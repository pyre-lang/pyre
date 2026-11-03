//! Legacy annotation propagation pass — transitional cutover input.
//!
//! TODO(retire-legacy-annotator): ad-hoc `ValueType`-flat-enum annotator
//! with no upstream `rpython/` counterpart — the orthodox port is
//! [`crate::annotator::annrpython::RPythonAnnotator`] (`annrpython.py`),
//! which produces per-`Variable.annotation` `SomeValue` shells directly.
//!
//! This file remains because the cutover transitional path
//! ([`crate::translator::rtyper::cutover::dual_gate_check_with_registry`]
//! + [`crate::translator::rtyper::cutover::lift_callee_to_pygraph`]) and
//!   the Skip-fallback in
//!   [`crate::codewriter::codewriter::transform_graph_to_jitcode`]
//!   still drive this `annotate()` for its side-effect: populating each
//!   `Variable.annotation` cell via `setbinding_with`.  A
//!   follow-up retirement slice drops both consumers and this file
//!   together once the dual-gate Skip categories close.
//!
//! Propagates ValueType annotations through the graph by analyzing
//! each op's inputs and computing the output type. Scheduled blocks
//! run to fixpoint (`complete_pending_blocks`) when Block.inputargs
//! (Phi nodes) need widening.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::annotator::annrpython::apply_renaming;
use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::description::DescEntry;
use crate::annotator::model::{
    SomeInstance, SomeInteger, SomeNone, SomeObjectTrait, SomePtr, SomeTypeOf, SomeValue, s_uint,
    r#typeof,
};
use crate::codewriter::annotation_state::{somevalue_to_valuetype, valuetype_to_someshell};
use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{CallTarget, FunctionGraph, Link, LinkArg, OpKind, ValueType, VecFieldPart};
use crate::translator::rtyper::error::TyperError;

enum Incoming {
    Flat(ValueType),
    Annotation(Rc<SomeValue>),
}

/// Run annotation propagation to fixpoint, writing each binding
/// directly into the orthodox `Variable.annotation` cell
/// (`flowspace/model.py:Variable.annotation`).
///
/// RPython equivalent: `RPythonAnnotator.complete()` →
/// `complete_pending_blocks` — processes scheduled blocks until the
/// pending set is empty.  All bindings land on each
/// `Variable.annotation` cell; callers that want a flat
/// `ValueType` discriminator read via `read_binding` (tests) or
/// `somevalue_to_valuetype(&var.annotation.borrow())`
/// (production).
pub fn annotate(graph: &FunctionGraph) {
    annotate_with_bookkeeper(graph, None);
}

/// Like [`annotate`], attaching a classed `SomeInstance` when a declared
/// `Ref(Some(root))` resolves through the per-translation bookkeeper
/// (`bookkeeper.py getuniqueclassdef`). `rclass.py getgcflavor` reads
/// `_alloc_flavor_` off that classdef.
pub fn annotate_with_bookkeeper(graph: &FunctionGraph, bookkeeper: Option<&Rc<Bookkeeper>>) {
    annotate_inner(graph, bookkeeper).unwrap_or_else(|e| panic!("{e}"));
}

fn annotate_inner(
    graph: &FunctionGraph,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<(), TyperError> {
    complete_pending_blocks(graph, bookkeeper).map_err(|e| {
        let text = e.to_string();
        if text.contains("UnionError:") {
            TyperError::message(format!("{}: {text}", graph.name))
        } else {
            e
        }
    })
}

fn complete_pending_blocks(
    graph: &FunctionGraph,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<(), TyperError> {
    // RPython parity: `annrpython.py:RPythonAnnotator.complete()`
    // (`annrpython.py typeof([v_last_exc_value])`) writes
    // exceptblock annotations only when a `follow_raise_link` actually
    // reaches the block — they stay `None` for unreachable exceptblocks.
    // No annotator-stage pre-seed runs here; the rtyper-equivalent
    // exception type/value concretetype seed is handled in
    // `legacy_resolve::resolve_types` (`rtyper.setup_block_entry` writes
    // `Signed` for `etype`, `GcRef` for `evalue` unconditionally at
    // rtyper time, not annotation time — `rpython/rtyper/rtyper.py:
    // setup_block_entry`).  Doing the seed at the annotation layer
    // here would clash with `setbinding`'s monotonicity assert when
    // partial real-rtyper `SomeInstance(...)` annotations are already
    // attached from a panicked walk.
    // `returnblock.inputargs[0]` deliberately stays unseeded here.
    // `set_return(_, None)` wires a `LinkArg::Const(Constant(None,
    // concretetype=Some(VOID)))` (model.rs:set_return) and the
    // rtyper-layer projection in
    // `legacy_resolve::link_arg_concrete_type` honours
    // `value.concretetype` to materialise `ConcreteType::Void` on the
    // returnblock inputarg, matching
    // `pairtype(Repr, NoneRepr).convert_from_to → inputconst(Void, None)`
    // (`rpython/rtyper/rnone.py`).  Pre-seeding `Ref` at annotation
    // stage would collapse a real `Float`/`Int` return into
    // `union_type(Ref, Float|Int) == Unknown`.

    let mut owned_bk: Option<Rc<Bookkeeper>> = None;

    // `annrpython.py RPythonAnnotator.complete_pending_blocks`: a block
    // is processed only when scheduled (`addpendingblock` /
    // `schedulependingblock`). Seed the startblock; a link target is
    // pushed when `follow_link` / `mergeinputargs` generalizes the
    // target's inputargs, or the target was never processed
    // (`block not in self.annotated`). `processblock` / `flowin` then
    // walk that block's ops and `block.exits`. Drain the pending set;
    // there is no sweep cap. Blocks never scheduled from the startblock
    // stay unannotated.
    let n = graph.blocks.len();
    let mut pending: VecDeque<usize> = VecDeque::new();
    let mut in_pending = vec![false; n];
    let mut annotated = vec![false; n];

    let start = graph.startblock.0;
    pending.push_back(start);
    in_pending[start] = true;
    // Same extra roots as `live_block_flags` / `clear_unreachable_blocks`:
    // a second start (`generated::*` / `eval_loop_jit` body) is live for
    // `setconcretetype` but has no predecessor from `startblock`.
    for root in crate::model::secondary_entry_roots(graph) {
        let idx = root.0;
        if !in_pending[idx] {
            pending.push_back(idx);
            in_pending[idx] = true;
        }
    }

    while let Some(idx) = pending.pop_front() {
        in_pending[idx] = false;
        annotated[idx] = true;

        let block = &graph.blocks[idx];
        // Propagate annotations through ops in this block
        for op in &block.operations {
            if let Some(result) = op.result.as_ref() {
                let inferred = match &op.kind {
                    OpKind::ConstRefAddr(_)
                        if FunctionGraph::concretetype_of(result)
                            == crate::model::ConcreteType::Void =>
                    {
                        // Frozen PBC with no attributes:
                        // `rpbc.py SingleFrozenPBCRepr.lowleveltype = Void`.
                        ValueType::Void
                    }
                    _ => infer_op_type(&op.kind),
                };
                // `SomeObject.hint` returns `self` (unaryop.py). `same_as`
                // keeps PBC / Bool; a `SomeInteger` operand goes through
                // `integer_annotation_incoming` so a signedness-changing
                // `result_ty` is `r_uint` / `intmask` rather than identity
                // (`bind_array_view_len_in_block` still plants `same_as`
                // with `result_ty: Int`, same signedness, nonneg kept).
                let keep_operand = match &op.kind {
                    OpKind::Hint { value, .. } => read_binding_some(value),
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if opname == "same_as" => match read_binding_some(operand) {
                        Some(rc) if matches!(rc.as_ref(), SomeValue::Integer(_)) => None,
                        other => other,
                    },
                    _ => None,
                };
                let incoming = match keep_operand {
                    Some(rc)
                        if matches!(
                            rc.as_ref(),
                            SomeValue::PBC(_) | SomeValue::Integer(_) | SomeValue::Bool(_)
                        ) =>
                    {
                        Incoming::Annotation(rc)
                    }
                    _ => typed_null_raw_incoming(bookkeeper, &op.kind)
                        .or_else(|| typed_null_fn_incoming(bookkeeper, &op.kind))
                        .or_else(|| fn_ptr_annotation_incoming(bookkeeper, result, &op.kind))
                        .or_else(|| raw_ptr_annotation_incoming(result, &op.kind))
                        .or_else(|| type_someobject_incoming(&op.kind))
                        .or_else(|| option_payload_const_none_incoming(result, &op.kind))
                        .or_else(|| integer_annotation_incoming(&op.kind, bookkeeper))
                        .or_else(|| binop_integer_incoming(&op.kind))
                        .unwrap_or(Incoming::Flat(inferred)),
                };
                if let Incoming::Annotation(sv) = &incoming
                    && matches!(sv.as_ref(), SomeValue::PBC(_))
                {
                    merge_some_pbc(result, sv.as_ref())?;
                } else {
                    merge_op_result(result, incoming, bookkeeper)?;
                }
            }
        }

        // Cross-block propagation: Link args → target inputargs, per
        // `follow_link` / `mergeinputargs` which iterate
        // `for link in block.exits` and union each `link.args[i]`
        // annotation into `link.target.inputargs[i]`.
        for link in &block.exits {
            let target = link.target.0;
            let link_changed = if link_is_raise_like(link) {
                follow_raise_link(graph, link, bookkeeper, &mut owned_bk)?
            } else {
                follow_link(graph, link, bookkeeper, &mut owned_bk)?
            };
            enqueue_pending(
                &mut pending,
                &mut in_pending,
                &annotated,
                target,
                link_changed,
            );
        }
    }
    Ok(())
}

/// `schedulependingblock`: de-duplicated FIFO of block indices. Push
/// when `mergeinputargs` changed the target, or the target is not yet
/// in `annotated`.
fn enqueue_pending(
    pending: &mut VecDeque<usize>,
    in_pending: &mut [bool],
    annotated: &[bool],
    target: usize,
    bindings_changed: bool,
) {
    if !in_pending[target] && (bindings_changed || !annotated[target]) {
        in_pending[target] = true;
        pending.push_back(target);
    }
}

/// Read the legacy `ValueType` discriminator out of
/// `var.annotation`.  Returns `Unknown` when the slot is empty —
/// consistent with [`setbinding_with`]'s invariant (every non-`Unknown`
/// setbinding writes a paired shell; `Unknown` clears it). Kind
/// prediction (`getFrozenPBCRepr` → `SingleFrozenPBCRepr`) still maps
/// a single FrozenDesc `SomePBC` to `Void`.
fn read_binding(var: &Variable) -> ValueType {
    var.annotation
        .borrow()
        .as_ref()
        .map(|s| {
            if is_single_frozen_pbc(s.as_ref()) {
                // Register-kind prediction only: `getFrozenPBCRepr` →
                // `SingleFrozenPBCRepr` (`lowleveltype = Void`). The
                // annotation cell is still `SomePBC`, not BOTTOM.
                ValueType::Void
            } else if is_str_family_shell(s.as_ref()) {
                // `somevalue_to_valuetype(SomeString)` is already `Str`.
                // `SomeChar` is `issubclass(tp, str)` (`bookkeeper.py`
                // `immutablevalue`); `rstr.py` has one string type.
                ValueType::Str
            } else {
                somevalue_to_valuetype(s)
            }
        })
        .unwrap_or(ValueType::Unknown)
}

/// Read the precise `SomeValue` shell out of `var.annotation`, or
/// `None` when the producer left the slot empty.
fn read_binding_some(var: &Variable) -> Option<Rc<SomeValue>> {
    var.annotation.borrow().clone()
}

/// Write a `ValueType` binding to `var.annotation` via the matching
/// `SomeValue` shell.  Doubles as the cfg(test) seed helper that
/// fixtures use to attach annotations directly to the orthodox
/// `Variable.annotation` cell.
///
/// RPython `RPythonAnnotator.setbinding(arg, s_value)`
/// (`annrpython.py`):
///
/// ```python
/// def setbinding(self, arg, s_value):
///     if arg in self.bindings:
///         assert s_value.contains(self.bindings[arg])
///     self.bindings[arg] = s_value
/// ```
///
/// The containment check enforces monotonicity (`s_new ⊇ s_old`); a
/// non-monotone re-binding is a producer-side error.  `ValueType::
/// Unknown` has no upstream annotation-stage counterpart; setting it
/// clears any stale shell so the downstream rtyper fails fast at
/// `bindingrepr` instead of silently bridging to `GcRef`.
#[cfg(test)]
pub(crate) fn setbinding(var: &Variable, ty: ValueType) {
    setbinding_with(var, ty, None);
}

/// `SomeValue` shell `setbinding` would write for `ty`. A classed
/// `Ref(Some(root))` is `SomeInstance(classdef, can_be_None=False)`;
/// a str-family root is `SomeString` with `no_nul=False`
/// (`annotator/model.py` `SomeString.__init__` default); `Ref(None)` is
/// the classdef-less `ref_fallback_instance`.
fn shell_for(ty: &ValueType, bookkeeper: Option<&Rc<Bookkeeper>>) -> Option<SomeValue> {
    match ty {
        ValueType::Ref(Some(root))
            if crate::annotator::bookkeeper::is_gcref_type_spelling(root) =>
        {
            Some(crate::annotator::bookkeeper::gcref_ptr_annotation())
        }
        ValueType::Ref(Some(root)) if crate::front::mir::is_str_family_class_root(root) => {
            // `valuetype_to_someshell` shells this as `SomeString` with
            // `no_nul=False` (`SomeString.__init__` default); `no_nul`
            // needs `rlib/rstring.py` `assert_str0`. Interning a ClassDef
            // here would mint `SomeInstance(String)` that
            // `binaryop.py` `pairtype(SomeObject, SomeObject).union` cannot
            // join with `String::new()`.
            Some(crate::annotator::model::s_str())
        }
        ValueType::Ref(Some(root)) => bookkeeper
            .and_then(|bk| bk.classdef_for_known_struct_root(root))
            .map(|classdef| {
                SomeValue::Instance(SomeInstance::new(Some(classdef), false, Default::default()))
            })
            .or_else(|| valuetype_to_someshell(ty)),
        _ => valuetype_to_someshell(ty),
    }
}

fn setbinding_with(var: &Variable, ty: ValueType, bookkeeper: Option<&Rc<Bookkeeper>>) {
    let shell = shell_for(&ty, bookkeeper);
    if let Some(shell) = shell {
        let new_val = Rc::new(shell);
        let mut slot = var.annotation.borrow_mut();
        if let Some(existing) = slot.as_ref() {
            assert!(
                new_val.contains(existing.as_ref()),
                "legacy_annotator::setbinding: non-monotone re-binding at \
                 Variable {var:?}; new value {:?} does not contain previous \
                 value {:?} (annrpython.py:292)",
                new_val,
                existing.as_ref(),
            );
        }
        *slot = Some(new_val);
    } else {
        *var.annotation.borrow_mut() = None;
    }
}

/// Rewrite a declared `Ref(Some(root))` to the interned classdef name so
/// fixpoint rounds compare equal (`somevalue_to_valuetype` emits that
/// name). `rclass.py getgcflavor` keys off the classdef, not the front
/// leaf spelling. A str-family root is `Str` (`rstr.py` one string type;
/// `is_str_family_class_root`): incoming `Ref(Some("string::String"))`
/// must match a `Str` slot (`binaryop.py`
/// `pairtype(SomeString, SomeString).union`).
fn declared_ref_valuetype(ty: ValueType, bookkeeper: Option<&Rc<Bookkeeper>>) -> ValueType {
    match &ty {
        ValueType::Ref(Some(root)) if crate::front::mir::is_str_family_class_root(root) => {
            ValueType::Str
        }
        ValueType::Ref(Some(root)) => {
            if let Some(cd) = bookkeeper.and_then(|bk| bk.classdef_for_known_struct_root(root)) {
                ValueType::Ref(Some(cd.borrow().name.clone()))
            } else {
                ty
            }
        }
        _ => ty,
    }
}

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

/// `annrpython.py follow_link` / `follow_raise_link` `renaming`:
/// `for v_out, v_input in zip(link.args, link.target.inputargs):
///      renaming[v_out].append(v_input)`.
/// Only `LinkArg::Value` sources contribute; constants are absent.
/// Keys use the same `Rc<Variable>` identity as [`typeof_last_exc_value`].
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn link_renaming(graph: &FunctionGraph, link: &Link) -> HashMap<Rc<Variable>, Vec<Rc<Variable>>> {
    let target_block = graph.block(link.target);
    let mut renaming: HashMap<Rc<Variable>, Vec<Rc<Variable>>> = HashMap::new();
    for (src, dst) in link.args.iter().zip(target_block.inputargs.iter()) {
        let LinkArg::Value(v_out) = src else {
            continue;
        };
        renaming
            .entry(Rc::new(v_out.clone()))
            .or_default()
            .push(Rc::new(dst.clone()));
    }
    renaming
}

/// Live source annotation that `apply_renaming` rewrites: `SomeTypeOf`
/// or a `SomeBool` carrying `knowntypedata`.
fn link_arg_renaming_source(src: &LinkArg) -> Option<SomeValue> {
    let LinkArg::Value(var) = src else {
        return None;
    };
    let ann = read_binding_some(var)?;
    match ann.as_ref() {
        SomeValue::TypeOf(_) => Some(ann.as_ref().clone()),
        SomeValue::Bool(b) if b.knowntypedata.is_some() => Some(ann.as_ref().clone()),
        _ => None,
    }
}

fn follow_link(
    graph: &FunctionGraph,
    link: &Link,
    bookkeeper: Option<&Rc<Bookkeeper>>,
    owned_bk: &mut Option<Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    let mut changed = false;
    let renaming = link_renaming(graph, link);
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        if let Some(s_out) = link_arg_renaming_source(src) {
            let s_out = apply_renaming(s_out, &renaming);
            changed |= merge_binding(dst, Incoming::Annotation(Rc::new(s_out)), bookkeeper)?;
        } else {
            changed |= bind_link_arg(dst, src, bookkeeper, owned_bk)?;
        }
    }
    Ok(changed)
}

/// `annrpython.py follow_raise_link`: `typeof([v_last_exc_value])`
/// (`model.py typeof` / `unaryop.py type_SomeObject`). Empty args yield
/// `SomeType()`. Source-side `setbinding` only; dest uses
/// `typeof(renaming[v_last_exc_value])`.
fn typeof_last_exc_value(link: &Link) -> Rc<SomeValue> {
    let vars: Vec<Rc<Variable>> = link
        .last_exc_value
        .as_ref()
        .and_then(LinkArg::as_variable)
        .map(|v| vec![Rc::new(v.clone())])
        .unwrap_or_default();
    Rc::new(crate::annotator::model::r#typeof(&vars))
}

/// `annrpython.py follow_raise_link`: dest `s_out = typeof(renaming[v_last_exc_value])`
/// plus the `v_last_exc_type` const rule.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn raise_link_dest_etype(
    link: &Link,
    renaming: &HashMap<Rc<Variable>, Vec<Rc<Variable>>>,
) -> SomeValue {
    let renamed: Vec<Rc<Variable>> = link
        .last_exc_value
        .as_ref()
        .and_then(LinkArg::as_variable)
        .and_then(|v| renaming.get(&Rc::new(v.clone())).cloned())
        .unwrap_or_default();
    let mut s_out = r#typeof(&renamed);
    match link.last_exception.as_ref() {
        Some(LinkArg::Const(c)) => {
            s_out.set_const_box(c.clone());
        }
        Some(LinkArg::Value(v)) => {
            if let Some(ann) = read_binding_some(v)
                && ann.is_constant()
                && let Some(val) = ann.const_()
            {
                s_out.set_const_box(crate::flowspace::model::Constant::new(val.clone()));
            }
        }
        None => {}
    }
    s_out
}

fn follow_raise_link(
    graph: &FunctionGraph,
    link: &Link,
    bookkeeper: Option<&Rc<Bookkeeper>>,
    owned_bk: &mut Option<Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    let mut changed = false;
    if let Some(value) = link.last_exc_value.as_ref().and_then(|a| a.as_variable()) {
        changed |= merge_value_type(value, ValueType::Ref(None), bookkeeper)?;
    }
    if let Some(value) = link.last_exception.as_ref().and_then(|a| a.as_variable()) {
        // `setbinding(v_last_exc_type, typeof([v_last_exc_value]))`.
        changed |= merge_binding(
            value,
            Incoming::Annotation(typeof_last_exc_value(link)),
            bookkeeper,
        )?;
    }

    let renaming = link_renaming(graph, link);
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let incoming = if Some(src) == link.last_exception.as_ref() {
            // `s_out = typeof(renaming[v_last_exc_value])` names dest
            // exceptblock value inputargs.
            Incoming::Annotation(Rc::new(raise_link_dest_etype(link, &renaming)))
        } else if Some(src) == link.last_exc_value.as_ref() {
            Incoming::Flat(ValueType::Ref(None))
        } else if let Some(s_out) = link_arg_renaming_source(src) {
            let s_out = apply_renaming(s_out, &renaming);
            changed |= merge_binding(dst, Incoming::Annotation(Rc::new(s_out)), bookkeeper)?;
            continue;
        } else {
            changed |= bind_link_arg(dst, src, bookkeeper, owned_bk)?;
            continue;
        };
        changed |= merge_binding(dst, incoming, bookkeeper)?;
    }
    Ok(changed)
}

/// `annrpython.py follow_link` / `mergeinputargs`: a constant takes
/// `bookkeeper.immutablevalue(v_out.value)` and a variable its own
/// annotation, then the two are unioned. A bare `Repr` constant is a frozen
/// PBC (`SomePBC`). A `SomePBC` on either side of a link (the variable's
/// annotation, or a dest already `SomePBC`) unions through
/// `merge_some_pbc`; a non-Repr constant arriving at a `SomePBC` dest is
/// first `immutablevalue`'d. `getFrozenPBCRepr` must stay
/// `SingleFrozenPBCRepr`. Every other source merges through
/// `merge_binding`.
fn bind_link_arg(
    dst: &Variable,
    src: &LinkArg,
    bookkeeper: Option<&Rc<Bookkeeper>>,
    owned_bk: &mut Option<Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    match src {
        LinkArg::Const(c) if matches!(&c.value, ConstValue::Repr(_)) => {
            let bk = match bookkeeper {
                Some(bk) => bk,
                None => owned_bk.get_or_insert_with(|| Rc::new(Bookkeeper::new())),
            };
            return merge_frozen_repr_annotation(dst, &c.value, bk);
        }
        LinkArg::Const(c) if matches!(&c.value, ConstValue::HostObject(h) if h.is_class()) => {
            // `bookkeeper.py immutablevalue` `tp is type` → `SomeConstantType`
            // (`SomePBC` of a ClassDesc). `set_raise` / `set_raise_implicit`
            // put that class Constant on exceptblock slot 0; a flat
            // `const_value_type` `Ref(None)` would shell as `SomeInstance`.
            let bk = match bookkeeper {
                Some(bk) => bk,
                None => owned_bk.get_or_insert_with(|| Rc::new(Bookkeeper::new())),
            };
            let incoming = bk
                .immutablevalue(&c.value)
                .map_err(|e| TyperError::message(e.to_string()))?;
            return merge_binding(dst, Incoming::Annotation(Rc::new(incoming)), Some(bk));
        }
        LinkArg::Value(var) => {
            let var_ann = read_binding_some(var);
            let dst_is_pbc = matches!(read_binding_some(dst).as_deref(), Some(SomeValue::PBC(_)));
            if var_ann.is_none() && dst_is_pbc {
                return Ok(false);
            }
            if let Some(incoming) = var_ann
                && (matches!(incoming.as_ref(), SomeValue::PBC(_)) || dst_is_pbc)
            {
                return merge_some_pbc(dst, incoming.as_ref());
            }
        }
        LinkArg::Const(c) => {
            if matches!(read_binding_some(dst).as_deref(), Some(SomeValue::PBC(_))) {
                let bk = match bookkeeper {
                    Some(bk) => bk,
                    None => owned_bk.get_or_insert_with(|| Rc::new(Bookkeeper::new())),
                };
                let incoming = bk
                    .immutablevalue(&c.value)
                    .map_err(|e| TyperError::message(e.to_string()))?;
                return merge_some_pbc(dst, &incoming);
            }
        }
    }
    merge_binding(dst, incoming_from_link_arg(src, bookkeeper), bookkeeper)
}

/// `annrpython.py follow_link` / `mergeinputargs` union the source's
/// actual annotation (`s_out = self.annotation(v_out)`). `immutablevalue(None)`
/// is `SomeNone`. A `ByteStr` / `UniStr` constant is
/// `bookkeeper.py immutablevalue` `issubclass(tp, str)` → `SomeString` /
/// `SomeChar` when a bookkeeper is available; otherwise flat `Str`.
/// `ConstValue::Int` / `UInt` go through `immutablevalue` the same way
/// (`bookkeeper.py immutablevalue(int)` → `SomeInteger(nonneg = x >= 0)`).
fn incoming_from_link_arg(src: &LinkArg, bookkeeper: Option<&Rc<Bookkeeper>>) -> Incoming {
    match src {
        LinkArg::Value(v) => match v.annotation.borrow().as_ref() {
            Some(rc) => Incoming::Annotation(rc.clone()),
            None => Incoming::Flat(ValueType::Unknown),
        },
        LinkArg::Const(c) if matches!(c.value, ConstValue::None) => {
            Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new())))
        }
        LinkArg::Const(c) if matches!(c.value, ConstValue::ByteStr(_) | ConstValue::UniStr(_)) => {
            str_const_incoming(&c.value, bookkeeper)
        }
        LinkArg::Const(c) if matches!(c.value, ConstValue::Int(_) | ConstValue::UInt(_)) => {
            integer_const_incoming(&c.value, bookkeeper)
        }
        LinkArg::Const(c) => Incoming::Flat(const_value_type(&c.value)),
    }
}

/// `bookkeeper.py immutablevalue(int)` → `SomeInteger(nonneg = x >= 0)`;
/// `UInt` is the unsigned shell (`rarithmetic` `r_uint`). Without a
/// bookkeeper the flags are the same `SomeInteger::new` value.
fn integer_const_incoming(value: &ConstValue, bookkeeper: Option<&Rc<Bookkeeper>>) -> Incoming {
    if let Some(bk) = bookkeeper
        && let Ok(sv) = bk.immutablevalue(value)
    {
        return Incoming::Annotation(Rc::new(sv));
    }
    let shell = match value {
        ConstValue::Int(n) => SomeInteger::new(*n >= 0, false),
        ConstValue::UInt(_) => SomeInteger::new(false, true),
        _ => return Incoming::Flat(const_value_type(value)),
    };
    Incoming::Annotation(Rc::new(SomeValue::Integer(shell)))
}

/// `unaryop.py` `SomeObject.len` / `SomeString.len` and `SomeChar.ord`
/// return `SomeInteger(nonneg=True)`. `infer_op_type` still reports
/// flat `Int` for `read_binding`. Container-length `FieldRead` is the
/// same count (`rlist.py` `ll_length`; `("length", Signed)`), not a
/// possibly-negative struct field.
fn integer_annotation_incoming(
    kind: &OpKind,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Option<Incoming> {
    match kind {
        OpKind::ConstInt(n) => Some(integer_const_incoming(&ConstValue::Int(*n), bookkeeper)),
        OpKind::ConstUInt(n) => Some(integer_const_incoming(&ConstValue::UInt(*n), bookkeeper)),
        OpKind::ArrayLen { .. } | OpKind::VableArrayLen { .. } => {
            Some(nonneg_signed_integer_incoming())
        }
        OpKind::LoweredBlackholeOp { opname, .. }
            if matches!(opname.as_str(), "strlen" | "unicodelen" | "strgetitem") =>
        {
            Some(nonneg_signed_integer_incoming())
        }
        OpKind::Call { target, .. }
            if matches!(
                target.path_segments().as_deref(),
                Some(["__len"] | ["__strlen"] | ["__string_byte_getitem"])
            ) =>
        {
            Some(nonneg_signed_integer_incoming())
        }
        OpKind::FieldRead {
            field,
            ty: ValueType::Int,
            ..
        } if is_container_len_field(field) => Some(nonneg_signed_integer_incoming()),
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } if op == "same_as" => same_as_integer_incoming(operand, result_ty),
        _ => None,
    }
}

/// `rarithmetic.py` `r_uint(x)` / `intmask(x)` are signedness casts:
/// `ForTypeEntry.compute_result_annotation` → `SomeInteger(knowntype=
/// r_uint)`; `builtin.py` `rarith_intmask` → `SomeInteger()`.
/// `rffi.cast` (`ForceCastEntry.compute_result_annotation`) annotates
/// from the TARGET (`lltype_to_annotation`). Same signedness copies
/// the operand (`unaryop.py` identity; nonneg preserved).
fn same_as_integer_incoming(operand: &Variable, result_ty: &ValueType) -> Option<Incoming> {
    let rc = read_binding_some(operand)?;
    let SomeValue::Integer(i) = rc.as_ref() else {
        return None;
    };
    match result_ty {
        ValueType::Unsigned if !i.unsigned => Some(Incoming::Annotation(Rc::new(s_uint()))),
        ValueType::Int if i.unsigned => Some(Incoming::Annotation(Rc::new(SomeValue::Integer(
            SomeInteger::new(false, false),
        )))),
        _ => Some(Incoming::Annotation(rc)),
    }
}

/// `unaryop.py` `SomeObject.len` / `rlist.py` `ll_length`. `vec_part`
/// `Len` / `FatLen` are length words; `IntArray` / `FloatArray` `"len"`
/// is `typed_array_len_owner`. `FatMeta` is the dyn vtable word
/// (`rtuple.py` `TupleRepr.rtype_getitem` of `two_word_fat_struct`)
/// and keeps the element's signed `SomeInteger`. A signed field that
/// is not a length (`Buffer::Sub.size`, `last_instr`) stays out.
fn is_container_len_field(field: &crate::model::FieldDescriptor) -> bool {
    matches!(
        field.vec_part,
        Some(VecFieldPart::Len | VecFieldPart::FatLen)
    ) || (field.name == "len"
        && matches!(field.owner_root.as_deref(), Some("IntArray" | "FloatArray")))
}

/// `binaryop.py` `pairtype(SomeInteger, SomeInteger)`: `or_ = xor =
/// add = mul = _clone(union, [])` (`integer_arith_union`); `sub` →
/// `SomeInteger(knowntype=restype)` (`integer_sub`, nonneg False);
/// `and_` → `nonneg = int1.nonneg or int2.nonneg` (`integer_and`,
/// `integer_family_nonneg`). Two `SomeBool` operands take
/// `pairtype(SomeBool, SomeBool)` `and_` / `or_` / `xor`
/// (`bool_and_values` / `bool_or_values` / `bool_xor_values`).
/// `rarithmetic.py` `compute_restype` supplies knowntype (`bool, bool`
/// → `int`). Operand shells come from the live binding (`SomeInteger` /
/// `SomeBool`); a `ConstInt` / `ConstUInt` producer already stored that
/// shell via `bookkeeper.py immutablevalue`. Unbound operands and other
/// ops (shifts, div, mod, comparisons) stay on the Flat `result_ty`
/// path. A signedness `UnionError` from `integer_arith_union` also
/// falls back to Flat — the result slot merge still raises if that
/// conflicts.
fn binop_integer_incoming(kind: &OpKind) -> Option<Incoming> {
    use crate::annotator::binaryop::{
        bool_and_values, bool_or_values, bool_xor_values, integer_arith_union,
        integer_family_nonneg,
    };
    let OpKind::BinOp { op, lhs, rhs, .. } = kind else {
        return None;
    };
    let s0 = integer_family_operand(lhs)?;
    let s1 = integer_family_operand(rhs)?;
    let both_bool = matches!(
        (s0.as_ref(), s1.as_ref()),
        (SomeValue::Bool(_), SomeValue::Bool(_))
    );
    let result = match op.as_str() {
        "or" | "or_" | "or_ovf" | "bitor" if both_bool => bool_or_values(s0.as_ref(), s1.as_ref()),
        "xor" | "bitxor" | "xor_ovf" if both_bool => bool_xor_values(s0.as_ref(), s1.as_ref()),
        "and" | "and_" | "bitand" if both_bool => bool_and_values(s0.as_ref(), s1.as_ref()),
        "add" | "add_ovf" | "mul" | "mul_ovf" | "or" | "or_" | "or_ovf" | "bitor" | "xor"
        | "bitxor" | "xor_ovf" => integer_arith_union(s0.as_ref(), s1.as_ref()).ok()?,
        "sub" | "sub_ovf" => {
            let kt = crate::rlib::rarithmetic::compute_restype(s0.knowntype(), s1.knowntype());
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, kt))
        }
        "and" | "and_" | "bitand" => {
            let nonneg = integer_family_nonneg(s0.as_ref()) || integer_family_nonneg(s1.as_ref());
            let kt = crate::rlib::rarithmetic::compute_restype(s0.knowntype(), s1.knowntype());
            SomeValue::Integer(SomeInteger::new_with_knowntype(nonneg, kt))
        }
        _ => return None,
    };
    Some(Incoming::Annotation(Rc::new(result)))
}

fn integer_family_operand(var: &Variable) -> Option<Rc<SomeValue>> {
    let rc = read_binding_some(var)?;
    is_integer_family_shell(rc.as_ref()).then_some(rc)
}

fn nonneg_signed_integer_incoming() -> Incoming {
    Incoming::Annotation(Rc::new(SomeValue::Integer(SomeInteger::new(true, false))))
}

/// `bookkeeper.py immutablevalue` of a `str` constant. Len-1 is
/// `SomeChar`; `no_nul` follows the bytes. Without a bookkeeper the
/// walker stores flat `Str` (`rstr.py` one string type).
fn str_const_incoming(value: &ConstValue, bookkeeper: Option<&Rc<Bookkeeper>>) -> Incoming {
    if let Some(bk) = bookkeeper
        && let Ok(sv) = bk.immutablevalue(value)
    {
        Incoming::Annotation(Rc::new(sv))
    } else {
        Incoming::Flat(ValueType::Str)
    }
}

/// `unaryop.py type_SomeObject`: `op.type(v)` → `typeof([v])` (`SomeTypeOf`).
/// Target is the same `CallTarget::function_path(["type"])` `push_type_of`
/// emits (`flowcontext.py exc_from_raise`).
fn type_someobject_incoming(kind: &OpKind) -> Option<Incoming> {
    let OpKind::Call { target, args, .. } = kind else {
        return None;
    };
    if target != &CallTarget::function_path(["type"]) {
        return None;
    }
    let [LinkArg::Value(v)] = args.as_slice() else {
        return None;
    };
    Some(Incoming::Annotation(Rc::new(r#typeof(&[Rc::new(
        v.clone(),
    )]))))
}

/// Typed `null_fn` (`model.rs push_null_fn_ptr_with_spelling`) is
/// `lltype.nullptr(FuncType)` as constant `SomePtr`
/// (`builtin.rs fn_null_constant` / `ann_nullptr` `immutablevalue`).
fn typed_null_fn_incoming(bookkeeper: Option<&Rc<Bookkeeper>>, kind: &OpKind) -> Option<Incoming> {
    let bk = bookkeeper?;
    let OpKind::Call { target, args, .. } = kind else {
        return None;
    };
    if !matches!(
        target.path_segments().as_deref(),
        Some(["core", "ptr", "null_fn"])
    ) {
        return None;
    }
    let args_s: Vec<Option<SomeValue>> = match args.as_slice() {
        [] => Vec::new(),
        [LinkArg::Const(c)] => {
            let ConstValue::ByteStr(spelling) = &c.value else {
                return None;
            };
            let mut s_ty = crate::annotator::model::s_str0();
            s_ty.set_const_box(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::byte_str(spelling),
            ));
            vec![Some(s_ty)]
        }
        _ => return None,
    };
    match crate::annotator::builtin::fn_null_constant(
        bk,
        &args_s,
        &std::collections::HashMap::new(),
    ) {
        Ok(sv @ SomeValue::Ptr(_)) => Some(Incoming::Annotation(Rc::new(sv))),
        _ => None,
    }
}

/// A function-pointer FieldRead / Call / Input is `SomePtr(Ptr(FuncType))`
/// (`lltype.py` `SomePtr` / `getfunctionptr`), never `SomeInteger`.
/// `history.getkind` of `Ptr(FuncType)` stays `'int'`, so the op bank is
/// still Int. `pairtype(SomePtr, SomePtr).union` requires equal
/// `ll_ptrtype`; FieldRead / Call dests carry the `stamp_fn_ptr`
/// `Ptr(FuncType)`, and Input seeds through
/// `fn_ptr_somevalue_for_spelling_in` with this bookkeeper so a
/// `Result<(), E>` of the exception carrier is `Void`
/// (`fn_ptr_result_lltype` / `ann_nullptr`).
fn fn_ptr_annotation_incoming(
    bookkeeper: Option<&Rc<Bookkeeper>>,
    result: &Variable,
    kind: &OpKind,
) -> Option<Incoming> {
    let bk = bookkeeper?;
    match kind {
        OpKind::FieldRead {
            ty: ValueType::Int, ..
        }
        | OpKind::Call {
            result_ty: ValueType::Int,
            ..
        } => {
            // Dest `concretetype` is `Ptr(FuncType)` from `stamp_fn_ptr`
            // (`operation.rs` FieldRead `field_ty` / `place_ty`; Call dest).
            // Charon's field type is the alias-expanded `FnPtr`, so the
            // stamp covers `BuiltinCode.func` without a name-keyed alias.
            fn_ptr_from_dest_concretetype(result)
        }
        OpKind::Input {
            ty: ValueType::Int,
            class_root: Some(root),
            ..
        } => {
            crate::annotator::bookkeeper::fn_ptr_somevalue_for_spelling_in(root, Some(bk.as_ref()))
                .filter(somevalue_is_fn_ptr)
                .map(|sv| Incoming::Annotation(Rc::new(sv)))
        }
        _ => None,
    }
}

fn somevalue_is_fn_ptr(sv: &SomeValue) -> bool {
    use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
    matches!(sv, SomeValue::Ptr(p) if matches!(p.ll_ptrtype.TO, PtrTarget::Func(_)))
}

/// Dest `concretetype` stamped `Ptr(FuncType)` (`stamp_fn_ptr`, the
/// `stamp_raw_u8_ccharp` / `var_concretetype_is_ccharp` pattern). That
/// `ll_ptrtype` is minted through `fn_ptr_somevalue_for_spelling_in` so
/// it equals `fn_null_constant` of the same spelling.
fn fn_ptr_from_dest_concretetype(result: &Variable) -> Option<Incoming> {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    let ct = result.concretetype()?;
    let LowLevelType::Ptr(ptr_ty) = ct else {
        return None;
    };
    if !matches!(ptr_ty.TO, PtrTarget::Func(_)) {
        return None;
    }
    Some(Incoming::Annotation(Rc::new(SomeValue::Ptr(SomePtr::new(
        *ptr_ty,
    )))))
}

/// Typed `null_raw` (`model.rs push_null_raw_ptr_with_spelling`) is a
/// nullable `SomeInstance` of a flavor-raw class
/// (`builtin.rs raw_ptr_null_constant`).
fn typed_null_raw_incoming(bookkeeper: Option<&Rc<Bookkeeper>>, kind: &OpKind) -> Option<Incoming> {
    let bk = bookkeeper?;
    let OpKind::Call { target, args, .. } = kind else {
        return None;
    };
    if !matches!(
        target.path_segments().as_deref(),
        Some(["core", "ptr", "null_raw"])
    ) {
        return None;
    }
    let [LinkArg::Const(c)] = args.as_slice() else {
        return None;
    };
    let ConstValue::ByteStr(spelling) = &c.value else {
        return None;
    };
    let mut s_ty = crate::annotator::model::s_str0();
    s_ty.set_const_box(crate::flowspace::model::Constant::new(
        crate::flowspace::model::ConstValue::byte_str(spelling),
    ));
    match crate::annotator::builtin::raw_ptr_null_constant(
        bk,
        &[Some(s_ty)],
        &std::collections::HashMap::new(),
    ) {
        Ok(sv @ SomeValue::Instance(_)) => Some(Incoming::Annotation(Rc::new(sv))),
        _ => None,
    }
}

/// `lltype.py` `SomePtr` / `ann_malloc` / `ann_cast_int_to_ptr`. A raw
/// `*mut u8` / `*const u8` is `SomePtr(CCHARP)` (`rffi.py` `CCHARP`),
/// never `SomeInteger`. `infer_op_type` still reports the JIT bank
/// (`history.py` `getkind` of a raw `Ptr` is `'int'`).
fn raw_ptr_annotation_incoming(result: &Variable, kind: &OpKind) -> Option<Incoming> {
    match kind {
        OpKind::RawMalloc { owner, TYPE, .. } => {
            // Str-family TYPE/owner stays Flat so `infer_op_type` `Str`
            // wins (`rstr.py` `mallocstr` / `SomeString`). `ann_malloc` of
            // a typed container is `SomePtr(Ptr(s_T.const))`; a TYPE-less
            // byte buffer is `raw_alloc_ptr_somevalue` → `SomePtr(CCHARP)`.
            if crate::front::mir::is_str_family_class_root(owner)
                || crate::front::mir::is_rstr_str_layout_declaration(owner)
            {
                return None;
            }
            Some(Incoming::Annotation(Rc::new(someptr_from_raw_malloc_type(
                TYPE.as_ref(),
            ))))
        }
        OpKind::Call {
            target,
            args,
            result_ty,
            ..
        } if matches!(
            target.path_segments().as_deref(),
            Some([.., "lltype", "cast_int_to_ptr"])
        ) =>
        {
            Some(Incoming::Annotation(Rc::new(someptr_from_cast_int_to_ptr(
                result, args, result_ty,
            ))))
        }
        OpKind::Call { target, .. } if is_get_raw_address_of_string(target) => {
            Some(ccharp_someptr_incoming())
        }
        OpKind::Call { .. } | OpKind::FieldRead { .. } | OpKind::Input { .. }
            if var_concretetype_is_ccharp(result) =>
        {
            Some(ccharp_someptr_incoming())
        }
        OpKind::FieldRead {
            field,
            ty: ValueType::Unsigned,
            ..
        } if field.vec_part == Some(VecFieldPart::FatData) => Some(ccharp_someptr_incoming()),
        _ => None,
    }
}

/// Same shell as `builtin.rs` `raw_alloc_ptr_somevalue`: `SomePtr` of
/// `rffi.py` `CCHARP`.
fn ccharp_someptr() -> SomeValue {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::lltypesystem::rffi::CCHARP;
    let LowLevelType::Ptr(ptr_ty) = (*CCHARP).clone() else {
        unreachable!("CCHARP is a pointer");
    };
    SomeValue::Ptr(SomePtr::new(*ptr_ty))
}

fn ccharp_someptr_incoming() -> Incoming {
    Incoming::Annotation(Rc::new(ccharp_someptr()))
}

/// `lltype.py` `ann_malloc`: `SomePtr(Ptr(s_T.const))` when TYPE is the
/// allocated Struct/Array (`Ptr::from_container_type`). `TYPE: None` is
/// a front-end scalar spill / `std::alloc` byte buffer
/// (`builtin.rs` `raw_alloc_ptr_somevalue` → `SomePtr(CCHARP)`).
#[allow(non_snake_case)]
fn someptr_from_raw_malloc_type(
    TYPE: Option<&crate::translator::rtyper::lltypesystem::lltype::LowLevelType>,
) -> SomeValue {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr};
    let Some(t) = TYPE else {
        return ccharp_someptr();
    };
    if let LowLevelType::Ptr(ptr_ty) = t {
        return SomeValue::Ptr(SomePtr::new((**ptr_ty).clone()));
    }
    match Ptr::from_container_type(t.clone()) {
        Ok(ptr) => SomeValue::Ptr(SomePtr::new(ptr)),
        Err(_) => ccharp_someptr(),
    }
}

/// `history.py` `getkind` of a raw `Ptr` is `'int'`. `resolve_types`
/// `commit_concretetype` keeps a `Ptr(Array)` (`rffi.py` `CCHARP`) and
/// a `Ptr(FuncType)` (`lltype.FuncType` / `getfunctionptr`) and will
/// not replace them with the kind-only overlay (`Signed` from
/// `somevalue_to_valuetype` of `Ptr(FuncType)`, `GcRef` from a
/// non-function `SomePtr`).
fn stamp_someptr_lltype(var: &Variable, sv: &SomeValue) {
    let SomeValue::Ptr(p) = sv else {
        return;
    };
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    var.set_concretetype(Some(LowLevelType::Ptr(Box::new(p.ll_ptrtype.clone()))));
}

/// `ann_cast_int_to_ptr(PtrT, s_int)` → `SomePtr(PtrT.const)`. Dest
/// already carrying a `Ptr` (`emit_dest_typed_cast_int_to_ptr`) keeps
/// that `PtrT`. 1-arg `cast_int_to_ptr_call` omits PtrT; dest is
/// `llmemory.GCREF` (`emit_host_cast_llop` / `result_ty: Ref`). Missing
/// PtrT on a non-Ref dest is `CCHARP` (`emit_ccharp_nullptr`).
fn someptr_from_cast_int_to_ptr(
    result: &Variable,
    args: &[LinkArg],
    result_ty: &ValueType,
) -> SomeValue {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    if let Some(LinkArg::Const(c)) = args.first()
        && let ConstValue::LowLevelType(ll) = &c.value
        && let LowLevelType::Ptr(ptr_ty) = ll.as_ref()
    {
        return SomeValue::Ptr(SomePtr::new((**ptr_ty).clone()));
    }
    if let Some(ct) = result.concretetype().as_ref()
        && let LowLevelType::Ptr(ptr_ty) = ct
    {
        return SomeValue::Ptr(SomePtr::new((**ptr_ty).clone()));
    }
    if FunctionGraph::concretetype_of(result) == crate::model::ConcreteType::GcRef
        || matches!(result_ty, ValueType::Ref(_))
    {
        return crate::annotator::bookkeeper::gcref_ptr_annotation();
    }
    ccharp_someptr()
}

fn is_get_raw_address_of_string(target: &CallTarget) -> bool {
    let Some(segs) = target.path_segments() else {
        return false;
    };
    matches!(
        segs.last().copied(),
        Some(crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING)
            | Some(crate::codewriter::get_raw_address::LL_GET_RAW_ADDRESS_OF_STRING)
    )
}

/// Production stamps `rffi.py` `CCHARP` / `CONST_CCHARP` on a declared
/// `*mut u8` / `*const u8` (`stamp_raw_u8_ccharp` / `stamp_as_ccharp`).
fn var_concretetype_is_ccharp(var: &Variable) -> bool {
    use crate::translator::rtyper::lltypesystem::rffi::{CCHARP, CONST_CCHARP};
    matches!(
        var.concretetype().as_ref(),
        Some(ct) if *ct == *CCHARP || *ct == *CONST_CCHARP
    )
}

/// Skip-walker refusal for a merged Frozen PBC the four-kind bank cannot
/// represent. `TyperError` Display / `panic!("{e}")` emit this text as the
/// whole message (or as a prefix when `where` is attached). Dual-gate
/// comparison matches this const, not a duplicate literal.
pub(crate) const MULTIPLE_FROZEN_PBC_UNSUPPORTED: &str =
    "MultipleFrozenPBCRepr is unsupported on this path (getFrozenPBCRepr)";

/// `getFrozenPBCRepr`: `follow_link` / `mergeinputargs` union incoming
/// annotations (`bookkeeper.immutablevalue` for a constant, `annotation(v)`
/// for a variable). A FrozenDesc union with more than one desc,
/// `can_be_none`, or a union error would be `MultipleFrozenPBCRepr` (or
/// `MultipleUnrelatedFrozenPBCRepr`). This skip-arm path cannot represent
/// that Ptr/Address, so refuse. Function/method/class PBCs are not this
/// repr (`FunctionsPBCRepr` / `MethodsPBCRepr` / `ClassesPBCRepr`).
pub(crate) fn multiple_frozen_pbc_unsupported() -> TyperError {
    TyperError::message(MULTIPLE_FROZEN_PBC_UNSUPPORTED)
}

/// `getFrozenPBCRepr` decides from the dest annotation after
/// `follow_link` / `mergeinputargs` have unioned every incoming edge
/// (`bookkeeper.immutablevalue` for a constant, `annotation(v)` for a
/// variable). For each `(target block, inputarg index)` that receives
/// at least one `LinkArg::Const(c)` with `ConstValue::Repr(_)`:
///
/// * Annotated dest: refuse `is_multiple_frozen_pbc` (`getFrozenPBCRepr`
///   would pick `MultipleFrozenPBCRepr`). A single FrozenDesc `SomePBC`
///   (`is_single_frozen_pbc`) is accepted. Other PBCs
///   (functions/methods/classes) propagate; a non-PBC annotation is
///   still `MultipleFrozenPBCRepr`.
/// * Unannotated dest (`resolve_types` without `annotate`): keep the
///   syntactic rule on that slot — distinct Repr constants, or a
///   non-Repr link arg, are `MultipleFrozenPBCRepr`.
pub(crate) fn refuse_merged_frozen_pbc_reprs(graph: &FunctionGraph) -> Result<(), TyperError> {
    // `follow_link` / `complete_pending_blocks` only walk startblock-reachable exits.
    let mut reachable = vec![false; graph.blocks.len()];
    for block in graph.iterblocks() {
        reachable[block.id.0] = true;
    }
    let mut slots: Vec<(usize, usize)> = Vec::new();
    for block in &graph.blocks {
        if !reachable[block.id.0] {
            continue;
        }
        for link in &block.exits {
            let n = graph
                .block(link.target)
                .inputargs
                .len()
                .min(link.args.len());
            for i in 0..n {
                if let LinkArg::Const(c) = &link.args[i]
                    && matches!(&c.value, ConstValue::Repr(_))
                    && !slots
                        .iter()
                        .any(|(t, idx)| *t == link.target.0 && *idx == i)
                {
                    slots.push((link.target.0, i));
                }
            }
        }
    }

    let mut unannotated: Vec<(usize, usize)> = Vec::new();
    for (target, index) in slots {
        let var = &graph.blocks[target].inputargs[index];
        match var.annotation.borrow().as_ref() {
            Some(s) => {
                if is_multiple_frozen_pbc(s.as_ref()) || !matches!(s.as_ref(), SomeValue::PBC(_)) {
                    return Err(multiple_frozen_pbc_unsupported());
                }
            }
            None => unannotated.push((target, index)),
        }
    }

    let mut incoming: Vec<(usize, usize, ConstValue)> = Vec::new();
    for block in &graph.blocks {
        if !reachable[block.id.0] {
            continue;
        }
        for link in &block.exits {
            let n = graph
                .block(link.target)
                .inputargs
                .len()
                .min(link.args.len());
            for i in 0..n {
                if !unannotated
                    .iter()
                    .any(|(t, idx)| *t == link.target.0 && *idx == i)
                {
                    continue;
                }
                if let LinkArg::Const(c) = &link.args[i]
                    && matches!(&c.value, ConstValue::Repr(_))
                {
                    if let Some((_, _, prev)) = incoming
                        .iter()
                        .find(|(t, idx, _)| *t == link.target.0 && *idx == i)
                    {
                        if prev != &c.value {
                            return Err(multiple_frozen_pbc_unsupported());
                        }
                    } else {
                        incoming.push((link.target.0, i, c.value.clone()));
                    }
                }
            }
        }
    }
    for block in &graph.blocks {
        if !reachable[block.id.0] {
            continue;
        }
        for link in &block.exits {
            let n = graph
                .block(link.target)
                .inputargs
                .len()
                .min(link.args.len());
            for i in 0..n {
                if !unannotated
                    .iter()
                    .any(|(t, idx)| *t == link.target.0 && *idx == i)
                {
                    continue;
                }
                let is_repr = matches!(
                    &link.args[i],
                    LinkArg::Const(c) if matches!(&c.value, ConstValue::Repr(_))
                );
                if !is_repr
                    && incoming
                        .iter()
                        .any(|(t, idx, _)| *t == link.target.0 && *idx == i)
                {
                    return Err(multiple_frozen_pbc_unsupported());
                }
            }
        }
    }
    Ok(())
}

fn is_single_frozen_pbc(s: &SomeValue) -> bool {
    match s {
        SomeValue::PBC(pbc) => {
            pbc.descriptions.len() == 1
                && !pbc.can_be_none
                && pbc
                    .descriptions
                    .values()
                    .all(|d| matches!(d, DescEntry::Frozen(_)))
        }
        _ => false,
    }
}

/// `getFrozenPBCRepr` would pick MultipleFrozenPBCRepr for this PBC.
fn is_multiple_frozen_pbc(s: &SomeValue) -> bool {
    match s {
        SomeValue::PBC(pbc) => {
            pbc.descriptions
                .values()
                .any(|d| matches!(d, DescEntry::Frozen(_)))
                && (pbc.descriptions.len() > 1 || pbc.can_be_none)
        }
        _ => false,
    }
}

fn merge_frozen_repr_annotation(
    dst: &Variable,
    value: &ConstValue,
    bookkeeper: &Rc<Bookkeeper>,
) -> Result<bool, TyperError> {
    // `bookkeeper.immutablevalue` `_freeze_` arm: `SomePBC([getdesc(x)])`.
    let incoming = bookkeeper
        .immutablevalue(value)
        .map_err(|e| TyperError::message(e.to_string()))?;
    merge_some_pbc(dst, &incoming)
}

/// `pairtype(SomePBC, SomePBC).union` unions description sets and
/// nullability; `getFrozenPBCRepr` is FrozenDesc-only, so only a
/// multiple/nullable Frozen PBC is refused here.
fn merge_some_pbc(dst: &Variable, incoming: &SomeValue) -> Result<bool, TyperError> {
    let mut slot = dst.annotation.borrow_mut();
    match slot.as_ref() {
        None => {
            *slot = Some(Rc::new(incoming.clone()));
            Ok(true)
        }
        Some(existing) => match crate::annotator::model::union(existing.as_ref(), incoming) {
            Ok(merged) => {
                if is_multiple_frozen_pbc(&merged) {
                    return Err(multiple_frozen_pbc_unsupported());
                }
                if merged != **existing {
                    *slot = Some(Rc::new(merged));
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Err(e) => Err(TyperError::message(format!(
                "{e}; MultipleFrozenPBCRepr is unsupported on this path (getFrozenPBCRepr)"
            ))),
        },
    }
}

fn merge_value_type(
    dst: &Variable,
    src_ty: ValueType,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    merge_binding(dst, Incoming::Flat(src_ty), bookkeeper)
}

/// Op-result binding (`annrpython.py consider_op` → `setbinding`): the
/// op's annotation is recomputed, not unioned with a second source.
/// `infer_op_type` is a `ValueType` projection with no flags or classdef,
/// so an existing instance annotation that projects to the same type is
/// what the op already produced and is kept.
fn merge_op_result(
    var: &Variable,
    incoming: Incoming,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    if let Incoming::Flat(ty) = &incoming
        && matches!(
            read_binding_some(var).as_deref(),
            Some(SomeValue::Instance(_))
        )
        && read_binding(var) == declared_ref_valuetype(ty.clone(), bookkeeper)
    {
        return Ok(false);
    }
    // Op-result only (`annrpython.py` `consider_op` → `setbinding`):
    // a recompute of `op.type(v)` may feed Flat `Ref(None)` (`result_ty`)
    // after `unaryop.py` `type_SomeObject` stored `SomeTypeOf`. Links
    // and merges do not take this skip.
    if let Incoming::Flat(ValueType::Ref(None)) = &incoming
        && matches!(
            read_binding_some(var).as_deref(),
            Some(SomeValue::TypeOf(_) | SomeValue::Type(_))
        )
    {
        return Ok(false);
    }
    merge_binding(var, incoming, bookkeeper)
}

/// Join `incoming` into `var.annotation`. Instance unions go through
/// `union` (`annrpython.py mergeinputargs`, `binaryop.py
/// pairtype(SomeInstance, SomeInstance).union`) so `can_be_None` is
/// kept. No common base stores the classdef-less root instance
/// (`rclass.py getinstancerepr(None)`). An empty slot keeps `SomeNone`;
/// `SomeNone` then joins any `noneify` shell in either order via
/// `model::union` (`binaryop.py pairtype(SomeObject, SomeNone).union`
/// / `pairtype(SomeNone, SomeObject).union` → `obj.noneify()`).
fn merge_binding(
    var: &Variable,
    incoming: Incoming,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    match incoming {
        Incoming::Flat(ty) => {
            if is_integer_family_flat(&ty)
                && read_binding_some(var)
                    .as_deref()
                    .is_some_and(is_integer_family_shell)
                && let Some(shell) = valuetype_to_someshell(&ty)
            {
                merge_integer_family_binding(var, Rc::new(shell), bookkeeper)
            } else {
                merge_flat(var, ty, bookkeeper)
            }
        }
        Incoming::Annotation(sv) => {
            if matches!(sv.as_ref(), SomeValue::None_(_)) {
                return merge_none_incoming(var, sv);
            }
            if let Some(existing) = read_binding_some(var)
                && matches!(existing.as_ref(), SomeValue::None_(_))
            {
                return store_annotation_union(var, existing.as_ref(), sv.as_ref());
            }
            match sv.as_ref() {
                SomeValue::Instance(_) => match read_binding_some(var) {
                    None => {
                        *var.annotation.borrow_mut() = Some(sv.clone());
                        Ok(true)
                    }
                    Some(existing) if matches!(existing.as_ref(), SomeValue::Instance(_)) => {
                        Ok(merge_instance_pair(var, existing.as_ref(), sv.as_ref()))
                    }
                    Some(_) => merge_flat(var, somevalue_to_valuetype(&sv), bookkeeper),
                },
                SomeValue::TypeOf(_) | SomeValue::Type(_) => {
                    merge_type_shaped_binding(var, sv, bookkeeper)
                }
                SomeValue::PBC(_)
                    if crate::annotator::model::is_some_constant_type(sv.as_ref()) =>
                {
                    merge_type_shaped_binding(var, sv, bookkeeper)
                }
                SomeValue::Integer(_) | SomeValue::Bool(_) => {
                    merge_integer_family_binding(var, sv, bookkeeper)
                }
                SomeValue::String(_)
                | SomeValue::Char(_)
                | SomeValue::UnicodeString(_)
                | SomeValue::UnicodeCodePoint(_) => merge_str_family_binding(var, sv, bookkeeper),
                SomeValue::Ptr(_) => merge_ptr_family_binding(var, sv, bookkeeper),
                _ => merge_flat(var, somevalue_to_valuetype(&sv), bookkeeper),
            }
        }
    }
}

/// Incoming `SomeNone` (`bookkeeper.py immutablevalue(None)` → `s_None`).
/// Empty slot keeps `SomeNone`; `SomeNone` ∪ `SomeNone` is identity;
/// any other bound shell goes through `union` → `noneify`.
fn merge_none_incoming(var: &Variable, sv: Rc<SomeValue>) -> Result<bool, TyperError> {
    match read_binding_some(var) {
        None => {
            *var.annotation.borrow_mut() = Some(sv);
            Ok(true)
        }
        Some(existing) if matches!(existing.as_ref(), SomeValue::None_(_)) => Ok(false),
        Some(existing) => store_annotation_union(var, existing.as_ref(), sv.as_ref()),
    }
}

/// `annotator::model::union` (`binaryop.py pairtype(SomeObject, SomeNone)
/// .union` / `pairtype(SomeNone, SomeObject).union`). A `noneify` failure
/// (`SomeInteger.noneify` → `SomeObject.noneify`) is `UnionError`.
fn store_annotation_union(
    var: &Variable,
    existing: &SomeValue,
    incoming: &SomeValue,
) -> Result<bool, TyperError> {
    match crate::annotator::model::union(existing, incoming) {
        Ok(u) if u != *existing => {
            stamp_someptr_lltype(var, &u);
            *var.annotation.borrow_mut() = Some(Rc::new(u));
            Ok(true)
        }
        Ok(_) => Ok(false),
        Err(_) => {
            let current = somevalue_to_valuetype(existing);
            let ty = somevalue_to_valuetype(incoming);
            Err(TyperError::message(format!(
                "UnionError: {current:?} ∪ {ty:?} at {} \
                 (binaryop.py pairtype(SomeObject, SomeNone).union)",
                var.name()
            )))
        }
    }
}

/// `result_exc.rs materialize_call_arg_constant` emits
/// `(OpKind::ConstNone, ConcreteType::GcRef)` for Option-payload `None`.
/// `emit_unit` / `AggregateShape::Void` keep `concretetype` Void — that
/// path stays `infer_op_type` Void (`rnone.py` `NoneRepr`, empty-return).
fn option_payload_const_none_incoming(result: &Variable, kind: &OpKind) -> Option<Incoming> {
    if !matches!(kind, OpKind::ConstNone) {
        return None;
    }
    if FunctionGraph::concretetype_of(result) == crate::model::ConcreteType::GcRef {
        Some(Incoming::Annotation(Rc::new(SomeValue::None_(
            SomeNone::new(),
        ))))
    } else {
        None
    }
}

fn is_str_family_shell(sv: &SomeValue) -> bool {
    matches!(
        sv,
        SomeValue::String(_)
            | SomeValue::Char(_)
            | SomeValue::UnicodeString(_)
            | SomeValue::UnicodeCodePoint(_)
    )
}

fn is_integer_family_shell(sv: &SomeValue) -> bool {
    matches!(sv, SomeValue::Integer(_) | SomeValue::Bool(_))
}

fn is_integer_family_flat(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Int | ValueType::Unsigned | ValueType::Bool)
}

/// `binaryop.py` `pairtype(SomeInteger, SomeInteger).union`. `SomeBool`
/// is a `SomeInteger` subclass so Bool joins here too. Flat `Int`
/// shells as `SomeInteger::default()` (`nonneg=false`); a signed value
/// that is not proven nonneg still UnionErrors against Unsigned.
fn merge_integer_family_binding(
    var: &Variable,
    incoming: Rc<SomeValue>,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    match read_binding_some(var) {
        None => {
            *var.annotation.borrow_mut() = Some(incoming);
            Ok(true)
        }
        Some(existing) if is_integer_family_shell(existing.as_ref()) => {
            match crate::annotator::model::union(existing.as_ref(), incoming.as_ref()) {
                Ok(u) if u != *existing => {
                    *var.annotation.borrow_mut() = Some(Rc::new(u));
                    Ok(true)
                }
                Ok(_) => Ok(false),
                Err(_) => {
                    let current = somevalue_to_valuetype(&existing);
                    let ty = somevalue_to_valuetype(&incoming);
                    Err(TyperError::message(format!(
                        "UnionError: {current:?} ∪ {ty:?} at {} \
                         (binaryop.py pairtype(SomeInteger, SomeInteger).union)",
                        var.name()
                    )))
                }
            }
        }
        Some(existing) if matches!(existing.as_ref(), SomeValue::Ptr(_)) => {
            union_ptr_object(var, existing.as_ref(), incoming.as_ref())
        }
        Some(_) => merge_flat(var, somevalue_to_valuetype(&incoming), bookkeeper),
    }
}

/// `llannotation.py` `pairtype(SomePtr, SomePtr).union`. Distinct
/// `ll_ptrtype` is `UnionError`; a non-Ptr cell goes through
/// `merge_flat` (`pairtype(SomeObject, SomeObject).union`).
fn merge_ptr_family_binding(
    var: &Variable,
    incoming: Rc<SomeValue>,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    match read_binding_some(var) {
        None => {
            stamp_someptr_lltype(var, incoming.as_ref());
            *var.annotation.borrow_mut() = Some(incoming);
            Ok(true)
        }
        Some(existing) if matches!(existing.as_ref(), SomeValue::Ptr(_)) => {
            // `consider_op` / `setbinding` recomputes the op result.
            // `llannotation.py pairtype(SomePtr, SomePtr).union` returns a
            // fresh non-constant `SomePtr(p1.ll_ptrtype)` (UnionError on
            // different types). `annrpython.py mergeinputargs` /
            // `model.py unionof` keep the old cell only when the union
            // equals it: same `ll_ptrtype` and the same constness/const
            // value stay unchanged (a recomputed `ann_nullptr`
            // `immutablevalue` must not churn); otherwise rebind through
            // `model::union`. Independently minted nullptr constants
            // compare equal through `_abstract_ptr.__eq__`.
            if let (SomeValue::Ptr(existing_ptr), SomeValue::Ptr(incoming_ptr)) =
                (existing.as_ref(), incoming.as_ref())
                && existing_ptr.ll_ptrtype == incoming_ptr.ll_ptrtype
                && someptr_same_const(existing_ptr, incoming_ptr)
            {
                return Ok(false);
            }
            match crate::annotator::model::union(existing.as_ref(), incoming.as_ref()) {
                Ok(u) if u != *existing => {
                    stamp_someptr_lltype(var, &u);
                    *var.annotation.borrow_mut() = Some(Rc::new(u));
                    Ok(true)
                }
                Ok(_) => Ok(false),
                Err(_) => {
                    let current = somevalue_to_valuetype(&existing);
                    let ty = somevalue_to_valuetype(&incoming);
                    Err(TyperError::message(format!(
                        "UnionError: {current:?} ∪ {ty:?} at {} \
                         (llannotation.py pairtype(SomePtr, SomePtr).union)",
                        var.name()
                    )))
                }
            }
        }
        Some(existing) if is_integer_family_shell(existing.as_ref()) => {
            union_ptr_object(var, existing.as_ref(), incoming.as_ref())
        }
        Some(_) => merge_flat(var, somevalue_to_valuetype(&incoming), bookkeeper),
    }
}

/// `llannotation.py` `pairtype(SomePtr, SomeObject).union`.
/// `history.getkind` of `Ptr(FuncType)` is `'int'`, so the flat
/// discriminator is Int, but the shells still UnionError.
fn union_ptr_object(
    var: &Variable,
    existing: &SomeValue,
    incoming: &SomeValue,
) -> Result<bool, TyperError> {
    match crate::annotator::model::union(existing, incoming) {
        Ok(u) if u != *existing => {
            stamp_someptr_lltype(var, &u);
            *var.annotation.borrow_mut() = Some(Rc::new(u));
            Ok(true)
        }
        Ok(_) => Ok(false),
        Err(_) => {
            let current = somevalue_to_valuetype(existing);
            let ty = somevalue_to_valuetype(incoming);
            Err(TyperError::message(format!(
                "UnionError: {current:?} ∪ {ty:?} at {} \
                 (llannotation.py pairtype(SomePtr, SomeObject).union)",
                var.name()
            )))
        }
    }
}

/// `binaryop.py` `pairtype(SomeString, SomeString).union` /
/// `pairtype(SomeChar, SomeString).union`. `bookkeeper.py`
/// `immutablevalue` of a `str` constant lands here as `SomeString` /
/// `SomeChar` instead of flattening through `somevalue_to_valuetype`.
fn merge_str_family_binding(
    var: &Variable,
    incoming: Rc<SomeValue>,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    match read_binding_some(var) {
        None => {
            *var.annotation.borrow_mut() = Some(incoming);
            Ok(true)
        }
        Some(existing) if is_str_family_shell(existing.as_ref()) => {
            match crate::annotator::model::union(existing.as_ref(), incoming.as_ref()) {
                Ok(u) if u != *existing => {
                    *var.annotation.borrow_mut() = Some(Rc::new(u));
                    Ok(true)
                }
                Ok(_) => Ok(false),
                Err(_) => merge_flat(var, ValueType::Str, bookkeeper),
            }
        }
        Some(_) => merge_flat(var, ValueType::Str, bookkeeper),
    }
}

fn is_type_shaped(sv: &SomeValue) -> bool {
    matches!(sv, SomeValue::TypeOf(_) | SomeValue::Type(_))
}

/// `binaryop.py pairtype(SomeTypeOf, SomeTypeOf)` merges `is_type_of`;
/// `pairtype(SomeType, SomeType)` / `pairtype(SomeType, SomeConstantType)`
/// / `pairtype(SomeConstantType, SomeType)` are owned by `annotator::union`.
/// Call that first when it already has the arm (PBC ∪ PBC).
fn union_type_shaped(existing: &SomeValue, incoming: &SomeValue) -> Result<SomeValue, TyperError> {
    match crate::annotator::model::union(existing, incoming) {
        Ok(u) => return Ok(u),
        Err(_) => {}
    }
    if let (SomeValue::TypeOf(a), SomeValue::TypeOf(b)) = (existing, incoming) {
        let mut vars = a.is_type_of.clone();
        for v in &b.is_type_of {
            if !vars.iter().any(|e| e == v) {
                vars.push(Rc::clone(v));
            }
        }
        let mut result = SomeTypeOf::new(vars);
        if a.is_constant() && b.is_constant() && a.base.const_box == b.base.const_box {
            result.base.const_box = a.base.const_box.clone();
        }
        return Ok(SomeValue::TypeOf(result));
    }
    Err(TyperError::message(format!(
        "UnionError: type-shaped {existing:?} ∪ {incoming:?} \
         (binaryop.py pairtype(SomeTypeOf, SomeTypeOf) / pairtype(SomeType, SomeType))"
    )))
}

/// `binaryop.py` `pairtype(SomeObject, SomeObject).union`.
fn union_into_binding(
    var: &Variable,
    existing: &SomeValue,
    incoming: &SomeValue,
) -> Result<bool, TyperError> {
    match crate::annotator::model::union(existing, incoming) {
        Ok(u) if u != *existing => {
            *var.annotation.borrow_mut() = Some(Rc::new(u));
            Ok(true)
        }
        Ok(_) => Ok(false),
        Err(_) => {
            let current = somevalue_to_valuetype(existing);
            let ty = somevalue_to_valuetype(incoming);
            Err(TyperError::message(format!(
                "UnionError: {current:?} ∪ {ty:?} at {} \
                 (binaryop.py pairtype(SomeObject, SomeObject).union)",
                var.name()
            )))
        }
    }
}

fn merge_type_shaped_binding(
    var: &Variable,
    incoming: Rc<SomeValue>,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    match read_binding_some(var) {
        None => {
            *var.annotation.borrow_mut() = Some(incoming);
            Ok(true)
        }
        Some(existing)
            if is_type_shaped(existing.as_ref())
                || crate::annotator::model::is_some_constant_type(existing.as_ref()) =>
        {
            let merged = union_type_shaped(existing.as_ref(), incoming.as_ref())?;
            if merged != *existing {
                *var.annotation.borrow_mut() = Some(Rc::new(merged));
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Some(existing) if matches!(existing.as_ref(), SomeValue::Impossible) => {
            merge_flat(var, somevalue_to_valuetype(&incoming), bookkeeper)
        }
        // `binaryop.py` has no `pairtype(SomeType, SomeInstance)` /
        // `pairtype(SomeTypeOf, SomeInstance)`; the mix is
        // `pairtype(SomeObject, SomeObject).union` → UnionError.
        // Flattening through `somevalue_to_valuetype` would shell
        // `Ref(None)` and let `SomeInstance` absorb it.
        Some(existing) => union_into_binding(var, existing.as_ref(), incoming.as_ref()),
    }
}

/// `SomeObject.__eq__` on `const` uses `_abstract_ptr.__eq__` (null-null)
/// is equal). Flow-space `Constant` keys an `LLPtr` by Hashable identity;
/// independently minted `ann_nullptr` cells still match as the same nullptr.
fn someptr_same_const(
    a: &crate::annotator::model::SomePtr,
    b: &crate::annotator::model::SomePtr,
) -> bool {
    match (&a.base.const_box, &b.base.const_box) {
        (None, None) => true,
        (Some(ca), Some(cb)) => match (&ca.value, &cb.value) {
            (ConstValue::LLPtr(pa), ConstValue::LLPtr(pb)) => {
                pa._TYPE == pb._TYPE && pa.as_ref() == pb.as_ref()
            }
            _ => ca == cb,
        },
        _ => false,
    }
}

fn merge_instance_pair(var: &Variable, existing: &SomeValue, incoming_sv: &SomeValue) -> bool {
    match crate::annotator::model::union(existing, incoming_sv) {
        Ok(u) => {
            if u != *existing {
                *var.annotation.borrow_mut() = Some(Rc::new(u));
                true
            } else {
                false
            }
        }
        Err(_) => {
            let can_be_none = match (existing, incoming_sv) {
                (SomeValue::Instance(a), SomeValue::Instance(b)) => a.can_be_none || b.can_be_none,
                _ => false,
            };
            let root =
                SomeValue::Instance(SomeInstance::new(None, can_be_none, Default::default()));
            if root != *existing {
                *var.annotation.borrow_mut() = Some(Rc::new(root));
                true
            } else {
                false
            }
        }
    }
}

fn merge_flat(
    var: &Variable,
    ty: ValueType,
    bookkeeper: Option<&Rc<Bookkeeper>>,
) -> Result<bool, TyperError> {
    let ty = declared_ref_valuetype(ty, bookkeeper);
    // Incoming `Unknown` is bottom (empty source): `union(x, bottom) = x`.
    if matches!(ty, ValueType::Unknown) {
        return Ok(false);
    }
    // Flat `Int` / `Unsigned` / `Bool` into a live Integer/Bool shell
    // keeps `nonneg` via `pairtype(SomeInteger, SomeInteger).union`.
    // `valuetype_to_someshell(Int)` is `nonneg=false`.
    if is_integer_family_flat(&ty)
        && read_binding_some(var)
            .as_deref()
            .is_some_and(is_integer_family_shell)
        && let Some(shell) = valuetype_to_someshell(&ty)
    {
        return merge_integer_family_binding(var, Rc::new(shell), bookkeeper);
    }
    let incoming = shell_for(&ty, bookkeeper);
    // Existing `SomeNone` ∪ a later concrete (`annrpython.py
    // mergeinputargs`; `binaryop.py pairtype(SomeNone, SomeObject).union`).
    // A Flat `Str` after an Option-payload `None` is this reverse of
    // `merge_none_incoming`.
    if let Some(existing) = read_binding_some(var)
        && matches!(existing.as_ref(), SomeValue::None_(_))
        && let Some(incoming_sv) = incoming.as_ref()
    {
        return store_annotation_union(var, existing.as_ref(), incoming_sv);
    }
    if let (Some(existing), Some(incoming_sv)) = (read_binding_some(var), incoming.as_ref())
        && matches!(existing.as_ref(), SomeValue::Instance(_))
        && matches!(incoming_sv, SomeValue::Instance(_))
    {
        Ok(merge_instance_pair(var, existing.as_ref(), incoming_sv))
    } else {
        let current = read_binding(var);
        let merged = union_type(&current, &ty);
        if merged == current {
            // `somevalue_to_valuetype` projects `TypeOf` / `Type` to
            // `Ref(None)`, so a classless instance is the same flat
            // type. `binaryop.py` `pairtype(SomeObject, SomeObject).union`
            // still raises.
            if let Some(existing) = read_binding_some(var)
                && matches!(existing.as_ref(), SomeValue::TypeOf(_) | SomeValue::Type(_))
                && let Some(incoming_sv) = incoming.as_ref()
            {
                return union_into_binding(var, existing.as_ref(), incoming_sv);
            }
            Ok(false)
        } else if matches!(merged, ValueType::Unknown)
            && !matches!(current, ValueType::Unknown)
            && !matches!(ty, ValueType::Unknown)
        {
            // `union_type` `_` arm: two bound concretes. Upstream
            // `binaryop.py` `pairtype(SomeObject, SomeObject).union`
            // raises `UnionError(obj1, obj2)`.
            Err(TyperError::message(format!(
                "UnionError: {current:?} ∪ {ty:?} at {} (binaryop.py pairtype(SomeObject, SomeObject).union)",
                var.name()
            )))
        } else {
            setbinding_with(var, merged, bookkeeper);
            Ok(true)
        }
    }
}

fn const_value_type(value: &ConstValue) -> ValueType {
    match value {
        ConstValue::Int(_)
        | ConstValue::RInt(_, _, _)
        | ConstValue::SpecTag(_)
        | ConstValue::AddressOffset(_)
        | ConstValue::InheritanceId { .. }
        | ConstValue::LLAddress(_)
        | ConstValue::AddressAsInt(_)
        | ConstValue::UniChar(_) => ValueType::Int,
        // `bookkeeper.py immutablevalue`: a bool constant is `SomeBool`.
        ConstValue::Bool(_) => ValueType::Bool,
        ConstValue::UInt(_) => ValueType::Unsigned,
        ConstValue::Int128(_) => ValueType::Int128,
        ConstValue::UInt128(_) => ValueType::UInt128,
        ConstValue::Float(_) | ConstValue::LongFloat(_) => ValueType::Float,
        ConstValue::SingleFloat(_) => ValueType::SingleFloat,
        ConstValue::Placeholder => ValueType::Unknown,
        // `bookkeeper.py immutablevalue` `issubclass(tp, str)` →
        // `SomeString` / `SomeChar`. Link constants take that shell
        // when a bookkeeper is present (`incoming_from_link_arg`);
        // the flat fallback is `Str` (`rstr.py` one string type).
        ConstValue::ByteStr(_) | ConstValue::UniStr(_) => ValueType::Str,
        // RPython `Constant(None)` is annotated as `SomeNone`
        // (`rpython/annotator/annrpython.py annotation immutablevalue(None)`
        // → `SomeNone()` per `annotator/model.py`), distinct from
        // `SomeInteger` / `SomeString`.  Pyre's `ValueType` lacks a
        // dedicated `SomeNone` variant, so collapse to `Ref` — the
        // closest match for the dominant downstream consumer where
        // None flows into a `Ptr` target and the rtyper emits
        // `inputconst(Ptr, None)` per `pairtype(NoneRepr, Repr).
        // convert_from_to` (`rpython/rtyper/rnone.py`).  For the
        // `Constant(None) → NoneRepr` target case (where the rtyper
        // returns `inputconst(Void, None)` per `rnone.py`),
        // construction sites such as `set_return` write the target
        // repr onto `Constant.concretetype`; the rtyping-layer
        // projection at `legacy_resolve::link_arg_concrete_type` then
        // honours that hint to materialise `Void` without forcing
        // every None through the annotation layer as `Void`.
        ConstValue::None
        | ConstValue::Atom(_)
        | ConstValue::Dict(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_)
        | ConstValue::Opaque(_) => ValueType::Ref(None),
        // `history.getkind` of `Ptr(FuncType)` is `'int'` (`FuncType._gckind
        // == 'raw'`). GC and other raw pointees stay `Ref` here.
        ConstValue::LLPtr(p) => match &p._TYPE.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(_) => ValueType::Int,
            _ => ValueType::Ref(None),
        },
        // Bare-constant repr selection after annotation: one FrozenDesc
        // (`getFrozenPBCRepr` → `SingleFrozenPBCRepr`).
        // `SingleFrozenPBCRepr.lowleveltype` is `Void`; `history.getkind(Void)`
        // is `"void"`. `follow_link` does not build the annotation shell
        // from this arm (`valuetype_to_someshell(Void)` is BOTTOM); it
        // writes `bookkeeper.immutablevalue` (`SomePBC`).
        ConstValue::Repr(_) => ValueType::Void,
    }
}

/// Infer the output type of an operation from its inputs.
///
/// RPython equivalent: annotator dispatch (e.g., `annotate_int_add`
/// returns `SomeInteger()`).
/// `rstr.py` `mallocstr` → `malloc(STR, n)` → `Ptr(STR)` / `SomeString`.
#[allow(non_snake_case)]
fn malloc_owner_is_str(
    owner: &str,
    TYPE: Option<&crate::translator::rtyper::lltypesystem::lltype::LowLevelType>,
) -> bool {
    if crate::front::mir::is_str_family_class_root(owner)
        || crate::front::mir::is_rstr_str_layout_declaration(owner)
    {
        return true;
    }
    let Some(t) = TYPE else {
        return false;
    };
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    let name = match t {
        LowLevelType::Struct(s) => s._name.as_str(),
        LowLevelType::Ptr(p) => match &p.TO {
            PtrTarget::Struct(s) => s._name.as_str(),
            _ => return false,
        },
        _ => return false,
    };
    crate::front::mir::is_str_family_class_root(name)
        || crate::front::mir::is_rstr_str_layout_declaration(name)
}

fn malloc_new_result_type(owner: &str) -> ValueType {
    if malloc_owner_is_str(owner, None) {
        ValueType::Str
    } else {
        ValueType::Ref(Some(owner.to_string()))
    }
}

fn infer_op_type(kind: &OpKind) -> ValueType {
    match kind {
        OpKind::Input { ty, .. } => ty.clone(),
        OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => ValueType::Int,
        OpKind::ConstUInt(_) => ValueType::Unsigned,
        OpKind::ConstInt128(_) => ValueType::Int128,
        OpKind::ConstSingleFloat(_) => ValueType::SingleFloat,
        OpKind::ConstUInt128(_) => ValueType::UInt128,
        OpKind::ConstBool(_) => ValueType::Bool,
        // `_we_are_jitted` symbolic carries its own concretetype
        // (`Bool` for `we_are_jitted() -> bool`).
        OpKind::ConstSymbolic { ty, .. } => ty.clone(),
        OpKind::ConstFloat(_) => ValueType::Float,
        // rstr `Ptr(STR)` (`rstr.py` `StringRepr.convert_const`).
        OpKind::ConstStr(_) => ValueType::Str,
        // Wrapped immortal `W_UnicodeObject` (Python str), not rstr.
        // Singleton instance pointer (unit-variant PBC).  The legacy
        // annotator sees this as a plain ref-typed value; concrete
        // class identity stays in the `HostObject` carrier itself.
        OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_) => {
            ValueType::Ref(None)
        }
        // The annotator `None` constant is `Void`-typed (`concretetype =
        // lltype.Void`), not a ref.  Only reaches the legacy annotator in a
        // graph that drops before `decompose_slice_args` consumes the
        // getslice `stop` operand (which discards it for `[start:]`).
        OpKind::ConstNone => ValueType::Void,
        OpKind::RawLoad { item_ty, .. } | OpKind::GcLoadIndexed { item_ty, .. } => {
            item_ty.clone()
        }
        OpKind::RawStore { .. } => ValueType::Void,
        OpKind::FieldRead { ty, .. } => ty.clone(),
        OpKind::FieldWrite { .. } => ValueType::Void,
        // `rstr.py` `mallocstr` → `malloc(STR, n)` → `Ptr(STR)` /
        // `SomeString`. A str-family owner is `Str`, not `Ref(PyObject)`.
        OpKind::New { owner } => malloc_new_result_type(owner),
        OpKind::NewWithVtable { owner, .. } => malloc_new_result_type(owner),
        // Fixed-size `lltype.malloc(T, flavor='raw')` returns a Signed
        // address except str-family TYPE/owner (`mallocstr`). Immortal
        // `alloc_varsize_with_gc_header_immortal` of a STR is dest-Str
        // after the Call (`GCRefRepr.convert_from_to`). `lltype.free`
        // returns nothing.
        OpKind::RawMalloc { owner, TYPE, .. } => {
            if malloc_owner_is_str(owner, TYPE.as_ref()) {
                ValueType::Str
            } else {
                ValueType::Int
            }
        }
        OpKind::RawFree { .. } => ValueType::Void,
        // `new_array_clear` yields a `Ref` to the freshly allocated,
        // zero-cleared items array (`Ptr(GcArray(OBJECTPTR))`).
        OpKind::NewArray { .. } | OpKind::NewArrayClear { .. } => ValueType::Ref(None),
        // `newlist_clear` yields a `Ref` to the freshly allocated
        // resizable-list `GcStruct` header (the `"list"` struct).
        OpKind::NewListClear { .. } => ValueType::Ref(None),
        OpKind::ArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::ArrayLen { .. } => ValueType::Int,
        OpKind::ArrayWrite { .. } => ValueType::Void,
        OpKind::InteriorFieldRead { item_ty, .. } => item_ty.clone(),
        OpKind::InteriorFieldWrite { .. } => ValueType::Void,
        // Frontend `DecodedConst::Str` is `Call(["__str_const", text])`
        // (`is_str_const_define`) with `result_ty: Ref(None)` (`getkind`
        // of rstr `Ptr(STR)` is `'r'`). Upstream that value is a bare
        // Constant; `bookkeeper.py immutablevalue` `issubclass(tp, str)`
        // → `SomeString`. `rstr.py` `StringRepr.convert_const`.
        OpKind::Call { .. } if crate::model::is_str_const_define(kind) => ValueType::Str,
        OpKind::Call {
            result_ty,
            target,
            args,
            ..
        } => {
            // When `result_ty` is `Ref(Some(root))` and the dest is
            // str-family, return `Str` (`is_str_family_class_root`;
            // `rstr.py` one string type). `Ref(None)` of a non-literal
            // callee stays `Ref` — residual `'r'` needs the callee
            // graph (`cutover.rs` `declared_return_annotation`).
            if let ValueType::Ref(Some(root)) = result_ty
                && crate::front::mir::is_str_family_class_root(root)
            {
                return ValueType::Str;
            }
            if result_ty != &ValueType::Unknown {
                return result_ty.clone();
            }
            infer_call_result_type(target, args)
        }
        OpKind::GuardTrue { .. } | OpKind::GuardFalse { .. } => ValueType::Void,
        OpKind::VableFieldRead { ty, .. } => ty.clone(),
        OpKind::VableFieldWrite { .. } => ValueType::Void,
        OpKind::VableArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::VableArrayWrite { .. } => ValueType::Void,
        // `arraylen_vable/rdd>i` — a length is a `Signed`, never the
        // element kind.
        OpKind::VableArrayLen { .. } => ValueType::Int,
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } if op == "same_as" => {
            if result_ty != &ValueType::Unknown {
                result_ty.clone()
            } else {
                read_binding(operand)
            }
        }
        // `hint(v, ...)` returns `v` unchanged — `rlib/jit.py:hint` is
        // identity at the annotation level, so the result carries the
        // operand's type (the same dispatch as `same_as`).
        OpKind::Hint { value, .. } => read_binding(value),
        // RPython `rfloat.py:rtype_neg` / `intop.py:rtype_neg`: `neg`
        // preserves the operand's lowleveltype (Float vs Int).  Pyre's
        // pre-jtransform graph emits a single `OpKind::UnaryOp` op="neg"
        // for both kinds, so resolve from the operand annotation when
        // `result_ty` is still Unknown.  Without this, `-z` where `z:
        // f64` annotates to Int (the previous default), which then
        // poisons every downstream phi-merge inputarg via union(Int,
        // Float) → Unknown (`rtyper.py setconcretetype` has no default).
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } if op == "neg" => {
            if result_ty != &ValueType::Unknown {
                result_ty.clone()
            } else {
                // RPython `intop.rtype_neg` / `rfloat.rtype_neg` dispatch
                // on the operand `SomeValue`'s lowleveltype.  Read the
                // orthodox `Variable.annotation` cell so a Float operand
                // keeps its Float result; absent/Unknown-cleared
                // operands default to Int matching the historical
                // `.types.get(...).unwrap_or(Int)` semantics for the
                // dominant "operand not yet annotated" case.
                read_binding_some(operand)
                    .map(|s| somevalue_to_valuetype(&s))
                    .unwrap_or(ValueType::Int)
            }
        }
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            if result_ty != &ValueType::Unknown {
                result_ty.clone()
            } else {
                ValueType::Int // Arithmetic defaults to Int
            }
        }
        OpKind::VableForce { .. }
        | OpKind::Live { .. }
        | OpKind::GuardValue { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::RecordKnownResult { .. }
        // jtransform.py — `record_quasiimmut_field` has no result.
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::JitForceQuasiImmutable { .. }
        // jtransform.py:1707,1718 — jit_merge_point / loop_header have no
        // result; upstream emits them with `op1 = SpaceOperation(..., None)`.
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. } => ValueType::Void,
        OpKind::CurrentTraceLength => ValueType::Int,
        // `jit.isconstant` / `jit.isvirtual` are declared at
        // `rlib/jit.py` returning `NonConstant(False)` — a
        // Python `bool`.  RPython annotates them as `SomeBool`, not
        // `SomeInteger`.
        OpKind::IsConstant { .. } | OpKind::IsVirtual { .. } => ValueType::Bool,
        // `isinstance(x, T)` is semantically a `bool`; the carried
        // `result_ty` is only a constructor hint and may be left as
        // `Unknown` by frontends that did not set it explicitly.
        // Returning `Unknown` here widens phi merges and degrades
        // downstream typing; enforce `Bool` directly so the inferred
        // type matches `rclass.py InstanceRepr.rtype_isinstance`'s
        // `LowLevelType::Bool` result.
        OpKind::IsInstance { .. } => ValueType::Bool,
        // RPython: vtable entry is a `Ptr(FuncType)` address.
        OpKind::VtableMethodPtr { .. } => ValueType::Int,
        // The class pointer, as the hint ops carry it.
        OpKind::GuardClass { .. } => ValueType::Int,
        OpKind::IndirectCall { result_ty, .. } => result_ty.clone(),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. }
        | OpKind::ConditionalCallValue { result_kind, .. } => kind_char_to_value_type(*result_kind),
        OpKind::ConditionalCall { .. } => ValueType::Void,
        OpKind::Abort { .. } => ValueType::Unknown,
        // `newtuple` yields a `Ref` to the freshly allocated tuple
        // object (RPython `SomeTuple` lowers to `Ptr<GcStruct>`).
        OpKind::NewTuple { .. } => ValueType::Ref(None),
        // `newlist` yields a `Ref` to the freshly allocated list object
        // (RPython `SomeList` lowers to `Ptr<GcStruct>`).
        OpKind::NewList { .. } => ValueType::Ref(None),
        // `getslice` yields a `Ref` to the freshly copied list object
        // (RPython `SomeList` lowers to `Ptr<GcStruct>`).
        OpKind::GetSlice { .. } => ValueType::Ref(None),
        // Already-lowered blackhole opnames (`strlen` / `strgetitem` /
        // `newstr` / …).  Spine B plants these from `jtransform_opname`;
        // Skip-spine string helpers and `__string_byte_getitem` /
        // `__strlen` rewrites plant the same opnames on Spine A so
        // `pyre_cpu` `bh_str*` can see a `W_UnicodeObject`.  The result
        // kind is a function of the opname.
        OpKind::LoweredBlackholeOp { opname, .. } => match opname.as_str() {
            "strlen" | "unicodelen" | "strgetitem" | "unicodegetitem" => ValueType::Int,
            // `n <= m < p`, a bool in the signed register bank.
            "int_between" => ValueType::Bool,
            "newstr" | "newunicode" => ValueType::Ref(None),
            "strsetitem" | "unicodesetitem" | "copystrcontent" | "copyunicodecontent" => {
                ValueType::Void
            }
            _ => ValueType::Unknown,
        },
        // `LoadStatic` carries the declared `ValueType` of the static
        // directly (extracted from the `syn::Item::Static.ty` at
        // `register::extract_static_decls`).
        OpKind::LoadStatic { ty, .. } => ty.clone(),
    }
}

fn kind_char_to_value_type(kind: char) -> ValueType {
    match kind {
        'i' => ValueType::Int,
        'r' => ValueType::Ref(None),
        'f' => ValueType::Float,
        'v' => ValueType::Void,
        _ => ValueType::Unknown,
    }
}

fn infer_call_result_type(
    target: &crate::model::CallTarget,
    args: &[crate::model::LinkArg],
) -> ValueType {
    if crate::call::is_int_arithmetic_target(target) {
        return ValueType::Int;
    }
    // Same synthetic as `is_str_const_define`, for a `result_ty: Unknown`
    // call that did not take the dedicated arm.
    if matches!(
        target,
        CallTarget::FunctionPath { segments, .. }
            if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const"
    ) {
        return ValueType::Str;
    }
    ValueType::Unknown
}

/// Merge two annotations (RPython `unionof()`).
///
/// `Unknown` is identity here (bottom / unbound). The `_` arm is the
/// conflict of two bound concretes; [`merge_flat`] reports
/// `binaryop.py` `pairtype(SomeObject, SomeObject).union` as `UnionError`.
fn union_type(a: &ValueType, b: &ValueType) -> ValueType {
    if a == b {
        return a.clone();
    }
    match (a, b) {
        (ValueType::Unknown, other) | (other, ValueType::Unknown) => {
            if other == &ValueType::Unknown {
                ValueType::Unknown
            } else {
                other.clone()
            }
        }
        (ValueType::Ref(_), ValueType::Ref(_)) => ValueType::Ref(None),
        // SomeBool ⊂ SomeInteger: `pair(SomeBool, SomeInteger).union`
        // resolves by inheritance to `pair(SomeInteger, SomeInteger).union`
        // (`binaryop.py`; there is no `pairtype(SomeBool, SomeInteger)`
        // override).  SomeBool is `nonneg=True, unsigned=False,
        // knowntype=bool` (`model.py`); bool→int is normalised at
        // `binaryop.py`.  Against signed Int the same-signedness
        // branch returns SomeInteger(knowntype=int) = Int.  Against Unsigned
        // (`unsigned=True, nonneg=True, knowntype=r_uint`) the differing-
        // signedness branch reaches `elif t1 is int` with `int1.nonneg ==
        // True`, so NO UnionError fires and `knowntype = r_uint` →
        // SomeInteger(unsigned) = Unsigned (`binaryop.py`).  The
        // UnionError at `binaryop.py` is reserved for SIGNED Int
        // (`nonneg=False`) ∪ Unsigned — the `(Int, Unsigned)` case left to
        // the `_` arm below.
        (ValueType::Bool, ValueType::Int) | (ValueType::Int, ValueType::Bool) => ValueType::Int,
        (ValueType::Bool, ValueType::Unsigned) | (ValueType::Unsigned, ValueType::Bool) => {
            ValueType::Unsigned
        }
        _ => ValueType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        CallTarget, ExitSwitch, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, ValueType,
        VecFieldPart, exception_exitcase,
    };

    #[test]
    fn shell_for_str_family_ref_is_somestring_even_with_struct_rows() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::model::{SomeString, SomeValue, union};
        let bk = Rc::new(Bookkeeper::new());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            "string::String".to_string(),
            vec![
                ("cap".to_string(), "usize".to_string()),
                ("ptr".to_string(), "*mut u8".to_string()),
                ("len".to_string(), "usize".to_string()),
            ],
        );
        bk.set_struct_fields(Rc::new(fields));
        let shell = shell_for(
            &ValueType::Ref(Some("string::String".to_string())),
            Some(&bk),
        )
        .expect("str-family Ref must project");
        let SomeValue::String(s) = &shell else {
            panic!(
                "str-family Ref must stay SomeString when String has struct rows, got {shell:?}"
            );
        };
        assert!(
            !s.inner.no_nul,
            "str-family Ref is SomeString(no_nul=False) (SomeString.__init__ default)"
        );
        let s_new = SomeValue::String(SomeString::new(false, false));
        let joined = union(&shell, &s_new).expect("str-family Ref must union with String::new()");
        assert!(
            matches!(joined, SomeValue::String(_)),
            "str-family Ref ∪ String::new() must stay SomeString, got {joined:?}"
        );
    }

    #[test]
    fn annotates_const_int() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        graph.set_return(entry, Some(v_var.clone()));

        annotate(&graph);
        assert_eq!(read_binding(&v_var), ValueType::Int);
    }

    #[test]
    fn const_repr_predicts_single_frozen_pbc_void() {
        // `SingleFrozenPBCRepr.lowleveltype` is `Void` (`rpbc.py`).
        let repr = ConstValue::Repr(crate::flowspace::model::ReprConst::new(
            crate::translator::rtyper::rint::signed_repr(),
        ));
        assert_eq!(const_value_type(&repr), ValueType::Void);
    }

    fn repr_const(
        handle: &std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr>,
    ) -> ConstValue {
        ConstValue::Repr(crate::flowspace::model::ReprConst::new(
            std::sync::Arc::clone(handle),
        ))
    }

    fn branch_start_to_preds(
        graph: &mut FunctionGraph,
        pred1: crate::model::BlockId,
        pred2: crate::model::BlockId,
    ) {
        let cond = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "switch".into(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_branch(graph.startblock, cond, pred1, vec![], pred2, vec![]);
    }

    fn two_pred_const_phi(
        left: ConstValue,
        right: ConstValue,
    ) -> (FunctionGraph, crate::flowspace::model::Variable) {
        let mut graph = FunctionGraph::new("repr_phi");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto_mixed(
            pred1,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(left))],
        );
        graph.set_goto_mixed(
            pred2,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(
                right,
            ))],
        );
        graph.set_return(join, Some(phi.clone()));
        (graph, phi)
    }

    #[test]
    fn same_repr_const_phi_is_void_some_pbc() {
        use crate::annotator::model::SomeObjectTrait;
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, phi) = two_pred_const_phi(repr_const(&handle), repr_const(&handle));
        annotate_inner(&graph, None).expect("same ReprConst is SingleFrozenPBCRepr");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::PBC(pbc)) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert!(pbc.is_constant());
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Void);
        assert!(!matches!(
            read_binding_some(&phi).as_deref(),
            Some(SomeValue::Impossible)
        ));
    }

    #[test]
    fn distinct_repr_const_phi_is_typer_error() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle_a: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let handle_b: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, _phi) = two_pred_const_phi(repr_const(&handle_a), repr_const(&handle_b));
        let err = annotate_inner(&graph, None).expect_err("distinct ReprConst merge");
        let msg = err.to_string();
        assert!(
            msg.contains("MultipleFrozenPBCRepr"),
            "error should name MultipleFrozenPBCRepr, got {msg}"
        );
        assert!(
            msg.contains("getFrozenPBCRepr"),
            "error should cite getFrozenPBCRepr, got {msg}"
        );
    }

    #[test]
    fn repr_const_mixed_with_non_repr_is_typer_error() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let (graph, _phi) = two_pred_const_phi(repr_const(&handle), ConstValue::Int(1));
        let err = annotate_inner(&graph, None).expect_err("Repr mixed with non-Repr");
        let msg = err.to_string();
        assert!(
            msg.contains("MultipleFrozenPBCRepr"),
            "error should name MultipleFrozenPBCRepr, got {msg}"
        );
        assert!(
            msg.contains("getFrozenPBCRepr"),
            "error should cite getFrozenPBCRepr, got {msg}"
        );
    }

    #[test]
    fn unreachable_pred_distinct_repr_is_accepted() {
        // `follow_link` / `complete_pending_blocks` never union a
        // startblock-unreachable predecessor into the dest annotation.
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle_a: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let handle_b: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let mut graph = FunctionGraph::new("unreachable_repr_phi");
        let entry = graph.startblock;
        let pred2 = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        graph.set_goto_mixed(
            entry,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(
                repr_const(&handle_a),
            ))],
        );
        graph.set_goto_mixed(
            pred2,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(
                repr_const(&handle_b),
            ))],
        );
        graph.set_return(join, Some(phi_args[0].clone()));
        refuse_merged_frozen_pbc_reprs(&graph).expect("unreachable distinct Repr does not join");
    }

    #[test]
    fn repr_const_and_same_singleton_variable_merge() {
        use crate::annotator::model::SomeObjectTrait;
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let c = repr_const(&handle);
        let bk = Rc::new(Bookkeeper::new());
        let s_pbc = bk
            .immutablevalue(&c)
            .expect("immutablevalue of Repr is SomePBC");

        let mut graph = FunctionGraph::new("repr_const_and_var");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();

        let src_var = graph.alloc_value_var();
        *src_var.annotation.borrow_mut() = Some(Rc::new(s_pbc));

        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto_mixed(
            pred1,
            join,
            vec![LinkArg::Const(crate::flowspace::model::Constant::new(c))],
        );
        graph.set_goto(pred2, join, vec![src_var]);
        graph.set_return(join, Some(phi.clone()));

        annotate_inner(&graph, Some(&bk)).expect("same singleton SomePBC merge");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::PBC(pbc)) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert!(pbc.is_constant());
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn hint_keeps_single_frozen_pbc() {
        use crate::translator::rtyper::rmodel::VoidRepr;

        let handle: std::sync::Arc<dyn crate::translator::rtyper::rmodel::Repr> =
            std::sync::Arc::new(VoidRepr::new());
        let bk = Rc::new(Bookkeeper::new());
        let s_pbc = bk
            .immutablevalue(&repr_const(&handle))
            .expect("immutablevalue of Repr is SomePBC");

        let mut graph = FunctionGraph::new("hint_pbc");
        let entry = graph.startblock;
        let value = graph.alloc_value_var();
        *value.annotation.borrow_mut() = Some(Rc::new(s_pbc));
        let result = graph
            .push_op_var(
                entry,
                OpKind::Hint {
                    value: value.clone(),
                    kind: crate::hints::HintKind::Promote,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result.clone()));

        annotate_inner(&graph, Some(&bk)).expect("hint keeps SomePBC");
        match read_binding_some(&result).as_deref() {
            Some(SomeValue::PBC(pbc)) => {
                assert_eq!(pbc.descriptions.len(), 1);
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
        assert!(!matches!(
            read_binding_some(&result).as_deref(),
            Some(SomeValue::Impossible)
        ));
        assert_eq!(read_binding(&result), ValueType::Void);
    }

    #[test]
    fn two_function_pbc_phi_merges_descriptions() {
        // `pairtype(SomePBC, SomePBC).union` unions FunctionDesc sets;
        // `getFrozenPBCRepr` is FrozenDesc-only (`FunctionsPBCRepr` later).
        use crate::annotator::description::FunctionDesc;
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;
        use std::cell::RefCell;

        let bk = Rc::new(Bookkeeper::new());
        let function_pbc = |name: &str| {
            SomeValue::PBC(SomePBC::new(
                vec![DescEntry::function(Rc::new(RefCell::new(
                    FunctionDesc::new(
                        bk.clone(),
                        None,
                        name,
                        Signature::new(vec![], None, None),
                        None,
                        None,
                    ),
                )))],
                false,
            ))
        };

        let mut graph = FunctionGraph::new("fn_pbc_phi");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();

        let src_a = graph.alloc_value_var();
        *src_a.annotation.borrow_mut() = Some(Rc::new(function_pbc("f1")));
        let src_b = graph.alloc_value_var();
        *src_b.annotation.borrow_mut() = Some(Rc::new(function_pbc("f2")));

        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto(pred1, join, vec![src_a]);
        graph.set_goto(pred2, join, vec![src_b]);
        graph.set_return(join, Some(phi.clone()));

        annotate_inner(&graph, Some(&bk)).expect("function PBC union must propagate");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::PBC(pbc)) => {
                assert_eq!(pbc.descriptions.len(), 2);
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn annotates_field_read_type() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base_var = graph.alloc_value_var();
        let v_var = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new("x", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v_var.clone()));

        annotate(&graph);
        assert_eq!(read_binding(&v_var), ValueType::Int);
    }

    #[test]
    fn annotates_call_with_int_args() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let a_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let b_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let result_var = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["w_int_add"]),
                    args: crate::model::call_args(vec![a_var.clone(), b_var.clone()]),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result_var.clone()));

        annotate(&graph);
        assert_eq!(read_binding(&a_var), ValueType::Int);
        assert_eq!(read_binding(&b_var), ValueType::Int);
        assert_eq!(read_binding(&result_var), ValueType::Int);
    }

    #[test]
    fn annotates_path_like_int_helper_call() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let a_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let b_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let result_var = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["crate", "math", "w_int_add"]),
                    args: crate::model::call_args(vec![a_var, b_var]),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result_var.clone()));

        annotate(&graph);
        assert_eq!(read_binding(&result_var), ValueType::Int);
    }

    #[test]
    fn propagates_across_blocks_via_phi() {
        // Test cross-block annotation propagation through Link args → inputargs
        let mut graph = FunctionGraph::new("phi_test");
        let entry = graph.startblock;

        // Entry: produce an Int value
        let val_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();

        // Create target block with one inputarg (Phi node)
        let (target, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();

        // Link: entry → target, passing val as the Phi arg
        graph.set_goto(entry, target, vec![val_var]);
        graph.set_return(target, Some(phi_var.clone()));

        annotate(&graph);
        // Phi should inherit Int from val via Link propagation
        assert_eq!(
            read_binding(&phi_var),
            ValueType::Int,
            "Phi node should receive Int annotation from Link args"
        );
    }

    #[test]
    fn raise_link_propagates_exception_pair_with_special_types() {
        // `annrpython.py follow_raise_link` binds the type slot with
        // `typeof([v_last_exc_value])` (`SomeTypeOf`). After rtyping,
        // `setup_block_entry` / `_convert_link` stamp `CLASSTYPE` whose
        // `getkind` is Signed.
        use super::super::legacy_resolve;
        use crate::codewriter::type_state::ConcreteType;

        let mut graph = FunctionGraph::new("raise_link");
        let entry = graph.startblock;
        let exc_block = graph.exceptblock;
        let (etype_var, evalue_var) = {
            let inputargs = &graph.block(exc_block).inputargs;
            (inputargs[0].clone(), inputargs[1].clone())
        };
        let last_exception_var = graph.alloc_value_var();
        let last_exc_value_var = graph.alloc_value_var();
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::from_variables(
                    &graph,
                    vec![last_exception_var.clone(), last_exc_value_var.clone()],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(last_exception_var.clone())),
                    Some(LinkArg::Value(last_exc_value_var.clone())),
                ),
            ],
        );

        annotate(&graph);
        assert!(
            matches!(
                read_binding_some(&last_exception_var).as_deref(),
                Some(SomeValue::TypeOf(_))
            ),
            "last_exception annotation is typeof([last_exc_value])"
        );
        assert_eq!(read_binding(&last_exception_var), ValueType::Ref(None));
        assert_eq!(read_binding(&last_exc_value_var), ValueType::Ref(None));
        assert!(
            matches!(
                read_binding_some(&etype_var).as_deref(),
                Some(SomeValue::TypeOf(_))
            ),
            "exceptblock[0] annotation is SomeTypeOf"
        );
        assert_eq!(read_binding(&etype_var), ValueType::Ref(None));
        assert_eq!(read_binding(&evalue_var), ValueType::Ref(None));

        legacy_resolve::resolve_types(&graph, &std::collections::HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&last_exception_var),
            ConcreteType::Signed,
            "raise-link seed overrides TypeOf projection with CLASSTYPE kind"
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&etype_var),
            ConcreteType::Signed
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&evalue_var),
            ConcreteType::GcRef
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&last_exc_value_var),
            ConcreteType::GcRef
        );
    }

    #[test]
    fn canraise_and_push_type_of_union_etype_as_typeof() {
        // Mixed exceptblock phi: a canraise `LastException` edge
        // (`follow_raise_link` typeof) and a `push_type_of` raise edge.
        // The `type` Call carries no pre-seeded annotation (`flowcontext.py
        // exc_from_raise` emits `op.type(w_value)`); the walker binds
        // `typeof([v])` (`unaryop.py type_SomeObject`). Both write
        // `SomeTypeOf`; `pairtype(SomeTypeOf, SomeTypeOf)` unions
        // `is_type_of`. `setup_block_entry` still stamps exceptblock[0] Signed.
        use super::super::legacy_resolve;
        use crate::codewriter::type_state::ConcreteType;

        let mut graph = FunctionGraph::new("mixed_raise_typeof");
        let entry = graph.startblock;
        let exc_block = graph.exceptblock;
        let raise_block = graph.create_block();
        let (etype_var, evalue_var) = {
            let inputargs = &graph.block(exc_block).inputargs;
            (inputargs[0].clone(), inputargs[1].clone())
        };
        let last_exception_var = graph.alloc_value_var();
        let last_exc_value_var = graph.alloc_value_var();
        let inst = graph
            .push_op_var(raise_block, OpKind::ConstRefNull, true)
            .unwrap();
        crate::front::exc_from_raise::set_raise_from_instance(&mut graph, raise_block, inst);
        let type_result = graph
            .block(raise_block)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { target, .. } if target == &CallTarget::function_path(["type"]) => {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("push_type_of emits op.type");
        assert!(
            type_result.annotation.borrow().is_none(),
            "front end does not pre-seed type() annotation"
        );
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::from_variables(&graph, vec![], raise_block, None),
                Link::from_variables(
                    &graph,
                    vec![last_exception_var.clone(), last_exc_value_var.clone()],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(last_exception_var.clone())),
                    Some(LinkArg::Value(last_exc_value_var.clone())),
                ),
            ],
        );

        annotate_inner(&graph, None).expect("canraise TypeOf ∪ push_type_of TypeOf");
        assert!(
            matches!(
                read_binding_some(&type_result).as_deref(),
                Some(SomeValue::TypeOf(_))
            ),
            "walker binds type() as SomeTypeOf (`unaryop.py type_SomeObject`)"
        );
        assert!(
            matches!(
                read_binding_some(&etype_var).as_deref(),
                Some(SomeValue::TypeOf(_))
            ),
            "exceptblock[0] stays SomeTypeOf across both raise edges"
        );
        assert_eq!(read_binding(&evalue_var), ValueType::Ref(None));

        legacy_resolve::resolve_types(&graph, &std::collections::HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&etype_var),
            ConcreteType::Signed,
            "exceptblock[0] is CLASSTYPE kind whether the incoming was \
             canraise or push_type_of"
        );
        assert_eq!(
            FunctionGraph::concretetype_of(&evalue_var),
            ConcreteType::GcRef
        );
    }

    #[test]
    fn raise_link_etype_typeof_names_dest_evalue() {
        // `follow_raise_link` dest `typeof(renaming[v_last_exc_value])`
        // names the exceptblock value inputarg, not the source
        // `last_exc_value`.
        let mut graph = FunctionGraph::new("raise_link_dest_typeof");
        let entry = graph.startblock;
        let exc_block = graph.exceptblock;
        let (etype_var, evalue_var) = {
            let inputargs = &graph.block(exc_block).inputargs;
            (inputargs[0].clone(), inputargs[1].clone())
        };
        let last_exception_var = graph.alloc_value_var();
        let last_exc_value_var = graph.alloc_value_var();
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::from_variables(
                    &graph,
                    vec![last_exception_var.clone(), last_exc_value_var.clone()],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(last_exception_var.clone())),
                    Some(LinkArg::Value(last_exc_value_var.clone())),
                ),
            ],
        );

        annotate(&graph);
        match read_binding_some(&etype_var).as_deref() {
            Some(SomeValue::TypeOf(t)) => {
                assert_eq!(
                    t.is_type_of.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
                    vec![&evalue_var]
                );
                assert_ne!(t.is_type_of[0].as_ref(), &last_exc_value_var);
            }
            other => panic!("exceptblock type is dest typeof, got {other:?}"),
        }
    }

    #[test]
    fn link_renames_typeof_to_dest_var() {
        // Block A computes `t = type(v)`; link A→B passes `(v, t)`.
        // `apply_renaming` rewrites `is_type_of` onto B's `v` inputarg.
        let mut graph = FunctionGraph::new("link_rename_typeof");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        let t = crate::front::exc_from_raise::push_type_of(&mut graph, entry, v.clone());
        let (b, b_args) = graph.create_block_with_arg_vars(2);
        let b_v = b_args[0].clone();
        let b_t = b_args[1].clone();
        graph.set_goto(entry, b, vec![v, t]);
        graph.set_return(b, Some(b_t.clone()));

        annotate(&graph);
        match read_binding_some(&b_t).as_deref() {
            Some(SomeValue::TypeOf(tof)) => {
                assert_eq!(
                    tof.is_type_of
                        .iter()
                        .map(|var| var.as_ref())
                        .collect::<Vec<_>>(),
                    vec![&b_v]
                );
            }
            other => panic!("B.t is renamed SomeTypeOf, got {other:?}"),
        }
    }

    #[test]
    fn link_typeof_var_not_on_link_is_sometype() {
        // Same as `link_renames_typeof_to_dest_var` but the link passes
        // only `t`. `apply_renaming` of `typeof([])` is `SomeType`.
        let mut graph = FunctionGraph::new("link_typeof_stripped");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        let t = crate::front::exc_from_raise::push_type_of(&mut graph, entry, v);
        let (b, b_args) = graph.create_block_with_arg_vars(1);
        let b_t = b_args[0].clone();
        graph.set_goto(entry, b, vec![t]);
        graph.set_return(b, Some(b_t.clone()));

        annotate(&graph);
        assert!(
            matches!(read_binding_some(&b_t).as_deref(), Some(SomeValue::Type(_))),
            "typeof([]) is SomeType when the is_type_of var is not on the link"
        );
    }

    #[test]
    fn typeof_phi_nullable_class_pbc_is_union_error() {
        // `unaryop.py type_SomeObject` → `SomeTypeOf`. A nullable class
        // PBC is a plain `SomePBC` (`SomePBC.noneify`), not
        // `SomeConstantType`. `pairtype(SomeObject, SomeObject).union`
        // is UnionError.
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::model::SomePBC;
        use crate::flowspace::model::HostObject;

        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class("C", vec![]);
        let classdesc =
            ClassDesc::new(&bk, pyobj, Some("C".to_string()), None, None).expect("ClassDesc::new");
        let nullable = SomeValue::PBC(SomePBC::new(vec![DescEntry::Class(classdesc)], true));

        let mut graph = FunctionGraph::new("typeof_nullable_class_pbc_phi");
        let pred_type = graph.create_block();
        let pred_pbc = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let inst = graph
            .push_op_var(pred_type, OpKind::ConstRefNull, true)
            .unwrap();
        let type_result = crate::front::exc_from_raise::push_type_of(&mut graph, pred_type, inst);
        let src_pbc = graph.alloc_value_var();
        *src_pbc.annotation.borrow_mut() = Some(Rc::new(nullable));
        branch_start_to_preds(&mut graph, pred_type, pred_pbc);
        graph.set_goto(pred_type, join, vec![type_result]);
        graph.set_goto(pred_pbc, join, vec![src_pbc]);
        graph.set_return(join, Some(phi_args[0].clone()));
        let err = annotate_inner(&graph, Some(&bk))
            .expect_err("SomeTypeOf ∪ nullable class PBC is UnionError");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    fn typeof_flat_ref_phi_graph(type_pred_is_true: bool) -> FunctionGraph {
        // Join of `type(v)` (`SomeTypeOf`) and a classless
        // `Input { ty: Ref(None) }` (`SomeInstance`). `set_branch`
        // walks the false link first.
        let mut graph = FunctionGraph::new("typeof_flat_ref_phi");
        let pred_type = graph.create_block();
        let pred_inst = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let inst = graph
            .push_op_var(pred_type, OpKind::ConstRefNull, true)
            .unwrap();
        let type_result = crate::front::exc_from_raise::push_type_of(&mut graph, pred_type, inst);
        let src_inst = graph
            .push_op_var(
                pred_inst,
                OpKind::Input {
                    name: "obj".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        if type_pred_is_true {
            branch_start_to_preds(&mut graph, pred_type, pred_inst);
        } else {
            branch_start_to_preds(&mut graph, pred_inst, pred_type);
        }
        graph.set_goto(pred_type, join, vec![type_result]);
        graph.set_goto(pred_inst, join, vec![src_inst]);
        graph.set_return(join, Some(phi_args[0].clone()));
        graph
    }

    #[test]
    fn typeof_phi_with_flat_ref_instance_is_union_error() {
        // `binaryop.py` has no `pairtype(SomeTypeOf, SomeInstance)`;
        // the mix is `pairtype(SomeObject, SomeObject).union`.
        for type_pred_is_true in [true, false] {
            let graph = typeof_flat_ref_phi_graph(type_pred_is_true);
            let err = annotate_inner(&graph, None)
                .expect_err("SomeTypeOf ∪ classless Input Ref(None) is UnionError");
            assert!(
                err.to_string().contains("UnionError"),
                "type_pred_is_true={type_pred_is_true} annotate_inner error: {err}"
            );
        }
    }

    #[test]
    fn type_op_rewalk_keeps_typeof() {
        // `annrpython.py` `consider_op` → `setbinding` recomputes the
        // `type()` result. A second walk of the same block must keep
        // `SomeTypeOf` (`unaryop.py` `type_SomeObject`).
        // False link first (`pred_bool`) stores `SomeBool`; the Int
        // predecessor widens the phi and reschedules the body.
        let mut graph = FunctionGraph::new("type_op_rewalk");
        let pred_int = graph.create_block();
        let pred_bool = graph.create_block();
        let (body, phi_args) = graph.create_block_with_arg_vars(1);
        let int_var = graph
            .push_op_var(pred_int, OpKind::ConstInt(1), true)
            .unwrap();
        let bool_var = graph
            .push_op_var(pred_bool, OpKind::ConstBool(true), true)
            .unwrap();
        let inst = graph.push_op_var(body, OpKind::ConstRefNull, true).unwrap();
        let type_result = crate::front::exc_from_raise::push_type_of(&mut graph, body, inst);
        branch_start_to_preds(&mut graph, pred_int, pred_bool);
        graph.set_goto(pred_int, body, vec![int_var]);
        graph.set_goto(pred_bool, body, vec![bool_var]);
        graph.set_return(body, Some(phi_args[0].clone()));
        annotate_inner(&graph, None).expect("type() rewalk keeps SomeTypeOf");
        assert!(
            matches!(
                read_binding_some(&type_result).as_deref(),
                Some(SomeValue::TypeOf(_))
            ),
            "rewalked type() stays SomeTypeOf"
        );
        assert_eq!(read_binding(&phi_args[0]), ValueType::Int);
    }

    #[test]
    fn propagates_float_return_into_returnblock() {
        let mut graph = FunctionGraph::new("float_return");
        let entry = graph.startblock;
        let base_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new("floatval", None),
                    ty: ValueType::Float,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result_var.clone()));

        annotate(&graph);
        let ret_var = graph.block(graph.returnblock).inputargs[0].clone();
        assert_eq!(read_binding(&result_var), ValueType::Float);
        assert_eq!(read_binding(&ret_var), ValueType::Float);
    }

    #[test]
    fn synthetic_void_return_carries_void_concretetype() {
        // `set_return(_, None)` wires `LinkArg::Const(Constant(None,
        // concretetype=Void))` per `flowcontext.py RETURN_VALUE` +
        // `:1232-1236`.  At the annotation layer, `Constant(None)`
        // projects to `Ref` (pyre's nearest mapping for upstream's
        // `SomeNone` — see `const_value_type`).  The rtyping-layer
        // projection at [`super::legacy_resolve::link_arg_concrete_type`]
        // then honours the construction-site `concretetype=Void` to
        // materialise `Void` on the returnblock inputarg, matching
        // `pairtype(Repr, NoneRepr).convert_from_to → inputconst(Void, None)`
        // (`rpython/rtyper/rnone.py`).
        use super::super::legacy_resolve;
        let mut graph = FunctionGraph::new("void_return");
        let entry = graph.startblock;
        graph.set_return(entry, None);

        annotate(&graph);
        let ret_var = graph.block(graph.returnblock).inputargs[0].clone();
        assert_eq!(
            read_binding(&ret_var),
            ValueType::Ref(None),
            "annotation-layer projection of Constant(None) follows SomeNone → Ref",
        );

        legacy_resolve::resolve_types(&graph, &std::collections::HashSet::new());
        assert_eq!(
            FunctionGraph::concretetype_of(&ret_var),
            crate::codewriter::type_state::ConcreteType::Void,
            "rtyping-layer projection honours Constant.concretetype=Void from set_return",
        );
    }

    #[test]
    fn union_bool_unsigned_is_unsigned() {
        // `binaryop.py:189-201`: SomeBool(nonneg=True) ∪ SomeInteger(unsigned)
        // reaches `elif t1 is int` with `int1.nonneg == True`, so no
        // UnionError fires and `knowntype = r_uint` → Unsigned.
        assert_eq!(
            union_type(&ValueType::Bool, &ValueType::Unsigned),
            ValueType::Unsigned
        );
        assert_eq!(
            union_type(&ValueType::Unsigned, &ValueType::Bool),
            ValueType::Unsigned
        );
    }

    #[test]
    fn union_bool_int_is_int() {
        assert_eq!(
            union_type(&ValueType::Bool, &ValueType::Int),
            ValueType::Int
        );
        assert_eq!(
            union_type(&ValueType::Int, &ValueType::Bool),
            ValueType::Int
        );
        assert_eq!(const_value_type(&ConstValue::Bool(true)), ValueType::Bool);
        assert_eq!(
            union_type(&const_value_type(&ConstValue::Bool(false)), &ValueType::Int),
            ValueType::Int
        );
    }

    #[test]
    fn union_int_unsigned_stays_unknown() {
        // `binaryop.py:191`: signed Int (`nonneg=False`) ∪ Unsigned raises
        // UnionError — the coarse enum widens it to Unknown.
        assert_eq!(
            union_type(&ValueType::Int, &ValueType::Unsigned),
            ValueType::Unknown
        );
        assert_eq!(
            union_type(&ValueType::Unsigned, &ValueType::Int),
            ValueType::Unknown
        );
    }

    fn instance_is_nullable(var: &Variable) -> bool {
        matches!(
            read_binding_some(var).as_deref(),
            Some(SomeValue::Instance(inst)) if inst.can_be_none
        )
    }

    /// `annrpython.py mergeinputargs` unions `SomeInstance` shells so a
    /// nullable classed annotation is not rebound through a non-nullable
    /// `setbinding` shell (`binaryop.py pairtype(SomeInstance,
    /// SomeInstance).union` keeps `can_be_None`).
    #[test]
    fn merge_binding_nullable_instance_stays_nullable() {
        use crate::annotator::classdesc::ClassDef;

        let bk = Rc::new(Bookkeeper::new());
        let _host = bk.intern_class_by_qualname("bytecode::CodeObject");
        let classdef = bk
            .classdef_for_known_struct_root("bytecode::CodeObject")
            .unwrap_or_else(|| ClassDef::new_standalone("bytecode::CodeObject", None));
        let mut graph = FunctionGraph::new("nullable_instance_merge");
        let seed =
            |graph: &mut FunctionGraph| {
                let var = graph.alloc_value_var();
                *var.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(
                    SomeInstance::new(Some(classdef.clone()), true, Default::default()),
                )));
                var
            };
        let v_none = seed(&mut graph);
        merge_binding(&v_none, Incoming::Flat(ValueType::Ref(None)), Some(&bk)).unwrap();
        assert!(
            instance_is_nullable(&v_none),
            "Ref(None) join keeps can_be_None"
        );
        let v_root = seed(&mut graph);
        merge_binding(
            &v_root,
            Incoming::Flat(ValueType::Ref(Some("bytecode::CodeObject".into()))),
            Some(&bk),
        )
        .unwrap();
        assert!(
            instance_is_nullable(&v_root),
            "Ref(Some(root)) join keeps can_be_None"
        );
    }

    /// Op-result re-inference keeps flags (`annrpython.py consider_op`
    /// `setbinding`); a same-class flat shell on a link intersects them
    /// (`binaryop.py pairtype(SomeInstance, SomeInstance).union`).
    #[test]
    fn op_result_keeps_flags_link_flat_intersects_them() {
        use crate::annotator::classdesc::ClassDef;

        let bk = Rc::new(Bookkeeper::new());
        let _host = bk.intern_class_by_qualname("bytecode::CodeObject");
        let classdef = bk
            .classdef_for_known_struct_root("bytecode::CodeObject")
            .unwrap_or_else(|| ClassDef::new_standalone("bytecode::CodeObject", None));
        let mut flags = std::collections::BTreeMap::new();
        flags.insert("access_directly".to_string(), true);
        let mut graph = FunctionGraph::new("op_result_keeps_flags");
        let seed =
            |graph: &mut FunctionGraph| {
                let var = graph.alloc_value_var();
                *var.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(
                    SomeInstance::new(Some(classdef.clone()), false, flags.clone()),
                )));
                var
            };
        let name = classdef.borrow().name.clone();

        // (a) op-result re-inference keeps the existing instance annotation.
        let var = seed(&mut graph);
        let before = read_binding_some(&var);
        let changed = merge_op_result(
            &var,
            Incoming::Flat(ValueType::Ref(Some(name.clone()))),
            Some(&bk),
        )
        .unwrap();
        assert!(!changed);
        let after = read_binding_some(&var);
        match (&before, &after) {
            (Some(a), Some(b)) => {
                assert!(Rc::ptr_eq(a, b) || a == b);
            }
            _ => panic!("op-result merge must keep the existing annotation"),
        }

        // (b) same keep with no bookkeeper.
        let var2 = seed(&mut graph);
        let changed = merge_op_result(
            &var2,
            Incoming::Flat(ValueType::Ref(Some(name.clone()))),
            None,
        )
        .unwrap();
        assert!(!changed);
        match read_binding_some(&var2).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef)),
                    "classdef must stay"
                );
                assert_eq!(inst.flags.get("access_directly"), Some(&true));
            }
            other => panic!("expected flagged instance, got {other:?}"),
        }

        // (c) a link-join flat shell intersects flags (keeps a flag only
        // when both sides have it).
        let var3 = seed(&mut graph);
        let changed =
            merge_binding(&var3, Incoming::Flat(ValueType::Ref(Some(name))), Some(&bk)).unwrap();
        assert!(changed);
        match read_binding_some(&var3).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef)),
                    "classdef must stay"
                );
                assert!(
                    inst.flags.is_empty(),
                    "flat shell has no flags, union drops them; got {:?}",
                    inst.flags
                );
            }
            other => panic!("expected instance with empty flags, got {other:?}"),
        }
    }

    #[test]
    fn link_keeps_nullable_source() {
        use crate::annotator::classdesc::ClassDef;

        let bk = Rc::new(Bookkeeper::new());
        let _host = bk.intern_class_by_qualname("bytecode::CodeObject");
        let classdef = bk
            .classdef_for_known_struct_root("bytecode::CodeObject")
            .unwrap_or_else(|| ClassDef::new_standalone("bytecode::CodeObject", None));
        let mut graph = FunctionGraph::new("link_keeps_nullable");
        let entry = graph.startblock;
        let name = classdef.borrow().name.clone();
        let src = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "v".into(),
                    ty: ValueType::Ref(Some(name)),
                    class_root: Some("bytecode::CodeObject".into()),
                },
                true,
            )
            .unwrap();
        *src.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            true,
            Default::default(),
        ))));
        let (target, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();
        graph.set_goto(entry, target, vec![src]);
        graph.set_return(target, Some(phi_var.clone()));
        annotate_with_bookkeeper(&graph, Some(&bk));
        match read_binding_some(&phi_var).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(inst.can_be_none);
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef))
                );
            }
            other => panic!("expected nullable instance of cd, got {other:?}"),
        }
    }

    #[test]
    fn none_const_joins_instance_as_nullable() {
        use crate::annotator::classdesc::ClassDef;
        use crate::flowspace::model::Constant;

        let bk = Rc::new(Bookkeeper::new());
        let _host = bk.intern_class_by_qualname("bytecode::CodeObject");
        let classdef = bk
            .classdef_for_known_struct_root("bytecode::CodeObject")
            .unwrap_or_else(|| ClassDef::new_standalone("bytecode::CodeObject", None));
        let mut graph = FunctionGraph::new("none_const_join");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let inst = graph.alloc_value_var();
        *inst.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ))));
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();
        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto(pred1, join, vec![inst]);
        graph.set_goto_mixed(
            pred2,
            join,
            vec![LinkArg::Const(Constant::new(ConstValue::None))],
        );
        graph.set_return(join, Some(phi_var.clone()));
        annotate_with_bookkeeper(&graph, Some(&bk));
        match read_binding_some(&phi_var).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(inst.can_be_none);
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef))
                );
            }
            other => panic!("expected nullable instance of cd, got {other:?}"),
        }
    }

    /// Empty slot keeps `SomeNone`; a later instance (or flat `Ref`)
    /// joins it in either arrival order (`annrpython.py mergeinputargs`
    /// → `model.unionof`; `binaryop.py pairtype(SomeNone,
    /// SomeInstance).union` → `noneify`).
    #[test]
    fn none_first_then_instance_is_nullable_instance() {
        use crate::annotator::classdesc::ClassDef;

        let bk = Rc::new(Bookkeeper::new());
        let _host = bk.intern_class_by_qualname("bytecode::CodeObject");
        let classdef = bk
            .classdef_for_known_struct_root("bytecode::CodeObject")
            .unwrap_or_else(|| ClassDef::new_standalone("bytecode::CodeObject", None));
        let mut graph = FunctionGraph::new("none_first_then_instance");
        let var = graph.alloc_value_var();
        let none = Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new())));
        merge_binding(&var, none, Some(&bk)).unwrap();
        match read_binding_some(&var).as_deref() {
            Some(SomeValue::None_(_)) => {}
            other => panic!("empty slot keeps SomeNone, got {other:?}"),
        }
        let inst = Incoming::Annotation(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ))));
        merge_binding(&var, inst, Some(&bk)).unwrap();
        match read_binding_some(&var).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(inst.can_be_none);
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef))
                );
            }
            other => panic!("expected nullable instance of cd, got {other:?}"),
        }

        let var2 = graph.alloc_value_var();
        merge_binding(
            &var2,
            Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new()))),
            Some(&bk),
        )
        .unwrap();
        merge_binding(&var2, Incoming::Flat(ValueType::Ref(None)), Some(&bk)).unwrap();
        match read_binding_some(&var2).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(inst.can_be_none);
            }
            other => panic!("expected nullable instance after flat Ref, got {other:?}"),
        }
    }

    /// `binaryop.py pairtype(SomeObject, SomeNone).union` /
    /// `pairtype(SomeNone, SomeObject).union` → `obj.noneify()`.
    /// `SomeString.noneify` sets `can_be_None=True`; `SomeInteger.noneify`
    /// raises (`SomeObject.noneify`).
    #[test]
    fn merge_binding_string_joins_none_via_noneify() {
        use crate::annotator::model::SomeString;

        let mut graph = FunctionGraph::new("string_joins_none");
        let var = graph.alloc_value_var();
        merge_binding(
            &var,
            Incoming::Annotation(Rc::new(SomeValue::String(SomeString::new(false, false)))),
            None,
        )
        .unwrap();
        merge_binding(
            &var,
            Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new()))),
            None,
        )
        .unwrap();
        match read_binding_some(&var).as_deref() {
            Some(SomeValue::String(s)) => assert!(s.can_be_none()),
            other => panic!("expected SomeString(can_be_None), got {other:?}"),
        }

        let var2 = graph.alloc_value_var();
        merge_binding(
            &var2,
            Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new()))),
            None,
        )
        .unwrap();
        merge_binding(
            &var2,
            Incoming::Annotation(Rc::new(SomeValue::String(SomeString::new(false, false)))),
            None,
        )
        .unwrap();
        match read_binding_some(&var2).as_deref() {
            Some(SomeValue::String(s)) => assert!(s.can_be_none()),
            other => panic!("expected SomeString(can_be_None) reverse, got {other:?}"),
        }
    }

    #[test]
    fn merge_binding_instance_then_none_is_nullable_instance() {
        use crate::annotator::classdesc::ClassDef;

        let bk = Rc::new(Bookkeeper::new());
        let _host = bk.intern_class_by_qualname("bytecode::CodeObject");
        let classdef = bk
            .classdef_for_known_struct_root("bytecode::CodeObject")
            .unwrap_or_else(|| ClassDef::new_standalone("bytecode::CodeObject", None));
        let mut graph = FunctionGraph::new("instance_then_none");
        let var = graph.alloc_value_var();
        merge_binding(
            &var,
            Incoming::Annotation(Rc::new(SomeValue::Instance(SomeInstance::new(
                Some(classdef.clone()),
                false,
                Default::default(),
            )))),
            Some(&bk),
        )
        .unwrap();
        merge_binding(
            &var,
            Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new()))),
            Some(&bk),
        )
        .unwrap();
        match read_binding_some(&var).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(inst.can_be_none);
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef))
                );
            }
            other => panic!("expected nullable instance of cd, got {other:?}"),
        }
    }

    #[test]
    fn merge_binding_int_then_none_is_union_error() {
        use crate::annotator::model::SomeInteger;

        let mut graph = FunctionGraph::new("int_then_none");
        let var = graph.alloc_value_var();
        merge_binding(
            &var,
            Incoming::Annotation(Rc::new(SomeValue::Integer(SomeInteger::new(false, false)))),
            None,
        )
        .unwrap();
        let err = merge_binding(
            &var,
            Incoming::Annotation(Rc::new(SomeValue::None_(SomeNone::new()))),
            None,
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("UnionError"),
            "SomeInteger.noneify raises UnionError, got {text}"
        );
    }

    /// `result_exc.rs materialize_call_arg_constant` stamps
    /// `ConstNone` + `GcRef` for Option-payload `None` (`bookkeeper.py
    /// immutablevalue(None)` → `s_None`). `emit_unit` /
    /// `AggregateShape::Void` keep Void (`rnone.py` `NoneRepr`).
    #[test]
    fn const_none_gcref_annotates_somenone_unit_stays_void() {
        let mut graph = FunctionGraph::new("const_none_payload");
        let entry = graph.startblock;
        let payload = graph.push_op_var(entry, OpKind::ConstNone, true).unwrap();
        FunctionGraph::set_concretetype_of_inline(&payload, crate::model::ConcreteType::GcRef);
        graph.set_return(entry, Some(payload.clone()));
        annotate(&graph);
        match read_binding_some(&payload).as_deref() {
            Some(SomeValue::None_(_)) => {}
            other => panic!("GcRef ConstNone annotates SomeNone, got {other:?}"),
        }

        let mut graph2 = FunctionGraph::new("const_none_unit");
        let entry2 = graph2.startblock;
        let unit = graph2.push_const_none(entry2);
        graph2.set_return(entry2, Some(unit.clone()));
        annotate(&graph2);
        assert_eq!(
            read_binding(&unit),
            ValueType::Void,
            "unit ConstNone stays Void (NoneRepr)"
        );
    }

    #[test]
    fn typed_null_raw_phi_keeps_raw_class() {
        use std::collections::HashSet;

        let bk = Rc::new(Bookkeeper::new());
        bk.set_raw_alloc_flavor_structs(HashSet::from(["X".to_string()]));
        let classdef = bk
            .getuniqueclassdef_for_struct_root("X")
            .expect("raw-flavor classdef");
        let mut graph = FunctionGraph::new("typed_null_raw_phi");
        let pred1 = graph.create_block();
        let pred2 = graph.create_block();
        let x_var = graph
            .push_op_var(
                pred1,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Ref(Some("X".into())),
                    class_root: Some("X".into()),
                },
                true,
            )
            .unwrap();
        let null_var = graph.push_null_raw_ptr_with_spelling(pred2, Some("X"));
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi_var = phi_args[0].clone();
        branch_start_to_preds(&mut graph, pred1, pred2);
        graph.set_goto(pred1, join, vec![x_var]);
        graph.set_goto(pred2, join, vec![null_var]);
        graph.set_return(join, Some(phi_var.clone()));
        annotate_with_bookkeeper(&graph, Some(&bk));
        match read_binding_some(&phi_var).as_deref() {
            Some(SomeValue::Instance(inst)) => {
                assert!(
                    inst.classdef
                        .as_ref()
                        .is_some_and(|cd| Rc::ptr_eq(cd, &classdef))
                );
                assert!(inst.can_be_none);
            }
            other => panic!("expected nullable instance of X, got {other:?}"),
        }
    }

    /// Typed `null_fn` (`push_null_fn_ptr_with_spelling`) annotates as
    /// constant `SomePtr` (`ann_nullptr` `immutablevalue`).
    #[test]
    fn typed_null_fn_annotates_as_constant_someptr() {
        let bk = Rc::new(Bookkeeper::new());
        for spelling in [None, Some("fn() -> i64")] {
            let mut graph = FunctionGraph::new("typed_null_fn");
            let entry = graph.startblock;
            let null_var = graph.push_null_fn_ptr_with_spelling(entry, spelling);
            graph.set_return(entry, Some(null_var.clone()));
            annotate_with_bookkeeper(&graph, Some(&bk));
            match read_binding_some(&null_var).as_deref() {
                Some(sv @ SomeValue::Ptr(_)) => {
                    use crate::annotator::model::SomeObjectTrait;
                    assert!(
                        sv.is_constant(),
                        "spelling {spelling:?} must be immutablevalue, got {sv:?}"
                    );
                }
                other => panic!("spelling {spelling:?}: expected constant SomePtr, got {other:?}"),
            }
        }
    }

    /// A join of constant `null_fn` (`ann_nullptr` `immutablevalue`) with a
    /// non-constant same-type `SomePtr` drops the constant
    /// (`llannotation.py pairtype(SomePtr, SomePtr).union`,
    /// `annrpython.py mergeinputargs` / `model.py unionof`). Two identical
    /// constants stay constant.
    #[test]
    fn typed_null_fn_join_drops_constant_unless_both_sides_match() {
        use crate::annotator::model::SomeObjectTrait;

        let bk = Rc::new(Bookkeeper::new());
        let spelling = "fn() -> i64";
        let s_fn =
            crate::annotator::bookkeeper::fn_ptr_somevalue_for_spelling_in(spelling, Some(&bk))
                .expect("fn spelling");
        assert!(
            matches!(&s_fn, SomeValue::Ptr(_)) && !s_fn.is_constant(),
            "Some of Option<fn> is a non-constant SomePtr, got {s_fn:?}"
        );

        // Constant arrives first, then a non-constant of the same
        // `ll_ptrtype`. Without the join fix the same-type arm keeps the
        // first arrival unchanged, so this stays a constant null.
        let mixed = {
            let mut graph = FunctionGraph::new("typed_null_fn_mixed_join");
            let pred_const = graph.create_block();
            let pred_live = graph.create_block();
            let null_var = graph.push_null_fn_ptr_with_spelling(pred_const, Some(spelling));
            let live_var = graph.alloc_value_var();
            *live_var.annotation.borrow_mut() = Some(Rc::new(s_fn.clone()));
            let (join, phi_args) = graph.create_block_with_arg_vars(1);
            let phi_var = phi_args[0].clone();
            branch_start_to_preds(&mut graph, pred_const, pred_live);
            graph.set_goto(pred_const, join, vec![null_var]);
            graph.set_goto(pred_live, join, vec![live_var]);
            graph.set_return(join, Some(phi_var.clone()));
            annotate_with_bookkeeper(&graph, Some(&bk));
            phi_var
        };
        match read_binding_some(&mixed).as_deref() {
            Some(sv @ SomeValue::Ptr(_)) => {
                assert!(
                    !sv.is_constant(),
                    "const null_fn ∪ live same-type ptr must drop the constant, got {sv:?}"
                );
            }
            other => panic!("expected non-constant SomePtr join, got {other:?}"),
        }

        let identical = {
            let mut graph = FunctionGraph::new("typed_null_fn_identical_join");
            let pred1 = graph.create_block();
            let pred2 = graph.create_block();
            let null1 = graph.push_null_fn_ptr_with_spelling(pred1, Some(spelling));
            let null2 = graph.push_null_fn_ptr_with_spelling(pred2, Some(spelling));
            let (join, phi_args) = graph.create_block_with_arg_vars(1);
            let phi_var = phi_args[0].clone();
            branch_start_to_preds(&mut graph, pred1, pred2);
            graph.set_goto(pred1, join, vec![null1]);
            graph.set_goto(pred2, join, vec![null2]);
            graph.set_return(join, Some(phi_var.clone()));
            annotate_with_bookkeeper(&graph, Some(&bk));
            phi_var
        };
        match read_binding_some(&identical).as_deref() {
            Some(sv @ SomeValue::Ptr(_)) => {
                assert!(
                    sv.is_constant(),
                    "join of two identical null_fn constants must stay constant, got {sv:?}"
                );
            }
            other => panic!("expected constant SomePtr join, got {other:?}"),
        }
    }

    fn builtin_code_fn_spelling() -> &'static str {
        "fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>"
    }

    fn startup_fn_spelling() -> &'static str {
        "fn(PyObjectRef, *const PyExecutionContext) -> Result<(), PyError>"
    }

    fn someptr_ll(sv: &SomeValue) -> &crate::translator::rtyper::lltypesystem::lltype::Ptr {
        match sv {
            SomeValue::Ptr(p) => &p.ll_ptrtype,
            other => panic!("expected SomePtr, got {other:?}"),
        }
    }

    fn fn_ptr_func_type(
        sv: &SomeValue,
    ) -> &crate::translator::rtyper::lltypesystem::lltype::FuncType {
        use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
        match &someptr_ll(sv).TO {
            PtrTarget::Func(ft) => ft,
            other => panic!("expected Ptr(FuncType), got {other:?}"),
        }
    }

    /// Typed `null_fn` ∪ FieldRead whose dest is `Ptr(FuncType)` (`stamp_fn_ptr`).
    #[test]
    fn typed_null_fn_phi_fn_fieldread_is_someptr() {
        use crate::annotator::bookkeeper::{Bookkeeper, fn_ptr_somevalue_for_spelling_in};
        let spelling = builtin_code_fn_spelling();
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("typed_null_fn_phi_fn_fieldread");
        let pred_null = graph.create_block();
        let pred_field = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let null_var = graph.push_null_fn_ptr_with_spelling(pred_null, Some(spelling));
        let base = graph
            .push_op_var(
                pred_field,
                OpKind::Input {
                    name: "code".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let field_var = graph
            .push_op_var(
                pred_field,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("func", Some("BuiltinCode".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let sv = fn_ptr_somevalue_for_spelling_in(spelling, Some(bk.as_ref()))
            .expect("fn spelling is SomePtr");
        stamp_someptr_lltype(&field_var, &sv);
        branch_start_to_preds(&mut graph, pred_null, pred_field);
        graph.set_goto(pred_null, join, vec![null_var.clone()]);
        graph.set_goto(pred_field, join, vec![field_var.clone()]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, Some(&bk)).expect("typed null_fn ∪ fn FieldRead");
        let phi_sv = read_binding_some(&phi).expect("phi annotated");
        let null_sv = read_binding_some(&null_var).expect("null_fn annotated");
        let field_sv = read_binding_some(&field_var).expect("FieldRead annotated");
        assert!(
            somevalue_is_fn_ptr(&phi_sv),
            "phi is SomePtr(FuncType), got {:?}",
            phi_sv
        );
        assert_eq!(
            someptr_ll(&null_sv),
            someptr_ll(&field_sv),
            "null_fn and FieldRead share ll_ptrtype (`pairtype(SomePtr, SomePtr).union`)"
        );
        assert_eq!(someptr_ll(&phi_sv), someptr_ll(&null_sv));
    }

    /// Typed `null_fn` ∪ Call whose dest is `Ptr(FuncType)` (`stamp_fn_ptr`).
    #[test]
    fn typed_null_fn_phi_fn_call_is_someptr() {
        use crate::annotator::bookkeeper::{Bookkeeper, fn_ptr_somevalue_for_spelling_in};
        let spelling = builtin_code_fn_spelling();
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("typed_null_fn_phi_fn_call");
        let pred_null = graph.create_block();
        let pred_call = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let null_var = graph.push_null_fn_ptr_with_spelling(pred_null, Some(spelling));
        let call_var = graph
            .push_op_var(
                pred_call,
                OpKind::Call {
                    target: CallTarget::function_path(["gateway", "builtin_code_get"]),
                    args: vec![],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let sv = fn_ptr_somevalue_for_spelling_in(spelling, Some(bk.as_ref()))
            .expect("fn spelling is SomePtr");
        stamp_someptr_lltype(&call_var, &sv);
        branch_start_to_preds(&mut graph, pred_null, pred_call);
        graph.set_goto(pred_null, join, vec![null_var.clone()]);
        graph.set_goto(pred_call, join, vec![call_var.clone()]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, Some(&bk)).expect("typed null_fn ∪ fn Call");
        let phi_sv = read_binding_some(&phi).expect("phi annotated");
        let null_sv = read_binding_some(&null_var).expect("null_fn annotated");
        let call_sv = read_binding_some(&call_var).expect("Call annotated");
        assert!(
            somevalue_is_fn_ptr(&phi_sv),
            "phi is SomePtr(FuncType), got {:?}",
            phi_sv
        );
        assert_eq!(
            someptr_ll(&null_sv),
            someptr_ll(&call_sv),
            "null_fn and stamped Call share ll_ptrtype"
        );
        assert_eq!(someptr_ll(&phi_sv), someptr_ll(&null_sv));
    }

    /// Typed `null_fn` ∪ `ConstInt(-1)` stays `UnionError`
    /// (`pairtype(SomePtr, SomeObject).union`).
    #[test]
    fn typed_null_fn_phi_constint_neg1_is_union_error() {
        use crate::annotator::bookkeeper::Bookkeeper;
        let spelling = builtin_code_fn_spelling();
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("typed_null_fn_phi_constint");
        let pred_null = graph.create_block();
        let pred_int = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let null_var = graph.push_null_fn_ptr_with_spelling(pred_null, Some(spelling));
        let int_var = graph
            .push_op_var(pred_int, OpKind::ConstInt(-1), true)
            .unwrap();
        branch_start_to_preds(&mut graph, pred_null, pred_int);
        graph.set_goto(pred_null, join, vec![null_var]);
        graph.set_goto(pred_int, join, vec![int_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, Some(&bk)).expect_err("typed null_fn ∪ ConstInt(-1)");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    /// `Result<(), PyError>` with the exception carrier is `Void` on both
    /// arms (`fn_ptr_result_lltype` / `ann_nullptr`).
    #[test]
    fn typed_null_fn_phi_startup_fn_fieldread_is_void_result() {
        use crate::annotator::bookkeeper::{Bookkeeper, fn_ptr_somevalue_for_spelling_in};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let spelling = startup_fn_spelling();
        let bk = Rc::new(Bookkeeper::new());
        bk.set_exception_carrier("error::PyErrorObject");
        bk.alias_exception_carrier_handle("error::PyError");
        let mut graph = FunctionGraph::new("typed_null_fn_phi_startup");
        let pred_null = graph.create_block();
        let pred_field = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let null_var = graph.push_null_fn_ptr_with_spelling(pred_null, Some(spelling));
        let base = graph
            .push_op_var(
                pred_field,
                OpKind::Input {
                    name: "def".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let field_var = graph
            .push_op_var(
                pred_field,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("startup", Some("BuiltinModuleDef".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let sv = fn_ptr_somevalue_for_spelling_in(spelling, Some(bk.as_ref()))
            .expect("fn spelling is SomePtr");
        stamp_someptr_lltype(&field_var, &sv);
        branch_start_to_preds(&mut graph, pred_null, pred_field);
        graph.set_goto(pred_null, join, vec![null_var.clone()]);
        graph.set_goto(pred_field, join, vec![field_var.clone()]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, Some(&bk)).expect("typed null_fn ∪ startup FieldRead");
        let phi_sv = read_binding_some(&phi).expect("phi annotated");
        let null_sv = read_binding_some(&null_var).expect("null_fn annotated");
        let field_sv = read_binding_some(&field_var).expect("FieldRead annotated");
        assert_eq!(
            fn_ptr_func_type(&null_sv).result,
            LowLevelType::Void,
            "carrier Result<(), E> is Void on null_fn"
        );
        assert_eq!(
            fn_ptr_func_type(&field_sv).result,
            LowLevelType::Void,
            "carrier Result<(), E> is Void on FieldRead"
        );
        assert_eq!(someptr_ll(&null_sv), someptr_ll(&field_sv));
        assert!(somevalue_is_fn_ptr(&phi_sv));
    }

    #[test]
    fn back_edge_chain_longer_than_sweep_cap_reaches_fixpoint() {
        // `complete_pending_blocks` drains until the pending set is
        // empty. Storage order walks startblock then later-created
        // blocks, so a 25-hop chain against that order needs more than
        // the old 20-sweep cap for the back edge to annotate the first
        // block.
        let mut graph = FunctionGraph::new("long_back_edge");
        let (header, header_args) = graph.create_block_with_arg_vars(1);
        let header_phi = header_args[0].clone();
        let mut chain: Vec<(crate::model::BlockId, crate::flowspace::model::Variable)> = Vec::new();
        for _ in 0..25 {
            let (id, args) = graph.create_block_with_arg_vars(1);
            chain.push((id, args[0].clone()));
        }
        let seed = graph
            .push_op_var(graph.startblock, OpKind::ConstInt(42), true)
            .unwrap();
        graph.set_goto(graph.startblock, chain[24].0, vec![seed]);
        for i in (1..25).rev() {
            graph.set_goto(chain[i].0, chain[i - 1].0, vec![chain[i].1.clone()]);
        }
        graph.set_goto(chain[0].0, header, vec![chain[0].1.clone()]);
        graph.set_return(header, Some(header_phi.clone()));

        annotate(&graph);
        assert_eq!(
            read_binding(&header_phi),
            ValueType::Int,
            "back-edge into the first chain block must reach fixpoint"
        );
    }

    #[test]
    fn unscheduled_block_receives_no_binding() {
        let mut graph = FunctionGraph::new("orphan_block");
        let entry = graph.startblock;
        let v_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        graph.set_return(entry, Some(v_var.clone()));

        let orphan = graph.create_block();
        let orphan_var = graph
            .push_op_var(orphan, OpKind::ConstInt(99), true)
            .unwrap();

        annotate(&graph);
        assert_eq!(read_binding(&v_var), ValueType::Int);
        assert_eq!(
            read_binding(&orphan_var),
            ValueType::Unknown,
            "a block never scheduled from the startblock stays unannotated"
        );
        assert!(
            orphan_var.annotation.borrow().is_none(),
            "an unreached slot stays empty (bottom)"
        );
    }

    #[test]
    fn int_float_phi_is_union_error() {
        // Direct slot: Int then Float. Empty is bottom; `union_type` of
        // two bound concretes is the `_` arm, and `merge_flat` reports
        // `binaryop.py` `pairtype(SomeObject, SomeObject).union` as
        // `UnionError`.
        let mut scratch = FunctionGraph::new("scratch");
        let slot = scratch.alloc_value_var();
        assert!(
            slot.annotation.borrow().is_none(),
            "unreached slot is bottom"
        );
        assert!(merge_flat(&slot, ValueType::Int, None).unwrap());
        let err = merge_flat(&slot, ValueType::Float, None).expect_err("Int ∪ Float is UnionError");
        assert!(
            err.to_string().contains("UnionError"),
            "merge_flat error: {err}"
        );
        assert_eq!(read_binding(&slot), ValueType::Int);

        let bound = scratch.alloc_value_var();
        assert!(merge_flat(&bound, ValueType::Int, None).unwrap());
        assert_eq!(
            merge_flat(&bound, ValueType::Unknown, None).unwrap(),
            false,
            "incoming Unknown (unbound source) into a bound slot is a no-op"
        );
        assert_eq!(read_binding(&bound), ValueType::Int);

        // Worklist: diamond Int/Float into one phi. `complete_pending_blocks`
        // drains until `merge_flat` raises UnionError.
        let mut graph = FunctionGraph::new("int_float_phi");
        let pred_int = graph.create_block();
        let pred_float = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let int_var = graph
            .push_op_var(pred_int, OpKind::ConstInt(1), true)
            .unwrap();
        let float_var = graph
            .push_op_var(pred_float, OpKind::ConstFloat(1.0f64.to_bits()), true)
            .unwrap();
        branch_start_to_preds(&mut graph, pred_int, pred_float);
        graph.set_goto(pred_int, join, vec![int_var]);
        graph.set_goto(pred_float, join, vec![float_var]);
        graph.set_return(join, Some(phi));

        let err = annotate_inner(&graph, None).expect_err("Int ∪ Float phi is UnionError");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn unsigned_and_str_same_shell_union_ok_signedness_mix_is_union_error() {
        // `binaryop.py` `pairtype(SomeInteger, SomeInteger).union` of two
        // `unsigned=True` integers succeeds. `read_binding` must report
        // `Unsigned` so a second incoming is not a false `Int ∪ Unsigned`.
        let mut scratch = FunctionGraph::new("scratch_unsigned_str");
        let unsigned_slot = scratch.alloc_value_var();
        assert!(merge_flat(&unsigned_slot, ValueType::Unsigned, None).unwrap());
        assert_eq!(
            merge_flat(&unsigned_slot, ValueType::Unsigned, None).unwrap(),
            false,
            "Unsigned ∪ Unsigned is identity"
        );
        assert_eq!(read_binding(&unsigned_slot), ValueType::Unsigned);

        // `binaryop.py` `pairtype(SomeString, SomeString).union` succeeds.
        let str_slot = scratch.alloc_value_var();
        assert!(merge_flat(&str_slot, ValueType::Str, None).unwrap());
        assert_eq!(
            merge_flat(&str_slot, ValueType::Str, None).unwrap(),
            false,
            "Str ∪ Str is identity"
        );
        assert_eq!(read_binding(&str_slot), ValueType::Str);

        // Signed `SomeInteger` (`nonneg=False`) ∪ unsigned is
        // `binaryop.py` UnionError (`pairtype(SomeInteger, SomeInteger).union`).
        let mix = scratch.alloc_value_var();
        assert!(merge_flat(&mix, ValueType::Int, None).unwrap());
        let err =
            merge_flat(&mix, ValueType::Unsigned, None).expect_err("Int ∪ Unsigned is UnionError");
        assert!(
            err.to_string().contains("UnionError"),
            "merge_flat error: {err}"
        );
        assert_eq!(read_binding(&mix), ValueType::Int);
    }

    #[test]
    fn const_int_nonneg_phi_unsigned_is_unsigned() {
        // `bookkeeper.py immutablevalue(int)` → `SomeInteger(nonneg = x >= 0)`.
        // `binaryop.py` `pairtype(SomeInteger, SomeInteger).union` of
        // nonneg signed ∪ unsigned wins `knowntype=r_uint`.
        let mut graph = FunctionGraph::new("const_int_phi_unsigned");
        let pred_const = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let const_var = graph
            .push_op_var(pred_const, OpKind::ConstInt(2), true)
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_const, pred_unsigned);
        graph.set_goto(pred_const, join, vec![const_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("ConstInt(2) ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn arraylen_phi_unsigned_is_unsigned() {
        // `unaryop.py` `SomeObject.len` → `SomeInteger(nonneg=True)`.
        let mut graph = FunctionGraph::new("arraylen_phi_unsigned");
        let pred_len = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let base = graph
            .push_op_var(
                pred_len,
                OpKind::Input {
                    name: "arr".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let len_var = graph
            .push_op_var(
                pred_len,
                OpKind::ArrayLen {
                    base,
                    array_type_id: None,
                    nolength: false,
                    gckind: crate::model::ArrayGcKind::Gc,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_len, pred_unsigned);
        graph.set_goto(pred_len, join, vec![len_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("ArrayLen ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn same_as_nonneg_int_phi_unsigned_is_unsigned() {
        // `unaryop.py` `same_as` keeps the operand; `bind_array_view_len_in_block`
        // plants `same_as` with `result_ty: Int`.
        let mut graph = FunctionGraph::new("same_as_phi_unsigned");
        let pred_len = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let const_var = graph
            .push_op_var(pred_len, OpKind::ConstInt(2), true)
            .unwrap();
        let same_var = graph
            .push_op_var(
                pred_len,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: const_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_len, pred_unsigned);
        graph.set_goto(pred_len, join, vec![same_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("same_as ConstInt(2) ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn string_byte_getitem_phi_unsigned_is_unsigned() {
        // Spine A `strgetitem` is Call `__string_byte_getitem`
        // (`unaryop.py` `SomeChar.ord` → `SomeInteger(nonneg=True)`).
        let mut graph = FunctionGraph::new("strgetitem_phi_unsigned");
        let pred_get = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let get_var = graph
            .push_op_var(
                pred_get,
                OpKind::Call {
                    target: CallTarget::function_path(["__string_byte_getitem"]),
                    args: vec![],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_get, pred_unsigned);
        graph.set_goto(pred_get, join, vec![get_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("__string_byte_getitem ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn const_int_negative_phi_const_uint_is_union_error() {
        // `binaryop.py` `pairtype(SomeInteger, SomeInteger).union`:
        // signed `nonneg=False` ∪ unsigned is UnionError.
        let mut graph = FunctionGraph::new("const_int_neg_phi_uint");
        let pred_signed = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let signed_var = graph
            .push_op_var(pred_signed, OpKind::ConstInt(-1), true)
            .unwrap();
        let unsigned_var = graph
            .push_op_var(pred_unsigned, OpKind::ConstUInt(0), true)
            .unwrap();
        branch_start_to_preds(&mut graph, pred_signed, pred_unsigned);
        graph.set_goto(pred_signed, join, vec![signed_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, None).expect_err("ConstInt(-1) ∪ ConstUInt(0)");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn link_const_int_zero_into_unsigned_slot_ok() {
        // `bookkeeper.py immutablevalue(int)` of `0` is nonneg; union
        // with Unsigned wins unsigned (`pairtype(SomeInteger, SomeInteger).union`).
        use crate::flowspace::model::Constant;
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("link_const_int_zero");
        let pred_unsigned = graph.create_block();
        let pred_const = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_unsigned, pred_const);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_goto_mixed(
            pred_const,
            join,
            vec![LinkArg::Const(Constant::new(ConstValue::Int(0)))],
        );
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, Some(&bk)).expect("link ConstValue::Int(0) ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn const_str_phi_with_str_call_is_somestring() {
        // `rstr.py` `StringRepr.convert_const`: `ConstStr` is `Ptr(STR)`.
        // `infer_op_type(Call { result_ty: Str })` copies `Str`.
        // `binaryop.py` `pairtype(SomeString, SomeString).union`.
        let mut graph = FunctionGraph::new("const_str_phi");
        let pred_const = graph.create_block();
        let pred_call = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let const_var = graph
            .push_op_var(pred_const, OpKind::ConstStr(b"object".to_vec()), true)
            .unwrap();
        let call_var = graph
            .push_op_var(
                pred_call,
                OpKind::Call {
                    target: CallTarget::function_path(["w_type_get_name"]),
                    args: vec![],
                    result_ty: ValueType::Str,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_const, pred_call);
        graph.set_goto(pred_const, join, vec![const_var]);
        graph.set_goto(pred_call, join, vec![call_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("ConstStr ∪ Call Str");
        assert!(
            matches!(
                read_binding_some(&phi).as_deref(),
                Some(SomeValue::String(_))
            ),
            "phi is SomeString, got {:?}",
            read_binding_some(&phi)
        );
        assert_eq!(read_binding(&phi), ValueType::Str);
    }

    #[test]
    fn str_const_call_ref_none_joins_str_slot() {
        // Frontend `DecodedConst::Str` is `Call(["__str_const", text],
        // result_ty: Ref(None))` (`is_str_const_define`). Upstream that
        // value is a bare Constant; `bookkeeper.py immutablevalue`
        // `issubclass(tp, str)` → `SomeString`.
        let mut graph = FunctionGraph::new("str_const_call_phi");
        let pred_const = graph.create_block();
        let pred_str = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let const_var = graph
            .push_op_var(
                pred_const,
                OpKind::Call {
                    target: CallTarget::function_path(["__str_const", "eq"]),
                    args: vec![],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let str_var = graph
            .push_op_var(
                pred_str,
                OpKind::Input {
                    name: "op".into(),
                    ty: ValueType::Str,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_const, pred_str);
        graph.set_goto(pred_const, join, vec![const_var]);
        graph.set_goto(pred_str, join, vec![str_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("__str_const Ref(None) ∪ Str");
        assert!(
            matches!(
                read_binding_some(&phi).as_deref(),
                Some(SomeValue::String(_))
            ),
            "phi is SomeString, got {:?}",
            read_binding_some(&phi)
        );
        assert_eq!(read_binding(&phi), ValueType::Str);
    }

    #[test]
    fn bytestr_link_const_joins_str_slot() {
        // `bookkeeper.py immutablevalue` `issubclass(tp, str)` →
        // `SomeString` / `SomeChar`.
        use crate::flowspace::model::Constant;
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("bytestr_link");
        let pred_str = graph.create_block();
        let pred_const = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let str_var = graph
            .push_op_var(
                pred_str,
                OpKind::Input {
                    name: "op".into(),
                    ty: ValueType::Str,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_str, pred_const);
        graph.set_goto(pred_str, join, vec![str_var]);
        graph.set_goto_mixed(
            pred_const,
            join,
            vec![LinkArg::Const(Constant::new(ConstValue::byte_str("eq")))],
        );
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, Some(&bk)).expect("ByteStr ∪ Str");
        assert!(
            matches!(
                read_binding_some(&phi).as_deref(),
                Some(SomeValue::String(_) | SomeValue::Char(_))
            ),
            "slot is SomeString/SomeChar, got {:?}",
            read_binding_some(&phi)
        );
        assert_eq!(read_binding(&phi), ValueType::Str);
    }

    #[test]
    fn str_family_ref_unions_with_str() {
        // `declared_ref_valuetype` rewrites str-family `Ref(Some(root))`
        // to `Str` (`is_str_family_class_root`; `rstr.py` one string type).
        let mut scratch = FunctionGraph::new("scratch_str_family_ref");
        let slot = scratch.alloc_value_var();
        assert!(merge_flat(&slot, ValueType::Str, None).unwrap());
        assert_eq!(
            merge_flat(&slot, ValueType::Ref(Some("string::String".into())), None).unwrap(),
            false,
            "str-family Ref ∪ Str is identity"
        );
        assert_eq!(read_binding(&slot), ValueType::Str);
        assert!(
            matches!(
                read_binding_some(&slot).as_deref(),
                Some(SomeValue::String(_))
            ),
            "slot stays SomeString, got {:?}",
            read_binding_some(&slot)
        );
    }

    #[test]
    fn object_ref_union_str_is_union_error() {
        // `binaryop.py` `pairtype(SomeObject, SomeObject).union`.
        let mut scratch = FunctionGraph::new("scratch_pyobject_str");
        let mix = scratch.alloc_value_var();
        assert!(merge_flat(&mix, ValueType::Str, None).unwrap());
        let err = merge_flat(
            &mix,
            ValueType::Ref(Some("pyobject::PyObject".into())),
            None,
        )
        .expect_err("PyObject Ref ∪ Str is UnionError");
        assert!(
            err.to_string().contains("UnionError"),
            "merge_flat error: {err}"
        );
        assert_eq!(read_binding(&mix), ValueType::Str);
    }

    #[test]
    fn str_family_malloc_phi_is_str() {
        // `rstr.py` `mallocstr` → `malloc(STR, n)` → `Ptr(STR)` /
        // `SomeString`. `try_alloc_bytes_block` Managed GC New ∪ immortal
        // STR RawMalloc.
        let mut graph = FunctionGraph::new("try_alloc_bytes_block_shape");
        let pred_gc = graph.create_block();
        let pred_immortal = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let gc_var = graph
            .push_op_var(
                pred_gc,
                OpKind::New {
                    owner: "unicodeobject::Utf8Str".into(),
                },
                true,
            )
            .unwrap();
        let immortal_var = graph
            .push_op_var(
                pred_immortal,
                OpKind::raw_malloc_with_flags(
                    "unicodeobject::Utf8Str",
                    None,
                    false,
                    true,
                    false,
                    vec![],
                    None,
                ),
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_gc, pred_immortal);
        graph.set_goto(pred_gc, join, vec![gc_var]);
        graph.set_goto(pred_immortal, join, vec![immortal_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("STR New ∪ STR RawMalloc");
        assert_eq!(read_binding(&phi), ValueType::Str);
        assert!(
            matches!(
                read_binding_some(&phi).as_deref(),
                Some(SomeValue::String(_))
            ),
            "phi is SomeString, got {:?}",
            read_binding_some(&phi)
        );
    }

    #[test]
    fn object_new_phi_str_is_union_error() {
        // A real `PyObject` New ∪ Str is still UnionError
        // (`binaryop.py` `pairtype(SomeObject, SomeObject).union`).
        let mut graph = FunctionGraph::new("pyobject_new_phi_str");
        let pred_obj = graph.create_block();
        let pred_str = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let obj_var = graph
            .push_op_var(
                pred_obj,
                OpKind::New {
                    owner: "pyobject::PyObject".into(),
                },
                true,
            )
            .unwrap();
        let str_var = graph
            .push_op_var(pred_str, OpKind::ConstStr(b"x".to_vec()), true)
            .unwrap();
        branch_start_to_preds(&mut graph, pred_obj, pred_str);
        graph.set_goto(pred_obj, join, vec![obj_var]);
        graph.set_goto(pred_str, join, vec![str_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, None).expect_err("PyObject New ∪ Str is UnionError");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn fatlen_fieldread_phi_unsigned_is_unsigned() {
        // `unaryop.py` `SomeObject.len` / `SomeString.len` →
        // `SomeInteger(nonneg=True)`. Slice `FatLen` is that count
        // (`rlist.py` `ll_length`).
        let mut graph = FunctionGraph::new("fatlen_phi_unsigned");
        let pred_len = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let base = graph
            .push_op_var(
                pred_len,
                OpKind::Input {
                    name: "window".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let len_var = graph
            .push_op_var(
                pred_len,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("as_bytes", None)
                        .with_vec_part(VecFieldPart::FatLen),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_len, pred_unsigned);
        graph.set_goto(pred_len, join, vec![len_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("FieldRead FatLen ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn size_fieldread_phi_unsigned_is_union_error() {
        // A signed field that is not a length stays `nonneg=False`
        // (`binaryop.py` `pairtype(SomeInteger, SomeInteger).union`).
        let mut graph = FunctionGraph::new("size_phi_unsigned");
        let pred_size = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let base = graph
            .push_op_var(
                pred_size,
                OpKind::Input {
                    name: "sub".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let size_var = graph
            .push_op_var(
                pred_size,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("size", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_size, pred_unsigned);
        graph.set_goto(pred_size, join, vec![size_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, None).expect_err("FieldRead size ∪ Unsigned");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn fatmeta_fieldread_is_signed_not_nonneg() {
        // `FatMeta` is the `&dyn Trait` metadata word (`rtuple.py`
        // `TupleRepr.rtype_getitem` of `two_word_fat_struct`), not
        // `unaryop.py` `SomeObject.len`. Flat `Int` is
        // `SomeInteger()` (`nonneg=False`).
        let mut graph = FunctionGraph::new("fatmeta_signed");
        let entry = graph.startblock;
        let base = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "slot".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let meta_var = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("slot", None).with_vec_part(VecFieldPart::FatMeta),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(meta_var.clone()));
        annotate_inner(&graph, None).expect("FieldRead FatMeta");
        match read_binding_some(&meta_var).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(!i.nonneg, "FatMeta is not a length, got {i:?}");
                assert!(!i.unsigned, "FatMeta stays signed, got {i:?}");
            }
            other => panic!("FatMeta is signed SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn fatmeta_fieldread_phi_unsigned_is_union_error() {
        // `FatMeta` keeps the element's signed `SomeInteger`
        // (`rtuple.py` `TupleRepr.rtype_getitem`). Joining with
        // unsigned is `binaryop.py`
        // `pairtype(SomeInteger, SomeInteger).union`.
        let mut graph = FunctionGraph::new("fatmeta_phi_unsigned");
        let pred_meta = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let base = graph
            .push_op_var(
                pred_meta,
                OpKind::Input {
                    name: "slot".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let meta_var = graph
            .push_op_var(
                pred_meta,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("slot", None).with_vec_part(VecFieldPart::FatMeta),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_meta, pred_unsigned);
        graph.set_goto(pred_meta, join, vec![meta_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, None).expect_err("FieldRead FatMeta ∪ Unsigned");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn binop_add_const_and_len_phi_unsigned_is_unsigned() {
        // `binaryop.py` `add = _clone(union, [])` (`integer_union`).
        // `ConstInt(1)` is nonneg (`bookkeeper.py immutablevalue`);
        // FatLen is `SomeObject.len`. Their union stays nonneg.
        let mut graph = FunctionGraph::new("binop_add_phi_unsigned");
        let pred_add = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let base = graph
            .push_op_var(
                pred_add,
                OpKind::Input {
                    name: "window".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let len_var = graph
            .push_op_var(
                pred_add,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("as_bytes", None)
                        .with_vec_part(VecFieldPart::FatLen),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let one = graph
            .push_op_var(pred_add, OpKind::ConstInt(1), true)
            .unwrap();
        let add_var = graph
            .push_op_var(
                pred_add,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: one,
                    rhs: len_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_add, pred_unsigned);
        graph.set_goto(pred_add, join, vec![add_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("BinOp add(ConstInt(1), FatLen) ∪ Unsigned");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn binop_sub_len_and_const_phi_unsigned_is_union_error() {
        // `binaryop.py` `sub` (`integer_sub`) returns
        // `SomeInteger(knowntype=restype)` with nonneg False.
        let mut graph = FunctionGraph::new("binop_sub_phi_unsigned");
        let pred_sub = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let base = graph
            .push_op_var(
                pred_sub,
                OpKind::Input {
                    name: "window".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let len_var = graph
            .push_op_var(
                pred_sub,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("as_bytes", None)
                        .with_vec_part(VecFieldPart::FatLen),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let one = graph
            .push_op_var(pred_sub, OpKind::ConstInt(1), true)
            .unwrap();
        let sub_var = graph
            .push_op_var(
                pred_sub,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len_var,
                    rhs: one,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_sub, pred_unsigned);
        graph.set_goto(pred_sub, join, vec![sub_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi));
        let err =
            annotate_inner(&graph, None).expect_err("BinOp sub(FatLen, ConstInt(1)) ∪ Unsigned");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn binop_add_two_equal_constants_is_not_constant() {
        // `binaryop.py` `_clone(union, [])`: add/mul/xor allocate a
        // fresh `SomeInteger` and never copy `const`.
        let bk = Rc::new(Bookkeeper::new());
        for op in ["add", "mul", "xor"] {
            let mut graph = FunctionGraph::new(format!("binop_{op}_two_const"));
            let entry = graph.startblock;
            let lhs = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
            let rhs = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
            let result = graph
                .push_op_var(
                    entry,
                    OpKind::BinOp {
                        op: op.into(),
                        lhs,
                        rhs,
                        result_ty: ValueType::Int,
                    },
                    true,
                )
                .unwrap();
            graph.set_return(entry, Some(result.clone()));
            annotate_inner(&graph, Some(&bk)).unwrap_or_else(|e| panic!("{op}(2, 2): {e}"));
            match read_binding_some(&result).as_deref() {
                Some(SomeValue::Integer(i)) => {
                    assert!(
                        !i.is_constant(),
                        "{op}(2, 2) must drop operand const, got {i:?}"
                    );
                    assert!(i.nonneg, "{op}(2, 2) is nonneg, got {i:?}");
                }
                other => panic!("{op}(2, 2) is SomeInteger, got {other:?}"),
            }
        }
    }

    #[test]
    fn binop_add_two_bools_is_integer() {
        // `pairtype(SomeInteger, SomeInteger)` add is `_clone(union)`;
        // `rarithmetic.py` `compute_restype(bool, bool)` is `int`.
        use crate::annotator::model::KnownType;
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("binop_add_two_bools");
        let entry = graph.startblock;
        let lhs = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let rhs = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result.clone()));
        annotate_inner(&graph, Some(&bk)).expect("bool + bool");
        match read_binding_some(&result).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert_eq!(i.base.knowntype, KnownType::Int);
                assert!(i.nonneg, "bool+bool is nonneg, got {i:?}");
            }
            other => panic!("bool + bool is SomeInteger(Int), got {other:?}"),
        }
    }

    #[test]
    fn binop_and_two_bools_is_bool() {
        // `pairtype(SomeBool, SomeBool).and_` returns `SomeBool`.
        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("binop_and_two_bools");
        let entry = graph.startblock;
        let lhs = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let rhs = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "and".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result.clone()));
        annotate_inner(&graph, Some(&bk)).expect("bool and bool");
        match read_binding_some(&result).as_deref() {
            Some(SomeValue::Bool(_)) => {}
            other => panic!("bool and bool is SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn binop_xor_two_true_constants_is_false() {
        // `pairtype(SomeBool, SomeBool).xor` folds `const ^ const`.
        // `ConstBool` ops do not go through `immutablevalue`, so the
        // operands are seeded the way `bookkeeper.py immutablevalue`
        // stamps `const_box` on a bool constant.
        let bk = Rc::new(Bookkeeper::new());
        let s_true = bk
            .immutablevalue(&ConstValue::Bool(true))
            .expect("immutablevalue of true");
        let mut graph = FunctionGraph::new("binop_xor_two_true");
        let entry = graph.startblock;
        let lhs = graph.alloc_value_var();
        let rhs = graph.alloc_value_var();
        *lhs.annotation.borrow_mut() = Some(Rc::new(s_true.clone()));
        *rhs.annotation.borrow_mut() = Some(Rc::new(s_true));
        let result = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "xor".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result.clone()));
        annotate_inner(&graph, Some(&bk)).expect("true ^ true");
        match read_binding_some(&result).as_deref() {
            Some(sv @ SomeValue::Bool(_)) => {
                assert!(sv.is_constant(), "true ^ true is constant, got {sv:?}");
                assert_eq!(sv.const_(), Some(&ConstValue::Bool(false)));
            }
            other => panic!("true ^ true is SomeBool const false, got {other:?}"),
        }
    }

    #[test]
    fn same_as_unsigned_of_integer_sub_phi_constint1_is_unsigned() {
        // `rarithmetic.py` `r_uint(x)` / `ForTypeEntry.compute_result_annotation`
        // → `SomeInteger(knowntype=r_uint)`. `parse_format` `rep as usize`
        // after `integer_sub` (`c - b'0'`) phis the else-arm `1`.
        let mut graph = FunctionGraph::new("same_as_uint_of_sub");
        let pred_cast = graph.create_block();
        let pred_one = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let c = graph
            .push_op_var(pred_cast, OpKind::ConstInt(i64::from(b'5')), true)
            .unwrap();
        let zero = graph
            .push_op_var(pred_cast, OpKind::ConstInt(i64::from(b'0')), true)
            .unwrap();
        let sub_var = graph
            .push_op_var(
                pred_cast,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: c,
                    rhs: zero,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let same_var = graph
            .push_op_var(
                pred_cast,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: sub_var,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let one = graph
            .push_op_var(pred_one, OpKind::ConstInt(1), true)
            .unwrap();
        branch_start_to_preds(&mut graph, pred_cast, pred_one);
        graph.set_goto(pred_cast, join, vec![same_var]);
        graph.set_goto(pred_one, join, vec![one]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("same_as Unsigned of integer_sub ∪ ConstInt(1)");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned, "unsigned won, got {i:?}");
            }
            other => panic!("phi is SomeInteger unsigned, got {other:?}"),
        }
        assert_eq!(read_binding(&phi), ValueType::Unsigned);
    }

    #[test]
    fn same_as_int_of_signed_nonneg_keeps_nonneg() {
        // Same signedness copies (`unaryop.py` identity);
        // `bind_array_view_len_in_block` plants `same_as` with
        // `result_ty: Int`.
        let mut graph = FunctionGraph::new("same_as_int_keeps_nonneg");
        let entry = graph.startblock;
        let const_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let same_var = graph
            .push_op_var(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: const_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(same_var.clone()));
        annotate_inner(&graph, None).expect("same_as Int of signed nonneg");
        match read_binding_some(&same_var).as_deref() {
            Some(SomeValue::Integer(i)) => {
                assert!(i.nonneg, "nonneg preserved, got {i:?}");
                assert!(!i.unsigned, "signedness unchanged, got {i:?}");
            }
            other => panic!("same_as is signed nonneg SomeInteger, got {other:?}"),
        }
        assert_eq!(read_binding(&same_var), ValueType::Int);
    }

    #[test]
    fn same_as_int_of_unsigned_phi_unsigned_is_union_error() {
        // `builtin.py` `rarith_intmask` → `SomeInteger()` (signed,
        // nonneg False). `binaryop.py` `pairtype(SomeInteger,
        // SomeInteger).union` rejects signed nonneg=False ∪ unsigned.
        let mut graph = FunctionGraph::new("same_as_int_of_uint");
        let pred_cast = graph.create_block();
        let pred_unsigned = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let unsigned_src = graph
            .push_op_var(
                pred_cast,
                OpKind::Input {
                    name: "u".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let same_var = graph
            .push_op_var(
                pred_cast,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: unsigned_src,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let unsigned_var = graph
            .push_op_var(
                pred_unsigned,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_cast, pred_unsigned);
        graph.set_goto(pred_cast, join, vec![same_var]);
        graph.set_goto(pred_unsigned, join, vec![unsigned_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, None).expect_err("same_as Int of Unsigned ∪ Unsigned");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    fn stamp_ccharp(var: &Variable) {
        use crate::translator::rtyper::lltypesystem::rffi::CCHARP;
        var.set_concretetype(Some((*CCHARP).clone()));
    }

    fn unsigned_ccharp_fieldread(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        field: &str,
    ) -> Variable {
        let base = graph
            .push_op_var(
                block,
                OpKind::Input {
                    name: "obj".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let ptr = graph
            .push_op_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new(field, None),
                    ty: ValueType::Unsigned,
                    pure: false,
                },
                true,
            )
            .unwrap();
        stamp_ccharp(&ptr);
        ptr
    }

    #[test]
    fn raw_malloc_phi_u8_fieldread_is_someptr() {
        // `ann_malloc` / `raw_alloc_ptr_somevalue` → `SomePtr(CCHARP)`.
        // `tuple_field_value_type("*mut u8")` stamps the same `CCHARP`.
        // `llannotation.py` `pairtype(SomePtr, SomePtr).union`.
        let mut graph = FunctionGraph::new("raw_malloc_phi_u8");
        let pred_malloc = graph.create_block();
        let pred_field = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let malloc_var = graph
            .push_op_var(
                pred_malloc,
                OpKind::raw_malloc_with_flags("alloc::raw", None, false, true, false, vec![], None),
                true,
            )
            .unwrap();
        let field_var = unsigned_ccharp_fieldread(&mut graph, pred_field, "cpy_ref");
        branch_start_to_preds(&mut graph, pred_malloc, pred_field);
        graph.set_goto(pred_malloc, join, vec![malloc_var]);
        graph.set_goto(pred_field, join, vec![field_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("RawMalloc ∪ *mut u8 FieldRead");
        assert!(
            matches!(read_binding_some(&phi).as_deref(), Some(SomeValue::Ptr(_))),
            "phi is SomePtr, got {:?}",
            read_binding_some(&phi)
        );
    }

    #[test]
    fn ccharp_nullptr_phi_u8_fieldread_is_someptr() {
        // `ann_cast_int_to_ptr` → `SomePtr(CCHARP)` (`emit_ccharp_nullptr`
        // Call `lltype.cast_int_to_ptr`).
        use crate::translator::rtyper::lltypesystem::rffi::CCHARP;
        let mut graph = FunctionGraph::new("ccharp_nullptr_phi_u8");
        let pred_null = graph.create_block();
        let pred_field = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let null_var = graph
            .push_op_var(
                pred_null,
                OpKind::Call {
                    target: CallTarget::function_path([
                        "rpython",
                        "rtyper",
                        "lltypesystem",
                        "lltype",
                        "cast_int_to_ptr",
                    ]),
                    args: vec![
                        LinkArg::from(ConstValue::LowLevelType(Box::new((*CCHARP).clone()))),
                        LinkArg::from(ConstValue::Int(0)),
                    ],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let field_var = unsigned_ccharp_fieldread(&mut graph, pred_field, "cpy_ref");
        branch_start_to_preds(&mut graph, pred_null, pred_field);
        graph.set_goto(pred_null, join, vec![null_var]);
        graph.set_goto(pred_field, join, vec![field_var]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("cast_int_to_ptr ∪ *mut u8 FieldRead");
        assert!(
            matches!(read_binding_some(&phi).as_deref(), Some(SomeValue::Ptr(_))),
            "phi is SomePtr, got {:?}",
            read_binding_some(&phi)
        );
    }

    #[test]
    fn constint_neg1_phi_u8_fieldread_is_union_error() {
        // `SomeInteger` ∪ `SomePtr` (`pairtype(SomeObject, SomeObject).union`).
        let mut graph = FunctionGraph::new("constint_neg1_phi_u8");
        let pred_int = graph.create_block();
        let pred_field = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let int_var = graph
            .push_op_var(pred_int, OpKind::ConstInt(-1), true)
            .unwrap();
        let field_var = unsigned_ccharp_fieldread(&mut graph, pred_field, "cpy_ref");
        branch_start_to_preds(&mut graph, pred_int, pred_field);
        graph.set_goto(pred_int, join, vec![int_var]);
        graph.set_goto(pred_field, join, vec![field_var]);
        graph.set_return(join, Some(phi));
        let err = annotate_inner(&graph, None).expect_err("ConstInt(-1) ∪ *mut u8 FieldRead");
        assert!(
            err.to_string().contains("UnionError"),
            "annotate_inner error: {err}"
        );
    }

    #[test]
    fn raw_malloc_typed_gcarray_is_someptr_of_ptr_type() {
        // `ann_malloc` → `SomePtr(Ptr(s_T.const))`. A phi of two mallocs
        // of the same TYPE is `pairtype(SomePtr, SomePtr).union`.
        use crate::translator::rtyper::lltypesystem::lltype::{Array, LowLevelType, Ptr};
        let array_ll = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let expected = Ptr::from_container_type(array_ll.clone()).expect("GcArray ptr");
        let mut graph = FunctionGraph::new("raw_malloc_typed_gcarray");
        let entry = graph.startblock;
        let malloc_var = graph
            .push_op_var(
                entry,
                OpKind::raw_malloc_with_flags(
                    "alloc::raw",
                    Some(array_ll.clone()),
                    false,
                    true,
                    false,
                    vec![],
                    None,
                ),
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(malloc_var.clone()));
        annotate_inner(&graph, None).expect("typed RawMalloc");
        match read_binding_some(&malloc_var).as_deref() {
            Some(SomeValue::Ptr(p)) => {
                assert_eq!(p.ll_ptrtype, expected, "ll_ptrtype is Ptr(GcArray)");
            }
            other => panic!("typed RawMalloc is SomePtr, got {other:?}"),
        }

        let mut graph = FunctionGraph::new("raw_malloc_typed_phi");
        let pred_a = graph.create_block();
        let pred_b = graph.create_block();
        let (join, phi_args) = graph.create_block_with_arg_vars(1);
        let phi = phi_args[0].clone();
        let a = graph
            .push_op_var(
                pred_a,
                OpKind::raw_malloc_with_flags(
                    "alloc::raw",
                    Some(array_ll.clone()),
                    false,
                    true,
                    false,
                    vec![],
                    None,
                ),
                true,
            )
            .unwrap();
        let b = graph
            .push_op_var(
                pred_b,
                OpKind::raw_malloc_with_flags(
                    "alloc::raw",
                    Some(array_ll),
                    false,
                    true,
                    false,
                    vec![],
                    None,
                ),
                true,
            )
            .unwrap();
        branch_start_to_preds(&mut graph, pred_a, pred_b);
        graph.set_goto(pred_a, join, vec![a]);
        graph.set_goto(pred_b, join, vec![b]);
        graph.set_return(join, Some(phi.clone()));
        annotate_inner(&graph, None).expect("phi of two typed RawMalloc");
        match read_binding_some(&phi).as_deref() {
            Some(SomeValue::Ptr(p)) => {
                assert_eq!(p.ll_ptrtype, expected, "phi ll_ptrtype is Ptr(GcArray)");
            }
            other => panic!("phi is SomePtr, got {other:?}"),
        }
    }
}
