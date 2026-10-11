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
//!   `Variable.annotation` cell via [`setbinding`].  A
//!   follow-up retirement slice drops both consumers and this file
//!   together once the dual-gate Skip categories close.
//!
//! Propagates ValueType annotations through the graph by analyzing
//! each op's inputs and computing the output type. Iterates to
//! fixpoint when Block.inputargs (Phi nodes) need widening.

use std::rc::Rc;

use crate::annotator::model::SomeValue;
use crate::codewriter::annotation_state::{somevalue_to_valuetype, valuetype_to_someshell};
use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{FunctionGraph, Link, LinkArg, OpKind, ValueType};

/// Run annotation propagation to fixpoint, writing each binding
/// directly into the orthodox `Variable.annotation` cell
/// (`flowspace/model.py:Variable.annotation`).
///
/// RPython equivalent: `RPythonAnnotator.complete()` — processes all
/// blocks until no annotation changes.  All bindings land on each
/// `Variable.annotation` cell; callers that want a flat
/// `ValueType` discriminator read via `read_binding` (tests) or
/// `somevalue_to_valuetype(&var.annotation.borrow())`
/// (production).
pub fn annotate(graph: &FunctionGraph) {
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

    // Process all blocks (simple single-pass for acyclic; loops need fixpoint)
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 20;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        for block in &graph.blocks {
            // Propagate annotations through ops in this block
            for op in &block.operations {
                if let Some(result) = op.result.as_ref() {
                    let inferred = infer_op_type(&op.kind);
                    let current = read_binding(result);
                    let merged = union_type(&current, &inferred);
                    if merged != current {
                        setbinding(result, merged);
                        changed = true;
                    }
                }
            }

            // Cross-block propagation: Link args → target inputargs, per
            // upstream `rpython/annotator/annrpython.py` fold pass which
            // iterates `for link in block.exits` and unions each
            // `link.args[i]` annotation into `link.target.inputargs[i]`.
            for link in &block.exits {
                let link_changed = if link_is_raise_like(link) {
                    follow_raise_link(graph, link)
                } else {
                    follow_link(graph, link)
                };
                changed |= link_changed;
            }
        }
    }
}

/// Read the legacy `ValueType` discriminator out of
/// `var.annotation`.  Returns `Unknown` when the slot is empty —
/// consistent with [`setbinding`]'s invariant (every non-`Unknown`
/// setbinding writes a paired shell; `Unknown` clears it).
fn read_binding(var: &Variable) -> ValueType {
    var.annotation
        .borrow()
        .as_ref()
        .map(|s| somevalue_to_valuetype(s))
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
pub(crate) fn setbinding(var: &Variable, ty: ValueType) {
    if let Some(shell) = valuetype_to_someshell(&ty) {
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

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

fn follow_link(graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        changed |= merge_value_type(dst, link_arg_type(src));
    }
    changed
}

fn follow_raise_link(graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    if let Some(value) = link.last_exc_value.as_ref().and_then(|a| a.as_variable()) {
        changed |= merge_value_type(value, ValueType::Ref(None));
    }
    if let Some(value) = link.last_exception.as_ref().and_then(|a| a.as_variable()) {
        changed |= merge_value_type(value, ValueType::Int);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ValueType::Int
        } else if Some(src) == link.last_exc_value.as_ref() {
            ValueType::Ref(None)
        } else {
            link_arg_type(src)
        };
        changed |= merge_value_type(dst, src_ty);
    }
    changed
}

fn link_arg_type(src: &LinkArg) -> ValueType {
    match src {
        LinkArg::Value(var) => read_binding(var),
        LinkArg::Const(value) => const_value_type(&value.value),
    }
}

fn merge_value_type(dst: &Variable, src_ty: ValueType) -> bool {
    let current = read_binding(dst);
    let merged = union_type(&current, &src_ty);
    if merged != current {
        setbinding(dst, merged);
        true
    } else {
        false
    }
}

fn const_value_type(value: &ConstValue) -> ValueType {
    match value {
        ConstValue::Int(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::AddressOffset(_)
        | ConstValue::InheritanceId { .. }
        | ConstValue::LLAddress(_) => ValueType::Int,
        ConstValue::Int128(_) => ValueType::Int128,
        ConstValue::UInt128(_) => ValueType::UInt128,
        ConstValue::Float(_) => ValueType::Float,
        ConstValue::Placeholder => ValueType::Unknown,
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
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_)
        | ConstValue::Opaque(_)
        | ConstValue::Repr(_) => ValueType::Ref(None),
    }
}

/// Infer the output type of an operation from its inputs.
///
/// RPython equivalent: annotator dispatch (e.g., `annotate_int_add`
/// returns `SomeInteger()`).
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
        // Singleton instance pointer (unit-variant PBC).  The legacy
        // annotator sees this as a plain ref-typed value; concrete
        // class identity stays in the `HostObject` carrier itself.
        OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
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
        OpKind::RawLoad { item_ty, .. } => item_ty.clone(),
        OpKind::RawStore { .. } => ValueType::Void,
        OpKind::FieldRead { ty, .. } => ty.clone(),
        OpKind::FieldWrite { .. } => ValueType::Void,
        OpKind::New { owner } => ValueType::Ref(Some(owner.clone())),
        OpKind::NewWithVtable { owner, .. } => ValueType::Ref(Some(owner.clone())),
        // Fixed-size `lltype.malloc(T, flavor='raw')` returns a Signed
        // address. `lltype.free(p, flavor='raw')` returns nothing.
        OpKind::RawMalloc { .. } => ValueType::Int,
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
        OpKind::Call {
            result_ty,
            target,
            args,
            ..
        } => {
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
        // Float) → Unknown → GcRef backfill.
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
        | OpKind::Live
        | OpKind::GuardValue { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::RecordKnownResult { .. }
        // jtransform.py — `record_quasiimmut_field` has no result.
        | OpKind::RecordQuasiImmutField { .. }
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
    _args: &[crate::model::LinkArg],
) -> ValueType {
    if crate::call::is_int_arithmetic_target(target) {
        return ValueType::Int;
    }
    ValueType::Unknown
}

/// Merge two annotations (RPython `unionof()`).
///
/// Returns the wider type. Unknown absorbs everything (top of lattice).
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
        CallTarget, ExitSwitch, FunctionGraph, Link, OpKind, ValueType, exception_exitcase,
    };

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
        assert_eq!(read_binding(&last_exception_var), ValueType::Int);
        assert_eq!(read_binding(&last_exc_value_var), ValueType::Ref(None));
        assert_eq!(read_binding(&etype_var), ValueType::Int);
        assert_eq!(read_binding(&evalue_var), ValueType::Ref(None));
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

        legacy_resolve::resolve_types(&graph);
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
}
