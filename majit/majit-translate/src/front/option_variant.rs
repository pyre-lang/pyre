//! Option / two-variant enum construction and diamond threading.
//!
//! Static [`emit_option_variant`] / [`emit_sum_variant`] mint a tagged enum
//! shell (variant ctor + `__discriminant` + optional `__pos_0`). Dynamic
//! [`emit_option_variant_dynamic`] / [`emit_sum_variant_dynamic`] branch on a
//! live discriminant and join at the existing continuation.
//! [`map_source`] / [`reproduce_exit_args`] / [`close_goto_mixed`] thread live
//! values through those arms. [`emit_narrow`] re-applies a pointee-class
//! narrowing a residual call carried into a diamond arm. [`slice_len_op`]
//! is the length read a slice bounds guard branches on.

use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
    ValueType,
};

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
                let mapped = map_source(sources, inputs, v)
                    .ok_or_else(|| format!("{name}: exit arg not threaded into diamond arm"))?;
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
/// Dynamic construction needs a plain goto whose arguments can be threaded
/// through both new arms.
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
            gckind: crate::model::ArrayGcKind::Gc,
        };
    } else {
        "__len"
    };
    OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: vec![marker.to_string()],
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        },
        args: crate::model::call_args(vec![base]),
        result_ty: ValueType::Int,
    }
}

/// Re-emit the `__cast_instance_intrinsic` pointee-class narrowing in `block` on
/// `value`, returning the narrowed value — the same cast `result_narrow_root`
/// appends after a `*mut <registered ADT>` call result.  `value` unchanged when
/// no narrowing was absorbed.
pub(crate) fn emit_narrow(
    graph: &mut FunctionGraph,
    block: BlockId,
    value: Variable,
    narrow_root: &Option<String>,
) -> Variable {
    let Some(root) = narrow_root else {
        return value;
    };
    let narrowed = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(narrowed.clone()),
        kind: crate::model::cast_instance_call(root.clone(), value),
    });
    narrowed
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
