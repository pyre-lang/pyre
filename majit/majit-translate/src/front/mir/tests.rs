use super::{harvest_declared_gc_facts, tyref_to_attr_value_type_with};

/// Classify a struct field [`TyRef`] into the RPython `lltype` register
/// class the annotator pre-fills into `FORCE_ATTRIBUTES_INTO_CLASSES`.
///
/// Like [`tyref_to_value_type`], this keeps the signed/unsigned split:
/// `{"Scalar": {"Integer": {"Unsigned": …}}}` shells to [`ValueType::Unsigned`] so
/// `valuetype_to_someshell`
/// picks `SomeInteger { unsigned: true }`, matching the per-field shells
/// the syn classifier produced for `u8`..`usize`.  `char` and every
/// signed width fold to `Int`; `bool`/`float` keep their classes; a
/// payload-carrying enum paints `Ref(Some(root))` so the FORCE-attr intern
/// keys the same classdef `Input.class_root` does; every other non-scalar
/// shape (named struct, reference, raw pointer, tuple, slice, array,
/// `Box`/`Rc`/`Arc` wrapper) folds to `Ref(None)` whose someshell ignores
/// the payload. A `repr(transparent)` scalar wrapper keeps its inner register
/// class, as it does at ordinary value sites.
pub(in crate::front::mir) fn tyref_to_attr_value_type(ty: &TyRef, llbc: &Llbc) -> ValueType {
    let gc_struct_ids = harvest_declared_gc_facts(llbc).gc_struct_ids();
    let raw_alloc_flavor_structs =
        crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(llbc));
    tyref_to_attr_value_type_with(
        ty,
        llbc,
        no_tombstoned_leaves(),
        &gc_struct_ids,
        &raw_alloc_flavor_structs,
    )
}

mod reachable_mask_tests {
    use super::super::{forward_reachable_mask, lowered_reachable_mask, successor_blocks};
    use majit_charon_reader::{Llbc, ullbc::Unstructured};
    use serde_json::{Value, json};

    fn span() -> Value {
        json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    }

    fn ty() -> Value {
        json!({"Deduplicated": 0})
    }

    fn place(i: u64) -> Value {
        json!({"kind": {"Local": i}, "ty": ty()})
    }

    fn bb(kind: Value, is_cleanup: bool) -> Value {
        let unwind = if is_cleanup { "Cleanup" } else { "Regular" };
        json!({
            "statements": [],
            "terminator": {"span": span(), "kind": kind},
            "kind": unwind
        })
    }

    fn body_of(blocks: Vec<Value>) -> (Llbc, Unstructured) {
        let llbc = Llbc::from_slice(
            br#"{"charon_version":"t","has_errors":false,"translated":{"crate_name":"c","fun_decls":[]}}"#,
        )
        .expect("empty artefact");
        let raw = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": [{"index": 0, "name": null, "span": span(), "ty": ty()}]
            },
            "body": blocks
        });
        let body: Unstructured = serde_json::from_value(raw).expect("fixture body");
        (llbc, body)
    }

    /// A one-arm `Ref` SwitchInt is niche-shaped; the panic default is a
    /// successor the lowered mask keeps.
    #[test]
    fn lowered_mask_keeps_niche_option_panic_default() {
        let switch = json!({"Switch": {
            "discr": {"Copy": place(0)},
            "targets": {"SwitchInt": [
                null,
                [[{"Ref": [{"Body": 18}, {"DynTrait": {"id": 0}}, "Mut"]}, 1]],
                2
            ]}
        }});
        let (llbc, body) = body_of(vec![
            bb(switch, false),
            bb(json!("Return"), false),
            bb(json!({"Abort": "UndefinedBehavior"}), false),
        ]);
        let lowered = lowered_reachable_mask(&llbc, &body);
        assert_eq!(
            lowered,
            vec![true, true, true],
            "niche Option panic default is an emitted If arm"
        );
        let forward = forward_reachable_mask(&llbc, &body);
        assert_eq!(forward, vec![true, true, true]);
    }

    /// Arms {0, 1} plus a panic default sit outside the niche shape, so
    /// the lowered mask drops that default.
    #[test]
    fn lowered_mask_drops_generic_switchint_panic_default() {
        let switch = json!({"Switch": {
            "discr": {"Copy": place(0)},
            "targets": {"SwitchInt": [
                null,
                [
                    [{"Scalar": {"Unsigned": ["U8", "0"]}}, 1],
                    [{"Scalar": {"Unsigned": ["U8", "1"]}}, 1]
                ],
                2
            ]}
        }});
        let (llbc, body) = body_of(vec![
            bb(switch, false),
            bb(json!("Return"), false),
            bb(json!({"Abort": "UndefinedBehavior"}), false),
        ]);
        let lowered = lowered_reachable_mask(&llbc, &body);
        assert_eq!(
            lowered,
            vec![true, true, false],
            "SwitchInt arms {{0, 1}} leave the panic default off the lowered mask"
        );
    }

    /// `strip_cleanup_blocks` left bodies unchanged when a normal successor
    /// targets a cleanup block (`kept_block_normal_successor_is_cleanup`).
    #[test]
    fn successor_blocks_keeps_a_normal_edge_into_cleanup() {
        let (llbc, body) = body_of(vec![
            bb(json!({"Goto": {"target": 1}}), false),
            bb(json!("UnwindResume"), true),
        ]);
        let succs = successor_blocks(&llbc, &body, 0, false);
        assert_eq!(succs, vec![1]);
    }
}

/// The unsubstituted generic `entries_item_ptr` has TypeVars and no
/// Charon layout. Translation stops rather than guess the items
/// offset (`array_field_items_offset`).
#[test]
fn unsubstituted_entries_item_ptr_stops_without_guessing_offset() {
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let err = super::lower_function(&llbc, "entries_item_ptr")
        .expect_err("generic entries_item_ptr has no items offset");
    assert!(err.to_string().contains("direct_arrayitems"), "{err}");
}

/// A one-word-item Vec header pointer is int-banked, matching
/// `descr.py get_type_flag` of a raw Ptr and `rrustvec.rs rust_vec_lltype`.
#[test]
fn rust_vec_header_pointer_tuple_field_is_int_banked() {
    use crate::model::ValueType;
    for spelling in ["*mut Vec<usize>", "*mut Vec<*mut u8>", "*mut Vec<f64>"] {
        assert_eq!(
            super::tuple_field_value_type(spelling),
            ValueType::Int,
            "{spelling}"
        );
    }
}

/// `history.py` `getkind` of a raw `Ptr` to a non-byte primitive is
/// `'int'`. A byte pointer stays `Ref` (GC erasure to `*mut u8`).
#[test]
fn raw_scalar_pointer_tuple_field_is_int_banked() {
    use crate::model::ValueType;
    for spelling in ["*mut usize", "*const usize", "*mut i64", "*mut f64"] {
        assert_eq!(
            super::tuple_field_value_type(spelling),
            ValueType::Int,
            "{spelling}"
        );
    }
    for spelling in ["*mut u8", "*const u8", "*mut *mut usize"] {
        assert_eq!(
            super::tuple_field_value_type(spelling),
            ValueType::Ref(None),
            "{spelling}"
        );
    }
}

use super::{no_tombstoned_leaves, scalar_replace_named_struct_aggregates};

use crate::flowspace::model::Variable;

use crate::model::{
    CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

use majit_charon_reader::{Llbc, ullbc::TyRef};

pub(in crate::front::mir) fn fixture_llbc() -> Llbc {
    Llbc::from_slice(
            br#"{"charon_version":"t","has_errors":false,"translated":{"crate_name":"c","fun_decls":[]}}"#,
        )
        .expect("empty artefact")
}

pub(in crate::front::mir) fn empty_llbc() -> Llbc {
    Llbc::from_slice(
            br#"{"charon_version":"t","has_errors":false,"translated":{"crate_name":"c","fun_decls":[],"files":[]}}"#,
        )
        .expect("empty llbc")
}

pub(in crate::front::mir) fn count_branch_calls(graph: &FunctionGraph) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "branch"
            )
        })
        .count()
}

pub(in crate::front::mir) fn struct_ctor_graph(pass_to_malloc: bool) -> FunctionGraph {
    struct_ctor_graph_escaping(pass_to_malloc, StructCtorEscape::UniqueLocal)
}

#[derive(Clone, Copy)]
enum StructCtorEscape {
    /// Field reads only; the aggregate itself does not escape.
    UniqueLocal,
    /// `return s` — a by-value move to the caller.
    Returned,
    /// Passed to a residual call that expects the by-value layout.
    CallArg,
    /// Copied into a successor phi, then mutated.
    Phi,
}

fn struct_ctor_graph_escaping(pass_to_malloc: bool, escape: StructCtorEscape) -> FunctionGraph {
    let mut graph = FunctionGraph::new("struct_ctor");
    let entry = graph.startblock;
    let owner = "error::DictKeyError";
    let result = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["error".to_string()],
                    "DictKeyError",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.to_string())),
            },
            true,
        )
        .expect("struct ctor");
    let payload = graph
        .push_op_var(entry, OpKind::ConstInt(1), true)
        .expect("payload");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: result.clone(),
            field: FieldDescriptor::new("kind", Some(owner.to_string())),
            value: LinkArg::Value(payload.clone()),
            ty: ValueType::Int,
        },
        false,
    );
    if pass_to_malloc {
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["runtime_object", "lltype", "malloc_typed"]),
                args: crate::model::call_args(vec![result.clone()]),
                result_ty: ValueType::Ref(Some(owner.to_string())),
            },
            true,
        );
    }
    match escape {
        StructCtorEscape::UniqueLocal => {
            let kind = graph
                .push_op_var(
                    entry,
                    OpKind::FieldRead {
                        base: result,
                        field: FieldDescriptor::new("kind", Some(owner.to_string())),
                        ty: ValueType::Int,
                        pure: true,
                    },
                    true,
                )
                .expect("field read");
            graph.set_return(entry, Some(kind));
        }
        StructCtorEscape::Returned => graph.set_return(entry, Some(result)),
        StructCtorEscape::CallArg => {
            graph.push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["slice", "index"]),
                    args: crate::model::call_args(vec![result]),
                    result_ty: ValueType::Int,
                },
                true,
            );
            graph.set_return(entry, None);
        }
        StructCtorEscape::Phi => {
            let (next, args) = graph.create_block_with_arg_vars(1);
            let copy = args[0].clone();
            graph.push_op_var(
                next,
                OpKind::FieldWrite {
                    base: copy,
                    field: FieldDescriptor::new("kind", Some(owner.to_string())),
                    value: LinkArg::Value(payload),
                    ty: ValueType::Int,
                },
                false,
            );
            graph.set_goto(entry, next, vec![result]);
            graph.set_return(next, None);
        }
    }
    graph
}

pub(in crate::front::mir) fn struct_ctor_ops(graph: &FunctionGraph) -> (usize, usize, usize) {
    let mut ctors = 0usize;
    let mut news = 0usize;
    let mut field_writes = 0usize;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        is_struct: true, ..
                    },
                ..
            } => ctors += 1,
            OpKind::New { owner } if owner == "error::DictKeyError" => news += 1,
            OpKind::FieldWrite { field, .. } if field.name == "kind" => field_writes += 1,
            _ => {}
        }
    }
    (ctors, news, field_writes)
}

pub(in crate::front::mir) fn empty_struct_attrs()
-> std::collections::HashMap<String, Vec<(String, ValueType)>> {
    std::collections::HashMap::new()
}

pub(in crate::front::mir) fn replace_struct_ctors(graph: &mut FunctionGraph) -> usize {
    scalar_replace_named_struct_aggregates(graph, &empty_struct_attrs())
}

pub(in crate::front::mir) fn same_as_count(graph: &FunctionGraph) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as"))
        .count()
}

pub(in crate::front::mir) fn struct_news(graph: &FunctionGraph) -> Vec<Variable> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::New { owner } if owner == "error::DictKeyError" => op.result.clone(),
            _ => None,
        })
        .collect()
}

/// A unique, unescaped named struct becomes its fields: the constructor
/// and its stores disappear, and the field read is the stored SSA value.
/// Transparent newtype wrappers are a different arm and stay a no-op alias.
#[test]
fn named_struct_aggregate_becomes_its_fields() {
    let mut graph = struct_ctor_graph(false);
    assert_eq!(struct_ctor_ops(&graph), (1, 0, 1));
    assert_eq!(replace_struct_ctors(&mut graph), 1);
    assert_eq!(struct_ctor_ops(&graph), (0, 0, 0));
    assert_eq!(same_as_count(&graph), 0);
    let payload = graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| match &op.kind {
            OpKind::ConstInt(1) => op.result.as_ref(),
            _ => None,
        })
    });
    assert_eq!(
        graph.block(graph.startblock).exits[0].args[0].as_variable(),
        payload,
        "the field read is the stored value"
    );
}

/// Returning the aggregate materialises one `New` at the escape. The
/// constructor is gone; later copies cannot share that object.
#[test]
fn returned_struct_ctor_materializes_a_fresh_new() {
    let mut graph = struct_ctor_graph_escaping(false, StructCtorEscape::Returned);
    assert_eq!(replace_struct_ctors(&mut graph), 1);
    assert_eq!(struct_ctor_ops(&graph), (0, 1, 1));
    let news = struct_news(&graph);
    assert_eq!(news.len(), 1);
    let returned = graph.block(graph.startblock).exits[0]
        .args
        .first()
        .and_then(LinkArg::as_variable);
    assert_eq!(returned, Some(&news[0]));
}

/// Passing the aggregate to a non-malloc call materialises one `New` at
/// that argument. The residual callee receives a distinct object.
#[test]
fn aggregate_passed_by_value_to_a_non_malloc_call() {
    let mut graph = struct_ctor_graph_escaping(false, StructCtorEscape::CallArg);
    assert_eq!(replace_struct_ctors(&mut graph), 1);
    assert_eq!(struct_ctor_ops(&graph), (0, 1, 1));
    let news = struct_news(&graph);
    assert_eq!(news.len(), 1);
    let call_arg = graph
        .block(graph.startblock)
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.as_slice() == ["slice", "index"] => {
                args.first().and_then(LinkArg::as_variable).cloned()
            }
            _ => None,
        });
    assert_eq!(call_arg.as_ref(), Some(&news[0]));
}

/// A phi copy plus a later field write is a distinct allocation: the
/// predecessor materialises a `New`, and the mutation writes that copy.
#[test]
fn aggregate_flowing_into_a_phi() {
    let mut graph = struct_ctor_graph_escaping(false, StructCtorEscape::Phi);
    assert_eq!(replace_struct_ctors(&mut graph), 1);
    assert_eq!(struct_ctor_ops(&graph), (0, 1, 2));
    let news = struct_news(&graph);
    assert_eq!(news.len(), 1);
    let entry = graph.block(graph.startblock);
    let passed = entry.exits[0].args.first().and_then(LinkArg::as_variable);
    assert_eq!(passed, Some(&news[0]));
    let join = &graph.blocks[entry.exits[0].target.0];
    let mutated = join.operations.iter().find_map(|op| match &op.kind {
        OpKind::FieldWrite { base, field, .. } if field.name == "kind" => Some(base.clone()),
        _ => None,
    });
    assert_eq!(mutated.as_ref(), join.inputargs.first());
}

/// `malloc_typed(T { .. })` is the boxing cluster. Its stack aggregate
/// stays a constructor so `fuse_boxing_alloc` can still see it.
#[test]
fn malloc_typed_struct_ctor_is_left_for_boxing_fusion() {
    let mut graph = struct_ctor_graph(true);
    assert_eq!(replace_struct_ctors(&mut graph), 0);
    assert_eq!(struct_ctor_ops(&graph), (1, 0, 1));
}

pub(in crate::front::mir) fn boxing_cluster_with_nested_header() -> FunctionGraph {
    let mut graph = FunctionGraph::new("w_new_int");
    let entry = graph.startblock;
    let header_owner = "ObjectHeader";
    let outer_owner = "W_IntObject";
    let header = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(Vec::new(), header_owner),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(header_owner.to_string())),
            },
            true,
        )
        .expect("header ctor");
    let class_static = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["charon_corpus", "INT_CLASS"]),
                args: Vec::new(),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .expect("class static");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: header.clone(),
            field: FieldDescriptor::new("ob_type", Some(header_owner.to_string())),
            value: LinkArg::Value(class_static),
            ty: ValueType::Ref(None),
        },
        false,
    );
    let outer = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(Vec::new(), outer_owner),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(outer_owner.to_string())),
            },
            true,
        )
        .expect("outer ctor");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: outer.clone(),
            field: FieldDescriptor::new("ob_header", Some(outer_owner.to_string())),
            value: LinkArg::Value(header),
            ty: ValueType::Ref(None),
        },
        false,
    );
    let boxed = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["runtime_object", "lltype", "malloc_typed"]),
                args: crate::model::call_args(vec![outer.clone()]),
                result_ty: ValueType::Ref(Some(outer_owner.to_string())),
            },
            true,
        )
        .expect("malloc_typed");
    graph.set_return(entry, Some(boxed));
    graph
}

pub(in crate::front::mir) fn boxing_cluster_ctor_new_counts(
    graph: &FunctionGraph,
) -> (usize, usize) {
    let mut ctors = 0usize;
    let mut news = 0usize;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        is_struct: true, ..
                    },
                ..
            } => ctors += 1,
            OpKind::New { .. } => news += 1,
            _ => {}
        }
    }
    (ctors, news)
}

/// The nested header stored into a `malloc_typed` aggregate is still
/// part of the boxing cluster. Rewriting it to `New` would allocate the
/// header separately and leave fusion looking at a `New` instead of a
/// constructor.
#[test]
fn malloc_typed_nested_header_ctor_is_left_for_boxing_fusion() {
    let mut graph = boxing_cluster_with_nested_header();
    assert_eq!(boxing_cluster_ctor_new_counts(&graph), (2, 0));
    assert_eq!(replace_struct_ctors(&mut graph), 0);
    assert_eq!(boxing_cluster_ctor_new_counts(&graph), (2, 0));
}

/// `*const () as *const Flavor` (`PyFrame::code`) must emit
/// `__cast_instance_intrinsic` so rewrite can `cast_ptr_to_int`.
/// UnaryOp `CastKind::RawPtr` is the Charon spelling; an Rvalue::Cast
/// whose kind is not the `RawPtr` atom still reaches the marker when
/// the dest is flavor-raw.
#[test]
fn flavor_raw_ptr_narrow_from_unit_emits_cast_instance() {
    let span = fixture_span();
    let vec_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [{"Scalar": {"Integer": {"Unsigned": "Usize"}}}],
                "const_generics": [],
                "trait_refs": []
            },
            "builtin": null
        }
    });
    let owner = struct_decl(
        0,
        ident_path(&["fixture", "RawFlavorStruct"]),
        serde_json::json!([{
            "name": "items",
            "ty": vec_ty,
            "attr_info": null
        }]),
        false,
    );
    let vec_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": fixture_item_meta(ident_path(&["alloc", "vec", "Vec"])),
        "kind": "Opaque"
    });
    let unit_ty = serde_json::json!({
        "Adt": {"builtin": "Tuple", "generics": {"types": []}}
    });
    let src_ty = serde_json::json!({"RawPtr": [unit_ty, "Const"]});
    let dest_ty = serde_json::json!({
        "RawPtr": [{
            "Adt": {
                "id": 0,
                "generics": {
                    "regions": [], "types": [],
                    "const_generics": [], "trait_refs": []
                },
                "builtin": null
            }
        }, "Const"]
    });
    let flavor_global = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(
            ident_path(&["fixture", "_alloc_flavor_RawFlavorStruct"])
        ),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": "raw"}, {"Ref": [
            "Erased",
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Shared"
        ]}]]}
    });
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let unary = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "unary_narrow"])),
        "signature": {
            "is_unsafe": true,
            "inputs": [src_ty.clone()],
            "output": dest_ty.clone()
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span, "ty": dest_ty.clone()},
                {"index": 1, "name": "p", "span": span, "ty": src_ty.clone()}
            ]},
            "body": [{
                "statements": [{
                    "span": span,
                    "kind": {"Assign": [
                        place(0, &dest_ty),
                        {"UnaryOp": [
                            {"Cast": {"RawPtr": [src_ty.clone(), dest_ty.clone()]}},
                            {"Move": place(1, &src_ty)}
                        ]}
                    ]}
                }],
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let rvalue = serde_json::json!({
        "def_id": 1,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "rvalue_narrow"])),
        "signature": {
            "is_unsafe": true,
            "inputs": [src_ty.clone()],
            "output": dest_ty.clone()
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span, "ty": dest_ty.clone()},
                {"index": 1, "name": "p", "span": span, "ty": src_ty.clone()}
            ]},
            "body": [{
                "statements": [{
                    "span": span,
                    "kind": {"Assign": [
                        place(0, &dest_ty),
                        {"Cast": [
                            "PtrToPtr",
                            {"Move": place(1, &src_ty)},
                            dest_ty.clone()
                        ]}
                    ]}
                }],
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [owner, vec_decl],
            "fun_decls": [unary, rvalue],
            "global_decls": [flavor_global],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let has_cast_instance = |graph: &crate::model::FunctionGraph| {
        graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                crate::model::cast_instance_root(&op.kind)
                    .is_some_and(|root| root.contains("RawFlavorStruct"))
            })
        })
    };
    let unary_graph = super::lower_function(&llbc, "unary_narrow").expect("lower unary_narrow");
    assert!(
        has_cast_instance(&unary_graph),
        "UnaryOp Cast RawPtr onto flavor-raw dest emits CAST_INSTANCE"
    );
    let rvalue_graph = super::lower_function(&llbc, "rvalue_narrow").expect("lower rvalue_narrow");
    assert!(
        has_cast_instance(&rvalue_graph),
        "Rvalue::Cast PtrToPtr onto flavor-raw dest emits CAST_INSTANCE"
    );
}

/// Lower one `pyre_object` function by its full `name_path`.
pub(in crate::front::mir) fn lower_object_fn(name_path: &str) -> crate::model::FunctionGraph {
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == name_path)
        .unwrap_or_else(|| panic!("{name_path} in pyre-object LLBC"));
    let context = super::LowerContext::new(&llbc);
    super::lower_fun_decl(&context, fd).unwrap_or_else(|err| panic!("lower {name_path}: {err:?}"))
}

pub(in crate::front::mir) fn calls_leaf(graph: &crate::model::FunctionGraph, leaf: &str) -> bool {
    use crate::model::{CallTarget, OpKind};
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().is_some_and(|s| s == leaf),
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == leaf,
            _ => false,
        })
}

/// Jitcode is produced before the GC transform, so no lowered graph reads
/// the header flag word: `dict_write_barrier`'s `GcHeader::has_flag` stays
/// the call it is in the source.
#[test]
fn dict_write_barrier_does_not_lower_the_header_flag_test() {
    use crate::model::OpKind;
    let graph = lower_object_fn("pyre_object::dictmultiobject::dict_write_barrier");
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        if let OpKind::FieldRead { field, .. } = &op.kind {
            assert_ne!(
                field.name, "tid_and_flags",
                "dict_write_barrier reads the GC header flag word"
            );
        }
    }
}

/// `w_type_set_w_doc` is `type_write_barrier(obj); t.w_doc = w_doc`.
/// The barrier call is dropped and the `setfield` stays; the backend
/// rewrite puts `COND_CALL_GC_WB` in front of it.
#[test]
fn type_write_barrier_call_leaves_only_the_setfield() {
    use crate::model::OpKind;
    let graph = lower_object_fn("pyre_object::typeobject::w_type_set_w_doc");
    assert!(
        !calls_leaf(&graph, "type_write_barrier"),
        "w_type_set_w_doc still calls type_write_barrier"
    );
    assert!(
        graph.blocks.iter().flat_map(|block| &block.operations).any(
            |op| matches!(&op.kind, OpKind::FieldWrite { field, .. } if field.name == "w_doc")
        ),
        "w_type_set_w_doc lost its w_doc setfield"
    );
}

/// `write_cell`'s in-place `ObjectMutableCell` arm is
/// `try_gc_write_barrier_prebuilt(cell); cell.w_value = w_value`. It does
/// not call `try_gc_owns_object` itself. jtransform
/// `drop_guarded_gc_write_barriers` drops that `gc_hook` barrier
/// when the store follows, so rewrite can put `COND_CALL_GC_WB` on
/// the `setfield_gc`.
#[test]
fn write_cell_calls_try_gc_write_barrier_prebuilt_before_w_value_setfield() {
    use crate::model::{CallTarget, OpKind};
    let graph = lower_object_fn("pyre_object::celldict::write_cell");
    assert!(
        !calls_leaf(&graph, "try_gc_owns_object"),
        "write_cell still calls try_gc_owns_object"
    );
    let is_barrier = |kind: &OpKind| {
        matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|s| s == "try_gc_write_barrier_prebuilt")
        )
    };
    let is_store =
        |kind: &OpKind| matches!(kind, OpKind::FieldWrite { field, .. } if field.name == "w_value");
    let mut barrier_before_store = false;
    for block in &graph.blocks {
        let mut saw_barrier = false;
        for op in &block.operations {
            if is_barrier(&op.kind) {
                saw_barrier = true;
            }
            if saw_barrier && is_store(&op.kind) {
                barrier_before_store = true;
            }
        }
        if saw_barrier {
            for succ in graph.successors(block.id) {
                if graph.blocks[succ.0]
                    .operations
                    .iter()
                    .any(|op| is_store(&op.kind))
                {
                    barrier_before_store = true;
                }
            }
        }
    }
    assert!(
        barrier_before_store,
        "write_cell's try_gc_write_barrier_prebuilt does not precede the w_value setfield"
    );
}

/// `ll_list_obj_capacity` is `len(l.items)` (`rlist.py`
/// `_ll_list_resize_hint` `allocated = len(l.items)`). The resizable
/// `ItemsBlock` may be NULL; `items_block_capacity` returns 0 for
/// NULL, so the call stays. PyPy's `ll_newemptylist` allocates a
/// zero-length array, which pyre does not have yet.
#[test]
fn items_block_capacity_on_a_resizable_list_stays_a_call() {
    use crate::model::OpKind;
    let graph = lower_object_fn("pyre_object::listobject::ll_list_obj_capacity");
    assert!(
        calls_leaf(&graph, "items_block_capacity"),
        "ll_list_obj_capacity must keep items_block_capacity"
    );
    let has_arraylen = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, OpKind::ArrayLen { .. }))
    });
    assert!(!has_arraylen, "the items header is not ll_fixed_length");
}

pub(in crate::front::mir) fn graph_call_leaves(graph: &FunctionGraph) -> Vec<String> {
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            _ => None,
        })
        .collect()
}

/// Anchor [`Lowering::fold_size_const_global`] to the real lowered
/// IR of `function_new_impl` (= reads `FUNCTION_OBJECT_SIZE`, a
/// `const usize = size_of::<Function>()`).  The global read must fold
/// to `Function`'s concrete byte size rather than residualize as an
/// unregisterable `FunctionPath` accessor call.  Ignored by default
/// (loads the real LLBC); run with `cargo test -p majit-translate
/// --lib fold_size_const_real -- --ignored`.
///
/// The literal is READ from the artefact this fold consumes, not
/// computed: `pyre-interpreter.ullbc` carries exactly one layout
/// entry for `Function`, `"size":192`, `"align":8`, `"repr_algo":"C"`,
/// with 23 `field_offsets` running `[0,16,24,…,184]`.  A source census
/// of `struct Function` reads the same 23 fields, and `184 + 8` is the
/// same 192.  Because there is a single entry, `select_target_layout`
/// resolves it through its sole-entry fallback even when `TARGET` is
/// unset, so a declining fold cannot be the reason this reds — the
/// second assertion below is what catches that case.
///
/// Re-take the literal from the artefact's `layout` entry whenever
/// `Function` changes.
#[test]
#[ignore]
fn fold_size_const_real_function_object_size() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "function_new_impl").expect("lower function_new_impl");

    let ops: Vec<&OpKind> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .map(|op| &op.kind)
        .collect();

    // The `size_of::<Function>()` const read folded to its byte size.
    // `usize` lowers as `ConstUInt` (`const_lit_to_op`).
    assert!(
        ops.iter().any(|k| matches!(k, OpKind::ConstUInt(192))),
        "expected a ConstUInt(192) for the folded FUNCTION_OBJECT_SIZE"
    );
    // No residual accessor call to the const remains.
    let residual = ops.iter().any(|k| {
        matches!(
            k,
            OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if segments.last().is_some_and(|s| s.ends_with("FUNCTION_OBJECT_SIZE"))
        )
    });
    assert!(
        !residual,
        "FUNCTION_OBJECT_SIZE must not residualize as a FunctionPath call"
    );
}

/// A concrete `Vec<T>` source is the translated-list case of PyPy's
/// `list.extend`; an arbitrary `IntoIterator` is not.  Pin both sides to
/// production LLBC so the frontend never broadens the bridge to Map or a
/// custom iterator.
#[test]
#[ignore]
fn vec_extend_retargets_only_concrete_vec_sources() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let count = |function: &str, leaf: &str| {
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|e| panic!("lower {function}: {e}"));
        graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &["vec", "Vec", leaf]
                )
            })
            .count()
    };

    for function in [
        "pyre_interpreter::argument::combine_starargs_wrapped",
        "pyre_interpreter::argument::combine_starstarargs_wrapped",
    ] {
        assert_eq!(count(function, "extend"), 0, "{function}");
        assert!(count(function, "extend_from_slice") >= 1, "{function}");
    }
    let map_only = "pyre_interpreter::baseobjspace::generator_unpack_into";
    assert_eq!(
        count(map_only, "extend_from_slice"),
        0,
        "{map_only}: an iterator source must not take the list-to-list bridge"
    );
    assert!(count(map_only, "extend") >= 1, "{map_only}");

    // This function has both a concrete Vec source and a Map source.  The
    // former may bridge, but the latter must remain visible as `extend`.
    let mixed = "pyre_interpreter::_structseq::new_instance_with_extra";
    assert!(count(mixed, "extend_from_slice") >= 1, "{mixed}");
    assert!(count(mixed, "extend") >= 1, "{mixed}");
}

/// Anchor the `&self` `Option::is_some` fold to the real lowered IR of
/// `callable_w` — `lookup_in_type(t, "__call__").is_some()`.  The
/// receiver is an owned `Option<PyObjectRef>` that Rust auto-refs to
/// `&Option` at the call, so the receiver type arrives as `Ref(Option)`;
/// the `_ref` guard + `tyref_ref_adt_def_id` peel it (the by-value
/// `tyref_is_option` would see the `Ref` and miss).  After the fold no
/// residual `is_some` Method call survives and a `__discriminant` read is
/// present.  Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn is_some_fold_real_callable_w() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "callable_w").expect("lower callable_w");

    let method_calls = |leaf: &str| {
        graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                        if name == leaf
                )
            })
            .count()
    };
    assert_eq!(
        method_calls("is_some"),
        0,
        "residual &Option::is_some Method call removed by the fold"
    );
    let disc_reads = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
    assert!(disc_reads >= 1, "the fold reads the Option __discriminant");
}

/// Anchor the `Option::or_else` fold to the real lowered IR of
/// `space_int` — `lookup(obj, "__int__").or_else(|| lookup(obj,
/// "__index__"))`.  `or_else` takes `self` by value (like the sibling
/// closure-select combinators), so the by-value receiver resolution
/// applies.  After the fold no residual `or_else` Method call survives,
/// the discriminant diamond branches, and the `Some` arm forwards the
/// receiver (no `Some::__pos_0` read introduced by the rewrite).  Ignored
/// by default (loads the real LLBC).
#[test]
#[ignore]
fn or_else_fold_real_space_int() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "space_int").expect("lower space_int");

    let or_else_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "or_else"
            )
        })
        .count();
    assert_eq!(
        or_else_calls, 0,
        "residual or_else Method call removed by the fold"
    );
    // The niladic `None`-arm closure lowers to a `call_once` Method call.
    let call_once = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "call_once"
            )
        })
        .count();
    assert!(
        call_once >= 1,
        "the None arm calls the or_else closure via call_once"
    );
}

/// `w_module_dict_is_object_strategy::promoted_const#0` constructs
/// `StrategyKind::Object` (`Aggregate` variant 2) and returns `&_1`.
/// That body is a `flowspace` `Constant`, not a residual call.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn promoted_const_fieldless_enum_folds_to_discriminant_int() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_object::dictmultiobject::w_module_dict_is_object_strategy",
    )
    .expect("lower w_module_dict_is_object_strategy");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.iter().any(|s| s.starts_with("promoted_const"))
            ))
            .count(),
        0,
        "promoted_const must fold to a constant, not a residual call"
    );
    assert!(
        ops().any(|op| matches!(&op.kind, OpKind::ConstInt(2))),
        "StrategyKind::Object is discriminant 2"
    );
    assert!(
        ops().all(|op| match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "kind" => {
                !field.taken_by_address
            }
            _ => true,
        }),
        "&StrategyKind is the tag, not the address of the field"
    );
}

/// Minimal `Llbc` carrying only `trait_impls` — the surface
/// [`resolve_trait_assoc_type`] consults.
pub(in crate::front::mir) fn llbc_with_trait_impls(trait_impls: serde_json::Value) -> Llbc {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": trait_impls,
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
}

pub(in crate::front::mir) fn window_span() -> serde_json::Value {
    serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 10}
    }})
}

pub(in crate::front::mir) fn window_meta(path: &[&str], is_local: bool) -> serde_json::Value {
    serde_json::json!({
        "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
        "span": window_span(),
        "source_text": null,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
        "is_local": is_local
    })
}

pub(in crate::front::mir) fn window_local(i: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"index": i, "name": null, "span": window_span(), "ty": ty})
}

pub(in crate::front::mir) fn window_place(
    kind: serde_json::Value,
    ty: &serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({"kind": kind, "ty": ty})
}

pub(in crate::front::mir) fn window_u8() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})
}

pub(in crate::front::mir) fn window_usize() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
}

pub(in crate::front::mir) fn window_empty_generics() -> serde_json::Value {
    serde_json::json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []})
}

pub(in crate::front::mir) fn window_slice() -> serde_json::Value {
    serde_json::json!({"Slice": [window_u8(), null]})
}

pub(in crate::front::mir) fn window_shared_slice() -> serde_json::Value {
    serde_json::json!({"Ref": ["Erased", window_slice(), "Shared"]})
}

pub(in crate::front::mir) fn window_mut_slice() -> serde_json::Value {
    serde_json::json!({"Ref": ["Erased", window_slice(), "Mut"]})
}

pub(in crate::front::mir) fn window_const_slice_ptr() -> serde_json::Value {
    serde_json::json!({"RawPtr": [window_slice(), "Const"]})
}

fn window_const_u8_ptr() -> serde_json::Value {
    serde_json::json!({"RawPtr": [window_u8(), "Const"]})
}

fn window_mut_u8_ptr() -> serde_json::Value {
    serde_json::json!({"RawPtr": [window_u8(), "Mut"]})
}

pub(in crate::front::mir) fn window_opaque_fun(
    def_id: u64,
    path: &[&str],
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": window_meta(path, false),
        "signature": {"is_unsafe": true, "inputs": inputs, "output": output},
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn window_llbc(
    types: Vec<serde_json::Value>,
    funs: Vec<serde_json::Value>,
) -> majit_charon_reader::Llbc {
    window_llbc_with_globals(types, funs, Vec::new())
}

fn window_llbc_with_globals(
    types: Vec<serde_json::Value>,
    funs: Vec<serde_json::Value>,
    globals: Vec<serde_json::Value>,
) -> majit_charon_reader::Llbc {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": types,
            "fun_decls": funs,
            "global_decls": globals,
            "trait_decls": [],
            "trait_impls": []
        }
    });
    majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
        .expect("window fixture Llbc parses")
}

pub(in crate::front::mir) fn window_graph_ops(
    graph: &crate::model::FunctionGraph,
) -> Vec<&crate::model::SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect()
}

pub(in crate::front::mir) fn window_call_leafs(graph: &crate::model::FunctionGraph) -> Vec<String> {
    use crate::model::{CallTarget, OpKind};
    window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            _ => None,
        })
        .collect()
}

fn window_has_join_chars(graph: &crate::model::FunctionGraph) -> bool {
    window_call_leafs(graph).iter().any(|leaf| {
        leaf == "ll_join_chars" || leaf == "ll_str2uintlist" || leaf == "ll_string2list"
    })
}

pub(in crate::front::mir) fn window_unwind_and_return_blocks()
-> (serde_json::Value, serde_json::Value) {
    (
        serde_json::json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": window_span(), "kind": "UnwindResume"}
        }),
        serde_json::json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": window_span(), "kind": "Return"}
        }),
    )
}

/// F1: a `*const [u8]` field read is the two-word window, not STR.
#[test]
fn window_f1_const_u8_slice_field_read_is_fat_window() {
    use crate::model::{OpKind, ValueType, VecFieldPart};

    let holder_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let holder_ref = serde_json::json!({"Ref": ["static", holder_adt, "Shared"]});
    let field_ty = window_const_slice_ptr();
    let holder = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "Holder"], true),
        "kind": {"Struct": [{
            "name": "data",
            "ty": field_ty,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        }]}
    });
    let slot_place = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Projection": [
                window_place(serde_json::json!({"Local": 1}), &holder_ref),
                "Deref"
            ]}), &holder_adt),
            {"Field": [null, 0]}
        ]}),
        &field_ty,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_f1_read"], true),
        "signature": {"is_unsafe": false, "inputs": [holder_ref], "output": field_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &field_ty),
                window_local(1, &holder_ref)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        window_place(serde_json::json!({"Local": 0}), &field_ty),
                        {"Use": [{"Copy": slot_place}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let llbc = window_llbc(vec![holder], vec![caller]);
    let graph = super::lower_function(&llbc, "window_f1_read").expect("lower F1");
    let ops = window_graph_ops(&graph);
    let fat_data = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, ty, .. }
                if field.name == "data"
                    && field.vec_part == Some(VecFieldPart::FatData)
                    && !matches!(ty, ValueType::Str)
        )
    });
    let fat_len = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, ty, .. }
                if field.name == "data"
                    && field.vec_part == Some(VecFieldPart::FatLen)
                    && !matches!(ty, ValueType::Str)
        )
    });
    let str_read = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, ty: ValueType::Str, .. } if field.name == "data"
        )
    });
    assert!(
        fat_data && fat_len,
        "F1 field read is FatData/FatLen; graph={}",
        graph.dump()
    );
    assert!(
        !str_read,
        "F1 field read is not getfield of Str; graph={}",
        graph.dump()
    );
    assert!(
        !window_has_join_chars(&graph),
        "F1 must not copy; graph={}",
        graph.dump()
    );
}

/// F3: `&*m` of `m: &mut [u8]` into a `&[u8]` dest is LIST→WINDOW.
#[test]
fn window_f3_mut_slice_reborrow_is_list_to_window() {
    let shared = window_shared_slice();
    let mut_slice = window_mut_slice();
    let bare = window_slice();
    let deref = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Local": 1}), &mut_slice),
            "Deref"
        ]}),
        &bare,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_f3_reborrow"], true),
        "signature": {"is_unsafe": false, "inputs": [mut_slice], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &shared),
                window_local(1, &mut_slice)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        window_place(serde_json::json!({"Local": 0}), &shared),
                        {"Ref": {"place": deref, "kind": "Shared", "ptr_metadata": null}}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller]);
    let graph = super::lower_function(&llbc, "window_f3_reborrow").expect("lower F3");
    assert!(
        !window_has_join_chars(&graph),
        "F3 &*m is a list-to-window borrow; graph={}",
        graph.dump()
    );
    let aliases_as_str = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            crate::model::OpKind::Call {
                result_ty: crate::model::ValueType::Str,
                ..
            }
        )
    });
    assert!(
        !aliases_as_str,
        "F3 must not alias the list as Str; graph={}",
        graph.dump()
    );
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST
        }),
        "F3 LIST→WINDOW is nonmoving_raw_ptr_for_resizable_list; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let buf_field = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            crate::model::OpKind::FieldRead { field, .. } if field.name == "buf"
        )
    });
    assert!(
        !buf_field,
        "F3 must not FieldRead buf; graph={}",
        graph.dump()
    );
}

/// B1: `Vec<u8>` reborrowed into a `&[u8]` window dest is LIST→WINDOW
/// via `rgc.nonmoving_raw_ptr_for_resizable_list`, never `FieldRead "buf"`.
#[test]
fn window_vec_u8_as_slice_is_list_to_window() {
    use crate::model::OpKind;

    let u8_ty = window_u8();
    let vec_generics = serde_json::json!({
        "regions": [], "types": [u8_ty], "const_generics": [], "trait_refs": []
    });
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": vec_generics}
    });
    let vec_ref = serde_json::json!({"Ref": ["Erased", vec_ty.clone(), "Mut"]});
    let shared = window_shared_slice();
    let vec_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["alloc", "vec", "Vec"], false),
        "kind": {"Struct": []}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_slice = window_opaque_fun(
        1,
        &["alloc", "vec", "<Impl>", "as_slice"],
        vec![vec_ref.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_vec_u8_as_slice"], true),
        "signature": {"is_unsafe": false, "inputs": [vec_ref], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &shared),
                window_local(1, &vec_ref)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &vec_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![vec_decl], vec![caller, as_slice]);
    let graph =
        super::lower_function(&llbc, "window_vec_u8_as_slice").expect("lower Vec<u8> as_slice");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST
        }),
        "Vec<u8> as_slice is nonmoving_raw_ptr_for_resizable_list; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let buf_field = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "buf"
        )
    });
    assert!(
        !buf_field,
        "Vec<u8> as_slice must not FieldRead buf; graph={}",
        graph.dump()
    );
    assert!(
        !window_has_join_chars(&graph),
        "Vec<u8> as_slice is a list-to-window borrow; graph={}",
        graph.dump()
    );
    assert!(
        !leafs.iter().any(|leaf| leaf == "as_slice"),
        "Vec<u8> as_slice is identity, not a residual call; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// A `&&[u8]` reborrow of a STR view is identity on the one `Ptr(STR)`
/// word: the dest is not stamped as a second string pointer.
#[test]
fn str_view_shared_slice_reborrow_is_one_word() {
    use crate::translator::rtyper::lltypesystem::rstr::STRPTR;

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let view_ref = serde_json::json!({"Ref": ["Erased", shared.clone(), "Shared"]});
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, _) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let view_place = window_place(serde_json::json!({"Local": 2}), &shared);
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "str_view_reborrow"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref], "output": view_ref},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &view_ref),
                window_local(1, &utf8_ref),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 2}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &view_ref),
                            {"Ref": {"place": view_place, "kind": "Shared", "ptr_metadata": null}}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                }
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller, as_bytes]);
    let graph = super::lower_function(&llbc, "str_view_reborrow")
        .expect("lower &&[u8] reborrow of a STR view");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING || leaf == "__strlen"
        }),
        "&&[u8] reborrow of a STR view is identity; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let returned = graph
        .blocks
        .iter()
        .flat_map(|block| block.exits.iter())
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .and_then(crate::model::LinkArg::as_variable)
        .expect("return carries the view word");
    assert_eq!(
        returned.concretetype().as_ref(),
        Some(&*STRPTR),
        "&&[u8] reborrow keeps one Ptr(STR) word; graph={}",
        graph.dump()
    );
}

/// F4: `&mut [u8]` written through `as_mut_ptr` does not copy.
#[test]
fn window_f4_mut_slice_as_mut_ptr_write_is_not_a_copy() {
    use crate::model::OpKind;

    let mut_slice = window_mut_slice();
    let mut_ptr = window_mut_u8_ptr();
    let u8_ty = window_u8();
    let (unwind, _) = window_unwind_and_return_blocks();
    let store_place = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Local": 0}), &mut_ptr),
            "Deref"
        ]}),
        &u8_ty,
    );
    let one = serde_json::json!({"Const": [
        {"Integer": {"Unsigned": ["U8", "1"]}},
        u8_ty
    ]});
    let as_mut_ptr = window_opaque_fun(
        1,
        &["core", "slice", "<Impl>", "as_mut_ptr"],
        vec![mut_slice.clone()],
        mut_ptr.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_f4_as_mut_ptr"], true),
        "signature": {"is_unsafe": true, "inputs": [mut_slice], "output": mut_ptr},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &mut_ptr),
                window_local(1, &mut_slice)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &mut_slice)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &mut_ptr)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            store_place,
                            {"Use": [one, "No"]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                }
            ]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller, as_mut_ptr]);
    let graph = super::lower_function(&llbc, "window_f4_as_mut_ptr").expect("lower F4");
    assert!(
        !window_has_join_chars(&graph),
        "F4 as_mut_ptr write is not a copy; graph={}",
        graph.dump()
    );
    let residual_as_mut_ptr = window_call_leafs(&graph)
        .iter()
        .any(|leaf| leaf == "as_mut_ptr");
    assert!(
        !residual_as_mut_ptr,
        "F4 as_mut_ptr is identity on the list items; graph={}",
        graph.dump()
    );
    let wrote = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::RawStore { .. } | OpKind::ArrayWrite { .. }
        )
    });
    assert!(
        wrote,
        "F4 writes through the mut pointer; graph={}",
        graph.dump()
    );
}

/// F5: `from_raw_parts` of a non-STR address is a window, not STR.
#[test]
fn window_f5_from_raw_parts_of_non_str_address_is_window() {
    use crate::model::{OpKind, ValueType};

    let shared = window_shared_slice();
    let ptr = window_const_u8_ptr();
    let n = window_usize();
    let (unwind, ret) = window_unwind_and_return_blocks();
    let from_raw = window_opaque_fun(
        1,
        &["core", "slice", "raw", "from_raw_parts"],
        vec![ptr.clone(), n.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_f5_from_raw_parts"], true),
        "signature": {"is_unsafe": true, "inputs": [ptr, n], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &shared),
                window_local(1, &ptr),
                window_local(2, &n)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 1}), &ptr)},
                            {"Copy": window_place(serde_json::json!({"Local": 2}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller, from_raw]);
    let graph = super::lower_function(&llbc, "window_f5_from_raw_parts").expect("lower F5");
    assert!(
        !window_has_join_chars(&graph),
        "F5 is a window of (p, n); graph={}",
        graph.dump()
    );
    let paints_str = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                result_ty: ValueType::Str,
                ..
            } | OpKind::FieldRead {
                ty: ValueType::Str,
                ..
            }
        )
    });
    assert!(
        !paints_str,
        "F5 from_raw_parts dest is not Str; graph={}",
        graph.dump()
    );
}

/// `fn f() -> &'static [u8] { g() }` with opaque `g`: a nullary call
/// whose dest is a shared `&[u8]` window must not index `args[0]`.
#[test]
fn window_nullary_opaque_call_to_shared_slice_does_not_panic() {
    let shared = window_shared_slice();
    let (unwind, ret) = window_unwind_and_return_blocks();
    let g = window_opaque_fun(1, &["fixture", "g"], vec![], shared.clone());
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_nullary_g"], true),
        "signature": {"is_unsafe": false, "inputs": [], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 0, "locals": [
                window_local(0, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller, g]);
    let _ = super::lower_function(&llbc, "window_nullary_g");
}

/// `fn f(b: &[u8]) -> i64 { g(b) }` with opaque window-param `g`: pass
/// the two parameter words, never the list shim.
#[test]
fn window_param_passed_to_opaque_window_fn_is_identity() {
    use crate::model::{CallTarget, LinkArg, OpKind};

    let shared = window_shared_slice();
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let (unwind, ret) = window_unwind_and_return_blocks();
    let g = window_opaque_fun(1, &["fixture", "g"], vec![shared.clone()], i64_ty.clone());
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_pass_param"], true),
        "signature": {"is_unsafe": true, "inputs": [shared], "output": i64_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &i64_ty),
                window_local(1, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &i64_ty)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller, g]);
    let graph =
        super::lower_function(&llbc, "window_pass_param").expect("lower window param pass-through");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST
        }),
        "window param must not take the list shim; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let inputs: Vec<_> = window_graph_ops(&graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { .. } => op.result.clone(),
            _ => None,
        })
        .collect();
    let g_args: Option<Vec<_>> =
        window_graph_ops(&graph)
            .into_iter()
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("g") => Some(
                    args.iter()
                        .filter_map(LinkArg::as_variable)
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            });
    let g_args = g_args.expect("opaque g is called");
    assert_eq!(
        g_args.len(),
        2,
        "g receives the two window words; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert_eq!(
        inputs.len(),
        2,
        "window param is two Input words; graph={}",
        graph.dump()
    );
    assert_eq!(
        g_args,
        inputs,
        "g receives the two parameter words unchanged; graph={}",
        graph.dump()
    );
}

/// `fn f(s: &[u8]) -> Vec<u8> { s.to_vec() }` of a window parameter is
/// `rffi.charpsize2str` of the two window words.
#[test]
fn window_param_to_vec_is_charpsize2str() {
    use crate::model::{CallTarget, LinkArg, OpKind};

    let shared = window_shared_slice();
    let u8_ty = window_u8();
    let vec_generics = serde_json::json!({
        "regions": [], "types": [u8_ty], "const_generics": [], "trait_refs": []
    });
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": vec_generics}
    });
    let vec_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["alloc", "vec", "Vec"], false),
        "kind": {"Struct": []}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let to_vec = window_opaque_fun(
        1,
        &["alloc", "slice", "<Impl>", "to_vec"],
        vec![shared.clone()],
        vec_ty.clone(),
    );
    let charpsize2str = window_opaque_fun(
        2,
        &["majit_rlib", "rffi", "buffer", "charpsize2str"],
        vec![window_mut_u8_ptr(), window_usize()],
        vec_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_param_to_vec"], true),
        "signature": {"is_unsafe": false, "inputs": [shared], "output": vec_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &vec_ty),
                window_local(1, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &vec_ty)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![vec_decl], vec![caller, to_vec, charpsize2str]);
    let graph =
        super::lower_function(&llbc, "window_param_to_vec").expect("lower window param to_vec");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "charpsize2str"),
        "window to_vec is charpsize2str; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        !leafs.iter().any(|leaf| leaf == "list"),
        "window to_vec must not call the list builtin; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        !leafs
            .iter()
            .any(|leaf| leaf == "getitem" || leaf == "getanyitem"),
        "window to_vec must not getitem the data word; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let inputs: Vec<_> = window_graph_ops(&graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { .. } => op.result.clone(),
            _ => None,
        })
        .collect();
    let cps: Option<(Vec<_>, Option<u64>)> =
        window_graph_ops(&graph)
            .into_iter()
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target:
                        CallTarget::FunctionPath {
                            segments,
                            fun_decl_id,
                            ..
                        },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("charpsize2str") => Some((
                    args.iter()
                        .filter_map(LinkArg::as_variable)
                        .cloned()
                        .collect::<Vec<_>>(),
                    *fun_decl_id,
                )),
                _ => None,
            });
    let (cps_args, fun_decl_id) = cps.expect("charpsize2str is called");
    assert_eq!(
        fun_decl_id,
        Some(2),
        "charpsize2str carries its FunDecl id; graph={}",
        graph.dump()
    );
    assert_eq!(
        cps_args,
        inputs,
        "charpsize2str receives the two window words; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// `owner_to_window` of an unclassified owner is `Unsupported`.
#[test]
fn owner_to_window_unclassified_owner_is_unsupported() {
    let shared = window_shared_slice();
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_unclassified_owner"], true),
        "signature": {"is_unsafe": false, "inputs": [i64_ty], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &shared),
                window_local(1, &i64_ty)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        window_place(serde_json::json!({"Local": 0}), &shared),
                        {"Use": [{"Copy": window_place(serde_json::json!({"Local": 1}), &i64_ty)}, "No"]}
                    ]}
                }],
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller]);
    let err = super::lower_function(&llbc, "window_unclassified_owner")
        .expect_err("unclassified owner_to_window");
    match err {
        super::LowerError::Unsupported(msg) => {
            assert!(
                msg.contains("unclassified owner"),
                "expected unclassified owner, got {msg}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

/// `fn f(b: &[u8]) { let r = &*b; g(r) }`: the Deref reborrow of a
/// window param stays the two parameter words. Bare `[u8]` is the
/// window referent, not a W2 list, so the list shim is not applied.
#[test]
fn window_param_deref_reborrow_passed_to_opaque_window_fn_is_identity() {
    use crate::model::{CallTarget, LinkArg, OpKind};

    let shared = window_shared_slice();
    let bare = window_slice();
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let (unwind, ret) = window_unwind_and_return_blocks();
    let g = window_opaque_fun(1, &["fixture", "g"], vec![shared.clone()], i64_ty.clone());
    let deref = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Local": 1}), &shared),
            "Deref"
        ]}),
        &bare,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_pass_deref"], true),
        "signature": {"is_unsafe": true, "inputs": [shared], "output": i64_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &i64_ty),
                window_local(1, &shared),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &shared),
                            {"Ref": {"place": deref, "kind": "Shared", "ptr_metadata": null}}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &i64_ty)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller, g]);
    let graph = super::lower_function(&llbc, "window_pass_deref")
        .expect("lower window param Deref reborrow");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST
        }),
        "Deref of a window param must not take the list shim; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let inputs: Vec<_> = window_graph_ops(&graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { .. } => op.result.clone(),
            _ => None,
        })
        .collect();
    let g_args: Option<Vec<_>> =
        window_graph_ops(&graph)
            .into_iter()
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("g") => Some(
                    args.iter()
                        .filter_map(LinkArg::as_variable)
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            });
    let g_args = g_args.expect("opaque g is called");
    assert_eq!(
        g_args,
        inputs,
        "g receives the two parameter words; graph={}",
        graph.dump()
    );
}

/// `fn f(s: &[u8]) -> &[u8] { tail(s) }` with opaque `tail`: a real
/// callee is not an identity of its first argument.
#[test]
fn window_opaque_tail_of_u8_slice_keeps_call() {
    let shared = window_shared_slice();
    let (unwind, ret) = window_unwind_and_return_blocks();
    let tail = window_opaque_fun(
        1,
        &["fixture", "tail"],
        vec![shared.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_opaque_tail"], true),
        "signature": {"is_unsafe": true, "inputs": [shared], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &shared),
                window_local(1, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![], vec![caller, tail]);
    let graph =
        super::lower_function(&llbc, "window_opaque_tail").expect("lower opaque tail of a window");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "tail"),
        "opaque tail must remain a Call; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// `fn f(s: &str) -> &[u8] { g(s) }` with opaque user `g`: a STR
/// argument does not prove the callee is an identity view.
#[test]
fn window_opaque_str_to_u8_slice_keeps_call() {
    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let g = window_opaque_fun(1, &["fixture", "g"], vec![utf8_ref.clone()], shared.clone());
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_opaque_str_g"], true),
        "signature": {"is_unsafe": true, "inputs": [utf8_ref], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &shared),
                window_local(1, &utf8_ref)
            ]},
            "body": [
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller, g]);
    let graph = super::lower_function(&llbc, "window_opaque_str_g")
        .expect("lower opaque STR→window callee");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "g"),
        "opaque g taking STR must remain a Call; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// A subslice of a `from_raw_parts` window is `window_subslice`, not
/// STR `getslice`.
#[test]
fn window_from_raw_parts_subslice_is_not_getslice() {
    use crate::model::OpKind;

    let shared = window_shared_slice();
    let ptr = window_const_u8_ptr();
    let n = window_usize();
    let range_generics = serde_json::json!({
        "regions": [],
        "types": [n.clone()],
        "const_generics": [],
        "trait_refs": []
    });
    let range_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": range_generics.clone()}
    });
    let range = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["core", "ops", "range", "Range"], true),
        "kind": {"Struct": [
            {"name": "start", "ty": n, "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}},
            {"name": "end", "ty": n, "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let from_raw = window_opaque_fun(
        1,
        &["core", "slice", "raw", "from_raw_parts"],
        vec![ptr.clone(), n.clone()],
        shared.clone(),
    );
    let generic_slice = serde_json::json!({
        "Ref": ["Erased", {"Slice": [{"TypeVar": {"Bound": [0, 0]}}, null]}, "Shared"]
    });
    let index = window_opaque_fun(
        2,
        &["core", "slice", "index", "<Impl>", "index"],
        vec![generic_slice, range_adt.clone()],
        shared.clone(),
    );
    let range_place = window_place(serde_json::json!({"Local": 6}), &range_adt);
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_from_raw_parts_subslice"], true),
        "signature": {"is_unsafe": true, "inputs": [ptr, n, n, n], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 4, "locals": [
                window_local(0, &shared),
                window_local(1, &ptr),
                window_local(2, &n),
                window_local(3, &n),
                window_local(4, &n),
                window_local(5, &shared),
                window_local(6, &range_adt)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 1}), &ptr)},
                            {"Copy": window_place(serde_json::json!({"Local": 2}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 5}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            range_place,
                            {"Aggregate": [
                                {"Adt": [
                                    {"id": 0, "generics": range_generics.clone()},
                                    0,
                                    null
                                ]},
                                [
                                    {"Copy": window_place(serde_json::json!({"Local": 3}), &n)},
                                    {"Copy": window_place(serde_json::json!({"Local": 4}), &n)}
                                ]
                            ]}
                        ]}
                    }],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 5}), &shared)},
                            {"Move": window_place(serde_json::json!({"Local": 6}), &range_adt)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![range], vec![caller, from_raw, index]);
    let graph = super::lower_function(&llbc, "window_from_raw_parts_subslice")
        .expect("lower from_raw_parts window subslice");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs
            .iter()
            .any(|leaf| leaf.starts_with("__getslice_") || leaf == "getslice"),
        "window subslice is not getslice; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let has_getslice = window_graph_ops(&graph)
        .iter()
        .any(|op| matches!(&op.kind, OpKind::GetSlice { .. }));
    assert!(
        !has_getslice,
        "window subslice has no GetSlice op; graph={}",
        graph.dump()
    );
}

/// Match arms join at a window local, then a subslice: the real
/// `Buffer::as_bytes` Sub shape. The join is a window, so the index
/// is `window_subslice` (`direct_ptradd`), not STR `GetSlice`.
#[test]
fn window_match_join_then_subslice_is_not_getslice() {
    use crate::model::OpKind;

    let shared = window_shared_slice();
    let ptr = window_const_u8_ptr();
    let n = window_usize();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let range_generics = serde_json::json!({
        "regions": [],
        "types": [n.clone()],
        "const_generics": [],
        "trait_refs": []
    });
    let range_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": range_generics.clone()}
    });
    let range = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["core", "ops", "range", "Range"], true),
        "kind": {"Struct": [
            {"name": "start", "ty": n, "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}},
            {"name": "end", "ty": n, "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let from_raw = window_opaque_fun(
        1,
        &["core", "slice", "raw", "from_raw_parts"],
        vec![ptr.clone(), n.clone()],
        shared.clone(),
    );
    let generic_slice = serde_json::json!({
        "Ref": ["Erased", {"Slice": [{"TypeVar": {"Bound": [0, 0]}}, null]}, "Shared"]
    });
    let index = window_opaque_fun(
        2,
        &["core", "slice", "index", "<Impl>", "index"],
        vec![generic_slice, range_adt.clone()],
        shared.clone(),
    );
    let range_place = window_place(serde_json::json!({"Local": 9}), &range_adt);
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_match_join_subslice"], true),
        "signature": {"is_unsafe": true, "inputs": [bool_ty, ptr, n, ptr, n, n, n], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 7, "locals": [
                window_local(0, &shared),
                window_local(1, &bool_ty),
                window_local(2, &ptr),
                window_local(3, &n),
                window_local(4, &ptr),
                window_local(5, &n),
                window_local(6, &n),
                window_local(7, &n),
                window_local(8, &shared),
                window_local(9, &range_adt)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Switch": {
                        "discr": {"Copy": window_place(serde_json::json!({"Local": 1}), &bool_ty)},
                        "targets": {"If": [2, 3]}
                    }}}
                },
                unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 2}), &ptr)},
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 8}), &shared)
                    }, "target": 4, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 4}), &ptr)},
                            {"Copy": window_place(serde_json::json!({"Local": 5}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 8}), &shared)
                    }, "target": 4, "on_unwind": 1}}}
                },
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            range_place,
                            {"Aggregate": [
                                {"Adt": [
                                    {"id": 0, "generics": range_generics.clone()},
                                    0,
                                    null
                                ]},
                                [
                                    {"Copy": window_place(serde_json::json!({"Local": 6}), &n)},
                                    {"Copy": window_place(serde_json::json!({"Local": 7}), &n)}
                                ]
                            ]}
                        ]}
                    }],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 8}), &shared)},
                            {"Move": window_place(serde_json::json!({"Local": 9}), &range_adt)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 5, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![range], vec![caller, from_raw, index]);
    let graph = super::lower_function(&llbc, "window_match_join_subslice")
        .expect("lower match-join window subslice");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| {
            leaf.starts_with("__getslice_") || leaf == "getslice" || leaf.contains("ll_listslice")
        }),
        "join-then-subslice is not getslice/ll_listslice; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let has_getslice = window_graph_ops(&graph)
        .iter()
        .any(|op| matches!(&op.kind, OpKind::GetSlice { .. }));
    assert!(
        !has_getslice,
        "join-then-subslice has no GetSlice op; graph={}",
        graph.dump()
    );
    assert!(
        leafs.iter().any(|leaf| leaf == "direct_ptradd"),
        "join-then-subslice is window_subslice (direct_ptradd); leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// STR→WINDOW: `s.as_bytes()` of a `Utf8Str` stays a `Ptr(STR)` view at
/// the definition; `{chars, length}` is borrowed at the window return.
#[test]
fn window_str_to_window_as_bytes_of_utf8str() {
    use crate::model::{OpKind, ValueType};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_str_to_window"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &shared),
                window_local(1, &utf8_ref)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller, as_bytes]);
    let graph = super::lower_function(&llbc, "window_str_to_window").expect("lower STR→WINDOW");
    assert!(
        !window_has_join_chars(&graph),
        "STR→WINDOW is a borrow of chars+len; graph={}",
        graph.dump()
    );
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs
            .iter()
            .any(|leaf| { leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING })
            && leafs.iter().any(|leaf| leaf == "__strlen"),
        "STR→WINDOW plants get_raw_address_of_string and length; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let paints_str = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: crate::model::CallTarget::FunctionPath { segments, .. },
                result_ty: ValueType::Str,
                ..
            } if segments.last().map(String::as_str) == Some("as_bytes")
        )
    });
    assert!(
        !paints_str,
        "as_bytes dest is a window, not Str; graph={}",
        graph.dump()
    );
    assert!(
        !leafs.iter().any(|leaf| leaf == "as_bytes"),
        "as_bytes is identity, not a residual call; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// A `#[repr(transparent)]` STR-pointer newtype `as_bytes` is a W1
/// view no-op: no Call and no `getattr("as_bytes")`.
#[test]
fn newtype_str_as_bytes_is_view_noop() {
    use crate::model::{CallTarget, OpKind};

    let block_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let k_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let k_value = serde_json::json!({"Value": [0, k_adt.clone()]});
    let k_ref = serde_json::json!({"Ref": ["Erased", k_adt.clone(), "Shared"]});
    let block_ptr = serde_json::json!({"RawPtr": [block_adt.clone(), "Mut"]});
    let shared = window_shared_slice();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let block = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "bytesobject", "BytesBlock"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let key = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["fixture", "K"], true),
        "kind": {"Struct": [
            {"name": null, "ty": block_ptr, "attr_info": null}
        ]},
        "layout": [{
            "key": "fixture-target",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"transparent": true}
            }
        }]
    });
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": k_value,
        "kind": "InherentImplBlock"
    }}});
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["K", 0]},
                impl_seg,
                {"Ident": ["as_bytes", 0]}
            ],
            "span": window_span(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {"is_unsafe": false, "inputs": [k_ref.clone()], "output": shared.clone()},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": window_span(), "ty": shared.clone()},
                {"index": 1, "name": "self", "span": window_span(), "ty": k_ref.clone()}
            ]},
            "body": [{
                "statements": [],
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let eq = window_opaque_fun(
        2,
        &["core", "slice", "cmp", "<Impl>", "eq"],
        vec![shared.clone(), shared.clone()],
        bool_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "newtype_str_as_bytes_eq"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [k_ref.clone(), k_ref.clone()],
            "output": bool_ty.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &bool_ty),
                window_local(1, &k_ref),
                window_local(2, &k_ref),
                window_local(3, &shared),
                window_local(4, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &k_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 3}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &k_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 4}), &shared)
                    }, "target": 3, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &shared)},
                            {"Copy": window_place(serde_json::json!({"Local": 4}), &shared)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &bool_ty)
                    }, "target": 4, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![block, key], vec![caller, as_bytes, eq]);
    let graph = super::lower_function(&llbc, "newtype_str_as_bytes_eq")
        .expect("lower newtype as_bytes equality");
    let ops = window_graph_ops(&graph);
    let as_bytes_call = ops.iter().any(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => segments.last().map(String::as_str) == Some("as_bytes"),
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } => name == "as_bytes",
        _ => false,
    });
    assert!(
        !as_bytes_call,
        "newtype as_bytes is a view no-op; graph={}",
        graph.dump()
    );
}

/// An `as_*` whose dest is `Option<&[u8]>` is not a view no-op: the
/// Option is a different representation class from the STR receiver.
#[test]
fn as_star_into_option_dest_is_not_a_view_noop() {
    use crate::model::{CallTarget, OpKind};

    let block_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let k_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let k_value = serde_json::json!({"Value": [0, k_adt.clone()]});
    let k_ref = serde_json::json!({"Ref": ["Erased", k_adt.clone(), "Shared"]});
    let block_ptr = serde_json::json!({"RawPtr": [block_adt.clone(), "Mut"]});
    let option_ty = option_adt_node(2, window_shared_slice());
    let block = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "bytesobject", "BytesBlock"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let key = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["fixture", "K"], true),
        "kind": {"Struct": [
            {"name": null, "ty": block_ptr, "attr_info": null}
        ]},
        "layout": [{
            "key": "fixture-target",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"transparent": true}
            }
        }]
    });
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": k_value,
        "kind": "InherentImplBlock"
    }}});
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_opt = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["K", 0]},
                impl_seg,
                {"Ident": ["as_opt", 0]}
            ],
            "span": window_span(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {"is_unsafe": false, "inputs": [k_ref.clone()], "output": option_ty.clone()},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": window_span(), "ty": option_ty.clone()},
                {"index": 1, "name": "self", "span": window_span(), "ty": k_ref.clone()}
            ]},
            "body": [{
                "statements": [],
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "f"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [k_ref.clone()],
            "output": option_ty.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &option_ty),
                window_local(1, &k_ref)
            ]},
            "body": [
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &k_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &option_ty)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![block, key, option_enum(2)], vec![caller, as_opt]);
    let graph = super::lower_function(&llbc, "f").expect("lower as_opt into Option dest");
    let ops = window_graph_ops(&graph);
    let as_opt_call = ops.iter().any(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => segments.last().map(String::as_str) == Some("as_opt"),
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } => name == "as_opt",
        _ => false,
    });
    assert!(
        as_opt_call,
        "as_opt into Option<&[u8]> is not a view no-op; graph={}",
        graph.dump()
    );
}

/// Heap `String::as_bytes` into a shared `&[u8]` parameter keeps the
/// residual `as_bytes` call; the receiver is not a W1 STR.
#[test]
fn heap_string_as_bytes_into_window_param_keeps_call() {
    let string_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let string_ref = serde_json::json!({"Ref": ["Erased", string_adt.clone(), "Shared"]});
    let shared = window_shared_slice();
    let unit = serde_json::json!({"Tuple": []});
    let string_td = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["alloc", "string", "String"], true),
        "kind": {"Struct": [
            {"name": "ptr", "ty": window_usize(), "attr_info": null},
            {"name": "len", "ty": window_usize(), "attr_info": null},
            {"name": "cap", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["alloc", "string", "<Impl>", "as_bytes"],
        vec![string_ref.clone()],
        shared.clone(),
    );
    let callee = window_opaque_fun(
        2,
        &["fixture", "window_param"],
        vec![shared.clone()],
        unit.clone(),
    );
    let s_place = window_place(serde_json::json!({"Local": 1}), &string_adt);
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "string_as_bytes_window_param"], true),
        "signature": {"is_unsafe": false, "inputs": [string_adt.clone()], "output": unit},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &unit),
                window_local(1, &string_adt),
                window_local(2, &string_ref),
                window_local(3, &shared)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &string_ref),
                            {"Ref": {"place": s_place, "kind": "Shared", "ptr_metadata": null}}
                        ]}
                    }],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &string_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 3}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 3}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &unit)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![string_td], vec![caller, as_bytes, callee]);
    let graph = super::lower_function(&llbc, "string_as_bytes_window_param")
        .expect("lower String::as_bytes into window param");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "as_bytes"),
        "heap String::as_bytes keeps the residual call; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        !leafs
            .iter()
            .any(|leaf| leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING),
        "heap String::as_bytes is not get_raw_address_of_string; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// Heap `String::as_ref` into a shared `&[u8]` parameter keeps the
/// residual `as_ref` call; the receiver is not a W1 STR.
#[test]
fn heap_string_as_ref_bytes_into_window_param_keeps_call() {
    let string_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let string_ref = serde_json::json!({"Ref": ["Erased", string_adt.clone(), "Shared"]});
    let shared = window_shared_slice();
    let unit = serde_json::json!({"Tuple": []});
    let string_td = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["alloc", "string", "String"], true),
        "kind": {"Struct": [
            {"name": "ptr", "ty": window_usize(), "attr_info": null},
            {"name": "len", "ty": window_usize(), "attr_info": null},
            {"name": "cap", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_ref = window_opaque_fun(
        1,
        &["alloc", "string", "<Impl>", "as_ref"],
        vec![string_ref.clone()],
        shared.clone(),
    );
    let callee = window_opaque_fun(
        2,
        &["fixture", "window_param"],
        vec![shared.clone()],
        unit.clone(),
    );
    let s_place = window_place(serde_json::json!({"Local": 1}), &string_adt);
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "string_as_ref_window_param"], true),
        "signature": {"is_unsafe": false, "inputs": [string_adt.clone()], "output": unit},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &unit),
                window_local(1, &string_adt),
                window_local(2, &string_ref),
                window_local(3, &shared)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &string_ref),
                            {"Ref": {"place": s_place, "kind": "Shared", "ptr_metadata": null}}
                        ]}
                    }],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &string_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 3}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 3}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &unit)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![string_td], vec![caller, as_ref, callee]);
    let graph = super::lower_function(&llbc, "string_as_ref_window_param")
        .expect("lower String::as_ref into window param");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "as_ref"),
        "heap String::as_ref keeps the residual call; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        !leafs
            .iter()
            .any(|leaf| leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING),
        "heap String::as_ref is not get_raw_address_of_string; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// Keep-gain: `FmtArg::Bytes(qualname.as_bytes())` writes a window
/// into a window field.
#[test]
fn window_keep_gain_fmtarg_bytes_pos0_is_window_write() {
    use crate::model::{OpKind, ValueType, VecFieldPart};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let fmt_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let fmt_arg = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["error", "FmtArg"], true),
        "kind": {"Enum": [{
            "name": "Bytes",
            "fields": [{
                "name": null,
                "ty": shared,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
            }],
            "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
        }]}
    });
    let (unwind, _) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_keep_gain_fmtarg"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref], "output": fmt_adt},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &fmt_adt),
                window_local(1, &utf8_ref),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 2}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &fmt_adt),
                            {"Aggregate": [
                                {"Adt": [1, 0, null]},
                                [{"Move": window_place(serde_json::json!({"Local": 2}), &shared)}]
                            ]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                }
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8, fmt_arg], vec![caller, as_bytes]);
    let graph = super::lower_function(&llbc, "window_keep_gain_fmtarg").expect("lower Keep-gain");
    assert!(
        !window_has_join_chars(&graph),
        "Keep-gain write is not a copy; graph={}",
        graph.dump()
    );
    let ops = window_graph_ops(&graph);
    let data_write = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldWrite { field, ty, .. }
                if field.name == "__pos_0"
                    && field.vec_part == Some(VecFieldPart::FatData)
                    && !matches!(ty, ValueType::Str)
        )
    });
    let len_write = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldWrite { field, ty, .. }
                if field.name == "__pos_0"
                    && field.vec_part == Some(VecFieldPart::FatLen)
                    && !matches!(ty, ValueType::Str)
        )
    });
    let str_or_list_write = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldWrite { field, ty: ValueType::Str | ValueType::Ref(_), .. }
                if field.name == "__pos_0" && field.vec_part.is_none()
        )
    });
    assert!(
        data_write && len_write,
        "Keep-gain __pos_0 is FatData/FatLen; graph={}",
        graph.dump()
    );
    assert!(
        !str_or_list_write,
        "Keep-gain field and value are windows; graph={}",
        graph.dump()
    );
}

/// A STR-source local passed to `fn f(d: &[u8])` borrows a window at
/// the call (`rffi.get_raw_address_of_string`), not at the definition.
#[test]
fn str_view_borrows_window_at_window_param() {
    use crate::model::{CallTarget, OpKind};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let unit = serde_json::json!({"Tuple": []});
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let callee = window_opaque_fun(
        2,
        &["fixture", "window_param"],
        vec![shared.clone()],
        unit.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "str_view_window_param"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref], "output": unit},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &unit),
                window_local(1, &utf8_ref),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 2}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &unit)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller, as_bytes, callee]);
    let graph = super::lower_function(&llbc, "str_view_window_param")
        .expect("lower STR view at window param");
    assert!(
        !window_has_join_chars(&graph),
        "window borrow is not a copy; graph={}",
        graph.dump()
    );
    let at_call = graph.blocks.iter().any(|block| {
        let has_window_ops = block.operations.iter().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => matches!(
                segments.last().map(String::as_str),
                Some("__strlen") | Some(crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING)
            ),
            _ => false,
        });
        let has_callee = block.operations.iter().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => segments.last().map(String::as_str) == Some("window_param") && args.len() == 2,
            _ => false,
        });
        has_window_ops && has_callee
    });
    assert!(
        at_call,
        "emit_str_to_window is at the window-param call; graph={}",
        graph.dump()
    );
    let buf_field = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "buf"
        )
    });
    assert!(
        !buf_field,
        "STR→window must not FieldRead buf; graph={}",
        graph.dump()
    );
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs
            .iter()
            .any(|leaf| { leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING })
            && leafs.iter().any(|leaf| leaf == "__strlen"),
        "STR→window data word is GET_RAW_ADDRESS_OF_STRING plus __strlen; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// `as_bytes().iter()` of a STR view is identity then the
/// `("slice","iter")` bridge (`fun_decl_id: None`). The FunDecl input
/// is generic `&[T]`; keeping that id would annotate the STR as List.
#[test]
fn str_view_iter_borrows_window_at_slice_iter() {
    use crate::model::{CallTarget, OpKind};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let u8_ty = window_u8();
    let generic_slice = serde_json::json!({
        "Ref": ["Erased", {"Slice": [{"TypeVar": {"Bound": [0, 0]}}, null]}, "Shared"]
    });
    let iter_generics = serde_json::json!({
        "regions": [], "types": [u8_ty], "const_generics": [], "trait_refs": []
    });
    let iter_ty = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": iter_generics}
    });
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let iter_td = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["core", "slice", "iter", "Iter"], false),
        "kind": {"Struct": []}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let slice_iter = window_opaque_fun(
        2,
        &["core", "slice", "<Impl>", "iter"],
        vec![generic_slice],
        iter_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "str_view_slice_iter"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref], "output": iter_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &iter_ty),
                window_local(1, &utf8_ref),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 2}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &iter_ty)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8, iter_td], vec![caller, as_bytes, slice_iter]);
    let graph = super::lower_function(&llbc, "str_view_slice_iter")
        .expect("lower STR-view as_bytes().iter()");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs
            .iter()
            .any(|leaf| leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING),
        "as_bytes of a STR view stays identity; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        leafs
            .iter()
            .any(|leaf| leaf == crate::front::join_chars::LL_STR2UINTLIST),
        "slice::iter of a STR view copies through ll_str2uintlist; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let iter_bridge = window_graph_ops(&graph).iter().any(|op| match &op.kind {
        OpKind::Call {
            target:
                CallTarget::FunctionPath {
                    segments,
                    fun_decl_id,
                    ..
                },
            args,
            ..
        } => {
            segments.as_slice() == ["core".to_string(), "slice".to_string(), "iter".to_string()]
                && fun_decl_id.is_none()
                && args.len() == 1
        }
        _ => false,
    });
    assert!(
        iter_bridge,
        "slice::iter of a STR view is the slice-iter bridge; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// `fn f(s: STR) -> Vec<u8> { s.as_bytes().to_vec() }` copies the STR
/// itself through `ll_str2uintlist`.
#[test]
fn str_view_to_vec_passes_str_to_ll_str2uintlist() {
    use crate::model::{CallTarget, LinkArg, OpKind};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let u8_ty = window_u8();
    let vec_generics = serde_json::json!({
        "regions": [], "types": [u8_ty], "const_generics": [], "trait_refs": []
    });
    let vec_ty = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": vec_generics}
    });
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let vec_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["alloc", "vec", "Vec"], false),
        "kind": {"Struct": []}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let to_vec = window_opaque_fun(
        2,
        &["alloc", "slice", "<Impl>", "to_vec"],
        vec![shared.clone()],
        vec_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "str_view_to_vec"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref], "output": vec_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &vec_ty),
                window_local(1, &utf8_ref),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 2}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &vec_ty)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8, vec_decl], vec![caller, as_bytes, to_vec]);
    let graph = super::lower_function(&llbc, "str_view_to_vec")
        .expect("lower STR-view as_bytes().to_vec()");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs
            .iter()
            .any(|leaf| leaf == crate::front::join_chars::LL_STR2UINTLIST),
        "to_vec of a STR view copies through ll_str2uintlist; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let inputs: Vec<_> = window_graph_ops(&graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { .. } => op.result.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(
        inputs.len(),
        1,
        "STR param is one Input word; graph={}",
        graph.dump()
    );
    let ll_block = graph
        .blocks
        .iter()
        .find(|block| {
            block.operations.iter().any(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => {
                    segments.last().map(String::as_str)
                        == Some(crate::front::join_chars::LL_STR2UINTLIST)
                }
                _ => false,
            })
        })
        .expect("ll_str2uintlist is called");
    let ll_args: Vec<_> = ll_block
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(String::as_str)
                == Some(crate::front::join_chars::LL_STR2UINTLIST) =>
            {
                Some(
                    args.iter()
                        .filter_map(LinkArg::as_variable)
                        .cloned()
                        .collect::<Vec<_>>(),
                )
            }
            _ => None,
        })
        .expect("ll_str2uintlist is called");
    // as_bytes is identity: the dest is the Input, or a same_as of it.
    // The to_vec block receives that word as an SSA rename.
    let view_src = {
        let input = inputs[0].clone();
        window_graph_ops(&graph)
            .into_iter()
            .find_map(|op| match &op.kind {
                OpKind::UnaryOp {
                    op: name, operand, ..
                } if name == "same_as" && *operand == input => op.result.clone(),
                _ => None,
            })
            .unwrap_or(input)
    };
    let str_var = graph
        .blocks
        .iter()
        .find_map(|pred| {
            pred.exits.iter().find_map(|link| {
                if link.target != ll_block.id {
                    return None;
                }
                link.args
                    .iter()
                    .zip(&ll_block.inputargs)
                    .find_map(|(arg, iarg)| {
                        (arg.as_variable() == Some(&view_src)).then(|| iarg.clone())
                    })
            })
        })
        .unwrap_or(view_src);
    assert_eq!(
        ll_args,
        vec![str_var],
        "ll_str2uintlist receives the STR; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        !leafs.iter().any(|leaf| leaf == "charpsize2str"),
        "to_vec of a STR view is not charpsize2str; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// `import_head` does `name.as_bytes().iter().position(|&b| b == b'.')`.
/// That is RPython `str.find(".")` (`rpython_str_find_char`), not a
/// residual `Iter::position` FunDecl whose empty GraphCache stub
/// unions every callsite's iterator slot.
#[test]
fn import_head_str_view_position_is_str_find() {
    use crate::model::{CallTarget, OpKind};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-interpreter.ullbc");
    let graph = super::lower_function(&llbc, "pyre_interpreter::importing::import_head")
        .expect("lower import_head");
    let mut has_find = false;
    let mut residual_position = 0usize;
    let mut invented_charpsize = 0usize;
    let mut str2uintlist = 0usize;
    let mut str_const_needle = 0usize;
    let mut raw_const_str = 0usize;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "find" => has_find = true,
            OpKind::ConstStr(_) => raw_const_str += 1,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                if segments.as_slice() == ["slice", "iter", "Iter", "position"]
                    || (segments.last().map(String::as_str) == Some("position")
                        && segments.iter().any(|s| s == "Iter" || s == "iterator"))
                {
                    residual_position += 1;
                }
                if segments.as_slice()
                    == ["rpython", "rtyper", "lltypesystem", "rffi", "charpsize2str"]
                {
                    invented_charpsize += 1;
                }
                if segments.as_slice() == [crate::front::join_chars::LL_STR2UINTLIST] {
                    str2uintlist += 1;
                }
                if segments.len() == 2 && segments[0] == "__str_const" {
                    str_const_needle += 1;
                }
            }
            _ => {}
        }
    }
    assert!(
        has_find,
        "import_head must emit str.find; residual_position={residual_position} graph={}",
        graph.dump()
    );
    assert_eq!(
        str2uintlist,
        0,
        "import_head position is str.find; no ll_str2uintlist copy; graph={}",
        graph.dump()
    );
    assert_eq!(
        residual_position,
        0,
        "import_head must not residualize Iter::position; graph={}",
        graph.dump()
    );
    assert_eq!(
        invented_charpsize,
        0,
        "import_head must not invent rffi.charpsize2str; graph={}",
        graph.dump()
    );
    assert_eq!(
        raw_const_str,
        0,
        "find needle must be __str_const, not ConstStr; graph={}",
        graph.dump()
    );
    assert!(
        str_const_needle >= 1,
        "find needle must be __str_const; graph={}",
        graph.dump()
    );
}

/// B3: `bytes = utf8_payload_bytes(h); s = from_utf8_unchecked(bytes)`
/// stays a STR view: identity, no residual `from_utf8_unchecked`, no
/// STR→window at the definition of `bytes`.
#[test]
fn utf8_payload_bytes_from_utf8_unchecked_is_str_view_identity() {
    use crate::model::{CallTarget, OpKind};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let payload = window_opaque_fun(
        1,
        &["pyre_object", "unicodeobject", "utf8_payload_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let from_utf8 = window_opaque_fun(
        2,
        &["core", "str", "converts", "from_utf8_unchecked"],
        vec![shared.clone()],
        str_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "utf8_payload_from_utf8"], true),
        "signature": {"is_unsafe": true, "inputs": [utf8_ref], "output": str_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &str_ty),
                window_local(1, &utf8_ref),
                window_local(2, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 2}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &str_ty)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller, payload, from_utf8]);
    let graph = super::lower_function(&llbc, "utf8_payload_from_utf8")
        .expect("lower utf8_payload_bytes then from_utf8_unchecked");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| leaf == "from_utf8_unchecked"),
        "from_utf8_unchecked of a STR view is identity; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        !leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING || leaf == "__strlen"
        }),
        "bytes stays a STR view at definition; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let residual_payload = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("utf8_payload_bytes")
        )
    });
    assert!(
        !residual_payload,
        "utf8_payload_bytes is a STR-view alias; graph={}",
        graph.dump()
    );
}

/// B3: `from_utf8_unchecked(&bytes[a..])` of a W1 STR view lowers
/// through the STR getslice (`rstr.py` `LLHelpers.ll_stringslice_startstop`).
#[test]
fn from_utf8_unchecked_of_str_view_rangefrom_is_getslice() {
    use crate::model::{CallTarget, OpKind};

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let n = window_usize();
    let rangefrom_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let rangefrom = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["core", "ops", "range", "RangeFrom"], true),
        "kind": {"Struct": [{
            "name": "start",
            "ty": n,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        }]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let payload = window_opaque_fun(
        1,
        &["pyre_object", "unicodeobject", "utf8_payload_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let generic_slice = serde_json::json!({
        "Ref": ["Erased", {"Slice": [{"TypeVar": {"Bound": [0, 0]}}, null]}, "Shared"]
    });
    let index = window_opaque_fun(
        2,
        &["core", "slice", "index", "<Impl>", "index"],
        vec![generic_slice, rangefrom_adt.clone()],
        shared.clone(),
    );
    let from_utf8 = window_opaque_fun(
        3,
        &["core", "str", "converts", "from_utf8_unchecked"],
        vec![shared.clone()],
        str_ty.clone(),
    );
    let range_place = window_place(serde_json::json!({"Local": 4}), &rangefrom_adt);
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "utf8_payload_getslice_from_utf8"], true),
        "signature": {"is_unsafe": true, "inputs": [utf8_ref, n], "output": str_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &str_ty),
                window_local(1, &utf8_ref),
                window_local(2, &n),
                window_local(3, &shared),
                window_local(4, &rangefrom_adt),
                window_local(5, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 3}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            range_place,
                            {"Aggregate": [
                                {"Adt": [1, 0, null]},
                                [{"Copy": window_place(serde_json::json!({"Local": 2}), &n)}]
                            ]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &shared)},
                            {"Move": window_place(serde_json::json!({"Local": 4}), &rangefrom_adt)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 5}), &shared)
                    }, "target": 3, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 3},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 5}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &str_ty)
                    }, "target": 4, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(
        vec![utf8, rangefrom],
        vec![caller, payload, index, from_utf8],
    );
    let graph = super::lower_function(&llbc, "utf8_payload_getslice_from_utf8")
        .expect("lower from_utf8_unchecked of STR-view RangeFrom");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| leaf == "from_utf8_unchecked"),
        "from_utf8_unchecked of a STR-view subslice is identity; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        leafs.iter().any(|leaf| leaf.starts_with("__getslice_")),
        "STR-view subslice is __getslice_*; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let residual_index = window_graph_ops(&graph).iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("index")
        )
    });
    assert!(
        !residual_index,
        "slice::index of a STR view is getslice; graph={}",
        graph.dump()
    );
}

/// Mixed W1 `w_bytes_data` / window `w_bytearray_data` dest is
/// `bytes_like_data`: the window arm uses the two-word return ABI
/// (`SLICE_LEN_SLOT_NEW` / extra arg / `TAKE`).
#[test]
fn mixed_w1_and_window_return_passes_len_slot() {
    use crate::model::{CallTarget, OpKind};

    let obj_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let obj_ref = serde_json::json!({"Ref": ["Erased", obj_adt, "Shared"]});
    let shared = window_shared_slice();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let pyobj = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "bytesobject", "W_BytesObject"], true),
        "kind": {"Struct": [
            {"name": "data", "ty": shared, "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let w_bytes_data = window_opaque_fun(
        1,
        &["pyre_object", "bytesobject", "w_bytes_data"],
        vec![obj_ref.clone()],
        shared.clone(),
    );
    let w_bytearray_data = window_opaque_fun(
        2,
        &["pyre_object", "bytearrayobject", "w_bytearray_data"],
        vec![obj_ref.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "bytesobject", "bytes_like_data"], true),
        "signature": {"is_unsafe": true, "inputs": [obj_ref, bool_ty], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &shared),
                window_local(1, &obj_ref),
                window_local(2, &bool_ty)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Switch": {
                        "discr": {"Copy": window_place(serde_json::json!({"Local": 2}), &bool_ty)},
                        "targets": {"If": [2, 3]}
                    }}}
                },
                unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &obj_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 4, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &obj_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 4, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![pyobj], vec![caller, w_bytes_data, w_bytearray_data]);
    let graph = super::lower_function(&llbc, "bytes_like_data")
        .expect("lower mixed w_bytes_data / w_bytearray_data");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "ll_slice_len_slot_new")
            && leafs.iter().any(|leaf| leaf == "ll_slice_len_slot_take"),
        "window-return call passes SLICE_LEN_SLOT_NEW / TAKE; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let window_call_has_slot = window_graph_ops(&graph).iter().any(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } => segments.last().map(String::as_str) == Some("w_bytearray_data") && args.len() >= 2,
        _ => false,
    });
    assert!(
        window_call_has_slot,
        "w_bytearray_data takes the length slot argument; graph={}",
        graph.dump()
    );
}

/// `s.as_bytes()[i]` of a `Utf8Str` is `__string_byte_getitem`, not a
/// 1-byte `RawLoad`.
#[test]
fn str_view_index_is_strgetitem() {
    use crate::model::OpKind;

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ref = serde_json::json!({"Ref": ["Erased", utf8_adt, "Shared"]});
    let shared = window_shared_slice();
    let u8_ty = window_u8();
    let n = window_usize();
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![utf8_ref.clone()],
        shared.clone(),
    );
    let generic_slice = serde_json::json!({
        "Ref": ["Erased", {"Slice": [{"TypeVar": {"Bound": [0, 0]}}, null]}, "Shared"]
    });
    let index = window_opaque_fun(
        2,
        &["core", "slice", "index", "<Impl>", "index"],
        vec![generic_slice, n.clone()],
        u8_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "str_view_index"], true),
        "signature": {"is_unsafe": false, "inputs": [utf8_ref, n], "output": u8_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &u8_ty),
                window_local(1, &utf8_ref),
                window_local(2, &n),
                window_local(3, &shared)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &utf8_ref)}],
                        "dest": window_place(serde_json::json!({"Local": 3}), &shared)
                    }, "target": 2, "on_unwind": 1}}}
                },
                unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &shared)},
                            {"Copy": window_place(serde_json::json!({"Local": 2}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &u8_ty)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller, as_bytes, index]);
    let graph = super::lower_function(&llbc, "str_view_index").expect("lower STR-view index");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "__string_byte_getitem"),
        "as_bytes()[i] is string byte getitem; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let raw_load = window_graph_ops(&graph)
        .iter()
        .any(|op| matches!(&op.kind, OpKind::RawLoad { .. }));
    assert!(
        !raw_load,
        "as_bytes()[i] is not a 1-byte RawLoad; graph={}",
        graph.dump()
    );
}

#[test]
fn positional_shapes_derive_distinct_pointer_aware_rows_from_the_spelling() {
    let rows = |shape| super::positional_shape_metadata(shape).unwrap().0;
    assert_eq!(
        rows("Array<i64;1>"),
        vec![crate::front::semantic::FieldRow::positional(
            "__pos_0", "i64"
        )]
    );
    assert_eq!(
        rows("Array<bool;2>"),
        vec![
            crate::front::semantic::FieldRow::positional("__pos_0", "bool"),
            crate::front::semantic::FieldRow::positional("__pos_1", "bool"),
        ]
    );
    assert_eq!(
        rows("Tuple<Union,Union>"),
        vec![
            crate::front::semantic::FieldRow::positional("__pos_0", "&Union"),
            crate::front::semantic::FieldRow::positional("__pos_1", "&Union"),
        ],
        "instance-valued tuple items use the pointer repr recorded by their FORCE attrs",
    );
    assert_eq!(
        super::positional_shape_metadata("Tuple<Union,Union>")
            .unwrap()
            .1,
        vec![
            ("__pos_0".into(), ValueType::Ref(None)),
            ("__pos_1".into(), ValueType::Ref(None))
        ],
    );
    assert_eq!(super::positional_shape_metadata("Tuple<>"), None);
    assert_ne!(
        majit_ir::descr::positional_shape_id("Array<i64;1>"),
        majit_ir::descr::positional_shape_id("Array<bool;2>")
    );
}

/// A `&'static dyn Trait` field is two words. The vtable-slot read of a
/// method call through that field is `getfield` of the metadata word
/// (`ClassRepr.getclsfield`), not of the data word.
#[test]
fn dyn_trait_field_vtable_slot_reads_the_metadata_word() {
    use crate::model::{CallTarget, FieldDescriptor, OpKind, VecFieldPart};

    let span = serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 10}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let empty_generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let holder_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": empty_generics}
    });
    let vtable_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": empty_generics}
    });
    let holder_ref = serde_json::json!({"Ref": ["static", holder_adt, "Shared"]});
    let fat_ty = serde_json::json!({"Ref": ["static", {"DynTrait": {}}, "Shared"]});
    let vtable_ptr = serde_json::json!({"RawPtr": [vtable_adt, "Const"]});
    let fn_ptr =
        serde_json::json!({"FnPtr": {"inputs": [fat_ty.clone()], "output": {"Tuple": []}}});
    let unit = serde_json::json!({"Tuple": []});
    let field_attr = serde_json::json!({
        "attributes": [], "inline": null, "rename": null, "public": true
    });
    let holder = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "Holder"]),
        "kind": {"Struct": [{
            "name": "slot",
            "ty": fat_ty,
            "attr_info": field_attr
        }]}
    });
    let vtable = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "Storage", "{vtable}"]),
        "kind": {"Struct": [{
            "name": "method_head",
            "ty": fn_ptr,
            "attr_info": field_attr
        }]}
    });
    let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
    let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span, "ty": ty});
    let slot_place = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Projection": [
                place(serde_json::json!({"Local": 1}), &holder_ref),
                "Deref"
            ]}), &holder_adt),
            {"Field": [null, 0]}
        ]}),
        &fat_ty,
    );
    let metadata_place = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Local": 2}), &fat_ty),
            "PtrMetadata"
        ]}),
        &vtable_ptr,
    );
    let slot_field = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Projection": [
                metadata_place,
                "Deref"
            ]}), &vtable_adt),
            {"Field": [null, 0]}
        ]}),
        &fn_ptr,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "call_head"]),
        "signature": {"is_unsafe": false, "inputs": [holder_ref], "output": unit},
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": [
                local(0, &unit),
                local(1, &holder_ref),
                local(2, &fat_ty),
                local(3, &fn_ptr)
            ]},
            "body": [
                {
                    "statements": [
                        {"span": span, "kind": {"Assign": [
                            place(serde_json::json!({"Local": 2}), &fat_ty),
                            {"Use": [{"Copy": slot_place}, "No"]}
                        ]}},
                        {"span": span, "kind": {"Assign": [
                            place(serde_json::json!({"Local": 3}), &fn_ptr),
                            {"UnaryOp": [
                                {"Cast": {"RawPtr": fn_ptr.clone()}},
                                {"Copy": slot_field}
                            ]}
                        ]}}
                    ],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": {"Call": {
                        "call": {
                            "func": {"Dynamic": [{"Copy": place(
                                serde_json::json!({"Local": 3}),
                                &fn_ptr
                            )}, {"regions": []}]},
                            "args": [{"Move": place(
                                serde_json::json!({"Local": 2}),
                                &fat_ty
                            )}],
                            "dest": place(serde_json::json!({"Local": 0}), &unit)
                        },
                        "target": 1,
                        "on_unwind": 2
                    }}}
                },
                {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
            ]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [holder, vtable],
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["fixture", "Storage"]),
                "methods": [{"skip_binder": {"name": "head"}}]
            }],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = super::lower_function(&llbc, "call_head").expect("lower call_head");

    let indirect: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::IndirectCall {
                funcptr,
                family_key: Some(family),
                ..
            } => Some((funcptr.clone(), family.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        indirect.len(),
        1,
        "one dyn method IndirectCall: {indirect:?}"
    );
    assert_eq!(indirect[0].1, ("Storage".to_string(), "head".to_string()));

    let producer = |var: &crate::flowspace::model::Variable| {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|op| op.result.as_ref() == Some(var))
            .map(|op| &op.kind)
    };
    let funcptr = &indirect[0].0;
    let slot_read = producer(funcptr).expect("vtable slot producer");
    let OpKind::FieldRead {
        base: slot_base,
        field: FieldDescriptor {
            name: slot_name, ..
        },
        ..
    } = slot_read
    else {
        panic!("vtable slot must be a FieldRead, got {slot_read:?}");
    };
    assert_eq!(slot_name, "method_head");

    let mut base = slot_base.clone();
    if let Some(OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    }) = producer(&base)
        && segments
            .last()
            .is_some_and(|s| s == "__cast_instance_intrinsic")
    {
        base = args[0].as_variable().expect("cast receiver").clone();
    }
    let meta = producer(&base).expect("metadata-word producer");
    let OpKind::FieldRead { field, .. } = meta else {
        panic!("metadata word must be a FieldRead, got {meta:?}");
    };
    assert_eq!(field.name, "slot");
    assert_eq!(
        field.vec_part,
        Some(VecFieldPart::FatMeta),
        "vtable slot base is the metadata word, not the data word: {meta:?}"
    );
}

/// `w_dict_getitem_str_hashed` reads `imp` on two strategy arms, merges,
/// then reads `method_*` after a passthrough. The FatMeta metadata word
/// must be an inputarg of the method_* block: a dominating use of the
/// merge phi is not a definition (`lookup_operand`).
#[test]
fn dyn_trait_vtable_slot_base_is_defined_after_a_passthrough() {
    use crate::model::{FieldDescriptor, OpKind, VecFieldPart};

    let span = serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 10}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let empty_generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let holder_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": empty_generics}
    });
    let vtable_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": empty_generics}
    });
    let holder_ref = serde_json::json!({"Ref": ["static", holder_adt, "Shared"]});
    let fat_ty = serde_json::json!({"Ref": ["static", {"DynTrait": {}}, "Shared"]});
    let vtable_ptr = serde_json::json!({"RawPtr": [vtable_adt, "Const"]});
    let fn_ptr =
        serde_json::json!({"FnPtr": {"inputs": [fat_ty.clone()], "output": {"Tuple": []}}});
    let unit = serde_json::json!({"Tuple": []});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let field_attr = serde_json::json!({
        "attributes": [], "inline": null, "rename": null, "public": true
    });
    let holder = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "Holder"]),
        "kind": {"Struct": [{
            "name": "slot",
            "ty": fat_ty,
            "attr_info": field_attr
        }]}
    });
    let vtable = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "Storage", "{vtable}"]),
        "kind": {"Struct": [{
            "name": "method_head",
            "ty": fn_ptr,
            "attr_info": field_attr
        }]}
    });
    let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
    let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span, "ty": ty});
    let slot_place = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Projection": [
                place(serde_json::json!({"Local": 1}), &holder_ref),
                "Deref"
            ]}), &holder_adt),
            {"Field": [null, 0]}
        ]}),
        &fat_ty,
    );
    let assign_slot = serde_json::json!({"span": span, "kind": {"Assign": [
        place(serde_json::json!({"Local": 3}), &fat_ty),
        {"Use": [{"Copy": slot_place}, "No"]}
    ]}});
    let metadata_place = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Local": 3}), &fat_ty),
            "PtrMetadata"
        ]}),
        &vtable_ptr,
    );
    let slot_field = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Projection": [
                metadata_place,
                "Deref"
            ]}), &vtable_adt),
            {"Field": [null, 0]}
        ]}),
        &fn_ptr,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "call_head_merged"]),
        "signature": {
            "is_unsafe": false,
            "inputs": [holder_ref, bool_ty],
            "output": unit
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 2, "locals": [
                local(0, &unit),
                local(1, &holder_ref),
                local(2, &bool_ty),
                local(3, &fat_ty),
                local(4, &fn_ptr)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": {"Switch": {
                        "discr": {"Copy": place(serde_json::json!({"Local": 2}), &bool_ty)},
                        "targets": {"If": [1, 2]}
                    }}}
                },
                {
                    "statements": [assign_slot.clone()],
                    "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}
                },
                {
                    "statements": [assign_slot],
                    "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": {"Goto": {"target": 4}}}
                },
                {
                    "statements": [{"span": span, "kind": {"Assign": [
                        place(serde_json::json!({"Local": 4}), &fn_ptr),
                        {"UnaryOp": [
                            {"Cast": {"RawPtr": [fn_ptr.clone(), fn_ptr.clone()]}},
                            {"Copy": slot_field}
                        ]}
                    ]}}],
                    "terminator": {"span": span, "kind": {"Call": {
                        "call": {
                            "func": {"Dynamic": [{"Copy": place(
                                serde_json::json!({"Local": 4}),
                                &fn_ptr
                            )}, {"regions": []}]},
                            "args": [{"Move": place(
                                serde_json::json!({"Local": 3}),
                                &fat_ty
                            )}],
                            "dest": place(serde_json::json!({"Local": 0}), &unit)
                        },
                        "target": 5,
                        "on_unwind": 6
                    }}}
                },
                {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
            ]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [holder, vtable],
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["fixture", "Storage"]),
                "methods": [{"skip_binder": {"name": "head"}}]
            }],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = super::lower_function(&llbc, "call_head_merged").expect("lower call_head_merged");

    let mut method_reads = 0usize;
    for block in &graph.blocks {
        let mut defined: std::collections::HashSet<u64> =
            block.inputargs.iter().map(|var| var.id()).collect();
        for op in &block.operations {
            for operand in crate::inline::op_variable_refs(&op.kind) {
                assert!(
                    defined.contains(&operand.id()),
                    "block {:?} uses undefined operand {:?} in {:?}",
                    block.id,
                    operand,
                    op.kind
                );
            }
            if let OpKind::FieldRead {
                field: FieldDescriptor { name, .. },
                ..
            } = &op.kind
                && name == "method_head"
            {
                method_reads += 1;
            }
            if let Some(result) = &op.result {
                defined.insert(result.id());
            }
        }
    }
    assert!(
        method_reads > 0,
        "lowered no method_head read\n{}",
        graph.dump()
    );
    let fat_lens: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "slot" && field.vec_part == Some(VecFieldPart::FatMeta)
            )
        })
        .collect();
    assert!(
        !fat_lens.is_empty(),
        "the vtable word must be a FatMeta of slot\n{}",
        graph.dump()
    );
}

/// Two one-call graphs sharing a synthetic `core::intrinsics::transmute`
/// declaration.  The first has the portal's scalar-to-fieldless-enum
/// shape; the second is an equal-size register-bank crossing.
fn transmute_lowering_fixture() -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
                .collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let u64_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let f64_ty = serde_json::json!({"Scalar": {"Float": "F64"}});
    let raw_ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]
    });
    let fnptr_ty = serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "is_unsafe": true,
                "inputs": [],
                "output": raw_ptr_ty.clone()
            }
        }
    });
    let instruction_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let caller = |def_id: u64, name: &str, src_ty: serde_json::Value, dst_ty: serde_json::Value| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta(&["fixture", name], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [src_ty.clone()],
                "output": dst_ty.clone()
            },
            "body": {
                "Unstructured": {
                    "span": span(),
                    "locals": {
                        "arg_count": 1,
                        "locals": [
                            {"index": 0, "name": null, "span": span(), "ty": dst_ty.clone()},
                            {"index": 1, "name": "value", "span": span(), "ty": src_ty.clone()}
                        ]
                    },
                    "body": [
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {
                                "span": span(),
                                "kind": {
                                    "Call": {
                                        "call": {
                                            "func": {
                                                "Regular": {
                                                    "kind": {"Fun": 2},
                                                    "generics": {
                                                        "regions": [],
                                                        "types": [src_ty.clone(), dst_ty.clone()],
                                                        "const_generics": [],
                                                        "trait_refs": []
                                                    }
                                                }
                                            },
                                            "args": [{
                                                "Copy": {"kind": {"Local": 1}, "ty": src_ty.clone()}
                                            }],
                                            "dest": {"kind": {"Local": 0}, "ty": dst_ty.clone()}
                                        },
                                        "target": 2,
                                        "on_unwind": 1
                                    }
                                }
                            }
                        },
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span(), "kind": "UnwindResume"}
                        },
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span(), "kind": "Return"}
                        }
                    ]
                }
            }
        })
    };
    let instruction = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "Instruction"], true),
        "kind": {
            "Enum": [
                {"name": "LoadConst", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}},
                {"name": "ReturnValue", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}}
            ]
        },
        "layout": [{
            "key": "fixture-target",
            "value": {
                "size": 1,
                "align": 1,
                "variant_layouts": [
                    {"field_offsets": []},
                    {"field_offsets": []}
                ],
                "repr": {"transparent": false}
            }
        }]
    });
    let transmute = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["core", "intrinsics", "transmute"], false),
        "signature": {
            "is_unsafe": true,
            "inputs": [{"TypeVar": {"Free": 0}}],
            "output": {"TypeVar": {"Free": 1}}
        },
        "body": {"Intrinsic": {"name": "transmute", "arg_names": ["src"]}}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [instruction],
            "fun_decls": [
                caller(0, "transmute_u8_to_instruction", u8_ty, instruction_ty),
                caller(1, "transmute_u64_to_f64", u64_ty, f64_ty),
                transmute,
                caller(3, "transmute_rawptr_to_fnptr", raw_ptr_ty, fnptr_ty)
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("transmute fixture Llbc parses")
}

/// Foreign inherent `Range::contains` whose FunDecl is Opaque while
/// the TypeDecl is a generic Struct (`TypeVar` fields, so it is not
/// raw storage). Mirrors `core::ops::range::Range::contains`.
pub(in crate::front::mir) fn range_contains_fixture() -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let item_meta = |name: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": name,
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let typevar = serde_json::json!({"TypeVar": {"Free": 0}});
    let range_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics.clone()}
    });
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let field_attr = serde_json::json!({
        "attributes": [], "inline": null, "rename": null, "public": true
    });
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, range_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let range = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(vec![
            ident("core"), ident("ops"), ident("range"), ident("Range")
        ], false),
        "kind": {"Struct": [
            {"name": "start", "ty": typevar.clone(), "attr_info": field_attr},
            {"name": "end", "ty": typevar, "attr_info": field_attr}
        ]}
    });
    let contains = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(
            vec![
                ident("core"), ident("ops"), ident("range"), impl_seg, ident("contains")
            ],
            false,
        ),
        "signature": {
            "is_unsafe": false,
            "inputs": [range_ty.clone(), i64_ty.clone()],
            "output": bool_ty.clone()
        },
        "body": "Opaque"
    });
    let caller = {
        let locals = vec![
            serde_json::json!({
                "index": 0,
                "name": null,
                "span": span(),
                "ty": bool_ty.clone()
            }),
            serde_json::json!({
                "index": 1,
                "name": "range",
                "span": span(),
                "ty": range_ty.clone()
            }),
            serde_json::json!({
                "index": 2,
                "name": "item",
                "span": span(),
                "ty": i64_ty.clone()
            }),
        ];
        serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(vec![ident("fixture"), ident("range_contains")], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [range_ty.clone(), i64_ty.clone()],
                "output": bool_ty.clone()
            },
            "body": {
                "Unstructured": {
                    "span": span(),
                    "locals": {
                        "arg_count": 2,
                        "locals": locals
                    },
                    "body": [
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {
                                "span": span(),
                                "kind": {
                                    "Call": {
                                        "call": {
                                            "func": {
                                                "Regular": {
                                                    "kind": {"Fun": 1},
                                                    "generics": generics.clone()
                                                }
                                            },
                                            "args": [
                                                {"Copy": {"kind": {"Local": 1}, "ty": range_ty.clone()}},
                                                {"Copy": {"kind": {"Local": 2}, "ty": i64_ty.clone()}}
                                            ],
                                            "dest": {"kind": {"Local": 0}, "ty": bool_ty.clone()}
                                        },
                                        "target": 2,
                                        "on_unwind": 1
                                    }
                                }
                            }
                        },
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span(), "kind": "UnwindResume"}
                        },
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span(), "kind": "Return"}
                        }
                    ]
                }
            }
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [range],
            "fun_decls": [caller, contains],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("Range::contains fixture Llbc parses")
}

#[test]
fn same_size_int_bank_transmute_lowers_to_same_as() {
    let llbc = transmute_lowering_fixture();
    let graph = super::lower_function(&llbc, "transmute_u8_to_instruction")
        .expect("lower u8-to-Instruction transmute");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();

    assert!(!ops.iter().any(|op| matches!(op.kind, OpKind::Call { .. })));
    assert!(
        ops.iter()
            .all(|op| !matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as")),
        "same-bank transmute binds the destination to the input"
    );
    let input = ops
        .iter()
        .find_map(|op| match (&op.result, &op.kind) {
            (Some(result), OpKind::Input { name, .. }) if name == "value" => Some(result),
            _ => None,
        })
        .expect("value input");
    let returned = graph
        .blocks
        .iter()
        .find_map(|block| {
            block
                .exits
                .iter()
                .find_map(|link| (link.target == graph.returnblock).then(|| link.args.first()))
        })
        .flatten()
        .and_then(LinkArg::as_variable);
    assert_eq!(returned, Some(input));
}

fn scalar_method_call_fixture(
    caller_name: &str,
    callee_name: &[&str],
    src_ty: serde_json::Value,
    dst_ty: serde_json::Value,
) -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| {
                    if *segment == "<Impl>" {
                        serde_json::json!({"Impl": {"kind": "InherentImplBlock"}})
                    } else {
                        serde_json::json!({"Ident": [segment, 0]})
                    }
                })
                .collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", caller_name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [src_ty.clone()],
            "output": dst_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span(), "ty": dst_ty.clone()},
                        {"index": 1, "name": "value", "span": span(), "ty": src_ty.clone()}
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 1},
                                                "generics": {
                                                    "regions": [],
                                                    "types": [src_ty.clone(), dst_ty.clone()],
                                                    "const_generics": [],
                                                    "trait_refs": []
                                                }
                                            }
                                        },
                                        "args": [{
                                            "Copy": {"kind": {"Local": 1}, "ty": src_ty.clone()}
                                        }],
                                        "dest": {"kind": {"Local": 0}, "ty": dst_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(callee_name, false),
        "signature": {
            "is_unsafe": true,
            "inputs": [src_ty.clone()],
            "output": dst_ty.clone()
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [caller, callee],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("scalar method fixture Llbc parses")
}

fn std_extern_call_fixture(
    caller_name: &str,
    callee_name: &[&str],
    arg_tys: &[serde_json::Value],
    dest_ty: serde_json::Value,
) -> Llbc {
    std_extern_call_fixture_with_types(caller_name, callee_name, arg_tys, dest_ty, vec![])
}

pub(in crate::front::mir) fn std_extern_call_fixture_with_types(
    caller_name: &str,
    callee_name: &[&str],
    arg_tys: &[serde_json::Value],
    dest_ty: serde_json::Value,
    type_decls: Vec<serde_json::Value>,
) -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| {
                    if *segment == "<Impl>" {
                        serde_json::json!({"Impl": {"kind": "InherentImplBlock"}})
                    } else {
                        serde_json::json!({"Ident": [segment, 0]})
                    }
                })
                .collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let mut locals = vec![serde_json::json!({
        "index": 0,
        "name": null,
        "span": span(),
        "ty": dest_ty.clone()
    })];
    for (i, ty) in arg_tys.iter().enumerate() {
        locals.push(serde_json::json!({
            "index": i + 1,
            "name": format!("arg{i}"),
            "span": span(),
            "ty": ty
        }));
    }
    let call_args: Vec<serde_json::Value> = arg_tys
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            serde_json::json!({
                "Copy": {"kind": {"Local": i + 1}, "ty": ty}
            })
        })
        .collect();
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", caller_name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": arg_tys.to_vec(),
            "output": dest_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": arg_tys.len(),
                    "locals": locals
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 1},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": call_args,
                                        "dest": {"kind": {"Local": 0}, "ty": dest_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(callee_name, false),
        "signature": {
            "is_unsafe": true,
            "inputs": arg_tys.to_vec(),
            "output": dest_ty.clone()
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": [caller, callee],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("std extern fixture Llbc parses")
}

/// Trait `Clone::clone` of `self_ty` (`&Self` argument, owned dest).
pub(in crate::front::mir) fn trait_clone_call_fixture(
    caller_name: &str,
    self_ty: serde_json::Value,
    type_decls: Vec<serde_json::Value>,
) -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
                .collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let recv_ty = serde_json::json!({"Ref": ["_", self_ty.clone(), "Shared"]});
    let locals = vec![
        serde_json::json!({
            "index": 0,
            "name": null,
            "span": span(),
            "ty": self_ty.clone()
        }),
        serde_json::json!({
            "index": 1,
            "name": "arg0",
            "span": span(),
            "ty": recv_ty.clone()
        }),
    ];
    let trait_ref = serde_json::json!({
        "kind": {"BuiltinOrAuto": {"builtin_data": "Clone"}},
        "trait_decl_ref": {
            "skip_binder": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [self_ty.clone()],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", caller_name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [recv_ty.clone()],
            "output": self_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 1,
                    "locals": locals
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Trait": [trait_ref, 0]},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [{
                                            "Copy": {"kind": {"Local": 1}, "ty": recv_ty.clone()}
                                        }],
                                        "dest": {"kind": {"Local": 0}, "ty": self_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let clone_trait = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["core", "clone", "Clone"], false),
        "methods": [{"skip_binder": {"name": "clone"}}]
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [clone_trait],
            "assoc_item_names": [{"types": [], "methods": ["clone"], "consts": []}],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("trait clone fixture Llbc parses")
}

pub(in crate::front::mir) fn call_leafs(ops: &[&SpaceOperation]) -> Vec<String> {
    ops.iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            _ => None,
        })
        .collect()
}

#[test]
fn icf_identity_token_lowers_without_a_call_op() {
    // Keep the unit dest live the way `mem_forget_returns_a_defined_unit_value`
    // does: a later consume reads it, so `transform_dead_op_vars` cannot
    // drop `emit_unit`'s ConstNone. The call's Regular generics carry a
    // const generic so the body is `icf_identity_token::<N>()`.
    let span = serde_json::json!({"data": {"file_id": 0,
            "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics =
        serde_json::json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let token_generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [[
            {"Integer": {"Unsigned": ["U64", "1"]}},
            {"Scalar": {"Integer": {"Unsigned": "U64"}}}
        ]],
        "trait_refs": []
    });
    let unit = serde_json::json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics}});
    let word = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let place = |id, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
    let file = serde_json::json!({
        "charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture", "type_decls": [],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "call_icf_identity_token"]),
                 "signature": {"is_unsafe": false, "inputs": [word], "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "value", "span": span, "ty": word},
                        {"index": 2, "name": "unit", "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": token_generics}},
                                "args": [], "dest": place(2, &unit)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Copy": place(2, &unit)}], "dest": place(0, &word)},
                            "target": 3, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["majit_ir", "icf", "icf_identity_token"]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit}, "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(&["fixture", "consume_unit"]),
                 "signature": {"is_unsafe": false, "inputs": [unit], "output": word}, "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("unit fixture parses");
    let graph =
        super::lower_function(&llbc, "call_icf_identity_token").expect("lower icf_identity_token");
    let ops = graph_ops(&graph);
    assert!(
        !call_leafs(&ops)
            .iter()
            .any(|leaf| *leaf == "icf_identity_token"),
        "icf_identity_token::<N>() must lower without a call op; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(op.kind, OpKind::ConstNone)),
        "the unit destination is emit_unit's ConstNone; ops={ops:?}"
    );
}

#[test]
fn copy_scalar_clone_is_the_value_itself() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let recv = serde_json::json!({"Ref": ["_", i64_ty.clone(), "Shared"]});
    let llbc = std_extern_call_fixture(
        "clone_i64",
        &["core", "clone", "impls", "<Impl>", "clone"],
        &[recv],
        i64_ty,
    );
    let graph = super::lower_function(&llbc, "clone_i64").expect("lower Copy clone");
    let ops = graph_ops(&graph);
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "clone"),
        "Copy clone must not residualize; ops={ops:?}"
    );
}

#[test]
fn non_copy_clone_lookalike_stays_residual() {
    let adt = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let recv = serde_json::json!({"Ref": ["_", adt.clone(), "Shared"]});
    let llbc = std_extern_call_fixture(
        "clone_vec",
        &["core", "clone", "impls", "<Impl>", "clone"],
        &[recv],
        adt,
    );
    let graph = super::lower_function(&llbc, "clone_vec").expect("lower non-Copy clone");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "clone"),
        "a non-Copy clone must stay residual; ops={ops:?}"
    );
}

#[test]
fn clone_lookalike_path_stays_residual() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let recv = serde_json::json!({"Ref": ["_", i64_ty.clone(), "Shared"]});
    let llbc = std_extern_call_fixture(
        "clone_blanket",
        &["core", "clone", "<Impl>", "clone"],
        &[recv],
        i64_ty,
    );
    let graph = super::lower_function(&llbc, "clone_blanket").expect("lower lookalike clone");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "clone"),
        "core::clone::<Impl>::clone must not use the impls identity; ops={ops:?}"
    );
}

pub(in crate::front::mir) fn named_adt(
    id: u64,
    types: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": id,
            "generics": {
                "regions": [],
                "types": types,
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

pub(in crate::front::mir) fn type_decl_struct(
    id: u64,
    path: &[&str],
    fields: Vec<serde_json::Value>,
) -> serde_json::Value {
    let span = serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    });
    serde_json::json!({
        "def_id": id,
        "item_meta": {
            "name": path
                .iter()
                .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
                .collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": false
        },
        "kind": {"Struct": fields}
    })
}

#[test]
fn trait_clone_of_build_hasher_default_is_identity() {
    let hasher = named_adt(0, vec![named_adt(1, vec![])]);
    let llbc = trait_clone_call_fixture(
        "clone_hasher",
        hasher,
        vec![
            type_decl_struct(0, &["core", "hash", "BuildHasherDefault"], vec![]),
            type_decl_struct(1, &["fixture", "Fnv1aHasher"], vec![]),
        ],
    );
    let graph =
        super::lower_function(&llbc, "clone_hasher").expect("lower BuildHasherDefault Clone");
    let ops = graph_ops(&graph);
    let leaves = call_leafs(&ops);
    assert!(
        !leaves.iter().any(|leaf| *leaf == "clone"),
        "Copy ZST Clone::clone must be identity; ops={ops:?}"
    );
    assert!(
        !leaves.iter().any(|leaf| *leaf == "list"),
        "BuildHasherDefault clone must not become list; ops={ops:?}"
    );
}

pub(in crate::front::mir) fn i64_ty() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})
}

pub(in crate::front::mir) fn vec_of(elem: serde_json::Value) -> serde_json::Value {
    named_adt(0, vec![elem])
}

fn string_ty() -> serde_json::Value {
    named_adt(1, vec![])
}

pub(in crate::front::mir) fn vec_type_decls() -> Vec<serde_json::Value> {
    vec![type_decl_struct(0, &["alloc", "vec", "Vec"], vec![])]
}

fn vec_string_type_decls() -> Vec<serde_json::Value> {
    let ptr = serde_json::json!({
        "name": "ptr",
        "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
        "attr_info": {
            "attributes": [],
            "inline": null,
            "rename": null,
            "public": true
        }
    });
    vec![
        type_decl_struct(0, &["alloc", "vec", "Vec"], vec![]),
        type_decl_struct(1, &["alloc", "string", "String"], vec![ptr]),
    ]
}

pub(in crate::front::mir) fn trait_clone_kind_payload(
    self_ty: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!([
        {
            "kind": {"BuiltinOrAuto": {"builtin_data": "Clone"}},
            "trait_decl_ref": {
                "skip_binder": {
                    "id": 0,
                    "generics": {
                        "regions": [],
                        "types": [self_ty],
                        "const_generics": [],
                        "trait_refs": []
                    }
                }
            }
        },
        0
    ])
}

fn tuple_ty(elems: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": elems,
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

fn assert_trait_clone_stays_residual(
    caller: &str,
    self_ty: serde_json::Value,
    decls: Vec<serde_json::Value>,
) {
    let llbc = trait_clone_call_fixture(caller, self_ty.clone(), decls);
    let payload = trait_clone_kind_payload(self_ty);
    assert!(
        super::specialized_trait_clone_list_target(&payload, &llbc).is_none(),
        "CallKind::Trait retarget must not intercept a non-identity-element Vec"
    );
    let graph = super::lower_function(&llbc, caller).expect("lower non-identity Vec Clone");
    let ops = graph_ops(&graph);
    let leaves = call_leafs(&ops);
    assert!(
        leaves.iter().any(|leaf| *leaf == "clone"),
        "non-identity-element Vec Clone::clone must stay residual; ops={ops:?}"
    );
    assert!(
        !leaves.iter().any(|leaf| *leaf == "list"),
        "non-identity-element Vec Clone::clone must not become list; ops={ops:?}"
    );
}

#[test]
fn trait_clone_of_vec_of_vec_is_not_list() {
    assert_trait_clone_stays_residual(
        "clone_vec_of_vec",
        vec_of(vec_of(i64_ty())),
        vec_type_decls(),
    );
}

#[test]
fn trait_clone_of_vec_of_string_is_not_list() {
    assert_trait_clone_stays_residual(
        "clone_vec_of_string",
        vec_of(string_ty()),
        vec_string_type_decls(),
    );
}

#[test]
fn trait_clone_of_nested_vec_tuple_is_not_list() {
    let self_ty = tuple_ty(vec![vec_of(vec_of(i64_ty())), i64_ty()]);
    assert_trait_clone_stays_residual("clone_nested_vec_tuple", self_ty, vec_type_decls());
}

#[test]
fn trait_clone_of_vec_pair_is_two_list_copies_and_tuple() {
    let vec_ty = vec_of(i64_ty());
    let tuple = tuple_ty(vec![vec_ty.clone(), vec_ty]);
    let llbc = trait_clone_call_fixture("clone_vec_pair", tuple, vec_type_decls());
    let graph = super::lower_function(&llbc, "clone_vec_pair").expect("lower (Vec, Vec) Clone");
    let ops = graph_ops(&graph);
    let list_calls = call_leafs(&ops)
        .iter()
        .filter(|leaf| *leaf == "list")
        .count();
    assert_eq!(
        list_calls, 2,
        "(Vec, Vec) Clone::clone must emit two list copies; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "clone"),
        "(Vec, Vec) Clone::clone must not residualize; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "__pos_0"
        )),
        "(Vec, Vec) clone must read field 0; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "__pos_1"
        )),
        "(Vec, Vec) clone must read field 1; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            }
        )),
        "(Vec, Vec) clone must construct a tuple; ops={ops:?}"
    );
}

#[test]
fn trait_clone_of_user_struct_stays_residual() {
    let user = named_adt(0, vec![]);
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let field = serde_json::json!({
        "name": "payload",
        "ty": i64_ty,
        "attr_info": {
            "attributes": [],
            "inline": null,
            "rename": null,
            "public": true
        }
    });
    let llbc = trait_clone_call_fixture(
        "clone_user",
        user,
        vec![type_decl_struct(0, &["fixture", "UserClone"], vec![field])],
    );
    let graph = super::lower_function(&llbc, "clone_user").expect("lower user Clone");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "clone"),
        "a non-Copy user Clone must stay residual; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "list"),
        "a user Clone must not become list; ops={ops:?}"
    );
}

#[test]
fn scalar_default_is_the_zero_constant() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = std_extern_call_fixture(
        "default_i64",
        &["core", "default", "<Impl>", "default"],
        &[],
        i64_ty,
    );
    let graph = super::lower_function(&llbc, "default_i64").expect("lower scalar default");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(op.kind, OpKind::ConstInt(0))),
        "i64::default must become ConstInt(0); ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "scalar default must not residualize; ops={ops:?}"
    );
}

#[test]
fn pointer_default_is_a_null_constant() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    });
    let llbc = std_extern_call_fixture(
        "default_mut_ptr",
        &["core", "ptr", "mut_ptr", "<Impl>", "default"],
        &[],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "default_mut_ptr").expect("lower ptr default");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(op.kind, OpKind::ConstRefNull)),
        "*mut T::default must become ConstRefNull; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "pointer default must not residualize; ops={ops:?}"
    );
}

#[test]
fn default_lookalike_path_stays_residual() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = std_extern_call_fixture(
        "default_trait",
        &["core", "default", "Default", "default"],
        &[],
        i64_ty,
    );
    let graph = super::lower_function(&llbc, "default_trait").expect("lower lookalike default");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "Default::default must stay residual; ops={ops:?}"
    );
}

#[test]
fn fixture_fn_default_returning_unit_keeps_the_call() {
    let unit = serde_json::json!({"Adt": {
        "id": 0,
        "builtin": "Tuple",
        "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
    }});
    let llbc = std_extern_call_fixture("call_default", &["fixture", "default"], &[], unit);
    let graph =
        super::lower_function(&llbc, "call_default").expect("lower fixture fn default() -> ()");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "a fixture fn default() -> () keeps its call op; ops={ops:?}"
    );
}

fn std_trait_call_fixture(
    caller_name: &str,
    trait_path: &[&str],
    method: &str,
    arg_tys: &[serde_json::Value],
    dest_ty: serde_json::Value,
    type_decls: Vec<serde_json::Value>,
) -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
                .collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let mut locals = vec![serde_json::json!({
        "index": 0,
        "name": null,
        "span": span(),
        "ty": dest_ty.clone()
    })];
    for (i, ty) in arg_tys.iter().enumerate() {
        locals.push(serde_json::json!({
            "index": i + 1,
            "name": format!("arg{i}"),
            "span": span(),
            "ty": ty
        }));
    }
    let call_args: Vec<serde_json::Value> = arg_tys
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            serde_json::json!({
                "Copy": {"kind": {"Local": i + 1}, "ty": ty}
            })
        })
        .collect();
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", caller_name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": arg_tys.to_vec(),
            "output": dest_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": arg_tys.len(),
                    "locals": locals
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Trait": [
                                                    {
                                                        "kind": "Unknown",
                                                        "trait_decl_ref": {
                                                            "skip_binder": {"id": 0}
                                                        }
                                                    },
                                                    0
                                                ]},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": call_args,
                                        "dest": {"kind": {"Local": 0}, "ty": dest_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let trait_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(trait_path, false),
        "methods": []
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [trait_decl],
            "trait_impls": [],
            "assoc_item_names": [{
                "types": [],
                "methods": [method],
                "consts": []
            }]
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("std trait fixture Llbc parses")
}

#[test]
fn trait_as_ref_str_of_string_is_identity() {
    let string_adt = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let recv = serde_json::json!({"Ref": ["Erased", string_adt, "Shared"]});
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let string_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["alloc", "string", "String"], false),
        "kind": {"Struct": [
            {"name": "ptr", "ty": window_usize(), "attr_info": null},
            {"name": "len", "ty": window_usize(), "attr_info": null},
            {"name": "cap", "ty": window_usize(), "attr_info": null}
        ]}
    });
    let llbc = std_trait_call_fixture(
        "string_as_ref_str",
        &["core", "convert", "AsRef"],
        "as_ref",
        &[recv],
        str_ty,
        vec![string_decl],
    );
    let graph = super::lower_function(&llbc, "string_as_ref_str")
        .expect("lower <String as AsRef<str>>::as_ref");
    let ops = graph_ops(&graph);
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "as_ref"),
        "Trait AsRef::as_ref of a string-family receiver to &str is identity; ops={ops:?} graph={}",
        graph.dump()
    );
}

#[test]
fn trait_default_of_i64_is_zero() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = std_trait_call_fixture(
        "trait_default_i64",
        &["core", "default", "Default"],
        "default",
        &[],
        i64_ty,
        vec![],
    );
    let graph = super::lower_function(&llbc, "trait_default_i64").expect("lower Trait Default i64");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(op.kind, OpKind::ConstInt(0))),
        "<i64 as Default>::default must become ConstInt(0); ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "Trait Default of i64 must not residualize; ops={ops:?}"
    );
}

#[test]
fn user_trait_named_default_is_not_integer_zero() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = std_trait_call_fixture(
        "user_trait_default_i64",
        &["pyre_test", "Default"],
        "default",
        &[],
        i64_ty,
        vec![],
    );
    let graph =
        super::lower_function(&llbc, "user_trait_default_i64").expect("lower user-defined Default");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter().any(|op| matches!(op.kind, OpKind::ConstInt(0))),
        "a user-defined Default must not become integer zero; ops={ops:?}"
    );
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "a user-defined Default stays a residual trait call; ops={ops:?}"
    );
}

#[test]
fn trait_default_of_wtf8buf_is_empty_str() {
    let wtf8buf = serde_json::json!({"Adt": {
        "id": 0,
        "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
    }});
    let llbc = std_trait_call_fixture(
        "trait_default_wtf8buf",
        &["core", "default", "Default"],
        "default",
        &[],
        wtf8buf,
        vec![code_struct(0, &["rustpython_wtf8", "Wtf8Buf"])],
    );
    let graph =
        super::lower_function(&llbc, "trait_default_wtf8buf").expect("lower Trait Default Wtf8Buf");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.as_slice() == ["__str_const", ""]
        )),
        "<Wtf8Buf as Default>::default must become __str_const \"\"; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "default"),
        "Trait Default of Wtf8Buf must not residualize; ops={ops:?}"
    );
}

fn fn_item_call_once_fixture(
    callee_name: &[&str],
    n_inputs: usize,
    tuple_elem_tys: &[serde_json::Value],
    dest_ty: serde_json::Value,
) -> Llbc {
    let empty = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    fn_item_call_once_fixture_with(
        callee_name,
        n_inputs,
        tuple_elem_tys,
        dest_ty,
        empty,
        None,
        false,
        &["core", "ops", "function", "FnOnce"],
    )
}

fn fn_item_call_once_fixture_with(
    callee_name: &[&str],
    n_inputs: usize,
    tuple_elem_tys: &[serde_json::Value],
    dest_ty: serde_json::Value,
    fn_def_generics: serde_json::Value,
    decl_generics: Option<serde_json::Value>,
    dont_look_inside: bool,
    trait_path: &[&str],
) -> Llbc {
    let span = serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    });
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| {
                    if *segment == "<Impl>" {
                        serde_json::json!({"Impl": {"kind": "InherentImplBlock"}})
                    } else {
                        serde_json::json!({"Ident": [segment, 0]})
                    }
                })
                .collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let callee_inputs: Vec<serde_json::Value> = (0..n_inputs)
        .map(|i| {
            tuple_elem_tys
                .get(i)
                .cloned()
                .unwrap_or_else(|| str_ty.clone())
        })
        .collect();
    let tuple_ty = serde_json::json!({"Adt": {
        "builtin": "Tuple",
        "generics": {
            "regions": [],
            "types": tuple_elem_tys,
            "const_generics": [],
            "trait_refs": []
        }
    }});
    let fn_ty = serde_json::json!({
        "FnDef": {
            "kind": {"Fun": 1},
            "generics": fn_def_generics.clone()
        }
    });
    let fn_const = serde_json::json!({"Const": [{
            "FnDef": {
                "kind": {"Fun": 1},
                "generics": fn_def_generics.clone()
            }
        }, fn_ty]});
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "call_once_fndef"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [tuple_ty.clone()],
            "output": dest_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span, "ty": dest_ty.clone()},
                        {"index": 1, "name": "args", "span": span, "ty": tuple_ty.clone()}
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Trait": [
                                                    {
                                                        "kind": "Unknown",
                                                        "trait_decl_ref": {
                                                            "skip_binder": {"id": 0}
                                                        }
                                                    },
                                                    0
                                                ]},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [
                                            fn_const,
                                            {"Copy": {"kind": {"Local": 1}, "ty": tuple_ty.clone()}}
                                        ],
                                        "dest": {"kind": {"Local": 0}, "ty": dest_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }
        }
    });
    let mut callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(callee_name, dont_look_inside),
        "signature": {
            "is_unsafe": false,
            "inputs": callee_inputs,
            "output": dest_ty.clone()
        },
        "body": "Opaque"
    });
    if let Some(decl_generics) = decl_generics {
        callee["generics"] = decl_generics;
    }
    let trait_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(trait_path, false),
        "methods": []
    });
    let mut global_decls = Vec::new();
    if dont_look_inside {
        let leaf = *callee_name.last().expect("callee leaf");
        let marker_leaf = format!("_jit_look_inside_{leaf}");
        global_decls.push(serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", &marker_leaf], true),
            "global_kind": "NamedConst",
            "ty": {"Scalar": "Bool"},
            "value": [{"Bool": false}, {"Scalar": "Bool"}]
        }));
    }
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [caller, callee],
            "global_decls": global_decls,
            "trait_decls": [trait_decl],
            "trait_impls": [],
            "assoc_item_names": [{
                "types": [],
                "methods": ["call_once"],
                "consts": []
            }]
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("FnDef call_once fixture Llbc parses")
}

#[test]
fn fn_item_call_once_of_str_is_empty_is_len_eq_zero() {
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let llbc = fn_item_call_once_fixture(
        &["core", "str", "<Impl>", "is_empty"],
        1,
        &[str_ty],
        bool_ty,
    );
    let graph = super::lower_function(&llbc, "call_once_fndef")
        .expect("lower FnOnce::call_once of str::is_empty");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.as_slice() == ["__len"]
        )),
        "FnDef call_once of str::is_empty must become __len; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(op.kind, OpKind::ConstInt(0))),
        "FnDef call_once of str::is_empty must compare with 0; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq")),
        "FnDef call_once of str::is_empty must become eq; ops={ops:?}"
    );
    let leafs = call_leafs(&ops);
    assert!(
        !leafs
            .iter()
            .any(|leaf| *leaf == "call_once" || *leaf == "is_empty"),
        "FnDef call_once of str::is_empty must not residualize; ops={ops:?}"
    );
}

#[test]
fn fn_item_call_once_arity_mismatch_stays_residual() {
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let llbc = fn_item_call_once_fixture(
        &["core", "str", "<Impl>", "is_empty"],
        1,
        &[str_ty.clone(), str_ty],
        bool_ty,
    );
    let graph = super::lower_function(&llbc, "call_once_fndef")
        .expect("lower arity-mismatched FnOnce::call_once");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "call_once"),
        "FnDef call_once whose Args arity differs from the FunDecl must decline; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.as_slice() == ["__len"]
        )),
        "an arity mismatch must not unpack into str::is_empty; ops={ops:?}"
    );
}

#[test]
fn fn_item_call_once_keeps_fndef_type_argument() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = fn_item_call_once_fixture_with(
        &["fixture", "generic_item"],
        1,
        &[i64_ty.clone()],
        i64_ty.clone(),
        serde_json::json!({
            "regions": [],
            "types": [i64_ty],
            "const_generics": [],
            "trait_refs": []
        }),
        Some(serde_json::json!({
            "regions": [],
            "types": [{"index": 0, "name": "T"}],
            "const_generics": [],
            "trait_clauses": []
        })),
        true,
        &["core", "ops", "function", "FnOnce"],
    );
    let graph = super::lower_function(&llbc, "call_once_fndef")
        .expect("lower FnOnce::call_once of a generic FnDef");
    let ops = graph_ops(&graph);
    let leafs = call_leafs(&ops);
    assert!(
        leafs
            .iter()
            .any(|leaf| leaf.contains("__inst_") && leaf.contains("i64")),
        "FnDef type argument must stay on the rewritten call; ops={ops:?}"
    );
    assert!(
        !leafs.iter().any(|leaf| *leaf == "call_once"),
        "a readable generic FnDef must unpack; ops={ops:?}"
    );
}

#[test]
fn user_trait_named_fn_once_is_not_retargeted() {
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let empty = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let llbc = fn_item_call_once_fixture_with(
        &["core", "str", "<Impl>", "is_empty"],
        1,
        &[str_ty],
        bool_ty,
        empty,
        None,
        false,
        &["pyre_test", "FnOnce"],
    );
    let graph = super::lower_function(&llbc, "call_once_fndef")
        .expect("lower user-defined FnOnce::call_once");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "call_once"),
        "a user-defined FnOnce must not retarget to the FnDef; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.as_slice() == ["__len"]
        )),
        "a user-defined FnOnce must not unpack into str::is_empty; ops={ops:?}"
    );
}

fn fn_item_call_once_gc_mut_ref_fixture() -> Llbc {
    let span = serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    });
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
                .collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let mut_ref = serde_json::json!({
        "Ref": ["Erased", {"RawPtr": [{"Adt": {"id": 7}}, "Mut"]}, "Mut"]
    });
    let dest_ty = serde_json::json!({"Adt": {
        "builtin": "Tuple",
        "generics": generics.clone()
    }});
    let tuple_ty = serde_json::json!({"Adt": {
        "builtin": "Tuple",
        "generics": {
            "regions": [],
            "types": [mut_ref.clone()],
            "const_generics": [],
            "trait_refs": []
        }
    }});
    let fn_ty = serde_json::json!({
        "FnDef": {
            "kind": {"Fun": 1},
            "generics": generics.clone()
        }
    });
    let fn_const = serde_json::json!({"Const": [{
            "FnDef": {
                "kind": {"Fun": 1},
                "generics": generics.clone()
            }
        }, fn_ty]});
    let locals = serde_json::json!([
        {"index": 0, "name": null, "span": span, "ty": dest_ty.clone()},
        {"index": 1, "name": "args", "span": span, "ty": tuple_ty.clone()}
    ]);
    let unwind_and_return = serde_json::json!([
        {
            "statements": [],
            "terminator": {"span": span, "kind": "UnwindResume"}
        },
        {
            "statements": [],
            "terminator": {"span": span, "kind": "Return"}
        }
    ]);
    let once_caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "call_once_fndef"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [tuple_ty.clone()],
            "output": dest_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {"arg_count": 1, "locals": locals.clone()},
                "body": [
                    {
                        "statements": [],
                        "terminator": {
                            "span": span,
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Trait": [
                                                    {
                                                        "kind": "Unknown",
                                                        "trait_decl_ref": {
                                                            "skip_binder": {"id": 0}
                                                        }
                                                    },
                                                    0
                                                ]},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [
                                            fn_const,
                                            {"Copy": {"kind": {"Local": 1}, "ty": tuple_ty.clone()}}
                                        ],
                                        "dest": {"kind": {"Local": 0}, "ty": dest_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    unwind_and_return[0].clone(),
                    unwind_and_return[1].clone()
                ]
            }
        }
    });
    let field0 = serde_json::json!({
        "kind": {"Projection": [
            {"kind": {"Local": 1}, "ty": tuple_ty.clone()},
            {"Field": [null, 0]}
        ]},
        "ty": mut_ref.clone()
    });
    let direct_caller = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["fixture", "direct_mut_ref"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [tuple_ty.clone()],
            "output": dest_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {"arg_count": 1, "locals": locals},
                "body": [
                    {
                        "statements": [],
                        "terminator": {
                            "span": span,
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 1},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [{"Copy": field0}],
                                        "dest": {"kind": {"Local": 0}, "ty": dest_ty.clone()}
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    unwind_and_return[0].clone(),
                    unwind_and_return[1].clone()
                ]
            }
        }
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["fixture", "takes_mut_ref"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [mut_ref.clone()],
            "output": dest_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span, "ty": dest_ty.clone()},
                        {"index": 1, "name": "slot", "span": span, "ty": mut_ref}
                    ]
                },
                "body": [{
                    "statements": [],
                    "terminator": {"span": span, "kind": "Return"}
                }]
            }
        }
    });
    let trait_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["core", "ops", "function", "FnOnce"], false),
        "methods": []
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [once_caller, callee, direct_caller],
            "global_decls": [],
            "trait_decls": [trait_decl],
            "trait_impls": [],
            "assoc_item_names": [{
                "types": [],
                "methods": ["call_once"],
                "consts": []
            }]
        }
    });
    Llbc::from_slice(file.to_string().as_bytes())
        .expect("FnDef call_once &mut T fixture Llbc parses")
}

#[test]
fn fn_item_call_once_gc_mut_ref_matches_direct_call() {
    let llbc = fn_item_call_once_gc_mut_ref_fixture();
    let once = super::lower_function(&llbc, "call_once_fndef")
        .expect("lower FnOnce::call_once of takes_mut_ref");
    let direct =
        super::lower_function(&llbc, "direct_mut_ref").expect("lower direct takes_mut_ref(args.0)");
    let once_kinds = startblock_op_kind_names(&once);
    let direct_kinds = startblock_op_kind_names(&direct);
    assert_eq!(
        once_kinds, direct_kinds,
        "rewritten FnOnce call must use the same call-block op kinds as a direct f(&mut x); once={once_kinds:?} direct={direct_kinds:?}"
    );
    let once_ops = graph_ops(&once);
    assert!(
        !call_leafs(&once_ops)
            .iter()
            .any(|leaf| *leaf == "call_once"),
        "FnDef call_once of takes_mut_ref must unpack; ops={once_ops:?}"
    );
    assert!(
        once_ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. } if field.name == "value"
        )),
        "the rewritten &mut T argument must wrap a MutRef cell; ops={once_ops:?}"
    );
}

#[test]
fn ptr_add_of_bytes_is_unscaled_direct_ptradd() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "add_u8",
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        &[ptr_ty.clone(), usize_ty],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "add_u8").expect("lower *mut u8::add");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "*mut u8::add must become direct_ptradd; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mul")),
        "*mut u8::add must not scale; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add")),
        "*mut u8::add must not become int_add at the front; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "add"),
        "*mut u8::add must not residualize; ops={ops:?}"
    );
}

#[test]
fn ptr_add_of_i64_scales_count_then_direct_ptradd() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "add_i64",
        &["core", "ptr", "const_ptr", "<Impl>", "add"],
        &[ptr_ty.clone(), usize_ty],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "add_i64").expect("lower *const i64::add");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter().any(|op| matches!(op.kind, OpKind::ConstInt(8))),
        "*const i64::add must leave scaling to rewrite_op_direct_ptradd; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mul")),
        "*const i64::add must not scale the count; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "*const i64::add must become direct_ptradd; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add")),
        "*const i64::add must not become int_add at the front; ops={ops:?}"
    );
}

#[test]
fn ptr_add_of_i64_emits_unscaled_count() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "add_i64_unscaled",
        &["core", "ptr", "const_ptr", "<Impl>", "add"],
        &[ptr_ty.clone(), usize_ty],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "add_i64_unscaled").expect("lower *const i64::add");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter().any(|op| matches!(op.kind, OpKind::ConstInt(8))),
        "item scaling belongs to rewrite_op_direct_ptradd; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mul")),
        "the count must stay unscaled; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty: ValueType::Int,
                ..
            } if segments.last().map(String::as_str) == Some("direct_ptradd")
        )),
        "direct_ptradd result is a raw pointer (int kind); ops={ops:?}"
    );
}

#[test]
fn ptr_sub_of_bytes_is_unscaled_direct_ptradd_with_negated_count() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "sub_u8",
        &["core", "ptr", "mut_ptr", "<Impl>", "sub"],
        &[ptr_ty.clone(), usize_ty],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "sub_u8").expect("lower *mut u8::sub");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "*mut u8::sub must become direct_ptradd; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "neg")),
        "*mut u8::sub must negate the byte count; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mul")),
        "*mut u8::sub must not scale; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "sub"),
        "*mut u8::sub must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_neg_of_i64_is_int_neg() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_neg_i64",
        &["core", "num", "<Impl>", "wrapping_neg"],
        &[i64_ty.clone()],
        i64_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_neg_i64").expect("lower i64::wrapping_neg");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "neg")),
        "i64::wrapping_neg must become int_neg; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_neg"),
        "i64::wrapping_neg must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_shl_of_usize_is_lshift() {
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_shl_usize",
        &["core", "num", "<Impl>", "wrapping_shl"],
        &[usize_ty.clone(), u32_ty],
        usize_ty,
    );
    let graph =
        super::lower_function(&llbc, "wrapping_shl_usize").expect("lower usize::wrapping_shl");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lshift")),
        "usize::wrapping_shl must become lshift; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_shl"),
        "usize::wrapping_shl must not residualize; ops={ops:?}"
    );
}

fn rhs_defined_by(
    ops: &[&SpaceOperation],
    rhs: &Variable,
    kind_is: impl Fn(&OpKind) -> bool,
) -> bool {
    ops.iter()
        .any(|op| op.result.as_ref() == Some(rhs) && kind_is(&op.kind))
}

#[test]
fn wrapping_add_of_u32_is_add_then_mask() {
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_add_u32",
        &["core", "num", "<Impl>", "wrapping_add"],
        &[u32_ty.clone(), u32_ty.clone()],
        u32_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_add_u32").expect("lower u32::wrapping_add");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add")),
        "u32::wrapping_add must become add; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| match &op.kind {
            OpKind::BinOp { op, rhs, .. } if op == "and" => {
                rhs_defined_by(&ops, rhs, |kind| {
                    matches!(kind, OpKind::ConstUInt(0xFFFF_FFFF))
                })
            }
            _ => false,
        }),
        "u32::wrapping_add must mask with 0xFFFF_FFFF; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_add"),
        "u32::wrapping_add must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_add_of_i32_is_add_then_sign_extend() {
    let i32_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_add_i32",
        &["core", "num", "<Impl>", "wrapping_add"],
        &[i32_ty.clone(), i32_ty.clone()],
        i32_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_add_i32").expect("lower i32::wrapping_add");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add")),
        "i32::wrapping_add must become add; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| match &op.kind {
            OpKind::BinOp { op, rhs, .. } if op == "signext" => {
                rhs_defined_by(&ops, rhs, |kind| matches!(kind, OpKind::ConstInt(4)))
            }
            _ => false,
        }),
        "i32::wrapping_add must signext by 4 bytes; ops={ops:?}"
    );
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "signext"))
            .count(),
        1,
        "i32::wrapping_add must emit one signext; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| {
            matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lshift" || op == "rshift")
        }),
        "i32::wrapping_add must not lshift/rshift; ops={ops:?}"
    );
    assert!(
        call_leafs(&ops).is_empty(),
        "i32::wrapping_add must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_div_of_u32_stays_a_call() {
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_div_u32",
        &["core", "num", "<Impl>", "wrapping_div"],
        &[u32_ty.clone(), u32_ty.clone()],
        u32_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_div_u32").expect("lower u32::wrapping_div");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_div"),
        "u32::wrapping_div must stay a call; ops={ops:?}"
    );
}

fn has_binop(ops: &[&SpaceOperation], name: &str) -> bool {
    ops.iter()
        .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == name))
}

fn has_and_const_uint(ops: &[&SpaceOperation], value: u64) -> bool {
    ops.iter().any(|op| match &op.kind {
        OpKind::BinOp { op, rhs, .. } if op == "and" => rhs_defined_by(
            ops,
            rhs,
            |kind| matches!(kind, OpKind::ConstUInt(n) if *n == value),
        ),
        _ => false,
    })
}

#[test]
fn wrapping_shl_of_u32_is_lshift_then_mask() {
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_shl_u32",
        &["core", "num", "<Impl>", "wrapping_shl"],
        &[u32_ty.clone(), u32_ty.clone()],
        u32_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_shl_u32").expect("lower u32::wrapping_shl");
    let ops = graph_ops(&graph);
    assert!(
        has_binop(&ops, "lshift"),
        "u32::wrapping_shl must become lshift; ops={ops:?}"
    );
    assert!(
        has_and_const_uint(&ops, 31),
        "u32::wrapping_shl must mask the count with 31; ops={ops:?}"
    );
    assert!(
        has_and_const_uint(&ops, 0xFFFF_FFFF),
        "u32::wrapping_shl must mask the result with 0xFFFF_FFFF; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_shl"),
        "u32::wrapping_shl must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_shr_of_u32_is_rshift_without_truncation() {
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_shr_u32",
        &["core", "num", "<Impl>", "wrapping_shr"],
        &[u32_ty.clone(), u32_ty.clone()],
        u32_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_shr_u32").expect("lower u32::wrapping_shr");
    let ops = graph_ops(&graph);
    assert!(
        has_binop(&ops, "rshift"),
        "u32::wrapping_shr must become rshift; ops={ops:?}"
    );
    assert!(
        has_and_const_uint(&ops, 31),
        "u32::wrapping_shr must mask the count with 31; ops={ops:?}"
    );
    assert!(
        !has_and_const_uint(&ops, 0xFFFF_FFFF),
        "u32::wrapping_shr must not truncate; the receiver is already canonical; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_shr"),
        "u32::wrapping_shr must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_neg_of_u32_is_neg_then_mask() {
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_neg_u32",
        &["core", "num", "<Impl>", "wrapping_neg"],
        &[u32_ty.clone()],
        u32_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_neg_u32").expect("lower u32::wrapping_neg");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "neg")),
        "u32::wrapping_neg must become int_neg; ops={ops:?}"
    );
    assert!(
        has_and_const_uint(&ops, 0xFFFF_FFFF),
        "u32::wrapping_neg must mask with 0xFFFF_FFFF; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_neg"),
        "u32::wrapping_neg must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_add_of_usize_is_bare_add() {
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_add_usize",
        &["core", "num", "<Impl>", "wrapping_add"],
        &[usize_ty.clone(), usize_ty.clone()],
        usize_ty,
    );
    let graph =
        super::lower_function(&llbc, "wrapping_add_usize").expect("lower usize::wrapping_add");
    let ops = graph_ops(&graph);
    assert!(
        has_binop(&ops, "add"),
        "usize::wrapping_add at the host word must become add; ops={ops:?}"
    );
    assert!(
        !has_and_const_uint(&ops, 0xFFFF_FFFF),
        "usize::wrapping_add at the host word must not truncate; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_add"),
        "usize::wrapping_add must not residualize; ops={ops:?}"
    );
}

#[test]
fn wrapping_neg_of_usize_is_bare_neg() {
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_neg_usize",
        &["core", "num", "<Impl>", "wrapping_neg"],
        &[usize_ty.clone()],
        usize_ty,
    );
    let graph =
        super::lower_function(&llbc, "wrapping_neg_usize").expect("lower usize::wrapping_neg");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "neg")),
        "usize::wrapping_neg at the host word must become int_neg; ops={ops:?}"
    );
    assert!(
        !has_and_const_uint(&ops, 0xFFFF_FFFF),
        "usize::wrapping_neg at the host word must not truncate; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_neg"),
        "usize::wrapping_neg must not residualize; ops={ops:?}"
    );
}

fn is_lltype_direct_ptradd(op: &SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty: ValueType::Int,
            ..
        } if segments.last().map(String::as_str) == Some("direct_ptradd")
            && segments.iter().any(|s| s == "lltype")
    )
}

fn return_merge_arm_value_types(graph: &crate::model::FunctionGraph) -> Vec<ValueType> {
    use crate::model::LinkArg;
    let returnblock = graph.returnblock;
    let mut tys = Vec::new();
    for block in &graph.blocks {
        for link in &block.exits {
            if link.target != returnblock {
                continue;
            }
            for arg in &link.args {
                let LinkArg::Value(v) = arg else {
                    continue;
                };
                let def = graph
                    .blocks
                    .iter()
                    .flat_map(|b| b.operations.iter())
                    .find(|op| op.result.as_ref() == Some(v));
                if let Some(ty) = def.and_then(|op| match &op.kind {
                    OpKind::Call { result_ty, .. }
                    | OpKind::BinOp { result_ty, .. }
                    | OpKind::UnaryOp { result_ty, .. } => Some(result_ty.clone()),
                    OpKind::Input { ty, .. } => Some(ty.clone()),
                    OpKind::ConstRefNull => Some(ValueType::Ref(None)),
                    OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => Some(ValueType::Int),
                    OpKind::ConstUInt(_) => Some(ValueType::Unsigned),
                    _ => None,
                }) {
                    tys.push(ty);
                }
            }
        }
    }
    tys
}

fn return_merge_arm_kinds_agree(tys: &[ValueType]) -> bool {
    tys.len() >= 2
        && tys
            .windows(2)
            .all(|w| std::mem::discriminant(&w[0]) == std::mem::discriminant(&w[1]))
}

#[test]
fn null_or_offset_pointer_return_arms_share_one_value_type() {
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load real pyre-object LLBC");
    let graph = super::lower_function(&llbc, "items_block_items_base")
        .expect("lower items_block_items_base");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "null_mut"),
        "null arm must stay null_mut(); ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add")),
        "offset arm must not become int_add at the front; ops={ops:?}"
    );
    let arm_tys = return_merge_arm_value_types(&graph);
    assert_eq!(
        arm_tys.len(),
        2,
        "items_block_items_base returns through two arms; got {arm_tys:?}"
    );
    assert!(
        arm_tys.iter().all(|ty| matches!(ty, ValueType::Ref(_))),
        "both return arms must be Ref; got {arm_tys:?}"
    );
    assert!(
        return_merge_arm_kinds_agree(&arm_tys),
        "null_mut and the offset arm must share one ValueType; got {arm_tys:?}"
    );
}

/// `tupleobject.py wraptuple` length 2 is `list_w[0]`, `list_w[1]`,
/// `newtuple2`. Those reads are `rlist.py ll_getitem_fast` —
/// `getarrayitem` of a constant index, not `direct_ptradd`. The
/// items-base accessor result is the `GcArray(OBJECTPTR)` list
/// (`[PyObject]`), so `getitem` uses ListRepr rather than
/// `getitem_SomeInstance`.
#[test]
fn wraptuple_len2_item_reads_are_getarrayitem() {
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load real pyre-object LLBC");
    let graph = super::lower_function(&llbc, "wraptuple").expect("lower wraptuple");
    let ops = graph_ops(&graph);
    let array_reads = ops
        .iter()
        .filter(|op| matches!(&op.kind, OpKind::ArrayRead { .. }))
        .count();
    assert!(
        array_reads >= 2,
        "length-2 wraptuple must getarrayitem both slots; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops)
            .iter()
            .any(|leaf| *leaf == "add" || *leaf == "wrapping_add"),
        "length-2 item pointer must not remain a residual add; ops={ops:?}"
    );
    let items_base = ops
        .iter()
        .copied()
        .find(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("tuple_items_base")
            )
        })
        .unwrap_or_else(|| panic!("wraptuple must call tuple_items_base; ops={ops:?}"));
    let items_base_result = items_base
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("tuple_items_base must define a result; ops={ops:?}"));
    let narrowed_to_object_list = ops.iter().any(|op| {
        crate::model::cast_instance_root(&op.kind) == Some("[pyobject::PyObject]")
            && matches!(
                &op.kind,
                OpKind::Call { args, .. }
                    if matches!(
                        args.first(),
                        Some(LinkArg::Value(value)) if value == items_base_result
                    )
            )
    });
    assert!(
        narrowed_to_object_list,
        "tuple_items_base result must annotate as [pyobject::PyObject] GcArray; ops={ops:?}"
    );
}

#[test]
fn ptr_add_lookalike_stays_residual() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let llbc = std_extern_call_fixture(
        "wrapping_add",
        &["core", "ptr", "mut_ptr", "<Impl>", "wrapping_add"],
        &[ptr_ty.clone(), usize_ty],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "wrapping_add").expect("lower wrapping_add");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| *leaf == "wrapping_add"),
        "wrapping_add must stay residual; ops={ops:?}"
    );
}

#[test]
fn ptr_write_of_scalar_is_a_raw_store() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    });
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let unit = serde_json::json!({
        "Adt": {"id": 0, "builtin": "Tuple", "generics": {
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        }}
    });
    let llbc = std_extern_call_fixture(
        "write_i64",
        &["core", "ptr", "write"],
        &[ptr_ty, i64_ty],
        unit,
    );
    let graph = super::lower_function(&llbc, "write_i64").expect("lower ptr::write");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::RawStore {
                itemsize: 8,
                is_item_signed: true,
                ..
            }
        )),
        "ptr::write of i64 must become RawStore; ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| *leaf == "write"),
        "scalar ptr::write must not residualize; ops={ops:?}"
    );
}

#[test]
fn ptr_write_lookalike_stays_residual() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    });
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let unit = serde_json::json!({
        "Adt": {"id": 0, "builtin": "Tuple", "generics": {
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        }}
    });
    let llbc = std_extern_call_fixture(
        "write_unaligned",
        &["core", "ptr", "write_unaligned"],
        &[ptr_ty, i64_ty],
        unit,
    );
    let graph = super::lower_function(&llbc, "write_unaligned").expect("lower write_unaligned");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops)
            .iter()
            .any(|leaf| *leaf == "write_unaligned"),
        "write_unaligned must stay residual; ops={ops:?}"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(op.kind, OpKind::RawStore { .. })),
        "write_unaligned must not become RawStore; ops={ops:?}"
    );
}

#[test]
fn vec_new_retargets_to_the_empty_list_constructor() {
    let vec_ty = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let llbc = std_extern_call_fixture("vec_new", &["alloc", "vec", "<Impl>", "new"], &[], vec_ty);
    let graph = super::lower_function(&llbc, "vec_new").expect("lower Vec::new");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &["vec".to_string(), "Vec".to_string(), "new".to_string()]
        )),
        "alloc::vec::<Impl>::new must retarget to vec::Vec::new; ops={ops:?}"
    );
}

#[test]
fn vec_push_is_not_retargeted_to_list_append() {
    let vec_ty = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let unit = serde_json::json!({
        "Adt": {"id": 0, "builtin": "Tuple", "generics": {
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        }}
    });
    let llbc = std_extern_call_fixture(
        "vec_push",
        &["alloc", "vec", "<Impl>", "push"],
        &[vec_ty, i64_ty],
        unit,
    );
    let graph = super::lower_function(&llbc, "vec_push").expect("lower Vec::push");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &[
                "alloc".to_string(),
                "vec".to_string(),
                "<Impl>".to_string(),
                "push".to_string()
            ]
        )),
        "alloc::vec::<Impl>::push must stay residual; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &["vec".to_string(), "Vec".to_string(), "push".to_string()]
        )),
        "Vec::push must not be a front-end list-append retarget; ops={ops:?}"
    );
}

#[test]
fn one_word_item_vec_std_calls_retarget_to_the_vec_helpers() {
    use majit_ir::rvec::{VecItemKind, VecOp, vec_helper_path};
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![i64_ty.clone()])}
    });
    let unit =
        serde_json::json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics(vec![])}});
    let i64_ptr = serde_json::json!({"RawPtr": [i64_ty.clone(), "Const"]});
    let i64_mut_ptr = serde_json::json!({"RawPtr": [i64_ty.clone(), "Mut"]});
    let vec_decl = || {
        let name: Vec<serde_json::Value> = ["alloc", "vec", "Vec"]
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        serde_json::json!({
            "def_id": 0,
            "item_meta": {
                "name": name,
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }},
                "source_text": null,
                "attr_info": {
                    "attributes": [], "inline": null, "rename": null, "public": true
                },
                "is_local": false
            },
            "kind": {"Struct": []}
        })
    };
    let cases = [
        ("vec_new", "new", vec![], vec_ty.clone(), VecOp::NewEmpty),
        (
            "vec_with_capacity",
            "with_capacity",
            vec![usize_ty],
            vec_ty.clone(),
            VecOp::NewHint,
        ),
        (
            "vec_push",
            "push",
            vec![vec_ty.clone(), i64_ty],
            unit,
            VecOp::Append,
        ),
        (
            "vec_as_ptr",
            "as_ptr",
            vec![vec_ty.clone()],
            i64_ptr,
            VecOp::Items,
        ),
        (
            "vec_as_mut_ptr",
            "as_mut_ptr",
            vec![vec_ty.clone()],
            i64_mut_ptr,
            VecOp::Items,
        ),
    ];
    for (caller, leaf, arg_tys, dest_ty, op) in cases {
        let llbc = std_extern_call_fixture_with_types(
            caller,
            &["alloc", "vec", "<Impl>", leaf],
            &arg_tys,
            dest_ty,
            vec![vec_decl()],
        );
        let graph = super::lower_function(&llbc, caller).expect("lower Vec call");
        let expected: Vec<String> = vec_helper_path(op, VecItemKind::Int)
            .split("::")
            .map(str::to_string)
            .collect();
        let ops = graph_ops(&graph);
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if *segments == expected
            )),
            "Vec::{leaf} on Vec<i64> must call {expected:?}; ops={ops:?}"
        );
    }
}

#[test]
fn one_word_item_vec_set_len_writes_the_len_word() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![i64_ty.clone()])}
    });
    let unit =
        serde_json::json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics(vec![])}});
    let vec_decl = {
        let name: Vec<serde_json::Value> = ["alloc", "vec", "Vec"]
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        serde_json::json!({
            "def_id": 0,
            "item_meta": {
                "name": name,
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }},
                "source_text": null,
                "attr_info": {
                    "attributes": [], "inline": null, "rename": null, "public": true
                },
                "is_local": false
            },
            "kind": {"Struct": []}
        })
    };
    let llbc = std_extern_call_fixture_with_types(
        "vec_set_len",
        &["alloc", "vec", "<Impl>", "set_len"],
        &[vec_ty, usize_ty],
        unit,
        vec![vec_decl],
    );
    let graph = super::lower_function(&llbc, "vec_set_len").expect("lower Vec::set_len");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, ty: ValueType::Int, .. }
                if field.name == "len"
                    && field.owner_root.as_deref() == Some("alloc::vec::Vec")
                    && field.vec_part.is_none()
        )),
        "Vec::set_len must write the header len word; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. }
                if field.vec_part == Some(crate::model::VecFieldPart::Len)
        )),
        "Vec::set_len must not mint a parentless len.len FieldWrite; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "set_len")
        )),
        "Vec::set_len must not remain a FunctionPath; ops={ops:?}"
    );
}

#[test]
fn one_word_copy_nonoverlapping_is_slice_arraycopy() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let i64_ptr = serde_json::json!({"RawPtr": [i64_ty.clone(), "Const"]});
    let i64_mut_ptr = serde_json::json!({"RawPtr": [i64_ty, "Mut"]});
    let unit =
        serde_json::json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics(vec![])}});
    let llbc = std_extern_call_fixture_with_types(
        "copy_i64",
        &["core", "ptr", "copy_nonoverlapping"],
        &[i64_ptr, i64_mut_ptr, usize_ty],
        unit,
        vec![],
    );
    let graph = super::lower_function(&llbc, "copy_i64").expect("lower ptr::copy_nonoverlapping");
    let ops = graph_ops(&graph);
    let expected: Vec<String> = majit_ir::rvec::slice_helper_path(
        majit_ir::rvec::SliceOp::ArrayCopy,
        majit_ir::rvec::VecItemKind::Int,
    )
    .split("::")
    .map(str::to_string)
    .collect();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments == &expected && args.len() == 5
        )),
        "copy_nonoverlapping of *i64 must be ll_slice_arraycopy_i; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "copy_nonoverlapping")
        )),
        "copy_nonoverlapping must not remain a FunctionPath; ops={ops:?}"
    );
}

#[test]
fn byte_item_vec_as_ptr_retargets_to_the_kind_independent_helper() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![u8_ty.clone()])}
    });
    let u8_ptr = serde_json::json!({"RawPtr": [u8_ty.clone(), "Const"]});
    let u8_mut_ptr = serde_json::json!({"RawPtr": [u8_ty, "Mut"]});
    let vec_decl = {
        let name: Vec<serde_json::Value> = ["alloc", "vec", "Vec"]
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        serde_json::json!({
            "def_id": 0,
            "item_meta": {
                "name": name,
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }},
                "source_text": null,
                "attr_info": {
                    "attributes": [], "inline": null, "rename": null, "public": true
                },
                "is_local": false
            },
            "kind": {"Struct": []}
        })
    };
    let expected =
        vec![crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST.to_string()];
    for (caller, leaf, dest_ty) in [
        ("vec_u8_as_ptr", "as_ptr", u8_ptr),
        ("vec_u8_as_mut_ptr", "as_mut_ptr", u8_mut_ptr),
    ] {
        let llbc = std_extern_call_fixture_with_types(
            caller,
            &["alloc", "vec", "<Impl>", leaf],
            &[vec_ty.clone()],
            dest_ty,
            vec![vec_decl.clone()],
        );
        let graph = super::lower_function(&llbc, caller).expect("lower Vec<u8> as_ptr");
        let ops = graph_ops(&graph);
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if *segments == expected
            )),
            "Vec::{leaf} on Vec<u8> must call {expected:?}; ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some(leaf)
            )),
            "Vec::{leaf} on Vec<u8> must not stay residual; ops={ops:?}"
        );
    }
}

#[test]
fn a_vec_extends_from_a_gcarray_word_through_ll_extend() {
    use majit_ir::rvec::{VecItemKind, VecOp, vec_helper_path};
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let named = |path: &[&str], def_id: u64| {
        let name: Vec<serde_json::Value> = path
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": name,
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }},
                "source_text": null,
                "attr_info": {
                    "attributes": [], "inline": null, "rename": null, "public": true
                },
                "is_local": false
            },
            "kind": {"Struct": []}
        })
    };
    let obj_adt = serde_json::json!({
        "Adt": {"id": 1, "generics": generics(vec![])}
    });
    let objptr = serde_json::json!({"RawPtr": [obj_adt, "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![objptr.clone()])}
    });
    let recv = serde_json::json!({"Ref": ["_", vec_ty, "Mut"]});
    let slice = serde_json::json!({
        "Ref": ["_", {"Slice": [objptr, null]}, "Shared"]
    });
    let unit =
        serde_json::json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics(vec![])}});
    let llbc = std_extern_call_fixture_with_types(
        "extend_from_object_slice",
        &["alloc", "vec", "<Impl>", "extend_from_slice"],
        &[recv, slice],
        unit,
        vec![named(&["alloc", "vec", "Vec"], 0), named(&["PyObject"], 1)],
    );
    let graph = super::lower_function(&llbc, "extend_from_object_slice")
        .expect("lower Vec::extend_from_slice of an object slice");
    let expected: Vec<String> = vec_helper_path(VecOp::Extend, VecItemKind::Ref)
        .split("::")
        .map(str::to_string)
        .collect();
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if *segments == expected && args.len() == 2
        )),
        "object-slice extend_from_slice must call {expected:?} with two args; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("extend_from_slice")
        )),
        "extend_from_slice must not remain as a residual; ops={ops:?}"
    );
}

fn object_pointer_vec_tys() -> (
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    Vec<serde_json::Value>,
) {
    let named = |path: &[&str], def_id: u64| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(ident_path(path)),
            "kind": {"Struct": []}
        })
    };
    let obj_adt = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let objptr = serde_json::json!({"RawPtr": [obj_adt, "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 1, "generics": {"types": [objptr.clone()]}}
    });
    let recv = serde_json::json!({"Ref": ["'_", vec_ty, "Mut"]});
    let slice = serde_json::json!({
        "Ref": ["'_", {"Slice": [objptr, null]}, "Mut"]
    });
    (
        recv,
        slice,
        serde_json::json!({"Tuple": []}),
        vec![named(&["PyObject"], 0), named(&["alloc", "vec", "Vec"], 1)],
    )
}

#[test]
fn object_vec_deref_mut_rotate_calls_ll_slice_rotate() {
    use majit_ir::rvec::{SliceOp, VecItemKind, slice_helper_path};
    let (recv, slice, unit, types) = object_pointer_vec_tys();
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        let mut meta = fixture_item_meta(ident_path(path));
        meta["is_local"] = serde_json::json!(is_local);
        meta
    };
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let place =
        |i: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": i}, "ty": ty});
    let local = |i: u64, name: Option<&str>, ty: &serde_json::Value| serde_json::json!({"index": i, "name": name, "span": span(), "ty": ty});
    let copy = |i: u64, ty: &serde_json::Value| serde_json::json!({"Copy": place(i, ty)});
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "object_vec_deref_mut_rotate"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [recv.clone(), usize_ty.clone()],
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 2,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("v"), &recv),
                        local(2, Some("k"), &usize_ty),
                        local(3, Some("s"), &slice)
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 1},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [copy(1, &recv)],
                                        "dest": place(3, &slice)
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 2},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [copy(3, &slice), copy(2, &usize_ty)],
                                        "dest": place(0, &unit)
                                    },
                                    "target": 4,
                                    "on_unwind": 3
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let deref_mut = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["alloc", "vec", "<Impl>", "deref_mut"], false),
        "signature": {
            "is_unsafe": true,
            "inputs": [recv.clone()],
            "output": slice.clone()
        },
        "body": "Opaque"
    });
    let rotate = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["core", "slice", "<Impl>", "rotate_right"], false),
        "signature": {
            "is_unsafe": true,
            "inputs": [slice.clone(), usize_ty.clone()],
            "output": unit.clone()
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": types,
            "fun_decls": [caller, deref_mut, rotate],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes())
        .expect("object vec deref_mut rotate fixture parses");
    let graph = super::lower_function(&llbc, "object_vec_deref_mut_rotate")
        .expect("lower deref_mut then <[T]>::rotate_right of Vec<PyObjectRef>");
    let expected: Vec<String> = slice_helper_path(SliceOp::RotateRight, VecItemKind::Ref)
        .split("::")
        .map(str::to_string)
        .collect();
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if *segments == expected && args.len() == 3
        )),
        "<[T]>::rotate_right after Vec<PyObjectRef>::deref_mut must call {expected:?}; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("deref_mut")
                || segments.last().map(String::as_str) == Some("rotate_right")
                && segments != &expected
        )),
        "deref_mut / slice rotate_right must not remain residual; ops={ops:?}"
    );
}

#[test]
fn an_object_array_literal_unsize_is_newlist() {
    let generics = || {
        serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        })
    };
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path.iter().map(|segment| serde_json::json!({"Ident": [segment, 0]})).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let named = |path: &[&str], def_id: u64| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta(path, false),
            "kind": {"Struct": []}
        })
    };
    let obj_adt = serde_json::json!({"Adt": {"id": 0, "generics": generics()}});
    let objptr = serde_json::json!({"RawPtr": [obj_adt, "Mut"]});
    let array_len = serde_json::json!([
        {"Integer": {"Unsigned": ["Usize", "3"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let array_ty = serde_json::json!({"Array": [objptr.clone(), array_len, null]});
    let array_ref = serde_json::json!({"Ref": ["_", array_ty.clone(), "Shared"]});
    let slice_ref = serde_json::json!({
        "Ref": ["_", {"Slice": [objptr.clone(), null]}, "Shared"]
    });
    let unit = serde_json::json!({
        "Adt": {"id": 1, "builtin": "Tuple", "generics": generics()}
    });
    let place =
        |i: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": i}, "ty": ty});
    let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "span": span(), "comments_before": []});
    let assign = |dest: u64, dest_ty: &serde_json::Value, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest, dest_ty), rvalue]}))
    };
    let mv = |i: u64, ty: &serde_json::Value| serde_json::json!({"Move": place(i, ty)});
    let copy = |i: u64, ty: &serde_json::Value| serde_json::json!({"Copy": place(i, ty)});
    let usize_lit = serde_json::json!({
        "Const": [
            {"Integer": {"Unsigned": ["Usize", "3"]}},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
        ]
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "pass_object_array_literal"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [objptr.clone(), objptr.clone(), objptr.clone()],
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 3,
                    "locals": [
                        local(0, &unit),
                        local(1, &objptr),
                        local(2, &objptr),
                        local(3, &objptr),
                        local(4, &array_ty),
                        local(5, &array_ref),
                        local(6, &slice_ref)
                    ]
                },
                "body": [
                    {
                        "statements": [
                            assign(4, &array_ty, serde_json::json!({
                                "Aggregate": [
                                    {"Array": [objptr.clone(), usize_lit.clone()]},
                                    [mv(1, &objptr), mv(2, &objptr), mv(3, &objptr)]
                                ]
                            })),
                            assign(5, &array_ref, serde_json::json!({
                                "Ref": {
                                    "place": place(4, &array_ty),
                                    "kind": "Shared",
                                    "ptr_metadata": null
                                }
                            })),
                            assign(6, &slice_ref, serde_json::json!({
                                "UnaryOp": [
                                    {"Cast": {"Unsize": [
                                        array_ref.clone(),
                                        slice_ref.clone(),
                                        {"Length": usize_lit}
                                    ]}},
                                    mv(5, &array_ref)
                                ]
                            }))
                        ],
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 1},
                                                "generics": generics()
                                            }
                                        },
                                        "args": [copy(6, &slice_ref)],
                                        "dest": place(0, &unit)
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["fixture", "take_object_slice"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [slice_ref],
            "output": unit
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [named(&["PyObject"], 0)],
            "fun_decls": [caller, callee],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc =
        Llbc::from_slice(file.to_string().as_bytes()).expect("object array literal fixture parses");
    let graph = super::lower_function(&llbc, "pass_object_array_literal")
        .expect("lower object array literal passed as &[PyObjectRef]");
    let ops = graph_ops(&graph);
    let newlist = ops
        .iter()
        .find(|op| matches!(&op.kind, OpKind::NewList { args } if args.len() == 3));
    let Some(newlist) = newlist else {
        panic!("object array literal unsized to a slice must become newlist; ops={ops:?}");
    };
    let list = newlist
        .result
        .as_ref()
        .expect("newlist produces a value")
        .clone();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(String::as_str) == Some("take_object_slice")
                && args.iter().any(|arg| arg.as_variable() == Some(&list))
        )),
        "the slice parameter must receive the newlist; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name == "Array" || majit_ir::descr::is_shaped_array_name(name)
        )),
        "the array literal must not remain an Array synthetic ctor; ops={ops:?}"
    );
}

#[test]
fn slice_from_ref_of_an_object_pointer_is_newlist() {
    let generics = || {
        serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        })
    };
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path.iter().map(|segment| serde_json::json!({"Ident": [segment, 0]})).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let named = |path: &[&str], def_id: u64| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta(path, false),
            "kind": {"Struct": []}
        })
    };
    let obj_adt = serde_json::json!({"Adt": {"id": 0, "generics": generics()}});
    let objptr = serde_json::json!({"RawPtr": [obj_adt, "Mut"]});
    let obj_ref = serde_json::json!({"Ref": ["_", objptr.clone(), "Shared"]});
    let slice_ref = serde_json::json!({
        "Ref": ["_", {"Slice": [objptr.clone(), null]}, "Shared"]
    });
    let unit = serde_json::json!({
        "Adt": {"id": 1, "builtin": "Tuple", "generics": generics()}
    });
    let place =
        |i: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": i}, "ty": ty});
    let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "span": span(), "comments_before": []});
    let assign = |dest: u64, dest_ty: &serde_json::Value, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest, dest_ty), rvalue]}))
    };
    let mv = |i: u64, ty: &serde_json::Value| serde_json::json!({"Move": place(i, ty)});
    let copy = |i: u64, ty: &serde_json::Value| serde_json::json!({"Copy": place(i, ty)});
    let call =
        |fun: u64, args: serde_json::Value, dest: u64, dest_ty: &serde_json::Value, target: u64| {
            serde_json::json!({"Call": {"call": {
                "func": {"Regular": {
                    "kind": {"Fun": fun},
                    "generics": generics()
                }},
                "args": args,
                "dest": {"kind": {"Local": dest}, "ty": dest_ty}
            }, "target": target, "on_unwind": 1}})
        };
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "pass_object_from_ref"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [objptr.clone()],
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, &unit),
                        local(1, &objptr),
                        local(2, &obj_ref),
                        local(3, &slice_ref)
                    ]
                },
                "body": [
                    {
                        "statements": [
                            assign(2, &obj_ref, serde_json::json!({
                                "Ref": {
                                    "place": place(1, &objptr),
                                    "kind": "Shared",
                                    "ptr_metadata": null
                                }
                            }))
                        ],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": call(1, serde_json::json!([mv(2, &obj_ref)]), 3, &slice_ref, 2)
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span(),
                            "kind": call(2, serde_json::json!([copy(3, &slice_ref)]), 0, &unit, 3)
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let from_ref = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["core", "slice", "raw", "from_ref"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [obj_ref],
            "output": slice_ref.clone()
        },
        "body": "Opaque"
    });
    let callee = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["fixture", "take_object_slice"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [slice_ref],
            "output": unit
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [named(&["PyObject"], 0)],
            "fun_decls": [caller, from_ref, callee],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc =
        Llbc::from_slice(file.to_string().as_bytes()).expect("object from_ref fixture parses");
    let graph = super::lower_function(&llbc, "pass_object_from_ref")
        .expect("lower slice::from_ref of a PyObjectRef");
    let ops = graph_ops(&graph);
    let newlist = ops
        .iter()
        .find(|op| matches!(&op.kind, OpKind::NewList { args } if args.len() == 1));
    let Some(newlist) = newlist else {
        panic!("slice::from_ref of a GC pointer must become newlist; ops={ops:?}");
    };
    let list = newlist
        .result
        .as_ref()
        .expect("newlist produces a value")
        .clone();
    let passed_to_callee = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(String::as_str) == Some("take_object_slice")
                && args.iter().any(|arg| arg.as_variable() == Some(&list))
        )
    });
    // `from_ref` is a Call terminator, so the slice consumer lives
    // in the successor. The NewList result is the Goto argument
    // (`flowspace/model.py` `Link.args`); the callee reads the
    // successor's `inputargs` slot.
    let passed_across_edge = graph.blocks.iter().any(|block| {
        block
            .exits
            .iter()
            .any(|link| link.args.iter().any(|arg| arg.as_variable() == Some(&list)))
    });
    assert!(
        passed_to_callee || passed_across_edge,
        "the slice parameter must receive the newlist; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("from_ref")
        )),
        "from_ref must not remain residual; ops={ops:?}"
    );
}

#[test]
fn an_object_array_literal_unsize_mut_is_not_newlist() {
    let generics = || {
        serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        })
    };
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path.iter().map(|segment| serde_json::json!({"Ident": [segment, 0]})).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let named = |path: &[&str], def_id: u64| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta(path, false),
            "kind": {"Struct": []}
        })
    };
    let obj_adt = serde_json::json!({"Adt": {"id": 0, "generics": generics()}});
    let objptr = serde_json::json!({"RawPtr": [obj_adt, "Mut"]});
    let array_len = serde_json::json!([
        {"Integer": {"Unsigned": ["Usize", "3"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let array_ty = serde_json::json!({"Array": [objptr.clone(), array_len, null]});
    let array_ref = serde_json::json!({"Ref": ["_", array_ty.clone(), "Mut"]});
    let slice_ref = serde_json::json!({
        "Ref": ["_", {"Slice": [objptr.clone(), null]}, "Mut"]
    });
    let unit = serde_json::json!({
        "Adt": {"id": 1, "builtin": "Tuple", "generics": generics()}
    });
    let place =
        |i: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": i}, "ty": ty});
    let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "span": span(), "comments_before": []});
    let assign = |dest: u64, dest_ty: &serde_json::Value, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest, dest_ty), rvalue]}))
    };
    let mv = |i: u64, ty: &serde_json::Value| serde_json::json!({"Move": place(i, ty)});
    let copy = |i: u64, ty: &serde_json::Value| serde_json::json!({"Copy": place(i, ty)});
    let usize_lit = serde_json::json!({
        "Const": [
            {"Integer": {"Unsigned": ["Usize", "3"]}},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
        ]
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", "pass_object_array_literal_mut"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [objptr.clone(), objptr.clone(), objptr.clone()],
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 3,
                    "locals": [
                        local(0, &unit),
                        local(1, &objptr),
                        local(2, &objptr),
                        local(3, &objptr),
                        local(4, &array_ty),
                        local(5, &array_ref),
                        local(6, &slice_ref)
                    ]
                },
                "body": [
                    {
                        "statements": [
                            assign(4, &array_ty, serde_json::json!({
                                "Aggregate": [
                                    {"Array": [objptr.clone(), usize_lit.clone()]},
                                    [mv(1, &objptr), mv(2, &objptr), mv(3, &objptr)]
                                ]
                            })),
                            assign(5, &array_ref, serde_json::json!({
                                "Ref": {
                                    "place": place(4, &array_ty),
                                    "kind": "Mut",
                                    "ptr_metadata": null
                                }
                            })),
                            assign(6, &slice_ref, serde_json::json!({
                                "UnaryOp": [
                                    {"Cast": {"Unsize": [
                                        array_ref.clone(),
                                        slice_ref.clone(),
                                        {"Length": usize_lit}
                                    ]}},
                                    mv(5, &array_ref)
                                ]
                            }))
                        ],
                        "terminator": {
                            "span": span(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 1},
                                                "generics": generics()
                                            }
                                        },
                                        "args": [copy(6, &slice_ref)],
                                        "dest": place(0, &unit)
                                    },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }
        }
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["fixture", "take_object_slice_mut"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [slice_ref],
            "output": unit
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [named(&["PyObject"], 0)],
            "fun_decls": [caller, callee],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes())
        .expect("object array literal mut fixture parses");
    let graph = super::lower_function(&llbc, "pass_object_array_literal_mut")
        .expect("lower object array literal passed as &mut [PyObjectRef]");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::NewList { .. })),
        "object array literal unsized to &mut [T] must not become newlist; ops={ops:?}"
    );
}

#[test]
fn object_vec_as_slice_is_narrowed_to_the_object_gcarray() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let named = |path: &[&str], def_id: u64| {
        let name: Vec<serde_json::Value> = path
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": name,
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }},
                "source_text": null,
                "attr_info": {
                    "attributes": [], "inline": null, "rename": null, "public": true
                },
                "is_local": false
            },
            "kind": {"Struct": []}
        })
    };
    let obj_adt = serde_json::json!({
        "Adt": {"id": 1, "generics": generics(vec![])}
    });
    let objptr = serde_json::json!({"RawPtr": [obj_adt, "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![objptr.clone()])}
    });
    let recv = serde_json::json!({"Ref": ["_", vec_ty, "Shared"]});
    let slice = serde_json::json!({
        "Ref": ["_", {"Slice": [objptr, null]}, "Shared"]
    });
    let llbc = std_extern_call_fixture_with_types(
        "object_vec_as_slice",
        &["alloc", "vec", "<Impl>", "as_slice"],
        &[recv],
        slice,
        vec![named(&["alloc", "vec", "Vec"], 0), named(&["PyObject"], 1)],
    );
    let graph = super::lower_function(&llbc, "object_vec_as_slice")
        .expect("lower Vec<PyObjectRef>::as_slice");
    let ops = graph_ops(&graph);
    let gcarray = ops.iter().find(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.join("::") == "pyre_object::gc_roots::gcarray_from_pyobject_vec"
        )
    });
    let Some(gcarray) = gcarray else {
        panic!("Vec<PyObjectRef>::as_slice must call gcarray_from_pyobject_vec; ops={ops:?}");
    };
    let view = gcarray
        .result
        .as_ref()
        .expect("gcarray_from_pyobject_vec produces a value");
    assert!(
        ops.iter().any(|op| {
            crate::model::cast_instance_root(&op.kind) == Some("[PyObject]")
                && matches!(
                    &op.kind,
                    OpKind::Call { args, .. }
                        if args.first().and_then(LinkArg::as_variable) == Some(view)
                )
        }),
        "gcarray_from_pyobject_vec must be narrowed through the position ListDef; ops={ops:?}"
    );
}

#[test]
fn vec_new_lookalike_stays_on_its_own_path() {
    let vec_ty = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let llbc = std_extern_call_fixture(
        "vec_with_capacity",
        &["alloc", "vec", "<Impl>", "with_capacity"],
        &[serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})],
        vec_ty,
    );
    let graph =
        super::lower_function(&llbc, "vec_with_capacity").expect("lower Vec::with_capacity");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "with_capacity")
        )),
        "Vec::with_capacity must not take the Vec::new retarget; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &["vec".to_string(), "Vec".to_string(), "new".to_string()]
        )),
        "Vec::with_capacity must not become vec::Vec::new; ops={ops:?}"
    );
}

pub(in crate::front::mir) fn graph_ops(
    graph: &crate::model::FunctionGraph,
) -> Vec<&SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect()
}

fn startblock_op_kind_names(graph: &crate::model::FunctionGraph) -> Vec<String> {
    graph
        .block(graph.startblock)
        .operations
        .iter()
        .map(|op| {
            let dbg = format!("{:?}", op.kind);
            dbg.split(['(', '{'])
                .next()
                .unwrap_or(&dbg)
                .trim()
                .to_string()
        })
        .collect()
}

#[test]
fn int_of_float_projects_to_cast_float_to_int() {
    let llbc = scalar_method_call_fixture(
        "to_int_unchecked_i64",
        &["core", "f64", "<Impl>", "to_int_unchecked"],
        serde_json::json!({"Scalar": {"Float": "F64"}}),
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
    );
    let graph = super::lower_function(&llbc, "to_int_unchecked_i64")
        .expect("lower f64::to_int_unchecked::<i64>");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, result_ty, .. }
                if op == "cast_float_to_int" && *result_ty == ValueType::Int
        )),
        "f64::to_int_unchecked::<i64> must become cast_float_to_int; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(op.kind, OpKind::Call { .. })),
        "f64::to_int_unchecked::<i64> must not residualize; ops={ops:?}"
    );
}

#[test]
fn f64_to_bits_projects_to_float2longlong() {
    let llbc = scalar_method_call_fixture(
        "f64_to_bits",
        &["core", "f64", "<Impl>", "to_bits"],
        serde_json::json!({"Scalar": {"Float": "F64"}}),
        serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}}),
    );
    let graph = super::lower_function(&llbc, "f64_to_bits").expect("lower f64::to_bits");
    let ops = graph_ops(&graph);
    assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if matches!(segments.as_slice(), [a, b] if a == "longlong2float" && b == "float2longlong")
            )),
            "f64::to_bits must become float2longlong; ops={ops:?}"
        );
}

#[test]
fn f32_to_bits_is_not_projected() {
    let llbc = scalar_method_call_fixture(
        "f32_to_bits",
        &["core", "f32", "<Impl>", "to_bits"],
        serde_json::json!({"Scalar": {"Float": "F32"}}),
        serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}}),
    );
    let graph = super::lower_function(&llbc, "f32_to_bits").expect("lower f32::to_bits");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => matches!(
                segments.as_slice(),
                [a, b] if a == "longlong2float"
                    && (b == "float2longlong" || b == "longlong2float")
            ),
            _ => false,
        }),
        "f32::to_bits must not become the f64 bitcast; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "to_bits")
        )),
        "f32::to_bits must stay a residual call; ops={ops:?}"
    );
}

#[test]
fn i128_to_int_unchecked_is_not_projected() {
    let llbc = scalar_method_call_fixture(
        "to_int_unchecked_i128",
        &["core", "f64", "<Impl>", "to_int_unchecked"],
        serde_json::json!({"Scalar": {"Float": "F64"}}),
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I128"}}}),
    );
    let graph = super::lower_function(&llbc, "to_int_unchecked_i128")
        .expect("lower f64::to_int_unchecked::<i128>");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, .. } if op == "cast_float_to_int"
        )),
        "to_int_unchecked::<i128> must not become the 64-bit cast; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "to_int_unchecked")
        )),
        "to_int_unchecked::<i128> must stay residual; ops={ops:?}"
    );
}

#[test]
fn f64_i64_transmute_projects_to_float_bytes_llop() {
    let llbc = scalar_method_call_fixture(
        "transmute_f64_to_i64",
        &["core", "intrinsics", "transmute"],
        serde_json::json!({"Scalar": {"Float": "F64"}}),
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
    );
    let graph =
        super::lower_function(&llbc, "transmute_f64_to_i64").expect("lower transmute::<f64, i64>");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, result_ty, .. }
                if op == "convert_float_bytes_to_longlong" && *result_ty == ValueType::Int
        )),
        "transmute::<f64, i64> must become convert_float_bytes_to_longlong; ops={ops:?}"
    );
}

fn assert_unary_llop(ops: &[&SpaceOperation], name: &str, result_ty: ValueType) {
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, result_ty: ty, .. } if op == name && *ty == result_ty
        )),
        "{name} llop missing; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(op.kind, OpKind::Call { .. })),
        "{name} must not residualize; ops={ops:?}"
    );
}

fn assert_lltype_cast_call(ops: &[&SpaceOperation], name: &str) {
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == name)
                && segments.windows(2).any(|w| w[0] == "lltype" && w[1] == name)
        )),
        "{name} host call missing; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, .. } if op == name
        )),
        "{name} must not be a UnaryOp; ops={ops:?}"
    );
}

#[test]
fn host_scalar_calls_project_to_llops() {
    let ptr_ty = serde_json::json!({"RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let f64_ty = serde_json::json!({"Scalar": {"Float": "F64"}});
    let unary_cases = [
        (
            "float2longlong",
            "pyre_float2longlong",
            ["pyre_object", "longlong2float", "float2longlong"],
            f64_ty,
            i64_ty.clone(),
            "convert_float_bytes_to_longlong",
            ValueType::Int,
        ),
        (
            "longlong2float",
            "pyre_longlong2float",
            ["pyre_interpreter", "longlong2float", "longlong2float"],
            i64_ty.clone(),
            serde_json::json!({"Scalar": {"Float": "F64"}}),
            "convert_longlong_bytes_to_float",
            ValueType::Float,
        ),
    ];
    for (name, func, path, arg_ty, ret_ty, llop, result_ty) in unary_cases {
        let llbc = scalar_method_call_fixture(func, &path, arg_ty, ret_ty);
        let lowered = format!("case {name}: lower {func}");
        let graph = super::lower_function(&llbc, func).expect(&lowered);
        let ops = graph_ops(&graph);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_unary_llop(&ops, llop, result_ty);
        }));
        if let Err(payload) = result {
            panic!("case {name}: {}", panic_payload_mir(&payload));
        }
    }
    let cast_cases = [
        (
            "cast_ptr_to_int",
            "pyre_cast_ptr_to_int",
            ["pyre_object", "lltype", "cast_ptr_to_int"],
            ptr_ty.clone(),
            serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
        ),
        (
            "cast_int_to_ptr",
            "pyre_cast_int_to_ptr",
            ["pyre_object", "lltype", "cast_int_to_ptr"],
            serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
            ptr_ty,
        ),
    ];
    for (name, func, path, arg_ty, ret_ty) in cast_cases {
        let llbc = scalar_method_call_fixture(func, &path, arg_ty, ret_ty);
        let lowered = format!("case {name}: lower {func}");
        let graph = super::lower_function(&llbc, func).expect(&lowered);
        let ops = graph_ops(&graph);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_lltype_cast_call(&ops, name);
        }));
        if let Err(payload) = result {
            panic!("case {name}: {}", panic_payload_mir(&payload));
        }
    }
}

fn panic_payload_mir(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "assertion failed".to_string()
    }
}

#[test]
fn lookalike_float2longlong_path_is_not_projected() {
    let llbc = scalar_method_call_fixture(
        "lookalike_float2longlong",
        &["pyre_object", "my_longlong2float", "float2longlong"],
        serde_json::json!({"Scalar": {"Float": "F64"}}),
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
    );
    let graph = super::lower_function(&llbc, "lookalike_float2longlong")
        .expect("lower look-alike float2longlong path");
    let ops = graph_ops(&graph);
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, .. }
                if op == "convert_float_bytes_to_longlong"
                    || op == "convert_longlong_bytes_to_float"
                    || op == "cast_ptr_to_int"
                    || op == "cast_int_to_ptr"
        )),
        "substring look-alike must not become a host llop; ops={ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "float2longlong")
        )),
        "look-alike path must stay a residual call; ops={ops:?}"
    );
}

#[test]
fn equal_size_int_to_float_transmute_stays_residual() {
    let llbc = transmute_lowering_fixture();
    let graph =
        super::lower_function(&llbc, "transmute_u64_to_f64").expect("lower u64-to-f64 transmute");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();

    assert!(
        !ops.iter()
            .any(|op| { matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as") })
    );
    assert_eq!(
        ops.iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["core", "intrinsics", "transmute"]
                )
            })
            .count(),
        1,
        "bank-crossing transmute must retain the residual intrinsic call"
    );
}

#[test]
fn equal_size_rawptr_to_fnptr_transmute_lowers_to_cast_ptr_to_int() {
    // `history.getkind`: data pointer is `ref`, `Ptr(FuncType)` is
    // `int`.  Equal-size transmute is `lltype.cast_ptr_to_int`
    // (`rbuiltin.py` `rtype_cast_ptr_to_int`).
    let llbc = transmute_lowering_fixture();
    let graph = super::lower_function(&llbc, "transmute_rawptr_to_fnptr")
        .expect("lower rawptr-to-fnptr transmute");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();

    assert!(
        !ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["core", "intrinsics", "transmute"]
            )
        }),
        "ptr↔int transmute must not stay a residual intrinsic: {ops:?}"
    );
    assert!(
        ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments
                    == &[
                        "rpython",
                        "rtyper",
                        "lltypesystem",
                        "lltype",
                        "cast_ptr_to_int",
                    ]
            )
        }),
        "expected cast_ptr_to_int host call, ops={ops:?}"
    );
}

pub(in crate::front::mir) fn fixture_span() -> serde_json::Value {
    serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 1}
    }})
}

pub(in crate::front::mir) fn fixture_item_meta(name: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "span": fixture_span(),
        "source_text": null,
        "attr_info": {
            "attributes": [],
            "inline": null,
            "rename": null,
            "public": true
        },
        "is_local": true
    })
}

pub(in crate::front::mir) fn ident_path(path: &[&str]) -> serde_json::Value {
    serde_json::Value::Array(
        path.iter()
            .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
            .collect(),
    )
}

pub(in crate::front::mir) fn transparent_layout() -> serde_json::Value {
    serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 8,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0]}],
            "repr": {"repr_algo": "Rust", "transparent": true}
        }
    }])
}

pub(in crate::front::mir) fn struct_decl(
    def_id: u64,
    name: serde_json::Value,
    fields: serde_json::Value,
    transparent: bool,
) -> serde_json::Value {
    let mut decl = serde_json::json!({
        "def_id": def_id,
        "item_meta": fixture_item_meta(name),
        "kind": {"Struct": fields}
    });
    if transparent {
        decl["layout"] = transparent_layout();
    }
    decl
}

pub(in crate::front::mir) fn i64_field() -> serde_json::Value {
    serde_json::json!([{
        "name": null,
        "ty": {"Scalar": {"Integer": {"Signed": "I64"}}},
        "attr_info": null
    }])
}

/// `fn f<T>(x: &T)` stays generic on the declaration. The call
/// instantiates `T` with a `repr(transparent)` scalar wrapper, so the
/// residual must be refused; the same callee with a non-transparent
/// `T` still lowers.
#[test]
fn generic_residual_borrow_of_transparent_scalar_is_refused() {
    let span = || {
        serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }})
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path
                .iter()
                .map(|segment| serde_json::json!({"Ident": [segment, 0]}))
                .collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let struct_decl = |def_id: u64, leaf: &str, transparent: bool| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta(&["fixture", leaf], true),
            "kind": {"Struct": [{
                "name": null,
                "ty": {"Scalar": {"Integer": {"Signed": "I64"}}},
                "attr_info": null
            }]},
            "layout": [{
                "key": "fixture-target",
                "value": {
                    "size": 8,
                    "align": 8,
                    "variant_layouts": [{"field_offsets": [0]}],
                    "repr": {"repr_algo": "Rust", "transparent": transparent}
                }
            }]
        })
    };
    let adt = |def_id: u64| {
        serde_json::json!({
            "Adt": {"id": def_id, "generics": {"types": []}}
        })
    };
    let borrowed = |inner: serde_json::Value| serde_json::json!({"Ref": ["_", inner, "Shared"]});
    let type_var = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let unit = serde_json::json!({"Tuple": []});
    let caller = |def_id: u64, name: &str, type_arg: serde_json::Value| {
        let arg_ty = borrowed(type_arg.clone());
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta(&["fixture", name], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [arg_ty.clone()],
                "output": unit.clone()
            },
            "body": {
                "Unstructured": {
                    "span": span(),
                    "locals": {
                        "arg_count": 1,
                        "locals": [
                            {"index": 0, "name": null, "span": span(), "ty": unit.clone()},
                            {"index": 1, "name": "value", "span": span(), "ty": arg_ty.clone()}
                        ]
                    },
                    "body": [
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {
                                "span": span(),
                                "kind": {
                                    "Call": {
                                        "call": {
                                            "func": {
                                                "Regular": {
                                                    "kind": {"Fun": 2},
                                                    "generics": {
                                                        "regions": [],
                                                        "types": [type_arg],
                                                        "const_generics": [],
                                                        "trait_refs": []
                                                    }
                                                }
                                            },
                                            "args": [{
                                                "Copy": {
                                                    "kind": {"Local": 1},
                                                    "ty": arg_ty.clone()
                                                }
                                            }],
                                            "dest": {"kind": {"Local": 0}, "ty": unit.clone()}
                                        },
                                        "target": 2,
                                        "on_unwind": 1
                                    }
                                }
                            }
                        },
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span(), "kind": "UnwindResume"}
                        },
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span(), "kind": "Return"}
                        }
                    ]
                }
            }
        })
    };
    let callee = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["fixture", "f"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [borrowed(type_var)],
            "output": unit
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                struct_decl(0, "Word", true),
                struct_decl(1, "Plain", false)
            ],
            "fun_decls": [
                caller(0, "pass_word", adt(0)),
                caller(1, "pass_plain", adt(1)),
                callee
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");

    let refused = super::lower_function(&llbc, "pass_word")
        .expect_err("borrow of a transparent scalar wrapper must be refused");
    let refused = refused.to_string();
    assert!(
        refused.contains("repr(transparent) scalar"),
        "refusal must name the scalar borrow, got {refused}"
    );

    let graph =
        super::lower_function(&llbc, "pass_plain").expect("a non-transparent T must still lower");
    let ops = graph_ops(&graph);
    assert!(
        call_leafs(&ops).iter().any(|leaf| leaf == "f"),
        "non-transparent instantiation must stay a residual call; ops={ops:?}"
    );
}

pub(in crate::front::mir) fn ptr_ty() -> serde_json::Value {
    serde_json::json!({"RawPtr": [usize_ty(), "Const"]})
}

fn opaque_fun(def_id: u64, path: &[&str]) -> serde_json::Value {
    let ty = ptr_ty();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": true, "inputs": [ty], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ty},
                {"index": 1, "name": "p", "span": span_json(), "ty": ty}
            ]},
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn chars_offset_global(def_id: u64, leaf: &str) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(
            &["pyre_object", "lowlevel_string", leaf],
            "",
            true
        ),
        "global_kind": "NamedConst",
        "ty": usize_ty(),
        "value": {"Value": [0, [{"Call": [{"kind": {"Fun": 0}, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}, []]}, usize_ty()]]}
    })
}

/// `from_raw_parts((base as *const u8).add(OFFSET), len(base))`.
/// `len_local` selects which argument the length call peels to.
fn str_chars_view_fun(offset_leaf: &str, len_local: u64) -> Llbc {
    let ty = ptr_ty();
    let usize_t = usize_ty();
    let cast = |src: u64, dst: u64| {
        serde_json::json!({"Assign": [
            {"kind": {"Local": dst}, "ty": ty},
            {"UnaryOp": [
                {"Cast": {"RawPtr": [ty, ty]}},
                {"Move": {"kind": {"Local": src}, "ty": ty}}
            ]}
        ]})
    };
    let copy = |src: u64, dst: u64, slot_ty: serde_json::Value| {
        serde_json::json!({"Assign": [
            {"kind": {"Local": dst}, "ty": slot_ty},
            {"Use": [{"Copy": {"kind": {"Local": src}, "ty": slot_ty}}, "No"]}
        ]})
    };
    let call =
        |fun: u64, args: serde_json::Value, dest: u64, dest_ty: serde_json::Value, target: u64| {
            serde_json::json!({"Call": {"call": {
                "func": {"Regular": {
                    "kind": {"Fun": fun},
                    "generics": empty_generics()
                }},
                "args": args,
                "dest": {"kind": {"Local": dest}, "ty": dest_ty}
            }, "target": target, "on_unwind": 4}})
        };
    let stmt = |kind: serde_json::Value| serde_json::json!({"span": span_json(), "kind": kind});
    let viewer = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(
            &["fixture", "not_the_enclosing_name"],
            "",
            true
        ),
        "signature": {"is_unsafe": true, "inputs": [ty, ty], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 2, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ty},
                {"index": 1, "name": "base", "span": span_json(), "ty": ty},
                {"index": 2, "name": "other", "span": span_json(), "ty": ty},
                {"index": 3, "name": null, "span": span_json(), "ty": ty},
                {"index": 4, "name": null, "span": span_json(), "ty": ty},
                {"index": 5, "name": "len", "span": span_json(), "ty": usize_t},
                {"index": 6, "name": null, "span": span_json(), "ty": ty},
                {"index": 7, "name": null, "span": span_json(), "ty": ty},
                {"index": 8, "name": null, "span": span_json(), "ty": ty},
                {"index": 9, "name": null, "span": span_json(), "ty": usize_t},
                {"index": 10, "name": null, "span": span_json(), "ty": ty}
            ]},
            "body": [
                {
                    "statements": [stmt(copy(len_local, 3, ty.clone())), stmt(cast(3, 4))],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": call(
                        1,
                        serde_json::json!([{"Move": {"kind": {"Local": 4}, "ty": ty}}]),
                        5,
                        usize_t.clone(),
                        1
                    )}
                },
                {
                    "statements": [stmt(copy(1, 6, ty.clone())), stmt(cast(6, 7))],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": call(
                        2,
                        serde_json::json!([
                            {"Move": {"kind": {"Local": 7}, "ty": ty}},
                            {"Copy": {"kind": {"Global": {
                                "generics": empty_generics(),
                                "id": 1
                            }}, "ty": usize_t}}
                        ]),
                        8,
                        ty.clone(),
                        2
                    )}
                },
                {
                    "statements": [stmt(copy(5, 9, usize_t.clone()))],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": call(
                        3,
                        serde_json::json!([
                            {"Move": {"kind": {"Local": 8}, "ty": ty}},
                            {"Move": {"kind": {"Local": 9}, "ty": usize_t}}
                        ]),
                        10,
                        ty.clone(),
                        3
                    )}
                },
                {
                    "statements": [stmt(serde_json::json!({"Assign": [
                        {"kind": {"Local": 0}, "ty": ty},
                        {"Use": [{"Move": {"kind": {"Local": 10}, "ty": ty}}, "No"]}
                    ]}))],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }
            ]
        }}
    });
    const_artifact(
        "fixture",
        serde_json::json!([
            viewer,
            opaque_fun(
                1,
                &["pyre_object", "lowlevel_string", "bh_lowlevel_string_len"]
            ),
            opaque_fun(2, &["core", "ptr", "const_ptr", "<Impl>", "add"]),
            opaque_fun(3, &["core", "slice", "raw", "from_raw_parts"])
        ]),
        serde_json::json!([null, chars_offset_global(1, offset_leaf)]),
    )
}

fn graph_calls_leaf(graph: &crate::model::FunctionGraph, leaf: &str) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: crate::model::CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|seg| seg == leaf)
            )
        })
}

#[test]
fn str_chars_view_from_raw_parts_aliases_same_base() {
    let llbc = str_chars_view_fun("LOWLEVEL_STRING_CHARS_OFFSET", 1);
    let graph =
        super::lower_function(&llbc, "fixture::not_the_enclosing_name").expect("chars view lowers");
    assert!(
        !graph_calls_leaf(&graph, "from_raw_parts"),
        "same-base chars view must alias the STR, graph: {graph:#?}"
    );
    let returned = graph.blocks.iter().find_map(|block| {
        block.exits.iter().find_map(|exit| {
            (exit.target == graph.returnblock).then(|| {
                exit.args.iter().find_map(|arg| match arg {
                    crate::model::LinkArg::Value(var) => Some(var.id()),
                    _ => None,
                })
            })?
        })
    });
    let input_id = graph
        .blocks
        .first()
        .and_then(|block| block.inputargs.first())
        .map(|var| var.id());
    let returned_id = returned.expect("the chars view returns a value");
    let input_id = input_id.expect("the STR base is a block argument");
    let threaded = graph.blocks.iter().any(|block| {
        block.exits.iter().any(|exit| {
            exit.args
                .iter()
                .any(|arg| matches!(arg, crate::model::LinkArg::Value(var) if var.id() == input_id))
                && graph.blocks.iter().any(|target| {
                    target.id == exit.target
                        && target.inputargs.iter().any(|arg| arg.id() == returned_id)
                })
        })
    }) || returned_id == input_id;
    assert!(
        threaded,
        "the slice destination must be the STR base (input {input_id}, return {returned_id})"
    );
}

#[test]
fn str_chars_view_from_raw_parts_keeps_other_shapes() {
    let other_offset = str_chars_view_fun("LOWLEVEL_STR_BASE_SIZE", 1);
    let graph = super::lower_function(&other_offset, "fixture::not_the_enclosing_name")
        .expect("other offset still lowers");
    assert!(
        graph_calls_leaf(&graph, "from_raw_parts"),
        "a different offset must not drop its length"
    );
    let other_base = str_chars_view_fun("LOWLEVEL_STRING_CHARS_OFFSET", 2);
    let graph = super::lower_function(&other_base, "fixture::not_the_enclosing_name")
        .expect("different base still lowers");
    assert!(
        graph_calls_leaf(&graph, "from_raw_parts"),
        "a length from a different base must not drop n"
    );
}

/// `p.add/sub(LOWLEVEL_STRING_CHARS_OFFSET)` with no string-header peel.
fn ptr_offset_chars_const_fun(
    caller_name: &str,
    callee_path: &[&str],
    ptr_ty: serde_json::Value,
) -> Llbc {
    let usize_t = usize_ty();
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", caller_name], "", true),
        "signature": {"is_unsafe": true, "inputs": [ptr_ty], "output": ptr_ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ptr_ty},
                {"index": 1, "name": "p", "span": span_json(), "ty": ptr_ty}
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": empty_generics()
                        }},
                        "args": [
                            {"Copy": {"kind": {"Local": 1}, "ty": ptr_ty}},
                            {"Copy": {"kind": {"Global": {
                                "generics": empty_generics(),
                                "id": 1
                            }}, "ty": usize_t}}
                        ],
                        "dest": {"kind": {"Local": 0}, "ty": ptr_ty}
                    }, "target": 2, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta_json(callee_path, "", false),
        "signature": {
            "is_unsafe": true,
            "inputs": [ptr_ty, usize_t],
            "output": ptr_ty
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [caller, callee],
            "global_decls": [null, chars_offset_global(1, "LOWLEVEL_STRING_CHARS_OFFSET")],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("chars-offset add fixture parses")
}

fn u8_raw_ptr_ty(mutability: &str) -> serde_json::Value {
    serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, mutability]
    })
}

fn graph_has_direct_ptradd(graph: &crate::model::FunctionGraph) -> bool {
    graph_ops(graph)
        .iter()
        .any(|op| is_lltype_direct_ptradd(op))
}

#[test]
fn u8_add_of_string_chars_offset_stays_direct_ptradd() {
    let ptr_ty = u8_raw_ptr_ty("Mut");
    let llbc = ptr_offset_chars_const_fun(
        "add_u8_chars_offset",
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "add_u8_chars_offset")
        .expect("lower *mut u8::add of LOWLEVEL_STRING_CHARS_OFFSET");
    assert!(
        graph_has_direct_ptradd(&graph),
        "a u8 pointer add of the chars offset keeps its displacement; graph={graph:#?}"
    );
}

#[test]
fn u8_sub_of_string_chars_offset_stays_direct_ptradd() {
    let ptr_ty = u8_raw_ptr_ty("Mut");
    let llbc = ptr_offset_chars_const_fun(
        "sub_u8_chars_offset",
        &["core", "ptr", "mut_ptr", "<Impl>", "sub"],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "sub_u8_chars_offset")
        .expect("lower *mut u8::sub of LOWLEVEL_STRING_CHARS_OFFSET");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "a u8 pointer sub of the chars offset keeps its displacement; ops={ops:?}"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "neg")),
        "the sub must negate the count; ops={ops:?}"
    );
}

#[test]
fn i64_add_of_string_chars_offset_stays_direct_ptradd() {
    let ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Const"]
    });
    let llbc = ptr_offset_chars_const_fun(
        "add_i64_chars_offset",
        &["core", "ptr", "const_ptr", "<Impl>", "add"],
        ptr_ty,
    );
    let graph = super::lower_function(&llbc, "add_i64_chars_offset")
        .expect("lower *const i64::add of LOWLEVEL_STRING_CHARS_OFFSET");
    assert!(
        graph_has_direct_ptradd(&graph),
        "a non-byte pointer add of the chars offset keeps its displacement; graph={graph:#?}"
    );
}

fn header_as_u8_add_chars_offset_fun(header_path: &[&str]) -> Llbc {
    let u8_ptr = u8_raw_ptr_ty("Const");
    let header_ptr = serde_json::json!({
        "RawPtr": [{
            "Adt": {
                "id": 0,
                "generics": empty_generics()
            }
        }, "Const"]
    });
    let usize_t = usize_ty();
    let isize_t = serde_json::json!({"Scalar": {"Integer": {"Signed": "Isize"}}});
    let header_adt = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(header_path, "", true),
        "kind": {"Struct": [
            {"name": "hash", "ty": isize_t, "attr_info": null},
            {"name": "length", "ty": usize_t, "attr_info": null}
        ]},
        "layout": null
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "header_chars_view"], "", true),
        "signature": {"is_unsafe": true, "inputs": [header_ptr], "output": u8_ptr},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": u8_ptr},
                {"index": 1, "name": "s", "span": span_json(), "ty": header_ptr},
                {"index": 2, "name": null, "span": span_json(), "ty": u8_ptr}
            ]},
            "body": [
                {
                    "statements": [{
                        "span": span_json(),
                        "kind": {"Assign": [
                            {"kind": {"Local": 2}, "ty": u8_ptr},
                            {"UnaryOp": [
                                {"Cast": {"RawPtr": [header_ptr, u8_ptr]}},
                                {"Move": {"kind": {"Local": 1}, "ty": header_ptr}}
                            ]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": empty_generics()
                        }},
                        "args": [
                            {"Move": {"kind": {"Local": 2}, "ty": u8_ptr}},
                            {"Copy": {"kind": {"Global": {
                                "generics": empty_generics(),
                                "id": 1
                            }}, "ty": usize_t}}
                        ],
                        "dest": {"kind": {"Local": 0}, "ty": u8_ptr}
                    }, "target": 2, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta_json(
            &["core", "ptr", "const_ptr", "<Impl>", "add"],
            "",
            false
        ),
        "signature": {
            "is_unsafe": true,
            "inputs": [u8_ptr, usize_t],
            "output": u8_ptr
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [header_adt],
            "fun_decls": [caller, callee],
            "global_decls": [null, chars_offset_global(1, "LOWLEVEL_STRING_CHARS_OFFSET")],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes())
        .expect("header-as-u8 chars-offset add fixture parses")
}

/// `fn f(s: *mut Utf8Str) -> usize { unsafe { (*s).length } }` —
/// `ll_stringslice_startstop`'s `(*s1).length`. A `*mut Utf8Str` is
/// rstr `Ptr(STR)` (`rpython/rtyper/lltypesystem/rstr.py` `STR`), so
/// the read is `LLHelpers.ll_strlen` and must not recast the STR as a
/// Utf8Str class instance.
#[test]
fn raw_utf8str_length_read_lowers_without_a_class_cast() {
    use crate::model::OpKind;

    let utf8_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let utf8_ptr = serde_json::json!({"RawPtr": [utf8_adt.clone(), "Mut"]});
    let n = window_usize();
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": n.clone(), "attr_info": null}
        ]}
    });
    let length_place = window_place(
        serde_json::json!({"Projection": [
            window_place(
                serde_json::json!({"Projection": [
                    window_place(serde_json::json!({"Local": 1}), &utf8_ptr),
                    "Deref"
                ]}),
                &utf8_adt
            ),
            {"Field": [null, 1]}
        ]}),
        &n,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "raw_utf8str_length"], true),
        "signature": {"is_unsafe": true, "inputs": [utf8_ptr], "output": n},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &n),
                window_local(1, &utf8_ptr)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        window_place(serde_json::json!({"Local": 0}), &n),
                        {"Use": [{"Copy": length_place}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let llbc = window_llbc(vec![utf8], vec![caller]);
    let graph =
        super::lower_function(&llbc, "raw_utf8str_length").expect("lower (*mut Utf8Str).length");
    let ops = window_graph_ops(&graph);
    let utf8str_casts: Vec<&str> = ops
        .iter()
        .filter_map(|op| crate::model::cast_instance_root(&op.kind))
        .filter(|root| majit_ir::descr::class_root_leaf(root) == "Utf8Str")
        .collect();
    assert!(
        utf8str_casts.is_empty(),
        "(*s).length of *mut Utf8Str must not recast STR as a class; casts={utf8str_casts:?} graph={}",
        graph.dump()
    );
    let length_read = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. }
                if field.name == "length"
                    && field.owner_root.as_deref().is_some_and(|root| {
                        majit_ir::descr::class_root_leaf(root) == "Utf8Str"
                    })
        )
    });
    assert!(
        length_read,
        "STR length is FieldRead length (`LLHelpers.ll_strlen`); graph={}",
        graph.dump()
    );
}

/// `fn f(h: &mut Holder, p: *mut UnicodeValueStorage) { h.value = p; }`
/// — the field-write dest overlay recasts the STR pointer onto the
/// declared class so `setattr("value")` unions with FORCE String.
#[test]
fn raw_unicode_value_storage_field_write_emits_a_class_cast() {
    use crate::model::OpKind;

    let storage_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let holder_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let storage_ptr = serde_json::json!({"RawPtr": [storage_adt.clone(), "Mut"]});
    let holder_ref = serde_json::json!({"Ref": ["Erased", holder_adt.clone(), "Mut"]});
    let n = window_usize();
    let storage = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(
            &["pyre_object", "unicodeobject", "UnicodeValueStorage"],
            true
        ),
        "kind": {"Struct": [
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": n.clone(), "attr_info": null}
        ]}
    });
    let holder = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["fixture", "Holder"], true),
        "kind": {"Struct": [{
            "name": "value",
            "ty": storage_ptr.clone(),
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        }]}
    });
    let field_place = window_place(
        serde_json::json!({"Projection": [
            window_place(
                serde_json::json!({"Projection": [
                    window_place(serde_json::json!({"Local": 1}), &holder_ref),
                    "Deref"
                ]}),
                &holder_adt
            ),
            {"Field": [null, 0]}
        ]}),
        &storage_ptr,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "store_unicode_value"], true),
        "signature": {
            "is_unsafe": true,
            "inputs": [holder_ref.clone(), storage_ptr.clone()],
            "output": storage_ptr.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &storage_ptr),
                window_local(1, &holder_ref),
                window_local(2, &storage_ptr)
            ]},
            "body": [{
                "statements": [
                    {
                        "span": window_span(),
                        "kind": {"Assign": [
                            field_place,
                            {"Use": [{"Copy": window_place(
                                serde_json::json!({"Local": 2}),
                                &storage_ptr
                            )}, "No"]}
                        ]}
                    },
                    {
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &storage_ptr),
                            {"Use": [{"Copy": window_place(
                                serde_json::json!({"Local": 2}),
                                &storage_ptr
                            )}, "No"]}
                        ]}
                    }
                ],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let llbc = window_llbc(vec![storage, holder], vec![caller]);
    let graph = super::lower_function(&llbc, "store_unicode_value")
        .expect("lower (*h).value = p for *mut UnicodeValueStorage");
    let ops = window_graph_ops(&graph);
    let storage_casts: Vec<&str> = ops
        .iter()
        .filter_map(|op| crate::model::cast_instance_root(&op.kind))
        .filter(|root| majit_ir::descr::class_root_leaf(root) == "UnicodeValueStorage")
        .collect();
    assert!(
        !storage_casts.is_empty(),
        "field write of *mut UnicodeValueStorage must recast onto the class; graph={}",
        graph.dump()
    );
    let value_write = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. } if field.name == "value"
        )
    });
    assert!(
        value_write,
        "the store is FieldWrite value; graph={}",
        graph.dump()
    );
}

#[test]
fn utf8str_as_u8_add_of_chars_offset_aliases_the_header() {
    let llbc = header_as_u8_add_chars_offset_fun(&["pyre_object", "unicodeobject", "Utf8Str"]);
    let graph = super::lower_function(&llbc, "header_chars_view")
        .expect("lower (Utf8Str as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)");
    assert!(
        !graph_has_direct_ptradd(&graph),
        "s.chars is the STR, not a displaced byte pointer; graph={graph:#?}"
    );
    assert!(
        !call_leafs(&graph_ops(&graph))
            .iter()
            .any(|leaf| *leaf == "add"),
        "the header chars add must not residualize; graph={graph:#?}"
    );
}

/// `(s as *const Utf8Str as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)`
/// for `s: &Utf8Str` is still `s.chars` (`ll_stritem_nonneg`).
fn header_ref_as_u8_add_chars_offset_fun(header_path: &[&str]) -> Llbc {
    let u8_ptr = u8_raw_ptr_ty("Const");
    let header_adt_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": empty_generics()
        }
    });
    let header_ptr = serde_json::json!({
        "RawPtr": [header_adt_ty, "Const"]
    });
    let header_ref = serde_json::json!({
        "Ref": ["Erased", header_adt_ty, "Shared"]
    });
    let usize_t = usize_ty();
    let isize_t = serde_json::json!({"Scalar": {"Integer": {"Signed": "Isize"}}});
    let header_adt = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(header_path, "", true),
        "kind": {"Struct": [
            {"name": "hash", "ty": isize_t, "attr_info": null},
            {"name": "length", "ty": usize_t, "attr_info": null}
        ]},
        "layout": null
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "header_chars_view"], "", true),
        "signature": {"is_unsafe": true, "inputs": [header_ref], "output": u8_ptr},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": u8_ptr},
                {"index": 1, "name": "s", "span": span_json(), "ty": header_ref},
                {"index": 2, "name": null, "span": span_json(), "ty": header_ptr},
                {"index": 3, "name": null, "span": span_json(), "ty": u8_ptr}
            ]},
            "body": [
                {
                    "statements": [
                        {
                            "span": span_json(),
                            "kind": {"Assign": [
                                {"kind": {"Local": 2}, "ty": header_ptr},
                                {"UnaryOp": [
                                    {"Cast": {"RawPtr": [header_ref, header_ptr]}},
                                    {"Move": {"kind": {"Local": 1}, "ty": header_ref}}
                                ]}
                            ]}
                        },
                        {
                            "span": span_json(),
                            "kind": {"Assign": [
                                {"kind": {"Local": 3}, "ty": u8_ptr},
                                {"UnaryOp": [
                                    {"Cast": {"RawPtr": [header_ptr, u8_ptr]}},
                                    {"Move": {"kind": {"Local": 2}, "ty": header_ptr}}
                                ]}
                            ]}
                        }
                    ],
                    "terminator": {"span": span_json(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": empty_generics()
                        }},
                        "args": [
                            {"Move": {"kind": {"Local": 3}, "ty": u8_ptr}},
                            {"Copy": {"kind": {"Global": {
                                "generics": empty_generics(),
                                "id": 1
                            }}, "ty": usize_t}}
                        ],
                        "dest": {"kind": {"Local": 0}, "ty": u8_ptr}
                    }, "target": 2, "on_unwind": 1}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta_json(
            &["core", "ptr", "const_ptr", "<Impl>", "add"],
            "",
            false
        ),
        "signature": {
            "is_unsafe": true,
            "inputs": [u8_ptr, usize_t],
            "output": u8_ptr
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [header_adt],
            "fun_decls": [caller, callee],
            "global_decls": [null, chars_offset_global(1, "LOWLEVEL_STRING_CHARS_OFFSET")],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes())
        .expect("header-ref-as-u8 chars-offset add fixture parses")
}

#[test]
fn utf8str_ref_as_u8_add_of_chars_offset_aliases_the_header() {
    let llbc = header_ref_as_u8_add_chars_offset_fun(&["pyre_object", "unicodeobject", "Utf8Str"]);
    let graph = super::lower_function(&llbc, "header_chars_view").expect(
        "lower (&Utf8Str as *const Utf8Str as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)",
    );
    assert!(
        !graph_has_direct_ptradd(&graph),
        "s.chars from &Utf8Str is the STR, not a displaced byte pointer; graph={graph:#?}"
    );
    assert!(
        !call_leafs(&graph_ops(&graph))
            .iter()
            .any(|leaf| *leaf == "add"),
        "the header-ref chars add must not residualize; graph={graph:#?}"
    );
}

#[test]
fn other_module_utf8str_as_u8_add_of_chars_offset_stays_direct_ptradd() {
    let llbc = header_as_u8_add_chars_offset_fun(&["other_crate", "Utf8Str"]);
    let graph = super::lower_function(&llbc, "header_chars_view")
        .expect("lower (other Utf8Str as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)");
    assert!(
        graph_has_direct_ptradd(&graph),
        "a same-leaf Utf8Str outside pyre_object::unicodeobject is not rstr STR; graph={graph:#?}"
    );
}

pub(in crate::front::mir) fn span_json() -> serde_json::Value {
    serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 16}
    }})
}

pub(in crate::front::mir) fn item_meta_json(
    path: &[&str],
    source_text: &str,
    is_local: bool,
) -> serde_json::Value {
    serde_json::json!({
        "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
        "span": span_json(),
        "source_text": source_text,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
        "is_local": is_local
    })
}

pub(in crate::front::mir) fn u32_ty() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}})
}

pub(in crate::front::mir) fn usize_ty() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
}

pub(in crate::front::mir) fn empty_generics() -> serde_json::Value {
    serde_json::json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []})
}

pub(in crate::front::mir) fn init_fun(
    def_id: u64,
    path: &[&str],
    body: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": u32_ty()},
        "src": {"GlobalInitializer": {"id": def_id}},
        "body": body
    })
}

pub(in crate::front::mir) fn named_const_global(
    def_id: u64,
    path: &[&str],
    source_text: &str,
    is_local: bool,
    global_kind: &str,
    init: u64,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, source_text, is_local),
        "global_kind": global_kind,
        "ty": u32_ty(),
        "value": {"Value": [0, [{"Call": [{"kind": {"Fun": init}, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}, []]}, u32_ty()]]}
    })
}

pub(in crate::front::mir) fn reader_fun(path: &[&str], global_id: u64) -> serde_json::Value {
    let ty = u32_ty();
    serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": ty},
                        {"Use": [{"Copy": {
                            "kind": {"Global": {"generics": empty_generics(), "id": global_id}},
                            "ty": ty
                        }}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

pub(in crate::front::mir) fn const_artifact(
    crate_name: &str,
    fun_decls: serde_json::Value,
    global_decls: serde_json::Value,
) -> Llbc {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": crate_name,
            "type_decls": [],
            "fun_decls": fun_decls,
            "global_decls": global_decls,
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("const fixture Llbc parses")
}

fn char_scalar_ty() -> serde_json::Value {
    serde_json::json!({"Scalar": "Char"})
}

fn char_const_expr(ch: &str) -> serde_json::Value {
    serde_json::json!([{"Char": ch}, char_scalar_ty()])
}

fn char_init_body(ch: &str) -> serde_json::Value {
    let ty = char_scalar_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": ty}
        ]},
        "body": [{
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 0}, "ty": ty},
                    {"Use": [{"Const": char_const_expr(ch)}, "No"]}
                ]}
            }],
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn char_init_fun(def_id: u64, path: &[&str], body: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": char_scalar_ty()},
        "src": {"GlobalInitializer": {"id": def_id}},
        "body": body
    })
}

fn char_named_const_global(
    def_id: u64,
    path: &[&str],
    source_text: &str,
    init: u64,
) -> serde_json::Value {
    let ty = char_scalar_ty();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, source_text, true),
        "global_kind": "NamedConst",
        "ty": ty,
        "value": {"Value": [0, [{"Call": [{"kind": {"Fun": init}, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}, []]}, ty]]}
    })
}

fn char_reader_fun(path: &[&str], global_id: u64) -> serde_json::Value {
    let ty = char_scalar_ty();
    serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": ty},
                        {"Use": [{"Copy": {
                            "kind": {"Global": {"generics": empty_generics(), "id": global_id}},
                            "ty": ty
                        }}, "No"]}
                    ]}
                }],
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn char_literal_reader_fun(path: &[&str], ch: &str) -> serde_json::Value {
    let ty = char_scalar_ty();
    serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": ty},
                        {"Use": [{"Const": char_const_expr(ch)}, "No"]}
                    ]}
                }],
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn unichr_ops(graph: &crate::model::FunctionGraph) -> Vec<&SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.first().map(String::as_str)
                    == Some(crate::runtime_names::shims::UNICHR)
            )
        })
        .collect()
}

#[test]
fn named_char_const_and_char_literal_emit_unichr() {
    let path = ["fixture", "CHAR_CONST"];
    let llbc = const_artifact(
        "fixture",
        serde_json::json!([
            char_reader_fun(&["fixture", "read_named"], 1),
            char_init_fun(1, &path, char_init_body(">")),
            char_literal_reader_fun(&["fixture", "read_lit"], ">")
        ]),
        serde_json::json!([
            null,
            char_named_const_global(1, &path, "const CHAR_CONST: char = '>';", 1)
        ]),
    );
    let named = super::lower_function(&llbc, "read_named").expect("lower named char const");
    let lit = super::lower_function(&llbc, "read_lit").expect("lower char literal");
    let named_ops = unichr_ops(&named);
    let lit_ops = unichr_ops(&lit);
    assert_eq!(
        named_ops.len(),
        1,
        "named char const emits __majit_unichr: {named:#?}"
    );
    assert_eq!(
        lit_ops.len(),
        1,
        "char literal emits __majit_unichr: {lit:#?}"
    );
    let has_code = |graph: &crate::model::FunctionGraph| {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(op.kind, OpKind::ConstInt(0x3e)))
    };
    assert!(
        has_code(&named),
        "named char const is code point 0x3e: {named:#?}"
    );
    assert!(has_code(&lit), "char literal is code point 0x3e: {lit:#?}");
}

pub(in crate::front::mir) fn nullary_calls_ending(
    graph: &crate::model::FunctionGraph,
    leaf: &str,
) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if args.is_empty() && segments.last().map(String::as_str) == Some(leaf)
            )
        })
        .count()
}

/// `*mut GCREFOpaque` and `*const (Vec<*mut PyObject>, Vec<*mut PyObject>)`
/// — the kwargs unerase dest (`KwargsDictStrategy.copy` `unerase`).
pub(in crate::front::mir) fn kwargs_unerase_ptr_tys() -> (serde_json::Value, serde_json::Value) {
    let empty_g = empty_generics();
    let adt = |id: u64| {
        serde_json::json!({
            "Adt": {"id": id, "generics": empty_g.clone()}
        })
    };
    let gcref_ptr = serde_json::json!({"RawPtr": [adt(0), "Mut"]});
    let vec_item = serde_json::json!({"RawPtr": [adt(1), "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {
            "id": 2,
            "generics": {
                "regions": [],
                "types": [vec_item],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let tuple_ty = serde_json::json!({
        "Adt": {
            "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": [vec_ty.clone(), vec_ty],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let dest_ptr = serde_json::json!({"RawPtr": [tuple_ty, "Const"]});
    (gcref_ptr, dest_ptr)
}

fn kwargs_unerase_type_decls() -> Vec<serde_json::Value> {
    vec![
        type_decl_struct(0, &["majit_gc", "header", "GCREFOpaque"], vec![]),
        type_decl_struct(1, &["fixture", "PyObject"], vec![]),
        type_decl_struct(2, &["alloc", "vec", "Vec"], vec![]),
    ]
}

fn gctype_impl_row(self_id: u64) -> serde_json::Value {
    serde_json::json!({
        "impl_trait": {
            "id": 0,
            "generics": {"types": [{
                "Value": [self_id, {
                    "Adt": {
                        "id": self_id,
                        "generics": empty_generics()
                    }
                }]
            }]}
        }
    })
}

fn gctype_trait_decl() -> serde_json::Value {
    serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["majit_gc", "header", "GcType"], "", false)
    })
}

pub(in crate::front::mir) fn kwargs_unerase_cast_llbc(src_is_gcref: bool) -> Llbc {
    let (gcref_ptr, dest_ptr) = kwargs_unerase_ptr_tys();
    let src_ty = if src_is_gcref {
        gcref_ptr.clone()
    } else {
        serde_json::json!({
            "RawPtr": [{
                "Adt": {"id": 1, "generics": empty_generics()}
            }, "Mut"]
        })
    };
    let span = span_json();
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "unerase"], "", true),
        "signature": {
            "is_unsafe": true,
            "inputs": [src_ty.clone()],
            "output": dest_ptr.clone()
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span, "ty": dest_ptr.clone()},
                {"index": 1, "name": "dstorage", "span": span, "ty": src_ty.clone()}
            ]},
            "body": [{
                "statements": [{
                    "span": span,
                    "kind": {"Assign": [
                        place(0, &dest_ptr),
                        {"UnaryOp": [
                            {"Cast": {"RawPtr": [src_ty.clone(), dest_ptr.clone()]}},
                            {"Move": place(1, &src_ty)}
                        ]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let impls = if src_is_gcref {
        serde_json::json!([gctype_impl_row(0)])
    } else {
        serde_json::json!([gctype_impl_row(1)])
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": kwargs_unerase_type_decls(),
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [gctype_trait_decl()],
            "trait_impls": impls
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("kwargs unerase fixture parses")
}

#[test]
fn gcref_cast_to_tuple_of_vecs_emits_cast_instance_marker() {
    let llbc = kwargs_unerase_cast_llbc(true);
    let graph = super::lower_function(&llbc, "unerase").expect("lower GCREF unerase");
    let roots: Vec<&str> = graph_ops(&graph)
        .iter()
        .filter_map(|op| crate::model::cast_instance_root(&op.kind))
        .collect();
    assert!(
        roots
            .iter()
            .any(|root| majit_ir::descr::is_shaped_tuple_name(root)),
        "GCREF as *const (Vec, Vec) must emit __cast_instance_intrinsic; roots={roots:?}"
    );
}

#[test]
fn non_gcref_cast_to_tuple_of_vecs_aliases() {
    let llbc = kwargs_unerase_cast_llbc(false);
    let graph = super::lower_function(&llbc, "unerase").expect("lower PyObject unerase");
    let roots: Vec<&str> = graph_ops(&graph)
        .iter()
        .filter_map(|op| crate::model::cast_instance_root(&op.kind))
        .collect();
    assert!(
        roots
            .iter()
            .all(|root| !majit_ir::descr::is_shaped_tuple_name(root)),
        "a non-GCREF source must keep today's alias; roots={roots:?}"
    );
}

/// Anchor the unerase marker to real `kwargs_storage`. Ignored by
/// default (loads pyre-object.ullbc).
#[test]
#[ignore]
fn kwargs_storage_unerase_emits_cast_instance() {
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-object.ullbc");
    let graph = super::lower_function(&llbc, "kwargs_storage").expect("lower kwargs_storage");
    let roots: Vec<&str> = graph_ops(&graph)
        .iter()
        .filter_map(|op| crate::model::cast_instance_root(&op.kind))
        .collect();
    assert!(
        roots
            .iter()
            .any(|root| majit_ir::descr::is_shaped_tuple_name(root)),
        "kwargs_storage unerase must emit Tuple<Vec,Vec> marker; roots={roots:?}"
    );
}

pub(in crate::front::mir) fn fixture_ty(v: serde_json::Value) -> TyRef {
    serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [0, v]
    }))
    .expect("fixture TyRef parses")
}

pub(in crate::front::mir) fn adt_ty(def_id: u64) -> TyRef {
    fixture_ty(serde_json::json!({
        "Adt": {"id": def_id, "generics": {"types": []}}
    }))
}

pub(in crate::front::mir) fn type_decl_meta(path: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
        "span": {"data": {
            "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
        }},
        "source_text": null,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
        "is_local": true
    })
}

/// `fn as usize` is Int-banked (`history.getkind` of `Ptr(FuncType)`),
/// but `rbuiltin.py gen_cast` still starts from a Ptr origin: Signed
/// `cast_ptr_to_int` then `r_uint`. A bare `r_uint` is the integer
/// signedness flip and fails rtyping (`PtrRepr` to `IntegerRepr`).
#[test]
fn fn_ptr_as_usize_emits_cast_ptr_to_int_then_r_uint() {
    let span = serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    });
    let fn_ptr = serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [],
                "output": {
                    "Adt": {
                        "id": 0,
                        "builtin": "Tuple",
                        "generics": {"types": []}
                    }
                },
                "is_unsafe": false
            }
        }
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let local = |index: u64, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": null,
            "span": span,
            "ty": ty
        })
    };
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["fn_as_usize", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [fn_ptr.clone()],
            "output": usize_ty.clone()
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": [
                local(0, &usize_ty),
                local(1, &fn_ptr)
            ]},
            "body": [{
                "statements": [{
                    "span": span,
                    "kind": {"Assign": [
                        place(0, &usize_ty),
                        {"UnaryOp": [
                            {"Cast": {"Scalar": [fn_ptr.clone(), usize_ty.clone()]}},
                            {"Copy": place(1, &fn_ptr)}
                        ]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = super::lower_function(&llbc, "fn_as_usize").expect("lower fn_as_usize");

    let calls: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                result_ty,
            } => Some((
                segments.last().map(String::as_str),
                op.result.as_ref(),
                args.as_slice(),
                result_ty,
            )),
            _ => None,
        })
        .collect();
    let cast_idx = calls
        .iter()
        .position(|(leaf, _, _, _)| *leaf == Some("cast_ptr_to_int"))
        .expect("fn as usize emits cast_ptr_to_int");
    let retype_idx = calls
        .iter()
        .position(|(leaf, _, _, _)| *leaf == Some("r_uint"))
        .expect("fn as usize emits r_uint");
    assert_eq!(
        retype_idx,
        cast_idx + 1,
        "rbuiltin.py gen_cast is cast_ptr_to_int then r_uint, got {calls:?}"
    );
    let (_, signed, cast_args, cast_ty) = &calls[cast_idx];
    assert_eq!(**cast_ty, ValueType::Int);
    let (_, result, retype_args, retype_ty) = &calls[retype_idx];
    assert_eq!(**retype_ty, ValueType::Unsigned);
    assert_eq!(
        retype_args.first().and_then(LinkArg::as_variable),
        signed.as_deref(),
        "r_uint retypes the Signed cast_ptr_to_int result"
    );
    assert!(result.is_some());
    assert_eq!(cast_args.len(), 1);
}

pub(in crate::front::mir) fn vec_decl_json() -> serde_json::Value {
    let name: Vec<serde_json::Value> = ["alloc", "vec", "Vec"]
        .iter()
        .map(|s| serde_json::json!({"Ident": [s, 0]}))
        .collect();
    serde_json::json!({
        "def_id": 0,
        "item_meta": {
            "name": name,
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }},
            "source_text": null,
            "attr_info": {
                "attributes": [], "inline": null, "rename": null, "public": true
            },
            "is_local": false
        },
        "kind": {"Struct": []}
    })
}

pub(in crate::front::mir) fn vec_usize_ty_json() -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [usize_ty()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

pub(in crate::front::mir) fn as_usize_cast_fixture(
    name: &str,
    src_ty: serde_json::Value,
    type_decls: Vec<serde_json::Value>,
    extra_assign: Option<(serde_json::Value, serde_json::Value, serde_json::Value)>,
) -> Llbc {
    let span = serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    });
    let usize_ty = usize_ty();
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let local = |index: u64, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": null,
            "span": span,
            "ty": ty
        })
    };
    let mut locals = vec![local(0, &usize_ty), local(1, &src_ty)];
    let mut statements = Vec::new();
    let (cast_src_ty, cast_src_index) = if let Some((tmp_ty, tmp_cast, tmp_src)) = extra_assign {
        locals.push(local(2, &tmp_ty));
        statements.push(serde_json::json!({
            "span": span,
            "kind": {"Assign": [
                place(2, &tmp_ty),
                {"UnaryOp": [tmp_cast, tmp_src]}
            ]}
        }));
        (tmp_ty, 2u64)
    } else {
        (src_ty.clone(), 1u64)
    };
    let cast_src_place = place(cast_src_index, &cast_src_ty);
    statements.push(serde_json::json!({
        "span": span,
        "kind": {"Assign": [
            place(0, &usize_ty),
            {"UnaryOp": [
                {"Cast": {"Scalar": [cast_src_ty, usize_ty.clone()]}},
                {"Copy": cast_src_place}
            ]}
        ]}
    }));
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": [name, 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [src_ty],
            "output": usize_ty
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": locals},
            "body": [{
                "statements": statements,
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
}

fn path_call_leaves(
    graph: &crate::model::FunctionGraph,
) -> Vec<(Option<&str>, Option<&Variable>, &[LinkArg], &ValueType)> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                result_ty,
            } => Some((
                segments.last().map(String::as_str),
                op.result.as_ref(),
                args.as_slice(),
                result_ty,
            )),
            _ => None,
        })
        .collect()
}

pub(in crate::front::mir) fn assert_cast_ptr_to_int_then_r_uint(
    graph: &crate::model::FunctionGraph,
    what: &str,
) {
    let calls = path_call_leaves(graph);
    let cast_idx = calls
        .iter()
        .position(|(leaf, _, _, _)| *leaf == Some("cast_ptr_to_int"))
        .unwrap_or_else(|| panic!("{what} emits cast_ptr_to_int, got {calls:?}"));
    let retype_idx = calls
        .iter()
        .position(|(leaf, _, _, _)| *leaf == Some("r_uint"))
        .unwrap_or_else(|| panic!("{what} emits r_uint, got {calls:?}"));
    assert_eq!(
        retype_idx,
        cast_idx + 1,
        "rbuiltin.py gen_cast is cast_ptr_to_int then r_uint, got {calls:?}"
    );
    let (_, signed, cast_args, cast_ty) = &calls[cast_idx];
    assert_eq!(**cast_ty, ValueType::Int);
    let (_, result, retype_args, retype_ty) = &calls[retype_idx];
    assert_eq!(**retype_ty, ValueType::Unsigned);
    assert_eq!(
        retype_args.first().and_then(LinkArg::as_variable),
        signed.as_deref(),
        "r_uint retypes the Signed cast_ptr_to_int result"
    );
    assert!(result.is_some());
    assert_eq!(cast_args.len(), 1);
}

/// `&mut Vec<T> as *mut Vec<T> as usize` is `rbuiltin.py gen_cast` of a
/// Ptr origin: Signed `cast_ptr_to_int` then `r_uint`. A bare `r_uint`
/// keeps the `SomeRustVec` annotation and fails to join an integer
/// `raw_ptradd` argument (`mergeinputargs`: Integer ∪ RustVec).
#[test]
fn vec_header_as_usize_emits_cast_ptr_to_int_then_r_uint() {
    let vec_ty = vec_usize_ty_json();
    let ref_ty = serde_json::json!({"Ref": ["_", vec_ty.clone(), "Mut"]});
    let raw_ty = serde_json::json!({"RawPtr": [vec_ty, "Mut"]});
    let span_place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let llbc = as_usize_cast_fixture(
        "vec_header_i",
        ref_ty.clone(),
        vec![vec_decl_json()],
        Some((
            raw_ty.clone(),
            serde_json::json!({"Cast": {"RawPtr": [ref_ty.clone(), raw_ty.clone()]}}),
            serde_json::json!({"Copy": span_place(1, &ref_ty)}),
        )),
    );
    let graph = super::lower_function(&llbc, "vec_header_i").expect("lower vec_header_i");
    assert_cast_ptr_to_int_then_r_uint(&graph, "&mut Vec as *mut Vec as usize");
}

pub(in crate::front::mir) fn rows(spec: &[(&str, &str)]) -> Vec<crate::front::semantic::FieldRow> {
    spec.iter()
        .map(|(n, t)| crate::front::semantic::FieldRow::named(*n, *t))
        .collect()
}

pub(in crate::front::mir) fn code_struct(def_id: u64, path: &[&str]) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "kind": {"Struct": [{
            "name": "n",
            "ty": {"Scalar": {"Integer": {"Unsigned": "U64"}}},
            "attr_info": null
        }]}
    })
}

pub(in crate::front::mir) fn llbc_with_types(
    crate_name: &str,
    types: Vec<serde_json::Value>,
    funs: Vec<serde_json::Value>,
) -> Llbc {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": crate_name,
            "type_decls": types,
            "fun_decls": funs,
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
}

pub(in crate::front::mir) fn adt_node(def_id: u64) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": def_id,
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        }
    })
}

pub(in crate::front::mir) fn option_enum(def_id: u64) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": [
                {"Ident": ["core", 0]},
                {"Ident": ["option", 0]},
                {"Ident": ["Option", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Enum": [
            {
                "name": "None",
                "fields": [],
                "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
            },
            {
                "name": "Some",
                "fields": [{"name": null, "ty": {"TypeVar": {"Bound": [0, 0]}}, "attr_info": null}],
                "discriminant": {"Scalar": {"Signed": ["Isize", "1"]}}
            }
        ]}
    })
}

pub(in crate::front::mir) fn option_adt_node(
    def_id: u64,
    arg: serde_json::Value,
) -> serde_json::Value {
    option_adt_node_from_types(def_id, vec![arg])
}

pub(in crate::front::mir) fn option_adt_node_from_types(
    def_id: u64,
    types: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": def_id,
            "generics": {
                "regions": [],
                "types": types,
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

pub(in crate::front::mir) fn option_int_arg(signed: bool) -> serde_json::Value {
    if signed {
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})
    } else {
        serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
    }
}

fn option_i64_ty() -> serde_json::Value {
    option_adt_node(0, option_int_arg(true))
}

fn option_impl_seg() -> serde_json::Value {
    serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, option_i64_ty()]},
        "kind": "InherentImplBlock"
    }}})
}

fn option_map_or_fn_item_llbc(fn_item: serde_json::Value) -> Llbc {
    let i64_ty = interior_i64();
    let opt_ty = option_i64_ty();
    let fn_id = fn_item
        .get("def_id")
        .and_then(serde_json::Value::as_u64)
        .expect("fn item def_id");
    let fn_ty = serde_json::json!({
        "FnDef": {
            "kind": {"Fun": fn_id},
            "generics": empty_generics()
        }
    });
    let map_or = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["core", 0]},
                {"Ident": ["option", 0]},
                option_impl_seg(),
                {"Ident": ["map_or", 0]}
            ],
            "span": interior_span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": false
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [opt_ty.clone(), i64_ty.clone(), fn_ty.clone()],
            "output": i64_ty.clone()
        },
        "body": "Opaque"
    });
    let caller = interior_caller(
        "map_or_fn_item",
        1,
        vec![opt_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("opt"), &opt_ty),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &opt_ty)),
                        interior_const_i64("-1", &i64_ty),
                        serde_json::json!({"Const": [{
                                "FnDef": {
                                    "kind": {"Fun": fn_id},
                                    "generics": empty_generics()
                                }
                            }, fn_ty]}),
                    ],
                    interior_place(0, &i64_ty),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(vec![], interior_return()),
        ],
    );
    llbc_with_types(
        "fixture",
        vec![option_enum(0)],
        vec![caller, map_or, fn_item],
    )
}

fn graph_has_map_or(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "map_or"
        ) || matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("map_or")
        )
    })
}

fn graph_has_fn_const(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        matches!(
            &op.kind,
            OpKind::Call { target, .. } if crate::model::fn_const_segments(target).is_some()
        )
    })
}

/// `opt.map_or(-1, i64::from)` with reflexive `From<i64> for i64`: residual
/// `map_or` and the 0-arg `__fn_const` define are gone; the Some arm is
/// identity on the payload (ordinary `is_reflexive_from`).
#[test]
fn map_or_i64_from_fn_item_is_identity_select() {
    let from = interior_opaque(
        2,
        &["core", "convert", "<Impl>", "from"],
        vec![interior_i64()],
        interior_i64(),
    );
    let llbc = option_map_or_fn_item_llbc(from);
    let graph = lower_interior(&llbc, "map_or_fn_item");
    assert!(
        !graph_has_map_or(&graph),
        "residual map_or must be rewritten; ops={:?}",
        graph_ops(&graph)
    );
    assert!(
        !graph_has_fn_const(&graph),
        "function-item define must not survive; ops={:?}",
        graph_ops(&graph)
    );
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("from")
            )),
        "reflexive From is identity, not a residual from call; ops={:?}",
        graph_ops(&graph)
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "__pos_0"
            )),
        "the Some arm still reads the payload"
    );
    assert_eq!(
        graph.blocks[graph.startblock.0].exits.len(),
        2,
        "A branches to Some/None arms"
    );
}

/// `opt.map_or(-1, plus_one)` with a local function item: residual `map_or`
/// and `__fn_const` are gone; the Some arm calls `plus_one` directly.
#[test]
fn map_or_plus_one_fn_item_calls_the_function() {
    let plus_one = interior_opaque(
        2,
        &["fixture", "plus_one"],
        vec![interior_i64()],
        interior_i64(),
    );
    let llbc = option_map_or_fn_item_llbc(plus_one);
    let graph = lower_interior(&llbc, "map_or_fn_item");
    assert!(
        !graph_has_map_or(&graph),
        "residual map_or must be rewritten; ops={:?}",
        graph_ops(&graph)
    );
    assert!(
        !graph_has_fn_const(&graph),
        "function-item define must not survive; ops={:?}",
        graph_ops(&graph)
    );
    let named: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("plus_one")
            )
        })
        .collect();
    assert_eq!(
        named.len(),
        1,
        "the Some arm calls plus_one; ops={:?}",
        graph_ops(&graph)
    );
    let OpKind::Call { args, .. } = &named[0].kind else {
        panic!("plus_one call");
    };
    assert_eq!(args.len(), 1, "the Some arm calls plus_one(payload)");
    assert_eq!(
        graph.blocks[graph.startblock.0].exits.len(),
        2,
        "A branches to Some/None arms"
    );
}

/// Anchor the `?`-operator fold to the real lowered IR of
/// `baseobjspace::lookup` (`let w_type = type(obj)?; lookup_in_type(..)`).
/// The break arm (`None => return None`) narrows the `from_residual`
/// result through a `__cast_instance_intrinsic` recast; before the break-arm
/// purity gate peeled that recast, `option_try` declined and the residual
/// `Try::branch` Method-call stayed, walling ~50 heads reaching `lookup`.
/// After: zero residual `branch` Method-call.  Ignored by default (loads
/// the ~440MB real LLBC); run with `cargo test -p majit-translate --lib
/// option_try_recast_break_arm_real -- --ignored`.
#[test]
#[ignore]
fn option_try_recast_break_arm_real_lookup() {
    use crate::model::{CallTarget, ExitCase, ExitSwitch, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "lookup").expect("lower lookup");
    let branch_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                    if name == "branch" && args.len() == 1
            )
        })
        .count();
    assert_eq!(
        branch_calls, 0,
        "lookup: residual Try::branch Method-call after the recast-break-arm fold"
    );
    // `lookup`'s `r#type(obj)?` is a niche `Option<NonNull>` `?`, rewritten
    // by `option_try` into a `ne` pointer null-test switch.  Because the
    // switch value is a `SomeBool` (`ne`), the some/none exits MUST carry
    // `ExitCase::Bool`, not `ExitCase::Const(Int)` — else rtype rejects the
    // bool exitswitch against Int cases ("not a bool: Int(1)").
    let mut niche_try_switches = 0usize;
    for b in &graph.blocks {
        let Some(ExitSwitch::Value(sw)) = &b.exitswitch else {
            continue;
        };
        let switches_on_ne = b.operations.iter().any(|op| {
            op.result.as_ref() == Some(sw)
                && matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne")
        });
        if !switches_on_ne {
            continue;
        }
        for l in &b.exits {
            assert!(
                matches!(&l.exitcase, Some(ExitCase::Bool(_))),
                "lookup: niche `?` null-test switch exit must be ExitCase::Bool, got {:?}",
                l.exitcase
            );
        }
        niche_try_switches += 1;
    }
    assert!(
        niche_try_switches >= 1,
        "lookup: expected a niche `?` `ne` null-test switch block"
    );

    // The `?` break arm returns the enclosing `Option<PyObjectRef>`'s
    // `None`.  That type is a nullable PyObject pointer, not a constructed
    // Rust `Option::None` instance.  Pin both halves of the representation:
    // no Option None ctor survives, and a null pointer narrowed to
    // PyObject is forwarded to the return block.
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor {
                        name,
                        owner_path,
                        ..
                    },
                    ..
                } if name == "None" && owner_path.iter().any(|part| part == "Option")
            )
        }),
        "lookup: niche `?` break arm must not construct Option::None"
    );
    let null_results: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments
                == &[
                    "core".to_string(),
                    "ptr".to_string(),
                    "null_mut".to_string(),
                ] =>
            {
                op.result.clone()
            }
            _ => None,
        })
        .collect();
    let narrowed_nulls: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| {
            let OpKind::Call { args, .. } = &op.kind else {
                return None;
            };
            if crate::model::cast_instance_root(&op.kind) == Some("PyObject")
                && args
                    .first()
                    .and_then(crate::model::LinkArg::as_variable)
                    .is_some_and(|v| null_results.contains(v))
            {
                op.result.clone()
            } else {
                None
            }
        })
        .collect();
    assert!(
        graph
            .blocks
            .iter()
            .any(|block| block.exits.iter().any(|link| {
                link.target == graph.returnblock
                    && link
                        .args
                        .iter()
                        .any(|arg| matches!(arg, LinkArg::Value(v) if narrowed_nulls.contains(v)))
            })),
        "lookup: niche `?` break arm must return a PyObject-narrowed null"
    );
}

/// `builtins::kwarg_get` opens with `let dict = kwargs?;` on its
/// `Option<PyObjectRef>` parameter, so the `Try::branch` sits in the
/// startblock after the parameters' `Input` ops and no block forwards the
/// Option into it.  `option_try` splits the startblock before the branch
/// and rewrites the `?` like any other site: no residual `branch` call.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn option_try_on_a_parameter_real_kwarg_get() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "kwarg_get").expect("lower kwarg_get");
    let branch_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, args, .. }
                    if name == "branch" && args.len() == 1
            )
        })
        .count();
    assert_eq!(
        branch_calls, 0,
        "kwarg_get: residual Try::branch Method-call on the parameter `?`"
    );
}

/// Real-LLBC anchor for the `Option::is_some_and` closure-select fold:
/// `is_mmap` is `type(obj).is_some_and(|tp| ptr::eq(tp.as_ptr(),
/// mmap_type()))`.  After the `option_closure_select` post-pass there must
/// be no residual `is_some_and` `Method` call, and block A must branch two
/// ways (the `Some`/`None` discriminant diamond).  `#[ignore]`d (loads the
/// ~440MB real LLBC); run with `cargo test -p majit-translate --lib
/// is_some_and_is_mmap_real -- --ignored`.
#[test]
#[ignore]
fn is_some_and_is_mmap_real() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "is_mmap").expect("lower is_mmap");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "is_some_and"
            )
        })
        .count();
    assert_eq!(
        residual, 0,
        "is_mmap: residual is_some_and call after the closure-select rewrite"
    );
    let branching = graph.blocks.iter().filter(|b| b.exits.len() == 2).count();
    assert!(
        branching >= 1,
        "is_mmap: expected the Some/None discriminant branch (a 2-exit block)"
    );
}

/// PyPy's frame stack clears store `None` into a `W_Root` list.  The Rust
/// sentinel `PY_NULL` must therefore retain the `PyObject` class boundary
/// while becoming nullable; a classdef-less null would widen the shared
/// `FixedObjectArray` item to the erased top reference.
#[test]
#[ignore]
fn popvalue_py_null_narrows_to_nullable_instance() {
    use crate::model::OpKind;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::pyframe::<Impl>::popvalue_maybe_none",
    )
    .expect("lower PyFrame::popvalue_maybe_none");
    let has_nullable_instance = graph.blocks.iter().any(|block| {
        block.operations.windows(2).any(|ops| {
            matches!(
                &ops[0].kind,
                OpKind::Call { target, .. } if target.to_string() == "core::ptr::null_mut"
            ) && crate::model::cast_instance_root(&ops[1].kind) == Some(super::W_ROOT_CLASS_ROOT)
        })
    });
    assert!(
        has_nullable_instance,
        "PY_NULL stack clear must lower as null_mut followed by a PyObject narrow"
    );
}

/// `call_method`'s `Result<Option<PyObjectRef>, PyError>::Ok.__pos_0`
/// is the niche pointer. The read must narrow to nullable `PyObject`
/// before it merges with the `PY_NULL` arm.
#[test]
#[ignore]
fn call_method_ok_payload_narrows_to_the_instance_class() {
    use crate::model::OpKind;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::baseobjspace::call_method")
        .expect("lower call_method");
    let narrowed = graph.blocks.iter().any(|block| {
        block.operations.windows(2).any(|ops| {
            let OpKind::FieldRead { field, .. } = &ops[0].kind else {
                return false;
            };
            field.name == "__pos_0"
                && field
                    .owner_root
                    .as_deref()
                    .is_some_and(|owner| owner.contains("Result") && owner.contains("PyObject"))
                && crate::model::cast_instance_root(&ops[1].kind) == Some(super::W_ROOT_CLASS_ROOT)
        })
    });
    assert!(
        narrowed,
        "call_method Ok payload read must be followed by a PyObject narrow"
    );
}

/// Re-type the small checked-in `option_source` graph with a chosen
/// payload and lower it. This preserves the exact Charon constructor CFG
/// used by the existing corpus harness while avoiding a large real-LLBC
/// load. The function has one `Some(value)` arm and one niladic `None`
/// arm, so the resulting graph directly exposes whether the classifier
/// chose aggregate construction or the nullable-pointer identities.
fn lower_option_source_with_payload(payload: serde_json::Value) -> FunctionGraph {
    lower_option_fn_retyped("option_source", payload, false, false, false)
}

fn lower_option_source_with_payload_ext(
    payload: serde_json::Value,
    inject_layoutless_nominal: bool,
) -> FunctionGraph {
    lower_option_fn_retyped(
        "option_source",
        payload,
        false,
        inject_layoutless_nominal,
        false,
    )
}

fn lower_option_source_with_aliased_payload(payload: serde_json::Value) -> FunctionGraph {
    lower_option_fn_retyped("option_source", payload, true, false, false)
}

fn lower_option_fn_retyped(
    fn_name: &str,
    mut payload: serde_json::Value,
    through_alias: bool,
    inject_layoutless_nominal: bool,
    inject_string: bool,
) -> FunctionGraph {
    fn replace_dedup(
        value: &mut serde_json::Value,
        payload_id: u64,
        output_id: u64,
        payload: &serde_json::Value,
        option_ty: &serde_json::Value,
    ) {
        if let Some(id) = value
            .as_object()
            .and_then(|obj| obj.get("Deduplicated"))
            .and_then(serde_json::Value::as_u64)
        {
            if id == payload_id {
                *value = payload.clone();
            } else if id == output_id {
                *value = option_ty.clone();
            }
            return;
        }
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    replace_dedup(value, payload_id, output_id, payload, option_ty);
                }
            }
            serde_json::Value::Object(values) => {
                for value in values.values_mut() {
                    replace_dedup(value, payload_id, output_id, payload, option_ty);
                }
            }
            _ => {}
        }
    }

    let path = crate::runtime_names::artifacts::CHARON_CORPUS_ULLBC;
    let bytes = std::fs::read(path).expect("read checked-in corpus LLBC");
    let mut file: serde_json::Value =
        serde_json::from_slice(&bytes).expect("parse checked-in corpus JSON");
    let translated = file
        .get_mut("translated")
        .and_then(serde_json::Value::as_object_mut)
        .expect("corpus translated object");

    if through_alias {
        let decls = translated
            .get_mut("type_decls")
            .and_then(serde_json::Value::as_array_mut)
            .expect("corpus type_decls");
        // Charon `type_by_id` indexes the table by position, so the
        // alias id is the slot this push occupies, not max(def_id)+1.
        let alias_id = decls.len() as u64;
        let template = decls
            .iter()
            .find(|decl| {
                decl.get("kind")
                    .and_then(|kind| kind.get("Alias"))
                    .is_some()
            })
            .cloned()
            .expect("corpus has a type alias to copy item_meta from");
        let mut alias_decl = template;
        alias_decl
            .as_object_mut()
            .expect("type decl object")
            .insert("def_id".to_string(), serde_json::json!(alias_id));
        alias_decl
            .as_object_mut()
            .expect("type decl object")
            .insert(
                "kind".to_string(),
                serde_json::json!({ "Alias": payload.clone() }),
            );
        if let Some(meta) = alias_decl
            .get_mut("item_meta")
            .and_then(|m| m.as_object_mut())
        {
            meta.insert(
                "name".to_string(),
                serde_json::json!([
                    {"Ident": ["charon_corpus", 0]},
                    {"Ident": ["ObjectPtr", 0]}
                ]),
            );
        }
        decls.push(alias_decl);
        payload = serde_json::json!({
            "Adt": {
                "id": alias_id,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        });
    }

    let option_def_id = translated
        .get("type_decls")
        .and_then(serde_json::Value::as_array)
        .and_then(|decls| {
            decls.iter().find_map(|decl| {
                let names = decl.get("item_meta")?.get("name")?.as_array()?;
                let path: Vec<&str> = names
                    .iter()
                    .filter_map(|seg| seg.get("Ident")?.get(0)?.as_str())
                    .collect();
                (path.as_slice() == ["core", "option", "Option"])
                    .then(|| decl.get("def_id")?.as_u64())?
            })
        })
        .expect("Option type declaration in corpus");
    if inject_layoutless_nominal {
        let decls = translated
            .get_mut("type_decls")
            .and_then(serde_json::Value::as_array_mut)
            .expect("corpus type_decls");
        // `type_by_id` indexes this array by `def_id`. A hole is the
        // only free index a copy can occupy.
        let hole = decls
            .iter()
            .position(serde_json::Value::is_null)
            .expect("corpus type_decls has a null hole");
        let mut copy = decls
            .iter()
            .find(|decl| decl.get("def_id").and_then(serde_json::Value::as_u64) == Some(4))
            .expect("sized nominal def-id 4")
            .clone();
        copy["def_id"] = serde_json::json!(hole);
        copy.as_object_mut()
            .expect("type decl object")
            .remove("layout");
        decls[hole] = copy;
        // The fixture payload names Adt 1000 as the stand-in. Point it
        // at the hole the copy actually occupies.
        fn retarget(value: &mut serde_json::Value, hole: u64) {
            if value.get("id").and_then(serde_json::Value::as_u64) == Some(1000) {
                value["id"] = serde_json::json!(hole);
            }
            match value {
                serde_json::Value::Array(items) => {
                    for item in items {
                        retarget(item, hole);
                    }
                }
                serde_json::Value::Object(map) => {
                    for item in map.values_mut() {
                        retarget(item, hole);
                    }
                }
                _ => {}
            }
        }
        retarget(&mut payload, hole as u64);
    }
    if inject_string {
        let decls = translated
            .get_mut("type_decls")
            .and_then(serde_json::Value::as_array_mut)
            .expect("corpus type_decls");
        let hole = decls
            .iter()
            .position(serde_json::Value::is_null)
            .expect("corpus type_decls has a null hole");
        // `SomeRawStruct` is a sized nominal struct; rename the copy to
        // `alloc::string::String` so the payload is a rust-heap header.
        let mut copy = decls
            .iter()
            .find(|decl| decl.get("def_id").and_then(serde_json::Value::as_u64) == Some(5))
            .expect("sized nominal def-id 5")
            .clone();
        copy["def_id"] = serde_json::json!(hole);
        if let Some(meta) = copy.get_mut("item_meta").and_then(|m| m.as_object_mut()) {
            meta.insert(
                "name".to_string(),
                serde_json::json!([
                    {"Ident": ["alloc", 0]},
                    {"Ident": ["string", 0]},
                    {"Ident": ["String", 0]}
                ]),
            );
        }
        decls[hole] = copy;
        payload = serde_json::json!({
            "Adt": {
                "id": hole,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        });
    }

    let fun = translated
        .get_mut("fun_decls")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|decls| {
            decls.iter_mut().find(|decl| {
                decl.get("item_meta")
                    .and_then(|meta| meta.get("name"))
                    .and_then(serde_json::Value::as_array)
                    .and_then(|names| names.last())
                    .and_then(|seg| seg.get("Ident"))
                    .and_then(|ident| ident.get(0))
                    .and_then(serde_json::Value::as_str)
                    == Some(fn_name)
            })
        })
        .unwrap_or_else(|| panic!("{fn_name} declaration in corpus"));
    let signature = fun
        .get("signature")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("{fn_name} signature"));
    let payload_id = signature
        .get("inputs")
        .and_then(serde_json::Value::as_array)
        .and_then(|inputs| inputs.get(1))
        .and_then(|ty| ty.get("Deduplicated"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| panic!("{fn_name} payload dedup id"));
    let output_id = signature
        .get("output")
        .and_then(|ty| ty.get("Deduplicated"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| panic!("{fn_name} output dedup id"));
    let option_ty = serde_json::json!({
        "Adt": {
            "id": option_def_id ,
            "generics": {
                "regions": [],
                "types": [payload.clone()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    replace_dedup(fun, payload_id, output_id, &payload, &option_ty);

    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    super::lower_function(&llbc, fn_name).unwrap_or_else(|e| panic!("lower retyped {fn_name}: {e}"))
}

fn niche_ctor_shape(graph: &FunctionGraph) -> (usize, usize) {
    let ops = graph.blocks.iter().flat_map(|block| &block.operations);
    // `rnone.py` `rtype_is_None`: a GC Ptr None is `null_mut` (ref
    // bank); `Option<fn>` is `null_fn` and a raw-address niche is
    // `null_raw` (int bank).
    let nulls = ops
        .clone()
        .filter(|op| {
            matches!(&op.kind, OpKind::Call { target, .. }
            if matches!(
                target.to_string().as_str(),
                "core::ptr::null_mut" | "core::ptr::null_fn" | "core::ptr::null_raw"
            ))
        })
        .count();
    let transparent_ctors = ops
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )
        })
        .count();
    (nulls, transparent_ctors)
}

/// `std::ptr::null_mut::<RawStruct>()` of a declared-Raw ADT is
/// `lltype.nullptr(T)` (`null_raw`) with T's class root, and must
/// not recast that null through `__cast_instance_intrinsic`.
#[test]
fn null_mut_of_raw_struct_is_null_raw_without_cast() {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt);
    let caller = interior_caller(
        "null_raw_ptr",
        0,
        vec![],
        ptr.clone(),
        vec![interior_local(0, None, &ptr)],
        vec![
            interior_bb(
                vec![],
                interior_call(1, vec![], interior_place(0, &ptr), 2, 1),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(vec![], interior_return()),
        ],
    );
    let null_mut = interior_opaque(1, &["std", "ptr", "null_mut"], vec![], ptr);
    let llbc = llbc_with_types(
        "fixture",
        vec![interior_struct(
            0,
            &["fixture", "RawStruct"],
            &[("word", interior_i64())],
            8,
            &[0],
            false,
        )],
        vec![caller, null_mut],
    );
    let graph = lower_interior(&llbc, "null_raw_ptr");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();
    let null_raw = ops
        .iter()
        .find(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. } if target.to_string() == "core::ptr::null_raw"
            )
        })
        .unwrap_or_else(|| panic!("expected core::ptr::null_raw; ops={ops:?}"));
    match &null_raw.kind {
        OpKind::Call {
            args, result_ty, ..
        } => {
            assert_eq!(result_ty, &crate::model::ValueType::Int);
            match args.as_slice() {
                [LinkArg::Const(c)] => {
                    assert_eq!(
                        c.value.as_pystr(),
                        Some("RawStruct"),
                        "null_raw must carry T's class-root spelling"
                    );
                }
                other => panic!("expected one class-root spelling, got {other:?}"),
            }
        }
        other => panic!("expected null_raw call, got {other:?}"),
    }
    let null_raw_var = null_raw
        .result
        .as_ref()
        .expect("null_raw produces a result");
    assert!(
        ops.iter()
            .all(|op| crate::model::cast_instance_of(&op.kind, null_raw_var).is_none()),
        "null_raw must not be followed by __cast_instance_intrinsic; ops={ops:?}"
    );
    assert!(
        ops.iter().all(|op| {
            !matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.as_slice()
                    == ["core", "ptr", "null_mut"]
                    || segments.as_slice() == ["std", "ptr", "null_mut"]
                    || segments.as_slice() == ["core", "ptr", "null"]
                    || segments.as_slice() == ["std", "ptr", "null"]
            )
        }),
        "raw-pointee null_mut must not remain as a GC null; ops={ops:?}"
    );
}

#[test]
fn niche_option_raw_nominal_ptr_without_layout_is_null_test() {
    use crate::model::OpKind;
    // Cross-crate pointees often have a type decl and no `layout`
    // entry. `Option<*mut That>` is still the nullable pointer:
    // `None` is null and the discriminant is a null test, not a
    // `__discriminant` load of the pointee header.
    let payload = serde_json::json!({
        "RawPtr": [
            {
                "Adt": {
                    "id": 1000 ,
                    "generics": {
                        "regions": [], "types": [],
                        "const_generics": [], "trait_refs": []
                    }
                }
            },
            "Mut"
        ]
    });
    let graph = lower_option_source_with_payload_ext(payload, true);
    let (null_muts, transparent_ctors) = niche_ctor_shape(&graph);
    assert_eq!(null_muts, 1, "None must lower to one null pointer");
    assert_eq!(transparent_ctors, 0, "Some must be the pointer identity");
    let discriminant_reads = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
    assert_eq!(
        discriminant_reads, 0,
        "a layout-less nominal raw pointer must not read __discriminant"
    );
}

#[test]
fn niche_option_type_alias_of_raw_nominal_ptr_none_is_null() {
    use crate::model::OpKind;
    // `type ObjectPtr = *mut HostRegistry` is the Charon spelling of
    // `type PyObjectRef = *mut PyObject`: the Option payload names the
    // alias Adt, not the `RawPtr`.  Discriminant / Some / None must
    // still see one nullable pointer word.
    let payload = serde_json::json!({
        "RawPtr": [
            {
                "Adt": {
                    "id": 4,
                    "generics": {
                        "regions": [], "types": [],
                        "const_generics": [], "trait_refs": []
                    }
                }
            },
            "Mut"
        ]
    });
    let graph = lower_option_source_with_aliased_payload(payload);
    let (null_muts, transparent_ctors) = niche_ctor_shape(&graph);
    assert_eq!(
        null_muts, 1,
        "None of an alias-to-raw-nominal-ptr Option must lower to one null pointer"
    );
    assert_eq!(
        transparent_ctors, 0,
        "Some of an alias-to-raw-nominal-ptr Option must be the payload identity"
    );
    let discriminant_reads = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant"
            )
        })
        .count();
    assert_eq!(
        discriminant_reads, 0,
        "a pointer-niche Option must not read an aggregate __discriminant"
    );
}

#[test]
fn niche_option_nominal_pointers_none_is_null_some_is_identity() {
    let corpus = Llbc::load(crate::runtime_names::artifacts::CHARON_CORPUS_ULLBC)
        .expect("load checked-in corpus LLBC");
    let box_def_id = corpus
        .iter_type_decls()
        .find(|td| td.item_meta.name_path() == "alloc::boxed::Box")
        .map(|td| td.def_id)
        .expect("Box type declaration in corpus");
    let raw_nominal = serde_json::json!({
        "RawPtr": [
            {
                "Adt": {
                    "id": 4 ,
                    "generics": {
                        "regions": [], "types": [],
                        "const_generics": [], "trait_refs": []
                    }
                }
            },
            "Mut"
        ]
    });
    let boxed = serde_json::json!({
        "Adt": {
            "id": box_def_id ,
            "generics": {
                "regions": [],
                "types": [{
                    "Adt": {
                        "id": 4 ,
                        "generics": {
                            "regions": [], "types": [],
                            "const_generics": [], "trait_refs": []
                        }
                    }
                }],
                "const_generics": [], "trait_refs": []
            }
        }
    });
    let cases = [
        (
            "raw_nominal_ptr",
            raw_nominal,
            "None must lower to one null pointer",
            "Some(raw nominal pointer) must be the payload identity, with no Option aggregate",
        ),
        (
            "boxed_nominal",
            boxed,
            "None must lower to one null owning pointer",
            "Some(Box<nominal>) must be the payload identity, with no Option aggregate",
        ),
    ];
    for (name, payload, none_msg, some_msg) in cases {
        let graph = lower_option_source_with_payload(payload);
        let (null_muts, transparent_ctors) = niche_ctor_shape(&graph);
        assert_eq!(null_muts, 1, "case {name}: {none_msg}");
        assert_eq!(transparent_ctors, 0, "case {name}: {some_msg}");
    }
}

/// `Rvalue::Discriminant` of `Option<&[u8]>` (and Charon's
/// `Option<[u8]>` spelling) is the nullable-window null test:
/// `is_(data, None)` then `eq(_, false)`. The data word is CCHARP;
/// `UInt(0)` is not a pointer null.
#[test]
fn option_u8_window_discriminant_is_pointer_null_test() {
    use crate::model::OpKind;

    let disc_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "Isize"}}});
    let payloads = [
        ("Option<&[u8]>", window_shared_slice()),
        ("Option<[u8]>", window_slice()),
    ];
    for (name, payload) in payloads {
        let option_ty = option_adt_node(0, payload);
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": window_meta(&["fixture", "window_disc"], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [option_ty.clone()],
                "output": disc_ty.clone()
            },
            "body": {"Unstructured": {
                "span": window_span(),
                "locals": {"arg_count": 1, "locals": [
                    window_local(0, &disc_ty),
                    window_local(1, &option_ty)
                ]},
                "body": [{
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &disc_ty),
                            {"Discriminant": window_place(
                                serde_json::json!({"Local": 1}),
                                &option_ty
                            )}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                }]
            }}
        });
        let llbc = window_llbc(vec![option_enum(0)], vec![caller]);
        let graph = super::lower_function(&llbc, "window_disc")
            .unwrap_or_else(|err| panic!("{name}: lower Discriminant: {err}"));
        let ops: Vec<&crate::model::SpaceOperation> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        let const_uint_zeros = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::ConstUInt(0)))
            .count();
        assert_eq!(
            const_uint_zeros,
            0,
            "{name}: window Discriminant must not emit ConstUInt(0); graph={}",
            graph.dump()
        );
        let ne_ops = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne"))
            .count();
        assert_eq!(
            ne_ops,
            0,
            "{name}: window Discriminant must not emit BinOp ne; graph={}",
            graph.dump()
        );
        let none_ids: std::collections::HashSet<u64> = ops
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::ConstNone => op.result.as_ref().map(|v| v.id()),
                _ => None,
            })
            .collect();
        assert!(
            !none_ids.is_empty(),
            "{name}: window Discriminant must emit ConstNone; graph={}",
            graph.dump()
        );
        let is_none_ids: std::collections::HashSet<u64> = ops
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::BinOp { op: name, rhs, .. }
                    if name == "is_" && none_ids.contains(&rhs.id()) =>
                {
                    op.result.as_ref().map(|v| v.id())
                }
                _ => None,
            })
            .collect();
        assert!(
            !is_none_ids.is_empty(),
            "{name}: window Discriminant must emit is_(data, None); graph={}",
            graph.dump()
        );
        let false_ids: std::collections::HashSet<u64> = ops
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::ConstBool(false) => op.result.as_ref().map(|v| v.id()),
                _ => None,
            })
            .collect();
        let some_eq = ops.iter().any(|op| match &op.kind {
            OpKind::BinOp {
                op: name, lhs, rhs, ..
            } => name == "eq" && is_none_ids.contains(&lhs.id()) && false_ids.contains(&rhs.id()),
            _ => false,
        });
        assert!(
            some_eq,
            "{name}: window Discriminant must emit eq(is_none, false); graph={}",
            graph.dump()
        );
        let disc_reads = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. } if field.name == "__discriminant"
                )
            })
            .count();
        assert_eq!(
            disc_reads,
            0,
            "{name}: window Discriminant must not read __discriminant; graph={}",
            graph.dump()
        );
    }
}

/// `match opt { Some(w) => w, None => default }` over `Option<&[u8]>`
/// (and Charon's `Option<[u8]>` spelling): the `Some` payload is the
/// Option's window, not a tagged `Option<[u8]>::Some` instance.
#[test]
fn option_u8_window_some_payload_is_the_window() {
    use crate::model::{CallTarget, ExitCase, LinkArg, OpKind};

    let disc_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "Isize"}}});
    let payloads = [
        ("Option<&[u8]>", window_shared_slice()),
        ("Option<[u8]>", window_slice()),
    ];
    for (name, payload) in payloads {
        let option_ty = option_adt_node(0, payload.clone());
        // `unwrap_or` returns `&[u8]`: a window dest. Charon's
        // `Option<[u8]>` spelling still projects `Some.0` as `[u8]`.
        let dest_ty = window_shared_slice();
        let opt_place = window_place(serde_json::json!({"Local": 1}), &option_ty);
        let default_place = window_place(serde_json::json!({"Local": 2}), &dest_ty);
        let ret_place = window_place(serde_json::json!({"Local": 0}), &dest_ty);
        let disc_place = window_place(serde_json::json!({"Local": 3}), &disc_ty);
        let some_payload = window_place(
            serde_json::json!({"Projection": [
                window_place(
                    serde_json::json!({"Projection": [opt_place.clone(), {"Downcast": 1}]}),
                    &option_ty
                ),
                {"Field": [1, 0]}
            ]}),
            &payload,
        );
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": window_meta(&["fixture", "window_some_payload"], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [option_ty.clone(), dest_ty.clone()],
                "output": dest_ty.clone()
            },
            "body": {"Unstructured": {
                "span": window_span(),
                "locals": {"arg_count": 2, "locals": [
                    window_local(0, &dest_ty),
                    window_local(1, &option_ty),
                    window_local(2, &dest_ty),
                    window_local(3, &disc_ty)
                ]},
                "body": [
                    {
                        "statements": [{
                            "span": window_span(),
                            "kind": {"Assign": [
                                disc_place.clone(),
                                {"Discriminant": opt_place.clone()}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {"span": window_span(), "kind": {"Switch": {
                            "discr": {"Copy": disc_place.clone()},
                            "targets": {"SwitchInt": [
                                null,
                                [
                                    [{"Scalar": {"Signed": ["Isize", "1"]}}, 1],
                                    [{"Scalar": {"Signed": ["Isize", "0"]}}, 2]
                                ],
                                2
                            ]}
                        }}}
                    },
                    {
                        "statements": [{
                            "span": window_span(),
                            "kind": {"Assign": [
                                ret_place.clone(),
                                {"Use": [{"Copy": some_payload}, "No"]}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {"span": window_span(), "kind": "Return"}
                    },
                    {
                        "statements": [{
                            "span": window_span(),
                            "kind": {"Assign": [
                                ret_place.clone(),
                                {"Use": [{"Copy": default_place}, "No"]}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {"span": window_span(), "kind": "Return"}
                    }
                ]
            }}
        });
        let llbc = window_llbc(vec![option_enum(0)], vec![caller]);
        let graph = super::lower_function(&llbc, "window_some_payload")
            .unwrap_or_else(|err| panic!("{name}: lower Some payload: {err}"));
        let ops: Vec<&crate::model::SpaceOperation> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        let cast_instance = ops
            .iter()
            .any(|op| crate::model::cast_instance_root(&op.kind).is_some());
        assert!(
            !cast_instance,
            "{name}: Some arm must not emit __cast_instance_intrinsic; graph={}",
            graph.dump()
        );
        let pos0 = ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "__pos_0"
            )
        });
        assert!(
            !pos0,
            "{name}: Some arm must not getattr \"__pos_0\"; graph={}",
            graph.dump()
        );
        let cast_int_to_ptr = ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("cast_int_to_ptr")
            )
        });
        assert!(
            !cast_int_to_ptr,
            "{name}: Some arm must not emit cast_int_to_ptr; graph={}",
            graph.dump()
        );
        // The niche `is_some` switch's true arm is `Some`. After the
        // branch the data word is the successor's first inputarg
        // (SSA-renamed `arg1`), not the startblock's `arg1_0`.
        let start = graph.block(graph.startblock);
        let some_link = start
            .exits
            .iter()
            .find(|link| link.exitcase == Some(ExitCase::Bool(true)))
            .unwrap_or_else(|| {
                panic!(
                    "{name}: Some is the true arm of the niche is_some switch; graph={}",
                    graph.dump()
                )
            });
        let some_block = graph.block(some_link.target);
        let some_data = some_block
            .inputargs
            .first()
            .unwrap_or_else(|| panic!("{name}: Some arm carries opt data; graph={}", graph.dump()));
        let some_returns_opt = if some_link.target == graph.returnblock {
            matches!(some_link.args.first(), Some(LinkArg::Value(v)) if v == some_data)
        } else {
            some_block.exits.iter().any(|link| {
                link.target == graph.returnblock
                    && matches!(link.args.first(), Some(LinkArg::Value(v)) if v == some_data)
            })
        };
        assert!(
            some_returns_opt,
            "{name}: Some arm must return opt's data word; graph={}",
            graph.dump()
        );
    }
}

/// Real-LLBC anchor for the `FixedObjectArray` Index intercept.  The
/// intercept returns before `lower_call`'s ordinary result-narrowing tail,
/// so an object element must carry its own `PyObject` narrow immediately
/// after the lifted ArrayRead.  Ignored by default because it loads the
/// large interpreter snapshot.
#[test]
#[ignore]
fn load_local_value_arrayread_narrows_instance_item() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::eval::<Impl>::load_local_value")
        .expect("lower eval::<Impl>::load_local_value");
    let has_instance_narrow = graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if crate::model::cast_instance_root(&op.kind)
                .is_some_and(|root| root.ends_with("PyObject"))
        )
    });
    assert!(
        has_instance_narrow,
        "FixedObjectArray getitem must recast its GCREF item to PyObject"
    );
}

/// `range_list_length` stores the PyPy strategy length in an RPython
/// Signed cell and converts it to Rust's indexing carrier.  The opaque
/// core `TryFrom` shell must be decomposed by the MIR front rather than
/// surviving as a residual foreign call that blocks the whole list
/// strategy closure.
#[test]
#[ignore]
fn range_list_length_lowers_signed_to_usize_try_from() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real object LLBC");
    let source = super::lower_function(&llbc, "pyre_object::listobject::range_state_value")
        .expect("lower listobject::range_state_value");
    assert!(
        source
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::ArrayRead {
                    item_ty: ValueType::Int,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == "[i64]"
            )),
        "TypedItemsBlock i64 dereference must be a GcArray(Signed) read"
    );
    let graph = super::lower_function(&llbc, "pyre_object::listobject::range_list_length")
        .expect("lower listobject::range_list_length");
    let residuals: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments
                .iter()
                .any(|segment| segment == "ptr_try_from_impls") =>
            {
                Some(segments.join("::"))
            }
            _ => None,
        })
        .collect();
    assert!(
        residuals.is_empty(),
        "signed-to-usize conversion must not leave an opaque core call: {residuals:?}"
    );
    let unwrap_residuals = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "unwrap"
            )
        })
        .count();
    assert_eq!(
        unwrap_residuals, 0,
        "checked conversion Result::unwrap must become a discriminant guard"
    );
    for name in [
        "pyre_object::listobject::range_state_value",
        "pyre_object::listobject::range_list_length",
        "pyre_object::listobject::range_list_start_step",
        "pyre_object::listobject::sizehint_state_value",
    ] {
        let lowered = super::lower_function(&llbc, name).unwrap_or_else(|err| {
            panic!("lower {name}: {err:?}");
        });
        let raw_loads: Vec<_> = lowered
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| matches!(op.kind, OpKind::RawLoad { .. }))
            .collect();
        assert!(
            raw_loads.is_empty(),
            "{name}: a TypedItemsBlock word is getarrayitem, not raw_load: {raw_loads:?}"
        );
        if name.ends_with("sizehint_state_value") {
            let ret = lowered
                .blocks
                .iter()
                .flat_map(|block| &block.exits)
                .find(|link| link.target == lowered.returnblock)
                .and_then(|link| link.args.first())
                .and_then(crate::model::LinkArg::as_variable)
                .cloned()
                .expect("sizehint_state_value returns the loaded word");
            let mut current = ret;
            let mut producer = None;
            for _ in 0..32 {
                if let Some(op) = lowered.blocks.iter().find_map(|block| {
                    block
                        .operations
                        .iter()
                        .find_map(|op| (op.result.as_ref() == Some(&current)).then_some(&op.kind))
                }) {
                    producer = Some(op);
                    break;
                }
                let mut next = None;
                for block in &lowered.blocks {
                    let Some(slot) = block.inputargs.iter().position(|arg| arg == &current) else {
                        continue;
                    };
                    for pred in &lowered.blocks {
                        for exit in &pred.exits {
                            if exit.target == block.id
                                && let Some(src) =
                                    exit.args.get(slot).and_then(|arg| arg.as_variable())
                                && src != &current
                            {
                                next = Some(src.clone());
                            }
                        }
                    }
                }
                match next {
                    Some(src) => current = src,
                    None => break,
                }
            }
            assert!(
                matches!(
                    producer,
                    Some(OpKind::ArrayRead {
                        item_ty: ValueType::Int,
                        array_type_id: Some(array_type_id),
                        ..
                    }) if array_type_id == "[i64]"
                ),
                "{name}: returned word must be the index-0 Signed read, got {producer:?}"
            );
        }
    }
    let store = super::lower_function(&llbc, "pyre_object::listobject::set_sizehint_state_value")
        .expect("lower set_sizehint_state_value");
    let raw_stores: Vec<_> = store
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(op.kind, OpKind::RawStore { .. }))
        .collect();
    assert!(
        raw_stores.is_empty(),
        "set_sizehint_state_value must not raw_store a GcArray cell: {raw_stores:?}"
    );
    assert!(
        store
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::ArrayWrite {
                    item_ty: ValueType::Int,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == "[i64]"
            )),
        "set_sizehint_state_value must setarrayitem the Signed cell"
    );
}

/// `complex_val` returns the RPython spelling of an optional pair.  The
/// arithmetic helpers call `unwrap` only after their numeric dispatch has
/// established the value is present, so the opaque Rust combinator must be
/// translated to the ordinary payload guard instead of becoming a method
/// lookup on an Option shell.
#[test]
#[ignore]
fn complex_arithmetic_unwraps_lower_to_payload_guards() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    for name in [
        "pyre_interpreter::objspace::descroperation::complex_add",
        "pyre_interpreter::objspace::descroperation::complex_sub",
        "pyre_interpreter::objspace::descroperation::complex_mul",
        "pyre_interpreter::objspace::descroperation::complex_truediv",
        "pyre_interpreter::objspace::descroperation::complex_neg",
    ] {
        let graph = super::lower_function(&llbc, name)
            .unwrap_or_else(|err| panic!("lower {name}: {err:?}"));
        let residuals: Vec<_> = graph
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(block, bb)| {
                bb.operations.iter().filter_map(move |op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        ..
                    } if name == "unwrap" => Some((block, op.clone(), bb.exits.clone())),
                    _ => None,
                })
            })
            .collect();
        assert!(
            residuals.is_empty(),
            "{name}: Option::unwrap must become a payload guard: {residuals:#?}"
        );
    }
}

/// The unwrapped-list storage adapters are Rust spellings of
/// RPython's `l.items` GcArray.  A live `push` must therefore end in the
/// same `setarrayitem` shape as `rlist.ll_append_noresize`, never in the
/// synthetic raw-pointer write that blocks annotation.  The adapter body
/// itself must also be free of the opaque Rust `offset_of!` getter; its
/// target-layout constant is part of the translated GcArray shape.
#[test]
#[ignore]
fn unwrapped_array_pushes_lower_to_rpython_gcarray_writes() {
    use crate::model::{CallTarget, OpKind};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real object LLBC");
    // `offset_leaf` is `None` for the two string rows on purpose: their
    // `base` is `items_block_items_base(self.block)`, a call with no
    // offset constant in the body, so a "no surviving offset read"
    // assertion over them would hold whatever the fold did.  Their real
    // content is the `push` half below.
    for (base, function, expected_array, expected_item, offset_leaf) in [
        (
            "pyre_object::int_array::<Impl>::base",
            "pyre_object::int_array::<Impl>::push",
            "[i64]",
            ValueType::Int,
            Some("TYPED_ITEMS_BLOCK_ITEMS_OFFSET"),
        ),
        (
            "pyre_object::float_array::<Impl>::base",
            "pyre_object::float_array::<Impl>::push",
            "[f64]",
            ValueType::Float,
            Some("TYPED_ITEMS_BLOCK_ITEMS_OFFSET"),
        ),
        (
            "pyre_object::bytes_array::<Impl>::base",
            "pyre_object::bytes_array::<Impl>::push",
            super::STRING_GCREF_GCARRAY_TYPE_ID,
            ValueType::Str,
            None,
        ),
        (
            "pyre_object::unicode_array::<Impl>::base",
            "pyre_object::unicode_array::<Impl>::push",
            super::STRING_GCREF_GCARRAY_TYPE_ID,
            ValueType::Str,
            None,
        ),
    ] {
        if let Some(offset_leaf) = offset_leaf {
            let base_graph = super::lower_function(&llbc, base)
                .unwrap_or_else(|err| panic!("lower {base}: {err}"));
            assert!(
                !base_graph
                    .blocks
                    .iter()
                    .flat_map(|block| &block.operations)
                    .any(|op| matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.last().is_some_and(|leaf| leaf == offset_leaf)
                    )),
                "{base} must fold the target-layout items offset"
            );
        }
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|err| panic!("lower {function}: {err}"));
        let operations: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::ArrayWrite {
                    item_ty,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == expected_array && item_ty == &expected_item
            )),
            "{function} must write {expected_array}/{expected_item:?} through setarrayitem"
        );
        assert!(
            !operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["__deref_write".to_string()]
            )),
            "{function} must not retain a raw-pointer write"
        );
    }

    for function in [
        "pyre_object::bytes_array::<Impl>::as_slice",
        "pyre_object::unicode_array::<Impl>::as_slice",
    ] {
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|err| panic!("lower {function}: {err}"));
        assert!(
            !graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().is_some_and(|leaf| leaf == "from_raw_parts")
                )),
            "{function} must be the direct RPython list-items view"
        );
    }

    for function in [
        "pyre_object::bytes_array::<Impl>::pop",
        "pyre_object::unicode_array::<Impl>::pop",
        "pyre_object::bytes_array::<Impl>::remove",
        "pyre_object::unicode_array::<Impl>::remove",
    ] {
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|err| panic!("lower {function}: {err}"));
        let operations: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            operations.iter().any(|operation| matches!(
                &operation.kind,
                OpKind::ArrayRead {
                    item_ty: ValueType::Str,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == super::STRING_GCREF_GCARRAY_TYPE_ID
            )),
            "{function} must read its external string through GcArray(GCREF)"
        );
        assert!(
            operations.iter().any(|operation| matches!(
                &operation.kind,
                OpKind::ArrayWrite {
                    item_ty: ValueType::Str,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == super::STRING_GCREF_GCARRAY_TYPE_ID
            )),
            "{function} must clear the released slot through the same GcArray(GCREF)"
        );
        assert!(
            !operations.iter().any(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|leaf| {
                    leaf == "from_raw_parts" || leaf == "__deref_write"
                })
            )),
            "{function} must not retain a Rust raw-slice/read-write boundary"
        );
        if function.ends_with("::remove") {
            assert!(
                operations.iter().any(|operation| matches!(
                    &operation.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &[crate::runtime_names::shims::LL_ARRAYMOVE.to_string()]
                )),
                "{function} must restore ptr::copy to RPython rgc.ll_arraymove"
            );
            assert!(
                !operations.iter().any(|operation| matches!(
                    &operation.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.as_slice() == ["core", "ptr", "copy"]
                        || segments.as_slice() == ["core", "ptr", "mut_ptr", "<Impl>", "add"]
                )),
                "{function} must not retain Rust pointer-copy arithmetic"
            );
        }
    }
}

/// `opcode_for_iter` matches `Option<H::Value>`. The sole impl binds
/// `Value = *mut PyObject`, a null niche. The generic body spells that
/// payload as a trait projection; the discriminant must be a null test.
#[test]
#[ignore]
fn opcode_for_iter_option_assoc_payload_is_null_test() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::pyopcode::opcode_for_iter")
        .expect("lower opcode_for_iter");
    let null_muts = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(&op.kind, OpKind::Call { target, .. }
                    if target.to_string() == "core::ptr::null_mut")
        })
        .count();
    assert!(
        null_muts >= 1,
        "opcode_for_iter Option match must null-test the pointer (null_muts={null_muts})"
    );
}

#[test]
fn niche_option_non_niche_payloads_remain_aggregate() {
    let scalar_ptr = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]
    });
    let tuple = serde_json::json!({
        "Adt": {
            "id": 0, "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": [
                    {"Scalar": {"Integer": {"Signed": "I64"}}},
                    { "Scalar": "Bool" }
                ],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cases = [
        (
            "raw_scalar_ptr",
            scalar_ptr,
            "Option<*mut u8> must not collapse None to the payload null word",
            "Option<*mut u8> must retain the tagged Some and None aggregates",
        ),
        (
            "tuple",
            tuple,
            "Option<(i64, bool)> must not collapse None to a payload null word",
            "Option<(i64, bool)> must retain the tagged Some and None aggregates",
        ),
    ];
    for (name, payload, none_msg, agg_msg) in cases {
        let graph = lower_option_source_with_payload(payload);
        let (null_muts, transparent_ctors) = niche_ctor_shape(&graph);
        assert_eq!(null_muts, 0, "case {name}: {none_msg}");
        assert!(transparent_ctors >= 2, "case {name}: {agg_msg}");
    }
}

/// `Option<String>` match is RPython's nullable `SomeString`:
/// `s is None` (`pairtype(SomeObject, SomeObject).is_`, rtyped by
/// `rtype_is_` / `ptr_iszero`), and the `Some` payload is the string
/// itself. The corpus `option_question_mark` `?` is a `Try::branch`
/// match; injecting `alloc::string::String` as the Option payload
/// pins that the front does not emit `getattr('__discriminant')`.
#[test]
fn niche_option_string_match_is_none_identity() {
    use crate::model::OpKind;
    let graph = lower_option_fn_retyped(
        "option_question_mark",
        serde_json::Value::Null,
        false,
        false,
        true,
    );
    let (null_muts, transparent_ctors) = niche_ctor_shape(&graph);
    assert!(
        null_muts >= 1,
        "Option<String> None must lower to a null string pointer, got {null_muts}"
    );
    assert_eq!(
        transparent_ctors, 0,
        "Option<String> Some must be the string identity, with no Option aggregate"
    );
    let discriminant_reads = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant")
            })
            .count();
    assert_eq!(
        discriminant_reads, 0,
        "Option<String> match must not read an aggregate __discriminant"
    );
    let some_payload_reads = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref().is_some_and(|owner| {
                            owner.contains("Option") && owner.ends_with("::Some")
                        })
            )
        })
        .count();
    assert_eq!(
        some_payload_reads, 0,
        "Option<String> Some payload must be the string itself"
    );
    let has_is = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_"));
    assert!(
        has_is,
        "Option<String> match must contain an identity null-test"
    );
}

/// Niche `Option<*mut T>` `eq` call against a named trait impl.
/// Charon spells the callee as `core::option::<Impl>::eq` with
/// `{"Impl": {"Trait": id}}` and `first_is_self`.
fn niche_option_ptr_eq_llbc(trait_name: &[&str]) -> Llbc {
    let span = serde_json::json!({"data": {"file_id": 0,
            "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let meta = |name: Vec<serde_json::Value>| {
        serde_json::json!({
            "name": name,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let payload_ty = serde_json::json!({
        "Adt": {"id": 1, "generics": generics}
    });
    let ptr_ty = serde_json::json!({"RawPtr": [payload_ty, "Mut"]});
    let option_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [ptr_ty.clone()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let option_ref = serde_json::json!({"Ref": ["Erased", option_ty.clone(), "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let impl_seg = serde_json::json!({"Impl": {"Trait": 0}});
    let option_value = serde_json::json!({"Value": [0, option_ty.clone()]});
    let trait_idents: Vec<serde_json::Value> = trait_name.iter().copied().map(ident).collect();
    let place = |id, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("core"), ident("option"), ident("Option")]),
                    "kind": {"Enum": [
                        {
                            "name": "None",
                            "fields": [],
                            "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}
                        },
                        {
                            "name": "Some",
                            "fields": [{
                                "name": null,
                                "is_positional": true,
                                "ty": ptr_ty.clone(),
                                "attr_info": null
                            }],
                            "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}
                        }
                    ]}
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![
                        ident("fixture"),
                        ident("pyobject"),
                        ident("PyObject"),
                    ]),
                    "kind": {"Struct": []}
                }
            ],
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("fixture"), ident("cmp_option_ptr_eq")]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [option_ty.clone(), ptr_ty.clone()],
                        "output": bool_ty.clone()
                    },
                    "body": {"Unstructured": {"span": span,
                        "locals": {"arg_count": 2, "locals": [
                            {"index": 0, "name": null, "span": span, "ty": bool_ty.clone()},
                            {"index": 1, "name": "opt", "span": span, "ty": option_ty.clone()},
                            {"index": 2, "name": "p", "span": span, "ty": ptr_ty.clone()},
                            {"index": 3, "name": "some", "span": span, "ty": option_ty.clone()},
                            {"index": 4, "name": null, "span": span, "ty": option_ref.clone()},
                            {"index": 5, "name": null, "span": span, "ty": option_ref.clone()},
                            {"index": 6, "name": null, "span": span, "ty": bool_ty.clone()}
                        ]},
                        "body": [
                            {"statements": [
                                {
                                    "span": span,
                                    "kind": {"Assign": [
                                        place(3, &option_ty),
                                        {"Aggregate": [
                                            {"Adt": [0, 1, null]},
                                            [{"Copy": place(2, &ptr_ty)}]
                                        ]}
                                    ]}
                                },
                                {
                                    "span": span,
                                    "kind": {"Assign": [
                                        place(4, &option_ref),
                                        {"Ref": {
                                            "place": place(1, &option_ty),
                                            "kind": "Shared",
                                            "ptr_metadata": null
                                        }}
                                    ]}
                                },
                                {
                                    "span": span,
                                    "kind": {"Assign": [
                                        place(5, &option_ref),
                                        {"Ref": {
                                            "place": place(3, &option_ty),
                                            "kind": "Shared",
                                            "ptr_metadata": null
                                        }}
                                    ]}
                                }
                            ], "terminator": {"span": span, "kind": {"Call": {
                                "call": {
                                    "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                    "args": [
                                        {"Copy": place(4, &option_ref)},
                                        {"Copy": place(5, &option_ref)}
                                    ],
                                    "dest": place(6, &bool_ty)
                                },
                                "target": 1, "on_unwind": 2
                            }}}},
                            {"statements": [{
                                "span": span,
                                "kind": {"Assign": [
                                    place(0, &bool_ty),
                                    {"Use": [{"Copy": place(6, &bool_ty)}, "Yes"]}
                                ]}
                            }], "terminator": {"span": span, "kind": "Return"}},
                            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                        ]
                    }}
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![
                        ident("core"), ident("option"), impl_seg, ident("eq")
                    ]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [option_ref.clone(), option_ref.clone()],
                        "output": bool_ty
                    },
                    "body": "Opaque"
                }
            ],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(trait_idents),
                "methods": [{"skip_binder": {"name": "eq"}}]
            }],
            "trait_impls": [{
                "impl_trait": {
                    "id": 0,
                    "generics": {"types": [option_value.clone(), option_value]}
                },
                "methods": []
            }]
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("option-ptr-eq fixture parses")
}

/// `opt == Some(p)` on niche `Option<*mut T>`. Charon spells the
/// derived `PartialEq` as `core::option::<Impl>::eq` with
/// `first_is_self`, so `impl_method_owner` would emit
/// `CallTarget::Method { name: "eq" }`. The niche fold must win
/// and emit one address `BinOp("eq")` (`rclass.py` `rtype_eq` →
/// `ptr_eq`; `rptr.py` `PtrRepr.rtype_eq`).
#[test]
fn niche_option_ptr_eq_some_lowers_to_binop() {
    use crate::model::{CallTarget, OpKind};

    let llbc = niche_option_ptr_eq_llbc(&["core", "cmp", "PartialEq"]);
    let graph = super::lower_function(&llbc, "cmp_option_ptr_eq").expect("lower cmp_option_ptr_eq");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "eq"
            ))
            .count(),
        0,
        "no residual `eq` Method call survives the niche fold"
    );
    assert_eq!(
        ops()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq"))
            .count(),
        1,
        "the comparison lowers to one address `BinOp(eq)`"
    );
}

/// An extension trait's `eq` on niche `Option` keeps the call
/// (`CallControl.graphs_from` / `FunctionReprBase.call`). Only
/// `core::cmp::PartialEq` lowers to `BinOp`.
#[test]
fn niche_option_ext_eq_stays_a_call() {
    use crate::model::{CallTarget, OpKind};

    let llbc = niche_option_ptr_eq_llbc(&["fixture", "Ext"]);
    let graph = super::lower_function(&llbc, "cmp_option_ptr_eq").expect("lower cmp_option_ptr_eq");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq"))
            .count(),
        0,
        "an Ext::eq on Option is not the PartialEq BinOp fold"
    );
    assert!(
        ops().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "eq",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("eq"),
            _ => false,
        }),
        "Ext::eq on Option stays a call"
    );
}

/// `gettypeobject` (`gettypefor(..).map_or(PY_NULL, |p| p.as_ptr())`)
/// must fold its `gettypefor` match through the niche
/// `Option<NonNull<PyObject>>` path now that `gettypefor` returns a
/// one-word niche Option: the map_or rewrite uses a `ptr_ne` null-test
/// (`base != null_mut()`), NOT an aggregate `__discriminant` FieldRead,
/// and the null is a `null_mut()` call (repr-adaptive) rather than a
/// fixed-GCREF `ConstRefNull`.  Guards against a regression where the
/// niche detector stops recognising the `NonNull` payload (which would
/// re-introduce the classdef-less `__discriminant` base) or where the
/// null reverts to `ConstRefNull` (which cannot rtype against the
/// pointer's `InstanceRepr`).  Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn niche_option_gettypeobject_folds_to_ptr_null_test() {
    use crate::model::OpKind;

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "gettypeobject").expect("lower gettypeobject");

    // The niche null is a `null_mut()` call, never a `ConstRefNull`.
    let const_ref_nulls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(&op.kind, OpKind::ConstRefNull))
        .count();
    assert_eq!(
        const_ref_nulls, 0,
        "gettypeobject: niche null must be null_mut(), not a fixed-GCREF ConstRefNull"
    );
    let null_muts = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(&op.kind, OpKind::Call { target, .. }
                    if target.to_string() == "core::ptr::null_mut")
        })
        .count();
    assert!(
        null_muts >= 1,
        "gettypeobject: expected a null_mut() feeding the niche ptr null-test"
    );
    // The map_or niche discriminant contains a `ne` pointer null-test, and no
    // `__discriminant` FieldRead survives on the niche `gettypefor` result.
    let has_ne = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne"));
    assert!(
        has_ne,
        "gettypeobject: expected a `ne` pointer null-test discriminant"
    );
    let discriminant_reads = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(&op.kind, OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant")
        })
        .count();
    assert_eq!(
        discriminant_reads, 0,
        "gettypeobject: niche Option must not read an aggregate __discriminant"
    );
}

/// `object_functionstr_type_name` is a pure `match r#type(w_obj) { Some(tp)
/// => .., None => .. }` on the niche `Option<NonNull<PyObject>>`.  The
/// `match` lowers `Rvalue::Discriminant` to a `ne(base, null_mut())` bool
/// null-test and closes the block with a `SwitchInt` on that bool.  Because
/// the switch value is a `SomeBool` (`ne`), its exits MUST carry
/// `ExitCase::Bool` — not `ExitCase::Const(Int)` — or rtype rejects the
/// bool exitswitch against its Int exitcases (`BoolRepr::convert_const(Int)`
/// → "not a bool: Int(1)").  Guards the `lower_switch` niche bool-route.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn niche_option_match_switch_closes_with_bool_exitcase() {
    use crate::model::{ExitCase, ExitSwitch, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "object_functionstr_type_name")
        .expect("lower object_functionstr_type_name");

    // The niche match's Some-discriminant is `not (base is None)`. It is
    // routed through the `If` path (`set_branch`), which may wrap the final
    // bool in an idempotent `bool` UnaryOp. Locate switches on that final
    // `eq(is_none, false)` value (or its wrapper).
    let is_result_ids: std::collections::HashSet<u64> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::BinOp { op: name, .. } if name == "is_" => op.result.as_ref().map(|v| v.id()),
            _ => None,
        })
        .collect();
    let some_result_ids: std::collections::HashSet<u64> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::BinOp { op: name, lhs, .. }
                if name == "eq" && is_result_ids.contains(&lhs.id()) =>
            {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect();
    let mut checked = 0usize;
    for b in &graph.blocks {
        let Some(ExitSwitch::Value(sw)) = &b.exitswitch else {
            continue;
        };
        // The exitswitch is a `bool(ne)` wrap (set_branch) or the raw `ne`.
        let switches_on_nulltest = b.operations.iter().any(|op| {
            op.result.as_ref() == Some(sw)
                && match &op.kind {
                    OpKind::BinOp { op: name, lhs, .. } => {
                        name == "eq" && is_result_ids.contains(&lhs.id())
                    }
                    OpKind::UnaryOp {
                        op: name, operand, ..
                    } => name == "bool" && some_result_ids.contains(&operand.id()),
                    _ => false,
                }
        });
        if !switches_on_nulltest {
            continue;
        }
        // Every exit of a niche null-test switch must be a bool case, never
        // an Int const case.
        for l in &b.exits {
            assert!(
                matches!(&l.exitcase, Some(ExitCase::Bool(_))),
                "object_functionstr_type_name: niche null-test switch exit must be \
                     ExitCase::Bool, got {:?}",
                l.exitcase
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 1,
        "object_functionstr_type_name: expected a niche identity-null-test switch block"
    );
}

/// Anchor the fieldless-enum `PartialEq` fold to the real lowered IR of
/// `strategy_is` — `current.strategy_kind() == expected.strategy_kind()`
/// over the fieldless `StrategyKind`.  Both operands come back in the int
/// bank, so the derived `eq` must fold to `BinOp("eq")`; leaving it a
/// `Method` call blocks the annotator with `Cannot find attribute "eq" on
/// Integer`.  Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn fieldless_enum_eq_fold_real_strategy_is() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_object::dictmultiobject::strategy_is")
        .expect("lower strategy_is");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "eq"
            ))
            .count(),
        0,
        "no residual `eq` Method call survives the fold"
    );
    assert_eq!(
        ops()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq"))
            .count(),
        1,
        "the discriminant comparison lowers to one int `BinOp(eq)`"
    );
}

/// `&mut PyObjectRef` parameters are one-field GC cells. Stores and
/// loads of `*lhs` / `*rhs` are field operations on `MutRef<…>`, and
/// the pointer word is not a raw load or store.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn mut_ref_cell_try_dispatch_binary_special() {
    use crate::model::{OpKind, ValueType};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::objspace::descroperation::try_dispatch_binary_special",
    )
    .expect("lower try_dispatch_binary_special");
    let ops = || {
        graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
    };
    assert!(
        ops().all(|op| { !matches!(op.kind, OpKind::RawStore { .. } | OpKind::RawLoad { .. }) }),
        "a GC-reference `&mut` cell does not raw-load or raw-store the pointer word"
    );
    assert!(
        ops().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. } if mut_ref_value_field(field)
        )),
        "stores through the cell write the `value` field"
    );
    assert!(
        ops().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if mut_ref_value_field(field)
        )),
        "reads through the cell read the `value` field"
    );
    let start = graph.block(graph.startblock);
    let mut cell_params = 0usize;
    for op in &start.operations {
        let OpKind::Input {
            name,
            ty,
            class_root,
        } = &op.kind
        else {
            continue;
        };
        if name != "lhs" && name != "rhs" {
            continue;
        }
        let root = class_root.as_deref().unwrap_or("");
        assert!(
            root.starts_with("MutRef<"),
            "{name} class_root {root:?} is not the cell"
        );
        assert_eq!(ty, &ValueType::Ref(Some(root.to_string())), "{name}");
        cell_params += 1;
    }
    assert_eq!(cell_params, 2, "lhs and rhs are the cell parameters");
}

/// `sub_impl` passes `&mut a` / `&mut b` into
/// `try_dispatch_binary_special`. The cell is built in the caller,
/// passed as the argument, and read back into the local after the call.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn mut_ref_cell_copy_sub_impl() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::objspace::descroperation::sub_impl",
    )
    .expect("lower sub_impl");
    let mut found = false;
    for block in &graph.blocks {
        let ops = &block.operations;
        for (idx, op) in ops.iter().enumerate() {
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            if !target_is_try_dispatch(target) {
                continue;
            }
            let before = ops[..idx].iter().any(|earlier| {
                matches!(
                    &earlier.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } if name.starts_with("MutRef<")
                )
            });
            let after = ops[idx + 1..].iter().any(|later| {
                matches!(
                    &later.kind,
                    OpKind::FieldRead { field, .. } if mut_ref_value_field(field)
                )
            });
            if before && after {
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    assert!(
        found,
        "a MutRef constructor precedes try_dispatch_binary_special and a value FieldRead follows it in the same block"
    );
}

pub(in crate::front::mir) fn mut_ref_value_field(field: &FieldDescriptor) -> bool {
    field.name == "value"
        && field
            .owner_root
            .as_deref()
            .is_some_and(|owner| owner.starts_with("MutRef<"))
}

fn target_is_try_dispatch(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            super::fmt_path_ends_with(segments, &["try_dispatch_binary_special"])
        }
        CallTarget::Method { name, .. } => name == "try_dispatch_binary_special",
        _ => false,
    }
}

/// Every variable carrying the result of a `FunctionPath` call whose
/// last segment is `name`, followed through identity casts and through
/// the links into later blocks.
fn call_results_named(graph: &FunctionGraph, name: &str) -> std::collections::HashSet<u64> {
    let mut ids: std::collections::HashSet<u64> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some(name) => {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect();
    loop {
        let mut grew = false;
        for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
                && segments.last().map(String::as_str) == Some("__cast_instance_intrinsic")
                && let Some(LinkArg::Value(v)) = args.first()
                && ids.contains(&v.id())
                && let Some(result) = &op.result
            {
                grew |= ids.insert(result.id());
            }
        }
        for link in graph.blocks.iter().flat_map(|b| b.exits.iter()) {
            let target = graph.block(link.target);
            for (arg, input) in link.args.iter().zip(&target.inputargs) {
                if let LinkArg::Value(v) = arg
                    && ids.contains(&v.id())
                {
                    grew |= ids.insert(input.id());
                }
            }
        }
        if !grew {
            return ids;
        }
    }
}

/// Rust `==` on two raw pointers compares addresses: the flowspace `is_`
/// (`rptr.py` `rtype_is_` → `ptr_eq`), not the value `eq` that an rstr
/// operand dispatches to `ll_streq`. `is_w`'s bytes arm compares two
/// `w_bytes_block` pointers that way. Ignored by default (loads the real
/// LLBC).
#[test]
#[ignore]
fn raw_pointer_eq_lowers_to_is() {
    let path = format!(
        "{}/../../target/llbc/pyre-object.ullbc",
        env!("CARGO_MANIFEST_DIR")
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(&llbc, "pyre_object::pyobject::is_w").expect("lower is_w");
    let blocks = call_results_named(&graph, "w_bytes_block");
    let compares: Vec<&str> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::BinOp { op, lhs, rhs, .. }
                if blocks.contains(&lhs.id()) && blocks.contains(&rhs.id()) =>
            {
                Some(op.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(compares, ["is_"], "w_bytes_block(a) == w_bytes_block(b)");
}

/// Rust `!=` on two raw pointers is `is_(a, b) == False`: flowspace has
/// no `is not` operation. `w_exception_base_defaults` tests
/// `w_class != PY_NULL`. Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn raw_pointer_ne_lowers_to_negated_is() {
    let path = format!(
        "{}/../../target/llbc/pyre-object.ullbc",
        env!("CARGO_MANIFEST_DIR")
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_object::interp_exceptions::w_exception_base_defaults",
    )
    .expect("lower w_exception_base_defaults");
    let classes = call_results_named(&graph, "lookup_exc_class_for_kind");
    let ops: Vec<&SpaceOperation> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .collect();
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, lhs, .. }
                if op == "ne" && classes.contains(&lhs.id()))),
        "w_class != PY_NULL must not stay a value `ne`"
    );
    let is_results: std::collections::HashSet<u64> = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::BinOp { op: name, lhs, .. } if name == "is_" && classes.contains(&lhs.id()) => {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect();
    assert_eq!(is_results.len(), 1, "w_class != PY_NULL starts with is_");
    let false_consts: std::collections::HashSet<u64> = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::ConstBool(false) => op.result.as_ref().map(|v| v.id()),
            _ => None,
        })
        .collect();
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, lhs, rhs, .. }
                if op == "eq" && is_results.contains(&lhs.id())
                    && false_consts.contains(&rhs.id()))),
        "w_class != PY_NULL is eq(is_(w_class, NULL), False)"
    );
}

/// `jit_ll_int2dec` (`ll_str.py` `ll_int2dec`, elidable) returns an rstr
/// `STR`.  Its residual FUNC.RESULT token is `str`, so the stub callers
/// annotate against is `SomeString` — the value `StringBuilder.build()`
/// passes to the same `space.newutf8` — not the classdef-less instance of
/// `ref`.  Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn string_residual_result_token_is_str() {
    use crate::translator::rtyper::cutover::{STR_RETURN_TYPE, residual_return_shell};
    let path = format!(
        "{}/../../target/llbc/pyre-object.ullbc",
        env!("CARGO_MANIFEST_DIR")
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let program =
        super::build_semantic_program_from_llbc(&llbc).expect("build semantic pyre-object program");
    let int2dec = program
        .functions
        .iter()
        .find(|f| f.name == "jit_ll_int2dec")
        .expect("jit_ll_int2dec is lowered");
    assert_eq!(int2dec.return_type.as_deref(), Some(STR_RETURN_TYPE));
    assert!(
        matches!(
            residual_return_shell(int2dec.return_type.as_deref()),
            Some(crate::annotator::model::SomeValue::String(_))
        ),
        "the `str` residual result is SomeString"
    );
}

/// `rutf8.py` `Utf8StringBuilder` keeps an `rstring.py` `StringBuilder`
/// in `_s`.  `new` / `append_utf8` / `build` are the `SomeStringBuilder`
/// operations (`__majit_stringbuilder_*`) on that attribute, and the
/// receiver `self._s` is the `getfield` of the builder reference, not the
/// address of an inline substructure.  Ignored by default (loads the real
/// LLBC).
#[test]
#[ignore]
fn utf8_string_builder_methods_lower_to_string_builder_markers() {
    use crate::runtime_names::shims::{
        STRINGBUILDER_APPEND, STRINGBUILDER_BUILD, STRINGBUILDER_NEW,
    };
    let path = format!(
        "{}/../../target/llbc/pyre-object.ullbc",
        env!("CARGO_MANIFEST_DIR")
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    for (method, marker) in [
        ("new", STRINGBUILDER_NEW),
        ("append_utf8", STRINGBUILDER_APPEND),
        ("build", STRINGBUILDER_BUILD),
    ] {
        let name = format!("pyre_object::rutf8::<Impl>::{method}");
        let graph = super::lower_function(&llbc, &name).expect("lower Utf8StringBuilder");
        let ops: Vec<&SpaceOperation> = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .collect();
        let marker_args: Vec<&Vec<LinkArg>> = ops
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.len() == 1 && segments[0] == marker => Some(args),
                _ => None,
            })
            .collect();
        assert_eq!(marker_args.len(), 1, "{name} calls {marker} once");
        let s_reads: Vec<&SpaceOperation> = ops
            .iter()
            .copied()
            .filter(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "_s"))
            .collect();
        assert!(
            s_reads.iter().all(|op| !matches!(&op.kind,
                    OpKind::FieldRead { field, .. } if field.taken_by_address)),
            "{name}: `self._s` is not taken by address"
        );
        if method != "new" {
            let receiver = match marker_args[0].first() {
                Some(LinkArg::Value(v)) => v.id(),
                other => panic!("{name}: {marker} receiver {other:?}"),
            };
            assert!(
                s_reads
                    .iter()
                    .any(|op| op.result.as_ref().map(|v| v.id()) == Some(receiver)),
                "{name}: the {marker} receiver is the `self._s` getfield"
            );
        }
    }
}

/// `get_w_locals` is `getdebug_data().map_or(PY_NULL, |data| data.w_locals)`
/// over `Option<&FrameDebugData>` — a SHARED-reference niche Option. Its
/// `Some` payload must alias the base pointer (no aggregate `__pos_0`
/// FieldRead), the same fold the `&mut`/`NonNull` niche arms already apply.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn shared_ref_niche_option_get_w_locals_no_pos0_read() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "get_w_locals").expect("lower get_w_locals");
    let pos0_reads = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0"))
        .count();
    assert_eq!(
        pos0_reads, 0,
        "get_w_locals: shared-ref niche Option<&FrameDebugData> Some payload \
             must alias the base pointer, not read an aggregate __pos_0"
    );
}

/// Regression: a `&mut self` setter's niche `Option<&mut Self>` must STILL
/// fold (the widening must not perturb the pre-existing mut arm). Ignored
/// (loads the real LLBC).
#[test]
#[ignore]
fn mut_ref_niche_option_set_chunk_size_still_no_pos0_read() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    // Any &mut-self __majit_wrap setter; set_CHUNK_SIZE takes `&mut self`.
    let graph = super::lower_function(&llbc, "__majit_wrap_set_CHUNK_SIZE").expect("lower setter");
    let pos0_reads = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(&op.kind, OpKind::FieldRead { field, .. }
            if field.name == "__pos_0"
                && field.owner_root.as_deref().is_some_and(|owner| {
                    owner.contains("Option") && owner.ends_with("::Some")
                }))
        })
        .count();
    assert_eq!(
        pos0_reads, 0,
        "the &mut niche arm must keep folding __pos_0"
    );
}

/// PyPy `BuiltinCode.docstring` is a nullable RPython string and
/// `getdocstring` delegates to `space.newtext_or_none`: there is no enum
/// object carrying a `__discriminant` attribute. Rust spells the field as
/// `Option<&'static str>`, but the front's `SomeString` projection must
/// survive through that Option as one nullable GC pointer.
#[test]
#[ignore]
fn builtin_docstring_option_str_is_nullable_somestring() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "builtin_code_get_docstring")
        .expect("lower builtin_code_get_docstring");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(&op.kind, OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant"))
            .count(),
        0,
        "Option<&str> must test the nullable SomeString, not read an enum tag"
    );
    assert!(
        ops().any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_")),
        "Option<&str> match must contain an identity null-test"
    );
}

/// PyPy `W_TextIOWrapper.pending_bytes` is `None` or one ordinary list.
/// The Rust `Option<Vec<Vec<u8>>>` spelling therefore translates to a
/// nullable `SomeList`, not an enum shell around the list.
#[test]
#[ignore]
fn textio_pending_bytes_option_vec_is_nullable_somelist() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "write_flush").expect("lower TextIOWrapper.write_flush");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(&op.kind, OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant"))
            .count(),
        0,
        "Option<Vec<_>> must test the nullable SomeList, not read an enum tag"
    );
    assert!(
        !ops().any(|op| matches!(&op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "is_some" || name == "is_none")),
        "the Option predicate must not survive as a residual Rust method"
    );
}

/// RPython `ll_getitem_fast` indexes a list's items array after checking
/// the list's logical length. The pyre object-list path must keep that
/// direct shape instead of manufacturing a Rust fat slice with
/// `from_raw_parts(base, self.length)`: such a slice is neither an RPython
/// source operation nor representable by the one-word GC-array model.
/// Loads the pyre-object LLBC, so ignored by default.
#[test]
#[ignore]
fn object_list_getitem_keeps_logical_length_without_a_raw_slice() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(&llbc, "w_list_getitem").expect("lower w_list_getitem");
    let from_raw_parts_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("from_raw_parts")
            )
        })
        .count();
    assert_eq!(
        from_raw_parts_calls, 0,
        "object-list getitem must read logical length and items directly, \
             like rlist.ll_getitem_fast"
    );
}

/// PyPy `float_repr` reaches `rfloat.formatd`, whose translated return is
/// one low-level string pointer. The real interpreter LLBC must therefore
/// retarget RustPython's three-word `String` formatter to the one-word
/// `Utf8Str*` (`rstr.py` `rpy_string`) residual wrapper.
#[test]
#[ignore]
fn float_repr_uses_the_rpython_string_residual_abi() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-interpreter LLBC");
    let graph = super::lower_function(&llbc, "format_float_repr").expect("lower format_float_repr");
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "str")),
        "float to_string must not lower to UnaryOp str (int_str/f>r): {graph:#?}"
    );
    let paths: Vec<Vec<String>> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(paths.iter().any(|segments| {
        segments
            == &[
                "pyre_interpreter".to_string(),
                "display".to_string(),
                "jit_format_float_repr_rstr".to_string(),
            ]
    }));
    assert!(!paths.iter().any(|segments| {
        segments
            == &[
                "rustpython_literal".to_string(),
                "float".to_string(),
                "to_string".to_string(),
            ]
    }));
}

/// PyPy's complex repr keeps its positive-zero/sign branch in
/// `W_ComplexObject.descr_repr` and calls `repr_format` for each component.
/// The interpreter graph must retain that shape rather than collapsing the
/// whole operation into RustPython's foreign complex formatter.
#[test]
#[ignore]
fn complex_repr_keeps_the_pypy_builder_and_float_formatter_calls() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-interpreter LLBC");
    let graph = super::lower_function(&llbc, "complex_repr_string").expect("lower complex repr");
    let paths: Vec<Vec<String>> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(paths.iter().any(|segments| {
        segments
            == &[
                "pyre_interpreter".to_string(),
                "typedef".to_string(),
                "jit_format_complex_component_repr_rstr".to_string(),
            ]
    }));
    assert!(!paths.iter().any(|segments| {
        segments
            == &[
                "rustpython_literal".to_string(),
                "complex".to_string(),
                "to_string".to_string(),
            ]
    }));
    assert!(!paths.iter().any(|segments| {
        segments
            == &[
                "core".to_string(),
                "f64".to_string(),
                "<Impl>".to_string(),
                "is_sign_positive".to_string(),
            ]
    }));
    assert!(paths.iter().any(|segments| {
        segments == &["longlong2float".to_string(), "float2longlong".to_string()]
    }));
    assert!(
        !paths.iter().any(|segments| {
            segments
                .last()
                .is_some_and(|leaf| leaf == "push" || leaf == "push_str")
        }),
        "complex repr String appends must use the RPython builder lift: {paths:#?}"
    );
}

/// Regression: `RBigInt::digits` uses `from_raw_parts(base, capacity)` — the
/// block's own length — so its header alias IS sound and must STILL fold
/// (the P1 fix removes only the object-list arm). The residual
/// `from_raw_parts` must be gone. Loads the majit-rlib LLBC, ignored.
#[test]
#[ignore]
fn rbigint_digits_still_folds_capacity_length_view() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    let graph = super::lower_function(&llbc, "rbigint::<Impl>::digits")
        .or_else(|_| super::lower_function(&llbc, "digits"))
        .expect("lower digits");
    let from_raw_parts_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("from_raw_parts")
            )
        })
        .count();
    assert_eq!(
        from_raw_parts_calls, 0,
        "rbigint digits (capacity-length view) must still fold to the \
             items-block header, leaving no residual from_raw_parts"
    );
}

/// `rbuilder.py ll_grow_by` spells its growth as three `ovfcheck`s under
/// one `except OverflowError: raise MemoryError`.  The port matches each
/// `checked_add` on the spot, so all three must lift to `add_ovf` with an
/// OverflowError edge and leave no residual `checked_add`.  Loads the real
/// pyre-object LLBC, so ignored by default.
#[test]
#[ignore]
fn rbuilder_ll_grow_by_ovfchecks_lift_to_add_ovf() {
    use crate::model::{CallTarget, ExitSwitch, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(&llbc, "ll_grow_by").expect("lower ll_grow_by");
    let residual_checked = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("checked_add")
            )
        })
        .count();
    assert_eq!(residual_checked, 0, "graph: {graph:#?}");
    let ovf_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .filter(|b| {
            matches!(
                b.operations.last().map(|op| &op.kind),
                Some(OpKind::BinOp { op, .. }) if op == "add_ovf"
            )
        })
        .count();
    assert_eq!(ovf_blocks, 3, "graph: {graph:#?}");
}

/// `bytes_block_chars` is `rstr.py` `ll_chars`: the
/// `from_raw_parts(chars, length)` view must fold to the managed `STR`
/// header instead of surviving as an opaque Rust call. Loads the real
/// pyre-object LLBC, so ignored by default.
#[test]
#[ignore]
fn bytes_block_chars_folds_header_length_view() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(&llbc, "bytes_block_chars").expect("lower bytes_block_chars");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("from_raw_parts")
            )
        })
        .collect::<Vec<_>>();
    assert!(
        residual.is_empty(),
        "bytes_block_chars must lower to the managed bytes-block array; \
             residual calls: {residual:#?}; graph: {graph:#?}"
    );
    let dead_storage_reads = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.owner_root.as_deref() == Some("BytesBlock")
                        && matches!(field.name.as_str(), "chars" | "length")
            )
        })
        .count();
    assert_eq!(
        dead_storage_reads, 0,
        "the owner alias must let the ordinary dead-op pass remove the \
             interior chars/length shell; graph: {graph:#?}"
    );
}

/// The Rust interpreter reloads a possibly-moved pointer explicitly;
/// RPython's translated graph instead gets the updated GCREF from
/// `gc_restore_root`.  The source adapter must therefore lower to an
/// identity graph and must not leak the `majit_gc` query into annotation.
/// Loads the real pyre-object LLBC, so ignored by default.
#[test]
#[ignore]
fn gc_current_object_address_adapter_folds_to_identity() {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(&llbc, "try_gc_current_object_address")
        .expect("lower try_gc_current_object_address");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str)
                        == Some("gc_current_object_address")
            )
        })
        .collect::<Vec<_>>();
    assert!(
        residual.is_empty(),
        "the RPython-shaped graph reloads its GCREF from the root map; \
             residual forwarding queries: {residual:#?}; graph: {graph:#?}"
    );

    let input = graph
        .block(graph.startblock)
        .inputargs
        .first()
        .expect("adapter input");
    let returned = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .expect("adapter return link");
    assert!(
        matches!(returned, LinkArg::Value(value) if value == input),
        "the adapter must return its input GCREF unchanged; graph: {graph:#?}"
    );
}

/// One `__cast_instance_intrinsic` whose root is a `{vtable}` struct.
/// The slot read's base is that cast when the metadata word is a raw
/// pointer; the operand is the word itself.
fn peel_vtable_instance_cast(graph: &FunctionGraph, kind: &OpKind) -> OpKind {
    let Some(root) = crate::model::cast_instance_root(kind) else {
        return kind.clone();
    };
    if !root.ends_with("::{vtable}") {
        return kind.clone();
    }
    let OpKind::Call { args, .. } = kind else {
        return kind.clone();
    };
    let Some(inner) = args.first().and_then(crate::model::LinkArg::as_variable) else {
        return kind.clone();
    };
    let inner_ops = producers_of(graph, inner);
    if inner_ops.len() == 1 {
        inner_ops[0].kind.clone()
    } else {
        kind.clone()
    }
}

fn producers_of<'a>(graph: &'a FunctionGraph, var: &Variable) -> Vec<&'a SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| op.result.as_ref() == Some(var))
        .collect()
}

/// `method_<name>` must be loaded from the vtable word of the same
/// field the call passes as `self`. The data word is the strategy
/// static; indexing it reads the bytes that follow that static.
fn assert_strategy_slot_indexes_vtable(
    graph: &FunctionGraph,
    function: &str,
    method: &str,
    funcptr: &Variable,
    args: &[Variable],
) {
    let producers = producers_of(graph, funcptr);
    assert_eq!(
        producers.len(),
        1,
        "{function}::{method} funcptr producers: {producers:?}"
    );
    let OpKind::FieldRead {
        base: vtable,
        field,
        ty: ValueType::Int,
        ..
    } = &producers[0].kind
    else {
        panic!(
            "{function}::{method} slot must be an int FieldRead: {:?}",
            producers[0].kind
        );
    };
    assert_eq!(field.name, format!("method_{method}"));
    assert!(field.vec_part.is_none());
    let meta_ops = producers_of(graph, vtable);
    assert_eq!(
        meta_ops.len(),
        1,
        "{function}::{method} vtable-word producers: {meta_ops:?}"
    );
    // A raw-pointer deref of the metadata word narrows through
    // `__cast_instance_intrinsic[<Trait>::{vtable}]` before the slot
    // read. The cast is a classdef paint; the word it retypes is the
    // metadata field read.
    let meta_kind = peel_vtable_instance_cast(graph, &meta_ops[0].kind);
    let OpKind::FieldRead {
        base: struct_base,
        field: meta,
        ty: ValueType::Ref(None),
        ..
    } = &meta_kind
    else {
        panic!("{function}::{method} vtable word must be a ref FieldRead: {meta_kind:?}");
    };
    assert_eq!(
        meta.vec_part,
        Some(crate::model::VecFieldPart::FatVtable),
        "{function}::{method} metadata field: {meta:?}"
    );
    let data_words: Vec<&Variable> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                base, field: data, ..
            } if base == struct_base
                && data.name == meta.name
                && data.owner_root == meta.owner_root
                && data.vec_part.is_none() =>
            {
                op.result.as_ref()
            }
            _ => None,
        })
        .collect();
    assert!(
        args.first()
            .is_some_and(|recv| forwards_to_any(graph, recv, &data_words)),
        "{function}::{method} receiver must stay the data word of {}.{}: args={args:?} data={data_words:?}",
        meta.owner_root.as_deref().unwrap_or("?"),
        meta.name
    );
}

/// `start` is one of `targets`, or a chain of one-predecessor block
/// arguments that ends at one of them. A live field read is re-bound
/// at each join; the call still receives that read.
fn forwards_to_any(graph: &FunctionGraph, start: &Variable, targets: &[&Variable]) -> bool {
    let mut current = start.clone();
    let mut seen = Vec::new();
    for _ in 0..64 {
        if targets.iter().any(|target| *target == &current) {
            return true;
        }
        if seen.iter().any(|seen| seen == &current) {
            return false;
        }
        seen.push(current.clone());
        if !producers_of(graph, &current).is_empty() {
            return false;
        }
        let Some(phi) = graph
            .blocks
            .iter()
            .find(|block| block.inputargs.iter().any(|arg| arg == &current))
        else {
            return false;
        };
        let Some(index) = phi.inputargs.iter().position(|arg| arg == &current) else {
            return false;
        };
        let preds = graph.predecessors(phi.id);
        if preds.len() != 1 {
            return false;
        }
        let Some(next) = graph
            .block(preds[0])
            .exits
            .iter()
            .find(|link| link.target == phi.id)
            .and_then(|link| link.args.get(index))
            .and_then(crate::model::LinkArg::as_variable)
        else {
            return false;
        };
        current = next.clone();
    }
    false
}

/// The Rust trait-object spelling of PyPy's
/// `W_DictMultiObject.getitem` / `getitem_str` strategy dispatch must
/// retain both halves RPython's PBC call carries: the concrete vtable
/// field read and the closed `DictStrategy` implementation family.
/// This uses the real extracted owner because a hand-written fixture can
/// accidentally make the trait leaf and vtable field shape agree.
#[test]
#[ignore]
fn dict_strategy_vtable_call_keeps_its_family_identity() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real pyre-object LLBC");
    for (function, method) in [
        ("w_dict_lookup", "getitem"),
        ("w_dict_getitem_str_hashed", "getitem_str_hashed"),
    ] {
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|error| panic!("lower {function}: {error}"));
        let calls: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::IndirectCall {
                    funcptr,
                    family_key: Some(family_key),
                    ..
                } => Some((funcptr.clone(), family_key.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "{function} must carry one DictStrategy vtable call: {calls:?}"
        );
        assert_eq!(
            calls[0].1,
            ("DictStrategy".to_string(), method.to_string()),
            "{function}'s MIR vtable slot must name the same PBC family as the trait impls"
        );
        let funcptr = &calls[0].0;
        let producers: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| op.result.as_ref() == Some(funcptr))
            .collect();
        assert_eq!(
            producers.len(),
            1,
            "{function}'s indirect funcptr must have one concrete producer"
        );
        assert!(
            matches!(
                &producers[0].kind,
                OpKind::FieldRead { field, ty: ValueType::Int, .. }
                    if field.name == format!("method_{method}")
            ),
            "{function}'s vtable method pointer must be a raw-pointer/int FieldRead: {:?}",
            producers[0].kind
        );
    }

    // `w_dict_lookup_checked` reads `DictStrategyRef.imp` (`&dyn
    // DictStrategy`) and calls through `ptr_metadata` of that field.
    // The slot load has to index the vtable word. The call receiver
    // stays the data word: a strategy static, not the vtable.
    {
        let function = "w_dict_lookup_checked";
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|error| panic!("lower {function}: {error}"));
        let calls: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::IndirectCall {
                    funcptr,
                    args,
                    family_key: Some(family_key),
                    ..
                } if family_key.0 == "DictStrategy" => {
                    Some((funcptr.clone(), args.clone(), family_key.1.clone()))
                }
                _ => None,
            })
            .collect();
        assert!(
            calls.iter().any(|(_, _, method)| method == "getitem"),
            "{function} must dispatch DictStrategy::getitem: {calls:?}"
        );
        for (funcptr, args, method) in &calls {
            assert_strategy_slot_indexes_vtable(&graph, function, method, funcptr, args);
        }
    }

    let program =
        super::build_semantic_program_from_llbc(&llbc).expect("build semantic pyre-object program");
    for method in ["getitem", "getitem_str_hashed"] {
        let members: Vec<_> = program
            .functions
            .iter()
            .filter(|function| {
                function.name == method && function.trait_root.as_deref() == Some("DictStrategy")
            })
            .map(|function| (function.self_ty_root.clone(), function.return_type.clone()))
            .collect();
        assert!(
            members.iter().any(|(owner, _)| owner.is_some()),
            "DictStrategy.{method} must retain concrete impl owners: {members:?}"
        );
        assert!(
            members
                .iter()
                .all(|(_, result)| result.as_deref() == Some("ref")),
            "every DictStrategy.{method} PBC member must carry FuncType.RESULT: {members:?}"
        );
    }
}

/// `Vec<PyObjectRef>::deref_mut` plus `iter_mut` and a store through the
/// slot write the vec's own item buffer (`ll_vec_items` / `ll_slice_setitem_fast`).
/// A `gcarray_from_pyobject_vec` copy would drop those stores.
#[test]
fn object_vec_deref_mut_iter_mut_store_writes_through_the_buffer() {
    use crate::model::{CallTarget, OpKind};
    let span = fixture_span();
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let obj = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let objptr = serde_json::json!({"RawPtr": [obj, "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 1, "generics": {"types": [objptr.clone()]}}
    });
    let vec_mut = serde_json::json!({"Ref": ["'_", vec_ty, "Mut"]});
    let slice_ty = serde_json::json!({
        "Ref": ["'_", {"Slice": [objptr.clone(), null]}, "Mut"]
    });
    let iter_ty = serde_json::json!({
        "Adt": {"id": 2, "generics": {"types": [objptr.clone()]}}
    });
    let opt_ty = serde_json::json!({
        "Adt": {"id": 3, "generics": {"types": [
            {"Ref": ["'_", objptr.clone(), "Mut"]}
        ]}}
    });
    let slot_ty = serde_json::json!({"Ref": ["'_", objptr.clone(), "Mut"]});
    let iter_mut_ref = serde_json::json!({"Ref": ["'_", iter_ty.clone(), "Mut"]});
    let unit = serde_json::json!({"Tuple": []});
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| serde_json::json!({"index": index, "name": name, "span": span, "ty": ty});
    let call = |fun: u64,
                args: Vec<serde_json::Value>,
                dest: serde_json::Value,
                target: u64,
                statements: Vec<serde_json::Value>| {
        serde_json::json!({
            "statements": statements,
            "terminator": {"span": span, "kind": {"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": generics}},
                    "args": args,
                    "dest": dest
                },
                "target": target,
                "on_unwind": 5
            }}}
        })
    };
    let init_ret = serde_json::json!({"span": span, "kind": {"Assign": [
        place(0, &unit),
        {"Aggregate": ["Tuple", []]}
    ]}});
    let meta = |path: &[&str]| fixture_item_meta(ident_path(path));
    let fun = |id: u64,
               path: &[&str],
               inputs: Vec<serde_json::Value>,
               output: serde_json::Value,
               body: serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": meta(path),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
            "body": body
        })
    };
    let field_place = serde_json::json!({
        "kind": {"Projection": [place(5, &opt_ty), {"Field": [null, 0]}]},
        "ty": slot_ty
    });
    let deref_slot = serde_json::json!({
        "kind": {"Projection": [place(6, &slot_ty), "Deref"]},
        "ty": objptr
    });
    let refresh_body = serde_json::json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 2, "locals": [
            local(0, Some("ret"), &unit),
            local(1, Some("vec"), &vec_mut),
            local(2, Some("new"), &objptr),
            local(3, Some("slice"), &slice_ty),
            local(4, Some("iter"), &iter_ty),
            local(5, Some("opt"), &opt_ty),
            local(6, Some("slot"), &slot_ty),
            local(7, Some("iter_ref"), &iter_mut_ref),
        ]},
        "body": [
            call(1, vec![serde_json::json!({"Copy": place(1, &vec_mut)})], place(3, &slice_ty), 1, vec![init_ret]),
            call(2, vec![serde_json::json!({"Copy": place(3, &slice_ty)})], place(4, &iter_ty), 2, vec![]),
            call(3, vec![serde_json::json!({"Copy": place(7, &iter_mut_ref)})], place(5, &opt_ty), 3, vec![
                serde_json::json!({"span": span, "kind": {"Assign": [
                    place(7, &iter_mut_ref),
                    {"Ref": {"place": place(4, &iter_ty), "kind": "Mut", "ptr_metadata": null}}
                ]}})
            ]),
            {"statements": [
                {"span": span, "kind": {"Assign": [
                    place(6, &slot_ty),
                    {"Use": [{"Copy": field_place}, "No"]}
                ]}},
                {"span": span, "kind": {"Assign": [
                    deref_slot,
                    {"Use": [{"Copy": place(2, &objptr)}, "No"]}
                ]}}
            ], "terminator": {"span": span, "kind": {"Goto": {"target": 4}}}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ]
    }});
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["PyObject"]), "kind": {"Struct": []}},
                {"def_id": 1, "item_meta": meta(&["alloc", "vec", "Vec"]), "kind": {"Struct": []}},
                {"def_id": 2, "item_meta": meta(&["core", "slice", "iter", "IterMut"]), "kind": {"Struct": []}},
                {"def_id": 3, "item_meta": meta(&["core", "option", "Option"]), "kind": {"Enum": []}},
            ],
            "fun_decls": [
                fun(0, &["fixture", "refresh"], vec![vec_mut.clone(), objptr.clone()], unit.clone(), refresh_body),
                fun(1, &["alloc", "vec", "<Impl>", "deref_mut"], vec![vec_mut.clone()], slice_ty.clone(), serde_json::json!("Opaque")),
                fun(2, &["core", "slice", "<Impl>", "iter_mut"], vec![slice_ty.clone()], iter_ty.clone(), serde_json::json!("Opaque")),
                fun(3, &["core", "slice", "iter", "<Impl>", "next"], vec![iter_mut_ref.clone()], opt_ty.clone(), serde_json::json!("Opaque")),
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = super::lower_function(&llbc, "refresh")
        .unwrap_or_else(|err| panic!("lower refresh: {err}"));
    let leaves: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            _ => None,
        })
        .collect();
    assert!(
        leaves.iter().any(|l| l == "ll_vec_items_r"),
        "deref_mut must alias the vec buffer: {leaves:?}"
    );
    assert!(
        leaves.iter().any(|l| l == "ll_slice_setitem_fast_r"),
        "iter_mut store must write through ptr: {leaves:?}"
    );
    assert!(
        !leaves.iter().any(|l| l == "gcarray_from_pyobject_vec"),
        "deref_mut must not copy into a gcarray: {leaves:?}"
    );
}

fn object_vec_whole_list_fixture(
    name: &str,
    callee_path: &[&str],
    callee_id: u64,
    vec_mut: bool,
    dest_is_mut_slice: bool,
    extra_funs: Vec<serde_json::Value>,
    extra_blocks: Vec<serde_json::Value>,
    extra_locals: Vec<serde_json::Value>,
    extra_args: Vec<serde_json::Value>,
    extra_inputs: Vec<serde_json::Value>,
) -> (Llbc, crate::model::FunctionGraph) {
    let span = fixture_span();
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let obj = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let objptr = serde_json::json!({"RawPtr": [obj, "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 1, "generics": {"types": [objptr.clone()]}}
    });
    let vec_ref = if vec_mut {
        serde_json::json!({"Ref": ["'_", vec_ty, "Mut"]})
    } else {
        serde_json::json!({"Ref": ["'_", vec_ty, "Shared"]})
    };
    let slice_kind = if dest_is_mut_slice { "Mut" } else { "Shared" };
    let slice_ty = serde_json::json!({
        "Ref": ["'_", {"Slice": [objptr.clone(), null]}, slice_kind]
    });
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| serde_json::json!({"index": index, "name": name, "span": span, "ty": ty});
    let mut locals = vec![
        local(0, Some("ret"), &slice_ty),
        local(1, Some("vec"), &vec_ref),
    ];
    locals.extend(extra_locals);
    let mut inputs = vec![vec_ref.clone()];
    inputs.extend(extra_inputs);
    let arg_count = inputs.len() as u64;
    let mut args = vec![serde_json::json!({"Copy": place(1, &vec_ref)})];
    args.extend(extra_args);
    let mut body_blocks = vec![serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span, "kind": {"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": callee_id}, "generics": generics}},
                "args": args,
                "dest": place(0, &slice_ty)
            },
            "target": 1,
            "on_unwind": extra_blocks.len() as u64 + 2
        }}}
    })];
    body_blocks.extend(extra_blocks);
    body_blocks.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span, "kind": "Return"}
    }));
    body_blocks.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span, "kind": "UnwindResume"}
    }));
    let meta = |path: &[&str]| fixture_item_meta(ident_path(path));
    let fun = |id: u64,
               path: &[&str],
               ins: Vec<serde_json::Value>,
               output: serde_json::Value,
               body: serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": meta(path),
            "signature": {"is_unsafe": false, "inputs": ins, "output": output},
            "body": body
        })
    };
    let view_body = serde_json::json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": arg_count, "locals": locals},
        "body": body_blocks
    }});
    let mut fun_decls = vec![
        fun(0, &["fixture", name], inputs, slice_ty.clone(), view_body),
        fun(
            callee_id,
            callee_path,
            vec![vec_ref],
            slice_ty,
            serde_json::json!("Opaque"),
        ),
    ];
    fun_decls.extend(extra_funs);
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["PyObject"]), "kind": {"Struct": []}},
                {"def_id": 1, "item_meta": meta(&["alloc", "vec", "Vec"]), "kind": {"Struct": []}},
            ],
            "fun_decls": fun_decls,
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph =
        super::lower_function(&llbc, name).unwrap_or_else(|err| panic!("lower {name}: {err}"));
    (llbc, graph)
}

fn object_vec_call_leaves(graph: &crate::model::FunctionGraph) -> Vec<String> {
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            _ => None,
        })
        .collect()
}

/// `Vec<PyObjectRef>::deref` (`&vec`) is the length-prefixed object array.
#[test]
fn object_vec_deref_builds_the_gcarray() {
    let (_llbc, graph) = object_vec_whole_list_fixture(
        "view",
        &["alloc", "vec", "<Impl>", "deref"],
        1,
        false,
        false,
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    );
    let leaves = object_vec_call_leaves(&graph);
    assert!(
        leaves.iter().any(|l| l == "gcarray_from_pyobject_vec"),
        "deref must copy into a gcarray: {leaves:?}"
    );
    let input = graph
        .block(graph.startblock)
        .inputargs
        .first()
        .expect("vec input");
    let returned = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .expect("return link");
    assert!(
        !matches!(returned, crate::model::LinkArg::Value(value) if value == input),
        "deref must not alias the header; graph: {graph:#?}"
    );
}

/// `Vec<PyObjectRef>::deref_mut` (`&mut vec`) is the header address, the
/// same `SomeList` as `&mut vec[..]`. A gcarray copy would drop stores.
#[test]
fn object_vec_deref_mut_aliases_the_header() {
    let (_llbc, graph) = object_vec_whole_list_fixture(
        "view_mut",
        &["alloc", "vec", "<Impl>", "deref_mut"],
        1,
        true,
        true,
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    );
    let leaves = object_vec_call_leaves(&graph);
    assert!(
        !leaves.iter().any(|l| l == "gcarray_from_pyobject_vec"),
        "deref_mut must not copy into a gcarray: {leaves:?}"
    );
    assert!(
        !leaves.iter().any(|l| l == "ll_vec_items_r"),
        "deref_mut must not split the header into a pair: {leaves:?}"
    );
    let input = graph
        .block(graph.startblock)
        .inputargs
        .first()
        .expect("vec input");
    let returned = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .expect("return link");
    assert!(
        matches!(returned, crate::model::LinkArg::Value(value) if value == input),
        "deref_mut must alias the vec header; graph: {graph:#?}"
    );
}

/// `&mut vec` passed to `shadow_stack_copy_range` is that vec. The call
/// is `shadow_stack_copy_range_into_vec`.
#[test]
fn object_vec_deref_mut_retargets_shadow_stack_copy_into_vec() {
    let span = fixture_span();
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let obj = serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
    let objptr = serde_json::json!({"RawPtr": [obj, "Mut"]});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 1, "generics": {"types": [objptr.clone()]}}
    });
    let vec_mut = serde_json::json!({"Ref": ["'_", vec_ty, "Mut"]});
    let slice_ty = serde_json::json!({
        "Ref": ["'_", {"Slice": [objptr.clone(), null]}, "Mut"]
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let unit = serde_json::json!({"Tuple": []});
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| serde_json::json!({"index": index, "name": name, "span": span, "ty": ty});
    let meta = |path: &[&str]| fixture_item_meta(ident_path(path));
    let fun = |id: u64,
               path: &[&str],
               inputs: Vec<serde_json::Value>,
               output: serde_json::Value,
               body: serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": meta(path),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
            "body": body
        })
    };
    let init_ret = serde_json::json!({"span": span, "kind": {"Assign": [
        place(0, &unit),
        {"Aggregate": ["Tuple", []]}
    ]}});
    let copy_body = serde_json::json!({"Unstructured": {
        "span": span,
        "locals": {"arg_count": 2, "locals": [
            local(0, Some("ret"), &unit),
            local(1, Some("base"), &usize_ty),
            local(2, Some("vec"), &vec_mut),
            local(3, Some("slice"), &slice_ty),
        ]},
        "body": [
            {"statements": [init_ret], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": [{"Copy": place(2, &vec_mut)}],
                    "dest": place(3, &slice_ty)
                },
                "target": 1,
                "on_unwind": 3
            }}}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                    "args": [
                        {"Copy": place(1, &usize_ty)},
                        {"Copy": place(3, &slice_ty)}
                    ],
                    "dest": place(0, &unit)
                },
                "target": 2,
                "on_unwind": 3
            }}}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ]
    }});
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["PyObject"]), "kind": {"Struct": []}},
                {"def_id": 1, "item_meta": meta(&["alloc", "vec", "Vec"]), "kind": {"Struct": []}},
            ],
            "fun_decls": [
                fun(0, &["fixture", "copy_into"], vec![usize_ty.clone(), vec_mut.clone()], unit.clone(), copy_body),
                fun(1, &["alloc", "vec", "<Impl>", "deref_mut"], vec![vec_mut.clone()], slice_ty.clone(), serde_json::json!("Opaque")),
                fun(2, &["pyre_object", "gc_roots", "shadow_stack_copy_range"], vec![usize_ty, slice_ty], unit, serde_json::json!("Opaque")),
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = super::lower_function(&llbc, "copy_into")
        .unwrap_or_else(|err| panic!("lower copy_into: {err}"));
    let paths: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.join("::")),
            _ => None,
        })
        .collect();
    assert!(
        paths
            .iter()
            .any(|p| p.ends_with("shadow_stack_copy_range_into_vec")),
        "deref_mut dest must retarget the copy: {paths:?}"
    );
    assert!(
        !paths
            .iter()
            .any(|p| p.ends_with("gcarray_from_pyobject_vec")),
        "deref_mut must not copy into a gcarray: {paths:?}"
    );
}

fn corpus_llbc() -> Llbc {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc");
    Llbc::load(path).expect("load corpus")
}

fn corpus_call_leaves(name: &str) -> Vec<(String, usize)> {
    let graph = super::lower_function(&corpus_llbc(), name).unwrap_or_else(|e| panic!("{e}"));
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => segments.last().map(|leaf| (leaf.clone(), args.len())),
            _ => None,
        })
        .collect()
}

fn corpus_call_paths(name: &str) -> Vec<String> {
    let graph = super::lower_function(&corpus_llbc(), name).unwrap_or_else(|e| panic!("{e}"));
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.join("::")),
            _ => None,
        })
        .collect()
}

/// `drop(x)` of an integer binds nothing and calls nothing.
#[test]
fn mem_drop_of_int_lowers_to_no_call() {
    for name in ["mem_drop_i64", "mem_drop_usize"] {
        let calls = corpus_call_paths(name);
        assert!(calls.is_empty(), "{name}: {calls:?}");
    }
}

/// `drop(roots)` before the function ends is the guard's close.
#[test]
fn mem_drop_of_root_scope_lowers_to_one_close() {
    let calls = corpus_call_paths("mem_drop_root_scope");
    let closes = calls
        .iter()
        .filter(|path| path.split("::").any(|seg| seg == "root_scope_close"))
        .count();
    assert_eq!(closes, 1, "{calls:?}");
    assert!(
        !calls.iter().any(|path| path.contains("mem::drop")),
        "{calls:?}"
    );
}

/// `vec![item; count]` on a one-word `Vec` is one `ll_vec_alloc_and_set_*`
/// call. `alloc::vec::from_elem` has no address.
#[test]
fn repeat_vec_lowers_to_alloc_and_set() {
    for (name, leaf) in [
        ("repeat_vec_i64", "ll_vec_alloc_and_set_i"),
        ("repeat_vec_ptr", "ll_vec_alloc_and_set_r"),
    ] {
        let calls = corpus_call_leaves(name);
        assert_eq!(
            calls
                .iter()
                .filter(|(got, n)| got == leaf && *n == 2)
                .count(),
            1,
            "{name}: {calls:?}"
        );
        assert!(
            !calls.iter().any(|(got, _)| got == "from_elem"),
            "{name}: {calls:?}"
        );
    }
}

/// A pair-slice index fail arm calls `ll_slice_bounds_panic`, a DLI
/// residual `remove_assertion_errors` cannot fold away.
#[test]
fn pair_slice_index_keeps_the_bounds_panic() {
    let calls = corpus_call_leaves("first_ref");
    assert!(
        calls
            .iter()
            .any(|(leaf, n)| leaf == "ll_slice_bounds_panic" && *n == 0),
        "{calls:?}"
    );
}

/// `dst.copy_from_slice(src)` over two pair slices is
/// `ll_slice_arraycopy(src, dst, 0, 0, dst_len)`.
#[test]
fn pair_slice_copy_from_slice_is_arraycopy_of_the_destination_length() {
    let calls = corpus_call_leaves("copy_slice_from_slice");
    assert!(
        calls
            .iter()
            .any(|(leaf, n)| leaf == "ll_slice_arraycopy_r" && *n == 5),
        "{calls:?}"
    );
    assert!(
        !calls.iter().any(|(leaf, _)| leaf == "copy_from_slice"),
        "{calls:?}"
    );
}

/// `<[T]>::as_ptr` / `as_mut_ptr` on a pair return the pointer word.
#[test]
fn pair_slice_as_ptr_is_the_pointer_word() {
    for name in ["ptr_of_slice", "mut_ptr_of_slice"] {
        let graph =
            super::lower_function(&corpus_llbc(), name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let calls = corpus_call_leaves(name);
        assert!(
            !calls
                .iter()
                .any(|(leaf, _)| leaf == "as_ptr" || leaf == "as_mut_ptr"),
            "{name}: {calls:?}"
        );
        let input = graph
            .block(graph.startblock)
            .inputargs
            .first()
            .expect("pair pointer input");
        let returned = graph
            .blocks
            .iter()
            .flat_map(|b| b.exits.iter())
            .find(|link| link.target == graph.returnblock)
            .and_then(|link| link.args.first())
            .expect("return link");
        assert!(
            matches!(returned, LinkArg::Value(value) if value == input),
            "{name} must return its pointer word; graph: {graph:#?}"
        );
    }
}

/// The Sized gate must reject an UNSIZED nominal ADT: a DST newtype
/// (`rustpython_wtf8::Wtf8 { bytes: [u8] }`) IS a nominal ADT with a def-id,
/// but `&Wtf8` is a two-word fat pointer. The layout for such a type carries
/// `size: null`, so the layout-size predicate the recognizer composes must
/// see `None` (reject), while a sized `W_*` pointee carries a concrete size
/// (accept). Loads the real LLBC (needs Charon-resolved layouts), ignored.
#[test]
#[ignore]
fn sized_gate_rejects_unsized_wtf8_accepts_sized_wstruct() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let target = std::env::var("TARGET").unwrap_or_default();

    // Resolve a type's layout size by its flattened name (as the recognizer
    // does after `adt_node_def_id`): find its def-id, read layout.size.
    let size_of = |suffix: &str| -> Option<Option<u64>> {
        llbc.iter_type_decls()
            .find(|td| td.item_meta.name_path().ends_with(suffix))
            .map(|td| td.layout_for_target(&llbc, &target).and_then(|l| l.size))
    };

    // Wtf8: a DST newtype → layout size is `None` → the gate rejects it.
    assert_eq!(
        size_of("rustpython_wtf8::Wtf8"),
        Some(None),
        "Wtf8 is an unsized DST newtype; its layout size must be None so the \
             shared-ref niche gate rejects Option<&Wtf8>"
    );
    // FrameDebugData: a sized struct → concrete layout size → gate accepts.
    assert!(
        matches!(size_of("pyframe::FrameDebugData"), Some(Some(_))),
        "FrameDebugData is Sized; its layout size must be Some so the \
             shared-ref niche win keeps folding"
    );
}

/// End-to-end: `str_slice_args -> Result<Option<&Wtf8>, PyError>` (matched by
/// `startswith`/`endswith`). Because `Wtf8` is an unsized DST, its
/// `Option<&Wtf8>` must NOT niche-fold — no pointer identity-null discriminant
/// against a `null_mut()` may appear, or the fat pointer's length word is
/// lost. The residual/aggregate Option handling must survive. Loads the real
/// LLBC, ignored.
#[test]
#[ignore]
fn str_slice_args_option_wtf8_not_niche_folded() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "str_slice_args").expect("lower str_slice_args");
    // The niche discriminant fold emits an identity test with `null_mut()`; a declined
    // Option keeps a `null_mut` builtin ONLY when genuinely niche. Assert no
    // `null_mut` call is synthesised for this graph's Option<&Wtf8> — the fat
    // ref must not be treated as a one-word niche.
    let null_mut_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("null_mut")
            )
        })
        .count();
    assert_eq!(
        null_mut_calls, 0,
        "Option<&Wtf8> is a fat-ref DST option and must not niche-fold to a \
             pointer-null test (no null_mut discriminant)"
    );
}

/// PyPy carries `call_function` arguments as one dynamic RPython list.
/// Keep that producer shape: a fixed array followed by a prefix copy is
/// unnecessary even though general RangeTo now lowers through getslice.
#[test]
#[ignore]
fn get_and_call_function_has_no_residual_array_index() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "get_and_call_function").expect("lower get_and_call_function");
    let calls_path = |want: &[&str]| -> usize {
        let want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &want
                )
            })
            .count()
    };
    assert_eq!(
        calls_path(&["core", "array", "<Impl>", "index"]),
        0,
        "PyPy-shaped dynamic arguments must not leave a fixed-array index"
    );
    assert_eq!(
        calls_path(&["__getslice_rangeto"]),
        0,
        "dynamic arguments are already a list; no prefix-slice marker is needed"
    );
}

/// The generator throw producer validates its preserved positional count
/// as 1..=3.  Delegation must keep those fixed call shapes visible rather
/// than routing them through `<[T; 3]>::index(&args, ..argc)`.
#[test]
#[ignore]
fn throw_yield_from_has_no_residual_array_index() {
    use crate::model::OpKind;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "throw_yield_from").expect("lower throw_yield_from");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["core", "array", "<Impl>", "index"]
            )
        })
        .count();
    assert_eq!(
        residual, 0,
        "fixed throw arities must not leave an array index"
    );
}

#[test]
#[ignore]
fn clear_references_niche_bool_then_has_no_option_aggregate() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "clear_references").expect("lower clear_references");
    let option_aggregate_ops = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } => name == "Option<*mut PyObject>",
            OpKind::FieldWrite { field, .. } => {
                field.name == "__discriminant"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|owner| owner.contains("Option<*mut PyObject>"))
            }
            _ => false,
        })
        .count();
    assert_eq!(
        option_aggregate_ops, 0,
        "bool::then producing Option<PyObjectRef> must use RPython's nullable-pointer \
             representation (Some(ptr)=ptr, None=null), not an Option aggregate"
    );
    let is_instance_narrow = |op: &crate::model::SpaceOperation| {
        crate::model::cast_instance_root(&op.kind) == Some("PyObject")
    };
    let join = graph.blocks.iter().find_map(|block| {
        let has_call_once = block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "call_once"
            )
        });
        (has_call_once && block.operations.iter().any(is_instance_narrow))
            .then(|| block.exits.first().map(|link| link.target))
            .flatten()
    });
    assert!(
        join.is_some(),
        "the niche Some payload must retain PyObject/W_Root class identity across call_once"
    );
    let narrowed_none = graph.blocks.iter().any(|block| {
        block.exits.first().map(|link| link.target) == join
            && block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["core", "ptr", "null_mut"]
                )
            })
            && block.operations.iter().any(is_instance_narrow)
    });
    assert!(
        narrowed_none,
        "the niche None arm must be the same nullable PyObject/W_Root repr as the Some arm"
    );
}

#[test]
#[ignore]
fn struct_ptr_write_lowers_to_per_field_initialization() {
    use crate::model::{CallTarget, OpKind};
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/llbc/");
    let llbcs = [
        Llbc::load(format!("{root}pyre-object.ullbc")).expect("load pyre-object LLBC"),
        Llbc::load(format!("{root}pyre-interpreter.ullbc")).expect("load pyre-interpreter LLBC"),
        Llbc::load(format!("{root}pyre-jit.ullbc")).expect("load pyre-jit LLBC"),
    ];
    let names = [
        "w_cell_new",
        "alloc_instance_object",
        "w_str_from_wtf8_managed",
    ];
    let program = super::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &llbcs,
        crate::HostStaticAddrs::default(),
        &[],
        &names,
    )
    .expect("build production-shaped semantic program");
    for (name, owner, expected_fields) in [
        (
            "w_cell_new",
            "Cell",
            &["ob_type", "w_class", "contents", "family"][..],
        ),
        (
            "alloc_instance_object",
            "W_ObjectObject",
            &["ob_type", "w_class", "map", "storage"][..],
        ),
        (
            "w_str_from_wtf8_managed",
            "W_UnicodeObject",
            &[
                "ob_type",
                "w_class",
                "value",
                "byte_len",
                "len",
                "w_slots",
                "index_storage",
                "hash",
            ][..],
        ),
    ] {
        let graph = &program
            .functions
            .iter()
            .find(|function| function.name == name)
            .unwrap_or_else(|| panic!("missing semantic function {name}"))
            .graph();
        assert!(
                !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &["core", "ptr", "write"])
                }),
                "{name}: generic core::ptr::write must not survive as a shared callee"
            );
        // A graph can cast more than one raw pointer to the owner (the
        // other arm's fallback allocation); the one the write targeted
        // carries the stores.
        let destinations: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match (&op.result, &op.kind) {
                (Some(result), OpKind::Call { .. })
                    if crate::model::cast_instance_root(&op.kind) == Some(owner) =>
                {
                    Some(result)
                }
                _ => None,
            })
            .collect();
        assert!(
            !destinations.is_empty(),
            "{name}: typed raw allocation destination"
        );
        let fields: Vec<Vec<_>> = destinations
            .iter()
            .map(|destination| {
                graph
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .filter_map(|op| match &op.kind {
                        OpKind::FieldWrite { base, field, .. } if base == *destination => {
                            Some(field.name.as_str())
                        }
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        assert!(
            fields.iter().any(|fields| fields == expected_fields),
            "{name}: source field order, got {fields:?}"
        );
        assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. }
                        if field.name == "ob_header" || field.name == "ob"
                )
            }),
            "{name}: inlined header field must be stored as its leaves"
        );
    }
}

#[test]
#[ignore]
fn atomic_from_ptr_lowers_to_an_identity_view() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "hash_slot").expect("lower hash_slot");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("from_ptr"))
            }),
            "AtomicI64::from_ptr must not survive as a graph-less core call"
        );
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "hash")
        }),
        "the identity view must retain its underlying unicode hash-field read"
    );
}

#[test]
#[ignore]
fn atomic_new_lowers_to_its_inner_value() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "w_specialised_tuple_ii_new")
        .expect("lower w_specialised_tuple_ii_new");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("new")
                        && segments.iter().any(|s| s.starts_with("Atomic")))
            }),
            "AtomicI64::new must not survive as a graph-less core call"
        );
}

#[test]
#[ignore]
fn unicode_hash_atomic_store_is_the_exact_residual_choke_point() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let residuals = super::dont_look_inside_set_of(&llbc);
    assert!(
        residuals.contains("unicodeobject::w_str_set_hash"),
        "the foreign AtomicI64::store must stay behind its pyre-owned wrapper"
    );
    assert!(
        !residuals.contains("unicodeobject::w_str_get_hash"),
        "the ordinary scalar memo read remains visible to the generated JIT"
    );
}

#[test]
#[ignore]
fn deque_compare_scalar_slice_get_lowers_to_a_guarded_item_read() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "deque_compare").expect("lower deque_compare");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["core".to_string(), "slice".to_string(), "<Impl>".to_string(), "get".to_string()])
            }),
            "scalar slice get must not survive as an opaque core call"
        );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(op.kind, OpKind::ArrayRead { .. })),
        "the successful get arm must read the guarded element"
    );
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor {
                        name,
                        owner_path,
                        ..
                    },
                    ..
                } if (name == "Some" || name == "None")
                    && owner_path.iter().any(|part| part.contains("Option<*mut PyObject>"))
            ) || matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. }
                    if field
                        .owner_root
                        .as_deref()
                        .is_some_and(|owner| owner.starts_with("core::option::Option<"))
            )
        }),
        "Option<&PyObjectRef> must stay a nullable PyObject pointer, not an Option aggregate"
    );
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["core", "ptr", "null_mut"]
            )
        }),
        "the out-of-bounds arm must materialize the nullable pointer's null"
    );
}

/// Slice `get` is not object-pointer-specific, and a literal index has no
/// Place type.  This production helper exercises both the generalized
/// Option payload recognizer and its generic-type fallback for `get(1)`.
#[test]
#[ignore]
fn slice_get_constant_index_lowers_to_a_guarded_item_read() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-module.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let name = "pyre_module::module::unicodedata::interp_ucd::char_and_default";
    let graph =
        super::lower_function(&llbc, name).unwrap_or_else(|err| panic!("lower {name}: {err:?}"));
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["core", "slice", "<Impl>", "get"])
            }),
            "{name}: scalar slice get must not survive as an opaque core call"
        );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(op.kind, OpKind::ArrayRead { .. })),
        "{name}: successful get arm must read the guarded item"
    );
}

/// `args.get(1..).unwrap_or(&[])` is Rust's checked spelling of PyPy's
/// ordinary `args[1:]`.  Keep `get`'s Option semantics in a `start <= len`
/// diamond and materialize the successful payload as the deferred
/// `getslice(start, None)` marker.
#[test]
#[ignore]
fn slice_get_rangefrom_lowers_to_a_guarded_getslice() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-module.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let name = "pyre_module::module::unicodedata::interp_ucd::ucd_method_args";
    let graph =
        super::lower_function(&llbc, name).unwrap_or_else(|err| panic!("lower {name}: {err:?}"));
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["core", "slice", "<Impl>", "get"])
            }),
            "{name}: RangeFrom slice get must not survive as an opaque core call"
        );
    assert!(
            graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["__getslice_rangefrom"])
            }),
            "{name}: successful get arm must carry the deferred getslice"
        );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "le")),
        "{name}: RangeFrom get must guard start <= len"
    );
}

#[test]
#[ignore]
fn native_load_fast_preserves_unit_operand_rtype() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    ))
    .expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::pyopcode::execute_load_fast")
        .expect("lower native load_fast");
    let mut checked = 0;
    for block in &graph.blocks {
        for arg in block.exits.iter().flat_map(|link| &link.args) {
            let crate::model::LinkArg::Value(var) = arg else {
                continue;
            };
            if crate::model::FunctionGraph::concretetype_of(var) != crate::model::ConcreteType::Void
            {
                continue;
            }
            checked += 1;
            assert!(
                block.inputargs.contains(var)
                    || block.operations.iter().any(|op| {
                        op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstNone)
                    }),
                "unit edge operand {var:?} has no definition in {:?}",
                block.id
            );
        }
    }
    assert!(
        checked > 0,
        "fixture must exercise actual unit-valued edges"
    );
}

use serde_json::{Value, json};

fn str_builder_span() -> Value {
    json!({"data": {"file_id": 0,
            "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}})
}

fn str_builder_generics() -> Value {
    json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []})
}

fn ident_seg(name: &str) -> Value {
    json!({"Ident": [name, 0]})
}

fn str_builder_meta(path: Vec<Value>, is_local: bool) -> Value {
    json!({
        "name": path,
        "span": str_builder_span(),
        "source_text": null,
        "is_local": is_local,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
    })
}

fn wtf8_adt() -> Value {
    json!({"Adt": {"id": 0, "generics": str_builder_generics()}})
}

fn outcome_adt() -> Value {
    json!({"Adt": {"id": 1, "generics": str_builder_generics()}})
}

fn ref_wtf8(kind: &str) -> Value {
    json!({"Ref": [{"Erased": null}, wtf8_adt(), kind]})
}

pub(in crate::front::mir) fn unit_ty() -> Value {
    json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": str_builder_generics()}})
}

pub(in crate::front::mir) fn word_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U64"}}})
}

pub(in crate::front::mir) fn place_of(id: u64, ty: &Value) -> Value {
    json!({"kind": {"Local": id}, "ty": ty})
}

pub(in crate::front::mir) fn local_of(index: u64, ty: &Value) -> Value {
    json!({"index": index, "name": null, "span": str_builder_span(), "ty": ty})
}

pub(in crate::front::mir) fn assign_stmt(place: Value, rvalue: Value) -> Value {
    json!({"span": str_builder_span(), "kind": {"Assign": [place, rvalue]}})
}

pub(in crate::front::mir) fn ref_rvalue(place: Value, kind: &str) -> Value {
    json!({"Ref": {"place": place, "kind": kind, "ptr_metadata": null}})
}

fn use_move(place: Value) -> Value {
    json!({"Use": [{"Move": place}, "No"]})
}

fn use_copy(place: Value) -> Value {
    json!({"Use": [{"Copy": place}, "No"]})
}

pub(in crate::front::mir) fn move_op(place: Value) -> Value {
    json!({"Move": place})
}

fn copy_op(place: Value) -> Value {
    json!({"Copy": place})
}

pub(in crate::front::mir) fn bb(statements: Vec<Value>, term: Value) -> Value {
    json!({
        "statements": statements,
        "terminator": {"span": str_builder_span(), "kind": term}
    })
}

pub(in crate::front::mir) fn call_term(
    fun: u64,
    args: Vec<Value>,
    dest: Value,
    target: u64,
    unwind: u64,
) -> Value {
    json!({"Call": {
        "call": {
            "func": {"Regular": {"kind": {"Fun": fun}, "generics": str_builder_generics()}},
            "args": args,
            "dest": dest
        },
        "target": target,
        "on_unwind": unwind
    }})
}

pub(in crate::front::mir) fn field_place(local: u64, base_ty: &Value, field_ty: &Value) -> Value {
    json!({
        "kind": {"Projection": [place_of(local, base_ty), {"Field": [0, 0]}]},
        "ty": field_ty
    })
}

fn wtf8buf_decl() -> Value {
    json!({
        "def_id": 0,
        "item_meta": type_decl_meta(&["rustpython_wtf8", "Wtf8Buf"]),
        "kind": "Opaque"
    })
}

fn outcome_decl() -> Value {
    json!({
        "def_id": 1,
        "item_meta": type_decl_meta(&["fixture", "Outcome"]),
        "kind": {"Enum": [
            {
                "name": "Continue",
                "fields": [{
                    "name": null,
                    "ty": wtf8_adt(),
                    "attr_info": {
                        "attributes": [], "inline": null, "rename": null, "public": false
                    }
                }],
                "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}
            },
            {
                "name": "Break",
                "fields": [],
                "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}
            }
        ]}
    })
}

pub(in crate::front::mir) fn opaque_decl(
    def_id: u64,
    path: &[&str],
    inputs: Vec<Value>,
    output: Value,
) -> Value {
    json!({
        "def_id": def_id,
        "item_meta": str_builder_meta(
            path.iter().copied().map(ident_seg).collect(),
            false,
        ),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": "Opaque"
    })
}

fn wtf8buf_impl_seg() -> Value {
    json!({"Impl": {"Ty": {
        "params": {
            "regions": [], "types": [], "const_generics": [], "trait_clauses": [],
            "regions_outlive": [], "types_outlive": [], "trait_type_constraints": []
        },
        "skip_binder": {"Value": [0, wtf8_adt()]},
        "kind": "InherentImplBlock"
    }}})
}

fn push_str_decl() -> Value {
    json!({
        "def_id": 2,
        "item_meta": str_builder_meta(vec![
            ident_seg("rustpython_wtf8"),
            ident_seg("Wtf8Buf"),
            wtf8buf_impl_seg(),
            ident_seg("push_str"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_wtf8("Mut"), word_ty()],
            "output": unit_ty()
        },
        "body": "Opaque"
    })
}

fn wtf8buf_new_decl() -> Value {
    json!({
        "def_id": 1,
        "item_meta": str_builder_meta(vec![
            ident_seg("rustpython_wtf8"),
            ident_seg("Wtf8Buf"),
            wtf8buf_impl_seg(),
            ident_seg("new"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": wtf8_adt()
        },
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn char_ty() -> Value {
    json!({"Scalar": "Char"})
}

fn const_char_op(ch: &str) -> Value {
    json!({"Const": [{"Char": ch}, {"Scalar": "Char"}]})
}

fn push_char_decl() -> Value {
    json!({
        "def_id": 2,
        "item_meta": str_builder_meta(vec![
            ident_seg("rustpython_wtf8"),
            ident_seg("Wtf8Buf"),
            wtf8buf_impl_seg(),
            ident_seg("push"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_wtf8("Mut"), char_ty()],
            "output": unit_ty()
        },
        "body": "Opaque"
    })
}

fn mem_take_decl() -> Value {
    json!({
        "def_id": 3,
        "item_meta": str_builder_meta(vec![
            ident_seg("core"),
            ident_seg("mem"),
            ident_seg("take"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_wtf8("Mut")],
            "output": wtf8_adt()
        },
        "body": "Opaque"
    })
}

fn as_bytes_decl() -> Value {
    json!({
        "def_id": 3,
        "item_meta": str_builder_meta(vec![
            ident_seg("rustpython_wtf8"),
            ident_seg("Wtf8Buf"),
            wtf8buf_impl_seg(),
            ident_seg("as_bytes"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_wtf8("Shared")],
            "output": bytes_slice_ty()
        },
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn string_adt() -> Value {
    json!({"Adt": {"id": 0, "builtin": null, "generics": str_builder_generics()}})
}

pub(in crate::front::mir) fn string_decl() -> Value {
    json!({
        "def_id": 0,
        "item_meta": type_decl_meta(&["alloc", "string", "String"]),
        "kind": {"Struct": [
            {"name": "ptr", "ty": window_usize(), "attr_info": null},
            {"name": "len", "ty": window_usize(), "attr_info": null},
            {"name": "cap", "ty": window_usize(), "attr_info": null}
        ]}
    })
}

fn string_impl_seg() -> Value {
    json!({"Impl": {"Ty": {
        "params": {
            "regions": [], "types": [], "const_generics": [], "trait_clauses": [],
            "regions_outlive": [], "types_outlive": [], "trait_type_constraints": []
        },
        "skip_binder": {"Value": [0, string_adt()]},
        "kind": "InherentImplBlock"
    }}})
}

pub(in crate::front::mir) fn ref_string(kind: &str) -> Value {
    json!({"Ref": [{"Erased": null}, string_adt(), kind]})
}

fn string_new_decl() -> Value {
    json!({
        "def_id": 1,
        "item_meta": str_builder_meta(vec![
            ident_seg("alloc"),
            ident_seg("string"),
            ident_seg("String"),
            string_impl_seg(),
            ident_seg("new"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": string_adt()
        },
        "body": "Opaque"
    })
}

fn string_push_str_decl() -> Value {
    json!({
        "def_id": 2,
        "item_meta": str_builder_meta(vec![
            ident_seg("alloc"),
            ident_seg("string"),
            ident_seg("String"),
            string_impl_seg(),
            ident_seg("push_str"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_string("Mut"), word_ty()],
            "output": unit_ty()
        },
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn string_as_bytes_decl() -> Value {
    json!({
        "def_id": 3,
        "item_meta": str_builder_meta(vec![
            ident_seg("alloc"),
            ident_seg("string"),
            ident_seg("String"),
            string_impl_seg(),
            ident_seg("as_bytes"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_string("Shared")],
            "output": window_shared_slice()
        },
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn fmt_arguments_adt() -> Value {
    json!({"Adt": {"id": 1, "builtin": null, "generics": str_builder_generics()}})
}

pub(in crate::front::mir) fn fmt_arguments_decl() -> Value {
    json!({
        "def_id": 1,
        "item_meta": type_decl_meta(&["core", "fmt", "Arguments"]),
        "kind": "Opaque"
    })
}

pub(in crate::front::mir) fn fmt_format_decl(def_id: u64) -> Value {
    json!({
        "def_id": def_id,
        "item_meta": str_builder_meta(vec![
            ident_seg("alloc"),
            ident_seg("fmt"),
            ident_seg("format"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [fmt_arguments_adt()],
            "output": string_adt()
        },
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn hint_must_use_decl(def_id: u64) -> Value {
    json!({
        "def_id": def_id,
        "item_meta": str_builder_meta(vec![
            ident_seg("core"),
            ident_seg("hint"),
            ident_seg("must_use"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [string_adt()],
            "output": string_adt()
        },
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn deref_place(local: u64, base_ty: &Value, pointee_ty: &Value) -> Value {
    json!({
        "kind": {"Projection": [place_of(local, base_ty), "Deref"]},
        "ty": pointee_ty
    })
}

fn format_as_bytes_window_name() -> &'static str {
    "format_string_as_bytes_window_param"
}

pub(in crate::front::mir) fn format_as_bytes_window_llbc() -> majit_charon_reader::Llbc {
    let string = string_adt();
    let args_ty = fmt_arguments_adt();
    let unit = unit_ty();
    let shared = ref_string("Shared");
    let window = window_shared_slice();
    let slice = window_slice();
    let name = format_as_bytes_window_name();
    llbc_with_types(
        "fixture",
        vec![string_decl(), fmt_arguments_decl()],
        vec![
            subject_decl(
                name,
                1,
                vec![args_ty.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &args_ty),
                    local_of(2, &string),
                    local_of(3, &string),
                    local_of(4, &shared),
                    local_of(5, &window),
                    local_of(6, &window),
                ],
                vec![
                    bb(
                        vec![],
                        call_term(
                            1,
                            vec![move_op(place_of(1, &args_ty))],
                            place_of(2, &string),
                            1,
                            5,
                        ),
                    ),
                    bb(
                        vec![],
                        call_term(
                            2,
                            vec![move_op(place_of(2, &string))],
                            place_of(3, &string),
                            2,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(4, &shared),
                            ref_rvalue(place_of(3, &string), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(4, &shared))],
                            place_of(5, &window),
                            3,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(6, &window),
                            ref_rvalue(deref_place(5, &window, &slice), "Shared"),
                        )],
                        call_term(
                            4,
                            vec![move_op(place_of(6, &window))],
                            place_of(0, &unit),
                            4,
                            5,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            fmt_format_decl(1),
            hint_must_use_decl(2),
            string_as_bytes_decl(),
            opaque_decl(4, &["fixture", "g"], vec![window], unit),
        ],
    )
}

fn string_as_ref_bytes_decl() -> Value {
    json!({
        "def_id": 3,
        "item_meta": str_builder_meta(vec![
            ident_seg("alloc"),
            ident_seg("string"),
            ident_seg("String"),
            string_impl_seg(),
            ident_seg("as_ref"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_string("Shared")],
            "output": window_shared_slice()
        },
        "body": "Opaque"
    })
}

fn deref_decl() -> Value {
    json!({
        "def_id": 4,
        "item_meta": str_builder_meta(vec![
            ident_seg("rustpython_wtf8"),
            ident_seg("Wtf8Buf"),
            wtf8buf_impl_seg(),
            ident_seg("deref"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_wtf8("Shared")],
            "output": ref_wtf8("Shared")
        },
        "body": "Opaque"
    })
}

fn bytes_slice_ty() -> Value {
    json!({"Ref": [{"Erased": null}, {"Adt": {
            "id": 0,
            "builtin": "Slice",
            "generics": {
                "regions": [],
                "types": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}],
                "const_generics": [],
                "trait_refs": []
            }
        }}, "Shared"]})
}

pub(in crate::front::mir) fn bool_ty() -> Value {
    json!({"Scalar": "Bool"})
}

pub(in crate::front::mir) fn subject_decl(
    name: &str,
    arg_count: u64,
    inputs: Vec<Value>,
    output: Value,
    locals: Vec<Value>,
    blocks: Vec<Value>,
) -> Value {
    json!({
        "def_id": 0,
        "item_meta": str_builder_meta(vec![ident_seg("fixture"), ident_seg(name)], true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": {"Unstructured": {
            "span": str_builder_span(),
            "locals": {"arg_count": arg_count, "locals": locals},
            "body": blocks
        }}
    })
}

fn lower_owned_concat(
    name: &str,
    types: Vec<Value>,
    funs: Vec<Value>,
) -> crate::model::FunctionGraph {
    let llbc = llbc_with_types("fixture", types, funs);
    super::lower_function(&llbc, name).unwrap_or_else(|err| panic!("{name} failed to lower: {err}"))
}

fn has_str_add(graph: &FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
}

fn has_path_leaf(graph: &FunctionGraph, leaf: &str) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| {
            matches!(&op.kind, OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|segment| segment == leaf))
        })
}

fn leftover_push_or_take(graph: &FunctionGraph) -> Vec<Vec<String>> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                let leaf = segments.last().map(String::as_str);
                let is_string_push = leaf == Some("push")
                    && segments.iter().any(|s| s == "String" || s == "Wtf8Buf");
                let is_mem_take = leaf == Some("take")
                    && segments.windows(2).any(|w| w[0] == "mem" && w[1] == "take");
                (is_string_push || is_mem_take).then(|| segments.clone())
            }
            _ => None,
        })
        .collect()
}

fn assert_builder_identity_lowering(graph: &FunctionGraph, name: &str) {
    assert!(
        !has_path_leaf(graph, "push_str"),
        "{name}: push_str must not stay residual: {graph:#?}"
    );
    assert!(
        !has_str_add(graph),
        "{name}: identity-view builder must use ll_build, not ll_strconcat: {graph:#?}"
    );
    for marker in [
        "__majit_stringbuilder_new",
        "__majit_stringbuilder_append",
        "__majit_stringbuilder_build",
    ] {
        assert!(
            has_path_leaf(graph, marker),
            "{name}: missing builder marker {marker}: {graph:#?}"
        );
    }
}

/// `Wtf8Buf::new(); buf.push('%'); mem::take(&mut buf)` — W4 + W5:
/// the take borrow is clean, `push(char)` becomes a one-char str, and
/// neither `push` nor `take` remains a FunctionPath.
#[test]
fn owned_ctor_push_char_take_lowers() {
    let wtf8 = wtf8_adt();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let graph = lower_owned_concat(
        "ctor_push_char_take",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "ctor_push_char_take",
                0,
                vec![],
                wtf8.clone(),
                vec![
                    local_of(0, &wtf8),
                    local_of(1, &wtf8),
                    local_of(2, &borrow),
                    local_of(3, &unit),
                    local_of(4, &borrow),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(1, &wtf8), 1, 4)),
                    bb(
                        vec![assign_stmt(
                            place_of(2, &borrow),
                            ref_rvalue(place_of(1, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(2, &borrow)), const_char_op("%")],
                            place_of(3, &unit),
                            2,
                            4,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(4, &borrow),
                            ref_rvalue(place_of(1, &wtf8), "Mut"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(4, &borrow))],
                            place_of(0, &wtf8),
                            3,
                            4,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            wtf8buf_new_decl(),
            push_char_decl(),
            mem_take_decl(),
        ],
    );
    let leftover = leftover_push_or_take(&graph);
    assert!(
        leftover.is_empty(),
        "push(char)/take must not stay residual: {leftover:#?} {graph:#?}"
    );
    let has_str_const = graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.first().map(String::as_str) == Some("__str_const"))
            })
        });
    assert!(
        has_path_leaf(&graph, crate::runtime_names::shims::UNICHAR_AS_UTF8) || has_str_const,
        "push(char) must become unichr_as_utf8 or a one-char str const: {graph:#?}"
    );
}

/// `Wtf8Buf::new(); buf.push(ch); return buf` — a variable code point
/// piece is `__majit_unichr_as_utf8`, not a leftover SomeInteger.
#[test]
fn owned_ctor_push_char_local_lowers() {
    let wtf8 = wtf8_adt();
    let ch = char_ty();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let graph = lower_owned_concat(
        "ctor_push_char_local",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "ctor_push_char_local",
                1,
                vec![ch.clone()],
                wtf8.clone(),
                vec![
                    local_of(0, &wtf8),
                    local_of(1, &ch),
                    local_of(2, &wtf8),
                    local_of(3, &borrow),
                    local_of(4, &unit),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &wtf8), 1, 4)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &borrow),
                            ref_rvalue(place_of(2, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &borrow)), copy_op(place_of(1, &ch))],
                            place_of(4, &unit),
                            2,
                            4,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(0, &wtf8),
                            use_move(place_of(2, &wtf8)),
                        )],
                        json!("Return"),
                    ),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            wtf8buf_new_decl(),
            push_char_decl(),
        ],
    );
    let leftover = leftover_push_or_take(&graph);
    assert!(
        leftover.is_empty(),
        "push(char) must not stay residual: {leftover:#?} {graph:#?}"
    );
    assert!(
        has_path_leaf(&graph, crate::runtime_names::shims::UNICHAR_AS_UTF8),
        "a variable code point must go through unichr_as_utf8: {graph:#?}"
    );
}

/// `Wtf8Buf::new(); buf.push_str(s); buf.as_bytes()` — a Shared identity
/// view is `ll_build` (`AbstractStringBuilderRepr.rtype_method_build`),
/// not an escape that residualizes `push_str`.
#[test]
fn owned_ctor_push_str_as_bytes_lowers_to_stringbuilder() {
    let wtf8 = wtf8_adt();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let shared = ref_wtf8("Shared");
    let bytes = bytes_slice_ty();
    let graph = lower_owned_concat(
        "ctor_push_as_bytes",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "ctor_push_as_bytes",
                1,
                vec![word.clone()],
                bytes.clone(),
                vec![
                    local_of(0, &bytes),
                    local_of(1, &word),
                    local_of(2, &wtf8),
                    local_of(3, &borrow),
                    local_of(4, &unit),
                    local_of(5, &shared),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &wtf8), 1, 4)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &borrow),
                            ref_rvalue(place_of(2, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &borrow)), copy_op(place_of(1, &word))],
                            place_of(4, &unit),
                            2,
                            4,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(5, &shared),
                            ref_rvalue(place_of(2, &wtf8), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(5, &shared))],
                            place_of(0, &bytes),
                            3,
                            4,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            wtf8buf_new_decl(),
            push_str_decl(),
            as_bytes_decl(),
        ],
    );
    assert_builder_identity_lowering(&graph, "ctor_push_as_bytes");
}

/// `String::new(); s.push_str(x); g(s.as_bytes())` with `g(&[u8])`.
/// `ll_build` (`AbstractStringBuilderRepr.rtype_method_build`) makes
/// the receiver a `Ptr(STR)`, so the window dest is a W1 STR view of
/// the built string rather than a residual `as_bytes` of the builder.
#[test]
fn builder_as_bytes_into_window_param_builds_the_str() {
    let string = string_adt();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_string("Mut");
    let shared = ref_string("Shared");
    let window = window_shared_slice();
    let graph = lower_owned_concat(
        "builder_as_bytes_window_param",
        vec![string_decl()],
        vec![
            subject_decl(
                "builder_as_bytes_window_param",
                1,
                vec![word.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &word),
                    local_of(2, &string),
                    local_of(3, &borrow),
                    local_of(4, &unit),
                    local_of(5, &shared),
                    local_of(6, &window),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &string), 1, 5)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &borrow),
                            ref_rvalue(place_of(2, &string), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &borrow)), copy_op(place_of(1, &word))],
                            place_of(4, &unit),
                            2,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(5, &shared),
                            ref_rvalue(place_of(2, &string), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(5, &shared))],
                            place_of(6, &window),
                            3,
                            5,
                        ),
                    ),
                    bb(
                        vec![],
                        call_term(
                            4,
                            vec![move_op(place_of(6, &window))],
                            place_of(0, &unit),
                            4,
                            5,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            string_new_decl(),
            string_push_str_decl(),
            string_as_bytes_decl(),
            opaque_decl(4, &["fixture", "g"], vec![window.clone()], unit),
        ],
    );
    assert!(
        has_path_leaf(&graph, crate::runtime_names::shims::STRINGBUILDER_BUILD),
        "as_bytes of a builder into a window param must ll_build: {graph:#?}"
    );
    assert!(
        !has_path_leaf(&graph, "as_bytes"),
        "as_bytes must not stay residual: {graph:#?}"
    );
    let new_ids: std::collections::HashSet<u64> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::STRINGBUILDER_NEW) =>
            {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect();
    let fed_by_new = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => {
                let leaf = segments.last().map(String::as_str);
                let watches_g = leaf == Some("g");
                let watches_window = leaf
                    == Some(crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING)
                    || leaf == Some("__strlen");
                (watches_g || watches_window) && args.iter().any(|arg| new_ids.contains(&arg.id()))
            }
            _ => false,
        });
    assert!(
        !fed_by_new,
        "g (or the window conversion feeding it) must not take the StringBuilder: {graph:#?}"
    );
}

fn format_call_result_ids(graph: &FunctionGraph) -> std::collections::HashSet<u64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if super::fmt_path_ends_with(segments, &["fmt", "format"]) => {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect()
}

fn g_or_str_window_fed_by(graph: &FunctionGraph, ids: &std::collections::HashSet<u64>) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => {
                let leaf = segments.last().map(String::as_str);
                let watches_g = leaf == Some("g");
                let watches_window = leaf
                    == Some(crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING)
                    || leaf == Some("__strlen");
                (watches_g || watches_window) && args.iter().any(|arg| ids.contains(&arg.id()))
            }
            _ => false,
        })
}

/// Format dests plus `__cast_instance_intrinsic` aliases and SSA
/// renames across block links (`emit_str_to_window` reads the header
/// in a successor of the format block).
fn ids_flowing_from(
    graph: &FunctionGraph,
    seeds: &std::collections::HashSet<u64>,
) -> std::collections::HashSet<u64> {
    let mut ids = seeds.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for block in &graph.blocks {
            for op in &block.operations {
                let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if segments.last().map(String::as_str) != Some("__cast_instance_intrinsic") {
                    continue;
                }
                let Some(src) = args.first().and_then(|a| a.as_variable()) else {
                    continue;
                };
                if ids.contains(&src.id())
                    && let Some(result) = &op.result
                    && ids.insert(result.id())
                {
                    changed = true;
                }
            }
            for link in &block.exits {
                let target = graph.block(link.target);
                for (i, arg) in link.args.iter().enumerate() {
                    let Some(src) = arg.as_variable() else {
                        continue;
                    };
                    if ids.contains(&src.id())
                        && let Some(inp) = target.inputargs.get(i)
                        && ids.insert(inp.id())
                    {
                        changed = true;
                    }
                }
            }
        }
    }
    ids
}

/// `format(args)` → `must_use` → `&` → `as_bytes` → `g(&[u8])`.
/// The format dest is a W1 STR (`rstr.py` `do_stringformat`).
#[test]
fn format_string_as_bytes_into_window_param_binds_the_str() {
    let name = format_as_bytes_window_name();
    let llbc = format_as_bytes_window_llbc();
    let graph = super::lower_function(&llbc, name)
        .unwrap_or_else(|err| panic!("{name} failed to lower: {err}"));
    assert!(
        !has_path_leaf(&graph, "as_bytes"),
        "as_bytes of a format String must not stay residual: {graph:#?}"
    );
    assert!(
        !has_path_leaf(&graph, "ll_slice_len_slot_new")
            && !has_path_leaf(&graph, "ll_slice_len_slot_take"),
        "format as_bytes must not use SLICE_LEN_SLOT_NEW/TAKE: {graph:#?}"
    );
    let format_ids = format_call_result_ids(&graph);
    assert!(
        !format_ids.is_empty(),
        "fmt::format must remain in the graph: {graph:#?}"
    );
    let format_ids = ids_flowing_from(&graph, &format_ids);
    assert!(
        g_or_str_window_fed_by(&graph, &format_ids),
        "g (or the STR→window conversion feeding it) must take the format result: {graph:#?}"
    );
}

/// The `String` local is also defined by `String::new` on another
/// `If` arm, so it is not a format_str local and `as_bytes` stays.
#[test]
fn mixed_format_and_heap_string_as_bytes_keeps_call() {
    let string = string_adt();
    let args_ty = fmt_arguments_adt();
    let unit = unit_ty();
    let flag = bool_ty();
    let shared = ref_string("Shared");
    let window = window_shared_slice();
    let slice = window_slice();
    let name = "mixed_format_heap_string_as_bytes_window_param";
    let graph = lower_owned_concat(
        name,
        vec![string_decl(), fmt_arguments_decl()],
        vec![
            subject_decl(
                name,
                2,
                vec![args_ty.clone(), flag.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &args_ty),
                    local_of(2, &flag),
                    local_of(3, &string),
                    local_of(4, &string),
                    local_of(5, &shared),
                    local_of(6, &window),
                    local_of(7, &window),
                ],
                vec![
                    bb(
                        vec![],
                        json!({"Switch": {
                            "discr": {"Copy": place_of(2, &flag)},
                            "targets": {"If": [1, 2]}
                        }}),
                    ),
                    bb(
                        vec![],
                        call_term(
                            5,
                            vec![move_op(place_of(1, &args_ty))],
                            place_of(3, &string),
                            3,
                            7,
                        ),
                    ),
                    bb(vec![], call_term(1, vec![], place_of(3, &string), 3, 7)),
                    bb(
                        vec![],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &string))],
                            place_of(4, &string),
                            4,
                            7,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(5, &shared),
                            ref_rvalue(place_of(4, &string), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(5, &shared))],
                            place_of(6, &window),
                            5,
                            7,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(7, &window),
                            ref_rvalue(deref_place(6, &window, &slice), "Shared"),
                        )],
                        call_term(
                            4,
                            vec![move_op(place_of(7, &window))],
                            place_of(0, &unit),
                            6,
                            7,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            string_new_decl(),
            hint_must_use_decl(2),
            string_as_bytes_decl(),
            opaque_decl(4, &["fixture", "g"], vec![window], unit),
            fmt_format_decl(5),
        ],
    );
    assert!(
        has_path_leaf(&graph, "as_bytes"),
        "mixed format/new String::as_bytes keeps the residual call: {graph:#?}"
    );
    assert!(
        !has_path_leaf(
            &graph,
            crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING
        ),
        "mixed format/new as_bytes is not get_raw_address_of_string: {graph:#?}"
    );
}

fn shadowed_builder_as_bytes_graph(
    name: &str,
    types: Vec<Value>,
    builder: &Value,
    mut_borrow: &Value,
    shared_borrow: &Value,
    new_decl: Value,
    push_decl: Value,
    as_bytes_decl: Value,
) -> FunctionGraph {
    let word = word_ty();
    let unit = unit_ty();
    let flag = bool_ty();
    let window = window_shared_slice();
    lower_owned_concat(
        name,
        types,
        vec![
            subject_decl(
                name,
                3,
                vec![word.clone(), flag.clone(), window.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &word),
                    local_of(2, &flag),
                    local_of(3, &window),
                    local_of(4, builder),
                    local_of(5, mut_borrow),
                    local_of(6, &unit),
                    local_of(7, shared_borrow),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(4, builder), 1, 7)),
                    bb(
                        vec![assign_stmt(
                            place_of(5, mut_borrow),
                            ref_rvalue(place_of(4, builder), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![
                                move_op(place_of(5, mut_borrow)),
                                copy_op(place_of(1, &word)),
                            ],
                            place_of(6, &unit),
                            2,
                            7,
                        ),
                    ),
                    bb(
                        vec![],
                        json!({"Switch": {
                            "discr": {"Copy": place_of(2, &flag)},
                            "targets": {"If": [3, 4]}
                        }}),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(7, shared_borrow),
                            ref_rvalue(place_of(4, builder), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(7, shared_borrow))],
                            place_of(3, &window),
                            5,
                            7,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &window),
                            use_copy(place_of(3, &window)),
                        )],
                        json!({"Goto": {"target": 5}}),
                    ),
                    bb(
                        vec![],
                        call_term(
                            4,
                            vec![move_op(place_of(3, &window))],
                            place_of(0, &unit),
                            6,
                            7,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            new_decl,
            push_decl,
            as_bytes_decl,
            opaque_decl(4, &["fixture", "g"], vec![window], unit),
        ],
    )
}

fn stringbuilder_new_ids(graph: &FunctionGraph) -> std::collections::HashSet<u64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::STRINGBUILDER_NEW) =>
            {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect()
}

fn assert_shadowed_builder_as_bytes_builds_the_str(graph: &FunctionGraph, name: &str) {
    assert!(
        has_path_leaf(graph, crate::runtime_names::shims::STRINGBUILDER_NEW),
        "{name}: accumulator must be in builder mode: {graph:#?}"
    );
    assert!(
        has_path_leaf(graph, crate::runtime_names::shims::STRINGBUILDER_BUILD),
        "{name}: as_bytes of a shadowed builder dest must ll_build: {graph:#?}"
    );
    assert!(
        !has_path_leaf(graph, "as_bytes"),
        "{name}: as_bytes must not stay residual: {graph:#?}"
    );
    let new_ids = stringbuilder_new_ids(graph);
    let g_fed = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(String::as_str) == Some("g") => {
                args.iter().any(|arg| new_ids.contains(&arg.id()))
            }
            _ => false,
        });
    assert!(
        !g_fed,
        "{name}: g must not take the StringBuilder: {graph:#?}"
    );
    let join_ids: Vec<_> = graph
        .blocks
        .iter()
        .filter(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().map(String::as_str) == Some("g")
                )
            })
        })
        .map(|block| block.id)
        .collect();
    let join_ptr_is_new = graph.blocks.iter().any(|block| {
        block.exits.iter().any(|link| {
            join_ids.contains(&link.target)
                && link
                    .args
                    .iter()
                    .any(|arg| arg.as_variable().is_some_and(|v| new_ids.contains(&v.id())))
        })
    });
    assert!(
        !join_ptr_is_new,
        "{name}: bb_a→bb_join pointer must not be STRINGBUILDER_NEW: {graph:#?}"
    );
}

/// Mixed defs of a builder `String::as_bytes` window with a `&[u8]`
/// parameter: dest keeps a length shadow, so `ll_build`
/// (`AbstractStringBuilderRepr.rtype_method_build`) must still fire.
#[test]
fn shadowed_builder_string_as_bytes_builds_the_str() {
    let string = string_adt();
    let borrow = ref_string("Mut");
    let shared = ref_string("Shared");
    let graph = shadowed_builder_as_bytes_graph(
        "shadowed_builder_string_as_bytes",
        vec![string_decl()],
        &string,
        &borrow,
        &shared,
        string_new_decl(),
        string_push_str_decl(),
        string_as_bytes_decl(),
    );
    assert_shadowed_builder_as_bytes_builds_the_str(&graph, "shadowed_builder_string_as_bytes");
}

/// Mixed defs of a builder `Wtf8Buf::as_bytes` window with a `&[u8]`
/// parameter: dest keeps a length shadow, so `ll_build`
/// (`AbstractStringBuilderRepr.rtype_method_build`) must still fire.
#[test]
fn shadowed_builder_wtf8buf_as_bytes_builds_the_str() {
    let wtf8 = wtf8_adt();
    let borrow = ref_wtf8("Mut");
    let shared = ref_wtf8("Shared");
    let graph = shadowed_builder_as_bytes_graph(
        "shadowed_builder_wtf8buf_as_bytes",
        vec![wtf8buf_decl()],
        &wtf8,
        &borrow,
        &shared,
        wtf8buf_new_decl(),
        push_str_decl(),
        as_bytes_decl(),
    );
    assert_shadowed_builder_as_bytes_builds_the_str(&graph, "shadowed_builder_wtf8buf_as_bytes");
}

/// `String::new(); s.push_str(x); g(s.as_ref())` with `g(&[u8])`.
/// `<String as AsRef<[u8]>>::as_ref` is `as_bytes`, so `ll_build`
/// (`AbstractStringBuilderRepr.rtype_method_build`) makes the receiver
/// a `Ptr(STR)` and the window dest is a W1 STR view of the built
/// string rather than a residual `as_ref` of the builder.
#[test]
fn builder_as_ref_bytes_into_window_param_builds_the_str() {
    let string = string_adt();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_string("Mut");
    let shared = ref_string("Shared");
    let window = window_shared_slice();
    let graph = lower_owned_concat(
        "builder_as_ref_window_param",
        vec![string_decl()],
        vec![
            subject_decl(
                "builder_as_ref_window_param",
                1,
                vec![word.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &word),
                    local_of(2, &string),
                    local_of(3, &borrow),
                    local_of(4, &unit),
                    local_of(5, &shared),
                    local_of(6, &window),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &string), 1, 5)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &borrow),
                            ref_rvalue(place_of(2, &string), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &borrow)), copy_op(place_of(1, &word))],
                            place_of(4, &unit),
                            2,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(5, &shared),
                            ref_rvalue(place_of(2, &string), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(5, &shared))],
                            place_of(6, &window),
                            3,
                            5,
                        ),
                    ),
                    bb(
                        vec![],
                        call_term(
                            4,
                            vec![move_op(place_of(6, &window))],
                            place_of(0, &unit),
                            4,
                            5,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            string_new_decl(),
            string_push_str_decl(),
            string_as_ref_bytes_decl(),
            opaque_decl(4, &["fixture", "g"], vec![window.clone()], unit),
        ],
    );
    assert!(
        has_path_leaf(&graph, crate::runtime_names::shims::STRINGBUILDER_BUILD),
        "as_ref of a builder into a window param must ll_build: {graph:#?}"
    );
    assert!(
        !has_path_leaf(&graph, "as_ref"),
        "as_ref must not stay residual: {graph:#?}"
    );
    let new_ids: std::collections::HashSet<u64> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::STRINGBUILDER_NEW) =>
            {
                op.result.as_ref().map(|v| v.id())
            }
            _ => None,
        })
        .collect();
    let fed_by_new = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => {
                let leaf = segments.last().map(String::as_str);
                let watches_g = leaf == Some("g");
                let watches_window = leaf
                    == Some(crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING)
                    || leaf == Some("__strlen");
                (watches_g || watches_window) && args.iter().any(|arg| new_ids.contains(&arg.id()))
            }
            _ => false,
        });
    assert!(
        !fed_by_new,
        "g (or the window conversion feeding it) must not take the StringBuilder: {graph:#?}"
    );
}

/// `Wtf8Buf::new(); buf.push_str(s); Deref::deref(&buf).as_bytes()` —
/// the Shared borrow is consumed by string-family deref, then as_bytes.
/// Both are identity; `ll_build` fires at the deref.
#[test]
fn owned_ctor_push_str_deref_as_bytes_lowers_to_stringbuilder() {
    let wtf8 = wtf8_adt();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let shared = ref_wtf8("Shared");
    let bytes = bytes_slice_ty();
    let graph = lower_owned_concat(
        "ctor_push_deref_as_bytes",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "ctor_push_deref_as_bytes",
                1,
                vec![word.clone()],
                bytes.clone(),
                vec![
                    local_of(0, &bytes),
                    local_of(1, &word),
                    local_of(2, &wtf8),
                    local_of(3, &borrow),
                    local_of(4, &unit),
                    local_of(5, &shared),
                    local_of(6, &shared),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &wtf8), 1, 5)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &borrow),
                            ref_rvalue(place_of(2, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &borrow)), copy_op(place_of(1, &word))],
                            place_of(4, &unit),
                            2,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(5, &shared),
                            ref_rvalue(place_of(2, &wtf8), "Shared"),
                        )],
                        call_term(
                            4,
                            vec![move_op(place_of(5, &shared))],
                            place_of(6, &shared),
                            3,
                            5,
                        ),
                    ),
                    bb(
                        vec![],
                        call_term(
                            3,
                            vec![move_op(place_of(6, &shared))],
                            place_of(0, &bytes),
                            4,
                            5,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            wtf8buf_new_decl(),
            push_str_decl(),
            as_bytes_decl(),
            deref_decl(),
        ],
    );
    assert_builder_identity_lowering(&graph, "ctor_push_deref_as_bytes");
}

/// Exclusive-branch `as_bytes` on one builder: two `ll_build` points
/// (`AbstractStringBuilderRepr.rtype_method_build` may run on each
/// terminal read). `push_str` must not stay residual.
#[test]
fn owned_ctor_branched_as_bytes_lowers_to_stringbuilder() {
    let wtf8 = wtf8_adt();
    let flag = bool_ty();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let shared = ref_wtf8("Shared");
    let bytes = bytes_slice_ty();
    let graph = lower_owned_concat(
        "ctor_branched_as_bytes",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "ctor_branched_as_bytes",
                2,
                vec![flag.clone(), word.clone()],
                bytes.clone(),
                vec![
                    local_of(0, &bytes),
                    local_of(1, &flag),
                    local_of(2, &word),
                    local_of(3, &wtf8),
                    local_of(4, &borrow),
                    local_of(5, &unit),
                    local_of(6, &shared),
                    local_of(7, &borrow),
                    local_of(8, &unit),
                    local_of(9, &shared),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(3, &wtf8), 1, 7)),
                    bb(
                        vec![],
                        json!({"Switch": {
                            "discr": copy_op(place_of(1, &flag)),
                            "targets": {"If": [2, 4]}
                        }}),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(4, &borrow),
                            ref_rvalue(place_of(3, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(4, &borrow)), copy_op(place_of(2, &word))],
                            place_of(5, &unit),
                            3,
                            7,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(6, &shared),
                            ref_rvalue(place_of(3, &wtf8), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(6, &shared))],
                            place_of(0, &bytes),
                            6,
                            7,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(7, &borrow),
                            ref_rvalue(place_of(3, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(7, &borrow)), copy_op(place_of(2, &word))],
                            place_of(8, &unit),
                            5,
                            7,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(9, &shared),
                            ref_rvalue(place_of(3, &wtf8), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(9, &shared))],
                            place_of(0, &bytes),
                            6,
                            7,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            wtf8buf_new_decl(),
            push_str_decl(),
            as_bytes_decl(),
        ],
    );
    assert_builder_identity_lowering(&graph, "ctor_branched_as_bytes");
}

/// An owned `Wtf8Buf` defined by a non-constructor call, then `push_str`,
/// rebinds as `ll_strconcat`.  The producer is not a builder ctor, so the
/// append must not become a `StringBuilder` operation.
#[test]
fn owned_call_result_push_str_lowers_to_ll_strconcat() {
    let wtf8 = wtf8_adt();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let graph = lower_owned_concat(
        "owned_call_result_concat",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "owned_call_result_concat",
                1,
                vec![word.clone()],
                wtf8.clone(),
                vec![
                    local_of(0, &wtf8),
                    local_of(1, &word),
                    local_of(2, &wtf8),
                    local_of(3, &borrow),
                    local_of(4, &unit),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &wtf8), 1, 3)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &borrow),
                            ref_rvalue(place_of(2, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(3, &borrow)), copy_op(place_of(1, &word))],
                            place_of(4, &unit),
                            2,
                            3,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(0, &wtf8),
                            use_move(place_of(2, &wtf8)),
                        )],
                        json!("Return"),
                    ),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            opaque_decl(1, &["fixture", "prefix"], vec![], wtf8),
            push_str_decl(),
        ],
    );
    assert!(
        has_str_add(&graph),
        "a call-defined owned buffer must lower push_str to ll_strconcat: {graph:#?}"
    );
    assert!(
        !has_path_leaf(&graph, "push_str"),
        "push_str must not stay residual: {graph:#?}"
    );
    assert!(
        !has_path_leaf(&graph, "__majit_stringbuilder_new"),
        "a non-ctor producer must not become StringBuilder::new: {graph:#?}"
    );
}

/// `let mut out = producer()?;` binds `out` by moving the `Continue`
/// payload.  That single definition is an owned `Wtf8Buf`, so `push_str`
/// rebinds it as `ll_strconcat`.
#[test]
fn continue_payload_move_push_str_lowers_to_ll_strconcat() {
    let wtf8 = wtf8_adt();
    let flow = outcome_adt();
    let word = word_ty();
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let graph = lower_owned_concat(
        "continue_payload_concat",
        vec![wtf8buf_decl(), outcome_decl()],
        vec![
            subject_decl(
                "continue_payload_concat",
                1,
                vec![word.clone()],
                wtf8.clone(),
                vec![
                    local_of(0, &wtf8),
                    local_of(1, &word),
                    local_of(2, &flow),
                    local_of(3, &wtf8),
                    local_of(4, &borrow),
                    local_of(5, &unit),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &flow), 1, 3)),
                    bb(
                        vec![
                            assign_stmt(place_of(3, &wtf8), use_move(field_place(2, &flow, &wtf8))),
                            assign_stmt(
                                place_of(4, &borrow),
                                ref_rvalue(place_of(3, &wtf8), "Mut"),
                            ),
                        ],
                        call_term(
                            2,
                            vec![move_op(place_of(4, &borrow)), copy_op(place_of(1, &word))],
                            place_of(5, &unit),
                            2,
                            3,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(0, &wtf8),
                            use_move(place_of(3, &wtf8)),
                        )],
                        json!("Return"),
                    ),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            opaque_decl(1, &["fixture", "prefix"], vec![], flow),
            push_str_decl(),
        ],
    );
    assert!(
        has_str_add(&graph),
        "a Continue-payload accumulator must lower push_str to ll_strconcat: {graph:#?}"
    );
    assert!(
        !has_path_leaf(&graph, "push_str"),
        "push_str must not stay residual: {graph:#?}"
    );
}

/// A `&mut Wtf8Buf` parameter is the caller's buffer.  `push_str` stays a
/// residual call.
#[test]
fn mut_wtf8buf_parameter_push_str_stays_residual() {
    let unit = unit_ty();
    let borrow = ref_wtf8("Mut");
    let word = word_ty();
    let graph = lower_owned_concat(
        "mut_param_keeps_push",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "mut_param_keeps_push",
                2,
                vec![borrow.clone(), word.clone()],
                unit.clone(),
                vec![local_of(0, &unit), local_of(1, &borrow), local_of(2, &word)],
                vec![
                    bb(
                        vec![],
                        call_term(
                            2,
                            vec![move_op(place_of(1, &borrow)), copy_op(place_of(2, &word))],
                            place_of(0, &unit),
                            1,
                            2,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            opaque_decl(1, &["fixture", "prefix"], vec![], unit.clone()),
            push_str_decl(),
        ],
    );
    assert!(
        has_path_leaf(&graph, "push_str"),
        "a &mut Wtf8Buf parameter must keep its residual push_str: {graph:#?}"
    );
    assert!(
        !has_str_add(&graph),
        "a parameter accumulator must not become ll_strconcat: {graph:#?}"
    );
}

/// An owned buffer that is also borrowed for a call other than an append
/// is not a clean accumulator, so `push_str` stays residual.
#[test]
fn owned_buffer_borrowed_outside_append_stays_residual() {
    let wtf8 = wtf8_adt();
    let word = word_ty();
    let unit = unit_ty();
    let shared = ref_wtf8("Shared");
    let borrow = ref_wtf8("Mut");
    let graph = lower_owned_concat(
        "escaped_borrow_keeps_push",
        vec![wtf8buf_decl()],
        vec![
            subject_decl(
                "escaped_borrow_keeps_push",
                1,
                vec![word.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &word),
                    local_of(2, &wtf8),
                    local_of(3, &shared),
                    local_of(4, &unit),
                    local_of(5, &borrow),
                ],
                vec![
                    bb(vec![], call_term(1, vec![], place_of(2, &wtf8), 1, 4)),
                    bb(
                        vec![assign_stmt(
                            place_of(3, &shared),
                            ref_rvalue(place_of(2, &wtf8), "Shared"),
                        )],
                        call_term(
                            3,
                            vec![move_op(place_of(3, &shared))],
                            place_of(4, &unit),
                            2,
                            4,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(5, &borrow),
                            ref_rvalue(place_of(2, &wtf8), "Mut"),
                        )],
                        call_term(
                            2,
                            vec![move_op(place_of(5, &borrow)), copy_op(place_of(1, &word))],
                            place_of(0, &unit),
                            3,
                            4,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            opaque_decl(1, &["fixture", "prefix"], vec![], wtf8),
            push_str_decl(),
            opaque_decl(3, &["fixture", "escape"], vec![shared], unit),
        ],
    );
    assert!(
        has_path_leaf(&graph, "push_str"),
        "a buffer borrowed outside an append must keep residual push_str: {graph:#?}"
    );
    assert!(
        !has_str_add(&graph),
        "an escaped borrow must not become ll_strconcat: {graph:#?}"
    );
}

/// RPython `AbstractStringBuilderRepr.rtype_method_append` accepts both a
/// whole string and `SomeChar`.  `bytearray_repr_string` mixes
/// `String::push_str` and `String::push(char)` in one loop, so omitting the
/// char arm prevents the whole fresh accumulator from lifting.
#[test]
#[ignore]
fn bytearray_repr_mixed_string_appends_lift_to_rpython_builder() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "bytearray_repr_string").expect("lower bytearray_repr_string");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|leaf| leaf == "push" || leaf == "push_str"))
            }),
            "String::push/push_str must not survive the RPython builder lift"
        );
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.iter().any(|segment| segment == "fmt")
                        || segments.last().is_some_and(|leaf| leaf == "new_lower_hex"))
            }),
            "Rust fmt shell must collapse to PyPy's two-nibble lookup"
        );
    assert!(
            graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.first().is_some_and(|s| s == "__str_const")
                        && segments.get(1).is_some_and(|s| s == "\\x"))
            }),
            "the collapse must keep upstream's `buf.append('\\\\x')` \
             (bytearrayobject.py `descr_repr`), not drop it with the fmt shell"
        );
    for opname in ["rshift", "and"] {
        assert!(
            graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|op| { matches!(&op.kind, OpKind::BinOp { op, .. } if op == opname) }),
            "missing PyPy byte-hex operation {opname}"
        );
    }
    for marker in [
        "__majit_stringbuilder_new",
        "__majit_stringbuilder_append",
        "__majit_stringbuilder_build",
    ] {
        assert!(
                graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &[marker])
                }),
                "missing builder marker {marker}"
            );
    }
    for block in &graph.blocks {
        let mut defined: std::collections::HashSet<u64> =
            block.inputargs.iter().map(|var| var.id()).collect();
        for op in &block.operations {
            for operand in crate::inline::op_variable_refs(&op.kind) {
                assert!(
                    defined.contains(&operand.id()),
                    "block {:?} uses undefined operand {:?} in {:?}",
                    block.id,
                    operand,
                    op.kind
                );
            }
            if let Some(result) = &op.result {
                defined.insert(result.id());
            }
        }
    }
}

/// PyPy's `bytesobject.py::string_escape_encode` is an ordinary
/// StringBuilder loop.  The interpreter now carries that loop directly,
/// so the frozen production LLBC must lower without the former external
/// `rustpython_literal::escape::AsciiEscape` iterator.
#[test]
#[ignore]
fn bytes_repr_lifts_without_external_ascii_escape() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "bytes_repr_string").expect("lower bytes_repr_string");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.iter().any(|segment| segment == "AsciiEscape"))
            }),
            "external AsciiEscape must not survive the PyPy loop port"
        );
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|segment| segment == "HEX"))
            }),
            "the borrowed HEX table must be a prebuilt constant, not an accessor call"
        );
    for marker in [
        "__majit_stringbuilder_new",
        "__majit_stringbuilder_append",
        "__majit_stringbuilder_build",
    ] {
        assert!(
                graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &[marker])
                }),
                "missing builder marker {marker}"
            );
    }
}

/// PyPy's `unicodeobject.py::_repr_function` is the ordinary
/// `rutf8.make_utf8_escape_function(pass_printable=True, quotes=True)`
/// StringBuilder loop.  The production graph must retain that loop and
/// its prebuilt lowercase-hex table without the former RustPython escape
/// iterator or a named-const accessor.
#[test]
#[ignore]
fn unicode_repr_lifts_without_external_unicode_escape() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "format_wtf8_repr").expect("lower format_wtf8_repr");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.iter().any(|segment| segment == "UnicodeEscape"))
            }),
            "external UnicodeEscape must not survive the PyPy loop port"
        );
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|segment| segment == "HEX"))
            }),
            "the borrowed HEX table must be a prebuilt constant, not an accessor call"
        );
    for marker in [
        "__majit_stringbuilder_new",
        "__majit_stringbuilder_append",
        "__majit_stringbuilder_build",
    ] {
        assert!(
                graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &[marker])
                }),
                "missing builder marker {marker}"
            );
    }
}

/// PyPy initializes `W_UnicodeObject._index_storage` with
/// `lltype.nullptr(UTF8_INDEX_STORAGE)`, where the storage is a typed
/// `GcArray`.  Rust spells the same slot `*mut Vec<Utf8LocElem>`; its null
/// constructor value must therefore be narrowed to the projected list
/// repr before the field write, not left as a classdef-less pointer that
/// later conflicts with `create_utf8_index_storage`'s list value.
#[test]
#[ignore]
fn unicode_index_storage_null_keeps_gcarray_repr() {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let graph = super::lower_function(&llbc, "w_str_from_wtf8").expect("lower w_str_from_wtf8");
    let value = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldWrite { field, value, .. } if field.name == "index_storage" => match value
            {
                LinkArg::Value(value) => Some(value.clone()),
                LinkArg::Const(_) => None,
            },
            _ => None,
        })
        .expect("W_UnicodeObject.index_storage write");
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                op.result.as_ref() == Some(&value)
                    && matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if crate::model::cast_instance_root(&op.kind)
                            .is_some_and(|root| root.starts_with("Vec<"))
                    )
            }),
        "index_storage null must narrow to the Vec/GcArray list repr"
    );
}

#[test]
fn str_traced_codepoint_bound_some_ctors_keep_i64_instantiation() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "str_traced_codepoint_bound")
        .expect("lower str_traced_codepoint_bound");
    let some_owners: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                ..
            } if name == "Some" => Some(owner_path.join("::")),
            _ => None,
        })
        .collect();
    assert!(
        !some_owners.is_empty(),
        "expected Some constructors, got none"
    );
    for owner in &some_owners {
        assert!(
            owner.contains("<i64>"),
            "Some of Option<i64> must intern under the instantiation, got {owner:?} from {some_owners:?}"
        );
    }
    let none_owners: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                ..
            } if name == "None" => Some(owner_path.join("::")),
            _ => None,
        })
        .collect();
    for owner in &none_owners {
        assert!(
            owner.contains("<i64>"),
            "None of Option<i64> must intern under the instantiation, got {owner:?}"
        );
    }
}

#[test]
fn str_idx_params_unrooted_pos0_owners_are_option_i64() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "str_idx_params_unrooted")
        .expect("lower str_idx_params_unrooted");
    let pos0: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. }
                if field.name == "__pos_0"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|o| o.contains("Option")) =>
            {
                Some(field.owner_root.clone().unwrap_or_default())
            }
            _ => None,
        })
        .collect();
    assert!(
        !pos0.is_empty() && pos0.iter().all(|o| o.contains("<i64>")),
        "expected Option<i64>::Some payload reads, got {pos0:?}"
    );
}

#[test]
fn iobase_peek_dict_does_not_getattr_template_option_some() {
    // `iobase_payload_dict_slot` returns `Option<*mut PyObjectRef>`.
    // The `?` must not getattr `__pos_0` on the template `Option::Some`.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "iobase_peek_dict").expect("lower iobase_peek_dict");
    let template_pos0 = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::FieldRead { field, .. }
                if field.name == "__pos_0"
                    && field.owner_root.as_deref().is_some_and(|o| {
                        majit_ir::descr::strip_instantiation_suffix(o).ends_with("Option::Some")
                            || majit_ir::descr::strip_instantiation_suffix(o) == "Option::Some"
                    })
                    && !field.owner_root.as_deref().is_some_and(|o| o.contains('<')) =>
            {
                true
            }
            _ => false,
        });
    assert!(
        !template_pos0,
        "iobase_peek_dict must not read template Option::Some.__pos_0"
    );
}

#[test]
fn match_on_call_returned_option_usize_reads_instantiated_some_pos0() {
    // `w_list_getitem_inner` does `ll_getitem_index(...)?` on
    // `Option<usize>`. The payload read must key `Option<usize>::Some`,
    // the same ClassDef the callee's `Some` constructor interned
    // (`bookkeeper.py` `getuniqueclassdef(cls)`).
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "w_list_getitem_inner").expect("lower w_list_getitem_inner");
    let pos0: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. }
                if field.name == "__pos_0"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|o| o.contains("Option")) =>
            {
                Some(field.owner_root.clone().unwrap_or_default())
            }
            _ => None,
        })
        .collect();
    assert!(
        !pos0.is_empty() && pos0.iter().all(|o| o.contains("<usize>")),
        "expected Option<usize>::Some payload reads, got {pos0:?}"
    );
}

#[test]
fn std_gc_array_size_does_not_getattr_template_option_some() {
    // `usize::checked_add(..)?` twice. The payload read must key
    // `Option<usize>::Some`, not the unsuffixed template
    // (`bookkeeper.py` `getuniqueclassdef(cls)`).
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_object::object_array::std_gc_array_size")
        .expect("lower std_gc_array_size");
    let pos0: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. }
                if field.name == "__pos_0"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|o| o.contains("Option")) =>
            {
                Some(field.owner_root.clone().unwrap_or_default())
            }
            _ => None,
        })
        .collect();
    assert!(
        !pos0.is_empty() && pos0.iter().all(|o| o.contains("<usize>")),
        "expected Option<usize>::Some payload reads, got {pos0:?}"
    );
    let some_owners: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                ..
            } if name == "Some" => Some(owner_path.join("::")),
            _ => None,
        })
        .collect();
    for owner in &some_owners {
        assert!(
            owner.contains("<usize>"),
            "Some of Option<usize> must intern under the instantiation, got {owner:?} from {some_owners:?}"
        );
    }
}

#[test]
fn strategy_is_kind_and_expected_share_int_signedness() {
    // `current.kind == expected` over fieldless `StrategyKind`. The
    // field getattr and the parameter are Signed (`Int`): Rust's
    // default discriminant is `isize`. Charon's physical tag may
    // spell `u8` for layout width; the annotation stays Signed.
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_object::dictmultiobject::strategy_is")
        .expect("lower strategy_is");
    let kind_ty = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldRead { field, ty, .. } if field.name == "kind" => Some(ty.clone()),
            _ => None,
        });
    let expected_ty = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::Input { ty, .. } if matches!(ty, ValueType::Int | ValueType::Unsigned) => {
                Some(ty.clone())
            }
            _ => None,
        });
    assert_eq!(kind_ty.as_ref(), Some(&ValueType::Int));
    assert_eq!(
        kind_ty, expected_ty,
        "fieldless enum discriminant must be Signed, kind={kind_ty:?} expected={expected_ty:?}"
    );
}

/// The actual phaseA hitter must construct Some on the same owner as
/// ordinary source constructors, never write payload onto bare Option.
#[test]
#[ignore]
fn checked_sub_constructs_real_some_in_peekvalue_maybe_none() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "peekvalue_maybe_none")
        .expect("lower real checked_sub caller");
    let constructors: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                ..
            } => Some((name, owner_path)),
            _ => None,
        })
        .collect();
    assert!(
        !constructors
            .iter()
            .any(|(name, _)| majit_ir::descr::strip_instantiation_suffix(name) == "Option")
    );
    assert!(
        constructors
            .iter()
            .any(|(name, owner)| name.as_str() == "Some"
                && owner.last().is_some_and(|root| root == "Option<usize>")),
        "checked_sub must use the same Some owner as a source constructor: {constructors:?}"
    );
}

/// `Wtf8::as_str` is fallible for a lone surrogate. Keep its validity
/// branch and concrete Result variants, not an unconditional identity.
#[test]
#[ignore]
fn wtf8_as_str_lowers_to_runtime_tagged_string_result() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "wtf8_display_string").expect("lower wtf8_display_string");

    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();
    assert!(!ops.iter().any(|op| matches!(&op.kind,
            OpKind::Call { target: CallTarget::SyntheticTransparentCtor { name, .. }, .. }
                if matches!(majit_ir::descr::strip_instantiation_suffix(name), "Option" | "Result"))),
            "native fallible conversion must construct concrete variants, not an enum root");
    assert!(
            !ops.iter().any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["Wtf8", "as_str"])
            }),
            "the foreign Rust Result must not cross the translated graph: {graph:#?}"
        );
    assert!(
            ops.iter().any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, result_ty: ValueType::Bool, .. }
                    if segments.last().map(String::as_str) == Some("wtf8_key_is_utf8"))
            }),
            "the lone-surrogate decision must remain a runtime scalar: {graph:#?}"
        );
    assert!(
            ops.iter().any(|op| {
                matches!(&op.kind, OpKind::FieldWrite { field, ty: ValueType::Str, .. }
                    if field.name == "__pos_0" && field.owner_root.as_deref().is_some_and(|owner| owner.ends_with("::Ok")))
            }),
            "the Result Ok payload must keep the lifted string repr: {graph:#?}"
        );
}

/// `panic!("x {} y", n)` / `assert_eq!(a, b, "m {}", c)` / live
/// `format!("line {}", n)` — the `removeassert.py` melt of abort-bound
/// `format!` construction (`remove_asserts` / `kill_assertion_link`).
pub(in crate::front::mir) fn leftover_fmt_message_paths(
    graph: &crate::model::FunctionGraph,
) -> Vec<Vec<String>> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| {
            let crate::model::OpKind::Call { target, .. } = &op.kind else {
                return None;
            };
            let segs: Vec<String> = match target {
                crate::model::CallTarget::FunctionPath { segments, .. } => segments.clone(),
                crate::model::CallTarget::Method {
                    name,
                    resolved_path,
                    ..
                } => resolved_path
                    .as_ref()
                    .map(|p| p.segments.clone())
                    .unwrap_or_else(|| vec![name.clone()]),
                _ => return None,
            };
            let leaf = segs
                .last()
                .map(|s| crate::front::clause_spec::unspecialized_leaf(s).to_string());
            let is_fmt = super::fmt_path_ends_with(&segs, &["fmt", "format"])
                || segs.iter().any(|s| s == "Arguments")
                || (segs.iter().any(|s| s == "Argument")
                    && leaf.as_deref().is_some_and(|l| l.starts_with("new_")));
            is_fmt.then_some(segs)
        })
        .collect()
}

fn has_implicit_assertion_error(graph: &crate::model::FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .exits
            .iter()
            .any(|link| oefmt_link_is_assert_raise(link, graph.exceptblock))
    })
}

fn regular_call_block(
    fun_id: u64,
    args: Vec<serde_json::Value>,
    dest: serde_json::Value,
    target: u64,
    unwind: u64,
) -> serde_json::Value {
    serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {
            "span": span_json(),
            "kind": {
                "Call": {
                    "call": {
                        "func": {
                            "Regular": {
                                "kind": {"Fun": fun_id},
                                "generics": empty_generics()
                            }
                        },
                        "args": args,
                        "dest": dest
                    },
                    "target": target,
                    "on_unwind": unwind
                }
            }
        }
    })
}

fn fmt_chain_opaque_fun(
    def_id: u64,
    path: &[&str],
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", false),
        "signature": {
            "is_unsafe": false,
            "inputs": inputs,
            "output": output
        },
        "body": "Opaque"
    })
}

fn fmt_placeholder_types() -> (
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    Vec<serde_json::Value>,
) {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let unit = serde_json::json!({"Tuple": []});
    let argument_ty = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": empty_generics()}
    });
    let arguments_ty = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": empty_generics()}
    });
    let str_ty = serde_json::json!({
        "Adt": {"builtin": "Str", "generics": empty_generics()}
    });
    let types = vec![
        serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["core", "fmt", "rt", "Argument"], "", false),
            "kind": {"Struct": []}
        }),
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["core", "fmt", "Arguments"], "", false),
            "kind": {"Struct": []}
        }),
    ];
    (i64_ty, unit, argument_ty, arguments_ty, str_ty, types)
}

fn fmt_placeholder_callees(
    argument_ty: &serde_json::Value,
    arguments_ty: &serde_json::Value,
    str_ty: &serde_json::Value,
    i64_ty: &serde_json::Value,
    unit: &serde_json::Value,
    abort_leaf: &str,
) -> Vec<serde_json::Value> {
    vec![
        fmt_chain_opaque_fun(
            1,
            &["core", "fmt", "rt", "Argument", "new_display"],
            vec![i64_ty.clone()],
            argument_ty.clone(),
        ),
        fmt_chain_opaque_fun(
            2,
            &["core", "fmt", "Arguments", "new"],
            vec![argument_ty.clone()],
            arguments_ty.clone(),
        ),
        fmt_chain_opaque_fun(
            3,
            &["alloc", "fmt", "format"],
            vec![arguments_ty.clone()],
            str_ty.clone(),
        ),
        fmt_chain_opaque_fun(
            4,
            &["core", "panicking", abort_leaf],
            vec![str_ty.clone()],
            unit.clone(),
        ),
    ]
}

fn fmt_placeholder_chain_blocks(
    i64_ty: &serde_json::Value,
    argument_ty: &serde_json::Value,
    arguments_ty: &serde_json::Value,
    str_ty: &serde_json::Value,
    n_local: u64,
    arg_local: u64,
    args_local: u64,
    formatted_local: u64,
    after_format: u64,
    unwind: u64,
) -> Vec<serde_json::Value> {
    let copy = |i: u64, ty: &serde_json::Value| serde_json::json!({"Copy": {"kind": {"Local": i}, "ty": ty}});
    let dest = |i: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": i}, "ty": ty});
    vec![
        regular_call_block(
            1,
            vec![copy(n_local, i64_ty)],
            dest(arg_local, argument_ty),
            1,
            unwind,
        ),
        regular_call_block(
            2,
            vec![copy(arg_local, argument_ty)],
            dest(args_local, arguments_ty),
            2,
            unwind,
        ),
        regular_call_block(
            3,
            vec![copy(args_local, arguments_ty)],
            dest(formatted_local, str_ty),
            after_format,
            unwind,
        ),
    ]
}

/// `panic!("x {} y", n)`: the placeholder chain feeds only
/// `panic_fmt`.  `removeassert.py` melts it to a bare implicit
/// `AssertionError` raise.
#[test]
fn panic_placeholder_format_melts_to_implicit_assertion_error() {
    let (i64_ty, unit, argument_ty, arguments_ty, str_ty, types) = fmt_placeholder_types();
    let unwind = 4u64;
    let mut body = fmt_placeholder_chain_blocks(
        &i64_ty,
        &argument_ty,
        &arguments_ty,
        &str_ty,
        1,
        2,
        3,
        4,
        3,
        unwind,
    );
    body.push(regular_call_block(
        4,
        vec![serde_json::json!({"Copy": {"kind": {"Local": 4}, "ty": str_ty}})],
        serde_json::json!({"kind": {"Local": 0}, "ty": unit}),
        5,
        unwind,
    ));
    body.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": "UnwindResume"}
    }));
    body.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": "Return"}
    }));
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "panic_placeholder"], "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [i64_ty.clone()],
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span_json(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span_json(), "ty": unit},
                        {"index": 1, "name": "n", "span": span_json(), "ty": i64_ty},
                        {"index": 2, "name": "arg", "span": span_json(), "ty": argument_ty},
                        {"index": 3, "name": "args", "span": span_json(), "ty": arguments_ty},
                        {"index": 4, "name": "msg", "span": span_json(), "ty": str_ty}
                    ]
                },
                "body": body
            }
        }
    });
    let mut funs = vec![caller];
    funs.extend(fmt_placeholder_callees(
        &argument_ty,
        &arguments_ty,
        &str_ty,
        &i64_ty,
        &unit,
        "panic_fmt",
    ));
    let llbc = llbc_with_types("fixture", types, funs);
    let graph =
        super::lower_function(&llbc, "panic_placeholder").expect("lower panic!(\"x {} y\", n)");
    let leftover = leftover_fmt_message_paths(&graph);
    assert!(
        leftover.is_empty(),
        "panic! placeholder format must melt: leftover={leftover:?} graph={graph:#?}"
    );
    assert!(
        has_implicit_assertion_error(&graph),
        "panic! must remain RaiseImplicit(AssertionError): {graph:#?}"
    );
    let start = graph.block(graph.startblock);
    let input_ops = start
        .operations
        .iter()
        .filter(|op| matches!(op.kind, crate::model::OpKind::Input { .. }))
        .count();
    assert_eq!(
        input_ops,
        start.inputargs.len(),
        "melt must keep startblock Input ops (`stamp_declared`): {graph:#?}"
    );
}

/// `assert_eq!(a, b, "m {}", c)`: the fail arm is the same placeholder
/// chain into `assert_failed`.  Format / Arguments must not remain.
#[test]
fn assert_eq_placeholder_format_melts_to_implicit_assertion_error() {
    let (i64_ty, unit, argument_ty, arguments_ty, str_ty, types) = fmt_placeholder_types();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let unwind = 6u64;
    let copy = |i: u64, ty: &serde_json::Value| serde_json::json!({"Copy": {"kind": {"Local": i}, "ty": ty}});
    let dest = |i: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": i}, "ty": ty});
    let body = vec![
        serde_json::json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {
                "span": span_json(),
                "kind": {
                    "Switch": {
                        "discr": {"Copy": {"kind": {"Local": 1}, "ty": bool_ty}},
                        "targets": {"If": [1, 2]}
                    }
                }
            }
        }),
        serde_json::json!({
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 0}, "ty": unit},
                    {"Aggregate": ["Tuple", []]}
                ]}
            }],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }),
        regular_call_block(1, vec![copy(2, &i64_ty)], dest(3, &argument_ty), 3, unwind),
        regular_call_block(
            2,
            vec![copy(3, &argument_ty)],
            dest(4, &arguments_ty),
            4,
            unwind,
        ),
        regular_call_block(3, vec![copy(4, &arguments_ty)], dest(5, &str_ty), 5, unwind),
        regular_call_block(4, vec![copy(5, &str_ty)], dest(0, &unit), 1, unwind),
        serde_json::json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "UnwindResume"}
        }),
    ];
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "assert_eq_placeholder"], "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [bool_ty.clone(), i64_ty.clone()],
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span_json(),
                "locals": {
                    "arg_count": 2,
                    "locals": [
                        {"index": 0, "name": null, "span": span_json(), "ty": unit},
                        {"index": 1, "name": "eq", "span": span_json(), "ty": bool_ty},
                        {"index": 2, "name": "c", "span": span_json(), "ty": i64_ty},
                        {"index": 3, "name": "arg", "span": span_json(), "ty": argument_ty},
                        {"index": 4, "name": "args", "span": span_json(), "ty": arguments_ty},
                        {"index": 5, "name": "msg", "span": span_json(), "ty": str_ty}
                    ]
                },
                "body": body
            }
        }
    });
    let mut funs = vec![caller];
    funs.extend(fmt_placeholder_callees(
        &argument_ty,
        &arguments_ty,
        &str_ty,
        &i64_ty,
        &unit,
        "assert_failed",
    ));
    let llbc = llbc_with_types("fixture", types, funs);
    let graph = super::lower_function(&llbc, "assert_eq_placeholder")
        .expect("lower assert_eq!(a, b, \"m {}\", c)");
    let leftover = leftover_fmt_message_paths(&graph);
    assert!(
        leftover.is_empty(),
        "assert_eq! placeholder format must melt: leftover={leftover:?} graph={graph:#?}"
    );
}

/// `let s = format!("line {}", n); use(s)` is not abort-bound.
/// `is_fmt_format_call` + `traces_to_str_const` stay for a literal;
/// a placeholder `format!` keeps the format call or the existing
/// str-concat expansion.
#[test]
fn live_format_placeholder_is_not_melted() {
    let (i64_ty, unit, argument_ty, arguments_ty, str_ty, types) = fmt_placeholder_types();
    let unwind = 4u64;
    let mut body = fmt_placeholder_chain_blocks(
        &i64_ty,
        &argument_ty,
        &arguments_ty,
        &str_ty,
        1,
        2,
        3,
        0,
        3,
        unwind,
    );
    body.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": "Return"}
    }));
    body.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": "UnwindResume"}
    }));
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "live_format_placeholder"], "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [i64_ty.clone()],
            "output": str_ty.clone()
        },
        "body": {
            "Unstructured": {
                "span": span_json(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span_json(), "ty": str_ty},
                        {"index": 1, "name": "n", "span": span_json(), "ty": i64_ty},
                        {"index": 2, "name": "arg", "span": span_json(), "ty": argument_ty},
                        {"index": 3, "name": "args", "span": span_json(), "ty": arguments_ty}
                    ]
                },
                "body": body
            }
        }
    });
    let mut funs = vec![caller];
    funs.extend(fmt_placeholder_callees(
        &argument_ty,
        &arguments_ty,
        &str_ty,
        &i64_ty,
        &unit,
        "panic_fmt",
    ));
    let llbc = llbc_with_types("fixture", types, funs);
    let graph = super::lower_function(&llbc, "live_format_placeholder")
        .expect("lower live format!(\"line {}\", n)");
    let leftover = leftover_fmt_message_paths(&graph);
    let has_str_concat =
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| match &op.kind {
                crate::model::OpKind::BinOp { op: name, .. } if name == "add" => true,
                crate::model::OpKind::UnaryOp { op: name, .. } if name == "str" => true,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.first().map(String::as_str) == Some("__str_const") => true,
                _ => false,
            });
    assert!(
        !leftover.is_empty() || has_str_concat,
        "live format! must keep its format call or the str-concat expansion: \
             leftover={leftover:?} graph={graph:#?}"
    );
    assert!(
        !has_implicit_assertion_error(&graph),
        "live format! must not become RaiseImplicit(AssertionError): {graph:#?}"
    );
}

fn oefmt_call_paths(graph: &crate::model::FunctionGraph) -> Vec<Vec<String>> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect()
}

fn oefmt_mentions_cow(graph: &crate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| {
            let cow_in = |s: &str| s.contains("Cow");
            match &op.kind {
                OpKind::Call {
                    target, result_ty, ..
                } => {
                    let path = match target {
                        CallTarget::FunctionPath { segments, .. } => {
                            segments.iter().any(|s| cow_in(s))
                        }
                        CallTarget::Method {
                            name,
                            receiver_root,
                            resolved_path,
                            ..
                        } => {
                            cow_in(name)
                                || receiver_root.as_deref().is_some_and(cow_in)
                                || resolved_path
                                    .as_ref()
                                    .is_some_and(|p| p.segments.iter().any(|s| cow_in(s)))
                        }
                        CallTarget::SyntheticTransparentCtor {
                            name, owner_path, ..
                        } => cow_in(name) || owner_path.iter().any(|s| cow_in(s)),
                        _ => false,
                    };
                    let ty = match result_ty {
                        ValueType::Ref(Some(root)) => cow_in(root),
                        _ => false,
                    };
                    path || ty
                }
                OpKind::FieldRead { field, ty, .. } | OpKind::FieldWrite { field, ty, .. } => {
                    field.owner_root.as_deref().is_some_and(cow_in)
                        || matches!(ty, ValueType::Ref(Some(root)) if cow_in(root))
                }
                _ => false,
            }
        })
}

fn oefmt_is_empty_str_const(op: &crate::model::SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments.as_slice() == ["__str_const", ""]
    )
}

fn oefmt_link_is_assert_raise(link: &crate::model::Link, except: crate::model::BlockId) -> bool {
    use crate::flowspace::model::ConstValue;
    if link.target != except {
        return false;
    }
    let Some(LinkArg::Const(ty)) = link.args.first() else {
        return false;
    };
    let ConstValue::HostObject(class) = &ty.value else {
        return false;
    };
    if !class.is_class() || !class.qualname().contains("AssertionError") {
        return false;
    }
    let Some(LinkArg::Const(value)) = link.args.get(1) else {
        return false;
    };
    let ConstValue::HostObject(inst) = &value.value else {
        return false;
    };
    inst.is_instance()
        && inst.instance_args().is_some_and(|args| {
            matches!(
                args.first(),
                Some(ConstValue::UniStr(msg))
                    if msg == "implicit AssertionError shouldn't occur"
            )
        })
}

/// `decode_utf8_with_errors` Err arm is `RaiseImplicit(AssertionError)`
/// (`rtyper/debug.py ll_assert`), never a fabricated `""`.
/// `remove_assertion_errors` (simplify.py) may prune that arm.
fn oefmt_decode_utf8_err_arm_is_assert_raise(graph: &crate::model::FunctionGraph) {
    let decode_bb = graph
            .blocks
            .iter()
            .find(|block| {
                block.operations.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.ends_with(&["typedef".to_string(), "decode_utf8_with_errors".to_string()])
                ))
            })
            .expect("decode_utf8_with_errors call");
    assert!(
        !decode_bb.operations.iter().any(oefmt_is_empty_str_const),
        "decode block must not fabricate \"\": {decode_bb:#?}"
    );
    let except = graph.exceptblock;
    let mut saw_assert_raise = false;
    for link in &decode_bb.exits {
        if oefmt_link_is_assert_raise(link, except) {
            saw_assert_raise = true;
            continue;
        }
        let succ = graph.block(link.target);
        assert!(
            !succ.operations.iter().any(oefmt_is_empty_str_const),
            "decode successor must not fabricate \"\": {succ:#?}"
        );
        if succ
            .exits
            .iter()
            .any(|exit| oefmt_link_is_assert_raise(exit, except))
        {
            saw_assert_raise = true;
        }
    }
    assert!(
        saw_assert_raise || decode_bb.exits.len() == 1,
        "Err arm must RaiseImplicit(AssertionError) or be pruned by \
             remove_assertion_errors: {graph:#?}"
    );
}

/// Checked `Wtf8::from_bytes` is `rutf8.py check_utf8` plus the
/// Option Some/None wrap (`unicodehelper.py str_decode_utf8` fast arm).
#[test]
fn oefmt_wtf8_from_bytes_lowers_to_check_utf8_option() {
    use crate::model::{CallTarget, OpKind};
    let interpreter = Llbc::load(crate::runtime_names::artifacts::INTERPRETER_ULLBC)
        .expect("load interpreter LLBC");
    let object =
        Llbc::load(crate::runtime_names::artifacts::OBJECT_ULLBC).expect("load object LLBC");
    let program = super::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter, object],
        crate::HostStaticAddrs {
            error_carrier: crate::ErrorCarrierSpec {
                carrier_path: "pyre_interpreter::error::PyError",
                carrier_class: "",
                carrier_wrappers: &[],
                to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
                from_exc_object: Some(("PyError", "from_exc_object")),
            },
            ..Default::default()
        },
        &["error"],
        &["oefmt"],
    )
    .expect("lower oefmt");
    let graph = program
        .functions
        .iter()
        .find(|f| f.name == "oefmt")
        .expect("oefmt graph")
        .graph();
    let paths = oefmt_call_paths(graph);
    assert!(
        !paths.iter().any(|segments| {
            segments.ends_with(&["Wtf8".to_string(), "from_bytes".to_string()])
        }),
        "checked Wtf8::from_bytes must not remain a residual: {paths:#?}"
    );
    assert!(
        paths.iter().any(|segments| {
            segments.ends_with(&["rutf8".to_string(), "check_utf8".to_string()])
        }),
        "rutf8.check_utf8 must remain in the caller: {paths:#?}"
    );
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();
    let check_utf8_argc = ops.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.ends_with(&["rutf8".to_string(), "check_utf8".to_string()]) => {
            Some(args.len())
        }
        _ => None,
    });
    assert_eq!(
        check_utf8_argc,
        Some(3),
        "check_utf8 takes the window pair plus allow_surrogates: {graph:#?}"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::ConstBool(true))),
        "allow_surrogates=True is ConstBool(true): {graph:#?}"
    );
    let has_null_mut = ops.iter().any(|op| {
        matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if segments == &["core".to_string(), "ptr".to_string(), "null_mut".to_string()])
    });
    let has_option_ctor = ops.iter().any(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { name, .. },
            ..
        } => matches!(name.as_str(), "Some" | "None"),
        _ => false,
    });
    assert!(
        has_null_mut || has_option_ctor,
        "Ok/Err of check_utf8 must construct Option Some/None: {graph:#?}"
    );
}

/// `String::from_utf8_lossy` is `unicodehelper.py str_decode_utf8`
/// replace / U+FFFD (`typedef::decode_utf8_with_errors`), never a Cow.
/// The Result Err arm is `RaiseImplicit(AssertionError)`
/// (`rtyper/debug.py ll_assert`), never a fabricated `""`.
#[test]
#[ignore]
fn oefmt_from_utf8_lossy_lowers_to_decode_utf8_replace() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "oefmt").expect("lower oefmt");
    let paths = oefmt_call_paths(&graph);
    assert!(
        !paths.iter().any(|segments| {
            segments
                .last()
                .is_some_and(|leaf| leaf == "from_utf8_lossy")
        }),
        "from_utf8_lossy must not remain a residual: {paths:#?}"
    );
    assert!(
        paths.iter().any(|segments| {
            segments.ends_with(&["typedef".to_string(), "decode_utf8_with_errors".to_string()])
        }),
        "decode_utf8_with_errors must remain in the caller: {paths:#?}"
    );
    assert!(
        !oefmt_mentions_cow(&graph),
        "from_utf8_lossy dest is a plain string, not Cow: {graph:#?}"
    );
    oefmt_decode_utf8_err_arm_is_assert_raise(&graph);
}

/// `decompose_valuefmt` `current: String` is a `StringBuilder`
/// (`rbuilder.py`): `push(char)` is `unichr_as_utf8` then append,
/// `mem::take` is `build()` plus a fresh builder.
#[test]
#[ignore]
fn decompose_valuefmt_lowers_string_accumulator_push_and_take() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "decompose_valuefmt").expect("lower decompose_valuefmt");
    let paths = oefmt_call_paths(&graph);
    let leftover: Vec<_> = paths
        .iter()
        .filter(|segments| {
            let leaf = segments.last().map(String::as_str);
            let is_string_push =
                leaf == Some("push") && segments.iter().any(|s| s == "String" || s == "Wtf8Buf");
            let is_mem_take = leaf == Some("take")
                && segments.windows(2).any(|w| w[0] == "mem" && w[1] == "take");
            let is_str_contains = leaf == Some("contains") && segments.iter().any(|s| s == "str");
            is_string_push || is_mem_take || is_str_contains
        })
        .cloned()
        .collect();
    assert!(
        leftover.is_empty(),
        "String/Wtf8Buf::push, mem::take, and str::contains must not remain residual: {leftover:#?}"
    );
    // The `contains` dest feeds an If whose false arm is only
    // `fmt::Arguments::from_str` + Panic.  That chain is
    // `set_raise_implicit(AssertionError)`, so `remove_assertion_errors`
    // promotes the If and `prune_dead_phis` reclaims the pure BinOp.
    // Live dest is asserted by `str_contains_lowers_as_binop_contains`.
    assert!(
        paths.iter().any(|segments| {
            segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::UNICHAR_AS_UTF8)
                || (segments.first().map(String::as_str) == Some("__str_const"))
        }),
        "push(char) must become unichr_as_utf8 or a one-char str const: {paths:#?}"
    );
    assert!(
        paths.iter().any(|segments| {
            matches!(
                segments.last().map(String::as_str),
                Some(crate::runtime_names::shims::STRINGBUILDER_APPEND)
                    | Some(crate::runtime_names::shims::STRINGBUILDER_BUILD)
                    | Some(crate::runtime_names::shims::STRINGBUILDER_NEW)
            )
        }) || graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, crate::model::OpKind::BinOp { op, .. } if op == "add"))
        }),
        "the accumulator must append (builder or ll_strconcat): {graph:#?}"
    );
    let leftover_some_casts: Vec<&str> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| {
            crate::model::cast_instance_root(&op.kind).and_then(|root| {
                (root.contains("Option") && root.ends_with("::Some")).then_some(root)
            })
        })
        .collect();
    assert!(
        leftover_some_casts.is_empty(),
        "inner next payload downcasts must collapse; leftover {leftover_some_casts:#?}"
    );
    let residual_next: Vec<_> = paths
        .iter()
        .filter(|segments| {
            crate::front::iter_next::is_iterator_next_target(&CallTarget::FunctionPath {
                segments: (*segments).clone(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            })
        })
        .cloned()
        .collect();
    assert!(
        residual_next.is_empty(),
        "Iterator::next must rewrite to [__iter_next]: {residual_next:#?}"
    );
}

/// `oefmt_single_t` `valuefmt.as_ptr()` is `rffi.get_raw_address_of_string`
/// (`__majit_get_raw_address_of_string`).  A leftover
/// `core::str::<Impl>::as_ptr` FunctionPath is the prepass wall.
#[test]
#[ignore]
fn oefmt_single_t_lowers_str_as_ptr_to_get_raw_address() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "oefmt_single_t").expect("lower oefmt_single_t");
    let paths = oefmt_call_paths(&graph);
    let leftover: Vec<_> = paths
        .iter()
        .filter(|segments| {
            let leaf = segments.last().map(String::as_str);
            leaf == Some("as_ptr") && segments.iter().any(|s| s == "str")
        })
        .cloned()
        .collect();
    assert!(
        leftover.is_empty(),
        "str::as_ptr must not remain residual: {leftover:#?}"
    );
    assert!(
        paths.iter().any(|segments| {
            segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING)
        }),
        "as_ptr must become get_raw_address_of_string: {paths:#?}"
    );
}

/// `core::str::<Impl>::contains` with a returned dest, so the BinOp
/// survives `prune_dead_phis`.  A `char` Pattern is `unichr_as_utf8`
/// then Contains; `&str` is Contains on the string (`rstr.py`
/// `rtype_contains`).
#[test]
fn str_contains_lowers_as_binop_contains() {
    let str_ty = str_ref_ty();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let char_ty = serde_json::json!({"Scalar": "Char"});
    let graph = super::lower_function(
        &std_extern_call_fixture(
            "str_contains_char",
            &["core", "str", "<Impl>", "contains"],
            &[str_ty.clone(), char_ty],
            bool_ty.clone(),
        ),
        "str_contains_char",
    )
    .expect("lower str::contains(char)");
    let paths = oefmt_call_paths(&graph);
    assert!(
        !paths.iter().any(|segments| {
            segments.last().map(String::as_str) == Some("contains")
                && segments.iter().any(|s| s == "str")
        }),
        "str::contains must not remain residual: {paths:#?}"
    );
    let binops: Vec<&str> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            crate::model::OpKind::BinOp { op, .. } => Some(op.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        binops.iter().any(|op| *op == "contains"),
        "char needle must become BinOp contains; binops={binops:?} paths={paths:#?}"
    );
    assert!(
        paths.iter().any(|segments| {
            segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::UNICHAR_AS_UTF8)
        }),
        "char needle must become unichr_as_utf8: {paths:#?}"
    );

    let graph = super::lower_function(
        &std_extern_call_fixture(
            "str_contains_str",
            &["core", "str", "<Impl>", "contains"],
            &[str_ty.clone(), str_ty],
            bool_ty,
        ),
        "str_contains_str",
    )
    .expect("lower str::contains(&str)");
    let paths = oefmt_call_paths(&graph);
    let binops: Vec<&str> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            crate::model::OpKind::BinOp { op, .. } => Some(op.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        binops.iter().any(|op| *op == "contains"),
        "&str needle must become BinOp contains; binops={binops:?} paths={paths:#?}"
    );
    assert!(
        !paths.iter().any(|segments| {
            segments.last().map(String::as_str)
                == Some(crate::runtime_names::shims::UNICHAR_AS_UTF8)
        }),
        "&str needle must not go through unichr_as_utf8: {paths:#?}"
    );
}

/// `exception_descr_str_wtf8` indexes the Python unicode carrier through
/// `Wtf8::index(RangeFrom)`. The frontend already models Wtf8 as
/// RPython's string repr, so this is its ordinary `getslice(start, None)`
/// and the Rust trait shim must be gone before annotation.
#[test]
#[ignore]
fn wtf8_rangefrom_lowers_to_getslice_marker() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "exception_descr_str_wtf8")
        .expect("lower exception_descr_str_wtf8");
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["Wtf8".to_string(), "index".to_string()])
            }),
            "Wtf8 RangeFrom index must not survive as an opaque foreign call"
        );
    assert!(
            graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["__getslice_rangefrom".to_string()])
            }),
            "runtime RangeFrom should use the post-annotation getslice marker"
        );
}

/// PyPy `typeobject._check_surrogate` delegates to the elidable
/// `rutf8.check_utf8`; it does not build a unicode iterator in the caller.
/// Keep that exact boundary visible in the real extracted graph.
#[test]
#[ignore]
fn check_surrogate_calls_rutf8_without_wtf8_iterator_next() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "check_surrogate").expect("lower check_surrogate");
    let call_paths: Vec<&[String]> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.as_slice()),
            _ => None,
        })
        .collect();
    assert!(
        call_paths.iter().any(|segments| {
            segments.ends_with(&["rutf8".to_string(), "check_utf8".to_string()])
        }),
        "PyPy's rutf8.check_utf8 boundary must remain in the caller"
    );
    assert!(
        !call_paths
            .iter()
            .any(|segments| matches!(*segments, [owner, method]
                    if owner == "Wtf8CodePoints" && (method == "next" || method == "nth"))),
        "the local Wtf8 iterator reimplementation must not return: {call_paths:#?}"
    );
}

/// RPython `rutf8.codepoint_at_pos` reads each encoded byte as
/// `ord(code[pos])`. The Rust byte view must retain that exact string
/// getitem + ord shape and never fall back to the crate iterator.
#[test]
#[ignore]
fn rutf8_codepoint_at_pos_lowers_byte_reads_to_ord() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "codepoint_at_pos").expect("lower rutf8.codepoint_at_pos");
    let paths: Vec<&[String]> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.as_slice()),
            _ => None,
        })
        .collect();
    assert!(
        paths
            .iter()
            .any(|segments| *segments == ["__string_byte_getitem"]),
        "byte reads must reach the adapter's getitem + ord marker: {paths:#?}"
    );
    assert!(
        !paths
            .iter()
            .any(|segments| *segments == ["Wtf8CodePoints", "next"]),
        "RPython's direct decoder must not be replaced by Wtf8 iteration"
    );
}

/// `PyError::new` initializes three `PyObjectRef` fields with
/// `null_mut()`. These are typed nullable W_Root slots, not erased root
/// references; each aggregate write must consume a PyObject-narrowed
/// value so `InstanceRepr._setup_repr` and later reads agree.
#[test]
#[ignore]
fn pyerror_new_narrows_null_object_fields() {
    use crate::model::{LinkArg, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::error::<Impl>::new")
        .expect("lower PyError::new");
    for field_name in ["exc_object", "w_name_context", "w_obj_context"] {
        let value = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite { field, value, .. } if field.name == field_name => {
                    match value {
                        LinkArg::Value(value) => Some(value.clone()),
                        LinkArg::Const(_) => None,
                    }
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("PyError::{field_name} aggregate write"));
        assert!(
            graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                op.result.as_ref() == Some(&value)
                    && crate::model::cast_instance_root(&op.kind) == Some("PyObject")
            }),
            "PyError::{field_name} null must retain the declared PyObject class"
        );
    }
}

/// The same declared-slot rule applies after construction.  PyPy's
/// `OperationError` object attributes never lose their W_Root class when
/// assigned in a method, so an ordinary MIR projection write must carry
/// the same narrowing as a struct-literal initializer.
#[test]
#[ignore]
fn pyerror_projection_writes_retain_instance_class() {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::error::<Impl>::enrich_attribute_error",
    )
    .expect("lower PyError::enrich_attribute_error");
    let mut writes = 0;
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::FieldWrite { field, value, .. } = &op.kind else {
            continue;
        };
        if !matches!(field.name.as_str(), "w_name_context" | "w_obj_context") {
            continue;
        }
        writes += 1;
        let LinkArg::Value(value) = value else {
            panic!(
                "PyError::{} write must use a materialized object value",
                field.name
            )
        };
        assert!(
            graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|producer| {
                    producer.result.as_ref() == Some(value)
                        && matches!(
                            &producer.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if crate::model::cast_instance_root(&producer.kind)
                                == Some("PyObject")
                        )
                }),
            "PyError::{} projection write must retain the declared PyObject class",
            field.name
        );
    }
    assert!(writes >= 2, "expected both PyError context-field writes");
}

/// `W_DictObject.dstrategy` is PyPy's ordinary instance attribute.  Rust
/// carries the singleton strategy through `&'static DictStrategyRef`, but
/// assigning that reference must still feed `setattr` a classed instance;
/// otherwise the field's constructor annotation and setter annotation meet
/// as `SomeInstance(DictStrategyRef) ∪ SomePtr`.
#[test]
#[ignore]
fn dict_strategy_setter_retains_instance_class() {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_object::dictmultiobject::w_dict_set_strategy")
        .expect("lower w_dict_set_strategy");
    let value = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldWrite { field, value, .. } if field.name == "dstrategy" => match value {
                LinkArg::Value(value) => Some(value.clone()),
                LinkArg::Const(_) => None,
            },
            _ => None,
        })
        .expect("W_DictObject.dstrategy write");
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            op.result.as_ref() == Some(&value)
                && matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if crate::model::cast_instance_root(&op.kind)
                        .is_some_and(|root| root.ends_with("DictStrategyRef"))
                )
        }),
        "dstrategy assignment must retain the declared DictStrategyRef class"
    );
}

/// `BufferRequest` is a payload-free enum and therefore travels through
/// the translated graph as its integer discriminant.  Its inherent
/// methods remain statically selected Rust functions; lowering them as
/// Python bound-method lookups would ask `SomeInteger` for
/// `require_contiguous` / `flags` and block annotation.
#[test]
#[ignore]
fn buffer_request_scalar_methods_keep_direct_graph_paths() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::baseobjspace::buffer_bytes")
        .expect("lower buffer_bytes");

    for method in ["require_contiguous", "flags"] {
        assert!(
            graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.last().map(String::as_str) == Some(method)
                            && segments.iter().any(|part| part == "BufferRequest")
                    )
                }),
            "BufferRequest::{method} must be a direct static call"
        );
        assert!(
            !graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::Method { name, .. },
                            ..
                        } if name == method
                    )
                }),
            "BufferRequest::{method} must not become an integer getattr"
        );
    }
}

/// PyPy's `instantiate(W_IntObject)` is lowered to a real allocation plus
/// `intval` store, never left as the Rust helper that accepted the
/// stack-built aggregate.  The real LLBC constructor must therefore leave
/// this frontend with `NewWithVtable`, not residual `malloc_typed`.
#[test]
#[ignore]
fn w_int_new_fuses_typed_allocation() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function_with_static_addrs(
        &llbc,
        "pyre_object::intobject::w_int_new",
        crate::HostStaticAddrs {
            pytypes: &[("pyobject::INT_TYPE", 0x1020_3040)],
            ..crate::HostStaticAddrs::default()
        },
    )
    .expect("lower w_int_new");
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(&op.kind, OpKind::NewWithVtable { owner, .. }
                    if owner.ends_with("W_IntObject"))
        }),
        "w_int_new must lower instantiate(W_IntObject) to NewWithVtable: {graph:#?}"
    );
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("malloc_typed")
            )
        }),
        "w_int_new must not retain residual malloc_typed: {graph:#?}"
    );
}

/// `forwarded_exporter` uses
/// `W_PickleBuffer::from_obj(obj).map(|pb| -> Result<_, PyError> { ... })`.
/// The closure graph follows the uniform Result exception ABI, while
/// `Option::map` must retain that Result as data inside `Some`; anchor the
/// late synthesized call to the ordinary catch-and-rewrap caller rule.
#[test]
#[ignore]
fn forwarded_exporter_map_rewraps_scoped_result() {
    use crate::model::{CallTarget, ExitSwitch, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    // `lower_function`'s default host table names NO error carrier, and
    // the carrier is what `tyref_is_result_of_carrier` compares `E`
    // against — with the default the closure's `Result<*mut PyObject,
    // PyError>` is not a scoped result, `call_once_result_exc` stays
    // `None`, and the rewrap this test grades can never be reached.
    // Pass the same spec `pyre-jit-trace`'s prepass installs.
    let graph = super::lower_function_with_static_addrs(
        &llbc,
        "forwarded_exporter",
        crate::HostStaticAddrs {
            error_carrier: crate::ErrorCarrierSpec {
                carrier_path: "pyre_interpreter::error::PyError",
                carrier_class: "",
                carrier_wrappers: &[],
                to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
                from_exc_object: Some(("PyError", "from_exc_object")),
            },
            ..crate::HostStaticAddrs::default()
        },
    )
    .expect("lower forwarded_exporter");

    let call_block = graph
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
        .expect("Option::map closure call_once block");
    assert!(
        matches!(call_block.exitswitch, Some(ExitSwitch::LastException)),
        "the Result-returning closure call must expose normal/exception successors"
    );
    for variant in ["Ok", "Err"] {
        assert_eq!(
            graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .filter(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } if name == variant
                ))
                .count(),
            1,
            "the caller must rebuild one Result::{variant} shell"
        );
    }
}

/// PyPy's `_IOBase.writelines_w` receives `self` directly from the
/// gateway and raises on a missing receiver.  The Rust gateway spells the
/// same boundary as `args.first().copied().ok_or_else(...)?`; after the
/// combined lowering neither the foreign Option combinator nor the Result
/// `?` shell may survive, and the niladic closure's error must terminate on
/// the graph exception edge.
#[test]
#[ignore]
fn iobase_writelines_ok_or_else_try_is_value_or_exception() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function_with_static_addrs(
        &llbc,
        "iobase_writelines",
        crate::HostStaticAddrs {
            error_carrier: crate::ErrorCarrierSpec {
                carrier_path: "pyre_interpreter::error::PyError",
                carrier_class: "",
                carrier_wrappers: &[],
                to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
                from_exc_object: Some(("PyError", "from_exc_object")),
            },
            ..crate::HostStaticAddrs::default()
        },
    )
    .expect("lower iobase_writelines");

    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "ok_or_else"
            )
        }),
        "the foreign Option::ok_or_else residual must be gone"
    );
    let error_arm = graph
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
        .expect("the None arm calls its error closure");
    assert!(
        error_arm
            .exits
            .iter()
            .any(|link| link.target == graph.exceptblock),
        "the None arm must raise through the native graph exception edge"
    );
}

/// A by-value fixed array is an RPython list-model iteration source, just
/// like the already-covered `&[T]` loop: its `IntoIter::next` must become
/// the native `iter`/`next` vertical rather than remain a foreign core
/// call.  Loads the real interpreter LLBC, so ignored by default.
#[test]
#[ignore]
fn fixed_array_for_loop_has_no_residual_into_iter_next() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "memoryview_is_native_buffer_descr")
        .expect("lower memoryview_is_native_buffer_descr");
    let residual: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("next") => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        residual.is_empty(),
        "fixed-array for-loop left residual next calls: {residual:?}"
    );
}

/// Native lock acquisition and the process-global method metadata lookup
/// are genuine runtime boundaries.  PyPy reaches the former through
/// `rthread.py`'s `llexternal` calls; none may expose Rust's
/// `Result<_, PoisonError<_>>::unwrap_or_else` shell to the source graph.
#[test]
#[ignore]
fn native_lock_and_method_metadata_leaves_are_residual() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let residuals = super::dont_look_inside_set_of(&llbc);
    for expected in [
        "module::_queue::queue_lock",
        "module::_queue::simplequeue_wait_for_item",
        "module::thread::lock_class::lock_state",
        "module::thread::lock_class::<Impl>::acquire_timed",
        "module::thread::rlock_class::lock_state",
        "module::thread::rlock_class::<Impl>::acquire_timed",
        "function::object_class_method_name",
    ] {
        assert!(
            residuals.contains(expected),
            "native/runtime leaf {expected} must remain residual; got {residuals:?}"
        );
    }
}

/// A prebuilt holder is an instance even when it has fields. Preserve its
/// registered identity and class when installing it into a dict, matching
/// Bookkeeper.immutablevalue and W_DictObject's strategy instance slot.
#[test]
#[ignore = "requires the extracted pyre-object LLBC"]
fn prebuilt_strategy_holder_keeps_its_instance_repr() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("load object LLBC");
    let addr = 0x1234_5678;
    let refs = [("dictmultiobject::EMPTY_KWARGS_DICT_STRATEGY_REF", addr)];
    let graph = super::lower_function_with_static_addrs(
        &llbc,
        "pyre_object::dictmultiobject::w_dict_new_kwargs",
        crate::HostStaticAddrs {
            refs: &refs,
            ..Default::default()
        },
    )
    .expect("lower kwargs dict constructor");
    let raw = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::ConstRefAddr(value) if *value == addr => op.result.clone(),
            _ => None,
        })
        .expect("the holder retains its registered prebuilt address");
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(&op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        args,
                        result_ty: ValueType::Ref(Some(owner)),
                    } if crate::model::cast_instance_root(&op.kind)
                        == Some("dictmultiobject::DictStrategyRef")
                        && owner == "dictmultiobject::DictStrategyRef"
                        && args.first().and_then(crate::model::LinkArg::as_variable)
                            == Some(&raw))
            }),
        "a field-bearing prebuilt holder must enter annotation as its own instance"
    );

    // This registry row is the Python object returned by the accessor,
    // not the opaque OnceLock global that initializes it. Do not infer
    // the container's class from the registered object's address.
    let refs = [("noneobject::NONE_SINGLETON", addr)];
    let graph = super::lower_function_with_static_addrs(
        &llbc,
        "pyre_object::noneobject::w_none",
        crate::HostStaticAddrs {
            refs: &refs,
            ..Default::default()
        },
    )
    .expect("lower singleton accessor");
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(&op.kind, OpKind::ConstRefAddr(value) if *value == addr))
    );
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(&op.kind,
                OpKind::Call { result_ty: ValueType::Ref(Some(owner)), .. }
                    if owner.contains("OnceLock")))
    );
}

/// `baseobjspace.py newbool` is a branch on the value and a prebuilt
/// constant per arm. Each arm's `get_or_init` on a registered singleton
/// is that constant, so no `get_or_init` call survives the lowering.
#[test]
#[ignore = "requires the extracted pyre-object LLBC"]
fn bool_singleton_accessor_lowers_to_prebuilt_constants() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("load object LLBC");
    let (w_true, w_false) = (0x1234_5678, 0x1234_9abc);
    let refs = [
        ("boolobject::TRUE_SINGLETON", w_true),
        ("boolobject::FALSE_SINGLETON", w_false),
    ];
    let graph = super::lower_function_with_static_addrs(
        &llbc,
        "pyre_object::boolobject::w_bool_from",
        crate::HostStaticAddrs {
            refs: &refs,
            ..Default::default()
        },
    )
    .expect("lower w_bool_from");
    let ops = || graph.blocks.iter().flat_map(|block| &block.operations);
    for addr in [w_true, w_false] {
        assert!(ops().any(|op| matches!(&op.kind, OpKind::ConstRefAddr(value) if *value == addr)));
    }
    // The `OnceLock<usize>` word is read back as the GC reference it
    // stores, with no `cast_int_to_ptr` of the prebuilt constant.
    assert!(!ops().any(|op| matches!(&op.kind,
    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
        if matches!(
            segments.last().map(String::as_str),
            Some("get_or_init" | "cast_int_to_ptr")
        ))));
}

/// `policy.py look_inside_graph` reads `_jit_look_inside_` first and
/// otherwise lets `_reject_function` reject an `_elidable_function_`
/// unconditionally, so both marker spellings name one set of body-less
/// residual boundaries.
#[test]
fn both_opaque_markers_reject_a_body_and_the_override_admits_it() {
    let hint = |s: &str| vec![s.to_string()];
    assert!(super::hints_reject_body(&hint("dont_look_inside")));
    assert!(super::hints_reject_body(&hint("elidable")));
    assert!(!super::hints_reject_body(&hint("jit_look_inside")));
    assert!(!super::hints_reject_body(&hint("unroll_safe")));
    assert!(!super::hints_reject_body(&[]));
    // `_jit_look_inside_ = True` overrides the elidable rejection.
    assert!(!super::hints_reject_body(&[
        "elidable".to_string(),
        "jit_look_inside".to_string(),
    ]));
}

/// RPython evaluates `SHIFT`, `HASH_BITS`, `_HASH_SHIFT`, and
/// `HASH_MODULUS` while building `_hash_long`'s flow graph.  The extracted
/// Rust graph must likewise contain their integer literals, not synthetic
/// zero-argument accessors that the annotator cannot resolve.  Loads the
/// real interpreter LLBC, so ignored by default.
#[test]
#[ignore]
fn hash_long_has_no_residual_constant_accessors() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "_hash_long").expect("lower _hash_long");
    let residual: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if matches!(
                segments.last().map(String::as_str),
                Some("SHIFT" | "HASH_BITS" | "HASH_SHIFT" | "HASH_MODULUS")
            ) =>
            {
                Some(segments.clone())
            }
            _ => None,
        })
        .collect();
    assert!(
        residual.is_empty(),
        "_hash_long left residual constant accessors: {residual:?}"
    );
}

/// `Option<usize>` has no null niche. `cap.checked_mul(..)?` in
/// `try_typed_items_block_layout` is `Try::branch` on that enum, which
/// must become a discriminant switch rather than `getattr("branch")`.
#[test]
fn try_typed_items_block_layout_branches_on_usize_discriminant() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/majit-rlib.ullbc"
    );
    let llbc = Llbc::load(path).expect("load majit-rlib.ullbc");
    let graph = super::lower_function(&llbc, "try_typed_items_block_layout")
        .expect("lower try_typed_items_block_layout");
    let branch = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "branch"
            )
        })
        .count();
    assert_eq!(branch, 0, "Option<usize> ? left a residual branch call");
    let discs = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant"
            )
        })
        .count();
    assert!(
        discs >= 1,
        "non-niche Option<usize> must switch on __discriminant"
    );
}

#[test]
fn charpsize2str_cp_input_is_ccharp() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/majit-rlib.ullbc"
    );
    let llbc = Llbc::load(path).expect("load majit-rlib.ullbc");
    let graph = super::lower_function(&llbc, "charpsize2str").expect("lower charpsize2str");
    let mut cp_root = None;
    let mut cp_ty = None;
    let mut list_ptr = 0usize;
    let mut copy_bytes = 0usize;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::Input {
                name,
                ty,
                class_root,
                ..
            } if name == "cp" => {
                cp_root = class_root.clone();
                cp_ty = Some(ty.clone());
            }
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                if segments.last().map(String::as_str)
                    == Some(crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST)
                {
                    list_ptr += 1;
                }
                if segments.last().map(String::as_str) == Some("copy_bytes") {
                    copy_bytes += 1;
                }
            }
            _ => {}
        }
    }
    assert_eq!(
        cp_root.as_deref(),
        Some("CCHARP"),
        "charpsize2str cp is CCHARP; graph={}",
        graph.dump()
    );
    assert_eq!(
        cp_ty,
        Some(ValueType::Unsigned),
        "CCHARP getkind is int; graph={}",
        graph.dump()
    );
    assert!(
        list_ptr >= 1,
        "Vec<u8>::as_mut_ptr is nonmoving_raw_ptr_for_resizable_list; graph={}",
        graph.dump()
    );
    assert!(
        copy_bytes >= 1,
        "charpsize2str still calls copy_bytes; graph={}",
        graph.dump()
    );
}

/// `rbigint.py` `ptwotable` is a prebuilt list. `_x_mul` must read that
/// array, not a residual accessor named `PTWOTABLE`.
#[test]
fn x_mul_ptwotable_is_a_prebuilt_int_array() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/majit-rlib.ullbc"
    );
    let llbc = Llbc::load(path).expect("load majit-rlib.ullbc");
    let graph = super::lower_function(&llbc, "_x_mul").expect("lower _x_mul");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("PTWOTABLE")
            )
        })
        .count();
    assert_eq!(residual, 0, "_x_mul left a PTWOTABLE accessor call");
    let tables: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.first().map(String::as_str) == Some("__const_int_array") => {
                Some(segments.clone())
            }
            _ => None,
        })
        .collect();
    assert!(
        tables.iter().any(|segments| {
            segments.len() == 64 && segments[1] == "0" && segments[2] == "1" && segments[63] == "62"
        }),
        "_x_mul must bake ptwotable[i] = i for i in 0..63, got {tables:?}"
    );
}

/// CPython 3.14's hashable-slice mixer spells PyPy's `xxrotate` expansion
/// as Rust `u64::rotate_left`.  The front must recover the same unsigned
/// shift/or graph rather than leave the opaque core method behind.
#[test]
#[ignore]
fn slice_hash_has_no_residual_rotate_left() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "slice_hash_value").expect("lower slice_hash_value");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("rotate_left")
            )
        })
        .count();
    assert_eq!(residual, 0, "slice hash left rotate_left residuals");
    assert!(
        graph.blocks.iter().flat_map(|block| &block.operations).any(
            |op| matches!(&op.kind, OpKind::BinOp { op, result_ty: ValueType::Unsigned, .. }
                    if op == "or")
        ),
        "slice hash must contain PyPy's unsigned shift/or rotate expansion"
    );
}

/// `w_method_new` pins three roots with `for slot in save_point..save_point
/// + 3`.  RPython represents that loop through `builtin_range` and
/// `RangeIteratorRepr`; the Rust `core::ops::range::Range<usize>` shell
/// must therefore be consumed by `front::range_iter`, not enter the
/// annotator as one shared foreign ClassDef whose `start` field can merge
/// with unrelated Range instantiations.  Loads the real object LLBC, so
/// ignored by default.
#[test]
#[ignore]
fn method_root_scan_uses_rpython_range_iterator() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real object LLBC");
    let graph = super::lower_function(&llbc, "pyre_object::function::w_method_new")
        .expect("lower w_method_new");
    let operations: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();

    let range_shells: Vec<_> = operations
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    result_ty: ValueType::Ref(Some(owner)),
                    ..
                } if owner.ends_with("ops::range::Range")
            )
        })
        .collect();
    assert!(
        range_shells.is_empty(),
        "the Rust Range shell must be consumed before annotation: {range_shells:#?}"
    );
    assert!(
        operations.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &[crate::runtime_names::shims::RANGE.to_string()]
        )),
        "w_method_new must route its root scan through RPython builtin_range"
    );
    assert!(
        operations.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &["__iter_next".to_string()]
        )),
        "w_method_new must lower the range loop through the native next op"
    );
}

pub(in crate::front::mir) fn disc_site(
    kind: super::DiscCombinator,
    result_var: Variable,
) -> super::DiscCombinatorSite {
    super::DiscCombinatorSite {
        kind,
        result_var,
        call_args: Vec::new(),
        recv_owner: "core::result::Result".into(),
        recv_tag0_owner: "core::result::Result::Ok".into(),
        recv_tag1_owner: "core::result::Result::Err".into(),
        payload0_ty: ValueType::Int,
        payload1_ty: ValueType::Int,
        payload0_class: None,
        payload1_class: None,
        result_owner: "core::result::Result".into(),
        result_tag0_owner: "core::result::Result::Ok".into(),
        result_tag1_owner: "core::result::Result::Err".into(),
        result_payload0_ty: ValueType::Int,
        result_payload1_ty: ValueType::Int,
        result_payload0_class: None,
        result_payload1_class: None,
        call_once_owner: "test::closure".into(),
        fn_item_segments: None,
        fn_item_ctor: None,
        args_tuple_suffix: String::new(),
        call_result_ty: ValueType::Int,
        call_result_class: None,
        carrier_recv: false,
        recv_niche: false,
        scalar_niche: false,
        result_niche: false,
        result_scalar_niche: false,
        niche_null_cast: None,
        int_null: None,
    }
}

pub(in crate::front::mir) fn option_disc_site(
    kind: super::DiscCombinator,
    result_var: Variable,
) -> super::DiscCombinatorSite {
    let mut site = disc_site(kind, result_var);
    site.recv_owner = "core::option::Option".into();
    site.recv_tag1_owner = "core::option::Option::Some".into();
    site.result_owner = "core::option::Option".into();
    site.result_tag1_owner = "core::option::Option::Some".into();
    site.call_result_ty = ValueType::Bool;
    site
}

pub(in crate::front::mir) fn count_method_calls(graph: &FunctionGraph, method: &str) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == method
            )
        })
        .count()
}

pub(in crate::front::mir) fn build_two_arg_combinator(method: &str) -> (FunctionGraph, Variable) {
    let mut graph = FunctionGraph::new("test_disc_combinator");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let extra = graph.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method(method, Some("Result".into())),
                args: crate::model::call_args(vec![recv, extra]),
                result_ty: ValueType::Int,
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    (graph, result)
}

pub(in crate::front::mir) fn option_ok_or_else_site(
    result_var: Variable,
) -> super::DiscCombinatorSite {
    let mut site = option_disc_site(super::DiscCombinator::OptionOkOrElse, result_var);
    site.result_owner = "core::result::Result".into();
    site.result_tag0_owner = "core::result::Result::Ok".into();
    site.result_tag1_owner = "core::result::Result::Err".into();
    site.result_payload0_ty = ValueType::Ref(None);
    site.result_payload1_ty = ValueType::Ref(None);
    site.call_result_ty = ValueType::Ref(None);
    site
}

#[derive(Clone, Copy, Debug)]
pub(in crate::front::mir) enum OptionRecvRepr {
    PointerNiche,
    ScalarNiche,
    Aggregate,
}

/// Pointer niche rtypes `is_` to `ptr_nonzero` (`rnone.py`
/// `rtype_is_None` / `CanBeNull.rtype_bool`). Scalar niche rtypes
/// `ne` 0 to `int_is_true` (`rint.py` `IntegerRepr.rtype_bool`).
/// Aggregate keeps `__discriminant` and `Some.__pos_0`.
pub(in crate::front::mir) fn assert_option_recv_repr(
    graph: &FunctionGraph,
    repr: OptionRecvRepr,
    method: &str,
) {
    let mut saw_disc = false;
    let mut saw_pos0_on_option = false;
    let mut saw_is = false;
    let mut saw_ne = false;
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    saw_disc = true;
                }
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref() == Some("core::option::Option::Some") =>
                {
                    saw_pos0_on_option = true;
                }
                OpKind::BinOp { op, .. } if op == "is_" => saw_is = true,
                OpKind::BinOp { op, .. } if op == "ne" => saw_ne = true,
                _ => {}
            }
        }
    }
    match repr {
        OptionRecvRepr::PointerNiche => {
            assert!(!saw_disc, "{method} pointer niche read __discriminant");
            assert!(
                !saw_pos0_on_option,
                "{method} pointer niche read Option Some.__pos_0"
            );
            assert!(
                saw_is,
                "{method} pointer niche missing rnone.py rtype_is_None null test"
            );
        }
        OptionRecvRepr::ScalarNiche => {
            assert!(!saw_disc, "{method} scalar niche read __discriminant");
            assert!(
                !saw_pos0_on_option,
                "{method} scalar niche read Option Some.__pos_0"
            );
            assert!(
                saw_ne,
                "{method} scalar niche missing rint.py IntegerRepr.rtype_bool test"
            );
        }
        OptionRecvRepr::Aggregate => {
            assert!(
                saw_disc,
                "{method} aggregate Option missing __discriminant read"
            );
            assert!(
                saw_pos0_on_option,
                "{method} aggregate Option missing Some.__pos_0 read"
            );
        }
    }
}

fn bool_exit_target(
    graph: &FunctionGraph,
    block: crate::model::BlockId,
    taken: bool,
) -> crate::model::BlockId {
    graph.blocks[block.0]
        .exits
        .iter()
        .find(|link| link.exitcase == Some(crate::model::ExitCase::Bool(taken)))
        .unwrap_or_else(|| panic!("missing Bool({taken}) exit on {block:?}"))
        .target
}

fn goto_result_var(graph: &FunctionGraph, block: crate::model::BlockId) -> Variable {
    let exits = &graph.blocks[block.0].exits;
    assert_eq!(exits.len(), 1, "arm {block:?} should goto the join");
    exits[0]
        .args
        .first()
        .and_then(LinkArg::as_variable)
        .cloned()
        .expect("filter arm result is a Variable")
}

fn option_variant_ctor(kind: &OpKind) -> Option<&str> {
    match kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { name, .. },
            ..
        } if name == "Some" || name == "None" => Some(name.as_str()),
        _ => None,
    }
}

fn is_null_mut_call(kind: &OpKind) -> bool {
    match kind {
        OpKind::Call { target, .. } => target.to_string() == "core::ptr::null_mut",
        _ => false,
    }
}

fn producer_of<'a>(block: &'a crate::model::Block, var: &Variable) -> Option<&'a SpaceOperation> {
    block
        .operations
        .iter()
        .find(|op| op.result.as_ref() == Some(var))
}

fn block_has_method(graph: &FunctionGraph, block: crate::model::BlockId, method: &str) -> bool {
    graph.blocks[block.0].operations.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == method
        )
    })
}

/// Drop / receiver-None arm of `rewire_option_filter_diamond`:
/// `emit_option_result_none` by the result representation.
fn assert_filter_none_value(
    graph: &FunctionGraph,
    arm: crate::model::BlockId,
    repr: OptionRecvRepr,
    which: &str,
) {
    let result = goto_result_var(graph, arm);
    let block = &graph.blocks[arm.0];
    let producer = producer_of(block, &result)
        .unwrap_or_else(|| panic!("filter {repr:?} {which} result is not produced in the arm"));
    match repr {
        OptionRecvRepr::PointerNiche => {
            assert!(
                block.operations.iter().any(|op| is_null_mut_call(&op.kind)),
                "filter pointer {which} missing rnone.py null_mut"
            );
            assert!(
                is_null_mut_call(&producer.kind)
                    || crate::model::cast_instance_root(&producer.kind).is_some(),
                "filter pointer {which} join is not the null (or its payload cast), got {:?}",
                producer.kind
            );
        }
        OptionRecvRepr::ScalarNiche => {
            assert!(
                matches!(producer.kind, OpKind::ConstUInt(0)),
                "filter scalar {which} must be rint.py IntegerRepr zero, got {:?}",
                producer.kind
            );
        }
        OptionRecvRepr::Aggregate => {
            assert_eq!(
                option_variant_ctor(&producer.kind),
                Some("None"),
                "filter aggregate {which} must be emit_option_result_none's None"
            );
            assert!(
                block.operations.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, base, .. }
                        if field.name == "__discriminant" && *base == result
                )),
                "filter aggregate {which} missing None __discriminant write"
            );
            assert!(
                !block.operations.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, base, .. }
                        if field.name == "__pos_0" && *base == result
                )),
                "filter aggregate {which} wrote Some.__pos_0 onto None"
            );
        }
    }
}

/// Keep arm of `rewire_option_filter_diamond`. A niche result is the
/// receiver word (`result_niche` / `result_scalar_niche`). An aggregate
/// rebuilds `Some` from `option_recv_payload` via `emit_option_variant`.
fn assert_filter_keep_arm(
    graph: &FunctionGraph,
    keep: crate::model::BlockId,
    repr: OptionRecvRepr,
) {
    let result = goto_result_var(graph, keep);
    let block = &graph.blocks[keep.0];
    match repr {
        OptionRecvRepr::PointerNiche | OptionRecvRepr::ScalarNiche => {
            assert!(
                block.operations.is_empty(),
                "filter {repr:?} keep re-wrapped the receiver word"
            );
            assert_eq!(
                Some(&result),
                block.inputargs.last(),
                "filter {repr:?} keep must pass the receiver word through to the join"
            );
        }
        OptionRecvRepr::Aggregate => {
            let producer = producer_of(block, &result)
                .expect("filter aggregate keep result produced in keep arm");
            assert_eq!(
                option_variant_ctor(&producer.kind),
                Some("Some"),
                "filter aggregate keep rebuilds Some via emit_option_variant"
            );
            let payload = block
                .inputargs
                .last()
                .expect("filter keep threads the payload");
            assert!(
                block.operations.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, value, .. }
                        if field.name == "__pos_0"
                            && value.as_variable() == Some(payload)
                )),
                "filter aggregate keep missing Some.__pos_0 write of the threaded payload"
            );
        }
    }
}

fn check_option_filter_repr(repr: OptionRecvRepr) {
    let method = "filter";
    let (mut graph, result) = build_two_arg_combinator(method);
    let mut site = option_disc_site(super::DiscCombinator::OptionFilter, result);
    match repr {
        OptionRecvRepr::PointerNiche => {
            site.recv_niche = true;
            site.result_niche = true;
            site.niche_null_cast = Some(("pyobject::PyObject".into(), ValueType::Ref(None)));
        }
        OptionRecvRepr::ScalarNiche => {
            site.scalar_niche = true;
            site.result_scalar_niche = true;
            site.recv_owner = "core::option::Option<NonZeroUsize>".into();
            site.result_owner = "core::option::Option<NonZeroUsize>".into();
            site.payload1_ty = ValueType::Unsigned;
            site.result_payload1_ty = ValueType::Unsigned;
        }
        OptionRecvRepr::Aggregate => {}
    }
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1, "{method} {repr:?} must rewrite");
    assert_eq!(
        count_method_calls(&graph, method),
        0,
        "{method} {repr:?} left a residual call"
    );
    assert_eq!(
        count_method_calls(&graph, "call_once"),
        1,
        "{method} {repr:?} predicate must run once"
    );
    let start = graph.startblock;
    assert_eq!(graph.blocks[start.0].exits.len(), 2);
    assert_option_recv_repr(&graph, repr, method);

    let recv = graph.blocks[start.0].operations[0]
        .result
        .clone()
        .expect("filter fixture receiver");
    match repr {
        OptionRecvRepr::PointerNiche => {
            assert!(
                graph.blocks[start.0].operations.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::BinOp { op, lhs, .. } if op == "is_" && *lhs == recv
                )),
                "filter pointer niche missing rnone.py rtype_is_None on the receiver"
            );
        }
        OptionRecvRepr::ScalarNiche => {
            let ne = graph.blocks[start.0]
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::BinOp { op, lhs, rhs, .. } if op == "ne" && *lhs == recv => {
                        Some(rhs.clone())
                    }
                    _ => None,
                });
            let rhs = ne.expect(
                "filter scalar niche missing rint.py IntegerRepr.rtype_bool ne on the receiver",
            );
            let zero = producer_of(&graph.blocks[start.0], &rhs)
                .expect("filter scalar niche tag zero is in the start block");
            assert!(
                matches!(zero.kind, OpKind::ConstUInt(0)),
                "filter scalar niche must compare against unsigned 0, got {:?}",
                zero.kind
            );
        }
        OptionRecvRepr::Aggregate => {
            assert!(
                graph.blocks[start.0].operations.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::FieldRead { field, base, .. }
                        if field.name == "__discriminant" && *base == recv
                )),
                "filter aggregate missing __discriminant read of the receiver"
            );
        }
    }

    let some = bool_exit_target(&graph, start, true);
    let none = bool_exit_target(&graph, start, false);
    assert_eq!(
        graph.blocks[some.0].exits.len(),
        2,
        "filter {repr:?} Some arm switches on the predicate"
    );
    assert!(
        block_has_method(&graph, some, "call_once"),
        "filter {repr:?} predicate must run on the Some/keep path"
    );
    assert!(
        !block_has_method(&graph, start, "call_once"),
        "filter {repr:?} predicate ran before the tag test"
    );
    let keep = bool_exit_target(&graph, some, true);
    let drop = bool_exit_target(&graph, some, false);
    assert!(
        !block_has_method(&graph, keep, "call_once"),
        "filter {repr:?} predicate ran inside the keep arm"
    );
    let join = graph.blocks[keep.0].exits[0].target;
    assert_eq!(graph.blocks[drop.0].exits[0].target, join);
    assert_eq!(graph.blocks[none.0].exits[0].target, join);
    assert_filter_none_value(&graph, none, repr, "receiver-None arm");
    assert_filter_none_value(&graph, drop, repr, "drop arm");
    assert_filter_keep_arm(&graph, keep, repr);
}

/// Three shapes: `filter` × pointer niche / scalar niche / aggregate
/// Option (`emit_option_recv_tag` / `option_recv_payload` /
/// `emit_option_result_none`).
#[test]
fn option_filter_repr_switch() {
    for repr in [
        OptionRecvRepr::PointerNiche,
        OptionRecvRepr::ScalarNiche,
        OptionRecvRepr::Aggregate,
    ] {
        check_option_filter_repr(repr);
    }
}

/// `Option::<i64>::Some` as the constructor shape
/// `aggregate_shape` resolves for a `Some(x)` aggregate.
pub(in crate::front::mir) fn some_ctor_shape() -> super::AggregateShape {
    super::AggregateShape::Ctor(super::AggregateCtor {
        target: CallTarget::synthetic_transparent_enum_variant_ctor(
            vec!["core".into(), "option".into(), "Option<i64>".into()],
            "Some".to_string(),
            1,
        ),
        result_owner: "core::option::Option<i64>::Some".into(),
        owner_id: None,
        owner_path: vec!["core".into(), "option".into(), "Option<i64>".into()],
        ctor_name: "Some".into(),
        fields: vec![super::AggregateField {
            name: "__pos_0".into(),
            void: false,
            narrow_root: None,
            window: false,
        }],
    })
}

/// Operations that build `Option::Some` (the variant constructor) and
/// that call the constructor function item.
pub(in crate::front::mir) fn some_ctor_ops(graph: &FunctionGraph) -> (usize, usize) {
    let ops = || graph.blocks.iter().flat_map(|block| &block.operations);
    let built = ops()
        .filter(|op| {
            matches!(&op.kind, OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, variant_tag: Some(1), .. },
                    ..
                } if name == "Some")
        })
        .count();
    let called = ops()
        .filter(|op| {
            matches!(&op.kind, OpKind::Call {
                    target: target @ CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("Some")
                    && crate::model::fn_const_segments(target).is_none())
        })
        .count();
    (built, called)
}

/// `Result::<i64, i64>::Ok` as the constructor shape.
pub(in crate::front::mir) fn ok_ctor_shape() -> super::AggregateShape {
    let owner_path: Vec<String> = vec!["core".into(), "result".into(), "Result<i64,i64>".into()];
    super::AggregateShape::Ctor(super::AggregateCtor {
        target: CallTarget::synthetic_transparent_enum_variant_ctor(
            owner_path.clone(),
            "Ok".to_string(),
            0,
        ),
        result_owner: "core::result::Result<i64,i64>::Ok".into(),
        owner_id: None,
        owner_path,
        ctor_name: "Ok".into(),
        fields: vec![super::AggregateField {
            name: "__pos_0".into(),
            void: false,
            narrow_root: None,
            window: false,
        }],
    })
}

pub(in crate::front::mir) fn push_fn_const(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    path: &[&str],
) -> Variable {
    let mut segments = vec![crate::model::FN_CONST_HEAD.to_string()];
    segments.extend(path.iter().map(|s| s.to_string()));
    graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Int,
            },
            true,
        )
        .unwrap()
}

#[test]
fn desugar_mix_result_question_mark_lowers_to_result_switch() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc");
    let llbc = Llbc::load(path).expect("load corpus");
    let graph = super::lower_function(&llbc, "desugar_mix").expect("lowering");
    let residual_branch = count_method_calls(&graph, "branch");
    let residual_from_residual = count_method_calls(&graph, "from_residual");
    assert_eq!(
        residual_branch, 0,
        "desugar_mix: residual Try::branch after Result `?` lowering"
    );
    assert_eq!(
        residual_from_residual, 0,
        "desugar_mix: residual from_residual after Result `?` lowering"
    );
    let result_disc = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref().is_some_and(|owner| owner.contains("Result"))
            )
        })
        .count();
    assert!(
        result_disc >= 1,
        "desugar_mix: expected a Result discriminant switch"
    );
}

#[test]
fn opaque_bitflags_empty_global_read_stays_residual() {
    let path = ["bitflags", "traits", "<Impl>", "EMPTY"];
    let mut opaque_init = init_fun(1, &path, serde_json::json!("Opaque"));
    opaque_init["item_meta"]["is_local"] = serde_json::json!(false);
    let artifact = const_artifact(
        "fixture",
        serde_json::json!([reader_fun(&["fixture", "read_empty"], 1), opaque_init]),
        serde_json::json!([
            null,
            named_const_global(1, &path, "", false, "NamedConst", 1)
        ]),
    );
    let graph = super::lower_function(&artifact, "read_empty").expect("lower");
    let ints: Vec<i64> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op.kind {
            OpKind::ConstInt(n) => Some(n),
            _ => None,
        })
        .collect();
    assert!(
        ints.is_empty(),
        "opaque Bits::EMPTY must not leaf-fold to 0, got {ints:?}"
    );
    assert_eq!(
        nullary_calls_ending(&graph, "EMPTY"),
        1,
        "the Opaque Bits::EMPTY read stays a nullary call: {graph:?}"
    );
}

/// `Gc` from a majit-gc `GcType` impl, from field 0 of that type
/// (and that field's own field 0), from a struct whose field 0 is
/// already `Gc` (`Struct._note_inlined_into`), and `Raw` for a type
/// with no impl, for a struct headed by a raw field 0, and for an
/// impl of another crate's trait that is also named `GcType`.
#[test]
fn declared_gckind_follows_gctype_impl_and_field0_parent() {
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |segs: &[&str]| {
        serde_json::json!({
            "name": segs.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let adt = |id: u64| {
        serde_json::json!({
            "Value": [id, {"Adt": {"id": id, "generics": {
                "regions": [], "types": [], "const_generics": [], "trait_refs": []
            }}}]
        })
    };
    let struct_decl = |def_id: u64, segs: &[&str], field0: Option<u64>| {
        let fields: Vec<serde_json::Value> = field0
            .into_iter()
            .map(|id| serde_json::json!({"name": "head", "ty": adt(id)}))
            .collect();
        serde_json::json!({
            "def_id": def_id,
            "item_meta": meta(segs),
            "kind": {"Struct": fields}
        })
    };
    let trait_decl =
        |def_id: u64, segs: &[&str]| serde_json::json!({"def_id": def_id, "item_meta": meta(segs)});
    let impl_row = |trait_id: u64, self_id: u64| {
        serde_json::json!({
            "impl_trait": {"id": trait_id, "generics": {"types": [adt(self_id)]}}
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                struct_decl(0, &["fixture", "ByImpl"], Some(1)),
                struct_decl(1, &["fixture", "Parent"], Some(2)),
                struct_decl(2, &["fixture", "Inner"], None),
                struct_decl(3, &["fixture", "RawOne"], None),
                struct_decl(4, &["fixture", "Foreign"], None),
                struct_decl(5, &["fixture", "HeadedByImpl"], Some(0)),
                struct_decl(6, &["fixture", "HeadedByInner"], Some(2)),
                struct_decl(7, &["fixture", "HeadedByRaw"], Some(3)),
            ],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [
                trait_decl(0, &["majit_gc", "header", "GcType"]),
                trait_decl(1, &["other_crate", "GcType"]),
            ],
            "trait_impls": [impl_row(0, 0), impl_row(1, 4)]
        }
    });
    let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
        .expect("fixture Llbc parses");
    let kinds = super::declared_gckind_by_name(std::slice::from_ref(&llbc));
    assert_eq!(kinds.get("ByImpl"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("Parent"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("Inner"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("RawOne"), Some(&GcKind::Raw));
    assert_eq!(kinds.get("Foreign"), Some(&GcKind::Raw));
    assert_eq!(kinds.get("HeadedByImpl"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("HeadedByInner"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("HeadedByRaw"), Some(&GcKind::Raw));
}

/// Layout gckind follows a `GcType` impl, including a generic template
/// seed for an instantiation. `struct_id_is_declared_gc` is the one
/// membership test (`history.py` `getkind` of `Ptr(TO)` follows
/// `TO._gckind`).
#[test]
fn declared_gc_layout_gckind_follows_gctype_or_template() {
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |segs: &[&str]| {
        serde_json::json!({
            "name": segs.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let adt = |id: u64| {
        serde_json::json!({
            "Adt": {"id": id, "generics": {
                "regions": [], "types": [], "const_generics": [], "trait_refs": []
            }}
        })
    };
    let g_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "G"]),
        "kind": {"Struct": []}
    });
    let gg_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "GG"]),
        "kind": {"Struct": [{"name": "head", "ty": {"TypeVar": {"Bound": [0, 0]}}}]}
    });
    let n_decl = serde_json::json!({
        "def_id": 2,
        "item_meta": meta(&["fixture", "N"]),
        "kind": {"Struct": []}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [g_decl, gg_decl, n_decl],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["majit_gc", "header", "GcType"])
            }],
            "trait_impls": [
                {"impl_trait": {"id": 0, "generics": {"types": [{"Value": [0, adt(0)]}]}}},
                {"impl_trait": {"id": 0, "generics": {"types": [{"Value": [1, adt(1)]}]}}}
            ]
        }
    });
    let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
        .expect("fixture Llbc parses");
    let kinds = super::declared_gckind_by_name(std::slice::from_ref(&llbc));
    assert_eq!(kinds.get("G"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("N"), Some(&GcKind::Raw));
    assert_eq!(kinds.get("GG"), Some(&GcKind::Gc));
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let gg_sid = super::struct_id_for_type_decl(llbc.type_by_id(1).expect("GG decl"));
    let _registry =
        crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
            ("GG".to_string(), Some(gg_sid)),
            ("fixture::GG".to_string(), Some(gg_sid)),
        ]));
    let inst = majit_ir::descr::struct_id_for_name("GG<i64>").expect("instantiated GG<i64>");
    assert_ne!(inst, gg_sid, "struct_id_for_name instantiates <i64>");
    assert!(
        super::struct_id_is_declared_gc(&gc, inst, "GG<i64>"),
        "GG<i64> is declared gc via the template seed"
    );
}

fn interior_span() -> serde_json::Value {
    serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    })
}

pub(in crate::front::mir) fn interior_meta(path: &[&str], is_local: bool) -> serde_json::Value {
    serde_json::json!({
        "name": path.iter().map(|segment| {
            if *segment == "<Impl>" {
                serde_json::json!({"Impl": {"kind": "InherentImplBlock"}})
            } else {
                serde_json::json!({"Ident": [segment, 0]})
            }
        }).collect::<Vec<_>>(),
        "span": interior_span(),
        "source_text": null,
        "attr_info": {
            "attributes": [],
            "inline": null,
            "rename": null,
            "public": true
        },
        "is_local": is_local
    })
}

pub(in crate::front::mir) fn interior_i64() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})
}

pub(in crate::front::mir) fn interior_usize() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
}

fn interior_u32() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}})
}

fn interior_adt(id: u64) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": id,
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

pub(in crate::front::mir) fn interior_raw_mut(pointee: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"RawPtr": [pointee, "Mut"]})
}

fn interior_shared(inner: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Ref": [null, inner, "Shared"]})
}

fn interior_layout(size: u64, offsets: &[u64], transparent: bool) -> serde_json::Value {
    serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": size,
            "align": 8,
            "variant_layouts": [{"field_offsets": offsets}],
            "repr": {"transparent": transparent}
        }
    }])
}

fn interior_struct(
    def_id: u64,
    path: &[&str],
    fields: &[(&str, serde_json::Value)],
    size: u64,
    offsets: &[u64],
    transparent: bool,
) -> serde_json::Value {
    let fields: Vec<_> = fields
        .iter()
        .map(|(name, ty)| serde_json::json!({"name": name, "ty": ty, "attr_info": null}))
        .collect();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": interior_meta(path, true),
        "kind": {"Struct": fields},
        "layout": interior_layout(size, offsets, transparent)
    })
}

pub(in crate::front::mir) fn interior_opaque(
    def_id: u64,
    path: &[&str],
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": interior_meta(path, false),
        "signature": {"is_unsafe": true, "inputs": inputs, "output": output},
        "body": "Opaque"
    })
}

pub(in crate::front::mir) fn interior_local(
    index: u64,
    name: Option<&str>,
    ty: &serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({"index": index, "name": name, "span": interior_span(), "ty": ty})
}

fn interior_place(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"kind": {"Local": index}, "ty": ty})
}

fn interior_deref(
    index: u64,
    ptr_ty: &serde_json::Value,
    pointee: &serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "kind": {"Projection": [interior_place(index, ptr_ty), "Deref"]},
        "ty": pointee
    })
}

fn interior_field(
    index: u64,
    ptr_ty: &serde_json::Value,
    pointee: &serde_json::Value,
    field_ty: &serde_json::Value,
    field_idx: u64,
) -> serde_json::Value {
    serde_json::json!({
        "kind": {
            "Projection": [
                interior_deref(index, ptr_ty, pointee),
                {"Field": [null, field_idx]}
            ]
        },
        "ty": field_ty
    })
}

fn interior_stmt(kind: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"kind": kind, "span": interior_span()})
}

fn interior_copy(place: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Copy": place})
}

fn interior_use(place: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Use": [interior_copy(place), "No"]})
}

fn interior_assign(dest: serde_json::Value, rvalue: serde_json::Value) -> serde_json::Value {
    interior_stmt(serde_json::json!({"Assign": [dest, rvalue]}))
}

fn interior_const_i64(text: &str, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Const": [{"Integer": {"Signed": ["I64", text]}}, ty]})
}

fn interior_const_usize(text: &str, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", text]}}, ty]})
}

fn interior_call(
    fun: u64,
    args: Vec<serde_json::Value>,
    dest: serde_json::Value,
    target: u64,
    unwind: u64,
) -> serde_json::Value {
    serde_json::json!({
        "span": interior_span(),
        "kind": {
            "Call": {
                "call": {
                    "func": {
                        "Regular": {
                            "kind": {"Fun": fun},
                            "generics": {
                                "regions": [],
                                "types": [],
                                "const_generics": [],
                                "trait_refs": []
                            }
                        }
                    },
                    "args": args,
                    "dest": dest
                },
                "target": target,
                "on_unwind": unwind
            }
        }
    })
}

pub(in crate::front::mir) fn interior_bb(
    stmts: Vec<serde_json::Value>,
    term: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({"statements": stmts, "terminator": term})
}

pub(in crate::front::mir) fn interior_return() -> serde_json::Value {
    serde_json::json!({"span": interior_span(), "kind": "Return"})
}

fn interior_unwind() -> serde_json::Value {
    serde_json::json!({"span": interior_span(), "kind": "UnwindResume"})
}

pub(in crate::front::mir) fn interior_caller(
    name: &str,
    arg_count: u64,
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
    locals: Vec<serde_json::Value>,
    body: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": 0,
        "item_meta": interior_meta(&["fixture", name], true),
        "signature": {"is_unsafe": true, "inputs": inputs, "output": output},
        "body": {
            "Unstructured": {
                "span": interior_span(),
                "locals": {"arg_count": arg_count, "locals": locals},
                "body": body
            }
        }
    })
}

fn interior_point() -> serde_json::Value {
    let i64_ty = interior_i64();
    interior_struct(
        0,
        &["fixture", "Point"],
        &[("x", i64_ty.clone()), ("y", i64_ty)],
        16,
        &[0, 8],
        false,
    )
}

fn lower_interior(llbc: &Llbc, name: &str) -> FunctionGraph {
    super::lower_function(llbc, name).unwrap_or_else(|err| panic!("lower {name}: {err:?}"))
}

/// `*self = fresh` through `&mut Self` of type `types[0]` (`def_id` 0).
fn lower_deref_store_mut_self(
    types: Vec<serde_json::Value>,
) -> Result<FunctionGraph, super::LowerError> {
    let s_ty = interior_adt(0);
    let mut_self = serde_json::json!({"Ref": [null, s_ty.clone(), "Mut"]});
    let unit = serde_json::json!({
        "Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}
    });
    let caller = interior_caller(
        "f",
        2,
        vec![mut_self.clone(), s_ty.clone()],
        unit.clone(),
        vec![
            interior_local(0, None, &unit),
            interior_local(1, Some("self"), &mut_self),
            interior_local(2, Some("fresh"), &s_ty),
        ],
        vec![interior_bb(
            vec![interior_assign(
                interior_deref(1, &mut_self, &s_ty),
                serde_json::json!({
                    "Use": [{"Move": interior_place(2, &s_ty)}, "No"]
                }),
            )],
            interior_return(),
        )],
    );
    let llbc = llbc_with_types("fixture", types, vec![caller]);
    super::lower_function(&llbc, "f")
}

fn field_read_write_names<'a>(ops: &'a [&SpaceOperation]) -> (Vec<&'a str>, Vec<&'a str>) {
    let writes = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } => Some(field.name.as_str()),
            _ => None,
        })
        .collect();
    let reads = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => Some(field.name.as_str()),
            _ => None,
        })
        .collect();
    (writes, reads)
}

fn interior_read_block(graph: &FunctionGraph) -> &crate::model::Block {
    graph
        .blocks
        .iter()
        .find(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::InteriorFieldRead { .. }))
        })
        .expect("InteriorFieldRead block")
}

fn assert_no_interior(ops: &[&SpaceOperation]) {
    assert!(
        ops.iter().all(|op| !matches!(
            op.kind,
            OpKind::InteriorFieldRead { .. } | OpKind::InteriorFieldWrite { .. }
        )),
        "struct-field ptr::add must not fire; ops={ops:?}"
    );
}

fn assert_direct_ptradd_not_residual_add(ops: &[&SpaceOperation]) {
    assert!(
        ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "declined struct offset must stay direct_ptradd; ops={ops:?}"
    );
    assert!(
        !call_leafs(ops)
            .iter()
            .any(|leaf| leaf == "add" || leaf == "sub"),
        "declined struct offset must not residualize; ops={ops:?}"
    );
}

/// `(*p.add(i)).y` is `getinteriorfield` (`rewrite_op_getinteriorfield`).
/// The base is the array object, which is the add's pointer argument.
#[test]
fn interior_field_read_of_ptr_add() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "read_point",
        2,
        vec![ptr.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(3, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &i64_ty),
                    interior_use(interior_field(3, &ptr, &adt, &i64_ty, 1)),
                )],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, add]);
    let graph = lower_interior(&llbc, "read_point");
    let ops = graph_ops(&graph);
    let reads: Vec<_> = ops
        .iter()
        .filter(|op| matches!(op.kind, OpKind::InteriorFieldRead { .. }))
        .copied()
        .collect();
    assert_eq!(reads.len(), 1, "ops={ops:?}");
    let block = interior_read_block(&graph);
    assert!(
        block.inputargs.len() >= 2,
        "header and index stay live; inputargs={:?}",
        block.inputargs
    );
    let OpKind::InteriorFieldRead {
        base,
        index,
        field,
        item_ty,
        array_type_id,
    } = &reads[0].kind
    else {
        unreachable!()
    };
    assert_eq!(field.name, "y");
    assert_eq!(field.owner_root.as_deref(), Some("Point"));
    assert_eq!(field.base_is_deref, Some(true));
    assert_eq!(item_ty, &ValueType::Int);
    assert_eq!(array_type_id.as_deref(), Some("GcArray<Point>"));
    assert_eq!(base, &block.inputargs[0]);
    assert_eq!(index, &block.inputargs[1]);
    assert!(
        ops.iter()
            .all(|op| !matches!(op.kind, OpKind::FieldRead { .. })),
        "the array object is the pointer argument, not a FieldRead; ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| leaf == "add"),
        "ops={ops:?}"
    );
}

/// `(*p.add(i)).y = v` is `setinteriorfield`. Field index 1 keeps the
/// store off the offset-0 aggregate move.
#[test]
fn interior_field_write_of_ptr_add() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "write_point",
        3,
        vec![ptr.clone(), usize_ty.clone(), i64_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("value"), &i64_ty),
            interior_local(4, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(4, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![
                    interior_assign(
                        interior_field(4, &ptr, &adt, &i64_ty, 1),
                        interior_use(interior_place(3, &i64_ty)),
                    ),
                    interior_assign(
                        interior_place(0, &i64_ty),
                        interior_use(interior_place(3, &i64_ty)),
                    ),
                ],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, add]);
    let graph = lower_interior(&llbc, "write_point");
    let ops = graph_ops(&graph);
    let writes: Vec<_> = ops
        .iter()
        .filter(|op| matches!(op.kind, OpKind::InteriorFieldWrite { .. }))
        .copied()
        .collect();
    assert_eq!(writes.len(), 1, "ops={ops:?}");
    let block = graph
        .blocks
        .iter()
        .find(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::InteriorFieldWrite { .. }))
        })
        .expect("InteriorFieldWrite block");
    assert!(
        block.inputargs.len() >= 3,
        "inputargs={:?}",
        block.inputargs
    );
    let OpKind::InteriorFieldWrite {
        base,
        index,
        value,
        field,
        item_ty,
        array_type_id,
    } = &writes[0].kind
    else {
        unreachable!()
    };
    assert_eq!(field.name, "y");
    assert_eq!(field.base_is_deref, Some(true));
    assert_eq!(item_ty, &ValueType::Int);
    assert_eq!(array_type_id.as_deref(), Some("GcArray<Point>"));
    assert_eq!(base, &block.inputargs[0]);
    assert_eq!(index, &block.inputargs[1]);
    assert_eq!(value, &block.inputargs[2]);
    assert!(
        !ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| leaf == "add"),
        "ops={ops:?}"
    );
}

/// `*self = fresh` through `&mut Self` is the field-wise structcopy
/// (`rffi.py` `_get_structcopy_fn`) even when a field is
/// `Atomic<usize>` (`tyref_atomic_inner_value_type`).
#[test]
fn deref_store_of_mut_self_emits_field_writes() {
    let s_ty = interior_adt(0);
    let t_ty = interior_adt(1);
    let usize_ty = interior_usize();
    let p_ty = interior_raw_mut(t_ty);
    let atomic_ty = serde_json::json!({
        "Adt": {
            "id": 2,
            "generics": {
                "regions": [],
                "types": [usize_ty],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let mut_self = serde_json::json!({"Ref": [null, s_ty.clone(), "Mut"]});
    let unit = serde_json::json!({
        "Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}
    });
    let s_decl = interior_struct(
        0,
        &["fixture", "S"],
        &[("p", p_ty), ("n", atomic_ty)],
        16,
        &[0, 8],
        false,
    );
    let t_decl = interior_struct(1, &["fixture", "T"], &[], 0, &[], false);
    // Charon emits `Atomic<T>` as Opaque with no layout; the stored
    // word is the type argument (`tyref_atomic_arg`).
    let atomic_decl = serde_json::json!({
        "def_id": 2,
        "item_meta": interior_meta(&["core", "sync", "atomic", "Atomic"], false),
        "kind": "Opaque",
        "layout": null
    });
    let caller = interior_caller(
        "f",
        2,
        vec![mut_self.clone(), s_ty.clone()],
        unit.clone(),
        vec![
            interior_local(0, None, &unit),
            interior_local(1, Some("self"), &mut_self),
            interior_local(2, Some("fresh"), &s_ty),
        ],
        vec![interior_bb(
            vec![interior_assign(
                interior_deref(1, &mut_self, &s_ty),
                serde_json::json!({
                    "Use": [{"Move": interior_place(2, &s_ty)}, "No"]
                }),
            )],
            interior_return(),
        )],
    );
    let llbc = llbc_with_types("fixture", vec![s_decl, t_decl, atomic_decl], vec![caller]);
    let graph = lower_interior(&llbc, "f");
    let ops = graph_ops(&graph);
    let write_names: Vec<&str> = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } => Some(field.name.as_str()),
            _ => None,
        })
        .collect();
    let read_names: Vec<&str> = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => Some(field.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(write_names, ["p", "n"], "ops={ops:?}");
    assert_eq!(read_names, ["p", "n"], "ops={ops:?}");
    let atomic_write_ty = ops.iter().find_map(|op| match &op.kind {
        OpKind::FieldWrite { field, ty, .. } if field.name == "n" => Some(ty),
        _ => None,
    });
    assert_eq!(atomic_write_ty, Some(&ValueType::Unsigned), "ops={ops:?}");
}

/// An inline nested-struct field is a ContainerType. `rffi.py`
/// `_get_structcopy_fn.copyfn` would recurse into its leaves;
/// `move_plan` has no recursive plan yet, so the store declines.
#[test]
fn deref_store_of_nested_struct_field_has_no_move_plan() {
    let s_ty = interior_adt(0);
    let inner_ty = interior_adt(1);
    let u32_ty = interior_u32();
    let usize_ty = interior_usize();
    let s_decl = interior_struct(
        0,
        &["fixture", "S"],
        &[("inner", inner_ty), ("n", usize_ty)],
        16,
        &[0, 8],
        false,
    );
    let inner_decl = interior_struct(
        1,
        &["fixture", "Inner"],
        &[("a", u32_ty.clone()), ("b", u32_ty)],
        8,
        &[0, 4],
        false,
    );
    let err = lower_deref_store_mut_self(vec![s_decl, inner_decl])
        .expect_err("nested container field must decline the move plan");
    let msg = err.to_string();
    assert!(
        msg.contains("deref store is not a field, array element, or raw word"),
        "s_ty={s_ty:?} err={msg}"
    );
}

/// `Atomic<u32>` is Opaque; the stored word is `T` (`tyref_atomic_arg`).
/// The Struct arm still builds a field-wise move plan.
#[test]
fn deref_store_of_atomic_u32_field_emits_field_writes() {
    let u32_ty = interior_u32();
    let usize_ty = interior_usize();
    let atomic_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [u32_ty],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let s_decl = interior_struct(
        0,
        &["fixture", "S"],
        &[("n", atomic_ty), ("extra", usize_ty)],
        16,
        &[0, 8],
        false,
    );
    let atomic_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": interior_meta(&["core", "sync", "atomic", "Atomic"], false),
        "kind": "Opaque",
        "layout": null
    });
    let graph = lower_deref_store_mut_self(vec![s_decl, atomic_decl])
        .unwrap_or_else(|err| panic!("Atomic<u32> field must keep a move plan: {err:?}"));
    let ops = graph_ops(&graph);
    let (write_names, read_names) = field_read_write_names(&ops);
    assert_eq!(write_names, ["n", "extra"], "ops={ops:?}");
    assert_eq!(read_names, ["n", "extra"], "ops={ops:?}");
    let atomic_write_ty = ops.iter().find_map(|op| match &op.kind {
        OpKind::FieldWrite { field, ty, .. } if field.name == "n" => Some(ty),
        _ => None,
    });
    assert_eq!(atomic_write_ty, Some(&ValueType::Unsigned), "ops={ops:?}");
}

/// A fieldless (C-like) enum field is its scalar tag, so the Struct
/// arm still builds a field-wise move plan.
#[test]
fn deref_store_of_fieldless_enum_field_emits_field_writes() {
    let e_ty = interior_adt(1);
    let usize_ty = interior_usize();
    let s_decl = interior_struct(
        0,
        &["fixture", "S"],
        &[("tag", e_ty), ("n", usize_ty)],
        16,
        &[0, 8],
        false,
    );
    let e_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": interior_meta(&["fixture", "Kind"], true),
        "kind": {"Enum": [
            {"name": "A", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}},
            {"name": "B", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}}
        ]},
        "layout": [{
            "key": "fixture-target",
            "value": {
                "size": 4,
                "align": 4,
                "discriminator": {"Branch": {"offset": 0, "int_ty": {"Unsigned": "U32"}}},
                "variant_layouts": [
                    {"field_offsets": []},
                    {"field_offsets": []}
                ],
                "repr": {"transparent": false}
            }
        }]
    });
    let graph = lower_deref_store_mut_self(vec![s_decl, e_decl])
        .unwrap_or_else(|err| panic!("C-like enum field must keep a move plan: {err:?}"));
    let ops = graph_ops(&graph);
    let (write_names, read_names) = field_read_write_names(&ops);
    assert_eq!(write_names, ["tag", "n"], "ops={ops:?}");
    assert_eq!(read_names, ["tag", "n"], "ops={ops:?}");
}

/// A signed field constant stays a pooled int. `field_write_inline_const`
/// hands `LinkArg::Const` to the write, and `link_arg_as_alias_op`
/// materialises `ConstInt` because `InteriorFieldWrite.value` is a
/// Variable. The return assign is a second `ConstInt`.
#[test]
fn interior_field_write_of_const_int() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "write_point_const",
        2,
        vec![ptr.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(3, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![
                    interior_assign(
                        interior_field(3, &ptr, &adt, &i64_ty, 1),
                        serde_json::json!({"Use": [interior_const_i64("7", &i64_ty), "No"]}),
                    ),
                    interior_assign(
                        interior_place(0, &i64_ty),
                        serde_json::json!({"Use": [interior_const_i64("7", &i64_ty), "No"]}),
                    ),
                ],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, add]);
    let graph = lower_interior(&llbc, "write_point_const");
    let ops = graph_ops(&graph);
    let consts: Vec<&Variable> = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::ConstInt(7) => op.result.as_ref(),
            _ => None,
        })
        .collect();
    assert_eq!(consts.len(), 2, "ops={ops:?}");
    let OpKind::InteriorFieldWrite {
        value,
        field,
        array_type_id,
        ..
    } = ops
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::InteriorFieldWrite { .. } => Some(&op.kind),
            _ => None,
        })
        .expect("InteriorFieldWrite")
    else {
        unreachable!()
    };
    assert_eq!(field.name, "y");
    assert_eq!(array_type_id.as_deref(), Some("GcArray<Point>"));
    assert_eq!(value, consts[0]);
}

/// Two value reads of one `ptr::add` are two interior reads. The
/// census accepts `fields >= 1`. Both reads feed the returned sum, so
/// neither is a dead store.
#[test]
fn interior_field_two_reads_of_one_add() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "read_point_both",
        2,
        vec![ptr.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("slot"), &ptr),
            interior_local(4, Some("x"), &i64_ty),
            interior_local(5, Some("y"), &i64_ty),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(3, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![
                    interior_assign(
                        interior_place(4, &i64_ty),
                        interior_use(interior_field(3, &ptr, &adt, &i64_ty, 0)),
                    ),
                    interior_assign(
                        interior_place(5, &i64_ty),
                        interior_use(interior_field(3, &ptr, &adt, &i64_ty, 1)),
                    ),
                    interior_assign(
                        interior_place(0, &i64_ty),
                        serde_json::json!({"BinaryOp": [
                            "Add",
                            interior_copy(interior_place(4, &i64_ty)),
                            interior_copy(interior_place(5, &i64_ty))
                        ]}),
                    ),
                ],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, add]);
    let graph = lower_interior(&llbc, "read_point_both");
    let block = interior_read_block(&graph);
    let names: Vec<&str> = block
        .operations
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::InteriorFieldRead {
                base,
                index,
                field,
                array_type_id,
                ..
            } => {
                assert_eq!(base, &block.inputargs[0]);
                assert_eq!(index, &block.inputargs[1]);
                assert_eq!(array_type_id.as_deref(), Some("GcArray<Point>"));
                Some(field.name.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(names, ["x", "y"]);
}

/// `entry_ptr(&dict).add(i)` recovers the `entries` header.
/// `getinteriorfield` adds `array_items_base`, so the interior
/// base is that `FieldRead`, not the item pointer the call returns.
/// `prune_dead_phis` drops an unread item-pointer phi; the lock is that
/// the interior base is the field-read result and not an inputarg.
#[test]
fn interior_field_entry_ptr_reads_entries_header() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let dict = interior_adt(0);
    let entry = interior_adt(1);
    let dict_ref = interior_shared(dict.clone());
    let entry_ptr_ty = interior_raw_mut(entry.clone());
    let dict_decl = interior_struct(
        0,
        &["fixture", "Dict"],
        &[("pad", i64_ty.clone()), ("entries", entry_ptr_ty.clone())],
        16,
        &[0, 8],
        false,
    );
    let entry_decl = interior_struct(
        1,
        &["fixture", "Entry"],
        &[("pad", i64_ty.clone()), ("value", i64_ty.clone())],
        16,
        &[0, 8],
        false,
    );
    let caller = interior_caller(
        "read_entry_at",
        2,
        vec![dict_ref.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("d"), &dict_ref),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("item"), &entry_ptr_ty),
            interior_local(4, Some("slot"), &entry_ptr_ty),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![interior_copy(interior_place(1, &dict_ref))],
                    interior_place(3, &entry_ptr_ty),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![],
                interior_call(
                    2,
                    vec![
                        interior_copy(interior_place(3, &entry_ptr_ty)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(4, &entry_ptr_ty),
                    3,
                    1,
                ),
            ),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &i64_ty),
                    interior_use(interior_field(4, &entry_ptr_ty, &entry, &i64_ty, 1)),
                )],
                interior_return(),
            ),
        ],
    );
    let entry_ptr_fn = interior_opaque(
        1,
        &["pyre_object", "rordereddict", "<Impl>", "entry_ptr"],
        vec![dict_ref],
        entry_ptr_ty.clone(),
    );
    let add = interior_opaque(
        2,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![entry_ptr_ty.clone(), usize_ty],
        entry_ptr_ty,
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![dict_decl, entry_decl],
        vec![caller, entry_ptr_fn, add],
    );
    let graph = lower_interior(&llbc, "read_entry_at");
    let ops = graph_ops(&graph);
    let block = interior_read_block(&graph);
    let field_ops: Vec<_> = block
        .operations
        .iter()
        .filter(|op| matches!(op.kind, OpKind::FieldRead { .. }))
        .collect();
    assert_eq!(field_ops.len(), 1, "ops={ops:?}");
    let OpKind::FieldRead {
        base: field_base,
        field: entries,
        ..
    } = &field_ops[0].kind
    else {
        unreachable!()
    };
    assert_eq!(entries.name, "entries");
    assert_eq!(entries.base_is_deref, Some(true));
    assert!(
        block.inputargs.len() >= 2,
        "inputargs={:?}",
        block.inputargs
    );
    assert_eq!(field_base, &block.inputargs[0]);
    let OpKind::InteriorFieldRead {
        base,
        index,
        field,
        array_type_id,
        ..
    } = block
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::InteriorFieldRead { .. } => Some(&op.kind),
            _ => None,
        })
        .expect("InteriorFieldRead")
    else {
        unreachable!()
    };
    assert_eq!(field.name, "value");
    assert_eq!(array_type_id.as_deref(), Some("GcArray<Entry>"));
    assert_eq!(index, &block.inputargs[1]);
    assert_eq!(base, field_ops[0].result.as_ref().unwrap());
    assert!(
        block.inputargs.iter().all(|arg| arg != base),
        "interior base must be the entries FieldRead, not the item pointer; inputargs={:?}",
        block.inputargs
    );
    assert!(
        call_leafs(&ops).iter().any(|leaf| leaf == "entry_ptr"),
        "ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| leaf == "add"),
        "ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "ops={ops:?}"
    );
}

/// `entries_item_ptr(header).add(i)` uses `header` itself. The call
/// stays residual; its result is not given a second length word.
#[test]
fn interior_field_entries_item_ptr_reuses_header_arg() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let entry = interior_adt(0);
    let entry_ptr_ty = interior_raw_mut(entry.clone());
    let header_ty =
        interior_raw_mut(serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}}));
    let entry_decl = interior_struct(
        0,
        &["fixture", "Entry"],
        &[("pad", i64_ty.clone()), ("value", i64_ty.clone())],
        16,
        &[0, 8],
        false,
    );
    let caller = interior_caller(
        "read_entries_item",
        2,
        vec![header_ty.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("header"), &header_ty),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("item"), &entry_ptr_ty),
            interior_local(4, Some("slot"), &entry_ptr_ty),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![interior_copy(interior_place(1, &header_ty))],
                    interior_place(3, &entry_ptr_ty),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![],
                interior_call(
                    2,
                    vec![
                        interior_copy(interior_place(3, &entry_ptr_ty)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(4, &entry_ptr_ty),
                    3,
                    1,
                ),
            ),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &i64_ty),
                    interior_use(interior_field(4, &entry_ptr_ty, &entry, &i64_ty, 1)),
                )],
                interior_return(),
            ),
        ],
    );
    let items = interior_opaque(
        1,
        &["pyre_object", "rordereddict_entries", "entries_item_ptr"],
        vec![header_ty.clone()],
        entry_ptr_ty.clone(),
    );
    let add = interior_opaque(
        2,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![entry_ptr_ty.clone(), usize_ty],
        entry_ptr_ty,
    );
    let llbc = llbc_with_types("fixture", vec![entry_decl], vec![caller, items, add]);
    let graph = lower_interior(&llbc, "read_entries_item");
    let ops = graph_ops(&graph);
    let block = interior_read_block(&graph);
    assert!(
        block.inputargs.len() >= 2,
        "inputargs={:?}",
        block.inputargs
    );
    let OpKind::InteriorFieldRead {
        base,
        index,
        field,
        array_type_id,
        ..
    } = block
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::InteriorFieldRead { .. } => Some(&op.kind),
            _ => None,
        })
        .expect("InteriorFieldRead")
    else {
        unreachable!()
    };
    assert_eq!(field.name, "value");
    assert_eq!(array_type_id.as_deref(), Some("GcArray<Entry>"));
    assert_eq!(base, &block.inputargs[0]);
    assert_eq!(index, &block.inputargs[1]);
    assert!(
        ops.iter()
            .all(|op| !matches!(op.kind, OpKind::FieldRead { .. })),
        "entries_item_ptr's argument is already the header; ops={ops:?}"
    );
    assert!(
        call_leafs(&ops)
            .iter()
            .any(|leaf| leaf == "entries_item_ptr"),
        "ops={ops:?}"
    );
    assert!(
        !call_leafs(&ops).iter().any(|leaf| leaf == "add"),
        "ops={ops:?}"
    );
    assert!(
        !ops.iter().any(|op| is_lltype_direct_ptradd(op)),
        "ops={ops:?}"
    );
}

#[test]
fn interior_field_ptr_sub_stays_direct_ptradd() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "sub_point",
        2,
        vec![ptr.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(3, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &i64_ty),
                    interior_use(interior_field(3, &ptr, &adt, &i64_ty, 1)),
                )],
                interior_return(),
            ),
        ],
    );
    let sub = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "sub"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, sub]);
    let graph = lower_interior(&llbc, "sub_point");
    let ops = graph_ops(&graph);
    assert_no_interior(&ops);
    assert_direct_ptradd_not_residual_add(&ops);
}

#[test]
fn interior_field_const_index_stays_direct_ptradd() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "read_point_at_const",
        1,
        vec![ptr.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_const_usize("3", &usize_ty),
                    ],
                    interior_place(2, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &i64_ty),
                    interior_use(interior_field(2, &ptr, &adt, &i64_ty, 1)),
                )],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, add]);
    let graph = lower_interior(&llbc, "read_point_at_const");
    let ops = graph_ops(&graph);
    assert_no_interior(&ops);
    assert_direct_ptradd_not_residual_add(&ops);
}

/// `&(*p.add(i)).y` is an address, not a value read, so the add stays
/// `direct_ptradd`. The census test is the guard; this locks the lowering.
#[test]
fn interior_field_address_of_stays_direct_ptradd() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let ret = interior_shared(i64_ty.clone());
    let field = interior_field(3, &ptr, &adt, &i64_ty, 1);
    let caller = interior_caller(
        "addr_point",
        2,
        vec![ptr.clone(), usize_ty.clone()],
        ret.clone(),
        vec![
            interior_local(0, None, &ret),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(3, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &ret),
                    serde_json::json!({
                        "Ref": {"place": field, "kind": "Shared", "ptr_metadata": null}
                    }),
                )],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![interior_point()], vec![caller, add]);
    let graph = lower_interior(&llbc, "addr_point");
    let ops = graph_ops(&graph);
    assert_no_interior(&ops);
    assert_direct_ptradd_not_residual_add(&ops);
}

/// A `repr(transparent)` pointee is the inner word, not a GcArray of
/// structs. `tyref_transparent_nonzst_field` collapses the field read.
#[test]
fn interior_field_transparent_pointee_stays_direct_ptradd() {
    let i64_ty = interior_i64();
    let usize_ty = interior_usize();
    let wrapper = interior_struct(
        0,
        &["fixture", "Wrapper"],
        &[("inner", i64_ty.clone())],
        8,
        &[0],
        true,
    );
    let adt = interior_adt(0);
    let ptr = interior_raw_mut(adt.clone());
    let caller = interior_caller(
        "read_wrapper",
        2,
        vec![ptr.clone(), usize_ty.clone()],
        i64_ty.clone(),
        vec![
            interior_local(0, None, &i64_ty),
            interior_local(1, Some("p"), &ptr),
            interior_local(2, Some("i"), &usize_ty),
            interior_local(3, Some("slot"), &ptr),
        ],
        vec![
            interior_bb(
                vec![],
                interior_call(
                    1,
                    vec![
                        interior_copy(interior_place(1, &ptr)),
                        interior_copy(interior_place(2, &usize_ty)),
                    ],
                    interior_place(3, &ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(
                vec![interior_assign(
                    interior_place(0, &i64_ty),
                    interior_use(interior_field(3, &ptr, &adt, &i64_ty, 0)),
                )],
                interior_return(),
            ),
        ],
    );
    let add = interior_opaque(
        1,
        &["core", "ptr", "mut_ptr", "<Impl>", "add"],
        vec![ptr.clone(), usize_ty],
        ptr,
    );
    let llbc = llbc_with_types("fixture", vec![wrapper], vec![caller, add]);
    let graph = lower_interior(&llbc, "read_wrapper");
    let ops = graph_ops(&graph);
    assert_no_interior(&ops);
    assert_direct_ptradd_not_residual_add(&ops);
    assert!(
        ops.iter().all(|op| !matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "inner"
        )),
        "the transparent word is the pointer; ops={ops:?}"
    );
}

/// `entry_valid` on a real artefact: `(*entry_ptr(self).add(slot)).f_valid`
/// (`RDict::entry_valid`). Ignored by default (loads pyre-object.ullbc).
/// `cargo test -p majit-translate --lib interior_field_entry_valid_real -- --ignored`
#[test]
#[ignore]
fn interior_field_entry_valid_real() {
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-object.ullbc");
    let graph = super::lower_function(&llbc, "entry_valid").expect("lower entry_valid");
    let ops = graph_ops(&graph);
    let read = ops.iter().find_map(|op| match &op.kind {
        OpKind::InteriorFieldRead {
            base,
            field,
            array_type_id,
            ..
        } if field.name == "f_valid" => Some((base, array_type_id)),
        _ => None,
    });
    let Some((base, array_type_id)) = read else {
        panic!("entry_valid must read f_valid via getinteriorfield; ops={ops:?}");
    };
    assert!(
        array_type_id
            .as_deref()
            .is_some_and(|id| id.contains("Entry")),
        "array id {array_type_id:?}"
    );
    let header = ops.iter().find(|op| op.result.as_ref() == Some(base));
    match header.map(|op| &op.kind) {
        Some(OpKind::FieldRead { field, .. }) => assert_eq!(field.name, "entries"),
        other => panic!("f_valid base must be FieldRead entries, got {other:?}"),
    }
}

fn str_ref_ty() -> serde_json::Value {
    serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]})
}

pub(in crate::front::mir) fn scalar_ty(width: &str) -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": width}}})
}

/// One-argument call of opaque `core::str::<leaf>` on `&str`.
fn string_view_call_llbc(leaf: &str, dest_ty: serde_json::Value) -> Llbc {
    let arg_ty = str_ref_ty();
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "string_view_call"], "", true),
        "signature": {"is_unsafe": false, "inputs": [arg_ty], "output": dest_ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": dest_ty},
                {"index": 1, "name": "s", "span": span_json(), "ty": arg_ty}
            ]},
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": empty_generics()
                        }},
                        "args": [{"Copy": {"kind": {"Local": 1}, "ty": arg_ty}}],
                        "dest": {"kind": {"Local": 0}, "ty": dest_ty}
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}
            }, {
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "UnwindResume"}
            }, {
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta_json(&["core", "str", leaf], "", false),
        "signature": {"is_unsafe": false, "inputs": [arg_ty], "output": dest_ty},
        "body": "Opaque"
    });
    llbc_with_types("fixture", vec![], vec![caller, callee])
}

fn returned_value_is_start_input(graph: &FunctionGraph) -> bool {
    let returned_id = graph.blocks.iter().find_map(|block| {
        block.exits.iter().find_map(|exit| {
            (exit.target == graph.returnblock).then(|| {
                exit.args.iter().find_map(|arg| match arg {
                    LinkArg::Value(var) => Some(var.id()),
                    _ => None,
                })
            })?
        })
    });
    let input_id = graph
        .block(graph.startblock)
        .inputargs
        .first()
        .map(|var| var.id());
    let (Some(returned_id), Some(input_id)) = (returned_id, input_id) else {
        return false;
    };
    returned_id == input_id
        || graph.blocks.iter().any(|block| {
            block.exits.iter().any(|exit| {
                exit.args
                    .iter()
                    .any(|arg| matches!(arg, LinkArg::Value(var) if var.id() == input_id))
                    && graph.blocks.iter().any(|target| {
                        target.id == exit.target
                            && target.inputargs.iter().any(|arg| arg.id() == returned_id)
                    })
            })
        })
}

#[test]
fn str_chars_is_not_a_byte_view_identity() {
    let chars = super::lower_function(
        &string_view_call_llbc("chars", usize_ty()),
        "string_view_call",
    )
    .expect("chars lowers");
    assert!(
        graph_calls_leaf(&chars, crate::runtime_names::shims::UTF8_ITER),
        "str::chars must rewrite to __majit_utf8iter, graph: {chars:#?}"
    );
    assert!(
        !graph_calls_leaf(&chars, "chars"),
        "str::chars must not remain a residual chars call, graph: {chars:#?}"
    );
    assert!(
        !returned_value_is_start_input(&chars),
        "str::chars must not alias the string, graph: {chars:#?}"
    );

    // Dest is a W1 view (`&str`), not a window return: `&[u8]` as the
    // function output takes the pair-return ABI and converts the
    // aliased STR at Return (`emit_str_to_window`), which is not the
    // identity the assertions check.
    for leaf in ["as_bytes", "bytes"] {
        let graph = super::lower_function(
            &string_view_call_llbc(leaf, str_ref_ty()),
            "string_view_call",
        )
        .unwrap_or_else(|err| panic!("{leaf} lowers: {err}"));
        assert!(
            !graph_calls_leaf(&graph, leaf),
            "{leaf} stays the byte-view identity, graph: {graph:#?}"
        );
        assert!(
            returned_value_is_start_input(&graph),
            "{leaf} aliases the receiver, graph: {graph:#?}"
        );
    }
}

#[test]
fn str_as_ptr_rewrites_to_get_raw_address_of_string() {
    let dest = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Const"]
    });
    let graph = super::lower_function(
        &std_extern_call_fixture(
            "str_as_ptr",
            &["core", "str", "<Impl>", "as_ptr"],
            &[str_ref_ty()],
            dest,
        ),
        "str_as_ptr",
    )
    .expect("lower str::as_ptr");
    assert!(
        graph_calls_leaf(
            &graph,
            crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING
        ),
        "str::as_ptr must rewrite to get_raw_address_of_string, graph: {graph:#?}"
    );
    assert!(
        !graph_calls_leaf(&graph, "as_ptr"),
        "str::as_ptr must not remain residual, graph: {graph:#?}"
    );
}

#[test]
fn str_len_rewrites_to_strlen_marker() {
    let dest = usize_ty();
    let graph = super::lower_function(
        &std_extern_call_fixture(
            "str_len",
            &["core", "str", "<Impl>", "len"],
            &[str_ref_ty()],
            dest,
        ),
        "str_len",
    )
    .expect("lower str::len");
    assert!(
        graph_calls_leaf(&graph, "__strlen"),
        "str::len must rewrite to __strlen, graph: {graph:#?}"
    );
    assert!(
        !graph_calls_leaf(&graph, "len"),
        "str::len must not remain residual, graph: {graph:#?}"
    );
}

fn default_call_llbc(kind: serde_json::Value, layout_value: Option<serde_json::Value>) -> Llbc {
    let dest_ty = serde_json::json!({"Adt": {
        "id": 0,
        "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
    }});
    let mut decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "RandomState"], "", false),
        "kind": kind
    });
    if let Some(value) = layout_value {
        decl["layout"] = serde_json::json!([{"key": "fixture-one-target", "value": value}]);
    }
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "call_default"], "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": dest_ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": dest_ty}
            ]},
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": empty_generics()
                        }},
                        "args": [],
                        "dest": {"kind": {"Local": 0}, "ty": dest_ty}
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}
            }, {
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "UnwindResume"}
            }, {
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    });
    let callee = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta_json(&["fixture", "RandomState", "default"], "", false),
        "signature": {"is_unsafe": false, "inputs": [], "output": dest_ty},
        "body": "Opaque"
    });
    llbc_with_types("fixture", vec![decl], vec![caller, callee])
}

#[test]
fn default_call_stays_a_call_regardless_of_dest_layout() {
    // The name says `RandomState`. Result size does not drop the call.
    let two_words = default_call_llbc(
        serde_json::json!({"Struct": [
            {"name": "k0", "ty": scalar_ty("U64"), "attr_info": null},
            {"name": "k1", "ty": scalar_ty("U64"), "attr_info": null}
        ]}),
        Some(serde_json::json!({
            "size": 16,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0, 8]}],
            "repr": {"repr_algo": "Rust", "align_modif": null, "transparent": false}
        })),
    );
    let graph = super::lower_function(&two_words, "call_default").expect("two-word default");
    assert!(
        graph_calls_leaf(&graph, "default"),
        "a non-zero default stays a call, graph: {graph:#?}"
    );

    let zst = default_call_llbc(
        serde_json::json!({"Struct": []}),
        Some(serde_json::json!({
            "size": 0,
            "align": 1,
            "variant_layouts": [{"field_offsets": []}],
            "repr": {"repr_algo": "Rust", "align_modif": null, "transparent": false}
        })),
    );
    let graph = super::lower_function(&zst, "call_default").expect("zero-sized default");
    assert!(
        graph_calls_leaf(&graph, "default"),
        "a named empty class default stays a call, graph: {graph:#?}"
    );

    let aligned = default_call_llbc(
        serde_json::json!({"Struct": []}),
        Some(serde_json::json!({
            "size": 8,
            "align": 8,
            "variant_layouts": [{"field_offsets": []}],
            "repr": {"repr_algo": "Rust", "align_modif": null, "transparent": false}
        })),
    );
    let graph =
        super::lower_function(&aligned, "call_default").expect("aligned field-less default");
    assert!(
        graph_calls_leaf(&graph, "default"),
        "a field-less struct with a non-zero size stays a call, graph: {graph:#?}"
    );

    let no_layout = default_call_llbc(serde_json::json!({"Struct": []}), None);
    let graph = super::lower_function(&no_layout, "call_default")
        .expect("fieldless default without layout");
    assert!(
        graph_calls_leaf(&graph, "default"),
        "a fieldless default without a recorded size stays a call, graph: {graph:#?}"
    );
}

/// Both operands are a copy of `*mut ItemsBlock`. `wrappeditems` is
/// published as the fixed list (`[*mut PyObject]`, `ll_fixed_items`);
/// `items` stays the `ItemsBlock` header.
fn items_base_operand_llbc(owner_path: &[&str], field: &str) -> Llbc {
    let items_ptr = interior_raw_mut(interior_adt(1));
    let owner = interior_struct(0, owner_path, &[(field, items_ptr.clone())], 8, &[0], false);
    let items_block = interior_struct(1, &["object_array", "ItemsBlock"], &[], 8, &[], false);
    let owner_ptr = interior_raw_mut(interior_adt(0));
    let owner_ty = interior_adt(0);
    let caller = interior_caller(
        "read_items_base",
        1,
        vec![owner_ptr.clone()],
        items_ptr.clone(),
        vec![
            interior_local(0, None, &items_ptr),
            interior_local(1, Some("obj"), &owner_ptr),
            interior_local(2, Some("block"), &items_ptr),
        ],
        vec![
            interior_bb(
                vec![interior_assign(
                    interior_place(2, &items_ptr),
                    interior_use(interior_field(1, &owner_ptr, &owner_ty, &items_ptr, 0)),
                )],
                interior_call(
                    1,
                    vec![interior_copy(interior_place(2, &items_ptr))],
                    interior_place(0, &items_ptr),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(vec![], interior_return()),
        ],
    );
    let callee = interior_opaque(
        1,
        &["pyre_object", "object_array", "items_block_items_base"],
        vec![items_ptr.clone()],
        items_ptr,
    );
    llbc_with_types("fixture", vec![owner, items_block], vec![caller, callee])
}

fn field_read_of<'a>(ops: &'a [&'a SpaceOperation], name: &str) -> &'a Variable {
    ops.iter()
        .find_map(|op| match (&op.result, &op.kind) {
            (Some(result), OpKind::FieldRead { field, .. }) if field.name == name => Some(result),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing FieldRead {name}; ops={ops:?}"))
}

fn link_carries(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|block| {
        block.exits.iter().any(|link| {
            link.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(value) if value == var))
        })
    })
}

fn items_base_call<'a>(ops: &'a [&'a SpaceOperation]) -> Option<&'a SpaceOperation> {
    ops.iter().copied().find(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("items_block_items_base")
        )
    })
}

/// `items_block_items_base` on a fixed-list GcArray is `ll_fixed_items`:
/// the call is dropped and the operand flows onward. The same callee
/// on an `ItemsBlock` header stays a call.
#[test]
fn items_block_items_base_on_a_fixed_list_is_the_operand() {
    let llbc = items_base_operand_llbc(&["tupleobject", "W_TupleObject"], "wrappeditems");
    let graph = lower_interior(&llbc, "read_items_base");
    let ops = graph_ops(&graph);
    let field = field_read_of(&ops, "wrappeditems");
    assert!(
        items_base_call(&ops).is_none(),
        "fixed-list items_block_items_base must be dropped; ops={ops:?}"
    );
    assert!(
        link_carries(&graph, field),
        "the fixed list itself must flow onward; ops={ops:?}"
    );
}

#[test]
fn items_block_items_base_on_an_items_block_stays_a_call() {
    let llbc = items_base_operand_llbc(&["listobject", "W_ListObject"], "items");
    let graph = lower_interior(&llbc, "read_items_base");
    let ops = graph_ops(&graph);
    let field = field_read_of(&ops, "items");
    let call = items_base_call(&ops).unwrap_or_else(|| {
        panic!("ItemsBlock items_block_items_base must stay a call; ops={ops:?}")
    });
    match &call.kind {
        OpKind::Call { args, .. } => {
            assert!(
                matches!(args.first(), Some(LinkArg::Value(value)) if value == field),
                "the call must take the ItemsBlock field; ops={ops:?}"
            );
        }
        other => panic!("expected a call, got {other:?}"),
    }
    let result = call
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("the call must define a result; ops={ops:?}"));
    let result_is_used = link_carries(&graph, result)
        || ops.iter().any(|op| match &op.kind {
            OpKind::Call { args, .. } => args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(value) if value == result)),
            _ => false,
        });
    assert!(
        result_is_used,
        "the ItemsBlock call result must be used; ops={ops:?}"
    );
    assert!(
        !link_carries(&graph, field),
        "the ItemsBlock header must not replace the call; ops={ops:?}"
    );
}

/// `items_block_capacity` is `len(l.items)` of the ITEMARRAY. A
/// published fixed list (`wrappeditems`) lowers to `arraylen_gc`. A
/// resizable `ItemsBlock` header (`l.items`) stays a call: empty
/// lists keep NULL storage.
fn items_capacity_operand_llbc(owner_path: &[&str], field: &str) -> Llbc {
    let items_ptr = interior_raw_mut(interior_adt(1));
    let usize_ty = interior_usize();
    let owner = interior_struct(0, owner_path, &[(field, items_ptr.clone())], 8, &[0], false);
    let items_block = interior_struct(1, &["object_array", "ItemsBlock"], &[], 8, &[], false);
    let owner_ptr = interior_raw_mut(interior_adt(0));
    let owner_ty = interior_adt(0);
    let caller = interior_caller(
        "read_items_capacity",
        1,
        vec![owner_ptr.clone()],
        usize_ty.clone(),
        vec![
            interior_local(0, None, &usize_ty),
            interior_local(1, Some("obj"), &owner_ptr),
            interior_local(2, Some("block"), &items_ptr),
        ],
        vec![
            interior_bb(
                vec![interior_assign(
                    interior_place(2, &items_ptr),
                    interior_use(interior_field(1, &owner_ptr, &owner_ty, &items_ptr, 0)),
                )],
                interior_call(
                    1,
                    vec![interior_copy(interior_place(2, &items_ptr))],
                    interior_place(0, &usize_ty),
                    2,
                    1,
                ),
            ),
            interior_bb(vec![], interior_unwind()),
            interior_bb(vec![], interior_return()),
        ],
    );
    let callee = interior_opaque(
        1,
        &["pyre_object", "object_array", "items_block_capacity"],
        vec![items_ptr],
        usize_ty,
    );
    llbc_with_types("fixture", vec![owner, items_block], vec![caller, callee])
}

fn items_capacity_call<'a>(ops: &'a [&'a SpaceOperation]) -> Option<&'a SpaceOperation> {
    ops.iter().copied().find(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("items_block_capacity")
        )
    })
}

fn items_capacity_arraylen<'a>(ops: &'a [&'a SpaceOperation]) -> Option<&'a SpaceOperation> {
    ops.iter().copied().find(|op| {
        matches!(
            &op.kind,
            OpKind::ArrayLen {
                array_type_id: Some(id),
                nolength: false,
                ..
            } if id == super::OBJECT_REF_GCARRAY_TYPE_ID
        )
    })
}

#[test]
fn items_block_capacity_on_a_fixed_list_is_arraylen() {
    let llbc = items_capacity_operand_llbc(&["tupleobject", "W_TupleObject"], "wrappeditems");
    let graph = lower_interior(&llbc, "read_items_capacity");
    let ops = graph_ops(&graph);
    let field = field_read_of(&ops, "wrappeditems");
    assert!(
        items_capacity_call(&ops).is_none(),
        "fixed-list items_block_capacity must be dropped; ops={ops:?}"
    );
    let len = items_capacity_arraylen(&ops)
        .unwrap_or_else(|| panic!("fixed-list capacity must be ArrayLen; ops={ops:?}"));
    match &len.kind {
        OpKind::ArrayLen { base, .. } => {
            assert!(
                base == field,
                "arraylen must take the wrappeditems field; ops={ops:?}"
            );
        }
        other => panic!("expected ArrayLen, got {other:?}"),
    }
}

#[test]
fn items_block_capacity_on_an_items_block_stays_a_call() {
    let llbc = items_capacity_operand_llbc(&["listobject", "W_ListObject"], "items");
    let graph = lower_interior(&llbc, "read_items_capacity");
    let ops = graph_ops(&graph);
    let field = field_read_of(&ops, "items");
    let call = items_capacity_call(&ops)
        .unwrap_or_else(|| panic!("resizable ItemsBlock capacity must stay a call; ops={ops:?}"));
    match &call.kind {
        OpKind::Call { args, .. } => {
            assert!(
                matches!(args.first(), Some(LinkArg::Value(value)) if value == field),
                "the call must take the ItemsBlock field; ops={ops:?}"
            );
        }
        other => panic!("expected a call, got {other:?}"),
    }
    assert!(
        items_capacity_arraylen(&ops).is_none(),
        "the items header is not ll_fixed_length; ops={ops:?}"
    );
}

/// `float_to_int_or_float` is `source = float_items.as_slice();
/// Vec::with_capacity(source.len())`. The as_slice fold used to keep
/// only `block` (`ll_items`); `source.len()` then became
/// `ll_fixed_length`. Record `l.length` (`ll_length`).
#[test]
fn float_array_slice_len_reads_logical_len() {
    let graph = lower_object_fn("pyre_object::listobject::float_to_int_or_float");
    let ops = graph_ops(&graph);
    let has_len = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, ty: ValueType::Int, .. }
                if field.name == "len"
                    && field.owner_root.as_deref() == Some("FloatArray")
        )
    });
    let has_block = ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. }
                if field.name == "block"
                    && field.owner_root.as_deref() == Some("FloatArray")
        )
    });
    let has_arraylen = ops
        .iter()
        .any(|op| matches!(&op.kind, OpKind::ArrayLen { .. }));
    assert!(
        has_len,
        "as_slice().len() must read FloatArray.len (ll_length); ops={ops:?}"
    );
    assert!(
        has_block,
        "as_slice must read FloatArray.block (ll_items); ops={ops:?}"
    );
    assert!(
        !has_arraylen,
        "view length is ll_length, not ll_fixed_length; ops={ops:?}"
    );
}

/// `[PyObject]` narrowing follows the ARRAY identity
/// (`SomeList.rtyper_makerepr`), not a graph-name test. `kwargs_at`
/// and `w_tuple_new_from_slice` share `&[PyObjectRef]`.
#[test]
fn object_array_index_narrows_from_the_array_type() {
    for name in [
        "pyre_object::kwargsdict::kwargs_at",
        "pyre_object::tupleobject::w_tuple_new_from_slice",
    ] {
        let graph = lower_object_fn(name);
        let ops = graph_ops(&graph);
        assert!(
            ops.iter()
                .any(|op| crate::model::cast_instance_root(&op.kind) == Some("[PyObject]")),
            "{name} must narrow the object-array index through [PyObject]; ops={ops:?}"
        );
    }
}

/// `ll_list_ascii_setitem_fast` is `l.ascii_items[index] = item`.
/// The write must use `ll_items` (`block`), the same base the paired
/// read uses (`ll_setitem_fast`).
#[test]
fn ascii_setitem_writes_items_block() {
    let graph = lower_object_fn("pyre_object::listobject::ll_list_ascii_setitem_fast");
    let ops = graph_ops(&graph);
    let write = ops.iter().find_map(|op| match &op.kind {
        OpKind::ArrayWrite {
            base,
            array_type_id: Some(array_type_id),
            ..
        } if array_type_id == super::STRING_GCREF_GCARRAY_TYPE_ID => Some(base),
        _ => None,
    });
    let Some(base) = write else {
        panic!("ascii_setitem must emit setarrayitem on [str]; ops={ops:?}");
    };
    let producer = ops
        .iter()
        .find(|op| op.result.as_ref() == Some(base))
        .unwrap_or_else(|| panic!("ArrayWrite base must be defined; ops={ops:?}"));
    let items = if crate::model::cast_instance_root(&producer.kind) == Some("[str]") {
        match &producer.kind {
            OpKind::Call { args, .. } => args
                .first()
                .and_then(LinkArg::as_variable)
                .unwrap_or_else(|| panic!("[str] cast needs a value arg; ops={ops:?}")),
            other => panic!("[str] narrow is a call, got {other:?}"),
        }
    } else {
        base
    };
    let items_op = ops
        .iter()
        .find(|op| op.result.as_ref() == Some(items))
        .unwrap_or_else(|| panic!("write base must be a FieldRead of block; ops={ops:?}"));
    match &items_op.kind {
        OpKind::FieldRead { field, .. } => {
            assert_eq!(field.name, "block", "write base is ll_items; ops={ops:?}");
            assert_eq!(
                field.owner_root.as_deref(),
                Some("UnicodeArray"),
                "ascii items live in UnicodeArray; ops={ops:?}"
            );
        }
        other => panic!("write base must be FieldRead block, got {other:?}"),
    }
}
