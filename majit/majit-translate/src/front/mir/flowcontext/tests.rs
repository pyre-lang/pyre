use super::simplify_lowered_graph_expanding;

use super::harden_duplicate_leaf_metadata;

use super::{
    cast_pointer_marker_op, dont_look_inside_return_token, no_tombstoned_leaves,
    push_cast_ptr_to_int, push_direct_ptradd, push_ptr_to_unsigned_cast,
    scalar_replace_named_struct_aggregates, tyref_to_attr_value_type_for_struct_field,
};

use crate::flowspace::model::Variable;

use crate::model::{
    CallTarget, FieldDescriptor, FrameState, FunctionGraph, LinkArg, OpKind, SpaceOperation,
    ValueType, VecFieldPart,
};

use majit_charon_reader::{Llbc, ullbc::TyRef};

use crate::front::mir::tests::{
    adt_node, boxing_cluster_ctor_new_counts, boxing_cluster_with_nested_header, calls_leaf,
    code_struct, count_branch_calls, count_method_calls, disc_site, empty_llbc, empty_struct_attrs,
    fixture_item_meta, fixture_llbc, graph_call_leaves, graph_ops, i64_field, ident_path,
    interior_bb, interior_caller, interior_i64, interior_local, interior_meta, interior_opaque,
    interior_raw_mut, interior_return, interior_usize, leftover_fmt_message_paths, llbc_with_types,
    lower_object_fn, ok_ctor_shape, option_adt_node, option_enum, option_int_arg,
    option_ok_or_else_site, push_fn_const, replace_struct_ctors, rows, same_as_count,
    some_ctor_ops, some_ctor_shape, struct_ctor_graph, struct_ctor_ops, struct_decl, struct_news,
    tyref_to_attr_value_type, window_call_leafs, window_empty_generics, window_graph_ops,
    window_llbc, window_local, window_meta, window_mut_u8_ptr, window_opaque_fun, window_place,
    window_shared_slice, window_slice, window_span, window_u8, window_unwind_and_return_blocks,
    window_usize,
};

/// Per-graph simplification after lowering — the model-layer slice of
/// RPython `simplify_graph(graph)` (`simplify.py`), which
/// upstream runs on every freshly built flow graph.  Only the passes
/// the Abort → `RaiseImplicit` fold needs are wired:
///
/// - `eliminate_empty_blocks` (simplify.py) collapses the empty
///   raise block between a discriminant switch's `else
///   unreachable!()` exit and `exceptblock`, exposing the
///   `[Constant(AssertionError), …]` link to the next pass.
/// - `retarget_assert_raise_blocks` points an edge into a bare
///   implicit-raise block at `exceptblock` instead, which is where
///   upstream's flow space puts it and what the next pass matches.
/// - `remove_assertion_errors` (simplify.py) prunes the
///   shouldn't-occur branch and promotes the surviving exit to an
///   unconditional link.
/// - `prune_dead_phis` (`transform_dead_op_vars`, simplify.py)
///   melts the now-dead condition ops — the `__discriminant`
///   FieldRead feeding the dropped exitswitch
///   (`removeassert.py` `remove_asserts` "now melt away the (hopefully) dead
///   operation that compute the condition").  Upstream leaves this to
///   the later backendopt sweep (`backendopt/all.py`); the model
///   layer has no later sweep, so it runs here — see `sweep_dead_vars`
///   below for when.
///
/// `sweep_dead_vars` selects which of the two call sites this is.  The
/// end-of-lowering call passes `true` and runs `prune_dead_phis`
/// unconditionally, as `all_passes` does.  The result-exc pre-pass passes
/// `false`: `rewire_result_exc_call_sites` walks the `Result` diamond by
/// following one `Variable` identity from link args into the target's
/// inputargs, and a `Result` that is dead downstream is exactly what the
/// sweep removes — the walk then fails the whole graph with "block N's exit
/// does not carry the tracked value".  Measured: three `_io` `__init__`
/// graphs stopped lowering, taking the `reader_reset_buf` /
/// `writer_reset_buf` callees they queue with them.  The pre-pass is
/// pyre-only and exists so the diamond matcher sees a pruned discriminant
/// switch; keeping the sweep out of it leaves one sweep per graph, which is
/// what upstream has.
fn simplify_lowered_graph(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    sweep_dead_vars: bool,
    guard_words: &std::collections::HashSet<Variable>,
) {
    simplify_lowered_graph_expanding(
        graph,
        struct_field_attrs,
        sweep_dead_vars,
        guard_words,
        &|_, _| None,
    )
    .expect("inline-struct expansion is off");
}

/// A missing Charon layout on `GcEntries { length: usize, items: [T; 0] }`
/// still has to place items at the length word rounded up to `T`'s
/// alignment (`array_items_base`). A 4-byte wasm32 word and an 8-aligned
/// `Entry<i64, PyObjectRef>` therefore sit at 8, not 4; a 64-bit word
/// already satisfies that alignment.
#[test]
fn array_field_tail_offset_rounds_length_word_to_element_align() {
    assert_eq!(
        super::Lowering::array_field_tail_offset_fallback(4, Some(8)),
        Some(8)
    );
    assert_eq!(
        super::Lowering::array_field_tail_offset_fallback(8, Some(8)),
        Some(8)
    );
    assert_eq!(
        super::Lowering::array_field_tail_offset_fallback(4, None),
        None
    );
}

/// `get_array_token` uses the actual array type. An 8-aligned source
/// tail must not pick up a 16-aligned `.cast()` dest.
#[test]
fn array_tail_elem_align_prefers_source_array_over_cast_dest() {
    assert_eq!(
        super::Lowering::array_tail_instantiated_elem_align(Some(8), None, Some(16)),
        Some(8)
    );
    assert_eq!(
        super::Lowering::array_tail_instantiated_elem_align(None, Some(8), Some(16)),
        Some(8)
    );
    assert_eq!(
        super::Lowering::array_tail_instantiated_elem_align(None, None, Some(8)),
        Some(8)
    );
}

/// `direct_arrayitems` constructs a pointer to item zero. Unknown
/// items offset stops translation rather than emit a different op.
#[test]
fn unknown_array_items_offset_stops_direct_arrayitems() {
    let err = super::Lowering::array_field_items_offset(None, "GcEntries", "items")
        .expect_err("unknown items offset must stop translation");
    assert!(err.to_string().contains("direct_arrayitems"), "{err}");
    assert_eq!(
        super::Lowering::array_field_items_offset(Some(8), "GcEntries", "items").unwrap(),
        8
    );
}

/// `entries_item_ptr` of `GcEntries<i64, PyObjectRef>` is
/// `addr_of_mut!((*header).items).cast()`. The generic `GcEntries`
/// declaration has no Charon layout and unresolved type variables,
/// so the items offset is the length word rounded up to the
/// instantiated `Entry` alignment (`array_items_base`). `Entry`
/// always contains `f_hash: u64`, so that offset is 8.
#[test]
fn entries_item_ptr_gcentries_i64_gcref_items_offset_is_8() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let name_path = "pyre_object::rordereddict_entries::entries_item_ptr";
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == name_path)
        .unwrap_or_else(|| panic!("{name_path} in pyre-object LLBC"));
    let gcref_id = llbc
        .iter_type_decls()
        .find(|td| td.item_meta.name_path() == "pyre_object::pyobject::PyObject")
        .unwrap_or_else(|| panic!("PyObject in pyre-object LLBC"))
        .def_id;
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let gcref_ty = serde_json::json!({
        "RawPtr": [
            {
                "Adt": {
                    "id": gcref_id,
                    "generics": {
                        "regions": [],
                        "types": [],
                        "const_generics": [],
                        "trait_refs": []
                    },
                    "builtin": null
                }
            },
            "Mut"
        ]
    });
    let body = crate::front::clause_spec::substituted_unstructured(
        fd,
        &llbc,
        &[],
        &[i64_ty, gcref_ty],
        &[],
    )
    .expect("substituted entries_item_ptr body");
    let context = super::LowerContext::new(&llbc);
    let struct_fields = super::struct_field_registry_from_rows(&context.struct_fields);
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let jitdriver_receiver_roots =
        crate::codewriter::jtransform::default_jitdriver_receiver_roots();
    let graph = super::lower_unstructured_with_static_addrs_and_attrs(
        &llbc,
        fd,
        &body,
        crate::HostStaticAddrs::default(),
        &jitdriver_receiver_roots,
        &context.struct_field_attrs,
        &struct_fields,
        &context.dont_look_inside,
        &context.tombstoned_leaves,
        &context.gc_struct_ids,
        &context.immutable_fields,
        &context.raw_alloc_flavor_structs,
        accum.has_builder,
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        None,
        true,
        None,
    )
    .unwrap_or_else(|err| panic!("lower substituted {name_path}: {err:?}"));

    // `emit_array_tail_item_ptr` is `cast_ptr_to_int` + ConstInt(offset)
    // + add + `cast_int_to_ptr`. The address-taken `items` FieldRead is
    // unused after that rewrite and may be dropped.
    let mut offset = None;
    let mut saw_cast_int_to_ptr = false;
    for block in &graph.blocks {
        for (i, op) in block.operations.iter().enumerate() {
            if let OpKind::ConstInt(n) = op.kind {
                let add = block.operations.get(i + 1).map(|next| &next.kind);
                if matches!(
                    add,
                    Some(OpKind::BinOp { op, .. }) if op == "add"
                ) {
                    offset = Some(n);
                }
            }
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
                && segments.last().is_some_and(|s| s == "cast_int_to_ptr")
            {
                saw_cast_int_to_ptr = true;
            }
        }
    }
    assert!(
        saw_cast_int_to_ptr,
        "substituted entries_item_ptr must emit direct_arrayitems (cast_int_to_ptr)"
    );
    assert_eq!(
        offset,
        Some(8),
        "GcEntries items sit at the length word rounded up to Entry alignment"
    );
}

/// `inline_vec_field_part` emits the component read from the SSI
/// local (`ssa.py` `SSA_to_SSI`), not the dominating producer.
#[test]
fn inline_vec_field_part_uses_ssi_local_as_field_read_base() {
    let mut graph = FunctionGraph::new("inline_vec_ssi_base");
    let pred = graph.startblock;
    let target = graph.create_block();
    let base = graph
        .push_op_var(
            pred,
            OpKind::Input {
                name: "holder".into(),
                ty: ValueType::Ref(Some("Holder".into())),
                class_root: None,
            },
            true,
        )
        .unwrap();
    let vec_var = graph.alloc_value_var();
    graph.block_mut(pred).operations.push(SpaceOperation {
        result: Some(vec_var.clone()),
        kind: OpKind::FieldRead {
            base: base.clone(),
            field: FieldDescriptor::new("items", Some("Holder".into())).with_inline_vec(true),
            ty: ValueType::Ref(None),
            pure: true,
        },
    });
    graph.set_goto(pred, target, vec![]);

    let len = super::inline_vec_field_part(
        &mut graph,
        target,
        &vec_var,
        VecFieldPart::Len,
        ValueType::Int,
    )
    .expect("inline vec len read");
    let op = graph
        .block(target)
        .operations
        .iter()
        .find(|op| op.result.as_ref() == Some(&len))
        .expect("len FieldRead");
    match &op.kind {
        OpKind::FieldRead {
            base: read_base, ..
        } => {
            assert!(
                graph.variable_defined_in_block(target, read_base),
                "component read base is defined in the use block"
            );
            assert_ne!(
                read_base.id(),
                base.id(),
                "component read does not bind the dominating producer"
            );
        }
        other => panic!("expected FieldRead, got {other:?}"),
    }
}

/// `dyn_receiver_data_word` emits FatData from the SSI local, not the
/// dominating producer (`ssa.py` `SSA_to_SSI`).
#[test]
fn dyn_receiver_data_word_uses_ssi_local_as_field_read_base() {
    let mut graph = FunctionGraph::new("dyn_receiver_ssi_base");
    let pred = graph.startblock;
    let target = graph.create_block();
    let base = graph
        .push_op_var(
            pred,
            OpKind::Input {
                name: "strategy".into(),
                ty: ValueType::Ref(Some("Strategy".into())),
                class_root: None,
            },
            true,
        )
        .unwrap();
    let recv = graph.alloc_value_var();
    graph.block_mut(pred).operations.push(SpaceOperation {
        result: Some(recv.clone()),
        kind: OpKind::FieldRead {
            base: base.clone(),
            field: FieldDescriptor::new("imp", Some("Strategy".into()))
                .with_vec_part(VecFieldPart::FatMeta),
            ty: ValueType::Int,
            pure: true,
        },
    });
    graph.set_goto(pred, target, vec![]);

    let data =
        super::emit_dyn_receiver_data_word(&mut graph, target, &recv).expect("fat data word");
    let op = graph
        .block(target)
        .operations
        .iter()
        .find(|op| op.result.as_ref() == Some(&data))
        .expect("FatData FieldRead");
    match &op.kind {
        OpKind::FieldRead {
            base: read_base, ..
        } => {
            assert!(
                graph.variable_defined_in_block(target, read_base),
                "FatData read base is defined in the use block"
            );
            assert_ne!(
                read_base.id(),
                base.id(),
                "FatData read does not bind the dominating producer"
            );
        }
        other => panic!("expected FieldRead, got {other:?}"),
    }
}

/// View `.len()` must not put a predecessor Variable on a consumer
/// link (`FrameState.getoutputargs`, `checkgraph` `definevar`).
#[test]
fn array_view_len_link_operand_is_defined_in_the_consumer_block() {
    let mut graph = FunctionGraph::new("array_view_len_consumer");
    let pred = graph.startblock;
    let consumer = graph.create_block();
    let len_var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(pred).operations.push(SpaceOperation {
        result: Some(len_var.clone()),
        kind: OpKind::ConstInt(3),
    });
    graph.set_goto(pred, consumer, vec![]);

    assert!(
        !graph.variable_defined_in_block(consumer, &len_var),
        "predecessor length is not a definition in the consumer"
    );
    let bound = super::bind_array_view_len_in_block(&mut graph, consumer, len_var.clone());
    assert_ne!(
        bound.id(),
        len_var.id(),
        "consumer must not name the predecessor Variable on a link"
    );
    assert!(
        graph.variable_defined_in_block(consumer, &bound),
        "same_as result is defined in the consumer (`checkgraph` definevar)"
    );
    match graph.block(consumer).operations.last().map(|op| &op.kind) {
        Some(OpKind::UnaryOp { op, operand, .. }) if op == "same_as" => {
            assert_eq!(operand.id(), len_var.id());
        }
        other => panic!("expected same_as of the predecessor length, got {other:?}"),
    }
    graph.set_return(consumer, Some(bound.clone()));
    match graph.block(consumer).exits[0].args.as_slice() {
        [LinkArg::Value(v)] => {
            assert!(
                graph.variable_defined_in_block(consumer, v),
                "return link operand is defined in the source block"
            );
            assert_eq!(v.id(), bound.id());
        }
        other => panic!("expected one return value, got {other:?}"),
    }
    let same = super::bind_array_view_len_in_block(&mut graph, pred, len_var.clone());
    assert_eq!(
        same.id(),
        len_var.id(),
        "a length already defined in the block is reused"
    );
}

fn result_flow_llbc() -> Llbc {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|seg| serde_json::json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        })
    };
    let enum_decl = |def_id: u64, path: &[&str], variants: &[(&str, u64, u64)]| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": meta(path),
            "kind": {"Enum": variants.iter().map(|(name, disc, tvar)| serde_json::json!({
                "name": name,
                "fields": [{"name": null, "ty": {"TypeVar": {"Bound": [0, tvar]}}}],
                "discriminant": {"Scalar": {"Signed": ["Isize", disc.to_string()]}}
            })).collect::<Vec<_>>()},
            "layout": []
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                enum_decl(0, &["core", "result", "Result"], &[("Ok", 0, 0), ("Err", 1, 1)]),
                // `ControlFlow<B, C>`: Continue(C) is variant 0 (type var 1),
                // Break(B) is variant 1 (type var 0).
                enum_decl(1, &["core", "ops", "control_flow", "ControlFlow"], &[("Continue", 0, 1), ("Break", 1, 0)]),
                // Charon `type_by_id` indexes `type_decls` (`Llbc::type_by_id`).
                // `Result<Infallible, i64>` at id 2 so the Ok arg is a
                // named empty enum, not `??adt#99`.
                enum_decl(2, &["core", "convert", "Infallible"], &[]),
            ],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("result/controlflow llbc")
}

fn branch_adt_ty(def_id: u64, args: &[serde_json::Value]) -> TyRef {
    serde_json::from_value(serde_json::json!({
        "Adt": {
            "id": def_id,
            "generics": {"regions": [], "types": args, "const_generics": [], "trait_refs": []}
        }
    }))
    .expect("adt ty")
}

fn branch_graph(payloads: crate::model::ResultBranchPayloads) -> FunctionGraph {
    let mut graph = FunctionGraph::new("result_branch_payload");
    let entry = graph.startblock;
    let operand = graph
        .push_op_var(entry, OpKind::ConstInt(1), true)
        .expect("operand");
    let result = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::method("branch", Some("core::result::Result".into()))
                    .with_branch_payloads(payloads),
                args: vec![LinkArg::Value(operand)],
                result_ty: ValueType::Ref(Some("core::ops::control_flow::ControlFlow".into())),
            },
            true,
        )
        .expect("branch");
    graph.set_return(entry, Some(result));
    graph
}

fn payload_field_tys(graph: &FunctionGraph) -> (Vec<ValueType>, Vec<ValueType>) {
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        match &op.kind {
            OpKind::FieldRead { field, ty, .. } if field.name == "__pos_0" => {
                reads.push(ty.clone());
            }
            OpKind::FieldWrite { field, ty, .. } if field.name == "__pos_0" => {
                writes.push(ty.clone());
            }
            _ => {}
        }
    }
    (reads, writes)
}

#[test]
fn result_branch_match_uses_int_payloads_from_the_type_decls() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let llbc = result_flow_llbc();
    let result_ty = branch_adt_ty(0, &[i64_ty.clone(), u8_ty.clone()]);
    // ControlFlow<B, C = U8, I64>: Break is B, Continue is C.
    let flow_ty = branch_adt_ty(1, &[u8_ty, i64_ty]);
    let tombstoned = std::collections::HashSet::new();
    let payloads = crate::model::ResultBranchPayloads {
        ok: super::adt_variant_payload_type(
            &result_ty,
            "Ok",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        err: super::adt_variant_payload_type(
            &result_ty,
            "Err",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        continue_ty: super::adt_variant_payload_type(
            &flow_ty,
            "Continue",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        break_ty: super::adt_variant_payload_type(
            &flow_ty,
            "Break",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
    };
    assert_eq!(payloads.ok, Some(ValueType::Int));
    assert_eq!(payloads.err, Some(ValueType::Unsigned));
    assert_eq!(payloads.continue_ty, Some(ValueType::Int));
    assert_eq!(payloads.break_ty, Some(ValueType::Unsigned));
    let mut graph = branch_graph(payloads);
    super::lower_result_branch_to_control_flow(&mut graph);
    let (reads, writes) = payload_field_tys(&graph);
    assert_eq!(reads, vec![ValueType::Int, ValueType::Unsigned]);
    assert_eq!(writes, vec![ValueType::Int, ValueType::Unsigned]);
}

#[test]
fn result_branch_break_of_infallible_result_builds_err() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let llbc = result_flow_llbc();
    let infallible = serde_json::json!({
        "Adt": {"id": 2, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}
    });
    let residual = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {"regions": [], "types": [infallible, i64_ty.clone()], "const_generics": [], "trait_refs": []}
        }
    });
    let result_ty = branch_adt_ty(0, &[i64_ty.clone(), i64_ty.clone()]);
    // ControlFlow<Result<Infallible, i64>, i64>: type 0 is Break, type 1 is Continue.
    let flow_ty = branch_adt_ty(1, &[residual, i64_ty]);
    let tombstoned = std::collections::HashSet::new();
    let payloads = crate::model::ResultBranchPayloads {
        ok: super::adt_variant_payload_type(
            &result_ty,
            "Ok",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        err: super::adt_variant_payload_type(
            &result_ty,
            "Err",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        continue_ty: super::adt_variant_payload_type(
            &flow_ty,
            "Continue",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        break_ty: super::adt_variant_payload_type(
            &flow_ty,
            "Break",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
    };
    assert_eq!(payloads.ok, Some(ValueType::Int));
    assert_eq!(payloads.err, Some(ValueType::Int));
    assert_eq!(payloads.continue_ty, Some(ValueType::Int));
    assert_ne!(
        payloads.break_ty, payloads.err,
        "Result<Infallible, i64> is not banked as i64"
    );
    let mut graph = branch_graph(payloads);
    super::lower_result_branch_to_control_flow(&mut graph);
    let mut ctors = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } = &op.kind
            {
                ctors.push(name.clone());
            }
        }
    }
    assert!(
        ctors.iter().any(|name| name == "Err"),
        "Break payload must be Result::Err, got {ctors:?}"
    );
    let (reads, writes) = payload_field_tys(&graph);
    assert_eq!(reads, vec![ValueType::Int, ValueType::Int]);
    assert!(
        writes.iter().any(|ty| matches!(
            ty,
            ValueType::Ref(Some(name)) if name.starts_with("result::Result<")
        )),
        "Break stores the Result value, got {writes:?}"
    );
}

#[test]
fn result_branch_match_emits_nothing_for_void_payloads() {
    let unit = serde_json::json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}});
    let llbc = result_flow_llbc();
    let result_ty = branch_adt_ty(0, &[unit.clone(), unit.clone()]);
    let flow_ty = branch_adt_ty(1, &[unit.clone(), unit]);
    let tombstoned = std::collections::HashSet::new();
    let payloads = crate::model::ResultBranchPayloads {
        ok: super::adt_variant_payload_type(
            &result_ty,
            "Ok",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        err: super::adt_variant_payload_type(
            &result_ty,
            "Err",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        continue_ty: super::adt_variant_payload_type(
            &flow_ty,
            "Continue",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
        break_ty: super::adt_variant_payload_type(
            &flow_ty,
            "Break",
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new(),
        ),
    };
    assert_eq!(payloads.ok, None);
    assert_eq!(payloads.err, None);
    assert_eq!(payloads.continue_ty, None);
    assert_eq!(payloads.break_ty, None);
    let mut graph = branch_graph(payloads);
    super::lower_result_branch_to_control_flow(&mut graph);
    let (reads, writes) = payload_field_tys(&graph);
    assert!(reads.is_empty(), "void payload emits no read: {reads:?}");
    assert!(writes.is_empty(), "void payload emits no field: {writes:?}");
    assert_eq!(count_branch_calls(&graph), 0);
}

/// A block emptied *after* the head `eliminate_empty_blocks` must still
/// be rewired past, so the values riding the link into it do not survive
/// lowering as dead link args.
///
/// Shape reproduced from `pyre_interpreter::pyframe::<Impl>::set_locals_w`:
/// the forwarding block holds one dead unit-`()` constructor, so it is
/// *not* empty when the head call runs; `prune_dead_boxing_remnants` /
/// `remove_dead_aggregates` drop that constructor, and the block is empty
/// only from then on.  The array the entry block writes to rides the link
/// into it, and `prune_dead_phis` cannot reclaim that link arg because the
/// forwarding block reuses the predecessor's `Variable` identities, so
/// per-variable liveness sees the array as read — by the predecessor's own
/// `ArrayWrite`.
#[test]
fn block_emptied_after_the_head_collapse_is_still_rewired_past() {
    let unit_ctor = || OpKind::Call {
        target: CallTarget::synthetic_transparent_ctor("Tuple"),
        args: Vec::new(),
        result_ty: ValueType::Ref(Some("Tuple".to_string())),
    };

    let mut graph = FunctionGraph::new("test");
    let entry = graph.startblock;
    let array = graph
        .push_op_var(entry, OpKind::ConstInt(1), true)
        .expect("array var");
    let index = graph
        .push_op_var(entry, OpKind::ConstInt(0), true)
        .expect("index var");
    let value = graph
        .push_op_var(entry, OpKind::ConstInt(7), true)
        .expect("value var");
    // Side-effecting, so `prune_dead_phis` pins `array` in `read_vars`.
    graph.push_op_var(
        entry,
        OpKind::ArrayWrite {
            base: array.clone(),
            index,
            value: LinkArg::Value(value),
            item_ty: ValueType::Int,
            array_type_id: None,
            nolength: false,
            gckind: crate::model::ArrayGcKind::Gc,
        },
        false,
    );
    // The dead unit-`()` value the entry block also builds.
    graph.push_op_var(entry, unit_ctor(), true);

    let forwarding = graph.create_block();
    // The forwarding block reuses the predecessor's `Variable`, which is
    // what defeats positional liveness in the real graph.
    graph.block_mut(forwarding).inputargs = vec![array.clone()];
    graph.push_op_var(forwarding, unit_ctor(), true);
    graph.set_goto(entry, forwarding, vec![array.clone()]);
    graph.set_return(forwarding, None);

    simplify_lowered_graph(
        &mut graph,
        &std::collections::HashMap::new(),
        true,
        &std::collections::HashSet::new(),
    );

    let entry_exit = &graph.block(entry).exits[0];
    assert_eq!(
        entry_exit.target, graph.returnblock,
        "the emptied forwarding block must be rewired past"
    );
    assert!(
        !entry_exit
            .args
            .iter()
            .any(|arg| arg.as_variable() == Some(&array)),
        "the array must not survive as a dead link arg: {:?}",
        entry_exit.args
    );
    assert!(
        graph.block(forwarding).inputargs.is_empty(),
        "the orphaned block must be cleared, not left naming the array: {:?}",
        graph.block(forwarding).inputargs
    );
}

fn pair_field_attrs() -> std::collections::HashMap<String, Vec<(String, ValueType)>> {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert(
        "Pair".to_string(),
        vec![
            ("a".to_string(), ValueType::Int),
            ("b".to_string(), ValueType::Int),
        ],
    );
    attrs
}

fn push_pair_ctor(graph: &mut FunctionGraph, block: crate::model::BlockId) -> Variable {
    graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["pair".to_string()],
                    "Pair",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some("Pair".to_string())),
            },
            true,
        )
        .expect("pair ctor")
}

fn push_pair_write(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    base: &Variable,
    field: &str,
    value: Variable,
) {
    graph.push_op_var(
        block,
        OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor::new(field, Some("Pair".to_string())),
            value: LinkArg::Value(value),
            ty: ValueType::Int,
        },
        false,
    );
}

fn push_pair_read(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    base: &Variable,
    field: &str,
) -> Variable {
    graph
        .push_op_var(
            block,
            OpKind::FieldRead {
                base: base.clone(),
                field: FieldDescriptor::new(field, Some("Pair".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        )
        .expect("field read")
}

fn graph_has_ctor_or_new(graph: &FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::New { .. } => true,
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        is_struct: true, ..
                    },
                ..
            } => true,
            _ => false,
        })
}

fn same_as_operand(graph: &FunctionGraph, result: &Variable) -> Option<Variable> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::UnaryOp {
                op: name, operand, ..
            } if name == "same_as" && op.result.as_ref() == Some(result) => Some(operand.clone()),
            _ => None,
        })
}

/// `malloc.py` `flowin` across one link: the field value is the new link
/// arg, and the successor read is an alias of that input.
#[test]
fn malloc_removal_follows_a_link() {
    let mut graph = FunctionGraph::new("malloc_removal_link");
    let entry = graph.startblock;
    let x = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![x.clone()];
    let agg = push_pair_ctor(&mut graph, entry);
    push_pair_write(&mut graph, entry, &agg, "a", x.clone());
    let (next, _) = graph.create_block_with_arg_vars(0);
    graph.block_mut(next).inputargs = vec![agg.clone()];
    let read = push_pair_read(&mut graph, next, &agg, "a");
    graph.set_goto(entry, next, vec![agg]);
    graph.set_return(next, Some(read.clone()));

    scalar_replace_named_struct_aggregates(&mut graph, &pair_field_attrs());

    assert!(!graph_has_ctor_or_new(&graph));
    assert_eq!(
        graph.block(entry).exits[0].args.first(),
        Some(&LinkArg::Value(x))
    );
    let field_input = graph.block(next).inputargs.first().cloned();
    assert_eq!(same_as_operand(&graph, &read), field_input);
}

/// Two mallocs of one struct join. The join reads a per-field input.
#[test]
fn malloc_removal_merges_two_ctors_at_a_join() {
    let mut graph = FunctionGraph::new("malloc_removal_join");
    let entry = graph.startblock;
    let x = graph.alloc_value_var();
    let y = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![x.clone(), y.clone()];
    let cond = graph
        .push_op_var(entry, OpKind::ConstBool(true), true)
        .expect("cond");
    let (left, _) = graph.create_block_with_arg_vars(0);
    let (right, _) = graph.create_block_with_arg_vars(0);
    let (join, _) = graph.create_block_with_arg_vars(0);
    let agg_l = push_pair_ctor(&mut graph, left);
    push_pair_write(&mut graph, left, &agg_l, "a", x);
    let agg_r = push_pair_ctor(&mut graph, right);
    push_pair_write(&mut graph, right, &agg_r, "a", y.clone());
    graph.block_mut(join).inputargs = vec![agg_l.clone()];
    let read = push_pair_read(&mut graph, join, &agg_l, "a");
    graph.set_goto(left, join, vec![agg_l]);
    graph.set_goto(right, join, vec![agg_r]);
    graph.set_return(join, Some(read.clone()));
    graph.block_mut(entry).exitswitch = Some(crate::model::ExitSwitch::Value(cond));
    graph.block_mut(entry).exits = vec![
        crate::model::Link::from_variables(
            &graph,
            vec![],
            left,
            Some(crate::model::ExitCase::Bool(true)),
        )
        .with_prevblock(entry),
        crate::model::Link::from_variables(
            &graph,
            vec![],
            right,
            Some(crate::model::ExitCase::Bool(false)),
        )
        .with_prevblock(entry),
    ];

    scalar_replace_named_struct_aggregates(&mut graph, &pair_field_attrs());

    assert!(!graph_has_ctor_or_new(&graph));
    assert_eq!(graph.block(join).inputargs.len(), 2);
    assert_eq!(
        graph.block(left).exits[0]
            .args
            .first()
            .and_then(LinkArg::as_variable),
        graph.block(entry).inputargs.first()
    );
    assert_eq!(
        graph.block(right).exits[0].args.first(),
        Some(&LinkArg::Value(y))
    );
    assert_eq!(
        same_as_operand(&graph, &read),
        graph.block(join).inputargs.first().cloned()
    );
}

/// A returned aggregate escapes. `foreign_field_ops` keeps the `New` fallback.
#[test]
fn malloc_removal_keeps_an_escaping_aggregate() {
    let mut graph = FunctionGraph::new("malloc_removal_escape");
    let entry = graph.startblock;
    let x = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![x.clone()];
    let agg = push_pair_ctor(&mut graph, entry);
    push_pair_write(&mut graph, entry, &agg, "a", x);
    let (next, _) = graph.create_block_with_arg_vars(0);
    graph.block_mut(next).inputargs = vec![agg.clone()];
    push_pair_read(&mut graph, next, &agg, "a");
    graph.set_goto(entry, next, vec![agg.clone()]);
    graph.set_return(next, Some(agg));

    scalar_replace_named_struct_aggregates(&mut graph, &pair_field_attrs());

    assert!(graph_has_ctor_or_new(&graph));
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| { matches!(op.kind, OpKind::New { ref owner } if owner == "Pair") })
    );
}

/// `has_gcstruct_a_vtable(OBJECT)` is false: the header type itself
/// (first field is the typeptr) lowers to plain `new`.
#[test]
fn malloc_removal_header_type_is_plain_new() {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert(
        "PyObject".to_string(),
        vec![
            ("ob_type".to_string(), ValueType::Ref(None)),
            ("payload".to_string(), ValueType::Int),
        ],
    );
    let mut graph = FunctionGraph::new("malloc_removal_header_new");
    let entry = graph.startblock;
    let payload = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![payload.clone()];
    let agg = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["pyobject".to_string()],
                    "PyObject",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some("PyObject".to_string())),
            },
            true,
        )
        .expect("ctor");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: agg.clone(),
            field: FieldDescriptor::new("payload", Some("PyObject".to_string())),
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    let (next, _) = graph.create_block_with_arg_vars(0);
    graph.block_mut(next).inputargs = vec![agg.clone()];
    graph.set_goto(entry, next, vec![agg.clone()]);
    graph.set_return(next, Some(agg));

    scalar_replace_named_struct_aggregates(&mut graph, &attrs);

    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(op.kind, OpKind::New { ref owner } if owner == "PyObject"))
    );
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(op.kind, OpKind::NewWithVtable { .. }))
    );
}

/// `rewrite_op_malloc`: first-field chain of inlined structs reaches
/// the typeptr header → `new_with_vtable`.
#[test]
fn malloc_removal_stamps_nested_header_typeptr_as_new_with_vtable() {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert(
        "PyObject".to_string(),
        vec![("ob_type".to_string(), ValueType::Ref(None))],
    );
    attrs.insert(
        "Boxed".to_string(),
        vec![
            (
                "ob_header".to_string(),
                ValueType::Ref(Some("PyObject".into())),
            ),
            ("payload".to_string(), ValueType::Int),
        ],
    );
    let mut graph = FunctionGraph::new("malloc_removal_nested_typeptr");
    let entry = graph.startblock;
    let payload = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![payload.clone()];
    let cls = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1020_3040), true)
        .expect("cls");
    let header = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["pyobject".to_string()],
                    "PyObject",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some("PyObject".to_string())),
            },
            true,
        )
        .expect("header");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: header.clone(),
            field: FieldDescriptor::new("ob_type", Some("PyObject".to_string())),
            value: LinkArg::Value(cls),
            ty: ValueType::Ref(None),
        },
        false,
    );
    let agg = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["boxed".to_string()],
                    "Boxed",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some("Boxed".to_string())),
            },
            true,
        )
        .expect("ctor");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: agg.clone(),
            field: FieldDescriptor::new("ob_header", Some("Boxed".to_string())),
            value: LinkArg::Value(header),
            ty: ValueType::Ref(Some("PyObject".to_string())),
        },
        false,
    );
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: agg.clone(),
            field: FieldDescriptor::new("payload", Some("Boxed".to_string())),
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    let (next, _) = graph.create_block_with_arg_vars(0);
    graph.block_mut(next).inputargs = vec![agg.clone()];
    graph.set_goto(entry, next, vec![agg.clone()]);
    graph.set_return(next, Some(agg));

    scalar_replace_named_struct_aggregates(&mut graph, &attrs);

    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::NewWithVtable { owner, .. } if owner == "Boxed"
                )
            }),
        "first-field chain to typeptr header must become NewWithVtable"
    );
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(op.kind, OpKind::New { ref owner } if owner == "Boxed")),
        "vtable-bearing Boxed must not lower to plain New"
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "ob_type"
            )),
        "the typeptr store stays until fuse_boxing_alloc / rewrite_op_setfield"
    );
}

/// A vtable-bearing aggregate can also carry a flattened `ob_type` write
/// (`Function.ob` is the nested header, and a field write names `ob_type`).
/// `emit_materialized_struct_copy` must keep that store so
/// `fuse_boxing_alloc` / `resolve_header_plan` still sees the constant
/// type word.
#[test]
fn malloc_removal_keeps_a_flattened_typeptr_store_on_a_vtable_copy() {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert(
        "PyObject".to_string(),
        vec![("ob_type".to_string(), ValueType::Ref(None))],
    );
    attrs.insert(
        "Function".to_string(),
        vec![
            ("ob".to_string(), ValueType::Ref(Some("PyObject".into()))),
            ("code".to_string(), ValueType::Ref(None)),
        ],
    );
    let mut graph = FunctionGraph::new("malloc_removal_flat_typeptr");
    let entry = graph.startblock;
    let code = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![code.clone()];
    let cls = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1020_3040), true)
        .expect("cls");
    let agg = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["function".to_string()],
                    "Function",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some("Function".to_string())),
            },
            true,
        )
        .expect("ctor");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: agg.clone(),
            field: FieldDescriptor::new("ob_type", Some("Function".to_string())),
            value: LinkArg::Value(cls),
            ty: ValueType::Ref(None),
        },
        false,
    );
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: agg.clone(),
            field: FieldDescriptor::new("code", Some("Function".to_string())),
            value: LinkArg::Value(code),
            ty: ValueType::Ref(None),
        },
        false,
    );
    let (next, _) = graph.create_block_with_arg_vars(0);
    graph.block_mut(next).inputargs = vec![agg.clone()];
    graph.set_goto(entry, next, vec![agg.clone()]);
    graph.set_return(next, Some(agg));

    scalar_replace_named_struct_aggregates(&mut graph, &attrs);

    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::NewWithVtable { owner, .. } if owner == "Function"
                )
            }),
        "Function's nested header makes the ctor new_with_vtable"
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "ob_type"
            )),
        "flattened ob_type store must survive scalar_replace"
    );
}

/// A field that is never stored is the typed zero (`malloc.py` `flatconstants`).
#[test]
fn malloc_removal_reads_zero_for_an_unwritten_field() {
    let mut graph = FunctionGraph::new("malloc_removal_zero");
    let entry = graph.startblock;
    let x = graph.alloc_value_var();
    graph.block_mut(entry).inputargs = vec![x.clone()];
    let agg = push_pair_ctor(&mut graph, entry);
    push_pair_write(&mut graph, entry, &agg, "a", x);
    let (next, _) = graph.create_block_with_arg_vars(0);
    graph.block_mut(next).inputargs = vec![agg.clone()];
    let read = push_pair_read(&mut graph, next, &agg, "b");
    graph.set_goto(entry, next, vec![agg]);
    graph.set_return(next, Some(read));

    scalar_replace_named_struct_aggregates(&mut graph, &pair_field_attrs());

    assert!(!graph_has_ctor_or_new(&graph));
    let zero = graph.block(entry).exits[0].args.get(1).cloned();
    assert_eq!(
        zero,
        Some(LinkArg::Const(crate::flowspace::model::Constant::new(
            crate::flowspace::model::ConstValue::Int(0)
        )))
    );
}

/// `execute_generator_frame` residualizes `get_eval_fn` as `indirect_call`.
/// The resume struct is the argument; a stack alias would be the walker's
/// C frame. Materialise `New` at that escape the same way `Call` does.
#[test]
fn aggregate_passed_by_value_to_an_indirect_call() {
    let mut graph = FunctionGraph::new("struct_ctor_indirect");
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
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    let funcptr = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
        .expect("funcptr");
    graph.push_op_var(
        entry,
        OpKind::IndirectCall {
            funcptr,
            args: vec![result],
            graphs: None,
            family_key: None,
            result_ty: ValueType::Int,
        },
        true,
    );
    graph.set_return(entry, None);

    assert_eq!(replace_struct_ctors(&mut graph), 1);
    assert_eq!(struct_ctor_ops(&graph), (0, 1, 1));
    let news = struct_news(&graph);
    assert_eq!(news.len(), 1);
    let call_arg = graph
        .block(graph.startblock)
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::IndirectCall { args, .. } => args.first().cloned(),
            _ => None,
        });
    assert_eq!(call_arg.as_ref(), Some(&news[0]));
}

/// `FunctionReprBase.call` builds one `vlist` for the single
/// `indirect_call`. The same aggregate in two argument positions is
/// one object; a second aggregate is a second object.
#[test]
fn aggregate_passed_twice_to_an_indirect_call_shares_one_copy() {
    let mut graph = FunctionGraph::new("struct_ctor_indirect_dup");
    let entry = graph.startblock;
    let owner = "error::DictKeyError";
    let first = graph
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
        .expect("first struct ctor");
    let first_payload = graph
        .push_op_var(entry, OpKind::ConstInt(1), true)
        .expect("first payload");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: first.clone(),
            field: FieldDescriptor::new("kind", Some(owner.to_string())),
            value: LinkArg::Value(first_payload),
            ty: ValueType::Int,
        },
        false,
    );
    let second = graph
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
        .expect("second struct ctor");
    let second_payload = graph
        .push_op_var(entry, OpKind::ConstInt(2), true)
        .expect("second payload");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: second.clone(),
            field: FieldDescriptor::new("kind", Some(owner.to_string())),
            value: LinkArg::Value(second_payload),
            ty: ValueType::Int,
        },
        false,
    );
    let funcptr = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
        .expect("funcptr");
    graph.push_op_var(
        entry,
        OpKind::IndirectCall {
            funcptr,
            args: vec![first.clone(), first, second],
            graphs: None,
            family_key: None,
            result_ty: ValueType::Int,
        },
        true,
    );
    graph.set_return(entry, None);

    assert_eq!(replace_struct_ctors(&mut graph), 2);
    assert_eq!(struct_ctor_ops(&graph), (0, 2, 2));
    let news = struct_news(&graph);
    assert_eq!(news.len(), 2);
    let call_args = graph
        .block(graph.startblock)
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::IndirectCall { args, .. } => Some(args.clone()),
            _ => None,
        })
        .expect("indirect call");
    assert_eq!(call_args.len(), 3);
    assert_eq!(call_args[0], call_args[1]);
    assert_ne!(call_args[0], call_args[2]);
    assert!(news.contains(&call_args[0]));
    assert!(news.contains(&call_args[2]));
    assert_ne!(news[0], news[1]);
}

/// The successor parameter is the aggregate variable itself. The body
/// must read the materialized copy, not the constructor variable the
/// replacement removed.
#[test]
fn shared_inputarg_reads_the_materialized_struct_copy() {
    let mut graph = FunctionGraph::new("struct_ctor_shared_phi");
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
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    let (join, _) = graph.create_block_with_arg_vars(1);
    graph.block_mut(join).inputargs = vec![result.clone()];
    graph.push_op_var(
        join,
        OpKind::Call {
            target: CallTarget::function_path(["slice", "index"]),
            args: crate::model::call_args(vec![result.clone()]),
            result_ty: ValueType::Int,
        },
        true,
    );
    graph.set_goto(entry, join, vec![result.clone()]);
    graph.set_return(join, None);

    assert_eq!(replace_struct_ctors(&mut graph), 1);
    let news = struct_news(&graph);
    assert_eq!(news.len(), 1);
    let passed = graph.block(entry).exits[0]
        .args
        .first()
        .and_then(LinkArg::as_variable)
        .cloned();
    assert_eq!(passed.as_ref(), Some(&news[0]));
    let join_block = graph.block(join);
    let call_arg = join_block.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call { args, .. } => args.first().and_then(LinkArg::as_variable).cloned(),
        _ => None,
    });
    assert_eq!(call_arg.as_ref(), join_block.inputargs.first());
    assert_eq!(call_arg, passed);
}

/// Two edges into one shared parameter each pass their own copy, and the
/// join reads one phi rather than the first edge's object.
#[test]
fn two_edges_into_a_shared_inputarg_keep_both_copies() {
    let mut graph = FunctionGraph::new("struct_ctor_two_edges");
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
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    let cond = graph
        .push_op_var(entry, OpKind::ConstBool(true), true)
        .expect("cond");
    let (join, _) = graph.create_block_with_arg_vars(1);
    graph.block_mut(join).inputargs = vec![result.clone()];
    graph.push_op_var(
        join,
        OpKind::Call {
            target: CallTarget::function_path(["slice", "index"]),
            args: crate::model::call_args(vec![result.clone()]),
            result_ty: ValueType::Int,
        },
        true,
    );
    graph.set_return(join, None);
    graph.block_mut(entry).exitswitch = Some(crate::model::ExitSwitch::Value(cond));
    graph.block_mut(entry).exits = vec![
        crate::model::Link::from_variables(
            &graph,
            vec![result.clone()],
            join,
            Some(crate::model::ExitCase::Bool(true)),
        )
        .with_prevblock(entry),
        crate::model::Link::from_variables(
            &graph,
            vec![result.clone()],
            join,
            Some(crate::model::ExitCase::Bool(false)),
        )
        .with_prevblock(entry),
    ];

    assert_eq!(replace_struct_ctors(&mut graph), 1);
    let news = struct_news(&graph);
    assert_eq!(news.len(), 2);
    let passed: Vec<_> = graph
        .block(entry)
        .exits
        .iter()
        .map(|link| link.args.first().and_then(LinkArg::as_variable).cloned())
        .collect();
    assert_eq!(passed[0].as_ref(), Some(&news[0]));
    assert_eq!(passed[1].as_ref(), Some(&news[1]));
    assert_ne!(passed[0], passed[1]);
    let join_block = graph.block(join);
    let call_arg = join_block.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call { args, .. } => args.first().and_then(LinkArg::as_variable).cloned(),
        _ => None,
    });
    assert_eq!(call_arg.as_ref(), join_block.inputargs.first());
    assert_ne!(call_arg, passed[0]);
    assert_ne!(call_arg, passed[1]);
}

/// Two by-value copies of one aggregate are two allocations. Mutating
/// one copy's field does not write the other.
#[test]
fn by_value_copy_then_mutation_does_not_change_the_other() {
    let mut graph = FunctionGraph::new("struct_ctor_copy");
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
    let (next, args) = graph.create_block_with_arg_vars(2);
    let orig = args[0].clone();
    let copy = args[1].clone();
    graph.push_op_var(
        next,
        OpKind::FieldWrite {
            base: copy.clone(),
            field: FieldDescriptor::new("kind", Some(owner.to_string())),
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    graph.push_op_var(
        next,
        OpKind::FieldRead {
            base: orig.clone(),
            field: FieldDescriptor::new("kind", Some(owner.to_string())),
            ty: ValueType::Int,
            pure: true,
        },
        true,
    );
    graph.set_goto(entry, next, vec![result.clone(), result]);
    graph.set_return(next, None);

    assert_eq!(replace_struct_ctors(&mut graph), 1);
    let news = struct_news(&graph);
    assert_eq!(news.len(), 2);
    assert_ne!(news[0], news[1]);
    let entry = graph.block(graph.startblock);
    let link_args: Vec<&Variable> = entry.exits[0]
        .args
        .iter()
        .filter_map(LinkArg::as_variable)
        .collect();
    assert_eq!(link_args, vec![&news[0], &news[1]]);
    let join = &graph.blocks[entry.exits[0].target.0];
    assert_eq!(join.inputargs.len(), 2);
    assert_ne!(join.inputargs[0], join.inputargs[1]);
    let mutated = join.operations.iter().find_map(|op| match &op.kind {
        OpKind::FieldWrite { base, field, .. } if field.name == "kind" => Some(base.clone()),
        _ => None,
    });
    let read = join.operations.iter().find_map(|op| match &op.kind {
        OpKind::FieldRead { base, field, .. } if field.name == "kind" => Some(base.clone()),
        _ => None,
    });
    assert_eq!(mutated.as_ref(), Some(&join.inputargs[1]));
    assert_eq!(read.as_ref(), Some(&join.inputargs[0]));
    assert_ne!(mutated, read);
}

/// The rewrite is the last step of the final simplify, after boxing fusion
/// and the consumer rewrites that still match a constructor. The pre-pass
/// simplify must not steal those constructors.
#[test]
fn final_simplify_rewrites_struct_ctors_prepass_does_not() {
    let mut graph = struct_ctor_graph(false);
    simplify_lowered_graph(
        &mut graph,
        &empty_struct_attrs(),
        false,
        &std::collections::HashSet::new(),
    );
    assert_eq!(struct_ctor_ops(&graph), (1, 0, 1));
    simplify_lowered_graph(
        &mut graph,
        &empty_struct_attrs(),
        true,
        &std::collections::HashSet::new(),
    );
    assert_eq!(struct_ctor_ops(&graph), (0, 0, 0));
    assert_eq!(same_as_count(&graph), 0);
}

/// A field-less struct has no size descr (`all_fielddescrs` is empty),
/// so a by-value escape keeps the constructor rather than allocating.
#[test]
fn field_less_struct_ctor_is_not_allocated() {
    let mut graph = FunctionGraph::new("unit_struct");
    let entry = graph.startblock;
    let owner = "ops::RangeFull";
    let result = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["ops".to_string()],
                    "RangeFull",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.to_string())),
            },
            true,
        )
        .expect("struct ctor");
    graph.set_return(entry, Some(result));
    assert_eq!(replace_struct_ctors(&mut graph), 0);
    let ctors = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor {
                        is_struct: true,
                        ..
                    },
                    ..
                }
            )
        })
        .count();
    let news = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(&op.kind, OpKind::New { .. }))
        .count();
    assert_eq!(ctors, 1);
    assert_eq!(news, 0);
}

/// A closure environment has no registered layout, so it is never a `New`.
#[test]
fn closure_env_ctor_is_not_allocated() {
    let mut graph = FunctionGraph::new("closure_ctor");
    let entry = graph.startblock;
    let owner = "body::closure";
    let result = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["body".to_string()],
                    "closure",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.to_string())),
            },
            true,
        )
        .expect("closure ctor");
    let payload = graph
        .push_op_var(entry, OpKind::ConstInt(1), true)
        .expect("capture");
    graph.push_op_var(
        entry,
        OpKind::FieldWrite {
            base: result.clone(),
            field: FieldDescriptor::new("capture", Some(owner.to_string())),
            value: LinkArg::Value(payload),
            ty: ValueType::Int,
        },
        false,
    );
    graph.set_return(entry, Some(result));
    assert_eq!(replace_struct_ctors(&mut graph), 0);
    let news = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(&op.kind, OpKind::New { .. }))
        .count();
    assert_eq!(news, 0);
}

#[test]
fn declined_disc_combinator_leaves_disconnected_blocks_that_cleanup_removes() {
    let mut graph = FunctionGraph::new("result_map_missing_closure");
    let entry = graph.startblock;
    let recv = graph
        .push_op_var(entry, OpKind::ConstInt(0), true)
        .expect("receiver");
    let result = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::method("map", Some("core::result::Result".to_string())),
                args: crate::model::call_args(vec![recv]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .expect("map");
    let cont = graph.create_block();
    graph.set_goto(entry, cont, vec![]);
    graph.set_return(cont, None);
    let site = super::DiscCombinatorSite {
        kind: super::DiscCombinator::ResultMap,
        result_var: result,
        call_args: Vec::new(),
        recv_owner: "core::result::Result".into(),
        recv_tag0_owner: "core::result::Result::Ok".into(),
        recv_tag1_owner: "core::result::Result::Err".into(),
        payload0_ty: ValueType::Int,
        payload1_ty: ValueType::Int,
        payload0_class: None,
        payload1_class: None,
        payload0_is_raw_u8: false,
        payload1_is_raw_u8: false,
        result_owner: "core::result::Result".into(),
        result_tag0_owner: "core::result::Result::Ok".into(),
        result_tag1_owner: "core::result::Result::Err".into(),
        result_payload0_ty: ValueType::Int,
        result_payload1_ty: ValueType::Int,
        result_payload0_class: None,
        result_payload1_class: None,
        call_once_owner: String::new(),
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
    };
    let live_before: std::collections::HashSet<_> =
        graph.blocks.iter().map(|block| block.id).collect();
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 0);
    let leftovers: Vec<_> = graph
        .blocks
        .iter()
        .filter(|block| !live_before.contains(&block.id))
        .map(|block| block.id)
        .collect();
    assert!(
        !leftovers.is_empty(),
        "a declined combinator site must leave the blocks it created"
    );
    crate::model::clear_unreachable_blocks(&mut graph);
    for id in leftovers {
        let block = graph.block(id);
        assert!(
            block.operations.is_empty() && block.exits.is_empty(),
            "clear_unreachable_blocks must empty disconnected decline leftovers"
        );
    }
}

/// Final simplify must not steal the cluster either: fusion still has
/// to see the construct-on-stack spelling after class addresses land.
#[test]
fn final_simplify_leaves_boxing_cluster_ctors_including_nested_header() {
    let mut graph = boxing_cluster_with_nested_header();
    simplify_lowered_graph(
        &mut graph,
        &std::collections::HashMap::new(),
        true,
        &std::collections::HashSet::new(),
    );
    assert_eq!(boxing_cluster_ctor_new_counts(&graph), (2, 0));
}

/// A declared-Raw ADT pointer is a raw address word (`history.py`
/// `getkind` of a raw `Ptr` is `"int"`), so
/// `dont_look_inside_return_token` is `"raw:Owner"` — the int-bank
/// token `raw_address_owner_root` names. An ordinary looked-inside
/// function leaves FUNC.RESULT unstamped; the CFG return of
/// `get_raw_owner` is a reference (`flatten.py` `make_return`).
/// Residual callees keep the token through `stamp_return_token`.
#[test]
fn ordinary_raw_adt_pointer_does_not_stamp_i64_func_result() {
    use std::collections::{HashMap, HashSet};
    let owner = struct_decl(0, ident_path(&["fixture", "Owner"]), i64_field(), false);
    let fun = |def_id: u64, leaf: &str, output: serde_json::Value| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(ident_path(&["fixture", leaf])),
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": output
            },
            "body": "Opaque"
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [owner],
            "fun_decls": [
                fun(0, "get_raw_owner", serde_json::json!({
                    "RawPtr": [
                        {"Adt": {"id": 0, "generics": {
                            "regions": [], "types": [],
                            "const_generics": [], "trait_refs": []
                        }, "builtin": null}},
                        "Const"
                    ]
                })),
                fun(1, "returns_i64", serde_json::json!({
                    "Scalar": {"Integer": {"Signed": "I64"}}
                }))
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let empty_gc: HashSet<majit_ir::descr::StructId> = HashSet::new();
    let empty_names: HashSet<String> = HashSet::new();
    let empty_hints: HashMap<String, Vec<String>> = HashMap::new();
    let spec = crate::ErrorCarrierSpec::default();
    let header_of = |fd: &majit_charon_reader::ullbc::FunDecl| {
        super::SemanticFunctionHeader::new(
            &llbc,
            fd,
            fd.item_meta
                .name_path()
                .rsplit("::")
                .next()
                .unwrap()
                .to_string(),
            "fixture".into(),
            &fd.signature,
            &empty_names,
            &empty_names,
            &empty_names,
            &empty_hints,
            spec,
            &empty_gc,
            &std::collections::HashSet::<String>::new(),
            &format!(
                "fixture::{}",
                fd.item_meta.name_path().rsplit("::").next().unwrap()
            ),
        )
    };

    let ptr_fd = llbc.fn_by_id(0).expect("get_raw_owner");
    assert_eq!(
        dont_look_inside_return_token(
            &ptr_fd.signature.output,
            &llbc,
            spec,
            &empty_gc,
            &std::collections::HashSet::<String>::new()
        )
        .as_deref(),
        Some("raw:Owner"),
        "declared-Raw ADT pointer stays an int-bank residual token"
    );
    let ptr_header = header_of(ptr_fd);
    assert_ne!(
        ptr_header.return_type.as_deref(),
        Some("i64"),
        "ordinary looked-inside declared-Raw ADT pointer must not stamp FUNC.RESULT=i64"
    );
    assert!(
        ptr_header.return_type.is_none(),
        "CFG kind r is kept by leaving FUNC.RESULT unstamped, got {:?}",
        ptr_header.return_type
    );

    let i64_fd = llbc.fn_by_id(1).expect("returns_i64");
    assert_eq!(
        dont_look_inside_return_token(
            &i64_fd.signature.output,
            &llbc,
            spec,
            &empty_gc,
            &std::collections::HashSet::<String>::new()
        )
        .as_deref(),
        Some("i64")
    );
    assert_eq!(
        header_of(i64_fd).return_type.as_deref(),
        Some("i64"),
        "a machine i64 result still stamps FUNC.RESULT"
    );
}

/// Loads the real interpreter LLBC to anchor fixed-array effect identity
/// to `_random::Random::genrand32`.
#[test]
#[ignore]
fn random_genrand32_fixed_array_keeps_concrete_identity() {
    use crate::model::{OpKind, ValueType};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with("::genrand32"))
        .expect("_random::Random::genrand32 in interpreter LLBC");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower Random::genrand32");

    let mut reads = 0usize;
    let mut writes = 0usize;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::ArrayRead {
                item_ty,
                array_type_id,
                nolength,
                ..
            } => {
                reads += 1;
                assert_eq!(*item_ty, ValueType::Unsigned);
                assert_eq!(array_type_id.as_deref(), Some("[u32;624]"));
                assert!(*nolength);
            }
            OpKind::ArrayWrite {
                item_ty,
                array_type_id,
                nolength,
                ..
            } => {
                writes += 1;
                assert_eq!(*item_ty, ValueType::Unsigned);
                assert_eq!(array_type_id.as_deref(), Some("[u32;624]"));
                assert!(*nolength);
            }
            _ => {}
        }
    }
    assert!(reads >= 1);
    assert!(writes >= 1);
}

/// `UnsafeCell::{new,get,get_mut}` on `RBigIntGcRoot` is the cell word.
/// Charon spells the callee `core::cell::<Impl>::new`, and the annotator
/// then sees `cell::UnsafeCell::new`. The lowering aliases that call to
/// its operand, so the graph must not keep the call.
#[test]
fn rbigint_gc_root_unsafecell_calls_are_word_identity() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    for name in [
        "majit_rlib::rbigint::gc::<Impl>::new",
        "majit_rlib::rbigint::gc::<Impl>::set",
    ] {
        let fd = llbc
            .iter_local_fns()
            .find(|fd| fd.item_meta.name_path() == name)
            .unwrap_or_else(|| panic!("{name} in majit-rlib LLBC"));
        let context = super::LowerContext::new(&llbc);
        let graph =
            super::lower_fun_decl(&context, fd).unwrap_or_else(|e| panic!("lower {name}: {e}"));
        for op in graph.blocks.iter().flat_map(|block| &block.operations) {
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            let hit = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    segments.iter().any(|s| s.contains("UnsafeCell"))
                }
                CallTarget::Method { receiver_root, .. } => receiver_root
                    .as_deref()
                    .is_some_and(|root| root.contains("UnsafeCell")),
                _ => false,
            };
            assert!(!hit, "{name} still calls {target:?}");
        }
    }
}

/// `s.flag.get()` of `S { flag: Cell<bool> }` is that field's value read.
/// The existing UnsafeCell identity test loads real majit-rlib LLBC and
/// cannot spell a struct field plus `Cell::get`.
#[test]
fn cell_get_of_a_struct_field_is_that_fields_value_read() {
    use crate::model::{CallTarget, OpKind, ValueType};

    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let ptr_ty = window_mut_u8_ptr();
    let cell_adt = serde_json::json!({
        "Adt": {
            "id": 1,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [bool_ty.clone()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cell_ptr_adt = serde_json::json!({
        "Adt": {
            "id": 1,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [ptr_ty.clone()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cell_ref = serde_json::json!({"Ref": ["Erased", cell_adt.clone(), "Shared"]});
    let cell_ptr_ref = serde_json::json!({"Ref": ["Erased", cell_ptr_adt.clone(), "Shared"]});
    let s_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let s_ref = serde_json::json!({"Ref": ["Erased", s_adt.clone(), "Shared"]});
    let s_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "S"], true),
        "kind": {"Struct": [
            {
                "name": "flag",
                "ty": cell_adt.clone(),
                "attr_info": null
            },
            {
                "name": "ptr",
                "ty": cell_ptr_adt.clone(),
                "attr_info": null
            }
        ]}
    });
    let cell_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["core", "cell", "Cell"], false),
        "kind": "Opaque"
    });
    let cell_get = window_opaque_fun(
        2,
        &["core", "cell", "Cell", "get"],
        vec![cell_ref.clone()],
        bool_ty.clone(),
    );
    let user_cell_get = window_opaque_fun(
        4,
        &["fixture", "Cell", "get"],
        vec![cell_ref.clone()],
        bool_ty.clone(),
    );
    let flag_place = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Projection": [
                window_place(serde_json::json!({"Local": 1}), &s_ref),
                "Deref"
            ]}), &s_adt),
            {"Field": [null, 0]}
        ]}),
        &cell_adt,
    );
    let ptr_place = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Projection": [
                window_place(serde_json::json!({"Local": 1}), &s_ref),
                "Deref"
            ]}), &s_adt),
            {"Field": [null, 1]}
        ]}),
        &cell_ptr_adt,
    );
    let (unwind, ret) = window_unwind_and_return_blocks();
    let call_get =
        |fun: u64, arg_ty: &serde_json::Value, arg_local: u64, dest_ty: &serde_json::Value| {
            serde_json::json!({"Call": {"call": {
            "func": {"Regular": {
                "kind": {"Fun": fun},
                "generics": window_empty_generics()
            }},
            "args": [{"Copy": window_place(serde_json::json!({"Local": arg_local}), arg_ty)}],
            "dest": window_place(serde_json::json!({"Local": 0}), dest_ty)
        }, "target": 2, "on_unwind": 1}})
        };
    let f = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "f"], true),
        "signature": {"is_unsafe": false, "inputs": [s_ref.clone()], "output": bool_ty.clone()},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &bool_ty),
                window_local(1, &s_ref),
                window_local(2, &cell_ref)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &cell_ref),
                            {"Ref": {
                                "place": flag_place.clone(),
                                "kind": "Shared",
                                "ptr_metadata": null
                            }}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": call_get(2, &cell_ref, 2, &bool_ty)}
                },
                unwind.clone(),
                ret.clone()
            ]
        }}
    });
    let g = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["fixture", "g"], true),
        "signature": {"is_unsafe": false, "inputs": [cell_ref.clone()], "output": bool_ty.clone()},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &bool_ty),
                window_local(1, &cell_ref)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": call_get(2, &cell_ref, 1, &bool_ty)}
                },
                unwind.clone(),
                ret.clone()
            ]
        }}
    });
    let h = serde_json::json!({
        "def_id": 3,
        "item_meta": window_meta(&["fixture", "h"], true),
        "signature": {"is_unsafe": false, "inputs": [s_ref.clone()], "output": ptr_ty.clone()},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &ptr_ty),
                window_local(1, &s_ref),
                window_local(2, &cell_ptr_ref)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &cell_ptr_ref),
                            {"Ref": {
                                "place": ptr_place,
                                "kind": "Shared",
                                "ptr_metadata": null
                            }}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": call_get(2, &cell_ptr_ref, 2, &ptr_ty)}
                },
                unwind.clone(),
                ret.clone()
            ]
        }}
    });
    let user_get = serde_json::json!({
        "def_id": 5,
        "item_meta": window_meta(&["fixture", "user_get"], true),
        "signature": {"is_unsafe": false, "inputs": [s_ref.clone()], "output": bool_ty.clone()},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &bool_ty),
                window_local(1, &s_ref),
                window_local(2, &cell_ref)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &cell_ref),
                            {"Ref": {
                                "place": flag_place,
                                "kind": "Shared",
                                "ptr_metadata": null
                            }}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": call_get(4, &cell_ref, 2, &bool_ty)}
                },
                unwind,
                ret
            ]
        }}
    });
    let llbc = window_llbc(
        vec![s_decl, cell_decl],
        vec![f, g, cell_get, h, user_cell_get, user_get],
    );
    let f_graph = super::lower_function(&llbc, "f").expect("lower f");
    let f_ops: Vec<_> = f_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();
    assert!(
        f_ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, ty, .. }
                if field.name == "flag" && !field.taken_by_address && *ty == ValueType::Bool
        )),
        "f must FieldRead flag as a bool value; graph={}",
        f_graph.dump()
    );
    let call_target_ends_with_get = |op: &crate::model::SpaceOperation| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => segments.last().is_some_and(|leaf| leaf.ends_with("get")),
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } => name.ends_with("get"),
        _ => false,
    };
    assert!(
        f_ops.iter().all(|op| !call_target_ends_with_get(op)),
        "f must not keep Cell::get as a Call; graph={}",
        f_graph.dump()
    );
    let g_graph = super::lower_function(&llbc, "g").expect("lower g");
    assert!(
        g_graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(call_target_ends_with_get),
        "g(&Cell<bool>) keeps its Call; graph={}",
        g_graph.dump()
    );
    let h_graph = super::lower_function(&llbc, "h").expect("lower h");
    assert!(
        h_graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(call_target_ends_with_get),
        "h(s.ptr.get()) of Cell<*mut u8> keeps its Call; graph={}",
        h_graph.dump()
    );
    let user_graph = super::lower_function(&llbc, "user_get").expect("lower user_get");
    assert!(
        user_graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(call_target_ends_with_get),
        "a user Cell::get must keep its Call; graph={}",
        user_graph.dump()
    );
}

/// `live_rbigint` clones `&RBigInt` and passes the by-value result to
/// `RBigIntGcRoot::new`. The clone's destination is the `RBigInt` ADT,
/// which `tyref_to_value_type` paints `Ref(None)`. Without the
/// `__cast_instance_intrinsic` narrow, `annotationoftype` never calls
/// `getuniqueclassdef` and `SomeInstance.getattr("_digits")` sees
/// `classdef=None`.
#[test]
fn live_rbigint_clone_result_is_rbigint_instance() {
    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == "majit_rlib::rbigint::gc::live_rbigint")
        .expect("live_rbigint in majit-rlib LLBC");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower live_rbigint");
    let rooted = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| crate::model::cast_instance_root(&op.kind) == Some("rbigint::RBigInt"));
    assert!(
        rooted,
        "live_rbigint clone result must be cast to rbigint::RBigInt, got ops: {}",
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .map(|op| format!("{:?}", std::mem::discriminant(&op.kind)))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// `_int_bitwise_and` keeps its handles: one lives in an
/// `Option<RBigIntGcRoot>`. `live_rbigint`'s graph returns the value, so
/// the kept body roots it again with the real `RBigIntGcRoot::new`, which
/// keeps the body's `acquire_owner_root` decline.
#[test]
fn a_kept_handle_roots_the_producer_value_with_the_real_new() {
    use crate::model::{CallTarget, LinkArg, OpKind};

    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == "majit_rlib::rbigint::_int_bitwise_and")
        .expect("_int_bitwise_and in majit-rlib LLBC");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower _int_bitwise_and");
    let leaf_of = |op: &crate::model::SpaceOperation| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => segments.last().cloned(),
        _ => None,
    };
    let rerooted = graph.blocks.iter().any(|block| {
        let ops = &block.operations;
        ops.iter().enumerate().any(|(i, op)| {
            leaf_of(op).as_deref() == Some("live_rbigint")
                && ops[i + 1..].iter().any(|later| {
                    let OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        args,
                        ..
                    } = &later.kind
                    else {
                        return false;
                    };
                    segments.iter().any(|s| s == "RBigIntGcRoot")
                        && segments.last().is_some_and(|leaf| leaf == "new")
                        && args.len() == 1
                        && matches!(&args[0], LinkArg::Value(_))
                })
        })
    });
    assert!(
        rerooted,
        "live_rbigint's value is not rooted again by RBigIntGcRoot::new"
    );
    // The `bool::then` closure keeps its handle and returns it; the kept
    // caller stores it in the `Option<RBigIntGcRoot>` as a handle.
    let closure = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta.name_path() == "majit_rlib::rbigint::_int_bitwise_and::<Impl>::call_once"
        })
        .expect("_int_bitwise_and closure in majit-rlib LLBC");
    super::lower_fun_decl(&context, closure).expect("lower the _int_bitwise_and closure");
}

/// `live_rbigint` clones and roots the clone. Its handle only leaves
/// through the return place, so the graph returns the clone itself and
/// never enters `RBigIntGcRoot::new`, whose `acquire_owner_root` has no
/// JIT counterpart.
#[test]
fn live_rbigint_returns_the_rooted_value() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == "majit_rlib::rbigint::gc::live_rbigint")
        .expect("live_rbigint in majit-rlib LLBC");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower live_rbigint");
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        let names_handle = match target {
            CallTarget::FunctionPath { segments, .. } => segments
                .iter()
                .any(|s| s.contains("RBigIntGcRoot") || s.contains("owner_root")),
            CallTarget::Method { receiver_root, .. } => receiver_root
                .as_deref()
                .is_some_and(|root| root.contains("RBigIntGcRoot")),
            _ => false,
        };
        assert!(!names_handle, "live_rbigint still calls {target:?}");
    }
}

/// `ll_list_obj_setitem_fast` is `items_block_set_ref(l.items, index,
/// item)`. The call becomes the one `setarrayitem_gc` on the object
/// items array, the same collapse `FixedObjectArray::set_ref` gets.
/// The ArrayWrite base is the object-strategy unerase of the
/// `ItemsBlock` (`rerased.py` / `ListStrategy.unerase`).
#[test]
fn items_block_set_ref_call_is_one_array_write() {
    use crate::model::OpKind;
    let graph = lower_object_fn("pyre_object::listobject::ll_list_obj_setitem_fast");
    assert!(
        !calls_leaf(&graph, "items_block_set_ref"),
        "ll_list_obj_setitem_fast still calls items_block_set_ref"
    );
    let mut writes = Vec::new();
    let mut object_items_casts = Vec::new();
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        if crate::model::cast_instance_root(&op.kind) == Some("[pyobject::PyObject]") {
            if let Some(result) = op.result.as_ref() {
                object_items_casts.push(result.clone());
            }
        }
        if let OpKind::ArrayWrite {
            base,
            item_ty,
            array_type_id,
            ..
        } = &op.kind
        {
            writes.push((base.clone(), item_ty.clone(), array_type_id.clone()));
        }
    }
    assert_eq!(
        writes
            .iter()
            .map(|(_, item_ty, array_type_id)| (item_ty.clone(), array_type_id.clone()))
            .collect::<Vec<_>>(),
        vec![(
            crate::model::ValueType::Ref(None),
            Some(super::OBJECT_REF_GCARRAY_TYPE_ID.to_string())
        )],
        "expected one object-array setarrayitem"
    );
    assert!(
        object_items_casts.contains(&writes[0].0),
        "ArrayWrite base must be the [PyObject] unerase of the ItemsBlock"
    );
}

/// `roots[i] = GcRef(digits as usize)` must store `Ptr(GCREF)`, the
/// same word as `GcRef::NULL` (`ConstRefNull`). The transparent
/// `usize` field would otherwise store `Integer` and
/// `listdef.generalize` refuses `Ptr(GCREF) ∪ Integer`.
#[test]
fn gcref_address_cast_store_keeps_gcref_pointer() {
    use crate::model::OpKind;

    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta.name_path() == "majit_rlib::rbigint::<Impl>::with_size_reloading_digits"
        })
        .expect("with_size_reloading_digits");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower with_size_reloading_digits");
    let mut gcref_casts = 0;
    let mut pointer_stores = 0;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        if crate::model::cast_instance_root(&op.kind) == Some("GCREF") {
            gcref_casts += 1;
        }
        if let OpKind::ArrayWrite {
            item_ty: crate::model::ValueType::Ref(_),
            ..
        } = &op.kind
        {
            pointer_stores += 1;
        }
    }
    assert!(
        gcref_casts >= 2,
        "address-cast GcRef construction must retype to GCREF, casts={gcref_casts}"
    );
    assert!(
        pointer_stores >= 2,
        "GcRef list stores must be pointer words, pointer_stores={pointer_stores}"
    );
}

/// `W_Random::rnd` is `Random::from_obj(self.rnd).expect(...)`.
/// The `Option<&mut Random>` consumer must lower to the niche guard;
/// a residual `expect` method is `getattr("expect")` on the instance.
#[test]
fn random_rnd_expect_is_niche_guard() {
    use crate::model::{CallTarget, OpKind, ValueType};

    // `_random` lives in `pyre-module`, so its graphs are in that artefact.
    let path = crate::runtime_names::artifacts::MODULE_ULLBC;
    let llbc = Llbc::load(path).expect("load pyre-module LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == "pyre_module::module::_random::<Impl>::rnd")
        .expect("W_Random::rnd");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower rnd");
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } = &op.kind
        {
            assert_ne!(name.as_str(), "expect", "rnd still calls expect");
        }
        if let OpKind::FieldRead { field, .. } = &op.kind {
            assert_ne!(
                field.name.as_str(),
                "__discriminant",
                "niche Option<&mut Random> has no aggregate tag"
            );
        }
    }
    // The expect rewrite builds `not (opt is None)` (`rnone.py`
    // `rtype_is_None`); `remove_assertion_errors` then drops the
    // never-None arm, so the discriminant (and its `Constant(None)`)
    // is dead. The surviving Some path is `from_obj` narrowed to
    // Random. A GCREF `null_mut` operand is the collapse this
    // producer no longer emits.
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
                } if segments.as_slice() == ["core", "ptr", "null_mut"]
            )),
        "niche Option::expect is rtype_is_None, not a GCREF null_mut operand"
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty: ValueType::Ref(Some(root)),
                    ..
                } if segments.as_slice() == ["__cast_instance_intrinsic"]
                    && root.as_str() == "module::_random::Random"
            )),
        "from_obj's Option<&mut Random> narrows to the Random instance"
    );
}

#[test]
#[ignore]
fn random_wrapper_narrows_nullable_self_to_w_random() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with("::__majit_wrap_random"))
        .expect("_random::__majit_wrap_random");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower wrapper");
    let receiver = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                args,
                ..
            } if name == "random" => args.first().cloned(),
            _ => None,
        })
        .expect("random receiver");
    // The receiver reaches the `random` call across a block link, which
    // renames the value: `Variable` equality is object identity, so a flat
    // scan for an op whose `result` is this variable finds nothing.
    // `resolve_to_producer_op` walks the links back to the producing op —
    // the same resolver `extract_fmt_chain` opens with.
    let (block_id, idx) =
        super::resolve_to_producer_op(&graph, &receiver).expect("random receiver producer");
    let producer = &graph
        .blocks
        .iter()
        .find(|block| block.id == block_id)
        .expect("producer block")
        .operations[idx];
    assert_eq!(
        crate::model::cast_instance_root(&producer.kind),
        Some("W_Random")
    );
}

#[test]
fn resolve_to_producer_op_follows_cross_block_inputarg_link() {
    use super::resolve_to_producer_op;
    use crate::flowspace::model::Variable;
    use crate::model::{FunctionGraph, Link, OpKind, SpaceOperation};

    let mut graph = FunctionGraph::new("xblock");
    let a = graph.create_block();
    let b = graph.create_block();

    // block A produces `x` via a ConstInt op.
    let x = Variable::new();
    graph.block_mut(a).operations.push(SpaceOperation {
        result: Some(x.clone()),
        kind: OpKind::ConstInt(7),
    });

    // block B takes `w` as its single inputarg; A links to B passing
    // `x` for `w` (the cross-block threading the recognizer sees).
    let w = Variable::new();
    graph.block_mut(b).inputargs = vec![w.clone()];
    let link = Link::from_variables(&graph, vec![x.clone()], b, None).with_prevblock(a);
    graph.block_mut(a).exits = vec![link];

    // `w` (block B inputarg) back-traces to A's ConstInt op.
    assert_eq!(resolve_to_producer_op(&graph, &w), Some((a, 0)));
    // A direct op result resolves to itself.
    assert_eq!(resolve_to_producer_op(&graph, &x), Some((a, 0)));
    // An unrelated free Variable has no producer.
    assert_eq!(resolve_to_producer_op(&graph, &Variable::new()), None);
}

fn vec_alloc_and_set_r(result: Variable) -> SpaceOperation {
    use majit_ir::rvec::{VecItemKind, VecOp, vec_helper_path};
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::function_path(
                vec_helper_path(VecOp::AllocAndSet, VecItemKind::Ref).split("::"),
            ),
            args: crate::model::call_args(vec![Variable::new(), Variable::new()]),
            result_ty: ValueType::Int,
        },
    }
}

fn vec_call_result(result: Variable, leaf: &str) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::function_path(["other", leaf]),
            args: vec![],
            result_ty: ValueType::Int,
        },
    }
}

/// `rewrite_op_free` frees a header that reached the drop through a
/// return/unwind phi: both incoming Links carry the same in-graph
/// `ll_vec_alloc_and_set` result, so the drop emits `ll_vec_free`.
#[test]
fn rust_vec_drop_frees_phi_merged_header() {
    use super::{emit_rust_vec_drop_if_allocated, rust_vec_header_allocated_in_graph};
    use crate::model::Link;
    use majit_ir::rvec::VecItemKind;

    let mut graph = FunctionGraph::new("phi_vec_drop");
    let alloc_bb = graph.create_block();
    let ok_bb = graph.create_block();
    let err_bb = graph.create_block();
    let drop_bb = graph.create_block();

    let allocated = Variable::new();
    graph
        .block_mut(alloc_bb)
        .operations
        .push(vec_alloc_and_set_r(allocated.clone()));

    let ok_h = Variable::new();
    graph.block_mut(ok_bb).inputargs = vec![ok_h.clone()];
    let err_h = Variable::new();
    graph.block_mut(err_bb).inputargs = vec![err_h.clone()];
    let drop_h = Variable::new();
    graph.block_mut(drop_bb).inputargs = vec![drop_h.clone()];

    graph.block_mut(alloc_bb).exits = vec![
        Link::from_variables(&graph, vec![allocated.clone()], ok_bb, None).with_prevblock(alloc_bb),
        Link::from_variables(&graph, vec![allocated.clone()], err_bb, None)
            .with_prevblock(alloc_bb),
    ];
    graph.block_mut(ok_bb).exits =
        vec![Link::from_variables(&graph, vec![ok_h.clone()], drop_bb, None).with_prevblock(ok_bb)];
    graph.block_mut(err_bb).exits = vec![
        Link::from_variables(&graph, vec![err_h.clone()], drop_bb, None).with_prevblock(err_bb),
    ];

    // The drop header is a two-predecessor phi: `resolve_to_producer_op`
    // has no single producer, which is the leak the SSA walk closes.
    assert_eq!(super::resolve_to_producer_op(&graph, &drop_h), None);
    assert!(rust_vec_header_allocated_in_graph(&graph, &drop_h));
    assert!(emit_rust_vec_drop_if_allocated(
        &mut graph,
        drop_bb,
        &drop_h,
        VecItemKind::Ref,
    ));
    let leaves = graph_call_leaves(&graph);
    assert!(
        leaves.iter().any(|leaf| leaf == "ll_vec_free_r"),
        "phi-merged in-graph header must be freed: {leaves:?}"
    );
}

/// A header that is only a call result in this graph is not this
/// graph's `raw_malloc` allocation; `rewrite_op_free` of it would
/// free a pointer another graph (or a residual) owns.
#[test]
fn rust_vec_drop_does_not_free_call_result_header() {
    use super::{emit_rust_vec_drop_if_allocated, rust_vec_header_allocated_in_graph};
    use crate::model::Link;
    use majit_ir::rvec::VecItemKind;

    let mut graph = FunctionGraph::new("call_result_vec_drop");
    let call_bb = graph.create_block();
    let drop_bb = graph.create_block();

    let returned = Variable::new();
    graph
        .block_mut(call_bb)
        .operations
        .push(vec_call_result(returned.clone(), "returns_vec"));

    let drop_h = Variable::new();
    graph.block_mut(drop_bb).inputargs = vec![drop_h.clone()];
    graph.block_mut(call_bb).exits = vec![
        Link::from_variables(&graph, vec![returned.clone()], drop_bb, None).with_prevblock(call_bb),
    ];

    assert!(!rust_vec_header_allocated_in_graph(&graph, &drop_h));
    assert!(!rust_vec_header_allocated_in_graph(&graph, &returned));
    assert!(!emit_rust_vec_drop_if_allocated(
        &mut graph,
        drop_bb,
        &drop_h,
        VecItemKind::Ref,
    ));
    let leaves = graph_call_leaves(&graph);
    assert!(
        !leaves.iter().any(|leaf| leaf.starts_with("ll_vec_free")),
        "call-result header must not be freed by this graph: {leaves:?}"
    );
}

/// A phi that mixes an in-graph alloc with a call result cannot prove
/// ownership of the live header.
#[test]
fn rust_vec_drop_does_not_free_mixed_phi_header() {
    use super::{emit_rust_vec_drop_if_allocated, rust_vec_header_allocated_in_graph};
    use crate::model::Link;
    use majit_ir::rvec::VecItemKind;

    let mut graph = FunctionGraph::new("mixed_phi_vec_drop");
    let alloc_bb = graph.create_block();
    let call_bb = graph.create_block();
    let drop_bb = graph.create_block();

    let allocated = Variable::new();
    graph
        .block_mut(alloc_bb)
        .operations
        .push(vec_alloc_and_set_r(allocated.clone()));
    let returned = Variable::new();
    graph
        .block_mut(call_bb)
        .operations
        .push(vec_call_result(returned.clone(), "returns_vec"));

    let drop_h = Variable::new();
    graph.block_mut(drop_bb).inputargs = vec![drop_h.clone()];
    graph.block_mut(alloc_bb).exits = vec![
        Link::from_variables(&graph, vec![allocated.clone()], drop_bb, None)
            .with_prevblock(alloc_bb),
    ];
    graph.block_mut(call_bb).exits = vec![
        Link::from_variables(&graph, vec![returned.clone()], drop_bb, None).with_prevblock(call_bb),
    ];

    assert!(!rust_vec_header_allocated_in_graph(&graph, &drop_h));
    assert!(!emit_rust_vec_drop_if_allocated(
        &mut graph,
        drop_bb,
        &drop_h,
        VecItemKind::Ref,
    ));
    let leaves = graph_call_leaves(&graph);
    assert!(
        !leaves.iter().any(|leaf| leaf.starts_with("ll_vec_free")),
        "mixed phi must not free: {leaves:?}"
    );
}

/// A loop-carried copy of an in-graph `ll_vec_alloc_and_set` header is
/// the same raw pointer. The latch is visited before the alloc arm, so
/// its phi is a cycle; that cycle must not count as a foreign owner.
/// `rtype_free_non_gc_object` still emits `free`, and `rewrite_op_free`
/// rewrites flavor `raw` to `raw_free`.
#[test]
fn rust_vec_drop_frees_loop_carried_header() {
    use super::{emit_rust_vec_drop_if_allocated, rust_vec_header_allocated_in_graph};
    use crate::model::Link;
    use majit_ir::rvec::VecItemKind;

    let mut graph = FunctionGraph::new("loop_vec_drop");
    // Latch first so the header's incoming walk sees the back edge
    // before the allocating predecessor.
    let latch_bb = graph.create_block();
    let alloc_bb = graph.create_block();
    let header_bb = graph.create_block();
    let drop_bb = graph.create_block();

    let allocated = Variable::new();
    graph
        .block_mut(alloc_bb)
        .operations
        .push(vec_alloc_and_set_r(allocated.clone()));

    let carried = Variable::new();
    graph.block_mut(header_bb).inputargs = vec![carried.clone()];
    let latch_h = Variable::new();
    graph.block_mut(latch_bb).inputargs = vec![latch_h.clone()];
    let drop_h = Variable::new();
    graph.block_mut(drop_bb).inputargs = vec![drop_h.clone()];

    graph.block_mut(latch_bb).exits = vec![
        Link::from_variables(&graph, vec![latch_h.clone()], header_bb, None)
            .with_prevblock(latch_bb),
    ];
    graph.block_mut(alloc_bb).exits = vec![
        Link::from_variables(&graph, vec![allocated.clone()], header_bb, None)
            .with_prevblock(alloc_bb),
    ];
    graph.block_mut(header_bb).exits = vec![
        Link::from_variables(&graph, vec![carried.clone()], latch_bb, None)
            .with_prevblock(header_bb),
        Link::from_variables(&graph, vec![carried.clone()], drop_bb, None)
            .with_prevblock(header_bb),
    ];

    assert!(rust_vec_header_allocated_in_graph(&graph, &drop_h));
    assert!(rust_vec_header_allocated_in_graph(&graph, &latch_h));
    assert!(emit_rust_vec_drop_if_allocated(
        &mut graph,
        drop_bb,
        &drop_h,
        VecItemKind::Ref,
    ));
    let leaves = graph_call_leaves(&graph);
    assert!(
        leaves.iter().any(|leaf| leaf == "ll_vec_free_r"),
        "loop-carried in-graph header must be freed: {leaves:?}"
    );
}

/// A loop phi that also merges a call result is not wholly this
/// graph's malloc. `rewrite_op_free` must not run on that header.
#[test]
fn rust_vec_drop_does_not_free_loop_mixed_with_call_result() {
    use super::{emit_rust_vec_drop_if_allocated, rust_vec_header_allocated_in_graph};
    use crate::model::Link;
    use majit_ir::rvec::VecItemKind;

    let mut graph = FunctionGraph::new("loop_mixed_vec_drop");
    let latch_bb = graph.create_block();
    let alloc_bb = graph.create_block();
    let call_bb = graph.create_block();
    let header_bb = graph.create_block();
    let drop_bb = graph.create_block();

    let allocated = Variable::new();
    graph
        .block_mut(alloc_bb)
        .operations
        .push(vec_alloc_and_set_r(allocated.clone()));
    let returned = Variable::new();
    graph
        .block_mut(call_bb)
        .operations
        .push(vec_call_result(returned.clone(), "returns_vec"));

    let carried = Variable::new();
    graph.block_mut(header_bb).inputargs = vec![carried.clone()];
    let latch_h = Variable::new();
    graph.block_mut(latch_bb).inputargs = vec![latch_h.clone()];
    let drop_h = Variable::new();
    graph.block_mut(drop_bb).inputargs = vec![drop_h.clone()];

    graph.block_mut(latch_bb).exits = vec![
        Link::from_variables(&graph, vec![latch_h.clone()], header_bb, None)
            .with_prevblock(latch_bb),
    ];
    graph.block_mut(alloc_bb).exits = vec![
        Link::from_variables(&graph, vec![allocated.clone()], header_bb, None)
            .with_prevblock(alloc_bb),
    ];
    graph.block_mut(call_bb).exits = vec![
        Link::from_variables(&graph, vec![returned.clone()], latch_bb, None)
            .with_prevblock(call_bb),
    ];
    graph.block_mut(header_bb).exits = vec![
        Link::from_variables(&graph, vec![carried.clone()], latch_bb, None)
            .with_prevblock(header_bb),
        Link::from_variables(&graph, vec![carried.clone()], drop_bb, None)
            .with_prevblock(header_bb),
    ];

    assert!(!rust_vec_header_allocated_in_graph(&graph, &drop_h));
    assert!(!emit_rust_vec_drop_if_allocated(
        &mut graph,
        drop_bb,
        &drop_h,
        VecItemKind::Ref,
    ));
    let leaves = graph_call_leaves(&graph);
    assert!(
        !leaves.iter().any(|leaf| leaf.starts_with("ll_vec_free")),
        "loop mixed with a call result must not free: {leaves:?}"
    );
}

/// `&*(*p).f` reads the value the field holds; `&(*p).f` names the
/// field's address.
///
/// Both reach `build_rvalue`'s `Ref` / `RawPtr` arms as a
/// `PlaceKind::Projection`, so a bare `matches!(.., Projection(..))`
/// calls them both an address-of.  On `locals_cells_stack_w` the false
/// mark is not inert: `rewrite_op_getfield` reads
/// `suppresses_virtualizable` and skips the `vable_array_vars`
/// registration, so the read leaves the protocol and becomes a plain
/// `getarrayitem_gc` against a heap array that is only synchronised at
/// the three `sync_virtualizable_*` points — a stale read, not a slow
/// equivalent.  `locals_w!` expands to the deref-then-ref spelling, so
/// every one of its readers took that path.
#[test]
fn a_deref_last_projection_reads_a_value_rather_than_naming_an_address() {
    use super::Lowering;
    use majit_charon_reader::ullbc::{Place, PlaceKind, ProjectionElem};

    fn ty() -> TyRef {
        TyRef::Other(serde_json::Value::Null)
    }
    fn place(kind: PlaceKind) -> Place {
        Place { kind, ty: ty() }
    }
    fn field(inner: Place, name: &str) -> Place {
        place(PlaceKind::Projection(
            Box::new(inner),
            ProjectionElem::Tagged(serde_json::json!({ "Field": name })),
        ))
    }
    fn deref(inner: Place) -> Place {
        place(PlaceKind::Projection(
            Box::new(inner),
            ProjectionElem::Atom("Deref".to_string()),
        ))
    }

    let local = || place(PlaceKind::Local(0));
    let is_addr = Lowering::place_ref_is_address_of;

    // `&(*p).f` — the outermost step names a field, so the reference
    // stands for that field's address and the mark is owed.
    assert!(is_addr(&field(deref(local()), "locals_cells_stack_w")));

    // `&*(*p).f` — the outermost step dereferences what the field
    // holds, so the reference stands for the pointee.  The field read
    // underneath is an ordinary value read.
    assert!(!is_addr(&deref(field(
        deref(local()),
        "locals_cells_stack_w"
    ))));

    // `&*p`, the same shape one level in, and a bare local: neither
    // names a field address either.
    assert!(!is_addr(&deref(local())));
    assert!(!is_addr(&local()));

    // A nested field is still an address-of at its outermost step.
    assert!(is_addr(&field(field(deref(local()), "a"), "b")));
}

#[test]
fn extract_fmt_chain_recovers_pieces_and_args_cross_block() {
    use super::{FmtArgKind, extract_fmt_chain};
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, OpKind, SpaceOperation, ValueType,
    };

    // Reconstruct the real `format!("hi{}", ctx)` front-end shape:
    // block A builds the `Argument::new_display(&ctx)` through the
    // argument Tuple (`&(ctx,).0`); block B builds the args + pieces
    // arrays and `Arguments::new`. The args-array element is the
    // new_display result threaded across the A→B link, so extraction
    // must follow the cross-block inputarg back to it.
    let mut graph = FunctionGraph::new("fmt_chain");
    let a = graph.create_block();
    let b = graph.create_block();

    // ── block A: Argument::new_display(&ctx) via the arg Tuple ──
    let ctx = Variable::new();
    let tuple = Variable::new();
    graph.block_mut(a).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple".to_string())),
        },
    });
    graph.block_mut(a).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            value: crate::model::LinkArg::Value(ctx.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let arg_ref = Variable::new();
    graph.block_mut(a).operations.push(SpaceOperation {
        result: Some(arg_ref.clone()),
        kind: OpKind::FieldRead {
            base: tuple,
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let argument = Variable::new();
    graph.block_mut(a).operations.push(SpaceOperation {
        result: Some(argument.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    "fmt".to_string(),
                    "rt".to_string(),
                    "Argument".to_string(),
                    "new_display".to_string(),
                ],
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![arg_ref]),
            result_ty: ValueType::Ref(Some("Argument".to_string())),
        },
    });

    // block B takes the new_display result as its single inputarg.
    let arg_in = Variable::new();
    graph.block_mut(b).inputargs = vec![arg_in.clone()];
    let link = Link::from_variables(&graph, vec![argument], b, None).with_prevblock(a);
    graph.block_mut(a).exits = vec![link];

    // ── block B: args array, pieces array, Arguments::new ──
    let args_arr = Variable::new();
    graph.block_mut(b).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    graph.block_mut(b).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_arr.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
            value: crate::model::LinkArg::Value(arg_in),
            ty: ValueType::Ref(None),
        },
    });
    let pieces_arr = Variable::new();
    graph.block_mut(b).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    // `format!("hi{}")` packed template: [2, 'h', 'i', 0xC0, 0].
    for (i, byte) in [2i64, 104, 105, 0xC0, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(b).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(b).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: crate::model::LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(b).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    "fmt".to_string(),
                    "Arguments".to_string(),
                    "new".to_string(),
                ],
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![pieces_arr.clone(), args_arr]),
            result_ty: ValueType::Ref(Some("Arguments".to_string())),
        },
    });

    let chain = extract_fmt_chain(&graph, &fmt_args).expect("recognized fmt chain");
    assert_eq!(chain.pieces, vec!["hi".to_string(), String::new()]);
    assert_eq!(chain.args.len(), 1);
    assert_eq!(chain.args[0].kind, FmtArgKind::Display);
    // The recovered value is `ctx`, unwrapped through the arg Tuple.
    assert_eq!(chain.args[0].value.id(), ctx.id());

    // A non-`Arguments::new` producer is not recognized.
    assert!(extract_fmt_chain(&graph, &pieces_arr).is_none());
}

fn single_arg_display_fmt_fixture(
    ctx_input_root: Option<&str>,
) -> (
    crate::model::FunctionGraph,
    crate::model::BlockId,
    crate::model::BlockId,
    crate::model::BlockId,
    crate::flowspace::model::Variable,
    crate::flowspace::model::Variable,
    crate::flowspace::model::Variable,
    crate::flowspace::model::Variable,
) {
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    let mut graph = FunctionGraph::new("fmt_collapse");
    let b0 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();

    let ctx = Variable::new();
    graph.block_mut(b0).inputargs = vec![ctx.clone()];
    if let Some(root) = ctx_input_root {
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(ctx.clone()),
            kind: OpKind::Input {
                name: "ctx".to_string(),
                ty: ValueType::Ref(Some(root.to_string())),
                class_root: Some(root.to_string()),
            },
        });
    }
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple".to_string())),
        },
    });
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            value: LinkArg::Value(ctx.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let arg_ref = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(arg_ref.clone()),
        kind: OpKind::FieldRead {
            base: tuple,
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let new_display = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(new_display.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "rt", "Argument", "new_display"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![arg_ref]),
            result_ty: ValueType::Ref(None),
        },
    });
    let arg_in = Variable::new();
    graph.block_mut(bp).inputargs = vec![arg_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![new_display], bp, None).with_prevblock(b0)];

    // ── Bp: args array, pieces array, Arguments::new ──
    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_arr.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
            value: LinkArg::Value(arg_in.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, byte) in [2i64, 104, 105, 0xC0, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "Arguments", "new"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args], bf, None).with_prevblock(bp)];

    // ── Bf: alloc::fmt::format(args) → String ──
    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["alloc", "fmt", "format"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];
    (graph, b0, bp, bf, ctx, arg_in, fmt_args_in, formatted)
}

#[test]
fn collapse_fmt_chains_expands_single_arg_chain_to_str_concat() {
    use super::collapse_fmt_chains;
    use crate::model::{CallTarget, OpKind};

    let (mut graph, b0, bp, bf, ctx, arg_in, fmt_args_in, formatted) =
        single_arg_display_fmt_fixture(None);
    collapse_fmt_chains(&mut graph);

    // Bf's `alloc::fmt::format` is replaced by the orthodox
    // `"hi" + str(value)` expansion (pieces ["hi", ""]; the empty
    // trailing piece is skipped), reusing `formatted` on the final
    // `add` so the return link still forwards the String.
    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    assert_eq!(bf_block.operations.len(), 3);
    match &bf_block.operations[0].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } => {
            assert_eq!(segments, &vec!["__str_const".to_string(), "hi".to_string()]);
            assert!(args.is_empty());
        }
        other => panic!("Bf op[0] not a __str_const: {other:?}"),
    }
    match &bf_block.operations[1].kind {
        OpKind::UnaryOp { op, operand, .. } => {
            assert_eq!(op, "str");
            assert_eq!(operand.id(), fmt_args_in.id());
        }
        other => panic!("Bf op[1] not a str UnaryOp: {other:?}"),
    }
    match &bf_block.operations[2].kind {
        OpKind::BinOp { op, lhs, rhs, .. } => {
            assert_eq!(op, "add");
            assert_eq!(
                lhs.id(),
                bf_block.operations[0].result.as_ref().unwrap().id()
            );
            assert_eq!(
                rhs.id(),
                bf_block.operations[1].result.as_ref().unwrap().id()
            );
        }
        other => panic!("Bf op[2] not an add BinOp: {other:?}"),
    }
    assert_eq!(
        bf_block.operations[2].result.as_ref().unwrap().id(),
        formatted.id()
    );

    // The chain ops are gone: B0 keeps no Tuple/new_display ops, Bp
    // keeps no array/Arguments ops.
    let b0_block = graph.blocks.iter().find(|b| b.id == b0).unwrap();
    assert!(b0_block.operations.is_empty(), "B0 chain ops not deleted");
    let bp_block = graph.blocks.iter().find(|b| b.id == bp).unwrap();
    assert!(bp_block.operations.is_empty(), "Bp chain ops not deleted");

    // The rendered value threads straight through: B0→Bp forwards
    // `ctx`, Bp→Bf forwards Bp's inputarg (now bound to `ctx`).
    let b0_exit = &b0_block.exits[0];
    assert_eq!(b0_exit.args[0].as_variable().unwrap().id(), ctx.id());
    let bp_exit = &bp_block.exits[0];
    assert_eq!(bp_exit.args[0].as_variable().unwrap().id(), arg_in.id());
}

#[test]
fn collapse_fmt_chains_expands_lower_hex_to_mod_pct_x() {
    use super::{collapse_fmt_chains, fmt_path_ends_with};
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    // `format!("0x{:x}", n)` — default-spec LowerHex over an Int.
    let mut graph = FunctionGraph::new("fmt_hex_collapse");
    let b0 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();

    let ctx = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(ctx.clone()),
        kind: OpKind::Input {
            name: "n".into(),
            ty: ValueType::Int,
            class_root: None,
        },
    });
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple<i64>"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple<i64>".to_string())),
        },
    });
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple<i64>".to_string())),
            value: LinkArg::Value(ctx.clone()),
            ty: ValueType::Int,
        },
    });
    let arg_ref = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(arg_ref.clone()),
        kind: OpKind::FieldRead {
            base: tuple,
            field: FieldDescriptor::new("__pos_0", Some("Tuple<i64>".to_string())),
            ty: ValueType::Int,
            pure: false,
        },
    });
    let new_lower_hex = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(new_lower_hex.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "rt", "Argument", "new_lower_hex"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![arg_ref]),
            result_ty: ValueType::Ref(None),
        },
    });
    let arg_in = Variable::new();
    graph.block_mut(bp).inputargs = vec![arg_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![new_lower_hex], bp, None).with_prevblock(b0)];

    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_arr.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
            value: LinkArg::Value(arg_in.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, byte) in [2i64, b'0' as i64, b'x' as i64, 0xC0, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "Arguments", "new"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args], bf, None).with_prevblock(bp)];

    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["alloc", "fmt", "format"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];

    collapse_fmt_chains(&mut graph);

    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    let kinds: Vec<&OpKind> = bf_block.operations.iter().map(|op| &op.kind).collect();
    assert!(
        kinds.iter().any(|kind| matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &vec!["__str_const".to_string(), "0x".to_string()]
        )),
        "leading literal 0x: {kinds:?}"
    );
    assert!(
        kinds.iter().any(|kind| matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments == &vec!["__str_const".to_string(), "%x".to_string()]
        )),
        "hex placeholder is __str_const(\"%x\"): {kinds:?}"
    );
    assert!(
        kinds.iter().any(|kind| matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("r_uint")
        )),
        "signed interpolant is r_uint-cast: {kinds:?}"
    );
    let mod_op = kinds
        .iter()
        .find(|kind| matches!(kind, OpKind::BinOp { op, .. } if op == "mod"));
    let Some(OpKind::BinOp { lhs, .. }) = mod_op else {
        panic!("hex placeholder is BinOp mod: {kinds:?}");
    };
    let lhs_op = bf_block
        .operations
        .iter()
        .find(|op| op.result.as_ref().map(|r| r.id()) == Some(lhs.id()));
    assert!(
        matches!(
            lhs_op.map(|op| &op.kind),
            Some(OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            }) if segments == &vec!["__str_const".to_string(), "%x".to_string()]
        ),
        "mod lhs is __str_const(\"%x\"): {lhs_op:?}"
    );
    assert!(
        !bf_block.operations.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if fmt_path_ends_with(segments, &["fmt", "format"])
        )),
        "alloc::fmt::format must be consumed"
    );
    let getslice = bf_block
        .operations
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("__getslice_rangeto")
            )
        })
        .count();
    assert!(
        getslice <= 2,
        "hex collapse must not unroll digits: {getslice} __getslice_rangeto"
    );
}

#[test]
fn collapse_fmt_chains_masks_signed_i8_hex_to_width() {
    use super::collapse_fmt_chains;
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    // `format!("{:x}", n: i8)` — signed interpolant is r_uint then
    // `and 0xff` (`int_to_int_cast` And) before `'%x' %`.
    let mut graph = FunctionGraph::new("fmt_hex_collapse_i8");
    let b0 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();

    let ctx = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(ctx.clone()),
        kind: OpKind::Input {
            name: "n".into(),
            ty: ValueType::Int,
            class_root: None,
        },
    });
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple<i8>"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple<i8>".to_string())),
        },
    });
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple<i8>".to_string())),
            value: LinkArg::Value(ctx.clone()),
            ty: ValueType::Int,
        },
    });
    let arg_ref = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(arg_ref.clone()),
        kind: OpKind::FieldRead {
            base: tuple,
            field: FieldDescriptor::new("__pos_0", Some("Tuple<i8>".to_string())),
            ty: ValueType::Int,
            pure: false,
        },
    });
    let new_lower_hex = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(new_lower_hex.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "rt", "Argument", "new_lower_hex"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![arg_ref]),
            result_ty: ValueType::Ref(None),
        },
    });
    let arg_in = Variable::new();
    graph.block_mut(bp).inputargs = vec![arg_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![new_lower_hex], bp, None).with_prevblock(b0)];

    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_arr.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
            value: LinkArg::Value(arg_in.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, byte) in [2i64, b'0' as i64, b'x' as i64, 0xC0, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "Arguments", "new"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args], bf, None).with_prevblock(bp)];

    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["alloc", "fmt", "format"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];

    collapse_fmt_chains(&mut graph);

    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    let r_uint = bf_block.operations.iter().find(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("r_uint")
        )
    });
    let Some(r_uint_var) = r_uint.and_then(|op| op.result.clone()) else {
        panic!("i8 interpolant is r_uint-cast: {:?}", bf_block.operations);
    };
    let mask = bf_block
        .operations
        .iter()
        .find(|op| matches!(&op.kind, OpKind::ConstUInt(0xff)));
    let Some(mask_var) = mask.and_then(|op| op.result.clone()) else {
        panic!(
            "i8 width mask is ConstUInt(0xff): {:?}",
            bf_block.operations
        );
    };
    let and_op = bf_block.operations.iter().find(|op| {
        matches!(
            &op.kind,
            OpKind::BinOp {
                op,
                result_ty: ValueType::Unsigned,
                ..
            } if op == "and"
        )
    });
    let Some(OpKind::BinOp { lhs, rhs, .. }) = and_op.map(|op| &op.kind) else {
        panic!("i8 interpolant is and-masked: {:?}", bf_block.operations);
    };
    assert_eq!(lhs.id(), r_uint_var.id(), "and lhs is r_uint");
    assert_eq!(rhs.id(), mask_var.id(), "and rhs is 0xff");
    let and_var = and_op
        .and_then(|op| op.result.clone())
        .expect("and has a result");
    let mod_op = bf_block
        .operations
        .iter()
        .find(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mod"));
    let Some(OpKind::BinOp { rhs, .. }) = mod_op.map(|op| &op.kind) else {
        panic!("hex placeholder is BinOp mod: {:?}", bf_block.operations);
    };
    assert_eq!(rhs.id(), and_var.id(), "mod rhs is the width-masked r_uint");
}

#[test]
fn collapse_fmt_chains_takes_tuple_ctor_type_as_integer_proof() {
    use super::{collapse_fmt_chains, fmt_path_ends_with};
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    // `format!("0x{:x}", flags: i64)` — interpolant is a block input
    // with no producer op (loop-carried phi). `Tuple<i64>` ctor is the
    // integer proof `fmt_value_is_integer_hex` cannot supply.
    let mut graph = FunctionGraph::new("fmt_hex_collapse_i64_phi");
    let b0 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();

    let ctx = Variable::new();
    graph.block_mut(b0).inputargs = vec![ctx.clone()];
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple<i64>"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple<i64>".to_string())),
        },
    });
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple<i64>".to_string())),
            value: LinkArg::Value(ctx.clone()),
            ty: ValueType::Int,
        },
    });
    let arg_ref = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(arg_ref.clone()),
        kind: OpKind::FieldRead {
            base: tuple,
            field: FieldDescriptor::new("__pos_0", Some("Tuple<i64>".to_string())),
            ty: ValueType::Int,
            pure: false,
        },
    });
    let new_lower_hex = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(new_lower_hex.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "rt", "Argument", "new_lower_hex"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![arg_ref]),
            result_ty: ValueType::Ref(None),
        },
    });
    let arg_in = Variable::new();
    graph.block_mut(bp).inputargs = vec![arg_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![new_lower_hex], bp, None).with_prevblock(b0)];

    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_arr.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
            value: LinkArg::Value(arg_in.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, byte) in [2i64, b'0' as i64, b'x' as i64, 0xC0, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "Arguments", "new"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args], bf, None).with_prevblock(bp)];

    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["alloc", "fmt", "format"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];

    collapse_fmt_chains(&mut graph);

    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    let kinds: Vec<&OpKind> = bf_block.operations.iter().map(|op| &op.kind).collect();
    assert!(
        kinds.iter().any(|kind| matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("r_uint")
        )),
        "signed interpolant is r_uint-cast: {kinds:?}"
    );
    let mod_op = kinds
        .iter()
        .find(|kind| matches!(kind, OpKind::BinOp { op, .. } if op == "mod"));
    let Some(OpKind::BinOp { lhs, .. }) = mod_op else {
        panic!("hex placeholder is BinOp mod: {kinds:?}");
    };
    let lhs_op = bf_block
        .operations
        .iter()
        .find(|op| op.result.as_ref().map(|r| r.id()) == Some(lhs.id()));
    assert!(
        matches!(
            lhs_op.map(|op| &op.kind),
            Some(OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            }) if segments == &vec!["__str_const".to_string(), "%x".to_string()]
        ),
        "mod lhs is __str_const(\"%x\"): {lhs_op:?}"
    );
    assert!(
        !bf_block.operations.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if fmt_path_ends_with(segments, &["fmt", "format"])
        )),
        "alloc::fmt::format must be consumed"
    );
}

#[test]
fn collapse_fmt_chains_accepts_crate_stripped_string_root() {
    use super::collapse_fmt_chains;
    use crate::model::{CallTarget, OpKind};

    let (mut graph, b0, bp, bf, ctx, arg_in, fmt_args_in, formatted) =
        single_arg_display_fmt_fixture(Some("string::String"));
    collapse_fmt_chains(&mut graph);

    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    assert_eq!(
        bf_block.operations.len(),
        3,
        "crate-stripped string::String interpolant must collapse format!; graph={}",
        graph.dump()
    );
    match &bf_block.operations[0].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => {
            assert_eq!(segments, &vec!["__str_const".to_string(), "hi".to_string()]);
        }
        other => panic!("Bf op[0] not a __str_const: {other:?}"),
    }
    match &bf_block.operations[1].kind {
        OpKind::UnaryOp { op, operand, .. } => {
            assert_eq!(op, "str");
            assert_eq!(operand.id(), fmt_args_in.id());
        }
        other => panic!("Bf op[1] not a str UnaryOp: {other:?}"),
    }
    match &bf_block.operations[2].kind {
        OpKind::BinOp { op, .. } => assert_eq!(op, "add"),
        other => panic!("Bf op[2] not an add BinOp: {other:?}"),
    }
    assert_eq!(
        bf_block.operations[2].result.as_ref().unwrap().id(),
        formatted.id()
    );
    let b0_block = graph.blocks.iter().find(|b| b.id == b0).unwrap();
    assert!(
        b0_block
            .operations
            .iter()
            .all(|op| matches!(&op.kind, OpKind::Input { .. })),
        "B0 chain ops not deleted; ops={:?}",
        b0_block.operations
    );
    let bp_block = graph.blocks.iter().find(|b| b.id == bp).unwrap();
    assert!(bp_block.operations.is_empty(), "Bp chain ops not deleted");
    let b0_exit = &b0_block.exits[0];
    assert_eq!(b0_exit.args[0].as_variable().unwrap().id(), ctx.id());
    let bp_exit = &bp_block.exits[0];
    assert_eq!(bp_exit.args[0].as_variable().unwrap().id(), arg_in.id());
}

#[test]
fn wtf8_from_str_trait_impl_is_string_identity() {
    assert!(super::wtf8_string_identity_segments(&[
        "Wtf8Buf".to_string(),
        "<impl core::convert::From<rustpython_wtf8::Wtf8Buf,&str>>".to_string(),
        "from".to_string(),
    ]));
    assert!(super::wtf8_string_identity_segments(&[
        "rustpython_wtf8".to_string(),
        "Wtf8Buf".to_string(),
        "<impl core::convert::From<rustpython_wtf8::Wtf8Buf,&str>>".to_string(),
        "from".to_string(),
    ]));
    assert!(super::wtf8_string_identity_segments(&[
        "Wtf8Buf".to_string(),
        "from_string".to_string(),
    ]));
    assert!(super::wtf8_string_identity_segments(&[
        "Wtf8".to_string(),
        "new".to_string(),
    ]));
    assert!(!super::wtf8_string_identity_segments(&[
        "Wtf8Buf".to_string(),
        "<impl core::convert::From<rustpython_wtf8::Wtf8Buf,i32>>".to_string(),
        "from".to_string(),
    ]));
    assert!(
        !super::wtf8_string_identity_segments(&[
            "other".to_string(),
            "Wtf8".to_string(),
            "new".to_string(),
        ]),
        "an unrelated Wtf8::new is not the rustpython_wtf8 identity wrap"
    );
    assert!(!super::wtf8_string_identity_segments(&[
        "other".to_string(),
        "Wtf8Buf".to_string(),
        "from_string".to_string(),
    ]));
}

#[test]
fn class_identity_spelling_of_string_family_and_object_roots() {
    let empty = std::collections::HashSet::new();
    assert_eq!(
        super::class_identity_spelling("pyre_object::pyobject::PyObject", &empty),
        super::W_ROOT_CLASS_ROOT
    );
    assert_eq!(
        super::class_identity_spelling("pyre_object::dictmultiobject::ObjectKey", &empty),
        super::OBJECT_KEY_CLASS_ROOT
    );
    assert_eq!(
        super::class_identity_spelling("pyre_object::unicodeobject::Utf8Str", &empty),
        "unicodeobject::Utf8Str"
    );
    assert_eq!(
        super::class_identity_spelling("pyre_object::unicodeobject::UnicodeValueStorage", &empty),
        "unicodeobject::UnicodeValueStorage"
    );
    assert_eq!(
        super::class_identity_spelling("pyre_object::bytesobject::BytesBlock", &empty),
        "bytesobject::BytesBlock"
    );
    assert_eq!(
        super::class_identity_spelling(
            "pyre_object::lowlevel_string::LowlevelStringHeader",
            &empty
        ),
        "lowlevel_string::LowlevelStringHeader"
    );
    assert_eq!(
        super::class_identity_spelling("pyre_object::dictmultiobject::BytesKey", &empty),
        "dictmultiobject::BytesKey"
    );
    assert_eq!(
        super::class_identity_spelling("pyre_object::celldict::StrKey", &empty),
        "celldict::StrKey"
    );
    assert_eq!(
        super::class_identity_spelling("rustpython_wtf8::Wtf8", &empty),
        "Wtf8"
    );
    assert_eq!(
        super::class_identity_spelling("rustpython_wtf8::Wtf8Buf", &empty),
        "Wtf8Buf"
    );
}

#[test]
fn str_family_class_roots_reject_lookalikes() {
    for root in super::STR_FAMILY_CLASS_ROOTS {
        assert!(
            super::is_str_family_class_root(root),
            "{root} is a declared string-family class root"
        );
    }
    assert!(super::is_str_family_class_root("string::String"));
    assert!(super::is_str_family_class_root("str"));
    assert!(!super::is_str_family_class_root("other::Utf8Str"));
    assert!(!super::is_str_family_class_root("other::Wtf8Buf"));
    assert!(!super::is_str_family_class_root("other::BytesBlock"));
    assert!(!super::is_str_family_class_root(
        "other::LowlevelStringHeader"
    ));
    assert!(!super::is_str_family_class_root(
        "lowlevel_string::LowlevelStringHeader"
    ));
    assert!(!super::is_str_length_field_class_root("other::Utf8Str"));
    assert!(super::is_str_length_field_class_root(
        "unicodeobject::Utf8Str"
    ));
    assert!(super::is_str_length_field_class_root(
        "lowlevel_string::LowlevelStringHeader"
    ));
    assert!(!super::is_str_oopspec_class_root("other::Utf8Str"));
    assert!(super::is_str_oopspec_class_root("unicodeobject::Utf8Str"));
    assert!(super::is_str_oopspec_class_root("bytesobject::BytesBlock"));
}

#[test]
fn collapse_fmt_chains_quotes_debug_of_str() {
    use super::collapse_fmt_chains;
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    // `format!("hi{:?}", name)` — Debug of a string-family value is
    // Display plus quotes (`Debug for str`).
    let mut graph = FunctionGraph::new("fmt_collapse_debug_str");
    let b0 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();

    let ctx = Variable::new();
    graph.block_mut(b0).inputargs = vec![ctx.clone()];
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(ctx.clone()),
        kind: OpKind::Input {
            name: "ctx".into(),
            ty: ValueType::Str,
            class_root: None,
        },
    });
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple".to_string())),
        },
    });
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            value: LinkArg::Value(ctx.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let arg_ref = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(arg_ref.clone()),
        kind: OpKind::FieldRead {
            base: tuple,
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let new_debug = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(new_debug.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "rt", "Argument", "new_debug"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![arg_ref]),
            result_ty: ValueType::Ref(None),
        },
    });
    let arg_in = Variable::new();
    graph.block_mut(bp).inputargs = vec![arg_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![new_debug], bp, None).with_prevblock(b0)];

    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: args_arr.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
            value: LinkArg::Value(arg_in.clone()),
            ty: ValueType::Ref(None),
        },
    });
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, byte) in [2i64, 104, 105, 0xC0, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["fmt", "Arguments", "new"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args], bf, None).with_prevblock(bp)];

    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["alloc", "fmt", "format"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];

    collapse_fmt_chains(&mut graph);

    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    assert_eq!(bf_block.operations.len(), 5);
    match &bf_block.operations[0].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => {
            assert_eq!(
                segments,
                &vec!["__str_const".to_string(), "hi\"".to_string()]
            );
        }
        other => panic!("Bf op[0] not a quoted prefix __str_const: {other:?}"),
    }
    match &bf_block.operations[1].kind {
        OpKind::UnaryOp { op, .. } => assert_eq!(op, "str"),
        other => panic!("Bf op[1] not a str UnaryOp: {other:?}"),
    }
    match &bf_block.operations[2].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => {
            assert_eq!(segments, &vec!["__str_const".to_string(), "\"".to_string()]);
        }
        other => panic!("Bf op[2] not a quote __str_const: {other:?}"),
    }
    match &bf_block.operations[3].kind {
        OpKind::BinOp { op, .. } => assert_eq!(op, "add"),
        other => panic!("Bf op[3] not an add BinOp: {other:?}"),
    }
    match &bf_block.operations[4].kind {
        OpKind::BinOp { op, .. } => assert_eq!(op, "add"),
        other => panic!("Bf op[4] not an add BinOp: {other:?}"),
    }
    assert_eq!(
        bf_block.operations[4].result.as_ref().unwrap().id(),
        formatted.id()
    );
}

/// The rewrite is fail-safe: a site whose recorded format-result var has
/// no producer op (a stale / mismatched site) leaves the graph untouched.
#[test]
fn rewrite_debug_enum_fmt_site_declines_when_result_has_no_producer() {
    use super::{DebugEnumFmtCollapse, rewrite_debug_enum_fmt_site};
    use crate::flowspace::model::Variable;
    use crate::model::FunctionGraph;

    let mut graph = FunctionGraph::new("debug_enum_fmt_decline");
    let bf = graph.startblock;
    let func = graph
        .push_op_var(bf, crate::model::OpKind::ConstInt(0), true)
        .unwrap();
    graph.set_return(bf, None);
    let ops_before = graph.blocks[bf.0].operations.len();

    let site = DebugEnumFmtCollapse {
        // A var never produced by any op — no format block to split.
        format_result: Variable::new(),
        enum_value: func,
        pieces: vec!["a".to_string(), "b".to_string()],
        by_discr: vec![(0, "Invalid".to_string())],
        link_rewrites: Vec::new(),
        dead_results: Vec::new(),
        dead_bases: Vec::new(),
    };
    assert!(!rewrite_debug_enum_fmt_site(&mut graph, &site));
    assert_eq!(
        graph.blocks[bf.0].operations.len(),
        ops_before,
        "graph left untouched on decline"
    );
}

#[test]
fn collapse_fmt_chains_multi_expands_two_arg_chain() {
    use super::{collapse_fmt_chains_multi, fmt_path_ends_with, is_arguments_new_path};
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    // Reconstruct the `f(format!("a{}b{}c", x, y))` shape charon lowers:
    // B0 builds the 2-field arg Tuple + `new_display(&x)`, B1 reads the
    // 2nd field off a *threaded copy* of the tuple ref + `new_display(&y)`
    // (exercising the cross-block `unwrap_fmt_arg_tuple_ref` match), Bp
    // builds the args/pieces arrays + `Arguments::new`, Bf calls
    // `alloc::fmt::format`.
    let mut graph = FunctionGraph::new("fmt_collapse_multi");
    let b0 = graph.create_block();
    let b1 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();

    let fpath = |segs: &[&str]| CallTarget::FunctionPath {
        segments: segs.iter().map(|s| s.to_string()).collect(),
        fun_decl_id: None,
        specialization: crate::parse::GraphSpecialization::Regular,
    };

    // ── B0: Tuple{x, y} + FieldRead __pos_0 + new_display(&x) ──
    let x = Variable::new();
    let y = Variable::new();
    graph.block_mut(b0).inputargs = vec![x.clone(), y.clone()];
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple".to_string())),
        },
    });
    for (i, v) in [&x, &y].iter().enumerate() {
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: tuple.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Tuple".to_string())),
                value: LinkArg::Value((*v).clone()),
                ty: ValueType::Ref(None),
            },
        });
    }
    let ar0 = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(ar0.clone()),
        kind: OpKind::FieldRead {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let nd0 = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(nd0.clone()),
        kind: OpKind::Call {
            target: fpath(&["fmt", "rt", "Argument", "new_display"]),
            args: crate::model::call_args(vec![ar0.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    // B0 → B1 forwards the tuple ref and the first new_display.
    let tuple_in = Variable::new();
    let nd0_in = Variable::new();
    graph.block_mut(b1).inputargs = vec![tuple_in.clone(), nd0_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![tuple.clone(), nd0], b1, None).with_prevblock(b0)];

    // ── B1: FieldRead __pos_1 (off the threaded tuple copy) + new_display(&y) ──
    let ar1 = Variable::new();
    graph.block_mut(b1).operations.push(SpaceOperation {
        result: Some(ar1.clone()),
        kind: OpKind::FieldRead {
            base: tuple_in.clone(),
            field: FieldDescriptor::new("__pos_1", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let nd1 = Variable::new();
    graph.block_mut(b1).operations.push(SpaceOperation {
        result: Some(nd1.clone()),
        kind: OpKind::Call {
            target: fpath(&["fmt", "rt", "Argument", "new_display"]),
            args: crate::model::call_args(vec![ar1.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let nd0_p = Variable::new();
    let nd1_p = Variable::new();
    graph.block_mut(bp).inputargs = vec![nd0_p.clone(), nd1_p.clone()];
    graph.block_mut(b1).exits =
        vec![Link::from_variables(&graph, vec![nd0_in, nd1], bp, None).with_prevblock(b1)];

    // ── Bp: args array, pieces array, Arguments::new ──
    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, v) in [&nd0_p, &nd1_p].iter().enumerate() {
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value((*v).clone()),
                ty: ValueType::Ref(None),
            },
        });
    }
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    // Template for "a{}b{}c": 'a', ph, 'b', ph, 'c', term.
    for (i, byte) in [1i64, 97, 0xC0, 1, 98, 0xC0, 1, 99, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: fpath(&["fmt", "Arguments", "new"]),
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args.clone()], bf, None).with_prevblock(bp)];

    // ── Bf: alloc::fmt::format(args) → String ──
    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: fpath(&["alloc", "fmt", "format"]),
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];

    collapse_fmt_chains_multi(&mut graph);

    // Both `new_display` ctors are rendered in place to `str`.
    let find_str = |bid| {
        graph
            .blocks
            .iter()
            .find(|b| b.id == bid)
            .unwrap()
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::UnaryOp { op: o, operand, .. } if o == "str" => Some(operand.id()),
                _ => None,
            })
    };
    // The argument-Tuple round-trip is eliminated: the local placeholder
    // reads its value `x` directly; the threaded placeholder reads the
    // rebound carrier (`tuple_in`, now forwarding `y`), not the Tuple
    // field.
    assert_eq!(find_str(b0), Some(x.id()), "B0 new_display→str(x)");
    assert_eq!(
        find_str(b1),
        Some(tuple_in.id()),
        "B1 new_display→str(rebound carrier)"
    );

    // No `fmt` externs survive anywhere in the graph.
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
            {
                let bad = fmt_path_ends_with(segments, &["Argument", "new_display"])
                    || is_arguments_new_path(segments)
                    || fmt_path_ends_with(segments, &["fmt", "format"]);
                assert!(!bad, "residual fmt extern survived: {segments:?}");
            }
        }
    }

    // The Tuple ctor, its `__pos_N` writes, and the field read-backs are
    // all deleted — no StringRepr→PyObject-erased-field setattr survives.
    for b in &graph.blocks {
        for op in &b.operations {
            if let Some(r) = &op.result {
                assert_ne!(r.id(), tuple.id(), "Tuple ctor survived");
                assert_ne!(r.id(), ar0.id(), "__pos_0 FieldRead survived");
                assert_ne!(r.id(), ar1.id(), "__pos_1 FieldRead survived");
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                assert_ne!(base.id(), tuple.id(), "Tuple __pos_N FieldWrite survived");
            }
        }
    }
    // The B0→B1 link slot that forwarded the Tuple ref now forwards `y`.
    let b0_block = graph.blocks.iter().find(|b| b.id == b0).unwrap();
    assert_eq!(
        b0_block.exits[0].args[0].as_variable().map(|v| v.id()),
        Some(y.id()),
        "B0→B1 Tuple slot rebound to y"
    );

    // Bf's format result is `fmt_args_in`. The copy is not a new variable.
    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    assert!(
        bf_block
            .operations
            .iter()
            .all(|op| { !matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as") })
    );
    assert_eq!(
        bf_block.exits[0].args[0].as_variable().map(|v| v.id()),
        Some(fmt_args_in.id())
    );

    // Bp now folds the rendered values with the pieces (str_const + add)
    // and forwards the folded String where it forwarded `Arguments`.
    let bp_block = graph.blocks.iter().find(|b| b.id == bp).unwrap();
    let add_count = bp_block
        .operations
        .iter()
        .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
        .count();
    assert!(
        add_count >= 2,
        "Bp must fold both rendered args, got {add_count} adds"
    );
    let bp_exit_val = bp_block.exits[0].args[0].as_variable().unwrap().id();
    assert_ne!(
        bp_exit_val,
        fmt_args.id(),
        "Bp must forward the folded String, not Arguments"
    );
}

#[test]
fn collapse_fmt_chains_multi_eliminates_cross_block_local_tuple() {
    use super::{collapse_fmt_chains_multi, fmt_path_ends_with, is_arguments_new_path};
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
        ValueType,
    };

    // The `index_type_error` shape: `format!` splits the two placeholder
    // field reads across a straight-line block boundary, but B1 reads the
    // 2nd field off the *same* Tuple ref id (referenced cross-block by
    // dominance), not a threaded inputarg copy.  Both placeholders are
    // Local; the cross-block reader is admitted because B0 dominates B1
    // and every Tuple write lives in B0.
    let mut graph = FunctionGraph::new("fmt_collapse_cross_block_local");
    let b0 = graph.create_block();
    let b1 = graph.create_block();
    let bp = graph.create_block();
    let bf = graph.create_block();
    let bret = graph.create_block();
    // B0 is the entry so `block_dominates(b0, b1)` holds (b1 reachable
    // only through b0), mirroring the real `index_type_error` CFG.
    graph.startblock = b0;

    let fpath = |segs: &[&str]| CallTarget::FunctionPath {
        segments: segs.iter().map(|s| s.to_string()).collect(),
        fun_decl_id: None,
        specialization: crate::parse::GraphSpecialization::Regular,
    };

    // ── B0: Tuple{x, y} + both FieldWrites + FieldRead __pos_0 + new_display ──
    let x = Variable::new();
    let y = Variable::new();
    graph.block_mut(b0).inputargs = vec![x.clone(), y.clone()];
    let tuple = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Tuple"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Tuple".to_string())),
        },
    });
    for (i, v) in [&x, &y].iter().enumerate() {
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: tuple.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Tuple".to_string())),
                value: LinkArg::Value((*v).clone()),
                ty: ValueType::Ref(None),
            },
        });
    }
    let ar0 = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(ar0.clone()),
        kind: OpKind::FieldRead {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let nd0 = Variable::new();
    graph.block_mut(b0).operations.push(SpaceOperation {
        result: Some(nd0.clone()),
        kind: OpKind::Call {
            target: fpath(&["fmt", "rt", "Argument", "new_display"]),
            args: crate::model::call_args(vec![ar0.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    // B0 → B1 forwards only the first new_display; the Tuple ref is NOT a
    // link arg — B1 references it directly by id (dominance carry).
    let nd0_in = Variable::new();
    graph.block_mut(b1).inputargs = vec![nd0_in.clone()];
    graph.block_mut(b0).exits =
        vec![Link::from_variables(&graph, vec![nd0], b1, None).with_prevblock(b0)];

    // ── B1: FieldRead __pos_1 off the *same* tuple id + new_display(&y) ──
    let ar1 = Variable::new();
    graph.block_mut(b1).operations.push(SpaceOperation {
        result: Some(ar1.clone()),
        kind: OpKind::FieldRead {
            base: tuple.clone(),
            field: FieldDescriptor::new("__pos_1", Some("Tuple".to_string())),
            ty: ValueType::Ref(None),
            pure: false,
        },
    });
    let nd1 = Variable::new();
    graph.block_mut(b1).operations.push(SpaceOperation {
        result: Some(nd1.clone()),
        kind: OpKind::Call {
            target: fpath(&["fmt", "rt", "Argument", "new_display"]),
            args: crate::model::call_args(vec![ar1.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let nd0_p = Variable::new();
    let nd1_p = Variable::new();
    graph.block_mut(bp).inputargs = vec![nd0_p.clone(), nd1_p.clone()];
    graph.block_mut(b1).exits =
        vec![Link::from_variables(&graph, vec![nd0_in, nd1], bp, None).with_prevblock(b1)];

    // ── Bp: args array, pieces array, Arguments::new ──
    let args_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(args_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, v) in [&nd0_p, &nd1_p].iter().enumerate() {
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value((*v).clone()),
                ty: ValueType::Ref(None),
            },
        });
    }
    let pieces_arr = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(pieces_arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    for (i, byte) in [1i64, 97, 0xC0, 1, 98, 0xC0, 1, 99, 0].iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstInt(*byte),
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: pieces_arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: LinkArg::Value(v),
                ty: ValueType::Int,
            },
        });
    }
    let fmt_args = Variable::new();
    graph.block_mut(bp).operations.push(SpaceOperation {
        result: Some(fmt_args.clone()),
        kind: OpKind::Call {
            target: fpath(&["fmt", "Arguments", "new"]),
            args: crate::model::call_args(vec![pieces_arr, args_arr]),
            result_ty: ValueType::Ref(None),
        },
    });
    let fmt_args_in = Variable::new();
    graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
    graph.block_mut(bp).exits =
        vec![Link::from_variables(&graph, vec![fmt_args.clone()], bf, None).with_prevblock(bp)];

    // ── Bf: alloc::fmt::format(args) → String ──
    let formatted = Variable::new();
    graph.block_mut(bf).operations.push(SpaceOperation {
        result: Some(formatted.clone()),
        kind: OpKind::Call {
            target: fpath(&["alloc", "fmt", "format"]),
            args: crate::model::call_args(vec![fmt_args_in.clone()]),
            result_ty: ValueType::Ref(None),
        },
    });
    let ret = Variable::new();
    graph.block_mut(bret).inputargs = vec![ret];
    graph.block_mut(bf).exits =
        vec![Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf)];

    collapse_fmt_chains_multi(&mut graph);

    let find_str = |bid| {
        graph
            .blocks
            .iter()
            .find(|b| b.id == bid)
            .unwrap()
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::UnaryOp { op: o, operand, .. } if o == "str" => Some(operand.id()),
                _ => None,
            })
    };
    // Both placeholders are Local and read their values directly — the
    // cross-block reader in B1 reads `y`, not a Tuple field.
    assert_eq!(find_str(b0), Some(x.id()), "B0 new_display→str(x)");
    assert_eq!(
        find_str(b1),
        Some(y.id()),
        "B1 cross-block new_display→str(y)"
    );

    // No `fmt` externs survive.
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
            {
                let bad = fmt_path_ends_with(segments, &["Argument", "new_display"])
                    || is_arguments_new_path(segments)
                    || fmt_path_ends_with(segments, &["fmt", "format"]);
                assert!(!bad, "residual fmt extern survived: {segments:?}");
            }
        }
    }

    // Tuple ctor, both writes, and both field read-backs are gone.
    for b in &graph.blocks {
        for op in &b.operations {
            if let Some(r) = &op.result {
                assert_ne!(r.id(), tuple.id(), "Tuple ctor survived");
                assert_ne!(r.id(), ar0.id(), "__pos_0 FieldRead survived");
                assert_ne!(r.id(), ar1.id(), "__pos_1 FieldRead survived");
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                assert_ne!(base.id(), tuple.id(), "Tuple __pos_N FieldWrite survived");
            }
        }
    }
}

#[test]
fn dict_strategy_ref_imp_layout_is_dyn_fat() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    if !std::path::Path::new(path).is_file() {
        return;
    }
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let (_, _, fields, _, _, attrs, _, _) = super::derive_program_metadata(
        &llbc,
        &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc)),
    );
    let spelling = ["dictmultiobject::DictStrategyRef", "DictStrategyRef"]
        .iter()
        .find_map(|owner| fields.field_type(owner, "imp"))
        .expect("DictStrategyRef.imp layout row");
    assert!(
        crate::fat_ptr_layout::spelling_is_dyn_fat_ptr(spelling),
        "DictStrategyRef.imp must be two_word_fat_struct, got {spelling:?}"
    );
    let attr_rows = attrs
        .get("dictmultiobject::DictStrategyRef")
        .expect("DictStrategyRef FORCE attrs");
    assert!(
        attr_rows.iter().any(|(name, _)| name == "imp"),
        "FORCE attrs include imp: {attr_rows:?}"
    );
}

#[test]
fn cast_pointer_marker_carries_root_in_path_and_result_type() {
    use crate::model::{CallTarget, OpKind, ValueType};
    let arg = crate::flowspace::model::Variable::new();
    let op = cast_pointer_marker_op("W_CastTarget".to_string(), arg.clone());
    let OpKind::Call {
        target, result_ty, ..
    } = &op
    else {
        panic!("marker must be an OpKind::Call");
    };
    assert_eq!(target, &CallTarget::function_path(["__cast_pointer"]),);
    assert_eq!(crate::model::cast_pointer_root(&op), Some("W_CastTarget"));
    assert_eq!(result_ty, &ValueType::Ref(Some("W_CastTarget".to_string())));
}

/// The real `RangeBounds::contains` body is the caller of the two
/// comparisons (`start <= x && x < end` via `le`/`lt`). `int_between` is
/// not emitted: RPython only has that llop because source calls
/// `rarithmetic.int_between` (`rint.py`). Ignored by default (loads
/// the real LLBC); run with `cargo test -p majit-translate --lib
/// range_contains_real_body -- --ignored --nocapture`.
#[test]
#[ignore]
fn range_contains_real_body_carries_le_lt() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let ctx = super::LowerContext::new(&llbc);

    let mut bounds_contains = None;
    let mut range_contains = None;
    for fd in llbc.iter_fun_decls() {
        if !fd.has_unstructured_body() {
            continue;
        }
        let p = fd.item_meta.name_path();
        if !p.ends_with("::contains") || !p.contains("ops::range") {
            continue;
        }
        if p.contains("RangeBounds") {
            bounds_contains = Some(fd);
        } else if p.contains("RangeInclusive") {
            continue;
        } else if p.contains("Range") {
            range_contains = Some(fd);
        }
    }
    let bounds_fd = bounds_contains.expect("RangeBounds::contains FunDecl with a body");
    let bounds_graph = super::lower_fun_decl(&ctx, bounds_fd)
        .unwrap_or_else(|e| panic!("lower RangeBounds::contains: {e:?}"));

    let mut leaves: Vec<String> = Vec::new();
    let mut binops: Vec<String> = Vec::new();
    let mut int_between = 0usize;
    for op in bounds_graph.blocks.iter().flat_map(|b| b.operations.iter()) {
        match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                if let Some(leaf) = segments.last() {
                    leaves.push(leaf.clone());
                }
            }
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => leaves.push(name.clone()),
            OpKind::BinOp { op, .. } => binops.push(op.clone()),
            OpKind::LoweredBlackholeOp { opname, .. } if opname == "int_between" => {
                int_between += 1;
            }
            _ => {}
        }
    }
    eprintln!(
        "RangeBounds::contains path={} leaves={leaves:?} binops={binops:?} int_between={int_between}",
        bounds_fd.item_meta.name_path()
    );
    assert_eq!(int_between, 0, "body must not emit int_between");
    let has = |name: &str| leaves.iter().any(|l| l == name) || binops.iter().any(|b| b == name);
    assert!(has("start_bound"), "missing start_bound: {leaves:?}");
    assert!(has("end_bound"), "missing end_bound: {leaves:?}");
    assert!(has("le"), "missing le: leaves={leaves:?} binops={binops:?}");
    assert!(has("lt"), "missing lt: leaves={leaves:?} binops={binops:?}");

    if let Some(range_fd) = range_contains {
        let range_graph = super::lower_fun_decl(&ctx, range_fd)
            .unwrap_or_else(|e| panic!("lower Range::contains: {e:?}"));
        let mut range_leaves: Vec<String> = Vec::new();
        let mut range_between = 0usize;
        for op in range_graph.blocks.iter().flat_map(|b| b.operations.iter()) {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => {
                    if let Some(leaf) = segments.last() {
                        range_leaves.push(leaf.clone());
                    }
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } => range_leaves.push(name.clone()),
                OpKind::LoweredBlackholeOp { opname, .. } if opname == "int_between" => {
                    range_between += 1;
                }
                _ => {}
            }
        }
        eprintln!(
            "Range::contains path={} leaves={range_leaves:?} int_between={range_between}",
            range_fd.item_meta.name_path()
        );
        assert_eq!(range_between, 0);
    }
}

#[test]
fn map_err_drop_detection_requires_resolved_trait_metadata() {
    use super::{Lowering, Unstructured};
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(), "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    for (trait_name, dropless) in [(Some("Drop"), false), (None, false), (Some("Copy"), true)] {
        let traits = trait_name
            .map(|name| {
                vec![serde_json::json!({
                    "def_id": 0, "item_meta": meta(&["core", "ops", "drop", name])
                })]
            })
            .unwrap_or_default();
        let file = serde_json::json!({
            "charon_version": "0.1.201", "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [{"def_id": 0, "item_meta": meta(&["fixture", "Capture"]), "kind": {"Struct": []}}],
                "fun_decls": [], "global_decls": [], "trait_decls": traits,
                "trait_impls": [{"impl_trait": {"id": 0, "generics": {"types": [
                    {"Value": [1, {"Adt": {"id": 0, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}}]}
                ]}}}]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).unwrap();
        let body: Unstructured = serde_json::from_value(serde_json::json!({
            "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span.clone()
        }))
        .unwrap();
        let dont_look_inside = std::collections::HashSet::new();
        let accum = super::AccumulatorFacts::build(&llbc, &body);
        let tombstoned_leaves = std::collections::HashSet::new();
        let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
        let lowering = Lowering::new(
            &llbc,
            "fixture".into(),
            &body,
            crate::HostStaticAddrs::default(),
            &[],
            None,
            &dont_look_inside,
            &tombstoned_leaves,
            &gc_struct_ids_llbc,
            super::no_immutable_fields(),
            &std::collections::HashSet::<String>::new(),
            &accum,
            &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            lowering.type_decl_is_trivially_dropless(llbc.type_by_id(0).unwrap(), 0),
            dropless,
            "{trait_name:?}"
        );
        assert!(
            !lowering.type_decl_has_explicit_drop(99),
            "an unrelated owner must not be rejected"
        );
    }
}

/// Empty-array Unsize and a STR-view `as_bytes` join a window local:
/// both data words are `CCHARP` (`cast_int_to_ptr` / GET_RAW_ADDRESS).
#[test]
fn window_join_empty_array_unsize_and_str_view_is_ccharp() {
    let shared = window_shared_slice();
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let usize_zero = serde_json::json!([
        {"Integer": {"Unsigned": ["Usize", "0"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let array_ty = serde_json::json!({"Array": [window_u8(), usize_zero, null]});
    let array_ref = serde_json::json!({"Ref": ["Erased", array_ty, "Shared"]});
    let usize_lit = serde_json::json!({
        "Const": [
            {"Integer": {"Unsigned": ["Usize", "0"]}},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
        ]
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![str_ty.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "bytesobject", "w_bytes_empty"], true),
        "signature": {"is_unsafe": true, "inputs": [bool_ty, array_ref, str_ty], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 3, "locals": [
                window_local(0, &shared),
                window_local(1, &bool_ty),
                window_local(2, &array_ref),
                window_local(3, &str_ty)
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
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &shared),
                            {"UnaryOp": [
                                {"Cast": {"Unsize": [
                                    array_ref.clone(),
                                    shared.clone(),
                                    {"Length": usize_lit}
                                ]}},
                                {"Move": window_place(serde_json::json!({"Local": 2}), &array_ref)}
                            ]}
                        ]}
                    }],
                    "terminator": {"span": window_span(), "kind": {"Goto": {"target": 4}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 3}), &str_ty)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 4, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(Vec::new(), vec![caller, as_bytes]);
    let graph = super::lower_function(&llbc, "w_bytes_empty")
        .expect("lower empty-array unsize join STR view");
    let leafs = window_call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "cast_int_to_ptr"),
        "empty-array unsize data word is cast_int_to_ptr(CCHARP); leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        leafs
            .iter()
            .any(|leaf| { leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING }),
        "STR-view arm data word is GET_RAW_ADDRESS_OF_STRING; leafs={leafs:?} graph={}",
        graph.dump()
    );
}

/// A shared `&[u8]` reborrow of `&Box<[u8]>` stays the list: `len` /
/// index are `__len` / `ArrayRead`, never the resizable-list raw ptr.
#[test]
fn list_view_box_u8_slice_len_index_stays_the_list() {
    use crate::model::{OpKind, VecFieldPart};

    let u8_ty = window_u8();
    let n = window_usize();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let shared = window_shared_slice();
    let box_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [window_slice()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let box_ref = serde_json::json!({"Ref": ["Erased", box_ty, "Shared"]});
    let (unwind, ret) = window_unwind_and_return_blocks();
    let deref = window_opaque_fun(
        1,
        &["core", "ops", "deref", "<Impl>", "deref"],
        vec![box_ref.clone()],
        shared.clone(),
    );
    let len = window_opaque_fun(
        2,
        &["core", "slice", "<Impl>", "len"],
        vec![shared.clone()],
        n.clone(),
    );
    let generic_slice = serde_json::json!({
        "Ref": ["Erased", {"Slice": [{"TypeVar": {"Bound": [0, 0]}}, null]}, "Shared"]
    });
    let index = window_opaque_fun(
        3,
        &["core", "slice", "index", "<Impl>", "index"],
        vec![generic_slice, n.clone()],
        u8_ty.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "list_view_box_u8"], true),
        "signature": {"is_unsafe": false, "inputs": [box_ref, n], "output": bool_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &bool_ty),
                window_local(1, &box_ref),
                window_local(2, &n),
                window_local(3, &shared),
                window_local(4, &n),
                window_local(5, &u8_ty)
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
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 1}), &box_ref)}],
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
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 3}), &shared)}],
                        "dest": window_place(serde_json::json!({"Local": 4}), &n)
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
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &shared)},
                            {"Copy": window_place(serde_json::json!({"Local": 2}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 5}), &u8_ty)
                    }, "target": 4, "on_unwind": 1}}}
                },
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &bool_ty),
                            {"BinaryOp": [
                                "Lt",
                                {"Copy": window_place(serde_json::json!({"Local": 5}), &u8_ty)},
                                {"Copy": window_place(serde_json::json!({"Local": 4}), &n)}
                            ]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(Vec::new(), vec![caller, deref, len, index]);
    let graph = super::lower_function(&llbc, "list_view_box_u8")
        .expect("lower &Box<[u8]> list-view len/index");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| {
            leaf == crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST
                || leaf == "cast_int_to_ptr"
        }),
        "Box<[u8]> list view must not take a raw ptr; leafs={leafs:?} graph={}",
        graph.dump()
    );
    assert!(
        leafs.iter().any(|leaf| leaf == "__len"),
        "Box<[u8]> list view len is __len; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let getitem = window_graph_ops(&graph).iter().any(|op| {
        matches!(&op.kind, OpKind::ArrayRead { .. })
            || matches!(
                &op.kind,
                OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|leaf| {
                    leaf == "ll_getitem_nonneg" || leaf == "__getitem__"
                })
            )
    });
    assert!(
        getitem,
        "Box<[u8]> list view index is list getitem; leafs={leafs:?} graph={}",
        graph.dump()
    );

    // `fn f(s: &S) -> usize { s.kinds.len() }` with `S { kinds: Box<[u8]> }`.
    // The Box is a struct field read in `fat_box_vars`, so `<[T]>::len` is
    // FatLen of `kinds`, not a residual `__len`.
    let holder_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let holder_ref = serde_json::json!({"Ref": ["Erased", holder_adt, "Shared"]});
    let field_box_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [window_slice()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let usize_ty = window_usize();
    let holder = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "S"], true),
        "kind": {"Struct": [{
            "name": "kinds",
            "ty": field_box_ty,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        }]}
    });
    let kinds_place = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Projection": [
                window_place(serde_json::json!({"Local": 1}), &holder_ref),
                "Deref"
            ]}), &holder_adt),
            {"Field": [null, 0]}
        ]}),
        &field_box_ty,
    );
    let field_len = window_opaque_fun(
        1,
        &["core", "slice", "<Impl>", "len"],
        vec![field_box_ty.clone()],
        usize_ty.clone(),
    );
    let (field_unwind, field_ret) = window_unwind_and_return_blocks();
    let field_caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "kinds_len"], true),
        "signature": {"is_unsafe": false, "inputs": [holder_ref], "output": usize_ty},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &usize_ty),
                window_local(1, &holder_ref),
                window_local(2, &field_box_ty)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &field_box_ty),
                            {"Use": [{"Copy": kinds_place}, "No"]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Call": {"call": {
                        "func": {"Regular": {
                            "kind": {"Fun": 1},
                            "generics": window_empty_generics()
                        }},
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &field_box_ty)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &usize_ty)
                    }, "target": 2, "on_unwind": 1}}}
                },
                field_unwind,
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                },
                field_ret
            ]
        }}
    });
    let field_llbc = window_llbc(vec![holder], vec![field_caller, field_len]);
    let field_graph = super::lower_function(&field_llbc, "kinds_len").expect("lower S.kinds.len()");
    let field_ops = window_graph_ops(&field_graph);
    let field_leafs = window_call_leafs(&field_graph);
    let kinds_fat_len = field_ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { field, .. }
                if field.name == "kinds" && field.vec_part == Some(VecFieldPart::FatLen)
        )
    });
    assert!(
        kinds_fat_len,
        "Box<[u8]> field len is FatLen of kinds; leafs={field_leafs:?} graph={}",
        field_graph.dump()
    );
    assert!(
        !field_leafs.iter().any(|leaf| leaf == "__len"),
        "Box<[u8]> field len must not residualize __len; leafs={field_leafs:?} graph={}",
        field_graph.dump()
    );
}

/// A window local joined from a STR view and `from_raw_parts` of an
/// integer address has a CCHARP data word on both links.
#[test]
fn window_join_str_view_and_from_raw_parts_is_ccharp() {
    use crate::model::LinkArg;

    let shared = window_shared_slice();
    let str_ty = serde_json::json!({"Ref": ["Erased", {"Adt": {"builtin": "Str"}}, "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let n = window_usize();
    let (unwind, ret) = window_unwind_and_return_blocks();
    let as_bytes = window_opaque_fun(
        1,
        &["core", "str", "<Impl>", "as_bytes"],
        vec![str_ty.clone()],
        shared.clone(),
    );
    let from_raw = window_opaque_fun(
        2,
        &["core", "slice", "raw", "from_raw_parts"],
        vec![n.clone(), n.clone()],
        shared.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "window_join_from_raw_parts"], true),
        "signature": {"is_unsafe": true, "inputs": [bool_ty, n, n, str_ty], "output": shared},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 4, "locals": [
                window_local(0, &shared),
                window_local(1, &bool_ty),
                window_local(2, &n),
                window_local(3, &n),
                window_local(4, &str_ty)
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
                            "kind": {"Fun": 2},
                            "generics": window_empty_generics()
                        }},
                        "args": [
                            {"Copy": window_place(serde_json::json!({"Local": 2}), &n)},
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &n)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
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
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 4}), &str_ty)}],
                        "dest": window_place(serde_json::json!({"Local": 0}), &shared)
                    }, "target": 4, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(Vec::new(), vec![caller, as_bytes, from_raw]);
    let graph = super::lower_function(&llbc, "window_join_from_raw_parts")
        .expect("lower STR-view join from_raw_parts");
    let mut by_target: Vec<(crate::model::BlockId, Vec<&crate::model::Link>)> = Vec::new();
    for block in &graph.blocks {
        for link in &block.exits {
            if link.target == graph.returnblock || link.target == graph.exceptblock {
                continue;
            }
            if let Some((_, links)) = by_target.iter_mut().find(|(id, _)| *id == link.target) {
                links.push(link);
            } else {
                by_target.push((link.target, vec![link]));
            }
        }
    }
    let join_links = by_target
        .into_iter()
        .find(|(_, links)| links.len() == 2)
        .map(|(_, links)| links)
        .expect("window local has a two-predecessor join");
    for link in &join_links {
        let has_ccharp = link.args.iter().any(|arg| {
            LinkArg::as_variable(arg).is_some_and(|var| {
                var.concretetype()
                    .as_ref()
                    .is_some_and(super::lltype_is_ccharp)
            })
        });
        assert!(
            has_ccharp,
            "join link data word is CCHARP; args={:?} graph={}",
            link.args,
            graph.dump()
        );
    }
}

#[test]
fn unify_receiver_ty_bound_var_equals_actual_without_recursing() {
    // A bound TypeVar is compared with `actual` after
    // `strip_ty_indirections`. Recursing through subst on a
    // self-binding (`subst[0] = TypeVar(0)`) would loop.
    let llbc = empty_llbc();
    let tv0 = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let f64_ty = serde_json::json!({"Scalar": {"Float": "F64"}});
    let mut subst = vec![Some(tv0.clone())];
    let mut const_subst = Vec::new();
    assert!(!super::unify_receiver_ty(
        &tv0,
        &i64_ty,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
    let mut subst = vec![Some(i64_ty.clone())];
    let mut const_subst = Vec::new();
    assert!(super::unify_receiver_ty(
        &tv0,
        &i64_ty,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
    let mut subst = vec![Some(i64_ty.clone())];
    let mut const_subst = Vec::new();
    assert!(!super::unify_receiver_ty(
        &tv0,
        &f64_ty,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
}

#[test]
fn unify_receiver_ty_distinguishes_adt_const_generics() {
    // `Buffer<4>` must not match `Buffer<8>`. Declaration parameters
    // bind by position (`FunctionDesc.cachedgraph`).
    let llbc = empty_llbc();
    let cg = |n: &str| {
        serde_json::json!([
            {"Integer": {"Unsigned": ["Usize", n]}},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
        ])
    };
    let buf = |n: &str| {
        serde_json::json!({
            "Adt": {
                "id": 0,
                "generics": {"types": [], "const_generics": [cg(n)]}
            }
        })
    };
    let buf_var = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {"types": [], "const_generics": [{"Var": 0}]}
        }
    });
    let buf4 = buf("4");
    let buf8 = buf("8");
    let mut subst = Vec::new();
    let mut const_subst = Vec::new();
    assert!(super::unify_receiver_ty(
        &buf4,
        &buf4,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
    let mut subst = Vec::new();
    let mut const_subst = Vec::new();
    assert!(!super::unify_receiver_ty(
        &buf4,
        &buf8,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
    let mut subst = Vec::new();
    let mut const_subst = Vec::new();
    assert!(super::unify_receiver_ty(
        &buf_var,
        &buf4,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
    let row = serde_json::json!({
        "generics": {
            "const_generics": [{"Var": 0}],
            "trait_refs": []
        }
    });
    let got = super::reconstructed_impl_generics(&subst, &const_subst, &row)
        .expect("const N bound from Buffer<4>");
    assert_eq!(got["const_generics"], serde_json::json!([cg("4")]));
}

#[test]
fn reconstructed_impl_generics_binds_swapped_const_params() {
    // `impl<const N, const M> Trait for Pair<M, N>` with `Pair<2, 3>`
    // binds N=3, M=2 (`FunctionDesc.cachedgraph`).
    let llbc = empty_llbc();
    let cg = |n: &str| {
        serde_json::json!([
            {"Integer": {"Unsigned": ["Usize", n]}},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
        ])
    };
    let pair_pattern = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "types": [],
                "const_generics": [{"Var": 1}, {"Var": 0}]
            }
        }
    });
    let pair_actual = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "types": [],
                "const_generics": [cg("2"), cg("3")]
            }
        }
    });
    let mut subst = Vec::new();
    let mut const_subst = Vec::new();
    assert!(super::unify_receiver_ty(
        &pair_pattern,
        &pair_actual,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
    let row = serde_json::json!({
        "generics": {
            "const_generics": [{"Var": 0}, {"Var": 1}],
            "trait_refs": []
        }
    });
    let got =
        super::reconstructed_impl_generics(&subst, &const_subst, &row).expect("N and M bound");
    assert_eq!(got["const_generics"], serde_json::json!([cg("3"), cg("2")]));
    let mut subst = Vec::new();
    let mut const_subst = Vec::new();
    let pair_repeat = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "types": [],
                "const_generics": [{"Var": 0}, {"Var": 0}]
            }
        }
    });
    assert!(!super::unify_receiver_ty(
        &pair_repeat,
        &pair_actual,
        &mut subst,
        &mut const_subst,
        &llbc
    ));
}

/// Production `w_module_dict_setitem_str_internal` names the inherent
/// `ModuleDictStrategy::setitem_str` FunDecl, not the trait trampoline.
#[test]
fn w_module_dict_setitem_str_internal_calls_inherent_setitem_str() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipping: {path} is missing; run `python3 scripts/extract-llbc.py`");
        return;
    }
    let llbc = Llbc::load(path).expect("load pyre-object LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path()
                .ends_with("::w_module_dict_setitem_str_internal")
        })
        .expect("w_module_dict_setitem_str_internal in pyre-object LLBC");
    let context = super::LowerContext::new(&llbc);
    let graph =
        super::lower_fun_decl(&context, fd).expect("lower w_module_dict_setitem_str_internal");
    let mut bindings = Vec::new();
    for op in graph_ops(&graph) {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        match target {
            CallTarget::Method {
                name,
                resolved_path,
                fun_decl_id,
                ..
            } if name == "setitem_str" => bindings.push((
                *fun_decl_id,
                resolved_path
                    .as_ref()
                    .map(|p| p.segments.clone())
                    .unwrap_or_default(),
            )),
            CallTarget::FunctionPath {
                segments,
                fun_decl_id,
                ..
            } if segments.last().map(String::as_str) == Some("setitem_str") => {
                bindings.push((*fun_decl_id, segments.clone()))
            }
            _ => {}
        }
    }
    assert!(
        !bindings.is_empty(),
        "module-dict store must call setitem_str; ops={:?}",
        graph_ops(&graph)
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    );
    for (fun_decl_id, bound) in bindings {
        let id = fun_decl_id.expect("setitem_str call carries a FunDeclId");
        let callee = llbc.fn_by_id(id).expect("setitem_str FunDecl");
        assert_eq!(
            super::trait_impl_id_for_fundecl(callee),
            None,
            "module arm must name the inherent setitem_str, bound={bound:?} path={}",
            callee.item_meta.name_path()
        );
        assert_eq!(
            bound,
            super::registered_path_for_fun_decl(&llbc, callee).segments
        );
        assert!(
            !bound.iter().any(|s| s.starts_with("<impl ")),
            "inherent path must not be a trait-impl identity: {bound:?}"
        );
    }
}

#[test]
fn direct_ptradd_pointer_reuses_array_only_when_element_matches() {
    use crate::translator::rtyper::lltypesystem::lltype::{Array, LowLevelType, Ptr, PtrTarget};
    let byte = LowLevelType::Char;
    let wide = LowLevelType::Signed;
    let existing = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::new(byte.clone())),
    }));
    let same = super::lltype_for_direct_ptradd_pointer(Some(existing.clone()), &byte);
    assert_eq!(same, existing);
    let rebuilt = super::lltype_for_direct_ptradd_pointer(Some(existing), &wide);
    match &rebuilt {
        LowLevelType::Ptr(ptr) => match &ptr.TO {
            PtrTarget::Array(arr) => assert_eq!(arr.OF, wide),
            other => panic!("expected an array pointer, got {other:?}"),
        },
        other => panic!("expected a pointer, got {other:?}"),
    }
}

#[test]
fn atomic_field_layout_string_is_the_inner_scalar() {
    let atomic_decl = |def_id: u64, size: u64| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": [
                    {"Ident": ["core", 0]},
                    {"Ident": ["sync", 0]},
                    {"Ident": ["atomic", 0]},
                    {"Ident": ["Atomic", 0]}
                ],
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 8}
                }},
                "source_text": "pub struct Atomic;",
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": false
            },
            // Opaque: the scalar spelling comes from the `Atomic<T>`
            // type argument, not from reading its storage payload,
            // which a dependency artefact does not carry.
            "kind": "Opaque",
            "layout": [{
                "key": "x86_64-unknown-linux-gnu",
                "value": {
                    "size": size,
                    "align": size,
                    "variant_layouts": [{"field_offsets": [0]}],
                    "repr": {"repr_algo": "Rust", "transparent": false}
                }
            }]
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                null,
                atomic_decl(1, 8)
            ],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let adt_ty = |def_id: u64, args: serde_json::Value| {
        serde_json::from_value::<super::TyRef>(serde_json::json!({
            "Value": [8, {
                "Adt": {"id": def_id, "generics": {"types": args}}
            }]
        }))
        .expect("fixture TyRef parses")
    };

    // An integer atomic spells as the scalar it is layout-transparent
    // over, so the string-keyed field tables land on the same `Int` the
    // runtime publishes rather than their unknown-name `Ref` wildcard.
    // `AtomicUsize` is the alias `Atomic<usize>`.
    let usize_ty = adt_ty(
        1,
        serde_json::json!([{"Scalar": {"Integer": {"Unsigned": "Usize"}}}]),
    );
    assert_eq!(
        super::tyref_to_field_layout_string(
            &usize_ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "usize"
    );
    assert_eq!(
        crate::codewriter::call::get_type_flag("usize", &std::collections::HashSet::new()).1,
        majit_ir::value::Type::Int
    );

    // `AtomicPtr<T>` (`Atomic<*mut T>`) is a pointer word. Spell it as
    // the actual `*mut T` so fielddescrof does not treat the wrapper as
    // a nested struct, and so a non-GC pointee is not erased to
    // `PyObject`.
    let ptr_ty = adt_ty(
        1,
        serde_json::json!([{"RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]}]),
    );
    let ptr_str = super::tyref_to_field_layout_string(
        &ptr_ty,
        &llbc,
        &gc_struct_ids,
        &std::collections::HashSet::<String>::new(),
    );
    assert_eq!(ptr_str, "*mut u8");
    assert_eq!(
        crate::codewriter::call::get_type_flag(&ptr_str, &std::collections::HashSet::new()).1,
        majit_ir::value::Type::Ref
    );
}

#[test]
fn fieldless_enum_field_layout_is_its_discriminant_scalar() {
    let type_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["AccessMode", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 32}
            }},
            "source_text": "enum AccessMode { ReadWrite, ReadOnly }",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "kind": {"Enum": [
            {"name": "ReadWrite", "fields": [], "discriminant": 0, "attr_info": null},
            {"name": "ReadOnly", "fields": [], "discriminant": 1, "attr_info": null}
        ]},
        "layout": [{
            "key": "x86_64-unknown-linux-gnu",
            "value": {
                "size": 1,
                "align": 1,
                "discriminator": {"Branch": {
                    "offset": 0,
                    "int_ty": {"Unsigned": "U8"}
                }},
                "variant_layouts": [
                    {"field_offsets": []},
                    {"field_offsets": []}
                ],
                "repr": {"repr_algo": "Rust", "transparent": false}
            }
        }]
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [null, type_decl],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let enum_ty = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [8, {
            "Adt": {"id": 1, "generics": {"types": []}}
        }]
    }))
    .expect("fixture TyRef parses");
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();

    assert_eq!(
        super::tyref_to_field_layout_string(
            &enum_ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "u8"
    );
    assert_eq!(
        crate::codewriter::call::get_type_flag("u8", &std::collections::HashSet::new()),
        (
            majit_ir::descr::ArrayFlag::Unsigned,
            majit_ir::value::Type::Int,
            1,
        )
    );
}

/// A closure env whose capture is a fieldless enum (by shared
/// borrow, the rustc capture of a `Copy` tag) keeps that word as
/// the discriminant integer. `typed_ref_field_narrow_root` must
/// not emit `__cast_instance_intrinsic` against the enum class
/// (`rclass.py` `pairtype(InstanceRepr, InstanceRepr).convert_from_to`
/// has no integer operand), and the lowered graph must annotate.
#[test]
fn closure_capturing_fieldless_enum_annotates() {
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeInteger, SomeValue};
    use crate::translator::rtyper::call_registry::CallRegistry;
    use crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace;
    use std::rc::Rc;

    let span = serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 32}
    }});
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let enum_adt = serde_json::json!({"Adt": {"id": 1, "generics": generics}});
    let borrowed_enum = serde_json::json!({"Ref": ["_", enum_adt.clone(), "Shared"]});
    let closure_adt = serde_json::json!({"Adt": {"id": 2, "generics": generics}});
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let place =
        |id: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
    // rustc `call_once` of a capturing closure reads the tag as
    // `Discriminant(*&E)`. The place type after Deref is the enum;
    // the inner is the captured borrow.
    let deref_enum = serde_json::json!({
        "kind": {"Projection": [place(1, &borrowed_enum), "Deref"]},
        "ty": enum_adt
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                null,
                {
                    "def_id": 1,
                    "item_meta": {
                        "name": [{"Ident": ["fixture", 0]}, {"Ident": ["AccessMode", 0]}],
                        "span": span, "source_text": "enum AccessMode { ReadWrite, ReadOnly }",
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "kind": {"Enum": [
                        {"name": "ReadWrite", "fields": [], "discriminant": 0, "attr_info": null},
                        {"name": "ReadOnly", "fields": [], "discriminant": 1, "attr_info": null}
                    ]},
                    "layout": [{
                        "key": "x86_64-unknown-linux-gnu",
                        "value": {
                            "size": 1, "align": 1,
                            "discriminator": {"Branch": {
                                "offset": 0,
                                "int_ty": {"Unsigned": "U8"}
                            }},
                            "variant_layouts": [
                                {"field_offsets": []},
                                {"field_offsets": []}
                            ],
                            "repr": {"repr_algo": "Rust", "transparent": false}
                        }
                    }]
                },
                {
                    "def_id": 2,
                    "item_meta": {
                        "name": [
                            {"Ident": ["fixture", 0]},
                            {"Ident": ["capture", 0]},
                            {"Ident": ["closure", 0]}
                        ],
                        "span": span, "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": borrowed_enum.clone(),
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}
                    }]},
                    "layout": [{
                        "key": "x86_64-unknown-linux-gnu",
                        "value": {
                            "size": 1, "align": 1,
                            "variant_layouts": [{"field_offsets": [0]}],
                            "repr": {"repr_algo": "Rust", "transparent": false}
                        }
                    }],
                    "src": {"Closure": {"info": {"kind": "FnOnce"}}}
                }
            ],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [{"Ident": ["fixture", 0]}, {"Ident": ["capture", 0]}],
                    "span": span, "source_text": null,
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": true
                },
                "signature": {
                    "is_unsafe": false,
                    "inputs": [borrowed_enum.clone()],
                    "output": closure_adt.clone()
                },
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": closure_adt.clone()},
                        {"index": 1, "name": "op", "span": span, "ty": borrowed_enum.clone()}
                    ]},
                    "body": [
                        {"statements": [{
                            "span": span,
                            "kind": {"Assign": [
                                place(0, &closure_adt),
                                {"Aggregate": [
                                    {"Adt": [2, null, null, generics.clone()]},
                                    [{"Copy": place(1, &borrowed_enum)}]
                                ]}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                }}
            }, {
                "def_id": 1,
                "item_meta": {
                    "name": [{"Ident": ["fixture", 0]}, {"Ident": ["tag", 0]}],
                    "span": span, "source_text": null,
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": true
                },
                "signature": {
                    "is_unsafe": false,
                    "inputs": [borrowed_enum.clone()],
                    "output": u8_ty.clone()
                },
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": u8_ty.clone()},
                        {"index": 1, "name": "op", "span": span, "ty": borrowed_enum.clone()}
                    ]},
                    "body": [
                        {"statements": [{
                            "span": span,
                            "kind": {"Assign": [
                                place(0, &u8_ty),
                                {"Discriminant": deref_enum}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                }}
            }],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let borrowed_ty = serde_json::from_value::<super::TyRef>(borrowed_enum.clone())
        .expect("borrowed fieldless TyRef parses");
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let env = llbc.type_by_id(2).expect("closure env");

    assert!(
        !super::ty_is_raw_address(
            &borrowed_ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "a borrow of a fieldless enum is the tag integer, not a raw address"
    );
    assert_eq!(
        super::tyref_to_field_layout_string(
            &borrowed_ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "u8",
        "a borrow of a fieldless enum is the discriminant scalar"
    );
    assert_eq!(
        tyref_to_attr_value_type(&borrowed_ty, &llbc),
        ValueType::Int,
        "FORCE must seed Integer for a captured &E tag"
    );
    assert_eq!(
        tyref_to_attr_value_type_for_struct_field(
            &borrowed_ty,
            env,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Int,
        "closure-env FORCE rows match the Int-banked capture"
    );

    let graph = super::lower_function(&llbc, "capture").expect("lower capture");
    let cast_roots: Vec<&str> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| crate::model::cast_instance_root(&op.kind))
        .collect();
    assert!(
        cast_roots.is_empty(),
        "fieldless-enum capture must not emit __cast_instance_intrinsic; roots={cast_roots:?}"
    );

    let tag_graph = super::lower_function(&llbc, "tag").expect("lower tag");
    let tag_ops: Vec<&crate::model::OpKind> = tag_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| &op.kind)
        .collect();
    assert!(
        tag_ops
            .iter()
            .all(|kind| !matches!(kind, crate::model::OpKind::RawLoad { .. })),
        "Discriminant(*&E) of a fieldless enum is the tag, not a raw_load; ops={tag_ops:?}"
    );
    assert!(
        tag_ops.iter().all(|kind| {
            !matches!(
                kind,
                crate::model::OpKind::BinOp { op, .. } if op == "and_" || op == "&" || op == "and"
            )
        }),
        "fieldless Discriminant must not mask a loaded tag; ops={tag_ops:?}"
    );
    let tag_cast_roots: Vec<&str> = tag_ops
        .iter()
        .filter_map(|kind| crate::model::cast_instance_root(kind))
        .collect();
    assert!(
        tag_cast_roots.is_empty(),
        "fieldless Discriminant must not emit __cast_instance_intrinsic; roots={tag_cast_roots:?}"
    );

    let (_, _, fields, _, _, attrs, _, _) = super::derive_program_metadata(
        &llbc,
        &gc_struct_ids,
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc)),
    );
    let closure_key = "fixture::capture::closure";
    let attr_rows = attrs
        .get("capture::closure")
        .or_else(|| attrs.get(closure_key))
        .expect("closure FORCE rows");
    let layout_rows = fields
        .fields
        .get(closure_key)
        .or_else(|| fields.fields.get("capture::closure"))
        .cloned()
        .expect("closure layout rows");
    let layout_pairs: Vec<(String, String)> = layout_rows
        .iter()
        .map(|row| (row.name.clone(), row.ty.clone()))
        .collect();
    assert!(
        fields.fields.contains_key(closure_key),
        "intern keys the full closure path; keys={:?}",
        fields.fields.keys().collect::<Vec<_>>()
    );
    crate::annotator::classdesc::register_struct_fields_with_layout(
        closure_key,
        attr_rows,
        Some(layout_pairs.as_slice()),
    );
    crate::annotator::classdesc::register_struct_fields_with_layout(
        "capture::closure",
        attr_rows,
        Some(layout_pairs.as_slice()),
    );

    let ann = RPythonAnnotator::new(None, None, None, false);
    let bk = &ann.bookkeeper;
    bk.set_struct_fields(Rc::new(fields));
    let registry = CallRegistry::new(bk.clone());
    let lifted = function_graph_to_flowspace(&graph, &registry).expect("lift capture");
    crate::translator::simplify::simplify_graph(&lifted.graph.borrow(), None);
    let start = lifted.graph.borrow().startblock.clone();
    let n_inputs = start.borrow().inputargs.len();
    let cells: Vec<Option<SomeValue>> = (0..n_inputs)
        .map(|_| Some(SomeValue::Integer(SomeInteger::new(true, false))))
        .collect();
    ann.addpendingblock(&lifted.graph, &start, &cells);
    ann.complete_pending_blocks()
        .expect("a closure capturing a fieldless-enum tag annotates");

    let tag_ann = RPythonAnnotator::new(None, None, None, false);
    let tag_bk = &tag_ann.bookkeeper;
    tag_bk.set_struct_fields(Rc::new(
        super::derive_program_metadata(
            &llbc,
            &gc_struct_ids,
            super::no_immutable_fields(),
            &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(
                &llbc,
            )),
        )
        .2,
    ));
    let tag_registry = CallRegistry::new(tag_bk.clone());
    let tag_lifted = function_graph_to_flowspace(&tag_graph, &tag_registry).expect("lift tag");
    crate::translator::simplify::simplify_graph(&tag_lifted.graph.borrow(), None);
    let tag_start = tag_lifted.graph.borrow().startblock.clone();
    let tag_n = tag_start.borrow().inputargs.len();
    let tag_cells: Vec<Option<SomeValue>> = (0..tag_n)
        .map(|_| Some(SomeValue::Integer(SomeInteger::new(true, false))))
        .collect();
    tag_ann.addpendingblock(&tag_lifted.graph, &tag_start, &tag_cells);
    tag_ann
        .complete_pending_blocks()
        .expect("Discriminant of a borrowed fieldless-enum tag annotates");
}

/// Each receiver that reaches a `Vec` — an inline field, a `&mut Vec`
/// local, and a pointer loaded out of `Box<Vec<_>>` — becomes a getfield
/// of the buffer word. A declared virtualizable array is not a receiver
/// this helper is given: the caller keeps the vable array op.
#[test]
fn vec_index_retargets_each_receiver_shape() {
    use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType, VecFieldPart};

    fn input(graph: &mut FunctionGraph, name: &str) -> crate::flowspace::model::Variable {
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: name.into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .expect("input")
    }

    fn field_part(graph: &FunctionGraph, var: &crate::flowspace::model::Variable) -> VecFieldPart {
        graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } if op.result.as_ref() == Some(var) => {
                    field.vec_part
                }
                _ => None,
            })
            .expect("vec part")
    }

    let mut inline = FunctionGraph::new("inline_vec_field");
    let frame = input(&mut inline, "frame");
    let stack = inline
        .push_op_var(
            inline.startblock,
            OpKind::FieldRead {
                base: frame,
                field: FieldDescriptor::new("stack", Some("Vm".into())).with_inline_vec(true),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        )
        .expect("stack field");
    let bb = inline.startblock;
    let buf = super::retarget_vec_operand(&mut inline, bb, &stack, VecFieldPart::Buf);
    assert_ne!(buf, stack, "inline field keeps the original read");
    assert!(
        inline
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. }
                        if op.result.as_ref() == Some(&stack)
                            && field.inline_vec
                            && field.vec_part.is_none()
                )
            }),
        "the shared field read stays the whole Vec"
    );
    assert_eq!(field_part(&inline, &buf), VecFieldPart::Buf);

    let mut local = FunctionGraph::new("vec_local");
    let arg = input(&mut local, "v");
    let bb = local.startblock;
    let buf = super::retarget_vec_operand(&mut local, bb, &arg, VecFieldPart::Buf);
    assert_ne!(buf, arg);
    match &local.blocks[0].operations.last().unwrap().kind {
        OpKind::FieldRead { base, field, .. } => {
            assert_eq!(base, &arg, "&mut Vec local");
            assert_eq!(field.owner_root.as_deref(), Some("alloc::vec::Vec"));
            assert_eq!(field.vec_part, Some(VecFieldPart::Buf));
        }
        other => panic!("local receiver must getfield the buffer, got {other:?}"),
    }

    let mut boxed = FunctionGraph::new("box_vec");
    let frame = input(&mut boxed, "frame");
    let words = boxed
        .push_op_var(
            boxed.startblock,
            OpKind::FieldRead {
                base: frame,
                field: FieldDescriptor::new("operand_words", Some("GrainFrame".into())),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        )
        .expect("box pointer");
    let bb = boxed.startblock;
    let buf = super::retarget_vec_operand(&mut boxed, bb, &words, VecFieldPart::Len);
    assert_ne!(buf, words);
    match &boxed.blocks[0].operations.last().unwrap().kind {
        OpKind::FieldRead { base, field, .. } => {
            assert_eq!(base, &words, "Box<Vec> pointer");
            assert_eq!(field.vec_part, Some(VecFieldPart::Len));
            assert_eq!(field.name, "len");
        }
        other => panic!("box receiver must getfield the length word, got {other:?}"),
    }
}

/// `Vec::set_len` is `l.length = n`. An inline field keeps the owner;
/// a standalone receiver names `alloc::vec::Vec.len` so fielddescrof
/// can attach the Vec SizeDescr (`descr.py` `parent_descr`).
#[test]
fn vec_set_len_write_keeps_the_struct_owner() {
    use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType, VecFieldPart};

    let mut inline = FunctionGraph::new("inline_vec_set_len");
    let frame = inline
        .push_op_var(
            inline.startblock,
            OpKind::Input {
                name: "frame".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .expect("frame");
    let stack = inline
        .push_op_var(
            inline.startblock,
            OpKind::FieldRead {
                base: frame.clone(),
                field: FieldDescriptor::new("stack", Some("Vm".into())).with_inline_vec(true),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        )
        .expect("stack");
    let n = inline
        .push_op_var(inline.startblock, OpKind::ConstInt(0), true)
        .expect("n");
    match super::vec_set_len_write(&inline, &stack, n.clone()) {
        Some(OpKind::FieldWrite { base, field, .. }) => {
            assert_eq!(base, frame);
            assert_eq!(field.owner_root.as_deref(), Some("Vm"));
            assert_eq!(field.name, "stack");
            assert_eq!(field.vec_part, Some(VecFieldPart::Len));
        }
        other => panic!("inline set_len must write the owner field, got {other:?}"),
    }

    let mut local = FunctionGraph::new("vec_local_set_len");
    let arg = local
        .push_op_var(
            local.startblock,
            OpKind::Input {
                name: "v".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .expect("v");
    assert!(
        super::vec_set_len_write(&local, &arg, n).is_none(),
        "standalone Vec::set_len is not ll_set_length"
    )
}

/// `let v = &mut frame.items; let n = v.len(); v[0]` binds the inline
/// field once. Each use gets its own component read; neither read is
/// stacked on the other.
#[test]
fn shared_inline_vec_len_and_index_keep_the_field_read() {
    use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType, VecFieldPart};

    let mut graph = FunctionGraph::new("shared_inline_vec");
    let frame = graph
        .push_op_var(
            graph.startblock,
            OpKind::Input {
                name: "frame".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .expect("frame");
    let v = graph
        .push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: frame.clone(),
                field: FieldDescriptor::new("items", Some("Frame".into())).with_inline_vec(true),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        )
        .expect("items");
    let bb = graph.startblock;
    let len = super::retarget_vec_operand(&mut graph, bb, &v, VecFieldPart::Len);
    let elem = super::retarget_vec_operand(&mut graph, bb, &v, VecFieldPart::Buf);
    assert_ne!(len, v);
    assert_ne!(elem, v);
    assert_ne!(len, elem);
    let mut component_bases = Vec::new();
    let mut original_intact = false;
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        match &op.kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } if op.result.as_ref() == Some(&v) => {
                original_intact = field.inline_vec
                    && field.vec_part.is_none()
                    && field.name == "items"
                    && base == &frame;
                let _ = ty;
            }
            OpKind::FieldRead {
                base, field, ty, ..
            } if op.result.as_ref() == Some(&len) || op.result.as_ref() == Some(&elem) => {
                assert_eq!(
                    base, &frame,
                    "component read uses the struct, not a Vec word"
                );
                assert_eq!(field.name, "items");
                assert!(field.inline_vec);
                assert_ne!(base, &len);
                assert_ne!(base, &elem);
                component_bases.push((field.vec_part, ty.clone()));
            }
            _ => {}
        }
    }
    assert!(
        original_intact,
        "original field read must survive both uses"
    );
    assert_eq!(
        component_bases,
        vec![
            (Some(VecFieldPart::Len), ValueType::Int),
            (Some(VecFieldPart::Buf), ValueType::Ref(None)),
        ]
    );
}

#[test]
fn gcref_field_cast_to_tuple_graph_annotates_sometuple_and_pos0_list() {
    // Synthetic graph, not LLBC-backed: Input(GCREF) then unerase
    // marker then `__pos_0` FieldRead (`rtuple.py` `rtype_getitem`).
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::SomeValue;
    use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};
    use crate::translator::rtyper::call_registry::CallRegistry;
    use crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace;

    // Multi-word Vec items project to SomeList. Production kwargs
    // `Vec<*mut PyObject>` is one-word and projects to SomeRustVec
    // (`rust_vec_item_kind_for_spelling`); the lowering tests above
    // keep that spelling.
    let tuple_root = "Tuple<Vec<String>,Vec<String>>";
    let mut graph = FunctionGraph::new("kwargs_unerase");
    let entry = graph.startblock;
    let dstorage = graph
        .push_op_var(
            entry,
            OpKind::Input {
                name: "dstorage".to_string(),
                ty: ValueType::Ref(None),
                class_root: Some("GCREF".to_string()),
            },
            true,
        )
        .expect("input");
    graph.push_inputarg_var(entry, dstorage.clone());
    let unerased = graph
        .push_op_var(
            entry,
            crate::model::cast_instance_call(tuple_root, dstorage),
            true,
        )
        .expect("cast");
    let item0 = graph
        .push_op_var(
            entry,
            OpKind::FieldRead {
                base: unerased,
                field: FieldDescriptor::new("__pos_0", Some(tuple_root.to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        )
        .expect("pos_0");
    graph.set_return(entry, Some(item0));

    let marker = graph_ops(&graph)
        .iter()
        .find_map(|op| crate::model::cast_instance_root(&op.kind));
    assert_eq!(marker, Some(tuple_root));

    let ann = RPythonAnnotator::new(None, None, None, false);
    let registry = CallRegistry::new(ann.bookkeeper.clone());
    let lifted = function_graph_to_flowspace(&graph, &registry).expect("lift unerase graph");
    crate::translator::simplify::simplify_graph(&lifted.graph.borrow(), None);
    let start = lifted.graph.borrow().startblock.clone();
    let n_inputs = start.borrow().inputargs.len();
    let gcref = crate::translator::rtyper::llannotation::lltype_to_annotation(
        crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
    );
    let cells: Vec<Option<SomeValue>> = (0..n_inputs).map(|_| Some(gcref.clone())).collect();
    ann.addpendingblock(&lifted.graph, &start, &cells);
    ann.complete_pending_blocks()
        .expect("GCREF unerase then getitem(0) must annotate");
    let ret = lifted.graph.borrow().getreturnvar();
    let s_ret = ann.binding(&ret);
    assert!(
        matches!(s_ret, SomeValue::List(_)),
        "__pos_0 of the unerase tuple must annotate SomeList, got {s_ret:?}"
    );
}

#[test]
fn resolve_adt_field_keys_a_closure_capture_by_its_full_path() {
    use super::Lowering;
    use majit_charon_reader::ullbc::Unstructured;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let file = serde_json::json!({
        "charon_version": "0.1.201", "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [
                        {"Ident": ["fixture", 0]},
                        {"Ident": ["try_dispatch_binary_special", 0]},
                        {"Ident": ["closure", 0]}
                    ],
                    "span": span, "source_text": null,
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": true
                },
                "kind": {"Struct": [{
                    "name": null,
                    "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}
                }]}
            }],
            "fun_decls": [], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).unwrap();
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned_leaves = std::collections::HashSet::new();
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned_leaves,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let base = TyRef::Other(serde_json::json!({"Adt": {"id": 0, "builtin": null}}));
    let payload = serde_json::json!([null, 0]);
    let (owner_root, field_name, _, _, _) = lowering
        .resolve_adt_field(&base, &payload)
        .expect("closure field projection must resolve");
    assert_eq!(field_name, "__pos_0");
    assert_eq!(
        owner_root, "try_dispatch_binary_special::closure",
        "the shared leaf `closure` is withdrawn; the field keys the full path"
    );
}

/// A raw-pointer dest whose pointee holds a GC pointer that is not in
/// `_immutable_fields_` is not an overlay. `rewrite_op_getfield` raises
/// for a non-pure `getfield_raw_r`; `ann_cast_int_to_ptr` keeps that
/// dest off `SomePtr`.
#[test]
fn dest_overlay_rejects_raw_struct_with_mutable_gc_pointer_field() {
    use super::Lowering;
    use majit_charon_reader::ullbc::Unstructured;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let gc_header = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "pyobject", "PyObject"]),
        "kind": {"Struct": [{
            "name": "ob_refcnt",
            "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
            "attr_info": null
        }]}
    });
    let holder = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "MutableGcHolder"]),
        "kind": {"Struct": [
            {
                "name": "_discriminant",
                "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}},
                "attr_info": null
            },
            {
                "name": "w_obj",
                "ty": {"RawPtr": [adt_node(0), "Mut"]},
                "attr_info": null
            }
        ]}
    });
    let llbc = llbc_with_types("fixture", vec![gc_header, holder], vec![]);
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned_leaves = std::collections::HashSet::new();
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned_leaves,
        &gc_struct_ids,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let dest = TyRef::Other(serde_json::json!({"RawPtr": [adt_node(1), "Const"]}));
    assert!(
        lowering.dest_overlay_ptr_lltype(&dest).is_none(),
        "a mutable GC pointer field must not become SomePtr overlay"
    );
}

/// `LowlevelStringHeader` is the rstr `STR` `{hash, len}` view
/// (`rstr.py` `STR` / `LLHelpers.ll_strlen`). Dest overlay must not
/// recast it as a raw `SomePtr`; the 2-arg `cast_int_to_ptr` would
/// survive as CallMayForce inside a transparent helper.
#[test]
fn dest_overlay_rejects_lowlevel_string_header() {
    use super::Lowering;
    use majit_charon_reader::ullbc::Unstructured;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let header = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["pyre_object", "lowlevel_string", "LowlevelStringHeader"]),
        "kind": {"Struct": [
            {
                "name": "hash",
                "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}},
                "attr_info": null
            },
            {
                "name": "length",
                "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
                "attr_info": null
            }
        ]}
    });
    let llbc = llbc_with_types("fixture", vec![header], vec![]);
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned_leaves = std::collections::HashSet::new();
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned_leaves,
        &gc_struct_ids,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let dest = TyRef::Other(serde_json::json!({"RawPtr": [adt_node(0), "Const"]}));
    assert!(
        lowering.dest_overlay_ptr_lltype(&dest).is_none(),
        "LowlevelStringHeader is Ptr(STR), not a raw overlay SomePtr"
    );
}

#[test]
fn push_direct_ptradd_emits_the_lltype_helper() {
    let mut graph = FunctionGraph::new("test");
    let entry = graph.startblock;
    let ptr = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
        .expect("pointer value");
    FunctionGraph::set_concretetype_of_inline(&ptr, crate::model::ConcreteType::Signed);
    let before = ptr.concretetype();
    let count = graph
        .push_op_var(entry, OpKind::ConstInt(8), true)
        .expect("count");
    let result = push_direct_ptradd(
        &mut graph,
        entry,
        ptr.clone(),
        count.clone(),
        &crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Char,
    );
    assert_eq!(
        FunctionGraph::concretetype_of(&result),
        crate::model::ConcreteType::Signed,
        "direct_ptradd result is a raw pointer (int kind)"
    );
    match &graph.block(entry).operations.last().unwrap().kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            result_ty: ValueType::Int,
        } if segments.last().map(String::as_str) == Some("direct_ptradd") => {
            assert_eq!(ptr.concretetype(), before, "caller concretetype stays put");
            let crate::model::LinkArg::Value(got) = &args[0] else {
                panic!("pointer operand must be a variable, got {:?}", args[0]);
            };
            assert_ne!(got, &ptr, "a different item type needs a fresh variable");
            assert_eq!(args[1], crate::model::LinkArg::Value(count));
        }
        other => panic!("expected direct_ptradd call, got {other:?}"),
    }
}

/// A GC / classdef-less pointer fed to `direct_ptradd` is `rffi.cast`
/// onto `CArrayPtr` (`cast_ptr_to_int` then `cast_int_to_ptr`) so
/// `ann_direct_ptradd` sees `SomePtr`. `cast_ptr_to_int` alone would
/// make the operand `Integer`.
#[test]
fn push_direct_ptradd_rffi_casts_a_gc_pointer_operand() {
    let mut graph = FunctionGraph::new("test");
    let entry = graph.startblock;
    let ptr = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
        .expect("gc pointer");
    let count = graph
        .push_op_var(entry, OpKind::ConstInt(8), true)
        .expect("count");
    let result = push_direct_ptradd(
        &mut graph,
        entry,
        ptr.clone(),
        count,
        &crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
    );
    let ops = &graph.block(entry).operations;
    let leaves: Vec<&str> = ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str),
            _ => None,
        })
        .collect();
    assert!(
        leaves.contains(&"cast_ptr_to_int"),
        "rffi.cast starts at cast_ptr_to_int, got {leaves:?}"
    );
    assert!(
        leaves.contains(&"cast_int_to_ptr"),
        "rffi.cast ends at cast_int_to_ptr, got {leaves:?}"
    );
    let raw = ops
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(String::as_str) == Some("cast_int_to_ptr") => {
                assert_eq!(
                    args.len(),
                    2,
                    "rffi.cast emits cast_int_to_ptr(PTRTYPE, int)"
                );
                assert!(
                    matches!(
                        &args[0],
                        crate::model::LinkArg::Const(c)
                            if matches!(
                                c.value,
                                crate::flowspace::model::ConstValue::LowLevelType(_)
                            )
                    ),
                    "PtrT is the CArrayPtr constant, got {:?}",
                    args[0]
                );
                op.result.clone()
            }
            _ => None,
        })
        .expect("cast_int_to_ptr result");
    let call = ops.last().unwrap();
    match &call.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            result_ty: ValueType::Int,
        } if segments.last().map(String::as_str) == Some("direct_ptradd") => {
            assert_eq!(args[0], crate::model::LinkArg::Value(raw));
            assert_ne!(args[0], crate::model::LinkArg::Value(ptr));
            assert_eq!(call.result.as_ref(), Some(&result));
        }
        other => panic!("expected pointer direct_ptradd, got {other:?}"),
    }
}

#[test]
fn push_cast_ptr_to_int_emits_the_lltype_helper() {
    let mut graph = FunctionGraph::new("test");
    let entry = graph.startblock;
    let ptr = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
        .expect("pointer value");
    let signed = push_cast_ptr_to_int(&mut graph, entry, ptr.clone());
    assert_eq!(
        FunctionGraph::concretetype_of(&signed),
        crate::model::ConcreteType::Signed,
        "rbuiltin.py rtype_cast_ptr_to_int resulttype is Signed"
    );
    match &graph.block(entry).operations.last().unwrap().kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            result_ty: ValueType::Int,
        } if segments.last().map(String::as_str) == Some("cast_ptr_to_int") => {
            assert_eq!(args, &vec![ptr]);
        }
        other => panic!("expected cast_ptr_to_int call, got {other:?}"),
    }
    assert_eq!(
        graph
            .block(entry)
            .operations
            .last()
            .unwrap()
            .result
            .as_ref(),
        Some(&signed)
    );
}

#[test]
fn pointer_to_usize_cast_keeps_rpython_signed_then_unsigned_sequence() {
    let mut graph = FunctionGraph::new("test");
    let entry = graph.startblock;
    let ptr = graph
        .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
        .expect("pointer value");

    let (retype, result) = push_ptr_to_unsigned_cast(&mut graph, entry, ptr.clone());
    graph
        .block_mut(entry)
        .operations
        .push(crate::model::SpaceOperation {
            result: Some(result.clone()),
            kind: retype,
        });

    let ops = &graph.block(entry).operations;
    let (signed, signed_arg) = match &ops[1] {
        crate::model::SpaceOperation {
            result: Some(signed),
            kind:
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    result_ty: ValueType::Int,
                },
        } if segments.last().map(String::as_str) == Some("cast_ptr_to_int") => (signed, &args[0]),
        other => panic!("first cast must be Signed cast_ptr_to_int: {other:?}"),
    };
    assert_eq!(signed_arg, &ptr);
    match &ops[2] {
        crate::model::SpaceOperation {
            result: Some(actual_result),
            kind:
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    result_ty: ValueType::Unsigned,
                },
        } if segments.last().map(String::as_str) == Some("r_uint") => {
            assert_eq!(actual_result, &result);
            assert_eq!(&args[0], signed);
        }
        other => panic!("second cast must retype to Unsigned with r_uint: {other:?}"),
    }
}

/// `look_inside_iff` (`rlib/jit.py`) emits a `dont_look_inside` trampoline
/// that returns the same `Vec` as the original. Lowering records the
/// `*mut Vec<…>` container root `project_rust_vec` reads. The `i64`
/// FUNC.RESULT token is stamped later by `SemanticFunctionHeader` (`getkind`
/// of the raw header is `int`); the marker set is what makes that stamp
/// unconditional.
#[test]
fn look_inside_iff_vec_trampoline_records_rust_vec_root() {
    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    let context = super::LowerContext::new(&llbc);
    let opaque = super::dont_look_inside_set_of(&llbc);
    for (name, root) in [
        (
            "majit_rlib::lltypesystem::rvec::ll_vec_alloc_and_set_nonnull_r_trampoline",
            "*mut Vec<*mut u8>",
        ),
        (
            "majit_rlib::lltypesystem::rvec::ll_vec_alloc_and_set_nonnull_i_trampoline",
            "*mut Vec<usize>",
        ),
        (
            "majit_rlib::lltypesystem::rvec::ll_vec_alloc_and_set_nonnull_f_trampoline",
            "*mut Vec<f64>",
        ),
    ] {
        let fd = llbc
            .iter_local_fns()
            .find(|fd| fd.item_meta.name_path() == name)
            .unwrap_or_else(|| panic!("{name} in majit-rlib LLBC"));
        let policy = super::strip_crate_prefix(name);
        assert!(
            opaque.contains(&policy),
            "{policy} is the dont_look_inside trampoline"
        );
        let graph =
            super::lower_fun_decl(&context, fd).unwrap_or_else(|e| panic!("lower {name}: {e}"));
        assert_eq!(graph.return_container_root.as_deref(), Some(root), "{name}");
    }
}

#[test]
fn harden_withdraws_shape_divergent_bare_alias_and_tombstones_origin() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let a = rows(&[("handlerposition", "usize")]);
    let b = rows(&[
        ("valuestackdepth", "usize"),
        ("previous", "*mut FrameBlock"),
    ]);
    reg.fields.insert(
        "pyre_interpreter::pyopcode::FrameBlock".to_string(),
        a.clone(),
    );
    reg.fields
        .insert("pyre_interpreter::pyframe::FrameBlock".to_string(), b);
    // last-decl-wins bare alias as the dual-publish would leave it
    reg.fields.insert("FrameBlock".to_string(), a);
    let mut origins = std::collections::HashMap::new();
    // first-decl-wins origin as `or_insert` would leave it
    origins.insert("FrameBlock".to_string(), "pyopcode".to_string());
    let mut enums = std::collections::HashMap::new();

    harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, None);

    assert!(
        !reg.fields.contains_key("FrameBlock"),
        "shape-divergent duplicate leaf must lose its bare alias"
    );
    assert!(
        reg.fields
            .contains_key("pyre_interpreter::pyopcode::FrameBlock")
    );
    assert!(
        reg.fields
            .contains_key("pyre_interpreter::pyframe::FrameBlock")
    );
    assert_eq!(
        origins.get("FrameBlock").map(String::as_str),
        Some(""),
        "module-divergent duplicate leaf origin must be tombstoned"
    );
}

#[test]
fn harden_keeps_alias_for_equal_shape_same_module_duplicates() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let shape = rows(&[("x", "i64")]);
    reg.fields
        .insert("pyre_object::eval::Point".to_string(), shape.clone());
    reg.fields
        .insert("jit_artifact::eval::Point".to_string(), shape.clone());
    reg.fields.insert("Point".to_string(), shape.clone());
    let mut origins = std::collections::HashMap::new();
    origins.insert("Point".to_string(), "eval".to_string());
    let mut enums = std::collections::HashMap::new();

    harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, None);

    assert_eq!(reg.fields.get("Point"), Some(&shape));
    assert_eq!(origins.get("Point").map(String::as_str), Some("eval"));
}

#[test]
fn harden_deduplicates_full_and_canonical_aliases_by_struct_identity() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let shape = rows(&[("ob_header", "PyObject"), ("value", "*mut Wtf8Buf")]);
    let full = "pyre_object::unicodeobject::W_UnicodeObject";
    let canonical = "unicodeobject::W_UnicodeObject";
    reg.fields.insert(full.to_string(), shape.clone());
    reg.fields.insert(canonical.to_string(), shape.clone());
    reg.fields
        .insert("W_UnicodeObject".to_string(), shape.clone());
    let mut origins = std::collections::HashMap::new();
    origins.insert("W_UnicodeObject".to_string(), "unicodeobject".to_string());
    let mut enums = std::collections::HashMap::new();
    let sid = majit_ir::descr::StructId::from_canonical(canonical);
    let ids = std::collections::HashMap::from([
        (full.to_string(), Some(sid)),
        (canonical.to_string(), Some(sid)),
        ("W_UnicodeObject".to_string(), Some(sid)),
    ]);

    harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, Some(&ids));

    assert_eq!(reg.fields.get("W_UnicodeObject"), Some(&shape));
    assert_eq!(
        origins.get("W_UnicodeObject").map(String::as_str),
        Some("unicodeobject"),
        "two spellings of one StructId must not tombstone its origin"
    );
}

#[test]
fn harden_withdraws_discriminant_divergent_bare_enum_alias() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let mut origins = std::collections::HashMap::new();
    let map_a: std::collections::HashMap<i64, String> =
        [(0, "Continue".to_string()), (1, "Break".to_string())].into();
    let map_b: std::collections::HashMap<i64, String> = [(0, "Return".to_string())].into();
    let same_as_a = map_a.clone();
    let mut enums = std::collections::HashMap::new();
    enums.insert("pyre_interpreter::eval::StepResult".to_string(), map_a);
    enums.insert("jit_artifact::eval::StepResult".to_string(), map_b);
    // silent-winner bare alias as the dual-publish would leave it
    enums.insert("StepResult".to_string(), same_as_a.clone());
    enums.insert("pyre_object::flow::Verdict".to_string(), same_as_a.clone());
    enums.insert("jit_artifact::flow::Verdict".to_string(), same_as_a.clone());
    enums.insert("Verdict".to_string(), same_as_a.clone());
    // The `__discriminant`-only base rows as the enum arm registers
    // them — shape-identical across enums, so only the discriminant
    // divergence above can adjudicate the bare alias.
    let disc = rows(&[("__discriminant", "i64")]);
    reg.fields.insert("StepResult".to_string(), disc.clone());
    reg.fields.insert("Verdict".to_string(), disc.clone());

    harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, None);

    assert!(
        !enums.contains_key("StepResult"),
        "discriminant-divergent duplicate leaf must lose its bare alias"
    );
    assert!(enums.contains_key("pyre_interpreter::eval::StepResult"));
    assert!(enums.contains_key("jit_artifact::eval::StepResult"));
    assert_eq!(
        enums.get("Verdict"),
        Some(&same_as_a),
        "equal-map duplicates keep the alias"
    );
    // The bare enum-base row follows the discriminant signal: dropped
    // for the divergent leaf, kept for the equal-map one.
    assert!(
        !reg.fields.contains_key("StepResult"),
        "divergent enum base must lose its bare alias so it is not pre-minted as one merged class"
    );
    assert_eq!(
        reg.fields.get("Verdict"),
        Some(&disc),
        "equal-map enum base keeps its bare alias"
    );
}

#[test]
fn harden_leaves_unique_leaves_untouched() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let shape = rows(&[("ob_value", "i64")]);
    reg.fields.insert(
        "pyre_object::intobject::W_IntObject".to_string(),
        shape.clone(),
    );
    reg.fields.insert("W_IntObject".to_string(), shape.clone());
    let mut origins = std::collections::HashMap::new();
    origins.insert("W_IntObject".to_string(), "intobject".to_string());
    let mut enums = std::collections::HashMap::new();

    harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, None);

    assert_eq!(reg.fields.get("W_IntObject"), Some(&shape));
    assert_eq!(
        origins.get("W_IntObject").map(String::as_str),
        Some("intobject")
    );
}

#[test]
fn resolve_aggregate_adt_uses_dest_ty_generics_when_head_is_bare_id() {
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned = std::collections::HashSet::new();
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let kind = serde_json::json!({"Adt": [0, 1]});
    let dest = |signed: bool| {
        TyRef::Other(serde_json::json!({
            "Value": [7, option_adt_node(0, option_int_arg(signed))]
        }))
    };
    let owner_tail = |signed: bool| {
        let (path, leaf, _, _, _, _) = lowering
            .resolve_aggregate_adt(&kind, Some(&dest(signed)))
            .expect("Some constructor");
        assert_eq!(leaf, "Some");
        path.last().cloned().unwrap_or_default()
    };
    let usize_owner = owner_tail(false);
    let i64_owner = owner_tail(true);
    assert_eq!(usize_owner, "option::Option<usize>");
    assert_eq!(i64_owner, "option::Option<i64>");
    assert_ne!(
        usize_owner, i64_owner,
        "a bare type_id head must not intern both instantiations as Option"
    );
    let (bare_path, _, _, _, _, _) = lowering
        .resolve_aggregate_adt(&kind, None)
        .expect("presence check without dest_ty still resolves the ADT");
    assert_eq!(
        bare_path.last().map(String::as_str),
        Some("option::Option"),
        "without a dest_ty the presence check may keep the unsuffixed leaf"
    );
}

fn option_unit_arg() -> serde_json::Value {
    serde_json::json!({"Adt": {
        "id": 0,
        "builtin": "Tuple",
        "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
    }})
}

#[test]
fn resolve_aggregate_adt_option_unit_some_payload_is_void() {
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned = std::collections::HashSet::new();
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let kind = serde_json::json!({"Adt": [0, 1]});
    let dest = TyRef::Other(serde_json::json!({
        "Value": [7, option_adt_node(0, option_unit_arg())]
    }));
    let (_path, leaf, fields, _, _, _) = lowering
        .resolve_aggregate_adt(&kind, Some(&dest))
        .expect("Some constructor");
    assert_eq!(leaf, "Some");
    assert_eq!(
        fields
            .iter()
            .map(|(_, ty, _)| ty.as_str())
            .collect::<Vec<_>>(),
        vec!["()"],
        "Option<()> Some payload is the unit, so _names_without_voids drops it"
    );
    let shape = lowering
        .aggregate_shape(&kind, &dest, 1)
        .expect("Some(()) shape");
    match shape {
        super::AggregateShape::Ctor(ctor) => {
            assert_eq!(ctor.fields.len(), 1);
            assert!(
                ctor.fields[0].void,
                "unit payload occupies no slot and gets no FieldWrite"
            );
        }
        other => panic!("expected Ctor, got {other:?}"),
    }
}

fn range_inclusive_struct(def_id: u64) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": [
                {"Ident": ["core", 0]},
                {"Ident": ["ops", 0]},
                {"Ident": ["range", 0]},
                {"Ident": ["RangeInclusive", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Struct": [
            {"name": "start", "ty": {"TypeVar": {"Bound": [0, 0]}}, "attr_info": null},
            {"name": "end", "ty": {"TypeVar": {"Bound": [0, 0]}}, "attr_info": null},
            {"name": "exhausted", "ty": {"Scalar": "Bool"}, "attr_info": null}
        ]}
    })
}

#[test]
fn resolve_aggregate_adt_range_inclusive_uses_dest_ty_when_head_is_bare_id() {
    let llbc = llbc_with_types("fixture", vec![range_inclusive_struct(0)], vec![]);
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned = std::collections::HashSet::new();
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let kind = serde_json::json!({"Adt": [0]});
    let dest = |ty: serde_json::Value| {
        TyRef::Other(serde_json::json!({
            "Value": [7, option_adt_node(0, ty)]
        }))
    };
    let leaf_of = |ty: serde_json::Value| {
        let (_, leaf, _, _, _, _) = lowering
            .resolve_aggregate_adt(&kind, Some(&dest(ty)))
            .expect("RangeInclusive constructor");
        leaf
    };
    let i64_leaf = leaf_of(option_int_arg(true));
    let char_leaf = leaf_of(serde_json::json!({"Scalar": "Char"}));
    assert_eq!(i64_leaf, "ops::range::RangeInclusive<i64>");
    assert_eq!(char_leaf, "ops::range::RangeInclusive<char>");
    assert_ne!(
        i64_leaf, char_leaf,
        "a bare type_id head must not intern both instantiations as RangeInclusive"
    );
    let (_, bare_leaf, _, _, _, _) = lowering
        .resolve_aggregate_adt(&kind, None)
        .expect("presence check without dest_ty still resolves the ADT");
    assert_eq!(
        bare_leaf, "ops::range::RangeInclusive",
        "without a dest_ty the presence check may keep the unsuffixed identity"
    );
    // Product `new` spells the Aggregate head as a TypeDeclRef whose
    // `types` are still `TypeVar`. That is nonempty, so a "has any
    // type args" test interned `RangeInclusive<TypeVar#0>` for every
    // spec and unioned `start`. Prefer dest_ty when the head is not
    // concrete (`bookkeeper.py` `getuniqueclassdef`).
    let kind_tvar = serde_json::json!({"Adt": [{
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [{"TypeVar": {"Bound": [0, 0]}}],
                    "const_generics": [],
                    "trait_refs": []
                },
                "builtin": null
            }, null, null]});
    let leaf_tvar = |ty: serde_json::Value| {
        let (_, leaf, _, _, _, _) = lowering
            .resolve_aggregate_adt(&kind_tvar, Some(&dest(ty)))
            .expect("RangeInclusive constructor with TypeVar head");
        leaf
    };
    assert_eq!(
        leaf_tvar(option_int_arg(true)),
        "ops::range::RangeInclusive<i64>"
    );
    assert_eq!(
        leaf_tvar(serde_json::json!({"Scalar": "Char"})),
        "ops::range::RangeInclusive<char>"
    );
}

/// `Result::as_ref` stays a regular call to the extracted core body
/// `match *self { Ok(ref x) => Ok(x), Err(ref x) => Err(x) }`. The
/// tagged-sum Result repr is unchanged; only the caller rewrite is
/// gone (`guess_call_kind` regular).
#[test]
fn result_as_ref_keeps_regular_call_to_core_body() {
    use crate::model::{CallTarget, ExitSwitch, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    if !std::path::Path::new(path).is_file() {
        eprintln!("skipping: {path} is missing; run scripts/extract-llbc.py");
        return;
    }
    let llbc = Llbc::load(path).expect("load interpreter LLBC");
    let caller = super::lower_function(&llbc, "unicode_translate_error_str")
        .expect("lower unicode_translate_error_str");
    let as_ref_calls = caller
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "as_ref",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some("as_ref")
                    && segments.iter().any(|s| s == "result")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        as_ref_calls, 1,
        "caller keeps one regular call to the core as_ref body, got {as_ref_calls}"
    );

    let ctx = super::LowerContext::new(&llbc);
    let mut lowered_core_bodies = 0usize;
    for fd in llbc.iter_fun_decls() {
        if !fd.has_unstructured_body() {
            continue;
        }
        let name_path = fd.item_meta.name_path();
        if !name_path.ends_with("::result::<Impl>::as_ref") {
            continue;
        }
        let graph =
            super::lower_fun_decl(&ctx, fd).unwrap_or_else(|e| panic!("lower {name_path}: {e}"));
        let has_disc = graph.blocks.iter().any(|block| {
            matches!(block.exitswitch, Some(ExitSwitch::Value(_)))
                || block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. } if field.name == "__discriminant"
                    )
                })
        });
        assert!(has_disc, "{name_path}: core body matches on the Result tag");
        lowered_core_bodies += 1;
    }
    assert_eq!(
        lowered_core_bodies, 1,
        "extracted as_ref FunDecl bodies must lower, got {lowered_core_bodies}"
    );
}

/// Two `Code` declarations in one LLBC. `fixture::Code` strips to the
/// leaf; `fixture::other::Code` strips to `other::Code`.
fn duplicate_code_llbc() -> Llbc {
    llbc_with_types(
        "fixture",
        vec![
            code_struct(0, &["fixture", "Code"]),
            code_struct(1, &["fixture", "other", "Code"]),
        ],
        vec![],
    )
}

#[test]
fn single_llbc_lowering_tombstones_a_duplicate_leaf() {
    let llbc = duplicate_code_llbc();
    let tombstoned = super::tombstoned_leaves_of(&llbc);
    assert!(
        tombstoned.contains("Code"),
        "two Code declarations in one LLBC withdraw the bare leaf"
    );
    let context = super::LowerContext::new(&llbc);
    assert!(
        context.tombstoned_leaves.contains("Code"),
        "LowerContext paints with the LLBC tombstone set, not an empty one"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(1), &llbc, &tombstoned).as_deref(),
        Some("other::Code")
    );
}

#[test]
fn crate_root_tombstoned_leaf_paints_the_full_declaration_path() {
    let llbc = duplicate_code_llbc();
    let tombstoned = super::tombstoned_leaves_of(&llbc);
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &llbc, &tombstoned).as_deref(),
        Some("fixture::Code"),
        "crate::Code strips to the withdrawn leaf, so the paint keeps the full path"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &llbc, super::no_tombstoned_leaves())
            .as_deref(),
        Some("Code"),
        "without a tombstone the crate-root decl still paints the bare leaf"
    );

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let base = TyRef::Other(serde_json::json!({"Adt": {"id": 0, "builtin": null}}));
    let payload = serde_json::json!([null, 0]);
    let (owner_root, field_name, _, _, _) = lowering
        .resolve_adt_field(&base, &payload)
        .expect("field projection");
    assert_eq!(field_name, "n");
    assert_eq!(
        owner_root, "fixture::Code",
        "the field owner uses the same full path as the class root"
    );
}

#[test]
fn tombstoned_module_leaf_field_owner_matches_the_class_root() {
    let llbc = duplicate_code_llbc();
    let tombstoned = super::tombstoned_leaves_of(&llbc);
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(1), &llbc, &tombstoned).as_deref(),
        Some("other::Code")
    );
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let base = TyRef::Other(serde_json::json!({"Adt": {"id": 1, "builtin": null}}));
    let payload = serde_json::json!([null, 0]);
    let (owner_root, _, _, _, _) = lowering
        .resolve_adt_field(&base, &payload)
        .expect("field projection");
    assert_eq!(
        owner_root, "other::Code",
        "fixture::other::Code is tombstoned, so the field owner is the crate-stripped path"
    );
}

/// `a::same::Code` and `b::same::Code` strip to one key. The leaf is
/// tombstoned, and each declaration keeps its crate on the owner and
/// on the merged field row.
#[test]
fn cross_crate_same_module_leaf_keeps_distinct_rows() {
    fn code_with_field(def_id: u64, path: &[&str], field: &str, ty: &str) -> serde_json::Value {
        let mut decl = code_struct(def_id, path);
        decl["kind"]["Struct"][0]["name"] = serde_json::json!(field);
        decl["kind"]["Struct"][0]["ty"] =
            serde_json::json!({"Scalar": {"Integer": {"Unsigned": ty}}});
        decl
    }
    let a = llbc_with_types(
        "a",
        vec![code_with_field(0, &["a", "same", "Code"], "n", "U64")],
        vec![],
    );
    let b = llbc_with_types(
        "b",
        vec![code_with_field(0, &["b", "same", "Code"], "m", "U32")],
        vec![],
    );
    let mut facts = super::DuplicateLeafFacts::default();
    facts.absorb(super::DuplicateLeafFacts::discover(&a));
    facts.absorb(super::DuplicateLeafFacts::discover(&b));
    let cross = facts.tombstoned_leaves();
    assert!(cross.contains("Code"));
    assert!(
        cross.contains("same::Code"),
        "the colliding stripped path is recorded, not only the leaf"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &a, &cross).as_deref(),
        Some("a::same::Code")
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &b, &cross).as_deref(),
        Some("b::same::Code")
    );

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum_a = super::AccumulatorFacts::build(&a, &body);
    let gc_struct_ids_a = super::harvest_declared_gc_facts(&a).gc_struct_ids();
    let lowering_a = super::Lowering::new(
        &a,
        "a".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &cross,
        &gc_struct_ids_a,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum_a,
        &super::RootStackAnalyzer::new(&a, &super::RootStackState::new(&a)),
        false,
        false,
    )
    .unwrap();
    let base = TyRef::Other(serde_json::json!({"Adt": {"id": 0, "builtin": null}}));
    let payload = serde_json::json!([null, 0]);
    let (owner_a, field_a, _, id_a, _) = lowering_a
        .resolve_adt_field(&base, &payload)
        .expect("field projection");
    assert_eq!(owner_a, "a::same::Code");
    assert_eq!(field_a, "n");
    let kind = serde_json::json!({"Adt": [0, null]});
    let (ctor_path, ctor_leaf, _, ctor_id, _, _) = lowering_a
        .resolve_aggregate_adt(&kind, None)
        .expect("constructor");
    let ctor_owner = if ctor_path.is_empty() {
        ctor_leaf
    } else {
        format!("{}::{}", ctor_path.join("::"), ctor_leaf)
    };
    assert_eq!(ctor_owner, "a::same::Code");
    assert_eq!(
        id_a,
        Some(majit_ir::descr::StructId::from_canonical("a::same::Code"))
    );
    assert_eq!(
        ctor_id,
        majit_ir::descr::StructId::from_canonical("a::same::Code")
    );

    let accum_b = super::AccumulatorFacts::build(&b, &body);
    let gc_struct_ids_b = super::harvest_declared_gc_facts(&b).gc_struct_ids();
    let lowering_b = super::Lowering::new(
        &b,
        "b".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &cross,
        &gc_struct_ids_b,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum_b,
        &super::RootStackAnalyzer::new(&b, &super::RootStackState::new(&b)),
        false,
        false,
    )
    .unwrap();
    let (owner_b, field_b, _, id_b, _) = lowering_b
        .resolve_adt_field(&base, &payload)
        .expect("field projection");
    assert_eq!(owner_b, "b::same::Code");
    assert_eq!(field_b, "m");
    assert_eq!(
        id_b,
        Some(majit_ir::descr::StructId::from_canonical("b::same::Code"))
    );
    assert_ne!(id_a, id_b);

    let prog = super::build_semantic_program_from_llbcs(&[a, b]).expect("merged program");
    assert_eq!(
        prog.struct_fields
            .fields
            .get("a::same::Code")
            .map(|rows| rows[0].name.as_str()),
        Some("n")
    );
    assert_eq!(
        prog.struct_fields
            .fields
            .get("b::same::Code")
            .map(|rows| rows[0].name.as_str()),
        Some("m")
    );
    assert!(
        !prog.struct_fields.fields.contains_key("same::Code"),
        "the merged program must not keep one row for the shared stripped key"
    );
    assert_ne!(
        prog.struct_ids
            .get("a::same::Code")
            .and_then(|id| id.as_ref()),
        prog.struct_ids
            .get("b::same::Code")
            .and_then(|id| id.as_ref())
    );
}

#[test]
fn harden_withdraws_shape_divergent_variant_leaf_alias() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let shape_a = rows(&[("__pos_0", "i64")]);
    let shape_b = rows(&[("__pos_0", "*mut Foo"), ("__pos_1", "u32")]);
    // Two `Outcome` enums in different modules whose `Ok` variant rows
    // diverge — the bare `Outcome::Ok` alias is last-decl-wins (shape_b),
    // as the variant dual-publish leaves it.
    reg.fields
        .insert("crateX::moduleA::Outcome::Ok".to_string(), shape_a.clone());
    reg.fields
        .insert("moduleA::Outcome::Ok".to_string(), shape_a.clone());
    reg.fields
        .insert("crateY::moduleB::Outcome::Ok".to_string(), shape_b.clone());
    reg.fields
        .insert("moduleB::Outcome::Ok".to_string(), shape_b.clone());
    reg.fields.insert("Outcome::Ok".to_string(), shape_b);
    // A shape-identical `Status::Active` duplicate keeps its bare alias.
    let shape_s = rows(&[("__pos_0", "u8")]);
    reg.fields
        .insert("moduleC::Status::Active".to_string(), shape_s.clone());
    reg.fields
        .insert("moduleD::Status::Active".to_string(), shape_s.clone());
    reg.fields
        .insert("Status::Active".to_string(), shape_s.clone());

    let mut origins = std::collections::HashMap::new();
    let mut enums = std::collections::HashMap::new();

    harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, None);

    assert!(
        !reg.fields.contains_key("Outcome::Ok"),
        "shape-divergent variant-leaf alias must be withdrawn (bare last-segment pass missed it)"
    );
    assert!(reg.fields.contains_key("moduleA::Outcome::Ok"));
    assert!(reg.fields.contains_key("moduleB::Outcome::Ok"));
    assert_eq!(
        reg.fields.get("Status::Active"),
        Some(&shape_s),
        "shape-identical variant duplicate keeps its bare alias"
    );
}

#[test]
fn harden_keeps_a_struct_leaf_that_only_a_variant_shares() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    let state = rows(&[("int_max_str_digits", "i32")]);
    reg.fields
        .insert("module::sys::state::SysState".to_string(), state.clone());
    reg.fields.insert("SysState".to_string(), state.clone());
    let base = rows(&[("__discriminant", "i64")]);
    reg.fields
        .insert("baseobjspace::SpaceCacheInstance".to_string(), base.clone());
    reg.fields.insert("SpaceCacheInstance".to_string(), base);
    let variant = rows(&[("__pos_0", "RetainedSpaceCache<SysState>")]);
    reg.fields.insert(
        "baseobjspace::SpaceCacheInstance::SysState".to_string(),
        variant.clone(),
    );
    reg.fields
        .insert("SpaceCacheInstance::SysState".to_string(), variant);
    let mut origins = std::collections::HashMap::from([(
        "SysState".to_string(),
        "module::sys::state".to_string(),
    )]);
    let mut enums = std::collections::HashMap::new();

    let tombstoned = harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums, None);

    assert_eq!(
        reg.fields.get("SysState"),
        Some(&state),
        "a variant does not publish the bare leaf, so it cannot make the struct's alias ambiguous"
    );
    assert!(!tombstoned.contains("SysState"));
    assert_eq!(
        origins.get("SysState").map(String::as_str),
        Some("module::sys::state")
    );
}

#[test]
fn root_bracket_erasure_names_the_pin_a_shadow_stack_len_reads() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // `dunder_import_inner` reads a slot with `shadow_stack_len()` and
    // then `shadow_stack_get` of that depth. The len names the pin that
    // runs after every pin already executed.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow = |dest: u64, src: u64| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                6 => "pyre_object::gc_roots::pin_root",
                7 => "pyre_object::gc_roots::shadow_stack_get",
                8 => "pyre_object::gc_roots::shadow_stack_len",
                _ => "pyre_interpreter::misc::as_long",
            }
            .to_string(),
        )
    };
    // An unknown call keeps the bracket by naming the len, which is
    // watched. `pin_root` and `shadow_stack_len` are root API, so this
    // callback is not what admits them.
    let touches = |_: &RegularCall| false;
    let llbc = fixture_llbc();
    let assemble = |steps: Vec<(Vec<serde_json::Value>, serde_json::Value)>| -> Unstructured {
        let mut blocks = Vec::new();
        for (i, (stmts, mut term)) in steps.into_iter().enumerate() {
            let target = (i + 1) as u64;
            if let Some(call) = term.get_mut("Call") {
                call["target"] = serde_json::json!(target);
            }
            blocks.push(block(stmts, term));
        }
        let drop_at = blocks.len();
        blocks.push(block(vec![], drop_guard(3, (drop_at + 1) as u64)));
        blocks.push(block(vec![], serde_json::json!("Return")));
        serde_json::from_value(serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
            "body": blocks,
        }))
        .expect("fixture Unstructured parses")
    };
    let plan_of = |body: &Unstructured| {
        super::analyze_root_brackets_with(
            &llbc,
            body,
            &super::MovedOutLocals::with_set(body, bit_set::BitSet::new()),
            &name_of,
            &touches,
        )
    };
    let push = || (vec![], call(1, vec![], 3, 0));
    let len = || (vec![], call(8, vec![], 4, 0));
    let pin = |value: u64, dest: u64| (vec![], call(6, vec![copy(value)], dest, 0));
    let get = |index: u64| (vec![], call(7, vec![copy(index)], 6, 0));

    // The only pin is after the len, so the len names slot 0: local `_1`.
    let happy = assemble(vec![push(), len(), pin(1, 5), get(4)]);
    let plan = plan_of(&happy);
    assert!(
        plan.scopes.contains(3),
        "shadow_stack_len names the pin that follows it"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![1]));
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);
    // Callee 9 is `fixture::len_reload::call`, not a `gc_roots` leaf.
    // `lower_erased_root_bracket_call` answers that block through
    // `get_answer` so the skipped capture temps are never resolved.
    assert_eq!(plan.get_answer(3), Some(1usize));
    assert_eq!(plan.slot_temps.get(&4), Some(&3), "the len result");
    assert!(
        plan.slot_temps.get(&5).is_none(),
        "pin_root's result is the value, not a depth"
    );
    assert!(plan.base_results.get(&4).is_none());
    for bb in [1usize, 2, 3] {
        assert_eq!(plan.free_sites.get(&bb), Some(&3), "free call in bb{bb}");
    }

    // `_7 = copy _4` is the same depth. Both temps bind nothing.
    let copied = assemble(vec![
        push(),
        len(),
        pin(1, 5),
        (
            vec![assign(7, serde_json::json!({"Use": [copy(4), "No"]}))],
            call(7, vec![copy(7)], 6, 0),
        ),
    ]);
    let plan = plan_of(&copied);
    assert!(
        plan.scopes.contains(3),
        "a copy of the len names the same pin"
    );
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);
    assert_eq!(plan.slot_temps.get(&4), Some(&3));
    assert_eq!(plan.slot_temps.get(&7), Some(&3));

    // A pin already executed is not the one the len names. The get
    // answers with the second pin.
    let second = assemble(vec![push(), pin(1, 8), len(), pin(2, 9), get(4)]);
    let plan = plan_of(&second);
    assert!(
        plan.scopes.contains(3),
        "the len names the pin that runs after it"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![1, 2]));
    assert_eq!(plan.get_sites, vec![(4usize, 2usize)]);
    assert_eq!(plan.slot_temps.get(&4), Some(&3));
    assert!(plan.slot_temps.get(&8).is_none());
    assert!(plan.slot_temps.get(&9).is_none());

    // `len + 1` names the pin after the one the len names. A copy of
    // that sum is the argument temporary MIR builds for the get.
    let plus = assemble(vec![
        push(),
        len(),
        pin(1, 5),
        pin(2, 8),
        (
            vec![
                assign(
                    10,
                    serde_json::json!({"BinaryOp": ["Add", copy(4), usize_lit(1)]}),
                ),
                assign(11, serde_json::json!({"Use": [copy(10), "No"]})),
            ],
            call(7, vec![copy(11)], 6, 0),
        ),
    ]);
    let plan = plan_of(&plus);
    assert!(
        plan.scopes.contains(3),
        "len + 1 names the pin after the len's next pin"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![1, 2]));
    assert_eq!(plan.get_sites, vec![(4usize, 2usize)]);
    assert_eq!(plan.slot_temps.get(&10), Some(&3));
    assert_eq!(plan.slot_temps.get(&11), Some(&3));

    // A `(len + 0, len + 1, len + 2)` tuple whose fields feed the
    // gets names the same pins as the separate-local spelling.
    let field = |i: u64, f: u64| {
        serde_json::json!({"Copy": {
            "kind": {"Projection": [place(i), {"Field": [null, f]}]},
            "ty": ty()
        }})
    };
    let add = |dest: u64, k: u64| {
        assign(
            dest,
            serde_json::json!({"BinaryOp": ["Add", copy(4), usize_lit(k)]}),
        )
    };
    let separate = assemble(vec![
        push(),
        len(),
        pin(1, 5),
        pin(2, 8),
        pin(7, 9),
        (vec![add(10, 0)], call(7, vec![copy(10)], 6, 0)),
        (vec![add(11, 1)], call(7, vec![copy(11)], 6, 0)),
        (vec![add(12, 2)], call(7, vec![copy(12)], 6, 0)),
    ]);
    let packed = assemble(vec![
        push(),
        len(),
        pin(1, 5),
        pin(2, 8),
        pin(7, 9),
        (
            vec![
                add(10, 0),
                add(11, 1),
                add(12, 2),
                assign(
                    13,
                    serde_json::json!({
                        "Aggregate": ["Tuple", [copy(10), copy(11), copy(12)]]
                    }),
                ),
                assign(14, serde_json::json!({"Use": [field(13, 0), "No"]})),
                assign(15, serde_json::json!({"Use": [field(13, 1), "No"]})),
                assign(16, serde_json::json!({"Use": [field(13, 2), "No"]})),
            ],
            call(7, vec![copy(14)], 6, 0),
        ),
        (vec![], call(7, vec![copy(15)], 6, 0)),
        (vec![], call(7, vec![copy(16)], 6, 0)),
    ]);
    let separate_plan = plan_of(&separate);
    let packed_plan = plan_of(&packed);
    assert!(
        separate_plan.scopes.contains(3),
        "separate len+k locals name the three pins"
    );
    assert_eq!(packed_plan.scopes, separate_plan.scopes);
    assert_eq!(packed_plan.get_sites, separate_plan.get_sites);

    let rejected = [
        (
            "a get before its pin",
            assemble(vec![push(), len(), get(4), pin(1, 5)]),
        ),
        (
            "the len passed to an unknown call",
            assemble(vec![
                push(),
                len(),
                pin(1, 5),
                get(4),
                (vec![], call(5, vec![copy(4)], 9, 0)),
            ]),
        ),
        (
            "a len after the only pin",
            assemble(vec![push(), pin(1, 5), len(), get(4)]),
        ),
        (
            "a statement that also writes the len",
            assemble(vec![
                push(),
                (
                    vec![assign(4, serde_json::json!({"Use": [copy(1), "No"]}))],
                    call(8, vec![], 4, 0),
                ),
                pin(1, 5),
                get(4),
            ]),
        ),
        (
            "a second len writing the same local",
            assemble(vec![push(), len(), len(), pin(1, 5), get(4)]),
        ),
        (
            "a pin value assigned inside the bracket",
            assemble(vec![
                push(),
                len(),
                (
                    vec![assign(1, serde_json::json!({"Use": [copy(2), "No"]}))],
                    call(6, vec![copy(1)], 5, 0),
                ),
                get(4),
            ]),
        ),
        (
            "a borrow of the pin value",
            assemble(vec![
                push(),
                len(),
                (vec![borrow(9, 1)], call(6, vec![copy(1)], 5, 0)),
                get(4),
            ]),
        ),
    ];
    for (why, body) in rejected {
        let plan = plan_of(&body);
        assert!(!plan.scopes.contains(3), "{why} keeps the bracket");
    }
}

#[test]
fn root_bracket_erasure_erases_free_pins_indexed_by_len_after_open() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // `slice_unpack`: `push_roots`, `base = shadow_stack_len()` once
    // right after the open, three free `pin_root`s, then
    // `shadow_stack_get(base + k)`. Each pin is the k-th slot, the
    // same numbering `pin_roots` uses, and the scope's close rewinds
    // them.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                6 => "pyre_object::gc_roots::pin_root",
                7 => "pyre_object::gc_roots::shadow_stack_get",
                8 => "pyre_object::gc_roots::shadow_stack_len",
                _ => "pyre_interpreter::sliceobject::eval_slice_index",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    let assemble = |steps: Vec<(Vec<serde_json::Value>, serde_json::Value)>| -> Unstructured {
        let mut blocks = Vec::new();
        for (i, (stmts, mut term)) in steps.into_iter().enumerate() {
            let target = (i + 1) as u64;
            if let Some(call) = term.get_mut("Call") {
                call["target"] = serde_json::json!(target);
            }
            blocks.push(block(stmts, term));
        }
        let drop_at = blocks.len();
        blocks.push(block(vec![], drop_guard(3, (drop_at + 1) as u64)));
        blocks.push(block(vec![], serde_json::json!("Return")));
        serde_json::from_value(serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 3, "locals": (0..=20).map(local).collect::<Vec<_>>()},
            "body": blocks,
        }))
        .expect("fixture Unstructured parses")
    };
    let add = |dest: u64, k: u64| {
        assign(
            dest,
            serde_json::json!({"BinaryOp": ["Add", copy(4), usize_lit(k)]}),
        )
    };
    let body = assemble(vec![
        (vec![], call(1, vec![], 3, 0)),
        (vec![], call(8, vec![], 4, 0)),
        (vec![], call(6, vec![copy(1)], 5, 0)),
        (vec![], call(6, vec![copy(2)], 8, 0)),
        (vec![], call(6, vec![copy(7)], 9, 0)),
        (vec![add(10, 0)], call(7, vec![copy(10)], 11, 0)),
        (vec![add(12, 1)], call(7, vec![copy(12)], 13, 0)),
        (vec![add(14, 2)], call(7, vec![copy(14)], 15, 0)),
    ]);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &body,
        &super::MovedOutLocals::with_set(&body, bit_set::BitSet::new()),
        &name_of,
        &touches,
    );
    assert!(
        plan.scopes.contains(3),
        "len-after-open plus free pin_root x3 must erase"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![1, 2, 7]));
    assert_eq!(
        plan.get_sites,
        vec![(5usize, 1usize), (6usize, 2usize), (7usize, 7usize)]
    );
    assert_eq!(plan.slot_temps.get(&4), Some(&3), "the len result");
    for bb in [1usize, 2, 3, 4, 5, 6, 7] {
        assert_eq!(plan.free_sites.get(&bb), Some(&3), "free call in bb{bb}");
    }
}

/// Real-LLBC anchor for the unsigned `saturating_sub` wrapper itself.
/// In-lowering rewrite of `core::intrinsics::saturating_sub` replaced
/// the Call's goto with non-MIR diamond arms, so `lower_framestate`
/// `stub_dead_block`'d the Return continuation (`flowcontext.py`
/// `make_next_block`) and the wrapper became a raise-only stub. The
/// clamp rewrite now runs after the body lowers (`saturating_mul`);
/// the usize wrapper must keep the `lt`/`sub`
/// diamond and must not close startblock onto exceptblock.
/// `#[ignore]`d (loads the object LLBC); run with
/// `cargo test -p majit-translate --lib saturating_sub_wrapper_diamond_real
/// -- --ignored`.
#[test]
#[ignore]
fn saturating_sub_wrapper_diamond_real() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let llbc = Llbc::load(path).expect("load object LLBC");
    let context = super::LowerContext::new(&llbc);
    let mut saw = 0usize;
    for fd in llbc.iter_fun_decls() {
        if fd.item_meta.name_path() != "core::num::<Impl>::saturating_sub" {
            continue;
        }
        if !fd.has_unstructured_body() {
            continue;
        }
        let graph =
            super::lower_fun_decl(&context, fd).expect("lower core::num::<Impl>::saturating_sub");
        let start = &graph.blocks[graph.startblock.0];
        assert!(
            !(start.exits.len() == 1 && start.exits[0].target == graph.exceptblock),
            "{}: startblock is a raise-only stub (framestate stub_dead_block of the Return continuation)",
            graph.name
        );
        let lt_guards = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt"))
            .count();
        let subs = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "sub"))
            .count();
        assert!(
            lt_guards >= 1 && subs >= 1,
            "{}: expected the unsigned clamp diamond (lt={lt_guards}, sub={subs})",
            graph.name
        );
        saw += 1;
    }
    assert!(
        saw >= 1,
        "object LLBC: expected at least one unstructured core::num::<Impl>::saturating_sub"
    );
}

/// A translated opcode helper is a PyPy frame method, not a generic
/// classdef-less reference.  Its receiver is the per-frame red argument
/// that owns `pycode`, globals, and locals, so the front must retain the
/// concrete `PyFrame` root on the corresponding Input operation.
#[test]
#[ignore]
fn opcode_impl_receiver_keeps_pyframe_input_root() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::eval::<Impl>::pop_value")
        .expect("lower eval::<Impl>::pop_value");
    let receiver = graph
        .block(graph.startblock)
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::Input { class_root, .. }
                if op.result.as_ref() == graph.block(graph.startblock).inputargs.first() =>
            {
                class_root.as_deref()
            }
            _ => None,
        });
    assert_eq!(
        receiver,
        Some("PyFrame"),
        "opcode impl receiver must retain the concrete per-frame red identity"
    );

    let program = super::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        crate::HostStaticAddrs::default(),
        &[],
        &["pop_value", "pop_top", "build_list", "opcode_build_list"],
    )
    .expect("build filtered opcode program");
    let pyframe_roots: Vec<_> = program
        .struct_fields
        .fields
        .keys()
        .filter(|root| root.rsplit("::").next() == Some("PyFrame"))
        .cloned()
        .collect();
    assert!(
        pyframe_roots.len() > 1,
        "fixture must exercise more than one lookup spelling"
    );
    let mut pyframe_ids: Vec<_> = pyframe_roots
        .iter()
        .map(|root| {
            program
                .struct_ids
                .get(root)
                .copied()
                .flatten()
                .expect("every qualified PyFrame alias has an identity")
        })
        .collect();
    pyframe_ids.sort();
    pyframe_ids.dedup();
    assert_eq!(
        pyframe_ids.len(),
        1,
        "all PyFrame lookup spellings must name one class identity"
    );

    let mut trait_impl_owners = crate::TraitImplOwners::new();
    for function in &program.functions {
        let (Some(trait_qualified), Some(owner)) =
            (&function.trait_qualified, &function.self_ty_root)
        else {
            continue;
        };
        trait_impl_owners
            .entry(trait_qualified.clone())
            .or_default()
            .insert(owner.clone());
    }
    let unique_impls = crate::unique_trait_impl_roots(&program, trait_impl_owners);
    let frame_owner = unique_impls
        .iter()
        .find(|(trait_name, _)| trait_name.ends_with("::OpcodeStepExecutor"))
        .map(|(_, owner)| owner)
        .expect("OpcodeStepExecutor has one concrete owner");
    assert!(
        frame_owner.ends_with("::PyFrame"),
        "the production unique-impl census must preserve the qualified frame owner: \
             {frame_owner:?}"
    );
    assert_eq!(
        program.struct_ids.get(frame_owner).copied().flatten(),
        Some(pyframe_ids[0]),
        "the retained owner spelling must resolve to the frame's StructId"
    );
}

/// `[OpcodeStepExecutor, to_bool]` on the real interpreter LLBC is the
/// `PyFrame` override.  The census printed in a failure names which
/// uniqueness miss is live: several owner strings of one StructId, a
/// missing override row, or a direct path that stayed the trait default.
#[test]
#[ignore]
fn opcode_step_executor_to_bool_direct_path_is_pyframe_override() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    let mut owners = std::collections::BTreeSet::new();
    let mut to_bool_override = false;
    let mut to_bool_default = false;
    for fd in llbc.iter_local_fns() {
        let name_path = fd.item_meta.name_path();
        let trait_path = super::trait_impl_trait_path_for_fundecl(&llbc, fd);
        let trait_leaf = trait_path.as_deref().and_then(|p| p.rsplit("::").next());
        if trait_leaf == Some("OpcodeStepExecutor")
            && let Some((owner, method)) = super::impl_method_owner_for_fundecl(&llbc, fd)
        {
            owners.insert(owner);
            if method == "to_bool" {
                to_bool_override = true;
            }
        }
        if name_path.ends_with("::OpcodeStepExecutor::to_bool") {
            to_bool_default = true;
        }
    }
    let program = super::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(&llbc),
        crate::HostStaticAddrs::default(),
        &[],
        &["to_bool"],
    )
    .expect("build filtered to_bool program");
    let mut ids = Vec::new();
    for owner in &owners {
        let id = program
            .struct_ids
            .get(owner)
            .copied()
            .flatten()
            .unwrap_or_else(|| majit_ir::descr::StructId::from_canonical(owner));
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let leaf_identities = crate::distinct_struct_identities_by_leaf(&program);
    let mut call_control = crate::call::CallControl::new();
    crate::bind_trait_default_direct_paths(&program, &mut call_control, &leaf_identities);
    let graph = call_control
        .function_graphs()
        .get(&crate::CallPath::from_segments([
            "OpcodeStepExecutor",
            "to_bool",
        ]))
        .expect("direct path registered");
    assert!(
        to_bool_default && to_bool_override,
        "LLBC must contain both the trait default and the PyFrame override; \
             owners={owners:?} ids={}",
        ids.len()
    );
    assert_eq!(
        ids.len(),
        1,
        "OpcodeStepExecutor owners must be one struct identity; owners={owners:?}"
    );
    assert!(
        graph.name.contains("eval") && graph.name.contains("<Impl>"),
        "direct to_bool graph is {:?}; owners={owners:?} ({} strings, {} struct ids); \
             classified_override={to_bool_override}",
        graph.name,
        owners.len(),
        ids.len()
    );
    assert!(
        !graph.name.contains("OpcodeStepExecutor"),
        "raising default stayed on the direct path: {}",
        graph.name
    );
}

/// `Option<&[u8]>` None is a null window data word, not
/// `convert_const(None)` on `StringRepr`. `Option<&mut [u8]>` is a
/// list of unsigned.
#[test]
fn option_niche_null_cast_shared_u8_slice_is_window() {
    use crate::model::ValueType;

    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(&llbc, &body);
    let tombstoned = std::collections::HashSet::new();
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let slice = serde_json::json!({"Slice": [u8_ty, null]});
    let option_shared = TyRef::Other(option_adt_node(
        0,
        serde_json::json!({"Ref": ["Erased", slice.clone(), "Shared"]}),
    ));
    let option_mut = TyRef::Other(option_adt_node(
        0,
        serde_json::json!({"Ref": ["Erased", slice, "Mut"]}),
    ));
    assert!(
        matches!(
            lowering.option_niche_null_cast(&option_shared),
            Some((root, ty)) if root == "&[u8]" && !matches!(ty, ValueType::Str)
        ),
        "Option<&[u8]> None is a null window data word"
    );
    assert!(
        !matches!(
            lowering.option_niche_null_cast(&option_mut),
            Some((_, ValueType::Str))
        ),
        "Option<&mut [u8]> None is not a string root"
    );
}

/// A cell parameter's startblock input is the `MutRef<T>` class, not the
/// pointee. A `dont_look_inside` subject keeps the raw word.
#[test]
fn mut_ref_cell_param_input_names_the_cell_unless_residual() {
    use crate::model::OpKind;

    let llbc = fixture_llbc();
    let tomb = no_tombstoned_leaves();
    let gc_struct_ids = std::collections::HashSet::new();
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let pointee = serde_json::json!({"RawPtr": [{"Adt": {"id": 7}}, "Mut"]});
    let mut_ref = serde_json::json!({"Ref": ["Erased", pointee, "Mut"]});
    let locals: majit_charon_reader::ullbc::Locals = serde_json::from_value(serde_json::json!({
        "arg_count": 1,
        "locals": [
            {
                "index": 0,
                "name": null,
                "span": span,
                "ty": {"Scalar": {"Integer": {"Signed": "I64"}}}
            },
            {"index": 1, "name": "slot", "span": span, "ty": mut_ref}
        ]
    }))
    .unwrap();
    let pointee_ty: TyRef = serde_json::from_value(pointee).unwrap();
    let root = super::mut_ref_cell_root_of(
        &pointee_ty,
        &llbc,
        tomb,
        &gc_struct_ids,
        &std::collections::HashSet::<String>::new(),
    )
    .expect("raw GC pointer behind &mut is a cell");
    let slot = |graph: &FunctionGraph| {
        graph
            .block(graph.startblock)
            .operations
            .iter()
            .find(|op| matches!(&op.kind, OpKind::Input { name, .. } if name == "slot"))
            .expect("slot input")
            .clone()
    };

    let cells = super::gc_mut_ref_param_locals(
        &locals,
        &llbc,
        tomb,
        &gc_struct_ids,
        false,
        &std::collections::HashSet::<String>::new(),
    );
    assert!(cells.contains(&1));
    let mut graph = FunctionGraph::new("c::holder");
    super::pygraph_initial_block(
        &mut graph,
        &locals,
        &llbc,
        None,
        tomb,
        &cells,
        &gc_struct_ids,
        false,
        &std::collections::HashSet::<String>::new(),
    );
    let OpKind::Input { ty, class_root, .. } = &slot(&graph).kind else {
        panic!("slot op is an input");
    };
    assert_eq!(class_root.as_deref(), Some(root.as_str()));
    assert_eq!(ty, &ValueType::Ref(Some(root)));

    let residual = super::gc_mut_ref_param_locals(
        &locals,
        &llbc,
        tomb,
        &gc_struct_ids,
        true,
        &std::collections::HashSet::<String>::new(),
    );
    assert!(residual.is_empty());
    let mut residual_graph = FunctionGraph::new("c::holder");
    super::pygraph_initial_block(
        &mut residual_graph,
        &locals,
        &llbc,
        None,
        tomb,
        &residual,
        &gc_struct_ids,
        false,
        &std::collections::HashSet::<String>::new(),
    );
    let OpKind::Input { class_root, .. } = &slot(&residual_graph).kind else {
        panic!("slot op is an input");
    };
    assert!(
        !class_root.as_deref().unwrap_or("").starts_with("MutRef<"),
        "a residual subject's parameter keeps the raw word"
    );
}

/// A residual callee — no body, or `dont_look_inside` — is not given a
/// cell signature. The traced callee with the same `&mut` GC pointer is.
#[test]
fn mut_ref_cell_signature_skips_a_residual_callee() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let mut_ref = serde_json::json!({
        "Ref": ["Erased", {"RawPtr": [{"Adt": {"id": 7}}, "Mut"]}, "Mut"]
    });
    let item = |leaf: &str| {
        serde_json::json!({
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": [leaf, 0]}
            ],
            "span": span.clone(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        })
    };
    let signature = serde_json::json!({
        "is_unsafe": false,
        "inputs": [mut_ref],
        "output": {"Tuple": []}
    });
    let body = serde_json::json!({
        "Unstructured": {
            "locals": {"arg_count": 0, "locals": []},
            "body": [],
            "span": span.clone()
        }
    });
    let decl = |id: u64, leaf: &str, body: serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": item(leaf),
            "signature": signature.clone(),
            "body": body
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [
                decl(0, "traced", body.clone()),
                decl(1, "opaque", serde_json::Value::Null),
                decl(2, "marked", body)
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let mut dont_look_inside = std::collections::HashSet::new();
    dont_look_inside.insert("marked".to_string());
    assert!(!super::fun_decl_is_residual(
        llbc.fn_by_id(0).unwrap(),
        &dont_look_inside
    ));
    assert!(super::fun_decl_is_residual(
        llbc.fn_by_id(1).unwrap(),
        &dont_look_inside
    ));
    assert!(super::fun_decl_is_residual(
        llbc.fn_by_id(2).unwrap(),
        &dont_look_inside
    ));

    let subject: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []},
        "body": [],
        "span": span
    }))
    .unwrap();
    let accum = super::AccumulatorFacts::build(&llbc, &subject);
    let tombstoned_leaves = std::collections::HashSet::new();
    let gc_struct_ids = std::collections::HashSet::new();
    let lowering = super::Lowering::new(
        &llbc,
        "fixture::holder".into(),
        &subject,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned_leaves,
        &gc_struct_ids,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(&llbc, &super::RootStackState::new(&llbc)),
        false,
        false,
    )
    .unwrap();
    let call = |id: u64| {
        serde_json::from_value::<super::CallFunc>(serde_json::json!({
            "Regular": {
                "kind": {"Fun": id},
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }))
        .unwrap()
    };
    let traced = lowering
        .gc_mut_ref_signature(&call(0), &[])
        .expect("a traced &mut GC pointer is a cell");
    assert_eq!(traced.0, Some(0));
    assert!(traced.1[0].1.starts_with("MutRef<"));
    assert!(lowering.gc_mut_ref_signature(&call(1), &[]).is_none());
    assert!(lowering.gc_mut_ref_signature(&call(2), &[]).is_none());
}

/// `Arguments::new` holds `Option<Vec<PyObjectRef>>`. A `Vec` of one-word
/// items is the address of its raw header, never null, so that `Option`
/// keeps its enum tag instead of a null test, and no ref-kinded null
/// `Vec` joins the int-kinded header. Loads the interpreter LLBC, so
/// ignored by default.
#[test]
#[ignore]
fn option_of_one_word_item_vec_keeps_the_enum_tag() {
    use crate::model::OpKind;
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let context = super::LowerContext::new(&llbc);
    let fd = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta.name_path() == "pyre_interpreter::argument::<Impl>::new"
                && fd.signature.inputs.len() == 7
        })
        .expect("Arguments::new");
    let graph = super::lower_fun_decl(&context, fd).expect("lower Arguments::new");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert!(
        !ops().any(|op| matches!(&op.kind,
                OpKind::Call { result_ty: ValueType::Ref(Some(root)), .. }
                    if root.starts_with("Vec<"))),
        "no null `Vec` pointer is cast for the Option<Vec<PyObjectRef>> test"
    );
    assert!(
        ops().any(|op| matches!(&op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|s| s == "ll_vec_length_r"))),
        "the keyword-name count reads the header through ll_vec_length_r"
    );
}

#[test]
fn struct_ptr_write_copies_inlined_struct_field_leaves() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-interpreter LLBC");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .source_text
                .as_deref()
                .is_some_and(|text| text.starts_with("pub fn new(mut frame: PyFrame)"))
        })
        .expect("FrameBox::new");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower FrameBox::new");
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.iter().map(String::as_str).eq(["core", "ptr", "write"])
            )
        }),
        "FrameBox::new still has core::ptr::write"
    );
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "ob_header"
            )
        }),
        "FrameBox::new must not store the inlined ob_header as one word"
    );
    let header_leaves: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. }
                if matches!(field.name.as_str(), "ob_type" | "w_class")
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|owner| owner.ends_with("PyObject")) =>
            {
                Some(field.name.as_str())
            }
            _ => None,
        })
        .collect();
    assert!(
        header_leaves.contains(&"ob_type") && header_leaves.contains(&"w_class"),
        "offset-0 header is the `super` chain: inner PyObject descriptors, got {header_leaves:?}"
    );
}

#[test]
fn snapshot_frame_ctor_stores_inlined_header_as_leaves() {
    use crate::model::OpKind;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load pyre-interpreter LLBC");
    let fd = llbc
            .iter_local_fns()
            .find(|fd| {
                fd.item_meta.source_text.as_deref().is_some_and(|text| {
                    text.starts_with("fn build_snapshot_frame(&self, allocation: FrameLocalsArrayAllocation) -> PyFrame")
                })
            })
            .expect("build_snapshot_frame");
    let context = super::LowerContext::new(&llbc);
    let graph = super::lower_fun_decl(&context, fd).expect("lower build_snapshot_frame");
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "ob_header"
            )
        }),
        "PyFrame ctor must not store the inlined header as one word"
    );
    let header_leaves: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. }
                if matches!(field.name.as_str(), "ob_type" | "w_class")
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|owner| owner.ends_with("PyObject")) =>
            {
                Some(field.name.as_str())
            }
            _ => None,
        })
        .collect();
    assert!(
        header_leaves.contains(&"ob_type") && header_leaves.contains(&"w_class"),
        "offset-0 header is the `super` chain: inner PyObject descriptors, got {header_leaves:?}"
    );
}

/// Native constant slice bounds must retain their actual rtyper provenance
/// when simplification removes a forwarding block input.
#[test]
#[ignore]
fn slice_get_rangefrom_preserves_constant_bound_rtype() {
    use crate::translator::rtyper::call_registry::CallRegistry;
    use crate::translator::rtyper::cutover::{DualGateOutcome, dual_gate_check_with_registry};
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-module.ullbc"
    ))
    .expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_module::module::unicodedata::interp_ucd::ucd_method_args",
    )
    .expect("lower actual RangeFrom get caller");
    let program = super::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        crate::HostStaticAddrs::default(),
        &[],
        &["ucd_method_args"],
    )
    .expect("derive actual native registry metadata");
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    bk.set_struct_fields(std::rc::Rc::new(program.struct_fields));
    bk.set_enum_variant_by_discriminant(std::rc::Rc::new(program.enum_variant_by_discriminant));
    let registry = CallRegistry::new(bk);
    let result = dual_gate_check_with_registry(
        &graph,
        &registry,
        &crate::codewriter::call::GraphStore::default(),
    )
    .expect("dual gate must not error");
    let DualGateOutcome::Match { real_constants, .. } = result else {
        panic!("{result:?}");
    };
    let bound = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments == &["__getslice_rangefrom"] => Some(args[1].clone()),
            _ => None,
        })
        .expect("actual live slice bound");
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    assert_eq!(real_constants.get(&bound), Some(&LowLevelType::Unsigned));
    bound.set_concretetype(None);
    crate::codewriter::type_state::apply_from_flowspace_constants(&real_constants);
    assert_eq!(bound.concretetype(), Some(LowLevelType::Unsigned));
}

/// `bytearrayobject.py` `descr_repr` appends three values for a non-printable
/// byte — `'\\x'` and the two nibble characters.  The expansion consumed
/// the template's `\x` piece as a match condition only, so a JIT-compiled
/// repr rendered `bytearray(b'ff')` where CPython renders
/// `bytearray(b'\xff')`.  Assert the literal is emitted and folded first.
#[test]
fn lower_hex_byte_02_expansion_emits_the_backslash_x_literal() {
    use crate::model::{CallTarget, ConcreteType, FunctionGraph, OpKind};
    let mut graph = FunctionGraph::new("t");
    let value = graph.alloc_value_var_with_type(ConcreteType::Unknown);
    let result = graph.alloc_value_var_with_type(ConcreteType::Unknown);
    let pieces = vec!["\\x".to_string(), String::new()];
    let ops = super::emit_lower_hex_byte_02_ops(&mut graph, &pieces, &value, result.clone());

    let literal = ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.first().is_some_and(|s| s == "__str_const")
                        && segments.get(1).is_some_and(|s| s == "\\x"))
            })
            .expect("the template's leading `\\x` must be emitted, not only matched");
    let literal = literal.result.clone().expect("__str_const has a result");

    // `'\\x'` is upstream's FIRST buf.append, so it must be the left
    // operand of the first concat.
    let first_add = ops
        .iter()
        .find(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
        .expect("the parts fold through add");
    let OpKind::BinOp { lhs, .. } = &first_add.kind else {
        unreachable!("matched an add above")
    };
    assert_eq!(lhs.id(), literal.id(), "`\\x` must be concatenated first");

    assert_eq!(
        ops.last().and_then(|op| op.result.clone()).map(|v| v.id()),
        Some(result.id()),
        "the final op must reuse the displaced format result var"
    );
}

fn melt_abort_fixture_input(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    name: &str,
) -> crate::flowspace::model::Variable {
    let var = graph
        .push_op_var(
            block,
            OpKind::Input {
                name: name.into(),
                ty: ValueType::Ref(Some("Obj".into())),
                class_root: None,
            },
            true,
        )
        .unwrap();
    graph.push_inputarg_var(block, var.clone());
    var
}

fn melt_abort_field_write(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    base: crate::flowspace::model::Variable,
    field: &str,
    value: crate::flowspace::model::Variable,
) {
    graph.push_op_var(
        block,
        OpKind::FieldWrite {
            base,
            field: FieldDescriptor::new(field, Some("Obj".into())),
            value: LinkArg::Value(value),
            ty: ValueType::Int,
        },
        false,
    );
}

fn melt_abort_format_of(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    x: crate::flowspace::model::Variable,
) {
    let arg = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::function_path(["fmt", "rt", "Argument", "new_display"]),
                args: crate::model::call_args(vec![x]),
                result_ty: ValueType::Ref(Some("Argument".into())),
            },
            true,
        )
        .unwrap();
    let args = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::function_path(["fmt", "Arguments", "new"]),
                args: crate::model::call_args(vec![arg]),
                result_ty: ValueType::Ref(Some("Arguments".into())),
            },
            true,
        )
        .unwrap();
    graph.push_op_var(
        block,
        OpKind::Call {
            target: CallTarget::function_path(["alloc", "fmt", "format"]),
            args: crate::model::call_args(vec![args]),
            result_ty: ValueType::Str,
        },
        true,
    );
}

fn melt_abort_field_names(graph: &FunctionGraph) -> Vec<String> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } => Some(field.name.clone()),
            _ => None,
        })
        .collect()
}

/// `input obj; obj.field = v; panic!("{}", x)` — the store is a heap
/// side effect (`transform_dead_op_vars` never removes `setfield`).
#[test]
fn melt_keeps_field_write_into_input_object() {
    let mut graph = FunctionGraph::new("melt_input_store");
    let entry = graph.startblock;
    let obj = melt_abort_fixture_input(&mut graph, entry, "obj");
    let x = melt_abort_fixture_input(&mut graph, entry, "x");
    let v = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
    melt_abort_field_write(&mut graph, entry, obj, "field", v);
    melt_abort_format_of(&mut graph, entry, x);
    graph.set_raise_implicit(entry, "AssertionError");

    super::melt_abort_message_chain(&mut graph, entry);

    assert!(
        melt_abort_field_names(&graph).iter().any(|n| n == "field"),
        "store into an Input object must survive: {graph:#?}"
    );
    assert!(
        leftover_fmt_message_paths(&graph).is_empty(),
        "panic! format operands must still melt: leftover={:?} graph={graph:#?}",
        leftover_fmt_message_paths(&graph)
    );
}

/// `let inner = obj.inner; inner.f = v; panic!` — the base is a
/// FieldRead, not a fresh malloc, so the store stays.
#[test]
fn melt_keeps_field_write_through_field_read_base() {
    let mut graph = FunctionGraph::new("melt_field_read_store");
    let entry = graph.startblock;
    let obj = melt_abort_fixture_input(&mut graph, entry, "obj");
    let v = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
    let inner = graph
        .push_op_var(
            entry,
            OpKind::FieldRead {
                base: obj,
                field: FieldDescriptor::new("inner", Some("Obj".into())),
                ty: ValueType::Ref(Some("Inner".into())),
                pure: true,
            },
            true,
        )
        .unwrap();
    melt_abort_field_write(&mut graph, entry, inner, "f", v);
    graph.set_raise_implicit(entry, "AssertionError");

    super::melt_abort_message_chain(&mut graph, entry);

    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "inner"
        )),
        "FieldRead that feeds a live store must survive: {graph:#?}"
    );
    assert!(
        melt_abort_field_names(&graph).iter().any(|n| n == "f"),
        "store through a FieldRead base must survive: {graph:#?}"
    );
}

/// Pieces array constructed in a predecessor region block, filled,
/// then fed only to `Arguments::new` on the abort — melted
/// (`remove_simple_mallocs` of a fresh aggregate).
#[test]
fn melt_drops_field_writes_into_fresh_pieces_array() {
    let mut graph = FunctionGraph::new("melt_pieces_array");
    let pred = graph.startblock;
    let abort = graph.create_block();
    let byte = graph
        .push_op_var(pred, OpKind::ConstInt(b'x' as i64), true)
        .unwrap();
    let pieces = graph
        .push_op_var(
            pred,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor("Array"),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(Some("Array".into())),
            },
            true,
        )
        .unwrap();
    graph.push_op_var(
        pred,
        OpKind::FieldWrite {
            base: pieces.clone(),
            field: FieldDescriptor::new("__pos_0", Some("Array".into())),
            value: LinkArg::Value(byte),
            ty: ValueType::Int,
        },
        false,
    );
    let pieces_in = graph.alloc_value_var();
    graph.push_inputarg_var(abort, pieces_in.clone());
    graph.set_goto(pred, abort, vec![pieces]);
    graph
        .push_op_var(
            abort,
            OpKind::Call {
                target: CallTarget::function_path(["fmt", "Arguments", "new"]),
                args: crate::model::call_args(vec![pieces_in]),
                result_ty: ValueType::Ref(Some("Arguments".into())),
            },
            true,
        )
        .unwrap();
    graph.set_raise_implicit(abort, "AssertionError");

    super::melt_abort_message_chain(&mut graph, abort);

    let leftover = leftover_fmt_message_paths(&graph);
    assert!(
        leftover.is_empty(),
        "Arguments::new of a dead pieces array must melt: leftover={leftover:?} graph={graph:#?}"
    );
    assert!(
        melt_abort_field_names(&graph).is_empty(),
        "stores into the fresh pieces array must melt: {graph:#?}"
    );
    let has_array_ctor = graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name == "Array"
        )
    });
    assert!(
        !has_array_ctor,
        "dead pieces Array ctor must melt: {graph:#?}"
    );
}

/// A user method named `kind_name` whose result is unused before a
/// panic is not a `CanRemoveBuiltins` extern; the call stays.
#[test]
fn melt_keeps_user_method_named_kind_name() {
    let mut graph = FunctionGraph::new("melt_kind_name_method");
    let entry = graph.startblock;
    let obj = melt_abort_fixture_input(&mut graph, entry, "obj");
    graph.push_op_var(
        entry,
        OpKind::Call {
            target: CallTarget::method("kind_name", Some("Obj".into())),
            args: crate::model::call_args(vec![obj]),
            result_ty: ValueType::Str,
        },
        true,
    );
    graph.set_raise_implicit(entry, "AssertionError");

    super::melt_abort_message_chain(&mut graph, entry);

    let has_kind_name = graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "kind_name"
        )
    });
    assert!(
        has_kind_name,
        "user method kind_name must survive: {graph:#?}"
    );
}

/// The exception carrier's cached object is PyPy's `W_Root` field. Its
/// registry spelling must retain `PyObjectRef`; erasing it to a generic
/// reference makes `InstanceRepr._setup_repr` choose the root object
/// pointer while field reads expect the concrete PyObject repr.
#[test]
#[ignore]
fn pyerror_exc_object_registry_retains_instance_ref() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let (_, _, fields, _, _, _, _, _) = super::derive_program_metadata(
        &llbc,
        &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc)),
    );
    let (owner, ty) = fields
        .fields
        .iter()
        .filter(|(owner, _)| owner.ends_with("PyError"))
        .find_map(|(owner, rows)| {
            rows.iter()
                .find(|row| row.name == "exc_object")
                .map(|row| (owner.as_str(), row.ty.as_str()))
        })
        .map(|(owner, ty)| (owner.to_string(), ty.to_string()))
        .expect("PyError.exc_object registry row");
    assert!(
        ty == "PyObjectRef"
            || ty.ends_with("::PyObjectRef")
            || ty.ends_with("pyobject::PyObject")
            || ty == "*mut PyObject",
        "{owner}.exc_object must retain its concrete object-pointer spelling, got {ty}"
    );
    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    bk.set_struct_fields(std::rc::Rc::new(fields));
    let crate::annotator::model::SomeValue::Instance(projected) = bk.project_struct_field_type(&ty)
    else {
        panic!("{owner}.exc_object field must project to an instance annotation")
    };
    assert!(
        projected.classdef.is_some(),
        "{owner}.exc_object field must project to the concrete PyObject class"
    );
}

/// `mini_buffer_params` constructs `Option<(*mut u8, usize)>`.  The Rust
/// payload is inline, but the translator gives the internal aggregate one
/// GC-managed `Tuple<...>` class identity; the per-instantiation `Some`
/// row must therefore be seeded with that same identity before any shared
/// `InstanceRepr` can cache the variant layout.
#[test]
#[ignore]
fn mini_buffer_params_preregisters_tuple_payload_row() {
    use crate::annotator::model::SomeValue;
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real interpreter LLBC");
    let program = super::build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(&llbc),
        crate::HostStaticAddrs::default(),
        &["module::_cffi_backend::cbuffer"],
        &["mini_buffer_params", "call_once"],
    )
    .expect("build mini_buffer_params semantic program");
    assert!(
        program
            .functions
            .iter()
            .any(|function| function.name == "mini_buffer_params"),
        "mini_buffer_params semantic function"
    );
    let tuple_owner = program
        .functions
        .iter()
        .flat_map(|function| &function.graph().blocks)
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name == "Tuple<*mut u8,usize>" => Some(name.clone()),
            _ => None,
        })
        .expect("concrete tuple aggregate class");

    let (variant_owner, rows) = program
        .struct_fields
        .fields
        .iter()
        .find(|(owner, _)| owner.ends_with("Option<(*mut u8,usize)>::Some"))
        .expect("concrete Option<tuple>::Some registry row");
    assert_eq!(
        rows,
        &vec![("__pos_0".to_string(), "(*mut u8,usize)".to_string())],
        "the concrete tuple payload must not fall back to the ??T template"
    );
    let variant_owner = variant_owner.clone();

    let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    bk.set_struct_fields(std::rc::Rc::new(program.struct_fields));
    bk.set_enum_variant_by_discriminant(std::rc::Rc::new(program.enum_variant_by_discriminant));
    for root in bk.struct_root_names() {
        bk.getuniqueclassdef_for_struct_root(&root)
            .expect("project struct root");
    }
    bk.pre_register_enum_variant_classes();

    let enum_root = variant_owner
        .strip_suffix("::Some")
        .expect("Some-qualified owner");
    let variant = bk
        .getuniqueclassdef_for_enum_variant(enum_root, "Some")
        .expect("resolve pre-registered Some classdef");
    let variant = variant.borrow();
    let Some(SomeValue::Instance(payload)) = variant.attrs.get("__pos_0").map(|attr| &attr.s_value)
    else {
        panic!(
            "Some.__pos_0 must carry the concrete tuple class, got {:?}",
            variant.attrs.get("__pos_0").map(|attr| &attr.s_value)
        )
    };
    assert_eq!(
        payload
            .classdef
            .as_ref()
            .expect("tuple payload classdef")
            .borrow()
            .name,
        tuple_owner,
        "the pre-registered payload and constructor must share one class"
    );
    assert!(
        !variant.attrs["__pos_0"].readonly,
        "the constructor payload must be an instance field before repr setup"
    );
}

/// Niche `Option<*mut T>::ok_or_else` is a null test. `Some` is that
/// pointer — not `__discriminant` / `__pos_0` of the object.
#[test]
fn niche_option_ok_or_else_is_a_null_test() {
    let mut graph = FunctionGraph::new("test_niche_ok_or_else");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let env = graph.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("ok_or_else", Some("Option".into())),
                args: crate::model::call_args(vec![recv, env]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let mut site = option_ok_or_else_site(result);
    site.recv_niche = true;
    site.niche_null_cast = Some(("pyobject::PyObject".into(), ValueType::Ref(None)));
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "ok_or_else"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 1);
    let mut saw_disc = false;
    let mut saw_pos0_on_option = false;
    let mut saw_is = false;
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
                _ => {}
            }
        }
    }
    assert!(!saw_disc, "niche ok_or_else read __discriminant");
    assert!(
        !saw_pos0_on_option,
        "niche ok_or_else read Option Some.__pos_0"
    );
    assert!(saw_is, "niche ok_or_else did not null-test the pointer");
}

#[test]
fn option_ok_or_else_fn_item_calls_the_named_function() {
    let mut graph = FunctionGraph::new("test_ok_or_else_fn_item");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let f = graph.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("ok_or_else", Some("Option".into())),
                args: crate::model::call_args(vec![recv, f]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let mut site = option_ok_or_else_site(result);
    site.call_once_owner = String::new();
    site.fn_item_segments = Some(
        ["pyre_interpreter", "builtins", "reservation_failed"]
            .map(str::to_string)
            .to_vec(),
    );
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "ok_or_else"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 0);
    let fn_calls = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("reservation_failed")
            )
        })
        .count();
    assert_eq!(fn_calls, 1);
}

fn some_ctor_site(result: Variable) -> super::DiscCombinatorSite {
    let mut site = disc_site(super::DiscCombinator::ResultMap, result);
    site.call_once_owner = String::new();
    site.fn_item_segments = Some(
        ["core", "option", "Option", "Some"]
            .map(str::to_string)
            .to_vec(),
    );
    site.fn_item_ctor = Some(some_ctor_shape());
    site.call_result_ty = ValueType::Ref(None);
    site.result_payload0_ty = ValueType::Ref(None);
    site
}

#[test]
fn result_map_of_a_constructor_builds_the_variant() {
    // `r.map(Some)`: the `Ok` arm builds `Some(x)` the way the
    // `Rvalue::Aggregate` for `Some(x)` does, with no call of the
    // constructor and no `call_once`.
    let mut graph = FunctionGraph::new("test_result_map_some");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let some = push_fn_const(&mut graph, a, &["core", "option", "Option", "Some"]);
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("map", Some("Result".into())),
                args: crate::model::call_args(vec![recv, some]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[some_ctor_site(result)],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "map"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 0);
    assert_eq!(some_ctor_ops(&graph), (1, 0));
    let payload_owner_writes = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(&op.kind, OpKind::FieldWrite { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref() == Some("core::option::Option<i64>::Some"))
        })
        .count();
    assert_eq!(payload_owner_writes, 1);
    // The function-item define is no longer read; the dead-op sweep
    // drops it like the constant it stands for.
    crate::model::prune_dead_phis(&mut graph);
    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                matches!(&op.kind, OpKind::Call { target, .. }
                if crate::model::fn_const_segments(target).is_some())
            })
    );
}

#[test]
fn result_and_then_of_a_constructor_builds_the_variant() {
    // `r.and_then(Ok)`: the `Ok` arm builds `Ok(x)` as the aggregate
    // does.
    let mut graph = FunctionGraph::new("test_result_and_then_ok");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let ok = push_fn_const(&mut graph, a, &["core", "result", "Result", "Ok"]);
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("and_then", Some("Result".into())),
                args: crate::model::call_args(vec![recv, ok]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let mut site = disc_site(super::DiscCombinator::ResultAndThen, result);
    site.call_once_owner = String::new();
    site.fn_item_segments = Some(
        ["core", "result", "Result", "Ok"]
            .map(str::to_string)
            .to_vec(),
    );
    site.fn_item_ctor = Some(ok_ctor_shape());
    site.call_result_ty = ValueType::Ref(None);
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "and_then"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 0);
    let built = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(&op.kind, OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, variant_tag: Some(0), .. },
                    ..
                } if name == "Ok")
        })
        .count();
    assert_eq!(built, 1);
}

/// `od_malloc_i64_entries` returns `*mut GcEntries<i64, *mut PyObject>`
/// and a generic `Entry` local must not publish a type-variable row.
/// The bare `Entry` leaf keeps its type-variable `key`.
#[test]
fn concrete_i64_entry_rows_publish_one_qualified_spelling() {
    let obj_ty = 0u64;
    let entry = 1u64;
    let gc_entries = 2u64;
    let i64_ty = interior_i64();
    let bool_ty = interior_bool();
    let u64_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let typevar = |index: u64| serde_json::json!({"TypeVar": {"Bound": [0, index]}});
    let adt = |id: u64, types: Vec<serde_json::Value>| {
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
    };
    let field = |name: &str, ty: serde_json::Value| serde_json::json!({"name": name, "ty": ty, "attr_info": null});
    let struct_decl = |def_id: u64, path: &[&str], fields: Vec<serde_json::Value>| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": interior_meta(path, true),
            "kind": {"Struct": fields}
        })
    };
    let obj_ptr = interior_raw_mut(adt(obj_ty, vec![]));
    let concrete_gc = interior_raw_mut(adt(gc_entries, vec![i64_ty.clone(), obj_ptr.clone()]));
    let typevar_entry = interior_raw_mut(adt(entry, vec![typevar(0), typevar(1)]));
    let malloc = interior_opaque(
        10,
        &["pyre_object", "rordereddict", "od_malloc_i64_entries"],
        vec![interior_usize()],
        concrete_gc.clone(),
    );
    let generic = interior_caller(
        "generic_entry",
        0,
        vec![],
        bool_ty.clone(),
        vec![
            interior_local(0, None, &bool_ty),
            interior_local(1, Some("e"), &typevar_entry),
        ],
        vec![interior_bb(vec![], interior_return())],
    );
    let llbc = llbc_with_types(
        "pyre_object",
        vec![
            struct_decl(obj_ty, &["pyre_object", "pyobject", "PyObject"], vec![]),
            struct_decl(
                entry,
                &["pyre_object", "rordereddict_entries", "Entry"],
                vec![
                    field("key", typevar(0)),
                    field("f_valid", bool_ty.clone()),
                    field("value", typevar(1)),
                    field("f_hash", u64_ty),
                ],
            ),
            struct_decl(
                gc_entries,
                &["pyre_object", "rordereddict_entries", "GcEntries"],
                vec![
                    field("length", interior_usize()),
                    field(
                        "items",
                        serde_json::json!({
                            "Array": [
                                adt(entry, vec![i64_ty.clone(), obj_ptr.clone()]),
                                {"Value": {"Scalar": {"Unsigned": ["Usize", "0"]}}},
                                null
                            ]
                        }),
                    ),
                ],
            ),
        ],
        vec![malloc, generic],
    );
    use crate::model::ValueType;
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let gc_ptr = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [9100, concrete_gc.clone()]
    }))
    .expect("GcEntries ptr TyRef");
    assert!(
        !super::ty_is_raw_address(
            &gc_ptr,
            &llbc,
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        "*mut GcEntries is a GC ref"
    );
    assert!(
        matches!(
            super::tyref_to_value_type_with(
                &gc_ptr,
                &llbc,
                &std::collections::HashSet::new(),
                &gc,
                &std::collections::HashSet::<String>::new()
            ),
            ValueType::Ref(_)
        ),
        "*mut GcEntries lowers as Ref"
    );
    let state = super::CrateLoweringState::new(
        &llbc,
        &std::collections::HashSet::new(),
        std::collections::HashMap::new(),
        Default::default(),
        &gc,
        &std::collections::HashSet::new(),
        None,
        None,
    );
    let program = state.finish(Vec::new());
    let qualified = "pyre_object::rordereddict_entries::Entry<i64,*mut PyObject>";
    assert_eq!(
        program.struct_fields.fields.get(qualified),
        Some(&vec![
            crate::front::semantic::FieldRow::named("key", "i64"),
            crate::front::semantic::FieldRow::named("f_valid", "bool"),
            crate::front::semantic::FieldRow::named("value", "*mut PyObject"),
            crate::front::semantic::FieldRow::named("f_hash", "u64"),
        ])
    );
    assert!(program.known_struct_names.contains(qualified));
    let bare_key = program
        .struct_fields
        .fields
        .get("Entry")
        .and_then(|rows| rows.iter().find(|row| row.name == "key"))
        .map(|row| row.ty.as_str());
    assert_eq!(bare_key, Some("??TypeVar#0"));
    let published: Vec<_> = program
        .struct_fields
        .fields
        .keys()
        .filter(|key| {
            key.split('<')
                .next()
                .and_then(|base| base.rsplit("::").next())
                == Some("Entry")
                && key.contains("i64")
                && key.contains("PyObject")
        })
        .cloned()
        .collect();
    assert_eq!(published, vec![qualified.to_string()]);
    assert!(
        program
            .struct_fields
            .fields
            .keys()
            .all(|key| !key.contains("??"))
    );
}

fn interior_bool() -> serde_json::Value {
    serde_json::json!({"Scalar": "Bool"})
}

/// A merge phi inherits `concretetype` only when both predecessors agree
/// (`unionof` of equal lowleveltypes).
#[test]
fn merge_phi_concretetype_requires_predecessor_agreement() {
    let mut graph = FunctionGraph::new("merge_phi_ct_disagree");
    let v_ccharp = Variable::new();
    super::stamp_as_ccharp(&v_ccharp);
    let v_plain = Variable::new();
    let prev = FrameState {
        entries: vec![Some(v_ccharp.clone())],
        ..Default::default()
    };
    let ex = FrameState {
        entries: vec![Some(v_plain.clone())],
        ..Default::default()
    };
    let merged = prev
        .union(&ex, &mut graph)
        .expect("disagreeing-Variable union must succeed");
    super::copy_agreed_concretetypes(&prev, &ex, &merged);
    let phi = merged.entries[0]
        .as_ref()
        .expect("disagreement mints a fresh phi");
    assert!(
        phi.id() != v_ccharp.id() && phi.id() != v_plain.id(),
        "fresh phi is distinct from both predecessors"
    );
    assert!(
        phi.concretetype().is_none(),
        "disagreeing concretetypes leave the merge phi unset; ct={:?}",
        phi.concretetype()
    );

    let mut graph = FunctionGraph::new("merge_phi_ct_agree");
    let v_a = Variable::new();
    super::stamp_as_ccharp(&v_a);
    let v_b = Variable::new();
    super::stamp_as_ccharp(&v_b);
    let prev = FrameState {
        entries: vec![Some(v_a.clone())],
        ..Default::default()
    };
    let ex = FrameState {
        entries: vec![Some(v_b.clone())],
        ..Default::default()
    };
    let merged = prev
        .union(&ex, &mut graph)
        .expect("disagreeing-Variable union must succeed");
    super::copy_agreed_concretetypes(&prev, &ex, &merged);
    let phi = merged.entries[0]
        .as_ref()
        .expect("disagreement mints a fresh phi");
    assert!(
        phi.id() != v_a.id() && phi.id() != v_b.id(),
        "fresh phi is distinct from both predecessors"
    );
    assert!(
        phi.concretetype()
            .as_ref()
            .is_some_and(super::lltype_is_ccharp),
        "agreeing CCHARP stamps the merge phi; ct={:?}",
        phi.concretetype()
    );
}
