use crate::model::{CallTarget, LinkArg, OpKind};

use majit_charon_reader::{Llbc, ullbc::TyRef};

use crate::front::mir::tests::{
    call_leafs, fixture_item_meta, graph_ops, ident_path, llbc_with_trait_impls,
    std_extern_call_fixture_with_types, std_extern_call_fixture_with_types_dest, window_call_leafs,
    window_empty_generics, window_graph_ops, window_llbc, window_local, window_meta,
    window_opaque_fun, window_place, window_shared_slice, window_span,
    window_unwind_and_return_blocks, window_usize,
};

/// `k = w_bytes_data(key); entries.get(k)` keeps `k` as one `Ptr(STR)`
/// word (`space.bytes_w` / `BytesDictStrategy.unwrap`). The host
/// `RDict::get` (`rtype_method_get` → `ll_dict_get`) sees two args.
#[test]
fn w_bytes_data_local_stays_str_view_for_rdict_get() {
    use crate::model::{CallTarget, OpKind};

    let rdict_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let bytes_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let rdict_ref = serde_json::json!({"Ref": ["Erased", rdict_adt, "Shared"]});
    let bytes_ref = serde_json::json!({"Ref": ["Erased", bytes_adt, "Shared"]});
    let shared = window_shared_slice();
    let n = window_usize();
    let rdict = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "RDict"], true),
        "kind": {"Struct": []}
    });
    let w_bytes = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["pyre_object", "bytesobject", "W_BytesObject"], true),
        "kind": {"Struct": []}
    });
    let (unwind, ret) = window_unwind_and_return_blocks();
    let w_bytes_data = window_opaque_fun(
        1,
        &["pyre_object", "bytesobject", "w_bytes_data"],
        vec![bytes_ref.clone()],
        shared.clone(),
    );
    let get = window_opaque_fun(
        2,
        &["fixture", "RDict", "get"],
        vec![rdict_ref.clone(), shared.clone()],
        n.clone(),
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "w_bytes_data_rdict_get"], true),
        "signature": {"is_unsafe": false, "inputs": [rdict_ref, bytes_ref], "output": n},
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &n),
                window_local(1, &rdict_ref),
                window_local(2, &bytes_ref),
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
                        "args": [{"Copy": window_place(serde_json::json!({"Local": 2}), &bytes_ref)}],
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
                            {"Copy": window_place(serde_json::json!({"Local": 1}), &rdict_ref)},
                            {"Copy": window_place(serde_json::json!({"Local": 3}), &shared)}
                        ],
                        "dest": window_place(serde_json::json!({"Local": 0}), &n)
                    }, "target": 3, "on_unwind": 1}}}
                },
                ret
            ]
        }}
    });
    let llbc = window_llbc(vec![rdict, w_bytes], vec![caller, w_bytes_data, get]);
    let graph = crate::test_support::lower_function(&llbc, "w_bytes_data_rdict_get")
        .expect("lower STR-view RDict::get");
    let leafs = window_call_leafs(&graph);
    assert!(
        !leafs.iter().any(|leaf| {
            leaf == "direct_ptradd"
                || leaf == "__strlen"
                || leaf == crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING
        }),
        "STR-view key is not converted to a window; leafs={leafs:?} graph={}",
        graph.dump()
    );
    let get_args: Vec<usize> = window_graph_ops(&graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(String::as_str) == Some("get") => Some(args.len()),
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                args,
                ..
            } if name == "get" => Some(args.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        get_args.as_slice(),
        &[2],
        "RDict::get is receiver + STR key; graph={}",
        graph.dump()
    );
    assert!(
        !super::is_rdict_borrowed_key_method("fixture::NotRDictish::get"),
        "fixture::NotRDictish::get is not an RDict method"
    );
}

#[test]
fn rust_vec_std_leaf_accepts_both_callee_spellings() {
    let path = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    for p in [
        path(&["alloc", "vec", "<Impl>", "new"]),
        path(&["alloc", "vec", "Vec", "new"]),
        path(&["vec", "Vec", "new"]),
    ] {
        assert_eq!(super::rust_vec_std_leaf(&p), Some("new"), "{p:?}");
    }
    assert_eq!(
        super::rust_vec_std_leaf(&path(&["vec", "Vec", "try_reserve_exact"])),
        Some("try_reserve_exact")
    );
    for p in [
        path(&["alloc", "boxed", "<Impl>", "new"]),
        path(&["vec", "VecDeque", "new"]),
        path(&["new"]),
    ] {
        assert_eq!(super::rust_vec_std_leaf(&p), None, "{p:?}");
    }
    for (p, leaf) in [
        (path(&["alloc", "vec", "<Impl>", "deref_mut"]), "deref_mut"),
        (path(&["alloc", "vec", "Vec", "deref_mut"]), "deref_mut"),
        (path(&["vec", "Vec", "deref_mut"]), "deref_mut"),
        (
            path(&["alloc", "vec", "<Impl>", "as_mut_slice"]),
            "as_mut_slice",
        ),
        (path(&["vec", "Vec", "as_slice"]), "as_slice"),
    ] {
        assert_eq!(super::rust_vec_slice_view_leaf(&p), Some(leaf), "{p:?}");
    }
    for p in [
        path(&["alloc", "vec", "<Impl>", "extend_from_slice"]),
        path(&["alloc", "vec", "Vec", "extend_from_slice"]),
        path(&["vec", "Vec", "extend_from_slice"]),
    ] {
        assert_eq!(
            super::rust_vec_std_leaf(&p),
            Some("extend_from_slice"),
            "{p:?}"
        );
        assert!(super::path_is_vec_extend_from_slice(&p.join("::")), "{p:?}");
    }
    assert!(super::path_is_vec_slice_view("vec::Vec::deref_mut"));
    assert!(!super::path_is_vec_slice_view("vec::Vec::push"));
    assert!(super::path_is_vec_extend_from_slice(
        "alloc::vec::<Impl>::extend_from_slice"
    ));
    assert!(!super::path_is_vec_extend_from_slice("vec::Vec::extend"));
    assert!(super::path_is_build_refs_pair_abi(
        "pyre_interpreter::runtime_ops::build_list_from_refs"
    ));
    assert!(super::path_is_build_refs_pair_abi(
        "pyre_interpreter::build_tuple_from_refs"
    ));
    assert!(!super::path_is_build_refs_pair_abi(
        "pyre_interpreter::runtime_ops::build_map_from_refs"
    ));
    assert_eq!(
        super::rust_vec_std_leaf(&path(&[
            "vec",
            "Vec",
            "<impl core::ops::deref::Deref<alloc::vec::Vec<v0,v1>>>",
            "deref",
        ])),
        Some("deref")
    );
}

#[test]
fn rust_vec_slice_view_accepts_inherent_and_trait_impl_spellings() {
    let cases = [
        ("alloc::vec::<Impl>::deref", super::RustVecSliceView::Deref),
        (
            "alloc::vec::<Impl>::deref_mut",
            super::RustVecSliceView::DerefMut,
        ),
        (
            "alloc::vec::<Impl>::as_slice",
            super::RustVecSliceView::AsSlice,
        ),
        (
            "alloc::vec::<Impl>::as_mut_slice",
            super::RustVecSliceView::AsMutSlice,
        ),
        ("vec::Vec::deref", super::RustVecSliceView::Deref),
        ("vec::Vec::as_slice", super::RustVecSliceView::AsSlice),
        (
            "vec::Vec::<impl core::ops::deref::Deref<alloc::vec::Vec<v0,v1>>>::deref",
            super::RustVecSliceView::Deref,
        ),
        (
            "vec::Vec::<impl core::ops::deref::DerefMut<alloc::vec::Vec<v0,v1>>>::deref_mut",
            super::RustVecSliceView::DerefMut,
        ),
        (
            "vec::Vec::<impl core::ops::deref::Deref<alloc::vec::Vec<v0,v1>>>::as_slice",
            super::RustVecSliceView::AsSlice,
        ),
        (
            "alloc::vec::Vec::<impl core::ops::deref::Deref<alloc::vec::Vec<v0,v1>>>::as_mut_slice",
            super::RustVecSliceView::AsMutSlice,
        ),
    ];
    for (path, view) in cases {
        assert_eq!(super::rust_vec_slice_view(path), Some(view), "{path}");
    }
    assert_eq!(
        super::rust_vec_slice_view("core::slice::<Impl>::as_slice"),
        None
    );
    assert_eq!(
        super::rust_vec_slice_view(
            "vec::Vec::<impl core::ops::index::Index<core::ops::range::RangeFull>>::index"
        ),
        None
    );
    let index = [
        "vec",
        "Vec",
        "<impl core::ops::index::Index<core::ops::range::RangeFull>>",
        "index",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    assert_eq!(super::rust_vec_std_leaf(&index), Some("index"));
}

#[test]
fn signed_byte_vec_as_ptr_converts_the_header_and_wraps_ccharp() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let i8_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I8"}}});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![i8_ty.clone()])}
    });
    let i8_ptr = serde_json::json!({"RawPtr": [i8_ty.clone(), "Const"]});
    let i8_mut_ptr = serde_json::json!({"RawPtr": [i8_ty, "Mut"]});
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
    let expected = super::rust_vec_as_ptr_helper_segments();
    for (caller, leaf, dest_ty) in [
        ("vec_i8_as_ptr", "as_ptr", i8_ptr),
        ("vec_i8_as_mut_ptr", "as_mut_ptr", i8_mut_ptr),
    ] {
        let llbc = std_extern_call_fixture_with_types(
            caller,
            &["alloc", "vec", "<Impl>", leaf],
            &[vec_ty.clone()],
            dest_ty,
            vec![vec_decl.clone()],
        );
        let graph =
            crate::test_support::lower_function(&llbc, caller).expect("lower Vec<i8> as_ptr");
        let ops = graph_ops(&graph);
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if *segments == expected
            )),
            "Vec::{leaf} on Vec<i8> must call {expected:?}; ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some(leaf)
            )),
            "Vec::{leaf} on Vec<i8> must not stay residual; ops={ops:?}"
        );
        let as_ptr_arg = ops.iter().find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if *segments == expected => args.first().and_then(LinkArg::as_variable).cloned(),
            _ => None,
        });
        let as_ptr_arg = as_ptr_arg.expect("ll_vec_as_ptr argument");
        let r_uint_src = ops.iter().find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if op.result.as_ref() == Some(&as_ptr_arg)
                && segments.last().map(String::as_str) == Some("r_uint") =>
            {
                args.first().and_then(LinkArg::as_variable).cloned()
            }
            _ => None,
        });
        let r_uint_src = r_uint_src.expect("ll_vec_as_ptr argument is r_uint of the header");
        assert!(
            ops.iter().any(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if op.result.as_ref() == Some(&r_uint_src)
                    && segments.last().map(String::as_str) == Some("cast_ptr_to_int") =>
                {
                    true
                }
                _ => false,
            }),
            "Vec::{leaf} header is cast_ptr_to_int then r_uint; ops={ops:?}"
        );
        let items = ops.iter().find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if *segments == expected => op.result.clone(),
            _ => None,
        });
        let items = items.expect("ll_vec_as_ptr result");
        assert!(
            ops.iter().any(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("cast_int_to_ptr")
                    && args.len() == 2
                    && args.get(1).and_then(LinkArg::as_variable) == Some(&items) =>
                {
                    true
                }
                _ => false,
            }),
            "Vec::{leaf} items word is cast_int_to_ptr(CCHARP); ops={ops:?}"
        );
    }
}

#[test]
fn byte_item_vec_deref_splits_the_header_with_word_helpers() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![u8_ty.clone()])}
    });
    let u8_slice = serde_json::json!({
        "Ref": ["'_", {"Slice": [u8_ty, null]}, "Shared"]
    });
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
    for (caller, callee) in [
        ("vec_u8_deref", &["vec", "Vec", "deref"] as &[&str]),
        ("vec_u8_as_slice", &["alloc", "vec", "<Impl>", "as_slice"]),
    ] {
        let llbc = std_extern_call_fixture_with_types_dest(
            caller,
            callee,
            &[vec_ty.clone()],
            u8_slice.clone(),
            vec![vec_decl.clone()],
            true,
        );
        let graph = super::lower_function(&llbc, caller).expect("lower Vec<u8> slice view");
        let ops = graph_ops(&graph);
        let leafs = call_leafs(&ops);
        assert!(
            leafs.iter().any(|leaf| leaf == "ll_vec_items_i"),
            "{caller} must call ll_vec_items_i; leafs={leafs:?} ops={ops:?}"
        );
        assert!(
            leafs.iter().any(|leaf| leaf == "ll_vec_length_i"),
            "{caller} must call ll_vec_length_i; leafs={leafs:?} ops={ops:?}"
        );
        assert!(
            !leafs
                .iter()
                .any(|leaf| leaf == "deref" || leaf == "as_slice"),
            "{caller} must not stay residual; leafs={leafs:?}"
        );
    }
}

#[test]
fn payload_enum_vec_deref_stays_residual() {
    let generics = |types: Vec<serde_json::Value>| {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    };
    let slot_ty = serde_json::json!({
        "Adt": {"id": 1, "generics": generics(vec![])}
    });
    let vec_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics(vec![slot_ty.clone()])}
    });
    let slot_slice = serde_json::json!({
        "Ref": ["'_", {"Slice": [slot_ty, null]}, "Shared"]
    });
    let item_meta = |path: &[&str]| {
        let name: Vec<serde_json::Value> = path
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        serde_json::json!({
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
        })
    };
    let vec_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["alloc", "vec", "Vec"]),
        "kind": {"Struct": []}
    });
    let slot_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["fixture", "SlotValue"]),
        "kind": {"Enum": [
            {"name": "Int", "fields": [{
                "name": "v",
                "ty": {"Scalar": {"Integer": {"Signed": "I64"}}},
                "attr_info": null
            }]}
        ]}
    });
    let llbc = std_extern_call_fixture_with_types(
        "vec_slot_deref",
        &["vec", "Vec", "deref"],
        &[vec_ty],
        slot_slice,
        vec![vec_decl, slot_decl],
    );
    let graph =
        super::lower_function(&llbc, "vec_slot_deref").expect("payload-enum Vec deref must lower");
    let leafs = call_leafs(&graph_ops(&graph));
    assert!(
        leafs.iter().any(|leaf| leaf == "deref"),
        "Vec<SlotValue>::deref stays residual; leafs={leafs:?}"
    );
    assert!(
        !leafs.iter().any(|leaf| leaf.starts_with("ll_vec_items")),
        "payload-enum Vec deref is not a pair view; leafs={leafs:?}"
    );
}

/// `code.varnames.len()` and `code.localspluskinds[i]` share one field
/// read. Each use gets its own fat-pointer word; the shared read is
/// not a `Vec` component and is not rewritten in place.
#[test]
fn fat_box_len_and_index_keep_the_field_read() {
    use crate::flowspace::model::{ConstValue, Constant};
    use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType, VecFieldPart};

    let mut graph = FunctionGraph::new("fat_box_field");
    let code = graph
        .push_op_var(
            graph.startblock,
            OpKind::Input {
                name: "code".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .expect("code");
    let varnames = graph
        .push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: code.clone(),
                field: FieldDescriptor::new("varnames", Some("bytecode::CodeObject".into())),
                ty: ValueType::Ref(None),
                pure: true,
            },
            true,
        )
        .expect("varnames");
    let marked = vec![varnames.clone()];
    let bb = graph.startblock;
    let len = super::retarget_fat_operand(&mut graph, &marked, bb, &varnames, VecFieldPart::FatLen)
        .expect("same-block len");
    let data =
        super::retarget_fat_operand(&mut graph, &marked, bb, &varnames, VecFieldPart::FatData)
            .expect("same-block data");
    assert_ne!(len, varnames);
    assert_ne!(data, varnames);
    assert_ne!(len, data);
    assert!(
        super::retarget_fat_operand(&mut graph, &marked, bb, &code, VecFieldPart::FatLen).is_none()
    );
    assert!(
        super::retarget_fat_operand(&mut graph, &[], bb, &varnames, VecFieldPart::FatLen).is_none()
    );
    let mut saw_original = false;
    let mut saw_len = false;
    let mut saw_data = false;
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        let OpKind::FieldRead {
            base, field, ty, ..
        } = &op.kind
        else {
            continue;
        };
        if op.result.as_ref() == Some(&varnames) {
            saw_original = field.vec_part.is_none()
                && !field.inline_vec
                && field.name == "varnames"
                && field.owner_root.as_deref() == Some("bytecode::CodeObject")
                && base == &code;
        } else if op.result.as_ref() == Some(&len) {
            saw_len = field.vec_part == Some(VecFieldPart::FatLen)
                && field.name == "varnames"
                && field.owner_root.as_deref() != Some("alloc::vec::Vec")
                && base == &code
                && matches!(ty, ValueType::Int);
        } else if op.result.as_ref() == Some(&data) {
            saw_data = field.vec_part == Some(VecFieldPart::FatData)
                && base == &code
                && matches!(ty, ValueType::Ref(None));
        }
    }
    assert!(saw_original, "shared field read stays whole");
    assert!(saw_len, "len reads the metadata word");
    assert!(saw_data, "index reads the data word");

    let mut raw_graph = FunctionGraph::new("raw_fat_box");
    let raw_base = raw_graph
        .push_op_var(
            raw_graph.startblock,
            OpKind::Input {
                name: "raw".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            true,
        )
        .expect("raw");
    let raw_box = raw_graph
        .push_op_var(
            raw_graph.startblock,
            OpKind::FieldRead {
                base: raw_base,
                field: FieldDescriptor::new("items", Some("RawOwner".into()))
                    .with_owner_declared_gc(Some(false)),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        )
        .expect("items");
    let start = raw_graph.startblock;
    let raw_data = super::retarget_fat_operand(
        &mut raw_graph,
        &[raw_box.clone()],
        start,
        &raw_box,
        VecFieldPart::FatData,
    )
    .expect("raw data word");
    let raw_ty = raw_graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| {
            if op.result.as_ref() != Some(&raw_data) {
                return None;
            }
            match &op.kind {
                OpKind::FieldRead { ty, .. } => Some(ty.clone()),
                _ => None,
            }
        })
    });
    assert!(
        matches!(raw_ty, Some(ValueType::Int)),
        "a raw owner's fat data word is an address"
    );

    let (next, inputs) = graph.create_block_with_arg_vars(1);
    graph.set_goto(graph.startblock, next, vec![varnames.clone()]);
    let hopped =
        super::retarget_fat_operand(&mut graph, &marked, next, &inputs[0], VecFieldPart::FatLen)
            .expect("one phi hop");
    let block = graph.block(next);
    let last = block.operations.last().expect("phi retarget");
    assert_eq!(last.result.as_ref(), Some(&hopped));
    match &last.kind {
        OpKind::FieldRead {
            base, field, ty, ..
        } => {
            assert_eq!(base, &code);
            assert_eq!(field.vec_part, Some(VecFieldPart::FatLen));
            assert_eq!(field.name, "varnames");
            assert!(matches!(ty, ValueType::Int));
        }
        other => panic!("phi copy must retarget, got {other:?}"),
    }

    let mut other = FunctionGraph::new("fat_const_phi");
    let (merge, merge_inputs) = other.create_block_with_arg_vars(1);
    other.set_goto_mixed(
        other.startblock,
        merge,
        vec![LinkArg::Const(Constant::new(ConstValue::Int(1)))],
    );
    let dummy = other.alloc_value_var();
    assert!(
        super::retarget_fat_operand(
            &mut other,
            &[dummy],
            merge,
            &merge_inputs[0],
            VecFieldPart::FatLen,
        )
        .is_none(),
        "a constant phi is not a fat field"
    );
}

/// The Sized gate: a shared reference to a fat pointee (`&[T]`, spelled
/// `{"Ref":[_, {"Slice": elem}, "Shared"]}`) has no ADT def-id, so
/// `type_node_shared_ref_pointee` + `adt_node_def_id` must NOT accept it;
/// while a shared ref to a nominal `{"Adt": {"id": N}}` pointee
/// DOES. Uses the existing `Llbc::from_slice` minimal-fixture pattern
/// (`llbc_with_trait_impls` in this file), no real corpus.
/// A one-word-item `&[T]` / `&mut [T]` is a pair slice; a byte slice, a
/// borrowed array, a double borrow and a bare slice are not.
#[test]
fn pair_slice_item_kind_is_one_borrow_over_a_one_word_slice() {
    use majit_ir::rvec::VecItemKind;
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": fixture_item_meta(ident_path(&["PyObject"])),
                "kind": {"Struct": []}
            }],
            "fun_decls": [], "global_decls": [],
            "trait_decls": [], "trait_impls": [],
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let usize_ty = serde_json::json!({ "Scalar": { "Integer": { "Unsigned": "Usize" } } });
    let objptr = serde_json::json!({
        "RawPtr": [{"Adt": {"id": 0, "generics": {"types": []}}}, "Mut"]
    });
    let kind =
        |node: serde_json::Value| super::tyref_pair_slice_item_kind(&TyRef::Other(node), &llbc);
    assert_eq!(
        kind(serde_json::json!({ "Ref": ["'_", { "Slice": [usize_ty.clone(), null] }, "Shared"] })),
        Some(VecItemKind::Int)
    );
    assert_eq!(
        kind(
            serde_json::json!({ "Ref": ["'_", { "Slice": [{ "Scalar": { "Float": "F64" } }, null] }, "Mut"] })
        ),
        Some(VecItemKind::Float)
    );
    assert_eq!(
        kind(
            serde_json::json!({ "Ref": ["'_", { "Slice": [{ "Scalar": { "Integer": { "Unsigned": "U8" } } }, null] }, "Shared"] })
        ),
        None
    );
    assert!(super::tyref_is_fat_slice(
        &TyRef::Other(serde_json::json!({
            "Ref": ["'_", { "Slice": [{ "Scalar": { "Integer": { "Unsigned": "U8" } } }, null] }, "Shared"]
        })),
        &llbc,
    ));
    assert!(super::tyref_is_fat_slice(
        &TyRef::Other(serde_json::json!({
            "Ref": ["'_", { "Slice": [usize_ty.clone(), null] }, "Shared"]
        })),
        &llbc,
    ));
    assert!(!super::tyref_is_fat_slice(
        &TyRef::Other(serde_json::json!({
            "Ref": ["'_", { "Array": [usize_ty.clone(), { "Value": { "Scalar": { "Unsigned": ["Usize", "4"] } } }] }, "Shared"]
        })),
        &llbc,
    ));
    assert_eq!(
        kind(
            serde_json::json!({ "Ref": ["'_", { "Array": [usize_ty.clone(), { "Value": { "Scalar": { "Unsigned": ["Usize", "4"] } } }] }, "Shared"] })
        ),
        None
    );
    assert_eq!(
        kind(
            serde_json::json!({ "Ref": ["'_", { "Ref": ["'_", { "Slice": [usize_ty.clone(), null] }, "Shared"] }, "Shared"] })
        ),
        None
    );
    assert_eq!(kind(serde_json::json!({ "Slice": [usize_ty, null] })), None);
    assert_eq!(super::spelling_pair_field_kind("[PyObjectRef]"), None);
    assert_eq!(super::spelling_pair_field_kind("[*mut PyObject]"), None);
    assert_eq!(
        super::spelling_pair_field_kind("[*const pyre_object::PyObject]"),
        None
    );
    assert_eq!(
        super::spelling_pair_field_kind("[*mut u8]"),
        Some(VecItemKind::Int),
        "*mut u8 is rffi.CCHARP, a word"
    );
    assert_eq!(
        super::spelling_pair_field_kind("[GCREF]"),
        Some(VecItemKind::Ref)
    );
    assert!(super::gcref_item_spelling("GCREF"));
    assert!(super::gcref_item_spelling("majit_gc::GCREF"));
    assert!(!super::gcref_item_spelling("*mut u8"));
    assert_eq!(
        super::spelling_pair_field_kind("[usize]"),
        Some(VecItemKind::Int)
    );
    assert_eq!(
        kind(serde_json::json!({
            "Ref": ["'_", { "Slice": [objptr.clone(), null] }, "Mut"]
        })),
        Some(VecItemKind::Ref),
        "object-pointer &mut [T] is the {{ptr, len}} view"
    );
    assert_eq!(
        kind(serde_json::json!({
            "Ref": ["'_", { "Slice": [objptr, null] }, "Shared"]
        })),
        None,
        "object-pointer &[T] stays the length-prefixed object array"
    );
    // A slice of a Copy associated type is the same `{ptr, len}` fat
    // pointer. Unique-impl resolution spells a raw pointer; an
    // unresolved projection is still one word.
    let ptr = serde_json::json!({"RawPtr": [usize_ty.clone(), "Mut"]});
    let binding = serde_json::json!({
        "kind": {"TraitType": [0, 0]},
        "skip_binder": {"value": ptr}
    });
    let row = serde_json::json!({
        "impl_trait": {"id": 0, "generics": {"types": []}},
        "types": [binding],
        "implied_trait_refs": []
    });
    let unique = llbc_with_trait_impls(serde_json::json!([row.clone()]));
    let projection = serde_json::json!({
        "TraitType": [
            {"kind": {"Clause": {"Bound": [0, 0]}}, "trait_decl_ref": {"skip_binder": {"id": 0}}},
            0,
            {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        ]
    });
    let slice_of = |item: serde_json::Value| serde_json::json!({ "Ref": ["'_", { "Slice": [item, null] }, "Shared"] });
    assert_eq!(
        super::tyref_pair_slice_item_kind(&TyRef::Other(slice_of(projection.clone())), &unique),
        Some(VecItemKind::Int),
        "unique-impl *mut usize is a word (`rffi.CCHARP` / Address)"
    );
    let ambiguous = llbc_with_trait_impls(serde_json::json!([row.clone(), row]));
    assert_eq!(
        super::tyref_pair_slice_item_kind(&TyRef::Other(slice_of(projection)), &ambiguous),
        Some(VecItemKind::Ref)
    );
}
