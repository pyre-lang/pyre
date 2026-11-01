//! Atomic ordering must survive the Rust-input boundary. RPython's
//! flowcontext.FlowContext.record_block propagates FlowingError rather than
//! publishing a graph with an unsupported operation erased.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, lower_function};
use majit_translate::model::{CallTarget, FunctionGraph, LinkArg, OpKind};
use serde_json::{Value, json};

fn fixture(ordering: Option<&str>) -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let adt = |id| json!({"Adt": {"id": id, "generics": generics}});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    // `AtomicU64` is the alias `Atomic<u64>`.
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let receiver = json!({"Ref": ["Erased", atomic, "Shared"]});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let statements = ordering
        .map(|name| {
            let index = variants.iter().position(|v| *v == name).unwrap();
            json!([{"span": span, "kind": {"Assign": [place(2, &adt(1)),
            {"Aggregate": [{"Adt": [1, index, null, generics]}, []]}]}}])
        })
        .unwrap_or_else(|| json!([]));
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_atomic"]),
                 "signature": {"is_unsafe": false,
                    "inputs": if ordering.is_some() {vec![receiver.clone()]} else {vec![receiver.clone(), adt(1)]},
                    "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": if ordering.is_some() {1} else {2}, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "slot", "span": span, "ty": receiver},
                        {"index": 2, "name": "ordering", "span": span, "ty": adt(1)}
                    ]},
                    "body": [
                        {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Copy": place(1, &receiver)}, {"Copy": place(2, &adt(1))}],
                                "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [receiver, adt(1)], "output": word}, "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic fixture parses")
}

#[test]
fn atomic_load_relaxed_keeps_existing_lowering() {
    lower_function(&fixture(Some("Relaxed")), "read_atomic").expect("known Relaxed load");
}

#[test]
fn atomic_load_must_not_erase_ordering() {
    for ordering in [
        Some("Acquire"),
        Some("SeqCst"),
        Some("Release"),
        Some("AcqRel"),
        None,
    ] {
        let result = lower_function(&fixture(ordering), "read_atomic");
        assert!(
            matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering")),
            "{ordering:?}: must reject unsupported ordering, got {result:?}"
        );
    }
}

#[test]
#[ignore = "requires freshly extracted pyre-object LLBC; run extraction --check first"]
fn atomic_load_real_acquire_reader_of_quasi_field_lowers() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("load fresh pyre-object corpus");
    // Quasi-immutable `version_tag: AtomicU64`. The Acquire load lowers
    // to a FieldRead so `record_quasiimmut_field` still guards it.
    let graph = lower_function(&llbc, "w_type_get_version_tag")
        .expect("Acquire load of quasi version_tag lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "lowered w_type_get_version_tag has no version_tag field read"
    );
    // get_instantiate publishes the cached W_TypeObject to allocators. Its
    // readers and late writers are all GIL-serialized, so the source spells a
    // Relaxed load — the one ordering the frontend lowers to a plain read.
    let result = lower_function(&llbc, "get_instantiate");
    assert!(
        result.is_ok(),
        "get_instantiate: the Relaxed reader must keep lowering: {result:?}"
    );
}

#[test]
#[ignore = "loads target/llbc/pyre-interpreter.ullbc"]
fn acquire_quasi_w_globals_lowers_across_the_block_copy() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    ))
    .expect("fresh pyre-interpreter corpus");
    let graph = lower_function(&llbc, "w_code_get_w_globals")
        .expect("Acquire load of quasi w_globals lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. } if field.name == "w_globals"
            )
        })
    });
    assert!(
        saw,
        "lowered w_code_get_w_globals has no w_globals field read"
    );
}

enum Receiver {
    Borrow,
    Copied,
    ReassignedBase,
    ReassignedParam,
}

fn quasi_atomic_u64_field_fixture(
    owner: &str,
    field: &str,
    entries: Option<&str>,
    receiver: Receiver,
) -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let owner_adt = json!({"Adt": {"id": 2, "generics": generics, "builtin": null}});
    let owner_ref = json!({"Ref": ["Erased", owner_adt, "Shared"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let reassigned_param = matches!(receiver, Receiver::ReassignedParam);
    let ordering_local = if reassigned_param { 3 } else { 2 };
    let recv_bind = if reassigned_param { 4 } else { 3 };
    let field_base = match receiver {
        Receiver::ReassignedBase => 4,
        _ => 1,
    };
    let field_place = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(field_base, &owner_ref), "Deref"]}, "ty": owner_adt},
            {"Field": [null, 0]}
        ]},
        "ty": atomic
    });
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let attr = json!({"attributes": [], "inline": null, "rename": null, "public": true});
    let global_decls = match entries {
        Some(joined) => json!([{
            "def_id": 0,
            "item_meta": meta(&["fixture", &format!("_immutable_fields_{owner}")]),
            "global_kind": "NamedConst",
            "value": {"Value": [1, [{"Str": joined}, {"Ref": [
                "Erased",
                {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "Shared"
            ]}]]}
        }]),
        None => json!([]),
    };
    let mut locals = vec![
        json!({"index": 0, "name": null, "span": span, "ty": word}),
        json!({"index": 1, "name": "owner", "span": span, "ty": owner_ref}),
    ];
    if reassigned_param {
        locals.push(json!({"index": 2, "name": "other", "span": span, "ty": owner_ref}));
    }
    locals.push(json!({"index": ordering_local, "name": "ordering", "span": span, "ty": adt_ord}));
    locals.push(json!({"index": recv_bind, "name": "recv", "span": span, "ty": recv_ty}));
    let mut statements = vec![
        json!({"span": span, "kind": {"Assign": [place(ordering_local, &adt_ord),
            {"Aggregate": [{"Adt": [1, 1, null, generics]}, []]}]}}),
    ];
    if matches!(receiver, Receiver::ReassignedBase) {
        locals.push(json!({"index": 4, "name": "_b", "span": span, "ty": owner_ref}));
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(4, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(4, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
        );
    }
    statements.push(
        json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": field_place, "kind": "Shared", "ptr_metadata": null}}]}}),
    );
    if reassigned_param {
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(1, &owner_ref),
            {"Use": [{"Copy": place(2, &owner_ref)}, "Yes"]}]}}),
        );
    }
    let recv_local = if matches!(receiver, Receiver::Copied) {
        locals.push(json!({"index": 4, "name": "alias", "span": span, "ty": recv_ty}));
        statements.push(json!({"span": span, "kind": {"Assign": [place(4, &recv_ty),
            {"Use": [{"Copy": place(3, &recv_ty)}, "Yes"]}]}}));
        4
    } else {
        recv_bind
    };
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}},
                {"def_id": 2, "item_meta": meta(&["fixture", owner]),
                 "kind": {"Struct": [{
                    "name": field, "ty": atomic, "attr_info": attr
                 }]}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_field"]),
                 "signature": {"is_unsafe": false,
                    "inputs": if reassigned_param {
                        vec![owner_ref.clone(), owner_ref.clone()]
                    } else {
                        vec![owner_ref.clone()]
                    },
                    "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": if reassigned_param { 2 } else { 1 }, "locals": locals},
                    "body": [
                        {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Copy": place(recv_local, &recv_ty)}, {"Copy": place(ordering_local, &adt_ord)}],
                                "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [recv_ty, adt_ord], "output": word}, "body": "Opaque"}
            ], "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field fixture parses")
}

#[test]
fn acquire_quasi_atomic_field_lowers_and_non_quasi_still_errors() {
    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::Borrow,
        ),
        "read_field",
    )
    .expect("Acquire load of quasi AtomicU64 field lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "lowered quasi AtomicU64 field has no version_tag field read"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture("Plain", "hot_counter", None, Receiver::Borrow),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load of a non-quasi AtomicU64 field must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::Copied,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load through a copied receiver must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedBase,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load through a reassigned borrow base must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load through a reassigned owner parameter must stay unsupported, got {result:?}"
    );
}

fn quasi_atomic_u64_field_fixture_with_owner_call() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": generics}});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let owner_adt = json!({"Adt": {"id": 2, "generics": generics, "builtin": null}});
    let owner_ref = json!({"Ref": ["Erased", owner_adt, "Shared"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let field_place = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
            {"Field": [null, 0]}
        ]},
        "ty": atomic
    });
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let attr = json!({"attributes": [], "inline": null, "rename": null, "public": true});
    let global_decls = json!([{
        "def_id": 0,
        "item_meta": meta(&["fixture", "_immutable_fields_Owner"]),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": "version_tag?"}, {"Ref": [
            "Erased",
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Shared"
        ]}]]}
    }]);
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}},
                {"def_id": 2, "item_meta": meta(&["fixture", "Owner"]),
                 "kind": {"Struct": [{
                    "name": "version_tag", "ty": atomic, "attr_info": attr
                 }]}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_field"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [owner_ref.clone()],
                    "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "owner", "span": span, "ty": owner_ref},
                        {"index": 2, "name": "ordering", "span": span, "ty": adt_ord},
                        {"index": 3, "name": "recv", "span": span, "ty": recv_ty},
                        {"index": 4, "name": "ignored", "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [
                            {"span": span, "kind": {"Assign": [place(2, &adt_ord),
                                {"Aggregate": [{"Adt": [1, 1, null, generics]}, []]}]}},
                            {"span": span, "kind": {"Assign": [place(3, &recv_ty),
                                {"Ref": {"place": field_place, "kind": "Shared", "ptr_metadata": null}}]}}
                        ], "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Copy": place(1, &owner_ref)}],
                                "dest": place(4, &unit)}, "target": 1, "on_unwind": 3
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Copy": place(3, &recv_ty)}, {"Copy": place(2, &adt_ord)}],
                                "dest": place(0, &word)}, "target": 2, "on_unwind": 3
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [recv_ty, adt_ord], "output": word}, "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(&["fixture", "other"]),
                 "signature": {"is_unsafe": false, "inputs": [owner_ref], "output": unit}, "body": "Opaque"}
            ], "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field owner-call fixture parses")
}

fn call_leaf_is_other(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("other")
        }
        CallTarget::Method { name, .. } => name == "other",
        _ => false,
    }
}

fn success_path_ops(graph: &FunctionGraph) -> Vec<&majit_translate::model::SpaceOperation> {
    let mut ops = Vec::new();
    let mut id = graph.startblock;
    for _ in 0..graph.blocks.len() {
        let block = graph.block(id);
        ops.extend(&block.operations);
        if id == graph.returnblock {
            break;
        }
        let Some(link) = block
            .exits
            .iter()
            .find(|link| link.target != graph.exceptblock)
        else {
            break;
        };
        if link.target == id {
            break;
        }
        id = link.target;
    }
    ops
}

#[test]
fn acquire_quasi_atomic_load_reads_at_the_load_site() {
    let graph = lower_function(
        &quasi_atomic_u64_field_fixture_with_owner_call(),
        "read_field",
    )
    .expect("Acquire load after an intervening owner call lowers");
    let return_var = graph
        .blocks
        .iter()
        .find_map(|block| {
            block.exits.iter().find_map(|link| {
                if link.target != graph.returnblock {
                    return None;
                }
                match link.args.as_slice() {
                    [LinkArg::Value(var)] => Some(var.clone()),
                    _ => None,
                }
            })
        })
        .expect("read_field returns a value");
    let ops = success_path_ops(&graph);
    let other_idx = ops.iter().position(|op| match &op.kind {
        OpKind::Call { target, .. } => call_leaf_is_other(target),
        _ => false,
    });
    let load_read_idx = ops.iter().position(|op| {
        op.result.as_ref() == Some(&return_var)
            && matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "version_tag"
            )
    });
    let Some(load_read_idx) = load_read_idx else {
        panic!("returned value is not a version_tag FieldRead: {ops:?}");
    };
    match other_idx {
        Some(other_idx) => assert!(
            load_read_idx > other_idx,
            "load-site version_tag FieldRead must follow the intervening call, got {ops:?}"
        ),
        None => {
            let load_block = graph
                .blocks
                .iter()
                .find(|block| {
                    block.operations.iter().any(|op| {
                        op.result.as_ref() == Some(&return_var)
                            && matches!(
                                &op.kind,
                                OpKind::FieldRead { field, .. } if field.name == "version_tag"
                            )
                    })
                })
                .expect("load-site FieldRead lives in some block");
            let borrow_is_elsewhere = graph.blocks.iter().any(|block| {
                block.id != load_block.id
                    && block.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::FieldRead { field, .. } if field.name == "version_tag"
                        ) && op.result.as_ref() != Some(&return_var)
                    })
            });
            assert!(
                borrow_is_elsewhere,
                "returned FieldRead must be the load-site read, distinct from the borrow's: {ops:?}"
            );
        }
    }
}
