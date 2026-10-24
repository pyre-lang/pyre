//! Atomic ordering must survive the Rust-input boundary. RPython's
//! flowcontext.FlowContext.record_block propagates FlowingError rather than
//! publishing a graph with an unsupported operation erased.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, lower_function};
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
                        {"statements": [], "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
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

fn static_atomic_bool_load_llbc() -> Llbc {
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
    let bool_ty = json!({"Scalar": "Bool"});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [bool_ty.clone()], "const_generics": [], "trait_refs": []}}});
    let receiver = json!({"Ref": ["Erased", atomic.clone(), "Shared"]});
    let ordering = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let global_place = json!({
        "kind": {"Global": {"generics": generics, "id": 1}},
        "ty": atomic
    });
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let ordering_enum = variants
        .iter()
        .enumerate()
        .map(|(i, name)| {
            json!({
                "name": name, "fields": [],
                "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
            })
        })
        .collect::<Vec<_>>();
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "AtomicBool"]),
                 "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": ordering_enum}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "dev_mode_flag"]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": bool_ty.clone()},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 0, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": bool_ty},
                        {"index": 1, "name": "slot", "span": span, "ty": receiver},
                        {"index": 2, "name": "ordering", "span": span, "ty": ordering}
                    ]},
                    "body": [
                        {"statements": [
                            {"span": span, "kind": {"Assign": [
                                place(1, &receiver),
                                {"Ref": {"place": global_place, "kind": "Shared", "ptr_metadata": "None"}}
                            ]}},
                            {"span": span, "kind": {"Assign": [
                                place(2, &ordering),
                                {"Aggregate": [{"Adt": [1, 0, null, generics]}, []]}
                            ]}}
                        ], "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [
                                    {"Copy": place(1, &receiver)},
                                    {"Copy": place(2, &ordering)}
                                ],
                                "dest": place(0, &bool_ty)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "AtomicBool", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [receiver, ordering],
                    "output": bool_ty}, "body": "Opaque"}
            ],
            "global_decls": [
                null,
                {
                    "def_id": 1,
                    "item_meta": meta(&["fixture", "SYS_DEV_MODE"]),
                    "global_kind": {"Static": {
                        "is_mut": false, "is_safe": true, "is_thread_local": false
                    }},
                    "ty": atomic
                }
            ],
            "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("static atomic load fixture")
}

#[test]
fn static_atomic_bool_load_reads_through_the_address() {
    use majit_translate::model::{CallTarget, OpKind};
    let graph = lower_function(&static_atomic_bool_load_llbc(), "dev_mode_flag")
        .expect("lower static AtomicBool load");
    let kinds: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| &op.kind)
        .collect();
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, OpKind::ConstRefAddr(_))),
        "static AtomicBool is read through its address: {graph:?}"
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, OpKind::RawLoad { .. })),
        "AtomicBool::load of a static is a load from that address: {graph:?}"
    );
    assert!(
        !kinds.iter().any(|kind| matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("SYS_DEV_MODE")
                || segments.last().map(String::as_str) == Some("load")
        )),
        "interior-mut static must not residualize as FunctionPath: {graph:?}"
    );
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
fn atomic_load_real_acquire_readers_preserve_the_diagnostic() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-object.ullbc"
    ))
    .expect("load fresh pyre-object corpus");
    // The version tag publishes class mutations across threads: its Acquire
    // load may not become a scalar field read just because its enclosing
    // caller can be traced. A missing descriptor is not permission to
    // recover the old ordering-erasing lowering to satisfy a layout test.
    let result = lower_function(&llbc, "w_type_get_version_tag");
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")),
        "w_type_get_version_tag: the real Acquire reader must not become a plain field value: {result:?}"
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
#[ignore = "loads build/llbc/pyre-interpreter.ullbc"]
fn acquire_quasi_w_globals_lowers_across_the_block_copy() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-interpreter.ullbc"
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
