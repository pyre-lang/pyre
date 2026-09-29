//! `Cell::get` / `Cell::set` are a load / store of the inner word, and an
//! interior-mut static is still read through its address.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, OpKind};
use serde_json::{Value, json};

fn span() -> Value {
    json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}})
}

fn meta(path: &[&str]) -> Value {
    json!({
        "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
        "span": span(), "source_text": null, "is_local": true,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
    })
}

fn generics() -> Value {
    json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []})
}

fn usize_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
}

fn cell_ty() -> Value {
    json!({"Adt": {"id": 0, "generics": {
        "regions": [], "types": [usize_ty()], "const_generics": [], "trait_refs": []
    }}})
}

fn cell_ref() -> Value {
    json!({"Ref": ["Erased", cell_ty(), "Shared"]})
}

fn place(id: u64, ty: &Value) -> Value {
    json!({"kind": {"Local": id}, "ty": ty})
}

fn op_kinds(graph: &majit_translate::model::FunctionGraph) -> Vec<&OpKind> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| &op.kind)
        .collect()
}

fn calls_leaf(graph: &majit_translate::model::FunctionGraph, leaf: &str) -> bool {
    op_kinds(graph).iter().any(|kind| {
        matches!(
            kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some(leaf)
        )
    })
}

fn cell_get_llbc() -> Llbc {
    let receiver = cell_ref();
    let word = usize_ty();
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "cell", "Cell"]), "kind": "Opaque"}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_cell"]),
                 "signature": {"is_unsafe": false, "inputs": [receiver.clone()], "output": word},
                 "body": {"Unstructured": {"span": span(),
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span(), "ty": word},
                        {"index": 1, "name": "slot", "span": span(), "ty": receiver}
                    ]},
                    "body": [
                        {"statements": [], "terminator": {"span": span(), "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics()}},
                                "args": [{"Copy": place(1, &receiver)}],
                                "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "terminator": {"span": span(), "kind": "Return"}},
                        {"statements": [], "terminator": {"span": span(), "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "cell", "Cell", "get"]),
                 "signature": {"is_unsafe": false, "inputs": [receiver], "output": word},
                 "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("cell get fixture")
}

fn cell_set_llbc() -> Llbc {
    let receiver = cell_ref();
    let word = usize_ty();
    let unit = json!({"Tuple": []});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "cell", "Cell"]), "kind": "Opaque"}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "write_cell"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [receiver.clone(), word.clone()], "output": unit},
                 "body": {"Unstructured": {"span": span(),
                    "locals": {"arg_count": 2, "locals": [
                        {"index": 0, "name": null, "span": span(), "ty": unit},
                        {"index": 1, "name": "slot", "span": span(), "ty": receiver},
                        {"index": 2, "name": "val", "span": span(), "ty": word}
                    ]},
                    "body": [
                        {"statements": [], "terminator": {"span": span(), "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics()}},
                                "args": [
                                    {"Copy": place(1, &receiver)},
                                    {"Copy": place(2, &word)}
                                ],
                                "dest": place(0, &unit)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "terminator": {"span": span(), "kind": "Return"}},
                        {"statements": [], "terminator": {"span": span(), "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "cell", "Cell", "set"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [receiver, word.clone()], "output": unit},
                 "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("cell set fixture")
}

fn cell_static_get_llbc() -> Llbc {
    let receiver = cell_ref();
    let word = usize_ty();
    let cell = cell_ty();
    let global_place = json!({
        "kind": {"Global": {"generics": generics(), "id": 1}},
        "ty": cell
    });
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "cell", "Cell"]), "kind": "Opaque"}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_static_cell"]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": word},
                 "body": {"Unstructured": {"span": span(),
                    "locals": {"arg_count": 0, "locals": [
                        {"index": 0, "name": null, "span": span(), "ty": word},
                        {"index": 1, "name": "slot", "span": span(), "ty": receiver}
                    ]},
                    "body": [
                        {"statements": [{"span": span(), "kind": {"Assign": [
                            place(1, &receiver),
                            {"Ref": {"place": global_place, "kind": "Shared", "ptr_metadata": "None"}}
                        ]}}], "terminator": {"span": span(), "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics()}},
                                "args": [{"Copy": place(1, &receiver)}],
                                "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "terminator": {"span": span(), "kind": "Return"}},
                        {"statements": [], "terminator": {"span": span(), "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "cell", "Cell", "get"]),
                 "signature": {"is_unsafe": false, "inputs": [receiver], "output": word},
                 "body": "Opaque"}
            ],
            "global_decls": [
                null,
                {
                    "def_id": 1,
                    "item_meta": meta(&["fixture", "COUNTER"]),
                    "global_kind": {"Static": {
                        "is_mut": false, "is_safe": true, "is_thread_local": false
                    }},
                    "ty": cell
                }
            ],
            "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("cell static fixture")
}

#[test]
fn cell_get_lowers_to_an_impure_field_load() {
    let graph = lower_function(&cell_get_llbc(), "read_cell").expect("lower read_cell");
    assert!(
        !calls_leaf(&graph, "get"),
        "Cell::get must not stay a residual call: {graph:?}"
    );
    let reads: Vec<_> = op_kinds(&graph)
        .into_iter()
        .filter_map(|kind| match kind {
            OpKind::FieldRead {
                field, ty, pure, ..
            } => Some((field.name.as_str(), ty.clone(), *pure)),
            _ => None,
        })
        .collect();
    assert_eq!(
        reads,
        vec![("value", majit_translate::model::ValueType::Unsigned, false)],
        "Cell::get is a load of the inner word: {graph:?}"
    );
}

#[test]
fn cell_set_lowers_to_a_field_store() {
    let graph = lower_function(&cell_set_llbc(), "write_cell").expect("lower write_cell");
    assert!(
        !calls_leaf(&graph, "set"),
        "Cell::set must not stay a residual call: {graph:?}"
    );
    let writes: Vec<_> = op_kinds(&graph)
        .into_iter()
        .filter_map(|kind| match kind {
            OpKind::FieldWrite { field, ty, .. } => Some((field.name.as_str(), ty.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        writes,
        vec![("value", majit_translate::model::ValueType::Unsigned)],
        "Cell::set is a store of the inner word: {graph:?}"
    );
}

#[test]
fn cell_static_get_reads_through_the_static_address() {
    let graph = lower_function(&cell_static_get_llbc(), "read_static_cell")
        .expect("lower read_static_cell");
    assert!(
        !calls_leaf(&graph, "COUNTER"),
        "Cell static must not residualize as FunctionPath: {graph:?}"
    );
    assert!(
        !calls_leaf(&graph, "get"),
        "Cell::get on a static must not stay a residual call: {graph:?}"
    );
    let addrs: Vec<i64> = op_kinds(&graph)
        .into_iter()
        .filter_map(|kind| match kind {
            OpKind::ConstRefAddr(addr) => Some(*addr),
            _ => None,
        })
        .collect();
    assert_eq!(
        addrs.len(),
        1,
        "Cell static is one address, not a residual path: {graph:?}"
    );
    assert!(
        op_kinds(&graph).iter().any(|kind| matches!(
            kind,
            OpKind::FieldRead { field, pure, .. } if field.name == "value" && !*pure
        )),
        "Cell::get of a static is a load from that address: {graph:?}"
    );
}
