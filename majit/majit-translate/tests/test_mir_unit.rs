//! Erasing a call's effects must not erase its unit-valued SSA definition.
//! rpython/rtyper/rmodel.py::pairtype(Repr, VoidRepr).convert_from_to
//! returns a real Constant(None, Void), not an undefined variable.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{ConcreteType, FunctionGraph, LinkArg, OpKind};
use serde_json::json;

#[test]
fn mem_forget_returns_a_defined_unit_value() {
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
    let unit = json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics}});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let place = |id, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture", "type_decls": [],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "forget_and_return"]),
                 "signature": {"is_unsafe": false, "inputs": [word], "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "value", "span": span, "ty": word},
                        {"index": 2, "name": "unit", "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Move": place(1, &word)}], "dest": place(2, &unit)},
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
                {"def_id": 1, "item_meta": meta(&["core", "mem", "forget"]),
                 "signature": {"is_unsafe": false, "inputs": [word], "output": unit}, "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(&["fixture", "consume_unit"]),
                 "signature": {"is_unsafe": false, "inputs": [unit], "output": word}, "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("unit fixture parses");
    let graph = lower_function(&llbc, "forget_and_return").expect("unit return lowers");
    let mut units = 0;
    for block in &graph.blocks {
        for arg in block.exits.iter().flat_map(|link| &link.args) {
            let LinkArg::Value(var) = arg else { continue };
            if FunctionGraph::concretetype_of(var) != ConcreteType::Void {
                continue;
            }
            units += 1;
            assert!(
                block.inputargs.contains(var)
                    || block.operations.iter().any(|op| {
                        op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstNone)
                    }),
                "undefined unit on outgoing link from {:?}",
                block.id
            );
        }
    }
    assert!(
        units > 0,
        "fixture must exercise the erased call's returned unit"
    );
}

#[test]
fn icf_identity_token_is_erased_to_a_defined_unit_value() {
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
    let unit = json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": generics}});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let place = |id, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture", "type_decls": [],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "token_and_return"]),
                 "signature": {"is_unsafe": false, "inputs": [word], "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "value", "span": span, "ty": word},
                        {"index": 2, "name": "unit", "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1},
                                "generics": generics}},
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
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("token fixture parses");
    let graph = lower_function(&llbc, "token_and_return").expect("token call lowers");
    let calls: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| matches!(op.kind, OpKind::Call { .. }))
        .collect();
    assert!(
        calls.iter().all(|op| match &op.kind {
            OpKind::Call {
                target: majit_translate::model::CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.as_slice() != ["majit_ir", "icf", "icf_identity_token"],
            _ => true,
        }),
        "icf_identity_token must be erased, leftover {calls:?}"
    );
    let mut units = 0;
    for block in &graph.blocks {
        for arg in block.exits.iter().flat_map(|link| &link.args) {
            let LinkArg::Value(var) = arg else { continue };
            if FunctionGraph::concretetype_of(var) != ConcreteType::Void {
                continue;
            }
            units += 1;
            assert!(
                block.inputargs.contains(var)
                    || block.operations.iter().any(|op| {
                        op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstNone)
                    }),
                "undefined unit on outgoing link from {:?}",
                block.id
            );
        }
    }
    assert!(
        units > 0,
        "fixture must exercise the erased token's returned unit"
    );
}
