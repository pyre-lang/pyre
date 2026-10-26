//! A named empty class (`struct Marker {}`) is an instance, not
//! `lltype.Void`. Constructing it and passing it as a call argument must
//! not erase to `Constant(None, Void)` — `Bookkeeper.immutablevalue` of
//! an instance is `SomeInstance(classdef)` (`bookkeeper.py`);
//! `InstanceRepr.convert_const` materialises one prebuilt instance
//! (`rclass.py`); `FunctionDesc.pycall` unions it into a callee `Ref`
//! input (`description.py`). `()` and a noncapturing closure stay on
//! their own paths.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, ConcreteType, FunctionGraph, LinkArg, OpKind};
use serde_json::{Value, json};

fn empty_class_call() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: Vec<Value>| {
        json!({
            "name": path,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let ident = |name: &str| json!({"Ident": [name, 0]});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let adt = |id| json!({"Adt": {"id": id, "generics": generics}});
    let word = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let marker_ty = adt(0);
    let layout = json!([{
        "key": "fixture-target",
        "value": {"size": 0, "align": 1, "variant_layouts": [{"field_offsets": []}], "repr": {"transparent": false}}
    }]);
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("Marker")]),
                 "kind": {"Struct": []},
                 "layout": layout}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("use_marker")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 0, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "marker", "span": span, "ty": marker_ty}
                    ]},
                    "body": [
                        {"statements": [{"span": span, "kind": {"Assign": [place(1, &marker_ty),
                            {"Aggregate": [{"Adt": [0, null, null, generics]}, []]}]}}],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Move": place(1, &marker_ty)}],
                                "dest": place(0, &word)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(vec![ident("fixture"), ident("pass_marker")]),
                 "signature": {"is_unsafe": false, "inputs": [marker_ty], "output": word},
                 "body": "Opaque"}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("empty class fixture parses")
}

fn defined_by_const_none(
    graph: &FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstNone))
    })
}

fn operand_is_void_none(graph: &FunctionGraph, arg: &LinkArg) -> bool {
    let LinkArg::Value(var) = arg else {
        return false;
    };
    FunctionGraph::concretetype_of(var) == ConcreteType::Void || defined_by_const_none(graph, var)
}

#[test]
fn named_empty_class_passed_as_call_arg_is_not_void_none() {
    let graph = lower_function(&empty_class_call(), "use_marker").expect("use_marker lowers");
    let mut marker_operands = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            match target {
                CallTarget::Method { name, .. } if name == "pass_marker" && !args.is_empty() => {
                    marker_operands.push(&args[0]);
                }
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().map(String::as_str) == Some("pass_marker")
                        && !args.is_empty() =>
                {
                    marker_operands.push(&args[0]);
                }
                _ => {}
            }
        }
    }
    assert!(
        !marker_operands.is_empty(),
        "the lowered graph must pass Marker to pass_marker"
    );
    assert!(
        marker_operands
            .iter()
            .all(|arg| !operand_is_void_none(&graph, arg)),
        "the named empty class must stay an instance, not Constant(None, Void)"
    );
}
