//! A closure that captures nothing is a layout-size-0 ADT. Its flow value
//! still names the callable (`SomePBC`); only `FunctionsPBCRepr` /
//! `SingleFrozenPBCRepr` give that callable a Void repr. Erasing the
//! aggregate to `Constant(None, Void)` makes an inlined `Option::and_then`
//! annotate `getattr(Constant(None), "call_once")`.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, ConcreteType, FunctionGraph, LinkArg, OpKind};
use serde_json::{Value, json};

fn zst_closure_and_then() -> Llbc {
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
    let option_ty = json!({"Adt": {
        "id": 0,
        "generics": {"regions": [], "types": [{"Scalar": {"Integer": {"Signed": "I64"}}}], "const_generics": [], "trait_refs": []}
    }});
    let word = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let closure_ty = adt(1);
    let layout = json!([{
        "key": "fixture-target",
        "value": {"size": 0, "align": 1, "variant_layouts": [{"field_offsets": []}], "repr": {"transparent": false}}
    }]);
    let impl_seg = json!({"Impl": {"Ty": {
        "params": {"regions": [], "types": [], "const_generics": [], "trait_clauses": [], "regions_outlive": [], "types_outlive": [], "trait_type_constraints": []},
        "skip_binder": {"Value": [0, {"Adt": {"id": 0, "generics": generics}}]},
        "kind": "InherentImplBlock"
    }}});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("core"), ident("option"), ident("Option")]),
                 "kind": {"Enum": [
                    {"name": "None", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}},
                    {"name": "Some", "fields": [{"name": null, "ty": word, "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}}],
                     "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}}
                 ]}},
                {"def_id": 1, "item_meta": meta(vec![ident("fixture"), ident("use_and_then"), ident("closure")]),
                 "kind": {"Struct": []},
                 "layout": layout,
                 "src": {"Closure": {"info": {"kind": "FnOnce"}}}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("use_and_then")]),
                 "signature": {"is_unsafe": false, "inputs": [option_ty], "output": option_ty},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": option_ty},
                        {"index": 1, "name": "opt", "span": span, "ty": option_ty},
                        {"index": 2, "name": "closure", "span": span, "ty": closure_ty}
                    ]},
                    "body": [
                        {"statements": [{"span": span, "kind": {"Assign": [place(2, &closure_ty),
                            {"Aggregate": [{"Adt": [1, null, null, generics]}, []]}]}}],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Move": place(1, &option_ty)}, {"Move": place(2, &closure_ty)}],
                                "dest": place(0, &option_ty)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(vec![ident("core"), ident("option"), ident("Option"), impl_seg, ident("and_then")]),
                 "signature": {"is_unsafe": false, "inputs": [option_ty, closure_ty], "output": option_ty},
                 "body": "Opaque"}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("closure fixture parses")
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
fn noncapturing_closure_passed_to_and_then_is_not_void_none() {
    let graph = lower_function(&zst_closure_and_then(), "use_and_then").expect("and_then lowers");
    let mut closure_operands = Vec::new();
    let mut call_once_on_void = false;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            match target {
                CallTarget::Method { name, .. } if name == "and_then" && args.len() >= 2 => {
                    closure_operands.push(&args[1]);
                }
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().map(String::as_str) == Some("and_then")
                        && args.len() >= 2 =>
                {
                    closure_operands.push(&args[1]);
                }
                CallTarget::Method { name, .. } if name == "call_once" && !args.is_empty() => {
                    if operand_is_void_none(&graph, &args[0]) {
                        call_once_on_void = true;
                    }
                    closure_operands.push(&args[0]);
                }
                _ => {}
            }
        }
    }
    assert!(
        !closure_operands.is_empty(),
        "the lowered graph must pass the closure to and_then or call_once"
    );
    assert!(
        !call_once_on_void,
        "call_once must not read a Void None constant"
    );
    assert!(
        closure_operands
            .iter()
            .all(|arg| !operand_is_void_none(&graph, arg)),
        "the closure operand must keep its callable, not Constant(None, Void)"
    );
}
