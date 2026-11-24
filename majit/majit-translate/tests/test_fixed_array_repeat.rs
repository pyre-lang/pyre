//! `[u8; 16]` Repeat and Array aggregate must share one annotator class.
//!
//! `rffi.CFixedArray` / `CConfig.in6_addr.s6_addr` is a FixedSizeArray,
//! never unified with an integer. A Repeat of the fill byte that aliases
//! the dest to `Unsigned` unions with a by-value `[u8; 16]` Array ctor
//! (`sockaddr_in6_get_addr`) at the `buf` join.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, LinkArg, OpKind, ValueType};
use majit_translate::translator::rtyper::legacy_annotator;
use serde_json::{Value, json};

fn u8_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})
}

fn usize_const(n: &str) -> Value {
    json!([
        {"Integer": {"Unsigned": ["Usize", n]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ])
}

fn u8xn_ty(n: &str) -> Value {
    json!({"Array": [u8_ty(), usize_const(n), null]})
}

fn u8x16_ty() -> Value {
    u8xn_ty("16")
}

fn u8_const(v: &str) -> Value {
    json!({"Const": [{"Integer": {"Unsigned": ["U8", v]}}, u8_ty()]})
}

fn u8_zero() -> Value {
    u8_const("0")
}

fn span_json() -> Value {
    json!({
        "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
    })
}

fn item_meta(path: &[&str]) -> Value {
    let span = span_json();
    json!({
        "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
        "span": span,
        "source_text": null,
        "is_local": true,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
    })
}

fn place(id: u64, ty: &Value) -> Value {
    json!({"kind": {"Local": id}, "ty": ty})
}

fn repeat_u8xn(n: &str, fill: &str, fn_name: &str) -> Llbc {
    let span = span_json();
    let array = u8xn_ty(n);
    let file = json!({
        "charon_version": "0.1.281",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": item_meta(&["fixture", fn_name]),
                "signature": {"is_unsafe": false, "inputs": [], "output": array},
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {
                        "arg_count": 0,
                        "locals": [
                            {"index": 0, "name": null, "span": span, "ty": array}
                        ]
                    },
                    "body": [{
                        "statements": [{
                            "span": span,
                            "kind": {"Assign": [
                                place(0, &array),
                                {"Repeat": {
                                    "operand": u8_const(fill),
                                    "count": usize_const(n),
                                    "elem_is_copy": null,
                                    "elem_is_sized": null
                                }}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }]
                }}
            }],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("repeat fixture parses")
}

fn mix_repeat_and_array_aggregate() -> Llbc {
    let span = span_json();
    let array = u8x16_ty();
    let flag = json!({"Scalar": "Bool"});
    let zeros: Vec<Value> = (0..16).map(|_| u8_zero()).collect();
    let file = json!({
        "charon_version": "0.1.281",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": item_meta(&["fixture", "mix_u8x16"]),
                "signature": {"is_unsafe": false, "inputs": [flag], "output": array},
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {
                        "arg_count": 1,
                        "locals": [
                            {"index": 0, "name": null, "span": span, "ty": array},
                            {"index": 1, "name": "flag", "span": span, "ty": flag}
                        ]
                    },
                    "body": [
                        {
                            "statements": [],
                            "kind": "Regular",
                            "terminator": {"span": span, "kind": {"Switch": {
                                "discr": {"Read": place(1, &flag)},
                                "targets": {"If": [1, 2]}
                            }}}
                        },
                        {
                            "statements": [{
                                "span": span,
                                "kind": {"Assign": [
                                    place(0, &array),
                                    {"Repeat": {
                                        "operand": u8_zero(),
                                        "count": usize_const("16"),
                                        "elem_is_copy": null,
                                        "elem_is_sized": null
                                    }}
                                ]}
                            }],
                            "kind": "Regular",
                            "terminator": {"span": span, "kind": "Return"}
                        },
                        {
                            "statements": [{
                                "span": span,
                                "kind": {"Assign": [
                                    place(0, &array),
                                    {"Aggregate": [
                                        {"Array": [u8_ty(), usize_const("16"), null]},
                                        zeros
                                    ]}
                                ]}
                            }],
                            "kind": "Regular",
                            "terminator": {"span": span, "kind": "Return"}
                        }
                    ]
                }}
            }],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("u8x16 mix fixture parses")
}

fn mix_repeat_and_by_value_array_call() -> Llbc {
    let span = span_json();
    let array = u8x16_ty();
    let flag = json!({"Scalar": "Bool"});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let file = json!({
        "charon_version": "0.1.281",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": item_meta(&["fixture", "mix_u8x16_call"]),
                    "signature": {"is_unsafe": false, "inputs": [flag], "output": array},
                    "body": {"Unstructured": {
                        "span": span,
                        "locals": {
                            "arg_count": 1,
                            "locals": [
                                {"index": 0, "name": null, "span": span, "ty": array},
                                {"index": 1, "name": "flag", "span": span, "ty": flag}
                            ]
                        },
                        "body": [
                            {
                                "statements": [],
                                "kind": "Regular",
                                "terminator": {"span": span, "kind": {"Switch": {
                                    "discr": {"Read": place(1, &flag)},
                                    "targets": {"If": [1, 2]}
                                }}}
                            },
                            {
                                "statements": [{
                                    "span": span,
                                    "kind": {"Assign": [
                                        place(0, &array),
                                        {"Repeat": {
                                            "operand": u8_zero(),
                                            "count": usize_const("16"),
                                            "elem_is_copy": null,
                                            "elem_is_sized": null
                                        }}
                                    ]}
                                }],
                                "kind": "Regular",
                                "terminator": {"span": span, "kind": "Return"}
                            },
                            {
                                "statements": [],
                                "kind": "Regular",
                                "terminator": {"span": span, "kind": {"Call": {
                                    "call": {
                                        "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                        "args": []
                                    },
                                    "dest": place(0, &array),
                                    "target": 3,
                                    "on_unwind": 4
                                }}}
                            },
                            {
                                "statements": [],
                                "kind": "Regular",
                                "terminator": {"span": span, "kind": "Return"}
                            },
                            {
                                "statements": [],
                                "kind": "Regular",
                                "terminator": {"span": span, "kind": "UnwindResume"}
                            }
                        ]
                    }}
                },
                {
                    "def_id": 1,
                    "item_meta": item_meta(&["fixture", "sockaddr_in6_get_addr"]),
                    "signature": {"is_unsafe": false, "inputs": [], "output": array},
                    "body": "Opaque"
                }
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("u8x16 call mix fixture parses")
}

fn result_ty_of_arrayish_calls(graph: &majit_translate::model::FunctionGraph) -> Vec<ValueType> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty,
                ..
            } if segments.as_slice() == ["__array_repeat"] => Some(result_ty.clone()),
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                result_ty,
                ..
            } if name == "Array" || name.starts_with("Array<") => Some(result_ty.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn repeat_and_array_aggregate_of_u8x16_annotate() {
    let llbc = mix_repeat_and_array_aggregate();
    let graph = lower_function(&llbc, "mix_u8x16").expect("mix_u8x16 lowers");
    let array = ValueType::Ref(Some("Array<u8;16>".into()));
    let tys = result_ty_of_arrayish_calls(&graph);
    assert!(
        tys.iter().filter(|ty| *ty == &array).count() >= 2,
        "Repeat and Aggregate of [u8; 16] must both be Array<u8;16>, got {tys:?}"
    );
    assert!(
        tys.iter()
            .all(|ty| !matches!(ty, ValueType::Unsigned | ValueType::Int)),
        "neither arm of [u8; 16] may collapse to the fill integer, got {tys:?}"
    );
    legacy_annotator::annotate(&graph);
}

#[test]
fn repeat_fill_reaches_every_array_slot() {
    let llbc = repeat_u8xn("4", "7", "fill7");
    let graph = lower_function(&llbc, "fill7").expect("fill7 lowers");
    let fill = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match (&op.kind, &op.result) {
            (OpKind::ConstUInt(7), Some(v)) => Some(v.clone()),
            _ => None,
        })
        .expect("Repeat fill 7 is a ConstUInt");
    let mut pos_stores = [false; 4];
    let mut ctor_args_ok = false;
    let mut ctor_ty_ok = false;
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::FieldWrite {
                    field,
                    value: LinkArg::Value(v),
                    ..
                } if v == &fill => {
                    if let Some(idx) = field
                        .name
                        .strip_prefix("__pos_")
                        .and_then(|s| s.parse::<usize>().ok())
                        && idx < 4
                    {
                        pos_stores[idx] = true;
                    }
                }
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    args,
                    result_ty,
                } if name == "Array<u8;4>" || name.starts_with("Array<") => {
                    ctor_ty_ok |= *result_ty == ValueType::Ref(Some("Array<u8;4>".into()));
                    ctor_args_ok |= args.len() == 4
                        && args
                            .iter()
                            .all(|arg| arg.as_variable().is_some_and(|v| v == &fill));
                }
                _ => {}
            }
        }
    }
    assert!(
        pos_stores.iter().all(|&stored| stored) || ctor_args_ok,
        "fill 7 must reach all 4 slots via FieldWrite or ctor args, stores={pos_stores:?} ctor_args={ctor_args_ok}"
    );
    assert!(
        ctor_ty_ok || pos_stores.iter().all(|&stored| stored),
        "shaped [7u8; 4] must be Array<u8;4>, not the fill integer"
    );
    legacy_annotator::annotate(&graph);
}

#[test]
fn repeat_and_by_value_u8x16_call_annotate() {
    let llbc = mix_repeat_and_by_value_array_call();
    let graph = lower_function(&llbc, "mix_u8x16_call").expect("mix_u8x16_call lowers");
    let array = ValueType::Ref(Some("Array<u8;16>".into()));
    let tys = result_ty_of_arrayish_calls(&graph);
    assert!(
        tys.iter().any(|ty| ty == &array),
        "Repeat of [0u8; 16] must be Array<u8;16>, got {tys:?}"
    );
    assert!(
        tys.iter()
            .all(|ty| !matches!(ty, ValueType::Unsigned | ValueType::Int)),
        "neither arm of [u8; 16] may collapse to the fill integer, got {tys:?}"
    );
    legacy_annotator::annotate(&graph);
}
