//! A `Result<T, S>` whose variants are marked-exception or carrier payloads
//! lowers to raise and except links.
//!
//! `exception_sum_g` and `exception_sum_call_h` are separate functions. A
//! shared returnblock makes `merged_forward_target` fire on the hop into
//! that block, even when the hop itself is empty. The selector may share
//! its returnblock: its `from_residual` tails are deferred, not
//! tail-forwarded.

use std::collections::HashSet;

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TyRef;
use serde_json::{Value, json};

use crate::flowspace::model::Variable;
use crate::front::result_exc::{
    ExceptionSum, ExceptionSumPayload, exception_sum_of_result, tyref_is_result_of_carrier,
    tyref_lowers_as_result_exc,
};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind,
    exception_exitcase,
};

const E_ID: u64 = 0;
const S_ID: u64 = 1;
const T_ID: u64 = 2;
const CARRIER_ID: u64 = 3;
const RESULT_ID: u64 = 4;

const F_ID: u64 = 0;
const G_ID: u64 = 1;
const H_ID: u64 = 2;
const CALL_H_ID: u64 = 3;
const K_ID: u64 = 4;
const FROM_RESIDUAL_ID: u64 = 5;
const SRC_ID: u64 = 6;
const MATCH_ID: u64 = 7;
const OWN_ID: u64 = 8;
const NON_SUM_ID: u64 = 9;
const LEAK_SINK_ID: u64 = 10;
const LEAK_ID: u64 = 11;
const RETHROW_ID: u64 = 12;

fn carrier_spec() -> crate::ErrorCarrierSpec<'static> {
    crate::ErrorCarrierSpec {
        carrier_path: "carrier::PyError",
        carrier_class: "",
        carrier_wrappers: &[],
        to_exc_object: Some(&["carrier", "to_exc_object"]),
        from_exc_object: None,
    }
}

fn host() -> crate::HostStaticAddrs<'static> {
    crate::HostStaticAddrs {
        error_carrier: carrier_spec(),
        ..Default::default()
    }
}

fn span() -> Value {
    super::tests::window_span()
}

fn meta(path: &[&str], is_local: bool) -> Value {
    super::tests::window_meta(path, is_local)
}

fn usize_ty() -> Value {
    super::tests::window_usize()
}

fn u32_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U32"}}})
}

fn isize_ty() -> Value {
    json!({"Scalar": {"Integer": {"Signed": "Isize"}}})
}

fn adt_ref(id: u64) -> Value {
    json!({"Adt": {"id": id, "generics": super::tests::window_empty_generics()}})
}

fn named_field(name: &str, ty: &Value) -> Value {
    json!({"name": name, "ty": ty, "attr_info": null})
}

fn tuple_field(ty: &Value) -> Value {
    json!({"name": "_0", "is_positional": true, "ty": ty, "attr_info": null})
}

fn variant(name: &str, fields: Vec<Value>, disc: i64) -> Value {
    json!({
        "name": name,
        "fields": fields,
        "discriminant": {"Scalar": {"Signed": ["Isize", disc.to_string()]}}
    })
}

fn enum_decl(id: u64, path: &[&str], variants: Vec<Value>) -> Value {
    json!({
        "def_id": id,
        "item_meta": meta(path, true),
        "kind": {"Enum": variants}
    })
}

fn struct_decl(id: u64, path: &[&str]) -> Value {
    json!({
        "def_id": id,
        "item_meta": meta(path, true),
        "kind": {"Struct": []}
    })
}

fn marker(id: u64, path: &[&str], value: &str) -> Value {
    json!({
        "def_id": id,
        "item_meta": meta(path, true),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": value}, {"Deduplicated": 1}]]}
    })
}

fn local(index: u64, ty: &Value) -> Value {
    super::tests::window_local(index, ty)
}

fn place(index: u64, ty: &Value) -> Value {
    super::tests::window_place(json!({"Local": index}), ty)
}

fn copy_of(place: Value) -> Value {
    json!({"Read": place})
}

fn assign(dest: Value, rvalue: Value) -> Value {
    json!({"span": span(), "kind": {"Assign": [dest, rvalue]}})
}

fn regular(stmts: Vec<Value>, term: Value) -> Value {
    json!({
        "statements": stmts,
        "kind": "Regular",
        "terminator": {"span": span(), "kind": term}
    })
}

fn ret_block(stmts: Vec<Value>) -> Value {
    regular(stmts, json!("Return"))
}

fn unwind_block() -> Value {
    regular(vec![], json!("UnwindResume"))
}

fn aggregate(type_id: u64, variant_idx: u64, operands: Vec<Value>) -> Value {
    json!({"Aggregate": [{"Adt": [type_id, variant_idx, null]}, operands]})
}

fn aggregate_generic(
    type_id: u64,
    variant_idx: u64,
    types: &[Value],
    operands: Vec<Value>,
) -> Value {
    json!({"Aggregate": [
        {"Adt": [
            {
                "id": type_id,
                "generics": {
                    "regions": [],
                    "types": types,
                    "const_generics": [],
                    "trait_refs": []
                }
            },
            variant_idx,
            null
        ]},
        operands
    ]})
}

fn const_usize_use(text: &str) -> Value {
    let ty = usize_ty();
    json!({"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", text]}}, ty]}, "No"]})
}

fn use_copy(
    base: Value,
    base_ty: &Value,
    variant_idx: u64,
    field_idx: u64,
    field_ty: &Value,
) -> Value {
    let down = super::tests::window_place(
        json!({"Projection": [base, {"Downcast": variant_idx}]}),
        base_ty,
    );
    let field = super::tests::window_place(
        json!({"Projection": [down, {"Field": [variant_idx, field_idx]}]}),
        field_ty,
    );
    json!({"Use": [{"Read": field}, "No"]})
}

fn switch_int(discr: Value, arms: &[(i64, u64)], default_bb: u64) -> Value {
    let pairs: Vec<Value> = arms
        .iter()
        .map(|(disc, bb)| json!([{"Scalar": {"Signed": ["Isize", disc.to_string()]}}, bb]))
        .collect();
    json!({"Switch": {
        "discr": {"Read": discr},
        "targets": {"SwitchInt": [null, pairs, default_bb]}
    }})
}

fn call_term(fun: u64, args: Vec<Value>, dest: Value, target: u64, unwind: u64) -> Value {
    json!({"Call": {"call": {
        "func": {"Regular": {
            "kind": {"Fun": fun},
            "generics": super::tests::window_empty_generics()
        }},
        "args": args}, "dest": dest, "target": target, "on_unwind": unwind}})
}

fn unstructured(arg_count: u64, locals: Vec<Value>, body: Vec<Value>) -> Value {
    json!({"Unstructured": {
        "span": span(),
        "locals": {"arg_count": arg_count, "locals": locals},
        "body": body
    }})
}

fn fun(id: u64, name: &str, inputs: Vec<Value>, output: Value, body: Value) -> Value {
    json!({
        "def_id": id,
        "item_meta": meta(&["fixture", "synth", name], true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": body
    })
}

fn opaque(id: u64, name: &str, inputs: Vec<Value>, output: Value) -> Value {
    json!({
        "def_id": id,
        "item_meta": meta(&["fixture", "synth", name], false),
        "signature": {"is_unsafe": true, "inputs": inputs, "output": output},
        "body": "Opaque"
    })
}

fn parse_llbc(types: Vec<Value>, funs: Vec<Value>, globals: Vec<Value>) -> Llbc {
    let file = json!({
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
    Llbc::from_slice(file.to_string().as_bytes()).expect("exception sum fixture parses")
}

fn result_of(ok: &Value, err: &Value) -> Value {
    super::tests::result_adt_node(RESULT_ID, ok.clone(), err.clone())
}

fn fixture() -> Llbc {
    let u = usize_ty();
    let u32 = u32_ty();
    let isize = isize_ty();
    let e = adt_ref(E_ID);
    let s = adt_ref(S_ID);
    let t = adt_ref(T_ID);
    let carrier = adt_ref(CARRIER_ID);
    let sum_result = result_of(&u, &s);
    let carrier_result = result_of(&u, &carrier);
    let e_result = result_of(&u, &e);
    let non_sum_result = result_of(&u32, &t);

    let e_decl = enum_decl(
        E_ID,
        &["fixture", "synth", "E"],
        vec![
            variant("A", vec![named_field("n", &u)], 0),
            variant("B", vec![named_field("n", &u)], 1),
            variant("C", vec![named_field("n", &u)], 2),
        ],
    );
    let s_decl = enum_decl(
        S_ID,
        &["fixture", "synth", "S"],
        vec![
            variant("Own", vec![tuple_field(&e)], 0),
            variant("Py", vec![tuple_field(&carrier)], 1),
        ],
    );
    let t_decl = enum_decl(
        T_ID,
        &["fixture", "synth", "T"],
        vec![
            variant("X", vec![tuple_field(&u32)], 0),
            variant("Y", vec![tuple_field(&carrier)], 1),
        ],
    );

    let selector = fun(
        F_ID,
        "exception_sum_f",
        vec![isize.clone(), carrier.clone()],
        sum_result.clone(),
        unstructured(
            2,
            vec![
                local(0, &sum_result),
                local(1, &isize),
                local(2, &carrier),
                local(3, &u),
                local(4, &e),
                local(5, &s),
                local(6, &s),
                local(7, &e_result),
                local(8, &e),
                local(9, &u),
            ],
            vec![
                regular(
                    vec![],
                    switch_int(place(1, &isize), &[(0, 2), (1, 3), (2, 5)], 8),
                ),
                unwind_block(),
                ret_block(vec![
                    assign(place(3, &u), const_usize_use("1")),
                    assign(
                        place(4, &e),
                        aggregate(E_ID, 1, vec![copy_of(place(3, &u))]),
                    ),
                    assign(
                        place(5, &s),
                        aggregate(S_ID, 0, vec![copy_of(place(4, &e))]),
                    ),
                    assign(
                        place(0, &sum_result),
                        aggregate_generic(
                            RESULT_ID,
                            1,
                            &[u.clone(), s.clone()],
                            vec![copy_of(place(5, &s))],
                        ),
                    ),
                ]),
                regular(
                    vec![assign(
                        place(6, &s),
                        aggregate(S_ID, 1, vec![copy_of(place(2, &carrier))]),
                    )],
                    call_term(
                        FROM_RESIDUAL_ID,
                        vec![copy_of(place(6, &s))],
                        place(0, &sum_result),
                        4,
                        1,
                    ),
                ),
                ret_block(vec![]),
                regular(vec![], call_term(K_ID, vec![], place(7, &e_result), 6, 1)),
                regular(
                    vec![assign(
                        place(8, &e),
                        use_copy(place(7, &e_result), &e_result, 1, 0, &e),
                    )],
                    call_term(
                        FROM_RESIDUAL_ID,
                        vec![copy_of(place(8, &e))],
                        place(0, &sum_result),
                        7,
                        1,
                    ),
                ),
                ret_block(vec![]),
                ret_block(vec![
                    assign(place(9, &u), const_usize_use("0")),
                    assign(
                        place(0, &sum_result),
                        aggregate_generic(
                            RESULT_ID,
                            0,
                            &[u.clone(), s.clone()],
                            vec![copy_of(place(9, &u))],
                        ),
                    ),
                ]),
            ],
        ),
    );

    let same_type = fun(
        G_ID,
        "exception_sum_g",
        vec![],
        sum_result.clone(),
        unstructured(
            0,
            vec![local(0, &sum_result)],
            vec![
                regular(vec![], call_term(G_ID, vec![], place(0, &sum_result), 2, 1)),
                unwind_block(),
                ret_block(vec![]),
            ],
        ),
    );

    let carrier_tail = fun(
        CALL_H_ID,
        "exception_sum_call_h",
        vec![],
        sum_result.clone(),
        unstructured(
            0,
            vec![local(0, &carrier_result)],
            vec![
                regular(
                    vec![],
                    call_term(H_ID, vec![], place(0, &carrier_result), 2, 1),
                ),
                unwind_block(),
                ret_block(vec![]),
            ],
        ),
    );

    let caller = fun(
        MATCH_ID,
        "exception_sum_match",
        vec![],
        carrier_result.clone(),
        unstructured(
            0,
            vec![
                local(0, &carrier_result),
                local(1, &sum_result),
                local(2, &isize),
                local(3, &u),
                local(4, &s),
                local(5, &e),
                local(6, &u),
                local(7, &carrier),
            ],
            vec![
                regular(
                    vec![],
                    call_term(SRC_ID, vec![], place(1, &sum_result), 2, 1),
                ),
                unwind_block(),
                regular(
                    vec![assign(
                        place(2, &isize),
                        json!({"Discriminant": place(1, &sum_result)}),
                    )],
                    switch_int(place(2, &isize), &[(0, 3)], 4),
                ),
                ret_block(vec![
                    assign(
                        place(3, &u),
                        use_copy(place(1, &sum_result), &sum_result, 0, 0, &u),
                    ),
                    assign(
                        place(0, &carrier_result),
                        aggregate_generic(
                            RESULT_ID,
                            0,
                            &[u.clone(), carrier.clone()],
                            vec![copy_of(place(3, &u))],
                        ),
                    ),
                ]),
                regular(
                    vec![
                        assign(
                            place(4, &s),
                            use_copy(place(1, &sum_result), &sum_result, 1, 0, &s),
                        ),
                        assign(place(2, &isize), json!({"Discriminant": place(4, &s)})),
                    ],
                    switch_int(place(2, &isize), &[(0, 5)], 6),
                ),
                regular(
                    vec![
                        assign(place(5, &e), use_copy(place(4, &s), &s, 0, 0, &e)),
                        assign(place(2, &isize), json!({"Discriminant": place(5, &e)})),
                    ],
                    switch_int(place(2, &isize), &[(2, 7)], 8),
                ),
                ret_block(vec![
                    assign(
                        place(7, &carrier),
                        use_copy(place(4, &s), &s, 1, 0, &carrier),
                    ),
                    assign(
                        place(0, &carrier_result),
                        aggregate_generic(
                            RESULT_ID,
                            1,
                            &[u.clone(), carrier.clone()],
                            vec![copy_of(place(7, &carrier))],
                        ),
                    ),
                ]),
                ret_block(vec![
                    assign(place(6, &u), use_copy(place(5, &e), &e, 2, 0, &u)),
                    assign(
                        place(0, &carrier_result),
                        aggregate_generic(
                            RESULT_ID,
                            0,
                            &[u.clone(), carrier.clone()],
                            vec![copy_of(place(6, &u))],
                        ),
                    ),
                ]),
                ret_block(vec![
                    assign(place(6, &u), use_copy(place(5, &e), &e, 0, 0, &u)),
                    assign(
                        place(0, &carrier_result),
                        aggregate_generic(
                            RESULT_ID,
                            0,
                            &[u.clone(), carrier.clone()],
                            vec![copy_of(place(6, &u))],
                        ),
                    ),
                ]),
            ],
        ),
    );

    let own = fun(
        OWN_ID,
        "exception_sum_own",
        vec![],
        sum_result.clone(),
        unstructured(
            0,
            vec![
                local(0, &sum_result),
                local(1, &u),
                local(2, &e),
                local(3, &s),
            ],
            vec![ret_block(vec![
                assign(place(1, &u), const_usize_use("1")),
                assign(
                    place(2, &e),
                    aggregate(E_ID, 1, vec![copy_of(place(1, &u))]),
                ),
                assign(
                    place(3, &s),
                    aggregate(S_ID, 0, vec![copy_of(place(2, &e))]),
                ),
                assign(
                    place(0, &sum_result),
                    aggregate_generic(
                        RESULT_ID,
                        1,
                        &[u.clone(), s.clone()],
                        vec![copy_of(place(3, &s))],
                    ),
                ),
            ])],
        ),
    );

    let non_sum = fun(
        NON_SUM_ID,
        "exception_sum_t",
        vec![carrier.clone()],
        non_sum_result.clone(),
        unstructured(
            1,
            vec![local(0, &non_sum_result), local(1, &carrier), local(2, &t)],
            vec![ret_block(vec![
                assign(
                    place(2, &t),
                    aggregate(T_ID, 1, vec![copy_of(place(1, &carrier))]),
                ),
                assign(
                    place(0, &non_sum_result),
                    aggregate_generic(
                        RESULT_ID,
                        1,
                        &[u32.clone(), t.clone()],
                        vec![copy_of(place(2, &t))],
                    ),
                ),
            ])],
        ),
    );

    let leak = fun(
        LEAK_ID,
        "exception_sum_leak",
        vec![e.clone()],
        sum_result.clone(),
        unstructured(
            1,
            vec![
                local(0, &sum_result),
                local(1, &e),
                local(2, &s),
                local(3, &u),
                local(4, &u),
            ],
            vec![
                regular(
                    vec![assign(
                        place(2, &s),
                        aggregate(S_ID, 0, vec![copy_of(place(1, &e))]),
                    )],
                    call_term(
                        LEAK_SINK_ID,
                        vec![copy_of(place(2, &s))],
                        place(3, &u),
                        2,
                        1,
                    ),
                ),
                unwind_block(),
                ret_block(vec![
                    assign(place(4, &u), const_usize_use("0")),
                    assign(
                        place(0, &sum_result),
                        aggregate_generic(
                            RESULT_ID,
                            0,
                            &[u.clone(), s.clone()],
                            vec![copy_of(place(4, &u))],
                        ),
                    ),
                ]),
            ],
        ),
    );

    // Own arm: FieldRead of `S::Own`, then `return Err(S::Own(e))` in the
    // next block. The pure forward is skipped, so the handler input is
    // `e` with no producer. Carrier arm returns `Err` and reraises.
    let rethrow = fun(
        RETHROW_ID,
        "exception_sum_rethrow",
        vec![],
        sum_result.clone(),
        unstructured(
            0,
            vec![
                local(0, &sum_result),
                local(1, &sum_result),
                local(2, &isize),
                local(3, &u),
                local(4, &s),
                local(5, &e),
                local(6, &s),
                local(7, &carrier),
            ],
            vec![
                regular(
                    vec![],
                    call_term(SRC_ID, vec![], place(1, &sum_result), 2, 1),
                ),
                unwind_block(),
                regular(
                    vec![assign(
                        place(2, &isize),
                        json!({"Discriminant": place(1, &sum_result)}),
                    )],
                    switch_int(place(2, &isize), &[(0, 3)], 4),
                ),
                ret_block(vec![
                    assign(
                        place(3, &u),
                        use_copy(place(1, &sum_result), &sum_result, 0, 0, &u),
                    ),
                    assign(
                        place(0, &sum_result),
                        aggregate_generic(
                            RESULT_ID,
                            0,
                            &[u.clone(), s.clone()],
                            vec![copy_of(place(3, &u))],
                        ),
                    ),
                ]),
                regular(
                    vec![
                        assign(
                            place(4, &s),
                            use_copy(place(1, &sum_result), &sum_result, 1, 0, &s),
                        ),
                        assign(place(2, &isize), json!({"Discriminant": place(4, &s)})),
                    ],
                    switch_int(place(2, &isize), &[(0, 5)], 7),
                ),
                regular(
                    vec![assign(place(5, &e), use_copy(place(4, &s), &s, 0, 0, &e))],
                    json!({"Goto": {"target": 6}}),
                ),
                ret_block(vec![
                    assign(
                        place(6, &s),
                        aggregate(S_ID, 0, vec![copy_of(place(5, &e))]),
                    ),
                    assign(
                        place(0, &sum_result),
                        aggregate_generic(
                            RESULT_ID,
                            1,
                            &[u.clone(), s.clone()],
                            vec![copy_of(place(6, &s))],
                        ),
                    ),
                ]),
                ret_block(vec![
                    assign(
                        place(7, &carrier),
                        use_copy(place(4, &s), &s, 1, 0, &carrier),
                    ),
                    assign(
                        place(6, &s),
                        aggregate(S_ID, 1, vec![copy_of(place(7, &carrier))]),
                    ),
                    assign(
                        place(0, &sum_result),
                        aggregate_generic(
                            RESULT_ID,
                            1,
                            &[u.clone(), s.clone()],
                            vec![copy_of(place(6, &s))],
                        ),
                    ),
                ]),
            ],
        ),
    );

    parse_llbc(
        vec![
            e_decl,
            s_decl,
            t_decl,
            struct_decl(CARRIER_ID, &["carrier", "PyError"]),
            super::tests::result_enum(RESULT_ID),
        ],
        vec![
            selector,
            same_type,
            opaque(H_ID, "exception_sum_h", vec![], carrier_result.clone()),
            carrier_tail,
            opaque(K_ID, "exception_sum_k", vec![], e_result.clone()),
            opaque(
                FROM_RESIDUAL_ID,
                "from_residual",
                vec![s.clone()],
                sum_result.clone(),
            ),
            opaque(SRC_ID, "exception_sum_src", vec![], sum_result.clone()),
            caller,
            own,
            non_sum,
            opaque(LEAK_SINK_ID, "exception_sum_leak_sink", vec![s], u.clone()),
            leak,
            rethrow,
        ],
        vec![marker(
            0,
            &["fixture", "synth", "_exception_class_E"],
            "C(B)",
        )],
    )
}

fn output_of<'a>(llbc: &'a Llbc, name: &str) -> &'a TyRef {
    &llbc
        .local_fn(name)
        .unwrap_or_else(|| panic!("missing {name}"))
        .signature
        .output
}

fn token(llbc: &Llbc, name: &str) -> Option<String> {
    super::lltype::dont_look_inside_return_token(
        output_of(llbc, name),
        llbc,
        carrier_spec(),
        &HashSet::new(),
        &HashSet::new(),
    )
}

fn lower_named(llbc: &Llbc, name: &str) -> FunctionGraph {
    match super::lower_function_with_static_addrs(llbc, name, host()) {
        Ok(graph) => graph,
        Err(err) => panic!("{name}: {err}"),
    }
}

fn apply_runtime_edges(graph: &mut FunctionGraph, llbc: &Llbc) {
    let hints = super::lltype::llbc_exception_class_hints(llbc).clone();
    crate::codewriter::error_carrier_edges::lower_error_carrier_edges(
        graph,
        &crate::OwnedErrorCarrierSpec::own(carrier_spec()),
        &|name| crate::front::llbc_hints::exception_class_name_belongs(&hints, name),
    );
}

fn call_leafs(graph: &FunctionGraph) -> Vec<String> {
    super::tests::window_call_leafs(graph)
}

fn blocks_calling(graph: &FunctionGraph, leaf: &str) -> Vec<usize> {
    graph
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block.operations.iter().any(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => segments.last().map(String::as_str) == Some(leaf),
                _ => false,
            })
        })
        .map(|(index, _)| index)
        .collect()
}

fn producing<'a>(graph: &'a FunctionGraph, var: &Variable) -> Option<&'a OpKind> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| (op.result.as_ref() == Some(var)).then_some(&op.kind))
    })
}

fn follow_same_as(graph: &FunctionGraph, mut var: Variable) -> Variable {
    for _ in 0..4 {
        let Some(kind) = producing(graph, &var) else {
            break;
        };
        if let OpKind::UnaryOp { op, operand, .. } = kind
            && op == "same_as"
        {
            var = operand.clone();
            continue;
        }
        if (crate::model::cast_instance_root(kind).is_some()
            || crate::model::cast_pointer_root(kind).is_some())
            && let OpKind::Call { args, .. } = kind
            && let Some(next) = args.first().and_then(LinkArg::as_variable)
        {
            var = next.clone();
            continue;
        }
        break;
    }
    var
}

fn fresh_raises(graph: &FunctionGraph) -> Vec<Variable> {
    let mut out = Vec::new();
    for block in &graph.blocks {
        let [link] = block.exits.as_slice() else {
            continue;
        };
        if link.target != graph.exceptblock
            || link.last_exception.is_some()
            || link.last_exc_value.is_some()
        {
            continue;
        }
        let [LinkArg::Value(etype), LinkArg::Value(evalue)] = link.args.as_slice() else {
            continue;
        };
        let typed = block.operations.iter().any(|op| {
            op.result.as_ref() == Some(etype)
                && matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        args,
                        ..
                    } if segments.len() == 1
                        && segments[0] == "type"
                        && args.len() == 1
                        && args[0].as_variable() == Some(evalue)
                )
        });
        if typed {
            out.push(evalue.clone());
        }
    }
    out
}

#[derive(Debug)]
enum Raised {
    Ctor { name: String, owner: Vec<String> },
    Field { owner: String },
    Other,
}

fn classify(graph: &FunctionGraph, mut var: Variable) -> Raised {
    for _ in 0..8 {
        let Some(kind) = producing(graph, &var) else {
            return Raised::Other;
        };
        match kind {
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => var = operand.clone(),
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                ..
            } => {
                return Raised::Ctor {
                    name: name.clone(),
                    owner: owner_path.clone(),
                };
            }
            OpKind::FieldRead { field, .. } => {
                return Raised::Field {
                    owner: field.owner_root.clone().unwrap_or_default(),
                };
            }
            _ => return Raised::Other,
        }
    }
    Raised::Other
}

fn describe_raises(graph: &FunctionGraph) -> String {
    fresh_raises(graph)
        .iter()
        .map(|var| format!("{:?}", classify(graph, var.clone())))
        .collect::<Vec<_>>()
        .join("; ")
}

fn err_arg_leaf_is_e(owner: &str) -> bool {
    let Some(head) = owner.strip_suffix("::Err") else {
        return false;
    };
    let Some(open) = head.find('<') else {
        return false;
    };
    if !head.ends_with('>') {
        return false;
    }
    let interior = &head[open + 1..head.len() - 1];
    interior.split(',').any(|arg| {
        let leaf = arg.trim().rsplit("::").next().unwrap_or(arg);
        leaf == "E"
    })
}

fn ctor_rows(graph: &FunctionGraph) -> Vec<(String, Vec<String>)> {
    super::tests::window_graph_ops(graph)
        .into_iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                ..
            } => Some((name.clone(), owner_path.clone())),
            _ => None,
        })
        .collect()
}

fn forbidden_leaf(leaf: &str) -> bool {
    matches!(
        leaf,
        "from" | "from_residual" | "to_exc_object" | "pyerror_to_exc_object"
    )
}

fn assert_no_forbidden_leafs(graph: &FunctionGraph) {
    let bad: Vec<_> = call_leafs(graph)
        .into_iter()
        .filter(|leaf| forbidden_leaf(leaf))
        .collect();
    assert!(
        bad.is_empty(),
        "forbidden call leafs {bad:?}\n{}",
        graph.dump()
    );
}

fn assert_no_sum_ctor(graph: &FunctionGraph) {
    let rows = ctor_rows(graph);
    assert!(
        rows.iter()
            .all(|(_, owner)| owner != &vec!["synth::S".to_string()]),
        "sum ctor survived {rows:?}\n{}",
        graph.dump()
    );
}

fn is_reraise(graph: &FunctionGraph, link: &Link) -> bool {
    link.exitcase.as_ref() == Some(&exception_exitcase())
        && link.target == graph.exceptblock
        && link.last_exception.is_some()
        && link.last_exc_value.is_some()
        && link.llexitcase.is_none()
}

fn assert_tail_reraise(graph: &FunctionGraph, leaf: &str) {
    let callers = blocks_calling(graph, leaf);
    assert_eq!(
        callers.len(),
        1,
        "{leaf} callers {callers:?}\n{}",
        graph.dump()
    );
    let block = &graph.blocks[callers[0]];
    assert!(
        matches!(block.exitswitch, Some(ExitSwitch::LastException)),
        "{leaf} exitswitch {:?}\n{}",
        block.exitswitch,
        graph.dump()
    );
    assert_eq!(block.exits.len(), 2, "{leaf}\n{}", graph.dump());
    assert!(
        block.exits[0].exitcase.is_none(),
        "{leaf} normal exit {:?}\n{}",
        block.exits[0].exitcase,
        graph.dump()
    );
    assert!(
        is_reraise(graph, &block.exits[1]),
        "{leaf} exception link is not a reraise: {:?}\n{}",
        block.exits[1],
        graph.dump()
    );
    assert_no_sum_ctor(graph);
    assert_no_forbidden_leafs(graph);
    assert!(
        graph.return_type.is_none(),
        "{leaf} {:?}",
        graph.return_type
    );
}

fn class_name(exitcase: &Option<ExitCase>) -> Option<String> {
    match exitcase {
        Some(ExitCase::ExceptionClass(class)) => Some(class.classdef().borrow().name.clone()),
        _ => None,
    }
}

fn find_n(graph: &FunctionGraph, start: BlockId) -> (BlockId, String, Variable) {
    let mut id = start;
    for _ in 0..5 {
        let block = graph.block(id);
        if let Some((owner, base)) = block.operations.iter().find_map(|op| match &op.kind {
            OpKind::FieldRead { base, field, .. } if field.name == "n" => {
                Some((field.owner_root.clone().unwrap_or_default(), base.clone()))
            }
            _ => None,
        }) {
            return (id, owner, base);
        }
        if block.exits.len() != 1 || block.exitswitch.is_some() {
            break;
        }
        id = block.exits[0].target;
    }
    panic!("no n read from {start:?}\n{}", graph.dump());
}

fn exc_value_feeds(graph: &FunctionGraph, link: &Link, read_at: BlockId, base: &Variable) -> bool {
    let mut tracked = follow_same_as(graph, base.clone());
    let mut id = read_at;
    for _ in 0..5 {
        let block = graph.block(id);
        let Some(pos) = block.inputargs.iter().position(|arg| arg == &tracked) else {
            return false;
        };
        if id == link.target {
            return link.args.get(pos).and_then(LinkArg::as_variable)
                == link.last_exc_value.as_ref().and_then(LinkArg::as_variable);
        }
        let mut pred: Option<(BlockId, usize)> = None;
        for (source_index, source) in graph.blocks.iter().enumerate() {
            for (exit_index, exit) in source.exits.iter().enumerate() {
                if exit.target == id {
                    if pred.is_some() {
                        return false;
                    }
                    pred = Some((BlockId(source_index), exit_index));
                }
            }
        }
        let Some((source, exit_index)) = pred else {
            return false;
        };
        let Some(next) = graph.blocks[source.0].exits[exit_index]
            .args
            .get(pos)
            .and_then(LinkArg::as_variable)
            .cloned()
        else {
            return false;
        };
        tracked = next;
        id = source;
    }
    false
}

fn reaches_return(graph: &FunctionGraph, mut id: BlockId) -> bool {
    for _ in 0..6 {
        if id == graph.returnblock {
            return true;
        }
        let block = graph.block(id);
        if block.exits.len() != 1 {
            return false;
        }
        id = block.exits[0].target;
    }
    false
}

fn sum_of<'a>(llbc: &'a Llbc, name: &str) -> Option<ExceptionSum> {
    exception_sum_of_result(output_of(llbc, name), llbc, carrier_spec())
}

#[test]
fn exception_sum_predicate() {
    let llbc = fixture();
    let sum = sum_of(&llbc, "exception_sum_f").expect("S is an exception sum");
    assert_eq!(sum.identity, "synth::S");
    assert_eq!(sum.path, "fixture::synth::S");
    assert_eq!(sum.variants.len(), 2);
    match &sum.variants[0].payload {
        ExceptionSumPayload::Marked {
            enum_path,
            base_class,
            classes,
        } => {
            assert_eq!(enum_path, "synth::E");
            assert_eq!(base_class, "synth::E");
            assert_eq!(
                classes,
                &vec![
                    ("A".to_string(), "synth::E::A".to_string()),
                    ("B".to_string(), "synth::E::B".to_string()),
                    ("C".to_string(), "synth::E::C".to_string()),
                ]
            );
        }
        ExceptionSumPayload::Carrier => panic!("Own payload is the marked enum"),
    }
    assert!(matches!(
        sum.variants[1].payload,
        ExceptionSumPayload::Carrier
    ));
    assert!(sum_of(&llbc, "exception_sum_own").is_some());
    assert!(sum_of(&llbc, "exception_sum_rethrow").is_some());
    assert!(sum_of(&llbc, "exception_sum_g").is_some());
    assert!(sum_of(&llbc, "exception_sum_call_h").is_some());

    assert!(sum_of(&llbc, "exception_sum_h").is_none());
    assert!(tyref_is_result_of_carrier(
        output_of(&llbc, "exception_sum_h"),
        &llbc,
        carrier_spec()
    ));
    assert!(tyref_lowers_as_result_exc(
        output_of(&llbc, "exception_sum_h"),
        &llbc,
        carrier_spec()
    ));
    assert!(sum_of(&llbc, "exception_sum_match").is_none());
    assert!(tyref_is_result_of_carrier(
        output_of(&llbc, "exception_sum_match"),
        &llbc,
        carrier_spec()
    ));

    assert!(sum_of(&llbc, "exception_sum_k").is_none());
    assert!(!tyref_is_result_of_carrier(
        output_of(&llbc, "exception_sum_k"),
        &llbc,
        carrier_spec()
    ));
    assert!(!tyref_lowers_as_result_exc(
        output_of(&llbc, "exception_sum_k"),
        &llbc,
        carrier_spec()
    ));
    assert!(sum_of(&llbc, "exception_sum_t").is_none());
    assert!(!tyref_is_result_of_carrier(
        output_of(&llbc, "exception_sum_t"),
        &llbc,
        carrier_spec()
    ));
    assert!(!tyref_lowers_as_result_exc(
        output_of(&llbc, "exception_sum_t"),
        &llbc,
        carrier_spec()
    ));

    let bare_s: TyRef = serde_json::from_value(adt_ref(S_ID)).expect("S ty");
    assert!(exception_sum_of_result(&bare_s, &llbc, carrier_spec()).is_none());
    assert!(!tyref_is_result_of_carrier(&bare_s, &llbc, carrier_spec()));

    assert_eq!(token(&llbc, "exception_sum_f").as_deref(), Some("u64"));
    assert_eq!(token(&llbc, "exception_sum_g").as_deref(), Some("u64"));
    assert_eq!(token(&llbc, "exception_sum_own").as_deref(), Some("u64"));
    assert_eq!(token(&llbc, "exception_sum_call_h").as_deref(), Some("u64"));
    assert_eq!(token(&llbc, "exception_sum_match").as_deref(), Some("u64"));
    assert_eq!(token(&llbc, "exception_sum_k").as_deref(), Some("ref"));
    assert_eq!(token(&llbc, "exception_sum_t").as_deref(), Some("ref"));
}

#[test]
fn exception_sum_selector_raises() {
    let llbc = fixture();
    let graph = lower_named(&llbc, "exception_sum_f");
    let raises = fresh_raises(&graph);
    assert_eq!(
        raises.len(),
        3,
        "raises: {}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    let mut ctors = Vec::new();
    let mut fields = Vec::new();
    let mut others = 0usize;
    for var in &raises {
        match classify(&graph, var.clone()) {
            Raised::Ctor { name, owner } => ctors.push((name, owner)),
            Raised::Field { owner } => fields.push(owner),
            Raised::Other => others += 1,
        }
    }
    assert_eq!(
        ctors,
        vec![("B".to_string(), vec!["synth::E".to_string()])],
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert_eq!(
        fields.len(),
        1,
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert!(
        err_arg_leaf_is_e(&fields[0]),
        "k residual owner {}\n{}",
        fields[0],
        graph.dump()
    );
    assert_eq!(others, 1, "{}\n{}", describe_raises(&graph), graph.dump());
    let leafs = call_leafs(&graph);
    assert!(
        leafs.iter().any(|leaf| leaf == "exception_sum_k"),
        "k call missing {leafs:?}\n{}",
        graph.dump()
    );
    assert_eq!(
        leafs.iter().filter(|leaf| leaf.as_str() == "type").count(),
        3,
        "{leafs:?}\n{}",
        graph.dump()
    );
    assert_no_forbidden_leafs(&graph);
    assert_no_sum_ctor(&graph);
    assert!(graph.return_type.is_none(), "{:?}", graph.return_type);
    assert!(
        !graph
            .blocks
            .iter()
            .any(|block| block.exits.iter().any(|link| is_reraise(&graph, link))),
        "selector must not reraise; g and h are separate\n{}",
        graph.dump()
    );
}

#[test]
fn exception_sum_same_type_reraise() {
    let llbc = fixture();
    let graph = lower_named(&llbc, "exception_sum_g");
    assert_tail_reraise(&graph, "exception_sum_g");
}

#[test]
fn exception_sum_carrier_residual_propagates() {
    let llbc = fixture();
    let graph = lower_named(&llbc, "exception_sum_call_h");
    assert_tail_reraise(&graph, "exception_sum_h");
}

#[test]
fn exception_sum_caller_links() {
    let llbc = fixture();
    let graph = lower_named(&llbc, "exception_sum_match");
    let callers = blocks_calling(&graph, "exception_sum_src");
    assert_eq!(callers.len(), 1, "{callers:?}\n{}", graph.dump());
    let block = &graph.blocks[callers[0]];
    assert!(
        matches!(block.exitswitch, Some(ExitSwitch::LastException)),
        "{:?}\n{}",
        block.exitswitch,
        graph.dump()
    );
    assert_eq!(block.exits.len(), 4, "{}", graph.dump());
    let normal = &block.exits[0];
    assert!(
        normal.exitcase.is_none(),
        "{:?}\n{}",
        normal.exitcase,
        graph.dump()
    );
    assert!(normal.last_exception.is_none() && normal.last_exc_value.is_none());
    assert!(normal.llexitcase.is_none());
    assert!(
        reaches_return(&graph, normal.target),
        "Ok arm does not return\n{}",
        graph.dump()
    );

    let specific = &block.exits[1];
    let binding = &block.exits[2];
    let reraise = &block.exits[3];
    assert_eq!(
        class_name(&specific.exitcase).as_deref(),
        Some("synth::E::C")
    );
    assert_eq!(class_name(&binding.exitcase).as_deref(), Some("synth::E"));
    assert!(specific.llexitcase.is_none() && binding.llexitcase.is_none());
    assert!(specific.last_exception.is_some() && specific.last_exc_value.is_some());
    assert!(binding.last_exception.is_some() && binding.last_exc_value.is_some());
    assert!(
        is_reraise(&graph, reraise),
        "carrier arm {:?}\n{}",
        reraise.exitcase,
        graph.dump()
    );
    assert_eq!(
        reraise.args.len(),
        2,
        "reraise args {}\n{}",
        reraise.args.len(),
        graph.dump()
    );

    let (ec_at, ec_owner, ec_base) = find_n(&graph, specific.target);
    let (e_at, e_owner, e_base) = find_n(&graph, binding.target);
    assert_eq!(ec_owner, "synth::E::B", "{}", graph.dump());
    assert_eq!(e_owner, "synth::E::A", "{}", graph.dump());
    assert!(
        exc_value_feeds(&graph, specific, ec_at, &ec_base),
        "EC arm does not bind the caught instance\n{}",
        graph.dump()
    );
    assert!(
        exc_value_feeds(&graph, binding, e_at, &e_base),
        "E arm does not bind the caught instance\n{}",
        graph.dump()
    );
    assert!(
        graph.blocks.iter().all(|block| {
            block
                .exits
                .iter()
                .all(|link| !matches!(link.exitcase, Some(ExitCase::ErrorCarrier)))
        }),
        "carrier arm is a reraise\n{}",
        graph.dump()
    );
    assert_no_sum_ctor(&graph);
    assert!(graph.return_type.is_none(), "{:?}", graph.return_type);
}

#[test]
fn exception_sum_own_raise_skips_exc_conversion() {
    let llbc = fixture();
    let mut graph = lower_named(&llbc, "exception_sum_own");
    let before = fresh_raises(&graph);
    assert_eq!(
        before.len(),
        1,
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    match classify(&graph, before[0].clone()) {
        Raised::Ctor { name, owner } => {
            assert_eq!(name, "B");
            assert_eq!(owner, vec!["synth::E".to_string()]);
        }
        other => panic!("own raise {other:?}\n{}", graph.dump()),
    }
    assert_no_forbidden_leafs(&graph);
    apply_runtime_edges(&mut graph, &llbc);
    let after = fresh_raises(&graph);
    assert_eq!(
        after.len(),
        1,
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert_eq!(
        after[0],
        before[0],
        "own raise was rewritten\n{}",
        graph.dump()
    );
    match classify(&graph, after[0].clone()) {
        Raised::Ctor { name, owner } => {
            assert_eq!(name, "B");
            assert_eq!(owner, vec!["synth::E".to_string()]);
        }
        other => panic!("edges rewrote the own raise to {other:?}\n{}", graph.dump()),
    }
    assert_no_forbidden_leafs(&graph);
}

#[test]
fn exception_sum_err_payload_field_read_skips_exc_conversion() {
    let llbc = fixture();
    let mut graph = lower_named(&llbc, "exception_sum_f");
    let before = fresh_raises(&graph);
    let mut ctor = None;
    let mut field = None;
    let mut carrier = None;
    for var in &before {
        match classify(&graph, var.clone()) {
            Raised::Ctor { name, owner } => {
                assert_eq!(name, "B");
                assert_eq!(owner, vec!["synth::E".to_string()]);
                ctor = Some(var.clone());
            }
            Raised::Field { owner } => {
                assert!(
                    err_arg_leaf_is_e(&owner),
                    "k residual owner {owner}\n{}",
                    graph.dump()
                );
                field = Some(var.clone());
            }
            Raised::Other => carrier = Some(var.clone()),
        }
    }
    let ctor = ctor.expect("own ctor raise");
    let field = field.expect("Err payload field read");
    let carrier = carrier.expect("carrier raise");
    apply_runtime_edges(&mut graph, &llbc);
    let after = fresh_raises(&graph);
    assert!(
        after.iter().any(|var| var == &ctor),
        "ctor raise converted\n{}",
        graph.dump()
    );
    assert!(
        after.iter().any(|var| var == &field),
        "field-read raise converted\n{}",
        graph.dump()
    );
    assert!(
        !after.iter().any(|var| var == &carrier),
        "carrier raise was not converted\n{}",
        graph.dump()
    );
    assert_eq!(
        after.len(),
        3,
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert!(
        after
            .iter()
            .any(|var| raise_call_leaf(&graph, var) == Some("to_exc_object")),
        "carrier materialisation missing\n{}",
        graph.dump()
    );
}

#[test]
fn exception_sum_handler_reraise_skips_exc_conversion() {
    let llbc = fixture();
    let mut graph = lower_named(&llbc, "exception_sum_rethrow");
    assert_no_forbidden_leafs(&graph);
    let before = fresh_raises(&graph);
    assert_eq!(
        before.len(),
        1,
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert_eq!(
        crate::front::result_exc::raised_value_class_name(&graph, &before[0]).as_deref(),
        Some("synth::E"),
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    apply_runtime_edges(&mut graph, &llbc);
    let after = fresh_raises(&graph);
    assert_eq!(
        after.len(),
        1,
        "{}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert_eq!(
        after[0],
        before[0],
        "handler reraise was rewritten\n{}",
        graph.dump()
    );
    assert_eq!(
        crate::front::result_exc::raised_value_class_name(&graph, &after[0]).as_deref(),
        Some("synth::E"),
        "{}",
        graph.dump()
    );
    assert_no_forbidden_leafs(&graph);
}

fn raise_call_leaf(graph: &FunctionGraph, var: &Variable) -> Option<&'static str> {
    let kind = producing(graph, &follow_same_as(graph, var.clone()))?;
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return None;
    };
    match segments.last().map(String::as_str) {
        Some("to_exc_object") => Some("to_exc_object"),
        Some("pyerror_to_exc_object") => Some("pyerror_to_exc_object"),
        _ => None,
    }
}

#[test]
fn exception_sum_non_sum_unchanged() {
    let llbc = fixture();
    let graph = lower_named(&llbc, "exception_sum_t");
    assert_eq!(token(&llbc, "exception_sum_t").as_deref(), Some("ref"));
    assert_ne!(graph.return_type.as_deref(), Some("u64"));
    assert_ne!(graph.return_type.as_deref(), Some("u32"));
    let rows = ctor_rows(&graph);
    assert!(
        rows.iter()
            .any(|(name, owner)| name == "Y" && owner == &vec!["synth::T".to_string()]),
        "T::Y missing {rows:?}\n{}",
        graph.dump()
    );
    assert!(
        rows.iter().any(|(name, owner)| {
            name == "Err" && owner.iter().any(|seg| seg.contains("Result"))
        }),
        "Result::Err missing {rows:?}\n{}",
        graph.dump()
    );
    assert!(
        fresh_raises(&graph).is_empty(),
        "non-sum raised {}\n{}",
        describe_raises(&graph),
        graph.dump()
    );
    assert!(
        graph
            .blocks
            .iter()
            .all(|block| !matches!(block.exitswitch, Some(ExitSwitch::LastException))),
        "{}",
        graph.dump()
    );
    assert!(
        graph.blocks.iter().any(|block| {
            block
                .exits
                .iter()
                .any(|link| link.exitcase.is_none() && link.target == graph.returnblock)
        }),
        "non-sum does not return normally\n{}",
        graph.dump()
    );
}

#[test]
fn exception_sum_value_use_unsupported() {
    let llbc = fixture();
    let err = match super::lower_function_with_static_addrs(&llbc, "exception_sum_leak", host()) {
        Ok(graph) => panic!("leak lowered\n{}", graph.dump()),
        Err(err) => err.to_string(),
    };
    assert_eq!(
        err,
        "unsupported MIR: exception sum fixture::synth::S used as a value"
    );
}
