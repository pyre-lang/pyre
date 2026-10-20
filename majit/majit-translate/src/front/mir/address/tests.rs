/// Widening integer storage keeps every address bit (`rbuiltin.py` `gen_cast`).
#[test]
fn int_scalar_width_covers_word_accepts_widening_integer() {
    assert!(super::int_scalar_width_covers_word("U64", 8));
    assert!(super::int_scalar_width_covers_word("I64", 8));
    assert!(super::int_scalar_width_covers_word("U128", 8));
    assert!(!super::int_scalar_width_covers_word("U32", 8));
    assert!(super::int_scalar_width_covers_word("U64", 4));
    assert!(super::int_scalar_width_covers_word("U32", 4));
    assert!(!super::int_scalar_width_covers_word("U16", 4));
}

/// `a` calls `b` and `c`, both call `d`. `e` calls `f`, and `f` calls `e` and `d`.
#[test]
fn address_escape_walk_memo_matches_an_unmemoized_walk() {
    use super::{AddressEscapeWalk, function_address_escape, plain_local};
    use majit_charon_reader::Llbc;
    use serde_json::{Value, json};

    const A: u64 = 0;
    const B: u64 = 1;
    const C: u64 = 2;
    const D: u64 = 3;
    const E: u64 = 4;
    const F: u64 = 5;

    fn span() -> Value {
        json!({"Deduplicated": 0})
    }
    fn local_place(index: u64) -> Value {
        json!({"kind": {"Local": index}, "ty": {"Deduplicated": 0}})
    }
    fn bb(kind: Value) -> Value {
        json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span(), "kind": kind}
        })
    }
    fn call_term(fun: u64, target: u64, on_unwind: u64) -> Value {
        json!({
            "Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": [{"Move": local_place(1)}]
                },
                "dest": local_place(0),
                "target": target,
                "on_unwind": on_unwind
            }
        })
    }
    fn call_then_return(fun: u64) -> Vec<Value> {
        vec![
            bb(call_term(fun, 1, 2)),
            bb(json!("Return")),
            bb(json!("UnwindResume")),
        ]
    }
    fn fun_decl(def_id: u64, name: &str, body: Vec<Value>) -> Value {
        json!({
            "def_id": def_id,
            "item_meta": {
                "name": [{"Ident": [name, 0]}],
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [{"Deduplicated": 0}],
                "output": {"Deduplicated": 0}
            },
            "body": {"Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span(), "ty": {"Deduplicated": 0}},
                        {"index": 1, "name": "p", "span": span(), "ty": {"Deduplicated": 0}}
                    ]
                },
                "body": body
            }}
        })
    }

    let file = json!({
        "charon_version": "t",
        "has_errors": false,
        "translated": {
            "crate_name": "c",
            "fun_decls": [
                fun_decl(A, "a", vec![
                    bb(call_term(B, 1, 3)),
                    bb(call_term(C, 2, 3)),
                    bb(json!("Return")),
                    bb(json!("UnwindResume")),
                ]),
                fun_decl(B, "b", call_then_return(D)),
                fun_decl(C, "c", call_then_return(D)),
                fun_decl(D, "d", vec![bb(json!("Return"))]),
                fun_decl(E, "e", call_then_return(F)),
                fun_decl(F, "f", vec![
                    bb(call_term(E, 1, 3)),
                    bb(call_term(D, 2, 3)),
                    bb(json!("Return")),
                    bb(json!("UnwindResume")),
                ]),
            ],
            "files": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let entry = [plain_local(1, 1, 0)];

    let mut memo = AddressEscapeWalk::new();
    let a_memo = function_address_escape(&llbc, A, &entry, &mut memo);
    let e_memo = function_address_escape(&llbc, E, &entry, &mut memo);
    assert_eq!(memo.walk_count(D), 1);

    let mut raw_a = AddressEscapeWalk::without_memo();
    let a_raw = function_address_escape(&llbc, A, &entry, &mut raw_a);
    let mut raw_e = AddressEscapeWalk::without_memo();
    let e_raw = function_address_escape(&llbc, E, &entry, &mut raw_e);
    assert_eq!(a_memo, a_raw);
    assert_eq!(e_memo, e_raw);
    assert!(raw_a.walk_count(D) > 1);
}

/// Residual `&i64` of an oopspec callee is the key word, not a spill cell.
///
/// `jtransform.py` `handle_builtin_call` rewrites oopspecs, including
/// `ordereddict.lookup` → `_handle_dict_lookup_call` / `OS_DICT_LOOKUP`.
/// Copy-sized `Q` is one residual word (`lookup_key_jit_abi`
/// `with_residual_key`). Spilling a `malloc(8)` buffer and passing its
/// address makes that wrapper read the buffer pointer as the key bits.
#[test]
fn oopspec_residual_i64_ref_passes_the_word_not_a_spill_cell() {
    use crate::front::mir::lower_function;
    use crate::model::{CallTarget, LinkArg, OpKind};
    use majit_charon_reader::Llbc;
    use serde_json::{Value, json};

    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let local = |index: u64, name: Option<&str>, ty: &Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let place = |local: u64, ty: &Value| json!({"kind": {"Local": local}, "ty": ty});
    let word = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let key_ty = json!({"Ref": ["Erased", word.clone(), "Shared"]});
    let wrapper_locals = vec![
        local(0, None, &word),
        local(1, Some("key"), &word),
        local(2, None, &word),
    ];
    let wrapper = json!({
        "def_id": 0,
        "item_meta": meta(&["probe", "fwd_val"]),
        "signature": {"is_unsafe": false, "inputs": [word.clone()], "output": word.clone()},
        "body": {"Unstructured": {"span": span, "locals": {"arg_count": 1, "locals": wrapper_locals}, "body": [
            {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": [{"Read": place(1, &word)}]}, "dest": place(2, &word),
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &word),
                {"Use": [{"Read": place(2, &word)}, "Yes"]}
            ]}}], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}
    });
    let leaf = json!({
        "def_id": 1,
        "item_meta": meta(&["probe", "leaf"]),
        "signature": {"is_unsafe": false, "inputs": [key_ty], "output": word},
        "body": Value::Null
    });
    let oopspec = json!({
        "def_id": 0,
        "item_meta": meta(&["probe", "oopspec_leaf"]),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": "ordereddict.lookup"}, {"Deduplicated": 1}]]}
    });
    let file = json!({"charon_version": "0.1.201", "has_errors": false, "translated": {
        "crate_name": "probe", "type_decls": [], "fun_decls": [wrapper, leaf],
        "global_decls": [oopspec], "trait_decls": [], "trait_impls": []
    }});
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("wrapper fixture parses");
    let paths = super::oopspec_fn_paths(&llbc);
    assert!(
        paths.contains("leaf"),
        "ordereddict.lookup marker must name the residual leaf, got {paths:?}"
    );
    let graph =
        lower_function(&llbc, "fwd_val").unwrap_or_else(|err| panic!("lower fwd_val: {err}"));
    let has_malloc = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(op.kind, OpKind::RawMalloc { .. }))
    });
    assert!(
        !has_malloc,
        "oopspec residual `&i64` must not spill a key buffer\n{}",
        graph.dump()
    );
    let mut leaf_arg = None;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::Call { target, args, .. } = &op.kind else {
            continue;
        };
        let CallTarget::FunctionPath { segments, .. } = target else {
            continue;
        };
        if segments.last().map(String::as_str) != Some("leaf") {
            continue;
        }
        leaf_arg = args.first().and_then(LinkArg::as_variable).cloned();
    }
    let leaf_arg = leaf_arg.unwrap_or_else(|| panic!("missing leaf call\n{}", graph.dump()));
    let key = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Input { name, .. } if name == "key" => op.result.clone(),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing key input\n{}", graph.dump()));
    assert_eq!(
        leaf_arg,
        key,
        "oopspec residual `&i64` must receive the key word\n{}",
        graph.dump()
    );
}
