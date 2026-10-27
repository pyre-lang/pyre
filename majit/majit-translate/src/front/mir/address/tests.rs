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

/// `sink(a: *mut usize, b: *mut bool)` has no body. `b = &raw mut flag`
/// of a bool parameter sits next to a `*mut usize` argument.
///
/// A body-less callee is uninspectable (`callee_returns_spill_address`).
/// `flag` is caller storage, not a `rffi.scoped_alloc` /
/// `lltype.malloc(flavor='raw')` fresh cell (`spill_group_is_fresh_cell`).
/// Argument 0 is `p: *mut usize` with `SpillAddressSource::Unknown`
/// (`local_address_source` of a parameter with no assign), so
/// `attach_aliasing_raw_arguments` returns `unspilled_raw_argument`.
///
/// Dict STR `hash_out` is `w_dict_store_object_strategy_checked_inner`
/// passing `&mut keyhash` into `w_dict_object_setitem_callback_free`
/// (`dont_look_inside`, unstructured body): that call still spills.
/// Body-less `*mut bool` with a function-local cell, a by-value `usize`,
/// and a kind `'r'` result is `rffi.scoped_alloc`
/// (`fresh_cell_usize_beside_bool_spill_into_gcref_lowers`).
#[test]
fn raw_scalar_bodyless_callee_refuses_a_bool_beside_an_unknown_raw_arg() {
    use crate::front::mir::lower_function;
    use crate::front::mir::tests::{fixture_item_meta, fixture_span, i64_ty, ident_path};
    use majit_charon_reader::Llbc;
    use serde_json::json;

    let span = fixture_span();
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let bool_ty = json!({"Scalar": "Bool"});
    let usize_ty = json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let bool_ptr = json!({"RawPtr": [bool_ty, "Mut"]});
    let usize_ptr = json!({"RawPtr": [usize_ty, "Mut"]});
    let ret = i64_ty();
    let place = |index: u64, ty: &serde_json::Value| json!({"kind": {"Local": index}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let caller = json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(ident_path(&["probe", "write_flag"])),
        "signature": {
            "is_unsafe": false,
            "inputs": [bool_ty.clone(), usize_ptr.clone()],
            "output": ret.clone()
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 2, "locals": [
                local(0, None, &ret),
                local(1, Some("flag"), &bool_ty),
                local(2, Some("p"), &usize_ptr),
                local(3, Some("b"), &bool_ptr),
                local(4, None, &ret)
            ]},
            "body": [
                {
                    "statements": [
                        {"span": span, "kind": {"Assign": [place(3, &bool_ptr), {"RawPtr": {
                            "place": place(1, &bool_ty),
                            "kind": "Mut",
                            "ptr_metadata": null
                        }}]}}
                    ],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                            "args": [
                                {"Move": place(2, &usize_ptr)},
                                {"Move": place(3, &bool_ptr)}
                            ]
                        },
                        "dest": place(4, &ret),
                        "target": 1,
                        "on_unwind": 2
                    }}}
                },
                {
                    "statements": [{"span": span, "kind": {"Assign": [
                        place(0, &ret),
                        {"Use": [{"Read": place(4, &ret)}, "Yes"]}
                    ]}}],
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
    });
    let sink = json!({
        "def_id": 1,
        "item_meta": fixture_item_meta(ident_path(&["probe", "sink"])),
        "signature": {
            "is_unsafe": false,
            "inputs": [usize_ptr.clone(), bool_ptr.clone()],
            "output": ret.clone()
        },
        "body": null
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "probe",
            "type_decls": [],
            "fun_decls": [caller, sink],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    let err = lower_function(&llbc, "write_flag").expect_err(
        "a body-less callee with a parameter bool and an unknown *mut usize must stay unlowered",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("unspilled raw argument"),
        "attach_aliasing_raw_arguments must refuse this shape: {msg}"
    );
}

/// Dict STR `hash_out`: `w_dict_store_object_strategy_checked_inner`
/// passes `&mut keyhash` into `w_dict_object_setitem_callback_free`.
/// That callee has an unstructured body (`dont_look_inside`); the
/// caller still lowers and the out-parameter is a
/// `rffi.scoped_alloc` / `lltype.malloc(flavor='raw')` cell.
#[test]
fn raw_scalar_hash_out_production_still_spills() {
    use crate::model::{CallTarget, LinkArg, OpKind};

    let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
    let Some(llbc) = crate::test_support::load_ullbc_if_present(path) else {
        return;
    };
    let graph = crate::test_support::lower_function(
        llbc,
        "dictmultiobject::w_dict_store_object_strategy_checked_inner",
    )
    .unwrap_or_else(|err| panic!("production hash_out caller must lower: {err}"));
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let dump = ops
        .iter()
        .map(|op| format!("{:?}", op.kind))
        .collect::<Vec<_>>()
        .join("\n");
    let args = ops
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::Call { target, args, .. }
                if matches!(
                    target,
                    CallTarget::FunctionPath { segments, .. }
                        if segments.last().map(String::as_str)
                            == Some("w_dict_object_setitem_callback_free")
                ) =>
            {
                Some(args.as_slice())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing callback_free call\n{dump}"));
    assert!(
        args.get(5).and_then(LinkArg::as_variable).is_some(),
        "hash_out is not a variable: {args:?}"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(&op.kind, OpKind::RawMalloc { .. })),
        "production hash_out caller must emit a raw cell\n{dump}"
    );
}
