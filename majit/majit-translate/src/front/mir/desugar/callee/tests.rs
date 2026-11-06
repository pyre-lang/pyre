use super::{FnPtrFamily, fn_ptr_family_for, profilefunc_graphs_for_call};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::Operand;

use serde_json::json;

use crate::model::{CallTarget, OpKind};

use crate::front::mir::tests::{
    adt_node, assign_stmt, bb, call_term, const_artifact, deref_place, empty_generics,
    fixture_llbc, fmt_arguments_adt, fmt_arguments_decl, fmt_format_decl,
    format_as_bytes_window_llbc, hint_must_use_decl, item_meta_json, llbc_with_trait_impls,
    llbc_with_types, local_of, move_op, opaque_decl, option_adt_node, option_enum, place_of,
    ptr_ty, ref_rvalue, ref_string, scalar_ty, span_json, string_adt, string_as_bytes_decl,
    string_decl, subject_decl, type_decl_alias, type_decl_struct, unit_fn_ptr, unit_ty,
    window_shared_slice, window_slice,
};

/// Every row of the fn-pointer family decision, including the two the
/// fixture corpus structurally cannot reach: it declares one fn-pointer
/// alias (`HostCallback`) and that alias is safe, so no corpus-driven
/// test distinguishes `is_safe` either way.
///
/// The load-bearing row is `(unsafe, builtin shape) -> None`. Its
/// positive control is the row above it: drop the safety gate and
/// `(safe, builtin shape)` still passes while that row flips to
/// `Some(BuiltinWrapper)`, so the pair fails in exactly one direction.
#[test]
fn fn_ptr_family_names_the_wrapper_set_only_for_a_safe_pointer() {
    // (is_safe, is_builtin_shape, fnptr_indirect) -> family
    let rows = [
        (true, true, false, Some(FnPtrFamily::BuiltinWrapper)),
        (true, true, true, Some(FnPtrFamily::BuiltinWrapper)),
        (false, true, false, None),
        (false, true, true, None),
        (true, false, true, Some(FnPtrFamily::Unknown)),
        // An `unsafe fn` outside the wrapper shape keeps the unknown
        // family: `None` graphs claims nothing about the callee.
        (false, false, true, Some(FnPtrFamily::Unknown)),
        (true, false, false, None),
        (false, false, false, None),
    ];
    for (is_safe, is_builtin_shape, fnptr_indirect, expected) in rows {
        assert_eq!(
            fn_ptr_family_for(is_safe, is_builtin_shape, fnptr_indirect),
            expected,
            "is_safe={is_safe} is_builtin_shape={is_builtin_shape} \
                 fnptr_indirect={fnptr_indirect}"
        );
    }
}

fn fnptr_skip_binder(
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
) -> serde_json::Value {
    json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": inputs,
                "output": output,
                "is_unsafe": true
            }
        }
    })
}

fn move_operand_ty(ty: serde_json::Value) -> Operand {
    serde_json::from_value(json!({
        "Move": {
            "kind": {"Local": 0},
            "ty": ty
        }
    }))
    .expect("operand")
}

/// Charon may inline the `EqWHookFn` alias so the Dynamic operand is a
/// bare `FnPtr` (`TyRef::Other`) with no dedup id. Classification is the
/// skip_binder, not the enclosing body's name.
#[test]
fn operand_is_dict_eq_hook_fn_matches_inlined_fnptr_by_signature() {
    let obj = json!({"RawPtr": [{"Adt": {"id": 0}}, "Mut"]});
    let eq_fnptr = fnptr_skip_binder(vec![obj.clone(), obj.clone()], json!({"Scalar": "Bool"}));
    let hash_fnptr =
        fnptr_skip_binder(vec![obj], json!({"Scalar": {"Integer": {"Signed": "I64"}}}));
    let llbc = llbc_with_types(
        "pyre_object",
        vec![],
        vec![json!({
            "def_id": 0,
            "item_meta": item_meta_json(
                &["pyre_object", "dict_eq_hook", "register_eq_w_hook"],
                "",
                true,
            ),
            "signature": {
                "is_unsafe": false,
                "inputs": [{"Value": [1, eq_fnptr.clone()]}],
                "output": {"Tuple": []}
            },
            "body": "Opaque"
        })],
    );
    let inlined = move_operand_ty(eq_fnptr);
    assert!(
        super::operand_is_dict_eq_hook_fn(&inlined, &llbc),
        "inlined alias-less FnPtr of register_eq_w_hook's parameter"
    );
    let hash_w = move_operand_ty(hash_fnptr);
    assert!(
        !super::operand_is_dict_eq_hook_fn(&hash_w, &llbc),
        "hash_w hook shape is not the eq hook"
    );
}

/// `ll_keyeq_custom` is a named body whose `hlinvoke` is `indirect_call`
/// with `graphs is None` (`rordereddict.py`, `rptr.py`
/// `PtrRepr.rtype_simple_call`). The TLS hook must lower the same way.
#[test]
fn eq_w_hooked_lowers_the_tls_hook_as_indirect_call_with_unknown_family() {
    let Some(llbc) =
        crate::test_support::load_ullbc_if_present(crate::runtime_names::artifacts::OBJECT_ULLBC)
    else {
        return;
    };
    let graph = crate::test_support::lower_function(llbc, "pyre_object::dict_eq_hook::eq_w_hooked")
        .expect("lower eq_w_hooked");
    let mut unknown_family = 0usize;
    let mut dyn_call = 0usize;
    let kinds: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| {
            match &op.kind {
                OpKind::IndirectCall { graphs, .. } if graphs.is_none() => {
                    unknown_family += 1;
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("__dyn_call") => {
                    dyn_call += 1;
                }
                _ => {}
            }
            format!("{:?}", op.kind)
        })
        .collect();
    assert_eq!(
        dyn_call, 0,
        "eq_w_hooked must not residualize the hook as __dyn_call: {kinds:?}"
    );
    assert_eq!(
        unknown_family, 1,
        "eq_w_hooked must hlinvoke the stored hook as indirect_call \
         with graphs is None: {kinds:?}"
    );
}

#[test]
#[ignore]
fn builtin_wrapper_declaration_has_builtin_code_fn_type() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = crate::test_support::load_ullbc(path);
    let wrapper = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path_str()
                .ends_with("::__majit_wrap_random")
        })
        .expect("_random::__majit_wrap_random");
    assert!(super::fun_decl_is_builtin_code_fn(wrapper, llbc));
    let non_member = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path_str()
                .ends_with("::function::funccall_valuestack")
        })
        .expect("function::funccall_valuestack");
    assert!(!super::fun_decl_is_builtin_code_fn(non_member, llbc));
}

/// rustc's promoted-const wrapper copies one `Place::Global` and
/// returns a reference. That inner id is the NamedConst `getfunctionptr`
/// would name (`Transformer.handle_residual_call`).
#[test]
fn promoted_const_initializer_copies_the_inner_global() {
    let body: majit_charon_reader::ullbc::Unstructured = serde_json::from_value(serde_json::json!({
            "span": {"Deduplicated": 0},
            "locals": {"arg_count": 0, "locals": []},
            "body": [{
                "statements": [
                    {"span": {"Deduplicated": 0}, "kind": {"StorageLive": 1}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                        {"Use": [{"Read": {
                            "kind": {"Global": {
                                "id": 31,
                                "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
                            }},
                            "ty": {"Deduplicated": 0}
                        }}, "Yes"]}
                    ]}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
                        {"Ref": {
                            "place": {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                            "kind": "Shared",
                            "ptr_metadata": {"Const": {"Deduplicated": 0}}
                        }}
                    ]}}
                ],
                "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
                "kind": "Regular"
            }]
        }))
        .expect("promoted-const wrapper fixture parses");
    let place = super::promoted_const_copied_global(&body).expect("inner global");
    assert_eq!(super::inner_global_id(&place), 31);
}

/// rustc promotes a `&str` literal to `_1 = const C; _0 = &_1`.
#[test]
fn promoted_const_initializer_copies_the_inner_const() {
    let body: majit_charon_reader::ullbc::Unstructured =
        serde_json::from_value(serde_json::json!({
            "span": {"Deduplicated": 0},
            "locals": {"arg_count": 0, "locals": []},
            "body": [{
                "statements": [
                    {"span": {"Deduplicated": 0}, "kind": {"StorageLive": 1}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                        {"Use": [{"Const": {"Deduplicated": 20437}}, "Yes"]}
                    ]}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
                        {"Ref": {
                            "place": {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                            "kind": "Shared",
                            "ptr_metadata": {"Const": {"Deduplicated": 0}}
                        }}
                    ]}}
                ],
                "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
                "kind": "Regular"
            }]
        }))
        .expect("promoted-const literal wrapper fixture parses");
    let value = super::promoted_const_copied_const(&body).expect("inner const");
    assert_eq!(
        value.get("Deduplicated").and_then(|v| v.as_u64()),
        Some(20437)
    );
}

fn promoted_const_unstructured(
    blocks: serde_json::Value,
) -> majit_charon_reader::ullbc::Unstructured {
    serde_json::from_value(serde_json::json!({
        "span": {"Deduplicated": 0},
        "locals": {"arg_count": 0, "locals": []},
        "body": blocks
    }))
    .expect("promoted-const fixture parses")
}

fn global_place(id: u64) -> serde_json::Value {
    serde_json::json!({
        "kind": {"Global": {
            "id": id,
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        }},
        "ty": {"Deduplicated": 0}
    })
}

fn local_place(id: u64) -> serde_json::Value {
    serde_json::json!({"kind": {"Local": id}, "ty": {"Deduplicated": 0}})
}

fn assign_copy(dest: serde_json::Value, src: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "span": {"Deduplicated": 0},
        "kind": {"Assign": [dest, {"Use": [{"Read": src}, "Yes"]}]}
    })
}

fn assign_shared_ref(dest: serde_json::Value, src: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "span": {"Deduplicated": 0},
        "kind": {"Assign": [dest, {"Ref": {
            "place": src,
            "kind": "Shared",
            "ptr_metadata": {"Const": {"Deduplicated": 0}}
        }}]}
    })
}

fn return_bb(statements: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "statements": statements,
        "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
        "kind": "Regular"
    })
}

/// Shape (b): `_0 = copy Global(id); return`.
#[test]
fn promoted_const_initializer_copies_the_inner_global_by_value() {
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![assign_copy(
        local_place(0),
        global_place(31)
    ),])]));
    let place = super::promoted_const_copied_global(&body).expect("inner global");
    assert_eq!(super::inner_global_id(&place), 31);
}

/// `_1 = copy G; _0 = &(_1.field); return` is not a read of G.
#[test]
fn promoted_const_initializer_rejects_field_projection_ref() {
    let field = serde_json::json!({
        "kind": {"Projection": [local_place(1), {"Field": [null, 0]}]},
        "ty": {"Deduplicated": 0}
    });
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![
        assign_copy(local_place(1), global_place(31)),
        assign_shared_ref(local_place(0), field),
    ])]));
    assert!(super::promoted_const_copied_global(&body).is_none());
}

/// rustc's promoted-const wrapper does not emit a local-to-local copy
/// chain (`_2 = copy _1; _0 = &_2`). Reject it so a copy of a copy is
/// not folded as the global.
#[test]
fn promoted_const_initializer_rejects_local_to_local_copy_chain() {
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![
        assign_copy(local_place(1), global_place(31)),
        assign_copy(local_place(2), local_place(1)),
        assign_shared_ref(local_place(0), local_place(2)),
    ])]));
    assert!(super::promoted_const_copied_global(&body).is_none());
}

/// A global copied only in an unreachable block is not the returned value.
#[test]
fn promoted_const_initializer_rejects_unreachable_block_global() {
    let unreachable = return_bb(vec![
        assign_copy(local_place(1), global_place(31)),
        assign_shared_ref(local_place(0), local_place(1)),
    ]);
    let entry = serde_json::json!({
        "statements": [],
        "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
        "kind": "Regular"
    });
    let body = promoted_const_unstructured(serde_json::json!([entry, unreachable]));
    assert!(super::promoted_const_copied_global(&body).is_none());
    assert!(super::promoted_const_copied_const(&body).is_none());
}

/// `resolve_place` of a projected global falls back to the inner Global
/// and drops the field, so a copy of `G.field` is not accepted as G.
#[test]
fn promoted_const_initializer_rejects_projected_global() {
    let projected = serde_json::json!({
        "kind": {"Projection": [global_place(31), {"Field": [null, 0]}]},
        "ty": {"Deduplicated": 0}
    });
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![
        assign_copy(local_place(1), projected),
        assign_shared_ref(local_place(0), local_place(1)),
    ])]));
    assert!(super::promoted_const_copied_global(&body).is_none());
}

/// A Charon-instantiated FunDecl's function-item path uses the same
/// specialized leaf `registered_path_for_fun_decl` registers
/// (`FunctionDesc.cachedgraph`).
#[test]
fn fundecl_fn_item_segments_uses_the_registered_instance_leaf() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let ident = |s: &str| serde_json::json!({"Ident": [s, 0]});
    let args = serde_json::json!({
        "regions": [],
        "types": [{"Scalar": {"Integer": {"Signed": "I64"}}}],
        "const_generics": [],
        "trait_refs": []
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [
                        ident("fixture"),
                        ident("f"),
                        {"Instantiated": {
                            "params": {},
                            "kind": "Other",
                            "skip_binder": args
                        }}
                    ],
                    "span": span.clone(),
                    "source_text": null,
                    "attr_info": {
                        "attributes": [],
                        "inline": null,
                        "rename": null,
                        "public": true
                    },
                    "is_local": true
                },
                "signature": {
                    "is_unsafe": false,
                    "inputs": [],
                    "output": {"Scalar": "Bool"}
                },
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {"arg_count": 0, "locals": []},
                    "body": []
                }}
            }],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("instance fixture parses");
    let fd = llbc.fn_by_id(0).expect("f");
    let registered = super::registered_path_for_fun_decl(&llbc, fd);
    let item = super::fundecl_fn_item_segments(&llbc, fd);
    assert_eq!(item, registered.segments);
    let leaf = item.last().expect("leaf");
    assert!(leaf.starts_with("f__spec_"), "{leaf}");
    assert_ne!(leaf.as_str(), "f");
}

/// A monomorphized vtable keeps its method slots as `*const ()`, so the
/// `dyn` call casts the slot into a temporary and calls that; the slot
/// read is recovered from the temporary's single producer.
#[test]
fn a_mono_vtable_call_reads_the_slot_under_its_fn_ptr_cast() {
    let ty = ptr_ty();
    let slot = serde_json::json!({"Read": {"kind": {"Projection": [
            {"kind": {"Projection": [{"kind": {"Local": 1}, "ty": ty}, "Deref"]}, "ty": ty},
            {"Field": [null, 4]}
        ]}, "ty": ty}});
    let assign = |dst: u64, rvalue: serde_json::Value| {
        serde_json::json!({"span": span_json(), "kind": {"Assign": [
            {"kind": {"Local": dst}, "ty": ty},
            rvalue
        ]}})
    };
    let local = |index: u64| serde_json::json!({"index": index, "name": null, "span": span_json(), "ty": ty});
    let fun = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "dyn_call"], "", true),
        "signature": {"is_unsafe": true, "inputs": [ty], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [local(0), local(1), local(2), local(3)]},
            "body": [{
                "statements": [
                    assign(2, serde_json::json!({"UnaryOp": [
                        {"Cast": {"RawPtr": [ty, ty]}},
                        slot
                    ]})),
                    assign(3, serde_json::json!({"Use": [slot, "No"]}))
                ],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    });
    let llbc = const_artifact("fixture", serde_json::json!([fun]), serde_json::json!([]));
    let fd = llbc.iter_local_fns().next().expect("fixture fn");
    let body = fd.unstructured().expect("fixture body");
    let operand = |index: u64| {
        serde_json::from_value::<majit_charon_reader::ullbc::Operand>(serde_json::json!(
            {"Read": {"kind": {"Local": index}, "ty": ty}}
        ))
        .unwrap()
    };
    let peeled = super::vtable_slot_under_fn_ptr_cast(&llbc, &body, &operand(2))
        .expect("the cast temporary names its vtable slot");
    let expected =
        serde_json::from_value::<majit_charon_reader::ullbc::Operand>(slot.clone()).unwrap();
    assert_eq!(format!("{peeled:?}"), format!("{expected:?}"));
    assert!(
        super::vtable_slot_under_fn_ptr_cast(&llbc, &body, &operand(3)).is_none(),
        "a plain copy of the slot is not a cast temporary"
    );
}

/// The `Vec` index fold must accept a `usize` index.
///
/// `usize` types as `Unsigned`, not `Int`, so gating on `Int` alone left
/// both the read and write lifts dead for every real callsite — caught
/// only by the `#[ignore]`d real-LLBC anchor
/// (`vec_index_mut_fill_user_function_args_real`), which does not run in
/// CI.  This pins the acceptance set without the 440MB corpus.
#[test]
fn vec_index_gate_accepts_both_integer_banks_and_rejects_range() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let ty = |v: serde_json::Value| {
        serde_json::from_value::<super::TyRef>(serde_json::json!({
            "Value": [0, v]
        }))
        .expect("fixture TyRef parses")
    };
    let usize_ty = ty(serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}}));
    assert_eq!(
        super::tyref_to_value_type(&usize_ty, &llbc),
        crate::model::ValueType::Unsigned,
        "usize must type as Unsigned — the trap this gate fell into"
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert!(
        super::vec_index_type_is_scalar(
            &usize_ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "Vec index fold must accept a usize index"
    );
    assert!(
        super::vec_index_type_is_scalar(
            &ty(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})),
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "Vec index fold must accept a signed index"
    );
    assert!(
        !super::vec_index_type_is_scalar(
            &ty(serde_json::json!({ "Adt": { "id": 7  } })),
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "a Range index types as Ref and must stay rejected"
    );
}

#[test]
fn add_dest_single_deref_guard_classifies_uses() {
    use majit_charon_reader::ullbc::Unstructured;
    // brick 3's escape guard: the `.add`-result local `_1` may be
    // dereferenced once and never escape as a raw pointer.  Each
    // fixture defines `_1` with one bare-local write (standing in for
    // the `add` call, which the guard counts the same as any other
    // bare-local definition) and varies how `_1` is then used.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let deref_place = |i: u64| {
        serde_json::json!({
            "kind": {"Projection": [place_local(i), "Deref"]}, "ty": ty()
        })
    };
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    // A single-block body: `_1 = const` (the def) followed by `extra`,
    // returning.  `dest` = `_1`.
    let body_of = |extra: Vec<serde_json::Value>| -> Unstructured {
        let mut statements = vec![stmt(
            serde_json::json!({"Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]}),
        )];
        statements.extend(extra);
        let bb = serde_json::json!({
            "statements": statements,
            "terminator": {"kind": "Return"}
        });
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [bb]
        });
        serde_json::from_value(body_json).expect("fixture Unstructured parses")
    };

    // `_2 = *_1` — one deref read: accepted.
    let read = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]
    }))]);
    assert!(super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &read,
        1
    ));

    // `*_1 = _2` — one deref write: accepted.
    let write = body_of(vec![stmt(serde_json::json!({
        "Assign": [deref_place(1), {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &write,
        1
    ));

    // `_2 = _1` — the pointer escapes by value: rejected.
    let escape = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": place_local(1)}, "No"]}]
    }))]);
    assert!(!super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &escape,
        1
    ));

    // `_2 = *_1; _3 = *_1` — two derefs: rejected.
    let twice = body_of(vec![
        stmt(
            serde_json::json!({"Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]}),
        ),
        stmt(
            serde_json::json!({"Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]}),
        ),
    ]);
    assert!(!super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &twice,
        1
    ));

    // No use at all (just the def): rejected (no element load/store).
    let dead = body_of(vec![]);
    assert!(!super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &dead,
        1
    ));
}

#[test]
fn format_str_locals_follows_must_use_and_ref() {
    let llbc = format_as_bytes_window_llbc();
    let body = llbc
        .fn_by_id(0)
        .and_then(|fd| fd.unstructured())
        .expect("subject unstructured");
    let marks = super::format_str_locals(&body, &llbc);
    assert_eq!(
        marks.get(2).copied(),
        Some(true),
        "format dest is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(3).copied(),
        Some(true),
        "must_use dest is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(4).copied(),
        Some(true),
        "shared-ref temp is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(1).copied(),
        Some(false),
        "Arguments parameter is not format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(5).copied(),
        Some(false),
        "as_bytes dest is not format_str: {marks:?}"
    );
}

fn format_mut_borrowed_format_string_llbc() -> majit_charon_reader::Llbc {
    let string = string_adt();
    let args_ty = fmt_arguments_adt();
    let unit = unit_ty();
    let shared = ref_string("Shared");
    let mut_ref = ref_string("Mut");
    let window = window_shared_slice();
    let slice = window_slice();
    let name = "format_mut_borrowed_format_string";
    llbc_with_types(
        "fixture",
        vec![string_decl(), fmt_arguments_decl()],
        vec![
            subject_decl(
                name,
                1,
                vec![args_ty.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &args_ty),
                    local_of(2, &string),
                    local_of(3, &string),
                    local_of(4, &shared),
                    local_of(5, &window),
                    local_of(6, &window),
                    local_of(7, &mut_ref),
                ],
                vec![
                    bb(
                        vec![],
                        call_term(
                            1,
                            vec![move_op(place_of(1, &args_ty))],
                            place_of(2, &string),
                            1,
                            5,
                        ),
                    ),
                    bb(
                        vec![],
                        call_term(
                            2,
                            vec![move_op(place_of(2, &string))],
                            place_of(3, &string),
                            2,
                            5,
                        ),
                    ),
                    bb(
                        vec![
                            assign_stmt(
                                place_of(7, &mut_ref),
                                ref_rvalue(place_of(3, &string), "Mut"),
                            ),
                            assign_stmt(
                                place_of(4, &shared),
                                ref_rvalue(place_of(3, &string), "Shared"),
                            ),
                        ],
                        call_term(
                            3,
                            vec![move_op(place_of(4, &shared))],
                            place_of(5, &window),
                            3,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(6, &window),
                            ref_rvalue(deref_place(5, &window, &slice), "Shared"),
                        )],
                        call_term(
                            4,
                            vec![move_op(place_of(6, &window))],
                            place_of(0, &unit),
                            4,
                            5,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            fmt_format_decl(1),
            hint_must_use_decl(2),
            string_as_bytes_decl(),
            opaque_decl(4, &["fixture", "g"], vec![window], unit),
        ],
    )
}

#[test]
fn format_str_locals_declines_a_mut_borrowed_format_string() {
    let llbc = format_mut_borrowed_format_string_llbc();
    let body = llbc
        .fn_by_id(0)
        .and_then(|fd| fd.unstructured())
        .expect("subject unstructured");
    let marks = super::format_str_locals(&body, &llbc);
    assert_eq!(
        marks.get(2).copied(),
        Some(true),
        "format dest is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(3).copied(),
        Some(false),
        "must_use dest is declined by &mut: {marks:?}"
    );
    assert_eq!(
        marks.get(4).copied(),
        Some(false),
        "shared-ref of a declined local is not format_str: {marks:?}"
    );
}

#[test]
fn add_dest_struct_field_guard_classifies_uses() {
    use majit_charon_reader::ullbc::Unstructured;
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let deref_place = |i: u64| {
        serde_json::json!({
            "kind": {"Projection": [place_local(i), "Deref"]},
            "ty": ty()
        })
    };
    let field_place = |i: u64, idx: u64| {
        serde_json::json!({
            "kind": {"Projection": [deref_place(i), {"Field": [null, idx]}]},
            "ty": ty()
        })
    };
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let body_of = |extra: Vec<serde_json::Value>| -> Unstructured {
        let mut statements = vec![stmt(serde_json::json!({
            "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
        }))];
        statements.extend(extra);
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [{
                "statements": statements,
                "terminator": {"kind": "Return"}
            }]
        });
        serde_json::from_value(body_json).expect("fixture Unstructured parses")
    };
    let llbc = fixture_llbc();
    let accepts = |body: &Unstructured| super::add_dest_used_only_as_struct_field(&llbc, body, 1);

    let read = body_of(vec![
        stmt(serde_json::json!({"StorageLive": 1})),
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
        })),
    ]);
    assert!(accepts(&read), "one field read");

    let write = body_of(vec![stmt(serde_json::json!({
        "Assign": [field_place(1, 1), {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(accepts(&write), "one field write");

    let twice = body_of(vec![
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
        })),
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 1)}, "No"]}]
        })),
    ]);
    assert!(accepts(&twice), "two field reads of one add");

    let escape = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": place_local(1)}, "No"]}]
    }))]);
    assert!(!accepts(&escape), "the pointer escapes by value");

    let bare = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]
    }))]);
    assert!(!accepts(&bare), "a bare deref is not a struct field");

    let address = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {"place": field_place(1, 0), "kind": "Shared", "ptr_metadata": null}
        }]
    }))]);
    assert!(
        !accepts(&address),
        "&(*dest).field of an unresolved type is not a value read"
    );

    let field_place_ty = |i: u64, idx: u64, field_ty: serde_json::Value| {
        serde_json::json!({
            "kind": {"Projection": [deref_place(i), {"Field": [null, idx]}]},
            "ty": field_ty
        })
    };
    let address_word = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {
                "place": field_place_ty(1, 0, scalar_ty("Usize")),
                "kind": "Shared",
                "ptr_metadata": null
            }
        }]
    }))]);
    assert!(
        accepts(&address_word),
        "&(*dest).field of a copy scalar is a value read"
    );

    let address_typevar = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {
                "place": field_place_ty(1, 0, serde_json::json!({"TypeVar": {"Bound": [0, 0]}})),
                "kind": "Shared",
                "ptr_metadata": null
            }
        }]
    }))]);
    assert!(
        accepts(&address_typevar),
        "&(*dest).field of a TypeVar is a value read"
    );

    let deeper = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": {
            "kind": {"Projection": [field_place(1, 0), {"Field": [null, 0]}]},
            "ty": ty()
        }}, "No"]}]
    }))]);
    assert!(!accepts(&deeper), "a further projection escapes");

    let nested_index = body_of(vec![stmt(serde_json::json!({
        "Assign": [{
            "kind": {"Projection": [field_place(1, 1), {"Index": {
                "offset": {"Read": place_local(2)},
                "from_end": false
            }}]},
            "ty": ty()
        }, {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(
        accepts(&nested_index),
        "(*dest).field[i] is a struct-field use"
    );

    let dead = body_of(vec![]);
    assert!(!accepts(&dead), "no field use");

    let redefined = body_of(vec![
        stmt(serde_json::json!({
            "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
        })),
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
        })),
    ]);
    assert!(!accepts(&redefined), "a second definition stays residual");

    let call_body = {
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [{
                "statements": [stmt(serde_json::json!({
                    "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
                }))],
                "kind": "Regular",
                "terminator": {
                    "kind": {
                        "Call": {
                            "call": {
                                "func": {
                                    "Regular": {
                                        "kind": {"Fun": 1},
                                        "generics": {
                                            "regions": [],
                                            "types": [],
                                            "const_generics": [],
                                            "trait_refs": []
                                        }
                                    }
                                },
                                "args": [{"Read": field_place(1, 0)}]}, "dest": place_local(2),
                            "target": 0,
                            "on_unwind": 0
                        }
                    }
                }
            }]
        });
        serde_json::from_value::<Unstructured>(body_json).expect("call fixture parses")
    };
    assert!(accepts(&call_body), "a call argument is a value field use");
}

/// Distinctive `fn(bool) -> bool` so this fixture cannot collide with
/// `unit_fn_ptr` / BuiltinCodeFn. Member names are invented (`hook_a` /
/// `hook_b`); discovery keys on the `ProfileFunc` type, not those names.
fn bool_fn_ptr() -> serde_json::Value {
    json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [{"Scalar": "Bool"}],
                "output": {"Scalar": "Bool"},
                "is_unsafe": false
            }
        }
    })
}

fn const_fndef(fun: u64, ty: &serde_json::Value) -> serde_json::Value {
    json!({"Const": [{
        "FnDef": {
            "kind": {"Fun": fun},
            "generics": empty_generics()
        }
    }, ty]})
}

fn profilefunc_fun(
    def_id: u64,
    name: &str,
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
    arg_count: u64,
    locals: Vec<serde_json::Value>,
    blocks: Vec<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "def_id": def_id,
        "item_meta": item_meta_json(&["fixture", name], "", true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": arg_count, "locals": locals},
            "body": blocks
        }}
    })
}

fn matching_profilefunc_member(def_id: u64, name: &str) -> serde_json::Value {
    let ty = json!({"Scalar": "Bool"});
    profilefunc_fun(
        def_id,
        name,
        vec![ty.clone()],
        ty.clone(),
        1,
        vec![local_of(0, &ty), local_of(1, &ty)],
        vec![bb(vec![], json!("Return"))],
    )
}

fn profilefunc_family_llbc() -> Llbc {
    let bool_fn = bool_fn_ptr();
    let profilefunc_adt = adt_node(1);
    let option_ty = option_adt_node(0, profilefunc_adt.clone());
    let ec_ty = adt_node(2);
    let unit = json!({"Tuple": []});
    let types = vec![
        option_enum(0),
        type_decl_alias(1, &["fixture", "ProfileFunc"], bool_fn.clone()),
        type_decl_struct(
            2,
            &["fixture", "ExecutionContext"],
            vec![json!({
                "name": "profilefunc",
                "ty": option_ty,
                "attr_info": null
            })],
        ),
    ];
    let hook_a = matching_profilefunc_member(0, "hook_a");
    let hook_b = matching_profilefunc_member(1, "hook_b");
    let other_fn = profilefunc_fun(
        2,
        "other_fn",
        vec![],
        unit.clone(),
        0,
        vec![local_of(0, &unit)],
        vec![bb(vec![], json!("Return"))],
    );
    let setllprofile = profilefunc_fun(
        3,
        "setllprofile",
        vec![ec_ty.clone(), option_adt_node(0, profilefunc_adt.clone())],
        unit.clone(),
        2,
        vec![
            local_of(0, &unit),
            local_of(1, &ec_ty),
            local_of(2, &option_adt_node(0, profilefunc_adt.clone())),
            local_of(3, &profilefunc_adt),
        ],
        vec![bb(
            vec![assign_stmt(
                place_of(3, &profilefunc_adt),
                json!({"Use": [const_fndef(0, &bool_fn), "Yes"]}),
            )],
            json!("Return"),
        )],
    );
    let setprofile = profilefunc_fun(
        4,
        "setprofile",
        vec![ec_ty.clone()],
        unit.clone(),
        1,
        vec![local_of(0, &unit), local_of(1, &ec_ty)],
        vec![
            bb(
                vec![],
                call_term(
                    3,
                    vec![move_op(place_of(1, &ec_ty)), const_fndef(1, &bool_fn)],
                    place_of(0, &unit),
                    1,
                    2,
                ),
            ),
            bb(vec![], json!("Return")),
            bb(vec![], json!("UnwindResume")),
        ],
    );
    llbc_with_types(
        "fixture",
        types,
        vec![hook_a, hook_b, other_fn, setllprofile, setprofile],
    )
}

/// `discover_eval_hook_graphs` analogue: the family is the `ProfileFunc`
/// type plus stores into that slot (`setllprofile` / `setprofile`), not
/// baked member names.
#[test]
fn discover_profilefunc_graphs_keys_on_type_not_member_names() {
    let llbc = profilefunc_family_llbc();
    let graphs = super::discover_profilefunc_graphs(&llbc);
    assert!(
        graphs
            .iter()
            .any(|path| path == "hook_a" || path.ends_with("::hook_a")),
        "signature-matching / stored FnDef hook_a must be in the family, got {graphs:?}"
    );
    assert!(
        graphs
            .iter()
            .any(|path| path == "hook_b" || path.ends_with("::hook_b")),
        "signature-matching / setprofile-stored FnDef hook_b must be in the family, got {graphs:?}"
    );
    assert!(
        graphs
            .iter()
            .all(|path| !path.ends_with("other_fn") && path != "other_fn"),
        "a FunDecl whose fn type is not ProfileFunc stays out, got {graphs:?}"
    );
    let spellings = super::discover_profilefunc_spellings(&llbc);
    assert!(
        spellings
            .iter()
            .any(|s| s == "ProfileFunc" || s.ends_with("::ProfileFunc")),
        "alias leaf must be published, got {spellings:?}"
    );
    assert!(
        spellings.iter().any(|s| s.contains("fn(")),
        "expanded fn(...) after alias peel must be published, got {spellings:?}"
    );
}

#[test]
fn profilefunc_graphs_for_call_attaches_the_pbc_family() {
    let llbc = profilefunc_family_llbc();
    let operand: Operand = serde_json::from_value(move_op(place_of(0, &adt_node(1))))
        .expect("ProfileFunc local deserializes");
    let graphs = profilefunc_graphs_for_call(&llbc, &operand)
        .expect("FunctionsPBCRepr.call attaches c_graphs for ProfileFunc");
    assert!(
        graphs.iter().any(|path| {
            path.last_segment() == Some("hook_a") || path.display_name().ends_with("::hook_a")
        }),
        "hook_a must be in the attached family, got {graphs:?}"
    );
    assert!(
        graphs.iter().any(|path| {
            path.last_segment() == Some("hook_b") || path.display_name().ends_with("::hook_b")
        }),
        "hook_b must be in the attached family, got {graphs:?}"
    );
    let other: Operand = serde_json::from_value(move_op(place_of(0, &unit_fn_ptr())))
        .expect("unrelated fn local deserializes");
    assert!(
        profilefunc_graphs_for_call(&llbc, &other).is_none(),
        "a non-ProfileFunc fn pointer must not claim the family"
    );
}

#[test]
fn discover_profilefunc_graphs_ignores_unrelated_fn_pointer_fields() {
    let unit = json!({"Tuple": []});
    let types = vec![
        option_enum(0),
        type_decl_struct(
            1,
            &["fixture", "Hooks"],
            vec![json!({
                "name": "callback",
                "ty": option_adt_node(0, unit_fn_ptr()),
                "attr_info": null
            })],
        ),
    ];
    let other = profilefunc_fun(
        0,
        "plain_callback",
        vec![],
        unit.clone(),
        0,
        vec![local_of(0, &unit)],
        vec![bb(vec![], json!("Return"))],
    );
    let llbc = llbc_with_types("fixture", types, vec![other]);
    assert!(
        super::discover_profilefunc_graphs(&llbc).is_empty(),
        "a non-profilefunc Option<fn> field must not become a PBC family"
    );
    assert!(
        super::discover_profilefunc_spellings(&llbc).is_empty(),
        "no ProfileFunc alias / profilefunc field ⇒ no published spelling"
    );
}

#[test]
#[ignore]
fn interpreter_ullbc_discovers_app_profile_call_as_profilefunc_member() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = crate::test_support::load_ullbc(path);
    let graphs = super::discover_profilefunc_graphs(&llbc);
    assert!(
        graphs
            .iter()
            .any(|p| p.ends_with("app_profile_call") || p == "app_profile_call"),
        "executioncontext.py setprofile stores app_profile_call; got {graphs:?}"
    );
    let spellings = super::discover_profilefunc_spellings(&llbc);
    assert!(
        spellings
            .iter()
            .any(|s| s == "ProfileFunc" || s.ends_with("::ProfileFunc")),
        "ProfileFunc alias must be published from the interpreter artefact, got {spellings:?}"
    );
}
