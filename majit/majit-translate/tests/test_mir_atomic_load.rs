//! Atomic ordering must survive the Rust-input boundary. RPython's
//! flowcontext.FlowContext.record_block propagates FlowingError rather than
//! publishing a graph with an unsupported operation erased.

use majit_charon_reader::Llbc;
use majit_translate::flowspace::model::Variable;
use majit_translate::front::mir::{LowerError, lower_function};
use majit_translate::model::{BlockId, CallTarget, FunctionGraph, LinkArg, OpKind};
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
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
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
fn atomic_access_real_corpus_receivers_lower() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("load fresh pyre-object corpus");
    // Quasi-immutable `version_tag: AtomicU64`. The Acquire load lowers
    // to a FieldRead so `record_quasiimmut_field` still guards it.
    let graph = lower_function(&llbc, "w_type_get_version_tag")
        .expect("Acquire load of quasi version_tag lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "lowered w_type_get_version_tag has no version_tag field read"
    );
    // get_instantiate publishes the cached W_TypeObject to allocators. Its
    // readers and late writers are all GIL-serialized, so the source spells a
    // Relaxed load — the one ordering the frontend lowers to a plain read.
    let result = lower_function(&llbc, "get_instantiate");
    assert!(
        result.is_ok(),
        "get_instantiate: the Relaxed reader must keep lowering: {result:?}"
    );
    // `w_type_is_abstract`: `_t = &(*_c)` then `&((*_t).flag_abstract)`.
    // `flag_abstract?` is quasi-immutable; the Acquire load is that field's
    // read.
    let graph = lower_function(&llbc, "w_type_is_abstract")
        .expect("Acquire load of quasi flag_abstract lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "flag_abstract"
            )
        })
    });
    assert!(
        saw,
        "lowered w_type_is_abstract has no flag_abstract field read"
    );
    let interp = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    ))
    .expect("load fresh pyre-interpreter corpus");
    for name in [
        "dev_mode_flag",
        "monitoring_tool_name",
        "monitoring_set_tool_name",
    ] {
        let result = lower_function(&interp, name);
        assert!(
            result.is_ok(),
            "{name}: static-rooted Relaxed atomic must keep lowering: {result:?}"
        );
    }
    // `hash_slot` is `from_ptr(&raw mut (*(obj as *mut W_UnicodeObject)).hash)`.
    // `w_str_get_hash` is that slot's Relaxed load. The parent is a raw
    // pointer: no `ConstRefNull` may enter a block as a link arg.
    let hash_slot = lower_function(&llbc, "hash_slot").expect("from_ptr of hash field lowers");
    let hash = lower_function(&llbc, "w_str_get_hash")
        .expect("Relaxed load of hash through from_ptr lowers");
    assert!(
        !graph_has_raw_load(&hash),
        "w_str_get_hash must not raw_load the hash_slot value"
    );
    let hash_ret = graph_return_var(&hash).expect("w_str_get_hash returns a value");
    assert!(
        traces_to_call_named(&hash, &hash_ret, "hash_slot"),
        "w_str_get_hash must return the hash_slot call result"
    );
    for (name, graph) in [("hash_slot", &hash_slot), ("w_str_get_hash", &hash)] {
        let saw_null_link = graph.blocks.iter().any(|block| {
            block.exits.iter().any(|link| {
                link.args.iter().any(|arg| {
                    let Some(var) = arg.as_variable() else {
                        return false;
                    };
                    graph.blocks.iter().any(|b| {
                        b.operations.iter().any(|op| {
                            matches!(op.kind, OpKind::ConstRefNull)
                                && op.result.as_ref().is_some_and(|r| r.id() == var.id())
                        })
                    })
                })
            })
        });
        assert!(
            !saw_null_link,
            "{name} must not pass ConstRefNull as a link arg"
        );
    }
}

#[test]
#[ignore = "loads target/llbc/pyre-interpreter.ullbc"]
fn acquire_quasi_w_globals_lowers_across_the_block_copy() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
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

enum Receiver {
    Borrow,
    Copied,
    CopiedTwice,
    CopiedChain,
    CopiedFromTwoBorrows,
    CopiedThroughOwnerCopy,
    CopiedThroughOwnerCopyInOneBranch,
    CopiedThroughAnotherOwnerCopyInOneBranch,
    CopiedFromTwoOwners,
    CopiedFromTwoFields,
    BorrowedInTwoBranches,
    BorrowedTwoFieldsInTwoBranches,
    CopiedFromCopiesOfAReassignedOwner,
    CopiedThroughDiamond,
    ParamCopiedAfterCall,
    CopiedThenOverwritten,
    OverwrittenByCall,
    ReborrowOfCallResult,
    PinnedLocalField,
    PinnedLocalsInTwoBranches,
    BorrowThroughSubstructure,
    IndexReassignedBeforeBorrow,
    IndexReassignedAfterBorrow,
    IndexZeroCapturedAtBorrow,
    ReassignedBase,
    ReassignedParam,
    OwnerReassignedAfterCall,
    OwnerReassignedBeforeCopy,
    CopiedThenOwnerReassigned,
    CopiedLastFromAnotherOwner,
    MutBorrowedOwner,
    SuffixIndex,
    ReborrowedOwner,
    RawReborrowOfCallResult,
    IndexParamReassignedAfterLoad,
    ParamBaseLaterReassigned,
    DeepSubstructure,
    ReborrowKeepsIndexCapture,
    RepeatedIndexLocal,
    MutBorrowedSubstructure,
}

#[derive(Clone, Copy, Debug)]
enum AtomicOp {
    Load(&'static str),
    RelaxedStore,
    ReleaseStore,
}

fn quasi_atomic_u64_field_fixture(
    owner: &str,
    field: &str,
    entries: Option<&str>,
    receiver: Receiver,
    op: AtomicOp,
) -> Llbc {
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
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": generics}});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let owner_adt = json!({"Adt": {"id": 2, "generics": generics, "builtin": null}});
    let owner_ref = json!({"Ref": ["Erased", owner_adt, "Shared"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let other_param = matches!(
        receiver,
        Receiver::ReassignedParam
            | Receiver::OwnerReassignedAfterCall
            | Receiver::OwnerReassignedBeforeCopy
            | Receiver::CopiedThenOwnerReassigned
    );
    let param_copied_after_call = matches!(receiver, Receiver::ParamCopiedAfterCall);
    let extra_recv_param = matches!(
        receiver,
        Receiver::ParamCopiedAfterCall | Receiver::CopiedThenOverwritten
    );
    let extra_owner_param = matches!(
        receiver,
        Receiver::CopiedFromTwoOwners
            | Receiver::BorrowedInTwoBranches
            | Receiver::CopiedFromCopiesOfAReassignedOwner
            | Receiver::CopiedLastFromAnotherOwner
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
            | Receiver::PinnedLocalsInTwoBranches
    );
    let index_param = matches!(receiver, Receiver::IndexParamReassignedAfterLoad);
    let inner_param = matches!(receiver, Receiver::ParamBaseLaterReassigned);
    let merge = matches!(
        receiver,
        Receiver::CopiedFromTwoBorrows
            | Receiver::CopiedThroughOwnerCopy
            | Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
            | Receiver::CopiedFromTwoOwners
            | Receiver::CopiedFromTwoFields
            | Receiver::BorrowedInTwoBranches
            | Receiver::BorrowedTwoFieldsInTwoBranches
            | Receiver::CopiedFromCopiesOfAReassignedOwner
            | Receiver::CopiedThroughDiamond
            | Receiver::PinnedLocalsInTwoBranches
    );
    let pinned_local = matches!(
        receiver,
        Receiver::PinnedLocalField | Receiver::PinnedLocalsInTwoBranches
    );
    let borrow_through_substructure = matches!(
        receiver,
        Receiver::BorrowThroughSubstructure
            | Receiver::ParamBaseLaterReassigned
            | Receiver::MutBorrowedSubstructure
    );
    let dynamic_index = matches!(
        receiver,
        Receiver::IndexReassignedBeforeBorrow
            | Receiver::IndexReassignedAfterBorrow
            | Receiver::IndexZeroCapturedAtBorrow
            | Receiver::IndexParamReassignedAfterLoad
            | Receiver::ReborrowKeepsIndexCapture
            | Receiver::RepeatedIndexLocal
    );
    let is_store = matches!(op, AtomicOp::RelaxedStore | AtomicOp::ReleaseStore);
    let arg_count = 1
        + usize::from(other_param)
        + usize::from(is_store)
        + usize::from(extra_recv_param)
        + usize::from(extra_owner_param)
        + usize::from(index_param)
        + usize::from(inner_param)
        + usize::from(merge);
    let extra_data_param = arg_count - usize::from(merge);
    let value_local =
        extra_data_param - usize::from(extra_recv_param) - usize::from(extra_owner_param);
    let bool_ty = json!({"Scalar": "Bool"});
    let ordering_local = arg_count + 1;
    let recv_bind = arg_count + 2;
    let extra_local = arg_count + 3;
    let field_base = match receiver {
        Receiver::ReassignedBase | Receiver::ReborrowedOwner => extra_local,
        Receiver::CopiedFromCopiesOfAReassignedOwner => extra_local + 4,
        _ => 1,
    };
    let dest_ty = if is_store { unit.clone() } else { word.clone() };
    let suffix_index = matches!(receiver, Receiver::SuffixIndex);
    let array_field = suffix_index || dynamic_index;
    let usize_ty = json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let inner_adt = json!({"Adt": {"id": 3, "generics": generics, "builtin": null}});
    let inner_ref = json!({"Ref": ["Erased", inner_adt, "Shared"]});
    let owner_param_ty = if pinned_local {
        owner_adt.clone()
    } else {
        owner_ref.clone()
    };
    let array_len = json!([
        {"Integer": {"Unsigned": ["Usize", "2"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let array_ty = json!({"Array": [atomic, array_len, null]});
    let field_leaf_ty = if array_field {
        array_ty.clone()
    } else {
        atomic.clone()
    };
    let field_proj = if pinned_local {
        json!({
            "kind": {"Projection": [place(field_base, &owner_adt), {"Field": [null, 0]}]},
            "ty": field_leaf_ty.clone()
        })
    } else {
        json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(field_base, &owner_ref), "Deref"]}, "ty": owner_adt},
                {"Field": [null, 0]}
            ]},
            "ty": field_leaf_ty.clone()
        })
    };
    let field_place = if suffix_index {
        json!({
            "kind": {"Projection": [
                field_proj,
                {"Index": {"offset": {"Const": [
                    {"Integer": {"Unsigned": ["Usize", "1"]}},
                    {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
                ]}, "from_end": true}}
            ]},
            "ty": atomic
        })
    } else if dynamic_index {
        let index_local = if index_param {
            extra_data_param
        } else {
            extra_local
        };
        json!({
            "kind": {"Projection": [
                field_proj,
                {"Index": {"offset": {"Copy": place(index_local, &usize_ty)}, "from_end": false}}
            ]},
            "ty": atomic
        })
    } else {
        field_proj
    };
    let owner_field_ty = if borrow_through_substructure {
        inner_adt.clone()
    } else if array_field {
        array_ty.clone()
    } else {
        atomic.clone()
    };
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let ordering_name = match op {
        AtomicOp::Load(name) => name,
        AtomicOp::RelaxedStore => "Relaxed",
        AtomicOp::ReleaseStore => "Release",
    };
    let ordering_index = variants.iter().position(|v| *v == ordering_name).unwrap();
    let attr = json!({"attributes": [], "inline": null, "rename": null, "public": true});
    let global_decls = match entries {
        Some(joined) => json!([{
            "def_id": 0,
            "item_meta": meta(&["fixture", &format!("_immutable_fields_{owner}")]),
            "global_kind": "NamedConst",
            "value": {"Value": [1, [{"Str": joined}, {"Ref": [
                "Erased",
                {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "Shared"
            ]}]]}
        }]),
        None => json!([]),
    };
    let mut locals = vec![
        json!({"index": 0, "name": null, "span": span, "ty": dest_ty}),
        json!({"index": 1, "name": "owner", "span": span, "ty": owner_param_ty.clone()}),
    ];
    if other_param {
        locals.push(json!({"index": 2, "name": "other", "span": span, "ty": owner_ref}));
    }
    if is_store {
        locals.push(json!({"index": value_local, "name": "value", "span": span, "ty": word}));
    }
    if extra_recv_param {
        locals.push(json!({"index": extra_data_param, "name": "r", "span": span, "ty": recv_ty}));
    }
    if extra_owner_param {
        let extra_owner_name = match receiver {
            Receiver::CopiedFromCopiesOfAReassignedOwner => "p",
            Receiver::CopiedThroughAnotherOwnerCopyInOneBranch => "owner_b",
            _ => "owner2",
        };
        locals.push(
            json!({"index": extra_data_param, "name": extra_owner_name, "span": span, "ty": owner_param_ty}),
        );
    }
    if index_param {
        locals.push(json!({"index": extra_data_param, "name": "i", "span": span, "ty": usize_ty}));
    }
    if inner_param {
        locals.push(json!({"index": extra_data_param, "name": "s", "span": span, "ty": inner_ref}));
    }
    if merge {
        locals.push(json!({"index": arg_count, "name": "flag", "span": span, "ty": bool_ty}));
    }
    locals.push(json!({"index": ordering_local, "name": "ordering", "span": span, "ty": adt_ord}));
    locals.push(json!({"index": recv_bind, "name": "recv", "span": span, "ty": recv_ty}));
    let mut statements = vec![
        json!({"span": span, "kind": {"Assign": [place(ordering_local, &adt_ord),
            {"Aggregate": [{"Adt": [1, ordering_index, null, generics]}, []]}]}}),
    ];
    if matches!(receiver, Receiver::ReassignedBase) {
        locals.push(json!({"index": extra_local, "name": "_b", "span": span, "ty": owner_ref}));
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(extra_local, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(extra_local, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
        );
    }
    if matches!(receiver, Receiver::ReborrowedOwner) {
        locals.push(json!({"index": extra_local, "name": "_t", "span": span, "ty": owner_ref}));
        let owner_deref = json!({
            "kind": {"Projection": [place(1, &owner_ref), "Deref"]},
            "ty": owner_adt
        });
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(extra_local, &owner_ref),
            {"Ref": {"place": owner_deref, "kind": "Shared", "ptr_metadata": null}}]}}),
        );
    }
    let recv2 = extra_local + 1;
    let owner2 = if extra_owner_param
        && !matches!(receiver, Receiver::CopiedThroughAnotherOwnerCopyInOneBranch)
    {
        extra_data_param
    } else {
        extra_local + 2
    };
    if matches!(
        receiver,
        Receiver::CopiedThroughOwnerCopy
            | Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
    ) && matches!(receiver, Receiver::CopiedThroughOwnerCopy)
    {
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(owner2, &owner_ref),
                {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
        );
    }
    if matches!(receiver, Receiver::CopiedFromCopiesOfAReassignedOwner) {
        let o = extra_local + 3;
        let owner_a = extra_local + 4;
        let owner_b = extra_local + 5;
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(o, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(owner_a, &owner_ref),
            {"Use": [{"Copy": place(o, &owner_ref)}, "Yes"]}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(o, &owner_ref),
            {"Use": [{"Copy": place(extra_data_param, &owner_ref)}, "Yes"]}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(owner_b, &owner_ref),
            {"Use": [{"Copy": place(o, &owner_ref)}, "Yes"]}]}}),
        );
    }
    if dynamic_index && !index_param {
        locals.push(json!({"index": extra_local, "name": "i", "span": span, "ty": usize_ty}));
        if matches!(
            receiver,
            Receiver::IndexZeroCapturedAtBorrow | Receiver::ReborrowKeepsIndexCapture
        ) {
            // Signed 0 lowers as `ConstInt(0)`.
            statements.push(
                json!({"span": span, "kind": {"Assign": [place(extra_local, &usize_ty),
                {"Use": [{"Const": [{"Integer": {"Signed": ["Isize", "0"]}},
                    {"Scalar": {"Integer": {"Signed": "Isize"}}}]}, "Yes"]}]}}),
            );
        } else {
            let first = if matches!(
                receiver,
                Receiver::IndexReassignedBeforeBorrow | Receiver::RepeatedIndexLocal
            ) {
                "0"
            } else {
                "1"
            };
            statements.push(json!({"span": span, "kind": {"Assign": [place(extra_local, &usize_ty),
                {"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", first]}}, usize_ty.clone()]}, "Yes"]}]}}));
            if matches!(receiver, Receiver::IndexReassignedBeforeBorrow) {
                statements.push(json!({"span": span, "kind": {"Assign": [place(extra_local, &usize_ty),
                    {"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", "1"]}}, usize_ty.clone()]}, "Yes"]}]}}));
            }
        }
    }
    if matches!(
        receiver,
        Receiver::BorrowThroughSubstructure | Receiver::MutBorrowedSubstructure
    ) {
        locals.push(
            json!({"index": extra_local, "name": "s", "span": span, "ty": inner_ref.clone()}),
        );
        let inner_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
                {"Field": [null, 0]}
            ]},
            "ty": inner_adt.clone()
        });
        let tag_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(extra_local, &inner_ref), "Deref"]}, "ty": inner_adt},
                {"Field": [null, 0]}
            ]},
            "ty": atomic
        });
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(extra_local, &inner_ref),
            {"Ref": {"place": inner_place, "kind": "Shared", "ptr_metadata": null}}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": tag_place, "kind": "Shared", "ptr_metadata": null}}]}}),
        );
    }
    if !matches!(
        receiver,
        Receiver::BorrowedInTwoBranches
            | Receiver::BorrowedTwoFieldsInTwoBranches
            | Receiver::ReborrowOfCallResult
            | Receiver::PinnedLocalsInTwoBranches
            | Receiver::BorrowThroughSubstructure
            | Receiver::RawReborrowOfCallResult
            | Receiver::ParamBaseLaterReassigned
            | Receiver::DeepSubstructure
            | Receiver::RepeatedIndexLocal
            | Receiver::MutBorrowedSubstructure
    ) {
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": field_place.clone(), "kind": "Shared", "ptr_metadata": null}}]}}),
        );
    }
    if matches!(
        receiver,
        Receiver::IndexReassignedAfterBorrow | Receiver::IndexZeroCapturedAtBorrow
    ) {
        let after = if matches!(receiver, Receiver::IndexZeroCapturedAtBorrow) {
            "1"
        } else {
            "0"
        };
        statements.push(json!({"span": span, "kind": {"Assign": [place(extra_local, &usize_ty),
            {"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", after]}}, usize_ty.clone()]}, "Yes"]}]}}));
    }
    if matches!(
        receiver,
        Receiver::CopiedFromTwoBorrows
            | Receiver::CopiedThroughOwnerCopy
            | Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
            | Receiver::CopiedFromTwoOwners
            | Receiver::CopiedFromTwoFields
            | Receiver::CopiedFromCopiesOfAReassignedOwner
            | Receiver::CopiedLastFromAnotherOwner
    ) {
        let recv2_place = if matches!(
            receiver,
            Receiver::CopiedThroughOwnerCopy
                | Receiver::CopiedThroughOwnerCopyInOneBranch
                | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
                | Receiver::CopiedFromTwoOwners
                | Receiver::CopiedLastFromAnotherOwner
        ) {
            json!({
                "kind": {"Projection": [
                    {"kind": {"Projection": [place(owner2, &owner_ref), "Deref"]}, "ty": owner_adt},
                    {"Field": [null, 0]}
                ]},
                "ty": atomic.clone()
            })
        } else if matches!(receiver, Receiver::CopiedFromCopiesOfAReassignedOwner) {
            json!({
                "kind": {"Projection": [
                    {"kind": {"Projection": [place(extra_local + 5, &owner_ref), "Deref"]}, "ty": owner_adt},
                    {"Field": [null, 0]}
                ]},
                "ty": atomic.clone()
            })
        } else if matches!(receiver, Receiver::CopiedFromTwoFields) {
            json!({
                "kind": {"Projection": [
                    {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
                    {"Field": [null, 1]}
                ]},
                "ty": atomic.clone()
            })
        } else {
            field_place.clone()
        };
        if !matches!(
            receiver,
            Receiver::CopiedThroughOwnerCopyInOneBranch
                | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
        ) {
            statements.push(
                json!({"span": span, "kind": {"Assign": [place(recv2, &recv_ty),
                {"Ref": {"place": recv2_place, "kind": "Shared", "ptr_metadata": null}}]}}),
            );
        }
    }
    if matches!(receiver, Receiver::ParamBaseLaterReassigned) {
        let tag_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(extra_data_param, &inner_ref), "Deref"]}, "ty": inner_adt},
                {"Field": [null, 0]}
            ]},
            "ty": atomic
        });
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": tag_place, "kind": "Shared", "ptr_metadata": null}}]}}),
        );
    }
    if matches!(receiver, Receiver::MutBorrowedSubstructure) {
        let mut_ty = json!({"Ref": ["Erased", inner_ref, "Mut"]});
        locals.push(json!({"index": extra_local + 1, "name": "_m", "span": span, "ty": mut_ty}));
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(extra_local + 1, &mut_ty),
            {"Ref": {"place": place(extra_local, &inner_ref), "kind": "Mut", "ptr_metadata": null}}]}}),
        );
    }
    if matches!(receiver, Receiver::ReborrowKeepsIndexCapture) {
        let q = extra_local + 1;
        locals.push(json!({"index": q, "name": "q", "span": span, "ty": recv_ty}));
        let reborrow = json!({
            "kind": {"Projection": [place(recv_bind, &recv_ty), "Deref"]},
            "ty": atomic
        });
        statements.push(json!({"span": span, "kind": {"Assign": [place(q, &recv_ty),
            {"Ref": {"place": reborrow, "kind": "Shared", "ptr_metadata": null}}]}}));
        statements.push(json!({"span": span, "kind": {"Assign": [place(extra_local, &usize_ty),
            {"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", "1"]}}, usize_ty.clone()]}, "Yes"]}]}}));
    }
    if matches!(receiver, Receiver::RepeatedIndexLocal) {
        let s = extra_local + 1;
        let row_adt = json!({"Adt": {"id": 3, "generics": generics, "builtin": null}});
        let row_ref = json!({"Ref": ["Erased", row_adt, "Shared"]});
        locals.push(json!({"index": s, "name": "s", "span": span, "ty": row_ref}));
        let rows_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [
                    {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
                    {"Field": [null, 0]}
                ]}, "ty": json!({"Array": [row_adt.clone(), array_len.clone(), null]})},
                {"Index": {"offset": {"Copy": place(extra_local, &usize_ty)}, "from_end": false}}
            ]},
            "ty": row_adt
        });
        statements.push(json!({"span": span, "kind": {"Assign": [place(s, &row_ref),
            {"Ref": {"place": rows_place, "kind": "Shared", "ptr_metadata": null}}]}}));
        statements.push(json!({"span": span, "kind": {"Assign": [place(extra_local, &usize_ty),
            {"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", "1"]}}, usize_ty.clone()]}, "Yes"]}]}}));
        let cells_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [
                    {"kind": {"Projection": [place(s, &row_ref), "Deref"]}, "ty": row_adt},
                    {"Field": [null, 0]}
                ]}, "ty": array_ty.clone()},
                {"Index": {"offset": {"Copy": place(extra_local, &usize_ty)}, "from_end": false}}
            ]},
            "ty": atomic
        });
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": cells_place, "kind": "Shared", "ptr_metadata": null}}]}}),
        );
    }
    if matches!(receiver, Receiver::DeepSubstructure) {
        let mut prev_local = 1usize;
        let mut prev_ref = owner_ref.clone();
        let mut prev_adt = owner_adt.clone();
        for level in 0..9 {
            let s = extra_local + level;
            let inner_id = 3 + level;
            let next_adt = json!({"Adt": {"id": inner_id, "generics": generics, "builtin": null}});
            let next_ref = json!({"Ref": ["Erased", next_adt, "Shared"]});
            locals.push(
                json!({"index": s, "name": format!("s{level}"), "span": span, "ty": next_ref}),
            );
            let inner_place = json!({
                "kind": {"Projection": [
                    {"kind": {"Projection": [place(prev_local, &prev_ref), "Deref"]}, "ty": prev_adt},
                    {"Field": [null, 0]}
                ]},
                "ty": next_adt
            });
            statements.push(
                json!({"span": span, "kind": {"Assign": [place(s, &next_ref),
                {"Ref": {"place": inner_place, "kind": "Shared", "ptr_metadata": null}}]}}),
            );
            prev_local = s;
            prev_ref = next_ref;
            prev_adt = next_adt;
        }
        let tag_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(prev_local, &prev_ref), "Deref"]}, "ty": prev_adt},
                {"Field": [null, 0]}
            ]},
            "ty": atomic
        });
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": tag_place, "kind": "Shared", "ptr_metadata": null}}]}}),
        );
    }
    if matches!(receiver, Receiver::MutBorrowedOwner) {
        let mut_ty = json!({"Ref": ["Erased", owner_ref, "Mut"]});
        locals.push(json!({"index": extra_local, "name": "_m", "span": span, "ty": mut_ty}));
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(extra_local, &mut_ty),
            {"Ref": {"place": place(1, &owner_ref), "kind": "Mut", "ptr_metadata": null}}]}}),
        );
    }
    if matches!(
        receiver,
        Receiver::ReassignedParam | Receiver::OwnerReassignedBeforeCopy
    ) {
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(1, &owner_ref),
            {"Use": [{"Copy": place(2, &owner_ref)}, "Yes"]}]}}),
        );
    }
    let recv_local = if matches!(
        receiver,
        Receiver::Copied
            | Receiver::CopiedTwice
            | Receiver::CopiedChain
            | Receiver::CopiedFromTwoBorrows
            | Receiver::CopiedThroughOwnerCopy
            | Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
            | Receiver::CopiedFromTwoOwners
            | Receiver::CopiedFromTwoFields
            | Receiver::CopiedFromCopiesOfAReassignedOwner
            | Receiver::CopiedThroughDiamond
            | Receiver::CopiedThenOverwritten
            | Receiver::OverwrittenByCall
            | Receiver::OwnerReassignedBeforeCopy
            | Receiver::CopiedThenOwnerReassigned
            | Receiver::CopiedLastFromAnotherOwner
    ) {
        locals.push(json!({"index": extra_local, "name": "alias", "span": span, "ty": recv_ty}));
        if matches!(
            receiver,
            Receiver::CopiedFromTwoBorrows
                | Receiver::CopiedThroughOwnerCopy
                | Receiver::CopiedThroughOwnerCopyInOneBranch
                | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
                | Receiver::CopiedFromTwoOwners
                | Receiver::CopiedFromTwoFields
                | Receiver::CopiedFromCopiesOfAReassignedOwner
                | Receiver::CopiedLastFromAnotherOwner
        ) {
            locals.push(json!({"index": recv2, "name": "recv2", "span": span, "ty": recv_ty}));
        }
        if matches!(
            receiver,
            Receiver::CopiedThroughOwnerCopy
                | Receiver::CopiedThroughOwnerCopyInOneBranch
                | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
        ) {
            locals.push(json!({"index": owner2, "name": "owner2", "span": span, "ty": owner_ref}));
        }
        if matches!(receiver, Receiver::CopiedThroughDiamond) {
            let x = extra_local + 1;
            let y = extra_local + 2;
            locals.push(json!({"index": x, "name": "x", "span": span, "ty": recv_ty}));
            locals.push(json!({"index": y, "name": "y", "span": span, "ty": recv_ty}));
            statements.push(json!({"span": span, "kind": {"Assign": [place(x, &recv_ty),
                {"Use": [{"Copy": place(recv_bind, &recv_ty)}, "Yes"]}]}}));
            statements.push(json!({"span": span, "kind": {"Assign": [place(y, &recv_ty),
                {"Use": [{"Copy": place(x, &recv_ty)}, "Yes"]}]}}));
            extra_local
        } else {
            if !merge {
                statements.push(
                    json!({"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                    {"Use": [{"Copy": place(recv_bind, &recv_ty)}, "Yes"]}]}}),
                );
                if matches!(receiver, Receiver::CopiedTwice) {
                    statements.push(
                        json!({"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                        {"Use": [{"Copy": place(recv_bind, &recv_ty)}, "Yes"]}]}}),
                    );
                }
                if matches!(
                    receiver,
                    Receiver::CopiedFromTwoBorrows
                        | Receiver::CopiedThroughOwnerCopy
                        | Receiver::CopiedThroughOwnerCopyInOneBranch
                        | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
                        | Receiver::CopiedFromTwoOwners
                        | Receiver::CopiedFromCopiesOfAReassignedOwner
                        | Receiver::CopiedLastFromAnotherOwner
                ) {
                    statements.push(
                        json!({"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                        {"Use": [{"Copy": place(recv2, &recv_ty)}, "Yes"]}]}}),
                    );
                }
                if matches!(receiver, Receiver::CopiedThenOverwritten) {
                    statements.push(
                        json!({"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                        {"Use": [{"Copy": place(arg_count, &recv_ty)}, "Yes"]}]}}),
                    );
                }
            }
            if matches!(receiver, Receiver::CopiedChain) {
                let alias2 = extra_local + 1;
                locals
                    .push(json!({"index": alias2, "name": "alias2", "span": span, "ty": recv_ty}));
                statements.push(
                    json!({"span": span, "kind": {"Assign": [place(alias2, &recv_ty),
                    {"Use": [{"Copy": place(extra_local, &recv_ty)}, "Yes"]}]}}),
                );
                alias2
            } else {
                extra_local
            }
        }
    } else if matches!(receiver, Receiver::ReborrowKeepsIndexCapture) {
        extra_local + 1
    } else if matches!(receiver, Receiver::ReborrowOfCallResult) {
        locals.push(json!({"index": extra_local, "name": "_3", "span": span, "ty": recv_ty}));
        recv_bind
    } else if param_copied_after_call {
        arg_count
    } else {
        recv_bind
    };
    if matches!(receiver, Receiver::CopiedThenOwnerReassigned) {
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(1, &owner_ref),
            {"Use": [{"Copy": place(2, &owner_ref)}, "Yes"]}]}}),
        );
    }
    if matches!(receiver, Receiver::CopiedFromCopiesOfAReassignedOwner) {
        // Place ids are vec indices; pad extra_local+2 so o / owner_a /
        // owner_b land at extra_local+3/+4/+5 after alias and recv2.
        locals.push(json!({"index": extra_local + 2, "name": null, "span": span, "ty": owner_ref}));
        locals.push(json!({"index": extra_local + 3, "name": "o", "span": span, "ty": owner_ref}));
        locals.push(
            json!({"index": extra_local + 4, "name": "owner_a", "span": span, "ty": owner_ref}),
        );
        locals.push(
            json!({"index": extra_local + 5, "name": "owner_b", "span": span, "ty": owner_ref}),
        );
    }
    let keep = if matches!(
        receiver,
        Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
    ) {
        let keep = locals.len();
        locals.push(json!({"index": keep, "name": "keep", "span": span, "ty": owner_ref}));
        Some(keep)
    } else {
        None
    };
    let mut inputs = vec![owner_param_ty.clone()];
    if other_param {
        inputs.push(owner_ref.clone());
    }
    if is_store {
        inputs.push(word.clone());
    }
    if extra_recv_param {
        inputs.push(recv_ty.clone());
    }
    if extra_owner_param {
        inputs.push(owner_param_ty.clone());
    }
    if index_param {
        inputs.push(usize_ty.clone());
    }
    if inner_param {
        inputs.push(inner_ref.clone());
    }
    if merge {
        inputs.push(bool_ty.clone());
    }
    let call_args = if is_store {
        json!([
            {"Copy": place(recv_local, &recv_ty)},
            {"Copy": place(value_local, &word)},
            {"Copy": place(ordering_local, &adt_ord)}
        ])
    } else {
        json!([
            {"Copy": place(recv_local, &recv_ty)},
            {"Copy": place(ordering_local, &adt_ord)}
        ])
    };
    let (atomic_method, atomic_inputs, atomic_output) = if is_store {
        (
            "store",
            json!([recv_ty, word, adt_ord]),
            json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": generics}}),
        )
    } else {
        ("load", json!([recv_ty, adt_ord]), word.clone())
    };
    let return_statements = if param_copied_after_call {
        json!([{"span": span, "kind": {"Assign": [place(arg_count, &recv_ty),
            {"Use": [{"Copy": place(recv_bind, &recv_ty)}, "Yes"]}]}}])
    } else if matches!(receiver, Receiver::OwnerReassignedAfterCall) {
        json!([{"span": span, "kind": {"Assign": [place(1, &owner_ref),
            {"Use": [{"Copy": place(2, &owner_ref)}, "Yes"]}]}}])
    } else if index_param {
        json!([{"span": span, "kind": {"Assign": [place(extra_data_param, &usize_ty),
            {"Use": [{"Const": [{"Integer": {"Unsigned": ["Usize", "2"]}}, usize_ty.clone()]}, "Yes"]}]}}])
    } else if inner_param {
        let inner_place = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
                {"Field": [null, 0]}
            ]},
            "ty": inner_adt.clone()
        });
        json!([{"span": span, "kind": {"Assign": [place(extra_data_param, &inner_ref),
            {"Ref": {"place": inner_place, "kind": "Shared", "ptr_metadata": null}}]}}])
    } else if let Some(keep) = keep {
        json!([{"span": span, "kind": {"Assign": [place(keep, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}])
    } else {
        json!([])
    };
    let overwritten_by_call = matches!(receiver, Receiver::OverwrittenByCall);
    let reborrow_of_call_result = matches!(receiver, Receiver::ReborrowOfCallResult);
    let raw_reborrow_of_call_result = matches!(receiver, Receiver::RawReborrowOfCallResult);
    if raw_reborrow_of_call_result {
        let raw_ty = json!({"RawPtr": [atomic.clone(), "Const"]});
        locals.push(json!({"index": extra_local, "name": "p", "span": span, "ty": recv_ty}));
        locals.push(json!({"index": extra_local + 1, "name": "q", "span": span, "ty": raw_ty}));
    }
    let body_blocks = if overwritten_by_call {
        json!([
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                    "args": [],
                    "dest": place(recv_local, &recv_ty)}, "target": 1, "on_unwind": 3
            }}}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args,
                    "dest": place(0, &dest_ty)}, "target": 2, "on_unwind": 3
            }}}},
            {"statements": return_statements, "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    } else if reborrow_of_call_result {
        let call_deref = json!({
            "kind": {"Projection": [place(extra_local, &recv_ty), "Deref"]},
            "ty": atomic
        });
        json!([
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                    "args": [],
                    "dest": place(extra_local, &recv_ty)}, "target": 1, "on_unwind": 3
            }}}},
            {"statements": [
                {"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
                    {"Ref": {"place": call_deref, "kind": "Shared", "ptr_metadata": null}}]}}
            ], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args,
                    "dest": place(0, &dest_ty)}, "target": 2, "on_unwind": 3
            }}}},
            {"statements": return_statements, "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    } else if raw_reborrow_of_call_result {
        let raw_ty = json!({"RawPtr": [atomic.clone(), "Const"]});
        let call_deref = json!({
            "kind": {"Projection": [place(extra_local, &recv_ty), "Deref"]},
            "ty": atomic
        });
        json!([
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                    "args": [{"Copy": place(1, &owner_ref)}],
                    "dest": place(extra_local, &recv_ty)}, "target": 1, "on_unwind": 4
            }}}},
            {"statements": [
                {"span": span, "kind": {"Assign": [place(extra_local + 1, &raw_ty),
                    {"RawPtr": {"place": call_deref, "kind": "Const", "ptr_metadata": null}}]}}
            ], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 3}, "generics": generics}},
                    "args": [{"Copy": place(extra_local + 1, &raw_ty)}],
                    "dest": place(recv_bind, &recv_ty)}, "target": 2, "on_unwind": 4
            }}}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args,
                    "dest": place(0, &dest_ty)}, "target": 3, "on_unwind": 4
            }}}},
            {"statements": return_statements, "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    } else if merge {
        let (branch_a, branch_b) = if matches!(receiver, Receiver::CopiedThroughDiamond) {
            let x = extra_local + 1;
            let y = extra_local + 2;
            (
                json!([{"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                    {"Use": [{"Copy": place(x, &recv_ty)}, "Yes"]}]}}]),
                json!([{"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                    {"Use": [{"Copy": place(y, &recv_ty)}, "Yes"]}]}}]),
            )
        } else if matches!(
            receiver,
            Receiver::CopiedThroughOwnerCopyInOneBranch
                | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
        ) {
            let recv2_place = json!({
                "kind": {"Projection": [
                    {"kind": {"Projection": [place(owner2, &owner_ref), "Deref"]}, "ty": owner_adt},
                    {"Field": [null, 0]}
                ]},
                "ty": atomic.clone()
            });
            let owner2_src =
                if matches!(receiver, Receiver::CopiedThroughAnotherOwnerCopyInOneBranch) {
                    extra_data_param
                } else {
                    1
                };
            (
                json!([{"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                    {"Use": [{"Copy": place(recv_bind, &recv_ty)}, "Yes"]}]}}]),
                json!([
                    {"span": span, "kind": {"Assign": [place(owner2, &owner_ref),
                        {"Use": [{"Copy": place(owner2_src, &owner_ref)}, "Yes"]}]}},
                    {"span": span, "kind": {"Assign": [place(recv2, &recv_ty),
                        {"Ref": {"place": recv2_place, "kind": "Shared", "ptr_metadata": null}}]}},
                    {"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                        {"Use": [{"Copy": place(recv2, &recv_ty)}, "Yes"]}]}}
                ]),
            )
        } else if matches!(receiver, Receiver::PinnedLocalsInTwoBranches) {
            let place_b = json!({
                "kind": {"Projection": [place(owner2, &owner_adt), {"Field": [null, 0]}]},
                "ty": atomic.clone()
            });
            (
                json!([{"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
                    {"Ref": {"place": field_place.clone(), "kind": "Shared", "ptr_metadata": null}}]}}]),
                json!([{"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
                    {"Ref": {"place": place_b, "kind": "Shared", "ptr_metadata": null}}]}}]),
            )
        } else if matches!(
            receiver,
            Receiver::BorrowedInTwoBranches | Receiver::BorrowedTwoFieldsInTwoBranches
        ) {
            let place_b = if matches!(receiver, Receiver::BorrowedTwoFieldsInTwoBranches) {
                json!({
                    "kind": {"Projection": [
                        {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
                        {"Field": [null, 1]}
                    ]},
                    "ty": atomic.clone()
                })
            } else {
                json!({
                    "kind": {"Projection": [
                        {"kind": {"Projection": [place(owner2, &owner_ref), "Deref"]}, "ty": owner_adt},
                        {"Field": [null, 0]}
                    ]},
                    "ty": atomic.clone()
                })
            };
            (
                json!([{"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
                    {"Ref": {"place": field_place.clone(), "kind": "Shared", "ptr_metadata": null}}]}}]),
                json!([{"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
                    {"Ref": {"place": place_b, "kind": "Shared", "ptr_metadata": null}}]}}]),
            )
        } else {
            (
                json!([{"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                    {"Use": [{"Copy": place(recv_bind, &recv_ty)}, "Yes"]}]}}]),
                json!([{"span": span, "kind": {"Assign": [place(extra_local, &recv_ty),
                    {"Use": [{"Copy": place(recv2, &recv_ty)}, "Yes"]}]}}]),
            )
        };
        json!([
            {"statements": statements, "kind": "Regular", "terminator": {"span": span, "kind": {
                "Switch": {"discr": {"Copy": place(arg_count, &bool_ty)}, "targets": {"If": [1, 2]}}
            }}},
            {"statements": branch_a, "kind": "Regular", "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
            {"statements": branch_b, "kind": "Regular", "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args,
                    "dest": place(0, &dest_ty)}, "target": 4, "on_unwind": 5
            }}}},
            {"statements": return_statements, "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    } else {
        json!([
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args,
                    "dest": place(0, &dest_ty)}, "target": 1, "on_unwind": 2
            }}}},
            {"statements": return_statements, "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
        ])
    };
    let mut fun_decls = vec![
        json!({"def_id": 0, "item_meta": meta(&["fixture", "read_field"]),
        "signature": {"is_unsafe": false,
           "inputs": inputs,
           "output": dest_ty},
        "body": {"Unstructured": {"span": span,
           "locals": {"arg_count": arg_count, "locals": locals},
           "body": body_blocks
        }}}),
        json!({"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", atomic_method]),
             "signature": {"is_unsafe": false, "inputs": atomic_inputs, "output": atomic_output}, "body": "Opaque"}),
    ];
    if overwritten_by_call || reborrow_of_call_result || raw_reborrow_of_call_result {
        let get_ref_inputs = if raw_reborrow_of_call_result {
            json!([owner_ref.clone()])
        } else {
            json!([])
        };
        fun_decls.push(json!({
            "def_id": 2, "item_meta": meta(&["fixture", "get_ref"]),
            "signature": {"is_unsafe": false, "inputs": get_ref_inputs, "output": recv_ty}, "body": "Opaque"
        }));
    }
    if raw_reborrow_of_call_result {
        let raw_ty = json!({"RawPtr": [atomic.clone(), "Const"]});
        fun_decls.push(json!({
            "def_id": 3, "item_meta": meta(&["core", "sync", "atomic", "<Impl>", "from_ptr"]),
            "signature": {"is_unsafe": false, "inputs": [raw_ty], "output": recv_ty}, "body": "Opaque"
        }));
    }
    let row_adt = json!({"Adt": {"id": 3, "generics": generics, "builtin": null}});
    let rows_array_ty = json!({"Array": [row_adt.clone(), array_len.clone(), null]});
    let struct_fields = if matches!(receiver, Receiver::DeepSubstructure) {
        let inner1 = json!({"Adt": {"id": 3, "generics": generics, "builtin": null}});
        vec![json!({"name": "inner", "ty": inner1, "attr_info": attr.clone()})]
    } else if matches!(receiver, Receiver::RepeatedIndexLocal) {
        vec![json!({"name": "rows", "ty": rows_array_ty, "attr_info": attr.clone()})]
    } else if borrow_through_substructure {
        vec![json!({"name": "inner", "ty": owner_field_ty, "attr_info": attr.clone()})]
    } else if matches!(
        receiver,
        Receiver::CopiedFromTwoFields | Receiver::BorrowedTwoFieldsInTwoBranches
    ) {
        vec![
            json!({"name": field, "ty": owner_field_ty, "attr_info": attr.clone()}),
            json!({"name": "other_tag", "ty": owner_field_ty, "attr_info": attr.clone()}),
        ]
    } else {
        vec![json!({"name": field, "ty": owner_field_ty, "attr_info": attr.clone()})]
    };
    let mut type_decls = vec![
        json!({"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"}),
        json!({"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
             "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                "name": name, "fields": [],
                "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
             })).collect::<Vec<_>>()}}),
        json!({"def_id": 2, "item_meta": meta(&["fixture", owner]),
             "kind": {"Struct": struct_fields}}),
    ];
    if borrow_through_substructure {
        type_decls.push(json!({
            "def_id": 3, "item_meta": meta(&["fixture", "Inner"]),
            "kind": {"Struct": [{"name": field, "ty": atomic, "attr_info": attr}]}
        }));
    }
    if matches!(receiver, Receiver::RepeatedIndexLocal) {
        type_decls.push(json!({
            "def_id": 3, "item_meta": meta(&["fixture", "Row"]),
            "kind": {"Struct": [{"name": "cells", "ty": array_ty, "attr_info": attr}]}
        }));
    }
    if matches!(receiver, Receiver::DeepSubstructure) {
        for level in 0..9 {
            let id = 3 + level;
            let field_ty = if level == 8 {
                atomic.clone()
            } else {
                json!({"Adt": {"id": id + 1, "generics": generics, "builtin": null}})
            };
            let field_name = if level == 8 { field } else { "inner" };
            type_decls.push(json!({
                "def_id": id,
                "item_meta": meta(&["fixture", &format!("Inner{level}")]),
                "kind": {"Struct": [{"name": field_name, "ty": field_ty, "attr_info": attr}]}
            }));
        }
    }
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": fun_decls, "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field fixture parses")
}

#[test]
fn atomic_field_access_lowers_on_the_parent_captured_at_the_borrow() {
    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::Borrow,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    )
    .expect("Acquire load of quasi AtomicU64 field lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "lowered quasi AtomicU64 field has no version_tag field read"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::Borrow,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load of a non-quasi AtomicU64 field must stay unsupported, got {result:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::Copied,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    )
    .expect("Acquire load through a copied receiver lowers through the recorded borrow");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Acquire load through a copied receiver has no version_tag field read"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedBase,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    )
    .expect("Acquire load through a borrow base assigned twice before the borrow lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Acquire load through a borrow base assigned twice before the borrow has no version_tag field read"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    )
    .expect("Acquire load through a reassigned owner parameter lowers on the captured parent");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Acquire load through a reassigned owner parameter must read version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Acquire load through a reassigned owner parameter must not read version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::Borrow,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load of a stable AtomicU64 field borrow lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "hot_counter"
            )
        })
    });
    assert!(
        saw,
        "lowered Relaxed load of a stable borrow has no hot_counter field read"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a reassigned owner parameter lowers on the captured parent");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed load through a reassigned owner parameter must read version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Relaxed load through a reassigned owner parameter must not read version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::Copied,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a copied receiver lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through a copied receiver has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through a copied receiver must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedChain,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a chain of copied receivers lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through a chain of copied receivers has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through a chain of copied receivers must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughDiamond,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through copies that converge on one local lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through copies that converge on one local has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through copies that converge on one local must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedTwice,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a receiver copied twice from one borrow lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through a receiver copied twice from one borrow has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through a receiver copied twice from one borrow must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::Borrow,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store of a stable AtomicU64 field borrow lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldWrite { field, .. }
                    if field.name == "hot_counter"
            )
        })
    });
    assert!(
        saw,
        "lowered Relaxed store of a stable borrow has no hot_counter field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a reassigned owner parameter lowers on the captured parent");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed store through a reassigned owner parameter must write version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Relaxed store through a reassigned owner parameter must not write version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::Copied,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a copied receiver lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldWrite { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed store through a copied receiver has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedChain,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a chain of copied receivers lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldWrite { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed store through a chain of copied receivers has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughDiamond,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through copies that converge on one local lowers");
    assert!(
        graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through copies that converge on one local has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedTwice,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a receiver copied twice from one borrow lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldWrite { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed store through a receiver copied twice from one borrow has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoBorrows,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a receiver copied from two borrows of one place lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through a receiver copied from two borrows of one place has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through a receiver copied from two borrows of one place must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoBorrows,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a receiver copied from two borrows of one place lowers");
    assert!(
        graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through a receiver copied from two borrows of one place has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughOwnerCopy,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a receiver copied from borrows through a copied owner lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through a receiver copied from borrows through a copied owner has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through a receiver copied from borrows through a copied owner must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughOwnerCopy,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a receiver copied from borrows through a copied owner lowers");
    assert!(
        graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through a receiver copied from borrows through a copied owner has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughOwnerCopyInOneBranch,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a copy of a borrow whose owner copy one branch alone assigns lowers through the other borrow");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through a copy of a borrow whose owner copy one branch alone assigns has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through a copy of a borrow whose owner copy one branch alone assigns must not raw_load the field value"
    );
    let (block_id, base) = graph
        .blocks
        .iter()
        .find_map(|block| {
            block.operations.iter().find_map(|op| match &op.kind {
                OpKind::FieldRead { field, base, .. } if field.name == "version_tag" => {
                    Some((block.id, base.clone()))
                }
                _ => None,
            })
        })
        .expect("Relaxed load through a copy of a borrow whose owner copy one branch alone assigns has a version_tag FieldRead");
    let owner = &graph.block(graph.startblock).inputargs[0];
    assert!(
        traces_to_start_input(&graph, block_id, &base, owner),
        "Relaxed load through a copy of a borrow whose owner copy one branch alone assigns must read version_tag of owner, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughOwnerCopyInOneBranch,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a copy of a borrow whose owner copy one branch alone assigns lowers through the other borrow");
    assert!(
        graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through a copy of a borrow whose owner copy one branch alone assigns has no version_tag field write"
    );
    let (block_id, base) = graph
        .blocks
        .iter()
        .find_map(|block| {
            block.operations.iter().find_map(|op| match &op.kind {
                OpKind::FieldWrite { field, base, .. } if field.name == "version_tag" => {
                    Some((block.id, base.clone()))
                }
                _ => None,
            })
        })
        .expect("Relaxed store through a copy of a borrow whose owner copy one branch alone assigns has a version_tag FieldWrite");
    let owner = &graph.block(graph.startblock).inputargs[0];
    assert!(
        traces_to_start_input(&graph, block_id, &base, owner),
        "Relaxed store through a copy of a borrow whose owner copy one branch alone assigns must write version_tag of owner, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughAnotherOwnerCopyInOneBranch,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a copy of borrows of two owners, one borrowed in one branch alone, lowers");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    let extra_owner = merge_extra_owner(start);
    assert!(
        reached.iter().any(|v| v == &start[0]) && reached.iter().any(|v| v == extra_owner),
        "Relaxed load through a copy of borrows of two owners, one borrowed in one branch alone, must read version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughAnotherOwnerCopyInOneBranch,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a copy of borrows of two owners, one borrowed in one branch alone, lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed store through a copy of borrows of two owners, one borrowed in one branch alone, must write version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoOwners,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a receiver copied from borrows of two owners lowers");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed load through a receiver copied from borrows of two owners must read version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoOwners,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a receiver copied from borrows of two owners lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed store through a receiver copied from borrows of two owners must write version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::BorrowedInTwoBranches,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a receiver borrowed in two branches lowers");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed load through a receiver borrowed in two branches must read version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::BorrowedInTwoBranches,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    )
    .expect("Acquire load through a receiver borrowed in two branches lowers");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Acquire load through a receiver borrowed in two branches must read version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::BorrowedInTwoBranches,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a receiver borrowed in two branches lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed store through a receiver borrowed in two branches must write version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::BorrowedInTwoBranches,
            AtomicOp::ReleaseStore,
        ),
        "read_field",
    )
    .expect("Release store through a receiver borrowed in two branches lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Release store through a receiver borrowed in two branches must write version_tag of both owners, got {reached:?} inputs {start:?}"
    );

    for op in [
        AtomicOp::Load("Acquire"),
        AtomicOp::Load("Relaxed"),
        AtomicOp::RelaxedStore,
        AtomicOp::ReleaseStore,
    ] {
        let write = matches!(op, AtomicOp::RelaxedStore | AtomicOp::ReleaseStore);
        let graph = lower_function(
            &quasi_atomic_u64_field_fixture(
                "Owner",
                "version_tag",
                Some("version_tag?"),
                Receiver::ReborrowedOwner,
                op,
            ),
            "read_field",
        )
        .unwrap_or_else(|err| panic!("reborrowed owner {op:?} must lower: {err:?}"));
        let (block_id, base) = version_tag_field_base(&graph, write);
        let owner = &graph.block(graph.startblock).inputargs[0];
        assert!(
            traces_to_start_input(&graph, block_id, &base, owner),
            "reborrowed owner {op:?} must access version_tag of owner, got base {base:?}"
        );
    }

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromCopiesOfAReassignedOwner,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a receiver copied from borrows of copies of a reassigned owner lowers");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed load through a receiver copied from borrows of copies of a reassigned owner must read version_tag of owner and p, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromCopiesOfAReassignedOwner,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a receiver copied from borrows of copies of a reassigned owner lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    let reached = start_inputs_reached(&graph, block_id, &base);
    assert!(
        reached.iter().any(|v| v == &start[0])
            && reached.iter().any(|v| v == merge_extra_owner(start)),
        "Relaxed store through a receiver copied from borrows of copies of a reassigned owner must write version_tag of owner and p, got {reached:?} inputs {start:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OwnerReassignedAfterCall,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through an owner reassigned after the call lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed load through an owner reassigned after the call has no version_tag field read"
    );
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        !saw_raw,
        "Relaxed load through an owner reassigned after the call must not raw_load the field value"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OwnerReassignedAfterCall,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through an owner reassigned after the call lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldWrite { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "Relaxed store through an owner reassigned after the call has no version_tag field write"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OwnerReassignedBeforeCopy,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect(
        "Relaxed load through a copy of a borrow whose owner is reassigned before the copy lowers",
    );
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed load through a copy of a borrow whose owner is reassigned before the copy must read version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Relaxed load through a copy of a borrow whose owner is reassigned before the copy must not read version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OwnerReassignedBeforeCopy,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect(
        "Relaxed store through a copy of a borrow whose owner is reassigned before the copy lowers",
    );
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed store through a copy of a borrow whose owner is reassigned before the copy must write version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Relaxed store through a copy of a borrow whose owner is reassigned before the copy must not write version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThenOwnerReassigned,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect(
        "Relaxed load through a copy of a borrow whose owner is reassigned after the copy lowers",
    );
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed load through a copy of a borrow whose owner is reassigned after the copy must read version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Relaxed load through a copy of a borrow whose owner is reassigned after the copy must not read version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThenOwnerReassigned,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect(
        "Relaxed store through a copy of a borrow whose owner is reassigned after the copy lowers",
    );
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed store through a copy of a borrow whose owner is reassigned after the copy must write version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Relaxed store through a copy of a borrow whose owner is reassigned after the copy must not write version_tag of other, got base {base:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedLastFromAnotherOwner,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through the last copy reaching the call lowers");
    let start_inputs = &graph.block(graph.startblock).inputargs;
    assert!(
        start_inputs.len() >= 2,
        "start block should have owner then owner2 inputs, got {start_inputs:?}"
    );
    let owner = &start_inputs[0];
    let owner2 = start_inputs
        .last()
        .expect("owner2 is the last start-block parameter");
    assert_ne!(
        owner, owner2,
        "owner and owner2 must be distinct start-block inputs, got {start_inputs:?}"
    );
    let return_var = graph
        .blocks
        .iter()
        .find_map(|block| {
            block.exits.iter().find_map(|link| {
                if link.target != graph.returnblock {
                    return None;
                }
                match link.args.as_slice() {
                    [LinkArg::Value(var)] => Some(var.clone()),
                    _ => None,
                }
            })
        })
        .expect("read_field returns a value");
    let load_base = graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| match &op.kind {
            majit_translate::model::OpKind::FieldRead { field, base, .. }
                if field.name == "version_tag" && op.result.as_ref() == Some(&return_var) =>
            {
                Some(base.clone())
            }
            _ => None,
        })
    });
    assert_eq!(
        load_base.as_ref(),
        Some(owner2),
        "Relaxed load through the last copy reaching the call must read version_tag of owner2, got {load_base:?} inputs {start_inputs:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedLastFromAnotherOwner,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through the last copy reaching the call lowers");
    let start_inputs = &graph.block(graph.startblock).inputargs;
    assert!(
        start_inputs.len() >= 2,
        "store start block should have owner, value, then owner2 inputs, got {start_inputs:?}"
    );
    let owner = &start_inputs[0];
    let owner2 = start_inputs
        .last()
        .expect("owner2 is the last start-block parameter");
    assert_ne!(
        owner, owner2,
        "store owner and owner2 must be distinct start-block inputs, got {start_inputs:?}"
    );
    let store_base = graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| match &op.kind {
            majit_translate::model::OpKind::FieldWrite { field, base, .. }
                if field.name == "version_tag" =>
            {
                Some(base.clone())
            }
            _ => None,
        })
    });
    assert_eq!(
        store_base.as_ref(),
        Some(owner2),
        "Relaxed store through the last copy reaching the call must write version_tag of owner2, got {store_base:?} inputs {start_inputs:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ParamCopiedAfterCall,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a parameter copied into after the call lowers");
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        saw_raw,
        "Relaxed load through a parameter copied into after the call reads the incoming address"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ParamCopiedAfterCall,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a parameter copied into after the call lowers");
    assert!(
        !graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through a parameter copied into after the call must not write the recorded borrow's field"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThenOverwritten,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a copy overwritten by a parameter copy lowers");
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        saw_raw,
        "Relaxed load through a copy overwritten by a parameter copy reads the incoming address"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThenOverwritten,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a copy overwritten by a parameter copy lowers");
    assert!(
        !graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through a copy overwritten by a parameter copy must not write the recorded borrow's field"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OverwrittenByCall,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a copy overwritten by a call result lowers");
    let saw_raw = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, majit_translate::model::OpKind::RawLoad { .. }))
    });
    assert!(
        saw_raw,
        "Relaxed load through a copy overwritten by a call result reads the returned address"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OverwrittenByCall,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store through a copy overwritten by a call result lowers");
    assert!(
        !graph_has_field_write(&graph, "version_tag"),
        "Relaxed store through a copy overwritten by a call result must not write the recorded borrow's field"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReborrowOfCallResult,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load of a reborrow of a call result lowers");
    assert!(
        !graph_has_raw_load(&graph),
        "Relaxed load of a reborrow of a call result must not raw_load the value"
    );
    let reborrow_ret =
        graph_return_var(&graph).expect("reborrow-of-call-result load returns a value");
    assert!(
        traces_to_call_named(&graph, &reborrow_ret, "get_ref"),
        "Relaxed load of a reborrow of a call result must alias that call"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::MutBorrowedOwner,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a mutably borrowed owner must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::SuffixIndex,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("suffix index read from an array with no length")),
        "Relaxed load through a suffix index must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoFields,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through borrows of two fields of one owner must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoFields,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through borrows of two fields of one owner must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::BorrowedTwoFieldsInTwoBranches,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through two fields borrowed in two branches must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::BorrowedTwoFieldsInTwoBranches,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through two fields borrowed in two branches must stay unsupported, got {result:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::PinnedLocalField,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load of a pinned local field lowers");
    let (block_id, base) = version_tag_field_base(&graph, false);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed load of a pinned local field must read version_tag of the local owner"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::PinnedLocalField,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    )
    .expect("Relaxed store of a pinned local field lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed store of a pinned local field must write version_tag of the local owner"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::PinnedLocalsInTwoBranches,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through two pinned locals borrowed in two branches must stay unsupported, got {result:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tag",
            None,
            Receiver::BorrowThroughSubstructure,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a borrowed substructure lowers");
    let (block_id, base) = outermost_field_base(&graph, "tag", false);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Relaxed load through a borrowed substructure must read tag of start input owner"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tags",
            None,
            Receiver::IndexReassignedBeforeBorrow,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through an index reassigned before the borrow lowers");
    assert_eq!(
        array_read_index_const(&graph),
        Some(1),
        "Relaxed load through an index reassigned before the borrow must use the captured index 1"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tags",
            None,
            Receiver::IndexReassignedAfterBorrow,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through an index reassigned after the borrow lowers");
    let index = array_read_index_const(&graph);
    assert_eq!(
        index,
        Some(1),
        "Relaxed load through an index reassigned after the borrow must use the captured index 1, got {index:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tags",
            None,
            Receiver::IndexZeroCapturedAtBorrow,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through an index 0 captured at the borrow lowers");
    let index = array_read_index_const(&graph);
    assert_eq!(
        index,
        Some(0),
        "Relaxed load through an index 0 captured at the borrow must use the captured index 0, got {index:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tag",
            None,
            Receiver::RawReborrowOfCallResult,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a raw reborrow of a call result lowers");
    assert!(
        graph_has_raw_load(&graph),
        "Relaxed load through a raw reborrow of a call result must RawLoad"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tags",
            None,
            Receiver::IndexParamReassignedAfterLoad,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through an index parameter reassigned after the load lowers");
    assert!(
        graph_has_array_read(&graph) || graph_has_field_read(&graph, "tags"),
        "Relaxed load through an index parameter reassigned after the load must read the array field"
    );

    // `s` mixes a plain `&Inner` parameter with a later field borrow
    // `s = &(*owner).inner`, so the local is Invalid and the load of
    // `&(*s).tag` cannot fold.
    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tag",
            None,
            Receiver::ParamBaseLaterReassigned,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a param base later reassigned to a field borrow must stay unsupported, got {result:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tag",
            None,
            Receiver::DeepSubstructure,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through 10 nested substructure borrows lowers");
    assert!(
        graph_has_field_read(&graph, "tag"),
        "Relaxed load through 10 nested substructure borrows must read tag"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tags",
            None,
            Receiver::ReborrowKeepsIndexCapture,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through a reborrow keeps the index captured at the inner borrow");
    let index = array_read_index_const(&graph);
    assert_eq!(
        index,
        Some(0),
        "reborrow of an indexed field must use the index captured at the inner borrow, got {index:?}"
    );

    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "cells",
            None,
            Receiver::RepeatedIndexLocal,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    )
    .expect("Relaxed load through rows[i].cells[i] with i reassigned between borrows lowers");
    let indexes = array_read_index_consts(&graph);
    assert_eq!(
        indexes,
        vec![0, 1],
        "rows[i].cells[i] with i reassigned between borrows must read rows[0].cells[1], got {indexes:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "tag",
            None,
            Receiver::MutBorrowedSubstructure,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a mutably borrowed substructure must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReborrowOfCallResult,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a reborrow of a call result must stay unsupported, got {result:?}"
    );
}

fn quasi_atomic_u64_field_fixture_with_owner_call() -> Llbc {
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
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": generics}});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let owner_adt = json!({"Adt": {"id": 2, "generics": generics, "builtin": null}});
    let owner_ref = json!({"Ref": ["Erased", owner_adt, "Shared"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let field_place = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
            {"Field": [null, 0]}
        ]},
        "ty": atomic
    });
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let attr = json!({"attributes": [], "inline": null, "rename": null, "public": true});
    let global_decls = json!([{
        "def_id": 0,
        "item_meta": meta(&["fixture", "_immutable_fields_Owner"]),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": "version_tag?"}, {"Ref": [
            "Erased",
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Shared"
        ]}]]}
    }]);
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}},
                {"def_id": 2, "item_meta": meta(&["fixture", "Owner"]),
                 "kind": {"Struct": [{
                    "name": "version_tag", "ty": atomic, "attr_info": attr
                 }]}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_field"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [owner_ref.clone()],
                    "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "owner", "span": span, "ty": owner_ref},
                        {"index": 2, "name": "ordering", "span": span, "ty": adt_ord},
                        {"index": 3, "name": "recv", "span": span, "ty": recv_ty},
                        {"index": 4, "name": "ignored", "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [
                            {"span": span, "kind": {"Assign": [place(2, &adt_ord),
                                {"Aggregate": [{"Adt": [1, 1, null, generics]}, []]}]}},
                            {"span": span, "kind": {"Assign": [place(3, &recv_ty),
                                {"Ref": {"place": field_place, "kind": "Shared", "ptr_metadata": null}}]}}
                        ], "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Copy": place(1, &owner_ref)}],
                                "dest": place(4, &unit)}, "target": 1, "on_unwind": 3
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Copy": place(3, &recv_ty)}, {"Copy": place(2, &adt_ord)}],
                                "dest": place(0, &word)}, "target": 2, "on_unwind": 3
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [recv_ty, adt_ord], "output": word}, "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(&["fixture", "other"]),
                 "signature": {"is_unsafe": false, "inputs": [owner_ref], "output": unit}, "body": "Opaque"}
            ], "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field owner-call fixture parses")
}

fn call_leaf_is_other(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("other")
        }
        CallTarget::Method { name, .. } => name == "other",
        _ => false,
    }
}

fn success_path_ops(graph: &FunctionGraph) -> Vec<&majit_translate::model::SpaceOperation> {
    let mut ops = Vec::new();
    let mut id = graph.startblock;
    for _ in 0..graph.blocks.len() {
        let block = graph.block(id);
        ops.extend(&block.operations);
        if id == graph.returnblock {
            break;
        }
        let Some(link) = block
            .exits
            .iter()
            .find(|link| link.target != graph.exceptblock)
        else {
            break;
        };
        if link.target == id {
            break;
        }
        id = link.target;
    }
    ops
}

#[test]
fn acquire_quasi_atomic_load_reads_at_the_load_site() {
    let graph = lower_function(
        &quasi_atomic_u64_field_fixture_with_owner_call(),
        "read_field",
    )
    .expect("Acquire load after an intervening owner call lowers");
    let return_var = graph
        .blocks
        .iter()
        .find_map(|block| {
            block.exits.iter().find_map(|link| {
                if link.target != graph.returnblock {
                    return None;
                }
                match link.args.as_slice() {
                    [LinkArg::Value(var)] => Some(var.clone()),
                    _ => None,
                }
            })
        })
        .expect("read_field returns a value");
    let ops = success_path_ops(&graph);
    let other_idx = ops.iter().position(|op| match &op.kind {
        OpKind::Call { target, .. } => call_leaf_is_other(target),
        _ => false,
    });
    let load_read_idx = ops.iter().position(|op| {
        op.result.as_ref() == Some(&return_var)
            && matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "version_tag"
            )
    });
    let Some(load_read_idx) = load_read_idx else {
        panic!("returned value is not a version_tag FieldRead: {ops:?}");
    };
    match other_idx {
        Some(other_idx) => assert!(
            load_read_idx > other_idx,
            "load-site version_tag FieldRead must follow the intervening call, got {ops:?}"
        ),
        None => {
            let load_block = graph
                .blocks
                .iter()
                .find(|block| {
                    block.operations.iter().any(|op| {
                        op.result.as_ref() == Some(&return_var)
                            && matches!(
                                &op.kind,
                                OpKind::FieldRead { field, .. } if field.name == "version_tag"
                            )
                    })
                })
                .expect("load-site FieldRead lives in some block");
            let borrow_is_elsewhere = graph.blocks.iter().any(|block| {
                block.id != load_block.id
                    && block.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::FieldRead { field, .. } if field.name == "version_tag"
                        ) && op.result.as_ref() != Some(&return_var)
                    })
            });
            assert!(
                borrow_is_elsewhere,
                "returned FieldRead must be the load-site read, distinct from the borrow's: {ops:?}"
            );
        }
    }
}

fn graph_has_raw_load(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(op.kind, OpKind::RawLoad { .. }))
    })
}

fn graph_return_var(graph: &FunctionGraph) -> Option<Variable> {
    graph.blocks.iter().find_map(|block| {
        block.exits.iter().find_map(|link| {
            if link.target != graph.returnblock {
                return None;
            }
            match link.args.as_slice() {
                [LinkArg::Value(var)] => Some(var.clone()),
                _ => None,
            }
        })
    })
}

fn call_target_leaf_is(target: &CallTarget, name: &str) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some(name)
        }
        CallTarget::Method { name: method, .. } => method == name,
        _ => false,
    }
}

fn traces_to_call_named(graph: &FunctionGraph, var: &Variable, name: &str) -> bool {
    let mut work = vec![var.clone()];
    let mut seen: Vec<Variable> = Vec::new();
    while let Some(cur) = work.pop() {
        if seen.iter().any(|existing| existing == &cur) {
            continue;
        }
        seen.push(cur.clone());
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() == Some(&cur)
                    && let OpKind::Call { target, .. } = &op.kind
                    && call_target_leaf_is(target, name)
                {
                    return true;
                }
            }
            let Some(i) = block.inputargs.iter().position(|a| a == &cur) else {
                continue;
            };
            for source in &graph.blocks {
                for link in &source.exits {
                    if link.target != block.id {
                        continue;
                    }
                    if let Some(LinkArg::Value(v)) = link.args.get(i) {
                        work.push(v.clone());
                    }
                }
            }
        }
    }
    false
}

fn graph_has_field_write(graph: &FunctionGraph, name: &str) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == name
            )
        })
    })
}

fn version_tag_field_base(graph: &FunctionGraph, write: bool) -> (BlockId, Variable) {
    field_op_base(graph, "version_tag", write)
}

fn outermost_field_base(graph: &FunctionGraph, name: &str, write: bool) -> (BlockId, Variable) {
    let (mut block_id, mut base) = field_op_base(graph, name, write);
    loop {
        let mut found = None;
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() == Some(&base)
                    && let OpKind::FieldRead { base: inner, .. } = &op.kind
                {
                    found = Some((block.id, inner.clone()));
                }
            }
        }
        match found {
            Some((next_block, inner)) => {
                block_id = next_block;
                base = inner;
            }
            None => return (block_id, base),
        }
    }
}

fn graph_has_array_read(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
    })
}

fn graph_has_field_read(graph: &FunctionGraph, name: &str) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == name
            )
        })
    })
}

fn array_read_index_const(graph: &FunctionGraph) -> Option<u64> {
    array_read_index_consts(graph).into_iter().next()
}

fn array_read_index_consts(graph: &FunctionGraph) -> Vec<u64> {
    let mut indexes = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::ArrayRead { index, .. } => {
                    if let Some(n) = const_uint_of(graph, index) {
                        indexes.push(n);
                    }
                }
                OpKind::Call { target, args, .. } if call_target_leaf_is(target, "getitem") => {
                    if let Some(LinkArg::Value(idx)) = args.get(1)
                        && let Some(n) = const_uint_of(graph, idx)
                    {
                        indexes.push(n);
                    }
                }
                _ => {}
            }
        }
    }
    indexes
}

fn const_uint_of(graph: &FunctionGraph, var: &Variable) -> Option<u64> {
    for block in &graph.blocks {
        for op in &block.operations {
            if op.result.as_ref() != Some(var) {
                continue;
            }
            match &op.kind {
                OpKind::ConstInt(n) if *n >= 0 => return Some(*n as u64),
                OpKind::ConstUInt(n) => return Some(*n),
                _ => {}
            }
        }
    }
    None
}

fn field_op_base(graph: &FunctionGraph, name: &str, write: bool) -> (BlockId, Variable) {
    if write {
        return graph
            .blocks
            .iter()
            .find_map(|block| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::FieldWrite { field, base, .. } if field.name == name => {
                        Some((block.id, base.clone()))
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| panic!("{name} field write"));
    }
    // A borrow of the field also emits a FieldRead; the folded load is the
    // FieldRead that produces the returned value, walking through inputargs.
    let mut work: Vec<Variable> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block.exits.iter().filter_map(|link| {
                if link.target != graph.returnblock {
                    return None;
                }
                match link.args.as_slice() {
                    [LinkArg::Value(var)] => Some(var.clone()),
                    _ => None,
                }
            })
        })
        .collect();
    let mut seen: Vec<Variable> = Vec::new();
    while let Some(var) = work.pop() {
        if seen.iter().any(|existing| existing == &var) {
            continue;
        }
        seen.push(var.clone());
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() == Some(&var)
                    && let OpKind::FieldRead { field, base, .. } = &op.kind
                    && field.name == name
                {
                    return (block.id, base.clone());
                }
            }
            let Some(i) = block.inputargs.iter().position(|a| a == &var) else {
                continue;
            };
            for source in &graph.blocks {
                for link in &source.exits {
                    if link.target != block.id {
                        continue;
                    }
                    if let Some(LinkArg::Value(v)) = link.args.get(i) {
                        work.push(v.clone());
                    }
                }
            }
        }
    }
    panic!("{name} field read producing the return")
}

/// Extra owner parameter of a merge fixture: last data input before the
/// bool flag that keeps both switch arms live.
fn merge_extra_owner(start: &[Variable]) -> &Variable {
    start
        .get(start.len().saturating_sub(2))
        .expect("merge extra owner then flag")
}

/// Every start-block input `var` traces back to through link args.
fn start_inputs_reached(graph: &FunctionGraph, block: BlockId, var: &Variable) -> Vec<Variable> {
    if block == graph.startblock {
        if graph.block(block).inputargs.iter().any(|a| a == var) {
            return vec![var.clone()];
        }
        return Vec::new();
    }
    let Some(i) = graph.block(block).inputargs.iter().position(|a| a == var) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for source in &graph.blocks {
        for link in &source.exits {
            if link.target != block {
                continue;
            }
            let Some(LinkArg::Value(v)) = link.args.get(i) else {
                continue;
            };
            for input in start_inputs_reached(graph, source.id, v) {
                if !found.iter().any(|existing| existing == &input) {
                    found.push(input);
                }
            }
        }
    }
    found
}

/// `var`, used in `block`, is `input` of the start block on every path:
/// each incoming link passes, at `var`'s position in `block.inputargs`, a
/// value that is `input` in its own source block.
fn traces_to_start_input(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
    input: &Variable,
) -> bool {
    if block == graph.startblock {
        return var == input;
    }
    let Some(i) = graph.block(block).inputargs.iter().position(|a| a == var) else {
        return false;
    };
    let mut any = false;
    for source in &graph.blocks {
        for link in &source.exits {
            if link.target != block {
                continue;
            }
            any = true;
            let Some(LinkArg::Value(v)) = link.args.get(i) else {
                return false;
            };
            if !traces_to_start_input(graph, source.id, v, input) {
                return false;
            }
        }
    }
    any
}

fn graph_has_atomic_store_call(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::Call { target, .. } => match target {
                CallTarget::FunctionPath { segments, .. } => {
                    segments.last().map(String::as_str) == Some("store")
                        && segments
                            .iter()
                            .any(|s| s.contains("Atomic") || s == "atomic")
                }
                CallTarget::Method { name, .. } => name == "store",
                _ => false,
            },
            _ => false,
        })
    })
}

fn atomic_u64_field_store_fixture(
    owner: &str,
    field: &str,
    entries: Option<&str>,
    ordering: &str,
) -> Llbc {
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
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": generics}});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let owner_adt = json!({"Adt": {"id": 2, "generics": generics, "builtin": null}});
    let owner_ref = json!({"Ref": ["Erased", owner_adt, "Shared"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let field_place = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
            {"Field": [null, 0]}
        ]},
        "ty": atomic
    });
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let index = variants.iter().position(|v| *v == ordering).unwrap();
    let attr = json!({"attributes": [], "inline": null, "rename": null, "public": true});
    let global_decls = match entries {
        Some(joined) => json!([{
            "def_id": 0,
            "item_meta": meta(&["fixture", &format!("_immutable_fields_{owner}")]),
            "global_kind": "NamedConst",
            "value": {"Value": [1, [{"Str": joined}, {"Ref": [
                "Erased",
                {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "Shared"
            ]}]]}
        }]),
        None => json!([]),
    };
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}},
                {"def_id": 2, "item_meta": meta(&["fixture", owner]),
                 "kind": {"Struct": [{
                    "name": field, "ty": atomic, "attr_info": attr
                 }]}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "write_field"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [owner_ref.clone(), word.clone()],
                    "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 2, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit},
                        {"index": 1, "name": "owner", "span": span, "ty": owner_ref},
                        {"index": 2, "name": "value", "span": span, "ty": word},
                        {"index": 3, "name": "ordering", "span": span, "ty": adt_ord},
                        {"index": 4, "name": "recv", "span": span, "ty": recv_ty}
                    ]},
                    "body": [
                        {"statements": [
                            {"span": span, "kind": {"Assign": [place(3, &adt_ord),
                                {"Aggregate": [{"Adt": [1, index, null, generics]}, []]}]}},
                            {"span": span, "kind": {"Assign": [place(4, &recv_ty),
                                {"Ref": {"place": field_place, "kind": "Shared", "ptr_metadata": null}}]}}
                        ], "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [
                                    {"Copy": place(4, &recv_ty)},
                                    {"Copy": place(2, &word)},
                                    {"Copy": place(3, &adt_ord)}
                                ],
                                "dest": place(0, &unit)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "store"]),
                 "signature": {"is_unsafe": false, "inputs": [recv_ty, word, adt_ord], "output": unit}, "body": "Opaque"}
            ], "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field store fixture parses")
}

fn atomic_from_ptr_store_release_fixture() -> Llbc {
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
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let unit = json!({"Adt": {"id": 99, "builtin": "Tuple", "generics": generics}});
    let pointee = json!({"RawPtr": [word.clone(), "Mut"]});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let slot_ty = json!({"RawPtr": [word.clone(), "Mut"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let _ = pointee;
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "AtomicPtr"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "write_slot"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [slot_ty.clone(), word.clone()],
                    "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 2, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit},
                        {"index": 1, "name": "slot", "span": span, "ty": slot_ty},
                        {"index": 2, "name": "value", "span": span, "ty": word},
                        {"index": 3, "name": "ordering", "span": span, "ty": adt_ord},
                        {"index": 4, "name": "recv", "span": span, "ty": recv_ty}
                    ]},
                    "body": [
                        {"statements": [
                            {"span": span, "kind": {"Assign": [place(3, &adt_ord),
                                {"Aggregate": [{"Adt": [1, 3, null, generics]}, []]}]}}
                        ], "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Copy": place(1, &slot_ty)}],
                                "dest": place(4, &recv_ty)}, "target": 1, "on_unwind": 3
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [
                                    {"Copy": place(4, &recv_ty)},
                                    {"Copy": place(2, &word)},
                                    {"Copy": place(3, &adt_ord)}
                                ],
                                "dest": place(0, &unit)}, "target": 2, "on_unwind": 3
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "AtomicPtr", "store"]),
                 "signature": {"is_unsafe": false, "inputs": [recv_ty, word, adt_ord], "output": unit}, "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(&["core", "sync", "atomic", "<Impl>", "from_ptr"]),
                 "signature": {"is_unsafe": false, "inputs": [slot_ty], "output": recv_ty}, "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("from_ptr store fixture parses")
}

#[test]
fn release_store_of_quasi_field_through_reassigned_owner_lowers() {
    let graph = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
            AtomicOp::ReleaseStore,
        ),
        "read_field",
    )
    .expect("Release store of a quasi field through a reassigned owner parameter lowers");
    let (block_id, base) = version_tag_field_base(&graph, true);
    let start = &graph.block(graph.startblock).inputargs;
    assert!(
        traces_to_start_input(&graph, block_id, &base, &start[0]),
        "Release store through a reassigned owner parameter must write version_tag of owner, got base {base:?}"
    );
    assert!(
        !traces_to_start_input(&graph, block_id, &base, &start[1]),
        "Release store through a reassigned owner parameter must not write version_tag of other, got base {base:?}"
    );
}

#[test]
fn release_store_of_quasi_atomic_field_lowers_to_field_write() {
    let graph = lower_function(
        &atomic_u64_field_store_fixture("Owner", "version_tag", Some("version_tag?"), "Release"),
        "write_field",
    )
    .expect("Release store of quasi AtomicU64 field lowers");
    assert!(
        graph_has_field_write(&graph, "version_tag"),
        "Release store of quasi version_tag must lower to FieldWrite, got {graph:?}"
    );
    assert!(
        !graph_has_atomic_store_call(&graph),
        "Release store of quasi version_tag must not keep Atomic::store, got {graph:?}"
    );
}

#[test]
fn release_store_of_non_quasi_atomic_field_stays_a_call() {
    let graph = lower_function(
        &atomic_u64_field_store_fixture("Plain", "hot_counter", None, "Release"),
        "write_field",
    )
    .expect("Release store of a non-quasi field keeps lowering");
    assert!(
        !graph_has_field_write(&graph, "hot_counter"),
        "Release store of non-quasi hot_counter must not become FieldWrite, got {graph:?}"
    );
    assert!(
        graph_has_atomic_store_call(&graph),
        "Release store of non-quasi hot_counter must stay Atomic::store, got {graph:?}"
    );
}

#[test]
fn relaxed_store_folds_regardless_of_rank() {
    for (owner, field, entries) in [
        ("Owner", "version_tag", Some("version_tag?")),
        ("Plain", "hot_counter", None),
    ] {
        let graph = lower_function(
            &atomic_u64_field_store_fixture(owner, field, entries, "Relaxed"),
            "write_field",
        )
        .unwrap_or_else(|err| panic!("Relaxed store of {field} must fold: {err:?}"));
        assert!(
            graph_has_field_write(&graph, field),
            "Relaxed store of {field} must lower to FieldWrite, got {graph:?}"
        );
        assert!(
            !graph_has_atomic_store_call(&graph),
            "Relaxed store of {field} must not keep Atomic::store, got {graph:?}"
        );
    }
}

#[test]
fn atomic_from_ptr_store_release_stays_a_call() {
    let graph = lower_function(&atomic_from_ptr_store_release_fixture(), "write_slot")
        .expect("from_ptr store Release keeps lowering");
    assert!(
        !graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::FieldWrite { .. }))
        }),
        "AtomicPtr::from_ptr(slot).store(Release) must not invent a FieldWrite, got {graph:?}"
    );
    assert!(
        graph_has_atomic_store_call(&graph),
        "AtomicPtr::from_ptr(slot).store(Release) must stay a Call, got {graph:?}"
    );
}

#[test]
#[ignore = "requires freshly extracted pyre-object LLBC; run extraction --check first"]
fn atomic_store_real_release_writer_of_quasi_field_lowers() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("load fresh pyre-object corpus");
    let graph = lower_function(&llbc, "w_type_set_compares_by_identity_status")
        .expect("Release store of quasi compares_by_identity_status lowers");
    assert!(
        graph_has_field_write(&graph, "compares_by_identity_status"),
        "lowered w_type_set_compares_by_identity_status has no compares_by_identity_status FieldWrite"
    );
    assert!(
        !graph_has_atomic_store_call(&graph),
        "w_type_set_compares_by_identity_status must not keep Call AtomicU8::store"
    );
}
