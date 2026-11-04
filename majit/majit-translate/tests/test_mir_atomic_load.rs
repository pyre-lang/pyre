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
fn atomic_load_real_acquire_reader_of_quasi_field_lowers() {
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
    CopiedFromCopiesOfAReassignedOwner,
    CopiedThroughDiamond,
    ParamCopiedAfterCall,
    CopiedThenOverwritten,
    OverwrittenByCall,
    ReassignedBase,
    ReassignedParam,
    OwnerReassignedAfterCall,
    OwnerReassignedBeforeCopy,
    CopiedThenOwnerReassigned,
    CopiedLastFromAnotherOwner,
    MutBorrowedOwner,
    SuffixIndex,
}

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
            | Receiver::CopiedFromCopiesOfAReassignedOwner
            | Receiver::CopiedLastFromAnotherOwner
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
    );
    let merge = matches!(
        receiver,
        Receiver::CopiedFromTwoBorrows
            | Receiver::CopiedThroughOwnerCopy
            | Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
            | Receiver::CopiedFromTwoOwners
            | Receiver::CopiedFromCopiesOfAReassignedOwner
            | Receiver::CopiedThroughDiamond
    );
    let is_store = matches!(op, AtomicOp::RelaxedStore | AtomicOp::ReleaseStore);
    let arg_count = 1
        + usize::from(other_param)
        + usize::from(is_store)
        + usize::from(extra_recv_param)
        + usize::from(extra_owner_param);
    let value_local = arg_count - usize::from(extra_recv_param) - usize::from(extra_owner_param);
    let ordering_local = arg_count + 1;
    let recv_bind = arg_count + 2;
    let extra_local = arg_count + 3;
    let field_base = match receiver {
        Receiver::ReassignedBase => extra_local,
        Receiver::CopiedFromCopiesOfAReassignedOwner => extra_local + 4,
        _ => 1,
    };
    let dest_ty = if is_store { unit.clone() } else { word.clone() };
    let suffix_index = matches!(receiver, Receiver::SuffixIndex);
    let array_len = json!([
        {"Integer": {"Unsigned": ["Usize", "2"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let array_ty = json!({"Array": [atomic, array_len, null]});
    let field_proj = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(field_base, &owner_ref), "Deref"]}, "ty": owner_adt},
            {"Field": [null, 0]}
        ]},
        "ty": if suffix_index { array_ty.clone() } else { atomic.clone() }
    });
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
    } else {
        field_proj
    };
    let owner_field_ty = if suffix_index {
        array_ty
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
        json!({"index": 1, "name": "owner", "span": span, "ty": owner_ref}),
    ];
    if other_param {
        locals.push(json!({"index": 2, "name": "other", "span": span, "ty": owner_ref}));
    }
    if is_store {
        locals.push(json!({"index": value_local, "name": "value", "span": span, "ty": word}));
    }
    if extra_recv_param {
        locals.push(json!({"index": arg_count, "name": "r", "span": span, "ty": recv_ty}));
    }
    if extra_owner_param {
        let extra_owner_name = match receiver {
            Receiver::CopiedFromCopiesOfAReassignedOwner => "p",
            Receiver::CopiedThroughAnotherOwnerCopyInOneBranch => "owner_b",
            _ => "owner2",
        };
        locals.push(
            json!({"index": arg_count, "name": extra_owner_name, "span": span, "ty": owner_ref}),
        );
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
    let recv2 = extra_local + 1;
    let owner2 = if extra_owner_param
        && !matches!(receiver, Receiver::CopiedThroughAnotherOwnerCopyInOneBranch)
    {
        arg_count
    } else {
        extra_local + 2
    };
    if matches!(
        receiver,
        Receiver::CopiedThroughOwnerCopy
            | Receiver::CopiedThroughOwnerCopyInOneBranch
            | Receiver::CopiedThroughAnotherOwnerCopyInOneBranch
    ) {
        if matches!(receiver, Receiver::CopiedThroughOwnerCopy) {
            statements.push(
                json!({"span": span, "kind": {"Assign": [place(owner2, &owner_ref),
                {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}),
            );
        }
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
            {"Use": [{"Copy": place(arg_count, &owner_ref)}, "Yes"]}]}}),
        );
        statements.push(
            json!({"span": span, "kind": {"Assign": [place(owner_b, &owner_ref),
            {"Use": [{"Copy": place(o, &owner_ref)}, "Yes"]}]}}),
        );
    }
    statements.push(
        json!({"span": span, "kind": {"Assign": [place(recv_bind, &recv_ty),
            {"Ref": {"place": field_place.clone(), "kind": "Shared", "ptr_metadata": null}}]}}),
    );
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
    let mut inputs = vec![owner_ref.clone()];
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
        inputs.push(owner_ref.clone());
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
    } else if let Some(keep) = keep {
        json!([{"span": span, "kind": {"Assign": [place(keep, &owner_ref),
            {"Use": [{"Copy": place(1, &owner_ref)}, "Yes"]}]}}])
    } else {
        json!([])
    };
    let overwritten_by_call = matches!(receiver, Receiver::OverwrittenByCall);
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
                    arg_count
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
                "Switch": {"discr": {"Const": [{"Bool": true}, {"Scalar": "Bool"}]}, "targets": {"If": [1, 2]}}
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
    if overwritten_by_call {
        fun_decls.push(json!({
            "def_id": 2, "item_meta": meta(&["fixture", "get_ref"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": recv_ty}, "body": "Opaque"
        }));
    }
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
                    "name": field, "ty": owner_field_ty, "attr_info": attr
                 }]}}
            ],
            "fun_decls": fun_decls, "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field fixture parses")
}

#[test]
fn atomic_field_access_lowers_through_a_stable_borrow_only() {
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

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
            AtomicOp::Load("Acquire"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load through a reassigned owner parameter must stay unsupported, got {result:?}"
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

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::ReassignedParam,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a reassigned owner parameter must stay unsupported, got {result:?}"
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

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Plain",
            "hot_counter",
            None,
            Receiver::ReassignedParam,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a reassigned owner parameter must stay unsupported, got {result:?}"
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

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughAnotherOwnerCopyInOneBranch,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref m))
            if m.contains("atomic load ordering Relaxed")
                && m.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a copy of borrows of two owners, one borrowed in one branch alone, must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThroughAnotherOwnerCopyInOneBranch,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref m))
            if m.contains("atomic store ordering Relaxed")
                && m.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a copy of borrows of two owners, one borrowed in one branch alone, must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoOwners,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref m))
            if m.contains("atomic load ordering Relaxed")
                && m.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a receiver copied from borrows of two owners must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromTwoOwners,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref m))
            if m.contains("atomic store ordering Relaxed")
                && m.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a receiver copied from borrows of two owners must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromCopiesOfAReassignedOwner,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref m))
            if m.contains("atomic load ordering Relaxed")
                && m.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a receiver copied from borrows of copies of a reassigned owner must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedFromCopiesOfAReassignedOwner,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref m))
            if m.contains("atomic store ordering Relaxed")
                && m.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a receiver copied from borrows of copies of a reassigned owner must stay unsupported, got {result:?}"
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

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OwnerReassignedBeforeCopy,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a copy of a borrow whose owner is reassigned before the copy must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::OwnerReassignedBeforeCopy,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a copy of a borrow whose owner is reassigned before the copy must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThenOwnerReassigned,
            AtomicOp::Load("Relaxed"),
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed load through a copy of a borrow whose owner is reassigned after the copy must stay unsupported, got {result:?}"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::CopiedThenOwnerReassigned,
            AtomicOp::RelaxedStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Relaxed")
                && message.contains("requires address-preserving ordered lowering")),
        "Relaxed store through a copy of a borrow whose owner is reassigned after the copy must stay unsupported, got {result:?}"
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
fn release_store_of_quasi_field_through_reassigned_owner_is_unsupported() {
    let result = lower_function(
        &quasi_atomic_u64_field_fixture(
            "Owner",
            "version_tag",
            Some("version_tag?"),
            Receiver::ReassignedParam,
            AtomicOp::ReleaseStore,
        ),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic store ordering Release")
                && message.contains("requires address-preserving ordered lowering")),
        "Release store of a quasi field through a reassigned owner parameter must stay unsupported, got {result:?}"
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
