use super::Unstructured;

use crate::model::FunctionGraph;

use majit_charon_reader::{Llbc, ullbc::TyRef};

use crate::front::mir::tests::{fixture_llbc, graph_call_leaves, llbc_with_types};

impl<'b> crate::front::mir::MovedOutLocals<'b> {
    /// A fixed answer for `body`, so a test can state the move set.
    pub(in crate::front::mir) fn with_set(body: &'b Unstructured, set: bit_set::BitSet) -> Self {
        Self {
            body,
            llbc: None,
            set: std::cell::OnceCell::from(set),
        }
    }
}

fn owned_vec_drop_llbc(
    name: &str,
    arg_count: u64,
    locals: Vec<serde_json::Value>,
    blocks: Vec<serde_json::Value>,
    extra_types: Vec<serde_json::Value>,
    extra_funs: Vec<serde_json::Value>,
) -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str], is_local: bool| {
        serde_json::json!({
            "name": path.iter().map(|segment| {
                if *segment == "<Impl>" {
                    serde_json::json!({"Impl": {"kind": "InherentImplBlock"}})
                } else {
                    serde_json::json!({"Ident": [segment, 0]})
                }
            }).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let vec_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [i64_ty],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let unit = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let inputs: Vec<serde_json::Value> = locals
        .iter()
        .skip(1)
        .take(arg_count as usize)
        .filter_map(|local| local.get("ty").cloned())
        .collect();
    let subject = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["fixture", name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": inputs,
            "output": unit.clone()
        },
        "body": {
            "Unstructured": {
                "span": span(),
                "locals": {"arg_count": arg_count, "locals": locals},
                "body": blocks
            }
        }
    });
    let vec_new = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["alloc", "vec", "<Impl>", "new"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": vec_ty.clone()
        },
        "body": "Opaque"
    });
    let sink = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["fixture", "sink"], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [vec_ty],
            "output": unit
        },
        "body": "Opaque"
    });
    let vec_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["alloc", "vec", "Vec"], false),
        "kind": {"Struct": []}
    });
    let mut types = vec![vec_decl];
    types.extend(extra_types);
    let mut funs = vec![subject, vec_new, sink];
    funs.extend(extra_funs);
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": types,
            "fun_decls": funs,
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("owned vec drop fixture parses")
}

fn owned_vec_ty() -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [{"Scalar": {"Integer": {"Signed": "I64"}}}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

fn owned_vec_unit_ty() -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

fn owned_vec_span() -> serde_json::Value {
    serde_json::json!({
        "data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }
    })
}

fn owned_vec_local(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "name": null,
        "span": owned_vec_span(),
        "ty": ty
    })
}

fn owned_vec_place(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"kind": {"Local": index}, "ty": ty})
}

fn owned_vec_block(
    statements: Vec<serde_json::Value>,
    terminator: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "statements": statements,
        "terminator": {"span": owned_vec_span(), "kind": terminator}
    })
}

fn owned_vec_new_call(
    dest: u64,
    dest_ty: &serde_json::Value,
    target: u64,
    unwind: u64,
) -> serde_json::Value {
    serde_json::json!({"Call": {
        "call": {
            "func": {"Regular": {
                "kind": {"Fun": 1},
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }},
            "args": [],
            "dest": owned_vec_place(dest, dest_ty)
        },
        "target": target,
        "on_unwind": unwind
    }})
}

fn owned_vec_drop(
    local: u64,
    ty: &serde_json::Value,
    target: u64,
    unwind: u64,
) -> serde_json::Value {
    serde_json::json!({"Drop": {
        "place": owned_vec_place(local, ty),
        "fn_ptr": {"kind": {"Fun": 1}, "generics": {}},
        "target": target,
        "on_unwind": unwind
    }})
}

fn owned_vec_return_unit(unit: &serde_json::Value) -> serde_json::Value {
    owned_vec_block(
        vec![serde_json::json!({
            "span": owned_vec_span(),
            "kind": {"Assign": [
                owned_vec_place(0, unit),
                {"Aggregate": ["Tuple", []]}
            ]}
        })],
        serde_json::json!("Return"),
    )
}

fn owned_vec_leaves(graph: &FunctionGraph) -> Vec<String> {
    graph_call_leaves(graph)
}

fn owned_vec_has_free(graph: &FunctionGraph) -> bool {
    owned_vec_leaves(graph)
        .iter()
        .any(|leaf| leaf.starts_with("ll_vec_free"))
}

fn owned_vec_has_alloc(graph: &FunctionGraph) -> bool {
    owned_vec_leaves(graph)
        .iter()
        .any(|leaf| leaf.starts_with("ll_vec_newemptylist"))
}

fn drop_term_is_goto(
    body: &majit_charon_reader::ullbc::Unstructured,
    llbc: &Llbc,
    bb: usize,
) -> bool {
    matches!(
        body.body[bb].term(llbc),
        Ok(majit_charon_reader::ullbc::TermKind::Goto { .. })
    )
}

fn drop_term_is_drop(
    body: &majit_charon_reader::ullbc::Unstructured,
    llbc: &Llbc,
    bb: usize,
) -> bool {
    matches!(
        body.body[bb].term(llbc),
        Ok(majit_charon_reader::ullbc::TermKind::Drop { .. })
    )
}

/// A definitely-initialised `Vec` Drop stays a Drop and lowers to
/// `ll_vec_free_*` at that terminator, the `rewrite_op_free` site.
#[test]
fn rust_vec_drop_of_definitely_init_vec_frees_at_the_drop() {
    let vec_ty = owned_vec_ty();
    let unit = owned_vec_unit_ty();
    let name = "drop_init_vec";
    let llbc = owned_vec_drop_llbc(
        name,
        0,
        vec![owned_vec_local(0, &unit), owned_vec_local(1, &vec_ty)],
        vec![
            owned_vec_block(vec![], owned_vec_new_call(1, &vec_ty, 1, 3)),
            owned_vec_block(vec![], owned_vec_drop(1, &vec_ty, 2, 3)),
            owned_vec_return_unit(&unit),
            owned_vec_block(vec![], serde_json::json!("UnwindResume")),
        ],
        vec![],
        vec![],
    );
    let mut body = llbc.local_fn(name).unwrap().unstructured().unwrap();
    super::elaborate_vec_drops(&llbc, &mut body);
    assert!(
        drop_term_is_drop(&body, &llbc, 1),
        "definitely-init Drop stays a Drop"
    );
    assert_eq!(
        super::vec_init_at_drop(&body, &llbc, 1, 1),
        super::VecInit::Init
    );
    let graph = super::lower_function(&llbc, name).expect("lower drop_init_vec");
    let leaves = owned_vec_leaves(&graph);
    assert!(owned_vec_has_alloc(&graph), "must allocate: {leaves:?}");
    assert!(
        owned_vec_has_free(&graph),
        "definitely-init Drop must free at the Drop: {leaves:?}"
    );
}

/// A `Vec` moved into a call is uninitialised at the trailing Drop;
/// elaboration turns that Drop into a Goto, so this graph does not free.
#[test]
fn rust_vec_drop_after_move_into_call_does_not_free() {
    let vec_ty = owned_vec_ty();
    let unit = owned_vec_unit_ty();
    let name = "drop_moved_into_call_vec";
    let sink = serde_json::json!({"Call": {
        "call": {
            "func": {"Regular": {
                "kind": {"Fun": 2},
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }},
            "args": [{"Move": owned_vec_place(1, &vec_ty)}],
            "dest": owned_vec_place(2, &unit)
        },
        "target": 2,
        "on_unwind": 4
    }});
    let llbc = owned_vec_drop_llbc(
        name,
        0,
        vec![
            owned_vec_local(0, &unit),
            owned_vec_local(1, &vec_ty),
            owned_vec_local(2, &unit),
        ],
        vec![
            owned_vec_block(vec![], owned_vec_new_call(1, &vec_ty, 1, 4)),
            owned_vec_block(vec![], sink),
            owned_vec_block(vec![], owned_vec_drop(1, &vec_ty, 3, 4)),
            owned_vec_return_unit(&unit),
            owned_vec_block(vec![], serde_json::json!("UnwindResume")),
        ],
        vec![],
        vec![],
    );
    let mut body = llbc.local_fn(name).unwrap().unstructured().unwrap();
    super::elaborate_vec_drops(&llbc, &mut body);
    assert!(
        drop_term_is_goto(&body, &llbc, 2),
        "moved-out Drop becomes Goto"
    );
    let graph = super::lower_function(&llbc, name).expect("lower drop_moved_into_call_vec");
    let leaves = owned_vec_leaves(&graph);
    assert!(owned_vec_has_alloc(&graph), "must allocate: {leaves:?}");
    assert!(
        !owned_vec_has_free(&graph),
        "moved-into-call must not free: {leaves:?}"
    );
}

/// A `Vec` stored into a struct field is uninitialised at the trailing
/// Drop; this graph does not free the moved-out header.
#[test]
fn rust_vec_drop_after_store_into_struct_does_not_free() {
    let vec_ty = owned_vec_ty();
    let unit = owned_vec_unit_ty();
    let holder_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let name = "drop_stored_into_struct_vec";
    let store = serde_json::json!({
        "span": owned_vec_span(),
        "kind": {"Assign": [
            owned_vec_place(2, &holder_ty),
            {"Aggregate": [
                {"Adt": [1, null]},
                [{"Move": owned_vec_place(1, &vec_ty)}]
            ]}
        ]}
    });
    let holder_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["Holder", 0]}
            ],
            "span": owned_vec_span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        },
        "kind": {"Struct": [{
            "name": "items",
            "ty": vec_ty,
            "attr_info": null
        }]}
    });
    let llbc = owned_vec_drop_llbc(
        name,
        0,
        vec![
            owned_vec_local(0, &unit),
            owned_vec_local(1, &vec_ty),
            owned_vec_local(2, &holder_ty),
        ],
        vec![
            owned_vec_block(vec![], owned_vec_new_call(1, &vec_ty, 1, 4)),
            owned_vec_block(vec![store], serde_json::json!({"Goto": {"target": 2}})),
            owned_vec_block(vec![], owned_vec_drop(1, &vec_ty, 3, 4)),
            owned_vec_return_unit(&unit),
            owned_vec_block(vec![], serde_json::json!("UnwindResume")),
        ],
        vec![holder_decl],
        vec![],
    );
    let mut body = llbc.local_fn(name).unwrap().unstructured().unwrap();
    super::elaborate_vec_drops(&llbc, &mut body);
    assert!(
        drop_term_is_goto(&body, &llbc, 2),
        "stored-into-struct Drop becomes Goto"
    );
    let graph = super::lower_function(&llbc, name).expect("lower drop_stored_into_struct_vec");
    let leaves = owned_vec_leaves(&graph);
    assert!(owned_vec_has_alloc(&graph), "must allocate: {leaves:?}");
    assert!(
        !owned_vec_has_free(&graph),
        "stored-into-struct must not free: {leaves:?}"
    );
}

/// Mixed init at a `Vec` Drop is a drop-flag case: leave the Drop,
/// decline `maybe-initialised`, emit no free.
#[test]
fn rust_vec_drop_maybe_initialised_declines_without_free() {
    let vec_ty = owned_vec_ty();
    let unit = owned_vec_unit_ty();
    let cond_ty = serde_json::json!({"Scalar": "Bool"});
    let name = "drop_maybe_init_vec";
    let sink = serde_json::json!({"Call": {
        "call": {
            "func": {"Regular": {
                "kind": {"Fun": 2},
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }},
            "args": [{"Move": owned_vec_place(2, &vec_ty)}],
            "dest": owned_vec_place(3, &unit)
        },
        "target": 4,
        "on_unwind": 6
    }});
    let llbc = owned_vec_drop_llbc(
        name,
        1,
        vec![
            owned_vec_local(0, &unit),
            owned_vec_local(1, &cond_ty),
            owned_vec_local(2, &vec_ty),
            owned_vec_local(3, &unit),
        ],
        vec![
            owned_vec_block(vec![], owned_vec_new_call(2, &vec_ty, 1, 6)),
            owned_vec_block(
                vec![],
                serde_json::json!({"Switch": {
                    "discr": {"Copy": owned_vec_place(1, &cond_ty)},
                    "targets": {"If": [2, 3]}
                }}),
            ),
            owned_vec_block(vec![], sink),
            owned_vec_block(vec![], serde_json::json!({"Goto": {"target": 4}})),
            owned_vec_block(vec![], owned_vec_drop(2, &vec_ty, 5, 6)),
            owned_vec_return_unit(&unit),
            owned_vec_block(vec![], serde_json::json!("UnwindResume")),
        ],
        vec![],
        vec![],
    );
    let mut body = llbc.local_fn(name).unwrap().unstructured().unwrap();
    super::elaborate_vec_drops(&llbc, &mut body);
    assert!(
        drop_term_is_drop(&body, &llbc, 4),
        "maybe-init Drop is left unlowered"
    );
    assert_eq!(
        super::vec_init_at_drop(&body, &llbc, 4, 2),
        super::VecInit::Maybe
    );
    let graph = super::lower_function(&llbc, name).expect("lower drop_maybe_init_vec");
    let leaves = owned_vec_leaves(&graph);
    assert!(owned_vec_has_alloc(&graph), "must allocate: {leaves:?}");
    assert!(
        !owned_vec_has_free(&graph),
        "maybe-init must not free: {leaves:?}"
    );
}

#[test]
fn root_scope_drop_is_classified_and_live() {
    use bit_set::BitSet;
    use majit_charon_reader::ullbc::Unstructured;
    assert!(super::gc_root_scope_drop_glue_path(
        "pyre_object::gc_roots::RootScope::<Impl>::drop_in_place"
    ));
    assert!(!super::gc_root_scope_drop_glue_path(
        "alloc::vec::Vec::<Impl>::drop_in_place"
    ));
    assert!(super::frame_anchor_drop_glue_path(
        "pyre_interpreter::eval::FrameAnchor::<Impl>::drop_in_place"
    ));
    assert!(super::frame_anchor_drop_glue_path(
        "eval::FrameAnchor::drop"
    ));
    assert!(!super::frame_anchor_drop_glue_path(
        "pyre_object::gc_roots::RootScope::<Impl>::drop_in_place"
    ));
    let release = crate::model::OpKind::Call {
        target: crate::model::CallTarget::FunctionPath {
            segments: vec![
                "pyre_interpreter".to_string(),
                "eval".to_string(),
                "frame_anchor_release".to_string(),
            ],
            fun_decl_id: None,
        },
        args: Vec::new(),
        result_ty: crate::model::ValueType::Void,
    };
    assert!(super::is_frame_anchor_release_call(&release));
    assert!(super::is_shadow_stack_bracket_close(&release));
    assert!(!super::is_root_scope_drop_glue_call(&release));

    // `_1` is written in bb0 and read only by bb1's Drop.
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let bb0 = serde_json::json!({
        "statements": [stmt(serde_json::json!({
            "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
        }))],
        "kind": "Regular",
        "terminator": {"kind": {"Goto": {"target": 1}}}
    });
    let bb1 = serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"kind": {"Drop": {
            "place": place_local(1),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": 2,
            "on_unwind": 2
        }}}
    });
    let bb2 = serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"kind": "Return"}
    });
    let body_json = serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 0, "locals": [local(0), local(1)]},
        "body": [bb0, bb1, bb2]
    });
    let body: Unstructured =
        serde_json::from_value(body_json).expect("fixture Unstructured parses");

    let erased = super::compute_mir_liveness(&fixture_llbc(), &body, &[], &BitSet::new());
    assert!(
        !erased[1].contains(1),
        "an erased drop reads nothing, so the guard is not live-in at bb1"
    );

    let mut as_call = BitSet::new();
    as_call.insert(1);
    let lowered = super::compute_mir_liveness(&fixture_llbc(), &body, &[], &as_call);
    assert!(
        lowered[1].contains(1),
        "a drop lowered as its glue call reads the guard, so it is live-in at bb1"
    );
}

#[test]
fn a_move_operand_typed_unlike_the_local_is_not_a_move_of_it() {
    use majit_charon_reader::ullbc::Unstructured;
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    // `_1` is declared as the guard type (ty#7), `_2` as a `usize` (ty#0).
    let local = |i: u64, ty: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": {"Deduplicated": ty}});
    let body_of = |spelled_ty: u64| -> Unstructured {
        serde_json::from_value(serde_json::json!({
                "span": span(),
                "locals": {"arg_count": 0, "locals": [local(0, 0), local(1, 7), local(2, 0)]},
                "body": [{
                    "statements": [{
                        "kind": {"Assign": [
                            {"kind": {"Local": 2}, "ty": {"Deduplicated": 0}},
                            {"Use": [{"Move": {"kind": {"Local": 1}, "ty": {"Deduplicated": spelled_ty}}}, "No"]}
                        ]},
                        "comments_before": [],
                        "span": span()
                    }],
                    "kind": "Regular",
                    "terminator": {"kind": "Return"}
                }]
            }))
            .expect("fixture Unstructured parses")
    };
    assert!(super::moved_out_locals(&body_of(7), &fixture_llbc()).contains(1));
    assert!(
        !super::moved_out_locals(&body_of(0), &fixture_llbc()).contains(1),
        "a `usize` operand spelled under the guard's index does not move the guard"
    );
}

#[test]
fn owner_root_erasure_needs_every_handle_use_to_be_new_deref_or_drop() {
    use super::OwnerRootCallee;
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let borrow = |dest: u64, src: u64| {
        stmt(serde_json::json!({
            "Assign": [place(dest), {"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}]
        }))
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_local = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    // Callee ids: 1 = `RBigIntGcRoot::new`, 2 = `live_long_num`,
    // 3 = `<RBigIntGcRoot as Deref>::deref`, 4 = an unrelated function.
    let callee_of = |reg: &RegularCall| {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return OwnerRootCallee::Other;
        };
        match id {
            1 => OwnerRootCallee::New,
            2 => OwnerRootCallee::Producer,
            3 => OwnerRootCallee::Deref,
            _ => OwnerRootCallee::Other,
        }
    };
    //   bb0: _2 = <open>(move _1)      -> bb1
    //   bb1: _3 = &_2; _4 = <read>(_3) -> bb2
    //   bb2: _5 = int_xor(_4)          -> bb3
    //   bb3: drop(_2)                  -> bb4
    //   bb4: return
    let body_of = |open: u64, read: u64| -> Unstructured {
        serde_json::from_value(serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 1, "locals": (0..=6).map(local).collect::<Vec<_>>()},
            "body": [
                block(vec![], call(open, vec![copy(1)], 2, 1)),
                block(vec![borrow(3, 2)], call(read, vec![copy(3)], 4, 2)),
                block(vec![], call(4, vec![copy(4)], 5, 3)),
                block(vec![], drop_local(2, 4)),
                block(vec![], serde_json::json!("Return")),
            ]
        }))
        .expect("fixture Unstructured parses")
    };
    let handle = |l: usize| l == 2;
    let names_handle = |l: usize| l == 2 || l == 3;

    for open in [1, 2] {
        let body = body_of(open, 3);
        let plan = super::analyze_owner_roots_with(
            &fixture_llbc(),
            &body,
            handle,
            names_handle,
            callee_of,
        );
        assert!(
            plan.erases(2),
            "callee {open}: a deref-only handle is erased"
        );
    }

    // The auto-deref receiver: `_3 = &_2; _6 = &(*_3); deref(_6)`. The
    // reborrow is the first borrow's one read.
    let reborrow: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 1, "locals": (0..=6).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![copy(1)], 2, 1)),
            block(
                vec![
                    borrow(3, 2),
                    stmt(serde_json::json!({
                        "Assign": [place(6), {"Ref": {
                            "place": {"kind": {"Projection": [place(3), "Deref"]}, "ty": ty()},
                            "kind": "Shared",
                            "ptr_metadata": null
                        }}]
                    })),
                ],
                call(3, vec![copy(6)], 4, 2),
            ),
            block(vec![], drop_local(2, 3)),
            block(vec![], serde_json::json!("Return")),
        ]
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_owner_roots_with(
        &fixture_llbc(),
        &reborrow,
        handle,
        |l| l == 2 || l == 3 || l == 6,
        callee_of,
    );
    assert!(plan.erases(2));
    assert_eq!(plan.handle_of_borrow(3), Some(2));
    assert_eq!(plan.handle_of_borrow(6), Some(2));

    // The borrow reaches a function that is not `Deref`: the handle
    // itself escapes, so it stays.
    for open in [1, 2] {
        let escaping = body_of(open, 4);
        let plan = super::analyze_owner_roots_with(
            &fixture_llbc(),
            &escaping,
            handle,
            names_handle,
            callee_of,
        );
        assert!(
            plan.roots.is_empty(),
            "callee {open}: the escaping handle stays"
        );
    }

    // A handle parameter is owned by the caller.
    let plan = super::analyze_owner_roots_with(
        &fixture_llbc(),
        &body_of(1, 3),
        |l| l == 1 || l == 2,
        |l| (1..=3).contains(&l),
        callee_of,
    );
    assert!(plan.roots.is_empty());

    // Another local naming the handle's type (`Option<RBigIntGcRoot>`)
    // holds it in a way the pass does not model.
    let plan = super::analyze_owner_roots_with(
        &fixture_llbc(),
        &body_of(1, 3),
        handle,
        |l| (2..=3).contains(&l) || l == 6,
        callee_of,
    );
    assert!(plan.roots.is_empty());

    // `live_rbigint`: the return place is the handle, erased, so the
    // producer's graph returns the value.
    let producer: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 1, "locals": (0..=1).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![copy(1)], 0, 1)),
            block(vec![], serde_json::json!("Return")),
        ]
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_owner_roots_with(
        &fixture_llbc(),
        &producer,
        |l| l == 0,
        |l| l == 0,
        callee_of,
    );
    assert!(plan.erases(0));
}

#[test]
fn root_bracket_erasure_accepts_a_single_pin_read_back_through_base() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // The `W_CTypePrimitiveSigned.convert_from_object` shape:
    //
    //   bb0: _2 = push_roots()          -> bb1
    //   bb1: _3 = &_2; _4 = base(_3)    -> bb2
    //   bb2: _5 = &_2; _6 = pin_root(_5, _1)  -> bb3
    //   bb3: _7 = &_2; _8 = get(_7, _4) -> bb4
    //   bb4: drop(_2)                   -> bb5
    //   bb5: return
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let borrow = |dest: u64, src: u64| {
        stmt(serde_json::json!({
            "Assign": [place(dest), {"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}]
        }))
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    // `TermKind::Drop` requires Charon's drop-glue `fn_ptr` (#1689).
    // Without it `term()` fails and the analyzer treats the raw JSON as
    // an unmodelled mention of the guard, retiring the bracket.
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    // Callee ids: 1 = push_roots, 2 = base, 3 = pin_root, 4 = get,
    // 5 = an unrelated function, 6 = the receiver-free pin_root,
    // 7 = a function the root-stack analysis proves balanced.
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                2 => "pyre_object::gc_roots::{impl RootScope}::base",
                3 => "pyre_object::gc_roots::{impl RootScope}::pin_root",
                4 => "pyre_object::gc_roots::{impl RootScope}::get",
                6 => "pyre_object::gc_roots::pin_root",
                7 => "pyre_object::listobject::ll_list_obj_resize_ge",
                _ => "pyre_interpreter::misc::as_long",
            }
            .to_string(),
        )
    };
    // Only callee 7 is known to leave the root stack as it found it.
    let touches = |reg: &RegularCall| -> bool {
        !matches!(&reg.kind, CallKind::Fun(FunId::Regular { id: 7 }))
    };
    let body_of = |get_index: u64, last_callee: u64| -> Unstructured {
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
            "body": [
                block(vec![], call(1, vec![], 2, 1)),
                block(vec![borrow(3, 2)], call(2, vec![copy(3)], 4, 2)),
                block(vec![borrow(5, 2)], call(3, vec![copy(5), copy(1)], 6, 3)),
                block(vec![borrow(7, 2)], call(last_callee, vec![copy(7), copy(get_index)], 8, 4)),
                block(vec![], drop_guard(2, 5)),
                block(vec![], serde_json::json!("Return")),
            ]
        });
        serde_json::from_value(body_json).expect("fixture Unstructured parses")
    };

    // Read back through the guard's own `base()`: erased, and the read
    // is answered with the pinned local `_1`.
    let paired = body_of(4, 4);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &paired,
        &super::MovedOutLocals::with_set(&paired, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(plan.scopes.contains(2), "the guard `_2` must be erased");
    assert_eq!(plan.pins.get(&2), Some(&vec![1]));
    assert_eq!(plan.aliases.get(&3), Some(&2));
    assert_eq!(plan.base_results.get(&4), Some(&2));
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);

    // `let roots = push_roots();` leaves a borrowck-only `FakeRead` of the
    // guard inside the bracket.  It reads nothing at run time, so the
    // bracket is still erased.
    let mut fake_read = body_of(4, 4);
    fake_read.body[2].statements.insert(
        0,
        serde_json::from_value(stmt(
            serde_json::json!({"Borrowck": {"FakeRead": place(2)}}),
        ))
        .unwrap(),
    );
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &fake_read,
        &super::MovedOutLocals::with_set(&fake_read, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(2),
        "a FakeRead of the guard is not a use"
    );
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);

    // A free pin does not name the guard, but its slot still belongs to
    // that guard's rewind. This is the real unpackiterable drain shape.
    let free_pin: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 2, 1)),
            block(vec![], call(6, vec![copy(1)], 6, 2)),
            block(vec![], drop_guard(2, 3)),
            block(vec![], serde_json::json!("Return")),
        ]
    }))
    .unwrap();
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &free_pin,
        &super::MovedOutLocals::with_set(&free_pin, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(!plan.scopes.contains(2), "a free pin still needs its close");

    // `FunDecl::unstructured` rewrites every `on_unwind` to one
    // `UnwindResume` block. That terminator does not change the stack.
    // Use the paired get shape (P3a still erases it); a free `pin_root`
    // without a matching len stays open and would not test this closer.
    let call_uw = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 6
        }})
    };
    let drop_uw = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 6
        }})
    };
    let unwind_resume: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call_uw(1, vec![], 2, 1)),
            block(vec![borrow(3, 2)], call_uw(2, vec![copy(3)], 4, 2)),
            block(vec![borrow(5, 2)], call_uw(3, vec![copy(5), copy(1)], 6, 3)),
            block(vec![borrow(7, 2)], call_uw(4, vec![copy(7), copy(4)], 8, 4)),
            block(vec![], drop_uw(2, 5)),
            block(vec![], serde_json::json!("Return")),
            block(vec![], serde_json::json!("UnwindResume")),
        ]
    }))
    .unwrap();
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &unwind_resume,
        &super::MovedOutLocals::with_set(&unwind_resume, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(2),
        "an on_unwind UnwindResume does not keep the bracket"
    );

    // Charon 0.1.297 spells the former `Abort("UndefinedBehavior")` as
    // a first-class terminator. It is a never-returning sink and does
    // not change the stack.
    let undefined_behavior: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call_uw(1, vec![], 2, 1)),
            block(vec![borrow(3, 2)], call_uw(2, vec![copy(3)], 4, 2)),
            block(vec![borrow(5, 2)], call_uw(3, vec![copy(5), copy(1)], 6, 3)),
            block(vec![borrow(7, 2)], call_uw(4, vec![copy(7), copy(4)], 8, 4)),
            block(vec![], drop_uw(2, 5)),
            block(vec![], serde_json::json!("Return")),
            block(vec![], serde_json::json!("UndefinedBehavior")),
        ]
    }))
    .unwrap();
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &undefined_behavior,
        &super::MovedOutLocals::with_set(&undefined_behavior, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(2),
        "an on_unwind UndefinedBehavior does not keep the bracket"
    );

    // The same problem can follow a known method pin. An unmodelled
    // callee may append or overwrite a slot without borrowing this guard.
    for callee in [5, 6] {
        let mut foreign = body_of(4, 4);
        foreign.body[2].terminator.kind =
            serde_json::value::to_raw_value(&call(3, vec![copy(5), copy(1)], 6, 6)).unwrap();
        foreign.body.push(
            serde_json::from_value(block(vec![], call(callee, vec![copy(1)], 9, 3))).unwrap(),
        );
        let plan = super::analyze_root_brackets_with(
            &fixture_llbc(),
            &foreign,
            &super::MovedOutLocals::with_set(&foreign, bit_set::BitSet::new()),
            name_of,
            touches,
        );
        assert!(
            !plan.scopes.contains(2),
            "unknown stack effects keep the bracket"
        );
    }

    // A callee proved to leave the root stack as it found it does not
    // keep the bracket: `ll_append` spans `_ll_resize_ge` this way.
    let mut balanced = body_of(4, 4);
    balanced.body[2].terminator.kind =
        serde_json::value::to_raw_value(&call(3, vec![copy(5), copy(1)], 6, 6)).unwrap();
    balanced
        .body
        .push(serde_json::from_value(block(vec![], call(7, vec![copy(1)], 9, 3))).unwrap());
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &balanced,
        &super::MovedOutLocals::with_set(&balanced, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(2),
        "a balanced callee inside the bracket must not keep it"
    );
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);

    // The shape the real MIR has: a call argument is a fresh temporary,
    // so the index `get` names is `_9 = copy _4` rather than `_4`.
    let through_copy: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 2, 1)),
            block(vec![borrow(3, 2)], call(2, vec![copy(3)], 4, 2)),
            block(vec![borrow(5, 2)], call(3, vec![copy(5), copy(1)], 6, 3)),
            block(
                vec![
                    borrow(7, 2),
                    stmt(serde_json::json!({"Assign": [place(9), {"Use": [copy(4), "No"]}]})),
                ],
                call(4, vec![copy(7), copy(9)], 8, 4),
            ),
            block(vec![], drop_guard(2, 5)),
            block(vec![], serde_json::json!("Return")),
        ]
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &through_copy,
        &super::MovedOutLocals::with_set(&through_copy, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(2),
        "an index reached through a copy temporary must still pair"
    );
    assert_eq!(plan.base_results.get(&9), Some(&2));
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);

    // A pin saves the value, not the mutable MIR carrier that supplied
    // it. An assignment after the pin must not change get(base).
    let mut reassigned = body_of(4, 4);
    reassigned.body[2].terminator.kind =
        serde_json::value::to_raw_value(&call(3, vec![copy(5), copy(1)], 6, 6)).unwrap();
    reassigned
        .body
        .push(serde_json::from_value(block(vec![], call(5, vec![], 1, 3))).unwrap());
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &reassigned,
        &super::MovedOutLocals::with_set(&reassigned, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(2),
        "pin input must retain its saved value"
    );

    // One call site in a loop is several runtime pushes. The first slot
    // stays the first value even as the pin's local receives later ones.
    let mut repeated = body_of(4, 4);
    repeated.locals.arg_count = 0;
    // The loop's producer returns a new value into _1 each iteration.
    // It is one static definition, so definition counting alone cannot
    // distinguish the first saved value from later ones.
    repeated.body[1].terminator.kind =
        serde_json::value::to_raw_value(&call(2, vec![copy(3)], 4, 6)).unwrap();
    repeated.body[3].terminator.kind =
        serde_json::value::to_raw_value(&call(4, vec![copy(7), copy(4)], 8, 6)).unwrap();
    repeated
        .body
        .push(serde_json::from_value(block(vec![], call(5, vec![], 1, 2))).unwrap());
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &repeated,
        &super::MovedOutLocals::with_set(&repeated, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(2),
        "a pin may not repeat under one base"
    );

    // An inner pin before the outer pin occupies the outer base. Each
    // guard has exactly one static pin, but outer.get(base) reads the
    // inner value. The actual drops still run in LIFO order.
    let mut nested = body_of(4, 4);
    nested
        .locals
        .locals
        .push(serde_json::from_value(local(10)).unwrap());
    nested
        .locals
        .locals
        .push(serde_json::from_value(local(11)).unwrap());
    let pin_outer = std::mem::replace(
        &mut nested.body[2],
        serde_json::from_value(block(vec![], call(1, vec![], 9, 6))).unwrap(),
    );
    nested.body.push(
        serde_json::from_value(block(
            vec![borrow(10, 9)],
            call(3, vec![copy(10), copy(0)], 11, 7),
        ))
        .unwrap(),
    );
    nested.body.push(pin_outer);
    nested.body[4].terminator.kind = serde_json::value::to_raw_value(&drop_guard(9, 8)).unwrap();
    nested
        .body
        .push(serde_json::from_value(block(vec![], drop_guard(2, 5))).unwrap());
    // Produce the inner value before either scope opens, so _0 is a
    // defined pointer distinct from the outer pin's input _1.
    let open_outer = std::mem::replace(
        &mut nested.body[0],
        serde_json::from_value(block(vec![], call(5, vec![], 0, 9))).unwrap(),
    );
    nested.body.push(open_outer);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &nested,
        &super::MovedOutLocals::with_set(&nested, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(2),
        "outer base may already hold an inner pin"
    );

    // Read back through a slot this pass cannot name: the whole bracket
    // stays.
    let unpaired = body_of(9, 4);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &unpaired,
        &super::MovedOutLocals::with_set(&unpaired, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(2),
        "a read at an unnameable slot must keep its bracket"
    );

    // The guard reaching any other callee is an escape.
    let escaped = body_of(4, 5);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &escaped,
        &super::MovedOutLocals::with_set(&escaped, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(2),
        "a guard handed to an unmodelled callee must keep its bracket"
    );
}

#[test]
fn root_bracket_erasure_answers_a_pin_roots_run() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // The `with_roots!(a, b => callee(a))` shape:
    //
    //   bb0: _3 = push_roots()                                  -> bb1
    //   bb1: _19 = copy _1; _20 = copy _2; _6 = [move _19, move _20];
    //        _7 = &_6; _8 = &(*_7); _9 = move _8 as &[_] (Unsize);
    //        _4 = &_3; _5 = pin_roots(_4, move _9)               -> bb2
    //   bb2: _10 = callee(_1)                                   -> bb3
    //   bb3: _11 = &_3; _12 = copy _5; _13 = get(_11, _12)       -> bb4
    //   bb4: _14 = &_3; _15 = AddChecked(copy _5, 1); assert    -> bb5
    //   bb5: _16 = move _15.0; _17 = get(_14, move _16)          -> bb6
    //   bb6: drop(_3)                                           -> bb7
    //   bb7: return
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let field = |i: u64, f: u64| serde_json::json!({"kind": {"Projection": [place(i), {"Field": [null, f]}]}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let mv = |i: u64| serde_json::json!({"Move": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow_place = |dest: u64, src: serde_json::Value| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": src, "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let borrow = |dest: u64, src: u64| borrow_place(dest, place(src));
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                3 => "pyre_object::gc_roots::{impl RootScope}::pin_roots",
                4 => "pyre_object::gc_roots::{impl RootScope}::get",
                6 => "pyre_object::gc_roots::pin_roots",
                7 => "pyre_object::gc_roots::shadow_stack_get",
                _ => "pyre_object::listobject::ll_list_obj_resize_ge",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    let body_of = |elements: Vec<serde_json::Value>| -> Unstructured {
        serde_json::from_value(serde_json::json!({
                "span": span(),
                "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
                "body": [
                    block(vec![], call(1, vec![], 3, 1)),
                    block(
                        vec![
                            assign(19, serde_json::json!({"Use": [copy(1), "No"]})),
                            assign(20, serde_json::json!({"Use": [copy(2), "No"]})),
                            assign(6, serde_json::json!({"Aggregate": [{"Array": [ty(), usize_lit(2)]}, elements]})),
                            borrow(7, 6),
                            borrow_place(8, serde_json::json!({"kind": {"Projection": [place(7), "Deref"]}, "ty": ty()})),
                            assign(9, serde_json::json!({"UnaryOp": [{"Cast": {"Unsize": [ty(), ty(), {"Length": usize_lit(2)}]}}, mv(8)]})),
                            borrow(4, 3),
                        ],
                        call(3, vec![copy(4), mv(9)], 5, 2),
                    ),
                    block(vec![], call(5, vec![copy(1)], 10, 3)),
                    block(
                        vec![borrow(11, 3), assign(12, serde_json::json!({"Use": [copy(5), "No"]}))],
                        call(4, vec![copy(11), copy(12)], 13, 4),
                    ),
                    block(
                        vec![
                            borrow(14, 3),
                            assign(15, serde_json::json!({"BinaryOp": ["AddChecked", copy(5), usize_lit(1)]})),
                        ],
                        serde_json::json!({"Assert": {
                            "assert": {
                                "cond": {"Move": field(15, 1)},
                                "expected": false,
                                "check_kind": {"Overflow": [{"Add": "Wrap"}, copy(5), usize_lit(1)]}
                            },
                            "target": 5,
                            "on_unwind": 99
                        }}),
                    ),
                    block(
                        vec![assign(16, serde_json::json!({"Use": [{"Move": field(15, 0)}, "No"]}))],
                        call(4, vec![copy(14), mv(16)], 17, 6),
                    ),
                    block(vec![], drop_guard(3, 7)),
                    block(vec![], serde_json::json!("Return")),
                ],
            }))
            .expect("fixture Unstructured parses")
    };

    let run = body_of(vec![mv(19), mv(20)]);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &run,
        &super::MovedOutLocals::with_set(&run, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(plan.scopes.contains(3), "a pin_roots run must be erased");

    assert_eq!(plan.pins.get(&3), Some(&vec![19, 20]));
    assert_eq!(plan.get_sites, vec![(3usize, 19usize), (5usize, 20usize)]);
    for temp in [6, 7, 8, 9] {
        assert_eq!(plan.pin_temps.get(&temp), Some(&3), "argument temp _{temp}");
    }
    for base in [5, 12] {
        assert_eq!(plan.base_results.get(&base), Some(&3), "base _{base}");
    }

    // An element that is not a plain local leaves nothing to answer with.
    let constant = body_of(vec![mv(19), usize_lit(0)]);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &constant,
        &super::MovedOutLocals::with_set(&constant, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "a constant element keeps the bracket"
    );

    // `with_roots!` pins the `mut` locals themselves and writes each
    // `get` back into its own local: bb3 `_1 = get(..)`, bb5 `_2 = get(..)`.
    // The restores are the only definitions inside the bracket, so every
    // read there still sees the pinned value.
    let mut restored = body_of(vec![copy(1), copy(2)]);
    restored.body[3].set_terminator_kind(call(4, vec![copy(11), copy(12)], 1, 4));
    restored.body[5].set_terminator_kind(call(4, vec![copy(14), mv(16)], 2, 6));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &restored,
        &super::MovedOutLocals::with_set(&restored, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "a restore into the pinned local must still be erased"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![1, 2]));
    assert_eq!(plan.get_sites, vec![(3usize, 1usize), (5usize, 2usize)]);

    // Any other write to a pinned local inside the bracket changes what a
    // later `get` would be answered with.
    let mut clobbered = body_of(vec![copy(1), copy(2)]);
    clobbered.body[2].set_terminator_kind(call(5, vec![copy(1)], 1, 3));
    clobbered.body[3].set_terminator_kind(call(4, vec![copy(11), copy(12)], 1, 4));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &clobbered,
        &super::MovedOutLocals::with_set(&clobbered, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "a non-get write to a pinned local keeps the bracket"
    );

    // The free spelling: `let _r = push_roots(); let b = pin_roots(&[..]);
    // shadow_stack_get(b + k)` pins onto and reads from the one bracket
    // open around it.
    let mut free = body_of(vec![mv(19), mv(20)]);
    free.body[1].statements.pop();
    free.body[1].set_terminator_kind(call(6, vec![mv(9)], 5, 2));
    free.body[3].statements.remove(0);
    free.body[3].set_terminator_kind(call(7, vec![copy(12)], 13, 4));
    free.body[4].statements.remove(0);
    free.body[5].set_terminator_kind(call(7, vec![mv(16)], 17, 6));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &free,
        &super::MovedOutLocals::with_set(&free, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "a free pin_roots run must be erased"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![19, 20]));
    assert_eq!(plan.get_sites, vec![(3usize, 19usize), (5usize, 20usize)]);
    for bb in [1, 3, 5] {
        assert_eq!(plan.free_sites.get(&bb), Some(&3), "free call in bb{bb}");
    }

    // A read of a slot some other guard's pin filled is not this one's.
    let mut stray = body_of(vec![mv(19), mv(20)]);
    stray.body[3].set_terminator_kind(call(7, vec![copy(13)], 18, 4));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &stray,
        &super::MovedOutLocals::with_set(&stray, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "a free read at a slot this pass cannot name keeps the bracket"
    );
}

#[test]
fn root_bracket_erasure_answers_a_later_pin_roots_dest() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // `zip_two_tuple_next` / `tuple_iter_descr_next`: each cursor is a
    // `pin_roots(&[value])` of the same bracket. The later dest is that
    // run's first slot, not the guard's base.
    //
    //   bb0: _3 = push_roots()                         -> bb1
    //   bb1: _19 = copy _1; _6 = [move _19]; ...;
    //        _4 = pin_roots(move _9)                   -> bb2
    //   bb2: _10 = callee(_1)                          -> bb3
    //   bb3: _20 = copy _2; _12 = [move _20]; ...;
    //        _11 = pin_roots(move _15)                 -> bb4
    //   bb4: _16 = shadow_stack_get(_4)                -> bb5
    //   bb5: _17 = shadow_stack_get(_11)               -> bb6
    //   bb6: drop(_3)                                  -> bb7
    //   bb7: return
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let mv = |i: u64| serde_json::json!({"Move": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow_place = |dest: u64, src: serde_json::Value| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": src, "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let borrow = |dest: u64, src: u64| borrow_place(dest, place(src));
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                6 => "pyre_object::gc_roots::pin_roots",
                7 => "pyre_object::gc_roots::shadow_stack_get",
                _ => "pyre_object::listobject::ll_list_obj_resize_ge",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    let pin_slice = |copy_dest: u64, src: u64, arr: u64, r: u64, r2: u64, slice: u64| {
        vec![
            assign(copy_dest, serde_json::json!({"Use": [copy(src), "No"]})),
            assign(
                arr,
                serde_json::json!({"Aggregate": [{"Array": [ty(), usize_lit(1)]}, [mv(copy_dest)]]}),
            ),
            borrow(r, arr),
            borrow_place(
                r2,
                serde_json::json!({"kind": {"Projection": [place(r), "Deref"]}, "ty": ty()}),
            ),
            assign(
                slice,
                serde_json::json!({"UnaryOp": [{"Cast": {"Unsize": [ty(), ty(), {"Length": usize_lit(1)}]}}, mv(r2)]}),
            ),
        ]
    };
    let two_runs: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 3, 1)),
            block(pin_slice(19, 1, 6, 7, 8, 9), call(6, vec![mv(9)], 4, 2)),
            block(vec![], call(5, vec![copy(1)], 10, 3)),
            block(pin_slice(20, 2, 12, 13, 14, 15), call(6, vec![mv(15)], 11, 4)),
            block(vec![], call(7, vec![copy(4)], 16, 5)),
            block(vec![], call(7, vec![copy(11)], 17, 6)),
            block(vec![], drop_guard(3, 7)),
            block(vec![], serde_json::json!("Return")),
        ],
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &two_runs,
        &super::MovedOutLocals::with_set(&two_runs, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "a later pin_roots dest is that run's first slot"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![19, 20]));
    assert_eq!(plan.get_sites, vec![(4usize, 19usize), (5usize, 20usize)]);
    assert_eq!(plan.base_results.get(&4), Some(&3));
    assert_eq!(plan.base_results.get(&11), Some(&3));

    // The later pin after an early return: the get of its dest runs only
    // on that path, so the read still names one pin.
    let branched: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 3, 1)),
            block(pin_slice(19, 1, 6, 7, 8, 9), call(6, vec![mv(9)], 4, 2)),
            block(
                vec![],
                serde_json::json!({"Switch": {"discr": copy(1), "targets": {"If": [3, 6]}}}),
            ),
            block(pin_slice(20, 2, 12, 13, 14, 15), call(6, vec![mv(15)], 11, 4)),
            block(vec![], call(7, vec![copy(11)], 17, 5)),
            block(vec![], drop_guard(3, 7)),
            block(vec![], drop_guard(3, 7)),
            block(vec![], serde_json::json!("Return")),
        ],
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &branched,
        &super::MovedOutLocals::with_set(&branched, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "a later pin_roots behind an early return still erases when its dest is read on that path"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![19, 20]));
    assert_eq!(plan.get_sites, vec![(4usize, 20usize)]);

    // `first_dest + 1` on a path that may skip the later pin: the slot is
    // empty there, so the bracket stays.
    let skipped: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 3, 1)),
            block(pin_slice(19, 1, 6, 7, 8, 9), call(6, vec![mv(9)], 4, 2)),
            block(
                vec![],
                serde_json::json!({"Switch": {"discr": copy(1), "targets": {"If": [3, 4]}}}),
            ),
            block(pin_slice(20, 2, 12, 13, 14, 15), call(6, vec![mv(15)], 11, 4)),
            block(
                vec![assign(
                    18,
                    serde_json::json!({"BinaryOp": ["Add", copy(4), usize_lit(1)]}),
                )],
                call(7, vec![copy(18)], 16, 5),
            ),
            block(vec![], drop_guard(3, 6)),
            block(vec![], serde_json::json!("Return")),
        ],
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &skipped,
        &super::MovedOutLocals::with_set(&skipped, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "first_dest + 1 must not answer a pin_roots that did not run"
    );
}

#[test]
fn root_bracket_erasure_answers_an_incoming_slice_pin() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // `__majit_wrap_cdata_call`: `pin_roots(args)` of the incoming
    // `&[PyObjectRef]` parameter, `get(base)` of slot 0, and
    // `shadow_stack_copy_range(base + 1, rest)`.
    //
    //   bb0: _3 = push_roots()                            -> bb1
    //   bb1: _4 = &_3; _5 = pin_roots(_4, copy _1)        -> bb2
    //   bb2: _6 = &_3; _7 = copy _5; _8 = get(_6, _7)     -> bb3
    //   bb3: _9 = AddChecked(copy _5, 1); assert          -> bb4
    //   bb4: _10 = move _9.0; copy_range(_10, _2)         -> bb5
    //   bb5: drop(_3)                                     -> bb6
    //   bb6: return
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let field = |i: u64, f: u64| serde_json::json!({"kind": {"Projection": [place(i), {"Field": [null, f]}]}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let mv = |i: u64| serde_json::json!({"Move": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow = |dest: u64, src: u64| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                3 => "pyre_object::gc_roots::{impl RootScope}::pin_roots",
                4 => "pyre_object::gc_roots::{impl RootScope}::get",
                8 => "pyre_object::gc_roots::shadow_stack_copy_range",
                _ => "pyre_object::listobject::ll_list_obj_resize_ge",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 3, 1)),
            block(
                vec![borrow(4, 3)],
                call(3, vec![copy(4), copy(1)], 5, 2),
            ),
            block(
                vec![borrow(6, 3), assign(7, serde_json::json!({"Use": [copy(5), "No"]}))],
                call(4, vec![copy(6), copy(7)], 8, 3),
            ),
            block(
                vec![assign(
                    9,
                    serde_json::json!({"BinaryOp": ["AddChecked", copy(5), usize_lit(1)]}),
                )],
                serde_json::json!({"Assert": {
                    "assert": {
                        "cond": {"Move": field(9, 1)},
                        "expected": false,
                        "check_kind": {"Overflow": [{"Add": "Wrap"}, copy(5), usize_lit(1)]}
                    },
                    "target": 4,
                    "on_unwind": 99
                }}),
            ),
            block(
                vec![assign(10, serde_json::json!({"Use": [{"Move": field(9, 0)}, "No"]}))],
                call(8, vec![mv(10), copy(2)], 0, 5),
            ),
            block(vec![], drop_guard(3, 6)),
            block(vec![], serde_json::json!("Return")),
        ],
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &body,
        &super::MovedOutLocals::with_set(&body, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "an incoming-slice pin_roots must be erased"
    );
    assert_eq!(plan.slice_gets, vec![(2usize, 1usize, 0u64)]);
    assert_eq!(plan.slice_copies, vec![(4usize, 1usize, 1u64)]);
    assert_eq!(
        plan.free_sites.get(&4),
        Some(&3),
        "copy_range is this guard"
    );
}

#[test]
fn root_bracket_erasure_answers_an_incoming_elem_pin() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // `W_CTypeFunc._call` `n == 1` (`ctypefunc.py` `_call`): `base()`,
    // `pin_root(args_w[0])`, `get(base)`. The pin's temporary is
    // `StorageDead` before the get, so the answer is `getarrayitem` of
    // the incoming list at 0, not that local.
    //
    //   bb0: _3 = push_roots()                                 -> bb1
    //   bb1: _4 = &_3; _5 = base(_4)                           -> bb2
    //   bb2: _6 = 0; _7 = (*_1)[copy _6]; _8 = &_3;
    //        _9 = copy _7; pin_root(_8, move _9)               -> bb3
    //   bb3: convert()                                         -> bb4
    //   bb4: _10 = &_3; _11 = copy _5; get(_10, _11)           -> bb5
    //   bb5: drop(_3)                                          -> bb6
    //   bb6: return
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let mv = |i: u64| serde_json::json!({"Move": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let deref =
        |i: u64| serde_json::json!({"kind": {"Projection": [place(i), "Deref"]}, "ty": ty()});
    let index_place = |slice: u64, offset: u64| {
        serde_json::json!({"kind": {"Projection": [deref(slice), {"Index": {
                "offset": copy(offset),
                "from_end": false
            }}]}, "ty": ty()})
    };
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow = |dest: u64, src: u64| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                2 => "pyre_object::gc_roots::{impl RootScope}::base",
                3 => "pyre_object::gc_roots::{impl RootScope}::pin_root",
                4 => "pyre_object::gc_roots::{impl RootScope}::get",
                5 => "pyre_module::module::_cffi_backend::ctypeobj::convert_argument_from_object",
                _ => "pyre_object::listobject::ll_list_obj_resize_ge",
            }
            .to_string(),
        )
    };
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 3, 1)),
            block(vec![borrow(4, 3)], call(2, vec![copy(4)], 5, 2)),
            block(
                vec![
                    assign(6, serde_json::json!({"Use": [usize_lit(0), "No"]})),
                    assign(7, serde_json::json!({"Use": [{"Copy": index_place(1, 6)}, "Yes"]})),
                    borrow(8, 3),
                    assign(9, serde_json::json!({"Use": [copy(7), "Yes"]})),
                ],
                call(3, vec![mv(8), mv(9)], 10, 3),
            ),
            block(vec![], call(5, vec![], 14, 4)),
            block(
                vec![
                    borrow(12, 3),
                    assign(11, serde_json::json!({"Use": [copy(5), "No"]})),
                ],
                call(4, vec![copy(12), copy(11)], 13, 5),
            ),
            block(vec![], drop_guard(3, 6)),
            block(vec![], serde_json::json!("Return")),
        ],
    }))
    .expect("fixture Unstructured parses");
    let moved = super::MovedOutLocals::with_set(&body, bit_set::BitSet::new());
    for touches_callees in [false, true] {
        let touches = |reg: &RegularCall| {
            touches_callees && matches!(&reg.kind, CallKind::Fun(FunId::Regular { id }) if *id == 5)
        };
        let plan =
            super::analyze_root_brackets_with(&fixture_llbc(), &body, &moved, name_of, touches);
        assert!(
            plan.scopes.contains(3),
            "n==1 incoming-elem pin must be erased (touches={touches_callees})"
        );
        assert_eq!(
            plan.slice_gets,
            vec![(4usize, 1usize, 0u64)],
            "get(base) answers with args_w[0] (touches={touches_callees})"
        );
    }
}

#[test]
fn root_bracket_erasure_numbers_pins_in_order_and_reads_base_plus_k() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // The `w_list_append_inner` Object-arm shape:
    //
    //   bb0: _3 = push_roots()                       -> bb1
    //   bb1: _4 = &_3; _5 = base(_4)                 -> bb2
    //   bb2: _6 = &_3; _7 = pin_root(_6, _1)         -> bb3
    //   bb3: _8 = &_3; _9 = pin_root(_8, _2)         -> bb4
    //   bb4: _10 = resize_ge(_1)                     -> bb5
    //   bb5: _11 = &_3; _12 = copy _5; _13 = get(_11, _12)   -> bb6
    //   bb6: _14 = &_3; _15 = copy _5;
    //        _16 = AddChecked(copy _15, 1); assert(!_16.1)   -> bb7
    //   bb7: _17 = move _16.0; _18 = get(_14, move _17)      -> bb8
    //   bb8: drop(_3)                                -> bb9
    //   bb9: return
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let field = |i: u64, f: u64| serde_json::json!({"kind": {"Projection": [place(i), {"Field": [null, f]}]}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow = |dest: u64, src: u64| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let overflow_check = |sum: u64, lhs: u64, target: u64| {
        serde_json::json!({"Assert": {
            "assert": {
                "cond": {"Move": field(sum, 1)},
                "expected": false,
                "check_kind": {"Overflow": [{"Add": "Wrap"}, {"Move": place(lhs)}, usize_lit(1)]}
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                2 => "pyre_object::gc_roots::{impl RootScope}::base",
                3 => "pyre_object::gc_roots::{impl RootScope}::pin_root",
                4 => "pyre_object::gc_roots::{impl RootScope}::get",
                _ => "pyre_object::listobject::ll_list_obj_resize_ge",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    let blocks = |second_pin: serde_json::Value, pin2_block: Vec<serde_json::Value>| {
        let mut blocks = vec![
            block(vec![], call(1, vec![], 3, 1)),
            block(vec![borrow(4, 3)], call(2, vec![copy(4)], 5, 2)),
            block(vec![borrow(6, 3)], call(3, vec![copy(6), copy(1)], 7, 3)),
            second_pin,
            block(vec![], call(5, vec![copy(1)], 10, 5)),
            block(
                vec![
                    borrow(11, 3),
                    assign(12, serde_json::json!({"Use": [copy(5), "No"]})),
                ],
                call(4, vec![copy(11), copy(12)], 13, 6),
            ),
            block(
                vec![
                    borrow(14, 3),
                    assign(15, serde_json::json!({"Use": [copy(5), "No"]})),
                    assign(
                        16,
                        serde_json::json!({"BinaryOp": ["AddChecked", copy(15), usize_lit(1)]}),
                    ),
                ],
                overflow_check(16, 15, 7),
            ),
            block(
                vec![assign(
                    17,
                    serde_json::json!({"Use": [{"Move": field(16, 0)}, "No"]}),
                )],
                call(
                    4,
                    vec![copy(14), serde_json::json!({"Move": place(17)})],
                    18,
                    8,
                ),
            ),
            block(vec![], drop_guard(3, 9)),
            block(vec![], serde_json::json!("Return")),
        ];
        blocks.extend(pin2_block);
        blocks
    };
    let body_of = |blocks: Vec<serde_json::Value>| -> Unstructured {
        serde_json::from_value(serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 2, "locals": (0..=20).map(local).collect::<Vec<_>>()},
            "body": blocks,
        }))
        .expect("fixture Unstructured parses")
    };
    let second_pin = || block(vec![borrow(8, 3)], call(3, vec![copy(8), copy(2)], 9, 4));

    let two_pins = body_of(blocks(second_pin(), vec![]));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &two_pins,
        &super::MovedOutLocals::with_set(&two_pins, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(plan.scopes.contains(3), "both pins run once, in one order");
    assert_eq!(plan.pins.get(&3), Some(&vec![1, 2]));
    assert_eq!(plan.get_sites, vec![(5usize, 1usize), (7usize, 2usize)]);
    for temp in [16, 17] {
        assert_eq!(
            plan.slot_temps.get(&temp),
            Some(&3),
            "`base + 1` temp _{temp}"
        );
    }
    for base in [5, 12, 15] {
        assert_eq!(
            plan.base_results.get(&base),
            Some(&3),
            "`base()` copy _{base}"
        );
    }

    // The artefact's opener unwinds to the body's `UnwindResume`.  That
    // edge runs only when opening fails, so it is not inside the bracket.
    let mut unwinding = body_of(blocks(second_pin(), vec![]));
    let resume = unwinding.body.len();
    unwinding
        .body
        .push(serde_json::from_value(block(vec![], serde_json::json!("UnwindResume"))).unwrap());
    let mut kind = unwinding.body[0].terminator.kind_value().clone();
    kind["Call"]["on_unwind"] = serde_json::json!(resume);
    unwinding.body[0].set_terminator_kind(kind);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &unwinding,
        &super::MovedOutLocals::with_set(&unwinding, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "the opener's own unwind edge must not keep the bracket"
    );

    // A call inside the bracket unwinds to that same `UnwindResume`
    // once cleanup blocks are stripped. Visiting it must not refuse
    // a bracket whose normal path is still balanced.
    let mut inner_unwinding = body_of(blocks(second_pin(), vec![]));
    let resume = inner_unwinding.body.len();
    inner_unwinding
        .body
        .push(serde_json::from_value(block(vec![], serde_json::json!("UnwindResume"))).unwrap());
    let mut kind = inner_unwinding.body[4].terminator.kind_value().clone();
    kind["Call"]["on_unwind"] = serde_json::json!(resume);
    inner_unwinding.body[4].set_terminator_kind(kind);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &inner_unwinding,
        &super::MovedOutLocals::with_set(&inner_unwinding, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(3),
        "a call's stripped unwind must not keep the bracket"
    );

    // With one pin, `base + 1` names a slot nothing in this bracket
    // filled.
    let one_pin = body_of(blocks(
        block(vec![], serde_json::json!({"Goto": {"target": 4}})),
        vec![],
    ));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &one_pin,
        &super::MovedOutLocals::with_set(&one_pin, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "a read past the last pin must keep its bracket"
    );

    // The second pin on only one path: `base + 1` holds it on that path
    // alone, so the read cannot be answered by its value.
    let branch = block(
        vec![],
        serde_json::json!({"Switch": {"discr": copy(1), "targets": {"If": [10, 4]}}}),
    );
    let maybe_pinned = body_of(blocks(branch, vec![second_pin()]));
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &maybe_pinned,
        &super::MovedOutLocals::with_set(&maybe_pinned, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "a pin that does not run on every path keeps its bracket"
    );

    // The resize is the one call that may collect.  Unproved, it keeps the
    // bracket.
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &two_pins,
        &super::MovedOutLocals::with_set(&two_pins, bit_set::BitSet::new()),
        name_of,
        |reg: &RegularCall| matches!(&reg.kind, CallKind::Fun(FunId::Regular { id: 5 })),
    );
    assert!(
        !plan.scopes.contains(3),
        "a callee that may change the root stack keeps the bracket"
    );
}

#[test]
fn root_stack_analysis_charges_free_pin_callees_to_the_bracket_around_them() {
    use majit_charon_reader::ullbc::{RegularCall, Unstructured};
    // `capture_set_items` and `w_type_set_bases` call the receiver-free
    // `gc_roots::pin_root` and return with the slot still pushed: the
    // enclosing bracket's close is what rewinds it. A bracket spanning
    // either one, directly or through a caller, must stay.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let borrow = |dest: u64, src: u64| {
        stmt(serde_json::json!({
            "Assign": [place(dest), {"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}]
        }))
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let body = |blocks: Vec<serde_json::Value>| {
        serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
            "body": blocks
        })
    };
    let fun = |def_id: u64, path: &[&str], body: Option<serde_json::Value>| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Tuple": []}
            },
            "body": match body {
                Some(b) => serde_json::json!({"Unstructured": b}),
                None => serde_json::json!("Missing"),
            }
        })
    };
    let ret = || block(vec![], serde_json::json!("Return"));
    // Ids 0..=4 are the root API, 5.. the callees under test.
    let funs = vec![
        fun(0, &["pyre_object", "gc_roots", "RootScope", "drop"], None),
        fun(1, &["pyre_object", "gc_roots", "push_roots"], None),
        fun(2, &["pyre_object", "gc_roots", "RootScope", "base"], None),
        fun(3, &["pyre_object", "gc_roots", "RootScope", "get"], None),
        fun(4, &["pyre_object", "gc_roots", "pin_root"], None),
        fun(
            5,
            &["pyre_object", "setobject", "capture_set_items"],
            Some(body(vec![
                block(vec![], call(4, vec![copy(1)], 2, 1)),
                ret(),
            ])),
        ),
        fun(
            6,
            &["pyre_object", "typeobject", "w_type_set_bases"],
            Some(body(vec![
                block(vec![], call(4, vec![copy(1)], 2, 1)),
                block(vec![], call(8, vec![], 3, 2)),
                ret(),
            ])),
        ),
        // Reaches the free pin only through its callee.
        fun(
            7,
            &["pyre_object", "setobject", "w_set_insert_key_checked"],
            Some(body(vec![
                block(vec![], call(5, vec![copy(1)], 2, 1)),
                ret(),
            ])),
        ),
        fun(
            8,
            &["pyre_object", "typeobject", "type_write_barrier"],
            Some(body(vec![ret()])),
        ),
    ];
    let llbc = llbc_with_types("pyre_object", vec![], funs);
    let state = super::RootStackState::new(&llbc);
    let analyzer = super::RootStackAnalyzer::new(&llbc, &state);
    let direct = |id: u64| -> RegularCall {
        serde_json::from_value(serde_json::json!({"kind": {"Fun": id}, "generics": null}))
            .expect("fixture call parses")
    };
    for (id, name) in [
        (5, "capture_set_items"),
        (6, "w_type_set_bases"),
        (7, "a caller of capture_set_items"),
    ] {
        assert!(
            analyzer.regular_call_touches_root_stack(&direct(id)),
            "{name} leaves a pin for its caller's bracket to rewind"
        );
    }
    assert!(!analyzer.regular_call_touches_root_stack(&direct(8)));

    //   bb0: _2 = push_roots()          -> bb1
    //   bb1: _3 = &_2; _4 = base(_3)    -> bb2
    //   bb2: _9 = callee(_1)            -> bb3
    //   bb3: drop(_2)                   -> bb4
    //   bb4: return
    let spanning = |callee: u64| -> Unstructured {
        serde_json::from_value(body(vec![
            block(vec![], call(1, vec![], 2, 1)),
            block(vec![borrow(3, 2)], call(2, vec![copy(3)], 4, 2)),
            block(vec![], call(callee, vec![copy(1)], 9, 3)),
            block(vec![], drop_guard(2, 4)),
            ret(),
        ]))
        .expect("fixture Unstructured parses")
    };
    let name_of = |reg: &RegularCall| super::regular_call_name_path(reg, &llbc);
    let touches = |reg: &RegularCall| analyzer.regular_call_touches_root_stack(reg);
    for callee in [5, 6, 7] {
        let body = spanning(callee);
        let plan = super::analyze_root_brackets_with(
            &fixture_llbc(),
            &body,
            &super::MovedOutLocals::with_set(&body, bit_set::BitSet::new()),
            name_of,
            touches,
        );
        assert!(
            !plan.scopes.contains(2),
            "a bracket around free-pin callee {callee} must stay"
        );
    }
    let body = spanning(8);
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &body,
        &super::MovedOutLocals::with_set(&body, bit_set::BitSet::new()),
        name_of,
        touches,
    );
    assert!(
        plan.scopes.contains(2),
        "a bracket around a callee that touches no slot is erased"
    );
}

#[test]
fn len_names_next_pin_counts_only_pins_that_already_ran() {
    let set = |xs: &[usize]| -> bit_set::BitSet { xs.iter().copied().collect() };
    let mut dom = std::collections::HashMap::new();
    // The pin shares the len's block. Both are terminators, so that pin
    // is not after the len, and no later pin exists.
    dom.insert(1, set(&[1]));
    assert_eq!(super::len_names_next_pin(1, &[(1, 9)], &dom), None);

    // Pins at 0 and 2, len at 1: one pin has already run, and the next
    // pin dominates the len.
    dom.clear();
    dom.insert(0, set(&[0]));
    dom.insert(1, set(&[0, 1]));
    dom.insert(2, set(&[0, 1, 2]));
    assert_eq!(
        super::len_names_next_pin(1, &[(0, 10), (2, 11)], &dom),
        Some(1)
    );

    // A pin on a side path is not ordered against the len.
    dom.clear();
    dom.insert(1, set(&[1]));
    dom.insert(5, set(&[5]));
    assert_eq!(super::len_names_next_pin(1, &[(5, 9)], &dom), None);
}

#[test]
fn root_bracket_erasure_keeps_an_unnameable_read_back() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // A `shadow_stack_get` whose index is not this scope's base, a
    // `base + k`, or a `len + k` this pass can name keeps every op:
    // answering it would invent a pin.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                6 => "pyre_object::gc_roots::pin_root",
                7 => "pyre_object::gc_roots::shadow_stack_get",
                8 => "pyre_object::gc_roots::shadow_stack_len",
                _ => "pyre_interpreter::misc::as_long",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "span": span(),
        "locals": {"arg_count": 2, "locals": (0..=12).map(local).collect::<Vec<_>>()},
        "body": [
            block(vec![], call(1, vec![], 3, 1)),
            block(vec![], call(8, vec![], 4, 2)),
            block(vec![], call(6, vec![copy(1)], 5, 3)),
            block(vec![], call(7, vec![copy(9)], 6, 4)),
            block(vec![], drop_guard(3, 5)),
            block(vec![], serde_json::json!("Return")),
        ],
    }))
    .expect("fixture Unstructured parses");
    let plan = super::analyze_root_brackets_with(
        &fixture_llbc(),
        &body,
        &super::MovedOutLocals::with_set(&body, bit_set::BitSet::new()),
        &name_of,
        &touches,
    );
    assert!(
        !plan.scopes.contains(3),
        "a get at a slot this pass cannot name keeps the bracket"
    );
    assert!(plan.get_sites.is_empty());
    assert!(plan.free_sites.is_empty() || !plan.scopes.contains(3));
}

#[test]
fn root_bracket_erasure_answers_a_closure_get_of_a_captured_slot() {
    use majit_charon_reader::ullbc::{CallKind, FunId, RegularCall, Unstructured};
    // A closure whose body is only `shadow_stack_get` of a captured
    // base or len, plus a constant, is the read `dunder_import_inner`
    // spells for `reload_import`. The call answers with that slot.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let project = |base: serde_json::Value, elem: serde_json::Value| serde_json::json!({"kind": {"Projection": [base, elem]}, "ty": ty()});
    let deref = |base: serde_json::Value| project(base, serde_json::json!("Deref"));
    let field =
        |base: serde_json::Value, f: u64| project(base, serde_json::json!({"Field": [null, f]}));
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let mv = |i: u64| serde_json::json!({"Move": place(i)});
    let usize_lit = |k: u64| serde_json::json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let assign = |dest: u64, rvalue: serde_json::Value| {
        stmt(serde_json::json!({"Assign": [place(dest), rvalue]}))
    };
    let borrow = |dest: u64, src: u64| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let borrow_place = |dest: u64, src: serde_json::Value| {
        assign(
            dest,
            serde_json::json!({"Ref": {"place": src, "kind": "Shared", "ptr_metadata": null}}),
        )
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_place = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let use_op = |op: serde_json::Value| serde_json::json!({"Use": [op, "No"]});
    let ret = || block(vec![], serde_json::json!("Return"));
    let capture_blocks = || vec![block(vec![], call(7, vec![copy(1)], 0, 1)), ret()];
    let plus_blocks = || {
        vec![
            block(
                vec![
                    assign(3, use_op(serde_json::json!({"Move": field(place(2), 0)}))),
                    assign(
                        5,
                        use_op(serde_json::json!({"Copy": deref(field(deref(place(1)), 0))})),
                    ),
                    assign(6, use_op(copy(3))),
                    assign(
                        7,
                        serde_json::json!({"BinaryOp": ["AddChecked", copy(5), copy(6)]}),
                    ),
                ],
                serde_json::json!({"Assert": {
                    "assert": {
                        "cond": {"Move": field(place(7), 1)},
                        "expected": false,
                        "check_kind": {"Overflow": [{"Add": "Wrap"}, copy(5), copy(6)]}
                    },
                    "target": 1,
                    "on_unwind": 99
                }}),
            ),
            block(
                vec![assign(
                    4,
                    use_op(serde_json::json!({"Move": field(place(7), 0)})),
                )],
                call(7, vec![mv(4)], 0, 2),
            ),
            ret(),
        ]
    };
    let fun = |def_id: u64, segs: &[&str], arg_count: u64, blocks: Vec<serde_json::Value>| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": segs.iter().map(|s| serde_json::json!({"Ident": [*s, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Tuple": []}
            },
            "body": {"Unstructured": {
                "span": span(),
                "locals": {"arg_count": arg_count, "locals": (0..=8).map(local).collect::<Vec<_>>()},
                "body": blocks
            }}
        })
    };
    let mut plus = fun(8, &["fixture", "reload", "call"], 2, plus_blocks());
    // An unknown field stays in the file bytes, which is where dedup
    // constants are collected. Id 1 is the `usize` 1 the second call adds.
    plus["const_probe"] = serde_json::json!({
        "Value": [1, [{"Integer": {"Unsigned": ["Usize", "1"]}}, {"Tuple": []}]]
    });
    let mut funs = vec![serde_json::Value::Null; 8];
    funs.push(plus);
    funs.push(fun(
        9,
        &["fixture", "len_reload", "call"],
        2,
        capture_blocks(),
    ));
    funs.push(fun(
        10,
        &["fixture", "not_a_getter", "call"],
        2,
        vec![
            block(vec![], call(7, vec![copy(1)], 0, 1)),
            block(vec![], call(7, vec![copy(1)], 0, 2)),
            ret(),
        ],
    ));
    funs.push(fun(
        11,
        &["fixture", "self_add", "call"],
        2,
        vec![
            block(
                vec![assign(
                    3,
                    serde_json::json!({"BinaryOp": [
                        "AddChecked",
                        serde_json::json!({"Copy": deref(place(1))}),
                        serde_json::json!({"Copy": deref(place(1))})
                    ]}),
                )],
                call(7, vec![copy(3)], 0, 1),
            ),
            ret(),
        ],
    ));
    funs.push(fun(
        12,
        &["fixture", "one_arg", "call"],
        1,
        capture_blocks(),
    ));
    funs.push(fun(
        13,
        &["fixture", "foreign_get", "call"],
        2,
        vec![block(vec![], call(2, vec![copy(1)], 0, 1)), ret()],
    ));
    let call_unwind = |dest: u64, target: u64, on_unwind: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": 7}, "generics": null}},
                "args": [copy(1)],
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": on_unwind
        }})
    };
    funs.push(fun(
        14,
        &["fixture", "discarded_get", "call"],
        2,
        vec![block(vec![], call(7, vec![copy(1)], 3, 1)), ret()],
    ));
    funs.push(fun(
        15,
        &["fixture", "return_skips_get", "call"],
        2,
        vec![ret(), block(vec![], call(7, vec![copy(1)], 0, 2)), ret()],
    ));
    funs.push(fun(
        16,
        &["fixture", "getter_with_unwind", "call"],
        2,
        vec![
            block(vec![], call_unwind(0, 1, 2)),
            ret(),
            block(vec![], serde_json::json!("UnwindResume")),
        ],
    ));
    funs.push(fun(
        17,
        &["fixture", "return_on_unwind", "call"],
        2,
        vec![
            block(vec![], call_unwind(0, 1, 2)),
            block(vec![], serde_json::json!("UnwindResume")),
            ret(),
        ],
    ));
    let llbc = llbc_with_types("fixture", vec![], funs);
    let name_of = |reg: &RegularCall| -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        Some(
            match id {
                1 => "pyre_object::gc_roots::push_roots",
                2 => "fixture::shadow_stack_get",
                3 => "pyre_object::gc_roots::shadow_stack_len",
                4 => "pyre_object::gc_roots::pin_root",
                6 => "pyre_object::gc_roots::pin_roots",
                7 => "pyre_object::gc_roots::shadow_stack_get",
                8 => "fixture::reload::call",
                9 => "fixture::len_reload::call",
                10 => "fixture::not_a_getter::call",
                _ => "fixture::other::call",
            }
            .to_string(),
        )
    };
    let touches = |_: &RegularCall| false;
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 8, &name_of),
        Some(super::RootSlotGetter::CapturePlusArg)
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 9, &name_of),
        Some(super::RootSlotGetter::Capture)
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 10, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 11, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 12, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 13, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 1, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 14, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 15, &name_of),
        None
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 16, &name_of),
        Some(super::RootSlotGetter::Capture)
    );
    assert_eq!(
        super::function_is_root_slot_getter(&llbc, 17, &name_of),
        None
    );

    let assemble = |steps: Vec<(Vec<serde_json::Value>, serde_json::Value)>| -> Unstructured {
        let mut blocks = Vec::new();
        for (i, (stmts, mut term)) in steps.into_iter().enumerate() {
            let target = (i + 1) as u64;
            if let Some(call) = term.get_mut("Call") {
                call["target"] = serde_json::json!(target);
            } else if let Some(drop) = term.get_mut("Drop") {
                drop["target"] = serde_json::json!(target);
            }
            blocks.push(block(stmts, term));
        }
        let drop_at = blocks.len();
        blocks.push(block(vec![], drop_place(3, (drop_at + 1) as u64)));
        blocks.push(block(vec![], serde_json::json!("Return")));
        serde_json::from_value(serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 2, "locals": (0..=40).map(local).collect::<Vec<_>>()},
            "body": blocks,
        }))
        .expect("fixture Unstructured parses")
    };
    let plan_of = |body: &Unstructured| {
        super::analyze_root_brackets_with(
            &llbc,
            body,
            &super::MovedOutLocals::with_set(body, bit_set::BitSet::new()),
            &name_of,
            &touches,
        )
    };
    let pin_slice = |elems: Vec<serde_json::Value>| {
        let n = elems.len() as u64;
        vec![
            assign(19, use_op(copy(1))),
            assign(20, use_op(copy(2))),
            assign(
                6,
                serde_json::json!({"Aggregate": [{"Array": [ty(), usize_lit(n)]}, elems]}),
            ),
            borrow(7, 6),
            borrow_place(
                8,
                serde_json::json!({"kind": {"Projection": [place(7), "Deref"]}, "ty": ty()}),
            ),
            assign(
                9,
                serde_json::json!({"UnaryOp": [
                    {"Cast": {"Unsize": [ty(), ty(), {"Length": usize_lit(n)}]}},
                    mv(8)
                ]}),
            ),
        ]
    };
    let closure_of = |base: u64, borrow_tmp: u64, closure: u64, recv: u64| {
        vec![
            borrow(borrow_tmp, base),
            assign(
                closure,
                serde_json::json!({"Aggregate": [{"Adt": [3026, null]}, [mv(borrow_tmp)]]}),
            ),
            borrow(recv, closure),
        ]
    };
    let tuple = |k: serde_json::Value| serde_json::json!({"Aggregate": [{"Adt": [7, null]}, [k]]});
    let unit = || serde_json::json!({"Aggregate": [{"Adt": [0, null]}, []]});
    let dedup_one = || serde_json::json!({"Const": {"Deduplicated": 1}});

    let happy = assemble(vec![
        (vec![], call(1, vec![], 3, 0)),
        (pin_slice(vec![mv(19), mv(20)]), call(6, vec![mv(9)], 5, 0)),
        (
            {
                let mut stmts = closure_of(5, 23, 22, 24);
                stmts.push(assign(25, tuple(usize_lit(0))));
                stmts
            },
            call(8, vec![copy(24), copy(25)], 30, 0),
        ),
        (
            vec![assign(27, tuple(dedup_one()))],
            call(8, vec![copy(24), copy(27)], 31, 0),
        ),
        (vec![], drop_place(22, 0)),
    ]);
    let plan = plan_of(&happy);
    assert!(
        plan.scopes.contains(3),
        "a closure get of base + k is the pin_roots slot"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![19, 20]));
    assert_eq!(plan.get_sites, vec![(2usize, 19usize), (3usize, 20usize)]);
    assert_eq!(plan.base_results.get(&5), Some(&3));
    for temp in [6usize, 7, 8, 9] {
        assert_eq!(plan.pin_temps.get(&temp), Some(&3), "slice temp _{temp}");
    }
    for temp in [22usize, 23, 24, 25, 27] {
        assert_eq!(
            plan.capture_temps.get(&temp),
            Some(&3),
            "capture temp _{temp}"
        );
    }
    assert!(
        plan.capture_temps.get(&5).is_none(),
        "the captured base is the index, not a temp"
    );
    assert_eq!(plan.free_sites.get(&1), Some(&3));

    // Addend 1 with a single pin names a slot this bracket did not fill.
    let one_pin = assemble(vec![
        (vec![], call(1, vec![], 3, 0)),
        (pin_slice(vec![mv(19)]), call(6, vec![mv(9)], 5, 0)),
        (
            {
                let mut stmts = closure_of(5, 23, 22, 24);
                stmts.push(assign(25, tuple(usize_lit(1))));
                stmts
            },
            call(8, vec![copy(24), copy(25)], 30, 0),
        ),
        (vec![], drop_place(22, 0)),
    ]);
    let plan = plan_of(&one_pin);
    assert!(
        !plan.scopes.contains(3),
        "base + 1 with one pin keeps the bracket"
    );

    // A copy of the closure is a use the erased get would not perform.
    let escaped = assemble(vec![
        (vec![], call(1, vec![], 3, 0)),
        (pin_slice(vec![mv(19), mv(20)]), call(6, vec![mv(9)], 5, 0)),
        (
            {
                let mut stmts = closure_of(5, 23, 22, 24);
                stmts.push(assign(25, tuple(usize_lit(0))));
                stmts.push(assign(40, use_op(copy(22))));
                stmts
            },
            call(8, vec![copy(24), copy(25)], 30, 0),
        ),
        (
            vec![assign(27, tuple(dedup_one()))],
            call(8, vec![copy(24), copy(27)], 31, 0),
        ),
        (vec![], drop_place(22, 0)),
    ]);
    let plan = plan_of(&escaped);
    assert!(
        !plan.scopes.contains(3),
        "a closure that escapes keeps the bracket"
    );

    // A second call in the callee is not a read this pass can answer.
    let extra_call = assemble(vec![
        (vec![], call(1, vec![], 3, 0)),
        (pin_slice(vec![mv(19), mv(20)]), call(6, vec![mv(9)], 5, 0)),
        (
            {
                let mut stmts = closure_of(5, 23, 22, 24);
                stmts.push(assign(25, tuple(usize_lit(0))));
                stmts
            },
            call(10, vec![copy(24), copy(25)], 30, 0),
        ),
    ]);
    let plan = plan_of(&extra_call);
    assert!(
        !plan.scopes.contains(3),
        "a callee that is not only shadow_stack_get keeps the bracket"
    );

    // `shadow_stack_len` captured by a closure, then `pin_root`, then
    // `get(len)`. The len names the pin that follows it.
    let len_capture = assemble(vec![
        (vec![], call(1, vec![], 3, 0)),
        (vec![], call(3, vec![], 4, 0)),
        (vec![], call(4, vec![copy(1)], 5, 0)),
        (
            {
                let mut stmts = closure_of(4, 10, 11, 12);
                stmts.push(assign(13, unit()));
                stmts
            },
            call(9, vec![copy(12), copy(13)], 14, 0),
        ),
        (vec![], drop_place(11, 0)),
    ]);
    let plan = plan_of(&len_capture);
    assert!(
        plan.scopes.contains(3),
        "a closure get of the len is the next pin"
    );
    assert_eq!(plan.pins.get(&3), Some(&vec![1]));
    assert_eq!(plan.get_sites, vec![(3usize, 1usize)]);
    assert_eq!(plan.slot_temps.get(&4), Some(&3), "the len result");
    assert!(
        plan.slot_temps.get(&5).is_none(),
        "pin_root's result is not a depth"
    );
    for temp in [10usize, 11, 12, 13] {
        assert_eq!(
            plan.capture_temps.get(&temp),
            Some(&3),
            "len capture temp _{temp}"
        );
    }
    assert!(plan.capture_temps.get(&4).is_none(), "the len is the index");
    assert_eq!(plan.free_sites.get(&1), Some(&3), "the len call");
    assert_eq!(plan.free_sites.get(&2), Some(&3), "the pin call");
}

#[test]
fn root_stack_analysis_does_not_merge_a_cycle_closed_inside_a_bracket() {
    use majit_charon_reader::ullbc::RegularCall;
    // `outer` calls `inner` only inside a bracket it opens and closes, so
    // `inner`'s pin never reaches `outer`'s caller.  `inner` calls back
    // into `outer` and then pins for its own caller.  The two form a call
    // cycle, but only `inner` touches the stack: asking about `inner`
    // first must not leave `outer` with `inner`'s answer.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let drop_guard = |local: u64, target: u64| {
        serde_json::json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |terminator: serde_json::Value| serde_json::json!({"statements": [], "kind": "Regular", "terminator": {"kind": terminator}});
    let body = |blocks: Vec<serde_json::Value>| {
        serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
            "body": blocks
        })
    };
    let fun = |def_id: u64, path: &[&str], body: Option<serde_json::Value>| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Tuple": []}
            },
            "body": match body {
                Some(b) => serde_json::json!({"Unstructured": b}),
                None => serde_json::json!("Missing"),
            }
        })
    };
    let ret = || block(serde_json::json!("Return"));
    let funs = vec![
        fun(0, &["pyre_object", "gc_roots", "RootScope", "drop"], None),
        fun(1, &["pyre_object", "gc_roots", "push_roots"], None),
        fun(2, &["pyre_object", "gc_roots", "pin_root"], None),
        //   bb0: _2 = push_roots() -> bb1
        //   bb1: _3 = inner(_1)    -> bb2
        //   bb2: drop(_2)          -> bb3
        //   bb3: return
        fun(
            3,
            &["pyre_object", "fixture", "outer"],
            Some(body(vec![
                block(call(1, vec![], 2, 1)),
                block(call(4, vec![copy(1)], 3, 2)),
                block(drop_guard(2, 3)),
                ret(),
            ])),
        ),
        //   bb0: _2 = outer(_1)    -> bb1
        //   bb1: _3 = pin_root(_1) -> bb2
        //   bb2: return
        fun(
            4,
            &["pyre_object", "fixture", "inner"],
            Some(body(vec![
                block(call(3, vec![copy(1)], 2, 1)),
                block(call(2, vec![copy(1)], 3, 2)),
                ret(),
            ])),
        ),
    ];
    let llbc = llbc_with_types("pyre_object", vec![], funs);
    let direct = |id: u64| -> RegularCall {
        serde_json::from_value(serde_json::json!({"kind": {"Fun": id}, "generics": null}))
            .expect("fixture call parses")
    };
    let root_state = super::RootStackState::new(&llbc);
    let analyzer = super::RootStackAnalyzer::new(&llbc, &root_state);
    assert!(
        analyzer.regular_call_touches_root_stack(&direct(4)),
        "inner leaves a pin for its caller"
    );
    assert!(
        !analyzer.regular_call_touches_root_stack(&direct(3)),
        "outer closes every pin inner leaves, whichever is asked first"
    );
}

#[test]
fn root_stack_analysis_charges_no_unknown_body_in_an_artefact_below_the_api() {
    use majit_charon_reader::ullbc::RegularCall;
    // An artefact that names no part of the root-stack API sits below
    // it: a local body Charon left opaque, or a trait method no body in
    // the artefact implements, reaches the stack only through a callback
    // a crate above supplies.  Once the artefact names the API, the same
    // unknowns may be its helpers.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    };
    let block = |terminator: serde_json::Value| serde_json::json!({"statements": [], "kind": "Regular", "terminator": {"kind": terminator}});
    let body = |blocks: Vec<serde_json::Value>| {
        serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
            "body": blocks
        })
    };
    let fun = |def_id: u64, path: &[&str], body: serde_json::Value| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Tuple": []}
            },
            "body": body
        })
    };
    let artefact = |krate: &str, names_api: bool| {
        let mut funs = vec![
            fun(
                0,
                &[krate, "fixture", "opaque"],
                serde_json::json!("Opaque"),
            ),
            //   bb0: _2 = opaque(_1) -> bb1
            //   bb1: return
            fun(
                1,
                &[krate, "fixture", "caller"],
                serde_json::json!({"Unstructured": body(vec![
                    block(call(0, vec![copy(1)], 2, 1)),
                    block(serde_json::json!("Return")),
                ])}),
            ),
            // A required trait method: a declaration with no body.
            fun(
                2,
                &[krate, "fixture", "Hook", "run"],
                serde_json::json!("Missing"),
            ),
        ];
        if names_api {
            funs.push(fun(
                3,
                &[krate, "gc_roots", "push_roots"],
                serde_json::json!("Missing"),
            ));
        }
        llbc_with_types(krate, vec![], funs)
    };
    let direct: RegularCall =
        serde_json::from_value(serde_json::json!({"kind": {"Fun": 1}, "generics": null}))
            .expect("fixture call parses");
    let method: RegularCall = serde_json::from_value(
        serde_json::json!({"kind": {"Trait": [{}, 0, 2]}, "generics": null}),
    )
    .expect("fixture call parses");

    let below = artefact("majit_rlib", false);
    let root_state = super::RootStackState::new(&below);
    let analyzer = super::RootStackAnalyzer::new(&below, &root_state);
    assert!(
        !analyzer.regular_call_touches_root_stack(&direct),
        "an opaque local body below the API cannot reach the stack"
    );
    assert!(
        !analyzer.regular_call_touches_root_stack(&method),
        "an unimplemented trait method below the API cannot reach the stack"
    );

    let above = artefact("pyre_object", true);
    let root_state = super::RootStackState::new(&above);
    let analyzer = super::RootStackAnalyzer::new(&above, &root_state);
    assert!(analyzer.regular_call_touches_root_stack(&direct));
    assert!(analyzer.regular_call_touches_root_stack(&method));
}

#[test]
fn an_explicit_drop_of_a_root_guard_is_its_bracket_close() {
    use majit_charon_reader::ullbc::{PlaceKind, RegularCall, TermKind};
    // `let g = push_roots(); pin_root(a); drop(g);` spells the close as a
    // call Charon writes as `_4 = move _2; core::mem::drop(move _4)`,
    // and keeps the scope-end `Drop _2` a moved-out local no longer runs.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy = |i: u64| serde_json::json!({"Copy": place(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let move_into = |dest: u64, src: u64| {
        stmt(serde_json::json!({"Assign": [place(dest), {"Use": [{"Move": place(src)}, "No"]}]}))
    };
    let call = |id: u64, args: Vec<serde_json::Value>, dest: u64, target: u64, unwind: u64| {
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": id}, "generics": null}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": unwind
        }})
    };
    let drop_guard = |local: u64, target: u64, unwind: u64| {
        serde_json::json!({"Drop": {
            "kind": "Conditional",
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": unwind
        }})
    };
    let block = |statements: Vec<serde_json::Value>, terminator: serde_json::Value| serde_json::json!({"statements": statements, "terminator": {"kind": terminator}});
    let body = |blocks: Vec<serde_json::Value>| {
        serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 1, "locals": (0..=9).map(local).collect::<Vec<_>>()},
            "body": blocks
        })
    };
    let fun = |def_id: u64, path: &[&str], body: Option<serde_json::Value>| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Tuple": []}
            },
            "body": match body {
                Some(b) => serde_json::json!({"Unstructured": b}),
                None => serde_json::json!("Missing"),
            }
        })
    };
    let funs = vec![
        fun(
            0,
            &["pyre_object", "gc_roots", "RootScope", "drop_in_place"],
            None,
        ),
        fun(1, &["pyre_object", "gc_roots", "push_roots"], None),
        fun(2, &["pyre_object", "gc_roots", "pin_root"], None),
        fun(3, &["core", "mem", "drop"], None),
        //   bb0: _2 = push_roots()                      -> bb1
        //   bb1: _3 = pin_root(_1)                      -> bb2
        //   bb2: _4 = move _2; _5 = drop(move _4)       -> bb3, unwind bb6
        //   bb3: drop(_2)                               -> bb7
        //   bb4: drop(_2)                               -> bb5
        //   bb5: resume
        //   bb6: drop(_4)                               -> bb4
        //   bb7: return
        fun(
            4,
            &["pyre_object", "fixture", "operands"],
            Some(body(vec![
                block(vec![], call(1, vec![], 2, 1, 5)),
                block(vec![], call(2, vec![copy(1)], 3, 2, 4)),
                block(
                    vec![move_into(4, 2)],
                    call(3, vec![serde_json::json!({"Move": place(4)})], 5, 3, 6),
                ),
                block(vec![], drop_guard(2, 7, 5)),
                block(vec![], drop_guard(2, 5, 5)),
                block(vec![], serde_json::json!("UnwindResume")),
                block(vec![], drop_guard(4, 4, 5)),
                block(vec![], serde_json::json!("Return")),
            ])),
        ),
        // The same close on one arm only: the join's `Drop _2` runs on
        // the other arm, so it is neither the close nor dead.
        //   bb0: _2 = push_roots()                      -> bb1
        //   bb1: if _1 { bb2 } else { bb3 }
        //   bb2: _4 = move _2; _5 = drop(move _4)       -> bb3
        //   bb3: drop(_2)                               -> bb4
        //   bb4: return
        fun(
            5,
            &["pyre_object", "fixture", "one_arm"],
            Some(body(vec![
                block(vec![], call(1, vec![], 2, 1, 4)),
                block(
                    vec![],
                    serde_json::json!({"Switch": {"discr": copy(1), "targets": {"If": [2, 3]}}}),
                ),
                block(
                    vec![move_into(4, 2)],
                    call(3, vec![serde_json::json!({"Move": place(4)})], 5, 3, 4),
                ),
                block(vec![], drop_guard(2, 4, 4)),
                block(vec![], serde_json::json!("Return")),
            ])),
        ),
    ];
    let llbc = llbc_with_types("pyre_object", vec![], funs);
    let name_of = |reg: &RegularCall| super::regular_call_name_path(reg, &llbc);
    let drop_place =
        |u: &majit_charon_reader::ullbc::Unstructured, bb: usize| match u.body[bb].term(&llbc) {
            Ok(TermKind::Drop { place, target, .. }) => match place.kind {
                PlaceKind::Local(l) => Some((l, target)),
                _ => None,
            },
            _ => None,
        };
    let mut u = llbc.fn_by_id(4).unwrap().unstructured().unwrap();
    super::elaborate_explicit_root_closes(&llbc, &mut u, &name_of);
    // The temporary's only move is the `mem::drop` operand, and that
    // move is not a move-out. The close stays a call.
    assert_eq!(
        u.body[2].statements.len(),
        1,
        "the move into the call stays"
    );
    assert!(matches!(u.body[2].term(&llbc), Ok(TermKind::Call { .. })));
    assert_eq!(drop_place(&u, 3), Some((2, 7)));
    assert_eq!(drop_place(&u, 6), Some((4, 4)));
    assert_eq!(drop_place(&u, 4), Some((2, 5)));
    assert!(super::moved_out_locals(&u, &llbc).contains(2));

    let mut u = llbc.fn_by_id(5).unwrap().unstructured().unwrap();
    super::elaborate_explicit_root_closes(&llbc, &mut u, &name_of);
    assert_eq!(
        u.body[2].statements.len(),
        1,
        "a guard closed on one arm is left"
    );
    assert!(matches!(u.body[2].term(&llbc), Ok(TermKind::Call { .. })));
    assert_eq!(drop_place(&u, 3), Some((2, 4)));

    // The bracket closes every pin `operands` makes, so its caller sees
    // no change to the root stack.
    let root_state = super::RootStackState::new(&llbc);
    let analyzer = super::RootStackAnalyzer::new(&llbc, &root_state);
    let direct: RegularCall =
        serde_json::from_value(serde_json::json!({"kind": {"Fun": 4}, "generics": null}))
            .expect("fixture call parses");
    assert!(!analyzer.regular_call_touches_root_stack(&direct));
}

fn identity_item_meta(name: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "span": {"data": {
            "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
        }},
        "source_text": null,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
        "is_local": true
    })
}

fn identity_instantiated_name(
    crate_name: &str,
    item: &str,
    types: serde_json::Value,
    consts: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!([
        {"Ident": [crate_name, 0]},
        {"Ident": [item, 0]},
        {"Instantiated": {"params": {}, "kind": "Other", "skip_binder": {
            "regions": [],
            "types": types,
            "const_generics": consts,
            "trait_refs": []
        }}}
    ])
}

fn identity_plain_name(crate_name: &str, module: &str, item: &str) -> serde_json::Value {
    serde_json::json!([
        {"Ident": [crate_name, 0]},
        {"Ident": [module, 0]},
        {"Ident": [item, 0]}
    ])
}

fn identity_crate_item(crate_name: &str, item: &str) -> serde_json::Value {
    serde_json::json!([
        {"Ident": [crate_name, 0]},
        {"Ident": [item, 0]}
    ])
}

fn identity_bare_item(item: &str) -> serde_json::Value {
    serde_json::json!([{"Ident": [item, 0]}])
}

fn identity_unqualified_instantiated_name(
    item: &str,
    types: serde_json::Value,
    consts: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!([
        {"Ident": [item, 0]},
        {"Instantiated": {"params": {}, "kind": "Other", "skip_binder": {
            "regions": [],
            "types": types,
            "const_generics": consts,
            "trait_refs": []
        }}}
    ])
}

fn identity_fn_ptr_ty(
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": inputs,
                "output": output,
                "is_unsafe": false
            }
        }
    })
}

fn identity_char_const(value: &str) -> serde_json::Value {
    serde_json::json!([[
        {"Char": value},
        {"Scalar": "Char"}
    ]])
}

fn identity_signed(kind: &str) -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Signed": kind}}})
}

fn identity_usize_const(value: &str) -> serde_json::Value {
    serde_json::json!([[
        {"Integer": {"Unsigned": ["Usize", value]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]])
}

fn identity_struct_decl(
    def_id: u64,
    name: serde_json::Value,
    field_name: &str,
    field_ty: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": identity_item_meta(name),
        "kind": {"Struct": [{
            "name": field_name,
            "ty": field_ty,
            "attr_info": null
        }]}
    })
}

fn identity_opaque_decl(def_id: u64, name: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": identity_item_meta(name),
        "kind": "Opaque"
    })
}

fn identity_fun(def_id: u64, name: &str, output: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": identity_item_meta(serde_json::json!([
            {"Ident": ["fixture", 0]},
            {"Ident": [name, 0]}
        ])),
        "signature": {"is_unsafe": false, "inputs": [], "output": output},
        "body": null
    })
}

fn identity_adt_use(
    def_id: u64,
    types: serde_json::Value,
    consts: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": def_id,
            "generics": {
                "regions": [],
                "types": types,
                "const_generics": consts,
                "trait_refs": []
            }
        }
    })
}

fn published_field_ty<'a>(llbc: &'a Llbc, id: u64, field: &str) -> Option<&'a serde_json::Value> {
    let td = llbc.type_by_id(id)?;
    match &td.kind {
        majit_charon_reader::ullbc::TypeDeclKind::Struct(fields) => fields
            .iter()
            .find(|row| row.name.as_deref() == Some(field))
            .and_then(|row| match &row.ty {
                TyRef::Other(value) => Some(value),
                _ => None,
            }),
        _ => None,
    }
}

fn published_field_adt_id<'a>(
    llbc: &'a Llbc,
    id: u64,
    field: &str,
) -> Option<&'a serde_json::Value> {
    published_field_ty(llbc, id, field)?.get("Adt")?.get("id")
}

fn dest_kind_is_opaque(llbc: &Llbc, id: u64) -> bool {
    matches!(
        llbc.type_by_id(id).map(|td| &td.kind),
        Some(majit_charon_reader::ullbc::TypeDeclKind::Opaque)
    )
}

/// Two monomorphizations of one generic struct keep their own field
/// types. A crate-stripped suffix is used only when it is unique.
#[test]
fn published_structs_key_on_instantiated_identity() {
    let i32_ty = identity_signed("I32");
    let i64_ty = identity_signed("I64");
    let defining = llbc_with_types(
        "crate_a",
        vec![
            identity_struct_decl(
                0,
                identity_instantiated_name(
                    "crate_a",
                    "Container",
                    serde_json::json!([i32_ty.clone()]),
                    serde_json::json!([]),
                ),
                "payload",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                1,
                identity_instantiated_name(
                    "crate_a",
                    "Container",
                    serde_json::json!([i64_ty.clone()]),
                    serde_json::json!([]),
                ),
                "payload",
                i64_ty.clone(),
            ),
            identity_struct_decl(
                2,
                identity_plain_name("crate_a", "mod", "Unique"),
                "mark",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                3,
                identity_plain_name("crate_a", "mod", "Shared"),
                "from_a",
                i32_ty.clone(),
            ),
        ],
        vec![],
    );
    let other = llbc_with_types(
        "crate_b",
        vec![identity_struct_decl(
            0,
            identity_plain_name("crate_b", "mod", "Shared"),
            "from_b",
            i64_ty.clone(),
        )],
        vec![],
    );
    let mut rows = super::harvest_published_structs(&defining);
    rows.extend(super::harvest_published_structs(&other));
    assert!(
        rows.iter()
            .any(|row| row.path.contains("Container") && row.path.contains("i32")),
        "i32 instantiation must keep its own identity: {:?}",
        rows.iter().map(|row| &row.path).collect::<Vec<_>>()
    );
    assert!(
        rows.iter()
            .any(|row| row.path.contains("Container") && row.path.contains("i64")),
        "i64 instantiation must keep its own identity: {:?}",
        rows.iter().map(|row| &row.path).collect::<Vec<_>>()
    );
    assert!(
        !rows.iter().any(|row| row.path == "Container"
            || row.path == "mod::Unique"
            || row.path == "mod::Shared"),
        "harvest must not emit a crate-stripped alias row: {:?}",
        rows.iter().map(|row| &row.path).collect::<Vec<_>>()
    );

    let mut dest = llbc_with_types(
        "crate_c",
        vec![
            identity_opaque_decl(
                0,
                identity_instantiated_name(
                    "crate_a",
                    "Container",
                    serde_json::json!([i32_ty.clone()]),
                    serde_json::json!([]),
                ),
            ),
            identity_opaque_decl(
                1,
                identity_instantiated_name(
                    "crate_a",
                    "Container",
                    serde_json::json!([i64_ty.clone()]),
                    serde_json::json!([]),
                ),
            ),
            identity_opaque_decl(2, identity_plain_name("crate_c", "mod", "Unique")),
            identity_opaque_decl(3, identity_plain_name("crate_c", "mod", "Shared")),
        ],
        vec![],
    );
    super::apply_published_structs(&mut dest, &rows);
    assert_eq!(
        published_field_ty(&dest, 0, "payload"),
        Some(&i32_ty),
        "the i32 instantiation must receive the i32 payload field"
    );
    assert_eq!(
        published_field_ty(&dest, 1, "payload"),
        Some(&i64_ty),
        "the i64 instantiation must receive the i64 payload field"
    );
    assert_eq!(
        published_field_ty(&dest, 2, "mark"),
        Some(&i32_ty),
        "a unique crate-stripped suffix may still match"
    );
    assert!(
        matches!(
            dest.type_by_id(3).map(|td| &td.kind),
            Some(majit_charon_reader::ullbc::TypeDeclKind::Opaque)
        ),
        "an ambiguous crate-stripped suffix must stay opaque"
    );
}

/// `Container<T,4>` and `Container<T,8>` are distinct types, including
/// when nested as a type argument.
#[test]
fn identity_adt_includes_const_generics() {
    let i32_ty = identity_signed("I32");
    let container = identity_struct_decl(
        0,
        identity_plain_name("fixture", "types", "Container"),
        "payload",
        i32_ty.clone(),
    );
    let outer = identity_struct_decl(
        1,
        identity_plain_name("fixture", "types", "Outer"),
        "inner",
        identity_adt_use(
            0,
            serde_json::json!([i32_ty.clone()]),
            serde_json::json!([]),
        ),
    );
    let four = identity_fun(
        0,
        "new",
        identity_adt_use(
            0,
            serde_json::json!([i32_ty.clone()]),
            identity_usize_const("4"),
        ),
    );
    let eight = identity_fun(
        1,
        "new",
        identity_adt_use(
            0,
            serde_json::json!([i32_ty.clone()]),
            identity_usize_const("8"),
        ),
    );
    let nested_four = identity_fun(
        2,
        "wrap",
        identity_adt_use(
            1,
            serde_json::json!([identity_adt_use(
                0,
                serde_json::json!([i32_ty.clone()]),
                identity_usize_const("4"),
            )]),
            serde_json::json!([]),
        ),
    );
    let nested_eight = identity_fun(
        3,
        "wrap",
        identity_adt_use(
            1,
            serde_json::json!([identity_adt_use(
                0,
                serde_json::json!([i32_ty.clone()]),
                identity_usize_const("8"),
            )]),
            serde_json::json!([]),
        ),
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![container, outer],
        vec![four, eight, nested_four, nested_eight],
    );
    let id_of = |id| super::scope_owner_identity(&llbc, llbc.fn_by_id(id).expect("fun"));
    let four_id = id_of(0);
    let eight_id = id_of(1);
    assert_ne!(
        four_id, eight_id,
        "const generic arguments must distinguish Container<T,4> from Container<T,8>: {four_id} vs {eight_id}"
    );
    assert!(
        four_id.contains("|4") && eight_id.contains("|8"),
        "the identity must carry the const generic: {four_id} / {eight_id}"
    );
    let nested_four_id = id_of(2);
    let nested_eight_id = id_of(3);
    assert_ne!(
        nested_four_id, nested_eight_id,
        "nested const generics must distinguish Outer<Container<T,4>> from Outer<Container<T,8>>: {nested_four_id} vs {nested_eight_id}"
    );
}

/// A `monomorphize:false` field `Generic<T>` is keyed by the template
/// path; the use-site TypeDeclRef still carries the arguments.
#[test]
fn published_field_generic_adt_falls_back_to_template_path() {
    let i32_ty = identity_signed("I32");
    let defining = llbc_with_types(
        "crate_a",
        vec![
            identity_struct_decl(
                0,
                identity_crate_item("crate_a", "Generic"),
                "mark",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                1,
                identity_crate_item("crate_a", "Outer"),
                "inner",
                identity_adt_use(
                    0,
                    serde_json::json!([i32_ty.clone()]),
                    serde_json::json!([]),
                ),
            ),
        ],
        vec![],
    );
    let rows = super::harvest_published_structs(&defining);
    let mut dest = llbc_with_types(
        "crate_c",
        vec![
            identity_opaque_decl(0, identity_crate_item("crate_a", "Generic")),
            identity_opaque_decl(1, identity_crate_item("crate_a", "Outer")),
        ],
        vec![],
    );
    super::apply_published_structs(&mut dest, &rows);
    let inner_id = published_field_adt_id(&dest, 1, "inner");
    assert_eq!(
        inner_id.and_then(serde_json::Value::as_u64),
        Some(0),
        "Generic<i32> must remap onto the dest template Generic: {inner_id:?}"
    );
}

/// `lltype.FuncType` ARGS/RESULT distinguish two function-pointer
/// instantiations of one wrapper.
#[test]
fn published_structs_fn_ptr_args_keep_distinct_identities() {
    let i32_ty = identity_signed("I32");
    let i64_ty = identity_signed("I64");
    let fn_i32 = identity_fn_ptr_ty(vec![i32_ty.clone()], i32_ty.clone());
    let fn_i64 = identity_fn_ptr_ty(vec![], i64_ty.clone());
    let defining = llbc_with_types(
        "crate_a",
        vec![
            identity_struct_decl(
                0,
                identity_instantiated_name(
                    "crate_a",
                    "Wrapper",
                    serde_json::json!([fn_i32.clone()]),
                    serde_json::json!([]),
                ),
                "from_i32",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                1,
                identity_instantiated_name(
                    "crate_a",
                    "Wrapper",
                    serde_json::json!([fn_i64.clone()]),
                    serde_json::json!([]),
                ),
                "from_i64",
                i64_ty.clone(),
            ),
        ],
        vec![],
    );
    let rows = super::harvest_published_structs(&defining);
    let paths: Vec<&str> = rows.iter().map(|row| row.path.as_str()).collect();
    assert_eq!(paths.len(), 2, "both wrappers must publish: {paths:?}");
    assert_ne!(
        paths[0], paths[1],
        "fn-pointer arguments must distinguish the identities: {paths:?}"
    );

    let mut dest = llbc_with_types(
        "crate_c",
        vec![
            identity_opaque_decl(
                0,
                identity_instantiated_name(
                    "crate_a",
                    "Wrapper",
                    serde_json::json!([fn_i32]),
                    serde_json::json!([]),
                ),
            ),
            identity_opaque_decl(
                1,
                identity_instantiated_name(
                    "crate_a",
                    "Wrapper",
                    serde_json::json!([fn_i64]),
                    serde_json::json!([]),
                ),
            ),
        ],
        vec![],
    );
    super::apply_published_structs(&mut dest, &rows);
    assert_eq!(
        published_field_ty(&dest, 0, "from_i32"),
        Some(&i32_ty),
        "Wrapper<fn(i32) -> i32> must keep its own field"
    );
    assert_eq!(
        published_field_ty(&dest, 1, "from_i64"),
        Some(&i64_ty),
        "Wrapper<fn() -> i64> must keep its own field"
    );
}

/// A bare `Foo` row makes the `Foo` suffix ambiguous with `crate_a::Foo`.
#[test]
fn published_structs_unprefixed_identity_counts_in_suffix_uniqueness() {
    let i32_ty = identity_signed("I32");
    let i64_ty = identity_signed("I64");
    let defining_a = llbc_with_types(
        "crate_a",
        vec![
            identity_struct_decl(
                0,
                identity_crate_item("crate_a", "Foo"),
                "from_a",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                1,
                identity_crate_item("crate_a", "Wrapper"),
                "inner",
                identity_adt_use(0, serde_json::json!([]), serde_json::json!([])),
            ),
        ],
        vec![],
    );
    let defining_bare = llbc_with_types(
        "bare",
        vec![identity_struct_decl(
            0,
            identity_bare_item("Foo"),
            "from_bare",
            i64_ty.clone(),
        )],
        vec![],
    );
    let mut rows = super::harvest_published_structs(&defining_a);
    rows.extend(super::harvest_published_structs(&defining_bare));
    let mut dest = llbc_with_types(
        "crate_b",
        vec![
            identity_opaque_decl(0, identity_bare_item("Foo")),
            identity_opaque_decl(1, identity_crate_item("crate_b", "Foo")),
            identity_opaque_decl(2, identity_crate_item("crate_a", "Wrapper")),
        ],
        vec![],
    );
    super::apply_published_structs(&mut dest, &rows);
    assert_eq!(
        published_field_ty(&dest, 0, "from_bare"),
        Some(&i64_ty),
        "the exact unprefixed Foo identity still matches"
    );
    assert!(
        dest_kind_is_opaque(&dest, 1),
        "crate_b::Foo must stay opaque when Foo and crate_a::Foo both publish"
    );
    let inner_id = published_field_adt_id(&dest, 2, "inner");
    assert!(
        inner_id.and_then(|id| id.get("Path")).is_some(),
        "an ambiguous Foo dest must leave the nested ADT as Path: {inner_id:?}"
    );
}

/// `lltype.Char` is a distinct primitive; two char const instantiations
/// keep their own bodies.
#[test]
fn published_structs_char_const_args_keep_distinct_identities() {
    let i32_ty = identity_signed("I32");
    let i64_ty = identity_signed("I64");
    let defining = llbc_with_types(
        "crate_a",
        vec![
            identity_struct_decl(
                0,
                identity_instantiated_name(
                    "crate_a",
                    "Outer",
                    serde_json::json!([]),
                    identity_char_const("a"),
                ),
                "from_a",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                1,
                identity_instantiated_name(
                    "crate_a",
                    "Outer",
                    serde_json::json!([]),
                    identity_char_const("b"),
                ),
                "from_b",
                i64_ty.clone(),
            ),
        ],
        vec![],
    );
    let rows = super::harvest_published_structs(&defining);
    let paths: Vec<&str> = rows.iter().map(|row| row.path.as_str()).collect();
    assert_eq!(
        paths.len(),
        2,
        "both Outer instantiations must publish: {paths:?}"
    );
    assert_ne!(
        paths[0], paths[1],
        "char const arguments must distinguish the identities: {paths:?}"
    );
    assert!(
        paths.iter().any(|path| path.contains("char:a"))
            && paths.iter().any(|path| path.contains("char:b")),
        "identities must carry the char const: {paths:?}"
    );

    let mut dest = llbc_with_types(
        "crate_c",
        vec![
            identity_opaque_decl(
                0,
                identity_instantiated_name(
                    "crate_a",
                    "Outer",
                    serde_json::json!([]),
                    identity_char_const("a"),
                ),
            ),
            identity_opaque_decl(
                1,
                identity_instantiated_name(
                    "crate_a",
                    "Outer",
                    serde_json::json!([]),
                    identity_char_const("b"),
                ),
            ),
        ],
        vec![],
    );
    super::apply_published_structs(&mut dest, &rows);
    assert_eq!(
        published_field_ty(&dest, 0, "from_a"),
        Some(&i32_ty),
        "Outer<'a'> must keep its own field"
    );
    assert_eq!(
        published_field_ty(&dest, 1, "from_b"),
        Some(&i64_ty),
        "Outer<'b'> must keep its own field"
    );
}

/// `LowLevelType.__eq__` compares the outer type; a `::` inside a
/// generic argument is not a crate prefix.
#[test]
fn published_structs_crate_prefix_does_not_split_inside_generic_args() {
    let i32_ty = identity_signed("I32");
    let t_use = identity_adt_use(0, serde_json::json!([]), serde_json::json!([]));
    let defining = llbc_with_types(
        "crate_a",
        vec![
            identity_struct_decl(
                0,
                identity_crate_item("dep", "T"),
                "payload",
                i32_ty.clone(),
            ),
            identity_struct_decl(
                1,
                identity_unqualified_instantiated_name(
                    "WrapA",
                    serde_json::json!([t_use.clone()]),
                    serde_json::json!([]),
                ),
                "from_a",
                i32_ty.clone(),
            ),
        ],
        vec![],
    );
    let rows = super::harvest_published_structs(&defining);
    assert!(
        rows.iter()
            .any(|row| row.path.starts_with("WrapA<") && row.path.contains("dep::T")),
        "WrapA identity must keep the qualified argument: {:?}",
        rows.iter().map(|row| &row.path).collect::<Vec<_>>()
    );

    let mut dest = llbc_with_types(
        "crate_c",
        vec![
            identity_opaque_decl(0, identity_crate_item("dep", "T")),
            identity_opaque_decl(
                1,
                identity_unqualified_instantiated_name(
                    "WrapA",
                    serde_json::json!([t_use.clone()]),
                    serde_json::json!([]),
                ),
            ),
            identity_opaque_decl(
                2,
                identity_unqualified_instantiated_name(
                    "WrapB",
                    serde_json::json!([t_use]),
                    serde_json::json!([]),
                ),
            ),
        ],
        vec![],
    );
    super::apply_published_structs(&mut dest, &rows);
    assert_eq!(
        published_field_ty(&dest, 1, "from_a"),
        Some(&i32_ty),
        "WrapA<dep::T> must still receive its published fields"
    );
    assert!(
        dest_kind_is_opaque(&dest, 2),
        "WrapB<dep::T> must not take WrapA's body via a suffix split inside the argument"
    );
}
