use super::is_core_result_map_err_path;

use crate::flowspace::model::Variable;

use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, ValueType};

use crate::front::mir::tests::{
    OptionRecvRepr, assert_option_recv_repr, build_two_arg_combinator, count_branch_calls,
    count_method_calls, disc_site, ok_ctor_shape, option_disc_site, option_ok_or_else_site,
    push_fn_const, some_ctor_ops, some_ctor_shape,
};

fn branch_layout(disc_off: u64, payload_off: u64) -> majit_charon_reader::ullbc::TypeLayout {
    majit_charon_reader::ullbc::TypeLayout {
        size: Some(16),
        align: Some(8),
        variant_layouts: vec![
            majit_charon_reader::ullbc::VariantLayout {
                field_offsets: vec![payload_off],
                tagger: None,
            },
            majit_charon_reader::ullbc::VariantLayout {
                field_offsets: vec![payload_off],
                tagger: None,
            },
        ],
        discriminator: Some(serde_json::json!({
            "Branch": {"offset": disc_off, "int_ty": {"Signed": "I64"}}
        })),
        repr: None,
    }
}

fn graph_with_result_branch() -> (FunctionGraph, Variable) {
    let mut graph = FunctionGraph::new("result_branch");
    let entry = graph.startblock;
    let operand = graph
        .push_op_var(entry, OpKind::ConstInt(1), true)
        .expect("operand");
    let result = graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::method("branch", Some("core::result::Result".into())),
                args: vec![LinkArg::Value(operand)],
                result_ty: ValueType::Ref(Some("core::ops::control_flow::ControlFlow".into())),
            },
            true,
        )
        .expect("branch");
    graph.set_return(entry, Some(result.clone()));
    (graph, result)
}

#[test]
fn equal_result_and_control_flow_layouts_lower_branch_to_same_as() {
    let same = branch_layout(0, 8);
    let (mut graph, _result) = graph_with_result_branch();
    let entry = graph.startblock;
    let call_idx = graph.block(entry).operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "branch"
        )
    });
    let call_idx = call_idx.expect("branch call");
    let kind = std::mem::replace(
        &mut graph.block_mut(entry).operations[call_idx].kind,
        OpKind::ConstInt(0),
    );
    graph.block_mut(entry).operations[call_idx].kind =
        super::rewrite_result_branch_for_layouts(kind, Some(&same), Some(&same));
    assert_eq!(count_branch_calls(&graph), 0);
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .any(|op| { matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as") }),
        "equal layouts keep the bits with same_as"
    );
}

#[test]
fn niche_result_layout_lowers_branch_as_a_discriminant_match() {
    let explicit = branch_layout(0, 8);
    let niche = branch_layout(8, 0);
    assert!(!super::type_layouts_equal(&explicit, &niche));
    let (mut graph, _result) = graph_with_result_branch();
    let entry = graph.startblock;
    let call_idx = graph
        .block(entry)
        .operations
        .iter()
        .position(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "branch"
            )
        })
        .expect("branch call");
    let kind = graph.block(entry).operations[call_idx].kind.clone();
    let kept = super::rewrite_result_branch_for_layouts(kind, Some(&explicit), Some(&niche));
    assert!(
        matches!(
            kept,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "branch"
        ),
        "unequal layouts stay a branch call"
    );
    super::lower_result_branch_to_control_flow(&mut graph);
    assert_eq!(count_branch_calls(&graph), 0, "branch must not stay a call");
    let reads_disc = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant"
            )
        });
    assert!(reads_disc, "the match reads the Result discriminant");
    let mut ctors = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } = &op.kind
            {
                ctors.push(name.clone());
            }
        }
    }
    assert!(ctors.iter().any(|name| name == "Continue"), "{ctors:?}");
    assert!(ctors.iter().any(|name| name == "Break"), "{ctors:?}");
}

#[test]
fn map_err_capture_requires_the_core_result_callee() {
    for path in [
        "core::result::<Impl>::map_err",
        "std::result::<Impl>::map_err",
        "core::result::Result::map_err",
        "std::result::Result::map_err",
    ] {
        assert!(is_core_result_map_err_path(path), "{path}");
        assert!(super::is_core_result_method(path, "map_err"), "{path}");
    }
    for path in ["result::Result::map_err", "result::<Impl>::map_err"] {
        assert!(is_core_result_map_err_path(path), "{path}");
    }
    for path in [
        "map_err",
        "Result::map_err",
        "mycrate::result::<Impl>::map_err",
        "core::result::ResultExt::<Impl>::map_err",
        "foo::bar::map_err",
    ] {
        assert!(!is_core_result_map_err_path(path), "{path}");
    }
}

#[test]
fn std_extern_path_predicates_match_exact_full_paths_only() {
    assert!(super::is_core_clone_impls_clone_path(
        "core::clone::impls::<Impl>::clone"
    ));
    assert!(!super::is_core_clone_impls_clone_path(
        "core::clone::<Impl>::clone"
    ));
    assert!(!super::is_core_clone_impls_clone_path(
        "alloc::vec::<Impl>::clone"
    ));

    assert!(super::is_core_default_path(
        "core::default::<Impl>::default"
    ));
    assert!(super::is_core_default_path(
        "core::ptr::mut_ptr::<Impl>::default"
    ));
    assert!(!super::is_core_default_path(
        "core::ptr::const_ptr::<Impl>::default"
    ));
    assert!(!super::is_core_default_path(
        "core::default::Default::default"
    ));
    assert!(!super::is_core_default_path("core::tuple::<Impl>::default"));

    assert!(super::is_core_tuple_default_path(
        "core::tuple::<Impl>::default"
    ));
    assert!(!super::is_core_tuple_default_path(
        "core::default::<Impl>::default"
    ));
    assert!(!super::is_core_tuple_default_path(
        "core::tuple::Tuple::default"
    ));

    assert!(super::is_core_ptr_add_path(
        "core::ptr::mut_ptr::<Impl>::add"
    ));
    assert!(super::is_core_ptr_add_path(
        "core::ptr::const_ptr::<Impl>::add"
    ));
    assert!(!super::is_core_ptr_add_path(
        "core::ptr::mut_ptr::<Impl>::wrapping_add"
    ));
    assert!(!super::is_core_ptr_add_path(
        "core::ptr::mut_ptr::<Impl>::offset"
    ));
    assert!(super::is_core_ptr_sub_path(
        "core::ptr::mut_ptr::<Impl>::sub"
    ));
    assert!(super::is_core_ptr_sub_path(
        "core::ptr::const_ptr::<Impl>::sub"
    ));
    assert!(!super::is_core_ptr_sub_path(
        "core::ptr::mut_ptr::<Impl>::add"
    ));

    assert!(super::is_core_ptr_write_path("core::ptr::write"));
    assert!(super::is_core_ptr_write_path(
        "core::ptr::mut_ptr::<Impl>::write"
    ));
    assert!(!super::is_core_ptr_write_path("core::ptr::write_unaligned"));
    assert!(!super::is_core_ptr_write_path("core::ptr::write_bytes"));

    assert!(super::is_alloc_vec_new_segments(&[
        "alloc".into(),
        "vec".into(),
        "<Impl>".into(),
        "new".into(),
    ]));
    assert!(super::is_alloc_vec_new_segments(&[
        "alloc".into(),
        "vec".into(),
        "Vec".into(),
        "new".into(),
    ]));
    assert!(!super::is_alloc_vec_new_segments(&[
        "alloc".into(),
        "vec".into(),
        "<Impl>".into(),
        "with_capacity".into(),
    ]));
}

#[test]
fn result_map_lowers_to_discriminant_switch() {
    let (mut graph, result) = build_two_arg_combinator("map");
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[disc_site(super::DiscCombinator::ResultMap, result)],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "map"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 1);
    assert_eq!(graph.blocks[graph.startblock.0].exits.len(), 2);
}

#[test]
fn option_filter_lowers_to_discriminant_and_predicate_switch() {
    let mut graph = FunctionGraph::new("test_filter");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let env = graph.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("filter", Some("Option".into())),
                args: crate::model::call_args(vec![recv, env]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[option_disc_site(
            super::DiscCombinator::OptionFilter,
            result,
        )],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "filter"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 1);
    assert_eq!(graph.blocks[graph.startblock.0].exits.len(), 2);
    let pred_branches = graph
        .blocks
        .iter()
        .filter(|block| block.exits.len() == 2 && block.id != graph.startblock)
        .count();
    assert_eq!(pred_branches, 1, "the Some arm switches on the predicate");
}

/// `Option<*mut PyObject>::filter` is a nullable pointer. The tag is
/// `is_` against null, and `Some` is that pointer — not `__discriminant`
/// / `__pos_0` of the object.
#[test]
fn niche_option_filter_is_a_null_test() {
    let mut graph = FunctionGraph::new("test_niche_filter");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let env = graph.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("filter", Some("Option".into())),
                args: crate::model::call_args(vec![recv, env]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let mut site = option_disc_site(super::DiscCombinator::OptionFilter, result);
    site.recv_niche = true;
    site.result_niche = true;
    site.niche_null_cast = Some(("pyobject::PyObject".into(), ValueType::Ref(None)));
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "filter"), 0);
    let mut saw_disc = false;
    let mut saw_pos0 = false;
    let mut saw_is = false;
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    saw_disc = true;
                }
                OpKind::FieldRead { field, .. } if field.name == "__pos_0" => {
                    saw_pos0 = true;
                }
                OpKind::BinOp { op, .. } if op == "is_" => saw_is = true,
                _ => {}
            }
        }
    }
    assert!(!saw_disc, "niche filter read __discriminant");
    assert!(!saw_pos0, "niche filter read __pos_0");
    assert!(saw_is, "niche filter did not null-test the pointer");
}

#[test]
fn option_ok_or_else_lowers_to_discriminant_switch() {
    let (mut graph, result) = build_two_arg_combinator("ok_or_else");
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[option_ok_or_else_site(result)],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "ok_or_else"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 1);
    assert_eq!(graph.blocks[graph.startblock.0].exits.len(), 2);
}

fn option_ok_or_family_site(
    kind: super::DiscCombinator,
    result_var: Variable,
) -> super::DiscCombinatorSite {
    let mut site = option_disc_site(kind, result_var);
    site.result_owner = "core::result::Result".into();
    site.result_tag0_owner = "core::result::Result::Ok".into();
    site.result_tag1_owner = "core::result::Result::Err".into();
    site.result_payload0_ty = ValueType::Ref(None);
    site.result_payload1_ty = ValueType::Ref(None);
    if kind == super::DiscCombinator::OptionOkOrElse {
        site.call_result_ty = ValueType::Ref(None);
    }
    site
}

fn check_option_ok_or_family_repr(kind: super::DiscCombinator, method: &str, repr: OptionRecvRepr) {
    let (mut graph, result) = build_two_arg_combinator(method);
    let mut site = option_ok_or_family_site(kind, result);
    match repr {
        OptionRecvRepr::PointerNiche => {
            site.recv_niche = true;
            site.niche_null_cast = Some(("pyobject::PyObject".into(), ValueType::Ref(None)));
        }
        OptionRecvRepr::ScalarNiche => {
            site.scalar_niche = true;
            site.recv_owner = "core::option::Option<NonZeroUsize>".into();
            site.payload1_ty = ValueType::Unsigned;
        }
        OptionRecvRepr::Aggregate => {}
    }
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1, "{method} {repr:?} must rewrite");
    assert_eq!(
        count_method_calls(&graph, method),
        0,
        "{method} {repr:?} left a residual call"
    );
    assert_eq!(graph.blocks[graph.startblock.0].exits.len(), 2);
    assert_option_recv_repr(&graph, repr, method);
}

/// Six shapes: `ok_or` / `ok_or_else` × pointer niche / scalar niche /
/// aggregate Option.
#[test]
fn option_ok_or_family_repr_switch() {
    let kinds = [
        (super::DiscCombinator::OptionOkOr, "ok_or"),
        (super::DiscCombinator::OptionOkOrElse, "ok_or_else"),
    ];
    let reprs = [
        OptionRecvRepr::PointerNiche,
        OptionRecvRepr::ScalarNiche,
        OptionRecvRepr::Aggregate,
    ];
    for (kind, method) in kinds {
        for repr in reprs {
            check_option_ok_or_family_repr(kind, method, repr);
        }
    }
}

#[test]
fn result_ok_lowers_to_some_none_switch() {
    let mut graph = FunctionGraph::new("test_result_ok");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("ok", Some("Result".into())),
                args: crate::model::call_args(vec![recv]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let mut site = disc_site(super::DiscCombinator::ResultOk, result);
    site.result_owner = "core::option::Option".into();
    site.result_tag1_owner = "core::option::Option::Some".into();
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[site],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "ok"), 0);
    assert_eq!(graph.blocks[graph.startblock.0].exits.len(), 2);
}

#[test]
fn result_is_ok_replaces_the_call_with_a_tag_compare() {
    let mut graph = FunctionGraph::new("test_is_ok");
    let a = graph.startblock;
    let recv = graph.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
    let result = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method("is_ok", Some("Result".into())),
                args: crate::model::call_args(vec![recv]),
                result_ty: ValueType::Bool,
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    graph.set_goto(a, b, vec![result.clone()]);
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[disc_site(super::DiscCombinator::ResultIsOk, result)],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "is_ok"), 0);
    assert!(
        graph.blocks[a.0]
            .operations
            .iter()
            .any(|op| { matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq") })
    );
}

#[test]
fn result_and_then_lowers_to_discriminant_switch() {
    let (mut graph, result) = build_two_arg_combinator("and_then");
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[disc_site(super::DiscCombinator::ResultAndThen, result)],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "and_then"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 1);
    assert_eq!(graph.blocks[graph.startblock.0].exits.len(), 2);
}

/// A combinator call `result_exc` made can-raise (its result renamed to
/// the `Ok` payload), whose receiver is the `Ok`/`Err` shell pair
/// `catch_and_rewrap` rebuilt around a raising producer.
struct RaisingCombinator {
    graph: FunctionGraph,
    /// The `Ok` shell block.
    n: crate::model::BlockId,
    /// The `Err` shell block.
    e: crate::model::BlockId,
    /// The caught exception value `e` receives.
    caught_value: Variable,
    /// The combinator's normal target.
    b: crate::model::BlockId,
    /// The combinator's exception target.
    handler: crate::model::BlockId,
    call_args: Vec<LinkArg>,
}

/// `callable` is `None` for a closure env threaded as a block input, or
/// the path of a function item.  `rewrapped` builds the receiver as the
/// rebuilt shell pair; otherwise it is a plain input.  `to_exceptblock`
/// sends the exception exit to the graph's exceptblock instead of a
/// local handler.
fn raising_combinator(
    method: &str,
    callable: Option<&[&str]>,
    rewrapped: bool,
    to_exceptblock: bool,
) -> RaisingCombinator {
    use crate::model::{ExitSwitch, Link};
    let mut graph = FunctionGraph::new("test_raising_combinator");
    let p = graph.startblock;
    let env = graph.push_op_var(p, OpKind::ConstInt(7), true).unwrap();
    let (a, a_inputs) = graph.create_block_with_arg_vars(2);
    let (n, e, caught_value) = if rewrapped {
        let x = graph.alloc_value_var();
        let (n, n_inputs) = graph.create_block_with_arg_vars(2);
        let (e, e_inputs) = graph.create_block_with_arg_vars(3);
        graph.blocks[p.0]
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(x.clone()),
                kind: OpKind::Call {
                    target: CallTarget::function_path(["test", "produce"]),
                    args: Vec::new(),
                    result_ty: ValueType::Int,
                },
            });
        let va = graph.alloc_value_var();
        let vb = graph.alloc_value_var();
        let mut exc = Link::new_mixed(
            vec![
                LinkArg::Value(env.clone()),
                LinkArg::Value(va.clone()),
                LinkArg::Value(vb.clone()),
            ],
            e,
            Some(crate::model::exception_exitcase()),
        );
        exc.last_exception = Some(LinkArg::Value(va));
        exc.last_exc_value = Some(LinkArg::Value(vb));
        graph.set_control_flow_metadata(
            p,
            Some(ExitSwitch::LastException),
            vec![
                Link::new_mixed(
                    vec![LinkArg::Value(x), LinkArg::Value(env.clone())],
                    n,
                    None,
                ),
                exc,
            ],
        );
        let ok = crate::front::result_exc::build_shell(
            &mut graph,
            n,
            "Ok",
            n_inputs[0].clone(),
            ValueType::Int,
            "",
        );
        graph.set_goto(n, a, vec![ok, n_inputs[1].clone()]);
        let error = graph
            .push_op_var(
                e,
                OpKind::Call {
                    target: CallTarget::function_path(["test", "from_exc_object"]),
                    args: crate::model::call_args(vec![e_inputs[2].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let err = crate::front::result_exc::build_shell(
            &mut graph,
            e,
            "Err",
            error,
            ValueType::Ref(None),
            "",
        );
        graph.set_goto(e, a, vec![err, e_inputs[0].clone()]);
        (n, e, e_inputs[2].clone())
    } else {
        let recv = graph.push_op_var(p, OpKind::ConstInt(0), true).unwrap();
        graph.set_goto(p, a, vec![recv, env]);
        (p, p, graph.alloc_value_var())
    };
    let callable_var = match callable {
        Some(path) => push_fn_const(&mut graph, a, path),
        None => a_inputs[1].clone(),
    };
    let call_args = crate::model::call_args(vec![a_inputs[0].clone(), callable_var]);
    let payload = graph
        .push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::method(method, Some("Result".into())),
                args: call_args.clone(),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let (b, _) = graph.create_block_with_arg_vars(1);
    graph.set_return(b, None);
    let etype = graph.alloc_value_var();
    let evalue = graph.alloc_value_var();
    let (handler, raise_args) = if to_exceptblock {
        (
            graph.exceptblock,
            vec![
                LinkArg::Value(etype.clone()),
                LinkArg::Value(evalue.clone()),
            ],
        )
    } else {
        let (handler, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(handler, None);
        (handler, vec![LinkArg::Value(evalue.clone())])
    };
    let mut raise = Link::new_mixed(
        raise_args,
        handler,
        Some(crate::model::exception_exitcase()),
    );
    raise.last_exception = Some(LinkArg::Value(etype));
    raise.last_exc_value = Some(LinkArg::Value(evalue));
    graph.set_control_flow_metadata(
        a,
        Some(ExitSwitch::LastException),
        vec![
            Link::new_mixed(vec![LinkArg::Value(payload)], b, None),
            raise,
        ],
    );
    RaisingCombinator {
        graph,
        n,
        e,
        caught_value,
        b,
        handler,
        call_args,
    }
}

/// The site of a [`raising_combinator`]: its recorded result is the
/// pre-rename shell.
fn raising_site(
    kind: super::DiscCombinator,
    fixture: &mut RaisingCombinator,
    callable: Option<&[&str]>,
    ctor: Option<super::AggregateShape>,
) -> super::DiscCombinatorSite {
    let mut site = disc_site(kind, fixture.graph.alloc_value_var());
    site.call_args = fixture.call_args.clone();
    if let Some(path) = callable {
        site.call_once_owner = String::new();
        site.fn_item_segments = Some(path.iter().map(|s| s.to_string()).collect());
    }
    if ctor.is_some() {
        site.call_result_ty = ValueType::Ref(None);
        site.result_payload0_ty = ValueType::Ref(None);
    }
    site.fn_item_ctor = ctor;
    site
}

fn rewire_raising(fixture: &mut RaisingCombinator, site: super::DiscCombinatorSite) -> usize {
    let to_exc: &[&str] = &["test", "to_exc_object"];
    super::rewire_disc_combinator_sites(
        &mut fixture.graph,
        &[site],
        crate::ErrorCarrierSpec {
            to_exc_object: Some(to_exc),
            ..Default::default()
        },
    )
}

/// Operations of the blocks reachable from the start block.
fn live_ops(graph: &FunctionGraph) -> Vec<&crate::model::SpaceOperation> {
    graph
        .iterblocks_order()
        .into_iter()
        .flat_map(|id| &graph.blocks[id.0].operations)
        .collect()
}

fn live_calls_to(graph: &FunctionGraph, leaf: &str) -> usize {
    live_ops(graph)
        .into_iter()
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == leaf,
            OpKind::Call {
                target: target @ CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some(leaf)
                    && crate::model::fn_const_segments(target).is_none()
            }
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } => name == leaf,
            _ => false,
        })
        .count()
}

/// The rebuilt shells are gone: the `Ok` block calls `callable` on the
/// payload and the `Err` block forwards the caught exception pair to the
/// combinator's exception target.
fn assert_receiver_shells_dropped(fixture: &RaisingCombinator, method: &str) {
    let graph = &fixture.graph;
    assert_eq!(live_calls_to(graph, method), 0);
    assert_eq!(live_calls_to(graph, "Err"), 0);
    assert_eq!(live_calls_to(graph, "from_exc_object"), 0);
    assert_eq!(live_calls_to(graph, "to_exc_object"), 0);
    let e = &graph.blocks[fixture.e.0];
    assert!(e.operations.is_empty());
    let [exit] = e.exits.as_slice() else {
        panic!("Err block has one exit");
    };
    assert_eq!(exit.target, fixture.handler);
    assert!(exit.exitcase.is_none());
    assert_eq!(
        exit.args,
        vec![LinkArg::Value(fixture.caught_value.clone())]
    );
}

#[test]
fn raising_result_map_drops_the_rebuilt_receiver_shells() {
    let mut fixture = raising_combinator("map", None, true, false);
    let site = raising_site(super::DiscCombinator::ResultMap, &mut fixture, None, None);
    assert_eq!(rewire_raising(&mut fixture, site), 1);
    assert_receiver_shells_dropped(&fixture, "map");
    assert_eq!(live_calls_to(&fixture.graph, "Ok"), 0);
    let n = &fixture.graph.blocks[fixture.n.0];
    assert!(n.exitswitch.is_none());
    assert_eq!(n.exits[0].target, fixture.b);
    assert_eq!(count_method_calls(&fixture.graph, "call_once"), 1);
}

#[test]
fn raising_result_map_of_a_constructor_drops_the_rebuilt_receiver_shells() {
    // `r.map(Some)` on a rebuilt receiver: `Some(x)` on the producer's
    // normal edge, the producer's exception straight to the handler.
    let path: &[&str] = &["core", "option", "Option", "Some"];
    let mut fixture = raising_combinator("map", Some(path), true, false);
    let site = raising_site(
        super::DiscCombinator::ResultMap,
        &mut fixture,
        Some(path),
        Some(some_ctor_shape()),
    );
    assert_eq!(rewire_raising(&mut fixture, site), 1);
    assert_receiver_shells_dropped(&fixture, "map");
    assert_eq!(some_ctor_ops(&fixture.graph), (1, 0));
    let n = &fixture.graph.blocks[fixture.n.0];
    assert_eq!(n.exits[0].target, fixture.b);
    assert_eq!(count_method_calls(&fixture.graph, "call_once"), 0);
}

/// The `Ok` block ends in the callable as a raising call: normal exit to
/// the combinator's normal target, exception exit to its handler.
fn assert_ok_block_raises_through_callable(fixture: &RaisingCombinator, callee: &str) {
    use crate::model::ExitSwitch;
    let n = &fixture.graph.blocks[fixture.n.0];
    assert!(matches!(n.exitswitch, Some(ExitSwitch::LastException)));
    let last = n.operations.last().expect("raising call");
    let called = match &last.kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } => name.as_str(),
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => segments.last().map_or("", String::as_str),
        _ => "",
    };
    assert_eq!(called, callee);
    let [normal, raise] = n.exits.as_slice() else {
        panic!("raising call has two exits");
    };
    assert_eq!(normal.target, fixture.b);
    assert_eq!(
        normal.args,
        vec![LinkArg::Value(last.result.clone().unwrap())]
    );
    assert_eq!(raise.target, fixture.handler);
    assert!(raise.exitcase.is_some());
    assert_eq!(raise.args.first(), raise.last_exc_value.as_ref());
}

#[test]
fn raising_result_and_then_calls_the_closure_as_a_raising_call() {
    let mut fixture = raising_combinator("and_then", None, true, false);
    let site = raising_site(
        super::DiscCombinator::ResultAndThen,
        &mut fixture,
        None,
        None,
    );
    assert_eq!(rewire_raising(&mut fixture, site), 1);
    assert_receiver_shells_dropped(&fixture, "and_then");
    assert_ok_block_raises_through_callable(&fixture, "call_once");
}

#[test]
fn raising_result_and_then_calls_the_function_item_as_a_raising_call() {
    let path: &[&str] = &["test", "step"];
    let mut fixture = raising_combinator("and_then", Some(path), true, false);
    let site = raising_site(
        super::DiscCombinator::ResultAndThen,
        &mut fixture,
        Some(path),
        None,
    );
    assert_eq!(rewire_raising(&mut fixture, site), 1);
    assert_receiver_shells_dropped(&fixture, "and_then");
    assert_ok_block_raises_through_callable(&fixture, "step");
}

#[test]
fn raising_result_and_then_of_a_constructor_keeps_the_residual() {
    let path: &[&str] = &["core", "result", "Result", "Ok"];
    let mut fixture = raising_combinator("and_then", Some(path), true, false);
    let site = raising_site(
        super::DiscCombinator::ResultAndThen,
        &mut fixture,
        Some(path),
        Some(ok_ctor_shape()),
    );
    assert_eq!(rewire_raising(&mut fixture, site), 0);
    assert_eq!(count_method_calls(&fixture.graph, "and_then"), 1);
    assert_eq!(live_calls_to(&fixture.graph, "Err"), 1);
}

#[test]
fn raising_result_map_with_a_local_handler_keeps_the_residual() {
    // A receiver that is not a rebuilt shell pair stays a value; its
    // `Err` would reach a local handler expecting a shell.
    let mut fixture = raising_combinator("map", None, false, false);
    let site = raising_site(super::DiscCombinator::ResultMap, &mut fixture, None, None);
    assert_eq!(rewire_raising(&mut fixture, site), 0);
    assert_eq!(count_method_calls(&fixture.graph, "map"), 1);
}

/// The receiver branches on its discriminant; the `Err` arm raises the
/// carrier with `type(carrier)` and an exceptblock link. The codewriter
/// rewrites that raise to `to_exc_object`. Returns the `Ok` arm.
fn assert_err_arm_raises_to_the_exceptblock(fixture: &RaisingCombinator) -> crate::model::BlockId {
    let graph = &fixture.graph;
    let start = &graph.blocks[graph.startblock.0];
    let entry = &graph.blocks[start.exits[0].target.0];
    let arms: Vec<_> = entry.exits.iter().map(|link| link.target).collect();
    assert_eq!(arms.len(), 2);
    let err_arm = arms
        .iter()
        .map(|id| &graph.blocks[id.0])
        .find(|arm| matches!(arm.exits.as_slice(), [exit] if exit.target == graph.exceptblock))
        .expect("Err arm");
    let type_of = err_arm
        .operations
        .iter()
        .find(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["type".to_string()]
            )
        })
        .expect("Err arm records type(carrier)");
    let OpKind::Call { args, .. } = &type_of.kind else {
        unreachable!("matched a type() call");
    };
    let carrier = args[0].as_variable().expect("type() reads the carrier");
    assert_eq!(
        err_arm.exits[0].args[0].as_variable(),
        type_of.result.as_ref()
    );
    assert_eq!(err_arm.exits[0].args[1].as_variable(), Some(carrier));
    assert!(err_arm.exits[0].exitcase.is_none());
    *arms.iter().find(|id| **id != err_arm.id).unwrap()
}

#[test]
fn raising_result_map_of_a_constructor_raises_err_to_the_exceptblock() {
    let path: &[&str] = &["core", "option", "Option", "Some"];
    let mut fixture = raising_combinator("map", Some(path), false, true);
    let site = raising_site(
        super::DiscCombinator::ResultMap,
        &mut fixture,
        Some(path),
        Some(some_ctor_shape()),
    );
    assert_eq!(rewire_raising(&mut fixture, site), 1);
    assert_eq!(count_method_calls(&fixture.graph, "map"), 0);
    assert_eq!(some_ctor_ops(&fixture.graph), (1, 0));
    let ok_arm = assert_err_arm_raises_to_the_exceptblock(&fixture);
    let ok_arm = &fixture.graph.blocks[ok_arm.0];
    assert!(ok_arm.exitswitch.is_none());
    assert_eq!(ok_arm.exits[0].target, fixture.b);
}

#[test]
fn raising_result_and_then_raises_err_to_the_exceptblock() {
    use crate::model::ExitSwitch;
    let mut fixture = raising_combinator("and_then", None, false, true);
    let site = raising_site(
        super::DiscCombinator::ResultAndThen,
        &mut fixture,
        None,
        None,
    );
    assert_eq!(rewire_raising(&mut fixture, site), 1);
    assert_eq!(count_method_calls(&fixture.graph, "and_then"), 0);
    let ok_arm = assert_err_arm_raises_to_the_exceptblock(&fixture);
    let ok_arm = &fixture.graph.blocks[ok_arm.0];
    assert!(matches!(ok_arm.exitswitch, Some(ExitSwitch::LastException)));
    let [normal, raise] = ok_arm.exits.as_slice() else {
        panic!("raising call has two exits");
    };
    assert_eq!(normal.target, fixture.b);
    assert_eq!(raise.target, fixture.graph.exceptblock);
    assert_eq!(
        raise.args,
        vec![
            raise.last_exception.clone().unwrap(),
            raise.last_exc_value.clone().unwrap()
        ]
    );
}

#[test]
fn result_unwrap_or_else_calls_on_err() {
    let (mut graph, result) = build_two_arg_combinator("unwrap_or_else");
    let rewritten = super::rewire_disc_combinator_sites(
        &mut graph,
        &[disc_site(super::DiscCombinator::ResultUnwrapOrElse, result)],
        crate::ErrorCarrierSpec::default(),
    );
    assert_eq!(rewritten, 1);
    assert_eq!(count_method_calls(&graph, "unwrap_or_else"), 0);
    assert_eq!(count_method_calls(&graph, "call_once"), 1);
}

#[test]
fn core_option_result_method_paths() {
    assert!(super::is_core_option_method(
        "core::option::<Impl>::filter",
        "filter"
    ));
    assert!(super::is_core_result_method(
        "core::result::Result::map",
        "map"
    ));
    assert!(!super::is_core_result_method(
        "core::option::<Impl>::map",
        "map"
    ));
    assert!(!super::is_core_option_method(
        "mycrate::option::<Impl>::filter",
        "filter"
    ));
}

#[test]
fn classed_result_enum_owner_is_core_result_path() {
    use crate::front::result_exc::owner_is_core_result;
    use crate::model::ValueType;

    assert!(owner_is_core_result("core::result::Result"));
    assert!(owner_is_core_result("core::result::Result<Infallible, E>"));
    assert!(owner_is_core_result(
        "core::result::Result<Infallible, E>::Err"
    ));
    assert!(owner_is_core_result("result::Result<??scalar, i64>"));
    assert!(owner_is_core_result("std::result::Result"));
    assert!(
        !owner_is_core_result("mymod::Result<Infallible, E>"),
        "a same-leaf user enum is not core::result::Result"
    );
    assert!(
        !owner_is_core_result("Result<Infallible, E>"),
        "a bare Result leaf is not the std item"
    );

    let core = ValueType::Ref(Some("core::result::Result<Infallible, E>".to_string()));
    assert_eq!(
        super::classed_result_enum_owner(&core),
        Some("core::result::Result<Infallible, E>")
    );
    let stripped = ValueType::Ref(Some("result::Result<??scalar, i64>".to_string()));
    assert_eq!(
        super::classed_result_enum_owner(&stripped),
        Some("result::Result<??scalar, i64>")
    );
    let user = ValueType::Ref(Some("mymod::Result<Infallible, E>".to_string()));
    assert_eq!(
        super::classed_result_enum_owner(&user),
        None,
        "user Result must not intern as the std Result wrap"
    );
    let leaf = ValueType::Ref(Some("Result<Infallible, E>".to_string()));
    assert_eq!(super::classed_result_enum_owner(&leaf), None);
}
