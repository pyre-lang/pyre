//! `Result::map_err(result, closure)` → discriminant closure-select.
//!
//! Rust's foreign `core::result` body is opaque in LLBC.  Leaving the method
//! call residual is not executable for an embedded interpreter because there
//! is no monomorphisation-independent host address for its closure type.  The
//! equivalent RPython graph already has the ordinary two exception/value
//! branches (`translator/exceptiontransform.py::ExceptionTransformer.transform_block`);
//! this adapter restores that
//! shape before `result_exc` converts an exception-carrying result to native
//! graph exception edges.

use crate::flowspace::model::Variable;
use crate::front::option_variant::{
    close_goto_mixed, emit_narrow, emit_sum_variant, map_source, reproduce_exit_args,
};
use crate::front::result_exc::emit_callable;
use crate::model::{
    CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

#[derive(Clone)]
pub(crate) struct ResultMapErrSite {
    pub result_var: Variable,
    pub receiver_owner: String,
    pub receiver_ok_owner: String,
    pub receiver_err_owner: String,
    pub result_owner: String,
    pub result_ok_owner: String,
    pub result_err_owner: String,
    pub call_once_owner: String,
    pub ok_ty: ValueType,
    pub ok_class_root: Option<String>,
    pub err_ty: ValueType,
    pub err_class_root: Option<String>,
    pub mapped_err_ty: ValueType,
    pub mapped_err_class_root: Option<String>,
    pub args_tuple_suffix: String,
    /// Set when the mapper is a function item (`map_err(named_fn)`), not a
    /// closure. The Err arm is then a direct call, the same shape as
    /// `Option::map(opt, named_fn)`.
    pub fn_item_segments: Option<Vec<String>>,
    /// True only when every captured field recursively needs no destructor.
    /// Other closure environments stay on the fail-closed path until MIR Drop
    /// lowering can preserve their conditional destruction on the Ok arm.
    pub closure_env_is_trivially_dropless: bool,
}

fn is_map_err_call(kind: &OpKind) -> bool {
    match kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } if args.len() == 2 && name == "map_err" => true,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.len() == 2 && segments.last().map(String::as_str) == Some("map_err") => true,
        _ => false,
    }
}

/// `(result, call is the block's last op)`.
fn map_err_call_results(graph: &FunctionGraph) -> Vec<(Variable, bool)> {
    let mut results = Vec::new();
    for block in &graph.blocks {
        let n = block.operations.len();
        for (index, op) in block.operations.iter().enumerate() {
            if is_map_err_call(&op.kind)
                && let Some(result) = op.result.clone()
            {
                results.push((result, index + 1 == n));
            }
        }
    }
    results
}

pub(crate) fn rewire_result_map_err_sites(
    graph: &mut FunctionGraph,
    sites: &[ResultMapErrSite],
) -> usize {
    let calls = map_err_call_results(graph);
    // Simplify rewrites the call's result variable after the site was
    // recorded. When every recorded variable is gone, bind sites to the
    // surviving calls. Prefer calls that are still the block's last op:
    // an inlined helper can leave a second `map_err` mid-block.
    let stale = sites
        .iter()
        .all(|site| !calls.iter().any(|(result, _)| result == &site.result_var));
    let tail: Vec<&Variable> = calls
        .iter()
        .filter(|(_, is_tail)| *is_tail)
        .map(|(result, _)| result)
        .collect();
    let all: Vec<&Variable> = calls.iter().map(|(result, _)| result).collect();
    let binders: Option<&[&Variable]> = if stale && sites.len() == tail.len() {
        Some(tail.as_slice())
    } else if stale && sites.len() == all.len() {
        Some(all.as_slice())
    } else {
        None
    };
    let rebound: Vec<ResultMapErrSite> = if let Some(binders) = binders {
        sites
            .iter()
            .zip(binders)
            .map(|(site, result)| {
                let mut site = site.clone();
                site.result_var = (*result).clone();
                site
            })
            .collect()
    } else {
        sites.to_vec()
    };
    rebound
        .iter()
        .filter(|site| rewire_one(graph, site).is_ok())
        .count()
}

fn rewire_one(graph: &mut FunctionGraph, site: &ResultMapErrSite) -> Result<(), String> {
    let name = graph.name.clone();
    if !site.closure_env_is_trivially_dropless {
        return Err(format!(
            "{name}: Result::map_err closure captures values whose Ok-arm destruction is not lowered"
        ));
    }
    let a = graph
        .blocks
        .iter()
        .position(|block| {
            block
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: Result::map_err result has no producer block"))?;
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: Result::map_err producer vanished"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: Result::map_err is not the block's last op"
        ));
    }
    let (receiver, env) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } if name == "map_err" && args.len() == 2 => (args[0].clone(), args[1].clone()),
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.len() == 2 && segments.last().map(String::as_str) == Some("map_err") => {
            (args[0].clone(), args[1].clone())
        }
        other => return Err(format!("{name}: recorded map_err site changed: {other:?}")),
    };
    // When the mapped Result is consumed by `?`, result_exc runs first and
    // changes this call block to LastException exits.  The map_err rewrite
    // then completes that same decision: receiver Ok forwards its payload to
    // the normal edge, receiver Err calls the mapper and raises the mapped
    // value.  A plain map_err consumer keeps the value-encoded Result arms.
    let exception_lowered = matches!(
        graph.blocks[a].exitswitch,
        Some(crate::model::ExitSwitch::LastException)
    );
    let (saved_exit, saved_exception_exit) = if exception_lowered {
        if graph.blocks[a].exits.len() != 2 {
            return Err(format!(
                "{name}: Result::map_err LastException block does not have two exits"
            ));
        }
        let normal = graph.blocks[a]
            .exits
            .iter()
            .find(|exit| exit.exitcase.is_none())
            .cloned()
            .ok_or_else(|| format!("{name}: Result::map_err has no normal exception edge"))?;
        let exceptional = graph.blocks[a]
            .exits
            .iter()
            .find(|exit| exit.exitcase.is_some())
            .cloned()
            .ok_or_else(|| format!("{name}: Result::map_err has no exceptional edge"))?;
        (normal, Some(exceptional))
    } else {
        let [exit] = graph.blocks[a].exits.as_slice() else {
            return Err(format!("{name}: Result::map_err block has multiple exits"));
        };
        if graph.blocks[a].exitswitch.is_some()
            || exit.exitcase.is_some()
            || exit.last_exception.is_some()
            || exit.last_exc_value.is_some()
        {
            return Err(format!(
                "{name}: Result::map_err block exit is not a plain goto"
            ));
        }
        (exit.clone(), None)
    };
    let target = saved_exit.target;
    let mut carried = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(value) = arg
            && *value != site.result_var
            && !carried.contains(value)
        {
            carried.push(value.clone());
        }
    }

    let mut ok_sources = carried.clone();
    if !ok_sources.contains(&receiver) {
        ok_sources.push(receiver.clone().into_variable());
    }
    let mut err_sources = ok_sources.clone();
    if site.fn_item_segments.is_none() && !err_sources.contains(&env) {
        err_sources.push(env.clone().into_variable());
    }
    // Validate the complete exception destination before adding any blocks.
    // Like ExceptionTransformer.transform_block / insert_matching, a rejected
    // transformation must leave the original exception graph intact. Identity
    // inputs let the rewrap validator check the same mappings without issuing
    // fresh SSA variables or populating orphan call_once operations.
    if let Some(exceptional) = &saved_exception_exit {
        let last_exception = exceptional
            .last_exception
            .as_ref()
            .and_then(LinkArg::as_variable)
            .ok_or_else(|| format!("{name}: exceptional map_err edge lacks last_exception"))?;
        let last_exc_value = exceptional
            .last_exc_value
            .as_ref()
            .and_then(LinkArg::as_variable)
            .ok_or_else(|| format!("{name}: exceptional map_err edge lacks last_exc_value"))?;
        for arg in &exceptional.args {
            if let LinkArg::Value(value) = arg
                && value != last_exception
                && value != last_exc_value
                && !err_sources.contains(value)
            {
                return Err(format!(
                    "{name}: exceptional map_err edge carries an unthreaded value"
                ));
            }
        }
        if exceptional.target != graph.exceptblock {
            bypass_rewrap_handler_args(
                graph,
                exceptional,
                &site.result_var,
                &err_sources,
                &err_sources,
                &site.result_err_owner,
                &name,
            )?;
        }
    }
    let (ok_block, ok_inputs) = graph.create_block_with_arg_vars(ok_sources.len());
    let (err_block, err_inputs) = graph.create_block_with_arg_vars(err_sources.len());

    // `Ok(())`: no `__pos_0` field (`result_exc` `unit_ok`,
    // `pairtype(Repr, VoidRepr).convert_from_to` Constant(None, Void)).
    let ok_is_void = matches!(site.ok_ty, ValueType::Void);
    let ok_payload = if ok_is_void {
        emit_void_unit(graph, ok_block)
    } else {
        let receiver_ok = map_source(&ok_sources, &ok_inputs, &receiver)
            .ok_or_else(|| format!("{name}: Result receiver not threaded into Ok arm"))?;
        read_payload(
            graph,
            ok_block,
            receiver_ok,
            &site.receiver_ok_owner,
            site.ok_ty.clone(),
        )
    };
    let ok_payload = emit_narrow(graph, ok_block, ok_payload, &site.ok_class_root);
    let ok_result = if exception_lowered {
        ok_payload
    } else {
        emit_sum_variant(
            graph,
            ok_block,
            &site.result_owner,
            "Ok",
            0,
            if ok_is_void {
                None
            } else {
                Some((&site.result_ok_owner, ok_payload, site.ok_ty.clone()))
            },
        )
    };
    let ok_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &ok_result,
        &ok_sources,
        &ok_inputs,
        &name,
    )?;
    close_goto_mixed(graph, ok_block, target, ok_args);

    let receiver_err = map_source(&err_sources, &err_inputs, &receiver)
        .ok_or_else(|| format!("{name}: Result receiver not threaded into Err arm"))?;
    let err_payload = read_payload(
        graph,
        err_block,
        receiver_err,
        &site.receiver_err_owner,
        site.err_ty.clone(),
    );
    let err_payload = emit_narrow(graph, err_block, err_payload, &site.err_class_root);
    let env_err = if site.fn_item_segments.is_some() {
        None
    } else {
        Some(
            map_source(&err_sources, &err_inputs, &env)
                .ok_or_else(|| format!("{name}: closure env not threaded into Err arm"))?,
        )
    };
    let mapped = emit_callable(
        graph,
        err_block,
        env_err,
        &site.call_once_owner,
        site.fn_item_segments.as_deref(),
        None,
        Some((
            err_payload,
            site.err_ty.clone(),
            site.err_class_root.clone(),
        )),
        site.mapped_err_ty.clone(),
        &site.args_tuple_suffix,
    )?;
    let mapped = emit_narrow(graph, err_block, mapped, &site.mapped_err_class_root);
    if exception_lowered {
        // result_exc has already made the map_err call a can-raise site. Keep
        // that exact exceptional destination: a `?` site targets exceptblock,
        // while catch_and_rewrap targets its local Err-shell rebuilding arm.
        let exceptional = saved_exception_exit
            .as_ref()
            .expect("LastException form captured its exceptional edge");
        if exceptional.target != graph.exceptblock {
            // `catch_and_rewrap` only converted the old call into an exception
            // edge so its local handler could recreate the value-encoded
            // Result.  The mapper has already produced the carrier that handler
            // would reconstruct; build the Err shell here and bypass the
            // carrier -> exception object -> carrier round trip.  Besides an
            // avoidable allocation, PyError::from_exc_object cannot preserve
            // carrier-only state such as attach_tb or reraise_lasti.
            let err_result = emit_sum_variant(
                graph,
                err_block,
                &site.result_owner,
                "Err",
                1,
                Some((&site.result_err_owner, mapped, site.mapped_err_ty.clone())),
            );
            let (rewrap_target, rewrap_args) = bypass_rewrap_handler_args(
                graph,
                exceptional,
                &err_result,
                &err_sources,
                &err_inputs,
                &site.result_err_owner,
                &name,
            )?;
            close_goto_mixed(graph, err_block, rewrap_target, rewrap_args);
        } else {
            // `raise mapped`. `exc_from_raise` records `etype = type(mapped)`
            // and the exceptblock link. `codewriter::error_carrier_edges`
            // rewrites that raise to `to_exc_object(mapped)`.
            crate::front::exc_from_raise::set_raise_from_instance(graph, err_block, mapped);
        }
    } else {
        let err_result = emit_sum_variant(
            graph,
            err_block,
            &site.result_owner,
            "Err",
            1,
            Some((&site.result_err_owner, mapped, site.mapped_err_ty.clone())),
        );
        let err_args = reproduce_exit_args(
            &saved_exit,
            &site.result_var,
            &err_result,
            &err_sources,
            &err_inputs,
            &name,
        )?;
        close_goto_mixed(graph, err_block, target, err_args);
    }

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.truncate(call_idx);
    let disc = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: receiver.into_variable(),
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(site.receiver_owner.clone()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            ty: ValueType::Int,
            pure: true,
        },
    });
    graph.set_branch(a_id, disc, err_block, err_sources, ok_block, ok_sources);
    Ok(())
}

/// Close `block` with `raise carrier`.
///
/// `exc_from_raise` records `etype = type(carrier)` and the `(etype, evalue)`
/// link to `exceptblock`. `codewriter::error_carrier_edges` rewrites a raise
/// of that shape to `to_exc_object(carrier)`. Copying `carrier` into both
/// exception slots leaves no `type(evalue)` producer, so that pass treats the
/// link as an already-materialised propagation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raise_carrier_on_exception_edge(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    carrier: Variable,
    exceptional: &crate::model::Link,
    _sources: &[Variable],
    _inputs: &[Variable],
    _spec: crate::ErrorCarrierSpec<'_>,
    name: &str,
) -> Result<(), String> {
    if exceptional.target != graph.exceptblock {
        return Err(format!("{name}: carrier raise does not target exceptblock"));
    }
    if exceptional
        .last_exception
        .as_ref()
        .and_then(LinkArg::as_variable)
        .is_none()
        || exceptional
            .last_exc_value
            .as_ref()
            .and_then(LinkArg::as_variable)
            .is_none()
    {
        return Err(format!("{name}: exceptional edge lacks its exception pair"));
    }
    crate::front::exc_from_raise::set_raise_from_instance(graph, block, carrier);
    Ok(())
}

/// Bypass the local handler installed by `result_exc::catch_and_rewrap`.
///
/// That handler receives the original non-result live values followed by the
/// exception pair, converts `last_exc_value` back into the error carrier, and
/// builds the Err shell consumed by the untouched custom match.  A lowered
/// `map_err` Err arm already owns the mapped carrier, so route a shell built
/// from that exact value to the handler's successor and remap the other live
/// values through this arm's inputargs.
fn bypass_rewrap_handler_args(
    graph: &FunctionGraph,
    exceptional: &crate::model::Link,
    err_result: &Variable,
    err_sources: &[Variable],
    err_inputs: &[Variable],
    result_err_owner: &str,
    name: &str,
) -> Result<(crate::model::BlockId, Vec<LinkArg>), String> {
    let handler = &graph.blocks[exceptional.target.0];
    if handler.exitswitch.is_some() || handler.exits.len() != 1 {
        return Err(format!(
            "{name}: map_err rewrap handler is not a single plain forwarding block"
        ));
    }
    let exit = &handler.exits[0];
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: map_err rewrap handler exit is not a plain goto"
        ));
    }
    let shell = handler
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::Call { target, .. }
                if crate::front::result_exc::result_ctor_kind(target) == Some(true)
                    && op.result.as_ref().is_some_and(|result| {
                        handler.operations.iter().any(|write| {
                            matches!(
                                &write.kind,
                                OpKind::FieldWrite { base, field, .. }
                                    if base == result
                                        && field.name == "__pos_0"
                                        && field.owner_root.as_deref() == Some(result_err_owner)
                            )
                        })
                    }) =>
            {
                op.result.clone()
            }
            _ => None,
        })
        .ok_or_else(|| format!("{name}: map_err rewrap handler lacks its Err shell"))?;
    if exceptional.args.len() != handler.inputargs.len() {
        return Err(format!(
            "{name}: map_err rewrap handler input arity does not match its exceptional edge"
        ));
    }
    let mut args = Vec::with_capacity(exit.args.len());
    for arg in &exit.args {
        match arg {
            LinkArg::Const(value) => args.push(LinkArg::Const(value.clone())),
            LinkArg::Value(value) if *value == shell => {
                args.push(LinkArg::Value(err_result.clone()))
            }
            LinkArg::Value(value) => {
                let pos = handler
                    .inputargs
                    .iter()
                    .position(|input| input == value)
                    .ok_or_else(|| {
                        format!("{name}: map_err rewrap handler forwards an unknown value")
                    })?;
                let LinkArg::Value(source) = &exceptional.args[pos] else {
                    return Err(format!(
                        "{name}: map_err rewrap handler input is sourced from a constant"
                    ));
                };
                let remapped = map_source(err_sources, err_inputs, source).ok_or_else(|| {
                    format!("{name}: map_err rewrap handler carries an unthreaded value")
                })?;
                args.push(LinkArg::Value(remapped));
            }
        }
    }
    Ok((exit.target, args))
}

fn emit_void_unit(graph: &mut FunctionGraph, block: crate::model::BlockId) -> Variable {
    let unit = graph.alloc_value_var_with_type(crate::model::ConcreteType::Void);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(unit.clone()),
        kind: OpKind::ConstNone,
    });
    unit
}

fn read_payload(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    receiver: Variable,
    owner: &str,
    ty: ValueType,
) -> Variable {
    let payload = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: receiver,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(owner.to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            ty,
            pure: true,
        },
    });
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_exception_edges_leave_the_graph_unchanged() {
        for defect in ["last_exception", "last_exc_value", "unthreaded", "rewrap"] {
            let mut graph = FunctionGraph::new("map_err_invalid_exception");
            let entry = graph.startblock;
            let receiver = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
            let env = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
            let result = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::method("map_err", Some("Result".into())),
                        args: crate::model::call_args(vec![receiver, env]),
                        result_ty: ValueType::Ref(None),
                    },
                    true,
                )
                .unwrap();
            let (normal, _) = graph.create_block_with_arg_vars(1);
            graph.set_return(normal, None);
            let etype = graph.alloc_value_var();
            let evalue = graph.alloc_value_var();
            let mut exceptional = crate::model::Link::new_mixed(
                vec![
                    LinkArg::Value(etype.clone()),
                    LinkArg::Value(evalue.clone()),
                ],
                graph.exceptblock,
                Some(crate::model::exception_exitcase()),
            );
            exceptional.last_exception = Some(LinkArg::Value(etype));
            exceptional.last_exc_value = Some(LinkArg::Value(evalue));
            match defect {
                "last_exception" => exceptional.last_exception = None,
                "last_exc_value" => exceptional.last_exc_value = None,
                "unthreaded" => exceptional
                    .args
                    .push(LinkArg::Value(graph.alloc_value_var())),
                "rewrap" => exceptional.target = normal,
                _ => unreachable!(),
            }
            graph.set_control_flow_metadata(
                entry,
                Some(crate::model::ExitSwitch::LastException),
                vec![
                    crate::model::Link::new_mixed(
                        vec![LinkArg::Value(result.clone())],
                        normal,
                        None,
                    ),
                    exceptional,
                ],
            );
            let mut site = fixture_site(result);
            site.closure_env_is_trivially_dropless = true;
            let before = format!("{graph:?}");
            assert_eq!(
                rewire_result_map_err_sites(&mut graph, &[site]),
                0,
                "{defect}"
            );
            assert_eq!(format!("{graph:?}"), before, "{defect} mutated the graph");
        }
    }

    fn fixture_site(result_var: Variable) -> ResultMapErrSite {
        ResultMapErrSite {
            result_var,
            receiver_owner: "core::result::Result<i64,str>".into(),
            receiver_ok_owner: "core::result::Result<i64,str>::Ok".into(),
            receiver_err_owner: "core::result::Result<i64,str>::Err".into(),
            result_owner: "core::result::Result<i64,Error>".into(),
            result_ok_owner: "core::result::Result<i64,Error>::Ok".into(),
            result_err_owner: "core::result::Result<i64,Error>::Err".into(),
            call_once_owner: "fixture::closure".into(),
            ok_ty: ValueType::Int,
            ok_class_root: None,
            err_ty: ValueType::Str,
            err_class_root: None,
            mapped_err_ty: ValueType::Ref(Some("Error".into())),
            mapped_err_class_root: Some("Error".into()),
            args_tuple_suffix: "<str>".into(),
            fn_item_segments: None,
            closure_env_is_trivially_dropless: false,
        }
    }

    #[test]
    fn map_err_becomes_ok_passthrough_and_err_closure_arms() {
        let mut graph = FunctionGraph::new("map_err_fixture");
        let entry = graph.startblock;
        let receiver = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let env = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::method("map_err", Some("Result".into())),
                    args: crate::model::call_args(vec![receiver, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (join, _inputs) = graph.create_block_with_arg_vars(1);
        graph.set_return(join, None);
        graph.set_goto(entry, join, vec![result.clone()]);

        let mut site = fixture_site(result);
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site.clone()]), 0);
        assert!(graph.blocks[entry.0].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "map_err"
            )
        }));

        site.closure_env_is_trivially_dropless = true;
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site]), 1);

        let calls: Vec<&CallTarget> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target),
                _ => None,
            })
            .collect();
        assert!(
            !calls.iter().any(
                |target| matches!(target, CallTarget::Method { name, .. } if name == "map_err")
            )
        );
        assert_eq!(
            calls
                .iter()
                .filter(|target| matches!(target, CallTarget::Method { name, .. } if name == "call_once"))
                .count(),
            1,
            "only the Err arm invokes the mapper"
        );
        for variant in ["Ok", "Err"] {
            assert!(calls.iter().any(|target| {
                matches!(target, CallTarget::SyntheticTransparentCtor { name, .. } if name == variant)
            }));
        }
    }

    #[test]
    fn map_err_unit_ok_skips_payload_read() {
        let mut graph = FunctionGraph::new("map_err_unit_ok");
        let entry = graph.startblock;
        let receiver = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let env = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::method("map_err", Some("Result".into())),
                    args: crate::model::call_args(vec![receiver, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (join, _inputs) = graph.create_block_with_arg_vars(1);
        graph.set_return(join, None);
        graph.set_goto(entry, join, vec![result.clone()]);

        let mut site = fixture_site(result);
        site.ok_ty = ValueType::Void;
        site.receiver_owner = "core::result::Result<(),str>".into();
        site.receiver_ok_owner = "core::result::Result<(),str>::Ok".into();
        site.receiver_err_owner = "core::result::Result<(),str>::Err".into();
        site.result_owner = "core::result::Result<(),Error>".into();
        site.result_ok_owner = "core::result::Result<(),Error>::Ok".into();
        site.result_err_owner = "core::result::Result<(),Error>::Err".into();
        site.closure_env_is_trivially_dropless = true;
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site]), 1);

        let ok_arm = graph
            .blocks
            .iter()
            .find(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target: CallTarget::SyntheticTransparentCtor { name, .. }, .. }
                            if name == "Ok"
                    )
                })
            })
            .expect("Ok arm");
        assert!(
            !ok_arm.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. } if field.name == "__pos_0"
                )
            }),
            "unit Ok has no payload field"
        );
        assert!(
            !ok_arm.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. } if field.name == "__pos_0"
                )
            }),
            "unit Ok shell stores no __pos_0"
        );
        assert!(
            ok_arm
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ConstNone))
        );
    }

    #[test]
    fn exception_lowered_map_err_bypasses_rewrap_without_carrier_round_trip() {
        let mut graph = FunctionGraph::new("map_err_catch_fixture");
        let entry = graph.startblock;
        let receiver = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let env = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let carried = graph.push_op_var(entry, OpKind::ConstInt(3), true).unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::method("map_err", Some("Result".into())),
                    args: crate::model::call_args(vec![receiver, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (normal, _) = graph.create_block_with_arg_vars(2);
        graph.set_return(normal, None);
        let (catch, catch_inputs) = graph.create_block_with_arg_vars(3);
        let reconstructed = graph
            .push_op_var(
                catch,
                OpKind::Call {
                    target: CallTarget::method("from_exc_object", Some("Error".into())),
                    args: crate::model::call_args(vec![catch_inputs[2].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let old_shell = crate::front::result_exc::build_shell(
            &mut graph,
            catch,
            "Err",
            reconstructed,
            ValueType::Ref(None),
            "<i64,Error>",
        );
        let (downstream, _) = graph.create_block_with_arg_vars(2);
        graph.set_return(downstream, None);
        graph.set_goto(catch, downstream, vec![catch_inputs[0].clone(), old_shell]);
        let etype = graph.alloc_value_var();
        let evalue = graph.alloc_value_var();
        let mut exceptional = crate::model::Link::new_mixed(
            vec![
                LinkArg::Value(carried.clone()),
                LinkArg::Value(etype.clone()),
                LinkArg::Value(evalue.clone()),
            ],
            catch,
            Some(crate::model::exception_exitcase()),
        );
        exceptional.last_exception = Some(LinkArg::Value(etype));
        exceptional.last_exc_value = Some(LinkArg::Value(evalue));
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::Value(result.clone()),
                        LinkArg::Value(carried.clone()),
                    ],
                    normal,
                    None,
                ),
                exceptional,
            ],
        );

        let mut site = fixture_site(result);
        site.closure_env_is_trivially_dropless = true;
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site]), 1);

        let err_arm = graph
            .blocks
            .iter()
            .find(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                            if name == "call_once"
                    )
                })
            })
            .expect("Err arm must call the mapper");
        assert!(!err_arm.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["fixture".to_string(), "to_exc_object".to_string()]
            )
        }));
        assert_eq!(err_arm.exits.len(), 1);
        assert_eq!(err_arm.exits[0].target, downstream);
        assert_ne!(err_arm.exits[0].target, catch);
        assert_eq!(err_arm.exits[0].args.len(), 2);
        let carried_alias = err_arm.exits[0].args[0]
            .as_variable()
            .expect("carried value stays a Variable");
        assert!(err_arm.inputargs.contains(carried_alias));
        assert!(err_arm.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, value: LinkArg::Value(_), .. }
                    if field.owner_root.as_deref()
                        == Some("core::result::Result<i64,Error>::Err")
            )
        }));
    }

    #[test]
    fn exception_lowered_map_err_raises_the_mapped_carrier() {
        let mut graph = FunctionGraph::new("map_err_propagate_fixture");
        let entry = graph.startblock;
        let receiver = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let env = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::method("map_err", Some("Result".into())),
                    args: crate::model::call_args(vec![receiver, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (normal, _) = graph.create_block_with_arg_vars(1);
        graph.set_return(normal, None);
        let etype = graph.alloc_value_var();
        let evalue = graph.alloc_value_var();
        let mut exceptional = crate::model::Link::new_mixed(
            vec![
                LinkArg::Value(etype.clone()),
                LinkArg::Value(evalue.clone()),
            ],
            graph.exceptblock,
            Some(crate::model::exception_exitcase()),
        );
        exceptional.last_exception = Some(LinkArg::Value(etype));
        exceptional.last_exc_value = Some(LinkArg::Value(evalue));
        graph.set_control_flow_metadata(
            entry,
            Some(crate::model::ExitSwitch::LastException),
            vec![
                crate::model::Link::new_mixed(vec![LinkArg::Value(result.clone())], normal, None),
                exceptional,
            ],
        );

        let mut site = fixture_site(result);
        site.closure_env_is_trivially_dropless = true;
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site]), 1);

        let err_arm = graph
            .blocks
            .iter()
            .find(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                            if name == "call_once"
                    )
                })
            })
            .expect("Err arm must call the mapper");
        // `raise mapped`: `etype = type(mapped)`, then the exceptblock link.
        // The codewriter rewrites that raise to `to_exc_object(mapped)`.
        let type_of = err_arm
            .operations
            .last()
            .expect("type(mapped) closes the raise");
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &type_of.kind
        else {
            panic!("raise records type(mapped)");
        };
        assert_eq!(segments.as_slice(), ["type".to_string()]);
        let mapped = args[0].as_variable().expect("type() reads the carrier");
        assert_eq!(err_arm.exits.len(), 1);
        assert_eq!(err_arm.exits[0].target, graph.exceptblock);
        assert_eq!(
            err_arm.exits[0].args[0].as_variable(),
            type_of.result.as_ref()
        );
        assert_eq!(err_arm.exits[0].args[1].as_variable(), Some(mapped));
    }
}
