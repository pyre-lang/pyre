//! Restore core::result::Result::as_ref's two value branches from opaque LLBC.
//! The admitted payloads already have immutable RPython value representations
//! (rpython/rtyper/rint.py::IntegerRepr and
//! rpython/rtyper/lltypesystem/rstr.py::StringRepr). No native borrowed address
//! is produced.
//! Keep distinct source/destination enum owners, as rclass.InstanceRepr does.

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_sum_variant, map_source, reproduce_exit_args,
};
use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};

#[derive(Clone)]
pub(crate) struct ResultAsRefSite {
    pub result_var: Variable,
    pub receiver_owner: String,
    pub receiver_variants: [String; 2],
    pub result_owner: String,
    pub result_variants: [String; 2],
    pub payload_types: [ValueType; 2],
}

pub(crate) fn rewire_result_as_ref_sites(
    graph: &mut FunctionGraph,
    sites: &[ResultAsRefSite],
) -> usize {
    sites
        .iter()
        .filter(|site| rewire_one(graph, site).is_ok())
        .count()
}

fn rewire_one(graph: &mut FunctionGraph, site: &ResultAsRefSite) -> Result<(), String> {
    let a = graph
        .blocks
        .iter()
        .position(|block| {
            block
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or("Result::as_ref producer missing")?;
    // `tyref_to_value_type` paints split-eligible Result as
    // `Ref(Some(root))` (`bookkeeper.py` `getuniqueclassdef`), so
    // `lower_call` recasts the as_ref result (`result_narrow_root`).
    // The recast is identity (`cast_pointer`); this rewrite needs the
    // residual call last. Same collapse the checked-arith Option path
    // uses (`collapse_trailing_recasts_onto`).
    crate::front::result_exc::collapse_trailing_recasts_onto(graph, a, &site.result_var);
    let Some(op) = graph.blocks[a].operations.last() else {
        return Err("empty block".into());
    };
    if op.result.as_ref() != Some(&site.result_var) {
        return Err("as_ref is not last".into());
    }
    let receiver = match &op.kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone().into_variable(),
        _ => return Err("as_ref does not have one argument".into()),
    };
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err("as_ref needs one exit".into());
    };
    if graph.blocks[a].exitswitch.is_some()
        || exit.exitcase.is_some()
        || exit.last_exception.is_some()
        || exit.last_exc_value.is_some()
    {
        return Err("as_ref needs a plain continuation".into());
    }
    let saved_exit = exit.clone();
    let mut sources = vec![receiver.clone()];
    for arg in &saved_exit.args {
        if let LinkArg::Value(value) = arg
            && *value != site.result_var
            && !sources.contains(value)
        {
            sources.push(value.clone());
        }
    }
    let (ok, ok_inputs) = graph.create_block_with_arg_vars(sources.len());
    let (err, err_inputs) = graph.create_block_with_arg_vars(sources.len());
    for (tag, variant, block, inputs) in [(0, "Ok", ok, ok_inputs), (1, "Err", err, err_inputs)] {
        let base = map_source(&sources, &inputs, &receiver).expect("receiver is carried");
        let value = graph
            .push_op_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new(
                        "__pos_0",
                        Some(site.receiver_variants[tag].clone()),
                    ),
                    ty: site.payload_types[tag].clone(),
                    pure: true,
                },
                true,
            )
            .expect("payload read has a result");
        let result = emit_sum_variant(
            graph,
            block,
            &site.result_owner,
            variant,
            tag as i64,
            Some((
                &site.result_variants[tag],
                value,
                site.payload_types[tag].clone(),
            )),
        );
        let args = reproduce_exit_args(
            &saved_exit,
            &site.result_var,
            &result,
            &sources,
            &inputs,
            &graph.name,
        )?;
        close_goto_mixed(graph, block, saved_exit.target, args);
    }
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.pop();
    let disc = graph
        .push_op_var(
            a_id,
            OpKind::FieldRead {
                base: receiver.clone(),
                field: FieldDescriptor::new("__discriminant", Some(site.receiver_owner.clone())),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        )
        .expect("tag read has a result");
    // core::result::Result::as_ref: Ok(ref x) => Ok(x), Err(ref x) => Err(x).
    graph.set_branch(a_id, disc, err, sources.clone(), ok, sources);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    #[test]
    fn borrowed_result_keeps_both_payloads_and_distinct_variant_owners() {
        let mut graph = FunctionGraph::new("borrow_result");
        let receiver = graph.alloc_value_var();
        let live = graph.alloc_value_var();
        graph.block_mut(graph.startblock).inputargs = vec![receiver.clone(), live.clone()];
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("as_ref", Some("Result".into())),
                    args: crate::model::call_args(vec![receiver]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (continuation, inputs) = graph.create_block_with_arg_vars(2);
        let start = graph.startblock;
        graph.set_goto(start, continuation, vec![result.clone(), live]);
        graph.set_return(continuation, Some(inputs[0].clone()));
        let site = ResultAsRefSite {
            result_var: result,
            receiver_owner: "Result<i64,String>".into(),
            receiver_variants: [
                "Result<i64,String>::Ok".into(),
                "Result<i64,String>::Err".into(),
            ],
            result_owner: "Result<&i64,&String>".into(),
            result_variants: [
                "Result<&i64,&String>::Ok".into(),
                "Result<&i64,&String>::Err".into(),
            ],
            payload_types: [ValueType::Int, ValueType::Str],
        };
        assert_eq!(rewire_result_as_ref_sites(&mut graph, &[site.clone()]), 1);
        // bool(tag): true is Err=1, false is Ok=0. Each branch copies its
        // own payload into the destination's actual variant, never aliases
        // two source-level type identities onto a single object.
        for edge in &graph.block(start).exits {
            let tag = match edge.exitcase {
                Some(crate::model::ExitCase::Bool(value)) => usize::from(value),
                _ => panic!("as_ref must have boolean branch edges"),
            };
            let block = graph.block(edge.target);
            assert!(block.operations.iter().any(|op| matches!(&op.kind,
                OpKind::FieldRead { field, ty, .. }
                if field.owner_root.as_deref() == Some(site.receiver_variants[tag].as_str())
                    && *ty == site.payload_types[tag])));
            assert!(block.operations.iter().any(|op| matches!(&op.kind,
                OpKind::FieldWrite { field, ty, .. }
                if field.owner_root.as_deref() == Some(site.result_variants[tag].as_str())
                    && *ty == site.payload_types[tag])));
            assert_eq!(block.exits[0].target, continuation);
            assert_eq!(block.exits[0].args.len(), 2);
        }
    }

    #[test]
    fn borrowed_result_rewrites_through_instantiation_recast() {
        // `tyref_to_value_type` paints split-eligible Result as
        // `Ref(Some(root))`, so `lower_call` recasts the as_ref dest
        // (`result_narrow_root`). The Option sibling is
        // `rewrite_finds_checked_add_behind_instantiation_recast`.
        let mut graph = FunctionGraph::new("borrow_result_recast");
        let receiver = graph.alloc_value_var();
        let live = graph.alloc_value_var();
        graph.block_mut(graph.startblock).inputargs = vec![receiver.clone(), live.clone()];
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("as_ref", Some("Result".into())),
                    args: crate::model::call_args(vec![receiver]),
                    result_ty: ValueType::Ref(Some("Result<i64,String>".into())),
                },
                true,
            )
            .unwrap();
        let recast = graph
            .push_op_var(
                graph.startblock,
                crate::model::cast_instance_call("Result<i64,String>", result.clone()),
                true,
            )
            .unwrap();
        let (continuation, inputs) = graph.create_block_with_arg_vars(2);
        let start = graph.startblock;
        graph.set_goto(start, continuation, vec![recast, live]);
        graph.set_return(continuation, Some(inputs[0].clone()));
        let site = ResultAsRefSite {
            result_var: result,
            receiver_owner: "Result<i64,String>".into(),
            receiver_variants: [
                "Result<i64,String>::Ok".into(),
                "Result<i64,String>::Err".into(),
            ],
            result_owner: "Result<&i64,&String>".into(),
            result_variants: [
                "Result<&i64,&String>::Ok".into(),
                "Result<&i64,&String>::Err".into(),
            ],
            payload_types: [ValueType::Int, ValueType::Str],
        };
        assert_eq!(rewire_result_as_ref_sites(&mut graph, &[site.clone()]), 1);
        let start_block = graph.block(start);
        assert!(
            start_block
                .operations
                .iter()
                .all(|op| crate::model::cast_instance_root(&op.kind).is_none()),
            "the Result classdef recast is collapsed"
        );
        assert!(start_block.operations.iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if field.name == "__discriminant"
        )));
        for edge in &start_block.exits {
            let tag = match edge.exitcase {
                Some(crate::model::ExitCase::Bool(value)) => usize::from(value),
                _ => panic!("as_ref must have boolean branch edges"),
            };
            let block = graph.block(edge.target);
            assert!(block.operations.iter().any(|op| matches!(&op.kind,
                OpKind::FieldRead { field, ty, .. }
                if field.owner_root.as_deref() == Some(site.receiver_variants[tag].as_str())
                    && *ty == site.payload_types[tag])));
            assert_eq!(block.exits[0].target, continuation);
        }
    }
}
