//! The error carrier's exception edges, converted to the runtime exception
//! value.
//!
//! The front raises and catches the interpreter's error carrier
//! (`crate::ErrorCarrierSpec`) the way PyPy raises and catches
//! `OperationError`: `return Err(e)` becomes `raise e`, and a handler that
//! reads the error catches with the carrier class
//! ([`ExitCase::ErrorCarrier`]) and receives the carrier.  The annotator
//! and rtyper see that shape.
//!
//! The runtime exception value is a different object from the carrier.
//! The residual-call ABI stores the carrier's `to_exc_object`
//! materialisation in `BH_LAST_EXC_VALUE`, and `bh_classof` reads the
//! class off that object.  This pass rewrites the carrier's edges into
//! that domain on the graph the codewriter consumes, after annotation.
//! It is the carrier-level analogue of `translator/exceptiontransform.py`
//! fixing the exception representation after rtyping:
//!
//! - A direct raise `raise e` becomes `raise to_exc_object(e)`.
//! - An `except OperationError as e` link lands on a fresh block computing
//!   `e = from_exc_object(last_exc_value)`, and the link becomes catch-all.
//!   Every exception the runtime delivers is carrier-derived, so the class
//!   test the carrier exitcase asks for always holds.
//!
//! With the conversion in place, `fuse_kind_ctor_raise` folds a raise site's
//! literal-message constructor into its materialisation.
//!
//! A pipeline that names no carrier leaves the graph untouched.  So does a
//! carrier that is its own exception value (no `to_exc_object` /
//! `from_exc_object` declared), apart from turning its exitcase into the
//! catch-all.

use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, ConcreteType, ExitCase, FunctionGraph, Link, LinkArg, OpKind, ValueType,
};

pub fn lower_error_carrier_edges(
    graph: &mut FunctionGraph,
    carrier: &crate::OwnedErrorCarrierSpec,
) {
    if carrier.carrier_path.is_empty() {
        return;
    }
    let from_exc_object = carrier.from_exc_object.as_ref().map(|(_, method)| {
        let owner = crate::front::mir::strip_crate_prefix(&carrier.carrier_path);
        crate::parse::CallPath::for_impl_method(&owner, method).segments
    });
    lower_carrier_catches(graph, from_exc_object.as_deref());
    if let Some(to_exc_object) = carrier.to_exc_object.as_deref() {
        lower_carrier_raises(graph, to_exc_object);
        crate::front::result_exc::fuse_kind_ctor_raise(graph);
    }
}

/// `except OperationError as e` → catch-all link landing on
/// `e = from_exc_object(last_exc_value)`.
fn lower_carrier_catches(graph: &mut FunctionGraph, from_exc_object: Option<&[String]>) {
    for bi in 0..graph.blocks.len() {
        for li in 0..graph.blocks[bi].exits.len() {
            if graph.blocks[bi].exits[li].exitcase != Some(ExitCase::ErrorCarrier) {
                continue;
            }
            if let Some(segments) = from_exc_object {
                let link = graph.blocks[bi].exits[li].clone();
                if let Some(landing) = land_from_exc_object(graph, &link, segments) {
                    graph.blocks[bi].exits[li].target = landing;
                }
            }
            let link = &mut graph.blocks[bi].exits[li];
            link.exitcase = Some(crate::model::exception_exitcase());
            link.llexitcase = None;
        }
    }
}

/// A block taking `link`'s arguments, converting the caught
/// `last_exc_value` into the carrier, and forwarding to `link.target`.
///
/// `None` when the handler never receives the caught value (`Err(_)`):
/// there is nothing to convert.
fn land_from_exc_object(
    graph: &mut FunctionGraph,
    link: &Link,
    segments: &[String],
) -> Option<BlockId> {
    let caught = link
        .last_exc_value
        .as_ref()
        .and_then(LinkArg::as_variable)
        .cloned()?;
    let pos = link
        .args
        .iter()
        .position(|arg| matches!(arg, LinkArg::Value(v) if *v == caught))?;
    let (landing, inputs) = graph.create_block_with_arg_vars(link.args.len());
    for (input, arg) in inputs.iter().zip(&link.args) {
        if let LinkArg::Value(v) = arg {
            input.set_concretetype(v.concretetype.borrow().clone());
        }
    }
    FunctionGraph::set_concretetype_of_inline(&inputs[pos], ConcreteType::GcRef);
    let carrier = push_call(graph, landing, segments, inputs[pos].clone());
    let forwarded: Vec<Variable> = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| match &link.args[i] {
            LinkArg::Value(v) if *v == caught => carrier.clone(),
            _ => input.clone(),
        })
        .collect();
    graph.set_goto(landing, link.target, forwarded);
    Some(landing)
}

/// `raise e` → `raise to_exc_object(e)`.
fn lower_carrier_raises(graph: &mut FunctionGraph, to_exc_object: &[String]) {
    let exceptblock = graph.exceptblock;
    for bi in 0..graph.blocks.len() {
        let block_id = graph.blocks[bi].id;
        let block = &graph.blocks[bi];
        if block_id == exceptblock || block.exitswitch.is_some() {
            continue;
        }
        let [link] = block.exits.as_slice() else {
            continue;
        };
        if link.target != exceptblock
            || link.last_exception.is_some()
            || link.last_exc_value.is_some()
        {
            continue;
        }
        let [LinkArg::Value(etype), LinkArg::Value(evalue)] = link.args.as_slice() else {
            continue;
        };
        let (etype, evalue) = (etype.clone(), evalue.clone());
        // A raise builds its type as `etype = type(evalue)` in the raising
        // block (`exc_from_raise`).  A goto forwarding a caught
        // `(etype, evalue)` pair is a propagation and already carries the
        // runtime exception value.
        let Some(idx) = block.operations.iter().position(|op| {
            op.result.as_ref() == Some(&etype)
                && matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, args, .. }
                        if segments.len() == 1
                            && segments[0] == "type"
                            && args.len() == 1
                            && args[0].as_variable() == Some(&evalue)
                )
        }) else {
            continue;
        };
        // The raise below takes the type of the converted value instead.
        if !variable_used_outside(graph, bi, idx, &etype) {
            graph.blocks[bi].operations.remove(idx);
        }
        let exc = push_call(graph, block_id, to_exc_object, evalue);
        crate::front::exc_from_raise::set_raise_from_instance(graph, block_id, exc);
    }
}

/// True when `var` is read anywhere other than as the defining op at
/// `graph.blocks[block].operations[def_idx]` and the raise link of `block`.
fn variable_used_outside(
    graph: &FunctionGraph,
    block: usize,
    def_idx: usize,
    var: &Variable,
) -> bool {
    graph.blocks.iter().enumerate().any(|(bi, b)| {
        b.operations.iter().enumerate().any(|(oi, op)| {
            !(bi == block && oi == def_idx)
                && crate::front::result_exc::op_operand_vars(&op.kind).contains(var)
        }) || (bi != block
            && b.exits
                .iter()
                .any(|l| l.args.iter().any(|a| a.as_variable() == Some(var))))
            || matches!(&b.exitswitch, Some(crate::model::ExitSwitch::Value(v)) if v == var)
    })
}

fn push_call(
    graph: &mut FunctionGraph,
    block: BlockId,
    segments: &[String],
    arg: Variable,
) -> Variable {
    let result = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::function_path(segments.iter().cloned()),
                args: crate::model::call_args(vec![arg]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .expect("a conversion call produces a value");
    FunctionGraph::set_concretetype_of_inline(&result, ConcreteType::GcRef);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ExitSwitch;

    fn carrier() -> crate::OwnedErrorCarrierSpec {
        crate::OwnedErrorCarrierSpec {
            carrier_path: "pyre_interpreter::error::PyError".into(),
            carrier_class: String::new(),
            carrier_wrappers: Vec::new(),
            to_exc_object: Some(vec![
                "pyre_interpreter".into(),
                "error".into(),
                "pyerror_to_exc_object".into(),
            ]),
            from_exc_object: Some(("PyError".into(), "from_exc_object".into())),
        }
    }

    fn call_segments(op: &SpaceOperation) -> Option<&[String]> {
        match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments),
            _ => None,
        }
    }

    use crate::model::SpaceOperation;

    /// `raise e` stores `to_exc_object(e)`, typed by `type()` of that value.
    #[test]
    fn a_carrier_raise_stores_the_materialised_exception_object() {
        let mut graph = FunctionGraph::new("raise_carrier");
        let entry = graph.startblock;
        let e = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["make_error"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        crate::front::exc_from_raise::set_raise_from_instance(&mut graph, entry, e.clone());
        lower_error_carrier_edges(&mut graph, &carrier());

        let block = &graph.blocks[entry.0];
        let names: Vec<_> = block
            .operations
            .iter()
            .map(|op| call_segments(op).map(|s| s.join("::")))
            .collect();
        assert_eq!(
            names,
            vec![
                Some("make_error".to_string()),
                Some("pyre_interpreter::error::pyerror_to_exc_object".to_string()),
                Some("type".to_string()),
            ],
            "the carrier's own `type(e)` gives way to `type(to_exc_object(e))`"
        );
        let exc = block.operations[1].result.clone().unwrap();
        let [_, LinkArg::Value(raised)] = block.exits[0].args.as_slice() else {
            panic!("raise link");
        };
        assert_eq!(*raised, exc);
        assert_eq!(block.exits[0].target, graph.exceptblock);
    }

    /// `except OperationError as e` lands on `e = from_exc_object(caught)`
    /// through a catch-all link.
    #[test]
    fn a_carrier_catch_lands_on_from_exc_object_through_a_catch_all_link() {
        let mut graph = FunctionGraph::new("catch_carrier");
        let entry = graph.startblock;
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["may_raise"]),
                args: Vec::new(),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        let (normal, _) = graph.create_block_with_arg_vars(0);
        graph.set_return(normal, None);
        let (handler, handler_in) = graph.create_block_with_arg_vars(2);
        graph.set_return(handler, Some(handler_in[1].clone()));
        let etype = graph.alloc_value_var();
        let evalue = graph.alloc_value_var();
        let mut exc = Link::new_mixed(
            vec![
                LinkArg::Value(etype.clone()),
                LinkArg::Value(evalue.clone()),
            ],
            handler,
            Some(crate::model::error_carrier_exitcase()),
        );
        exc.last_exception = Some(LinkArg::Value(etype));
        exc.last_exc_value = Some(LinkArg::Value(evalue));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![Link::new_mixed(Vec::new(), normal, None), exc],
        );
        lower_error_carrier_edges(&mut graph, &carrier());

        let link = &graph.blocks[entry.0].exits[1];
        assert!(link.catches_all_exceptions());
        let landing = &graph.blocks[link.target.0];
        assert_eq!(landing.operations.len(), 1);
        assert_eq!(
            call_segments(&landing.operations[0]).map(|s| s.join("::")),
            Some("error::PyError::from_exc_object".to_string()),
        );
        let carrier_value = landing.operations[0].result.clone().unwrap();
        assert_eq!(landing.exits[0].target, handler);
        assert_eq!(
            landing.exits[0].args[1].as_variable(),
            Some(&carrier_value),
            "the handler receives the carrier in the caught value's slot"
        );
    }

    /// `Err(_)`: the handler never receives the caught value, so the link
    /// only turns catch-all and keeps its target.
    #[test]
    fn a_carrier_catch_that_drops_the_caught_value_needs_no_landing() {
        let mut graph = FunctionGraph::new("catch_carrier_unused");
        let entry = graph.startblock;
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["may_raise"]),
                args: Vec::new(),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        let (normal, _) = graph.create_block_with_arg_vars(0);
        graph.set_return(normal, None);
        let (handler, _) = graph.create_block_with_arg_vars(0);
        graph.set_return(handler, None);
        let mut exc = Link::new_mixed(
            Vec::new(),
            handler,
            Some(crate::model::error_carrier_exitcase()),
        );
        exc.last_exception = Some(LinkArg::Value(graph.alloc_value_var()));
        exc.last_exc_value = Some(LinkArg::Value(graph.alloc_value_var()));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![Link::new_mixed(Vec::new(), normal, None), exc],
        );
        let nblocks = graph.blocks.len();
        lower_error_carrier_edges(&mut graph, &carrier());

        let link = &graph.blocks[entry.0].exits[1];
        assert!(link.catches_all_exceptions());
        assert_eq!(link.target, handler);
        assert_eq!(graph.blocks.len(), nblocks);
    }

    /// No carrier named: the graph is left alone.
    #[test]
    fn a_pipeline_without_a_carrier_is_untouched() {
        let mut graph = FunctionGraph::new("no_carrier");
        let entry = graph.startblock;
        let e = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["make_error"]),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        crate::front::exc_from_raise::set_raise_from_instance(&mut graph, entry, e);
        let before = format!("{graph:?}");
        lower_error_carrier_edges(&mut graph, &crate::OwnedErrorCarrierSpec::default());
        assert_eq!(format!("{graph:?}"), before);
    }
}
