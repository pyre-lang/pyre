//! `w_set_items` must lift with every FieldRead base defined in its block
//! (the kwargsdict getitem family failed when a Signed FieldRead used an
//! undefined operand).

mod common;

use common::{OBJECT_LLBC, load_llbc_if_present, lower_named_with_static_addrs};
use majit_translate::model::{CallTarget, FunctionGraph, LinkArg, OpKind};
use majit_translate::{ErrorCarrierSpec, HostStaticAddrs};
use std::collections::HashSet;

const ERROR_CARRIER: ErrorCarrierSpec<'static> = ErrorCarrierSpec {
    carrier_path: "pyre_interpreter::error::PyError",
    carrier_class: "",
    carrier_wrappers: &[],
    to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
    from_exc_object: Some(("PyError", "from_exc_object")),
};

fn production_addrs() -> HostStaticAddrs<'static> {
    HostStaticAddrs {
        error_carrier: ERROR_CARRIER,
        ..Default::default()
    }
}

fn dump_graph(graph: &FunctionGraph) -> String {
    let mut listing = String::new();
    for block in &graph.blocks {
        listing.push_str(&format!(
            "\nblock {:?} dead={} inputargs: {:?}\n",
            block.id,
            block.dead,
            block.inputargs.iter().map(|v| v.id()).collect::<Vec<_>>()
        ));
        for (i, op) in block.operations.iter().enumerate() {
            listing.push_str(&format!(
                "  op[{i}] result={:?} {:?}\n",
                op.result.as_ref().map(|v| v.id()),
                op.kind
            ));
        }
        for (i, link) in block.exits.iter().enumerate() {
            listing.push_str(&format!(
                "  exit[{i}] -> {:?} args={:?}\n",
                link.target,
                link.args
                    .iter()
                    .map(|a| match a {
                        LinkArg::Value(v) => format!("v{}", v.id()),
                        LinkArg::Const(c) => format!("const({:?})", c.value),
                    })
                    .collect::<Vec<_>>()
            ));
        }
    }
    listing
}

fn op_vars(kind: &OpKind) -> Vec<&majit_translate::flowspace::model::Variable> {
    match kind {
        OpKind::FieldRead { base, .. } => vec![base],
        OpKind::FieldWrite { base, value, .. } => {
            let mut vars = vec![base];
            if let LinkArg::Value(v) = value {
                vars.push(v);
            }
            vars
        }
        _ => Vec::new(),
    }
}

#[test]
fn w_set_items_fieldread_bases_are_defined() {
    let Some(llbc) = load_llbc_if_present(OBJECT_LLBC) else {
        return;
    };
    let graph = match lower_named_with_static_addrs(llbc, "w_set_items", production_addrs()) {
        Ok(graph) => graph,
        Err(err) => panic!("w_set_items lower failed: {err}"),
    };
    let mut bad = Vec::new();
    for block in &graph.blocks {
        let mut defined: HashSet<u64> = block.inputargs.iter().map(|v| v.id()).collect();
        for (i, op) in block.operations.iter().enumerate() {
            if let OpKind::FieldRead {
                base, field, ty, ..
            } = &op.kind
                && !defined.contains(&base.id())
            {
                bad.push(format!(
                    "block {:?} op[{i}] FieldRead base v{} field={} ty={ty:?}",
                    block.id,
                    base.id(),
                    field.name
                ));
            }
            for var in op_vars(&op.kind) {
                if !defined.contains(&var.id()) && !matches!(&op.kind, OpKind::FieldRead { .. }) {
                    bad.push(format!(
                        "block {:?} op[{i}] uses undefined v{} in {:?}",
                        block.id,
                        var.id(),
                        op.kind
                    ));
                }
            }
            if let Some(r) = &op.result {
                defined.insert(r.id());
            }
        }
    }
    assert!(
        bad.is_empty(),
        "undefined FieldRead bases in w_set_items:\n{}\n{}",
        bad.join("\n"),
        dump_graph(&graph)
    );
}

#[test]
fn set_strategy_trampoline_keeps_its_vtable_call() {
    let Some(llbc) = load_llbc_if_present(OBJECT_LLBC) else {
        return;
    };
    let graph =
        match lower_named_with_static_addrs(llbc, "set_strategy_add_key_ptr", production_addrs()) {
            Ok(graph) => graph,
            Err(err) => panic!("set_strategy_add_key_ptr lower failed: {err}"),
        };
    eprintln!(
        "set_strategy_add_key_ptr name={:?} source_identity={:?}",
        graph.name, graph.source_identity
    );
    let mut self_calls = 0usize;
    let mut family_add = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("set_strategy_add_key_ptr") => {
                    self_calls += 1;
                }
                OpKind::IndirectCall { family_key, .. }
                    if family_key
                        .as_ref()
                        .map(|(trait_root, method)| (trait_root.as_str(), method.as_str()))
                        == Some(("SetStrategy", "add")) =>
                {
                    family_add += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        self_calls,
        0,
        "trampoline body must not Call its own FunctionPath\n{}",
        dump_graph(&graph)
    );
    assert!(
        family_add > 0,
        "trampoline body must keep IndirectCall family SetStrategy::add\n{}",
        dump_graph(&graph)
    );
}
