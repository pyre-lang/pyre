//! A Rust-heap `String` is not rstr `STR`. `type_name_of` returns `string::String`.
//! `getitem_slot`'s `format!("'{}' object is not subscriptable", …)` may
//! collapse via `str(value)` then `add` of rstr `STR` pieces; the
//! `type_name_of` result itself must not be an `add` operand.
mod common;

use common::{INTERPRETER_LLBC, load_llbc, lower_named_with_static_addrs};
use majit_translate::model::{CallTarget, LinkArg, OpKind, ValueType};
use majit_translate::{ErrorCarrierSpec, HostStaticAddrs};
use std::collections::HashSet;

fn addrs() -> HostStaticAddrs<'static> {
    HostStaticAddrs {
        error_carrier: ErrorCarrierSpec {
            carrier_path: "pyre_interpreter::error::PyError",
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
            from_exc_object: Some(("PyError", "from_exc_object")),
        },
        ..Default::default()
    }
}

fn lower(name: &str) -> majit_translate::model::FunctionGraph {
    let llbc = load_llbc(INTERPRETER_LLBC);
    lower_named_with_static_addrs(llbc, name, addrs())
        .unwrap_or_else(|e| panic!("lower {name}: {e:?}"))
}

#[test]
fn type_name_of_returns_instance_string_not_rstr() {
    let graph = lower("pyre_interpreter::error::type_name_of");
    assert!(
        !graph.return_is_str,
        "by-value String is not Ptr(STR): return_is_str={}",
        graph.return_is_str
    );
    assert_eq!(graph.return_class_root.as_deref(), Some("string::String"));
}

#[test]
fn getitem_slot_does_not_concat_rust_string_as_rstr() {
    let graph = lower("pyre_interpreter::baseobjspace::getitem_slot");
    let mut type_name_tys = Vec::new();
    let mut type_name_ids = Vec::new();
    let mut raw_string_adds = 0usize;
    let mut str_adds = 0usize;
    let mut format_calls = 0usize;
    let mut unary_str = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty,
                ..
            } = &op.kind
            {
                let leaf = segments.last().map(String::as_str).unwrap_or("");
                if leaf == "type_name_of" {
                    type_name_tys.push(result_ty.clone());
                    if let Some(r) = &op.result {
                        type_name_ids.push(r.id());
                    }
                }
            }
        }
    }
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::UnaryOp { op: uop, .. } if uop == "str" => {
                    unary_str += 1;
                }
                OpKind::BinOp {
                    op: bin,
                    lhs,
                    rhs,
                    result_ty,
                } if bin == "add" => {
                    if matches!(result_ty, ValueType::Ref(_) | ValueType::Str) {
                        str_adds += 1;
                    }
                    if type_name_ids
                        .iter()
                        .any(|id| lhs.id() == *id || rhs.id() == *id)
                    {
                        raw_string_adds += 1;
                    }
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => {
                    let leaf = segments.last().map(String::as_str).unwrap_or("");
                    if segments.iter().any(|s| s == "fmt") && leaf == "format" {
                        format_calls += 1;
                    }
                }
                _ => {}
            }
        }
    }
    assert!(
        type_name_tys.iter().all(|ty| !matches!(ty, ValueType::Str)),
        "type_name_of must not paint as Str: {type_name_tys:?}"
    );
    assert_eq!(
        raw_string_adds, 0,
        "ll_strconcat must not consume a type_name_of String header"
    );
    assert_eq!(
        format_calls, 0,
        "format! with a string-family interpolant must collapse; leftover format={format_calls}"
    );
    assert!(
        str_adds > 0,
        "collapse must emit add of rstr STR pieces: str_adds={str_adds}"
    );

    // Follow SSA through `__cast_instance_intrinsic` and block inputargs so
    // `str()` of the rust-heap String sits between `type_name_of` and `add`.
    // The front end emits `UnaryOp str` (`Repr.rtype_str`); jtransform
    // residualizes that op through `rust_string_str_helper` (not identity).
    let aliases = ssa_aliases_of(&graph, &type_name_ids);
    let mut conversion_results = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("jit_ll_str_from_rust_string") => {
                    panic!(
                        "front-end must emit UnaryOp str; jtransform names the host helper, \
                         got FunctionPath {segments:?}"
                    );
                }
                OpKind::UnaryOp {
                    op: uop, operand, ..
                } if uop == "str" && aliases.contains(&operand.id()) => {
                    if let Some(r) = &op.result {
                        assert_ne!(
                            r.id(),
                            operand.id(),
                            "str() of a rust-heap String must allocate a new STR, \
                             not alias the String header"
                        );
                        conversion_results.push(r.id());
                    }
                }
                _ => {}
            }
        }
    }
    let mut add_uses_converted = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::BinOp {
                op: bin, lhs, rhs, ..
            } = &op.kind
                && bin == "add"
                && conversion_results
                    .iter()
                    .any(|id| lhs.id() == *id || rhs.id() == *id)
            {
                add_uses_converted += 1;
            }
        }
    }
    assert!(
        !conversion_results.is_empty() && add_uses_converted > 0,
        "str() of type_name_of must sit between the call and add: \
         conversions={} add_uses_converted={add_uses_converted} unary_str={unary_str}",
        conversion_results.len()
    );
}

/// SSA aliases of `seeds`: `__cast_instance_intrinsic` results and block
/// inputargs forwarded from those ids.
fn ssa_aliases_of(graph: &majit_translate::model::FunctionGraph, seeds: &[u64]) -> HashSet<u64> {
    let mut ids: HashSet<u64> = seeds.iter().copied().collect();
    let mut changed = true;
    while changed {
        changed = false;
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                    && segments.iter().any(|s| s == "__cast_instance_intrinsic")
                    && let Some(r) = &op.result
                    && args.iter().any(|a| match a {
                        LinkArg::Value(v) => ids.contains(&v.id()),
                        _ => false,
                    })
                    && ids.insert(r.id())
                {
                    changed = true;
                }
            }
            for link in &block.exits {
                for (i, a) in link.args.iter().enumerate() {
                    if let LinkArg::Value(v) = a
                        && ids.contains(&v.id())
                        && let Some(target) = graph.blocks.iter().find(|b| b.id == link.target)
                        && let Some(inp) = target.inputargs.get(i)
                        && ids.insert(inp.id())
                    {
                        changed = true;
                    }
                }
            }
        }
    }
    ids
}
