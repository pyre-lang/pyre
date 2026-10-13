//! Promoted `(a..=b).contains(&x)` — `W_CType::is_primitive` un-promotes
//! rustc's `AnonConst` `RangeInclusive` so `front::range_contains` can
//! fold it to `int_between`.
//!
//! Shares the production `pyre-module.ullbc` with the other module-LLBC
//! integration tests; load is once per process.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, OpKind};
use std::sync::OnceLock;

const MODULE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-module.ullbc",
);

fn module_llbc() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(MODULE).expect("load pyre-module.ullbc"))
}

fn functionpath_calls_ending(
    graph: &majit_translate::model::FunctionGraph,
    tail: &[&str],
) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if path_ends_with(segments, tail)
            )
        })
        .count()
}

fn path_ends_with(segments: &[String], tail: &[&str]) -> bool {
    segments.len() >= tail.len()
        && segments[segments.len() - tail.len()..]
            .iter()
            .zip(tail)
            .all(|(s, t)| s.as_str() == *t)
}

/// `ctypeobj::is_primitive` is `(KIND_PRIM_CHAR..=KIND_PRIM_COMPLEX).contains(&self.kind)`.
/// rustc promotes the range to an `AnonConst`; without the un-promote the
/// Global read residualizes `…::is_primitive::promoted` as a 0-arg Call.
#[test]
fn is_primitive_promoted_range_folds_to_int_between() {
    let graph = lower_function(
        module_llbc(),
        "pyre_module::module::_cffi_backend::ctypeobj::<Impl>::is_primitive",
    )
    .expect("lower is_primitive");

    let residual_promoted = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if args.is_empty()
                && segments.iter().any(|s| s == "is_primitive")
                && segments.last().is_some_and(|s| {
                    s == "promoted" || s.starts_with("promoted#") || s == "<Builtin>"
                }) =>
            {
                true
            }
            _ => false,
        })
        .count();
    assert_eq!(
        residual_promoted, 0,
        "promoted RangeInclusive must not residualize as a 0-arg Call; graph={graph:?}"
    );

    assert_eq!(
        functionpath_calls_ending(&graph, &["range", "RangeInclusive", "new"]),
        0,
        "residual RangeInclusive::new removed"
    );
    assert_eq!(
        functionpath_calls_ending(&graph, &["range", "RangeInclusive", "contains"]),
        0,
        "residual RangeInclusive::contains removed"
    );

    let int_between: Vec<&[majit_translate::flowspace::model::Variable]> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::LoweredBlackholeOp { opname, args } if opname == "int_between" => {
                Some(args.as_slice())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        int_between.len(),
        1,
        "is_primitive must emit one int_between(KIND_PRIM_CHAR, kind, KIND_PRIM_COMPLEX + 1)"
    );

    let const_int_of = |var: &majit_translate::flowspace::model::Variable| -> Option<i64> {
        // The ConstInt may live in a predecessor; `SSA_to_SSI` binds a
        // fresh `Variable.copy()` through `Link.args` at each hop.
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![var.clone()];
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur.id()) {
                continue;
            }
            for block in &graph.blocks {
                for op in &block.operations {
                    if let (Some(result), OpKind::ConstInt(value)) = (&op.result, &op.kind)
                        && result == &cur
                    {
                        return Some(*value);
                    }
                }
                for (i, input) in block.inputargs.iter().enumerate() {
                    if input != &cur {
                        continue;
                    }
                    for pred in &graph.blocks {
                        for link in &pred.exits {
                            if link.target != block.id {
                                continue;
                            }
                            if let Some(incoming) = link.args.get(i).and_then(|a| a.as_variable()) {
                                stack.push(incoming.clone());
                            }
                        }
                    }
                }
            }
        }
        None
    };
    let args = int_between[0];
    assert_eq!(args.len(), 3);
    assert_eq!(
        const_int_of(&args[0]),
        Some(1),
        "lower bound is KIND_PRIM_CHAR"
    );
    assert_eq!(
        const_int_of(&args[2]),
        Some(9),
        "exclusive upper is KIND_PRIM_COMPLEX + 1"
    );
}
