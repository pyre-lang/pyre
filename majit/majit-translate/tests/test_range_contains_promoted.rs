//! Promoted `(a..=b).contains(&x)` — `W_CType::is_primitive` un-promotes
//! rustc's `AnonConst` `RangeInclusive` so `front::range_contains` can
//! fold it to `int_between`.
//!
//! Shares the production `pyre-module.ullbc` / `pyre-object.ullbc` with
//! the other LLBC integration tests; each load is once per process.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, OpKind};
use std::sync::OnceLock;

const MODULE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-module.ullbc",
);
const OBJECT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc",
);

fn module_llbc() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(MODULE).expect("load pyre-module.ullbc"))
}

fn object_llbc() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(OBJECT).expect("load pyre-object.ullbc"))
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

    let args = int_between[0];
    assert_eq!(args.len(), 3);
    assert_eq!(
        const_int_of(&graph, &args[0]),
        Some(1),
        "lower bound is KIND_PRIM_CHAR"
    );
    assert_eq!(
        const_int_of(&graph, &args[2]),
        Some(9),
        "exclusive upper is KIND_PRIM_COMPLEX + 1"
    );
}

/// The signed constant `var` is produced by. Follows `Variable.copy()`
/// inputargs (`ssa.py` `SSA_to_SSI`) only when every incoming path
/// agrees (`DataFlowFamilyBuilder` unifies a single family).
fn const_int_of(
    graph: &majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<i64> {
    const_int_of_inner(graph, var, &mut std::collections::HashSet::new())
}

fn const_int_of_inner(
    graph: &majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
    seen: &mut std::collections::HashSet<u64>,
) -> Option<i64> {
    use majit_translate::model::{LinkArg, OpKind};
    if !seen.insert(var.id()) {
        return None;
    }
    for block in &graph.blocks {
        for op in &block.operations {
            if let (Some(result), OpKind::ConstInt(value)) = (&op.result, &op.kind)
                && result == var
            {
                return Some(*value);
            }
        }
    }
    let mut agreed: Option<i64> = None;
    let mut saw_incoming = false;
    for block in graph.iterblocks() {
        if block.dead {
            continue;
        }
        let Some(slot) = block.inputargs.iter().position(|arg| arg == var) else {
            continue;
        };
        for pred in graph.iterblocks() {
            if pred.dead {
                continue;
            }
            for link in &pred.exits {
                if link.target != block.id {
                    continue;
                }
                let value = match link.args.get(slot) {
                    Some(LinkArg::Value(incoming)) => const_int_of_inner(graph, incoming, seen)?,
                    Some(LinkArg::Const(constant)) => match constant.value {
                        majit_translate::flowspace::model::ConstValue::Int(value) => value,
                        _ => return None,
                    },
                    None => return None,
                };
                if let Some(prev) = agreed {
                    if prev != value {
                        return None;
                    }
                } else {
                    agreed = Some(value);
                }
                saw_incoming = true;
            }
        }
    }
    if saw_incoming { agreed } else { None }
}

/// A phi merging distinct constants is not itself a constant
/// (`DataFlowFamilyBuilder.complete`).
#[test]
fn const_int_of_rejects_phi_of_distinct_constants() {
    use majit_translate::model::{FunctionGraph, OpKind, ValueType};
    let mut graph = FunctionGraph::new("const_int_mixed_phi");
    let entry = graph.startblock;
    let left = graph.create_block();
    let right = graph.create_block();
    let (join, join_args) = graph.create_block_with_arg_vars(1);
    let one = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
    let nine = graph.push_op_var(entry, OpKind::ConstInt(9), true).unwrap();
    let cond = graph
        .push_op_var(
            entry,
            OpKind::Input {
                name: "cond".into(),
                ty: ValueType::Bool,
                class_root: None,
            },
            true,
        )
        .unwrap();
    graph.set_branch(entry, cond, left, vec![], right, vec![]);
    graph.set_goto(left, join, vec![one]);
    graph.set_goto(right, join, vec![nine]);
    assert_eq!(const_int_of(&graph, &join_args[0]), None);
}

/// `w_small_int_const` uses live exclusive
/// `(SMALL_INT_CONST_FROM..SMALL_INT_CONST_TO).contains(&value)`.
/// The transparent `Range` ADT routes `contains` as `CallTarget::Method`.
#[test]
fn w_small_int_const_exclusive_range_folds() {
    let graph = lower_function(object_llbc(), "pyre_object::intobject::w_small_int_const")
        .expect("lower w_small_int_const");

    let residual_contains = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                args,
                ..
            } if args.len() == 2 && name == "contains" => true,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if args.len() == 2
                && segments
                    .last()
                    .is_some_and(|leaf| leaf == "contains" || leaf.starts_with("contains")) =>
            {
                true
            }
            _ => false,
        })
        .count();
    let int_between = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::LoweredBlackholeOp { opname, .. } if opname == "int_between"
            )
        })
        .count();
    assert_eq!(
        residual_contains, 0,
        "Range::contains must be lowered; graph has leftover contains"
    );
    assert!(
        int_between >= 1,
        "exclusive int range emits int_between; got {int_between}"
    );
}
