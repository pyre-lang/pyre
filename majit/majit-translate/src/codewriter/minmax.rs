//! `core::cmp::{min,max}` on a graph the rtyper never lifted.
//!
//! A lifted graph meets `rtype_builtin_min` / `rtype_builtin_max`
//! (`rbuiltin.py`), which `gendirectcall`s `ll_min` / `ll_max`.  A graph
//! that stays on the rich-`OpKind` spine never meets the rtyper, so the
//! Opaque `core::cmp::min` / `core::cmp::impls::<Impl>::min` call stays a
//! residual with no host symbol — the wall between a gateway body and
//! its trace whenever the body does `std::cmp::min(a, b)`.
//!
//! This module gives that spine the same answer: [`minmax_path`] mints
//! the `ll_min` / `ll_max` graph in the rich model and `jtransform`
//! redirects the call to it.  The graph is `rbuiltin.py`'s helper:
//!
//! ```python
//! def ll_min(i1, i2):
//!     if i1 < i2:
//!         return i1
//!     return i2
//!
//! def ll_max(i1, i2):
//!     if i1 > i2:
//!         return i1
//!     return i2
//! ```

use crate::codewriter::call::CallControl;
use crate::model::{FunctionGraph, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;

/// `rbuiltin.py ll_min`.
pub const LL_MIN: &str = "ll_min";
/// `rbuiltin.py ll_max`.
pub const LL_MAX: &str = "ll_max";

/// `Some(is_max)` when the op is a two-arg `core::cmp::{min,max}`
/// (free function or `impls::<Impl>` method).  The adapter uses the
/// same `core` + `cmp` + leaf test (`flowspace_adapter`).
pub fn is_cmp_minmax(op: &SpaceOperation) -> Option<bool> {
    let OpKind::Call {
        target: crate::model::CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = &op.kind
    else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    if segments.len() < 3 || segments[0] != "core" || segments[1] != "cmp" {
        return None;
    }
    match segments.last().map(String::as_str) {
        Some("min") => Some(false),
        Some("max") => Some(true),
        _ => None,
    }
}

/// The comparison leaf a `core::cmp::{eq,ne,lt,le,gt,ge}` FunctionPath
/// (free function or `impls::<Impl>` method) lowers to.  The adapter's
/// `nonraising_core_bridge_opname` uses the same `core` + `cmp` + leaf
/// test; the rich spine emits the like-named `BinOp` instead of leaving
/// the Opaque call residual.
pub fn cmp_binop_leaf(segments: &[String]) -> Option<&'static str> {
    if segments.len() < 3 || segments[0] != "core" || segments[1] != "cmp" {
        return None;
    }
    match segments.last().map(String::as_str) {
        Some("eq") => Some("eq"),
        Some("ne") => Some("ne"),
        Some("lt") => Some("lt"),
        Some("le") => Some("le"),
        Some("gt") => Some("gt"),
        Some("ge") => Some("ge"),
        _ => None,
    }
}

/// Whether two scalar banks may share a `BinOp` of `leaf`.  `eq`/`ne`
/// treat Signed and Unsigned as one machine word (`getkind` is `'int'`
/// for both); ordered compares keep the banks apart so `uint_lt` is
/// not answered by `int_lt`.  Float never mixes.  A missing bank is a
/// const operand — it follows the other side.
pub fn scalar_cmp_banks_compatible(
    lhs: Option<&ValueType>,
    rhs: Option<&ValueType>,
    leaf: &str,
) -> bool {
    let ordered = matches!(leaf, "lt" | "le" | "gt" | "ge");
    match (lhs, rhs) {
        (Some(a), Some(b)) => banks_agree(a, b, ordered),
        (Some(a), None) | (None, Some(a)) => is_scalar_cmp_bank(a),
        (None, None) => false,
    }
}

fn is_scalar_cmp_bank(ty: &ValueType) -> bool {
    matches!(
        ty,
        ValueType::Int | ValueType::Unsigned | ValueType::Float | ValueType::Bool
    )
}

fn banks_agree(lhs: &ValueType, rhs: &ValueType, ordered: bool) -> bool {
    match (lhs, rhs) {
        (ValueType::Float, ValueType::Float) => true,
        (ValueType::Float, _) | (_, ValueType::Float) => false,
        (ValueType::Unsigned, ValueType::Unsigned) => true,
        (ValueType::Unsigned, _) | (_, ValueType::Unsigned) => !ordered,
        (ValueType::Int | ValueType::Bool, ValueType::Int | ValueType::Bool) => true,
        _ => false,
    }
}

/// Assembler `op_kind_to_opname` prefixes a bare `lt` with `int_`.
/// Unsigned ordered compares must already carry the `uint_` prefix
/// (`rtyper.rs` `lowlevel_min_max_helper_graph` picks `uint_lt`).
pub fn scalar_cmp_opname(leaf: &str, lhs: Option<&ValueType>, rhs: Option<&ValueType>) -> String {
    let unsigned = matches!(
        (lhs, rhs),
        (Some(ValueType::Unsigned), Some(ValueType::Unsigned))
            | (Some(ValueType::Unsigned), None)
            | (None, Some(ValueType::Unsigned))
    );
    if unsigned && matches!(leaf, "lt" | "le" | "gt" | "ge") {
        format!("uint_{leaf}")
    } else {
        leaf.to_string()
    }
}

/// The value bank `ll_min` / `ll_max` can compare.  Refs have no
/// ordering helper; mixed banks stay residual.
pub fn minmax_value_ty(result_ty: &ValueType) -> Option<ValueType> {
    match result_ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Float => Some(result_ty.clone()),
        // `BoolRepr` compares through `as_int = signed_repr` (`rbool.py`).
        ValueType::Bool => Some(ValueType::Int),
        _ => None,
    }
}

/// The helper path for `(is_max, value_ty)`, minting and registering
/// the graph on first use.
pub fn minmax_path(cc: &mut CallControl, is_max: bool, value_ty: &ValueType) -> CallPath {
    let leaf = if is_max { LL_MAX } else { LL_MIN };
    let name = helper_name(leaf, value_ty);
    let path = CallPath::from_segments([name.as_str()]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_minmax_graph(&name, is_max, value_ty);
        cc.register_function_graph(path.clone(), graph);
        cc.add_candidate_graph(path.clone());
    }
    path
}

fn helper_name(leaf: &str, value_ty: &ValueType) -> String {
    let bank = match value_ty {
        ValueType::Float => "float",
        ValueType::Unsigned => "uint",
        _ => "int",
    };
    format!("{leaf}__{bank}")
}

/// `rbuiltin.py ll_min` / `ll_max` in the rich model: compare, then
/// return the winning argument on the true / false link.
pub fn build_ll_minmax_graph(name: &str, is_max: bool, value_ty: &ValueType) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;

    let i1 = graph.alloc_value_var();
    let i2 = graph.alloc_value_var();
    for (var, param) in [(&i1, "arg0"), (&i2, "arg1")] {
        graph.push_inputarg_var(start_block, var.clone());
        graph.push_op_with_result_var(
            start_block,
            OpKind::Input {
                name: param.to_string(),
                ty: value_ty.clone(),
                class_root: None,
            },
            var.clone(),
        );
    }

    let cond = {
        let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        graph
            .block_mut(start_block)
            .operations
            .push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::BinOp {
                    op: match (is_max, value_ty) {
                        (true, ValueType::Unsigned) => "uint_gt".into(),
                        (false, ValueType::Unsigned) => "uint_lt".into(),
                        (true, _) => "gt".into(),
                        (false, _) => "lt".into(),
                    },
                    lhs: i1.clone(),
                    rhs: i2.clone(),
                    result_ty: ValueType::Bool,
                },
            });
        res
    };
    let (take_first, take_first_args) = graph.create_block_with_arg_vars(1);
    let (take_second, take_second_args) = graph.create_block_with_arg_vars(1);
    graph.set_branch(
        start_block,
        cond,
        take_first,
        vec![i1],
        take_second,
        vec![i2],
    );
    let [first] = take_first_args.as_slice() else {
        unreachable!("take_first was created with one inputarg")
    };
    let [second] = take_second_args.as_slice() else {
        unreachable!("take_second was created with one inputarg")
    };
    graph.set_return(take_first, Some(first.clone()));
    graph.set_return(take_second, Some(second.clone()));
    graph
}

/// `rint.py` non-ovf `int_abs` in the rich model: `if x < 0: return -x; return x`.
/// `checked_abs` uses this as the unread-on-None wrapping payload.
pub const LL_INT_ABS: &str = "ll_int_abs";

pub fn int_abs_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_INT_ABS]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_int_abs_graph(LL_INT_ABS);
        cc.register_function_graph(path.clone(), graph);
        cc.add_candidate_graph(path.clone());
    }
    path
}

pub fn build_ll_int_abs_graph(name: &str) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;
    let x = graph.alloc_value_var();
    graph.push_inputarg_var(start_block, x.clone());
    graph.push_op_with_result_var(
        start_block,
        OpKind::Input {
            name: "arg0".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        x.clone(),
    );
    let zero = {
        let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        graph
            .block_mut(start_block)
            .operations
            .push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::ConstInt(0),
            });
        res
    };
    let cond = {
        let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        graph
            .block_mut(start_block)
            .operations
            .push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::BinOp {
                    op: "lt".into(),
                    lhs: x.clone(),
                    rhs: zero,
                    result_ty: ValueType::Bool,
                },
            });
        res
    };
    let (take_neg, take_neg_args) = graph.create_block_with_arg_vars(1);
    let (take_id, take_id_args) = graph.create_block_with_arg_vars(1);
    graph.set_branch(
        start_block,
        cond,
        take_neg,
        vec![x.clone()],
        take_id,
        vec![x],
    );
    let [neg_x] = take_neg_args.as_slice() else {
        unreachable!("take_neg was created with one inputarg")
    };
    let negated = {
        let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        graph.block_mut(take_neg).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::UnaryOp {
                op: "neg".into(),
                operand: neg_x.clone(),
                result_ty: ValueType::Int,
            },
        });
        res
    };
    graph.set_return(take_neg, Some(negated));
    let [id_x] = take_id_args.as_slice() else {
        unreachable!("take_id was created with one inputarg")
    };
    graph.set_return(take_id, Some(id_x.clone()));
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::Variable;
    use crate::model::CallTarget;

    fn call(segments: &[&str], nargs: usize) -> SpaceOperation {
        SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: segments.iter().map(|s| (*s).to_string()).collect(),
                    fun_decl_id: None,
                },
                args: crate::model::call_args((0..nargs).map(|_| Variable::new())),
                result_ty: ValueType::Int,
            },
        }
    }

    #[test]
    fn recognises_cmp_binop_leaves() {
        assert_eq!(
            cmp_binop_leaf(&["core".into(), "cmp".into(), "eq".into()]),
            Some("eq")
        );
        assert_eq!(
            cmp_binop_leaf(&[
                "core".into(),
                "cmp".into(),
                "impls".into(),
                "<Impl>".into(),
                "lt".into()
            ]),
            Some("lt")
        );
        assert_eq!(
            cmp_binop_leaf(&["core".into(), "cmp".into(), "min".into()]),
            None
        );
        assert_eq!(cmp_binop_leaf(&["foo".into(), "eq".into()]), None);
    }

    #[test]
    fn scalar_cmp_banks_keep_ordered_unsigned_apart() {
        assert!(scalar_cmp_banks_compatible(
            Some(&ValueType::Int),
            Some(&ValueType::Int),
            "eq"
        ));
        assert!(scalar_cmp_banks_compatible(
            Some(&ValueType::Int),
            Some(&ValueType::Unsigned),
            "eq"
        ));
        assert!(!scalar_cmp_banks_compatible(
            Some(&ValueType::Int),
            Some(&ValueType::Unsigned),
            "lt"
        ));
        assert!(scalar_cmp_banks_compatible(
            Some(&ValueType::Unsigned),
            Some(&ValueType::Unsigned),
            "lt"
        ));
        assert!(!scalar_cmp_banks_compatible(
            Some(&ValueType::Float),
            Some(&ValueType::Int),
            "eq"
        ));
        assert!(scalar_cmp_banks_compatible(
            Some(&ValueType::Int),
            None,
            "lt"
        ));
        assert!(!scalar_cmp_banks_compatible(None, None, "eq"));
        assert_eq!(
            scalar_cmp_opname("lt", Some(&ValueType::Unsigned), Some(&ValueType::Unsigned)),
            "uint_lt"
        );
        assert_eq!(
            scalar_cmp_opname("eq", Some(&ValueType::Unsigned), Some(&ValueType::Unsigned)),
            "eq"
        );
        assert_eq!(
            scalar_cmp_opname("lt", Some(&ValueType::Int), Some(&ValueType::Int)),
            "lt"
        );
    }

    #[test]
    fn unsigned_minmax_helper_uses_uint_compare() {
        let graph = build_ll_minmax_graph("ll_min__uint", false, &ValueType::Unsigned);
        let start = graph.block(graph.startblock);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "uint_lt"))
        );
    }

    #[test]
    fn int_abs_helper_negates_when_negative() {
        let graph = build_ll_int_abs_graph("ll_int_abs");
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 1);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt"))
        );
        assert_eq!(start.exits.len(), 2);
        assert!(graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "neg"))
        }));
        let mut cc = CallControl::new();
        let first = int_abs_path(&mut cc);
        let again = int_abs_path(&mut cc);
        assert_eq!(first, again);
    }

    #[test]
    fn recognises_free_and_impls_minmax() {
        assert_eq!(
            is_cmp_minmax(&call(&["core", "cmp", "min"], 2)),
            Some(false)
        );
        assert_eq!(is_cmp_minmax(&call(&["core", "cmp", "max"], 2)), Some(true));
        assert_eq!(
            is_cmp_minmax(&call(&["core", "cmp", "impls", "<Impl>", "min"], 2)),
            Some(false)
        );
        assert_eq!(
            is_cmp_minmax(&call(&["core", "cmp", "impls", "min"], 2)),
            Some(false)
        );
        assert_eq!(is_cmp_minmax(&call(&["core", "cmp", "eq"], 2)), None);
        assert_eq!(is_cmp_minmax(&call(&["core", "cmp", "min"], 1)), None);
        assert_eq!(is_cmp_minmax(&call(&["std", "cmp", "min"], 2)), None);
    }

    #[test]
    fn helper_graph_branches_on_lt_and_returns_the_winner() {
        let graph = build_ll_minmax_graph("ll_min__int", false, &ValueType::Int);
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 2);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt"))
        );
        assert_eq!(start.exits.len(), 2);
        assert_eq!(graph.blocks.len(), 5); // start / return / except / take_first / take_second
    }

    #[test]
    fn max_helper_compares_with_gt() {
        let graph = build_ll_minmax_graph("ll_max__int", true, &ValueType::Int);
        let start = graph.block(graph.startblock);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "gt"))
        );
    }

    #[test]
    fn helper_is_minted_once_and_is_a_regular_callee() {
        use crate::codewriter::call::CallKind;
        let mut cc = CallControl::new();
        let first = minmax_path(&mut cc, false, &ValueType::Int);
        let again = minmax_path(&mut cc, false, &ValueType::Int);
        assert_eq!(first, again);
        let max = minmax_path(&mut cc, true, &ValueType::Int);
        assert_ne!(first, max);
        let uint = minmax_path(&mut cc, false, &ValueType::Unsigned);
        assert_ne!(first, uint);
        let call = SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![first.last_segment().unwrap().to_string()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![Variable::new(), Variable::new()]),
                result_ty: ValueType::Int,
            },
        };
        assert_eq!(cc.guess_call_kind(&call), CallKind::Regular);
    }
}
