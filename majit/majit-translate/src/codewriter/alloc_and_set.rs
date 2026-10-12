//! `[v; N]` on a graph the rtyper never lifted.
//!
//! The front spells a Rust repeat as `__array_repeat(fill, count)`
//! (`front/mir.rs`). A lifted graph meets `rtype_alloc_and_set`
//! (`rlist.py`), which calls `ll_alloc_and_set`. A graph on the
//! rich-`OpKind` spine never meets the rtyper, so the marker survived
//! as a call with no graph — the same wall `codewriter::getslice`
//! closes for `__getslice_*`. [`ll_fixed_alloc_and_set_path`] mints
//! the fixed-layout family here and `jtransform` redirects the call.
//!
//! `jtransform.py do_fixed_newlist` raises
//! `NotSupported("variable or non-zero default value")` for a non-zero
//! fill. That case is `rlist.py ll_alloc_and_set`: clamp the count
//! (`op_int_force_ge_zero`), then `we_are_jitted` selects
//! `_ll_alloc_and_set_jit` or the nojit arm. The jit arm calls
//! `_ll_alloc_and_clear` when `_ll_zero_or_null`
//! (`do_fixed_newlist_clear` → `new_array_clear`) and otherwise
//! `_ll_alloc_and_set_nonnull`. `rlib/jit.py look_inside_iff` wraps
//! that loop: `unroll_safe` on the original, a `dont_look_inside`
//! trampoline, and
//! `if not we_are_jitted() or (isconstant(count) and count < 137)`.
//! The tracer follows the loop; this module does not emit N
//! `setarrayitem`s.
//!
//! The clear arm is one `NewArrayClear` of the consumer's ARRAY id.
//! Naming that helper `_ll_fixed_alloc_and_clear` would hit the global
//! `newlist_clear` oopspec and drop the id. A non-zero `i64` fill is
//! `NewArray` of the same id plus the traced `setarrayitem` loop
//! (`fixed_list_clears` is false for `i64`).

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, ConcreteType, ExitSwitch, FunctionGraph, LinkArg, OpKind, ValueType,
};
use crate::parse::CallPath;

/// `rlist.py _ll_alloc_and_set_nonnull`: `jit.isconstant(count) and count < 137`.
pub(crate) const LOOK_INSIDE_COUNT_LIMIT: i64 = 137;

/// Item kind and ARRAY identity of one fixed `[item] * count`.
pub(crate) struct FixedArrayRepeat {
    pub(crate) item_ty: ValueType,
    pub(crate) array_type_id: String,
    pub(crate) clears: bool,
}

/// Leaf names of the rich family for one ARRAY identity.
pub(crate) struct AllocAndSetNames {
    pub(crate) outer: String,
    pub(crate) jit: String,
    pub(crate) nonnull: String,
    pub(crate) wrapper: String,
    pub(crate) trampoline: String,
}

pub(crate) fn alloc_and_set_names(array_type_id: &str) -> AllocAndSetNames {
    let id = sanitize_array_id(array_type_id);
    let nonnull = format!("_ll_fixed_alloc_and_set_nonnull__{id}");
    AllocAndSetNames {
        outer: format!("ll_fixed_alloc_and_set__{id}"),
        jit: format!("_ll_fixed_alloc_and_set_jit__{id}"),
        wrapper: format!("{nonnull}_look_inside_iff"),
        trampoline: format!("{nonnull}_trampoline"),
        nonnull,
    }
}

/// The `ll_fixed_alloc_and_set` path for `identity`, minting the family
/// on first use. `LIST` is folded into the name, so the call is
/// `(count, item)`.
pub(crate) fn ll_fixed_alloc_and_set_path(
    cc: &mut CallControl,
    identity: &FixedArrayRepeat,
) -> CallPath {
    let names = alloc_and_set_names(&identity.array_type_id);
    let path = CallPath::from_segments([names.outer.as_str()]);
    if cc.has_function_graph(&path) {
        return path;
    }
    let orig = build_nonnull(&names, identity);
    let trampoline = build_trampoline(&names, identity);
    let wrapper = build_wrapper(&names, identity);
    let jit = build_jit_arm(&names, identity);
    let outer = build_outer(&names, identity);
    register(cc, &names.nonnull, orig, &["unroll_safe"]);
    register(cc, &names.trampoline, trampoline, &["dont_look_inside"]);
    register(cc, &names.wrapper, wrapper, &[]);
    register(cc, &names.jit, jit, &[]);
    register(cc, &names.outer, outer, &[]);
    path
}

fn register(cc: &mut CallControl, name: &str, graph: FunctionGraph, hints: &[&str]) {
    let path = CallPath::from_segments([name]);
    if hints.is_empty() {
        cc.register_function_graph(path.clone(), graph);
    } else {
        let hints = hints.iter().map(|hint| (*hint).to_string()).collect();
        cc.register_function_graph_with_hints(path.clone(), graph, hints);
    }
    cc.add_candidate_graph(path);
}

fn sanitize_array_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn item_concrete(item_ty: &ValueType) -> ConcreteType {
    match item_ty {
        ValueType::Float | ValueType::SingleFloat => ConcreteType::Float,
        ValueType::Ref(_) | ValueType::Str => ConcreteType::GcRef,
        ValueType::Void => ConcreteType::Void,
        _ => ConcreteType::Signed,
    }
}

fn stamp_exitswitch_int(graph: &FunctionGraph, block: BlockId) {
    let Some(ExitSwitch::Value(cond)) = &graph.block(block).exitswitch else {
        return;
    };
    let cond = cond.clone();
    FunctionGraph::set_concretetype_of_inline(&cond, ConcreteType::Signed);
}

fn push_value(
    graph: &mut FunctionGraph,
    block: BlockId,
    kind: OpKind,
    concrete: ConcreteType,
) -> Variable {
    let var = graph.alloc_value_var_with_type(concrete);
    graph.push_op_with_result_var(block, kind, var.clone());
    var
}

fn push_call(
    graph: &mut FunctionGraph,
    block: BlockId,
    segments: &[&str],
    args: Vec<Variable>,
    result_ty: ValueType,
    concrete: ConcreteType,
) -> Variable {
    push_value(
        graph,
        block,
        OpKind::Call {
            target: CallTarget::function_path(segments.iter().copied()),
            args: crate::model::call_args(args),
            result_ty,
        },
        concrete,
    )
}

fn entry_params(
    graph: &mut FunctionGraph,
    block: BlockId,
    identity: &FixedArrayRepeat,
) -> (Variable, Variable) {
    let count = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let item = graph.alloc_value_var_with_type(item_concrete(&identity.item_ty));
    graph.push_inputarg_var(block, count.clone());
    graph.push_inputarg_var(block, item.clone());
    graph.push_op_with_result_var(
        block,
        OpKind::Input {
            name: "count".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        count.clone(),
    );
    graph.push_op_with_result_var(
        block,
        OpKind::Input {
            name: "item".to_string(),
            ty: identity.item_ty.clone(),
            class_root: None,
        },
        item.clone(),
    );
    (count, item)
}

fn phi_pair(graph: &mut FunctionGraph, identity: &FixedArrayRepeat) -> (BlockId, Vec<Variable>) {
    let (block, args) = graph.create_block_with_arg_vars(2);
    FunctionGraph::set_concretetype_of_inline(&args[0], ConcreteType::Signed);
    FunctionGraph::set_concretetype_of_inline(&args[1], item_concrete(&identity.item_ty));
    (block, args)
}

fn call_and_return(graph: &mut FunctionGraph, block: BlockId, args: Vec<Variable>, callee: &str) {
    let result = push_call(
        graph,
        block,
        &[callee],
        args,
        ValueType::Ref(None),
        ConcreteType::GcRef,
    );
    graph.set_return(block, Some(result));
}

/// `ll_alloc_and_set` (`rlist.py`): `count = int_force_ge_zero(count)`,
/// then `we_are_jitted` → jit arm, else the original loop. Tracing folds
/// `we_are_jitted` true, so the nojit arm is the direct call of the loop
/// (zero and nonzero both write `item`).
///
/// `rarithmetic.int_force_ge_zero` lowers to `llop.int_force_ge_zero`.
/// `pyjitpl.py` records that unary in the exec-generated loop as
/// `opimpl_int_force_ge_zero` (`int_force_ge_zero/i>i`).
fn build_outer(names: &AllocAndSetNames, identity: &FixedArrayRepeat) -> FunctionGraph {
    let mut graph = FunctionGraph::new(&names.outer);
    let start = graph.startblock;
    let (count, item) = entry_params(&mut graph, start, identity);
    let count = push_value(
        &mut graph,
        start,
        OpKind::UnaryOp {
            op: "int_force_ge_zero".into(),
            operand: count,
            result_ty: ValueType::Int,
        },
        ConcreteType::Signed,
    );
    let jitted = push_call(
        &mut graph,
        start,
        &["majit_rlib", "jit", "we_are_jitted"],
        vec![],
        ValueType::Bool,
        ConcreteType::Signed,
    );
    let (jit_block, jit_args) = phi_pair(&mut graph, identity);
    let (nojit_block, nojit_args) = phi_pair(&mut graph, identity);
    call_and_return(&mut graph, jit_block, jit_args, &names.jit);
    call_and_return(&mut graph, nojit_block, nojit_args, &names.nonnull);
    graph.set_branch(
        start,
        jitted,
        jit_block,
        vec![count.clone(), item.clone()],
        nojit_block,
        vec![count, item],
    );
    stamp_exitswitch_int(&graph, start);
    graph
}

/// `_ll_alloc_and_set_jit`: `_ll_zero_or_null` → one `new_array_clear`
/// of the consumer ARRAY, else the `look_inside_iff` wrapper.
fn build_jit_arm(names: &AllocAndSetNames, identity: &FixedArrayRepeat) -> FunctionGraph {
    let mut graph = FunctionGraph::new(&names.jit);
    let start = graph.startblock;
    let (count, item) = entry_params(&mut graph, start, identity);
    let is_zero = push_item_is_zero(&mut graph, start, &item, &identity.item_ty);
    let (clear_block, clear_args) = graph.create_block_with_arg_vars(1);
    FunctionGraph::set_concretetype_of_inline(&clear_args[0], ConcreteType::Signed);
    let cleared = push_value(
        &mut graph,
        clear_block,
        OpKind::NewArrayClear {
            length: clear_args[0].clone(),
            item_ty: identity.item_ty.clone(),
            array_type_id: Some(identity.array_type_id.clone()),
        },
        ConcreteType::GcRef,
    );
    graph.set_return(clear_block, Some(cleared));
    let (nonzero, nonzero_args) = phi_pair(&mut graph, identity);
    call_and_return(&mut graph, nonzero, nonzero_args, &names.wrapper);
    graph.set_branch(
        start,
        is_zero,
        clear_block,
        vec![count.clone()],
        nonzero,
        vec![count, item],
    );
    stamp_exitswitch_int(&graph, start);
    graph
}

/// `not check` (`rlist.py _ll_zero_or_null`). `-0.0` is zero:
/// `float_is_true(-0.0)` is false, and `int_is_zero` of that flag is true.
fn push_item_is_zero(
    graph: &mut FunctionGraph,
    block: BlockId,
    item: &Variable,
    item_ty: &ValueType,
) -> Variable {
    match item_ty {
        ValueType::Ref(_) | ValueType::Str => push_value(
            graph,
            block,
            OpKind::UnaryOp {
                op: "ptr_iszero".into(),
                operand: item.clone(),
                result_ty: ValueType::Bool,
            },
            ConcreteType::Signed,
        ),
        ValueType::Float | ValueType::SingleFloat => {
            let flag = push_value(
                graph,
                block,
                OpKind::UnaryOp {
                    op: "float_is_true".into(),
                    operand: item.clone(),
                    result_ty: ValueType::Bool,
                },
                ConcreteType::Signed,
            );
            push_value(
                graph,
                block,
                OpKind::UnaryOp {
                    op: "int_is_zero".into(),
                    operand: flag,
                    result_ty: ValueType::Bool,
                },
                ConcreteType::Signed,
            )
        }
        _ => push_value(
            graph,
            block,
            OpKind::UnaryOp {
                op: "int_is_zero".into(),
                operand: item.clone(),
                result_ty: ValueType::Bool,
            },
            ConcreteType::Signed,
        ),
    }
}

/// `rlib/jit.py look_inside_iff` wrapper. The predicate is the oopspec
/// `jit.isconstant` (`majit_rlib::jit::isconstant`) and `count < 137`,
/// not a codewriter unroll.
fn build_wrapper(names: &AllocAndSetNames, identity: &FixedArrayRepeat) -> FunctionGraph {
    let mut graph = FunctionGraph::new(&names.wrapper);
    let start = graph.startblock;
    let (count, item) = entry_params(&mut graph, start, identity);
    let jitted = push_call(
        &mut graph,
        start,
        &["majit_rlib", "jit", "we_are_jitted"],
        vec![],
        ValueType::Bool,
        ConcreteType::Signed,
    );
    let (orig_block, orig_args) = phi_pair(&mut graph, identity);
    let (tramp_block, tramp_args) = phi_pair(&mut graph, identity);
    let (pred_block, pred_args) = phi_pair(&mut graph, identity);
    let (lt_block, lt_args) = phi_pair(&mut graph, identity);

    let is_constant = push_call(
        &mut graph,
        pred_block,
        &["majit_rlib", "jit", "isconstant"],
        vec![pred_args[0].clone()],
        ValueType::Bool,
        ConcreteType::Signed,
    );
    graph.set_branch(
        pred_block,
        is_constant,
        lt_block,
        pred_args.clone(),
        tramp_block,
        pred_args,
    );
    stamp_exitswitch_int(&graph, pred_block);

    let limit = push_value(
        &mut graph,
        lt_block,
        OpKind::ConstInt(LOOK_INSIDE_COUNT_LIMIT),
        ConcreteType::Signed,
    );
    let below = push_value(
        &mut graph,
        lt_block,
        OpKind::BinOp {
            op: "lt".into(),
            lhs: lt_args[0].clone(),
            rhs: limit,
            result_ty: ValueType::Bool,
        },
        ConcreteType::Signed,
    );
    graph.set_branch(
        lt_block,
        below,
        orig_block,
        lt_args.clone(),
        tramp_block,
        lt_args,
    );
    stamp_exitswitch_int(&graph, lt_block);

    graph.set_branch(
        start,
        jitted,
        pred_block,
        vec![count.clone(), item.clone()],
        orig_block,
        vec![count, item],
    );
    stamp_exitswitch_int(&graph, start);

    call_and_return(&mut graph, orig_block, orig_args, &names.nonnull);
    call_and_return(&mut graph, tramp_block, tramp_args, &names.trampoline);
    graph
}

/// `look_inside_iff`'s `@dont_look_inside` trampoline: one call of the
/// original loop.
fn build_trampoline(names: &AllocAndSetNames, identity: &FixedArrayRepeat) -> FunctionGraph {
    let mut graph = FunctionGraph::new(&names.trampoline);
    let start = graph.startblock;
    let (count, item) = entry_params(&mut graph, start, identity);
    call_and_return(&mut graph, start, vec![count, item], &names.nonnull);
    graph
}

/// `_ll_alloc_and_set_nonnull`: `new_array` (or `new_array_clear` when
/// `ARRAY.OF` is a GC pointer or a struct) then `while i < count:
/// setarrayitem`. One store in the body; `unroll_safe` lets the tracer
/// record the iterations.
fn build_nonnull(names: &AllocAndSetNames, identity: &FixedArrayRepeat) -> FunctionGraph {
    let mut graph = FunctionGraph::new(&names.nonnull);
    let start = graph.startblock;
    let (count, item) = entry_params(&mut graph, start, identity);
    let length = count.clone();
    let alloc = if identity.clears {
        OpKind::NewArrayClear {
            length,
            item_ty: identity.item_ty.clone(),
            array_type_id: Some(identity.array_type_id.clone()),
        }
    } else {
        OpKind::NewArray {
            length,
            item_ty: identity.item_ty.clone(),
            array_type_id: Some(identity.array_type_id.clone()),
        }
    };
    let arr = push_value(&mut graph, start, alloc, ConcreteType::GcRef);
    let zero = push_value(&mut graph, start, OpKind::ConstInt(0), ConcreteType::Signed);

    let (header, header_args) = graph.create_block_with_arg_vars(4);
    stamp_loop_phis(&header_args, identity);
    let (body, body_args) = graph.create_block_with_arg_vars(4);
    stamp_loop_phis(&body_args, identity);
    let (exit, exit_args) = graph.create_block_with_arg_vars(1);
    FunctionGraph::set_concretetype_of_inline(&exit_args[0], ConcreteType::GcRef);

    graph.set_goto(start, header, vec![arr, count, item.clone(), zero]);

    let h_arr = header_args[0].clone();
    let h_count = header_args[1].clone();
    let h_item = header_args[2].clone();
    let h_i = header_args[3].clone();
    let cont = push_value(
        &mut graph,
        header,
        OpKind::BinOp {
            op: "lt".into(),
            lhs: h_i.clone(),
            rhs: h_count.clone(),
            result_ty: ValueType::Bool,
        },
        ConcreteType::Signed,
    );
    graph.set_branch(
        header,
        cont,
        body,
        vec![h_arr.clone(), h_count.clone(), h_item.clone(), h_i.clone()],
        exit,
        vec![h_arr],
    );
    stamp_exitswitch_int(&graph, header);
    graph.set_return(exit, Some(exit_args[0].clone()));

    let b_arr = body_args[0].clone();
    let b_count = body_args[1].clone();
    let b_item = body_args[2].clone();
    let b_i = body_args[3].clone();
    let _ = graph.push_op_var(
        body,
        OpKind::ArrayWrite {
            base: b_arr.clone(),
            index: b_i.clone(),
            value: LinkArg::Value(b_item.clone()),
            item_ty: identity.item_ty.clone(),
            array_type_id: Some(identity.array_type_id.clone()),
            nolength: false,
        },
        false,
    );
    let one = push_value(&mut graph, body, OpKind::ConstInt(1), ConcreteType::Signed);
    let next = push_value(
        &mut graph,
        body,
        OpKind::BinOp {
            op: "add".into(),
            lhs: b_i,
            rhs: one,
            result_ty: ValueType::Int,
        },
        ConcreteType::Signed,
    );
    graph.set_goto(body, header, vec![b_arr, b_count, b_item, next]);
    graph
}

fn stamp_loop_phis(args: &[Variable], identity: &FixedArrayRepeat) {
    FunctionGraph::set_concretetype_of_inline(&args[0], ConcreteType::GcRef);
    FunctionGraph::set_concretetype_of_inline(&args[1], ConcreteType::Signed);
    FunctionGraph::set_concretetype_of_inline(&args[2], item_concrete(&identity.item_ty));
    FunctionGraph::set_concretetype_of_inline(&args[3], ConcreteType::Signed);
}
