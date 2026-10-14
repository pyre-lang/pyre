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
//! `_ll_alloc_and_set_jit` or `_ll_alloc_and_set_nojit`. The jit arm
//! calls `_ll_alloc_and_clear` when `_ll_zero_or_null`
//! (`do_fixed_newlist_clear` → `new_array_clear`) and otherwise
//! `_ll_alloc_and_set_nonnull`. The nojit arm is the same
//! `_ll_zero_or_null` test: zero fill uses cleared storage
//! (`rgc.ll_arrayclear` / `_ll_alloc_and_clear`), otherwise
//! `rgc.ll_arrayfill`. A Float fill compares bits (only `+0.0`
//! clears): these graphs lower Rust `vec![x; n]` / `[x; N]`, which
//! store `-0.0`. `rlib/jit.py look_inside_iff` wraps the loop:
//! `unroll_safe` on the original, a `dont_look_inside` trampoline, and
//! `if not we_are_jitted() or (isconstant(count) and count < 137)`.
//! The tracer follows the loop; this module does not emit N
//! `setarrayitem`s. The trampoline is residual: `look_inside_graph`
//! returns False for `_jit_look_inside_ = False`, so it is not a
//! candidate.
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
    pub(crate) nojit: String,
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
        nojit: format!("_ll_fixed_alloc_and_set_nojit__{id}"),
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
    let nojit = build_nojit(&names, identity);
    let outer = build_outer(&names, identity);
    // `look_inside_iff` (`rlib/jit.py`): `unroll_safe(func)` then
    // `func._annspecialcase_ = "specialize:call_location"` on the
    // original and the trampoline. The wrapper `f` does not get it.
    // Bind the kind-leaf residual address before residualizing the
    // trampoline: `getfunctionptr` of an ordinary translated function.
    bind_kind_trampoline_fnaddr(cc, &names.trampoline, &identity.item_ty);
    register(
        cc,
        &names.nonnull,
        orig,
        &["unroll_safe", "specialize:call_location"],
    );
    register(
        cc,
        &names.trampoline,
        trampoline,
        &["dont_look_inside", "specialize:call_location"],
    );
    register(cc, &names.wrapper, wrapper, &[]);
    register(cc, &names.jit, jit, &[]);
    register(cc, &names.nojit, nojit, &[]);
    register(cc, &names.outer, outer, &[]);
    path
}

fn register(cc: &mut CallControl, name: &str, mut graph: FunctionGraph, hints: &[&str]) {
    let path = CallPath::from_segments([name]);
    for hint in hints {
        graph.push_hint(*hint);
    }
    // BFS admits a callee only when `policy.look_inside_graph` is true
    // (`call.py find_all_graphs`). A helper minted after that walk skips
    // BFS; run the same gate on the flags and `look_inside_function`
    // that walk recorded so a `_jit_look_inside_ = False` trampoline
    // stays residual.
    cc.register_function_graph(path.clone(), graph);
    cc.add_candidate_graph_if_look_inside(path);
}

/// Kind leaf published by `HELPER_FNADDRS` for the residual trampoline.
/// Identity-specific graphs alias this address (`look_inside_iff`
/// trampoline is an ordinary translated function).
pub(crate) fn kind_trampoline_leaf(item_ty: &ValueType) -> &'static str {
    match item_ty {
        ValueType::Float | ValueType::SingleFloat => "_ll_fixed_alloc_and_set_nonnull_f_trampoline",
        ValueType::Ref(_) | ValueType::Str => "_ll_fixed_alloc_and_set_nonnull_r_trampoline",
        _ => "_ll_fixed_alloc_and_set_nonnull_i_trampoline",
    }
}

/// Crate-stripped `CallPath` `register_macro_helper_trace_fnaddr` binds
/// first for this kind's `HelperFnAddr.path`.
fn kind_trampoline_fnaddr_key(item_ty: &ValueType) -> CallPath {
    let leaf = kind_trampoline_leaf(item_ty);
    match item_ty {
        ValueType::Ref(_) | ValueType::Str => CallPath::from_segments(["object_array", leaf]),
        _ => CallPath::from_segments(["lltypesystem", "rlist", leaf]),
    }
}

/// `HelperFnAddr.path` (`concat!(module_path!(), "::", name)`) consumed
/// by `register_macro_helper_trace_fnaddr`.
#[cfg(test)]
fn kind_trampoline_helper_fnaddr_path(item_ty: &ValueType) -> String {
    let crate_name = match item_ty {
        ValueType::Ref(_) | ValueType::Str => "pyre_object",
        _ => "majit_rlib",
    };
    format!("{crate_name}::{}", kind_trampoline_fnaddr_key(item_ty))
}

fn bind_kind_trampoline_fnaddr(cc: &mut CallControl, trampoline: &str, item_ty: &ValueType) {
    let dest = CallPath::from_segments([trampoline]);
    if cc.lookup_function_fnaddr(&dest).is_some() {
        return;
    }
    let src = kind_trampoline_fnaddr_key(item_ty);
    if cc.lookup_function_fnaddr(&src).is_some() {
        cc.alias_function_fnaddr(dest, &src);
    }
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

fn push_void_call(
    graph: &mut FunctionGraph,
    block: BlockId,
    segments: &[&str],
    args: Vec<Variable>,
) {
    let _ = graph.push_op_var(
        block,
        OpKind::Call {
            target: CallTarget::function_path(segments.iter().copied()),
            args: crate::model::call_args(args),
            result_ty: ValueType::Void,
        },
        false,
    );
}

/// `LIST.ll_newlist(count)` for a fixed ARRAY: `new_array_clear` when
/// `ARRAY.OF` is a GC pointer, else uninitialised `new_array`.
fn push_newlist(
    graph: &mut FunctionGraph,
    block: BlockId,
    identity: &FixedArrayRepeat,
    length: Variable,
) -> Variable {
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
    push_value(graph, block, alloc, ConcreteType::GcRef)
}

/// `ll_alloc_and_set` (`rlist.py`): `count = int_force_ge_zero(count)`,
/// then `we_are_jitted` → jit arm, else `_ll_alloc_and_set_nojit`.
/// Tracing folds `we_are_jitted` true.
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
    call_and_return(&mut graph, nojit_block, nojit_args, &names.nojit);
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

/// Zero/null test for the fill. Non-float types follow `_ll_zero_or_null`
/// (`rlist.py`): `not item` for a Number/Primitive, `ptr_iszero` for a
/// Ptr. `_ll_zero_or_null` on a Float Primitive is also `not item`, so
/// `-0.0` would clear to `+0.0`; these graphs lower Rust `vec![x; n]` /
/// `[x; N]`, which store the fill's bit pattern, so the Float arm
/// compares bits and only `+0.0` takes the cleared arm.
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
            let bits = push_value(
                graph,
                block,
                OpKind::UnaryOp {
                    op: "convert_float_bytes_to_longlong".into(),
                    operand: item.clone(),
                    result_ty: ValueType::Int,
                },
                ConcreteType::Signed,
            );
            push_value(
                graph,
                block,
                OpKind::UnaryOp {
                    op: "int_is_zero".into(),
                    operand: bits,
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

/// `_ll_alloc_and_set_nojit` (`rlist.py`). The rtyper builder
/// (`build_ll_alloc_and_set_nojit_helper_graph`) returns a flowspace
/// `PyGraph`; this spine mirrors the same body in rich `OpKind`:
///
/// ```python
/// l = LIST.ll_newlist(count)
/// if _ll_zero_or_null(item):
///     if not malloc_zero_filled:
///         if not isinstance(typeOf(item), Ptr):
///             rgc.ll_arrayclear(l.ll_items())
///     return l
/// rgc.ll_arrayfill(l.ll_items(), item)
/// return l
/// ```
///
/// `malloc_zero_filled` is False in a translated program. A Ptr item
/// was already zeroed by `ll_newlist`'s `zero_gc_pointers_inside`
/// (`identity.clears`), so the zero arm skips `ll_arrayclear`. The
/// Float fill test is the bit compare in [`push_item_is_zero`]: `-0.0`
/// takes `ll_arrayfill`.
fn build_nojit(names: &AllocAndSetNames, identity: &FixedArrayRepeat) -> FunctionGraph {
    let mut graph = FunctionGraph::new(&names.nojit);
    let start = graph.startblock;
    let (count, item) = entry_params(&mut graph, start, identity);
    let arr = push_newlist(&mut graph, start, identity, count);
    let is_zero = push_item_is_zero(&mut graph, start, &item, &identity.item_ty);

    let (zero_block, zero_args) = graph.create_block_with_arg_vars(2);
    FunctionGraph::set_concretetype_of_inline(&zero_args[0], ConcreteType::GcRef);
    FunctionGraph::set_concretetype_of_inline(&zero_args[1], item_concrete(&identity.item_ty));
    // `not isinstance(typeOf(item), Ptr)`: only a non-Ptr item still
    // needs `rgc.ll_arrayclear`. The fixed receiver is the array ptr.
    if !identity.clears {
        push_void_call(
            &mut graph,
            zero_block,
            &["ll_arrayclear"],
            vec![zero_args[0].clone()],
        );
    }
    graph.set_return(zero_block, Some(zero_args[0].clone()));

    let (fill_block, fill_args) = graph.create_block_with_arg_vars(2);
    FunctionGraph::set_concretetype_of_inline(&fill_args[0], ConcreteType::GcRef);
    FunctionGraph::set_concretetype_of_inline(&fill_args[1], item_concrete(&identity.item_ty));
    push_void_call(
        &mut graph,
        fill_block,
        &["ll_arrayfill"],
        vec![fill_args[0].clone(), fill_args[1].clone()],
    );
    graph.set_return(fill_block, Some(fill_args[0].clone()));

    graph.set_branch(
        start,
        is_zero,
        zero_block,
        vec![arr.clone(), item.clone()],
        fill_block,
        vec![arr, item],
    );
    stamp_exitswitch_int(&graph, start);
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
    let arr = push_newlist(&mut graph, start, identity, count.clone());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::CallKind;
    use crate::model::SpaceOperation;

    fn i64_identity() -> FixedArrayRepeat {
        FixedArrayRepeat {
            item_ty: ValueType::Int,
            array_type_id: "[i64]".into(),
            clears: false,
        }
    }

    fn float_identity() -> FixedArrayRepeat {
        FixedArrayRepeat {
            item_ty: ValueType::Float,
            array_type_id: "[f64]".into(),
            clears: false,
        }
    }

    fn ptr_identity() -> FixedArrayRepeat {
        FixedArrayRepeat {
            item_ty: ValueType::Ref(None),
            array_type_id: crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID.into(),
            clears: true,
        }
    }

    fn mint(identity: &FixedArrayRepeat) -> (CallControl, AllocAndSetNames) {
        mint_with_kind_fnaddr(identity, Some(0x1000))
    }

    fn mint_with_kind_fnaddr(
        identity: &FixedArrayRepeat,
        addr: Option<i64>,
    ) -> (CallControl, AllocAndSetNames) {
        let mut cc = CallControl::new();
        if let Some(addr) = addr {
            cc.register_macro_helper_trace_fnaddr(
                &kind_trampoline_helper_fnaddr_path(&identity.item_ty),
                addr,
            );
        }
        let _ = ll_fixed_alloc_and_set_path(&mut cc, identity);
        let names = alloc_and_set_names(&identity.array_type_id);
        (cc, names)
    }

    fn call_op(name: &str) -> SpaceOperation {
        SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::function_path([name]),
                args: crate::model::call_args(vec![Variable::new(), Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        }
    }

    fn graph_ops(cc: &CallControl, name: &str) -> Vec<SpaceOperation> {
        let path = CallPath::from_segments([name]);
        cc.function_graphs()
            .get(&path)
            .unwrap_or_else(|| panic!("missing helper {name}"))
            .blocks
            .iter()
            .flat_map(|block| block.operations.clone())
            .collect()
    }

    fn calls_named(ops: &[SpaceOperation], name: &str) -> bool {
        ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.len() == 1 && segments[0] == name
            )
        })
    }

    fn graph_hints(cc: &CallControl, name: &str) -> Vec<String> {
        let path = CallPath::from_segments([name]);
        cc.function_graphs()
            .get(&path)
            .unwrap_or_else(|| panic!("missing helper {name}"))
            .hints
            .clone()
    }

    /// `dont_look_inside` is excluded by `look_inside_graph`, not by
    /// leaf name: the trampoline is registered, `graphs_from` is None,
    /// `guess_call_kind` is Residual, and the kind-leaf address is bound.
    #[test]
    fn trampoline_is_residual_not_a_candidate() {
        let (cc, names) = mint(&i64_identity());
        let tramp_path = CallPath::from_segments([names.trampoline.as_str()]);
        assert!(cc.has_function_graph(&tramp_path));
        assert!(
            !cc.is_candidate(&tramp_path),
            "look_inside_graph refuses `_jit_look_inside_ = False`"
        );
        let tramp_call = call_op(&names.trampoline);
        assert!(cc.graphs_from(&tramp_call).is_none());
        assert_eq!(cc.guess_call_kind(&tramp_call), CallKind::Residual);
        assert_eq!(cc.lookup_function_fnaddr(&tramp_path), Some(0x1000));
        let OpKind::Call { target, .. } = &tramp_call.kind else {
            panic!("call_op is a Call");
        };
        let binding = cc.fnaddr_binding_for_target(target);
        assert!(!binding.symbolic, "residual trampoline must be executable");
        assert!(!crate::codewriter::call::is_symbolic_fnaddr(binding.addr));

        for name in [
            &names.outer,
            &names.jit,
            &names.nojit,
            &names.nonnull,
            &names.wrapper,
        ] {
            let path = CallPath::from_segments([name.as_str()]);
            assert!(cc.is_candidate(&path), "{name} must stay a candidate");
            assert_eq!(cc.guess_call_kind(&call_op(name)), CallKind::Regular);
        }
    }

    /// `register_macro_helper_trace_fnaddr` consumes a `HelperFnAddr.path`
    /// and `bind_kind_trampoline_fnaddr` aliases that canonical key onto
    /// the identity trampoline.
    #[test]
    fn consumer_binds_each_kind_trampoline_from_helper_fnaddr_path() {
        for identity in [i64_identity(), float_identity(), ptr_identity()] {
            let mut cc = CallControl::new();
            cc.register_macro_helper_trace_fnaddr(
                &kind_trampoline_helper_fnaddr_path(&identity.item_ty),
                0x1000,
            );
            assert_eq!(
                cc.lookup_function_fnaddr(&kind_trampoline_fnaddr_key(&identity.item_ty)),
                Some(0x1000),
            );
            let _ = ll_fixed_alloc_and_set_path(&mut cc, &identity);
            let names = alloc_and_set_names(&identity.array_type_id);
            let tramp_path = CallPath::from_segments([names.trampoline.as_str()]);
            assert!(
                !cc.is_candidate(&tramp_path),
                "look_inside_graph refuses `_jit_look_inside_ = False`"
            );
            assert_eq!(cc.lookup_function_fnaddr(&tramp_path), Some(0x1000));
            let tramp_call = call_op(&names.trampoline);
            let OpKind::Call { target, .. } = &tramp_call.kind else {
                panic!("call_op is a Call");
            };
            let binding = cc.fnaddr_binding_for_target(target);
            assert!(!binding.symbolic, "{}", names.trampoline);
            assert!(!crate::codewriter::call::is_symbolic_fnaddr(binding.addr));
        }
    }

    fn family_names(names: &AllocAndSetNames) -> [&str; 5] {
        [
            names.outer.as_str(),
            names.jit.as_str(),
            names.nojit.as_str(),
            names.nonnull.as_str(),
            names.wrapper.as_str(),
        ]
    }

    /// Late mint uses the BFS policy's flags, not a freshly constructed
    /// policy with forced float/longlong/singlefloat support. Default
    /// `JitPolicy` has `supports_floats = False`, so a Float family is
    /// residual until BFS records the flags.
    #[test]
    fn float_family_is_residual_when_bfs_policy_has_no_floats() {
        let (cc, names) = mint(&float_identity());
        let tramp_path = CallPath::from_segments([names.trampoline.as_str()]);
        assert!(!cc.is_candidate(&tramp_path));
        for name in family_names(&names) {
            let path = CallPath::from_segments([name]);
            assert!(
                !cc.is_candidate(&path),
                "{name} must follow BFS supports_floats, not a forced-true policy"
            );
        }
    }

    /// After `find_all_graphs` records `supports_floats`, a later Float
    /// family is a candidate except the `dont_look_inside` trampoline.
    #[test]
    fn float_family_follows_bfs_policy_float_support() {
        let mut cc = CallControl::new();
        let portal = CallPath::from_segments(["fixture", "portal"]);
        cc.register_function_graph(portal.clone(), FunctionGraph::new("portal"));
        cc.mark_portal(portal);
        let mut policy = crate::policy::DefaultJitPolicy::new();
        policy.state.set_supports_floats(true);
        cc.find_all_graphs(&mut policy);
        let identity = float_identity();
        cc.register_macro_helper_trace_fnaddr(
            &kind_trampoline_helper_fnaddr_path(&identity.item_ty),
            0x1000,
        );
        let _ = ll_fixed_alloc_and_set_path(&mut cc, &identity);
        let names = alloc_and_set_names(&identity.array_type_id);
        let tramp_path = CallPath::from_segments([names.trampoline.as_str()]);
        assert!(cc.has_function_graph(&tramp_path));
        assert!(
            !cc.is_candidate(&tramp_path),
            "look_inside_graph refuses `_jit_look_inside_ = False`"
        );
        assert_eq!(cc.lookup_function_fnaddr(&tramp_path), Some(0x1000));
        for name in family_names(&names) {
            let path = CallPath::from_segments([name]);
            assert!(cc.is_candidate(&path), "{name} must stay a candidate");
            assert_eq!(cc.guess_call_kind(&call_op(name)), CallKind::Regular);
        }
    }

    /// `StopAtXPolicy.look_inside_function` is the BFS policy, so a late
    /// mint of a named helper stays opaque (`policy.py look_inside_graph`).
    #[test]
    fn late_family_respects_stop_at_x_policy() {
        let mut cc = CallControl::new();
        let portal = CallPath::from_segments(["fixture", "portal"]);
        cc.register_function_graph(portal.clone(), FunctionGraph::new("portal"));
        cc.mark_portal(portal);
        let identity = i64_identity();
        let names = alloc_and_set_names(&identity.array_type_id);
        let mut policy = crate::policy::StopAtXPolicy::new(vec![names.outer.clone()]);
        cc.find_all_graphs(&mut policy);
        cc.register_macro_helper_trace_fnaddr(
            &kind_trampoline_helper_fnaddr_path(&identity.item_ty),
            0x1000,
        );
        let _ = ll_fixed_alloc_and_set_path(&mut cc, &identity);
        let outer = CallPath::from_segments([names.outer.as_str()]);
        assert!(
            !cc.is_candidate(&outer),
            "StopAtXPolicy.look_inside_function must refuse the late outer"
        );
        for name in [
            names.jit.as_str(),
            names.nojit.as_str(),
            names.nonnull.as_str(),
            names.wrapper.as_str(),
        ] {
            let path = CallPath::from_segments([name]);
            assert!(cc.is_candidate(&path), "{name} must stay a candidate");
        }
        let tramp = CallPath::from_segments([names.trampoline.as_str()]);
        assert!(!cc.is_candidate(&tramp));
        assert_eq!(cc.lookup_function_fnaddr(&tramp), Some(0x1000));
    }

    /// Outer's false `we_are_jitted` arm is `_ll_alloc_and_set_nojit`,
    /// not the nonnull loop.
    #[test]
    fn outer_nojit_arm_calls_nojit_not_the_loop() {
        let (cc, names) = mint(&i64_identity());
        let oops = graph_ops(&cc, &names.outer);
        assert!(calls_named(&oops, &names.nojit));
        assert!(calls_named(&oops, &names.jit));
        assert!(!calls_named(&oops, &names.nonnull));
    }

    /// Non-Ptr nojit: `ll_newlist`, then the zero test selects
    /// `ll_arrayclear` or `ll_arrayfill`. Float compares bits so `-0.0`
    /// fills (only `+0.0` has bits 0 and clears).
    #[test]
    fn nojit_zero_nonptr_clears_and_nonzero_fills() {
        let (cc, names) = mint(&float_identity());
        let nops = graph_ops(&cc, &names.nojit);
        assert!(
            nops.iter()
                .any(|op| matches!(op.kind, OpKind::NewArray { .. })),
            "ll_newlist is uninitialised new_array for a non-Ptr item"
        );
        assert!(
            nops.iter().any(|op| {
                matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "convert_float_bytes_to_longlong")
            }),
            "Float zero test compares bits; -0.0 fills, only +0.0 clears"
        );
        assert!(
            !nops.iter().any(|op| {
                matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "float_is_true")
            }),
            "IEEE float_is_true would treat -0.0 as zero"
        );
        assert!(calls_named(&nops, "ll_arrayclear"));
        assert!(calls_named(&nops, "ll_arrayfill"));
        assert!(
            !nops
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayWrite { .. })),
            "nojit does not store item in a loop"
        );
    }

    /// Ptr-element nojit skips `ll_arrayclear`: `ll_newlist` already
    /// zeroed the slots (`zero_gc_pointers_inside`).
    #[test]
    fn nojit_ptr_item_skips_arrayclear() {
        let (cc, names) = mint(&ptr_identity());
        let nops = graph_ops(&cc, &names.nojit);
        assert!(
            nops.iter()
                .any(|op| matches!(op.kind, OpKind::NewArrayClear { .. }))
        );
        assert!(!calls_named(&nops, "ll_arrayclear"));
        assert!(calls_named(&nops, "ll_arrayfill"));
    }

    /// `look_inside_iff` stamps `specialize:call_location` on the
    /// original and the trampoline; the wrapper `f` does not get it.
    #[test]
    fn look_inside_iff_stamps_call_location_on_orig_and_trampoline() {
        let (cc, names) = mint(&i64_identity());
        assert!(
            graph_hints(&cc, &names.nonnull)
                .iter()
                .any(|h| h == "specialize:call_location")
        );
        assert!(
            graph_hints(&cc, &names.trampoline)
                .iter()
                .any(|h| h == "specialize:call_location")
        );
        assert!(
            !graph_hints(&cc, &names.wrapper)
                .iter()
                .any(|h| h == "specialize:call_location"),
            "look_inside_iff wrapper f does not get call_location"
        );
    }
}
