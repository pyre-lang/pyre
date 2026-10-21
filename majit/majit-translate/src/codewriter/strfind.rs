//! Skip-spine mint of `LLHelpers.ll_find` (`rstr.py`).
//!
//! `pairtype(AbstractStringRepr, AbstractStringRepr).rtype_contains`
//! (`rstr.py`) is `gendirectcall(ll_strlen)` then
//! `gendirectcall(ll_find, s1, s2, 0, end)` then `int_ne(vres, -1)`.
//! A graph that stays on the rich-`OpKind` spine never meets the rtyper, so
//! the front's `BinOp { op: "contains" }` would assemble as the unwired
//! `int_contains/rr>i`.
//!
//! This module gives that spine the helper `rtype_contains` would have
//! minted: [`ll_find_path`] registers [`build_ll_find_helper_graph`] as a
//! Spine-B candidate (`CallControl::register_opname_helper_graph`), matching
//! `codewriter::minmax` / `codewriter::getslice`.
//!
//! [`ll_find_char_path`] and [`search_elidable_path`] are the elidable
//! residual siblings (`LLHelpers.ll_find_char`, `rstring._search_elidable`).
//! Shape matches [`super::alloc_and_set::register`]: bind fnaddr →
//! `register_function_graph` → `add_candidate_graph_if_look_inside`.

use std::rc::Rc;

use crate::codewriter::call::CallControl;
use crate::model::{ConcreteType, FunctionGraph, OpKind, ValueType};
use crate::parse::CallPath;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::lltypesystem::rstr::{
    STRPTR, build_ll_find_char_helper_graph, build_ll_find_helper_graph,
    build_ll_search_helper_graph,
};

/// `LLHelpers.ll_find` (`rstr.py`).
pub const LL_FIND: &str = "ll_find";
/// `LLHelpers.ll_find_char` (`rstr.py`).
pub const LL_FIND_CHAR: &str = "ll_find_char";
/// `LLHelpers.ll_search` (`rstr.py`).
pub const LL_SEARCH: &str = "ll_search";
/// `rstring._search_elidable` (`rlib/rstring.py`).
pub const SEARCH_ELIDABLE: &str = "_search_elidable";

/// Native `HELPER_FNADDRS` leaf after crate-strip.
fn native_fnaddr_key() -> CallPath {
    CallPath::from_segments(["lltypesystem", "rstr", LL_FIND_CHAR])
}

fn search_elidable_native_fnaddr_key() -> CallPath {
    CallPath::from_segments(["rstring", SEARCH_ELIDABLE])
}

#[cfg(test)]
fn helper_fnaddr_path() -> String {
    format!("majit_rlib::lltypesystem::rstr::{LL_FIND_CHAR}")
}

#[cfg(test)]
fn search_elidable_fnaddr_path() -> String {
    format!("majit_rlib::rstring::{SEARCH_ELIDABLE}")
}

/// The helper path for `ll_find`, minting and registering the graph on first use.
///
/// Also registers the callees `ll_find_char`, `ll_search`, and
/// `_search_elidable` so neither spine meets an unknown callee.
pub fn ll_find_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_FIND]);
    if cc.jitcode_handle(&path).is_none() {
        let py = build_ll_find_helper_graph(LL_FIND, STRPTR.clone()).expect("ll_find helper graph");
        let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
        let flow = Rc::try_unwrap(graph)
            .expect("fresh ll_find helper is uniquely owned")
            .into_inner();
        cc.register_opname_helper_graph(path.clone(), flow);
    }
    let _ = ll_find_char_path(cc, &*STRPTR);
    let _ = ll_search_path(cc);
    path
}

/// The helper path for `ll_search`, minting a look-inside graph.
///
/// Upstream `LLHelpers.ll_search` has no JIT hint. The STR body is
/// `direct_call _search_elidable`.
pub fn ll_search_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_SEARCH]);
    if cc.jitcode_handle(&path).is_none() {
        let py = build_ll_search_helper_graph(LL_SEARCH, STRPTR.clone())
            .expect("ll_search helper graph");
        let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
        let flow = Rc::try_unwrap(graph)
            .expect("fresh ll_search helper is uniquely owned")
            .into_inner();
        cc.register_opname_helper_graph(path.clone(), flow);
    }
    let _ = search_elidable_path(cc);
    path
}

/// The helper path for `ll_find_char`, minting the elidable residual.
///
/// `look_inside_graph` refuses an elidable function, so the call stays
/// residual. The executable address is the `HELPER_FNADDRS` row of
/// `lltypesystem::rstr::ll_find_char` (`Ptr(STR)` only).
pub fn ll_find_char_path(cc: &mut CallControl, ptr_lltype: &LowLevelType) -> CallPath {
    let path = CallPath::from_segments([LL_FIND_CHAR]);
    if cc.has_function_graph(&path) {
        return path;
    }
    let _ = build_ll_find_char_helper_graph(LL_FIND_CHAR, ptr_lltype.clone())
        .expect("ll_find_char helper graph");
    if ptr_lltype == &*STRPTR {
        bind_ll_find_char_fnaddr(cc);
    }
    let mut graph = residual_ll_find_char_graph();
    graph.push_hint("elidable");
    cc.register_function_graph(path.clone(), graph);
    cc.add_candidate_graph_if_look_inside(path.clone());
    path
}

fn bind_ll_find_char_fnaddr(cc: &mut CallControl) {
    let dest = CallPath::from_segments([LL_FIND_CHAR]);
    if cc.lookup_function_fnaddr(&dest).is_some() {
        return;
    }
    let src = native_fnaddr_key();
    if cc.lookup_function_fnaddr(&src).is_some() {
        cc.alias_function_fnaddr(dest, &src);
    }
}

/// The helper path for `_search_elidable`, minting the elidable residual.
///
/// `look_inside_graph` refuses an elidable function, so the call stays
/// residual. The executable address is the `HELPER_FNADDRS` row of
/// `rstring::_search_elidable` (`Ptr(STR)` only).
pub fn search_elidable_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([SEARCH_ELIDABLE]);
    if cc.has_function_graph(&path) {
        return path;
    }
    bind_search_elidable_fnaddr(cc);
    let mut graph = residual_search_elidable_graph();
    graph.push_hint("elidable");
    cc.register_function_graph(path.clone(), graph);
    cc.add_candidate_graph_if_look_inside(path.clone());
    path
}

fn bind_search_elidable_fnaddr(cc: &mut CallControl) {
    let dest = CallPath::from_segments([SEARCH_ELIDABLE]);
    if cc.lookup_function_fnaddr(&dest).is_some() {
        return;
    }
    let src = search_elidable_native_fnaddr_key();
    if cc.lookup_function_fnaddr(&src).is_some() {
        cc.alias_function_fnaddr(dest, &src);
    }
}

/// Body is never traced (`look_inside_graph` refuses elidable). Four
/// arguments matching `ll_find_char(s, ch, start, end)`.
fn residual_ll_find_char_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new(LL_FIND_CHAR);
    let start = graph.startblock;
    let s = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    let ch = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let start_i = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let end = graph.alloc_value_var_with_type(ConcreteType::Signed);
    graph.push_inputarg_var(start, s.clone());
    graph.push_inputarg_var(start, ch.clone());
    graph.push_inputarg_var(start, start_i.clone());
    graph.push_inputarg_var(start, end.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "s".to_string(),
            ty: ValueType::Str,
            class_root: None,
        },
        s,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "ch".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        ch,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "start".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        start_i,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "end".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        end,
    );
    let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
    graph.push_op_with_result_var(start, OpKind::ConstInt(-1), result.clone());
    graph.set_return(start, Some(result));
    graph.with_return_type("i64")
}

/// Body is never traced (`look_inside_graph` refuses elidable). Five
/// arguments matching `_search_elidable(s1, s2, start, end, mode)`.
fn residual_search_elidable_graph() -> FunctionGraph {
    let mut graph = FunctionGraph::new(SEARCH_ELIDABLE);
    let start = graph.startblock;
    let s1 = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    let s2 = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    let start_i = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let end = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let mode = graph.alloc_value_var_with_type(ConcreteType::Signed);
    graph.push_inputarg_var(start, s1.clone());
    graph.push_inputarg_var(start, s2.clone());
    graph.push_inputarg_var(start, start_i.clone());
    graph.push_inputarg_var(start, end.clone());
    graph.push_inputarg_var(start, mode.clone());
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "s1".to_string(),
            ty: ValueType::Str,
            class_root: None,
        },
        s1,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "s2".to_string(),
            ty: ValueType::Str,
            class_root: None,
        },
        s2,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "start".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        start_i,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "end".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        end,
    );
    graph.push_op_with_result_var(
        start,
        OpKind::Input {
            name: "mode".to_string(),
            ty: ValueType::Int,
            class_root: None,
        },
        mode,
    );
    let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
    graph.push_op_with_result_var(start, OpKind::ConstInt(-1), result.clone());
    graph.set_return(start, Some(result));
    graph.with_return_type("i64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::{CallKind, is_symbolic_fnaddr};
    use crate::codewriter::codewriter::CodeWriter;
    use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
    use crate::codewriter::jtransform_opname::lower_graph;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph as FlowGraph, GraphFunc, Hlvalue,
        Link, SpaceOperation,
    };
    use crate::model::{CallTarget, OpKind};
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};
    use crate::translator::rtyper::rtyper::{constant_with_lltype, variable_with_lltype};
    use majit_ir::descr::ExtraEffect;

    fn mint() -> CallControl {
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&helper_fnaddr_path(), 0x1000);
        cc.register_macro_helper_trace_fnaddr(&search_elidable_fnaddr_path(), 0x2000);
        let _ = ll_find_path(&mut cc);
        cc
    }

    fn call_op() -> crate::model::SpaceOperation {
        crate::model::SpaceOperation {
            result: Some(crate::flowspace::model::Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::function_path([LL_FIND_CHAR]),
                args: crate::model::call_args(vec![
                    crate::flowspace::model::Variable::new(),
                    crate::flowspace::model::Variable::new(),
                    crate::flowspace::model::Variable::new(),
                    crate::flowspace::model::Variable::new(),
                ]),
                result_ty: ValueType::Int,
            },
        }
    }

    #[test]
    fn ll_find_char_path_is_residual_not_a_candidate() {
        let cc = mint();
        let path = CallPath::from_segments([LL_FIND_CHAR]);
        assert!(cc.has_function_graph(&path));
        assert!(
            !cc.is_candidate(&path),
            "look_inside_graph refuses an elidable function"
        );
        let graph = cc
            .function_graphs()
            .get(&path)
            .expect("registered ll_find_char graph");
        assert!(graph.func.elidable);
        let call = call_op();
        assert!(cc.graphs_from(&call).is_none());
        assert_eq!(cc.guess_call_kind(&call), CallKind::Residual);
        assert!(cc.is_elidable(&CallTarget::function_path([LL_FIND_CHAR])));
        assert_eq!(cc.lookup_function_fnaddr(&path), Some(0x1000));
        let OpKind::Call { target, .. } = &call.kind else {
            panic!("call_op is a Call");
        };
        let binding = cc.fnaddr_binding_for_target(target);
        assert!(
            !binding.symbolic,
            "ll_find_char residual must be executable"
        );
        assert!(!is_symbolic_fnaddr(binding.addr));
        assert_eq!(binding.addr, 0x1000);
    }

    /// `direct_call ll_find_char(s, ch, start, end)` becomes
    /// `residual_call_*` with `EF_ELIDABLE_CANNOT_RAISE` and a real fnaddr.
    #[test]
    fn ll_find_char_direct_call_lowers_to_elidable_residual() {
        let s = variable_with_lltype("s", STRPTR.clone());
        let ch = variable_with_lltype("ch", LowLevelType::Signed);
        let start = variable_with_lltype("start", LowLevelType::Signed);
        let end = variable_with_lltype("end", LowLevelType::Signed);
        let startblock = Block::shared(vec![
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(ch.clone()),
            Hlvalue::Variable(start.clone()),
            Hlvalue::Variable(end.clone()),
        ]);
        let return_var = variable_with_lltype("out", LowLevelType::Signed);
        let mut flow = FlowGraph::with_return_var(
            "ll_find_char_caller",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: vec![
                STRPTR.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
            result: LowLevelType::Signed,
        };
        let funcptr = functionptr(ft, LL_FIND_CHAR, None, Some(LL_FIND_CHAR.to_string()));
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                callee,
                Hlvalue::Variable(s),
                Hlvalue::Variable(ch),
                Hlvalue::Variable(start),
                Hlvalue::Variable(end),
            ],
            Hlvalue::Variable(out.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(out)],
                Some(flow.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let mut func = GraphFunc::new(
            "ll_find_char_caller",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func._jit_look_inside_ = Some(true);
        flow.func = Some(func);

        let caller = lower_graph(&flow);
        let mut cc = mint();
        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&caller);
        let ops = &transformed.graph.block(caller.startblock).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "ll_find_char must not inline: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    funcptr,
                    ..
                } => Some((descriptor.extra_info.extraeffect, funcptr.clone())),
                _ => None,
            })
            .expect("direct_call ll_find_char must rewrite to residual_call");
        assert_eq!(residual.0, ExtraEffect::ElidableCannotRaise);
        let fnaddr = ops.iter().find_map(|op| match &op.kind {
            OpKind::ConstFnAddr {
                value, symbolic, ..
            } => Some((*value, *symbolic)),
            _ => None,
        });
        let (addr, symbolic) = fnaddr.expect("residual funcptr is a ConstFnAddr");
        assert!(!symbolic, "ll_find_char fnaddr must be executable");
        assert!(!is_symbolic_fnaddr(addr));
        assert_eq!(addr, 0x1000);
    }

    #[test]
    fn ll_find_path_registers_search_callees() {
        let cc = mint();
        let find = CallPath::from_segments([LL_FIND]);
        let search = CallPath::from_segments([LL_SEARCH]);
        let find_char = CallPath::from_segments([LL_FIND_CHAR]);
        let elidable = CallPath::from_segments([SEARCH_ELIDABLE]);
        assert!(cc.jitcode_handle(&find).is_some());
        assert!(cc.is_candidate(&find));
        assert!(cc.jitcode_handle(&search).is_some());
        assert!(cc.is_candidate(&search));
        assert!(cc.has_function_graph(&find_char));
        assert!(!cc.is_candidate(&find_char));
        assert!(cc.has_function_graph(&elidable));
        assert!(
            !cc.is_candidate(&elidable),
            "look_inside_graph refuses an elidable function"
        );
        assert_eq!(cc.lookup_function_fnaddr(&find_char), Some(0x1000));
        assert_eq!(cc.lookup_function_fnaddr(&elidable), Some(0x2000));
        assert!(cc.is_elidable(&CallTarget::function_path([SEARCH_ELIDABLE])));
    }

    fn unwrap_pygraph(py: crate::flowspace::pygraph::PyGraph) -> FlowGraph {
        let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
        Rc::try_unwrap(graph)
            .expect("fresh helper is uniquely owned")
            .into_inner()
    }

    fn contains_graph() -> crate::model::FunctionGraph {
        let mut graph = crate::model::FunctionGraph::new("contains_through_ll_find");
        let lhs = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Str,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Str,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "contains".into(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result.clone()));
        lhs.set_concretetype(Some(STRPTR.clone()));
        rhs.set_concretetype(Some(STRPTR.clone()));
        crate::model::FunctionGraph::set_concretetype_of_inline(&result, ConcreteType::Signed);
        graph
    }

    fn residual_effect_and_fnaddrs(
        graph: &crate::model::FunctionGraph,
    ) -> (Vec<ExtraEffect>, Vec<(i64, bool)>) {
        let mut effects = Vec::new();
        let mut addrs = Vec::new();
        for block in graph.iter_blocks() {
            for op in &block.operations {
                match &op.kind {
                    OpKind::CallResidual { descriptor, .. } => {
                        effects.push(descriptor.extra_info.extraeffect);
                    }
                    OpKind::ConstFnAddr {
                        value, symbolic, ..
                    } => addrs.push((*value, *symbolic)),
                    _ => {}
                }
            }
        }
        (effects, addrs)
    }

    /// `contains` through `ll_find` residuals `ll_find_char` (m == 1) and
    /// `_search_elidable` (m > 1 via `ll_search`), both
    /// `ElidableCannotRaise` with real fnaddrs. `jit_record_exact_value`
    /// lives on the `ll_find` CFG; Spine B lowers it to
    /// `record_exact_value_i` when that rewrite is present.
    #[test]
    fn ll_find_contains_residuals_ll_find_char_and_search_elidable() {
        let contains = contains_graph();
        let mut cc = mint();
        let transformed = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&contains);
        let contains_ops = &transformed.graph.block(contains.startblock).operations;
        assert!(
            !contains_ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "contains"
            )),
            "leftover contains must not survive: {contains_ops:?}"
        );
        assert!(
            contains_ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "ll_find is an inline_call: {contains_ops:?}"
        );

        let find_char_flow = {
            let s = variable_with_lltype("s", STRPTR.clone());
            let ch = variable_with_lltype("ch", LowLevelType::Signed);
            let start = variable_with_lltype("start", LowLevelType::Signed);
            let end = variable_with_lltype("end", LowLevelType::Signed);
            let startblock = Block::shared(vec![
                Hlvalue::Variable(s.clone()),
                Hlvalue::Variable(ch.clone()),
                Hlvalue::Variable(start.clone()),
                Hlvalue::Variable(end.clone()),
            ]);
            let return_var = variable_with_lltype("out", LowLevelType::Signed);
            let mut flow = FlowGraph::with_return_var(
                "ll_find_char_from_contains",
                startblock.clone(),
                Hlvalue::Variable(return_var),
            );
            let ft = FuncType {
                args: vec![
                    STRPTR.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                result: LowLevelType::Signed,
            };
            let funcptr = functionptr(ft, LL_FIND_CHAR, None, Some(LL_FIND_CHAR.to_string()));
            let callee =
                constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
            let out = variable_with_lltype("out", LowLevelType::Signed);
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "direct_call",
                vec![
                    callee,
                    Hlvalue::Variable(s),
                    Hlvalue::Variable(ch),
                    Hlvalue::Variable(start),
                    Hlvalue::Variable(end),
                ],
                Hlvalue::Variable(out.clone()),
            ));
            startblock.closeblock(vec![
                Link::new(
                    vec![Hlvalue::Variable(out)],
                    Some(flow.returnblock.clone()),
                    None,
                )
                .into_ref(),
            ]);
            flow.func = Some(GraphFunc::new(
                "ll_find_char_from_contains",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
            flow
        };
        let find_char_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&find_char_flow));
        let (find_effects, find_addrs) = residual_effect_and_fnaddrs(&find_char_xf.graph);
        assert!(
            find_effects
                .iter()
                .any(|effect| *effect == ExtraEffect::ElidableCannotRaise),
            "ll_find_char residual is ElidableCannotRaise: {find_effects:?}"
        );
        assert!(
            find_addrs.iter().any(|(addr, symbolic)| *addr == 0x1000
                && !*symbolic
                && !is_symbolic_fnaddr(*addr)),
            "ll_find_char fnaddr is executable: {find_addrs:?}"
        );

        let search_flow = unwrap_pygraph(
            build_ll_search_helper_graph(LL_SEARCH, STRPTR.clone())
                .expect("ll_search helper graph"),
        );
        let search_lowered = lower_graph(&search_flow);
        let search_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&search_lowered);
        let (search_effects, search_addrs) = residual_effect_and_fnaddrs(&search_xf.graph);
        assert_eq!(search_effects, vec![ExtraEffect::ElidableCannotRaise]);
        assert!(
            search_addrs.iter().any(|(addr, symbolic)| *addr == 0x2000
                && !*symbolic
                && !is_symbolic_fnaddr(*addr)),
            "_search_elidable fnaddr is executable: {search_addrs:?}"
        );

        let find_py =
            build_ll_find_helper_graph(LL_FIND, STRPTR.clone()).expect("ll_find helper graph");
        let find_ops: Vec<String> = find_py
            .graph
            .borrow()
            .iterblocks()
            .iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            find_ops.iter().any(|op| op == "jit_record_exact_value"),
            "ll_find CFG records res < end: {find_ops:?}"
        );
    }

    fn leftover_call_targets(graph: &crate::model::FunctionGraph) -> Vec<String> {
        graph
            .iter_blocks()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target.to_string()),
                _ => None,
            })
            .collect()
    }

    /// `contains` through `ll_find` drains with no raw `direct_call` in any
    /// emitted jitcode. `ll_find_char` and `_search_elidable` are
    /// `residual_call_*` with `ElidableCannotRaise`; `ll_search` is an
    /// `inline_call_*` (`jtransform.py rewrite_op_direct_call`).
    #[test]
    fn contains_through_ll_find_emits_no_direct_call() {
        let contains = contains_graph();
        let mut cc = mint();
        let _ = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&contains);

        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut cc, &GraphTransformConfig::default());

        let direct_call_insns: Vec<&String> = codewriter
            .assembler
            .insns()
            .keys()
            .filter(|key| key.starts_with("direct_call"))
            .collect();
        assert!(
            direct_call_insns.is_empty(),
            "raw direct_call in assembled insns: {direct_call_insns:?}"
        );
        for (path, jitcode) in cc.jitcodes() {
            let dump = jitcode.dump();
            assert!(
                !dump
                    .split_whitespace()
                    .any(|tok| tok == "call" || tok.starts_with("direct_call")),
                "raw direct_call in {path}: {dump}"
            );
        }

        let find_flow =
            unwrap_pygraph(build_ll_find_helper_graph(LL_FIND, STRPTR.clone()).expect("ll_find"));
        let find_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&find_flow));
        let leftover = leftover_call_targets(&find_xf.graph);
        assert!(
            leftover.is_empty(),
            "ll_find leftover Call after rewrite_op_direct_call: {leftover:?}"
        );
        assert!(
            find_xf
                .graph
                .iter_blocks()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "ll_search is look-inside: inline_call"
        );
        let (find_effects, _) = residual_effect_and_fnaddrs(&find_xf.graph);
        assert!(
            find_effects
                .iter()
                .any(|effect| *effect == ExtraEffect::ElidableCannotRaise),
            "ll_find_char residual is ElidableCannotRaise: {find_effects:?}"
        );

        let search_flow = unwrap_pygraph(
            build_ll_search_helper_graph(LL_SEARCH, STRPTR.clone()).expect("ll_search"),
        );
        let search_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&search_flow));
        let leftover = leftover_call_targets(&search_xf.graph);
        assert!(
            leftover.is_empty(),
            "ll_search leftover Call after rewrite_op_direct_call: {leftover:?}"
        );
        let (search_effects, _) = residual_effect_and_fnaddrs(&search_xf.graph);
        assert_eq!(search_effects, vec![ExtraEffect::ElidableCannotRaise]);
    }

    /// `direct_call _search_elidable(...)` becomes `residual_call_*`
    /// with `EF_ELIDABLE_CANNOT_RAISE` and a real fnaddr.
    #[test]
    fn search_elidable_direct_call_lowers_to_elidable_residual() {
        let s1 = variable_with_lltype("s1", STRPTR.clone());
        let s2 = variable_with_lltype("s2", STRPTR.clone());
        let start = variable_with_lltype("start", LowLevelType::Signed);
        let end = variable_with_lltype("end", LowLevelType::Signed);
        let mode = variable_with_lltype("mode", LowLevelType::Signed);
        let startblock = Block::shared(vec![
            Hlvalue::Variable(s1.clone()),
            Hlvalue::Variable(s2.clone()),
            Hlvalue::Variable(start.clone()),
            Hlvalue::Variable(end.clone()),
            Hlvalue::Variable(mode.clone()),
        ]);
        let return_var = variable_with_lltype("out", LowLevelType::Signed);
        let mut flow = FlowGraph::with_return_var(
            "search_elidable_caller",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: vec![
                STRPTR.clone(),
                STRPTR.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
            result: LowLevelType::Signed,
        };
        let funcptr = functionptr(ft, SEARCH_ELIDABLE, None, Some(SEARCH_ELIDABLE.to_string()));
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                callee,
                Hlvalue::Variable(s1),
                Hlvalue::Variable(s2),
                Hlvalue::Variable(start),
                Hlvalue::Variable(end),
                Hlvalue::Variable(mode),
            ],
            Hlvalue::Variable(out.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(out)],
                Some(flow.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        flow.func = Some(GraphFunc::new(
            "search_elidable_caller",
            Constant::new(ConstValue::Dict(Default::default())),
        ));

        let caller = lower_graph(&flow);
        let mut cc = mint();
        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&caller);
        let ops = &transformed.graph.block(caller.startblock).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "_search_elidable must not inline: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    funcptr,
                    ..
                } => Some((descriptor.extra_info.extraeffect, funcptr.clone())),
                _ => None,
            })
            .expect("direct_call _search_elidable must rewrite to residual_call");
        assert_eq!(residual.0, ExtraEffect::ElidableCannotRaise);
        let fnaddr = ops.iter().find_map(|op| match &op.kind {
            OpKind::ConstFnAddr {
                value, symbolic, ..
            } => Some((*value, *symbolic)),
            _ => None,
        });
        let (addr, symbolic) = fnaddr.expect("residual funcptr is a ConstFnAddr");
        assert!(!symbolic, "_search_elidable fnaddr must be executable");
        assert!(!is_symbolic_fnaddr(addr));
        assert_eq!(addr, 0x2000);
    }
}
