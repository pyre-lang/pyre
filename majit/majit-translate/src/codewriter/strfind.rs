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
//! minted: [`ll_find_path`] registers [`build_ll_find_helper_graph_on`] as a
//! Spine-B candidate (`CallControl::register_opname_helper_graph`), matching
//! `codewriter::minmax` / `codewriter::getslice`.
//!
//! Callee funcptrs (`ll_find_char`, `ll_search`, `_search_elidable`,
//! `_search_normal`) carry their graphs (`HighLevelOp.gendirectcall` /
//! `getcallable`). [`ll_find_path`] walks those funcptrs depth-first
//! (`call.py CallControl.find_all_graphs`) and registers each graph
//! once. `policy.look_inside_graph` decides candidacy. A residual
//! callee published on `HELPER_FNADDRS` aliases that native fnaddr.

use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::codewriter::call::CallControl;
use crate::codewriter::jtransform_opname::lower_graph;
use crate::parse::CallPath;
use crate::policy::{DefaultJitPolicy, JitPolicy};
use crate::rlib::rstring::effect_graph_from_search_helper;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::lltypesystem::rstr::{
    STRPTR, UNICODEPTR, build_ll_find_char_helper_graph, build_ll_find_helper_graph_on,
    build_ll_search_helper_graph_on, direct_call_funcobjs, graph_ref_for_funcptr_id,
};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// `LLHelpers.ll_find` (`rstr.py`).
pub const LL_FIND: &str = "ll_find";
/// `LLHelpers.ll_find_char` (`rstr.py`).
pub const LL_FIND_CHAR: &str = "ll_find_char";
/// `LLHelpers.ll_search` (`rstr.py`).
pub const LL_SEARCH: &str = "ll_search";
/// `rstring._search_elidable` (`rlib/rstring.py`).
pub const SEARCH_ELIDABLE: &str = "_search_elidable";
/// `rstring._search_normal` (`rlib/rstring.py`).
pub const SEARCH_NORMAL: &str = "_search_normal";

#[cfg(test)]
fn helper_fnaddr_path() -> String {
    format!("majit_rlib::lltypesystem::rstr::{LL_FIND_CHAR}")
}

#[cfg(test)]
fn search_elidable_fnaddr_path() -> String {
    format!("majit_rlib::rstring::{SEARCH_ELIDABLE}")
}

#[cfg(test)]
fn search_normal_fnaddr_path() -> String {
    format!("majit_rlib::rstring::{SEARCH_NORMAL}")
}

/// The helper path for `ll_find`, minting and registering the graph on first use.
///
/// Callees `ll_find_char`, `ll_search`, and `_search_elidable` come off
/// the funcptrs `HighLevelOp.gendirectcall` attached, so this path
/// stays self-contained without a second registration site.
pub fn ll_find_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_FIND]);
    if cc.jitcode_handle(&path).is_none() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let py = build_ll_find_helper_graph_on(&rtyper, LL_FIND, STRPTR.clone())
            .expect("ll_find helper graph");
        register_direct_call_callees(cc, &rtyper, &py.graph.borrow());
        let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
        let flow = Rc::try_unwrap(graph)
            .expect("fresh ll_find helper is uniquely owned")
            .into_inner();
        cc.register_opname_helper_graph(path.clone(), flow);
    }
    path
}

/// The helper path for `ll_search`, minting a look-inside graph.
///
/// Upstream `LLHelpers.ll_search` has no JIT hint. The STR body is
/// `direct_call _search_elidable`. Skips when [`ll_find_path`] already
/// consumed the graph from the find helper's funcptr.
pub fn ll_search_path(cc: &mut CallControl) -> CallPath {
    let path = CallPath::from_segments([LL_SEARCH]);
    if callee_already_registered(cc, &path) {
        return path;
    }
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = RPythonTyper::new(&ann);
    let py = build_ll_search_helper_graph_on(&rtyper, LL_SEARCH, STRPTR.clone())
        .expect("ll_search helper graph");
    register_direct_call_callees(cc, &rtyper, &py.graph.borrow());
    let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
    let flow = Rc::try_unwrap(graph)
        .expect("fresh ll_search helper is uniquely owned")
        .into_inner();
    cc.register_opname_helper_graph(path.clone(), flow);
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
    let py = build_ll_find_char_helper_graph(LL_FIND_CHAR, ptr_lltype.clone())
        .expect("ll_find_char helper graph");
    debug_assert!(py.func._elidable_function_);
    register_callee_graph(cc, path.clone(), &py.graph.borrow(), true);
    path
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
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = RPythonTyper::new(&ann);
    let py = crate::rlib::rstring::build_search_elidable_helper_graph(
        &rtyper,
        SEARCH_ELIDABLE,
        STRPTR.clone(),
    )
    .expect("_search_elidable helper graph");
    register_direct_call_callees(cc, &rtyper, &py.graph.borrow());
    register_callee_graph(cc, path.clone(), &py.graph.borrow(), true);
    path
}

fn callee_already_registered(cc: &CallControl, path: &CallPath) -> bool {
    cc.has_function_graph(path) || cc.jitcode_handle(path).is_some() || cc.has_opname_graph(path)
}

/// Walk `direct_call` callee graphs depth-first, once per path
/// (`call.py CallControl.find_all_graphs`). Each callee is
/// `register_function_graph` then `add_candidate_graph_if_look_inside`;
/// `policy.look_inside_graph` decides candidacy. Nested callees of a
/// residual are registered for `getcalldescr` only.
fn register_direct_call_callees(
    cc: &mut CallControl,
    rtyper: &RPythonTyper,
    graph: &crate::flowspace::model::FunctionGraph,
) {
    visit_direct_call_callees(cc, rtyper, graph, true);
}

fn visit_direct_call_callees(
    cc: &mut CallControl,
    rtyper: &RPythonTyper,
    graph: &crate::flowspace::model::FunctionGraph,
    offer_candidate: bool,
) {
    for func in direct_call_funcobjs(graph) {
        let graph_id = func
            .graph
            .expect("HighLevelOp.gendirectcall funcptr carries graph");
        let callee = graph_ref_for_funcptr_id(rtyper, graph_id)
            .expect("funcobj.graph lives on translator.graphs");
        let path = CallPath::from_segments([func._name.as_str()]);
        if callee_already_registered(cc, &path) {
            continue;
        }
        register_callee_graph(cc, path.clone(), &callee.borrow(), offer_candidate);
        let nested_offer = offer_candidate && cc.is_candidate(&path);
        visit_direct_call_callees(cc, rtyper, &callee.borrow(), nested_offer);
    }
}

/// Register one callee. A look-inside graph is `lower_graph`; a refused
/// graph is effect-only so `getcalldescr` walks the real body. Residual
/// native fnaddr is the `HELPER_FNADDRS` row whose leaf is the callee
/// name (`ll_find_char`, `_search_elidable`, `_search_normal`).
fn register_callee_graph(
    cc: &mut CallControl,
    path: CallPath,
    callee: &crate::flowspace::model::FunctionGraph,
    offer_candidate: bool,
) {
    if callee_already_registered(cc, &path) {
        return;
    }
    if !offer_candidate {
        let graph = effect_graph_from_search_helper(callee);
        cc.register_function_graph(path, graph);
        return;
    }
    let effect = effect_graph_from_search_helper(callee);
    let mut probe = DefaultJitPolicy::new();
    let look_inside = probe.look_inside_graph(&effect);
    if look_inside {
        cc.register_function_graph(path.clone(), lower_graph(callee));
    } else {
        cc.register_function_graph(path.clone(), effect);
    }
    cc.add_candidate_graph_if_look_inside(path.clone());
    if cc.is_candidate(&path) {
        let _ = cc.get_jitcode(&path);
    } else {
        bind_residual_native_fnaddr(cc, &path);
    }
}

fn bind_residual_native_fnaddr(cc: &mut CallControl, dest: &CallPath) {
    if cc.lookup_function_fnaddr(dest).is_some() {
        return;
    }
    let Some(name) = dest.last_segment() else {
        return;
    };
    for src in [
        CallPath::from_segments(["lltypesystem", "rstr", name]),
        CallPath::from_segments(["rstring", name]),
    ] {
        if cc.lookup_function_fnaddr(&src).is_some() {
            cc.alias_function_fnaddr(dest.clone(), &src);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::{CallKind, is_symbolic_fnaddr};
    use crate::codewriter::codewriter::CodeWriter;
    use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph as FlowGraph, GraphFunc, Hlvalue,
        Link, SpaceOperation,
    };
    use crate::model::{CallTarget, ConcreteType, OpKind, ValueType};
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};
    use crate::translator::rtyper::lltypesystem::rstr::{
        build_ll_find_helper_graph, build_ll_search_helper_graph,
    };
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

    /// `contains` through `ll_find` residuals `ll_find_char` (m == 1,
    /// `ElidableCannotRaise`) and `_search_elidable` (m > 1 via
    /// `ll_search`, `ElidableOrMemoryError` from `_twoway_preprocess`).
    /// `jit_record_exact_value` lives on the `ll_find` CFG; Spine B
    /// lowers it to `record_exact_value_i` when that rewrite is present.
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
        assert_eq!(search_effects, vec![ExtraEffect::ElidableOrMemoryError]);
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
    /// emitted jitcode. `ll_find_char` is `ElidableCannotRaise`;
    /// `_search_elidable` is `ElidableOrMemoryError`; `ll_search` is an
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
        assert_eq!(search_effects, vec![ExtraEffect::ElidableOrMemoryError]);
    }

    /// `direct_call _search_elidable(...)` becomes `residual_call_*`
    /// with `EF_ELIDABLE_OR_MEMORYERROR` and a real fnaddr.
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
        assert_eq!(residual.0, ExtraEffect::ElidableOrMemoryError);
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

    /// rtyper path alone: callee funcptrs carry graphs, and lowering
    /// without [`ll_find_path`] residuals `ll_find_char`
    /// (`ElidableCannotRaise`) / `_search_elidable`
    /// (`ElidableOrMemoryError`) with executable fnaddrs and inlines
    /// `ll_search`.
    #[test]
    fn rtyper_ll_find_funcptrs_carry_graphs_and_lower_without_ll_find_path() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let py = build_ll_find_helper_graph_on(&rtyper, LL_FIND, STRPTR.clone())
            .expect("ll_find helper graph");

        let callees = direct_call_funcobjs(&py.graph.borrow());
        assert!(
            callees.iter().any(|func| func._name == LL_FIND_CHAR),
            "ll_find calls ll_find_char"
        );
        assert!(
            callees.iter().any(|func| func._name == LL_SEARCH),
            "ll_find calls ll_search"
        );
        for func in &callees {
            let graph_id = func
                .graph
                .expect("HighLevelOp.gendirectcall funcptr carries graph");
            assert!(
                graph_ref_for_funcptr_id(&rtyper, graph_id).is_some(),
                "{} graph lives on translator.graphs",
                func._name
            );
        }

        let search = callees
            .iter()
            .find(|func| func._name == LL_SEARCH)
            .expect("ll_search callee");
        let search_graph =
            graph_ref_for_funcptr_id(&rtyper, search.graph.expect("ll_search graph id"))
                .expect("ll_search graph lives on translator.graphs");
        let nested = direct_call_funcobjs(&search_graph.borrow());
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0]._name, SEARCH_ELIDABLE);
        let nested_id = nested[0]
            .graph
            .expect("_search_elidable funcptr carries graph");
        assert!(graph_ref_for_funcptr_id(&rtyper, nested_id).is_some());

        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&helper_fnaddr_path(), 0x1000);
        cc.register_macro_helper_trace_fnaddr(&search_elidable_fnaddr_path(), 0x2000);
        register_direct_call_callees(&mut cc, &rtyper, &py.graph.borrow());

        let find_char = CallPath::from_segments([LL_FIND_CHAR]);
        let search_path = CallPath::from_segments([LL_SEARCH]);
        let elidable = CallPath::from_segments([SEARCH_ELIDABLE]);
        assert!(cc.has_function_graph(&find_char));
        assert!(!cc.is_candidate(&find_char));
        assert!(cc.has_function_graph(&search_path));
        assert!(cc.is_candidate(&search_path));
        assert!(cc.has_function_graph(&elidable));
        assert!(!cc.is_candidate(&elidable));
        assert!(
            cc.jitcode_handle(&CallPath::from_segments([LL_FIND]))
                .is_none()
        );

        let find_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&py.graph.borrow()));
        assert!(
            find_xf
                .graph
                .iter_blocks()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "ll_search is look-inside: inline_call"
        );
        let (find_effects, find_addrs) = residual_effect_and_fnaddrs(&find_xf.graph);
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

        let search_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&search_graph.borrow()));
        let (search_effects, search_addrs) = residual_effect_and_fnaddrs(&search_xf.graph);
        assert_eq!(search_effects, vec![ExtraEffect::ElidableOrMemoryError]);
        assert!(
            search_addrs.iter().any(|(addr, symbolic)| *addr == 0x2000
                && !*symbolic
                && !is_symbolic_fnaddr(*addr)),
            "_search_elidable fnaddr is executable: {search_addrs:?}"
        );
    }

    fn calldescr(
        cc: &CallControl,
        name: &str,
        args: Vec<majit_ir::value::Type>,
    ) -> crate::codewriter::call::CallDescriptor {
        use crate::codewriter::call::AnalysisCache;
        use crate::model::CallTarget;
        use majit_ir::descr::OopSpecIndex;
        use majit_ir::value::Type;
        let n = args.len();
        let call = crate::model::SpaceOperation {
            result: Some(crate::flowspace::model::Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::function_path([name]),
                args: crate::model::call_args(
                    (0..n)
                        .map(|_| crate::flowspace::model::Variable::new())
                        .collect::<Vec<_>>(),
                ),
                result_ty: ValueType::Int,
            },
        };
        let mut cache = AnalysisCache::default();
        cc.getcalldescr(
            &call,
            args,
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        )
    }

    /// `getcalldescr` (`call.py`) on `ll_find_char`: elidable loop of
    /// `getarrayitem` / `char_eq`, no malloc → `EF_ELIDABLE_CANNOT_RAISE`.
    #[test]
    fn ll_find_char_calldescr_is_elidable_cannot_raise() {
        use majit_ir::value::Type;
        let cc = mint();
        let descr = calldescr(
            &cc,
            LL_FIND_CHAR,
            vec![Type::Ref, Type::Int, Type::Int, Type::Int],
        );
        assert_eq!(
            descr.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
        assert!(!descr.extra_info.can_collect);
        let graph = cc
            .function_graphs()
            .get(&CallPath::from_segments([LL_FIND_CHAR]))
            .expect("ll_find_char graph");
        assert!(
            graph
                .iter_block_ops()
                .any(|(_, op)| matches!(op.kind, OpKind::ArrayRead { .. })),
            "ll_find_char reads s.chars"
        );
    }

    /// `getcalldescr` (`call.py`) on `_search_elidable`: `_canraise`
    /// `"mem"` from `_twoway_preprocess` → `EF_ELIDABLE_OR_MEMORYERROR`.
    #[test]
    fn search_elidable_calldescr_is_elidable_or_memoryerror() {
        use majit_ir::value::Type;
        let cc = mint();
        let descr = calldescr(
            &cc,
            SEARCH_ELIDABLE,
            vec![Type::Ref, Type::Ref, Type::Int, Type::Int, Type::Int],
        );
        assert_eq!(
            descr.extra_info.extraeffect,
            ExtraEffect::ElidableOrMemoryError
        );
        let elidable = CallPath::from_segments([SEARCH_ELIDABLE]);
        assert!(
            nested_has_newarray(&cc, &elidable),
            "_twoway_preprocess table is a GC list malloc"
        );
    }

    fn nested_has_newarray(cc: &CallControl, path: &CallPath) -> bool {
        use std::collections::HashSet;
        fn walk(cc: &CallControl, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
            if !seen.insert(path.clone()) {
                return false;
            }
            let Some(graph) = cc.function_graphs().get(path) else {
                return false;
            };
            for (_, op) in graph.iter_block_ops() {
                if matches!(op.kind, OpKind::NewArray { .. }) {
                    return true;
                }
                if let OpKind::Call { target, .. } = &op.kind {
                    if let Some(segs) = target.path_segments() {
                        let nested = CallPath::from_segments(segs);
                        if walk(cc, &nested, seen) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        walk(cc, path, &mut HashSet::new())
    }

    /// UNICODE `ll_find` look-inside: `inline_call` `ll_search`;
    /// `_search_normal` is a loopy residual (`policy.look_inside_graph`
    /// / `unsafe_loopy_graphs`). Fresh `CallControl` so STR/UNICODE
    /// helper names do not collide.
    #[test]
    fn unicode_ll_find_inlines_ll_search_and_calls_search_normal_residual() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let py = build_ll_find_helper_graph_on(&rtyper, "ll_unicode_find", UNICODEPTR.clone())
            .expect("UNICODE ll_find");
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&search_normal_fnaddr_path(), 0x3000);
        register_direct_call_callees(&mut cc, &rtyper, &py.graph.borrow());
        let search = CallPath::from_segments([LL_SEARCH]);
        let normal = CallPath::from_segments([SEARCH_NORMAL]);
        assert!(cc.is_candidate(&search), "ll_search is look-inside");
        assert!(
            !cc.is_candidate(&normal),
            "_search_normal is residual (contains a loop)"
        );
        assert!(cc.has_function_graph(&normal));
        let mut policy = DefaultJitPolicy::new();
        let normal_graph = cc
            .function_graphs()
            .get(&normal)
            .expect("_search_normal graph");
        assert!(
            !policy.look_inside_graph(&normal_graph),
            "look_inside_graph refuses a loopy _search_normal"
        );
        assert!(
            policy.state().unsafe_loopy_graphs.contains(SEARCH_NORMAL),
            "_search_normal is in unsafe_loopy_graphs: {:?}",
            policy.state().unsafe_loopy_graphs
        );
        assert_eq!(cc.lookup_function_fnaddr(&normal), Some(0x3000));
        assert!(!cc.is_elidable(&CallTarget::function_path([SEARCH_NORMAL])));

        let find_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&py.graph.borrow()));
        assert!(
            find_xf
                .graph
                .iter_blocks()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "UNICODE ll_find inlines ll_search: {:?}",
            leftover_call_targets(&find_xf.graph)
        );

        let callees = direct_call_funcobjs(&py.graph.borrow());
        let search_func = callees
            .iter()
            .find(|func| func._name == LL_SEARCH)
            .expect("ll_search callee");
        let search_graph =
            graph_ref_for_funcptr_id(&rtyper, search_func.graph.expect("ll_search graph id"))
                .expect("ll_search graph");
        let nested = direct_call_funcobjs(&search_graph.borrow());
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0]._name, SEARCH_NORMAL);

        let search_xf = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&lower_graph(&search_graph.borrow()));
        let leftover = leftover_call_targets(&search_xf.graph);
        assert!(
            leftover.is_empty(),
            "UNICODE ll_search leftover Call: {leftover:?}"
        );
        assert!(
            !search_xf
                .graph
                .iter_blocks()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "UNICODE ll_search must not inline _search_normal"
        );
        let (search_effects, search_addrs) = residual_effect_and_fnaddrs(&search_xf.graph);
        assert!(
            search_effects.iter().any(|effect| !matches!(
                *effect,
                ExtraEffect::ElidableCannotRaise
                    | ExtraEffect::ElidableOrMemoryError
                    | ExtraEffect::ElidableCanRaise
            )),
            "_search_normal residual is not elidable: {search_effects:?}"
        );
        assert!(
            search_addrs.iter().any(|(addr, symbolic)| *addr == 0x3000
                && !*symbolic
                && !is_symbolic_fnaddr(*addr)),
            "_search_normal fnaddr is executable: {search_addrs:?}"
        );

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
    }
}
