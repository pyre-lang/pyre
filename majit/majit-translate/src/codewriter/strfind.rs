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
//! once by graph identity (`funcobj.graph` / `GraphKey::of`).
//! `cachedgraph` `name__<argtypes>` is the alt_name label
//! (`description.py FunctionDesc.cachedgraph`); Array pointees spell
//! the element type (`array_Char`). `policy.look_inside_graph` decides
//! candidacy. A residual callee aliases the `HELPER_FNADDRS` row whose
//! native takes the same STR / UNICODE layout.

use std::collections::HashSet;
use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::codewriter::call::{CallControl, CallKind};
use crate::codewriter::jtransform_opname::lower_graph;
use crate::flowspace::model::{ConstValue, FunctionGraph, Hlvalue, host_env_callable};
use crate::model::{CallTarget, OpKind, ValueType};
use crate::parse::CallPath;
use crate::policy::{DefaultJitPolicy, JitPolicy};
use crate::rlib::rstring::effect_graph_from_search_helper;
use crate::tool::sourcetools::valid_identifier;
use crate::translator::rtyper::lltypesystem::lltype::{
    _func, _ptr_obj, LowLevelType, PtrObj, PtrTarget,
};
use crate::translator::rtyper::lltypesystem::rstr::{
    STRPTR, UNICODEPTR, build_ll_find_char_helper_graph, build_ll_find_helper_graph_on,
    build_ll_search_helper_graph_on, chars_array_ptr_lltype_from_strptr, direct_call_funcobjs,
    graph_ref_for_funcptr_id,
};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// `LLHelpers.ll_find` (`rstr.py`).
pub const LL_FIND: &str = "ll_find";
/// `LLHelpers.ll_find_char` (`rstr.py`).
pub const LL_FIND_CHAR: &str = "ll_find_char";
/// UNICODE `ll_find_char` native published on `HELPER_FNADDRS`.
pub const LL_FIND_CHAR_UNICODE: &str = "ll_find_char_unicode";
/// `LLHelpers.ll_search` (`rstr.py`).
pub const LL_SEARCH: &str = "ll_search";
/// `rstring._search_elidable` (`rlib/rstring.py`).
pub const SEARCH_ELIDABLE: &str = "_search_elidable";
/// `rstring._search_normal` (`rlib/rstring.py`).
pub const SEARCH_NORMAL: &str = "_search_normal";
/// `rlist.py _ll_list_resize_hint_really` original body.
const LL_LIST_RESIZE_HINT_REALLY: &str = "_ll_list_resize_hint_really";
const LL_LIST_RESIZE_HINT_REALLY_F: &str = "_ll_list_resize_hint_really_f";
/// `look_inside_iff` wrapper residualized by `_ll_list_resize_ge`.
const LL_LIST_RESIZE_HINT_REALLY_WRAPPER: &str = "_ll_list_resize_hint_really_look_inside_iff";
/// `dont_look_inside` trampoline the wrapper's false arm `direct_call`s.
const LL_LIST_RESIZE_HINT_REALLY_TRAMPOLINE: &str = "_ll_list_resize_hint_really_trampoline";

#[cfg(test)]
fn helper_fnaddr_path() -> String {
    format!("majit_rlib::lltypesystem::rstr::{LL_FIND_CHAR}")
}

#[cfg(test)]
fn ll_find_char_unicode_fnaddr_path() -> String {
    format!("majit_rlib::lltypesystem::rstr::{LL_FIND_CHAR_UNICODE}")
}

#[cfg(test)]
fn search_elidable_fnaddr_path() -> String {
    format!("majit_rlib::rstring::{SEARCH_ELIDABLE}")
}

#[cfg(test)]
fn search_normal_fnaddr_path() -> String {
    format!("majit_rlib::rstring::{SEARCH_NORMAL}")
}

#[cfg(test)]
fn ll_list_resize_hint_really_fnaddr_path() -> String {
    format!("majit_rlib::lltypesystem::rlist::{LL_LIST_RESIZE_HINT_REALLY}")
}

#[cfg(test)]
fn ll_list_resize_hint_really_f_fnaddr_path() -> String {
    format!("majit_rlib::lltypesystem::rlist::{LL_LIST_RESIZE_HINT_REALLY_F}")
}

/// `description.py FunctionDesc.cachedgraph` `nameof` on one lltype.
/// `default_specialize` keys `cachedgraph` by the argument-lltype tuple;
/// the `name__<argtypes>` alt_name is only a label. Ptr uses
/// `TO.__name__ + "Ptr"` (`lltype.py Ptr.__name__`), Struct and
/// Primitive use `__name__` (`rpy_string`, `Signed`, `Char`). Array
/// spells `array_<OF>` so `Ptr(Array(Char))` and `Ptr(Array(UniChar))`
/// never share a label.
fn lltype_nameof(ty: &LowLevelType) -> String {
    match ty {
        LowLevelType::Ptr(p) => format!("{}Ptr", ptr_target_nameof(&p.TO)),
        LowLevelType::Struct(s) => s._name.clone(),
        LowLevelType::Array(arr) => format!("array_{}", lltype_nameof(&arr.OF)),
        LowLevelType::FixedSizeArray(arr) => {
            format!("array{}_{}", arr.length, lltype_nameof(&arr.OF))
        }
        LowLevelType::ForwardReference(fwd) => fwd
            .resolved()
            .as_ref()
            .map(lltype_nameof)
            .unwrap_or_else(|| "ForwardReference".to_string()),
        LowLevelType::Signed => "Signed".to_string(),
        LowLevelType::Unsigned => "Unsigned".to_string(),
        LowLevelType::SignedLongLong => "SignedLongLong".to_string(),
        LowLevelType::SignedLongLongLong => "SignedLongLongLong".to_string(),
        LowLevelType::UnsignedLongLong => "UnsignedLongLong".to_string(),
        LowLevelType::UnsignedLongLongLong => "UnsignedLongLongLong".to_string(),
        LowLevelType::Float => "Float".to_string(),
        LowLevelType::SingleFloat => "SingleFloat".to_string(),
        LowLevelType::LongFloat => "LongFloat".to_string(),
        LowLevelType::Char => "Char".to_string(),
        LowLevelType::UniChar => "UniChar".to_string(),
        LowLevelType::Bool => "Bool".to_string(),
        LowLevelType::Void => "Void".to_string(),
        LowLevelType::Address => "Address".to_string(),
        other => other.short_name(),
    }
}

fn ptr_target_nameof(to: &PtrTarget) -> String {
    match to {
        PtrTarget::Struct(s) => s._name.clone(),
        PtrTarget::ForwardReference(fwd) => fwd
            .resolved()
            .map(|real| match real {
                LowLevelType::Struct(s) => s._name,
                other => lltype_nameof(&other),
            })
            .unwrap_or_else(|| "ForwardReference".to_string()),
        PtrTarget::Array(arr) => format!("array_{}", lltype_nameof(&arr.OF)),
        PtrTarget::FixedSizeArray(arr) => {
            format!("array{}_{}", arr.length, lltype_nameof(&arr.OF))
        }
        PtrTarget::Func(_) => "func".to_string(),
        PtrTarget::Opaque(opaque) => opaque.tag.clone(),
    }
}

/// `FunctionDesc.cachedgraph` alt_name `name__<argtypes>`
/// (`annlowlevel.py LowLevelAnnotatorPolicy.default_specialize`).
pub fn specialized_helper_leaf(name: &str, args: &[LowLevelType]) -> String {
    let base = unspecialized_helper_name(name);
    let postfix = valid_identifier(args.iter().map(lltype_nameof).collect::<Vec<_>>().join("_"));
    format!("{base}__{postfix}")
}

/// CallPath leaf for one low-level specialization.
pub fn specialized_helper_path(name: &str, args: &[LowLevelType]) -> CallPath {
    CallPath::from_segments([specialized_helper_leaf(name, args)])
}

/// Strip a `name__<argtypes>` leaf back to `func._name`.
pub fn unspecialized_helper_name(name: &str) -> &str {
    name.split("__").next().unwrap_or(name)
}

fn ll_find_arg_types(ptr: &LowLevelType) -> [LowLevelType; 4] {
    [
        ptr.clone(),
        ptr.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
    ]
}

fn element_lltype_of_ptr(ptr: &LowLevelType) -> LowLevelType {
    match chars_array_ptr_lltype_from_strptr(ptr) {
        Ok(LowLevelType::Ptr(p)) => match p.TO {
            PtrTarget::Array(arr) => arr.OF,
            _ => LowLevelType::Signed,
        },
        _ => LowLevelType::Signed,
    }
}

fn ll_find_char_arg_types(ptr: &LowLevelType) -> [LowLevelType; 4] {
    [
        ptr.clone(),
        element_lltype_of_ptr(ptr),
        LowLevelType::Signed,
        LowLevelType::Signed,
    ]
}

fn ll_search_arg_types(ptr: &LowLevelType) -> [LowLevelType; 5] {
    [
        ptr.clone(),
        ptr.clone(),
        LowLevelType::Signed,
        LowLevelType::Signed,
        LowLevelType::Signed,
    ]
}

fn specialized_callee_leaf(func: &_func) -> String {
    specialized_helper_leaf(&func._name, &func.TYPE.args)
}

/// CallPath label for one callee. Prefer the graph's own alt_name when
/// the builder already stamped `name__<argtypes>`; otherwise derive it
/// from `func._name` plus argument lltypes (`cachedgraph` `nameof`).
fn callee_path(func: &_func, graph: &FunctionGraph) -> CallPath {
    let unspecialized = unspecialized_helper_name(&graph.name);
    if graph.name != unspecialized {
        CallPath::from_segments([graph.name.as_str()])
    } else {
        CallPath::from_segments([specialized_callee_leaf(func)])
    }
}

/// `cachedgraph` alt_name is `graph.name`, a label.
/// `call.py find_all_graphs` follows the graph object; GraphStore keys
/// by `(source_identity, name)`, so two distinct graphs must not share
/// a label.
fn stamp_specialized_graph_name(graph: &mut FunctionGraph, leaf: &str) {
    graph.name = leaf.to_string();
    if let Some(func) = graph.func.as_mut() {
        func.name = leaf.to_string();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StringElement {
    Str,
    Unicode,
}

fn string_element_of_ptr(ptr: &LowLevelType) -> Option<StringElement> {
    let chars_ptr = chars_array_ptr_lltype_from_strptr(ptr).ok()?;
    let LowLevelType::Ptr(p) = chars_ptr else {
        return None;
    };
    let PtrTarget::Array(arr) = p.TO else {
        return None;
    };
    match arr.OF {
        LowLevelType::Char => Some(StringElement::Str),
        LowLevelType::UniChar => Some(StringElement::Unicode),
        _ => None,
    }
}

fn string_element_of_args(args: &[LowLevelType]) -> Option<StringElement> {
    args.iter().find_map(string_element_of_ptr)
}

fn graph_arg_lltypes(graph: &FunctionGraph) -> Vec<LowLevelType> {
    graph
        .getargs()
        .iter()
        .filter_map(|hlv| match hlv {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        })
        .collect()
}

/// `description.py FunctionDesc.cachedgraph` stamps `graph.name` to
/// `name__<argtypes>` while the function object (and its oopspec) stays
/// the unspecialized identity. Helper graphs this walk registers under
/// that label are relabelled; a graphless funcptr and a HOST_ENV
/// `rlib/jit.py` builtin keep `func._name`. `jit_conditional_call`
/// stores the funcptr in args[1] (`rlist.py _ll_list_resize_ge`).
fn specialize_direct_call_funcptr_names(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        for op in &mut block.borrow_mut().operations {
            let funcptr_idx = match op.opname.as_str() {
                "direct_call" => 0,
                "jit_conditional_call" => 1,
                _ => continue,
            };
            let Some(Hlvalue::Constant(c)) = op.args.get_mut(funcptr_idx) else {
                continue;
            };
            let ConstValue::LLPtr(ptr) = &mut c.value else {
                continue;
            };
            let PtrObj::Strong(_ptr_obj::Func(func)) = &mut ptr._obj0 else {
                continue;
            };
            if func.graph.is_none() || host_env_jit_builtin_func(func) {
                continue;
            }
            func._name = specialized_callee_leaf(func);
        }
    }
}

/// HOST_ENV `majit_rlib.jit` / `jit` ExtRegistry identity
/// (`host_env_callable`). `cachedgraph` alt_name is a label, not a
/// new function object.
fn host_env_jit_builtin_func(func: &_func) -> bool {
    let segs: Vec<String> = func
        ._name
        .split("::")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let jit_module = matches!(
        segs.as_slice(),
        [crate_name, module, ..] if crate_name == "majit_rlib" && module == "jit"
    ) || matches!(segs.as_slice(), [module, ..] if module == "jit");
    jit_module && host_env_callable(&segs).is_some()
}

/// The helper path for `ll_find`, minting and registering the graph on first use.
///
/// Callees `ll_find_char`, `ll_search`, and `_search_elidable` come off
/// the funcptrs `HighLevelOp.gendirectcall` attached, so this path
/// stays self-contained without a second registration site. STR
/// specialization (`rtype_contains`).
pub fn ll_find_path(cc: &mut CallControl) -> CallPath {
    ll_find_path_on(cc, &STRPTR)
}

/// [`ll_find_path`] for one `Ptr(STR)` / `Ptr(UNICODE)` specialization.
pub fn ll_find_path_on(cc: &mut CallControl, ptr_lltype: &LowLevelType) -> CallPath {
    let path = specialized_helper_path(LL_FIND, &ll_find_arg_types(ptr_lltype));
    if cc.jitcode_handle(&path).is_none() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let leaf = specialized_helper_leaf(LL_FIND, &ll_find_arg_types(ptr_lltype));
        let py = build_ll_find_helper_graph_on(&rtyper, &leaf, ptr_lltype.clone())
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
    ll_search_path_on(cc, &STRPTR)
}

/// [`ll_search_path`] for one `Ptr(STR)` / `Ptr(UNICODE)` specialization.
pub fn ll_search_path_on(cc: &mut CallControl, ptr_lltype: &LowLevelType) -> CallPath {
    let path = specialized_helper_path(LL_SEARCH, &ll_search_arg_types(ptr_lltype));
    if callee_already_registered(cc, &path) {
        return path;
    }
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = RPythonTyper::new(&ann);
    let leaf = specialized_helper_leaf(LL_SEARCH, &ll_search_arg_types(ptr_lltype));
    let py = build_ll_search_helper_graph_on(&rtyper, &leaf, ptr_lltype.clone())
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
/// residual. The executable address is the `HELPER_FNADDRS` row of the
/// matching layout (`ll_find_char` for `Ptr(STR)`, `ll_find_char_unicode`
/// for `Ptr(UNICODE)`).
pub fn ll_find_char_path(cc: &mut CallControl, ptr_lltype: &LowLevelType) -> CallPath {
    let path = specialized_helper_path(LL_FIND_CHAR, &ll_find_char_arg_types(ptr_lltype));
    if cc.has_function_graph(&path) {
        return path;
    }
    let leaf = specialized_helper_leaf(LL_FIND_CHAR, &ll_find_char_arg_types(ptr_lltype));
    let py = build_ll_find_char_helper_graph(&leaf, ptr_lltype.clone())
        .expect("ll_find_char helper graph");
    debug_assert!(py.func._elidable_function_);
    register_callee_graph(cc, path.clone(), &py.graph.borrow(), true);
    path
}

/// The helper path for `_search_elidable`, minting the elidable residual.
///
/// `look_inside_graph` refuses an elidable function, so the call stays
/// residual. The executable address is the `HELPER_FNADDRS` row of
/// `rstring::_search_elidable` (`Ptr(STR)`).
pub fn search_elidable_path(cc: &mut CallControl) -> CallPath {
    search_elidable_path_on(cc, &STRPTR)
}

/// [`search_elidable_path`] for one string-pointer specialization.
pub fn search_elidable_path_on(cc: &mut CallControl, ptr_lltype: &LowLevelType) -> CallPath {
    let path = specialized_helper_path(SEARCH_ELIDABLE, &ll_search_arg_types(ptr_lltype));
    if cc.has_function_graph(&path) {
        return path;
    }
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = RPythonTyper::new(&ann);
    let leaf = specialized_helper_leaf(SEARCH_ELIDABLE, &ll_search_arg_types(ptr_lltype));
    // Residual of `ll_search`'s callee: `ll_search` already asserted,
    // so `value_can_be_none` is false (`transform_dead_code`).
    let py = crate::rlib::rstring::build_search_elidable_helper_graph(
        &rtyper,
        &leaf,
        ptr_lltype.clone(),
        false,
    )
    .expect("_search_elidable helper graph");
    register_direct_call_callees(cc, &rtyper, &py.graph.borrow());
    register_callee_graph(cc, path.clone(), &py.graph.borrow(), true);
    path
}

fn callee_already_registered(cc: &CallControl, path: &CallPath) -> bool {
    cc.has_function_graph(path) || cc.jitcode_handle(path).is_some() || cc.has_opname_graph(path)
}

fn registered_label_graph_name(cc: &CallControl, path: &CallPath) -> Option<String> {
    if let Some(graph) = cc.function_graphs().get(path) {
        return Some(graph.name.clone());
    }
    if callee_already_registered(cc, path) {
        return Some(path.last_segment().unwrap_or("").to_string());
    }
    None
}

fn stored_graph_arg_lltypes(cc: &CallControl, path: &CallPath) -> Option<Vec<LowLevelType>> {
    let graph = cc.function_graphs().get(path)?;
    let inputs = &graph.block(graph.startblock).inputargs;
    let mut types = Vec::with_capacity(inputs.len());
    for var in inputs {
        types.push(var.concretetype()?);
    }
    Some(types)
}

/// Walk `direct_call` callee graphs depth-first, once per graph object
/// (`call.py CallControl.find_all_graphs` follows the funcptr GRAPH;
/// `annlowlevel.py default_specialize` / `cachedgraph` caches by the
/// specialization key and uses `name__<argtypes>` only as alt_name).
/// Each callee is registered then `add_candidate_graph_if_look_inside`;
/// `policy.look_inside_graph` decides candidacy. Nested callees of a
/// residual are registered for `getcalldescr` only.
///
/// `seen` deduplicates graph identity inside one walk. A label already
/// registered as an effect-only graph is still offered as a candidate
/// when a later walk reaches it with `offer_candidate` — effect analysis
/// (`getcalldescr` / `effectinfo`) never marks a graph as seen for
/// candidacy (`call.py CallControl.find_all_graphs`).
fn register_direct_call_callees(
    cc: &mut CallControl,
    rtyper: &RPythonTyper,
    graph: &crate::flowspace::model::FunctionGraph,
) {
    visit_direct_call_callees(cc, rtyper, graph, true, &mut HashSet::new());
}

/// `call.py CallControl.guess_call_kind` on a `direct_call` funcobj.
/// `find_all_graphs` follows only `'regular'`. A missing graph is residual;
/// a recorded oopspec is builtin. Residual-from-`graphs_from` during this
/// walk is a graph not yet in the candidate set — still registered here.
fn follow_regular_direct_call(cc: &CallControl, func: &_func) -> bool {
    if func.graph.is_none() {
        return false;
    }
    if host_env_jit_builtin_func(func) {
        return false;
    }
    let name = unspecialized_helper_name(&func._name);
    let op = crate::model::SpaceOperation {
        result: None,
        kind: OpKind::Call {
            target: CallTarget::function_path(name.split("::").filter(|s| !s.is_empty())),
            args: Vec::new(),
            result_ty: ValueType::Void,
        },
    };
    match cc.guess_call_kind(&op) {
        CallKind::Builtin | CallKind::Recursive => false,
        CallKind::Regular | CallKind::Residual => true,
    }
}

/// `rlib/jit.py isconstant` / `isvirtual` body (`return NonConstant(False)`).
/// `find_all_graphs` follows only `'regular'`, so this stays in
/// `function_graphs` for `RaiseAnalyzer` and is not a jitcode candidate.
fn register_host_env_jit_body_graph(
    cc: &mut CallControl,
    rtyper: &RPythonTyper,
    func: &_func,
    seen: &mut HashSet<usize>,
) {
    let Some(graph_id) = func.graph else {
        return;
    };
    if !host_env_jit_builtin_func(func) {
        return;
    }
    if !seen.insert(graph_id) {
        return;
    }
    let Some(callee) = graph_ref_for_funcptr_id(rtyper, graph_id) else {
        return;
    };
    let path = CallPath::from_segments(func._name.split("::").filter(|s| !s.is_empty()));
    register_callee_graph(cc, path, &callee.borrow(), false);
}

fn visit_direct_call_callees(
    cc: &mut CallControl,
    rtyper: &RPythonTyper,
    graph: &crate::flowspace::model::FunctionGraph,
    offer_candidate: bool,
    seen: &mut HashSet<usize>,
) {
    specialize_direct_call_funcptr_names(graph);
    bind_ll_list_resize_hint_really_from_graph(cc, graph);
    for func in direct_call_funcobjs(graph) {
        if !follow_regular_direct_call(cc, &func) {
            register_host_env_jit_body_graph(cc, rtyper, &func, seen);
            continue;
        }
        let graph_id = func
            .graph
            .expect("regular direct_call funcptr carries graph");
        if !seen.insert(graph_id) {
            continue;
        }
        let callee = graph_ref_for_funcptr_id(rtyper, graph_id)
            .expect("funcobj.graph lives on translator.graphs");
        let path = callee_path(&func, &callee.borrow());
        if let Some(existing) = registered_label_graph_name(cc, &path) {
            if stored_graph_arg_lltypes(cc, &path).is_some_and(|args| args != func.TYPE.args) {
                panic!(
                    "specialized label {path} already names graph {existing}; colliding graph {}",
                    callee.borrow().name
                );
            }
            if offer_candidate && !cc.is_candidate(&path) {
                register_callee_graph(cc, path.clone(), &callee.borrow(), true);
                let nested_offer = cc.is_candidate(&path);
                visit_direct_call_callees(cc, rtyper, &callee.borrow(), nested_offer, seen);
            }
            continue;
        }
        let leaf = path
            .last_segment()
            .expect("specialized callee path has a leaf")
            .to_string();
        {
            let mut graph = callee.borrow_mut();
            stamp_specialized_graph_name(&mut graph, &leaf);
        }
        specialize_direct_call_funcptr_names(&callee.borrow());
        register_callee_graph(cc, path.clone(), &callee.borrow(), offer_candidate);
        let nested_offer = offer_candidate && cc.is_candidate(&path);
        visit_direct_call_callees(cc, rtyper, &callee.borrow(), nested_offer, seen);
    }
}

/// Share the flowspace graph for Spine-B drain. Blocks stay the same
/// `BlockRef`s; `FunctionGraph` itself is not `Clone`.
fn flowspace_graph_for_opname_drain(
    graph: &crate::flowspace::model::FunctionGraph,
) -> crate::flowspace::model::FunctionGraph {
    crate::flowspace::model::FunctionGraph {
        name: graph.name.clone(),
        startblock: graph.startblock.clone(),
        returnblock: graph.returnblock.clone(),
        exceptblock: graph.exceptblock.clone(),
        tag: graph.tag.clone(),
        func: graph.func.clone(),
        _source: graph._source.clone(),
        _seen_by_the_backend: std::cell::Cell::new(graph._seen_by_the_backend.get()),
    }
}

/// Register one callee as the search effect graph (`getcalldescr` walks
/// the real body). `add_candidate_graph_if_look_inside` decides
/// candidacy. Residual native fnaddr is the `HELPER_FNADDRS` row whose
/// native takes the same STR / UNICODE layout.
///
/// A look-inside helper is registered as an opname graph so drain uses
/// `jtransform_opname::lower_graph` (`LLHelpers.ll_search` `assert
/// value is not None` is `ptr_nonzero`). The lowered twin stays in
/// `function_graphs` for `getcalldescr` / Transformer.
fn register_callee_graph(
    cc: &mut CallControl,
    path: CallPath,
    callee: &crate::flowspace::model::FunctionGraph,
    offer_candidate: bool,
) {
    bind_ll_list_resize_hint_really_from_graph(cc, callee);
    bind_ll_list_resize_hint_really_fnaddr(cc, &path, &graph_arg_lltypes(callee));
    if cc.is_candidate(&path) {
        return;
    }
    if !offer_candidate {
        if callee_already_registered(cc, &path) {
            return;
        }
        let graph = effect_graph_from_search_helper(callee);
        cc.register_function_graph(path, graph);
        return;
    }
    let effect = effect_graph_from_search_helper(callee);
    let mut probe = DefaultJitPolicy::new();
    let look_inside = probe.look_inside_graph(&effect);
    if look_inside {
        cc.register_opname_helper_graph(path.clone(), flowspace_graph_for_opname_drain(callee));
        cc.register_function_graph(path.clone(), lower_graph(callee));
    } else if !callee_already_registered(cc, &path) {
        cc.register_function_graph(path.clone(), effect);
    }
    cc.add_candidate_graph_if_look_inside(path.clone());
    if cc.is_candidate(&path) {
        let _ = cc.get_jitcode(&path);
    } else {
        bind_residual_native_fnaddr(cc, &path, &graph_arg_lltypes(callee));
    }
}

/// `HELPER_FNADDRS` row whose native reads the same `chars` layout as
/// `args`. `None` when this port publishes no matching trampoline
/// (`annlowlevel.py default_specialize` still built a distinct graph).
fn string_element_of_path(dest: &CallPath) -> Option<StringElement> {
    let leaf = dest.last_segment()?;
    if leaf.contains("rpy_unicode") {
        Some(StringElement::Unicode)
    } else if leaf.contains("rpy_string") {
        Some(StringElement::Str)
    } else {
        None
    }
}

fn residual_native_fnaddr_row(dest: &CallPath, args: &[LowLevelType]) -> Option<CallPath> {
    let name = unspecialized_helper_name(dest.last_segment()?);
    let element = string_element_of_path(dest).or_else(|| string_element_of_args(args))?;
    match (name, element) {
        (LL_FIND_CHAR, StringElement::Str) => Some(CallPath::from_segments([
            "lltypesystem",
            "rstr",
            LL_FIND_CHAR,
        ])),
        (LL_FIND_CHAR, StringElement::Unicode) => Some(CallPath::from_segments([
            "lltypesystem",
            "rstr",
            LL_FIND_CHAR_UNICODE,
        ])),
        (SEARCH_ELIDABLE, StringElement::Str) => {
            Some(CallPath::from_segments(["rstring", SEARCH_ELIDABLE]))
        }
        (SEARCH_NORMAL, StringElement::Unicode) => {
            Some(CallPath::from_segments(["rstring", SEARCH_NORMAL]))
        }
        _ => None,
    }
}

fn bind_residual_native_fnaddr(cc: &mut CallControl, dest: &CallPath, args: &[LowLevelType]) {
    match residual_native_fnaddr_row(dest, args) {
        Some(src) => {
            if cc.lookup_function_fnaddr(&src).is_some() {
                cc.alias_function_fnaddr(dest.clone(), &src);
            }
        }
        None => {
            // `jtransform.py handle_residual_call` / `call.py CallControl.getcalldescr`
            // need a real funcptr. A residual with no `HELPER_FNADDRS` family is a
            // translation error; G5 (and a same-name HELPER row) bind `dest` first.
            if cc.lookup_function_fnaddr(dest).is_some() {
                return;
            }
            panic!(
                "residual synthesized helper {dest} has no HELPER_FNADDRS row family (call.py getcalldescr needs a real funcptr)"
            );
        }
    }
}

fn is_ll_list_resize_hint_really_family_leaf(name: &str) -> bool {
    matches!(
        unspecialized_helper_name(name),
        LL_LIST_RESIZE_HINT_REALLY
            | LL_LIST_RESIZE_HINT_REALLY_WRAPPER
            | LL_LIST_RESIZE_HINT_REALLY_TRAMPOLINE
    )
}

/// ITEM of `Ptr(GcStruct("list", items: Ptr(GcArray(ITEM))))`.
fn list_item_lltype_of_ptr(ptr: &LowLevelType) -> Option<LowLevelType> {
    let LowLevelType::Ptr(p) = ptr else {
        return None;
    };
    let PtrTarget::Struct(st) = &p.TO else {
        return None;
    };
    if st._name != "list" {
        return None;
    }
    let items = st._flds.get("items")?;
    let LowLevelType::Ptr(items_p) = items else {
        return None;
    };
    match &items_p.TO {
        PtrTarget::Array(arr) => Some(arr.OF.clone()),
        _ => None,
    }
}

fn ll_list_resize_hint_really_fnaddr_key() -> CallPath {
    CallPath::from_segments(["lltypesystem", "rlist", LL_LIST_RESIZE_HINT_REALLY])
}

fn ll_list_resize_hint_really_f_fnaddr_key() -> CallPath {
    CallPath::from_segments(["lltypesystem", "rlist", LL_LIST_RESIZE_HINT_REALLY_F])
}

/// Alias a synthesized `_ll_list_resize_hint_really*` CallPath onto the
/// `HELPER_FNADDRS` row of its ITEM (`Signed` or `Float`).
fn bind_ll_list_resize_hint_really_fnaddr(
    cc: &mut CallControl,
    dest: &CallPath,
    args: &[LowLevelType],
) {
    let Some(leaf) = dest.last_segment() else {
        return;
    };
    if !is_ll_list_resize_hint_really_family_leaf(leaf) {
        return;
    }
    let item = args.first().and_then(list_item_lltype_of_ptr);
    let Some(item) = item else {
        panic!(
            "_ll_list_resize_hint_really has no native row for item lltype {}",
            args.first()
                .map(|ty| ty.short_name())
                .unwrap_or_else(|| "missing".to_string())
        );
    };
    // One native body per `typeOf(l).TO.items.TO` specialization.
    let src = match item {
        LowLevelType::Signed => ll_list_resize_hint_really_fnaddr_key(),
        LowLevelType::Float => ll_list_resize_hint_really_f_fnaddr_key(),
        other => panic!(
            "_ll_list_resize_hint_really has no native row for item lltype {}",
            other.short_name()
        ),
    };
    if cc.lookup_function_fnaddr(dest).is_some() {
        return;
    }
    if cc.lookup_function_fnaddr(&src).is_some() {
        cc.alias_function_fnaddr(dest.clone(), &src);
    }
}

fn bind_ll_list_resize_hint_really_func(cc: &mut CallControl, func: &_func) {
    let unspecialized = unspecialized_helper_name(&func._name);
    if !is_ll_list_resize_hint_really_family_leaf(unspecialized) {
        return;
    }
    bind_ll_list_resize_hint_really_fnaddr(
        cc,
        &CallPath::from_segments([unspecialized]),
        &func.TYPE.args,
    );
    let specialized = specialized_callee_leaf(func);
    if specialized != unspecialized {
        bind_ll_list_resize_hint_really_fnaddr(
            cc,
            &CallPath::from_segments([specialized.as_str()]),
            &func.TYPE.args,
        );
    }
    if func._name != unspecialized && func._name != specialized {
        bind_ll_list_resize_hint_really_fnaddr(
            cc,
            &CallPath::from_segments([func._name.as_str()]),
            &func.TYPE.args,
        );
    }
}

fn bind_ll_list_resize_hint_really_from_graph(
    cc: &mut CallControl,
    graph: &crate::flowspace::model::FunctionGraph,
) {
    for func in direct_call_funcobjs(graph) {
        bind_ll_list_resize_hint_really_func(cc, &func);
    }
    bind_ll_list_resize_hint_really_fnaddr(
        cc,
        &CallPath::from_segments([graph.name.as_str()]),
        &graph_arg_lltypes(graph),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::{CallKind, is_symbolic_fnaddr};
    use crate::codewriter::codewriter::CodeWriter;
    use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph as FlowGraph, GraphFunc, GraphKey,
        Hlvalue, Link, SpaceOperation,
    };
    use crate::model::{CallTarget, ConcreteType, OpKind, ValueType};
    use crate::policy::JitPolicy;
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};
    use crate::translator::rtyper::lltypesystem::rstr::build_ll_find_helper_graph;
    use crate::translator::rtyper::rtyper::{constant_with_lltype, variable_with_lltype};
    use majit_ir::descr::ExtraEffect;

    fn mint() -> CallControl {
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&helper_fnaddr_path(), 0x1000);
        cc.register_macro_helper_trace_fnaddr(&search_elidable_fnaddr_path(), 0x2000);
        let _ = ll_find_path(&mut cc);
        cc
    }

    fn mint_str_and_unicode() -> CallControl {
        let mut cc = mint();
        cc.register_macro_helper_trace_fnaddr(&search_normal_fnaddr_path(), 0x3000);
        cc.register_macro_helper_trace_fnaddr(&ll_find_char_unicode_fnaddr_path(), 0x4000);
        let _ = ll_find_path_on(&mut cc, &UNICODEPTR);
        cc
    }

    fn str_ll_find_path() -> CallPath {
        specialized_helper_path(LL_FIND, &ll_find_arg_types(&STRPTR))
    }

    fn str_ll_search_path() -> CallPath {
        specialized_helper_path(LL_SEARCH, &ll_search_arg_types(&STRPTR))
    }

    fn str_ll_find_char_path() -> CallPath {
        specialized_helper_path(LL_FIND_CHAR, &ll_find_char_arg_types(&STRPTR))
    }

    fn str_search_elidable_path() -> CallPath {
        specialized_helper_path(SEARCH_ELIDABLE, &ll_search_arg_types(&STRPTR))
    }

    fn unicode_ll_search_path() -> CallPath {
        specialized_helper_path(LL_SEARCH, &ll_search_arg_types(&UNICODEPTR))
    }

    fn unicode_ll_find_char_path() -> CallPath {
        specialized_helper_path(LL_FIND_CHAR, &ll_find_char_arg_types(&UNICODEPTR))
    }

    fn unicode_search_normal_path() -> CallPath {
        specialized_helper_path(SEARCH_NORMAL, &ll_search_arg_types(&UNICODEPTR))
    }

    fn str_ll_find_char_leaf() -> String {
        specialized_helper_leaf(LL_FIND_CHAR, &ll_find_char_arg_types(&STRPTR))
    }

    fn str_search_elidable_leaf() -> String {
        specialized_helper_leaf(SEARCH_ELIDABLE, &ll_search_arg_types(&STRPTR))
    }

    fn transform_registered(cc: &mut CallControl, path: &CallPath) -> crate::model::FunctionGraph {
        let graph = cc
            .function_graphs()
            .get(path)
            .unwrap_or_else(|| panic!("registered {path}"))
            .clone();
        Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(cc)
            .transform(graph.as_ref())
            .graph
    }

    fn call_op() -> crate::model::SpaceOperation {
        let leaf = str_ll_find_char_leaf();
        crate::model::SpaceOperation {
            result: Some(crate::flowspace::model::Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::function_path([leaf]),
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
        let path = str_ll_find_char_path();
        assert!(
            cc.has_function_graph(&path),
            "missing {path}; registered {:?}",
            cc.function_graphs().keys()
        );
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
        assert!(cc.is_elidable(&CallTarget::function_path([str_ll_find_char_leaf()])));
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
        let leaf = str_ll_find_char_leaf();
        let funcptr = functionptr(ft, &leaf, None, Some(leaf.clone()));
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
        let find = str_ll_find_path();
        let search = str_ll_search_path();
        let find_char = str_ll_find_char_path();
        let elidable = str_search_elidable_path();
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
        assert!(cc.is_elidable(&CallTarget::function_path([str_search_elidable_leaf()])));
    }

    /// Effect-only registration of a shared helper (first reached under
    /// residual `_search_elidable`) does not block a later candidate
    /// offer (`call.py CallControl.find_all_graphs`). Reverse order ends
    /// in the same candidate + jitcode state.
    #[test]
    fn shared_helper_effect_only_then_candidate_becomes_candidate() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let elidable = crate::rlib::rstring::build_search_elidable_helper_graph(
            &rtyper,
            SEARCH_ELIDABLE,
            STRPTR.clone(),
            false,
        )
        .expect("_search_elidable");
        let prep = crate::rlib::rstring::build_twoway_preprocess_helper_graph(
            &rtyper,
            "_twoway_preprocess",
            STRPTR.clone(),
        )
        .expect("_twoway_preprocess");
        let path = specialized_helper_path("ll_newlist", &[LowLevelType::Signed]);

        // Nested callees of a residual (`offer_candidate == false`) are
        // registered for `getcalldescr` only (`call.py find_all_graphs`
        // under a non-candidate).
        let mut residual_first = CallControl::new();
        visit_direct_call_callees(
            &mut residual_first,
            &rtyper,
            &elidable.graph.borrow(),
            false,
            &mut HashSet::new(),
        );
        assert!(
            residual_first.has_function_graph(&path),
            "residual walk registers {path}: {:?}",
            residual_first.function_graphs().keys()
        );
        assert!(
            !residual_first.is_candidate(&path),
            "ll_newlist under residual _search_elidable is effect-only"
        );
        let func = direct_call_funcobjs(&prep.graph.borrow())
            .into_iter()
            .find(|func| unspecialized_helper_name(&func._name).ends_with("ll_newlist"))
            .expect("ll_newlist funcobj");
        let start = Block::shared(vec![]);
        let ret = variable_with_lltype("out", LowLevelType::Signed);
        let parent = FlowGraph::with_return_var("caller", start.clone(), Hlvalue::Variable(ret));
        let funcptr = functionptr(
            func.TYPE.clone(),
            &func._name,
            func.graph,
            func._callable.clone(),
        );
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", func.TYPE.result.clone());
        let mut call_args = vec![callee];
        for arg in &func.TYPE.args {
            call_args.push(Hlvalue::Variable(variable_with_lltype("x", arg.clone())));
        }
        start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            call_args,
            Hlvalue::Variable(out),
        ));
        start.closeblock(vec![
            Link::new(
                vec![constant_with_lltype(
                    ConstValue::Int(0),
                    LowLevelType::Signed,
                )],
                Some(parent.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        register_direct_call_callees(&mut residual_first, &rtyper, &parent);
        assert!(
            residual_first.is_candidate(&path),
            "later candidate walk offers the shared helper"
        );
        assert!(
            residual_first.jitcode_handle(&path).is_some(),
            "candidate helper has a jitcode"
        );

        let mut candidate_first = CallControl::new();
        register_direct_call_callees(&mut candidate_first, &rtyper, &parent);
        visit_direct_call_callees(
            &mut candidate_first,
            &rtyper,
            &elidable.graph.borrow(),
            false,
            &mut HashSet::new(),
        );
        assert!(candidate_first.is_candidate(&path));
        assert!(candidate_first.jitcode_handle(&path).is_some());
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
    /// `ll_search` narrowed `value`, so `_search_normal`
    /// `assert value is not None` is cut (`transform_dead_code`).
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
            let leaf = str_ll_find_char_leaf();
            let funcptr = functionptr(ft, &leaf, None, Some(leaf.clone()));
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

        let search_xf = transform_registered(&mut cc, &str_ll_search_path());
        let (search_effects, search_addrs) = residual_effect_and_fnaddrs(&search_xf);
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
        specialize_direct_call_funcptr_names(&find_flow);
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

        let search_xf = transform_registered(&mut cc, &str_ll_search_path());
        let leftover = leftover_call_targets(&search_xf);
        assert!(
            leftover.is_empty(),
            "ll_search leftover Call after rewrite_op_direct_call: {leftover:?}"
        );
        let (search_effects, _) = residual_effect_and_fnaddrs(&search_xf);
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
        let leaf = str_search_elidable_leaf();
        let funcptr = functionptr(ft, &leaf, None, Some(leaf.clone()));
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

        let find_char = str_ll_find_char_path();
        let search_path = str_ll_search_path();
        let elidable = str_search_elidable_path();
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
        assert!(cc.jitcode_handle(&str_ll_find_path()).is_none());

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

        let search_xf = transform_registered(&mut cc, &search_path);
        let (search_effects, search_addrs) = residual_effect_and_fnaddrs(&search_xf);
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
            &str_ll_find_char_leaf(),
            vec![Type::Ref, Type::Int, Type::Int, Type::Int],
        );
        assert_eq!(
            descr.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
        assert!(!descr.extra_info.can_collect);
        let graph = cc
            .function_graphs()
            .get(&str_ll_find_char_path())
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
    /// `ll_search` narrowed `value`, so `_search_normal`
    /// `assert value is not None` is cut (`transform_dead_code`).
    #[test]
    fn search_elidable_calldescr_is_elidable_or_memoryerror() {
        use majit_ir::value::Type;
        let cc = mint();
        let descr = calldescr(
            &cc,
            &str_search_elidable_leaf(),
            vec![Type::Ref, Type::Ref, Type::Int, Type::Int, Type::Int],
        );
        assert_eq!(
            descr.extra_info.extraeffect,
            ExtraEffect::ElidableOrMemoryError
        );
        let elidable = str_search_elidable_path();
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
    /// / `unsafe_loopy_graphs`). Shares the STR+UNICODE `CallControl`
    /// from [`mint_str_and_unicode`]; specializations are keyed by
    /// `name__<argtypes>`.
    #[test]
    fn unicode_ll_find_inlines_ll_search_and_calls_search_normal_residual() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let py = build_ll_find_helper_graph_on(&rtyper, "ll_unicode_find", UNICODEPTR.clone())
            .expect("UNICODE ll_find");
        let mut cc = mint_str_and_unicode();
        register_direct_call_callees(&mut cc, &rtyper, &py.graph.borrow());
        let search = unicode_ll_search_path();
        let normal = unicode_search_normal_path();
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
            policy
                .state()
                .unsafe_loopy_graphs
                .iter()
                .any(|name| unspecialized_helper_name(name) == SEARCH_NORMAL),
            "_search_normal is in unsafe_loopy_graphs: {:?}",
            policy.state().unsafe_loopy_graphs
        );
        assert_eq!(cc.lookup_function_fnaddr(&normal), Some(0x3000));
        let normal_leaf = specialized_helper_leaf(SEARCH_NORMAL, &ll_search_arg_types(&UNICODEPTR));
        assert!(!cc.is_elidable(&CallTarget::function_path([normal_leaf])));

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
        let (find_effects, find_addrs) = residual_effect_and_fnaddrs(&find_xf.graph);
        assert!(
            find_effects
                .iter()
                .any(|effect| *effect == ExtraEffect::ElidableCannotRaise),
            "UNICODE ll_find_char residual is ElidableCannotRaise: {find_effects:?}"
        );
        assert!(
            find_addrs.iter().any(|(addr, symbolic)| *addr == 0x4000
                && !*symbolic
                && !is_symbolic_fnaddr(*addr)),
            "UNICODE ll_find_char fnaddr is the UNICODE native: {find_addrs:?}"
        );

        let callees = direct_call_funcobjs(&py.graph.borrow());
        let search_func = callees
            .iter()
            .find(|func| unspecialized_helper_name(&func._name) == LL_SEARCH)
            .expect("ll_search callee");
        let search_graph =
            graph_ref_for_funcptr_id(&rtyper, search_func.graph.expect("ll_search graph id"))
                .expect("ll_search graph");
        let nested = direct_call_funcobjs(&search_graph.borrow());
        assert_eq!(nested.len(), 1);
        assert_eq!(unspecialized_helper_name(&nested[0]._name), SEARCH_NORMAL);

        let search_xf = transform_registered(&mut cc, &search);
        let leftover = leftover_call_targets(&search_xf);
        assert!(
            leftover.is_empty(),
            "UNICODE ll_search leftover Call: {leftover:?}"
        );
        assert!(
            !search_xf
                .iter_blocks()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::InlineCall { .. })),
            "UNICODE ll_search must not inline _search_normal"
        );
        let (search_effects, search_addrs) = residual_effect_and_fnaddrs(&search_xf);
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

    /// STR and UNICODE `ll_find` families on one `CallControl`.
    /// `default_specialize` / `cachedgraph` `name__<argtypes>` keeps
    /// each specialization's path, and each residual aliases only the
    /// native whose `chars` layout matches.
    #[test]
    fn one_callcontrol_registers_str_and_unicode_ll_find_families() {
        let cc = mint_str_and_unicode();
        let str_search = str_ll_search_path();
        let uni_search = unicode_ll_search_path();
        let str_find_char = str_ll_find_char_path();
        let uni_find_char = unicode_ll_find_char_path();
        let str_normal = specialized_helper_path(SEARCH_NORMAL, &ll_search_arg_types(&STRPTR));
        let uni_normal = unicode_search_normal_path();
        assert_ne!(str_search, uni_search);
        assert_ne!(str_find_char, uni_find_char);
        assert_ne!(str_normal, uni_normal);
        assert!(
            str_search
                .last_segment()
                .is_some_and(|leaf| leaf.contains("rpy_stringPtr")),
            "STR ll_search path names Ptr(STR): {str_search}"
        );
        assert!(
            uni_search
                .last_segment()
                .is_some_and(|leaf| leaf.contains("rpy_unicodePtr")),
            "UNICODE ll_search path names Ptr(UNICODE): {uni_search}"
        );
        assert!(
            str_find_char
                .last_segment()
                .is_some_and(|leaf| leaf.contains("rpy_stringPtr") && leaf.contains("_Char_")),
            "STR ll_find_char path names Ptr(STR) Char: {str_find_char}"
        );
        assert!(
            uni_find_char
                .last_segment()
                .is_some_and(|leaf| leaf.contains("rpy_unicodePtr") && leaf.contains("UniChar")),
            "UNICODE ll_find_char path names Ptr(UNICODE) UniChar: {uni_find_char}"
        );
        assert!(cc.has_function_graph(&str_search));
        assert!(cc.has_function_graph(&uni_search));
        assert!(cc.has_function_graph(&str_find_char));
        assert!(cc.has_function_graph(&uni_find_char));
        assert!(cc.has_function_graph(&uni_normal));
        assert_eq!(cc.lookup_function_fnaddr(&str_find_char), Some(0x1000));
        assert_eq!(cc.lookup_function_fnaddr(&uni_find_char), Some(0x4000));
        assert_eq!(
            cc.lookup_function_fnaddr(&str_search_elidable_path()),
            Some(0x2000)
        );
        assert_eq!(cc.lookup_function_fnaddr(&uni_normal), Some(0x3000));
    }

    /// Residual bind maps a specialization to the native of the same
    /// layout and leaves a gap unbound rather than aliasing the other
    /// element's row.
    #[test]
    fn residual_native_fnaddr_row_matches_element_layout() {
        let str_find_char_args = ll_find_char_arg_types(&STRPTR);
        let uni_find_char_args = ll_find_char_arg_types(&UNICODEPTR);
        let str_search_args = ll_search_arg_types(&STRPTR);
        let uni_search_args = ll_search_arg_types(&UNICODEPTR);
        assert_eq!(
            residual_native_fnaddr_row(&str_ll_find_char_path(), &str_find_char_args),
            Some(CallPath::from_segments([
                "lltypesystem",
                "rstr",
                LL_FIND_CHAR
            ]))
        );
        assert_eq!(
            residual_native_fnaddr_row(&unicode_ll_find_char_path(), &uni_find_char_args),
            Some(CallPath::from_segments([
                "lltypesystem",
                "rstr",
                LL_FIND_CHAR_UNICODE
            ]))
        );
        assert_eq!(
            residual_native_fnaddr_row(&str_search_elidable_path(), &str_search_args),
            Some(CallPath::from_segments(["rstring", SEARCH_ELIDABLE]))
        );
        assert_eq!(
            residual_native_fnaddr_row(&unicode_search_normal_path(), &uni_search_args),
            Some(CallPath::from_segments(["rstring", SEARCH_NORMAL]))
        );
        assert_eq!(
            residual_native_fnaddr_row(
                &specialized_helper_path(SEARCH_ELIDABLE, &uni_search_args),
                &uni_search_args
            ),
            None,
            "UNICODE _search_elidable has no native row"
        );
        assert_eq!(
            residual_native_fnaddr_row(
                &specialized_helper_path(SEARCH_NORMAL, &str_search_args),
                &str_search_args
            ),
            None,
            "STR _search_normal has no native row"
        );
    }

    fn residual_named_helper(name: &str, arg: LowLevelType) -> FlowGraph {
        let x = variable_with_lltype("x", arg);
        let start = Block::shared(vec![Hlvalue::Variable(x)]);
        let ret = variable_with_lltype("out", LowLevelType::Signed);
        let mut graph =
            FlowGraph::with_return_var(name.to_string(), start.clone(), Hlvalue::Variable(ret));
        start.closeblock(vec![
            Link::new(
                vec![constant_with_lltype(
                    ConstValue::Int(0),
                    LowLevelType::Signed,
                )],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let mut func = GraphFunc::new(name, Constant::new(ConstValue::Dict(Default::default())));
        func._elidable_function_ = true;
        graph.func = Some(func);
        graph
    }

    /// `call.py find_all_graphs` follows graph identity. Two distinct
    /// graphs that share `func._name` and would have spelled the same
    /// `arrayPtr` label (STR vs UNICODE `chars`) both register, with
    /// `array_Char` / `array_UniChar` alt_names.
    #[test]
    fn callcontrol_registers_distinct_graphs_that_share_func_name() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let char_ptr = chars_array_ptr_lltype_from_strptr(&STRPTR).expect("STR chars");
        let uni_ptr = chars_array_ptr_lltype_from_strptr(&UNICODEPTR).expect("UNICODE chars");
        let char_leaf = specialized_helper_leaf("helper", &[char_ptr.clone()]);
        let uni_leaf = specialized_helper_leaf("helper", &[uni_ptr.clone()]);
        assert_ne!(char_leaf, uni_leaf);
        assert!(
            char_leaf.contains("array_Char"),
            "Char array label: {char_leaf}"
        );
        assert!(
            uni_leaf.contains("array_UniChar"),
            "UniChar array label: {uni_leaf}"
        );

        let char_graph = residual_named_helper("helper", char_ptr.clone()).into_ref();
        let uni_graph = residual_named_helper("helper", uni_ptr.clone()).into_ref();
        ann.translator
            .graphs
            .borrow_mut()
            .extend([char_graph.clone(), uni_graph.clone()]);
        let char_id = GraphKey::of(&char_graph).as_usize();
        let uni_id = GraphKey::of(&uni_graph).as_usize();
        assert_ne!(char_id, uni_id);

        let start = Block::shared(vec![]);
        let ret = variable_with_lltype("out", LowLevelType::Signed);
        let parent = FlowGraph::with_return_var("caller", start.clone(), Hlvalue::Variable(ret));
        for (arg, graph_id) in [(char_ptr, char_id), (uni_ptr, uni_id)] {
            let ft = FuncType {
                args: vec![arg.clone()],
                result: LowLevelType::Signed,
            };
            let funcptr = functionptr(ft, "helper", Some(graph_id), Some("helper".into()));
            let callee =
                constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
            let out = variable_with_lltype("out", LowLevelType::Signed);
            start.borrow_mut().operations.push(SpaceOperation::new(
                "direct_call",
                vec![callee, Hlvalue::Variable(variable_with_lltype("x", arg))],
                Hlvalue::Variable(out),
            ));
        }
        start.closeblock(vec![
            Link::new(
                vec![constant_with_lltype(
                    ConstValue::Int(0),
                    LowLevelType::Signed,
                )],
                Some(parent.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut cc = CallControl::new();
        // The fixture stands for a translated function; getfunctionptr gives it an address.
        cc.register_macro_helper_trace_fnaddr(&format!("fixture::{char_leaf}"), 0x7000);
        cc.register_macro_helper_trace_fnaddr(&format!("fixture::{uni_leaf}"), 0x7008);
        assert_eq!(
            cc.lookup_function_fnaddr(&CallPath::from_segments([char_leaf.as_str()])),
            Some(0x7000)
        );
        register_direct_call_callees(&mut cc, &rtyper, &parent);
        let char_path = CallPath::from_segments([char_leaf.as_str()]);
        let uni_path = CallPath::from_segments([uni_leaf.as_str()]);
        assert_ne!(char_path, uni_path);
        assert!(
            cc.has_function_graph(&char_path),
            "missing {char_path}; registered {:?}",
            cc.function_graphs().keys()
        );
        assert!(
            cc.has_function_graph(&uni_path),
            "missing {uni_path}; registered {:?}",
            cc.function_graphs().keys()
        );
        let char_stored = cc
            .function_graphs()
            .get(&char_path)
            .expect("Char helper graph");
        let uni_stored = cc
            .function_graphs()
            .get(&uni_path)
            .expect("UniChar helper graph");
        assert!(!std::rc::Rc::ptr_eq(&char_stored, &uni_stored));
    }

    /// `_ll_list_resize_ge` is an ordinary traced graph (`rlist.py
    /// _ll_list_resize_ge`). The strfind walk offers it as a candidate;
    /// `lower_graph` transduces `jit_conditional_call` instead of demoting.
    #[test]
    fn ll_list_resize_ge_registers_as_a_candidate() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let prep = crate::rlib::rstring::build_twoway_preprocess_helper_graph(
            &rtyper,
            "_twoway_preprocess",
            STRPTR.clone(),
        )
        .expect("_twoway_preprocess");
        let graphs = ann.translator.graphs.borrow();
        let ge = graphs
            .iter()
            .find(|g| g.borrow().name == "_ll_list_resize_ge")
            .expect("_twoway_preprocess mints _ll_list_resize_ge")
            .clone();
        drop(graphs);
        let mut cc = CallControl::new();
        // rstring.py _twoway_preprocess is loopy: policy.py look_inside_graph rejects it, so its
        // callees register effect-only.
        visit_direct_call_callees(
            &mut cc,
            &rtyper,
            &prep.graph.borrow(),
            false,
            &mut HashSet::new(),
        );
        let path = CallPath::from_segments(["_ll_list_resize_ge"]);
        register_callee_graph(&mut cc, path.clone(), &ge.borrow(), true);
        assert!(
            cc.is_candidate(&path),
            "_ll_list_resize_ge must be a candidate; registered {:?}",
            cc.function_graphs().keys()
        );
    }

    /// `FunctionDesc.cachedgraph` labels the helper graph; the
    /// `rlib/jit.py isconstant` funcobj keeps its unspecialized name so
    /// `rewrite_op_direct_call` still hits `jit.isconstant`.
    #[test]
    fn specialize_keeps_isconstant_funcptr_name() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let prep = crate::rlib::rstring::build_twoway_preprocess_helper_graph(
            &rtyper,
            "_twoway_preprocess",
            STRPTR.clone(),
        )
        .expect("_twoway_preprocess");
        let graphs = ann.translator.graphs.borrow();
        let ge = graphs
            .iter()
            .find(|g| g.borrow().name == "_ll_list_resize_ge")
            .expect("_twoway_preprocess mints _ll_list_resize_ge")
            .clone();
        drop(graphs);
        let mut cc = CallControl::new();
        // rstring.py _twoway_preprocess is loopy: policy.py look_inside_graph rejects it, so its
        // callees register effect-only.
        visit_direct_call_callees(
            &mut cc,
            &rtyper,
            &prep.graph.borrow(),
            false,
            &mut HashSet::new(),
        );
        let path = CallPath::from_segments(["_ll_list_resize_ge"]);
        register_callee_graph(&mut cc, path, &ge.borrow(), true);

        let mut isconstant_names = Vec::new();
        for graph in ann.translator.graphs.borrow().iter() {
            for func in direct_call_funcobjs(&graph.borrow()) {
                let unspecialized = unspecialized_helper_name(&func._name);
                if unspecialized.rsplit("::").next() == Some("isconstant") {
                    isconstant_names.push(func._name.clone());
                }
            }
        }
        assert!(
            isconstant_names
                .iter()
                .any(|name| name == "majit_rlib::jit::isconstant"),
            "isconstant funcptr identity stays unspecialized, got {isconstant_names:?}"
        );
        assert!(
            isconstant_names.iter().all(|name| !name.contains("__")),
            "cachedgraph alt_name is not the isconstant funcobj, got {isconstant_names:?}"
        );

        let wrapper = ann
            .translator
            .graphs
            .borrow()
            .iter()
            .find(|g| unspecialized_helper_name(&g.borrow().name) == "_ll_alloc_and_set_isconstant")
            .expect("_ll_alloc_and_set_isconstant family graph")
            .clone();
        let model = lower_graph(&wrapper.borrow());
        let calls: Vec<_> = model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => target.path_segments(),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segs| segs.as_slice() == ["majit_rlib", "jit", "isconstant"]),
            "lower_graph still emits Call to jit.isconstant, got {calls:?}"
        );

        cc.mark_oopspec(
            CallPath::from_segments(["majit_rlib", "jit", "isconstant"]),
            "jit.isconstant".to_string(),
        );
        let rewritten = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&model);
        assert!(
            rewritten
                .graph
                .iter_blocks()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(op.kind, OpKind::IsConstant { .. })),
            "rewrite_op_direct_call still rewrites jit.isconstant"
        );

        let isc_path = CallPath::from_segments(["majit_rlib", "jit", "isconstant"]);
        assert!(
            cc.has_function_graph(&isc_path),
            "isconstant body graph is in function_graphs for RaiseAnalyzer"
        );
        assert!(
            !cc.is_candidate(&isc_path),
            "find_all_graphs follows only regular; isconstant is Builtin"
        );
    }

    /// `_ll_list_resize_ge`'s residual CondCall of the look_inside_iff
    /// wrapper aliases the Signed `_ll_list_resize_hint_really` HELPER
    /// row, as does the `dont_look_inside` trampoline.
    #[test]
    fn ll_list_resize_ge_conditional_call_funcptr_is_executable() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let prep = crate::rlib::rstring::build_twoway_preprocess_helper_graph(
            &rtyper,
            "_twoway_preprocess",
            STRPTR.clone(),
        )
        .expect("_twoway_preprocess");
        let graphs = ann.translator.graphs.borrow();
        let ge = graphs
            .iter()
            .find(|g| g.borrow().name == "_ll_list_resize_ge")
            .expect("_twoway_preprocess mints _ll_list_resize_ge")
            .clone();
        drop(graphs);
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&ll_list_resize_hint_really_fnaddr_path(), 0x5000);
        // rstring.py _twoway_preprocess is loopy: policy.py look_inside_graph rejects it, so its
        // callees register effect-only.
        visit_direct_call_callees(
            &mut cc,
            &rtyper,
            &prep.graph.borrow(),
            false,
            &mut HashSet::new(),
        );
        let path = CallPath::from_segments(["_ll_list_resize_ge"]);
        register_callee_graph(&mut cc, path.clone(), &ge.borrow(), true);
        assert!(
            cc.is_candidate(&path),
            "_ll_list_resize_ge must be a candidate; registered {:?}",
            cc.function_graphs().keys()
        );

        let wrapper = CallPath::from_segments([LL_LIST_RESIZE_HINT_REALLY_WRAPPER]);
        assert_eq!(cc.lookup_function_fnaddr(&wrapper), Some(0x5000));
        let trampoline = CallPath::from_segments([LL_LIST_RESIZE_HINT_REALLY_TRAMPOLINE]);
        assert_eq!(cc.lookup_function_fnaddr(&trampoline), Some(0x5000));

        for func in direct_call_funcobjs(&ge.borrow()) {
            if !is_ll_list_resize_hint_really_family_leaf(&func._name) {
                continue;
            }
            let spec = CallPath::from_segments([specialized_callee_leaf(&func).as_str()]);
            assert_eq!(
                cc.lookup_function_fnaddr(&spec),
                Some(0x5000),
                "specialized {spec} must alias the HELPER row"
            );
        }

        let model = lower_graph(&ge.borrow());
        let fn_consts: Vec<_> = model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } if args.is_empty() => {
                    let segs = target.path_segments()?;
                    if segs.first() == Some(&crate::model::FN_CONST_HEAD) {
                        Some(target.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        assert!(
            !fn_consts.is_empty(),
            "lower_graph emits __fn_const of the CondCall target"
        );
        for funcptr in &fn_consts {
            let binding = cc.fnaddr_binding_for_target(funcptr);
            assert!(
                !binding.symbolic,
                "ConditionalCall funcptr must be executable: {funcptr:?}"
            );
            assert!(!is_symbolic_fnaddr(binding.addr));
            assert_eq!(binding.addr, 0x5000);
        }

        cc.mark_oopspec(
            CallPath::from_segments(["majit_rlib", "jit", "conditional_call3"]),
            "jit.conditional_call".to_string(),
        );
        let transformed = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&model);
        let funcptrs: Vec<_> = transformed
            .graph
            .iter_blocks()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::ConditionalCall { funcptr, .. } => Some(funcptr.clone()),
                _ => None,
            })
            .collect();
        assert!(
            !funcptrs.is_empty(),
            "_ll_list_resize_ge must emit ConditionalCall"
        );
        for funcptr in &funcptrs {
            let binding = cc.fnaddr_binding_for_target(funcptr);
            assert!(
                !binding.symbolic,
                "ConditionalCall funcptr must be executable: {funcptr:?}"
            );
            assert!(!is_symbolic_fnaddr(binding.addr));
            assert_eq!(binding.addr, 0x5000);
        }
    }

    #[test]
    #[should_panic(expected = "_ll_list_resize_hint_really has no native row")]
    fn ll_list_resize_hint_really_char_item_has_no_native_row() {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr, Struct};

        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Char)),
        }));
        let list = Struct::gc(
            "list",
            vec![
                ("length".to_string(), LowLevelType::Signed),
                ("items".to_string(), items_ptr),
            ],
        );
        let ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(list),
        }));
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&ll_list_resize_hint_really_fnaddr_path(), 0x5000);
        bind_ll_list_resize_hint_really_fnaddr(
            &mut cc,
            &CallPath::from_segments([LL_LIST_RESIZE_HINT_REALLY_WRAPPER]),
            &[ptr, LowLevelType::Signed, LowLevelType::Bool],
        );
    }

    fn signed_list_ptr() -> LowLevelType {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr, Struct};

        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Signed)),
        }));
        let list = Struct::gc(
            "list",
            vec![
                ("length".to_string(), LowLevelType::Signed),
                ("items".to_string(), items_ptr),
            ],
        );
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(list),
        }))
    }

    fn float_list_ptr() -> LowLevelType {
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr, Struct};

        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Float)),
        }));
        let list = Struct::gc(
            "list",
            vec![
                ("length".to_string(), LowLevelType::Signed),
                ("items".to_string(), items_ptr),
            ],
        );
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(list),
        }))
    }

    #[test]
    fn ll_list_resize_hint_really_float_item_binds_float_row() {
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&ll_list_resize_hint_really_fnaddr_path(), 0x5000);
        cc.register_macro_helper_trace_fnaddr(&ll_list_resize_hint_really_f_fnaddr_path(), 0x6000);
        let dest = CallPath::from_segments([LL_LIST_RESIZE_HINT_REALLY_WRAPPER]);
        bind_ll_list_resize_hint_really_fnaddr(
            &mut cc,
            &dest,
            &[float_list_ptr(), LowLevelType::Signed, LowLevelType::Bool],
        );
        assert_eq!(cc.lookup_function_fnaddr(&dest), Some(0x6000));
    }

    /// A residual synthesized helper with no `HELPER_FNADDRS` family is a
    /// translation error (`call.py CallControl.getcalldescr` needs a real
    /// funcptr).
    #[test]
    #[should_panic(expected = "has no HELPER_FNADDRS row family")]
    fn residual_synthesized_helper_without_row_family_is_a_translation_error() {
        let mut graph = residual_named_helper("no_helper_fnaddrs_row_family", LowLevelType::Signed);
        if let Some(func) = graph.func.as_mut() {
            func._elidable_function_ = false;
            func._jit_look_inside_ = Some(false);
        }
        let mut cc = CallControl::new();
        register_callee_graph(
            &mut cc,
            CallPath::from_segments(["no_helper_fnaddrs_row_family"]),
            &graph,
            true,
        );
    }

    /// G5 aliases the look_inside_iff wrapper onto the Signed HELPER row
    /// before `bind_residual_native_fnaddr` checks for a row family.
    #[test]
    fn ll_list_resize_hint_really_wrapper_is_bound_before_residual_check() {
        let mut graph =
            residual_named_helper(LL_LIST_RESIZE_HINT_REALLY_WRAPPER, signed_list_ptr());
        if let Some(func) = graph.func.as_mut() {
            func._elidable_function_ = false;
            func._jit_look_inside_ = Some(false);
        }
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr(&ll_list_resize_hint_really_fnaddr_path(), 0x5000);
        let path = CallPath::from_segments([LL_LIST_RESIZE_HINT_REALLY_WRAPPER]);
        register_callee_graph(&mut cc, path.clone(), &graph, true);
        assert!(
            !cc.is_candidate(&path),
            "look_inside_graph refuses `_jit_look_inside_ = False`"
        );
        assert_eq!(cc.lookup_function_fnaddr(&path), Some(0x5000));
    }
}
