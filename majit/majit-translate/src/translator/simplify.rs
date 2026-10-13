//! RPython `rpython/translator/simplify.py` — graph-level transformations
//! invoked from `RPythonAnnotator.simplify()` (annrpython.py)
//! and `simplify_graph` (simplify.py+).
//!
//! Only the subset reachable from the annotator port lands here
//! initially; downstream simplification passes arrive with the
//! rtyper port.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphKey, GraphRef,
    HOST_ENV, Hlvalue, HostObject, LinkRef, SpaceOperation, Variable, checkgraph, mkentrymap,
};
use crate::translator::backendopt::ssa::DataFlowFamilyBuilder;
use crate::translator::rtyper::lltypesystem::{lloperation, lltype};
use crate::translator::translator::TranslationContext;

/// RPython `eliminate_empty_blocks(graph)` (simplify.py).
///
/// ```python
/// def eliminate_empty_blocks(graph):
///     """Eliminate basic blocks that do not contain any operations.
///     When this happens, we need to replace the preceeding link with
///     the following link.  Arguments of the links should be updated."""
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target, (
///                 "the graph contains an empty infinite loop")
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// The Rust port mirrors upstream's single-threaded mutation pattern:
/// walk each snapshot link, collapse chains of empty successor
/// blocks, and rewrite the link's `args` / `target` in place.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn eliminate_empty_blocks(graph: &FunctionGraph) {
    // upstream: `for link in list(graph.iterlinks()):`
    for link_ref in graph.iterlinks() {
        loop {
            // upstream: `while not link.target.operations:`
            let (is_empty, has_switch, has_exits, next_exit, target_rc) = {
                let link = link_ref.borrow();
                let Some(target) = link.target.as_ref() else {
                    break;
                };
                let target_b = target.borrow();
                let is_empty = target_b.operations.is_empty();
                let has_switch = target_b.exitswitch.is_some();
                let has_exits = !target_b.exits.is_empty();
                let next_exit = target_b.exits.first().cloned();
                (is_empty, has_switch, has_exits, next_exit, target.clone())
            };
            if !is_empty {
                break;
            }
            if has_switch {
                // upstream: `if block1.exitswitch is not None: break`.
                break;
            }
            if !has_exits {
                // upstream: `if not block1.exits: break`.
                break;
            }
            // upstream: `exit = block1.exits[0]`.
            let Some(exit_ref) = next_exit else {
                break;
            };

            // upstream: `assert block1 is not exit.target, ...`.
            {
                let exit = exit_ref.borrow();
                if let Some(exit_target) = &exit.target {
                    assert!(
                        !std::rc::Rc::ptr_eq(exit_target, &target_rc),
                        "the graph contains an empty infinite loop"
                    );
                }
            }

            // Build `subst = dict(zip(block1.inputargs, link.args))` —
            // the Link carries `Vec<LinkArg> = Vec<Option<Hlvalue>>`,
            // so we keep entries with concrete values only and skip
            // the transient-merge `None` slots.
            let subst: HashMap<Variable, Hlvalue> = {
                let target_b = target_rc.borrow();
                let link = link_ref.borrow();
                target_b
                    .inputargs
                    .iter()
                    .zip(link.args.iter())
                    .filter_map(|(formal, actual)| {
                        let Hlvalue::Variable(v) = formal else {
                            return None;
                        };
                        actual.as_ref().map(|a| (v.clone(), a.clone()))
                    })
                    .collect()
            };

            // upstream: `link.args = [v.replace(subst) for v in exit.args]`.
            let new_args: Vec<Option<Hlvalue>> = {
                let exit = exit_ref.borrow();
                exit.args
                    .iter()
                    .map(|arg| {
                        arg.as_ref().map(|v| match v {
                            Hlvalue::Variable(var) => subst
                                .get(var)
                                .cloned()
                                .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
                            other => other.clone(),
                        })
                    })
                    .collect()
            };

            // upstream: `link.target = exit.target`.
            let new_target = exit_ref.borrow().target.clone();

            {
                let mut link = link_ref.borrow_mut();
                link.args = new_args;
                link.target = new_target;
            }
            // upstream: the while-loop re-evaluates on the rewritten
            // link so chains of empty blocks collapse in one pass.
        }
    }
}

/// Thin wrapper around `Hlvalue::replace` kept for back-compat with
/// this file's internal call shape — `Hlvalue::replace` is now
/// polymorphic (`HashMap<Variable, Hlvalue>`) in line with upstream.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn rename_hl(v: &Hlvalue, mapping: &HashMap<Variable, Hlvalue>) -> Hlvalue {
    match v {
        Hlvalue::Variable(var) => mapping
            .get(var)
            .cloned()
            .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
        Hlvalue::Constant(_) => v.clone(),
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn rename_op(op: &SpaceOperation, mapping: &HashMap<Variable, Hlvalue>) -> SpaceOperation {
    // upstream: `op = op.replace(renaming)`.
    let mut new_op = SpaceOperation {
        opname: op.opname.clone(),
        args: op.args.iter().map(|a| rename_hl(a, mapping)).collect(),
        result: rename_hl(&op.result, mapping),
        offset: op.offset,
    };
    // upstream: indirect_call with a Constant callee is rewritten back
    // to direct_call (simplify.py:293-297).
    if new_op.opname == "indirect_call" && matches!(new_op.args.first(), Some(Hlvalue::Constant(_)))
    {
        assert!(
            matches!(new_op.args.last(), Some(Hlvalue::Constant(_))),
            "indirect_call's trailing argument must be a Constant"
        );
        new_op.args.pop();
        new_op.opname = "direct_call".to_string();
    }
    new_op
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn rename_link_args(link: &LinkRef, mapping: &HashMap<Variable, Hlvalue>) -> LinkRef {
    // upstream's `link.replace(mapping)` path — rebuild a fresh Link
    // with renamed args / last_exception / last_exc_value.
    let src = link.borrow();
    let mut newlink = crate::flowspace::model::Link::new_mergeable(
        src.args
            .iter()
            .map(|a| a.as_ref().map(|v| rename_hl(v, mapping)))
            .collect(),
        src.target.clone(),
        src.exitcase.clone(),
    );
    newlink.prevblock = src.prevblock.clone();
    newlink.llexitcase = src.llexitcase.clone();
    newlink.last_exception = src.last_exception.as_ref().map(|v| rename_hl(v, mapping));
    newlink.last_exc_value = src.last_exc_value.as_ref().map(|v| rename_hl(v, mapping));
    newlink.into_ref()
}

/// RPython `replace_exitswitch_by_constant(block, const)`
/// (simplify.py:36-48).
///
/// ```python
/// def replace_exitswitch_by_constant(block, const):
///     newexits = [link for link in block.exits
///                      if link.exitcase == const.value]
///     if len(newexits) == 0:
///         newexits = [link for link in block.exits
///                      if link.exitcase == 'default']
///     assert len(newexits) == 1
///     newexits[0].exitcase = None
///     if hasattr(newexits[0], 'llexitcase'):
///         newexits[0].llexitcase = None
///     block.exitswitch = None
///     block.recloseblock(*newexits)
///     return newexits
/// ```
pub fn replace_exitswitch_by_constant(block: &BlockRef, const_: &Constant) -> Vec<LinkRef> {
    // upstream: `if link.exitcase == const.value` — our `Link.exitcase`
    // is `Option<Hlvalue>` wrapping a Constant carrying `ConstValue`;
    // compare on the inner `ConstValue`.
    let cases_eq = |link: &LinkRef| match &link.borrow().exitcase {
        Some(Hlvalue::Constant(c)) => c.value == const_.value,
        _ => false,
    };
    let default_case = |link: &LinkRef| {
        matches!(
            &link.borrow().exitcase,
            Some(Hlvalue::Constant(c)) if c.value.string_eq("default")
        )
    };

    let exits_snapshot: Vec<LinkRef> = block.borrow().exits.to_vec();
    let mut newexits: Vec<LinkRef> = exits_snapshot
        .iter()
        .filter(|l| cases_eq(l))
        .cloned()
        .collect();
    if newexits.is_empty() {
        newexits = exits_snapshot
            .iter()
            .filter(|l| default_case(l))
            .cloned()
            .collect();
    }
    assert_eq!(
        newexits.len(),
        1,
        "replace_exitswitch_by_constant: no unique surviving exit"
    );
    {
        let mut l = newexits[0].borrow_mut();
        l.exitcase = None;
        l.llexitcase = None;
    }
    block.borrow_mut().exitswitch = None;
    block.recloseblock(newexits.clone());
    newexits
}

/// RPython `CanRemove = {...}` (simplify.py).
///
/// ```python
/// CanRemove = {}
/// for _op in '''
///         newtuple newlist newdict bool
///         is_ id type issubtype isinstance repr str len hash getattr getitem
///         pos neg abs hex oct ord invert add sub mul
///         truediv floordiv div mod divmod pow lshift rshift and_ or_
///         xor int float long lt le eq ne gt ge cmp coerce contains
///         iter get'''.split():
///     CanRemove[_op] = True
/// from rpython.rtyper.lltypesystem.lloperation import enum_ops_without_sideeffects
/// for _op in enum_ops_without_sideeffects():
///     CanRemove[_op] = True
/// ```
///
/// The Rust port populates the high-level opname list verbatim and
/// then runs the same `enum_ops_without_sideeffects()` extension over
/// the ported `LL_OPERATIONS` table.
fn can_remove_opnames() -> &'static HashSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        // upstream: high-level opname list (simplify.py:406-413).
        let mut set: HashSet<&'static str> = [
            "newtuple",
            "newlist",
            "newdict",
            "bool",
            "is_",
            "id",
            "type",
            "issubtype",
            "isinstance",
            "repr",
            "str",
            "len",
            "hash",
            "getattr",
            "getitem",
            "pos",
            "neg",
            "abs",
            "hex",
            "oct",
            "ord",
            "invert",
            "add",
            "sub",
            "mul",
            "truediv",
            "floordiv",
            "div",
            "mod",
            "divmod",
            "pow",
            "lshift",
            "rshift",
            "and_",
            "or_",
            "xor",
            "int",
            "float",
            "long",
            "lt",
            "le",
            "eq",
            "ne",
            "gt",
            "ge",
            "cmp",
            "coerce",
            "contains",
            "iter",
            "get",
        ]
        .into_iter()
        .collect();
        // upstream: `for _op in enum_ops_without_sideeffects(): CanRemove[_op] = True`
        // (simplify.py:414-416).
        set.extend(lloperation::enum_ops_without_sideeffects(false));
        set
    })
}

/// RPython `CanRemoveBuiltins = {hasattr: True}` (simplify.py).
///
/// Compared by HostObject identity against `HOST_ENV.lookup_builtin("hasattr")`.
fn can_remove_builtins() -> Vec<HostObject> {
    let mut v = Vec::new();
    if let Some(h) = HOST_ENV.lookup_builtin("hasattr") {
        v.push(h);
    }
    v
}

/// RPython `get_graph(arg, translator)` (simplify.py).
///
/// ```python
/// def get_graph(arg, translator):
///     if isinstance(arg, Variable):
///         return None
///     f = arg.value
///     if not isinstance(f, lltype._ptr):
///         return None
///     try:
///         funcobj = f._obj
///     except lltype.DelayedPointer:
///         return None
///     try:
///         return funcobj.graph
///     except AttributeError:
///         return None
/// ```
///
pub fn get_graph(arg: &Hlvalue, translator: &TranslationContext) -> Option<GraphRef> {
    // upstream: `if isinstance(arg, Variable): return None`.
    let Hlvalue::Constant(c) = arg else {
        return None;
    };
    // upstream: `if not isinstance(f, lltype._ptr): return None`.
    let ConstValue::LLPtr(f) = &c.value else {
        return None;
    };
    // upstream: `try: funcobj = f._obj except lltype.DelayedPointer: return None`.
    let funcobj = match f._obj() {
        Ok(lltype::_ptr_obj::Func(funcobj)) => funcobj,
        Ok(lltype::_ptr_obj::Struct(_)) => return None,
        Ok(lltype::_ptr_obj::Array(_)) => return None,
        Ok(lltype::_ptr_obj::Opaque(_)) => return None,
        Ok(lltype::_ptr_obj::Wref(_)) => return None,
        Ok(lltype::_ptr_obj::Subarray(_)) => return None,
        Ok(lltype::_ptr_obj::ArrayLenRef(_)) => return None,
        Ok(lltype::_ptr_obj::EndMarker(_)) => return None,
        Ok(lltype::_ptr_obj::IntCast(_)) => return None,
        Err(lltype::DelayedPointer) => return None,
    };
    // upstream: `try: return funcobj.graph except AttributeError: return None`.
    let graph_key = funcobj.graph?;
    translator
        .graphs
        .borrow()
        .iter()
        .find(|graph| GraphKey::of(graph).as_usize() == graph_key)
        .cloned()
}

/// RPython `op_has_side_effects(op)` (simplify.py).
///
/// ```python
/// def op_has_side_effects(op):
///     return lloperation.LL_OPERATIONS[op.opname].sideeffects
/// ```
///
/// Upstream calls this only on ll-lowered graphs where every opname
/// is present in `LL_OPERATIONS`. The Rust port is reachable from
/// `has_no_side_effects` via the newly-ported `buildrtyper()` /
/// `rtyper_already_seen()` chain before the rtyper actually lowers
/// anything (`RPythonTyper` is a skeleton carrying only the
/// `already_seen` set right now). Missing opnames — high-level
/// `newlist` / `contains` / `simple_call`, for instance — therefore
/// still flow through here and must be treated **conservatively
/// as having side effects** instead of panicking.
pub fn op_has_side_effects(op: &SpaceOperation) -> bool {
    lloperation::ll_operations()
        .get(op.opname.as_str())
        .map(|entry| entry.sideeffects)
        // Conservative fallback: unknown opname → assume it has side
        // effects so dead-op elimination keeps it live. Mirrors the
        // upstream contract (LL_OPERATIONS is complete after ll
        // lowering) without introducing a panic before lowering
        // lands.
        .unwrap_or(true)
}

/// RPython `rec_op_has_side_effects(translator, op, seen=None)`
/// (simplify.py:377-392).
///
/// ```python
/// def rec_op_has_side_effects(translator, op, seen=None):
///     if op.opname == "direct_call":
///         g = get_graph(op.args[0], translator)
///         if g is None:
///             return True
///         if not has_no_side_effects(translator, g, seen):
///             return True
///     elif op.opname == "indirect_call":
///         graphs = op.args[-1].value
///         if graphs is None:
///             return True
///         for g in graphs:
///             if not has_no_side_effects(translator, g, seen):
///                 return True
///     else:
///         return op_has_side_effects(op)
/// ```
///
pub fn rec_op_has_side_effects(
    translator: &TranslationContext,
    op: &SpaceOperation,
    seen: Option<&HashSet<GraphKeyForSeen>>,
) -> bool {
    if op.opname == "direct_call" {
        let Some(callee_arg) = op.args.first() else {
            return true;
        };
        let g = get_graph(callee_arg, translator);
        let Some(g) = g else {
            return true;
        };
        if !has_no_side_effects(translator, &g, seen) {
            return true;
        }
        false
    } else if op.opname == "indirect_call" {
        let Some(Hlvalue::Constant(c_graphs)) = op.args.last() else {
            return true;
        };
        let Some(graph_keys) = c_graphs.value.graphs() else {
            return true;
        };
        let Some(graphs) = resolve_graph_family(translator, graph_keys) else {
            return true;
        };
        for g in &graphs {
            if !has_no_side_effects(translator, g, seen) {
                return true;
            }
        }
        false
    } else {
        op_has_side_effects(op)
    }
}

fn resolve_graph_family(
    translator: &TranslationContext,
    graph_keys: &[usize],
) -> Option<Vec<GraphRef>> {
    let graphs = translator.graphs.borrow();
    let mut resolved: Vec<GraphRef> = Vec::with_capacity(graph_keys.len());
    for graph_key in graph_keys {
        let graph = graphs
            .iter()
            .find(|graph| GraphKey::of(graph).as_usize() == *graph_key)?;
        resolved.push(graph.clone());
    }
    Some(resolved)
}

/// Marker for `has_no_side_effects`'s `seen` set. Upstream stores graph
/// identities as dict keys; Rust uses [`GraphKey`] plus a thin wrapper
/// so the set's type is local to this module.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphKeyForSeen(usize);

impl GraphKeyForSeen {
    fn of(g: &GraphRef) -> Self {
        GraphKeyForSeen(std::rc::Rc::as_ptr(g) as usize)
    }
}

/// RPython `has_no_side_effects(translator, graph, seen=None)`
/// (simplify.py:355-375).
///
/// ```python
/// def has_no_side_effects(translator, graph, seen=None):
///     if translator.rtyper is None:
///         return False
///     else:
///         if graph.startblock not in translator.rtyper.already_seen:
///             return False
///     if seen is None:
///         seen = {}
///     elif graph in seen:
///         return True
///     newseen = seen.copy()
///     newseen[graph] = True
///     for block in graph.iterblocks():
///         if block is graph.exceptblock:
///             return False
///         for op in block.operations:
///             if rec_op_has_side_effects(translator, op, newseen):
///                 return False
///     return True
/// ```
///
/// Upstream short-circuits to `False` whenever `translator.rtyper is
/// None`. The Rust port now carries the real `TranslationContext.rtyper`
/// field, so this function follows the same branch structure. Full
/// post-rtyper equivalence still depends on the lloperation table and
/// call→graph resolution.
pub fn has_no_side_effects(
    translator: &TranslationContext,
    graph: &GraphRef,
    seen: Option<&HashSet<GraphKeyForSeen>>,
) -> bool {
    // upstream: `if translator.rtyper is None: return False`.
    if !translator_has_rtyper(translator) {
        return false;
    }
    // upstream: `if graph.startblock not in translator.rtyper.already_seen: return False`.
    if !rtyper_already_seen(translator, graph) {
        return false;
    }
    // upstream: seen-cycle guard.
    let key = GraphKeyForSeen::of(graph);
    let mut newseen: HashSet<GraphKeyForSeen> = match seen {
        None => HashSet::new(),
        Some(s) => {
            if s.contains(&key) {
                return true;
            }
            s.clone()
        }
    };
    newseen.insert(key);
    // upstream: walk blocks; bail on the except block or any op with
    // side effects.
    let exceptblock_key = BlockKey::of(&graph.borrow().exceptblock);
    for block in graph.borrow().iterblocks() {
        if BlockKey::of(&block) == exceptblock_key {
            return false;
        }
        let b = block.borrow();
        for op in &b.operations {
            if rec_op_has_side_effects(translator, op, Some(&newseen)) {
                return false;
            }
        }
    }
    true
}

/// RPython `translator.rtyper is None` guard.
fn translator_has_rtyper(translator: &TranslationContext) -> bool {
    translator.rtyper().is_some()
}

/// RPython `graph.startblock in translator.rtyper.already_seen`.
fn rtyper_already_seen(translator: &TranslationContext, graph: &GraphRef) -> bool {
    let Some(rtyper) = translator.rtyper() else {
        return false;
    };
    rtyper
        .already_seen
        .borrow()
        .contains_key(&BlockKey::of(&graph.borrow().startblock))
}

/// RPython `transform_dead_op_vars(graph, translator=None)`
/// (simplify.py:397-401). Thin wrapper around
/// `transform_dead_op_vars_in_blocks`.
pub fn transform_dead_op_vars(graph: &FunctionGraph, translator: Option<&TranslationContext>) {
    transform_dead_op_vars_in_blocks(
        &graph.iterblocks(),
        1,
        translator,
        Some(graph.startblock.clone()),
    );
}

/// RPython `transform_dead_op_vars_in_blocks(blocks, graphs, translator=None)`
/// (simplify.py:422-524). Rust carries `graphs_len` plus the optional
/// single-graph start block instead of a heterogeneous Python list:
/// upstream only branches on `len(graphs) == 1`, and the multi-graph
/// arm recovers start blocks through `translator.annotator`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn transform_dead_op_vars_in_blocks(
    blocks: &[BlockRef],
    graphs_len: usize,
    translator: Option<&TranslationContext>,
    single_graph_startblock: Option<BlockRef>,
) {
    // upstream: `set_of_blocks = set(blocks)`.
    let set_of_blocks: HashSet<BlockKey> = blocks.iter().map(BlockKey::of).collect();
    // upstream:
    // if len(graphs) == 1:
    //     start_blocks = {graphs[0].startblock}
    // else:
    //     start_blocks = {translator.annotator.annotated[block].startblock
    //                     for block in blocks}
    let start_blocks: HashSet<BlockKey> = if graphs_len == 1 {
        let mut start_blocks = HashSet::new();
        let single_graph_startblock = single_graph_startblock
            .expect("single-graph transform_dead_op_vars_in_blocks needs startblock");
        start_blocks.insert(BlockKey::of(&single_graph_startblock));
        start_blocks
    } else if blocks.is_empty() {
        // upstream: `{translator.annotator.annotated[block].startblock
        // for block in blocks}` is a lazy set comprehension — when
        // `blocks` is empty the comprehension never dereferences
        // `translator.annotator`. `complete_helpers()` reaches this
        // branch whenever `complete()` added no fresh blocks, so it
        // must be a silent no-op instead of a panic.
        HashSet::new()
    } else {
        let translator =
            translator.expect("multi-graph transform_dead_op_vars_in_blocks needs translator");
        let annotator = translator
            .annotator()
            .expect("multi-graph transform_dead_op_vars_in_blocks needs translator.annotator");
        let annotated = annotator.annotated.borrow();
        let mut start_blocks = HashSet::new();
        for block in blocks {
            // Upstream indexes `translator.annotator.annotated[block]`
            // directly here. By the time `complete_helpers()` reaches
            // `simplify(block_subset=...)`, `complete()` has already
            // raised on blocked (`False`) entries, so every block in
            // the subset must resolve to its owning graph.
            let graph = annotated
                .get(&BlockKey::of(block))
                .and_then(|graph| graph.as_ref())
                .expect(
                    "transform_dead_op_vars_in_blocks: block_subset contains an unannotated/blocked block in the multi-graph path",
                );
            start_blocks.insert(BlockKey::of(&graph.borrow().startblock));
        }
        start_blocks
    };
    let can_remove_ops = can_remove_opnames();
    let can_remove_builtins_list = can_remove_builtins();

    // `read_vars`: set of Variables whose value is read downstream.
    let mut read_vars: HashSet<Variable> = HashSet::new();
    // `dependencies`: map from Var → [dependency Vars].
    let mut dependencies: HashMap<Variable, HashSet<Variable>> = HashMap::new();

    let canremove_op = |op: &SpaceOperation, block: &BlockRef, idx: usize| -> bool {
        if !can_remove_ops.contains(op.opname.as_str()) {
            return false;
        }
        // upstream: `op is not block.raising_op`.
        //
        // `block.raising_op` is `block.operations[-1]` when
        // `block.canraise()` (`flowspace/model.rs`), so positional
        // identity (`idx == operations.len()-1`) reproduces upstream's
        // Python object-identity check. Value equality would
        // mis-classify a structurally-identical op earlier in the
        // block as the raising op — e.g. the same opname+args may
        // appear twice in `remove_assertion_errors` input.
        let b = block.borrow();
        if !b.canraise() {
            return true;
        }
        idx + 1 != b.operations.len()
    };
    let add_read_var = |read_vars: &mut HashSet<Variable>, v: &Hlvalue| {
        if let Hlvalue::Variable(var) = v {
            read_vars.insert(var.clone());
        }
    };
    let add_dep =
        |deps: &mut HashMap<Variable, HashSet<Variable>>, dest: &Hlvalue, src: &Hlvalue| {
            if let (Hlvalue::Variable(dv), Hlvalue::Variable(sv)) = (dest, src) {
                deps.entry(dv.clone()).or_default().insert(sv.clone());
            }
        };

    // upstream: `for block in blocks: compute read_vars + dependencies`.
    for block in blocks {
        let b = block.borrow();
        for (idx, op) in b.operations.iter().enumerate() {
            if !canremove_op(op, block, idx) {
                // upstream: `read_vars.update(op.args)`.
                for a in &op.args {
                    add_read_var(&mut read_vars, a);
                }
            } else {
                for a in &op.args {
                    add_dep(&mut dependencies, &op.result, a);
                }
            }
        }
        // upstream: `if isinstance(block.exitswitch, Variable): read_vars.add(block.exitswitch)`.
        if let Some(Hlvalue::Variable(sw)) = &b.exitswitch {
            read_vars.insert(sw.clone());
        }

        if !b.exits.is_empty() {
            for link in &b.exits {
                let link_b = link.borrow();
                let Some(target) = link_b.target.as_ref() else {
                    continue;
                };
                let target_key = BlockKey::of(target);
                let target_inputargs = target.borrow().inputargs.clone();
                let link_args: Vec<Option<Hlvalue>> = link_b.args.clone();
                let in_set = set_of_blocks.contains(&target_key);
                for (arg_opt, tgt) in link_args.iter().zip(target_inputargs.iter()) {
                    let Some(arg) = arg_opt else { continue };
                    if !in_set {
                        // upstream: `read_vars.add(arg); read_vars.add(targetarg)`.
                        add_read_var(&mut read_vars, arg);
                        add_read_var(&mut read_vars, tgt);
                    } else {
                        add_dep(&mut dependencies, tgt, arg);
                    }
                }
            }
        } else {
            // upstream: return/except blocks implicitly use inputargs.
            for a in &b.inputargs {
                add_read_var(&mut read_vars, a);
            }
        }

        // upstream: start block's inputargs always live.
        if start_blocks.contains(&BlockKey::of(block)) {
            for a in &b.inputargs {
                add_read_var(&mut read_vars, a);
            }
        }
    }

    // upstream: `flow_read_var_backward(set(read_vars))`.
    let mut pending: Vec<Variable> = read_vars.iter().cloned().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for prev in deps {
                if read_vars.insert(prev.clone()) {
                    pending.push(prev);
                }
            }
        }
    }

    // upstream: iterate blocks again, prune removable ops whose result
    // isn't read, prune link args whose target-inputarg isn't read.
    for block in blocks {
        // upstream: backward walk over operations.
        let ops_len = block.borrow().operations.len();
        let mut i = ops_len;
        while i > 0 {
            i -= 1;
            let (op, result_used, opname, first_arg_builtin) = {
                let b = block.borrow();
                let op = b.operations[i].clone();
                let result_used = match &op.result {
                    Hlvalue::Variable(v) => read_vars.contains(v),
                    Hlvalue::Constant(_) => true,
                };
                let first_arg_builtin = match op.args.first() {
                    Some(Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(h),
                        ..
                    })) => Some(h.clone()),
                    _ => None,
                };
                let opname = op.opname.clone();
                (op, result_used, opname, first_arg_builtin)
            };
            if result_used {
                continue;
            }
            if canremove_op(&op, block, i) {
                block.borrow_mut().operations.remove(i);
            } else if opname == "simple_call" {
                if let Some(h) = first_arg_builtin
                    && can_remove_builtins_list.iter().any(|b| b == &h)
                {
                    block.borrow_mut().operations.remove(i);
                }
            } else if opname == "direct_call"
                && let Some(trans) = translator
            {
                let Some(callee_arg) = op.args.first() else {
                    continue;
                };
                if let Some(graph) = get_graph(callee_arg, trans) {
                    // upstream: `op is not block.raising_op` —
                    // positional identity matches upstream object
                    // identity because `raising_op` is
                    // `operations[-1]` (`flowspace/model.rs`'s `canraise`).
                    let is_raising = {
                        let b = block.borrow();
                        b.canraise() && i + 1 == b.operations.len()
                    };
                    if has_no_side_effects(trans, &graph, None) && !is_raising {
                        block.borrow_mut().operations.remove(i);
                    }
                }
            }
        }

        // upstream: output vars never used — drop their positions from
        // link.args. Must happen before block.inputargs is shrunk so
        // the same-index cross-block invariant holds.
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.to_vec();
        for link in exits_snapshot {
            let target = link.borrow().target.clone();
            let Some(target) = target else { continue };
            let target_inputargs = target.borrow().inputargs.clone();
            let args_len = link.borrow().args.len();
            assert_eq!(args_len, target_inputargs.len(), "link arity mismatch");
            // upstream: back-to-front deletion.
            let mut i = args_len;
            while i > 0 {
                i -= 1;
                let tgt = &target_inputargs[i];
                let drop = match tgt {
                    Hlvalue::Variable(v) => !read_vars.contains(v),
                    Hlvalue::Constant(_) => false,
                };
                if drop {
                    link.borrow_mut().args.remove(i);
                }
            }
        }
    }

    // upstream: final pass — drop unused inputargs (matching link.args
    // are already gone).
    for block in blocks {
        let inputargs = block.borrow().inputargs.clone();
        let mut i = inputargs.len();
        while i > 0 {
            i -= 1;
            let drop = match &inputargs[i] {
                Hlvalue::Variable(v) => !read_vars.contains(v),
                Hlvalue::Constant(_) => false,
            };
            if drop {
                block.borrow_mut().inputargs.remove(i);
            }
        }
    }
}

/// RPython `class Representative` (simplify.py).
///
/// Info payload attached to each UF partition in
/// `remove_identical_vars_SSA`. Stores the per-partition
/// "representative" value; `absorb` is a no-op so the info that wins
/// the weighted union keeps its `rep`.
#[derive(Clone, Debug)]
pub struct Representative {
    rep: Hlvalue,
}

impl crate::tool::algo::unionfind::UnionFindInfo for Representative {
    fn absorb(&mut self, _other: Self) {}
}

/// RPython `all_equal(lst)` (simplify.py).
pub fn all_equal(lst: &[Hlvalue]) -> bool {
    match lst.first() {
        None => true,
        Some(first) => lst.iter().skip(1).all(|x| x == first),
    }
}

/// RPython `isspecialvar(v)` (simplify.py).
pub fn isspecialvar(v: &Hlvalue) -> bool {
    match v {
        Hlvalue::Variable(var) => {
            let p = var.name_prefix();
            p == "last_exception_" || p == "last_exc_value_"
        }
        _ => false,
    }
}

/// Extends `Block::renamevariables` with the `last_exception` /
/// `last_exc_value` rewrites that upstream's `remove_identical_vars_SSA`
/// requires. `Block::renamevariables` matches RPython's canonical shape
/// (args / operations / exitswitch / link.args) but stops short of the
/// exception-extras Links carry; this helper closes that gap.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn renamevariables_hl(block: &BlockRef, mapping: &HashMap<Variable, Hlvalue>) {
    if mapping.is_empty() {
        return;
    }
    let new_ops: Vec<SpaceOperation> = {
        let b = block.borrow();
        b.operations
            .iter()
            .map(|op| rename_op(op, mapping))
            .collect()
    };
    let new_exitswitch = {
        let b = block.borrow();
        b.exitswitch.as_ref().map(|sw| rename_hl(sw, mapping))
    };
    {
        let mut b = block.borrow_mut();
        b.operations = new_ops;
        b.exitswitch = new_exitswitch;
    }
    // link.args are Vec<Option<Hlvalue>>; rename in place.
    let exits_snapshot: Vec<LinkRef> = block.borrow().exits.to_vec();
    for link in exits_snapshot {
        let mut l = link.borrow_mut();
        l.args = l
            .args
            .iter()
            .map(|arg| arg.as_ref().map(|v| rename_hl(v, mapping)))
            .collect();
        if let Some(sw) = &l.last_exception {
            l.last_exception = Some(rename_hl(sw, mapping));
        }
        if let Some(sw) = &l.last_exc_value {
            l.last_exc_value = Some(rename_hl(sw, mapping));
        }
    }
}

/// RPython `remove_identical_vars_SSA(graph)` (simplify.py).
///
/// Upstream variant that uses its own `UnionFind(Representative)`
/// instead of `DataFlowFamilyBuilder`, inlining the phi-node collapse
/// loop over block inputs with linked `link.args[i]` as phi sources.
#[allow(non_snake_case)]
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn remove_identical_vars_SSA(graph: &FunctionGraph) {
    use crate::tool::algo::unionfind::UnionFind;

    let mut uf: UnionFind<Hlvalue, Representative> =
        UnionFind::new(|k: &Hlvalue| Representative { rep: k.clone() });

    // upstream: `entrymap = mkentrymap(graph); del entrymap[startblock];
    // entrymap.pop(returnblock, None); entrymap.pop(exceptblock, None)`.
    let mut entrymap = mkentrymap(graph);
    entrymap.remove(&BlockKey::of(&graph.startblock));
    entrymap.remove(&BlockKey::of(&graph.returnblock));
    entrymap.remove(&BlockKey::of(&graph.exceptblock));

    // upstream: `inputs[block] = zip(block.inputargs, zip(*[link.args
    // for link in links]))`. Rust: Vec<(Variable, Vec<Hlvalue>)> per
    // block. We key by BlockKey and carry a BlockRef alongside for
    // later iteration.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    let mut inputs: HashMap<BlockKey, (BlockRef, Vec<(Variable, Vec<Hlvalue>)>)> = HashMap::new();
    for (bkey, links) in entrymap.iter() {
        let block = links
            .first()
            .and_then(|l| l.borrow().target.clone())
            .expect("entrymap link missing target");
        let inputargs = block.borrow().inputargs.clone();
        let mut phis: Vec<(Variable, Vec<Hlvalue>)> = Vec::with_capacity(inputargs.len());
        for (i, input) in inputargs.iter().enumerate() {
            let Hlvalue::Variable(input_v) = input else {
                continue;
            };
            let mut phi_args: Vec<Hlvalue> = Vec::with_capacity(links.len());
            for link in links {
                let la: Vec<Option<Hlvalue>> = link.borrow().args.clone();
                let Some(Some(a)) = la.get(i).cloned() else {
                    panic!(
                        "remove_identical_vars_SSA: link.args[{i}] missing at block {:?}",
                        bkey
                    );
                };
                phi_args.push(a);
            }
            phis.push((input_v.clone(), phi_args));
        }
        inputs.insert(bkey.clone(), (block.clone(), phis));
    }

    // upstream: `progress = True; while progress: ... for block in
    // inputs: if simplify_phis(block): progress = True`.
    let block_keys: Vec<BlockKey> = inputs.keys().cloned().collect();
    let mut progress = true;
    while progress {
        progress = false;
        for bkey in &block_keys {
            if simplify_phis_inner(&mut uf, inputs.get_mut(bkey).unwrap()) {
                progress = true;
            }
        }
    }

    // upstream: `renaming = dict((key, uf[key].rep) for key in uf)` —
    // filter to Variable keys only (Constants-as-keys occur when
    // linked args brought them into the UF, but we only rename
    // Variables in block storage slots).
    let renaming: HashMap<Variable, Hlvalue> = {
        let keys: Vec<Hlvalue> = uf.keys().cloned().collect();
        let mut out: HashMap<Variable, Hlvalue> = HashMap::new();
        for k in keys {
            let Hlvalue::Variable(kv) = k.clone() else {
                continue;
            };
            if let Some(info) = uf.get(&k) {
                out.insert(kv, info.rep.clone());
            }
        }
        out
    };

    // upstream: rewrite each block's inputargs and every incoming
    // link's args to match the pruned `inputs[block]`. The order
    // matters — inputargs shrink before renamevariables runs so the
    // eliminated inputs' references in ops get the rep instead.
    for (bkey, (block, phis)) in inputs.iter() {
        let links = entrymap
            .get(bkey)
            .cloned()
            .expect("entrymap lookup consistent with inputs");
        let new_inputs: Vec<Hlvalue> = phis
            .iter()
            .map(|(v, _)| Hlvalue::Variable(v.clone()))
            .collect();
        // `per_link_args[link_idx][phi_idx] = phis[phi_idx].1[link_idx]`.
        let per_link_args: Vec<Vec<Option<Hlvalue>>> = (0..links.len())
            .map(|li| phis.iter().map(|(_, pa)| Some(pa[li].clone())).collect())
            .collect();
        block.borrow_mut().inputargs = new_inputs;
        assert_eq!(links.len(), per_link_args.len());
        for (link, args) in links.iter().zip(per_link_args) {
            link.borrow_mut().args = args;
        }
    }

    for (block, _) in inputs.values() {
        renamevariables_hl(block, &renaming);
    }
}

/// Inner of `remove_identical_vars_SSA`'s `simplify_phis(block)` closure
/// (simplify.py:555-573).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn simplify_phis_inner(
    uf: &mut crate::tool::algo::unionfind::UnionFind<Hlvalue, Representative>,
    slot: &mut (BlockRef, Vec<(Variable, Vec<Hlvalue>)>),
) -> bool {
    let (_block, phis) = slot;
    let mut to_remove: Vec<usize> = Vec::new();
    let mut unique_phis: HashMap<Vec<Hlvalue>, Variable> = HashMap::new();
    for (i, (input, phi_args)) in phis.iter().enumerate() {
        // upstream: `new_args = [uf.find_rep(arg) for arg in phi_args]`.
        let new_args: Vec<Hlvalue> = phi_args.iter().map(|a| uf.find_rep(a.clone())).collect();
        // upstream: `if all_equal(new_args) and not isspecialvar(new_args[0]):`
        let first = new_args.first().cloned();
        if let Some(first) = first
            && all_equal(&new_args)
            && !isspecialvar(&first)
        {
            uf.union(first, Hlvalue::Variable(input.clone()));
            to_remove.push(i);
            continue;
        }
        // upstream: else branch — group by identical phi-tuple.
        let key = new_args;
        if let Some(existing) = unique_phis.get(&key).cloned() {
            uf.union(
                Hlvalue::Variable(existing),
                Hlvalue::Variable(input.clone()),
            );
            to_remove.push(i);
        } else {
            unique_phis.insert(key, input.clone());
        }
    }
    // upstream: `for i in reversed(to_remove): del phis[i]`.
    for i in to_remove.iter().rev() {
        phis.remove(*i);
    }
    !to_remove.is_empty()
}

/// RPython `remove_identical_vars(graph)` (simplify.py).
///
/// ```python
/// def remove_identical_vars(graph):
///     """When the same variable is passed multiple times into the next
///     block, pass it only once."""
///     builder = DataFlowFamilyBuilder(graph)
///     variable_families = builder.get_variable_families()  # vertical removal
///     while True:
///         if not builder.merge_identical_phi_nodes():    # horizontal removal
///             break
///         if not builder.complete():                     # vertical removal
///             break
///     for block, links in mkentrymap(graph).items():
///         if block is graph.startblock:
///             continue
///         renaming = {}
///         family2blockvar = {}
///         kills = []
///         for i, v in enumerate(block.inputargs):
///             v1 = variable_families.find_rep(v)
///             if v1 in family2blockvar:
///                 renaming[v] = family2blockvar[v1]
///                 kills.append(i)
///             else:
///                 family2blockvar[v1] = v
///         if renaming:
///             block.renamevariables(renaming)
///             kills.reverse()
///             for i in kills:
///                 del block.inputargs[i]
///                 for link in links:
///                     del link.args[i]
/// ```
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn remove_identical_vars(graph: &FunctionGraph) {
    let mut builder = DataFlowFamilyBuilder::new(graph);
    // upstream: `builder.get_variable_families()` triggers initial
    // vertical removal (calls `complete()`); we discard the returned
    // handle and drive the subsequent steps on the same builder.
    let _ = builder.get_variable_families();
    loop {
        // upstream: horizontal removal first, then vertical.
        if !builder.merge_identical_phi_nodes() {
            break;
        }
        if !builder.complete() {
            break;
        }
    }
    let variable_families = &mut builder.variable_families;

    let entrymap = mkentrymap(graph);
    for (block_key, links) in entrymap.iter() {
        // upstream: `if block is graph.startblock: continue`.
        if *block_key == BlockKey::of(&graph.startblock) {
            continue;
        }
        let block = links
            .first()
            .and_then(|l| l.borrow().target.clone())
            .expect("link.target missing");

        let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut family2blockvar: HashMap<Hlvalue, Variable> = HashMap::new();
        let mut kills: Vec<usize> = Vec::new();

        let inputargs_snapshot: Vec<Hlvalue> = block.borrow().inputargs.clone();
        for (i, input) in inputargs_snapshot.iter().enumerate() {
            let Hlvalue::Variable(v) = input else {
                continue;
            };
            let v1 = variable_families.find_rep(Hlvalue::Variable(v.clone()));
            if let Some(existing) = family2blockvar.get(&v1) {
                renaming.insert(v.clone(), Hlvalue::Variable(existing.clone()));
                kills.push(i);
            } else {
                family2blockvar.insert(v1, v.clone());
            }
        }
        if !renaming.is_empty() {
            block.borrow_mut().renamevariables(&renaming);
            // upstream: `kills.reverse()` + `del block.inputargs[i]`
            // + `del link.args[i]` for each incoming link.
            kills.reverse();
            for i in &kills {
                block.borrow_mut().inputargs.remove(*i);
                for link in links {
                    link.borrow_mut().args.remove(*i);
                }
            }
        }
    }
}

/// RPython `simplify_exceptions(graph)` (simplify.py).
///
/// Collapses the `except Exception:` chain-of-is_/issubtype tests
/// produced by the flowspace into a single list of `exitcase=cls`
/// links on the raising block.
///
/// ```python
/// def simplify_exceptions(graph):
///     renaming = {}
///     for block in graph.iterblocks():
///         if not (block.canraise
///                 and block.exits[-1].exitcase is Exception):
///             continue
///         covered = [link.exitcase for link in block.exits[1:-1]]
///         seen = []
///         preserve = list(block.exits[:-1])
///         exc = block.exits[-1]
///         last_exception = exc.last_exception
///         last_exc_value = exc.last_exc_value
///         query = exc.target
///         switches = []
///         while len(query.exits) == 2:
///             newrenaming = {}
///             for lprev, ltarg in zip(exc.args, query.inputargs):
///                 newrenaming[ltarg] = lprev.replace(renaming)
///             op = query.operations[0]
///             if not (op.opname in ("is_", "issubtype") and
///                     op.args[0].replace(newrenaming) == last_exception):
///                 break
///             renaming.update(newrenaming)
///             case = query.operations[0].args[-1].value
///             assert issubclass(case, py.builtin.BaseException)
///             lno, lyes = query.exits
///             assert lno.exitcase == False and lyes.exitcase == True
///             if case not in seen:
///                 is_covered = False
///                 for cov in covered:
///                     if issubclass(case, cov):
///                         is_covered = True
///                         break
///                 if not is_covered:
///                     switches.append( (case, lyes) )
///                 seen.append(case)
///             exc = lno
///             query = exc.target
///         if Exception not in seen:
///             switches.append( (Exception, exc) )
///         exits = []
///         for case, oldlink in switches:
///             link = oldlink.replace(renaming)
///             link.last_exception = last_exception
///             link.last_exc_value = last_exc_value
///             renaming2 = {}
///             for v in link.getextravars():
///                 renaming2[v] = Variable(v)
///             link = link.replace(renaming2)
///             link.exitcase = case
///             exits.append(link)
///         block.recloseblock(*(preserve + exits))
/// ```
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn simplify_exceptions(graph: &FunctionGraph) {
    let exception_class = HOST_ENV
        .lookup_builtin("Exception")
        .expect("HOST_ENV missing Exception");
    let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();

    for block in graph.iterblocks() {
        // upstream: `if not (block.canraise and block.exits[-1].exitcase is Exception): continue`.
        let (canraise, last_is_exception, exits_snapshot) = {
            let b = block.borrow();
            let canraise = b.canraise();
            let last_is_exception = b
                .exits
                .last()
                .and_then(|l| l.borrow().exitcase.clone())
                .map(|c| match c {
                    Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(h),
                        ..
                    }) => h == exception_class,
                    _ => false,
                })
                .unwrap_or(false);
            let exits_snapshot: Vec<LinkRef> = b.exits.to_vec();
            (canraise, last_is_exception, exits_snapshot)
        };
        if !(canraise && last_is_exception) {
            continue;
        }

        // upstream: `covered = [link.exitcase for link in block.exits[1:-1]]`.
        let covered: Vec<Option<Hlvalue>> = exits_snapshot
            .iter()
            .skip(1)
            .take(exits_snapshot.len().saturating_sub(2))
            .map(|l| l.borrow().exitcase.clone())
            .collect();
        // upstream: `preserve = list(block.exits[:-1])`.
        let preserve: Vec<LinkRef> =
            exits_snapshot[..exits_snapshot.len().saturating_sub(1)].to_vec();
        let mut seen: Vec<Hlvalue> = Vec::new();
        let mut switches: Vec<(Hlvalue, LinkRef)> = Vec::new();

        // upstream: `exc = block.exits[-1]; ... query = exc.target`.
        let mut exc: LinkRef = exits_snapshot
            .last()
            .cloned()
            .expect("canraise block has ≥2 exits");
        let last_exception = exc.borrow().last_exception.clone();
        let last_exc_value = exc.borrow().last_exc_value.clone();
        let mut query: Option<BlockRef> = exc.borrow().target.clone();

        // upstream: `while len(query.exits) == 2:`.
        while let Some(q) = query.clone() {
            if q.borrow().exits.len() != 2 {
                break;
            }
            // upstream: `newrenaming[ltarg] = lprev.replace(renaming)`.
            let mut newrenaming: HashMap<Variable, Hlvalue> = HashMap::new();
            let exc_args: Vec<Option<Hlvalue>> = exc.borrow().args.clone();
            let q_inputargs = q.borrow().inputargs.clone();
            for (lprev_opt, ltarg) in exc_args.iter().zip(q_inputargs.iter()) {
                let Some(lprev) = lprev_opt else { continue };
                let Hlvalue::Variable(tgt_v) = ltarg else {
                    continue;
                };
                let replaced = rename_hl(lprev, &renaming);
                newrenaming.insert(tgt_v.clone(), replaced);
            }
            // upstream: `op = query.operations[0]`.
            let q_ops = q.borrow().operations.clone();
            let Some(op) = q_ops.first() else { break };
            // upstream: `if not (op.opname in ("is_", "issubtype") and
            //              op.args[0].replace(newrenaming) == last_exception): break`.
            if !(op.opname == "is_" || op.opname == "issubtype") {
                break;
            }
            let Some(last_exc_hl) = last_exception.as_ref() else {
                break;
            };
            let Some(op_arg0) = op.args.first() else {
                break;
            };
            if rename_hl(op_arg0, &newrenaming) != *last_exc_hl {
                break;
            }
            // upstream: `renaming.update(newrenaming)`.
            for (k, v) in newrenaming.into_iter() {
                renaming.insert(k, v);
            }
            // upstream: `case = query.operations[0].args[-1].value`.
            let Some(case_arg) = op.args.last() else {
                break;
            };
            let case_hl = case_arg.clone();
            // upstream: `assert issubclass(case, BaseException)` — debug.
            debug_assert!(match &case_hl {
                Hlvalue::Constant(Constant {
                    value: ConstValue::HostObject(h),
                    ..
                }) => h.is_class(),
                _ => false,
            });
            // upstream: `lno, lyes = query.exits; assert lno.exitcase == False and
            // lyes.exitcase == True`. Our boolean switch invariant
            // (checkgraph) maps exits[0] → False, exits[1] → True.
            let q_exits = q.borrow().exits.clone();
            let Some(lno) = q_exits.first().cloned() else {
                break;
            };
            let Some(lyes) = q_exits.get(1).cloned() else {
                break;
            };
            // upstream: `if case not in seen: ... switches.append((case, lyes))`.
            if !seen.contains(&case_hl) {
                let is_covered = covered.iter().any(|cov_opt| match cov_opt {
                    Some(cov) => is_exitcase_subclass(&case_hl, cov),
                    None => false,
                });
                if !is_covered {
                    switches.push((case_hl.clone(), lyes));
                }
                seen.push(case_hl);
            }
            // upstream: `exc = lno; query = exc.target`.
            exc = lno;
            query = exc.borrow().target.clone();
        }
        // upstream: `if Exception not in seen: switches.append((Exception, exc))`.
        let exception_hlvalue = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
            exception_class.clone(),
        )));
        if !seen.contains(&exception_hlvalue) {
            switches.push((exception_hlvalue.clone(), exc));
        }

        // upstream: `exits = []; for case, oldlink in switches:
        //   link = oldlink.replace(renaming); ... link.exitcase = case; exits.append(link)`.
        let mut new_exits: Vec<LinkRef> = Vec::new();
        for (case_hl, oldlink) in switches {
            let link = rename_link_args(&oldlink, &renaming);
            {
                let mut lb = link.borrow_mut();
                lb.last_exception = last_exception.clone();
                lb.last_exc_value = last_exc_value.clone();
            }
            // upstream: `renaming2[v] = Variable(v)` for each extra
            // var — fresh identity, upstream Variable(v) copies prefix
            // via rename; our Variable::copy does the same.
            let mut renaming2: HashMap<Variable, Hlvalue> = HashMap::new();
            {
                let lb = link.borrow();
                if let Some(Hlvalue::Variable(v)) = &lb.last_exception {
                    renaming2.insert(v.clone(), Hlvalue::Variable(v.copy()));
                }
                if let Some(Hlvalue::Variable(v)) = &lb.last_exc_value {
                    renaming2.insert(v.clone(), Hlvalue::Variable(v.copy()));
                }
            }
            // Apply renaming2 in place on the fresh link's args, etc.
            let link2 = rename_link_args(&link, &renaming2);
            link2.borrow_mut().exitcase = Some(case_hl);
            new_exits.push(link2);
        }
        let mut combined: Vec<LinkRef> = preserve;
        combined.extend(new_exits);
        block.recloseblock(combined);
    }
}

/// RPython `coalesce_bool(graph)` (simplify.py).
///
/// Collapses the two-step pattern
/// ```text
/// block: bool(v); exitswitch=result_v
///   ├─False→ block2: bool(result_v_again); exitswitch=...
///   └─True→  block2_same
/// ```
/// into a direct jump from `block` to `block2`'s true/false targets.
/// Upstream walks candidate bool-ended blocks, rewrites each
/// `bool`-exitcase-keyed exit to point at `block2`'s outgoing arm, and
/// repeats until a fixed point.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn coalesce_bool(graph: &FunctionGraph) {
    // upstream: list of (block, [(case, target_block)]) candidate tuples.
    let mut candidates: Vec<(BlockRef, Vec<(bool, BlockRef)>)> = Vec::new();
    for block in graph.iterblocks() {
        let is_bool = {
            let b = block.borrow();
            b.operations
                .last()
                .map(|op| op.opname == "bool")
                .unwrap_or(false)
        };
        if !is_bool {
            continue;
        }
        let tgts = has_bool_exitpath(&block);
        if !tgts.is_empty() {
            candidates.push((block, tgts));
        }
    }

    while let Some((cand, tgts)) = candidates.pop() {
        let cand_exits_snapshot: Vec<LinkRef> = cand.borrow().exits.to_vec();
        let mut new_exits: Vec<LinkRef> = cand_exits_snapshot.clone();

        for (case_bool, tgt) in tgts {
            // upstream: `exit = cand.exits[case]` where `case` is bool
            // (Python bool subclasses int → cand.exits[0/1]).
            let idx = if case_bool { 1 } else { 0 };
            let Some(exit_link) = cand_exits_snapshot.get(idx).cloned() else {
                continue;
            };
            // upstream: `rrenaming = dict(zip(tgt.inputargs, exit.args));
            //  rrenaming[tgt.operations[0].result] = cand.operations[-1].result`.
            let mut rrenaming: HashMap<Variable, Hlvalue> = HashMap::new();
            let tgt_inputargs = tgt.borrow().inputargs.clone();
            let exit_args: Vec<Option<Hlvalue>> = exit_link.borrow().args.clone();
            for (ltarg, lprev_opt) in tgt_inputargs.iter().zip(exit_args.iter()) {
                let Hlvalue::Variable(tgt_v) = ltarg else {
                    continue;
                };
                let Some(lprev) = lprev_opt else { continue };
                rrenaming.insert(tgt_v.clone(), lprev.clone());
            }
            let tgt_first_result = {
                let b = tgt.borrow();
                b.operations
                    .first()
                    .map(|op| op.result.clone())
                    .and_then(|r| match r {
                        Hlvalue::Variable(v) => Some(v),
                        _ => None,
                    })
            };
            let cand_last_result = {
                let b = cand.borrow();
                b.operations.last().map(|op| op.result.clone())
            };
            if let (Some(tgt_first), Some(cand_last)) = (tgt_first_result, cand_last_result) {
                rrenaming.insert(tgt_first, cand_last);
            }
            // upstream: `newlink = tgt.exits[case].copy(rename)`.
            let tgt_exits = tgt.borrow().exits.clone();
            let Some(tgt_exit) = tgt_exits.get(idx).cloned() else {
                continue;
            };
            let newlink = rename_link_args(&tgt_exit, &rrenaming);
            new_exits[idx] = newlink;
        }
        cand.recloseblock(new_exits);
        // upstream: retry against the same cand in case another bool-
        // chain opened up.
        let again = has_bool_exitpath(&cand);
        if !again.is_empty() {
            candidates.push((cand, again));
        }
    }
}

/// Helper for `coalesce_bool` — mirrors upstream's nested
/// `has_bool_exitpath(block)` closure (simplify.py).
fn has_bool_exitpath(block: &BlockRef) -> Vec<(bool, BlockRef)> {
    let mut tgts: Vec<(bool, BlockRef)> = Vec::new();
    let b = block.borrow();
    let Some(start_op) = b.operations.last() else {
        return tgts;
    };
    let Some(cond_v) = start_op.args.first().cloned() else {
        return tgts;
    };
    if b.exitswitch.as_ref() != Some(&start_op.result) {
        return tgts;
    }
    for exit in &b.exits {
        let exit_b = exit.borrow();
        let Some(tgt) = exit_b.target.clone() else {
            continue;
        };
        // upstream: `if tgt == block: continue`.
        if std::rc::Rc::ptr_eq(&tgt, block) {
            continue;
        }
        let tgt_b = tgt.borrow();
        if tgt_b.operations.len() != 1 || tgt_b.operations[0].opname != "bool" {
            continue;
        }
        let tgt_op = &tgt_b.operations[0];
        if tgt_b.exitswitch.as_ref() != Some(&tgt_op.result) {
            continue;
        }
        // upstream: `rrenaming.get(tgt_op.args[0]) == cond_v`.
        let Some(tgt_op_arg0) = tgt_op.args.first() else {
            continue;
        };
        let Hlvalue::Variable(arg_var) = tgt_op_arg0 else {
            continue;
        };
        // Build rrenaming just enough to check `arg_var → cond_v`.
        let position = tgt_b
            .inputargs
            .iter()
            .position(|ia| matches!(ia, Hlvalue::Variable(v) if v == arg_var));
        let Some(pos) = position else { continue };
        let Some(link_arg) = exit_b.args.get(pos).cloned().flatten() else {
            continue;
        };
        if link_arg != cond_v {
            continue;
        }
        let case_bool = match &exit_b.exitcase {
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Bool(b),
                ..
            })) => *b,
            _ => continue,
        };
        drop(tgt_b);
        tgts.push((case_bool, tgt));
    }
    tgts
}

/// RPython `transform_xxxitem(graph)` (simplify.py).
///
/// ```python
/// def transform_xxxitem(graph):
///     # xxx setitem too
///     for block in graph.iterblocks():
///         if block.canraise:
///             last_op = block.raising_op
///             if last_op.opname == 'getitem':
///                 postfx = []
///                 for exit in block.exits:
///                     if exit.exitcase is IndexError:
///                         postfx.append('idx')
///                 if postfx:
///                     Op = getattr(op, '_'.join(['getitem'] + postfx))
///                     newop = Op(*last_op.args)
///                     newop.result = last_op.result
///                     block.operations[-1] = newop
/// ```
///
/// `op.getitem_idx` is the sole descendant: when `IndexError` appears
/// in a `getitem` block's exit cases, we rewrite the raising op to
/// `getitem_idx`. Upstream's open-ended postfx list (`['idx', 'key']`
/// etc.) does not materialise — `op.getitem_idx_key` does exist but
/// the pass only builds an `'idx'` postfx.
pub fn transform_xxxitem(graph: &FunctionGraph) {
    let index_error = HOST_ENV
        .lookup_builtin("IndexError")
        .expect("HOST_ENV missing IndexError");
    for block in graph.iterblocks() {
        let (canraise, is_getitem) = {
            let b = block.borrow();
            let canraise = b.canraise();
            let is_getitem = canraise
                && b.raising_op()
                    .map(|op| op.opname == "getitem")
                    .unwrap_or(false);
            (canraise, is_getitem)
        };
        if !canraise || !is_getitem {
            continue;
        }
        // upstream: `for exit in block.exits: if exit.exitcase is IndexError: postfx.append('idx')`.
        let has_idx = {
            let b = block.borrow();
            b.exits.iter().any(|link| match &link.borrow().exitcase {
                Some(Hlvalue::Constant(Constant {
                    value: ConstValue::HostObject(h),
                    ..
                })) => h == &index_error,
                _ => false,
            })
        };
        if !has_idx {
            continue;
        }
        // upstream: `newop = Op(*last_op.args); newop.result = last_op.result;
        // block.operations[-1] = newop`. The Rust port rewrites opname
        // in place since SpaceOperation is already the concrete carrier.
        let mut b = block.borrow_mut();
        if let Some(op) = b.operations.last_mut() {
            op.opname = "getitem_idx".to_string();
        }
    }
}

/// RPython `remove_dead_exceptions(graph)` (simplify.py).
///
/// ```python
/// def remove_dead_exceptions(graph):
///     """Exceptions can be removed if they are unreachable"""
///     def issubclassofmember(cls, seq):
///         for member in seq:
///             if member and issubclass(cls, member):
///                 return True
///         return False
///     for block in list(graph.iterblocks()):
///         if not block.canraise:
///             continue
///         exits = []
///         seen = []
///         for link in block.exits:
///             case = link.exitcase
///             if issubclassofmember(case, seen):
///                 continue
///             while len(exits) > 1:
///                 prev = exits[-1]
///                 if not (issubclass(prev.exitcase, link.exitcase) and
///                     prev.target is link.target and prev.args == link.args):
///                     break
///                 exits.pop()
///             exits.append(link)
///             seen.append(case)
///         block.recloseblock(*exits)
/// ```
pub fn remove_dead_exceptions(graph: &FunctionGraph) {
    // upstream: `for block in list(graph.iterblocks())` — snapshot.
    for block in graph.iterblocks() {
        if !block.borrow().canraise() {
            continue;
        }
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.to_vec();
        let mut new_exits: Vec<LinkRef> = Vec::new();
        let mut seen: Vec<Option<Hlvalue>> = Vec::new();

        for link in exits_snapshot {
            let case = link.borrow().exitcase.clone();

            // upstream: `if issubclassofmember(case, seen): continue`.
            if issubclassofmember(case.as_ref(), &seen) {
                continue;
            }

            // upstream: merge the previous case if it's a subclass of
            // the current one and the link shape matches.
            while new_exits.len() > 1 {
                let prev = new_exits.last().expect("len > 1 ⇒ non-empty");
                let (subclass_ok, target_same, args_same) = {
                    let prev_b = prev.borrow();
                    let link_b = link.borrow();
                    let subclass_ok = match (&prev_b.exitcase, &link_b.exitcase) {
                        (Some(a), Some(b)) => is_exitcase_subclass(a, b),
                        _ => false,
                    };
                    let target_same = match (prev_b.target.as_ref(), link_b.target.as_ref()) {
                        (Some(a), Some(b)) => std::rc::Rc::ptr_eq(a, b),
                        _ => false,
                    };
                    let args_same = prev_b.args == link_b.args;
                    (subclass_ok, target_same, args_same)
                };
                if !(subclass_ok && target_same && args_same) {
                    break;
                }
                new_exits.pop();
            }
            new_exits.push(link);
            seen.push(case);
        }
        block.recloseblock(new_exits);
    }
}

fn issubclassofmember(cls: Option<&Hlvalue>, seq: &[Option<Hlvalue>]) -> bool {
    let Some(cls) = cls else {
        return false;
    };
    for member in seq {
        // upstream: `if member and issubclass(cls, member): return True`.
        let Some(member) = member else { continue };
        if is_exitcase_subclass(cls, member) {
            return true;
        }
    }
    false
}

/// `issubclass(exitcase_cls, other_cls)` over Hlvalue-wrapped Constant
/// HostObjects. Returns `false` for any non-class payload (upstream's
/// Python `issubclass` raises TypeError on non-classes, but upstream
/// only ever calls this with exception classes).
fn is_exitcase_subclass(cls: &Hlvalue, other: &Hlvalue) -> bool {
    match (cls, other) {
        (
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(a),
                ..
            }),
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(b),
                ..
            }),
        ) => a.is_subclass_of(b),
        _ => false,
    }
}

/// RPython `remove_assertion_errors(graph)` (simplify.py).
///
/// ```python
/// def remove_assertion_errors(graph):
///     """Remove branches that go directly to raising an AssertionError,
///     assuming that AssertionError shouldn't occur at run-time.  Note that
///     this is how implicit exceptions are removed (see _implicit_ in
///     flowcontext.py).
///     """
///     for block in list(graph.iterblocks()):
///         for i in range(len(block.exits)-1, -1, -1):
///             exit = block.exits[i]
///             if not (exit.target is graph.exceptblock and
///                     exit.args[0] == Constant(AssertionError)):
///                 continue
///             if len(block.exits) < 2:
///                 break
///             if block.canraise:
///                 if exit.exitcase is None:
///                     break
///                 if len(block.exits) == 2:
///                     block.exitswitch = None
///                     exit.exitcase = None
///             lst = list(block.exits)
///             del lst[i]
///             block.recloseblock(*lst)
/// ```
pub fn remove_assertion_errors(graph: &FunctionGraph) {
    // upstream: `for block in list(graph.iterblocks())` — snapshot
    // blocks so recloseblock mid-iteration is safe.
    let assert_err_class: HostObject = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock_key = BlockKey::of(&graph.exceptblock);

    for block in graph.iterblocks() {
        let mut i = block.borrow().exits.len();
        while i > 0 {
            i -= 1;
            let (targets_except, args_is_assert_err, canraise, exitcase_none, exits_len) = {
                let b = block.borrow();
                let Some(exit) = b.exits.get(i) else {
                    break;
                };
                let exit_b = exit.borrow();
                let targets_except = exit_b
                    .target
                    .as_ref()
                    .map(|t| BlockKey::of(t) == exceptblock_key)
                    .unwrap_or(false);
                let args_first = exit_b.args.first().cloned().flatten();
                let args_is_assert_err = matches!(
                    args_first,
                    Some(Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(ref h),
                        ..
                    })) if h == &assert_err_class
                );
                (
                    targets_except,
                    args_is_assert_err,
                    b.canraise(),
                    exit_b.exitcase.is_none(),
                    b.exits.len(),
                )
            };

            // upstream: `if not (exit.target is graph.exceptblock and
            // exit.args[0] == Constant(AssertionError)): continue`.
            if !(targets_except && args_is_assert_err) {
                continue;
            }
            // upstream: `if len(block.exits) < 2: break`.
            if exits_len < 2 {
                break;
            }
            if canraise {
                // upstream: `if exit.exitcase is None: break`.
                if exitcase_none {
                    break;
                }
                // upstream: `if len(block.exits) == 2: block.exitswitch
                // = None; exit.exitcase = None` — the surviving exit
                // (the non-i one) gets promoted to an unconditional
                // link below once we delete exit i.
                if exits_len == 2 {
                    // upstream mutates the `exit` being removed to
                    // have `exitcase = None`, but since we drop it in
                    // the same block this is effectively no-op for
                    // the outcome. Still mirror the mutation for
                    // parity — a reader reconstructing upstream
                    // semantics shouldn't be surprised.
                    block.borrow_mut().exitswitch = None;
                    if let Some(e) = block.borrow().exits.get(i) {
                        e.borrow_mut().exitcase = None;
                    }
                }
            }
            // upstream: `lst = list(block.exits); del lst[i];
            // block.recloseblock(*lst)`.
            let mut lst: Vec<LinkRef> = block.borrow().exits.to_vec();
            lst.remove(i);
            block.recloseblock(lst);
        }
    }
}

/// RPython `transform_ovfcheck(graph)` (simplify.py).
///
/// Rewrites `simple_call(ovfcheck, result_of_prev_op)` into the
/// `_ovf` variant of the previous operation. When `ovfcheck` sits at
/// the start of its block, upstream collapses the predecessor link
/// first via `join_blocks` and re-enters; the Rust port mirrors that
/// fixpoint loop.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn transform_ovfcheck(graph: &FunctionGraph) {
    // upstream `covf = Constant(rarithmetic.ovfcheck)`. The sentinel
    // lives on the `rpython.rlib.rarithmetic` module; looking it up
    // through the module keeps `find_global("ovfcheck")` upstream-
    // accurate (unqualified `ovfcheck` still raises ImportError).
    let ovfcheck_sentinel: HostObject = HOST_ENV
        .import_module("rpython.rlib.rarithmetic")
        .and_then(|m| m.module_get("ovfcheck"))
        .expect("rpython.rlib.rarithmetic missing ovfcheck sentinel");

    loop {
        let mut any_block_needs_merge = false;

        for block in graph.iterblocks() {
            // upstream: `for i in range(len(block.operations)-1, -1, -1)`.
            let ops_len = block.borrow().operations.len();
            let mut i = ops_len;
            while i > 0 {
                i -= 1;
                let (is_ovfcheck_call, result_of_call) = {
                    let b = block.borrow();
                    let op = &b.operations[i];
                    let matches_ovfcheck = op.opname == "simple_call"
                        && matches!(
                            op.args.first(),
                            Some(Hlvalue::Constant(Constant {
                                value: ConstValue::HostObject(h),
                                ..
                            })) if h == &ovfcheck_sentinel
                        );
                    (matches_ovfcheck, op.result.clone())
                };
                if !is_ovfcheck_call {
                    continue;
                }

                // upstream: hard case — `ovfcheck` at block start.
                if i == 0 {
                    let entrymap = mkentrymap(graph);
                    let Some(links) = entrymap.get(&BlockKey::of(&block)) else {
                        break;
                    };
                    assert_eq!(
                        links.len(),
                        1,
                        "ovfcheck at block start requires single entry"
                    );
                    let Some(prevblock) = links[0]
                        .borrow()
                        .prevblock
                        .as_ref()
                        .and_then(|w| w.upgrade())
                    else {
                        break;
                    };
                    {
                        let first_exit = prevblock.borrow().exits.first().cloned();
                        if let Some(fe) = first_exit {
                            assert!(matches!(
                                fe.borrow().target.as_ref(),
                                Some(t) if std::rc::Rc::ptr_eq(t, &block)
                            ));
                        }
                    }
                    prevblock.borrow_mut().exitswitch = None;
                    prevblock.recloseblock(vec![links[0].clone()]);
                    any_block_needs_merge = true;
                    break;
                }

                // upstream: `op1 = block.operations[i - 1]`. Must be
                // overflow-capable.
                let (op1_opname, op1_result) = {
                    let b = block.borrow();
                    let op1 = &b.operations[i - 1];
                    (op1.opname.clone(), op1.result.clone())
                };
                let kind = crate::flowspace::operation::OpKind::from_opname(&op1_opname)
                    .unwrap_or_else(|| {
                        panic!("ovfcheck on unknown opname {op1_opname} in {}", graph.name)
                    });
                let ovf_kind = kind.ovf_variant().unwrap_or_else(|| {
                    panic!(
                        "ovfcheck in {}: Operation {op1_opname} has no overflow variant",
                        graph.name
                    )
                });
                // upstream: `op1_ovf = op1.ovfchecked()`; we just flip
                // opname in place — args/result/offset already match.
                {
                    let mut b = block.borrow_mut();
                    b.operations[i - 1].opname = ovf_kind.opname().to_string();
                    b.operations.remove(i);
                }
                let Hlvalue::Variable(result_v) = result_of_call else {
                    continue;
                };
                let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
                renaming.insert(result_v, op1_result);
                renamevariables_hl(&block, &renaming);
            }
        }

        if !any_block_needs_merge {
            break;
        }
        // upstream: after merging, re-run join_blocks + retry.
        join_blocks(graph);
    }
}

fn is_stop_iteration_exitcase(exitcase: Option<&Hlvalue>) -> bool {
    let stop_iteration = HOST_ENV
        .lookup_builtin("StopIteration")
        .expect("HOST_ENV missing StopIteration");
    matches!(
        exitcase,
        Some(Hlvalue::Constant(Constant {
            value: ConstValue::HostObject(obj),
            ..
        })) if obj == &stop_iteration
    )
}

/// RPython `class DetectorFailed` (simplify.py).
///
/// The Rust detector routes failure through `Option`/early returns, but the
/// upstream exception name is part of the public parity surface for this
/// module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorFailed;

impl std::fmt::Display for DetectorFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DetectorFailed")
    }
}

impl std::error::Error for DetectorFailed {}

pub struct ListComprehensionDetector<'a> {
    graph: &'a FunctionGraph,
    loops: Vec<(BlockRef, BlockRef, Hlvalue)>,
    newlist_v: HashMap<Hlvalue, BlockRef>,
    variable_families: &'a mut crate::tool::algo::unionfind::UnionFind<Hlvalue, ()>,
    reachable_cache: HashMap<(BlockKey, BlockKey, BlockKey), bool>,
    vmeth: Hlvalue,
    vlistfamily: Hlvalue,
    vlistcone: HashMap<BlockKey, bool>,
    escapes: HashMap<BlockKey, bool>,
}

impl<'a> ListComprehensionDetector<'a> {
    fn enum_blocks_with_vlist_from(
        &mut self,
        fromblock: &BlockRef,
        avoid: &BlockRef,
    ) -> Vec<BlockRef> {
        let mut found: HashSet<BlockKey> = HashSet::from([BlockKey::of(avoid)]);
        let mut pending = vec![fromblock.clone()];
        let mut result = Vec::new();
        while let Some(block) = pending.pop() {
            let block_key = BlockKey::of(&block);
            if found.contains(&block_key) {
                continue;
            }
            if !self.vlist_alive(&block) {
                continue;
            }
            result.push(block.clone());
            found.insert(block_key);
            for exit in block.borrow().exits.iter() {
                if let Some(target) = exit.borrow().target.clone() {
                    pending.push(target);
                }
            }
        }
        result
    }

    fn enum_reachable_blocks(
        &mut self,
        fromblock: &BlockRef,
        stop_at: &BlockRef,
        stay_within: Option<&HashSet<BlockKey>>,
    ) -> Vec<BlockRef> {
        if BlockKey::of(fromblock) == BlockKey::of(stop_at) {
            return Vec::new();
        }
        let mut found: HashSet<BlockKey> = HashSet::from([BlockKey::of(stop_at)]);
        let mut pending = vec![fromblock.clone()];
        let mut result = Vec::new();
        while let Some(block) = pending.pop() {
            let block_key = BlockKey::of(&block);
            if found.contains(&block_key) {
                continue;
            }
            found.insert(block_key);
            for exit in block.borrow().exits.iter() {
                let Some(target) = exit.borrow().target.clone() else {
                    continue;
                };
                let target_key = BlockKey::of(&target);
                if stay_within.is_none_or(|blocks| blocks.contains(&target_key)) {
                    result.push(target.clone());
                    pending.push(target);
                }
            }
        }
        result
    }

    fn reachable_within(
        &mut self,
        fromblock: &BlockRef,
        toblock: &BlockRef,
        avoid: &BlockRef,
        stay_within: &HashSet<BlockKey>,
    ) -> bool {
        if BlockKey::of(toblock) == BlockKey::of(avoid) {
            return false;
        }
        self.enum_reachable_blocks(fromblock, avoid, Some(stay_within))
            .into_iter()
            .any(|block| BlockKey::of(&block) == BlockKey::of(toblock))
    }

    fn reachable(&mut self, fromblock: &BlockRef, toblock: &BlockRef, avoid: &BlockRef) -> bool {
        let to_key = BlockKey::of(toblock);
        let avoid_key = BlockKey::of(avoid);
        if to_key == avoid_key {
            return false;
        }
        let from_key = BlockKey::of(fromblock);
        if let Some(result) =
            self.reachable_cache
                .get(&(from_key.clone(), to_key.clone(), avoid_key.clone()))
        {
            return *result;
        }
        let mut future = vec![fromblock.clone()];
        for block in self.enum_reachable_blocks(fromblock, avoid, None) {
            self.reachable_cache.insert(
                (from_key.clone(), BlockKey::of(&block), avoid_key.clone()),
                true,
            );
            if BlockKey::of(&block) == to_key {
                return true;
            }
            future.push(block);
        }
        for block in future {
            self.reachable_cache.insert(
                (BlockKey::of(&block), to_key.clone(), avoid_key.clone()),
                false,
            );
        }
        false
    }

    fn contains_vlist_values(&mut self, args: &[Hlvalue]) -> Option<Hlvalue> {
        for arg in args {
            if self.variable_families.find_rep(arg.clone()) == self.vlistfamily {
                return Some(arg.clone());
            }
        }
        None
    }

    fn contains_vlist_linkargs(&mut self, args: &[Option<Hlvalue>]) -> Option<Hlvalue> {
        for arg in args {
            let Some(arg) = arg else {
                continue;
            };
            if self.variable_families.find_rep(arg.clone()) == self.vlistfamily {
                return Some(arg.clone());
            }
        }
        None
    }

    fn vlist_alive(&mut self, block: &BlockRef) -> bool {
        let key = BlockKey::of(block);
        if let Some(result) = self.vlistcone.get(&key) {
            return *result;
        }
        let inputargs = block.borrow().inputargs.clone();
        let result = self.contains_vlist_values(&inputargs).is_some();
        self.vlistcone.insert(key, result);
        result
    }

    fn vlist_escapes(&mut self, block: &BlockRef) -> bool {
        let key = BlockKey::of(block);
        if let Some(result) = self.escapes.get(&key) {
            return *result;
        }
        let operations = block.borrow().operations.clone();
        let result = operations.iter().any(|op| {
            if op.result == self.vmeth {
                return false;
            }
            if op.opname == "getitem" {
                return false;
            }
            self.contains_vlist_values(&op.args).is_some()
        });
        self.escapes.insert(key, result);
        result
    }

    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    fn run(&mut self, vlist: Hlvalue, vmeth: Hlvalue, appendblock: BlockRef) -> Result<(), ()> {
        let append_ops = appendblock.borrow().operations.clone();
        for hlop in &append_ops {
            if hlop.opname == "simple_call" && hlop.args.first() == Some(&vmeth) {
                continue;
            }
            if hlop.args.iter().any(|arg| arg == &vmeth) {
                return Err(());
            }
        }
        for link in appendblock.borrow().exits.iter() {
            if link
                .borrow()
                .args
                .iter()
                .any(|arg| arg.as_ref() == Some(&vmeth))
            {
                return Err(());
            }
        }

        self.vmeth = vmeth;
        self.vlistfamily = self.variable_families.find_rep(vlist);
        let newlistblock = self
            .newlist_v
            .get(&self.vlistfamily)
            .cloned()
            .expect("newlist family must exist");
        self.vlistcone = HashMap::from([(BlockKey::of(&newlistblock), true)]);
        self.escapes = HashMap::from([
            (BlockKey::of(&self.graph.returnblock), true),
            (BlockKey::of(&self.graph.exceptblock), true),
        ]);

        let (loopnextblock, iterblock, viterfamily, loopbody, stopblocks, exactlength) = {
            let mut accepted = None;
            let loops = self.loops.clone();
            for (loopnextblock, iterblock, viterfamily) in loops {
                if !self.vlist_alive(&loopnextblock) {
                    continue;
                }
                if self.reachable(&newlistblock, &appendblock, &iterblock) {
                    continue;
                }
                if self.reachable(&loopnextblock, &iterblock, &newlistblock) {
                    continue;
                }
                if self.reachable(&appendblock, &appendblock, &loopnextblock) {
                    continue;
                }
                let stopblocks: Vec<BlockRef> = loopnextblock
                    .borrow()
                    .exits
                    .iter()
                    .filter(|link| link.borrow().exitcase.is_some())
                    .filter_map(|link| link.borrow().target.clone())
                    .collect();
                let mut stop_reaches_append = false;
                for stopblock in &stopblocks {
                    if self.reachable(stopblock, &appendblock, &newlistblock) {
                        stop_reaches_append = true;
                        break;
                    }
                }
                if stop_reaches_append {
                    continue;
                }
                let mut loopbody: HashMap<BlockKey, BlockRef> = HashMap::new();
                for block in self.graph.iterblocks() {
                    if self.vlist_alive(&block) && self.reachable(&block, &appendblock, &iterblock)
                    {
                        loopbody.insert(BlockKey::of(&block), block);
                    }
                }
                if !loopbody.contains_key(&BlockKey::of(&appendblock)) {
                    continue;
                }
                let loopheader = self.enum_blocks_with_vlist_from(&newlistblock, &loopnextblock);
                assert_eq!(BlockKey::of(&loopheader[0]), BlockKey::of(&newlistblock));
                let mut escapes = false;
                for block in loopheader.iter().cloned().chain(loopbody.values().cloned()) {
                    assert!(self.vlist_alive(&block));
                    if self.vlist_escapes(&block) {
                        escapes = true;
                        break;
                    }
                }
                if escapes {
                    continue;
                }
                let loopbody_keys: HashSet<BlockKey> = loopbody.keys().cloned().collect();
                let exactlength = !self.reachable_within(
                    &loopnextblock,
                    &loopnextblock,
                    &appendblock,
                    &loopbody_keys,
                );
                accepted = Some((
                    loopnextblock,
                    iterblock,
                    viterfamily,
                    loopbody,
                    stopblocks,
                    exactlength,
                ));
                break;
            }
            accepted.ok_or(())?
        };

        assert!(!loopbody.contains_key(&BlockKey::of(&iterblock)));
        assert!(loopbody.contains_key(&BlockKey::of(&loopnextblock)));
        for stopblock in &stopblocks {
            assert!(!loopbody.contains_key(&BlockKey::of(stopblock)));
        }

        let iter_exit = iterblock
            .borrow()
            .exits
            .first()
            .cloned()
            .expect("iterblock must have a single exit");
        let vlist = self
            .contains_vlist_linkargs(&iter_exit.borrow().args)
            .expect("iterblock exit should carry the new list");
        let iterable = {
            let operations = iterblock.borrow().operations.clone();
            let mut found = None;
            for hlop in &operations {
                let res = self.variable_families.find_rep(hlop.result.clone());
                if res == viterfamily {
                    found = hlop.args.first().cloned();
                    break;
                }
            }
            found.expect("lost 'iter' operation")
        };
        let mut hint_items = HashMap::new();
        hint_items.insert(
            ConstValue::byte_str(if exactlength {
                "maxlength"
            } else {
                "maxlength_inexact"
            }),
            ConstValue::Bool(true),
        );
        let hint_result = match &vlist {
            Hlvalue::Variable(v) => Hlvalue::Variable(v.copy()),
            other => panic!("hint target must be a Variable, got {other:?}"),
        };
        let hint = SpaceOperation::new(
            "hint",
            vec![
                vlist.clone(),
                iterable,
                Hlvalue::Constant(Constant::new(ConstValue::Dict(hint_items))),
            ],
            hint_result.clone(),
        );
        iterblock.borrow_mut().operations.push(hint);
        {
            let mut link = iter_exit.borrow_mut();
            for arg in &mut link.args {
                if arg.as_ref() == Some(&vlist) {
                    *arg = Some(hint_result.clone());
                }
            }
        }

        for block in loopbody.values() {
            let exits = block.borrow().exits.clone();
            for link in exits {
                let Some(target) = link.borrow().target.clone() else {
                    continue;
                };
                if loopbody.contains_key(&BlockKey::of(&target)) {
                    continue;
                }
                let Some(vlist) = self.contains_vlist_linkargs(&link.borrow().args) else {
                    continue;
                };
                let mut hints =
                    HashMap::from([(ConstValue::byte_str("fence"), ConstValue::Bool(true))]);
                if exactlength
                    && BlockKey::of(block) == BlockKey::of(&loopnextblock)
                    && stopblocks
                        .iter()
                        .any(|stopblock| BlockKey::of(stopblock) == BlockKey::of(&target))
                {
                    hints.insert(ConstValue::byte_str("exactlength"), ConstValue::Bool(true));
                }
                let newblock = crate::translator::unsimplify::insert_empty_block(&link, vec![]);
                let index = link
                    .borrow()
                    .args
                    .iter()
                    .position(|arg| arg.as_ref() == Some(&vlist))
                    .expect("vlist must stay on the rewritten link");
                let vlist2 = newblock.borrow().inputargs[index].clone();
                let vlist3 = match &vlist2 {
                    Hlvalue::Variable(v) => Hlvalue::Variable(v.copy()),
                    other => panic!("fence hint target must be a Variable, got {other:?}"),
                };
                newblock.borrow_mut().inputargs[index] = vlist3.clone();
                let hint = SpaceOperation::new(
                    "hint",
                    vec![
                        vlist3,
                        Hlvalue::Constant(Constant::new(ConstValue::Dict(hints))),
                    ],
                    vlist2,
                );
                newblock.borrow_mut().operations.push(hint);
            }
        }
        Ok(())
    }
}

/// RPython `detect_list_comprehension(graph)` (simplify.py).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn detect_list_comprehension(graph: &FunctionGraph) {
    let mut variable_families = DataFlowFamilyBuilder::new(graph).into_variable_families();
    let c_append = Constant::new(ConstValue::byte_str("append"));
    let mut newlist_v: HashMap<Hlvalue, BlockRef> = HashMap::new();
    let mut iter_v: HashMap<Hlvalue, BlockRef> = HashMap::new();
    let mut append_v: Vec<(Hlvalue, Hlvalue, BlockRef)> = Vec::new();
    let mut loopnextblocks: Vec<(BlockRef, Hlvalue)> = Vec::new();

    for block in graph.iterblocks() {
        let block_b = block.borrow();
        if block_b.operations.len() == 1
            && block_b.operations[0].opname == "next"
            && block_b.canraise()
            && block_b.exits.len() >= 2
        {
            let has_none_case = block_b
                .exits
                .iter()
                .any(|link| link.borrow().exitcase.is_none());
            let has_stop_iteration = block_b
                .exits
                .iter()
                .any(|link| is_stop_iteration_exitcase(link.borrow().exitcase.as_ref()));
            if has_none_case && has_stop_iteration {
                loopnextblocks.push((block.clone(), block_b.operations[0].args[0].clone()));
                continue;
            }
        }
        for op in &block_b.operations {
            if op.opname == "newlist" && op.args.is_empty() {
                let vlist = variable_families.find_rep(op.result.clone());
                newlist_v.insert(vlist, block.clone());
            }
            if op.opname == "iter" {
                let viter = variable_families.find_rep(op.result.clone());
                iter_v.insert(viter, block.clone());
            }
        }
    }

    let mut loops: Vec<(BlockRef, BlockRef, Hlvalue)> = Vec::new();
    for (block, viter) in loopnextblocks {
        let viterfamily = variable_families.find_rep(viter);
        if let Some(iterblock) = iter_v.get(&viterfamily).cloned() {
            let iterblock_b = iterblock.borrow();
            if iterblock_b.exits.len() == 1
                && iterblock_b.exitswitch.is_none()
                && iterblock_b.exits[0]
                    .borrow()
                    .target
                    .as_ref()
                    .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&block))
            {
                loops.push((block, iterblock.clone(), viterfamily));
            }
        }
    }
    if newlist_v.is_empty() || loops.is_empty() {
        return;
    }

    for block in graph.iterblocks() {
        let ops = block.borrow().operations.clone();
        for i in 0..ops.len().saturating_sub(1) {
            let op = &ops[i];
            if op.opname == "getattr"
                && op.args.get(1) == Some(&Hlvalue::Constant(c_append.clone()))
            {
                let vlist = variable_families.find_rep(op.args[0].clone());
                if newlist_v.contains_key(&vlist) {
                    for op2 in ops.iter().skip(i + 1) {
                        if op2.opname == "simple_call"
                            && op2.args.len() == 2
                            && op2.args[0] == op.result
                        {
                            append_v.push((op.args[0].clone(), op.result.clone(), block.clone()));
                            break;
                        }
                    }
                }
            }
        }
    }
    if append_v.is_empty() {
        return;
    }

    let mut detector = ListComprehensionDetector {
        graph,
        loops,
        newlist_v,
        variable_families: &mut variable_families,
        reachable_cache: HashMap::new(),
        vmeth: Hlvalue::Constant(Constant::new(ConstValue::Placeholder)),
        vlistfamily: Hlvalue::Constant(Constant::new(ConstValue::Placeholder)),
        vlistcone: HashMap::new(),
        escapes: HashMap::new(),
    };
    let mut graphmutated = false;
    for (vlist, vmeth, block) in append_v {
        if graphmutated {
            detect_list_comprehension(graph);
            return;
        }
        if detector.run(vlist, vmeth, block).is_ok() {
            graphmutated = true;
        }
    }
}

/// RPython `all_passes = [...]` (simplify.py) + `simplify_graph`
/// (simplify.py:1075-1081).
///
/// `all_passes` upstream is a list of callables; the Rust port uses a
/// `fn(&FunctionGraph)` slice so every entry has the same call shape.
/// `transform_dead_op_vars(graph, translator=None)` and
/// `SSA_to_SSI(graph, annotator=None)` are wrapped to match — both
/// accept `translator=None` / `annotator=None` by default and the
/// Rust wrappers pass `None`.
#[allow(non_upper_case_globals)]
pub const all_passes: &[fn(&FunctionGraph)] = &[
    dead_op_vars_shim,
    eliminate_empty_blocks,
    remove_assertion_errors,
    remove_identical_vars_SSA,
    constfold_exitswitch,
    remove_trivial_links,
    ssa_to_ssi_shim,
    coalesce_bool,
    transform_ovfcheck,
    simplify_exceptions,
    transform_xxxitem,
    remove_dead_exceptions,
];

fn dead_op_vars_shim(graph: &FunctionGraph) {
    transform_dead_op_vars(graph, None);
}

fn ssa_to_ssi_shim(graph: &FunctionGraph) {
    crate::translator::backendopt::ssa::ssa_to_ssi(graph, None);
}

/// RPython `simplify_graph(graph, passes=True)` (simplify.py).
///
/// ```python
/// def simplify_graph(graph, passes=True):
///     if passes is True:
///         passes = all_passes
///     for pass_ in passes:
///         pass_(graph)
///     checkgraph(graph)
/// ```
pub fn simplify_graph(graph: &FunctionGraph, passes: Option<&[fn(&FunctionGraph)]>) {
    let default_passes = all_passes;
    let passes = passes.unwrap_or(default_passes);
    for pass_ in passes {
        pass_(graph);
    }
    checkgraph(graph);
}

/// RPython `cleanup_graph(graph)` (simplify.py).
///
/// ```python
/// def cleanup_graph(graph):
///     checkgraph(graph)
///     eliminate_empty_blocks(graph)
///     join_blocks(graph)
///     remove_identical_vars(graph)
///     checkgraph(graph)
/// ```
pub fn cleanup_graph(graph: &FunctionGraph) {
    checkgraph(graph);
    eliminate_empty_blocks(graph);
    join_blocks(graph);
    remove_identical_vars(graph);
    checkgraph(graph);
}

/// RPython `constfold_exitswitch(graph)` (simplify.py).
///
/// ```python
/// def constfold_exitswitch(graph):
///     block = graph.startblock
///     seen = set([block])
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         target = link.target
///         if target in seen:
///             continue
///         source = link.prevblock
///         switch = source.exitswitch
///         if (isinstance(switch, Constant) and not source.canraise):
///             exits = replace_exitswitch_by_constant(source, switch)
///             stack.extend(exits)
///         else:
///             seen.add(target)
///             stack.extend(target.exits)
/// ```
pub fn constfold_exitswitch(graph: &FunctionGraph) {
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.to_vec();

    while let Some(link) = stack.pop() {
        let (prev_rc, target_rc) = {
            let l = link.borrow();
            (
                l.prevblock.as_ref().and_then(|w| w.upgrade()),
                l.target.clone(),
            )
        };
        let Some(target_rc) = target_rc else {
            continue;
        };
        if seen.contains(&BlockKey::of(&target_rc)) {
            continue;
        }
        let Some(prev_rc) = prev_rc else {
            continue;
        };

        // upstream: `switch = source.exitswitch`.
        let (is_const_switch, const_val, is_canraise) = {
            let b = prev_rc.borrow();
            match &b.exitswitch {
                Some(Hlvalue::Constant(c)) => (true, Some(c.clone()), b.canraise()),
                _ => (false, None, b.canraise()),
            }
        };

        if is_const_switch && !is_canraise {
            let const_val = const_val.expect("const_val set when is_const_switch");
            let new_exits = replace_exitswitch_by_constant(&prev_rc, &const_val);
            stack.extend(new_exits);
        } else {
            seen.insert(BlockKey::of(&target_rc));
            let more: Vec<LinkRef> = target_rc.borrow().exits.to_vec();
            stack.extend(more);
        }
    }
}

/// RPython `remove_trivial_links(graph)` (simplify.py).
///
/// ```python
/// def remove_trivial_links(graph):
///     """Remove trivial links by merging their source and target blocks
///
///     A link is trivial if it has no arguments, is the single exit of its
///     source and the single parent of its target.
///     """
///     entrymap = mkentrymap(graph)
///     block = graph.startblock
///     seen = set([block])
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         if link.target in seen:
///             continue
///         source = link.prevblock
///         target = link.target
///         if (not link.args and source.exitswitch is None and
///                 len(entrymap[target]) == 1 and
///                 target.exits):  # stop at the returnblock
///             assert len(source.exits) == 1
///             source.operations.extend(target.operations)
///             source.exitswitch = newexitswitch = target.exitswitch
///             source.recloseblock(*target.exits)
///             stack.extend(source.exits)
///         else:
///             seen.add(target)
///             stack.extend(target.exits)
/// ```
pub fn remove_trivial_links(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.to_vec();

    while let Some(link) = stack.pop() {
        let (prev_rc, target_rc, link_args_empty) = {
            let l = link.borrow();
            let prev = l.prevblock.as_ref().and_then(|w| w.upgrade());
            let target = l.target.clone();
            // upstream `not link.args` — treat `Vec<Option<Hlvalue>>` as
            // empty when there are no args at all. Transient-merge
            // `None` slots are valid args in upstream semantics.
            let args_empty = l.args.is_empty();
            (prev, target, args_empty)
        };
        let Some(target_rc) = target_rc else {
            continue;
        };
        if seen.contains(&BlockKey::of(&target_rc)) {
            continue;
        }
        let Some(prev_rc) = prev_rc else {
            continue;
        };

        let source_switch_none = prev_rc.borrow().exitswitch.is_none();
        let target_entry_count = entrymap
            .get(&BlockKey::of(&target_rc))
            .map(Vec::len)
            .unwrap_or(0);
        let target_has_exits = !target_rc.borrow().exits.is_empty();

        let is_trivial =
            link_args_empty && source_switch_none && target_entry_count == 1 && target_has_exits;

        if is_trivial {
            assert_eq!(
                prev_rc.borrow().exits.len(),
                1,
                "remove_trivial_links: source block has exitswitch=None but != 1 exit"
            );
            // upstream: `source.operations.extend(target.operations)`.
            let target_ops = target_rc.borrow().operations.clone();
            prev_rc.borrow_mut().operations.extend(target_ops);
            // upstream: `source.exitswitch = target.exitswitch`.
            let new_switch = target_rc.borrow().exitswitch.clone();
            prev_rc.borrow_mut().exitswitch = new_switch;
            // upstream: `source.recloseblock(*target.exits)`.
            let target_exits: Vec<LinkRef> = target_rc.borrow().exits.to_vec();
            prev_rc.recloseblock(target_exits);
            let source_exits: Vec<LinkRef> = prev_rc.borrow().exits.to_vec();
            stack.extend(source_exits);
        } else {
            seen.insert(BlockKey::of(&target_rc));
            let more: Vec<LinkRef> = target_rc.borrow().exits.to_vec();
            stack.extend(more);
        }
    }
}

/// RPython `join_blocks(graph)` (simplify.py).
///
/// ```python
/// def join_blocks(graph):
///     """Links can be deleted if they are the single exit of a block and
///     the single entry point of the next block.  When this happens, we can
///     append all the operations of the following block to the preceeding
///     block (but renaming variables with the appropriate arguments.)
///     """
///     entrymap = mkentrymap(graph)
///     block = graph.startblock
///     seen = {block: True}
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         if (link.prevblock.exitswitch is None and
///             len(entrymap[link.target]) == 1 and
///             link.target.exits):  # stop at the returnblock
///             assert len(link.prevblock.exits) == 1
///             renaming = {}
///             for vprev, vtarg in zip(link.args, link.target.inputargs):
///                 renaming[vtarg] = vprev
///             def rename_op(op):
///                 op = op.replace(renaming)
///                 # special case...
///                 if op.opname == 'indirect_call':
///                     if isinstance(op.args[0], Constant):
///                         assert isinstance(op.args[-1], Constant)
///                         del op.args[-1]
///                         op.opname = 'direct_call'
///                 return op
///             for op in link.target.operations:
///                 link.prevblock.operations.append(rename_op(op))
///             exits = []
///             for exit in link.target.exits:
///                 newexit = exit.replace(renaming)
///                 exits.append(newexit)
///             if link.target.exitswitch:
///                 newexitswitch = link.target.exitswitch.replace(renaming)
///             else:
///                 newexitswitch = None
///             link.prevblock.exitswitch = newexitswitch
///             link.prevblock.recloseblock(*exits)
///             if (isinstance(newexitswitch, Constant) and
///                     not link.prevblock.canraise):
///                 exits = replace_exitswitch_by_constant(link.prevblock,
///                                                        newexitswitch)
///             stack.extend(exits)
///         else:
///             if link.target not in seen:
///                 stack.extend(link.target.exits)
///                 seen[link.target] = True
/// ```
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn join_blocks(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));

    // upstream: `stack = list(block.exits)`.
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.to_vec();

    while let Some(link) = stack.pop() {
        // Snapshot the fields we need before re-borrowing the Link
        // itself (Rust borrow-checker keeps mutation and reads split).
        let (prev_rc, target_rc, target_inputargs, link_args_all_some, link_args) = {
            let l = link.borrow();
            let prev = l.prevblock.as_ref().and_then(|w| w.upgrade());
            let target = l.target.clone();
            let target_inputargs = target
                .as_ref()
                .map(|t| t.borrow().inputargs.clone())
                .unwrap_or_default();
            let args_all_some = l.args.iter().all(|a| a.is_some());
            let args: Vec<Hlvalue> = l.args.iter().filter_map(|a| a.clone()).collect();
            (prev, target, target_inputargs, args_all_some, args)
        };
        let Some(prev_rc) = prev_rc else {
            continue;
        };
        let Some(target_rc) = target_rc else {
            continue;
        };

        let prev_switch_is_none = prev_rc.borrow().exitswitch.is_none();
        let target_entry_count = entrymap
            .get(&BlockKey::of(&target_rc))
            .map(Vec::len)
            .unwrap_or(0);
        let target_has_exits = !target_rc.borrow().exits.is_empty();

        // upstream: `link.prevblock.exitswitch is None and
        // len(entrymap[link.target]) == 1 and link.target.exits`.
        let can_join = prev_switch_is_none
            && target_entry_count == 1
            && target_has_exits
            && link_args_all_some;

        if can_join {
            assert_eq!(
                prev_rc.borrow().exits.len(),
                1,
                "join_blocks: prevblock has exitswitch=None but != 1 exit"
            );

            // upstream: `renaming[vtarg] = vprev` for each pair.
            let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
            for (vprev, vtarg) in link_args.iter().zip(target_inputargs.iter()) {
                if let Hlvalue::Variable(tgt) = vtarg {
                    renaming.insert(tgt.clone(), vprev.clone());
                }
            }

            // upstream: append renamed target.operations to prevblock.
            let target_ops: Vec<SpaceOperation> = target_rc.borrow().operations.clone();
            for op in &target_ops {
                prev_rc
                    .borrow_mut()
                    .operations
                    .push(rename_op(op, &renaming));
            }

            // upstream: `exits = [exit.replace(renaming) for exit in target.exits]`.
            let target_exits: Vec<LinkRef> = target_rc.borrow().exits.to_vec();
            let mut new_exits: Vec<LinkRef> = target_exits
                .iter()
                .map(|e| rename_link_args(e, &renaming))
                .collect();

            // upstream: `newexitswitch = target.exitswitch.replace(renaming) or None`.
            let new_exitswitch = target_rc
                .borrow()
                .exitswitch
                .as_ref()
                .map(|sw| rename_hl(sw, &renaming));
            prev_rc.borrow_mut().exitswitch = new_exitswitch.clone();
            prev_rc.recloseblock(new_exits.clone());

            // upstream: constant-fold the new switch when prevblock is
            // not a can-raise block (simplify.py:311-314).
            if let Some(Hlvalue::Constant(const_)) = &new_exitswitch
                && !prev_rc.borrow().canraise()
            {
                new_exits = replace_exitswitch_by_constant(&prev_rc, const_);
            }

            stack.extend(new_exits);
        } else {
            // upstream: `if link.target not in seen: stack.extend(link.target.exits)`.
            let target_key = BlockKey::of(&target_rc);
            if !seen.contains(&target_key) {
                seen.insert(target_key);
                let more: Vec<LinkRef> = target_rc.borrow().exits.to_vec();
                stack.extend(more);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphKey, GraphRef, Hlvalue, HostObject, Link,
        Variable, c_last_exception,
    };
    use crate::translator::rtyper::lltypesystem::lltype;
    use crate::translator::translator::TranslationContext;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn signed_var() -> Variable {
        let v = Variable::new();
        v.set_concretetype(Some(lltype::LowLevelType::Signed));
        v
    }

    fn fn_ptr_var() -> Variable {
        let v = Variable::new();
        v.set_concretetype(Some(lltype::LowLevelType::Ptr(Box::new(lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![lltype::LowLevelType::Signed],
                result: lltype::LowLevelType::Signed,
            }),
        }))));
        v
    }

    fn test_functionptr_void(graph: &GraphRef) -> lltype::_ptr {
        lltype::getfunctionptr(graph, |_| Ok(lltype::LowLevelType::Void)).unwrap()
    }

    #[test]
    fn get_graph_reads_llptr_funcobj_graph() {
        let start = Block::shared(vec![]);
        let ret = Variable::new();
        ret.set_concretetype(Some(lltype::LowLevelType::Void));
        let graph: GraphRef = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "callee",
            start,
            Hlvalue::Variable(ret),
        )));
        let translator = TranslationContext::new();
        translator.graphs.borrow_mut().push(graph.clone());

        let arg = Hlvalue::Constant(Constant::new(ConstValue::LLPtr(Box::new(
            test_functionptr_void(&graph),
        ))));

        let got = get_graph(&arg, &translator).expect("expected graph");
        assert_eq!(GraphKey::of(&got), GraphKey::of(&graph));
    }

    #[test]
    fn get_graph_returns_none_for_delayed_pointer() {
        let translator = TranslationContext::new();
        let arg = Hlvalue::Constant(Constant::new(ConstValue::LLPtr(Box::new(
            lltype::_ptr::new(
                lltype::Ptr {
                    TO: lltype::PtrTarget::Func(lltype::FuncType {
                        args: vec![],
                        result: lltype::LowLevelType::Void,
                    }),
                },
                Err(lltype::DelayedPointer),
            ),
        ))));

        assert!(get_graph(&arg, &translator).is_none());
    }

    #[test]
    fn detect_list_comprehension_inserts_maxlength_and_fence_hints() {
        let iterable = Variable::named("iterable");
        let start = Block::shared(vec![Hlvalue::Variable(iterable.clone())]);
        let graph = FunctionGraph::new("lc", start.clone());

        let vlist0 = Variable::named("vlist0");
        let viter0 = Variable::named("viter0");
        let loopnext = Block::shared(vec![
            Hlvalue::Variable(vlist0.copy()),
            Hlvalue::Variable(viter0.copy()),
        ]);
        let append = Block::shared(vec![
            Hlvalue::Variable(vlist0.copy()),
            Hlvalue::Variable(viter0.copy()),
            Hlvalue::Variable(Variable::named("item")),
        ]);
        let after = Block::shared(vec![Hlvalue::Variable(vlist0.copy())]);

        let vmeth = Variable::named("vmeth");
        start.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![],
            Hlvalue::Variable(vlist0.clone()),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "iter",
            vec![Hlvalue::Variable(iterable.clone())],
            Hlvalue::Variable(viter0.clone()),
        ));
        start.borrow_mut().closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(vlist0.clone()),
                    Hlvalue::Variable(viter0.clone()),
                ],
                Some(loopnext.clone()),
                None,
            )
            .into_ref(),
        ]);

        let item_from_next = Variable::named("item_from_next");
        let loop_iter_arg = loopnext.borrow().inputargs[1].clone();
        loopnext.borrow_mut().operations.push(SpaceOperation::new(
            "next",
            vec![loop_iter_arg],
            Hlvalue::Variable(item_from_next.clone()),
        ));
        loopnext.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        let stop_iteration = HOST_ENV
            .lookup_builtin("StopIteration")
            .expect("HOST_ENV missing StopIteration");
        let loop_success_args = {
            let loopnext_b = loopnext.borrow();
            vec![
                loopnext_b.inputargs[0].clone(),
                loopnext_b.inputargs[1].clone(),
                Hlvalue::Variable(item_from_next.clone()),
            ]
        };
        let loop_stop_args = {
            let loopnext_b = loopnext.borrow();
            vec![loopnext_b.inputargs[0].clone()]
        };
        loopnext.borrow_mut().closeblock(vec![
            Link::new(loop_success_args, Some(append.clone()), None).into_ref(),
            Link::new(
                loop_stop_args,
                Some(after.clone()),
                Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                    stop_iteration,
                )))),
            )
            .into_ref(),
        ]);

        let append_inputargs = append.borrow().inputargs.clone();
        let list_in_append = append_inputargs[0].clone();
        let item_in_append = append_inputargs[2].clone();
        let mul_result = Variable::named("mul_result");
        append.borrow_mut().operations.push(SpaceOperation::new(
            "getattr",
            vec![
                list_in_append.clone(),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("append"))),
            ],
            Hlvalue::Variable(vmeth.clone()),
        ));
        append.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![
                item_in_append.clone(),
                Hlvalue::Constant(Constant::new(ConstValue::Int(17))),
            ],
            Hlvalue::Variable(mul_result.clone()),
        ));
        append.borrow_mut().operations.push(SpaceOperation::new(
            "simple_call",
            vec![
                Hlvalue::Variable(vmeth.clone()),
                Hlvalue::Variable(mul_result.clone()),
            ],
            Hlvalue::Variable(Variable::named("append_res")),
        ));
        let append_backedge_args = {
            let append_b = append.borrow();
            vec![append_b.inputargs[0].clone(), append_b.inputargs[1].clone()]
        };
        append.borrow_mut().closeblock(vec![
            Link::new(append_backedge_args, Some(loopnext.clone()), None).into_ref(),
        ]);

        let after_args = after.borrow().inputargs.clone();
        after.borrow_mut().closeblock(vec![
            Link::new(after_args, Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        detect_list_comprehension(&graph);

        let summary = crate::flowspace::model::summary(&graph);
        assert_eq!(summary.get("newlist"), Some(&1));
        assert_eq!(summary.get("iter"), Some(&1));
        assert_eq!(summary.get("next"), Some(&1));
        assert_eq!(summary.get("getattr"), Some(&1));
        assert_eq!(summary.get("simple_call"), Some(&1));
        assert_eq!(summary.get("mul"), Some(&1));
        assert_eq!(summary.get("hint"), Some(&2));
    }

    #[test]
    fn detect_list_comprehension_ignores_non_builtin_stop_iteration_namesake() {
        let iterable = Variable::named("iterable");
        let start = Block::shared(vec![Hlvalue::Variable(iterable.clone())]);
        let graph = FunctionGraph::new("lc_namesake", start.clone());

        let vlist0 = Variable::named("vlist0");
        let viter0 = Variable::named("viter0");
        let loopnext = Block::shared(vec![
            Hlvalue::Variable(vlist0.copy()),
            Hlvalue::Variable(viter0.copy()),
        ]);
        let append = Block::shared(vec![
            Hlvalue::Variable(vlist0.copy()),
            Hlvalue::Variable(viter0.copy()),
            Hlvalue::Variable(Variable::named("item")),
        ]);
        let after = Block::shared(vec![Hlvalue::Variable(vlist0.copy())]);

        let vmeth = Variable::named("vmeth");
        start.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![],
            Hlvalue::Variable(vlist0.clone()),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "iter",
            vec![Hlvalue::Variable(iterable.clone())],
            Hlvalue::Variable(viter0.clone()),
        ));
        start.borrow_mut().closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(vlist0.clone()),
                    Hlvalue::Variable(viter0.clone()),
                ],
                Some(loopnext.clone()),
                None,
            )
            .into_ref(),
        ]);

        let item_from_next = Variable::named("item_from_next");
        let loop_iter_arg = loopnext.borrow().inputargs[1].clone();
        loopnext.borrow_mut().operations.push(SpaceOperation::new(
            "next",
            vec![loop_iter_arg],
            Hlvalue::Variable(item_from_next.clone()),
        ));
        loopnext.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        let stop_iteration = HostObject::new_class("pkg.StopIteration", vec![]);
        let loop_success_args = {
            let loopnext_b = loopnext.borrow();
            vec![
                loopnext_b.inputargs[0].clone(),
                loopnext_b.inputargs[1].clone(),
                Hlvalue::Variable(item_from_next.clone()),
            ]
        };
        let loop_stop_args = {
            let loopnext_b = loopnext.borrow();
            vec![loopnext_b.inputargs[0].clone()]
        };
        loopnext.borrow_mut().closeblock(vec![
            Link::new(loop_success_args, Some(append.clone()), None).into_ref(),
            Link::new(
                loop_stop_args,
                Some(after.clone()),
                Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                    stop_iteration,
                )))),
            )
            .into_ref(),
        ]);

        let append_inputargs = append.borrow().inputargs.clone();
        let list_in_append = append_inputargs[0].clone();
        let item_in_append = append_inputargs[2].clone();
        append.borrow_mut().operations.push(SpaceOperation::new(
            "getattr",
            vec![
                list_in_append.clone(),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("append"))),
            ],
            Hlvalue::Variable(vmeth.clone()),
        ));
        append.borrow_mut().operations.push(SpaceOperation::new(
            "simple_call",
            vec![Hlvalue::Variable(vmeth.clone()), item_in_append.clone()],
            Hlvalue::Variable(Variable::named("append_res")),
        ));
        let append_backedge_args = {
            let append_b = append.borrow();
            vec![append_b.inputargs[0].clone(), append_b.inputargs[1].clone()]
        };
        append.borrow_mut().closeblock(vec![
            Link::new(append_backedge_args, Some(loopnext.clone()), None).into_ref(),
        ]);

        let after_args = after.borrow().inputargs.clone();
        after.borrow_mut().closeblock(vec![
            Link::new(after_args, Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        detect_list_comprehension(&graph);

        let summary = crate::flowspace::model::summary(&graph);
        assert_eq!(summary.get("hint"), None);
    }

    #[test]
    fn elimination_collapses_single_empty_block() {
        // start -> empty (no ops, 1 formal arg) -> return
        // empty.exits[0].args = [empty.inputargs[0]]
        // After: start.exits[0] rewrites to point at return with the
        // substituted arg from start's link.args.
        let formal = Variable::new();
        let empty = Block::shared(vec![Hlvalue::Variable(formal.clone())]);
        let start = Block::shared(vec![]);

        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // start -> empty with args [int(42)]
        let actual = Hlvalue::Constant(crate::flowspace::model::Constant::new(
            crate::flowspace::model::ConstValue::Int(42),
        ));
        let start_link = Link::new(vec![actual], Some(empty.clone()), None).into_ref();
        start.borrow_mut().closeblock(vec![start_link]);

        // empty -> return with args [formal]
        let empty_link = Link::new(
            vec![Hlvalue::Variable(formal.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        empty.borrow_mut().closeblock(vec![empty_link]);

        eliminate_empty_blocks(&graph);

        // start -> returnblock now (empty collapsed).
        let s = start.borrow();
        let exit = s.exits[0].borrow();
        assert!(Rc::ptr_eq(exit.target.as_ref().unwrap(), &returnblock));
        // And the link args carry the substituted constant.
        assert!(matches!(
            exit.args[0].as_ref(),
            Some(Hlvalue::Constant(c)) if matches!(c.value, crate::flowspace::model::ConstValue::Int(42))
        ));
    }

    #[test]
    fn elimination_keeps_blocks_with_operations() {
        // start -> body (has an op) -> return — body is NOT empty.
        let v_in = Variable::new();
        let v_out = Variable::new();
        let body = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        body.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "op".to_string(),
                vec![Hlvalue::Variable(v_in.clone())],
                Hlvalue::Variable(v_out.clone()),
            ));
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(Variable::new())],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link]);

        let body_link = Link::new(
            vec![Hlvalue::Variable(v_out)],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.borrow_mut().closeblock(vec![body_link]);

        eliminate_empty_blocks(&graph);

        // start -> body unchanged.
        let s = start.borrow();
        let exit = s.exits[0].borrow();
        assert!(Rc::ptr_eq(exit.target.as_ref().unwrap(), &body));
    }

    #[test]
    fn join_blocks_merges_linear_chain() {
        // start -> mid -> return.
        // `mid` has one entry (from start) and non-empty exits ⇒ its
        // operations get folded into start, and start takes over mid's
        // exit to returnblock.
        let v_in = Variable::new();
        let v_out = Variable::new();
        let mid = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Variable(v_in.clone()),
                    Hlvalue::Constant(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::Int(1),
                    )),
                ],
                Hlvalue::Variable(v_out.clone()),
            ));

        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(mid.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(v_out.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        join_blocks(&graph);

        // `mid`'s op should now be on `start`, with v_in renamed to
        // start_in in the folded op's args.
        let s = start.borrow();
        assert_eq!(s.operations.len(), 1);
        assert_eq!(s.operations[0].opname, "int_add");
        assert!(matches!(
            &s.operations[0].args[0],
            Hlvalue::Variable(v) if *v == start_in
        ));
        // start's exit should now target returnblock (mid got folded).
        assert_eq!(s.exits.len(), 1);
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn join_blocks_keeps_diamond() {
        // start -> {a, b} — both target the same block `merge`. `merge`
        // has 2 entries ⇒ join_blocks must NOT fold either into start
        // (entrymap[merge] != 1).
        let v = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let merge = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // Boolean switch on `v`: False -> merge, True -> merge.
        start.borrow_mut().exitswitch = Some(Hlvalue::Variable(v.clone()));
        let left = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(merge.clone()),
            Some(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::Bool(false),
            ))),
        )
        .into_ref();
        let right = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(merge.clone()),
            Some(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::Bool(true),
            ))),
        )
        .into_ref();
        start
            .borrow_mut()
            .closeblock(vec![left.clone(), right.clone()]);
        left.borrow_mut().prevblock = Some(Rc::downgrade(&start));
        right.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let merge_link = Link::new(
            vec![merge.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        merge.borrow_mut().closeblock(vec![merge_link.clone()]);
        merge_link.borrow_mut().prevblock = Some(Rc::downgrade(&merge));

        let ops_before = merge.borrow().operations.len();
        join_blocks(&graph);
        let ops_after = merge.borrow().operations.len();

        // Start still has 2 exits, merge still has 1 op-list, returnblock unchanged.
        assert_eq!(start.borrow().exits.len(), 2);
        assert_eq!(ops_before, ops_after);
    }

    #[test]
    fn constfold_exitswitch_picks_matching_exit() {
        // start has a Bool(true) exitswitch with two exits:
        //   exitcase=False -> left
        //   exitcase=True  -> right (should survive)
        // After folding, start has a single exit with exitcase=None
        // targeting `right`.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let v = Variable::new();
        let left = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let right = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(C::new(CV::Bool(true))));
        let l_link = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(left.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(false)))),
        )
        .into_ref();
        let r_link = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(right.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(true)))),
        )
        .into_ref();
        start
            .borrow_mut()
            .closeblock(vec![l_link.clone(), r_link.clone()]);
        l_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));
        r_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        // left and right each exit to returnblock so the graph is
        // structurally closed (though checkgraph is not invoked here).
        let left_end = Link::new(
            vec![left.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        left.borrow_mut().closeblock(vec![left_end.clone()]);
        left_end.borrow_mut().prevblock = Some(Rc::downgrade(&left));
        let right_end = Link::new(
            vec![right.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        right.borrow_mut().closeblock(vec![right_end.clone()]);
        right_end.borrow_mut().prevblock = Some(Rc::downgrade(&right));

        constfold_exitswitch(&graph);

        let s = start.borrow();
        assert!(s.exitswitch.is_none());
        assert_eq!(s.exits.len(), 1);
        assert!(s.exits[0].borrow().exitcase.is_none());
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &right
        ));
    }

    #[test]
    fn remove_trivial_links_merges_empty_link() {
        // start -> mid (via trivial link, no args) -> return.
        // `mid` has 1 op and 1 entry/exit. The source→mid link has no
        // args, so remove_trivial_links folds mid into start.
        let v = Variable::new();
        let mid = Block::shared(vec![]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "int_zero",
                vec![],
                Hlvalue::Variable(v.clone()),
            ));
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // Trivial link start -> mid (no args — mid.inputargs also empty).
        let start_link = Link::new(vec![], Some(mid.clone()), None).into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        remove_trivial_links(&graph);

        let s = start.borrow();
        // mid's op now lives on start.
        assert_eq!(s.operations.len(), 1);
        assert_eq!(s.operations[0].opname, "int_zero");
        // start's exit now targets returnblock directly.
        assert_eq!(s.exits.len(), 1);
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn remove_identical_vars_dedupes_duplicate_phi_inputs() {
        // A block receives two input args whose phi nodes pull from
        // the same Constant on every incoming link. `remove_identical_vars`
        // should collapse them to a single inputarg.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let v_a = Variable::new();
        let v_b = Variable::new();
        let body = Block::shared(vec![
            Hlvalue::Variable(v_a.clone()),
            Hlvalue::Variable(v_b.clone()),
        ]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // start -> body with [Const(1), Const(1)] — both phis see the
        // same value so merge_identical_phi_nodes unifies v_a / v_b.
        let link_sb = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::Int(1))),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link_sb]);
        let link_br = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![link_br.clone()]);

        remove_identical_vars(&graph);

        // body should now have exactly one inputarg — the duplicate
        // was pruned, and the entrymap link's args list shrank to
        // match.
        assert_eq!(body.borrow().inputargs.len(), 1);
        assert_eq!(start.borrow().exits[0].borrow().args.len(), 1);
    }

    #[test]
    fn transform_xxxitem_rewrites_getitem_to_idx() {
        // Block raising `getitem` with an IndexError exit → opname
        // swaps to `getitem_idx`.
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HOST_ENV as HE, LAST_EXCEPTION, SpaceOperation as SO,
        };
        let index_err = HE.lookup_builtin("IndexError").unwrap();
        let v_in = Variable::new();
        let block = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        block.borrow_mut().operations.push(SO::new(
            "getitem",
            vec![
                Hlvalue::Variable(v_in.clone()),
                Hlvalue::Constant(C::new(CV::Int(0))),
            ],
            Hlvalue::Variable(Variable::new()),
        ));
        block.borrow_mut().exitswitch =
            Some(Hlvalue::Constant(C::new(CV::Atom(LAST_EXCEPTION.clone()))));
        let graph = FunctionGraph::new("f", block.clone());
        let ok_link = Link::new(
            vec![block.borrow().operations[0].result.clone()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let err_link = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::HostObject(index_err.clone()))),
                Hlvalue::Constant(C::new(CV::HostObject(index_err.clone()))),
            ],
            Some(graph.exceptblock.clone()),
            Some(Hlvalue::Constant(C::new(CV::HostObject(index_err.clone())))),
        )
        .into_ref();
        block.closeblock(vec![ok_link, err_link]);

        transform_xxxitem(&graph);

        assert_eq!(block.borrow().operations[0].opname, "getitem_idx");
    }

    #[test]
    fn remove_identical_vars_ssa_dedupes_duplicate_phis() {
        // Same fixture as remove_identical_vars_dedupes_duplicate_phi_inputs
        // — verify the SSA variant collapses constant-fed duplicate
        // phis to a single input.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let v_a = Variable::new();
        let v_b = Variable::new();
        let body = Block::shared(vec![
            Hlvalue::Variable(v_a.clone()),
            Hlvalue::Variable(v_b.clone()),
        ]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let link_sb = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::Int(1))),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link_sb]);
        let link_br = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![link_br]);

        remove_identical_vars_SSA(&graph);
        // body lost at least one inputarg.
        assert!(body.borrow().inputargs.len() < 2);
    }

    #[test]
    fn transform_dead_op_vars_drops_unused_pure_ops() {
        // Block: two unused `add` ops whose results don't flow into
        // link.args / exitswitch / return. transform_dead_op_vars
        // should drop them.
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation as SO};
        let v_live = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v_live.clone())]);
        // dead_a = add v_live, 1
        // dead_b = add v_live, 2
        let dead_a = Variable::new();
        let dead_b = Variable::new();
        start.borrow_mut().operations.push(SO::new(
            "add",
            vec![
                Hlvalue::Variable(v_live.clone()),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Hlvalue::Variable(dead_a.clone()),
        ));
        start.borrow_mut().operations.push(SO::new(
            "add",
            vec![
                Hlvalue::Variable(v_live.clone()),
                Hlvalue::Constant(C::new(CV::Int(2))),
            ],
            Hlvalue::Variable(dead_b.clone()),
        ));

        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Variable(v_live.clone())],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);

        transform_dead_op_vars(&graph, None);

        // Both dead ops are gone; block.operations is empty.
        assert!(start.borrow().operations.is_empty());
    }

    #[test]
    fn transform_dead_op_vars_removes_side_effect_free_direct_call() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation as SO};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator = &ann.translator;

        let callee_arg = signed_var();
        let callee_res = signed_var();
        let callee_start = Block::shared(vec![Hlvalue::Variable(callee_arg.clone())]);
        callee_start.borrow_mut().operations.push(SO::new(
            "int_add",
            vec![
                Hlvalue::Variable(callee_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(123))),
            ],
            Hlvalue::Variable(callee_res.clone()),
        ));
        let callee = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "callee",
            callee_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let callee_link = Link::new(
            vec![Hlvalue::Variable(callee_res.clone())],
            Some(callee.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        callee_start.closeblock(vec![callee_link]);

        let caller_arg = signed_var();
        let call_res = signed_var();
        let live_res = signed_var();
        let caller_start = Block::shared(vec![Hlvalue::Variable(caller_arg.clone())]);
        caller_start.borrow_mut().operations.push(SO::new(
            "direct_call",
            vec![
                Hlvalue::Constant(C::new(CV::LLPtr(Box::new(test_functionptr_void(&callee))))),
                Hlvalue::Variable(caller_arg.clone()),
            ],
            Hlvalue::Variable(call_res.clone()),
        ));
        caller_start.borrow_mut().operations.push(SO::new(
            "int_mul",
            vec![
                Hlvalue::Variable(caller_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(12))),
            ],
            Hlvalue::Variable(live_res.clone()),
        ));
        let caller = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "caller",
            caller_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let caller_link = Link::new(
            vec![Hlvalue::Variable(live_res.clone())],
            Some(caller.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        caller_start.closeblock(vec![caller_link]);

        translator.graphs.borrow_mut().push(callee.clone());
        translator.graphs.borrow_mut().push(caller.clone());
        let rtyper = translator.buildrtyper();
        rtyper.mark_already_seen(&callee_start);
        rtyper.mark_already_seen(&caller_start);

        transform_dead_op_vars(&caller.borrow(), Some(translator));

        let ops = &caller_start.borrow().operations;
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opname, "int_mul");
    }

    #[test]
    fn transform_dead_op_vars_removes_recursive_side_effect_free_direct_call() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation as SO};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator = &ann.translator;

        let rec_arg = signed_var();
        let rec_call_res = signed_var();
        let rec_live_res = signed_var();
        let rec_start = Block::shared(vec![Hlvalue::Variable(rec_arg.clone())]);
        let rec_graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "rec",
            rec_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let rec_ptr = Hlvalue::Constant(C::new(CV::LLPtr(Box::new(test_functionptr_void(
            &rec_graph,
        )))));
        rec_start.borrow_mut().operations.push(SO::new(
            "direct_call",
            vec![rec_ptr.clone(), Hlvalue::Variable(rec_arg.clone())],
            Hlvalue::Variable(rec_call_res.clone()),
        ));
        rec_start.borrow_mut().operations.push(SO::new(
            "int_add",
            vec![
                Hlvalue::Variable(rec_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Hlvalue::Variable(rec_live_res.clone()),
        ));
        let rec_link = Link::new(
            vec![Hlvalue::Variable(rec_live_res.clone())],
            Some(rec_graph.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        rec_start.closeblock(vec![rec_link]);

        let caller_arg = signed_var();
        let call_res = signed_var();
        let live_res = signed_var();
        let caller_start = Block::shared(vec![Hlvalue::Variable(caller_arg.clone())]);
        caller_start.borrow_mut().operations.push(SO::new(
            "direct_call",
            vec![rec_ptr, Hlvalue::Variable(caller_arg.clone())],
            Hlvalue::Variable(call_res.clone()),
        ));
        caller_start.borrow_mut().operations.push(SO::new(
            "int_add",
            vec![
                Hlvalue::Variable(caller_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(42))),
            ],
            Hlvalue::Variable(live_res.clone()),
        ));
        let caller = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "caller",
            caller_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let caller_link = Link::new(
            vec![Hlvalue::Variable(live_res.clone())],
            Some(caller.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        caller_start.closeblock(vec![caller_link]);

        translator.graphs.borrow_mut().push(rec_graph.clone());
        translator.graphs.borrow_mut().push(caller.clone());
        let rtyper = translator.buildrtyper();
        rtyper.mark_already_seen(&rec_start);
        rtyper.mark_already_seen(&caller_start);

        transform_dead_op_vars(&caller.borrow(), Some(translator));

        let ops = &caller_start.borrow().operations;
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opname, "int_add");
    }

    #[test]
    fn transform_dead_op_vars_removes_side_effect_free_indirect_call_family() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation as SO};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator = &ann.translator;

        let mk_pure_graph = |name: &str, delta: i64| {
            let arg = signed_var();
            let res = signed_var();
            let start = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
            start.borrow_mut().operations.push(SO::new(
                "int_add",
                vec![
                    Hlvalue::Variable(arg.clone()),
                    Hlvalue::Constant(C::new(CV::Int(delta))),
                ],
                Hlvalue::Variable(res.clone()),
            ));
            let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
                name,
                start.clone(),
                Hlvalue::Variable(signed_var()),
            )));
            let link = Link::new(
                vec![Hlvalue::Variable(res)],
                Some(graph.borrow().returnblock.clone()),
                None,
            )
            .into_ref();
            start.closeblock(vec![link]);
            (graph, start)
        };

        let (callee_a, callee_a_start) = mk_pure_graph("f1", 1);
        let (callee_b, callee_b_start) = mk_pure_graph("f2", 2);

        let wrapper_arg = signed_var();
        let runtime_funcptr = fn_ptr_var();
        let wrapper_call_res = signed_var();
        let wrapper_live_res = signed_var();
        let wrapper_start = Block::shared(vec![
            Hlvalue::Variable(wrapper_arg.clone()),
            Hlvalue::Variable(runtime_funcptr.clone()),
        ]);
        wrapper_start.borrow_mut().operations.push(SO::new(
            "indirect_call",
            vec![
                Hlvalue::Variable(runtime_funcptr),
                Hlvalue::Variable(wrapper_arg.clone()),
                Hlvalue::Constant(C::new(CV::Graphs(vec![
                    GraphKey::of(&callee_a).as_usize(),
                    GraphKey::of(&callee_b).as_usize(),
                ]))),
            ],
            Hlvalue::Variable(wrapper_call_res),
        ));
        wrapper_start.borrow_mut().operations.push(SO::new(
            "int_mul",
            vec![
                Hlvalue::Variable(wrapper_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(42))),
            ],
            Hlvalue::Variable(wrapper_live_res.clone()),
        ));
        let wrapper = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "wrapper",
            wrapper_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let wrapper_link = Link::new(
            vec![Hlvalue::Variable(wrapper_live_res.clone())],
            Some(wrapper.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        wrapper_start.closeblock(vec![wrapper_link]);

        let caller_arg = signed_var();
        let caller_runtime_funcptr = fn_ptr_var();
        let caller_call_res = signed_var();
        let caller_live_res = signed_var();
        let caller_start = Block::shared(vec![
            Hlvalue::Variable(caller_arg.clone()),
            Hlvalue::Variable(caller_runtime_funcptr.clone()),
        ]);
        caller_start.borrow_mut().operations.push(SO::new(
            "direct_call",
            vec![
                Hlvalue::Constant(C::new(CV::LLPtr(Box::new(test_functionptr_void(&wrapper))))),
                Hlvalue::Variable(caller_arg.clone()),
                Hlvalue::Variable(caller_runtime_funcptr),
            ],
            Hlvalue::Variable(caller_call_res),
        ));
        caller_start.borrow_mut().operations.push(SO::new(
            "int_add",
            vec![
                Hlvalue::Variable(caller_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(7))),
            ],
            Hlvalue::Variable(caller_live_res.clone()),
        ));
        let caller = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "caller",
            caller_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let caller_link = Link::new(
            vec![Hlvalue::Variable(caller_live_res.clone())],
            Some(caller.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        caller_start.closeblock(vec![caller_link]);

        translator.graphs.borrow_mut().push(callee_a);
        translator.graphs.borrow_mut().push(callee_b);
        translator.graphs.borrow_mut().push(wrapper.clone());
        translator.graphs.borrow_mut().push(caller.clone());
        let rtyper = translator.buildrtyper();
        rtyper.mark_already_seen(&callee_a_start);
        rtyper.mark_already_seen(&callee_b_start);
        rtyper.mark_already_seen(&wrapper_start);
        rtyper.mark_already_seen(&caller_start);

        transform_dead_op_vars(&caller.borrow(), Some(translator));

        let ops = &caller_start.borrow().operations;
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opname, "int_add");
    }

    #[test]
    fn transform_dead_op_vars_keeps_indirect_call_with_unknown_family() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation as SO};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator = &ann.translator;

        let wrapper_arg = signed_var();
        let runtime_funcptr = fn_ptr_var();
        let wrapper_call_res = signed_var();
        let wrapper_live_res = signed_var();
        let wrapper_start = Block::shared(vec![
            Hlvalue::Variable(wrapper_arg.clone()),
            Hlvalue::Variable(runtime_funcptr.clone()),
        ]);
        wrapper_start.borrow_mut().operations.push(SO::new(
            "indirect_call",
            vec![
                Hlvalue::Variable(runtime_funcptr),
                Hlvalue::Variable(wrapper_arg.clone()),
                Hlvalue::Constant(C::new(CV::None)),
            ],
            Hlvalue::Variable(wrapper_call_res),
        ));
        wrapper_start.borrow_mut().operations.push(SO::new(
            "int_add",
            vec![
                Hlvalue::Variable(wrapper_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(7))),
            ],
            Hlvalue::Variable(wrapper_live_res.clone()),
        ));
        let wrapper = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "wrapper",
            wrapper_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let wrapper_link = Link::new(
            vec![Hlvalue::Variable(wrapper_live_res.clone())],
            Some(wrapper.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        wrapper_start.closeblock(vec![wrapper_link]);

        let caller_arg = signed_var();
        let caller_runtime_funcptr = fn_ptr_var();
        let caller_call_res = signed_var();
        let caller_live_res = signed_var();
        let caller_start = Block::shared(vec![
            Hlvalue::Variable(caller_arg.clone()),
            Hlvalue::Variable(caller_runtime_funcptr.clone()),
        ]);
        caller_start.borrow_mut().operations.push(SO::new(
            "direct_call",
            vec![
                Hlvalue::Constant(C::new(CV::LLPtr(Box::new(test_functionptr_void(&wrapper))))),
                Hlvalue::Variable(caller_arg.clone()),
                Hlvalue::Variable(caller_runtime_funcptr),
            ],
            Hlvalue::Variable(caller_call_res),
        ));
        caller_start.borrow_mut().operations.push(SO::new(
            "int_mul",
            vec![
                Hlvalue::Variable(caller_arg.clone()),
                Hlvalue::Constant(C::new(CV::Int(3))),
            ],
            Hlvalue::Variable(caller_live_res.clone()),
        ));
        let caller = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "caller",
            caller_start.clone(),
            Hlvalue::Variable(signed_var()),
        )));
        let caller_link = Link::new(
            vec![Hlvalue::Variable(caller_live_res.clone())],
            Some(caller.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        caller_start.closeblock(vec![caller_link]);

        translator.graphs.borrow_mut().push(wrapper.clone());
        translator.graphs.borrow_mut().push(caller.clone());
        let rtyper = translator.buildrtyper();
        rtyper.mark_already_seen(&wrapper_start);
        rtyper.mark_already_seen(&caller_start);

        transform_dead_op_vars(&caller.borrow(), Some(translator));

        let ops = &caller_start.borrow().operations;
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opname, "direct_call");
        assert_eq!(ops[1].opname, "int_mul");
    }

    #[test]
    #[should_panic(
        expected = "transform_dead_op_vars_in_blocks: block_subset contains an unannotated/blocked block in the multi-graph path"
    )]
    fn transform_dead_op_vars_in_blocks_rejects_blocked_entries_in_multi_graph_path() {
        use crate::annotator::annrpython::RPythonAnnotator;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator = &ann.translator;

        let graph_a = Rc::new(RefCell::new(FunctionGraph::new(
            "a",
            Block::shared(vec![Hlvalue::Variable(Variable::new())]),
        )));
        let graph_b = Rc::new(RefCell::new(FunctionGraph::new(
            "b",
            Block::shared(vec![Hlvalue::Variable(Variable::new())]),
        )));
        let block_a = graph_a.borrow().startblock.clone();
        let block_b = graph_b.borrow().startblock.clone();

        translator.graphs.borrow_mut().push(graph_a.clone());
        translator.graphs.borrow_mut().push(graph_b.clone());

        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&block_a), Some(graph_a.clone()));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&block_b), None);

        transform_dead_op_vars_in_blocks(
            &[block_a, block_b],
            2,
            Some(translator),
            Some(graph_a.borrow().startblock.clone()),
        );
    }

    #[test]
    fn remove_assertion_errors_drops_assertion_branch() {
        // A canraise block with two exits:
        //   exit[0] (exitcase=None)        → returnblock (non-ex path)
        //   exit[1] (exitcase=AssertionError cls) → exceptblock
        //     with args=[Constant(AssertionError), Constant(instance)]
        // remove_assertion_errors should drop exit[1] and promote the
        // surviving exit — but because exit[0].exitcase was already
        // None and exits_len goes from 2→1, the exitswitch is cleared.
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HOST_ENV as HE, SpaceOperation as SO,
        };
        let assert_err = HE.lookup_builtin("AssertionError").unwrap();
        let v_in = Variable::new();
        let body = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        // A trivial op so canraise's raising_op check passes.
        body.borrow_mut().operations.push(SO::new(
            "simple_call",
            vec![Hlvalue::Variable(v_in.clone())],
            Hlvalue::Variable(Variable::new()),
        ));
        body.borrow_mut().exitswitch = Some(Hlvalue::Constant(C::new(CV::Atom(
            crate::flowspace::model::LAST_EXCEPTION.clone(),
        ))));

        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();
        let exceptblock = graph.exceptblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![start_link]);

        // Non-exceptional exit (exitcase=None) body → return.
        let ok_link = Link::new(
            vec![body.borrow().operations[0].result.clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        // Exceptional exit (exitcase=AssertionError) body → exceptblock.
        let err_link = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::HostObject(assert_err.clone()))),
                Hlvalue::Constant(C::new(CV::HostObject(assert_err.clone()))),
            ],
            Some(exceptblock.clone()),
            Some(Hlvalue::Constant(C::new(CV::HostObject(
                assert_err.clone(),
            )))),
        )
        .into_ref();
        body.closeblock(vec![ok_link.clone(), err_link.clone()]);

        remove_assertion_errors(&graph);

        // body should have exactly one exit (the non-exceptional one).
        assert_eq!(body.borrow().exits.len(), 1);
        assert!(body.borrow().exitswitch.is_none());
        assert!(Rc::ptr_eq(
            body.borrow().exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn transform_ovfcheck_rewrites_add_to_add_ovf() {
        // Block operations: `t = add v1 v2`; `r = simple_call(ovfcheck, t)`.
        // After transform_ovfcheck, the block should contain only
        // `r = add_ovf v1 v2` (t absorbed into r via renaming).
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HOST_ENV as HE, SpaceOperation as SO,
        };
        let ovf_sentinel = HE
            .import_module("rpython.rlib.rarithmetic")
            .and_then(|m| m.module_get("ovfcheck"))
            .unwrap();
        let v1 = Variable::new();
        let v2 = Variable::new();
        let t = Variable::new();
        let r = Variable::new();
        let start = Block::shared(vec![
            Hlvalue::Variable(v1.clone()),
            Hlvalue::Variable(v2.clone()),
        ]);
        start.borrow_mut().operations.push(SO::new(
            "add",
            vec![Hlvalue::Variable(v1.clone()), Hlvalue::Variable(v2.clone())],
            Hlvalue::Variable(t.clone()),
        ));
        start.borrow_mut().operations.push(SO::new(
            "simple_call",
            vec![
                Hlvalue::Constant(C::new(CV::HostObject(ovf_sentinel.clone()))),
                Hlvalue::Variable(t.clone()),
            ],
            Hlvalue::Variable(r.clone()),
        ));

        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Variable(r.clone())],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link.clone()]);

        transform_ovfcheck(&graph);

        let b = start.borrow();
        assert_eq!(b.operations.len(), 1);
        assert_eq!(b.operations[0].opname, "add_ovf");
        // The `return` link's arg was renamed from r → t.
        let link_arg = link.borrow().args[0].clone();
        assert!(matches!(link_arg, Some(Hlvalue::Variable(v)) if v == t));
    }

    #[test]
    fn simplify_graph_runs_all_passes_on_valid_graph() {
        // Smoke-test the driver end-to-end: minimum valid graph, all
        // 12 passes applied, checkgraph bookend. Must not panic.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("g", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(7)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        simplify_graph(&graph, None);
    }

    #[test]
    fn cleanup_graph_composes_passes_on_valid_graph() {
        // Smoke test — cleanup_graph on a minimum valid graph must
        // not panic (checkgraph bookends + the three inner passes are
        // all idempotent on a well-formed shape).
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("g", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(7)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);

        cleanup_graph(&graph);
    }

    #[test]
    fn remove_trivial_links_preserves_links_carrying_args() {
        // start -> mid with 1 arg. `link.args` is non-empty so
        // remove_trivial_links must NOT fold (args-less precondition
        // fails); mid's op stays on mid.
        let v_in = Variable::new();
        let mid = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "noop",
                vec![Hlvalue::Variable(v_in.clone())],
                Hlvalue::Variable(Variable::new()),
            ));
        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(mid.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(Variable::new())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        remove_trivial_links(&graph);

        // start still has no operations (not folded).
        assert!(start.borrow().operations.is_empty());
        // start's exit still targets mid (not rewired).
        assert!(Rc::ptr_eq(
            start.borrow().exits[0].borrow().target.as_ref().unwrap(),
            &mid
        ));
    }
}
