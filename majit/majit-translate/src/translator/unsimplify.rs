//! RPython `rpython/translator/unsimplify.py`.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{
    Block, BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link,
    LinkArg, LinkRef, SpaceOperation, Variable, checkgraph,
};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::translator::TranslationContext;

/// RPython `unsimplify.varoftype(concretetype, name=None)`
/// (unsimplify.py).
pub fn varoftype(concretetype: LowLevelType, name: Option<&str>) -> Variable {
    let var = name.map(Variable::named).unwrap_or_default();
    var.set_concretetype(Some(concretetype));
    var
}

/// RPython `unsimplify.insert_empty_block(link, newops=[])`
/// (unsimplify.py).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn insert_empty_block(link: &LinkRef, newops: Vec<SpaceOperation>) -> BlockRef {
    let link_args = link.borrow().args.clone();
    let target = link
        .borrow()
        .target
        .clone()
        .expect("insert_empty_block requires link.target");

    // `vars = {}` with insertion-ordered iteration to mirror CPython
    // dict semantics. `order` preserves first-insertion order; the
    // HashMap holds the (Variable, keep) pair keyed by Variable id.
    let mut vars_by_id: HashMap<u64, (Variable, bool)> = HashMap::new();
    let mut order: Vec<u64> = Vec::new();

    // `for v in link.args: vars[v] = True` — unconditional assign.
    for arg in &link_args {
        if let Some(Hlvalue::Variable(v)) = arg {
            let id = v.id();
            match vars_by_id.get_mut(&id) {
                Some((_existing, existing_keep)) => {
                    *existing_keep = true;
                }
                None => {
                    vars_by_id.insert(id, (v.clone(), true));
                    order.push(id);
                }
            }
        }
    }
    for op in &newops {
        // `for v in op.args: vars.setdefault(v, True)` — insert True
        // only when absent; never promote an existing False (e.g. a
        // prior op's result) back to True.
        for arg in &op.args {
            if let Hlvalue::Variable(v) = arg {
                let id = v.id();
                if let std::collections::hash_map::Entry::Vacant(e) = vars_by_id.entry(id) {
                    e.insert((v.clone(), true));
                    order.push(id);
                }
            }
        }
        // `vars[op.result] = False` — hard overwrite, demoting prior
        // True (e.g. from a link.arg or earlier op.arg) to False.
        if let Hlvalue::Variable(v) = &op.result {
            let id = v.id();
            match vars_by_id.get_mut(&id) {
                Some((_existing, existing_keep)) => {
                    *existing_keep = false;
                }
                None => {
                    vars_by_id.insert(id, (v.clone(), false));
                    order.push(id);
                }
            }
        }
    }

    let vars: Vec<Variable> = order
        .into_iter()
        .filter_map(|id| vars_by_id.remove(&id))
        .filter_map(|(var, keep)| keep.then_some(var))
        .collect();

    let mut mapping: HashMap<Variable, Hlvalue> = HashMap::new();
    for v in &vars {
        mapping.insert(v.clone(), Hlvalue::Variable(v.copy()));
    }

    let newblock = Block::shared(vars.iter().cloned().map(Hlvalue::Variable).collect());
    {
        let mut nb = newblock.borrow_mut();
        nb.operations.extend(newops);
    }
    // New outgoing edge (`Link::new`); GraphStore origin is None. The
    // incoming `link` is retargeted in place and keeps its lift stamp.
    let exit = Link::new_mergeable(link_args.clone(), Some(target), None).into_ref();
    newblock.closeblock(vec![exit]);
    newblock.borrow_mut().renamevariables(&mapping);

    let mut link_mut = link.borrow_mut();
    link_mut.args = vars.into_iter().map(Hlvalue::Variable).map(Some).collect();
    link_mut.target = Some(newblock.clone());
    drop(link_mut);

    newblock
}

/// RPython `unsimplify.insert_empty_startblock(graph)`
/// (unsimplify.py).
pub fn insert_empty_startblock(graph: &mut FunctionGraph) {
    let vars: Vec<Hlvalue> = graph
        .startblock
        .borrow()
        .inputargs
        .iter()
        .map(copy_hlvalue)
        .collect();
    let newblock = Block::shared(vars.clone());
    newblock.closeblock(vec![
        Link::new(vars, Some(graph.startblock.clone()), None).into_ref(),
    ]);
    graph.startblock = newblock;
}

/// RPython `unsimplify.starts_with_empty_block(graph)`
/// (unsimplify.py).
pub fn starts_with_empty_block(graph: &FunctionGraph) -> bool {
    let startblock = graph.startblock.borrow();
    startblock.operations.is_empty()
        && startblock.exitswitch.is_none()
        && startblock.exits.first().is_some_and(|link| {
            link.borrow().args == graph.getargs().into_iter().map(Some).collect::<Vec<_>>()
        })
}

/// RPython `unsimplify.call_initial_function(translator, initial_func, annhelper=None)`
/// (unsimplify.py).
///
/// Upstream receives a Python callable and turns it into a low-level
/// function pointer through `MixLevelHelperAnnotator.constfunc`. pyre's
/// source-translation path already represents such targets as `Constant`
/// funcptr values, so this Rust boundary takes that post-`constfunc`
/// carrier directly and preserves the graph rewrite itself line-by-line.
pub fn call_initial_function(translator: &TranslationContext, c_initial_func: Constant) {
    let entry_point = translator
        .entry_point_graph
        .borrow()
        .clone()
        .expect("call_initial_function requires translator.entry_point_graph");
    let (args, old_startblock) = {
        let graph = entry_point.borrow();
        (
            graph.getargs().iter().map(copy_hlvalue).collect::<Vec<_>>(),
            graph.startblock.clone(),
        )
    };
    let extrablock = Block::shared(args.clone());
    let v_none = varoftype(LowLevelType::Void, None);
    extrablock.borrow_mut().operations = vec![SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_initial_func)],
        Hlvalue::Variable(v_none),
    )];
    extrablock.closeblock(vec![Link::new(args, Some(old_startblock), None).into_ref()]);
    entry_point.borrow_mut().startblock = extrablock;
    checkgraph(&entry_point.borrow());
}

/// RPython `unsimplify.call_final_function(translator, final_func, annhelper=None)`
/// (unsimplify.py).
///
/// See [`call_initial_function`] for the Rust boundary around upstream's
/// `annhelper.constfunc` step.
pub fn call_final_function(translator: &TranslationContext, c_final_func: Constant) {
    let entry_point = translator
        .entry_point_graph
        .borrow()
        .clone()
        .expect("call_final_function requires translator.entry_point_graph");
    let (returnblock, returnvar, blocks) = {
        let graph = entry_point.borrow();
        (
            graph.returnblock.clone(),
            copy_hlvalue(&graph.getreturnvar()),
            graph.iterblocks(),
        )
    };
    let extrablock = Block::shared(vec![returnvar.clone()]);
    let v_none = varoftype(LowLevelType::Void, None);
    extrablock.borrow_mut().operations = vec![SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(c_final_func)],
        Hlvalue::Variable(v_none),
    )];
    extrablock.closeblock(vec![
        Link::new(vec![returnvar], Some(returnblock.clone()), None).into_ref(),
    ]);

    let return_key = BlockKey::of(&returnblock);
    let extrablock_key = BlockKey::of(&extrablock);
    for block in blocks {
        if BlockKey::of(&block) == extrablock_key {
            continue;
        }
        for link in &block.borrow().exits {
            let points_to_return = link
                .borrow()
                .target
                .as_ref()
                .is_some_and(|target| BlockKey::of(target) == return_key);
            if points_to_return {
                link.borrow_mut().target = Some(extrablock.clone());
            }
        }
    }
    checkgraph(&entry_point.borrow());
}

/// RPython `unsimplify.split_block(block, index, _forcelink=None)`
/// at `unsimplify.py`.
///
/// Returns a Link whose `prevblock` is `block` (truncated to operations
/// before `index`) and whose `target` is a fresh block holding the
/// moved-out operations and the original outgoing edges. Each Variable
/// produced before `index` and consumed at-or-after `index` is renamed
/// in the new block (a fresh `Variable::copy()`), and is passed through
/// the synthetic link's args.
///
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn split_block(block: &BlockRef, index: usize, _forcelink: Option<&[Hlvalue]>) -> LinkRef {
    {
        let b = block.borrow();
        assert!(
            index <= b.operations.len(),
            "split_block: index out of range (got {}, len={})",
            index,
            b.operations.len(),
        );
        if b.canraise() {
            assert!(
                index < b.operations.len(),
                "split_block: c_last_exception block requires index < len(ops)"
            );
        }
    }

    // varmap: each old Variable that is produced BEFORE `index` and
    // referenced AT or AFTER `index` (in moved ops, surviving exits,
    // or exitswitch). Ordered by first encounter so the synthetic
    // link's args mirror upstream's `linkargs = varmap.keys()` (CPython
    // dict iteration order = insertion order).
    let mut varmap_order: Vec<Variable> = Vec::new();
    let mut varmap: HashMap<Variable, Variable> = HashMap::new();
    let mut vars_produced_in_new_block: HashSet<Variable> = HashSet::new();

    // Step 1: walk the moved operations (>= index). For each op's
    // args, intern any Variable not already produced inside the moved
    // suffix; then mark op.result as produced.
    let mut moved_ops: Vec<SpaceOperation> = Vec::new();
    {
        let ops_after: Vec<SpaceOperation> = block.borrow().operations[index..].to_vec();
        for op in &ops_after {
            for arg in &op.args {
                if let Hlvalue::Variable(v) = arg {
                    intern_var(
                        v,
                        &mut varmap_order,
                        &mut varmap,
                        &vars_produced_in_new_block,
                    );
                }
            }
            // `op.replace(repl)`: substitute every Variable in the
            // cumulative varmap. Variables in
            // `vars_produced_in_new_block` were intentionally not
            // added to varmap, so they survive unchanged. Upstream
            // `vars_produced_in_new_block.add(op.result)` runs AFTER
            // `op.replace(repl)` — match that order so a moved op's
            // result is available to *subsequent* moved ops via
            // varmap-skip but is not pre-routed through varmap.
            let rename: HashMap<Variable, Hlvalue> = varmap
                .iter()
                .map(|(k, v)| (k.clone(), Hlvalue::Variable(v.clone())))
                .collect();
            let new_op = op.replace(&rename);
            if let Hlvalue::Variable(rv) = &op.result {
                vars_produced_in_new_block.insert(rv.clone());
            }
            moved_ops.push(new_op);
        }
    }

    // Step 2: detach the existing exits from the old block; they will
    // become the new block's exits after `link.args` rename. Upstream
    // `block.exits = None` is a sentinel — Rust clears the Vec.
    let links: Vec<LinkRef> = std::mem::take(&mut block.borrow_mut().exits);

    // Step 3: rename surviving exits' link args. Upstream skips
    // `link.last_exception` and `link.last_exc_value` because those
    // Variables are introduced by entering the link, not produced by
    // the predecessor block.
    for link in &links {
        let (last_exception, last_exc_value, l_args) = {
            let l = link.borrow();
            (
                l.last_exception.clone(),
                l.last_exc_value.clone(),
                l.args.clone(),
            )
        };
        let mut new_args: Vec<LinkArg> = Vec::with_capacity(l_args.len());
        for arg in l_args {
            let is_last_exc = match &arg {
                Some(a) => Some(a) == last_exception.as_ref() || Some(a) == last_exc_value.as_ref(),
                None => false,
            };
            if is_last_exc {
                new_args.push(arg);
                continue;
            }
            match arg {
                Some(Hlvalue::Variable(v)) => {
                    intern_var(
                        &v,
                        &mut varmap_order,
                        &mut varmap,
                        &vars_produced_in_new_block,
                    );
                    let renamed = varmap
                        .get(&v)
                        .cloned()
                        .map(Hlvalue::Variable)
                        .unwrap_or(Hlvalue::Variable(v));
                    new_args.push(Some(renamed));
                }
                other => new_args.push(other),
            }
        }
        link.borrow_mut().args = new_args;
    }

    // Step 4: rename block.exitswitch (becomes the new block's
    // exitswitch).
    let renamed_exitswitch = match block.borrow_mut().exitswitch.take() {
        None => None,
        Some(Hlvalue::Constant(c)) => Some(Hlvalue::Constant(c)),
        Some(Hlvalue::Variable(v)) => {
            intern_var(
                &v,
                &mut varmap_order,
                &mut varmap,
                &vars_produced_in_new_block,
            );
            let renamed = varmap
                .get(&v)
                .cloned()
                .map(Hlvalue::Variable)
                .unwrap_or(Hlvalue::Variable(v));
            Some(renamed)
        }
    };

    // Step 5: choose the synthetic link args. Upstream
    // `unsimplify.py` either uses `linkargs = varmap.keys()` or,
    // with `_forcelink`, the explicit list supplied by the jit-merge
    // rewrite. Missing varmap members are legal only for Void variables:
    // upstream recreates them in the target block with
    // `same_as(Constant(None, lltype.Void), varmap[v])`, inserted just
    // before the first moved op that consumes the recreated value.
    let link_args: Vec<Hlvalue> = if let Some(forcelink) = _forcelink {
        assert_eq!(
            index, 0,
            "unsimplify.py:87 split_block _forcelink requires index == 0"
        );
        let linkargs = forcelink.to_vec();
        for v in varmap_order.clone() {
            if linkargs
                .iter()
                .any(|arg| matches!(arg, Hlvalue::Variable(arg_v) if arg_v == &v))
            {
                continue;
            }
            if v.concretetype() != Some(LowLevelType::Void) {
                panic!(
                    "The variable {v:?} of type {:?} was not explicitly listed in _forcelink. \
                     This issue can be caused by a jitdriver.jit_merge_point() where some \
                     variable containing an int or str or instance is actually known to be \
                     constant, e.g. always 42.",
                    v.concretetype()
                );
            }
            let c = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ));
            let w = varmap
                .get(&v)
                .cloned()
                .expect("split_block: varmap_order member missing from varmap");
            let w_hv = Hlvalue::Variable(w.clone());
            let newop = SpaceOperation::new("same_as", vec![c], w_hv.clone());
            let insert_at = moved_ops
                .iter()
                .position(|op| op.args.iter().any(|arg| arg == &w_hv))
                .unwrap_or(moved_ops.len());
            moved_ops.insert(insert_at, newop);
        }
        linkargs
    } else {
        varmap_order
            .iter()
            .cloned()
            .map(Hlvalue::Variable)
            .collect()
    };

    // Step 6: assemble the new block. Its inputargs are
    // `[get_new_name(v) for v in linkargs]`; the default branch therefore
    // receives the varmap values in insertion order, and `_forcelink`
    // receives exactly the explicitly requested values, renamed as
    // upstream would do.
    let new_inputs: Vec<Hlvalue> = link_args
        .iter()
        .map(|arg| {
            get_new_name_for_split_input(
                arg,
                &mut varmap,
                &mut varmap_order,
                &vars_produced_in_new_block,
            )
        })
        .collect();
    let newblock = Block::shared(new_inputs);
    {
        let mut nb = newblock.borrow_mut();
        nb.operations = moved_ops;
        nb.exitswitch = renamed_exitswitch;
    }
    newblock.recloseblock(links);

    // Step 7: build the synthetic link from the truncated old block to
    // the new block. Its args are the original linkargs; the new
    // block's inputargs receive the renamed copies by position.
    let link = Link::new(link_args, Some(newblock), None).into_ref();

    // Step 8: truncate the old block and re-close it with the synthetic
    // link as its sole exit. exitswitch becomes None (it was moved to
    // the new block in step 4).
    {
        let mut b = block.borrow_mut();
        b.operations.truncate(index);
        b.exitswitch = None;
    }
    block.recloseblock(vec![link.clone()]);

    link
}

/// Helper for `split_block`. Adds `v` to `varmap` (with a fresh
/// `Variable::copy()`) unless it's already present or has been marked
/// as produced inside the moved suffix.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn intern_var(
    v: &Variable,
    varmap_order: &mut Vec<Variable>,
    varmap: &mut HashMap<Variable, Variable>,
    vars_produced_in_new_block: &HashSet<Variable>,
) {
    if vars_produced_in_new_block.contains(v) {
        return;
    }
    if !varmap.contains_key(v) {
        varmap.insert(v.clone(), v.copy());
        varmap_order.push(v.clone());
    }
}

fn copy_hlvalue(value: &Hlvalue) -> Hlvalue {
    match value {
        Hlvalue::Variable(v) => Hlvalue::Variable(v.copy()),
        Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
    }
}

/// `split_block`'s Rust equivalent of upstream `get_new_name(var)` for
/// the final `Block([get_new_name(v) for v in linkargs])` construction.
/// Constants pass through unchanged; Variables get their varmap copy,
/// creating one on demand for explicit `_forcelink` values that were not
/// otherwise needed by the moved suffix.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn get_new_name_for_split_input(
    arg: &Hlvalue,
    varmap: &mut HashMap<Variable, Variable>,
    varmap_order: &mut Vec<Variable>,
    vars_produced_in_new_block: &HashSet<Variable>,
) -> Hlvalue {
    match arg {
        Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
        Hlvalue::Variable(v) => {
            if vars_produced_in_new_block.contains(v) {
                return Hlvalue::Variable(v.clone());
            }
            if !varmap.contains_key(v) {
                varmap.insert(v.clone(), v.copy());
                varmap_order.push(v.clone());
            }
            Hlvalue::Variable(varmap.get(v).cloned().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn var(name: &str) -> Variable {
        Variable::named(name)
    }

    fn hv(v: &Variable) -> Hlvalue {
        Hlvalue::Variable(v.clone())
    }

    fn function_constant(name: &str) -> Constant {
        Constant::new(ConstValue::byte_str(name.as_bytes()))
    }

    fn simple_return_graph() -> (Rc<RefCell<FunctionGraph>>, Variable, Variable) {
        let arg = var("arg");
        let result = var("result");
        let start = Block::shared(vec![hv(&arg)]);
        let graph = FunctionGraph::with_return_var("entry", start.clone(), hv(&result));
        start.closeblock(vec![
            Link::new(vec![hv(&arg)], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        (Rc::new(RefCell::new(graph)), arg, result)
    }

    fn make_link(args: Vec<Variable>, target_inputargs: usize) -> LinkRef {
        let target = Block::shared(
            (0..target_inputargs)
                .map(|i| hv(&var(&format!("t{i}"))))
                .collect(),
        );
        Link::new(args.iter().map(hv).collect(), Some(target), None).into_ref()
    }

    /// `vars.setdefault(v, True)` on a variable whose prior `False`
    /// came from an earlier op.result must NOT promote it back to
    /// True. Covers a two-step conversion chain `a → c → d`, where
    /// `c` is an intermediate produced by the first op and consumed
    /// by the second: upstream keeps `c` out of the kept-vars set.
    #[test]
    fn op_arg_does_not_override_earlier_op_result_false_keep_flag() {
        let a = var("a");
        let b = var("b");
        let c = var("c");
        let d = var("d");
        let link = make_link(vec![a.clone(), b.clone()], 2);
        let newops = vec![
            SpaceOperation::new("convert1", vec![hv(&a)], hv(&c)),
            SpaceOperation::new("convert2", vec![hv(&c)], hv(&d)),
        ];

        let newblock = insert_empty_block(&link, newops);

        // Kept vars = [a, b]; `c` (earlier op.result) stays False even
        // after appearing as a later op.arg.
        assert_eq!(newblock.borrow().inputargs.len(), 2);
        assert_eq!(link.borrow().args.len(), 2);
    }

    /// `vars[op.result] = False` must hard-overwrite a prior True
    /// placed by `link.args`. Covers the case where an outgoing-link
    /// variable is reused as the result of an injected op.
    #[test]
    fn op_result_overwrites_link_arg_keep_flag_to_false() {
        let v = var("v");
        let w = var("w");
        let link = make_link(vec![v.clone()], 1);
        let newops = vec![SpaceOperation::new("produce_v", vec![hv(&w)], hv(&v))];

        let newblock = insert_empty_block(&link, newops);

        // `v` was True from link.args then False from op.result; drop
        // it. Only `w` (new from op.args) remains kept.
        assert_eq!(newblock.borrow().inputargs.len(), 1);
        assert_eq!(link.borrow().args.len(), 1);
    }

    #[test]
    fn varoftype_sets_concretetype_and_optional_name() {
        let unnamed = varoftype(LowLevelType::Void, None);
        assert_eq!(unnamed.concretetype(), Some(LowLevelType::Void));

        let named = varoftype(LowLevelType::Signed, Some("count"));
        assert_eq!(named.concretetype(), Some(LowLevelType::Signed));
        // `rename` runs `name.translate(PY_IDENTIFIER) + '_'`, so the
        // stored prefix carries the trailing underscore.
        assert_eq!(named.name_prefix(), "count_");
    }

    #[test]
    fn insert_empty_startblock_matches_unsimplify_shape() {
        let (graph, original_arg, _result) = simple_return_graph();
        let original_start = graph.borrow().startblock.clone();

        insert_empty_startblock(&mut graph.borrow_mut());

        let graph_b = graph.borrow();
        assert!(starts_with_empty_block(&graph_b));
        let new_start = graph_b.startblock.borrow();
        assert_eq!(new_start.inputargs.len(), 1);
        let Hlvalue::Variable(new_arg) = &new_start.inputargs[0] else {
            panic!("startblock input must be Variable")
        };
        assert_ne!(new_arg.id(), original_arg.id());
        assert_eq!(
            BlockKey::of(
                new_start.exits[0]
                    .borrow()
                    .target
                    .as_ref()
                    .expect("empty startblock must target old start")
            ),
            BlockKey::of(&original_start)
        );
    }

    #[test]
    fn call_initial_function_inserts_direct_call_before_entry_start() {
        let (graph, _arg, _result) = simple_return_graph();
        let translator = TranslationContext::new();
        *translator.entry_point_graph.borrow_mut() = Some(graph.clone());
        let old_start = graph.borrow().startblock.clone();

        call_initial_function(&translator, function_constant("init"));

        let graph_b = graph.borrow();
        let start_b = graph_b.startblock.borrow();
        assert_eq!(start_b.operations.len(), 1);
        assert_eq!(start_b.operations[0].opname, "direct_call");
        assert_eq!(
            BlockKey::of(
                start_b.exits[0]
                    .borrow()
                    .target
                    .as_ref()
                    .expect("initial block must target old start")
            ),
            BlockKey::of(&old_start)
        );
    }

    #[test]
    fn call_final_function_redirects_normal_return_through_finalizer_block() {
        let (graph, _arg, _result) = simple_return_graph();
        let translator = TranslationContext::new();
        *translator.entry_point_graph.borrow_mut() = Some(graph.clone());
        let returnblock = graph.borrow().returnblock.clone();

        call_final_function(&translator, function_constant("finish"));

        let graph_b = graph.borrow();
        let start_link = graph_b.startblock.borrow().exits[0].clone();
        let final_block = start_link
            .borrow()
            .target
            .clone()
            .expect("normal return must be redirected through final block");
        assert_ne!(BlockKey::of(&final_block), BlockKey::of(&returnblock));
        let final_b = final_block.borrow();
        assert_eq!(final_b.operations.len(), 1);
        assert_eq!(final_b.operations[0].opname, "direct_call");
        assert_eq!(
            BlockKey::of(
                final_b.exits[0]
                    .borrow()
                    .target
                    .as_ref()
                    .expect("final block must target returnblock")
            ),
            BlockKey::of(&returnblock)
        );
    }

    /// Helper: build a block with `inputargs` and `ops`, terminated
    /// with a single fall-through exit to a fresh return block whose
    /// inputargs match `link_args`.
    fn build_block(
        inputargs: Vec<Variable>,
        ops: Vec<SpaceOperation>,
        link_args: Vec<Variable>,
    ) -> (BlockRef, BlockRef) {
        let block = Block::shared(inputargs.iter().map(hv).collect());
        block.borrow_mut().operations = ops;
        let returnblock = Block::shared(link_args.iter().map(hv).collect::<Vec<_>>());
        let exit = Link::new(
            link_args.iter().map(hv).collect(),
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        block.closeblock(vec![exit]);
        (block, returnblock)
    }

    /// `split_block(block, 0)` with no ops to leave behind: the old
    /// block becomes empty and exits via the synthetic link, the new
    /// block carries every op, and varmap covers every input arg
    /// reachable in the moved suffix.
    #[test]
    fn split_at_zero_moves_all_ops_to_new_block() {
        let a = var("a");
        let b = var("b");
        let r = var("r");
        let ops = vec![SpaceOperation::new("int_add", vec![hv(&a), hv(&b)], hv(&r))];
        let (block, _ret) = build_block(vec![a.clone(), b.clone()], ops, vec![r.clone()]);

        let synth = split_block(&block, 0, None);

        // Old block has no ops, single exit = synth, no exitswitch.
        let b_borrowed = block.borrow();
        assert!(b_borrowed.operations.is_empty());
        assert_eq!(b_borrowed.exits.len(), 1);
        assert!(b_borrowed.exitswitch.is_none());
        drop(b_borrowed);

        // Synthetic link's args carry the OLD names (a, b — both used
        // by the moved op). Insertion order preserved.
        let synth_args = synth.borrow().args.clone();
        let arg_ids: Vec<u64> = synth_args
            .iter()
            .filter_map(|a| match a {
                Some(Hlvalue::Variable(v)) => Some(v.id()),
                _ => None,
            })
            .collect();
        assert_eq!(arg_ids, vec![a.id(), b.id()]);

        // The new block (synth.target) has fresh inputargs (different
        // ids) but the same op shape with renamed args.
        let target = synth.borrow().target.clone().unwrap();
        let target_b = target.borrow();
        assert_eq!(target_b.inputargs.len(), 2);
        for (input, original) in target_b.inputargs.iter().zip([&a, &b]) {
            let Hlvalue::Variable(v) = input else {
                panic!("expected Variable inputarg")
            };
            assert_ne!(v.id(), original.id(), "renamed");
        }
        assert_eq!(target_b.operations.len(), 1);
    }

    /// Splitting in the middle of a 2-op block produces a 1-op
    /// prefix and a 1-op suffix, with the intermediate variable
    /// passed through the synthetic link.
    #[test]
    fn split_in_middle_threads_intermediate_var_through_synth_link() {
        let a = var("a");
        let mid = var("mid");
        let r = var("r");
        let ops = vec![
            SpaceOperation::new("int_neg", vec![hv(&a)], hv(&mid)),
            SpaceOperation::new("int_neg", vec![hv(&mid)], hv(&r)),
        ];
        let (block, _ret) = build_block(vec![a.clone()], ops, vec![r.clone()]);

        let synth = split_block(&block, 1, None);

        // Old block has the first op only; new block has the second.
        assert_eq!(block.borrow().operations.len(), 1);
        let target = synth.borrow().target.clone().unwrap();
        assert_eq!(target.borrow().operations.len(), 1);

        // Synth link carries `mid` (the only Variable consumed by the
        // moved op); `a` was already consumed in the prefix.
        let synth_args = synth.borrow().args.clone();
        let arg_ids: Vec<u64> = synth_args
            .iter()
            .filter_map(|a| match a {
                Some(Hlvalue::Variable(v)) => Some(v.id()),
                _ => None,
            })
            .collect();
        assert_eq!(arg_ids, vec![mid.id()]);

        // The moved op's arg references the new block's renamed input,
        // not the original `mid`.
        let target_b = target.borrow();
        let Hlvalue::Variable(input0) = &target_b.inputargs[0] else {
            panic!()
        };
        let Hlvalue::Variable(moved_arg0) = &target_b.operations[0].args[0] else {
            panic!()
        };
        assert_eq!(moved_arg0.id(), input0.id());
        assert_ne!(moved_arg0.id(), mid.id());
    }

    /// `split_block(block, len(ops))` produces a synthetic link with
    /// no args (no Variables surface in the suffix or exits beyond
    /// what the prefix already produced and consumed). The new block
    /// holds zero ops but the original exits.
    #[test]
    fn split_at_end_yields_empty_suffix_block() {
        let a = var("a");
        let r = var("r");
        let ops = vec![SpaceOperation::new("int_neg", vec![hv(&a)], hv(&r))];
        let (block, _ret) = build_block(vec![a.clone()], ops, vec![r.clone()]);

        let synth = split_block(&block, 1, None);

        // Old block has the original op; new block has none.
        assert_eq!(block.borrow().operations.len(), 1);
        let target = synth.borrow().target.clone().unwrap();
        assert!(target.borrow().operations.is_empty());

        // Synth link carries `r` (referenced by the surviving exit).
        let synth_args = synth.borrow().args.clone();
        let arg_ids: Vec<u64> = synth_args
            .iter()
            .filter_map(|a| match a {
                Some(Hlvalue::Variable(v)) => Some(v.id()),
                _ => None,
            })
            .collect();
        assert_eq!(arg_ids, vec![r.id()]);
    }

    #[test]
    fn split_with_forcelink_recreates_missing_void_var() {
        let a = var("a");
        let void_v = var("void_v");
        void_v.set_concretetype(Some(LowLevelType::Void));
        let r = var("r");
        let ops = vec![SpaceOperation::new("use_void", vec![hv(&void_v)], hv(&r))];
        let (block, _ret) = build_block(vec![a.clone(), void_v.clone()], ops, vec![r.clone()]);

        let synth = split_block(&block, 0, Some(&[hv(&a)]));
        let target = synth.borrow().target.clone().unwrap();
        let target_b = target.borrow();

        // Only the explicit forcelink arg is passed through as a new
        // inputarg; the omitted Void var is recreated inside the suffix.
        assert_eq!(target_b.inputargs.len(), 1);
        assert_eq!(target_b.operations.len(), 2);
        assert_eq!(target_b.operations[0].opname, "same_as");
        assert!(matches!(
            &target_b.operations[0].args[0],
            Hlvalue::Constant(c)
                if c.value == ConstValue::None && c.concretetype == Some(LowLevelType::Void)
        ));
        let recreated = target_b.operations[0].result.clone();
        assert_eq!(target_b.operations[1].args[0], recreated);
    }

    #[test]
    #[should_panic(expected = "was not explicitly listed in _forcelink")]
    fn split_with_forcelink_rejects_missing_nonvoid_var() {
        let a = var("a");
        let b = var("b");
        b.set_concretetype(Some(LowLevelType::Signed));
        let r = var("r");
        let ops = vec![SpaceOperation::new("int_neg", vec![hv(&b)], hv(&r))];
        let (block, _ret) = build_block(vec![a.clone(), b.clone()], ops, vec![r.clone()]);
        let _ = split_block(&block, 0, Some(&[hv(&a)]));
    }
}
