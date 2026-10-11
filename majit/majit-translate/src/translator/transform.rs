//! RPython `rpython/translator/transform.py` — annotation-driven
//! graph transformations invoked from `RPythonAnnotator.simplify()`
//! (annrpython.py).
//!
//! Module header reproduced verbatim from upstream (transform.py:1-6):
//!
//! ```text
//! Flow Graph Transformation
//!
//! The difference between simplification and transformation is that
//! transformation is based on annotations; it runs after the annotator
//! completed.
//! ```
//!
//! Port coverage:
//!   - `checkgraphs` (transform.py)
//!   - `fully_annotated_blocks` (transform.py)
//!   - `transform_allocate` (transform.py)
//!   - `transform_extend_with_str_slice` (transform.py)
//!   - `transform_extend_with_char_count` (transform.py)
//!   - `transform_list_contains` (transform.py)
//!   - `transform_dead_op_vars` (transform.py) — forwards to
//!     [`crate::translator::simplify::transform_dead_op_vars_in_blocks`].
//!   - `transform_dead_code` + `cutoff_alwaysraising_block`.
//!   - `default_extra_passes` + `transform_graph`.
//!
//! `insert_ll_stackcheck` (transform.py) is rtyper-phase and
//! therefore out of scope for the annotator-phase port.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::model::{
    KnownType, SomeInstance, SomeObjectTrait, SomeValue, SomeValueTag, s_impossible_value, r#typeof,
};
use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, GraphKey, GraphRef, HOST_ENV, Hlvalue,
    HostObject, Link, LinkKey, LinkRef, SpaceOperation, Variable, checkgraph,
};

/// RPython `transform.py` — `checkgraphs(self, blocks)`.
///
/// ```python
/// def checkgraphs(self, blocks):
///     seen = set()
///     for block in blocks:
///         graph = self.annotated[block]
///         if graph not in seen:
///             checkgraph(graph)
///             seen.add(graph)
/// ```
///
/// Upstream indexes `ann.annotated[block]` directly here. By the time
/// `transform_graph()` runs on an explicit `block_subset`
/// (`complete_helpers()`'s path), `complete()` has already rejected
/// blocked (`False`) entries, so every supplied block must resolve to
/// its owning graph.
pub fn checkgraphs(ann: &RPythonAnnotator, blocks: &[BlockRef]) {
    // upstream: `seen = set()`.
    let mut seen: HashSet<GraphKey> = HashSet::new();
    let annotated = ann.annotated.borrow();
    for block in blocks {
        // upstream: `graph = self.annotated[block]`.
        let graph = annotated
            .get(&BlockKey::of(block))
            .and_then(|graph| graph.as_ref())
            .expect("checkgraphs: block_subset contains an unannotated/blocked block");
        // upstream: `if graph not in seen:`.
        let gkey = GraphKey::of(graph);
        if seen.insert(gkey) {
            // upstream: `checkgraph(graph)`.
            checkgraph(&graph.borrow());
        }
    }
}

/// RPython `transform.py` — `fully_annotated_blocks(self)`.
///
/// ```python
/// def fully_annotated_blocks(self):
///     """Ignore blocked blocks."""
///     for block, is_annotated in self.annotated.iteritems():
///         if is_annotated:
///             yield block
/// ```
///
/// Upstream returns a generator; the Rust port materialises the
/// filtered set into a `Vec<BlockRef>` since `transform_graph`'s
/// caller converts the generator into a dict immediately after.
pub fn fully_annotated_blocks(ann: &RPythonAnnotator) -> Vec<BlockRef> {
    let annotated = ann.annotated.borrow();
    let all_blocks = ann.all_blocks.borrow();
    let mut result = Vec::new();
    for (key, value) in annotated.iter() {
        // upstream `if is_annotated:` — `annotated[block] == False`
        // lowers to `Option<GraphRef>::None` in the Rust port.
        if value.is_none() {
            continue;
        }
        let block = all_blocks
            .get(key)
            .expect("fully_annotated_blocks: annotated block missing from all_blocks");
        result.push(block.clone());
    }
    result
}

/// Signature of an `extra_pass` function threaded through
/// `transform_graph`. Upstream calls `pass_(ann, block_subset)` where
/// `block_subset` is a Python dict keyed by Block; the Rust port takes
/// a `&[BlockRef]` slice since the callers only iterate it (no dict
/// lookups downstream).
pub(crate) type TransformPass = fn(&RPythonAnnotator, &[BlockRef]);

/// RPython `transform.py` — `default_extra_passes = [...]`.
///
/// ```python
/// default_extra_passes = [
///     transform_allocate,
///     transform_extend_with_str_slice,
///     transform_extend_with_char_count,
///     transform_list_contains,
/// ]
/// ```
///
/// Keep the default pass list identical to upstream. These rewrites run
/// after annotation completed, so `transform_graph` must not silently
/// drop them just because later phases have not yet ported the lowered
/// opnames.
pub(crate) const DEFAULT_EXTRA_PASSES: &[TransformPass] = &[
    transform_allocate,
    transform_extend_with_str_slice,
    transform_extend_with_char_count,
    transform_list_contains,
];

/// RPython `transform.py:253-272` — `transform_graph(ann, extra_passes,
/// block_subset)`.
///
/// ```python
/// def transform_graph(ann, extra_passes=None, block_subset=None):
///     """Apply set of transformations available."""
///     # WARNING: this produces incorrect results if the graph has been
///     #          modified by t.simplify() after it had been annotated.
///     if extra_passes is None:
///         extra_passes = default_extra_passes
///     if block_subset is None:
///         block_subset = fully_annotated_blocks(ann)
///     if not isinstance(block_subset, dict):
///         block_subset = dict.fromkeys(block_subset)
///     if ann.translator:
///         checkgraphs(ann, block_subset)
///     transform_dead_code(ann, block_subset)
///     for pass_ in extra_passes:
///         pass_(ann, block_subset)
///     # do this last, after the previous transformations had a
///     # chance to remove dependency on certain variables
///     transform_dead_op_vars(ann, block_subset)
///     if ann.translator:
///         checkgraphs(ann, block_subset)
/// ```
///
/// The `ann.translator` guard collapses to an unconditional invocation
/// because the Rust port always owns a `TranslationContext`
/// (annrpython.py default-constructs one when absent).
pub fn transform_graph(
    ann: &RPythonAnnotator,
    extra_passes: Option<&[TransformPass]>,
    block_subset: Option<&[BlockRef]>,
) {
    // upstream: `if extra_passes is None: extra_passes = default_extra_passes`.
    let extra_passes = extra_passes.unwrap_or(DEFAULT_EXTRA_PASSES);

    // upstream: `if block_subset is None: block_subset = fully_annotated_blocks(ann)` +
    // `block_subset = dict.fromkeys(block_subset)`. The Rust port
    // materialises both paths into a `Vec<BlockRef>` with duplicates
    // collapsed via BlockKey identity.
    let subset_owned: Vec<BlockRef> = match block_subset {
        Some(b) => b.to_vec(),
        None => fully_annotated_blocks(ann),
    };
    let subset = dedupe_blocks(&subset_owned);

    // upstream: `if ann.translator: checkgraphs(ann, block_subset)`.
    checkgraphs(ann, &subset);
    // upstream: `transform_dead_code(ann, block_subset)`.
    transform_dead_code(ann, &subset);
    // upstream: `for pass_ in extra_passes: pass_(ann, block_subset)`.
    for pass_ in extra_passes {
        pass_(ann, &subset);
    }
    // upstream: do this last, after previous passes had a chance to
    // remove dependency on certain variables.
    transform_dead_op_vars(ann, &subset);
    // upstream: `if ann.translator: checkgraphs(ann, block_subset)`.
    checkgraphs(ann, &subset);
}

/// Collapse duplicates by block identity, mirroring upstream's
/// `dict.fromkeys(block_subset)` de-duplication step (transform.py).
fn dedupe_blocks(blocks: &[BlockRef]) -> Vec<BlockRef> {
    let mut seen: HashSet<BlockKey> = HashSet::new();
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        if seen.insert(BlockKey::of(b)) {
            out.push(b.clone());
        }
    }
    out
}

/// RPython `transform_dead_code(self, block_subset)` from `transform.py`.
///
/// ```python
/// def transform_dead_code(self, block_subset):
///     """Remove dead code: these are the blocks that are not annotated at all
///     because the annotation considered that no conditional jump could reach
///     them."""
///     for block in block_subset:
///         for link in block.exits:
///             if link not in self.links_followed:
///                 lst = list(block.exits)
///                 lst.remove(link)
///                 block.exits = tuple(lst)
///                 if not block.exits:
///                     cutoff_alwaysraising_block(self, block)
///                 elif block.canraise:
///                     if block.exits[0].exitcase is not None:
///                         # killed the non-exceptional path!
///                         cutoff_alwaysraising_block(self, block)
///                 if len(block.exits) == 1:
///                     block.exitswitch = None
///                     block.exits[0].exitcase = None
/// ```
pub fn transform_dead_code(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        // upstream: `for link in block.exits:`. The mutation below
        // happens to `block.exits` while we iterate, so snapshot first
        // (matches upstream's implicit `list(...)` copy through
        // identity — Python lets you mutate a list during iteration
        // provided you don't resize, but `block.exits = tuple(lst)`
        // rebinds the attribute which changes subsequent comparisons).
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.to_vec();
        for link in exits_snapshot {
            let followed = ann
                .links_followed
                .borrow()
                .contains_key(&LinkKey::of(&link));
            if followed {
                continue;
            }
            // upstream: remove this link from block.exits.
            {
                let mut blk = block.borrow_mut();
                blk.exits.retain(|l| !Rc::ptr_eq(l, &link));
            }
            let no_exits_left = block.borrow().exits.is_empty();
            let canraise = block.borrow().canraise();
            let first_exitcase_nontrivial = {
                let blk = block.borrow();
                blk.exits
                    .first()
                    .map(|l| l.borrow().exitcase.is_some())
                    .unwrap_or(false)
            };

            // upstream: `if not block.exits: cutoff_alwaysraising_block(...)`.
            if no_exits_left {
                cutoff_alwaysraising_block(ann, block);
            } else if canraise && first_exitcase_nontrivial {
                // upstream: `killed the non-exceptional path!`.
                cutoff_alwaysraising_block(ann, block);
            }

            // upstream: `if len(block.exits) == 1: block.exitswitch =
            // None; block.exits[0].exitcase = None`.
            if block.borrow().exits.len() == 1 {
                let mut blk = block.borrow_mut();
                blk.exitswitch = None;
                if let Some(surviving) = blk.exits.first() {
                    surviving.borrow_mut().exitcase = None;
                }
            }
        }
    }
}

/// Stable, process-comparable identification for dead-code cutoff diagnostics.
pub(crate) struct CutoffBlockTrace {
    graph_name: String,
    n: usize,
    total: usize,
    block_position: Option<usize>,
    operations: usize,
}

impl std::fmt::Display for CutoffBlockTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "graph={:?} n={} total={} block={} ops={}",
            self.graph_name,
            self.n,
            self.total,
            self.block_position
                .map(|position| position.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.operations,
        )
    }
}

fn cutoff_block_trace_with_counts(
    ann: &RPythonAnnotator,
    block: &BlockRef,
    n: usize,
    total: usize,
) -> CutoffBlockTrace {
    let graph = ann
        .annotated
        .borrow()
        .get(&BlockKey::of(block))
        .and_then(|graph| graph.clone());
    let (graph_name, block_position) = match graph {
        Some(graph) => {
            let graph = graph.borrow();
            let position = graph
                .iterblocks()
                .iter()
                .position(|candidate| Rc::ptr_eq(candidate, block));
            (graph.name.clone(), position)
        }
        None => ("<unowned>".to_string(), None),
    };
    CutoffBlockTrace {
        graph_name,
        n,
        total,
        block_position,
        operations: total,
    }
}

/// Capture the same block identification used by `[DTRACE-CUT]` so the
/// per-block panic isolator can name a failed cutoff without using pointers.
pub(crate) fn cutoff_block_trace(ann: &RPythonAnnotator, block: &BlockRef) -> CutoffBlockTrace {
    let (n, total) = {
        let blk = block.borrow();
        let n = blk
            .operations
            .iter()
            .position(|op| match &op.result {
                Hlvalue::Variable(v) => v.annotation.borrow().is_none(),
                Hlvalue::Constant(_) => false,
            })
            .unwrap_or(blk.operations.len());
        (n, blk.operations.len())
    };
    cutoff_block_trace_with_counts(ann, block, n, total)
}

/// RPython `transform.py` — `cutoff_alwaysraising_block(self, block)`.
///
/// ```python
/// def cutoff_alwaysraising_block(self, block):
///     "Fix a block whose end can never be reached at run-time."
///     can_succeed    = [op for op in block.operations
///                          if op.result.annotation is not None]
///     cannot_succeed = [op for op in block.operations
///                          if op.result.annotation is None]
///     n = len(can_succeed)
///     assert can_succeed == block.operations[:n]
///     assert cannot_succeed == block.operations[n:]
///     assert 0 <= n < len(block.operations)
///     del block.operations[n+1:]
///     self.setbinding(block.operations[n].result, annmodel.s_ImpossibleValue)
///     graph = self.annotated[block]
///     msg = "Call to %r should have raised an exception" % (getattr(graph, 'func', None),)
///     c1 = Constant(AssertionError)
///     c2 = Constant(AssertionError(msg))
///     errlink = Link([c1, c2], graph.exceptblock)
///     block.recloseblock(errlink, *block.exits)
///     self.links_followed[errlink] = True
///     etype, evalue = graph.exceptblock.inputargs
///     s_type = annmodel.SomeTypeOf([evalue])
///     s_value = annmodel.SomeInstance(self.bookkeeper.getuniqueclassdef(Exception))
///     self.setbinding(etype, s_type)
///     self.setbinding(evalue, s_value)
///     self.bookkeeper.getuniqueclassdef(AssertionError)
/// ```
pub fn cutoff_alwaysraising_block(ann: &RPythonAnnotator, block: &BlockRef) {
    // upstream: partition block.operations by whether op.result has
    // an annotation. SpaceOperation.result is Hlvalue — only the
    // Variable variant carries `annotation`.
    let (n, total) = {
        let blk = block.borrow();
        let can_succeed: Vec<_> = blk
            .operations
            .iter()
            .map(|op| match &op.result {
                Hlvalue::Variable(v) => v.annotation.borrow().is_some(),
                Hlvalue::Constant(_) => true,
            })
            .collect();
        let n = can_succeed.iter().filter(|annotated| **annotated).count();
        // transform.py::cutoff_alwaysraising_block checks both partitions:
        // successfully annotated operations must precede the failing tail.
        assert!(
            can_succeed[..n].iter().all(|annotated| *annotated),
            "cutoff_alwaysraising_block: annotated operations must form a prefix"
        );
        assert!(
            can_succeed[n..].iter().all(|annotated| !*annotated),
            "cutoff_alwaysraising_block: unannotated operations must form a suffix"
        );
        (n, blk.operations.len())
    };
    if crate::determinism_trace_enabled() {
        let trace = cutoff_block_trace_with_counts(ann, block, n, total);
        eprintln!("[DTRACE-CUT] {trace}");
    }
    // transform.py::cutoff_alwaysraising_block: a successful operation prefix
    // must end at an operation that cannot succeed. Missing followed links
    // or preseeded annotations are not an alternative successful cutoff.
    assert!(
        n < total,
        "cutoff_alwaysraising_block: no failing op (n={n}, total={total})"
    );

    // upstream: `del block.operations[n+1:]`.
    block.borrow_mut().operations.truncate(n + 1);
    // upstream: `self.setbinding(block.operations[n].result,
    // annmodel.s_ImpossibleValue)`. Same mutation pattern as
    // annrpython.rs.
    {
        let mut blk = block.borrow_mut();
        if let Hlvalue::Variable(v) = &mut blk.operations[n].result {
            ann.setbinding(v, s_impossible_value());
        }
    }

    // upstream: `graph = self.annotated[block]`.
    let graph: GraphRef = {
        let annotated = ann.annotated.borrow();
        annotated
            .get(&BlockKey::of(block))
            .and_then(|g| g.clone())
            .expect("cutoff_alwaysraising_block: block not annotated")
    };

    // upstream: `msg = "Call to %r should have raised an exception" %
    // (getattr(graph, 'func', None),)`.
    let func_repr = graph
        .borrow()
        .func
        .as_ref()
        .map(|f| format!("{:?}", f))
        .unwrap_or_else(|| "None".to_string());
    let msg = format!("Call to {func_repr} should have raised an exception");

    // upstream: `c1 = Constant(AssertionError); c2 = Constant(AssertionError(msg))`.
    let assert_err_class = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let c1 = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
        assert_err_class.clone(),
    )));
    let err_instance =
        HostObject::new_instance(assert_err_class.clone(), vec![ConstValue::byte_str(msg)]);
    let c2 = Hlvalue::Constant(Constant::new(ConstValue::HostObject(err_instance)));

    // upstream: `errlink = Link([c1, c2], graph.exceptblock)`.
    let exceptblock = graph.borrow().exceptblock.clone();
    let errlink = Link::new(vec![c1, c2], Some(exceptblock.clone()), None).into_ref();

    // upstream: `block.recloseblock(errlink, *block.exits)`.
    let mut new_exits: Vec<LinkRef> = vec![errlink.clone()];
    let existing_exits: Vec<LinkRef> = block.borrow().exits.to_vec();
    new_exits.extend(existing_exits);
    block.recloseblock(new_exits);

    // upstream: `self.links_followed[errlink] = True`.
    ann.links_followed
        .borrow_mut()
        .insert(LinkKey::of(&errlink), errlink.clone());

    // upstream: `etype, evalue = graph.exceptblock.inputargs`.
    let (etype_rc, evalue_rc) = {
        let ex = exceptblock.borrow();
        assert_eq!(ex.inputargs.len(), 2, "exceptblock inputargs len != 2");
        let t = match &ex.inputargs[0] {
            Hlvalue::Variable(v) => Rc::new(v.clone()),
            _ => panic!("exceptblock.inputargs[0] is not a Variable"),
        };
        let v = match &ex.inputargs[1] {
            Hlvalue::Variable(v) => Rc::new(v.clone()),
            _ => panic!("exceptblock.inputargs[1] is not a Variable"),
        };
        (t, v)
    };
    // upstream: `s_type = annmodel.SomeTypeOf([evalue])`.
    let s_type = r#typeof(std::slice::from_ref(&evalue_rc));
    // upstream: `s_value = annmodel.SomeInstance(
    //     self.bookkeeper.getuniqueclassdef(Exception))`.
    let exc_class = HOST_ENV
        .lookup_builtin("Exception")
        .expect("HOST_ENV missing Exception");
    let exc_classdef = ann
        .bookkeeper
        .getuniqueclassdef(&exc_class)
        .expect("bookkeeper.getuniqueclassdef(Exception) failed");
    let s_value = SomeValue::Instance(SomeInstance::new(
        Some(exc_classdef),
        false,
        Default::default(),
    ));

    // upstream: `self.setbinding(etype, s_type); self.setbinding(evalue, s_value)`.
    {
        let mut ex = exceptblock.borrow_mut();
        if let Hlvalue::Variable(etype_var) = &mut ex.inputargs[0] {
            ann.setbinding(etype_var, s_type);
        }
        if let Hlvalue::Variable(evalue_var) = &mut ex.inputargs[1] {
            ann.setbinding(evalue_var, s_value);
        }
    }
    // Drop the detached Rc<Variable>s — they were only needed to
    // synthesise s_type via typeof.
    drop(etype_rc);
    drop(evalue_rc);

    // upstream: `self.bookkeeper.getuniqueclassdef(AssertionError)`.
    // Side-effect only — register the class so later flow sees it.
    let _ = ann.bookkeeper.getuniqueclassdef(&assert_err_class);
}

/// RPython `transform.py` — `transform_dead_op_vars(ann, blocks)`.
///
/// ```python
/// def transform_dead_op_vars(ann, block_subset):
///     from rpython.translator.simplify import transform_dead_op_vars_in_blocks
///     transform_dead_op_vars_in_blocks(block_subset, ann.translator.graphs,
///             ann.translator)
/// ```
///
/// Delegates to `simplify::transform_dead_op_vars_in_blocks` with the
/// annotator's translator hooked up. Upstream passes the block subset
/// directly — `block_subset` may be smaller than any one graph's full
/// block list (e.g. `complete_helpers()` threads `self.added_blocks`
/// only). The pass mutates only those blocks; `dependencies` /
/// `read_vars` still flow correctly because cross-subset link targets
/// contribute `read_vars` for their inputargs via the `target not in
/// set_of_blocks` branch.
pub fn transform_dead_op_vars(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    use crate::translator::simplify;
    let translator = &ann.translator;
    let graphs_len = translator.graphs.borrow().len();
    let single_graph_startblock = translator
        .graphs
        .borrow()
        .first()
        .map(|graph| graph.borrow().startblock.clone());
    simplify::transform_dead_op_vars_in_blocks(
        block_subset,
        graphs_len,
        Some(translator),
        single_graph_startblock,
    );
}

/// RPython `transform.py` — `transform_allocate(self, block_subset)`.
///
/// ```python
/// def transform_allocate(self, block_subset):
///     """Transforms [a] * b to alloc_and_set(b, a) where b is int."""
///     for block in block_subset:
///         length1_lists = {}   # maps 'c' to 'a', in the above notation
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if (op.opname == 'newlist' and
///                 len(op.args) == 1):
///                 length1_lists[op.result] = op.args[0]
///             elif (op.opname == 'mul' and
///                   op.args[0] in length1_lists):
///                 new_op = SpaceOperation('alloc_and_set',
///                                         (op.args[1], length1_lists[op.args[0]]),
///                                         op.result)
///                 block.operations[i] = new_op
/// ```
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn transform_allocate(_ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        // upstream: maps `c` (newlist result) → `a` (newlist's only arg).
        let mut length1_lists: HashMap<Variable, Hlvalue> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            // upstream: `op = block.operations[i]`.
            let op = block.borrow().operations[i].clone();
            if op.opname == "newlist" && op.args.len() == 1 {
                if let Hlvalue::Variable(result_v) = &op.result {
                    length1_lists.insert(result_v.clone(), op.args[0].clone());
                }
            } else if op.opname == "mul" {
                // upstream: `op.args[0] in length1_lists`.
                let Some(Hlvalue::Variable(first_var)) = op.args.first() else {
                    continue;
                };
                let Some(a_value) = length1_lists.get(first_var) else {
                    continue;
                };
                let Some(b_value) = op.args.get(1) else {
                    continue;
                };
                // upstream: `SpaceOperation('alloc_and_set', (op.args[1], length1_lists[op.args[0]]), op.result)`.
                let new_op = SpaceOperation::with_offset(
                    "alloc_and_set",
                    vec![b_value.clone(), a_value.clone()],
                    op.result.clone(),
                    op.offset,
                );
                block.borrow_mut().operations[i] = new_op;
            }
        }
    }
}

/// RPython `transform.py` — `transform_extend_with_str_slice(self, block_subset)`.
///
/// ```python
/// def transform_extend_with_str_slice(self, block_subset):
///     """Transforms lst += string[x:y] to extend_with_str_slice"""
///     for block in block_subset:
///         slice_sources = {}
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if (op.opname == 'getslice' and
///                 self.gettype(op.args[0]) is str):
///                 slice_sources[op.result] = op.args
///             elif (op.opname == 'inplace_add' and
///                   op.args[1] in slice_sources and
///                   self.gettype(op.args[0]) is list):
///                 v_string, v_x, v_y = slice_sources[op.args[1]]
///                 new_op = SpaceOperation('extend_with_str_slice',
///                                         [op.args[0], v_x, v_y, v_string],
///                                         op.result)
///                 block.operations[i] = new_op
/// ```
///
/// `self.gettype(v) is str` routes through `ann.gettype(v) ==
/// KnownType::Str`, which — like upstream — matches both `SomeString`
/// and `SomeChar` because both set `knowntype = str`. The equivalent
/// `is list` check goes through `SomeValue::List(_)` since the Rust
/// port's [`KnownType`] enum collapses SomeList to [`KnownType::Other`]
/// (list variant pending).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn transform_extend_with_str_slice(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        // upstream: maps `b` (getslice result Variable) → [string, x, y].
        let mut slice_sources: HashMap<Variable, Vec<Hlvalue>> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            let op = block.borrow().operations[i].clone();
            if op.opname == "getslice" && op.args.len() >= 3 {
                // upstream: `self.gettype(op.args[0]) is str` — true
                // for both SomeString and SomeChar (both set
                // knowntype = str).
                if ann.gettype(&op.args[0]) == KnownType::Str
                    && let Hlvalue::Variable(result_v) = &op.result
                {
                    slice_sources.insert(result_v.clone(), op.args.clone());
                }
            } else if op.opname == "inplace_add" && op.args.len() == 2 {
                // upstream: `op.args[1] in slice_sources`.
                let Hlvalue::Variable(arg1_v) = &op.args[1] else {
                    continue;
                };
                let Some(source_args) = slice_sources.get(arg1_v) else {
                    continue;
                };
                // upstream: `self.gettype(op.args[0]) is list`.
                if !matches!(ann.annotation(&op.args[0]), Some(SomeValue::List(_))) {
                    continue;
                }
                // upstream: `v_string, v_x, v_y = slice_sources[op.args[1]]`.
                let v_string = source_args[0].clone();
                let v_x = source_args[1].clone();
                let v_y = source_args[2].clone();
                let new_op = SpaceOperation::with_offset(
                    "extend_with_str_slice",
                    vec![op.args[0].clone(), v_x, v_y, v_string],
                    op.result.clone(),
                    op.offset,
                );
                block.borrow_mut().operations[i] = new_op;
            }
        }
    }
}

/// RPython `transform.py` — `transform_extend_with_char_count(self, block_subset)`.
///
/// ```python
/// def transform_extend_with_char_count(self, block_subset):
///     """Transforms lst += char*count to extend_with_char_count"""
///     for block in block_subset:
///         mul_sources = {}
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if op.opname == 'mul':
///                 s0 = self.annotation(op.args[0])
///                 s1 = self.annotation(op.args[1])
///                 if (isinstance(s0, annmodel.SomeChar) and
///                     isinstance(s1, annmodel.SomeInteger)):
///                     mul_sources[op.result] = op.args[0], op.args[1]
///                 elif (isinstance(s1, annmodel.SomeChar) and
///                       isinstance(s0, annmodel.SomeInteger)):
///                     mul_sources[op.result] = op.args[1], op.args[0]
///             elif (op.opname == 'inplace_add' and
///                   op.args[1] in mul_sources and
///                   self.gettype(op.args[0]) is list):
///                 v_char, v_count = mul_sources[op.args[1]]
///                 new_op = SpaceOperation('extend_with_char_count',
///                                         [op.args[0], v_char, v_count],
///                                         op.result)
///                 block.operations[i] = new_op
/// ```
pub fn transform_extend_with_char_count(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    // upstream: `isinstance(s, annmodel.SomeInteger)` — Python's
    // `isinstance` walks the MRO, so `SomeBool` (which subclasses
    // `SomeInteger`) matches too. Reproduce that via
    // [`SomeValueTag::mro`] so `SomeBool` / `SomeInteger` both
    // qualify as integer counts.
    fn is_integer_subclass(s: &Option<SomeValue>) -> bool {
        s.as_ref()
            .is_some_and(|sv| sv.tag().mro().contains(&SomeValueTag::Integer))
    }

    for block in block_subset {
        // upstream: maps `b` (mul result Variable) → (char, count).
        #[expect(
            clippy::mutable_key_type,
            reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
        )]
        let mut mul_sources: HashMap<Variable, (Hlvalue, Hlvalue)> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            let op = block.borrow().operations[i].clone();
            if op.opname == "mul" && op.args.len() == 2 {
                // upstream: `s0 = self.annotation(op.args[0])`.
                let s0 = ann.annotation(&op.args[0]);
                let s1 = ann.annotation(&op.args[1]);
                let Hlvalue::Variable(result_v) = &op.result else {
                    continue;
                };
                // upstream: SomeChar * SomeInteger.
                if matches!(s0, Some(SomeValue::Char(_))) && is_integer_subclass(&s1) {
                    mul_sources.insert(result_v.clone(), (op.args[0].clone(), op.args[1].clone()));
                } else if matches!(s1, Some(SomeValue::Char(_))) && is_integer_subclass(&s0) {
                    // upstream: swap so char is first.
                    mul_sources.insert(result_v.clone(), (op.args[1].clone(), op.args[0].clone()));
                }
            } else if op.opname == "inplace_add" && op.args.len() == 2 {
                let Hlvalue::Variable(arg1_v) = &op.args[1] else {
                    continue;
                };
                let Some((v_char, v_count)) = mul_sources.get(arg1_v).cloned() else {
                    continue;
                };
                if !matches!(ann.annotation(&op.args[0]), Some(SomeValue::List(_))) {
                    continue;
                }
                let new_op = SpaceOperation::with_offset(
                    "extend_with_char_count",
                    vec![op.args[0].clone(), v_char, v_count],
                    op.result.clone(),
                    op.offset,
                );
                block.borrow_mut().operations[i] = new_op;
            }
        }
    }
}

/// RPython `transform.py` — `transform_list_contains(self, block_subset)`.
///
/// ```python
/// def transform_list_contains(self, block_subset):
///     """Transforms x in [2, 3]"""
///     for block in block_subset:
///         newlist_sources = {}
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if op.opname == 'newlist':
///                 newlist_sources[op.result] = op.args
///             elif op.opname == 'contains' and op.args[0] in newlist_sources:
///                 items = {}
///                 for v in newlist_sources[op.args[0]]:
///                     s = self.annotation(v)
///                     if not s.is_immutable_constant():
///                         break
///                     items[s.const] = None
///                 else:
///                     # all arguments of the newlist are annotation constants
///                     op.args[0] = Constant(items)
///                     s_dict = self.annotation(op.args[0])
///                     s_dict.dictdef.generalize_key(self.binding(op.args[1]))
/// ```
///
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn transform_list_contains(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        let mut newlist_sources: HashMap<Variable, Vec<Hlvalue>> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            let op = block.borrow().operations[i].clone();
            if op.opname == "newlist" {
                let Hlvalue::Variable(result_v) = &op.result else {
                    continue;
                };
                newlist_sources.insert(result_v.clone(), op.args.clone());
            } else if op.opname == "contains" {
                let Some(Hlvalue::Variable(arg0_v)) = op.args.first() else {
                    continue;
                };
                let Some(newlist_args) = newlist_sources.get(arg0_v) else {
                    continue;
                };
                let needle_binding = ann.binding(&op.args[1]);
                let mut items: HashMap<ConstValue, ConstValue> = HashMap::new();
                // Track the canonical (`hash(True) == hash(1) == hash(1.0)`)
                // form of every key we have already absorbed. Python's
                // dict keeps the FIRST inserted key on a hash collision
                // (see `Objects/dictobject.c:insertdict` "K != KEY"
                // branch), so we dedup via the canonical form but store
                // the original `ConstValue` to preserve the key's
                // inferred type. Without this split the rewritten
                // annotation would widen `[True]` from `SomeBool` to
                // `SomeInteger`, diverging from upstream.
                let mut seen_canonical: std::collections::HashSet<ConstValue> =
                    std::collections::HashSet::new();
                let mut all_constant = true;
                for v in newlist_args {
                    // Pre-filter on `Hlvalue::Constant`: a few internal
                    // flowspace variants have no Python equivalent and
                    // panic inside `Bookkeeper::immutablevalue`. Bail
                    // out before `ann.annotation(v)` dispatches there.
                    // Variables carry their cached annotation instead,
                    // so the `is_valid_contains_key` post-check below
                    // covers that side.
                    if let Hlvalue::Constant(c) = v
                        && !is_valid_contains_key(&c.value)
                    {
                        all_constant = false;
                        break;
                    }
                    let Some(s) = ann.annotation(v) else {
                        all_constant = false;
                        break;
                    };
                    if !s.is_immutable_constant() {
                        all_constant = false;
                        break;
                    }
                    let Some(key) = s.const_().cloned() else {
                        all_constant = false;
                        break;
                    };
                    if !is_valid_contains_key(&key) {
                        all_constant = false;
                        break;
                    }
                    // upstream `items[s.const] = None` relies on Python
                    // dict key semantics where `1 == 1.0 == True` all
                    // hash to the same bucket and the FIRST insertion
                    // wins. `ConstValue`'s derived `Eq`/`Hash` treats
                    // `Int(1)`, `Float(1.0)`, and `Bool(true)` as
                    // distinct, so dedup via `canonical_dict_key` but
                    // store the ORIGINAL key so the DictDef annotation
                    // keeps the caller's inferred type (e.g. `SomeBool`
                    // for `[True]`, not widened `SomeInteger`).
                    let canonical = canonical_dict_key(key.clone());
                    if seen_canonical.insert(canonical) {
                        items.insert(key, ConstValue::None);
                    }
                }
                if all_constant {
                    block.borrow_mut().operations[i].args[0] =
                        Hlvalue::Constant(Constant::new(ConstValue::Dict(items)));
                    let s_dict = ann
                        .annotation(&block.borrow().operations[i].args[0])
                        .expect("transform_list_contains: dict annotation missing");
                    let SomeValue::Dict(s_dict) = s_dict else {
                        panic!("transform_list_contains: rewritten contains lhs is not SomeDict");
                    };
                    s_dict
                        .dictdef
                        .generalize_key(&needle_binding)
                        .expect("transform_list_contains: generalize_key failed");
                }
            }
        }
    }
}

/// Map a [`ConstValue`] to the key Python's built-in `dict` would
/// bucket it under.
///
/// Python's numeric tower makes `True`, `1`, and `1.0` interchangeable
/// as dict keys: `hash(True) == hash(1) == hash(1.0)` and equality
/// spans the three. `ConstValue` derives `Eq`/`Hash` from its enum
/// shape, so `Bool(true)` / `Int(1)` / `Float(bits_of_1.0)` are three
/// separate keys.
///
/// Canonical rules used here:
///
/// * `Bool(b)` → `Int(0|1)` — upstream dict collapse.
/// * `Float(bits)` where the value is finite and equals its truncation
///   and fits an `i64` → `Int(value as i64)`. Non-integral / non-finite
///   floats keep their `Float` form (Python hashes `1.5` and `Int(1)`
///   into different buckets).
/// * `Tuple(items)` — recurse so `(True,)` and `(1,)` collapse the
///   same way upstream's Python dict would.
/// * Everything else returns as-is.
fn canonical_dict_key(cv: ConstValue) -> ConstValue {
    match cv {
        ConstValue::Bool(b) => ConstValue::Int(if b { 1 } else { 0 }),
        ConstValue::Float(bits) => {
            let f = f64::from_bits(bits);
            if f.is_finite() && f == f.trunc() && f >= i64::MIN as f64 && f < i64::MAX as f64 + 1.0
            {
                ConstValue::Int(f as i64)
            } else {
                ConstValue::Float(bits)
            }
        }
        ConstValue::Tuple(items) => {
            ConstValue::Tuple(items.into_iter().map(canonical_dict_key).collect())
        }
        other => other,
    }
}

/// Guard for rewriting `x in [k1, k2, ...]` into
/// `x in {k1: None, ...}`. The rewrite stays a pure optimisation:
/// any element that isn't a valid Python dict key defeats the rewrite
/// and the original `contains` op stays untouched. Upstream gets this
/// guarantee for free from Python's own dict (`TypeError: unhashable
/// type` raised on ill-typed inputs never reaches the rewrite); the
/// Rust port has to encode it explicitly because `ConstValue` has
/// no native hash/equality tie to Python dict semantics.
///
/// Rejected:
///   * `List` / `Dict` — unhashable in Python.
///   * `LLPtr` — RPython `_ptr.__hash__` raises `TypeError`, so a
///     Python dict cannot accept it as a key.
///   * `Code` / `Graphs` / `Atom` / `SpecTag` / `Placeholder` —
///     internal flowspace carriers with no Python equivalent.
///     `Bookkeeper::immutablevalue` returns an error for these, so
///     `ann.annotation(Constant(_internal))` would panic in the
///     caller before the hashability side could even be checked.
///
/// Tuples recurse so `(True, [1])` is rejected the same way `[1]`
/// would be on its own.
fn is_valid_contains_key(cv: &ConstValue) -> bool {
    match cv {
        ConstValue::Bool(_)
        | ConstValue::Int(_)
        | ConstValue::Int128(_)
        | ConstValue::UInt128(_)
        | ConstValue::Float(_)
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::None
        | ConstValue::HostObject(_)
        | ConstValue::Function(_)
        | ConstValue::AddressOffset(_)
        | ConstValue::InheritanceId { .. } => true,
        ConstValue::Tuple(items) => items.iter().all(is_valid_contains_key),
        ConstValue::List(_)
        | ConstValue::Dict(_)
        | ConstValue::Code(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::LLPtr(_)
        | ConstValue::LLAddress(_)
        | ConstValue::Atom(_)
        | ConstValue::SpecTag(_)
        | ConstValue::Placeholder
        | ConstValue::Opaque(_)
        | ConstValue::Repr(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, Variable,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    fn mk_minimum_graph(name: &str) -> GraphRef {
        // Minimum valid graph: startblock → returnblock via Constant(1).
        let start = Block::shared(vec![]);
        let graph: GraphRef = Rc::new(RefCell::new(FunctionGraph::new(name, start.clone())));
        let link = Link::new(
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
            Some(graph.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        graph
    }

    #[test]
    fn checkgraphs_dedups_by_graph_identity() {
        // Two blocks reachable from the same graph → checkgraph should
        // run only once (we can't observe that directly, but the Rust
        // port's `seen` set matches upstream semantics — we just assert
        // the function completes without panic for a valid graph).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let start = graph.borrow().startblock.clone();
        let returnblock = graph.borrow().returnblock.clone();

        // Populate `ann.annotated` so both blocks map to the same graph.
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&returnblock), Some(graph.clone()));
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&start), start.clone());
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&returnblock), returnblock.clone());

        checkgraphs(&ann, &[start.clone(), returnblock.clone()]);
    }

    #[test]
    #[should_panic(expected = "checkgraphs: block_subset contains an unannotated/blocked block")]
    fn checkgraphs_rejects_blocked_entries() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let start = graph.borrow().startblock.clone();
        let blocked = Block::shared(vec![]);

        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&blocked), None);

        checkgraphs(&ann, &[start, blocked]);
    }

    #[test]
    fn fully_annotated_blocks_skips_blocked_entries() {
        // One block fully annotated (Some(Some(graph))) and one block
        // with `None` (= upstream's `False`) should yield only the
        // former.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let a = Block::shared(vec![]);
        let b = Block::shared(vec![]);
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&a), Some(graph.clone()));
        ann.annotated.borrow_mut().insert(BlockKey::of(&b), None);
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&a), a.clone());
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&b), b.clone());

        let blocks = fully_annotated_blocks(&ann);
        assert_eq!(blocks.len(), 1);
        assert!(Rc::ptr_eq(&blocks[0], &a));
    }

    #[test]
    #[should_panic(expected = "fully_annotated_blocks: annotated block missing from all_blocks")]
    fn fully_annotated_blocks_rejects_missing_all_blocks_entry() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let block = Block::shared(vec![]);
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&block), Some(graph));

        let _ = fully_annotated_blocks(&ann);
    }

    #[test]
    fn transform_graph_runs_checkgraphs_twice_and_respects_empty_extra_passes() {
        // Wire a minimum valid graph through `transform_graph` with an
        // explicit block_subset. Mark `start`'s exit as followed so
        // `transform_dead_code` leaves it alone — otherwise it would
        // try to cutoff an empty startblock, which upstream semantics
        // disallow (cutoff requires at least one failing op).
        use crate::flowspace::model::LinkKey;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let start = graph.borrow().startblock.clone();
        ann.translator.graphs.borrow_mut().push(graph.clone());
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&start), start.clone());
        // Mark the single exit as followed so transform_dead_code
        // doesn't prune it.
        for link in &start.borrow().exits {
            ann.links_followed
                .borrow_mut()
                .insert(LinkKey::of(link), link.clone());
        }

        transform_graph(&ann, None, Some(std::slice::from_ref(&start)));
    }

    #[test]
    fn transform_graph_default_extra_passes_apply_allocate_rewrite() {
        use crate::flowspace::model::{LinkKey, SpaceOperation};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let a = Variable::new();
        let b = Variable::new();
        let c = Variable::new();
        let d = Variable::new();
        let start = Block::shared(vec![
            Hlvalue::Variable(a.clone()),
            Hlvalue::Variable(b.clone()),
        ]);
        let graph: GraphRef = Rc::new(RefCell::new(FunctionGraph::new("g", start.clone())));

        start.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Variable(a.clone())],
            Hlvalue::Variable(c.clone()),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![Hlvalue::Variable(c.clone()), Hlvalue::Variable(b.clone())],
            Hlvalue::Variable(d.clone()),
        ));
        let link = Link::new(
            vec![Hlvalue::Variable(d.clone())],
            Some(graph.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);

        ann.translator.graphs.borrow_mut().push(graph.clone());
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&start), start.clone());
        for link in &start.borrow().exits {
            ann.links_followed
                .borrow_mut()
                .insert(LinkKey::of(link), link.clone());
        }

        transform_graph(&ann, None, Some(std::slice::from_ref(&start)));

        let ops = &start.borrow().operations;
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opname, "alloc_and_set");
    }

    /// transform.py::cutoff_alwaysraising_block requires an operation whose
    /// result was never annotated. A fully annotated block with no exits is
    /// broken flow/followed-link bookkeeping, not a successful cutoff.
    #[test]
    #[should_panic(expected = "no failing op")]
    fn cutoff_rejects_fully_annotated_block_without_exits() {
        use crate::annotator::model::SomeInteger;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);
        let result = Variable::new();
        result
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                false, false,
            )))));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
            Hlvalue::Variable(result),
        ));
        cutoff_alwaysraising_block(&ann, &block);
    }

    #[test]
    #[should_panic(expected = "annotated operations must form a prefix")]
    fn cutoff_rejects_annotation_after_unannotated_operation() {
        use crate::annotator::model::SomeInteger;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);
        let late_result = Variable::new();
        late_result
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                false, false,
            )))));
        for result in [Variable::new(), late_result] {
            block.borrow_mut().operations.push(SpaceOperation::new(
                "same_as",
                vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
                Hlvalue::Variable(result),
            ));
        }
        cutoff_alwaysraising_block(&ann, &block);
    }

    #[test]
    fn transform_dead_code_prunes_unfollowed_exits() {
        // A conditional block with two exits: only the False branch is
        // followed. transform_dead_code drops the True branch and the
        // surviving exit gets exitswitch=None + exitcase=None.
        use crate::flowspace::model::{ConstValue as CV, Constant as C, LinkKey, SpaceOperation};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let v = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let body = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let graph: GraphRef = Rc::new(RefCell::new(crate::flowspace::model::FunctionGraph::new(
            "f",
            start.clone(),
        )));
        let returnblock = graph.borrow().returnblock.clone();

        // One op so cutoff_alwaysraising_block would have something to
        // work with, but we won't reach cutoff here.
        body.borrow_mut().operations.push(SpaceOperation::new(
            "noop",
            vec![Hlvalue::Variable(v.clone())],
            Hlvalue::Variable(Variable::new()),
        ));

        start.borrow_mut().exitswitch = Some(Hlvalue::Variable(v.clone()));
        let left = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(body.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(false)))),
        )
        .into_ref();
        let right = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(body.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(true)))),
        )
        .into_ref();
        start.closeblock(vec![left.clone(), right.clone()]);

        // Only the false branch is followed.
        ann.links_followed
            .borrow_mut()
            .insert(LinkKey::of(&left), left.clone());
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&body), Some(graph.clone()));

        let body_end = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![body_end]);

        transform_dead_code(&ann, std::slice::from_ref(&start));

        let s = start.borrow();
        assert_eq!(s.exits.len(), 1);
        assert!(s.exitswitch.is_none());
        assert!(s.exits[0].borrow().exitcase.is_none());
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &body
        ));
    }

    #[test]
    fn transform_allocate_rewrites_newlist_plus_mul_to_alloc_and_set() {
        // Port of upstream's shape:
        //   c = newlist(a)
        //   d = mul(c, b)
        //   -->
        //   d = alloc_and_set(b, a)
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);
        let a = Variable::new();
        let b = Variable::new();
        let c = Variable::new();
        let d = Variable::new();
        // newlist(a) → c
        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Variable(a.clone())],
            Hlvalue::Variable(c.clone()),
        ));
        // mul(c, b) → d
        block.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![Hlvalue::Variable(c.clone()), Hlvalue::Variable(b.clone())],
            Hlvalue::Variable(d.clone()),
        ));
        // Unrelated op to prove the pass doesn't touch foreign muls.
        block.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![
                Hlvalue::Variable(Variable::new()),
                Hlvalue::Constant(C::new(CV::Int(2))),
            ],
            Hlvalue::Variable(Variable::new()),
        ));

        transform_allocate(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        assert_eq!(ops[0].opname, "newlist");
        assert_eq!(ops[1].opname, "alloc_and_set");
        // upstream: args become (b, a).
        let Hlvalue::Variable(arg0) = &ops[1].args[0] else {
            panic!("alloc_and_set arg0 should be Variable");
        };
        assert_eq!(arg0, &b);
        let Hlvalue::Variable(arg1) = &ops[1].args[1] else {
            panic!("alloc_and_set arg1 should be Variable");
        };
        assert_eq!(arg1, &a);
        // Unrelated mul is left alone.
        assert_eq!(ops[2].opname, "mul");
    }

    #[test]
    fn transform_extend_with_str_slice_rewrites_when_types_match() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeList, SomeString, SomeValue};
        use crate::flowspace::model::SpaceOperation;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let s = Variable::new(); // string source
        let x = Variable::new();
        let y = Variable::new();
        let lst = Variable::new(); // list target
        let sliced = Variable::new(); // getslice result
        let result = Variable::new();

        // Pre-bind annotations so the pass's type checks succeed.
        s.annotation
            .replace(Some(std::rc::Rc::new(SomeValue::String(SomeString::new(
                false, false,
            )))));
        lst.annotation
            .replace(Some(std::rc::Rc::new(SomeValue::List(SomeList::new(
                ListDef::new(None, SomeValue::Impossible, false, false),
            )))));

        // getslice(s, x, y) → sliced
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getslice",
            vec![
                Hlvalue::Variable(s.clone()),
                Hlvalue::Variable(x.clone()),
                Hlvalue::Variable(y.clone()),
            ],
            Hlvalue::Variable(sliced.clone()),
        ));
        // inplace_add(lst, sliced) → result
        block.borrow_mut().operations.push(SpaceOperation::new(
            "inplace_add",
            vec![
                Hlvalue::Variable(lst.clone()),
                Hlvalue::Variable(sliced.clone()),
            ],
            Hlvalue::Variable(result.clone()),
        ));

        transform_extend_with_str_slice(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        assert_eq!(ops[0].opname, "getslice");
        assert_eq!(ops[1].opname, "extend_with_str_slice");
        // upstream: args become [lst, x, y, string].
        let Hlvalue::Variable(arg0) = &ops[1].args[0] else {
            panic!();
        };
        assert_eq!(arg0, &lst);
        let Hlvalue::Variable(arg1) = &ops[1].args[1] else {
            panic!();
        };
        assert_eq!(arg1, &x);
        let Hlvalue::Variable(arg2) = &ops[1].args[2] else {
            panic!();
        };
        assert_eq!(arg2, &y);
        let Hlvalue::Variable(arg3) = &ops[1].args[3] else {
            panic!();
        };
        assert_eq!(arg3, &s);
    }

    #[test]
    fn transform_extend_with_char_count_rewrites_when_types_match() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeChar, SomeInteger, SomeList, SomeValue};
        use crate::flowspace::model::SpaceOperation;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let lst = Variable::new();
        let ch = Variable::new();
        let count = Variable::new();
        let mul_result = Variable::new();
        let result = Variable::new();

        lst.annotation
            .replace(Some(std::rc::Rc::new(SomeValue::List(SomeList::new(
                ListDef::new(None, SomeValue::Impossible, false, false),
            )))));
        ch.annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Char(SomeChar::new(
                false,
            )))));
        count
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));

        block.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![
                Hlvalue::Variable(ch.clone()),
                Hlvalue::Variable(count.clone()),
            ],
            Hlvalue::Variable(mul_result.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "inplace_add",
            vec![
                Hlvalue::Variable(lst.clone()),
                Hlvalue::Variable(mul_result.clone()),
            ],
            Hlvalue::Variable(result.clone()),
        ));

        transform_extend_with_char_count(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        assert_eq!(ops[0].opname, "mul");
        assert_eq!(ops[1].opname, "extend_with_char_count");
        let Hlvalue::Variable(arg0) = &ops[1].args[0] else {
            panic!();
        };
        assert_eq!(arg0, &lst);
        let Hlvalue::Variable(arg1) = &ops[1].args[1] else {
            panic!();
        };
        assert_eq!(arg1, &ch);
        let Hlvalue::Variable(arg2) = &ops[1].args[2] else {
            panic!();
        };
        assert_eq!(arg2, &count);
    }

    #[test]
    fn transform_list_contains_rewrites_constant_newlist_to_dict_constant() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let list_v = Variable::new();
        let needle = Variable::new();
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![
                Hlvalue::Constant(C::new(CV::Int(2))),
                Hlvalue::Constant(C::new(CV::Int(3))),
            ],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        let Hlvalue::Constant(dict_c) = &ops[1].args[0] else {
            panic!("contains lhs should be rewritten to Constant(dict)");
        };
        let CV::Dict(items) = &dict_c.value else {
            panic!("contains lhs constant should carry Dict");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items.get(&CV::Int(2)), Some(&CV::None));
        assert_eq!(items.get(&CV::Int(3)), Some(&CV::None));
    }

    #[test]
    fn transform_list_contains_leaves_nonconstant_newlist_alone() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::SpaceOperation;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let source = Variable::new();
        let list_v = Variable::new();
        let needle = Variable::new();
        source
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Variable(source.clone())],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        assert!(matches!(ops[1].args[0], Hlvalue::Variable(_)));
    }

    #[test]
    fn transform_list_contains_leaves_llptr_constant_newlist_alone() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Block, FunctionGraph, Hlvalue, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype;
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let ret = Variable::new();
        ret.set_concretetype(Some(lltype::LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "callee",
            start,
            Hlvalue::Variable(ret),
        )));
        let ptr = lltype::getfunctionptr(&graph, lltype::_getconcretetype).unwrap();

        let list_v = Variable::new();
        let needle = Variable::new();
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Constant(Constant::new(ConstValue::LLPtr(
                Box::new(ptr),
            )))],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        assert!(
            matches!(ops[1].args[0], Hlvalue::Variable(_)),
            "RPython _ptr objects are unhashable, so LLPtr lists stay unreduced"
        );
    }

    #[test]
    fn transform_list_contains_skips_when_nested_list_in_tuple() {
        // `[([True],)]` carries an unhashable
        // list inside a tuple. Python would `TypeError` on
        // `items[tuple] = None`, so the rewrite must bail out and
        // leave the list-iteration semantics intact.
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let list_v = Variable::new();
        let needle = Variable::new();
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        let nested = CV::Tuple(vec![CV::List(vec![CV::Bool(true)])]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Constant(C::new(nested))],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        assert!(
            matches!(ops[1].args[0], Hlvalue::Variable(_)),
            "tuple-containing-list must not be rewritten to Constant(dict)"
        );
    }

    #[test]
    #[should_panic(expected = "transform_list_contains: generalize_key failed")]
    fn transform_list_contains_raises_for_function_constant_key_with_non_pbc_needle() {
        // Upstream does not preflight `generalize_key()`: once the
        // list constant rewrites to `Constant(dict)`, widening the dict
        // key cell with an incompatible non-PBC needle raises from the
        // union lattice.
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue as CV, Constant as C, GraphFunc, SpaceOperation};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let list_v = Variable::new();
        let needle = Variable::new();
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        let globals = C::new(CV::Dict(Default::default()));
        let gf = GraphFunc::new("inner", globals);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Constant(C::new(CV::Function(Box::new(gf))))],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));
    }

    #[test]
    fn transform_list_contains_rewrites_function_constant_key_for_pbc_needle() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{ConstValue as CV, Constant as C, GraphFunc, SpaceOperation};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let globals = C::new(CV::Dict(Default::default()));
        let gf = GraphFunc::new("inner", globals);
        let func_const = Hlvalue::Constant(C::new(CV::Function(Box::new(gf))));
        let needle_s = ann
            .annotation(&func_const)
            .expect("function constant should annotate to SomePBC");
        let SomeValue::PBC(_) = &needle_s else {
            panic!("expected function constant annotation to be SomePBC");
        };

        let list_v = Variable::new();
        let needle = Variable::new();
        needle.annotation.replace(Some(std::rc::Rc::new(needle_s)));
        let result = Variable::new();

        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![func_const],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        assert!(
            matches!(
                block.borrow().operations[1].args[0],
                Hlvalue::Constant(Constant {
                    value: ConstValue::Dict(_),
                    ..
                })
            ),
            "function-constant list should rewrite when the needle is a compatible SomePBC"
        );
    }

    #[test]
    #[should_panic(expected = "transform_list_contains: generalize_key failed")]
    fn transform_list_contains_raises_for_class_constant_key_with_non_pbc_needle() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HostObject, SpaceOperation,
        };

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let list_v = Variable::new();
        let needle = Variable::new();
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        let class = HostObject::new_class("pkg.C", vec![]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Constant(C::new(CV::HostObject(class)))],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));
    }

    #[test]
    fn transform_list_contains_rewrites_class_constant_key_for_pbc_needle() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HostObject, SpaceOperation,
        };

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let class_const = Hlvalue::Constant(C::new(CV::HostObject(HostObject::new_class(
            "pkg.C",
            vec![],
        ))));
        let needle_s = ann
            .annotation(&class_const)
            .expect("class constant should annotate to SomePBC");
        let SomeValue::PBC(_) = &needle_s else {
            panic!("expected class constant annotation to be SomePBC");
        };

        let list_v = Variable::new();
        let needle = Variable::new();
        needle.annotation.replace(Some(std::rc::Rc::new(needle_s)));
        let result = Variable::new();

        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![class_const],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        assert!(
            matches!(
                block.borrow().operations[1].args[0],
                Hlvalue::Constant(Constant {
                    value: ConstValue::Dict(_),
                    ..
                })
            ),
            "class-constant list should rewrite when the needle is a compatible SomePBC"
        );
    }

    #[test]
    fn transform_list_contains_canonicalises_nested_bool_in_tuple() {
        // `(True,)` and `(1,)` must collapse to
        // the same bucket under `canonical_dict_key` (recursive
        // numeric tower). Direct helper check — the end-to-end path
        // also needs a matching-typed needle to survive the
        // annotator's generalize_key union, which isn't the property
        // under test here.
        use crate::flowspace::model::ConstValue as CV;

        let a = canonical_dict_key(CV::Tuple(vec![CV::Bool(true)]));
        let b = canonical_dict_key(CV::Tuple(vec![CV::Int(1)]));
        assert_eq!(
            a, b,
            "canonical_dict_key must recurse so (True,) and (1,) collapse"
        );

        let c = canonical_dict_key(CV::Tuple(vec![CV::Float(1.0_f64.to_bits())]));
        assert_eq!(
            a, c,
            "integral Float(1.0) inside a tuple must also canonicalise to Int(1)"
        );
    }

    #[test]
    fn transform_list_contains_keeps_first_inserted_key_on_numeric_collision() {
        // Python dict keeps the FIRST inserted
        // key on hash collision — `{True: 1, 1: 2}` stays `{True: 2}`
        // with key type `bool`. The Rust rewrite must preserve the
        // caller's original `ConstValue` rather than canonicalising
        // to `Int(1)`; otherwise the subsequent annotation widens
        // `[True]` from `SomeBool` to `SomeInteger`.
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let list_v = Variable::new();
        let needle = Variable::new();
        needle
            .annotation
            .replace(Some(std::rc::Rc::new(SomeValue::Integer(
                SomeInteger::new(false, false),
            ))));
        let result = Variable::new();

        // [True, 1] — after rewrite the dict must retain Bool(true)
        // (first inserted), not Int(1).
        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![
                Hlvalue::Constant(C::new(CV::Bool(true))),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Hlvalue::Variable(list_v.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "contains",
            vec![
                Hlvalue::Variable(list_v.clone()),
                Hlvalue::Variable(needle.clone()),
            ],
            Hlvalue::Variable(result),
        ));

        transform_list_contains(&ann, std::slice::from_ref(&block));

        let ops = &block.borrow().operations;
        let Hlvalue::Constant(dict_c) = &ops[1].args[0] else {
            panic!("contains lhs should be rewritten to Constant(dict)");
        };
        let CV::Dict(items) = &dict_c.value else {
            panic!("contains lhs constant should carry Dict");
        };
        assert_eq!(items.len(), 1, "True and 1 must collapse to one key");
        assert!(
            items.contains_key(&CV::Bool(true)),
            "first-inserted Bool(true) must win over Int(1); got {items:?}"
        );
    }
}
