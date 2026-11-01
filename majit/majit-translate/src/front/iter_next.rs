//! Iterator `next()` → `next` op + StopIteration handler lowering.
//!
//! ## Positioning
//!
//! Layer 3 of the iterator vertical (annotator iter/next ops = Layer 1,
//! rtyper `ListIteratorRepr` = Layer 2): the front-end lift that turns a
//! Rust `Iterator::next()` call and its `Option` match into the graph's
//! native `next` op with a StopIteration exception edge.
//!
//! Rust source models `for x in it` as
//! ```text
//!     opt = Iterator::next(&mut it);    // -> Option<T>
//!     match opt { Some(x) => body, None => break }
//! ```
//! lowered (in MIR) as a `__discriminant` read on `opt` plus a two-way
//! `switchInt` (None = 0, Some = 1).  RPython's `next` op returns the
//! element directly and raises `StopIteration` at exhaustion
//! (`Next.canraise = [StopIteration, RuntimeError]`,
//! `rpython/flowspace/operation.py`; the annotator narrows that per
//! container — `SomeIterator.next.can_only_throw` is the callable
//! `_can_only_throw`, which appends `RuntimeError` only for a `SomeDict`
//! container, `rpython/annotator/unaryop.py`).  This
//! module rewrites the value-encoded Option diamond into that exception
//! representation — the mirror of [`crate::front::result_exc`]'s
//! `Result`/`?` rewrite.  The Option diamond is the same shape minus the
//! `branch()` indirection, and its exception edge targets the loop-break
//! arm (a local `StopIteration` catch) instead of `exceptblock` — the
//! `try: x = next(it) except StopIteration: break` shape.  The rewrite
//! gates on the iterator tracing back to an `iter` op
//! (`originates_from_iter_op`), so only a list iterator reaches it, and
//! `ListIteratorRepr::rtype_next` raises solely `StopIteration` — the
//! block carries no catch-all propagation edge (the `RuntimeError` half is
//! `Next.canraise`'s conservative pair, which `_can_only_throw` already
//! drops for a non-dict container, and `flowin` drops the unhandled
//! remainder).
//!
//! ## The rewrite (`rewire_one_next_site`)
//!
//! Block A holds the `next()` residual call producing `opt`; block C is
//! its single successor — the discriminant switch.  The rewrite:
//! 1. replaces A's residual call with the native `next` op (the
//!    `[__iter_next]` marker the [`crate::translator::rtyper::flowspace_adapter`]
//!    maps to the raising flowspace `next` op), reusing `opt` as the
//!    element result;
//! 2. closes A with `LastException` exits — normal → the `Some` arm
//!    (`opt` IS the element on every normal-link path; every
//!    `cast(opt, Option::Some).__pos_0` in a block dominated by that
//!    link collapses to the element, including a later payload-compare
//!    or match-guard block), `StopIteration` → the `None` arm (loop
//!    break).
//!
//! The `None` arm carries no value in RPython (`ll_listnext` raises
//! `StopIteration` without a result), so any `Option` scrutinee the Rust MIR
//! threads through the break continuation is a value-encoded SSA merge with
//! no RPython counterpart.  `collect_transitive_dead_slots` follows that
//! thread across its whole forward chain and prunes every slot that is never
//! read, switched on, or escaped — a chain reaching a genuine use declines.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the unregistered `next` callee
//! makes the rtyper census Skip the graph (no regression vs the legacy
//! walker).

use crate::flowspace::model::{ConstValue, Constant, Variable};
use crate::front::result_exc::{
    assert_block_pure_besides, assert_single_pred, back_substitute, collapse_pos0_read,
    field_read_named_of_image, follow_single_exit, op_operand_vars, split_diamond_exits,
};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind,
    SpaceOperation, ValueType,
};

/// The `[__iter_next]` FunctionPath marker the rewrite emits in place of
/// the residual `Iterator::next()` call.  `flowspace_adapter::translate_op`
/// maps it to the raising flowspace `next` op (`operation.rs` `OpKind::Next`).
pub(crate) fn next_op_segments() -> Vec<String> {
    vec!["__iter_next".to_string()]
}

/// `true` iff `segments` is the `[__iter_next]` marker.
pub(crate) fn is_iter_next_segments(segments: &[String]) -> bool {
    segments.len() == 1 && segments[0] == "__iter_next"
}

/// `true` iff the residual call target is an `Iterator::next()` — a
/// `next`-leaf method or FunctionPath.  Combined with an `Option` return
/// type at the recording site, this records a `next`-diamond candidate;
/// the rewrite itself validates the surrounding match shape.
pub(crate) fn is_iterator_next_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "next",
        CallTarget::FunctionPath { segments, .. } => segments.last().is_some_and(|s| s == "next"),
        CallTarget::Indirect { method_name, .. } => method_name == "next",
        _ => false,
    }
}

/// `true` iff `segments` is the `core::slice` `iter` constructor the
/// front-end lowers a slice/Vec/array iterator to.  Two spellings reach
/// here: `is_concrete_iter_constructor`'s `into_iter` rewrite collapses to
/// `["core", "slice", "iter"]`, while a direct `<[T]>::iter` method call
/// (e.g. `boxed_slice.iter()`) stays as the raw method path
/// `["core", "slice", "<Impl>", "iter"]`.  Both name the same constructor —
/// `flowspace_adapter::nonraising_core_bridge_opname` already maps both to
/// the `iter` flowspace op feeding `ListIteratorRepr` by keying on the
/// `slice` family + `iter` leaf, so mirror that family/leaf match here
/// rather than the exact-length form (which missed the method spelling and
/// declined every `boxed_slice.iter()` for-loop).
pub(crate) fn is_iter_op_segments(segments: &[String]) -> bool {
    if !segments.last().is_some_and(|s| s == "iter") {
        return false;
    }
    // `is_concrete_iter_constructor` emits `["core", "slice", "iter"]`;
    // a direct `<[T]>::iter` keeps `["core", "slice", "<Impl>", "iter"]`.
    // `impl_method_owner_for_fundecl` / `trait_default_call_segments`
    // strip the crate, so the same family also arrives as `slice::…::iter`.
    match segments {
        [a, b, ..] if a == "core" && b == "slice" => true,
        [a, ..] if a == "slice" && segments.len() >= 2 => true,
        _ => false,
    }
}

/// `true` iff `var` originates — directly or through loop-carried block
/// inputargs — from an `iter` op (a list iterator).  A backward walk: a
/// var produced by an `iter` op is a list iterator; a var that is a block
/// inputarg is traced to each predecessor's link arg in the matching slot
/// (so the loop header's iterator phi resolves through its entry edge to
/// the pre-loop `iter` op, while the back edge re-threads an already-seen
/// var).  Conservative: a var produced by any other op (a reborrow, a
/// foreign iterator constructor) is not followed, so the walk returns
/// `true` only on a positively-confirmed `iter` source.
pub(crate) fn originates_from_iter_op(graph: &FunctionGraph, var: &Variable) -> bool {
    iter_op_container_with(graph, &BackEdges::build(graph), var).is_some()
}

fn iter_probe(op: &SpaceOperation) -> Option<Variable> {
    match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_iter_op_segments(segments) => args.first().cloned().map(LinkArg::into_variable),
        _ => None,
    }
}

fn range_probe(op: &SpaceOperation) -> Option<()> {
    match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments.len() == 1 && segments[0] == crate::runtime_names::shims::RANGE => Some(()),
        _ => None,
    }
}

fn utf8_iter_probe(op: &SpaceOperation) -> Option<Variable> {
    match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.len() == 1 && segments[0] == crate::runtime_names::shims::UTF8_ITER => {
            args.first().cloned().map(LinkArg::into_variable)
        }
        _ => None,
    }
}

/// The container an `iter` op constructed `var` over, when the backward walk
/// confirms one.  `originates_from_iter_op` is this with the container
/// discarded; [`iter_next_item_type`] needs it, because the container is the
/// only thing that separates the two iterator reprs.
///
/// `__majit_utf8iter(s)` is the `str::chars` constructor (`Utf8StringIterator`);
/// it is not a slice `iter`, so the probe is a sibling of [`iter_probe`]
/// rather than an extension of [`is_iter_op_segments`].
pub(crate) fn iter_op_container_with(
    graph: &FunctionGraph,
    edges: &BackEdges,
    var: &Variable,
) -> Option<Variable> {
    walk_back_with(graph, edges, var, |op| {
        iter_probe(op).or_else(|| utf8_iter_probe(op))
    })
}

/// Walk backwards from `var` to the op that produced it, asking `probe`
/// about each candidate, and stop at the first `Some`.
///
/// Two steps per var: (1) an op whose result is the var — hand it to
/// `probe`; (2) a block inputarg — trace each predecessor's link arg in
/// the matching slot, so a loop header's phi resolves through its entry
/// edge to the pre-loop definition while the back edge re-threads an
/// already-seen var.
///
/// Conservative: a var produced by an op `probe` declines is not followed
/// any further, so a result is always a positively-confirmed source rather
/// than the absence of a contrary one.
///
/// One call indexes every op result and every incoming link in the same
/// block/op order the scan below used to walk, then reads those indexes.
/// The visited set, the stack order, and the first `Some` from `probe` are
/// unchanged. The indexes live only for this call: a later rewrite mutates
/// the graph.
pub(crate) fn walk_back_to_source<T>(
    graph: &FunctionGraph,
    var: &Variable,
    probe: impl Fn(&SpaceOperation) -> Option<T>,
) -> Option<T> {
    let edges = BackEdges::build(graph);
    walk_back_with(graph, &edges, var, probe)
}

pub(crate) fn walk_back_with<T>(
    graph: &FunctionGraph,
    edges: &BackEdges,
    var: &Variable,
    probe: impl Fn(&SpaceOperation) -> Option<T>,
) -> Option<T> {
    let mut visited: Vec<Variable> = Vec::new();
    let mut stack: Vec<Variable> = vec![var.clone()];
    while let Some(v) = stack.pop() {
        if visited.contains(&v) {
            continue;
        }
        visited.push(v.clone());
        let Some(vi) = edges.var_index(&v) else {
            continue;
        };
        let (prod_start, prod_end) = edges.producer_range[vi];
        for k in prod_start..prod_end {
            let (bi, oi) = edges.producer_at[k as usize];
            let op = &graph.blocks[bi as usize].operations[oi as usize];
            if let Some(found) = probe(op) {
                return Some(found);
            }
        }
        let (slot_start, slot_end) = edges.slot_range[vi];
        for k in slot_start..slot_end {
            let (bi, pos) = edges.slot_at[k as usize];
            let bi = bi as usize;
            let pos = pos as usize;
            let (in_start, in_end) = edges.incoming_range[bi];
            for ik in in_start..in_end {
                let (pred, ei) = edges.incoming_at[ik as usize];
                let link = &graph.blocks[pred as usize].exits[ei as usize];
                if let Some(LinkArg::Value(src)) = link.args.get(pos) {
                    stack.push(src.clone());
                }
            }
        }
    }
    None
}

/// Producers, inputarg slots, and incoming links for one backward walk.
///
/// Variable rows are a dense numbering of the ids that occur in this
/// graph, assigned in first-seen order. Ids minted for other graphs are
/// not rows. Each row's slice is in the order a scan of `graph.blocks`
/// visits that fact: op results in block/op order, the first inputarg
/// slot of the var in each block, and links that enter a block in
/// predecessor-block then exit order.
pub(crate) struct BackEdges {
    row_of: rustc_hash::FxHashMap<u64, u32>,
    producer_range: Vec<(u32, u32)>,
    producer_at: Vec<(u32, u32)>,
    incoming_range: Vec<(u32, u32)>,
    incoming_at: Vec<(u32, u32)>,
    slot_range: Vec<(u32, u32)>,
    slot_at: Vec<(u32, u32)>,
}

impl BackEdges {
    fn build(graph: &FunctionGraph) -> Self {
        let mut row_of: rustc_hash::FxHashMap<u64, u32> = rustc_hash::FxHashMap::default();
        let mut note = |id: u64| {
            let next = row_of.len() as u32;
            row_of.entry(id).or_insert(next);
        };
        for b in &graph.blocks {
            for op in &b.operations {
                if let Some(result) = &op.result {
                    note(result.id());
                }
            }
            for iv in &b.inputargs {
                note(iv.id());
            }
            for link in &b.exits {
                for arg in &link.args {
                    if let LinkArg::Value(src) = arg {
                        note(src.id());
                    }
                }
            }
        }
        if row_of.is_empty() {
            return Self::empty();
        }
        let n = row_of.len();
        let row = |id: u64| row_of[&id] as usize;
        let nblocks = graph.blocks.len();
        let mut prod_count = vec![0u32; n];
        let mut slot_count = vec![0u32; n];
        let mut last_block = vec![u32::MAX; n];
        let mut in_count = vec![0u32; nblocks];
        for (bi, b) in graph.blocks.iter().enumerate() {
            let bi_u = bi as u32;
            for op in &b.operations {
                if let Some(result) = &op.result {
                    prod_count[row(result.id())] += 1;
                }
            }
            for iv in &b.inputargs {
                let i = row(iv.id());
                if last_block[i] != bi_u {
                    last_block[i] = bi_u;
                    slot_count[i] += 1;
                }
            }
            for link in &b.exits {
                let target = link.target.0;
                if target < nblocks {
                    in_count[target] += 1;
                }
            }
        }
        let producer_range = exclusive_ranges(&prod_count);
        let slot_range = exclusive_ranges(&slot_count);
        let incoming_range = exclusive_ranges(&in_count);
        let mut producer_at =
            vec![(0u32, 0u32); producer_range.last().map(|r| r.1).unwrap_or(0) as usize];
        let mut slot_at = vec![(0u32, 0u32); slot_range.last().map(|r| r.1).unwrap_or(0) as usize];
        let mut incoming_at =
            vec![(0u32, 0u32); incoming_range.last().map(|r| r.1).unwrap_or(0) as usize];
        let mut prod_fill: Vec<u32> = producer_range.iter().map(|r| r.0).collect();
        let mut slot_fill: Vec<u32> = slot_range.iter().map(|r| r.0).collect();
        let mut in_fill: Vec<u32> = incoming_range.iter().map(|r| r.0).collect();
        last_block.fill(u32::MAX);
        for (bi, b) in graph.blocks.iter().enumerate() {
            let bi_u = bi as u32;
            for (oi, op) in b.operations.iter().enumerate() {
                if let Some(result) = &op.result {
                    let i = row(result.id());
                    let at = prod_fill[i] as usize;
                    producer_at[at] = (bi_u, oi as u32);
                    prod_fill[i] += 1;
                }
            }
            for (pos, iv) in b.inputargs.iter().enumerate() {
                let i = row(iv.id());
                if last_block[i] == bi_u {
                    continue;
                }
                last_block[i] = bi_u;
                let at = slot_fill[i] as usize;
                slot_at[at] = (bi_u, pos as u32);
                slot_fill[i] += 1;
            }
            for (ei, link) in b.exits.iter().enumerate() {
                let target = link.target.0;
                if target < nblocks {
                    let at = in_fill[target] as usize;
                    incoming_at[at] = (bi_u, ei as u32);
                    in_fill[target] += 1;
                }
            }
        }
        Self {
            row_of,
            producer_range,
            producer_at,
            incoming_range,
            incoming_at,
            slot_range,
            slot_at,
        }
    }

    fn empty() -> Self {
        Self {
            row_of: rustc_hash::FxHashMap::default(),
            producer_range: Vec::new(),
            producer_at: Vec::new(),
            incoming_range: Vec::new(),
            incoming_at: Vec::new(),
            slot_range: Vec::new(),
            slot_at: Vec::new(),
        }
    }

    fn var_index(&self, var: &Variable) -> Option<usize> {
        self.row_of.get(&var.id()).map(|&row| row as usize)
    }

    /// First op in block/op order whose result is `var`.
    fn first_producer(&self, var: &Variable) -> Option<(u32, u32)> {
        let i = self.var_index(var)?;
        let (start, end) = self.producer_range[i];
        (start < end).then(|| self.producer_at[start as usize])
    }
}

fn exclusive_ranges(counts: &[u32]) -> Vec<(u32, u32)> {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut cursor = 0u32;
    for count in counts {
        let start = cursor;
        cursor += count;
        ranges.push((start, cursor));
    }
    ranges
}

/// The element type the native `next` op yields for the iterator `var`.
///
/// `ListIteratorRepr::rtype_next` (`rlist.py ll_listnext`) hands back the
/// list's item repr, while `RangeIteratorRepr::rtype_next`
/// (`rrange.py ll_rangenext_*`) hands back a `Signed`.
///
/// Two reprs arrive here behind one `iter` op, and the op does not tell
/// them apart.  [`is_iter_op_segments`] admits any `core::slice::…::iter`,
/// and `front::range_iter` deliberately reroutes an exclusive int `Range`
/// for-loop onto the `range()` builtin plus that same bridge, so the
/// container the op was given is what separates them — hence the backward
/// walk rather than a test on the op.
///
/// Getting this wrong is not inert.  `result_ty` is read straight off the
/// op: `resolve_call_result_kind` (`codewriter/jtransform.rs`) consults the
/// rtyper's `concretetype` *only* when `result_ty` is `Unknown`, and
/// `authoritative_result_types` (`codewriter/type_state.rs`) then stamps the
/// derived kind back over it.  So a wrong answer here outranks the real
/// rtyper for every graph that gets a JitCode, not just the legacy tier: the
/// induction variable lands in the wrong register bank, and an `array[i]` in
/// the loop body assembles as a `getarrayitem_gc` with a ref-kind index,
/// which `assembler.rs` asserts on.
///
/// And the container alone was never enough, because a slice of non-GC
/// items is spelled exactly like a slice of references — the same
/// `core::slice::…::iter` that `is_concrete_iter_constructor` collapses
/// `Vec<T>` / `[T; N]` / `Box<[T]>` onto.  `charon-corpus`'s
/// `branch_loop_sum(slice: &[i64], ..)` folds `for &v in slice` today, and
/// answering `Ref` for every non-range container typed its `i64` element as
/// a GC reference.  Hence `recorded`: the element type read off the
/// `Option<T>` at the recording site, which is the only place it survives.
fn iter_next_item_type(
    graph: &FunctionGraph,
    edges: &BackEdges,
    iterator: &Variable,
    recorded: &ValueType,
) -> ValueType {
    // `rrange.py ll_rangenext_*` hands back a `Signed` whatever the Rust
    // range's own spelling is, so the range arm answers before the recorded
    // element type is consulted — a `0..n` over `usize` records `Unsigned`.
    let over_a_range = iter_op_container_with(graph, edges, iterator)
        .is_some_and(|container| produced_by_range_builtin(graph, edges, &container));
    if over_a_range {
        return ValueType::Int;
    }
    match recorded {
        // `rlist.py ll_listnext` hands back the list's item repr.  The
        // classdef is dropped: a `Ref(Some(root))` would seed a different
        // `SomeInstance` shell in `valuetype_to_someshell`, so keeping the
        // bare `Ref` is what leaves every GC-element graph that folds today
        // stamping a byte-identical `result_ty`.
        ValueType::Ref(_) => ValueType::Ref(None),
        other => other.clone(),
    }
}

/// Whether `var` traces back to `front::range_iter`'s reserved
/// `["__majit_range"]` call — the `range(a, b)` builtin standing in for an
/// exclusive int `Range`.
///
/// The same backward walk the iterator itself gets.  `range_iter` emits the
/// `range` call and the `iter` bridge adjacent, so today the container is
/// that call's direct result; a copy or a block boundary between them would
/// otherwise flip the answer back to `Ref`, and the failure mode is the
/// assembler reject [`iter_next_item_type`] exists to prevent rather than a
/// graceful degrade.
fn produced_by_range_builtin(graph: &FunctionGraph, edges: &BackEdges, var: &Variable) -> bool {
    walk_back_with(graph, edges, var, range_probe).is_some()
}

/// Transitive closure of dead forwarded inputarg slots starting from
/// `(start_block, start_slot)`.  Returns the full `(block, slot)` removal
/// set, or an abort reason if any node in the closure is read by an op,
/// tested by an exitswitch, or escapes to the return/except block.
/// Validate-before-mutate: callers mutate only on `Ok`.  RPython's
/// `ll_listnext` StopIteration exhaustion path carries no value, so a
/// forwarded-but-never-read `Option` thread is a Rust SSA-merge artifact
/// prunable along its whole forward chain; removing a never-read inputarg
/// slot drops only dead arguments from each predecessor link (the values
/// stay defined and used wherever else they are live).
pub(crate) fn collect_transitive_dead_slots(
    graph: &FunctionGraph,
    start_block: usize,
    start_slot: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let mut set: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(start_block, start_slot)];
    while let Some((bi, si)) = stack.pop() {
        if set.contains(&(bi, si)) {
            continue; // cycle / already in the removal set
        }
        // The return/except block inputargs are observable (return value /
        // exception payload), so a forward into them escapes — not removable.
        if bi == graph.returnblock.0 || bi == graph.exceptblock.0 {
            return Err(format!("escapes to return/except block {bi}"));
        }
        let Some(v) = graph.blocks[bi].inputargs.get(si).cloned() else {
            return Err(format!("block {bi} lacks inputarg slot {si}"));
        };
        // Genuinely observed → not removable.
        if graph.blocks[bi]
            .operations
            .iter()
            .any(|op| op_operand_vars(&op.kind).contains(&v))
        {
            return Err(format!("slot {si} of block {bi} read by an op"));
        }
        match &graph.blocks[bi].exitswitch {
            Some(ExitSwitch::Value(x)) if *x == v => {
                return Err(format!("slot {si} of block {bi} is the exitswitch"));
            }
            Some(ExitSwitch::Fused { args, .. }) if args.contains(&v) => {
                return Err(format!(
                    "slot {si} of block {bi} is a fused exitswitch operand"
                ));
            }
            _ => {}
        }
        set.push((bi, si));
        // Follow every onward forward of `v` to its target slot.
        for l in &graph.blocks[bi].exits {
            for (j, a) in l.args.iter().enumerate() {
                if matches!(a, LinkArg::Value(x) if *x == v) {
                    stack.push((l.target.0, j));
                }
            }
        }
    }
    Ok(set)
}

/// Every slot `carrier` is forwarded into from `block`'s exits, closed over
/// its onward forwards, when the whole chain is dead.  The enumerate Some arm
/// reads its payload through `__pos_0`; a copy of the Option threaded on
/// through never-read slots is the same Rust SSA-merge artifact the None arm
/// prunes, and packing `(count, item)` leaves it with no value to carry.
fn forwarded_dead_slots(
    graph: &FunctionGraph,
    block: usize,
    carrier: &Variable,
) -> Result<Vec<(usize, usize)>, String> {
    let mut set: Vec<(usize, usize)> = Vec::new();
    for link in &graph.blocks[block].exits {
        for (j, arg) in link.args.iter().enumerate() {
            if matches!(arg, LinkArg::Value(v) if v == carrier) {
                set.extend(collect_transitive_dead_slots(graph, link.target.0, j)?);
            }
        }
    }
    Ok(set)
}

/// The typed `StopIteration` exitcase the `next` block's break link
/// carries — the handler analogue of [`crate::model::exception_exitcase`]
/// (`Exception` catch-all), narrowed to the single exception the loop
/// catches.  `ConstValue::builtin` resolves the class to a `HostObject`,
/// the `Constant(HostObject(class))` shape `annrpython::flowin` matches.
fn stopiteration_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("StopIteration"))
}

/// Split the nullable-pointer form of an Option switch into `(None, Some)`.
/// `front::mir::lower_niche_option_switch` closes the identity-null predicate
/// `((ptr is null) == false)` through `set_branch`, so false is None and true
/// is Some.  This is the flow-graph shape RPython itself uses for a maybe-null
/// pointer (`ptr_nonzero`).
fn split_niche_bool_exits(exits: &[Link], name: &str) -> Result<(Link, Link), String> {
    if exits.len() != 2 {
        return Err(format!(
            "{name}: niche Option switch has {} exits, expected 2",
            exits.len()
        ));
    }
    let mut none: Option<Link> = None;
    let mut some: Option<Link> = None;
    for link in exits {
        match link.exitcase {
            Some(ExitCase::Bool(false)) => none = Some(link.clone()),
            Some(ExitCase::Bool(true)) => some = Some(link.clone()),
            _ => {
                return Err(format!(
                    "{name}: niche Option switch has non-bool exit case {:?}",
                    link.exitcase
                ));
            }
        }
    }
    match (none, some) {
        (Some(none), Some(some)) => Ok((none, some)),
        _ => Err(format!(
            "{name}: niche Option switch lacks the false/true pair"
        )),
    }
}

fn int_const(i: i64) -> LinkArg {
    LinkArg::Const(Constant::new(ConstValue::Int(i)))
}

fn bool_const(value: bool) -> LinkArg {
    LinkArg::Const(Constant::new(ConstValue::Bool(value)))
}

/// Rewrite every recorded `next()` call site into the `next` op +
/// StopIteration handler shape.  Fail-safe: a site whose surrounding
/// `Option` match does not fit the for-loop shape is left as the residual
/// call (Skip), so a mismatch never regresses a graph the legacy walker
/// already handled.  Returns the number of sites rewritten.
pub(crate) fn rewire_next_call_sites(
    graph: &mut FunctionGraph,
    sites: &[(Variable, ValueType)],
) -> usize {
    // One index for every site that has not yet mutated the graph. A site
    // that reached the mutation phase changed blocks and links, whether it
    // then succeeded or declined, so the next site builds a fresh index; a
    // decline during validation leaves the graph alone and keeps this one.
    let mut edges = BackEdges::build(graph);
    let mut rewritten = 0;
    for (opt, recorded_item_ty) in sites {
        let mut mutated = false;
        match rewire_one_next_site(graph, &edges, opt, recorded_item_ty, &mut mutated) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `next` call; the unregistered callee
                // makes the rtyper census Skip this graph (no regression).
            }
        }
        if mutated {
            edges = BackEdges::build(graph);
        }
    }
    rewritten
}

/// `true` iff `segments` is a `__cast_instance_intrinsic` narrow — the front-end
/// pointer-downcast marker (`front::mir` `Rvalue::Cast` #298 arm) the MIR
/// emits when an opaque `Ref` result is reinterpreted as a registered struct
/// root.  It lowers to `cast_pointer` (a pure alias), so it carries no
/// side effect and its result is bit-identical to its operand.
fn is_recast_narrow(kind: &OpKind) -> bool {
    crate::model::cast_instance_root(kind).is_some()
}

/// An unregistered `next()` returns an opaque `Ref`; the MIR immediately
/// narrows it to the concrete `Option<T>` with one or more pure
/// `__cast_instance_intrinsic` recasts (the front-end analogue of
/// `ListIteratorRepr::rtype_next`'s `recast`, `rpython/rtyper/rlist.py`).
/// Peel that trailing chain: starting from the raw `next` result at
/// `next_idx`, follow each contiguous recast whose sole operand is the prior
/// result, and return the chain's final result — the value the block forwards
/// as the `Option` scrutinee.  Every op from `next_idx+1` to the block end
/// must be part of the chain (so the native `next` op, replacing the whole
/// chain, becomes the block's last / raising op), and no chain intermediate
/// may be read by anything but the next recast (else collapsing the chain to
/// a single value would drop a live use).  With no recast (`next` registered,
/// or a list element that needs no narrow) the chain is length 0 and this
/// returns `opt` unchanged.  Any other shape declines (fail-safe).
///
/// Other residual combinators produce the same opaque `Ref` narrowed to the
/// concrete `Option<T>` by a `__cast_instance_intrinsic` recast tail, so they
/// peel the identical chain (the caller passes its own `op_label` for the
/// decline messages).
pub(crate) fn peel_recast_chain(
    graph: &FunctionGraph,
    a: usize,
    next_idx: usize,
    opt: &Variable,
    op_label: &str,
) -> Result<Variable, String> {
    let name = &graph.name;
    let ops = &graph.blocks[a].operations;
    let mut cur = opt.clone();
    let mut idx = next_idx;
    while idx + 1 < ops.len() {
        let follow = &ops[idx + 1];
        // The next op must be a recast whose single operand is `cur`.
        let OpKind::Call { args, .. } = &follow.kind else {
            return Err(format!(
                "{name}: op after {op_label} in block {a} is not the block terminator \
                 nor a recast narrow"
            ));
        };
        if !is_recast_narrow(&follow.kind) || args[0] != cur {
            return Err(format!(
                "{name}: op after {op_label} in block {a} is not a recast of the prior result"
            ));
        }
        // `cur` (a non-final chain intermediate) must be dead except for
        // this recast: no other op reads it AND it does not escape block A
        // on any exit link arg.  Otherwise truncating the chain to the
        // native `next` result would strand a live reference.
        let other_reads = ops
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx + 1)
            .any(|(_, op)| op_operand_vars(&op.kind).contains(&cur));
        if other_reads {
            return Err(format!(
                "{name}: recast operand in block {a} has a second reader — cannot peel"
            ));
        }
        let escapes = graph.blocks[a].exits.iter().any(|l| {
            l.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if *v == cur))
        });
        if escapes {
            return Err(format!(
                "{name}: recast operand in block {a} escapes on an exit link — cannot peel"
            ));
        }
        cur = follow
            .result
            .clone()
            .ok_or_else(|| format!("{name}: recast narrow in block {a} has no result"))?;
        idx += 1;
    }
    Ok(cur)
}

/// `cast(opt, Option<T>::Some)` — production MIR for an aggregate Option
/// match (`collapse_pos0_read` `cast_then_payload`).
fn is_option_some_recast(kind: &OpKind, of: &Variable) -> bool {
    crate::model::cast_instance_root(kind)
        .is_some_and(|root| matches!(root.rsplit("::").next(), Some("Some" | "Continue")))
        && op_operand_vars(kind).contains(of)
}

/// One Some-arm projection of the Option payload: `carrier.__pos_0`, or
/// `cast(carrier, Option::Some).__pos_0`.  `for (i, x) in it.enumerate()`
/// emits the latter twice (once per field).
struct PayloadProjection {
    recast_idx: Option<usize>,
    pos0_idx: usize,
    payload: Variable,
}

fn collect_option_payload_projections(
    ops: &[SpaceOperation],
    carrier: &Variable,
) -> Result<Vec<PayloadProjection>, String> {
    let mut projections = Vec::new();
    let mut recasts: Vec<(usize, Variable)> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        if !op_operand_vars(&op.kind).contains(carrier) {
            continue;
        }
        match &op.kind {
            OpKind::FieldRead { base, field, .. } if base == carrier && field.name == "__pos_0" => {
                let payload = op
                    .result
                    .clone()
                    .ok_or_else(|| "__pos_0 read without result".to_string())?;
                projections.push(PayloadProjection {
                    recast_idx: None,
                    pos0_idx: i,
                    payload,
                });
            }
            _ if is_option_some_recast(&op.kind, carrier) => {
                let narrowed = op
                    .result
                    .clone()
                    .ok_or_else(|| "variant cast without result".to_string())?;
                recasts.push((i, narrowed));
            }
            _ => {
                return Err("uses the payload outside a __pos_0 read".into());
            }
        }
    }
    for (recast_idx, narrowed) in recasts {
        let mut n_pos0 = 0;
        for (i, op) in ops.iter().enumerate() {
            if !op_operand_vars(&op.kind).contains(&narrowed) {
                continue;
            }
            match &op.kind {
                OpKind::FieldRead { base, field, .. }
                    if *base == narrowed && field.name == "__pos_0" =>
                {
                    let payload = op
                        .result
                        .clone()
                        .ok_or_else(|| "__pos_0 read without result".to_string())?;
                    projections.push(PayloadProjection {
                        recast_idx: Some(recast_idx),
                        pos0_idx: i,
                        payload,
                    });
                    n_pos0 += 1;
                }
                _ => {
                    return Err("uses the payload outside a __pos_0 read".into());
                }
            }
        }
        if n_pos0 == 0 {
            return Err("uses the payload outside a __pos_0 read".into());
        }
    }
    Ok(projections)
}

/// Several payload projections of one Option are one value. Keep the
/// first and retarget every later result onto it, dropping extra recasts.
fn unify_payload_projections(
    graph: &mut FunctionGraph,
    block_idx: usize,
    projections: &[PayloadProjection],
) {
    if projections.len() <= 1 {
        return;
    }
    let keep = projections[0].payload.clone();
    let keep_recast = projections[0].recast_idx;
    let extras: Vec<Variable> = projections[1..].iter().map(|p| p.payload.clone()).collect();
    let mut drop: Vec<usize> = Vec::new();
    for p in &projections[1..] {
        drop.push(p.pos0_idx);
        if let Some(r) = p.recast_idx
            && Some(r) != keep_recast
            && !drop.contains(&r)
        {
            drop.push(r);
        }
    }
    drop.sort_unstable();
    drop.dedup();
    for i in drop.into_iter().rev() {
        graph.blocks[block_idx].operations.remove(i);
    }
    let rename = |v: &Variable| -> Variable {
        if extras.iter().any(|e| e == v) {
            keep.clone()
        } else {
            v.clone()
        }
    };
    let block = &mut graph.blocks[block_idx];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (sw, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |b| b,
    );
    block.exitswitch = sw;
    block.exits = exits;
}

/// Blocks reachable from `start` by following exits, including `start`.
fn reachable_from(graph: &FunctionGraph, start: usize) -> Vec<bool> {
    let n = graph.blocks.len();
    let mut seen = vec![false; n];
    if start >= n {
        return seen;
    }
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(b) = stack.pop() {
        for link in &graph.blocks[b].exits {
            let t = link.target.0;
            if t < n && !seen[t] {
                seen[t] = true;
                stack.push(t);
            }
        }
    }
    seen
}

/// Blocks dominated by `root` when `root` is treated as a local CFG root:
/// `root` itself, and every block whose every predecessor is already in
/// the set.  A loop header that the Some arm re-enters is excluded,
/// because the entry edge (and the StopIteration path) also reach it.
fn cone_dominated_by(graph: &FunctionGraph, root: usize) -> Vec<bool> {
    let n = graph.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (src, block) in graph.blocks.iter().enumerate() {
        for link in &block.exits {
            let dst = link.target.0;
            if dst < n && !preds[dst].contains(&src) {
                preds[dst].push(src);
            }
        }
    }
    let mut in_cone = vec![false; n];
    if root >= n {
        return in_cone;
    }
    in_cone[root] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..n {
            if in_cone[b] || preds[b].is_empty() {
                continue;
            }
            if preds[b].iter().all(|&p| in_cone[p]) {
                in_cone[b] = true;
                changed = true;
            }
        }
    }
    in_cone
}

fn block_observes_var(block: &crate::model::Block, var: &Variable) -> bool {
    if block
        .operations
        .iter()
        .any(|op| op_operand_vars(&op.kind).contains(var))
    {
        return true;
    }
    match &block.exitswitch {
        Some(ExitSwitch::Value(v)) if v == var => true,
        Some(ExitSwitch::Fused { args, .. }) if args.contains(var) => true,
        _ => false,
    }
}

/// Slots of the Option payload (the element after rewrite) in blocks
/// dominated by the normal link's Some target.  `ll_listnext` /
/// `ListIteratorRepr.rtype_next` define the iteration value only on the
/// normal path; every Some-arm-reachable `__pos_0` of that value is the
/// element, including a later payload-compare or `Some(x) if` guard.
///
/// A successor slot joins that set only when every incoming link into
/// the block passes, at that position, a variable that is already a
/// member slot's payload.  Dominance of the successor by the Some arm
/// does not prove the other predecessors carry the same value: a
/// Some-arm diamond can merge the iterator Option with a freshly
/// constructed `Some(k)`, and collapsing `.__pos_0` would then rewrite
/// both paths to identity.  `ListIteratorRepr.rtype_next` /
/// `ll_listnext` return the iteration value directly and never
/// reinterpret an unrelated Option.  The Some-arm seed stays: its
/// incoming from the discriminant switch is the definition of the
/// payload, not a join.  A removed successor whose variable is still
/// read by a payload projection declines the whole lowering.
///
/// A block reachable from both the normal link and the StopIteration
/// link is the `Some(_) | None` merge: it must not read the payload
/// (the None arm carries no value).  A read there declines, fail-safe.
fn plan_normal_link_payload_slots(
    graph: &FunctionGraph,
    some_target: BlockId,
    none_target: BlockId,
    pos: usize,
    name: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let some = some_target.0;
    let none = none_target.0;
    if some == none {
        return Err(format!(
            "{name}: Some and None arms share a block — payload collapse is not fail-safe"
        ));
    }
    let cone = cone_dominated_by(graph, some);
    let some_reach = reachable_from(graph, some);
    let none_reach = reachable_from(graph, none);
    if graph.blocks[some].inputargs.get(pos).is_none() {
        return Err(format!("{name}: Some arm lacks payload slot {pos}"));
    }
    let mut slots: Vec<(usize, usize)> = vec![(some, pos)];
    let mut i = 0;
    while i < slots.len() {
        let (b, s) = slots[i];
        i += 1;
        let v = graph.blocks[b]
            .inputargs
            .get(s)
            .cloned()
            .ok_or_else(|| format!("{name}: block {b} lacks payload slot {s}"))?;
        collect_option_payload_projections(&graph.blocks[b].operations, &v)
            .map_err(|reason| format!("{name}: Some-arm-dominated block {b} {reason}"))?;
        for link in &graph.blocks[b].exits {
            let t = link.target.0;
            for (j, arg) in link.args.iter().enumerate() {
                if !matches!(arg, LinkArg::Value(x) if *x == v) {
                    continue;
                }
                if t < some_reach.len() && some_reach[t] && t < none_reach.len() && none_reach[t] {
                    let recv = graph.blocks[t].inputargs.get(j).cloned().ok_or_else(|| {
                        format!("{name}: shared Some(_)|None block {t} lacks slot {j}")
                    })?;
                    if block_observes_var(&graph.blocks[t], &recv) {
                        return Err(format!("{name}: shared Some(_)|None arm reads the payload"));
                    }
                    continue;
                }
                if t >= cone.len() || !cone[t] {
                    continue;
                }
                // The same element may occupy several inputarg slots of one
                // successor (duplicated link args).  Each slot is the same
                // payload; collapse both.  Distinct Option values on the
                // Some *link from C* are still declined above
                // (`payload_positions.len() > 1`).
                if !slots.contains(&(t, j)) {
                    slots.push((t, j));
                }
            }
        }
    }
    // Compute candidates as the one-incoming expansion above, then drop
    // any successor that has an incoming link whose arg at `j` is not a
    // member slot's variable, until the set is closed.
    let seed = (some, pos);
    loop {
        let mut keep: Vec<(usize, usize)> = Vec::new();
        for &(t, j) in &slots {
            if (t, j) == seed || incoming_links_all_pass_member_payload(graph, t, j, &slots) {
                keep.push((t, j));
                continue;
            }
            let v = graph.blocks[t]
                .inputargs
                .get(j)
                .cloned()
                .ok_or_else(|| format!("{name}: block {t} lacks payload slot {j}"))?;
            let projections =
                collect_option_payload_projections(&graph.blocks[t].operations, &v)
                    .map_err(|reason| format!("{name}: Some-arm-dominated block {t} {reason}"))?;
            if !projections.is_empty() {
                return Err(format!(
                    "{name}: Some-arm join slot {j} of block {t} is not the iterator \
                     element on every incoming link — payload collapse is not fail-safe"
                ));
            }
        }
        if keep.len() == slots.len() {
            break;
        }
        slots = keep;
    }
    Ok(slots)
}

/// `true` iff every link into block `t` passes, at position `j`, a
/// variable that is the payload inputarg of a member slot of that
/// link's source block.  A constant, a missing arg, a foreign variable,
/// or a predecessor that is not itself a member all disqualify.
fn incoming_links_all_pass_member_payload(
    graph: &FunctionGraph,
    t: usize,
    j: usize,
    slots: &[(usize, usize)],
) -> bool {
    let mut saw_incoming = false;
    for (src, block) in graph.blocks.iter().enumerate() {
        for link in &block.exits {
            if link.target.0 != t {
                continue;
            }
            saw_incoming = true;
            let Some(arg) = link.args.get(j) else {
                return false;
            };
            let LinkArg::Value(x) = arg else {
                return false;
            };
            let from_source_member = slots.iter().any(|&(b, s)| {
                b == src && graph.blocks[b].inputargs.get(s).is_some_and(|v| v == x)
            });
            if !from_source_member {
                return false;
            }
        }
    }
    saw_incoming
}

fn rewire_one_next_site(
    graph: &mut FunctionGraph,
    edges: &BackEdges,
    opt: &Variable,
    recorded_item_ty: &ValueType,
    mutated: &mut bool,
) -> Result<(), String> {
    let name = graph.name.clone();
    // Block A: the block whose op produces `opt` — the residual `next()`
    // call, closed by lower_call with a single forwarding exit.
    let (a, next_idx) = edges
        .first_producer(opt)
        .ok_or_else(|| format!("{name}: next() result var has no producer block"))?;
    let a = a as usize;
    let next_idx = next_idx as usize;
    // The residual `next()` result, before any trailing recast.  The
    // native op must replace this call — looking up the peeled
    // scrutinee instead leaves `slice::iter::Iter::next` /
    // `array::iter::IntoIter::next` in the block (`ll_listnext` never
    // residualizes).
    let raw_next_result = opt.clone();

    // Capture the iterator operand (the `next` op's single argument) from
    // the raw call, before peeling any recast narrows off its result.
    let (next_target, iter_arg) = match &graph.blocks[a].operations[next_idx].kind {
        OpKind::Call { target, args, .. } if args.len() == 1 => {
            (target.clone(), args[0].clone().into_variable())
        }
        other => {
            return Err(format!(
                "{name}: next() producer op is not a 1-arg call: {other:?}"
            ));
        }
    };
    // `Enumerate::next` over a list iterator is RPython's
    // `next(enumerate(lst))`: the adapter becomes the `enumerate` builtin
    // (`iter_adapter`), whose `next` yields the `(index, item)` tuple the
    // Some arm reads.  Validate-only here — mutation waits until the
    // diamond is confirmed.
    let enumerate_site =
        crate::front::iter_adapter::enumerate_list_inner(graph, edges, &next_target, &iter_arg)?;
    let next_iter = enumerate_site
        .as_ref()
        .map_or_else(|| iter_arg.clone(), |site| site.inner.clone());

    // Read the element type off the still-unmutated graph: the backward
    // walk to the `iter` op's container has to see the block structure the
    // recording site saw, and the dead forwarded-slot removal below rewrites
    // exactly that.  For Enumerate the recorded kind is the *inner* element
    // (the tuple's second item).
    let item_ty = iter_next_item_type(graph, edges, &next_iter, recorded_item_ty);

    // `lower_call` closes the block right after the raising call, so the
    // `next()` call is normally A's last op.  An UNREGISTERED `next()`
    // returns an opaque `Ref` that the MIR immediately recasts to the
    // concrete `Option<T>` via one or more pure `__cast_instance_intrinsic`
    // narrows (the front-end analogue of `ListIteratorRepr::rtype_next`'s
    // `recast`, `rpython/rtyper/rlist.py` `self.r_list.recast`).  Such a
    // narrow is a `cast_pointer` alias, so the native `next` op subsumes it:
    // peel the trailing chain and scrutinise its final result as the Option.
    let effective_opt = peel_recast_chain(graph, a, next_idx, opt, "next()")?;
    let opt = &effective_opt;

    // The 2-exit StopIteration-only shape is faithful ONLY for a list
    // iterator (`ListIteratorRepr::rtype_next` / `ll_listnext` raises
    // solely `StopIteration`).  A non-list iterator's `next` annotates
    // `OpKind::Next` over a non-list tag (poison / mis-tag panic) and its
    // `can_only_throw` carries the live `RuntimeError` half the shape
    // drops.  `is_iterator_next_target` alone cannot tell them apart, so
    // gate on the iterator tracing back to an `iter` op — the front-end's
    // slice/Vec/array iterator constructor (`front::mir`
    // `is_concrete_iter_constructor` + the `.iter()` bridge, both lowered
    // to `["core", "slice", "iter"]`).  Anything else declines (the
    // residual call keeps the rtyper Skip), so a non-list iterator never
    // reaches the rewrite.
    if enumerate_site.is_none() && iter_op_container_with(graph, edges, &iter_arg).is_none() {
        return Err(format!(
            "{name}: next() iterator operand does not originate from an iter op — \
             not a list-iterator for-loop"
        ));
    }
    // `Utf8StringIterator.next` (`rutf8.py`) yields an INT code point.
    // Rust `Chars::next` yields `char`, so the Some-arm element is
    // `unichr(codepoint)` (`builtin_unichr`). The raising `__iter_next`
    // stays last on A (`LastException`); the wrap sits on a forwarding
    // successor.
    let from_utf8 = walk_back_with(graph, edges, &next_iter, utf8_iter_probe).is_some();

    // A's single exit → C (the Option discriminant switch).  Unlike the
    // Result `?` diamond there is no intervening `branch()` block.
    let (c, opt_c) = follow_single_exit(graph, a, opt)
        .map_err(|e| format!("{name}: next call block exit: {e}"))?;
    assert_single_pred(graph, c, &name)?;

    // Block C has one of the two representation-faithful Option tests:
    //
    // * aggregate Option: `d = opt.__discriminant`; switch 0/1;
    // * one-word niche Option: `d = bool((opt is None) == false)`;
    //   switch false/true.  The None sentinel is `ConstNone`
    //   (`rnone.py` `rtype_is_None`; `pairtype(NoneRepr, Repr).convert_from_to`
    //   is `inputconst(r_to, None)`) or `null_mut` (`rtype_ptr_null` /
    //   `convert_const(None)`).  The direct `opt != null_mut()` spelling is
    //   also accepted for artefacts extracted before the identity-null fold.
    //
    // The latter is `ptr_nonzero` in an RPython graph. Rust uses it for a
    // by-value array/Vec iterator whose T is itself a non-null pointer. The
    // front folds the niche before this post-pass runs, so accepting both
    // shapes here is required for one list-iterator representation rather
    // than a separate Rust-specific iterator path.
    //
    // A tagged `Option<T>` Discriminant recasts the forwarded `opt` onto
    // the interned class (`cast_instance_call` of `owner_root`) so
    // `getattr __discriminant` hits the classdef. The Option still
    // escapes on the diamond exits, so `peel_recast_chain_from` of
    // `opt_c` aborts; the FieldRead is of the recast dest.
    let aggregate_disc =
        field_read_named_of_image(&graph.blocks[c].operations, &opt_c, "__discriminant");
    let (none_link, some_link, aggregate_payload, branch_value_vars) = if let Some((
        disc_idx,
        disc_var,
    )) = aggregate_disc
    {
        match &graph.blocks[c].exitswitch {
            Some(ExitSwitch::Value(v)) if *v == disc_var => {}
            other => {
                return Err(format!(
                    "{name}: block {c} exitswitch {other:?} is not the Option discriminant switch"
                ));
            }
        }
        // Block C is bypassed; only the discriminant read may carry an
        // effect in the aggregate shape.
        assert_block_pure_besides(graph, c, &[disc_idx], "discriminant", &name)?;
        let (none_link, some_link) = split_diamond_exits(&graph.blocks[c].exits, &name)?;
        let branch_value_vars = vec![disc_var.clone()];
        (none_link, some_link, true, branch_value_vars)
    } else {
        // The current MIR fold preserves PyPy identity semantics for
        // nullable strings/lists: `is_none = opt is null; d = is_none ==
        // false`.  This is the same `ptr_nonzero` predicate as the older
        // direct `ne(opt, null)` spelling, but unlike value inequality it
        // remains valid for every pointer repr.  Recognise both exact pure
        // forms; accepting an arbitrary boolean chain here could reverse the
        // None/Some arms and would not be a faithful `ll_listnext` rewrite.
        let ops = &graph.blocks[c].operations;
        let direct =
            ops.iter()
                .enumerate()
                .find_map(|(i, op)| match (&op.kind, op.result.as_ref()) {
                    (OpKind::BinOp { op, lhs, rhs, .. }, Some(result))
                        if op == "ne" && (*lhs == opt_c || *rhs == opt_c) =>
                    {
                        Some((
                            vec![i],
                            result.clone(),
                            if *lhs == opt_c {
                                rhs.clone()
                            } else {
                                lhs.clone()
                            },
                            vec![result.clone()],
                        ))
                    }
                    _ => None,
                });
        let identity_inversion = || {
            let (is_idx, is_none, null_var) =
                ops.iter().enumerate().find_map(|(i, op)| {
                    match (&op.kind, op.result.as_ref()) {
                        (OpKind::BinOp { op, lhs, rhs, .. }, Some(result))
                            if op == "is_" && (*lhs == opt_c || *rhs == opt_c) =>
                        {
                            Some((
                                i,
                                result.clone(),
                                if *lhs == opt_c {
                                    rhs.clone()
                                } else {
                                    lhs.clone()
                                },
                            ))
                        }
                        _ => None,
                    }
                })?;
            let (false_idx, false_var) = ops.iter().enumerate().find_map(|(i, op)| {
                matches!(op.kind, OpKind::ConstBool(false))
                    .then(|| op.result.clone().map(|v| (i, v)))
                    .flatten()
            })?;
            let (eq_idx, disc_var) = ops.iter().enumerate().find_map(|(i, op)| {
                match (&op.kind, op.result.as_ref()) {
                    (OpKind::BinOp { op, lhs, rhs, .. }, Some(result))
                        if op == "eq"
                            && ((*lhs == is_none && *rhs == false_var)
                                || (*rhs == is_none && *lhs == false_var)) =>
                    {
                        Some((i, result.clone()))
                    }
                    _ => None,
                }
            })?;
            Some((
                vec![is_idx, false_idx, eq_idx],
                disc_var.clone(),
                null_var,
                vec![is_none, false_var, disc_var],
            ))
        };
        let (mut pure_indices, disc_var, null_var, mut branch_value_vars) = direct
            .or_else(identity_inversion)
            .ok_or_else(|| {
                format!(
                    "{name}: block {c} lacks both an Option __discriminant read and a nullable-pointer test"
                )
            })?;
        // `push_niche_null_ptr` may recast the raw null to the Option
        // payload's concrete pointer repr (for example Wtf8Buf's
        // translated StringRepr).  `push_niche_is_some` with
        // `niche_is_none_uses_const_none` emits `ConstNone` instead
        // (`rnone.py` `rtype_is_None`; `pairtype(NoneRepr, Repr).convert_from_to`
        // is `inputconst(r_to, None)`).  Follow only an exact chain of pure
        // cast-pointer aliases back to `null_mut` or `ConstNone`; any other
        // producer is not evidence of the None sentinel and declines.
        let mut null_source = null_var;
        let mut null_indices = Vec::new();
        loop {
            let (idx, op) = ops
                .iter()
                .enumerate()
                .find(|(_, op)| op.result.as_ref() == Some(&null_source))
                .ok_or_else(|| {
                    format!("{name}: block {c} nullable-pointer test has no None sentinel")
                })?;
            null_indices.push(idx);
            match &op.kind {
                OpKind::ConstNone => break,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if args.is_empty() && segments == &["core", "ptr", "null_mut"] => break,
                OpKind::Call { args, .. } if is_recast_narrow(&op.kind) => {
                    null_source = args[0].clone().into_variable();
                }
                _ => {
                    return Err(format!(
                        "{name}: block {c} nullable-pointer test has no None sentinel"
                    ));
                }
            }
        }
        let (bool_idx, bool_var) = graph.blocks[c]
            .operations
            .iter()
            .enumerate()
            .find_map(|(i, op)| match (&op.kind, op.result.as_ref()) {
                (OpKind::UnaryOp { op, operand, .. }, Some(result))
                    if op == "bool" && *operand == disc_var =>
                {
                    Some((i, result.clone()))
                }
                _ => None,
            })
            .ok_or_else(|| format!("{name}: block {c} nullable-pointer test lacks bool wrap"))?;
        match &graph.blocks[c].exitswitch {
            Some(ExitSwitch::Value(v)) if *v == bool_var => {}
            other => {
                return Err(format!(
                    "{name}: block {c} exitswitch {other:?} is not the niche Option bool switch"
                ));
            }
        }
        pure_indices.extend(null_indices);
        pure_indices.push(bool_idx);
        assert_block_pure_besides(
            graph,
            c,
            &pure_indices,
            "nullable-pointer discriminant",
            &name,
        )?;
        let (none_link, some_link) = split_niche_bool_exits(&graph.blocks[c].exits, &name)?;
        branch_value_vars.push(bool_var.clone());
        (none_link, some_link, false, branch_value_vars)
    };
    let mut some_target = some_link.target;
    let none_target = none_link.target;

    // Some arm (normal exit): the `next` op result IS the element.  Map
    // the Some-link args back to A scope; the forwarded Option value
    // becomes the element result, the threaded discriminant the constant 1.
    let mut normal_args: Vec<LinkArg> = Vec::with_capacity(some_link.args.len());
    let mut payload_positions: Vec<usize> = Vec::new();
    for (i, arg) in some_link.args.iter().enumerate() {
        match arg {
            LinkArg::Const(c0) => normal_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) => {
                if *v == opt_c {
                    normal_args.push(LinkArg::Value(opt.clone()));
                    // Aggregate Some arms still read `__pos_0`; a nullable
                    // pointer is already its payload, so there is no field
                    // read to collapse in the niche form.
                    if aggregate_payload {
                        payload_positions.push(i);
                    }
                } else if branch_value_vars.contains(v) {
                    normal_args.push(if aggregate_payload {
                        int_const(1)
                    } else {
                        bool_const(true)
                    });
                } else {
                    let v_a = back_substitute(graph, &[(a, c)], v, &name)
                        .map_err(|e| format!("{e}; Some-link source variable id {}", v.id()))?;
                    normal_args.push(LinkArg::Value(v_a));
                }
            }
        }
    }

    // Payload collapse is planned before any edit (`plan_normal_link_payload_slots`
    // plus the one-block `collapse_pos0_read` checks).  Two or more Some-link
    // slots (the same Option threaded into several slots) could half-collapse
    // before a later `Err` — decline that unusual shape up front to keep the
    // "validate-before-mutate" fail-safe contract airtight.
    if payload_positions.len() > 1 {
        return Err(format!(
            "{name}: Option value threaded into {} Some-arm slots — multi-slot \
             payload collapse is not fail-safe",
            payload_positions.len()
        ));
    }
    if enumerate_site.is_some() && (!aggregate_payload || payload_positions.is_empty()) {
        return Err(format!(
            "{name}: Enumerate::next Some arm is not an aggregate Option payload — \
             the (i, item) tuple is read off __pos_0"
        ));
    }
    // Slots of a dead chain the Some arm forwards the payload into; pruned
    // together with the None arm's chain below.
    let mut some_dead: Vec<(usize, usize)> = Vec::new();
    // The enumerate Some arm collapses this read onto the `(index, item)`
    // tuple. Accept only `__pos_0` reads of the payload slot, or a slot
    // nothing reads (`for _ in ...enumerate()`). Any other use declines
    // here, before the first edit.
    if enumerate_site.is_some() {
        let pos = payload_positions[0];
        let carrier = graph.blocks[some_target.0]
            .inputargs
            .get(pos)
            .cloned()
            .ok_or_else(|| format!("{name}: enumerate Some arm lacks payload slot {pos}"))?;
        let block = &graph.blocks[some_target.0];
        let projections = collect_option_payload_projections(&block.operations, &carrier)
            .map_err(|reason| format!("{name}: enumerate Some arm {reason}"))?;
        let forwarded = block.exits.iter().any(|link| {
            link.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if v == &carrier))
        });
        let switched = match &block.exitswitch {
            Some(ExitSwitch::Value(v)) if v == &carrier => true,
            Some(ExitSwitch::Fused { args, .. }) if args.contains(&carrier) => true,
            _ => false,
        };
        if switched {
            return Err(format!(
                "{name}: enumerate Some arm uses the payload outside a __pos_0 read"
            ));
        }
        if projections.is_empty() {
            if forwarded {
                some_dead =
                    forwarded_dead_slots(graph, some_target.0, &carrier).map_err(|reason| {
                        format!("{name}: enumerate Some arm forwards a live payload — {reason}")
                    })?;
            }
        } else {
            // Duplicate payload projections are the same Some tuple
            // (`for (i, x)` recasts once per field). They are merged after
            // validation. A forward is accepted only into a dead chain.
            let tuples: Vec<Variable> = projections.iter().map(|p| p.payload.clone()).collect();
            let tuple_escapes = block.operations.iter().any(|op| {
                !matches!(
                    &op.kind,
                    OpKind::FieldRead { base, field, .. }
                        if tuples.contains(base)
                            && (field.name == "__pos_0" || field.name == "__pos_1")
                ) && op_operand_vars(&op.kind).iter().any(|v| tuples.contains(v))
            }) || block.exits.iter().any(|link| {
                link.args
                    .iter()
                    .any(|arg| matches!(arg, LinkArg::Value(v) if tuples.contains(v)))
            }) || match &block.exitswitch {
                Some(ExitSwitch::Value(v)) => tuples.contains(v),
                Some(ExitSwitch::Fused { args, .. }) => args.iter().any(|v| tuples.contains(v)),
                _ => false,
            };
            if tuple_escapes {
                return Err(format!(
                    "{name}: enumerate Some arm's (index, item) tuple leaves its item reads"
                ));
            }
            if forwarded {
                some_dead =
                    forwarded_dead_slots(graph, some_target.0, &carrier).map_err(|reason| {
                        format!("{name}: enumerate Some arm forwards a live payload — {reason}")
                    })?;
            }
        }
    }

    // After rewrite `opt` IS the element on the normal link
    // (`ll_listnext` / `rtype_next`).  Collapse every Some-arm-dominated
    // `cast(opt, Option::Some).__pos_0` pair — the first Some block and
    // any later payload-compare / `Some(x) if` guard — onto that element.
    // Validate the whole cone here, before the first edit.
    let payload_slots = if let Some(&pos) = payload_positions.first() {
        plan_normal_link_payload_slots(graph, some_target, none_target, pos, &name)?
    } else {
        Vec::new()
    };

    // None arm (StopIteration exit): the loop-break continuation.  RPython's
    // `ll_listnext` raises `StopIteration` with NO value on the exhaustion
    // path (`rpython/rtyper/lltypesystem/rlist.py` ll_listnext;
    // `rpython/rtyper/rlist.py` rtype_next — the iteration value is
    // defined solely on the normal path), so a forwarded `Option` scrutinee
    // `opt_c` is a Rust value-encoded SSA merge-thread with no RPython
    // counterpart.  Prune it and its whole transitive forward chain iff every
    // slot in the chain is dead — never read by an op, tested by an
    // exitswitch, or escaping to the return/except block.  A chain reaching a
    // genuine use declines (the residual call keeps the rtyper Skip).
    // `opt_c` can be forwarded into MORE THAN ONE slot of `none_link` (the
    // same Option merge-thread duplicated across slots); each occurrence roots
    // its own transitive chain.  Collect every such slot, not just the first,
    // so a surviving duplicate is not back-substituted onto the StopIteration
    // edge below (where the native `next` result is not produced).  If any
    // chain reaches a genuine use the whole rewrite declines.
    let opt_none_positions: Vec<usize> = none_link
        .args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, LinkArg::Value(v) if *v == opt_c).then_some(i))
        .collect();
    let mut dead: std::collections::BTreeMap<usize, std::collections::BTreeSet<usize>> =
        std::collections::BTreeMap::new();
    for opt_none_pos in opt_none_positions {
        let set = collect_transitive_dead_slots(graph, none_target.0, opt_none_pos).map_err(
            |reason| format!("{name}: None arm forwards a live Option value — {reason}"),
        )?;
        for (b, s) in set {
            dead.entry(b).or_default().insert(s);
        }
    }
    // A's new normal link is built from `normal_args`, which keeps every
    // Some-target slot, so the Some target itself must not lose one.
    if some_dead.iter().any(|&(b, _)| b == some_target.0) {
        return Err(format!(
            "{name}: enumerate Some arm's dead payload chain re-enters the Some target"
        ));
    }
    for (b, s) in some_dead {
        dead.entry(b).or_default().insert(s);
    }

    // Generalized arity guard: removing slot `s` from a block also drops arg
    // `s` from every predecessor link, so every such link must carry the
    // block's full pre-removal inputarg arity.  Enforce it for EVERY block in
    // the removal set — a loop-header re-entry can pull several blocks, even
    // several slots of `none_target`, into the chain — not just `none_target`.
    // Block A's pre-rewrite exit targets C (not a removal-set block), so A is
    // not yet among these; its new StopIteration link is built post-removal.
    // `edges` already lists those predecessor links in the same block/exit
    // order. `collapse_pos0_read` below rewrites exit args but keeps each
    // exit and its target, so the recorded slots still name the links.
    for &b in dead.keys() {
        let arity = graph.blocks[b].inputargs.len();
        let (start, end) = edges.incoming_range[b];
        for k in start..end {
            let (pred, exit_i) = edges.incoming_at[k as usize];
            let nargs = graph.blocks[pred as usize].exits[exit_i as usize]
                .args
                .len();
            if nargs != arity {
                return Err(format!(
                    "{name}: predecessor link to block {b} has arity {nargs} != {arity} — \
                     unsafe to prune the transitive dead-Option chain"
                ));
            }
        }
    }

    // A's StopIteration link to `none_target`: the None-arm link args with
    // EVERY `none_target` slot in the removal set dropped (not only the
    // directly-forwarded `opt_c` slot — a loop re-entry can add a second
    // `none_target` slot), the discriminant threaded as the const 0, and
    // surviving values back-substituted from C scope into A scope.
    let none_target_dead = dead.get(&none_target.0).cloned().unwrap_or_default();
    let mut none_args: Vec<LinkArg> = Vec::with_capacity(none_link.args.len());
    for (p, arg) in none_link.args.iter().enumerate() {
        if none_target_dead.contains(&p) {
            continue; // pruned slot
        }
        match arg {
            LinkArg::Const(c0) => none_args.push(LinkArg::Const(c0.clone())),
            LinkArg::Value(v) if branch_value_vars.contains(v) => {
                none_args.push(if aggregate_payload {
                    int_const(0)
                } else {
                    bool_const(false)
                });
            }
            LinkArg::Value(v) => {
                let v_a = back_substitute(graph, &[(a, c)], v, &name)
                    .map_err(|e| format!("{e}; None-link source variable id {}", v.id()))?;
                none_args.push(LinkArg::Value(v_a));
            }
        }
    }

    // All structural validation passed; mutate the graph.
    *mutated = true;

    if let Some(site) = &enumerate_site {
        crate::front::iter_adapter::rewrite_enumerate_to_builtin(graph, site);
    }

    // The Some-arm-dominated reads of the payload via `opt.__pos_0` (or
    // `cast(opt, Option::Some).__pos_0`) collapse to the carried element.
    // For enumerate that value is the `(index, item)` tuple.
    for &(block, pos) in &payload_slots {
        let carrier = graph.blocks[block].inputargs[pos].clone();
        if let Ok(projections) =
            collect_option_payload_projections(&graph.blocks[block].operations, &carrier)
        {
            unify_payload_projections(graph, block, &projections);
        }
        collapse_pos0_read(graph, BlockId(block), pos, &name)?;
        if enumerate_site.is_some() {
            let carrier = graph.blocks[block].inputargs[pos].clone();
            crate::front::iter_adapter::paint_enumerate_tuple_reads(
                graph, block, &carrier, &item_ty,
            );
        }
    }

    // Drop every dead slot in the transitive chain from its block's inputargs
    // and from every predecessor link feeding that block (descending indices
    // so earlier positions stay valid).  Each link's args are reduced only for
    // its target block's removed slots, keeping link arity == target inputarg
    // arity across the whole graph.  `none_args` already omits `none_target`'s
    // dead slots, and A's new StopIteration link is added below, so the
    // reduced arity stays consistent across all predecessors.
    for (&b, slots) in dead.iter().rev() {
        for &s in slots.iter().rev() {
            graph.blocks[b].inputargs.remove(s);
            let (start, end) = edges.incoming_range[b];
            for k in start..end {
                let (pred, exit_i) = edges.incoming_at[k as usize];
                graph.blocks[pred as usize].exits[exit_i as usize]
                    .args
                    .remove(s);
            }
        }
    }

    // Replace A's residual `next()` call with the native `next` op: the
    // `[__iter_next]` marker, the iterator as its single operand, `opt`
    // (the peeled scrutinee) reused as the element.  Truncating after the
    // raw call drops the recast tail so the native op is A's last / raising
    // op under the `LastException` exitswitch below.  The enumerate
    // rewrite above may have shifted `next_idx`.
    let next_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&raw_next_result))
        .ok_or_else(|| format!("{name}: next() producer op vanished before native replacement"))?;
    let next_result_ty = if enumerate_site.is_some() {
        ValueType::Ref(Some(crate::front::iter_adapter::enumerate_yield_owner(
            &item_ty,
        )))
    } else {
        item_ty
    };
    graph.blocks[a].operations.truncate(next_idx + 1);
    graph.blocks[a].operations[next_idx] = SpaceOperation {
        result: Some(opt.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: next_op_segments(),
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![match &enumerate_site {
                Some(site) => site.renamed(&iter_arg),
                None => iter_arg,
            }]),
            result_ty: next_result_ty,
        },
    };

    // Rewire A: `LastException` exits.
    //   normal        → Some arm (element)
    //   StopIteration → None arm (loop break)
    // `OpKind::Next.can_only_throw` is the conservative `[StopIteration,
    // RuntimeError]` default, but the `originates_from_iter_op` gate above
    // admits only a list iterator, and `ListIteratorRepr::rtype_next`
    // (`ll_listnext`) raises solely `StopIteration` — the `RuntimeError`
    // half is dict-iterator mutation-detection that never materialises
    // here.  The annotator's
    // `flowin` drops the unhandled-exception remainder, so no catch-all
    // propagation edge to `exceptblock` is synthesised (preserving the
    // front graph's "exceptblock edges == MIR unwind terminators"
    // invariant).
    let stop_etype = graph.alloc_value_var();
    let stop_evalue = graph.alloc_value_var();
    // Derive the low-level exit case for the typed `StopIteration` link in
    // the front, exactly as `SwitchInt` (`mir.rs`) and `set_branch`
    // (`model.rs`) do for their model-`Link`s: `flatten.rs`'s
    // `GotoIfExceptionMismatch` reads `link.llexitcase`, and the legacy
    // walker (unlike the rtyper's `_convert_link`) never populates it, so a
    // folded `next` diamond in a graph that falls to the legacy tier would
    // otherwise reach the codewriter with an unset case.  The
    // `Constant(HostObject(class))` the exitcase carries is the shape
    // `assembler::emit_llexitcase` encodes via `identity_id()`.
    let mut stopiter_link = Link::new_mixed(none_args, none_target, Some(stopiteration_exitcase()))
        .with_llexitcase_from_exitcase();
    stopiter_link.last_exception = Some(LinkArg::Value(stop_etype));
    stopiter_link.last_exc_value = Some(LinkArg::Value(stop_evalue));

    if from_utf8 && enumerate_site.is_none() && aggregate_payload && payload_positions.len() == 1 {
        let pos = payload_positions[0];
        let n = graph.blocks[some_target.0].inputargs.len();
        if pos < n {
            let (wrap_bb, wrap_inputs) = graph.create_block_with_arg_vars(n);
            let ch = crate::front::mir::push_unichr(graph, wrap_bb, wrap_inputs[pos].clone());
            let wrap_args: Vec<LinkArg> = wrap_inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if i == pos {
                        LinkArg::Value(ch.clone())
                    } else {
                        LinkArg::Value(v.clone())
                    }
                })
                .collect();
            graph.set_goto_mixed(wrap_bb, some_target, wrap_args);
            some_target = wrap_bb;
        }
    }

    let block_a = &mut graph.blocks[a];
    block_a.exitswitch = Some(ExitSwitch::LastException);
    block_a.exits = vec![
        Link::new_mixed(normal_args, some_target, None),
        stopiter_link,
    ];
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph holding just the `iter` op the fold anchors on, over a
    /// container that is either the `range()` builtin's result or an
    /// unrelated call's.  Returns the iterator variable.
    fn graph_with_iter(over_a_range: bool) -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_iter_next_item_type");
        let n = g.startblock;
        let container_segments = if over_a_range {
            vec![crate::runtime_names::shims::RANGE.to_string()]
        } else {
            vec![
                "some".to_string(),
                "container".to_string(),
                "make".to_string(),
            ]
        };
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: container_segments,
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        (g, it)
    }

    /// `ll_rangenext_up` returns `Signed` whatever the Rust range spells,
    /// so the range arm answers before the recorded element type — a
    /// `for i in 0..n` over `usize` records `Unsigned` and must still come
    /// back `Int`.  This is the guard against "just use the recorded type".
    #[test]
    fn a_range_container_answers_int_over_its_recorded_element() {
        let (g, it) = graph_with_iter(true);
        assert_eq!(
            iter_next_item_type(&g, &BackEdges::build(&g), &it, &ValueType::Unsigned),
            ValueType::Int,
        );
    }

    /// A GC-reference element keeps the bare `Ref`: the classdef is dropped
    /// so every graph that folded before this element type was carried
    /// still stamps the identical `result_ty`.
    #[test]
    fn a_gc_reference_element_answers_a_classdefless_ref() {
        let (g, it) = graph_with_iter(false);
        assert_eq!(
            iter_next_item_type(
                &g,
                &BackEdges::build(&g),
                &it,
                &ValueType::Ref(Some("PyObject".into()))
            ),
            ValueType::Ref(None),
        );
    }

    /// The arm this element type was carried for.  `charon-corpus`'s
    /// `branch_loop_sum(slice: &[i64], ..)` folds `for &v in slice` today and
    /// stamped `Ref` on an `i64`; a non-GC element must come back as itself
    /// so it lands in the int register bank.
    #[test]
    fn a_non_gc_slice_element_keeps_its_own_kind() {
        let (g, it) = graph_with_iter(false);
        for recorded in [ValueType::Int, ValueType::Unsigned, ValueType::Float] {
            assert_eq!(
                iter_next_item_type(&g, &BackEdges::build(&g), &it, &recorded),
                recorded,
                "{recorded:?} element must not be retyped as a GC reference",
            );
        }
    }

    #[test]
    fn nullable_pointer_option_exits_split_false_none_true_some() {
        let none_target = crate::model::BlockId(3);
        let some_target = crate::model::BlockId(4);
        let exits = vec![
            Link::new_mixed(Vec::new(), some_target, Some(ExitCase::Bool(true))),
            Link::new_mixed(Vec::new(), none_target, Some(ExitCase::Bool(false))),
        ];
        let (none, some) = split_niche_bool_exits(&exits, "test").expect("split niche exits");
        assert_eq!(none.target, none_target);
        assert_eq!(some.target, some_target);
    }

    fn slice_iter_next_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["slice", "iter", "Iter", "next"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    /// Residual `next()` plus a trailing `__cast_instance_intrinsic` recast
    /// in the same block — the production shape of `for &x in &[PyObjectRef]`.
    /// The rewrite must replace the residual call, not the recast; otherwise
    /// `slice::iter::Iter::next` survives next to `[__iter_next]` and the
    /// rtyper still Skips.
    #[test]
    fn recast_tail_does_not_leave_the_residual_next_call() {
        use crate::model::{ExitCase, ExitSwitch, FieldDescriptor, Link};

        let mut g = FunctionGraph::new("test_iter_next_recast_tail");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let it_a = a_args[0].clone();
        let raw = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_iter_next_target(),
                    args: crate::model::call_args(vec![it_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let opt = g
            .push_op_var(
                a,
                crate::model::cast_instance_call_result("GCREF", raw.clone(), ValueType::Ref(None)),
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Ref(None),
                pure: true,
            },
            true,
        );
        g.set_return(some_t, None);

        let (none_t, _none_args) = g.create_block_with_arg_vars(0);
        g.set_return(none_t, None);

        g.set_goto(n, a, vec![it]);
        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                Vec::new(),
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c)],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        // Capture the raw `next()` result, the way `front::mir` records it.
        let rewritten = rewire_next_call_sites(&mut g, &[(raw, ValueType::Ref(None))]);
        assert_eq!(rewritten, 1, "the recast-tailed for-loop must fold");
        let residual = g.blocks.iter().flat_map(|b| &b.operations).filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &["slice".to_string(), "iter".to_string(), "Iter".to_string(), "next".to_string()]
            )
        }).count();
        assert_eq!(residual, 0, "residual slice::iter::Iter::next must be gone");
        let native = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &["__iter_next".to_string()]
                )
            })
            .count();
        assert_eq!(
            native, 1,
            "native [__iter_next] replaces the residual and its recast"
        );
    }

    /// Identity-null niche whose None sentinel is `ConstNone` (`is_`),
    /// the Discriminant spelling `push_niche_is_some` emits for an
    /// instance-Ref (`rnone.py` `rtype_is_None`).  Counterpart of
    /// `iter_next_fold_real_identity_null_niche`: that ignored LLBC test
    /// records the same `for &x in &[PyObjectRef]` shape; this one
    /// constructs the diamond and asserts the rewrite fires.
    fn slice_iter_next_const_none_niche_graph(recast_none: bool) -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_iter_next_const_none_niche");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let it_a = a_args[0].clone();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_iter_next_target(),
                    args: crate::model::call_args(vec![it_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let none = g.push_const_none(c);
        let sentinel = if recast_none {
            g.push_op_var(
                c,
                crate::model::cast_instance_call_result("GCREF", none, ValueType::Ref(None)),
                true,
            )
            .unwrap()
        } else {
            none
        };
        let is_none = g
            .push_op_var(
                c,
                OpKind::BinOp {
                    op: "is_".to_string(),
                    lhs: opt_c,
                    rhs: sentinel,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let false_var = g.push_op_var(c, OpKind::ConstBool(false), true).unwrap();
        let disc = g
            .push_op_var(
                c,
                OpKind::BinOp {
                    op: "eq".to_string(),
                    lhs: is_none,
                    rhs: false_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        let (some_t, _some_args) = g.create_block_with_arg_vars(0);
        g.set_return(some_t, None);
        let (none_t, _none_args) = g.create_block_with_arg_vars(0);
        g.set_return(none_t, None);

        g.set_goto(n, a, vec![it]);
        g.set_goto(a, c, vec![opt.clone()]);
        g.set_branch(c, disc, some_t, Vec::new(), none_t, Vec::new());
        (g, opt)
    }

    #[test]
    fn const_none_identity_null_niche_rewrites_to_iter_next() {
        for recast_none in [false, true] {
            let (mut g, opt) = slice_iter_next_const_none_niche_graph(recast_none);
            let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
            assert_eq!(
                rewritten, 1,
                "recast_none={recast_none}: ConstNone + is_ niche must fold"
            );
            let residual = g
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .filter(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments
                            == &[
                                "slice".to_string(),
                                "iter".to_string(),
                                "Iter".to_string(),
                                "next".to_string()
                            ]
                    )
                })
                .count();
            assert_eq!(
                residual, 0,
                "recast_none={recast_none}: residual slice::iter::Iter::next must be gone"
            );
            let native: Vec<&crate::model::SpaceOperation> = g
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .filter(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if is_iter_next_segments(segments)
                    )
                })
                .collect();
            assert_eq!(
                native.len(),
                1,
                "recast_none={recast_none}: exactly one [__iter_next]"
            );
            match &native[0].kind {
                OpKind::Call {
                    target:
                        CallTarget::FunctionPath {
                            segments,
                            fun_decl_id,
                            specialization,
                        },
                    args,
                    result_ty,
                } => {
                    assert_eq!(segments, &next_op_segments());
                    assert_eq!(*fun_decl_id, None);
                    assert_eq!(*specialization, crate::parse::GraphSpecialization::Regular);
                    assert_eq!(args.len(), 1);
                    assert_eq!(*result_ty, ValueType::Ref(None));
                }
                other => {
                    panic!("recast_none={recast_none}: expected [__iter_next] Call, got {other:?}")
                }
            }
            let next_block = g
                .blocks
                .iter()
                .find(|b| {
                    b.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if is_iter_next_segments(segments)
                        )
                    })
                })
                .expect("native next block");
            assert_eq!(
                next_block.exitswitch,
                Some(ExitSwitch::LastException),
                "recast_none={recast_none}: rewritten next closes with LastException"
            );
        }
    }

    fn chars_next_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "str", "iter", "<Impl>", "next"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
    }

    /// `while let Some(ch) = s.chars().next()`: the iterator originates from
    /// `__majit_utf8iter`, so `Chars::next` rewrites to the `[__iter_next]`
    /// diamond.  The residual is not routed as `core::slice::iter`.
    #[test]
    fn utf8iter_chars_next_rewrites_to_iter_next_diamond() {
        use crate::model::{ExitCase, ExitSwitch, FieldDescriptor, Link};

        let mut g = FunctionGraph::new("test_utf8iter_chars_next");
        let n = g.startblock;
        let s = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["some".to_string(), "str".to_string(), "make".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![crate::runtime_names::shims::UTF8_ITER.to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![s]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let it_a = a_args[0].clone();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: chars_next_target(),
                    args: crate::model::call_args(vec![it_a]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        );
        g.set_return(some_t, None);

        let (none_t, _none_args) = g.create_block_with_arg_vars(0);
        g.set_return(none_t, None);

        g.set_goto(n, a, vec![it]);
        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                Vec::new(),
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c)],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "while-let Some over chars() must fold");
        let residual_chars_next = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().is_some_and(|s| s == "next")
                            && segments.iter().any(|s| s == "str")
                )
            })
            .count();
        assert_eq!(residual_chars_next, 0, "residual Chars::next must be gone");
        let residual_slice_iter = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if is_iter_op_segments(segments)
                )
            })
            .count();
        assert_eq!(
            residual_slice_iter, 0,
            "utf8iter must not be rewritten as core::slice::iter"
        );
        let native = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if is_iter_next_segments(segments)
                )
            })
            .count();
        assert_eq!(native, 1, "native [__iter_next] replaces Chars::next");
        let utf8iter = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.len() == 1
                            && segments[0] == crate::runtime_names::shims::UTF8_ITER
                )
            })
            .count();
        assert_eq!(utf8iter, 1, "__majit_utf8iter constructor must remain");
    }

    fn option_some_cast_roots(g: &FunctionGraph) -> Vec<String> {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| {
                crate::model::cast_instance_root(&op.kind).and_then(|root| {
                    (root.contains("Option") && root.ends_with("::Some")).then(|| root.to_string())
                })
            })
            .collect()
    }

    fn residual_iterator_next_count(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if is_iterator_next_target(target)
                )
            })
            .count()
    }

    /// `match it.next() { Some(c) if c == K => A, Some(_) | None => B }`.
    /// The Some arm extracts, switches on the payload, and the default
    /// (guard / `Some(_)`) block extracts again — the inner
    /// `decompose_valuefmt` shape.  After rewrite both downcasts collapse
    /// to the element; the shared `Some(_) | None` arm does not read it.
    #[test]
    fn match_next_guard_collapses_dominated_option_some_downcasts() {
        use crate::model::{ExitCase, ExitSwitch, FieldDescriptor, Link};

        let mut g = FunctionGraph::new("test_iter_next_match_guard");
        let n = g.startblock;
        let s = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["some".to_string(), "str".to_string(), "make".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![crate::runtime_names::shims::UTF8_ITER.to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![s]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let it_a = a_args[0].clone();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: chars_next_target(),
                    args: crate::model::call_args(vec![it_a]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        let narrowed = g
            .push_op_var(
                some_t,
                crate::model::cast_instance_call("Option<char>::Some", some_opt.clone()),
                true,
            )
            .unwrap();
        let payload = g
            .push_op_var(
                some_t,
                OpKind::FieldRead {
                    base: narrowed,
                    field: FieldDescriptor::new("__pos_0", None),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .unwrap();

        let (guard, guard_args) = g.create_block_with_arg_vars(1);
        let guard_opt = guard_args[0].clone();
        let guard_narrowed = g
            .push_op_var(
                guard,
                crate::model::cast_instance_call("Option<char>::Some", guard_opt),
                true,
            )
            .unwrap();
        g.push_op_var(
            guard,
            OpKind::FieldRead {
                base: guard_narrowed,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        );

        let (arm_a, _arm_a_args) = g.create_block_with_arg_vars(0);
        g.set_return(arm_a, None);
        let (shared, _shared_args) = g.create_block_with_arg_vars(0);
        g.set_return(shared, None);
        g.set_goto(guard, shared, Vec::new());

        let (none_t, _none_args) = g.create_block_with_arg_vars(0);
        g.set_goto(none_t, shared, Vec::new());

        g.block_mut(some_t).exitswitch = Some(ExitSwitch::Value(payload));
        g.block_mut(some_t).exits = vec![
            Link::new_mixed(
                Vec::new(),
                arm_a,
                Some(ExitCase::Const(ConstValue::Int(37))),
            )
            .with_prevblock(some_t),
            Link::new_mixed(
                vec![LinkArg::Value(some_opt)],
                guard,
                Some(ExitCase::Const(ConstValue::UniStr("default".into()))),
            )
            .with_prevblock(some_t),
        ];

        g.set_goto(n, a, vec![it]);
        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                Vec::new(),
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c)],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];

        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "match next() with a payload guard must fold");
        assert_eq!(
            residual_iterator_next_count(&g),
            0,
            "residual Chars::next must be gone"
        );
        assert_eq!(
            option_some_cast_roots(&g),
            Vec::<String>::new(),
            "Some-arm-dominated Option::Some downcasts must collapse to the element"
        );
        let native = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if is_iter_next_segments(segments)
                )
            })
            .count();
        assert_eq!(native, 1, "native [__iter_next] replaces Chars::next");
    }

    /// Residual `next()` Option match whose Some arm is still open.
    /// The None arm returns. Callers close `some_t`.
    fn open_some_arm_next_graph(
        name: &str,
    ) -> (FunctionGraph, Variable, BlockId, Variable, BlockId) {
        use crate::model::{ExitCase, ExitSwitch, FieldDescriptor, Link};

        let mut g = FunctionGraph::new(name);
        let n = g.startblock;
        let s = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["some".to_string(), "str".to_string(), "make".to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![crate::runtime_names::shims::UTF8_ITER.to_string()],
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![s]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let it_a = a_args[0].clone();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: chars_next_target(),
                    args: crate::model::call_args(vec![it_a]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(1);
        let opt_c = c_args[0].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: true,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(1);
        let some_opt = some_args[0].clone();
        let (none_t, _none_args) = g.create_block_with_arg_vars(0);
        g.set_return(none_t, None);

        g.set_goto(n, a, vec![it]);
        g.set_goto(a, c, vec![opt.clone()]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                Vec::new(),
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![LinkArg::Value(opt_c)],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];
        (g, opt, some_t, some_opt, none_t)
    }

    fn pos0_read_count(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. } if field.name == "__pos_0"
                )
            })
            .count()
    }

    fn native_iter_next_count(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if is_iter_next_segments(segments)
                )
            })
            .count()
    }

    /// Some-arm diamond: one predecessor forwards the iterator Option,
    /// the other passes a freshly constructed `Some(k)` into the same
    /// join slot, and the join reads `.__pos_0`.  Collapse would rewrite
    /// both paths to identity; the lowering declines and leaves the
    /// graph untouched.  `ll_listnext` never reinterprets an unrelated
    /// Option as the iteration value.
    #[test]
    fn some_arm_mixed_option_join_declines_payload_collapse() {
        use crate::model::{ExitCase, ExitSwitch, FieldDescriptor, Link};

        let (mut g, opt, some_t, some_opt, _none_t) =
            open_some_arm_next_graph("test_iter_next_mixed_some_join");
        let (right, _right_args) = g.create_block_with_arg_vars(0);
        let (join, join_args) = g.create_block_with_arg_vars(1);
        let join_opt = join_args[0].clone();
        g.push_op_var(
            join,
            OpKind::FieldRead {
                base: join_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        );
        g.set_return(join, None);

        let k = g.push_op_var(right, OpKind::ConstInt(7), true).unwrap();
        let constructed = g
            .push_op_var(
                right,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["core", "option", "Option", "Some"]
                            .iter()
                            .map(|s| (*s).to_string())
                            .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![k]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_goto(right, join, vec![constructed]);

        let flag = g
            .push_op_var(some_t, OpKind::ConstBool(true), true)
            .unwrap();
        g.block_mut(some_t).exitswitch = Some(ExitSwitch::Value(flag));
        g.block_mut(some_t).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(some_opt)],
                join,
                Some(ExitCase::Bool(true)),
            )
            .with_prevblock(some_t),
            Link::new_mixed(Vec::new(), right, Some(ExitCase::Bool(false))).with_prevblock(some_t),
        ];

        let before = g.clone();
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 0, "mixed Some-arm Option join must decline");
        assert_eq!(
            g, before,
            "declined lowering must leave the graph untouched"
        );
        assert_eq!(
            residual_iterator_next_count(&g),
            1,
            "residual Chars::next must remain"
        );
        assert_eq!(native_iter_next_count(&g), 0);
        assert_eq!(pos0_read_count(&g), 1, "join __pos_0 must not collapse");
    }

    /// Some-arm diamond where both predecessors pass the iterator
    /// Option into the join slot.  The join `.__pos_0` is the element
    /// on every path and still collapses.
    #[test]
    fn some_arm_uniform_option_join_still_collapses() {
        use crate::model::{ExitCase, ExitSwitch, FieldDescriptor, Link};

        let (mut g, opt, some_t, some_opt, _none_t) =
            open_some_arm_next_graph("test_iter_next_uniform_some_join");
        let (right, right_args) = g.create_block_with_arg_vars(1);
        let right_opt = right_args[0].clone();
        let (join, join_args) = g.create_block_with_arg_vars(1);
        let join_opt = join_args[0].clone();
        g.push_op_var(
            join,
            OpKind::FieldRead {
                base: join_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Int,
                pure: true,
            },
            true,
        );
        g.set_return(join, None);
        g.set_goto(right, join, vec![right_opt]);

        let flag = g
            .push_op_var(some_t, OpKind::ConstBool(true), true)
            .unwrap();
        g.block_mut(some_t).exitswitch = Some(ExitSwitch::Value(flag));
        g.block_mut(some_t).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(some_opt.clone())],
                join,
                Some(ExitCase::Bool(true)),
            )
            .with_prevblock(some_t),
            Link::new_mixed(
                vec![LinkArg::Value(some_opt)],
                right,
                Some(ExitCase::Bool(false)),
            )
            .with_prevblock(some_t),
        ];

        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "uniform Some-arm Option join must fold");
        assert_eq!(
            residual_iterator_next_count(&g),
            0,
            "residual Chars::next must be gone"
        );
        assert_eq!(native_iter_next_count(&g), 1);
        assert_eq!(
            pos0_read_count(&g),
            0,
            "join __pos_0 must collapse to the element"
        );
    }
}
