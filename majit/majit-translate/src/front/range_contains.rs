//! Std `Range` / `RangeInclusive` membership → compare ops.
//!
//! `(a..b).contains(&x)` is `a <= x && x < b`; `(a..=b).contains(&x)` is
//! `a <= x && x <= b`.  Clippy rewrote the interpreter's hand-written
//! inequalities into those idioms; this pass is the rtyper lowering of
//! the idiom, not a walker fold.
//!
//! ## Why not extract `core::ops::range`
//!
//! Charon leaves `core` Foreign (`front::slice_get` and the other
//! opaque-std rewrites cite the same wall).  `--include core::ops::range::_`
//! would make `Range::contains` Transparent, but the body is the generic
//! `RangeBounds::contains` default that matches on `Bound` and calls
//! `PartialOrd::{le,lt}`.  Those impls live in `core::cmp`, Charon does
//! not monomorphize, and the residual would still be an unpublished
//! `PartialOrd` fnaddr — the same subwalk abort, one call deeper.  The
//! cascade (iterators, `Bound`, `PartialOrd`) is the explosion that
//! retires route (a).
//!
//! ## Upstream analogue
//!
//! RPython has no std `Range` type; `x in c` is `rtyper.py`
//! [`translate_op_contains`] dispatching to the container repr's
//! `rtype_contains`.  The inequalities the idiom replaced lower through
//! `rint.py` [`IntegerRepr::rtype_le`] / `rfloat.py`
//! [`FloatRepr::rtype_le`] (`_rtype_compare_template` → `int_le` /
//! `float_le`).  This pass is that dispatch for the std range ADT:
//! recognize the receiver structurally, emit the same compares.
//!
//! [`translate_op_contains`]: rpython/rtyper/rtyper.py
//! [`IntegerRepr::rtype_le`]: rpython/rtyper/rint.py
//! [`FloatRepr::rtype_le`]: rpython/rtyper/rfloat.py
//!
//! ## Producers
//!
//! `RangeInclusive` has private fields, so rustc emits
//! `RangeInclusive::new(a, b)` — an Opaque FunctionPath the rtyper
//! census Skips.  Exclusive `Range` has public fields, so `a..b` is an
//! `Rvalue::Aggregate` the front already emits as
//! `SyntheticTransparentCtor` + `FieldWrite("start")` / `FieldWrite("end")`
//! (the same shell `front::range_iter` diverts for `for _ in a..b`).
//! `contains` is a 2-arg call (`args[0]` the receiver, `args[1]` the
//! `&x` value).  Exclusive `Range` is a transparent ADT, so
//! `first_is_self` routes it as `CallTarget::Method` (`receiver_root`
//! `"Range"`, the same dual as `front::slice_get::is_result_expect_call_target`).
//! Inclusive `RangeInclusive` stays an owner-qualified FunctionPath
//! because the foreign ADT has no extracted body.
//!
//! ## Compare shape
//!
//! `int_between(n, m, p)` is `n <= m < p` and assumes `n <= p`.
//! Exclusive int with constant bounds and `a <= b` is already that
//! predicate, so the fold emits `int_between(a, x, b)`.  Inclusive int
//! needs the successor `b + 1`, and only when both bounds are constants,
//! `a <= b`, and `b != i64::MAX` (then `b + 1` exists and `a <= b + 1`):
//!
//! ```text
//!     r = RangeInclusive::new(a, b)          // residual `new` call (block N)
//!     ...
//!     t = r.contains(&x)                     // residual `contains` call (block C)
//! becomes
//!     upper = b + 1                          // ConstInt; both bounds constant
//!     t     = int_between(a, x, upper)      // a <= x < b + 1
//! ```
//!
//! Every other signed-int shape keeps the two comparisons.  A
//! non-constant bound cannot prove `a <= b` (exclusive) / `a <= b + 1`
//! (inclusive).  `b == i64::MAX` has no representable successor on the
//! inclusive path.  `a > b` is an empty range, and the wrapping
//! expansion of `int_between` answers true for some `x` there.  Float
//! ranges always keep `bitand(le, lt|ge)`: there is no float
//! `int_between`, and `jtransform` rewrites the bare `le`/`lt`/`ge`
//! leaves to `float_*` from the operand bank.
//!
//! ## Cross-block shape
//!
//! Each `TermKind::Call` closes its block with a `set_goto`, so a
//! `new` / aggregate (block N) and `contains` (block C) sit in adjacent
//! blocks, C the single successor of N's Call (or goto) edge.  The range
//! result rides N → C on a goto link arg; the front's liveness-forwarding
//! keeps the same `Variable` identity across the edge (and onward toward
//! the value's drop), so the `contains` receiver may be that same
//! Variable or a fresh C inputarg fed positionally from it (framestate
//! SSA).  The fold traces the receiver back through the single-predecessor
//! link-arg chain to its producer, then threads the bounds `a` / `b` into
//! block C via [`FunctionGraph::ensure_variable_at_block`] (the same
//! single-predecessor carry-through) before splicing the compares.
//! Exclusive `Range` stores the bounds on the live `FieldWrite` chain
//! (re-read after `simplify_lowered_graph`, matching `front::range_iter`).
//!
//! ## Why the producer must be removed explicitly
//!
//! No sweep reclaims a dead `OpKind::Call { FunctionPath }`: `is_pure_op`
//! returns `false` for `Call`, so `prune_dead_phis` never removes it, and
//! `remove_dead_aggregates` only handles `SyntheticTransparentCtor` +
//! `Box::new_uninit`.  A fold that emits the compares at `contains` and
//! lets inclusive `new` "die" is UNSAFE — the orphaned `new` still walls
//! the graph.  Exclusive `Range` is a `SyntheticTransparentCtor`, which
//! that sweep *would* reclaim, but the construction `FieldWrite`s still
//! read the shell; they are removed with the ctor so a later consumer
//! cannot observe a half-deleted range.  The now-dead threaded range
//! value / link args are reclaimed by `prune_dead_phis` (the caller runs
//! it after any rewrite).
//!
//! ## Scope + fail-safe
//!
//! Signed word (`I64` / `Isize`) and `F64` element types.  The pyre
//! `W_Range` Python object is out of scope.  Iteration-form ranges
//! (`for i in a..b`) are excluded by the consumer-shape gate: an
//! iterator reads the range a second time as an OP operand, so a range
//! value read by any op other than the single `contains` being folded
//! (and the exclusive-`Range` construction `FieldWrite`s) declines.
//! Link-arg threading of the range value is liveness forwarding, not
//! consumption, so it does not count.  Every mismatch declines (leaves
//! BOTH residual calls / the ctor chain, census Skip) — no regression.

use std::collections::HashSet;

use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{
    BlockId, CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// Element bank of a std range the fold can lower.  Signed word matches
/// `int_between` / `int_le`; `F64` matches `float_le` after jtransform
/// prefixes the bare compare leaves from the operand bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RangeElemKind {
    SignedInt,
    Float,
}

/// A recognized `RangeInclusive::new(lo, hi)` call site captured during
/// body lowering (`front::mir`).  Carries the result var (the range) and
/// the two bounds so the fold can thread them into the `contains`
/// block.
#[derive(Clone)]
pub(crate) struct RangeInclusiveNewSite {
    /// The `new` call result (the `RangeInclusive` value) — locates the
    /// producer op and matches the `contains` receiver.
    pub result_var: Variable,
    /// Lower bound `a` — `int_between`'s first operand, or `le`'s lhs.
    pub lo: Variable,
    /// Inclusive upper bound `b`.  The signed-int op's exclusive limit is
    /// `b + 1`.
    pub hi: Variable,
    /// Element bank captured from the destination `RangeInclusive<T>`.
    pub elem: RangeElemKind,
}

/// A recognized exclusive `Range { start, end }` aggregate captured
/// during body lowering (`front::mir`).  Bounds are re-read from the
/// live `FieldWrite` chain at rewrite time, matching `front::range_iter`.
#[derive(Clone)]
pub(crate) struct RangeExclusiveNewSite {
    /// The `SyntheticTransparentCtor` result (the `Range` value).
    pub result_var: Variable,
    /// Element bank captured from the aggregate head's `T`.
    pub elem: RangeElemKind,
}

/// A recognized `Range` / `RangeInclusive` `contains(&self, &x)` call
/// site captured during body lowering (`front::mir`).  Carries the result
/// var (the membership bool) — the fold reuses it for `int_between` or
/// the compare pair.
#[derive(Clone)]
pub(crate) struct RangeContainsSite {
    /// The `contains` call result (the membership `bool`) — locates the
    /// producer op; reused as the `bitand` / `int_between` result.
    pub result_var: Variable,
}

/// Rewrite every recorded `(a..b).contains(&x)` / `(a..=b).contains(&x)`
/// call site into a bounds check.  Fail-safe: a site that does not match
/// the expected producer → `contains` shape is left untouched (residual
/// calls / the ctor chain survive, census Skip).  Returns the number of
/// sites rewritten.
pub(crate) fn rewire_range_contains_call_sites(
    graph: &mut FunctionGraph,
    new_sites: &[RangeInclusiveNewSite],
    exclusive_sites: &[RangeExclusiveNewSite],
    contains_sites: &[RangeContainsSite],
) -> usize {
    let mut rewritten = 0;
    for site in contains_sites {
        match rewire_one_range_contains_site(graph, site, new_sites, exclusive_sites) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave both residual calls / the ctor chain; the
                // unregistered callee keeps the rtyper census Skip.
            }
        }
    }
    rewritten
}

fn rewire_one_range_contains_site(
    graph: &mut FunctionGraph,
    site: &RangeContainsSite,
    new_sites: &[RangeInclusiveNewSite],
    exclusive_sites: &[RangeExclusiveNewSite],
) -> Result<(), String> {
    let name = graph.name.clone();

    // 1. Locate the `contains` producer op + its block C.
    let (c_idx, call_idx) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(&site.result_var))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: range contains result var has no producer op"))?;

    // 2. The producer must be the 2-arg `contains` call: `args[0]` the
    //    receiver, `args[1]` the `&x` value.  Transparent `Range` arrives
    //    as `CallTarget::Method`; opaque `RangeInclusive` as an
    //    owner-qualified FunctionPath.  Exclusive vs inclusive is read
    //    from the receiver's producer, not from a helper-name table.
    let (range_v, x) = match &graph.blocks[c_idx].operations[call_idx].kind {
        OpKind::Call { target, args, .. }
            if is_range_contains_call_target(target) && args.len() == 2 =>
        {
            (args[0].clone(), args[1].clone())
        }
        other => {
            return Err(format!(
                "{name}: range contains producer op is not the 2-arg `contains` call: {other:?}"
            ));
        }
    };

    // 3. Match the producer whose `result_var` threads to block C.
    //    Inclusive `new` and exclusive `Range { start, end }` both ride
    //    a goto link-arg that becomes a C inputarg, so `range_v` (the
    //    C-local receiver) is often a distinct Variable.  A site matches
    //    when its `result_var` is defined in a block reachable from C
    //    through single-predecessor edges only AND the receiver `range_v`
    //    is either that same Variable (identity preserved, including
    //    same-block aggregate + contains) or a C inputarg fed exclusively
    //    from that definition.
    let c_block = graph.blocks[c_idx].id;
    let producer = match_range_producer(graph, c_block, &range_v, new_sites, exclusive_sites)
        .ok_or_else(|| {
            format!("{name}: range contains receiver traces to no Range / RangeInclusive producer")
        })?;

    // 4. Consumer-shape gate (excludes iteration-form ranges, whose
    //    range carries a stateful `exhausted` field read a second time by
    //    an iterator's `into_iter` / `next`).  The range value must be
    //    read as an OP operand by exactly the `contains` op being folded
    //    (plus exclusive-`Range` construction `FieldWrite`s).  Dead
    //    framestate `Link.args` forwarding past the `contains` block is
    //    reclaimed by `prune_dead_phis` and does not count.
    if !range_value_single_op_consumer(graph, producer.result_var(), &site.result_var) {
        return Err(format!(
            "{name}: range value has a second consumer — declining (iteration form?)"
        ));
    }

    let (lo, hi) = producer
        .bounds(graph)
        .ok_or_else(|| format!("{name}: range construction has no unique live start/end bounds"))?;

    // 5. Thread `lo` / `hi` into block C as inputargs (single-pred carry-
    //    through).  Decline if either cannot be threaded (orphan / multi-
    //    predecessor block) rather than panicking.
    if !can_thread_to_block(graph, c_block, &lo) || !can_thread_to_block(graph, c_block, &hi) {
        return Err(format!(
            "{name}: range bounds cannot thread to the contains block"
        ));
    }
    let Some(lo_in_c) = graph.ensure_variable_at_block_local(c_block, &lo) else {
        return Err(format!(
            "{name}: range construction bounds threading failed at the contains block"
        ));
    };
    let Some(hi_in_c) = graph.ensure_variable_at_block_local(c_block, &hi) else {
        return Err(format!(
            "{name}: range construction bounds threading failed at the contains block"
        ));
    };

    // 6. Splice the bounds check into block C, replacing the `contains`
    //    op in place and reusing its result Variable.
    let inserts = build_range_contains_compares(
        graph,
        &site.result_var,
        lo_in_c,
        hi_in_c,
        x.into_variable(),
        producer.exclusive(),
        producer.elem(),
    );
    let ops = &mut graph.blocks[c_idx].operations;
    ops.remove(call_idx);
    for (offset, op) in inserts.into_iter().enumerate() {
        ops.insert(call_idx + offset, op);
    }

    // 7. Remove the now-unread producer.  After removal `lo` / `hi` stay
    //    live (threaded to C); the dead threaded range inputarg / link
    //    args are reclaimed by `prune_dead_phis`.
    let producer_result = producer.result_var().clone();
    if producer.exclusive() {
        remove_range_field_writes(graph, &producer_result);
    }
    remove_op_by_result(graph, &producer_result);

    Ok(())
}

/// Inclusive `new` vs exclusive `Range { start, end }` producer matched
/// against the `contains` receiver.
enum RangeProducer<'a> {
    Inclusive(&'a RangeInclusiveNewSite),
    Exclusive(&'a RangeExclusiveNewSite),
}

impl RangeProducer<'_> {
    fn result_var(&self) -> &Variable {
        match self {
            Self::Inclusive(s) => &s.result_var,
            Self::Exclusive(s) => &s.result_var,
        }
    }

    fn elem(&self) -> RangeElemKind {
        match self {
            Self::Inclusive(s) => s.elem,
            Self::Exclusive(s) => s.elem,
        }
    }

    fn exclusive(&self) -> bool {
        matches!(self, Self::Exclusive(_))
    }

    fn bounds(&self, graph: &FunctionGraph) -> Option<(Variable, Variable)> {
        match self {
            Self::Inclusive(s) => Some((s.lo.clone(), s.hi.clone())),
            Self::Exclusive(s) => crate::front::range_iter::live_range_bounds(graph, &s.result_var),
        }
    }
}

fn match_range_producer<'a>(
    graph: &FunctionGraph,
    c_block: BlockId,
    range_v: &Variable,
    new_sites: &'a [RangeInclusiveNewSite],
    exclusive_sites: &'a [RangeExclusiveNewSite],
) -> Option<RangeProducer<'a>> {
    if let Some(site) = new_sites
        .iter()
        .find(|s| range_matches_new_site(graph, c_block, range_v, &s.result_var))
    {
        return Some(RangeProducer::Inclusive(site));
    }
    exclusive_sites
        .iter()
        .find(|s| range_matches_new_site(graph, c_block, range_v, &s.result_var))
        .map(RangeProducer::Exclusive)
}

fn is_contains_leaf(segments: &[String]) -> bool {
    segments
        .last()
        .is_some_and(|leaf| crate::front::clause_spec::unspecialized_leaf(leaf) == "contains")
}

/// `Range` / `RangeInclusive` `contains(&self, &x)` — the dual
/// `CallTarget::Method` / `FunctionPath` shape `front::slice_get::is_result_expect_call_target`
/// uses for `Result::expect`.  Transparent `Range` is Method with
/// `receiver_root == "Range"`; opaque `RangeInclusive` is a FunctionPath
/// whose leaf is `contains` (`<Impl>::contains` after Charon's owner
/// resolution).
pub(crate) fn is_range_contains_call_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => {
            name == "contains"
                && matches!(receiver_root.as_deref(), Some("Range" | "RangeInclusive"))
        }
        CallTarget::FunctionPath { segments, .. } => is_contains_leaf(segments),
        _ => false,
    }
}

fn remove_range_field_writes(graph: &mut FunctionGraph, range_result: &Variable) {
    for block in &mut graph.blocks {
        block.operations.retain(
            |op| !matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == range_result),
        );
    }
}

/// `true` when the block-C receiver `range_v` is fed exactly by the
/// `new` result `new_result` — either the same Variable (identity
/// preserved across the boundary) or a C inputarg threaded from
/// `new_result` through a single-predecessor link-arg chain.
///
/// The trace is positional and exact so that a body containing two
/// distinct `a..=b` ranges never pairs a `contains` with the wrong
/// `new` (which would splice the wrong bounds): at each level the
/// receiver must be an inputarg whose SINGLE predecessor supplies the
/// value at the matching link-arg index, recursing on that supplied
/// Variable until it either reaches `new_result` (match) or a block that
/// defines it via an op (must then equal `new_result`).
fn range_matches_new_site(
    graph: &FunctionGraph,
    c_block: BlockId,
    range_v: &Variable,
    new_result: &Variable,
) -> bool {
    let mut cur_block = c_block;
    let mut cur_var = range_v.clone();
    let mut hops = 0usize;
    loop {
        if &cur_var == new_result {
            return true;
        }
        hops += 1;
        if hops > graph.blocks.len() {
            return false;
        }
        // `cur_var` must be a block inputarg to trace across the edge;
        // if it is defined by an op in `cur_block` and is not
        // `new_result`, it is a different value — no match.
        let Some(arg_idx) = graph
            .block(cur_block)
            .inputargs
            .iter()
            .position(|v| v == &cur_var)
        else {
            return false;
        };
        // Exactly one predecessor edge, carrying the source at `arg_idx`.
        let pred_edges: Vec<(BlockId, usize)> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                let bid = b.id;
                b.exits
                    .iter()
                    .enumerate()
                    .filter(move |(_, e)| e.target == cur_block)
                    .map(move |(i, _)| (bid, i))
            })
            .collect();
        if pred_edges.len() != 1 {
            return false;
        }
        let (pred_block, exit_idx) = pred_edges[0];
        let Some(LinkArg::Value(src)) = graph.block(pred_block).exits[exit_idx].args.get(arg_idx)
        else {
            return false;
        };
        cur_var = src.clone();
        cur_block = pred_block;
    }
}

/// `var` is defined in a block reachable from `block` by walking
/// single-predecessor edges only — the precondition under which
/// [`FunctionGraph::ensure_variable_at_block`] threads cleanly.  Mirrors
/// the private helper of the same name in `model::thread_undefined_op_operands`.
fn defined_via_single_pred_chain(graph: &FunctionGraph, block: BlockId, var: &Variable) -> bool {
    let mut cur = block;
    let mut hops = 0usize;
    loop {
        hops += 1;
        if hops > graph.blocks.len() {
            return false;
        }
        let preds = graph.predecessors(cur);
        if preds.len() != 1 {
            return false;
        }
        let p = preds[0];
        if graph.variable_defined_in_block(p, var) {
            return true;
        }
        cur = p;
    }
}

/// A `var` can be threaded to `block` iff it is already defined there or
/// reachable through a single-predecessor chain.  Same predicate
/// `ensure_variable_at_block` succeeds on; checked first so a
/// multi-predecessor / orphan case declines instead of panicking.
fn can_thread_to_block(graph: &FunctionGraph, block: BlockId, var: &Variable) -> bool {
    graph.variable_defined_in_block(block, var) || defined_via_single_pred_chain(graph, block, var)
}

/// `true` when the range value is CONSUMED by exactly the one
/// `contains` op being folded — the consumer-shape gate that excludes
/// iteration-form ranges (whose range carries a stateful `exhausted`
/// field an iterator's `into_iter` / `next` reads a second time).
///
/// The genuine soundness requirement is that no operation other than the
/// `contains` being folded READS the range value.  A range that merely
/// rides framestate `Link.args` past the `contains` block without being
/// read is dead once `new`/`contains` are removed — `prune_dead_phis`
/// (run by the caller after any rewrite) reclaims the dead inputarg /
/// link-arg forwards.  So link-arg forwarding is NOT a second consumer;
/// only an op operand, an `exitswitch` discriminator, or an exception
/// payload read is.
///
/// Framestate SSA can rename the range value into a fresh inputarg at
/// every forwarding block, so a purely backward positional trace cannot
/// see a read that happens under a renamed alias.  This computes the
/// FORWARD alias closure from the `new` result — the fixpoint set of
/// every Variable the range value flows into through `Link.args` — and
/// then rejects if any op/exitswitch/exception payload (other than the
/// folded `contains`) reads a Variable in that closure.  An iteration-form
/// range's `into_iter`/`next` reads the range as an op operand, so it is
/// caught even after a rename; a dead forward is not.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn range_value_single_op_consumer(
    graph: &FunctionGraph,
    new_result: &Variable,
    contains_result: &Variable,
) -> bool {
    use std::collections::HashSet;
    // Forward alias closure: every Variable the `new` result reaches
    // through positional `Link.args` forwarding.  Seed with the producer
    // result, then iterate to a fixpoint — a link arg in the closure adds
    // the target block's inputarg at the matching position (the renamed
    // alias in the successor).
    let mut closure: HashSet<Variable> = HashSet::new();
    closure.insert(new_result.clone());
    loop {
        let mut grew = false;
        for block in &graph.blocks {
            for link in &block.exits {
                let Some(target_idx) = graph.blocks.iter().position(|b| b.id == link.target) else {
                    continue;
                };
                let target_inputargs = &graph.blocks[target_idx].inputargs;
                for (arg_idx, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(var) = arg else { continue };
                    if closure.contains(var)
                        && let Some(iarg) = target_inputargs.get(arg_idx)
                        && closure.insert(iarg.clone())
                    {
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }

    let is_range_value = |v: &Variable| closure.contains(v);
    for block in &graph.blocks {
        for op in &block.operations {
            let reads_range = crate::inline::op_variable_refs(&op.kind)
                .iter()
                .any(is_range_value);
            if reads_range && op.result.as_ref() != Some(contains_result) {
                // Exclusive `Range { start, end }` construction writes the
                // shell's fields; those ops are removed with the ctor.
                // Anything else is a genuine second consumer (iteration
                // form?).
                let is_construction_write =
                    matches!(&op.kind, OpKind::FieldWrite { base, .. } if is_range_value(base));
                if !is_construction_write {
                    return false;
                }
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) if is_range_value(v) => return false,
            Some(crate::model::ExitSwitch::Fused { args, .. })
                if args.iter().any(is_range_value) =>
            {
                return false;
            }
            _ => {}
        }
        for link in &block.exits {
            if link
                .last_exception
                .as_ref()
                .and_then(LinkArg::as_variable)
                .is_some_and(is_range_value)
                || link
                    .last_exc_value
                    .as_ref()
                    .and_then(LinkArg::as_variable)
                    .is_some_and(is_range_value)
            {
                return false;
            }
        }
    }
    true
}

/// Remove the (single) op in the graph whose result is `result_var`.
fn remove_op_by_result(graph: &mut FunctionGraph, result_var: &Variable) {
    for block in &mut graph.blocks {
        if let Some(i) = block
            .operations
            .iter()
            .position(|op| op.result.as_ref() == Some(result_var))
        {
            block.operations.remove(i);
            return;
        }
    }
}

/// The signed constant `var` is produced by. Follows `Variable.copy()`
/// inputargs (`ssa.py` `SSA_to_SSI`) only when every incoming path
/// agrees (`DataFlowFamilyBuilder` unifies a single family).
fn const_int_of(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    const_int_of_inner(graph, var, &mut HashSet::new())
}

fn const_int_of_inner(
    graph: &FunctionGraph,
    var: &Variable,
    seen: &mut HashSet<u64>,
) -> Option<i64> {
    if !seen.insert(var.id()) {
        return None;
    }
    for block in &graph.blocks {
        for op in &block.operations {
            if let (Some(result), OpKind::ConstInt(value)) = (&op.result, &op.kind)
                && result == var
            {
                return Some(*value);
            }
        }
    }
    let mut agreed: Option<i64> = None;
    let mut saw_incoming = false;
    for block in graph.iterblocks() {
        if block.dead {
            continue;
        }
        let Some(slot) = block.inputargs.iter().position(|arg| arg == var) else {
            continue;
        };
        for pred in graph.iterblocks() {
            if pred.dead {
                continue;
            }
            for link in &pred.exits {
                if link.target != block.id {
                    continue;
                }
                let value = match link.args.get(slot) {
                    Some(LinkArg::Value(incoming)) => const_int_of_inner(graph, incoming, seen)?,
                    Some(LinkArg::Const(constant)) => match constant.value {
                        ConstValue::Int(value) => value,
                        _ => return None,
                    },
                    None => return None,
                };
                if let Some(prev) = agreed {
                    if prev != value {
                        return None;
                    }
                } else {
                    agreed = Some(value);
                }
                saw_incoming = true;
            }
        }
    }
    if saw_incoming { agreed } else { None }
}

/// `int_between(lo, x, upper)` bound to `result_var`.  The result is a
/// 0/1 integer in the signed register bank.  The predicate is
/// `lo <= x < upper` and requires `lo <= upper`; the caller proves that
/// from constant bounds before emitting this op.
fn int_between_op(
    result_var: &Variable,
    lo: Variable,
    x: Variable,
    upper: Variable,
) -> SpaceOperation {
    FunctionGraph::set_concretetype_of_inline(result_var, crate::model::ConcreteType::Signed);
    SpaceOperation {
        result: Some(result_var.clone()),
        kind: OpKind::LoweredBlackholeOp {
            opname: "int_between".to_string(),
            args: vec![lo, x, upper],
        },
    }
}

/// `bitand` of two ordered compares, bound to `result_var`.
///
/// Inclusive membership is `le(lo, x) && ge(hi, x)` (`lo <= x <= hi`).
/// Exclusive membership is `le(lo, x) && lt(x, hi)` (`lo <= x && x < hi`).
/// Exact for every pair of bounds, including an empty range (`lo > hi`)
/// and `hi == i64::MAX`.
fn build_range_contains_compare_pair(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    lo: Variable,
    hi: Variable,
    x: Variable,
    exclusive: bool,
) -> Vec<SpaceOperation> {
    let lo_le = graph.alloc_value_var();
    let hi_cmp = graph.alloc_value_var();
    let (hi_op, hi_lhs, hi_rhs) = if exclusive {
        ("lt", x.clone(), hi)
    } else {
        ("ge", hi, x.clone())
    };
    vec![
        SpaceOperation {
            result: Some(lo_le.clone()),
            kind: OpKind::BinOp {
                op: "le".to_string(),
                lhs: lo,
                rhs: x,
                result_ty: ValueType::Int,
            },
        },
        SpaceOperation {
            result: Some(hi_cmp.clone()),
            kind: OpKind::BinOp {
                op: hi_op.to_string(),
                lhs: hi_lhs,
                rhs: hi_rhs,
                result_ty: ValueType::Int,
            },
        },
        SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::BinOp {
                op: "bitand".to_string(),
                lhs: lo_le,
                rhs: hi_cmp,
                result_ty: ValueType::Int,
            },
        },
    ]
}

/// Replace `contains` with `int_between` when that op's contract holds,
/// and with the compare pair otherwise.
///
/// `int_between(n, m, p)` is `n <= m < p` and requires `n <= p`.
/// Exclusive signed-int membership is already that predicate (`p = hi`).
/// Inclusive signed-int membership needs a successor `upper = hi + 1`.
/// Both hold for constants precisely when `lo <= hi` and, on the
/// inclusive path, `hi + 1` does not wrap.  Float ranges have no
/// `int_between`; they always keep the two comparisons (`float_le` /
/// `float_lt` after jtransform prefixes the bare leaves).
fn build_range_contains_compares(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    lo: Variable,
    hi: Variable,
    x: Variable,
    exclusive: bool,
    elem: RangeElemKind,
) -> Vec<SpaceOperation> {
    if elem == RangeElemKind::Float {
        return build_range_contains_compare_pair(graph, result_var, lo, hi, x, exclusive);
    }
    let const_bounds = match (const_int_of(graph, &lo), const_int_of(graph, &hi)) {
        (Some(lo_value), Some(hi_value)) if lo_value <= hi_value => Some(hi_value),
        _ => None,
    };
    match (exclusive, const_bounds) {
        (true, Some(_)) => vec![int_between_op(result_var, lo, x, hi)],
        (false, Some(hi_value)) => match hi_value.checked_add(1) {
            Some(upper) => {
                let upper_v = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                vec![
                    SpaceOperation {
                        result: Some(upper_v.clone()),
                        kind: OpKind::ConstInt(upper),
                    },
                    int_between_op(result_var, lo, x, upper_v),
                ]
            }
            None => build_range_contains_compare_pair(graph, result_var, lo, hi, x, false),
        },
        // Keep the compare pair for every shape that cannot prove
        // `lo <= upper`: a non-constant `lo`, a non-constant `hi`,
        // `lo > hi`, and inclusive `hi == i64::MAX`.  `int_between`
        // requires `lo <= upper`.  A non-constant bound cannot
        // establish it, and `lo > hi` is an empty range — the wrapping
        // `int_sub` + `uint_lt` expansion answers true for some `x`
        // (for example `(10..=5)` against `10`).  Inclusive
        // `hi == i64::MAX` has no representable successor, so `hi + 1`
        // wraps.  The two comparisons are exact for every one of these
        // shapes, including an empty range.
        _ => build_range_contains_compare_pair(graph, result_var, lo, hi, x, exclusive),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "RangeInclusive", "new"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn contains_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "RangeInclusive", "contains"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn exclusive_contains_target() -> CallTarget {
        // Live exclusive `Range` is a transparent ADT, so `first_is_self`
        // emits `CallTarget::Method` (`__majit_wrap_math_isqrt`).
        CallTarget::method("contains", Some("Range".into()))
    }

    fn range_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "Range".to_string(),
        )
    }

    fn field_write(base: &Variable, name: &str, value: &Variable) -> OpKind {
        OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor {
                name: name.to_string(),
                owner_root: Some("core::ops::range::Range".to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        }
    }

    fn rewrite_inc(
        g: &mut FunctionGraph,
        new_sites: &[RangeInclusiveNewSite],
        contains_sites: &[RangeContainsSite],
    ) -> usize {
        rewire_range_contains_call_sites(g, new_sites, &[], contains_sites)
    }

    /// Match a `FunctionPath`'s trailing segments against `tail` so a
    /// crate-qualified spelling and the crate-stripped front-end spelling
    /// both resolve — the same contract as `front::mir::fmt_path_ends_with`.
    fn segments_end_with(segments: &[String], tail: &[&str]) -> bool {
        segments.len() >= tail.len()
            && segments[segments.len() - tail.len()..]
                .iter()
                .zip(tail)
                .enumerate()
                .all(|(i, (s, t))| {
                    if i + 1 == tail.len() {
                        crate::front::clause_spec::unspecialized_leaf(s) == *t
                    } else {
                        s.as_str() == *t
                    }
                })
    }

    fn contains_calls(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, args, .. }
                        if args.len() == 2 && is_range_contains_call_target(target)
                )
            })
            .count()
    }

    /// Count `FunctionPath` calls whose segments end with `tail`.
    fn functionpath_calls_ending(g: &FunctionGraph, tail: &[&str]) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments_end_with(segments, tail)
                )
            })
            .count()
    }

    /// `(result, args)` of every `int_between` in the graph.
    fn int_between_ops(g: &FunctionGraph) -> Vec<(Variable, Vec<Variable>)> {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|sop| match &sop.kind {
                OpKind::LoweredBlackholeOp { opname, args } if opname == "int_between" => Some((
                    sop.result.clone().expect("int_between result"),
                    args.clone(),
                )),
                _ => None,
            })
            .collect()
    }

    /// `int_between(lo, x, exclusive)` bound to `result`, and no leftover
    /// compare pair. Bound operands are defined in the contains block
    /// (`ssa.py` `SSA_to_SSI` / `Block.renamevariables`).
    fn assert_int_between(
        g: &FunctionGraph,
        result: &Variable,
        lo: &Variable,
        x: &Variable,
        exclusive: i64,
    ) {
        let mut found = None;
        for block in &g.blocks {
            for op in &block.operations {
                if let OpKind::LoweredBlackholeOp { opname, args } = &op.kind
                    && opname == "int_between"
                {
                    found = Some((
                        block.id,
                        op.result.clone().expect("int_between result"),
                        args.clone(),
                    ));
                }
            }
        }
        let (block, bound, args) = found.expect("one int_between");
        assert_eq!(&bound, result, "int_between reuses the contains result");
        assert_eq!(args.len(), 3);
        for (i, arg) in args.iter().enumerate() {
            assert!(
                g.variable_defined_in_block(block, arg),
                "int_between arg {i} must be defined in the contains block"
            );
        }
        assert_eq!(
            const_int_of(g, &args[0]),
            const_int_of(g, lo),
            "lower bound is lo"
        );
        assert_eq!(&args[1], x, "middle value is x");
        assert_eq!(
            const_int_of(g, &args[2]),
            Some(exclusive),
            "exclusive upper bound is hi + 1",
        );
        assert!(binop_results(g, "le").is_empty(), "no leftover le");
        assert!(binop_results(g, "ge").is_empty(), "no leftover ge");
        assert!(binop_results(g, "bitand").is_empty(), "no leftover bitand");
    }

    /// A phi that merges two different constants is not a ConstInt family
    /// (`DataFlowFamilyBuilder.complete`).
    #[test]
    fn const_int_of_rejects_phi_of_distinct_constants() {
        let mut g = FunctionGraph::new("const_int_mixed_phi");
        let entry = g.startblock;
        let left = g.create_block();
        let right = g.create_block();
        let (join, join_args) = g.create_block_with_arg_vars(1);
        let one = g.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let nine = g.push_op_var(entry, OpKind::ConstInt(9), true).unwrap();
        let cond = g
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        g.set_branch(entry, cond, left, vec![], right, vec![]);
        g.set_goto(left, join, vec![one]);
        g.set_goto(right, join, vec![nine]);
        assert_eq!(const_int_of(&g, &join_args[0]), None);
    }

    /// Result Variables of every `BinOp` with the given opname.
    fn binop_results(g: &FunctionGraph, op_name: &str) -> Vec<Variable> {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|sop| match &sop.kind {
                OpKind::BinOp { op, .. } if op == op_name => sop.result.clone(),
                _ => None,
            })
            .collect()
    }

    /// Build a two-block `new` → `contains` graph and assert the rewrite
    /// drops both residual FunctionPath calls (`new` + `contains`) and
    /// emits `int_between(lo, x, hi + 1)` bound to the original
    /// `contains` result var.
    #[test]
    fn rewrite_folds_new_contains_to_compares() {
        let mut g = FunctionGraph::new("test_range_contains");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        // Block C: one inputarg = the threaded range receiver.
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone(), x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range.clone(),
                lo: a.clone(),
                hi: b.clone(),
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the contains site must be rewritten");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            0,
            "residual new removed",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            0,
            "residual contains removed",
        );
        // `(0..=255).contains` → `int_between(0, x, 256)`.
        assert_int_between(&g, &contains, &a, &x, 256);
    }

    /// A second consumer of the range result (an iterator's stateful
    /// read, modeled here as a dummy op reading the threaded receiver)
    /// declines the fold.
    #[test]
    fn rewrite_declines_when_range_has_second_consumer() {
        let mut g = FunctionGraph::new("test_range_contains_second_consumer");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        // Second consumer: another op reads the `new` result directly.
        let _second = g.push_op_var(
            n,
            OpKind::UnaryOp {
                op: "invert".to_string(),
                operand: range.clone(),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone(), x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range.clone(),
                lo: a,
                hi: b,
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains,
            }],
        );
        assert_eq!(rewritten, 0, "a second range consumer declines the fold");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            1,
            "residual new survives",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            1,
            "residual contains survives",
        );
    }

    /// Framestate threading gives the successor a third identity for the
    /// range.  Carrying that alias beyond the middle `contains` block must
    /// keep both residual calls: deleting `new` would leave the successor's
    /// input undefined.
    #[test]
    fn rewrite_declines_when_successor_carries_fresh_range_alias() {
        let mut g = FunctionGraph::new("test_range_contains_successor_alias");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();

        // Middle block M consumes one alias in `contains`, then threads it
        // onward.  The successor S receives a fresh framestate inputarg and
        // consumes that alias independently.
        let (m, m_args) = g.create_block_with_arg_vars(1);
        let range_in_m = m_args[0].clone();
        let x = g.push_op_var(m, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                m,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_m.clone(), x]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (s, s_args) = g.create_block_with_arg_vars(1);
        let range_in_s = s_args[0].clone();
        let later_use = g
            .push_op_var(
                s,
                OpKind::UnaryOp {
                    op: "invert".to_string(),
                    operand: range_in_s,
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(s, Some(later_use));
        g.set_goto(m, s, vec![range_in_m]);
        g.set_goto(n, m, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains,
            }],
        );
        assert_eq!(rewritten, 0, "a successor range alias declines the fold");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            1,
            "residual new survives",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            1,
            "residual contains survives",
        );
    }

    /// A producer that is not the 2-arg `contains` FunctionPath (here a
    /// 1-arg call) declines (fail-safe).
    #[test]
    fn rewrite_declines_when_producer_not_contains_method() {
        let mut g = FunctionGraph::new("test_range_contains_wrong_producer");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        // A 1-arg call — not the 2-arg `contains` FunctionPath shape.
        let result = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(result.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite { result_var: result }],
        );
        assert_eq!(rewritten, 0, "a non-2-arg `contains` producer declines");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            1,
            "residual new survives",
        );
    }

    /// The range value rides framestate `Link.args` past the `contains`
    /// block into a successor that only forwards it onward (never reads it
    /// as an op operand) — the shape the real `setitem_bytearray`
    /// `(0..=255).contains(&v)` lowers to, where the loop-carried `v`
    /// framestate slot threads the dead range temporary through the whole
    /// tail CFG.  This is a DEAD forward, not a second consumer: after the
    /// fold removes `new`/`contains`, `prune_dead_phis` reclaims the dead
    /// inputarg / link args.  It must FOLD (contrast
    /// `rewrite_declines_when_successor_carries_fresh_range_alias`, where the
    /// successor READS the alias via an op).
    #[test]
    fn rewrite_folds_when_dead_range_alias_forwarded_past_contains() {
        let mut g = FunctionGraph::new("test_range_contains_dead_forward");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(255), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();

        // Block C: identity-preserved delivery of the range to `contains`.
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c.clone(), x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        // Successor S receives the range on a framestate link-arg but never
        // reads it — a dead forward.  Its only live op reads an unrelated
        // value (the `contains` result), and it returns that, not the range.
        let (s, s_args) = g.create_block_with_arg_vars(2);
        let range_in_s = s_args[0].clone();
        let other_in_s = s_args[1].clone();
        let ret = g
            .push_op_var(
                s,
                OpKind::UnaryOp {
                    op: "invert".to_string(),
                    operand: other_in_s,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(s, Some(ret));
        // C forwards the dead range plus the `contains` result onward.
        let _ = range_in_s;
        g.set_goto(c, s, vec![range_in_c.clone(), contains.clone()]);
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b,
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(
            rewritten, 1,
            "a dead range forward past contains must still fold"
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            0,
            "residual new removed",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            0,
            "residual contains removed",
        );
        assert_int_between(&g, &contains, &a, &x, 256);
    }

    /// A constant inclusive upper bound of `i64::MAX` has no `hi + 1`.
    /// Keep the two comparisons rather than wrapping the limit.
    #[test]
    fn rewrite_keeps_compares_when_upper_bound_is_i64_max() {
        let mut g = FunctionGraph::new("test_range_contains_max");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(i64::MAX), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the overflow case still folds");
        assert!(
            int_between_ops(&g).is_empty(),
            "i64::MAX must not wrap into int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
    }

    /// A non-constant inclusive upper bound may be `i64::MAX` at run
    /// time, so it keeps the two comparisons instead of `int_between`.
    #[test]
    fn rewrite_keeps_compares_for_a_nonconstant_upper_bound() {
        let mut g = FunctionGraph::new("test_range_contains_var_hi");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let hi_src = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let b = g
            .push_op_var(
                n,
                OpKind::UnaryOp {
                    op: "neg".to_string(),
                    operand: hi_src,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            int_between_ops(&g).is_empty(),
            "a run-time hi must not be wrapped into int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
        let _ = (a, b, x);
    }

    /// Constant bounds with `lo > hi` (`10..=5`) are an empty range.
    /// `int_between` assumes `lo <= upper`; keep the two comparisons.
    #[test]
    fn rewrite_keeps_compares_when_constant_lo_exceeds_hi() {
        let mut g = FunctionGraph::new("test_range_contains_empty");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(5), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(10), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "an empty constant range still folds");
        assert!(
            int_between_ops(&g).is_empty(),
            "lo > hi must not become int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
        let _ = (a, b, x);
    }

    /// A non-constant `lo` cannot prove `lo <= hi + 1`, even when `hi` is
    /// a constant other than `i64::MAX`.  Keep the two comparisons.
    #[test]
    fn rewrite_keeps_compares_for_a_nonconstant_lower_bound() {
        let mut g = FunctionGraph::new("test_range_contains_var_lo");
        let n = g.startblock;
        let lo_src = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let a = g
            .push_op_var(
                n,
                OpKind::UnaryOp {
                    op: "neg".to_string(),
                    operand: lo_src,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(42), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewrite_inc(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert!(
            int_between_ops(&g).is_empty(),
            "a run-time lo must not be wrapped into int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
        let _ = (a, b, x);
    }

    fn count_range_ctors(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    }
                )
            })
            .count()
    }

    fn count_field_writes(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(&op.kind, OpKind::FieldWrite { .. }))
            .count()
    }

    /// Exclusive `(0..10).contains(&x)` is already `0 <= x < 10`, so the
    /// fold emits `int_between(lo, x, hi)` (no `hi + 1`) and drops the
    /// `Range { start, end }` ctor + FieldWrites.
    #[test]
    fn rewrite_folds_exclusive_int_range_to_int_between() {
        let mut g = FunctionGraph::new("test_range_contains_exclusive");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: range_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::Range".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: field_write(&range, "start", &a),
        });
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: field_write(&range, "end", &b),
        });
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(4), true).unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: exclusive_contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[],
            &[RangeExclusiveNewSite {
                result_var: range,
                elem: RangeElemKind::SignedInt,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(
            rewritten, 1,
            "the exclusive contains site must be rewritten"
        );
        assert_eq!(count_range_ctors(&g), 0, "Range ctor removed");
        assert_eq!(count_field_writes(&g), 0, "start/end FieldWrites removed");
        assert_eq!(contains_calls(&g), 0, "residual contains removed");
        assert_int_between(&g, &contains, &a, &x, 10);
    }

    /// Inclusive float `(-1.0..=1.0).contains(&x)` has no `int_between`;
    /// it keeps `bitand(le, ge)` so jtransform can prefix `float_le` /
    /// `float_ge` from the operand bank.
    #[test]
    fn rewrite_folds_inclusive_float_range_to_compares() {
        let mut g = FunctionGraph::new("test_range_contains_float");
        let n = g.startblock;
        let a = g
            .push_op_var(n, OpKind::ConstFloat((-1.0f64).to_bits()), true)
            .unwrap();
        let b = g
            .push_op_var(n, OpKind::ConstFloat(1.0f64.to_bits()), true)
            .unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target(),
                    args: crate::model::call_args(vec![a.clone(), b.clone()]),
                    result_ty: ValueType::Ref(Some("RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g
            .push_op_var(c, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let contains = g
            .push_op_var(
                c,
                OpKind::Call {
                    target: contains_target(),
                    args: crate::model::call_args(vec![range_in_c, x]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        g.set_return(c, Some(contains.clone()));
        g.set_goto(n, c, vec![range.clone()]);

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                elem: RangeElemKind::Float,
            }],
            &[],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the float contains site must be rewritten");
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "new"]),
            0,
            "residual new removed",
        );
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "RangeInclusive", "contains"]),
            0,
            "residual contains removed",
        );
        assert!(
            int_between_ops(&g).is_empty(),
            "float range must not become int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "ge").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
    }
}
