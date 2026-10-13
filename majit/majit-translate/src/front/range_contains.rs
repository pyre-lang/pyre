//! `(a..=b).contains(&x)` → `int_between(a, x, b + 1)`;
//! `(a..b).contains(&x)` → `int_between(a, x, b)`.
//!
//! ## Positioning
//!
//! `RangeInclusive::new(a, b)` has an Opaque body in the LLBC, so the
//! front emits a residual `FunctionPath` call whose owner-qualified
//! segments end `["range", "RangeInclusive", "new"]` — an unregistered
//! callee the rtyper census Skips, dropping the enclosing graph to the
//! legacy walker.  `RangeInclusive::contains(&self, &x)` routes as an
//! owner-qualified `FunctionPath` too (the opaque foreign receiver has no
//! extracted body to route a `CallTarget::Method` through), ending
//! `["range", "RangeInclusive", "contains"]` with `args[0]` the
//! `&RangeInclusive` receiver and `args[1]` the `&x` value.  `new` is the
//! wall: it is emitted before `contains`, so the translation loop's
//! failure surfaces on `new` first.
//!
//! Exclusive `Range` (`a..b`) is a struct aggregate, not a `new` call:
//! `Rvalue::Aggregate(Adt(core::ops::range::Range), [a, b])` lowers as a
//! `SyntheticTransparentCtor` plus `FieldWrite("start")` / `FieldWrite("end")`.
//! `Range::contains(&self, &x)` has a struct receiver, so `first_is_self`
//! routes it as a 2-arg `CallTarget::Method { name: "contains" }`
//! (receiver `args[0]`, value `args[1]`).  Inclusive `RangeInclusive` is
//! an opaque foreign ADT, so its `contains` stays the owner-qualified
//! `FunctionPath` tail `["range", "RangeInclusive", "contains"]`.
//!
//! `int_between(n, m, p)` is the predicate `n <= m < p` and assumes
//! `n <= p`.  `RangeInclusive` is `a <= x <= b`, so the exclusive upper
//! bound is `b + 1`.  The fold emits `int_between(a, x, b + 1)` only when
//! both bounds are constants, `a <= b`, and `b != i64::MAX` (then `b + 1`
//! exists and `a <= b + 1`):
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
//! Exclusive `Range` is already `a <= x < b`, so the same op is
//! `int_between(a, x, b)` with no successor:
//!
//! ```text
//!     r = Range { start: a, end: b }         // ctor + FieldWrite(start/end)
//!     ...
//!     t = r.contains(&x)                     // residual `contains` call
//! becomes
//!     t = int_between(a, x, b)               // a <= x < b
//! ```
//!
//! Every other inclusive shape keeps `bitand(le(a, x), ge(b, x))`.  A
//! non-constant bound cannot prove `a <= b + 1`.  `b == i64::MAX` has no
//! representable successor, so `b + 1` wraps.  `a > b` is an empty range,
//! and the wrapping expansion of `int_between` answers true for some `x`
//! there.  Exclusive shapes that cannot prove `a <= b` keep
//! `bitand(le(a, x), gt(b, x))` (`a <= x && x < b`).
//!
//! ## Cross-block shape
//!
//! Each `TermKind::Call` closes its block with a `set_goto`, so `new`
//! (block N) and `contains` (block C) sit in adjacent blocks, C the
//! single successor of N's Call edge.  The `new` result rides N → C on a
//! goto link arg; the front's liveness-forwarding keeps the same
//! `Variable` identity across the edge (and onward toward the value's
//! drop), so the `contains` receiver may be that same Variable or a
//! fresh C inputarg fed positionally from it (framestate SSA).  The fold
//! traces the receiver back through the single-predecessor link-arg
//! chain to its `new` site, then threads the bounds `a` / `b` into block
//! C via [`FunctionGraph::ensure_variable_at_block`] (the same
//! single-predecessor carry-through) before splicing the compares.
//!
//! ## Why `new` must be removed explicitly
//!
//! No sweep reclaims a dead `OpKind::Call { FunctionPath }`: `is_pure_op`
//! returns `false` for `Call`, so `prune_dead_phis` never removes it, and
//! `remove_dead_aggregates` only handles `SyntheticTransparentCtor` +
//! `Box::new_uninit`.  A fold that emits the compares at `contains` and
//! lets `new` "die" is UNSAFE — the orphaned `new` still walls the graph.
//! So the fold removes the `new` call op itself; the now-dead threaded
//! range value / link args are reclaimed by `prune_dead_phis` (the caller
//! runs it after any rewrite).
//!
//! ## Scope + fail-safe
//!
//! Int-only (word-sized signed: `I64` / `Isize`, the same element gate as
//! the `RangeInclusive` capture).  Float ranges (`complex_pow`, the
//! asin/acos `(-1.0..=1.0)` pins) and the pyre `W_Range` Python object
//! are out of scope.  Iteration-form ranges (`for i in a..=b` /
//! `for i in a..b`) are excluded by the consumer-shape gate: an iterator
//! reads the range a second time (its stateful `exhausted` field, or
//! `next` on an exclusive `Range`) as an OP operand, so a range value
//! read by any op other than the single `contains` being folded declines.
//! Construction `FieldWrite`s of `start`/`end` on an exclusive aggregate
//! are the ctor, not a second consumer.  Link-arg threading of the range
//! value is liveness forwarding, not consumption, so it does not count.
//! Every mismatch declines (leaves BOTH residual calls / the ctor chain,
//! census Skip) — no regression.

use crate::flowspace::model::Variable;
use crate::model::{
    BlockId, CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// A recognized range construction captured during body lowering
/// (`front::mir`).  Inclusive sites are `RangeInclusive::new(lo, hi)`;
/// exclusive sites are the `Range { start, end }` aggregate.  Carries the
/// result var (the range) and the two int bounds so the fold can thread
/// them into the `contains` block.
#[derive(Clone)]
pub(crate) struct RangeInclusiveNewSite {
    /// The construction result (the range value) — locates the producer
    /// op and matches the `contains` receiver.
    pub result_var: Variable,
    /// Lower bound `a` — `int_between`'s first operand.
    pub lo: Variable,
    /// Upper bound `b`.  Inclusive: the op's exclusive limit is `b + 1`.
    /// Exclusive: `b` is already the exclusive limit.
    pub hi: Variable,
    /// `true` for exclusive `Range { start, end }` (`a..b`); `false` for
    /// inclusive `RangeInclusive::new` (`a..=b`).
    pub exclusive: bool,
}

/// A recognized `RangeInclusive::contains` (FunctionPath) /
/// `Range::contains` (Method, or FunctionPath) call site captured during
/// body lowering (`front::mir`).  Carries the result var (the membership
/// bool) — the fold reuses it for `int_between`.
#[derive(Clone)]
pub(crate) struct RangeContainsSite {
    /// The `contains` call result (the membership `bool`) — locates the
    /// producer op; reused as the `bitand` result.
    pub result_var: Variable,
}

/// Rewrite every recorded `(a..=b).contains(&x)` / `(a..b).contains(&x)`
/// call site into a bounds check.  Inclusive: `int_between(a, x, b + 1)`
/// when both bounds are constants with `a <= b` and `b != i64::MAX`,
/// otherwise the two comparisons.  Exclusive: `int_between(a, x, b)` when
/// both bounds are constants with `a <= b`, otherwise
/// `bitand(le(a, x), gt(b, x))`.  Fail-safe: a site that does not match
/// the expected construction → `contains` shape is left untouched (both
/// residual calls / the ctor chain survive, census Skip).  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_range_contains_call_sites(
    graph: &mut FunctionGraph,
    new_sites: &[RangeInclusiveNewSite],
    contains_sites: &[RangeContainsSite],
) -> usize {
    let mut rewritten = 0;
    for site in contains_sites {
        match rewire_one_range_contains_site(graph, site, new_sites) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave both residual calls; the unregistered `new`
                // callee keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_range_contains_site(
    graph: &mut FunctionGraph,
    site: &RangeContainsSite,
    new_sites: &[RangeInclusiveNewSite],
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
    //    range receiver, `args[1]` the `&x` value.  Inclusive
    //    `RangeInclusive` is opaque, so `contains` is an owner-qualified
    //    `FunctionPath` ending `["range", "RangeInclusive", "contains"]`.
    //    Exclusive `Range` is a struct, so `first_is_self` routes
    //    `contains` as `CallTarget::Method`; a FunctionPath spelling
    //    ending `["range", "Range", "contains"]` is accepted too.
    let (range_v, x, exclusive_contains) = match &graph.blocks[c_idx].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if args.len() == 2 => {
            if segments_end_with(segments, &["range", "RangeInclusive", "contains"]) {
                (args[0].clone(), args[1].clone(), false)
            } else if segments_end_with(segments, &["range", "Range", "contains"]) {
                (args[0].clone(), args[1].clone(), true)
            } else {
                return Err(format!(
                    "{name}: range contains producer op is not the 2-arg `contains` FunctionPath: {segments:?}"
                ));
            }
        }
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } if args.len() == 2 && name == "contains" => (args[0].clone(), args[1].clone(), true),
        other => {
            return Err(format!(
                "{name}: range contains producer op is not the 2-arg `contains` call: {other:?}"
            ));
        }
    };

    // 3. Match the construction site whose `result_var` threads to
    //    block C, with the same inclusive/exclusive polarity as the
    //    `contains` path.  Inclusive `new` is in block N (a single
    //    predecessor of C) and its result rides a goto link-arg that
    //    becomes a C inputarg, so `range_v` (the C-local receiver) is a
    //    distinct Variable.  Exclusive construction is the same
    //    threading of the aggregate result.  A site matches when its
    //    `result_var` is defined in a block reachable from C through
    //    single-predecessor edges only AND the receiver `range_v` is
    //    either that same Variable (rare, when identity is preserved) or
    //    a C inputarg fed exclusively from that definition.  The
    //    single-consumer gate below is what makes this sound; here we
    //    only need to pick the producing site.
    let c_block = graph.blocks[c_idx].id;
    let new_site = new_sites
        .iter()
        .find(|s| {
            s.exclusive == exclusive_contains
                && range_matches_new_site(graph, c_block, &range_v, &s.result_var)
        })
        .ok_or_else(|| {
            format!("{name}: range contains receiver traces to no matching construction site")
        })?;

    // Exclusive `Range { start, end }` bounds are re-read from the live
    // FieldWrite chain: `simplify_lowered_graph` can replace a bound
    // Variable while updating its consumers, so the captured `lo`/`hi`
    // may be stale.  Inclusive `new` args stay the captured Variables.
    let (lo, hi) = if new_site.exclusive {
        live_range_bounds(graph, &new_site.result_var).ok_or_else(|| {
            format!(
                "{name}: exclusive Range construction has no unique live start/end FieldWrite pair"
            )
        })?
    } else {
        (new_site.lo.clone(), new_site.hi.clone())
    };

    // 4. Consumer-shape gate (excludes iteration-form ranges, whose
    //    range carries a stateful `exhausted` field read a second time by
    //    an iterator's `into_iter` / `next`).  The range value must be
    //    read as an OP operand by exactly the `contains` op being folded;
    //    construction FieldWrites of `start`/`end` on an exclusive
    //    aggregate are the ctor, not a second consumer.  Dead framestate
    //    `Link.args` forwarding past the `contains` block is reclaimed by
    //    `prune_dead_phis` and does not count.
    if !range_value_single_op_consumer(
        graph,
        &new_site.result_var,
        &site.result_var,
        new_site.exclusive.then_some((&lo, &hi)),
    ) {
        return Err(format!(
            "{name}: range value has a second consumer — declining (iteration form?)"
        ));
    }

    // 5. Thread `lo` / `hi` into block C as inputargs (single-pred carry-
    //    through).  Decline if either cannot be threaded (orphan / multi-
    //    predecessor block) rather than panicking.
    if !can_thread_to_block(graph, c_block, &lo) || !can_thread_to_block(graph, c_block, &hi) {
        return Err(format!(
            "{name}: range construction bounds cannot thread to the contains block"
        ));
    }
    let lo_in_c = lo.clone();
    let hi_in_c = hi.clone();
    let ok = graph.ensure_variable_at_block(c_block, &lo_in_c)
        && graph.ensure_variable_at_block(c_block, &hi_in_c);
    if !ok {
        return Err(format!(
            "{name}: range construction bounds threading failed at the contains block"
        ));
    }

    // 6. Splice the bounds check into block C, replacing the `contains`
    //    op in place and reusing its result Variable.
    let inserts = build_range_contains_compares(
        graph,
        &site.result_var,
        lo_in_c,
        hi_in_c,
        x.into_variable(),
        new_site.exclusive,
    );
    let ops = &mut graph.blocks[c_idx].operations;
    ops.remove(call_idx);
    for (offset, op) in inserts.into_iter().enumerate() {
        ops.insert(call_idx + offset, op);
    }

    // 7. Remove the now-unread construction.  Inclusive: the residual
    //    `new` call.  Exclusive: the `start`/`end` FieldWrites then the
    //    aggregate ctor.  After removal `lo` / `hi` stay live (threaded
    //    to C); the dead threaded range inputarg / link args are
    //    reclaimed by `prune_dead_phis`.
    if new_site.exclusive {
        remove_range_field_writes(graph, &new_site.result_var);
    }
    remove_op_by_result(graph, &new_site.result_var);

    Ok(())
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
    exclusive_bounds: Option<(&Variable, &Variable)>,
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
                // Exclusive `Range { start, end }` construction writes
                // the captured bounds into the aggregate; those
                // FieldWrites are the ctor, not a second consumer.
                if exclusive_bounds
                    .is_some_and(|(lo, hi)| is_ctor_bound_field_write(&op.kind, new_result, lo, hi))
                {
                    continue;
                }
                // Some op other than the `contains` being folded reads
                // the range value — a second consumer (iteration form?).
                return false;
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

/// `true` when `kind` is a `start`/`end` FieldWrite of the exclusive
/// `Range` aggregate `range` storing the captured bound.
fn is_ctor_bound_field_write(
    kind: &OpKind,
    range: &Variable,
    lo: &Variable,
    hi: &Variable,
) -> bool {
    match kind {
        OpKind::FieldWrite {
            base,
            field,
            value: LinkArg::Value(v),
            ..
        } if base == range => {
            (field.name == "start" && v == lo) || (field.name == "end" && v == hi)
        }
        _ => false,
    }
}

/// The unique post-simplification `start` / `end` values written into one
/// exclusive `Range` shell.  A duplicate, missing, or non-variable field
/// value declines: the rewrite is valid only for the exact two-field Rust
/// `Range { start, end }` construction shape.
fn live_range_bounds(
    graph: &FunctionGraph,
    range_result: &Variable,
) -> Option<(Variable, Variable)> {
    let mut start = None;
    let mut end = None;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::FieldWrite {
            base,
            field,
            value: LinkArg::Value(value),
            ..
        } = &op.kind
        else {
            continue;
        };
        if base != range_result {
            continue;
        }
        let slot = match field.name.as_str() {
            "start" => &mut start,
            "end" => &mut end,
            _ => continue,
        };
        if slot.replace(value.clone()).is_some() {
            return None;
        }
    }
    Some((start?, end?))
}

/// Drop every `start` / `end` FieldWrite whose base is the exclusive
/// `Range` aggregate — after the fold the range value is unread, and a
/// surviving `setattr("start")` would wall on the removed ctor.
fn remove_range_field_writes(graph: &mut FunctionGraph, range_result: &Variable) {
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            !matches!(
                &op.kind,
                OpKind::FieldWrite { base, field, .. }
                    if base == range_result && (field.name == "start" || field.name == "end")
            )
        });
    }
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

/// The signed constant `var` is produced by, if its single producer is
/// `ConstInt`.
fn const_int_of(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| match (&op.result, &op.kind) {
                (Some(result), OpKind::ConstInt(value)) if result == var => Some(*value),
                _ => None,
            })
    })
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

/// Inclusive `lo <= x <= hi` as `bitand(le(lo, x), ge(hi, x))`; exclusive
/// `lo <= x < hi` as `bitand(le(lo, x), gt(hi, x))`.
///
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
    let hi_op = if exclusive { "gt" } else { "ge" };
    vec![
        SpaceOperation {
            result: Some(lo_le.clone()),
            kind: OpKind::BinOp {
                op: "le".to_string(),
                lhs: lo,
                rhs: x.clone(),
                result_ty: ValueType::Int,
            },
        },
        SpaceOperation {
            result: Some(hi_cmp.clone()),
            kind: OpKind::BinOp {
                op: hi_op.to_string(),
                lhs: hi,
                rhs: x,
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
/// `int_between(n, m, p)` is `n <= m < p` and requires `n <= p`.  Inclusive
/// `hi` therefore needs a successor `upper = hi + 1`, and the requirement
/// is `lo <= upper`.  Both hold for constants precisely when `lo <= hi`
/// and `hi + 1` does not wrap.  That successor is materialised as
/// `ConstInt(hi + 1)`.  Exclusive `hi` is already the exclusive limit, so
/// the op is `int_between(lo, x, hi)` when both bounds are constants and
/// `lo <= hi`.
fn build_range_contains_compares(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    lo: Variable,
    hi: Variable,
    x: Variable,
    exclusive: bool,
) -> Vec<SpaceOperation> {
    let exclusive_upper = match (const_int_of(graph, &lo), const_int_of(graph, &hi)) {
        (Some(lo_value), Some(hi_value)) if lo_value <= hi_value => {
            if exclusive {
                Some(hi_value)
            } else {
                // Inclusive: `int_between` only when `hi + 1` fits.
                // `checked_add` is `None` exactly for `hi == i64::MAX`.
                hi_value.checked_add(1)
            }
        }
        _ => None,
    };
    match exclusive_upper {
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
        // Keep the compare pair for every shape that cannot prove
        // `lo <= upper`: a non-constant `lo`, a non-constant `hi`,
        // `lo > hi`, and inclusive `hi == i64::MAX`.  `int_between`
        // requires `lo <= upper`.  A non-constant bound cannot establish
        // it, and `lo > hi` is an empty range — the wrapping `int_sub` +
        // `uint_lt` expansion answers true for some `x` (for example
        // `(10..=5)` against `10`).  Inclusive `hi == i64::MAX` has no
        // representable successor, so `hi + 1` wraps.  The two
        // comparisons are exact for every one of these shapes, including
        // an empty range.
        None => build_range_contains_compare_pair(graph, result_var, lo, hi, x, exclusive),
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

    fn exclusive_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "Range".to_string(),
        )
    }

    fn exclusive_contains_target() -> CallTarget {
        // Real exclusive `Range::contains` is `CallTarget::Method` because
        // the receiver is a struct ADT (`first_is_self`).
        CallTarget::Method {
            name: "contains".to_string(),
            receiver_root: Some("Range".into()),
            resolved_path: None,
            fun_decl_id: None,
            branch_payloads: None,
        }
    }

    fn exclusive_contains_functionpath_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ops", "range", "Range", "contains"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn count_contains_methods(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        ..
                    } if name == "contains"
                )
            })
            .count()
    }

    fn exclusive_field_write(base: &Variable, name: &str, value: &Variable) -> OpKind {
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

    fn count_range_ctors(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
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

    /// Exclusive `Range { start, end }` in block N, `contains` in block C.
    fn exclusive_new_contains_graph(
        name: &str,
        lo: i64,
        hi: i64,
        x_val: i64,
    ) -> (
        FunctionGraph,
        Variable,
        Variable,
        Variable,
        Variable,
        Variable,
    ) {
        let mut g = FunctionGraph::new(name);
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(lo), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(hi), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: exclusive_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::Range".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: exclusive_field_write(&range, "start", &a),
        });
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: exclusive_field_write(&range, "end", &b),
        });
        let (c, c_args) = g.create_block_with_arg_vars(1);
        let range_in_c = c_args[0].clone();
        let x = g.push_op_var(c, OpKind::ConstInt(x_val), true).unwrap();
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
        (g, range, a, b, x, contains)
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
    /// compare pair.
    fn assert_int_between(
        g: &FunctionGraph,
        result: &Variable,
        lo: &Variable,
        x: &Variable,
        exclusive: i64,
    ) {
        let sites = int_between_ops(g);
        assert_eq!(sites.len(), 1, "one int_between, got {sites:?}");
        let (bound, args) = &sites[0];
        assert_eq!(bound, result, "int_between reuses the contains result");
        assert_eq!(args.len(), 3);
        assert_eq!(&args[0], lo, "lower bound is lo");
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range.clone(),
                lo: a.clone(),
                hi: b.clone(),
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range.clone(),
                lo: a,
                hi: b,
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b,
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                exclusive: false,
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

        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                exclusive: false,
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

    /// `(1..1<<53).contains(&n)` — the isqrt gateway shape — folds to
    /// `int_between(1, n, 1<<53)` and drops the ctor + FieldWrites +
    /// residual `Range::contains`.
    #[test]
    fn rewrite_folds_exclusive_range_contains_to_int_between() {
        let (mut g, range, a, b, x, contains) =
            exclusive_new_contains_graph("test_exclusive_range_contains", 1, 1 << 53, 42);
        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                exclusive: true,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(
            rewritten, 1,
            "the exclusive contains site must be rewritten"
        );
        assert_eq!(count_range_ctors(&g), 0, "exclusive Range ctor removed");
        assert_eq!(
            count_contains_methods(&g),
            0,
            "residual exclusive contains removed",
        );
        assert_int_between(&g, &contains, &a, &x, 1 << 53);
        let _ = b;
    }

    /// FunctionPath `["range", "Range", "contains"]` is the same exclusive
    /// fold as `CallTarget::Method` — accepted when a site residualises
    /// without the Method hint.
    #[test]
    fn rewrite_folds_exclusive_functionpath_contains() {
        let (mut g, range, a, b, x, contains) =
            exclusive_new_contains_graph("test_exclusive_functionpath_contains", 1, 1 << 53, 42);
        // Replace the Method contains the helper emitted with the
        // FunctionPath spelling.
        for block in &mut g.blocks {
            for op in &mut block.operations {
                if let OpKind::Call { target, .. } = &mut op.kind
                    && matches!(target, CallTarget::Method { name, .. } if name == "contains")
                {
                    *target = exclusive_contains_functionpath_target();
                }
            }
        }
        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b.clone(),
                exclusive: true,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert_eq!(
            functionpath_calls_ending(&g, &["range", "Range", "contains"]),
            0
        );
        assert_int_between(&g, &contains, &a, &x, 1 << 53);
        let _ = b;
    }

    /// A second consumer of the exclusive range result declines the fold
    /// and leaves the ctor + contains residuals.
    #[test]
    fn rewrite_declines_exclusive_when_range_has_second_consumer() {
        let mut g = FunctionGraph::new("test_exclusive_range_second_consumer");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(1), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: exclusive_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::Range".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: exclusive_field_write(&range, "start", &a),
        });
        g.block_mut(n).operations.push(SpaceOperation {
            result: None,
            kind: exclusive_field_write(&range, "end", &b),
        });
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
                    target: exclusive_contains_target(),
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
                exclusive: true,
            }],
            &[RangeContainsSite {
                result_var: contains,
            }],
        );
        assert_eq!(rewritten, 0, "a second range consumer declines the fold");
        assert_eq!(count_range_ctors(&g), 1, "exclusive Range ctor survives");
        assert_eq!(
            count_contains_methods(&g),
            1,
            "residual exclusive contains survives",
        );
    }

    /// Constant exclusive bounds with `lo > hi` (`10..5`) are an empty
    /// range.  `int_between` assumes `lo <= upper`; keep `le` + `gt`.
    #[test]
    fn rewrite_keeps_compares_when_exclusive_lo_exceeds_hi() {
        let (mut g, range, a, b, x, contains) =
            exclusive_new_contains_graph("test_exclusive_range_empty", 10, 5, 10);
        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                exclusive: true,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "an empty exclusive range still folds");
        assert!(
            int_between_ops(&g).is_empty(),
            "lo > hi must not become int_between"
        );
        assert_eq!(binop_results(&g, "le").len(), 1);
        assert_eq!(binop_results(&g, "gt").len(), 1);
        assert_eq!(binop_results(&g, "bitand"), vec![contains]);
        let _ = x;
    }

    /// Exclusive `contains` with only an inclusive `new` site declines
    /// (polarity mismatch) — the float-range capture gate never records
    /// an exclusive site, so a leftover `Range::contains` stays residual.
    #[test]
    fn rewrite_declines_exclusive_contains_without_exclusive_ctor() {
        let (mut g, range, a, b, _x, contains) =
            exclusive_new_contains_graph("test_exclusive_float_decline", 0, 1, 0);
        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a,
                hi: b,
                exclusive: false,
            }],
            &[RangeContainsSite {
                result_var: contains,
            }],
        );
        assert_eq!(
            rewritten, 0,
            "exclusive contains does not pair with an inclusive new site"
        );
        assert_eq!(count_range_ctors(&g), 1, "exclusive Range ctor survives");
        assert_eq!(
            count_contains_methods(&g),
            1,
            "residual exclusive contains survives",
        );
    }

    /// Exclusive `hi == i64::MAX` is already the exclusive limit, so
    /// `int_between(lo, x, MAX)` is well-defined (no successor).
    #[test]
    fn rewrite_exclusive_i64_max_hi_uses_int_between() {
        let (mut g, range, a, b, x, contains) =
            exclusive_new_contains_graph("test_exclusive_range_max", 0, i64::MAX, 42);
        let rewritten = rewire_range_contains_call_sites(
            &mut g,
            &[RangeInclusiveNewSite {
                result_var: range,
                lo: a.clone(),
                hi: b,
                exclusive: true,
            }],
            &[RangeContainsSite {
                result_var: contains.clone(),
            }],
        );
        assert_eq!(rewritten, 1);
        assert_int_between(&g, &contains, &a, &x, i64::MAX);
    }
}
