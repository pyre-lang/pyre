//! `for _ in a..b` (exclusive int `Range`) and `for _ in a..=b`
//! (`RangeInclusive::new`) → `range(a, b)` / `range(a, b + 1)` + list-iter.
//!
//! ## Positioning
//!
//! A Rust `for _ in a..b` desugars to `IntoIterator::into_iter(a..b)` then a
//! `Iterator::next()` / `Option` match loop.  `a..b` lowers as an
//! `Rvalue::Aggregate(Adt(core::ops::range::Range), [a, b])`, which
//! `front::mir` emits as a `SyntheticTransparentCtor` + `FieldWrite("start")`
//! / `FieldWrite("end")` chain — i.e. `SomeInstance(classdef=
//! 'core.ops.range.Range')`.  The struct arm attaches no per-instantiation
//! suffix, so every `Range<T>` collapses onto one classdef and the `start` /
//! `end` fields union across `T` (`int ∪ <other>`), while the loop's
//! reflexive `into_iter` (`impl<I: Iterator> IntoIterator for I`) is an
//! identity alias that plants no `iter` op, so `front::iter_next`'s
//! `originates_from_iter_op` gate declines and the residual `next()` blocks
//! as `getattr(range, "next")`.
//!
//! The inclusive form `for _ in a..=b` is not an aggregate: MIR emits a call
//! `RangeInclusive::<T>::new(a, b)` (a `CallTarget::FunctionPath` whose
//! segments end in `["RangeInclusive", "new"]`), then the same reflexive
//! `into_iter` alias, then `<RangeInclusive<T> as Iterator>::next`.  RPython
//! has no inclusive range, so the orthodox spelling is `range(a, b + 1)`
//! (`builtin.builtin_range`).  A 64-bit inclusive upper bound is diverted
//! only when it is a known constant strictly less than `i64::MAX`, so
//! `b + 1` cannot overflow the Signed lowering.
//!
//! RPython models `range(a, b)` as a `SomeList` carrying `range_step`
//! (`builtin.builtin_range`), whose `RangeRepr` `make_iterator_repr` yields
//! the integer-induction `RangeIteratorRepr` — the orthodox path.  This pass
//! reroutes both int-range for-loops onto it:
//!
//! ```text
//!     r = Range{start=a, end=b}          // ctor + FieldWrite(start/end)
//!     ... for _ in r (reflexive into_iter alias) ... next(r)
//! becomes
//!     t = range(a, b)                    // `SomeList` (range_step = 1)
//!     r = iter(t)                        // `["core","slice","iter"]` → SomeIterator
//!     ... next(r) folded by front::iter_next into the native `next` op
//!
//!     r = RangeInclusive::new(a, b)      // Call FunctionPath
//!     ... for _ in r ... next(r)
//! becomes
//!     t = range(a, b + 1)
//!     r = iter(t)
//! ```
//!
//! `range` is emitted under the reserved `["__majit_range"]` spelling (see
//! [`range_builtin_call`]) and resolves through
//! `HOST_ENV.lookup_builtin("range")` →
//! `SomeBuiltin("range")` → `builtin_range` (`SomeList` with `range_step`);
//! `["core","slice","iter"]` is the same `iter`-op bridge the slice/Vec
//! for-loop uses (`is_concrete_iter_constructor`), and on a range-`SomeList`
//! receiver it still selects `RangeIteratorRepr`.  Rewriting the aggregate
//! or `RangeInclusive::new` call to that pair makes `originates_from_iter_op`
//! see the `iter` op, so the ALREADY-CAPTURED `next` site folds through the
//! existing diamond — no new iteration machinery, no `is_iter_op_segments`
//! change.
//!
//! ## Consumer gate + fail-safe
//!
//! Only a range whose value feeds EXACTLY its own construction
//! (`FieldWrite` on the exclusive aggregate; the inclusive `new` call
//! produces the value and does not read it) and this for-loop's `next` is
//! diverted; a range read by any other op (`v[a..b]` slice-index via
//! `Index::index`, a `.start` field read, a stored range) keeps the ordinary
//! ADT ctor / `RangeInclusive::new` path — turning it into a
//! `SomeList`/iterator would break that consumer.  Every structural mismatch
//! declines (leaves the ctor + FieldWrite chain or the `new` call, census
//! Skip), so a decline never regresses a graph the legacy walker already
//! handled. The inclusive `exhausted` flag is unobservable when the for-loop
//! is the range's only consumer, and an empty `a > b` range is empty in both
//! forms.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// `true` iff `segments` names `RangeInclusive::new`, including a
/// `FunctionDesc.cachedgraph` copy whose leaf is `new__spec_…`
/// (`clause_spec::spec_leaf`). The whole-program lowering specialises
/// the generic associated function; a for-loop `next` stays the
/// unspecialized `["ops","range","RangeInclusive","next"]` residual.
pub(crate) fn is_range_inclusive_new_segments(segments: &[String]) -> bool {
    segments.len() >= 2
        && segments[segments.len() - 2] == "RangeInclusive"
        && segments
            .last()
            .is_some_and(|leaf| leaf == "new" || leaf.starts_with("new__spec_"))
}

/// How a captured int-range site was constructed in MIR.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RangeSiteKind {
    /// `a..b` aggregate: `SyntheticTransparentCtor` + `start`/`end` FieldWrites.
    Exclusive,
    /// `RangeInclusive::new(a, b)` call. `narrow` is true when the element
    /// integer type is narrower than 64 bits, so `b + 1` cannot overflow the
    /// Signed lowering.
    Inclusive { narrow: bool },
}

/// A recognized int-range site captured during body lowering: either an
/// exclusive `Range { start, end }` aggregate (`front::mir`'s
/// `Rvalue::Aggregate` arm) or an inclusive `RangeInclusive::new(a, b)`
/// call (Call-terminator lowering).  Carries the range value the reflexive
/// `into_iter` aliases into the loop iterator. Exclusive bounds are read
/// from the live post-simplify FieldWrite chain; inclusive bounds are the
/// `new` call's two arguments, with `range(a, b + 1)` subject to the
/// 64-bit overflow gate.
#[derive(Clone)]
pub(crate) struct RangeNewSite {
    /// The range value — locates the ctor / `RangeInclusive::new` op and
    /// the `next` iterator's backward-traced origin.
    pub result_var: Variable,
    pub kind: RangeSiteKind,
}

/// Rewrite every captured int-range site that feeds a for-loop `next` into
/// the `range(a, b)` / `range(a, b + 1)` builtin + list-iter pair, so the
/// `front::iter_next` diamond (which runs immediately after) folds the loop.
/// Fail-safe: a site that does not match the expected for-loop shape is left
/// as the ordinary ctor + FieldWrite chain or `RangeInclusive::new` call
/// (census Skip). A 64-bit inclusive upper bound is diverted only when it
/// is a known constant strictly less than `i64::MAX`. Returns the number
/// of range sites rewritten.
pub(crate) fn rewire_range_iter_sites(
    graph: &mut FunctionGraph,
    range_sites: &[RangeNewSite],
    next_results: &[Variable],
) -> usize {
    // The origin trace and consumer-closure walks read only control flow
    // (block `inputargs` / `exits`), which the rewrite never mutates — it
    // touches only `operations` — so the value-flow adjacency is stable for
    // the whole pass.  Build it ONCE here instead of re-deriving it (with a
    // full-graph scan per hop) inside the helpers for every `next()` site ×
    // range candidate.
    let flow = ValueFlow::build(graph);
    let mut rewritten = 0;
    for next_opt in next_results {
        match rewire_one_range_iter_site(graph, &flow, next_opt, range_sites) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the ctor + FieldWrite chain; the shared Range
                // classdef keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

/// Value-flow adjacency over the graph's `Link` args ↔ target-block
/// `inputargs`, precomputed once per graph.  A positional `Link.arg[i]`
/// carrying variable `src` into a block whose `inputargs[i]` is `dst` is one
/// edge `src → dst`; `fwd` holds it forward (for the consumer-closure walk),
/// `back` reversed (for the iterator-origin trace).  Both walks read only
/// this adjacency, so a single build serves every `next()` site and range
/// candidate — turning each helper from a per-hop full-graph rescan into a
/// linear BFS.
struct ValueFlow {
    fwd: HashMap<Variable, Vec<Variable>>,
    back: HashMap<Variable, Vec<Variable>>,
}

impl ValueFlow {
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    fn build(graph: &FunctionGraph) -> Self {
        let mut fwd: HashMap<Variable, Vec<Variable>> = HashMap::new();
        let mut back: HashMap<Variable, Vec<Variable>> = HashMap::new();
        for block in &graph.blocks {
            for link in &block.exits {
                // `BlockId` doubles as the index into `graph.blocks`.
                let Some(target) = graph.blocks.get(link.target.0) else {
                    continue;
                };
                for (i, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(src) = arg else { continue };
                    let Some(dst) = target.inputargs.get(i) else {
                        continue;
                    };
                    fwd.entry(src.clone()).or_default().push(dst.clone());
                    back.entry(dst.clone()).or_default().push(src.clone());
                }
            }
        }
        Self { fwd, back }
    }
}

fn rewire_one_range_iter_site(
    graph: &mut FunctionGraph,
    flow: &ValueFlow,
    next_opt: &Variable,
    range_sites: &[RangeNewSite],
) -> Result<(), String> {
    let name = graph.name.clone();

    // 1. Locate the `next()` producer op and its iterator operand (args[0]).
    let (na, ni) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(next_opt))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: next() result var has no producer op"))?;
    let iter_arg = match &graph.blocks[na].operations[ni].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone(),
        other => {
            return Err(format!(
                "{name}: next() producer op is not a 1-arg call: {other:?}"
            ));
        }
    };

    // 2. Backward-trace the iterator operand to a captured int-range
    //    site (the reflexive `into_iter` aliases the range value into
    //    the loop iterator, so the trace reaches the ctor / `new` result var).
    let site = range_sites
        .iter()
        .find(|s| range_iter_origin_matches(flow, &iter_arg, &s.result_var))
        .ok_or_else(|| {
            format!(
                "{name}: next() iterator does not originate from a captured int-Range aggregate"
            )
        })?;
    let res = site.result_var.clone();
    let kind = site.kind;

    // 3. Consumer gate: the range value must feed ONLY its own construction
    //    (`FieldWrite`) and this for-loop's `next`.  A `v[a..b]` slice-index,
    //    a `.start` read, or a stored range reads it as a different op
    //    operand — diverting it to a `SomeList`/iterator would break that
    //    consumer, so decline.
    if !range_feeds_only_forloop(graph, flow, &res, next_opt) {
        return Err(format!(
            "{name}: range value has a non-iteration consumer — declining"
        ));
    }

    let iter_phi = (*iter_arg).clone();
    match kind {
        RangeSiteKind::Exclusive => rewire_exclusive_range_iter(graph, &name, &res)?,
        RangeSiteKind::Inclusive { narrow } => {
            rewire_inclusive_range_iter(graph, &name, &res, narrow)?
        }
    }
    fix_iterator_phi_links(graph, flow, &res, &iter_phi);
    Ok(())
}

/// Exclusive `a..b`: live FieldWrite bounds, sweep the FieldWrites, replace
/// the `SyntheticTransparentCtor` with `range(a, b)` + `iter`.
fn rewire_exclusive_range_iter(
    graph: &mut FunctionGraph,
    name: &str,
    res: &Variable,
) -> Result<(), String> {
    // 4. Re-read the bounds from the live FieldWrite chain.  The capture is
    //    made while lowering the MIR aggregate, but `simplify_lowered_graph`
    //    runs before this pass and can replace a bound Variable while
    //    updating its consumers.  The surviving FieldWrites are therefore
    //    the authoritative post-simplify flow-graph values — exactly the
    //    graph the annotator will consume.  Keeping the captured Variables
    //    here made a valid loop decline when (for example) `save_point` was
    //    forwarded through an emptied block in `w_method_new`.
    let (start, end) = live_range_bounds(graph, res).ok_or_else(|| {
        format!("{name}: range construction has no unique live start/end FieldWrite pair")
    })?;
    if !graph_defines(graph, &start) || !graph_defines(graph, &end) {
        return Err(format!("{name}: range bounds are not defined in the graph"));
    }

    // 5. Remove every `start` / `end` `FieldWrite` on the range value (any
    //    block): after the divert `res` is a `SomeIterator`, so a surviving
    //    `setattr("start")` would wall on it.
    for b in &mut graph.blocks {
        b.operations
            .retain(|op| !matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == res));
    }

    // 6. Replace the `SyntheticTransparentCtor` (still the producer of `res`
    //    after the FieldWrite sweep) with `t = range(a, b)` and bind `res`
    //    to `iter(t)` so the reflexive-into_iter alias and the captured
    //    `next` site see an iter-op-produced iterator.
    let (cb, ctor_idx) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| {
                    op.result.as_ref() == Some(res)
                        && matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::SyntheticTransparentCtor { .. },
                                ..
                            }
                        )
                })
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: range aggregate ctor op vanished"))?;
    let tmp = graph.alloc_value_var();
    graph.blocks[cb].operations[ctor_idx] = range_builtin_call(tmp.clone(), start, end);
    graph.blocks[cb]
        .operations
        .insert(ctor_idx + 1, slice_iter_call(res.clone(), tmp));
    Ok(())
}

/// Inclusive `a..=b`: replace `RangeInclusive::new(lo, hi)` with
/// `range(lo, hi + 1)` + `iter`. No FieldWrites to sweep.
fn rewire_inclusive_range_iter(
    graph: &mut FunctionGraph,
    name: &str,
    res: &Variable,
    narrow: bool,
) -> Result<(), String> {
    let (cb, call_idx) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(res))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: RangeInclusive::new producer vanished"))?;
    let (lo_arg, hi_arg) = match &graph.blocks[cb].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_range_inclusive_new_segments(segments) && args.len() == 2 => {
            (args[0].clone(), args[1].clone())
        }
        _ => {
            return Err(format!(
                "{name}: RangeInclusive::new producer is not a 2-arg FunctionPath ending in RangeInclusive::new"
            ));
        }
    };

    // Overflow gate before any mutation: a wide (64-bit) upper bound is
    // diverted only when `hi` is a known constant `c < i64::MAX`.
    if !narrow && !inclusive_hi_fits_signed(graph, &hi_arg) {
        return Err(format!(
            "{name}: 64-bit inclusive upper bound may overflow b+1"
        ));
    }
    inclusive_bound_ready(graph, &lo_arg, name)?;
    inclusive_bound_ready(graph, &hi_arg, name)?;

    let mut insert_at = call_idx;
    let lo = materialize_inclusive_bound(graph, cb, &mut insert_at, &lo_arg, name)?;
    let hi = materialize_inclusive_bound(graph, cb, &mut insert_at, &hi_arg, name)?;
    let one = graph.alloc_value_var();
    graph.blocks[cb].operations.insert(
        insert_at,
        SpaceOperation {
            result: Some(one.clone()),
            kind: OpKind::ConstInt(1),
        },
    );
    insert_at += 1;
    let hi1 = graph.alloc_value_var();
    graph.blocks[cb].operations.insert(
        insert_at,
        SpaceOperation {
            result: Some(hi1.clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: hi,
                rhs: one,
                result_ty: ValueType::Int,
            },
        },
    );
    insert_at += 1;
    let tmp = graph.alloc_value_var();
    graph.blocks[cb].operations[insert_at] = range_builtin_call(tmp.clone(), lo, hi1);
    graph.blocks[cb]
        .operations
        .insert(insert_at + 1, slice_iter_call(res.clone(), tmp));
    Ok(())
}

fn inclusive_bound_ready(graph: &FunctionGraph, arg: &LinkArg, name: &str) -> Result<(), String> {
    match arg {
        LinkArg::Value(v) => {
            if graph_defines(graph, v) {
                Ok(())
            } else {
                Err(format!("{name}: range bounds are not defined in the graph"))
            }
        }
        LinkArg::Const(c) => {
            c.value.as_int().map(|_| ()).ok_or_else(|| {
                format!("{name}: RangeInclusive::new arg is not an integer constant")
            })
        }
    }
}

/// Resolve a `RangeInclusive::new` operand to a graph Variable, inserting
/// a `ConstInt` immediately before the call when the operand is an integer
/// `LinkArg::Const`.
fn materialize_inclusive_bound(
    graph: &mut FunctionGraph,
    block: usize,
    insert_at: &mut usize,
    arg: &LinkArg,
    name: &str,
) -> Result<Variable, String> {
    match arg {
        LinkArg::Value(v) => {
            if !graph_defines(graph, v) {
                return Err(format!("{name}: range bounds are not defined in the graph"));
            }
            Ok(v.clone())
        }
        LinkArg::Const(c) => {
            let n = c.value.as_int().ok_or_else(|| {
                format!("{name}: RangeInclusive::new arg is not an integer constant")
            })?;
            let var = graph.alloc_value_var();
            graph.blocks[block].operations.insert(
                *insert_at,
                SpaceOperation {
                    result: Some(var.clone()),
                    kind: OpKind::ConstInt(n),
                },
            );
            *insert_at += 1;
            Ok(var)
        }
    }
}

/// `true` when `hi` is a known integer constant strictly less than
/// `i64::MAX` — a `LinkArg::Const`, or a Variable whose unique producer
/// is `OpKind::ConstInt`.
fn inclusive_hi_fits_signed(graph: &FunctionGraph, hi: &LinkArg) -> bool {
    let c = match hi {
        LinkArg::Const(c) => c.value.as_int(),
        LinkArg::Value(v) => unique_const_int_producer(graph, v),
    };
    matches!(c, Some(n) if n < i64::MAX)
}

fn unique_const_int_producer(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    let mut found = None;
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        if op.result.as_ref() != Some(var) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        match &op.kind {
            OpKind::ConstInt(n) => found = Some(*n),
            _ => return None,
        }
    }
    found
}

/// The iterator-phi slot must receive the iterator on every predecessor,
/// including an entry edge that still names the end bound (`w_tuple_len`
/// in `exception_match`).  `FlowContext.mergeblock` /
/// `FrameState.getoutputargs` thread each block's live locals as that
/// block's inputargs, so a `Link.args` slot is a Variable defined in
/// the source (`flowspace/model.py checkgraph`).  Planting the
/// preheader `iter()` result on every predecessor breaks loop
/// back-edges: the latch does not define that result.  Keep a
/// source-defined alias; otherwise thread the header phi through the
/// source (`FunctionGraph::ensure_variable_at_block`, the `setstate`
/// backfill).
fn fix_iterator_phi_links(
    graph: &mut FunctionGraph,
    flow: &ValueFlow,
    iter_var: &Variable,
    iter_phi: &Variable,
) {
    let n = graph.blocks.len();
    let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (ei, link) in block.exits.iter().enumerate() {
            if link.target.0 < n {
                incoming[link.target.0].push((bi, ei));
            }
        }
    }
    let mut fix: Vec<(usize, usize, usize)> = Vec::new();
    for (ti, preds) in incoming.iter().enumerate() {
        let arity = graph.blocks[ti].inputargs.len();
        for pos in 0..arity {
            let slot_var = graph.blocks[ti].inputargs.get(pos);
            if slot_var != Some(iter_var) && slot_var != Some(iter_phi) {
                continue;
            }
            for &(bi, ei) in preds {
                if graph.blocks[bi].exits[ei].args.get(pos).is_some() {
                    fix.push((bi, ei, pos));
                }
            }
        }
    }
    for (bi, ei, pos) in fix {
        let src = crate::model::BlockId(bi);
        let keep = match graph.blocks[bi].exits[ei].args.get(pos) {
            Some(LinkArg::Value(v))
                if graph.variable_defined_in_block(src, v)
                    && iterator_alias_of(flow, v, iter_var, iter_phi) =>
            {
                true
            }
            _ => false,
        };
        if keep {
            continue;
        }
        let replacement = if graph.variable_defined_in_block(src, iter_var) {
            iter_var.clone()
        } else if graph.variable_defined_in_block(src, iter_phi) {
            iter_phi.clone()
        } else if let Some(local) = graph.ensure_variable_at_block_local(src, iter_phi) {
            local
        } else {
            continue;
        };
        if let Some(slot) = graph.blocks[bi].exits[ei].args.get_mut(pos) {
            *slot = LinkArg::Value(replacement);
        }
    }
}

/// `true` iff `v` is the `iter()` result, the loop-header iterator phi, or
/// a block-inputarg alias that traces to the range value through
/// `ValueFlow` (`range_iter_origin_matches`).
fn iterator_alias_of(
    flow: &ValueFlow,
    v: &Variable,
    iter_var: &Variable,
    iter_phi: &Variable,
) -> bool {
    v == iter_var || v == iter_phi || range_iter_origin_matches(flow, v, iter_var)
}

/// `true` iff `iter_arg` traces — directly or through loop-carried block
/// inputargs — to the `Range` aggregate result `range_result`.  Mirrors
/// `iter_next::originates_from_iter_op`'s backward walk (a var that is a
/// block inputarg is traced to each predecessor's link arg in the matching
/// slot, so the loop-header iterator phi resolves through its entry edge to
/// the pre-loop range value), but keys on reaching the captured range value
/// instead of an `iter` op.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn range_iter_origin_matches(
    flow: &ValueFlow,
    iter_arg: &Variable,
    range_result: &Variable,
) -> bool {
    let mut visited: HashSet<Variable> = HashSet::new();
    let mut stack: Vec<Variable> = vec![iter_arg.clone()];
    while let Some(v) = stack.pop() {
        if !visited.insert(v.clone()) {
            continue;
        }
        if &v == range_result {
            return true;
        }
        // A var that is a block inputarg traces back to each predecessor's
        // link arg in the matching slot (`flow.back`); anything else
        // dead-ends.
        if let Some(srcs) = flow.back.get(&v) {
            stack.extend(srcs.iter().cloned());
        }
    }
    false
}

/// `true` when the range value is consumed by EXACTLY its own construction
/// (`FieldWrite` writes) and this for-loop's `next` op — the gate that keeps
/// a range read by any other op (slice-index, `.start`, a stored range) on
/// the ordinary ADT ctor path.  Computes the forward alias closure of the
/// range value (every Variable it flows into through positional `Link.args`,
/// including the renamed loop-iterator phi) and rejects if any op / exitswitch
/// / exception payload other than the construction writes or the `next` op
/// reads a closure member.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn range_feeds_only_forloop(
    graph: &FunctionGraph,
    flow: &ValueFlow,
    range_result: &Variable,
    next_opt: &Variable,
) -> bool {
    // Forward alias closure of the range value: every inputarg it flows into
    // through positional `Link.args` (including the renamed loop-iterator
    // phi), reached by BFS over the precomputed `fwd` adjacency.
    let mut closure: HashSet<Variable> = HashSet::new();
    let mut stack: Vec<Variable> = vec![range_result.clone()];
    while let Some(v) = stack.pop() {
        if !closure.insert(v.clone()) {
            continue;
        }
        if let Some(succs) = flow.fwd.get(&v) {
            stack.extend(succs.iter().cloned());
        }
    }

    let in_closure = |v: &Variable| closure.contains(v);
    for block in &graph.blocks {
        for op in &block.operations {
            let reads_range = crate::inline::op_variable_refs(&op.kind)
                .iter()
                .any(&in_closure);
            if !reads_range {
                continue;
            }
            // A construction `FieldWrite` (base in the closure) writes the
            // range's `start` / `end`; the `next` op reads the iterator.
            // Both are removed / folded by the rewrite; anything else is a
            // genuine second consumer.
            let is_construction_write =
                matches!(&op.kind, OpKind::FieldWrite { base, .. } if in_closure(base));
            let is_next = op.result.as_ref() == Some(next_opt);
            if !(is_construction_write || is_next) {
                return false;
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) if in_closure(v) => return false,
            Some(crate::model::ExitSwitch::Fused { args, .. }) if args.iter().any(&in_closure) => {
                return false;
            }
            _ => {}
        }
        for link in &block.exits {
            if link
                .last_exception
                .as_ref()
                .and_then(LinkArg::as_variable)
                .is_some_and(in_closure)
                || link
                    .last_exc_value
                    .as_ref()
                    .and_then(LinkArg::as_variable)
                    .is_some_and(in_closure)
            {
                return false;
            }
        }
    }
    true
}

/// `true` when `var` is produced by some op or is a block inputarg — i.e. it
/// is a well-defined value the rewritten `range()` call can reference.
fn graph_defines(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|b| {
        b.inputargs.iter().any(|iv| iv == var)
            || b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(var))
    })
}

/// The unique post-simplification `start` / `end` values written into one
/// captured `Range` shell.  A duplicate, missing, or non-variable field value
/// declines: the consumer rewrite is valid only for the exact two-field Rust
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

/// `t = range(start, end)` — the `range` builtin call the adapter resolves
/// through `HOST_ENV.lookup_builtin("range")` → `SomeBuiltin("range")` →
/// `builtin_range` (a `SomeList` carrying `range_step`).
///
/// Spelled with the reserved `__majit_range` segment rather than a bare
/// `["range"]`: the adapter resolves a single-segment path against the
/// `CallRegistry` before `HOST_ENV`, and pyre's registry is one flat
/// namespace keyed by leaf-only `CallPath`s, so a pyre free function named
/// `range` (`pyre_interpreter::module::_ast::convert::range`) registers as
/// `["range"]` and would capture this call. The reserved spelling has a
/// dedicated `translate_op` arm that binds the `HOST_ENV` singleton, keeping
/// the Arc identity `BUILTIN_TYPER` keys `rtype_builtin_range` on.
pub(crate) fn range_builtin_call(
    result: Variable,
    start: Variable,
    end: Variable,
) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![crate::runtime_names::shims::RANGE.to_string()],
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![start, end]),
            result_ty: ValueType::Ref(None),
        },
    }
}

/// `r = iter(t)` — the `["core","slice","iter"]` bridge op (the same
/// constructor the slice/Vec for-loop uses); on the range-`SomeList`
/// receiver `Repr::rtype_iter` selects `RangeIteratorRepr`.
pub(crate) fn slice_iter_call(result: Variable, container: Variable) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![container]),
            result_ty: ValueType::Ref(None),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn count_calls_ending(g: &FunctionGraph, tail: &[&str]) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(&op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.len() >= tail.len()
                            && segments[segments.len() - tail.len()..]
                                .iter()
                                .zip(tail)
                                .all(|(s, t)| s == t))
            })
            .count()
    }

    fn count_range_inclusive_new(g: &FunctionGraph) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if is_range_inclusive_new_segments(segments)
                )
            })
            .count()
    }

    fn dump_function_paths(g: &FunctionGraph) -> Vec<Vec<String>> {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect()
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

    /// Build a `for _ in a..b` shape — the range ctor + FieldWrites in the
    /// entry block, the reflexive-into_iter alias delivering the range to a
    /// loop header, `next(range)` in the header — and assert the rewrite
    /// drops the ctor + both FieldWrites and emits `range(a,b)` + `iter`.
    #[test]
    fn rewrite_diverts_int_range_forloop_to_range_plus_iter() {
        let mut g = FunctionGraph::new("test_range_iter");
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

        // Loop header H: inputarg = the iterator (entry-threaded range).
        let (h, h_args) = g.create_block_with_arg_vars(1);
        let it = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: crate::model::call_args(vec![it.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        g.set_goto(n, h, vec![range.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                kind: RangeSiteKind::Exclusive,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 1, "the range for-loop must be diverted");
        assert_eq!(count_range_ctors(&g), 0, "range ctor removed");
        assert_eq!(
            count_calls_ending(&g, &[crate::runtime_names::shims::RANGE]),
            1,
            "one range() builtin emitted"
        );
        assert_eq!(
            count_calls_ending(&g, &["core", "slice", "iter"]),
            1,
            "one iter op emitted"
        );
    }

    fn inclusive_new_target() -> CallTarget {
        CallTarget::function_path(["ops", "range", "RangeInclusive", "new"])
    }

    fn inclusive_new_spec_target() -> CallTarget {
        CallTarget::function_path([
            "ops",
            "range",
            "RangeInclusive",
            "new__spec_u8_0123456789abcdef",
        ])
    }

    /// `for _ in lo..=hi` — `RangeInclusive::new(lo, hi)` in the entry
    /// block, the reflexive-into_iter alias delivering the range to a loop
    /// header, `next(range)` in the header. `hi_op = None` makes `hi` a
    /// startblock inputarg (non-ConstInt).
    fn inclusive_range_forloop(
        name: &str,
        hi_op: Option<OpKind>,
    ) -> (FunctionGraph, Variable, Variable, Variable) {
        inclusive_range_forloop_with(name, hi_op, inclusive_new_target())
    }

    fn inclusive_range_forloop_with(
        name: &str,
        hi_op: Option<OpKind>,
        new_target: CallTarget,
    ) -> (FunctionGraph, Variable, Variable, Variable) {
        let mut g = FunctionGraph::new(name);
        let n = g.startblock;
        let lo = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let hi = match hi_op {
            Some(kind) => g.push_op_var(n, kind, true).unwrap(),
            None => {
                let v = g.alloc_value_var();
                g.block_mut(n).inputargs.push(v.clone());
                v
            }
        };
        let range = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: new_target,
                    args: crate::model::call_args(vec![lo, hi.clone()]),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeInclusive".into())),
                },
                true,
            )
            .unwrap();
        let (h, h_args) = g.create_block_with_arg_vars(1);
        let it = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("RangeInclusive".to_string())),
                    args: crate::model::call_args(vec![it.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        g.set_goto(n, h, vec![range.clone()]);
        (g, range, hi, opt)
    }

    fn assert_inclusive_diverted(g: &FunctionGraph, hi: &Variable) {
        assert_eq!(
            count_range_inclusive_new(g),
            0,
            "no RangeInclusive::new call left"
        );
        assert_eq!(
            count_calls_ending(g, &[crate::runtime_names::shims::RANGE]),
            1,
            "one range() builtin emitted"
        );
        assert_eq!(
            count_calls_ending(g, &["core", "slice", "iter"]),
            1,
            "one iter op emitted"
        );
        let add_result = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::BinOp { op: name, lhs, .. } if name == "add" && lhs == hi => {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("BinOp add whose lhs is hi");
        let range_second = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str)
                    == Some(crate::runtime_names::shims::RANGE)
                    && args.len() == 2 =>
                {
                    Some(args[1].clone())
                }
                _ => None,
            })
            .expect("range() second arg");
        assert_eq!(
            range_second.as_variable(),
            Some(&add_result),
            "range() second arg must be the add result, got {range_second:?}"
        );
    }

    fn assert_kind_of_canonical_diverts(graph: &FunctionGraph) {
        let n_new = count_range_inclusive_new(graph);
        let n_next = count_calls_ending(graph, &["RangeInclusive", "next"]);
        let n_range = count_calls_ending(graph, &[crate::runtime_names::shims::RANGE]);
        let n_iter = count_calls_ending(graph, &["core", "slice", "iter"]);
        assert!(
            n_range >= 1 && n_iter >= 1 && n_next == 0 && n_new == 0,
            "inclusive for-loop in kind_of_canonical_exc_class must divert: \
             new={n_new} next={n_next} range={n_range} iter={n_iter}\npaths={:#?}\n{}",
            dump_function_paths(graph),
            graph.dump()
        );
    }

    #[test]
    fn lower_kind_of_canonical_exc_class_diverts_inclusive_range() {
        use crate::front::mir::{
            build_semantic_program_from_llbcs_with_static_addrs_and_function_names, lower_function,
        };
        use majit_charon_reader::Llbc;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/llbc/pyre-object.ullbc"
        );
        let llbc = Llbc::load(path).expect("pyre-object.ullbc");
        let graph = lower_function(&llbc, "kind_of_canonical_exc_class")
            .expect("lower kind_of_canonical_exc_class");
        assert_kind_of_canonical_diverts(&graph);

        let llbc = Llbc::load(path).expect("pyre-object.ullbc");
        let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
            &[llbc],
            crate::HostStaticAddrs::default(),
            &[],
            &["kind_of_canonical_exc_class"],
        )
        .expect("semantic program");
        let func = program
            .functions
            .iter()
            .find(|f| f.name.contains("kind_of_canonical_exc_class"))
            .unwrap_or_else(|| {
                panic!(
                    "kind_of_canonical_exc_class missing; names={:?}",
                    program
                        .functions
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>()
                )
            });
        assert_kind_of_canonical_diverts(func.graph());
    }

    #[test]
    fn rewrite_diverts_inclusive_spec_suffixed_new() {
        let (mut g, range, hi, opt) = inclusive_range_forloop_with(
            "test_range_inclusive_spec_new",
            Some(OpKind::ConstInt(10)),
            inclusive_new_spec_target(),
        );
        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range,
                kind: RangeSiteKind::Inclusive { narrow: true },
            }],
            &[opt],
        );
        assert_eq!(
            rewritten, 1,
            "spec-suffixed RangeInclusive::new must be diverted"
        );
        assert_inclusive_diverted(&g, &hi);
    }

    /// Mirror of `rewrite_diverts_int_range_forloop_to_range_plus_iter` for
    /// `for _ in 0..=10` (`RangeInclusive::new`).
    #[test]
    fn rewrite_diverts_int_range_inclusive_forloop_to_range_plus_one() {
        let (mut g, range, hi, opt) =
            inclusive_range_forloop("test_range_inclusive_iter", Some(OpKind::ConstInt(10)));
        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range,
                kind: RangeSiteKind::Inclusive { narrow: true },
            }],
            &[opt],
        );
        assert_eq!(
            rewritten, 1,
            "the inclusive range for-loop must be diverted"
        );
        assert_inclusive_diverted(&g, &hi);
    }

    #[test]
    fn rewrite_inclusive_wide_constant_hi_is_diverted() {
        let (mut g, range, hi, opt) = inclusive_range_forloop(
            "test_range_inclusive_wide_const",
            Some(OpKind::ConstInt(10)),
        );
        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range,
                kind: RangeSiteKind::Inclusive { narrow: false },
            }],
            &[opt],
        );
        assert_eq!(
            rewritten, 1,
            "wide inclusive range with constant hi < i64::MAX must be diverted"
        );
        assert_inclusive_diverted(&g, &hi);
    }

    #[test]
    fn rewrite_inclusive_wide_nonconstant_hi_declines() {
        let (mut g, range, _hi, opt) =
            inclusive_range_forloop("test_range_inclusive_wide_nonconst", None);
        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range,
                kind: RangeSiteKind::Inclusive { narrow: false },
            }],
            &[opt],
        );
        assert_eq!(
            rewritten, 0,
            "wide inclusive range with non-ConstInt hi declines"
        );
        assert_eq!(
            count_range_inclusive_new(&g),
            1,
            "RangeInclusive::new call survives"
        );
        assert_eq!(
            count_calls_ending(&g, &[crate::runtime_names::shims::RANGE]),
            0,
            "no range() builtin emitted"
        );
    }

    #[test]
    fn rewrite_inclusive_wide_max_hi_declines() {
        let (mut g, range, _hi, opt) = inclusive_range_forloop(
            "test_range_inclusive_wide_max",
            Some(OpKind::ConstInt(i64::MAX)),
        );
        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range,
                kind: RangeSiteKind::Inclusive { narrow: false },
            }],
            &[opt],
        );
        assert_eq!(
            rewritten, 0,
            "wide inclusive range with hi == i64::MAX declines"
        );
        assert_eq!(
            count_range_inclusive_new(&g),
            1,
            "RangeInclusive::new call survives"
        );
    }

    /// The range value is threaded through TWO block hops (entry → mid →
    /// header) before `next()`, so the backward origin trace must cross two
    /// inputarg phis and the forward consumer closure must grow across both.
    /// Locks the transitive value-flow BFS that replaced the per-hop
    /// full-graph rescans.
    #[test]
    fn rewrite_diverts_range_threaded_through_two_blocks() {
        let mut g = FunctionGraph::new("test_range_iter_two_hops");
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

        // Intermediate block M just relays the range value onward.
        let (m, m_args) = g.create_block_with_arg_vars(1);
        let mid = m_args[0].clone();
        // Loop header H: inputarg = the iterator, reached only via M.
        let (h, h_args) = g.create_block_with_arg_vars(1);
        let it = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: crate::model::call_args(vec![it.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        g.set_goto(n, m, vec![range.clone()]);
        g.set_goto(m, h, vec![mid.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                kind: RangeSiteKind::Exclusive,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 1, "two-hop range for-loop must be diverted");
        assert_eq!(count_range_ctors(&g), 0, "range ctor removed");
        assert_eq!(
            count_calls_ending(&g, &[crate::runtime_names::shims::RANGE]),
            1,
            "one range() builtin emitted"
        );
        assert_eq!(
            count_calls_ending(&g, &["core", "slice", "iter"]),
            1,
            "one iter op emitted"
        );
        assert_link_args_defined_in_source(&g);
    }

    /// A predecessor that still names a non-iterator dummy must receive
    /// the header iterator as an SSI local (`ssa.py` `SSA_to_SSI`), not
    /// the header inputarg identity.
    #[test]
    fn rewrite_threads_iterator_ssi_local_onto_non_alias_predecessor() {
        let mut g = FunctionGraph::new("test_range_iter_ssi_pred");
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

        let mid = g.create_block();
        let dummy = g.push_op_var(mid, OpKind::ConstInt(0), true).unwrap();
        let header = g.create_block();
        g.block_mut(header).inputargs.push(range.clone());
        let opt = g
            .push_op_var(
                header,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: crate::model::call_args(vec![range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(header, Some(opt.clone()));
        g.set_goto(n, mid, vec![]);
        g.set_goto(mid, header, vec![dummy]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                kind: RangeSiteKind::Exclusive,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 1, "the range for-loop must be diverted");
        assert_link_args_defined_in_source(&g);
        let passed = g.block(mid).exits[0].args[0]
            .as_variable()
            .expect("iterator slot is a value");
        assert!(
            g.variable_defined_in_block(mid, passed),
            "predecessor link names the mid SSI local"
        );
        assert_ne!(
            passed, &range,
            "predecessor must not plant the header inputarg identity"
        );
    }

    fn assert_link_args_defined_in_source(graph: &FunctionGraph) {
        for (bi, block) in graph.blocks.iter().enumerate() {
            for link in &block.exits {
                for (arg_index, arg) in link.args.iter().enumerate() {
                    let Some(var) = arg.as_variable() else {
                        continue;
                    };
                    assert!(
                        graph.variable_defined_in_block(crate::model::BlockId(bi), var),
                        "undefined operand as Link.args[{arg_index}] entry \
                         (source block {bi} -> target block {})",
                        link.target.0
                    );
                }
            }
        }
    }

    /// `for _ in a..b { live }` — the range reaches the header, the body
    /// carries a loop-live local and the iterator alias, and the latch
    /// jumps back.  The rewrite must not plant the preheader `iter()`
    /// result on that back-edge (`FlowContext.mergeblock` /
    /// `FrameState.getoutputargs`).
    #[test]
    fn rewrite_loop_back_edge_threads_iterator_defined_in_source() {
        let mut g = FunctionGraph::new("test_range_iter_loop_back_edge");
        let n = g.startblock;
        let a = g.push_op_var(n, OpKind::ConstInt(0), true).unwrap();
        let b = g.push_op_var(n, OpKind::ConstInt(10), true).unwrap();
        let live = g.push_op_var(n, OpKind::ConstInt(1), true).unwrap();
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

        let (h, h_args) = g.create_block_with_arg_vars(2);
        let it = h_args[0].clone();
        let live_h = h_args[1].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: crate::model::call_args(vec![it.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (body, body_args) = g.create_block_with_arg_vars(2);
        let it_b = body_args[0].clone();
        let live_b = body_args[1].clone();
        let _keep_live = g
            .push_op_var(
                body,
                OpKind::UnaryOp {
                    op: "int_is_true".to_string(),
                    operand: live_b.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        g.set_goto(n, h, vec![range.clone(), live.clone()]);
        g.set_goto(h, body, vec![it.clone(), live_h.clone()]);
        g.set_goto(body, h, vec![it_b.clone(), live_b.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                kind: RangeSiteKind::Exclusive,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 1, "the looping range for-loop must be diverted");
        assert_eq!(count_range_ctors(&g), 0, "range ctor removed");
        assert_link_args_defined_in_source(&g);
        let latch = g
            .blocks
            .iter()
            .find(|block| {
                block
                    .exits
                    .iter()
                    .any(|link| link.target == h && block.id != n)
            })
            .expect("loop latch");
        let passed = latch.exits[0].args[0]
            .as_variable()
            .expect("iterator slot is a value");
        assert!(
            g.variable_defined_in_block(latch.id, passed),
            "back-edge iterator is a source-block inputarg or op result"
        );
        assert_ne!(
            passed, &range,
            "back-edge must not name the preheader iter() result"
        );
    }

    /// Synthetic LLBC for `for _ in 0..n { live += 1 }`: a Range aggregate,
    /// `next` in the header, and a body latch that does not mention the
    /// iterator.  `lower_function` must thread the iterator through that
    /// latch (`FlowContext.mergeblock`).
    #[test]
    fn lower_range_for_loop_back_edge_defines_link_args() {
        use crate::front::mir::lower_function;
        use majit_charon_reader::Llbc;

        let span = serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        });
        let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
        let unit_ty = serde_json::json!({
            "Adt": {"id": 2, "builtin": "Tuple", "generics": {
                "regions": [], "types": [], "const_generics": [], "trait_refs": []
            }}
        });
        let range_generics = serde_json::json!({
            "regions": [],
            "types": [usize_ty.clone()],
            "const_generics": [],
            "trait_refs": []
        });
        let range_ty = serde_json::json!({
            "Adt": {"id": 0, "generics": range_generics.clone()}
        });
        let option_ty = serde_json::json!({
            "Adt": {"id": 1, "generics": {
                "regions": [],
                "types": [usize_ty.clone()],
                "const_generics": [],
                "trait_refs": []
            }}
        });
        let item_meta = |path: &[&str]| {
            serde_json::json!({
                "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span,
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            })
        };
        let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
        let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
            serde_json::json!({
                "index": index,
                "name": name,
                "span": span,
                "ty": ty
            })
        };
        let stmt = |kind: serde_json::Value| serde_json::json!({"span": span, "kind": kind, "comments_before": []});
        let assign = |dest: u64, dest_ty: &serde_json::Value, rvalue: serde_json::Value| {
            stmt(serde_json::json!({"Assign": [place(dest, dest_ty), rvalue]}))
        };
        let copy =
            |index: u64, ty: &serde_json::Value| serde_json::json!({"Copy": place(index, ty)});
        let usize_lit = |k: &str| {
            serde_json::json!({
                "Const": [
                    {"Integer": {"Unsigned": ["Usize", k]}},
                    usize_ty.clone()
                ]
            })
        };
        let field_attr = serde_json::json!({
            "attributes": [], "inline": null, "rename": null, "public": true
        });
        let range_td = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["core", "ops", "range", "Range"]),
            "kind": {"Struct": [
                {"name": "start", "ty": usize_ty.clone(), "attr_info": field_attr},
                {"name": "end", "ty": usize_ty.clone(), "attr_info": field_attr}
            ]}
        });
        let option_td = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta(&["core", "option", "Option"]),
            "kind": {"Enum": [
                {"name": "None", "fields": [], "discriminant": 0, "attr_info": null},
                {"name": "Some", "fields": [
                    {"name": "0", "ty": usize_ty.clone(), "attr_info": null}
                ], "discriminant": 1, "attr_info": null}
            ]}
        });
        let unit_td = serde_json::json!({
            "def_id": 2,
            "item_meta": item_meta(&["core", "tuple", "Tuple"]),
            "kind": {"Struct": []}
        });
        let next_fn = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta(&["core", "iter", "traits", "iterator", "Iterator", "next"]),
            "signature": {
                "is_unsafe": false,
                "inputs": [range_ty.clone()],
                "output": option_ty.clone()
            },
            "body": "Opaque"
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "loop_range"]),
            "signature": {
                "is_unsafe": false,
                "inputs": [usize_ty.clone()],
                "output": unit_ty.clone()
            },
            "body": {"Unstructured": {
                "span": span,
                "locals": {"arg_count": 1, "locals": [
                    local(0, None, &unit_ty),
                    local(1, Some("n"), &usize_ty),
                    local(2, Some("start"), &usize_ty),
                    local(3, Some("range"), &range_ty),
                    local(4, Some("opt"), &option_ty),
                    local(5, Some("live"), &usize_ty)
                ]},
                "body": [
                    {
                        "statements": [
                            assign(2, &usize_ty, serde_json::json!({
                                "Use": [usize_lit("0"), "No"]
                            })),
                            assign(5, &usize_ty, serde_json::json!({
                                "Use": [usize_lit("1"), "No"]
                            })),
                            assign(3, &range_ty, serde_json::json!({
                                "Aggregate": [
                                    {"Adt": [
                                        {"id": 0, "generics": range_generics.clone()},
                                        0,
                                        null
                                    ]},
                                    [copy(2, &usize_ty), copy(1, &usize_ty)]
                                ]
                            }))
                        ],
                        "terminator": {"span": span, "kind": {"Goto": {"target": 1}}}
                    },
                    {
                        "statements": [],
                        "terminator": {"span": span, "kind": {"Call": {
                            "call": {
                                "func": {"Regular": {
                                    "kind": {"Fun": 1},
                                    "generics": {
                                        "regions": [], "types": [],
                                        "const_generics": [], "trait_refs": []
                                    }
                                }},
                                "args": [copy(3, &range_ty)],
                                "dest": place(4, &option_ty)
                            },
                            "target": 2,
                            "on_unwind": 4
                        }}}
                    },
                    {
                        "statements": [
                            assign(5, &usize_ty, serde_json::json!({
                                "BinaryOp": ["Add", copy(5, &usize_ty), usize_lit("1")]
                            }))
                        ],
                        "terminator": {"span": span, "kind": {"Goto": {"target": 3}}}
                    },
                    {
                        "statements": [],
                        "terminator": {"span": span, "kind": {"Goto": {"target": 1}}}
                    },
                    {
                        "statements": [],
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }}
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [range_td, option_td, unit_td],
                "fun_decls": [caller, next_fn],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("loop_range Llbc parses");
        let graph = lower_function(&llbc, "loop_range").expect("lower loop_range");
        assert_link_args_defined_in_source(&graph);
        assert!(
            graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &[crate::runtime_names::shims::RANGE.to_string()]
                        || segments == &["core".to_string(), "slice".to_string(), "iter".to_string()]
                )),
            "the exclusive-int Range loop must divert to range()+iter()\n{}",
            graph.dump()
        );
    }

    /// A range whose value is also read by a non-iteration op (here a
    /// `.start` field read modeled as a unary op on the range) keeps the
    /// ordinary ctor path (fail-safe): diverting it to an iterator would
    /// break that reader.
    #[test]
    fn rewrite_declines_when_range_has_second_consumer() {
        let mut g = FunctionGraph::new("test_range_iter_second_consumer");
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
        // Second consumer: an op reads the range value directly.
        let _second = g.push_op_var(
            n,
            OpKind::UnaryOp {
                op: "invert".to_string(),
                operand: range.clone(),
                result_ty: ValueType::Ref(None),
            },
            true,
        );

        let (h, h_args) = g.create_block_with_arg_vars(1);
        let it = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: crate::model::call_args(vec![it.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        g.set_goto(n, h, vec![range.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                kind: RangeSiteKind::Exclusive,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 0, "a second range consumer declines the divert");
        assert_eq!(count_range_ctors(&g), 1, "range ctor survives");
        assert_eq!(
            count_calls_ending(&g, &[crate::runtime_names::shims::RANGE]),
            0,
            "no range() builtin emitted"
        );
    }

    /// The header phi is the ctor result itself. The entry edge still
    /// passes the end bound and must be retargeted to `iter()`, because
    /// that block defines the iterator. The latch passes a carried local
    /// and does not define the preheader `iter()` result, so its link arg
    /// stays put.
    #[test]
    fn rewrite_retargets_entry_end_bound_and_keeps_latch_arg() {
        let mut g = FunctionGraph::new("test_range_iter_latch_arg");
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

        let (h, _) = g.create_block_with_arg_vars(1);
        // Framestate threads the range value as this block's inputarg, so
        // the phi variable is the ctor result, not a fresh name.
        g.block_mut(h).inputargs = vec![range.clone()];
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: CallTarget::method("next", Some("Range".to_string())),
                    args: crate::model::call_args(vec![range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(h, Some(opt.clone()));
        // Entry still names the end bound. The latch carries its own local.
        g.set_goto(n, h, vec![b.clone()]);
        let (latch, latch_args) = g.create_block_with_arg_vars(1);
        let carried = latch_args[0].clone();
        g.set_goto(latch, h, vec![carried.clone()]);

        let rewritten = rewire_range_iter_sites(
            &mut g,
            &[RangeNewSite {
                result_var: range.clone(),
                kind: RangeSiteKind::Exclusive,
            }],
            &[opt],
        );
        assert_eq!(rewritten, 1, "the range for-loop must be diverted");
        let entry_arg = &g.block(n).exits[0].args[0];
        assert!(
            matches!(entry_arg, LinkArg::Value(v) if v == &range),
            "entry edge that defines the iterator must pass it, got {entry_arg:?}"
        );
        let latch_arg = &g.block(latch).exits[0].args[0];
        assert!(
            matches!(latch_arg, LinkArg::Value(v) if v == &carried),
            "latch must keep the carried iterator, got {latch_arg:?}"
        );
    }
}
