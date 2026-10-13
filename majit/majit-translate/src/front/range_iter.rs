//! `for _ in a..b` (exclusive int `Range`) → `range(a, b)` + list-iter.
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
//! RPython models `range(a, b)` as a `SomeList` carrying `range_step`
//! (`builtin.builtin_range`), whose `RangeRepr` `make_iterator_repr` yields
//! the integer-induction `RangeIteratorRepr` — the orthodox path.  This pass
//! reroutes the exclusive-int-`Range` for-loop onto it:
//!
//! ```text
//!     r = Range{start=a, end=b}          // ctor + FieldWrite(start/end)
//!     ... for _ in r (reflexive into_iter alias) ... next(r)
//! becomes
//!     t = range(a, b)                    // `SomeList` (range_step = 1)
//!     r = iter(t)                        // `["core","slice","iter"]` → SomeIterator
//!     ... next(r) folded by front::iter_next into the native `next` op
//! ```
//!
//! `range` is emitted under the reserved `["__majit_range"]` spelling (see
//! [`range_builtin_call`]) and resolves through
//! `HOST_ENV.lookup_builtin("range")` →
//! `SomeBuiltin("range")` → `builtin_range` (`SomeList` with `range_step`);
//! `["core","slice","iter"]` is the same `iter`-op bridge the slice/Vec
//! for-loop uses (`is_concrete_iter_constructor`), and on a range-`SomeList`
//! receiver it still selects `RangeIteratorRepr`.  Rewriting the aggregate to
//! that pair makes `originates_from_iter_op` see the `iter` op, so the
//! ALREADY-CAPTURED `next` site folds through the existing diamond — no new
//! iteration machinery, no `is_iter_op_segments` change.
//!
//! ## Consumer gate + fail-safe
//!
//! Only a range whose value feeds EXACTLY its own construction
//! (`FieldWrite`) and this for-loop's `next` is diverted; a range read by any
//! other op (`v[a..b]` slice-index via `Index::index`, a `.start` field read,
//! a stored range) keeps the ordinary ADT ctor path — turning it into a
//! `SomeList`/iterator would break that consumer.  Every structural mismatch
//! declines (leaves the ctor + FieldWrite chain, census Skip), so a decline
//! never regresses a graph the legacy walker already handled.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized exclusive int-`Range { start, end }` aggregate captured
/// during body lowering (`front::mir`'s `Rvalue::Aggregate` arm).  Carries
/// the ctor result (the range value the reflexive `into_iter` aliases into
/// the loop iterator). The bounds are read from the live post-simplify
/// FieldWrite chain when the pass runs.
#[derive(Clone)]
pub(crate) struct RangeNewSite {
    /// The `SyntheticTransparentCtor` result (the `Range` value) — locates
    /// the ctor op and the `next` iterator's backward-traced origin.
    pub result_var: Variable,
}

/// Rewrite every captured exclusive-int-`Range` aggregate that feeds a
/// for-loop `next` into the `range(a, b)` builtin + list-iter pair, so the
/// `front::iter_next` diamond (which runs immediately after) folds the loop.
/// Fail-safe: a site that does not match the expected for-loop shape is left
/// as the ordinary ctor + FieldWrite chain (census Skip).  Returns the number
/// of range aggregates rewritten.
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

    // 2. Backward-trace the iterator operand to a captured int-`Range`
    //    aggregate (the reflexive `into_iter` aliases the range value into
    //    the loop iterator, so the trace reaches the ctor result var).
    let site = range_sites
        .iter()
        .find(|s| range_iter_origin_matches(flow, &iter_arg, &s.result_var))
        .ok_or_else(|| {
            format!(
                "{name}: next() iterator does not originate from a captured int-Range aggregate"
            )
        })?;
    let res = site.result_var.clone();

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

    // 4. Re-read the bounds from the live FieldWrite chain.  The capture is
    //    made while lowering the MIR aggregate, but `simplify_lowered_graph`
    //    runs before this pass and can replace a bound Variable while
    //    updating its consumers.  The surviving FieldWrites are therefore
    //    the authoritative post-simplify flow-graph values — exactly the
    //    graph the annotator will consume.  Keeping the captured Variables
    //    here made a valid loop decline when (for example) `save_point` was
    //    forwarded through an emptied block in `w_method_new`.
    let (start, end) = live_range_bounds(graph, &res).ok_or_else(|| {
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
            .retain(|op| !matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == &res));
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
                    op.result.as_ref() == Some(&res)
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
    let iter_var = res.clone();
    let iter_phi = (*iter_arg).clone();
    graph.blocks[cb].operations[ctor_idx] = range_builtin_call(tmp.clone(), start, end);
    graph.blocks[cb]
        .operations
        .insert(ctor_idx + 1, slice_iter_call(res, tmp));
    // The iterator-phi slot must receive the iterator on every predecessor,
    // including an entry edge that still names the end bound (`w_tuple_len`
    // in `exception_match`).  `FlowContext.mergeblock` /
    // `FrameState.getoutputargs` thread each block's live locals as that
    // block's inputargs, so a `Link.args` slot is a Variable defined in
    // the source (`flowspace/model.py checkgraph`).  Planting the
    // preheader `iter()` result on every predecessor breaks loop
    // back-edges: the latch does not define that result.  Keep a
    // source-defined alias; otherwise thread the header phi through the
    // source (`FunctionGraph::ensure_variable_at_block`, the `setstate`
    // backfill).
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
            if slot_var != Some(&iter_var) && slot_var != Some(&iter_phi) {
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
                    && iterator_alias_of(flow, v, &iter_var, &iter_phi) =>
            {
                true
            }
            _ => false,
        };
        if keep {
            continue;
        }
        let replacement = if graph.variable_defined_in_block(src, &iter_var) {
            iter_var.clone()
        } else if graph.variable_defined_in_block(src, &iter_phi) {
            iter_phi.clone()
        } else if graph.ensure_variable_at_block(src, &iter_phi) {
            iter_phi.clone()
        } else {
            continue;
        };
        if let Some(slot) = graph.blocks[bi].exits[ei].args.get_mut(pos) {
            *slot = LinkArg::Value(replacement);
        }
    }
    Ok(())
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
