//! `&s[k..]` / `&s[..k]` sub-slice and WTF-8 string indexes → orthodox
//! `getslice` copies.
//!
//! These recognizers are CAPSTONES, not independently wired primitives.
//! Range/RangeFrom are carried as synthetic markers until the flowspace
//! adapter, where they expand to ordinary `getslice` operations. This keeps
//! the slice operation out of any graph that drops to the legacy walker,
//! matching the existing RangeTo marker split.
//!
//! No arm carries an `end <= len` proof, and upstream asks for none:
//! `ll_listslice_startstop` (`rpython/rtyper/rlist.py`) CLAMPS an
//! oversized stop (`if stop > length: stop = length`) and states the rest as
//! `ll_assert`s, which translate out.  Non-negativity is proven one layer up,
//! in the annotator (`check_negative_slice`, `rpython/annotator/unaryop.py`),
//! and pyre's port raises there identically.  A `Range { start, end }` is the
//! direct Rust spelling of what PyPy writes as `buffer[start:end]`
//! (`RStringIO.read`, `rpython/rlib/rStringIO.py:148`;
//! `BufferedMixin._raw_write`, `pypy/module/_io/interp_bufferedio.py`),
//! so it is admitted on the same terms.  The frontend likewise strips
//! Rust's implicit range assertions as `TermKind::Assert`, following
//! `backendopt/removeassert.py`; retaining an extra proof only for the opaque
//! `Index::index` shim would be a non-upstream control-flow restriction.
//!
//! RangeTo uses the same StartStop contract for every defined stop. As in
//! RPython's AbstractBaseListRepr.rtype_getslice and ll_listslice_startstop,
//! no static-length, dominance, or array-mutation proof is needed: the runtime
//! helper receives the original stop and clamps it against the live length.
//!
//! The earlier failure was an unwired frontend `getslice` planted before a
//! graph fell through the legacy walker. Every admitted path now uses an
//! unregistered synthetic call expanded only by the rtyper flowspace adapter,
//! so no frontend `getslice` or Void bound is planted.
//!
//! ## Positioning
//!
//! `&s[k..]` on a `&[T]` slice desugars to
//! `<[T] as Index<RangeFrom<usize>>>::index(s, RangeFrom { start: k })`.
//! Its sibling `&s[..k]` uses `RangeTo { end: k }` and rewrites to
//! `getslice(s, 0, k)` (`SliceKind::StartStop`).
//! `lower_call` keeps `core::slice::index::<Impl>::index`,
//! `rustpython_wtf8::Wtf8::index`, and range-indexed `Vec::index` as
//! unregistered residuals (the gh#346 census wall); the scalar `Vec::index`
//! integer case is intercepted earlier (`is_vec_index_call`). The
//! `RangeFrom { start }` operand is built by `front::mir`'s
//! `Rvalue::Aggregate` arm as a `SyntheticTransparentCtor("RangeFrom")` +
//! `FieldWrite("start")` chain — i.e.
//! `SomeInstance(classdef='core.ops.range.RangeFrom')`, the same shape
//! `front::range_iter` sees for a `Range { start, end }` for-loop.
//!
//! RPython models `l[start:]` as a **copy**: the annotator's `getslice`
//! handler (`unaryop.py`) returns a fresh `listdef.offspring`, and the
//! rtyper lowers it through `AbstractBaseListRepr.rtype_getslice`
//! (`rlist.py`) to a `gendirectcall` of `ll_listslice_startonly`.
//! There is no borrowed-view concept upstream. This pass reroutes the residual
//! `slice::index` onto that orthodox path:
//!
//! ```text
//!     r = RangeFrom { start = k }        // ctor + FieldWrite(start)
//!     v = slice::index(s, r)             // residual index call
//! becomes
//!     v = getslice(s, k, None)           // StartOnly: stop is a const None
//! ```
//!
//! The `getslice` operand contract (`decompose_slice_args`,
//! `rtyper.rs`) selects `SliceKind::StartOnly` when args[2] (`stop`)
//! annotates as a constant `None`; `k` (a proven-nonnegative `usize` start)
//! becomes args[1]. The adapter supplies the annotated constant `None`, so the
//! `stop_is_none` branch fires and the `len-k` copy runs at runtime.
//!
//! ## Consumer gate + fail-safe (the UNWIRED hazard)
//!
//! `getslice` is UNWIRED at the assembler — a bare `getslice` opname has no
//! blackhole handler. In a graph that fully rtypes, `rtype_getslice` replaces
//! the op with a `direct_call` before assembly, so no bare primitive survives.
//! But a graph that plants `getslice` and then DROPS to the legacy walker
//! (blocked by a co-wall the recognizer does not touch) would carry the raw op
//! to the assembler default arm and break
//! `default_bh_builder_unwired_set_matches_task_85_snapshot` — the same hazard
//! the vec!/`NewList` recognizer has (`design-346-foreign-lib-cluster-epic.md`).
//! Unlike `slice_first`, whose planted ops (`ArrayRead`/`ConstInt`/`Some`/
//! `None`) are all wired, this pass is only safe once the rewritten graph
//! rtypes. The rewrite is therefore fail-safe by construction: a range whose
//! value feeds anything other than exactly its own construction
//! (`FieldWrite`) and this one `slice::index` call is left as the residual
//! ADT ctor + call chain (census Skip), so a decline never regresses a graph
//! the legacy walker already handled, and the snapshot test is the empirical
//! gate that no dropped graph carries a bare `getslice`.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized `RangeFrom { start }` aggregate feeding a residual
/// `core::slice::index::<Impl>::index(slice, range)` call, captured during
/// body lowering (`front::mir`'s `Rvalue::Aggregate` arm). Carries the ctor
/// result (the `RangeFrom` value the index call consumes as its `range`
/// operand) and the `start` bound threaded into the `getslice` op.
#[derive(Clone)]
pub(crate) struct SliceIndexRangeFromSite {
    /// The `SyntheticTransparentCtor` result (the `RangeFrom` value) — locates
    /// the ctor op and the consuming `slice::index` call's `range` operand.
    pub range_result: Variable,
    /// The `start` bound `k` — the `getslice` `start` argument (args[1]).
    pub start: Variable,
}

/// A recognized `RangeTo { end }` aggregate feeding a residual
/// `core::slice::index::<Impl>::index(slice, range)` call.
#[derive(Clone)]
pub(crate) struct SliceIndexRangeToSite {
    pub range_result: Variable,
    pub end: Variable,
}

/// A recognized `Range { start, end }` aggregate feeding a residual scalar
/// slice-index call. This is RPython's ordinary `getslice(start, stop)` shape;
/// bounds are retained separately so the transparent Rust range shell can be
/// removed after the exact-consumer proof succeeds.
#[derive(Clone)]
pub(crate) struct SliceIndexRangeSite {
    pub range_result: Variable,
    pub start: Variable,
    pub end: Variable,
}

pub(crate) fn rewire_slice_index_range_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceIndexRangeSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        let outcome = rewire_one_slice_index_site(
            graph,
            &site.range_result,
            SliceIndexBounds::Range {
                start: &site.start,
                end: &site.end,
            },
        );
        match outcome {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

/// Rewrite every captured `RangeFrom` aggregate that feeds exactly one
/// residual `slice::index` call into an orthodox `getslice(slice, start,
/// None)` op. Fail-safe: a site that does not match the expected single-
/// consumer shape is left as the ordinary ctor + FieldWrite + residual call
/// chain (census Skip). Returns the number of sites rewritten.
pub(crate) fn rewire_slice_index_rangefrom_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceIndexRangeFromSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_index_site(
            graph,
            &site.range_result,
            SliceIndexBounds::RangeFrom { start: &site.start },
        ) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the ctor + FieldWrite + residual `slice::index` call;
                // the unregistered callee keeps the rtyper census Skip for
                // this graph.
            }
        }
    }
    rewritten
}

/// Rewrite captured `RangeTo { end }` indexes through the synthetic-call
/// channel. Preserve the actual stop for StartStop; only the receiver's
/// `len - 1` selects RPython's specialized MinusOne helper.
pub(crate) fn rewire_slice_index_rangeto_sites(
    graph: &mut FunctionGraph,
    sites: &[SliceIndexRangeToSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_slice_index_rangeto_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual chain intact so legacy keeps its prior
                // census Skip and never sees a newly planted `getslice`.
            }
        }
    }
    rewritten
}

#[derive(Clone, Copy)]
enum SliceIndexBounds<'a> {
    RangeFrom {
        start: &'a Variable,
    },
    Range {
        start: &'a Variable,
        end: &'a Variable,
    },
    MinusOne {
        end: &'a Variable,
    },
    RangeTo {
        end: &'a Variable,
    },
}

fn rewire_one_slice_index_rangeto_site(
    graph: &mut FunctionGraph,
    site: &SliceIndexRangeToSite,
) -> Result<(), String> {
    let range = &site.range_result;
    let slice = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| {
            let OpKind::Call { args, .. } = &op.kind else {
                return None;
            };
            (is_slice_range_index_call(&op.kind) && args.len() == 2 && args[1] == *range)
                .then(|| args[0].clone().into_variable())
        })
        .ok_or_else(|| format!("{}: no RangeTo index consumer", graph.name))?;
    let bounds = if minus_one_end_matches(graph, &site.end, &slice) {
        SliceIndexBounds::MinusOne { end: &site.end }
    } else {
        SliceIndexBounds::RangeTo { end: &site.end }
    };
    rewire_one_slice_index_site(graph, range, bounds)
}

fn rewire_one_slice_index_site(
    graph: &mut FunctionGraph,
    range_result: &Variable,
    bounds: SliceIndexBounds<'_>,
) -> Result<(), String> {
    let name = graph.name.clone();
    let range = range_result.clone();

    // 1. Locate the residual `slice::index` call consuming the `RangeFrom`
    //    value as its `range` operand (args[1]). Capture its `slice` operand
    //    (args[0]) and result; the position is re-found after the sweeps
    //    below, since removing the ctor / FieldWrite shifts op indices.
    let (slice, index_result, index_result_ty) = graph
        .blocks
        .iter()
        .find_map(|b| {
            b.operations.iter().find_map(|op| {
                if !is_slice_range_index_call(&op.kind) {
                    return None;
                }
                let OpKind::Call { args, .. } = &op.kind else {
                    return None;
                };
                if args.len() != 2 || args[1] != range {
                    return None;
                }
                let result = op.result.as_ref()?;
                let OpKind::Call { result_ty, .. } = &op.kind else {
                    unreachable!()
                };
                Some((
                    args[0].clone().into_variable(),
                    result.clone(),
                    result_ty.clone(),
                ))
            })
        })
        .ok_or_else(|| format!("{name}: no slice::index call consumes the RangeFrom value"))?;

    // 2. Consumer gate: the `RangeFrom` value must feed EXACTLY its own
    //    construction (`FieldWrite`) and this one index call. A range read by
    //    a `.start` field read, a second index, or a stored range keeps the
    //    ordinary ADT ctor path — rewriting it would break that consumer.
    if !range_feeds_only_index(
        graph,
        &range,
        &index_result,
        matches!(
            bounds,
            SliceIndexBounds::Range { .. }
                | SliceIndexBounds::MinusOne { .. }
                | SliceIndexBounds::RangeTo { .. }
        ),
    ) {
        return Err(format!(
            "{name}: RangeFrom value has a non-index consumer — declining"
        ));
    }

    // 3. Validate the selected bounds before mutating. `decompose_slice_args`
    // (rtyper.rs) requires a non-negative start.  A slice-index consumer
    // statically selects `SliceIndex<usize>`, so its bounds retain the
    // unsigned/nonnegative annotation even when computed; the marker expands
    // only in the rtyper flowspace adapter. Nothing more is asked, because upstream
    // asks nothing more: `ll_listslice_startstop`
    // (`rpython/rtyper/rlist.py`) clamps an oversized stop and states
    // `start <= length` / `stop >= start` as `ll_assert`s.  The one extra
    // check below is `MinusOne`, whose end must really be
    // `sub(ArrayLen(slice), 1)` — that is the shape recognition itself, not a
    // bounds proof.
    match bounds {
        SliceIndexBounds::RangeFrom { start } => {
            if !graph_defines(graph, start) {
                return Err(format!(
                    "{name}: RangeFrom start is not defined in the graph"
                ));
            }
        }
        SliceIndexBounds::Range { start, end } => {
            if !graph_defines(graph, start) || !graph_defines(graph, end) {
                return Err(format!("{name}: Range bounds are not defined in the graph"));
            }
        }
        SliceIndexBounds::MinusOne { end } => {
            if !graph_defines(graph, end) {
                return Err(format!("{name}: MinusOne end is not defined in the graph"));
            }
            if !minus_one_end_matches(graph, end, &slice) {
                return Err(format!(
                    "{name}: RangeTo end is not sub(ArrayLen(slice), 1) — declining"
                ));
            }
        }
        SliceIndexBounds::RangeTo { end } => {
            if !graph_defines(graph, end) {
                return Err(format!("{name}: RangeTo end is not defined in the graph"));
            }
        }
    }

    // All structural validation passed; mutate the graph.

    // 4. Remove the range construction for RangeFrom and Range rewrites.
    // RangeTo (including MinusOne) leaves the ctor and its FieldWrite in
    //    place: the range can be carried by a block link without being read
    //    by an operation, and sweeping its sole defining ctor would orphan
    //    that link-carried value.
    if matches!(
        bounds,
        SliceIndexBounds::RangeFrom { .. } | SliceIndexBounds::Range { .. }
    ) {
        for b in &mut graph.blocks {
            b.operations.retain(|op| {
                let is_field_write =
                    matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == &range);
                let is_ctor = op.result.as_ref() == Some(&range)
                    && matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::SyntheticTransparentCtor { .. },
                            ..
                        }
                    );
                !(is_field_write || is_ctor)
            });
        }
    }

    // 5. Replace the residual `slice::index` call (re-located by result
    //    identity — the sweeps above shifted op indices).
    let (rb, ri) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| {
                    op.result.as_ref() == Some(&index_result) && is_slice_range_index_call(&op.kind)
                })
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: slice::index op vanished before rewrite"))?;
    match bounds {
        SliceIndexBounds::Range { start, end } => {
            let segments = getslice_marker_segments(graph, &slice, "__getslice_range");
            graph.blocks[rb].operations[ri] = SpaceOperation {
                result: Some(index_result),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments,
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![slice, start.clone(), end.clone()]),
                    result_ty: index_result_ty,
                },
            };
        }
        SliceIndexBounds::MinusOne { .. } | SliceIndexBounds::RangeTo { .. } => {
            let (segments, args) = match bounds {
                SliceIndexBounds::MinusOne { .. } => (
                    getslice_marker_segments(graph, &slice, "__getslice_minusone"),
                    vec![slice],
                ),
                SliceIndexBounds::RangeTo { end } => (
                    getslice_marker_segments(graph, &slice, "__getslice_rangeto"),
                    vec![slice, end.clone()],
                ),
                SliceIndexBounds::RangeFrom { .. } => unreachable!(),
                SliceIndexBounds::Range { .. } => unreachable!(),
            };
            graph.blocks[rb].operations[ri] = SpaceOperation {
                result: Some(index_result),
                kind: OpKind::Call {
                    target: CallTarget::function_path(segments),
                    args: crate::model::call_args(args),
                    result_ty: index_result_ty,
                },
            };
        }
        SliceIndexBounds::RangeFrom { start } => {
            let segments = getslice_marker_segments(graph, &slice, "__getslice_rangefrom");
            graph.blocks[rb].operations[ri] = SpaceOperation {
                result: Some(index_result),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments,
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![slice, start.clone()]),
                    result_ty: index_result_ty,
                },
            };
        }
    }
    Ok(())
}

fn reachable_from_start(graph: &FunctionGraph) -> std::collections::HashSet<crate::model::BlockId> {
    let mut reachable = std::collections::HashSet::new();
    let mut todo = vec![graph.startblock];
    while let Some(block) = todo.pop() {
        if reachable.insert(block) {
            todo.extend(graph.successors(block));
        }
    }
    reachable
}

/// Resolve a variable through the single-input block edges that can represent
/// an ordinary MIR copy. Multiple incoming values must converge to one
/// identity; disagreement is deliberately not guessed through.
fn resolve_block_alias(graph: &FunctionGraph, var: &Variable) -> Option<Variable> {
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    fn visit(
        graph: &FunctionGraph,
        var: &Variable,
        seen: &mut std::collections::HashSet<Variable>,
        reachable: &std::collections::HashSet<crate::model::BlockId>,
    ) -> Result<Option<Variable>, ()> {
        if !seen.insert(var.clone()) {
            // A loop-carried pass-through contributes no new definition.  A
            // non-cyclic incoming edge must still establish the unique root.
            return Ok(None);
        }
        // MIR is not strict SSA: an operation result can also be threaded
        // through a later block as an inputarg under the same Variable name.
        // The operation is the authoritative definition; do not follow the
        // downstream links and mistake that pass-through for a phi.
        if graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| op.result.as_ref() == Some(var))
        {
            return Ok(Some(var.clone()));
        }
        let Some((block_id, arg_index)) = graph.blocks.iter().find_map(|b| {
            b.inputargs
                .iter()
                .position(|arg| arg == var)
                .map(|i| (b.id, i))
        }) else {
            return Ok(Some(var.clone()));
        };
        let incoming: Option<Vec<Variable>> = graph
            .blocks
            .iter()
            .filter(|block| reachable.contains(&block.id))
            .flat_map(|b| &b.exits)
            .filter(|link| link.target == block_id)
            .map(|link| {
                link.args
                    .get(arg_index)
                    .and_then(LinkArg::as_variable)
                    .cloned()
            })
            .collect();
        let incoming: Vec<Variable> = incoming
            .ok_or(())?
            .into_iter()
            .filter(|candidate| candidate != var)
            .collect();
        if incoming.is_empty() {
            // Function entry arguments are authoritative external roots.
            return (block_id == graph.startblock)
                .then(|| var.clone())
                .map(Some)
                .ok_or(());
        }
        let mut root: Option<Variable> = None;
        for candidate in incoming {
            // flowspace.model.FunctionGraph.iterblocks visits each identity
            // once across the whole graph. Do the same for this backward walk:
            // every discovered definition propagates to the query's root, so
            // a repeated node contributes no new definition. A path-local set
            // enumerates every path through successive diamonds exponentially.
            let Some(candidate_root) = visit(graph, &candidate, seen, reachable)? else {
                continue;
            };
            if root
                .as_ref()
                .is_some_and(|established| established != &candidate_root)
            {
                return Err(());
            }
            root = Some(candidate_root);
        }
        Ok(root)
    }
    visit(
        graph,
        var,
        &mut std::collections::HashSet::new(),
        &reachable_from_start(graph),
    )
    .ok()
    .flatten()
}

/// The value under a chain of `__cast_instance_intrinsic` identity casts.
fn peel_instance_casts(graph: &FunctionGraph, var: &Variable) -> Variable {
    let mut current = var.clone();
    for _ in 0..8 {
        let Some(operand) = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| {
                (op.result.as_ref() == Some(&current))
                    .then(|| instance_cast_operand(&op.kind))
                    .flatten()
            })
        else {
            break;
        };
        current = resolve_block_alias(graph, &operand).unwrap_or(operand);
    }
    current
}

fn instance_cast_operand(kind: &OpKind) -> Option<Variable> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = kind
    else {
        return None;
    };
    if segments.as_slice() != [crate::runtime_names::shims::CAST_INSTANCE] {
        return None;
    }
    args.first().and_then(LinkArg::as_variable).cloned()
}

fn const_int_value(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    let var = resolve_block_alias(graph, var)?;
    graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| {
            (op.result.as_ref() == Some(&var)).then_some(match op.kind {
                OpKind::ConstInt(value) => Some(value),
                OpKind::ConstUInt(value) => i64::try_from(value).ok(),
                _ => None,
            })?
        })
}

fn minus_one_end_matches(graph: &FunctionGraph, end: &Variable, slice: &Variable) -> bool {
    let Some(end) = resolve_block_alias(graph, end) else {
        return false;
    };
    let Some(slice) = resolve_block_alias(graph, slice) else {
        return false;
    };
    let Some((lhs, rhs)) = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match (&op.result, &op.kind) {
            (
                Some(result),
                OpKind::BinOp {
                    op,
                    lhs,
                    rhs,
                    result_ty: ValueType::Unsigned,
                },
            ) if result == &end && op == "sub" => Some((lhs.clone(), rhs.clone())),
            _ => None,
        })
    else {
        return false;
    };
    let lhs = resolve_block_alias(graph, &lhs).unwrap_or(lhs);
    let rhs = resolve_block_alias(graph, &rhs).unwrap_or(rhs);
    // `arraylen` of an object array is taken on the `[PyObject]` cast of
    // the slice. The cast is the same value (`cast_opaque_ptr`), so
    // `len(cast(s)) - 1` is still the receiver's minus-one end.
    let slice = peel_instance_casts(graph, &slice);
    let has_len = graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        op.result.as_ref() == Some(&lhs)
            && matches!(&op.kind, OpKind::ArrayLen { base, .. }
                if resolve_block_alias(graph, base)
                    .is_some_and(|base| peel_instance_casts(graph, &base) == slice))
    });
    // The subtraction is unsigned, so Charon's faithful literal spelling is
    // `ConstUInt(1)`.  `const_int_value` accepts both integer banks only for
    // this compile-time equality proof; it does not change the SSA value's
    // unsigned annotation used by the guard and slice lowering.
    let has_one = const_int_value(graph, &rhs) == Some(1);
    has_len && has_one
}

/// `true` when `kind` is a residual range-index call over a Rust slice,
/// a `Vec` receiver, or the WTF-8 string view used for Python unicode —
/// the `&s[range]` residual whose `range` operand this pass folds.
/// The scalar `Vec`/`Constants` integer index is intercepted earlier in
/// `lower_call` (`is_vec_index_regular` / `constants_call_leaf`), so only the
/// range-indexed residual reaches here.
fn is_slice_range_index_call(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if slice_index_segments_match(segments)
    )
}

/// The residual callee spelling for `<[T]>::index` with a range index.
/// Charon runs `monomorphize:false`, so the callee keeps the inherent-impl
/// path `core::slice::index::<Impl>::index`. A `Vec` indexed by a range is
/// the same `rtype_getslice` (`rlist.py` `AbstractBaseListRepr.rtype_getslice`)
/// as the slice spelling: Rust `Vec` is the rtyper raw list repr and a slice
/// is the raw `{ptr,len}` view, so the rewrite keeps the list receiver.
fn slice_index_segments_match(segments: &[String]) -> bool {
    // `Wtf8` is the frontend's RPython `SomeString` spelling (mir.rs maps
    // Wtf8/Wtf8Buf/String/str to ValueType::Str). Its Index<Range*> impl is
    // therefore the same `str[start:stop]` operation as the slice spelling,
    // not a foreign method to residualize. PyPy's unicode paths use ordinary
    // slicing and RPython lowers both string and list slices through
    // `rtype_getslice`.
    //
    // `rustpython_wtf8::Wtf8` sits at its crate root, so its impl-owner
    // spelling carries no module segment and the anchored form IS two
    // segments; the crate-qualified spelling is accepted alongside it so the
    // arm is anchored wherever the crate name survives.
    let segs: Vec<&str> = segments
        .iter()
        .map(String::as_str)
        .filter(|seg| !crate::front::mir::is_trait_impl_identity_segment(seg))
        .collect();
    if segs == ["Wtf8", "index"] || segs == ["rustpython_wtf8", "Wtf8", "index"] {
        return true;
    }
    // Scalar `Vec::index` is intercepted in `lower_call` (`is_vec_index_call`)
    // before a residual is planted, so this arm only sees `Range*`.
    // Residual spelling after `impl_method_owner_for_fundecl`:
    // `["vec","Vec","index"]`. FunDecl `name_path`: `alloc::vec::<Impl>::index`.
    // A trait-impl identity segment is stripped so
    // `["vec","Vec","<impl Index<…>>","index"]` is the same owner/leaf
    // (`rlist.py` `AbstractBaseListRepr.rtype_getslice`).
    if vec_inherent_index_from_leafs(&segs) {
        return true;
    }
    let n = segs.len();
    if n < 4 || segs.first() != Some(&"core") {
        return false;
    }
    let slice_impl =
        n >= 5 && segs[n - 4] == "slice" && segs[n - 3] == "index" && segs[n - 2] == "<Impl>";
    let array_impl = segs[n - 3] == "array" && segs[n - 2] == "<Impl>";
    segs[n - 1] == "index" && (slice_impl || array_impl)
}

/// Inherent `Vec::index` (not `index_mut`). Accepts the impl-owner residual
/// `["vec","Vec","index"]` and crate-qualified / FunDecl forms that end in
/// `"vec","Vec","index"` or `"vec","<Impl>","index"`. A trait-impl identity
/// segment is ignored, matching [`crate::front::mir::trait_impl_identity_path`].
pub(crate) fn vec_inherent_index_segments(segments: &[String]) -> bool {
    let segs: Vec<&str> = segments
        .iter()
        .map(String::as_str)
        .filter(|seg| !crate::front::mir::is_trait_impl_identity_segment(seg))
        .collect();
    vec_inherent_index_from_leafs(&segs)
}

fn vec_inherent_index_from_leafs(segs: &[&str]) -> bool {
    let n = segs.len();
    n >= 3
        && segs[n - 1] == "index"
        && segs[n - 3] == "vec"
        && (segs[n - 2] == "Vec" || segs[n - 2] == "<Impl>")
}

/// `true` when the `RangeFrom` value is consumed by EXACTLY its own
/// construction (`FieldWrite`) and the one `slice::index` call whose result is
/// `index_result`. Computes the forward alias closure of the range value
/// (every Variable it flows into through positional `Link.args`) and rejects
/// if any op / exitswitch / exception payload other than the construction
/// writes or the index op reads a closure member. Mirrors
/// `range_iter::range_feeds_only_forloop`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub(crate) fn range_feeds_only_index(
    graph: &FunctionGraph,
    range_result: &Variable,
    index_result: &Variable,
    require_end_write: bool,
) -> bool {
    use std::collections::HashSet;
    let mut closure: HashSet<Variable> = HashSet::new();
    closure.insert(range_result.clone());
    let mut end_writes = 0;
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

    let in_closure = |v: &Variable| closure.contains(v);
    for block in &graph.blocks {
        for op in &block.operations {
            let reads_range = crate::inline::op_variable_refs(&op.kind)
                .iter()
                .any(&in_closure);
            if !reads_range {
                continue;
            }
            // A construction `FieldWrite` on the `RangeFrom` value writes its
            // `start`; the `slice::index` op reads the range as its `range`
            // operand. Both are removed / rewritten; anything else is a genuine
            // second consumer. The base must be the range value ITSELF, not any
            // aliased inputarg in the closure: `rewire_one_slice_index_site`'s
            // removal sweep drops only a `FieldWrite` with `base == range`, so
            // accepting an aliased-base write here would leave that write
            // behind after its ctor is removed — an undefined-operand shape
            // `flowspace_adapter` rejects. `front::mir` emits the ctor and the
            // `start` write in one block (`base == range`), so this loses no
            // real site; a range threaded across a block edge declines cleanly.
            let is_construction_write =
                matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == range_result);
            if matches!(
                &op.kind,
                OpKind::FieldWrite { base, field, .. }
                    if base == range_result && field.name == "end"
            ) {
                end_writes += 1;
            }
            let is_index = op.result.as_ref() == Some(index_result);
            if !(is_construction_write || is_index) {
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
    !require_end_write || end_writes == 1
}

/// Marker path. The leaf stays `__getslice_*`. A second segment carries the
/// item kind when the slice operand's list spelling or array identity names
/// one and the sibling-array scan does not. A scan hit keeps today's
/// one-segment marker: the codewriter still recovers that identity itself.
fn getslice_marker_segments(graph: &FunctionGraph, slice: &Variable, leaf: &str) -> Vec<String> {
    let mut segments = vec![leaf.to_string()];
    if let Some(suffix) = slice_marker_suffix(graph, slice) {
        segments.push(suffix);
    }
    segments
}

fn slice_marker_suffix(graph: &FunctionGraph, slice: &Variable) -> Option<String> {
    if crate::codewriter::getslice::array_identity_of_base(graph, slice).is_some() {
        return None;
    }
    let classes = crate::codewriter::getslice::LinkClasses::of(graph);
    let same = |v: &Variable| classes.same(v, slice);
    let mut from_input = None;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::ArrayLen {
                base,
                array_type_id: Some(id),
                ..
            } if same(base) => {
                if let Some((item_ty, array_type_id)) = identity_from_array_type_id(id) {
                    return crate::codewriter::getslice::listslice_marker_suffix(
                        &item_ty,
                        array_type_id.as_deref(),
                    );
                }
            }
            OpKind::Input {
                class_root: Some(root),
                ..
            } if from_input.is_none() && op.result.as_ref().is_some_and(&same) => {
                from_input = identity_from_list_spelling(root);
            }
            _ => {}
        }
    }
    let (item_ty, array_type_id) = from_input?;
    crate::codewriter::getslice::listslice_marker_suffix(&item_ty, array_type_id.as_deref())
}

/// `[i64]` / `[f64]` and the same spellings inside `Vec<T>` / `VecDeque<T>` /
/// `[T; N]`. Anything else stays unsuffixed.
fn identity_from_list_spelling(spelling: &str) -> Option<(ValueType, Option<String>)> {
    let stripped = spelling
        .trim()
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim();
    let elem = if let Some(rest) = stripped.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        match rest.find(';') {
            Some(i) => rest[..i].trim(),
            None => rest.trim(),
        }
    } else {
        let rest = stripped
            .strip_prefix("Vec<")
            .or_else(|| stripped.strip_prefix("VecDeque<"))?;
        rest.strip_suffix('>')?.trim()
    };
    if elem.is_empty() {
        return None;
    }
    identity_from_array_type_id(&format!("[{elem}]"))
}

fn identity_from_array_type_id(id: &str) -> Option<(ValueType, Option<String>)> {
    let inner = id.strip_prefix('[').and_then(|s| s.strip_suffix(']'))?;
    if inner.contains(';') || inner.contains('<') {
        return None;
    }
    let item_ty = match inner {
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
            ValueType::Int
        }
        "f64" => ValueType::Float,
        _ => return None,
    };
    Some((item_ty, Some(id.to_string())))
}

/// `true` when `var` is produced by some op or is a block inputarg — i.e. it
/// is a well-defined value the rewritten `getslice` op can reference.
fn graph_defines(graph: &FunctionGraph, var: &Variable) -> bool {
    graph.blocks.iter().any(|b| {
        b.inputargs.iter().any(|iv| iv == var)
            || b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(var))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ValueType;

    fn rangefrom_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "RangeFrom".to_string(),
        )
    }

    fn rangeto_ctor_target() -> CallTarget {
        CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["core".to_string(), "ops".to_string(), "range".to_string()],
            "RangeTo".to_string(),
        )
    }

    fn start_field_write(base: &Variable, value: &Variable) -> OpKind {
        OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor {
                name: "start".to_string(),
                owner_root: Some("core::ops::range::RangeFrom".to_string()),
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

    fn end_field_write(base: &Variable, value: &Variable) -> OpKind {
        named_end_field_write(base, value, "end")
    }

    fn named_end_field_write(base: &Variable, value: &Variable, name: &str) -> OpKind {
        OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor {
                name: name.to_string(),
                owner_root: Some("core::ops::range::RangeTo".to_string()),
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

    fn slice_index_call_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: vec![
                "core".into(),
                "slice".into(),
                "index".into(),
                "<Impl>".into(),
                "index".into(),
            ],
            fun_decl_id: None,
        }
    }

    fn build_rangeto_static_length_graph(
        bound: Option<i64>,
        site_on_true_edge: bool,
        static_repeat_count: bool,
    ) -> (FunctionGraph, SliceIndexRangeToSite) {
        let mut g = FunctionGraph::new("rangeto_static_length");
        let entry = g.startblock;
        let fill = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let repeat_args = if static_repeat_count {
            let count = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
            vec![fill.clone(), count]
        } else {
            vec![fill.clone()]
        };
        let receiver = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(repeat_args),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let end = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let (site_block, other_block, true_block) = if bound.is_some() {
            let (true_block, _) = g.create_block_with_arg_vars(0);
            let (false_block, _) = g.create_block_with_arg_vars(0);
            (
                if site_on_true_edge {
                    true_block
                } else {
                    false_block
                },
                if site_on_true_edge {
                    false_block
                } else {
                    true_block
                },
                true_block,
            )
        } else {
            (entry, entry, entry)
        };
        if let Some(bound) = bound {
            let bound_var = g.push_op_var(entry, OpKind::ConstInt(bound), true).unwrap();
            let cond = g
                .push_op_var(
                    entry,
                    OpKind::BinOp {
                        op: "le".into(),
                        lhs: end.clone(),
                        rhs: bound_var,
                        result_ty: ValueType::Bool,
                    },
                    true,
                )
                .unwrap();
            g.set_branch(entry, cond, true_block, vec![], other_block, vec![]);
        }
        let range = g
            .push_op_var(
                site_block,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(site_block).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let result = g
            .push_op_var(
                site_block,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![receiver, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        if bound.is_some() {
            g.set_return(site_block, Some(result));
            g.set_return(other_block, None);
        } else {
            g.set_return(site_block, Some(result));
        }
        (
            g,
            SliceIndexRangeToSite {
                range_result: range,
                end,
            },
        )
    }

    fn build_rangeto_minusone_graph(field_name: &str) -> (FunctionGraph, SliceIndexRangeToSite) {
        let mut g = FunctionGraph::new("rangeto_minusone");
        let entry = g.startblock;
        let slice = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let len = g
            .push_op_var(
                entry,
                OpKind::ArrayLen {
                    base: slice.clone(),
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let one = g.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let end = g
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len,
                    rhs: one,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(entry).operations.push(SpaceOperation {
            result: None,
            kind: named_end_field_write(&range, &end, field_name),
        });
        let result = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(entry, Some(result));
        (
            g,
            SliceIndexRangeToSite {
                range_result: range,
                end,
            },
        )
    }

    fn build_rangeto_arraylen_value_graph(
        different_base: bool,
        mutable_base: bool,
    ) -> (FunctionGraph, SliceIndexRangeToSite) {
        let mut g = FunctionGraph::new("rangeto_arraylen_value");
        let entry = g.startblock;
        let fill = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let count = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let receiver_a = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![fill.clone(), count.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let receiver_b = if different_base {
            g.push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![fill.clone(), count.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap()
        } else {
            receiver_a.clone()
        };
        let comparison_len = g
            .push_op_var(
                entry,
                OpKind::ArrayLen {
                    base: receiver_b,
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let bound = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let cond = g
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "le".into(),
                    lhs: comparison_len,
                    rhs: bound,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        let (site_block, site_args) = g.create_block_with_arg_vars(1);
        let (other_block, _) = g.create_block_with_arg_vars(0);
        g.set_branch(
            entry,
            cond,
            site_block,
            vec![receiver_a],
            other_block,
            vec![],
        );
        if mutable_base {
            g.block_mut(site_block).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["vec".into(), "Vec".into(), "push".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![site_args[0].clone(), fill]),
                    result_ty: ValueType::Void,
                },
            });
        }
        let site_slice = site_args[0].clone();
        let end = g
            .push_op_var(
                site_block,
                OpKind::ArrayLen {
                    base: site_slice,
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                site_block,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(site_block).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let result = g
            .push_op_var(
                site_block,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![site_args[0].clone(), range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(site_block, Some(result));
        g.set_return(other_block, None);
        (
            g,
            SliceIndexRangeToSite {
                range_result: range,
                end,
            },
        )
    }

    fn has_residual_slice_index(graph: &FunctionGraph) -> bool {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| is_slice_range_index_call(&op.kind))
    }

    /// Every general RangeTo keeps the original stop, receiver and result.
    /// In particular, a branch/phi or mutation cannot substitute a different
    /// ArrayLen for the captured stop. The marker also stays safe for legacy.
    fn assert_rangeto_rewrite(g: &mut FunctionGraph, site: SliceIndexRangeToSite) {
        let original = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find(|op| is_slice_range_index_call(&op.kind))
            .unwrap()
            .clone();
        let OpKind::Call {
            args, result_ty, ..
        } = &original.kind
        else {
            unreachable!()
        };
        let expected_args = vec![args[0].clone(), site.end.clone().into()];
        assert_eq!(rewire_slice_index_rangeto_sites(g, &[site]), 1);
        assert!(!has_residual_slice_index(g));
        let rewritten = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find(|op| op.result == original.result)
            .unwrap();
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            result_ty: new_ty,
        } = &rewritten.kind
        else {
            panic!("RangeTo must remain a synthetic call until flowspace adaptation");
        };
        assert_eq!(segments, &["__getslice_rangeto"]);
        assert_eq!(args, &expected_args);
        assert_eq!(new_ty, result_ty);
        assert!(
            !g.blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|op| matches!(&op.kind, OpKind::GetSlice { .. }))
        );
    }

    fn length_changing_call(base: &Variable) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["vec".into(), "Vec".into(), "push".into()],
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![base.clone()]),
            result_ty: ValueType::Void,
        }
    }

    fn arraylen_site_base(
        graph: &FunctionGraph,
        end: &Variable,
    ) -> (crate::model::BlockId, Variable) {
        graph
            .blocks
            .iter()
            .find_map(|block| {
                block.operations.iter().find_map(|op| {
                    matches!(
                        (&op.result, &op.kind),
                        (Some(result), OpKind::ArrayLen { .. }) if result == end
                    )
                    .then(|| {
                        let OpKind::ArrayLen { base, .. } = &op.kind else {
                            unreachable!()
                        };
                        (block.id, base.clone())
                    })
                })
            })
            .expect("ArrayLen site")
    }

    #[test]
    fn rangeto_preserves_end_across_unresolved_accessor_path() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let site_block = g
            .blocks
            .iter()
            .find_map(|block| block.inputargs.contains(&site.end).then_some(block.id))
            .unwrap_or_else(|| {
                g.blocks
                    .iter()
                    .find_map(|block| {
                        block
                            .operations
                            .iter()
                            .any(|op| {
                                matches!(&op.kind, OpKind::FieldWrite { base, .. } if base == &site.range_result)
                            })
                            .then_some(block.id)
                    })
                    .unwrap()
            });
        let receiver = g
            .blocks
            .iter()
            .find(|block| block.id == site_block)
            .unwrap()
            .inputargs[0]
            .clone();
        g.block_mut(site_block).operations.insert(
            0,
            SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["util".into(), "get".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![receiver.clone(), receiver]),
                    result_ty: ValueType::Unsigned,
                },
            },
        );

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_second_end_write_declines_rewrite() {
        let (mut g, site) = build_rangeto_static_length_graph(Some(8), true, true);
        let site_block = g
            .blocks
            .iter()
            .find_map(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| {
                        matches!(&op.kind, OpKind::FieldWrite { base, field, .. }
                            if base == &site.range_result && field.name == "end")
                    })
                    .then_some(block.id)
            })
            .unwrap();
        let replacement = g
            .push_op_var(site_block, OpKind::ConstInt(5), true)
            .unwrap();
        g.block_mut(site_block).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&site.range_result, &replacement),
        });

        assert_eq!(
            rewire_slice_index_rangeto_sites(&mut g, &[site]),
            0,
            "a post-construction end write must prevent substitution of the captured end"
        );
        assert!(has_residual_slice_index(&g));
    }

    /// Positional `RangeTo.end` spelling deliberately stays rejected.
    /// `is_construction_write` has never keyed on the field name, so the
    /// `end_writes` gate is not what admits a site — it only requires the
    /// write to be unique and named. Widening it to `__pos_0` therefore does
    /// not restore an older rewrite; it admits sites that only
    /// `resolve_block_alias` makes visible, and that widening was measured to
    /// clear no additional prepass subject.
    #[test]
    fn rangeto_minusone_positional_end_write_declines() {
        let (mut g, site) = build_rangeto_minusone_graph("__pos_0");

        assert_eq!(
            rewire_slice_index_rangeto_sites(&mut g, &[site]),
            0,
            "a positional MinusOne end write must decline"
        );
        assert!(has_residual_slice_index(&g));
    }

    fn build_rangeto_self_phi_graph(
        distinct_incoming: bool,
    ) -> (FunctionGraph, SliceIndexRangeToSite) {
        let mut g = FunctionGraph::new("rangeto_self_phi");
        let entry = g.startblock;
        let fill = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let count = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let receiver_a = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![fill.clone(), count.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let receiver_b = if distinct_incoming {
            g.push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![fill, count]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap()
        } else {
            receiver_a.clone()
        };
        let end = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let bound = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let cond = g
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "le".into(),
                    lhs: end.clone(),
                    rhs: bound,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        let (site, site_args) = g.create_block_with_arg_vars(1);
        let (other, _) = g.create_block_with_arg_vars(0);
        if distinct_incoming {
            let (fanout, _) = g.create_block_with_arg_vars(0);
            g.set_branch(entry, cond, fanout, vec![], other, vec![]);
            let fanout_cond = g
                .push_op_var(fanout, OpKind::ConstBool(true), true)
                .unwrap();
            g.set_branch(
                fanout,
                fanout_cond,
                site,
                vec![receiver_a],
                site,
                vec![receiver_b],
            );
        } else {
            g.set_branch(entry, cond, site, vec![receiver_a], other, vec![]);
        }
        let range = g
            .push_op_var(
                site,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(site).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let result = g
            .push_op_var(
                site,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![site_args[0].clone(), range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        if distinct_incoming {
            g.set_return(site, Some(result));
        } else {
            g.set_goto(site, site, vec![site_args[0].clone()]);
        }
        g.set_return(other, None);
        (
            g,
            SliceIndexRangeToSite {
                range_result: range,
                end,
            },
        )
    }

    #[test]
    fn rangeto_self_phi_rewrites() {
        let (mut g, site) = build_rangeto_self_phi_graph(false);
        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_distinct_phi_rewrites() {
        let (mut g, site) = build_rangeto_self_phi_graph(true);
        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_mixed_variable_constant_phi_rewrites() {
        let (mut g, site) = build_rangeto_self_phi_graph(false);
        let (site_block, phi) = g
            .blocks
            .iter()
            .find_map(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| is_slice_range_index_call(&op.kind))
                    .then(|| (block.id, block.inputargs[0].clone()))
            })
            .unwrap();
        let self_link = g
            .block_mut(site_block)
            .exits
            .iter_mut()
            .find(|link| link.target == site_block)
            .unwrap();
        self_link.args[0] = LinkArg::Const(crate::flowspace::model::Constant::new(
            crate::flowspace::model::ConstValue::Int(7),
        ));

        assert_eq!(
            resolve_block_alias(&g, &phi),
            None,
            "a constant incoming phi argument must make alias resolution fail closed"
        );
        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn block_alias_ignores_constant_from_unreachable_predecessor() {
        let (mut g, _) = build_rangeto_self_phi_graph(false);
        let (site_block, phi) = g
            .blocks
            .iter()
            .find_map(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| is_slice_range_index_call(&op.kind))
                    .then(|| (block.id, block.inputargs[0].clone()))
            })
            .unwrap();
        let root = resolve_block_alias(&g, &phi).unwrap();
        let unreachable = g.create_block();
        g.set_goto(unreachable, site_block, vec![root.clone()]);
        g.block_mut(unreachable).exits[0].args[0] = LinkArg::Const(
            crate::flowspace::model::Constant::new(crate::flowspace::model::ConstValue::Int(7)),
        );

        assert_eq!(
            resolve_block_alias(&g, &phi),
            Some(root),
            "a constant from an unreachable predecessor must not veto alias resolution"
        );
    }

    #[test]
    fn block_alias_converging_diamonds_share_the_visited_set() {
        // A small graph with billions of paths. Alias resolution must inspect
        // its definitions, not enumerate those paths; it must still reject a
        // different root entering halfway through the converging branches.
        for conflicting in [false, true] {
            let mut g = FunctionGraph::new("alias_diamonds");
            let entry = g.startblock;
            let root = g.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
            let other = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
            let cond = g.push_op_var(entry, OpKind::ConstBool(true), true).unwrap();
            let mut block = entry;
            let mut value = root.clone();
            for layer in 0..32 {
                let (left, left_args) = g.create_block_with_arg_vars(1);
                let (right, right_args) = g.create_block_with_arg_vars(1);
                let (join, join_args) = g.create_block_with_arg_vars(1);
                let right_value = if conflicting && layer == 16 {
                    other.clone()
                } else {
                    value.clone()
                };
                g.set_branch(
                    block,
                    cond.clone(),
                    left,
                    vec![value],
                    right,
                    vec![right_value],
                );
                g.set_goto(left, join, left_args);
                g.set_goto(right, join, right_args);
                block = join;
                value = join_args[0].clone();
            }
            assert_eq!(
                resolve_block_alias(&g, &value),
                (!conflicting).then_some(root)
            );
        }
    }

    #[test]
    fn rangeto_operation_result_precedes_downstream_inputarg() {
        let mut g = FunctionGraph::new("rangeto_result_inputarg_precedence");
        let entry = g.startblock;
        let fill = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let count = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let receiver = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![fill, count]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let end = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let bound = g.push_op_var(entry, OpKind::ConstInt(8), true).unwrap();
        let cond = g
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "le".into(),
                    lhs: end.clone(),
                    rhs: bound,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        let (site, _site_args) = g.create_block_with_arg_vars(1);
        // Deliberately reuse the operation result as the downstream inputarg.
        // The incoming edge is self-only and therefore unresolvable if the
        // inputarg walk is incorrectly given precedence.
        g.block_mut(site).inputargs[0] = receiver.clone();
        let other = g.create_block();
        g.set_branch(entry, cond, site, vec![receiver.clone()], other, vec![]);
        let range = g
            .push_op_var(
                site,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(site).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let result = g
            .push_op_var(
                site,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![receiver, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(site, Some(result));
        g.set_return(other, None);
        let site = SliceIndexRangeToSite {
            range_result: range,
            end,
        };

        assert_eq!(rewire_slice_index_rangeto_sites(&mut g, &[site]), 1);
        assert!(!has_residual_slice_index(&g));
        assert_eq!(
            g.blocks
                .iter()
                .flat_map(|b| &b.operations)
                .filter(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["__getslice_rangeto".to_string()]
                ))
                .count(),
            1
        );
    }

    #[test]
    fn rangeto_dominating_le_rewrites() {
        let (mut g, site) = build_rangeto_static_length_graph(Some(8), true, true);
        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_arraylen_value_graphs_rewrite() {
        let cases = [
            ("equivalent_values", false, false),
            ("different_bases", true, false),
            ("mutable_base", false, true),
        ];
        for (name, different_bases, mutable_base) in cases {
            let (mut g, site) = build_rangeto_arraylen_value_graph(different_bases, mutable_base);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_rangeto_rewrite(&mut g, site);
            }));
            if let Err(payload) = result {
                panic!("case {name}: {}", panic_payload_slice(&payload));
            }
        }
    }

    fn panic_payload_slice(payload: &Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else {
            "assertion failed".to_string()
        }
    }

    #[test]
    fn rangeto_preserves_end_across_prebranch_mutation_after_lhs_read() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let entry = g.startblock;
        let (lhs_index, base) = g
            .block(entry)
            .operations
            .iter()
            .enumerate()
            .find_map(|(index, op)| match &op.kind {
                OpKind::ArrayLen { base, .. } => Some((index, base.clone())),
                _ => None,
            })
            .expect("comparison ArrayLen");
        g.block_mut(entry).operations.insert(
            lhs_index + 1,
            SpaceOperation {
                result: None,
                kind: length_changing_call(&base),
            },
        );

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_preserves_end_across_shrink_between_end_and_lhs_reads() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let entry = g.startblock;
        let (lhs_index, base) = g
            .block(entry)
            .operations
            .iter()
            .enumerate()
            .find_map(|(index, op)| match &op.kind {
                OpKind::ArrayLen { base, .. } => Some((index, base.clone())),
                _ => None,
            })
            .expect("comparison ArrayLen");
        let (end_block, end_index) = g
            .blocks
            .iter()
            .find_map(|block| {
                block
                    .operations
                    .iter()
                    .position(|op| op.result.as_ref() == Some(&site.end))
                    .map(|index| (block.id, index))
            })
            .expect("end ArrayLen");
        let mut end_op = g.block_mut(end_block).operations.remove(end_index);
        let OpKind::ArrayLen { base: end_base, .. } = &mut end_op.kind else {
            panic!("end definition is ArrayLen")
        };
        *end_base = base.clone();
        g.block_mut(entry).operations.insert(lhs_index, end_op);
        g.block_mut(entry).operations.insert(
            lhs_index + 1,
            SpaceOperation {
                result: None,
                kind: length_changing_call(&base),
            },
        );

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_preserves_end_across_opaque_call_without_base_operand() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let entry = g.startblock;
        let owner = g.push_op_var(entry, OpKind::ConstInt(17), true).unwrap();
        let (site_block, _) = arraylen_site_base(&g, &site.end);
        g.block_mut(site_block).operations.insert(
            0,
            SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["some".into(), "opaque".into(), "mutator".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![owner]),
                    result_ty: ValueType::Void,
                },
            },
        );

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_preserves_end_across_arraywrite_publish_reload_mutation() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let entry = g.startblock;
        let (lhs_index, base) = g
            .block(entry)
            .operations
            .iter()
            .enumerate()
            .find_map(|(index, op)| match &op.kind {
                OpKind::ArrayLen { base, .. } => Some((index, base.clone())),
                _ => None,
            })
            .expect("comparison ArrayLen");
        let slot = g
            .block(entry)
            .operations
            .iter()
            .find_map(|op| {
                matches!(op.kind, OpKind::ConstInt(0))
                    .then(|| op.result.clone())
                    .flatten()
            })
            .expect("slot and index variable");
        let reloaded = g
            .push_op_var(
                entry,
                OpKind::ArrayRead {
                    base: slot.clone(),
                    index: slot.clone(),
                    item_ty: ValueType::Ref(None),
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let reload_op = g.block_mut(entry).operations.pop().unwrap();
        let prefix = [
            SpaceOperation {
                result: None,
                kind: OpKind::ArrayWrite {
                    base: slot.clone(),
                    index: slot,
                    value: LinkArg::Value(base),
                    item_ty: ValueType::Ref(None),
                    array_type_id: None,
                    nolength: false,
                },
            },
            reload_op,
            SpaceOperation {
                result: None,
                kind: length_changing_call(&reloaded),
            },
        ];
        g.block_mut(entry)
            .operations
            .splice(lhs_index..lhs_index, prefix);

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_ignores_mutation_strictly_after_later_read() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let (site_block, base) = arraylen_site_base(&g, &site.end);
        let end_index = g
            .block(site_block)
            .operations
            .iter()
            .position(|op| op.result.as_ref() == Some(&site.end))
            .unwrap();
        g.block_mut(site_block).operations.insert(
            end_index + 1,
            SpaceOperation {
                result: None,
                kind: length_changing_call(&base),
            },
        );

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_ignores_mutation_after_site() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let (site_block, base) = arraylen_site_base(&g, &site.end);
        let (after_block, _) = g.create_block_with_arg_vars(0);
        g.block_mut(after_block).operations.push(SpaceOperation {
            result: None,
            kind: length_changing_call(&base),
        });

        assert_rangeto_rewrite(&mut g, site);
        assert_ne!(site_block, after_block);
    }

    #[test]
    fn rangeto_preserves_end_across_mutation_between_bound_and_site() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let (site_block, _) = arraylen_site_base(&g, &site.end);
        let entry = g.startblock;
        let (between_block, between_args) = g.create_block_with_arg_vars(1);
        let cond = match g.block(entry).exitswitch.clone() {
            Some(crate::model::ExitSwitch::Value(cond)) => cond,
            _ => panic!("bound branch condition"),
        };
        let other_block = g
            .block(entry)
            .exits
            .iter()
            .find(|link| link.exitcase == Some(crate::model::ExitCase::Bool(false)))
            .map(|link| link.target)
            .expect("bound false edge");
        g.set_branch(
            entry,
            cond,
            between_block,
            vec![g.block(site_block).inputargs[0].clone()],
            other_block,
            vec![],
        );
        g.block_mut(between_block).operations.push(SpaceOperation {
            result: None,
            kind: length_changing_call(&between_args[0]),
        });
        g.block_mut(between_block).exits = vec![crate::model::Link::from_variables(
            &g,
            between_args,
            site_block,
            None,
        )];

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_preserves_end_across_mutation_after_site_in_loop() {
        let (mut g, site) = build_rangeto_arraylen_value_graph(false, false);
        let (site_block, base) = arraylen_site_base(&g, &site.end);
        let (loop_block, loop_args) = g.create_block_with_arg_vars(1);
        g.block_mut(loop_block).operations.push(SpaceOperation {
            result: None,
            kind: length_changing_call(&loop_args[0]),
        });
        g.block_mut(site_block).exitswitch = None;
        g.block_mut(site_block).exits = vec![crate::model::Link::from_variables(
            &g,
            vec![base.clone()],
            loop_block,
            None,
        )];
        g.block_mut(loop_block).exitswitch = None;
        g.block_mut(loop_block).exits = vec![crate::model::Link::from_variables(
            &g, loop_args, site_block, None,
        )];

        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_static_length_graphs_rewrite() {
        let cases = [
            ("without_bound", None, true, true),
            ("weak_bound", Some(9), true, true),
            ("wrong_edge", Some(8), false, true),
            ("unknown_repeat_count", Some(8), true, false),
        ];
        for (name, bound, edge, repeat_known) in cases {
            let (mut g, site) = build_rangeto_static_length_graph(bound, edge, repeat_known);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_rangeto_rewrite(&mut g, site);
            }));
            if let Err(payload) = result {
                panic!("case {name}: {}", panic_payload_slice(&payload));
            }
        }
    }

    /// Build the minimal `&s[k..]` shape — a `ConstInt(k)` start, a
    /// `RangeFrom` ctor + `start` FieldWrite, and a residual
    /// `slice::index(slice, range)` — and assert the rewrite drops the ctor +
    /// FieldWrite + residual call and emits `getslice(slice, k, None)`.
    #[test]
    fn rewrite_lifts_rangefrom_index_to_getslice() {
        let mut g = FunctionGraph::new("test_slice_index");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let k = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangefrom_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeFrom".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: start_field_write(&range, &k),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice.clone(), range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub.clone()]);

        let site = SliceIndexRangeFromSite {
            range_result: range.clone(),
            start: k.clone(),
        };
        let rewritten = rewire_slice_index_rangefrom_sites(&mut g, &[site]);
        assert_eq!(rewritten, 1, "the slice::index RangeFrom site is rewritten");

        // The residual `slice::index` call, the RangeFrom ctor, and the
        // `start` FieldWrite are all gone.
        let has_index = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .any(|op| is_slice_range_index_call(&op.kind));
        assert!(!has_index, "residual slice::index call removed");
        let has_ctor = g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )
        });
        assert!(!has_ctor, "RangeFrom ctor removed");
        let has_start_write = g.blocks.iter().flat_map(|blk| &blk.operations).any(
            |op| matches!(&op.kind, OpKind::FieldWrite { field, .. } if field.name == "start"),
        );
        assert!(!has_start_write, "start FieldWrite removed");

        // A single marker with (slice, k) is emitted. The adapter adds the
        // constant None only after annotation, so an unrelated phaseA failure
        // cannot leak an unwired getslice into the legacy walker.
        let marker_ops: Vec<_> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["__getslice_rangefrom".to_string()]
                )
            })
            .collect();
        assert_eq!(marker_ops.len(), 1, "exactly one RangeFrom marker");
        let OpKind::Call { args, .. } = &marker_ops[0].kind else {
            unreachable!()
        };
        assert_eq!(args.len(), 2, "marker has [slice, start]");
        assert_eq!(args[0], slice, "arg0 = slice receiver");
        assert_eq!(args[1], k, "arg1 = const start k");
    }

    #[test]
    fn minus_one_recognizer_requires_same_slice_array_len() {
        let mut g = FunctionGraph::new("minus_one_recognizer");
        let b = g.startblock;
        let slice = g.push_op_var(b, OpKind::ConstInt(0), true).unwrap();
        let other = g.push_op_var(b, OpKind::ConstInt(0), true).unwrap();
        let len = g
            .push_op_var(
                b,
                OpKind::ArrayLen {
                    base: slice.clone(),
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let one = g.push_op_var(b, OpKind::ConstUInt(1), true).unwrap();
        let end = g
            .push_op_var(
                b,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len,
                    rhs: one,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        assert!(minus_one_end_matches(&g, &end, &slice));
        assert!(!minus_one_end_matches(&g, &end, &other));
    }

    #[test]
    fn minus_one_recognizer_requires_unsigned_subtraction() {
        let mut g = FunctionGraph::new("minus_one_recognizer_signed");
        let b = g.startblock;
        let slice = g.push_op_var(b, OpKind::ConstInt(0), true).unwrap();
        let len = g
            .push_op_var(
                b,
                OpKind::ArrayLen {
                    base: slice.clone(),
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let one = g.push_op_var(b, OpKind::ConstInt(1), true).unwrap();
        let end = g
            .push_op_var(
                b,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len,
                    rhs: one,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        assert!(!minus_one_end_matches(&g, &end, &slice));
    }

    #[test]
    fn rewrite_minusone_keeps_link_carried_range_defined() {
        let mut g = FunctionGraph::new("test_slice_index_minusone_link");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::ArrayLen {
                    base: slice.clone(),
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let one = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let end = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len,
                    rhs: one,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(2);
        g.set_return(b, None);
        // The RangeTo value is carried to the successor but never read by an
        // operation there — precisely the shape that must not orphan it.
        g.set_goto(a, b, vec![sub, range.clone()]);

        let site = SliceIndexRangeToSite {
            range_result: range.clone(),
            end,
        };
        assert_eq!(
            rewire_slice_index_rangeto_sites(&mut g, &[site]),
            1,
            "the MinusOne slice::index site is rewritten"
        );
        assert!(g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments == &["__getslice_minusone".to_string()] && args.len() == 1
            )
        }));
        assert!(
            g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    }
                ) && op.result.as_ref() == Some(&range)
            }),
            "RangeTo ctor survives MinusOne rewrite"
        );
        assert!(
            g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
                matches!(&op.kind, OpKind::FieldWrite { base, field, .. }
                    if base == &range && field.name == "end")
            }),
            "RangeTo end FieldWrite survives MinusOne rewrite"
        );

        for block in &g.blocks {
            for link in &block.exits {
                for arg in &link.args {
                    if let LinkArg::Value(var) = arg {
                        assert!(
                            graph_defines(&g, var),
                            "link arg {var:?} must remain defined after MinusOne rewrite"
                        );
                    }
                }
            }
        }
    }

    /// `len` of an object array is taken on the `[PyObject]` cast. That
    /// length is still the slice's length, so `s[..len-1]` stays MinusOne.
    #[test]
    fn rewrite_minusone_sees_arraylen_through_instance_cast() {
        let mut g = FunctionGraph::new("test_slice_index_minusone_cast");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let cast = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("[PyObject]", slice.clone()),
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::ArrayLen {
                    base: cast,
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        let one = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let end = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: len,
                    rhs: one,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let _sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);
        let site = SliceIndexRangeToSite {
            range_result: range,
            end,
        };
        assert_eq!(rewire_slice_index_rangeto_sites(&mut g, &[site]), 1);
        assert!(g.blocks.iter().flat_map(|blk| &blk.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments == &["__getslice_minusone".to_string()] && args.len() == 1
            )
        }));
    }

    /// `&s[..k]` uses the existing StartStop contract: a synthetic constant
    /// zero start and the aggregate's real `end` value as stop.
    #[test]
    fn rewrite_general_rangeto_keeps_link_carried_range_defined() {
        let mut g = FunctionGraph::new("test_slice_index_rangeto");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let end_base = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let end_one = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let end = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: end_base,
                    rhs: end_one,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice.clone(), range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // Model the production failure shape: the now-dead RangeTo value is
        // also threaded into a successor phi beside the live slice result.
        let (b, b_args) = g.create_block_with_arg_vars(2);
        g.set_return(b, Some(b_args[0].clone()));
        g.set_goto(a, b, vec![sub, range.clone()]);

        let site = SliceIndexRangeToSite {
            range_result: range.clone(),
            end: end.clone(),
        };
        assert_rangeto_rewrite(&mut g, site);
        for block in &g.blocks {
            for link in &block.exits {
                for arg in &link.args {
                    if let LinkArg::Value(var) = arg {
                        assert!(graph_defines(&g, var), "dangling link arg {var:?}");
                    }
                }
            }
        }
    }

    /// A runtime stop is passed unchanged; no length proof is required.
    #[test]
    fn rewrite_preserves_runtime_end() {
        let mut g = FunctionGraph::new("test_slice_index_rangeto_runtime");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let end = g
            .push_op_var(
                a,
                OpKind::Input {
                    name: "end".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        g.push_inputarg_var(a, end.clone());
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub]);

        let site = SliceIndexRangeToSite {
            range_result: range,
            end,
        };
        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn rangeto_oversized_stop_is_left_for_runtime_clamping() {
        let (mut g, site) = build_rangeto_static_length_graph(None, true, true);
        let end_op = g
            .blocks
            .iter_mut()
            .flat_map(|b| &mut b.operations)
            .find(|op| op.result.as_ref() == Some(&site.end))
            .unwrap();
        end_op.kind = OpKind::ConstUInt(999);
        assert_rangeto_rewrite(&mut g, site);
    }

    #[test]
    fn range_to_rejects_index_mut_and_accepts_array_index() {
        assert!(!slice_index_segments_match(&[
            "core".into(),
            "slice".into(),
            "index".into(),
            "<Impl>".into(),
            "index_mut".into(),
        ]));
        assert!(slice_index_segments_match(&[
            "core".into(),
            "array".into(),
            "<Impl>".into(),
            "index".into(),
        ]));
    }

    #[test]
    fn range_to_accepts_vec_inherent_index_and_rejects_index_mut() {
        assert!(slice_index_segments_match(&[
            "vec".into(),
            "Vec".into(),
            "index".into(),
        ]));
        assert!(slice_index_segments_match(&[
            "alloc".into(),
            "vec".into(),
            "Vec".into(),
            "index".into(),
        ]));
        assert!(slice_index_segments_match(&[
            "alloc".into(),
            "vec".into(),
            "<Impl>".into(),
            "index".into(),
        ]));
        assert!(slice_index_segments_match(&[
            "vec".into(),
            "Vec".into(),
            "<impl core::ops::index::Index<alloc::vec::Vec<v0,v2>,v1>>".into(),
            "index".into(),
        ]));
        assert!(slice_index_segments_match(&[
            "Wtf8".into(),
            "<impl core::ops::index::Index<rustpython_wtf8::Wtf8,core::ops::range::RangeFrom<usize>>>"
                .into(),
            "index".into(),
        ]));
        assert!(!slice_index_segments_match(&[
            "vec".into(),
            "Vec".into(),
            "index_mut".into(),
        ]));
        assert!(!slice_index_segments_match(&[
            "alloc".into(),
            "vec".into(),
            "<Impl>".into(),
            "index_mut".into(),
        ]));
    }

    /// Residual `["vec","Vec","index"]` with a `RangeTo` is the same
    /// `rtype_getslice` rewrite as the slice spelling.
    #[test]
    fn rangeto_rewrites_vec_inherent_index() {
        let mut g = FunctionGraph::new("test_vec_inherent_rangeto");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let end = g
            .push_op_var(
                a,
                OpKind::Input {
                    name: "end".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        g.push_inputarg_var(a, end.clone());
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["vec".into(), "Vec".into(), "index".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![slice, range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub]);

        let site = SliceIndexRangeToSite {
            range_result: range,
            end,
        };
        assert_rangeto_rewrite(&mut g, site);
    }

    /// A RangeFrom whose `usize` start is a runtime value is retained as a
    /// marker; its unsigned annotation proves the start nonnegative to
    /// `decompose_slice_args` when the adapter expands it.
    #[test]
    fn rewrite_lifts_runtime_unsigned_start() {
        let mut g = FunctionGraph::new("test_slice_index_runtime");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        // A runtime start: the result of a BinOp, not a ConstInt.
        let x = g.push_op_var(a, OpKind::ConstInt(5), true).unwrap();
        let start = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: x.clone(),
                    rhs: x.clone(),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangefrom_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeFrom".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: start_field_write(&range, &start),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice.clone(), range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub.clone()]);

        let site = SliceIndexRangeFromSite {
            range_result: range.clone(),
            start: start.clone(),
        };
        let rewritten = rewire_slice_index_rangefrom_sites(&mut g, &[site]);
        assert_eq!(rewritten, 1, "runtime unsigned start is rewritten");
        assert!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        args,
                        ..
                    } if segments == &["__getslice_rangefrom".to_string()]
                        && args == &[slice.clone(), start.clone()]
                )),
            "runtime RangeFrom becomes an adapter marker"
        );
    }

    /// A RangeFrom read by a SECOND consumer (a `.start` field read, here
    /// modeled as another op reading the range value) declines — rewriting it
    /// would break that consumer.
    #[test]
    fn rewrite_declines_multi_consumer_range() {
        let mut g = FunctionGraph::new("test_slice_index_multi");
        let a = g.startblock;
        let slice = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let k = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let range = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: rangefrom_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeFrom".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: start_field_write(&range, &k),
        });
        let sub = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: slice_index_call_target(),
                    args: crate::model::call_args(vec![slice.clone(), range.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // A second consumer: a FieldRead of `.start` on the range value.
        g.push_op_var(
            a,
            OpKind::FieldRead {
                base: range.clone(),
                field: crate::model::FieldDescriptor {
                    name: "start".to_string(),
                    owner_root: Some("core::ops::range::RangeFrom".to_string()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                ty: ValueType::Int,
                pure: true,
            },
            true,
        )
        .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![sub.clone()]);

        let site = SliceIndexRangeFromSite {
            range_result: range.clone(),
            start: k.clone(),
        };
        let rewritten = rewire_slice_index_rangefrom_sites(&mut g, &[site]);
        assert_eq!(rewritten, 0, "a second range consumer declines the rewrite");
        assert!(
            g.blocks
                .iter()
                .flat_map(|blk| &blk.operations)
                .any(|op| is_slice_range_index_call(&op.kind)),
            "residual slice::index call left in place"
        );
    }

    /// An integer-item slice whose only array fact is the input spelling
    /// carries that kind. A length read of `[i64]` is not enough for the
    /// codewriter scan, which names an item only for the object array.
    #[test]
    fn integer_list_input_carries_item_kind_on_the_marker() {
        let mut g = FunctionGraph::new("int_slice");
        let entry = g.startblock;
        let slice = g.alloc_value_var();
        g.push_inputarg_var(entry, slice.clone());
        g.push_op_with_result_var(
            entry,
            OpKind::Input {
                name: "items".into(),
                ty: ValueType::Ref(None),
                class_root: Some("[i64]".into()),
            },
            slice.clone(),
        );
        g.push_op_var(
            entry,
            OpKind::ArrayLen {
                base: slice.clone(),
                array_type_id: Some("[i64]".into()),
                nolength: false,
            },
            true,
        );
        let end = g.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let range = g
            .push_op_var(
                entry,
                OpKind::Call {
                    target: rangeto_ctor_target(),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("core::ops::range::RangeTo".into())),
                },
                true,
            )
            .unwrap();
        g.block_mut(entry).operations.push(SpaceOperation {
            result: None,
            kind: end_field_write(&range, &end),
        });
        g.push_op_var(
            entry,
            OpKind::Call {
                target: slice_index_call_target(),
                args: crate::model::call_args(vec![slice, range.clone()]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
        g.set_return(entry, None);
        let site = SliceIndexRangeToSite {
            range_result: range,
            end,
        };
        assert_eq!(rewire_slice_index_rangeto_sites(&mut g, &[site]), 1);
        let segments = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.first().map(String::as_str) == Some("__getslice_rangeto") => {
                    Some(segments.clone())
                }
                _ => None,
            });
        assert_eq!(
            segments.as_deref(),
            Some(["__getslice_rangeto".to_string(), "int___i64_".to_string()].as_slice())
        );
    }
}
