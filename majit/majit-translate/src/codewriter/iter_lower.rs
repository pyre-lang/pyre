//! Iterator scalarisation on the legacy rich-`OpKind` spine.
//!
//! The front lowers a Rust `for` loop into deferred markers:
//! `it = core::slice::iter(x)`, then `item = __iter_next(it)` guarded by a
//! `StopIteration` exception edge (`front/iter_next.rs`), and — for
//! `for i in a..b` — `x = __majit_range(a, b)` feeding the same `iter`
//! bridge (`front/range_iter.rs`).  A lifted graph meets the rtyper, whose
//! `ListIteratorRepr::rtype_next` / range repr (`rlist.py`, `rrange.py`)
//! scalarise the iterator into loop-carried state.  A graph on the
//! rich-`OpKind` spine keeps the markers all the way to the codewriter,
//! where each becomes a symbolic residual no host symbol backs — a wall
//! between a builtin gateway body and its trace.
//!
//! This pass gives the spine the rtyper's answer.  The iterator value is
//! replaced by the container it walks, and an index is threaded as one
//! extra `inputarg` through every block that carries the iterator:
//!
//! * slice iteration (`ll_listnext`): the `__iter_next` block branches on
//!   `index < len(container)` into a fresh advance block reading
//!   `container[index]` and stepping `index + 1`;
//! * range iteration (`ll_rangenext_up`): the carried value is the range's
//!   stop (`RANGEITER` is `GcStruct("range", ("next", Signed), ("stop",
//!   Signed))`), the branch is `index < stop`, and the item is the index.
//!   Loop-reachable forwarders phi both fields, because `regalloc.py`
//!   `make_dependencies` only keeps a value live across a back-edge when
//!   that edge's `link.args` name it;
//! * enumerate iteration (`EnumerateIteratorRepr.rtype_next`, `rrange.py`):
//!   `x = __majit_enumerate(lst)` feeds the `iter` op, the carried value is
//!   the list, and the item is the `(index, lst[index])` tuple — the index
//!   read (`ll_getnextindex`) before the list iterator steps.
//!
//! Fail-safe: a site whose surrounding shape is not the plain loop the
//! front emits (an iterator escaping into another op, a phi slot mixing
//! iterators, a `next` outside a `StopIteration` block) is left alone and
//! keeps today's symbolic residual.

use std::collections::{HashMap, HashSet};

use crate::codewriter::getslice::{LinkClasses, array_identity_of_base};
use crate::codewriter::type_state::valuetype_to_concrete;
use crate::flowspace::model::{ConstValue, Constant, Variable};
use crate::front::result_exc::op_operand_vars;
use crate::model::{
    BlockId, CallTarget, ConcreteType, ExitCase, ExitSwitch, FieldDescriptor, FunctionGraph, Link,
    LinkArg, OpKind, ValueType,
};
use crate::runtime_names::shims::{ENUMERATE, RANGE};

/// Scalarise every lowerable iterator site in `graph`, one at a time —
/// each rewrite invalidates the link classes the next site is judged by.
pub fn lower_iterators(graph: &mut FunctionGraph) {
    // The rtyper's `rtype_next` meets SSI graphs, where the `__iter_next`
    // block carries the iterator through its inputargs. The front emits
    // global SSA, so a header that calls `__iter_next(it)` without that
    // inputarg would decline. Thread it first (`ssa.py SSA_to_SSI`) in
    // that case only. A seed / merge / back-edge-only body that already
    // has a member header must stay an extra_index carrier: RANGEITER is
    // `("next", Signed), ("stop", Signed)` (`rrange.py`), and
    // `checkgraph` (`flowspace/model.py`) forbids a seed from naming the
    // constructor's locals. Unconditional SSI would promote those
    // carriers into members and pass the constructor's stop name on the
    // back-edge.
    if next_sites_need_ssi(graph) {
        crate::model_ssa::ssa_to_ssi(graph);
    }
    // More sites than any real graph carries; a backstop, not a budget.
    for _ in 0..64 {
        if !lower_one_site(graph) {
            return;
        }
    }
}

/// What the iterator walks.
enum IterKind {
    /// A GC array: item reads come from the array itself.
    Slice {
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// `__majit_range(start, stop)`: the index is the item.
    Range { start: Variable, stop: Variable },
    /// `__majit_enumerate(lst)`: the item is the `(index, lst[index])`
    /// tuple.
    Enumerate {
        list: Variable,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
}

fn lower_one_site(graph: &mut FunctionGraph) -> bool {
    let anchors: Vec<(usize, usize)> = graph
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(bi, block)| {
            block
                .operations
                .iter()
                .enumerate()
                .filter(|(_, op)| is_slice_iter_call(&op.kind))
                .map(move |(oi, _)| (bi, oi))
        })
        .collect();
    for (bi, oi) in anchors {
        if lower_site(graph, bi, oi).is_ok() {
            return true;
        }
    }
    false
}

/// One `it = core::slice::iter(x)` site: validate the whole loop shape
/// first, then rewrite.  Any deviation declines before the first
/// mutation.
fn lower_site(graph: &mut FunctionGraph, d: usize, anchor_idx: usize) -> Result<(), String> {
    let d_id = BlockId(d);
    let (it, x) = {
        let op = &graph.blocks[d].operations[anchor_idx];
        let OpKind::Call { args, .. } = &op.kind else {
            return Err("anchor is not a call".into());
        };
        let Some(result) = &op.result else {
            return Err("iter constructor has no result".into());
        };
        (result.clone(), args[0].clone().into_variable())
    };
    if !in_scope(graph, d, &x) {
        return Err("iterated container out of scope".into());
    }

    // What the iterator walks: a range or enumerate marker feeding the
    // constructor in the same block, else a GC array with a known identity.
    let marker_site = graph.blocks[d]
        .operations
        .iter()
        .enumerate()
        .find_map(|(ri, op)| {
            if op.result.as_ref() != Some(&x) {
                return None;
            }
            is_marker_call(&op.kind, RANGE, 2)
                .or_else(|| is_marker_call(&op.kind, ENUMERATE, 1))
                .map(|args| (ri, args))
        });
    let kind = if let Some((_, args)) = &marker_site {
        for v in args {
            if !in_scope(graph, d, v) {
                return Err("marker operand out of scope".into());
            }
        }
        match args.as_slice() {
            [start, stop] => IterKind::Range {
                start: start.clone(),
                stop: stop.clone(),
            },
            [list] => {
                let Some((item_ty, array_type_id)) = array_identity_of_base(graph, list) else {
                    return Err("enumerated container has no array identity".into());
                };
                IterKind::Enumerate {
                    list: list.clone(),
                    item_ty,
                    array_type_id,
                }
            }
            _ => unreachable!("marker arities are 1 and 2"),
        }
    } else {
        let Some((item_ty, array_type_id)) = array_identity_of_base(graph, &x) else {
            return Err("container has no array identity".into());
        };
        IterKind::Slice {
            item_ty,
            array_type_id,
        }
    };
    let range_op_idx = marker_site.as_ref().map(|(ri, _)| *ri);
    if range_op_idx.is_some() {
        // The marker value must feed only the constructor: its op is
        // deleted alongside the anchor.
        for (bi, block) in graph.blocks.iter().enumerate() {
            for (oi, op) in block.operations.iter().enumerate() {
                if bi == d && (oi == anchor_idx || Some(oi) == range_op_idx) {
                    continue;
                }
                if op_operand_vars(&op.kind).contains(&x) {
                    return Err("range value escapes into an op".into());
                }
            }
        }
        if graph.blocks[d].exits.iter().any(|link| {
            link.args
                .iter()
                .any(|a| matches!(a, LinkArg::Value(v) if *v == x))
        }) {
            return Err("range value escapes through a link".into());
        }
    }
    let carried = match &kind {
        IterKind::Range { stop, .. } => stop.clone(),
        IterKind::Enumerate { list, .. } => list.clone(),
        IterKind::Slice { .. } => x.clone(),
    };

    let classes = LinkClasses::of(graph);
    let in_class = |v: &Variable| classes.same(v, &it);

    // Blocks whose inputargs carry the iterator, and at which slots.
    let mut member_positions: HashMap<usize, Vec<usize>> = HashMap::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        let positions: Vec<usize> = block
            .inputargs
            .iter()
            .enumerate()
            .filter(|(_, v)| in_class(v))
            .map(|(i, _)| i)
            .collect();
        if positions.is_empty() {
            continue;
        }
        if bi == d {
            return Err("constructor block also carries an iterator".into());
        }
        member_positions.insert(bi, positions);
    }

    // The `__iter_next` sites over this iterator: each must be its
    // block's closing op under a `StopIteration` exception switch.
    struct NextSite {
        block: usize,
        iter_arg: Variable,
        item: Variable,
        /// The enumerate `next`'s `(index, item)` tuple class.
        tuple_owner: Option<String>,
        some_link: Link,
        break_link: Link,
    }
    let mut next_sites: Vec<NextSite> = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        let sites: Vec<usize> = block
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| {
                is_marker_call(&op.kind, "__iter_next", 1).is_some_and(|args| in_class(&args[0]))
            })
            .map(|(i, _)| i)
            .collect();
        if sites.is_empty() {
            continue;
        }
        if sites.len() > 1 || sites[0] + 1 != block.operations.len() {
            return Err(format!("next site in block {bi} is not the closing op"));
        }
        let op = &block.operations[sites[0]];
        let Some(item) = op.result.clone() else {
            return Err("next call has no result".into());
        };
        let iter_arg =
            is_marker_call(&op.kind, "__iter_next", 1).expect("filtered above")[0].clone();
        let tuple_owner = match &op.kind {
            OpKind::Call {
                result_ty: ValueType::Ref(Some(owner)),
                ..
            } if owner.starts_with("Tuple<") => Some(owner.clone()),
            _ => None,
        };
        if matches!(kind, IterKind::Enumerate { .. }) && tuple_owner.is_none() {
            return Err(format!(
                "enumerate next in block {bi} does not yield a tuple"
            ));
        }
        if !matches!(block.exitswitch, Some(ExitSwitch::LastException)) {
            return Err(format!("next block {bi} is not an exception switch"));
        }
        if block.exits.len() != 2 {
            return Err(format!("next block {bi} has {} exits", block.exits.len()));
        }
        let mut some_link = None;
        let mut break_link = None;
        for link in &block.exits {
            match &link.exitcase {
                None => some_link = Some(link.clone()),
                Some(_) => break_link = Some(link.clone()),
            }
        }
        let (Some(some_link), Some(break_link)) = (some_link, break_link) else {
            return Err(format!("next block {bi} lacks the normal/handler pair"));
        };
        if break_link
            .args
            .iter()
            .any(|a| matches!(a, LinkArg::Value(v) if *v == item))
        {
            return Err("exhaustion link carries the item".into());
        }
        if !member_positions.contains_key(&bi) {
            return Err(format!("next block {bi} does not carry the iterator"));
        }
        next_sites.push(NextSite {
            block: bi,
            iter_arg,
            item,
            tuple_owner,
            some_link,
            break_link,
        });
    }

    // The iterator must be opaque everywhere else: no op reads or
    // produces a class value besides the constructor and the next sites,
    // and no exitswitch tests one.
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let is_anchor = bi == d && oi == anchor_idx;
            let is_range = bi == d && Some(oi) == range_op_idx;
            let is_next = next_sites
                .iter()
                .any(|s| s.block == bi && oi + 1 == block.operations.len());
            if is_anchor || is_range || is_next {
                continue;
            }
            if op_operand_vars(&op.kind).iter().any(&in_class) {
                return Err(format!("iterator read by op {oi} of block {bi}"));
            }
            if op.result.as_ref().is_some_and(|v| *v != it && in_class(v)) {
                return Err(format!("iterator slot written by op {oi} of block {bi}"));
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(v)) if in_class(v) => {
                return Err(format!("iterator is the exitswitch of block {bi}"));
            }
            Some(ExitSwitch::Fused { args, .. }) if args.iter().any(&in_class) => {
                return Err(format!(
                    "iterator is a fused exitswitch operand of block {bi}"
                ));
            }
            _ => {}
        }
    }

    // Every predecessor of a member block must supply the iterator as a
    // value. A constant in that slot means the phi mixes non-iterator
    // values. A non-member that feeds a member — loop-reachable merge or
    // constructor-adjacent seed — already carries the iterator object
    // (`ll_rangenext_up` keeps `iter.next` and `iter.stop` on it);
    // scalarisation phis both. `checkgraph` (`flowspace/model.py`)
    // forbids a seed from naming the constructor's locals, so the seed
    // is a carrier too: the constructor is the only block that still
    // passes the original `carried` / start.
    let mut extra_index_blocks: HashSet<usize> = HashSet::new();
    let mut grew = true;
    while grew {
        grew = false;
        for (si, block) in graph.blocks.iter().enumerate() {
            if si == d || member_positions.contains_key(&si) || extra_index_blocks.contains(&si) {
                continue;
            }
            for link in &block.exits {
                let feeds_index_carrier = member_positions.contains_key(&link.target.0)
                    || extra_index_blocks.contains(&link.target.0);
                if !feeds_index_carrier {
                    continue;
                }
                if let Some(positions) = member_positions.get(&link.target.0) {
                    for &p in positions {
                        if !matches!(link.args.get(p), Some(LinkArg::Value(_))) {
                            return Err(format!(
                                "iterator slot {p} of block {} fed a constant",
                                link.target.0
                            ));
                        }
                    }
                }
                extra_index_blocks.insert(si);
                grew = true;
            }
        }
    }
    for block in graph.blocks.iter() {
        for link in &block.exits {
            let Some(positions) = member_positions.get(&link.target.0) else {
                continue;
            };
            for &p in positions {
                if !matches!(link.args.get(p), Some(LinkArg::Value(_))) {
                    return Err(format!(
                        "iterator slot {p} of block {} fed a constant",
                        link.target.0
                    ));
                }
            }
        }
    }

    // mutation: everything below is committed

    let mut idx_of: HashMap<usize, Variable> = HashMap::new();
    for &bi in member_positions.keys().chain(extra_index_blocks.iter()) {
        let idx = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&idx, ConcreteType::Signed);
        idx_of.insert(bi, idx);
    }

    // The phi slots that used to hold the iterator now hold the carried
    // value.  Each slot is a fresh Variable: the later hydration step
    // (`apply_from_flowspace_variables`) maps a reused identity back to
    // the rtyper's iterator type, which would retag the stop/container
    // after fusion has already captured it.  The advance blocks inherit
    // their inputarg concretetypes from these slots, so the replacement
    // happens before anything copies their kinds.
    let carried_ct = carried.concretetype.borrow().clone();
    for (&bi, positions) in &member_positions {
        for &p in positions {
            let old = graph.blocks[bi].inputargs[p].clone();
            let fresh = graph.alloc_value_var();
            fresh.set_concretetype(carried_ct.clone());
            crate::model_ssa::renamevariables(graph, BlockId(bi), &old, &fresh);
            for site in &mut next_sites {
                if site.iter_arg == old {
                    site.iter_arg = fresh.clone();
                }
                rename_link_uses(&mut site.some_link, &old, &fresh);
                rename_link_uses(&mut site.break_link, &old, &fresh);
            }
        }
    }

    // Per-block SSA name of the RANGEITER `.stop` (or the slice / enumerate
    // container). Members already hold it in the rewritten iterator slot.
    // Loop-reachable forwarders get a fresh phi; advance blocks record the
    // copy they already take as an inputarg.
    let mut carried_of: HashMap<usize, Variable> = HashMap::new();
    for (&bi, positions) in &member_positions {
        carried_of.insert(bi, graph.blocks[bi].inputargs[positions[0]].clone());
    }
    for &bi in &extra_index_blocks {
        let c = graph.alloc_value_var();
        c.set_concretetype(carried_ct.clone());
        carried_of.insert(bi, c);
    }

    for site in &next_sites {
        let a = BlockId(site.block);
        let idx_a = idx_of[&site.block].clone();
        // Drop the closing `__iter_next` op; its block now branches on
        // the index instead of catching `StopIteration`.
        graph.block_mut(a).operations.pop();
        let bound = match &kind {
            IterKind::Slice { array_type_id, .. } | IterKind::Enumerate { array_type_id, .. } => {
                let len = push(
                    graph,
                    a,
                    OpKind::ArrayLen {
                        base: site.iter_arg.clone(),
                        array_type_id: array_type_id.clone(),
                        nolength: false,
                    },
                );
                FunctionGraph::set_concretetype_of_inline(&len, ConcreteType::Signed);
                len
            }
            IterKind::Range { .. } => site.iter_arg.clone(),
        };
        let cond = push(
            graph,
            a,
            OpKind::BinOp {
                op: "lt".into(),
                lhs: idx_a.clone(),
                rhs: bound,
                result_ty: ValueType::Bool,
            },
        );
        FunctionGraph::set_concretetype_of_inline(&cond, ConcreteType::Signed);
        let cond = push(
            graph,
            a,
            OpKind::UnaryOp {
                op: "bool".into(),
                operand: cond,
                result_ty: ValueType::Bool,
            },
        );
        FunctionGraph::set_concretetype_of_inline(&cond, ConcreteType::Signed);

        // The advance block: read the item, step the index, resume the
        // normal arm.  Its inputargs are the values the resume link
        // needs, plus the container and the index.
        let mut sources: Vec<Variable> = Vec::new();
        for arg in &site.some_link.args {
            if let LinkArg::Value(v) = arg
                && *v != site.item
                && !sources.contains(v)
            {
                sources.push(v.clone());
            }
        }
        for v in [&site.iter_arg, &idx_a] {
            if !sources.contains(v) {
                sources.push(v.clone());
            }
        }
        let (bp, bp_inputs) = graph.create_block_with_arg_vars(sources.len());
        for (src, dst) in sources.iter().zip(&bp_inputs) {
            let ct = src.concretetype.borrow().clone();
            dst.set_concretetype(ct);
        }
        let map: HashMap<Variable, Variable> = sources
            .iter()
            .cloned()
            .zip(bp_inputs.iter().cloned())
            .collect();
        let bp_idx = map[&idx_a].clone();
        match &kind {
            IterKind::Slice {
                item_ty,
                array_type_id,
            } => {
                graph.push_op_with_result_var(
                    bp,
                    OpKind::ArrayRead {
                        base: map[&site.iter_arg].clone(),
                        index: bp_idx.clone(),
                        item_ty: item_ty.clone(),
                        array_type_id: array_type_id.clone(),
                        nolength: crate::front::typestr::nolength_from_array_type_id(
                            array_type_id.as_deref(),
                        ),
                        pure: false,
                    },
                    site.item.clone(),
                );
            }
            IterKind::Range { .. } => {
                graph.push_op_with_result_var(
                    bp,
                    OpKind::UnaryOp {
                        op: "same_as".into(),
                        operand: bp_idx.clone(),
                        result_ty: ValueType::Int,
                    },
                    site.item.clone(),
                );
            }
            IterKind::Enumerate {
                item_ty,
                array_type_id,
                ..
            } => {
                let item = push(
                    graph,
                    bp,
                    OpKind::ArrayRead {
                        base: map[&site.iter_arg].clone(),
                        index: bp_idx.clone(),
                        item_ty: item_ty.clone(),
                        array_type_id: array_type_id.clone(),
                        nolength: crate::front::typestr::nolength_from_array_type_id(
                            array_type_id.as_deref(),
                        ),
                        pure: false,
                    },
                );
                FunctionGraph::set_concretetype_of_inline(&item, valuetype_to_concrete(item_ty));
                let owner = site
                    .tuple_owner
                    .clone()
                    .expect("validated: enumerate next names its tuple");
                graph.push_op_with_result_var(
                    bp,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor(&owner),
                        args: Vec::new(),
                        result_ty: ValueType::Ref(Some(owner.clone())),
                    },
                    site.item.clone(),
                );
                FunctionGraph::set_concretetype_of_inline(&site.item, ConcreteType::GcRef);
                for (pos, value, ty) in [
                    ("__pos_0", bp_idx.clone(), ValueType::Unsigned),
                    ("__pos_1", item, item_ty.clone()),
                ] {
                    graph.push_op_var(
                        bp,
                        OpKind::FieldWrite {
                            base: site.item.clone(),
                            field: FieldDescriptor::new(pos, Some(owner.clone())),
                            value: LinkArg::Value(value),
                            ty,
                        },
                        false,
                    );
                }
            }
        }
        let one = push(graph, bp, OpKind::ConstInt(1));
        FunctionGraph::set_concretetype_of_inline(&one, ConcreteType::Signed);
        let next_idx = push(
            graph,
            bp,
            OpKind::BinOp {
                op: "add".into(),
                lhs: bp_idx,
                rhs: one,
                result_ty: ValueType::Int,
            },
        );
        FunctionGraph::set_concretetype_of_inline(&next_idx, ConcreteType::Signed);

        let resume_args: Vec<LinkArg> = site
            .some_link
            .args
            .iter()
            .map(|arg| match arg {
                LinkArg::Value(v) if *v == site.item => LinkArg::Value(site.item.clone()),
                LinkArg::Value(v) => LinkArg::Value(map[v].clone()),
                other => other.clone(),
            })
            .collect();
        let resume = Link::new_mixed(resume_args, site.some_link.target, None);
        graph.set_control_flow_metadata(bp, None, vec![resume]);

        let exhausted = Link::new_mixed(
            site.break_link.args.clone(),
            site.break_link.target,
            Some(ExitCase::Bool(false)),
        )
        .with_llexitcase_from_exitcase();
        let advance = Link::new_mixed(
            sources.iter().cloned().map(LinkArg::Value).collect(),
            bp,
            Some(ExitCase::Bool(true)),
        )
        .with_llexitcase_from_exitcase();
        graph.set_control_flow_metadata(a, Some(ExitSwitch::Value(cond)), vec![exhausted, advance]);
        idx_of.insert(bp.0, next_idx);
        carried_of.insert(bp.0, map[&site.iter_arg].clone());
    }

    // Thread the index through every block that carries it: members and
    // loop-reachable forwarders. `rrange.py` `RangeIteratorRepr.rtype_next`
    // keeps that index on the iterator object as it flows through a merge.
    // Advance blocks minted below already take the index as an inputarg
    // on the constructed edge; they are in `idx_of` so a successor can
    // read the stepped index, but they are not in this carrier set.
    let index_carriers: HashSet<usize> = member_positions
        .keys()
        .copied()
        .chain(extra_index_blocks.iter().copied())
        .collect();
    for &bi in &index_carriers {
        graph.push_inputarg_var(BlockId(bi), idx_of[&bi].clone());
    }
    let init_arg = match &kind {
        IterKind::Range { start, .. } => LinkArg::Value(start.clone()),
        IterKind::Slice { .. } | IterKind::Enumerate { .. } => {
            LinkArg::Const(Constant::new(ConstValue::Int(0)))
        }
    };
    for bi in 0..graph.blocks.len() {
        for li in 0..graph.blocks[bi].exits.len() {
            let target = graph.blocks[bi].exits[li].target.0;
            if !index_carriers.contains(&target) {
                continue;
            }
            let arg = match idx_of.get(&bi) {
                Some(idx) => LinkArg::Value(idx.clone()),
                None => match &init_arg {
                    LinkArg::Const(_) => init_arg.clone(),
                    LinkArg::Value(v) if in_scope(graph, bi, v) => init_arg.clone(),
                    _ => panic!(
                        "iter_lower: block {bi} has no live index to pass \
                         (graph {:?})",
                        graph.name
                    ),
                },
            };
            graph.blocks[bi].exits[li].args.push(arg);
        }
    }

    // `regalloc.py` `make_dependencies` pops `link.args` from `die_at`, so
    // a value lives through a block only when a successor still names it.
    // Members already carry stop in the iterator slot; extra_index
    // forwarders (and a body that only names the iterator on the
    // back-edge) must too, otherwise `ConstInt(1)` / `args_base + i` in
    // the advance or body reuse the header's bound colour and the loop
    // compare reads the index.
    for &bi in &extra_index_blocks {
        graph.push_inputarg_var(BlockId(bi), carried_of[&bi].clone());
    }
    for bi in 0..graph.blocks.len() {
        for li in 0..graph.blocks[bi].exits.len() {
            let target = graph.blocks[bi].exits[li].target.0;
            if !extra_index_blocks.contains(&target) {
                continue;
            }
            let arg = LinkArg::Value(live_carried(graph, bi, &carried, &carried_of));
            graph.blocks[bi].exits[li].args.push(arg);
        }
    }

    // Retire the constructor: the iterator value becomes the carried
    // one, and the range or enumerate marker (whose only consumer was the
    // constructor) goes with it. Rewrite every non-member edge into a
    // member, not only `d`'s: a forwarding block may still name the
    // constructor's `it`. An extra_index predecessor passes its live
    // stop phi, not the constructor's global-SSA name — that name is
    // not live in the predecessor, so using it would leave the same
    // colour free for a loop temp (`regalloc.py` only interferes inside
    // a block).
    let removed: Vec<Variable> = match &kind {
        IterKind::Range { .. } | IterKind::Enumerate { .. } => vec![it.clone(), x.clone()],
        IterKind::Slice { .. } => vec![it.clone()],
    };
    graph
        .block_mut(d_id)
        .operations
        .retain(|op| !op.result.as_ref().is_some_and(|r| removed.contains(r)));
    for bi in 0..graph.blocks.len() {
        if member_positions.contains_key(&bi) {
            continue;
        }
        let names_it_into_member = graph.blocks[bi].exits.iter().any(|link| {
            member_positions.contains_key(&link.target.0)
                && link
                    .args
                    .iter()
                    .any(|a| matches!(a, LinkArg::Value(v) if *v == it))
        });
        if !names_it_into_member {
            continue;
        }
        let replacement = live_carried(graph, bi, &carried, &carried_of);
        for link in &mut graph.blocks[bi].exits {
            if !member_positions.contains_key(&link.target.0) {
                continue;
            }
            for arg in &mut link.args {
                if matches!(arg, LinkArg::Value(v) if *v == it) {
                    *arg = LinkArg::Value(replacement.clone());
                }
            }
        }
    }

    Ok(())
}

fn is_slice_iter_call(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } if args.len() == 1
        && segments.len() >= 3
        && segments[0] == "core"
        && segments[1] == "slice"
        && segments.last().is_some_and(|s| s == "iter"))
}

fn is_marker_call(kind: &OpKind, name: &str, arity: usize) -> Option<Vec<Variable>> {
    match kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.len() == 1 && segments[0] == name && args.len() == arity => {
            Some(crate::model::call_arg_vars(args))
        }
        _ => None,
    }
}

/// `v` is usable inside `block`: one of its inputargs or op results.
fn in_scope(graph: &FunctionGraph, block: usize, v: &Variable) -> bool {
    let b = &graph.blocks[block];
    b.inputargs.contains(v) || b.operations.iter().any(|op| op.result.as_ref() == Some(v))
}

/// The stop / container a block may pass on an exit. `checkgraph`
/// (`flowspace/model.py`) only lets a link name a Variable defined in
/// its source. Members and extra_index carriers already hold a live phi
/// in `carried_of`; using the constructor's global-SSA name would leave
/// that colour free for a loop temp (`regalloc.py` `make_dependencies`).
/// The constructor itself is not in `carried_of` and still passes the
/// original `carried`.
fn live_carried(
    graph: &FunctionGraph,
    bi: usize,
    carried: &Variable,
    carried_of: &HashMap<usize, Variable>,
) -> Variable {
    if let Some(phi) = carried_of.get(&bi) {
        return phi.clone();
    }
    if in_scope(graph, bi, carried) {
        carried.clone()
    } else {
        panic!(
            "iter_lower: block {bi} has no live stop to pass (graph {:?})",
            graph.name
        )
    }
}

/// True when some `__iter_next` reads an iterator that is not already an
/// inputarg of its block. Those headers need `ssa.py SSA_to_SSI` before
/// `rtype_next` can see a member. Seed / merge / back-edge-only bodies
/// that feed an already-member header do not.
fn next_sites_need_ssi(graph: &FunctionGraph) -> bool {
    let has_site = graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| is_slice_iter_call(&op.kind))
    });
    if !has_site {
        return false;
    }
    let classes = LinkClasses::of(graph);
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(args) = is_marker_call(&op.kind, "__iter_next", 1) else {
                continue;
            };
            let iter_arg = &args[0];
            if !block.inputargs.iter().any(|v| classes.same(v, iter_arg)) {
                return true;
            }
        }
    }
    false
}

fn push(graph: &mut FunctionGraph, block: BlockId, kind: OpKind) -> Variable {
    graph
        .push_op_var(block, kind, true)
        .expect("a value-producing op allocates its result")
}

fn rename_link_uses(link: &mut Link, old: &Variable, fresh: &Variable) {
    for arg in &mut link.args {
        if let LinkArg::Value(v) = arg
            && *v == *old
        {
            *v = fresh.clone();
        }
    }
    for extra in [&mut link.last_exception, &mut link.last_exc_value] {
        if let Some(LinkArg::Value(v)) = extra
            && *v == *old
        {
            *v = fresh.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn call(segments: &[&str], args: Vec<Variable>) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: segments.iter().map(|s| s.to_string()).collect(),
                fun_decl_id: None,
                generic_rust_args: Vec::new(),
            },
            args: crate::model::call_args(args),
            result_ty: ValueType::Ref(None),
        }
    }

    fn marker_count(graph: &FunctionGraph) -> usize {
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| {
                is_slice_iter_call(&op.kind)
                    || is_marker_call(&op.kind, "__iter_next", 1).is_some()
                    || is_marker_call(&op.kind, RANGE, 2).is_some()
                    || is_marker_call(&op.kind, ENUMERATE, 1).is_some()
            })
            .count()
    }

    fn stopiteration() -> ExitCase {
        ExitCase::Const(ConstValue::builtin("StopIteration"))
    }

    /// `flowspace/model.py` `checkgraph`: every Variable a block uses
    /// (op operand, exitswitch, `link.args`) is defined in that block.
    fn assert_model_checkgraph(graph: &FunctionGraph) {
        for (bi, block) in graph.blocks.iter().enumerate() {
            let mut defined: HashSet<Variable> = block.inputargs.iter().cloned().collect();
            for op in &block.operations {
                for v in op_operand_vars(&op.kind) {
                    assert!(
                        defined.contains(&v),
                        "checkgraph: block {bi} uses {} before definition (graph {:?})",
                        v.name(),
                        graph.name
                    );
                }
                if let Some(r) = &op.result {
                    defined.insert(r.clone());
                }
            }
            match &block.exitswitch {
                Some(ExitSwitch::Value(v)) => {
                    assert!(
                        defined.contains(v),
                        "checkgraph: block {bi} exitswitch {} used before definition (graph {:?})",
                        v.name(),
                        graph.name
                    );
                }
                Some(ExitSwitch::Fused { args, .. }) => {
                    for v in args {
                        assert!(
                            defined.contains(v),
                            "checkgraph: block {bi} fused exitswitch {} used before definition \
                             (graph {:?})",
                            v.name(),
                            graph.name
                        );
                    }
                }
                Some(ExitSwitch::LastException) | None => {}
            }
            for (li, link) in block.exits.iter().enumerate() {
                let n_in = graph.block(link.target).inputargs.len();
                assert_eq!(
                    link.args.len(),
                    n_in,
                    "checkgraph: block {bi} link[{li}] has {} args, target {} has {n_in} inputargs \
                     (graph {:?})",
                    link.args.len(),
                    link.target.0,
                    graph.name
                );
                for (ai, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(v) = arg else {
                        continue;
                    };
                    assert!(
                        defined.contains(v),
                        "checkgraph: block {bi} link[{li}].args[{ai}] {} used before definition \
                         (graph {:?})",
                        v.name(),
                        graph.name
                    );
                }
            }
        }
    }

    /// `for i in 0..n { use(i) }` in the front's shape.
    fn range_loop_graph() -> (FunctionGraph, Variable, BlockId, BlockId) {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let n = g.alloc_value_var();
        g.push_inputarg_var(d, n.clone());
        g.push_op_with_result_var(
            d,
            OpKind::Input {
                name: "n".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            n.clone(),
        );
        let zero = push(&mut g, d, OpKind::ConstInt(0));
        let t = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&[RANGE], vec![zero.clone(), n]), t.clone());
        let it = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&["core", "slice", "iter"], vec![t]), it.clone());

        let (head, head_args) = g.create_block_with_arg_vars(1);
        g.set_goto(d, head, vec![it]);
        let h_it = head_args[0].clone();
        let item = g.alloc_value_var();
        g.push_op_with_result_var(
            head,
            call(&["__iter_next"], vec![h_it.clone()]),
            item.clone(),
        );
        let (body, body_args) = g.create_block_with_arg_vars(2);
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link =
            Link::new_mixed(vec![LinkArg::Value(h_it), LinkArg::Value(item)], body, None);
        let mut stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        stop_link.last_exception = Some(LinkArg::Value(g.alloc_value_var()));
        stop_link.last_exc_value = Some(LinkArg::Value(g.alloc_value_var()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_goto(body, head, vec![body_args[0].clone()]);
        g.set_return(done, None);
        (g, zero, head, body)
    }

    #[test]
    fn a_range_loop_scalarises_to_an_index() {
        let (mut g, zero, head, body) = range_loop_graph();
        assert_eq!(marker_count(&g), 3);
        // The resolver committed the stop bound as an integer and the
        // iterator slot as a pointer; after scalarisation the slot carries
        // the stop, and every copy of it (the advance block's inputargs
        // included) must follow the integer kind.
        FunctionGraph::set_concretetype_of_inline(
            &g.block(g.startblock).inputargs[0],
            ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(&g.block(head).inputargs[0], ConcreteType::GcRef);
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);

        // The next block now branches on the index.
        let head_block = g.block(head);
        assert!(matches!(head_block.exitswitch, Some(ExitSwitch::Value(_))));
        assert_eq!(head_block.exits.len(), 2);
        assert_eq!(head_block.exits[0].exitcase, Some(ExitCase::Bool(false)));
        assert_eq!(head_block.exits[1].exitcase, Some(ExitCase::Bool(true)));
        // Head and body each carry the extra index slot.
        assert_eq!(head_block.inputargs.len(), 2);
        assert_eq!(g.block(body).inputargs.len(), 3);
        // The constructor block seeds the index with the range start.
        let seed = g.block(g.startblock).exits[0].args.last().unwrap();
        assert_eq!(seed, &LinkArg::Value(zero));
        // The advance block computes the item and the stepped index.
        let bp = g.block(head_block.exits[1].target);
        assert!(
            bp.operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as"))
        );
        assert!(
            bp.operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
        );
        // No Ref kind survives on the scalarised slots or their copies.
        assert_eq!(
            FunctionGraph::concretetype_of(&g.block(head).inputargs[0]),
            ConcreteType::Signed,
        );
        for input in &bp.inputargs {
            assert_ne!(
                FunctionGraph::concretetype_of(input),
                ConcreteType::GcRef,
                "advance block inherited the retired iterator kind",
            );
        }
    }

    /// Constructor → empty forwarder → header, the shape `get_and_call_function`
    /// emits between `Vec::with_capacity` and the `0..len` header.
    /// `RangeIteratorRepr.rtype_next` seeds start on that edge.
    #[test]
    fn a_range_loop_with_a_seed_forwarder_scalarises() {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let n = g.alloc_value_var();
        g.push_inputarg_var(d, n.clone());
        g.push_op_with_result_var(
            d,
            OpKind::Input {
                name: "n".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            n.clone(),
        );
        let zero = push(&mut g, d, OpKind::ConstInt(0));
        let t = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&[RANGE], vec![zero.clone(), n]), t.clone());
        let it = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&["core", "slice", "iter"], vec![t]), it.clone());
        let (fwd, _) = g.create_block_with_arg_vars(0);
        g.set_goto(d, fwd, vec![]);
        let (head, head_args) = g.create_block_with_arg_vars(1);
        g.set_goto(fwd, head, vec![it.clone()]);
        let h_it = head_args[0].clone();
        let item = g.alloc_value_var();
        g.push_op_with_result_var(
            head,
            call(&["__iter_next"], vec![h_it.clone()]),
            item.clone(),
        );
        let (body, body_args) = g.create_block_with_arg_vars(2);
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link =
            Link::new_mixed(vec![LinkArg::Value(h_it), LinkArg::Value(item)], body, None);
        let mut stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        stop_link.last_exception = Some(LinkArg::Value(g.alloc_value_var()));
        stop_link.last_exc_value = Some(LinkArg::Value(g.alloc_value_var()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_goto(body, head, vec![body_args[0].clone()]);
        g.set_return(done, None);
        FunctionGraph::set_concretetype_of_inline(
            &g.block(g.startblock).inputargs[0],
            ConcreteType::Signed,
        );
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);
        // Seed is a carrier: constructor passes start and stop; the
        // forwarder names its own phis (`checkgraph`).
        assert_eq!(g.block(fwd).inputargs.len(), 2);
        let d_args = &g.block(d).exits[0].args;
        assert_eq!(&d_args[0], &LinkArg::Value(zero.clone()));
        assert_eq!(
            g.block(fwd).exits[0].args.last().unwrap(),
            &LinkArg::Value(g.block(fwd).inputargs[0].clone()),
        );
    }

    /// Constructor → merge forwarder → header, with the body jumping back
    /// through the merge. The merge is loop-reachable and must phi the
    /// index; seeding start on that back-edge would restart the loop.
    #[test]
    fn a_range_loop_with_a_merge_forwarder_phis_the_index() {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let n = g.alloc_value_var();
        g.push_inputarg_var(d, n.clone());
        g.push_op_with_result_var(
            d,
            OpKind::Input {
                name: "n".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            n.clone(),
        );
        let zero = push(&mut g, d, OpKind::ConstInt(0));
        let t = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&[RANGE], vec![zero.clone(), n]), t.clone());
        let it = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&["core", "slice", "iter"], vec![t]), it.clone());
        let (fwd, _) = g.create_block_with_arg_vars(0);
        g.set_goto(d, fwd, vec![]);
        let (head, head_args) = g.create_block_with_arg_vars(1);
        g.set_goto(fwd, head, vec![it.clone()]);
        let h_it = head_args[0].clone();
        let item = g.alloc_value_var();
        g.push_op_with_result_var(
            head,
            call(&["__iter_next"], vec![h_it.clone()]),
            item.clone(),
        );
        let (body, body_args) = g.create_block_with_arg_vars(2);
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link =
            Link::new_mixed(vec![LinkArg::Value(h_it), LinkArg::Value(item)], body, None);
        let mut stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        stop_link.last_exception = Some(LinkArg::Value(g.alloc_value_var()));
        stop_link.last_exc_value = Some(LinkArg::Value(g.alloc_value_var()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_goto(body, fwd, vec![]);
        g.set_return(done, None);
        let _ = body_args;
        FunctionGraph::set_concretetype_of_inline(
            &g.block(g.startblock).inputargs[0],
            ConcreteType::Signed,
        );
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);
        // Index then stop: RANGEITER is `("next", Signed), ("stop", Signed)`.
        assert_eq!(g.block(fwd).inputargs.len(), 2);
        let d_args = &g.block(d).exits[0].args;
        assert_eq!(&d_args[d_args.len() - 2], &LinkArg::Value(zero.clone()));
        let body_exit = &g.block(body).exits[0].args;
        assert!(matches!(&body_exit[body_exit.len() - 2], LinkArg::Value(_)));
        assert_ne!(&body_exit[body_exit.len() - 2], &LinkArg::Value(zero));
    }

    /// Production `get_and_call_function` shape: the body uses `i` and
    /// names the iterator only on the back-edge (global-SSA), so it is an
    /// extra_index carrier rather than a member. Stop must still phi
    /// through that body, or `regalloc.py` colours `ConstInt(1)` with the
    /// header bound.
    #[test]
    fn a_range_loop_body_that_only_names_the_iterator_on_the_backedge_phis_stop() {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let n = g.alloc_value_var();
        g.push_inputarg_var(d, n.clone());
        g.push_op_with_result_var(
            d,
            OpKind::Input {
                name: "n".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            n.clone(),
        );
        let zero = push(&mut g, d, OpKind::ConstInt(0));
        let t = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&[RANGE], vec![zero, n.clone()]), t.clone());
        let it = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&["core", "slice", "iter"], vec![t]), it.clone());
        let (head, head_args) = g.create_block_with_arg_vars(1);
        g.set_goto(d, head, vec![it.clone()]);
        let h_it = head_args[0].clone();
        let item = g.alloc_value_var();
        g.push_op_with_result_var(
            head,
            call(&["__iter_next"], vec![h_it.clone()]),
            item.clone(),
        );
        let (body, _) = g.create_block_with_arg_vars(1);
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link = Link::new_mixed(vec![LinkArg::Value(item)], body, None);
        let mut stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        stop_link.last_exception = Some(LinkArg::Value(g.alloc_value_var()));
        stop_link.last_exc_value = Some(LinkArg::Value(g.alloc_value_var()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_goto(body, head, vec![it]);
        g.set_return(done, None);
        FunctionGraph::set_concretetype_of_inline(&g.block(d).inputargs[0], ConcreteType::Signed);
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);
        assert_eq!(g.block(head).inputargs.len(), 2);
        // item, index, stop
        assert_eq!(g.block(body).inputargs.len(), 3);
        let back = &g.block(body).exits[0].args;
        assert_eq!(
            &back[0],
            &LinkArg::Value(g.block(body).inputargs[2].clone()),
            "back-edge must pass the body's live stop, not the constructor name"
        );
        let bp_id = g.block(head).exits[1].target;
        let one = g
            .block(bp_id)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::ConstInt(1) => op.result.clone(),
                _ => None,
            })
            .expect("advance steps the index by 1");
        let stop = g.block(head).inputargs[0].clone();
        let ints = crate::codewriter::regalloc::perform_register_allocation(
            &g,
            crate::flatten::RegKind::Int,
        );
        let stop_color = ints
            .color_for_variable(&stop)
            .expect("header stop is coloured");
        let one_color = ints
            .color_for_variable(&one)
            .expect("ConstInt(1) is coloured");
        assert_ne!(
            stop_color, one_color,
            "header stop must interfere with the advance temp (`regalloc.py` die_at on link.args)"
        );
        assemble_graph(g);
    }

    /// Global SSA: the header calls `__iter_next(it)` without carrying `it`
    /// as an inputarg. `ssa.py SSA_to_SSI` threads it first so `rtype_next`
    /// sees a member. Extra_index carriers that already have a member
    /// header must not take this path.
    #[test]
    fn a_range_loop_whose_next_uses_the_constructor_name_still_scalarises() {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let n = g.alloc_value_var();
        g.push_inputarg_var(d, n.clone());
        g.push_op_with_result_var(
            d,
            OpKind::Input {
                name: "n".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            n.clone(),
        );
        let zero = push(&mut g, d, OpKind::ConstInt(0));
        let t = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&[RANGE], vec![zero, n]), t.clone());
        let it = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&["core", "slice", "iter"], vec![t]), it.clone());
        let (head, _) = g.create_block_with_arg_vars(0);
        g.set_goto(d, head, vec![]);
        let item = g.alloc_value_var();
        g.push_op_with_result_var(head, call(&["__iter_next"], vec![it.clone()]), item.clone());
        let (body, _) = g.create_block_with_arg_vars(1);
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link = Link::new_mixed(vec![LinkArg::Value(item)], body, None);
        let mut stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        stop_link.last_exception = Some(LinkArg::Value(g.alloc_value_var()));
        stop_link.last_exc_value = Some(LinkArg::Value(g.alloc_value_var()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_goto(body, head, vec![]);
        g.set_return(done, None);
        FunctionGraph::set_concretetype_of_inline(&g.block(d).inputargs[0], ConcreteType::Signed);
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);
        assert_eq!(g.block(head).inputargs.len(), 2);
    }

    /// The scalarised range slot keeps its integer kind through the rest of
    /// the codewriter: the rtyper typed that slot as the range iterator
    /// pointer, and nothing after jtransform may copy that type back.
    fn assemble_graph(g: FunctionGraph) {
        let path = crate::parse::CallPath::from_segments(["f"]);
        let mut callcontrol = crate::codewriter::call::CallControl::new();
        callcontrol.register_function_graph(path.clone(), g.clone());
        let mut codewriter = crate::codewriter::codewriter::CodeWriter::new();
        let jitcode = std::sync::Arc::new(crate::codewriter::jitcode::JitCode::new("f"));
        codewriter.transform_graph_to_jitcode(
            &g,
            &path,
            &mut callcontrol,
            &crate::GraphTransformConfig::default(),
            &jitcode,
            false,
            0,
        );
    }

    #[test]
    fn a_scalarised_range_slot_is_a_fresh_signed_variable() {
        let (mut g, _zero, head, _body) = range_loop_graph();
        FunctionGraph::set_concretetype_of_inline(
            &g.block(g.startblock).inputargs[0],
            ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(&g.block(head).inputargs[0], ConcreteType::GcRef);
        let original = g.block(head).inputargs[0].clone();
        lower_iterators(&mut g);
        let carried = &g.block(head).inputargs[0];
        assert_ne!(carried, &original);
        assert_eq!(
            FunctionGraph::concretetype_of(carried),
            ConcreteType::Signed,
        );
    }

    #[test]
    fn a_range_loop_assembles_through_the_codewriter() {
        let (g, _, _, _) = range_loop_graph();
        assemble_graph(g);
    }

    /// `for &w in args_w { use(w) }` in the front's shape, with a sibling
    /// length read establishing the array identity.
    fn slice_loop_graph() -> (FunctionGraph, BlockId, BlockId) {
        slice_loop_graph_with(false)
    }

    /// The same loop over `enumerate(args_w)` when `enumerate`: the
    /// marker feeds the `iter` op and `next` yields the `(i, w)` tuple.
    fn slice_loop_graph_with(enumerate: bool) -> (FunctionGraph, BlockId, BlockId) {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let l = g.alloc_value_var();
        g.push_inputarg_var(d, l.clone());
        g.push_op_with_result_var(
            d,
            OpKind::Input {
                name: "args_w".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            l.clone(),
        );
        push(
            &mut g,
            d,
            OpKind::ArrayLen {
                base: l.clone(),
                array_type_id: Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID.into()),
                nolength: false,
            },
        );
        let iterated = if enumerate {
            let e = g.alloc_value_var();
            g.push_op_with_result_var(d, call(&[ENUMERATE], vec![l.clone()]), e.clone());
            e
        } else {
            l.clone()
        };
        let it = g.alloc_value_var();
        g.push_op_with_result_var(
            d,
            call(&["core", "slice", "iter"], vec![iterated]),
            it.clone(),
        );

        let (head, head_args) = g.create_block_with_arg_vars(2);
        g.set_goto(d, head, vec![l, it]);
        let h_it = head_args[1].clone();
        let item = g.alloc_value_var();
        let mut next = call(&["__iter_next"], vec![h_it.clone()]);
        if enumerate && let OpKind::Call { result_ty, .. } = &mut next {
            *result_ty = ValueType::Ref(Some("Tuple<usize,Ptr>".into()));
        }
        g.push_op_with_result_var(head, next, item.clone());
        let (body, body_args) = g.create_block_with_arg_vars(3);
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link = Link::new_mixed(
            vec![
                LinkArg::Value(head_args[0].clone()),
                LinkArg::Value(h_it),
                LinkArg::Value(item),
            ],
            body,
            None,
        );
        let stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_goto(body, head, vec![body_args[0].clone(), body_args[1].clone()]);
        g.set_return(done, None);
        (g, head, body)
    }

    #[test]
    fn a_slice_loop_scalarises_to_indexed_reads() {
        let (mut g, head, body) = slice_loop_graph();
        assert_eq!(marker_count(&g), 2);
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);

        let head_block = g.block(head);
        assert!(matches!(head_block.exitswitch, Some(ExitSwitch::Value(_))));
        assert!(
            head_block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::ArrayLen { .. }))
        );
        assert_eq!(head_block.inputargs.len(), 3);
        assert_eq!(g.block(body).inputargs.len(), 4);
        // The constructor block seeds the index with zero.
        let seed = g.block(g.startblock).exits[0].args.last().unwrap();
        assert!(matches!(seed, LinkArg::Const(c) if c.value == ConstValue::Int(0)));
        // The advance block reads the item out of the array.
        let bp = g.block(head_block.exits[1].target);
        assert!(
            bp.operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }))
        );
    }

    #[test]
    fn an_enumerate_loop_builds_the_index_item_tuple() {
        let (mut g, head, body) = slice_loop_graph_with(true);
        assert_eq!(marker_count(&g), 3);
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), 0);
        assert_model_checkgraph(&g);

        let head_block = g.block(head);
        assert!(matches!(head_block.exitswitch, Some(ExitSwitch::Value(_))));
        // The carried value is the list itself: its length bounds the index.
        assert!(
            head_block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::ArrayLen { .. }))
        );
        assert_eq!(g.block(body).inputargs.len(), 4);
        let seed = g.block(g.startblock).exits[0].args.last().unwrap();
        assert!(matches!(seed, LinkArg::Const(c) if c.value == ConstValue::Int(0)));
        // The advance block reads `lst[index]` and packs `(index, item)`
        // into the tuple `next` yielded.
        let bp = g.block(head_block.exits[1].target);
        let bp_idx = bp.inputargs.last().unwrap().clone();
        let item = bp
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::ArrayRead { index, .. } if *index == bp_idx => op.result.clone(),
                _ => None,
            })
            .expect("item read at the index");
        let tuple = bp
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } if name == "Tuple<usize,Ptr>" => op.result.clone(),
                _ => None,
            })
            .expect("tuple ctor");
        let writes: Vec<(String, Variable)> = bp
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite {
                    base,
                    field,
                    value: LinkArg::Value(v),
                    ..
                } if *base == tuple => Some((field.name.clone(), v.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                ("__pos_0".to_string(), bp_idx),
                ("__pos_1".to_string(), item)
            ]
        );
        // The resume link hands the tuple to the loop body.
        assert!(
            bp.exits[0]
                .args
                .iter()
                .any(|a| matches!(a, LinkArg::Value(v) if *v == tuple))
        );
    }

    #[test]
    fn an_unknown_container_declines() {
        let mut g = FunctionGraph::new("f");
        let d = g.startblock;
        let l = g.alloc_value_var();
        g.push_inputarg_var(d, l.clone());
        let it = g.alloc_value_var();
        g.push_op_with_result_var(d, call(&["core", "slice", "iter"], vec![l]), it.clone());
        let (head, head_args) = g.create_block_with_arg_vars(1);
        g.set_goto(d, head, vec![it]);
        let item = g.alloc_value_var();
        g.push_op_with_result_var(
            head,
            call(&["__iter_next"], vec![head_args[0].clone()]),
            item,
        );
        let (done, _) = g.create_block_with_arg_vars(0);
        let some_link = Link::new_mixed(vec![], done, None);
        let stop_link = Link::new_mixed(vec![], done, Some(stopiteration()));
        g.set_control_flow_metadata(
            head,
            Some(ExitSwitch::LastException),
            vec![some_link, stop_link],
        );
        g.set_return(done, None);
        let before = marker_count(&g);
        lower_iterators(&mut g);
        assert_eq!(marker_count(&g), before);
    }
}
