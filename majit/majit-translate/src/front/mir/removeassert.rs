//! Assertion stripping (`rpython/translator/backendopt/removeassert.py`).

use super::*;

/// `alloc::fmt::format`, `fmt::Arguments` constructors, `fmt::rt::Argument::new_*`,
/// and `__str_const`.  Graph-less std externs analogous to
/// `simplify.py` `CanRemoveBuiltins`; rustc still evaluates them before
/// `panic_fmt`.  Interpreter functions with graphs (`kind_name`,
/// `expected_fmt_arg`) are not listed: `has_no_side_effects` walks a
/// flowspace graph under a rtyper and is not reachable from this
/// `FunctionGraph`-only pass, so those calls stay.
pub(in crate::front::mir) fn is_abort_message_call(segments: &[String]) -> bool {
    if segments.first().map(String::as_str) == Some("__str_const") {
        return true;
    }
    if fmt_path_ends_with(segments, &["fmt", "format"]) {
        return true;
    }
    let Some(raw_leaf) = segments.last() else {
        return false;
    };
    let leaf = crate::front::clause_spec::unspecialized_leaf(raw_leaf);
    if segments.iter().any(|s| s == "Argument") && leaf.starts_with("new_") {
        return true;
    }
    if segments.iter().any(|s| s == "Arguments")
        && (matches!(
            leaf,
            "new" | "new_v1" | "new_const" | "from_str" | "from_str_nonconst"
        ) || leaf.starts_with("new_"))
    {
        return true;
    }
    // `core::fmt::<Impl>::new` when the owner stayed an `<Impl>` placeholder
    // (`is_arguments_from_str`: inherent-impl `name_path` carries `<Impl>`,
    // not `Arguments`).  Needed for that Charon spelling of `Arguments::new`.
    leaf == "new"
        && segments.iter().any(|s| s == "fmt")
        && !segments.iter().any(|s| s == "Argument")
}

/// Front-end synthetic `__len` (`Rvalue::Len` / `<str>::is_empty` / vec
/// length): the `len` of `simplify.py` `CanRemove`.  `can_remove_op` does
/// not classify a `Call` as removable, so the exact single-segment
/// `FunctionPath` the front emits is matched here.  Never a Method name.
pub(in crate::front::mir) fn is_abort_message_synthetic_len(segments: &[String]) -> bool {
    segments.len() == 1 && segments[0] == "__len"
}

pub(in crate::front::mir) fn abort_message_call_target(target: &crate::model::CallTarget) -> bool {
    use crate::model::CallTarget;
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            is_abort_message_call(segments) || is_abort_message_synthetic_len(segments)
        }
        CallTarget::Method { resolved_path, .. } => resolved_path
            .as_ref()
            .is_some_and(|p| is_abort_message_call(&p.segments)),
        CallTarget::SyntheticTransparentCtor { name, .. } => {
            matches!(name.as_str(), "Tuple" | "Array" | "Some")
        }
        _ => false,
    }
}

/// Result ids of `is_fresh_aggregate_ctor` producers in the abort region.
/// A ctor in a predecessor region block filling the pieces array counts.
pub(in crate::front::mir) fn region_fresh_aggregate_ids(
    graph: &FunctionGraph,
    region: &std::collections::HashSet<BlockId>,
) -> std::collections::HashSet<u64> {
    graph
        .blocks
        .iter()
        .filter(|block| region.contains(&block.id))
        .flat_map(|block| &block.operations)
        .filter(|op| crate::model::is_fresh_aggregate_ctor(&op.kind))
        .filter_map(|op| op.result.as_ref().map(|r| r.id()))
        .collect()
}

pub(in crate::front::mir) fn is_abort_message_op(
    kind: &crate::model::OpKind,
    fresh_aggregates: &std::collections::HashSet<u64>,
) -> bool {
    use crate::model::OpKind;
    match kind {
        // `graph_body.rs` `stamp_declared` / `declaration` pin startblock
        // `Input` ops to the funcobj header. A unique-pred chain from the
        // startblock into an abort (a body that is only `panic!("…{}", x)`)
        // still has to keep those parameters even when they feed only the
        // melted message.
        OpKind::Input { .. } => false,
        OpKind::Call { target, .. } => abort_message_call_target(target),
        // `remove_simple_mallocs`: a `setfield` is dropped only when its
        // base is a fresh malloc in the region.  A store through an Input,
        // a FieldRead, a cast, or any other base is a heap side effect
        // (`transform_dead_op_vars` never removes `setfield`).
        OpKind::FieldWrite { base, .. } => fresh_aggregates.contains(&base.id()),
        _ => crate::inline::can_remove_op(kind),
    }
}

pub(in crate::front::mir) fn abort_message_op_is_live(
    op: &crate::model::SpaceOperation,
    fresh_aggregates: &std::collections::HashSet<u64>,
    live: &std::collections::HashSet<u64>,
) -> bool {
    use crate::model::OpKind;
    if !is_abort_message_op(&op.kind, fresh_aggregates) {
        return false;
    }
    match &op.kind {
        // A write into a live aggregate (the pieces / args array whose
        // `Arguments::new` result escaped) must keep its value operand.
        OpKind::FieldWrite { base, .. } => live.contains(&base.id()),
        _ => op.result.as_ref().is_some_and(|r| live.contains(&r.id())),
    }
}

pub(in crate::front::mir) fn link_raises_implicit_assertion_error(
    link: &crate::model::Link,
    exceptblock: BlockId,
    assert_err: &crate::flowspace::model::HostObject,
) -> bool {
    use crate::flowspace::model::ConstValue;
    use crate::model::LinkArg;
    link.target == exceptblock
        && matches!(
            link.args.first(),
            Some(LinkArg::Const(c))
                if matches!(&c.value, ConstValue::HostObject(h) if *h == *assert_err)
        )
}

/// Abort block plus the unique-predecessor chain that has no other successor
/// — the blocks rustc fills with `format_args!` before `panic_fmt`.
pub(in crate::front::mir) fn abort_message_region(
    graph: &FunctionGraph,
    abort: BlockId,
) -> Vec<BlockId> {
    let mut region = vec![abort];
    let mut cur = abort;
    for _ in 0..graph.blocks.len() {
        let preds = graph.predecessors(cur);
        if preds.len() != 1 {
            break;
        }
        let pred = preds[0];
        if graph.block(pred).exits.len() != 1 {
            break;
        }
        region.push(pred);
        cur = pred;
    }
    region
}

/// Melt abort-bound message construction (`removeassert.py` "melt away the
/// (hopefully) dead operation that compute the condition").  Ops whose
/// result is used only as the panicking call's format operands disappear;
/// a store into an escaped object, and a `format!` whose result is live,
/// stay.
pub(in crate::front::mir) fn melt_abort_message_chain(
    graph: &mut FunctionGraph,
    abort: BlockId,
) -> usize {
    use crate::model::LinkArg;
    let region = abort_message_region(graph, abort);
    let region_set: std::collections::HashSet<BlockId> = region.iter().copied().collect();
    let exceptblock = graph.exceptblock;
    let mut outside_used: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for block in &graph.blocks {
        let in_region = region_set.contains(&block.id);
        if !in_region {
            for op in &block.operations {
                for v in crate::inline::op_variable_refs(&op.kind) {
                    outside_used.insert(v.id());
                }
            }
        }
        for exit in &block.exits {
            if in_region && (region_set.contains(&exit.target) || exit.target == exceptblock) {
                continue;
            }
            for arg in &exit.args {
                if let Some(v) = arg.as_variable() {
                    outside_used.insert(v.id());
                }
            }
        }
    }
    let fresh_aggregates = region_fresh_aggregate_ids(graph, &region_set);
    let mut live = outside_used;
    for block in &graph.blocks {
        if !region_set.contains(&block.id) {
            continue;
        }
        for op in &block.operations {
            if is_abort_message_op(&op.kind, &fresh_aggregates) {
                continue;
            }
            for v in crate::inline::op_variable_refs(&op.kind) {
                live.insert(v.id());
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for bi in 0..graph.blocks.len() {
            let block_id = graph.blocks[bi].id;
            if !region_set.contains(&block_id) {
                continue;
            }
            let live_inputs: Vec<usize> = graph.blocks[bi]
                .inputargs
                .iter()
                .enumerate()
                .filter(|(_, inp)| live.contains(&inp.id()))
                .map(|(i, _)| i)
                .collect();
            let preds = graph.predecessors(block_id);
            for pred in preds {
                if !region_set.contains(&pred) {
                    continue;
                }
                for exit in &graph.block(pred).exits {
                    if exit.target != block_id {
                        continue;
                    }
                    for &i in &live_inputs {
                        if let Some(LinkArg::Value(v)) = exit.args.get(i)
                            && live.insert(v.id())
                        {
                            changed = true;
                        }
                    }
                }
            }
            let pull: Vec<u64> = graph.blocks[bi]
                .operations
                .iter()
                .filter(|op| abort_message_op_is_live(op, &fresh_aggregates, &live))
                .flat_map(|op| crate::inline::op_variable_refs(&op.kind))
                .map(|v| v.id())
                .collect();
            for id in pull {
                if live.insert(id) {
                    changed = true;
                }
            }
        }
    }
    let mut deleted = 0usize;
    for block in &mut graph.blocks {
        if !region_set.contains(&block.id) {
            continue;
        }
        let dead: Vec<usize> = block
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| {
                is_abort_message_op(&op.kind, &fresh_aggregates)
                    && !abort_message_op_is_live(op, &fresh_aggregates, &live)
            })
            .map(|(i, _)| i)
            .collect();
        if dead.is_empty() {
            continue;
        }
        deleted += dead.len();
        let drop: std::collections::HashSet<usize> = dead.into_iter().collect();
        let mut kept = Vec::with_capacity(block.operations.len() - drop.len());
        for (i, op) in block.operations.drain(..).enumerate() {
            if !drop.contains(&i) {
                kept.push(op);
            }
        }
        block.operations = kept;
    }
    deleted
}

pub(in crate::front::mir) fn melt_abort_message_ops(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::HOST_ENV;
    let assert_err = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock = graph.exceptblock;
    let aborts: Vec<BlockId> = graph
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                block.exits.as_slice(),
                [exit] if link_raises_implicit_assertion_error(exit, exceptblock, &assert_err)
            )
        })
        .map(|block| block.id)
        .collect();
    let mut deleted = 0usize;
    for abort in aborts {
        deleted += melt_abort_message_chain(graph, abort);
    }
    deleted
}

/// A block is collapsible into a bare implicit-`AssertionError` raise when
/// every op is a pure value / ctor / field op or a recognised `fmt`
/// message extern — the message-building work a Rust `panic!` / `assert!`
/// emits before its diverging call.  A `FieldWrite` is allowed only when
/// its base aggregate was constructed earlier in the same block (the
/// `(a, b)` tuple / `Some(_)` / pieces array the message builder fills),
/// so a write into an escaped object (`self.field = …; panic!()`) keeps
/// the block non-collapsible and its effect preserved.  Any other `Call`
/// — a real side-effecting function such as `Write::write_fmt` — likewise
/// blocks the collapse.
pub(in crate::front::mir) fn panic_block_is_pure_message(block: &crate::model::Block) -> bool {
    use crate::model::OpKind;
    let mut produced = std::collections::HashSet::new();
    for op in &block.operations {
        let pure = match &op.kind {
            OpKind::Input { .. } => false,
            OpKind::Call { target, .. } => abort_message_call_target(target),
            OpKind::FieldWrite { base, .. } => produced.contains(&base.id()),
            kind => crate::inline::can_remove_op(kind),
        };
        if !pure {
            return false;
        }
        if let Some(r) = &op.result {
            produced.insert(r.id());
        }
    }
    true
}

/// Collapse a Rust `panic!` / `assert!` message-building block chain into
/// a bare implicit-`AssertionError` raise, matching RPython's `_implicit_`
/// exception form (`flowcontext.py`; `remove_assertion_errors`,
/// simplify.py): RPython's implicit exceptions are direct raises
/// carrying no computed message, whereas Rust lowers `panic!("…{}", x)`
/// into a chain of message blocks (`fmt::rt::Argument::new_display`,
/// `fmt::Arguments::new`, the on-stack pieces / args arrays) ending in the
/// `Abort` → `set_raise_implicit(AssertionError)` exit.  Those message
/// blocks are graph-less host externs the rtyper can't type, so they wall
/// the dual gate.
///
/// Each message block has a single exit and only pure message ops
/// (`panic_block_is_pure_message`); the chain tail exits to `exceptblock`
/// carrying the `[AssertionError, value]` pair `set_raise_implicit`
/// installs.  Growing that property backward over single-exit edges yields
/// the set of collapsible blocks; the deciding block at the head (a `bool`
/// switch whose panic arm enters the chain) is not collapsible.  Its panic
/// edge is retargeted straight to `exceptblock` with the chain's
/// `[AssertionError, value]` args, leaving the message blocks unreachable
/// for `clear_unreachable_blocks`, after which `remove_assertion_errors`
/// prunes the now-direct AssertionError edge exactly as for an
/// already-direct implicit raise.
///
/// Returns the number of retargeted deciding edges so the caller can gate
/// the follow-up `clear_unreachable_blocks` / `simplify_lowered_graph`.
pub(in crate::front::mir) fn collapse_panic_message_chains(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::{ConstValue, HOST_ENV};
    use crate::model::LinkArg;
    use std::collections::HashMap;
    let assert_err = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock = graph.exceptblock;
    let exit_raises_assert = |link: &crate::model::Link| -> bool {
        link.target == exceptblock
            && matches!(
                link.args.first(),
                Some(LinkArg::Const(c))
                    if matches!(&c.value, ConstValue::HostObject(h) if *h == assert_err)
            )
    };
    // Grow the collapsible set backward over single-exit pure-message
    // edges, keyed to the `[AssertionError, value]` args the tail raise
    // carries.
    let mut collapsible: HashMap<crate::model::BlockId, Vec<LinkArg>> = HashMap::new();
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            if collapsible.contains_key(&block.id) || block.exits.len() != 1 {
                continue;
            }
            if !panic_block_is_pure_message(block) {
                continue;
            }
            let exit = &block.exits[0];
            let raise_args = if exit_raises_assert(exit) {
                Some(exit.args.clone())
            } else {
                collapsible.get(&exit.target).cloned()
            };
            if let Some(args) = raise_args {
                collapsible.insert(block.id, args);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Retarget every edge from a non-collapsible block into a collapsible
    // block straight to `exceptblock` with the chain's raise args; the
    // collapsible blocks themselves keep their edges and fall out as
    // unreachable.
    let mut redirected = 0usize;
    for block_idx in 0..graph.blocks.len() {
        if collapsible.contains_key(&graph.blocks[block_idx].id) {
            continue;
        }
        for ei in 0..graph.blocks[block_idx].exits.len() {
            let target = graph.blocks[block_idx].exits[ei].target;
            let Some(args) = collapsible.get(&target) else {
                continue;
            };
            let args = args.clone();
            let exit = &mut graph.blocks[block_idx].exits[ei];
            exit.target = exceptblock;
            exit.args = args;
            redirected += 1;
        }
    }
    redirected
}

/// `l.length` as a value defined in `block`.
///
/// View creation records that word outside `local_var`, so
/// `FrameState.getoutputargs` / `flowcontext.py` `make_next_block`
/// do not carry it. A later `.len()` / `Rvalue::Len` must not name
/// the predecessor Variable on a link (`checkgraph` `definevar`).
///
/// When `len_var` is already an inputarg or op result of `block`, it
/// is returned. Otherwise this emits `same_as` so the dest / link
/// operand is an op result of `block`. After the CFG closes,
/// `thread_undefined_op_operands` (`ensure_variable_at_block_local`,
/// `SSA_to_SSI`) threads the `same_as` operand. Calling
/// `ensure_variable_at_block_local` during lowering would add an
/// inputarg that framestate pass 2 then drops.
pub(crate) fn bind_array_view_len_in_block(
    graph: &mut FunctionGraph,
    block: BlockId,
    len_var: Variable,
) -> Variable {
    if graph.variable_defined_in_block(block, &len_var) {
        return len_var;
    }
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: len_var,
            result_ty: ValueType::Int,
        },
    });
    res
}
