//! Graph inlining utility — substitute Call ops with callee body graphs.
//!
//! **Note:** RPython's codewriter does NOT inline callee bodies into callers.
//! Instead, `jtransform.py` rewrites `direct_call` to `inline_call_*`
//! (referencing the callee's JitCode) and the meta-interpreter descends
//! into callee JitCode at runtime.
//!
//! This module provides graph-level body splicing for use cases where
//! actual body expansion is needed (e.g. analysis, testing). It is NOT
//! part of the RPython-orthodox codewriter pipeline.

use std::collections::{HashMap, HashSet};

use crate::call::{CallControl, CallKind};
use crate::model::{
    BlockId, CallFuncPtr, ExitSwitch, FunctionGraph, Link, OpKind, SpaceOperation,
    remap_control_flow_metadata_var,
};

fn remap_call_funcptr<
    F: Fn(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
>(
    funcptr: &CallFuncPtr,
    remap_var: &F,
) -> CallFuncPtr {
    match funcptr {
        CallFuncPtr::Target(target) => CallFuncPtr::Target(target.clone()),
        CallFuncPtr::Value(var) => CallFuncPtr::Value(remap_var(var)),
    }
}

/// Inline all `Regular` calls in the graph, consulting `CallControl`
/// for the inline/residual decision.
///
/// RPython equivalent: flow space auto-inlining + `backendopt/inline.py`.
///
/// Returns the number of call sites inlined.
pub fn inline_graph(
    graph: &mut FunctionGraph,
    call_control: &CallControl,
    max_depth: usize,
) -> usize {
    let mut total_inlined = 0;
    for _depth in 0..max_depth {
        let sites = find_inline_sites(graph, call_control);
        if sites.is_empty() {
            break;
        }
        // Process sites in reverse order so block indices remain valid
        // (later blocks are not affected by earlier inlining).
        for site in sites.into_iter().rev() {
            let _ = inline_call_site(graph, site);
            total_inlined += 1;
        }
    }
    total_inlined
}

/// A call site eligible for inlining.
struct InlineSite {
    block_id: BlockId,
    op_index: usize,
    callee: FunctionGraph,
}

/// Find all Call ops in the graph where `CallControl` says `Regular`.
fn find_inline_sites(graph: &FunctionGraph, call_control: &CallControl) -> Vec<InlineSite> {
    let mut sites = Vec::new();
    for block in &graph.blocks {
        for (op_idx, op) in block.operations.iter().enumerate() {
            // Inline only direct calls — RPython's inline pass skips
            // indirect family dispatch (each callee is resolved
            // dynamically at runtime, not statically at inline time).
            let target = match &op.kind {
                OpKind::Call { target, .. } => target,
                _ => continue,
            };
            if call_control.guess_call_kind(op) != CallKind::Regular {
                continue;
            }
            let callee = match call_control.direct_graph_for(target) {
                Some(g) => FunctionGraph::clone(&g),
                None => continue,
            };
            sites.push(InlineSite {
                block_id: block.id,
                op_index: op_idx,
                callee,
            });
        }
    }
    sites
}

/// Inline a single call site.
///
/// Algorithm:
/// 1. Split the caller block at the call site into "before" and "after"
/// 2. Remap all callee values/blocks to fresh IDs in the caller graph
/// 3. Connect: before → callee entry (passing call args)
/// 4. Connect: callee Return → merge block (passing return value)
/// 5. Merge block continues with the "after" ops
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn inline_call_site(
    graph: &mut FunctionGraph,
    site: InlineSite,
) -> Option<crate::flowspace::model::Variable> {
    let InlineSite {
        block_id,
        op_index,
        callee,
    } = site;

    // Extract the call op details
    let block = &graph.blocks[block_id.0];
    let call_op = &block.operations[op_index];
    let (call_args, call_result) = match &call_op.kind {
        OpKind::Call { args, .. } => (args.clone(), call_op.result.clone()),
        _ => unreachable!("InlineSite should point to a Call op"),
    };

    // Separate ops into before-call and after-call.  Upstream
    // `rpython/flowspace/model.py` treats `Block.exitswitch` +
    // `Block.exits` as the single CFG source of truth, so they move
    // together with the ops they guard into the merge block.
    let after_ops: Vec<SpaceOperation> = block.operations[op_index + 1..].to_vec();
    let after_exitswitch = block.exitswitch.clone();
    let after_exits = block.exits.clone();
    // `Inliner.do_inline` captures `original_passon_vars` from the link
    // `unsimplify.split_block` builds: every caller value the after-block
    // still uses. `Inliner.passon_vars` then copies that list onto each
    // inlined block. Collected before the call op is deleted.
    let passon = caller_passon_vars(&after_ops, &after_exitswitch, &after_exits, &call_result);

    // Truncate the original block to before-call ops only
    graph.blocks[block_id.0].operations.truncate(op_index);

    // Remap callee values and blocks
    let value_map = remap_callee_values(graph, &callee);
    let block_map = remap_callee_blocks(graph, &callee);

    // Create merge block for after-call ops
    // Upstream `backendopt/inline.py copy_block` copies caller-block-after
    // ops + exits into a fresh afterblock whenever there is something
    // to preserve.  Pyre creates the merge block when (a) after-call
    // ops exist, (b) the caller block was already closed with exits or
    // an exitswitch, or (c) the call produced a result that downstream
    // code consumes.  (Order matches upstream's `(exits) or (stmts)`
    // guard.)
    let caller_was_closed = !after_exits.is_empty() || after_exitswitch.is_some();
    let merge_block_id = if !after_ops.is_empty() || caller_was_closed || call_result.is_some() {
        let (merge_id, merge_args) = if let Some(original_result_var) = call_result {
            let (id, arg_vars) = graph.create_block_with_arg_vars(1);
            // The merge block's inputarg replaces the original call result.
            // Remap every reference to `original_result_var` → `arg_vars[0]` in
            // after-call ops and exit metadata so the phi-node-style
            // merge carries the callee's return value forward.
            let merge_inputarg_var = arg_vars[0].clone();
            let remapped_after_ops =
                remap_value_in_ops(&after_ops, &original_result_var, &merge_inputarg_var);
            graph.blocks[id.0].operations = remapped_after_ops;
            let (remapped_switch, remapped_exits) = remap_control_flow_metadata_var(
                &after_exitswitch,
                &after_exits,
                |var| {
                    if var == &original_result_var {
                        merge_inputarg_var.clone()
                    } else {
                        var.clone()
                    }
                },
                |b| b,
            );
            graph.set_control_flow_metadata(id, remapped_switch, remapped_exits);
            (id, arg_vars)
        } else {
            let id = graph.create_block();
            graph.blocks[id.0].operations = after_ops;
            let (remapped_switch, remapped_exits) = remap_control_flow_metadata_var(
                &after_exitswitch,
                &after_exits,
                |var| var.clone(),
                |b| b,
            );
            graph.set_control_flow_metadata(id, remapped_switch, remapped_exits);
            (id, Vec::<crate::flowspace::model::Variable>::new())
        };

        Some((merge_id, merge_args))
    } else {
        None
    };

    // Copy callee blocks into the graph
    let callee_entry = *block_map.get(&callee.startblock).unwrap();

    for callee_block in &callee.blocks {
        let new_block_id = block_map[&callee_block.id];

        // Remap inputargs (callee Variables → caller-graph Variables
        // via value_map).
        let new_inputargs: Vec<crate::flowspace::model::Variable> = callee_block
            .inputargs
            .iter()
            .map(|var| value_map[var].clone())
            .collect();
        graph.blocks[new_block_id.0].inputargs = new_inputargs;

        // Remap ops
        let new_ops: Vec<SpaceOperation> = callee_block
            .operations
            .iter()
            .map(|op| remap_op(op, &value_map))
            .collect();
        graph.blocks[new_block_id.0].operations = new_ops;

        // Identify the callee's canonical returnblock by ID, matching
        // upstream `inline.py` `rewire_returnblock`, which reads
        // `graph_to_inline.returnblock`
        // and rewires its exits to point at the caller's afterblock.
        let is_returnblock = callee_block.id == callee.returnblock;
        if is_returnblock {
            // `inline.py` `rewire_returnblock`:
            //   copiedreturnblock = copy_block(self.graph_to_inline.returnblock)
            //   linkargs = ([copiedreturnblock.inputargs[0]] + passon_vars)
            //   linkfrominlined = Link(linkargs, afterblock)
            //   copiedreturnblock.exitswitch = None
            //   copiedreturnblock.recloseblock(linkfrominlined)
            // When there is no afterblock (the call was the caller
            // block's only statement and the block terminates unclosed),
            // forward to the caller graph's canonical returnblock so
            // the inlined function's return value becomes the caller's
            // return value — still a Goto into a final block, matching
            // upstream's `exits=[Link(..., returnblock)]` shape.
            let ret_val: Option<crate::flowspace::model::Variable> = callee_block
                .inputargs
                .first()
                .map(|var| value_map[var].clone());
            let caller_returnblock = graph.returnblock;
            let (target, arg_vars) = match (&merge_block_id, ret_val) {
                (Some((merge_id, merge_args)), Some(remapped_ret)) => {
                    if merge_args.is_empty() {
                        (*merge_id, vec![])
                    } else {
                        (*merge_id, vec![remapped_ret])
                    }
                }
                (Some((merge_id, _)), None) => (*merge_id, vec![]),
                (None, Some(remapped_ret)) => (caller_returnblock, vec![remapped_ret]),
                (None, None) => (caller_returnblock, vec![]),
            };
            graph.set_goto(new_block_id, target, arg_vars);
        } else {
            // Preserve the callee block's upstream CFG shape (single
            // goto, can-raise, typed-exception, bool-branch) with
            // renamed values and blocks.  `set_control_flow_metadata`
            // stamps `prevblock` on every link per
            // `flowspace/model.py`.
            let (exitswitch, exits) = remap_control_flow_metadata_var(
                &callee_block.exitswitch,
                &callee_block.exits,
                |var| value_map[var].clone(),
                |b| block_map[&b],
            );
            graph.set_control_flow_metadata(new_block_id, exitswitch, exits);
        }
    }
    // `Inliner.rewire_exceptblock_no_guard`: an edge into the copied
    // exceptblock targets the host `graph.exceptblock` and keeps only
    // etype and evalue. The guarded rewire stays unreachable.
    rewire_copied_exceptblock(graph, &block_map, callee.exceptblock);

    // Connect caller's before-block to callee entry.
    //
    // `Inliner.do_inline` (`backendopt/inline.py`) binds call arguments
    // only through the entry Link: `passon_args` is the call args plus
    // `original_passon_vars`, and `copy_block` starts the copied
    // startblock from `block.inputargs`. Uses inside the callee keep
    // the copied parameter Variables; they are never rewritten onto
    // the caller's argument Variable. Rewriting them (`remap_value_in_graph`)
    // made a FieldRead name a value defined in the before-block, after
    // which `prune_dead_phis` dropped the now-unread startblock
    // inputarg together with that link column (`w_set_items` →
    // `SetStrategyRef::deref` left `v18.imp` undefined).
    //
    // A callee whose parameters live only as `OpKind::Input` ops (empty
    // `startblock.inputargs`) still has to grow those Input results into
    // inputargs so the same Link can bind them. `flowspace/model.py`
    // `checkgraph` requires each use to be a block inputarg or op result.
    let callee_entry_block = &callee.blocks[callee.startblock.0];
    if graph.blocks[callee_entry.0].inputargs.is_empty() {
        let input_vars: Vec<crate::flowspace::model::Variable> = callee_entry_block
            .operations
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::Input { .. }))
            .filter_map(|op| op.result.clone())
            .collect();
        let n = input_vars.len().min(call_args.len());
        for var in input_vars.iter().take(n) {
            graph.blocks[callee_entry.0]
                .inputargs
                .push(value_map[var].clone());
        }
    }

    // Set the before-block's terminator to jump to callee entry.
    // `set_terminator` resyncs `exits`/`exitswitch` to a single Goto
    // link with `prevblock = block_id`, replacing the caller block's
    // original exits (which have already been moved to the merge block
    // above).
    //
    // RPython `get_new_name` keeps Constants on the entry Link
    // (`backendopt/inline.py`).  `call_arg_vars` would drop them and
    // shift later arguments.
    let inputarg_count = graph.blocks[callee_entry.0].inputargs.len();
    let entry_args: Vec<crate::model::LinkArg> =
        call_args.into_iter().take(inputarg_count).collect();
    graph.set_goto_mixed(block_id, callee_entry, entry_args);
    // `rewire_returnblock` does `linkargs = [return] + passon_vars` after
    // `copy_block` has already appended `passon_vars` to every inlined
    // block. Thread the same caller Variables (not per-block copies) once
    // the entry and return links exist, so `set_goto` still sees the
    // one-result merge. `jtransform.join_blocks` runs on this global-SSA
    // graph, before `ssa.SSA_to_SSI`, and renames a joined block's
    // inputarg to the link arg. That arg has to be an inputarg or op
    // result of the source block; otherwise the renamed use has no def
    // (`liveness.compute_liveness` on `w_dict_getitem_str_hashed`'s
    // `hash_`).
    if let Some((merge_id, _)) = merge_block_id.as_ref() {
        let merge_id = *merge_id;
        for var in &passon {
            graph.ensure_variable_at_block(merge_id, var);
        }
    }
    merge_block_id.and_then(|(_, args)| args.into_iter().next())
}

/// Caller values live across the inlined call.
///
/// `unsimplify.split_block` puts each such value on the link into the
/// after-block and skips `last_exception` / `last_exc_value`, which the
/// link itself creates. The call result is not passed through: the merge
/// inputarg replaces it. Values the after-ops produce stay in the merge.
fn caller_passon_vars(
    after_ops: &[SpaceOperation],
    after_exitswitch: &Option<ExitSwitch>,
    after_exits: &[Link],
    call_result: &Option<crate::flowspace::model::Variable>,
) -> Vec<crate::flowspace::model::Variable> {
    let defined_after: HashSet<u64> = after_ops
        .iter()
        .filter_map(|op| op.result.as_ref().map(|var| var.id()))
        .collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut push = |var: &crate::flowspace::model::Variable| {
        if call_result.as_ref().is_some_and(|result| result == var) {
            return;
        }
        if defined_after.contains(&var.id()) {
            return;
        }
        if seen.insert(var.id()) {
            out.push(var.clone());
        }
    };
    for op in after_ops {
        for var in op_variable_refs(&op.kind) {
            push(&var);
        }
    }
    match after_exitswitch {
        Some(ExitSwitch::Value(var)) => push(var),
        Some(ExitSwitch::Fused { args, .. }) => {
            for var in args {
                push(var);
            }
        }
        Some(ExitSwitch::LastException) | None => {}
    }
    for link in after_exits {
        for arg in &link.args {
            let Some(var) = arg.as_variable() else {
                continue;
            };
            let carried_exception = link
                .last_exception
                .as_ref()
                .is_some_and(|exc| exc.as_variable().as_ref() == Some(&var))
                || link
                    .last_exc_value
                    .as_ref()
                    .is_some_and(|exc| exc.as_variable().as_ref() == Some(&var));
            if !carried_exception {
                push(var);
            }
        }
    }
    out
}

/// Retarget edges that enter the copied exceptblock.
///
/// `Inliner.rewire_exceptblock_no_guard` walks those entry links, truncates
/// each to etype and evalue, and points them at the host exceptblock. The
/// copied exceptblock's own outgoing edges stay; once nothing targets it,
/// it is unreachable.
fn rewire_copied_exceptblock(
    graph: &mut FunctionGraph,
    block_map: &HashMap<BlockId, BlockId>,
    callee_except: BlockId,
) {
    let Some(&copied_except) = block_map.get(&callee_except) else {
        return;
    };
    let host_except = graph.exceptblock;
    if copied_except == host_except {
        return;
    }
    for &copied_id in block_map.values() {
        if copied_id == copied_except {
            continue;
        }
        for exit in &mut graph.blocks[copied_id.0].exits {
            if exit.target != copied_except {
                continue;
            }
            exit.args.truncate(2);
            exit.target = host_except;
        }
    }
}

/// Splice one direct call. Returns whether `op_index` was a `Call`.
///
/// A use of the call result outside the call block's after-ops (the
/// monotonic lowering reuses that variable in a later block) is retargeted
/// at the merge inputarg. Framestate already rewrites the call block's
/// exit links inside [`inline_call_site`].
pub(crate) fn splice_direct_call(
    graph: &mut FunctionGraph,
    block_id: BlockId,
    op_index: usize,
    callee: FunctionGraph,
) -> bool {
    let Some(call_op) = graph
        .blocks
        .get(block_id.0)
        .and_then(|block| block.operations.get(op_index))
    else {
        return false;
    };
    let call_result = match &call_op.kind {
        OpKind::Call { .. } => call_op.result.clone(),
        _ => return false,
    };
    let merge = inline_call_site(
        graph,
        InlineSite {
            block_id,
            op_index,
            callee,
        },
    );
    if let (Some(old), Some(new)) = (call_result, merge)
        && old != new
    {
        retarget_var_uses(graph, &old, &new);
    }
    true
}

/// Replace operand uses of `old_var` with `new_var`. Definitions stay:
/// the call that produced `old_var` has already been removed.
fn retarget_var_uses(
    graph: &mut FunctionGraph,
    old_var: &crate::flowspace::model::Variable,
    new_var: &crate::flowspace::model::Variable,
) {
    let remap = |var: &crate::flowspace::model::Variable| {
        if var == old_var {
            new_var.clone()
        } else {
            var.clone()
        }
    };
    for block in &mut graph.blocks {
        for arg in &mut block.inputargs {
            if arg == old_var {
                *arg = new_var.clone();
            }
        }
        for op in &mut block.operations {
            op.kind = remap_op_kind(&op.kind, &remap);
        }
        let (exitswitch, exits) =
            remap_control_flow_metadata_var(&block.exitswitch, &block.exits, remap, |block_id| {
                block_id
            });
        block.exitswitch = exitswitch;
        block.exits = exits;
    }
}

/// Allocate fresh caller-graph Variables for every callee-graph
/// Variable referenced inside the callee.  Returns the callee-Variable
/// → caller-Variable rename map consumed by [`remap_op`] and the
/// cross-graph remap closures in [`inline_call_site`].
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn remap_callee_values(
    graph: &mut FunctionGraph,
    callee: &FunctionGraph,
) -> HashMap<crate::flowspace::model::Variable, crate::flowspace::model::Variable> {
    let mut map: HashMap<crate::flowspace::model::Variable, crate::flowspace::model::Variable> =
        HashMap::new();
    // Collect all Variables used in the callee
    for block in &callee.blocks {
        for var in block.inputargs.iter() {
            map.entry(var.clone())
                .or_insert_with(|| graph.alloc_value_var());
        }
        for op in &block.operations {
            if let Some(result) = op.result.as_ref() {
                map.entry(result.clone())
                    .or_insert_with(|| graph.alloc_value_var());
            }
            for var in op_variable_refs(&op.kind) {
                map.entry(var).or_insert_with(|| graph.alloc_value_var());
            }
        }
        // Upstream `rpython/flowspace/model.py getvariables`
        // walks `link.args` for every exit in addition to ops.  The
        // exitswitch variable is always a block-local value referenced
        // by the raising op / branch condition, so it is already in
        // `op_variable_refs` — but the per-link args must be copied here.
        // Upstream `rpython/translator/backendopt/inline.py:268-269
        // copy_link` also renames `link.last_exception` and
        // `link.last_exc_value`, so the extravars must be present in
        // the value map before `remap_control_flow_metadata` runs.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    map.entry(var.clone())
                        .or_insert_with(|| graph.alloc_value_var());
                }
            }
            if let Some(arg) = &link.last_exception
                && let Some(var) = arg.as_variable()
            {
                map.entry(var.clone())
                    .or_insert_with(|| graph.alloc_value_var());
            }
            if let Some(arg) = &link.last_exc_value
                && let Some(var) = arg.as_variable()
            {
                map.entry(var.clone())
                    .or_insert_with(|| graph.alloc_value_var());
            }
        }
        if let Some(crate::model::ExitSwitch::Value(cond)) = &block.exitswitch {
            map.entry(cond.clone())
                .or_insert_with(|| graph.alloc_value_var());
        }
    }
    map
}

/// Allocate fresh BlockIds for all blocks in the callee graph.
fn remap_callee_blocks(
    graph: &mut FunctionGraph,
    callee: &FunctionGraph,
) -> HashMap<BlockId, BlockId> {
    let mut map = HashMap::new();
    for block in &callee.blocks {
        let new_id = graph.create_block();
        map.insert(block.id, new_id);
    }
    map
}

/// Remap a single Op's operand and result Variables across the
/// inline boundary.  `value_map` carries the callee Variable →
/// caller Variable rename allocated by [`remap_callee_values`];
/// every operand Variable is looked up directly (falling back to
/// the source Variable if it is not in the map — e.g. a Constant
/// or an external-graph reference).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn remap_op(
    op: &SpaceOperation,
    value_map: &HashMap<crate::flowspace::model::Variable, crate::flowspace::model::Variable>,
) -> SpaceOperation {
    let remap_var = |var: &crate::flowspace::model::Variable| {
        value_map.get(var).cloned().unwrap_or_else(|| var.clone())
    };
    let result = op.result.as_ref().map(remap_var);
    let kind = remap_op_kind(&op.kind, &remap_var);
    SpaceOperation { result, kind }
}

pub(crate) fn remap_op_kind(
    kind: &OpKind,
    remap_var: &impl Fn(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
) -> OpKind {
    match kind {
        OpKind::Input {
            name,
            ty,
            class_root,
        } => OpKind::Input {
            name: name.clone(),
            ty: ty.clone(),
            class_root: class_root.clone(),
        },
        OpKind::ConstInt(v) => OpKind::ConstInt(*v),
        OpKind::ConstFnAddr {
            value,
            path,
            symbolic,
        } => OpKind::ConstFnAddr {
            value: *value,
            path: path.clone(),
            symbolic: *symbolic,
        },
        OpKind::RawLoad {
            base,
            offset,
            item_ty,
            itemsize,
            is_item_signed,
        } => OpKind::RawLoad {
            base: remap_var(base),
            offset: remap_var(offset),
            item_ty: item_ty.clone(),
            itemsize: *itemsize,
            is_item_signed: *is_item_signed,
        },
        OpKind::GcLoadIndexed {
            base,
            index,
            scale,
            base_ofs,
            item_ty,
            itemsize,
            is_item_signed,
        } => OpKind::GcLoadIndexed {
            base: remap_var(base),
            index: remap_var(index),
            scale: *scale,
            base_ofs: *base_ofs,
            item_ty: item_ty.clone(),
            itemsize: *itemsize,
            is_item_signed: *is_item_signed,
        },
        OpKind::RawStore {
            base,
            offset,
            value,
            item_ty,
            itemsize,
            is_item_signed,
        } => OpKind::RawStore {
            base: remap_var(base),
            offset: remap_var(offset),
            value: remap_var(value),
            item_ty: item_ty.clone(),
            itemsize: *itemsize,
            is_item_signed: *is_item_signed,
        },
        OpKind::ConstUInt(v) => OpKind::ConstUInt(*v),
        OpKind::ConstInt128(v) => OpKind::ConstInt128(*v),
        OpKind::ConstSingleFloat(v) => OpKind::ConstSingleFloat(*v),
        OpKind::ConstUInt128(v) => OpKind::ConstUInt128(*v),
        OpKind::ConstBool(v) => OpKind::ConstBool(*v),
        OpKind::ConstSymbolic { tag, ty } => OpKind::ConstSymbolic {
            tag: *tag,
            ty: ty.clone(),
        },
        OpKind::ConstFloat(bits) => OpKind::ConstFloat(*bits),
        OpKind::ConstStr(bytes) => OpKind::ConstStr(bytes.clone()),
        OpKind::ConstInternedStr(bytes) => OpKind::ConstInternedStr(bytes.clone()),
        OpKind::ConstRef(obj) => OpKind::ConstRef(obj.clone()),
        OpKind::ConstRefNull => OpKind::ConstRefNull,
        OpKind::ConstNone => OpKind::ConstNone,
        OpKind::ConstRefAddr(addr) => OpKind::ConstRefAddr(*addr),
        OpKind::New { owner } => OpKind::New {
            owner: owner.clone(),
        },
        OpKind::NewWithVtable { owner, vtable } => OpKind::NewWithVtable {
            owner: owner.clone(),
            vtable: *vtable,
        },
        OpKind::RawMalloc {
            owner,
            zero,
            track_allocation,
            add_memory_pressure,
            extra_flags,
            length,
        } => OpKind::RawMalloc {
            owner: owner.clone(),
            zero: *zero,
            track_allocation: *track_allocation,
            add_memory_pressure: *add_memory_pressure,
            extra_flags: extra_flags.clone(),
            length: length.as_ref().map(remap_var),
        },
        OpKind::RawFree { ptr } => OpKind::RawFree {
            ptr: remap_var(ptr),
        },
        OpKind::NewArray {
            length,
            item_ty,
            array_type_id,
        } => OpKind::NewArray {
            length: remap_var(length),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::NewArrayClear {
            length,
            item_ty,
            array_type_id,
        } => OpKind::NewArrayClear {
            length: remap_var(length),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::NewListClear {
            length,
            item_ty,
            array_type_id,
        } => OpKind::NewListClear {
            length: remap_var(length),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        } => OpKind::FieldRead {
            base: remap_var(base),
            field: field.clone(),
            ty: ty.clone(),
            pure: *pure,
        },
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } => OpKind::FieldWrite {
            base: remap_var(base),
            field: field.clone(),
            value: value.map_value(remap_var),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
            array_type_id,
            nolength,
            pure,
        } => OpKind::ArrayRead {
            base: remap_var(base),
            index: remap_var(index),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
            // Preserve the source foldable/immutable flag through the
            // inline clone — never hardcode it (rlist.py:724
            // ll_getitem_foldable_nonneg).
            pure: *pure,
        },
        OpKind::ArrayLen {
            base,
            array_type_id,
            nolength,
        } => OpKind::ArrayLen {
            base: remap_var(base),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
            array_type_id,
            nolength,
        } => OpKind::ArrayWrite {
            base: remap_var(base),
            index: remap_var(index),
            value: value.map_value(remap_var),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldRead {
            base: remap_var(base),
            index: remap_var(index),
            field: field.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldWrite {
            base: remap_var(base),
            index: remap_var(index),
            field: field.clone(),
            value: remap_var(value),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => OpKind::Call {
            target: target.clone(),
            args: args
                .iter()
                .map(|arg| arg.map_value(|var| remap_var(var)))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_var(cond),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_var(cond),
        },
        OpKind::GuardValue { value, kind_char } => OpKind::GuardValue {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        } => OpKind::VtableMethodPtr {
            receiver: remap_var(receiver),
            trait_root: trait_root.clone(),
            method_name: method_name.clone(),
        },
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            family_key,
            result_ty,
        } => OpKind::IndirectCall {
            funcptr: remap_var(funcptr),
            args: args.iter().map(&remap_var).collect(),
            graphs: graphs.clone(),
            family_key: family_key.clone(),
            result_ty: result_ty.clone(),
        },
        OpKind::RecordQuasiImmutField {
            base,
            field,
            mutate_field,
        } => OpKind::RecordQuasiImmutField {
            base: remap_var(base),
            field: field.clone(),
            mutate_field: mutate_field.clone(),
        },
        OpKind::JitForceQuasiImmutable { base, mutate_field } => OpKind::JitForceQuasiImmutable {
            base: base.map_value(remap_var),
            mutate_field: mutate_field.clone(),
        },
        OpKind::VableFieldRead {
            base,
            field_index,
            ty,
        } => OpKind::VableFieldRead {
            base: remap_var(base),
            field_index: *field_index,
            ty: ty.clone(),
        },
        OpKind::VableFieldWrite {
            base,
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            base: remap_var(base),
            field_index: *field_index,
            value: value.map_value(remap_var),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            base,
            array_index,
            elem_index,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayRead {
            base: remap_var(base),
            array_index: *array_index,
            elem_index: remap_var(elem_index),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::VableArrayLen {
            base,
            array_index,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayLen {
            base: remap_var(base),
            array_index: *array_index,
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::VableArrayWrite {
            base,
            array_index,
            elem_index,
            value,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayWrite {
            base: remap_var(base),
            array_index: *array_index,
            elem_index: remap_var(elem_index),
            value: value.map_value(remap_var),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::BinOp {
            op,
            lhs,
            rhs,
            result_ty,
        } => OpKind::BinOp {
            op: op.clone(),
            lhs: remap_var(lhs),
            rhs: remap_var(rhs),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_var(operand),
            result_ty: result_ty.clone(),
        },
        OpKind::VableForce { base } => OpKind::VableForce {
            base: remap_var(base),
        },
        OpKind::Hint { value, kind } => OpKind::Hint {
            value: remap_var(value),
            kind: *kind,
        },
        OpKind::JitDebug { args } => OpKind::JitDebug {
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::RecordKnownResult {
            result_value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::RecordKnownResult {
            result_value: remap_var(result_value),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::AssertGreen { value, kind_char } => OpKind::AssertGreen {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::CurrentTraceLength => OpKind::CurrentTraceLength,
        OpKind::IsConstant { value, kind_char } => OpKind::IsConstant {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::IsVirtual { value, kind_char } => OpKind::IsVirtual {
            value: remap_var(value),
            kind_char: *kind_char,
        },
        OpKind::GuardClass { base } => OpKind::GuardClass {
            base: remap_var(base),
        },
        OpKind::IsInstance {
            obj,
            class_carrier,
            result_ty,
        } => OpKind::IsInstance {
            obj: remap_var(obj),
            class_carrier: remap_var(class_carrier),
            result_ty: result_ty.clone(),
        },
        OpKind::Live => OpKind::Live,
        OpKind::JitMergePoint {
            jitdriver_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            green_classes,
            red_classes,
        } => OpKind::JitMergePoint {
            jitdriver_index: *jitdriver_index,
            greens_i: greens_i.iter().map(&remap_var).collect(),
            greens_r: greens_r.iter().map(&remap_var).collect(),
            greens_f: greens_f.iter().map(&remap_var).collect(),
            reds_i: reds_i.iter().map(&remap_var).collect(),
            reds_r: reds_r.iter().map(&remap_var).collect(),
            reds_f: reds_f.iter().map(&remap_var).collect(),
            green_classes: green_classes.clone(),
            red_classes: red_classes.clone(),
        },
        OpKind::LoopHeader { jitdriver_index } => OpKind::LoopHeader {
            jitdriver_index: *jitdriver_index,
        },
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallElidable {
            funcptr: remap_call_funcptr(funcptr, remap_var),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            indirect_targets,
        } => OpKind::CallResidual {
            funcptr: remap_call_funcptr(funcptr, remap_var),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
            indirect_targets: indirect_targets.clone(),
        },
        OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallMayForce {
            funcptr: remap_call_funcptr(funcptr, remap_var),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            result_kind,
            arg_classes,
        } => OpKind::InlineCall {
            jitcode: jitcode.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
            arg_classes: arg_classes.clone(),
        },
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            result_kind,
            green_classes,
            red_classes,
        } => OpKind::RecursiveCall {
            jd_index: *jd_index,
            greens_i: greens_i.iter().map(&remap_var).collect(),
            greens_r: greens_r.iter().map(&remap_var).collect(),
            greens_f: greens_f.iter().map(&remap_var).collect(),
            reds_i: reds_i.iter().map(&remap_var).collect(),
            reds_r: reds_r.iter().map(&remap_var).collect(),
            reds_f: reds_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
            green_classes: green_classes.clone(),
            red_classes: red_classes.clone(),
        },
        OpKind::ConditionalCall {
            condition,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
        } => OpKind::ConditionalCall {
            condition: remap_var(condition),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
        },
        OpKind::ConditionalCallValue {
            value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::ConditionalCallValue {
            value: remap_var(value),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: args_i.iter().map(&remap_var).collect(),
            args_r: args_r.iter().map(&remap_var).collect(),
            args_f: args_f.iter().map(&remap_var).collect(),
            result_kind: *result_kind,
        },
        OpKind::Abort { kind } => OpKind::Abort { kind: kind.clone() },
        OpKind::NewTuple { args } => OpKind::NewTuple {
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::NewList { args } => OpKind::NewList {
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::GetSlice { args } => OpKind::GetSlice {
            args: args.iter().map(&remap_var).collect(),
        },
        OpKind::LoweredBlackholeOp { opname, args } => OpKind::LoweredBlackholeOp {
            opname: opname.clone(),
            args: args.iter().map(|a| a.map_value(&remap_var)).collect(),
        },
        OpKind::LoadStatic {
            segments,
            ty,
            value,
        } => OpKind::LoadStatic {
            segments: segments.clone(),
            ty: ty.clone(),
            value: value.clone(),
        },
    }
}

/// Variable-identity primary walker — clones every `Variable` operand
/// directly out of [`OpKind`]'s arm fields.
///
/// RPython parity — upstream `SpaceOperation.args` is already a
/// `Vec<Hlvalue>` where each `Hlvalue::Variable` carries the
/// authoritative operand identity (`flowspace/model.py`).  Pyre's
/// `OpKind` already stores `flowspace::model::Variable` per operand,
/// so this walker needs no graph round-trip.
pub fn op_variable_refs(kind: &OpKind) -> Vec<crate::flowspace::model::Variable> {
    let clone_var = |var: &crate::flowspace::model::Variable| var.clone();
    match kind {
        OpKind::RawLoad { base, offset, .. } => vec![clone_var(base), clone_var(offset)],
        OpKind::GcLoadIndexed { base, index, .. } => vec![clone_var(base), clone_var(index)],
        OpKind::RawStore {
            base,
            offset,
            value,
            ..
        } => vec![clone_var(base), clone_var(offset), clone_var(value)],
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstInt128(_)
        | OpKind::ConstSingleFloat(_)
        | OpKind::ConstUInt128(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_)
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::LoadStatic { .. }
        | OpKind::New { .. }
        | OpKind::NewWithVtable { .. } => {
            vec![]
        }
        OpKind::RawMalloc { length, .. } => length.iter().map(clone_var).collect(),
        OpKind::RawFree { ptr } => vec![clone_var(ptr)],
        OpKind::NewTuple { args } => args.iter().map(clone_var).collect(),
        OpKind::NewList { args } => args.iter().map(clone_var).collect(),
        // `new_array_clear(v_length, arraydescr)` — only the length Variable
        // is an SSA operand; the arraydescr is a descriptor, not a value.
        OpKind::NewArray { length, .. } | OpKind::NewArrayClear { length, .. } => {
            vec![clone_var(length)]
        }
        // `newlist_clear(v_length, ...)` — same operand shape: only the
        // length Variable is an SSA operand; the struct/array descrs are
        // descriptors, not values.
        OpKind::NewListClear { length, .. } => vec![clone_var(length)],
        OpKind::GetSlice { args } => args.iter().map(clone_var).collect(),
        OpKind::LoweredBlackholeOp { args, .. } => args
            .iter()
            .filter_map(|a| a.as_variable().map(clone_var))
            .collect(),
        OpKind::VableForce { base } => vec![clone_var(base)],
        OpKind::Hint { value, .. } => vec![clone_var(value)],
        OpKind::JitMergePoint {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut v = Vec::with_capacity(
                greens_i.len()
                    + greens_r.len()
                    + greens_f.len()
                    + reds_i.len()
                    + reds_r.len()
                    + reds_f.len(),
            );
            v.extend(greens_i.iter().map(&clone_var));
            v.extend(greens_r.iter().map(&clone_var));
            v.extend(greens_f.iter().map(&clone_var));
            v.extend(reds_i.iter().map(&clone_var));
            v.extend(reds_r.iter().map(&clone_var));
            v.extend(reds_f.iter().map(&clone_var));
            v
        }
        OpKind::FieldRead { base, .. } => vec![clone_var(base)],
        OpKind::FieldWrite { base, value, .. } => {
            // `setfield_gc` stores either a register operand or an inline
            // constant; only a `Variable` value contributes an SSA
            // reference (a `Const` literal carries no defining op).
            let mut refs = vec![clone_var(base)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::ArrayRead { base, index, .. } => vec![clone_var(base), clone_var(index)],
        OpKind::ArrayLen { base, .. } => vec![clone_var(base)],
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            // `setarrayitem_gc` stores either a register operand or an
            // inline constant; only a `Variable` value contributes an SSA
            // reference (a `Const` literal carries no defining op).
            let mut refs = vec![clone_var(base), clone_var(index)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::InteriorFieldRead { base, index, .. } => vec![clone_var(base), clone_var(index)],
        OpKind::InteriorFieldWrite {
            base, index, value, ..
        } => vec![clone_var(base), clone_var(index), clone_var(value)],
        OpKind::Call { args, .. } => args
            .iter()
            .filter_map(|arg| arg.as_variable().map(clone_var))
            .collect(),
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => vec![clone_var(cond)],
        OpKind::GuardValue { value, .. } => vec![clone_var(value)],
        OpKind::AssertGreen { value, .. }
        | OpKind::IsConstant { value, .. }
        | OpKind::IsVirtual { value, .. } => vec![clone_var(value)],
        OpKind::GuardClass { base } => vec![clone_var(base)],
        OpKind::IsInstance {
            obj, class_carrier, ..
        } => {
            vec![clone_var(obj), clone_var(class_carrier)]
        }
        OpKind::VtableMethodPtr { receiver, .. } => vec![clone_var(receiver)],
        OpKind::IndirectCall { funcptr, args, .. } => {
            let mut v = vec![clone_var(funcptr)];
            v.extend(args.iter().map(&clone_var));
            v
        }
        OpKind::RecordQuasiImmutField { base, .. } => vec![clone_var(base)],
        OpKind::JitForceQuasiImmutable { base, .. } => {
            if let Some(var) = base.as_variable() {
                vec![clone_var(var)]
            } else {
                Vec::new()
            }
        }
        OpKind::JitDebug { args, .. } => args.iter().map(&clone_var).collect(),
        OpKind::VableFieldRead { base, .. } => vec![clone_var(base)],
        OpKind::VableFieldWrite { base, value, .. } => {
            // Only a `Variable` value contributes an SSA reference; a
            // `setfield_vable_i` inline `Const` carries no defining op.
            let mut refs = vec![clone_var(base)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::VableArrayRead {
            base, elem_index, ..
        } => vec![clone_var(base), clone_var(elem_index)],
        OpKind::VableArrayLen { base, .. } => vec![clone_var(base)],
        OpKind::VableArrayWrite {
            base,
            elem_index,
            value,
            ..
        } => {
            let mut refs = vec![clone_var(base), clone_var(elem_index)];
            if let Some(var) = value.as_variable() {
                refs.push(clone_var(var));
            }
            refs
        }
        OpKind::BinOp { lhs, rhs, .. } => vec![clone_var(lhs), clone_var(rhs)],
        OpKind::UnaryOp { operand, .. } => vec![clone_var(operand)],
        OpKind::CallElidable {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallResidual {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        }
        | OpKind::CallMayForce {
            funcptr,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = match funcptr {
                CallFuncPtr::Target(_) => Vec::new(),
                CallFuncPtr::Value(var) => vec![clone_var(var)],
            };
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::InlineCall {
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs: Vec<crate::flowspace::model::Variable> =
                args_i.iter().map(&clone_var).collect();
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::ConditionalCall {
            condition,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = vec![clone_var(condition)];
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::ConditionalCallValue {
            value,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = vec![clone_var(value)];
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::RecordKnownResult {
            result_value,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            let mut refs = vec![clone_var(result_value)];
            refs.extend(args_i.iter().map(&clone_var));
            refs.extend(args_r.iter().map(&clone_var));
            refs.extend(args_f.iter().map(&clone_var));
            refs
        }
        OpKind::RecursiveCall {
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut refs: Vec<crate::flowspace::model::Variable> =
                greens_i.iter().map(&clone_var).collect();
            refs.extend(greens_r.iter().map(&clone_var));
            refs.extend(greens_f.iter().map(&clone_var));
            refs.extend(reds_i.iter().map(&clone_var));
            refs.extend(reds_r.iter().map(&clone_var));
            refs.extend(reds_f.iter().map(&clone_var));
            refs
        }
    }
}

/// `true` iff `kind` is side-effect-free for the existing folding,
/// CSE, and control-flow-shape consumers of this predicate.
///
/// Dead-operation removal uses [`can_remove_op`] instead.  RPython keeps
/// `LLOp.is_pure()` (`lloperation.py`) and `CanRemove`
/// (`simplify.py:411-423`) as distinct predicates.
pub fn is_pure_op(kind: &OpKind) -> bool {
    match kind {
        // `raw_load` reads mutable raw memory and `raw_store` writes it:
        // neither is pure. `gc_load_indexed` reads a mutable GC object.
        OpKind::RawLoad { .. } | OpKind::RawStore { .. } | OpKind::GcLoadIndexed { .. } => false,
        // `new` / `new_with_vtable` / `new_array_clear` heap-allocate fresh
        // objects: NOT pure.  CSE must never coalesce two allocations, or
        // Python `is` object identity would break.
        OpKind::New { .. }
        | OpKind::NewWithVtable { .. }
        | OpKind::RawMalloc { .. }
        | OpKind::RawFree { .. }
        | OpKind::NewArray { .. }
        | OpKind::NewArrayClear { .. }
        | OpKind::NewListClear { .. }
        // `newlist` subclasses `HLOperation` (`operation.py`), not
        // `PureOperation`; its DCE authorization comes from the high-level
        // `simplify.py CanRemove` list alone.
        | OpKind::NewList { .. } => false,
        // `OpKind::ConstInt` / `OpKind::ConstFloat` materialize a
        // `Variable` for a literal in pyre's IR.  There
        // is NO upstream `int_constant` op — RPython's `Constant` is
        // a value class (`model.py` `Constant`), not an
        // operation, so it appears inline in `op.args` rather than
        // as a standalone op in `block.operations`.  Pyre's
        // op-shaped representation is forced by the
        // `Block.operations: Vec<Op>` /
        // `Op.result: Option<Variable>` design: every value the
        // graph produces must be materialised through an op so a
        // backing `Variable` can be minted for it.  Returning `true` here
        // is the dataflow-equivalent of upstream's "Constant args
        // pin nothing" behaviour: the const op is removed by
        // `prune_dead_phis` Step 5 when its result is unread, which
        // mirrors upstream's "dead Constant simply doesn't appear in
        // any live op's args" outcome.
        //
        // `Input` is structurally pure: inputarg-shaped Input ops
        // are protected from `model::prune_dead_phis` Step 5 sweep
        // by their result `Variable` being pinned in `read_vars` (Step
        // 1+3+dependency-routing); naked Input ops (legacy frontend
        // fallback) are removed by Step 5 when their result is
        // dead.  Returned as `true` here for consistency with the
        // dependency-routing classification.
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstInt128(_)
        | OpKind::ConstSingleFloat(_)
        | OpKind::ConstUInt128(_)
        | OpKind::ConstBool(_)
        // `_we_are_jitted` symbolic const — a pure `Constant`, removable
        // when its result is unread (folded to a real const by
        // `jtransform` before assembly).
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_)
        // Pure reads — `getfield(_pure) getarrayitem(_pure)
        // getinteriorfield` in `enum_ops_without_sideeffects`.
        | OpKind::FieldRead { .. }
        | OpKind::ArrayRead { .. }
        | OpKind::ArrayLen { .. }
        | OpKind::InteriorFieldRead { .. }
        // Pure virtualizable reads — no heap mutation.
        | OpKind::VableFieldRead { .. }
        | OpKind::VableArrayRead { .. }
        | OpKind::VableArrayLen { .. }
        // Pure vtable slot read — `cast_pointer + getfield` chain
        // collapsed into one op (see `OpKind::VtableMethodPtr` doc).
        | OpKind::VtableMethodPtr { .. }
        // `newtuple` is `PureOperation` (`operation.py`).
        | OpKind::NewTuple { .. }
        // `getslice` is registered `pure=True` for flowspace folding/CSE
        // (`operation.py:461`), but its possible exception excludes it from
        // dead-op removal; see `can_remove_op`.
        | OpKind::GetSlice { .. }
        // The high-level `isinstance` entry in `simplify.py:411-418`
        // `CanRemove` authorizes DCE.  Its lowering reads subclass-range
        // fields and emits `int_between`; there is no
        // `instance_isinstance` lloperation row.
        | OpKind::IsInstance { .. }
        // `LoadStatic` reads a `static` declaration's compile-time
        // address — equivalent to `LOAD_GLOBAL` → Constant lookup,
        // pure.
        | OpKind::LoadStatic { .. } => true,
        // Per-opname classification for `OpKind::BinOp` mirrors
        // `simplify.CanRemove` and
        // `enum_ops_without_sideeffects()` for binary ops.  Pyre's
        // `OpKind::BinOp` carries the opname as a string field, so
        // the parity-correct classification is opname-keyed rather
        // than enum-blanket — short-circuit `&&` / `||` (Rust
        // logical-and / -or) emit `BinOp{op:"and"/"or"}` which have
        // no flowspace peer and must surface a fail-loud TyperError
        // at the rtyper rather than be silently DCE'd.
        OpKind::BinOp { op, .. } => is_pure_binop_opname(op),
        // Per-opname classification mirrors `enum_ops_without_sideeffects()`'s
        // `LL_OPERATIONS[opname].sideeffects` lookup
        // (`lloperation.enum_ops_without_sideeffects`).
        // Pyre's `OpKind::UnaryOp` carries the opname as a string
        // field, so the parity-correct classification is opname-keyed
        // rather than enum-blanket.
        OpKind::UnaryOp { op, .. } => is_pure_unary_opname(op),
        // Side-effecting writes / calls / guards / markers / aborts.
        // `direct_call`-family ops are routed here even when the
        // callee is elidable: `simplify.transform_dead_op_vars`'s `canremove`
        // split treats `direct_call` as side-effecting (args go
        // straight to `read_vars`), and the same function performs a
        // separate elidable-graph removal that requires `translator`
        // to be supplied (pyre's call site passes `translator=None`,
        // so the removal arm is unreachable).  The post-jtransform
        // `OpKind::CallElidable` shape arrives after `prune_dead_phis`
        // has already run, so its classification here is moot for the
        // current pipeline ordering.
        OpKind::FieldWrite { .. }
        | OpKind::ArrayWrite { .. }
        | OpKind::InteriorFieldWrite { .. }
        | OpKind::VableFieldWrite { .. }
        | OpKind::VableArrayWrite { .. }
        | OpKind::Call { .. }
        | OpKind::IndirectCall { .. }
        | OpKind::CallResidual { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::CallElidable { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. }
        | OpKind::ConditionalCall { .. }
        | OpKind::ConditionalCallValue { .. }
        | OpKind::RecordKnownResult { .. }
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::JitForceQuasiImmutable { .. }
        | OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::GuardClass { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::IsConstant { .. }
        | OpKind::IsVirtual { .. }
        | OpKind::CurrentTraceLength
        | OpKind::JitDebug { .. }
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. }
        | OpKind::Live
        | OpKind::VableForce { .. }
        | OpKind::Hint { .. }
        | OpKind::Abort { .. } => false,

        // `LoweredBlackholeOp` carries register-shaped blackhole insns whose
        // side-effect class is opname-dependent — the same split `op_can_raise`
        // (`call.rs`) and upstream `lloperation.LL_OPERATIONS` (sideeffects /
        // canfold) make.  The read family
        // (`strlen`/`unicodelen`/`strgetitem`/`unicodegetitem`) is
        // pure/removable, as is `int_between` (`canfold`, no heap effect);
        // the alloc/store/copy family
        // (`newstr`/`newunicode`/`strsetitem`/`unicodesetitem`/
        // `copystrcontent`/`copyunicodecontent`) mutates.  Any unlisted opname
        // falls through to side-effecting, so the dead-op sweep never drops a
        // store or an allocation by default.
        OpKind::LoweredBlackholeOp { opname, .. } => matches!(
            opname.as_str(),
            "strlen"
                | "unicodelen"
                | "strgetitem"
                | "unicodegetitem"
                | "int_between"
        ),
    }
}

/// `true` iff `kind` may be removed when its result has no readers.
/// Models RPython `simplify.py CanRemove`, including the default
/// `raising_is_ok=False` used by `enum_ops_without_sideeffects()`.
///
pub fn can_remove_op(kind: &OpKind) -> bool {
    match kind {
        // `newlist` is explicitly listed in `simplify.py:413` even though it
        // is not a flowspace `PureOperation` (`operation.py`).
        OpKind::NewList { .. } => true,
        // `getslice` is absent from `simplify.py CanRemove` and is
        // raising at `lloperation.py:578`, so
        // `enum_ops_without_sideeffects()` does not add it either.
        OpKind::GetSlice { .. } => false,
        // `front::mir`'s function-item define-op stands for a `Constant`
        // (`model.py`); one left unread after a combinator names the
        // function directly goes the way of a dead constant.
        OpKind::Call { target, args, .. }
            if args.is_empty() && crate::model::fn_const_segments(target).is_some() =>
        {
            true
        }
        _ => is_pure_op(kind),
    }
}

/// Whitelist of `OpKind::UnaryOp` opnames that are side-effect-free
/// upstream — direct port of the unary entries in
/// `simplify.CanRemove`
/// + `enum_ops_without_sideeffects()`
///   (`lloperation.enum_ops_without_sideeffects`).
///
/// Any opname not in this list is treated as side-effecting so the
/// dead-op DCE pass does not silently remove it.  Notably absent:
/// `not` — Python's `not` is control flow and `flowspace.operation` does not
/// register it; pyre's `flowspace_adapter` requires the
/// frontend to desugar `!x` away before reaching the rtyper, so DCE
/// must surface a live `not` op to the rtyper rather than silently
/// dropping a dead one.  `front::mir` collapses Rust deref `*x`
/// (no flowspace peer) when it lowers `UnaryOp`/`Deref`, so the
/// frontend never emits it.
/// Whitelist of `OpKind::BinOp` opnames that are side-effect-free
/// upstream — direct port of the binary entries in
/// `simplify.CanRemove` plus
/// `enum_ops_without_sideeffects()`.
///
/// Notable omissions:
///   - `and` / `or` (no trailing underscore): Rust `&&` / `||`
///     short-circuit operators; `flowspace.operation` does not
///     register them as binary operators (they are control flow),
///     and `translator/rtyper/flowspace_adapter.rs` requires
///     the frontend to desugar them before reaching the rtyper.  DCE
///     must keep dead occurrences alive so the rtyper surfaces the
///     fail-loud TyperError instead of silently dropping them.  The
///     trailing-underscore canonical names `and_` / `or_` are PyPy's
///     registered bitwise AND/OR operations and are
///     pure and listed below.
///   - `*_assign` (Rust compound assignments): `inplace_*` upstream;
///     `simplify.py:CanRemove` does not include `inplace_*`, and
///     compound assignments mutate their LHS so DCE shouldn't drop
///     a live-LHS write just because the result vid is unread.
///   - `unknown_binop`: pyre fallback for unsupported ops;
///     fail-loud territory.
///
/// CanRemove entries that do not surface as `OpKind::BinOp` or
/// `OpKind::UnaryOp` (handled elsewhere or shape-incompatible):
///   - `newtuple` / `newlist` / `newdict` — collection constructors,
///     surfaced by frontend lowering passes, not via BinOp/UnaryOp.
///   - `getattr` — variable arity (`getattr(o, name [, default])`),
///     pyre lowers as `OpKind::FieldRead` / Call rather than BinOp.
///   - `get` — three-argument descriptor `__get__`; pyre
///     has no TernaryOp surface.  Drop is the call-DCE's concern.
fn is_pure_binop_opname(opname: &str) -> bool {
    if matches!(
        opname,
        // `simplify.CanRemove` entries registered through
        // `flowspace.operation.add_operator`.
        "add"
        | "sub"
        | "mul"
        | "div"
        | "truediv"
        | "floordiv"
        | "mod"
        | "divmod"
        | "pow"
        | "lshift"
        | "rshift"
        // Canonical PyPy bitwise binops `and_`, `or_`, and `xor`. These are the
        // trailing-underscore forms that surface after
        // `flowspace_adapter.rs`'s `normalize_binop_name` rewrites
        // pyre's `bitand`/`bitor`/`bitxor`; both forms are pure so the
        // DCE whitelist accepts pre- and post-normalize names.
        | "and_"
        | "or_"
        | "xor"
        | "bitand"
        | "bitor"
        | "bitxor"
        // Comparisons from `simplify.CanRemove`.
        | "lt"
        | "le"
        | "eq"
        | "ne"
        | "gt"
        | "ge"
        // Remaining two-argument `CanRemove` entries: identity/type tests,
        // indexing, comparison, coercion, and containment.
        | "is_"
        | "issubtype"
        | "isinstance"
        | "getitem"
        | "cmp"
        | "coerce"
        | "contains"
    ) {
        return true;
    }
    // Post-rtyper / lltype binops — `enum_ops_without_sideeffects()`
    // `lloperation.enum_ops_without_sideeffects` registers `int_add int_sub int_mul
    // int_lt ... uint_add ... float_add ...` with sideeffects=False.
    // Pyre's BinOp arrives here pre-rtyper (frontend names like
    // `add`); the post-rtyper shape is also accepted so a future
    // post-rtyper DCE call site requires no further widening.
    let integer_prefix_match = opname
        .strip_prefix("int_")
        .or_else(|| opname.strip_prefix("uint_"));
    if let Some(suffix) = integer_prefix_match {
        return matches!(
            suffix,
            "add"
                | "sub"
                | "mul"
                | "floordiv"
                | "mod"
                | "lshift"
                | "rshift"
                | "and"
                | "or"
                | "xor"
                | "lt"
                | "le"
                | "eq"
                | "ne"
                | "gt"
                | "ge"
        );
    }
    if let Some(suffix) = opname.strip_prefix("float_") {
        // `lloperation.py:246-261` defines only arithmetic through true
        // division and the six comparisons, and explicitly leaves
        // `float_mod` to `math.fmod`.
        return matches!(
            suffix,
            "add" | "sub" | "mul" | "truediv" | "lt" | "le" | "eq" | "ne" | "gt" | "ge"
        );
    }
    false
}

fn is_pure_unary_opname(opname: &str) -> bool {
    matches!(
        opname,
        // `simplify.py` CanRemove — every entry registered as
        // a 1-arg `add_operator(...)` at `flowspace/operation.py`.
        //   id     operation.py
        //   type   operation.py:447
        //   repr   operation.py
        //   str    operation.py:451
        //   len    operation.py
        //   hash   operation.py
        //   pos    operation.py:465
        //   neg    operation.py:466
        //   bool   operation.py:467
        //   abs    operation.py
        //   hex    operation.py:470
        //   oct    operation.py:471
        //   ord    operation.py
        //   invert operation.py:474
        //   int    operation.py:488
        //   float  operation.py:490
        //   long   operation.py:491
        //   iter   operation.py:582 (Iter HLOperation, arity=1)
        "id"
        | "type"
        | "repr"
        | "str"
        | "len"
        | "hash"
        | "pos"
        | "neg"
        | "bool"
        | "abs"
        | "hex"
        | "oct"
        | "ord"
        | "invert"
        | "int"
        | "float"
        | "long"
        | "iter"
        // `same_as` — `jtransform.py rewrite_op_same_as`
        // explicitly drops the op + aliases the result.
        | "same_as"
        // `cast_pointer cast_ptr_to_int cast_int_to_ptr ...` — pure
        // RPython `enum_ops_without_sideeffects` peers.
        | "cast_pointer"
        | "cast_ptr_to_int"
        | "cast_int_to_ptr"
        | "cast_opaque_ptr"
    )
}

/// Replace all occurrences of `old_var` with `new_var` in a list of ops.
fn remap_value_in_ops(
    ops: &[SpaceOperation],
    old_var: &crate::flowspace::model::Variable,
    new_var: &crate::flowspace::model::Variable,
) -> Vec<SpaceOperation> {
    let remap_var = |var: &crate::flowspace::model::Variable| {
        if var == old_var {
            new_var.clone()
        } else {
            var.clone()
        }
    };
    ops.iter()
        .map(|op| SpaceOperation {
            result: op.result.clone(),
            kind: remap_op_kind(&op.kind, &remap_var),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::CallControl;
    use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
    use crate::parse::CallPath;

    #[test]
    fn dead_op_removal_and_float_purity_use_their_rpython_tables() {
        let newlist = OpKind::NewList { args: vec![] };
        assert!(!is_pure_op(&newlist));
        assert!(can_remove_op(&newlist));

        let getslice = OpKind::GetSlice { args: vec![] };
        assert!(is_pure_op(&getslice));
        assert!(!can_remove_op(&getslice));

        for name in [
            "float_add",
            "float_sub",
            "float_mul",
            "float_truediv",
            "float_lt",
            "float_le",
            "float_eq",
            "float_ne",
            "float_gt",
            "float_ge",
        ] {
            assert!(is_pure_binop_opname(name), "{name} is an lltype float op");
        }
        for name in [
            "float_floordiv",
            "float_mod",
            "float_lshift",
            "float_rshift",
            "float_and",
            "float_or",
            "float_xor",
        ] {
            assert!(
                !is_pure_binop_opname(name),
                "{name} is absent from the lltype float table"
            );
        }
    }

    #[test]
    fn lowered_blackhole_op_purity_is_opname_aware() {
        let pure = |name: &str| {
            is_pure_op(&OpKind::LoweredBlackholeOp {
                opname: name.to_string(),
                args: vec![],
            })
        };
        // Read family: pure/removable when the result is dead.
        for name in [
            "strlen",
            "unicodelen",
            "strgetitem",
            "unicodegetitem",
            "int_between",
        ] {
            assert!(pure(name), "{name} is pure");
        }
        // Alloc / store / copy family: side-effecting, must never be dropped.
        for name in [
            "newstr",
            "newunicode",
            "strsetitem",
            "unicodesetitem",
            "copystrcontent",
            "copyunicodecontent",
        ] {
            assert!(!pure(name), "{name} mutates and must be kept");
        }
    }

    fn make_simple_callee() -> FunctionGraph {
        // Callee: fn callee(base) -> value { ArrayRead(base, const(0)) }
        let mut g = FunctionGraph::new("callee");
        let entry = g.startblock;
        let base_var = g
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let idx_var = g.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let result = g.push_op_var(
            entry,
            OpKind::ArrayRead {
                base: base_var,
                index: idx_var,
                item_ty: ValueType::Ref(None),
                array_type_id: None,
                nolength: false,
                pure: false,
            },
            true,
        );
        g.set_return(entry, result);
        g
    }

    #[test]
    fn inline_single_call() {
        // Caller: fn caller() { v = Call("callee", [base]); Return v }
        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let base_var = caller
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: crate::model::call_args(vec![base_var]),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

        let callee = make_simple_callee();

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["callee"]), callee);
        cc.find_all_graphs_for_tests();

        let count = inline_graph(&mut caller, &cc, 3);
        assert_eq!(count, 1);

        // After inlining: the graph should have ArrayRead from the callee
        let has_array_read = caller
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }));
        assert!(
            has_array_read,
            "inlined graph should contain ArrayRead from callee"
        );

        // Should NOT have the original Call op
        let has_call = caller
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::Call { .. }));
        assert!(!has_call, "Call op should be replaced by inlined body");
    }

    #[test]
    fn splice_binds_call_arg_through_entry_inputarg() {
        // `Inliner.do_inline` keeps callee uses on copied startblock
        // inputargs. Rewriting them onto the caller's argument left
        // the ArrayRead base undefined once `prune_dead_phis` dropped
        // the unread parameter column.
        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let base_var = caller
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: crate::model::call_args(vec![base_var.clone()]),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

        assert!(splice_direct_call(
            &mut caller,
            entry,
            1,
            make_simple_callee()
        ));
        crate::model::prune_dead_phis(&mut caller);

        let mut saw_array_read = false;
        for block in &caller.blocks {
            let defined: std::collections::HashSet<u64> = block
                .inputargs
                .iter()
                .map(|var| var.id())
                .chain(
                    block
                        .operations
                        .iter()
                        .filter_map(|op| op.result.as_ref().map(|var| var.id())),
                )
                .collect();
            for op in &block.operations {
                let OpKind::ArrayRead { base, .. } = &op.kind else {
                    continue;
                };
                saw_array_read = true;
                assert!(
                    defined.contains(&base.id()),
                    "ArrayRead base must be a block inputarg or op result, not the \
                     caller's dominating argument\n{}",
                    caller.dump()
                );
            }
        }
        assert!(saw_array_read, "spliced graph lost the callee ArrayRead");
    }

    #[test]
    fn splice_rewires_exception_exit_to_the_host_exceptblock() {
        use crate::model::{Link, LinkArg};

        let mut callee = FunctionGraph::new("raising");
        let etype = callee.alloc_value_var();
        let evalue = callee.alloc_value_var();
        let extra = callee.alloc_value_var();
        callee.set_control_flow_metadata(
            callee.startblock,
            None,
            vec![Link::new_mixed(
                vec![
                    LinkArg::Value(etype),
                    LinkArg::Value(evalue),
                    LinkArg::Value(extra),
                ],
                callee.exceptblock,
                None,
            )],
        );

        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["raising"]),
                args: crate::model::call_args(Vec::<crate::flowspace::model::Variable>::new()),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);
        assert!(splice_direct_call(&mut caller, entry, 0, callee));

        let host = caller.exceptblock;
        let raising: Vec<_> = caller
            .blocks
            .iter()
            .flat_map(|block| block.exits.iter())
            .filter(|exit| exit.target == host)
            .collect();
        assert!(
            raising.iter().any(|exit| exit.args.len() == 2),
            "exception edge must reach the host exceptblock with etype and evalue\n{}",
            caller.dump()
        );
        assert!(
            caller
                .blocks
                .iter()
                .flat_map(|block| block.exits.iter())
                .all(|exit| exit.args.len() <= 2 || exit.target != host),
            "host exception edge keeps only etype and evalue\n{}",
            caller.dump()
        );
        assert!(
            caller
                .blocks
                .iter()
                .flat_map(|block| block.exits.iter())
                .all(|exit| exit.args.len() != 3),
            "the callee's third exception arg must not survive the splice\n{}",
            caller.dump()
        );
    }

    #[test]
    fn inline_keeps_constant_call_args_on_the_entry_link() {
        use crate::flowspace::model::{ConstValue, Variable};
        use crate::model::LinkArg;

        let mut callee = FunctionGraph::new("const_callee");
        let param = Variable::new();
        callee.blocks[callee.startblock.0].inputargs = vec![param.clone()];
        callee.set_return(callee.startblock, Some(param));

        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let result = caller
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["const_callee"]),
                    args: vec![LinkArg::from(ConstValue::Int(7))],
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        caller.set_return(entry, Some(result));

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["const_callee"]), callee);
        cc.find_all_graphs_for_tests();
        assert_eq!(inline_graph(&mut caller, &cc, 1), 1);

        let entry_link = caller.blocks[caller.startblock.0]
            .exits
            .first()
            .expect("caller jumps to inlined entry");
        assert_eq!(entry_link.args.len(), 1);
        assert!(
            matches!(
                &entry_link.args[0],
                LinkArg::Const(c) if c.value == ConstValue::Int(7)
            ),
            "constant positional arg must stay on the entry Link"
        );
    }

    #[test]
    fn inline_preserves_residual_calls() {
        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["unknown_fn"]),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

        let cc = CallControl::new(); // empty — no graphs registered
        let count = inline_graph(&mut caller, &cc, 3);
        assert_eq!(count, 0);

        // Call should still be there
        let has_call = caller
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::Call { .. }));
        assert!(has_call, "residual Call should be preserved");
    }

    #[test]
    fn inline_two_levels() {
        // inner: fn inner(x) -> ArrayRead(x, 0)
        let inner = make_simple_callee();

        // outer: fn outer(base) -> Call("callee", [base])
        let mut outer = FunctionGraph::new("outer");
        let entry = outer.startblock;
        let base_var = outer
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = outer.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: crate::model::call_args(vec![base_var]),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        outer.set_return(entry, result);

        // caller: fn caller(x) -> Call("outer", [x])
        let mut caller = FunctionGraph::new("caller");
        let centry = caller.startblock;
        let x_var = caller
            .push_op_var(
                centry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            centry,
            OpKind::Call {
                target: CallTarget::function_path(["outer"]),
                args: crate::model::call_args(vec![x_var]),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(centry, result);

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["callee"]), inner);
        cc.register_function_graph(CallPath::from_segments(["outer"]), outer);
        cc.find_all_graphs_for_tests();

        let count = inline_graph(&mut caller, &cc, 3);
        assert!(count >= 2, "should inline at least 2 levels, got {count}");

        // After 2-level inlining: should have ArrayRead, no Call ops
        let has_array_read = caller
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(&op.kind, OpKind::ArrayRead { .. }));
        assert!(
            has_array_read,
            "2-level inlined graph should have ArrayRead"
        );
    }

    /// Post-inline regression: every block whose terminator is a
    /// control-flow op (Goto/Branch) must carry matching `Block.exits`
    /// metadata, and every resulting Link must stamp `prevblock` with
    /// the block it exits.  RPython `flowspace/model.py` keeps
    /// `exitswitch`/`exits` as the single CFG source of truth, so pyre
    /// must not let the inline rewrite produce terminator/exits drift
    /// or `prevblock = None` links.
    #[test]
    fn inline_preserves_exits_and_prevblock_invariants() {
        let callee = make_simple_callee();

        let mut caller = FunctionGraph::new("caller");
        let entry = caller.startblock;
        let base_var = caller
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "base".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = caller.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: crate::model::call_args(vec![base_var]),
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        caller.set_return(entry, result);

        let mut cc = CallControl::new();
        cc.register_function_graph(CallPath::from_segments(["callee"]), callee);
        cc.find_all_graphs_for_tests();

        let count = inline_graph(&mut caller, &cc, 3);
        assert!(count >= 1, "callee should inline at least once");

        for block in &caller.blocks {
            // Upstream `flowspace/model.py Block` — a closed block (one
            // with `exitswitch.is_some()` or at least one exit) always
            // carries both: the exitswitch names the branch condition,
            // the exits hold every outgoing `Link`.  An unclosed block
            // (startblock before its first closeblock, or a freshly
            // created merge block with no afterblock payload) has
            // `exits=()` and `exitswitch=None`, matching the initial
            // state in `FunctionGraph.__init__`.
            assert!(
                block.is_closed() || !block.exits.is_empty() || block.exitswitch.is_none(),
                "block {:?} exits/exitswitch out of sync after inline",
                block.id
            );
            for link in &block.exits {
                assert_eq!(
                    link.prevblock,
                    Some(block.id),
                    "link in block {:?} targeting {:?} missing prevblock stamp",
                    block.id,
                    link.target
                );
            }
        }
    }
}
