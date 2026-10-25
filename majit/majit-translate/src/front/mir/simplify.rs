//! Graph simplification (`rpython/translator/simplify.py`).

use super::*;

/// Restamp `ArrayWrite` / `ArrayRead` `item_ty` to `Ref` for arrays that
/// receive a `GcRef(p as usize)` word.
pub(in crate::front::mir) fn retarget_gcref_array_items(graph: &mut FunctionGraph) {
    let mut pointer_words: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();
    for block in &graph.blocks {
        for op in &block.operations {
            if crate::model::cast_instance_root(&op.kind) == Some("GCREF")
                && let Some(result) = &op.result
            {
                pointer_words.insert(result.id());
            }
        }
    }
    // Each block's incoming links, counted once: the walk below reads only
    // blocks with exactly one, and it rewrites no link.
    let mut incoming: rustc_hash::FxHashMap<_, (usize, &crate::model::Link)> =
        rustc_hash::FxHashMap::default();
    for link in graph.blocks.iter().flat_map(|pred| pred.exits.iter()) {
        incoming.entry(link.target).or_insert((0, link)).0 += 1;
    }
    let mut grew = true;
    while grew {
        grew = false;
        for block in &graph.blocks {
            let Some(&(1, link)) = incoming.get(&block.id) else {
                continue;
            };
            for (input, arg) in block.inputargs.iter().zip(link.args.iter()) {
                let Some(src) = arg.as_variable() else {
                    continue;
                };
                if pointer_words.contains(&src.id()) && pointer_words.insert(input.id()) {
                    grew = true;
                }
            }
        }
    }
    let is_word = |var: &crate::flowspace::model::Variable| pointer_words.contains(&var.id());
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            if let OpKind::ArrayWrite { value, item_ty, .. } = &mut op.kind
                && value.as_variable().is_some_and(is_word)
            {
                *item_ty = ValueType::Ref(None);
            }
        }
    }
    // A second pass marks array bases that received a pointer word, then
    // their reads. Bases are variables, not the words themselves.
    let mut pointer_arrays: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::ArrayWrite { base, value, .. } = &op.kind
                && value.as_variable().is_some_and(is_word)
            {
                pointer_arrays.insert(base.id());
            }
        }
    }
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            if let OpKind::ArrayRead { base, item_ty, .. } = &mut op.kind
                && pointer_arrays.contains(&base.id())
            {
                *item_ty = ValueType::Ref(None);
            }
        }
    }
}

pub(in crate::front::mir) fn simplify_lowered_graph_expanding(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    sweep_dead_vars: bool,
    guard_words: &std::collections::HashSet<Variable>,
    expand_inline_struct: &dyn Fn(&str, &str) -> Option<crate::model::InlineStructExpansion>,
) -> Result<(), LowerError> {
    // `GcRef(p as usize)` is a GCREF word (`ConstRefNull` for `GcRef::NULL`).
    // The element place type still peels to `usize`, so an `ArrayWrite`
    // emitted before its block link existed kept `item_ty: Unsigned`.
    // Restamp once the producer walk can see the cast.
    retarget_gcref_array_items(graph);
    // Collapse the operation-less blocks the MIR lowering leaves behind.
    // This is not the no-op `eliminate_empty_blocks`' own doc describes —
    // that claim is about the AST front, whose tree-recursive `Expr::Match`
    // / `Expr::If` lowering emits the post-collapse CFG directly.  This is
    // the MIR front, and Charon MIR is full of empty blocks: measured over
    // the LLBC corpus, the call rewires a link on 998 of 1255 graphs.
    crate::model::eliminate_empty_blocks(graph);
    // `eliminate_empty_blocks` (simplify.py) rewires each incoming
    // link past an operation-less block and leaves the bypassed block
    // orphaned, still carrying its exits.  Upstream every later consumer
    // walks the graph through `iterblocks()` (flowspace/model.py), so
    // the orphan and its stale links are invisible; the model-layer
    // passes below scan `graph.blocks` directly, so the stale links stay
    // in view — `prune_dead_phis` then trims a shared reachable target's
    // dead inputargs without trimming the orphan's link
    // (transform_dead_op_vars walks reachable blocks only), the
    // arity-desynced link misaligns `remove_duplicate_inputargs`'s
    // positional phi columns, and the legacy annotator's link fold
    // (`follow_link`, annrpython.py binding propagation) unions the
    // shifted kinds into live phi inputargs — surfacing as an
    // assembler Int/Ref register-kind mismatch.  Empty the orphans
    // eagerly so the direct-scan passes see the same surface
    // `iterblocks()` exposes upstream.
    crate::model::clear_unreachable_blocks(graph);
    // `reload_top_root` / `try_gc_current_object_address` are gctransform
    // identities pyre spells in source. Peel a `cast(x, GCREF)` that only
    // feeds them so the annotator sees `x` (`peel_recast_to_source`).
    crate::front::result_exc::retarget_gc_erasure_identity_args(graph);
    // Rust's `ptr::write(raw as *mut T, T { .. })` is the source spelling of
    // RPython's already-allocated object followed by one setfield per member.
    // Lower that aggregate copy before allocation fusion so generic
    // `core::ptr::write<T>` never becomes one shared FunctionDesc whose `T`
    // argument would merge unrelated object classes.
    let mut dirty =
        crate::model::lower_struct_ptr_writes_with(graph, struct_field_attrs, expand_inline_struct)
            > 0;
    // Lower the allocation idiom `malloc[_typed](W_Object { ... })` to a
    // native `NewWithVtable` + payload stores before the dead-aggregate sweep,
    // which then reclaims the orphaned construct-on-stack ctor and header
    // field writes.
    dirty |=
        crate::model::fuse_boxing_alloc_with(graph, struct_field_attrs, expand_inline_struct) > 0;
    // After boxing fusion has read the header store, a leftover by-value
    // write of an inlined struct is one setfield per inner leaf.
    dirty |= crate::model::lower_inlined_struct_field_writes(graph, expand_inline_struct)
        .map_err(LowerError::Unsupported)?
        > 0;
    // Reclaim boxing-cluster remnants (fused header ctors/casts, and a
    // `vec![…]` box whose consumer became a `newlist`) using dependency-flow
    // liveness (`transform_dead_op_vars`, simplify.py) with the
    // malloc-removal exemption (`remove_simple_mallocs`, malloc.py).  Unlike
    // `remove_dead_aggregates` below, it treats a `Link.arg` as a dependency
    // of the target inputarg, not an unconditional read, so it reaches a box
    // threaded across a block boundary.  Runs unconditionally: a `vec!`-only
    // graph has `fused == 0`, and the pass is conservative (fresh-alloc
    // producers + scoped store exemption) so it is a no-op on graphs with no
    // such remnants.  A removal leaves dangling threaded inputargs / link args
    // the `prune_dead_phis` below reclaims.
    dirty |= crate::model::prune_dead_boxing_remnants(graph) > 0;
    // Drop dead aggregate constructions (malloc + field stores whose
    // result is never read) before the dead-op sweep — `prune_dead_phis`
    // keeps them because a `FieldWrite` is side-effecting, so its `base`
    // pins the aggregate (`remove_simple_mallocs`, malloc.py).
    dirty |= crate::model::remove_dead_aggregates(graph) > 0;
    // Bring the raise into the shape `remove_assertion_errors` matches: the
    // MIR front builds the exception in a block of its own, where upstream's
    // flow space links straight to `exceptblock`.
    dirty |= crate::model::retarget_assert_raise_blocks(graph) > 0;
    dirty |= crate::model::remove_assertion_errors(graph) > 0;
    // Constant-condition arms (`if WITHPREBUILTINT { … }` with the
    // config const folded by `const_eval_global`) collapse to the
    // taken link and the dead arm is emptied — the registry lift
    // (`translate_op`) walks blocks by index, so a disconnected arm
    // reading an unliftable static (`SMALL_INTS`) would otherwise
    // still fail the whole graph.  `constant_fold_graph` link
    // folding (backendopt/constfold.py).
    if crate::model::fold_constant_exitswitch(graph) > 0 {
        crate::model::clear_unreachable_blocks(graph);
        dirty = true;
    }
    // `transform_dead_op_vars` is the first entry in `all_passes`
    // (simplify.py) and `simplify_graph` (:1080-1086) applies every
    // pass to every graph — there is no "only if something was removed"
    // condition upstream.  The gate that used to stand here was written
    // when the front made a single-predecessor block's inputargs the
    // predecessor's own Variables, which left the sweep unable to tell a
    // read in the predecessor from a read in the target; it removed
    // little, so running it only after another pass had already changed
    // something looked free.  Now that the entry framestate is copied
    // (`flowcontext.py`) the sweep does real work on graphs no other
    // pass touches — `_check_stack_index` threads the virtualizable array
    // into a successor that never reads it, and nothing else in this
    // sequence reports a removal for that graph.
    if sweep_dead_vars || dirty {
        // `ssa.py` `SSA_to_SSI` `record_used_var(block.exitswitch)` /
        // `block.renamevariables({v: w})` rewrites the use onto the
        // family member before `transform_dead_op_vars` runs. An
        // exitswitch that still names the producer while a copy sits
        // in `inputargs` looks unused to prune and is dropped; the
        // later thread then sees no family member.
        crate::model::thread_undefined_op_operands(graph);
        crate::model::prune_dead_phis(graph);
    }
    // Collapse again, now that every dead-code sweep above has run.
    //
    // Upstream orders `all_passes` (simplify.py, applied in list
    // order by `simplify_graph`, :1080-1086) as `transform_dead_op_vars`
    // *then* `eliminate_empty_blocks`, so empty-block elimination observes
    // the dead-op sweep's output.  The head call above runs before all of
    // it, which inverts that.  A block emptied by one of the sweeps —
    // `prune_dead_boxing_remnants` dropping the unit-`()` constructor from
    // a return-forwarding block is the shape seen in practice — is
    // otherwise never rewired past, and the values riding the link into it
    // survive to the end of lowering as dead link args.
    //
    // Upstream has exactly *one* `eliminate_empty_blocks` in `all_passes`,
    // so two calls is not upstream's shape.  It is the minimum that holds
    // upstream's ordering property here, because pyre has dead-code passes
    // with no upstream analogue (`fuse_boxing_alloc`,
    // `prune_dead_boxing_remnants`, `remove_dead_aggregates`) on *both*
    // sides of the collapse: the head call is load-bearing for the MIR
    // front's own empty blocks, and one call cannot occupy both positions.
    crate::model::eliminate_empty_blocks(graph);
    // Mandatory, not redundant.  `eliminate_empty_blocks` rewires the
    // incoming link past the block and leaves it orphaned, still carrying
    // its inputargs; the model-layer passes scan `graph.blocks` directly
    // rather than through upstream's `iterblocks()`, so those inputargs
    // stay in view.  Measured counterexample: with the `clear` omitted,
    // `set_locals_w`'s orphaned block kept the virtualizable array in its
    // inputargs and `prune_dead_phis` did not launder it — the prune
    // trimmed that orphan from 8 inputargs to 4 and the array was still
    // among them.  Ordering the prune first does not remove the need.
    crate::model::clear_unreachable_blocks(graph);
    // Re-thread boxing-cluster operands the dead-var sweeps above stripped out
    // of the `NewWithVtable`-chain blocks' inputargs.  Runs after those sweeps
    // so no later dead-var pass can remove the threaded inputarg, restoring the
    // adapter's per-block operand invariant for cross-block boxing clusters
    // (e.g. `w_int_new`, whose `intval` payload and
    // `__cast_instance_intrinsic` return chain span the blocks split by the
    // `get_instantiate` / `gc_interp::enabled` calls).
    crate::model::thread_undefined_op_operands(graph);
    // Named-struct aggregates that boxing fusion did not consume become
    // their fields. Only the final simplify runs this — the pre-pass
    // still has consumer rewrites (`range_iter`, slice-index) that match
    // the constructor.
    if sweep_dead_vars {
        scalar_replace_named_struct_aggregates(graph, struct_field_attrs);
        crate::model::lower_inlined_struct_field_writes(graph, expand_inline_struct)
            .map_err(LowerError::Unsupported)?;
    }
    // `get().0 as *mut T` is emitted as `cast_int_to_ptr` while the
    // loaded word is still a reference field. After empty-block
    // elimination the cast's operand is that load (through block
    // arguments). Fold it back so the reference is what the cast returns.
    fold_int_to_ptr_of_ref_load(graph, guard_words);
    refuse_inlined_struct_field_writes(graph, expand_inline_struct)?;
    Ok(())
}

/// A `FieldWrite` whose field type is an inlined struct has no field
/// descr (`heaptracker.py all_fielddescrs` recursion). The expand pass
/// above must have replaced every one; a survivor is a front-end gap.
pub(in crate::front::mir) fn refuse_inlined_struct_field_writes(
    graph: &FunctionGraph,
    expand_inline_struct: &dyn Fn(&str, &str) -> Option<crate::model::InlineStructExpansion>,
) -> Result<(), LowerError> {
    let malloc_args = crate::model::typed_malloc_arg_vars(graph);
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::FieldWrite { base, field, .. } = &op.kind else {
                continue;
            };
            if malloc_args.contains(&base.id()) {
                continue;
            }
            let Some(owner) = field.owner_root.as_deref() else {
                continue;
            };
            if expand_inline_struct(owner, &field.name).is_some() {
                return Err(LowerError::Unsupported(format!(
                    "by-value store of inlined struct field {owner}.{} was not lowered to leaf stores",
                    field.name
                )));
            }
        }
    }
    Ok(())
}

/// Drop `cast_int_to_ptr` of a reference-typed field load. The place type
/// of the cast is an integer, but the value was loaded as a reference, so
/// the integer bank does not hold it.
pub(in crate::front::mir) fn fold_int_to_ptr_of_ref_load(
    graph: &mut FunctionGraph,
    guard_words: &std::collections::HashSet<Variable>,
) {
    let mut aliases: Vec<(Variable, Variable)> = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(result) = op.result.clone() else {
                continue;
            };
            let Some(arg) = int_to_ptr_operand(op) else {
                continue;
            };
            if ref_load_root(graph, &arg, guard_words) {
                aliases.push((result, arg));
            }
        }
    }
    let mut folded = false;
    for (result, arg) in &aliases {
        if forward_identity(graph, result, arg) {
            folded = true;
        }
    }
    if folded {
        crate::model::eliminate_empty_blocks(graph);
        crate::model::clear_unreachable_blocks(graph);
    }
}

pub(in crate::front::mir) fn int_to_ptr_operand(op: &SpaceOperation) -> Option<Variable> {
    match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            result_ty: ValueType::Ref(_),
        } if segments
            .last()
            .is_some_and(|leaf| leaf == "cast_int_to_ptr") =>
        {
            args.first().and_then(LinkArg::as_variable).cloned()
        }
        OpKind::UnaryOp {
            op,
            operand,
            result_ty: ValueType::Ref(_),
        } if op == "cast_int_to_ptr" => Some(operand.clone()),
        _ => None,
    }
}

pub(in crate::front::mir) fn ref_load_root(
    graph: &FunctionGraph,
    var: &Variable,
    guard_words: &std::collections::HashSet<Variable>,
) -> bool {
    let Some((block_id, idx)) = resolve_to_producer_op(graph, var) else {
        return false;
    };
    let Some(op) = graph
        .blocks
        .iter()
        .find(|block| block.id == block_id)
        .and_then(|block| block.operations.get(idx))
    else {
        return false;
    };
    matches!(
        op.kind,
        OpKind::FieldRead {
            ty: ValueType::Ref(_),
            ..
        }
    ) && op
        .result
        .as_ref()
        .is_some_and(|result| guard_words.contains(result))
}

/// Whether `target` is `lltype::malloc` / `malloc_typed` / the managed and
/// stable flavors — the boxing-cluster allocator `fuse_boxing_alloc` keys on.
pub(in crate::front::mir) fn call_target_is_gc_malloc(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    let [.., parent, leaf] = segments.as_slice() else {
        return false;
    };
    parent == "lltype"
        && matches!(
            leaf.as_str(),
            "malloc" | "malloc_typed" | "malloc_typed_managed" | "malloc_typed_stable"
        )
}
