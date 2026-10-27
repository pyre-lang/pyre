//! What the virtualizable array may and may not do, checked against the
//! shipped interpreter LLBC rather than a hand-built graph.
//!
//! The codewriter's virtualizable protocol requires the array to be
//! consumed by an array operation in its defining block: `vable_array_vars`
//! is rebuilt per block, so anything that carries the array elsewhere —
//! a call argument, a link argument — reaches `_check_no_vable_array` and
//! aborts the build with "a virtualizable array is passed around".

mod common;

use common::{interpreter_llbc, lower_context_for};
use majit_translate::flowspace::model::Variable;
use majit_translate::front::mir::{LowerContext, lower_fun_decl};
use majit_translate::model::{CallTarget, FunctionGraph, LinkArg, OpKind};

/// Load the shipped interpreter LLBC once and share it across every test.
/// `Llbc` is read-only after `load`, so a single parse behind
/// `common::interpreter_llbc` is sufficient. `None` means the artefact is
/// absent, which degrades the tests to a skip rather than a failure on a
/// tree that has not run the extraction.
fn interpreter_context() -> Option<&'static LowerContext<'static>> {
    Some(lower_context_for(interpreter_llbc()?))
}

/// Lower `pyframe::<Impl>::<leaf>` out of the shipped interpreter LLBC.
fn lower_pyframe_method(leaf: &str) -> Option<FunctionGraph> {
    let llbc = interpreter_llbc()?;
    let suffix = format!("::{leaf}");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with(&suffix))
        .unwrap_or_else(|| panic!("{leaf} present in the shipped LLBC"));
    Some(
        lower_fun_decl(interpreter_context()?, fd)
            .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}")),
    )
}

/// Every graph in the LLBC whose name path ends with `::<leaf>`.  Five
/// distinct functions in `pyframe` render as `pyframe::<Impl>::new`, so a
/// test that wants one of them has to pick it out by shape.
fn lower_all_named(leaf: &str) -> Option<Vec<FunctionGraph>> {
    let llbc = interpreter_llbc()?;
    let context = interpreter_context()?;
    let suffix = format!("::{leaf}");
    Some(
        llbc.iter_local_fns()
            .filter(|fd| fd.item_meta.name_path().ends_with(&suffix))
            .filter_map(|fd| lower_fun_decl(context, fd).ok())
            .collect(),
    )
}

/// The `locals_cells_stack_w` read and the block it happens in.
fn virtualizable_array_read(graph: &FunctionGraph) -> (Variable, usize) {
    let mut found = None;
    for (index, block) in graph.blocks.iter().enumerate() {
        for op in &block.operations {
            if let OpKind::FieldRead { field, .. } = &op.kind
                && field.name == "locals_cells_stack_w"
            {
                found = Some((op.result.clone().expect("the read binds a result"), index));
            }
        }
    }
    found.expect("a read of the virtualizable array")
}

/// Every `Variable` carrying `seed`'s value, followed across link arguments
/// into the target block's `inputargs`.
///
/// `flowcontext.py newstate = state.copy()` freshens a variable at every
/// block boundary, so a value produced in one block and used after a branch
/// is a different `Variable` at the use site.  A search that compares
/// identity stops at the first branch without this closure.
fn link_aliases(graph: &FunctionGraph, seed: &Variable) -> Vec<Variable> {
    let mut carried = vec![seed.clone()];
    let mut changed = true;
    while changed {
        changed = false;
        for block in &graph.blocks {
            for link in &block.exits {
                for (position, arg) in link.args.iter().enumerate() {
                    let LinkArg::Value(value) = arg else {
                        continue;
                    };
                    if !carried.contains(value) {
                        continue;
                    }
                    let Some(target) = graph.block(link.target).inputargs.get(position) else {
                        continue;
                    };
                    if !carried.contains(target) {
                        carried.push(target.clone());
                        changed = true;
                    }
                }
            }
        }
    }
    carried
}

/// `PyFrame::_check_stack_index` is `index >= self.stack_base() && index <
/// locals_w!(self).len()` — a read of the virtualizable array whose only
/// consumer is `len`.
#[test]
fn vable_array_len_lowers_to_arraylen_not_a_call() {
    let Some(graph) = lower_pyframe_method("_check_stack_index") else {
        return;
    };
    let (array_var, defining_block) = virtualizable_array_read(&graph);

    let mut len_block = None;
    for (index, block) in graph.blocks.iter().enumerate() {
        for op in &block.operations {
            match &op.kind {
                OpKind::ArrayLen { base, .. } if *base == array_var => {
                    len_block = Some(index);
                }
                // The `__len` retargeting `is_container_len` gives the
                // non-virtualizable length-prefixed containers.
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments == &["__len".to_string()] && args.iter().any(|a| a == &array_var) => {
                    panic!(
                        "the virtualizable array reaches a `__len` call as an argument, \
                         which trips `_check_no_vable_array` via `handle_residual_call`"
                    );
                }
                _ => {}
            }
        }
    }

    assert_eq!(
        len_block,
        Some(defining_block),
        "`ArrayLen` must consume the array in the block that reads it"
    );
}

/// The array must not leave the block that reads it.
///
/// `_check_no_vable_array` (jtransform.py) rejects a graph whose
/// `Link.args` carry a virtualizable array, because the target block's
/// `vable_array_vars` is rebuilt per block and cannot follow the array
/// across the edge.  Lowering the `len` to `ArrayLen` is not on its own
/// enough: the array can still ride a **dead** link arg out of its
/// defining block, and it did — `_check_stack_index` threaded it twice
/// into a successor whose only operation never touches it.
///
/// `prune_dead_phis` (`transform_dead_op_vars`) is what removes such a
/// link arg, and it can only do so when the target's inputargs are
/// Variables distinct from the link args feeding them
/// (`flowcontext.py newstate = state.copy()`).  This asserts the
/// outcome rather than the mechanism, so it stays meaningful if the
/// freshening moves.
#[test]
fn vable_array_never_rides_a_link_out_of_its_defining_block() {
    let Some(graph) = lower_pyframe_method("_check_stack_index") else {
        return;
    };
    let (array_var, defining_block) = virtualizable_array_read(&graph);

    for (index, block) in graph.blocks.iter().enumerate() {
        for (exit_index, link) in block.exits.iter().enumerate() {
            let position = link
                .args
                .iter()
                .position(|arg| matches!(arg, LinkArg::Value(v) if *v == array_var));
            assert!(
                position.is_none(),
                "the virtualizable array escapes block {index} on exit {exit_index} \
                 (arg {}) to block {:?}; `_check_no_vable_array` rejects that link",
                position.unwrap(),
                link.target,
            );
        }
    }

    for (index, block) in graph.blocks.iter().enumerate() {
        assert!(
            !block.inputargs.contains(&array_var),
            "the virtualizable array is an inputarg of block {index}, so it was \
             threaded in from elsewhere rather than read in block {defining_block}",
        );
    }
}

/// `FrameLocalsRoot::new` registers the frame's array **slot** as a GC
/// root:
///
/// ```ignore
/// let slot = addr_of_mut!((*frame_ptr).locals_cells_stack_w) as *mut *mut u8;
/// let registered = try_gc_add_root(slot);
/// ```
///
/// The front models a reference as an alias of its referent, so that
/// `&raw mut` arrives as a `FieldRead` of the virtualizable array field
/// and the address is handed to a call.  It must be marked
/// `taken_by_address` — otherwise the codewriter records the array in
/// `vable_array_vars`, **drops** the read, and `try_gc_add_root` is left
/// with an undefined operand (with `_check_no_vable_array` wired, it
/// aborts the build instead, which is how this was found).
///
/// Marking it does not make the aliasing right: the getfield still yields
/// the array's value where the source asked for the slot's address, the
/// same as every other place-address in the corpus.  It stops the
/// virtualizable path from turning that into a dropped operand.
#[test]
fn address_of_the_vable_array_slot_is_marked_not_a_read() {
    let Some(graphs) = lower_all_named("new") else {
        return;
    };

    let mut checked = 0;
    for graph in &graphs {
        for block in &graph.blocks {
            for op in &block.operations {
                let OpKind::FieldRead { field, .. } = &op.kind else {
                    continue;
                };
                if field.name != "locals_cells_stack_w" {
                    continue;
                }
                let Some(result) = op.result.as_ref() else {
                    continue;
                };
                // Only the read whose consumer is the root registration.  The
                // registration sits behind a short-circuit test of the frame's
                // mobility, so the call is in a later block and the slot
                // reaches it under a freshened `Variable`.
                let carried = link_aliases(graph, result);
                let feeds_add_root = graph.blocks.iter().any(|b| {
                    b.operations.iter().any(|o| {
                        matches!(
                            &o.kind,
                            OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, args, .. }
                                if segments.last().is_some_and(|s| s == "try_gc_add_root")
                                    && args.iter().any(|a| carried.contains(a))
                        )
                    })
                });
                if !feeds_add_root {
                    continue;
                }
                checked += 1;
                assert!(
                    field.taken_by_address,
                    "the `addr_of_mut!` projection feeding `try_gc_add_root` in {:?} \
                     is recorded as a plain read of the virtualizable array",
                    graph.name,
                );
                assert!(
                    field.suppresses_virtualizable(),
                    "an address-of projection must suppress the virtualizable lowering",
                );
            }
        }
    }
    // Residual `FrameLocalsRoot` helpers compute the slot only inside
    // `dont_look_inside`, so look-inside graphs have `checked == 0`.
    // A look-inside graph that still has the old shape must keep
    // `taken_by_address` (asserted above).
    let _ = checked;
}

/// Every `locals_w!` read in `fast2locals` is consumed by an array operation
/// in the block that reads it.
///
/// This is the other half of what the virtualizable protocol needs, and it
/// is checked separately because the two fail independently: an unmarked
/// read that escapes its block trips `_check_no_vable_array`, and a read
/// consumed in place but marked `taken_by_address` silently lowers to
/// `getarrayitem_gc`.  `fast2locals` is the graph a policy that admitted
/// loops would open first (`pyframe.py` decorates it `@jit.unroll_safe`),
/// so it is where both have to hold.
#[test]
fn every_vable_array_read_in_fast2locals_is_consumed_in_its_block() {
    let Some(graph) = lower_pyframe_method("fast2locals") else {
        return;
    };

    let mut checked = 0;
    for (index, block) in graph.blocks.iter().enumerate() {
        let reads: Vec<Variable> = block
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "locals_cells_stack_w" => {
                    op.result.clone()
                }
                _ => None,
            })
            .collect();
        for array_var in reads {
            checked += 1;
            let consumed = block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::ArrayRead { base, .. }
                        | OpKind::ArrayWrite { base, .. }
                        | OpKind::ArrayLen { base, .. }
                        if *base == array_var
                )
            });
            assert!(
                consumed,
                "the virtualizable array read in block {index} of {:?} has no array \
                 operation consuming it there",
                graph.name,
            );
        }
    }
    assert!(
        checked > 0,
        "no read of the virtualizable array in `fast2locals`"
    );
}

/// REGRESSION GUARD (#24): `frame_locals_proxy_snapshot` carries the
/// `unroll_safe` hint, and this pins the read shape that lets it.
///
/// The hint is the arming switch, not a consumer: it lets the codewriter
/// walk a graph with a backedge, and only then does the virtualizable
/// protocol decide whether each `locals_w!` read becomes a
/// `VableArrayRead` or a plain `getarrayitem_gc` — the latter being a
/// STALE read of a heap array synchronised only at entry/exit and guard
/// failure, not a slower equivalent.  Two preconditions decide it, and
/// they fail independently: the read must not be `taken_by_address`, and
/// it must be consumed by an array operation in its own block.
///
/// The reads once lived in an `insert_slot` **closure**, which charon
/// renders as its own function, so a probe that lowered only the outer
/// method would have found no reads at all and passed vacuously.  The
/// walk still covers every function whose path mentions the method,
/// closures included, so reintroducing one cannot hide from it.
#[test]
fn report_vable_array_shape_of_frame_locals_proxy_snapshot() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let mut graphs: Vec<FunctionGraph> = Vec::new();
    let context = interpreter_context().expect("interpreter LLBC is present");
    for fd in llbc.iter_local_fns() {
        let path = fd.item_meta.name_path();
        if path.contains("frame_locals_proxy_snapshot") {
            let g = lower_fun_decl(context, fd).unwrap_or_else(|e| panic!("lower {path}: {e:?}"));
            eprintln!("[vable-probe] lowered {path}");
            graphs.push(g);
        }
    }
    assert!(
        !graphs.is_empty(),
        "no `frame_locals_proxy_snapshot` graph in the shipped LLBC"
    );

    let mut reads = 0;
    let mut by_address = 0;
    let mut unconsumed = 0;
    let mut escapes: Vec<String> = Vec::new();

    for graph in &graphs {
        for (index, block) in graph.blocks.iter().enumerate() {
            for op in &block.operations {
                let OpKind::FieldRead { field, .. } = &op.kind else {
                    continue;
                };
                if field.name != "locals_cells_stack_w" {
                    continue;
                }
                let Some(result) = op.result.as_ref() else {
                    continue;
                };
                reads += 1;
                if field.suppresses_virtualizable() {
                    by_address += 1;
                    eprintln!(
                        "[vable-probe] {:?} block {index}: read is taken_by_address \
                         => lowers to a PLAIN getarrayitem_gc (stale read)",
                        graph.name
                    );
                }
                let consumed = block.operations.iter().any(|o| {
                    matches!(
                        &o.kind,
                        OpKind::ArrayRead { base, .. }
                            | OpKind::ArrayWrite { base, .. }
                            | OpKind::ArrayLen { base, .. }
                            if base == result
                    )
                });
                if !consumed {
                    unconsumed += 1;
                    eprintln!(
                        "[vable-probe] {:?} block {index}: read NOT consumed by an \
                         array op in its own block",
                        graph.name
                    );
                }
                // The escape routes `check_no_vable_array` names.
                for (bi, b) in graph.blocks.iter().enumerate() {
                    if b.inputargs.contains(result) {
                        escapes.push(format!("{:?}: inputarg of block {bi}", graph.name));
                    }
                    for (ei, link) in b.exits.iter().enumerate() {
                        if link
                            .args
                            .iter()
                            .any(|a| matches!(a, LinkArg::Value(v) if v == result))
                        {
                            escapes.push(format!(
                                "{:?}: link argument, block {bi} exit {ei}",
                                graph.name
                            ));
                        }
                    }
                    for o in &b.operations {
                        if let OpKind::Call { args, .. } = &o.kind
                            && args.iter().any(|a| a == result)
                        {
                            escapes.push(format!("{:?}: call argument in block {bi}", graph.name));
                        }
                    }
                }
            }
        }
    }

    eprintln!(
        "[vable-probe] SUMMARY graphs={} reads={reads} taken_by_address={by_address} \
         unconsumed={unconsumed} escapes={}",
        graphs.len(),
        escapes.len()
    );
    for e in &escapes {
        eprintln!("[vable-probe]   escape: {e}");
    }
    assert!(
        reads > 0,
        "no `locals_cells_stack_w` read found in any \
         `frame_locals_proxy_snapshot` graph — probe is inert"
    );
    // The escape list was collected and printed but never asserted on, so a
    // read that rode a link out of its defining block would print and pass —
    // and that is precisely what `_check_no_vable_array`'s `link argument`
    // route refuses.
    assert!(
        escapes.is_empty(),
        "`frame_locals_proxy_snapshot`'s virtualizable-array read escapes its \
         defining block: {escapes:?}"
    );

    // Pin the measured shape rather than only reporting it.  Both reads now
    // pair: the body is a plain loop, so each `locals_w!(self)` copies the
    // raw pointer out of the field into a temp and borrows `&(*temp)` —
    // deref-last, which is not an address-of.  While the body was a closure
    // the capture path was truncated at that raw-pointer deref, so the
    // closure captured the field itself and the borrow read as
    // `&(*self).locals_cells_stack_w` — field-last, a genuine address-of,
    // correctly suppressed.  That is what kept the hint off this body, and
    // why `fast2locals`, which never had a closure, could always carry it.
    //
    // If this assertion fails, the body has regressed: check whether a
    // closure was reintroduced over `locals_cells_stack_w`.
    assert_eq!(
        (by_address, unconsumed),
        (0, 0),
        "`frame_locals_proxy_snapshot`'s virtualizable-array read changed shape"
    );
}
