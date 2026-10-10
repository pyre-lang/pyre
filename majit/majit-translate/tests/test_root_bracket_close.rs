//! A lowered root bracket closes, in the crate that owns the guard and in a
//! crate that only imports it.
//!
//! The `Drop` of a `RootScope` is the bracket's `pop_roots`: it rewinds the
//! shadow stack to the length the matching `push_roots` captured. A lowering
//! that forwards past that terminator leaves the bracket open, and the slots
//! it pinned stay reachable for the life of the thread — every later
//! collection walks them, so a loop that opens one per iteration turns a
//! constant root set into a growing one.
//!
//! The close is spelled as a call taking the guard by reference rather than as
//! two field reads and a truncate, because a crate that imports `RootScope`
//! sees an opaque stub with no fields. Reading the fields would confine the
//! close to `pyre-object`, which is why `pyre_interpreter` is checked here and
//! not just `pyre_object`.
//!
//! Two things make the close conditional on more than the `Drop` arm existing,
//! and each has a fixture below. The guard has to reach the dropping block:
//! liveness counts a `Drop` place as a use only because the close reads it, and
//! without that the binding survives only when the dropping block happens to be
//! lowered right after the defining one. And the guard has to be one this body
//! still owns: a drop the artefact stamps `Conditional` runs under an
//! initialisation flag it does not carry, so a guard the body moves out of
//! keeps its bracket open rather than rewinding a stack its new owner holds.
//!
//! A third case owes no close at all. `erased_root_bracket_guards` names the
//! brackets the lowering takes out of the jitcode entirely: those open nothing
//! and pin nothing, so there is no shadow stack to rewind. Every assertion here
//! is stated over the brackets that survive that pass, which is why the
//! fixtures below are bodies whose brackets it keeps.

mod common;

use common::{
    INTERPRETER_LLBC, MODULE_LLBC, OBJECT_LLBC, interpreter_llbc, lower_context_for, module_llbc,
    object_llbc,
};
use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{PlaceKind, SwitchTargets, TermKind, TyRef, Unstructured};
use majit_translate::CallPath;
use majit_translate::HostStaticAddrs;
use majit_translate::call::CallControl;
use majit_translate::codewriter::jtransform::{GraphTransformConfig, Transformer};
use majit_translate::front::mir::{
    LowerContext, erased_root_bracket_guards, lower_fun_decl, lower_fun_decl_with_static_addrs,
};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};

fn fun_decl_named<'a>(llbc: &'a Llbc, leaf: &str) -> &'a majit_charon_reader::FunDecl {
    let suffix = format!("::{leaf}");
    llbc.iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with(&suffix))
        .unwrap_or_else(|| panic!("{leaf} present in the shipped LLBC"))
}

fn lower_fun(llbc: &Llbc, context: &LowerContext<'_>, leaf: &str) -> FunctionGraph {
    lower_fun_decl(context, fun_decl_named(llbc, leaf))
        .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"))
}

fn lower_named(llbc: &'static Llbc, leaf: &str) -> FunctionGraph {
    lower_fun(llbc, lower_context_for(llbc), leaf)
}

/// Same class-static bucket the production driver stamps
/// (`jit_fnaddr.rs jit_static_pytype_addrs`). `fuse_boxing_alloc` needs
/// a constant `ob_type` address to emit `NewWithVtable`; an empty
/// `HostStaticAddrs.pytypes` leaves `&LIST_TYPE` as a residual call.
fn lower_named_with_pytypes(
    llbc: &'static Llbc,
    leaf: &str,
    pytypes: &[(&str, i64)],
) -> FunctionGraph {
    lower_fun_decl_with_static_addrs(
        lower_context_for(llbc),
        fun_decl_named(llbc, leaf),
        HostStaticAddrs {
            pytypes,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"))
}

/// Count calls whose path ends with `leaf`, over every block of the graph.
fn calls_to(graph: &FunctionGraph, leaf: &str) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().is_some_and(|s| s == leaf),
            _ => false,
        })
        .count()
}

fn assert_bracket_closes(llbc: &'static Llbc, leaf: &str) {
    let graph = lower_named(llbc, leaf);
    let opened = calls_to(&graph, "push_roots");
    let closed = calls_to(&graph, "root_scope_close");
    assert!(
        opened > 0,
        "{leaf} is the fixture for a lowered root bracket, but it opens none"
    );
    assert!(
        closed > 0,
        "{leaf} opens {opened} root bracket(s) and closes none: every slot the \
         bracket pins stays reachable for the life of the thread"
    );
}

#[test]
fn bracket_closes_for_each_caller_shape() {
    let cases: &[(&str, fn() -> Option<&'static Llbc>, &str)] = &[
        ("owns_guard", object_llbc, "w_tuple_items_copy_as_vec"),
        (
            "imports_guard",
            interpreter_llbc,
            "call_function_impl_result",
        ),
        // `do_call` n>1 is `pin_roots(args_w)` plus `get(args_slot + i)`:
        // leftover-slice erasure takes that bracket out. `do_call_fargs`
        // loops `pin_root` of each argument, so the depth is not a
        // compile-time offset and the imported-crate close stays.
        ("cffi", module_llbc, "do_call_fargs"),
    ];
    for (name, load, leaf) in cases {
        let Some(llbc) = load() else {
            continue;
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_bracket_closes(llbc, leaf);
        }));
        if let Err(payload) = result {
            let msg = if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else {
                "assertion failed".to_string()
            };
            panic!("case {name}: {msg}");
        }
    }
}

/// A body whose guard is dropped several blocks away from where it is bound.
/// The binding reaches that block only through the drop block's `inputargs`,
/// which liveness supplies only because a `Drop` place counts as a use.
///
/// Stated over every body that has such a drop rather than against named
/// fixtures: the shapes the erasure keeps are not the ones it keeps tomorrow,
/// and a fixture list that goes empty asserts nothing while still passing.
#[test]
fn bracket_closes_when_the_drop_is_not_adjacent_to_the_binding() {
    let Some(llbc) = object_llbc() else { return };
    let mut bodies = 0usize;
    let context = lower_context_for(llbc);
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let erased = erased_root_bracket_guards(llbc, fd, &body);
        let moved = moved_out_locals(&body);
        let opens = opener_blocks(llbc, &body);
        let distant = guard_drop_sites(llbc, &body).any(|(bb, local)| {
            !moved.contains(&local)
                && !erased.contains(&(local as usize))
                && opens.get(&local).is_some_and(|open| *open != bb)
        });
        if !distant {
            continue;
        }
        bodies += 1;
        let Ok(graph) = lower_fun_decl(&context, fd) else {
            continue;
        };
        assert!(
            reachable_closes(&graph) > 0,
            "{} drops a surviving guard in a block that does not bind it and              lowers no close",
            fd.item_meta.name_path()
        );
    }
    assert!(
        bodies > 10,
        "only {bodies} bodies drop a surviving guard away from its binding; the          fixture population is too small to prove anything"
    );
}

/// `local -> block` for each guard a `push_roots` call binds in this body.
fn opener_blocks(llbc: &Llbc, body: &Unstructured) -> std::collections::HashMap<u64, usize> {
    let mut out = std::collections::HashMap::new();
    for (i, bb) in body.body.iter().enumerate() {
        let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
            continue;
        };
        let PlaceKind::Local(local) = call.dest.kind else {
            continue;
        };
        if ty_is_root_scope(llbc, &call.dest.ty) {
            out.insert(local, i);
        }
    }
    out
}

/// A bracket left open must belong to a guard the body moved out of, or to one
/// the erasure took out of the jitcode.
///
/// From the move onward the bracket is the new owner's, and rewinding it at
/// the moved-from local's `Drop` would truncate a shadow stack that owner is
/// still using — `DictOperationGuard::new` pins before it moves, so those pins
/// are what the rewind would drop. Charon stamps such a drop `Conditional`:
/// the destructor runs only if the place still holds a value, and the flag
/// that decides it is not in the artefact. "This body never moves the guard"
/// is the only proof of definite initialisation available, so a guard without
/// it keeps its bracket open.
///
/// An erased guard is the other way a drop lowers no close, and it is not a
/// retention: the bracket publishes nothing, so there is no slot to rewind.
///
/// Stated over every body that drops a guard rather than against named
/// fixtures: any other function that stops closing is the regression this
/// guards against, whatever it is called.
#[test]
fn only_a_moved_out_or_erased_guard_keeps_its_bracket_open() {
    let Some(llbc) = object_llbc() else { return };
    let mut dropping = 0usize;
    let context = lower_context_for(llbc);
    let mut left_open: Vec<String> = Vec::new();
    let mut moved_out = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        if guard_drop_sites(llbc, &body).next().is_none() {
            continue;
        }
        dropping += 1;
        let Ok(graph) = lower_fun_decl(&context, fd) else {
            continue;
        };
        if calls_to(&graph, "root_scope_close") > 0 {
            continue;
        }
        let name = fd.item_meta.name_path().to_string();
        let moved = moved_out_locals(&body);
        let erased = erased_root_bracket_guards(llbc, fd, &body);
        let excused = guard_drop_sites(llbc, &body)
            .any(|(_, local)| moved.contains(&local) || erased.contains(&(local as usize)));
        assert!(
            excused,
            "{name} drops a root-bracket guard, lowers no close for it, and neither \
             moves the guard out nor has it erased: every slot the bracket pins \
             stays reachable for the life of the thread"
        );
        if guard_drop_sites(llbc, &body).any(|(_, local)| moved.contains(&local)) {
            moved_out += 1;
        }
        left_open.push(name);
    }
    assert!(
        dropping > 100,
        "only {dropping} bodies drop a guard; the artefact looks wrong, so a pass \
         here would prove nothing"
    );
    assert!(
        moved_out > 0,
        "no function left a bracket open by moving its guard out, so that case has \
         no live fixture and this test asserted nothing about it"
    );
}

/// The locals this body moves out of.
fn moved_out_locals(body: &Unstructured) -> std::collections::HashSet<u64> {
    let mut moved = std::collections::HashSet::new();
    let mut stack: Vec<&serde_json::Value> = Vec::new();
    for bb in &body.body {
        stack.extend(bb.statements.iter().map(|st| st.kind_value()));
        stack.push(bb.terminator.kind_value());
    }
    while let Some(node) = stack.pop() {
        match node {
            serde_json::Value::Object(map) => {
                if let Some(local) = map
                    .get("Move")
                    .and_then(|place| place.get("kind"))
                    .and_then(|kind| kind.get("Local"))
                    .and_then(serde_json::Value::as_u64)
                {
                    moved.insert(local);
                }
                stack.extend(map.values());
            }
            serde_json::Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    moved
}

/// Whether `ty` resolves to the root-bracket guard's own ADT.
fn ty_is_root_scope(llbc: &Llbc, ty: &TyRef) -> bool {
    let TyRef::Dedup { id } = ty else {
        return false;
    };
    llbc.dedup_to_adt_def_id(*id)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .is_some_and(|decl| decl.item_meta.name_path().ends_with("gc_roots::RootScope"))
}

/// Most of the brackets that survive the erasure must actually close.
///
/// The per-fixture tests above pin named shapes; this pins the aggregate, so a
/// rewrite that starts bypassing the block a close sits in shows up as a
/// coverage drop rather than as silence. A rewrite that redirects a block's
/// exits deletes whatever the bypassed chain carried, and the close is not
/// exempt — measured, three bodies here lose one that way.
///
/// Both artefacts are counted. The erasure answers most of `pyre-object`'s
/// brackets, so that crate alone no longer carries a population large enough
/// for this to prove anything.
#[test]
fn nearly_every_dropped_bracket_closes() {
    // Derive metadata once per program and share each context across workers.
    let programs: Vec<_> = [object_llbc(), interpreter_llbc()]
        .into_iter()
        .flatten()
        .map(|llbc| (llbc, lower_context_for(llbc)))
        .collect();
    let mut work: Vec<(
        &LowerContext<'_>,
        &majit_charon_reader::ullbc::FunDecl,
        usize,
    )> = Vec::new();
    for (llbc, context) in &programs {
        for fd in llbc.iter_local_fns() {
            let Some(body) = fd.unstructured() else {
                continue;
            };
            let want = owed_closes(llbc, fd, &body);
            if want == 0 {
                continue;
            }
            work.push((context, fd, want));
        }
    }
    let bodies = work.len();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(work.len().max(1));
    let chunk = work.len().div_ceil(threads).max(1);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = work
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    let mut closed = 0usize;
                    let mut short = Vec::new();
                    for &(context, fd, want) in slice {
                        let Ok(graph) = lower_fun_decl(context, fd) else {
                            continue;
                        };
                        if reachable_closes(&graph) >= want {
                            closed += 1;
                        } else {
                            short.push(fd.item_meta.name_path().to_string());
                        }
                    }
                    (closed, short)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("census worker"))
            .collect::<Vec<_>>()
    });
    let closed: usize = results.iter().map(|(c, _)| *c).sum();
    let short: Vec<String> = results.into_iter().flat_map(|(_, s)| s).collect();
    assert!(
        bodies > 100,
        "only {bodies} bodies drop a surviving guard; the artefact looks wrong, so \
         a pass here would prove nothing"
    );
    assert!(
        closed * 100 >= bodies * 95,
        "only {closed} of {bodies} guard-dropping bodies close every bracket they \
         drop; short: {short:?}"
    );
}

/// The closes a body owes: one per drop of an unmoved, unerased guard, counting
/// only the drops the front lowers. A cleanup-only drop is not one of them —
/// the front does not follow `on_unwind`.
fn owed_closes(
    llbc: &Llbc,
    fd: &majit_charon_reader::ullbc::FunDecl,
    body: &Unstructured,
) -> usize {
    let moved = moved_out_locals(body);
    let erased = erased_root_bracket_guards(llbc, fd, body);
    let live = reachable_without_unwind(llbc, body);
    guard_drop_sites(llbc, body)
        .filter(|(bb, local)| {
            live[*bb] && !moved.contains(local) && !erased.contains(&(*local as usize))
        })
        .count()
}

/// `(block, local)` for every drop of a root-bracket guard, moved or not.
fn guard_drop_sites<'a>(
    llbc: &'a Llbc,
    body: &'a Unstructured,
) -> impl Iterator<Item = (usize, u64)> + 'a {
    body.body.iter().enumerate().filter_map(move |(i, bb)| {
        let Ok(TermKind::Drop { place, .. }) = bb.term(llbc) else {
            return None;
        };
        let PlaceKind::Local(local) = place.kind else {
            return None;
        };
        ty_is_root_scope(llbc, &place.ty).then_some((i, local))
    })
}

/// MIR blocks reachable from the entry without taking an unwind edge.
fn reachable_without_unwind(llbc: &Llbc, body: &Unstructured) -> Vec<bool> {
    let mut seen = vec![false; body.body.len()];
    let mut stack = vec![0usize];
    while let Some(b) = stack.pop() {
        if b >= seen.len() || seen[b] {
            continue;
        }
        seen[b] = true;
        let Ok(term) = body.body[b].term(llbc) else {
            continue;
        };
        match term {
            TermKind::Goto { target }
            | TermKind::Call { target, .. }
            | TermKind::Assert { target, .. }
            | TermKind::Drop { target, .. } => stack.push(target as usize),
            TermKind::Switch { targets, .. } => match targets {
                SwitchTargets::If(a, b) => stack.extend([a as usize, b as usize]),
                SwitchTargets::SwitchInt(_, arms, default) => {
                    stack.extend(arms.into_iter().map(|(_, t)| t as usize));
                    stack.push(default as usize);
                }
            },
            _ => {}
        }
    }
    seen
}

/// Closes in blocks the lowered graph can still reach. A close in a block a
/// rewrite orphaned is one the function no longer runs.
fn reachable_closes(graph: &FunctionGraph) -> usize {
    let mut seen = vec![false; graph.blocks.len()];
    let mut stack = vec![graph.startblock.0];
    let mut found = 0usize;
    while let Some(b) = stack.pop() {
        if b >= graph.blocks.len() || seen[b] {
            continue;
        }
        seen[b] = true;
        found += graph.blocks[b]
            .operations
            .iter()
            .filter(|op| {
                matches!(&op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().is_some_and(|s| s == "root_scope_close"))
            })
            .count();
        stack.extend(graph.blocks[b].exits.iter().map(|e| e.target.0));
    }
    found
}

/// A bracket whose slots are all constant offsets from the body's own depth
/// leaves no shadow-stack call in the lowered graph:
/// `w_range_iter_one_arg_new` opens a scope around its allocation.
#[test]
fn a_constant_offset_bracket_is_scalar_replaced() {
    let Some(llbc) = object_llbc() else { return };
    let graph = lower_named(llbc, "w_range_iter_one_arg_new");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "normalize_roots",
        "shadow_stack_get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "w_range_iter_one_arg_new still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// `ll_listslice_inner` opens, pins, and closes on every return so a
/// BINARY_SLICE caller can erase its own bracket and walk this body.
#[test]
fn ll_listslice_inner_is_depth_neutral() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
    let names = majit_translate::front::mir::discover_depth_neutral_fns(llbc);
    assert!(
        names
            .iter()
            .any(|n| n.ends_with("::ll_listslice_inner") || n.ends_with("ll_listslice_inner")),
        "ll_listslice_inner must be depth-neutral, got {names:?}"
    );
}

/// The inner list-copy body is scalar-replaced: the walk records the
/// strategy copy, not residual `push_roots`.
#[test]
fn ll_listslice_inner_bracket_is_scalar_replaced() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
    let graph = lower_named(llbc, "ll_listslice_inner");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "normalize_roots",
        "shadow_stack_get",
        "pin_root",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "ll_listslice_inner still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// `funccall_flat_from_args` pins 2..=6 slots per `match nargs` arm, so
/// the join after that match is `AtMerge`. The erase pass models that
/// join and takes `push_roots` / `pin_roots` / `get` out of the jitcode.
#[test]
fn funccall_flat_from_args_bracket_is_scalar_replaced() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
    let fd = fun_decl_named(llbc, "funccall_flat_from_args");
    let body = fd
        .unstructured()
        .expect("funccall_flat_from_args has a body");
    let erased = erased_root_bracket_guards(llbc, fd, &body);
    assert!(
        !erased.is_empty(),
        "funccall_flat_from_args must erase its RootScope; guards={erased:?}"
    );
    let graph = lower_named(llbc, "funccall_flat_from_args");
    for leaf in [
        "push_roots",
        "pin_roots",
        "pin_root",
        "get",
        "shadow_stack_get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "funccall_flat_from_args still calls {leaf} after the nargs-pin merge was erased"
        );
    }
}

/// `eval_slice_index` opens, pins, and closes on every return. The
/// path-sensitive walk classifies that body as depth-neutral.
#[test]
fn eval_slice_index_is_depth_neutral() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
    let names = majit_translate::front::mir::discover_depth_neutral_fns(llbc);
    assert!(
        names
            .iter()
            .any(|n| n.ends_with("::eval_slice_index") || n.ends_with("eval_slice_index")),
        "eval_slice_index must be depth-neutral, got {names:?}"
    );
    let graph = lower_named(llbc, "eval_slice_index");
    for leaf in [
        "push_roots",
        "pin_root",
        "shadow_stack_get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "eval_slice_index still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// `zip_two_tuple_next` pins across `tuple_iter_descr_next`. Observes
/// inside its Open is erasable, so the helper jitcode has no root bracket.
#[test]
fn zip_two_tuple_next_bracket_is_scalar_replaced() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
    let graph = lower_named(llbc, "zip_two_tuple_next");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "normalize_roots",
        "shadow_stack_get",
        "pin_root",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "zip_two_tuple_next still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// `binary_slice_values_inner` opens a RootScope, batch-publishes the three
/// operands, and calls `eval_slice_index`. A stack-sensitive callee used
/// to refuse the whole rewrite; the walker then aborted on `push_roots`.
#[test]
fn binary_slice_values_inner_bracket_is_scalar_replaced() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let graph = lower_named(llbc, "binary_slice_values_inner");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "normalize_roots",
        "shadow_stack_get",
        "pin_root",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "binary_slice_values_inner still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// The linked translation publishes `pyre-object`'s sensitive set onto
/// the interpreter artefact. Inner still erases: its object callees that
/// open-and-close a `RootScope` are depth-neutral, not a name exemption.
#[test]
fn binary_slice_values_inner_erases_with_linked_object_sensitive_set() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load object llbc");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load interpreter llbc");
    let context = LowerContext::new(&interpreter);
    let (mut sensitive, mut leaves, mut params, mut ret_idx, mut observes) =
        majit_translate::front::mir::discover_stack_fn_effects(&object);
    object.register_stack_sensitive_fns(sensitive.iter().cloned());
    object.register_stack_observes_fns(observes.iter().cloned());
    object.register_stack_leaves_above_fns(leaves.iter().cloned());
    object.register_stack_param_slots_fns(params.iter().cloned());
    object.register_stack_returns_index_fns(ret_idx.iter().cloned());
    let mut neutral = majit_translate::front::mir::discover_depth_neutral_fns(&object);
    object.register_stack_depth_neutral_fns(neutral.iter().cloned());
    interpreter.register_stack_sensitive_fns(sensitive.iter().cloned());
    interpreter.register_stack_observes_fns(observes.iter().cloned());
    interpreter.register_stack_depth_neutral_fns(neutral.iter().cloned());
    interpreter.register_stack_leaves_above_fns(leaves.iter().cloned());
    interpreter.register_stack_param_slots_fns(params.iter().cloned());
    interpreter.register_stack_returns_index_fns(ret_idx.iter().cloned());
    let (sens, more_leaves, more_params, more_ret, more_observes) =
        majit_translate::front::mir::discover_stack_fn_effects(&interpreter);
    sensitive.extend(sens);
    observes.extend(more_observes);
    leaves.extend(more_leaves);
    params.extend(more_params);
    ret_idx.extend(more_ret);
    interpreter.register_stack_sensitive_fns(sensitive);
    interpreter.register_stack_observes_fns(observes);
    interpreter.register_stack_leaves_above_fns(leaves);
    interpreter.register_stack_param_slots_fns(params);
    interpreter.register_stack_returns_index_fns(ret_idx);
    neutral.extend(majit_translate::front::mir::discover_depth_neutral_fns(
        &interpreter,
    ));
    interpreter.register_stack_depth_neutral_fns(neutral);
    interpreter.mark_stack_sensitive_fns_complete();
    let graph = lower_fun(&interpreter, &context, "binary_slice_values_inner");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "normalize_roots",
        "shadow_stack_get",
        "pin_root",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "binary_slice_values_inner still calls {leaf} after the linked object set was published"
        );
    }
}

/// `|| shadow_stack_get(len)` after `pin_root` is a RootSlotGetter
/// closure. Publishing other crates' root-stack effects lets
/// `analyze_root_brackets` erase the bracket; the callee is the closure
/// `call`, not a `gc_roots` leaf, so lowering answers it from `get_sites`.
#[test]
fn a_captured_len_getter_lowers_once_root_stack_effects_are_published() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load object llbc");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load interpreter llbc");
    let mut crates = vec![object.crate_name().to_string()];
    let mut touching = majit_translate::front::mir::harvest_root_stack_touching_paths(&object);
    crates.push(interpreter.crate_name().to_string());
    touching.extend(majit_translate::front::mir::harvest_root_stack_touching_paths(&interpreter));
    interpreter.set_root_stack_effects(crates, touching);
    let context = LowerContext::new(&interpreter);
    let graph = lower_fun(&interpreter, &context, "type_get_annotations");
    assert!(
        !graph.blocks.is_empty(),
        "a captured-len getter must lower once root-stack effects are published"
    );
}

/// `ll_listslice_new_int_list` pins source and dest around `newlist` +
/// `ll_arraycopy`. The native RootScope is scalar-replaced so a
/// BINARY_SLICE sub-walk records the copy, not residual `pin_root`.
#[test]
fn ll_listslice_new_int_list_bracket_is_scalar_replaced() {
    let Some(llbc) = object_llbc() else { return };
    let graph = lower_named(llbc, "ll_listslice_new_int_list");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "normalize_roots",
        "shadow_stack_get",
        "pin_root",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "ll_listslice_new_int_list still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// `format_int_decimal` roots the fill, the sign and the digits across the
/// collecting `mallocstr` helpers. The FORMAT_WITH_SPEC int fold descends its
/// body, so the bracket must be scalar-replaced rather than left as residual
/// shadow-stack calls the sub-walk cannot trace.
#[test]
fn format_int_decimal_bracket_is_scalar_replaced() {
    let Some(llbc) = object_llbc() else { return };
    let graph = lower_named(llbc, "format_int_decimal");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "publish_roots",
        "pin_roots",
        "shadow_stack_get",
        "pin_root",
        "base",
        "get",
        "set",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "format_int_decimal still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// BUILD_SET inlines `set_add_value` (`flatten.rs
/// inline_call_targets::SET_ADD`). A bracket left as residual shadow-stack
/// calls makes the walker residualize the whole body through its fnaddr.
#[test]
fn set_add_value_bracket_is_scalar_replaced() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let graph = lower_named(llbc, "set_add_value");
    for leaf in [
        "push_roots",
        "shadow_stack_len",
        "pin_roots",
        "shadow_stack_get",
        "pin_root",
        "base",
        "get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "set_add_value still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// `int_ll_newlist` is `@oopspec("newlist(length)")`. After jtransform the
/// call in `ll_listslice_new_int_list` is `new_array` of
/// `LIST_INT_ITEMS_ARRAY`, not a residual Call.
#[test]
fn ll_listslice_new_int_list_newlist_becomes_new_array() {
    let Some(llbc) = object_llbc() else { return };
    let graph = lower_named(llbc, "ll_listslice_new_int_list");
    let mut cc = CallControl::new();
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } = &op.kind
        else {
            continue;
        };
        if segments.last().is_some_and(|s| s == "int_ll_newlist") {
            cc.mark_oopspec(
                CallPath::from_segments(segments.iter().map(String::as_str)),
                "newlist(length)".to_string(),
            );
        }
    }
    let config = GraphTransformConfig::default();
    let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
    let out = transformer.transform(&graph);
    let kinds: Vec<String> = out
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::NewArray { .. } => Some("NewArray".to_string()),
            OpKind::NewArrayClear { .. } => Some("NewArrayClear".to_string()),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(format!(
                "Call:{}",
                segments.last().cloned().unwrap_or_default()
            )),
            OpKind::CallResidual { .. } => Some("CallResidual".to_string()),
            _ => None,
        })
        .collect();
    let new_arrays = kinds.iter().filter(|k| *k == "NewArray").count();
    assert!(
        new_arrays > 0,
        "int_ll_newlist oopspec must become NewArray, ops={kinds:?} notes={:?}",
        out.notes
    );
    assert_eq!(
        calls_to(&out.graph, "int_ll_newlist"),
        0,
        "int_ll_newlist must not remain a residual call after newlist(length) rewrite"
    );
}

/// `from_storage_and_strategy` is `instantiate` plus field stores.
/// `w_list_adopt_int_items` is that cluster: `malloc_typed_managed` fuses
/// to `NewWithVtable`, and the live Integer storage is the oopspec
/// `list.int_set_items` / `list.int_set_len` pair. The class address is
/// the production `pyobject::LIST_TYPE` row (`jit_static_pytype_addrs`).
#[test]
fn w_list_adopt_int_items_fuses_to_new_with_vtable() {
    let Some(llbc) = object_llbc() else { return };
    const LIST_TYPE_ADDR: i64 = 0x00C0_FFEE;
    let graph = lower_named_with_pytypes(
        llbc,
        "w_list_adopt_int_items",
        &[("pyobject::LIST_TYPE", LIST_TYPE_ADDR)],
    );
    let kinds: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .map(|op| match &op.kind {
            OpKind::NewWithVtable { owner, .. } => format!("NewWithVtable:{owner}"),
            OpKind::New { owner } => format!("New:{owner}"),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => format!("Call:{}", segments.last().cloned().unwrap_or_default()),
            OpKind::FieldWrite { field, .. } => format!("FieldWrite:{}", field.name),
            other => format!("{other:?}")
                .split('(')
                .next()
                .unwrap_or("op")
                .to_string(),
        })
        .collect();
    let nwv = kinds
        .iter()
        .filter(|k| k.starts_with("NewWithVtable:") && k.contains("W_ListObject"))
        .count();
    assert_eq!(
        nwv, 1,
        "adopt must record new_with_vtable of W_ListObject, ops={kinds:?}"
    );
    assert_eq!(
        calls_to(&graph, "malloc_typed_managed"),
        0,
        "malloc_typed_managed must fuse rather than stay residual"
    );
    assert_eq!(
        calls_to(&graph, "ll_list_int_set_items") + calls_to(&graph, "ll_list_int_set_len"),
        2,
        "live Integer storage is the oopspec set_items / set_len pair"
    );
}

/// Print how many bracketed bodies the scalar replacement rewrites, and why it
/// refuses the rest.  A measurement, not a gate.
#[test]
#[ignore]
fn shadow_stack_erase_census() {
    for (name, load) in [
        ("pyre-object", object_llbc as fn() -> Option<&'static Llbc>),
        ("pyre-interpreter", interpreter_llbc),
        ("pyre-module", module_llbc),
    ] {
        if std::env::var("CENSUS_ONLY").is_ok_and(|only| only != name) {
            continue;
        }
        let Some(llbc) = load() else { continue };
        let census = majit_translate::front::mir::shadow_stack_erase_census(llbc);
        for (bucket, bodies) in &census {
            eprintln!("[census {name}] {bucket} {}", bodies.len());
        }
        if let Ok(filter) = std::env::var("CENSUS_LIST") {
            if filter == "*" {
                for (bucket, bodies) in &census {
                    for body in bodies {
                        eprintln!("[census {name}] {bucket}: {body}");
                    }
                }
            } else {
                for body in census.get(filter.as_str()).into_iter().flatten() {
                    eprintln!("[census {name}] {filter}: {body}");
                }
            }
        }
    }
}

/// Nested exclusive RootScopes in `zip_two_tuple_next` close before they
/// merge with a sibling that never opened them. Closing must drop that
/// set's pin-count key; leftover keys refuse the rewrite and leave a
/// residual unpublished `push_roots`. FOR_ITER over zip already calls
/// `zip_two_tuple_next` as a residual; the body is not descended.
#[test]
fn zip_two_tuple_next_erases_the_nested_root_brackets() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(&object);
    interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
    let context = LowerContext::new(&interpreter);
    let graph = lower_fun(&interpreter, &context, "zip_two_tuple_next");
    for leaf in [
        "push_roots",
        "pin_root",
        "pin_roots",
        "shadow_stack_len",
        "shadow_stack_get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "zip_two_tuple_next still calls {leaf} after nested-scope erasure"
        );
    }
}

/// `slice_unpack` pins with free `pin_root` onto a `shadow_stack_len` base.
/// Scalar replacement refuses it (`calls-stack-sensitive-fn`: `__index__`),
/// and [`RootBracketPlan`] erases the bracket, including the len call.
#[test]
fn slice_unpack_erases_the_len_named_free_pin_bracket() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(&object);
    interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
    let context = LowerContext::new(&interpreter);
    let graph = lower_fun(&interpreter, &context, "slice_unpack");
    for leaf in [
        "push_roots",
        "pin_root",
        "pin_roots",
        "shadow_stack_len",
        "shadow_stack_get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "slice_unpack still calls {leaf} after the len-named free pins were erased"
        );
    }
}

/// `__majit_wrap_cdata_call` pins the incoming `&[PyObjectRef]` and copies
/// the tail through `shadow_stack_copy_range`. Production harvests
/// pyre-object's root-stack effects before lowering pyre-module, the same
/// order `lib.rs` uses; the incoming-slice plan must erase that bracket so
/// the walker is not left with a residual `push_roots` at wrap pc=41.
/// `W_CData.call` (`cdataobj.py descr_call`) passes `args_w` through, so a
/// callee that opens its own bracket (`W_CTypeFunc._call`) must not keep
/// wrap's pins in the jitcode.
#[test]
fn wrap_cdata_call_erases_the_incoming_slice_bracket() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
        || !std::path::Path::new(MODULE_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let module = Llbc::load(MODULE_LLBC).expect("load pyre-module");
    let mut crates = Vec::new();
    let mut touching = Vec::new();
    for llbc in [&object, &interpreter, &module] {
        llbc.set_root_stack_effects(crates.clone(), touching.clone());
        touching.extend(majit_translate::front::mir::harvest_root_stack_touching_paths(llbc));
        crates.push(llbc.crate_name().to_string());
    }
    let context = LowerContext::new(&module);
    let graph = lower_fun(&module, &context, "__majit_wrap_cdata_call");
    for leaf in [
        "push_roots",
        "pin_roots",
        "pin_root",
        "shadow_stack_get",
        "shadow_stack_copy_range",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "wrap still calls {leaf} after incoming-slice erasure"
        );
    }
    assert!(
        calls_to(&graph, "copy_object_slice_range_into_vec") > 0,
        "wrap must rewrite copy_range to copy_object_slice_range_into_vec"
    );
    assert!(
        calls_to(&graph, "call") > 0,
        "wrap must still reach ctypefunc::call"
    );
    assert!(
        calls_to(&graph, "ll_vec_alloc_and_set_r") > 0,
        "wrap must allocate the rest-args vec"
    );
    assert!(
        calls_to(&graph, "ll_vec_free_r") > 0,
        "wrap must free the rest-args vec; drop elaboration keeps the \
         definitely-init Drop of rest and rewrite_op_free frees that header"
    );
}

/// `try_new_for_call_with_closure_and_globals_obj` publishes `live` (the
/// `[code, globals, closure, *args]` Vec) through `pin_roots`. That is
/// the incoming-slice shape (`cdataobj.py descr_call` / `W_CData.call`
/// passes `args_w` through; `pyframe.py Frame.__init__` livevars are
/// that list), so the scalar replacement takes the bracket out of the
/// jitcode and the walker is not left with a residual `push_roots`.
#[test]
fn try_new_for_call_erases_the_published_slice_bracket() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let graph = lower_named(llbc, "try_new_for_call_with_closure_and_globals_obj");
    for leaf in ["push_roots", "pin_roots", "root_scope_close"] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "try_new still calls {leaf} after published-slice erasure"
        );
    }
}

/// `finish_for_call_with_globals_obj` publishes a 4-element array then
/// the incoming `args` slice. Both runs erase; a leftover `push_roots`
/// would residualize the whole constructor before `new_frame.run`.
#[test]
fn finish_for_call_erases_the_published_slice_bracket() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let graph = lower_named(llbc, "finish_for_call_with_globals_obj");
    for leaf in ["push_roots", "pin_roots", "publish", "root_scope_close"] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "finish_for_call still calls {leaf} after published-slice erasure"
        );
    }
}

/// `do_call` `n == 1` pins `args_w[0]` with `base()` + `pin_root` + `get(base)`.
/// The pin temporary is `StorageDead` before the get, so erasure rewrites
/// the get as `getarrayitem` of the incoming list. `n > 1` is
/// `pin_roots(args_w)` plus `get(args_slot + i)`: leftover-slice erasure
/// takes that bracket out the same way (`cdataobj.py descr_call` /
/// `pyframe.py Frame.__init__` livevars).
#[test]
fn do_call_erases_the_n1_incoming_elem_bracket() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
        || !std::path::Path::new(MODULE_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let module = Llbc::load(MODULE_LLBC).expect("load pyre-module");
    let mut crates = Vec::new();
    let mut touching = Vec::new();
    for llbc in [&object, &interpreter, &module] {
        llbc.set_root_stack_effects(crates.clone(), touching.clone());
        touching.extend(majit_translate::front::mir::harvest_root_stack_touching_paths(llbc));
        crates.push(llbc.crate_name().to_string());
    }
    let context = LowerContext::new(&module);
    let graph = lower_fun(&module, &context, "do_call");
    assert_eq!(
        calls_to(&graph, "push_roots"),
        0,
        "n==1 incoming-elem and n>1 incoming-slice push_roots must both erase"
    );
    assert_eq!(
        calls_to(&graph, "pin_roots"),
        0,
        "n>1 pin_roots(args_w) must be erased as leftover-slice"
    );
    assert_eq!(
        calls_to(&graph, "pin_root"),
        0,
        "n==1 pin_root(args_w[0]) must be erased"
    );
    assert_eq!(
        calls_to(&graph, "base"),
        0,
        "n==1 base() must be erased with the incoming-elem bracket"
    );
    let array_reads = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
        .count();
    assert!(
        array_reads > 0,
        "get(base) / get(args_slot + i) must rewrite to getarrayitem of args_w"
    );
}

/// `w_code_frame_stores_global` publishes `[obj, w_globals]`, then on the
/// miss path calls `publish_code_slot_store_rooting` which returns an owned
/// nested `RootScope`. The leftover is that dest (and ParamSlots methods
/// of it), so scalar replacement takes the outer bracket out.
#[test]
fn w_code_frame_stores_global_erases_the_outer_root_bracket() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(&object);
    interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
    let context = LowerContext::new(&interpreter);
    let graph = lower_fun(&interpreter, &context, "w_code_frame_stores_global");
    for leaf in ["push_roots", "pin_roots", "publish", "root_scope_close"] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "w_code_frame_stores_global still calls {leaf} after owned-scope erasure"
        );
    }
    assert!(
        calls_to(&graph, "publish_code_slot_store_rooting") > 0,
        "the nested owned-scope helper keeps its own bracket"
    );
}

/// `isinstance` (`abstract_isinstance_w`) pins `obj` and `classinfo` across
/// the `__instancecheck__` lookup and call. Its brackets must erase, or the
/// `isinstance` gateway descent stops at a residual `push_roots`.
#[test]
fn isinstance_bracket_is_scalar_replaced() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(&object);
    interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
    let context = LowerContext::new(&interpreter);
    let graph = lower_fun(&interpreter, &context, "baseobjspace::isinstance");
    for leaf in [
        "push_roots",
        "pin_roots",
        "pin_root",
        "shadow_stack_get",
        "root_scope_close",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "isinstance still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}
