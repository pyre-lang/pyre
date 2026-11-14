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
    INTERPRETER_LLBC, MODULE_LLBC, OBJECT_LLBC, interpreter_llbc, load_llbc, module_llbc,
    object_llbc, with_llbc_link_state, with_shared_lowering, with_shared_lowering_many,
};
use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, CallKind, FunId, JsonVal, Operand, PlaceKind, SwitchTargets, TermKind, TyRef,
    Unstructured,
};
use majit_translate::CallPath;
use majit_translate::HostStaticAddrs;
use majit_translate::call::CallControl;
use majit_translate::codewriter::jtransform::{GraphTransformConfig, Transformer};
use majit_translate::front::mir::{
    LowerContext, RootStackState, erased_root_bracket_guards, lower_fun_decl,
    lower_fun_decl_with_root_stack, lower_fun_decl_with_static_addrs,
};
use majit_translate::model::{CallFuncPtr, CallTarget, FunctionGraph, OpKind};

fn fun_decl_named<'a>(llbc: &'a Llbc, leaf: &str) -> &'a majit_charon_reader::FunDecl {
    let suffix = format!("::{leaf}");
    llbc.iter_local_fns()
        .find(|fd| fd.item_meta.name_path_str().ends_with(&suffix))
        .unwrap_or_else(|| panic!("{leaf} present in the shipped LLBC"))
}

fn lower_fun(llbc: &Llbc, context: &LowerContext<'_>, leaf: &str) -> FunctionGraph {
    lower_fun_decl(context, fun_decl_named(llbc, leaf))
        .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"))
}

fn lower_named(llbc: &'static Llbc, leaf: &str) -> FunctionGraph {
    with_shared_lowering(llbc, |ctx| lower_fun(llbc, ctx, leaf))
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
    with_shared_lowering(llbc, |ctx| {
        lower_fun_decl_with_static_addrs(
            ctx,
            fun_decl_named(llbc, leaf),
            HostStaticAddrs {
                pytypes,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"))
    })
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
    let cases: &[(&str, fn() -> &'static Llbc, &str)] = &[
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
        let llbc = load();
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
    let llbc = object_llbc();
    with_shared_lowering(llbc, |context| {
        let mut bodies = 0usize;
        let root_stack_state = RootStackState::new(llbc);
        for fd in llbc.iter_local_fns() {
            let Some(body) = fd.unstructured() else {
                continue;
            };
            let erased = erased_root_bracket_guards(llbc, fd, &body, &root_stack_state);
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
            let Ok(graph) = lower_fun_decl_with_root_stack(context, fd, &root_stack_state) else {
                continue;
            };
            assert!(
                reachable_closes(&graph) > 0,
                "{} drops a surviving guard in a block that does not bind it and              lowers no close",
                fd.item_meta.name_path_str()
            );
        }
        assert!(
            bodies > 10,
            "only {bodies} bodies drop a surviving guard away from its binding; the          fixture population is too small to prove anything"
        );
    });
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
    let llbc = object_llbc();
    with_shared_lowering(llbc, |context| {
        let mut dropping = 0usize;
        let root_stack_state = RootStackState::new(llbc);
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
            let Ok(graph) = lower_fun_decl_with_root_stack(context, fd, &root_stack_state) else {
                continue;
            };
            if calls_to(&graph, "root_scope_close") > 0 {
                continue;
            }
            let name = fd.item_meta.name_path().to_string();
            let moved = moved_out_locals(&body);
            let erased = erased_root_bracket_guards(llbc, fd, &body, &root_stack_state);
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
    });
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
        .is_some_and(|decl| {
            decl.item_meta
                .name_path_str()
                .ends_with("gc_roots::RootScope")
        })
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
    let llbcs: Vec<&'static Llbc> = vec![object_llbc(), interpreter_llbc()];
    with_shared_lowering_many(&llbcs, |ctxs| {
        // Derive metadata once per program and share each context across workers.
        let mut work: Vec<(
            &'static Llbc,
            &'static LowerContext<'static>,
            &majit_charon_reader::ullbc::FunDecl,
            usize,
        )> = Vec::new();
        for (i, llbc) in llbcs.iter().enumerate() {
            let context = ctxs[i];
            let root_stack_state = RootStackState::new(llbc);
            for fd in llbc.iter_local_fns() {
                let Some(body) = fd.unstructured() else {
                    continue;
                };
                let want = owed_closes(llbc, fd, &body, &root_stack_state);
                if want == 0 {
                    continue;
                }
                work.push((*llbc, context, fd, want));
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
                        for &(llbc, context, fd, want) in slice {
                            let state = RootStackState::new(llbc);
                            let Ok(graph) = lower_fun_decl_with_root_stack(context, fd, &state)
                            else {
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
    });
}

/// The closes a body owes: one per drop of an unmoved, unerased guard, counting
/// only the drops the front lowers. A cleanup-only drop is not one of them —
/// the front does not follow `on_unwind`.
fn owed_closes(
    llbc: &Llbc,
    fd: &majit_charon_reader::ullbc::FunDecl,
    body: &Unstructured,
    root_stack_state: &RootStackState,
) -> usize {
    let moved = moved_out_locals(body);
    let erased = erased_root_bracket_guards(llbc, fd, body, root_stack_state);
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

/// Trailing-segment suffix, matching `Llbc::local_fn`.
fn name_path_ends_with(path: &str, suffix: &str) -> bool {
    path == suffix || path.ends_with(&format!("::{suffix}"))
}

/// Locals whose Call dest the MIR front folds to a constant bool
/// (`jit::we_are_jitted` → true, `majit_gc::gc_alloc_failed_pending` → false).
fn folded_switch_bools(llbc: &Llbc, body: &Unstructured) -> Vec<Option<bool>> {
    let mut folded = Vec::new();
    for bb in &body.body {
        let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
            continue;
        };
        let PlaceKind::Local(n) = call.dest.kind else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            continue;
        };
        let Some(decl) = llbc.fn_by_id(*id) else {
            continue;
        };
        let path = decl.item_meta.name_path_str();
        let value = if name_path_ends_with(&path, "jit::we_are_jitted") {
            true
        } else if name_path_ends_with(&path, "majit_gc::gc_alloc_failed_pending") {
            false
        } else {
            continue;
        };
        let idx = n as usize;
        if idx >= folded.len() {
            folded.resize(idx + 1, None);
        }
        folded[idx] = Some(value);
    }
    folded
}

/// Whether a `SwitchInt` arm scalar is the integer 0. `None` if the JSON
/// is not a scalar we can read.
fn switch_int_arm_is_zero(scalar: &JsonVal) -> Option<bool> {
    if let Some(b) = scalar.get("Bool").and_then(serde_json::Value::as_bool) {
        return Some(!b);
    }
    let inner = scalar.get("Scalar").or_else(|| scalar.get("Integer"))?;
    let obj = inner.as_object()?;
    for payload in obj.values() {
        let value = match payload {
            serde_json::Value::Array(items) if items.len() >= 2 => &items[1],
            other => other,
        };
        if let Some(n) = value.as_i64() {
            return Some(n == 0);
        }
        if let Some(n) = value.as_u64() {
            return Some(n == 0);
        }
        if let Some(text) = value.as_str() {
            return text.parse::<i128>().ok().map(|n| n == 0);
        }
        if let Some(b) = value.as_bool() {
            return Some(!b);
        }
    }
    None
}

/// Target of the 0 arm, or `default` when every readable arm is nonzero.
/// `None` when no 0 arm is visible and at least one scalar is unreadable.
fn switch_int_zero_target(arms: &[(JsonVal, u64)], default: u64) -> Option<u64> {
    let mut zero = None;
    let mut unreadable = false;
    for (scalar, target) in arms {
        match switch_int_arm_is_zero(scalar) {
            Some(true) => zero = Some(*target),
            Some(false) => {}
            None => unreadable = true,
        }
    }
    if let Some(target) = zero {
        Some(target)
    } else if unreadable {
        None
    } else {
        Some(default)
    }
}

/// MIR blocks reachable from the entry without taking an unwind edge.
/// Blocks reachable only through the constant arm the MIR front folds away
/// (`gc_alloc_failed_pending()` is `false`, `we_are_jitted()` is `true`) owe
/// no close, because the lowered graph does not contain them.
fn reachable_without_unwind(llbc: &Llbc, body: &Unstructured) -> Vec<bool> {
    let folded = folded_switch_bools(llbc, body);
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
            TermKind::Switch { discr, targets } => {
                let folded_discr = match &discr {
                    Operand::Copy(place) | Operand::Move(place) => match place.kind {
                        PlaceKind::Local(n) => folded.get(n as usize).copied().flatten(),
                        _ => None,
                    },
                    _ => None,
                };
                match targets {
                    SwitchTargets::If(then_bb, else_bb) => match folded_discr {
                        Some(true) => stack.push(then_bb as usize),
                        Some(false) => stack.push(else_bb as usize),
                        None => stack.extend([then_bb as usize, else_bb as usize]),
                    },
                    SwitchTargets::SwitchInt(_, arms, default) => match folded_discr {
                        Some(true) => stack.push(default as usize),
                        Some(false) => match switch_int_zero_target(&arms, default) {
                            Some(target) => stack.push(target as usize),
                            None => {
                                stack.extend(arms.into_iter().map(|(_, t)| t as usize));
                                stack.push(default as usize);
                            }
                        },
                        None => {
                            stack.extend(arms.into_iter().map(|(_, t)| t as usize));
                            stack.push(default as usize);
                        }
                    },
                }
            }
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
    let llbc = object_llbc();
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
    let llbc = object_llbc();
    with_llbc_link_state(llbc, || {
        majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
        let names = majit_translate::front::mir::discover_depth_neutral_fns(llbc);
        assert!(
            names.iter().any(|id| id.contains("ll_listslice_inner")),
            "ll_listslice_inner must be depth-neutral"
        );
    });
}

/// The inner list-copy body is scalar-replaced: the walk records the
/// strategy copy, not residual `push_roots`.
#[test]
fn ll_listslice_inner_bracket_is_scalar_replaced() {
    let llbc = object_llbc();
    with_llbc_link_state(llbc, || {
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
    });
}

/// `funccall_flat_from_args` pins 2..=6 slots per `match nargs` arm, so
/// the join after that match is `AtMerge`. A later `match nargs` with
/// matching suffix gets is the same scrutinee (`constfold.constant_diffuse`
/// / `rewire_link_for_known_exitswitch`); erasure takes `push_roots` /
/// `pin_roots` / `get` out of the jitcode.
#[test]
fn funccall_flat_from_args_bracket_is_scalar_replaced() {
    let llbc = interpreter_llbc();
    with_llbc_link_state(llbc, || {
        majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
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
    });
}

/// `eval_slice_index` opens, pins, and closes on every return. The
/// path-sensitive walk classifies that body as depth-neutral.
#[test]
fn eval_slice_index_is_depth_neutral() {
    let llbc = interpreter_llbc();
    with_llbc_link_state(llbc, || {
        majit_translate::front::mir::ensure_stack_sensitive_fns(llbc);
        let names = majit_translate::front::mir::discover_depth_neutral_fns(llbc);
        assert!(
            names.iter().any(|id| id.contains("eval_slice_index")),
            "eval_slice_index must be depth-neutral"
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
    });
}

/// `zip_two_tuple_next` pins across `tuple_iter_descr_next`. Observes
/// inside its Open is erasable, so the helper jitcode has no root bracket.
#[test]
fn zip_two_tuple_next_bracket_is_scalar_replaced() {
    let llbc = interpreter_llbc();
    with_llbc_link_state(llbc, || {
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
    });
}

/// `binary_slice_values_inner` opens a RootScope, batch-publishes the three
/// operands, and calls `eval_slice_index`. A stack-sensitive callee used
/// to refuse the whole rewrite; the walker then aborted on `push_roots`.
#[test]
fn binary_slice_values_inner_bracket_is_scalar_replaced() {
    let llbc = interpreter_llbc();
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
/// Harvested strings are `scope_owner_identity`.
#[test]
fn binary_slice_values_inner_erases_with_linked_object_sensitive_set() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(object, || {
        with_llbc_link_state(interpreter, || {
            let context = LowerContext::new(interpreter);
            let (mut sensitive, mut leaves, mut params, mut ret_idx, mut observes, mut neutral) =
                majit_translate::front::mir::discover_stack_fn_effects(object);
            object.register_stack_sensitive_fns(sensitive.iter().cloned());
            object.register_stack_observes_fns(observes.iter().cloned());
            object.register_stack_leaves_above_fns(leaves.iter().cloned());
            object.register_stack_param_slots_fns(params.iter().cloned());
            object.register_stack_returns_index_fns(ret_idx.iter().cloned());
            object.register_stack_depth_neutral_fns(neutral.iter().cloned());
            interpreter.register_stack_sensitive_fns(sensitive.iter().cloned());
            interpreter.register_stack_observes_fns(observes.iter().cloned());
            interpreter.register_stack_depth_neutral_fns(neutral.iter().cloned());
            interpreter.register_stack_leaves_above_fns(leaves.iter().cloned());
            interpreter.register_stack_param_slots_fns(params.iter().cloned());
            interpreter.register_stack_returns_index_fns(ret_idx.iter().cloned());
            let (sens, more_leaves, more_params, more_ret, more_observes, more_neutral) =
                majit_translate::front::mir::discover_stack_fn_effects(interpreter);
            sensitive.extend(sens);
            observes.extend(more_observes);
            leaves.extend(more_leaves);
            params.extend(more_params);
            ret_idx.extend(more_ret);
            neutral.extend(more_neutral);
            interpreter.register_stack_sensitive_fns(sensitive);
            interpreter.register_stack_observes_fns(observes);
            interpreter.register_stack_leaves_above_fns(leaves);
            interpreter.register_stack_param_slots_fns(params);
            interpreter.register_stack_returns_index_fns(ret_idx);
            interpreter.register_stack_depth_neutral_fns(neutral);
            interpreter.mark_stack_sensitive_fns_complete();
            let graph = lower_fun(interpreter, &context, "binary_slice_values_inner");
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
        })
    });
}

/// `scope_owner_identity` splits the `<Impl>` homonyms that
/// `ItemMeta::name_path` collapses. `GraphAnalyzer.analyze_direct_call`
/// / `DependencyTracker` key by graph, never by a rendered name.
#[test]
fn stack_effect_identity_splits_impl_homonyms() {
    use majit_translate::front::mir::scope_owner_identity;
    let object = object_llbc();
    assert_scope_owner_identity_injective(object, "pyre-object");
    let get_ids: Vec<String> = object
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .filter(|fd| {
            let path = fd.item_meta.name_path_str();
            path.ends_with("::get") && path.contains("gc_roots::<Impl>")
        })
        .map(|fd| scope_owner_identity(object, fd))
        .collect();
    let mut unique = get_ids.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        get_ids.len(),
        "gc_roots::<Impl>::get identities must be injective, got {get_ids:?}"
    );
    let root_scope = unique
        .iter()
        .find(|id| {
            id.starts_with("pyre_object::gc_roots::<impl pyre_object::gc_roots::RootScope>::get|")
        })
        .cloned();
    let rooted_items = unique
        .iter()
        .find(|id| {
            id.starts_with("pyre_object::gc_roots::<impl pyre_object::gc_roots::RootedItems>::get|")
        })
        .cloned();
    let rooted_once = unique
        .iter()
        .find(|id| {
            id.starts_with(
                "pyre_object::gc_roots::<impl pyre_object::gc_roots::RootedOnceRef>::get|",
            )
        })
        .cloned();
    assert!(
        root_scope.is_some(),
        "missing RootScope identity in {unique:?}"
    );
    assert!(
        rooted_items.is_some(),
        "missing RootedItems identity in {unique:?}"
    );
    assert!(
        rooted_once.is_some(),
        "missing RootedOnceRef identity in {unique:?}"
    );
    assert_ne!(root_scope, rooted_items);
    assert_ne!(root_scope, rooted_once);
    assert_ne!(rooted_items, rooted_once);
    let interpreter = interpreter_llbc();
    assert_scope_owner_identity_injective(interpreter, "pyre-interpreter");
    assert_scope_owner_identity_injective(module_llbc(), "pyre-module");
    let release_ids: Vec<String> = interpreter
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .filter(|fd| {
            let path = fd.item_meta.name_path_str();
            path.ends_with("::release") && path.contains("baseobjspace::<Impl>")
        })
        .map(|fd| scope_owner_identity(interpreter, fd))
        .collect();
    let mut unique_release = release_ids.clone();
    unique_release.sort();
    unique_release.dedup();
    assert_eq!(
        unique_release.len(),
        release_ids.len(),
        "baseobjspace::<Impl>::release identities must be injective, got {release_ids:?}"
    );
    let simple_buf = unique_release
        .iter()
        .find(|id| {
            id.starts_with(
                "pyre_interpreter::baseobjspace::<impl pyre_interpreter::baseobjspace::SimpleBufferBytes>::release|",
            )
        })
        .cloned();
    let lock = unique_release
        .iter()
        .find(|id| {
            id.starts_with(
                "pyre_interpreter::baseobjspace::<impl pyre_interpreter::baseobjspace::Lock>::release|",
            )
        })
        .cloned();
    assert!(
        simple_buf.is_some(),
        "missing SimpleBufferBytes identity in {unique_release:?}"
    );
    assert!(
        lock.is_some(),
        "missing Lock identity in {unique_release:?}"
    );
    assert_ne!(simple_buf, lock);
    let saturating: Vec<String> = object
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .filter(|fd| fd.item_meta.name_path_str().ends_with("::saturating_add"))
        .map(|fd| scope_owner_identity(object, fd))
        .collect();
    let mut unique_sat = saturating.clone();
    unique_sat.sort();
    unique_sat.dedup();
    assert_eq!(
        unique_sat.len(),
        saturating.len(),
        "saturating_add identities must be injective, got {saturating:?}"
    );
    assert!(
        saturating.len() >= 2,
        "expected at least two saturating_add bodies, got {saturating:?}"
    );
    assert!(
        unique_sat
            .iter()
            .any(|id| id.starts_with("core::num::<impl usize>::saturating_add|")),
        "missing usize saturating_add in {unique_sat:?}"
    );
    assert!(
        unique_sat
            .iter()
            .any(|id| id.starts_with("core::num::<impl i64>::saturating_add|")),
        "missing i64 saturating_add in {unique_sat:?}"
    );
    let drops: Vec<String> = object
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .filter(|fd| {
            let path = fd.item_meta.name_path_str();
            path.ends_with("::drop_in_place") || path.contains("drop_in_place")
        })
        .map(|fd| scope_owner_identity(object, fd))
        .collect();
    let mut unique_drops = drops.clone();
    unique_drops.sort();
    unique_drops.dedup();
    assert_eq!(
        unique_drops.len(),
        drops.len(),
        "drop_in_place identities must be injective, got {drops:?}"
    );
    assert!(
        unique_drops
            .iter()
            .any(|id| id.starts_with("core::ptr::drop_in_place|")),
        "missing core::ptr::drop_in_place in {unique_drops:?}"
    );
    assert!(
        unique_drops
            .iter()
            .any(|id| id.contains("::<impl ") && id.contains("::drop_in_place|")),
        "missing impl drop_in_place in {unique_drops:?}"
    );
    let owners: Vec<String> = interpreter
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .filter(|fd| {
            fd.item_meta
                .name_path_str()
                .contains("typedef::method_owner::OWNER")
        })
        .map(|fd| scope_owner_identity(interpreter, fd))
        .collect();
    let mut unique_owners = owners.clone();
    unique_owners.sort();
    unique_owners.dedup();
    assert_eq!(
        unique_owners.len(),
        owners.len(),
        "typedef OWNER identities must be injective, got {owners:?}"
    );
    assert!(
        unique_owners
            .iter()
            .any(|id| id.starts_with("pyre_interpreter::typedef::method_owner::OWNER|")),
        "missing OWNER in {unique_owners:?}"
    );
    assert!(
        unique_owners
            .iter()
            .any(|id| id.contains("method_owner::OWNER#1|")),
        "missing OWNER#1 in {unique_owners:?}"
    );
    let extern_root_scope: Vec<String> = interpreter
        .iter_fun_decls()
        .filter(|fd| {
            let path = fd.item_meta.name_path_str();
            path.contains("gc_roots::<Impl>") && path.ends_with("::get")
        })
        .map(|fd| scope_owner_identity(interpreter, fd))
        .filter(|id| id.contains("RootScope"))
        .collect();
    let Some(local) = root_scope.as_ref() else {
        panic!("RootScope::get local identity missing");
    };
    assert!(
        extern_root_scope.iter().any(|id| id == local),
        "extern RootScope::get {extern_root_scope:?} must equal defining {local}"
    );
    eprintln!("identity RootScope::get = {local}");
    eprintln!(
        "identity RootedItems::get = {}",
        rooted_items.as_deref().unwrap_or("-")
    );
    eprintln!(
        "identity RootedOnceRef::get = {}",
        rooted_once.as_deref().unwrap_or("-")
    );
    eprintln!(
        "identity SimpleBufferBytes::release = {}",
        simple_buf.as_deref().unwrap_or("-")
    );
    eprintln!(
        "identity Lock::release = {}",
        lock.as_deref().unwrap_or("-")
    );
    for (i, id) in saturating.iter().take(2).enumerate() {
        eprintln!("identity saturating_add[{i}] = {id}");
    }
    for (i, id) in drops.iter().take(2).enumerate() {
        eprintln!("identity drop_in_place[{i}] = {id}");
    }
    for (i, id) in owners.iter().take(2).enumerate() {
        eprintln!("identity OWNER[{i}] = {id}");
    }
    eprintln!("identity RootScope::get extern = {local}");
}

fn assert_scope_owner_identity_injective(llbc: &Llbc, crate_label: &str) {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for fd in llbc.iter_local_fns().filter(|fd| fd.body.is_some()) {
        let identity = majit_translate::front::mir::scope_owner_identity(llbc, fd);
        let segs = format!("{:?}", fd.item_meta.name);
        groups
            .entry(identity)
            .or_default()
            .push((fd.item_meta.name_path(), segs));
    }
    let collisions: Vec<(String, Vec<(String, String)>)> = groups
        .into_iter()
        .filter(|(_, names)| names.len() > 1)
        .collect();
    if !collisions.is_empty() {
        eprintln!(
            "{crate_label} scope_owner_identity collision groups: {}",
            collisions.len()
        );
        for (identity, names) in &collisions {
            eprintln!("{crate_label} collision identity={identity}");
            for (name, segs) in names {
                eprintln!("{crate_label} collision body name={name} segs={segs}");
            }
        }
    }
    assert!(
        collisions.is_empty(),
        "{crate_label} scope_owner_identity collisions: {} groups",
        collisions.len()
    );
}

/// `|| shadow_stack_get(len)` after `pin_root` is a RootSlotGetter
/// closure. Publishing other crates' root-stack effects lets
/// `analyze_root_brackets` erase the bracket; the callee is the closure
/// `call`, not a `gc_roots` leaf, so lowering answers it from `get_sites`.
#[test]
fn a_captured_len_getter_lowers_once_root_stack_effects_are_published() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let mut crates = vec![object.crate_name().to_string()];
        let mut touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        crates.push(interpreter.crate_name().to_string());
        touching
            .extend(majit_translate::front::mir::harvest_root_stack_touching_paths(interpreter));
        interpreter.set_root_stack_effects(crates, touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "type_get_annotations");
        assert!(
            !graph.blocks.is_empty(),
            "a captured-len getter must lower once root-stack effects are published"
        );
    });
}

/// `ll_listslice_new_int_list` pins source and dest around `newlist` +
/// `ll_arraycopy`. The native RootScope is scalar-replaced so a
/// BINARY_SLICE sub-walk records the copy, not residual `pin_root`.
#[test]
fn ll_listslice_new_int_list_bracket_is_scalar_replaced() {
    let llbc = object_llbc();
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
    let llbc = object_llbc();
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
    let llbc = interpreter_llbc();
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
    let llbc = object_llbc();
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

/// The public `ll_list_int_resize_hint_really` is the look_inside_iff
/// wrapper (`iff` / trampoline / `_orig_*`). `newlist(length)` lives in
/// the `_orig_` body, which is the graph a constant-capacity grow inlines.
/// After jtransform that call is `new_array`, and the `try_gc_write_barrier`
/// before `int_set_items` is the gctransform `ll_writebarrier` that must
/// not survive as a residual (a residual call forces the virtual list on
/// `list_append_subscr_fresh`).
#[test]
fn ll_list_int_resize_hint_really_newlist_becomes_new_array() {
    let llbc = object_llbc();
    let graph = lower_named(llbc, "_orig_ll_list_int_resize_hint_really");
    let mut pre_calls: Vec<String> = Vec::new();
    let mut newlist_result_tys: Vec<String> = Vec::new();
    let mut cc = CallControl::new();
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty,
            ..
        } = &op.kind
        else {
            continue;
        };
        let Some(leaf) = segments.last() else {
            continue;
        };
        pre_calls.push(leaf.clone());
        if leaf == "int_ll_newlist" {
            newlist_result_tys.push(format!("{result_ty:?}"));
        }
        let spec = match leaf.as_str() {
            "int_ll_newlist" => Some("newlist(length)"),
            "ll_list_int_set_items" => Some("list.int_set_items(l, items)"),
            "ll_list_int_items" => Some("list.int_items(l)"),
            "ll_list_int_set_len" => Some("list.int_set_len(l, n)"),
            "ll_list_int_length" => Some("list.int_len(l)"),
            _ => None,
        };
        if let Some(spec) = spec {
            cc.mark_oopspec(
                CallPath::from_segments(segments.iter().map(String::as_str)),
                spec.to_string(),
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
            OpKind::FieldWrite { field, .. } => Some(format!("FieldWrite:{}", field.name)),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(format!(
                "Call:{}",
                segments.last().cloned().unwrap_or_default()
            )),
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(fp),
                ..
            } => {
                let path = out
                    .graph
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .find_map(|def| {
                        if def.result.as_ref() == Some(fp) {
                            match &def.kind {
                                OpKind::ConstFnAddr { path, .. } => Some(path.clone()),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "unknown".into());
                Some(format!("CallResidual:{path}"))
            }
            OpKind::CallResidual { .. } => Some("CallResidual".to_string()),
            _ => None,
        })
        .collect();
    assert!(
        kinds.iter().any(|k| k == "NewArray"),
        "int_ll_newlist oopspec must become NewArray, pre_calls={pre_calls:?} \
         newlist_result_tys={newlist_result_tys:?} ops={kinds:?} notes={:?}",
        out.notes
    );
    assert_eq!(
        calls_to(&out.graph, "int_ll_newlist"),
        0,
        "int_ll_newlist must not remain a residual call after newlist(length) rewrite"
    );
    assert_eq!(
        calls_to(&out.graph, "try_gc_write_barrier") + calls_to(&out.graph, "list_write_barrier"),
        0,
        "the barrier before int_set_items must drop, pre_calls={pre_calls:?} \
         ops={kinds:?} notes={:?}",
        out.notes
    );
    assert!(
        kinds
            .iter()
            .all(|k| { !k.contains("list_write_barrier") && !k.contains("try_gc_write_barrier") }),
        "write barrier must not survive as CallResidual, pre_calls={pre_calls:?} \
         ops={kinds:?} notes={:?}",
        out.notes
    );
}

/// Production `apply_harvested_funcobj_hints` is what the codewriter
/// consults for `newlist(length)`. The sibling test injects
/// `mark_oopspec` on the exact CallPath; this one only stamps harvested
/// `oopspec:` tokens, which is the path `list_append_subscr_fresh` takes.
#[test]
fn harvested_int_ll_newlist_oopspec_becomes_new_array() {
    let llbc = object_llbc();
    let harvested =
        majit_translate::front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(llbc));
    let newlist_keys: Vec<(String, Vec<String>)> = harvested
        .iter()
        .filter(|(k, hints)| {
            k.contains("int_ll_newlist") || hints.iter().any(|h| h.starts_with("oopspec:newlist"))
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let graph = lower_named(llbc, "_orig_ll_list_int_resize_hint_really");
    let mut cc = CallControl::new();
    for (path_str, hints) in &harvested {
        if hints.iter().all(|h| !h.starts_with("oopspec:")) {
            continue;
        }
        // `apply_harvested_funcobj_hints` / `free_function_alias_paths`
        // register the crate-stripped harvest key and the defining-crate
        // spelling the callsite uses (`pyre_object::listobject::…`).
        let stripped: Vec<&str> = path_str.split("::").collect();
        cc.mark_decorator_hints(&CallPath::from_segments(stripped.iter().copied()), hints);
        let mut crate_prefixed = vec!["pyre_object"];
        crate_prefixed.extend(stripped.iter().copied());
        cc.mark_decorator_hints(&CallPath::from_segments(crate_prefixed), hints);
    }
    let mut oopspec_hits: Vec<String> = Vec::new();
    let mut newlist_result_tys: Vec<String> = Vec::new();
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::Call {
            target, result_ty, ..
        } = &op.kind
        else {
            continue;
        };
        let CallTarget::FunctionPath { segments, .. } = target else {
            continue;
        };
        let Some(leaf) = segments.last() else {
            continue;
        };
        if *leaf == "int_ll_newlist" {
            newlist_result_tys.push(format!("{result_ty:?}"));
            oopspec_hits.push(format!(
                "{} -> {:?}",
                segments.join("::"),
                cc.get_oopspec(target)
            ));
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
    assert!(
        kinds.iter().any(|k| k == "NewArray"),
        "harvested newlist(length) must become NewArray, newlist_keys={newlist_keys:?} \
         oopspec_hits={oopspec_hits:?} newlist_result_tys={newlist_result_tys:?} \
         ops={kinds:?} notes={:?}",
        out.notes
    );
}

/// `gil.py GILThreadLocals.gil_ready` is a harvested `_alloc_flavor_ = raw`
/// `static mut`. The Global emits an Address-stamped ConstRefAddr and
/// `__cast_instance_intrinsic` onto the class root so getattr of
/// `gil_ready` annotates; the instance pointer is int-kinded
/// (`descr.py get_type_flag` of a raw Ptr).
#[test]
fn orig_gil_ready_word_static_is_int_banked_address() {
    let llbc = object_llbc();
    let harvested = majit_translate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(
        std::slice::from_ref(llbc),
    );
    assert!(
        harvested.iter().any(|s| s.contains("GilReadyState")),
        "harvested flavor set must include GilReadyState, got {harvested:?}"
    );
    let graph = with_shared_lowering(llbc, |ctx| {
        lower_fun_decl_with_static_addrs(
            ctx,
            fun_decl_named(llbc, "gil_ready_word"),
            HostStaticAddrs {
                refs: &[("gil_ready::GIL_READY_STATE", 0x1111)],
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("lower gil_ready_word: {e:?}"))
    });
    let mut const_ref = Vec::new();
    let mut casts = 0usize;
    let mut cast_kind = None;
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        match &op.kind {
            OpKind::ConstRefAddr(addr) => {
                let ct = op.result.as_ref().map(FunctionGraph::concretetype_of);
                let ll = op.result.as_ref().and_then(|v| v.concretetype());
                const_ref.push((*addr, ct, ll));
            }
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty,
                ..
            } if segments.last().map(String::as_str) == Some("__cast_instance_intrinsic") => {
                casts += 1;
                cast_kind = Some((
                    op.result.as_ref().map(FunctionGraph::concretetype_of),
                    result_ty.clone(),
                ));
            }
            _ => {}
        }
    }
    assert_eq!(
        const_ref
            .iter()
            .map(|(addr, _, _)| *addr)
            .collect::<Vec<_>>(),
        vec![0x1111],
        "gil_ready_word must load GIL_READY_STATE as ConstRefAddr: {const_ref:?} graph={graph:?}"
    );
    assert!(
        const_ref.iter().any(|(_, ct, ll)| {
            *ct == Some(majit_translate::ConcreteType::Signed)
                && matches!(
                    ll,
                    Some(majit_translate::translator::rtyper::lltypesystem::lltype::LowLevelType::Address)
                )
        }),
        "GIL_READY_STATE address word must be llmemory.Address, got {const_ref:?} graph={graph:?}"
    );
    assert_eq!(
        casts, 1,
        "raw-flavor STRUCT static wraps as the class instance: {graph:?}"
    );
    assert_eq!(
        cast_kind,
        Some((
            Some(majit_translate::ConcreteType::Signed),
            majit_translate::ValueType::Int
        )),
        "raw instance pointer is int-kinded: {graph:?}"
    );
}

/// `jtransform.py rewrite_op_getfield` of `gil_ready?` is
/// `-live-` + `record_quasiimmut_field` + `getfield_*_pure`. A residual
/// Call of the static field would leave the sub-walk with a symbolic
/// fnaddr (`get_jitcode_shell_falls_back_to_symbolic_fnaddr`).
#[test]
fn orig_gil_ready_word_transforms_to_quasiimmut_field_read() {
    let llbc = object_llbc();
    let ranks = majit_translate::front::llbc_hints::harvest_immutable_fields_from_llbcs(
        std::slice::from_ref(llbc),
    );
    assert!(
        ranks.get("GilReadyState").is_some_and(|fields| {
            fields
                .iter()
                .any(|(name, rank)| name == "gil_ready" && rank.is_quasi_immutable())
        }),
        "harvested _immutable_fields_ for GilReadyState: {:?}",
        ranks.get("GilReadyState")
    );
    let flavors = majit_translate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(
        std::slice::from_ref(llbc),
    );
    let graph = with_shared_lowering(llbc, |ctx| {
        lower_fun_decl_with_static_addrs(
            ctx,
            fun_decl_named(llbc, "gil_ready_word"),
            HostStaticAddrs {
                refs: &[("gil_ready::GIL_READY_STATE", 0x1111)],
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("lower gil_ready_word: {e:?}"))
    });
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "gil_ready"
            )),
        "gil_ready_word front graph must FieldRead gil_ready: {graph:?}"
    );
    let mut cc = CallControl::new();
    cc.immutable_fields_by_struct = ranks;
    cc.raw_alloc_flavor_structs = flavors;
    let out = Transformer::new(&GraphTransformConfig::default())
        .with_callcontrol(&mut cc)
        .transform(&graph);
    let kinds: Vec<String> = out
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .map(|op| match &op.kind {
            OpKind::RecordQuasiImmutField {
                field,
                mutate_field,
                ..
            } => {
                format!("RecordQuasiImmutField:{}:{}", field.name, mutate_field.name)
            }
            OpKind::FieldRead { field, pure, .. } => {
                format!("FieldRead:{}:pure={pure}", field.name)
            }
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => format!("Call:{}", segments.join("::")),
            OpKind::CallResidual { .. } => "CallResidual".to_string(),
            OpKind::ConstRefAddr(_) => "ConstRefAddr".to_string(),
            OpKind::Live { .. } => "Live".to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    assert!(
        kinds
            .iter()
            .any(|k| k == "RecordQuasiImmutField:gil_ready:mutate_gil_ready"),
        "rewrite_op_getfield must emit record_quasiimmut_field for gil_ready?: {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "FieldRead:gil_ready:pure=true"),
        "quasi-immut gil_ready read must be getfield_*_pure: {kinds:?}"
    );
    assert!(
        !kinds.iter().any(|k| k.starts_with("CallResidual")
            || (k.starts_with("Call:") && !k.contains("__cast_instance_intrinsic"))),
        "gil_ready_word must not residualize the static field read: {kinds:?}"
    );
}

/// `_orig_ll_list_int_resize_hint_really` is the look_inside body the
/// LIST_EXTEND unroll walks. Its `int_ll_newlist` must become `new_array`
/// the same way `ll_listslice_new_int_list` does; a residual call aborts
/// the helper subwalk at that pc.
#[test]
fn orig_ll_list_int_resize_hint_really_newlist_becomes_new_array() {
    let llbc = object_llbc();
    let hints =
        majit_translate::front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(llbc));
    let harvested: Vec<(String, Vec<String>)> = hints
        .iter()
        .filter(|(k, _)| k.contains("int_ll_newlist") || k.contains("ll_list_int_resize_hint"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let graph = lower_named(llbc, "_orig_ll_list_int_resize_hint_really");
    let mut call_shapes = Vec::new();
    let mut cc = CallControl::new();
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty,
            ..
        } = &op.kind
        else {
            continue;
        };
        if segments.last().is_some_and(|s| s == "int_ll_newlist") {
            let ct = op
                .result
                .as_ref()
                .and_then(|v| v.concretetype())
                .map(|t| format!("{t:?}"));
            call_shapes.push((segments.join("::"), format!("{result_ty:?}"), ct));
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
        "int_ll_newlist in _orig_ll_list_int_resize_hint_really must become NewArray, \
         harvested={harvested:?} call_shapes={call_shapes:?} ops={kinds:?} notes={:?}",
        out.notes
    );
    assert_eq!(
        calls_to(&out.graph, "int_ll_newlist"),
        0,
        "int_ll_newlist must not remain a residual call after newlist(length) rewrite; \
         harvested={harvested:?} call_shapes={call_shapes:?}"
    );
    assert_eq!(
        calls_to(&out.graph, "list_write_barrier"),
        0,
        "list_write_barrier must not remain a residual; the items setfield \
         takes COND_CALL_GC_WB from rewrite.py; \
         harvested={harvested:?} call_shapes={call_shapes:?} ops={kinds:?}"
    );

    // Dual-gate Skip kind-collapses the result to OBJECTPTR. The Call's
    // result_ty still names TypedItemsBlock; recover the ARRAY from that
    // spelling, the way `newlist_clear` recovers FixedObjectArray.
    let collapsed = lower_named(llbc, "_orig_ll_list_int_resize_hint_really");
    let mut cc = CallControl::new();
    for op in collapsed.blocks.iter().flat_map(|b| &b.operations) {
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
            if let Some(result) = &op.result {
                result.set_concretetype(None);
            }
        }
    }
    let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
    let out = transformer.transform(&collapsed);
    let collapsed_new_arrays = out
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| matches!(op.kind, OpKind::NewArray { .. }))
        .count();
    assert!(
        collapsed_new_arrays > 0,
        "kind-collapsed TypedItemsBlock result must still become NewArray, \
         call_shapes={call_shapes:?} notes={:?}",
        out.notes
    );
}

/// `from_storage_and_strategy` is `instantiate` plus field stores.
/// `w_list_adopt_int_items` is that cluster: `malloc_typed_managed` fuses
/// to `NewWithVtable`, and the live Integer storage is the oopspec
/// `list.int_set_items` / `list.int_set_len` pair. The class address is
/// the production `pyobject::LIST_TYPE` row (`jit_static_pytype_addrs`).
#[test]
fn w_list_adopt_int_items_fuses_to_new_with_vtable() {
    let llbc = object_llbc();
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
        ("pyre-object", object_llbc as fn() -> &'static Llbc),
        ("pyre-interpreter", interpreter_llbc),
        ("pyre-module", module_llbc),
    ] {
        if std::env::var("CENSUS_ONLY").is_ok_and(|only| only != name) {
            continue;
        }
        let llbc = load();
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
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "zip_two_tuple_next");
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
    });
}

/// `FrameBox::new` closes with `drop(frame_root)`, which Charon spells as
/// `_t = Move(guard); core::mem::drop(Move(_t))`. Production harvests
/// pyre-object's root-stack effects before lowering the interpreter, the
/// same order `lib.rs` uses; that Call is Close of the watched Guard so
/// the bracket is not a residual `push_roots`.
#[test]
fn frame_box_new_erases_the_explicit_drop_root_bracket() {
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
    let fd = interpreter
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .source_text
                .as_deref()
                .is_some_and(|text| text.starts_with("pub fn new(mut frame: PyFrame)"))
        })
        .expect("FrameBox::new");
    let graph =
        lower_fun_decl(&context, fd).unwrap_or_else(|e| panic!("lower FrameBox::new: {e:?}"));
    let leftover: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                let leaf = segments.last().map(String::as_str).unwrap_or("");
                let forbidden = matches!(
                    leaf,
                    "push_roots"
                        | "pin_root"
                        | "publish"
                        | "normalize"
                        | "get"
                        | "root_scope_close"
                ) || segments
                    .iter()
                    .map(String::as_str)
                    .eq(["core", "mem", "drop"]);
                forbidden.then_some(segments.join("::"))
            }
            _ => None,
        })
        .collect();
    assert!(
        leftover.is_empty(),
        "FrameBox::new still calls root-bracket leaves after explicit-drop erasure: {leftover:?}"
    );
}

/// `slice_unpack` pins with free `pin_root` onto a `shadow_stack_len` base.
/// Scalar replacement refuses it (`calls-stack-sensitive-fn`: `__index__`),
/// and [`RootBracketPlan`] erases the bracket, including the len call.
#[test]
fn slice_unpack_erases_the_len_named_free_pin_bracket() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "slice_unpack");
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
    });
}

/// `long_rshift` captures the `pin_roots` index in closures
/// (`|| shadow_stack_get(base)` / `|| shadow_stack_get(base + 1)`).
/// Scalar replacement answers those calls as slot copies so the
/// codewriter never sees the bracket (`gctransform` `push_roots` runs
/// after `make_jitcodes`).
#[test]
fn long_rshift_erases_the_closure_capture_bracket() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "long_rshift");
        for leaf in [
            "push_roots",
            "pin_roots",
            "shadow_stack_get",
            "root_scope_close",
        ] {
            assert_eq!(
                calls_to(&graph, leaf),
                0,
                "long_rshift still calls {leaf} after the closure capture was erased"
            );
        }
    });
}

/// `__majit_wrap_dict_descr_keys` pins an incoming `&[PyObjectRef]` then
/// a later `pin_root`. Scalar replacement answers `get(base)` from the
/// list and `get(len)` from the extra slot so the codewriter never sees
/// the bracket (`gctransform` `push_roots` runs after `make_jitcodes`).
#[test]
fn dict_descr_keys_erases_the_incoming_slice_bracket() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "__majit_wrap_dict_descr_keys");
        for leaf in [
            "push_roots",
            "pin_roots",
            "shadow_stack_get",
            "root_scope_close",
        ] {
            assert_eq!(
                calls_to(&graph, leaf),
                0,
                "__majit_wrap_dict_descr_keys still calls {leaf} after the incoming-slice bracket was erased"
            );
        }
    });
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
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    let module = load_llbc(MODULE_LLBC);
    with_llbc_link_state(object, || {
        with_llbc_link_state(interpreter, || {
            with_llbc_link_state(module, || {
                let mut crates = Vec::new();
                let mut touching = Vec::new();
                for llbc in [object, interpreter, module] {
                    llbc.set_root_stack_effects(crates.clone(), touching.clone());
                    touching.extend(
                        majit_translate::front::mir::harvest_root_stack_touching_paths(llbc),
                    );
                    crates.push(llbc.crate_name().to_string());
                }
                let context = LowerContext::new(module);
                let graph = lower_fun(module, &context, "__majit_wrap_cdata_call");
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
            });
        });
    });
}

/// `try_new_for_call_with_closure_and_globals_obj` publishes `live` (the
/// `[code, globals, closure, *args]` Vec) through `pin_roots`. That is
/// the incoming-slice shape (`cdataobj.py descr_call` / `W_CData.call`
/// passes `args_w` through; `pyframe.py Frame.__init__` livevars are
/// that list), so the scalar replacement takes the bracket out of the
/// jitcode and the walker is not left with a residual `push_roots`.
#[test]
fn try_new_for_call_erases_the_published_slice_bracket() {
    let llbc = interpreter_llbc();
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
    let llbc = interpreter_llbc();
    let graph = lower_named(llbc, "finish_for_call_with_globals_obj");
    for leaf in ["push_roots", "pin_roots", "publish", "root_scope_close"] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "finish_for_call still calls {leaf} after published-slice erasure"
        );
    }
}

fn callee_leaf<'a>(
    llbc: &'a Llbc,
    call: &majit_charon_reader::ullbc::CallPayload,
) -> Option<&'a str> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    llbc.fn_by_id(*id)?
        .item_meta
        .name_path_str()
        .rsplit("::")
        .next()
}

/// FunDecl ids of `FrameLocalsRoot::new`: the inherent `new` whose body
/// calls `register_frame_locals_slot`. Tests may name that helper; the
/// erase pass itself keys only `gc_hook`.
fn frame_locals_root_new_ids(llbc: &Llbc) -> Vec<u64> {
    llbc.iter_fun_decls()
        .filter(|fd| {
            fd.item_meta.name_path().ends_with("::new")
                && fd.unstructured().is_some_and(|body| {
                    body.body.iter().any(|bb| {
                        matches!(
                            bb.term_ref(llbc),
                            Ok(TermKind::Call { call, .. })
                                if callee_leaf(llbc, &call) == Some("register_frame_locals_slot")
                        )
                    })
                })
        })
        .map(|fd| fd.def_id)
        .collect()
}

fn call_fun_decl_id(target: &CallTarget) -> Option<u64> {
    match target {
        CallTarget::FunctionPath { fun_decl_id, .. } | CallTarget::Method { fun_decl_id, .. } => {
            *fun_decl_id
        }
        _ => None,
    }
}

fn calls_to_frame_locals_root_new(graph: &FunctionGraph, ids: &[u64]) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call { target, .. } => {
                call_fun_decl_id(target).is_some_and(|id| ids.contains(&id))
            }
            _ => false,
        })
        .count()
}

/// `finish_for_call_with_globals_obj` roots `addr_of_mut!(frame.locals_cells_stack_w)`
/// through a RAII guard over a by-value local aggregate. The slot is already
/// an SSA Ref in this graph, so the opener is not a jitcode op
/// (`shadowstack.py` `push_roots` / `pop_roots` run after `make_jitcodes`).
#[test]
fn finish_for_call_erases_the_frame_locals_root_registration() {
    let llbc = interpreter_llbc();
    let ids = frame_locals_root_new_ids(llbc);
    assert!(
        !ids.is_empty(),
        "interpreter LLBC must contain FrameLocalsRoot::new"
    );
    let graph = lower_named(llbc, "finish_for_call_with_globals_obj");
    assert_eq!(
        calls_to_frame_locals_root_new(&graph, &ids),
        0,
        "finish_for_call still calls FrameLocalsRoot::new"
    );
    for leaf in [
        "register_frame_locals_slot",
        "try_gc_add_root",
        "unregister_frame_locals_slot",
        "try_gc_remove_root",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "finish_for_call still calls {leaf} after the local-aggregate root erase"
        );
    }
}

/// A guard over a heap frame pointer (`into_raw` / `as_mut_ptr`) is not
/// a scalar-replaced local aggregate. The opener stays in the jitcode.
#[test]
fn into_generator_named_keeps_heap_frame_root_registration() {
    let llbc = interpreter_llbc();
    let ids = frame_locals_root_new_ids(llbc);
    assert!(
        !ids.is_empty(),
        "interpreter LLBC must contain FrameLocalsRoot::new"
    );
    let graph = lower_named(llbc, "into_generator_named");
    assert_eq!(
        calls_to_frame_locals_root_new(&graph, &ids),
        1,
        "into_generator_named must keep the heap-pointer FrameLocalsRoot::new"
    );
}

/// `do_call` `n == 1` pins `args_w[0]` with `base()` + `pin_root` + `get(base)`.
/// The pin temporary is `StorageDead` before the get, so erasure rewrites
/// the get as `getarrayitem` of the incoming list. `n > 1` is
/// `pin_roots(args_w)` plus `get(args_slot + i)`: leftover-slice erasure
/// takes that bracket out the same way (`cdataobj.py descr_call` /
/// `pyframe.py Frame.__init__` livevars).
#[test]
fn do_call_erases_the_n1_incoming_elem_bracket() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    let module = load_llbc(MODULE_LLBC);
    with_llbc_link_state(object, || {
        with_llbc_link_state(interpreter, || {
            with_llbc_link_state(module, || {
                let mut crates = Vec::new();
                let mut touching = Vec::new();
                for llbc in [object, interpreter, module] {
                    llbc.set_root_stack_effects(crates.clone(), touching.clone());
                    touching.extend(
                        majit_translate::front::mir::harvest_root_stack_touching_paths(llbc),
                    );
                    crates.push(llbc.crate_name().to_string());
                }
                let context = LowerContext::new(module);
                let graph = lower_fun(module, &context, "do_call");
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
            });
        });
    });
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
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "w_code_frame_stores_global");
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
    });
}

/// `isinstance` (`abstract_isinstance_w`) pins `obj` and `classinfo` across
/// the `__instancecheck__` lookup and call. Its brackets must erase, or the
/// `isinstance` gateway descent stops at a residual `push_roots`.
#[test]
fn isinstance_bracket_is_scalar_replaced() {
    let object = load_llbc(OBJECT_LLBC);
    let interpreter = load_llbc(INTERPRETER_LLBC);
    with_llbc_link_state(interpreter, || {
        let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(object);
        interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
        let context = LowerContext::new(interpreter);
        let graph = lower_fun(interpreter, &context, "baseobjspace::isinstance");
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
    });
}
