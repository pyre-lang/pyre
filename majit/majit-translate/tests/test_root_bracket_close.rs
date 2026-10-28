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
//! The close is spelled as a call taking the guard's save_point word rather
//! than a reference or two field reads, because a crate that imports
//! `RootScope` sees an opaque stub with no fields. Reading the fields would
//! confine the close to `pyre-object`, which is why `pyre_interpreter` is
//! checked here and not just `pyre_object`. The residual argument is Int so
//! the jitcode has no `cast_int_to_ptr` before the close.
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
    INTERPRETER_LLBC, MODULE_LLBC, OBJECT_LLBC, interpreter_llbc, module_llbc, object_llbc,
};
use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TypeDeclKind;
use majit_charon_reader::ullbc::{PlaceKind, SwitchTargets, TermKind, TyRef, Unstructured};
use majit_translate::CallPath;
use majit_translate::HostStaticAddrs;
use majit_translate::call::CallControl;
use majit_translate::codewriter::jtransform::{GraphTransformConfig, Transformer};
use majit_translate::front::mir::{
    LowerContext, apply_published_structs, erased_root_bracket_guards, harvest_published_structs,
    lower_fun_decl, lower_fun_decl_with_static_addrs,
};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
use std::sync::OnceLock;

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

/// Body-carrying FunDecls from defining crates so an Opaque import of a
/// wrapper is a regular call into that crate's graph, matching the merged
/// program (`StructFieldRegistry::foreign_callables`).
fn linked_context(llbc: &'static Llbc) -> &'static LowerContext<'static> {
    static OBJECT: OnceLock<LowerContext<'static>> = OnceLock::new();
    static INTERPRETER: OnceLock<LowerContext<'static>> = OnceLock::new();
    static MODULE: OnceLock<LowerContext<'static>> = OnceLock::new();
    let extras: Vec<&Llbc> = match llbc.crate_name() {
        "pyre_interpreter" => object_llbc().into_iter().collect(),
        "pyre_module" => object_llbc()
            .into_iter()
            .chain(interpreter_llbc())
            .collect(),
        _ => Vec::new(),
    };
    let slot = match llbc.crate_name() {
        "pyre_object" => &OBJECT,
        "pyre_interpreter" => &INTERPRETER,
        "pyre_module" => &MODULE,
        other => panic!("linked_context: unexpected crate {other}"),
    };
    slot.get_or_init(|| {
        let mut context = LowerContext::new(llbc);
        for extra in &extras {
            context.note_loaded_crate_bodies(extra);
        }
        context
    })
}

fn context_with_extras<'a>(llbc: &'a Llbc, extras: &[&Llbc]) -> LowerContext<'a> {
    let mut context = LowerContext::new(llbc);
    for extra in extras {
        context.note_loaded_crate_bodies(extra);
    }
    context
}

fn lower_named(llbc: &'static Llbc, leaf: &str) -> FunctionGraph {
    lower_fun(llbc, linked_context(llbc), leaf)
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
        linked_context(llbc),
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
    let closed = calls_to(&graph, "root_scope_close_word");
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
        // `do_call` erases the `n==1` incoming-elem pin; `n>1` stays
        // because `convert_argument_from_object` LeavesAbove. `call_varargs`
        // still opens a surviving bracket: `complete_argtypes` is param_slots
        // so scalar replacement refuses, and RootBracketPlan needs a const
        // slot for `get(args_base + i)`.
        ("cffi", module_llbc, "call_varargs"),
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

/// Residual `push_roots` returns the guard as its save_point Int. The close
/// takes that word as Int (`root_scope_close_word`); a Ref argument would
/// make jtransform insert `cast_int_to_ptr` of a non-GC word.
#[test]
fn close_takes_the_guard_word_as_int() {
    let cases: &[(&str, fn() -> Option<&'static Llbc>, &str)] = &[
        ("owns_guard", object_llbc, "w_tuple_items_copy_as_vec"),
        (
            "imports_guard",
            interpreter_llbc,
            "call_function_impl_result",
        ),
        ("bytes_concat", interpreter_llbc, "bytes_concat"),
    ];
    for (name, load, leaf) in cases {
        let Some(llbc) = load() else {
            continue;
        };
        let graph = lower_named(llbc, leaf);
        let mut found = 0usize;
        for op in graph.blocks.iter().flat_map(|b| &b.operations) {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !segments
                .last()
                .is_some_and(|s| s == "root_scope_close_word")
            {
                continue;
            }
            found += 1;
            assert_eq!(args.len(), 1, "{name}/{leaf}: close has one arg");
            let arg = args[0]
                .as_variable()
                .unwrap_or_else(|| panic!("{name}/{leaf}: close arg is a variable"));
            let from_cast = graph.blocks.iter().flat_map(|b| &b.operations).any(|prev| {
                prev.result.as_ref() == Some(arg)
                    && match &prev.kind {
                        OpKind::UnaryOp { op, .. } => op == "cast_int_to_ptr",
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } => segments.last().is_some_and(|s| s == "cast_int_to_ptr"),
                        _ => false,
                    }
            });
            assert!(
                !from_cast,
                "{name}/{leaf}: close arg is produced by cast_int_to_ptr"
            );
        }
        assert!(
            found > 0,
            "{name}/{leaf} is the fixture for a surviving root bracket, but it closes none"
        );
    }
}

/// The wrapper's sized field is the word (`tyref_transparent_nonzst_field` /
/// `tyref_transparent_inner_value_type`), including through `&` / `&mut`.
/// A FieldRead would getattr that field on Integer.
#[test]
fn wrapper_sized_field_projection_is_the_word() {
    let Some(llbc) = object_llbc() else { return };
    let context = linked_context(llbc);
    let mut bodies = 0usize;
    for fd in llbc.iter_local_fns() {
        let path = fd.item_meta.name_path();
        let projects = path.ends_with("::root_scope_close")
            || (path.ends_with("::base") && path.contains("gc_roots"));
        if !projects {
            continue;
        }
        let Ok(graph) = lower_fun_decl(&context, fd) else {
            continue;
        };
        bodies += 1;
        for op in graph.blocks.iter().flat_map(|b| &b.operations) {
            if let OpKind::FieldRead { field, .. } = &op.kind {
                panic!("{path} still reads field {}", field.name);
            }
        }
    }
    assert!(
        bodies > 0,
        "object LLBC has no lowerable wrapper-field projection body"
    );
}

/// Production copies the defining crate's field list onto an importer's
/// Opaque wrapper. The copy has to keep the unique scalar-word field so
/// identity can fire (`tyref_transparent_inner_value_type`).
#[test]
fn published_wrapper_on_jit_artefact_keeps_scalar_word_field() {
    const JIT_LLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-jit.ullbc"
    );
    if !std::path::Path::new(OBJECT_LLBC).is_file() || !std::path::Path::new(JIT_LLBC).is_file() {
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load object llbc");
    let mut jit = Llbc::load(JIT_LLBC).expect("load jit llbc");
    let rows = harvest_published_structs(&object);
    let harvested = rows
        .iter()
        .filter(|row| row.path.ends_with("RootScope"))
        .map(|row| {
            (
                row.path.clone(),
                row.fields
                    .iter()
                    .map(|f| (f.name.clone(), format!("{}", f.ty)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    apply_published_structs(&mut jit, &rows);
    let td = jit
        .iter_type_decls()
        .find(|td| td.item_meta.name_path().ends_with("::RootScope"))
        .expect("jit artefact names the wrapper");
    let fields = match &td.kind {
        TypeDeclKind::Struct(fields) => fields
            .iter()
            .map(|f| (f.name.clone(), format!("{:?}", f.ty)))
            .collect::<Vec<_>>(),
        other => {
            panic!(
                "jit wrapper after publish is {other:?}, harvested={harvested:?}, published={}",
                td.published_struct_copy
            )
        }
    };
    assert!(
        td.published_struct_copy,
        "jit wrapper is a published copy; harvested={harvested:?} fields={fields:?}"
    );
    assert!(
        td.is_repr_transparent(),
        "jit wrapper keeps repr(transparent); fields={fields:?}"
    );
    let marker = fields
        .iter()
        .find(|(name, _)| name.as_deref() == Some("_not_send"));
    let Some((_, ty)) = marker else {
        panic!("published wrapper has no marker field; fields={fields:?}");
    };
    assert!(
        ty.contains("PhantomData") && ty.contains("Path"),
        "jit published marker stays Path-form PhantomData; fields={fields:?}"
    );
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
    let context = linked_context(llbc);
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
    let context = linked_context(llbc);
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
        if calls_to(&graph, "root_scope_close_word") > 0 {
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
        .map(|llbc| (llbc, linked_context(llbc)))
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
                        if segments.last().is_some_and(|s| s == "root_scope_close_word"))
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
        "root_scope_close_word",
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
        "root_scope_close_word",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "ll_listslice_inner still calls {leaf} after the bracket was scalar-replaced"
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
        "root_scope_close_word",
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
        "root_scope_close_word",
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
        "root_scope_close_word",
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
    let context = context_with_extras(&interpreter, &[&object]);
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
        "root_scope_close_word",
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
    let context = context_with_extras(&interpreter, &[&object]);
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
        "root_scope_close_word",
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
        "shadow_stack_set",
        "root_scope_close_word",
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
        "root_scope_close_word",
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
    let context = context_with_extras(&interpreter, &[&object]);
    let graph = lower_fun(&interpreter, &context, "slice_unpack");
    for leaf in [
        "push_roots",
        "pin_root",
        "pin_roots",
        "shadow_stack_len",
        "shadow_stack_get",
        "root_scope_close_word",
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
    let context = context_with_extras(&module, &[&object, &interpreter]);
    let graph = lower_fun(&module, &context, "__majit_wrap_cdata_call");
    for leaf in [
        "push_roots",
        "pin_roots",
        "pin_root",
        "shadow_stack_get",
        "shadow_stack_copy_range",
        "root_scope_close_word",
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

/// `do_call` `n == 1` pins `args_w[0]` with `base()` + `pin_root` +
/// `get(base)`. RootBracketPlan rewrites that as `getarrayitem` of the
/// incoming list. `n > 1` `pin_roots(args_w)` aliases the incoming
/// slice, but the body also calls `convert_argument_from_object`, whose
/// pointer arm `pin_roots`s a local `Vec` (`pointer_convert_argument_from_object`).
/// That callee LeavesAbove, so whole-body erase refuses
/// (`refuses_erase`); the n>1 bracket stays.
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
    let context = context_with_extras(&module, &[&object, &interpreter]);
    let graph = lower_fun(&module, &context, "do_call");
    assert_eq!(
        calls_to(&graph, "push_roots"),
        1,
        "n==1 push_roots is erased; n>1 stays (LeavesAbove convert_argument_from_object)"
    );
    assert_eq!(
        calls_to(&graph, "pin_roots"),
        1,
        "n>1 pin_roots(args_w) stays with that bracket"
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
        "n==1 get(base) must rewrite to getarrayitem of args_w"
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
    let context = context_with_extras(&interpreter, &[&object]);
    let graph = lower_fun(&interpreter, &context, "baseobjspace::isinstance");
    for leaf in [
        "push_roots",
        "pin_roots",
        "pin_root",
        "shadow_stack_get",
        "root_scope_close_word",
    ] {
        assert_eq!(
            calls_to(&graph, leaf),
            0,
            "isinstance still calls {leaf} after the bracket was scalar-replaced"
        );
    }
}

/// CALL_KW binder bodies: `_match_signature` loses its interpreter root
/// brackets so the walker traces ordinary locals. `arguments_from_call_kw`
/// publishes the packed keyword-name `Vec` (`pin_roots(&names)`); that
/// buffer is not a GC object, so `pop_roots` / `expand_pop_roots`
/// (`gc_restore_root`) cannot reload it. Erase keeps that bracket.
#[test]
fn call_kw_binder_bodies_erase_root_brackets() {
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
    let context = context_with_extras(&interpreter, &[&object]);
    let names_erase = majit_translate::front::mir::shadow_stack_erase_named(
        &interpreter,
        "::arguments_from_call_kw",
    );
    assert!(
        names_erase.starts_with("publish-slice-not-incoming"),
        "arguments_from_call_kw pin_roots(&names) is a local Vec, erase={names_erase}"
    );
    let erase =
        majit_translate::front::mir::shadow_stack_erase_named(&interpreter, "::match_signature");
    assert_eq!(erase, "erased", "match_signature erase={erase}");
    let graph = lower_fun(&interpreter, &context, "match_signature");
    let repeat_tys: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty,
                ..
            } if segments.as_slice() == ["__array_repeat"] => Some(result_ty.clone()),
            _ => None,
        })
        .collect();
    assert!(!repeat_tys.is_empty(), "cap arm is [0i64; N] Repeat");
    for ty in &repeat_tys {
        assert!(
            matches!(ty, ValueType::Ref(Some(owner)) if owner.starts_with("Array<i64;")),
            "Repeat dest_ty stamps Array<i64;N>, got {ty:?}"
        );
    }
    let mut cc = CallControl::new();
    let config = GraphTransformConfig::default();
    let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
    let out = transformer.transform(&graph);
    let leftover = out
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.as_slice() == ["__array_repeat"] => true,
            OpKind::ConstFnAddr { path, .. } if path == "__array_repeat" => true,
            _ => false,
        });
    assert!(
        !leftover,
        "jtransform rewrites zero Repeat to NewArrayClear"
    );
    assert!(
        out.graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::NewArrayClear {
                        array_type_id: Some(id),
                        ..
                    } if id == "[i64]"
                )
            }),
        "cap arm [0i64; N] is NewArrayClear of [i64]"
    );
    assert_eq!(
        calls_to(&graph, "push_roots"),
        0,
        "match_signature push_roots"
    );
    assert_eq!(
        calls_to(&graph, "pin_roots"),
        0,
        "match_signature pin_roots"
    );
    assert_eq!(calls_to(&graph, "pin_root"), 0, "match_signature pin_root");
    assert_eq!(
        calls_to(&graph, "pin_root_gcarray"),
        0,
        "match_signature pin_root_gcarray"
    );
    assert_eq!(
        calls_to(&graph, "shadow_stack_len"),
        0,
        "match_signature len"
    );
    assert_eq!(
        calls_to(&graph, "shadow_stack_get"),
        0,
        "match_signature get"
    );
    assert_eq!(
        calls_to(&graph, "shadow_stack_set"),
        0,
        "match_signature set"
    );
    assert_eq!(
        calls_to(&graph, "shadow_stack_copy_range"),
        0,
        "match_signature copy_range"
    );
    assert_eq!(
        calls_to(&graph, "root_scope_close_word"),
        0,
        "match_signature close"
    );
}

fn int_family(ty: &ValueType) -> bool {
    matches!(
        ty,
        ValueType::Int
            | ValueType::Unsigned
            | ValueType::Bool
            | ValueType::Int128
            | ValueType::UInt128
    )
}

fn defining_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| (op.result.as_ref() == Some(var)).then_some(&op.kind))
    })
}

fn root_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    let mut current = var.clone();
    for _ in 0..32 {
        if let Some(op) = defining_op(graph, &current) {
            return Some(op);
        }
        let mut next = None;
        for block in &graph.blocks {
            let Some(slot) = block.inputargs.iter().position(|arg| arg == &current) else {
                continue;
            };
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target != block.id {
                        continue;
                    }
                    if let Some(arg) = exit.args.get(slot).and_then(|a| a.as_variable()) {
                        next = Some(arg.clone());
                        break;
                    }
                }
                if next.is_some() {
                    break;
                }
            }
            if next.is_some() {
                break;
            }
        }
        current = next?;
    }
    None
}

fn op_kind_label(op: &OpKind) -> String {
    match op {
        OpKind::Input {
            name,
            ty,
            class_root,
        } => {
            format!("Input({name}, {ty:?}, class={class_root:?})")
        }
        OpKind::ConstInt(n) => format!("ConstInt({n})"),
        OpKind::ConstUInt(n) => format!("ConstUInt({n})"),
        OpKind::ConstBool(b) => format!("ConstBool({b})"),
        OpKind::FieldRead { field, ty, .. } => {
            format!(
                "FieldRead({}.{}, {ty:?})",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name
            )
        }
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => format!("Call({})", segments.join("::")),
        other => format!("{other:?}").chars().take(120).collect::<String>(),
    }
}

/// `_match_signature` is walked as an orthodox helper with a hardcoded
/// 6-ref + 1-int packing (`MATCH_SIGNATURE_DESCENT`). A transparent-scalar
/// peel that moves a parameter into the other bank, or a `getfield(ob_type)`
/// whose base is an Int word (RootScope save_point, a bool flag), is the
/// GuardClass-at-0x1 crash. Production lowering publishes RootScope from
/// object into interpreter, so this graph uses that same universe.
#[test]
fn match_signature_input_banks_and_guard_class_bases() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let mut interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let published = harvest_published_structs(&object);
    apply_published_structs(&mut interpreter, &published);
    let touching = majit_translate::front::mir::harvest_root_stack_touching_paths(&object);
    interpreter.set_root_stack_effects(vec![object.crate_name().to_string()], touching);
    let context = context_with_extras(&interpreter, &[&object]);
    let graph = lower_fun(&interpreter, &context, "match_signature");

    let start = graph.block(graph.startblock);
    let mut inputs: Vec<(String, ValueType, Option<String>)> = Vec::new();
    for arg in &start.inputargs {
        let found = start.operations.iter().find_map(|op| match &op.kind {
            OpKind::Input {
                name,
                ty,
                class_root,
            } if op.result.as_ref() == Some(arg) => {
                Some((name.clone(), ty.clone(), class_root.clone()))
            }
            _ => None,
        });
        let (name, ty, class_root) =
            found.unwrap_or_else(|| panic!("startblock inputarg {} has no Input op", arg.name()));
        inputs.push((name, ty, class_root));
    }
    let n_ref = inputs
        .iter()
        .filter(|(_, ty, _)| {
            matches!(
                ty,
                ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder
            )
        })
        .count();
    let n_int = inputs.iter().filter(|(_, ty, _)| int_family(ty)).count();
    let dump_inputs: Vec<String> = inputs
        .iter()
        .map(|(n, t, c)| format!("{n}: {t:?} class={c:?}"))
        .collect();

    let mut flag_reads = Vec::new();
    for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
        if let OpKind::FieldRead {
            base, field, ty, ..
        } = &op.kind
            && (field.name == "jit_few_keywords" || field.name == "methodcall")
        {
            let producer = root_op(&graph, base)
                .map(op_kind_label)
                .unwrap_or_else(|| format!("unrooted {}", base.name()));
            flag_reads.push((field.name.clone(), ty.clone(), producer));
        }
    }

    let mut cc = CallControl::new();
    let config = GraphTransformConfig::default();
    let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
    let out = transformer.transform(&graph);

    let mut guard_class = Vec::new();
    for op in out.graph.blocks.iter().flat_map(|b| b.operations.iter()) {
        if let OpKind::GuardClass { base } = &op.kind {
            let producer = root_op(&out.graph, base)
                .map(op_kind_label)
                .unwrap_or_else(|| format!("unrooted {}", base.name()));
            guard_class.push(producer);
        }
    }

    let summary = format!(
        "inputs({} ref, {} int): {}\nflag_reads: {flag_reads:?}\nguard_class: {guard_class:?}",
        n_ref,
        n_int,
        dump_inputs.join(" | "),
    );

    assert_eq!(n_ref, 6, "MATCH_SIGNATURE_DESCENT packs 6 refs; {summary}");
    assert_eq!(
        n_int, 1,
        "MATCH_SIGNATURE_DESCENT packs 1 int (blindargs); {summary}"
    );
    for (name, ty, _) in &flag_reads {
        assert!(
            int_family(ty),
            "{name} FieldRead is {ty:?}, would seed const_int into a Ref getfield; {summary}"
        );
    }
    for producer in &guard_class {
        let int_base = producer.starts_with("Input(")
            && (producer.contains("Int")
                || producer.contains("Unsigned")
                || producer.contains("Bool"))
            || producer.starts_with("ConstInt")
            || producer.starts_with("ConstUInt")
            || producer.starts_with("ConstBool");
        assert!(
            !int_base,
            "GuardClass base is Int-produced ({producer}); {summary}"
        );
    }

    let collect = fun_decl_named(&interpreter, "_orig_collect_keyword_args");
    let collect_graph = lower_fun_decl(&context, collect)
        .unwrap_or_else(|e| panic!("lower _orig_collect_keyword_args: {e:?}"));
    let mut push_roots_ty = None;
    let mut close_word = 0usize;
    for op in collect_graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
    {
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty,
            ..
        } = &op.kind
        else {
            continue;
        };
        match segments.last().map(String::as_str) {
            Some("push_roots") => push_roots_ty = Some(result_ty.clone()),
            Some("root_scope_close_word") => close_word += 1,
            _ => {}
        }
    }
    assert!(
        matches!(push_roots_ty, Some(ValueType::Unsigned | ValueType::Int)),
        "RootScope peel keeps push_roots as the save_point word, got {push_roots_ty:?}"
    );
    assert!(
        close_word > 0,
        "_orig_collect_keyword_args must close the guard as a word"
    );
}

/// `_check_not_duplicate_kwargs` copies the published existing-names
/// slice into `existing_now`, a separately allocated buffer.
/// `pop_roots` / `expand_pop_roots` (`gc_restore_root`) reloads
/// livevars; a no-op copy would leave that buffer holding stale words.
/// The public `look_inside_iff` name is a trampoline; the rooted body
/// is `_orig_check_not_duplicate_kwargs`.
#[test]
fn check_not_duplicate_kwargs_keeps_copy_into_existing_now() {
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
    let erase = majit_translate::front::mir::shadow_stack_erase_named(
        &interpreter,
        "::_orig_check_not_duplicate_kwargs",
    );
    assert!(
        erase.starts_with("copy-range-dest-not-published-slice"),
        "existing_now is a distinct buffer, erase={erase}"
    );
}

/// `match_keywords` stores into the cap mapping must use the same
/// length-prefixed `[i64]` identity as `match_signature`'s Repeat /
/// fill. A headerless `[i64; N]` write is a different descr, so the
/// walker's heapcache keeps the `-1` init and the compiled CALL_KW
/// loop fills the default instead of the keyword.
#[test]
fn match_keywords_mapping_stores_share_signature_array_identity() {
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
    let context = context_with_extras(&interpreter, &[&object]);
    let graph = lower_fun(&interpreter, &context, "_orig_match_keywords");
    let writes: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayWrite {
                array_type_id,
                nolength,
                item_ty: ValueType::Int,
                ..
            } => Some((
                array_type_id.clone().unwrap_or_else(|| "<none>".into()),
                *nolength,
            )),
            _ => None,
        })
        .collect();
    assert!(
        !writes.is_empty(),
        "_orig_match_keywords stores into the cap mapping"
    );
    for (id, nolength) in &writes {
        assert_eq!(id, "[i64]", "mapping store identity {id}");
        assert!(!nolength, "mapping store {id} is length-prefixed");
    }
}
