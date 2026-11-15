//! Erase a GC-root registration guard whose slot lives in a local aggregate.
//!
//! Upstream never puts root registration in a jitcode. `PyFrame.__init__`
//! (`pypy/interpreter/pyframe.py`) builds a GC object; root slots are
//! emitted by `ShadowStackFrameworkGCTransformer.push_roots` /
//! `pop_roots` (`rpython/memory/gctransform/shadowstack.py`) after
//! `warmspot.py` `make_jitcodes`. pyre spells the same span as a RAII
//! guard that calls `gc_hook::try_gc_add_root` / `try_gc_remove_root`.
//! When the translator scalar-replaces the by-value aggregate
//! (`scalar_replace_named_struct_aggregates`), the slot is already an SSA
//! Ref rooted by the jitcode's own root sets (jitframe gcmap, blackhole
//! `registers_r`, recorder) — the argument [`super::shadow_stack_erase`]
//! makes for `RootScope`. The opener is then not a jitcode op, and its
//! `Drop` is already `Goto` (`emit_place_drop` returns `Ok(false)`).
//! Erasing the opener closes that register-without-unregister
//! asymmetry. A guard over a pointer that came from the heap stays.
//!
//! Sibling of [`super::shadow_stack_erase`] and [`super::owner_root_guard`].

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, CallKind, CallPayload, FunDecl, FunId, NameSeg, Operand, Place, PlaceKind,
    ProjectionElem, Rvalue, StmtKind, TermKind, TyRef, TypeDeclKind, Unstructured,
};

/// Per-crate memo of [`root_registration_effect`], keyed by FunDecl id
/// (`GraphAnalyzer._analyzed_calls`). `Mutex` so [`super::LowerContext`]
/// (shared across `lower_fun_decl` workers, `OnceLock` in tests) stays `Sync`.
pub(super) type EffectMemo = Mutex<HashMap<u64, Option<RootRegistration>>>;

const GC_HOOK_MODULE: &str = "gc_hook";
const ADD_ROOT_LEAF: &str = "try_gc_add_root";
const REMOVE_ROOT_LEAF: &str = "try_gc_remove_root";
const ICF_IDENTITY_TOKEN: &str = "majit_ir::icf::icf_identity_token";
const DROP_TRAIT: &str = "core::ops::drop::Drop";
const EFFECT_DEPTH_LIMIT: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RootRegistration {
    Add,
    Remove,
}

/// Direct-call effect of `fun_id` (`graphanalyze.py`
/// `GraphAnalyzer.analyze_direct_call`). `Some(k)` when the body is the
/// `gc_hook` primitive of kind `k`, or every `Call` has that same effect
/// (at least one) and statements write only body locals.
pub(super) fn root_registration_effect(
    fun_id: u64,
    llbc: &Llbc,
    cache: &EffectMemo,
) -> Option<RootRegistration> {
    analyze_effect(fun_id, llbc, cache, 0, &mut Vec::new())
}

/// Effect of the type's `Drop::drop` impl. `None` when the type has no
/// Drop impl or when any field of it needs drop glue.
pub(super) fn drop_glue_effect(
    ty: &TyRef,
    llbc: &Llbc,
    cache: &EffectMemo,
) -> Option<RootRegistration> {
    let adt_id = tyref_adt_id(ty, llbc)?;
    let decl = llbc.type_by_id(adt_id)?;
    if adt_fields_need_drop(decl, llbc, 0, &mut Vec::new()) {
        return None;
    }
    let drop_id = drop_method_id(llbc, adt_id)?;
    root_registration_effect(drop_id, llbc, cache)
}

/// The call's first argument is a borrow of a non-argument, non-pointer
/// local aggregate with no projection.
pub(super) fn slot_is_local_aggregate(
    body: &Unstructured,
    llbc: &Llbc,
    _mir_bb: usize,
    arg: &Operand,
) -> bool {
    let n = body.locals.locals.len();
    let defs = super::local_def_counts(body, llbc, n);
    follow_to_local_aggregate(arg, body, llbc, &defs, 0, &mut Vec::new())
}

/// Per-local `dest_only_dropped`: the local is never read, moved, borrowed
/// or projected; its only uses are `Drop` / `StorageLive` / `StorageDead`
/// of the bare local (plus the defining Call dest).
///
/// `drop(guard)` is `_t = Move(guard); core::mem::drop(Move(_t))`
/// (`explicit_drop_guard_local`). That chain is the same close as a
/// `Drop` terminator: the guard is destroyed and its value is unused.
/// `#276` `finish_for_call_with_globals_obj` spells the MemoryError
/// paths that way (`drop(_locals_root)`), so a local-aggregate
/// `FrameLocalsRoot` stays off the jitcode.
pub(super) fn locals_only_dropped(body: &Unstructured, llbc: &Llbc) -> Vec<bool> {
    let n = body.locals.locals.len();
    let mut only = vec![true; n];
    let drop_chain = mem_drop_move_chain_locals(body, llbc);
    let mark = |only: &mut [bool], local: Option<usize>| {
        if let Some(i) = local {
            if i < only.len() {
                only[i] = false;
            }
        }
    };
    for bb in &body.body {
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Borrowck(_)) => {
                }
                Ok(StmtKind::Assign(place, rvalue)) => {
                    if assign_is_mem_drop_move_step(place, rvalue, &drop_chain) {
                        continue;
                    }
                    mark(&mut only, place_base_local(place));
                    rvalue_mark_used(rvalue, &mut only);
                }
                Ok(StmtKind::PlaceMention(place)) => {
                    mark(&mut only, place_base_local(place));
                }
                Ok(StmtKind::Assert(assert)) => {
                    operand_mark_used(&assert.cond, &mut only);
                }
                Ok(StmtKind::Unknown) | Err(_) => {
                    unknown_stmt_mark_used(st.kind_value(), &mut only);
                }
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Drop { place, .. }) => {
                if !matches!(place.kind, PlaceKind::Local(_)) {
                    mark(&mut only, place_base_local(place));
                }
            }
            Ok(TermKind::Call { call, .. }) => {
                if !matches!(call.dest.kind, PlaceKind::Local(_)) {
                    mark(&mut only, place_base_local(&call.dest));
                }
                if mem_drop_operand_local(call, llbc).is_none() {
                    for op in &call.args {
                        operand_mark_used(op, &mut only);
                    }
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                operand_mark_used(discr, &mut only);
            }
            Ok(TermKind::Assert { assert, .. }) => {
                operand_mark_used(&assert.cond, &mut only);
            }
            Ok(_) => {}
            Err(_) => {
                only.fill(false);
                return only;
            }
        }
    }
    only
}

/// `core::mem::drop` / `core::ptr::drop_in_place` of a local
/// (`shadow_stack_erase` `mem_drop_scope_local`).
pub(super) fn mem_drop_operand_local(call: &CallPayload, llbc: &Llbc) -> Option<usize> {
    let id = call_fun_id(call)?;
    let path = llbc.fn_by_id(id)?.item_meta.name_path_str();
    if path != "core::mem::drop" && path != "core::ptr::drop_in_place" {
        return None;
    }
    if call.args.len() != 1 {
        return None;
    }
    match &call.args[0] {
        Operand::Move(place) | Operand::Copy(place) => match place.kind {
            PlaceKind::Local(local) => Some(local as usize),
            _ => None,
        },
        Operand::Const(_) => None,
    }
}

/// Locals on a same-block `_t = Move(guard); core::mem::drop(Move(_t))`
/// chain (`explicit_drop_guard_local`).
fn mem_drop_move_chain_locals(body: &Unstructured, llbc: &Llbc) -> HashSet<usize> {
    mem_drop_chains(body, llbc).into_iter().flatten().collect()
}

fn mem_drop_chains(body: &Unstructured, llbc: &Llbc) -> Vec<HashSet<usize>> {
    let mut chains = Vec::new();
    for bb in &body.body {
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let Some(mut cur) = mem_drop_operand_local(call, llbc) else {
            continue;
        };
        let mut chain = HashSet::new();
        chain.insert(cur);
        for stmt in bb.statements.iter().rev() {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(d) if d as usize == cur) {
                continue;
            }
            let Rvalue::Use(Operand::Move(src), _) = rvalue else {
                break;
            };
            let PlaceKind::Local(src) = src.kind else {
                break;
            };
            cur = src as usize;
            chain.insert(cur);
        }
        chains.push(chain);
    }
    chains
}

/// Dest locals whose opener [`super::Lowering::lower_erased_gc_root_registration_call`]
/// erases, plus the `mem::drop` temps of those dests. Stripped from
/// `block_live_in` the way `RootBracketPlan::scopes` are: the guard binds
/// no Variable, so no edge may pass one.
pub(super) fn erased_registration_locals(
    body: &Unstructured,
    llbc: &Llbc,
    cache: &EffectMemo,
) -> HashSet<usize> {
    let only = locals_only_dropped(body, llbc);
    let chains = mem_drop_chains(body, llbc);
    let mut out = HashSet::new();
    for (mir_bb, bb) in body.body.iter().enumerate() {
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        let dest = dest as usize;
        if !registration_call_is_erased(body, llbc, mir_bb, call, dest, cache, &only) {
            continue;
        }
        out.insert(dest);
        for chain in &chains {
            if chain.contains(&dest) {
                out.extend(chain.iter().copied());
            }
        }
    }
    out
}

fn registration_call_is_erased(
    body: &Unstructured,
    llbc: &Llbc,
    mir_bb: usize,
    call: &CallPayload,
    dest_local: usize,
    cache: &EffectMemo,
    only_dropped: &[bool],
) -> bool {
    if !only_dropped.get(dest_local).copied().unwrap_or(false) {
        return false;
    }
    let Some(fun_id) = call_fun_id(call) else {
        return false;
    };
    let Some(first) = call.args.first() else {
        return false;
    };
    if !slot_is_local_aggregate(body, llbc, mir_bb, first) {
        return false;
    }
    if root_registration_effect(fun_id, llbc, cache) != Some(RootRegistration::Add) {
        return false;
    }
    drop_glue_effect(&call.dest.ty, llbc, cache) == Some(RootRegistration::Remove)
}

fn assign_is_mem_drop_move_step(
    place: &Place,
    rvalue: &Rvalue,
    drop_chain: &HashSet<usize>,
) -> bool {
    let PlaceKind::Local(dest) = place.kind else {
        return false;
    };
    if !drop_chain.contains(&(dest as usize)) {
        return false;
    }
    let Rvalue::Use(Operand::Move(src), _) = rvalue else {
        return false;
    };
    matches!(src.kind, PlaceKind::Local(src) if drop_chain.contains(&(src as usize)))
}

pub(super) fn call_fun_id(call: &CallPayload) -> Option<u64> {
    match &call.func {
        CallFunc::Regular(reg) => match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => Some(*id),
            _ => None,
        },
        CallFunc::Dynamic(..) | CallFunc::Unknown => None,
    }
}

fn analyze_effect(
    fun_id: u64,
    llbc: &Llbc,
    cache: &EffectMemo,
    depth: u32,
    visiting: &mut Vec<u64>,
) -> Option<RootRegistration> {
    if depth > EFFECT_DEPTH_LIMIT {
        return None;
    }
    if let Some(cached) = memo_get(cache, fun_id) {
        return cached;
    }
    if visiting.contains(&fun_id) {
        return None;
    }
    visiting.push(fun_id);
    let result = analyze_effect_uncached(fun_id, llbc, cache, depth, visiting);
    visiting.pop();
    memo_insert(cache, fun_id, result);
    result
}

fn memo_get(cache: &EffectMemo, fun_id: u64) -> Option<Option<RootRegistration>> {
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&fun_id)
        .copied()
}

fn memo_insert(cache: &EffectMemo, fun_id: u64, result: Option<RootRegistration>) {
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(fun_id, result);
}

fn analyze_effect_uncached(
    fun_id: u64,
    llbc: &Llbc,
    cache: &EffectMemo,
    depth: u32,
    visiting: &mut Vec<u64>,
) -> Option<RootRegistration> {
    let fd = llbc.fn_by_id(fun_id)?;
    if let Some(kind) = primitive_effect(fd) {
        return Some(kind);
    }
    if is_erased_identity_token(fd) {
        return None;
    }
    let body = fd.unstructured()?;
    body_effect(&body, llbc, cache, depth, visiting)
}

fn body_effect(
    body: &Unstructured,
    llbc: &Llbc,
    cache: &EffectMemo,
    depth: u32,
    visiting: &mut Vec<u64>,
) -> Option<RootRegistration> {
    // `GraphAnalyzer.analyze_direct_call` walks each block's operations
    // and `block.exits`. Charon `on_unwind` is not an RPython exit
    // (`successor_blocks`); a Panic/Assert/Abort/UB on a normal edge is.
    // Local rejection first: a Drop, non-local dest, or unknown terminator
    // here is already `None`; walking callees first would re-analyze the
    // same non-wrapper bodies from every caller.
    let forward = super::forward_reachable_mask(llbc, body);
    let mut callees: Vec<u64> = Vec::new();
    for (i, bb) in body.body.iter().enumerate() {
        if !forward.get(i).copied().unwrap_or(false) {
            continue;
        }
        if !statements_write_only_locals(bb) {
            return None;
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Drop { place, .. }) => {
                // A Drop of a type with no glue is not a registration
                // effect. `OwnerRootGuard` / `Option<OwnerRootGuard>`
                // (`tyref_is_erased_drop`) are that no-op: the pin field
                // of `FrameLocalsRoot` after #2628.
                if ty_needs_drop_glue(&place.ty, llbc, 0, &mut Vec::new()) {
                    return None;
                }
            }
            Ok(TermKind::Call { call, .. }) => {
                if !place_writes_only_local(&call.dest) {
                    return None;
                }
                let callee = call_fun_id(call)?;
                let fd = llbc.fn_by_id(callee)?;
                if call_is_registration_neutral(call, fd, llbc) {
                    continue;
                }
                callees.push(callee);
            }
            Ok(TermKind::Return | TermKind::Goto { .. } | TermKind::Switch { .. }) => {}
            Ok(
                TermKind::Assert { .. }
                | TermKind::Panic { .. }
                | TermKind::Abort(_)
                | TermKind::UndefinedBehavior
                | TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::Unknown,
            )
            | Err(_) => return None,
        }
    }
    if callees.is_empty() {
        return None;
    }
    let mut kind: Option<RootRegistration> = None;
    for callee in callees {
        let effect = analyze_effect(callee, llbc, cache, depth + 1, visiting)?;
        match kind {
            None => kind = Some(effect),
            Some(k) if k == effect => {}
            Some(_) => return None,
        }
    }
    kind
}

fn statements_write_only_locals(bb: &majit_charon_reader::ullbc::BasicBlock) -> bool {
    for st in &bb.statements {
        match st.stmt_kind_ref() {
            Ok(StmtKind::Assign(place, _)) => {
                if !place_writes_only_local(place) {
                    return false;
                }
            }
            Ok(
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::Borrowck(_)
                | StmtKind::PlaceMention(_),
            ) => {}
            Ok(StmtKind::Assert(_)) => return false,
            Ok(StmtKind::Unknown) | Err(_) => {
                if !unknown_stmt_writes_only_locals(st.kind_value()) {
                    return false;
                }
            }
        }
    }
    true
}

fn unknown_stmt_writes_only_locals(kind: &serde_json::Value) -> bool {
    // `RaiseAnalyzer.analyze_simple_operation`: an unknown operation is
    // treated as raising. Only SetDiscriminant / Deinit of a local dest
    // are admitted; every other statement kind is an effect.
    if let Some(place) = set_discriminant_place(kind) {
        return place_writes_only_local(&place);
    }
    if let Some(place) = deinit_place(kind) {
        return place_writes_only_local(&place);
    }
    false
}

fn unknown_stmt_mark_used(kind: &serde_json::Value, only: &mut [bool]) {
    if let Some(place) = set_discriminant_place(kind) {
        if let Some(i) = place_base_local(&place) {
            if i < only.len() {
                only[i] = false;
            }
        }
    }
    if let Some(place) = deinit_place(kind) {
        if let Some(i) = place_base_local(&place) {
            if i < only.len() {
                only[i] = false;
            }
        }
    }
}

fn set_discriminant_place(kind: &serde_json::Value) -> Option<Place> {
    let arr = kind.get("SetDiscriminant")?.as_array()?;
    serde_json::from_value(arr.first()?.clone()).ok()
}

fn deinit_place(kind: &serde_json::Value) -> Option<Place> {
    serde_json::from_value(kind.get("Deinit")?.clone()).ok()
}

fn primitive_effect(fd: &FunDecl) -> Option<RootRegistration> {
    let path = fd.item_meta.name_path_str();
    let mut has_hook = false;
    let mut leaf = "";
    for seg in path.split("::") {
        if seg == GC_HOOK_MODULE {
            has_hook = true;
        }
        leaf = seg;
    }
    if !has_hook {
        return None;
    }
    match leaf {
        ADD_ROOT_LEAF => Some(RootRegistration::Add),
        REMOVE_ROOT_LEAF => Some(RootRegistration::Remove),
        _ => None,
    }
}

/// `Lowering::is_icf_identity_token`: empty `nomem` asm, erased before
/// a jitcode op is emitted. It is not a registration call.
fn is_erased_identity_token(fd: &FunDecl) -> bool {
    fd.item_meta.name_path_str() == ICF_IDENTITY_TOKEN
}

/// Calls that are not a registration effect: identity tokens, erased
/// `OwnerRootGuard` ops (the pin of `FrameLocalsRoot`), and residual
/// bool queries (`ptr::is_null`, an Opaque `bool` dest). Remaining
/// callees still have to share one Add/Remove effect.
fn call_is_registration_neutral(call: &CallPayload, fd: &FunDecl, llbc: &Llbc) -> bool {
    if is_erased_identity_token(fd) {
        return true;
    }
    if super::owner_root_guard::classify(call, llbc).is_some() {
        return true;
    }
    if primitive_effect(fd).is_some() {
        return false;
    }
    fd.unstructured().is_none() && dest_is_bool(&call.dest.ty, llbc)
}

fn dest_is_bool(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) =
        super::tyref_node(ty, llbc).and_then(|n| super::strip_ty_indirections(n, llbc))
    else {
        return false;
    };
    match node.get("Scalar") {
        Some(serde_json::Value::String(s)) => s == "Bool",
        Some(serde_json::Value::Object(map)) => map.contains_key("Bool"),
        _ => false,
    }
}

fn follow_to_local_aggregate(
    op: &Operand,
    body: &Unstructured,
    llbc: &Llbc,
    defs: &[usize],
    depth: u32,
    visiting: &mut Vec<usize>,
) -> bool {
    if depth > 16 {
        return false;
    }
    let place = match op {
        Operand::Copy(p) | Operand::Move(p) => p,
        Operand::Const(_) => return false,
    };
    let PlaceKind::Local(l) = place.kind else {
        return false;
    };
    let l = l as usize;
    if visiting.contains(&l) {
        return false;
    }
    if defs.get(l).copied() != Some(1) {
        return false;
    }
    visiting.push(l);
    let ok = match unique_assign_rvalue(body, llbc, l) {
        Some(Rvalue::Use(next, _)) => {
            follow_to_local_aggregate(next, body, llbc, defs, depth + 1, visiting)
        }
        Some(Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. }) => {
            match &place.kind {
                PlaceKind::Local(_) => ends_at_local_aggregate(place, body, llbc),
                // `&raw mut (*p)` / `&mut *p`: the Deref is a reborrow of a
                // pointer temporary. Follow `p`; a heap pointer or argument
                // still fails at that local's unique def.
                PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) if elem == "Deref" => {
                    let next = Operand::Copy((**inner).clone());
                    follow_to_local_aggregate(&next, body, llbc, defs, depth + 1, visiting)
                }
                _ => false,
            }
        }
        Some(Rvalue::Cast(_, next, dest_ty)) => {
            let src_ty = super::operand_tyref(next);
            src_ty.is_some_and(|src| ty_is_pointer(src, llbc))
                && ty_is_pointer(dest_ty, llbc)
                && follow_to_local_aggregate(next, body, llbc, defs, depth + 1, visiting)
        }
        _ => false,
    };
    visiting.pop();
    ok
}

fn ends_at_local_aggregate(place: &Place, body: &Unstructured, llbc: &Llbc) -> bool {
    let PlaceKind::Local(base) = place.kind else {
        return false;
    };
    let base = base as usize;
    let arg_count = body.locals.arg_count as usize;
    base > arg_count
        && body
            .locals
            .locals
            .get(base)
            .is_some_and(|local| !ty_is_pointer(&local.ty, llbc))
}

fn unique_assign_rvalue<'a>(
    body: &'a Unstructured,
    llbc: &Llbc,
    local: usize,
) -> Option<&'a Rvalue> {
    let forward = super::forward_reachable_mask(llbc, body);
    let mut found = None;
    for (i, bb) in body.body.iter().enumerate() {
        if !forward.get(i).copied().unwrap_or(false) {
            continue;
        }
        for st in &bb.statements {
            if let Ok(StmtKind::Assign(place, rv)) = st.stmt_kind_ref()
                && matches!(place.kind, PlaceKind::Local(l) if l as usize == local)
            {
                if found.is_some() {
                    return None;
                }
                found = Some(rv);
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && matches!(call.dest.kind, PlaceKind::Local(l) if l as usize == local)
        {
            return None;
        }
    }
    found
}

fn drop_method_id(llbc: &Llbc, adt_id: u64) -> Option<u64> {
    llbc.iter_fun_decls().find_map(|fd| {
        (fn_leaf(fd) == Some("drop")
            && impl_self_id(fd, llbc) == Some(adt_id)
            && impl_is_drop_trait(fd, llbc))
        .then_some(fd.def_id)
    })
}

fn impl_is_drop_trait(fd: &FunDecl, llbc: &Llbc) -> bool {
    let Some(last_ident) = fd
        .item_meta
        .name
        .iter()
        .rposition(|seg| matches!(seg, NameSeg::Ident { .. }))
    else {
        return false;
    };
    if last_ident == 0 {
        return false;
    }
    let Some(trait_impl_id) = fd.item_meta.name[last_ident - 1].impl_trait_id() else {
        return false;
    };
    let Some(ti) = llbc.trait_impls_raw().get(trait_impl_id as usize) else {
        return false;
    };
    let Some(trait_id) = ti
        .get("impl_trait")
        .and_then(|t| t.get("id"))
        .and_then(serde_json::Value::as_u64)
    else {
        return false;
    };
    llbc.trait_by_id(trait_id)
        .is_some_and(|td| td.item_meta.name_path_str() == DROP_TRAIT)
}

fn impl_self_id(fd: &FunDecl, llbc: &Llbc) -> Option<u64> {
    let segs = &fd.item_meta.name;
    let last_ident = segs
        .iter()
        .rposition(|seg| matches!(seg, NameSeg::Ident { .. }))?;
    if last_ident == 0 {
        return None;
    }
    let impl_seg = &segs[last_ident - 1];
    if !impl_seg.is_impl() {
        return None;
    }
    super::resolve_impl_owner_adt_def_id_free(llbc, impl_seg)
}

fn fn_leaf(fd: &FunDecl) -> Option<&str> {
    match fd.item_meta.name.last()? {
        NameSeg::Ident { ident: (leaf, _) } => Some(leaf.as_str()),
        NameSeg::Builtin { kind, .. } => Some(kind.as_str()),
        _ => None,
    }
}

fn tyref_adt_id(ty: &TyRef, llbc: &Llbc) -> Option<u64> {
    let node = super::tyref_node(ty, llbc)?;
    let node = super::strip_ty_indirections(node, llbc)?;
    super::adt_node_def_id(node)
}

fn adt_fields_need_drop(
    decl: &majit_charon_reader::ullbc::TypeDecl,
    llbc: &Llbc,
    depth: usize,
    seen: &mut Vec<u64>,
) -> bool {
    match &decl.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => fields
            .iter()
            .any(|field| ty_needs_drop_glue(&field.ty, llbc, depth + 1, seen)),
        TypeDeclKind::Enum(variants) => variants.iter().any(|variant| {
            variant
                .fields
                .iter()
                .any(|field| ty_needs_drop_glue(&field.ty, llbc, depth + 1, seen))
        }),
        TypeDeclKind::Alias(_) | TypeDeclKind::Opaque | TypeDeclKind::Unknown => true,
    }
}

fn ty_needs_drop_glue(ty: &TyRef, llbc: &Llbc, depth: usize, seen: &mut Vec<u64>) -> bool {
    if depth > 16 {
        return true;
    }
    // The pin is an erased GCREF word (`owner_root_guard` Drop is
    // nothing). It must not make `drop_glue_effect` refuse the
    // `FrameLocalsRoot` Drop that is `try_gc_remove_root`.
    if super::owner_root_guard::tyref_is_erased_drop(ty, llbc) {
        return false;
    }
    let Some(node) =
        super::tyref_node(ty, llbc).and_then(|n| super::strip_ty_indirections(n, llbc))
    else {
        return true;
    };
    if node.get("Scalar").is_some()
        || node.get("Ref").is_some()
        || node.get("RawPtr").is_some()
        || node.get("FnDef").is_some()
        || node.get("FnPtr").is_some()
        || node.get("Never").is_some()
    {
        return false;
    }
    let Some(id) = super::adt_node_def_id(node) else {
        return true;
    };
    if seen.contains(&id) {
        return false;
    }
    if llbc.has_explicit_drop_impl(id, |expr| {
        super::resolve_tyexpr_to_adt_def_id_free(llbc, expr)
    }) {
        return true;
    }
    let Some(decl) = llbc.type_by_id(id) else {
        return true;
    };
    seen.push(id);
    let needs = adt_fields_need_drop(decl, llbc, depth, seen);
    seen.pop();
    needs
}

fn ty_is_pointer(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) =
        super::tyref_node(ty, llbc).and_then(|n| super::strip_ty_indirections(n, llbc))
    else {
        return false;
    };
    node.get("Ref").is_some() || node.get("RawPtr").is_some()
}

fn place_base_local(place: &Place) -> Option<usize> {
    let mut p = place;
    loop {
        match &p.kind {
            PlaceKind::Local(i) => return Some(*i as usize),
            PlaceKind::Projection(inner, _) => p = inner,
            PlaceKind::Global { .. } | PlaceKind::Unknown => return None,
        }
    }
}

fn place_has_deref(place: &Place) -> bool {
    let mut p = place;
    loop {
        match &p.kind {
            PlaceKind::Projection(inner, elem) => {
                if matches!(elem, ProjectionElem::Atom(s) if s == "Deref") {
                    return true;
                }
                p = inner;
            }
            _ => return false,
        }
    }
}

/// Assignment dest whose base is a bare `PlaceKind::Local` with no
/// `Deref` and no Global/static projection.
fn place_writes_only_local(place: &Place) -> bool {
    place_base_local(place).is_some() && !place_has_deref(place)
}

fn operand_mark_used(op: &Operand, only: &mut [bool]) {
    match op {
        Operand::Copy(p) | Operand::Move(p) => {
            if let Some(i) = place_base_local(p) {
                if i < only.len() {
                    only[i] = false;
                }
            }
        }
        Operand::Const(_) => {}
    }
}

fn rvalue_mark_used(rv: &Rvalue, only: &mut [bool]) {
    match rv {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat { operand: op, .. }
        | Rvalue::ShallowInitBox(op, _) => operand_mark_used(op, only),
        Rvalue::BinaryOp(_, a, b) => {
            operand_mark_used(a, only);
            operand_mark_used(b, only);
        }
        Rvalue::Ref { place, .. }
        | Rvalue::RawPtr { place, .. }
        | Rvalue::Discriminant(place)
        | Rvalue::Len(place) => {
            if let Some(i) = place_base_local(place) {
                if i < only.len() {
                    only[i] = false;
                }
            }
        }
        Rvalue::Aggregate(_, ops) => {
            for op in ops {
                operand_mark_used(op, only);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::Llbc;

    fn span() -> serde_json::Value {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    }

    fn ident(name: &str) -> serde_json::Value {
        serde_json::json!({"Ident": [name, 0]})
    }

    fn item_meta(path: &[&str], is_local: bool) -> serde_json::Value {
        serde_json::json!({
            "name": path.iter().map(|s| ident(s)).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    }

    fn unit_ty() -> serde_json::Value {
        serde_json::json!({
            "Adt": {
                "id": 0,
                "builtin": "Tuple",
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    }

    fn bool_ty() -> serde_json::Value {
        serde_json::json!({"Scalar": "Bool"})
    }

    fn i64_ty() -> serde_json::Value {
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})
    }

    fn local(index: u64, ty: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "index": index,
            "name": null,
            "span": span(),
            "ty": ty
        })
    }

    fn place_local(index: u64, ty: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({"kind": {"Local": index}, "ty": ty})
    }

    fn stmt(kind: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"span": span(), "kind": kind})
    }

    fn block(
        statements: Vec<serde_json::Value>,
        terminator: serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "statements": statements,
            "kind": "Regular",
            "terminator": {"span": span(), "kind": terminator}
        })
    }

    fn call_add_root(target: u64) -> serde_json::Value {
        let bool_ty = bool_ty();
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {
                    "kind": {"Fun": 1},
                    "generics": {
                        "regions": [],
                        "types": [],
                        "const_generics": [],
                        "trait_refs": []
                    }
                }},
                "args": []
            },
            "dest": place_local(1, &bool_ty),
            "target": target,
            "on_unwind": 0
        }})
    }

    fn call_mem_drop(operand_local: u64, ty: &serde_json::Value, target: u64) -> serde_json::Value {
        let unit = unit_ty();
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {
                    "kind": {"Fun": 2},
                    "generics": {
                        "regions": [],
                        "types": [],
                        "const_generics": [],
                        "trait_refs": []
                    }
                }},
                "args": [{"Move": place_local(operand_local, ty)}]
            },
            "dest": place_local(0, &unit),
            "target": target,
            "on_unwind": 0
        }})
    }

    fn wrapper_llbc(
        extra_locals: Vec<serde_json::Value>,
        blocks: Vec<serde_json::Value>,
        globals: Vec<serde_json::Value>,
    ) -> Llbc {
        wrapper_llbc_with_funs(extra_locals, blocks, globals, vec![])
    }

    fn wrapper_llbc_with_funs(
        extra_locals: Vec<serde_json::Value>,
        blocks: Vec<serde_json::Value>,
        globals: Vec<serde_json::Value>,
        extra_funs: Vec<serde_json::Value>,
    ) -> Llbc {
        let unit = unit_ty();
        let bool_ty = bool_ty();
        let mut locals = vec![local(0, &unit), local(1, &bool_ty)];
        locals.extend(extra_locals);
        let wrapper = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "wrap"], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": unit.clone()
            },
            "body": {"Unstructured": {
                "span": span(),
                "locals": {"arg_count": 0, "locals": locals},
                "body": blocks
            }}
        });
        let add_root = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta(
                &["pyre_object", "gc_hook", "try_gc_add_root"],
                false
            ),
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": bool_ty
            },
            "body": "Opaque"
        });
        let mem_drop = serde_json::json!({
            "def_id": 2,
            "item_meta": item_meta(&["core", "mem", "drop"], false),
            "signature": {
                "is_unsafe": false,
                "inputs": [unit.clone()],
                "output": unit.clone()
            },
            "body": "Opaque"
        });
        let mut fun_decls = vec![wrapper, add_root, mem_drop];
        fun_decls.extend(extra_funs);
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": fun_decls,
                "global_decls": globals,
                "trait_decls": [],
                "trait_impls": []
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("registration wrapper fixture parses")
    }

    fn wrapper_effect(
        extra_locals: Vec<serde_json::Value>,
        blocks: Vec<serde_json::Value>,
        globals: Vec<serde_json::Value>,
    ) -> Option<RootRegistration> {
        let llbc = wrapper_llbc(extra_locals, blocks, globals);
        let cache = Mutex::new(HashMap::new());
        root_registration_effect(0, &llbc, &cache)
    }

    fn i64_zero() -> serde_json::Value {
        let ty = i64_ty();
        serde_json::json!({"Const": [
            {"Integer": {"Signed": ["I64", "0"]}},
            ty
        ]})
    }

    /// A wrapper that only assigns a local and calls `try_gc_add_root`
    /// is `RootRegistration::Add` (`GraphAnalyzer.analyze_direct_call`
    /// of a body whose operations are local stores plus that call).
    #[test]
    fn local_only_wrapper_is_add() {
        let i64_ty = i64_ty();
        let effect = wrapper_effect(
            vec![local(2, &i64_ty)],
            vec![
                block(
                    vec![stmt(serde_json::json!({"Assign": [
                        place_local(2, &i64_ty),
                        {"Use": [i64_zero(), "No"]}
                    ]}))],
                    call_add_root(1),
                ),
                block(vec![], serde_json::json!("Return")),
            ],
            vec![],
        );
        assert_eq!(effect, Some(RootRegistration::Add));
    }

    /// `drop(guard)` is `_t = Move(guard); core::mem::drop(Move(_t))`.
    /// That chain is dest-only-dropped, the same close as a `Drop`
    /// terminator (`explicit_drop_guard_local`).
    #[test]
    fn mem_drop_move_chain_is_only_dropped() {
        let bool_ty = bool_ty();
        let llbc = wrapper_llbc(
            vec![local(2, &bool_ty)],
            vec![
                block(vec![], call_add_root(1)),
                block(
                    vec![stmt(serde_json::json!({"Assign": [
                        place_local(2, &bool_ty),
                        {"Use": [{"Move": place_local(1, &bool_ty)}, "No"]}
                    ]}))],
                    call_mem_drop(2, &bool_ty, 2),
                ),
                block(vec![], serde_json::json!("Return")),
            ],
            vec![],
        );
        let body = llbc
            .fn_by_id(0)
            .and_then(|fd| fd.unstructured())
            .expect("wrapper body");
        let only = locals_only_dropped(&body, &llbc);
        assert!(
            only.get(1).copied().unwrap_or(false),
            "add-root dest moved only into mem::drop: {only:?}"
        );
    }

    /// A store to a `PlaceKind::Global` is an effect, even when the
    /// rest of the body is a registration call.
    #[test]
    fn global_assignment_is_not_registration() {
        let i64_ty = i64_ty();
        let global_place = serde_json::json!({
            "kind": {"Global": {
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                },
                "id": 0
            }},
            "ty": i64_ty
        });
        let global = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "COUNTER"], true),
            "global_kind": "Static",
            "ty": i64_ty
        });
        let effect = wrapper_effect(
            vec![],
            vec![
                block(
                    vec![stmt(serde_json::json!({"Assign": [
                        global_place,
                        {"Use": [i64_zero(), "No"]}
                    ]}))],
                    call_add_root(1),
                ),
                block(vec![], serde_json::json!("Return")),
            ],
            vec![global],
        );
        assert_eq!(effect, None);
    }

    /// Terminator `Assert` can fail; `RaiseAnalyzer.analyze_simple_operation`
    /// would treat an unknown/raising op as top.
    #[test]
    fn failing_assertion_is_not_registration() {
        let bool_ty = bool_ty();
        let assert_term = serde_json::json!({"Assert": {
            "assert": {
                "cond": {"Read": place_local(1, &bool_ty)},
                "expected": true,
                "check_kind": "NullCheck"
            },
            "target": 1,
            "on_unwind": 0
        }});
        let effect = wrapper_effect(
            vec![],
            vec![block(vec![], call_add_root(1)), block(vec![], assert_term)],
            vec![],
        );
        assert_eq!(effect, None);
    }

    /// Panic / Abort / UndefinedBehavior on a normal edge are effects.
    #[test]
    fn panic_abort_ub_are_not_registration() {
        let panic_name = serde_json::json!([
            {"Ident": ["core", 0]},
            {"Ident": ["panicking", 0]},
            {"Ident": ["panic", 0]}
        ]);
        let terms = [
            serde_json::json!({"Panic": {"name": panic_name, "on_unwind": 0}}),
            serde_json::json!({"Abort": "UndefinedBehavior"}),
            serde_json::json!("UndefinedBehavior"),
        ];
        for term in terms {
            let effect = wrapper_effect(
                vec![],
                vec![block(vec![], call_add_root(1)), block(vec![], term.clone())],
                vec![],
            );
            assert_eq!(effect, None, "terminator {term} must refuse erasure");
        }
    }

    fn call_opaque_bool(fun_id: u64, dest_local: u64, target: u64) -> serde_json::Value {
        let bool_ty = bool_ty();
        serde_json::json!({"Call": {
            "call": {
                "func": {"Regular": {
                    "kind": {"Fun": fun_id},
                    "generics": {
                        "regions": [],
                        "types": [],
                        "const_generics": [],
                        "trait_refs": []
                    }
                }},
                "args": []
            },
            "dest": place_local(dest_local, &bool_ty),
            "target": target,
            "on_unwind": 0
        }})
    }

    fn drop_local(index: u64, ty: &serde_json::Value, target: u64) -> serde_json::Value {
        serde_json::json!({"Drop": {
            "place": place_local(index, ty),
            "fn_ptr": {
                "kind": {"Fun": 0},
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            },
            "target": target,
            "on_unwind": 0
        }})
    }

    /// An Opaque `bool` dest (`ptr::is_null`, `try_gc_owns_object`) is
    /// not a registration effect. Next to `try_gc_add_root` the body is
    /// still `Add`.
    #[test]
    fn opaque_bool_query_plus_add_is_add() {
        let bool_ty = bool_ty();
        let query = serde_json::json!({
            "def_id": 3,
            "item_meta": item_meta(&["core", "ptr", "is_null"], false),
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": bool_ty
            },
            "body": "Opaque"
        });
        let llbc = wrapper_llbc_with_funs(
            vec![],
            vec![
                block(vec![], call_opaque_bool(3, 1, 1)),
                block(vec![], call_add_root(2)),
                block(vec![], serde_json::json!("Return")),
            ],
            vec![],
            vec![query],
        );
        let cache = Mutex::new(HashMap::new());
        assert_eq!(
            root_registration_effect(0, &llbc, &cache),
            Some(RootRegistration::Add)
        );
    }

    /// A Drop of a type with no glue (`i64`) is not a registration
    /// effect. Next to `try_gc_add_root` the body is still `Add`.
    #[test]
    fn nonglue_drop_plus_add_is_add() {
        let i64_ty = i64_ty();
        let effect = wrapper_effect(
            vec![local(2, &i64_ty)],
            vec![
                block(vec![], drop_local(2, &i64_ty, 1)),
                block(vec![], call_add_root(2)),
                block(vec![], serde_json::json!("Return")),
            ],
            vec![],
        );
        assert_eq!(effect, Some(RootRegistration::Add));
    }

    /// An unknown statement is an effect (`analyze_simple_operation`
    /// returns True / top for an unknown opname).
    #[test]
    fn unknown_statement_is_not_registration() {
        let effect = wrapper_effect(
            vec![],
            vec![
                block(
                    vec![stmt(serde_json::json!({"NotAStatement": true}))],
                    call_add_root(1),
                ),
                block(vec![], serde_json::json!("Return")),
            ],
            vec![],
        );
        assert_eq!(effect, None);
    }
}
