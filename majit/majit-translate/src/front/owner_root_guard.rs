//! `OwnerRootGuard` is one `Ref` word in a lowered graph.
//!
//! Upstream never puts root-slot management in a jitcode. The slots are
//! created by `ShadowStackFrameworkGCTransformer`
//! (`memory/gctransform/shadowstack.py` `push_roots` / `pop_roots`), which
//! runs after `warmspot.py` `make_jitcodes` has read the graphs. The JIT
//! keeps its own references rooted (the jitframe gcmap, the blackhole's
//! `registers_r`, the recorder). pyre spells that slot as
//! [`PATH`] in interpreter source, so the codewriter sees the reference the
//! guard roots and none of the slot calls. The native build keeps executing
//! the real guard.
//!
//! `new(r)` is `r`. `get` reads the guard's word back (`GcRef` is
//! `repr(transparent)`, so the word is the `GcRef`). Projecting that
//! word's `usize` field and casting it to a pointer reads the same
//! reference: the integer bank does not hold it. `set(v)` writes `v`
//! into the local or the field that holds the guard. `Drop` of the guard,
//! and of `Option<OwnerRootGuard>`, is nothing. `Option<OwnerRootGuard>`
//! is the nullable `Option<Ref>` this front end already lowers for a
//! one-word reference payload (`None` is null, `Some(p)` is `p`).
//!
//! A callee that still runs as Rust observes the real layout. A pointer to
//! the guard, or to an aggregate that contains one by value, passed to such
//! a callee declines the graph.

use std::collections::HashSet;

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, CallKind, CallPayload, FunDecl, FunId, NameSeg, Place, PlaceKind, ProjectionElem,
    TermKind, TyRef, TypeDeclKind,
};

use super::operand_tyref;

/// Charon declaration path of the guard. Matched in full, never by the leaf.
pub(super) const PATH: &str = "majit_gc::shadow_stack::OwnerRootGuard";

/// A guard operation the lowering erases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Op {
    /// `OwnerRootGuard::new(r)` — the result is `r`.
    New,
    /// `guard.get()` — the result is the guard's word.
    Get,
    /// `guard.set(v)` — `v` replaces the guard's word in place.
    Set,
    /// `Drop::drop` / `drop_in_place` of the guard or of `Option` of it.
    Drop,
    /// `Option<OwnerRootGuard>::take` — read the word, write null.
    OptionTake,
}

pub(super) fn is_guard_path(path: &str) -> bool {
    path == PATH || path.ends_with(&format!("::{PATH}"))
}

fn type_id_is_guard(id: u64, llbc: &Llbc) -> bool {
    llbc.type_by_id(id)
        .is_some_and(|td| is_guard_path(&td.item_meta.name_path()))
}

/// `node` is the guard ADT, after dedup and hash-cons wrappers.
pub(super) fn type_node_is_guard(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(node) = super::strip_ty_indirections(node, llbc) else {
        return false;
    };
    super::adt_node_def_id(node).is_some_and(|id| type_id_is_guard(id, llbc))
}

pub(super) fn tyref_is_guard(ty: &TyRef, llbc: &Llbc) -> bool {
    super::tyref_node(ty, llbc).is_some_and(|node| type_node_is_guard(node, llbc))
}

pub(super) fn tyref_is_option_of_guard(ty: &TyRef, llbc: &Llbc) -> bool {
    crate::front::result_exc::tyref_option_payload(ty, llbc)
        .is_some_and(|payload| tyref_is_guard(&payload, llbc))
}

/// A drop of this place lowers to nothing: the guard, or `Option` of it.
pub(super) fn tyref_is_erased_drop(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_guard(ty, llbc) || tyref_is_option_of_guard(ty, llbc) {
        return true;
    }
    ref_pointee_ty(ty, llbc).is_some_and(|pointee| {
        tyref_is_guard(&pointee, llbc) || tyref_is_option_of_guard(&pointee, llbc)
    })
}

fn fn_leaf(fd: &FunDecl) -> Option<&str> {
    match fd.item_meta.name.last()? {
        NameSeg::Ident { ident: (leaf, _) } => Some(leaf.as_str()),
        NameSeg::Other(value) => builtin_leaf(value),
    }
}

fn builtin_leaf(value: &serde_json::Value) -> Option<&str> {
    let builtin = value.as_object()?.get("Builtin")?;
    if let Some(arr) = builtin.as_array() {
        return arr.first()?.as_str();
    }
    None
}

/// Adt id of the inherent or trait impl this method belongs to.
fn impl_self_id(fd: &FunDecl, llbc: &Llbc) -> Option<u64> {
    let segs = &fd.item_meta.name;
    let last_ident = segs
        .iter()
        .rposition(|seg| matches!(seg, NameSeg::Ident { .. }))?;
    if last_ident == 0 {
        return None;
    }
    let NameSeg::Other(value) = &segs[last_ident - 1] else {
        return None;
    };
    let payload = value.as_object()?.get("Impl")?;
    super::resolve_impl_owner_adt_def_id_free(llbc, payload)
}

fn ref_pointee_ty(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    let node = super::tyref_node(ty, llbc)?;
    let node = super::strip_ty_indirections(node, llbc)?;
    let obj = node.as_object()?;
    let inner = if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
        arr.get(1)?
    } else if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
        arr.first()?
    } else {
        return None;
    };
    serde_json::from_value(inner.clone()).ok()
}

/// Peel every pointer layer, then say whether the pointee contains a guard
/// by value. A field that is itself a pointer does not: its width does not
/// change when the guard's layout does.
pub(super) fn pointer_exposes_guard(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(start) =
        super::tyref_node(ty, llbc).and_then(|node| super::strip_ty_indirections(node, llbc))
    else {
        return false;
    };
    let mut node = start;
    let mut saw_pointer = false;
    for _ in 0..8 {
        let Some(obj) = node.as_object() else {
            break;
        };
        let inner = if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            arr.get(1)
        } else if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
            arr.first()
        } else {
            None
        };
        let Some(inner) = inner else {
            break;
        };
        saw_pointer = true;
        let Some(next) = super::strip_ty_indirections(inner, llbc) else {
            // A pointer whose pointee does not resolve still names a layout
            // this lowering may have changed.
            return true;
        };
        node = next;
    }
    saw_pointer && contains_guard_by_value(node, llbc, 0, &mut Vec::new())
}

fn contains_guard_by_value<'a>(
    node: &'a serde_json::Value,
    llbc: &'a Llbc,
    depth: usize,
    seen: &mut Vec<u64>,
) -> bool {
    if depth > 32 {
        return true;
    }
    let Some(node) = super::strip_ty_indirections(node, llbc) else {
        return false;
    };
    // A pointer-sized field is not the guard's bytes.
    if node.get("Ref").is_some() || node.get("RawPtr").is_some() {
        return false;
    }
    let Some(adt) = node.get("Adt").and_then(serde_json::Value::as_object) else {
        return false;
    };
    if super::adt_node_def_id(node).is_some_and(|id| type_id_is_guard(id, llbc)) {
        return true;
    }
    if let Some(types) = adt
        .get("generics")
        .and_then(serde_json::Value::as_object)
        .and_then(|generics| generics.get("types"))
        .and_then(serde_json::Value::as_array)
        && types
            .iter()
            .any(|ty| contains_guard_by_value(ty, llbc, depth + 1, seen))
    {
        return true;
    }
    let Some(id) = super::adt_node_def_id(node) else {
        return false;
    };
    if seen.contains(&id) {
        return false;
    }
    seen.push(id);
    let contained = llbc.type_by_id(id).is_some_and(|td| match &td.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => fields
            .iter()
            .any(|field| contains_ty(&field.ty, llbc, depth + 1, seen)),
        TypeDeclKind::Enum(variants) => variants.iter().any(|variant| {
            variant
                .fields
                .iter()
                .any(|field| contains_ty(&field.ty, llbc, depth + 1, seen))
        }),
        TypeDeclKind::Alias(_) | TypeDeclKind::Opaque | TypeDeclKind::Unknown => false,
    });
    seen.pop();
    contained
}

fn contains_ty(ty: &TyRef, llbc: &Llbc, depth: usize, seen: &mut Vec<u64>) -> bool {
    let Some(node) = super::tyref_node(ty, llbc) else {
        return false;
    };
    contains_guard_by_value(node, llbc, depth, seen)
}

fn arg_is_erased_drop(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_erased_drop(ty, llbc)
}

fn is_destructor_leaf(leaf: &str) -> bool {
    matches!(leaf, "drop" | "drop_in_place" | "DropGlue")
}

fn is_option_take_path(path: &str) -> bool {
    path.contains("option::<Impl>::take") || path.contains("::option::") && path.ends_with("::take")
}

/// Classify a call the lowering erases, from the callee declaration and the
/// types the call actually passes. The guard is named by [`PATH`], not by a
/// leaf that another type could share.
pub(super) fn classify(call: &CallPayload, llbc: &Llbc) -> Option<Op> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let leaf = fn_leaf(fd)?;
    if impl_self_id(fd, llbc).is_some_and(|id| type_id_is_guard(id, llbc)) {
        return match leaf {
            "new" => Some(Op::New),
            "get" => Some(Op::Get),
            "set" => Some(Op::Set),
            "drop" => Some(Op::Drop),
            _ => None,
        };
    }
    let arg_tys: Vec<Option<&TyRef>> = call.args.iter().map(operand_tyref).collect();
    if is_destructor_leaf(leaf)
        && arg_tys
            .iter()
            .flatten()
            .any(|ty| arg_is_erased_drop(ty, llbc))
    {
        return Some(Op::Drop);
    }
    if leaf == "take" && is_option_take_path(&fd.item_meta.name_path()) {
        let pointee_is_guard_option = arg_tys.iter().flatten().any(|ty| {
            tyref_is_option_of_guard(ty, llbc)
                || ref_pointee_ty(ty, llbc)
                    .is_some_and(|pointee| tyref_is_option_of_guard(&pointee, llbc))
        });
        if pointee_is_guard_option {
            return Some(Op::OptionTake);
        }
    }
    None
}

fn callee_name(call: &CallPayload, llbc: &Llbc) -> String {
    let CallFunc::Regular(reg) = &call.func else {
        return "<dynamic>".to_string();
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return "<indirect>".to_string();
    };
    llbc.fn_by_id(*id)
        .map(|fd| fd.item_meta.name_path())
        .unwrap_or_else(|| format!("<fun {id}>"))
}

fn callee_stays_residual(
    call: &CallPayload,
    llbc: &Llbc,
    dont_look_inside: &HashSet<String>,
) -> bool {
    let CallFunc::Regular(reg) = &call.func else {
        return true;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return true;
    };
    let Some(fd) = llbc.fn_by_id(*id) else {
        return true;
    };
    if fd.unstructured().is_none() {
        return true;
    }
    let path = super::strip_crate_prefix(&fd.item_meta.name_path());
    dont_look_inside.contains(&path)
}

/// `is_some` / `is_none` on `Option<OwnerRootGuard>` are the nullable-pointer
/// predicate (`front::option_is_none`), not a call into `core`.
fn niche_option_predicate(call: &CallPayload, llbc: &Llbc) -> bool {
    let CallFunc::Regular(reg) = &call.func else {
        return false;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    let Some(fd) = llbc.fn_by_id(*id) else {
        return false;
    };
    let Some(leaf) = fn_leaf(fd) else {
        return false;
    };
    if !matches!(leaf, "is_some" | "is_none") {
        return false;
    }
    let path = fd.item_meta.name_path();
    if !path.contains("option::<Impl>") && !path.contains("::option::") {
        return false;
    }
    call.args.iter().any(|arg| {
        operand_tyref(arg).is_some_and(|ty| {
            tyref_is_option_of_guard(ty, llbc)
                || ref_pointee_ty(ty, llbc)
                    .is_some_and(|pointee| tyref_is_option_of_guard(&pointee, llbc))
        })
    })
}

/// `Some(callee)` when `call` passes a pointer to a guard-containing value
/// into a callee that stays residual. Erased guard operations are not escapes.
pub(super) fn residual_escape(
    call: &CallPayload,
    llbc: &Llbc,
    dont_look_inside: &HashSet<String>,
) -> Option<String> {
    if classify(call, llbc).is_some() || niche_option_predicate(call, llbc) {
        return None;
    }
    let exposes = call
        .args
        .iter()
        .any(|arg| operand_tyref(arg).is_some_and(|ty| pointer_exposes_guard(ty, llbc)));
    if !exposes || !callee_stays_residual(call, llbc, dont_look_inside) {
        return None;
    }
    Some(callee_name(call, llbc))
}

/// Every local graph that [`residual_escape`] would decline, as
/// `(graph path, callee path)`.
pub(super) fn residual_escape_census(llbc: &Llbc) -> Vec<(String, String)> {
    let dont_look_inside = super::dont_look_inside_set_of(llbc);
    let mut out = Vec::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let graph = fd.item_meta.name_path();
        let forward_blocks = super::forward_reachable_mask(llbc, &body);
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
                continue;
            };
            if let Some(callee) = residual_escape(&call, llbc, &dont_look_inside) {
                out.push((graph.clone(), callee));
            }
        }
    }
    out
}

/// Where `set` writes. A projection of the `Some` payload of
/// `Option<OwnerRootGuard>` is the option word itself.
pub(super) fn guard_sink_place(place: &Place, llbc: &Llbc) -> Place {
    if let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind
        && elem.as_object().and_then(|map| map.get("Field")).is_some()
        && tyref_is_option_of_guard(&inner.ty, llbc)
    {
        return guard_sink_place(inner, llbc);
    }
    super::clone_place(place)
}
