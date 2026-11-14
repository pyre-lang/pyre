//! `RBigIntGcRoot` is the `RBigInt` it roots in a lowered graph.
//!
//! Upstream never puts root-slot management in a jitcode. The slots are
//! created by `ShadowStackFrameworkGCTransformer`
//! (`memory/gctransform/shadowstack.py` `push_roots` / `pop_roots`), which
//! runs after `warmspot.py` `make_jitcodes` has read the graphs. In
//! `rbigint.py` the local is the `rbigint` instance. pyre spells the slot
//! as [`PATH`] in interpreter source (`majit_rlib::rbigint::gc`), so the
//! codewriter sees that instance and none of the slot calls. The native
//! build keeps executing the real handle.
//!
//! The shared reasoning is [`super::owner_root_guard`]: a Rust GC-root
//! RAII type is not a jitcode value. The handle is not a pointer word —
//! it is the by-value `RBigInt` in its `value` field (after `UnsafeCell`).
//! `new(v)` is `v`. `deref` / `deref_mut` is a borrow of that value.
//! `set(v)` writes `v` into the place. `normalize` is
//! `RBigInt::_normalize` on the erased value. `Drop` of the handle, and
//! of `Option` of it, is nothing. After erasure `Option<RBigIntGcRoot>`
//! is a tagged `Option<RBigInt>` while `Option<&RBigInt>` is the thin
//! null-niche. The extracted LLBC has no unstructured body for
//! `as_ref` / `as_deref` (and the `mut` forms), so those calls lower as
//! that tagged-to-niche conversion: `__discriminant` then the `Some`
//! payload or a `None` null (`emit_enum_disc_read` / `emit_payload_read`
//! / `emit_niche_null` / `close_call_diamond`). After erasure `Deref` of
//! the handle is identity, so `as_ref` and `as_deref` are the same
//! conversion. `core::array::from_fn::<T, N, F>(f)` has no unstructured
//! body, so a residual call would see the handle layout. The front end
//! lowers it as the array literal `[f(0), …, f(N-1)]`: each element is
//! an ordinary `FnMut::call_mut` of `F` with `(&mut f, (i,))`, then the
//! same Array aggregate a `[v0, …, vN-1]` literal builds. A call whose
//! `N` is not concrete or whose closure body cannot be found stays
//! residual, and [`residual_escape`] declines a handle-containing
//! destination.
//!
//! A callee that still runs as Rust observes the real layout. A handle
//! reachable by value or through any number of pointers, at any depth
//! inside an argument or destination type, declines the graph.

use std::collections::HashSet;

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, CallKind, CallPayload, FunDecl, FunId, NameSeg, TermKind, TyRef, TypeDeclKind,
};

use super::operand_tyref;

/// Charon declaration path of the handle. Matched in full, never by the leaf.
pub(super) const PATH: &str = "majit_rlib::rbigint::gc::RBigIntGcRoot";

const RBIGINT_PATH: &str = "majit_rlib::rbigint::RBigInt";

const OPTION_PATH: &str = "core::option::Option";

/// A handle operation the lowering erases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Op {
    /// `RBigIntGcRoot::new(v)` — the result is `v`.
    New,
    /// `<RBigIntGcRoot as Deref>::deref` / `DerefMut::deref_mut` — the
    /// result is the borrow of the erased `RBigInt`.
    Deref,
    /// `handle.set(v)` — `v` replaces the value in place.
    Set,
    /// `handle.normalize()` — `RBigInt::_normalize` on the erased value.
    Normalize,
    /// `Drop::drop` / `drop_in_place` of the handle or of `Option` of it.
    Drop,
    /// `Option<RBigIntGcRoot>::as_deref` / `as_ref` (and `mut`) — tagged
    /// `Option<RBigInt>` to the null-niche `Option<&RBigInt>`.
    OptionAsDeref,
}

pub(super) fn is_root_path(path: &str) -> bool {
    path == PATH
        || path
            .strip_suffix(PATH)
            .is_some_and(|head| head.ends_with("::"))
}

fn is_rbigint_path(path: &str) -> bool {
    path == RBIGINT_PATH || path == "rbigint::RBigInt" || path.ends_with("::rbigint::RBigInt")
}

fn is_rbigint_decl_path(path: &str) -> bool {
    path == RBIGINT_PATH
        || path
            .strip_suffix(RBIGINT_PATH)
            .is_some_and(|head| head.ends_with("::"))
}

fn type_id_is_root(id: u64, llbc: &Llbc) -> bool {
    let Some(td) = llbc.type_by_id(id) else {
        return false;
    };
    if is_root_path(td.item_meta.name_path_str()) {
        return true;
    }
    let TypeDeclKind::Alias(body) = &td.kind else {
        return false;
    };
    type_node_is_root(body.get("aliased_ty").unwrap_or(body), llbc)
}

fn type_id_is_rbigint(id: u64, llbc: &Llbc) -> bool {
    llbc.type_by_id(id)
        .is_some_and(|td| is_rbigint_path(td.item_meta.name_path_str()))
}

fn type_id_is_option(id: u64, llbc: &Llbc) -> bool {
    llbc.type_by_id(id)
        .is_some_and(|td| td.item_meta.name_path_str() == OPTION_PATH)
}

/// `node` is the handle ADT, after dedup, hash-cons wrappers, and aliases.
pub(super) fn type_node_is_root(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(node) = super::type_node_peel_aliases(node, llbc) else {
        return false;
    };
    super::adt_node_def_id(node).is_some_and(|id| {
        llbc.type_by_id(id)
            .is_some_and(|td| is_root_path(td.item_meta.name_path_str()))
    })
}

pub(super) fn tyref_is_root(ty: &TyRef, llbc: &Llbc) -> bool {
    super::tyref_node(ty, llbc).is_some_and(|node| type_node_is_root(node, llbc))
}

pub(super) fn tyref_is_option_of_root(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) =
        super::tyref_node(ty, llbc).and_then(|n| super::type_node_peel_aliases(n, llbc))
    else {
        return false;
    };
    crate::front::result_exc::tyref_option_payload(&TyRef::Other(node.clone().into()), llbc)
        .is_some_and(|payload| tyref_is_root(&payload, llbc))
}

/// A drop of this place lowers to nothing: the handle, or `Option` of it.
pub(super) fn tyref_is_erased_drop(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_root(ty, llbc) || tyref_is_option_of_root(ty, llbc) {
        return true;
    }
    ref_pointee_ty(ty, llbc).is_some_and(|pointee| {
        tyref_is_root(&pointee, llbc) || tyref_is_option_of_root(&pointee, llbc)
    })
}

/// The `RBigInt` type the handle's `value` field stores after `UnsafeCell`.
/// `id` is the handle's type-decl id, or an alias of it; every caller
/// already matched [`type_node_is_root`] on that node. An importing crate's
/// declaration is `Opaque` until a published body is copied onto it, so
/// this falls back to the `RBigInt` declaration at [`RBIGINT_PATH`].
pub(super) fn erased_value_ty(id: u64, llbc: &Llbc) -> Option<TyRef> {
    let id = resolved_root_id(id, llbc)?;
    if let Some(ty) = erased_value_from_fields(id, llbc) {
        return Some(ty);
    }
    rbigint_tyref_by_path(llbc)
}

fn resolved_root_id(id: u64, llbc: &Llbc) -> Option<u64> {
    let td = llbc.type_by_id(id)?;
    if is_root_path(td.item_meta.name_path_str()) {
        return Some(id);
    }
    let TypeDeclKind::Alias(body) = &td.kind else {
        return None;
    };
    let node = body.get("aliased_ty").unwrap_or(body);
    let peeled = super::type_node_peel_aliases(node, llbc)?;
    super::adt_node_def_id(peeled).filter(|&peeled_id| {
        llbc.type_by_id(peeled_id)
            .is_some_and(|td| is_root_path(td.item_meta.name_path_str()))
    })
}

fn erased_value_from_fields(id: u64, llbc: &Llbc) -> Option<TyRef> {
    let td = llbc.type_by_id(id)?;
    let fields = super::type_decl_struct_fields(td, llbc)?;
    let value = fields
        .iter()
        .find(|field| field.name.as_deref() == Some("value"))?;
    if super::tyref_is_unsafecell(&value.ty, llbc) {
        super::tyref_adt_generics_type_arg(&value.ty, 0, llbc)
    } else {
        Some(super::clone_tyref(&value.ty))
    }
}

fn rbigint_tyref_by_path(llbc: &Llbc) -> Option<TyRef> {
    llbc.iter_type_decls().find_map(|td| {
        is_rbigint_decl_path(td.item_meta.name_path_str())
            .then(|| super::type_decl_adt_tyref(td.def_id))
    })
}

fn rbigint_type_node(id: u64, llbc: &Llbc) -> Option<serde_json::Value> {
    let ty = erased_value_ty(id, llbc)?;
    super::tyref_node(&ty, llbc)
        .and_then(|node| super::strip_ty_indirections(node, llbc))
        .cloned()
}

/// Every `RBigIntGcRoot` declaration in this artefact can be rewritten to
/// `RBigInt`. Classify erases values only when this is true.
fn root_erasure_ready(llbc: &Llbc) -> bool {
    llbc.iter_type_decls().all(|td| {
        !is_root_path(td.item_meta.name_path_str()) || erased_value_ty(td.def_id, llbc).is_some()
    })
}

/// Substitute the handle (and wrappers / generic arguments that mention it)
/// with the erased `RBigInt` type. `None` when `ty` does not name the handle.
pub(super) fn erase_tyref(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    let node = super::tyref_node(ty, llbc)?;
    rewrite_type_node(node, llbc, 0).map(|node| TyRef::Other(node.into()))
}

fn rewrite_type_node(
    node: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> Option<serde_json::Value> {
    if depth > 24 {
        return None;
    }
    let node = super::type_node_peel_aliases(node, llbc)?;
    if type_node_is_root(node, llbc) {
        return rbigint_type_node(super::adt_node_def_id(node)?, llbc);
    }
    let obj = node.as_object()?;
    if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
        let inner = arr.get(1)?;
        let rewritten = rewrite_type_node(inner, llbc, depth + 1)?;
        let mut new_arr = arr.clone();
        new_arr[1] = rewritten;
        return Some(serde_json::json!({"Ref": new_arr}));
    }
    if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
        let inner = arr.first()?;
        let rewritten = rewrite_type_node(inner, llbc, depth + 1)?;
        let mut new_arr = arr.clone();
        new_arr[0] = rewritten;
        return Some(serde_json::json!({"RawPtr": new_arr}));
    }
    if let Some(arr) = obj.get("Array").and_then(serde_json::Value::as_array) {
        let inner = arr.first()?;
        let rewritten = rewrite_type_node(inner, llbc, depth + 1)?;
        let mut new_arr = arr.clone();
        new_arr[0] = rewritten;
        return Some(serde_json::json!({"Array": new_arr}));
    }
    if let Some(arr) = obj.get("Slice").and_then(serde_json::Value::as_array) {
        let inner = arr.first()?;
        let rewritten = rewrite_type_node(inner, llbc, depth + 1)?;
        let mut new_arr = arr.clone();
        new_arr[0] = rewritten;
        return Some(serde_json::json!({"Slice": new_arr}));
    }
    if let Some(items) = obj.get("Tuple").and_then(serde_json::Value::as_array) {
        return rewrite_seq(items, "Tuple", llbc, depth);
    }
    if let Some(adt) = obj.get("Adt").and_then(serde_json::Value::as_object) {
        let types = adt
            .get("generics")
            .and_then(serde_json::Value::as_object)
            .and_then(|generics| generics.get("types"))
            .and_then(serde_json::Value::as_array)?;
        let rewritten: Vec<Option<serde_json::Value>> = types
            .iter()
            .map(|ty| rewrite_type_node(ty, llbc, depth + 1))
            .collect();
        if rewritten.iter().all(Option::is_none) {
            return None;
        }
        let new_types: Vec<serde_json::Value> = rewritten
            .into_iter()
            .zip(types.iter())
            .map(|(rewritten, ty)| rewritten.unwrap_or_else(|| ty.clone()))
            .collect();
        let mut new_adt = adt.clone();
        if let Some(generics) = new_adt
            .get_mut("generics")
            .and_then(serde_json::Value::as_object_mut)
        {
            generics.insert("types".to_string(), serde_json::Value::Array(new_types));
        }
        return Some(serde_json::json!({"Adt": new_adt}));
    }
    None
}

fn rewrite_seq(
    items: &[serde_json::Value],
    key: &str,
    llbc: &Llbc,
    depth: usize,
) -> Option<serde_json::Value> {
    let rewritten: Vec<Option<serde_json::Value>> = items
        .iter()
        .map(|ty| rewrite_type_node(ty, llbc, depth + 1))
        .collect();
    if rewritten.iter().all(Option::is_none) {
        return None;
    }
    let new_items: Vec<serde_json::Value> = rewritten
        .into_iter()
        .zip(items.iter())
        .map(|(rewritten, ty)| rewritten.unwrap_or_else(|| ty.clone()))
        .collect();
    Some(serde_json::json!({ key: new_items }))
}

fn fn_leaf(fd: &FunDecl) -> Option<&str> {
    match fd.item_meta.name.last()? {
        NameSeg::Ident { ident: (leaf, _) } => Some(leaf.as_str()),
        NameSeg::Builtin { kind, .. } => Some(kind.as_str()),
        _ => None,
    }
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
    let impl_seg = &segs[last_ident - 1];
    if !impl_seg.is_impl() {
        return None;
    }
    super::resolve_impl_owner_adt_def_id_free(llbc, impl_seg)
}

fn ref_pointee_ty(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    let node = super::tyref_node(ty, llbc)?;
    let node = super::strip_ty_indirections(node, llbc)?;
    let obj = node.as_object()?;
    let inner = if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
        arr.get(1)?
    } else {
        let arr = obj.get("RawPtr").and_then(serde_json::Value::as_array)?;
        arr.first()?
    };
    serde_json::from_value(inner.clone()).ok()
}

/// A handle is reachable from `node` by value or through any number of
/// pointers (`Ref`, `RawPtr`) at any depth: tuple items, array/slice
/// elements, ADT generic arguments, and struct/union/enum fields.
/// `CallControl.getcalldescr` (`rpython/jit/codewriter/call.py`) requires
/// the residual callee to see the same representation the caller built.
fn contains_root<'a>(
    node: &'a serde_json::Value,
    llbc: &'a Llbc,
    depth: usize,
    seen: &mut Vec<u64>,
) -> bool {
    if depth > 32 {
        return true;
    }
    let Some(node) = super::type_node_peel_aliases(node, llbc) else {
        return false;
    };
    if let Some(arr) = node.get("Ref").and_then(serde_json::Value::as_array) {
        return arr
            .get(1)
            .is_some_and(|ty| contains_root(ty, llbc, depth + 1, seen));
    }
    if let Some(arr) = node.get("RawPtr").and_then(serde_json::Value::as_array) {
        return arr
            .first()
            .is_some_and(|ty| contains_root(ty, llbc, depth + 1, seen));
    }
    if let Some(items) = node.get("Tuple").and_then(serde_json::Value::as_array) {
        return items
            .iter()
            .any(|ty| contains_root(ty, llbc, depth + 1, seen));
    }
    if let Some(arr) = node
        .get("Array")
        .or_else(|| node.get("Slice"))
        .and_then(serde_json::Value::as_array)
    {
        return arr
            .first()
            .is_some_and(|ty| contains_root(ty, llbc, depth + 1, seen));
    }
    let Some(adt) = node.get("Adt").and_then(serde_json::Value::as_object) else {
        return false;
    };
    if super::adt_node_def_id(node).is_some_and(|id| type_id_is_root(id, llbc)) {
        return true;
    }
    if let Some(types) = adt
        .get("generics")
        .and_then(serde_json::Value::as_object)
        .and_then(|generics| generics.get("types"))
        .and_then(serde_json::Value::as_array)
        && types
            .iter()
            .any(|ty| contains_root(ty, llbc, depth + 1, seen))
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
        TypeDeclKind::Alias(body) => contains_root(
            body.get("aliased_ty").unwrap_or(body),
            llbc,
            depth + 1,
            seen,
        ),
        TypeDeclKind::Opaque | TypeDeclKind::Unknown => false,
    });
    seen.pop();
    contained
}

fn contains_ty(ty: &TyRef, llbc: &Llbc, depth: usize, seen: &mut Vec<u64>) -> bool {
    let Some(node) = super::tyref_node(ty, llbc) else {
        return false;
    };
    contains_root(node, llbc, depth, seen)
}

fn type_exposes_root(ty: &TyRef, llbc: &Llbc) -> bool {
    let id = match ty {
        TyRef::Dedup { id } => Some(*id),
        TyRef::Inline { value: (id, _) } => Some(*id),
        TyRef::Other(_) => None,
    };
    if let Some(id) = id
        && let Some(cached) = llbc.type_flag_memo_get(id)
    {
        return cached;
    }
    let value = contains_ty(ty, llbc, 0, &mut Vec::new());
    if let Some(id) = id {
        llbc.type_flag_memo_set(id, value);
    }
    value
}

fn arg_is_erased_drop(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_erased_drop(ty, llbc)
}

fn path_is(path: &str, item: &str) -> bool {
    path == item
        || path
            .strip_suffix(item)
            .is_some_and(|head| head.ends_with("::"))
}

/// Charon `PathElem::Builtin(DropGlue, n)` — the drop-glue item whose
/// `name_path` leaf is rendered `drop_in_place`.
fn is_drop_glue_item(fd: &FunDecl) -> bool {
    fd.item_meta
        .template_name()
        .last()
        .is_some_and(|seg| matches!(seg, NameSeg::Builtin { kind, .. } if kind == "DropGlue"))
}

/// `SimpleCall.eval` (`rpython/flowspace/operation.py`) keys special
/// cases by callable identity, never by the leaf name.
fn is_destructor_callable(fd: &FunDecl) -> bool {
    if is_drop_glue_item(fd) {
        return true;
    }
    let path = fd.item_meta.name_path_str();
    path_is(path, "core::ptr::drop_in_place") || path_is(path, "core::ops::drop::Drop::drop")
}

fn is_deref_callable(fd: &FunDecl) -> bool {
    let path = fd.item_meta.name_path_str();
    path_is(path, "core::ops::deref::Deref::deref")
        || path_is(path, "core::ops::deref::DerefMut::deref_mut")
}

fn arg_is_root_or_borrow(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_root(ty, llbc)
        || ref_pointee_ty(ty, llbc).is_some_and(|pointee| tyref_is_root(&pointee, llbc))
}

/// Classify a call the lowering erases, from the callee declaration and the
/// types the call actually passes. The handle is named by [`PATH`], not by a
/// leaf that another type could share. Erasure of values is gated on
/// [`root_erasure_ready`]: a handle whose `RBigInt` type cannot be resolved
/// is not rewritten.
pub(super) fn classify(call: &CallPayload, llbc: &Llbc) -> Option<Op> {
    let op = classify_op(call, llbc)?;
    root_erasure_ready(llbc).then_some(op)
}

fn classify_op(call: &CallPayload, llbc: &Llbc) -> Option<Op> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let leaf = fn_leaf(fd)?;
    if impl_self_id(fd, llbc).is_some_and(|id| type_id_is_root(id, llbc)) {
        return match leaf {
            "new" => Some(Op::New),
            "deref" | "deref_mut" => Some(Op::Deref),
            "set" => Some(Op::Set),
            "normalize" => Some(Op::Normalize),
            "drop" => Some(Op::Drop),
            _ => None,
        };
    }
    let arg_tys: Vec<Option<&TyRef>> = call.args.iter().map(operand_tyref).collect();
    if is_destructor_callable(fd)
        && arg_tys
            .iter()
            .flatten()
            .any(|ty| arg_is_erased_drop(ty, llbc))
    {
        return Some(Op::Drop);
    }
    if is_deref_callable(fd)
        && arg_tys
            .iter()
            .flatten()
            .any(|ty| arg_is_root_or_borrow(ty, llbc))
    {
        return Some(Op::Deref);
    }
    if is_option_inherent_method(fd, llbc, &["as_deref", "as_deref_mut", "as_ref", "as_mut"])
        && arg_tys.iter().flatten().any(|ty| {
            tyref_is_option_of_root(ty, llbc)
                || ref_pointee_ty(ty, llbc)
                    .is_some_and(|pointee| tyref_is_option_of_root(&pointee, llbc))
        })
    {
        return Some(Op::OptionAsDeref);
    }
    None
}

fn is_option_inherent_method(fd: &FunDecl, llbc: &Llbc, leaves: &[&str]) -> bool {
    fn_leaf(fd).is_some_and(|leaf| leaves.contains(&leaf))
        && impl_self_id(fd, llbc).is_some_and(|id| type_id_is_option(id, llbc))
}

/// The defining-crate `RBigInt::_normalize` declaration.
pub(super) fn rbigint_normalize_decl(llbc: &Llbc) -> Option<&FunDecl> {
    llbc.iter_fun_decls().find(|fd| {
        fn_leaf(fd) == Some("_normalize")
            && impl_self_id(fd, llbc).is_some_and(|id| type_id_is_rbigint(id, llbc))
    })
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
    let path = super::strip_crate_prefix(fd.item_meta.name_path_str());
    dont_look_inside.contains(&path)
}

/// A handle is reachable from this call, but [`erased_value_ty`] cannot
/// resolve `RBigInt`. Classify would otherwise erase `new` / `deref`
/// while type lowering kept the handle.
pub(super) fn unresolved_escape(call: &CallPayload, llbc: &Llbc) -> bool {
    let arg_exposes = call
        .args
        .iter()
        .any(|arg| operand_tyref(arg).is_some_and(|ty| type_exposes_root(ty, llbc)));
    if !(arg_exposes || type_exposes_root(&call.dest.ty, llbc)) {
        return false;
    }
    !root_erasure_ready(llbc)
}

/// `Some(callee)` when `call` passes a handle reachable by value or
/// through any number of pointers, at any depth, into a callee that stays
/// residual, or receives one as its destination. Erased handle operations
/// are not escapes. A core `Option` body that this front end lowers is
/// not residual (`callee_stays_residual` is false when the unstructured
/// body is present). `core::array::from_fn` that this front end lowers
/// to `[f(0), …, f(N-1)]` is not residual.
pub(super) fn residual_escape(
    call: &CallPayload,
    llbc: &Llbc,
    dont_look_inside: &HashSet<String>,
) -> Option<String> {
    if classify(call, llbc).is_some() {
        return None;
    }
    if super::desugar::std_call::array_from_fn_plan(call, llbc).is_some() {
        return None;
    }
    let arg_exposes = call
        .args
        .iter()
        .any(|arg| operand_tyref(arg).is_some_and(|ty| type_exposes_root(ty, llbc)));
    let dest_exposes = type_exposes_root(&call.dest.ty, llbc);
    if !(arg_exposes || dest_exposes) || !callee_stays_residual(call, llbc, dont_look_inside) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front::mir::tests::{
        fixture_item_meta, ident_path, llbc_with_types as parse_llbc_types,
        llbc_with_types_impls as parse_llbc_types_impls, named_adt, type_decl_alias,
        type_decl_struct,
    };

    fn llbc_with_types(
        crate_name: &str,
        types: Vec<serde_json::Value>,
        funs: Vec<serde_json::Value>,
    ) -> Llbc {
        parse_llbc_types(crate_name, with_rbigint(types), funs)
    }

    fn llbc_with_types_impls(
        crate_name: &str,
        types: Vec<serde_json::Value>,
        funs: Vec<serde_json::Value>,
        impls: Vec<serde_json::Value>,
    ) -> Llbc {
        parse_llbc_types_impls(crate_name, with_rbigint(types), funs, impls)
    }

    const ROOT_PATH: &[&str] = &["majit_rlib", "rbigint", "gc", "RBigIntGcRoot"];

    fn usize_ty() -> serde_json::Value {
        serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
    }

    fn unit_ty() -> serde_json::Value {
        serde_json::json!({"Tuple": []})
    }

    fn root_adt() -> serde_json::Value {
        named_adt(0, vec![])
    }

    fn option_of_root() -> serde_json::Value {
        named_adt(1, vec![root_adt()])
    }

    fn shared_ref(pointee: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"Ref": ["Erased", pointee, "Shared"]})
    }

    fn mut_ptr(pointee: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"RawPtr": [pointee, "Mut"]})
    }

    fn ty(node: serde_json::Value) -> TyRef {
        TyRef::Other(node.into())
    }

    fn root_type_decl() -> serde_json::Value {
        type_decl_struct(0, ROOT_PATH, vec![])
    }

    fn opaque_root_type_decl() -> serde_json::Value {
        let mut td = root_type_decl();
        td["kind"] = serde_json::json!("Opaque");
        td
    }

    fn rbigint_type_decl(id: u64) -> serde_json::Value {
        type_decl_struct(id, &["majit_rlib", "rbigint", "RBigInt"], vec![])
    }

    fn with_rbigint(mut types: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        let id = types.len() as u64;
        types.push(rbigint_type_decl(id));
        types
    }

    fn option_type_decl() -> serde_json::Value {
        type_decl_struct(1, &["core", "option", "Option"], vec![])
    }

    fn option_method_name(leaf: &str, opt_ty: &serde_json::Value) -> serde_json::Value {
        serde_json::json!([
            {"Ident": ["core", 0]},
            {"Ident": ["option", 0]},
            {"Impl": {"Ty": {
                "skip_binder": {"Value": [0, opt_ty]},
                "kind": "InherentImplBlock"
            }}},
            {"Ident": [leaf, 0]}
        ])
    }

    fn residual_fun(
        def_id: u64,
        path: &[&str],
        inputs: Vec<serde_json::Value>,
        output: serde_json::Value,
    ) -> serde_json::Value {
        residual_fun_named(def_id, ident_path(path), inputs, output)
    }

    fn residual_fun_named(
        def_id: u64,
        name: serde_json::Value,
        inputs: Vec<serde_json::Value>,
        output: serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(name),
            "signature": {
                "is_unsafe": false,
                "inputs": inputs,
                "output": output
            },
            "body": null
        })
    }

    fn fun_with_body(
        def_id: u64,
        path: &[&str],
        inputs: Vec<serde_json::Value>,
        output: serde_json::Value,
    ) -> serde_json::Value {
        fun_with_body_named(def_id, ident_path(path), inputs, output)
    }

    fn fun_with_body_named(
        def_id: u64,
        name: serde_json::Value,
        inputs: Vec<serde_json::Value>,
        output: serde_json::Value,
    ) -> serde_json::Value {
        let span = serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }});
        let arg_ty = inputs[0].clone();
        let out_ty = output.clone();
        let locals = vec![
            serde_json::json!({
                "index": 0,
                "name": null,
                "span": span,
                "ty": out_ty
            }),
            serde_json::json!({
                "index": 1,
                "name": null,
                "span": span,
                "ty": arg_ty
            }),
        ];
        serde_json::json!({
            "def_id": def_id,
            "item_meta": fixture_item_meta(name),
            "signature": {
                "is_unsafe": false,
                "inputs": inputs,
                "output": output
            },
            "body": {"Unstructured": {
                "span": span,
                "locals": {"arg_count": 1, "locals": locals},
                "body": [{"statements": [], "terminator": {"kind": "Return"}}]
            }}
        })
    }

    fn call_payload(
        fun_id: u64,
        arg_ty: serde_json::Value,
        dest_ty: serde_json::Value,
    ) -> CallPayload {
        CallPayload {
            func: serde_json::from_value(serde_json::json!({
                "Regular": {"kind": {"Fun": fun_id}, "generics": null}
            }))
            .expect("fixture CallFunc parses"),
            args: serde_json::from_value(serde_json::json!([{
                "Read": {"kind": {"Local": 1}, "ty": arg_ty}
            }]))
            .expect("fixture args parse"),
            dest: serde_json::from_value(serde_json::json!({
                "kind": {"Local": 0}, "ty": dest_ty
            }))
            .expect("fixture dest Place parses"),
        }
    }

    /// A nested reference to the handle is an escape: the residual callee
    /// would dereference the pointer as the real 3-field handle.
    #[test]
    fn nested_reference_to_the_handle_exposes_the_root() {
        let holder_fields = vec![serde_json::json!({
            "name": "t",
            "ty": {"Tuple": [root_adt(), usize_ty()]},
            "attr_info": null
        })];
        let llbc = llbc_with_types(
            "fixture",
            vec![
                root_type_decl(),
                type_decl_struct(1, &["fixture", "Holder"], holder_fields),
            ],
            vec![],
        );
        let tuple = serde_json::json!({"Tuple": [root_adt(), usize_ty()]});
        let array = serde_json::json!({"Array": [root_adt(), {"value": "32"}, null]});
        let slice = serde_json::json!({"Slice": [root_adt(), null]});
        assert!(type_exposes_root(&ty(shared_ref(tuple.clone())), &llbc));
        assert!(type_exposes_root(&ty(shared_ref(array)), &llbc));
        assert!(type_exposes_root(&ty(shared_ref(slice)), &llbc));
        assert!(type_exposes_root(
            &ty(shared_ref(named_adt(1, vec![]))),
            &llbc
        ));
        assert!(type_exposes_root(&ty(tuple), &llbc));

        let ptr_tuple = serde_json::json!({"Tuple": [shared_ref(root_adt()), usize_ty()]});
        assert!(type_exposes_root(&ty(shared_ref(ptr_tuple)), &llbc));
        let empty = serde_json::json!({"Tuple": [usize_ty(), usize_ty()]});
        assert!(!type_exposes_root(&ty(shared_ref(empty.clone())), &llbc));
        assert!(!type_exposes_root(&ty(empty), &llbc));
    }

    /// A residual callee that takes the handle by value, or returns one,
    /// declines. A core body that this front end lowers does not.
    #[test]
    fn residual_escape_declines_by_value_arg_and_result() {
        let empty = HashSet::new();
        let residual = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "take_root"],
                vec![root_adt()],
                unit_ty(),
            )],
        );
        let by_value_arg = call_payload(0, root_adt(), unit_ty());
        assert_eq!(
            residual_escape(&by_value_arg, &residual, &empty).as_deref(),
            Some("fixture::take_root")
        );

        let returns = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "make_root"],
                vec![usize_ty()],
                root_adt(),
            )],
        );
        let by_value_dest = call_payload(0, usize_ty(), root_adt());
        assert_eq!(
            residual_escape(&by_value_dest, &returns, &empty).as_deref(),
            Some("fixture::make_root")
        );

        let array_dest = serde_json::json!({"Array": [root_adt(), {"value": "32"}, null]});
        let returns_array = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "make_table"],
                vec![usize_ty()],
                array_dest.clone(),
            )],
        );
        let array_result = call_payload(0, usize_ty(), array_dest.clone());
        assert_eq!(
            residual_escape(&array_result, &returns_array, &empty).as_deref(),
            Some("fixture::make_table")
        );

        let lowered = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![fun_with_body(
                0,
                &["fixture", "take_root"],
                vec![root_adt()],
                unit_ty(),
            )],
        );
        assert_eq!(residual_escape(&by_value_arg, &lowered, &empty), None);
    }

    /// A handle reachable only through a pointer still declines a residual
    /// call: `(&RBigIntGcRoot,)`, a `*mut` field, `Option<&RBigIntGcRoot>`.
    /// A tuple of plain ints does not.
    #[test]
    fn residual_escape_declines_nested_pointers_to_the_handle() {
        let empty = HashSet::new();
        let ref_tuple = serde_json::json!({"Tuple": [shared_ref(root_adt())]});
        let by_ref_tuple = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "take_ref_tuple"],
                vec![ref_tuple.clone()],
                unit_ty(),
            )],
        );
        let arg = call_payload(0, ref_tuple, unit_ty());
        assert_eq!(
            residual_escape(&arg, &by_ref_tuple, &empty).as_deref(),
            Some("fixture::take_ref_tuple")
        );

        let ptr_fields = vec![serde_json::json!({
            "name": "p",
            "ty": mut_ptr(root_adt()),
            "attr_info": null
        })];
        let holder = named_adt(1, vec![]);
        let by_ptr_field = llbc_with_types(
            "fixture",
            vec![
                root_type_decl(),
                type_decl_struct(1, &["fixture", "Holder"], ptr_fields),
            ],
            vec![residual_fun(
                0,
                &["fixture", "make_holder"],
                vec![usize_ty()],
                holder.clone(),
            )],
        );
        let dest = call_payload(0, usize_ty(), holder);
        assert_eq!(
            residual_escape(&dest, &by_ptr_field, &empty).as_deref(),
            Some("fixture::make_holder")
        );

        let opt_ref = named_adt(1, vec![shared_ref(root_adt())]);
        let by_opt_ref = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "take_opt_ref"],
                vec![opt_ref.clone()],
                unit_ty(),
            )],
        );
        let opt_arg = call_payload(0, opt_ref, unit_ty());
        assert_eq!(
            residual_escape(&opt_arg, &by_opt_ref, &empty).as_deref(),
            Some("fixture::take_opt_ref")
        );

        let ints = serde_json::json!({"Tuple": [usize_ty(), usize_ty()]});
        let by_ints = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "take_ints"],
                vec![ints.clone()],
                unit_ty(),
            )],
        );
        let ints_arg = call_payload(0, ints, unit_ty());
        assert_eq!(residual_escape(&ints_arg, &by_ints, &empty), None);
    }

    /// A free fn named `drop` / `deref` is an ordinary call. The generic
    /// `core::ptr::drop_in_place` of the handle is `Op::Drop`.
    #[test]
    fn classify_matches_destructor_and_deref_by_full_path() {
        let free_drop = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![fun_with_body(
                0,
                &["other", "drop"],
                vec![root_adt()],
                unit_ty(),
            )],
        );
        let drop_call = call_payload(0, root_adt(), unit_ty());
        assert_eq!(classify(&drop_call, &free_drop), None);

        let free_deref = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![fun_with_body(
                0,
                &["other", "deref"],
                vec![shared_ref(root_adt())],
                unit_ty(),
            )],
        );
        let deref_call = call_payload(0, shared_ref(root_adt()), unit_ty());
        assert_eq!(classify(&deref_call, &free_deref), None);

        let drop_in_place = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["core", "ptr", "drop_in_place"],
                vec![shared_ref(root_adt())],
                unit_ty(),
            )],
        );
        let dip = call_payload(0, shared_ref(root_adt()), unit_ty());
        assert_eq!(classify(&dip, &drop_in_place), Some(Op::Drop));
    }

    /// `as_ref` / `as_deref` on `Option<RBigIntGcRoot>` classify as the
    /// tagged-to-niche conversion, so they are not residual escapes.
    #[test]
    fn option_as_ref_is_not_an_erased_alias() {
        let empty = HashSet::new();
        let arg = shared_ref(option_of_root());
        let dest = serde_json::json!({"Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [shared_ref(root_adt())],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let residual = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![residual_fun_named(
                0,
                option_method_name("as_ref", &option_of_root()),
                vec![arg.clone()],
                dest.clone(),
            )],
        );
        let call = call_payload(0, arg.clone(), dest.clone());
        assert!(
            ref_pointee_ty(&ty(arg.clone()), &residual)
                .is_some_and(|pointee| tyref_is_option_of_root(&pointee, &residual)),
            "fixture Option<RBigIntGcRoot> must be recognized"
        );
        assert_eq!(classify(&call, &residual), Some(Op::OptionAsDeref));
        assert_eq!(residual_escape(&call, &residual, &empty), None);

        let lowered = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![fun_with_body_named(
                0,
                option_method_name("as_deref", &option_of_root()),
                vec![arg.clone()],
                dest.clone(),
            )],
        );
        let deref_call = call_payload(0, arg, dest);
        assert_eq!(classify(&deref_call, &lowered), Some(Op::OptionAsDeref));
        assert_eq!(residual_escape(&deref_call, &lowered, &empty), None);
    }

    fn array_len_3(elem: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"Array": [
            elem,
            [{"Integer": {"Unsigned": ["Usize", "3"]}}, usize_ty()],
            null
        ]})
    }

    fn tuple_of(elem: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"Adt": {
            "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": [elem],
                "const_generics": [],
                "trait_refs": []
            }
        }})
    }

    fn closure_type_decl() -> serde_json::Value {
        let empty = serde_json::json!({
            "regions": [],
            "types": [],
            "const_generics": [],
            "trait_refs": []
        });
        let mut td = type_decl_struct(1, &["fixture", "Env"], vec![]);
        td["src"] = serde_json::json!({"Closure": {"info": {
            "kind": "FnMut",
            "closure_body": {"id": 1, "generics": empty.clone()},
            "fn_mut_impl": {
                "regions": [],
                "regions_outlive": [],
                "types_outlive": [],
                "skip_binder": {"id": 0, "generics": empty}
            },
            "fn_once_impl": null,
            "fn_impl": null
        }}});
        td
    }

    fn call_mut_fun(output: serde_json::Value) -> serde_json::Value {
        let span = serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }});
        let recv = serde_json::json!({"Ref": ["Erased", named_adt(1, vec![]), "Mut"]});
        let args = tuple_of(usize_ty());
        serde_json::json!({
            "def_id": 1,
            "item_meta": fixture_item_meta(ident_path(&["fixture", "call_mut"])),
            "signature": {
                "is_unsafe": false,
                "inputs": [recv.clone(), args.clone()],
                "output": output.clone()
            },
            "body": {"Unstructured": {
                "span": span,
                "locals": {"arg_count": 2, "locals": [
                    {"index": 0, "name": null, "span": span, "ty": output},
                    {"index": 1, "name": null, "span": span, "ty": recv},
                    {"index": 2, "name": null, "span": span, "ty": args}
                ]},
                "body": [{"statements": [], "kind": "Regular",
                    "terminator": {"span": span, "kind": "Return"}}]
            }}
        })
    }

    /// Opaque `array::from_fn` that produces `[RBigIntGcRoot; N]` without a
    /// lowerable closure body stays a residual escape of the handle array.
    #[test]
    fn array_from_fn_of_root_without_closure_is_a_residual_escape() {
        let empty = HashSet::new();
        let array_dest = array_len_3(root_adt());
        let llbc = llbc_with_types(
            "fixture",
            vec![root_type_decl()],
            vec![residual_fun(
                0,
                &["core", "array", "from_fn"],
                vec![unit_ty()],
                array_dest.clone(),
            )],
        );
        let call = call_payload(0, unit_ty(), array_dest);
        assert_eq!(classify(&call, &llbc), None);
        assert_eq!(
            residual_escape(&call, &llbc, &empty).as_deref(),
            Some("core::array::from_fn")
        );
    }

    /// `array::from_fn` of a handle array whose closure `call_mut` is a
    /// local body is `[f(0), …, f(N-1)]`, not a residual escape.
    #[test]
    fn array_from_fn_of_root_with_closure_is_not_a_residual_escape() {
        let empty = HashSet::new();
        let array_dest = array_len_3(root_adt());
        let from_fn = residual_fun(
            0,
            &["core", "array", "from_fn"],
            vec![named_adt(1, vec![])],
            array_dest.clone(),
        );
        let trait_impl = serde_json::json!({
            "def_id": 0,
            "methods": [{
                "kind": {"TraitMethod": [0, 0]},
                "skip_binder": {"id": 1, "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }}
            }]
        });
        let llbc = llbc_with_types_impls(
            "fixture",
            vec![root_type_decl(), closure_type_decl()],
            vec![from_fn, call_mut_fun(root_adt())],
            vec![trait_impl],
        );
        let call = call_payload(0, named_adt(1, vec![]), array_dest);
        assert_eq!(classify(&call, &llbc), None);
        assert_eq!(residual_escape(&call, &llbc, &empty), None);
    }

    fn erased_is_rbigint(ty: &TyRef, llbc: &Llbc) -> bool {
        super::super::tyref_node(ty, llbc)
            .and_then(super::super::adt_node_def_id)
            .is_some_and(|id| type_id_is_rbigint(id, llbc))
    }

    /// An importing crate's handle declaration is Opaque. The erased type
    /// is still `RBigInt`, resolved by its full path.
    #[test]
    fn opaque_handle_erases_to_rbigint_by_path() {
        let llbc = llbc_with_types("fixture", vec![opaque_root_type_decl()], vec![]);
        let erased = erase_tyref(&ty(root_adt()), &llbc).expect("opaque handle erases");
        assert!(
            erased_is_rbigint(&erased, &llbc),
            "Opaque RBigIntGcRoot must erase to RBigInt"
        );
    }

    /// Neither the handle fields nor the `RBigInt` declaration are present.
    /// Classify does not erase values, and the call is an unresolved escape.
    #[test]
    fn opaque_handle_without_rbigint_declines() {
        let empty = HashSet::new();
        let llbc = parse_llbc_types(
            "fixture",
            vec![opaque_root_type_decl()],
            vec![residual_fun(
                0,
                &["fixture", "take_root"],
                vec![root_adt()],
                unit_ty(),
            )],
        );
        let call = call_payload(0, root_adt(), unit_ty());
        assert!(erase_tyref(&ty(root_adt()), &llbc).is_none());
        assert_eq!(classify(&call, &llbc), None);
        assert!(unresolved_escape(&call, &llbc));
        assert_eq!(
            residual_escape(&call, &llbc, &empty).as_deref(),
            Some("fixture::take_root")
        );
    }

    /// `type Root = RBigIntGcRoot` is the handle and erases to `RBigInt`.
    #[test]
    fn alias_of_the_handle_is_recognised_and_erased() {
        let empty = HashSet::new();
        let alias = named_adt(1, vec![]);
        let llbc = llbc_with_types(
            "fixture",
            vec![
                root_type_decl(),
                type_decl_alias(
                    1,
                    &["fixture", "Root"],
                    serde_json::json!({"aliased_ty": root_adt()}),
                ),
            ],
            vec![residual_fun(
                0,
                &["fixture", "take_root"],
                vec![alias.clone()],
                unit_ty(),
            )],
        );
        assert!(
            type_node_is_root(&alias, &llbc),
            "alias of the handle must be recognised"
        );
        let erased = erase_tyref(&ty(alias.clone()), &llbc).expect("alias erases");
        assert!(erased_is_rbigint(&erased, &llbc));
        let call = call_payload(0, alias, unit_ty());
        assert_eq!(
            residual_escape(&call, &llbc, &empty).as_deref(),
            Some("fixture::take_root")
        );
    }

    /// A struct field typed as `type RootRef = &RBigIntGcRoot` still
    /// exposes the handle to a residual callee.
    #[test]
    fn alias_of_ref_to_handle_field_is_a_residual_escape() {
        let empty = HashSet::new();
        let holder = named_adt(2, vec![]);
        let field = serde_json::json!({
            "name": "p",
            "ty": named_adt(1, vec![]),
            "attr_info": null
        });
        let llbc = llbc_with_types(
            "fixture",
            vec![
                root_type_decl(),
                type_decl_alias(
                    1,
                    &["fixture", "RootRef"],
                    serde_json::json!({"aliased_ty": shared_ref(root_adt())}),
                ),
                type_decl_struct(2, &["fixture", "Holder"], vec![field]),
            ],
            vec![residual_fun(
                0,
                &["fixture", "take_holder"],
                vec![holder.clone()],
                unit_ty(),
            )],
        );
        assert!(type_exposes_root(&ty(holder.clone()), &llbc));
        let call = call_payload(0, holder, unit_ty());
        assert_eq!(
            residual_escape(&call, &llbc, &empty).as_deref(),
            Some("fixture::take_holder")
        );
    }

    /// A function that merely lives under `example::option` is not the
    /// inherent `core::option::Option` method. `SimpleCall.eval` keys
    /// special cases by callable identity.
    #[test]
    fn example_option_as_ref_is_not_classified() {
        let arg = shared_ref(option_of_root());
        let dest = named_adt(1, vec![shared_ref(root_adt())]);
        let llbc = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![fun_with_body(
                0,
                &["example", "option", "as_ref"],
                vec![arg.clone()],
                dest.clone(),
            )],
        );
        let call = call_payload(0, arg, dest);
        assert_eq!(classify(&call, &llbc), None);
    }

    /// `is_some` on `&Option<RBigIntGcRoot>` is an ordinary callee:
    /// a bodyless (or `dont_look_inside`) instance is a residual escape,
    /// and an unstructured body is not.
    #[test]
    fn option_is_some_is_judged_like_any_other_callee() {
        let empty = HashSet::new();
        let arg = shared_ref(option_of_root());
        let dest = serde_json::json!({"Scalar": {"Bool": null}});
        let residual = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![residual_fun_named(
                0,
                option_method_name("is_some", &option_of_root()),
                vec![arg.clone()],
                dest.clone(),
            )],
        );
        let call = call_payload(0, arg.clone(), dest.clone());
        assert_eq!(classify(&call, &residual), None);
        assert_eq!(
            residual_escape(&call, &residual, &empty).as_deref(),
            Some("core::option::<Impl>::is_some")
        );

        let lowered = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![fun_with_body_named(
                0,
                option_method_name("is_some", &option_of_root()),
                vec![arg.clone()],
                dest.clone(),
            )],
        );
        assert_eq!(residual_escape(&call, &lowered, &empty), None);

        let opaque_path = super::super::strip_crate_prefix(
            lowered
                .fn_by_id(0)
                .expect("is_some fixture")
                .item_meta
                .name_path_str(),
        );
        let mut dont_look_inside = HashSet::new();
        dont_look_inside.insert(opaque_path);
        assert_eq!(
            residual_escape(&call, &lowered, &dont_look_inside).as_deref(),
            Some("core::option::<Impl>::is_some")
        );
    }

    /// The inherent `core::option::Option::as_deref` is the tagged-to-niche
    /// conversion, identified by impl owner, not by the path substring.
    #[test]
    fn core_option_as_deref_is_classified() {
        let arg = shared_ref(option_of_root());
        let dest = named_adt(1, vec![shared_ref(root_adt())]);
        let llbc = llbc_with_types(
            "fixture",
            vec![root_type_decl(), option_type_decl()],
            vec![residual_fun_named(
                0,
                option_method_name("as_deref", &option_of_root()),
                vec![arg.clone()],
                dest.clone(),
            )],
        );
        let call = call_payload(0, arg, dest);
        assert_eq!(classify(&call, &llbc), Some(Op::OptionAsDeref));
    }
}
