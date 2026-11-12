//! `FrameAnchor` is one `Ref` word in a lowered graph: the frame.
//!
//! Upstream never puts a frame-root slot in a jitcode. The live Python
//! frame is a register (`pyframe.py` `execute_frame` / `MIFrame.force`).
//! pyre spells the host slot as [`PATH`] so native rustc still runs the
//! real LIFO push; this front end keeps the frame reference and none of
//! the slot calls. The shared reasoning is [`super::owner_root_guard`]: a
//! Rust GC-root RAII type is not a jitcode value.
//!
//! `new(frame)` / `from_raw(frame)` / `frame_anchor_push(frame)` is
//! `frame`. `live` / `frame_anchor_live` is the anchor word, which is
//! that same frame. `Drop` of the anchor, and `frame_anchor_release`,
//! emit nothing. `&FrameAnchor` is the word, not an address of the host
//! `{depth}` layout.
//!
//! A callee that still runs as Rust observes the real layout. A pointer
//! to the anchor, or to an aggregate that contains one by value, passed
//! to such a callee declines the graph.

use std::collections::HashSet;

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, CallKind, CallPayload, FunDecl, FunId, NameSeg, TermKind, TyRef, TypeDeclKind,
};

use super::operand_tyref;

/// Charon declaration path of the anchor. Matched in full, never by the leaf.
pub(super) const PATH: &str = "pyre_interpreter::eval::FrameAnchor";

const FRAME_PATH: &str = "pyre_interpreter::pyframe::PyFrame";

const FREE_FNS: &[&str] = &[
    "frame_anchor_push",
    "frame_anchor_live",
    "frame_anchor_release",
    "frame_anchor_push_jit_abi",
    "frame_anchor_live_jit_abi",
    "frame_anchor_release_jit_abi",
    "frame_anchor_new_jit_abi",
    "frame_anchor_live_method_jit_abi",
];

/// An anchor operation the lowering erases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Op {
    /// `FrameAnchor::new` / `from_raw` / `frame_anchor_push` — the result
    /// is the frame argument.
    New,
    /// `FrameAnchor::live` / `frame_anchor_live` — the result is the
    /// anchor word (the frame).
    Live,
    /// `Drop::drop` / `drop_in_place` / `frame_anchor_release`.
    Drop,
}

pub(super) fn is_anchor_path(path: &str) -> bool {
    path == PATH
        || path.ends_with(&format!("::{PATH}"))
        || path == "eval::FrameAnchor"
        || super::strip_crate_prefix(path) == "eval::FrameAnchor"
}

fn is_frame_type_path(path: &str) -> bool {
    path == FRAME_PATH
        || path.ends_with(&format!("::{FRAME_PATH}"))
        || path == "pyframe::PyFrame"
        || super::strip_crate_prefix(path) == "pyframe::PyFrame"
}

fn type_id_is_anchor(id: u64, llbc: &Llbc) -> bool {
    llbc.type_by_id(id)
        .is_some_and(|td| is_anchor_path(td.item_meta.name_path_str()))
}

/// `node` is the anchor ADT, after dedup and hash-cons wrappers.
pub(super) fn type_node_is_anchor(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(node) = super::strip_ty_indirections(node, llbc) else {
        return false;
    };
    super::adt_node_def_id(node).is_some_and(|id| type_id_is_anchor(id, llbc))
}

pub(super) fn tyref_is_anchor(ty: &TyRef, llbc: &Llbc) -> bool {
    super::tyref_node(ty, llbc).is_some_and(|node| type_node_is_anchor(node, llbc))
}

/// `&FrameAnchor` / `&mut FrameAnchor`. A raw pointer is the host layout.
pub(super) fn tyref_is_borrow_of_anchor(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) =
        super::tyref_node(ty, llbc).and_then(|n| super::strip_ty_indirections(n, llbc))
    else {
        return false;
    };
    let Some(arr) = node
        .as_object()
        .and_then(|obj| obj.get("Ref"))
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    arr.get(1)
        .is_some_and(|inner| type_node_is_anchor(inner, llbc))
}

pub(super) fn tyref_is_option_of_anchor(ty: &TyRef, llbc: &Llbc) -> bool {
    crate::front::result_exc::tyref_option_payload(ty, llbc)
        .is_some_and(|payload| tyref_is_anchor(&payload, llbc))
}

/// A drop of this place lowers to nothing: the anchor, or `Option` of it.
pub(super) fn tyref_is_erased_drop(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_anchor(ty, llbc) || tyref_is_option_of_anchor(ty, llbc) {
        return true;
    }
    ref_pointee_ty(ty, llbc).is_some_and(|pointee| {
        tyref_is_anchor(&pointee, llbc) || tyref_is_option_of_anchor(&pointee, llbc)
    })
}

/// Class root of the `PyFrame` pointee `new` takes / `live` returns.
pub(super) fn frame_class_root(
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    super::llbc_derived(llbc)
        .frame_class_root
        .get_or_init(|| {
            let mut found: Option<String> = None;
            for td in llbc.iter_type_decls() {
                if !is_frame_type_path(td.item_meta.name_path_str()) {
                    continue;
                }
                let ty = super::type_decl_adt_tyref(td.def_id);
                let Some(root) = super::tyref_node(&ty, llbc).and_then(|node| {
                    super::adt_node_class_root_with(node, llbc, super::no_tombstoned_leaves())
                }) else {
                    continue;
                };
                match found.as_deref() {
                    None => found = Some(root),
                    Some(existing) if existing == root => {}
                    Some(_) => return None,
                }
            }
            found
        })
        .clone()
        .or_else(|| {
            // A fixture without a `PyFrame` decl still names the pointee
            // when `tombstoned` would have changed the spelling.
            let _ = tombstoned;
            None
        })
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

/// Peel every pointer layer, then say whether the pointee contains an
/// anchor by value. A field that is itself a pointer does not: its width
/// does not change when the anchor's layout does.
pub(super) fn pointer_exposes_anchor(ty: &TyRef, llbc: &Llbc) -> bool {
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
            return true;
        };
        node = next;
    }
    saw_pointer && contains_anchor_by_value(node, llbc, 0, &mut Vec::new())
}

fn contains_anchor_by_value<'a>(
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
    if node.get("Ref").is_some() || node.get("RawPtr").is_some() {
        return false;
    }
    let Some(adt) = node.get("Adt").and_then(serde_json::Value::as_object) else {
        return false;
    };
    if super::adt_node_def_id(node).is_some_and(|id| type_id_is_anchor(id, llbc)) {
        return true;
    }
    if let Some(types) = adt
        .get("generics")
        .and_then(serde_json::Value::as_object)
        .and_then(|generics| generics.get("types"))
        .and_then(serde_json::Value::as_array)
        && types
            .iter()
            .any(|ty| contains_anchor_by_value(ty, llbc, depth + 1, seen))
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
    contains_anchor_by_value(node, llbc, depth, seen)
}

fn is_destructor_leaf(leaf: &str) -> bool {
    matches!(leaf, "drop" | "drop_in_place" | "DropGlue")
}

fn module_of(path: &str) -> &str {
    path.rsplit_once("::")
        .map(|(module, _)| module)
        .unwrap_or(path)
}

fn is_free_anchor_fn(path: &str, leaf: &str) -> bool {
    if !FREE_FNS.contains(&leaf) {
        return false;
    }
    let module = module_of(PATH);
    let stripped = super::strip_crate_prefix(path);
    path == format!("{module}::{leaf}")
        || path.ends_with(&format!("::{module}::{leaf}"))
        || stripped == format!("eval::{leaf}")
}

fn op_from_leaf(leaf: &str) -> Option<Op> {
    match leaf {
        "new"
        | "from_raw"
        | "frame_anchor_push"
        | "frame_anchor_push_jit_abi"
        | "frame_anchor_new_jit_abi" => Some(Op::New),
        "live"
        | "frame_anchor_live"
        | "frame_anchor_live_jit_abi"
        | "frame_anchor_live_method_jit_abi" => Some(Op::Live),
        "drop" | "drop_in_place" | "frame_anchor_release" | "frame_anchor_release_jit_abi" => {
            Some(Op::Drop)
        }
        _ => None,
    }
}

fn arg_is_erased_drop(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_erased_drop(ty, llbc)
}

/// Classify a call the lowering erases, from the callee declaration and the
/// types the call actually passes. The anchor is named by [`PATH`], not by a
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
    let path = fd.item_meta.name_path();
    if impl_self_id(fd, llbc).is_some_and(|id| type_id_is_anchor(id, llbc)) {
        return op_from_leaf(leaf);
    }
    if is_free_anchor_fn(&path, leaf) {
        return op_from_leaf(leaf);
    }
    let dest_anchor = tyref_is_anchor(&call.dest.ty, llbc);
    let recv_anchor = call
        .args
        .iter()
        .next()
        .and_then(operand_tyref)
        .is_some_and(|ty| tyref_is_anchor(ty, llbc) || tyref_is_borrow_of_anchor(ty, llbc));
    if dest_anchor || recv_anchor {
        return op_from_leaf(leaf);
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
    None
}

/// A helper whose host body still pushes the native slot. Erasing it
/// would strip the index it returns; the identity is modeled at the
/// call site. `SharedOpcodeHandler::anchor` is the same skip: output is
/// the frame, and the host `new` inside must not become a returned slot.
pub(super) fn is_helper(fd: &FunDecl, llbc: &Llbc) -> bool {
    let Some(leaf) = fn_leaf(fd) else {
        return false;
    };
    let path = fd.item_meta.name_path();
    if impl_self_id(fd, llbc).is_some_and(|id| type_id_is_anchor(id, llbc)) {
        return matches!(leaf, "new" | "from_raw" | "live" | "drop" | "drop_in_place");
    }
    if is_anchor_path(&path)
        && matches!(leaf, "new" | "from_raw" | "live" | "drop" | "drop_in_place")
    {
        return true;
    }
    if matches!(
        leaf,
        "new" | "from_raw" | "live" | "drop" | "drop_in_place" | "anchor"
    ) && (tyref_is_anchor(&fd.signature.output, llbc)
        || fd
            .signature
            .inputs
            .first()
            .is_some_and(|ty| tyref_is_anchor(ty, llbc) || tyref_is_borrow_of_anchor(ty, llbc)))
    {
        return true;
    }
    is_free_anchor_fn(&path, leaf)
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
    if !fd.has_unstructured_body() {
        return true;
    }
    let path = super::strip_crate_prefix(&fd.item_meta.name_path());
    dont_look_inside.contains(&path)
}

/// `is_some` / `is_none` on `Option<FrameAnchor>` call the core
/// predicate bodies. Those bodies' `Discriminant` fold is a
/// nullable-pointer test, not an escape of the anchor into a residual
/// callee.
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
            tyref_is_option_of_anchor(ty, llbc)
                || ref_pointee_ty(ty, llbc)
                    .is_some_and(|pointee| tyref_is_option_of_anchor(&pointee, llbc))
        })
    })
}

/// `Some(callee)` when `call` passes a pointer to an anchor-containing
/// value into a callee that stays residual. Erased anchor operations
/// are not escapes.
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
        .any(|arg| operand_tyref(arg).is_some_and(|ty| pointer_exposes_anchor(ty, llbc)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::ullbc::{Operand, Place, PlaceKind};
    use serde_json::json;

    fn span() -> serde_json::Value {
        json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    }

    fn ident(s: &str) -> serde_json::Value {
        json!({"Ident": [s, 0]})
    }

    fn item_meta(name: &[&str], def_id: u64) -> serde_json::Value {
        json!({
            "def_id": def_id,
            "item_meta": {
                "name": name.iter().copied().map(ident).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            }
        })
    }

    fn type_decl(def_id: u64, name: &[&str]) -> serde_json::Value {
        let mut td = item_meta(name, def_id);
        td["kind"] = json!({"Struct": []});
        td
    }

    /// Host `FrameAnchor` is `repr(transparent)` over `usize`. The paint
    /// must win over that peel so a declared RESULT is Ref.
    fn frame_anchor_decl(def_id: u64) -> serde_json::Value {
        let mut td = type_decl(def_id, &["pyre_interpreter", "eval", "FrameAnchor"]);
        td["kind"] = json!({"Struct": [{
            "name": "depth",
            "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
            "attr_info": null
        }]});
        td["layout"] = json!([{
            "key": "fixture-target",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"repr_algo": "Rust", "transparent": true}
            }
        }]);
        td
    }

    fn opaque_fun(
        def_id: u64,
        name: &[&str],
        inputs: Vec<serde_json::Value>,
        output: serde_json::Value,
    ) -> serde_json::Value {
        let mut fd = item_meta(name, def_id);
        fd["signature"] = json!({
            "is_unsafe": false,
            "inputs": inputs,
            "output": output
        });
        fd["body"] = json!("Opaque");
        fd
    }

    fn adt(id: u64) -> serde_json::Value {
        json!({
            "Adt": {
                "id": id,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    }

    fn fixture() -> Llbc {
        let frame = adt(1);
        let anchor = adt(0);
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "type_decls": [
                    frame_anchor_decl(0),
                    type_decl(1, &["pyre_interpreter", "pyframe", "PyFrame"])
                ],
                "fun_decls": [
                    null,
                    opaque_fun(
                        1,
                        &["pyre_interpreter", "eval", "FrameAnchor", "new"],
                        vec![frame.clone()],
                        anchor.clone()
                    ),
                    opaque_fun(
                        2,
                        &["pyre_interpreter", "eval", "FrameAnchor", "live"],
                        vec![anchor.clone()],
                        frame.clone()
                    ),
                    opaque_fun(
                        3,
                        &["pyre_interpreter", "eval", "frame_anchor_push"],
                        vec![frame.clone()],
                        json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
                    ),
                    opaque_fun(
                        4,
                        &["pyre_interpreter", "eval", "frame_anchor_live"],
                        vec![json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})],
                        frame.clone()
                    ),
                    opaque_fun(
                        5,
                        &["pyre_interpreter", "eval", "frame_anchor_release"],
                        vec![json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})],
                        json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": {
                            "regions": [], "types": [], "const_generics": [], "trait_refs": []
                        }}})
                    ),
                    opaque_fun(
                        6,
                        &["other", "frame_anchor_push"],
                        vec![frame],
                        json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
                    )
                ]
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc")
    }

    fn call(fun: u64, args: Vec<Operand>, dest_ty: TyRef) -> CallPayload {
        CallPayload {
            func: CallFunc::Regular(majit_charon_reader::ullbc::RegularCall {
                kind: CallKind::Fun(FunId::Regular { id: fun }),
                generics: Default::default(),
            }),
            args,
            dest: Place {
                kind: PlaceKind::Local(0),
                ty: dest_ty,
            },
        }
    }

    fn local_operand(i: u64, ty: TyRef) -> Operand {
        Operand::Copy(Place {
            kind: PlaceKind::Local(i),
            ty,
        })
    }

    #[test]
    fn frame_anchor_path_matches_the_eval_type_only() {
        assert!(is_anchor_path(PATH));
        assert!(is_anchor_path("eval::FrameAnchor"));
        assert!(!is_anchor_path("other::FrameAnchor"));
        assert!(!is_anchor_path("pyre_interpreter::eval::frame_anchor_push"));
    }

    #[test]
    fn frame_anchor_classify_new_live_and_release() {
        let llbc = fixture();
        let frame_ty = TyRef::Other(adt(1));
        let anchor_ty = TyRef::Other(adt(0));
        let usize_ty = TyRef::Other(json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}}));
        assert_eq!(
            classify(
                &call(
                    1,
                    vec![local_operand(1, frame_ty.clone())],
                    anchor_ty.clone()
                ),
                &llbc
            ),
            Some(Op::New)
        );
        assert_eq!(
            classify(
                &call(
                    2,
                    vec![local_operand(1, anchor_ty.clone())],
                    frame_ty.clone()
                ),
                &llbc
            ),
            Some(Op::Live)
        );
        assert_eq!(
            classify(
                &call(
                    3,
                    vec![local_operand(1, frame_ty.clone())],
                    usize_ty.clone()
                ),
                &llbc
            ),
            Some(Op::New)
        );
        assert_eq!(
            classify(
                &call(4, vec![local_operand(1, usize_ty.clone())], frame_ty),
                &llbc
            ),
            Some(Op::Live)
        );
        assert_eq!(
            classify(&call(5, vec![local_operand(1, usize_ty)], anchor_ty), &llbc),
            Some(Op::Drop)
        );
    }

    #[test]
    fn other_frame_anchor_push_is_not_an_anchor_op() {
        let llbc = fixture();
        let frame_ty = TyRef::Other(adt(1));
        let usize_ty = TyRef::Other(json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}}));
        assert_eq!(
            classify(&call(6, vec![local_operand(1, frame_ty)], usize_ty), &llbc),
            None
        );
    }

    #[test]
    fn frame_anchor_tyref_is_the_eval_adt() {
        let llbc = fixture();
        let anchor = TyRef::Other(adt(0));
        let frame = TyRef::Other(adt(1));
        assert!(tyref_is_anchor(&anchor, &llbc));
        assert!(!tyref_is_anchor(&frame, &llbc));
        assert!(tyref_is_erased_drop(&anchor, &llbc));
        assert_eq!(
            frame_class_root(&llbc, crate::front::mir::no_tombstoned_leaves()).as_deref(),
            Some("pyframe::PyFrame")
        );
        let borrowed = TyRef::Other(json!({"Ref": ["Erased", adt(0), "Shared"]}));
        assert!(tyref_is_borrow_of_anchor(&borrowed, &llbc));
        assert!(!tyref_is_borrow_of_anchor(&anchor, &llbc));
        let painted = crate::front::mir::tyref_to_value_type(&anchor, &llbc);
        assert!(
            matches!(painted, crate::model::ValueType::Ref(Some(ref root)) if root.ends_with("PyFrame")),
            "FrameAnchor paints as the frame Ref, got {painted:?}"
        );
        let painted_borrow = crate::front::mir::tyref_to_value_type(&borrowed, &llbc);
        assert!(
            matches!(painted_borrow, crate::model::ValueType::Ref(Some(ref root)) if root.ends_with("PyFrame")),
            "&FrameAnchor is the same Ref, got {painted_borrow:?}"
        );
        let empty_gc: std::collections::HashSet<majit_ir::descr::StructId> =
            std::collections::HashSet::new();
        let empty_raw: std::collections::HashSet<String> = std::collections::HashSet::new();
        assert_eq!(
            crate::front::mir::dont_look_inside_return_token(
                &anchor,
                &llbc,
                crate::ErrorCarrierSpec::default(),
                &empty_gc,
                &empty_raw,
            )
            .as_deref(),
            Some("ref"),
            "declared FUNC.RESULT of a FrameAnchor signature is Ref"
        );
    }

    fn span_stmt() -> serde_json::Value {
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}})
    }

    fn ident_seg(s: &str) -> serde_json::Value {
        json!({"Ident": [s, 0]})
    }

    fn item_meta_path(path: &[&str], is_local: bool) -> serde_json::Value {
        json!({
            "name": path.iter().copied().map(ident_seg).collect::<Vec<_>>(),
            "span": span_stmt(),
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

    /// `new` aliases the frame argument and `Drop` of the dest emits no call.
    #[test]
    fn frame_anchor_new_aliases_the_frame_and_drop_emits_nothing() {
        let frame = adt(1);
        let anchor = adt(0);
        let unit = json!({
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
        });
        let frame_ref = json!({"Ref": ["Erased", frame.clone(), "Mut"]});
        let place = |i: u64, ty: &serde_json::Value| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64, ty: &serde_json::Value| json!({"index": i, "name": null, "span": span_stmt(), "ty": ty});
        let call_new = json!({"Call": {
            "call": {
                "func": {"Regular": {
                    "kind": {"Fun": 1},
                    "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
                }},
                "args": [{"Read": place(1, &frame_ref)}]
            },
            "dest": place(2, &anchor),
            "target": 1,
            "on_unwind": 3
        }});
        let drop_anchor = json!({"Drop": {
            "place": place(2, &anchor),
            "fn_ptr": {"kind": {"Fun": 1}, "generics": {}},
            "target": 2,
            "on_unwind": 3
        }});
        let assign_unit = json!({
            "span": span_stmt(),
            "kind": {"Assign": [place(0, &unit), {"Aggregate": ["Tuple", []]}]}
        });
        let subject = json!({
            "def_id": 0,
            "item_meta": item_meta_path(&["fixture", "frame_anchor_new_and_drop"], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [frame_ref.clone()],
                "output": unit.clone()
            },
            "body": {"Unstructured": {
                "span": span_stmt(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, &unit),
                        local(1, &frame_ref),
                        local(2, &anchor)
                    ]
                },
                "body": [
                    {"statements": [], "terminator": {"span": span_stmt(), "kind": call_new}},
                    {"statements": [], "terminator": {"span": span_stmt(), "kind": drop_anchor}},
                    {"statements": [assign_unit], "terminator": {"span": span_stmt(), "kind": "Return"}},
                    {"statements": [], "terminator": {"span": span_stmt(), "kind": "UnwindResume"}, "kind": "Cleanup"}
                ]
            }}
        });
        let new_fd = json!({
            "def_id": 1,
            "item_meta": item_meta_path(&["pyre_interpreter", "eval", "FrameAnchor", "new"], false),
            "signature": {
                "is_unsafe": false,
                "inputs": [frame_ref],
                "output": anchor
            },
            "body": "Opaque"
        });
        let file = json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [
                    frame_anchor_decl(0),
                    type_decl(1, &["pyre_interpreter", "pyframe", "PyFrame"])
                ],
                "fun_decls": [subject, new_fd],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let graph = crate::test_support::lower_function(&llbc, "frame_anchor_new_and_drop")
            .expect("lower frame_anchor_new_and_drop");
        let leaves = crate::front::mir::tests::graph_call_leaves(&graph);
        assert!(
            leaves.iter().all(|leaf| {
                leaf != "new"
                    && leaf != "from_raw"
                    && leaf != "live"
                    && leaf != "frame_anchor_push"
                    && leaf != "frame_anchor_live"
                    && leaf != "frame_anchor_release"
                    && leaf != "drop"
                    && leaf != "drop_in_place"
            }),
            "FrameAnchor new/Drop must emit no call, got {leaves:?}"
        );
    }
}
