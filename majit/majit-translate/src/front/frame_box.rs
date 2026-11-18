//! `FrameBox` is the GC `PyFrame` word in a lowered graph.
//!
//! `baseobjspace.py createframe` returns the malloc'd `W_Root`. `function.py
//! Function.funccall` does `return new_frame.run`. There is no wrapper
//! object. pyre's `FrameBox` is a host RAII handle (`ptr` plus
//! `OwnerRootGuard`) so the native build can pin the frame until
//! `execute_frame` links it. The JIT already erases `OwnerRootGuard` to that
//! one reference; the handle around it is the same word.
//!
//! Construction `FrameBox { ptr, owner_root: Some(OwnerRootGuard::new(r)) }`
//! is `r`. A field read of `ptr` or `owner_root` is the handle. Drop is
//! nothing: a GC frame is reclaimed by reachability (`pyframe.py class
//! PyFrame(W_Root)`; `executioncontext.py leave` frees nothing).
//!
//! A callee that still runs as Rust observes the real two-word layout. A
//! pointer to the handle passed to such a callee declines the graph the
//! same way a pointer to `OwnerRootGuard` does.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{TyRef, TypeDeclKind};

/// Charon declaration leaf of the handle. Matched with the `pyframe`
/// parent so a user type of the same leaf stays a struct.
const LEAF: &str = "FrameBox";
const PARENT: &str = "pyframe";

pub(super) fn is_frame_box_path(path: &str) -> bool {
    let Some((parent, leaf)) = path.rsplit_once("::") else {
        return false;
    };
    leaf == LEAF && parent.rsplit("::").next() == Some(PARENT)
}

pub(in crate::front::mir) fn owner_is_frame_box(owner: &str) -> bool {
    is_frame_box_path(owner)
}

fn type_id_is_frame_box(id: u64, llbc: &Llbc) -> bool {
    llbc.type_by_id(id)
        .is_some_and(|td| is_frame_box_path(&td.item_meta.name_path()))
}

pub(super) fn type_node_is_frame_box(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(node) = super::strip_ty_indirections(node, llbc) else {
        return false;
    };
    super::adt_node_def_id(node).is_some_and(|id| type_id_is_frame_box(id, llbc))
}

pub(super) fn tyref_is_frame_box(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(mut node) = super::tyref_node(ty, llbc) else {
        return false;
    };
    for _ in 0..8 {
        let Some(stripped) = super::strip_ty_indirections(node, llbc) else {
            return false;
        };
        if type_node_is_frame_box(stripped, llbc) {
            return true;
        }
        let Some(obj) = stripped.as_object() else {
            return false;
        };
        let inner = if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            arr.get(1)
        } else {
            obj.get("RawPtr")
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first())
        };
        let Some(inner) = inner else {
            return false;
        };
        node = inner;
    }
    false
}

/// `ptr` and `owner_root` both store the PyFrame GCREF after the guard
/// is erased. Either projection is the handle.
pub(super) fn field_is_frame_gcref(name: &str) -> bool {
    name == "ptr" || name == "owner_root"
}

/// Operand of the Aggregate that is already the GCREF livevar.
///
/// `owner_root` is `Option<OwnerRootGuard>` — the translated livevar
/// (`frame_ptr` reads that slot). `ptr` is the same address as a raw
/// field. Prefer `owner_root` so a `None` fallback still has `ptr`.
pub(super) fn gcref_operand_index(ty: &TyRef, llbc: &Llbc) -> Option<usize> {
    let node = super::tyref_node(ty, llbc)?;
    let node = super::strip_ty_indirections(node, llbc)?;
    let id = super::adt_node_def_id(node)?;
    let td = llbc.type_by_id(id)?;
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return None;
    };
    let named = |want: &str| {
        fields
            .iter()
            .position(|field| field.name.as_deref() == Some(want))
    };
    named("owner_root").or_else(|| named("ptr"))
}

#[cfg(test)]
mod tests {
    use super::{field_is_frame_gcref, is_frame_box_path, owner_is_frame_box};

    #[test]
    fn frame_box_path_needs_the_pyframe_parent() {
        assert!(is_frame_box_path("pyre_interpreter::pyframe::FrameBox"));
        assert!(is_frame_box_path("pyframe::FrameBox"));
        assert!(owner_is_frame_box("pyframe::FrameBox"));
        assert!(!is_frame_box_path("other::FrameBox"));
        assert!(!is_frame_box_path("FrameBox"));
    }

    #[test]
    fn both_handle_fields_are_the_gcref_word() {
        assert!(field_is_frame_gcref("ptr"));
        assert!(field_is_frame_gcref("owner_root"));
        assert!(!field_is_frame_gcref("pycode"));
    }
}
