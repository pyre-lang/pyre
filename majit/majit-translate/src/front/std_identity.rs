//! Transparent one-field std wrappers and scalar Copy combinators as
//! identity / zero, not residual calls.
//!
//! RPython has no `Cell` / `Ref` / `MutexGuard` / `Box` / `AtomicUsize`
//! types: the field *is* the value.  Charon still emits residual
//! `FunctionPath` / `Method` calls to those bodies (Opaque in the LLBC),
//! and every caller then dies at `translate_op` with an unregistered
//! `CallRegistry` miss.  A pointer cast or a transparent one-field
//! layout is `same_as`.  `Cell::get` / `Cell::set` are a field
//! load/store of that word (`getfield` / `setfield` on the owning
//! struct).  A payload that sits beside other fields is a field read.
//! An opaque layout stays a residual call.  `Default` on a machine-word
//! integer / bool is a typed zero.
//!
//! `core::mem::replace` / `swap` / `take` are not rewritten here.  The
//! front models `&mut T` as the referent value (`Rvalue::Ref` aliases
//! the place's Variable), so the call's first argument is the old `T`,
//! not the slot.  The write needs the borrowed place, recorded on
//! `Lowering::atomic_ref_place` and applied in `lower_call`
//! (`try_lower_mem_exchange`): a read of that place, then a store of
//! the new value.  That is `old = a[i]; a[i] = new`
//! (`getarrayitem_gc`/`setarrayitem_gc` or `getfield_gc`/`setfield_gc`).

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FieldDescriptor, OpKind, ValueType};

fn path_has(segments: &[String], needle: &str) -> bool {
    segments.iter().any(|s| s.as_str() == needle)
}

fn is_std_crate(name: &str) -> bool {
    // `sync::...` is the std lock path with the `std::` prefix already peeled.
    matches!(name, "core" | "std" | "alloc" | "sync")
}

fn is_std_fn_path(segments: &[String]) -> bool {
    segments.first().is_some_and(|seg| is_std_crate(seg))
}

fn is_std_type_path(path: Option<&str>) -> bool {
    path.is_some_and(|p| p.split("::").next().is_some_and(is_std_crate))
}

fn path_leaf(path: Option<&str>) -> Option<&str> {
    path.map(|p| p.rsplit("::").next().unwrap_or(p))
}

fn function_leaf(segments: &[String]) -> Option<&str> {
    segments.last().map(String::as_str)
}

pub(crate) fn is_identity_wrapper_target(
    target: &CallTarget,
    receiver_path: Option<&str>,
    dest_path: Option<&str>,
    receiver_is_thin_box: bool,
) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => {
            if !is_std_type_path(receiver_root.as_deref()) && !is_std_type_path(receiver_path) {
                return false;
            }
            match (name.as_str(), path_leaf(receiver_root.as_deref())) {
                ("get", Some("Cell")) => true,
                ("deref" | "deref_mut", Some("Ref") | Some("MutexGuard")) => true,
                _ => false,
            }
        }
        CallTarget::FunctionPath { segments, .. } => {
            let leaf = function_leaf(segments);
            // `boxed::Box::as_mut` is the decl path after `alloc` is peeled.
            // It is not a std-crate prefix, so the Box-method check runs
            // before the std-path gate.
            if is_builtin_box_as_ref_or_mut_path(segments) {
                return receiver_is_thin_box;
            }
            if !is_std_fn_path(segments) {
                return false;
            }
            let recv = path_leaf(receiver_path);
            let dest = path_leaf(dest_path);
            match leaf {
                Some("new")
                    if dest == Some("Cell")
                        || path_has(segments, "Cell") && !path_has(segments, "RefCell") =>
                {
                    true
                }
                Some("new") if path_has(segments, "atomic") => true,
                Some("get")
                    if recv == Some("Cell")
                        || path_has(segments, "Cell")
                        || is_core_cell_impl_leaf(segments, "get") =>
                {
                    true
                }
                Some("deref" | "deref_mut")
                    if matches!(recv, Some("Ref") | Some("MutexGuard"))
                        || path_has(segments, "MutexGuard")
                        || (path_has(segments, "Ref") && !path_has(segments, "RefCell")) =>
                {
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// The concrete `Copy` impls, spelled in full — the same items
/// [`crate::front::mir::is_core_clone_impls_clone_path`] /
/// [`crate::front::mir::is_core_default_path`] recognise on the
/// `RegularCall` route. Both routes call those predicates.
///
/// A leaf-name match would also take `core::clone::<Impl>::clone` and
/// `core::default::Default::default`, which are the blanket impl and the
/// trait item. Charon does not monomorphise them, so the destination
/// width these folds read describes the declaration and not the
/// instantiation, and folding either one answers for every `T` a caller
/// might have instantiated. A method spelling carries no module path at
/// all, so it can never be shown to be one of these items.
pub(crate) fn is_clone_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            crate::front::mir::is_core_clone_impls_clone_path(&segments.join("::"))
        }
        _ => false,
    }
}

pub(crate) fn is_default_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            crate::front::mir::is_core_default_path(&segments.join("::"))
        }
        _ => false,
    }
}

/// Machine-word integer / bool `Copy` — the only `Clone::clone` /
/// `Default::default` instantiations with one uniform answer.
pub(crate) fn is_scalar_copy_width(
    dest_int_atom: Option<&str>,
    dest_uint_atom: Option<&str>,
    dest_is_bool: bool,
) -> bool {
    dest_is_bool
        || dest_int_atom.is_some_and(crate::front::checked_arith::is_ovf_width_int_atom)
        || dest_uint_atom.is_some_and(crate::front::checked_arith_uint::is_word_sized_uint_atom)
}

/// Rewrite a residual std wrapper call when its low-level type says how.
///
/// `rewrite_op_cast_pointer` forwards to `rewrite_op_same_as`: a pointer
/// cast is identity. A one-field struct whose payload bank equals the
/// wrapper's is that same identity. Any other visible payload is a field
/// read (`getfield`) of `value` / `data`, or of the sole field when the
/// banks differ. An opaque layout declines — aliasing the wrapper register
/// would read the guard, not the payload.
///
/// `banks_agree` is the caller's verdict that the wrapper and the
/// destination occupy the same register bank (`flatten.py getkind`).
/// It gates the identity arm only.
pub(crate) fn lower_std_primitive_op(
    op_kind: OpKind,
    receiver_path: Option<&str>,
    dest_path: Option<&str>,
    dest_int_atom: Option<&str>,
    dest_uint_atom: Option<&str>,
    dest_is_bool: bool,
    banks_agree: bool,
    layout: Option<&[(String, ValueType)]>,
    receiver_is_thin_box: bool,
) -> OpKind {
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = &op_kind
    else {
        return op_kind;
    };
    // `Cell::get` is a load of the inner word (`getfield` on the owning
    // struct). `same_as` would alias the cell address as `T`. Layout may
    // be missing when Charon left `Cell` Opaque; the destination type is
    // still `T`.
    if args.len() == 1 && is_std_cell_method(target, receiver_path, "get") {
        let Some(operand) = args[0].as_variable().cloned() else {
            return op_kind;
        };
        let name = layout
            .and_then(|fields| fields.first())
            .map(|(name, _)| name.as_str())
            .unwrap_or("value");
        return cell_word_read(
            operand,
            name,
            cell_field_owner(receiver_path),
            result_ty.clone(),
        );
    }
    if args.len() == 2 && is_std_cell_method(target, receiver_path, "set") {
        let Some(base) = args[0].as_variable().cloned() else {
            return op_kind;
        };
        let (name, field_ty) = match layout.and_then(|fields| fields.first()) {
            Some((name, field_ty)) => (name.as_str(), field_ty.clone()),
            None => ("value", ValueType::Unknown),
        };
        return cell_word_write(
            base,
            name,
            cell_field_owner(receiver_path),
            args[1].clone(),
            field_ty,
        );
    }
    if args.len() == 1
        && is_identity_wrapper_target(target, receiver_path, dest_path, receiver_is_thin_box)
    {
        let Some(operand) = args[0].as_variable().cloned() else {
            return op_kind;
        };
        let owner = if function_leaf_is(target, "new") {
            dest_path
        } else {
            receiver_path
        };
        // `Box::as_ref` / `Box::as_mut` are a pointer cast whatever the
        // layout is.
        if banks_agree && is_box_as_ref_or_mut(target, receiver_is_thin_box) {
            return same_as(operand, result_ty.clone());
        }
        if let Some(fields) = layout {
            if fields.len() == 1 {
                let (name, field_ty) = &fields[0];
                if banks_agree && same_value_bank(field_ty, result_ty) {
                    return same_as(operand, result_ty.clone());
                }
                if !function_leaf_is(target, "new") {
                    return payload_field_read(operand, name, owner, result_ty.clone());
                }
                return op_kind;
            }
            if !function_leaf_is(target, "new")
                && let Some((name, _)) = fields
                    .iter()
                    .find(|(name, _)| name == "value" || name == "data")
            {
                return payload_field_read(operand, name, owner, result_ty.clone());
            }
            return op_kind;
        }
        return op_kind;
    }
    if args.len() == 1
        && is_clone_target(target)
        && is_scalar_copy_width(dest_int_atom, dest_uint_atom, dest_is_bool)
    {
        let Some(operand) = args[0].as_variable().cloned() else {
            return op_kind;
        };
        return OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand,
            result_ty: result_ty.clone(),
        };
    }
    if args.is_empty() && is_default_target(target) {
        if dest_int_atom.is_some_and(crate::front::checked_arith::is_ovf_width_int_atom) {
            return OpKind::ConstInt(0);
        }
        if dest_uint_atom.is_some_and(crate::front::checked_arith_uint::is_word_sized_uint_atom) {
            return OpKind::ConstUInt(0);
        }
        if dest_is_bool {
            return OpKind::ConstBool(false);
        }
    }
    op_kind
}

fn is_core_cell_impl_leaf(segments: &[String], leaf: &str) -> bool {
    let parts: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        parts.as_slice(),
        ["core", "cell", "<Impl>", l] | ["std", "cell", "<Impl>", l] if *l == leaf
    )
}

fn is_std_cell_method(target: &CallTarget, receiver_path: Option<&str>, leaf: &str) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => {
            name == leaf
                && path_leaf(receiver_root.as_deref()) == Some("Cell")
                && is_std_type_path(receiver_root.as_deref().or(receiver_path))
        }
        CallTarget::FunctionPath { segments, .. } => {
            function_leaf(segments) == Some(leaf)
                && is_std_fn_path(segments)
                && !path_has(segments, "RefCell")
                && (path_has(segments, "Cell")
                    || is_core_cell_impl_leaf(segments, leaf)
                    || path_leaf(receiver_path) == Some("Cell") && is_std_type_path(receiver_path))
        }
        _ => false,
    }
}

fn cell_field_owner(receiver_path: Option<&str>) -> Option<&str> {
    receiver_path
        .filter(|path| is_std_type_path(Some(path)))
        .or(Some("core::cell::Cell"))
}

fn function_leaf_is(target: &CallTarget, leaf: &str) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => function_leaf(segments) == Some(leaf),
        CallTarget::Method { name, .. } => name == leaf,
        _ => false,
    }
}

/// Inherent `alloc::boxed::Box::as_ref` / `as_mut`.
///
/// `impl_method_owner_for_fundecl` records the owner ADT path plus the
/// leaf (`boxed::Box::as_mut` once `alloc` is peeled, or the full
/// `alloc::boxed::Box::as_mut`). A free function whose leaf is `as_mut`
/// is a different decl and stays a call. The receiver still has to be
/// a thin Box (`receiver_is_thin_box`).
fn is_builtin_box_as_ref_or_mut_path(segments: &[String]) -> bool {
    let parts: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        parts.as_slice(),
        ["alloc", "boxed", "Box", "as_ref" | "as_mut"] | ["boxed", "Box", "as_ref" | "as_mut"]
    )
}

fn is_box_as_ref_or_mut(target: &CallTarget, receiver_is_thin_box: bool) -> bool {
    if !receiver_is_thin_box {
        return false;
    }
    match target {
        CallTarget::FunctionPath { segments, .. } => is_builtin_box_as_ref_or_mut_path(segments),
        _ => false,
    }
}

/// Inherent `core::option::Option::as_deref` / `as_deref_mut`.
///
/// `ItemMeta::name_path` renders the inherent impl segment as `<Impl>`
/// (`core::option::<Impl>::as_deref_mut`). `impl_method_owner_for_fundecl`
/// resolves that impl's owner ADT and emits `option::Option::as_deref_mut`.
/// A path that merely contains an `option` segment is not this item.
fn is_option_as_deref_path(segments: &[String]) -> bool {
    let parts: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        parts.as_slice(),
        ["core", "option", "<Impl>", "as_deref" | "as_deref_mut"]
            | ["option", "Option", "as_deref" | "as_deref_mut"]
            | ["core", "option", "Option", "as_deref" | "as_deref_mut"]
            | [
                "core",
                "option",
                "Option",
                "<Impl>",
                "as_deref" | "as_deref_mut"
            ]
    )
}

fn is_option_as_deref(target: &CallTarget, decl_is_option_deref: bool) -> bool {
    if decl_is_option_deref {
        return true;
    }
    match target {
        CallTarget::FunctionPath { segments, .. } => is_option_as_deref_path(segments),
        _ => false,
    }
}

/// `Option::as_deref` / `as_deref_mut` on a null-niche `Option<Box<T>>`
/// (or `Option<&T>` / `Option<&mut T>`). The option word is the payload
/// pointer and `None` is null, so the returned `Option<&T>` is that word.
/// `rmodel.py` nullable pointer (`can_be_none`). A non-niche option stays
/// a residual call: its payload is not the option word.
///
/// `decl_is_option_deref` is the call-site verdict from the callee
/// `FunDecl`: `impl_method_owner_for_fundecl` names
/// `option::Option::as_deref` / `as_deref_mut`. A `CallTarget::Method`
/// keeps only the leaf, so the decl verdict is what identifies that
/// spelling. A `FunctionPath` is identified by `is_option_as_deref_path`.
pub(crate) fn lower_niche_option_deref(
    op_kind: OpKind,
    niche_option_ptr: bool,
    decl_is_option_deref: bool,
) -> OpKind {
    if !niche_option_ptr {
        return op_kind;
    }
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = &op_kind
    else {
        return op_kind;
    };
    if !is_option_as_deref(target, decl_is_option_deref) {
        return op_kind;
    }
    let Some(operand) = args.first().and_then(|arg| arg.as_variable()).cloned() else {
        return op_kind;
    };
    same_as(operand, result_ty.clone())
}

fn same_as(operand: Variable, result_ty: ValueType) -> OpKind {
    OpKind::UnaryOp {
        op: "same_as".to_string(),
        operand,
        result_ty,
    }
}

fn payload_field_read(
    operand: Variable,
    name: &str,
    owner: Option<&str>,
    result_ty: ValueType,
) -> OpKind {
    OpKind::FieldRead {
        base: operand,
        field: FieldDescriptor::new(name, owner.map(str::to_string)),
        ty: result_ty,
        pure: true,
    }
}

/// Interior-mutable cell word. `pure: false` so a later store is visible
/// (`bookkeeper.py` does not treat `UnsafeCell` as `immutablevalue`).
fn cell_word_read(
    operand: Variable,
    name: &str,
    owner: Option<&str>,
    result_ty: ValueType,
) -> OpKind {
    OpKind::FieldRead {
        base: operand,
        field: FieldDescriptor::new(name, owner.map(str::to_string)),
        ty: result_ty,
        pure: false,
    }
}

fn cell_word_write(
    base: Variable,
    name: &str,
    owner: Option<&str>,
    value: crate::model::LinkArg,
    ty: ValueType,
) -> OpKind {
    OpKind::FieldWrite {
        base,
        field: FieldDescriptor::new(name, owner.map(str::to_string)),
        value,
        ty,
    }
}

/// Real JIT banks only. Non-value kinds never compare equal, so an unknown
/// payload cannot become `same_as`.
fn same_value_bank(lhs: &ValueType, rhs: &ValueType) -> bool {
    fn bank(ty: &ValueType) -> Option<u8> {
        match ty {
            ValueType::Int | ValueType::Unsigned | ValueType::Bool => Some(0),
            ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => Some(1),
            ValueType::Float => Some(2),
            ValueType::Void
            | ValueType::State
            | ValueType::Unknown
            | ValueType::Int128
            | ValueType::UInt128
            | ValueType::SingleFloat => None,
        }
    }
    matches!((bank(lhs), bank(rhs)), (Some(a), Some(b)) if a == b)
}

/// `(self_ty, borrowed)` is a core/alloc `Borrow` view whose method body
/// is not extracted: `<T as Borrow<T>>`, `<&T as Borrow<T>>`,
/// `<&mut T as Borrow<T>>`, `<String as Borrow<str>>`,
/// `<Vec<T> as Borrow<[T]>>`.
pub(crate) fn is_identity_borrow_pair(
    self_ty: &serde_json::Value,
    borrowed: &serde_json::Value,
    llbc: &majit_charon_reader::Llbc,
) -> bool {
    let self_ty = peel_ty(self_ty, llbc, 0);
    let borrowed = peel_ty(borrowed, llbc, 0);
    if ref_pointee(self_ty, "Shared").is_some_and(|pointee| ty_eq(pointee, borrowed, llbc))
        || ref_pointee(self_ty, "Mut").is_some_and(|pointee| ty_eq(pointee, borrowed, llbc))
    {
        return true;
    }
    if ty_eq(self_ty, borrowed, llbc) {
        return true;
    }
    if is_named_adt(self_ty, llbc, "alloc::string::String") && is_builtin(borrowed, "Str") {
        return true;
    }
    vec_elem(self_ty, llbc)
        .is_some_and(|elem| slice_elem(borrowed, llbc).is_some_and(|item| ty_eq(elem, item, llbc)))
}

fn peel_ty<'a>(
    v: &'a serde_json::Value,
    llbc: &'a majit_charon_reader::Llbc,
    depth: usize,
) -> &'a serde_json::Value {
    if depth > 8 {
        return v;
    }
    let Some(obj) = v.as_object() else {
        return v;
    };
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc
            .dedup_body(id)
            .map(|body| peel_ty(body, llbc, depth + 1))
            .unwrap_or(v);
    }
    if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return peel_ty(&arr[1], llbc, depth + 1);
    }
    v
}

fn ty_eq(
    lhs: &serde_json::Value,
    rhs: &serde_json::Value,
    llbc: &majit_charon_reader::Llbc,
) -> bool {
    peel_ty(lhs, llbc, 0) == peel_ty(rhs, llbc, 0)
}

fn ref_pointee<'a>(node: &'a serde_json::Value, kind: &str) -> Option<&'a serde_json::Value> {
    let arr = node.get("Ref")?.as_array()?;
    if arr.get(2).and_then(serde_json::Value::as_str) != Some(kind) {
        return None;
    }
    arr.get(1)
}

fn is_builtin(node: &serde_json::Value, name: &str) -> bool {
    node.get("Adt")
        .and_then(crate::front::mir::type_decl_ref_builtin)
        == Some(name)
}

fn is_named_adt(node: &serde_json::Value, llbc: &majit_charon_reader::Llbc, path: &str) -> bool {
    let Some(id) = node
        .get("Adt")
        .and_then(serde_json::Value::as_object)
        .and_then(crate::front::mir::type_decl_ref_adt_id)
    else {
        return false;
    };
    llbc.type_by_id(id)
        .is_some_and(|td| td.item_meta.name_path() == path)
}

fn vec_elem<'a>(
    node: &'a serde_json::Value,
    llbc: &'a majit_charon_reader::Llbc,
) -> Option<&'a serde_json::Value> {
    if !is_named_adt(node, llbc, "alloc::vec::Vec") {
        return None;
    }
    node.pointer("/Adt/generics/types/0")
}

fn slice_elem<'a>(
    node: &'a serde_json::Value,
    llbc: &'a majit_charon_reader::Llbc,
) -> Option<&'a serde_json::Value> {
    let elem = node.get("Slice")?.as_array()?.first()?;
    Some(peel_ty(elem, llbc, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ValueType, call_args};

    fn path(segments: &[&str]) -> CallTarget {
        CallTarget::FunctionPath {
            segments: segments.iter().map(|s| s.to_string()).collect(),
            fun_decl_id: None,
            generic_rust_args: Vec::new(),
        }
    }

    fn call(target: CallTarget, args: Vec<Variable>, result_ty: ValueType) -> OpKind {
        OpKind::Call {
            target,
            args: call_args(args),
            result_ty,
        }
    }

    fn dummy_var() -> Variable {
        let mut g = crate::model::FunctionGraph::new("std_identity_dummy");
        g.push_op_var(g.startblock, OpKind::ConstInt(1), true)
            .unwrap()
    }

    #[test]
    fn cell_get_is_a_load_when_layout_is_unknown() {
        // `Cell::get(&Cell<T>) -> T` is a load of the inner word, even
        // when Charon left `Cell` Opaque without a field list.
        let v = dummy_var();
        let get = lower_std_primitive_op(
            call(
                path(&["core", "cell", "Cell", "get"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            Some("core::cell::Cell"),
            None,
            Some("I64"),
            None,
            false,
            false,
            None,
            false,
        );
        match get {
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                assert_eq!(base, v);
                assert_eq!(field.name, "value");
                assert_eq!(ty, ValueType::Int);
                assert!(!pure);
            }
            other => panic!("Cell::get must be a field load, got {other:?}"),
        }
    }

    #[test]
    fn cell_get_and_atomic_new_are_identity() {
        let v = dummy_var();
        let get = lower_std_primitive_op(
            call(
                path(&["core", "cell", "Cell", "get"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            Some("core::cell::Cell"),
            None,
            Some("I64"),
            None,
            false,
            true,
            Some(&[("value".into(), ValueType::Int)]),
            false,
        );
        match get {
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                assert_eq!(base, v);
                assert_eq!(field.name, "value");
                assert_eq!(ty, ValueType::Int);
                assert!(!pure);
            }
            other => panic!("Cell::get must be a field load, got {other:?}"),
        }

        let new = lower_std_primitive_op(
            call(
                path(&["core", "sync", "atomic", "<Impl>", "new"]),
                vec![v.clone()],
                ValueType::Unsigned,
            ),
            None,
            Some("core::sync::atomic::AtomicUsize"),
            None,
            Some("Usize"),
            false,
            true,
            Some(&[("v".into(), ValueType::Unsigned)]),
            false,
        );
        assert!(
            matches!(new, OpKind::UnaryOp { ref op, .. } if op == "same_as"),
            "AtomicUsize::new is identity on the inner usize"
        );
    }

    #[test]
    fn cell_impl_get_and_set_are_the_inner_word() {
        let v = dummy_var();
        let stored = dummy_var();
        let get = lower_std_primitive_op(
            call(
                path(&["core", "cell", "<Impl>", "get"]),
                vec![v.clone()],
                ValueType::Unsigned,
            ),
            None,
            None,
            None,
            Some("Usize"),
            false,
            false,
            None,
            false,
        );
        match get {
            OpKind::FieldRead {
                base, field, pure, ..
            } => {
                assert_eq!(base, v);
                assert_eq!(field.name, "value");
                assert!(!pure);
            }
            other => panic!("core::cell::<Impl>::get must be a field load, got {other:?}"),
        }

        let set = lower_std_primitive_op(
            call(
                path(&["core", "cell", "Cell", "set"]),
                vec![v.clone(), stored.clone()],
                ValueType::Void,
            ),
            Some("core::cell::Cell"),
            None,
            None,
            None,
            false,
            false,
            Some(&[("value".into(), ValueType::Unsigned)]),
            false,
        );
        match set {
            OpKind::FieldWrite {
                base,
                field,
                value,
                ty,
            } => {
                assert_eq!(base, v);
                assert_eq!(field.name, "value");
                assert_eq!(value.as_variable(), Some(&stored));
                assert_eq!(ty, ValueType::Unsigned);
            }
            other => panic!("Cell::set must be a field store, got {other:?}"),
        }
    }

    #[test]
    fn user_cell_get_stays_a_call() {
        let v = dummy_var();
        let get = lower_std_primitive_op(
            call(path(&["my_crate", "Cell", "get"]), vec![v], ValueType::Int),
            Some("my_crate::Cell"),
            None,
            Some("I64"),
            None,
            false,
            true,
            Some(&[
                ("value".into(), ValueType::Int),
                ("flag".into(), ValueType::Bool),
            ]),
            false,
        );
        assert!(
            matches!(&get, OpKind::Call { .. }),
            "a user Cell::get must not become a field read, got {get:?}"
        );
    }

    #[test]
    fn cell_new_uses_dest_leaf_when_impl_is_erased() {
        let v = dummy_var();
        let lowered = lower_std_primitive_op(
            call(
                path(&["core", "cell", "<Impl>", "new"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            None,
            Some("core::cell::Cell"),
            None,
            None,
            false,
            true,
            Some(&[("value".into(), ValueType::Int)]),
            false,
        );
        assert!(matches!(
            lowered,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));

        let refcell = lower_std_primitive_op(
            call(
                path(&["core", "cell", "<Impl>", "new"]),
                vec![v],
                ValueType::Ref(None),
            ),
            None,
            Some("core::cell::RefCell"),
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(refcell, OpKind::Call { .. }),
            "RefCell::new is not a one-field identity"
        );
    }

    #[test]
    fn box_as_ref_and_mutex_guard_deref_are_identity() {
        let v = dummy_var();
        let as_ref = lower_std_primitive_op(
            call(
                path(&["alloc", "boxed", "Box", "as_ref"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("alloc::boxed::Box"),
            None,
            None,
            None,
            false,
            true,
            None,
            true,
        );
        assert!(matches!(
            as_ref,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));

        let multi = lower_std_primitive_op(
            call(
                path(&["alloc", "boxed", "Box", "as_ref"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("alloc::boxed::Box"),
            None,
            None,
            None,
            false,
            true,
            Some(&[
                ("0".into(), ValueType::Ref(None)),
                ("1".into(), ValueType::Ref(None)),
            ]),
            true,
        );
        assert!(
            matches!(multi, OpKind::UnaryOp { ref op, .. } if op == "same_as"),
            "Box::as_ref stays a pointer cast when the layout is not one field"
        );

        let as_mut = lower_std_primitive_op(
            call(
                path(&["alloc", "boxed", "Box", "as_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("alloc::boxed::Box"),
            None,
            None,
            None,
            false,
            true,
            None,
            true,
        );
        assert!(matches!(
            as_mut,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));
        let peeled = lower_std_primitive_op(
            call(
                path(&["boxed", "Box", "as_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("boxed::Box"),
            None,
            None,
            None,
            false,
            true,
            None,
            true,
        );
        assert!(
            matches!(peeled, OpKind::UnaryOp { ref op, .. } if op == "same_as"),
            "boxed::Box::as_mut is the same pointer cast as alloc::boxed::Box::as_mut, got {peeled:?}"
        );
        let user_box = lower_std_primitive_op(
            call(
                path(&["my_crate", "Box", "as_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("my_crate::Box"),
            None,
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(user_box, OpKind::Call { .. }),
            "a user type named Box::as_mut is not the alloc Box pointer cast, got {user_box:?}"
        );
        let my_box = lower_std_primitive_op(
            call(
                path(&["my_crate", "MyBox", "as_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("my_crate::MyBox"),
            None,
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(my_box, OpKind::Call { .. }),
            "MyBox::as_mut is not rewritten, got {my_box:?}"
        );
        let peeled_without_type = lower_std_primitive_op(
            call(
                path(&["boxed", "Box", "as_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("boxed::Box"),
            None,
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(peeled_without_type, OpKind::Call { .. }),
            "boxed:: spelling without the Box ADT stays a call, got {peeled_without_type:?}"
        );
        let helper_as_mut = lower_std_primitive_op(
            call(
                path(&["helper", "as_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("alloc::boxed::Box"),
            None,
            None,
            None,
            false,
            true,
            None,
            true,
        );
        assert!(
            matches!(helper_as_mut, OpKind::Call { .. }),
            "helper::as_mut on a Box stays a call, got {helper_as_mut:?}"
        );

        let as_deref_mut = lower_niche_option_deref(
            call(
                path(&["core", "option", "<Impl>", "as_deref_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            true,
            false,
        );
        match as_deref_mut {
            OpKind::UnaryOp {
                ref op,
                ref operand,
                ..
            } => {
                assert_eq!(op, "same_as");
                assert_eq!(operand, &v);
            }
            other => {
                panic!("niche Option<Box<T>>::as_deref_mut is the pointer word, got {other:?}")
            }
        }
        let as_deref = lower_niche_option_deref(
            call(
                path(&["option", "Option", "as_deref"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            true,
            false,
        );
        assert!(matches!(
            as_deref,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));
        // `CallTarget::Method` keeps the leaf. The fold runs only when the
        // callee decl is `option::Option::as_deref`.
        let method_leaf = lower_niche_option_deref(
            call(
                CallTarget::method("as_deref_mut", Some("Option".into())),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            true,
            false,
        );
        assert!(
            matches!(method_leaf, OpKind::Call { .. }),
            "an Option leaf without the decl path stays a call, got {method_leaf:?}"
        );
        let method_decl = lower_niche_option_deref(
            call(
                CallTarget::method("as_deref_mut", Some("Option".into())),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            true,
            true,
        );
        assert!(
            matches!(method_decl, OpKind::UnaryOp { ref op, .. } if op == "same_as"),
            "the decl-identified Option::as_deref_mut is the pointer word, got {method_decl:?}"
        );
        let foreign_option = lower_niche_option_deref(
            call(
                path(&["my_crate", "option", "as_deref_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            true,
            false,
        );
        assert!(
            matches!(foreign_option, OpKind::Call { .. }),
            "my_crate::option::as_deref_mut stays a call, got {foreign_option:?}"
        );
        let tagged = lower_niche_option_deref(
            call(
                path(&["core", "option", "Option", "as_deref_mut"]),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            false,
            false,
        );
        assert!(
            matches!(tagged, OpKind::Call { .. }),
            "a non-niche Option::as_deref_mut stays a call"
        );

        let deref = lower_std_primitive_op(
            call(
                CallTarget::method("deref", Some("MutexGuard".into())),
                vec![v.clone()],
                ValueType::Ref(Some("Vec".into())),
            ),
            Some("sync::poison::mutex::MutexGuard"),
            None,
            None,
            None,
            false,
            true,
            Some(&[
                ("data".into(), ValueType::Ref(Some("Vec".into()))),
                ("poison".into(), ValueType::Ref(None)),
            ]),
            false,
        );
        match deref {
            OpKind::FieldRead { field, base, .. } => {
                assert_eq!(field.name, "data");
                assert_eq!(base, v);
            }
            other => panic!("expected a field read of MutexGuard.data, got {other:?}"),
        }
    }

    #[test]
    fn ref_deref_reads_value_field_not_the_guard() {
        let v = dummy_var();
        let deref = lower_std_primitive_op(
            call(
                path(&["core", "cell", "Ref", "deref"]),
                vec![v.clone()],
                ValueType::Ref(Some("Vec".into())),
            ),
            Some("core::cell::Ref"),
            None,
            None,
            None,
            false,
            true,
            Some(&[
                ("value".into(), ValueType::Ref(Some("Vec".into()))),
                ("borrow".into(), ValueType::Ref(None)),
            ]),
            false,
        );
        match deref {
            OpKind::FieldRead { field, base, .. } => {
                assert_eq!(field.name, "value");
                assert_eq!(base, v);
            }
            other => panic!("expected a field read of Ref.value, got {other:?}"),
        }
    }

    #[test]
    fn word_sized_clone_and_default_lower_to_identity_and_zero() {
        let v = dummy_var();
        let clone = lower_std_primitive_op(
            call(
                path(&["core", "clone", "impls", "<Impl>", "clone"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
            None,
            false,
        );
        assert!(matches!(
            clone,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));

        let default_i = lower_std_primitive_op(
            call(
                path(&["core", "default", "<Impl>", "default"]),
                vec![],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
            None,
            false,
        );
        assert!(matches!(default_i, OpKind::ConstInt(0)));

        let default_u = lower_std_primitive_op(
            call(
                path(&["core", "default", "<Impl>", "default"]),
                vec![],
                ValueType::Unsigned,
            ),
            None,
            None,
            None,
            Some("Usize"),
            false,
            true,
            None,
            false,
        );
        assert!(matches!(default_u, OpKind::ConstUInt(0)));

        let default_bool = lower_std_primitive_op(
            call(
                path(&["core", "default", "<Impl>", "default"]),
                vec![],
                ValueType::Bool,
            ),
            None,
            None,
            None,
            None,
            true,
            true,
            None,
            false,
        );
        assert!(matches!(default_bool, OpKind::ConstBool(false)));
    }

    #[test]
    fn adt_clone_default_and_mem_replace_stay_residual() {
        let v = dummy_var();
        let clone_adt = lower_std_primitive_op(
            call(
                CallTarget::method("clone", Some("W_ListObject".into())),
                vec![v.clone()],
                ValueType::Ref(Some("W_ListObject".into())),
            ),
            Some("pyre_object::listobject::W_ListObject"),
            Some("pyre_object::listobject::W_ListObject"),
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(matches!(clone_adt, OpKind::Call { .. }));

        let tuple_default = lower_std_primitive_op(
            call(
                path(&["core", "tuple", "<Impl>", "default"]),
                vec![],
                ValueType::Ref(None),
            ),
            None,
            Some("Tuple"),
            None,
            None,
            false,
            true,
            None,
            false,
        );
        assert!(matches!(tuple_default, OpKind::Call { .. }));

        let replace = call(
            path(&["core", "mem", "replace"]),
            vec![v.clone(), dummy_var()],
            ValueType::Int,
        );
        let replace = lower_std_primitive_op(
            replace,
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(replace, OpKind::Call { .. }),
            "replace cannot lower without the borrowed Place"
        );
    }

    #[test]
    fn i32_clone_is_not_word_sized_copy() {
        let v = dummy_var();
        let clone = lower_std_primitive_op(
            call(
                path(&["core", "clone", "impls", "<Impl>", "clone"]),
                vec![v],
                ValueType::Int,
            ),
            None,
            None,
            Some("I32"),
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(clone, OpKind::Call { .. }),
            "i32 is not the machine-word Copy width"
        );
    }

    /// The blanket impl and the trait item share the leaf name with the
    /// concrete `Copy` impls and arrive at the machine word width, so a
    /// leaf-name match folds them too. `mir.rs` guards the same pair
    /// through `lower_function`; this pins the predicate itself.
    #[test]
    fn the_blanket_clone_and_the_default_trait_item_stay_residual() {
        let v = dummy_var();
        let blanket = lower_std_primitive_op(
            call(
                path(&["core", "clone", "<Impl>", "clone"]),
                vec![v],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(blanket, OpKind::Call { .. }),
            "core::clone::<Impl>::clone is not the impls identity"
        );

        let trait_item = lower_std_primitive_op(
            call(
                path(&["core", "default", "Default", "default"]),
                vec![],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
            None,
            false,
        );
        assert!(
            matches!(trait_item, OpKind::Call { .. }),
            "Default::default names no impl to read a zero from"
        );
    }

    #[test]
    fn clone_and_default_targets_use_the_regular_call_predicates() {
        let clone_impl = path(&["core", "clone", "impls", "<Impl>", "clone"]);
        let blanket = path(&["core", "clone", "<Impl>", "clone"]);
        let default_impl = path(&["core", "default", "<Impl>", "default"]);
        let ptr_default = path(&["core", "ptr", "mut_ptr", "<Impl>", "default"]);
        let trait_item = path(&["core", "default", "Default", "default"]);
        assert!(is_clone_target(&clone_impl));
        assert!(!is_clone_target(&blanket));
        assert!(is_default_target(&default_impl));
        assert!(is_default_target(&ptr_default));
        assert!(!is_default_target(&trait_item));
        assert_eq!(
            is_clone_target(&clone_impl),
            crate::front::mir::is_core_clone_impls_clone_path("core::clone::impls::<Impl>::clone")
        );
        assert_eq!(
            is_default_target(&ptr_default),
            crate::front::mir::is_core_default_path("core::ptr::mut_ptr::<Impl>::default")
        );
    }
}
