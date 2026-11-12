//! Every translator-seen `DictStrategy` / `SetStrategy` impl is registered
//! on `PYRE_DYN_IMPL_VTABLES`. A new impl without `register_dyn_impl_vtable!`
//! fails this test.

mod common;

use common::{INTERPRETER_LLBC, OBJECT_LLBC, load_llbc_if_present};
use majit_charon_reader::ullbc::{Operand, PlaceKind, Rvalue, StmtKind};
use majit_charon_reader::{GlobalDecl, Llbc};
use serde_json::Value;

fn strategy_impl_leaves(llbc: &Llbc) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for row in llbc.trait_impls_raw() {
        let Some(impl_trait) = row.get("impl_trait") else {
            continue;
        };
        let Some(trait_id) = impl_trait.get("id").and_then(Value::as_u64) else {
            continue;
        };
        let Some(decl) = llbc.trait_by_id(trait_id) else {
            continue;
        };
        let trait_path = decl.item_meta.name_path();
        let trait_leaf = trait_path.rsplit("::").next().unwrap_or(&trait_path);
        let concrete_trait = matches!(trait_leaf, "DictStrategy" | "SetStrategy");
        let unwrapped_set = trait_leaf == "AbstractUnwrappedSetStrategy";
        if !concrete_trait && !unwrapped_set {
            continue;
        }
        let Some(self_ty) = impl_trait
            .get("generics")
            .and_then(|generics| generics.get("types"))
            .and_then(Value::as_array)
            .and_then(|types| types.first())
        else {
            continue;
        };
        let Some(leaf) = ty_adt_leaf(llbc, self_ty) else {
            continue;
        };
        let registered_trait = if unwrapped_set {
            "SetStrategy"
        } else {
            trait_leaf
        };
        out.push((registered_trait.to_string(), leaf));
    }
    out
}

fn ty_adt_leaf(llbc: &Llbc, ty: &Value) -> Option<String> {
    let obj = ty.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
        let def_id = llbc.dedup_to_adt_def_id(id)?;
        let td = llbc.type_by_id(def_id)?;
        return Some(
            td.item_meta
                .name_path()
                .rsplit("::")
                .next()
                .unwrap_or("")
                .to_string(),
        );
    }
    if let Some(arr) = obj.get("Value").and_then(Value::as_array)
        && let Some(body) = arr.get(1)
        && let Some(def_id) = body
            .get("Adt")
            .and_then(|adt| adt.get("id"))
            .and_then(Value::as_u64)
    {
        let td = llbc.type_by_id(def_id)?;
        return Some(
            td.item_meta
                .name_path()
                .rsplit("::")
                .next()
                .unwrap_or("")
                .to_string(),
        );
    }
    None
}

fn operand_str_literal(llbc: &Llbc, op: &Operand, local_strs: &[Option<String>]) -> Option<String> {
    match op {
        Operand::Const(expr) => llbc
            .const_expr_literal(expr)?
            .get("Str")?
            .as_str()
            .map(str::to_string),
        Operand::Copy(place) | Operand::Move(place) => match &place.kind {
            PlaceKind::Local(i) => local_strs.get(*i as usize).cloned().flatten(),
            _ => None,
        },
    }
}

fn dyn_impl_desc_strings(llbc: &Llbc, gd: &GlobalDecl) -> Option<(String, String)> {
    let init_id = gd.init_fun_id()?;
    let body = llbc.fn_by_id(init_id)?.unstructured()?;
    let n = body.locals.locals.len();
    let mut local_strs = vec![None; n];
    for block in &body.body {
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind() else {
                continue;
            };
            let PlaceKind::Local(dest) = place.kind else {
                continue;
            };
            let dest = dest as usize;
            if dest >= n {
                continue;
            }
            if let Rvalue::Use(Operand::Const(expr), _) = rvalue {
                local_strs[dest] = llbc
                    .const_expr_literal(&expr)
                    .and_then(|lit| lit.get("Str")?.as_str().map(str::to_string));
            }
        }
    }
    for block in &body.body {
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(_, Rvalue::Aggregate(_, operands))) = stmt.stmt_kind() else {
                continue;
            };
            let Some(impl_key) = operands
                .first()
                .and_then(|op| operand_str_literal(llbc, op, &local_strs))
            else {
                continue;
            };
            let Some(trait_leaf) = operands
                .get(1)
                .and_then(|op| operand_str_literal(llbc, op, &local_strs))
            else {
                continue;
            };
            return Some((impl_key, trait_leaf));
        }
    }
    None
}

fn dyn_impl_vtable_desc_adt_path(path: &str) -> bool {
    path == "pyre_object::lltype::DynImplVtableDesc"
}

fn global_ty_is_dyn_impl_vtable_desc(llbc: &Llbc, gd: &GlobalDecl) -> bool {
    llbc.global_ty_adt_def_id(gd)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| dyn_impl_vtable_desc_adt_path(&td.item_meta.name_path()))
}

fn registered_dyn_impls(llbc: &Llbc) -> Vec<(String, String)> {
    llbc.iter_global_decls()
        .filter_map(|gd| {
            if !global_ty_is_dyn_impl_vtable_desc(llbc, gd) {
                return None;
            }
            let (impl_key, trait_leaf) = dyn_impl_desc_strings(llbc, gd)?;
            let type_leaf = impl_key.rsplit("::").next()?.to_string();
            Some((trait_leaf, type_leaf))
        })
        .collect()
}

fn impl_is_registered(trait_leaf: &str, type_leaf: &str, registered: &[(String, String)]) -> bool {
    registered
        .iter()
        .any(|(t, ty)| t == trait_leaf && ty == type_leaf)
}

#[test]
fn translator_seen_dict_and_set_strategy_impls_are_registered() {
    let object = load_llbc_if_present(OBJECT_LLBC).unwrap_or_else(|| {
        panic!("{OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
    });
    let interpreter = load_llbc_if_present(INTERPRETER_LLBC).unwrap_or_else(|| {
        panic!("{INTERPRETER_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
    });
    let mut impls = strategy_impl_leaves(object);
    impls.extend(strategy_impl_leaves(interpreter));
    impls.sort();
    impls.dedup();
    let mut registered = registered_dyn_impls(object);
    registered.extend(registered_dyn_impls(interpreter));
    let missing: Vec<(String, String)> = impls
        .into_iter()
        .filter(|(trait_leaf, type_leaf)| !impl_is_registered(trait_leaf, type_leaf, &registered))
        .collect();
    assert!(
        missing.is_empty(),
        "translator-seen DictStrategy/SetStrategy impls with no register_dyn_impl_vtable! \
         (add the macro at the impl): {missing:?}\nregistered: {registered:?}"
    );
}
