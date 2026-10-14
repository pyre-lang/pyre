//! Every translator-seen `DictStrategy` / `SetStrategy` impl is registered
//! on `PYRE_DYN_IMPL_VTABLES`. A new impl without `register_dyn_impl_vtable!`
//! fails this test.

mod common;

use common::{INTERPRETER_LLBC, OBJECT_LLBC, load_llbc_if_present};
use majit_charon_reader::Llbc;
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

fn registered_dyn_impl_globals(llbc: &Llbc) -> Vec<String> {
    llbc.iter_global_decls()
        .map(|gd| gd.item_meta.name_path())
        .filter(|path| path.contains("DYN_IMPL_"))
        .collect()
}

fn impl_is_registered(_trait_leaf: &str, type_leaf: &str, globals: &[String]) -> bool {
    let marker = format!("DYN_IMPL_{type_leaf}");
    globals
        .iter()
        .any(|path| path.rsplit("::").next() == Some(marker.as_str()))
}

#[test]
fn translator_seen_dict_and_set_strategy_impls_are_registered() {
    let Some(object) = load_llbc_if_present(OBJECT_LLBC) else {
        eprintln!("skipping: {OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
        return;
    };
    let Some(interpreter) = load_llbc_if_present(INTERPRETER_LLBC) else {
        eprintln!("skipping: {INTERPRETER_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
        return;
    };
    let mut impls = strategy_impl_leaves(object);
    impls.extend(strategy_impl_leaves(interpreter));
    impls.sort();
    impls.dedup();
    let mut globals = registered_dyn_impl_globals(object);
    globals.extend(registered_dyn_impl_globals(interpreter));
    let missing: Vec<(String, String)> = impls
        .into_iter()
        .filter(|(trait_leaf, type_leaf)| !impl_is_registered(trait_leaf, type_leaf, &globals))
        .collect();
    assert!(
        missing.is_empty(),
        "translator-seen DictStrategy/SetStrategy impls with no register_dyn_impl_vtable! \
         (add the macro at the impl): {missing:?}\nregistered globals: {globals:?}"
    );
}
