use super::{
    FnPtrFamily, collect_fnptr_family_inputs, discover_fnptr_field_families, fn_ptr_family_for,
    profilefunc_graphs_for_call, publish_fnptr_field_families_across, resolve_fnptr_field_families,
};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::Operand;

use serde_json::json;

use crate::front::mir::tests::{
    adt_node, assign_stmt, bb, call_term, const_artifact, deref_place, empty_generics,
    fixture_llbc, fmt_arguments_adt, fmt_arguments_decl, fmt_format_decl,
    format_as_bytes_window_llbc, hint_must_use_decl, item_meta_json, llbc_with_trait_impls,
    llbc_with_types, local_of, move_op, opaque_decl, option_adt_node, option_enum, place_of,
    ptr_ty, ref_rvalue, ref_string, scalar_ty, span_json, string_adt, string_as_bytes_decl,
    string_decl, subject_decl, type_decl_alias, type_decl_struct, unit_fn_ptr, unit_ty,
    window_shared_slice, window_slice,
};

/// Every row of the fn-pointer family decision, including the two the
/// fixture corpus structurally cannot reach: it declares one fn-pointer
/// alias (`HostCallback`) and that alias is safe, so no corpus-driven
/// test distinguishes `is_safe` either way.
///
/// The load-bearing row is `(unsafe, builtin shape) -> None`. Its
/// positive control is the row above it: drop the safety gate and
/// `(safe, builtin shape)` still passes while that row flips to
/// `Some(BuiltinWrapper)`, so the pair fails in exactly one direction.
#[test]
fn fn_ptr_family_names_the_wrapper_set_only_for_a_safe_pointer() {
    // (is_safe, is_builtin_shape, fnptr_indirect) -> family
    let rows = [
        (true, true, false, Some(FnPtrFamily::BuiltinWrapper)),
        (true, true, true, Some(FnPtrFamily::BuiltinWrapper)),
        (false, true, false, None),
        (false, true, true, None),
        (true, false, true, Some(FnPtrFamily::Unknown)),
        // An `unsafe fn` outside the wrapper shape keeps the unknown
        // family: `None` graphs claims nothing about the callee.
        (false, false, true, Some(FnPtrFamily::Unknown)),
        (true, false, false, None),
        (false, false, false, None),
    ];
    for (is_safe, is_builtin_shape, fnptr_indirect, expected) in rows {
        assert_eq!(
            fn_ptr_family_for(is_safe, is_builtin_shape, fnptr_indirect),
            expected,
            "is_safe={is_safe} is_builtin_shape={is_builtin_shape} \
                 fnptr_indirect={fnptr_indirect}"
        );
    }
}

fn fnptr_ty() -> serde_json::Value {
    unit_fn_ptr()
}

fn adt_ty(id: u64) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": id,
            "generics": empty_generics()
        }
    })
}

fn fn_def_const(fun_id: u64) -> serde_json::Value {
    serde_json::json!({"Const": [{
        "FnDef": {
            "kind": {"Fun": fun_id},
            "generics": empty_generics()
        }
    }, fnptr_ty()]})
}

fn fn_def_to_fn_ptr(operand: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"UnaryOp": [{"Cast": "FnDefToFnPtr"}, operand]})
}

fn local_ty(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "index": index,
        "name": null,
        "span": span_json(),
        "ty": ty
    })
}

fn typed_local(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"kind": {"Local": index}, "ty": ty})
}

fn move_local(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Move": typed_local(index, ty)})
}

fn copy_local(index: u64, ty: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({"Read": typed_local(index, ty)})
}

fn assign_local(
    index: u64,
    ty: &serde_json::Value,
    rvalue: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "span": span_json(),
        "kind": {"Assign": [typed_local(index, ty), rvalue]}
    })
}

fn unit_fun(def_id: u64, path: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [local_ty(0, &unit_ty())]},
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn table_type(def_id: u64, path: &[&str], fields: &[&str]) -> serde_json::Value {
    let field_decls: Vec<serde_json::Value> = fields
        .iter()
        .map(|name| {
            serde_json::json!({
                "name": name,
                "ty": fnptr_ty(),
                "attr_info": null
            })
        })
        .collect();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "kind": {"Struct": field_decls}
    })
}

fn table_static(def_id: u64, path: &[&str], type_id: u64, init_fun: u64) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "global_kind": "Static",
        "ty": adt_ty(type_id),
        "value": {"Value": [0, [{"Call": [{"kind": {"Fun": init_fun}, "generics": empty_generics()}, []]}, adt_ty(type_id)]]}
    })
}

fn table_initializer(
    def_id: u64,
    path: &[&str],
    global_id: u64,
    type_id: u64,
    field_rvalues: Vec<serde_json::Value>,
) -> serde_json::Value {
    table_aggregate_fun(
        def_id,
        path,
        serde_json::json!({"GlobalInitializer": {"id": global_id}}),
        type_id,
        field_rvalues,
    )
}

fn table_runtime_constructor(
    def_id: u64,
    path: &[&str],
    type_id: u64,
    field_rvalues: Vec<serde_json::Value>,
) -> serde_json::Value {
    table_aggregate_fun(
        def_id,
        path,
        serde_json::json!("Normal"),
        type_id,
        field_rvalues,
    )
}

fn table_aggregate_fun(
    def_id: u64,
    path: &[&str],
    src: serde_json::Value,
    type_id: u64,
    field_rvalues: Vec<serde_json::Value>,
) -> serde_json::Value {
    let struct_ty = adt_ty(type_id);
    let n_fields = field_rvalues.len() as u64;
    let mut locals = vec![local_ty(0, &struct_ty)];
    for i in 1..=n_fields {
        locals.push(local_ty(i, &fnptr_ty()));
    }
    let mut statements = Vec::new();
    for (i, rvalue) in field_rvalues.into_iter().enumerate() {
        statements.push(assign_local(i as u64 + 1, &fnptr_ty(), rvalue));
    }
    let operands: Vec<serde_json::Value> =
        (1..=n_fields).map(|i| move_local(i, &fnptr_ty())).collect();
    statements.push(assign_local(
        0,
        &struct_ty,
        serde_json::json!({"Aggregate": [{"Adt": [type_id, null]}, operands]}),
    ));
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": struct_ty},
        "src": src,
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": locals},
            "body": [{
                "statements": statements,
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

/// Charon indexes `*_decls` by `def_id` (`fn_by_id` / `type_by_id` /
/// `global_by_id` are O(1) array lookups). Pad unused slots with `null`.
fn pad_by_def_id(decls: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let max_id = decls
        .iter()
        .filter_map(|decl| decl.get("def_id")?.as_u64())
        .max()
        .unwrap_or(0);
    let mut out = vec![serde_json::Value::Null; max_id as usize + 1];
    for decl in decls {
        let id = decl
            .get("def_id")
            .and_then(serde_json::Value::as_u64)
            .expect("decl def_id") as usize;
        out[id] = decl;
    }
    out
}

fn fnptr_family_llbc(
    types: Vec<serde_json::Value>,
    funs: Vec<serde_json::Value>,
    globals: Vec<serde_json::Value>,
) -> Llbc {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "c",
            "type_decls": pad_by_def_id(types),
            "fun_decls": pad_by_def_id(funs),
            "global_decls": pad_by_def_id(globals),
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fnptr family fixture parses")
}

fn family_members<'a>(
    rows: &'a [(String, String, String)],
    struct_path: &str,
    field: &str,
) -> Vec<&'a str> {
    rows.iter()
        .filter(|(path, name, _)| path == struct_path && name == field)
        .map(|(_, _, callpath)| callpath.as_str())
        .collect()
}

/// Families follow the FunDecl stored in the field (`row_of_graphs`), not
/// a FunDecl whose last path segment equals the field name.
#[test]
fn fnptr_field_family_follows_stored_functions_not_field_names() {
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["left", "right"])],
        vec![
            unit_fun(1, &["c", "mod", "TABLE", "foo"]),
            unit_fun(2, &["c", "mod", "bar"]),
            unit_fun(3, &["c", "mod", "TABLE", "left"]),
            table_initializer(
                20,
                &["c", "mod", "TABLE"],
                10,
                0,
                vec![
                    fn_def_to_fn_ptr(fn_def_const(1)),
                    fn_def_to_fn_ptr(fn_def_const(2)),
                ],
            ),
        ],
        vec![table_static(10, &["c", "mod", "TABLE"], 0, 20)],
    );
    let rows = discover_fnptr_field_families(&llbc);
    // Members are registered_path_for_fun_decl: the FunDecl identity keeps the crate Ident.
    assert_eq!(
        family_members(&rows, "mod::Table", "left"),
        ["c::mod::TABLE::foo"],
        "left stores nested foo, not a FunDecl named left: {rows:?}"
    );
    assert_eq!(
        family_members(&rows, "mod::Table", "right"),
        ["c::mod::bar"],
        "right stores top-level bar: {rows:?}"
    );
    assert!(
        !rows.iter().any(|(_, _, path)| path == "mod::TABLE::left"),
        "a FunDecl named left that is not stored is not a family member: {rows:?}"
    );
}

/// Swapping the initializer operands swaps the families. The left slot
/// peels `Use` of an `FnDef` then `FnDefToFnPtr` of that copy.
#[test]
fn fnptr_field_family_follows_swapped_initializer() {
    let llbc = swapped_use_peel_llbc();
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::Table", "left"),
        ["c::mod::bar"],
        "swapped left stores bar: {rows:?}"
    );
    assert_eq!(
        family_members(&rows, "mod::Table", "right"),
        ["c::mod::foo"],
        "swapped right stores foo: {rows:?}"
    );
}

fn swapped_use_peel_llbc() -> Llbc {
    let struct_ty = adt_ty(0);
    let fn_ty = fnptr_ty();
    let statements = vec![
        assign_local(
            3,
            &fn_ty,
            serde_json::json!({"Use": [fn_def_const(2), "No"]}),
        ),
        assign_local(1, &fn_ty, fn_def_to_fn_ptr(copy_local(3, &fn_ty))),
        assign_local(2, &fn_ty, fn_def_to_fn_ptr(fn_def_const(1))),
        assign_local(
            0,
            &struct_ty,
            serde_json::json!({"Aggregate": [{"Adt": [0, null]}, [
                move_local(1, &fn_ty),
                move_local(2, &fn_ty)
            ]]}),
        ),
    ];
    let init = serde_json::json!({
        "def_id": 20,
        "item_meta": item_meta_json(&["c", "mod", "TABLE"], "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": struct_ty},
        "src": {"GlobalInitializer": {"id": 10}},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                local_ty(0, &struct_ty),
                local_ty(1, &fn_ty),
                local_ty(2, &fn_ty),
                local_ty(3, &fn_ty)
            ]},
            "body": [{
                "statements": statements,
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    });
    fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["left", "right"])],
        vec![
            unit_fun(1, &["c", "mod", "foo"]),
            unit_fun(2, &["c", "mod", "bar"]),
            init,
        ],
        vec![table_static(10, &["c", "mod", "TABLE"], 0, 20)],
    )
}

/// Two types that share a leaf name keep distinct family keys.
#[test]
fn fnptr_field_family_keys_use_full_defining_type_path() {
    let llbc = fnptr_family_llbc(
        vec![
            table_type(0, &["c", "a", "Table"], &["slot"]),
            table_type(1, &["c", "b", "Table"], &["slot"]),
        ],
        vec![
            unit_fun(1, &["c", "a", "foo"]),
            unit_fun(2, &["c", "b", "bar"]),
            table_initializer(
                20,
                &["c", "a", "A"],
                10,
                0,
                vec![fn_def_to_fn_ptr(fn_def_const(1))],
            ),
            table_initializer(
                21,
                &["c", "b", "B"],
                11,
                1,
                vec![fn_def_to_fn_ptr(fn_def_const(2))],
            ),
        ],
        vec![
            table_static(10, &["c", "a", "A"], 0, 20),
            table_static(11, &["c", "b", "B"], 1, 21),
        ],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "a::Table", "slot"),
        ["c::a::foo"],
        "a::Table.slot family: {rows:?}"
    );
    assert_eq!(
        family_members(&rows, "b::Table", "slot"),
        ["c::b::bar"],
        "b::Table.slot family: {rows:?}"
    );
    assert!(
        !rows.iter().any(|(path, _, _)| path == "Table"),
        "family key keeps the defining type path, not the leaf: {rows:?}"
    );
    assert!(
        super::fnptr_field_family_contains(&llbc, "a::Table", "slot"),
        "lookup uses the full key"
    );
    assert!(
        !super::fnptr_field_family_contains(&llbc, "Table", "slot"),
        "the leaf alone is not a family key"
    );
}

/// A static that stores `f` plus a runtime constructor that stores `g`
/// closes the family over both (`s_pbc.descriptions` is the union).
#[test]
fn fnptr_field_family_unions_static_and_runtime_stores() {
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            table_initializer(
                20,
                &["c", "mod", "TABLE"],
                10,
                0,
                vec![fn_def_to_fn_ptr(fn_def_const(1))],
            ),
            table_runtime_constructor(
                21,
                &["c", "mod", "make"],
                0,
                vec![fn_def_to_fn_ptr(fn_def_const(2))],
            ),
        ],
        vec![table_static(10, &["c", "mod", "TABLE"], 0, 20)],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"],
        "static f and runtime g: {rows:?}"
    );
}

/// A runtime store of a parameter with no Regular call site is not a
/// recoverable FunDecl, so the family is unpublished even when a static
/// also stores `f`.
#[test]
fn fnptr_field_family_unpublished_when_a_store_is_not_a_fndef() {
    let table = adt_ty(0);
    let table_ptr = serde_json::json!({"RawPtr": [table.clone(), "Mut"]});
    let fn_ty = fnptr_ty();
    let field_place = serde_json::json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [
                {"kind": {"Local": 1}, "ty": table_ptr.clone()},
                "Deref"
            ]}, "ty": table.clone()},
            {"Field": [null, 0]}
        ]},
        "ty": fn_ty.clone()
    });
    let poison = serde_json::json!({
        "def_id": 21,
        "item_meta": item_meta_json(&["c", "mod", "install"], "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [table_ptr.clone(), fn_ty.clone()],
            "output": unit_ty()
        },
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 2, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &table_ptr),
                local_ty(2, &fn_ty)
            ]},
            "body": [{
                "statements": [serde_json::json!({
                    "span": span_json(),
                    "kind": {"Assign": [field_place, serde_json::json!({"Use": [move_local(2, &fn_ty), "No"]})]}
                })],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    });
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            table_initializer(
                20,
                &["c", "mod", "TABLE"],
                10,
                0,
                vec![fn_def_to_fn_ptr(fn_def_const(1))],
            ),
            poison,
        ],
        vec![table_static(10, &["c", "mod", "TABLE"], 0, 20)],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert!(
        family_members(&rows, "mod::Table", "slot").is_empty(),
        "a parameter store unpublishes the family: {rows:?}"
    );
    assert!(
        !super::fnptr_field_family_contains(&llbc, "mod::Table", "slot"),
        "fnptr_struct_field_target must not name an unknown family"
    );
}

/// A diamond that assigns `f` / `g` to one local then Aggregates that
/// local publishes the union. Last-write-wins in physical block order
/// would keep only `g`.
#[test]
fn fnptr_field_family_unions_diamond_local_assigns() {
    let struct_ty = adt_ty(0);
    let fn_ty = fnptr_ty();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let diamond = serde_json::json!({
        "def_id": 20,
        "item_meta": item_meta_json(&["c", "mod", "make"], "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [bool_ty.clone()],
            "output": struct_ty.clone()
        },
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                local_ty(0, &struct_ty),
                local_ty(1, &bool_ty),
                local_ty(2, &fn_ty)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Switch": {
                        "discr": {"Read": typed_local(1, &bool_ty)},
                        "targets": {"If": [1, 2]}
                    }}}
                },
                {
                    "statements": [assign_local(2, &fn_ty, fn_def_to_fn_ptr(fn_def_const(1)))],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Goto": {"target": 3}}}
                },
                {
                    "statements": [assign_local(2, &fn_ty, fn_def_to_fn_ptr(fn_def_const(2)))],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Goto": {"target": 3}}}
                },
                {
                    "statements": [assign_local(
                        0,
                        &struct_ty,
                        serde_json::json!({"Aggregate": [{"Adt": [0, null]}, [
                            move_local(2, &fn_ty)
                        ]]})
                    )],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    });
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            diamond,
        ],
        vec![],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"],
        "diamond union of f and g: {rows:?}"
    );
}

fn table_param_constructor(def_id: u64, path: &[&str], type_id: u64) -> serde_json::Value {
    let struct_ty = adt_ty(type_id);
    let fn_ty = fnptr_ty();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [fn_ty.clone()],
            "output": struct_ty.clone()
        },
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                local_ty(0, &struct_ty),
                local_ty(1, &fn_ty)
            ]},
            "body": [{
                "statements": [assign_local(
                    0,
                    &struct_ty,
                    serde_json::json!({"Aggregate": [{"Adt": [type_id, null]}, [
                        move_local(1, &fn_ty)
                    ]]})
                )],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn call_ctor_with_fn(
    def_id: u64,
    path: &[&str],
    ctor_id: u64,
    stored_fun: u64,
    type_id: u64,
) -> serde_json::Value {
    let struct_ty = adt_ty(type_id);
    let fn_ty = fnptr_ty();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &fn_ty),
                local_ty(2, &struct_ty)
            ]},
            "body": [
                {
                    "statements": [assign_local(
                        1,
                        &fn_ty,
                        fn_def_to_fn_ptr(fn_def_const(stored_fun))
                    )],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": ctor_id},
                                "generics": empty_generics()
                            }},
                            "args": [move_local(1, &fn_ty)]
                        },
                        "dest": typed_local(2, &struct_ty),
                        "target": 1,
                        "on_unwind": 1
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    })
}

/// A constructor that stores its fn-pointer parameter closes the family
/// over FunDecls passed at Regular call sites (`FunctionDesc.pycall`
/// unions arguments over call sites). A Dynamic read+call through the
/// field follows both members (`FunctionReprBase.call` `c_graphs`).
#[test]
fn fnptr_field_family_unions_constructor_parameter_call_sites() {
    let table = adt_ty(0);
    let table_ptr = serde_json::json!({"RawPtr": [table.clone(), "Const"]});
    let fn_ty = fnptr_ty();
    let field_place = serde_json::json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [
                {"kind": {"Local": 1}, "ty": table_ptr.clone()},
                "Deref"
            ]}, "ty": table},
            {"Field": [null, 0]}
        ]},
        "ty": fn_ty.clone()
    });
    let harvested = dynamic_call_fun(
        30,
        &["c", "mod", "harvested_caller"],
        vec![assign_local(
            2,
            &fn_ty,
            serde_json::json!({"Use": [{"Read": field_place}, "No"]}),
        )],
        copy_local(2, &fn_ty),
        vec![
            local_ty(0, &unit_ty()),
            local_ty(1, &table_ptr),
            local_ty(2, &fn_ty),
        ],
        1,
        vec![table_ptr],
    );
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            table_param_constructor(21, &["c", "mod", "make"], 0),
            call_ctor_with_fn(22, &["c", "mod", "call_f"], 21, 1, 0),
            call_ctor_with_fn(23, &["c", "mod", "call_g"], 21, 2, 0),
            harvested,
        ],
        vec![],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"],
        "constructor parameter call sites union f and g: {rows:?}"
    );
    let cg = crate::memory::gctransform::framework::build(&llbc);
    let harvested_callees = cg.callees.get(&30).expect("harvested caller");
    assert!(
        harvested_callees.contains(&1) && harvested_callees.contains(&2),
        "read+call through the field follows both members: {harvested_callees:?}"
    );
    assert!(
        !cg.indirect.contains(&30),
        "harvested Dynamic is analyze_indirect_call, not top_result"
    );
}

fn opaque_fun(
    def_id: u64,
    path: &[&str],
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "src": "Normal",
        "body": "Opaque"
    })
}

/// Constructor stores live in one artefact and Regular call sites in
/// another (`FunctionDesc.pycall` unions arguments over the program).
#[test]
fn fnptr_field_family_unions_constructor_parameter_across_artefacts() {
    let ctor = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![table_param_constructor(21, &["c", "mod", "make"], 0)],
        vec![],
    );
    let callers = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            opaque_fun(21, &["c", "mod", "make"], vec![fnptr_ty()], adt_ty(0)),
            call_ctor_with_fn(22, &["c", "mod", "call_f"], 21, 1, 0),
            call_ctor_with_fn(23, &["c", "mod", "call_g"], 21, 2, 0),
        ],
        vec![],
    );
    let ctor_only = discover_fnptr_field_families(&ctor);
    assert!(
        family_members(&ctor_only, "mod::Table", "slot").is_empty(),
        "constructor artefact has no Regular call sites: {ctor_only:?}"
    );
    let mut inputs = collect_fnptr_family_inputs(&ctor);
    inputs.merge(collect_fnptr_family_inputs(&callers));
    let rows = resolve_fnptr_field_families(&inputs);
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"],
        "call sites in the other artefact union f and g: {rows:?}"
    );
    let published = publish_fnptr_field_families_across(&[&ctor, &callers]);
    assert_eq!(
        family_members(&published, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"]
    );
    assert_eq!(
        family_members(
            &discover_fnptr_field_families(&callers),
            "mod::Table",
            "slot"
        ),
        ["c::mod::f", "c::mod::g"],
        "seeded artefact sees the program-wide family"
    );
}

fn dynamic_call_ctor_with_fn(
    def_id: u64,
    path: &[&str],
    ctor_id: u64,
    stored_fun: u64,
    type_id: u64,
) -> serde_json::Value {
    let struct_ty = adt_ty(type_id);
    let fn_ty = fnptr_ty();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &fn_ty),
                local_ty(2, &fn_ty),
                local_ty(3, &struct_ty)
            ]},
            "body": [
                {
                    "statements": [
                        assign_local(1, &fn_ty, fn_def_to_fn_ptr(fn_def_const(ctor_id))),
                        assign_local(2, &fn_ty, fn_def_to_fn_ptr(fn_def_const(stored_fun)))
                    ],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Dynamic": [move_local(1, &fn_ty), {"regions": []}]},
                            "args": [move_local(2, &fn_ty)]
                        },
                        "dest": typed_local(3, &struct_ty),
                        "target": 1,
                        "on_unwind": 1
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    })
}

/// Direct `make(f)` plus a Dynamic call of `make` through a closed fn
/// pointer with `g` unions both members (`FunctionDesc.pycall` /
/// `Bookkeeper.consider_call_site` of that `SomePBC`).
#[test]
fn fnptr_field_family_unions_resolvable_dynamic_call_of_constructor() {
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            table_param_constructor(21, &["c", "mod", "make"], 0),
            call_ctor_with_fn(22, &["c", "mod", "call_f"], 21, 1, 0),
            dynamic_call_ctor_with_fn(23, &["c", "mod", "call_g_dyn"], 21, 2, 0),
        ],
        vec![],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"],
        "Regular make(f) and Dynamic make(g) union: {rows:?}"
    );
}

fn unresolvable_dynamic_taking_ctor_address(
    def_id: u64,
    path: &[&str],
    ctor_id: u64,
    stored_fun: u64,
    type_id: u64,
) -> serde_json::Value {
    let struct_ty = adt_ty(type_id);
    let fn_ty = fnptr_ty();
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [fn_ty.clone()],
            "output": unit_ty()
        },
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &fn_ty),
                local_ty(2, &fn_ty),
                local_ty(3, &fn_ty),
                local_ty(4, &struct_ty)
            ]},
            "body": [
                {
                    "statements": [
                        assign_local(2, &fn_ty, fn_def_to_fn_ptr(fn_def_const(ctor_id))),
                        assign_local(3, &fn_ty, fn_def_to_fn_ptr(fn_def_const(stored_fun)))
                    ],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Dynamic": [move_local(1, &fn_ty), {"regions": []}]},
                            "args": [move_local(3, &fn_ty)]
                        },
                        "dest": typed_local(4, &struct_ty),
                        "target": 1,
                        "on_unwind": 1
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    })
}

/// Direct `make(f)` plus an unresolvable Dynamic call, with `make`
/// used as a value, cannot close over `f` (`SomePBC` of every
/// address-taken FunDecl).
#[test]
fn fnptr_field_family_unknown_when_unresolvable_dynamic_can_call_constructor() {
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            table_param_constructor(21, &["c", "mod", "make"], 0),
            call_ctor_with_fn(22, &["c", "mod", "call_f"], 21, 1, 0),
            unresolvable_dynamic_taking_ctor_address(23, &["c", "mod", "poison"], 21, 2, 0),
        ],
        vec![],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert!(
        family_members(&rows, "mod::Table", "slot").is_empty(),
        "unresolvable Dynamic while make is a value unpublishes: {rows:?}"
    );
    assert!(
        !super::fnptr_field_family_contains(&llbc, "mod::Table", "slot"),
        "an incomplete input must not publish a smaller closed family"
    );
}

fn wrapper_reassign_then_ctor(
    def_id: u64,
    path: &[&str],
    ctor_id: u64,
    reassigned_fun: u64,
    type_id: u64,
) -> serde_json::Value {
    let struct_ty = adt_ty(type_id);
    let fn_ty = fnptr_ty();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [bool_ty.clone(), fn_ty.clone()],
            "output": unit_ty()
        },
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 2, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &bool_ty),
                local_ty(2, &fn_ty),
                local_ty(3, &struct_ty)
            ]},
            "body": [
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Switch": {
                        "discr": {"Read": typed_local(1, &bool_ty)},
                        "targets": {"If": [1, 2]}
                    }}}
                },
                {
                    "statements": [assign_local(
                        2,
                        &fn_ty,
                        fn_def_to_fn_ptr(fn_def_const(reassigned_fun))
                    )],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Goto": {"target": 3}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Goto": {"target": 3}}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": ctor_id},
                                "generics": empty_generics()
                            }},
                            "args": [move_local(2, &fn_ty)]
                        },
                        "dest": typed_local(3, &struct_ty),
                        "target": 4,
                        "on_unwind": 4
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    })
}

fn call_wrapper_with_fn(
    def_id: u64,
    path: &[&str],
    wrapper_id: u64,
    stored_fun: u64,
) -> serde_json::Value {
    let fn_ty = fnptr_ty();
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &fn_ty)
            ]},
            "body": [
                {
                    "statements": [assign_local(
                        1,
                        &fn_ty,
                        fn_def_to_fn_ptr(fn_def_const(stored_fun))
                    )],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": wrapper_id},
                                "generics": empty_generics()
                            }},
                            "args": [
                                {"Const": [{"Bool": true}, bool_ty]},
                                move_local(1, &fn_ty)
                            ]
                        },
                        "dest": typed_local(0, &unit_ty()),
                        "target": 1,
                        "on_unwind": 1
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    })
}

/// A wrapper that receives `f`, conditionally does `f = g`, then calls
/// `make(f)` unions the incoming argument with the reassignment
/// (`RPythonAnnotator.mergeinputargs`).
#[test]
fn fnptr_field_family_unions_parameter_with_conditional_reassignment() {
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            table_param_constructor(21, &["c", "mod", "make"], 0),
            wrapper_reassign_then_ctor(22, &["c", "mod", "wrapper"], 21, 2, 0),
            call_wrapper_with_fn(23, &["c", "mod", "call_wrapper"], 22, 1),
        ],
        vec![],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f", "c::mod::g"],
        "incoming f and reassigned g: {rows:?}"
    );
}

fn forward_param_to_callees(
    def_id: u64,
    path: &[&str],
    callee_ids: &[u64],
    callee_output: &serde_json::Value,
) -> serde_json::Value {
    let fn_ty = fnptr_ty();
    let mut locals = vec![local_ty(0, &unit_ty()), local_ty(1, &fn_ty)];
    for i in 0..callee_ids.len() as u64 {
        locals.push(local_ty(2 + i, callee_output));
    }
    let mut blocks = Vec::new();
    for (i, callee_id) in callee_ids.iter().enumerate() {
        let i = i as u64;
        blocks.push(serde_json::json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {
                        "kind": {"Fun": callee_id},
                        "generics": empty_generics()
                    }},
                    "args": [copy_local(1, &fn_ty)]
                },
                "dest": typed_local(2 + i, callee_output),
                "target": i + 1,
                "on_unwind": i + 1
            }}}
        }));
    }
    blocks.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": "Return"}
    }));
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {
            "is_unsafe": false,
            "inputs": [fn_ty],
            "output": unit_ty()
        },
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": locals},
            "body": blocks
        }}
    })
}

fn call_wrappers_with_fn(
    def_id: u64,
    path: &[&str],
    wrapper_ids: &[u64],
    stored_fun: u64,
) -> serde_json::Value {
    let fn_ty = fnptr_ty();
    let mut locals = vec![local_ty(0, &unit_ty()), local_ty(1, &fn_ty)];
    for i in 0..wrapper_ids.len() as u64 {
        locals.push(local_ty(2 + i, &unit_ty()));
    }
    let mut blocks = vec![serde_json::json!({
        "statements": [assign_local(
            1,
            &fn_ty,
            fn_def_to_fn_ptr(fn_def_const(stored_fun))
        )],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": {"Call": {
            "call": {
                "func": {"Regular": {
                    "kind": {"Fun": wrapper_ids[0]},
                    "generics": empty_generics()
                }},
                "args": [copy_local(1, &fn_ty)]
            },
            "dest": typed_local(2, &unit_ty()),
            "target": 1,
            "on_unwind": 1
        }}}
    })];
    for (i, wrapper_id) in wrapper_ids.iter().enumerate().skip(1) {
        let i = i as u64;
        blocks.push(serde_json::json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {
                        "kind": {"Fun": wrapper_id},
                        "generics": empty_generics()
                    }},
                    "args": [copy_local(1, &fn_ty)]
                },
                "dest": typed_local(2 + i, &unit_ty()),
                "target": i + 1,
                "on_unwind": i + 1
            }}}
        }));
    }
    blocks.push(serde_json::json!({
        "statements": [],
        "kind": "Regular",
        "terminator": {"span": span_json(), "kind": "Return"}
    }));
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": locals},
            "body": blocks
        }}
    })
}

/// A 25-layer diamond of wrappers (two callers per layer, 2^25 paths to
/// the constructor) closes over the one FunDecl constant at the top in
/// linear time (`RPythonAnnotator.complete` worklist, not a DFS of paths).
#[test]
fn fnptr_field_family_resolves_diamond_wrapper_chain_in_linear_time() {
    const LAYERS: usize = 25;
    let f_id = 1u64;
    let make_id = 2u64;
    let top_id = 3u64;
    let wrapper_id = |layer: usize, which: usize| 10 + (layer as u64) * 2 + (which as u64);
    let mut names = Vec::new();
    for layer in 0..LAYERS {
        for which in 0..2 {
            names.push(format!("w{layer:02}{}", if which == 0 { 'a' } else { 'b' }));
        }
    }
    let mut funs = vec![
        unit_fun(f_id, &["c", "mod", "f"]),
        table_param_constructor(make_id, &["c", "mod", "make"], 0),
    ];
    let table_ty = adt_ty(0);
    let unit = unit_ty();
    for layer in 0..LAYERS {
        let callees: Vec<u64> = if layer == 0 {
            vec![make_id]
        } else {
            vec![wrapper_id(layer - 1, 0), wrapper_id(layer - 1, 1)]
        };
        let output = if layer == 0 { &table_ty } else { &unit };
        for which in 0..2 {
            let name = &names[layer * 2 + which];
            funs.push(forward_param_to_callees(
                wrapper_id(layer, which),
                &["c", "mod", name],
                &callees,
                output,
            ));
        }
    }
    funs.push(call_wrappers_with_fn(
        top_id,
        &["c", "mod", "top"],
        &[wrapper_id(LAYERS - 1, 0), wrapper_id(LAYERS - 1, 1)],
        f_id,
    ));
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["slot"])],
        funs,
        vec![],
    );
    let started = std::time::Instant::now();
    let rows = discover_fnptr_field_families(&llbc);
    let elapsed = started.elapsed();
    assert_eq!(
        family_members(&rows, "mod::Table", "slot"),
        ["c::mod::f"],
        "diamond chain closes over the top FunDecl: {rows:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "diamond chain of {LAYERS} layers took {elapsed:?}"
    );
}

/// `a = phi(b, f)`, `b = phi(a, g)`: querying `a` then `b` from one
/// function still closes `b` over both FunDecls. Caching a cycle-cut
/// `ClosedCopy` on `b` while `a` is on the stack would keep only `g`.
#[test]
fn fnptr_field_family_unions_cyclic_local_phis_on_second_query() {
    let fn_ty = fnptr_ty();
    let table_a = adt_ty(0);
    let table_b = adt_ty(1);
    let cycle = serde_json::json!({
        "def_id": 23,
        "item_meta": item_meta_json(&["c", "mod", "cycle"], "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                local_ty(0, &unit_ty()),
                local_ty(1, &fn_ty),
                local_ty(2, &fn_ty),
                local_ty(3, &table_a),
                local_ty(4, &table_b)
            ]},
            "body": [
                {
                    "statements": [
                        assign_local(1, &fn_ty, fn_def_to_fn_ptr(fn_def_const(1))),
                        assign_local(2, &fn_ty, fn_def_to_fn_ptr(fn_def_const(2))),
                        assign_local(
                            1,
                            &fn_ty,
                            serde_json::json!({"Use": [copy_local(2, &fn_ty), "No"]})
                        ),
                        assign_local(
                            2,
                            &fn_ty,
                            serde_json::json!({"Use": [copy_local(1, &fn_ty), "No"]})
                        )
                    ],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 21},
                                "generics": empty_generics()
                            }},
                            "args": [copy_local(1, &fn_ty)]
                        },
                        "dest": typed_local(3, &table_a),
                        "target": 1,
                        "on_unwind": 1
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 22},
                                "generics": empty_generics()
                            }},
                            "args": [copy_local(2, &fn_ty)]
                        },
                        "dest": typed_local(4, &table_b),
                        "target": 2,
                        "on_unwind": 2
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    });
    let llbc = fnptr_family_llbc(
        vec![
            table_type(0, &["c", "mod", "TableA"], &["slot"]),
            table_type(1, &["c", "mod", "TableB"], &["slot"]),
        ],
        vec![
            unit_fun(1, &["c", "mod", "f"]),
            unit_fun(2, &["c", "mod", "g"]),
            table_param_constructor(21, &["c", "mod", "make_a"], 0),
            table_param_constructor(22, &["c", "mod", "make_b"], 1),
            cycle,
        ],
        vec![],
    );
    let rows = discover_fnptr_field_families(&llbc);
    assert_eq!(
        family_members(&rows, "mod::TableA", "slot"),
        ["c::mod::f", "c::mod::g"],
        "first query a unions f and g: {rows:?}"
    );
    assert_eq!(
        family_members(&rows, "mod::TableB", "slot"),
        ["c::mod::f", "c::mod::g"],
        "second query b must still include f: {rows:?}"
    );
}

fn dynamic_call_fun(
    def_id: u64,
    path: &[&str],
    statements: Vec<serde_json::Value>,
    dyn_operand: serde_json::Value,
    locals: Vec<serde_json::Value>,
    arg_count: u64,
    inputs: Vec<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": unit_ty()},
        "src": "Normal",
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": arg_count, "locals": locals},
            "body": [
                {
                    "statements": statements,
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Dynamic": [dyn_operand, {"regions": []}]},
                            "args": []
                        },
                        "dest": typed_local(0, &unit_ty()),
                        "target": 1,
                        "on_unwind": 1
                    }}}
                },
                {
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }
            ]
        }}
    })
}

/// A Dynamic call through a harvested FnPtr field is `analyze_indirect_call`
/// over the family members, not `graphs is None`.
#[test]
fn fnptr_field_dynamic_call_follows_family_members() {
    let table = adt_ty(0);
    let table_ptr = serde_json::json!({"RawPtr": [table.clone(), "Const"]});
    let fn_ty = fnptr_ty();
    let field_place = serde_json::json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [
                {"kind": {"Local": 1}, "ty": table_ptr.clone()},
                "Deref"
            ]}, "ty": table},
            {"Field": [null, 0]}
        ]},
        "ty": fn_ty.clone()
    });
    let harvested = dynamic_call_fun(
        30,
        &["c", "mod", "harvested_caller"],
        vec![assign_local(
            2,
            &fn_ty,
            serde_json::json!({"Use": [{"Read": field_place}, "No"]}),
        )],
        copy_local(2, &fn_ty),
        vec![
            local_ty(0, &unit_ty()),
            local_ty(1, &table_ptr),
            local_ty(2, &fn_ty),
        ],
        1,
        vec![table_ptr],
    );
    let residual = dynamic_call_fun(
        31,
        &["c", "mod", "residual_caller"],
        vec![],
        copy_local(1, &fn_ty),
        vec![local_ty(0, &unit_ty()), local_ty(1, &fn_ty)],
        1,
        vec![fn_ty],
    );
    let llbc = fnptr_family_llbc(
        vec![table_type(0, &["c", "mod", "Table"], &["left", "right"])],
        vec![
            unit_fun(1, &["c", "mod", "foo"]),
            unit_fun(2, &["c", "mod", "bar"]),
            table_initializer(
                20,
                &["c", "mod", "TABLE"],
                10,
                0,
                vec![
                    fn_def_to_fn_ptr(fn_def_const(1)),
                    fn_def_to_fn_ptr(fn_def_const(2)),
                ],
            ),
            harvested,
            residual,
        ],
        vec![table_static(10, &["c", "mod", "TABLE"], 0, 20)],
    );
    let cg = crate::memory::gctransform::framework::build(&llbc);
    let harvested_callees = cg.callees.get(&30).expect("harvested caller");
    assert!(
        harvested_callees.contains(&1),
        "left stores foo: {harvested_callees:?}"
    );
    assert!(
        !harvested_callees.contains(&2),
        "right is a different slot: {harvested_callees:?}"
    );
    assert!(
        !cg.indirect.contains(&30),
        "harvested Dynamic is analyze_indirect_call, not top_result"
    );
    assert!(
        cg.indirect.contains(&31),
        "a Dynamic call through a non-harvested pointer stays unresolved"
    );
    let residual_callees = cg.callees.get(&31).cloned().unwrap_or_default();
    assert!(
        residual_callees.is_empty(),
        "non-harvested Dynamic has no family edges: {residual_callees:?}"
    );
}

#[test]
#[ignore]
fn builtin_wrapper_declaration_has_builtin_code_fn_type() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = crate::test_support::load_ullbc(path);
    let wrapper = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path_str()
                .ends_with("::__majit_wrap_random")
        })
        .expect("_random::__majit_wrap_random");
    assert!(super::fun_decl_is_builtin_code_fn(wrapper, llbc));
    let non_member = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path_str()
                .ends_with("::function::funccall_valuestack")
        })
        .expect("function::funccall_valuestack");
    assert!(!super::fun_decl_is_builtin_code_fn(non_member, llbc));
}

/// rustc's promoted-const wrapper copies one `Place::Global` and
/// returns a reference. That inner id is the NamedConst `getfunctionptr`
/// would name (`Transformer.handle_residual_call`).
#[test]
fn promoted_const_initializer_copies_the_inner_global() {
    let body: majit_charon_reader::ullbc::Unstructured = serde_json::from_value(serde_json::json!({
            "span": {"Deduplicated": 0},
            "locals": {"arg_count": 0, "locals": []},
            "body": [{
                "statements": [
                    {"span": {"Deduplicated": 0}, "kind": {"StorageLive": 1}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                        {"Use": [{"Read": {
                            "kind": {"Global": {
                                "id": 31,
                                "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
                            }},
                            "ty": {"Deduplicated": 0}
                        }}, "Yes"]}
                    ]}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
                        {"Ref": {
                            "place": {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                            "kind": "Shared",
                            "ptr_metadata": {"Const": {"Deduplicated": 0}}
                        }}
                    ]}}
                ],
                "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
                "kind": "Regular"
            }]
        }))
        .expect("promoted-const wrapper fixture parses");
    let place = super::promoted_const_copied_global(&body).expect("inner global");
    assert_eq!(super::inner_global_id(&place), 31);
}

/// rustc promotes a `&str` literal to `_1 = const C; _0 = &_1`.
#[test]
fn promoted_const_initializer_copies_the_inner_const() {
    let body: majit_charon_reader::ullbc::Unstructured =
        serde_json::from_value(serde_json::json!({
            "span": {"Deduplicated": 0},
            "locals": {"arg_count": 0, "locals": []},
            "body": [{
                "statements": [
                    {"span": {"Deduplicated": 0}, "kind": {"StorageLive": 1}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                        {"Use": [{"Const": {"Deduplicated": 20437}}, "Yes"]}
                    ]}},
                    {"span": {"Deduplicated": 0}, "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
                        {"Ref": {
                            "place": {"kind": {"Local": 1}, "ty": {"Deduplicated": 0}},
                            "kind": "Shared",
                            "ptr_metadata": {"Const": {"Deduplicated": 0}}
                        }}
                    ]}}
                ],
                "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
                "kind": "Regular"
            }]
        }))
        .expect("promoted-const literal wrapper fixture parses");
    let value = super::promoted_const_copied_const(&body).expect("inner const");
    assert_eq!(
        value.get("Deduplicated").and_then(|v| v.as_u64()),
        Some(20437)
    );
}

fn promoted_const_unstructured(
    blocks: serde_json::Value,
) -> majit_charon_reader::ullbc::Unstructured {
    serde_json::from_value(serde_json::json!({
        "span": {"Deduplicated": 0},
        "locals": {"arg_count": 0, "locals": []},
        "body": blocks
    }))
    .expect("promoted-const fixture parses")
}

fn global_place(id: u64) -> serde_json::Value {
    serde_json::json!({
        "kind": {"Global": {
            "id": id,
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        }},
        "ty": {"Deduplicated": 0}
    })
}

fn local_place(id: u64) -> serde_json::Value {
    serde_json::json!({"kind": {"Local": id}, "ty": {"Deduplicated": 0}})
}

fn assign_copy(dest: serde_json::Value, src: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "span": {"Deduplicated": 0},
        "kind": {"Assign": [dest, {"Use": [{"Read": src}, "Yes"]}]}
    })
}

fn assign_shared_ref(dest: serde_json::Value, src: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "span": {"Deduplicated": 0},
        "kind": {"Assign": [dest, {"Ref": {
            "place": src,
            "kind": "Shared",
            "ptr_metadata": {"Const": {"Deduplicated": 0}}
        }}]}
    })
}

fn return_bb(statements: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "statements": statements,
        "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
        "kind": "Regular"
    })
}

/// Shape (b): `_0 = copy Global(id); return`.
#[test]
fn promoted_const_initializer_copies_the_inner_global_by_value() {
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![assign_copy(
        local_place(0),
        global_place(31)
    ),])]));
    let place = super::promoted_const_copied_global(&body).expect("inner global");
    assert_eq!(super::inner_global_id(&place), 31);
}

/// `_1 = copy G; _0 = &(_1.field); return` is not a read of G.
#[test]
fn promoted_const_initializer_rejects_field_projection_ref() {
    let field = serde_json::json!({
        "kind": {"Projection": [local_place(1), {"Field": [null, 0]}]},
        "ty": {"Deduplicated": 0}
    });
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![
        assign_copy(local_place(1), global_place(31)),
        assign_shared_ref(local_place(0), field),
    ])]));
    assert!(super::promoted_const_copied_global(&body).is_none());
}

/// rustc's promoted-const wrapper does not emit a local-to-local copy
/// chain (`_2 = copy _1; _0 = &_2`). Reject it so a copy of a copy is
/// not folded as the global.
#[test]
fn promoted_const_initializer_rejects_local_to_local_copy_chain() {
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![
        assign_copy(local_place(1), global_place(31)),
        assign_copy(local_place(2), local_place(1)),
        assign_shared_ref(local_place(0), local_place(2)),
    ])]));
    assert!(super::promoted_const_copied_global(&body).is_none());
}

/// A global copied only in an unreachable block is not the returned value.
#[test]
fn promoted_const_initializer_rejects_unreachable_block_global() {
    let unreachable = return_bb(vec![
        assign_copy(local_place(1), global_place(31)),
        assign_shared_ref(local_place(0), local_place(1)),
    ]);
    let entry = serde_json::json!({
        "statements": [],
        "terminator": {"span": {"Deduplicated": 0}, "kind": "Return"},
        "kind": "Regular"
    });
    let body = promoted_const_unstructured(serde_json::json!([entry, unreachable]));
    assert!(super::promoted_const_copied_global(&body).is_none());
    assert!(super::promoted_const_copied_const(&body).is_none());
}

/// `resolve_place` of a projected global falls back to the inner Global
/// and drops the field, so a copy of `G.field` is not accepted as G.
#[test]
fn promoted_const_initializer_rejects_projected_global() {
    let projected = serde_json::json!({
        "kind": {"Projection": [global_place(31), {"Field": [null, 0]}]},
        "ty": {"Deduplicated": 0}
    });
    let body = promoted_const_unstructured(serde_json::json!([return_bb(vec![
        assign_copy(local_place(1), projected),
        assign_shared_ref(local_place(0), local_place(1)),
    ])]));
    assert!(super::promoted_const_copied_global(&body).is_none());
}

/// A Charon-instantiated FunDecl's function-item path uses the same
/// specialized leaf `registered_path_for_fun_decl` registers
/// (`FunctionDesc.cachedgraph`).
#[test]
fn fundecl_fn_item_segments_uses_the_registered_instance_leaf() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let ident = |s: &str| serde_json::json!({"Ident": [s, 0]});
    let args = serde_json::json!({
        "regions": [],
        "types": [{"Scalar": {"Integer": {"Signed": "I64"}}}],
        "const_generics": [],
        "trait_refs": []
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [
                        ident("fixture"),
                        ident("f"),
                        {"Instantiated": {
                            "params": {},
                            "kind": "Other",
                            "skip_binder": args
                        }}
                    ],
                    "span": span.clone(),
                    "source_text": null,
                    "attr_info": {
                        "attributes": [],
                        "inline": null,
                        "rename": null,
                        "public": true
                    },
                    "is_local": true
                },
                "signature": {
                    "is_unsafe": false,
                    "inputs": [],
                    "output": {"Scalar": "Bool"}
                },
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {"arg_count": 0, "locals": []},
                    "body": []
                }}
            }],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("instance fixture parses");
    let fd = llbc.fn_by_id(0).expect("f");
    let registered = super::registered_path_for_fun_decl(&llbc, fd);
    let item = super::fundecl_fn_item_segments(&llbc, fd);
    let harvest = super::spec_segments(&llbc, fd, &super::registered_leaf(&llbc, fd));
    assert_eq!(item, harvest);
    let leaf = item.last().expect("leaf");
    assert!(leaf.starts_with("f__spec_"), "{leaf}");
    assert_ne!(leaf.as_str(), "f");
    assert_eq!(
        registered.segments.last().map(String::as_str),
        Some(leaf.as_str())
    );
}

/// A call that names a specialized copy stamps identity with that copy's
/// leaf and `cachedgraph` key, not the base FunDecl leaf.
#[test]
fn identity_path_for_fun_decl_uses_the_call_leaf_and_specialization() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let ident = |s: &str| serde_json::json!({"Ident": [s, 0]});
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [ident("fixture"), ident("load_attr")],
                    "span": span.clone(),
                    "source_text": null,
                    "attr_info": {
                        "attributes": [],
                        "inline": null,
                        "rename": null,
                        "public": true
                    },
                    "is_local": true
                },
                "signature": {
                    "is_unsafe": false,
                    "inputs": [],
                    "output": {"Scalar": "Bool"}
                },
                "body": {"Unstructured": {
                    "span": span,
                    "locals": {"arg_count": 0, "locals": []},
                    "body": []
                }}
            }],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("load_attr fixture parses");
    let fd = llbc.fn_by_id(0).expect("load_attr");
    let base = super::registered_path_for_fun_decl(&llbc, fd);
    assert_eq!(base.segments.last().map(String::as_str), Some("load_attr"));
    assert_eq!(
        base.specialization,
        crate::parse::GraphSpecialization::Regular
    );
    let spec_leaf = "load_attr__spec_PyFrame_ab";
    let spec = super::identity_path_for_fun_decl(
        &llbc,
        fd,
        spec_leaf,
        crate::parse::GraphSpecialization::AccessDirect,
    );
    assert_eq!(spec.segments.last().map(String::as_str), Some(spec_leaf));
    assert_eq!(
        spec.specialization,
        crate::parse::GraphSpecialization::AccessDirect
    );
    assert_ne!(spec.segments, base.segments);
}

/// One Charon `NameSeg` in an `inherent_adt_impl_make_llbc` family fixture.
#[derive(Clone, Copy)]
enum FixtureSeg<'a> {
    Ident(&'a str),
    /// Inherent `impl Self` whose TypeDecl is `id`.
    Inherent(u64),
    /// `impl Trait for Self` (`Trait` is the `trait_impls` row).
    TraitImpl {
        self_ty: u64,
        trait_impl: u64,
    },
}

/// JSON LLBC: inherent `fn make` methods whose Self is an ADT TypeDecl.
/// `types` is `(def_id, ident path)`; `methods` is
/// `(fun def_id, impl-module idents, self TypeDecl id)`.
fn inherent_adt_impl_make_llbc(types: &[(u64, &[&str])], methods: &[(u64, &[&str], u64)]) -> Llbc {
    let types: Vec<(u64, Vec<FixtureSeg<'_>>)> = types
        .iter()
        .map(|(id, path)| (*id, path.iter().copied().map(FixtureSeg::Ident).collect()))
        .collect();
    let funs: Vec<(u64, Vec<FixtureSeg<'_>>)> = methods
        .iter()
        .map(|(def_id, module, self_id)| {
            let mut segs: Vec<FixtureSeg<'_>> =
                module.iter().copied().map(FixtureSeg::Ident).collect();
            segs.push(FixtureSeg::Inherent(*self_id));
            segs.push(FixtureSeg::Ident("make"));
            (*def_id, segs)
        })
        .collect();
    fixture_adt_llbc(&types, &funs, &[], &[])
}

/// Sparse `type_decls` / `fun_decls` / `trait_decls` / `trait_impls`
/// indexed by `def_id`. `traits` is `(id, ident path)`; `trait_impls`
/// is `(impl_id, trait_id, self TypeDecl id)`.
fn fixture_adt_llbc(
    types: &[(u64, Vec<FixtureSeg<'_>>)],
    funs: &[(u64, Vec<FixtureSeg<'_>>)],
    traits: &[(u64, &[&str])],
    trait_impls: &[(u64, u64, u64)],
) -> Llbc {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let ident = |s: &str| serde_json::json!({"Ident": [s, 0]});
    let meta = |name: Vec<serde_json::Value>| {
        serde_json::json!({
            "name": name,
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let adt = |id: u64| {
        serde_json::json!({
            "Adt": {"id": id, "generics": generics.clone(), "builtin": null}
        })
    };
    let ty = |id: u64| serde_json::json!({"Value": [id, adt(id)]});
    let seg_json = |seg: FixtureSeg<'_>| match seg {
        FixtureSeg::Ident(s) => ident(s),
        FixtureSeg::Inherent(id) => serde_json::json!({"Impl": {"Ty": {
            "skip_binder": ty(id),
            "kind": "InherentImplBlock"
        }}}),
        FixtureSeg::TraitImpl {
            self_ty,
            trait_impl,
        } => serde_json::json!({"Impl": {
            "Ty": {"skip_binder": ty(self_ty), "kind": "TraitImpl"},
            "Trait": trait_impl
        }}),
    };
    let fill = |rows: Vec<(u64, serde_json::Value)>| {
        let n = rows
            .iter()
            .map(|(id, _)| *id)
            .max()
            .map(|id| id + 1)
            .unwrap_or(0);
        let mut out = vec![serde_json::Value::Null; n as usize];
        for (id, value) in rows {
            out[id as usize] = value;
        }
        out
    };
    let type_decls = fill(
        types
            .iter()
            .map(|(id, path)| {
                (
                    *id,
                    serde_json::json!({
                        "def_id": id,
                        "item_meta": meta(path.iter().copied().map(seg_json).collect()),
                        "kind": {"Struct": []}
                    }),
                )
            })
            .collect(),
    );
    let fun_decls = fill(
        funs.iter()
            .map(|(def_id, path)| {
                (
                    *def_id,
                    serde_json::json!({
                        "def_id": def_id,
                        "item_meta": meta(path.iter().copied().map(seg_json).collect()),
                        "signature": {
                            "is_unsafe": false,
                            "inputs": [],
                            "output": {"Scalar": "Bool"}
                        },
                        "body": {"Unstructured": {
                            "span": span.clone(),
                            "locals": {"arg_count": 0, "locals": []},
                            "body": []
                        }}
                    }),
                )
            })
            .collect(),
    );
    let trait_decls = fill(
        traits
            .iter()
            .map(|(id, path)| {
                (
                    *id,
                    serde_json::json!({
                        "def_id": id,
                        "item_meta": meta(path.iter().copied().map(ident).collect()),
                        "methods": []
                    }),
                )
            })
            .collect(),
    );
    let trait_impl_decls = fill(
        trait_impls
            .iter()
            .map(|(impl_id, trait_id, self_ty)| {
                (
                    *impl_id,
                    serde_json::json!({
                        "impl_trait": {
                            "id": trait_id,
                            "generics": {"types": [adt(*self_ty)], "const_generics": []}
                        },
                        "methods": []
                    }),
                )
            })
            .collect(),
    );
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "c",
            "type_decls": type_decls,
            "fun_decls": fun_decls,
            "global_decls": [],
            "trait_decls": trait_decls,
            "trait_impls": trait_impl_decls
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("inherent ADT impl fixture parses")
}

fn identity_segs(llbc: &Llbc, id: u64) -> Vec<String> {
    super::registered_path_for_fun_decl(llbc, llbc.fn_by_id(id).expect("fun")).segments
}

/// Two inherent impls in one module whose Self types are `a::T` and
/// `b::T` (non-generic). Harvest `for_impl_method` already names the
/// TypeDecl (`a::T::make` / `b::T::make`). The FunDecl identity is the
/// TypeDecl path plus the method (`c::a::T::make` / `c::b::T::make`);
/// last-Ident-only Self (`T`) would collapse both onto `c::m::T::make`
/// (`Bookkeeper.getdesc` keys the function object).
///
/// `fnptr_family_llbc` / `unit_fun` mint Ident-only `item_meta` paths
/// and cannot carry an Impl payload; this fixture is the same JSON
/// shape as `lltype/tests.rs`
/// `spec_segments_distinguishes_impls_that_share_a_self_leaf`.
#[test]
fn registered_path_for_fun_decl_distinguishes_inherent_impls_that_share_a_self_leaf() {
    let llbc = inherent_adt_impl_make_llbc(
        &[(0, &["c", "a", "T"]), (1, &["c", "b", "T"])],
        &[(1, &["c", "m"], 0), (2, &["c", "m"], 1)],
    );
    let a = llbc.fn_by_id(1).expect("impl a::T");
    let b = llbc.fn_by_id(2).expect("impl b::T");
    let a_harvest = super::harvest_path_for_fun_decl(&llbc, a).segments;
    let b_harvest = super::harvest_path_for_fun_decl(&llbc, b).segments;
    assert_eq!(
        a_harvest,
        crate::parse::CallPath::for_impl_method("a::T", "make").segments
    );
    assert_eq!(
        b_harvest,
        crate::parse::CallPath::for_impl_method("b::T", "make").segments
    );
    assert_ne!(a_harvest, b_harvest, "harvest a::T vs b::T collapsed");
    let a_id = super::registered_path_for_fun_decl(&llbc, a).segments;
    let b_id = super::registered_path_for_fun_decl(&llbc, b).segments;
    assert_ne!(
        a_id, b_id,
        "identity a::T vs b::T collapsed: {a_id:?} {b_id:?}"
    );
    assert_eq!(
        a_id,
        ["c", "a", "T", "make"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        b_id,
        ["c", "b", "T", "make"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    );
}

/// Suffix-overlap of the impl module with Self's TypeDecl path is not
/// an identity: `impl` in `c::x::m` for `c::m::T` and `impl` in `c::x`
/// for `c::x::m::T` both became `c::x::m::T::make` under that rule.
#[test]
fn registered_path_for_fun_decl_distinguishes_inherent_impls_whose_module_overlaps_self() {
    let llbc = inherent_adt_impl_make_llbc(
        &[(0, &["c", "m", "T"]), (1, &["c", "x", "m", "T"])],
        &[(1, &["c", "x", "m"], 0), (2, &["c", "x"], 1)],
    );
    let in_xm = identity_segs(&llbc, 1);
    let in_x = identity_segs(&llbc, 2);
    assert_ne!(
        in_xm, in_x,
        "identity c::m::T vs c::x::m::T collapsed: {in_xm:?} {in_x:?}"
    );
    assert_eq!(
        in_xm,
        ["c", "m", "T", "make"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        in_x,
        ["c", "x", "m", "T", "make"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    );
}

/// Same-module `impl T`: identity is the TypeDecl path plus the method,
/// which matches the pre-250 spelling `c::m::T::make`.
#[test]
fn registered_path_for_fun_decl_same_module_inherent_impl_is_type_decl_path() {
    let llbc = inherent_adt_impl_make_llbc(&[(0, &["c", "m", "T"])], &[(1, &["c", "m"], 0)]);
    assert_eq!(
        identity_segs(&llbc, 1),
        ["c", "m", "T", "make"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    );
}

/// `impl A { fn build() { struct T; impl T { fn make() {} } } }` vs
/// `impl B { fn build() { struct T; impl T { fn make() {} } } }`.
/// TypeDecl paths carry the enclosing inherent Impl; Ident-only Self
/// (`c::build::T::make`) collides (`Bookkeeper.getdesc` keys the
/// function object).
#[test]
fn registered_path_for_fun_decl_distinguishes_nested_type_decls_in_inherent_impls() {
    let llbc = fixture_adt_llbc(
        &[
            (0, vec![FixtureSeg::Ident("c"), FixtureSeg::Ident("A")]),
            (1, vec![FixtureSeg::Ident("c"), FixtureSeg::Ident("B")]),
            (
                2,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::Inherent(0),
                    FixtureSeg::Ident("build"),
                    FixtureSeg::Ident("T"),
                ],
            ),
            (
                3,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::Inherent(1),
                    FixtureSeg::Ident("build"),
                    FixtureSeg::Ident("T"),
                ],
            ),
        ],
        &[
            (
                1,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::Inherent(2),
                    FixtureSeg::Ident("make"),
                ],
            ),
            (
                2,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::Inherent(3),
                    FixtureSeg::Ident("make"),
                ],
            ),
        ],
        &[],
        &[],
    );
    let in_a = identity_segs(&llbc, 1);
    let in_b = identity_segs(&llbc, 2);
    assert_ne!(
        in_a, in_b,
        "nested TypeDecl T in impl A vs impl B collapsed: {in_a:?} {in_b:?}"
    );
}

/// Same nested `struct T; impl T { fn make }`, but the enclosing scope
/// is `impl X for A { fn build() }` vs `impl Y for A { fn build() }`
/// (same Self, different traits).
#[test]
fn registered_path_for_fun_decl_distinguishes_nested_type_decls_in_trait_impls() {
    let llbc = fixture_adt_llbc(
        &[
            (0, vec![FixtureSeg::Ident("c"), FixtureSeg::Ident("A")]),
            (
                1,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::TraitImpl {
                        self_ty: 0,
                        trait_impl: 0,
                    },
                    FixtureSeg::Ident("build"),
                    FixtureSeg::Ident("T"),
                ],
            ),
            (
                2,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::TraitImpl {
                        self_ty: 0,
                        trait_impl: 1,
                    },
                    FixtureSeg::Ident("build"),
                    FixtureSeg::Ident("T"),
                ],
            ),
        ],
        &[
            (
                1,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::Inherent(1),
                    FixtureSeg::Ident("make"),
                ],
            ),
            (
                2,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::Inherent(2),
                    FixtureSeg::Ident("make"),
                ],
            ),
        ],
        &[(0, &["c", "X"]), (1, &["c", "Y"])],
        &[(0, 0, 0), (1, 1, 0)],
    );
    let in_x = identity_segs(&llbc, 1);
    let in_y = identity_segs(&llbc, 2);
    assert_ne!(
        in_x, in_y,
        "nested TypeDecl T in impl X for A vs impl Y for A collapsed: {in_x:?} {in_y:?}"
    );
}

/// FunDecl nested in a trait-impl method:
/// `impl X for A { fn build() { fn helper() {} } }` vs
/// `impl Y for A { fn build() { fn helper() {} } }`. The last Impl is
/// the trait impl; `scope_owner_impl_seg` / last-Impl `render_trait_impl`
/// must keep X vs Y.
#[test]
fn registered_path_for_fun_decl_distinguishes_helpers_nested_in_trait_impl_methods() {
    let llbc = fixture_adt_llbc(
        &[(0, vec![FixtureSeg::Ident("c"), FixtureSeg::Ident("A")])],
        &[
            (
                1,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::TraitImpl {
                        self_ty: 0,
                        trait_impl: 0,
                    },
                    FixtureSeg::Ident("build"),
                    FixtureSeg::Ident("helper"),
                ],
            ),
            (
                2,
                vec![
                    FixtureSeg::Ident("c"),
                    FixtureSeg::TraitImpl {
                        self_ty: 0,
                        trait_impl: 1,
                    },
                    FixtureSeg::Ident("build"),
                    FixtureSeg::Ident("helper"),
                ],
            ),
        ],
        &[(0, &["c", "X"]), (1, &["c", "Y"])],
        &[(0, 0, 0), (1, 1, 0)],
    );
    let in_x = identity_segs(&llbc, 1);
    let in_y = identity_segs(&llbc, 2);
    assert_ne!(
        in_x, in_y,
        "helper nested in impl X for A vs impl Y for A collapsed: {in_x:?} {in_y:?}"
    );
}

/// A monomorphized vtable keeps its method slots as `*const ()`, so the
/// `dyn` call casts the slot into a temporary and calls that; the slot
/// read is recovered from the temporary's single producer.
#[test]
fn a_mono_vtable_call_reads_the_slot_under_its_fn_ptr_cast() {
    let ty = ptr_ty();
    let slot = serde_json::json!({"Read": {"kind": {"Projection": [
            {"kind": {"Projection": [{"kind": {"Local": 1}, "ty": ty}, "Deref"]}, "ty": ty},
            {"Field": [null, 4]}
        ]}, "ty": ty}});
    let assign = |dst: u64, rvalue: serde_json::Value| {
        serde_json::json!({"span": span_json(), "kind": {"Assign": [
            {"kind": {"Local": dst}, "ty": ty},
            rvalue
        ]}})
    };
    let local = |index: u64| serde_json::json!({"index": index, "name": null, "span": span_json(), "ty": ty});
    let fun = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "dyn_call"], "", true),
        "signature": {"is_unsafe": true, "inputs": [ty], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 1, "locals": [local(0), local(1), local(2), local(3)]},
            "body": [{
                "statements": [
                    assign(2, serde_json::json!({"UnaryOp": [
                        {"Cast": {"RawPtr": [ty, ty]}},
                        slot
                    ]})),
                    assign(3, serde_json::json!({"Use": [slot, "No"]}))
                ],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    });
    let llbc = const_artifact("fixture", serde_json::json!([fun]), serde_json::json!([]));
    let fd = llbc.iter_local_fns().next().expect("fixture fn");
    let body = fd.unstructured().expect("fixture body");
    let operand = |index: u64| {
        serde_json::from_value::<majit_charon_reader::ullbc::Operand>(serde_json::json!(
            {"Read": {"kind": {"Local": index}, "ty": ty}}
        ))
        .unwrap()
    };
    let peeled = super::vtable_slot_under_fn_ptr_cast(&llbc, &body, &operand(2))
        .expect("the cast temporary names its vtable slot");
    let expected =
        serde_json::from_value::<majit_charon_reader::ullbc::Operand>(slot.clone()).unwrap();
    assert_eq!(format!("{peeled:?}"), format!("{expected:?}"));
    assert!(
        super::vtable_slot_under_fn_ptr_cast(&llbc, &body, &operand(3)).is_none(),
        "a plain copy of the slot is not a cast temporary"
    );
}

/// The `Vec` index fold must accept a `usize` index.
///
/// `usize` types as `Unsigned`, not `Int`, so gating on `Int` alone left
/// both the read and write lifts dead for every real callsite — caught
/// only by the `#[ignore]`d real-LLBC anchor
/// (`vec_index_mut_fill_user_function_args_real`), which does not run in
/// CI.  This pins the acceptance set without the 440MB corpus.
#[test]
fn vec_index_gate_accepts_both_integer_banks_and_rejects_range() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let ty = |v: serde_json::Value| {
        serde_json::from_value::<super::TyRef>(serde_json::json!({
            "Value": [0, v]
        }))
        .expect("fixture TyRef parses")
    };
    let usize_ty = ty(serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}}));
    assert_eq!(
        super::tyref_to_value_type(&usize_ty, &llbc),
        crate::model::ValueType::Unsigned,
        "usize must type as Unsigned — the trap this gate fell into"
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert!(
        super::vec_index_type_is_scalar(
            &usize_ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new(),
            None,
        ),
        "Vec index fold must accept a usize index"
    );
    assert!(
        super::vec_index_type_is_scalar(
            &ty(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})),
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new(),
            None,
        ),
        "Vec index fold must accept a signed index"
    );
    assert!(
        !super::vec_index_type_is_scalar(
            &ty(serde_json::json!({ "Adt": { "id": 7  } })),
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new(),
            None,
        ),
        "a Range index types as Ref and must stay rejected"
    );
}

#[test]
fn add_dest_single_deref_guard_classifies_uses() {
    use majit_charon_reader::ullbc::Unstructured;
    // brick 3's escape guard: the `.add`-result local `_1` may be
    // dereferenced once and never escape as a raw pointer.  Each
    // fixture defines `_1` with one bare-local write (standing in for
    // the `add` call, which the guard counts the same as any other
    // bare-local definition) and varies how `_1` is then used.
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let deref_place = |i: u64| {
        serde_json::json!({
            "kind": {"Projection": [place_local(i), "Deref"]}, "ty": ty()
        })
    };
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    // A single-block body: `_1 = const` (the def) followed by `extra`,
    // returning.  `dest` = `_1`.
    let body_of = |extra: Vec<serde_json::Value>| -> Unstructured {
        let mut statements = vec![stmt(
            serde_json::json!({"Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]}),
        )];
        statements.extend(extra);
        let bb = serde_json::json!({
            "statements": statements,
            "terminator": {"kind": "Return"}
        });
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [bb]
        });
        serde_json::from_value(body_json).expect("fixture Unstructured parses")
    };

    // `_2 = *_1` — one deref read: accepted.
    let read = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]
    }))]);
    assert!(super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &read,
        1
    ));

    // `*_1 = _2` — one deref write: accepted.
    let write = body_of(vec![stmt(serde_json::json!({
        "Assign": [deref_place(1), {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &write,
        1
    ));

    // `_2 = _1` — the pointer escapes by value: rejected.
    let escape = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": place_local(1)}, "No"]}]
    }))]);
    assert!(!super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &escape,
        1
    ));

    // `_2 = *_1; _3 = *_1` — two derefs: rejected.
    let twice = body_of(vec![
        stmt(
            serde_json::json!({"Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]}),
        ),
        stmt(
            serde_json::json!({"Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]}),
        ),
    ]);
    assert!(!super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &twice,
        1
    ));

    // No use at all (just the def): rejected (no element load/store).
    let dead = body_of(vec![]);
    assert!(!super::add_dest_used_only_as_single_deref(
        &fixture_llbc(),
        &dead,
        1
    ));
}

#[test]
fn format_str_locals_follows_must_use_and_ref() {
    let llbc = format_as_bytes_window_llbc();
    let body = llbc
        .fn_by_id(0)
        .and_then(|fd| fd.unstructured())
        .expect("subject unstructured");
    let marks = super::format_str_locals(&body, &llbc);
    assert_eq!(
        marks.get(2).copied(),
        Some(true),
        "format dest is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(3).copied(),
        Some(true),
        "must_use dest is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(4).copied(),
        Some(true),
        "shared-ref temp is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(1).copied(),
        Some(false),
        "Arguments parameter is not format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(5).copied(),
        Some(false),
        "as_bytes dest is not format_str: {marks:?}"
    );
}

fn format_mut_borrowed_format_string_llbc() -> majit_charon_reader::Llbc {
    let string = string_adt();
    let args_ty = fmt_arguments_adt();
    let unit = unit_ty();
    let shared = ref_string("Shared");
    let mut_ref = ref_string("Mut");
    let window = window_shared_slice();
    let slice = window_slice();
    let name = "format_mut_borrowed_format_string";
    llbc_with_types(
        "fixture",
        vec![string_decl(), fmt_arguments_decl()],
        vec![
            subject_decl(
                name,
                1,
                vec![args_ty.clone()],
                unit.clone(),
                vec![
                    local_of(0, &unit),
                    local_of(1, &args_ty),
                    local_of(2, &string),
                    local_of(3, &string),
                    local_of(4, &shared),
                    local_of(5, &window),
                    local_of(6, &window),
                    local_of(7, &mut_ref),
                ],
                vec![
                    bb(
                        vec![],
                        call_term(
                            1,
                            vec![move_op(place_of(1, &args_ty))],
                            place_of(2, &string),
                            1,
                            5,
                        ),
                    ),
                    bb(
                        vec![],
                        call_term(
                            2,
                            vec![move_op(place_of(2, &string))],
                            place_of(3, &string),
                            2,
                            5,
                        ),
                    ),
                    bb(
                        vec![
                            assign_stmt(
                                place_of(7, &mut_ref),
                                ref_rvalue(place_of(3, &string), "Mut"),
                            ),
                            assign_stmt(
                                place_of(4, &shared),
                                ref_rvalue(place_of(3, &string), "Shared"),
                            ),
                        ],
                        call_term(
                            3,
                            vec![move_op(place_of(4, &shared))],
                            place_of(5, &window),
                            3,
                            5,
                        ),
                    ),
                    bb(
                        vec![assign_stmt(
                            place_of(6, &window),
                            ref_rvalue(deref_place(5, &window, &slice), "Shared"),
                        )],
                        call_term(
                            4,
                            vec![move_op(place_of(6, &window))],
                            place_of(0, &unit),
                            4,
                            5,
                        ),
                    ),
                    bb(vec![], json!("Return")),
                    bb(vec![], json!("UnwindResume")),
                ],
            ),
            fmt_format_decl(1),
            hint_must_use_decl(2),
            string_as_bytes_decl(),
            opaque_decl(4, &["fixture", "g"], vec![window], unit),
        ],
    )
}

#[test]
fn format_str_locals_declines_a_mut_borrowed_format_string() {
    let llbc = format_mut_borrowed_format_string_llbc();
    let body = llbc
        .fn_by_id(0)
        .and_then(|fd| fd.unstructured())
        .expect("subject unstructured");
    let marks = super::format_str_locals(&body, &llbc);
    assert_eq!(
        marks.get(2).copied(),
        Some(true),
        "format dest is format_str: {marks:?}"
    );
    assert_eq!(
        marks.get(3).copied(),
        Some(false),
        "must_use dest is declined by &mut: {marks:?}"
    );
    assert_eq!(
        marks.get(4).copied(),
        Some(false),
        "shared-ref of a declined local is not format_str: {marks:?}"
    );
}

#[test]
fn add_dest_struct_field_guard_classifies_uses() {
    use majit_charon_reader::ullbc::Unstructured;
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let deref_place = |i: u64| {
        serde_json::json!({
            "kind": {"Projection": [place_local(i), "Deref"]},
            "ty": ty()
        })
    };
    let field_place = |i: u64, idx: u64| {
        serde_json::json!({
            "kind": {"Projection": [deref_place(i), {"Field": [null, idx]}]},
            "ty": ty()
        })
    };
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let body_of = |extra: Vec<serde_json::Value>| -> Unstructured {
        let mut statements = vec![stmt(serde_json::json!({
            "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
        }))];
        statements.extend(extra);
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [{
                "statements": statements,
                "terminator": {"kind": "Return"}
            }]
        });
        serde_json::from_value(body_json).expect("fixture Unstructured parses")
    };
    let llbc = fixture_llbc();
    let accepts = |body: &Unstructured| super::add_dest_used_only_as_struct_field(&llbc, body, 1);
    let elem_accepts = |body: &Unstructured| super::add_dest_used_as_struct_elem(&llbc, body, 1);

    let read = body_of(vec![
        stmt(serde_json::json!({"StorageLive": 1})),
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
        })),
    ]);
    assert!(accepts(&read), "one field read");

    let write = body_of(vec![stmt(serde_json::json!({
        "Assign": [field_place(1, 1), {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(accepts(&write), "one field write");

    let twice = body_of(vec![
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
        })),
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 1)}, "No"]}]
        })),
    ]);
    assert!(accepts(&twice), "two field reads of one add");

    let escape = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": place_local(1)}, "No"]}]
    }))]);
    assert!(!accepts(&escape), "the pointer escapes by value");
    assert!(
        !elem_accepts(&escape),
        "a pointer copied into an unused local has no elem"
    );

    let bare = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": deref_place(1)}, "No"]}]
    }))]);
    assert!(!accepts(&bare), "a bare deref is not a struct field");

    let address = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {"place": field_place(1, 0), "kind": "Shared", "ptr_metadata": null}
        }]
    }))]);
    assert!(
        !accepts(&address),
        "&(*dest).field of an unresolved type is not a value read"
    );

    let field_place_ty = |i: u64, idx: u64, field_ty: serde_json::Value| {
        serde_json::json!({
            "kind": {"Projection": [deref_place(i), {"Field": [null, idx]}]},
            "ty": field_ty
        })
    };
    let address_word = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {
                "place": field_place_ty(1, 0, scalar_ty("Usize")),
                "kind": "Shared",
                "ptr_metadata": null
            }
        }]
    }))]);
    assert!(
        accepts(&address_word),
        "&(*dest).field of a copy scalar is a value read"
    );

    let address_typevar = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {
                "place": field_place_ty(1, 0, serde_json::json!({"TypeVar": {"Bound": [0, 0]}})),
                "kind": "Shared",
                "ptr_metadata": null
            }
        }]
    }))]);
    assert!(
        accepts(&address_typevar),
        "&(*dest).field of a TypeVar is a value read"
    );
    assert!(
        elem_accepts(&address),
        "&(*dest).field is an element use for getitem SSA"
    );

    let reborrow = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {"place": deref_place(1), "kind": "Mut", "ptr_metadata": null}
        }]
    }))]);
    assert!(
        elem_accepts(&reborrow),
        "&mut *dest is an element use for getitem SSA"
    );

    let deeper = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": {
            "kind": {"Projection": [field_place(1, 0), {"Field": [null, 0]}]},
            "ty": ty()
        }}, "No"]}]
    }))]);
    assert!(!accepts(&deeper), "a further projection escapes");

    let nested_index = body_of(vec![stmt(serde_json::json!({
        "Assign": [{
            "kind": {"Projection": [field_place(1, 1), {"Index": {
                "offset": {"Read": place_local(2)},
                "from_end": false
            }}]},
            "ty": ty()
        }, {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(
        accepts(&nested_index),
        "(*dest).field[i] is a struct-field use"
    );

    let dead = body_of(vec![]);
    assert!(!accepts(&dead), "no field use");

    let redefined = body_of(vec![
        stmt(serde_json::json!({
            "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
        })),
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
        })),
    ]);
    assert!(!accepts(&redefined), "a second definition stays residual");

    let call_body = {
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [{
                "statements": [stmt(serde_json::json!({
                    "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
                }))],
                "kind": "Regular",
                "terminator": {
                    "kind": {
                        "Call": {
                            "call": {
                                "func": {
                                    "Regular": {
                                        "kind": {"Fun": 1},
                                        "generics": {
                                            "regions": [],
                                            "types": [],
                                            "const_generics": [],
                                            "trait_refs": []
                                        }
                                    }
                                },
                                "args": [{"Read": field_place(1, 0)}]}, "dest": place_local(2),
                            "target": 0,
                            "on_unwind": 0
                        }
                    }
                }
            }]
        });
        serde_json::from_value::<Unstructured>(body_json).expect("call fixture parses")
    };
    assert!(accepts(&call_body), "a call argument is a value field use");

    let elem_escape_call = {
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
            "body": [{
                "statements": [
                    stmt(serde_json::json!({
                        "Assign": [place_local(1), {"Use": [{"Const": null}, "No"]}]
                    })),
                    stmt(serde_json::json!({
                        "Assign": [place_local(2), {
                            "Ref": {"place": field_place(1, 0), "kind": "Shared", "ptr_metadata": null}
                        }]
                    })),
                ],
                "kind": "Regular",
                "terminator": {
                    "kind": {
                        "Call": {
                            "call": {
                                "func": {
                                    "Regular": {
                                        "kind": {"Fun": 1},
                                        "generics": {
                                            "regions": [],
                                            "types": [],
                                            "const_generics": [],
                                            "trait_refs": []
                                        }
                                    }
                                },
                                "args": [{"Read": place_local(1)}]
                            },
                            "dest": place_local(2),
                            "target": 0,
                            "on_unwind": 0
                        }
                    }
                }
            }]
        });
        serde_json::from_value::<Unstructured>(body_json).expect("elem+escape call fixture parses")
    };
    assert!(
        !elem_accepts(&elem_escape_call),
        "a by-value call argument of dest escapes even with a Ref elem use"
    );
}

/// Distinctive `fn(bool) -> bool` so this fixture cannot collide with
/// `unit_fn_ptr` / BuiltinCodeFn. Member names are invented (`hook_a` /
/// `hook_b`); discovery keys on the `ProfileFunc` type, not those names.
fn bool_fn_ptr() -> serde_json::Value {
    json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [{"Scalar": "Bool"}],
                "output": {"Scalar": "Bool"},
                "is_unsafe": false
            }
        }
    })
}

fn const_fndef(fun: u64, ty: &serde_json::Value) -> serde_json::Value {
    json!({"Const": [{
        "FnDef": {
            "kind": {"Fun": fun},
            "generics": empty_generics()
        }
    }, ty]})
}

fn profilefunc_fun(
    def_id: u64,
    name: &str,
    inputs: Vec<serde_json::Value>,
    output: serde_json::Value,
    arg_count: u64,
    locals: Vec<serde_json::Value>,
    blocks: Vec<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "def_id": def_id,
        "item_meta": item_meta_json(&["fixture", name], "", true),
        "signature": {"is_unsafe": false, "inputs": inputs, "output": output},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": arg_count, "locals": locals},
            "body": blocks
        }}
    })
}

fn matching_profilefunc_member(def_id: u64, name: &str) -> serde_json::Value {
    let ty = json!({"Scalar": "Bool"});
    profilefunc_fun(
        def_id,
        name,
        vec![ty.clone()],
        ty.clone(),
        1,
        vec![local_of(0, &ty), local_of(1, &ty)],
        vec![bb(vec![], json!("Return"))],
    )
}

fn profilefunc_family_llbc() -> Llbc {
    let bool_fn = bool_fn_ptr();
    let profilefunc_adt = adt_node(1);
    let option_ty = option_adt_node(0, profilefunc_adt.clone());
    let ec_ty = adt_node(2);
    let unit = json!({"Tuple": []});
    let types = vec![
        option_enum(0),
        type_decl_alias(1, &["fixture", "ProfileFunc"], bool_fn.clone()),
        type_decl_struct(
            2,
            &["fixture", "ExecutionContext"],
            vec![json!({
                "name": "profilefunc",
                "ty": option_ty,
                "attr_info": null
            })],
        ),
    ];
    let hook_a = matching_profilefunc_member(0, "hook_a");
    let hook_b = matching_profilefunc_member(1, "hook_b");
    let other_fn = profilefunc_fun(
        2,
        "other_fn",
        vec![],
        unit.clone(),
        0,
        vec![local_of(0, &unit)],
        vec![bb(vec![], json!("Return"))],
    );
    let setllprofile = profilefunc_fun(
        3,
        "setllprofile",
        vec![ec_ty.clone(), option_adt_node(0, profilefunc_adt.clone())],
        unit.clone(),
        2,
        vec![
            local_of(0, &unit),
            local_of(1, &ec_ty),
            local_of(2, &option_adt_node(0, profilefunc_adt.clone())),
            local_of(3, &profilefunc_adt),
        ],
        vec![bb(
            vec![assign_stmt(
                place_of(3, &profilefunc_adt),
                json!({"Use": [const_fndef(0, &bool_fn), "Yes"]}),
            )],
            json!("Return"),
        )],
    );
    let setprofile = profilefunc_fun(
        4,
        "setprofile",
        vec![ec_ty.clone()],
        unit.clone(),
        1,
        vec![local_of(0, &unit), local_of(1, &ec_ty)],
        vec![
            bb(
                vec![],
                call_term(
                    3,
                    vec![move_op(place_of(1, &ec_ty)), const_fndef(1, &bool_fn)],
                    place_of(0, &unit),
                    1,
                    2,
                ),
            ),
            bb(vec![], json!("Return")),
            bb(vec![], json!("UnwindResume")),
        ],
    );
    let trace_profile = dynamic_call_fun(
        5,
        &["fixture", "trace_profile"],
        vec![],
        copy_local(1, &profilefunc_adt),
        vec![local_ty(0, &unit), local_ty(1, &profilefunc_adt)],
        1,
        vec![profilefunc_adt],
    );
    llbc_with_types(
        "fixture",
        types,
        vec![
            hook_a,
            hook_b,
            other_fn,
            setllprofile,
            setprofile,
            trace_profile,
        ],
    )
}

/// `discover_eval_hook_graphs` analogue: the family is the `ProfileFunc`
/// type plus stores into that slot (`setllprofile` / `setprofile`), not
/// baked member names.
#[test]
fn discover_profilefunc_graphs_keys_on_type_not_member_names() {
    let llbc = profilefunc_family_llbc();
    let graphs = super::discover_profilefunc_graphs(&llbc);
    assert!(
        graphs
            .iter()
            .any(|path| path == "hook_a" || path.ends_with("::hook_a")),
        "signature-matching / stored FnDef hook_a must be in the family, got {graphs:?}"
    );
    assert!(
        graphs
            .iter()
            .any(|path| path == "hook_b" || path.ends_with("::hook_b")),
        "signature-matching / setprofile-stored FnDef hook_b must be in the family, got {graphs:?}"
    );
    assert!(
        graphs
            .iter()
            .all(|path| !path.ends_with("other_fn") && path != "other_fn"),
        "a FunDecl whose fn type is not ProfileFunc stays out, got {graphs:?}"
    );
    let spellings = super::discover_profilefunc_spellings(&llbc);
    assert!(
        spellings
            .iter()
            .any(|s| s == "ProfileFunc" || s.ends_with("::ProfileFunc")),
        "alias leaf must be published, got {spellings:?}"
    );
    assert!(
        spellings.iter().any(|s| s.contains("fn(")),
        "expanded fn(...) after alias peel must be published, got {spellings:?}"
    );
}

/// `eval_hook_graphs_for_call` analogue: a Dynamic call whose operand is
/// the `ProfileFunc` alias carries the published family (`call.py
/// graphs_from` of `indirect_call`). Returning `None` here leaves
/// `graphs is None` and BFS residual-skips `app_profile_call`.
#[test]
fn profilefunc_graphs_for_call_returns_published_family() {
    let llbc = profilefunc_family_llbc();
    llbc.set_profilefunc_graphs(vec!["hook_a".into(), "hook_b".into()]);
    let operand: majit_charon_reader::ullbc::Operand =
        serde_json::from_value(copy_local(0, &adt_node(1))).expect("ProfileFunc operand parses");
    let graphs = super::profilefunc_graphs_for_call(&llbc, &operand)
        .expect("ProfileFunc-typed operand must name the published family");
    let names: Vec<String> = graphs.iter().map(|p| p.canonical_key()).collect();
    assert!(
        names
            .iter()
            .any(|n| n == "hook_a" || n.ends_with("::hook_a")),
        "published hook_a must be attached, got {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n == "hook_b" || n.ends_with("::hook_b")),
        "published hook_b must be attached, got {names:?}"
    );
}

/// `_trace`'s `profilefunc(...)` Dynamic call is `indirect_call` with
/// `c_graphs` (`rpbc.py FunctionReprBase.call`), so `find_all_graphs`
/// follows the family. Residual/`__dyn_call` leaves AccessDirect copies
/// outside `jit_graphs`.
#[test]
fn profilefunc_dynamic_call_attaches_published_family() {
    let llbc = profilefunc_family_llbc();
    llbc.set_profilefunc_graphs(vec!["hook_a".into(), "hook_b".into()]);
    let graph = crate::front::mir::lower_function(&llbc, "trace_profile")
        .expect("lower Dynamic ProfileFunc caller");
    let families: Vec<Vec<String>> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            crate::model::OpKind::IndirectCall {
                graphs: Some(graphs),
                ..
            } => Some(graphs.iter().map(|p| p.canonical_key()).collect()),
            _ => None,
        })
        .collect();
    assert_eq!(
        families.len(),
        1,
        "exactly one IndirectCall with graphs, got {families:?}; ops={:?}",
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .map(|op| format!("{:?}", op.kind))
            .collect::<Vec<_>>()
    );
    let names = &families[0];
    assert!(
        names
            .iter()
            .any(|n| n == "hook_a" || n.ends_with("::hook_a")),
        "hook_a must be on the indirect_call family, got {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n == "hook_b" || n.ends_with("::hook_b")),
        "hook_b must be on the indirect_call family, got {names:?}"
    );
}

#[test]
fn profilefunc_graphs_for_call_attaches_the_pbc_family() {
    let llbc = profilefunc_family_llbc();
    let operand: Operand = serde_json::from_value(move_op(place_of(0, &adt_node(1))))
        .expect("ProfileFunc local deserializes");
    let graphs = profilefunc_graphs_for_call(&llbc, &operand)
        .expect("FunctionsPBCRepr.call attaches c_graphs for ProfileFunc");
    assert!(
        graphs.iter().any(|path| {
            path.last_segment() == Some("hook_a") || path.display_name().ends_with("::hook_a")
        }),
        "hook_a must be in the attached family, got {graphs:?}"
    );
    assert!(
        graphs.iter().any(|path| {
            path.last_segment() == Some("hook_b") || path.display_name().ends_with("::hook_b")
        }),
        "hook_b must be in the attached family, got {graphs:?}"
    );
    let other: Operand = serde_json::from_value(move_op(place_of(0, &unit_fn_ptr())))
        .expect("unrelated fn local deserializes");
    assert!(
        profilefunc_graphs_for_call(&llbc, &other).is_none(),
        "a non-ProfileFunc fn pointer must not claim the family"
    );
}

#[test]
fn discover_profilefunc_graphs_ignores_unrelated_fn_pointer_fields() {
    let unit = json!({"Tuple": []});
    let types = vec![
        option_enum(0),
        type_decl_struct(
            1,
            &["fixture", "Hooks"],
            vec![json!({
                "name": "callback",
                "ty": option_adt_node(0, unit_fn_ptr()),
                "attr_info": null
            })],
        ),
    ];
    let other = profilefunc_fun(
        0,
        "plain_callback",
        vec![],
        unit.clone(),
        0,
        vec![local_of(0, &unit)],
        vec![bb(vec![], json!("Return"))],
    );
    let llbc = llbc_with_types("fixture", types, vec![other]);
    assert!(
        super::discover_profilefunc_graphs(&llbc).is_empty(),
        "a non-profilefunc Option<fn> field must not become a PBC family"
    );
    assert!(
        super::discover_profilefunc_spellings(&llbc).is_empty(),
        "no ProfileFunc alias / profilefunc field ⇒ no published spelling"
    );
}

#[test]
#[ignore]
fn interpreter_ullbc_discovers_app_profile_call_as_profilefunc_member() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = crate::test_support::load_ullbc(path);
    let graphs = super::discover_profilefunc_graphs(&llbc);
    assert!(
        graphs
            .iter()
            .any(|p| p.ends_with("app_profile_call") || p == "app_profile_call"),
        "executioncontext.py setprofile stores app_profile_call; got {graphs:?}"
    );
    let spellings = super::discover_profilefunc_spellings(&llbc);
    assert!(
        spellings
            .iter()
            .any(|s| s == "ProfileFunc" || s.ends_with("::ProfileFunc")),
        "ProfileFunc alias must be published from the interpreter artefact, got {spellings:?}"
    );
}
