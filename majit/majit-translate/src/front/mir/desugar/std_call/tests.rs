use majit_charon_reader::{
    Llbc,
    ullbc::{RegularCall, TyRef, Unstructured},
};

use super::{
    AccumulatorFacts, Lowering, RootStackAnalyzer, RootStackState, deref_callee_fun_decl,
    harvest_declared_gc_facts, impl_owner_path, no_immutable_fields,
};

fn with_fixture_lowering(llbc: &Llbc, f: impl FnOnce(&Lowering<'_>)) {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = AccumulatorFacts::build(llbc, &body);
    let tombstoned = std::collections::HashSet::new();
    let gc_struct_ids_llbc = harvest_declared_gc_facts(llbc).gc_struct_ids();
    let lowering = Lowering::new(
        llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &RootStackAnalyzer::new(llbc, &RootStackState::new(llbc)),
        false,
        false,
    )
    .unwrap();
    f(&lowering);
}

fn deref_impl_fixture(owner_path: &[&str]) -> (Llbc, RegularCall) {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        })
    };
    let owner_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": {"types": []}}
    });
    let owner_td = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(owner_path),
        "kind": {"Struct": []}
    });
    let mut name = owner_path
        .iter()
        .map(|s| serde_json::json!({"Ident": [s, 0]}))
        .collect::<Vec<_>>();
    name.push(serde_json::json!({"Impl": {
        "Ty": {"skip_binder": {"Value": [0, owner_adt.clone()]}},
        "Trait": 0
    }}));
    name.push(serde_json::json!({"Ident": ["deref", 0]}));
    let deref_fd = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": name,
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [{"Ref": [null, owner_adt.clone(), "Shared"]}],
            "output": {"Tuple": []}
        },
        "body": null
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [owner_td],
            "fun_decls": [null, deref_fd],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "ops", "deref", "Deref"]),
                "methods": [{"skip_binder": {"name": "deref"}}]
            }],
            "trait_impls": [{
                "impl_trait": {"id": 0, "generics": {"types": [owner_adt]}},
                "generics": {"types": [], "const_generics": [], "trait_refs": []},
                "methods": [{
                    "kind": {"TraitMethod": [0, 0]},
                    "skip_binder": {"id": 1}
                }]
            }]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = serde_json::from_value::<RegularCall>(serde_json::json!({
        "kind": {"Trait": [{
            "kind": {"TraitImpl": {
                "id": 0,
                "generics": {"types": [], "const_generics": [], "trait_refs": []}
            }}
        }, 0]},
        "generics": {}
    }))
    .expect("TraitImpl deref call parses");
    (llbc, call)
}

/// A user `foo::Vec` Deref impl is not identity. The leaf `Vec` would
/// alias it with `alloc::vec::Vec`; the declaration path does not.
#[test]
fn user_vec_deref_is_not_container_identity() {
    let slice_ty = TyRef::Other(serde_json::json!({
        "Ref": [null, {"Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, null]}, "Shared"]
    }));
    let i64_ty = TyRef::Other(serde_json::json!({
        "Ref": [null, {"Scalar": {"Integer": {"Signed": "I64"}}}, "Shared"]
    }));

    let (llbc, call) = deref_impl_fixture(&["foo", "Vec"]);
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            !lo.is_container_identity_deref_for(&call, None, None),
            "foo::Vec Deref must keep its body"
        );
        assert!(
            !lo.is_container_identity_deref_for(&call, None, Some(&slice_ty)),
            "foo::Vec Deref is not a slice identity either"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["alloc", "vec", "Vec"]);
    let fd = deref_callee_fun_decl(&call, None, &llbc).expect("Vec Deref FunDecl");
    assert_eq!(
        impl_owner_path(&llbc, fd).as_deref(),
        Some("alloc::vec::Vec")
    );
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            lo.is_container_identity_deref_for(&call, None, None),
            "alloc::vec::Vec Deref is identity in the lifted list model"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["alloc", "string", "String"]);
    let fd = deref_callee_fun_decl(&call, None, &llbc).expect("String Deref FunDecl");
    assert_eq!(
        impl_owner_path(&llbc, fd).as_deref(),
        Some("alloc::string::String")
    );
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            lo.is_container_identity_deref_for(&call, None, None),
            "alloc::string::String Deref is identity in the lifted string model"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["alloc", "boxed", "Box"]);
    let fd = deref_callee_fun_decl(&call, None, &llbc).expect("Box Deref FunDecl");
    assert_eq!(
        impl_owner_path(&llbc, fd).as_deref(),
        Some("alloc::boxed::Box")
    );
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            lo.is_container_identity_deref_for(&call, None, Some(&slice_ty)),
            "alloc::boxed::Box<[T]> Deref is identity"
        );
        assert!(
            !lo.is_container_identity_deref_for(&call, None, Some(&i64_ty)),
            "alloc::boxed::Box<i64> Deref is a thin pointer, not identity"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["foo", "Box"]);
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            !lo.is_container_identity_deref_for(&call, None, Some(&slice_ty)),
            "foo::Box with a slice dest is not alloc::boxed::Box"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["foo", "alloc", "vec", "Vec"]);
    let fd = deref_callee_fun_decl(&call, None, &llbc).expect("foo::alloc::vec::Vec Deref FunDecl");
    assert_eq!(
        impl_owner_path(&llbc, fd).as_deref(),
        Some("foo::alloc::vec::Vec")
    );
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            !lo.is_container_identity_deref_for(&call, None, None),
            "foo::alloc::vec::Vec Deref is not alloc::vec::Vec"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["foo", "alloc", "string", "String"]);
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            !lo.is_container_identity_deref_for(&call, None, None),
            "foo::alloc::string::String Deref is not alloc::string::String"
        );
    });

    let (llbc, call) = deref_impl_fixture(&["foo", "alloc", "boxed", "Box"]);
    with_fixture_lowering(&llbc, |lo| {
        assert!(
            !lo.is_container_identity_deref_for(&call, None, Some(&slice_ty)),
            "foo::alloc::boxed::Box with a slice dest is not alloc::boxed::Box"
        );
    });
}
