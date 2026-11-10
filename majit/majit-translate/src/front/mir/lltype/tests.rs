use super::{
    adt_def_is_raw_storage, adt_field_read_value_type, adt_node_class_root_identity,
    charon_type_value_to_ast_string, charon_type_value_to_enum_arg, dont_look_inside_return_token,
    field_ty_is_raw_word, is_cow_str_family_class_root, is_niche_option_gc_payload_spelling,
    is_oparg_newtype_spelling, is_os_path_family_root, is_raw_ptr_to_str_family_spelling,
    is_rstr_str_layout_declaration, is_rustc_pointer_niche_spelling, is_str_family_class_root,
    json_ty_is_by_value_copy_aggregate, json_ty_is_thin_pointer_element,
    json_ty_packed_copy_struct_byte_size, json_ty_scalar_element_spelling, no_immutable_fields,
    no_tombstoned_leaves, payload_tuple_suffix, raw_field_address_value_type,
    raw_ptr_pointee_is_non_adt_spelling, shaped_array_parts, type_decl_is_closure_env,
    tyref_array_suffix, tyref_enum_payload_value_type, tyref_fn_ptr_input_class_root,
    tyref_is_closure_env, tyref_is_cow_str, tyref_is_gcref_spelling, tyref_is_niche_option,
    tyref_is_option_of_string_value, tyref_is_ptr_to_int_cast_origin, tyref_is_raw_byte_ptr,
    tyref_is_raw_c_void_pp, tyref_is_raw_c_void_ptr, tyref_is_string_family, tyref_is_void_zst,
    tyref_niche_option_gc_payload_spelling, tyref_positional_aggregate_root,
    tyref_ptr_pointee_is_vec, tyref_raw_ptr_to_classed_instance_spelling,
    tyref_ref_param_class_root, tyref_shaped_tuple_root, tyref_to_attr_value_type_for_struct_field,
    tyref_to_value_type, tyref_tuple_suffix, unique_unsafecell_slice_deref_field_index,
};

use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};

use majit_charon_reader::{
    Llbc,
    ullbc::{Locals, RegularCall, TyRef, TypeDeclKind},
};

use crate::front::mir::tests::{
    adt_node, adt_ty, as_usize_cast_fixture, assert_cast_ptr_to_int_then_r_uint, call_leafs,
    code_struct, empty_generics, empty_llbc, fixture_item_meta, fixture_llbc, fixture_span,
    fixture_ty, graph_ops, i64_field, i64_ty, ident_path, item_meta_json, kwargs_unerase_cast_llbc,
    kwargs_unerase_ptr_tys, llbc_with_trait_impls, llbc_with_types, mut_ref_value_field,
    mut_u8_ty_json, named_adt, opaque_type_decl, option_adt_node, option_adt_node_from_types,
    option_enum, option_int_arg, option_shared_ref, range_contains_fixture, result_adt_node,
    result_enum, span_json, struct_decl, trait_clone_call_fixture, trait_clone_kind_payload,
    transparent_layout, type_decl_alias, type_decl_meta, type_decl_struct,
    tyref_to_attr_value_type, unit_fn_ptr, usize_ty, vec_decl_json, vec_of, vec_type_decls,
    vec_usize_ty_json, window_const_slice_ptr, window_empty_generics, window_llbc, window_local,
    window_meta, window_mut_slice, window_place, window_shared_slice, window_slice, window_span,
    window_usize,
};

#[test]
fn crate_prefixed_utf8str_is_str_family() {
    assert!(is_str_family_class_root(
        "pyre_object::unicodeobject::Utf8Str"
    ));
    assert!(is_str_family_class_root(
        "pyre_object::unicodeobject::UnicodeValueStorage"
    ));
    assert!(is_str_family_class_root("unicodeobject::Utf8Str"));
    assert!(is_str_family_class_root("Utf8Str"));
    assert!(!is_str_family_class_root("other::Utf8Str"));
    assert!(!is_str_family_class_root("other::Wtf8Buf"));
    assert!(!is_str_family_class_root(
        "other_crate::nested::unicodeobject::Utf8Str"
    ));
    assert!(!is_str_family_class_root(
        "pyre_object::lowlevel_string::LowlevelStringHeader"
    ));
    assert!(!is_str_family_class_root(
        "lowlevel_string::LowlevelStringHeader"
    ));
    assert!(!is_str_family_class_root("LowlevelStringHeader"));
    assert!(!is_str_family_class_root("other::LowlevelStringHeader"));
    assert!(is_rstr_str_layout_declaration(
        "pyre_object::lowlevel_string::LowlevelStringHeader"
    ));
    assert!(is_rstr_str_layout_declaration(
        "lowlevel_string::LowlevelStringHeader"
    ));
    assert!(is_rstr_str_layout_declaration("LowlevelStringHeader"));
    assert!(!is_rstr_str_layout_declaration(
        "other::LowlevelStringHeader"
    ));
    assert!(!is_rstr_str_layout_declaration("other::Utf8Str"));
    assert!(!is_rstr_str_layout_declaration("alloc::string::String"));
    assert!(!is_rstr_str_layout_declaration("string::String"));
    assert!(is_str_family_class_root("string::String"));
    // Family tables keep the dict-key wrappers; layout (`node_is_string_value`) does not.
    assert!(is_str_family_class_root("celldict::StrKey"));
    assert!(is_str_family_class_root("dictmultiobject::BytesKey"));
    assert!(is_str_family_class_root("pyre_object::celldict::StrKey"));
    assert!(!is_str_family_class_root("other::StrKey"));
    assert!(!is_str_family_class_root("other::BytesKey"));
    assert!(!is_rstr_str_layout_declaration("celldict::StrKey"));
    assert!(!is_rstr_str_layout_declaration("dictmultiobject::BytesKey"));
}

fn cow_enum_decl(def_id: u64) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": [
                {"Ident": ["alloc", 0]},
                {"Ident": ["borrow", 0]},
                {"Ident": ["Cow", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Enum": [
            {
                "name": "Borrowed",
                "fields": [{"name": null, "ty": {"TypeVar": {"Bound": [0, 0]}}, "attr_info": null}],
                "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
            },
            {
                "name": "Owned",
                "fields": [{"name": null, "ty": {"TypeVar": {"Bound": [0, 0]}}, "attr_info": null}],
                "discriminant": {"Scalar": {"Signed": ["Isize", "1"]}}
            }
        ]}
    })
}

fn builtin_str_node() -> serde_json::Value {
    serde_json::json!({"Adt": {"builtin": "Str", "generics": window_empty_generics()}})
}

fn cow_ty(def_id: u64, payload: serde_json::Value) -> TyRef {
    TyRef::Other(named_adt(def_id, vec![payload]))
}

/// `Cow<str>` is rstr (`unicodehelper.py str_decode_utf8`). `Cow<[u8]>`
/// stays the Cow ADT. A live Cow value paints `ValueType::Str` so
/// `Option::map` mergeinputargs sees SomeString on every producer.
#[test]
fn cow_str_paints_as_str_cow_bytes_stays_instance() {
    let string_decl = serde_json::json!({
        "def_id": 2,
        "item_meta": window_meta(&["alloc", "string", "String"], false),
        "kind": {"Struct": []}
    });
    let llbc = llbc_with_types(
        "fixture",
        vec![cow_enum_decl(0), option_enum(1), string_decl],
        vec![],
    );
    let cow_str = cow_ty(0, builtin_str_node());
    let cow_ref_str = TyRef::Other(serde_json::json!({
        "Ref": ["Erased", named_adt(0, vec![builtin_str_node()]), "Shared"]
    }));
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let cow_bytes = cow_ty(
        0,
        serde_json::json!({
            "Adt": {"builtin": "Slice", "generics": {"types": [u8_ty]}}
        }),
    );
    let cow_string = cow_ty(
        0,
        serde_json::json!({
            "Adt": {"id": 2, "builtin": null, "generics": window_empty_generics()}
        }),
    );
    let option_cow_str = TyRef::Other(option_adt_node(1, named_adt(0, vec![builtin_str_node()])));

    assert!(tyref_is_cow_str(&cow_str, &llbc));
    assert!(tyref_is_cow_str(&cow_ref_str, &llbc));
    assert!(tyref_is_cow_str(&cow_string, &llbc));
    assert!(!tyref_is_cow_str(&cow_bytes, &llbc));
    assert!(!tyref_is_cow_str(&option_cow_str, &llbc));

    assert!(tyref_is_string_family(&cow_str, &llbc));
    assert!(!tyref_is_string_family(&cow_bytes, &llbc));

    assert!(matches!(
        tyref_to_value_type(&cow_str, &llbc),
        ValueType::Str
    ));
    assert!(matches!(
        tyref_to_value_type(&cow_ref_str, &llbc),
        ValueType::Str
    ));
    assert!(matches!(
        tyref_to_value_type(&cow_string, &llbc),
        ValueType::Str
    ));
    assert!(
        !matches!(tyref_to_value_type(&cow_bytes, &llbc), ValueType::Str),
        "Cow<[u8]> must stay the Cow ADT, got {:?}",
        tyref_to_value_type(&cow_bytes, &llbc)
    );
    let option_cow_bytes = TyRef::Other(option_adt_node(
        1,
        named_adt(
            0,
            vec![serde_json::json!({
                "Adt": {"builtin": "Slice", "generics": {"types": [u8_ty]}}
            })],
        ),
    ));
    assert!(
        matches!(tyref_to_value_type(&option_cow_str, &llbc), ValueType::Str),
        "Option<Cow<str>> is a nullable SomeString, got {:?}",
        tyref_to_value_type(&option_cow_str, &llbc)
    );
    assert!(
        tyref_is_option_of_string_value(&option_cow_str, &llbc),
        "Option<Cow<str>> is the nullable-string niche"
    );
    assert!(
        tyref_is_niche_option(&option_cow_str, &llbc),
        "Option<Cow<str>> is_some is ptr_nonzero"
    );
    assert!(
        !tyref_is_option_of_string_value(&option_cow_bytes, &llbc),
        "Option<Cow<[u8]>> stays the Cow ADT"
    );
    assert!(
        !matches!(
            tyref_to_value_type(&option_cow_bytes, &llbc),
            ValueType::Str
        ),
        "Option<Cow<[u8]>> stays tagged, got {:?}",
        tyref_to_value_type(&option_cow_bytes, &llbc)
    );
}

/// Class-root spellings of `Cow<str-family>` are SomeString
/// (`unicodehelper.py str_decode_utf8`). `Cow<[u8]>` stays the Cow ADT.
#[test]
fn cow_str_class_root_is_str_family_cow_bytes_is_not() {
    for root in [
        "borrow::Cow<str>",
        "alloc::borrow::Cow<str>",
        "Cow<str>",
        "Cow<&str>",
        "borrow::Cow<&str>",
        "Cow<'static, str>",
        "borrow::Cow<String>",
        "borrow::Cow<string::String>",
    ] {
        assert!(
            is_cow_str_family_class_root(root),
            "{root} is Cow<str-family>"
        );
        assert!(
            !is_str_family_class_root(root),
            "{root} is not an rstr header"
        );
    }
    for root in [
        "borrow::Cow<[u8]>",
        "Cow<[u8]>",
        "alloc::borrow::Cow<[u8]>",
        "borrow::Cow",
        "Cow",
        "option::Option<str>",
    ] {
        assert!(
            !is_cow_str_family_class_root(root),
            "{root} is not Cow<str-family>"
        );
    }
}

#[test]
fn os_path_family_roots_are_str_family_and_reject_lookalikes() {
    for root in [
        "PathBuf",
        "path::PathBuf",
        "std::path::PathBuf",
        "Path",
        "path::Path",
        "OsString",
        "ffi::OsString",
        "std::ffi::OsString",
        "ffi::os_str::OsString",
        "OsStr",
        "ffi::OsStr",
    ] {
        assert!(
            is_os_path_family_root(root),
            "{root} is a std path/ffi spelling"
        );
        assert!(
            is_str_family_class_root(root),
            "{root} projects as the rpath.py s_Str0 string family"
        );
    }
    assert!(!is_os_path_family_root("other::PathBuf"));
    assert!(!is_os_path_family_root("other::OsString"));
    assert!(!is_str_family_class_root("other::PathBuf"));
    assert!(!is_str_family_class_root("other::OsString"));
    assert!(!is_os_path_family_root("String"));
    assert!(!is_os_path_family_root("string::String"));
}

#[test]
fn oparg_newtype_spelling_is_the_compiler_core_wrapper() {
    for spelling in [
        "ConstIdx",
        "LoadAttr",
        "OpArg",
        "bytecode::oparg::ConstIdx",
        "bytecode::oparg::LoadAttr",
        "oparg::ConstIdx",
        "rustpython_compiler_core::bytecode::oparg::ConstIdx",
        "rustpython_compiler_core::bytecode::oparg::LoadAttr",
        "&ConstIdx",
        "&mut LoadAttr",
    ] {
        assert!(
            is_oparg_newtype_spelling(spelling),
            "{spelling} is a newtype_oparg! wrapper"
        );
    }
    assert!(!is_oparg_newtype_spelling("other::ConstIdx"));
    assert!(!is_oparg_newtype_spelling("other::LoadAttr"));
    assert!(!is_oparg_newtype_spelling("String"));
    assert!(!is_oparg_newtype_spelling("u32"));
}

#[test]
fn raw_ptr_to_str_family_spelling_is_the_star_prefix() {
    assert!(is_raw_ptr_to_str_family_spelling(
        "*mut unicodeobject::Utf8Str"
    ));
    assert!(is_raw_ptr_to_str_family_spelling(
        "*const unicodeobject::Utf8Str"
    ));
    assert!(is_raw_ptr_to_str_family_spelling("*mut string::String"));
    assert!(!is_raw_ptr_to_str_family_spelling("unicodeobject::Utf8Str"));
    assert!(!is_raw_ptr_to_str_family_spelling(
        "&unicodeobject::Utf8Str"
    ));
    assert!(!is_raw_ptr_to_str_family_spelling("string::String"));
    assert!(!is_raw_ptr_to_str_family_spelling("str"));
    assert!(!is_raw_ptr_to_str_family_spelling(
        "*mut pyobject::PyObject"
    ));
}

fn classed_raw_ptr_llbc() -> Llbc {
    let mut opaque = type_decl_struct(5, &["ffi_raw", "NativeSession"], vec![]);
    opaque["kind"] = serde_json::json!("Opaque");
    llbc_with_types(
        "fixture",
        vec![
            type_decl_struct(0, &["pyre_object", "pyobject", "PyObject"], vec![]),
            type_decl_alias(1, &["pyre_object", "pyobject", "PyObjectRef"], {
                serde_json::json!({"RawPtr": [adt_node(0), "Mut"]})
            }),
            type_decl_struct(
                2,
                &["foo", "bar_state"],
                vec![serde_json::json!({
                    "name": "sess",
                    "ty": {"RawPtr": [adt_node(5), "Mut"]},
                    "attr_info": null
                })],
            ),
            type_decl_struct(3, &["pyre_object", "unicodeobject", "Utf8Str"], vec![]),
            type_decl_struct(4, &["majit_gc", "header", "GCREFOpaque"], vec![]),
            opaque,
        ],
        vec![],
    )
}

#[test]
fn raw_ptr_to_classed_instance_uses_the_pointee_declaration() {
    let llbc = classed_raw_ptr_llbc();
    let spelling = |ty: &TyRef| tyref_raw_ptr_to_classed_instance_spelling(ty, &llbc);
    let object = spelling(&TyRef::Other(raw_mut_adt(0)));
    assert!(
        object
            .as_deref()
            .is_some_and(|s| s.starts_with("*mut ") && s.contains("PyObject")),
        "declared *mut PyObject is classed, got {object:?}"
    );
    let alias = spelling(&TyRef::Other(adt_node(1)));
    assert!(
        alias.as_deref().is_some_and(|s| s.starts_with("*mut ")
            && s.contains("PyObject")
            && !s.contains("PyObjectRef")),
        "PyObjectRef alias peels to the pointer spelling, got {alias:?}"
    );
    let const_object = spelling(&TyRef::Other(serde_json::json!({
        "RawPtr": [adt_node(0), "Shared"]
    })));
    assert!(
        const_object
            .as_deref()
            .is_some_and(|s| s.starts_with("*const ") && s.contains("PyObject")),
        "declared *const PyObject is classed, got {const_object:?}"
    );
    let lower = spelling(&TyRef::Other(raw_mut_adt(2)));
    assert!(
        lower
            .as_deref()
            .is_some_and(|s| s.starts_with("*mut ") && s.contains("bar_state")),
        "lowercase declared struct is classed, got {lower:?}"
    );
    let opaque = spelling(&TyRef::Other(raw_mut_adt(5)));
    assert!(
        opaque
            .as_deref()
            .is_some_and(|s| s.starts_with("*mut ") && s.contains("NativeSession")),
        "declared Opaque pointee is classed, got {opaque:?}"
    );
    assert_eq!(
        spelling(&TyRef::Other(raw_mut_adt(99))),
        None,
        "*mut SomeUndeclared (uppercase, no TypeDecl) is not classed"
    );
    assert_eq!(
        spelling(&TyRef::Other(adt_node(0))),
        None,
        "owned PyObject is not a raw pointer"
    );
    assert_eq!(
        spelling(&TyRef::Other(option_shared_ref(adt_node(0)))),
        None,
        "&PyObject is not a raw pointer"
    );
    assert_eq!(
        spelling(&TyRef::Other(raw_mut_adt(3))),
        None,
        "str-family pointee is not a classed instance"
    );
    assert_eq!(
        spelling(&TyRef::Other(raw_mut_adt(4))),
        None,
        "GCREFOpaque pointee is not a classed instance"
    );
    assert_eq!(
        spelling(&TyRef::Other(mut_u8_ty_json())),
        None,
        "*mut u8 is not a classed instance"
    );
    assert!(raw_ptr_pointee_is_non_adt_spelling("*mut u8", "u8"));
    assert!(raw_ptr_pointee_is_non_adt_spelling(
        "*mut GCREFOpaque",
        "GCREFOpaque"
    ));
    assert!(raw_ptr_pointee_is_non_adt_spelling(
        "*mut unicodeobject::Utf8Str",
        "unicodeobject::Utf8Str"
    ));
}

#[test]
fn derive_program_metadata_harvests_opaque_adt_as_declared_layout_identity() {
    let llbc = classed_raw_ptr_llbc();
    let (_, _, fields, _, _, _, _, ids) = super::derive_program_metadata(
        &llbc,
        &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &std::collections::HashSet::new(),
        &[],
    );
    assert!(
        ids.get("ffi_raw::NativeSession")
            .and_then(|slot| slot.as_ref())
            .is_some(),
        "Opaque TypeDecl must receive a StructId (lltype.py Struct._first_struct)"
    );
    assert!(
        fields.fields.get("ffi_raw::NativeSession").is_some(),
        "Opaque TypeDecl must have a harvested field row"
    );
    assert!(
        ids.get("foo::bar_state")
            .and_then(|slot| slot.as_ref())
            .is_some(),
        "lowercase declared struct still has a StructId"
    );
    assert!(
        ids.get("SomeUndeclared").is_none(),
        "undeclared uppercase name has no layout identity"
    );
}

/// A Deduplicated body first reached at the walk cutoff is still walked
/// when the same id appears later at a shallow depth. `seen_dedup` is
/// the cycle guard; entering `llbc.dedup_body` restarts depth at 0.
#[test]
fn deduplicated_opaque_raw_ptr_survives_truncated_first_visit() {
    let opaque = opaque_type_decl(0, &["ffi_raw", "NativeSession"]);
    let seed = type_decl_struct(
        1,
        &["fixture", "Seed"],
        vec![serde_json::json!({
            "name": "ty",
            "ty": {"Value": [7, {"RawPtr": [adt_node(0), "Mut"]}]},
            "attr_info": null
        })],
    );
    let llbc = llbc_with_types("fixture", vec![opaque, seed], vec![]);
    assert!(
        llbc.dedup_body(7).is_some(),
        "Value:[7, RawPtr] must fill dedup_body"
    );
    let node = serde_json::json!({"Deduplicated": 7});
    let mut ids = std::collections::HashSet::new();
    let mut seen_dedup = std::collections::HashSet::new();
    super::super::collect_opaque_raw_ptr_from_node(&llbc, &node, &mut ids, &mut seen_dedup, 24);
    super::super::collect_opaque_raw_ptr_from_node(&llbc, &node, &mut ids, &mut seen_dedup, 0);
    assert!(
        ids.contains(&0),
        "Opaque pointee of a Deduplicated body must survive a truncated first visit; ids={ids:?}"
    );
}

#[test]
fn niche_option_gc_payload_spelling_is_option_of_instance_or_string() {
    let intern_none = |s: &str| is_niche_option_gc_payload_spelling(s, None);
    assert!(intern_none("option::Option<string::String>"));
    assert!(intern_none("Option<*mut unicodeobject::Utf8Str>"));
    assert!(intern_none("Option<path::PathBuf>"));
    assert!(intern_none("option::Option<PathBuf>"));
    assert!(intern_none("Option<ffi::OsString>"));
    assert!(!intern_none("Option<other::PathBuf>"));
    assert!(intern_none("Option<borrow::Cow<str>>"));
    assert!(intern_none("option::Option<Cow<'static, str>>"));
    assert!(intern_none("Option<alloc::borrow::Cow<String>>"));
    assert!(!intern_none("Option<borrow::Cow<[u8]>>"));
    assert!(!intern_none("Option<pyobject::PyObject>"));
    assert!(!intern_none("Option<W_BufferedReader>"));
    assert!(!intern_none("pyobject::PyObject"));
    assert!(!intern_none("string::String"));
    assert!(!intern_none("Option<fn()>"));
    assert!(!intern_none("Option<i64>"));
    assert!(!intern_none("Option<*mut u8>"));
    assert!(!intern_none("Option<&[u8]>"));
    assert!(!intern_none("Option<*mut GCREFOpaque>"));
    // Uppercase / `::` spelling is not GC without a harvested layout.
    assert!(!intern_none("Option<&pyobject::PyObject>"));
    assert!(!intern_none("Option<*mut pyobject::PyObject>"));
    assert!(!intern_none("Option<&W_BufferedReader>"));
    assert!(!intern_none("Option<NonNull<W_BufferedReader>>"));
    assert!(!intern_none("Option<&*mut PyObject>"));
    assert!(!intern_none("Option<*mut SomeUndeclared>"));

    let (layouts, _ids) = crate::test_support::gc_layout_table_for(&[
        "pyobject::PyObject",
        "PyObject",
        "W_BufferedReader",
        "SomeUndeclared",
    ]);
    let intern = |s: &str| is_niche_option_gc_payload_spelling(s, Some(&layouts));
    assert!(intern("Option<&pyobject::PyObject>"));
    assert!(intern("Option<*mut pyobject::PyObject>"));
    assert!(intern("Option<&W_BufferedReader>"));
    assert!(intern("Option<NonNull<W_BufferedReader>>"));
    assert!(intern("Option<&*mut PyObject>"));
    assert!(intern("Option<&*mut pyobject::PyObject>"));
    assert!(intern("Option<*mut PyObject>"));
    // Stamp peels `type PyObjectRef = *mut PyObject` via
    // `type_node_peel_aliases`; intern has the class-root spelling, not
    // the TypeDecl, so an unresolved alias leaf is not GC.
    assert!(!intern("Option<PyObjectRef>"));
    assert!(!intern("Option<&PyObjectRef>"));
    assert!(intern("Option<*mut SomeUndeclared>"));
    assert!(!intern("Option<pyobject::PyObject>"));
    assert!(!intern("Option<W_BufferedReader>"));
    assert!(!intern("Option<*mut u8>"));
}

#[test]
fn is_rustc_pointer_niche_spelling_matches_rustc_option_niche() {
    assert!(is_rustc_pointer_niche_spelling("&T"));
    assert!(is_rustc_pointer_niche_spelling("&mut T"));
    assert!(is_rustc_pointer_niche_spelling("NonNull<object::PyObject>"));
    assert!(is_rustc_pointer_niche_spelling(
        "core::ptr::non_null::NonNull<object::PyObject>"
    ));
    assert!(is_rustc_pointer_niche_spelling("Box<T>"));
    assert!(is_rustc_pointer_niche_spelling("alloc::boxed::Box<T>"));
    assert!(is_rustc_pointer_niche_spelling("Arc<T>"));
    assert!(is_rustc_pointer_niche_spelling("Rc<T>"));
    assert!(!is_rustc_pointer_niche_spelling("*mut object::PyObject"));
    assert!(!is_rustc_pointer_niche_spelling("*const object::PyObject"));
    assert!(!is_rustc_pointer_niche_spelling("error::PyError"));
    assert!(!is_rustc_pointer_niche_spelling("i64"));
    assert!(!is_rustc_pointer_niche_spelling("Box"));
    assert!(!is_rustc_pointer_niche_spelling("fn()"));
    assert!(!is_rustc_pointer_niche_spelling(
        "Option<NonNull<object::PyObject>>"
    ));
}

/// `node_is_string_value` identifies the rstr `STR` layout set by
/// declaration path, not the last path component. `other::Utf8Str` is a
/// different type. `StrKey` / `BytesKey` wrap a STR pointer.
#[test]
fn node_is_string_value_uses_declaration_identity() {
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let lookalike = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["other", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let header = serde_json::json!({
        "def_id": 2,
        "item_meta": window_meta(&["pyre_object", "lowlevel_string", "LowlevelStringHeader"], true),
        "kind": {"Struct": []}
    });
    let header_lookalike = serde_json::json!({
        "def_id": 3,
        "item_meta": window_meta(&["other", "LowlevelStringHeader"], true),
        "kind": {"Struct": []}
    });
    let rust_heap = serde_json::json!({
        "def_id": 4,
        "item_meta": window_meta(&["alloc", "string", "String"], true),
        "kind": {"Struct": []}
    });
    let str_key = serde_json::json!({
        "def_id": 5,
        "item_meta": window_meta(&["pyre_object", "celldict", "StrKey"], true),
        "kind": {"Struct": []}
    });
    let bytes_key = serde_json::json!({
        "def_id": 6,
        "item_meta": window_meta(&["pyre_object", "dictmultiobject", "BytesKey"], true),
        "kind": {"Struct": []}
    });
    let str_key_lookalike = serde_json::json!({
        "def_id": 7,
        "item_meta": window_meta(&["other", "StrKey"], true),
        "kind": {"Struct": []}
    });
    let utf8_leaf = serde_json::json!({
        "def_id": 8,
        "item_meta": window_meta(&["Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let bytes_leaf = serde_json::json!({
        "def_id": 9,
        "item_meta": window_meta(&["BytesBlock"], true),
        "kind": {"Struct": []}
    });
    let header_leaf = serde_json::json!({
        "def_id": 10,
        "item_meta": window_meta(&["LowlevelStringHeader"], true),
        "kind": {"Struct": []}
    });
    let llbc = window_llbc(
        vec![
            utf8,
            lookalike,
            header,
            header_lookalike,
            rust_heap,
            str_key,
            bytes_key,
            str_key_lookalike,
            utf8_leaf,
            bytes_leaf,
            header_leaf,
        ],
        vec![],
    );
    let adt = |id: u64| {
        TyRef::Other(serde_json::json!({
            "Adt": {"id": id, "builtin": null, "generics": window_empty_generics()}
        }))
    };
    let raw = |id: u64| {
        TyRef::Other(serde_json::json!({
            "RawPtr": [{
                "Adt": {"id": id, "builtin": null, "generics": window_empty_generics()}
            }, "Mut"]
        }))
    };
    assert!(super::tyref_is_string_value(&adt(0), &llbc));
    assert!(!super::tyref_is_string_value(&adt(1), &llbc));
    // Header is the lltype `{hash, len}` overlay (`STR_LENGTH_FIELD_CLASS_ROOTS`),
    // rstr `STR` layout (`is_rstr_str_layout_declaration`).
    assert!(super::tyref_is_string_value(&adt(2), &llbc));
    assert!(is_rstr_str_layout_declaration(
        "pyre_object::lowlevel_string::LowlevelStringHeader"
    ));
    assert!(!super::tyref_is_string_value(&adt(3), &llbc));
    assert!(!super::tyref_is_string_value(&adt(4), &llbc));
    assert!(!super::tyref_is_string_value(&adt(5), &llbc));
    assert!(!super::tyref_is_string_value(&adt(6), &llbc));
    assert!(!super::tyref_is_string_value(&adt(7), &llbc));
    // Crate-root leaves of the old layout set stay string values;
    // `other::Utf8Str` above does not.
    assert!(super::tyref_is_string_value(&adt(8), &llbc));
    assert!(super::tyref_is_string_value(&adt(9), &llbc));
    assert!(super::tyref_is_string_value(&adt(10), &llbc));
    assert!(super::tyref_raw_ptr_pointee_is_string_value(&raw(0), &llbc));
    assert!(!super::tyref_raw_ptr_pointee_is_string_value(
        &raw(1),
        &llbc
    ));
    assert!(super::tyref_raw_ptr_pointee_is_string_value(&raw(2), &llbc));
    assert!(!super::tyref_raw_ptr_pointee_is_string_value(
        &raw(3),
        &llbc
    ));
    assert!(!super::tyref_raw_ptr_pointee_is_string_value(
        &raw(4),
        &llbc
    ));
    assert!(!super::tyref_raw_ptr_pointee_is_string_value(
        &raw(5),
        &llbc
    ));
    assert!(!super::tyref_raw_ptr_pointee_is_string_value(
        &raw(6),
        &llbc
    ));
    assert!(matches!(
        tyref_to_value_type(&adt(0), &llbc),
        ValueType::Str
    ));
    assert!(!matches!(
        tyref_to_value_type(&adt(5), &llbc),
        ValueType::Str
    ));
    assert!(!matches!(
        tyref_to_value_type(&raw(5), &llbc),
        ValueType::Str
    ));
    assert!(!matches!(
        tyref_to_value_type(&raw(6), &llbc),
        ValueType::Str
    ));
}

/// rstr `STR` pointee (`Utf8Str`) is GC (`TO._gckind == 'gc'`) even
/// without a `GcType` seed. A raw struct cannot hold that pointer
/// (`adt_def_is_raw_storage`).
#[test]
fn utf8str_pointee_is_not_a_raw_word() {
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let llbc = window_llbc(vec![utf8], vec![]);
    let node = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let gc = std::collections::HashSet::new();
    assert!(
        !super::pointee_is_raw_word(&node, &llbc, &gc),
        "Utf8Str pointee is rstr STR, not a raw word"
    );
}

/// rstr `STR` identity is the declaration path
/// (`is_rstr_str_layout_declaration`). `other::Utf8Str` is a different
/// type (`rstr.py` `GcStruct('rpy_string')`, `TO._gckind == 'gc'`).
#[test]
fn raw_ptr_to_other_utf8str_pointee_follows_declaration() {
    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let lookalike = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["other", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let owner_rstr = struct_decl(
        2,
        ident_path(&["fixture", "RstrOwner"]),
        serde_json::json!([{
            "name": "p",
            "ty": {"RawPtr": [adt_node(0), "Mut"]},
            "attr_info": null
        }]),
        false,
    );
    let owner_other = struct_decl(
        3,
        ident_path(&["fixture", "OtherOwner"]),
        serde_json::json!([{
            "name": "p",
            "ty": {"RawPtr": [adt_node(1), "Mut"]},
            "attr_info": null
        }]),
        false,
    );
    let llbc = window_llbc(vec![utf8, lookalike, owner_rstr, owner_other], vec![]);
    let adt = |id: u64| {
        serde_json::json!({
            "Adt": {"id": id, "builtin": null, "generics": window_empty_generics()}
        })
    };
    let gc = std::collections::HashSet::new();
    let immut = no_immutable_fields();
    assert!(
        !super::pointee_is_raw_word(&adt(0), &llbc, &gc),
        "unicodeobject::Utf8Str is rstr STR"
    );
    assert!(
        super::pointee_is_raw_word(&adt(1), &llbc, &gc),
        "other::Utf8Str is not rstr STR"
    );
    assert!(
        !adt_def_is_raw_storage(2, &llbc, &gc, immut, &mut Vec::new()),
        "owner of *mut unicodeobject::Utf8Str is not raw storage"
    );
    assert!(
        adt_def_is_raw_storage(3, &llbc, &gc, immut, &mut Vec::new()),
        "owner of *mut other::Utf8Str is classified by its own declaration"
    );
}

/// A `dont_look_inside` residual of `*mut Utf8Str` is nullable even when
/// Utf8Str has no `GcType` impl. `registered_gc_ptr_class_root_with` is
/// the gckind answer and stays None; nullability follows the raw
/// pointer (`model.py` `SomeString.noneify`). `binaryop.py`
/// `is__default` must not fold `is_(ret, None)`. A by-value / `&`
/// string stays non-nullable. rstr `STR` is still a GC pointee
/// (`is_rstr_str_family_spelling` / `pointee_is_raw_word`).
#[test]
fn dont_look_inside_return_raw_str_ptr_is_nullable() {
    use crate::annotator::binaryop::is__default;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::model::{SomeValue, s_none};
    use crate::flowspace::model::{Hlvalue, Variable};
    use crate::flowspace::operation::{HLOperation, OpKind};

    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let llbc = window_llbc(vec![utf8], vec![]);
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert!(
        gc.is_empty(),
        "Utf8Str has no GcType impl; gc_struct_ids must stay empty"
    );
    let empty = std::collections::HashSet::<String>::new();
    let spec = crate::ErrorCarrierSpec::default();
    let adt = TyRef::Other(serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    }));
    let raw = TyRef::Other(serde_json::json!({
        "RawPtr": [{
            "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
        }, "Mut"]
    }));
    let shared = TyRef::Other(serde_json::json!({
        "Ref": ["Erased", {
            "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
        }, "Shared"]
    }));

    assert!(super::dont_look_inside_return_is_str(&raw, &llbc, spec));
    assert!(super::dont_look_inside_return_can_be_none(
        &raw, &llbc, spec, &gc, &empty
    ));
    assert!(super::dont_look_inside_return_is_str(&adt, &llbc, spec));
    assert!(!super::dont_look_inside_return_can_be_none(
        &adt, &llbc, spec, &gc, &empty
    ));
    assert!(super::dont_look_inside_return_is_str(&shared, &llbc, spec));
    assert!(!super::dont_look_inside_return_can_be_none(
        &shared, &llbc, spec, &gc, &empty
    ));

    let intern = |can_be_none: bool| {
        let s = crate::codewriter::annotation_state::valuetype_to_someshell(&ValueType::Str)
            .expect("Str shells as SomeString");
        if can_be_none {
            Bookkeeper::nullable_raw_ptr_annotation(s)
        } else {
            s
        }
    };
    let is_none_of = |s: SomeValue| {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let mut ret = Variable::named("ret");
        let mut n = Variable::named("n");
        ann.setbinding(&mut ret, s);
        ann.setbinding(&mut n, s_none());
        is__default(
            &ann,
            &HLOperation::new(
                OpKind::Is,
                vec![Hlvalue::Variable(ret), Hlvalue::Variable(n)],
            ),
        )
    };

    let SomeValue::String(s) = intern(true) else {
        panic!("raw *mut Utf8Str intern is SomeString");
    };
    assert!(
        s.inner.can_be_none,
        "raw *mut Utf8Str intern is SomeString(can_be_none=true) (SomeString.noneify)"
    );
    assert!(
        is_none_of(SomeValue::String(s)).const_().is_none(),
        "is_(nullable SomeString, None) must not fold (is__default)"
    );

    let SomeValue::String(by_value) = intern(false) else {
        panic!("by-value / & string intern is SomeString");
    };
    assert!(
        !by_value.inner.can_be_none,
        "by-value / & string intern stays SomeString(can_be_none=false)"
    );
    assert_eq!(
        is_none_of(SomeValue::String(by_value)).const_(),
        Some(&crate::flowspace::model::ConstValue::Bool(false)),
        "is_(non-nullable SomeString, None) folds to False (is__default)"
    );
}

fn family_ptr_newtype_llbc(
    inner_path: &[&str],
    wrapper_path: &[&str],
    wrap_name: &str,
    unwrap_name: &str,
) -> Llbc {
    pointer_newtype_llbc(inner_path, wrapper_path, wrap_name, unwrap_name, false)
}

/// `gc_object` stores a `Vec` in the pointee (`adt_path_is_gc_container`),
/// so `*mut T` is a GC object pointer. Scalar fields alone are raw storage
/// and the wrapper would erase to that address.
fn pointer_newtype_llbc(
    inner_path: &[&str],
    wrapper_path: &[&str],
    wrap_name: &str,
    unwrap_name: &str,
    gc_object: bool,
) -> Llbc {
    let inner_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    });
    let wrapper_adt = serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    });
    let inner_ptr = serde_json::json!({"RawPtr": [inner_adt.clone(), "Mut"]});
    let inner_fields = if gc_object {
        let vec_adt = serde_json::json!({
            "Adt": {"id": 2, "builtin": null, "generics": window_empty_generics()}
        });
        serde_json::json!([
            {"name": "items", "ty": vec_adt, "attr_info": null}
        ])
    } else {
        serde_json::json!([
            {"name": "hash", "ty": {"Scalar": {"Integer": {"Signed": "Isize"}}}, "attr_info": null},
            {"name": "length", "ty": window_usize(), "attr_info": null}
        ])
    };
    let inner = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(inner_path, true),
        "kind": {"Struct": inner_fields}
    });
    let wrapper = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(wrapper_path, true),
        "kind": {"Struct": [
            {"name": null, "ty": inner_ptr.clone(), "attr_info": null}
        ]},
        "layout": transparent_layout()
    });
    let wrap = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", wrap_name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [inner_ptr.clone()],
            "output": wrapper_adt.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &wrapper_adt),
                window_local(1, &inner_ptr)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        window_place(serde_json::json!({"Local": 0}), &wrapper_adt),
                        {"Aggregate": [
                            {"Adt": [
                                {"id": 1, "generics": window_empty_generics()},
                                null,
                                null
                            ]},
                            [{"Move": window_place(
                                serde_json::json!({"Local": 1}),
                                &inner_ptr
                            )}]
                        ]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let field_place = window_place(
        serde_json::json!({"Projection": [
            window_place(serde_json::json!({"Local": 1}), &wrapper_adt),
            {"Field": [null, 0]}
        ]}),
        &inner_ptr,
    );
    let unwrap = serde_json::json!({
        "def_id": 1,
        "item_meta": window_meta(&["fixture", unwrap_name], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [wrapper_adt.clone()],
            "output": inner_ptr.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &inner_ptr),
                window_local(1, &wrapper_adt)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        window_place(serde_json::json!({"Local": 0}), &inner_ptr),
                        {"Use": [{"Read": field_place}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let mut types = vec![inner, wrapper];
    if gc_object {
        types.push(serde_json::json!({
            "def_id": 2,
            "item_meta": window_meta(&["alloc", "vec", "Vec"], false),
            "kind": {"Struct": []}
        }));
        // `*mut Obj` is a GC instance pointer only when Obj is in
        // `gc_struct_ids` (`GcType` / first-field ancestor). A Vec field
        // is not that declaration.
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": types,
                "fun_decls": [wrap, unwrap],
                "global_decls": [],
                "trait_decls": [{
                    "def_id": 0,
                    "item_meta": window_meta(&["majit_gc", "header", "GcType"], false)
                }],
                "trait_impls": [gctype_impl_for_adt(0)]
            }
        });
        return Llbc::from_slice(file.to_string().as_bytes())
            .expect("gc object pointer newtype fixture parses");
    }
    window_llbc(types, vec![wrap, unwrap])
}

fn wrapper_tyref() -> TyRef {
    TyRef::Other(serde_json::json!({
        "Adt": {"id": 1, "builtin": null, "generics": window_empty_generics()}
    }))
}

fn graph_has_wrapper_ctor_or_pos0(graph: &FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            } => true,
            OpKind::FieldWrite { field, .. } if field.name == "__pos_0" => true,
            OpKind::FieldRead { field, .. } if field.name == "__pos_0" => true,
            _ => false,
        })
}

/// A transparent newtype is its field's low-level value.
/// `StrKey(*mut Utf8Str)`, `BytesKey(*mut BytesBlock)`, and
/// `other::StrKey(*mut Utf8Str)` are `ValueType::Str`: construction and
/// `.0` are the identity (`AbstractTypedStrategy` keys by the unwrapped
/// rstr). A GC object pointer peels the same way: `ObjPtrKey(*mut Obj)`
/// is that `Ref`. `erased_str_ptr_newtype_annotates_somestring_on_construct_field_and_receiver`
/// covers the annotator side.
#[test]
fn str_family_dict_key_newtype_constructs_as_str() {
    let str_llbc = family_ptr_newtype_llbc(
        &["pyre_object", "unicodeobject", "Utf8Str"],
        &["pyre_object", "celldict", "StrKey"],
        "wrap_str_key",
        "unwrap_str_key",
    );
    let bytes_llbc = family_ptr_newtype_llbc(
        &["pyre_object", "bytesobject", "BytesBlock"],
        &["pyre_object", "dictmultiobject", "BytesKey"],
        "wrap_bytes_key",
        "unwrap_bytes_key",
    );
    let wrapper = wrapper_tyref();
    assert!(
        matches!(tyref_to_value_type(&wrapper, &str_llbc), ValueType::Str),
        "StrKey(ptr) is the inner STR, not a nominal instance"
    );
    assert!(
        matches!(tyref_to_value_type(&wrapper, &bytes_llbc), ValueType::Str),
        "BytesKey(ptr) is the inner STR, not a nominal instance"
    );

    let wrap_str =
        crate::test_support::lower_function(&str_llbc, "wrap_str_key").expect("lower StrKey wrap");
    let unwrap_str = crate::test_support::lower_function(&str_llbc, "unwrap_str_key")
        .expect("lower StrKey unwrap");
    let wrap_bytes = crate::test_support::lower_function(&bytes_llbc, "wrap_bytes_key")
        .expect("lower BytesKey wrap");
    let unwrap_bytes = crate::test_support::lower_function(&bytes_llbc, "unwrap_bytes_key")
        .expect("lower BytesKey unwrap");
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&wrap_str),
        "StrKey(ptr) must not instantiate + setattr(__pos_0); graph={}",
        wrap_str.dump()
    );
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&unwrap_str),
        "StrKey.0 must be identity; graph={}",
        unwrap_str.dump()
    );
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&wrap_bytes),
        "BytesKey(ptr) must not instantiate + setattr(__pos_0); graph={}",
        wrap_bytes.dump()
    );
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&unwrap_bytes),
        "BytesKey.0 must be identity; graph={}",
        unwrap_bytes.dump()
    );

    let lookalike = family_ptr_newtype_llbc(
        &["pyre_object", "unicodeobject", "Utf8Str"],
        &["other", "StrKey"],
        "wrap_lookalike",
        "unwrap_lookalike",
    );
    assert!(
        matches!(tyref_to_value_type(&wrapper, &lookalike), ValueType::Str),
        "other::StrKey(*mut Utf8Str) is the inner STR"
    );
    let wrap_lookalike = crate::test_support::lower_function(&lookalike, "wrap_lookalike")
        .expect("lower lookalike wrap");
    let unwrap_lookalike = crate::test_support::lower_function(&lookalike, "unwrap_lookalike")
        .expect("lower lookalike unwrap");
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&wrap_lookalike),
        "other::StrKey(ptr) is the STR pointer; graph={}",
        wrap_lookalike.dump()
    );
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&unwrap_lookalike),
        "other::StrKey.0 is the STR pointer; graph={}",
        unwrap_lookalike.dump()
    );

    // `ObjPtrKey(*mut Obj)` is the inner Ref: construction is identity.
    let obj_llbc = pointer_newtype_llbc(
        &["fixture", "Obj"],
        &["fixture", "ObjPtrKey"],
        "wrap_obj_ptr",
        "unwrap_obj_ptr",
        true,
    );
    let wrap_obj = crate::test_support::lower_function(&obj_llbc, "wrap_obj_ptr")
        .expect("lower ObjPtrKey wrap");
    let unwrap_obj = crate::test_support::lower_function(&obj_llbc, "unwrap_obj_ptr")
        .expect("lower ObjPtrKey unwrap");
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&wrap_obj),
        "ObjPtrKey(ptr) is the inner Ref, not a constructor; graph={}",
        wrap_obj.dump()
    );
    assert!(
        !graph_has_wrapper_ctor_or_pos0(&unwrap_obj),
        "ObjPtrKey.0 is identity; graph={}",
        unwrap_obj.dump()
    );
}

/// `shadow_stack_get(base) as *mut UnicodeValueStorage` in the real
/// `function_new_impl` body must lower to `__cast_instance_intrinsic`
/// with a Utf8Str-family root and `result_ty == Str`
/// (`op_gc_restore_root` recasts to `Ptr(STR)`).
#[test]
fn function_new_impl_reload_emits_cast_instance_str() {
    let interp = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let Some(llbc) = crate::test_support::load_ullbc_if_present(interp) else {
        return;
    };
    let graph =
        crate::test_support::lower_function(llbc, "pyre_interpreter::function::function_new_impl")
            .expect("lower function_new_impl");
    let casts: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call { result_ty, .. } => crate::model::cast_instance_root(&op.kind)
                .map(|root| (root.to_string(), result_ty.clone())),
            _ => None,
        })
        .collect();
    assert!(
        casts
            .iter()
            .any(|(root, ty)| { is_str_family_class_root(root) && matches!(ty, ValueType::Str) }),
        "function_new_impl reload must dest-is-str; casts={casts:?}"
    );
}

mod slice_or_vec_container_root_spelling_tests {
    use crate::front::mir::slice_or_vec_container_root_spelling;

    #[test]
    fn slice_and_vec_spellings_are_list_residuals() {
        assert_eq!(slice_or_vec_container_root_spelling("&[u8]"), None);
        assert_eq!(
            slice_or_vec_container_root_spelling("&mut [u8]"),
            Some("&mut [u8]")
        );
        assert_eq!(
            slice_or_vec_container_root_spelling("*mut [u8]"),
            Some("*mut [u8]")
        );
        assert_eq!(
            slice_or_vec_container_root_spelling("*mut Vec<Utf8LocElem>"),
            Some("*mut Vec<Utf8LocElem>")
        );
        assert_eq!(
            slice_or_vec_container_root_spelling("Vec<u8>"),
            Some("Vec<u8>")
        );
        assert_eq!(
            slice_or_vec_container_root_spelling("*mut Vec<*mut u8>"),
            Some("*mut Vec<*mut u8>")
        );
        assert_eq!(
            slice_or_vec_container_root_spelling("Vec<*mut u8>"),
            Some("Vec<*mut u8>")
        );
        assert_eq!(
            slice_or_vec_container_root_spelling("*mut Vec<usize>"),
            Some("*mut Vec<usize>")
        );
        assert_eq!(slice_or_vec_container_root_spelling("&Engine"), None);
        assert_eq!(slice_or_vec_container_root_spelling("*mut PyObject"), None);
        assert_eq!(slice_or_vec_container_root_spelling("[u8; 4]"), None);
        assert_eq!(slice_or_vec_container_root_spelling("*mut [u8; 4]"), None);
        assert_eq!(slice_or_vec_container_root_spelling("&[u8; 4]"), None);
    }
}

mod f64_bitcast_path_tests {
    use crate::front::mir::path_ends_with_segments;

    #[test]
    fn f64_bit_casts_are_path_suffixes_not_substrings() {
        let cases = [
            (
                "from_bits_core",
                "core::f64::<Impl>::from_bits",
                "f64::<Impl>::from_bits",
                true,
            ),
            (
                "from_bits_std",
                "std::f64::<Impl>::from_bits",
                "f64::<Impl>::from_bits",
                true,
            ),
            (
                "from_bits_position",
                "rhai::grain::vm::jit::position_from_bits",
                "f64::<Impl>::from_bits",
                false,
            ),
            (
                "from_bits_step_flags",
                "rhai::grain::format::StepFlags::<Impl>::from_bits",
                "f64::<Impl>::from_bits",
                false,
            ),
            (
                "to_bits_core",
                "core::f64::<Impl>::to_bits",
                "f64::<Impl>::to_bits",
                true,
            ),
            (
                "to_bits_position",
                "some_host::position_to_bits",
                "f64::<Impl>::to_bits",
                false,
            ),
        ];
        for (name, path, suffix, want) in cases {
            assert_eq!(path_ends_with_segments(path, suffix), want, "case {name}");
        }
    }
}

fn literal_i64() -> TyRef {
    serde_json::from_value(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})).unwrap()
}

fn shared_i64() -> TyRef {
    serde_json::from_value(
            serde_json::json!({"Ref": ["Erased", {"Scalar": {"Integer": {"Signed": "I64"}}}, "Shared"]}),
        )
        .unwrap()
}

fn shared_of(pointee: serde_json::Value) -> TyRef {
    TyRef::Other(serde_json::json!({"Ref": ["Erased", pointee, "Shared"]}))
}

/// `PartialOrd<&B> for &A` formals are `&&T`. One-level
/// `tyref_scalar_pointee_value_type` declines a pointer-to-pointer;
/// `input_param_value_type` peels to the scalar so mergeinputargs
/// sees Integer ∪ Integer (or Float ∪ Float).
#[test]
fn input_param_peels_nested_ref_of_primitive_to_the_scalar() {
    use crate::model::ValueType;
    let llbc = fixture_llbc();
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let f64_ty = serde_json::json!({"Scalar": {"Float": "F64"}});
    let char_ty = serde_json::json!({"Scalar": "Char"});
    let one = shared_of(i64_ty.clone());
    let two = shared_of(serde_json::json!({"Ref": ["Erased", i64_ty.clone(), "Shared"]}));
    let two_f64 = shared_of(serde_json::json!({"Ref": ["Erased", f64_ty, "Shared"]}));
    let two_char = shared_of(serde_json::json!({"Ref": ["Erased", char_ty, "Shared"]}));
    let range = shared_of(serde_json::json!({"Adt": {"id": 0}}));
    let no_tombstones = std::collections::HashSet::new();
    assert_eq!(
        super::input_param_value_type(
            ValueType::Ref(None),
            &one,
            &llbc,
            false,
            false,
            &no_tombstones
        ),
        ValueType::Int
    );
    assert_eq!(
        super::input_param_value_type(
            ValueType::Ref(None),
            &two,
            &llbc,
            false,
            false,
            &no_tombstones
        ),
        ValueType::Int
    );
    assert_eq!(
        super::input_param_value_type(
            ValueType::Ref(None),
            &two_f64,
            &llbc,
            false,
            false,
            &no_tombstones
        ),
        ValueType::Float
    );
    assert_eq!(
        super::input_param_value_type(
            ValueType::Ref(None),
            &two_char,
            &llbc,
            false,
            false,
            &no_tombstones
        ),
        ValueType::Int
    );
    assert_eq!(
        super::input_param_value_type(
            ValueType::Ref(None),
            &range,
            &llbc,
            false,
            false,
            &no_tombstones
        ),
        ValueType::Ref(None),
        "&Range stays a reference"
    );
    assert_eq!(
        super::tyref_scalar_pointee_value_type(&two, &llbc),
        None,
        "call-result banking still declines &&i64"
    );
}

#[test]
fn tyref_char_class_root_follows_nested_refs() {
    let llbc = fixture_llbc();
    let char_ty = serde_json::json!({"Scalar": "Char"});
    let by_value = TyRef::Other(char_ty.clone());
    let one = shared_of(char_ty.clone());
    let two = shared_of(serde_json::json!({"Ref": ["Erased", char_ty, "Shared"]}));
    let i64 = TyRef::Other(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}));
    assert_eq!(
        super::tyref_char_class_root(&by_value, &llbc).as_deref(),
        Some("char")
    );
    assert_eq!(
        super::tyref_char_class_root(&one, &llbc).as_deref(),
        Some("char")
    );
    assert_eq!(
        super::tyref_char_class_root(&two, &llbc).as_deref(),
        Some("char")
    );
    assert_eq!(super::tyref_char_class_root(&i64, &llbc), None);
}

/// A Ref pointee is `{"Deduplicated": id}` of the Scalar
/// (`tyref_ref_pointee_value_type`), not `TyRef::Dedup`. The
/// primitive match must follow that wrapper so a `&[i64]` iterator
/// item is Signed and `retype_word` emits `intmask` (`intmask` /
/// `cast_uint_to_int`) after `ll_slice_getitem_fast_i`.
#[test]
fn nested_dedup_i64_pointee_classifies_as_signed() {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Seed", 0]}],
                    "span": {"data": {
                        "file_id": 0,
                        "beg": {"line": 1, "col": 0},
                        "end": {"line": 1, "col": 1}
                    }},
                    "source_text": null,
                    "attr_info": {
                        "attributes": [], "inline": null, "rename": null, "public": true
                    },
                    "is_local": true
                },
                "kind": {"Struct": [{
                    "name": null,
                    "ty": {"Value": [7, {"Scalar": {"Integer": {"Signed": "I64"}}}]},
                    "attr_info": null
                }]},
                "layout": null
            }],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let pointee = TyRef::Other(serde_json::json!({"Deduplicated": 7}));
    assert_eq!(tyref_to_value_type(&pointee, &llbc), ValueType::Int);
    assert_eq!(tyref_to_attr_value_type(&pointee, &llbc), ValueType::Int);
    let borrowed = TyRef::Other(serde_json::json!({
        "Ref": ["Erased", {"Deduplicated": 7}, "Shared"]
    }));
    assert_eq!(
        tyref_to_value_type(&borrowed, &llbc),
        ValueType::Ref(None),
        "&i64 stays a reference; only its pointee is Signed"
    );
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert_eq!(
        super::tyref_ref_pointee_value_type(
            &borrowed,
            &llbc,
            super::no_tombstoned_leaves(),
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        Some(ValueType::Int)
    );
}

/// `&i64` and `i64` are different banks. `slice::Iter<i64>`'s type
/// argument is the scalar; the closure reads the borrow.
#[test]
fn map_collect_slice_iter_records_the_borrow_the_closure_reads() {
    let llbc = empty_llbc();
    let scalar = literal_i64();
    let borrow = shared_i64();
    assert_eq!(tyref_to_value_type(&scalar, &llbc), ValueType::Int);
    assert_eq!(
        tyref_to_value_type(&borrow, &llbc),
        ValueType::Ref(None),
        "&i64 and i64 must not share a bank"
    );
    assert!(super::iterator_adds_a_reference("core::slice::iter::Iter"));
    assert_eq!(
        super::map_collect_payload_value_type(&scalar, &llbc, true),
        ValueType::Ref(None),
        "Map::collect must record Iterator::Item &i64, not the type argument i64"
    );
}

/// `Enumerate<slice::Iter<i64>>::next` yields `(usize, &i64)`; the
/// `(index, item)` tuple of `enumerate(lst)` holds the list's item repr
/// (`EnumerateIteratorRepr.rtype_next` converts to `items_r[1]`), so the
/// reference the slice iterator added is peeled exactly as for a plain
/// `next`.
#[test]
fn enumerate_slice_iter_next_records_the_list_item() {
    let llbc = empty_llbc();
    let borrow = shared_i64();
    let node = match &borrow {
        TyRef::Other(v) => v.clone(),
        other => panic!("fixture ref must be inline, got {other:?}"),
    };
    assert_eq!(tyref_to_value_type(&borrow, &llbc), ValueType::Ref(None));
    let peeled = super::iterator_payload_element(&node, &llbc, true)
        .and_then(|item| serde_json::from_value::<TyRef>(item.clone()).ok())
        .map(|ty| tyref_to_value_type(&ty, &llbc));
    assert_eq!(peeled, Some(ValueType::Int));
}

/// Opaque `core::cell::Ref` / `RefMut` / `RefCell` type as generic 0.
/// RPython has no RefCell; the borrow flag has no lltype counterpart.
#[test]
fn std_cell_ref_wrappers_type_as_generic0() {
    let llbc = llbc_with_types(
        "core",
        vec![
            opaque_type_decl(0, &["core", "cell", "Ref"]),
            opaque_type_decl(1, &["core", "cell", "RefMut"]),
            opaque_type_decl(2, &["core", "cell", "RefCell"]),
            type_decl_struct(3, &["alloc", "vec", "Vec"], vec![]),
            opaque_type_decl(4, &["my_crate", "cell", "Ref"]),
        ],
        vec![],
    );
    let i64 = i64_ty();
    let vec_i64 = named_adt(3, vec![i64.clone()]);
    let ref_i64 = TyRef::Other(named_adt(0, vec![i64.clone()]));
    assert_eq!(
        tyref_to_value_type(&ref_i64, &llbc),
        ValueType::Int,
        "Ref<i64> occupies i64's bank"
    );
    let refmut_i64 = TyRef::Other(named_adt(1, vec![i64.clone()]));
    assert_eq!(tyref_to_value_type(&refmut_i64, &llbc), ValueType::Int);
    let refcell_i64 = TyRef::Other(named_adt(2, vec![i64.clone()]));
    assert_eq!(tyref_to_value_type(&refcell_i64, &llbc), ValueType::Int);

    let ref_vec = TyRef::Other(named_adt(0, vec![vec_i64.clone()]));
    assert_eq!(
        tyref_to_value_type(&ref_vec, &llbc),
        ValueType::Int,
        "Ref<Vec<i64>> occupies the RustVec int bank"
    );
    let refcell_vec = TyRef::Other(named_adt(2, vec![vec_i64]));
    assert_eq!(tyref_to_value_type(&refcell_vec, &llbc), ValueType::Int);

    let borrowed_ref = TyRef::Other(serde_json::json!({
        "Ref": ["Erased", named_adt(0, vec![i64.clone()]), "Shared"]
    }));
    assert_eq!(
        tyref_to_value_type(&borrowed_ref, &llbc),
        ValueType::Int,
        "&Ref<i64> peels to the same bank as Ref<i64>"
    );

    let user = TyRef::Other(named_adt(4, vec![i64]));
    assert_eq!(
        tyref_to_value_type(&user, &llbc),
        ValueType::Ref(None),
        "a user cell::Ref is not the std wrapper"
    );
}

#[test]
fn fn_ptr_value_type_is_int_and_input_class_root_is_fn() {
    let llbc = fixture_llbc();
    let ty = TyRef::Other(serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [{"Slice": [{"Adt": {"id": 0}}]}],
                "output": {"Scalar": {"Integer": {"Signed": "I64"}}},
                "is_unsafe": false
            }
        }
    }));
    assert_eq!(
        tyref_to_value_type(&ty, &llbc),
        ValueType::Int,
        "history.getkind of Ptr(FuncType) is int"
    );
    assert_eq!(
        tyref_fn_ptr_input_class_root(&ty, &llbc).as_deref(),
        Some("fn([??adt#0]) -> i64"),
        "Int-banked fn-pointer params carry the fn(inputs) -> output spelling"
    );
    assert_eq!(
        tyref_to_attr_value_type(&ty, &llbc),
        ValueType::Int,
        "a stored fn field uses the same Int bank as a value site"
    );
}

#[test]
fn stamp_fn_ptr_spelling_equals_fn_null_constant() {
    use crate::annotator::bookkeeper::{Bookkeeper, fn_ptr_somevalue_for_spelling_in};
    use crate::annotator::builtin::fn_null_constant;
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::{ConstValue, Constant, Variable};
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    use std::collections::HashMap;
    use std::rc::Rc;

    let spelling = "fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>";
    let var = Variable::named("fnptr");
    super::stamp_fn_ptr_spelling(&var, spelling, crate::ErrorCarrierSpec::default());
    let bk = Rc::new(Bookkeeper::new());
    let s_ll = fn_ptr_somevalue_for_spelling_in(spelling, Some(bk.as_ref())).expect("fn spelling");
    let SomeValue::Ptr(p_ll) = &s_ll else {
        panic!("fn_ptr_somevalue_for_spelling_in is the lltype helper, got {s_ll:?}");
    };
    let mut s_ty = crate::annotator::model::s_str0();
    s_ty.set_const_box(Constant::new(ConstValue::byte_str(spelling)));
    let s_null = fn_null_constant(&bk, &[Some(s_ty)], &HashMap::new()).expect("typed null_fn");
    assert!(
        matches!(s_null, SomeValue::None_(_)),
        "spelled null_fn annotates as s_None, got {s_null:?}"
    );
    let ct = var.concretetype().expect("dest stamped");
    let LowLevelType::Ptr(ptr_ty) = ct else {
        panic!("stamp must be Ptr, got {ct:?}");
    };
    assert!(
        matches!(ptr_ty.TO, PtrTarget::Func(_)),
        "stamp is Ptr(FuncType)"
    );
    assert_eq!(
        *ptr_ty, p_ll.ll_ptrtype,
        "stamp_fn_ptr equals func_type_from_fn_spelling of the same spelling"
    );

    let s_untyped = fn_null_constant(&bk, &[], &HashMap::new()).expect("untyped null_fn");
    assert!(
        matches!(s_untyped, SomeValue::Ptr(_)),
        "untyped ptr::null_fn stays SomePtr, got {s_untyped:?}"
    );
}

#[test]
fn stamp_fn_ptr_spelling_carrier_result_is_void() {
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::builtin::fn_null_constant;
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::{ConstValue, Constant, Variable};
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    use std::collections::HashMap;
    use std::rc::Rc;

    let spelling = "fn(PyObjectRef, *const PyExecutionContext) -> Result<(), PyError>";
    let spec = crate::ErrorCarrierSpec {
        carrier_path: "error::PyError",
        carrier_class: "error::PyErrorObject",
        carrier_wrappers: &[],
        to_exc_object: None,
        from_exc_object: None,
    };
    let var = Variable::named("startup");
    super::stamp_fn_ptr_spelling(&var, spelling, spec);
    let bk = Rc::new(Bookkeeper::new());
    bk.set_exception_carrier("error::PyErrorObject");
    bk.alias_exception_carrier_handle("error::PyError");
    let mut s_ty = crate::annotator::model::s_str0();
    s_ty.set_const_box(Constant::new(ConstValue::byte_str(spelling)));
    let s_null = fn_null_constant(&bk, &[Some(s_ty)], &HashMap::new()).expect("typed null_fn");
    assert!(
        matches!(s_null, SomeValue::None_(_)),
        "spelled null_fn annotates as s_None, got {s_null:?}"
    );
    let s_ll =
        crate::annotator::bookkeeper::fn_ptr_somevalue_for_spelling_in(spelling, Some(bk.as_ref()))
            .expect("fn spelling");
    let SomeValue::Ptr(p_ll) = &s_ll else {
        panic!("lltype helper is SomePtr, got {s_ll:?}");
    };
    let ct = var.concretetype().expect("dest stamped");
    let LowLevelType::Ptr(ptr_ty) = ct else {
        panic!("stamp must be Ptr, got {ct:?}");
    };
    let PtrTarget::Func(ft) = &ptr_ty.TO else {
        panic!("stamp is Ptr(FuncType)");
    };
    assert_eq!(
        ft.result,
        LowLevelType::Void,
        "carrier Result<(), E> is Void (`fn_ptr_result_lltype`)"
    );
    assert_eq!(
        *ptr_ty, p_ll.ll_ptrtype,
        "stamp_fn_ptr with the carrier equals func_type_from_fn_spelling"
    );
}

#[test]
fn stamp_fn_ptr_spelling_equals_fn_null_constant_for_carrier_results() {
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::builtin::fn_null_constant;
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::{ConstValue, Constant, Variable};
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    use std::collections::HashMap;
    use std::rc::Rc;

    let spec = crate::ErrorCarrierSpec {
        carrier_path: "error::PyError",
        carrier_class: "error::PyErrorObject",
        carrier_wrappers: &[],
        to_exc_object: None,
        from_exc_object: None,
    };
    let bk = Rc::new(Bookkeeper::new());
    bk.set_exception_carrier(spec.classdef_path());
    if !spec.carrier_class.is_empty() {
        bk.alias_exception_carrier_handle(spec.carrier_path);
    }
    for spelling in [
        "fn() -> Result<(), PyError>",
        "fn() -> Result<PyObjectRef, PyError>",
    ] {
        let var = Variable::named("fnptr");
        super::stamp_fn_ptr_spelling(&var, spelling, spec);
        let mut s_ty = crate::annotator::model::s_str0();
        s_ty.set_const_box(Constant::new(ConstValue::byte_str(spelling)));
        let s_null = fn_null_constant(&bk, &[Some(s_ty)], &HashMap::new())
            .unwrap_or_else(|e| panic!("{spelling}: typed null_fn: {e}"));
        assert!(
            matches!(s_null, SomeValue::None_(_)),
            "{spelling}: spelled null_fn is s_None, got {s_null:?}"
        );
        let s_ll = crate::annotator::bookkeeper::fn_ptr_somevalue_for_spelling_in(
            spelling,
            Some(bk.as_ref()),
        )
        .unwrap_or_else(|| panic!("{spelling}: fn spelling"));
        let SomeValue::Ptr(p_ll) = &s_ll else {
            panic!("{spelling}: lltype helper is SomePtr, got {s_ll:?}");
        };
        let ct = var
            .concretetype()
            .unwrap_or_else(|| panic!("{spelling}: dest stamped"));
        let LowLevelType::Ptr(ptr_ty) = ct else {
            panic!("{spelling}: stamp must be Ptr, got {ct:?}");
        };
        assert!(
            matches!(ptr_ty.TO, PtrTarget::Func(_)),
            "{spelling}: stamp is Ptr(FuncType)"
        );
        assert_eq!(
            *ptr_ty, p_ll.ll_ptrtype,
            "{spelling}: stamp_fn_ptr equals func_type_from_fn_spelling of the same spec"
        );
    }
}

#[test]
fn type_alias_of_fn_ptr_peels_to_functype() {
    use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
    use crate::translator::rtyper::rclass::OBJECTPTR;

    let fnptr = serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [{
                    "RawPtr": [{"Adt": {"id": 1, "generics": empty_generics()}}, "Mut"]
                }],
                "output": {
                    "RawPtr": [{"Adt": {"id": 1, "generics": empty_generics()}}, "Mut"]
                },
                "is_unsafe": false
            }
        }
    });
    let llbc = llbc_with_types(
        "fixture",
        vec![
            type_decl_alias(
                0,
                &["fixture", "HookFn"],
                serde_json::json!({"aliased_ty": fnptr}),
            ),
            type_decl_struct(1, &["fixture", "PyObject"], vec![]),
        ],
        vec![],
    );
    let ty = TyRef::Other(serde_json::json!({
        "Adt": {"id": 0, "generics": empty_generics()}
    }));
    assert!(
        super::type_node_is_fn_ptr(
            super::tyref_node(&ty, &llbc).expect("alias Adt node"),
            &llbc
        ),
        "TypeDeclKind::Alias {{aliased_ty: FnPtr}} is a function pointer"
    );
    let ft = super::func_type_from_fn_ptr_ty(&ty, &llbc, crate::ErrorCarrierSpec::default())
        .expect("alias FnPtr lowers to FuncType");
    assert_eq!(ft.args, vec![OBJECTPTR.clone()]);
    assert_eq!(ft.result, *OBJECTPTR);

    let var = crate::flowspace::model::Variable::named("hook");
    super::stamp_fn_ptr(&var, &ty, &llbc, crate::ErrorCarrierSpec::default());
    let ct = var.concretetype().expect("dest stamped");
    let crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Ptr(ptr_ty) = ct else {
        panic!("stamp must be Ptr, got {ct:?}");
    };
    assert!(
        matches!(ptr_ty.TO, PtrTarget::Func(_)),
        "stamp is Ptr(FuncType)"
    );
}

#[test]
fn fn_ptr_str_family_pointer_is_strptr_not_objectptr() {
    use crate::translator::rtyper::lltypesystem::rstr::STRPTR;
    use crate::translator::rtyper::rclass::OBJECTPTR;

    let utf8 = type_decl_struct(0, &["unicodeobject", "Utf8Str"], vec![]);
    let inst = type_decl_struct(1, &["owner", "Inst"], vec![]);
    let other = type_decl_struct(2, &["other", "Utf8Str"], vec![]);
    let llbc = llbc_with_types("fixture", vec![utf8, inst, other], vec![]);
    let raw = |id: u64| {
        serde_json::json!({
            "RawPtr": [{"Adt": {"id": id, "generics": empty_generics()}}, "Const"]
        })
    };
    let fnptr = |id: u64| {
        TyRef::Other(serde_json::json!({
            "FnPtr": {
                "skip_binder": {
                    "inputs": [raw(id)],
                    "output": {"Tuple": []},
                    "is_unsafe": false
                }
            }
        }))
    };
    let spec = crate::ErrorCarrierSpec::default();
    let str_ft = super::func_type_from_fn_ptr_ty(&fnptr(0), &llbc, spec).expect("Utf8Str fn-ptr");
    assert_eq!(
        str_ft.args,
        vec![STRPTR.clone()],
        "FUNC.ARGS of *const unicodeobject::Utf8Str is STRPTR; ast={}",
        super::tyref_to_ast_string(&TyRef::Other(raw(0)), &llbc)
    );
    let inst_ft = super::func_type_from_fn_ptr_ty(&fnptr(1), &llbc, spec).expect("Inst fn-ptr");
    assert_eq!(inst_ft.args, vec![OBJECTPTR.clone()]);
    let other_ty = TyRef::Other(raw(2));
    let other_ft =
        super::func_type_from_fn_ptr_ty(&fnptr(2), &llbc, spec).expect("other::Utf8Str fn-ptr");
    assert!(
        !super::tyref_raw_ptr_pointee_is_string_value(&other_ty, &llbc),
        "other::Utf8Str is not the STR layout; intern ast={}",
        super::tyref_to_ast_string(&other_ty, &llbc)
    );
    assert_eq!(
        other_ft.args,
        vec![OBJECTPTR.clone()],
        "other::Utf8Str is not the STR layout; intern ast={}",
        super::tyref_to_ast_string(&other_ty, &llbc)
    );
    let str_ty = super::fn_ptr_tyref_to_lltype(&TyRef::Other(raw(0)), &llbc, spec);
    assert_eq!(str_ty, *STRPTR);
}

#[test]
fn hkey_alias_of_c_void_ptr_is_voidp() {
    let c_void_adt = serde_json::json!({
        "Adt": {"id": 0, "generics": empty_generics()}
    });
    let voidp = serde_json::json!({"RawPtr": [c_void_adt, "Mut"]});
    let llbc = llbc_with_types(
        "fixture",
        vec![
            serde_json::json!({
                "def_id": 0,
                "item_meta": fixture_item_meta(ident_path(&["core", "ffi", "c_void"])),
                "kind": "Opaque"
            }),
            type_decl_alias(1, &["fixture", "HKEY"], voidp.clone()),
            type_decl_alias(
                2,
                &["fixture", "VOIDPP"],
                serde_json::json!({"RawPtr": [voidp, "Mut"]}),
            ),
            type_decl_struct(3, &["majit_gc", "header", "GCREFOpaque"], vec![]),
            type_decl_alias(
                4,
                &["majit_gc", "header", "GCREF"],
                serde_json::json!({"RawPtr": [{
                    "Adt": {"id": 3, "generics": empty_generics()}
                }, "Mut"]}),
            ),
        ],
        vec![],
    );
    let hkey = TyRef::Other(serde_json::json!({
        "Adt": {"id": 1, "generics": empty_generics()}
    }));
    let voidpp = TyRef::Other(serde_json::json!({
        "Adt": {"id": 2, "generics": empty_generics()}
    }));
    let gcref = TyRef::Other(serde_json::json!({
        "Adt": {"id": 4, "generics": empty_generics()}
    }));
    assert!(
        tyref_is_raw_c_void_ptr(&hkey, &llbc),
        "type HKEY = *mut c_void is VOIDP"
    );
    assert!(
        tyref_is_raw_c_void_pp(&voidpp, &llbc),
        "type VOIDPP = *mut *mut c_void is VOIDPP"
    );
    assert!(
        !tyref_is_raw_c_void_ptr(&gcref, &llbc),
        "type GCREF = *mut GCREFOpaque is not VOIDP"
    );
    assert!(
        !tyref_is_raw_c_void_pp(&gcref, &llbc),
        "GCREF is not VOIDPP"
    );
}

/// Charon dests intern `*mut c_void` as `Deduplicated` (`prepare_file`
/// `null_mut` dest). The intern body is `RawPtr` of `c_void`; VOIDP
/// recognition has to resolve that table, not `type_decls[id]`.
#[test]
fn interned_c_void_ptr_dest_is_voidp() {
    let intern_ptr = 1u64;
    let intern_void = 2u64;
    let llbc = llbc_with_types(
        "fixture",
        vec![
            serde_json::json!({
                "def_id": 0,
                "item_meta": fixture_item_meta(ident_path(&["core", "ffi", "c_void"])),
                "kind": "Opaque"
            }),
            type_decl_alias(
                1,
                &["fixture", "InternSeed"],
                serde_json::json!({
                    "Value": [intern_ptr, {
                        "RawPtr": [
                            {"Value": [intern_void, {
                                "Adt": {
                                    "id": 0,
                                    "generics": empty_generics(),
                                    "builtin": null
                                }
                            }]},
                            "Mut"
                        ]
                    }]
                }),
            ),
        ],
        vec![],
    );
    let ty = TyRef::Dedup { id: intern_ptr };
    assert!(
        tyref_is_raw_c_void_ptr(&ty, &llbc),
        "Deduplicated *mut c_void intern is VOIDP"
    );
    assert!(
        !tyref_is_raw_c_void_pp(&ty, &llbc),
        "Deduplicated *mut c_void intern is not VOIDPP"
    );
    assert_eq!(
        tyref_enum_payload_value_type(
            &ty,
            &llbc,
            &no_tombstoned_leaves(),
            &std::collections::HashSet::new(),
            &std::collections::HashSet::new(),
        ),
        ValueType::Int,
        "Result Ok *mut c_void is VOIDP int-bank, not GCREF"
    );
}

/// `Option<*mut *mut c_void>` is tagged: the inner VOIDP word may already
/// be null, so rustc keeps a discriminant. `Option<*mut PyObjectRef>`
/// stays the GC pointer niche.
#[test]
fn option_of_voidpp_is_not_pointer_niche() {
    let c_void_adt = serde_json::json!({
        "Adt": {"id": 1, "generics": empty_generics()}
    });
    let voidp = serde_json::json!({"RawPtr": [c_void_adt, "Mut"]});
    let voidpp = serde_json::json!({"RawPtr": [voidp.clone(), "Mut"]});
    let gc_obj_adt = serde_json::json!({
        "Adt": {"id": 2, "generics": empty_generics()}
    });
    let gc_obj_ptr = serde_json::json!({"RawPtr": [gc_obj_adt, "Mut"]});
    let gc_obj_ptr_ptr = serde_json::json!({"RawPtr": [gc_obj_ptr, "Mut"]});
    let hkey = serde_json::json!({
        "Adt": {"id": 3, "generics": empty_generics()}
    });
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            serde_json::json!({
                "def_id": 1,
                "item_meta": fixture_item_meta(ident_path(&["core", "ffi", "c_void"])),
                "kind": "Opaque"
            }),
            type_decl_struct(2, &["fixture", "GcObj"], vec![]),
            type_decl_alias(3, &["fixture", "HKEY"], voidp),
        ],
        vec![],
    );
    let option_voidpp = TyRef::Other(option_adt_node(0, voidpp));
    let option_gc_obj_ptr_ptr = TyRef::Other(option_adt_node(0, gc_obj_ptr_ptr));
    let option_phkey = TyRef::Other(option_adt_node(
        0,
        serde_json::json!({"RawPtr": [hkey, "Mut"]}),
    ));
    assert!(
        !super::tyref_is_niche_option(&option_voidpp, &llbc),
        "Option<*mut *mut c_void> keeps the tag"
    );
    assert!(
        super::tyref_niche_option_payload_paint(&option_voidpp, &llbc, &no_tombstoned_leaves())
            .is_none(),
        "tagged VOIDPP Option is not Int-painted as a VOIDP niche"
    );
    assert!(
        super::tyref_is_niche_option(&option_gc_obj_ptr_ptr, &llbc),
        "Option<*mut *mut GcObj> stays the GC pointer niche"
    );
    assert!(
        !super::tyref_is_niche_option(&option_phkey, &llbc),
        "Option<*mut HKEY> is Option of VOIDPP via alias"
    );
}

#[test]
fn fn_ptr_ty_keeps_gcref_result() {
    use crate::translator::rtyper::lltypesystem::lltype::{GCREF, LowLevelType};

    let gcrefopaque_adt = serde_json::json!({
        "Adt": {"id": 0, "generics": empty_generics()}
    });
    let gcref_adt = serde_json::json!({
        "Adt": {"id": 3, "generics": empty_generics()}
    });
    let llbc = llbc_with_types(
        "fixture",
        vec![
            type_decl_struct(0, &["majit_gc", "header", "GCREFOpaque"], vec![]),
            type_decl_struct(1, &["fixture", "PyObject"], vec![]),
            type_decl_struct(2, &["alloc", "vec", "Vec"], vec![]),
            type_decl_alias(
                3,
                &["majit_gc", "header", "GCREF"],
                serde_json::json!({"RawPtr": [gcrefopaque_adt.clone(), "Mut"]}),
            ),
        ],
        vec![],
    );
    let ty = TyRef::Other(serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [
                    {"Scalar": {"Integer": {"Unsigned": "U32"}}},
                    {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
                    {"RawPtr": [gcref_adt.clone(), "Mut"]},
                    {"RawPtr": [{"Scalar": "Bool"}, "Mut"]}
                ],
                "output": gcref_adt,
                "is_unsafe": true
            }
        }
    }));
    let ft = super::func_type_from_fn_ptr_ty(&ty, &llbc, crate::ErrorCarrierSpec::default())
        .expect("hook FnPtr lowers");
    assert_eq!(ft.args.len(), 4);
    assert_eq!(ft.args[0], LowLevelType::Unsigned);
    assert_eq!(ft.args[1], LowLevelType::Unsigned);
    assert_eq!(
        ft.args[2], *GCREF,
        "*mut GCREF FUNC arg is the GCREF root slot, not OBJECTPTR"
    );
    assert_eq!(
        ft.args[3],
        LowLevelType::Signed,
        "*mut bool FUNC arg is the slot address, not OBJECTPTR"
    );
    assert_eq!(
        ft.result, *GCREF,
        "GCREF FUNC.RESULT stays GCREF, not OBJECTPTR"
    );

    let opaque_out = TyRef::Other(serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [],
                "output": gcrefopaque_adt,
                "is_unsafe": false
            }
        }
    }));
    let ft_opaque =
        super::func_type_from_fn_ptr_ty(&opaque_out, &llbc, crate::ErrorCarrierSpec::default())
            .expect("GCREFOpaque result FnPtr lowers");
    assert_eq!(
        ft_opaque.result, *GCREF,
        "bare GCREFOpaque FUNC.RESULT is GCREF"
    );

    // `AllocNurseryCollectingTypedRootedFn`: `*mut GcRef` / `GcRef`
    // (`majit_ir::GcRef`) is the same GCREF word.
    let gcref_word = serde_json::json!({
        "Adt": {"id": 4, "generics": empty_generics()}
    });
    let llbc_gcref = llbc_with_types(
        "fixture",
        vec![
            type_decl_struct(0, &["majit_gc", "header", "GCREFOpaque"], vec![]),
            type_decl_struct(1, &["fixture", "PyObject"], vec![]),
            type_decl_struct(2, &["alloc", "vec", "Vec"], vec![]),
            type_decl_alias(
                3,
                &["majit_gc", "header", "GCREF"],
                serde_json::json!({"RawPtr": [gcrefopaque_adt.clone(), "Mut"]}),
            ),
            type_decl_struct(4, &["majit_ir", "GcRef"], vec![]),
        ],
        vec![],
    );
    let ty_gcref = TyRef::Other(serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [
                    {"Scalar": {"Integer": {"Unsigned": "U32"}}},
                    {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
                    {"RawPtr": [gcref_word.clone(), "Mut"]},
                    {"RawPtr": [{"Scalar": "Bool"}, "Mut"]}
                ],
                "output": gcref_word,
                "is_unsafe": true
            }
        }
    }));
    let ft_gcref =
        super::func_type_from_fn_ptr_ty(&ty_gcref, &llbc_gcref, crate::ErrorCarrierSpec::default())
            .expect("GcRef hook FnPtr lowers");
    assert_eq!(ft_gcref.args.len(), 4);
    assert_eq!(
        ft_gcref.args[2], *GCREF,
        "*mut GcRef FUNC arg is the GCREF root slot, not OBJECTPTR"
    );
    assert_eq!(
        ft_gcref.args[3],
        LowLevelType::Signed,
        "*mut bool FUNC arg is the slot address, not OBJECTPTR"
    );
    assert_eq!(
        ft_gcref.result, *GCREF,
        "GcRef FUNC.RESULT is GCREF (`llmemory.py` GCREF)"
    );
}

#[test]
fn function_pointer_uses_the_raw_pointer_int_bank() {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [], "fun_decls": [], "global_decls": [],
            "trait_decls": [], "trait_impls": [],
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let ty = TyRef::Other(serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [],
                "output": {"Scalar": {"Integer": {"Signed": "I64"}}},
                "is_unsafe": false
            }
        }
    }));

    assert_eq!(
        tyref_to_value_type(&ty, &llbc),
        ValueType::Int,
        "RPython history.getkind(Ptr(FuncType)) uses the int bank"
    );
}

#[test]
fn a_tuple_param_names_the_shape_its_reads_name() {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [], "fun_decls": [], "global_decls": [],
            "trait_decls": [], "trait_impls": [],
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let tuple = serde_json::json!({
        "Adt": {
            "id": 0, "builtin": "Tuple",
            "generics": {
                "types": [
                    { "Scalar": { "Integer": { "Signed": "I64" } } },
                    { "Scalar": "Bool" }
                ]
            }
        }
    });
    let owner = format!(
        "Tuple{}",
        tyref_tuple_suffix(&TyRef::Other(tuple.clone()), &llbc)
    );
    assert!(majit_ir::descr::is_shaped_tuple_name(&owner), "{owner}");
    // By value (a closure's args tuple) and behind a borrow (the read
    // place of `(*p).N` is the deref'd tuple) both name the read owner.
    let by_value = TyRef::Other(tuple.clone());
    let borrowed = TyRef::Other(serde_json::json!({ "Ref": ["Erased", tuple, "Shared"] }));
    assert_eq!(
        tyref_shaped_tuple_root(&by_value, &llbc),
        Some(owner.clone())
    );
    assert_eq!(
        tyref_shaped_tuple_root(&borrowed, &llbc),
        Some(owner.clone())
    );
    let unit = TyRef::Other(serde_json::json!({
        "Adt": { "id": 0, "builtin": "Tuple", "generics": { "types": [] } }
    }));
    assert_eq!(tyref_shaped_tuple_root(&unit, &llbc), None);
    let int = TyRef::Other(serde_json::json!({ "Scalar": { "Integer": { "Signed": "I64" } } }));
    assert_eq!(tyref_shaped_tuple_root(&int, &llbc), None);
    // The TypeDecl leaf of a builtin Tuple is `"Tuple"`. Input
    // class_root must keep the shaped name so `call_once` args
    // seed `SomeTuple` (`rtuple.py` `TupleRepr`).
    assert_eq!(
        tyref_ref_param_class_root(&by_value, &llbc, no_tombstoned_leaves(), None),
        Some(owner)
    );

    // A `call_once` `tupled_args` is a hash-consed 1-tuple of `fn`.
    // The TypeDecl leaf is still `"Tuple"`; the Input class_root
    // must be the shaped `Tuple<fn(...)>` so `getitem(0)` is
    // `SomePtr` (`binaryop.py` `getitem((tup1, int2))`).
    let fn_tuple = r#"{
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [{
                    "def_id": 0,
                    "item_meta": {
                        "name": [{"Builtin": [{"Tuple": 1}, 0]}],
                        "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}},
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": false
                    },
                    "kind": {"Struct": []}
                }, {
                    "def_id": 1,
                    "item_meta": {
                        "name": [{"Ident": ["majit_gc", 0]}, {"Ident": ["header", 0]}, {"Ident": ["GCREFOpaque", 0]}],
                        "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}},
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": false
                    },
                    "kind": "Opaque"
                }],
                "fun_decls": [{
                    "def_id": 0,
                    "item_meta": {
                        "name": [{"Ident": ["fixture", 0]}, {"Ident": ["call_once", 0]}],
                        "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}},
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [{"Value": [3456, {"Adt": {
                            "id": 0,
                            "builtin": "Tuple",
                            "generics": {
                                "regions": [],
                                "types": [{"Value": [287, {"FnPtr": {
                                    "regions": [],
                                    "skip_binder": {
                                        "is_unsafe": false,
                                        "abi": "Rust",
                                        "is_variadic": false,
                                        "inputs": [
                                            {"Scalar": {"Integer": {"Unsigned": "U32"}}},
                                            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
                                        ],
                                        "output": {"RawPtr": [
                                            {"Adt": {"id": 1, "generics": {
                                                "regions": [], "types": [],
                                                "const_generics": [], "trait_refs": []
                                            }, "builtin": null}},
                                            "Mut"
                                        ]}
                                    }
                                }}]}],
                                "const_generics": [],
                                "trait_refs": []
                            }
                        }}]}],
                        "output": {"Scalar": "Bool"}
                    },
                    "body": "Opaque"
                }],
                "global_decls": [], "trait_decls": [], "trait_impls": []
            }
        }"#;
    let fn_llbc =
        Llbc::from_slice(fn_tuple.as_bytes()).expect("fn-pointer tuple fixture Llbc parses");
    let tupled = TyRef::Dedup { id: 3456 };
    let shaped = tyref_shaped_tuple_root(&tupled, &fn_llbc)
        .expect("dedup Tuple<fn> must have a shaped root");
    assert!(
        majit_ir::descr::is_shaped_tuple_name(&shaped),
        "shaped root must not be the bare Tuple leaf: {shaped}"
    );
    assert!(
        shaped.contains("fn("),
        "shaped root must name the fn payload: {shaped}"
    );
    assert_eq!(
        tyref_ref_param_class_root(&tupled, &fn_llbc, no_tombstoned_leaves(), None),
        Some(shaped.clone())
    );
    let empty_gc: std::collections::HashSet<majit_ir::descr::StructId> =
        std::collections::HashSet::new();
    assert_eq!(
        dont_look_inside_return_token(
            &by_value,
            &llbc,
            crate::ErrorCarrierSpec::default(),
            &empty_gc,
            &std::collections::HashSet::<String>::new(),
        )
        .as_deref(),
        Some("Tuple<i64,bool>"),
        "a pair tuple must keep SomeTuple items, not collapse to ref"
    );

    // Construction keys `__pos_N` off `tyref_tuple_suffix` /
    // `tyref_positional_aggregate_root`. A hash-consed or `Value`-wrapped
    // destination must recover the same shaped owner the read side uses,
    // otherwise assemble asks `getattr(Tuple, __pos_0)`.
    assert!(
        !tyref_tuple_suffix(&tupled, &fn_llbc).is_empty(),
        "dedup Tuple<fn> construction suffix must not collapse to unit Tuple"
    );
    assert_eq!(
        tyref_positional_aggregate_root(&tupled, &fn_llbc),
        Some(shaped.clone())
    );
    let hashed = TyRef::Other(serde_json::json!({
        "HashConsedValue": [99, {
            "Adt": {
                "id": 0, "builtin": "Tuple",
                "generics": {
                    "types": [
                        { "Scalar": { "Integer": { "Signed": "I64" } } },
                        { "Scalar": "Bool" }
                    ]
                }
            }
        }]
    }));
    assert_eq!(tyref_tuple_suffix(&hashed, &llbc), "<i64,bool>");
    assert_eq!(
        tyref_positional_aggregate_root(&hashed, &llbc),
        Some("Tuple<i64,bool>".to_string())
    );
    let valued = TyRef::Other(serde_json::json!({
        "Value": [7, {
            "Adt": {
                "id": 0, "builtin": "Tuple",
                "generics": {
                    "types": [
                        { "Scalar": { "Integer": { "Signed": "I64" } } },
                        { "Scalar": "Bool" }
                    ]
                }
            }
        }]
    }));
    assert_eq!(tyref_tuple_suffix(&valued, &llbc), "<i64,bool>");
    assert_eq!(
        tyref_positional_aggregate_root(&valued, &llbc),
        Some("Tuple<i64,bool>".to_string())
    );
}

/// A one-word pointer to a harvested `_alloc_flavor_ = raw` ADT is
/// front `Ref(Some(root))` so the annotator seeds `SomeInstance`.
/// The int bank is `get_type_flag` / `valuetype_to_concrete_for_legacy`
/// (`history.getkind` of a raw `Ptr`). Storage-raw classification stays
/// false: a `Vec` field fails `adt_def_is_raw_storage`. An empty
/// flavor set keeps the residual token as `ref`.
#[test]
fn flavor_raw_struct_borrow_is_int_banked_with_class_root() {
    let span = fixture_span();
    let vec_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [{"Scalar": {"Integer": {"Unsigned": "Usize"}}}],
                "const_generics": [],
                "trait_refs": []
            },
            "builtin": null
        }
    });
    let owner = struct_decl(
        0,
        ident_path(&["fixture", "RawFlavorStruct"]),
        serde_json::json!([{
            "name": "items",
            "ty": vec_ty,
            "attr_info": null
        }]),
        false,
    );
    let vec_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": fixture_item_meta(ident_path(&["alloc", "vec", "Vec"])),
        "kind": "Opaque"
    });
    let ref_ty = serde_json::json!({
        "Ref": [
            "Erased",
            {"Adt": {
                "id": 0,
                "generics": {
                    "regions": [], "types": [],
                    "const_generics": [], "trait_refs": []
                },
                "builtin": null
            }},
            "Shared"
        ]
    });
    let unit_ty = serde_json::json!({
        "Adt": {"builtin": "Tuple", "generics": {"types": []}}
    });
    let flavor_global = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(
            ident_path(&["fixture", "_alloc_flavor_RawFlavorStruct"])
        ),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": "raw"}, {"Ref": [
            "Erased",
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Shared"
        ]}]]}
    });
    let inspect = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "inspect"])),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_ty],
            "output": unit_ty
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    {"index": 0, "name": null, "span": span, "ty": unit_ty},
                    {"index": 1, "name": "subject", "span": span, "ty": ref_ty}
                ]
            },
            "body": [{
                "statements": [],
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let get_ptr = serde_json::json!({
        "def_id": 1,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "get_ptr"])),
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": ref_ty
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [owner, vec_decl],
            "fun_decls": [inspect, get_ptr],
            "global_decls": [flavor_global],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let harvested =
        crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc));
    assert!(
        harvested.iter().any(|s| s.contains("RawFlavorStruct")),
        "harvested flavor set: {harvested:?}"
    );
    let empty: std::collections::HashSet<String> = std::collections::HashSet::new();
    let empty_gc: std::collections::HashSet<majit_ir::descr::StructId> =
        std::collections::HashSet::new();
    let inspect_fd = llbc.fn_by_id(0).expect("inspect");
    let param_ty = &inspect_fd.signature.inputs[0];
    assert!(
        !super::ty_is_raw_address(param_ty, &llbc, &empty_gc, &harvested),
        "flavor-raw &RawFlavorStruct is not storage-raw; ty_is_raw_address stays false"
    );
    assert!(
        !super::ty_is_raw_address(param_ty, &llbc, &empty_gc, &empty),
        "empty flavor set keeps ty_is_raw_address false"
    );
    let harvested_vt = super::tyref_to_value_type_with(
        param_ty,
        &llbc,
        super::no_tombstoned_leaves(),
        &empty_gc,
        &harvested,
    );
    assert!(
        matches!(
            &harvested_vt,
            ValueType::Ref(Some(root)) if root.contains("RawFlavorStruct")
        ),
        "flavor-raw &RawFlavorStruct is front Ref(Some(root)), got {harvested_vt:?}"
    );
    let empty_vt = super::tyref_to_value_type_with(
        param_ty,
        &llbc,
        super::no_tombstoned_leaves(),
        &empty_gc,
        &empty,
    );
    assert!(
        matches!(empty_vt, ValueType::Ref(_)),
        "empty flavor set keeps the front type Ref, got {empty_vt:?}"
    );
    let word = crate::layout::target_word_size();
    let owner = harvested
        .iter()
        .find(|s| s.contains("RawFlavorStruct"))
        .expect("harvested intern key");
    let spelling = format!("&{owner}");
    let (flag, bank, size) = crate::codewriter::call::get_type_flag(&spelling, &harvested);
    assert_eq!(flag, majit_ir::descr::ArrayFlag::Unsigned);
    assert_eq!(bank, majit_ir::value::Type::Int);
    assert_eq!(size, word);
    let (empty_flag, empty_bank, _) = crate::codewriter::call::get_type_flag(&spelling, &empty);
    assert_eq!(empty_flag, majit_ir::descr::ArrayFlag::Pointer);
    assert_eq!(empty_bank, majit_ir::value::Type::Ref);
    let spec = crate::ErrorCarrierSpec::default();
    let get_ptr_fd = llbc.fn_by_id(1).expect("get_ptr");
    let harvested_token = dont_look_inside_return_token(
        &get_ptr_fd.signature.output,
        &llbc,
        spec,
        &empty_gc,
        &harvested,
    );
    assert!(
        harvested_token
            .as_deref()
            .is_some_and(|t| t.contains("RawFlavorStruct") && t.starts_with("raw:")),
        "residual FUNC.RESULT of &RawFlavorStruct is raw:<owner>, got {harvested_token:?}"
    );
    assert_eq!(
            dont_look_inside_return_token(
                &get_ptr_fd.signature.output,
                &llbc,
                spec,
                &empty_gc,
                &empty,
            )
            .as_deref(),
            Some("ref"),
            "empty flavor set keeps the residual token as ref"
        );
    let graph = crate::test_support::lower_function(&llbc, "inspect").expect("lower inspect");
    let input = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Input {
                name,
                ty,
                class_root,
            } if name == "subject" => Some((ty.clone(), class_root.clone())),
            _ => None,
        })
        .expect("inspect parameter Input");
    assert!(
        matches!(
            &input.0,
            ValueType::Ref(Some(root)) if root.contains("RawFlavorStruct")
        ),
        "flavor-raw &RawFlavorStruct parameter is Ref(Some(root)), got {:?}",
        input.0
    );
    let root = input.1.expect("Ref parameter carries the class root");
    assert!(
        root.contains("RawFlavorStruct"),
        "class root names RawFlavorStruct, got {root}"
    );
    let input_var = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Input { name, .. } if name == "subject" => op.result.clone(),
            _ => None,
        })
        .expect("inspect parameter result");
    crate::translator::rtyper::legacy_annotator::annotate(&graph);
    crate::translator::rtyper::legacy_resolve::resolve_types(&graph, &harvested);
    assert_eq!(
        FunctionGraph::concretetype_of(&input_var),
        crate::model::ConcreteType::Signed,
        "valuetype_to_concrete_for_legacy(Ref(Some(flavor))) is Signed"
    );
    for var in graph.iter_variables() {
        *var.annotation.borrow_mut() = None;
        *var.concretetype.borrow_mut() = None;
    }
    crate::translator::rtyper::legacy_annotator::annotate(&graph);
    crate::translator::rtyper::legacy_resolve::resolve_types(&graph, &empty);
    assert_eq!(
        FunctionGraph::concretetype_of(&input_var),
        crate::model::ConcreteType::GcRef,
        "empty flavor set cannot int-bank at the legacy layer"
    );
}

/// Named-ADT arguments stay the omitted-default spelling of one class
/// (`struct_instantiation_suffix`). `raw_address_owner_root` must not
/// append them: `CodeObject<ConstantData>` and `CodeObject` intern as
/// two ClassDefs and `mergeinputargs` cannot unify them.
#[test]
fn flavor_raw_generic_named_arg_keeps_unsuffixed_class_root() {
    let span = fixture_span();
    let owner = struct_decl(
        0,
        ident_path(&["fixture", "RawFlavorStruct"]),
        empty_fields(),
        false,
    );
    let named_arg = struct_decl(
        1,
        ident_path(&["fixture", "NamedArg"]),
        empty_fields(),
        false,
    );
    let named_arg_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [], "types": [],
                "const_generics": [], "trait_refs": []
            },
            "builtin": null
        }
    });
    let ref_ty = serde_json::json!({
        "Ref": [
            "Erased",
            {"Adt": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [named_arg_ty],
                    "const_generics": [],
                    "trait_refs": []
                },
                "builtin": null
            }},
            "Shared"
        ]
    });
    let flavor_global = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(
            ident_path(&["fixture", "_alloc_flavor_RawFlavorStruct"])
        ),
        "global_kind": "NamedConst",
        "value": {"Value": [1, [{"Str": "raw"}, {"Ref": [
            "Erased",
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Shared"
        ]}]]}
    });
    let inspect = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "inspect"])),
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_ty],
            "output": {"Adt": {"builtin": "Tuple", "generics": {"types": []}}}
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    {"index": 0, "name": null, "span": span, "ty": {"Adt": {"builtin": "Tuple", "generics": {"types": []}}}},
                    {"index": 1, "name": "subject", "span": span, "ty": ref_ty}
                ]
            },
            "body": [{
                "statements": [],
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [owner, named_arg],
            "fun_decls": [inspect],
            "global_decls": [flavor_global],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let harvested =
        crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc));
    let empty_gc: std::collections::HashSet<majit_ir::descr::StructId> =
        std::collections::HashSet::new();
    let inspect_fd = llbc.fn_by_id(0).expect("inspect");
    let param_ty = &inspect_fd.signature.inputs[0];
    let class_root = super::tyref_class_root_with(param_ty, &llbc, super::no_tombstoned_leaves())
        .expect("pointee class root");
    assert!(
        !class_root.contains('<'),
        "tyref_class_root_with stays unsuffixed for a named-ADT arg, got {class_root}"
    );
    let vt = super::tyref_to_value_type_with(
        param_ty,
        &llbc,
        super::no_tombstoned_leaves(),
        &empty_gc,
        &harvested,
    );
    match vt {
        ValueType::Ref(Some(root)) => {
            assert_eq!(
                root, class_root,
                "flavor-raw front Ref root must match tyref_class_root_with"
            );
            assert!(
                !root.contains('<'),
                "flavor-raw pointer does not intern a suffixed ClassDef, got {root}"
            );
        }
        other => panic!("expected Ref(Some(unsuffixed root)), got {other:?}"),
    }
}

#[test]
fn type_arg_splits_per_instantiation_defers_singlefloat_unit_and_empty() {
    use super::type_arg_splits_per_instantiation;
    // Bare named heap classes split (1-word GC pointer in the
    // erased model).
    assert!(type_arg_splits_per_instantiation("Tuple"));
    assert!(type_arg_splits_per_instantiation("W_Object"));
    assert!(type_arg_splits_per_instantiation("str"));
    // Nested generics and tuples are also 1-word GC pointers in the
    // erased model (`Result<StepResult<…>, PyError>` etc.).
    assert!(type_arg_splits_per_instantiation("Option<i32>"));
    assert!(type_arg_splits_per_instantiation("(A,B)"));
    // Scalar integer/bool/char primitives split too: each carries a
    // well-defined int-banked repr, so the per-instantiation `__pos_0`
    // annotates to that repr instead of a program-wide union.
    for p in [
        "bool", "char", "u8", "u32", "u64", "u128", "usize", "i8", "i64", "i128", "isize",
    ] {
        assert!(type_arg_splits_per_instantiation(p), "{p} must split");
    }
    // DoubleFloat is float-banked. SingleFloat is int-banked in RPython,
    // so it remains deferred until the textual descriptor path agrees.
    assert!(type_arg_splits_per_instantiation("f64"));
    // Unit/degenerate atoms have no materialised payload field.
    for p in ["f32", "()", ""] {
        assert!(!type_arg_splits_per_instantiation(p), "{p} must not split");
    }
    for p in [
        "TypeVar",
        "TypeVar#0",
        "??TypeVar#0",
        "??unresolved_dedup#5",
    ] {
        assert!(
            !type_arg_splits_per_instantiation(p),
            "{p} is not a concrete instantiation"
        );
    }
    use super::type_arg_is_concrete;
    assert!(type_arg_is_concrete("()"));
    assert!(type_arg_is_concrete("f32"));
    assert!(type_arg_is_concrete("i64"));
    assert!(type_arg_is_concrete("Result<(), PyError>"));
    assert!(type_arg_is_concrete("??scalar"));
    assert!(!type_arg_is_concrete("Result<??scalar, PyError>"));
    assert!(type_arg_is_concrete("Never"));
    assert!(!type_arg_is_concrete("TypeVar#0"));
    assert!(!type_arg_is_concrete("Result<TypeVar, E>"));
    assert!(!type_arg_is_concrete("??TypeVar#0"));
    assert!(!type_arg_is_concrete("??adt#99"));
    assert!(!type_arg_is_concrete("Result<??adt#99, PyError>"));
    use super::enum_instantiation_suffix_from_args;
    assert_eq!(
        enum_instantiation_suffix_from_args(&["??scalar".into(), "PyError".into()]).as_deref(),
        Some("<??scalar,PyError>"),
        "whole-arg ??scalar is Charon Never and still suffixes Result"
    );
    assert!(!type_arg_is_concrete("Option<??scalar>"));
    assert!(!type_arg_is_concrete("??TraitType"));
}

#[test]
fn range_inclusive_known_receiver_paints_instantiation_class() {
    // `bookkeeper.py` `getuniqueclassdef`: a known RangeInclusive<i64>
    // receiver is that instantiation's class, not a classdef-less
    // `Ref(None)`. `start_bound` then getattr's `start` on it.
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let name: Vec<serde_json::Value> = ["core", "ops", "range", "RangeInclusive"]
        .iter()
        .map(|s| serde_json::json!({"Ident": [s, 0]}))
        .collect();
    let range_inc = serde_json::json!({
        "def_id": 0,
        "item_meta": {
            "name": name,
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Struct": [
            {"name": "start", "ty": i64_ty.clone()},
            {"name": "end", "ty": i64_ty.clone()},
            {"name": "exhausted", "ty": {"Scalar": "Bool"}}
        ]}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [range_inc],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let node = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [i64_ty],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    assert_eq!(
        super::adt_node_class_root_identity(&node, &llbc, super::no_tombstoned_leaves()),
        Some("ops::range::RangeInclusive<i64>".to_string())
    );
}

fn generic_struct_adt(id: serde_json::Value, type_arg: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": id,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [type_arg],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

#[test]
fn foreign_generic_slot_instantiations_print_and_intern_distinct() {
    // Harvested foreign-body crate, scalar args: `Slot<i64>` and
    // `Slot<f64>` are their own classes (`bookkeeper.py`
    // `getuniqueclassdef`).
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let f64_ty = serde_json::json!({"Scalar": {"Float": "F64"}});
    let tvar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let slot = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["cell", "Slot"], "", false),
        "kind": {"Struct": [
            {"name": "value", "ty": tvar, "attr_info": null}
        ]}
    });
    let body = serde_json::json!({
        "Unstructured": {
            "span": span_json(),
            "locals": {
                "arg_count": 0,
                "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": {"Scalar": "Bool"}}
                ]
            },
            "body": [{"statements": [], "terminator": {"span": span_json(), "kind": "Return"}}]
        }
    });
    let getter = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["cell", "get"], "", false),
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": {"Scalar": "Bool"}
        },
        "body": body
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [slot],
            "fun_decls": [getter],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    assert!(llbc.has_body_under_crate_root("cell"));
    let node_i64 = generic_struct_adt(serde_json::json!(0), i64_ty);
    let node_f64 = generic_struct_adt(serde_json::json!(0), f64_ty);
    let printed_i64 = charon_type_value_to_ast_string(&node_i64, &llbc, 0);
    let printed_f64 = charon_type_value_to_ast_string(&node_f64, &llbc, 0);
    assert_eq!(printed_i64, "Slot<i64>");
    assert_eq!(printed_f64, "Slot<f64>");
    assert_eq!(
        super::adt_node_class_root_identity(&node_i64, &llbc, super::no_tombstoned_leaves()),
        Some("Slot<i64>".to_string())
    );
    assert_eq!(
        super::adt_node_class_root_identity(&node_f64, &llbc, super::no_tombstoned_leaves()),
        Some("Slot<f64>".to_string())
    );
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::front::StructFieldRegistry;
    let bk = std::rc::Rc::new(Bookkeeper::new());
    let mut reg = StructFieldRegistry::default();
    reg.foreign_body_roots.push("cell".to_string());
    reg.fields.insert(
        printed_i64.clone(),
        vec![("value".to_string(), "i64".to_string())],
    );
    reg.fields.insert(
        printed_f64.clone(),
        vec![("value".to_string(), "f64".to_string())],
    );
    bk.set_struct_fields(std::rc::Rc::new(reg));
    let i64_host = bk.intern_class_by_qualname(&printed_i64);
    let f64_host = bk.intern_class_by_qualname(&printed_f64);
    assert_ne!(
        i64_host, f64_host,
        "foreign Slot instantiations intern as different ClassDefs"
    );
    let cd_i = bk.getuniqueclassdef(&i64_host).expect("Slot<i64> classdef");
    let cd_f = bk.getuniqueclassdef(&f64_host).expect("Slot<f64> classdef");
    assert!(!std::rc::Rc::ptr_eq(&cd_i, &cd_f));
}

#[test]
fn range_inclusive_self_after_subst_paints_instantiation_class() {
    // Product `start_bound` encodes `&self` as Dedup → Ref → Adt
    // whose type arg is Dedup → TypeVar. `substitute_type_vars`
    // copies those wrappers; the painted class is then
    // `RangeInclusive<i64>` (`bookkeeper.py` `getuniqueclassdef`).
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let name: Vec<serde_json::Value> = ["core", "ops", "range", "RangeInclusive"]
        .iter()
        .map(|s| serde_json::json!({"Ident": [s, 0]}))
        .collect();
    let tvar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let adt = serde_json::json!({"Adt": {
        "id": 0,
        "builtin": null,
        "generics": {
            "regions": [],
            "types": [{"Value": [5, tvar]}],
            "const_generics": [],
            "trait_refs": []
        }
    }});
    let self_ty = serde_json::json!({
        "Ref": [{"Static": null}, {"Value": [40, adt]}, "Shared"]
    });
    let range_inc = serde_json::json!({
        "def_id": 0,
        "item_meta": {
            "name": name,
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Struct": [
            {"name": "start", "ty": {"TypeVar": {"Bound": [0, 0]}}},
            {"name": "end", "ty": {"TypeVar": {"Bound": [0, 0]}}},
            {"name": "exhausted", "ty": {"Scalar": "Bool"}}
        ]}
    });
    let body = serde_json::json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    {"index": 0, "name": null, "span": span, "ty": {"Scalar": "Bool"}},
                    {"index": 1, "name": "self", "span": span, "ty": {"Value": [1, self_ty]}}
                ]
            },
            "body": [{"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}}]
        }
    });
    let file = serde_json::json!({
        "charon_version": "t",
        "has_errors": false,
        "translated": {
            "crate_name": "c",
            "type_decls": [range_inc],
            "fun_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [{"Ident": ["start_bound", 0]}],
                    "span": span,
                    "source_text": null,
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": false
                },
                "signature": {
                    "is_unsafe": false,
                    "inputs": [{"Deduplicated": 1}],
                    "output": {"Scalar": "Bool"}
                },
                "body": body
            }],
            "files": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let fd = llbc.fn_by_id(0).expect("start_bound");
    let copied =
        crate::front::clause_spec::substituted_unstructured(fd, &llbc, &[], &[i64_ty], &[])
            .expect("substituted start_bound body");
    let self_ty = &copied.locals.locals[1].ty;
    assert_eq!(
        super::tyref_ref_param_class_root(self_ty, &llbc, super::no_tombstoned_leaves(), None),
        Some("ops::range::RangeInclusive<i64>".to_string())
    );
}

#[test]
fn enum_suffix_keeps_unit_ok_when_err_splits() {
    use super::enum_instantiation_suffix_from_args;
    // `Result<(), PyError>` vs `Result<(), (i32, String)>` must not
    // share `Result::Err.__pos_0` (`native_result` `map_err`).
    assert_eq!(
        enum_instantiation_suffix_from_args(&["()".into(), "OsError".into()]).as_deref(),
        Some("<(),OsError>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["()".into(), "(i32, String)".into()]).as_deref(),
        Some("<(),(i32, String)>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["i64".into(), "OsError".into()]).as_deref(),
        Some("<i64,OsError>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["()".into(), "()".into()]),
        None
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["f32".into(), "f32".into()]),
        None
    );
    assert_eq!(enum_instantiation_suffix_from_args(&["()".into()]), None);
    // Nested unresolved args are not an instantiation: `Option<??scalar>`
    // does not start with `??`, so prefix-only splitting interned
    // `ControlFlow<Option<??scalar>, i64>` and getattr `from_residual`
    // blocked on that classdef (`w_range_fields_i64` `?`).
    assert_eq!(
        enum_instantiation_suffix_from_args(&["Option<??scalar>".into(), "i64".into()]),
        None
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["i64".into(), "Option<??scalar>".into()]),
        None
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["Result<TypeVar, E>".into()]),
        None
    );
    // Whole-arg unresolved associated type: the sibling concrete
    // payload still intern's (`getuniqueclassdef`). Nested unresolved
    // above stays out.
    assert_eq!(
        enum_instantiation_suffix_from_args(&["u32".into(), "??TraitType".into()]).as_deref(),
        Some("<u32,??TraitType>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["??TraitType".into(), "i64".into()]).as_deref(),
        Some("<??TraitType,i64>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["??TypeVar#0".into(), "i64".into()]),
        None
    );
    // A leaked hasher `fn() -> ()` Instantiated arg is concrete (the
    // `{"Tuple": []}` unit, not `??Tuple`).
    assert_eq!(
        enum_instantiation_suffix_from_args(&["WordHasher".into(), "fn() -> ()".into()]).as_deref(),
        Some("<WordHasher,fn() -> ()>")
    );
    // Charon hash-cons `{"Value":[id,"Never"]}` is a type-name string.
    // Residual dest_ty of Try::branch spells that uninhabited ok as a
    // non-object payload (`??scalar`); dest_adt still intern's.
    let llbc = empty_llbc();
    assert_eq!(
        charon_type_value_to_ast_string(&serde_json::json!({"Value": [0, "Never"]}), &llbc, 0),
        "Never"
    );
    assert_eq!(
        charon_type_value_to_ast_string(&serde_json::json!({"Value": [0, null]}), &llbc, 0),
        "??scalar"
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["Never".into(), "BytecodeCorruption".into()])
            .as_deref(),
        Some("<Never,BytecodeCorruption>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["??scalar".into(), "PyError".into()]).as_deref(),
        Some("<??scalar,PyError>")
    );
    assert_eq!(
        enum_instantiation_suffix_from_args(&["Option<Never>".into(), "i64".into()]).as_deref(),
        Some("<Option<Never>,i64>")
    );
}

#[test]
fn object_items_pointer_names_its_element_list() {
    use super::object_ref_items_list_root;

    let object_decl = struct_decl(
        2,
        ident_path(&["pyobject", "PyObject"]),
        empty_fields(),
        false,
    );
    let (llbc, _) = load_handle(vec![(2, object_decl)], serde_json::json!([]), 2);
    let items = TyRef::Other(serde_json::json!({
        "RawPtr": [
            {"RawPtr": [
                {"Adt": {"id": 2, "generics": {"types": []}}},
                "Mut"
            ]},
            "Mut"
        ]
    }));
    assert_eq!(
        object_ref_items_list_root(&items, &llbc).as_deref(),
        Some("[PyObject]")
    );
    assert_eq!(super::list_spelling_item("[PyObject]"), Some("PyObject"));
    assert_eq!(super::list_spelling_item("[PyObject; 0]"), None);
    let scalar = TyRef::Other(serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    }));
    assert_eq!(object_ref_items_list_root(&scalar, &llbc), None);
    let header = TyRef::Other(serde_json::json!({
        "RawPtr": [
            {"Adt": {"id": 2, "generics": {"types": []}}},
            "Mut"
        ]
    }));
    assert_eq!(
        object_ref_items_list_root(&header, &llbc),
        None,
        "a pointer to the object itself is not the items pointer"
    );
    assert_eq!(
        super::object_stored_ptr_items_list_root(&header, &llbc).as_deref(),
        Some("[PyObject]"),
        "a stored *mut Class is the items-list element"
    );
    assert_eq!(
        super::object_stored_ptr_items_list_root(&items, &llbc),
        None,
        "an items pointer (*mut *mut Class) is not a stored element"
    );
    assert_eq!(
        super::object_stored_ptr_class_root(&header, &llbc).as_deref(),
        Some("PyObject"),
        "a stored *mut Class names the element class"
    );
    assert_eq!(
        super::object_stored_ptr_class_root(&items, &llbc),
        None,
        "an items pointer (*mut *mut Class) is not a stored element"
    );
}

#[test]
fn object_pointer_slice_names_its_element_list() {
    use super::object_pointer_slice_list_root;

    let object_decl = struct_decl(
        2,
        ident_path(&["pyobject", "PyObject"]),
        empty_fields(),
        false,
    );
    let (llbc, _) = load_handle(vec![(2, object_decl)], serde_json::json!([]), 2);
    let objptr = serde_json::json!({
        "RawPtr": [
            {"Adt": {"id": 2, "generics": {"types": []}}},
            "Mut"
        ]
    });
    let slice = TyRef::Other(serde_json::json!({
        "Ref": ["_", {"Slice": [objptr.clone(), null]}, "Shared"]
    }));
    assert_eq!(
        object_pointer_slice_list_root(&slice, &llbc).as_deref(),
        Some("[PyObject]")
    );
    let raw = TyRef::Other(serde_json::json!({
        "RawPtr": [{"Slice": [objptr, null]}, "Const"]
    }));
    assert_eq!(
        object_pointer_slice_list_root(&raw, &llbc).as_deref(),
        Some("[PyObject]")
    );
}

#[test]
fn items_block_base_accessor_gate_excludes_deref_in_place() {
    use super::graph_is_items_block_base_accessor;

    // The object and typed-array accessors whose `.add(*_ITEMS_OFFSET)`
    // body returns the interior items pointer for descr-based consumption
    // are safe to collapse to the receiver, regardless of crate prefix.
    assert!(graph_is_items_block_base_accessor(
        "pyre_object::object_array::items_block_items_base"
    ));
    assert!(graph_is_items_block_base_accessor(
        "pyre_object::object_array::items_block_items_ptr"
    ));
    assert!(graph_is_items_block_base_accessor(
        "jit_artifact::object_array::items_block_items_base"
    ));
    assert!(graph_is_items_block_base_accessor(
        "majit_rlib::lltypesystem::rlist::typed_items_block_items_base"
    ));
    assert!(graph_is_items_block_base_accessor(
        "pyre_object::int_array::<Impl>::base"
    ));
    assert!(graph_is_items_block_base_accessor(
        "pyre_object::float_array::<Impl>::base"
    ));
    assert!(graph_is_items_block_base_accessor(
        "pyre_object::tupleobject::tuple_items_base"
    ));

    // Bodies that dereference a `.add(NAMED_OFFSET)` interior pointer
    // in place must NOT be aliased — the offset is load-bearing.
    assert!(!graph_is_items_block_base_accessor(
        "pyre_object::tupleobject::w_tuple_getitem_known"
    ));
    assert!(!graph_is_items_block_base_accessor(
        "pyre_interpreter::runtime_ops::load_global_str_extern"
    ));
    // A leaf collision in another module must not widen the gate.
    assert!(!graph_is_items_block_base_accessor(
        "pyre_object::other_mod::items_block_items_base_helper"
    ));
    assert!(!graph_is_items_block_base_accessor(
        "pyre_object::my_int_array::<Impl>::base"
    ));
}

/// brick 1 collapses an accessor to its receiver — the block header — and
/// only brick 3's `base_size`-bearing array descr adds the item offset
/// back. Every accessor brick 1 rewrites must therefore have a matching
/// brick-3 family: object accessors use `GcArray(OBJECTPTR)`, while the
/// typed accessor is deliberately excluded from that predicate and uses
/// the separate `GcArray(Signed|Float)` gate.
#[test]
fn the_two_bricks_agree_on_every_reference_items_base_accessor() {
    use super::{graph_is_items_block_base_accessor, is_object_items_block_base_accessor};

    for name in [
        "pyre_object::object_array::items_block_items_base",
        "pyre_object::object_array::items_block_items_ptr",
        "jit_artifact::object_array::items_block_items_base",
        "pyre_object::tupleobject::tuple_items_base",
    ] {
        assert!(
            is_object_items_block_base_accessor(name),
            "brick 3 declines {name}, which brick 1 rewrites"
        );
        assert!(graph_is_items_block_base_accessor(name));
    }

    // The typed block is not an object-array accessor; its own brick-3
    // gate supplies the scalar array operation.
    let typed = "majit_rlib::lltypesystem::rlist::typed_items_block_items_base";
    assert!(graph_is_items_block_base_accessor(typed));
    assert!(!is_object_items_block_base_accessor(typed));

    // Neither gate is keyed on a crate: a host crate laying its block out
    // the same way is recognised by the same module-qualified suffix. The
    // name below is a shape, not a crate this workspace contains.
    let host = "some_host::runtime::object_array::items_block_items_base";
    assert!(is_object_items_block_base_accessor(host));
    assert!(graph_is_items_block_base_accessor(host));

    // A leaf collision still cannot widen either gate. Note this case
    // proves less than it looks: it is refused for TWO independent
    // reasons — the leaf is `…_helper`, and the module is `other_mod` —
    // so it passes even under a plain `ends_with`, and it never tested
    // whether the comparison respects `::` boundaries at all.
    let collision = "some_host::other_mod::items_block_items_base_helper";
    assert!(!is_object_items_block_base_accessor(collision));
    assert!(!graph_is_items_block_base_accessor(collision));

    // The case that does test it. Every segment of the key appears, the
    // leaf is exact, and the whole key is a byte-suffix of the path — so
    // `ends_with` accepts it — yet the module is `my_object_array` and not
    // `object_array`, so a path comparison must refuse it. This is the one
    // input whose answer changed when the gates moved off `ends_with`.
    let neighbour = "some_host::runtime::my_object_array::items_block_items_base";
    assert!(!is_object_items_block_base_accessor(neighbour));
    assert!(!graph_is_items_block_base_accessor(neighbour));

    // The same defect on the other predicate's own arm.
    let neighbour_typed = "majit_rlib::lltypesystem::my_rlist::typed_items_block_items_base";
    assert!(!graph_is_items_block_base_accessor(neighbour_typed));
}

/// `BytesArray` / `UnicodeArray` methods share `items_block_items_base`
/// with object lists. The `[PyObject]` call-result narrow must not fire
/// on those graphs (`SomeString` items, `string_items_elem_ptr_add`).
#[test]
fn string_array_graphs_skip_the_object_items_list_narrow() {
    use super::graph_is_string_items_owner;

    assert!(graph_is_string_items_owner(
        "pyre_object::bytes_array::<Impl>::push"
    ));
    assert!(graph_is_string_items_owner(
        "pyre_object::bytes_array::<Impl>::set"
    ));
    assert!(graph_is_string_items_owner(
        "pyre_object::unicode_array::<Impl>::set"
    ));
    assert!(!graph_is_string_items_owner(
        "pyre_object::tupleobject::wraptuple"
    ));
    assert!(!graph_is_string_items_owner(
        "pyre_object::listobject::w_list_make_empty"
    ));
    assert!(!graph_is_string_items_owner(
        "pyre_object::my_bytes_array::<Impl>::push"
    ));
}

/// Anchor PyPy's single string-valued `W_UnicodeObject._utf8` shape to
/// the real pyre-object LLBC.  Rust stores that value behind
/// `*mut Wtf8Buf`, but both its canonical annotation metadata and the
/// managed storage-box call must collapse to the one RPython string value.
#[test]
#[ignore]
fn unicode_storage_matches_real_object_llbc() {
    use crate::model::{CallTarget, OpKind, ValueType};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = crate::test_support::load_ullbc(path);
    let (_, _, fields, _, _, attrs, _, _) = super::derive_program_metadata(
        llbc,
        &super::harvest_declared_gc_facts(llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(llbc)),
        &[],
    );
    assert!(
        fields
            .fields
            .get("pyre_object::unicodeobject::W_UnicodeObject")
            .is_some_and(|rows| {
                rows.iter()
                    .any(|row| row.name == "value" && row.ty == "*mut Wtf8Buf")
            }),
        "the declaration metadata must retain the physical storage row"
    );
    assert!(
        attrs
            .get("unicodeobject::W_UnicodeObject")
            .is_some_and(|rows| rows.contains(&("value".to_string(), ValueType::Str))),
        "W_UnicodeObject.value must project as PyPy's string-valued _utf8 field"
    );

    let graph = crate::test_support::lower_function(llbc, "w_str_from_wtf8_managed")
        .expect("lower managed string constructor");
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if super::fmt_path_ends_with(segments, &["gc_alloc_storage_box"])
            )
        }),
        "the Wtf8 storage box must be an RPython-string identity"
    );
}

fn constants_accessor_fun_decl_id(kind: &OpKind, leaf: &str) -> Option<u64> {
    use crate::model::CallTarget;
    let OpKind::Call { target, .. } = kind else {
        return None;
    };
    let matches = match target {
        CallTarget::FunctionPath { segments, .. } => {
            super::fmt_path_ends_with(segments, &["Constants", leaf])
        }
        CallTarget::Method {
            name,
            resolved_path,
            receiver_root,
            ..
        } => {
            name == leaf
                && (receiver_root
                    .as_deref()
                    .is_some_and(|root| root.ends_with("Constants"))
                    || resolved_path.as_ref().is_some_and(|path| {
                        path.segments.iter().any(|segment| segment == "Constants")
                    }))
        }
        _ => false,
    };
    if !matches {
        return None;
    }
    Some(
        target
            .fun_decl_id()
            .unwrap_or_else(|| panic!("Constants::{leaf} call has no fun_decl_id, got {target:?}")),
    )
}

/// Anchor compiler-core's exact
/// `Constants(Box<[C]>)::{deref,index}` storage shape to the real
/// interpreter LLBC.  `constant_at` is `pyopcode.py getconstant_w`
/// (a `co_consts_w` field read on the promoted code); it must not
/// FieldRead the compiler `Constants` wrapper.  `code_getdocstring`
/// still projects that wrapper, through the translated `deref` body.
///
/// `#[ignore]` is deliberate and has a precondition, not a verdict: this
/// loads a 667 MB artefact that only exists after `extract-llbc.py` has run,
/// and takes ~11 s.  Run it explicitly after a fresh extraction:
///
/// ```text
/// cargo test -p majit-translate --lib \
///     constants_wrapper_access_matches_real_interpreter_llbc -- --ignored
/// ```
///
/// An ignored test states a belief until someone runs it, so this must be
/// exercised explicitly whenever the extracted interpreter LLBC changes.
#[test]
#[ignore]
fn constants_wrapper_access_matches_real_interpreter_llbc() {
    use crate::model::{CallTarget, OpKind, ValueType};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = crate::test_support::load_ullbc(path);
    let (_, _, fields, _, _, attrs, _, _) = super::derive_program_metadata(
        llbc,
        &super::harvest_declared_gc_facts(llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(llbc)),
        &[],
    );
    assert_eq!(
        fields.fields.get("Constants"),
        Some(&vec![crate::front::semantic::FieldRow::positional(
            "__pos_0",
            "Box<[ConstantData]>"
        )])
    );
    assert_eq!(
        attrs.get("bytecode::Constants"),
        Some(&vec![("__pos_0".to_string(), ValueType::Ref(None))])
    );
    // A payload-carrying variant inherits the enum base's
    // `__discriminant` as its first row (`rclass.py` InstanceRepr
    // subclass with the super-field), then the payload field.
    assert_eq!(
        fields.fields.get("ConstantData::Integer"),
        Some(&vec![
            crate::front::semantic::FieldRow::named("__discriminant", "u64"),
            crate::front::semantic::FieldRow::named("value", "BigInt"),
        ])
    );
    assert_eq!(
        fields.fields.get("ConstantData::Complex"),
        Some(&vec![
            crate::front::semantic::FieldRow::named("__discriminant", "u64"),
            crate::front::semantic::FieldRow::named("value", "Complex<f64>"),
        ])
    );
    assert_eq!(
        fields.fields.get("ConstantData::Str"),
        Some(&vec![
            crate::front::semantic::FieldRow::named("__discriminant", "u64"),
            crate::front::semantic::FieldRow::named("value", "Wtf8Buf"),
        ])
    );
    assert_eq!(
        fields.fields.get("ConstantData::Code"),
        Some(&vec![
            crate::front::semantic::FieldRow::named("__discriminant", "u64"),
            crate::front::semantic::FieldRow::named("code", "Box<CodeObject<ConstantData>>"),
        ])
    );
    assert_eq!(
        fields.fields.get("Complex"),
        Some(&vec![
            crate::front::semantic::FieldRow::named("re", "f64"),
            crate::front::semantic::FieldRow::named("im", "f64"),
        ])
    );
    // `attrs` is keyed by the crate-stripped path, and `strip_crate_prefix`
    // splits on the FIRST `::` — so `num_complex::Complex` strips to the bare
    // leaf `Complex` while `compiler_core::bytecode::Constants` above keeps its
    // module qualifier as `bytecode::Constants`. The two assertions read as a
    // symmetric pair and are not: path depth decides whether the key retains a
    // qualifier.
    assert_eq!(
        attrs.get("Complex"),
        Some(&vec![
            ("re".to_string(), ValueType::Float),
            ("im".to_string(), ValueType::Float),
        ])
    );
    let constant_at = crate::test_support::lower_function(llbc, "constant_at")
        .expect("lower ConstantOpcodeHandler::constant_at");
    let constant_ops: Vec<_> = constant_at
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    // `ConstIdx` is a `#[repr(transparent)] u32` oparg wrapper, so the
    // front-end types it `Unsigned` — whose register class *is* `'int'`
    // (`getkind(Unsigned) == 'int'`, `model.rs`), signedness being an
    // rtyper-level distinction only. `Int | Unsigned` is the pairing every
    // bank-blind consumer already uses; naming the input as well keeps this
    // from being satisfied by some unrelated integer argument.
    assert_eq!(
        constant_ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Input {
                        name,
                        ty: ValueType::Int | ValueType::Unsigned,
                        ..
                    } if name == "index"
                )
            })
            .count(),
        1,
        "ConstIdx input must use RPython's integer oparg representation"
    );
    assert_eq!(
        constant_ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. }
                        if field.name == "__pos_0"
                            && field.owner_root.as_deref() == Some("Constants")
                )
            })
            .count(),
        0,
        "getconstant_w reads co_consts_w, not compiler Constants.__pos_0"
    );
    assert!(
        constant_ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "co_consts_w"
            )
        }),
        "getconstant_w reads self.getcode().co_consts_w"
    );
    assert!(
        !constant_ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if super::fmt_path_ends_with(segments, &["w_code_const"])
            )
        }),
        "getconstant_w indexes co_consts_w directly"
    );
    for op in &constant_ops {
        if let Some(id) = constants_accessor_fun_decl_id(&op.kind, "index") {
            let fd = llbc
                .fn_by_id(id)
                .unwrap_or_else(|| panic!("Constants::index fun_decl_id={id} missing"));
            assert!(
                fd.has_unstructured_body(),
                "Constants::index FunDecl {} is Opaque",
                fd.item_meta.name_path()
            );
        }
    }

    let getdocstring = crate::test_support::lower_function(llbc, "code_getdocstring")
        .expect("lower code_getdocstring");
    let doc_ops: Vec<_> = getdocstring
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    assert_eq!(
        doc_ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. }
                        if field.name == "__pos_0"
                            && field.owner_root.as_deref() == Some("Constants")
                )
            })
            .count(),
        0,
        "code_getdocstring projects Constants through the translated deref body"
    );
    let mut deref_ids = Vec::new();
    for op in &doc_ops {
        if let Some(id) = constants_accessor_fun_decl_id(&op.kind, "deref") {
            let fd = llbc
                .fn_by_id(id)
                .unwrap_or_else(|| panic!("Constants::deref fun_decl_id={id} missing"));
            assert!(
                fd.has_unstructured_body(),
                "Constants::deref FunDecl {} is Opaque",
                fd.item_meta.name_path()
            );
            deref_ids.push(id);
        }
    }
    assert!(
        !deref_ids.is_empty(),
        "code_getdocstring must call translated Constants::deref"
    );
    crate::test_support::with_lowering(llbc, |context| {
        let mut deref_field_reads = 0usize;
        for id in deref_ids {
            let fd = llbc.fn_by_id(id).expect("Constants::deref FunDecl");
            let deref_graph = super::lower_fun_decl(context, fd)
                .unwrap_or_else(|e| panic!("lower Constants::deref: {e:?}"));
            deref_field_reads += deref_graph
                .blocks
                .iter()
                .flat_map(|block| block.operations.iter())
                .filter(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. }
                            if field.name == "__pos_0"
                                && field.owner_root.as_deref().is_some_and(|root| root.ends_with("Constants"))
                    )
                })
                .count();
        }
        assert!(
            deref_field_reads >= 1,
            "translated Constants::deref reads Constants.__pos_0"
        );
    });

    let load_const = crate::test_support::lower_function(llbc, "load_const_value")
        .expect("lower rbigint-consuming load_const_value");
    let load_ops: Vec<_> = load_const
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    assert_eq!(
        load_ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Input {
                        name,
                        class_root: Some(root),
                        ..
                    } if name == "constant"
                        && (root == "ConstantData" || root.ends_with("::ConstantData"))
                )
            })
            .count(),
        1
    );

    // `PyFrame::bigint_constant(&RBigInt)` is the LOAD_CONST consumer
    // that boxes the already-translated payload. Anchor its ABI rewrite
    // to the real interpreter LLBC: no by-value `w_long_new` call may
    // survive, and the existing GC reference must flow to
    // `w_long_from_raw`.
    let mut boxed_handlers = 0usize;
    crate::test_support::with_lowering(llbc, |context| {
        for fd in llbc.iter_local_fns().filter(|fd| {
            fd.item_meta.name_path().ends_with("::bigint_constant") && fd.unstructured().is_some()
        }) {
            let graph = super::lower_fun_decl(context, fd)
                .unwrap_or_else(|e| panic!("lower {}: {e:?}", fd.item_meta.name_path()));
            let calls: Vec<&[String]> = graph
                .blocks
                .iter()
                .flat_map(|block| block.operations.iter())
                .filter_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } => Some(segments.as_slice()),
                    _ => None,
                })
                .collect();
            if calls
                .iter()
                .any(|segments| segments.last().map(String::as_str) == Some("w_long_from_raw"))
            {
                boxed_handlers += 1;
                assert!(!calls.iter().any(|segments| {
                    matches!(
                        segments.last().map(String::as_str),
                        Some(
                            "w_long_new"
                                | "w_long_new_fresh_rbigint_handle"
                                | "box_bigint_constant"
                        )
                    )
                }));
            }
        }
        // The loop above filters to bodies (`unstructured().is_some()`), which
        // drops the trait declaration and leaves the single impl. `== 1` reds if
        // a second body-carrying impl appears: the inner ABI check runs only
        // inside the `w_long_from_raw` arm, so a duplicate that reverted to the
        // by-value ABI would fall out of that arm and go ungraded.
        assert_eq!(
            boxed_handlers, 1,
            "real LOAD_CONST bigint handler must use the RBigInt pointer boxing ABI"
        );
    });
    // `load_const_value` receives `&ConstantData` already. Manufacturing a
    // one-word ArrayRead of the compiler's `Vec<ConstantData>` would stride
    // into the middle of an inline enum element; the graph must not grow
    // one.
    assert_eq!(
        load_ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::ArrayRead { .. }))
            .count(),
        0
    );
    assert!(!load_ops.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("toint")
        )
    }));
    let call_leaves: Vec<&str> = load_ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str),
            _ => None,
        })
        .collect();
    assert_eq!(
        load_ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().map(String::as_str) == Some("jit_bigint_toint")
                )
            })
            .count(),
        1,
        "load_const_value calls jit_bigint_toint once; leaves={call_leaves:?}"
    );
    for residual in ["jit_bigint_to_i64_value_or_zero", "jit_bigint_to_i64_fits"] {
        assert_eq!(
            load_ops
                .iter()
                .filter(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.last().map(String::as_str) == Some(residual)
                    )
                })
                .count(),
            0,
            "{residual} is not the load_const_value toint residual"
        );
    }
    assert_eq!(
            load_ops
                .iter()
                .filter(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldWrite { field, .. }
                            if field.name == "__discriminant"
                                && field.owner_root.as_deref().is_some_and(|root| root.contains("Result<i64,RBigIntError>"))
                    )
                })
                .count(),
            0,
            "load_const_value does not write a Result<i64,RBigIntError> discriminant"
        );
}

/// Two impls bind the associated type differently. A `TraitImpl` ref
/// reads that impl's binding; anything else is not unique.
#[test]
fn trait_assoc_impl_ref_resolves_its_own_binding() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let binding = |ty: &serde_json::Value| {
        serde_json::json!({
            "kind": {"TraitType": [0, 0]},
            "skip_binder": {"value": ty}
        })
    };
    let row = |ty: &serde_json::Value| {
        serde_json::json!({
            "impl_trait": {"id": 0, "generics": {"types": []}},
            "types": [binding(ty)],
            "implied_trait_refs": []
        })
    };
    let impl_ref = |id: u64| {
        serde_json::json!({
            "kind": {"TraitImpl": {"id": id, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}},
            "trait_decl_ref": {"skip_binder": {"id": 0}}
        })
    };
    let llbc = llbc_with_trait_impls(serde_json::json!([row(&i64_ty), row(&bool_ty)]));
    let assoc = serde_json::json!(0);
    assert_eq!(
        super::resolve_trait_assoc_type_value(&impl_ref(0), &assoc, &llbc),
        Some(&i64_ty)
    );
    assert_eq!(
        super::resolve_trait_assoc_type_value(&impl_ref(1), &assoc, &llbc),
        Some(&bool_ty)
    );
    let clause = serde_json::json!({
        "kind": {"Clause": {"Bound": [0, 0]}},
        "trait_decl_ref": {"skip_binder": {"id": 0}}
    });
    assert_eq!(
        super::resolve_trait_assoc_type_value(&clause, &assoc, &llbc),
        None
    );
}

#[test]
fn trait_assoc_pred_generics_select_among_impls() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let char_ty = serde_json::json!({"Scalar": "Char"});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let binding = |ty: &serde_json::Value| {
        serde_json::json!({
            "kind": {"TraitType": [0, 0]},
            "skip_binder": {"value": ty}
        })
    };
    let row = |self_ty: &serde_json::Value, bound: &serde_json::Value| {
        serde_json::json!({
            "impl_trait": {"id": 0, "generics": {"types": [self_ty]}},
            "types": [binding(bound)],
            "implied_trait_refs": []
        })
    };
    let llbc = llbc_with_trait_impls(serde_json::json!([
        row(&i64_ty, &bool_ty),
        row(&char_ty, &i64_ty)
    ]));
    let assoc = serde_json::json!(0);
    let clause = |self_ty: &serde_json::Value| {
        serde_json::json!({
            "kind": {"Clause": {"Bound": [0, 0]}},
            "trait_decl_ref": {
                "skip_binder": {
                    "id": 0,
                    "generics": {
                        "regions": [],
                        "types": [self_ty],
                        "const_generics": [],
                        "trait_refs": []
                    }
                }
            }
        })
    };
    assert_eq!(
        super::resolve_trait_assoc_type_value(&clause(&i64_ty), &assoc, &llbc),
        Some(&bool_ty)
    );
    assert_eq!(
        super::resolve_trait_assoc_type_value(&clause(&char_ty), &assoc, &llbc),
        Some(&i64_ty)
    );
    let projection = |self_ty: &serde_json::Value| {
        serde_json::json!({
            "TraitType": [
                clause(self_ty),
                0,
                {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
            ]
        })
    };
    assert_eq!(
        charon_type_value_to_ast_string(&projection(&i64_ty), &llbc, 0),
        "bool"
    );
    assert_eq!(
        super::enum_instantiation_suffix_from_args(&[
            "u32".into(),
            charon_type_value_to_ast_string(&projection(&i64_ty), &llbc, 0),
        ])
        .as_deref(),
        Some("<u32,bool>")
    );
}

#[test]
fn trait_assoc_pred_generics_ambiguous_or_open_stay_unresolved() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let char_ty = serde_json::json!({"Scalar": "Char"});
    let binding = |ty: &serde_json::Value| {
        serde_json::json!({
            "kind": {"TraitType": [0, 0]},
            "skip_binder": {"value": ty}
        })
    };
    let row = |self_ty: &serde_json::Value, bound: &serde_json::Value| {
        serde_json::json!({
            "impl_trait": {"id": 0, "generics": {"types": [self_ty]}},
            "types": [binding(bound)],
            "implied_trait_refs": []
        })
    };
    let two_for_i64 = llbc_with_trait_impls(serde_json::json!([
        row(&i64_ty, &bool_ty),
        row(&i64_ty, &char_ty)
    ]));
    let assoc = serde_json::json!(0);
    let clause = serde_json::json!({
        "kind": {"Clause": {"Bound": [0, 0]}},
        "trait_decl_ref": {
            "skip_binder": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [i64_ty],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }
    });
    assert_eq!(
        super::resolve_trait_assoc_type_value(&clause, &assoc, &two_for_i64),
        None
    );
    let open = serde_json::json!({
        "kind": {"Clause": {"Bound": [0, 0]}},
        "trait_decl_ref": {
            "skip_binder": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [{"TypeVar": {"Bound": [0, 0]}}],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }
    });
    let one_impl = llbc_with_trait_impls(serde_json::json!([row(&i64_ty, &bool_ty)]));
    // Unique-impl still answers when the predicate is open: there is
    // only one impl of the trait.
    assert_eq!(
        super::resolve_trait_assoc_type_value(&open, &assoc, &one_impl),
        Some(&bool_ty)
    );
}

#[test]
fn zero_sized_fields_are_payload_free_but_mixed_fields_are_not() {
    let span = || {
        serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }})
    };
    let item_meta = |name: &str| {
        serde_json::json!({
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": [name, 0]}
            ],
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        })
    };
    let adt = |def_id: u64, hash: u64| {
        serde_json::json!({"Value": [hash, {
            "Adt": {
                "id": def_id,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }]})
    };
    let field =
        |ty: serde_json::Value| serde_json::json!({"name": null, "ty": ty, "attr_info": null});
    let variant = |name: &str, fields: Vec<serde_json::Value>, discriminant: u64| {
        serde_json::json!({
            "name": name,
            "fields": fields,
            "discriminant": {"Scalar": {"Unsigned": ["U8", discriminant.to_string()]}}
        })
    };
    let layout = |size: u64, field_offsets: Vec<u64>| {
        serde_json::json!([{
            "key": "fixture-target",
            "value": {
                "size": size,
                "align": 1,
                "variant_layouts": [{"field_offsets": field_offsets}],
                "repr": {"transparent": false}
            }
        }])
    };
    let phantom = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta("Phantom"),
        "kind": {"Struct": []},
        "layout": layout(0, vec![])
    });
    let only_zst = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta("OnlyZst"),
        "kind": {"Enum": [variant("Value", vec![field(adt(0, 10))], 0)]},
        "layout": layout(1, vec![0])
    });
    let mixed = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta("Mixed"),
        "kind": {"Enum": [variant(
            "Value",
            vec![
                field(adt(0, 11)),
                field(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}))
            ],
            0
        )]},
        "layout": layout(16, vec![0, 8])
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [phantom, only_zst, mixed],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let phantom_ty = serde_json::from_value::<TyRef>(adt(0, 20)).expect("phantom TyRef");
    let only_zst_ty = serde_json::from_value::<TyRef>(adt(1, 21)).expect("ZST enum TyRef");
    let mixed_ty = serde_json::from_value::<TyRef>(adt(2, 22)).expect("mixed enum TyRef");

    assert!(super::tyref_is_zero_sized(&phantom_ty, &llbc));
    assert!(super::type_decl_is_fieldless_enum(
        llbc.type_by_id(1).expect("OnlyZst declaration"),
        &llbc
    ));
    assert!(!super::type_decl_is_fieldless_enum(
        llbc.type_by_id(2).expect("Mixed declaration"),
        &llbc
    ));
    assert_eq!(
        super::tyref_to_value_type(&only_zst_ty, &llbc),
        ValueType::Int
    );
    assert_eq!(
        super::tyref_to_value_type(&mixed_ty, &llbc),
        ValueType::Ref(Some("Mixed".to_string()))
    );
}

/// A mixed enum (fieldless variants plus a payload-carrying one) is a
/// `SomeInstance` of its interned enum-base classdef, not the untyped
/// `Ref(None)` fallback.  The intern key is the ADT leaf
/// `tyref_class_root` / `Input.class_root` already use, so
/// `getuniqueclassdef_for_struct_root` and `__cast_instance_intrinsic`
/// resolve the same class object the bookkeeper intern's for variant
/// subclasses.  A fieldless-only sibling stays `Int`.
#[test]
fn payload_carrying_enum_paints_typed_ref_of_intern_leaf() {
    let span = || {
        serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }})
    };
    let item_meta = |name: &str| {
        serde_json::json!({
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": [name, 0]}
            ],
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        })
    };
    let adt = |def_id: u64, hash: u64| {
        serde_json::json!({"Value": [hash, {
            "Adt": {
                "id": def_id,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }]})
    };
    let field = |name: &str, ty: serde_json::Value| serde_json::json!({"name": name, "ty": ty, "attr_info": null});
    let variant = |name: &str, fields: Vec<serde_json::Value>, discriminant: u64| {
        serde_json::json!({
            "name": name,
            "fields": fields,
            "discriminant": {"Scalar": {"Unsigned": ["U8", discriminant.to_string()]}}
        })
    };
    let layout = |size: u64, field_offsets: Vec<u64>| {
        serde_json::json!([{
            "key": "fixture-target",
            "value": {
                "size": size,
                "align": 1,
                "variant_layouts": [{"field_offsets": field_offsets}],
                "repr": {"transparent": false}
            }
        }])
    };
    // Fieldless-only sibling: every variant is payload-free, so the
    // value is the discriminant integer.
    let fieldless = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta("AccessMode"),
        "kind": {"Enum": [
            variant("ReadWrite", vec![], 0),
            variant("ReadOnly", vec![], 1)
        ]},
        "layout": layout(1, vec![])
    });
    // Mixed: fieldless variants plus one payload-carrying variant.
    // Shape of `Code` (ten unit variants + `Int { signed }`) without
    // naming that enum — the paint is driven by the payload, not the
    // ident.
    let mixed = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta("Opcode"),
        "kind": {"Enum": [
            variant("Nop", vec![], 0),
            variant("LoadConst", vec![], 1),
            variant(
                "Int",
                vec![field("signed", serde_json::json!({"Scalar": "Bool"}))],
                2
            )
        ]},
        "layout": layout(2, vec![1])
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [fieldless, mixed],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let fieldless_ty = serde_json::from_value::<TyRef>(adt(0, 10)).expect("fieldless TyRef");
    let mixed_ty = serde_json::from_value::<TyRef>(adt(1, 11)).expect("mixed TyRef");

    assert!(super::type_decl_is_fieldless_enum(
        llbc.type_by_id(0).expect("AccessMode declaration"),
        &llbc
    ));
    assert!(!super::type_decl_is_fieldless_enum(
        llbc.type_by_id(1).expect("Opcode declaration"),
        &llbc
    ));
    assert_eq!(
        super::tyref_to_value_type(&fieldless_ty, &llbc),
        ValueType::Int
    );
    assert_eq!(
        super::tyref_to_value_type(&mixed_ty, &llbc),
        ValueType::Ref(Some("Opcode".to_string())),
        "payload-carrying enum must paint Ref(Some(leaf)), not Ref(None)"
    );
    assert_eq!(
        tyref_to_attr_value_type(&mixed_ty, &llbc),
        ValueType::Ref(Some("Opcode".to_string())),
        "FORCE-attr paint must agree with the value-site paint"
    );
    assert_eq!(
        super::tyref_class_root(&mixed_ty, &llbc).as_deref(),
        Some("Opcode"),
        "intern key must be the Input.class_root leaf"
    );
}

/// `eval::Code` and `module::struct::Code` share a leaf. Harden clears
/// that leaf's origin, and the bare token is not a class. The enum's
/// value must paint the declaration that still has the
/// `__discriminant` row (`module::struct::Code`), while a leaf harden
/// left alone stays bare.
#[test]
fn tombstoned_duplicate_leaf_paints_the_declaration_path() {
    let span = || {
        serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }})
    };
    let item_meta = |segs: &[&str]| {
        serde_json::json!({
            "name": segs.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        })
    };
    let adt = |def_id: u64, hash: u64| {
        serde_json::json!({"Value": [hash, {
            "Adt": {
                "id": def_id,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }]})
    };
    let variant = |name: &str, fields: Vec<serde_json::Value>, discriminant: u64| {
        serde_json::json!({
            "name": name,
            "fields": fields,
            "discriminant": {"Scalar": {"Unsigned": ["U8", discriminant.to_string()]}}
        })
    };
    let field = |name: &str| {
        serde_json::json!({
            "name": name,
            "ty": {"Scalar": "Bool"},
            "attr_info": null
        })
    };
    let enum_code = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(&["pyre_interpreter", "module", "struct", "Code"]),
        "kind": {"Enum": [
            variant("Pad", vec![], 0),
            variant("Int", vec![field("signed")], 1)
        ]}
    });
    let struct_code = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(&["pyre_interpreter", "eval", "Code"]),
        "kind": {"Struct": [field("name")]}
    });
    let opcode = serde_json::json!({
        "def_id": 2,
        "item_meta": item_meta(&["fixture", "Opcode"]),
        "kind": {"Enum": [
            variant("Nop", vec![], 0),
            variant("Int", vec![field("signed")], 1)
        ]}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "pyre_interpreter",
            "type_decls": [enum_code, struct_code, opcode],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let code_ty = serde_json::from_value::<TyRef>(adt(0, 30)).expect("Code TyRef");
    let opcode_ty = serde_json::from_value::<TyRef>(adt(2, 31)).expect("Opcode TyRef");
    let tombstoned = ["Code".to_string()].into_iter().collect();
    assert_eq!(
        super::tyref_to_value_type_with(
            &code_ty,
            &llbc,
            &tombstoned,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Ref(Some("module::struct::Code".to_string()))
    );
    assert_eq!(
        super::tyref_class_root_with(&code_ty, &llbc, &tombstoned).as_deref(),
        Some("module::struct::Code")
    );
    assert_eq!(
        super::tyref_to_value_type(&opcode_ty, &llbc),
        ValueType::Ref(Some("Opcode".to_string())),
        "a leaf harden did not clear keeps its bare intern key"
    );
}

/// `rclass.InstanceRepr._setup_repr` declares a subclass as
/// `MkStruct(name, ('super', rbase.object_type), *llfields)`, so
/// `heaptracker.all_fielddescrs` walks the embedded base first and the
/// inherited field holds the same index in the subclass as in the base.
/// A shelled variant's one inherited field is the base's synthetic
/// `__discriminant`, so it must be the variant subclass's own first row —
/// otherwise the payload row claims slot 0, the slot the tag's descr also
/// claims, and a virtual's payload store replaces the tag store.
///
/// A payload-less variant registers no rows at all: a lone
/// `__discriminant` row is the sentinel `StructFieldRegistry::is_enum_base`
/// reads to tell an enum base from a variant subclass.
#[test]
fn a_shelled_variant_carries_the_inherited_discriminant_as_its_first_row() {
    let span = || {
        serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }})
    };
    let item_meta = |name: &str| {
        serde_json::json!({
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": [name, 0]}
            ],
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        })
    };
    let int_field = |name: &str| {
        serde_json::json!({
            "name": name,
            "ty": {"Scalar": {"Integer": {"Signed": "I64"}}},
            "attr_info": null
        })
    };
    let variant = |name: &str, fields: Vec<serde_json::Value>, discriminant: u64| {
        serde_json::json!({
            "name": name,
            "fields": fields,
            "discriminant": {"Scalar": {"Unsigned": ["U8", discriminant.to_string()]}}
        })
    };
    // No `layout` entry at all — the case that hid this defect.  The
    // variant offsets then fall back to packing from byte 0, which is the
    // inherited discriminant's byte, so `payload_seats_on_tag` takes its
    // `None` arm and the enum is shelled.
    let step = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta("Step"),
        "kind": {"Enum": [
            variant("Continue", vec![], 0),
            variant(
                "CloseLoop",
                vec![int_field("loop_header_pc")],
                1
            )
        ]},
        "layout": []
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [step],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let (_, _, fields, _, _, _, exact_layouts, _) = super::derive_program_metadata(
        &llbc,
        &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc)),
        &[],
    );

    assert_eq!(
        fields.fields.get("fixture::Step::CloseLoop"),
        Some(&vec![
            crate::front::semantic::FieldRow::named("__discriminant", "i64"),
            crate::front::semantic::FieldRow::named("loop_header_pc", "i64"),
        ]),
        "the payload rows must follow the inherited tag, not start at slot 0"
    );
    assert_eq!(
        fields.fields.get("Step::CloseLoop"),
        fields.fields.get("fixture::Step::CloseLoop"),
        "the dual-published spellings must carry the same rows"
    );
    assert_eq!(
        fields.fields.get("fixture::Step::Continue"),
        Some(&vec![]),
        "a payload-less variant inherits nothing to seat and stays row-free"
    );
    assert_eq!(
        fields.fields.get("fixture::Step"),
        Some(&vec![crate::front::semantic::FieldRow::named(
            "__discriminant",
            "i64"
        )]),
        "the base still carries the tag alone"
    );

    let variant_sid = majit_ir::descr::StructId::from_canonical("Step::CloseLoop");
    let offsets = &exact_layouts
        .get(&variant_sid)
        .expect("shelled variant registers an exact layout")
        .field_offsets;
    assert_eq!(offsets.get("__discriminant"), Some(&0));
    assert_eq!(offsets.get("loop_header_pc"), Some(&8));
}

/// A pointer-niche payload enum (`CodePageFail { NoTranslation, Os(PyError) }`
/// with `PyError = NonNull<PyObject>`) records its tag as `Branch.int_ty =
/// Isize`. That width has a field-type spelling, so the base registers
/// `__discriminant` and the payload at byte 0 seats on the tag — the same
/// explicit shell a missing layout takes.
#[test]
fn a_pointer_niche_payload_enum_takes_the_explicit_shell() {
    let span = || {
        serde_json::json!({"data": {
            "file_id": 0,
            "beg": {"line": 1, "col": 0},
            "end": {"line": 1, "col": 1}
        }})
    };
    let item_meta = |name: &str| {
        serde_json::json!({
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": [name, 0]}
            ],
            "span": span(),
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": true
        })
    };
    let ptr_field = |name: &str| {
        serde_json::json!({
            "name": name,
            "ty": {"RawPtr": [
                {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "Mut"
            ]},
            "attr_info": null
        })
    };
    let variant = |name: &str, fields: Vec<serde_json::Value>, discriminant: u64| {
        serde_json::json!({
            "name": name,
            "fields": fields,
            "discriminant": {"Scalar": {"Unsigned": ["U8", discriminant.to_string()]}}
        })
    };
    let fail = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta("Fail"),
        "kind": {"Enum": [
            variant("NoTranslation", vec![], 0),
            variant("Os", vec![ptr_field("os")], 1)
        ]},
        "layout": [{
            "key": "fixture-target",
            "value": {
                "size": 8,
                "align": 8,
                "discriminator": {"Branch": {
                    "offset": 0,
                    "int_ty": {"Signed": "Isize"},
                    "children": [[{"start": 0, "end": 0}, {"Known": 0}]],
                    "fallback": {"Known": 1}
                }},
                "variant_layouts": [
                    {"field_offsets": []},
                    {"field_offsets": [0]}
                ],
                "repr": {"transparent": false}
            }
        }]
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [fail],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let (_, _, fields, _, _, _, exact_layouts, _) = super::derive_program_metadata(
        &llbc,
        &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(&llbc)),
        &[],
    );

    assert_eq!(
        fields.fields.get("fixture::Fail"),
        Some(&vec![crate::front::semantic::FieldRow::named(
            "__discriminant",
            "isize"
        )]),
        "the spelled Isize tag is the base's one __discriminant row"
    );

    let os_rows = fields
        .fields
        .get("fixture::Fail::Os")
        .expect("payload variant registers rows");
    assert_eq!(
        os_rows
            .first()
            .map(|row| (row.name.as_str(), row.ty.as_str())),
        Some(("__discriminant", "isize")),
        "the payload rows must start with the inherited tag, not the pointer field"
    );

    let variant_sid = majit_ir::descr::StructId::from_canonical("Fail::Os");
    let offsets = &exact_layouts
        .get(&variant_sid)
        .expect("shelled variant registers an exact layout")
        .field_offsets;
    assert_eq!(offsets.get("__discriminant"), Some(&0));
    let payload_off = offsets.get("os").copied();
    assert!(
        payload_off.is_some_and(|off| off != 0),
        "the payload must not alias the tag at byte 0, got {payload_off:?}"
    );
}

/// An `ArrayRead` element is addressable two ways: a scalar names its
/// spelling so the descr computes the real width, or a thin pointer takes
/// the identity-less mint's one-target-word assumption, which is already
/// right for it. Anything else — including a scalar `get_type_flag` has no
/// row for — is not addressable and the site stays residual.
#[test]
fn a_scalar_element_names_its_width_and_a_thin_pointer_needs_no_name() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let spelling = |ty: serde_json::Value| json_ty_scalar_element_spelling(&ty, &llbc);
    let uint = |w: &str| serde_json::json!({"Scalar": {"Integer": {"Unsigned": w}}});
    let int = |w: &str| serde_json::json!({"Scalar": {"Integer": {"Signed": w}}});
    let named_adt = serde_json::json!({"Adt": {"id": 7, "generics": {"types": []}}});

    // Every width `get_type_flag` computes, narrow ones included — these
    // are the `bytearray` / `rbigint` digit / `rutf8` index populations
    // that `ValueType::Unsigned` cannot tell apart from `usize`.
    for (charon, rust) in [
        ("U8", "u8"),
        ("U16", "u16"),
        ("U32", "u32"),
        ("U64", "u64"),
        ("Usize", "usize"),
    ] {
        assert_eq!(spelling(uint(charon)).as_deref(), Some(rust));
    }
    assert_eq!(spelling(int("I16")).as_deref(), Some("i16"));
    assert_eq!(spelling(int("Isize")).as_deref(), Some("isize"));
    assert_eq!(
        spelling(serde_json::json!({"Scalar": {"Float": "F32"}})).as_deref(),
        Some("f32")
    );
    assert_eq!(
        spelling(serde_json::json!({"Scalar": "Bool"})).as_deref(),
        Some("bool")
    );

    // `char` is `UniChar`: named, so the descr strides by its 4 bytes
    // instead of the identity-less word. The object form is
    // `{"Char": "<codepoint>"}`.
    assert_eq!(
        spelling(serde_json::json!({"Scalar": "Char"})).as_deref(),
        Some("char")
    );
    assert_eq!(
        spelling(serde_json::json!({"Scalar": {"Char": "97"}})).as_deref(),
        Some("char")
    );
    assert_eq!(
        crate::codewriter::call::get_type_flag("char", &std::collections::HashSet::new()),
        (
            majit_ir::descr::ArrayFlag::Unsigned,
            majit_ir::value::Type::Int,
            4
        )
    );
    // A named ADT is the element itself, not a pointer to one, so its size
    // is whatever the struct is — `String` is three words. Packed Copy
    // detection needs the TypeDecl field list; a missing decl declines.
    assert_eq!(spelling(named_adt.clone()), None);
    assert_eq!(
        json_ty_packed_copy_struct_byte_size(&named_adt, &llbc),
        None
    );

    let thin = |ty: serde_json::Value| json_ty_is_thin_pointer_element(&ty, &llbc);
    assert!(thin(
        serde_json::json!({"Ref": ["_", named_adt.clone(), "Shared"]})
    ));
    assert!(thin(serde_json::json!({"RawPtr": [named_adt, "Mut"]})));
    assert!(!thin(uint("U8")));
}

fn named_copy_field(name: &str, ty: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"name": name, "ty": ty, "attr_info": null})
}

#[test]
fn json_ty_is_by_value_copy_aggregate_accepts_named_copy_structs() {
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let pair = type_decl_struct(
        0,
        &["fixture", "CodePair"],
        vec![
            named_copy_field("op", u8_ty.clone()),
            named_copy_field("arg", u8_ty.clone()),
        ],
    );
    let unsafe_cell = type_decl_struct(
        1,
        &["core", "cell", "UnsafeCell"],
        vec![named_copy_field(
            "value",
            serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}}),
        )],
    );
    let atomic = type_decl_struct(
        2,
        &["core", "sync", "atomic", "AtomicU64"],
        vec![named_copy_field("v", adt_node(1))],
    );
    let unique = type_decl_struct(
        3,
        &["core", "ptr", "unique", "Unique"],
        vec![named_copy_field(
            "pointer",
            serde_json::json!({"RawPtr": [u8_ty.clone(), "Mut"]}),
        )],
    );
    let entry = type_decl_struct(
        4,
        &["fixture", "Entry"],
        vec![named_copy_field("key", u8_ty.clone())],
    );
    let option = option_enum(5);
    let arg_marker = opaque_type_decl(
        6,
        &["rustpython_compiler_core", "bytecode", "instruction", "Arg"],
    );
    let instruction = serde_json::json!({
        "def_id": 7,
        "item_meta": {
            "name": [
                {"Ident": ["rustpython_compiler_core", 0]},
                {"Ident": ["bytecode", 0]},
                {"Ident": ["instruction", 0]},
                {"Ident": ["Instruction", 0]}
            ],
            "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}},
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Enum": [
            {"name": "Nop", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}},
            {"name": "LoadName", "fields": [{
                "name": null,
                "ty": adt_node(6),
                "attr_info": null
            }], "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}}
        ]}
    });
    let code_unit = type_decl_struct(
        8,
        &["rustpython_compiler_core", "bytecode", "CodeUnit"],
        vec![
            named_copy_field("op", adt_node(7)),
            named_copy_field("arg", u8_ty.clone()),
        ],
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![
            pair,
            unsafe_cell,
            atomic,
            unique,
            entry,
            option,
            arg_marker,
            instruction,
            code_unit,
        ],
        vec![],
    );
    assert!(
        json_ty_is_by_value_copy_aggregate(&adt_node(0), &llbc),
        "a two-field Copy struct is the getarrayitem item"
    );
    assert!(
        !json_ty_is_by_value_copy_aggregate(&adt_node(2), &llbc),
        "AtomicU64 is interior-mutable; get must not copy the slot"
    );
    assert!(
        !json_ty_is_by_value_copy_aggregate(&adt_node(3), &llbc),
        "a pointer field is not a by-value Copy item"
    );
    let option_ty = serde_json::json!({
        "Adt": {
            "id": 5,
            "generics": {
                "regions": [],
                "types": [adt_node(4)],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    assert!(
        !json_ty_is_by_value_copy_aggregate(&option_ty, &llbc),
        "Option<Entry> is an enum slot, not a Copy struct item"
    );
    assert!(
        !json_ty_is_by_value_copy_aggregate(&u8_ty, &llbc),
        "a scalar stays on the existing spelling arm"
    );
    assert!(
        json_ty_is_by_value_copy_aggregate(&adt_node(8), &llbc),
        "CodeUnit is a Copy struct of a C-like opcode enum plus a byte"
    );
}

#[test]
fn json_ty_is_by_value_copy_aggregate_rejects_interior_mut_and_heap_payloads() {
    let atomic = type_decl_struct(
        0,
        &["core", "sync", "atomic", "AtomicU64"],
        vec![named_copy_field(
            "v",
            serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}}),
        )],
    );
    let llbc = llbc_with_types("fixture", vec![atomic], vec![]);
    assert!(!json_ty_is_by_value_copy_aggregate(&adt_node(0), &llbc));
}

#[test]
fn unique_unsafecell_slice_deref_field_index_picks_the_cell_slice() {
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let pair = type_decl_struct(
        0,
        &["fixture", "CodePair"],
        vec![
            named_copy_field("op", u8_ty.clone()),
            named_copy_field("arg", u8_ty.clone()),
        ],
    );
    let typevar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let unsafecell = type_decl_struct(
        1,
        &["core", "cell", "UnsafeCell"],
        vec![named_copy_field("value", typevar)],
    );
    let box_slice = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [{"Slice": [adt_node(0), null]}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cell_ty = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [box_slice],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let holder = type_decl_struct(
        2,
        &["fixture", "CellPairUnits"],
        vec![
            named_copy_field("units", cell_ty),
            named_copy_field("pad", u8_ty),
        ],
    );
    let llbc = llbc_with_types("fixture", vec![pair, unsafecell, holder], vec![]);
    let recv = TyRef::Other(serde_json::json!({
        "Ref": ["_", {
            "Adt": {
                "id": 2,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }, "Shared"]
    }));
    let dest = TyRef::Other(serde_json::json!({
        "Ref": ["_", {"Slice": [adt_node(0), null]}, "Shared"]
    }));
    assert_eq!(
        unique_unsafecell_slice_deref_field_index(&recv, &dest, &llbc),
        Some(0)
    );
}

/// A slice of a unique two-segment Copy struct keeps the TypeDecl leaf.
/// `strip_crate_prefix("fixture::CodePair")` is `CodePair`, so Place Index
/// `array_type_id` stays `[CodePair]`.
#[test]
fn unique_leaf_copy_struct_slice_keeps_leaf_spelling() {
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let pair = type_decl_struct(
        0,
        &["fixture", "CodePair"],
        vec![
            named_copy_field("op", u8_ty.clone()),
            named_copy_field("arg", u8_ty),
        ],
    );
    let llbc = llbc_with_types("fixture", vec![pair], vec![]);
    let slice = TyRef::Other(serde_json::json!({"Slice": [adt_node(0), null]}));
    assert_eq!(super::tyref_to_ast_string(&slice, &llbc), "[CodePair]");
}

/// Two TypeDecls share the leaf `CopyInstrWord`. Harvest of
/// `UnsafeCell<Box<[CopyInstrWord]>>` qualifies the slice element
/// (`field_row_struct_identity`) as the crate-stripped identity, the
/// same identity ArrayRead / InteriorFieldRead use
/// (`adt_node_class_root_identity`). `tyref_to_ast_string` keeps the
/// TypeDecl leaf. Projecting that field is a SomeList whose item is
/// that instance, so getitem (`listdef.py` `read_item`) is not
/// Impossible.
#[test]
fn harvested_copy_struct_slice_item_is_class_identity_not_duplicate_leaf() {
    use crate::annotator::model::SomeValue;
    use std::rc::Rc;

    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let bytecode_word = type_decl_struct(
        0,
        &["fixture", "bytecode", "CopyInstrWord"],
        vec![
            named_copy_field("op", u8_ty.clone()),
            named_copy_field("arg", u8_ty.clone()),
        ],
    );
    let gc_word = type_decl_struct(
        1,
        &["fixture", "gc", "CopyInstrWord"],
        vec![
            named_copy_field("wrappers", usize_ty.clone()),
            named_copy_field("pinned", usize_ty),
        ],
    );
    let typevar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let unsafecell = type_decl_struct(
        2,
        &["core", "cell", "UnsafeCell"],
        vec![named_copy_field("value", typevar)],
    );
    let box_slice = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [{"Slice": [adt_node(0), null]}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cell_ty = serde_json::json!({
        "Adt": {
            "id": 2,
            "generics": {
                "regions": [],
                "types": [box_slice],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let holder = type_decl_struct(
        3,
        &["fixture", "CellInstrUnits"],
        vec![
            named_copy_field("units", cell_ty),
            named_copy_field("pad", u8_ty),
        ],
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![bytecode_word, gc_word, unsafecell, holder],
        vec![],
    );

    let slice = TyRef::Other(serde_json::json!({"Slice": [adt_node(0), null]}));
    assert_eq!(
        super::tyref_to_ast_string(&slice, &llbc),
        "[CopyInstrWord]",
        "tyref_to_ast_string keeps the TypeDecl leaf"
    );
    let tombstoned = super::tombstoned_leaves_of(&llbc);
    assert!(
        tombstoned.contains("CopyInstrWord"),
        "duplicate leaf is withdrawn; tombstoned={tombstoned:?}"
    );
    assert_eq!(
        adt_node_class_root_identity(&adt_node(0), &llbc, &tombstoned).as_deref(),
        Some("bytecode::CopyInstrWord"),
        "ArrayRead / InteriorFieldRead FLAG_STRUCT item identity"
    );

    let program = crate::test_support::build_semantic_program_from_llbc(&llbc)
        .expect("harvest CellInstrUnits");
    let units_ty = program
        .struct_fields
        .fields
        .get("CellInstrUnits")
        .or_else(|| program.struct_fields.fields.get("fixture::CellInstrUnits"))
        .and_then(|rows| rows.iter().find(|row| row.name == "units"))
        .map(|row| row.ty.clone())
        .unwrap_or_default();
    assert!(
        units_ty.contains("[bytecode::CopyInstrWord]"),
        "harvested units field carries the ArrayRead item identity; units_ty={units_ty:?}; keys={:?}",
        program.struct_fields.fields.keys().collect::<Vec<_>>()
    );
    assert!(
        program.struct_fields.fields.get("CopyInstrWord").is_none(),
        "harden withdraws the duplicate leaf; keys={:?}",
        program.struct_fields.fields.keys().collect::<Vec<_>>()
    );

    let bk = Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
    bk.set_struct_fields(Rc::new(program.struct_fields));
    let projected = bk.project_struct_field_type(&units_ty);
    let SomeValue::List(list) = &projected else {
        panic!("units projects to SomeList, got {projected:?}");
    };
    let item = list.listdef.read_item(None);
    assert!(
        !matches!(item, SomeValue::Impossible),
        "getitem read_item must not be Impossible; SomeList={list:?} item={item:?}"
    );
    let SomeValue::Instance(inst) = &item else {
        panic!("list item is SomeInstance of the Copy struct, got {item:?}");
    };
    let classdef = inst.classdef.as_ref().expect("named instance");
    assert_eq!(
        classdef.borrow().name,
        "bytecode::CopyInstrWord",
        "list item matches ArrayRead class identity"
    );
    let _ = bk
        .getuniqueclassdef_for_struct_root("bytecode::CopyInstrWord")
        .expect("element classdef projects");
    assert!(
        classdef.borrow().attrs.contains_key("arg"),
        "element rows are the bytecode CopyInstrWord fields; attrs={:?}",
        classdef.borrow().attrs.keys().collect::<Vec<_>>()
    );
}

/// `Vec::index_mut` of `Vec<Box<T>>` is one word per element when `T` has
/// a concrete size. `IndexMut::index_mut` then lowers to `ArrayRead`
/// (`getarrayitem`) instead of staying a residual call.
#[test]
fn a_box_of_a_sized_adt_is_a_thin_pointer_element() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = serde_json::json!({
        "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Payload", 0]}],
        "span": span, "source_text": null,
        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
        "is_local": true
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": meta,
                "kind": {"Struct": []},
                "layout": [{
                    "key": "fixture-target",
                    "value": {
                        "size": 16,
                        "align": 8,
                        "variant_layouts": [{"field_offsets": []}],
                        "repr": {"transparent": false}
                    }
                }]
            }],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let payload = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        }
    });
    let boxed = serde_json::json!({
        "Adt": {
            "id": 1, "builtin": "Box",
            "generics": {"regions": [], "types": [payload], "const_generics": [], "trait_refs": []}
        }
    });
    assert!(
        json_ty_is_thin_pointer_element(&boxed, &llbc),
        "Box<sized ADT> is one pointer word, so index_mut can emit ArrayRead"
    );
    let fat_box = serde_json::json!({
        "Adt": {
            "id": 1, "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [{"Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, null]}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    assert!(
        !json_ty_is_thin_pointer_element(&fat_box, &llbc),
        "Box<[u8]> is a fat pointer and must not take the one-word stride"
    );
}

#[test]
fn field_layout_keeps_reference_repr_instead_of_embedding_the_referent() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let shared = super::TyRef::Other(serde_json::json!({
        "Ref": ["_", {"Scalar": {"Integer": {"Signed": "I64"}}}, "Shared"]
    }));
    let mutable = super::TyRef::Other(serde_json::json!({
        "Ref": ["_", {"Scalar": {"Integer": {"Signed": "I64"}}}, "Mut"]
    }));
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();

    assert_eq!(
        super::tyref_to_field_layout_string(
            &shared,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "&i64"
    );
    assert_eq!(
        super::tyref_to_field_layout_string(
            &mutable,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "&mut i64"
    );
}

/// CAST intern suffixes keep `&T` / `&mut T` (`charon_type_value_to_enum_arg`).
/// Class-root lookup still strips the referent (`charon_type_value_to_ast_string`).
/// Stripping the CAST suffix made `Option<&T>` intern as the tagged
/// `Option<T>` class (`model.py` `SomeObject.noneify`).
#[test]
fn charon_ref_renders_ampersand_prefix() {
    let llbc = empty_llbc();
    let tomb = no_tombstoned_leaves();
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let shared = serde_json::json!({"Ref": ["Erased", i64_ty.clone(), "Shared"]});
    let mutable = serde_json::json!({"Ref": ["Erased", i64_ty, "Mut"]});
    assert_eq!(
        charon_type_value_to_enum_arg(&shared, &llbc, 0, &tomb),
        "&i64"
    );
    assert_eq!(
        charon_type_value_to_enum_arg(&mutable, &llbc, 0, &tomb),
        "&mut i64"
    );
    assert_eq!(charon_type_value_to_ast_string(&shared, &llbc, 0), "i64");
    assert_eq!(charon_type_value_to_ast_string(&mutable, &llbc, 0), "i64");
}

/// Dest intern KEY of a generic ADT keeps `&` only when the referent is
/// raw storage (`adt_def_is_raw_storage`). `Option<&Raw>` is then not
/// the tagged `Option<Raw>` class (`model.py` `SomeObject.noneify`),
/// and not `Option<i64>`. A scalar `&i64` is the integer itself, so
/// the prefix is erased. Top-level `&T` still strips.
#[test]
fn option_of_ref_to_raw_storage_keeps_ampersand() {
    let raw = struct_decl(
        1,
        ident_path(&["fixture", "RawCell"]),
        serde_json::json!([{
            "name": "word",
            "ty": {"Scalar": {"Integer": {"Signed": "I32"}}},
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types("fixture", vec![option_enum(0), raw], vec![]);
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let opt_ref_raw = option_adt_node(0, option_shared_ref(adt_node(1)));
    let opt_raw = option_adt_node(0, adt_node(1));
    let opt_ref_i64 = option_adt_node(0, option_shared_ref(i64_ty.clone()));
    let opt_i64 = option_adt_node(0, i64_ty.clone());
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(opt_ref_raw), &llbc),
        "Option<&RawCell>",
        "dest intern KEY of Option<&Raw> keeps &"
    );
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(opt_raw), &llbc),
        "Option<RawCell>",
        "by-value Option<Raw> is the tagged class"
    );
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(opt_ref_i64), &llbc),
        "Option<i64>",
        "dest intern KEY of Option<&i64> erases scalar &"
    );
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(opt_i64), &llbc),
        "Option<i64>",
        "Option<i64> dest intern KEY is the tagged integer Option"
    );
    assert_eq!(
        charon_type_value_to_ast_string(&option_shared_ref(i64_ty), &llbc, 0),
        "i64",
        "top-level &T still strips for class-root lookup"
    );
}

/// Variant payload `&mut Raw` is the interior pointer word
/// (`history.getkind` of `Ptr` when `TO._gckind == raw`). Parameter dest
/// overlay of the same borrow stays `Ref` (`ty_is_raw_address` /
/// `RootScope::get`).
#[test]
fn enum_payload_of_ref_to_raw_storage_is_int() {
    let raw = struct_decl(
        1,
        ident_path(&["fixture", "RawCell"]),
        serde_json::json!([{
            "name": "word",
            "ty": {"Scalar": {"Integer": {"Signed": "I32"}}},
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types("fixture", vec![option_enum(0), raw], vec![]);
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let tomb = no_tombstoned_leaves();
    let flavors = std::collections::HashSet::<String>::new();
    let mut_ref = TyRef::Other(serde_json::json!({
        "Ref": ["Erased", adt_node(1), "Mut"]
    }));
    assert_eq!(
        tyref_to_value_type(&mut_ref, &llbc),
        ValueType::Ref(None),
        "a borrowed raw struct parameter stays a reference"
    );
    assert_eq!(
        tyref_enum_payload_value_type(&mut_ref, &llbc, tomb, &gc, &flavors),
        ValueType::Int,
        "Result Ok / Option Some of &mut Raw is the interior pointer word"
    );
    assert_eq!(
        adt_field_read_value_type(
            &mut_ref, &mut_ref, true, false, None, &llbc, tomb, &gc, &flavors
        ),
        ValueType::Int,
        "enum payload FieldRead of &mut Raw follows intern ROW Int"
    );
    assert_eq!(
        adt_field_read_value_type(
            &mut_ref, &mut_ref, false, true, None, &llbc, tomb, &gc, &flavors
        ),
        ValueType::Int,
        "closure-env capture of &mut Raw is the interior pointer word"
    );
}

/// Closure-argument tuple of `&Raw` keeps `&` so `and_then` write and
/// `call_once` read share `Tuple<&RawCell>` (`render_adt_type_args`).
#[test]
fn payload_tuple_suffix_of_ref_to_raw_storage_keeps_ampersand() {
    let raw = struct_decl(
        0,
        ident_path(&["fixture", "RawCell"]),
        serde_json::json!([{
            "name": "word",
            "ty": {"Scalar": {"Integer": {"Signed": "I32"}}},
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types("fixture", vec![raw], vec![]);
    let payload = TyRef::Other(option_shared_ref(adt_node(0)));
    assert_eq!(
        payload_tuple_suffix(&payload, &llbc),
        "<&RawCell>",
        "Tuple intern KEY of &Raw keeps &"
    );
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let scalar = TyRef::Other(option_shared_ref(i64_ty));
    assert_eq!(
        payload_tuple_suffix(&scalar, &llbc),
        "<i64>",
        "Tuple intern KEY of &i64 erases scalar &"
    );
}

/// Charon `monomorphize:false`: a field typed `Option<Raw>` is the
/// Option template plus type args. The template (TypeVar payload) is
/// not raw storage; the instantiation is, so the owner is too.
#[test]
fn option_instantiation_of_raw_payload_is_a_raw_word() {
    let option = option_enum(0);
    let raw = struct_decl(
        1,
        ident_path(&["fixture", "RawCell"]),
        serde_json::json!([{
            "name": "word",
            "ty": {"Scalar": {"Integer": {"Signed": "I32"}}},
            "attr_info": null
        }]),
        false,
    );
    let owner = struct_decl(
        2,
        ident_path(&["fixture", "OwnerBag"]),
        serde_json::json!([{
            "name": "cell",
            "ty": option_adt_node(0, adt_node(1)),
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types("fixture", vec![option, raw, owner], vec![]);
    let gc = std::collections::HashSet::new();
    let immut = no_immutable_fields();
    assert!(
        !adt_def_is_raw_storage(0, &llbc, &gc, immut, &mut Vec::new()),
        "Option TypeDecl template (TypeVar payload) is not raw storage"
    );
    let inst = TyRef::Other(option_adt_node(0, adt_node(1)));
    assert!(
        field_ty_is_raw_word(&inst, &llbc, &gc, immut, &mut Vec::new()),
        "Option<RawCell> instantiation is a raw word"
    );
    assert!(
        adt_def_is_raw_storage(2, &llbc, &gc, immut, &mut Vec::new()),
        "owner of Option<RawCell> is raw storage"
    );
}

/// Nested `Option<Option<&T>>` re-enters the Option def by value.
/// `lltype.py` `Struct._first_struct` classifies from the declared field
/// types, so a GC referent is not raw storage and a raw referent is.
#[test]
fn option_of_option_of_ref_is_not_raw_storage_for_gc_referent() {
    let option = option_enum(0);
    let gc_t = struct_decl(
        1,
        ident_path(&["pyre_object", "pyobject", "PyObject"]),
        serde_json::json!([]),
        false,
    );
    let raw = struct_decl(
        2,
        ident_path(&["fixture", "RawCell"]),
        serde_json::json!([{
            "name": "word",
            "ty": {"Scalar": {"Integer": {"Signed": "I32"}}},
            "attr_info": null
        }]),
        false,
    );
    let nested = |inner: serde_json::Value| option_adt_node(0, option_adt_node(0, inner));
    let owner_gc = struct_decl(
        3,
        ident_path(&["fixture", "GcSlot"]),
        serde_json::json!([{
            "name": "slot",
            "ty": nested(option_shared_ref(adt_node(1))),
            "attr_info": null
        }]),
        false,
    );
    let owner_raw = struct_decl(
        4,
        ident_path(&["fixture", "RawSlot"]),
        serde_json::json!([{
            "name": "slot",
            "ty": nested(option_shared_ref(adt_node(2))),
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![option, gc_t, raw, owner_gc, owner_raw],
        vec![],
    );
    let gc = std::collections::HashSet::new();
    let immut = no_immutable_fields();
    let inst_gc = TyRef::Other(nested(option_shared_ref(adt_node(1))));
    let inst_raw = TyRef::Other(nested(option_shared_ref(adt_node(2))));
    assert!(
        !field_ty_is_raw_word(&inst_gc, &llbc, &gc, immut, &mut Vec::new()),
        "Option<Option<&PyObject>> is not a raw word"
    );
    assert!(
        !adt_def_is_raw_storage(3, &llbc, &gc, immut, &mut Vec::new()),
        "owner of Option<Option<&PyObject>> is not raw storage"
    );
    assert!(
        field_ty_is_raw_word(&inst_raw, &llbc, &gc, immut, &mut Vec::new()),
        "Option<Option<&RawCell>> is a raw word"
    );
    assert!(
        adt_def_is_raw_storage(4, &llbc, &gc, immut, &mut Vec::new()),
        "owner of Option<Option<&RawCell>> is raw storage"
    );
}

/// Scalar-only type args do not substitute: `Option<i64>` stays the
/// TypeDecl template (TypeVar payload), so an owner of that field is
/// not a raw-word owner (`getfield_gc` of the tagged Instance).
#[test]
fn option_instantiation_of_scalar_payload_is_not_a_raw_word() {
    let option = option_enum(0);
    let owner = struct_decl(
        1,
        ident_path(&["fixture", "OwnerBag"]),
        serde_json::json!([{
            "name": "slot",
            "ty": option_adt_node(0, serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})),
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types("fixture", vec![option, owner], vec![]);
    let gc = std::collections::HashSet::new();
    let immut = no_immutable_fields();
    let inst = TyRef::Other(option_adt_node(
        0,
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}}),
    ));
    assert!(
        !field_ty_is_raw_word(&inst, &llbc, &gc, immut, &mut Vec::new()),
        "Option<i64> instantiation stays the tagged Instance intern"
    );
    assert!(
        !adt_def_is_raw_storage(1, &llbc, &gc, immut, &mut Vec::new()),
        "owner of Option<i64> is not raw storage"
    );
}

/// `Atomic<T>` is Opaque; the stored word is `T` (`tyref_atomic_arg`).
/// A scalar or a pointer to a storage-raw ADT is a raw word. A pointer
/// to a classed instance is not (`rclass.py` `getgcflavor`; `lltype.py`
/// `Struct.__init__` refuses a non-raw container inside a raw Struct;
/// `quasiimmut.py` `QuasiImmut` is an ordinary instance).
#[test]
fn atomic_ptr_to_classed_instance_is_not_a_raw_word() {
    fn atomic_of(atomic_id: u64, arg: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "Adt": {
                "id": atomic_id,
                "generics": {
                    "regions": [],
                    "types": [arg],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    }
    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let i32_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I32"}}});
    let vec = vec_decl_json();
    let atomic = opaque_type_decl(1, &["core", "sync", "atomic", "Atomic"]);
    let classed = struct_decl(
        2,
        ident_path(&["fixture", "ClassedInst"]),
        serde_json::json!([{
            "name": "items",
            "ty": vec_of(serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})),
            "attr_info": null
        }]),
        false,
    );
    let raw_cell = struct_decl(
        3,
        ident_path(&["fixture", "RawCell"]),
        serde_json::json!([{
            "name": "word",
            "ty": i32_ty,
            "attr_info": null
        }]),
        false,
    );
    let classed_ptr = serde_json::json!({"RawPtr": [adt_node(2), "Mut"]});
    let raw_ptr = serde_json::json!({"RawPtr": [adt_node(3), "Mut"]});
    let owner_of_classed = struct_decl(
        4,
        ident_path(&["fixture", "ClassedPtrOwner"]),
        serde_json::json!([{
            "name": "slot",
            "ty": atomic_of(1, classed_ptr.clone()),
            "attr_info": null
        }]),
        false,
    );
    let owner_of_raw = struct_decl(
        5,
        ident_path(&["fixture", "RawPtrOwner"]),
        serde_json::json!([{
            "name": "slot",
            "ty": atomic_of(1, raw_ptr.clone()),
            "attr_info": null
        }]),
        false,
    );
    let owner_of_scalar = struct_decl(
        6,
        ident_path(&["fixture", "ScalarAtomicOwner"]),
        serde_json::json!([{
            "name": "n",
            "ty": atomic_of(1, u32_ty.clone()),
            "attr_info": null
        }]),
        false,
    );
    let view = struct_decl(
        7,
        ident_path(&["fixture", "AtomicPtrView"]),
        serde_json::json!([
            {
                "name": "ptr",
                "ty": option_shared_ref(atomic_of(1, classed_ptr.clone())),
                "attr_info": null
            },
            {
                "name": "lock",
                "ty": option_shared_ref(atomic_of(1, classed_ptr.clone())),
                "attr_info": null
            }
        ]),
        false,
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![
            vec,
            atomic,
            classed,
            raw_cell,
            owner_of_classed,
            owner_of_raw,
            owner_of_scalar,
            view,
        ],
        vec![],
    );
    let gc = std::collections::HashSet::new();
    let immut = no_immutable_fields();
    let raw = |ty: serde_json::Value| {
        field_ty_is_raw_word(&TyRef::Other(ty), &llbc, &gc, immut, &mut Vec::new())
    };
    assert!(
        raw(atomic_of(1, u32_ty)),
        "Atomic<u32> stored word is the scalar"
    );
    assert!(
        raw(atomic_of(1, raw_ptr)),
        "Atomic<*mut RawCell> stored word is a C pointer"
    );
    assert!(
        !raw(atomic_of(1, classed_ptr.clone())),
        "Atomic<*mut ClassedInst> is a pointer to a classed instance"
    );
    assert!(
        !raw(option_shared_ref(atomic_of(1, classed_ptr))),
        "&Atomic<*mut ClassedInst> is the same stored word as the Atomic"
    );
    assert!(
        !adt_def_is_raw_storage(2, &llbc, &gc, immut, &mut Vec::new()),
        "classed instance with a Vec field is not raw storage"
    );
    assert!(
        adt_def_is_raw_storage(3, &llbc, &gc, immut, &mut Vec::new()),
        "RawCell of one integer is raw storage"
    );
    assert!(
        !adt_def_is_raw_storage(4, &llbc, &gc, immut, &mut Vec::new()),
        "owner of AtomicPtr to a classed instance is not a raw-word owner"
    );
    assert!(
        adt_def_is_raw_storage(5, &llbc, &gc, immut, &mut Vec::new()),
        "owner of AtomicPtr to a storage-raw ADT is a raw-word owner"
    );
    assert!(
        adt_def_is_raw_storage(6, &llbc, &gc, immut, &mut Vec::new()),
        "owner of Atomic<u32> is a raw-word owner"
    );
    assert!(
        !adt_def_is_raw_storage(7, &llbc, &gc, immut, &mut Vec::new()),
        "borrowed view of AtomicPtr to a classed instance is not a raw-word owner"
    );
}

/// Dest intern KEY of `Option<&TransparentWord>` erases `&` because the
/// borrow is the integer word (`tyref_is_borrowed_transparent_scalar` /
/// nested `repr(transparent)` wrap). A non-transparent one-integer
/// tuple struct is still interior `Ptr(T)` and keeps `&`.
#[test]
fn option_of_ref_to_integer_word_erases_ampersand() {
    let inner = struct_decl(
        1,
        ident_path(&["fixture", "InnerWord"]),
        serde_json::json!([{
            "name": null,
            "ty": {"Scalar": {"Integer": {"Unsigned": "U16"}}},
            "attr_info": null
        }]),
        true,
    );
    let outer = struct_decl(
        2,
        ident_path(&["fixture", "OuterWord"]),
        serde_json::json!([{
            "name": null,
            "ty": adt_node(1),
            "attr_info": null
        }]),
        true,
    );
    let raw_word = struct_decl(
        3,
        ident_path(&["fixture", "RawWord"]),
        serde_json::json!([{
            "name": null,
            "ty": {"Scalar": {"Integer": {"Signed": "I32"}}},
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types(
        "fixture",
        vec![option_enum(0), inner, outer, raw_word],
        vec![],
    );
    assert_eq!(
        super::tyref_to_ast_string(
            &TyRef::Other(option_adt_node(0, option_shared_ref(adt_node(1)))),
            &llbc
        ),
        "Option<InnerWord>",
        "dest intern KEY of Option<&transparent u16 wrap> erases &"
    );
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(option_adt_node(0, adt_node(1))), &llbc),
        "Option<InnerWord>",
        "by-value Option<transparent u16 wrap> is the integer Option"
    );
    assert_eq!(
        super::tyref_to_ast_string(
            &TyRef::Other(option_adt_node(0, option_shared_ref(adt_node(2)))),
            &llbc
        ),
        "Option<OuterWord>",
        "dest intern KEY of Option<&nested transparent wrap> erases &"
    );
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(option_adt_node(0, adt_node(2))), &llbc),
        "Option<OuterWord>",
        "by-value Option<nested transparent wrap> is the integer Option"
    );
    assert_eq!(
        super::tyref_to_ast_string(
            &TyRef::Other(option_adt_node(0, option_shared_ref(adt_node(3)))),
            &llbc
        ),
        "Option<&RawWord>",
        "dest intern KEY of Option<&non-transparent integer tuple> keeps &"
    );
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(option_adt_node(0, adt_node(3))), &llbc),
        "Option<RawWord>",
        "by-value Option<non-transparent integer tuple> is the tagged class"
    );
}

/// `harvest_declared_gc_facts` walks a tuple-struct field0 through
/// `field_is_zst` → `is_unit_type` → `charon_type_value_to_ast_string`.
/// That spelling used to `get_or_init` `value_type_inputs` and deadlock
/// on the same OnceLock (`orig_parse_format_link_args_are_defined`).
#[test]
fn render_adt_type_args_does_not_reenter_value_type_inputs() {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let inner = struct_decl(
        0,
        ident_path(&["fixture", "InnerWrap"]),
        serde_json::json!([{
            "name": null,
            "ty": option_adt_node(1, i64_ty.clone()),
            "attr_info": null
        }]),
        false,
    );
    let option = option_enum(1);
    let outer = struct_decl(
        2,
        ident_path(&["fixture", "OuterBag"]),
        serde_json::json!([{
            "name": "cell",
            "ty": adt_node(0),
            "attr_info": null
        }]),
        false,
    );
    let llbc = llbc_with_types("fixture", vec![inner, option, outer], vec![]);
    assert_eq!(
        super::tyref_to_ast_string(&TyRef::Other(option_adt_node(1, i64_ty)), &llbc),
        "Option<i64>",
        "first dest intern KEY must not wait on value_type_inputs"
    );
}

/// `[T]` (`{"Slice": elem}`), `str` (the `builtin: "Str"` ADT) and
/// `dyn Trait` are unsized.  A pointer to any of them is fat, and the
/// width proof has to decline every one of them.
#[test]
fn both_charon_spellings_of_an_unsized_pointee_decline() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let borrow = |pointee: serde_json::Value| {
        json_ty_is_thin_pointer_element(
            &serde_json::json!({"Ref": ["_", pointee, "Shared"]}),
            &llbc,
        )
    };
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let builtin = |name: &str, args: serde_json::Value| serde_json::json!({"Adt": {"id": 0, "builtin": name, "generics": {"types": args}}});

    assert!(!borrow(serde_json::json!({"Slice": [u8_ty.clone(), null]})));
    assert!(!borrow(builtin("Str", serde_json::json!([]))));
    assert!(!borrow(serde_json::json!({"DynTrait": {}})));

    let slice_adt = TyRef::Other(builtin("Slice", serde_json::json!([u8_ty])));
    let spelling = super::tyref_to_ast_string(&slice_adt, &llbc);
    assert_eq!(spelling, "[u8]");
    assert!(majit_ir::descr::is_list_container_spelling(&spelling));

    // A fixed-length array IS sized, so a pointer to one stays thin.
    let len = serde_json::json!([
        {"Integer": {"Unsigned": ["Usize", "14"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    assert!(borrow(serde_json::json!({"Array": [u8_ty, len, null]})));
}

#[test]
fn dyn_pointer_metadata_is_one_pointer_level() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let dyn_trait = serde_json::json!({"DynTrait": {}});
    let shared = |pointee: serde_json::Value| {
        super::TyRef::Other(serde_json::json!({"Ref": ["_", pointee, "Shared"]}))
    };
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    assert!(super::tyref_is_dyn_pointer(
        &shared(dyn_trait.clone()),
        &llbc
    ));
    assert!(!super::tyref_is_dyn_pointer(
        &shared(serde_json::json!({"Slice": [u8_ty, null]})),
        &llbc
    ));
    assert!(!super::tyref_is_dyn_pointer(
        &shared(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})),
        &llbc
    ));
}

/// `PtrMetadata` of `Box<dyn T>` is the vtable word (`FatMeta`), not a
/// slice length. `tyref_is_dyn_pointer` peels Ref/Value only, so the
/// Box ADT would otherwise be classed as `FatLen`.
#[test]
fn box_dyn_ptr_metadata_is_fat_meta() {
    use crate::model::{OpKind, VecFieldPart};

    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let dyn_trait = serde_json::json!({"DynTrait": {}});
    let empty_generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let boxed = |pointee: serde_json::Value| {
        super::TyRef::Other(serde_json::json!({
            "Adt": {
                "id": 0, "builtin": "Box",
                "generics": {
                    "regions": [],
                    "types": [pointee],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }))
    };
    let shared = |pointee: serde_json::Value| {
        super::TyRef::Other(serde_json::json!({"Ref": ["_", pointee, "Shared"]}))
    };
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let slice = serde_json::json!({"Slice": [u8_ty, null]});
    assert!(super::tyref_is_fat_dyn(&boxed(dyn_trait.clone()), &llbc));
    assert!(super::tyref_is_fat_dyn(&shared(dyn_trait.clone()), &llbc));
    assert!(!super::tyref_is_fat_dyn(&boxed(slice.clone()), &llbc));
    assert!(!super::tyref_is_fat_dyn(&shared(slice), &llbc));
    assert!(!super::tyref_is_dyn_pointer(
        &boxed(dyn_trait.clone()),
        &llbc
    ));

    let span = serde_json::json!({"data": {
        "file_id": 0,
        "beg": {"line": 1, "col": 0},
        "end": {"line": 1, "col": 10}
    }});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let holder_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": empty_generics}
    });
    let holder_ref = serde_json::json!({"Ref": ["static", holder_adt, "Shared"]});
    let fat_ty = serde_json::json!({
        "Adt": {
            "id": 1, "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [dyn_trait],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let meta_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let field_attr = serde_json::json!({
        "attributes": [], "inline": null, "rename": null, "public": true
    });
    let holder = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "Holder"]),
        "kind": {"Struct": [{
            "name": "slot",
            "ty": fat_ty,
            "attr_info": field_attr
        }]}
    });
    let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
    let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span, "ty": ty});
    let slot_place = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Projection": [
                place(serde_json::json!({"Local": 1}), &holder_ref),
                "Deref"
            ]}), &holder_adt),
            {"Field": [null, 0]}
        ]}),
        &fat_ty,
    );
    let metadata_place = place(
        serde_json::json!({"Projection": [
            place(serde_json::json!({"Local": 2}), &fat_ty),
            "PtrMetadata"
        ]}),
        &meta_ty,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "read_box_dyn_meta"]),
        "signature": {"is_unsafe": false, "inputs": [holder_ref], "output": meta_ty},
        "body": {"Unstructured": {
            "span": span,
            "locals": {"arg_count": 1, "locals": [
                local(0, &meta_ty),
                local(1, &holder_ref),
                local(2, &fat_ty)
            ]},
            "body": [
                {
                    "statements": [
                        {"span": span, "kind": {"Assign": [
                            place(serde_json::json!({"Local": 2}), &fat_ty),
                            {"Use": [{"Read": slot_place}, "No"]}
                        ]}},
                        {"span": span, "kind": {"Assign": [
                            place(serde_json::json!({"Local": 0}), &meta_ty),
                            {"Use": [{"Read": metadata_place}, "No"]}
                        ]}}
                    ],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": "Return"}
                }
            ]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [holder],
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = crate::test_support::lower_function(&llbc, "read_box_dyn_meta")
        .expect("lower read_box_dyn_meta");
    let fat_meta: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "slot" && field.vec_part == Some(VecFieldPart::FatMeta)
            )
        })
        .collect();
    assert!(
        !fat_meta.is_empty(),
        "PtrMetadata of Box<dyn T> must be FatMeta of slot, not FatLen\n{}",
        graph.dump()
    );
    let fat_len = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "slot" && field.vec_part == Some(VecFieldPart::FatLen)
            )
        })
    });
    assert!(
        !fat_len,
        "PtrMetadata of Box<dyn T> must not emit FatLen\n{}",
        graph.dump()
    );
}

/// Charon wraps a `DynTrait` predicate in `Value`. The principal trait
/// is the first clause's `TraitDeclRef.id` after that peel (`rclass.py`
/// `getvtable`).
#[test]
fn tyref_dyn_trait_decl_id_peels_value_wrapped_dyn_predicate() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let binder = serde_json::json!({
        "binder": {
            "params": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_clauses": [{
                    "clause_id": 0,
                    "origin": "Dyn",
                    "trait_": {
                        "regions": [],
                        "outlives": [],
                        "skip_binder": {
                            "id": 7,
                            "generics": {
                                "regions": [],
                                "types": [],
                                "const_generics": [],
                                "trait_refs": []
                            }
                        }
                    }
                }]
            }
        }
    });
    let wrapped = super::TyRef::Other(serde_json::json!({
        "Ref": ["static", {"Value": [1870, {"DynTrait": binder.clone()}]}, "Shared"]
    }));
    assert!(super::tyref_is_fat_dyn(&wrapped, &llbc));
    assert_eq!(super::tyref_dyn_trait_decl_id(&wrapped, &llbc), Some(7));
    let outer_value = super::TyRef::Other(serde_json::json!({
        "Value": [1869, {
            "Ref": ["static", {"Value": [1870, {"DynTrait": binder}]}, "Shared"]
        }]
    }));
    assert!(super::tyref_is_fat_dyn(&outer_value, &llbc));
    assert_eq!(super::tyref_dyn_trait_decl_id(&outer_value, &llbc), Some(7));
}

#[test]
fn window_classifier_predicates() {
    use crate::model::ValueType;

    let utf8 = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["pyre_object", "unicodeobject", "Utf8Str"], true),
        "kind": {"Struct": []}
    });
    let llbc = window_llbc(vec![utf8, option_enum(1)], vec![]);
    let shared = TyRef::Other(window_shared_slice());
    let mut_slice = TyRef::Other(window_mut_slice());
    let bare = TyRef::Other(window_slice());
    let const_ptr = TyRef::Other(window_const_slice_ptr());
    let utf8_ty = TyRef::Other(serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": window_empty_generics()}
    }));
    assert!(super::tyref_is_shared_u8_slice(&shared, &llbc));
    assert!(super::tyref_is_shared_u8_slice(&const_ptr, &llbc));
    assert!(!super::tyref_is_shared_u8_slice(&mut_slice, &llbc));
    assert!(!super::tyref_is_shared_u8_slice(&utf8_ty, &llbc));
    assert!(super::tyref_is_bare_u8_slice(&bare, &llbc));
    assert!(!super::tyref_is_bare_u8_slice(&shared, &llbc));
    assert!(!super::tyref_is_string_value(&shared, &llbc));
    assert!(super::tyref_is_string_value(&utf8_ty, &llbc));
    assert!(super::tyref_is_u8_window(&shared, &llbc));
    assert!(super::tyref_is_u8_window(&const_ptr, &llbc));
    assert!(!super::tyref_is_u8_window(&mut_slice, &llbc));
    let box_u8 = TyRef::Other(serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [window_slice()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    }));
    let box_u8_ref = TyRef::Other(serde_json::json!({"Ref": ["Erased", {
            "Adt": {
                "id": 0,
                "builtin": "Box",
                "generics": {
                    "regions": [],
                    "types": [window_slice()],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        }, "Shared"]}));
    assert!(super::tyref_is_box_u8_slice(&box_u8, &llbc));
    assert!(super::tyref_is_box_u8_slice(&box_u8_ref, &llbc));
    assert!(super::tyref_is_u8_list(&box_u8, &llbc));
    assert!(super::tyref_is_u8_list(&box_u8_ref, &llbc));
    assert!(super::tyref_is_fixed_u8_list(&box_u8, &llbc));
    assert!(!super::tyref_is_u8_window(&box_u8, &llbc));
    let tomb = std::collections::HashSet::new();
    let gc = std::collections::HashSet::new();
    assert!(!matches!(
        super::tyref_to_value_type_with(
            &shared,
            &llbc,
            &tomb,
            &gc,
            &std::collections::HashSet::<String>::new(),
        ),
        ValueType::Str
    ));
    let option_shared = TyRef::Other(option_adt_node(1, window_shared_slice()));
    let option_bare = TyRef::Other(option_adt_node(1, window_slice()));
    let option_mut = TyRef::Other(option_adt_node(
        1,
        serde_json::json!({"Ref": ["Erased", window_slice(), "Mut"]}),
    ));
    assert!(super::tyref_is_option_of_shared_u8_slice(
        &option_shared,
        &llbc
    ));
    assert!(super::tyref_is_option_of_shared_u8_slice(
        &option_bare,
        &llbc
    ));
    assert!(!super::tyref_is_option_of_shared_u8_slice(
        &option_mut,
        &llbc
    ));
    assert!(!matches!(
        super::tyref_to_value_type_with(
            &option_shared,
            &llbc,
            &tomb,
            &gc,
            &std::collections::HashSet::<String>::new(),
        ),
        ValueType::Str
    ));
    assert!(!super::dont_look_inside_return_is_str(
        &shared,
        &llbc,
        crate::ErrorCarrierSpec::default()
    ));
}

#[test]
fn fixed_array_suffix_separates_item_type_and_length() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let array = |item: serde_json::Value, len: &str| {
        let len = serde_json::json!([
            {"Integer": {"Unsigned": ["Usize", len]}},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
        ]);
        TyRef::Other(serde_json::json!({ "Array": [item, len, null] }))
    };
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});

    assert_eq!(
        tyref_array_suffix(&array(i64_ty.clone(), "1"), &llbc),
        "<i64;1>"
    );
    assert_eq!(tyref_array_suffix(&array(i64_ty, "2"), &llbc), "<i64;2>");
    assert_eq!(tyref_array_suffix(&array(bool_ty, "1"), &llbc), "<bool;1>");
    assert_eq!(shaped_array_parts("Array<[i64;2];3>"), Some(("[i64;2]", 3)));
}

#[test]
fn field_row_struct_identity_qualifies_array_of_adt_element() {
    // `charon_adt_to_ast_string` renders the ADT leaf, so a harvested
    // `[Entry;0]` would collide with a second TypeDecl of the same leaf.
    // The array-of-ADT row carries the TypeDecl path, the same way a
    // by-value ADT field does.
    let llbc = llbc_with_types(
        "host",
        vec![
            type_decl_struct(0, &["host", "rordereddict_entries", "Entry"], vec![]),
            type_decl_struct(1, &["host", "counter", "Entry"], vec![]),
        ],
        vec![],
    );
    let array_len = serde_json::json!([
        {"Integer": {"Unsigned": ["Usize", "0"]}},
        {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
    ]);
    let array_ty = TyRef::Other(serde_json::json!({
        "Array": [
            named_adt(
                0,
                vec![
                    serde_json::json!({"TypeVar": {"Bound": [0, 0]}}),
                    serde_json::json!({"TypeVar": {"Bound": [0, 1]}}),
                ],
            ),
            array_len,
            null
        ]
    }));
    let gc = std::collections::HashSet::<majit_ir::descr::StructId>::new();
    let raw = std::collections::HashSet::<String>::new();
    let layout = super::tyref_to_field_layout_string(&array_ty, &llbc, &gc, &raw);
    assert_eq!(layout, "[Entry<??TypeVar#0,??TypeVar#1>;0]");
    assert_eq!(
        super::field_row_struct_identity(&array_ty, &llbc, &layout).as_deref(),
        Some("[rordereddict_entries::Entry<??TypeVar#0,??TypeVar#1>;0]")
    );

    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let primitive = TyRef::Other(serde_json::json!({
        "Array": [
            i64_ty,
            [
                {"Integer": {"Unsigned": ["Usize", "0"]}},
                {"Scalar": {"Integer": {"Unsigned": "Usize"}}}
            ],
            null
        ]
    }));
    let primitive_layout = super::tyref_to_field_layout_string(&primitive, &llbc, &gc, &raw);
    assert_eq!(primitive_layout, "[i64;0]");
    assert_eq!(
        super::field_row_struct_identity(&primitive, &llbc, &primitive_layout),
        None
    );
}

#[test]
fn field_row_struct_identity_qualifies_slice_element_behind_box_and_unsafecell() {
    // `charon_adt_to_ast_string` renders the ADT leaf, so a harvested
    // `[CopyInstrWord]` would collide with a second TypeDecl of the same
    // leaf. The slice arm qualifies only a by-value Copy struct element
    // (`json_ty_is_by_value_copy_aggregate`, the FLAG_STRUCT item
    // `adt_node_class_root_identity` names). A slice-of-Copy-struct row,
    // including `[T]` reached through `Box` / `UnsafeCell`, carries the
    // TypeDecl path the same way a by-value ADT field does and keeps
    // the wrapper spelling.
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let typevar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let llbc = llbc_with_types(
        "host",
        vec![
            type_decl_struct(
                0,
                &["host", "bytecode", "CopyInstrWord"],
                vec![
                    named_copy_field("op", u8_ty.clone()),
                    named_copy_field("arg", u8_ty.clone()),
                ],
            ),
            type_decl_struct(
                1,
                &["host", "gc", "CopyInstrWord"],
                vec![
                    named_copy_field("wrappers", u8_ty.clone()),
                    named_copy_field("pinned", u8_ty),
                ],
            ),
            type_decl_struct(
                2,
                &["core", "cell", "UnsafeCell"],
                vec![named_copy_field("value", typevar)],
            ),
        ],
        vec![],
    );
    let word = adt_node(0);
    let box_slice = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [{"Slice": [word.clone(), null]}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cell_ty = TyRef::Other(serde_json::json!({
        "Adt": {
            "id": 2,
            "generics": {
                "regions": [],
                "types": [box_slice],
                "const_generics": [],
                "trait_refs": []
            }
        }
    }));
    let gc = std::collections::HashSet::<majit_ir::descr::StructId>::new();
    let raw = std::collections::HashSet::<String>::new();
    let cell_layout = super::tyref_to_field_layout_string(&cell_ty, &llbc, &gc, &raw);
    assert_eq!(cell_layout, "UnsafeCell<Box<[CopyInstrWord]>>");
    assert_eq!(
        super::field_row_struct_identity(&cell_ty, &llbc, &cell_layout).as_deref(),
        Some("UnsafeCell<Box<[bytecode::CopyInstrWord]>>")
    );

    let slice_ty = TyRef::Other(serde_json::json!({"Slice": [word.clone(), null]}));
    let slice_layout = super::tyref_to_field_layout_string(&slice_ty, &llbc, &gc, &raw);
    assert_eq!(slice_layout, "[CopyInstrWord]");
    assert_eq!(
        super::field_row_struct_identity(&slice_ty, &llbc, &slice_layout).as_deref(),
        Some("[bytecode::CopyInstrWord]")
    );

    let box_ty = TyRef::Other(serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [{"Slice": [word, null]}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    }));
    let box_layout = super::tyref_to_field_layout_string(&box_ty, &llbc, &gc, &raw);
    assert_eq!(box_layout, "Box<[CopyInstrWord]>");
    assert_eq!(
        super::field_row_struct_identity(&box_ty, &llbc, &box_layout).as_deref(),
        Some("Box<[bytecode::CopyInstrWord]>")
    );

    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let primitive = TyRef::Other(serde_json::json!({"Slice": [i64_ty, null]}));
    let primitive_layout = super::tyref_to_field_layout_string(&primitive, &llbc, &gc, &raw);
    assert_eq!(primitive_layout, "[i64]");
    assert_eq!(
        super::field_row_struct_identity(&primitive, &llbc, &primitive_layout),
        None
    );
}

#[test]
fn field_row_struct_identity_keeps_slice_of_gc_pointer_struct() {
    // `RDict.entries` is `UnsafeCell<Box<[Entry<IdentityKey, PyObjectRef>]>>`.
    // The slice arm must not qualify a GC-pointer-carrying struct: that
    // spelling stays `tyref_to_field_layout_string`.
    let typevar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let gc_ptr = serde_json::json!({
        "RawPtr": [
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Mut"
        ]
    });
    let llbc = llbc_with_types(
        "host",
        vec![
            type_decl_struct(
                0,
                &["host", "rordereddict_entries", "Entry"],
                vec![
                    named_copy_field("key", gc_ptr.clone()),
                    named_copy_field("value", gc_ptr),
                ],
            ),
            type_decl_struct(
                1,
                &["core", "cell", "UnsafeCell"],
                vec![named_copy_field("value", typevar)],
            ),
        ],
        vec![],
    );
    let entry = named_adt(
        0,
        vec![
            serde_json::json!({"TypeVar": {"Bound": [0, 0]}}),
            serde_json::json!({"TypeVar": {"Bound": [0, 1]}}),
        ],
    );
    assert!(
        !json_ty_is_by_value_copy_aggregate(&entry, &llbc),
        "a GC/raw pointer field is not a by-value Copy item"
    );
    let box_slice = serde_json::json!({
        "Adt": {
            "id": 0,
            "builtin": "Box",
            "generics": {
                "regions": [],
                "types": [{"Slice": [entry, null]}],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let cell_ty = TyRef::Other(serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [box_slice],
                "const_generics": [],
                "trait_refs": []
            }
        }
    }));
    let gc = std::collections::HashSet::<majit_ir::descr::StructId>::new();
    let raw = std::collections::HashSet::<String>::new();
    let cell_layout = super::tyref_to_field_layout_string(&cell_ty, &llbc, &gc, &raw);
    assert_eq!(
        cell_layout,
        "UnsafeCell<Box<[Entry<??TypeVar#0,??TypeVar#1>]>>"
    );
    assert_eq!(
        super::field_row_struct_identity(&cell_ty, &llbc, &cell_layout),
        None
    );
}

#[test]
fn positional_call_result_recovers_its_exact_tuple_repr_owner() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let pair = TyRef::Other(serde_json::json!({
        "Adt": {
            "id": 0, "builtin": "Tuple",
            "generics": {"types": [usize_ty.clone(), usize_ty]}
        }
    }));
    let scalar = TyRef::Other(serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}}));

    assert_eq!(
        tyref_positional_aggregate_root(&pair, &llbc).as_deref(),
        Some("Tuple<usize,usize>")
    );
    assert_eq!(tyref_positional_aggregate_root(&scalar, &llbc), None);
}

#[test]
fn bodyless_trait_method_on_a_trait_object_uses_indirect_dispatch() {
    let method = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Storage", 0]}, {"Ident": ["head", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 10}
            }},
            "source_text": "fn head(&self);",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {
            "is_unsafe": false,
            // The `&self` the source text declares. Its shape does not
            // matter to the classifier, only that the receiver is there.
            "inputs": [{"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}}],
            "output": {"Tuple": []}
        },
        "body": "Missing"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [null, method],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Storage", 0]}],
                    "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}}},
                    "source_text": "trait Storage",
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": true
                },
                "methods": [{"skip_binder": {"name": "head", "signature": {"inputs": [{"Adt": {"id": 0, "builtin": "Tuple"}}]}}}]
            }],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        "kind": {"Trait": [{"kind": "Dyn", "trait_decl_ref": {"skip_binder": {"id": 0}}}, 0]},
        "generics": {}
    }))
    .expect("fixture trait call parses");

    assert_eq!(
        super::abstract_trait_call_target(&call, &llbc),
        Some(("Storage".to_string(), "head".to_string()))
    );
}

#[test]
fn a_mono_trait_object_call_names_its_method_from_assoc_item_names() {
    // Under `--monomorphize` the trait declaration of a sibling crate
    // carries no method rows and its vtable struct is opaque, so Charon
    // leaves `x.getitem_str(..)` on a `&dyn DictStrategy` as a
    // `CallKind::Trait` with a `Dyn` reference. The method index is then
    // named only by the crate's `assoc_item_names` table.
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Storage", 0]}],
                    "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}}},
                    "source_text": "trait Storage",
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": false
                },
                "methods": []
            }],
            "assoc_item_names": [{"types": [], "methods": ["len", "head"], "consts": []}],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        "kind": {"Trait": [{"kind": "Dyn", "trait_decl_ref": {"skip_binder": {"id": 0}}}, 1]},
        "generics": {}
    }))
    .expect("fixture trait call parses");

    assert_eq!(
        super::abstract_trait_call_target(&call, &llbc),
        Some(("Storage".to_string(), "head".to_string()))
    );
}

#[test]
fn bodyless_trait_method_behind_a_clause_is_not_indirect() {
    // `fn body<H: Storage>(h: &mut H) { h.head() }`: the declaration of
    // `head` is as bodyless as the trait-object case above, but the
    // reference is the bound on `H`, which whoever instantiated `body`
    // has already resolved. The implementation is unique, so the direct
    // `[Storage, head]` path names the impl's graph and the callee stays
    // inside the inline closure instead of becoming a residual call.
    let method = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Storage", 0]}, {"Ident": ["head", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 10}
            }},
            "source_text": "fn head(&self);",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [{"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}}],
            "output": {"Tuple": []}
        },
        "body": "Missing"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [null, method],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 1]}}}, 0, 1]},
        "generics": {}
    }))
    .expect("fixture trait call parses");

    assert_eq!(super::abstract_trait_call_target(&call, &llbc), None);
}

#[test]
fn a_hash_consed_trait_ref_wrapping_a_deduplicated_one_still_reads_dyn() {
    // Charon's two indirections nest: the reference arrives as
    // `Value: [id, {Deduplicated: n}]`. Peeling one of each in a
    // fixed order leaves the inner `Deduplicated` standing, the `kind`
    // read misses, and a genuine trait-object call is declined — a
    // classification that would then depend on how the reference happened
    // to be serialized. The receiver type below registers body 9 so the
    // inner hop has somewhere to land.
    let method = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Storage", 0]}, {"Ident": ["head", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 10}
            }},
            "source_text": "fn head(&self);",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [{"Value": [9, {"kind": "Dyn", "trait_decl_ref": {"skip_binder": {"id": 0}}}]}],
            "output": {"Tuple": []}
        },
        "body": "Missing"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [null, method],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Storage", 0]}],
                    "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}}},
                    "source_text": "trait Storage",
                    "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                    "is_local": true
                },
                "methods": [{"skip_binder": {"name": "head", "signature": {"inputs": [{"Adt": {"id": 0}}]}}}]
            }],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        "kind": {"Trait": [{"Value": [7, {"Deduplicated": 9}]}, 0]},
        "generics": {}
    }))
    .expect("fixture trait call parses");

    assert_eq!(
        super::abstract_trait_call_target(&call, &llbc),
        Some(("Storage".to_string(), "head".to_string()))
    );
}

#[test]
fn receiverless_trait_associated_function_is_not_indirect() {
    // `fn type_id() -> u64` on a trait: the implementation is chosen by
    // the type argument at the call site, and there is no instance the
    // dispatch could read it from. `lower_indirect_calls` would take the
    // call's first argument for a receiver, so this has to stay direct.
    let method = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["GcType", 0]}, {"Ident": ["type_id", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 10}
            }},
            "source_text": "fn type_id() -> u64;",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": {"Scalar": {"Integer": {"Unsigned": "U64"}}}
        },
        "body": "Missing"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [null, method],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        // `Dyn` so the receiver guard is what rejects this, not the
        // reference kind.
        "kind": {"Trait": [{"kind": "Dyn"}, 0, 1]},
        "generics": {}
    }))
    .expect("fixture trait call parses");

    assert_eq!(super::abstract_trait_call_target(&call, &llbc), None);
}

#[test]
fn trait_method_on_known_receiver_names_the_impl_graph() {
    // `description.py` `ClassDesc.lookup`: RangeBounds::start_bound
    // on a Range receiver is that impl's graph, not the trait path.
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
    let range_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": {"types": []}}
    });
    let range_td = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["core", "ops", "range", "Range"]),
        "kind": {"Struct": []}
    });
    let start_bound = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["core", 0]},
                {"Ident": ["ops", 0]},
                {"Ident": ["range", 0]},
                {"Ident": ["Range", 0]},
                {"Impl": {
                    "Ty": {"skip_binder": {"Value": [0, range_adt.clone()]}},
                    "Trait": 0
                }},
                {"Ident": ["start_bound", 0]}
            ],
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [{"Ref": [null, range_adt.clone(), "Shared"]}],
            "output": {"Tuple": []}
        },
        "body": {"Unstructured": {
            "span": span.clone(),
            "locals": {"arg_count": 1, "locals": []},
            "body": []
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [range_td],
            "fun_decls": [null, start_bound],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "ops", "range", "RangeBounds"]),
                "methods": [{"skip_binder": {"name": "start_bound"}}]
            }],
            "trait_impls": [{
                "impl_trait": {"id": 0, "generics": {"types": [range_adt.clone()]}},
                "methods": [{
                    "kind": {"TraitMethod": [0, 0]},
                    "skip_binder": {"id": 1}
                }]
            }]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let recv = TyRef::Other(serde_json::json!({
        "Ref": [null, range_adt, "Shared"]
    }));
    let payload = serde_json::json!([
        {
            "kind": {"Clause": {"Bound": [0, 0]}},
            "trait_decl_ref": {"skip_binder": {"id": 0}}
        },
        0
    ]);
    let identity =
        crate::front::clause_spec::render_trait_impl(&llbc, 0).expect("RangeBounds impl identity");
    let got = super::trait_method_on_known_receiver(Some(&recv), &payload, &llbc);
    assert_eq!(
        got.map(|(_, segs, _)| segs),
        Some(super::trait_impl_identity_path(
            "ops::range::Range",
            &identity,
            "start_bound"
        ))
    );
}

#[test]
fn trait_box_slice_deref_is_container_identity() {
    // `description.py` `ClassDesc.lookup`: `<Box<[T]> as Deref>::deref`
    // is that impl's FunDecl. The impl is bodyless (`Box::deref`); the
    // dest is a slice, so the lifted list model aliases — same rule as
    // `Vec`/`String`, not a `["Deref","deref"]` residual.
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
    let box_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": "Box", "generics": {"types": []}}
    });
    let box_td = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["alloc", "boxed", "Box"]),
        "kind": {"Struct": []}
    });
    let deref_fd = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["alloc", 0]},
                {"Ident": ["boxed", 0]},
                {"Ident": ["Box", 0]},
                {"Impl": {
                    "Ty": {"skip_binder": {"Value": [0, box_adt.clone()]}},
                    "Trait": 0
                }},
                {"Ident": ["deref", 0]}
            ],
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [{"Ref": [null, box_adt.clone(), "Shared"]}],
            "output": {"Tuple": []}
        },
        "body": null
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [box_td],
            "fun_decls": [null, deref_fd],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "ops", "deref", "Deref"]),
                "methods": [{"skip_binder": {"name": "deref"}}]
            }],
            "trait_impls": [{
                "impl_trait": {"id": 0, "generics": {"types": [box_adt.clone()]}},
                "generics": {"types": [], "const_generics": [], "trait_refs": []},
                "methods": [{
                    "kind": {"TraitMethod": [0, 0]},
                    "skip_binder": {"id": 1}
                }]
            }]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let recv = TyRef::Other(serde_json::json!({
        "Ref": [null, box_adt, "Shared"]
    }));
    let slice_ty = TyRef::Other(serde_json::json!({
        "Ref": [null, {"Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, null]}, "Shared"]
    }));
    let i64_ty = TyRef::Other(serde_json::json!({
        "Ref": [null, {"Scalar": {"Integer": {"Signed": "I64"}}}, "Shared"]
    }));
    let trait_impl_call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        "kind": {"Trait": [{
            "kind": {"TraitImpl": {
                "id": 0,
                "generics": {"types": [], "const_generics": [], "trait_refs": []}
            }}
        }, 0]},
        "generics": {}
    }))
    .expect("TraitImpl deref call parses");
    let clause_call = serde_json::from_value::<super::RegularCall>(serde_json::json!({
        "kind": {"Trait": [{
            "kind": {"Clause": {"Bound": [0, 0]}},
            "trait_decl_ref": {"skip_binder": {"id": 0}}
        }, 0]},
        "generics": {}
    }))
    .expect("Clause deref call parses");

    let fd = super::deref_callee_fun_decl(&trait_impl_call, None, &llbc)
        .expect("TraitImpl names Box::deref");
    assert_eq!(
        super::deref_impl_owner_leaf(&llbc, fd).as_deref(),
        Some("Box")
    );
    assert!(super::is_deref_call(&trait_impl_call, &llbc));
    assert!(super::is_deref_call_for(&clause_call, Some(&recv), &llbc));
    assert!(super::tyref_strips_to_slice(&slice_ty, &llbc));
    assert!(!super::tyref_strips_to_slice(&i64_ty, &llbc));
    assert!(super::container_identity_deref_owner(
        "alloc::boxed::Box",
        Some(&slice_ty),
        &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "alloc::boxed::Box",
        Some(&i64_ty),
        &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "alloc::boxed::Box",
        None,
        &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "Constants",
        Some(&slice_ty),
        &llbc
    ));
    assert_eq!(
        super::impl_owner_path(&llbc, fd).as_deref(),
        Some("alloc::boxed::Box")
    );
    assert!(super::container_identity_deref_owner(
        "alloc::vec::Vec",
        None,
        &llbc
    ));
    assert!(super::container_identity_deref_owner(
        "alloc::string::String",
        None,
        &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "foo::Vec", None, &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "foo::alloc::vec::Vec",
        None,
        &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "foo::alloc::string::String",
        None,
        &llbc
    ));
    assert!(!super::container_identity_deref_owner(
        "foo::alloc::boxed::Box",
        Some(&slice_ty),
        &llbc
    ));
}

#[test]
fn user_trait_deref_method_is_not_ops_deref() {
    // `description.py` `FunctionDesc.pycall` follows the selected
    // implementation through `annotator.recursivecall`. A user trait
    // `MyDeref::deref` on a string-family receiver keeps its body;
    // only `<String as core::ops::deref::Deref>::deref` is the
    // identity-deref rewrite.
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
    let string_adt = serde_json::json!({
        "Adt": {"id": 0, "generics": {"types": []}}
    });
    let string_td = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["alloc", "string", "String"]),
        "kind": {"Struct": []}
    });
    let fundecl = |def_id: u64, impl_id: u64, segs: &[&str]| {
        let mut name: Vec<serde_json::Value> = segs
            .iter()
            .map(|s| serde_json::json!({"Ident": [s, 0]}))
            .collect();
        name.push(serde_json::json!({
            "Impl": {
                "Ty": {"skip_binder": {"Value": [0, string_adt.clone()]}},
                "Trait": impl_id
            }
        }));
        name.push(serde_json::json!({"Ident": ["deref", 0]}));
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": name,
                "span": span.clone(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": false
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [{"Ref": [null, string_adt.clone(), "Shared"]}],
                "output": {"Ref": [null, {"Adt": {"id": 0, "builtin": "Str", "generics": {"types": []}}}, "Shared"]}
            },
            "body": null
        })
    };
    let trait_impl = |trait_id: u64, fn_id: u64| {
        serde_json::json!({
            "impl_trait": {"id": trait_id, "generics": {"types": [string_adt.clone()]}},
            "generics": {"types": [], "const_generics": [], "trait_refs": []},
            "methods": [{
                "kind": {"TraitMethod": [trait_id, 0]},
                "skip_binder": {"id": fn_id}
            }]
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [string_td],
            "fun_decls": [
                null,
                fundecl(1, 0, &["alloc", "string"]),
                fundecl(2, 1, &["fixture"])
            ],
            "global_decls": [],
            "trait_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["core", "ops", "deref", "Deref"]),
                    "methods": [{"skip_binder": {"name": "deref"}}]
                },
                {
                    "def_id": 1,
                    "item_meta": {
                        "name": [{"Ident": ["fixture", 0]}, {"Ident": ["MyDeref", 0]}],
                        "span": span.clone(),
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "methods": [{"skip_binder": {"name": "deref"}}]
                }
            ],
            "trait_impls": [trait_impl(0, 1), trait_impl(1, 2)]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let ops_deref_fd = llbc.fn_by_id(1).expect("String as Deref::deref");
    let my_deref_fd = llbc.fn_by_id(2).expect("String as MyDeref::deref");
    assert!(super::is_deref_fundecl(ops_deref_fd, &llbc));
    assert!(!super::is_deref_fundecl(my_deref_fd, &llbc));

    let recv = TyRef::Other(serde_json::json!({
        "Ref": [null, string_adt, "Shared"]
    }));
    let fun_call = |id: u64| {
        serde_json::from_value::<super::RegularCall>(serde_json::json!({
            "kind": {"Fun": id},
            "generics": {}
        }))
        .expect("Fun Regular call parses")
    };
    let trait_impl_call = |impl_id: u64| {
        serde_json::from_value::<super::RegularCall>(serde_json::json!({
            "kind": {"Trait": [{
                "kind": {"TraitImpl": {
                    "id": impl_id,
                    "generics": {"types": [], "const_generics": [], "trait_refs": []}
                }}
            }, 0]},
            "generics": {}
        }))
        .expect("TraitImpl call parses")
    };
    let clause_call = |trait_id: u64| {
        serde_json::from_value::<super::RegularCall>(serde_json::json!({
            "kind": {"Trait": [{
                "kind": {"Clause": {"Bound": [0, 0]}},
                "trait_decl_ref": {"skip_binder": {"id": trait_id}}
            }, 0]},
            "generics": {}
        }))
        .expect("Clause call parses")
    };

    assert!(super::is_deref_call(&fun_call(1), &llbc));
    assert!(!super::is_deref_call(&fun_call(2), &llbc));
    assert!(super::is_deref_call(&trait_impl_call(0), &llbc));
    assert!(!super::is_deref_call(&trait_impl_call(1), &llbc));
    assert!(super::is_deref_call_for(
        &clause_call(0),
        Some(&recv),
        &llbc
    ));
    assert!(!super::is_deref_call_for(
        &clause_call(1),
        Some(&recv),
        &llbc
    ));
    // Unresolved Trait fallback: full declared path, not the last segment.
    assert!(super::is_deref_call(&clause_call(0), &llbc));
    assert!(!super::is_deref_call(&clause_call(1), &llbc));
}

#[test]
fn trait_impl_for_ref_selects_on_double_ref_receiver() {
    // `impl Trait for &Foo` with `fn method(&self)`: the first
    // argument is `&&Foo`. `ClassDesc.lookup` matches the method's
    // first-input type, keeping every reference layer.
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
    let foo_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": {"types": []}}
    });
    let ref_foo = serde_json::json!({"Ref": [null, foo_adt.clone(), "Shared"]});
    let ref_ref_foo = serde_json::json!({"Ref": [null, ref_foo.clone(), "Shared"]});
    let foo_td = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "Foo"]),
        "kind": {"Struct": []}
    });
    let method = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["Foo", 0]},
                {"Impl": {
                    "Ty": {"skip_binder": {"Value": [0, ref_foo.clone()]}},
                    "Trait": 0
                }},
                {"Ident": ["method", 0]}
            ],
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "signature": {
            "is_unsafe": false,
            "inputs": [ref_ref_foo.clone()],
            "output": {"Tuple": []}
        },
        "body": {"Unstructured": {
            "span": span.clone(),
            "locals": {"arg_count": 1, "locals": []},
            "body": []
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [foo_td.clone()],
            "fun_decls": [null, method.clone()],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["fixture", "Trait"]),
                "methods": [{"skip_binder": {"name": "method"}}]
            }],
            "trait_impls": [{
                "impl_trait": {"id": 0, "generics": {"types": [ref_foo.clone()]}},
                "generics": {"types": [], "const_generics": [], "trait_refs": []},
                "methods": [{
                    "kind": {"TraitMethod": [0, 0]},
                    "skip_binder": {"id": 1}
                }]
            }]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let recv = TyRef::Other(ref_ref_foo);
    let payload = serde_json::json!([
        {
            "kind": {"Clause": {"Bound": [0, 0]}},
            "trait_decl_ref": {"skip_binder": {"id": 0}}
        },
        0
    ]);
    let got = super::trait_impl_match_for_receiver(&recv, &payload, &llbc);
    assert_eq!(
        got.map(|(impl_id, fn_id, _)| (impl_id, fn_id)),
        Some((0, 1)),
        "impl Trait for &Foo must match a &&Foo argument"
    );

    // A matched impl whose `reconstructed_impl_generics` is None is
    // unresolvable (`ClassDesc.lookup`); the other candidate is not unique.
    let mut method_ok = method.clone();
    method_ok["def_id"] = serde_json::json!(2);
    let file_two = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [foo_td],
            "fun_decls": [null, method, method_ok],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["fixture", "Trait"]),
                "methods": [{"skip_binder": {"name": "method"}}]
            }],
            "trait_impls": [
                {
                    "impl_trait": {"id": 0, "generics": {"types": [ref_foo.clone()]}},
                    "generics": {
                        "types": [{"TypeVar": {"Bound": [0, 0]}}],
                        "const_generics": [],
                        "trait_refs": []
                    },
                    "methods": [{
                        "kind": {"TraitMethod": [0, 0]},
                        "skip_binder": {"id": 1}
                    }]
                },
                {
                    "impl_trait": {"id": 0, "generics": {"types": [ref_foo]}},
                    "generics": {"types": [], "const_generics": [], "trait_refs": []},
                    "methods": [{
                        "kind": {"TraitMethod": [0, 0]},
                        "skip_binder": {"id": 2}
                    }]
                }
            ]
        }
    });
    let llbc_two = Llbc::from_slice(file_two.to_string().as_bytes()).expect("two-impl Llbc parses");
    let got_two = super::trait_impl_match_for_receiver(&recv, &payload, &llbc_two);
    assert_eq!(
        got_two.map(|(impl_id, fn_id, _)| (impl_id, fn_id)),
        None,
        "a matched impl that cannot reconstruct generics is unresolvable"
    );
}

#[test]
fn spec_segments_distinguishes_scalar_trait_impls() {
    // `CallControl.graphs_from`: `impl PartialOrd<char> for char` and
    // `impl PartialOrd<i64> for i64` are distinct function objects.
    // `name_path` is `cmp::impls::<Impl>::le` for both; the impl's
    // trait ref and Self type is the identity (`Bookkeeper.getdesc`).
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
            "is_local": false
        })
    };
    let char_ty = serde_json::json!({"Scalar": "Char"});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let le_fd = |def_id: u64, impl_id: u64| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": meta(vec![
                ident("core"),
                ident("cmp"),
                ident("impls"),
                serde_json::json!({"Impl": {"Trait": impl_id}}),
                ident("le"),
            ]),
            "signature": {
                "is_unsafe": false,
                "inputs": [{"Scalar": "Bool"}],
                "output": {"Scalar": "Bool"}
            },
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 2, "locals": []},
                "body": []
            }}
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [null, le_fd(1, 0), le_fd(2, 1)],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(vec![ident("core"), ident("cmp"), ident("PartialOrd")]),
                "methods": [{"skip_binder": {"name": "le"}}]
            }],
            "trait_impls": [
                {"impl_trait": {"id": 0, "generics": {"types": [char_ty.clone(), char_ty.clone()]}}, "methods": []},
                {"impl_trait": {"id": 0, "generics": {"types": [i64_ty.clone(), i64_ty.clone()]}}, "methods": []}
            ]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let char_le = llbc.fn_by_id(1).expect("char le");
    let i64_le = llbc.fn_by_id(2).expect("i64 le");
    assert_eq!(super::impl_method_owner_for_fundecl(&llbc, char_le), None);
    assert_eq!(super::impl_method_owner_for_fundecl(&llbc, i64_le), None);
    let char_path = super::spec_segments(&llbc, char_le, "le");
    let i64_path = super::spec_segments(&llbc, i64_le, "le");
    let char_identity =
        crate::front::clause_spec::render_trait_impl(&llbc, 0).expect("char PartialOrd identity");
    let i64_identity =
        crate::front::clause_spec::render_trait_impl(&llbc, 1).expect("i64 PartialOrd identity");
    assert_ne!(char_identity, i64_identity);
    assert_eq!(
        char_path,
        super::trait_impl_identity_path("cmp::impls::<Impl>", &char_identity, "le")
    );
    assert_eq!(
        i64_path,
        super::trait_impl_identity_path("cmp::impls::<Impl>", &i64_identity, "le")
    );
    assert_ne!(char_path, i64_path);
    assert_eq!(super::fundecl_fn_item_segments(&llbc, char_le), char_path);
    assert_eq!(super::fundecl_fn_item_segments(&llbc, i64_le), i64_path);
    assert_eq!(
        super::registered_path_for_fun_decl(&llbc, char_le).segments,
        char_path
    );
    assert_eq!(
        super::registered_path_for_fun_decl(&llbc, i64_le).segments,
        i64_path
    );
    let mut char_fn =
        crate::front::semantic::SemanticFunction::with_empty_graph("le", "le", None, None, None);
    char_fn.module_path = "cmp::impls::<Impl>".to_string();
    char_fn.trait_impl_id = Some(0);
    char_fn.trait_impl_identity = Some(char_identity.clone());
    let mut i64_fn =
        crate::front::semantic::SemanticFunction::with_empty_graph("le", "le", None, None, None);
    i64_fn.module_path = "cmp::impls::<Impl>".to_string();
    i64_fn.trait_impl_id = Some(1);
    i64_fn.trait_impl_identity = Some(i64_identity.clone());
    assert_ne!(
        super::semantic_function_dedup_key(&char_fn),
        super::semantic_function_dedup_key(&i64_fn)
    );
    let mut char_other_artifact = char_fn.clone();
    char_other_artifact.trait_impl_id = Some(12);
    assert_eq!(
        super::semantic_function_dedup_key(&char_fn),
        super::semantic_function_dedup_key(&char_other_artifact)
    );
}

/// Inherent `Owner::setitem` and `impl Store for Owner { fn setitem }`
/// share a leaf. Charon records two FunDeclIds; the call graph follows
/// each one (`method-call-expr`, `CallControl.graphs_from`).
fn inherent_vs_trait_setitem_fixture() -> Llbc {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 10}
    }});
    let ident = |s: &str| serde_json::json!({"Ident": [s, 0]});
    let meta = |name: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": name,
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": is_local
        })
    };
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let owner_adt = serde_json::json!({
        "Adt": {"id": 0, "generics": generics.clone(), "builtin": null}
    });
    let owner_ty = serde_json::json!({"Value": [0, owner_adt.clone()]});
    let self_ty = serde_json::json!({"Ref": [null, owner_ty.clone(), "Mut"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let field_attr = serde_json::json!({
        "attributes": [], "inline": null, "rename": null, "public": true
    });
    let owner = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(vec![
            ident("fixture"), ident("celldict"), ident("Owner")
        ], true),
        "kind": {"Struct": [{
            "name": "items",
            "ty": {"Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}]},
            "attr_info": field_attr
        }]}
    });
    let inherent_impl = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": owner_ty.clone(),
        "kind": "InherentImplBlock"
    }}});
    let trait_impl_seg = serde_json::json!({"Impl": {"Trait": 0}});
    let method_locals = serde_json::json!([
        {"index": 0, "name": null, "span": span.clone(), "ty": bool_ty.clone()},
        {"index": 1, "name": "self", "span": span.clone(), "ty": self_ty.clone()}
    ]);
    let return_body = serde_json::json!({
        "Unstructured": {
            "span": span.clone(),
            "locals": {"arg_count": 1, "locals": method_locals},
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span.clone(), "kind": "Return"}
            }]
        }
    });
    let inherent_setitem = serde_json::json!({
        "def_id": 2,
        "item_meta": meta(vec![
            ident("fixture"), ident("celldict"), inherent_impl, ident("setitem")
        ], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [self_ty.clone()],
            "output": bool_ty.clone()
        },
        "body": return_body.clone()
    });
    let trait_setitem = serde_json::json!({
        "def_id": 3,
        "item_meta": meta(vec![
            ident("fixture"), ident("celldict"), trait_impl_seg, ident("setitem")
        ], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [self_ty.clone()],
            "output": bool_ty.clone()
        },
        "body": return_body
    });
    let caller = |def_id: u64, name: &str, callee: u64| {
        let locals = vec![
            serde_json::json!({
                "index": 0, "name": null, "span": span.clone(), "ty": bool_ty.clone()
            }),
            serde_json::json!({
                "index": 1, "name": "self", "span": span.clone(), "ty": self_ty.clone()
            }),
        ];
        serde_json::json!({
            "def_id": def_id,
            "item_meta": meta(vec![ident("fixture"), ident(name)], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [self_ty.clone()],
                "output": bool_ty.clone()
            },
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 1, "locals": locals},
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span.clone(),
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": callee},
                                                "generics": generics.clone()
                                            }
                                        },
                                        "args": [{
                                            "Read": {
                                                "kind": {"Local": 1},
                                                "ty": self_ty.clone()
                                            }
                                        }]}, "dest": {
                                            "kind": {"Local": 0},
                                            "ty": bool_ty.clone()
                                        },
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span.clone(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span.clone(), "kind": "Return"}
                    }
                ]
            }}
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [owner],
            "fun_decls": [
                caller(0, "call_inherent", 2),
                caller(1, "call_trait", 3),
                inherent_setitem,
                trait_setitem
            ],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(vec![
                    ident("fixture"), ident("celldict"), ident("Store")
                ], true),
                "methods": [{"skip_binder": {"name": "setitem"}}]
            }],
            "trait_impls": [{
                "impl_trait": {
                    "id": 0,
                    "generics": {"types": [owner_ty]}
                },
                "methods": []
            }]
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("inherent vs trait setitem fixture parses")
}

fn first_call_binding(graph: &crate::model::FunctionGraph) -> (Option<u64>, Vec<String>) {
    use crate::model::{CallTarget, OpKind};
    for op in graph_ops(graph) {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        return match target {
            CallTarget::Method {
                resolved_path,
                fun_decl_id,
                ..
            } => (
                *fun_decl_id,
                resolved_path
                    .as_ref()
                    .map(|path| path.segments.clone())
                    .unwrap_or_default(),
            ),
            CallTarget::FunctionPath {
                segments,
                fun_decl_id,
                ..
            } => (*fun_decl_id, segments.clone()),
            other => panic!("expected a FunDecl call, got {other:?}"),
        };
    }
    panic!("caller graph has no Call")
}

#[test]
fn spec_segments_distinguishes_inherent_and_trait_impl_methods() {
    use crate::parse::CallPath;

    let llbc = inherent_vs_trait_setitem_fixture();
    let inherent = llbc.fn_by_id(2).expect("inherent setitem");
    let trait_impl = llbc.fn_by_id(3).expect("trait setitem");
    let inherent_owner =
        super::impl_method_owner_for_fundecl(&llbc, inherent).expect("inherent Owner");
    let trait_owner = super::impl_method_owner_for_fundecl(&llbc, trait_impl).expect("trait Owner");
    assert_eq!(
        inherent_owner,
        ("celldict::Owner".to_string(), "setitem".to_string())
    );
    assert_eq!(trait_owner, inherent_owner);
    assert_eq!(super::trait_impl_id_for_fundecl(inherent), None);
    assert_eq!(super::trait_impl_id_for_fundecl(trait_impl), Some(0));
    let inherent_path = super::spec_segments(&llbc, inherent, "setitem");
    let trait_path = super::spec_segments(&llbc, trait_impl, "setitem");
    let identity =
        crate::front::clause_spec::render_trait_impl(&llbc, 0).expect("Store for Owner identity");
    assert_eq!(
        inherent_path,
        CallPath::for_impl_method("celldict::Owner", "setitem").segments
    );
    assert_eq!(
        trait_path,
        super::trait_impl_identity_path("celldict::Owner", &identity, "setitem")
    );
    assert_ne!(inherent_path, trait_path);
    assert!(!inherent_path.iter().any(|s| s.starts_with("<impl ")));
    assert!(trait_path.iter().any(|s| s.starts_with("<impl ")));
    assert_eq!(
        super::registered_path_for_fun_decl(&llbc, inherent).segments,
        inherent_path
    );
    assert_eq!(
        super::registered_path_for_fun_decl(&llbc, trait_impl).segments,
        trait_path
    );

    let inherent_graph =
        crate::test_support::lower_function(&llbc, "call_inherent").expect("lower call_inherent");
    let trait_graph =
        crate::test_support::lower_function(&llbc, "call_trait").expect("lower call_trait");
    let (inherent_id, inherent_bound) = first_call_binding(&inherent_graph);
    let (trait_id, trait_bound) = first_call_binding(&trait_graph);
    assert_eq!(inherent_id, Some(2));
    assert_eq!(trait_id, Some(3));
    assert_eq!(inherent_bound, inherent_path);
    assert_eq!(trait_bound, trait_path);
}

/// Foreign payload-enum associated fn whose FunDecl is Opaque (the
/// owner crate is not extracted) while the TypeDecl is a full Enum.
/// Mirrors `majit_gc::GcAllocOutcome::from_hook`: one non-`self`
/// argument, result the payload enum.  A sibling `self` method on
/// the same owner (`allocated_or_abort`) stays out of the harvest.
fn gc_alloc_outcome_from_hook_fixture() -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let item_meta = |name: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": name,
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
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let outcome_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics.clone()}
    });
    let raw_ptr_ty = serde_json::json!({
        "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]
    });
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, outcome_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let outcome = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(vec![ident("majit_gc"), ident("GcAllocOutcome")], false),
        "kind": {
            "Enum": [
                {
                    "name": "Allocated",
                    "fields": [{
                        "name": null,
                        "is_positional": true,
                        "ty": raw_ptr_ty.clone(),
                        "attr_info": null
                    }],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}
                },
                {
                    "name": "Failed",
                    "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}
                },
                {
                    "name": "NoRoute",
                    "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", "2"]}}
                }
            ]
        }
    });
    let from_hook = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(
            vec![ident("majit_gc"), impl_seg.clone(), ident("from_hook")],
            false,
        ),
        "signature": {
            "is_unsafe": false,
            "inputs": [raw_ptr_ty.clone()],
            "output": outcome_ty.clone()
        },
        "body": "Opaque"
    });
    let allocated_or_abort = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta(
            vec![ident("majit_gc"), impl_seg, ident("allocated_or_abort")],
            false,
        ),
        "signature": {
            "is_unsafe": false,
            "inputs": [outcome_ty.clone(), {"Scalar": {"Integer": {"Unsigned": "Usize"}}}],
            "output": raw_ptr_ty
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [outcome],
            "fun_decls": [from_hook, allocated_or_abort],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes())
        .expect("GcAllocOutcome::from_hook fixture Llbc parses")
}

#[test]
fn foreign_payload_enum_associated_fn_is_harvested_as_function_path() {
    // Call-site lookup is `impl_method_owner_for_fundecl` +
    // `strip_crate_prefix` → `["GcAllocOutcome", "from_hook"]`.
    // The harvest must use that same key and keep the payload-enum
    // class root so the residual intern's a ClassDef, not a
    // classdef-less `SomeInstance`.
    let llbc = gc_alloc_outcome_from_hook_fixture();
    let harvested = super::collect_foreign_opaque_method_externals(&llbc);
    let row = harvested.iter().find(|(segments, _, _)| {
        segments == &["GcAllocOutcome".to_string(), "from_hook".to_string()]
    });
    assert!(
        row.is_some(),
        "from_hook must be harvested under [GcAllocOutcome, from_hook]: {harvested:?}"
    );
    let (_, _, result_ty) = row.unwrap();
    assert_eq!(
        result_ty,
        &crate::model::ValueType::Ref(Some("GcAllocOutcome".into()))
    );
    assert!(
        !harvested
            .iter()
            .any(|(segments, _, _)| segments.last().map(String::as_str)
                == Some("allocated_or_abort")),
        "RawPtr self-method on a payload enum is declined by result type: {harvested:?}"
    );
}

/// Foreign opaque rust-heap `alloc::string::String::new`: TypeDecl is
/// Opaque (the alloc crate is not extracted), FunDecl is a zero-arg
/// inherent method whose result is that ADT. Harvest must keep the
/// str-family class root so the residual annotates `SomeString`, not a
/// classdef-less `SomeInstance`.
fn string_new_opaque_fixture() -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let item_meta = |name: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": name,
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
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let string_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics.clone()}
    });
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, string_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let string_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(
            vec![ident("alloc"), ident("string"), ident("String")],
            false,
        ),
        "kind": "Opaque"
    });
    let string_new = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(
            vec![ident("alloc"), ident("string"), impl_seg, ident("new")],
            false,
        ),
        "signature": {
            "is_unsafe": false,
            "inputs": [],
            "output": string_ty
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [string_decl],
            "fun_decls": [string_new],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("String::new opaque fixture Llbc parses")
}

#[test]
fn foreign_opaque_string_new_keeps_str_family_root() {
    let llbc = string_new_opaque_fixture();
    let harvested = super::collect_foreign_opaque_method_externals(&llbc);
    let row = harvested.iter().find(|(segments, _, _)| {
        segments.last().map(String::as_str) == Some("new")
            && segments.iter().any(|s| s == "String" || s == "string")
    });
    assert!(
        row.is_some(),
        "String::new must be harvested: {harvested:?}"
    );
    let (_, _, result_ty) = row.unwrap();
    let crate::model::ValueType::Ref(Some(root)) = result_ty else {
        panic!("String::new residual must keep a named Ref, got {result_ty:?}");
    };
    assert!(
        super::is_str_family_class_root(root),
        "String::new residual class root must be str-family, got {root:?}"
    );
}

/// Foreign opaque associated constructor: TypeDecl is Opaque, FunDecl is
/// a one-argument inherent method whose result is that ADT and whose
/// first input is the payload, not `self`. `rthread.py` `allocate_lock`
/// residualizes `ll_allocate_lock` the same way — the lock constructor
/// is `extfunc.py` `register_external`, not a look-inside graph.
fn opaque_associated_constructor_fixture() -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let item_meta = |name: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": name,
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
    };
    let generics = serde_json::json!({
        "regions": [],
        "types": [],
        "const_generics": [],
        "trait_refs": []
    });
    let mutex_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics.clone()}
    });
    let payload_ty = serde_json::json!({"Scalar": "Bool"});
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, mutex_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let mutex_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(
            vec![ident("lock_api"), ident("mutex"), ident("Mutex")],
            false,
        ),
        "kind": "Opaque"
    });
    let mutex_new = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(
            vec![ident("lock_api"), ident("mutex"), impl_seg, ident("new")],
            false,
        ),
        "signature": {
            "is_unsafe": false,
            "inputs": [payload_ty],
            "output": mutex_ty
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [mutex_decl],
            "fun_decls": [mutex_new],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes())
        .expect("opaque associated constructor fixture Llbc parses")
}

#[test]
fn foreign_opaque_associated_constructor_is_harvested() {
    let llbc = opaque_associated_constructor_fixture();
    let harvested = super::collect_foreign_opaque_method_externals(&llbc);
    let row = harvested.iter().find(|(segments, _, _)| {
        segments.last().map(String::as_str) == Some("new") && segments.iter().any(|s| s == "Mutex")
    });
    assert!(
        row.is_some(),
        "one-arg opaque associated constructor must be harvested: {harvested:?}"
    );
    let (segments, signature, result_ty) = row.unwrap();
    assert_eq!(
        signature.argnames.len(),
        1,
        "payload argument: {segments:?}"
    );
    assert_eq!(
        result_ty,
        &crate::model::ValueType::Ref(None),
        "opaque constructor residual is classdef-less SomeInstance, got {result_ty:?}"
    );
}

/// Foreign opaque `HashMap` method returning pointer-niche `Option<P>`.
fn hashmap_opaque_option_method_fixture(
    method: &str,
    payload: serde_json::Value,
    self_mut: bool,
    gc_pointee: bool,
) -> Llbc {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let item_meta = |name: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": name,
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
    };
    let generics = empty_generics();
    let hashmap_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": generics.clone()}
    });
    let option_of = option_adt_node(1, payload);
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, hashmap_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let hashmap_decl = opaque_type_decl(0, &["alloc", "collections", "hashmap", "HashMap"]);
    let option_decl = option_enum(1);
    let mut type_decls = vec![hashmap_decl, option_decl];
    let mut trait_decls = Vec::new();
    let mut trait_impls = Vec::new();
    if gc_pointee {
        type_decls.push(code_struct(2, &["pyre_object", "pyobject", "PyObject"]));
        let gc_obj_ty = serde_json::json!({
            "Adt": {"id": 2, "generics": generics.clone()}
        });
        trait_decls.push(serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(
                vec![ident("majit_gc"), ident("header"), ident("GcType")],
                false,
            ),
        }));
        trait_impls.push(serde_json::json!({
            "impl_trait": {
                "id": 0,
                "generics": {"types": [{"Value": [2, gc_obj_ty]}]}
            }
        }));
    }
    let self_kind = if self_mut { "Mut" } else { "Shared" };
    let hashmap_method = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta(
            vec![
                ident("alloc"),
                ident("collections"),
                ident("hashmap"),
                impl_seg,
                ident(method),
            ],
            false,
        ),
        "signature": {
            "is_unsafe": false,
            "inputs": [
                {"Ref": ["Erased", hashmap_ty, self_kind]},
                {"Ref": ["Erased", {"Scalar": {"Integer": {"Unsigned": "Usize"}}}, "Shared"]},
            ],
            "output": option_of
        },
        "body": "Opaque"
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": type_decls,
            "fun_decls": [hashmap_method],
            "global_decls": [],
            "trait_decls": trait_decls,
            "trait_impls": trait_impls
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("HashMap opaque fixture Llbc parses")
}

/// Foreign opaque `HashMap::get` returning pointer-niche `Option<&V>`.
/// `tyref_to_value_type_with` names the pointee and drops Option
/// nullability; collection keeps the Option spelling so intern can
/// noneify (`Bookkeeper::niche_option_gc_payload_annotation`).
fn hashmap_get_option_ref_fixture() -> Llbc {
    let generics = empty_generics();
    let gc_obj_ty = serde_json::json!({
        "Adt": {"id": 2, "generics": generics}
    });
    hashmap_opaque_option_method_fixture("get", option_shared_ref(gc_obj_ty), false, true)
}

/// Opaque `HashMap::get_mut` of a scalar value: `Option<&mut i64>`.
fn hashmap_get_mut_option_mut_i64_fixture() -> Llbc {
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    hashmap_opaque_option_method_fixture(
        "get_mut",
        serde_json::json!({"Ref": ["Erased", i64_ty, "Mut"]}),
        true,
        false,
    )
}

#[test]
fn foreign_opaque_hashmap_get_keeps_niche_option_spelling() {
    let llbc = hashmap_get_option_ref_fixture();
    let harvested = super::collect_foreign_opaque_method_externals(&llbc);
    let row = harvested.iter().find(|(segments, _, _)| {
        segments.last().map(String::as_str) == Some("get")
            && segments.iter().any(|s| s == "HashMap" || s == "hashmap")
    });
    assert!(
        row.is_some(),
        "HashMap::get must be harvested: {harvested:?}"
    );
    let (_, _, result_ty) = row.unwrap();
    let crate::model::ValueType::Ref(Some(root)) = result_ty else {
        panic!("HashMap::get residual must keep a named Ref, got {result_ty:?}");
    };
    assert!(
        root.contains("Option<") && root.contains('&') && root.contains("PyObject"),
        "HashMap::get residual must keep Option<&T> spelling, got {root:?}"
    );
}

#[test]
fn foreign_opaque_hashmap_get_mut_scalar_does_not_keep_niche_option_spelling() {
    let llbc = hashmap_get_mut_option_mut_i64_fixture();
    let harvested = super::collect_foreign_opaque_method_externals(&llbc);
    let row = harvested.iter().find(|(segments, _, _)| {
        segments.last().map(String::as_str) == Some("get_mut")
            && segments.iter().any(|s| s == "HashMap" || s == "hashmap")
    });
    if let Some((_, _, result_ty)) = row {
        match result_ty {
            crate::model::ValueType::Ref(Some(root)) => {
                assert!(
                    !(root.contains("Option<") && root.contains("&mut") && root.contains("i64")),
                    "scalar HashMap::get_mut must not keep Option<&mut i64> spelling, got {root:?}"
                );
                assert!(
                    !super::is_niche_option_gc_payload_spelling(root, None),
                    "scalar HashMap::get_mut must not stamp a GC-niche Option, got {root:?}"
                );
            }
            _ => {}
        }
    }
}

#[test]
fn range_contains_is_not_harvested_as_function_path() {
    // A bodyless `Range::contains` is not harvested as an opaque
    // external; the FunctionPath stays unregistered (census Skip).
    let llbc = range_contains_fixture();
    let harvested = super::collect_foreign_opaque_method_externals(&llbc);
    assert!(
        !harvested
            .iter()
            .any(|(segments, _, _)| segments.last().map(String::as_str) == Some("contains")),
        "Range::contains must not be harvested: {harvested:?}"
    );
}

fn gc_ref_ty() -> serde_json::Value {
    serde_json::json!({"RawPtr": [named_adt(1, vec![]), "Mut"]})
}

fn vec_gcref_type_decls() -> Vec<serde_json::Value> {
    vec![
        type_decl_struct(0, &["alloc", "vec", "Vec"], vec![]),
        type_decl_struct(1, &["pyre_object", "pyobject", "PyObject"], vec![]),
    ]
}

fn assert_trait_clone_is_list(
    caller: &str,
    self_ty: serde_json::Value,
    decls: Vec<serde_json::Value>,
) {
    let llbc = trait_clone_call_fixture(caller, self_ty, decls);
    let graph = crate::test_support::lower_function(&llbc, caller)
        .expect("lower identity-element Vec Clone");
    let ops = graph_ops(&graph);
    let leaves = call_leafs(&ops);
    assert!(
        leaves.iter().any(|leaf| *leaf == "list"),
        "Vec Clone::clone with identity-copy T must become the list builtin; ops={ops:?}"
    );
    assert!(
        !leaves.iter().any(|leaf| *leaf == "clone"),
        "Vec Clone::clone with identity-copy T must not residualize; ops={ops:?}"
    );
}

#[test]
fn trait_clone_of_vec_i64_is_list_builtin() {
    let vec_ty = vec_of(i64_ty());
    let payload = trait_clone_kind_payload(vec_ty.clone());
    let llbc = trait_clone_call_fixture("clone_vec_i64", vec_ty.clone(), vec_type_decls());
    assert_eq!(
        super::specialized_trait_clone_list_target(&payload, &llbc).map(|(path, _)| path),
        Some(vec!["list".to_string()]),
        "CallKind::Trait retarget must send Vec<i64>::clone to list"
    );
    assert_trait_clone_is_list("clone_vec_i64", vec_ty, vec_type_decls());
}

#[test]
fn trait_clone_of_vec_gcref_is_list_builtin() {
    let vec_ty = vec_of(gc_ref_ty());
    let payload = trait_clone_kind_payload(vec_ty.clone());
    let llbc = trait_clone_call_fixture("clone_vec_gcref", vec_ty.clone(), vec_gcref_type_decls());
    assert_eq!(
        super::specialized_trait_clone_list_target(&payload, &llbc).map(|(path, _)| path),
        Some(vec!["list".to_string()]),
        "CallKind::Trait retarget must send Vec<GC-ref>::clone to list"
    );
    assert_trait_clone_is_list("clone_vec_gcref", vec_ty, vec_gcref_type_decls());
}

#[test]
fn transparent_scalar_struct_uses_inner_register_bank() {
    let type_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["Word", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 16}
            }},
            "source_text": "struct Word(i64);",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "kind": {"Struct": [{
            "name": null,
            "ty": {"Value": [7, {"Scalar": {"Integer": {"Signed": "I64"}}}]},
            "attr_info": null
        }]},
        "layout": [{
            "key": "x86_64-unknown-linux-gnu",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"repr_algo": "Rust", "transparent": true}
            }
        }]
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [null, type_decl],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let word_ty = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [8, {
            "Adt": {"id": 1, "generics": {"types": []}}
        }]
    }))
    .expect("fixture TyRef parses");

    assert_eq!(
        super::tyref_to_value_type(&word_ty, &llbc),
        crate::model::ValueType::Int
    );
    assert!(
        super::tyref_by_value_struct_decl(
            &word_ty,
            &llbc,
            no_tombstoned_leaves(),
            &std::collections::HashSet::new(),
            &std::collections::HashSet::new(),
        )
        .is_none(),
        "a transparent wrapper over a scalar is that scalar's one-word field"
    );
}

/// `BuildHasherDefault<H>` is Opaque in std LLBC and has no layout.
/// RPython `RDict` has no hasher field (`rordereddict.py`).
#[test]
fn build_hasher_default_without_layout_is_void_zst() {
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": fixture_item_meta(ident_path(&["core", "hash", "BuildHasherDefault"])),
                "kind": "Opaque"
            }],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let ty = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [1, {
            "Adt": {"id": 0, "generics": {"types": []}}
        }]
    }))
    .expect("fixture TyRef parses");
    assert!(
        super::tyref_is_zero_sized(&ty, &llbc),
        "Opaque BuildHasherDefault without layout is Void"
    );
    assert!(
        super::tyref_is_void_zst(&ty, &llbc),
        "hasher param / field is lltype.Void"
    );
    assert!(
        super::tyref_default_is_unit(&ty, &llbc),
        "Trait Default of Opaque BuildHasherDefault is unit"
    );
}

fn infallible_enum(def_id: u64) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": fixture_item_meta(ident_path(&["core", "convert", "Infallible"])),
        "kind": {"Enum": []}
    })
}

fn phantom_data_opaque(def_id: u64) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": fixture_item_meta(ident_path(&["core", "marker", "PhantomData"])),
        "kind": "Opaque"
    })
}

fn inhabited_never_struct(def_id: u64) -> serde_json::Value {
    type_decl_struct(
        def_id,
        &["Never"],
        vec![serde_json::json!({
            "name": null,
            "ty": {"Scalar": {"Integer": {"Unsigned": "U32"}}},
            "attr_info": null
        })],
    )
}

fn inhabited_infallible_enum(def_id: u64) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": fixture_item_meta(ident_path(&["Infallible"])),
        "kind": {"Enum": [{
            "name": "A",
            "fields": [],
            "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
        }]}
    })
}

/// `Option<U>` is Void only when `U` is an uninhabited atom.
/// `PhantomData<Infallible>` is Void via the PhantomData leaf.
#[test]
fn option_of_uninhabited_atom_is_void_zst() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            infallible_enum(1),
            phantom_data_opaque(2),
            type_decl_struct(3, &["alloc", "vec", "Vec"], vec![]),
        ],
        vec![],
    );
    let opt = |arg: serde_json::Value| TyRef::Other(option_adt_node(0, arg));

    let inf_atom = TyRef::Other(adt_node(1));
    assert!(super::tyref_decl_is_uninhabited(&inf_atom, &llbc));
    assert!(super::tyref_is_void_zst(&inf_atom, &llbc));
    let never_kind = TyRef::Other(serde_json::json!("Never"));
    assert!(super::tyref_is_never(&never_kind, &llbc));
    assert!(super::tyref_is_void_zst(&never_kind, &llbc));

    for (label, arg) in [
        ("Never", serde_json::json!("Never")),
        ("!", serde_json::json!("!")),
        ("Infallible", adt_node(1)),
        ("Value Never", serde_json::json!({"Value": [99, "Never"]})),
    ] {
        let ty = opt(arg);
        assert!(super::tyref_is_option_of_uninhabited(&ty, &llbc), "{label}");
        assert!(super::tyref_is_void_zst(&ty, &llbc), "{label}");
        assert_eq!(
            super::tyref_to_value_type(&ty, &llbc),
            crate::model::ValueType::Void,
            "{label}"
        );
    }

    let opt_ptr = opt(serde_json::json!({"RawPtr": [adt_node(1), "Const"]}));
    assert!(!super::tyref_is_option_of_uninhabited(&opt_ptr, &llbc));
    assert_ne!(
        super::tyref_to_value_type(&opt_ptr, &llbc),
        crate::model::ValueType::Void
    );

    let opt_vec = opt(named_adt(3, vec![adt_node(1)]));
    assert!(!super::tyref_is_option_of_uninhabited(&opt_vec, &llbc));
    assert_ne!(
        super::tyref_to_value_type(&opt_vec, &llbc),
        crate::model::ValueType::Void
    );

    let phantom = named_adt(2, vec![adt_node(1)]);
    let opt_phantom = opt(phantom.clone());
    assert!(!super::tyref_is_option_of_uninhabited(&opt_phantom, &llbc));
    assert_ne!(
        super::tyref_to_value_type(&opt_phantom, &llbc),
        crate::model::ValueType::Void
    );

    let phantom_ty = TyRef::Other(phantom);
    assert!(
        super::tyref_is_void_zst(&phantom_ty, &llbc),
        "PhantomData<Infallible> is Void via PhantomData"
    );
    assert!(!super::tyref_is_option_of_uninhabited(&phantom_ty, &llbc));
}

/// `struct Never(u32)` and `enum Infallible { A }` keep their storage.
/// `_names_without_voids` (`lltype.py`) drops a field whose lowered
/// type IS Void; a user Adt named Never / Infallible is inhabited.
#[test]
fn inhabited_never_struct_and_infallible_enum_are_not_void_zst() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            inhabited_never_struct(1),
            inhabited_infallible_enum(2),
            type_decl_struct(
                3,
                &["Holder"],
                vec![
                    serde_json::json!({
                        "name": "n",
                        "ty": adt_node(1),
                        "attr_info": null
                    }),
                    serde_json::json!({
                        "name": "i",
                        "ty": adt_node(2),
                        "attr_info": null
                    }),
                    serde_json::json!({
                        "name": "on",
                        "ty": option_adt_node(0, adt_node(1)),
                        "attr_info": null
                    }),
                    serde_json::json!({
                        "name": "oi",
                        "ty": option_adt_node(0, adt_node(2)),
                        "attr_info": null
                    }),
                ],
            ),
        ],
        vec![],
    );
    let never_ty = TyRef::Other(adt_node(1));
    let inf_ty = TyRef::Other(adt_node(2));
    assert!(!super::tyref_is_never(&never_ty, &llbc));
    assert!(!super::tyref_decl_is_uninhabited(&never_ty, &llbc));
    assert!(!super::tyref_is_void_zst(&never_ty, &llbc));
    assert!(!super::tyref_decl_is_uninhabited(&inf_ty, &llbc));
    assert!(!super::tyref_is_void_zst(&inf_ty, &llbc));

    let opt_never = TyRef::Other(option_adt_node(0, adt_node(1)));
    let opt_inf = TyRef::Other(option_adt_node(0, adt_node(2)));
    assert!(!super::tyref_is_option_of_uninhabited(&opt_never, &llbc));
    assert!(!super::tyref_is_void_zst(&opt_never, &llbc));
    assert!(!super::tyref_is_option_of_uninhabited(&opt_inf, &llbc));
    assert!(!super::tyref_is_void_zst(&opt_inf, &llbc));

    let td = llbc.type_by_id(3).expect("Holder TypeDecl");
    let TypeDeclKind::Struct(fields) = &td.kind else {
        panic!("Holder is a struct");
    };
    for f in fields {
        let label = f.name.as_deref().unwrap_or("?");
        assert!(
            !super::tyref_is_void_zst(&f.ty, &llbc),
            "Holder.{label} stays stored"
        );
    }
}

/// `type T = BuildHasherDefault<H>` peels to the hasher ZST
/// (`type_node_peel_aliases`). RPython `RDict` has no hasher field
/// (`rordereddict.py`).
#[test]
fn alias_of_build_hasher_default_is_void_zst() {
    let hasher = named_adt(1, vec![adt_node(2)]);
    let llbc = llbc_with_types(
        "fixture",
        vec![
            type_decl_alias(
                0,
                &["dictmultiobject", "ObjectKeyBuildHasher"],
                hasher.clone(),
            ),
            type_decl_struct(1, &["core", "hash", "BuildHasherDefault"], vec![]),
            type_decl_struct(2, &["dictmultiobject", "ObjectKeyHasher"], vec![]),
        ],
        vec![],
    );
    let alias_ty = TyRef::Other(adt_node(0));
    assert!(
        super::tyref_is_zero_sized(&alias_ty, &llbc),
        "ObjectKeyBuildHasher alias peels to BuildHasherDefault"
    );
    assert!(super::tyref_is_void_zst(&alias_ty, &llbc));
    let direct = TyRef::Other(hasher);
    assert!(super::tyref_is_void_zst(&direct, &llbc));
}

/// `RDict.hash_builder: S` instantiated at `BuildHasherDefault<H>` is
/// Void (`substitute_spill_typevars`). `_names_without_voids`
/// (`lltype.py`) drops a field whose lowered type IS Void.
#[test]
fn substituted_build_hasher_default_typevar_field_is_void_zst() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            type_decl_struct(
                0,
                &["rordereddict", "RDict"],
                vec![serde_json::json!({
                    "name": "hash_builder",
                    "ty": {"TypeVar": {"Bound": [0, 2]}},
                    "attr_info": null
                })],
            ),
            type_decl_struct(1, &["dictmultiobject", "BytesKey"], vec![]),
            type_decl_struct(2, &["pyobject", "PyObject"], vec![]),
            type_decl_struct(3, &["core", "hash", "BuildHasherDefault"], vec![]),
            type_decl_struct(4, &["dictmultiobject", "ObjectKeyHasher"], vec![]),
        ],
        vec![],
    );
    let field_ty = TyRef::Other(serde_json::json!({"TypeVar": {"Bound": [0, 2]}}));
    assert!(!super::tyref_is_void_zst(&field_ty, &llbc));
    let owner = named_adt(
        0,
        vec![adt_node(1), adt_node(2), named_adt(3, vec![adt_node(4)])],
    );
    let subst = crate::front::mir::substitute_spill_typevars(&field_ty, &owner, &llbc)
        .expect("S instantiates to BuildHasherDefault");
    assert!(
        super::tyref_is_void_zst(&subst, &llbc),
        "instantiated hasher field is lltype.Void"
    );
}

#[test]
fn empty_tuple_tyref_is_void() {
    // `()` is `lltype.Void` (`history.py` `getkind`). Charon emits
    // an empty Tuple ADT; intern as Void so `Result<(), E>` does
    // not join Tuple ∪ OsError (`native_result`).
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let ty = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}
    }))
    .expect("fixture TyRef parses");
    assert!(super::is_unit_type(&ty, &llbc));
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        crate::model::ValueType::Void
    );
}

/// `rtuple.py TupleRepr` preserves each item's low-level representation.
#[test]
fn shared_byte_window_tuple_keeps_both_words_without_aliasing_a_mutable_list() {
    let llbc = empty_llbc();
    let tuple = |borrow| {
        TyRef::Other(serde_json::json!({"Adt": {
            "builtin": "Tuple",
            "generics": {"types": [{"Ref": ["'_", {
                "Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, null]
            }, borrow]}]}
        }}))
    };
    let shared = tyref_positional_aggregate_root(&tuple("Shared"), &llbc).unwrap();
    let mutable = tyref_positional_aggregate_root(&tuple("Mut"), &llbc).unwrap();
    assert_ne!(shared, mutable, "window and byte list have distinct reprs");
    let cc = crate::codewriter::call::CallControl::new();
    let layout = cc
        .struct_layout_for(&shared)
        .expect("shared window tuple layout");
    let word = crate::layout::target_word_size();
    assert_eq!(layout.size, 2 * word);
    assert_eq!(
        layout
            .fields
            .iter()
            .map(|field| (field.name.as_str(), field.offset))
            .collect::<Vec<_>>(),
        vec![("__pos_0", 0), ("__pos_0.len", word)]
    );
    let mutable_layout = cc
        .struct_layout_for(&mutable)
        .expect("mutable byte-list tuple layout");
    assert!(
        mutable_layout
            .fields
            .iter()
            .all(|field| field.name != "__pos_0.len")
    );
}

fn indexed_rows(rows: Vec<(u64, serde_json::Value)>) -> serde_json::Value {
    let max = rows.iter().map(|(id, _)| *id).max().unwrap_or(0) as usize;
    let mut slots = vec![serde_json::Value::Null; max + 1];
    for (id, row) in rows {
        slots[id as usize] = row;
    }
    serde_json::Value::Array(slots)
}

fn empty_fields() -> serde_json::Value {
    serde_json::json!([])
}

fn raw_ptr_field(pointee: u64) -> serde_json::Value {
    serde_json::json!([{
        "name": null,
        "ty": {"RawPtr": [
            {"Adt": {"id": pointee, "generics": {"types": []}}},
            "Mut"
        ]},
        "attr_info": null
    }])
}

fn u64_field() -> serde_json::Value {
    serde_json::json!([{
        "name": null,
        "ty": {"Scalar": {"Integer": {"Unsigned": "U64"}}},
        "attr_info": null
    }])
}

fn two_i64_fields() -> serde_json::Value {
    serde_json::json!([
        {
            "name": null,
            "ty": {"Scalar": {"Integer": {"Signed": "I64"}}},
            "attr_info": null
        },
        {
            "name": null,
            "ty": {"Scalar": {"Integer": {"Signed": "I64"}}},
            "attr_info": null
        }
    ])
}

fn adt_body(def_id: u64) -> serde_json::Value {
    serde_json::json!({"Adt": {"id": def_id, "generics": {"types": []}}})
}

fn deref_impl(
    trait_id: u64,
    self_ty: serde_json::Value,
    targets: Vec<serde_json::Value>,
    witnesses: Vec<serde_json::Value>,
) -> serde_json::Value {
    let types = targets
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            serde_json::json!({
                "kind": {"TraitType": [trait_id, index]},
                "skip_binder": {"value": target}
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "impl_trait": {
            "id": trait_id,
            "generics": {
                "regions": [],
                "types": [self_ty],
                "const_generics": [],
                "trait_refs": []
            }
        },
        "types": types,
        "witnesses": witnesses
    })
}

fn handle_trait_decls() -> serde_json::Value {
    indexed_rows(vec![
        (
            0,
            serde_json::json!({
                "def_id": 0,
                "item_meta": fixture_item_meta(ident_path(&[
                    "majit_gc", "header", "GcType"
                ])),
                "methods": []
            }),
        ),
        (
            1,
            serde_json::json!({
                "def_id": 1,
                "item_meta": fixture_item_meta(ident_path(&[
                    "core", "ops", "deref", "Deref"
                ])),
                "methods": []
            }),
        ),
    ])
}

/// `harvest_declared_gc_facts` seeds `gc_struct_ids` from a majit-gc
/// `GcType` impl (`declared_gckind_follows_gctype_impl_and_field0_parent`).
fn gctype_impl_for_adt(type_id: u64) -> serde_json::Value {
    serde_json::json!({
        "impl_trait": {
            "id": 0,
            "generics": {"types": [{"Value": [type_id, adt_body(type_id)]}]}
        }
    })
}

fn fixture_gc_class_leaf(decl: &serde_json::Value) -> bool {
    let Some(name) = decl
        .pointer("/item_meta/name")
        .and_then(|name| name.as_array())
    else {
        return false;
    };
    let Some(leaf) = name
        .last()
        .and_then(|seg| seg.get("Ident"))
        .and_then(|ident| ident.as_array())
        .and_then(|ident| ident.first())
        .and_then(|ident| ident.as_str())
    else {
        return false;
    };
    matches!(leaf, "PyObject" | "PyErrorObject" | "OtherClass")
}

fn with_fixture_gctype_impls(
    type_decls: &[(u64, serde_json::Value)],
    trait_impls: serde_json::Value,
) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = type_decls
        .iter()
        .filter(|(_, decl)| fixture_gc_class_leaf(decl))
        .map(|(id, _)| gctype_impl_for_adt(*id))
        .collect();
    match trait_impls {
        serde_json::Value::Array(rest) => {
            rows.extend(rest);
            serde_json::Value::Array(rows)
        }
        other => {
            rows.push(other);
            serde_json::Value::Array(rows)
        }
    }
}

fn load_handle(
    type_decls: Vec<(u64, serde_json::Value)>,
    trait_impls: serde_json::Value,
    query: u64,
) -> (Llbc, TyRef) {
    let trait_impls = with_fixture_gctype_impls(&type_decls, trait_impls);
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": indexed_rows(type_decls),
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": trait_impls
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let ty = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [9000 + query, adt_body(query)]
    }))
    .expect("query TyRef parses");
    (llbc, ty)
}

fn paint_handle(
    type_decls: Vec<(u64, serde_json::Value)>,
    trait_impls: serde_json::Value,
    query: u64,
) -> ValueType {
    let (llbc, ty) = load_handle(type_decls, trait_impls, query);
    super::tyref_to_value_type(&ty, &llbc)
}

fn peel_transparent_inner(llbc: &Llbc, ty: &TyRef) -> Option<ValueType> {
    let gc = super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    super::tyref_transparent_inner_value_type(
        ty,
        llbc,
        no_tombstoned_leaves(),
        &gc,
        &std::collections::HashSet::<String>::new(),
    )
}

fn assert_handle_class(llbc: &Llbc, ty: &TyRef, class: Option<&str>) {
    let tomb = no_tombstoned_leaves();
    assert_eq!(
        super::tyref_class_root_with(ty, llbc, tomb).as_deref(),
        class
    );
    assert_eq!(
        super::enum_payload_instance_class_root(ty, llbc, tomb).as_deref(),
        class
    );
}

fn pointer_handle_decls() -> Vec<(u64, serde_json::Value)> {
    vec![
        (
            1,
            struct_decl(
                1,
                ident_path(&["fixture", "PyObject"]),
                empty_fields(),
                false,
            ),
        ),
        (
            2,
            struct_decl(
                2,
                ident_path(&["fixture", "PyErrorObject"]),
                empty_fields(),
                false,
            ),
        ),
        (
            3,
            struct_decl(
                3,
                ident_path(&["fixture", "PyError"]),
                raw_ptr_field(1),
                true,
            ),
        ),
        (
            4,
            struct_decl(
                4,
                ident_path(&["fixture", "OtherClass"]),
                empty_fields(),
                false,
            ),
        ),
        (
            5,
            struct_decl(
                5,
                ident_path(&["alloc", "vec", "Vec"]),
                empty_fields(),
                false,
            ),
        ),
    ]
}

fn pyerror_deref_impl(target_dedup: u64, target_adt: u64) -> serde_json::Value {
    deref_impl(
        1,
        serde_json::json!({"Deduplicated": 22}),
        vec![serde_json::json!({"Deduplicated": target_dedup})],
        vec![
            serde_json::json!({"Value": [22, adt_body(3)]}),
            serde_json::json!({"Value": [target_dedup, adt_body(target_adt)]}),
        ],
    )
}

/// A transparent GC-ref wrapper keeps its own class. The handle intern's
/// as a subclass of `Deref::Target`; the payload row stays a pointer to
/// that target.
#[test]
fn transparent_gc_handle_keeps_the_wrapper_class() {
    let decls = pointer_handle_decls();
    let blanket = serde_json::json!({
        "impl_trait": {
            "id": 1,
            "generics": {"types": [{"TypeVar": {"Bound": [0, 0]}}]}
        },
        "types": [{
            "kind": {"TraitType": [1, 0]},
            "skip_binder": {"value": adt_body(4)}
        }]
    });
    let (llbc, ty) = load_handle(
        decls.clone(),
        serde_json::json!([null, blanket, pyerror_deref_impl(23, 2)]),
        3,
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        ValueType::Ref(Some("PyError".into()))
    );
    assert_handle_class(&llbc, &ty, Some("PyError"));
    // The payload row is one pointer to the Deref target. A bare
    // `PyErrorObject` would be laid out as the whole object.
    assert_eq!(
        super::tyref_to_field_layout_string(
            &ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "*mut PyErrorObject"
    );
    // A captured `&PyError` keeps the handle class, not the payload
    // pointer `transparent_handle_field_spelling` would paint after
    // peeling `&`.
    let shared = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [9104, {"Ref": ["Erased", adt_body(3), "Shared"]}]
    }))
    .expect("shared-borrow TyRef parses");
    assert_eq!(
        super::tyref_to_field_layout_string(
            &shared,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "&PyError"
    );
    let ptr = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [9103, {"RawPtr": [adt_body(3), "Mut"]}]
    }))
    .expect("pointer TyRef parses");
    assert_eq!(
        super::enum_payload_instance_class_root(&ptr, &llbc, no_tombstoned_leaves()).as_deref(),
        Some("PyError")
    );
    assert_eq!(
        super::tyref_to_field_layout_string(
            &ptr,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "*mut PyErrorObject"
    );

    let same_class = paint_handle(
        decls.clone(),
        serde_json::json!([pyerror_deref_impl(23, 2), pyerror_deref_impl(25, 2)]),
        3,
    );
    assert_eq!(same_class, ValueType::Ref(Some("PyError".into())));

    let (llbc, ty) = load_handle(decls.clone(), serde_json::json!([null]), 3);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        ValueType::Ref(Some("PyObject".into()))
    );
    assert_handle_class(&llbc, &ty, Some("PyError"));
    assert_eq!(
        super::tyref_to_field_layout_string(
            &ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "PyError"
    );

    let mut scalar_decls = decls.clone();
    scalar_decls[2] = (
        3,
        struct_decl(3, ident_path(&["fixture", "Word"]), i64_field(), true),
    );
    let (llbc, ty) = load_handle(
        scalar_decls,
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert_eq!(super::tyref_to_value_type(&ty, &llbc), ValueType::Int);
    assert_handle_class(&llbc, &ty, Some("Word"));
    assert_eq!(
        super::tyref_to_field_layout_string(
            &ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "i64"
    );

    let split = paint_handle(
        decls.clone(),
        serde_json::json!([pyerror_deref_impl(23, 2), pyerror_deref_impl(24, 4)]),
        3,
    );
    assert_eq!(split, ValueType::Ref(Some("PyObject".into())));

    let ambiguous = deref_impl(
        1,
        serde_json::json!({"Deduplicated": 22}),
        vec![
            serde_json::json!({"Deduplicated": 23}),
            serde_json::json!({"Deduplicated": 24}),
        ],
        vec![
            serde_json::json!({"Value": [22, adt_body(3)]}),
            serde_json::json!({"Value": [23, adt_body(2)]}),
            serde_json::json!({"Value": [24, adt_body(4)]}),
        ],
    );
    assert_eq!(
        paint_handle(decls.clone(), serde_json::json!([ambiguous]), 3),
        ValueType::Ref(Some("PyObject".into()))
    );

    let vec_target = serde_json::json!({
        "Adt": {
            "id": 5,
            "generics": {"types": [{"Scalar": {"Integer": {"Signed": "I64"}}}]}
        }
    });
    let no_class = deref_impl(
        1,
        serde_json::json!({"Deduplicated": 22}),
        vec![serde_json::json!({"Deduplicated": 26})],
        vec![
            serde_json::json!({"Value": [22, adt_body(3)]}),
            serde_json::json!({"Value": [26, vec_target]}),
        ],
    );
    assert_eq!(
        paint_handle(decls, serde_json::json!([no_class]), 3),
        ValueType::Ref(Some("PyObject".into()))
    );

    let mut copy_name = ident_path(&["fixture", "Handle"]);
    copy_name
        .as_array_mut()
        .expect("path")
        .push(serde_json::json!({
            "Instantiated": {"skip_binder": {"types": [{"Scalar": "Bool"}]}}
        }));
    let mut other_copy = ident_path(&["fixture", "Handle"]);
    other_copy
        .as_array_mut()
        .expect("path")
        .push(serde_json::json!({
            "Instantiated": {"skip_binder": {"types": [
                {"Scalar": {"Integer": {"Signed": "I64"}}}
            ]}}
        }));
    let copies = vec![
        (
            1,
            struct_decl(
                1,
                ident_path(&["fixture", "PyObject"]),
                empty_fields(),
                false,
            ),
        ),
        (
            2,
            struct_decl(
                2,
                ident_path(&["fixture", "PyErrorObject"]),
                empty_fields(),
                false,
            ),
        ),
        (6, struct_decl(6, copy_name, raw_ptr_field(1), true)),
        (7, struct_decl(7, other_copy, raw_ptr_field(1), true)),
    ];
    let copy_impl = deref_impl(
        1,
        serde_json::json!({"Deduplicated": 22}),
        vec![serde_json::json!({"Deduplicated": 23})],
        vec![
            serde_json::json!({"Value": [22, adt_body(6)]}),
            serde_json::json!({"Value": [23, adt_body(2)]}),
        ],
    );
    let (llbc, ty) = load_handle(copies, serde_json::json!([copy_impl]), 7);
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        ValueType::Ref(Some("Handle".into()))
    );
    assert_handle_class(&llbc, &ty, Some("Handle"));
}

/// Intern ROW of a transparent handle is `*mut {Deref::Target}` for every
/// type-argument shape CAST still spells as the handle leaf: numeric `Adt`,
/// hash-cons `Value` / `Deduplicated` / `HashConsedValue`, a published
/// Path-form id, and a `type T = Handle` alias. A bare class name would
/// intern the payload as a by-value nested struct.
#[test]
fn intern_payload_handle_agrees_across_type_arg_shapes() {
    let mut decls = pointer_handle_decls();
    decls.push((
        6,
        type_decl_alias(6, &["fixture", "alias_mod", "PyError"], adt_body(3)),
    ));
    let (llbc, _) = load_handle(decls, serde_json::json!([pyerror_deref_impl(23, 2)]), 3);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw_alloc = std::collections::HashSet::<String>::new();
    let tomb = no_tombstoned_leaves();
    let payload = |arg: serde_json::Value| -> Vec<String> {
        let adt = serde_json::json!({
            "id": 0,
            "generics": {"types": [arg]}
        });
        super::render_adt_payload_type_args(
            adt.as_object().expect("adt map"),
            &llbc,
            0,
            &gc_struct_ids,
            &raw_alloc,
            tomb,
        )
    };
    let expected = payload(adt_body(3));
    assert_eq!(
        expected,
        vec!["*mut PyErrorObject".to_string()],
        "numeric Adt of the handle is *mut Deref::Target"
    );
    assert_eq!(
        payload(serde_json::json!({"Value": [22, adt_body(3)]})),
        expected,
        "Value-wrapped handle Adt"
    );
    assert_eq!(
        payload(serde_json::json!({"Deduplicated": 22})),
        expected,
        "Deduplicated handle Adt"
    );
    assert_eq!(
        payload(serde_json::json!({"HashConsedValue": [22, adt_body(3)]})),
        expected,
        "HashConsedValue-wrapped handle Adt"
    );
    assert_eq!(
        payload(serde_json::json!({
            "Adt": {"id": {"Path": "fixture::PyError"}, "generics": {"types": []}}
        })),
        expected,
        "Path-form TypeDeclRef of the handle"
    );
    assert_eq!(payload(adt_body(6)), expected, "type alias of the handle");
}

/// A `repr(transparent)` wrapper whose unique non-ZST field is a GC
/// pointer peels to that `Ref`. Construction and `.0` are identity.
/// The wrapper stays only when it is itself a registered GC class.
#[test]
fn transparent_gc_ptr_wrapper_peels_unless_wrapper_is_gc_class() {
    let gc_object_ref = ValueType::Ref(Some("PyObject".into()));

    // Handle intern (`Deref::Target`) keeps the wrapper class on the
    // value, but construction / `.0` still peel: the wrapper is not a
    // GcType struct (`gc_struct_ids` has `PyErrorObject`, not `PyError`).
    let (llbc, ty) = load_handle(
        pointer_handle_decls(),
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
    );
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let pyerror_sid = super::struct_id_for_type_decl(llbc.type_by_id(3).expect("PyError decl"));
    assert!(
        !gc.contains(&pyerror_sid),
        "PyError is a handle, not a GcType struct"
    );
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        ValueType::Ref(Some("PyError".into()))
    );
    assert_eq!(
        peel_transparent_inner(&llbc, &ty),
        Some(gc_object_ref.clone())
    );

    // `IdentityKey` is the same shape: `repr(transparent)` over a GC
    // pointer, no `GcType` on the wrapper. Peel to the field `Ref`.
    // Native construction / `.0` are identity on that word.
    let mut identity_decls = pointer_handle_decls();
    identity_decls.push((
        6,
        struct_decl(
            6,
            ident_path(&["fixture", "IdentityKey"]),
            raw_ptr_field(1),
            true,
        ),
    ));
    let (llbc, ty) = load_handle(identity_decls, serde_json::json!([null]), 6);
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let identity_sid =
        super::struct_id_for_type_decl(llbc.type_by_id(6).expect("IdentityKey decl"));
    assert!(
        !gc.contains(&identity_sid),
        "IdentityKey is a dict-key word, not a GcType struct"
    );
    assert_eq!(
        peel_transparent_inner(&llbc, &ty),
        Some(gc_object_ref.clone())
    );
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        gc_object_ref.clone()
    );

    // The wrapper itself carries `GcType`: keep it so the allocated
    // object is not interned as its payload pointer.
    let mut gc_wrapper_decls = pointer_handle_decls();
    gc_wrapper_decls.push((
        6,
        struct_decl(
            6,
            ident_path(&["fixture", "GcWrapper"]),
            raw_ptr_field(1),
            true,
        ),
    ));
    let (llbc, ty) = load_handle(
        gc_wrapper_decls,
        serde_json::json!([gctype_impl_for_adt(6)]),
        6,
    );
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let wrapper_sid = super::struct_id_for_type_decl(llbc.type_by_id(6).expect("GcWrapper decl"));
    assert!(
        gc.contains(&wrapper_sid),
        "GcType impl registers the wrapper"
    );
    assert_eq!(peel_transparent_inner(&llbc, &ty), None);
}

/// A by-value store of a `#[repr(transparent)]` GC handle into an
/// enum-variant payload is one `setfield` of the container field
/// (`heaptracker.all_fielddescrs` / `Transformer.zero_contents`: a Ptr
/// is not an inlined STRUCT). The FieldWrite name is `__pos_0`.
#[test]
fn transparent_handle_enum_payload_store_is_one_fieldwrite() {
    let mut decls = pointer_handle_decls();
    decls.push((6, result_enum(6)));
    let handle_adt = serde_json::json!({
        "Adt": {"id": 3, "generics": window_empty_generics()}
    });
    let object_ptr = serde_json::json!({
        "RawPtr": [
            {"Adt": {"id": 1, "generics": window_empty_generics()}},
            "Mut"
        ]
    });
    let result_ty = result_adt_node(6, object_ptr, handle_adt.clone());
    let result_ref = serde_json::json!({"Ref": ["Erased", result_ty.clone(), "Mut"]});
    let unit = serde_json::json!({"Adt": {"builtin": "Tuple", "generics": {"types": []}}});
    let field_place = window_place(
        serde_json::json!({"Projection": [
            window_place(
                serde_json::json!({"Projection": [
                    window_place(serde_json::json!({"Local": 1}), &result_ref),
                    "Deref"
                ]}),
                &result_ty
            ),
            {"Field": [1, 0]}
        ]}),
        &handle_adt,
    );
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "store_handle_into_enum_payload"], true),
        "signature": {
            "is_unsafe": true,
            "inputs": [result_ref.clone(), handle_adt.clone()],
            "output": unit.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 2, "locals": [
                window_local(0, &unit),
                window_local(1, &result_ref),
                window_local(2, &handle_adt)
            ]},
            "body": [{
                "statements": [{
                    "span": window_span(),
                    "kind": {"Assign": [
                        field_place,
                        {"Use": [{"Read": window_place(
                            serde_json::json!({"Local": 2}),
                            &handle_adt
                        )}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let trait_impls =
        with_fixture_gctype_impls(&decls, serde_json::json!([pyerror_deref_impl(23, 2)]));
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": indexed_rows(decls),
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": trait_impls
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let handle_ty = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [9003, adt_body(3)]
    }))
    .expect("handle TyRef parses");
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw_alloc = std::collections::HashSet::<String>::new();
    assert!(
        super::tyref_by_value_struct_decl(
            &handle_ty,
            &llbc,
            no_tombstoned_leaves(),
            &gc_struct_ids,
            &raw_alloc
        )
        .is_none(),
        "a transparent GC handle is a Ptr leaf, not an inlined STRUCT"
    );
    let graph = crate::test_support::lower_function(&llbc, "store_handle_into_enum_payload")
        .expect("lower store of a transparent handle into an enum payload");
    let writes: Vec<&str> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } => Some(field.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        writes,
        ["__pos_0"],
        "enum-variant handle payload is one FieldWrite of the container field; graph={}",
        graph.dump()
    );
}

/// `repr(transparent)` construction of a GC-pointer handle is the
/// identity on that word (`history.getkind` of a GC `Ptr`). There is no
/// second object and no `__pos_0` FieldWrite.
#[test]
fn transparent_handle_ctor_is_identity_of_the_payload_pointer() {
    let decls = pointer_handle_decls();
    let handle_adt = serde_json::json!({
        "Adt": {"id": 3, "generics": window_empty_generics()}
    });
    let object_ptr = serde_json::json!({
        "RawPtr": [
            {"Adt": {"id": 1, "generics": window_empty_generics()}},
            "Mut"
        ]
    });
    let carrier_ptr = serde_json::json!({
        "RawPtr": [
            {"Adt": {"id": 2, "generics": window_empty_generics()}},
            "Mut"
        ]
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "ctor_handle_from_object_cast"], true),
        "signature": {
            "is_unsafe": true,
            "inputs": [carrier_ptr.clone()],
            "output": handle_adt.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &handle_adt),
                window_local(1, &carrier_ptr),
                window_local(2, &object_ptr)
            ]},
            "body": [{
                "statements": [
                    {
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 2}), &object_ptr),
                            {"Cast": [
                                "PtrToPtr",
                                {"Move": window_place(
                                    serde_json::json!({"Local": 1}),
                                    &carrier_ptr
                                )},
                                object_ptr.clone()
                            ]}
                        ]}
                    },
                    {
                        "span": window_span(),
                        "kind": {"Assign": [
                            window_place(serde_json::json!({"Local": 0}), &handle_adt),
                            {"Aggregate": [
                                {"Adt": [
                                    {"id": 3, "generics": window_empty_generics()},
                                    0,
                                    null
                                ]},
                                [{"Move": window_place(
                                    serde_json::json!({"Local": 2}),
                                    &object_ptr
                                )}]
                            ]}
                        ]}
                    }
                ],
                "kind": "Regular",
                "terminator": {"span": window_span(), "kind": "Return"}
            }]
        }}
    });
    let trait_impls =
        with_fixture_gctype_impls(&decls, serde_json::json!([pyerror_deref_impl(23, 2)]));
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": indexed_rows(decls),
            "fun_decls": [caller],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": trait_impls
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let graph = crate::test_support::lower_function(&llbc, "ctor_handle_from_object_cast")
        .expect("lower transparent-handle Ctor of a PyObject-cast payload");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect();
    assert!(
        ops.iter()
            .all(|op| !matches!(op.kind, OpKind::FieldWrite { .. })),
        "transparent-handle construction is identity, not a FieldWrite; graph={}",
        graph.dump()
    );
    assert!(
        ops.iter().all(|op| !matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            }
        )),
        "transparent-handle construction is identity, not a synthetic ctor; graph={}",
        graph.dump()
    );
}

/// The same handle built through a constructor fn item
/// (`src: "AdtConstructor"` → `constructor_fn_item_shape` →
/// `emit_aggregate_value`) is the payload operand.
#[test]
fn transparent_handle_ctor_fn_item_is_identity_of_the_payload() {
    let decls = pointer_handle_decls();
    let handle_adt = serde_json::json!({
        "Adt": {"id": 3, "generics": window_empty_generics()}
    });
    let object_ptr = serde_json::json!({
        "RawPtr": [
            {"Adt": {"id": 1, "generics": window_empty_generics()}},
            "Mut"
        ]
    });
    let ctor_fn = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "PyError"], true),
        "src": "AdtConstructor",
        "signature": {
            "is_unsafe": true,
            "inputs": [object_ptr.clone()],
            "output": handle_adt.clone()
        },
        "body": "Opaque"
    });
    let trait_impls =
        with_fixture_gctype_impls(&decls, serde_json::json!([pyerror_deref_impl(23, 2)]));
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": indexed_rows(decls),
            "fun_decls": [ctor_fn],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": trait_impls
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let handle_ty = serde_json::from_value::<TyRef>(handle_adt).expect("handle TyRef parses");
    let fd = llbc
        .fn_by_id(0)
        .expect("AdtConstructor FunDecl is in the fixture");
    with_fixture_lowering(&llbc, |lowering| {
        let shape = lowering
            .constructor_fn_item_shape(fd, &handle_ty)
            .expect("constructor_fn_item_shape of the handle");
        assert_eq!(
            shape,
            crate::front::mir::AggregateShape::Operand {
                index: 0,
                transparent: true
            }
        );
        let mut graph = FunctionGraph::new("ctor_handle_fn_item");
        let block = graph.startblock;
        let payload = graph.alloc_value_var();
        let built =
            crate::front::mir::emit_aggregate_value(&mut graph, block, &shape, &[payload.clone()])
                .expect("emit constructor-fn-item aggregate");
        assert_eq!(built, payload, "transparent ctor is the payload word");
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.is_empty(),
            "identity construction emits no ops; graph={}",
            graph.dump()
        );
    });
}

/// A by-value Copy newtype around one integer word is that integer
/// even without `repr(transparent)` (`history.getkind` /
/// `call.py getcalldescr`). `VersionTag(u64)` is this shape.
#[test]
fn by_value_integer_newtype_types_as_the_inner_integer() {
    let mut decls = pointer_handle_decls();
    decls.push((
        6,
        struct_decl(6, ident_path(&["fixture", "Word"]), i64_field(), false),
    ));
    decls.push((
        7,
        struct_decl(7, ident_path(&["fixture", "Serial"]), u64_field(), false),
    ));
    decls.push((
        8,
        struct_decl(8, ident_path(&["fixture", "Pair"]), two_i64_fields(), false),
    ));
    let (llbc, word_ty) = load_handle(decls.clone(), serde_json::json!([null]), 6);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw_alloc = std::collections::HashSet::<String>::new();
    assert_eq!(super::tyref_to_value_type(&word_ty, &llbc), ValueType::Int);
    assert_eq!(
        super::tyref_to_field_layout_string(&word_ty, &llbc, &gc_struct_ids, &raw_alloc),
        "i64"
    );
    let (llbc, serial_ty) = load_handle(decls.clone(), serde_json::json!([null]), 7);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw_alloc = std::collections::HashSet::<String>::new();
    assert_eq!(
        super::tyref_to_value_type(&serial_ty, &llbc),
        ValueType::Unsigned
    );
    assert_eq!(
        super::tyref_to_field_layout_string(&serial_ty, &llbc, &gc_struct_ids, &raw_alloc),
        "u64"
    );
    assert!(
        super::tyref_by_value_struct_decl(
            &word_ty,
            &llbc,
            no_tombstoned_leaves(),
            &gc_struct_ids,
            &raw_alloc
        )
        .is_none(),
        "integer newtype is one word, not an inlined STRUCT"
    );
    assert!(
        super::tyref_by_value_struct_decl(
            &serial_ty,
            &llbc,
            no_tombstoned_leaves(),
            &gc_struct_ids,
            &raw_alloc
        )
        .is_none()
    );
    let (llbc, pair_ty) = load_handle(decls.clone(), serde_json::json!([null]), 8);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw_alloc = std::collections::HashSet::<String>::new();
    assert!(
        super::tyref_by_value_struct_decl(
            &pair_ty,
            &llbc,
            no_tombstoned_leaves(),
            &gc_struct_ids,
            &raw_alloc
        )
        .is_some(),
        "a two-field struct stays an inlined STRUCT"
    );
    assert_eq!(
        paint_handle(decls.clone(), serde_json::json!([null]), 8),
        ValueType::Ref(None)
    );

    decls.push((
        9,
        struct_decl(
            9,
            ident_path(&["core", "ops", "range", "RangeFrom"]),
            serde_json::json!([{
                "name": "start",
                "ty": {"Scalar": {"Integer": {"Unsigned": "U64"}}},
                "attr_info": null
            }]),
            false,
        ),
    ));
    let (llbc, range_ty) = load_handle(decls.clone(), serde_json::json!([null]), 9);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw_alloc = std::collections::HashSet::<String>::new();
    assert_eq!(
        super::tyref_to_value_type(&range_ty, &llbc),
        ValueType::Ref(None),
        "RangeFrom.start is a named field, not a newtype wrap"
    );
    assert!(
        super::tyref_by_value_struct_decl(
            &range_ty,
            &llbc,
            no_tombstoned_leaves(),
            &gc_struct_ids,
            &raw_alloc
        )
        .is_some()
    );

    let mut gc_int = pointer_handle_decls();
    gc_int[0] = (
        1,
        struct_decl(1, ident_path(&["fixture", "PyObject"]), i64_field(), false),
    );
    assert_eq!(
        paint_handle(gc_int, serde_json::json!([null]), 1),
        ValueType::Ref(None)
    );
}

/// `&self` of a one-integer tuple struct is an address. `self.0` is
/// that integer (`history.getkind` of the field), not the Ref receiver.
#[test]
fn integer_newtype_shared_ref_method_returns_the_inner_integer() {
    let span = fixture_span();
    let empty_generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let word_adt = serde_json::json!({
        "Adt": {"id": 0, "builtin": null, "generics": empty_generics}
    });
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let word_ref = serde_json::json!({"Ref": ["Erased", word_adt, "Shared"]});
    let word_decl = struct_decl(0, ident_path(&["fixture", "Word"]), i64_field(), false);
    let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
    let field_place = place(
        serde_json::json!({"Projection": [
            place(
                serde_json::json!({"Projection": [
                    place(serde_json::json!({"Local": 1}), &word_ref),
                    "Deref"
                ]}),
                &word_adt
            ),
            {"Field": [null, 0]}
        ]}),
        &i64_ty,
    );
    let as_raw = serde_json::json!({
        "def_id": 0,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "Word", "as_raw"])),
        "signature": {
            "is_unsafe": false,
            "inputs": [word_ref],
            "output": i64_ty
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    {"index": 0, "name": null, "span": span, "ty": i64_ty},
                    {"index": 1, "name": "self", "span": span, "ty": word_ref}
                ]
            },
            "body": [{
                "statements": [{
                    "span": span,
                    "kind": {"Assign": [
                        place(serde_json::json!({"Local": 0}), &i64_ty),
                        {"Use": [{"Read": field_place}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [word_decl],
            "fun_decls": [as_raw],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let word_ty = TyRef::Other(word_adt);
    let ref_ty = TyRef::Other(word_ref);
    assert_eq!(super::tyref_to_value_type(&word_ty, &llbc), ValueType::Int);
    assert!(
        matches!(
            super::tyref_to_value_type(&ref_ty, &llbc),
            ValueType::Ref(_)
        ),
        "&Word is an address, got {:?}",
        super::tyref_to_value_type(&ref_ty, &llbc)
    );
    let graph = super::lower_function(&llbc, "as_raw").expect("lower as_raw");
    let reads: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, ty: ValueType::Int, .. }
                    if field.name == "__pos_0" || field.name == "_0"
            )
        })
        .collect();
    assert_eq!(
        reads.len(),
        1,
        "(*self).0 of &Word must be a FieldRead of the integer, graph={}",
        graph.dump()
    );
}

fn opaque_handle_decl(def_id: u64, name: serde_json::Value, size: u64) -> serde_json::Value {
    let mut decl = serde_json::json!({
        "def_id": def_id,
        "item_meta": fixture_item_meta(name),
        "kind": "Opaque"
    });
    let mut layout = transparent_layout();
    layout[0]["value"]["size"] = serde_json::json!(size);
    layout[0]["value"]["align"] = serde_json::json!(size);
    decl["layout"] = layout;
    decl
}

fn load_handle_with_width(
    type_decls: Vec<(u64, serde_json::Value)>,
    trait_impls: serde_json::Value,
    query: u64,
    pointer_size: Option<u8>,
) -> (Llbc, TyRef) {
    let trait_impls = with_fixture_gctype_impls(&type_decls, trait_impls);
    let mut translated = serde_json::json!({
        "crate_name": "fixture",
        "type_decls": indexed_rows(type_decls),
        "fun_decls": [],
        "global_decls": [],
        "trait_decls": handle_trait_decls(),
        "trait_impls": trait_impls
    });
    if let Some(width) = pointer_size {
        translated["target_information"] = serde_json::json!([{
            "key": "fixture-target",
            "value": {
                "target_pointer_size": width,
                "is_little_endian": true
            }
        }]);
    }
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": translated
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let ty = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [9000 + query, adt_body(query)]
    }))
    .expect("query TyRef parses");
    (llbc, ty)
}

/// An external `repr(transparent)` handle is `Opaque`. The wrapper
/// keeps its own class; the payload row is a pointer to `Deref::Target`.
#[test]
fn opaque_transparent_gc_handle_keeps_the_wrapper_class() {
    let mut decls = pointer_handle_decls();
    decls[2] = (
        3,
        opaque_handle_decl(3, ident_path(&["fixture", "PyError"]), 8),
    );
    let (llbc, ty) = load_handle_with_width(
        decls.clone(),
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        ValueType::Ref(Some("PyError".into()))
    );
    assert_handle_class(&llbc, &ty, Some("PyError"));
    assert_eq!(
        super::tyref_to_field_layout_string(
            &ty,
            &llbc,
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "*mut PyErrorObject"
    );

    // The target struct is `Opaque` in a crate that never reads its
    // fields. The class is still the nominal leaf.
    let mut opaque_target = decls.clone();
    opaque_target[1] = (
        2,
        serde_json::json!({
            "def_id": 2,
            "item_meta": fixture_item_meta(ident_path(&["fixture", "PyErrorObject"])),
            "kind": "Opaque"
        }),
    );
    let (llbc, ty) = load_handle_with_width(
        opaque_target,
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        ValueType::Ref(Some("PyError".into()))
    );
    assert_handle_class(&llbc, &ty, Some("PyError"));

    // No extraction target: the width is unknown, so the opaque
    // view does not guess a class.
    let (llbc, ty) = load_handle_with_width(
        decls.clone(),
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        None,
    );
    assert_eq!(super::tyref_to_value_type(&ty, &llbc), ValueType::Ref(None));
    assert_handle_class(&llbc, &ty, Some("PyError"));

    // A narrower transparent word is not this handle.
    decls[2] = (
        3,
        opaque_handle_decl(3, ident_path(&["fixture", "Narrow"]), 4),
    );
    let (llbc, ty) = load_handle_with_width(
        decls.clone(),
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    assert_eq!(super::tyref_to_value_type(&ty, &llbc), ValueType::Ref(None));
    assert_handle_class(&llbc, &ty, Some("Narrow"));

    // A linked scalar keeps its bank even when a `Deref` impl is
    // present. The class rule runs before the scalar peel.
    decls[2] = (
        3,
        opaque_handle_decl(3, ident_path(&["fixture", "Word"]), 8),
    );
    let (llbc, ty) = load_handle_with_width(
        decls,
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    llbc.register_transparent_scalar_kinds([(
        "fixture::Word".to_string(),
        majit_charon_reader::TransparentScalarKind::Signed,
    )]);
    assert_eq!(super::tyref_to_value_type(&ty, &llbc), ValueType::Int);
    assert_handle_class(&llbc, &ty, Some("Word"));
}

/// An imported Opaque pointer handle is a GC `&mut T` cell. The
/// defining crate sees a transparent struct; the importer sees
/// `TypeDeclKind::Opaque` (`opaque_pointer_handle`).
#[test]
fn opaque_imported_handle_mut_ref_is_a_cell() {
    let mut decls = pointer_handle_decls();
    decls[2] = (
        3,
        opaque_handle_decl(3, ident_path(&["fixture", "PyError"]), 8),
    );
    let (llbc, ty) = load_handle_with_width(
        decls,
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let tomb = super::no_tombstoned_leaves();
    let raw = std::collections::HashSet::<String>::new();
    assert!(
        super::mut_ref_cell_root_of(&ty, &llbc, tomb, &gc_struct_ids, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "Opaque PyError is a MutRef payload"
    );
    let mut_ty: TyRef = serde_json::from_value(serde_json::json!({
        "Ref": ["Erased", {"Value": [9003, adt_body(3)]}, "Mut"]
    }))
    .unwrap();
    assert!(
        super::tyref_gc_mut_ref_cell_root(&mut_ty, &llbc, tomb, &gc_struct_ids, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "&mut Opaque PyError occupies a MutRef cell"
    );
}

/// `apply_published_structs` fills the Opaque import; the word is
/// still a MutRef cell (`published_struct_copy`).
#[test]
fn published_imported_handle_mut_ref_is_a_cell() {
    let defining_decls = pointer_handle_decls();
    let (defining, _) = load_handle_with_width(
        defining_decls,
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    let rows = crate::front::mir::harvest_published_structs(&defining);
    let mut dest_decls = pointer_handle_decls();
    dest_decls[2] = (
        3,
        opaque_handle_decl(3, ident_path(&["fixture", "PyError"]), 8),
    );
    let (mut llbc, ty) = load_handle_with_width(
        dest_decls,
        serde_json::json!([pyerror_deref_impl(23, 2)]),
        3,
        Some(8),
    );
    crate::front::mir::apply_published_structs(&mut llbc, &rows);
    assert!(
        llbc.type_by_id(3)
            .is_some_and(|td| td.published_struct_copy),
        "PyError import is a published struct copy"
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let tomb = super::no_tombstoned_leaves();
    let raw = std::collections::HashSet::<String>::new();
    assert!(
        super::mut_ref_cell_root_of(&ty, &llbc, tomb, &gc_struct_ids, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "published PyError is a MutRef payload"
    );
    let mut_ty: TyRef = serde_json::from_value(serde_json::json!({
        "Ref": ["Erased", {"Value": [9003, adt_body(3)]}, "Mut"]
    }))
    .unwrap();
    assert!(
        super::tyref_gc_mut_ref_cell_root(&mut_ty, &llbc, tomb, &gc_struct_ids, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "&mut published PyError occupies a MutRef cell"
    );
}

/// `celldict.py` `erase({})`: a by-value transparent newtype intern's
/// as the inner struct, so unique-graph `mergeinputargs` unifies the
/// wrapper with that operand (`rordereddict.rtype_len`).
#[test]
fn by_value_transparent_newtype_interns_as_inner_struct() {
    let inner = {
        let mut decl = struct_decl(1, ident_path(&["fixture", "RDict"]), i64_field(), false);
        decl["layout"] = serde_json::json!([{
            "key": "fixture-target",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"repr_algo": "Rust", "transparent": false}
            }
        }]);
        decl
    };
    let wrapper_field = serde_json::json!([{
        "name": null,
        "ty": {"Adt": {"id": 1, "generics": {"types": []}}},
        "attr_info": null
    }]);
    let wrapper = struct_decl(
        2,
        ident_path(&["fixture", "ModuleDictStorage"]),
        wrapper_field,
        true,
    );
    let alias = serde_json::json!({
        "def_id": 3,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "ModuleDictEntries"])),
        "kind": {"Alias": {
            "aliased_ty": {"Adt": {"id": 1, "generics": {"types": []}}}
        }}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": indexed_rows(vec![(1, inner), (2, wrapper), (3, alias)]),
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let tomb = no_tombstoned_leaves();
    let storage = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [2, {"Adt": {"id": 2, "generics": {"types": []}}}]
    }))
    .expect("storage TyRef parses");
    let entries = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [3, {"Adt": {"id": 3, "generics": {"types": []}}}]
    }))
    .expect("entries TyRef parses");
    let dict = serde_json::from_value::<TyRef>(serde_json::json!({
        "Value": [1, {"Adt": {"id": 1, "generics": {"types": []}}}]
    }))
    .expect("dict TyRef parses");
    assert_eq!(
        super::tyref_class_root_with(&storage, &llbc, tomb).as_deref(),
        Some("RDict")
    );
    assert_eq!(
        super::tyref_class_root_with(&entries, &llbc, tomb).as_deref(),
        Some("RDict")
    );
    assert_eq!(
        super::tyref_class_root_with(&dict, &llbc, tomb).as_deref(),
        Some("RDict")
    );
}

/// A transparent wrapper of `*mut PyObject` stays the wrapper.
/// `StrKey`'s `*mut Utf8Str` and a scalar newtype still erase to the word.
#[test]
fn borrowed_one_scalar_struct_uses_the_field_bank() {
    let type_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [{"Ident": ["fixture", 0]}, {"Ident": ["DepthWord", 0]}],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 4, "col": 1}
            }},
            "source_text": "struct Anchor { depth: usize, _not_send: () }",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "kind": {"Struct": [
            {
                "name": "depth",
                "ty": {"Value": [7, {"Scalar": {"Integer": {"Unsigned": "Usize"}}}]},
                "attr_info": null
            },
            {
                "name": "_not_send",
                "ty": {"Value": [9, {"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}}]},
                "attr_info": null
            }
        ]},
        "layout": [{
            "key": "aarch64-apple-darwin",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0, 8]}],
                "repr": {"repr_algo": "Rust", "transparent": true}
            }
        }]
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [null, type_decl],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let borrowed = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [8, {
            "Ref": ["r", {
                "Adt": {"id": 1, "generics": {"types": []}}
            }, "Shared"]
        }]
    }))
    .expect("borrowed TyRef parses");
    assert_eq!(
        super::tyref_to_value_type(&borrowed, &llbc),
        crate::model::ValueType::Unsigned
    );
}

#[test]
fn linked_transparent_definition_types_opaque_dependency_view() {
    let type_decl = |kind: serde_json::Value| {
        serde_json::json!({
            "def_id": 1,
            "item_meta": {
                "name": [{"Ident": ["shared", 0]}, {"Ident": ["Word", 0]}],
                "span": {"data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 16}
                }},
                "source_text": "struct Word(i64);",
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "kind": kind,
            "layout": [{
                "key": "x86_64-unknown-linux-gnu",
                "value": {
                    "size": 8,
                    "align": 8,
                    "variant_layouts": [{"field_offsets": [0]}],
                    "repr": {"repr_algo": "Rust", "transparent": true}
                }
            }]
        })
    };
    let artifact = |kind| {
        serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [null, type_decl(kind)],
                "fun_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        })
    };
    let defining = Llbc::from_slice(
        artifact(serde_json::json!({"Struct": [{
            "name": null,
            "ty": {"Value": [7, {"Scalar": {"Integer": {"Signed": "I64"}}}]},
            "attr_info": null
        }]}))
        .to_string()
        .as_bytes(),
    )
    .unwrap();
    let opaque =
        Llbc::from_slice(artifact(serde_json::json!("Opaque")).to_string().as_bytes()).unwrap();
    let llbcs = vec![defining, opaque];
    super::link_transparent_scalar_types(&llbcs);
    let jitdriver_receiver_roots =
        crate::codewriter::jtransform::default_jitdriver_receiver_roots();
    super::super::build_semantic_program_from_llbc_with_static_addrs_filtered(
        &llbcs[1],
        crate::HostStaticAddrs::default(),
        &jitdriver_receiver_roots,
        None,
        None,
        super::no_tombstoned_leaves(),
    )
    .unwrap();
    let linked_ty = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [8, {
            "Adt": {"id": 1, "generics": {"types": []}}
        }]
    }))
    .unwrap();
    assert_eq!(
        super::tyref_to_value_type(&linked_ty, &llbcs[1]),
        crate::model::ValueType::Int
    );
    assert_eq!(
        tyref_to_attr_value_type(&linked_ty, &llbcs[1]),
        crate::model::ValueType::Int
    );
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbcs[1]).gc_struct_ids();
    assert_eq!(
        super::tyref_to_field_layout_string(
            &linked_ty,
            &llbcs[1],
            &gc_struct_ids,
            &std::collections::HashSet::<String>::new()
        ),
        "i64"
    );
}

#[test]
fn trait_assoc_type_resolves_via_unique_impl() {
    // `C::Name` with `impl Trait#1 for X { type Name = bool }` as
    // the LLBC's only impl of trait 1 renders the bound type.
    let llbc = llbc_with_trait_impls(serde_json::json!([
        null,
        {
            "impl_trait": { "id": 1 },
            "types": [{
                "kind": { "TraitType": [1, 0] },
                "skip_binder": { "value": { "Scalar": "Bool" } }
            }]
        }
    ]));
    let projection = serde_json::json!({
        "TraitType": [
            { "trait_decl_ref": { "skip_binder": { "id": 1 } } },
            0,
            { "regions": [], "types": [], "const_generics": [], "trait_refs": [] }
        ]
    });
    assert_eq!(
        charon_type_value_to_ast_string(&projection, &llbc, 0),
        "bool"
    );
}

#[test]
fn trait_assoc_type_keeps_fallback_when_impl_ambiguous_or_missing() {
    // Two impls of trait 1 → instantiation-dependent → fallback.
    let two_impls = llbc_with_trait_impls(serde_json::json!([
        {
            "impl_trait": { "id": 1 },
            "types": [{
                "kind": { "TraitType": [1, 0] },
                "skip_binder": { "value": { "Scalar": "Bool" } }
            }]
        },
        {
            "impl_trait": { "id": 1 },
            "types": [{
                "kind": { "TraitType": [1, 0] },
                "skip_binder": { "value": { "Scalar": "Char" } }
            }]
        }
    ]));
    let projection = serde_json::json!({
        "TraitType": [
            { "trait_decl_ref": { "skip_binder": { "id": 1 } } },
            0,
            { "regions": [], "types": [], "const_generics": [], "trait_refs": [] }
        ]
    });
    assert_eq!(
        charon_type_value_to_ast_string(&projection, &two_impls, 0),
        "??TraitType"
    );
    // No impl at all → fallback too.
    let no_impls = llbc_with_trait_impls(serde_json::json!([]));
    assert_eq!(
        charon_type_value_to_ast_string(&projection, &no_impls, 0),
        "??TraitType"
    );
}

#[test]
fn raw_ptr_pointee_container_root_names_a_tuple_of_vecs() {
    let llbc = kwargs_unerase_cast_llbc(true);
    let (_, dest_ptr) = kwargs_unerase_ptr_tys();
    let root = super::raw_ptr_pointee_container_root(&dest_ptr, &llbc)
        .expect("tuple-of-vecs dest must be a container root");
    assert!(
        majit_ir::descr::is_shaped_tuple_name(&root),
        "dest spelling must be Tuple<...>, got {root}"
    );
    assert!(
        root.contains("Vec<") && root.contains("PyObject"),
        "tuple items must be Vec of PyObject pointers, got {root}"
    );
}

fn shared_borrow_of_usize() -> TyRef {
    fixture_ty(serde_json::json!({
        "Ref": [
            {"Erased": null},
            {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
            "Shared"
        ]
    }))
}

fn shared_borrow_of_u8() -> TyRef {
    fixture_ty(serde_json::json!({
        "Ref": [
            {"Erased": null},
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Shared"
        ]
    }))
}

fn llbc_with_ordinary_and_closure_env() -> Llbc {
    // def_id 0: ordinary struct named `closure` with src Normal — the
    // name leaf must not decide env-ness.
    // def_id 1: compiler-generated closure env named `Env` with src Closure.
    let file = serde_json::json!({
        "charon_version": "0.1.201", "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": type_decl_meta(&["fixture", "BorrowedByte", "closure"]),
                    "kind": {"Struct": [{
                        "name": "byte",
                        "ty": {"Ref": [{"Erased": null}, {"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Shared"]},
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
                    }]},
                    "src": "Normal"
                },
                {
                    "def_id": 1,
                    "item_meta": type_decl_meta(&["fixture", "try_dispatch_binary_special", "Env"]),
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": {"Ref": [{"Erased": null}, {"Scalar": {"Integer": {"Unsigned": "Usize"}}}, "Shared"]},
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}
                    }]},
                    "src": {"Closure": {"info": {"kind": "FnOnce"}}}
                }
            ],
            "fun_decls": [], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
}

#[test]
fn type_decl_is_closure_env_reads_src_not_the_name() {
    let llbc = llbc_with_ordinary_and_closure_env();
    let named_closure = llbc.type_by_id(0).expect("ordinary struct");
    let env = llbc.type_by_id(1).expect("closure env");
    assert!(
        !type_decl_is_closure_env(named_closure),
        "a struct whose leaf is `closure` but whose src is Normal is not an env"
    );
    assert!(
        type_decl_is_closure_env(env),
        "src Closure identifies the env even when the leaf is not `closure`"
    );
    assert!(!tyref_is_closure_env(&adt_ty(0), &llbc));
    assert!(tyref_is_closure_env(&adt_ty(1), &llbc));
}

#[test]
fn adt_field_read_peels_shared_borrow_of_usize_to_unsigned() {
    // `try_dispatch_binary_special::closure::call` captures `operands:
    // usize` by shared borrow.  `Rvalue::Ref` aliases that integer, so
    // the field read must be Unsigned; leaving it Ref assembles
    // `int_add/ri>i` at pc 10 of jitcode `call`.
    let llbc = llbc_with_ordinary_and_closure_env();
    let ref_usize = shared_borrow_of_usize();
    let ref_u8 = shared_borrow_of_u8();
    assert_eq!(
        tyref_to_value_type(&ref_usize, &llbc),
        ValueType::Ref(None),
        "the global projection stays non-peeling for &usize"
    );
    assert_eq!(
        adt_field_read_value_type(
            &ref_u8,
            &ref_u8,
            false,
            false,
            None,
            &llbc,
            no_tombstoned_leaves(),
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Ref(None),
        "an ordinary struct's &u8 field is a stored pointer"
    );
    assert_eq!(
        adt_field_read_value_type(
            &ref_usize,
            &ref_usize,
            false,
            true,
            None,
            &llbc,
            no_tombstoned_leaves(),
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Unsigned,
        "a closure-env &usize capture is the usize"
    );
    let ordinary = llbc.type_by_id(0).expect("ordinary struct");
    let env = llbc.type_by_id(1).expect("closure env");
    assert_eq!(
        tyref_to_attr_value_type(&ref_u8, &llbc),
        ValueType::Ref(None),
        "the generic attr projection does not peel a stored &u8"
    );
    assert_eq!(
        tyref_to_attr_value_type_for_struct_field(
            &ref_u8,
            ordinary,
            &llbc,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Ref(None),
        "FORCE-attr rows for an ordinary struct stay Ref"
    );
    assert_eq!(
        tyref_to_attr_value_type_for_struct_field(
            &ref_usize,
            env,
            &llbc,
            &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Unsigned,
        "FORCE-attr rows for a closure-env capture match the Int-banked getfield"
    );
}

/// `#[repr(transparent)] struct Anchor { depth: usize, _not_send: PhantomData }`
/// plus a rustc closure env that captures `&Anchor`. Mirrors
/// `eval::FrameAnchor` / `_trace::{closure#2}.__pos_5`.
fn llbc_with_transparent_anchor_closure_env() -> (Llbc, TyRef, TyRef) {
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let phantom_adt = serde_json::json!({"Adt": {"id": 0, "generics": generics}});
    let anchor_adt = serde_json::json!({"Adt": {"id": 1, "generics": generics}});
    let borrowed_anchor =
        serde_json::json!({"Ref": [{"Erased": null}, anchor_adt.clone(), "Shared"]});
    let file = serde_json::json!({
        "charon_version": "0.1.201", "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": type_decl_meta(&["core", "marker", "PhantomData"]),
                    "kind": {"Struct": []},
                    "layout": [{
                        "key": "x86_64-unknown-linux-gnu",
                        "value": {
                            "size": 0, "align": 1,
                            "variant_layouts": [{"field_offsets": []}],
                            "repr": {"repr_algo": "Rust", "transparent": false}
                        }
                    }],
                    "src": "Normal"
                },
                {
                    "def_id": 1,
                    "item_meta": type_decl_meta(&["eval", "FrameAnchor"]),
                    "kind": {"Struct": [
                        {
                            "name": "depth",
                            "ty": usize_ty,
                            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
                        },
                        {
                            "name": "_not_send",
                            "ty": phantom_adt,
                            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}
                        }
                    ]},
                    "layout": [{
                        "key": "x86_64-unknown-linux-gnu",
                        "value": {
                            "size": 8, "align": 8,
                            "variant_layouts": [{"field_offsets": [0, 8]}],
                            "repr": {"repr_algo": "Rust", "transparent": true}
                        }
                    }],
                    "src": "Normal"
                },
                {
                    "def_id": 2,
                    "item_meta": type_decl_meta(&["fixture", "Holder"]),
                    "kind": {"Struct": [{
                        "name": "anchor",
                        "ty": borrowed_anchor.clone(),
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
                    }]},
                    "src": "Normal"
                },
                {
                    "def_id": 3,
                    "item_meta": type_decl_meta(&["fixture", "_trace", "closure"]),
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": borrowed_anchor.clone(),
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}
                    }]},
                    "layout": [{
                        "key": "x86_64-unknown-linux-gnu",
                        "value": {
                            "size": 8, "align": 8,
                            "variant_layouts": [{"field_offsets": [0]}],
                            "repr": {"repr_algo": "Rust", "transparent": false}
                        }
                    }],
                    "src": {"Closure": {"info": {"kind": "Fn"}}}
                }
            ],
            "fun_decls": [], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let borrowed = serde_json::from_value::<TyRef>(borrowed_anchor).expect("borrowed TyRef");
    let by_value = adt_ty(1);
    (llbc, borrowed, by_value)
}

/// A closure env that captures `&TransparentWrapper` of a scalar must
/// paint that field as the inner word, matching `Rvalue::Ref` aliasing
/// of the local. Layout `&eval::FrameAnchor` would project
/// `Ptr(FrameAnchor)` against the stored Integer
/// (`annotator/model.py` `unionof`).
#[test]
fn closure_env_capture_of_borrowed_transparent_scalar_is_the_inner_word() {
    let (llbc, borrowed, by_value) = llbc_with_transparent_anchor_closure_env();
    let gc_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw = std::collections::HashSet::<String>::new();
    let tomb = no_tombstoned_leaves();
    assert_eq!(
        tyref_to_value_type(&by_value, &llbc),
        ValueType::Unsigned,
        "by-value transparent wrapper is its depth word"
    );
    assert_eq!(
        tyref_to_value_type(&borrowed, &llbc),
        ValueType::Unsigned,
        "a borrow of the wrapper is the same word"
    );
    assert_eq!(
        super::tyref_to_field_layout_string(&by_value, &llbc, &gc_ids, &raw),
        "usize",
        "by-value layout is the inner scalar"
    );
    assert_eq!(
        super::tyref_to_field_layout_string(&borrowed, &llbc, &gc_ids, &raw),
        "usize",
        "a captured &Wrapper layout is the inner scalar, not &eval::FrameAnchor"
    );
    let ordinary = llbc.type_by_id(2).expect("ordinary holder");
    let env = llbc.type_by_id(3).expect("closure env");
    assert!(type_decl_is_closure_env(env));
    assert!(!type_decl_is_closure_env(ordinary));
    assert_eq!(
        tyref_to_attr_value_type_for_struct_field(&borrowed, env, &llbc, &gc_ids, &raw),
        ValueType::Unsigned,
        "closure-env FORCE of a captured &Wrapper is the inner word"
    );
    assert_eq!(
        adt_field_read_value_type(
            &borrowed, &borrowed, false, true, None, &llbc, tomb, &gc_ids, &raw
        ),
        ValueType::Unsigned,
        "closure-env getfield of a captured &Wrapper is the inner word"
    );
}

fn raw_mut_u8() -> TyRef {
    fixture_ty(serde_json::json!({
        "RawPtr": [
            {"Scalar": {"Integer": {"Unsigned": "U8"}}},
            "Mut"
        ]
    }))
}

fn raw_mut_i8() -> TyRef {
    fixture_ty(serde_json::json!({
        "RawPtr": [
            {"Scalar": {"Integer": {"Signed": "I8"}}},
            "Mut"
        ]
    }))
}

#[test]
fn adt_field_read_raw_byte_ptr_bank() {
    // `rffi.py` `CCHARP`: a raw `*mut u8` field is CCHARP (Unsigned bank) in
    // any owner. `history.py` `getkind` of a raw `Ptr` is `'int'`: a raw
    // owner's `*mut i8` is the address word; a GC owner's stays the erased
    // GCREF.
    let llbc = llbc_with_ordinary_and_closure_env();
    let gc_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let tomb = no_tombstoned_leaves();
    let raw_u8 = raw_mut_u8();
    let raw_i8 = raw_mut_i8();
    assert_eq!(
        adt_field_read_value_type(
            &raw_u8,
            &raw_u8,
            false,
            false,
            Some(true),
            &llbc,
            tomb,
            &gc_ids,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Unsigned,
        "a GC owner's *mut u8 is CCHARP"
    );
    assert_eq!(
        adt_field_read_value_type(
            &raw_u8,
            &raw_u8,
            false,
            false,
            None,
            &llbc,
            tomb,
            &gc_ids,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Unsigned,
        "an unknown owner's *mut u8 is CCHARP"
    );
    assert_eq!(
        adt_field_read_value_type(
            &raw_u8,
            &raw_u8,
            false,
            false,
            Some(false),
            &llbc,
            tomb,
            &gc_ids,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Unsigned,
        "a raw owner's *mut u8 is CCHARP"
    );
    assert_eq!(
        adt_field_read_value_type(
            &raw_i8,
            &raw_i8,
            false,
            false,
            Some(false),
            &llbc,
            tomb,
            &gc_ids,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Int,
        "a raw owner's *mut i8 is the address word"
    );
    assert_eq!(
        adt_field_read_value_type(
            &raw_i8,
            &raw_i8,
            false,
            false,
            Some(true),
            &llbc,
            tomb,
            &gc_ids,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Ref(None),
        "a GC owner's *mut i8 stays the erased GCREF"
    );
}

#[test]
fn raw_field_address_value_type_matches_field_read() {
    let llbc = llbc_with_ordinary_and_closure_env();
    let raw_u8 = raw_mut_u8();
    assert_eq!(
        raw_field_address_value_type(&raw_u8, &raw_u8, Some(false), &llbc),
        Some(ValueType::Unsigned),
        "a raw owner's *mut u8 is CCHARP in any owner"
    );
    assert_eq!(
        raw_field_address_value_type(&raw_u8, &raw_u8, Some(true), &llbc),
        Some(ValueType::Unsigned),
        "a GC owner's *mut u8 is CCHARP in any owner"
    );
    assert_eq!(
        raw_field_address_value_type(&raw_mut_i8(), &raw_mut_i8(), Some(true), &llbc),
        None,
        "a GC owner's *mut i8 is not the raw-address bank"
    );
}

/// `*mut u8 as usize` is the type-erased address (`tyref_is_raw_byte_ptr`);
/// it is not a Vec header and must not take the Int-banked pointer arm's
/// predicate. The Ref → Unsigned sequence is unchanged.
#[test]
fn mut_u8_as_usize_emits_cast_ptr_to_int_then_r_uint() {
    let src = mut_u8_ty_json();
    let llbc = as_usize_cast_fixture("u8_as_usize", src, vec![], None);
    let graph =
        crate::test_support::lower_function(&llbc, "u8_as_usize").expect("lower *mut u8 as usize");
    assert_cast_ptr_to_int_then_r_uint(&graph, "*mut u8 as usize");
    assert!(!tyref_is_ptr_to_int_cast_origin(
        &TyRef::Other(mut_u8_ty_json()),
        &llbc
    ));
}

/// The pointer-to-int origin is a function pointer or a pointer /
/// reference to `alloc::vec::Vec`. A raw `*mut u8` buffer, a `*mut usize`
/// address, and a `Vec` by value are not that shape.
#[test]
fn ptr_to_int_cast_origin_is_fn_ptr_or_ptr_to_vec() {
    let llbc = {
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [vec_decl_json()],
                "fun_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
    };
    let vec_ty = vec_usize_ty_json();
    let fn_ptr = TyRef::Other(serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [],
                "output": {
                    "Adt": {
                        "id": 0,
                        "builtin": "Tuple",
                        "generics": {"types": []}
                    }
                },
                "is_unsafe": false
            }
        }
    }));
    assert!(tyref_is_ptr_to_int_cast_origin(&fn_ptr, &llbc));
    assert!(tyref_ptr_pointee_is_vec(
        &TyRef::Other(serde_json::json!({"Ref": ["_", vec_ty.clone(), "Mut"]})),
        &llbc
    ));
    assert!(tyref_ptr_pointee_is_vec(
        &TyRef::Other(serde_json::json!({"RawPtr": [vec_ty.clone(), "Mut"]})),
        &llbc
    ));
    assert!(tyref_is_ptr_to_int_cast_origin(
        &TyRef::Other(serde_json::json!({"RawPtr": [vec_ty.clone(), "Mut"]})),
        &llbc
    ));
    assert!(!tyref_ptr_pointee_is_vec(
        &TyRef::Other(vec_ty.clone()),
        &llbc
    ));
    assert!(!tyref_is_ptr_to_int_cast_origin(
        &TyRef::Other(vec_ty),
        &llbc
    ));
    assert!(!tyref_is_ptr_to_int_cast_origin(
        &TyRef::Other(mut_u8_ty_json()),
        &llbc
    ));
    assert!(!tyref_is_ptr_to_int_cast_origin(
        &TyRef::Other(serde_json::json!({
            "RawPtr": [{"Scalar": {"Integer": {"Unsigned": "Usize"}}}, "Mut"]
        })),
        &llbc
    ));
    assert!(!tyref_is_ptr_to_int_cast_origin(
        &TyRef::Other(usize_ty()),
        &llbc
    ));
}

/// Only a byte pointee spells the type-erased address; a pointee-typed
/// raw pointer and a wider primitive array head must stay narrows /
/// aliases, since erasing those would drop a pointee class the
/// annotator still needs.
#[test]
fn raw_byte_ptr_recognizes_only_the_byte_pointee() {
    let llbc = llbc_with_trait_impls(serde_json::json!([]));
    let ptr_to = |pointee: serde_json::Value| {
        TyRef::Other(serde_json::json!({ "RawPtr": [pointee, "Mut"] }))
    };
    for pointee in ["U8", "I8"] {
        assert!(
            tyref_is_raw_byte_ptr(
                &ptr_to(serde_json::json!({"Scalar": {"Integer": {"Unsigned": pointee}}})),
                &llbc
            ),
            "*mut {pointee} is the erased address pointer"
        );
    }
    assert!(!tyref_is_raw_byte_ptr(
        &ptr_to(serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})),
        &llbc
    ));
    assert!(!tyref_is_raw_byte_ptr(
        &ptr_to(serde_json::json!({"Adt": {"id": 0, "builtin": "Str"}})),
        &llbc
    ));
    // A bare `u8` (no pointer around it) is not an address either.
    assert!(!tyref_is_raw_byte_ptr(
        &TyRef::Other(serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})),
        &llbc
    ));
}

#[test]
fn string_storage_pointer_seeds_the_rpython_string_field_repr() {
    let wtf8buf = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["rustpython_wtf8", 0]},
                {"Ident": ["Wtf8Buf", 0]}
            ],
            "span": {"data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 16}
            }},
            "source_text": "pub struct Wtf8Buf { ... }",
            "attr_info": {
                "attributes": [], "inline": null, "rename": null, "public": true
            },
            "is_local": false
        },
        "kind": "Opaque",
        "layout": null
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [null, wtf8buf],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [null, {
                "def_id": 1,
                "item_meta": {
                    "name": [
                        {"Ident": ["core", 0]},
                        {"Ident": ["convert", 0]},
                        {"Ident": ["Into", 0]}
                    ],
                    "span": {"data": {
                        "file_id": 0,
                        "beg": {"line": 1, "col": 0},
                        "end": {"line": 1, "col": 16}
                    }},
                    "source_text": "pub trait Into<T> { ... }",
                    "attr_info": {
                        "attributes": [], "inline": null, "rename": null, "public": true
                    },
                    "is_local": false
                }
            }],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let pointee = serde_json::json!({
        "Adt": {"id": 1, "generics": {"types": []}}
    });
    let storage_ptr = TyRef::Other(serde_json::json!({
        "RawPtr": [pointee, "Mut"]
    }));

    assert_eq!(
        tyref_to_attr_value_type(&storage_ptr, &llbc),
        ValueType::Str,
        "W_UnicodeObject.value is the Rust storage spelling of PyPy's string-valued _utf8 field"
    );
    assert_eq!(
        super::tyref_to_value_type(&storage_ptr, &llbc),
        ValueType::Str,
        "the owning storage pointer must remain the same RPython string at return and phi sites"
    );
    assert_eq!(
        super::tyref_to_value_type(&TyRef::Other(pointee.clone()), &llbc),
        ValueType::Str,
        "a bare Wtf8Buf value is the same RPython string representation"
    );

    let generic_message = TyRef::Other(serde_json::json!({
        "TypeVar": {"Bound": [0, 0]}
    }));
    let generics = majit_charon_reader::FunGenerics {
        has_types: false,
        has_const_generics: false,
        trait_clauses: vec![majit_charon_reader::TraitClauseBound {
            trait_id: Some(1),
            types: vec![serde_json::json!({"TypeVar": {"Bound": [1, 0]}}), pointee],
        }],
    };
    assert_eq!(
        super::tyref_generic_trait_bound_root(&generic_message, &llbc, Some(&generics),).as_deref(),
        Some("String"),
        "impl Into<Wtf8Buf> message parameters seed the one RPython string input cell"
    );
}

#[test]
fn entry_instantiation_splits_key_class() {
    let llbc = llbc_with_types(
        "pyre_object",
        vec![code_struct(
            0,
            &["pyre_object", "rordereddict_entries", "Entry"],
        )],
        vec![],
    );
    let u64_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let node = |args: Vec<serde_json::Value>| {
        serde_json::json!({
            "Adt": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": args,
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    };
    let tomb = std::collections::HashSet::new();
    let a = super::adt_node_class_root_with(&node(vec![u64_ty, i64_ty.clone()]), &llbc, &tomb);
    let b = super::adt_node_class_root_with(&node(vec![i64_ty.clone(), i64_ty]), &llbc, &tomb);
    assert_ne!(a, b, "Entry<K,V> instantiations must not share one class");
    assert!(a.as_deref().unwrap_or("").contains('<'), "got {a:?}");
}

#[test]
fn tyref_enum_instantiation_suffix_unwraps_value_wrapper() {
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let wrapped = TyRef::Other(serde_json::json!({
        "Value": [99, option_adt_node(0, option_int_arg(true))]
    }));
    assert_eq!(
        super::tyref_enum_instantiation_suffix(&wrapped, &llbc, super::no_tombstoned_leaves()),
        "<i64>"
    );
    let bare_id = TyRef::Other(serde_json::json!({"Adt": {"id": 0}}));
    assert_eq!(
        super::tyref_enum_instantiation_suffix(&bare_id, &llbc, super::no_tombstoned_leaves()),
        "",
        "a TypeDeclRef with no generics and no instantiation() has no suffix"
    );
}

#[test]
fn option_integer_instantiations_paint_distinct_class_roots() {
    // A returned `Option<usize>` annotates as that class, not the
    // template `Option` (`bookkeeper.py` `getuniqueclassdef(cls)`).
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let dest = |signed: bool| {
        TyRef::Other(serde_json::json!({
            "Value": [7, option_adt_node(0, option_int_arg(signed))]
        }))
    };
    let usize_ty = super::tyref_to_value_type(&dest(false), &llbc);
    let i64_ty = super::tyref_to_value_type(&dest(true), &llbc);
    assert_eq!(
        usize_ty,
        crate::model::ValueType::Ref(Some("option::Option<usize>".into()))
    );
    assert_eq!(
        i64_ty,
        crate::model::ValueType::Ref(Some("option::Option<i64>".into()))
    );
    assert_ne!(usize_ty, i64_ty);
}

#[test]
fn zero_variant_enum_publishes_empty_discriminant_map() {
    let empty = |def_id: u64, path: &[&str]| {
        let span = serde_json::json!({"data": {
            "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
        }});
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": path.iter().map(|seg| serde_json::json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
                "span": span,
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": false
            },
            "kind": {"Enum": []},
            "layout": []
        })
    };
    let llbc = llbc_with_types(
        "fixture",
        vec![
            empty(0, &["core", "convert", "Infallible"]),
            empty(1, &["mymod", "Void"]),
        ],
        vec![],
    );
    let (_, _, _, enums, _, _, _, _) = super::derive_program_metadata(
        &llbc,
        &super::harvest_declared_gc_facts(&llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &std::collections::HashSet::new(),
        &[],
    );
    for key in [
        "core::convert::Infallible",
        "convert::Infallible",
        "Infallible",
        "mymod::Void",
        "Void",
    ] {
        assert!(
            enums.get(key).is_some_and(|m| m.is_empty()),
            "{key} must be a harvested zero-variant enum, got {:?}",
            enums.get(key)
        );
    }
}

fn raw_mut_u8_arg() -> serde_json::Value {
    serde_json::json!({"RawPtr": [
        {"Scalar": {"Integer": {"Unsigned": "U8"}}},
        "Mut"
    ]})
}

#[test]
fn result_integer_instantiations_paint_distinct_class_roots() {
    // `Result::ok` / `unwrap_or_default` match on the interned tagged
    // struct (`InstanceRepr.rtype_getattr` `__discriminant`, rclass.py).
    // Each instantiation is its own class (`bookkeeper.py`
    // `getuniqueclassdef(cls)`), so `__pos_0` does not union i32 with
    // Layout / String. A Result parameter seeds the same class
    // (`tyref_ref_param_class_root`).
    let llbc = llbc_with_types("fixture", vec![result_enum(0)], vec![]);
    let dest = |ok: serde_json::Value, err: serde_json::Value| {
        TyRef::Other(serde_json::json!({
            "Value": [7, result_adt_node(0, ok, err)]
        }))
    };
    let i32_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I32"}}});
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    let i32_dest = dest(i32_ty, u8_ty.clone());
    let usize_dest = dest(usize_ty, u8_ty);
    let i32_u8 = super::tyref_to_value_type(&i32_dest, &llbc);
    let usize_u8 = super::tyref_to_value_type(&usize_dest, &llbc);
    assert_eq!(
        i32_u8,
        crate::model::ValueType::Ref(Some("result::Result<i32,u8>".into()))
    );
    assert_eq!(
        usize_u8,
        crate::model::ValueType::Ref(Some("result::Result<usize,u8>".into()))
    );
    assert_ne!(i32_u8, usize_u8);
    assert_eq!(
        super::tyref_ref_param_class_root(&i32_dest, &llbc, super::no_tombstoned_leaves(), None),
        Some("result::Result<i32,u8>".into())
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&usize_dest, &llbc, super::no_tombstoned_leaves(), None),
        Some("result::Result<usize,u8>".into())
    );
}

/// `Option::map_or` stays a regular call to the extracted core body
/// `match self { Some(t) => f(t), None => default }`. The Option repr
/// is unchanged; only the caller rewrite is gone (`guess_call_kind`
/// regular).
#[test]
fn option_map_or_keeps_regular_call_to_core_body() {
    use crate::model::{CallTarget, ExitSwitch, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let Some(llbc) = crate::test_support::load_ullbc_if_present(path) else {
        return;
    };
    let caller = crate::test_support::lower_function(llbc, "get_or_create_extra_locals")
        .expect("lower get_or_create_extra_locals");
    let map_or_calls = caller
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "map_or",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some("map_or")
                    && segments.iter().any(|s| s == "option")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        map_or_calls, 1,
        "caller keeps one regular call to the core map_or body, got {map_or_calls}"
    );

    crate::test_support::with_lowering(llbc, |ctx| {
        let mut lowered_core_bodies = 0usize;
        for fd in llbc.iter_fun_decls() {
            if !fd.has_unstructured_body() {
                continue;
            }
            let name_path = fd.item_meta.name_path();
            if !name_path.ends_with("::option::<Impl>::map_or") {
                continue;
            }
            let graph =
                super::lower_fun_decl(ctx, fd).unwrap_or_else(|e| panic!("lower {name_path}: {e}"));
            let has_disc = graph.blocks.iter().any(|block| {
                matches!(block.exitswitch, Some(ExitSwitch::Value(_)))
                    || block.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::FieldRead { field, .. } if field.name == "__discriminant"
                        )
                    })
                    || block.operations.iter().any(
                        |op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_" || op == "ne"),
                    )
            });
            assert!(has_disc, "{name_path}: core body matches on the Option tag");
            let has_ft = graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        ..
                    } => name == "call_once" || name == "call" || name == "call_mut",
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } => {
                        let leaf = segments.last().map(String::as_str);
                        leaf == Some("call_once")
                            || leaf == Some("call")
                            || leaf == Some("call_mut")
                    }
                    _ => false,
                })
            });
            assert!(
                has_ft,
                "{name_path}: core body calls f(t); ops={:?}",
                graph_ops(&graph)
            );
            lowered_core_bodies += 1;
        }
        assert_eq!(
            lowered_core_bodies, 1,
            "extracted map_or FunDecl bodies must lower, got {lowered_core_bodies}"
        );
    });
}

/// `Option::{map, and_then, unwrap_or_else, or_else, is_some_and}` stay
/// regular calls to the extracted core bodies. The Option repr is
/// unchanged; only the caller rewrite is gone (`guess_call_kind`
/// regular). Each body matches on the receiver tag and calls `f(x)` /
/// `f()`.
#[test]
fn option_map_family_keeps_regular_call_to_core_body() {
    use crate::model::{CallTarget, ExitSwitch, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let Some(llbc) = crate::test_support::load_ullbc_if_present(path) else {
        return;
    };
    crate::test_support::with_lowering(llbc, |ctx| {
        let leaves = [
            "map",
            "and_then",
            "unwrap_or_else",
            "or_else",
            "is_some_and",
        ];
        for leaf in leaves {
            let suffix = format!("::option::<Impl>::{leaf}");
            let mut lowered_core_bodies = 0usize;
            for fd in llbc.iter_fun_decls() {
                if !fd.has_unstructured_body() {
                    continue;
                }
                let name_path = fd.item_meta.name_path();
                if !name_path.ends_with(&suffix) {
                    continue;
                }
                let graph = super::lower_fun_decl(ctx, fd)
                    .unwrap_or_else(|e| panic!("lower {name_path}: {e}"));
                let has_disc = graph.blocks.iter().any(|block| {
                    matches!(block.exitswitch, Some(ExitSwitch::Value(_)))
                        || block.operations.iter().any(|op| {
                            matches!(
                                &op.kind,
                                OpKind::FieldRead { field, .. } if field.name == "__discriminant"
                            )
                        })
                        || block.operations.iter().any(
                            |op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_" || op == "ne"),
                        )
                });
                assert!(has_disc, "{name_path}: core body matches on the Option tag");
                let has_ft = graph.blocks.iter().any(|block| {
                    block.operations.iter().any(|op| match &op.kind {
                        OpKind::Call {
                            target: CallTarget::Method { name, .. },
                            ..
                        } => name == "call_once" || name == "call" || name == "call_mut",
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } => {
                            let leaf = segments.last().map(String::as_str);
                            leaf == Some("call_once")
                                || leaf == Some("call")
                                || leaf == Some("call_mut")
                        }
                        _ => false,
                    })
                });
                assert!(
                    has_ft,
                    "{name_path}: core body calls f(x)/f(); ops={:?}",
                    graph_ops(&graph)
                );
                lowered_core_bodies += 1;
            }
            assert_eq!(
                lowered_core_bodies, 1,
                "extracted {leaf} FunDecl bodies must lower, got {lowered_core_bodies}"
            );
        }
    });

    let caller = crate::test_support::lower_function(llbc, "install_applevel_builtins")
        .expect("lower install_applevel_builtins");
    let is_some_and_calls = caller
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "is_some_and",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some("is_some_and")
                    && segments.iter().any(|s| s == "option")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        is_some_and_calls, 2,
        "caller keeps two regular calls to the core is_some_and body, got {is_some_and_calls}"
    );
}

fn option_core_regular_call_count(graph: &crate::model::FunctionGraph, leaf: &str) -> usize {
    use crate::model::{CallTarget, OpKind};
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == leaf,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some(leaf)
                    && segments.iter().any(|s| s == "option")
            }
            _ => false,
        })
        .count()
}

fn option_core_body_matches_tag(graph: &crate::model::FunctionGraph) -> bool {
    use crate::model::{ExitSwitch, OpKind};
    graph.blocks.iter().any(|block| {
        matches!(block.exitswitch, Some(ExitSwitch::Value(_)))
            || block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. } if field.name == "__discriminant"
                )
            })
            || block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "is_" || op == "ne"))
    })
}

fn option_core_body_calls_fnonce(graph: &crate::model::FunctionGraph) -> bool {
    use crate::model::{CallTarget, OpKind};
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "call_once" || name == "call" || name == "call_mut",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                let leaf = segments.last().map(String::as_str);
                leaf == Some("call_once") || leaf == Some("call") || leaf == Some("call_mut")
            }
            _ => false,
        })
    })
}

/// `Option::map_or_else` / `Option::flatten` stay regular calls of the
/// extracted core bodies (`CHARON_CORE_INCLUDE`), same as `map_or`.
///
/// `lower_function` has no `SpecQueue`, so the caller may still spell
/// `CallTarget::Method`; production `enqueue_spec` names the extracted
/// body once the FunDecl is Unstructured and the call's `trait_refs`
/// are concrete (`TraitImpl`, not `Clause`).
#[test]
fn option_map_or_else_and_flatten_keep_regular_call_to_core_body() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let Some(llbc) = crate::test_support::load_ullbc_if_present(path) else {
        return;
    };

    let map_or_else_caller = crate::test_support::lower_function(llbc, "concat_operand_name")
        .expect("lower concat_operand_name");
    let map_or_else_calls = option_core_regular_call_count(&map_or_else_caller, "map_or_else");
    assert_eq!(
        map_or_else_calls, 1,
        "concat_operand_name keeps one regular call to the core map_or_else body, got {map_or_else_calls}"
    );

    crate::test_support::with_lowering(llbc, |ctx| {
        let mut lowered_map_or_else = 0usize;
        let mut lowered_flatten = 0usize;
        for fd in llbc.iter_fun_decls() {
            if !fd.has_unstructured_body() {
                continue;
            }
            let name_path = fd.item_meta.name_path();
            let is_map_or_else = name_path.ends_with("::option::<Impl>::map_or_else");
            let is_flatten = name_path.ends_with("::option::<Impl>::flatten");
            if !is_map_or_else && !is_flatten {
                continue;
            }
            let graph =
                super::lower_fun_decl(ctx, fd).unwrap_or_else(|e| panic!("lower {name_path}: {e}"));
            assert!(
                option_core_body_matches_tag(&graph),
                "{name_path}: core body matches on the Option tag"
            );
            if is_map_or_else {
                assert!(
                    option_core_body_calls_fnonce(&graph),
                    "{name_path}: core body calls d()/f(t); ops={:?}",
                    graph_ops(&graph)
                );
                lowered_map_or_else += 1;
            } else {
                lowered_flatten += 1;
            }
        }
        assert_eq!(
            lowered_map_or_else, 1,
            "extracted map_or_else FunDecl bodies must lower, got {lowered_map_or_else}"
        );
        assert_eq!(
            lowered_flatten, 1,
            "extracted flatten FunDecl bodies must lower, got {lowered_flatten}"
        );

        let mut flatten_caller_calls = 0usize;
        for fd in llbc.iter_fun_decls() {
            if !fd.has_unstructured_body() {
                continue;
            }
            if !fd
                .item_meta
                .name_path()
                .ends_with("::type_fully_qualified_name")
            {
                continue;
            }
            if let Ok(graph) = super::lower_fun_decl(ctx, fd) {
                flatten_caller_calls += option_core_regular_call_count(&graph, "flatten");
            }
        }
        assert!(
            flatten_caller_calls >= 1,
            "type_fully_qualified_name keeps a regular call to the core flatten body, got {flatten_caller_calls}"
        );
    });
}

fn raw_store_struct(def_id: u64) -> serde_json::Value {
    let mut decl = struct_decl(
        def_id,
        ident_path(&["fixture", "RawStore"]),
        serde_json::json!([
            {
                "name": "ptr",
                "ty": {"RawPtr": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Mut"]},
                "attr_info": null
            },
            {
                "name": "len",
                "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}},
                "attr_info": null
            }
        ]),
        false,
    );
    decl["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 16,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0, 8]}],
            "repr": {"transparent": false}
        }
    }]);
    decl
}

fn raw_storage_enum(def_id: u64, payload_id: u64) -> serde_json::Value {
    let mut decl = serde_json::json!({
        "def_id": def_id,
        "item_meta": fixture_item_meta(ident_path(&["fixture", "RawEnum"])),
        "kind": {"Enum": [{
            "name": "Mapped",
            "fields": [{
                "name": null,
                "ty": adt_node(payload_id),
                "attr_info": null
            }],
            "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
        }]}
    });
    decl["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 16,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0]}],
            "repr": {"transparent": false}
        }
    }]);
    decl
}

fn nested_option_raw_enum_nodes() -> (serde_json::Value, serde_json::Value) {
    let inner = option_adt_node(0, option_shared_ref(adt_node(2)));
    let outer = option_adt_node(0, inner.clone());
    (inner, outer)
}

/// rustc layout of `Option<Option<&RawEnum>>` is inner pointer-niche
/// (one word) plus an outer tag (two words). The outer cannot reuse the
/// inner None (`rclass.py` `InstanceRepr.convert_const(None)` /
/// `rnone.py` `rtype_is_None`).
#[test]
fn tagged_outer_option_of_niche_raw_enum_borrow_is_not_raw_address() {
    let llbc = llbc_with_types(
        "fixture",
        vec![option_enum(0), raw_store_struct(1), raw_storage_enum(2, 1)],
        vec![],
    );
    let empty_gc: std::collections::HashSet<majit_ir::descr::StructId> =
        std::collections::HashSet::new();
    let empty_s: std::collections::HashSet<String> = std::collections::HashSet::new();
    let tomb = super::no_tombstoned_leaves();
    let (inner_node, outer_node) = nested_option_raw_enum_nodes();
    let inner = TyRef::Other(inner_node);
    let outer = TyRef::Other(outer_node);
    let mut stack = Vec::new();
    assert!(
        super::adt_def_is_raw_storage(
            2,
            &llbc,
            &empty_gc,
            super::no_immutable_fields(),
            &mut stack
        ),
        "RawEnum with a raw-struct payload is storage-raw"
    );
    assert!(
        super::tyref_is_niche_option(&inner, &llbc),
        "Option<&RawEnum> is the pointer niche"
    );
    assert!(
        !super::tyref_is_niche_option(&outer, &llbc),
        "Option<Option<&RawEnum>> keeps the outer tag"
    );
    assert!(
        super::ty_is_raw_address(&inner, &llbc, &empty_gc, &empty_s),
        "inner Option<&RawEnum> is the int-bank null word"
    );
    assert!(
        !super::ty_is_raw_address(&outer, &llbc, &empty_gc, &empty_s),
        "tagged outer Option must not inherit the inner raw-address bank"
    );
    assert_eq!(
        super::tyref_niche_option_gc_payload_spelling(&inner, &llbc, &empty_gc, &empty_s),
        None,
        "raw-enum borrow Option is not a GC payload stamp"
    );
    assert_eq!(
        super::tyref_niche_option_gc_payload_spelling(&outer, &llbc, &empty_gc, &empty_s),
        None,
        "tagged outer Option is not a GC payload stamp"
    );
    let inner_vt = super::tyref_to_value_type_with(&inner, &llbc, tomb, &empty_gc, &empty_s);
    let outer_vt = super::tyref_to_value_type_with(&outer, &llbc, tomb, &empty_gc, &empty_s);
    assert!(
        matches!(inner_vt, ValueType::Int),
        "inner Option<&RawEnum> paints Int, got {inner_vt:?}"
    );
    match &outer_vt {
        ValueType::Ref(Some(root)) => {
            assert!(
                root.contains("Option"),
                "outer tagged Option class root names Option, got {root}"
            );
        }
        other => panic!("outer Option<Option<&RawEnum>> paints Ref(class), got {other:?}"),
    }
}

/// Extracted `flatten` body for `Option<Option<&RawEnum>>`: match the
/// outer tag, yield the inner niche word. Self is the tagged Option
/// class (`Ref`), not the inner address (`Int`), so the match must not
/// `__cast_instance_intrinsic` an Integer onto `Option<Option<RawEnum>>`.
#[test]
fn flatten_of_tagged_option_of_raw_enum_borrow_keeps_ref_self() {
    let (inner, outer) = nested_option_raw_enum_nodes();
    let disc_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "Isize"}}});
    let opt_place = window_place(serde_json::json!({"Local": 1}), &outer);
    let ret_place = window_place(serde_json::json!({"Local": 0}), &inner);
    let disc_place = window_place(serde_json::json!({"Local": 2}), &disc_ty);
    let some_payload = window_place(
        serde_json::json!({"Projection": [
            window_place(
                serde_json::json!({"Projection": [opt_place.clone(), {"Downcast": 1}]}),
                &outer
            ),
            {"Field": [1, 0]}
        ]}),
        &inner,
    );
    let none_aggregate = serde_json::json!({
        "Aggregate": [
            {"Adt": [
                {"id": 0, "generics": {
                    "regions": [],
                    "types": [option_shared_ref(adt_node(2))],
                    "const_generics": [],
                    "trait_refs": []
                }},
                0,
                null
            ]},
            []
        ]
    });
    let flatten = serde_json::json!({
        "def_id": 0,
        "item_meta": window_meta(&["fixture", "flatten"], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [outer.clone()],
            "output": inner.clone()
        },
        "body": {"Unstructured": {
            "span": window_span(),
            "locals": {"arg_count": 1, "locals": [
                window_local(0, &inner),
                window_local(1, &outer),
                window_local(2, &disc_ty)
            ]},
            "body": [
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            disc_place.clone(),
                            {"Discriminant": opt_place.clone()}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": {"Switch": {
                        "discr": {"Read": disc_place.clone()},
                        "targets": {"SwitchInt": [
                            null,
                            [
                                [{"Scalar": {"Signed": ["Isize", "1"]}}, 1],
                                [{"Scalar": {"Signed": ["Isize", "0"]}}, 2]
                            ],
                            2
                        ]}
                    }}}
                },
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            ret_place.clone(),
                            {"Use": [{"Read": some_payload}, "No"]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                },
                {
                    "statements": [{
                        "span": window_span(),
                        "kind": {"Assign": [
                            ret_place.clone(),
                            none_aggregate
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": window_span(), "kind": "Return"}
                }
            ]
        }}
    });
    let llbc = llbc_with_types(
        "fixture",
        vec![option_enum(0), raw_store_struct(1), raw_storage_enum(2, 1)],
        vec![flatten],
    );
    let graph = crate::test_support::lower_function(&llbc, "flatten")
        .unwrap_or_else(|err| panic!("lower flatten of Option<Option<&RawEnum>>: {err}"));
    assert!(
        option_core_body_matches_tag(&graph),
        "flatten matches the outer Option tag; graph={}",
        graph.dump()
    );
    let self_input = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .find_map(|op| match &op.kind {
            OpKind::Input { name, ty, .. } if name == "arg1" || name == "self" => Some(ty.clone()),
            _ => None,
        });
    let self_ty = self_input.expect("flatten stamps an Input for self");
    assert!(
        matches!(self_ty, ValueType::Ref(Some(ref root)) if root.contains("Option")),
        "flatten self of tagged Option<Option<&RawEnum>> is Ref(Option class), got {self_ty:?}; graph={}",
        graph.dump()
    );
    let pos0_int = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, ty: ValueType::Int, .. } if field.name == "__pos_0"
            )
        })
    });
    assert!(
        pos0_int,
        "Some.__pos_0 of the tagged outer is the inner niche Int word; graph={}",
        graph.dump()
    );
    let w_root_cast = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            crate::model::cast_instance_root(&op.kind).is_some_and(|root| root.contains("PyObject"))
        })
    });
    assert!(
        !w_root_cast,
        "inner niche word must not recast through W_ROOT; graph={}",
        graph.dump()
    );
}

#[test]
fn result_of_tombstoned_layout_leafs_paint_distinct_class_roots() {
    // `alloc::layout::Layout` and `typeobject::Layout` share the leaf
    // `Layout`. A leaf suffix would intern one
    // `result::Result<Layout,LayoutError>::Ok` whose `__pos_0` unions
    // them (`bookkeeper.py` `getuniqueclassdef`). The suffix spells
    // tombstoned-leaf type arguments with `class_identity_spelling`.
    let llbc = llbc_with_types(
        "fixture",
        vec![
            result_enum(0),
            opaque_type_decl(1, &["alloc", "layout", "Layout"]),
            struct_decl(
                2,
                ident_path(&["pyre_object", "typeobject", "Layout"]),
                empty_fields(),
                false,
            ),
            struct_decl(
                3,
                ident_path(&["alloc", "layout", "LayoutError"]),
                empty_fields(),
                false,
            ),
        ],
        vec![],
    );
    let tomb: std::collections::HashSet<String> = ["Layout".to_string()].into_iter().collect();
    let dest = |ok: serde_json::Value| {
        TyRef::Other(serde_json::json!({
            "Value": [7, result_adt_node(0, ok, adt_node(3))]
        }))
    };
    let alloc = dest(adt_node(1));
    let typeobject = dest(adt_node(2));
    let alloc_root = super::tyref_to_value_type_with(
        &alloc,
        &llbc,
        &tomb,
        &Default::default(),
        &Default::default(),
    );
    let typeobject_root = super::tyref_to_value_type_with(
        &typeobject,
        &llbc,
        &tomb,
        &Default::default(),
        &Default::default(),
    );
    let crate::model::ValueType::Ref(Some(alloc_name)) = &alloc_root else {
        panic!("expected Result class for alloc Layout, got {alloc_root:?}");
    };
    let crate::model::ValueType::Ref(Some(typeobject_name)) = &typeobject_root else {
        panic!("expected Result class for typeobject Layout, got {typeobject_root:?}");
    };
    assert_ne!(
        alloc_name, typeobject_name,
        "tombstoned Layout leaves must not share Result::Ok"
    );
    assert!(
        alloc_name.contains("layout::Layout"),
        "alloc Layout suffix should keep the module path, got {alloc_name}"
    );
    assert!(
        typeobject_name.contains("typeobject::Layout"),
        "typeobject Layout suffix should keep the module path, got {typeobject_name}"
    );
    assert_eq!(
        super::adt_head_instantiation_suffix(
            result_adt_node(0, adt_node(1), adt_node(3))
                .get("Adt")
                .and_then(|a| a.as_object())
                .unwrap(),
            &llbc,
            &tomb,
        )
        .as_deref(),
        Some("<layout::Layout,LayoutError>")
    );
    assert_eq!(
        super::adt_head_instantiation_suffix(
            result_adt_node(0, adt_node(2), adt_node(3))
                .get("Adt")
                .and_then(|a| a.as_object())
                .unwrap(),
            &llbc,
            &tomb,
        )
        .as_deref(),
        Some("<typeobject::Layout,LayoutError>")
    );
}

#[test]
fn result_with_pointer_like_err_still_paints_class_root() {
    // Result keeps its tag even when `E` is pointer-like
    // (`adt_node_class_root_identity`). The parameter carries the same
    // interned class (`bookkeeper.py` `getuniqueclassdef`).
    let llbc = llbc_with_types("fixture", vec![result_enum(0)], vec![]);
    let node = result_adt_node(0, option_int_arg(true), raw_mut_u8_arg());
    let ty = TyRef::Other(node);
    let painted = super::tyref_to_value_type(&ty, &llbc);
    match &painted {
        crate::model::ValueType::Ref(Some(root)) => {
            assert!(
                root.starts_with("result::Result<") && root.contains("i64"),
                "Result<i64, *mut u8> must intern as a Result class, got {root:?}"
            );
        }
        other => panic!("expected Ref(Some(Result<...>)), got {other:?}"),
    }
    let root = super::tyref_ref_param_class_root(&ty, &llbc, super::no_tombstoned_leaves(), None);
    assert!(
        root.as_deref()
            .is_some_and(|r| r.starts_with("result::Result<") && r.contains("i64")),
        "Result<i64, *mut u8> param must intern as a Result class, got {root:?}"
    );
}

#[test]
fn option_of_raw_pointer_stays_classdef_less() {
    // `Option<*mut u8>` is the null-word niche, not an enum class.
    // `*mut u8` is not a GC pointee (`is_niche_option_gc_payload_spelling`
    // with no layout is false); the Input stamp stays classdef-less. A GC pointee such as
    // `*mut pyobject::PyObject` is a different test.
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let node = option_adt_node(0, raw_mut_u8_arg());
    assert_eq!(
        super::adt_node_class_root_with(&node, &llbc, super::no_tombstoned_leaves()),
        None
    );
    let ty = TyRef::Other(node);
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, super::no_tombstoned_leaves(), None),
        None
    );
}

/// FunDecl template `Option<v0>` is a parameter of `Try::branch`. Value
/// paint stays classdef-less (recasting a call result onto the template
/// unions `__pos_0`); the Input `class_root` is the harvested enum base
/// so getattr `__discriminant` finds the template (`bookkeeper.py`
/// `getuniqueclassdef`).
#[test]
fn option_of_type_parameter_param_seeds_enum_template() {
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let tomb = super::no_tombstoned_leaves();
    let tvar = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
    let node = option_adt_node(0, tvar);
    assert_eq!(
        super::adt_node_class_root_identity(&node, &llbc, tomb),
        None,
        "value paint of Option<v0> stays classdef-less"
    );
    let ty = TyRef::Other(node.clone());
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, tomb, None),
        Some("option::Option".into()),
        "Option<v0> param seeds the harvested Option template"
    );
    let borrowed = TyRef::Other(option_shared_ref(node));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        Some("option::Option".into()),
        "&Option<v0> param seeds the harvested Option template"
    );
}

/// `Option<()>::branch` is a spec copy whose readable leaf is `__`
/// (`valid_identifier("()")`). `()` does not split, so the param still
/// seeds the harvested Option template for getattr `__discriminant`.
#[test]
fn option_of_unit_param_seeds_enum_template() {
    let llbc = llbc_with_types("fixture", vec![option_enum(0)], vec![]);
    let tomb = super::no_tombstoned_leaves();
    let unit = serde_json::json!({"Tuple": []});
    let node = option_adt_node(0, unit);
    assert_eq!(
        super::adt_node_class_root_identity(&node, &llbc, tomb),
        None,
        "value paint of Option<()> stays classdef-less"
    );
    let ty = TyRef::Other(node.clone());
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, tomb, None),
        Some("option::Option".into()),
        "Option<()> param seeds the harvested Option template"
    );
    let borrowed = TyRef::Other(option_shared_ref(node));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        Some("option::Option".into()),
        "&Option<()> param seeds the harvested Option template"
    );
}

fn fixture_llbc_with_gctype(
    types: Vec<serde_json::Value>,
    gctype_type_ids: &[u64],
    extra_impls: Vec<serde_json::Value>,
) -> Llbc {
    let mut trait_impls: Vec<serde_json::Value> = gctype_type_ids
        .iter()
        .copied()
        .map(gctype_impl_for_adt)
        .collect();
    trait_impls.extend(extra_impls);
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": types,
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": trait_impls
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
}

fn option_payload_stamp_decls() -> Vec<serde_json::Value> {
    let gcrefopaque = serde_json::json!({
        "Adt": {"id": 1, "generics": empty_generics()}
    });
    let object_adt = serde_json::json!({
        "Adt": {"id": 3, "generics": empty_generics()}
    });
    vec![
        option_enum(0),
        type_decl_struct(1, &["majit_gc", "header", "GCREFOpaque"], vec![]),
        type_decl_alias(2, &["majit_gc", "header", "GCREF"], {
            serde_json::json!({"RawPtr": [gcrefopaque, "Mut"]})
        }),
        type_decl_struct(3, &["pyre_object", "pyobject", "PyObject"], vec![]),
        type_decl_alias(4, &["pyre_object", "pyobject", "PyObjectRef"], {
            serde_json::json!({"RawPtr": [object_adt, "Mut"]})
        }),
        type_decl_struct(5, &["object_array", "TypedItemsBlock"], vec![]),
    ]
}

fn option_payload_stamp_llbc() -> Llbc {
    fixture_llbc_with_gctype(
        option_payload_stamp_decls(),
        &[3, 5],
        vec![serde_json::json!({
            "impl_trait": {
                "id": 1,
                "generics": {"types": [adt_node(3)]}
            },
            "types": [{
                "kind": {"TraitType": [1, 0]},
                "skip_binder": {"value": raw_mut_adt(3)}
            }]
        })],
    )
}

fn option_adt_ty(payload: serde_json::Value) -> serde_json::Value {
    option_adt_node(0, payload)
}

fn raw_mut_adt(id: u64) -> serde_json::Value {
    serde_json::json!({"RawPtr": [adt_node(id), "Mut"]})
}

fn stamp_option_self_input(llbc: &Llbc, opt_ty: &serde_json::Value) -> (ValueType, Option<String>) {
    let inputs = stamp_option_method_inputs(llbc, opt_ty, None);
    inputs
        .into_iter()
        .find(|(name, _, _)| name == "self")
        .map(|(_, ty, root)| (ty, root))
        .expect("self Input")
}

fn stamp_option_method_inputs(
    llbc: &Llbc,
    opt_ty: &serde_json::Value,
    closure_ty: Option<&serde_json::Value>,
) -> Vec<(String, ValueType, Option<String>)> {
    let span = fixture_span();
    let mut locals_json = vec![
        serde_json::json!({"index": 0, "name": null, "span": span, "ty": {"Tuple": []}}),
        serde_json::json!({"index": 1, "name": "self", "span": span, "ty": opt_ty}),
    ];
    let arg_count = if let Some(closure_ty) = closure_ty {
        locals_json.push(serde_json::json!({
            "index": 2, "name": "f", "span": span, "ty": closure_ty
        }));
        2
    } else {
        1
    };
    let locals: Locals = serde_json::from_value(serde_json::json!({
        "arg_count": arg_count,
        "locals": locals_json
    }))
    .expect("fixture Locals");
    let mut graph = FunctionGraph::new("option::Option::unwrap_or_else");
    let empty_s: std::collections::HashSet<String> = std::collections::HashSet::new();
    let empty_u: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let gc_struct_ids = super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    super::super::flowcontext::pygraph_initial_block(
        &mut graph,
        &locals,
        llbc,
        None,
        &empty_s,
        &empty_u,
        &empty_u,
        &gc_struct_ids,
        false,
        &empty_s,
    );
    graph
        .block(graph.startblock)
        .operations
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input {
                name,
                ty,
                class_root,
            } => Some((name.clone(), ty.clone(), class_root.clone())),
            _ => None,
        })
        .collect()
}

/// Substituted pointer-niche `Option<P>` Input stamp from `local.ty`.
/// GCREF → `SomePtr` class_root; GC instance / `&PyObjectRef` /
/// `*mut TypedItemsBlock` → `Option<{payload}>` so intern noneifies
/// when the pointee has a Gc layout. `Option<i64>` is the tagged enum
/// and stays `option::Option<i64>`.
#[test]
fn substituted_pointer_niche_option_param_input_stamp() {
    let llbc = option_payload_stamp_llbc();
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let empty_s: std::collections::HashSet<String> = std::collections::HashSet::new();
    let helper =
        |ty: &TyRef| tyref_niche_option_gc_payload_spelling(ty, &llbc, &gc_struct_ids, &empty_s);
    let (layouts, _ids) = crate::test_support::gc_layout_table_for(&[
        "PyObject",
        "pyobject::PyObject",
        "TypedItemsBlock",
    ]);
    let intern = |s: &str| is_niche_option_gc_payload_spelling(s, Some(&layouts));

    let gcref_ty = TyRef::Other(option_adt_ty(raw_mut_adt(1)));
    let gcref_root = helper(&gcref_ty);
    assert!(
        gcref_root
            .as_deref()
            .is_some_and(crate::annotator::bookkeeper::is_gcref_type_spelling),
        "Option<*mut GCREFOpaque> stamps a GCREF class_root, got {gcref_root:?}"
    );
    let (gcref_ty_in, gcref_class) = stamp_option_self_input(&llbc, &option_adt_ty(raw_mut_adt(1)));
    assert!(
        !matches!(gcref_ty_in, ValueType::Void),
        "GCREF Option self is not Void"
    );
    assert!(
        gcref_class
            .as_deref()
            .is_some_and(crate::annotator::bookkeeper::is_gcref_type_spelling),
        "Input stamp of Option<*mut GCREFOpaque> is GCREF, got {gcref_class:?}"
    );

    let py_ty = option_adt_ty(raw_mut_adt(3));
    let (py_in, py_class) = stamp_option_self_input(&llbc, &py_ty);
    assert!(
        !matches!(py_in, ValueType::Void),
        "Option<*mut PyObject> self is not Void"
    );
    assert!(
        py_class.as_deref().is_some_and(intern),
        "Option<*mut PyObject> stamps a niche Option, got {py_class:?}"
    );
    assert!(
        py_class.as_deref().is_some_and(|r| r.contains("PyObject")),
        "payload spelling names PyObject, got {py_class:?}"
    );

    let borrowed_raw = serde_json::json!({"Ref": ["Erased", raw_mut_adt(3), "Shared"]});
    let (borrowed_raw_in, borrowed_raw_class) =
        stamp_option_self_input(&llbc, &option_adt_ty(borrowed_raw));
    assert!(
        !matches!(borrowed_raw_in, ValueType::Void),
        "Option<&*mut PyObject> self is not Void"
    );
    assert!(
        borrowed_raw_class.as_deref().is_some_and(intern),
        "Option<&*mut PyObject> stamps a niche Option, got {borrowed_raw_class:?}"
    );

    let borrowed_ref = serde_json::json!({"Ref": ["Erased", adt_node(4), "Shared"]});
    let (borrowed_in, borrowed_class) =
        stamp_option_self_input(&llbc, &option_adt_ty(borrowed_ref));
    assert!(
        !matches!(borrowed_in, ValueType::Void),
        "Option<&PyObjectRef> self is not Void"
    );
    assert!(
        borrowed_class.as_deref().is_some_and(intern),
        "Option<&PyObjectRef> stamps a niche Option, got {borrowed_class:?}"
    );
    assert_eq!(
        borrowed_class, borrowed_raw_class,
        "Option<&PyObjectRef> stamps the alias-resolved pointer, got {borrowed_class:?}"
    );

    let block_ty = option_adt_ty(raw_mut_adt(5));
    let (block_in, block_class) = stamp_option_self_input(&llbc, &block_ty);
    assert!(
        !matches!(block_in, ValueType::Void),
        "Option<*mut TypedItemsBlock> self is not Void"
    );
    assert!(
        block_class.as_deref().is_some_and(intern),
        "Option<*mut TypedItemsBlock> stamps a niche Option, got {block_class:?}"
    );
    assert!(
        block_class
            .as_deref()
            .is_some_and(|r| r.contains("TypedItemsBlock")),
        "payload spelling names TypedItemsBlock, got {block_class:?}"
    );

    let alias_ty = option_adt_ty(adt_node(4));
    let (alias_in, alias_class) = stamp_option_self_input(&llbc, &alias_ty);
    assert!(
        !matches!(alias_in, ValueType::Void),
        "Option<PyObjectRef> self is not Void"
    );
    assert!(
        alias_class.as_deref().is_some_and(intern),
        "Option<PyObjectRef> stamps a niche Option, got {alias_class:?}"
    );
    assert_eq!(
        alias_class, py_class,
        "Option<PyObjectRef> stamps the alias-resolved pointer, got {alias_class:?}"
    );

    let borrowed_object = serde_json::json!({"Ref": ["Erased", adt_node(3), "Shared"]});
    let (borrowed_object_in, borrowed_object_class) =
        stamp_option_self_input(&llbc, &option_adt_ty(borrowed_object));
    assert!(
        !matches!(borrowed_object_in, ValueType::Void),
        "Option<&PyObject> self is not Void"
    );
    assert!(
        borrowed_object_class.as_deref().is_some_and(intern),
        "Option<&PyObject> stamps a niche Option, got {borrowed_object_class:?}"
    );

    let borrowed_block = serde_json::json!({"Ref": ["Erased", adt_node(5), "Shared"]});
    let (borrowed_block_in, borrowed_block_class) =
        stamp_option_self_input(&llbc, &option_adt_ty(borrowed_block));
    assert!(
        !matches!(borrowed_block_in, ValueType::Void),
        "Option<&TypedItemsBlock> self is not Void"
    );
    assert!(
        borrowed_block_class.as_deref().is_some_and(intern),
        "Option<&TypedItemsBlock> stamps a niche Option, got {borrowed_block_class:?}"
    );

    let assoc = serde_json::json!({
        "TraitType": [
            { "trait_decl_ref": { "skip_binder": { "id": 1 } } },
            0,
            { "regions": [], "types": [], "const_generics": [], "trait_refs": [] }
        ]
    });
    let (assoc_in, assoc_class) = stamp_option_self_input(&llbc, &option_adt_ty(assoc));
    assert!(
        !matches!(assoc_in, ValueType::Void),
        "Option<H::Value> self is not Void"
    );
    assert!(
        assoc_class.as_deref().is_some_and(intern),
        "unique-impl TraitType payload *mut PyObject stamps a niche Option, got {assoc_class:?}"
    );

    let i64_ty = option_adt_ty(option_int_arg(true));
    let (i64_in, i64_class) = stamp_option_self_input(&llbc, &i64_ty);
    assert!(
        !matches!(i64_in, ValueType::Void),
        "Option<i64> self is not Void"
    );
    assert_eq!(
        i64_class.as_deref(),
        Some("option::Option<i64>"),
        "tagged Option<i64> keeps its enum class_root, got {i64_class:?}"
    );
    assert!(
        !i64_class.as_deref().is_some_and(intern),
        "Option<i64> is not a pointer-niche GC payload"
    );
}

/// A pointee named `PyObject` that is not in `gc_struct_ids` must not
/// noneify. The Input stamp keys on the declaration, not the leaf name.
#[test]
fn undeclared_named_object_pointee_is_not_noneify_stamp() {
    let llbc = llbc_with_types("fixture", option_payload_stamp_decls(), vec![]);
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert!(
        gc_struct_ids.is_empty(),
        "fixture PyObject without a GcType impl is not harvested"
    );
    let empty_s: std::collections::HashSet<String> = std::collections::HashSet::new();
    let py_ty = option_adt_ty(raw_mut_adt(3));
    let helper = tyref_niche_option_gc_payload_spelling(
        &TyRef::Other(py_ty.clone()),
        &llbc,
        &gc_struct_ids,
        &empty_s,
    );
    assert_eq!(
        helper, None,
        "undeclared PyObject pointee must not stamp, got {helper:?}"
    );
    let (py_in, py_class) = stamp_option_self_input(&llbc, &py_ty);
    assert!(
        !matches!(py_in, ValueType::Void),
        "Option<*mut PyObject> self is not Void"
    );
    assert!(
        !py_class
            .as_deref()
            .is_some_and(|s| is_niche_option_gc_payload_spelling(s, None)),
        "Input class_root of undeclared PyObject is not a GC payload, got {py_class:?}"
    );
}

/// `Option<&RawEnum>` is the int-bank null word (`borrowed_pointee_is_raw_enum`),
/// not a GC instance. Intern's layout gckind
/// (`is_niche_option_gc_payload_spelling`) does not noneify an
/// undeclared uppercase `&T`; rtype then fails `cast_int_to_ptr`
/// InstanceRepr → IntegerRepr if intern did.
#[test]
fn substituted_raw_enum_borrow_option_is_not_gc_payload_stamp() {
    let mut option = option_enum(0);
    option["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 8,
            "align": 8,
            "variant_layouts": [
                {"field_offsets": []},
                {"field_offsets": [0]}
            ],
            "repr": {"transparent": false}
        }
    }]);
    let mut raw_enum = local_payload_enum(1);
    raw_enum["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 16,
            "align": 8,
            "variant_layouts": [
                {"field_offsets": []},
                {"field_offsets": [8]}
            ],
            "repr": {"transparent": false}
        }
    }]);
    let llbc = llbc_with_types("fixture", vec![option, raw_enum], vec![]);
    let empty_gc: std::collections::HashSet<majit_ir::descr::StructId> =
        std::collections::HashSet::new();
    let empty_s: std::collections::HashSet<String> = std::collections::HashSet::new();
    let opt_ty = option_adt_ty(option_shared_ref(adt_node(1)));
    let helper = tyref_niche_option_gc_payload_spelling(
        &TyRef::Other(opt_ty.clone()),
        &llbc,
        &empty_gc,
        &empty_s,
    );
    assert_eq!(
        helper, None,
        "Option<&RawEnum> must not stamp a GC Option spelling, got {helper:?}"
    );
    let (ty_in, class) = stamp_option_self_input(&llbc, &opt_ty);
    assert!(
        !matches!(ty_in, ValueType::Void),
        "raw-enum Option self is not Void"
    );
    assert!(
        !class
            .as_deref()
            .is_some_and(|s| is_niche_option_gc_payload_spelling(s, None)),
        "Input class_root of Option<&RawEnum> is not a GC payload, got {class:?}"
    );
}

/// Niched `unwrap_or_else` keeps `self` as an Input even when Charon
/// records a size-0 layout on the generic Option decl. The ZST closure
/// `f` stays an Input too (`tyref_keeps_callable_identity`).
#[test]
fn niched_unwrap_or_else_keeps_self_input() {
    let mut option = option_enum(0);
    option["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 0,
            "align": 8,
            "variant_layouts": [
                {"field_offsets": []},
                {"field_offsets": [0]}
            ],
            "repr": {"transparent": false}
        }
    }]);
    let llbc = fixture_llbc_with_gctype(
        vec![
            option,
            type_decl_struct(1, &["pyre_object", "pyobject", "PyObject"], vec![]),
            {
                let mut closure = type_decl_struct(
                    2,
                    &["object_array", "alloc_typed_items_block", "closure"],
                    vec![],
                );
                closure["src"] = serde_json::json!({"Closure": {"info": {"kind": "FnOnce"}}});
                closure
            },
        ],
        &[1],
        vec![],
    );
    let (layouts, _ids) = crate::test_support::gc_layout_table_for(&["PyObject"]);
    let intern = |s: &str| is_niche_option_gc_payload_spelling(s, Some(&layouts));
    let opt_ty = option_adt_ty(raw_mut_adt(1));
    let closure_ty = adt_node(2);
    let ty = TyRef::Other(opt_ty.clone());
    assert!(
        !tyref_is_void_zst(&ty, &llbc),
        "pointer-niche Option is not Void even with size-0 generic layout"
    );
    let inputs = stamp_option_method_inputs(&llbc, &opt_ty, Some(&closure_ty));
    let names: Vec<&str> = inputs.iter().map(|(n, _, _)| n.as_str()).collect();
    assert!(
        names.contains(&"self"),
        "niched unwrap_or_else keeps self Input, got {names:?}"
    );
    assert!(
        names.contains(&"f"),
        "ZST closure f stays an Input, got {names:?}"
    );
    let self_input = inputs
        .iter()
        .find(|(n, _, _)| n == "self")
        .expect("self Input");
    assert!(
        !matches!(self_input.1, ValueType::Void),
        "self is a payload word, not Void"
    );
    assert!(
        self_input.2.as_deref().is_some_and(intern),
        "self class_root is the niche Option spelling, got {:?}",
        self_input.2
    );
    let f_input = inputs.iter().find(|(n, _, _)| n == "f").expect("f Input");
    assert!(
        !matches!(f_input.1, ValueType::Void),
        "ZST closure is not dropped as Void"
    );
}

fn local_payload_enum(def_id: u64) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": [
                {"Ident": ["fixture", 0]},
                {"Ident": ["RawEnum", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "kind": {"Enum": [
            {
                "name": "Empty",
                "fields": [],
                "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
            },
            {
                "name": "Word",
                "fields": [{"name": null, "ty": {"Value": [0, {"Literal": "Int"}]}, "attr_info": null}],
                "discriminant": {"Scalar": {"Signed": ["Isize", "1"]}}
            }
        ]}
    })
}

/// A payload enum with no type arguments is not a FunDecl template.
/// `tyref_split_core_enum_param_root` must not seed it; Input class_root
/// comes from `adt_node_class_root_identity`.
#[test]
fn unsuffixed_local_payload_enum_param_is_not_a_template_seed() {
    let llbc = llbc_with_types("fixture", vec![local_payload_enum(0)], vec![]);
    let tomb = super::no_tombstoned_leaves();
    let node = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let ty = TyRef::Other(node.clone());
    assert_eq!(
        super::tyref_split_core_enum_param_root(&ty, &llbc, tomb),
        None,
        "Buffer-shaped enum has no type args; do not seed the Option<()> path"
    );
    assert_eq!(
        super::adt_node_class_root_identity(&node, &llbc, tomb).as_deref(),
        Some("RawEnum"),
        "value/param paint still names the harvested identity"
    );
}

/// A named word-sized tuple struct is a tagged Option payload, not a
/// pointer niche. `getattr __discriminant` needs the interned class.
#[test]
fn option_of_word_struct_interns_tagged_class() {
    let llbc = llbc_with_types(
        "fixture",
        vec![option_enum(0), code_struct(1, &["hash", "WordHasher"])],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let node = option_adt_node(0, adt_node(1));
    let root = super::adt_node_class_root_with(&node, &llbc, tomb);
    assert_eq!(root.as_deref(), Some("option::Option<WordHasher>"));
    let ty = TyRef::Other(node);
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        crate::model::ValueType::Ref(Some("option::Option<WordHasher>".into()))
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, tomb, None),
        Some("option::Option<WordHasher>".into())
    );
    let borrowed = TyRef::Other(option_shared_ref(option_adt_node(0, adt_node(1))));
    assert_eq!(
        super::tyref_to_value_type(&borrowed, &llbc),
        crate::model::ValueType::Ref(Some("option::Option<WordHasher>".into()))
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        Some("option::Option<WordHasher>".into())
    );
}

/// `Option<&NamedStruct>` is the one-word null niche even when the
/// pointee TypeDecl has no concrete layout size. Last-field walk sees
/// a sized word field; intern stays classdef-less.
#[test]
fn option_of_shared_ref_to_generic_struct_is_niche() {
    let llbc = llbc_with_types(
        "fixture",
        vec![option_enum(0), code_struct(1, &["hash", "WordHasher"])],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let node = option_adt_node(0, option_shared_ref(adt_node(1)));
    let ty = TyRef::Other(node.clone());
    assert!(
        super::tyref_is_niche_option(&ty, &llbc),
        "Option<&WordHasher> is the nullable pointer niche"
    );
    assert_eq!(
        super::adt_node_class_root_with(&node, &llbc, tomb),
        None,
        "niche Option<&Struct> must not intern a tagged class"
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, tomb, None),
        None
    );
}

/// Thin `Arc<T>` / `Rc<T>` is a one-word owning pointer, so Option of
/// either is the null niche. Fat `Arc<[T]>` keeps the tag.
#[test]
fn option_of_thin_arc_or_rc_is_niche() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            type_decl_struct(1, &["alloc", "sync", "Arc"], vec![]),
            type_decl_struct(2, &["alloc", "rc", "Rc"], vec![]),
            type_decl_struct(3, &["jit", "JitCode"], vec![]),
        ],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let jitcode = named_adt(3, vec![]);
    let option_arc = option_adt_node(0, named_adt(1, vec![jitcode.clone()]));
    let option_rc = option_adt_node(0, named_adt(2, vec![jitcode]));
    let ty_arc = TyRef::Other(option_arc.clone());
    let ty_rc = TyRef::Other(option_rc.clone());
    assert!(
        super::tyref_is_niche_option(&ty_arc, &llbc),
        "Option of a thin Arc is the nullable pointer niche"
    );
    assert!(
        super::tyref_is_niche_option(&ty_rc, &llbc),
        "Option of a thin Rc is the nullable pointer niche"
    );
    assert_eq!(
        super::adt_node_class_root_with(&option_arc, &llbc, tomb),
        None,
        "niche Option of Arc must not intern a tagged class"
    );
    let fat_inner = serde_json::json!({
        "Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, null]
    });
    let option_fat = TyRef::Other(option_adt_node(0, named_adt(1, vec![fat_inner])));
    assert!(
        !super::tyref_is_niche_option(&option_fat, &llbc),
        "Option of a fat Arc slice keeps the tag"
    );
}

/// A DST newtype whose last field is a Slice is two words. The
/// shared-ref arm must not treat a missing layout size as thin.
#[test]
fn option_of_shared_ref_to_dst_newtype_is_not_pointer_niche() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let dst = serde_json::json!({
        "def_id": 1,
        "item_meta": {
            "name": [
                {"Ident": ["bytes", 0]},
                {"Ident": ["Bytes", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "kind": {"Struct": [{
            "name": "bytes",
            "ty": {"Slice": [{"Scalar": {"Integer": {"Unsigned": "U8"}}}, null]},
            "attr_info": null
        }]}
    });
    let llbc = llbc_with_types("fixture", vec![option_enum(0), dst], vec![]);
    let node = option_adt_node(0, option_shared_ref(adt_node(1)));
    let ty = TyRef::Other(node);
    assert!(
        !super::tyref_is_niche_option(&ty, &llbc),
        "Option<&Bytes> is a fat pointer, not a one-word null niche"
    );
}

/// Extra Instantiated args beside Option's payload (a hasher phantom
/// `fn() -> H`) are not the niche test. Intern still uses the tagged
/// Option class so `getattr __discriminant` resolves.
#[test]
fn option_with_extra_fn_ptr_arg_still_interns() {
    let llbc = llbc_with_types(
        "fixture",
        vec![option_enum(0), code_struct(1, &["hash", "WordHasher"])],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let node = option_adt_node_from_types(0, vec![adt_node(1), unit_fn_ptr()]);
    let root = super::adt_node_class_root_with(&node, &llbc, tomb);
    assert!(
        root.as_deref()
            .is_some_and(|r| r.starts_with("option::Option<") && r.contains("WordHasher")),
        "tagged Option with a leaked fn-ptr Instantiated arg must intern, got {root:?}"
    );
    let ty = TyRef::Other(node);
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, tomb, None),
        root.clone()
    );
    let borrowed = TyRef::Other(option_shared_ref(option_adt_node_from_types(
        0,
        vec![adt_node(1), unit_fn_ptr()],
    )));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        root
    );
}

fn generic_adt_node(def_id: u64, types: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": def_id,
            "generics": {
                "regions": [],
                "types": types,
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

fn option_enum_instantiated(def_id: u64, skip_binder: serde_json::Value) -> serde_json::Value {
    let mut decl = option_enum(def_id);
    decl["item_meta"]["name"]
        .as_array_mut()
        .expect("option name")
        .push(serde_json::json!({
            "Instantiated": {
                "params": {},
                "kind": "Other",
                "skip_binder": skip_binder
            }
        }));
    decl
}

/// A tagged Option whose payload is a three-argument std container
/// (`HashMap<K,V,BuildHasherDefault<H>>`) intern's so
/// `getattr __discriminant` resolves. The hasher is a nested ADT,
/// not a pointer niche.
#[test]
fn option_of_nested_hasher_map_interns_tagged_class() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            code_struct(1, &["hash", "WordHasher"]),
            code_struct(2, &["core", "hash", "BuildHasherDefault"]),
            code_struct(3, &["alloc", "collections", "hashmap", "HashMap"]),
        ],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let hasher = generic_adt_node(2, vec![adt_node(1)]);
    let map = generic_adt_node(3, vec![adt_node(1), adt_node(1), hasher]);
    let node = option_adt_node(0, map);
    let root = super::adt_node_class_root_with(&node, &llbc, tomb);
    assert!(
        root.as_deref()
            .is_some_and(|r| r.starts_with("option::Option<") && r.contains("HashMap")),
        "tagged Option of HashMap<_,_,BuildHasherDefault<_>> must intern, got {root:?}"
    );
    let borrowed = TyRef::Other(option_shared_ref(node));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        root
    );
}

/// Charon-monomorphized Option is a TypeDecl whose Instantiated
/// segment carries the payload; the Adt node has no use-site
/// generics. Intern still paints the tagged class.
#[test]
fn option_instantiated_type_decl_interns_tagged_class() {
    let skip = serde_json::json!({
        "regions": [],
        "types": [adt_node(1)],
        "const_generics": [],
        "trait_refs": []
    });
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum_instantiated(0, skip),
            code_struct(1, &["hash", "WordHasher"]),
        ],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let node = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        }
    });
    let root = super::adt_node_class_root_with(&node, &llbc, tomb);
    assert_eq!(
        root.as_deref(),
        Some("option::Option<WordHasher>"),
        "Instantiated Option TypeDecl with empty use-site generics must intern, got {root:?}"
    );
    let borrowed = TyRef::Other(option_shared_ref(node));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        Some("option::Option<WordHasher>".into())
    );
}

/// Instantiated skip_binder is itself hash-consed (`Deduplicated` of
/// the instantiated Adt). Intern follows that body to the payload.
#[test]
fn option_instantiated_dedup_skip_binder_interns_tagged_class() {
    let instantiated_adt = option_adt_node(0, adt_node(1));
    let carrier = {
        let mut s = code_struct(2, &["hash", "SkipCarrier"]);
        s["kind"]["Struct"][0]["ty"] = serde_json::json!({"Value": [77, instantiated_adt]});
        s
    };
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum_instantiated(0, serde_json::json!({"Deduplicated": 77})),
            code_struct(1, &["hash", "WordHasher"]),
            carrier,
        ],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let node = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}
        }
    });
    let root = super::adt_node_class_root_with(&node, &llbc, tomb);
    assert_eq!(
        root.as_deref(),
        Some("option::Option<WordHasher>"),
        "Deduplicated Instantiated skip_binder must intern, got {root:?}"
    );
    let borrowed = TyRef::Other(option_shared_ref(node));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        Some("option::Option<WordHasher>".into())
    );
}

/// `Option<RDict<K, *mut T, BuildHasherDefault<H>>>` is a tagged
/// enum (the dict is multi-word). Intern paints the Option class.
#[test]
fn option_of_dict_with_hasher_and_raw_ptr_interns_tagged_class() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            code_struct(1, &["hash", "WordHasher"]),
            code_struct(2, &["core", "hash", "BuildHasherDefault"]),
            code_struct(3, &["dict", "rordereddict", "RDict"]),
            code_struct(4, &["object", "Header"]),
        ],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let hasher = generic_adt_node(2, vec![adt_node(1)]);
    let raw = serde_json::json!({"RawPtr": [adt_node(4), "Mut"]});
    let dict = generic_adt_node(3, vec![adt_node(1), raw, hasher]);
    let node = option_adt_node(0, dict);
    let root = super::adt_node_class_root_with(&node, &llbc, tomb);
    assert!(
        root.as_deref()
            .is_some_and(|r| r.starts_with("option::Option<") && r.contains("RDict")),
        "tagged Option of RDict<_,_,BuildHasherDefault<_>> must intern, got {root:?}"
    );
    let borrowed = TyRef::Other(option_shared_ref(node));
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        root
    );
}

#[test]
fn control_flow_value_and_param_share_instantiation_class() {
    // A split core ControlFlow value and parameter take the same instantiation class root.
    let llbc = llbc_with_types("fixture", vec![control_flow_enum(0)], vec![]);
    let node = result_adt_node(0, option_int_arg(true), option_int_arg(true));
    assert_eq!(
        super::adt_node_class_root_with(&node, &llbc, super::no_tombstoned_leaves()),
        Some("ops::control_flow::ControlFlow<i64,i64>".into())
    );
    let ty = TyRef::Other(node);
    assert_eq!(
        super::tyref_to_value_type(&ty, &llbc),
        crate::model::ValueType::Ref(Some("ops::control_flow::ControlFlow<i64,i64>".into()))
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&ty, &llbc, super::no_tombstoned_leaves(), None),
        Some("ops::control_flow::ControlFlow<i64,i64>".into())
    );
}

fn control_flow_enum(def_id: u64) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": [
                {"Ident": ["core", 0]},
                {"Ident": ["ops", 0]},
                {"Ident": ["control_flow", 0]},
                {"Ident": ["ControlFlow", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Enum": [
            {
                "name": "Continue",
                "fields": [{"name": null, "ty": {"TypeVar": {"Bound": [0, 1]}}, "attr_info": null}],
                "discriminant": {"Scalar": {"Signed": ["Isize", "0"]}}
            },
            {
                "name": "Break",
                "fields": [{"name": null, "ty": {"TypeVar": {"Bound": [0, 0]}}, "attr_info": null}],
                "discriminant": {"Scalar": {"Signed": ["Isize", "1"]}}
            }
        ]}
    })
}

#[test]
fn control_flow_integer_instantiations_paint_distinct_class_roots() {
    // `?` builds `ControlFlow::Continue(payload)` (`unaryop.py`
    // `setattr` `__pos_0`). Each instantiation is its own class
    // (`bookkeeper.py` `getuniqueclassdef(cls)`), so Continue(i32)
    // does not `generalize_attr` against Continue(()) (`int ∪ None`,
    // `classdesc.py`).
    let llbc = llbc_with_types("fixture", vec![control_flow_enum(0)], vec![]);
    let dest = |cont: serde_json::Value, brk: serde_json::Value| {
        TyRef::Other(serde_json::json!({
            "Value": [7, result_adt_node(0, brk, cont)]
        }))
    };
    let i32_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I32"}}});
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let usize_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}});
    // ControlFlow<B, C>: type 0 is Break, type 1 is Continue.
    let i32_u8 = super::tyref_to_value_type(&dest(i32_ty, u8_ty.clone()), &llbc);
    let usize_u8 = super::tyref_to_value_type(&dest(usize_ty, u8_ty), &llbc);
    assert_eq!(
        i32_u8,
        crate::model::ValueType::Ref(Some("ops::control_flow::ControlFlow<u8,i32>".into()))
    );
    assert_eq!(
        usize_u8,
        crate::model::ValueType::Ref(Some("ops::control_flow::ControlFlow<u8,usize>".into()))
    );
    assert_ne!(i32_u8, usize_u8);
}

#[test]
fn a_unique_crate_root_struct_keeps_its_bare_leaf() {
    let llbc = llbc_with_types(
        "charon_corpus",
        vec![code_struct(0, &["charon_corpus", "ClassObject"])],
        vec![],
    );
    let tombstoned = super::tombstoned_leaves_of(&llbc);
    assert!(
        !tombstoned.contains("ClassObject"),
        "a crate-root decl stores an empty origin module without being a duplicate"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &llbc, &tombstoned).as_deref(),
        Some("ClassObject"),
        "the class-static narrow and the field registry both key the bare leaf"
    );
}

/// An Opaque scalar-word projection takes the declared name from the
/// defining crate's harvest. `__pos_N` is only the missing-row fallback.
#[test]
fn opaque_struct_field_name_uses_harvested_declared_name() {
    let mut reg = crate::front::semantic::StructFieldRegistry::default();
    reg.fields.insert(
        "error::PyError".to_string(),
        vec![("kind".to_string(), "i32".to_string())],
    );
    assert_eq!(
        super::opaque_struct_field_name("pyre_interpreter::error::PyError", 0, &reg),
        "kind"
    );
    assert_eq!(
        super::opaque_struct_field_name("error::PyError", 0, &reg),
        "kind"
    );
    assert_eq!(
        super::opaque_struct_field_name("unknown::Owner", 0, &reg),
        "__pos_0"
    );
    assert_eq!(
        super::opaque_struct_field_name("error::PyError", 1, &reg),
        "__pos_1"
    );
}

/// Two same-leaf structs intern as distinct class identities. The
/// crate-stripped `module::Type` is the intern key
/// (`bookkeeper.py` `Bookkeeper.getdesc`); the bare leaf is display-only.
#[test]
fn two_same_leaf_structs_paint_distinct_class_roots() {
    let llbc = llbc_with_types(
        "fixture",
        vec![
            code_struct(0, &["fixture", "a", "Dup"]),
            code_struct(1, &["fixture", "b", "Dup"]),
        ],
        vec![],
    );
    let tomb = super::no_tombstoned_leaves();
    let a = TyRef::Other(adt_node(0));
    let b = TyRef::Other(adt_node(1));
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &llbc, tomb).as_deref(),
        Some("a::Dup")
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(1), &llbc, tomb).as_deref(),
        Some("b::Dup")
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&a, &llbc, tomb, None).as_deref(),
        Some("a::Dup")
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&b, &llbc, tomb, None).as_deref(),
        Some("b::Dup")
    );
}

#[test]
fn cross_llbc_duplicate_leaf_is_tombstoned_before_lowering() {
    let interpreter = llbc_with_types(
        "pyre_interpreter",
        vec![code_struct(0, &["pyre_interpreter", "eval", "FrameBlock"])],
        vec![],
    );
    let jit = llbc_with_types(
        "pyre_jit",
        vec![code_struct(0, &["pyre_jit", "trace", "FrameBlock"])],
        vec![],
    );
    assert!(
        !super::tombstoned_leaves_of(&interpreter).contains("FrameBlock"),
        "each artefact alone has one FrameBlock"
    );
    assert!(!super::tombstoned_leaves_of(&jit).contains("FrameBlock"));
    let mut facts = super::DuplicateLeafFacts::default();
    facts.absorb(super::DuplicateLeafFacts::discover(
        &interpreter,
        &std::collections::HashSet::new(),
    ));
    facts.absorb(super::DuplicateLeafFacts::discover(
        &jit,
        &std::collections::HashSet::new(),
    ));
    let cross = facts.tombstoned_leaves();
    assert!(
        cross.contains("FrameBlock"),
        "the merged verdict withdraws a leaf that collides only across LLBCs"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &interpreter, &cross).as_deref(),
        Some("eval::FrameBlock")
    );
}

/// `a::Foo` and `b::Foo` mint one `StructId` from the stripped leaf, so
/// the identity table sees a single declaration. The merge still
/// withdraws `Foo` because the defining crates differ.
#[test]
fn cross_llbc_crate_root_leaf_is_tombstoned_when_crates_differ() {
    let a = llbc_with_types("a", vec![code_struct(0, &["a", "Foo"])], vec![]);
    let b = llbc_with_types("b", vec![code_struct(0, &["b", "Foo"])], vec![]);
    assert!(!super::tombstoned_leaves_of(&a).contains("Foo"));
    assert!(!super::tombstoned_leaves_of(&b).contains("Foo"));
    let mut facts = super::DuplicateLeafFacts::default();
    facts.absorb(super::DuplicateLeafFacts::discover(
        &a,
        &std::collections::HashSet::new(),
    ));
    facts.absorb(super::DuplicateLeafFacts::discover(
        &b,
        &std::collections::HashSet::new(),
    ));
    let cross = facts.tombstoned_leaves();
    assert!(
        cross.contains("Foo"),
        "the same stripped key from two crates tombstones the leaf"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &a, &cross).as_deref(),
        Some("a::Foo")
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &b, &cross).as_deref(),
        Some("b::Foo")
    );
}

/// `alloc::layout::Layout` is Opaque in Charon, so it never enters
/// `struct_fields`. `typeobject::Layout` is a workspace struct. The
/// merged verdict still withdraws `Layout` so Result instantiations
/// intern two Ok classdefs.
#[test]
fn opaque_and_struct_same_leaf_are_tombstoned_across_llbc() {
    let alloc = llbc_with_types(
        "alloc",
        vec![opaque_type_decl(0, &["alloc", "layout", "Layout"])],
        vec![],
    );
    let obj = llbc_with_types(
        "pyre_object",
        vec![code_struct(0, &["pyre_object", "typeobject", "Layout"])],
        vec![],
    );
    assert!(
        !super::tombstoned_leaves_of(&alloc).contains("Layout"),
        "each artefact alone has one Layout"
    );
    assert!(!super::tombstoned_leaves_of(&obj).contains("Layout"));
    let mut facts = super::DuplicateLeafFacts::default();
    facts.absorb(super::DuplicateLeafFacts::discover(
        &alloc,
        &std::collections::HashSet::new(),
    ));
    facts.absorb(super::DuplicateLeafFacts::discover(
        &obj,
        &std::collections::HashSet::new(),
    ));
    let cross = facts.tombstoned_leaves();
    assert!(
        cross.contains("Layout"),
        "Opaque layout::Layout beside typeobject::Layout withdraws the leaf"
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &alloc, &cross).as_deref(),
        Some("layout::Layout")
    );
    assert_eq!(
        super::adt_node_class_root_with(&adt_node(0), &obj, &cross).as_deref(),
        Some("typeobject::Layout")
    );
}

/// The interpreter LLBC repeats `pyre_object::pyobject::PyType` as a
/// foreign struct. That copy is not a second definition and must not
/// withdraw the `PyType` leaf (or intern keys move off
/// `pyobject::PyType` and `set_struct_storage` panics).
#[test]
fn foreign_struct_copy_does_not_tombstone_pytype() {
    let obj = llbc_with_types(
        "pyre_object",
        vec![code_struct(0, &["pyre_object", "pyobject", "PyType"])],
        vec![],
    );
    let interp = llbc_with_types(
        "pyre_interpreter",
        vec![code_struct(0, &["pyre_object", "pyobject", "PyType"])],
        vec![],
    );
    let mut facts = super::DuplicateLeafFacts::default();
    facts.absorb(super::DuplicateLeafFacts::discover(
        &obj,
        &std::collections::HashSet::new(),
    ));
    facts.absorb(super::DuplicateLeafFacts::discover(
        &interp,
        &std::collections::HashSet::new(),
    ));
    let cross = facts.tombstoned_leaves();
    assert!(
        !cross.contains("PyType"),
        "foreign PyType copy must not withdraw the leaf, got {cross:?}"
    );
    assert!(
        !cross.contains("pyobject::PyType"),
        "foreign PyType copy must not mark the stripped path cross-crate, got {cross:?}"
    );
}

/// A field typed by its decl's own type parameter has no register class
/// the decl can name, so its FORCE row is `Unknown` and nothing is forced:
/// `Range<Idx>::start` and `Option<T>::Some.0` take the annotation of the
/// values written to them.  A concretely typed field keeps its class.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn type_parameter_fields_force_no_attribute_real() {
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = crate::test_support::load_ullbc(path);
    let (_, _, _, _, _, attrs, _, _) = super::derive_program_metadata(
        llbc,
        &super::harvest_declared_gc_facts(llbc).gc_struct_ids(),
        super::no_immutable_fields(),
        &crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(std::slice::from_ref(llbc)),
        &[],
    );
    let field = |owner: &str, name: &str| {
        attrs
            .get(owner)
            .unwrap_or_else(|| panic!("{owner} registered"))
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, ty)| ty.clone())
            .unwrap_or_else(|| panic!("{owner}.{name} registered"))
    };
    assert_eq!(field("ops::range::Range", "start"), ValueType::Unknown);
    assert_eq!(field("ops::range::Range", "end"), ValueType::Unknown);
    assert_eq!(field("option::Option::Some", "__pos_0"), ValueType::Unknown);
    assert_eq!(
        field("pyframe::FrameBlock", "valuestackdepth"),
        ValueType::Unsigned
    );
}

fn option_of_string_value_llbc() -> Llbc {
    llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            code_struct(1, &["rustpython_wtf8", "Wtf8"]),
            code_struct(2, &["alloc", "string", "String"]),
            code_struct(3, &["rustpython_wtf8", "Wtf8Buf"]),
            code_struct(4, &["std", "path", "PathBuf"]),
            code_struct(5, &["std", "ffi", "OsString"]),
            code_struct(6, &["other", "PathBuf"]),
        ],
        vec![],
    )
}

fn with_fixture_lowering(llbc: &Llbc, f: impl FnOnce(&super::Lowering<'_>)) {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let body: super::Unstructured = serde_json::from_value(serde_json::json!({
        "locals": {"arg_count": 0, "locals": []}, "body": [], "span": span
    }))
    .unwrap();
    let dont_look_inside = std::collections::HashSet::new();
    let accum = super::AccumulatorFacts::build(llbc, &body);
    let tombstoned = std::collections::HashSet::new();
    let gc_struct_ids_llbc = super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    let lowering = super::Lowering::new(
        llbc,
        "fixture".into(),
        &body,
        crate::HostStaticAddrs::default(),
        &[],
        None,
        &dont_look_inside,
        &tombstoned,
        &gc_struct_ids_llbc,
        super::no_immutable_fields(),
        &std::collections::HashSet::<String>::new(),
        &accum,
        &super::RootStackAnalyzer::new(llbc, &super::RootStackState::new(llbc)),
        false,
        false,
    )
    .unwrap();
    f(&lowering);
}

/// `deref_cast_root` reinterprets only a one-word handle (`Box<T>`,
/// `repr(transparent)` one-pointer wrappers). A three-field
/// non-transparent handle is `PtrRepr.rtype_getattr` field access.
#[test]
fn deref_cast_root_is_one_word_handle_only() {
    let payload_adt = serde_json::json!({
        "Adt": {"id": 0, "generics": {"types": []}}
    });
    let three_field_adt = serde_json::json!({
        "Adt": {"id": 1, "generics": {"types": []}}
    });
    let word_adt = serde_json::json!({
        "Adt": {"id": 2, "generics": {"types": []}}
    });
    let opaque_word_adt = serde_json::json!({
        "Adt": {"id": 3, "generics": {"types": []}}
    });
    let box_adt = serde_json::json!({
        "Adt": {
            "id": 99,
            "builtin": "Box",
            "generics": {"types": [payload_adt.clone()]}
        }
    });
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let payload = struct_decl(
        0,
        ident_path(&["fixture", "Payload"]),
        serde_json::json!([{
            "name": "n",
            "ty": i64_ty.clone(),
            "attr_info": null
        }]),
        false,
    );
    let mut payload = payload;
    payload["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 8,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0]}],
            "repr": {"transparent": false}
        }
    }]);
    let three_field = struct_decl(
        1,
        ident_path(&["fixture", "Handle"]),
        serde_json::json!([
            {"name": "value", "ty": i64_ty.clone(), "attr_info": null},
            {"name": "slot", "ty": i64_ty.clone(), "attr_info": null},
            {"name": "_not_send", "ty": i64_ty, "attr_info": null}
        ]),
        false,
    );
    let mut three_field = three_field;
    three_field["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 24,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0, 8, 16]}],
            "repr": {"repr_algo": "Rust", "transparent": false}
        }
    }]);
    let word = opaque_handle_decl(2, ident_path(&["fixture", "Word"]), 8);
    let mut opaque_word = opaque_handle_decl(3, ident_path(&["fixture", "Guard"]), 8);
    opaque_word["layout"][0]["value"]["repr"]["transparent"] = serde_json::json!(false);
    let dest_ty = TyRef::Other(serde_json::json!({
        "Ref": [null, payload_adt.clone(), "Shared"]
    }));
    let deref_fun = |def_id: u64, impl_id: u64, handle: serde_json::Value| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": {
                "name": [
                    {"Ident": ["fixture", 0]},
                    {"Ident": ["Handle", 0]},
                    {"Impl": {
                        "Ty": {"skip_binder": {"Value": [0, handle.clone()]}},
                        "Trait": impl_id
                    }},
                    {"Ident": ["deref", 0]}
                ],
                "span": fixture_span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": false
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [{"Ref": [null, handle, "Shared"]}],
                "output": {"Ref": [null, payload_adt.clone(), "Shared"]}
            },
            "body": "Opaque"
        })
    };
    let deref_trait_impl = |fn_id: u64, handle: serde_json::Value| {
        serde_json::json!({
            "impl_trait": {"id": 0, "generics": {"types": [handle]}},
            "generics": {"types": [], "const_generics": [], "trait_refs": []},
            "methods": [{
                "kind": {"TraitMethod": [0, 0]},
                "skip_binder": {"id": fn_id}
            }]
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "target_information": [{
                "key": "fixture-target",
                "value": {
                    "target_pointer_size": 8,
                    "is_little_endian": true
                }
            }],
            "type_decls": [payload, three_field, word, opaque_word],
            "fun_decls": [
                deref_fun(0, 0, three_field_adt.clone()),
                deref_fun(1, 1, box_adt.clone()),
                deref_fun(2, 2, word_adt.clone()),
                deref_fun(3, 3, opaque_word_adt.clone())
            ],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": {
                    "name": [
                        {"Ident": ["core", 0]},
                        {"Ident": ["ops", 0]},
                        {"Ident": ["deref", 0]},
                        {"Ident": ["Deref", 0]}
                    ],
                    "span": fixture_span(),
                    "source_text": null,
                    "attr_info": {
                        "attributes": [],
                        "inline": null,
                        "rename": null,
                        "public": true
                    },
                    "is_local": false
                },
                "methods": [{"skip_binder": {"name": "deref"}}]
            }],
            "trait_impls": [
                deref_trait_impl(0, three_field_adt),
                deref_trait_impl(1, box_adt),
                deref_trait_impl(2, word_adt),
                deref_trait_impl(3, opaque_word_adt)
            ]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");
    let call = |id: u64| {
        serde_json::from_value::<RegularCall>(serde_json::json!({
            "kind": {"Fun": id},
            "generics": {}
        }))
        .expect("RegularCall parses")
    };
    with_fixture_lowering(&llbc, |lowering| {
        assert_eq!(
            lowering.deref_cast_root(&call(0), &dest_ty, None),
            None,
            "a three-field non-transparent handle is not a one-word reinterpret"
        );
        assert_eq!(
            lowering
                .deref_cast_root(&call(1), &dest_ty, None)
                .as_deref(),
            Some("Payload"),
            "Box<T> keeps the one-word cast"
        );
        assert_eq!(
            lowering
                .deref_cast_root(&call(2), &dest_ty, None)
                .as_deref(),
            Some("Payload"),
            "a transparent one-word handle keeps the one-word cast"
        );
        assert_eq!(
            lowering
                .deref_cast_root(&call(3), &dest_ty, None)
                .as_deref(),
            Some("Payload"),
            "an opaque one-word guard (MutexGuard-shaped) keeps the one-word cast"
        );
    });
}

#[test]
fn option_of_shared_str_paints_as_str_and_is_niche() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_str = TyRef::Other(option_adt_node(
        0,
        option_shared_ref(serde_json::json!({"Adt": {"builtin": "Str"}})),
    ));
    assert!(
        matches!(
            super::tyref_to_value_type(&option_str, &llbc),
            ValueType::Str
        ),
        "Option<&str> is a nullable SomeString"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_str),
            "Option<&str> is the nullable-string niche"
        );
    });
    let borrowed = TyRef::Other(option_shared_ref(option_adt_node(
        0,
        option_shared_ref(serde_json::json!({"Adt": {"builtin": "Str"}})),
    )));
    assert!(
        matches!(super::tyref_to_value_type(&borrowed, &llbc), ValueType::Str),
        "&Option<&str> is a nullable SomeString"
    );
}

#[test]
fn option_of_shared_wtf8_paints_as_str_and_is_niche() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_wtf8 = TyRef::Other(option_adt_node(0, option_shared_ref(adt_node(1))));
    assert!(
        matches!(
            super::tyref_to_value_type(&option_wtf8, &llbc),
            ValueType::Str
        ),
        "Option<&Wtf8> is a nullable SomeString"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_wtf8),
            "Option<&Wtf8> is the nullable-string niche"
        );
    });
}

#[test]
fn option_of_wtf8buf_paints_as_str_and_is_niche() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_wtf8buf = TyRef::Other(option_adt_node(0, adt_node(3)));
    assert!(
        matches!(
            super::tyref_to_value_type(&option_wtf8buf, &llbc),
            ValueType::Str
        ),
        "Option<Wtf8Buf> is a nullable SomeString"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_wtf8buf),
            "Option<Wtf8Buf> is the nullable-string niche"
        );
    });
}

#[test]
fn option_of_string_is_ref_paint_and_niche() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_string = TyRef::Other(option_adt_node(0, adt_node(2)));
    let string = TyRef::Other(adt_node(2));
    let option_ref_string = TyRef::Other(option_adt_node(0, option_shared_ref(adt_node(2))));
    let option_paint = super::tyref_to_value_type(&option_string, &llbc);
    let string_paint = super::tyref_to_value_type(&string, &llbc);
    assert!(
        !matches!(option_paint, ValueType::Str),
        "Option<String> is not a nullable SomeString"
    );
    assert!(
        matches!(string_paint, ValueType::Ref(_)),
        "a rust-heap String itself stays Ref"
    );
    assert_eq!(
        option_paint, string_paint,
        "Option<String> value paint is the payload String's Ref"
    );
    assert_eq!(
        tyref_to_attr_value_type(&option_string, &llbc),
        tyref_to_attr_value_type(&string, &llbc),
        "Option<String> attr paint is the payload String's Ref"
    );
    let tomb = super::no_tombstoned_leaves();
    let option_param = super::tyref_ref_param_class_root(&option_string, &llbc, tomb, None);
    let string_param = super::tyref_ref_param_class_root(&string, &llbc, tomb, None);
    assert_eq!(
        option_param, string_param,
        "Option<String> param class root is the payload String's class root"
    );
    assert!(
        option_param.is_some(),
        "String class root must be named, got {option_param:?}"
    );
    assert_eq!(
        super::tyref_input_class_root(&option_string, &llbc, tomb),
        super::tyref_input_class_root(&string, &llbc, tomb),
        "Option<String> input class root is the payload String's class root"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_string),
            "Option<String> is the rust-heap-string niche"
        );
        assert!(
            matches!(
                lowering.option_niche_null_cast(&option_string),
                Some((_, ValueType::Ref(_)))
            ),
            "Option<String> None is a null Ref"
        );
        assert!(
            !lowering.enum_payload_is_nullable_instance_ptr(&option_string),
            "Option<String> is SomeString (`rstr.py`), not a W_ROOT GC instance"
        );
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_ref_string),
            "Option<&String> is the rust-heap-string niche"
        );
        assert_eq!(
            super::tyref_to_value_type(&option_ref_string, &llbc),
            string_paint,
            "Option<&String> value paint is the payload String's Ref"
        );
        assert_eq!(
            super::tyref_ref_param_class_root(&option_ref_string, &llbc, tomb, None),
            string_param,
            "Option<&String> param class root is the payload String's class root"
        );
        let borrowed = TyRef::Other(option_shared_ref(option_adt_node(0, adt_node(2))));
        assert_eq!(
            super::tyref_to_value_type(&borrowed, &llbc),
            option_paint,
            "&Option<String> value paint is the payload String's Ref"
        );
        assert_eq!(
            tyref_to_attr_value_type(&borrowed, &llbc),
            tyref_to_attr_value_type(&option_string, &llbc),
            "&Option<String> attr paint is the payload String's Ref"
        );
        assert_eq!(
            super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
            string_param,
            "&Option<String> param class root is the payload String's class root"
        );
    });
}

/// `rpath.py` `_posix_rnormpath` annotates `PathBuf` as `s_Str0`. The
/// Option is the same rust-heap-string niche as `Option<String>`: `Some`
/// is the path string, `None` is a null Ref, and there is no
/// `__discriminant` aggregate. A lookalike `other::PathBuf` stays tagged.
#[test]
fn option_of_pathbuf_is_ref_paint_and_niche() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_pathbuf = TyRef::Other(option_adt_node(0, adt_node(4)));
    let pathbuf = TyRef::Other(adt_node(4));
    let option_os_string = TyRef::Other(option_adt_node(0, adt_node(5)));
    let option_lookalike = TyRef::Other(option_adt_node(0, adt_node(6)));
    let option_paint = super::tyref_to_value_type(&option_pathbuf, &llbc);
    let pathbuf_paint = super::tyref_to_value_type(&pathbuf, &llbc);
    assert!(
        !matches!(option_paint, ValueType::Str),
        "Option<PathBuf> is not a nullable SomeString paint"
    );
    assert!(
        matches!(pathbuf_paint, ValueType::Ref(_)),
        "PathBuf itself stays Ref like a rust-heap String"
    );
    assert_eq!(
        option_paint, pathbuf_paint,
        "Option<PathBuf> value paint is the payload PathBuf's Ref"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_pathbuf),
            "Option<PathBuf> is the rpath.py s_Str0 string niche"
        );
        assert!(
            matches!(
                lowering.option_niche_null_cast(&option_pathbuf),
                Some((_, ValueType::Ref(_)))
            ),
            "Option<PathBuf> None is a null Ref"
        );
        assert!(
            !lowering.enum_payload_is_nullable_instance_ptr(&option_pathbuf),
            "Option<PathBuf> is SomeString (`rstr.py` / `rpath.py` s_Str0), not a W_ROOT GC instance"
        );
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_os_string),
            "Option<OsString> is the rpath.py s_Str0 string niche"
        );
        assert!(
            !lowering.enum_payload_is_nullable_instance_ptr(&option_os_string),
            "Option<OsString> is SomeString, not a W_ROOT GC instance"
        );
        assert!(
            !lowering.tyref_is_niche_option_ptr(&option_lookalike),
            "Option<other::PathBuf> stays a tagged aggregate"
        );
    });
    let empty_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let empty_raw: std::collections::HashSet<String> = std::collections::HashSet::new();
    let stamp = super::tyref_niche_option_gc_payload_spelling(
        &option_pathbuf,
        &llbc,
        &empty_ids,
        &empty_raw,
    );
    assert!(
        stamp.as_deref().is_some_and(|s| s.contains("PathBuf")),
        "Option<PathBuf> intern stamp is Option<PathBuf>, got {stamp:?}"
    );
}

/// `<Option<String> as PartialEq>::eq` operands are `&Option<String>`.
/// The Option paints as the payload String's `Ref`, so a `BinOp eq`
/// would be instance identity (`rclass.py` `rtype_eq` → `ptr_eq`).
/// The operand is not admitted; the call stays on the ordinary path.
#[test]
fn option_of_string_eq_operand_is_not_string_binop() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_string = TyRef::Other(option_adt_node(0, adt_node(2)));
    let operand = TyRef::Other(option_shared_ref(option_adt_node(0, adt_node(2))));
    assert!(
        matches!(
            super::tyref_to_value_type(&option_string, &llbc),
            ValueType::Ref(_)
        ),
        "Option<String> paint is a Ref"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            !lowering.option_niche_eq_ne_operand(&operand),
            "Option<String> eq operand is not the string BinOp shortcut"
        );
    });
}

/// `<Option<&str> as PartialEq>::eq` operands are `&Option<&str>`.
/// The Option paints `Str`, so derived equality is string contents
/// (`rstr.py` `AbstractStringRepr.rtype_eq` → `ll_streq`).
#[test]
fn option_of_shared_str_eq_operand_is_string_binop() {
    use crate::model::ValueType;
    let llbc = option_of_string_value_llbc();
    let option_str = option_adt_node(
        0,
        option_shared_ref(serde_json::json!({"Adt": {"builtin": "Str"}})),
    );
    let option_ty = TyRef::Other(option_str.clone());
    let operand = TyRef::Other(option_shared_ref(option_str));
    assert!(
        matches!(
            super::tyref_to_value_type(&option_ty, &llbc),
            ValueType::Str
        ),
        "Option<&str> paint is Str"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.option_niche_eq_ne_operand(&operand),
            "Option<&str> eq operand is the string BinOp shortcut"
        );
    });
}

fn option_of_object_pointer_slice_llbc() -> Llbc {
    llbc_with_types(
        "fixture",
        vec![
            option_enum(0),
            code_struct(1, &["pyre_object", "pyobject", "PyObject"]),
        ],
        vec![],
    )
}

fn object_pointer_slice_node(kind: &str) -> serde_json::Value {
    let item = serde_json::json!({"RawPtr": [adt_node(1), "Mut"]});
    let slice = serde_json::json!({"Slice": [item, null]});
    serde_json::json!({"Ref": ["Erased", slice, kind]})
}

/// `Option<&[PyObjectRef]>` is the nullable object GcArray list.
/// `Discriminant` is a null test; the param seeds `SomeList` of
/// `PyObject` (`rlist.py` `FixedSizeListRepr`).
#[test]
fn option_of_object_pointer_slice_is_nullable_list() {
    use crate::model::ValueType;
    let llbc = option_of_object_pointer_slice_llbc();
    let option_ty = TyRef::Other(option_adt_node(0, object_pointer_slice_node("Shared")));
    let list_root = Some("[pyobject::PyObject]".to_string());
    assert_eq!(
        super::tyref_to_value_type(&option_ty, &llbc),
        ValueType::Ref(list_root.clone()),
        "Option<&[*mut PyObject]> paints as the nullable object list"
    );
    let tomb = super::no_tombstoned_leaves();
    assert_eq!(
        super::tyref_ref_param_class_root(&option_ty, &llbc, tomb, None),
        list_root,
        "Option<&[*mut PyObject]> param seeds SomeList of PyObject"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_ty),
            "Option<&[*mut PyObject]> is the nullable-list niche"
        );
        assert_eq!(
            lowering.option_niche_null_cast(&option_ty),
            Some((
                "[pyobject::PyObject]".to_string(),
                ValueType::Ref(Some("[pyobject::PyObject]".into()))
            )),
            "None is convert_const(None) on that ListRepr"
        );
    });
}

/// Charon's `Option<[PyObject]>` spelling of the same slice.
#[test]
fn option_of_bare_object_pointer_slice_is_nullable_list() {
    use crate::model::ValueType;
    let llbc = option_of_object_pointer_slice_llbc();
    let item = serde_json::json!({"RawPtr": [adt_node(1), "Mut"]});
    let option_ty = TyRef::Other(option_adt_node(
        0,
        serde_json::json!({"Slice": [item, null]}),
    ));
    assert_eq!(
        super::tyref_to_value_type(&option_ty, &llbc),
        ValueType::Ref(Some("[pyobject::PyObject]".into())),
        "Charon Option<[PyObject]> paints as the nullable object list"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_ty),
            "Charon Option<[PyObject]> is the nullable-list niche"
        );
    });
}

fn vec_struct(def_id: u64) -> serde_json::Value {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": [
                {"Ident": ["alloc", 0]},
                {"Ident": ["vec", 0]},
                {"Ident": ["Vec", 0]}
            ],
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": false
        },
        "kind": {"Struct": [{
            "name": "buf",
            "ty": {"TypeVar": {"Bound": [0, 0]}},
            "attr_info": null
        }]}
    })
}

fn vec_adt_node(def_id: u64, arg: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "Adt": {
            "id": def_id,
            "generics": {
                "regions": [],
                "types": [arg],
                "const_generics": [],
                "trait_refs": []
            }
        }
    })
}

/// `Option<Vec<Vec<u8>>>` is the nullable list. `Discriminant` is a
/// null test; the param seeds `SomeList` of `Vec<u8>`.
#[test]
fn option_of_vec_of_vec_u8_is_nullable_list() {
    use crate::model::ValueType;
    let llbc = llbc_with_types("fixture", vec![option_enum(0), vec_struct(1)], vec![]);
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let vec_vec_u8 = vec_adt_node(1, vec_adt_node(1, u8_ty));
    let option_ty = TyRef::Other(option_adt_node(0, vec_vec_u8.clone()));
    let list_root = Some("Vec<Vec<u8>>".to_string());
    assert_eq!(
        super::tyref_to_value_type(&option_ty, &llbc),
        ValueType::Ref(list_root.clone()),
        "Option<Vec<Vec<u8>>> paints as the nullable list"
    );
    let tomb = super::no_tombstoned_leaves();
    assert_eq!(
        super::tyref_ref_param_class_root(&option_ty, &llbc, tomb, None),
        list_root,
        "Option<Vec<Vec<u8>>> param seeds SomeList"
    );
    let borrowed = TyRef::Other(option_shared_ref(option_adt_node(0, vec_vec_u8)));
    assert_eq!(
        super::tyref_to_value_type(&borrowed, &llbc),
        ValueType::Ref(list_root.clone()),
        "&Option<Vec<Vec<u8>>> value paint is the nullable list"
    );
    assert_eq!(
        super::tyref_ref_param_class_root(&borrowed, &llbc, tomb, None),
        list_root,
        "&Option<Vec<Vec<u8>>> param seeds SomeList"
    );
    with_fixture_lowering(&llbc, |lowering| {
        assert!(
            lowering.tyref_is_niche_option_ptr(&option_ty),
            "Option<Vec<Vec<u8>>> is the nullable-list niche"
        );
        assert_eq!(
            lowering.option_niche_null_cast(&option_ty),
            Some((
                "Vec<Vec<u8>>".to_string(),
                ValueType::Ref(Some("Vec<Vec<u8>>".into()))
            )),
            "None is convert_const(None) on that ListRepr"
        );
    });
}

/// Mut object-pointer slices are the `{ptr, len}` pair at call sites
/// (`try_lower_pair_unwrap_or`). They are not the shared GcArray list
/// niche this conversion added.
#[test]
fn option_of_mut_object_pointer_slice_is_not_gcarray_list() {
    let llbc = option_of_object_pointer_slice_llbc();
    let option_ty = TyRef::Other(option_adt_node(0, object_pointer_slice_node("Mut")));
    assert!(
        !super::tyref_is_option_of_object_pointer_slice(&option_ty, &llbc),
        "mut object-pointer slice is not the shared GcArray Option niche"
    );
}

/// `None` of `Option<&mut H>` for a GC handle is the nullable MutRef cell
/// (`gc_mut_ref_input_stamp`), not payload `H`. `Option<&H>` and
/// `Option<*mut H>` keep the payload class
/// (`option_payload_instance_class_root`).
#[test]
fn option_gc_mut_ref_none_is_mut_ref_cell_null() {
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": path,
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let handle_ty = adt_node(0);
    let payload_ty = adt_node(1);
    let mut_ref = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Mut"]});
    let shared_ref = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Shared"]});
    let raw_ptr = serde_json::json!({"RawPtr": [handle_ty.clone(), "Mut"]});
    let option_mut = option_adt_node(2, mut_ref);
    let option_shared = option_adt_node(2, shared_ref);
    let option_raw = option_adt_node(2, raw_ptr);
    let unit = serde_json::json!({"Tuple": []});
    let layout = |transparent: bool, size: u64| {
        serde_json::json!([{
            "key": "fixture-target",
            "value": {
                "size": size,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"transparent": transparent}
            }
        }])
    };
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": name,
            "span": span,
            "ty": ty
        })
    };
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let takes = |id: u64, leaf: &str, opt_ty: &serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": meta(vec![ident("fixture"), ident(leaf)], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [opt_ty.clone()],
                "output": unit
            },
            "body": {"Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("opt"), opt_ty)
                    ]
                },
                "body": [{
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": "Return"}
                }]
            }}
        })
    };
    let caller = |id: u64, leaf: &str, callee: u64, opt_ty: &serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": meta(vec![ident("fixture"), ident(leaf)], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": unit
            },
            "body": {"Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 0,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("none"), opt_ty)
                    ]
                },
                "body": [
                    {
                        "statements": [{
                            "span": span,
                            "kind": {"Assign": [
                                place(1, opt_ty),
                                {"Aggregate": [
                                    {"Adt": [2, 0, null]},
                                    []
                                ]}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": callee},
                                                "generics": empty_generics()
                                            }
                                        },
                                        "args": [{"Move": place(1, opt_ty)}]
                                    },
                                    "dest": place(0, &unit),
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }}
        })
    };
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("fixture"), ident("Handle")], true),
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": {"RawPtr": [payload_ty, "Mut"]},
                        "attr_info": null
                    }]},
                    "layout": layout(true, 8)
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![ident("fixture"), ident("Payload")], true),
                    "kind": {"Struct": []},
                    "layout": layout(false, 0)
                },
                option_enum(2)
            ],
            "fun_decls": [
                caller(0, "call_mut_none", 1, &option_mut),
                takes(1, "takes_mut", &option_mut),
                caller(2, "call_shared_none", 3, &option_shared),
                takes(3, "takes_shared", &option_shared),
                caller(4, "call_raw_none", 5, &option_raw),
                takes(5, "takes_raw", &option_raw)
            ],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": [gctype_impl_for_adt(1)]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let option_mut_ty: TyRef = serde_json::from_value(option_mut.clone()).unwrap();
    let option_shared_ty: TyRef = serde_json::from_value(option_shared.clone()).unwrap();
    let option_raw_ty: TyRef = serde_json::from_value(option_raw.clone()).unwrap();
    let tomb = no_tombstoned_leaves();
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw = std::collections::HashSet::<String>::new();
    assert!(
        super::tyref_option_gc_mut_ref_cell_root(&option_mut_ty, &llbc, tomb, &gc, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "Option<&mut Handle> occupies a MutRef cell"
    );
    assert!(
        super::tyref_option_gc_mut_ref_cell_root(&option_shared_ty, &llbc, tomb, &gc, &raw)
            .is_none(),
        "Option<&Handle> is not a MutRef cell"
    );
    assert!(
        super::tyref_option_gc_mut_ref_cell_root(&option_raw_ty, &llbc, tomb, &gc, &raw).is_none(),
        "Option<*mut Handle> is not a MutRef cell"
    );
    with_fixture_lowering(&llbc, |lowering| {
        let mut_cast = lowering.option_niche_null_cast(&option_mut_ty);
        assert!(
            matches!(
                &mut_cast,
                Some((root, ValueType::Ref(Some(r))))
                    if r == root && root.starts_with("MutRef<") && root.contains("Handle")
            ),
            "None of Option<&mut Handle> is MutRef, got {mut_cast:?}"
        );
        let shared_cast = lowering.option_niche_null_cast(&option_shared_ty);
        assert!(
            matches!(
                &shared_cast,
                Some((root, ValueType::Ref(Some(r))))
                    if r == root && !root.starts_with("MutRef<") && root.contains("Handle")
            ),
            "None of Option<&Handle> is payload Handle, got {shared_cast:?}"
        );
        let raw_cast = lowering.option_niche_null_cast(&option_raw_ty);
        assert!(
            matches!(
                &raw_cast,
                Some((root, ValueType::Ref(Some(r))))
                    if r == root && !root.starts_with("MutRef<") && root.contains("Handle")
            ),
            "None of Option<*mut Handle> is payload Handle, got {raw_cast:?}"
        );
    });
    let cast_roots = |name: &str| {
        let graph = crate::test_support::lower_function(&llbc, name)
            .unwrap_or_else(|err| panic!("lower {name}: {err}"));
        graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .filter_map(|op| crate::model::cast_instance_root(&op.kind).map(str::to_string))
            .collect::<Vec<_>>()
    };
    let mut_roots = cast_roots("call_mut_none");
    assert!(
        mut_roots
            .iter()
            .any(|root| root.starts_with("MutRef<") && root.contains("Handle")),
        "call_mut_none lowers None as MutRef, got {mut_roots:?}"
    );
    assert!(
        mut_roots
            .iter()
            .all(|root| root.starts_with("MutRef<") || !root.contains("Handle")),
        "call_mut_none must not cast None to payload Handle, got {mut_roots:?}"
    );
    let shared_roots = cast_roots("call_shared_none");
    assert!(
        shared_roots
            .iter()
            .any(|root| !root.starts_with("MutRef<") && root.contains("Handle")),
        "call_shared_none lowers None as payload Handle, got {shared_roots:?}"
    );
    let raw_roots = cast_roots("call_raw_none");
    assert!(
        raw_roots
            .iter()
            .any(|root| !root.starts_with("MutRef<") && root.contains("Handle")),
        "call_raw_none lowers None as payload Handle, got {raw_roots:?}"
    );
    let takes_mut =
        crate::test_support::lower_function(&llbc, "takes_mut").expect("lower takes_mut");
    let opt_input = takes_mut
        .block(takes_mut.startblock)
        .operations
        .iter()
        .find(|op| matches!(&op.kind, OpKind::Input { name, .. } if name == "opt"))
        .expect("opt input");
    let OpKind::Input { ty, class_root, .. } = &opt_input.kind else {
        panic!("opt is an input");
    };
    let class_root = class_root
        .as_deref()
        .expect("Option<&mut Handle> Input carries a class_root");
    assert!(
        class_root.starts_with("Option<MutRef<"),
        "takes_mut Input is nullable MutRef, got {class_root}"
    );
    match ty {
        ValueType::Ref(Some(root)) => assert!(
            root.starts_with("MutRef<"),
            "takes_mut value root is MutRef, got {root}"
        ),
        other => panic!("takes_mut Input is Ref(MutRef), got {other:?}"),
    }
}

#[test]
fn mut_ref_cell_root_names_a_gc_pointer_only() {
    let llbc = fixture_llbc();
    let tomb = no_tombstoned_leaves();
    let gc = std::collections::HashSet::new();
    let i64 = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let i64_ty: TyRef = serde_json::from_value(i64.clone()).unwrap();
    assert!(
        super::mut_ref_cell_root_of(
            &i64_ty,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        )
        .is_none()
    );
    let raw_i64: TyRef =
        serde_json::from_value(serde_json::json!({"RawPtr": [i64, "Mut"]})).unwrap();
    assert!(
        super::mut_ref_cell_root_of(
            &raw_i64,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        )
        .is_none()
    );
    let obj = serde_json::json!({"Adt": {"id": 7}});
    let obj_ty: TyRef = serde_json::from_value(obj.clone()).unwrap();
    assert!(
        super::mut_ref_cell_root_of(
            &obj_ty,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        )
        .is_none()
    );
    let raw_obj: TyRef =
        serde_json::from_value(serde_json::json!({"RawPtr": [obj, "Mut"]})).unwrap();
    assert_eq!(
        super::mut_ref_cell_root_of(
            &raw_obj,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        )
        .as_deref(),
        Some("MutRef<*mut ??adt#7>")
    );
}

/// `&mut self` on a transparent GC handle is a FunctionPath cell call.
/// `&self` on the same type stays `CallTarget::Method`.
#[test]
fn mut_ref_gc_handle_method_is_function_path_cell() {
    use crate::model::{CallTarget, OpKind};

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": path,
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let adt = |id: u64| {
        serde_json::json!({
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
    };
    let payload_ty = adt(1);
    let handle_ty = adt(0);
    let mut_self = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Mut"]});
    let shared_self = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Shared"]});
    let unit = serde_json::json!({"Tuple": []});
    let layout = |transparent: bool| {
        serde_json::json!([{
            "key": "fixture-target",
            "value": {
                "size": 8,
                "align": 8,
                "variant_layouts": [{"field_offsets": [0]}],
                "repr": {"transparent": transparent}
            }
        }])
    };
    let field = |name: Option<&str>, ty: serde_json::Value| serde_json::json!({"name": name, "ty": ty, "attr_info": null});
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, handle_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": name,
            "span": span,
            "ty": ty
        })
    };
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let method_body = |self_ty: &serde_json::Value| {
        serde_json::json!({
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("self"), self_ty)
                    ]
                },
                "body": [{
                    "statements": [],
                    "kind": "Regular",
                    "terminator": {"span": span, "kind": "Return"}
                }]
            }
        })
    };
    let method_decl = |id: u64, leaf: &str, self_ty: &serde_json::Value| {
        serde_json::json!({
            "def_id": id,
            "item_meta": meta(vec![
                ident("fixture"),
                ident("Handle"),
                impl_seg.clone(),
                ident(leaf),
            ], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [self_ty.clone()],
                "output": unit
            },
            "body": method_body(self_ty)
        })
    };
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(vec![ident("fixture"), ident("caller")], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [handle_ty.clone()],
            "output": unit
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("h"), &handle_ty),
                        local(2, None, &mut_self),
                        local(3, None, &shared_self)
                    ]
                },
                "body": [
                    {
                        "statements": [{
                            "span": span,
                            "kind": {"Assign": [
                                place(2, &mut_self),
                                {"Ref": {
                                    "place": place(1, &handle_ty),
                                    "kind": "Mut",
                                    "ptr_metadata": null
                                }}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
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
                                        "args": [{"Move": place(2, &mut_self)}]}, "dest": place(0, &unit),
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [{
                            "span": span,
                            "kind": {"Assign": [
                                place(3, &shared_self),
                                {"Ref": {
                                    "place": place(1, &handle_ty),
                                    "kind": "Shared",
                                    "ptr_metadata": null
                                }}
                            ]}
                        }],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
                            "kind": {
                                "Call": {
                                    "call": {
                                        "func": {
                                            "Regular": {
                                                "kind": {"Fun": 2},
                                                "generics": {
                                                    "regions": [],
                                                    "types": [],
                                                    "const_generics": [],
                                                    "trait_refs": []
                                                }
                                            }
                                        },
                                        "args": [{"Move": place(3, &shared_self)}]}, "dest": place(0, &unit),
                                    "target": 3,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }
        }
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("fixture"), ident("Handle")], true),
                    "kind": {"Struct": [field(
                        None,
                        serde_json::json!({"RawPtr": [payload_ty, "Mut"]})
                    )]},
                    "layout": layout(true)
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![ident("fixture"), ident("Payload")], true),
                    "kind": {"Struct": []}
                }
            ],
            "fun_decls": [
                caller,
                method_decl(1, "set_flag", &mut_self),
                method_decl(2, "get_flag", &shared_self)
            ],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": [gctype_impl_for_adt(1)]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let handle: TyRef = serde_json::from_value(handle_ty.clone()).unwrap();
    let mut_ty: TyRef = serde_json::from_value(mut_self.clone()).unwrap();
    let tomb = no_tombstoned_leaves();
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    assert!(
        super::mut_ref_cell_root_of(
            &handle,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        )
        .is_some(),
        "Handle behind &mut is a cell pointee"
    );
    assert!(
        super::tyref_gc_mut_ref_cell_root(
            &mut_ty,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        )
        .is_some(),
        "&mut Handle is a cell parameter"
    );
    let graph = crate::test_support::lower_function(&llbc, "caller").expect("lower caller");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let mut_call = ops
        .iter()
        .find(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => super::fmt_path_ends_with(segments, &["Handle", "set_flag"]),
            _ => false,
        })
        .expect("set_flag is a FunctionPath");
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "set_flag"
        )),
        "set_flag must not route as Method"
    );
    let mut_idx = ops
        .iter()
        .position(|op| std::ptr::eq(*op, *mut_call))
        .expect("set_flag index");
    assert!(
        ops[..mut_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name.starts_with("MutRef<")
        )),
        "a MutRef constructor precedes set_flag"
    );
    assert!(
        ops[mut_idx + 1..].iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldRead { field, .. } if mut_ref_value_field(field)
        )),
        "a value FieldRead follows set_flag"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "get_flag"
        )),
        "get_flag stays CallTarget::Method"
    );
}

/// Result Err peel of a GC handle: the Place arm wraps the payload as
/// a MutRef cell before the `&mut self` call (`install_gc_mut_ref_call`).
#[test]
fn result_err_peel_wraps_gc_handle_as_mut_ref_cell() {
    use crate::model::{CallTarget, OpKind};

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": path,
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let adt = |id: u64| {
        serde_json::json!({
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
    };
    let payload_ty = adt(1);
    let handle_ty = adt(0);
    let mut_self = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Mut"]});
    let unit = serde_json::json!({"Tuple": []});
    let result_ty = result_adt_node(2, unit.clone(), handle_ty.clone());
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, handle_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": name,
            "span": span,
            "ty": ty
        })
    };
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let err_place = serde_json::json!({
        "kind": {"Projection": [place(1, &result_ty), {"Field": [1, 0]}]},
        "ty": handle_ty
    });
    let method_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(vec![
            ident("fixture"),
            ident("Handle"),
            impl_seg,
            ident("set_flag"),
        ], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [mut_self.clone()],
            "output": unit
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    local(0, None, &unit),
                    local(1, Some("self"), &mut_self)
                ]
            },
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(vec![ident("fixture"), ident("caller")], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [result_ty.clone()],
            "output": unit
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("res"), &result_ty)
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
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
                                        "args": [{"Move": err_place}]
                                    },
                                    "dest": place(0, &unit),
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }
        }
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("fixture"), ident("Handle")], true),
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": {"RawPtr": [payload_ty.clone(), "Mut"]},
                        "attr_info": null
                    }]},
                    "layout": [{
                        "key": "fixture-target",
                        "value": {
                            "size": 8,
                            "align": 8,
                            "variant_layouts": [{"field_offsets": [0]}],
                            "repr": {"transparent": true}
                        }
                    }]
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![ident("fixture"), ident("Payload")], true),
                    "kind": {"Struct": []}
                },
                result_enum(2)
            ],
            "fun_decls": [caller, method_decl],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": [gctype_impl_for_adt(1)]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let graph = crate::test_support::lower_function(&llbc, "caller").expect("lower caller");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let call_idx = ops
        .iter()
        .position(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => super::fmt_path_ends_with(segments, &["Handle", "set_flag"]),
            _ => false,
        })
        .expect("set_flag is a FunctionPath");
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name.starts_with("MutRef<")
        )),
        "Result Err peel emits a MutRef constructor before set_flag"
    );
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. } if mut_ref_value_field(field)
        )),
        "Result Err peel writes FieldWrite value on the cell"
    );
}

/// `&mut Result<P, H>` Err peel of a transparent one-field GC handle:
/// wrap copy-out writes the Err payload field (`Err.__pos_0`), the handle
/// word `expand_pop_roots` reloads. A leaf-copy through the handle's
/// inner field would getattr `Err.__pos_0.__pos_0`.
#[test]
fn result_err_peel_mut_ref_copy_out_writes_err_pos0() {
    use crate::model::{CallTarget, OpKind};

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": path,
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let adt = |id: u64| {
        serde_json::json!({
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
    };
    let payload_ty = adt(1);
    let handle_ty = adt(0);
    let mut_self = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Mut"]});
    let unit = serde_json::json!({"Tuple": []});
    let result_ty = result_adt_node(2, unit.clone(), handle_ty.clone());
    let mut_result = serde_json::json!({"Ref": ["Erased", result_ty.clone(), "Mut"]});
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, handle_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": name,
            "span": span,
            "ty": ty
        })
    };
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let deref_result = serde_json::json!({
        "kind": {"Projection": [place(1, &mut_result), "Deref"]},
        "ty": result_ty
    });
    let err_place = serde_json::json!({
        "kind": {"Projection": [deref_result, {"Field": [1, 0]}]},
        "ty": handle_ty
    });
    let method_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(vec![
            ident("fixture"),
            ident("Handle"),
            impl_seg,
            ident("set_flag"),
        ], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [mut_self.clone()],
            "output": unit
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    local(0, None, &unit),
                    local(1, Some("self"), &mut_self)
                ]
            },
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(vec![ident("fixture"), ident("caller")], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [mut_result.clone()],
            "output": unit
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("res"), &mut_result)
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
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
                                        "args": [{"Move": err_place}]
                                    },
                                    "dest": place(0, &unit),
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }
        }
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("fixture"), ident("Handle")], true),
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": {"RawPtr": [payload_ty.clone(), "Mut"]},
                        "attr_info": null
                    }]},
                    "layout": [{
                        "key": "fixture-target",
                        "value": {
                            "size": 8,
                            "align": 8,
                            "variant_layouts": [{"field_offsets": [0]}],
                            "repr": {"transparent": true}
                        }
                    }]
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![ident("fixture"), ident("Payload")], true),
                    "kind": {"Struct": []}
                },
                result_enum(2)
            ],
            "fun_decls": [caller, method_decl],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": [gctype_impl_for_adt(1)]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let graph = crate::test_support::lower_function(&llbc, "caller").expect("lower caller");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let call_idx = ops
        .iter()
        .position(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => super::fmt_path_ends_with(segments, &["Handle", "set_flag"]),
            _ => false,
        })
        .expect("set_flag is a FunctionPath");
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name.starts_with("MutRef<")
        )),
        "Err peel of &mut Result wraps a MutRef cell before set_flag; graph={}",
        graph.dump()
    );
    let copy_out: Vec<_> = ops[call_idx + 1..]
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite { field, .. } if !mut_ref_value_field(field) => {
                Some((field.owner_root.clone(), field.name.clone()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        copy_out.len(),
        1,
        "wrap copy-out is one FieldWrite of the Err payload; got {copy_out:?}; graph={}",
        graph.dump()
    );
    let (owner, name) = &copy_out[0];
    assert_eq!(
        name,
        "__pos_0",
        "copy-out writes Err.__pos_0, the handle word; got {owner:?}.{name}; graph={}",
        graph.dump()
    );
    assert!(
        owner
            .as_deref()
            .is_some_and(|owner| owner.ends_with("::Err")),
        "copy-out owner is the Err variant; got {owner:?}.{name}; graph={}",
        graph.dump()
    );
}

/// By-value GC handle auto-ref: WholeLocal wraps the payload as a
/// MutRef cell (`emit_gc_mut_ref_cell`).
#[test]
fn by_value_gc_handle_auto_ref_wraps_as_mut_ref_cell() {
    use crate::model::{CallTarget, OpKind};

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": path,
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let adt = |id: u64| {
        serde_json::json!({
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
    };
    let payload_ty = adt(1);
    let handle_ty = adt(0);
    let mut_self = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Mut"]});
    let unit = serde_json::json!({"Tuple": []});
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, handle_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": name,
            "span": span,
            "ty": ty
        })
    };
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let method_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(vec![
            ident("fixture"),
            ident("Handle"),
            impl_seg,
            ident("set_flag"),
        ], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [mut_self.clone()],
            "output": unit
        },
        "body": {"Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 1,
                "locals": [
                    local(0, None, &unit),
                    local(1, Some("self"), &mut_self)
                ]
            },
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span, "kind": "Return"}
            }]
        }}
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(vec![ident("fixture"), ident("caller")], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [handle_ty.clone()],
            "output": unit
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("h"), &handle_ty)
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
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
                                        "args": [{"Move": place(1, &handle_ty)}]
                                    },
                                    "dest": place(0, &unit),
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }
        }
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(vec![ident("fixture"), ident("Handle")], true),
                    "kind": {"Struct": [{
                        "name": null,
                        "ty": {"RawPtr": [payload_ty, "Mut"]},
                        "attr_info": null
                    }]},
                    "layout": [{
                        "key": "fixture-target",
                        "value": {
                            "size": 8,
                            "align": 8,
                            "variant_layouts": [{"field_offsets": [0]}],
                            "repr": {"transparent": true}
                        }
                    }]
                },
                {
                    "def_id": 1,
                    "item_meta": meta(vec![ident("fixture"), ident("Payload")], true),
                    "kind": {"Struct": []}
                }
            ],
            "fun_decls": [caller, method_decl],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": [gctype_impl_for_adt(1)]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let graph = crate::test_support::lower_function(&llbc, "caller").expect("lower caller");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let call_idx = ops
        .iter()
        .position(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => super::fmt_path_ends_with(segments, &["Handle", "set_flag"]),
            _ => false,
        })
        .expect("set_flag is a FunctionPath");
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name.starts_with("MutRef<")
        )),
        "by-value handle auto-ref emits a MutRef constructor before set_flag"
    );
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. } if mut_ref_value_field(field)
        )),
        "by-value handle auto-ref writes FieldWrite value on the cell"
    );
}

/// Imported Opaque handle + Opaque method FunDecl: wrap from the
/// signature (`gc_mut_ref_signature` does not skip a body-less
/// `&mut T` GC formal).
#[test]
fn opaque_gc_handle_auto_ref_wraps_as_mut_ref_cell() {
    use crate::model::{CallTarget, OpKind};

    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |path: Vec<serde_json::Value>, is_local: bool| {
        serde_json::json!({
            "name": path,
            "span": span,
            "source_text": null,
            "attr_info": {
                "attributes": [],
                "inline": null,
                "rename": null,
                "public": true
            },
            "is_local": is_local
        })
    };
    let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
    let handle_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let mut_self = serde_json::json!({"Ref": ["Erased", handle_ty.clone(), "Mut"]});
    let unit = serde_json::json!({"Tuple": []});
    let impl_seg = serde_json::json!({"Impl": {"Ty": {
        "skip_binder": {"Value": [0, handle_ty.clone()]},
        "kind": "InherentImplBlock"
    }}});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| {
        serde_json::json!({
            "index": index,
            "name": name,
            "span": span,
            "ty": ty
        })
    };
    let place = |index: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": index}, "ty": ty});
    let method_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(vec![
            ident("fixture"),
            ident("Handle"),
            impl_seg,
            ident("set_flag"),
        ], false),
        "signature": {
            "is_unsafe": false,
            "inputs": [mut_self.clone()],
            "output": unit
        },
        "body": "Opaque"
    });
    let caller = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(vec![ident("fixture"), ident("caller")], true),
        "signature": {
            "is_unsafe": false,
            "inputs": [handle_ty.clone()],
            "output": unit
        },
        "body": {
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        local(0, None, &unit),
                        local(1, Some("context"), &handle_ty)
                    ]
                },
                "body": [
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {
                            "span": span,
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
                                        "args": [{"Move": place(1, &handle_ty)}]
                                    },
                                    "dest": place(0, &unit),
                                    "target": 2,
                                    "on_unwind": 1
                                }
                            }
                        }
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "kind": "Regular",
                        "terminator": {"span": span, "kind": "Return"}
                    }
                ]
            }
        }
    });
    let handle_decl = opaque_handle_decl(0, ident_path(&["fixture", "Handle"]), 8);
    let payload_decl = struct_decl(
        1,
        ident_path(&["fixture", "Payload"]),
        empty_fields(),
        false,
    );
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "target_information": [{
                "key": "fixture-target",
                "value": {
                    "target_pointer_size": 8,
                    "is_little_endian": true
                }
            }],
            "type_decls": [handle_decl, payload_decl],
            "fun_decls": [caller, method_decl],
            "global_decls": [],
            "trait_decls": handle_trait_decls(),
            "trait_impls": [
                gctype_impl_for_adt(1),
                deref_impl(
                    1,
                    serde_json::json!({"Value": [0, adt_body(0)]}),
                    vec![serde_json::json!({"Value": [1, adt_body(1)]})],
                    vec![
                        serde_json::json!({"Value": [0, adt_body(0)]}),
                        serde_json::json!({"Value": [1, adt_body(1)]}),
                    ],
                )
            ]
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let handle: TyRef = serde_json::from_value(handle_ty.clone()).unwrap();
    let mut_ty: TyRef = serde_json::from_value(mut_self.clone()).unwrap();
    let tomb = no_tombstoned_leaves();
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let raw = std::collections::HashSet::<String>::new();
    assert!(
        super::mut_ref_cell_root_of(&handle, &llbc, tomb, &gc, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "Opaque Handle is a MutRef payload, ty={:?}",
        super::tyref_to_value_type(&handle, &llbc)
    );
    assert!(
        super::tyref_gc_mut_ref_cell_root(&mut_ty, &llbc, tomb, &gc, &raw)
            .as_deref()
            .is_some_and(|root| root.starts_with("MutRef<")),
        "&mut Opaque Handle occupies a MutRef cell"
    );
    let graph = crate::test_support::lower_function(&llbc, "caller").expect("lower caller");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let call_idx = ops
        .iter()
        .position(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => super::fmt_path_ends_with(segments, &["Handle", "set_flag"]),
            _ => false,
        })
        .expect("set_flag is a FunctionPath");
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, .. },
                ..
            } if name.starts_with("MutRef<")
        )),
        "Opaque handle auto-ref emits a MutRef constructor before set_flag, graph={}",
        graph.dump()
    );
    assert!(
        ops[..call_idx].iter().any(|op| matches!(
            &op.kind,
            OpKind::FieldWrite { field, .. } if mut_ref_value_field(field)
        )),
        "Opaque handle auto-ref writes FieldWrite value on the cell"
    );
}

/// A Float `to_string` receiver, including `&f64`, is excluded from
/// `UnaryOp("str")`. Int/Bool/Char and a non-float Ref are not.
#[test]
fn to_string_receiver_is_float_looks_through_refs() {
    use majit_charon_reader::ullbc::TyRef;
    let llbc = empty_llbc();
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let tomb = super::no_tombstoned_leaves();
    let is_float = |ty: &TyRef| {
        super::to_string_receiver_is_float(
            ty,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new(),
        )
    };
    let ty = |value: serde_json::Value| -> TyRef { serde_json::from_value(value).unwrap() };
    assert!(is_float(&ty(
        serde_json::json!({"Scalar": {"Float": "F64"}})
    )));
    assert!(is_float(&ty(serde_json::json!({
        "Ref": ["Erased", {"Scalar": {"Float": "F64"}}, "Shared"]
    }))));
    assert!(is_float(&ty(serde_json::json!({
        "Ref": [
            "Erased",
            {"Ref": ["Erased", {"Scalar": {"Float": "F64"}}, "Shared"]},
            "Shared"
        ]
    }))));
    assert!(!is_float(&ty(
        serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})
    )));
    assert!(!is_float(&ty(serde_json::json!({"Scalar": "Bool"}))));
    assert!(!is_float(&ty(serde_json::json!({"Scalar": "Char"}))));
    assert!(!is_float(&ty(serde_json::json!({
        "Ref": ["Erased", {"Scalar": {"Integer": {"Signed": "I64"}}}, "Shared"]
    }))));
}

#[test]
fn shared_ref_sized_gate_accepts_adt_rejects_slice() {
    // Minimal Llbc: `type_node_shared_ref_pointee` only walks the node it
    // is handed (no dedup ids in these inline fixtures), so an empty
    // translated section suffices.
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [], "fun_decls": [], "global_decls": [],
            "trait_decls": [], "trait_impls": [],
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses");

    // &[i64] — fat pointer, Slice pointee, no ADT def-id → rejected.
    let slice_ref = serde_json::json!({
        "Ref": ["'_", { "Slice": [{"Scalar": {"Integer": {"Signed": "I64"}}}, null] }, "Shared"]
    });
    let slice_pointee = super::type_node_shared_ref_pointee(&slice_ref, &llbc)
        .expect("shared-ref pointee extracted");
    assert!(
        super::adt_node_def_id(slice_pointee).is_none(),
        "a &[T] fat-pointer pointee must not pass the Sized ADT gate"
    );

    // &SomeAdt — thin pointer, nominal ADT pointee, def-id present → accepted.
    let adt_ref = serde_json::json!({
        "Ref": ["'_", { "Adt": { "id": 196 , "generics": {} } }, "Shared"]
    });
    let adt_pointee =
        super::type_node_shared_ref_pointee(&adt_ref, &llbc).expect("shared-ref pointee extracted");
    assert_eq!(
        super::adt_node_def_id(adt_pointee),
        Some(196),
        "a &SizedAdt pointee must pass the Sized ADT gate"
    );

    // &mut — the shared-ref helper must NOT match a Mut ref.
    let mut_ref = serde_json::json!({
        "Ref": ["'_", { "Adt": { "id": 196 , "generics": {} } }, "Mut"]
    });
    assert!(
        super::type_node_shared_ref_pointee(&mut_ref, &llbc).is_none(),
        "type_node_shared_ref_pointee must not match a &mut ref"
    );
}

/// A monomorphized copy's module is its generic item's: the
/// `Instantiated` segment after the leaf is not a path segment.
#[test]
fn fundecl_module_of_an_instance_ends_at_its_leaf() {
    let fd: majit_charon_reader::ullbc::FunDecl = serde_json::from_value(serde_json::json!({
        "def_id": 0,
        "item_meta": {
            "name": [
                {"Ident": ["c", 0]},
                {"Ident": ["m", 0]},
                {"Ident": ["f", 0]},
                {"Instantiated": {"params": {}, "kind": "Other", "skip_binder": {
                    "regions": [], "types": [{"Deduplicated": 0}],
                    "const_generics": [], "trait_refs": []}}}
            ],
            "span": {"Deduplicated": 0},
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        },
        "signature": {"is_unsafe": false, "inputs": [], "output": {"Deduplicated": 0}},
        "body": null
    }))
    .unwrap();
    assert_eq!(super::fundecl_module(&fd).as_deref(), Some("c::m"));
}

/// `func.__module__`: a free function's module ends at its leaf, a
/// method's at its impl segment, and the crate stays on the path.
#[test]
#[ignore]
fn fundecl_module_is_the_crate_qualified_defining_module() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-module.ullbc"
    );
    let llbc = crate::test_support::load_ullbc(path);
    let module_of = |name: &str| {
        let fd = llbc
            .iter_local_fns()
            .find(|fd| fd.item_meta.name_path() == name)
            .unwrap_or_else(|| panic!("no fun decl {name}"));
        super::fundecl_module(fd)
    };
    assert_eq!(
        module_of("pyre_module::module::unicodedata::interp_ucd::char_and_default").as_deref(),
        Some("pyre_module::module::unicodedata::interp_ucd")
    );
    let method = llbc
        .iter_local_fns()
        .find(|fd| {
            fd.item_meta
                .name_path()
                .starts_with("pyre_module::module::")
                && fd.item_meta.name.iter().any(|seg| seg.is_impl())
        })
        .expect("a method under pyre_module::module");
    let module = super::fundecl_module(method).expect("method module");
    let name_path = method.item_meta.name_path();
    assert!(
        name_path.starts_with(&format!("{module}::<")),
        "{name_path}: module {module} must end at the impl segment"
    );
}

/// `policy.py` `look_inside_graph` is false for `dont_look_inside` and
/// `elidable`, so the call is `jtransform.py` `handle_residual_call`
/// through `helper_return_to_i64` / `nullable_pointer_option`: one
/// pointer word. The declaration must be in the CallRegistry or the
/// caller's lift records "not registered in CallRegistry". Rust's own
/// `Option<*mut T>` is still two words (`ty_is_raw_address`), so the
/// `unsafe` collector keeps declining it, and a scalar pointee or a
/// `Vec` stays out of both.
#[test]
fn policy_opaque_nullable_pointer_option_is_a_residual_stub() {
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 1, "col": 0},
                "end": {"line": 1, "col": 1}
            }
        })
    };
    let item_meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|segment| serde_json::json!({"Ident": [segment, 0]})).collect::<Vec<_>>(),
            "span": span(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false},
            "is_local": true
        })
    };
    let generics = || {
        serde_json::json!({
            "regions": [],
            "types": [],
            "const_generics": [],
            "trait_refs": []
        })
    };
    let adt = |id: u64, args: Vec<serde_json::Value>| {
        let mut g = generics();
        g["types"] = serde_json::Value::Array(args);
        serde_json::json!({"Adt": {"id": id, "generics": g}})
    };
    let raw_mut = |pointee: serde_json::Value| serde_json::json!({"RawPtr": [pointee, "Mut"]});
    let option_of = |inner: serde_json::Value| adt(0, vec![inner]);
    let result_of = |ok: serde_json::Value, err: serde_json::Value| adt(4, vec![ok, err]);
    let nominal = adt(1, vec![]);
    let sized = adt(2, vec![]);
    let alias = adt(3, vec![]);
    let host_error = adt(5, vec![]);
    let vec_of_nominal = adt(6, vec![nominal.clone()]);
    let open_ended = adt(7, vec![]);
    let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});

    let fun = |id: u64, leaf: &str, output: serde_json::Value, is_unsafe: bool| {
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(&["fixture", "owner", leaf]),
            "signature": {"is_unsafe": is_unsafe, "inputs": [], "output": output},
            "body": "Opaque"
        })
    };
    let look_inside_false = |id: u64, leaf: &str| {
        let marker = format!("_jit_look_inside_{leaf}");
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(&["fixture", "owner", marker.as_str()]),
            "global_kind": "NamedConst",
            "value": [{"Bool": false}, {"Scalar": "Bool"}]
        })
    };
    let elidable_marker = |id: u64, leaf: &str| {
        let marker = format!("_elidable_function_{leaf}");
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(&["fixture", "owner", marker.as_str()]),
            "global_kind": "NamedConst"
        })
    };
    let ty =
        |id: u64, path: &[&str], kind: serde_json::Value, layout: Option<serde_json::Value>| {
            let mut decl = serde_json::json!({
                "def_id": id,
                "item_meta": item_meta(path),
                "kind": kind
            });
            if let Some(layout) = layout {
                decl["layout"] = layout;
            }
            decl
        };

    let direct = option_of(raw_mut(nominal.clone()));
    let sized_option = option_of(raw_mut(sized));
    let alias_option = option_of(alias);
    let nested = option_of(raw_mut(raw_mut(nominal.clone())));
    let result_option = result_of(direct.clone(), host_error.clone());
    let scalar_option = option_of(raw_mut(u8_ty));
    let unsized_option = option_of(raw_mut(open_ended));
    let result_scalar = result_of(scalar_option.clone(), host_error);

    let funs = vec![
        fun(0, "word_result", i64_ty, false),
        fun(1, "direct_nullable", direct, true),
        fun(2, "sized_nullable", sized_option, false),
        fun(3, "alias_nullable", alias_option, false),
        fun(4, "nested_nullable", nested, false),
        fun(5, "result_nullable", result_option, false),
        fun(6, "elidable_nullable", option_of(raw_mut(nominal)), false),
        fun(7, "scalar_option", scalar_option, true),
        fun(8, "vec_return", vec_of_nominal, false),
        fun(9, "unsized_option", unsized_option, false),
        fun(10, "result_scalar", result_scalar, false),
    ];
    let globals = vec![
        look_inside_false(0, "word_result"),
        look_inside_false(1, "direct_nullable"),
        look_inside_false(2, "sized_nullable"),
        look_inside_false(3, "alias_nullable"),
        look_inside_false(4, "nested_nullable"),
        look_inside_false(5, "result_nullable"),
        elidable_marker(6, "elidable_nullable"),
        look_inside_false(7, "scalar_option"),
        look_inside_false(8, "vec_return"),
        look_inside_false(9, "unsized_option"),
        look_inside_false(10, "result_scalar"),
    ];
    let types = vec![
        ty(
            0,
            &["core", "option", "Option"],
            serde_json::json!("Opaque"),
            None,
        ),
        // A `Vec` field keeps the pointee off the raw-storage path
        // (`adt_path_is_gc_container`), so `Option<PointerAlias>` stays
        // the `ref` token instead of the raw-address `i64`.
        ty(
            1,
            &["fixture", "Nominal"],
            serde_json::json!({"Struct": [{
                "name": "items",
                "ty": adt(6, vec![serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}})]),
                "attr_info": null
            }]}),
            None,
        ),
        ty(
            2,
            &["fixture", "Sized"],
            serde_json::json!({"Struct": []}),
            Some(serde_json::json!([{"key": "any", "value": {"size": 8}}])),
        ),
        ty(
            3,
            &["fixture", "PointerAlias"],
            serde_json::json!({"Alias": {"RawPtr": [adt(1, vec![]), "Mut"]}}),
            None,
        ),
        ty(
            4,
            &["core", "result", "Result"],
            serde_json::json!("Opaque"),
            None,
        ),
        ty(
            5,
            &["fixture", "HostError"],
            serde_json::json!({"Struct": []}),
            None,
        ),
        ty(
            6,
            &["alloc", "vec", "Vec"],
            serde_json::json!({"Struct": []}),
            None,
        ),
        ty(
            7,
            &["fixture", "Unsized"],
            serde_json::json!({"Struct": []}),
            Some(serde_json::json!([{"key": "any", "value": {}}])),
        ),
    ];
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": types,
            "fun_decls": funs,
            "global_decls": globals,
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("nullable-option fixture");
    let gc_struct_ids = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let carrier = crate::ErrorCarrierSpec {
        carrier_path: "fixture::HostError",
        carrier_class: "",
        carrier_wrappers: &[],
        to_exc_object: None,
        from_exc_object: None,
    };
    let policy = super::collect_policy_opaque_fn_stubs_from_llbc(
        &llbc,
        &gc_struct_ids,
        carrier,
        &std::collections::HashSet::<String>::new(),
    );
    let unsafe_stubs = super::collect_unsafe_fn_stubs_from_llbc(
        &llbc,
        &gc_struct_ids,
        carrier,
        &std::collections::HashSet::<String>::new(),
    );
    fn leaf_of<'a>(
        specs: &'a [crate::translator::rtyper::cutover::ResidualStubSpec],
        leaf: &str,
    ) -> Option<&'a crate::translator::rtyper::cutover::ResidualStubSpec> {
        specs.iter().find(|(segments, _, _, _, _)| {
            segments.len() == 3
                && segments[0] == "fixture"
                && segments[1] == "owner"
                && segments[2] == leaf
        })
    }
    for leaf in [
        "word_result",
        "direct_nullable",
        "sized_nullable",
        "alias_nullable",
        "nested_nullable",
        "result_nullable",
        "elidable_nullable",
    ] {
        let row = leaf_of(&policy, leaf).unwrap_or_else(|| panic!("{leaf} must be registered"));
        // `sized_nullable` is `Option<*mut storage-raw Struct>`: overlay
        // paint is int-bank SomePtr (`history.py` `getkind` of
        // `Ptr(Struct Raw)`), so the residual token is `i64`. A Vec
        // field keeps Nominal off that path (`adt_path_is_gc_container`).
        let token = match leaf {
            "word_result" | "sized_nullable" => "i64",
            _ => "ref",
        };
        assert_eq!(row.2.as_deref(), Some(token), "{leaf}");
    }
    for leaf in [
        "scalar_option",
        "vec_return",
        "unsized_option",
        "result_scalar",
    ] {
        assert!(
            leaf_of(&policy, leaf).is_none(),
            "{leaf} is not a one-word nullable pointer"
        );
    }
    assert!(
        leaf_of(&unsafe_stubs, "direct_nullable").is_none(),
        "an unsafe fn keeps Rust's two-word Option<*mut T> return"
    );
    assert!(leaf_of(&unsafe_stubs, "scalar_option").is_none());
}

/// The elidable half of the same carrier, on the real declarations.
/// `interp_exceptions::lookup_exc_class_for_kind` is `#[elidable]` and
/// reads a process-global slot the MIR driver cannot lower, so it has no
/// `SemanticFunction` at all — exactly the callee whose absent registry
/// key failed `typedef::type`'s lift.
#[test]
#[ignore]
fn elidable_declaration_keeps_a_functiondesc_without_a_lowered_body() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    let llbc = crate::test_support::load_ullbc(path);
    let gc_struct_ids = super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    let specs = super::collect_policy_opaque_fn_stubs_from_llbc(
        llbc,
        &gc_struct_ids,
        crate::ErrorCarrierSpec::default(),
        &std::collections::HashSet::<String>::new(),
    );
    let expected = [
        "pyre_object",
        "interp_exceptions",
        "lookup_exc_class_for_kind",
    ];
    let (_, signature, token, _, _) = specs
        .iter()
        .find(|(segments, _, _, _, _)| {
            segments
                .iter()
                .map(String::as_str)
                .eq(expected.iter().copied())
        })
        .expect("an elidable declaration must have a residual stub");
    assert_eq!(signature.argnames.len(), 1);
    assert_eq!(
        token.as_deref(),
        Some("ref"),
        "a residual *mut registered-GC-struct return is the ref token; the class root is on the graph"
    );
}

#[test]
#[ignore]
fn getset_direct_residual_keeps_a_functiondesc_without_a_jitcode_body() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = crate::test_support::load_ullbc(path);
    let gc_struct_ids = super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    let specs = super::collect_policy_opaque_fn_stubs_from_llbc(
        llbc,
        &gc_struct_ids,
        crate::ErrorCarrierSpec::default(),
        &std::collections::HashSet::<String>::new(),
    );
    let expected = ["pyre_interpreter", "typedef", "call_getset_fget_direct"];
    let (_, signature, token, _, _) = specs
        .iter()
        .find(|(segments, _, _, _, _)| {
            segments
                .iter()
                .map(String::as_str)
                .eq(expected.iter().copied())
        })
        .expect("dont_look_inside getset trampoline must have a residual stub");

    assert_eq!(signature.argnames.len(), 3);
    assert!(
        crate::translator::rtyper::cutover::residual_return_shell(token.as_deref()).is_some(),
        "the declared Result<PyObjectRef, PyError> return must be annotatable"
    );
}

/// `metaclass_call_override` is `#[dont_look_inside]` and returns
/// `Option<PyObjectRef>`. The original Rust ABI is two words; the
/// residual target is the helper trampoline (`None` = null). The
/// policy-opaque stub must therefore be registered so
/// `_check_len_result` → `compare` → `call_function_impl_result` can
/// annotate past the call.
#[test]
#[ignore]
fn dont_look_inside_option_objectptr_residual_is_nullable_gcref() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = crate::test_support::load_ullbc(path);
    let gc_struct_ids = super::harvest_declared_gc_facts(llbc).gc_struct_ids();
    let specs = super::collect_policy_opaque_fn_stubs_from_llbc(
        llbc,
        &gc_struct_ids,
        crate::ErrorCarrierSpec::default(),
        &std::collections::HashSet::<String>::new(),
    );
    let expected = ["pyre_interpreter", "call", "metaclass_call_override"];
    let (_, signature, token, _, _) = specs
        .iter()
        .find(|(segments, _, _, _, _)| {
            segments
                .iter()
                .map(String::as_str)
                .eq(expected.iter().copied())
        })
        .expect("dont_look_inside Option<PyObjectRef> residual must have a stub");
    assert_eq!(signature.argnames.len(), 1);
    assert!(
        crate::translator::rtyper::cutover::residual_return_shell(token.as_deref()).is_some(),
        "Option<PyObjectRef> residual must shell as one nullable ref word"
    );
}

/// A `GcType` struct whose field 0 is a fieldless enum does not mark
/// that enum Gc. A `GcType` struct whose field 0 is a plain struct
/// does (`lltype.py` `Struct._first_struct`).
#[test]
fn declared_gc_field0_closure_follows_struct_not_enum() {
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |segs: &[&str]| {
        serde_json::json!({
            "name": segs.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let adt = |id: u64| {
        serde_json::json!({
            "Value": [id, {"Adt": {"id": id, "generics": {
                "regions": [], "types": [], "const_generics": [], "trait_refs": []
            }}}]
        })
    };
    let e_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "E"]),
        "kind": {"Enum": [
            {"name": "A", "fields": [], "discriminant": 0, "attr_info": null},
            {"name": "B", "fields": [], "discriminant": 1, "attr_info": null}
        ]}
    });
    let s_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "S"]),
        "kind": {"Struct": [{"name": "head", "ty": adt(0)}]}
    });
    let p_decl = serde_json::json!({
        "def_id": 2,
        "item_meta": meta(&["fixture", "P"]),
        "kind": {"Struct": []}
    });
    let t_decl = serde_json::json!({
        "def_id": 3,
        "item_meta": meta(&["fixture", "T"]),
        "kind": {"Struct": [{"name": "head", "ty": adt(2)}]}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [e_decl, s_decl, p_decl, t_decl],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["majit_gc", "header", "GcType"])
            }],
            "trait_impls": [
                {"impl_trait": {"id": 0, "generics": {"types": [adt(1)]}}},
                {"impl_trait": {"id": 0, "generics": {"types": [adt(3)]}}}
            ]
        }
    });
    let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
        .expect("fixture Llbc parses");
    let kinds = super::declared_gckind_by_name(std::slice::from_ref(&llbc));
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let e_sid = super::struct_id_for_type_decl(llbc.type_by_id(0).expect("E decl"));
    let p_sid = super::struct_id_for_type_decl(llbc.type_by_id(2).expect("P decl"));
    assert_eq!(kinds.get("E"), Some(&GcKind::Raw));
    assert!(
        !gc.contains(&e_sid),
        "fieldless enum E is not a Struct._first_struct parent"
    );
    assert_eq!(kinds.get("P"), Some(&GcKind::Gc));
    assert!(gc.contains(&p_sid), "plain struct P is field 0 of GcType T");
    assert_eq!(kinds.get("S"), Some(&GcKind::Gc));
    assert_eq!(kinds.get("T"), Some(&GcKind::Gc));
}

/// A `GcType` struct whose field 0 is a by-value one-integer newtype
/// does not mark that newtype Gc. `history.getkind` of the FIELD is the
/// inner integer (`descr.py get_field_descr`).
#[test]
fn declared_gc_field0_closure_skips_integer_newtype() {
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |segs: &[&str]| {
        serde_json::json!({
            "name": segs.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let adt = |id: u64| {
        serde_json::json!({
            "Value": [id, {"Adt": {"id": id, "generics": {
                "regions": [], "types": [], "const_generics": [], "trait_refs": []
            }}}]
        })
    };
    let v_decl = serde_json::json!({
        "def_id": 0,
        "item_meta": meta(&["fixture", "Serial"]),
        "kind": {"Struct": [{
            "name": null,
            "ty": {"Scalar": {"Integer": {"Unsigned": "U64"}}},
            "attr_info": null
        }]}
    });
    let m_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "Owner"]),
        "kind": {"Struct": [{"name": "version", "ty": adt(0)}]}
    });
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [v_decl, m_decl],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["majit_gc", "header", "GcType"])
            }],
            "trait_impls": [
                {"impl_trait": {"id": 0, "generics": {"types": [adt(1)]}}}
            ]
        }
    });
    let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
        .expect("fixture Llbc parses");
    let kinds = super::declared_gckind_by_name(std::slice::from_ref(&llbc));
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let v_sid = super::struct_id_for_type_decl(llbc.type_by_id(0).expect("Serial decl"));
    assert_eq!(kinds.get("Serial"), Some(&GcKind::Raw));
    assert!(
        !gc.contains(&v_sid),
        "one-integer newtype is not a Struct._first_struct parent"
    );
    assert_eq!(kinds.get("Owner"), Some(&GcKind::Gc));
    let serial_ty = serde_json::from_value::<TyRef>(adt(0)).expect("Serial TyRef");
    assert_eq!(
        super::tyref_to_value_type(&serial_ty, &llbc),
        ValueType::Unsigned
    );
    assert_eq!(
        super::tyref_to_field_layout_string(
            &serial_ty,
            &llbc,
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        "u64"
    );
}

/// A transparent handle wrapping an alias of `*mut Header` is a GC
/// pointer when `Header` is field 0 of a `GcType` struct.
/// `history.getkind` of that pointer is ref.
#[test]
fn transparent_handle_over_gctype_header_pointer_is_ref() {
    use crate::model::ValueType;
    let span = serde_json::json!({"data": {
        "file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}
    }});
    let meta = |segs: &[&str]| {
        serde_json::json!({
            "name": segs.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    };
    let adt = |id: u64| {
        serde_json::json!({
            "Adt": {"id": id, "generics": {
                "regions": [], "types": [], "const_generics": [], "trait_refs": []
            }}
        })
    };
    let raw_mut = |id: u64| serde_json::json!({"RawPtr": [adt(id), "Mut"]});
    let header = serde_json::json!({
        "def_id": 1,
        "item_meta": meta(&["fixture", "Header"]),
        "kind": {"Struct": [
            {"name": "ob_type", "ty": {"RawPtr": [adt(1), "Const"]}},
            {"name": "w_class", "ty": raw_mut(1)}
        ]}
    });
    let handle_object = serde_json::json!({
        "def_id": 2,
        "item_meta": meta(&["fixture", "HandleObject"]),
        "kind": {"Struct": [
            {"name": "ob_header", "ty": {"Value": [1, adt(1)]}}
        ]}
    });
    let header_ref = serde_json::json!({
        "def_id": 3,
        "item_meta": meta(&["fixture", "HeaderRef"]),
        "kind": {"Alias": raw_mut(1)}
    });
    let mut handle = serde_json::json!({
        "def_id": 4,
        "item_meta": meta(&["fixture", "Handle"]),
        "kind": {"Struct": [
            {"name": null, "ty": {"Value": [3, adt(3)]}}
        ]}
    });
    handle["layout"] = serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 8,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0]}],
            "repr": {"repr_algo": "Rust", "transparent": true}
        }
    }]);
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [
                serde_json::json!({"def_id": 0, "item_meta": meta(&["unused", "Pad"]), "kind": {"Struct": []}}),
                header,
                handle_object,
                header_ref,
                handle
            ],
            "fun_decls": [],
            "global_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["majit_gc", "header", "GcType"])
            }],
            "trait_impls": [{
                "impl_trait": {
                    "id": 0,
                    "generics": {"types": [{"Value": [2, adt(2)]}]}
                }
            }]
        }
    });
    let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
        .expect("fixture Llbc parses");
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let header_sid = super::struct_id_for_type_decl(llbc.type_by_id(1).expect("Header decl"));
    let object_sid = super::struct_id_for_type_decl(llbc.type_by_id(2).expect("HandleObject decl"));
    assert!(gc.contains(&object_sid), "GcType impl seeds HandleObject");
    assert!(
        gc.contains(&header_sid),
        "field-0 of HandleObject is Header"
    );
    let tomb = super::no_tombstoned_leaves();
    let raw_ptr = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [9100, raw_mut(1)]
    }))
    .expect("raw ptr TyRef");
    assert!(
        !super::ty_is_raw_address(
            &raw_ptr,
            &llbc,
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        "*mut Header is Ptr(GC)"
    );
    assert!(matches!(
        super::tyref_to_value_type_with(
            &raw_ptr,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Ref(_)
    ));
    let alias = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [3, adt(3)]
    }))
    .expect("alias TyRef");
    assert!(
        !super::ty_is_raw_address(
            &alias,
            &llbc,
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        "HeaderRef alias peels to *mut Header"
    );
    assert!(matches!(
        super::tyref_to_value_type_with(
            &alias,
            &llbc,
            tomb,
            &gc,
            &std::collections::HashSet::<String>::new()
        ),
        ValueType::Ref(_)
    ));
    let handle = serde_json::from_value::<super::TyRef>(serde_json::json!({
        "Value": [4, adt(4)]
    }))
    .expect("handle TyRef");
    assert!(!super::ty_is_raw_address(
        &handle,
        &llbc,
        &gc,
        &std::collections::HashSet::<String>::new()
    ));
    assert!(
        matches!(
            super::tyref_to_value_type_with(
                &handle,
                &llbc,
                tomb,
                &gc,
                &std::collections::HashSet::<String>::new()
            ),
            ValueType::Ref(_)
        ),
        "transparent handle over the GC pointer is Ref, not Int"
    );
}

#[test]
fn registered_gc_ptr_gcrefopaque_has_no_class_root() {
    // `GCREFOpaque` has no class root (`llmemory.GCREF`). A `PyObjectRef`
    // residual stays `"ref"` + class root.
    let llbc = kwargs_unerase_cast_llbc(true);
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let empty = std::collections::HashSet::<String>::new();
    let fd = llbc.fn_by_id(0).expect("unerase");
    let input = fd.signature.inputs.first().expect("dstorage");
    assert_eq!(
        super::registered_gc_ptr_class_root_with(
            input,
            &llbc,
            super::no_tombstoned_leaves(),
            &gc,
            &empty,
        ),
        None
    );
    assert_eq!(
        dont_look_inside_return_token(
            input,
            &llbc,
            crate::ErrorCarrierSpec::default(),
            &gc,
            &empty,
        )
        .as_deref(),
        Some(crate::translator::rtyper::cutover::GCREF_RETURN_TYPE)
    );
    assert_eq!(
        super::dont_look_inside_return_class_root(
            input,
            &llbc,
            crate::ErrorCarrierSpec::default(),
            super::no_tombstoned_leaves(),
            &gc,
            &empty,
        ),
        None
    );
}

/// `llmemory.py` `GCREF = lltype.Ptr(GCREFOpaque)`: exactly one pointer
/// onto `GCREFOpaque`. A pointer to a GCREF cell is a different type.
#[test]
fn tyref_is_gcref_spelling_is_exactly_one_pointer_onto_gcrefopaque() {
    let llbc = kwargs_unerase_cast_llbc(true);
    let gc = super::harvest_declared_gc_facts(&llbc).gc_struct_ids();
    let empty = std::collections::HashSet::<String>::new();
    let empty_g = empty_generics();
    let adt = |id: u64| {
        serde_json::json!({
            "Adt": {"id": id, "builtin": null, "generics": empty_g.clone()}
        })
    };
    let ty = |v: serde_json::Value| {
        serde_json::from_value::<TyRef>(serde_json::json!({ "Value": [0, v] }))
            .expect("fixture TyRef")
    };
    let token = |ty: &TyRef| {
        dont_look_inside_return_token(ty, &llbc, crate::ErrorCarrierSpec::default(), &gc, &empty)
    };
    let gcref_alias = ty(adt(3));
    let mut_gcrefopaque = ty(serde_json::json!({"RawPtr": [adt(0), "Mut"]}));
    let ref_gcrefopaque = ty(serde_json::json!({"Ref": ["Erased", adt(0), "Mut"]}));
    let mut_gcref = ty(serde_json::json!({"RawPtr": [adt(3), "Mut"]}));
    let ref_gcref = ty(serde_json::json!({"Ref": ["Erased", adt(3), "Mut"]}));
    let mut_mut_gcrefopaque = ty(serde_json::json!({
        "RawPtr": [{"RawPtr": [adt(0), "Mut"]}, "Mut"]
    }));

    assert!(
        tyref_is_gcref_spelling(&gcref_alias, &llbc),
        "bare GCREF alias"
    );
    assert_eq!(token(&gcref_alias).as_deref(), Some("gcref"));
    assert!(
        tyref_is_gcref_spelling(&mut_gcrefopaque, &llbc),
        "*mut GCREFOpaque"
    );
    assert_eq!(token(&mut_gcrefopaque).as_deref(), Some("gcref"));
    assert!(
        tyref_is_gcref_spelling(&ref_gcrefopaque, &llbc),
        "&GCREFOpaque"
    );
    assert_eq!(token(&ref_gcrefopaque).as_deref(), Some("gcref"));

    assert!(
        !tyref_is_gcref_spelling(&mut_gcref, &llbc),
        "*mut GCREF is a pointer to a GCREF cell"
    );
    let mut_gcref_token = token(&mut_gcref);
    assert_ne!(mut_gcref_token.as_deref(), Some("gcref"));
    assert!(
        !tyref_is_gcref_spelling(&ref_gcref, &llbc),
        "&GCREF is a borrow of a GCREF cell"
    );
    assert!(
        !tyref_is_gcref_spelling(&mut_mut_gcrefopaque, &llbc),
        "*mut *mut GCREFOpaque is two pointer levels"
    );
    let mut_mut_token = token(&mut_mut_gcrefopaque);
    assert_ne!(mut_mut_token.as_deref(), Some("gcref"));
    // Slot-pointer residual token (`dont_look_inside_return_token`).
    assert_eq!(
        mut_gcref_token.as_deref(),
        Some("ref"),
        "*mut GCREF slot token"
    );
    assert_eq!(
        mut_mut_token.as_deref(),
        Some("ref"),
        "*mut *mut GCREFOpaque slot token"
    );
}

/// `heaptracker.all_fielddescrs` and `_reccopy` walk the whole finite by-value
/// structure. A Rust struct cannot contain itself by value, so twenty nested
/// levels still yield every leaf.
#[test]
fn collect_by_value_struct_leaves_walks_the_whole_finite_nesting() {
    let field = |name: &str, ty: serde_json::Value| serde_json::json!({"name": name, "ty": ty, "attr_info": null});
    let mut decls = vec![type_decl_struct(
        0,
        &["fixture", "N0"],
        vec![field("x", i64_ty())],
    )];
    for k in 1..=20u64 {
        let name = format!("N{k}");
        decls.push(type_decl_struct(
            k,
            &["fixture", name.as_str()],
            vec![field("inner", named_adt(k - 1, vec![]))],
        ));
    }
    let llbc = window_llbc(decls, vec![]);
    let outer = llbc.type_by_id(20).expect("N20 decl");
    let outer_ty = TyRef::Other(named_adt(20, vec![]));
    let leaves = super::collect_by_value_struct_leaves(
        outer,
        &outer_ty,
        &llbc,
        super::no_tombstoned_leaves(),
        &std::collections::HashSet::new(),
        &std::collections::HashSet::new(),
    );
    assert_eq!(leaves.len(), 1, "one scalar leaf at the innermost struct");
    let dotted = format!("{}{}", leaves[0].super_path, leaves[0].field.name);
    let expected = format!("{}x", "inner.".repeat(20));
    assert_eq!(dotted, expected);
}

#[test]
fn foreign_byte_pointer_stub_keeps_the_c_address_bank() {
    let mut funs = Vec::new();
    for (id, signed, external) in [(0, false, true), (1, true, true), (2, false, false)] {
        let byte = if signed {
            serde_json::json!({"Scalar": {"Integer": {"Signed": "I8"}}})
        } else {
            serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})
        };
        funs.push(serde_json::json!({
            "def_id": id,
            "item_meta": {
                "name": [{"Ident": ["fixture", 0]}, {"Ident": [format!("f{id}"), 0]}],
                "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}},
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false},
                "is_local": true
            },
            "signature": {"is_unsafe": true, "inputs": [], "output": {"RawPtr": [byte, "Mut"]}},
            "body": if external { serde_json::json!({"Extern": {}}) } else { serde_json::json!("Opaque") }
        }));
    }
    let llbc = Llbc::from_slice(
        serde_json::json!({
            "charon_version": "0.1.201", "has_errors": false,
            "translated": {"crate_name": "fixture", "type_decls": [], "fun_decls": funs,
                "global_decls": [], "trait_decls": [], "trait_impls": []}
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();
    let specs = super::collect_unsafe_fn_stubs_from_llbc(
        &llbc,
        &std::collections::HashSet::new(),
        crate::ErrorCarrierSpec::default(),
        &std::collections::HashSet::new(),
    );
    let token = |leaf: &str| {
        specs
            .iter()
            .find(|(path, _, _, _, _)| path.last().is_some_and(|p| p == leaf))
            .unwrap()
            .2
            .as_deref()
    };
    assert_eq!(token("f0"), Some("i64"));
    assert_eq!(token("f1"), Some("i64"));
    assert_eq!(
        token("f2"),
        Some("ref"),
        "Rust's erased managed-byte pointer is unchanged"
    );
}

fn field_row_visited() -> std::collections::HashSet<String> {
    std::collections::HashSet::new()
}

fn note_class_identity(reg: &mut crate::front::StructFieldRegistry, spelling: &str) {
    let id = majit_ir::descr::StructId::from_canonical(spelling);
    reg.class_identity.insert(
        spelling.to_string(),
        Some(crate::front::semantic::ClassIdentity {
            id,
            spelling: spelling.to_string(),
        }),
    );
}

fn slot_block_registry(value_ty: &str) -> crate::front::StructFieldRegistry {
    let mut reg = crate::front::StructFieldRegistry::default();
    reg.fields.insert(
        "storage::SlotBlock".to_string(),
        vec![
            ("length".to_string(), "usize".to_string()),
            ("items".to_string(), "[storage::Slot; 0]".to_string()),
        ],
    );
    reg.fields.insert(
        "storage::Slot".to_string(),
        vec![
            ("f_valid".to_string(), "bool".to_string()),
            ("value".to_string(), value_ty.to_string()),
        ],
    );
    reg.note_declared_gc_name("storage::SlotBlock");
    reg
}

#[test]
fn field_row_classed_pointer_is_objectptr() {
    use crate::translator::rtyper::rclass::OBJECTPTR;
    let mut reg = crate::front::StructFieldRegistry::default();
    note_class_identity(&mut reg, "owner::Inst");
    let got = super::field_row_lltype(&reg, "*mut owner::Inst", &mut field_row_visited())
        .expect("classed pointer field");
    assert_eq!(got, OBJECTPTR.clone());
}

#[test]
fn field_row_unknown_pointer_declines() {
    let reg = crate::front::StructFieldRegistry::default();
    assert!(super::field_row_lltype(&reg, "*mut owner::Inst", &mut field_row_visited()).is_none());
    assert!(
        super::field_row_lltype(&reg, "*const owner::Inst", &mut field_row_visited()).is_none()
    );
    assert!(super::field_row_lltype(&reg, "&mut owner::Inst", &mut field_row_visited()).is_none());
}

#[test]
fn field_row_fat_pointer_declines() {
    let reg = crate::front::StructFieldRegistry::default();
    assert!(super::field_row_lltype(&reg, "&[u8]", &mut field_row_visited()).is_none());
    assert!(super::field_row_lltype(&reg, "&dyn Trait", &mut field_row_visited()).is_none());
    assert!(super::field_row_lltype(&reg, "&str", &mut field_row_visited()).is_none());
    assert!(super::field_row_lltype(&reg, "*mut [u8]", &mut field_row_visited()).is_none());
}

#[test]
fn field_row_scalar_pointer_is_address() {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    let reg = crate::front::StructFieldRegistry::default();
    let got = super::field_row_lltype(&reg, "*mut u8", &mut field_row_visited())
        .expect("scalar pointer field");
    assert_eq!(got, LowLevelType::Address);
}

#[test]
fn field_row_declared_gc_pointer_is_gcref() {
    use crate::translator::rtyper::lltypesystem::lltype::GCREF;
    let mut reg = crate::front::StructFieldRegistry::default();
    reg.note_declared_gc_name("owner::GcAdt");
    let got = super::field_row_lltype(&reg, "*mut owner::GcAdt", &mut field_row_visited())
        .expect("declared-GC pointer field");
    assert_eq!(got, (*GCREF).clone());
}

#[test]
fn field_row_str_family_pointer_is_strptr_ahead_of_declared_gc_and_classed() {
    use crate::translator::rtyper::lltypesystem::rstr::STRPTR;
    let mut reg = crate::front::StructFieldRegistry::default();
    reg.note_declared_gc_name("unicodeobject::Utf8Str");
    note_class_identity(&mut reg, "unicodeobject::Utf8Str");
    let got = super::field_row_lltype(
        &reg,
        "*mut unicodeobject::Utf8Str",
        &mut field_row_visited(),
    )
    .expect("str-family pointer field");
    assert_eq!(got, *STRPTR);
    let got = super::field_row_lltype(
        &reg,
        "*const bytesobject::BytesBlock",
        &mut field_row_visited(),
    )
    .expect("BytesBlock pointer field");
    assert_eq!(got, *STRPTR);
}

#[test]
fn field_row_raw_word_pointer_is_ptr_struct() {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    let mut reg = crate::front::StructFieldRegistry::default();
    reg.fields.insert(
        "host::RawWord".to_string(),
        vec![("save_point".to_string(), "usize".to_string())],
    );
    reg.raw_word_owners.insert("host::RawWord".to_string());
    let got = super::field_row_lltype(&reg, "*mut host::RawWord", &mut field_row_visited())
        .expect("raw-word pointer field");
    match got {
        LowLevelType::Ptr(p) => match p.TO {
            PtrTarget::Struct(_) => {}
            other => panic!("raw-word pointer must be Ptr(Struct), got {other:?}"),
        },
        other => panic!("raw-word pointer must be Ptr, got {other:?}"),
    }
}

#[test]
fn struct_item_gcarray_classed_pointer_field_interns_objectptr() {
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    use crate::translator::rtyper::rclass::OBJECTPTR;

    let mut reg = slot_block_registry("*mut owner::Inst");
    note_class_identity(&mut reg, "owner::Inst");
    let s = super::struct_item_flexible_gcarray_block_annotation(
        &reg,
        "storage::SlotBlock",
        "[storage::Slot; 0]",
    )
    .expect("classed pointer ITEM field intern");
    let SomeValue::Ptr(ptr) = s else {
        panic!("must intern SomePtr(GcArray), got {s:?}")
    };
    let PtrTarget::Array(arr) = &ptr.ll_ptrtype.TO else {
        panic!("must intern Ptr(GcArray), got {:?}", ptr.ll_ptrtype.TO)
    };
    let LowLevelType::Struct(st) = &arr.OF else {
        panic!("ITEM must be STRUCT, got {:?}", arr.OF)
    };
    assert_eq!(
        st.getattr_field_type("value"),
        Some(OBJECTPTR.clone()),
        "classed pointer field is the instance pointer"
    );
}

#[test]
fn struct_item_gcarray_str_family_pointer_field_interns_strptr() {
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    use crate::translator::rtyper::lltypesystem::rstr::STRPTR;

    let mut reg = slot_block_registry("*mut unicodeobject::Utf8Str");
    reg.note_declared_gc_name("unicodeobject::Utf8Str");
    note_class_identity(&mut reg, "unicodeobject::Utf8Str");
    let s = super::struct_item_flexible_gcarray_block_annotation(
        &reg,
        "storage::SlotBlock",
        "[storage::Slot; 0]",
    )
    .expect("str-family pointer ITEM field intern");
    let SomeValue::Ptr(ptr) = s else {
        panic!("must intern SomePtr(GcArray), got {s:?}")
    };
    let PtrTarget::Array(arr) = &ptr.ll_ptrtype.TO else {
        panic!("must intern Ptr(GcArray), got {:?}", ptr.ll_ptrtype.TO)
    };
    let LowLevelType::Struct(st) = &arr.OF else {
        panic!("ITEM must be STRUCT, got {:?}", arr.OF)
    };
    assert_eq!(
        st.getattr_field_type("value"),
        Some(STRPTR.clone()),
        "str-family pointer field is STRPTR, not GCREF or OBJECTPTR"
    );
}

#[test]
fn struct_item_gcarray_unknown_pointer_field_declines() {
    let reg = slot_block_registry("*mut owner::Inst");
    assert!(
        super::struct_item_flexible_gcarray_block_annotation(
            &reg,
            "storage::SlotBlock",
            "[storage::Slot; 0]",
        )
        .is_none(),
        "undecidable pointer ITEM field must decline"
    );
}
