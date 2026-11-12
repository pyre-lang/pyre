use super::{
    DecodedConst, bitflags_trait_empty_const, decode_literal, is_class_pytype_assoc_const,
    primitive_float_const, scalar_to_const_value, static_key_segments,
};

use crate::flowspace::model::ConstValue;

use crate::model::OpKind;

use majit_charon_reader::{Llbc, ullbc::NameSeg};

use crate::front::mir::tests::{
    const_artifact, empty_generics, empty_llbc, fixture_llbc, init_fun, item_meta_json,
    llbc_with_types, llbc_with_types_funs_globals, named_const_global, nullary_calls_ending,
    option_enum, reader_fun, scalar_ty, span_json, u32_ty, unit_fn_ptr, usize_ty,
};

/// `Isize`/`Usize` truncate to the target word (`lltype.py` `cast_primitive`).
#[test]
fn const_cast_int_with_word_truncates_isize_usize_to_target_width() {
    let usize_ty = serde_json::json!({"Integer": {"Unsigned": "Usize"}});
    let isize_ty = serde_json::json!({"Integer": {"Signed": "Isize"}});
    assert_eq!(
        super::const_cast_int_with_word(0x1_0000_0001, &usize_ty, 8),
        Some(0x1_0000_0001)
    );
    assert_eq!(
        super::const_cast_int_with_word(0xffff_ffff, &isize_ty, 8),
        Some(0xffff_ffff)
    );
    assert_eq!(
        super::const_cast_int_with_word(0x1_0000_0001, &usize_ty, 4),
        Some(1)
    );
    assert_eq!(
        super::const_cast_int_with_word(0xffff_ffff, &isize_ty, 4),
        Some(-1)
    );
}

#[test]
fn pytype_identity_lane_selects_the_trait_item_before_the_impl_owner() {
    let trait_path = "pyre_object::lltype::PyreClassPyTypeOf";
    assert!(is_class_pytype_assoc_const(
        "pyre_object::functional::<Impl>::PYTYPE",
        trait_path,
    ));
    assert!(!is_class_pytype_assoc_const(
        "pyre_object::functional::<Impl>::DESCRIPTOR",
        trait_path,
    ));
    assert!(!is_class_pytype_assoc_const(
        "pyre_object::functional::<Impl>::PYNAME",
        trait_path,
    ));
    assert!(!is_class_pytype_assoc_const(
        "other_crate::<Impl>::PYTYPE",
        "other_crate::lltype::PyreClassPyTypeOf",
    ));
}

#[test]
fn decode_char_literal_is_its_int_code_point() {
    let llbc = empty_llbc();
    for (lit, code) in [(">", 0x3e_u32), ("\u{0}", 0), ("\u{10ffff}", 0x10ffff)] {
        let json = serde_json::json!({ "Char": lit });
        assert!(
            matches!(decode_literal(&json), Ok(DecodedConst::UniChar(n)) if n == code),
            "char literal {lit:?} decodes to UniChar({code})",
        );
        // SwitchInt arms rtype through `AbstractUniCharRepr.convert_const`.
        assert!(
            matches!(
                scalar_to_const_value(&json, &llbc),
                Some(ConstValue::UniStr(ref s)) if s == lit
            ),
            "char SwitchInt arm {lit:?} is a length-1 UniStr",
        );
        let scalar = serde_json::json!({ "Scalar": { "Char": ["Char", code.to_string()] } });
        assert!(
            matches!(
                scalar_to_const_value(&scalar, &llbc),
                Some(ConstValue::UniStr(ref s)) if s == lit
            ),
            "Scalar Char code point {code} is a length-1 UniStr",
        );
    }
    assert!(
        decode_literal(&serde_json::json!({ "Str": ">" }))
            .is_ok_and(|c| matches!(c, DecodedConst::Str(ref s) if s == ">"))
    );
}

#[test]
fn decode_scalar_i128_and_u128_preserves_full_width() {
    let signed = serde_json::json!({
        "Scalar": {
            "Signed": ["I128", i128::MIN.to_string()]
        }
    });
    assert!(matches!(
        decode_literal(&signed).expect("decode I128"),
        DecodedConst::Int128(value) if value == i128::MIN
    ));

    let unsigned = serde_json::json!({
        "Scalar": {
            "Unsigned": ["U128", u128::MAX.to_string()]
        }
    });
    assert!(matches!(
        decode_literal(&unsigned).expect("decode U128"),
        DecodedConst::UInt128(value) if value == u128::MAX
    ));

    let word = serde_json::json!({
        "Scalar": {
            "Unsigned": ["Usize", "624"]
        }
    });
    assert!(matches!(
        decode_literal(&word).expect("decode usize"),
        DecodedConst::UInt(624)
    ));
}

#[test]
fn named_const_harvest_scopes_defining_crate_folds() {
    use super::OpKind;
    let mut folds = std::collections::HashMap::new();
    folds.insert(
        "pyre_object::intobject::W_INT_USER_GC_TYPE_ID".to_string(),
        OpKind::ConstUInt(185),
    );
    let _guard = super::push_named_const_folds(folds);
    assert!(matches!(
        super::named_const_fold_for_path("pyre_object::intobject::W_INT_USER_GC_TYPE_ID"),
        Some(OpKind::ConstUInt(185))
    ));
    assert!(super::named_const_fold_for_path("missing::CONST").is_none());
    drop(_guard);
    assert!(
        super::named_const_fold_for_path("pyre_object::intobject::W_INT_USER_GC_TYPE_ID").is_none()
    );
}

#[test]
fn named_const_merge_restarts_when_a_crate_repeats() {
    use super::OpKind;
    let one = |path: &str, v: u64| {
        let mut m = std::collections::HashMap::new();
        m.insert(path.to_string(), OpKind::ConstUInt(v));
        m
    };
    // A fresh table for this test, restored when the guard drops.
    let _guard = super::push_named_const_folds(std::collections::HashMap::new());
    super::merge_named_const_folds("pyre_object", one("pyre_object::A", 1));
    super::merge_named_const_folds("pyre_interpreter", one("pyre_interpreter::B", 2));
    assert!(super::named_const_fold_for_path("pyre_object::A").is_some());
    assert!(super::named_const_fold_for_path("pyre_interpreter::B").is_some());
    // `pyre_object` again means a second invocation started: the
    // earlier invocation's entries must not survive into it.
    super::merge_named_const_folds("pyre_object", one("pyre_object::A", 1));
    assert!(super::named_const_fold_for_path("pyre_object::A").is_some());
    assert!(super::named_const_fold_for_path("pyre_interpreter::B").is_none());
}

#[test]
fn const_eval_binop_declines_width_and_sign_ambiguous_folds() {
    use super::{ConstLit, const_eval_binop};
    use serde_json::json;

    // Wrapping Add/Sub/Mul: the i64 carrier has no operand width, so
    // `255u8 + 1` (wraps to 0, not 256) must residualize.
    assert!(
        const_eval_binop(
            &json!({ "Add": "Wrap" }),
            ConstLit::Int(255),
            ConstLit::Int(1)
        )
        .is_none()
    );
    assert!(
        const_eval_binop(
            &json!({ "Sub": "Wrap" }),
            ConstLit::Int(0),
            ConstLit::Int(1)
        )
        .is_none()
    );
    assert!(
        const_eval_binop(
            &json!({ "Mul": "Wrap" }),
            ConstLit::Int(16),
            ConstLit::Int(16)
        )
        .is_none()
    );
    // Non-wrapping (panic-checked) forms still fold: a valid const
    // initializer never overflows, so checked_* is exact.
    assert!(matches!(
        const_eval_binop(
            &json!({ "Add": "Panic" }),
            ConstLit::Int(255),
            ConstLit::Int(1)
        ),
        Some(ConstLit::Int(256))
    ));

    // Right shift with a negative carrier is signedness-ambiguous
    // (`u64::MAX >> 1` collapsed to `-1 >> 1`) — decline.
    assert!(
        const_eval_binop(
            &json!({ "Shr": "Wrap" }),
            ConstLit::Int(-1),
            ConstLit::Int(1)
        )
        .is_none()
    );
    // A non-negative left operand shifts identically signed/unsigned.
    assert!(matches!(
        const_eval_binop(
            &json!({ "Shr": "Wrap" }),
            ConstLit::Int(0x7fff_ffff),
            ConstLit::Int(1)
        ),
        Some(ConstLit::Int(0x3fff_ffff))
    ));

    // Census fold targets survive: SMALL_MAX = `(1i64<<62)-1` builds
    // from a Shl and a SubChecked; SMALL_MIN negates the same Shl.
    assert!(matches!(
        const_eval_binop(&json!({ "Shl": "Panic" }), ConstLit::Int(1), ConstLit::Int(62)),
        Some(ConstLit::Int(v)) if v == 1i64 << 62
    ));
    assert!(matches!(
        const_eval_binop(&json!("SubChecked"), ConstLit::Int(1i64 << 62), ConstLit::Int(1)),
        Some(ConstLit::Checked(v, false)) if v == (1i64 << 62) - 1
    ));
    assert!(matches!(
        const_eval_binop(&json!("AddChecked"), ConstLit::Int(1i64 << 62), ConstLit::Int(1)),
        Some(ConstLit::Checked(v, false)) if v == (1i64 << 62) + 1
    ));
}

/// `pub const MEMBER_DIRECT_FLAG: u32 = 1 << 31` (typedef.rs) records a
/// mixed-signedness pair, because an unsuffixed shift amount defaults to
/// `i32` while the shifted value is `u32`.  Charon plants the
/// shift-overflow assert (`31i32 as u32 < 32u32`) ahead of it, which is a
/// mixed pair too, so both halves have to fold or the const falls through
/// to a residual accessor call no registry can bind.
#[test]
fn const_eval_binop_folds_mixed_signedness_shifts_and_comparisons() {
    use super::{ConstLit, const_eval_binop};
    use serde_json::json;

    // The shift itself: `1u32 << 31i32`.
    assert!(matches!(
        const_eval_binop(
            &json!({ "Shl": "Wrap" }),
            ConstLit::UInt(1),
            ConstLit::Int(31)
        ),
        Some(ConstLit::UInt(v)) if v == 1u64 << 31
    ));
    // The overflow assert's comparison: `31u32 < 32u32` reached with the
    // cast's signedness dropped into the `Int` carrier.
    assert!(matches!(
        const_eval_binop(&json!("Lt"), ConstLit::Int(31), ConstLit::UInt(32)),
        Some(ConstLit::Bool(true))
    ));
    assert!(matches!(
        const_eval_binop(&json!("Ge"), ConstLit::UInt(32), ConstLit::Int(31)),
        Some(ConstLit::Bool(true))
    ));

    // A negative signed side makes the two carriers disagree on the
    // value — stay declined, as the `Shr` sign guard already does.
    assert!(
        const_eval_binop(
            &json!({ "Shl": "Wrap" }),
            ConstLit::UInt(1),
            ConstLit::Int(-1)
        )
        .is_none()
    );
    assert!(const_eval_binop(&json!("Lt"), ConstLit::Int(-1), ConstLit::UInt(0)).is_none());

    // Mixed arithmetic stays declined: the pair leaves the result *type*
    // undetermined, and Rust needs an explicit cast to write it at all.
    assert!(
        const_eval_binop(
            &json!({ "Add": "Panic" }),
            ConstLit::UInt(1),
            ConstLit::Int(1)
        )
        .is_none()
    );
}

#[test]
fn const_narrow_to_target_truncates_to_the_destination_width() {
    use super::{ConstLit, const_narrow_to_target};
    use serde_json::json;

    let u8_ty = json!({ "Integer": { "Unsigned": "U8" } });
    assert!(matches!(
        const_narrow_to_target(Some(&u8_ty), ConstLit::UInt(400)),
        ConstLit::UInt(144)
    ));
    // A signed destination re-extends rather than masking.
    let i8_ty = json!({ "Integer": { "Signed": "I8" } });
    assert!(matches!(
        const_narrow_to_target(Some(&i8_ty), ConstLit::UInt(200)),
        ConstLit::Int(-56)
    ));
    // The u32 flag itself is already in range and must survive intact.
    let u32_ty = json!({ "Integer": { "Unsigned": "U32" } });
    assert!(matches!(
        const_narrow_to_target(Some(&u32_ty), ConstLit::UInt(1 << 31)),
        ConstLit::UInt(v) if v == 1u64 << 31
    ));
    // Non-integer and absent destinations pass the value through.
    assert!(matches!(
        const_narrow_to_target(None, ConstLit::Int(7)),
        ConstLit::Int(7)
    ));
    let bool_ty = json!({ "Bool": [] });
    assert!(matches!(
        const_narrow_to_target(Some(&bool_ty), ConstLit::Bool(true)),
        ConstLit::Bool(true)
    ));
}

fn const_expr_integer(signed: bool, width: &str, text: &str) -> serde_json::Value {
    let int = if signed {
        serde_json::json!({"Signed": [width, text]})
    } else {
        serde_json::json!({"Unsigned": [width, text]})
    };
    let ty_int = if signed {
        serde_json::json!({"Signed": width})
    } else {
        serde_json::json!({"Unsigned": width})
    };
    serde_json::json!([
        {"Integer": int},
        {"Scalar": {"Integer": ty_int}}
    ])
}

fn const_expr_cast(
    inner: serde_json::Value,
    cast_kind: serde_json::Value,
    dest_ty: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!([{"Cast": [inner, cast_kind]}, dest_ty])
}

#[test]
fn decode_constant_cast_narrows_integer_and_rejects_lossy() {
    use super::{DecodedConst, LowerError, decode_constant};

    let llbc = fixture_llbc();
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let narrowed = decode_constant(
        &llbc,
        &const_expr_cast(
            const_expr_integer(true, "I32", "300"),
            serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}}),
            u8_ty,
        ),
    );
    assert!(
        matches!(narrowed, Ok(DecodedConst::UInt(44))),
        "300 as u8 is 44"
    );

    let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
    let sign = decode_constant(
        &llbc,
        &const_expr_cast(
            const_expr_integer(true, "I64", "-1"),
            serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}}),
            u32_ty,
        ),
    );
    assert!(
        matches!(sign, Ok(DecodedConst::UInt(4_294_967_295))),
        "-1 i64 as u32 is 4294967295"
    );

    let f64_ty = serde_json::json!({"Scalar": {"Float": "F64"}});
    let int_to_float = decode_constant(
        &llbc,
        &const_expr_cast(
            const_expr_integer(true, "I32", "300"),
            serde_json::json!({"Scalar": {"Float": "F64"}}),
            f64_ty,
        ),
    );
    assert!(
        matches!(
            int_to_float,
            Err(LowerError::Unsupported(ref msg)) if msg.contains("Cast constant not folded")
        ),
        "int→float Cast must not identity-fold"
    );

    let raw_ptr = decode_constant(
        &llbc,
        &const_expr_cast(
            const_expr_integer(true, "I64", "0"),
            serde_json::json!({"RawPtr": {"Deduplicated": 0}}),
            serde_json::json!({"RawPtr": {"Deduplicated": 0}}),
        ),
    );
    assert!(
        matches!(
            raw_ptr,
            Err(LowerError::Unsupported(ref msg)) if msg.contains("Cast constant not folded")
        ),
        "RawPtr Cast must not identity-fold"
    );
}

#[test]
fn decode_constant_ptr_is_not_the_pointee_and_ref_matches_10_04() {
    use super::{LowerError, decode_constant};

    let llbc = fixture_llbc();
    let seven = const_expr_integer(false, "U8", "7");
    let ptr = serde_json::json!([
        {"Ptr": [{"kind": {"Value": seven.clone()}}]},
        {"RawPtr": ["Erased", {"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Const"]}
    ]);
    let ptr_decoded = decode_constant(&llbc, &ptr);
    assert!(
        matches!(
            ptr_decoded,
            Err(LowerError::Unsupported(ref msg)) if msg.contains("raw pointer constant")
        ),
        "Ptr must stay a pointer (`PtrRepr.convert_const`)"
    );

    let shared = serde_json::json!([
        {"Ref": [{"kind": {"Value": seven}}]},
        {"Ref": ["Erased", {"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Shared"]}
    ]);
    let ref_decoded = decode_constant(&llbc, &shared);
    assert!(
        matches!(
            ref_decoded,
            Err(LowerError::Unsupported(ref msg)) if msg.contains("reference constant")
        ),
        "10.04 decode_constant had no Ref peel"
    );
}

#[test]
fn decode_constant_opaque_typeid_pointer_is_the_opaque_path() {
    use super::{DecodedConst, decode_constant};

    let llbc = fixture_llbc();
    let expr = serde_json::json!([
        {"Opaque": "pointer into a `TypeId`"},
        {"RawPtr": ["Erased", {"Scalar": {"Integer": {"Unsigned": "U8"}}}, "Const"]}
    ]);
    match decode_constant(&llbc, &expr) {
        Ok(DecodedConst::Str(s)) => {
            assert_eq!(s, "opaque:pointer into a `TypeId`")
        }
        Ok(_) => panic!("Opaque TypeId must take the Opaque path, got a non-Str constant"),
        Err(err) => panic!("Opaque TypeId must take the Opaque path, got {err}"),
    }
}

/// A fieldless-enum discriminant is the by-value constant
/// (`tyref_to_value_type` Int / `AggregateShape::Discriminant` ConstInt).
#[test]
fn const_lit_to_op_fieldless_enum_disc_is_constint() {
    use crate::model::OpKind;
    assert!(matches!(
        super::const_lit_to_op(super::ConstLit::EnumDisc(2)),
        Some(OpKind::ConstInt(2))
    ));
}

fn unsigned_const(width: &str, text: &str) -> serde_json::Value {
    serde_json::json!([{"Integer": {"Unsigned": [width, text]}}, u32_ty()])
}

fn literal_init_body(value: &str) -> serde_json::Value {
    let ty = u32_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": ty}
        ]},
        "body": [{
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 0}, "ty": ty},
                    {"Use": [{"Const": unsigned_const("U32", value)}, "No"]}
                ]}
            }],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn add_init_body(lhs: &str, rhs: &str) -> serde_json::Value {
    let ty = u32_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": ty}
        ]},
        "body": [{
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 0}, "ty": ty},
                    {"BinaryOp": [
                        "Add",
                        {"Const": unsigned_const("U32", lhs)},
                        {"Const": unsigned_const("U32", rhs)}
                    ]}
                ]}
            }],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn call_init_body() -> serde_json::Value {
    let ty = u32_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": ty}
        ]},
        "body": [{
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {
                        "kind": {"Fun": 2},
                        "generics": empty_generics()
                    }},
                    "args": []}, "dest": {"kind": {"Local": 0}, "ty": ty},
                "target": 1,
                "on_unwind": 1
            }}}
        }, {
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn unsigned_usize_const(text: &str) -> serde_json::Value {
    serde_json::json!([{"Integer": {"Unsigned": ["Usize", text]}}, usize_ty()])
}

fn size_of_fun(def_id: u64) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(&["core", "mem", "size_of"], "", false),
        "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
        "body": "Opaque"
    })
}

fn init_usize_fun(def_id: u64, path: &[&str], body: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
        "src": {"GlobalInitializer": {"id": def_id}},
        "body": body
    })
}

fn named_usize_const_global(
    def_id: u64,
    path: &[&str],
    source_text: &str,
    is_local: bool,
    init: u64,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, source_text, is_local),
        "global_kind": "NamedConst",
        "ty": usize_ty(),
        "value": {"Value": [0, [{"Call": [{"kind": {"Fun": init}, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}, []]}, usize_ty()]]}
    })
}

fn size_of_init_body(size_of_id: u64, arg_ty: serde_json::Value) -> serde_json::Value {
    let dest_ty = usize_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": dest_ty}
        ]},
        "body": [{
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {
                        "kind": {"Fun": size_of_id},
                        "generics": {
                            "regions": [],
                            "types": [arg_ty],
                            "const_generics": [],
                            "trait_refs": []
                        }
                    }},
                    "args": []}, "dest": {"kind": {"Local": 0}, "ty": dest_ty},
                "target": 2,
                "on_unwind": 1
            }}}
        }, {
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "UnwindResume"}
        }, {
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn size_of_usize_init_body(size_of_id: u64) -> serde_json::Value {
    size_of_init_body(size_of_id, usize_ty())
}

fn size_of_call_fun(
    def_id: u64,
    path: &[&str],
    size_of_id: u64,
    arg_ty: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
        "body": size_of_init_body(size_of_id, arg_ty)
    })
}

fn nominal_adt_type_decl(def_id: u64, path: &[&str], ptr_metadata: &str) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(path, "", true),
        "kind": {"Struct": []},
        "ptr_metadata": ptr_metadata
    })
}

fn raw_ptr_to_nominal_adt(id: u64) -> serde_json::Value {
    serde_json::json!({
        "RawPtr": [{
            "Adt": {
                "id": id,
                "generics": empty_generics(),
                "builtin": null
            }
        }, "Mut"]
    })
}

fn two_times_size_of_usize_init_body(size_of_id: u64) -> serde_json::Value {
    let ty = usize_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": ty},
            {"index": 1, "name": null, "span": span_json(), "ty": ty}
        ]},
        "body": [{
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {
                        "kind": {"Fun": size_of_id},
                        "generics": {
                            "regions": [],
                            "types": [ty],
                            "const_generics": [],
                            "trait_refs": []
                        }
                    }},
                    "args": []}, "dest": {"kind": {"Local": 1}, "ty": ty},
                "target": 1,
                "on_unwind": 1
            }}}
        }, {
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 0}, "ty": ty},
                    {"BinaryOp": [
                        "Mul",
                        {"Const": unsigned_usize_const("2")},
                        {"Read": {"kind": {"Local": 1}, "ty": ty}}
                    ]}
                ]}
            }],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn add_named_const_init_body(global_id: u64, rhs: &str) -> serde_json::Value {
    let ty = usize_ty();
    serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": ty}
        ]},
        "body": [{
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 0}, "ty": ty},
                    {"BinaryOp": [
                        "Add",
                        {"Read": {"kind": {"Global": {
                            "generics": empty_generics(),
                            "id": global_id
                        }}, "ty": ty}},
                        {"Const": unsigned_usize_const(rhs)}
                    ]}
                ]}
            }],
            "terminator": {"span": span_json(), "kind": "Return"}
        }]
    }})
}

fn reader_usize_fun(path: &[&str], global_id: u64) -> serde_json::Value {
    let ty = usize_ty();
    serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(path, "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": ty},
                        {"Use": [{"Read": {
                            "kind": {"Global": {"generics": empty_generics(), "id": global_id}},
                            "ty": ty
                        }}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn folded_uints(graph: &crate::model::FunctionGraph) -> Vec<u64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op.kind {
            OpKind::ConstUInt(n) => Some(n),
            _ => None,
        })
        .collect()
}

#[test]
fn discover_foldable_const_lits_harvests_named_const_literal() {
    let path = ["pyre_object", "intobject", "W_INT_USER_GC_TYPE_ID"];
    let defining = const_artifact(
        "pyre_object",
        serde_json::json!([null, init_fun(1, &path, literal_init_body("185"))]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &path,
                "pub const W_INT_USER_GC_TYPE_ID: u32 = 185;",
                true,
                "NamedConst",
                1
            )
        ]),
    );
    assert_eq!(
        super::discover_foldable_const_lits(&defining),
        vec![(
            "pyre_object::intobject::W_INT_USER_GC_TYPE_ID".into(),
            OpKind::ConstUInt(185)
        )]
    );
}

#[test]
fn discover_foldable_const_lits_folds_bitflag_from_element_bits() {
    let path = ["rustpython_compiler_core", "bytecode", "CO_FAST_CELL"];
    let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let flag_names = ["ArgPos", "ArgKw", "ArgVar", "Hidden", "Local", "Cell"];
    let variants: Vec<_> = flag_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            serde_json::json!({
                "name": name,
                "fields": [],
                "discriminant": {"Unsigned": ["U8", (index + 1).to_string()]},
            })
        })
        .collect();
    let enum_decl = serde_json::json!({
        "def_id": 1,
        "item_meta": item_meta_json(
            &["rustpython_compiler_core", "bytecode", "CoFastFlag"],
            "",
            false
        ),
        "kind": {"Enum": variants},
    });
    let opaque = |def_id: u64, leaf: &str| {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta_json(
                &["rustpython_compiler_core", "bytecode", "<Impl>", leaf],
                "",
                false
            ),
            "signature": {"is_unsafe": false, "inputs": [], "output": u8_ty},
            "body": "Opaque"
        })
    };
    let generics = empty_generics();
    let adt = serde_json::json!({
        "id": 1,
        "generics": generics,
        "builtin": null
    });
    let body = serde_json::json!({"Unstructured": {
        "span": span_json(),
        "locals": {"arg_count": 0, "locals": [
            {"index": 0, "name": null, "span": span_json(), "ty": u8_ty},
            {"index": 1, "name": null, "span": span_json(), "ty": u8_ty},
            {"index": 2, "name": null, "span": span_json(), "ty": u8_ty},
            {"index": 3, "name": null, "span": span_json(), "ty": u8_ty},
            {"index": 4, "name": null, "span": span_json(), "ty": u8_ty}
        ]},
        "body": [{
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 1}, "ty": u8_ty},
                    {"Aggregate": [{"Adt": [adt, 5, null]}, []]}
                ]}
            }],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                    "args": [{"Move": {"kind": {"Local": 1}, "ty": u8_ty}}]}, "dest": {"kind": {"Local": 2}, "ty": u8_ty},
                "target": 1,
                "on_unwind": 3
            }}}
        }, {
            "statements": [{
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 3}, "ty": u8_ty},
                    {"Ref": {
                        "place": {"kind": {"Local": 2}, "ty": u8_ty},
                        "kind": "Shared",
                        "ptr_metadata": null
                    }}
                ]}
            }, {
                "span": span_json(),
                "kind": {"Assign": [
                    {"kind": {"Local": 4}, "ty": u8_ty},
                    {"Ref": {
                        "place": {"kind": {"Projection": [
                            {"kind": {"Local": 3}, "ty": u8_ty},
                            "Deref"
                        ]}, "ty": u8_ty},
                        "kind": "Shared",
                        "ptr_metadata": null
                    }}
                ]}
            }],
            "terminator": {"span": span_json(), "kind": {"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": 3}, "generics": generics}},
                    "args": [{"Move": {"kind": {"Local": 4}, "ty": u8_ty}}]}, "dest": {"kind": {"Local": 0}, "ty": u8_ty},
                "target": 2,
                "on_unwind": 3
            }}}
        }, {
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "Return"}
        }, {
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span_json(), "kind": "UnwindResume"}
        }]
    }});
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "rustpython_compiler_core",
            "type_decls": [null, enum_decl],
            "fun_decls": [
                null,
                init_fun(1, &path, body),
                opaque(2, "from_element"),
                opaque(3, "bits")
            ],
            "global_decls": [null, named_const_global(
                1,
                &path,
                "pub const CO_FAST_CELL: u8 = CoFastFlags::from_element(CoFastFlag::Cell).bits();",
                false,
                "NamedConst",
                1
            )],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("bitflag const llbc");
    // Variant index 5 is Cell. Its discriminant is 6, so the mask is
    // 1 << 6. Shifting by the index would yield 32.
    assert_eq!(
        super::discover_foldable_const_lits(&llbc),
        vec![(path.join("::"), OpKind::ConstUInt(64))]
    );
}

#[test]
fn discover_foldable_const_lits_skips_thread_local_static_mut_and_call() {
    let tl = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            init_fun(1, &["fixture", "TLS"], literal_init_body("1"))
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &["fixture", "TLS"],
                "thread_local!",
                true,
                "ThreadLocal",
                1
            )
        ]),
    );
    let static_mut = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            init_fun(1, &["fixture", "COUNTER"], literal_init_body("0"))
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &["fixture", "COUNTER"],
                "pub static mut COUNTER: u32 = 0;",
                true,
                "Static",
                1
            )
        ]),
    );
    let call = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            init_fun(1, &["fixture", "COMPUTED"], call_init_body())
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &["fixture", "COMPUTED"],
                "pub const COMPUTED: u32 = f();",
                true,
                "NamedConst",
                1
            )
        ]),
    );
    assert!(super::discover_foldable_const_lits(&tl).is_empty());
    assert!(super::discover_foldable_const_lits(&static_mut).is_empty());
    assert!(super::discover_foldable_const_lits(&call).is_empty());
}

fn transparent_bitflag_initializer_fixture(transparent: bool) -> Llbc {
    use serde_json::{Value, json};
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
    let boolean = json!({"Scalar": "Bool"});
    let adt = |id| json!({"Adt": {"id": id, "builtin": null, "generics": empty_generics()}});
    let flag = adt(0);
    let flags = adt(1);
    let flags_ref = json!({"Ref": ["Erased", flags.clone(), "Shared"]});
    let place = |i, ty: &Value| json!({"kind": {"Local": i}, "ty": ty});
    let read = |i, ty: &Value| json!({"Read": place(i, ty)});
    let literal = |n: &str| json!({"Const": [{"Integer": {"Unsigned": ["U8", n]}}, word.clone()]});
    let assign = |i, ty: &Value, value: Value| json!({"span": span_json(), "kind": {"Assign": [place(i, ty), value]}});
    let block = |statements: Vec<Value>, term: Value| {
        json!({
            "statements": statements, "kind": "Regular", "terminator": {"span": span_json(), "kind": term}
        })
    };
    let call = |id, args: Vec<Value>, dest, ty: &Value, target| {
        json!({"Call": {
            "call": {"func": {"Regular": {"kind": {"Fun": id}, "generics": empty_generics()}},
                "args": args}, "dest": place(dest, ty), "target": target, "on_unwind": target
        }})
    };
    let body = |args, tys: Vec<Value>, blocks: Vec<Value>| {
        json!({"Unstructured": {
            "span": span_json(), "locals": {"arg_count": args, "locals": tys.iter().enumerate().map(|(i, ty)|
                json!({"index": i, "name": null, "span": span_json(), "ty": ty})).collect::<Vec<_>>()}, "body": blocks
        }})
    };
    let function = |id, name, inputs: Vec<Value>, output: Value, mir: Value| {
        json!({
            "def_id": id, "item_meta": item_meta_json(&["fixture", name], "", true),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": output}, "body": mir
        })
    };
    let init = function(
        0,
        "MASK",
        vec![],
        word.clone(),
        body(
            0,
            vec![word.clone(), flag.clone(), flags.clone(), flags_ref.clone()],
            vec![
                block(
                    vec![assign(
                        1,
                        &flag,
                        json!({"Aggregate": [{"Adt": [{"id": 0}, 0, null]}, []]}),
                    )],
                    call(1, vec![read(1, &flag)], 2, &flags, 1),
                ),
                block(
                    vec![assign(
                        3,
                        &flags_ref,
                        json!({"Ref": {"place": place(2, &flags), "kind": "Shared", "ptr_metadata": null}}),
                    )],
                    call(2, vec![read(3, &flags_ref)], 0, &word, 2),
                ),
                block(vec![], json!("Return")),
            ],
        ),
    );
    let from_element = function(
        1,
        "from_element",
        vec![flag.clone()],
        flags.clone(),
        body(
            1,
            vec![
                flags.clone(),
                flag.clone(),
                word.clone(),
                boolean.clone(),
                word.clone(),
            ],
            vec![
                block(
                    vec![
                        assign(2, &word, json!({"Discriminant": place(1, &flag)})),
                        assign(
                            3,
                            &boolean,
                            json!({"BinaryOp": ["Lt", read(2, &word), literal("8")]}),
                        ),
                    ],
                    json!({"Switch": {"discr": read(3, &boolean), "targets": {"If": [1, 2]}}}),
                ),
                block(
                    vec![
                        assign(
                            4,
                            &word,
                            json!({"BinaryOp": [{"Shl": "Wrap"}, literal("1"), read(2, &word)]}),
                        ),
                        assign(
                            0,
                            &flags,
                            json!({"Aggregate": [{"Adt": [{"id": 1}, null, null]}, [read(4, &word)]]}),
                        ),
                    ],
                    json!("Return"),
                ),
                block(vec![], json!("UnwindResume")),
            ],
        ),
    );
    let pointee =
        json!({"kind": {"Projection": [place(1, &flags_ref), "Deref"]}, "ty": flags.clone()});
    let bits = function(
        2,
        "bits",
        vec![flags_ref.clone()],
        word.clone(),
        body(
            1,
            vec![word.clone(), flags_ref.clone()],
            vec![block(
                vec![assign(
                    0,
                    &word,
                    json!({"Use": [{"Read": {
            "kind": {"Projection": [pointee, {"Field": [null, 0]}]}, "ty": word.clone()
        }}, "Yes"]}),
                )],
                json!("Return"),
            )],
        ),
    );
    let types = vec![
        json!({"def_id": 0, "item_meta": item_meta_json(&["fixture", "Flag"], "", true),
        "kind": {"Enum": [{"name": "Cell", "fields": [], "discriminant": {"Unsigned": ["U8", "6"]}}]}}),
        json!({"def_id": 1, "item_meta": item_meta_json(&["fixture", "Flags"], "", true),
            "layout": [{"key": "fixture", "value": {"repr": {
                "repr_algo": "Rust", "align_modif": null, "transparent": transparent
            }, "variant_layouts": []}}],
            "kind": {"Struct": [{"name": null, "ty": word.clone(), "attr_info": null}]}}),
    ];
    let global = named_const_global(0, &["fixture", "MASK"], "", true, "NamedConst", 0);
    let file = json!({"charon_version": "0.1.301", "has_errors": false, "translated": {
        "crate_name": "fixture", "type_decls": types, "fun_decls": [init, from_element, bits],
        "global_decls": [global], "trait_decls": [], "trait_impls": []
    }});
    Llbc::from_slice(file.to_string().as_bytes()).unwrap()
}

#[test]
fn transparent_bitflag_initializer_folds_its_body() {
    let llbc = transparent_bitflag_initializer_fixture(true);
    assert_eq!(
        super::discover_foldable_const_lits(&llbc),
        vec![("fixture::MASK".to_string(), OpKind::ConstUInt(64))]
    );
}

#[test]
fn nontransparent_bitflag_initializer_is_not_peeled() {
    let llbc = transparent_bitflag_initializer_fixture(false);
    assert!(super::discover_foldable_const_lits(&llbc).is_empty());
}

#[test]
fn discover_foldable_const_lits_skips_non_unique_name_path() {
    let path = ["fixture", "<Impl>", "CPYTHON_IMMUTABLETYPE"];
    let shared = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            init_fun(1, &path, literal_init_body("1")),
            init_fun(2, &path, literal_init_body("0"))
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &path,
                "const CPYTHON_IMMUTABLETYPE: bool = true;",
                true,
                "NamedConst",
                1
            ),
            named_const_global(
                2,
                &path,
                "const CPYTHON_IMMUTABLETYPE: bool = false;",
                true,
                "NamedConst",
                2
            )
        ]),
    );
    assert!(super::discover_foldable_const_lits(&shared).is_empty());
    super::register_ambiguous_impl_foldable_const_lits(&shared);
    let first = shared.foldable_const_lit(1);
    let second = shared.foldable_const_lit(2);
    assert!(
        first.is_some() && second.is_some() && first != second,
        "each impl const keeps its own literal, got {first:?} {second:?}"
    );
}

#[test]
fn discover_foldable_const_lits_harvests_size_of_usize() {
    let path = ["fixture", "LOWLEVEL_STRING_LEN_OFFSET"];
    let defining = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            init_usize_fun(1, &path, size_of_usize_init_body(2)),
            size_of_fun(2)
        ]),
        serde_json::json!([
            null,
            named_usize_const_global(
                1,
                &path,
                "pub const LOWLEVEL_STRING_LEN_OFFSET: usize = size_of::<usize>();",
                true,
                1
            )
        ]),
    );
    let word = crate::layout::target_word_size() as u64;
    assert_eq!(
        super::discover_foldable_const_lits(&defining),
        vec![(
            "fixture::LOWLEVEL_STRING_LEN_OFFSET".into(),
            OpKind::ConstUInt(word)
        )]
    );

    // Charon spells a nominal ADT as `"builtin": null`. The inline
    // `size_of` fold shares `size_align_const_from_tyexpr` with
    // `fold_size_const_global`; `nominal_adt_pointee` folds only when
    // that ADT's declaration has `ptr_metadata: "None"` (a thin
    // `lltype.Ptr`, one word). `size_of::<*mut Adt>()` must not
    // residualize as a `size_of` Call.
    let inline = llbc_with_types(
        "fixture",
        vec![nominal_adt_type_decl(0, &["fixture", "Adt"], "None")],
        vec![
            size_of_call_fun(0, &["fixture", "ptr_size"], 1, raw_ptr_to_nominal_adt(0)),
            size_of_fun(1),
        ],
    );
    let graph = crate::test_support::lower_function(&inline, "ptr_size")
        .expect("lower size_of of *mut nominal ADT");
    let word = crate::layout::target_word_size() as i64;
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(op.kind, OpKind::ConstInt(n) if n == word)),
        "size_of of a raw pointer to a nominal ADT with ptr_metadata None folds to the target word: {graph:#?}"
    );
    assert_eq!(
        nullary_calls_ending(&graph, "size_of"),
        0,
        "the folded size_of must not remain a residual Call: {graph:#?}"
    );

    let fat = llbc_with_types(
        "fixture",
        vec![nominal_adt_type_decl(0, &["fixture", "FatAdt"], "Length")],
        vec![
            size_of_call_fun(
                0,
                &["fixture", "fat_ptr_size"],
                1,
                raw_ptr_to_nominal_adt(0),
            ),
            size_of_fun(1),
        ],
    );
    let graph = crate::test_support::lower_function(&fat, "fat_ptr_size")
        .expect("lower size_of of *mut Length ADT");
    assert_eq!(
        nullary_calls_ending(&graph, "size_of"),
        1,
        "size_of of a raw pointer to a Length-metadata ADT must stay a Call: {graph:#?}"
    );
}

#[test]
fn discover_foldable_const_lits_harvests_composed_size_of_arith() {
    let chars_path = ["fixture", "LOWLEVEL_STRING_CHARS_OFFSET"];
    let base_path = ["fixture", "LOWLEVEL_STR_BASE_SIZE"];
    let defining = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            null,
            size_of_fun(2),
            init_usize_fun(3, &chars_path, two_times_size_of_usize_init_body(2)),
            init_usize_fun(4, &base_path, add_named_const_init_body(1, "1"))
        ]),
        serde_json::json!([
            null,
            named_usize_const_global(
                1,
                &chars_path,
                "pub const LOWLEVEL_STRING_CHARS_OFFSET: usize = 2 * size_of::<usize>();",
                true,
                3
            ),
            named_usize_const_global(
                2,
                &base_path,
                "pub const LOWLEVEL_STR_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET + 1;",
                true,
                4
            )
        ]),
    );
    let word = crate::layout::target_word_size() as u64;
    let harvested = super::discover_foldable_const_lits(&defining);
    assert_eq!(
        harvested,
        vec![
            (
                "fixture::LOWLEVEL_STRING_CHARS_OFFSET".into(),
                OpKind::ConstUInt(word * 2)
            ),
            (
                "fixture::LOWLEVEL_STR_BASE_SIZE".into(),
                OpKind::ConstUInt(word * 2 + 1)
            ),
        ]
    );
}

#[test]
fn linked_foldable_const_lit_folds_foreign_composed_size_of() {
    let chars_path = ["fixture", "LOWLEVEL_STRING_CHARS_OFFSET"];
    let base_path = ["fixture", "LOWLEVEL_STR_BASE_SIZE"];
    let defining = const_artifact(
        "fixture",
        serde_json::json!([
            null,
            null,
            size_of_fun(2),
            init_usize_fun(3, &chars_path, two_times_size_of_usize_init_body(2)),
            init_usize_fun(4, &base_path, add_named_const_init_body(1, "1"))
        ]),
        serde_json::json!([
            null,
            named_usize_const_global(
                1,
                &chars_path,
                "pub const LOWLEVEL_STRING_CHARS_OFFSET: usize = 2 * size_of::<usize>();",
                true,
                3
            ),
            named_usize_const_global(
                2,
                &base_path,
                "pub const LOWLEVEL_STR_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET + 1;",
                true,
                4
            )
        ]),
    );
    let mut opaque_init = init_usize_fun(1, &base_path, serde_json::json!("Opaque"));
    opaque_init["item_meta"]["is_local"] = serde_json::json!(false);
    let dependent = const_artifact(
        "dependent",
        serde_json::json!([
            reader_usize_fun(&["dependent", "read_base"], 1),
            opaque_init
        ]),
        serde_json::json!([
            null,
            named_usize_const_global(
                1,
                &base_path,
                "pub const LOWLEVEL_STR_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET + 1;",
                false,
                1
            )
        ]),
    );
    let llbcs = [defining, dependent];
    super::register_foldable_const_lits(Vec::new());
    let unlinked =
        crate::test_support::lower_function(&llbcs[1], "read_base").expect("unlinked lowers");
    assert!(
        folded_uints(&unlinked).is_empty(),
        "a Foreign size_of-arith const without a harvest must not fold: {unlinked:?}"
    );
    assert_eq!(
        nullary_calls_ending(&unlinked, "LOWLEVEL_STR_BASE_SIZE"),
        1,
        "the unlinked read stays a nullary call on the const path"
    );

    super::link_transparent_scalar_types(&llbcs);
    let graph = crate::test_support::lower_function(&llbcs[1], "read_base").expect("linked lowers");
    let word = crate::layout::target_word_size() as u64;
    assert_eq!(folded_uints(&graph), vec![word * 2 + 1]);
    assert_eq!(
        nullary_calls_ending(&graph, "LOWLEVEL_STR_BASE_SIZE"),
        0,
        "the linked composed size_of read must not remain a nullary call: {graph:?}"
    );
}

#[test]
fn linked_foldable_const_lit_folds_foreign_global_to_literal() {
    let path = ["pyre_object", "intobject", "W_INT_USER_GC_TYPE_ID"];
    let defining = const_artifact(
        "pyre_object",
        serde_json::json!([null, init_fun(1, &path, literal_init_body("185"))]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &path,
                "pub const W_INT_USER_GC_TYPE_ID: u32 = 185;",
                true,
                "NamedConst",
                1
            )
        ]),
    );
    let mut opaque_init = init_fun(1, &path, serde_json::json!("Opaque"));
    opaque_init["item_meta"]["is_local"] = serde_json::json!(false);
    let dependent = const_artifact(
        "pyre_interpreter",
        serde_json::json!([reader_fun(&["pyre_interpreter", "read_id"], 1), opaque_init]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &path,
                "pub const W_INT_USER_GC_TYPE_ID: u32 = 185;",
                false,
                "NamedConst",
                1
            )
        ]),
    );
    let llbcs = [defining, dependent];
    super::register_foldable_const_lits(Vec::new());
    let unlinked =
        crate::test_support::lower_function(&llbcs[1], "read_id").expect("unlinked lowers");
    assert!(
        folded_uints(&unlinked).is_empty(),
        "a Foreign const without a harvest must not fold: {unlinked:?}"
    );
    assert_eq!(
        nullary_calls_ending(&unlinked, "W_INT_USER_GC_TYPE_ID"),
        1,
        "the unlinked read stays a nullary call on the const path"
    );

    super::link_transparent_scalar_types(&llbcs);
    let graph = crate::test_support::lower_function(&llbcs[1], "read_id").expect("linked lowers");
    assert_eq!(folded_uints(&graph), vec![185]);
    assert_eq!(
        nullary_calls_ending(&graph, "W_INT_USER_GC_TYPE_ID"),
        0,
        "the linked read must not remain a nullary call: {graph:?}"
    );
}

#[test]
fn linked_foldable_const_lit_keys_the_full_name_path() {
    let a = const_artifact(
        "crate_a",
        serde_json::json!([
            null,
            init_fun(1, &["crate_a", "SIZE"], literal_init_body("4"))
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &["crate_a", "SIZE"],
                "pub const SIZE: u32 = 4;",
                true,
                "NamedConst",
                1
            )
        ]),
    );
    let b = const_artifact(
        "crate_b",
        serde_json::json!([
            null,
            init_fun(1, &["crate_b", "SIZE"], literal_init_body("8"))
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &["crate_b", "SIZE"],
                "pub const SIZE: u32 = 8;",
                true,
                "NamedConst",
                1
            )
        ]),
    );
    let llbcs = [a, b];
    super::link_transparent_scalar_types(&llbcs);
    assert_eq!(
        super::attached_foldable_lit(&llbcs[0], 1),
        Some(OpKind::ConstUInt(4))
    );
    assert_eq!(
        super::attached_foldable_lit(&llbcs[1], 1),
        Some(OpKind::ConstUInt(8))
    );
    assert!(super::attached_foldable_lit(&llbcs[0], 0).is_none());
}

#[test]
fn linked_foldable_const_lit_local_body_wins_over_linked() {
    let path = ["fixture", "VALUE"];
    let local = const_artifact(
        "fixture",
        serde_json::json!([
            reader_fun(&["fixture", "read_value"], 1),
            init_fun(1, &path, add_init_body("90", "9"))
        ]),
        serde_json::json!([
            null,
            named_const_global(
                1,
                &path,
                "pub const VALUE: u32 = 90 + 9;",
                true,
                "NamedConst",
                1
            )
        ]),
    );
    super::attach_foldable_const_lits(&local, &[("fixture::VALUE".into(), OpKind::ConstUInt(185))]);
    let graph =
        crate::test_support::lower_function(&local, "read_value").expect("local body lowers");
    assert_eq!(folded_uints(&graph), vec![99]);
    assert_eq!(nullary_calls_ending(&graph, "VALUE"), 0);
}

#[test]
#[should_panic(expected = "inconsistent linked definitions")]
fn register_foldable_const_lits_asserts_inconsistent_values() {
    super::register_foldable_const_lits([
        ("crate::SIZE".into(), OpKind::ConstUInt(4)),
        ("crate::SIZE".into(), OpKind::ConstUInt(8)),
    ]);
}

#[test]
fn impl_associated_const_literal_is_not_shared_across_decl_ids() {
    let path = ["fixture", "<Impl>", "EMPTY"];
    let a = const_artifact(
        "crate_a",
        serde_json::json!([
            reader_fun(&["crate_a", "read_a"], 1),
            init_fun(1, &path, literal_init_body("1"))
        ]),
        serde_json::json!([
            null,
            named_const_global(1, &path, "const EMPTY: u32 = 1;", true, "NamedConst", 1)
        ]),
    );
    let b = const_artifact(
        "crate_b",
        serde_json::json!([
            reader_fun(&["crate_b", "read_b"], 1),
            init_fun(1, &path, literal_init_body("7"))
        ]),
        serde_json::json!([
            null,
            named_const_global(1, &path, "const EMPTY: u32 = 7;", true, "NamedConst", 1)
        ]),
    );
    let llbcs = [a, b];
    super::link_transparent_scalar_types(&llbcs);
    let ga = crate::test_support::lower_function(&llbcs[0], "read_a").expect("a lowers");
    let gb = crate::test_support::lower_function(&llbcs[1], "read_b").expect("b lowers");
    assert_eq!(folded_uints(&ga), vec![1]);
    assert_eq!(folded_uints(&gb), vec![7]);
}

#[test]
fn primitive_float_const_folds_infinity_and_f64_named_consts() {
    match primitive_float_const(&[
        "core".into(),
        "f64".into(),
        "<Impl>".into(),
        "INFINITY".into(),
    ]) {
        Some(OpKind::ConstFloat(bits)) => assert_eq!(bits, f64::INFINITY.to_bits()),
        other => panic!("expected ConstFloat(INFINITY), got {other:?}"),
    }
    match primitive_float_const(&["core".into(), "f64".into(), "<Impl>".into(), "NAN".into()]) {
        Some(OpKind::ConstFloat(bits)) => assert_eq!(bits, f64::NAN.to_bits()),
        other => panic!("expected ConstFloat(NAN), got {other:?}"),
    }
    match primitive_float_const(&[
        "std".into(),
        "f64".into(),
        "<Impl>".into(),
        "NEG_INFINITY".into(),
    ]) {
        Some(OpKind::ConstFloat(bits)) => assert_eq!(bits, f64::NEG_INFINITY.to_bits()),
        other => panic!("expected ConstFloat(NEG_INFINITY), got {other:?}"),
    }
    match primitive_float_const(&["std".into(), "f64".into(), "consts".into(), "PI".into()]) {
        Some(OpKind::ConstFloat(bits)) => {
            assert_eq!(bits, std::f64::consts::PI.to_bits())
        }
        other => panic!("expected ConstFloat(PI), got {other:?}"),
    }
    match primitive_float_const(&["core".into(), "f64".into(), "consts".into(), "E".into()]) {
        Some(OpKind::ConstFloat(bits)) => assert_eq!(bits, std::f64::consts::E.to_bits()),
        other => panic!("expected ConstFloat(E), got {other:?}"),
    }
    assert!(
        primitive_float_const(&[
            "core".into(),
            "f64".into(),
            "consts".into(),
            "NO_SUCH".into()
        ])
        .is_none()
    );
    assert!(
        primitive_float_const(&["core".into(), "f64".into(), "<Impl>".into(), "RADIX".into()])
            .is_none()
    );
    assert!(primitive_float_const(&["unrelated".into()]).is_none());
}

#[test]
fn bitflags_trait_empty_const_folds_zero_and_rejects_all() {
    assert!(
        bitflags_trait_empty_const(&[
            "bitflags".into(),
            "traits".into(),
            "<Impl>".into(),
            "EMPTY".into(),
        ])
        .is_none(),
        "EMPTY is not a leaf fold; it comes from the impl initializer"
    );
    assert!(
        bitflags_trait_empty_const(&[
            "bitflags".into(),
            "traits".into(),
            "<Impl>".into(),
            "ALL".into()
        ])
        .is_none(),
        "Bits::ALL is width-specific under the same non-injective path"
    );
    assert!(
        bitflags_trait_empty_const(&[
            "core".into(),
            "traits".into(),
            "<Impl>".into(),
            "EMPTY".into()
        ])
        .is_none()
    );
    assert!(bitflags_trait_empty_const(&["unrelated".into()]).is_none());
}

#[test]
fn opaque_f64_nan_global_read_folds_to_const_float() {
    super::register_foldable_const_lits(Vec::new());
    let path = ["core", "f64", "<Impl>", "NAN"];
    let mut opaque_init = init_fun(1, &path, serde_json::json!("Opaque"));
    opaque_init["item_meta"]["is_local"] = serde_json::json!(false);
    let artifact = const_artifact(
        "fixture",
        serde_json::json!([reader_fun(&["fixture", "read_nan"], 1), opaque_init]),
        serde_json::json!([
            null,
            named_const_global(1, &path, "", false, "NamedConst", 1)
        ]),
    );
    let graph = crate::test_support::lower_function(&artifact, "read_nan").expect("lower");
    let bits: Vec<u64> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op.kind {
            OpKind::ConstFloat(bits) => Some(bits),
            _ => None,
        })
        .collect();
    assert_eq!(bits, vec![f64::NAN.to_bits()]);
    assert_eq!(
        nullary_calls_ending(&graph, "NAN"),
        0,
        "the Opaque f64::NAN read must not remain a nullary call: {graph:?}"
    );
}

/// Charon records a function-local static as
/// `Ident [leaf, n]` under the enclosing function. `name_path`
/// drops `n` (except `closure#N`), so two `OWNER` statics inside
/// `method_owner` collapse onto one key. The Global-place key keeps
/// `n` as `leaf#n` for `n > 0`.
#[test]
fn function_local_static_key_keeps_ident_disambiguator() {
    let owner = |n: u64| {
        static_key_segments(&[
            NameSeg::Ident {
                ident: ("pyre_interpreter".into(), 0),
            },
            NameSeg::Ident {
                ident: ("typedef".into(), 0),
            },
            NameSeg::Ident {
                ident: ("method_owner".into(), 0),
            },
            NameSeg::Ident {
                ident: ("OWNER".into(), n),
            },
        ])
    };
    assert_eq!(
        owner(0).join("::"),
        "pyre_interpreter::typedef::method_owner::OWNER"
    );
    assert_eq!(
        owner(1).join("::"),
        "pyre_interpreter::typedef::method_owner::OWNER#1"
    );
    assert_eq!(
        owner(56).join("::"),
        "pyre_interpreter::typedef::method_owner::OWNER#56"
    );
    assert_ne!(owner(0), owner(1));
    // Trait-impl Other segments still render as a label, matching
    // `name_path`, so existing `f64::<Impl>::NAN` keys keep matching.
    let impl_seg = NameSeg::Impl {
        trait_id: Some(1),
        skip_binder: None,
    };
    assert_eq!(
        static_key_segments(&[
            NameSeg::Ident {
                ident: ("core".into(), 0),
            },
            NameSeg::Ident {
                ident: ("f64".into(), 0),
            },
            impl_seg,
            NameSeg::Ident {
                ident: ("NAN".into(), 0),
            },
        ]),
        ["core", "f64", "<Impl>", "NAN"]
    );
    // `closure#N` keeps the same spelling `name_path` already used.
    assert_eq!(
        static_key_segments(&[NameSeg::Ident {
            ident: ("closure".into(), 3),
        }]),
        ["closure#3"]
    );
}

/// Two function-local statics that share a leaf bind distinct
/// `HostStaticAddrs` rows once the Ident disambiguator is part of
/// the key. Suffix matching is segment-exact, so `OWNER#1` does
/// not bind the un-numbered `OWNER` row.
#[test]
fn function_local_static_globals_resolve_through_disambiguated_keys() {
    super::register_foldable_const_lits(Vec::new());
    let owner = |n: u64| {
        [
            ("fixture", 0u64),
            ("typedef", 0),
            ("method_owner", 0),
            ("OWNER", n),
        ]
    };
    let artifact = const_artifact(
        "fixture",
        serde_json::json!([
            reader_fun_at(0, &["fixture", "read_owner_0"], 1),
            reader_fun_at(1, &["fixture", "read_owner_1"], 2),
        ]),
        serde_json::json!([
            null,
            static_global_with_idents(1, &owner(0)),
            static_global_with_idents(2, &owner(1)),
        ]),
    );
    let refs = [
        ("typedef::method_owner::OWNER", 0x1111i64),
        ("typedef::method_owner::OWNER#1", 0x2222i64),
    ];
    let addrs = crate::HostStaticAddrs {
        refs: &refs,
        ..Default::default()
    };
    let graph0 =
        crate::test_support::lower_function_with_static_addrs(&artifact, "read_owner_0", addrs)
            .expect("lower OWNER");
    let graph1 =
        crate::test_support::lower_function_with_static_addrs(&artifact, "read_owner_1", addrs)
            .expect("lower OWNER#1");
    let addrs0: Vec<i64> = graph0
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op.kind {
            OpKind::ConstRefAddr(addr) => Some(addr),
            _ => None,
        })
        .collect();
    let addrs1: Vec<i64> = graph1
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op.kind {
            OpKind::ConstRefAddr(addr) => Some(addr),
            _ => None,
        })
        .collect();
    assert_eq!(addrs0, vec![0x1111]);
    assert_eq!(addrs1, vec![0x2222]);
    assert_eq!(
        nullary_calls_ending(&graph0, "OWNER")
            + nullary_calls_ending(&graph1, "OWNER")
            + nullary_calls_ending(&graph1, "OWNER#1"),
        0,
        "disambiguated statics must not remain residual calls: {graph0:?} {graph1:?}"
    );
}

fn fnptrcell_type_decl(kind: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["majit_gc", "hook_cell", "FnPtrCell"], "", true),
        "kind": kind
    })
}

fn fnptrcell_global_llbc(kind: serde_json::Value) -> majit_charon_reader::Llbc {
    let cell_ty = serde_json::json!({
        "Adt": {"id": 0, "generics": empty_generics()}
    });
    let funs = vec![serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "read_cell"], "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": cell_ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": cell_ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": cell_ty},
                        {"Use": [{"Read": {
                            "kind": {"Global": {"generics": empty_generics(), "id": 1}},
                            "ty": cell_ty
                        }}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })];
    let globals = vec![
        serde_json::Value::Null,
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "CELL"], "", true),
            "global_kind": "Static",
            "ty": cell_ty
        }),
    ];
    llbc_with_types_funs_globals("fixture", vec![fnptrcell_type_decl(kind)], funs, globals)
}

fn fnptrcell_refs() -> crate::HostStaticAddrs<'static> {
    crate::HostStaticAddrs {
        refs: &[("fixture::CELL", 0x1111)],
        ..Default::default()
    }
}

fn const_ref_addrs(graph: &crate::model::FunctionGraph) -> Vec<i64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match op.kind {
            OpKind::ConstRefAddr(addr) => Some(addr),
            _ => None,
        })
        .collect()
}

fn call_paths_in(graph: &crate::model::FunctionGraph) -> Vec<Vec<String>> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: crate::model::CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect()
}

/// An Opaque `FnPtrCell` Global with a `refs` row is the cell's address,
/// not a 0-arg Call to the static path.
#[test]
fn fnptrcell_opaque_global_emits_bare_const_ref_addr() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = fnptrcell_global_llbc(serde_json::json!("Opaque"));
    let graph = super::lower_function_with_static_addrs(&llbc, "read_cell", fnptrcell_refs())
        .expect("FnPtrCell Global lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    let cell = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match op.kind {
            OpKind::ConstRefAddr(_) => op.result.clone(),
            _ => None,
        })
        .expect("ConstRefAddr dest");
    assert_eq!(
        crate::model::FunctionGraph::concretetype_of(&cell),
        crate::model::ConcreteType::Signed,
        "llmemory.Address getkind is int"
    );
    assert!(
        matches!(
            cell.concretetype(),
            Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Address)
        ),
        "FnPtrCell address is Address, got {:?}",
        cell.concretetype()
    );
    assert_eq!(
        nullary_calls_ending(&graph, "CELL"),
        0,
        "FnPtrCell Global must not residualize: {graph:?}"
    );
    assert!(
        call_paths_in(&graph).iter().all(
            |segments| segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        ),
        "Opaque FnPtrCell must not wrap ConstRefAddr: {graph:?}"
    );
}

/// A Struct-kind `FnPtrCell` still emits a bare ConstRefAddr: the type
/// lane runs before `refs_static_struct_root` would wrap the address as
/// `SomeInstance(FnPtrCell)`.
#[test]
fn fnptrcell_struct_global_does_not_cast_instance() {
    super::register_foldable_const_lits(Vec::new());
    let kind = serde_json::json!({"Struct": [{
        "name": "ptr",
        "ty": {"RawPtr": [{"Tuple": []}, "Mut"]},
        "attr_info": null
    }]});
    let llbc = fnptrcell_global_llbc(kind);
    let graph = super::lower_function_with_static_addrs(&llbc, "read_cell", fnptrcell_refs())
        .expect("Struct FnPtrCell Global lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    assert!(
        call_paths_in(&graph).iter().all(
            |segments| segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        ),
        "FnPtrCell must not wrap ConstRefAddr as an instance: {graph:?}"
    );
}

fn fnptrcell_get_llbc() -> majit_charon_reader::Llbc {
    let cell = serde_json::json!({
        "Adt": {"id": 0, "generics": empty_generics()}
    });
    let cell_ref = serde_json::json!({"Ref": ["Erased", cell, "Shared"]});
    let option_fn = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [unit_fn_ptr()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let generics = empty_generics();
    let funs = vec![
        serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "call_get"], "", true),
            "signature": {"is_unsafe": false, "inputs": [cell_ref], "output": option_fn},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 1, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": option_fn},
                    {"index": 1, "name": "cell", "span": span_json(), "ty": cell_ref}
                ]},
                "body": [
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                            "args": [{"Read": {"kind": {"Local": 1}, "ty": cell_ref}}]}, "dest": {"kind": {"Local": 0}, "ty": option_fn},
                        "target": 1,
                        "on_unwind": 2
                    }}}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": "Return"}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": "UnwindResume"}}
                ]
            }}
        }),
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["majit_gc", "hook_cell", "<Impl>", "get"], "", false),
            "signature": {"is_unsafe": false, "inputs": [cell_ref], "output": option_fn},
            "body": "Opaque"
        }),
    ];
    llbc_with_types(
        "fixture",
        vec![
            fnptrcell_type_decl(serde_json::json!("Opaque")),
            option_enum(1),
        ],
        funs,
    )
}

/// `FnPtrCell::get` is a signed word `raw_load` of `Ptr(FuncType)`
/// (`NicheIntNull::FnPtr`, `history.getkind` `'int'`).
#[test]
fn fnptrcell_get_lowers_to_signed_raw_load_of_func_ptr() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = fnptrcell_get_llbc();
    let graph = super::lower_function(&llbc, "call_get").expect("FnPtrCell::get lowers");
    let loads: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(op.kind, OpKind::RawLoad { .. }))
        .collect();
    assert_eq!(loads.len(), 1, "exactly one RawLoad: {graph:?}");
    match &loads[0].kind {
        OpKind::RawLoad {
            item_ty,
            itemsize,
            is_item_signed,
            ..
        } => {
            assert_eq!(*item_ty, crate::model::ValueType::Int);
            assert_eq!(*itemsize, crate::layout::target_word_size());
            assert!(*is_item_signed, "fn-ptr load is signed kind 'i'");
        }
        other => panic!("expected RawLoad, got {other:?}"),
    }
    let dest = loads[0].result.as_ref().expect("RawLoad dest");
    match dest.concretetype() {
        Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Ptr(p))
            if matches!(
                p.TO,
                crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(_)
            ) => {}
        other => panic!("load dest must be Ptr(FuncType), got {other:?}"),
    }
    assert_eq!(
        crate::model::FunctionGraph::concretetype_of(dest),
        crate::model::ConcreteType::Signed,
        "history.getkind of Ptr(FuncType) is int"
    );
    assert_eq!(
        nullary_calls_ending(&graph, "get"),
        0,
        "FnPtrCell::get must not residualize: {graph:?}"
    );
}

fn i64_fn_ptr() -> serde_json::Value {
    serde_json::json!({
        "FnPtr": {
            "skip_binder": {
                "inputs": [],
                "output": {"Scalar": {"Integer": {"Signed": "I64"}}},
                "is_unsafe": false
            }
        }
    })
}

fn oncelock_fn_generics() -> serde_json::Value {
    serde_json::json!({
        "regions": [],
        "types": [i64_fn_ptr()],
        "const_generics": [],
        "trait_refs": []
    })
}

fn oncelock_fn_type_decl(
    kind: serde_json::Value,
    layout: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut td = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["std", "sync", "once_lock", "OnceLock"], "", false),
        "kind": kind
    });
    if let Some(layout) = layout {
        td["layout"] = layout;
    }
    td
}

fn oncelock_fn_adt() -> serde_json::Value {
    serde_json::json!({
        "Adt": {"id": 0, "generics": oncelock_fn_generics()}
    })
}

fn oncelock_fn_global_llbc(
    kind: serde_json::Value,
    layout: Option<serde_json::Value>,
) -> majit_charon_reader::Llbc {
    let cell_ty = oncelock_fn_adt();
    let funs = vec![serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "read_cell"], "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": cell_ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": cell_ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": cell_ty},
                        {"Use": [{"Read": {
                            "kind": {"Global": {"generics": empty_generics(), "id": 1}},
                            "ty": cell_ty
                        }}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })];
    let globals = vec![
        serde_json::Value::Null,
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "CELL"], "", true),
            "global_kind": "Static",
            "ty": cell_ty
        }),
    ];
    llbc_with_types_funs_globals(
        "fixture",
        vec![oncelock_fn_type_decl(kind, layout)],
        funs,
        globals,
    )
}

fn oncelock_fn_value_layout() -> serde_json::Value {
    serde_json::json!([{
        "key": "fixture-target",
        "value": {
            "size": 16,
            "align": 8,
            "variant_layouts": [{"field_offsets": [0, 8]}],
            "repr": {"transparent": false}
        }
    }])
}

fn oncelock_fn_struct_kind() -> serde_json::Value {
    serde_json::json!({"Struct": [
        {"name": "once", "ty": {"Scalar": {"Integer": {"Unsigned": "Usize"}}}, "attr_info": null},
        {"name": "value", "ty": i64_fn_ptr(), "attr_info": null}
    ]})
}

/// An Opaque `OnceLock<fn>` Global with a `refs` row is the cell's address,
/// not a 0-arg Call to the static path.
#[test]
fn oncelock_fn_opaque_global_emits_bare_const_ref_addr() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = oncelock_fn_global_llbc(serde_json::json!("Opaque"), None);
    let graph = super::lower_function_with_static_addrs(&llbc, "read_cell", fnptrcell_refs())
        .expect("OnceLock<fn> Global lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    let cell = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match op.kind {
            OpKind::ConstRefAddr(_) => op.result.clone(),
            _ => None,
        })
        .expect("ConstRefAddr dest");
    assert_eq!(
        crate::model::FunctionGraph::concretetype_of(&cell),
        crate::model::ConcreteType::Signed,
        "llmemory.Address getkind is int"
    );
    assert!(
        matches!(
            cell.concretetype(),
            Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Address)
        ),
        "OnceLock<fn> address is Address, got {:?}",
        cell.concretetype()
    );
    assert_eq!(
        nullary_calls_ending(&graph, "CELL"),
        0,
        "OnceLock<fn> Global must not residualize: {graph:?}"
    );
    assert!(
        call_paths_in(&graph).iter().all(
            |segments| segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        ),
        "Opaque OnceLock<fn> must not wrap ConstRefAddr: {graph:?}"
    );
}

/// A Struct-kind `OnceLock<fn>` still emits a bare ConstRefAddr: the type
/// lane runs before `refs_static_struct_root` would wrap the address as
/// `SomeInstance(OnceLock)`.
#[test]
fn oncelock_fn_struct_global_does_not_cast_instance() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = oncelock_fn_global_llbc(oncelock_fn_struct_kind(), Some(oncelock_fn_value_layout()));
    let graph = super::lower_function_with_static_addrs(&llbc, "read_cell", fnptrcell_refs())
        .expect("Struct OnceLock<fn> Global lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    assert!(
        call_paths_in(&graph).iter().all(
            |segments| segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        ),
        "OnceLock<fn> must not wrap ConstRefAddr as an instance: {graph:?}"
    );
}

fn usize_scalar() -> serde_json::Value {
    serde_json::json!({"Scalar": {"Integer": {"Unsigned": "Usize"}}})
}

fn oncelock_usize_global_llbc() -> majit_charon_reader::Llbc {
    let generics = serde_json::json!({
        "regions": [],
        "types": [usize_scalar()],
        "const_generics": [],
        "trait_refs": []
    });
    let cell_ty = serde_json::json!({"Adt": {"id": 0, "generics": generics}});
    let kind = serde_json::json!({"Struct": [
        {"name": "once", "ty": {"Tuple": []}, "attr_info": null},
        {"name": "value", "ty": usize_scalar(), "attr_info": null}
    ]});
    let funs = vec![serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "read_cell"], "", true),
        "signature": {"is_unsafe": false, "inputs": [], "output": cell_ty},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": cell_ty}
            ]},
            "body": [{
                "statements": [{
                    "span": span_json(),
                    "kind": {"Assign": [
                        {"kind": {"Local": 0}, "ty": cell_ty},
                        {"Use": [{"Read": {
                            "kind": {"Global": {"generics": empty_generics(), "id": 1}},
                            "ty": cell_ty
                        }}, "No"]}
                    ]}
                }],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })];
    let globals = vec![
        serde_json::Value::Null,
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "CELL"], "", true),
            "global_kind": "Static",
            "ty": cell_ty
        }),
    ];
    llbc_with_types_funs_globals(
        "fixture",
        vec![oncelock_fn_type_decl(
            kind,
            Some(oncelock_fn_value_layout()),
        )],
        funs,
        globals,
    )
}

/// A Struct `OnceLock<usize>` singleton Global's refs row is the
/// initialized Python object (`bookkeeper.immutablevalue`), not the
/// container. `refs_static_struct_root` must not wrap it as
/// `SomeInstance(OnceLock)` or `get_or_init` loses its ConstRefAddr
/// receiver.
#[test]
fn oncelock_usize_struct_global_does_not_cast_instance() {
    super::register_foldable_const_lits(Vec::new());
    let graph = super::lower_function_with_static_addrs(
        &oncelock_usize_global_llbc(),
        "read_cell",
        fnptrcell_refs(),
    )
    .expect("Struct OnceLock<usize> Global lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    assert!(
        call_paths_in(&graph).iter().all(|segments| {
            segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        }),
        "OnceLock<usize> must not wrap ConstRefAddr as an instance: {graph:?}"
    );
}

fn std_once_lock_unstructured_get_or_init(def_id: u64) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": item_meta_json(
            &["std", "sync", "once_lock", "<Impl>", "get_or_init"],
            "",
            false
        ),
        "signature": {"is_unsafe": false, "inputs": [], "output": {"Tuple": []}},
        "body": {"Unstructured": {
            "span": span_json(),
            "locals": {"arg_count": 0, "locals": [
                {"index": 0, "name": null, "span": span_json(), "ty": {"Tuple": []}}
            ]},
            "body": [{
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span_json(), "kind": "Return"}
            }]
        }}
    })
}

fn oncelock_usize_harvested_std_global_llbc() -> majit_charon_reader::Llbc {
    let generics = serde_json::json!({
        "regions": [],
        "types": [usize_scalar()],
        "const_generics": [],
        "trait_refs": []
    });
    let cell_ty = serde_json::json!({"Adt": {"id": 0, "generics": generics}});
    let funs = vec![
        serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "read_cell"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": cell_ty},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": cell_ty}
                ]},
                "body": [{
                    "statements": [{
                        "span": span_json(),
                        "kind": {"Assign": [
                            {"kind": {"Local": 0}, "ty": cell_ty},
                            {"Use": [{"Read": {
                                "kind": {"Global": {"generics": empty_generics(), "id": 1}},
                                "ty": cell_ty
                            }}, "No"]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        }),
        std_once_lock_unstructured_get_or_init(1),
    ];
    let globals = vec![
        serde_json::Value::Null,
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "CELL"], "", true),
            "global_kind": "Static",
            "ty": cell_ty
        }),
    ];
    llbc_with_types_funs_globals(
        "fixture",
        vec![oncelock_fn_type_decl(
            serde_json::json!({"Struct": [
                {"name": "once", "ty": {"Tuple": []}, "attr_info": null},
                {"name": "value", "ty": usize_scalar(), "attr_info": null}
            ]}),
            Some(oncelock_fn_value_layout()),
        )],
        funs,
        globals,
    )
}

/// Harvested `std` FunDecl bodies make `OnceLock<usize>` a foreign
/// generic struct (`foreign_body_generic_struct`), so
/// `tyref_class_root_with` would wrap the refs address as
/// `SomeInstance(OnceLock)`. The OnceLock Global lane must emit the
/// bare `ConstRefAddr` first (`bookkeeper.py Bookkeeper.immutablevalue`).
#[test]
fn oncelock_usize_harvested_std_global_does_not_cast_instance() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = oncelock_usize_harvested_std_global_llbc();
    assert!(
        llbc.has_body_under_crate_root("std"),
        "fixture must harvest a std unstructured body"
    );
    let graph = super::lower_function_with_static_addrs(&llbc, "read_cell", fnptrcell_refs())
        .expect("harvested-std OnceLock<usize> Global lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    assert!(
        call_paths_in(&graph).iter().all(|segments| {
            segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        }),
        "harvested std OnceLock<usize> must not wrap ConstRefAddr: {graph:?}"
    );
}

fn oncelock_usize_get_or_init_llbc() -> majit_charon_reader::Llbc {
    let generics = serde_json::json!({
        "regions": [],
        "types": [usize_scalar()],
        "const_generics": [],
        "trait_refs": []
    });
    let cell_ty = serde_json::json!({"Adt": {"id": 0, "generics": generics}});
    let cell_ref = serde_json::json!({"Ref": ["Erased", cell_ty, "Shared"]});
    let usize_ref = serde_json::json!({"Ref": ["Erased", usize_scalar(), "Shared"]});
    let init_ty = unit_fn_ptr();
    let empty = empty_generics();
    let funs = vec![
        serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "read_cell"], "", true),
            "signature": {"is_unsafe": false, "inputs": [init_ty], "output": usize_ref},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 1, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ref},
                    {"index": 1, "name": "init", "span": span_json(), "ty": init_ty},
                    {"index": 2, "name": "cell", "span": span_json(), "ty": cell_ty}
                ]},
                "body": [
                    {"statements": [{
                        "span": span_json(),
                        "kind": {"Assign": [
                            {"kind": {"Local": 2}, "ty": cell_ty},
                            {"Use": [{"Read": {
                                "kind": {"Global": {"generics": empty.clone(), "id": 1}},
                                "ty": cell_ty
                            }}, "No"]}
                        ]}
                    }],
                    "kind": "Regular",
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {"kind": {"Fun": 1}, "generics": empty}},
                            "args": [
                                {"Read": {"kind": {"Local": 2}, "ty": cell_ref}},
                                {"Read": {"kind": {"Local": 1}, "ty": init_ty}}
                            ]}, "dest": {"kind": {"Local": 0}, "ty": usize_ref},
                        "target": 1,
                        "on_unwind": 2
                    }}}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": "Return"}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": "UnwindResume"}}
                ]
            }}
        }),
        {
            let mut get_or_init = std_once_lock_unstructured_get_or_init(1);
            get_or_init["signature"] = serde_json::json!({
                "is_unsafe": false,
                "inputs": [cell_ref, init_ty],
                "output": usize_ref
            });
            get_or_init
        },
    ];
    let globals = vec![
        serde_json::Value::Null,
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "CELL"], "", true),
            "global_kind": "Static",
            "ty": cell_ty
        }),
    ];
    llbc_with_types_funs_globals(
        "fixture",
        vec![oncelock_fn_type_decl(
            serde_json::json!({"Struct": [
                {"name": "once", "ty": {"Tuple": []}, "attr_info": null},
                {"name": "value", "ty": usize_scalar(), "attr_info": null}
            ]}),
            Some(oncelock_fn_value_layout()),
        )],
        funs,
        globals,
    )
}

/// `ONCE.get_or_init(f)` on a refs-keyed `OnceLock<usize>` is
/// `Bookkeeper.immutablevalue` of the registered object
/// (`cache.py Cache.getorbuild`). The harvested-std wrap must not
/// hide the ConstRefAddr receiver from the fold.
#[test]
fn oncelock_usize_get_or_init_folds_to_prebuilt() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = oncelock_usize_get_or_init_llbc();
    assert!(llbc.has_body_under_crate_root("std"));
    let graph = super::lower_function_with_static_addrs(&llbc, "read_cell", fnptrcell_refs())
        .expect("OnceLock<usize>::get_or_init lowers");
    assert_eq!(const_ref_addrs(&graph), vec![0x1111]);
    assert!(
        call_paths_in(&graph)
            .iter()
            .all(|segments| !segments.iter().any(|s| s == "get_or_init")),
        "get_or_init must fold to the refs row: {graph:?}"
    );
    assert!(
        call_paths_in(&graph).iter().all(|segments| {
            segments.last().map(String::as_str) != Some("__cast_instance_intrinsic")
        }),
        "get_or_init receiver must stay a bare ConstRefAddr: {graph:?}"
    );
}

fn oncelock_fn_get_llbc() -> majit_charon_reader::Llbc {
    oncelock_fn_get_llbc_with_layout(Some(oncelock_fn_value_layout()))
}

fn oncelock_fn_get_llbc_with_layout(
    layout: Option<serde_json::Value>,
) -> majit_charon_reader::Llbc {
    let cell = oncelock_fn_adt();
    let cell_ref = serde_json::json!({"Ref": ["Erased", cell, "Shared"]});
    let option_fn = serde_json::json!({
        "Adt": {
            "id": 1,
            "generics": {
                "regions": [],
                "types": [i64_fn_ptr()],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let generics = empty_generics();
    let funs = vec![
        serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "call_get"], "", true),
            "signature": {"is_unsafe": false, "inputs": [cell_ref], "output": option_fn},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 1, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": option_fn},
                    {"index": 1, "name": "cell", "span": span_json(), "ty": cell_ref}
                ]},
                "body": [
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                            "args": [{"Read": {"kind": {"Local": 1}, "ty": cell_ref}}]}, "dest": {"kind": {"Local": 0}, "ty": option_fn},
                        "target": 1,
                        "on_unwind": 2
                    }}}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": "Return"}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span_json(), "kind": "UnwindResume"}}
                ]
            }}
        }),
        serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["std", "sync", "once_lock", "<Impl>", "get"], "", false),
            "signature": {"is_unsafe": false, "inputs": [cell_ref], "output": option_fn},
            "body": "Opaque"
        }),
    ];
    llbc_with_types(
        "fixture",
        vec![
            oncelock_fn_type_decl(oncelock_fn_struct_kind(), layout),
            option_enum(1),
        ],
        funs,
    )
}

/// `OnceLock<fn>::get` is a signed word `raw_load` of `Ptr(FuncType)` at
/// the LLBC `value` field offset (`NicheIntNull::FnPtr`,
/// `history.getkind` `'int'`).
#[test]
fn oncelock_fn_get_lowers_to_signed_raw_load_of_func_ptr_at_value_offset() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = oncelock_fn_get_llbc();
    let graph = super::lower_function(&llbc, "call_get").expect("OnceLock::get lowers");
    let loads: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(op.kind, OpKind::RawLoad { .. }))
        .collect();
    assert_eq!(loads.len(), 1, "exactly one RawLoad: {graph:?}");
    match &loads[0].kind {
        OpKind::RawLoad {
            offset,
            item_ty,
            itemsize,
            is_item_signed,
            ..
        } => {
            assert_eq!(*item_ty, crate::model::ValueType::Int);
            assert_eq!(*itemsize, crate::layout::target_word_size());
            assert!(*is_item_signed, "fn-ptr load is signed kind 'i'");
            let offset_consts: Vec<_> = graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .filter_map(|op| match (&op.kind, op.result.as_ref()) {
                    (OpKind::ConstInt(n), Some(v)) if v.id() == offset.id() => Some(*n),
                    _ => None,
                })
                .collect();
            assert_eq!(
                offset_consts,
                vec![8],
                "value field offset from LLBC layout"
            );
        }
        other => panic!("expected RawLoad, got {other:?}"),
    }
    let dest = loads[0].result.as_ref().expect("RawLoad dest");
    match dest.concretetype() {
        Some(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Ptr(p))
            if matches!(
                p.TO,
                crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(_)
            ) => {}
        other => panic!("load dest must be Ptr(FuncType), got {other:?}"),
    }
    assert_eq!(
        crate::model::FunctionGraph::concretetype_of(dest),
        crate::model::ConcreteType::Signed,
        "history.getkind of Ptr(FuncType) is int"
    );
    assert_eq!(
        nullary_calls_ending(&graph, "get"),
        0,
        "OnceLock::get must not residualize: {graph:?}"
    );
}

/// Generic `OnceLock` leaves Charon `field_offsets` `chosen: null`.
/// `OnceLock` is `repr(Rust)` with a platform-dependent layout, so the
/// fold requires Charon's chosen `value` offset and leaves the Call.
#[test]
fn oncelock_fn_get_value_offset_without_chosen_field_offsets() {
    super::register_foldable_const_lits(Vec::new());
    let llbc = oncelock_fn_get_llbc_with_layout(None);
    let graph = super::lower_function(&llbc, "call_get")
        .expect("OnceLock::get residualizes when value offset is unchosen");
    let loads: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| matches!(op.kind, OpKind::RawLoad { .. }))
        .collect();
    assert_eq!(
        loads.len(),
        0,
        "must not fold without chosen offsets: {graph:?}"
    );
    assert!(
        call_paths_in(&graph)
            .iter()
            .any(|segments| segments.last().map(String::as_str) == Some("get")),
        "OnceLock::get Call must stay: {graph:?}"
    );
}

fn reader_fun_at(def_id: u64, path: &[&str], global_id: u64) -> serde_json::Value {
    let mut fun = reader_fun(path, global_id);
    fun["def_id"] = serde_json::json!(def_id);
    fun
}

fn static_global_with_idents(def_id: u64, idents: &[(&str, u64)]) -> serde_json::Value {
    serde_json::json!({
        "def_id": def_id,
        "item_meta": {
            "name": idents.iter().map(|(s, d)| serde_json::json!({"Ident": [s, d]})).collect::<Vec<_>>(),
            "span": span_json(),
            "source_text": "",
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false},
            "is_local": true
        },
        "global_kind": "Static",
        "ty": u32_ty()
    })
}

fn layout_struct_llbc(
    fields: &[(&str, serde_json::Value)],
    repr: serde_json::Value,
    size: Option<u64>,
    align: Option<u64>,
    field_offsets: Option<&[u64]>,
) -> Llbc {
    let field_json: Vec<_> = fields
        .iter()
        .map(|(name, ty)| serde_json::json!({"name": name, "ty": ty, "attr_info": null}))
        .collect();
    let mut value = serde_json::json!({
        "repr": repr,
        "variant_layouts": []
    });
    if let Some(size) = size {
        value["size"] = serde_json::json!(size);
    }
    if let Some(align) = align {
        value["align"] = serde_json::json!(align);
    }
    if let Some(offsets) = field_offsets {
        value["variant_layouts"] = serde_json::json!([{"field_offsets": offsets}]);
    }
    let decl = serde_json::json!({
        "def_id": 0,
        "item_meta": item_meta_json(&["fixture", "Pair"], "", true),
        "kind": {"Struct": field_json},
        "layout": [{"key": "fixture-one-target", "value": value}]
    });
    llbc_with_types("fixture", vec![decl], vec![])
}

fn adt0() -> serde_json::Value {
    serde_json::json!({"Adt": {"id": 0}})
}

#[test]
fn declaration_order_layout_requires_fixed_repr() {
    let u8_ty = scalar_ty("U8");
    let u64_ty = scalar_ty("U64");
    let rust_repr =
        serde_json::json!({"repr_algo": "Rust", "align_modif": null, "transparent": false});
    let c_repr = serde_json::json!({"repr_algo": "C", "align_modif": null, "transparent": false});

    let concrete = layout_struct_llbc(
        &[("a", u8_ty.clone()), ("b", u64_ty.clone())],
        rust_repr.clone(),
        Some(24),
        Some(8),
        Some(&[0, 8]),
    );
    assert_eq!(
        super::mem_size_align(&concrete, &adt0(), &[], 0),
        Some((24, 8)),
        "a concrete rustc layout wins for repr(Rust)"
    );

    let symbolic_rust = layout_struct_llbc(
        &[("a", u8_ty.clone()), ("b", u64_ty.clone())],
        rust_repr.clone(),
        None,
        None,
        None,
    );
    assert_eq!(super::mem_size_align(&symbolic_rust, &adt0(), &[], 0), None);
    assert_eq!(
        super::eval_offset_of(&symbolic_rust, &serde_json::json!([{"id": 0}, 0, 1])),
        None
    );

    let symbolic_c = layout_struct_llbc(
        &[("a", u8_ty.clone()), ("b", u64_ty.clone())],
        c_repr,
        None,
        None,
        None,
    );
    assert_eq!(
        super::mem_size_align(&symbolic_c, &adt0(), &[], 0),
        Some((16, 8))
    );
    assert_eq!(
        super::eval_offset_of(&symbolic_c, &serde_json::json!([{"id": 0}, 0, 0])),
        Some(0)
    );
    assert_eq!(
        super::eval_offset_of(&symbolic_c, &serde_json::json!([{"id": 0}, 0, 1])),
        Some(8)
    );

    let transparent = layout_struct_llbc(
        &[("inner", u64_ty.clone())],
        serde_json::json!({"repr_algo": "Rust", "align_modif": null, "transparent": true}),
        None,
        None,
        None,
    );
    assert_eq!(
        super::mem_size_align(&transparent, &adt0(), &[], 0),
        Some((8, 8))
    );

    let packed = layout_struct_llbc(
        &[("a", u8_ty), ("b", u64_ty)],
        serde_json::json!({"repr_algo": "C", "align_modif": {"Pack": 1}, "transparent": false}),
        None,
        None,
        None,
    );
    assert_eq!(super::mem_size_align(&packed, &adt0(), &[], 0), None);

    let rustc_offset = layout_struct_llbc(
        &[("a", scalar_ty("U8")), ("b", scalar_ty("U32"))],
        rust_repr,
        None,
        None,
        Some(&[0, 4]),
    );
    assert_eq!(
        super::eval_offset_of(&rustc_offset, &serde_json::json!([{"id": 0}, 0, 1])),
        Some(4),
        "a recorded field offset is used before the declaration-order walk"
    );
    assert_eq!(super::mem_size_align(&rustc_offset, &adt0(), &[], 0), None);
}

/// `fold_op_list` keeps any operation it cannot prove safe. Fold a
/// const-struct local only when every use is a field read.
#[test]
fn const_struct_fold_requires_field_reads_only() {
    use majit_charon_reader::ullbc::Unstructured;
    let span = || {
        serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let field_place = |i: u64, idx: u64| {
        serde_json::json!({
            "kind": {"Projection": [place_local(i), {"Field": [null, idx]}]},
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
    let accepts = |body: &Unstructured, dest: usize| {
        super::const_struct_uses_are_field_reads(&llbc, body, dest)
    };

    let reads = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [{"Read": field_place(1, 0)}, "No"]}]
    }))]);
    assert!(accepts(&reads, 1), "field reads only");

    let store = body_of(vec![stmt(serde_json::json!({
        "Assign": [field_place(1, 0), {"Use": [{"Read": place_local(2)}, "No"]}]
    }))]);
    assert!(!accepts(&store, 1), "projected store");

    let address = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {
            "Ref": {"place": place_local(1), "kind": "Shared", "ptr_metadata": null}
        }]
    }))]);
    assert!(!accepts(&address, 1), "address-taking");

    let whole = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(2), {"Discriminant": place_local(1)}]
    }))]);
    assert!(!accepts(&whole, 1), "whole-value consumer");

    let copy_then_field = body_of(vec![
        stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Read": place_local(1)}, "No"]}]
        })),
        stmt(serde_json::json!({
            "Assign": [place_local(0), {"Use": [{"Read": field_place(2, 0)}, "No"]}]
        })),
    ]);
    assert!(
        accepts(&copy_then_field, 1),
        "single-assignment copy then field read"
    );

    let copy_to_ret = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(0), {"Use": [{"Read": place_local(1)}, "No"]}]
    }))]);
    assert!(
        !accepts(&copy_to_ret, 1),
        "copy to the return place is a whole-value consumer"
    );

    assert!(
        !accepts(&reads, 0),
        "return place is a whole-value consumer"
    );
}
