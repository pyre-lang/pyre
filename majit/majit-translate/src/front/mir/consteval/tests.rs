use super::{
    DecodedConst, bitflags_trait_empty_const, decode_literal, is_class_pytype_assoc_const,
    primitive_float_const, scalar_to_const_value, static_key_segments, tyref_to_value_type,
};

use crate::flowspace::model::ConstValue;

use crate::model::{CallTarget, OpKind, ValueType};

use majit_charon_reader::{Llbc, ullbc::{NameSeg, TyRef}};

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
                    "args": [],
                    "dest": {"kind": {"Local": 0}, "ty": ty}
                },
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
                    "args": [],
                    "dest": {"kind": {"Local": 0}, "ty": dest_ty}
                },
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
                    "args": [],
                    "dest": {"kind": {"Local": 1}, "ty": ty}
                },
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
                        {"Copy": {"kind": {"Local": 1}, "ty": ty}}
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
                        {"Copy": {"kind": {"Global": {
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
                        {"Use": [{"Copy": {
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
                    "args": [{"Move": {"kind": {"Local": 1}, "ty": u8_ty}}],
                    "dest": {"kind": {"Local": 2}, "ty": u8_ty}
                },
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
                    "args": [{"Move": {"kind": {"Local": 4}, "ty": u8_ty}}],
                    "dest": {"kind": {"Local": 0}, "ty": u8_ty}
                },
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

    fn immutable_static_kind() -> serde_json::Value {
        serde_json::json!({
            "Static": {"is_mut": false, "is_safe": true, "is_thread_local": false}
        })
    }

    fn prebuilt_fn_ptr_ty() -> serde_json::Value {
        serde_json::json!({"FnPtr": {"inputs": [], "output": {"Tuple": []}}})
    }

    fn prebuilt_unit_ty() -> serde_json::Value {
        serde_json::json!({"Tuple": []})
    }

    fn prebuilt_adt_ty(id: u64) -> serde_json::Value {
        serde_json::json!({"Adt": {"id": id, "generics": empty_generics()}})
    }

    fn prebuilt_fn_def_const(fun_id: u64, ty: &serde_json::Value) -> serde_json::Value {
        serde_json::json!([{"FnDef": {"kind": {"Fun": fun_id}, "generics": empty_generics()}}, ty])
    }

    fn u8_field_struct(def_id: u64, path: &[&str], field: &str) -> serde_json::Value {
        let field_attr = serde_json::json!({
            "attributes": [], "inline": null, "rename": null, "public": true
        });
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta_json(path, "", true),
            "kind": {"Struct": [{
                "name": field,
                "ty": {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "attr_info": field_attr
            }]}
        })
    }

    fn types_only_llbc(type_decls: Vec<serde_json::Value>) -> Llbc {
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": type_decls,
                "fun_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("type fixture")
    }

    fn scalar_struct_static_llbc(
        type_path: &[&str],
        static_path: &[&str],
        reader_path: &[&str],
    ) -> Llbc {
        let adt = prebuilt_adt_ty(0);
        let reader = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(reader_path, "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": adt},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": adt}
                ]},
                "body": [{
                    "statements": [{
                        "span": span_json(),
                        "kind": {"Assign": [
                            {"kind": {"Local": 0}, "ty": adt},
                            {"Use": [{"Copy": {
                                "kind": {"Global": {"generics": empty_generics(), "id": 1}},
                                "ty": adt
                            }}, "No"]}
                        ]}
                    }],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [u8_field_struct(0, type_path, "bits")],
                "fun_decls": [reader],
                "global_decls": [
                    null,
                    {
                        "def_id": 1,
                        "item_meta": item_meta_json(static_path, "", true),
                        "global_kind": immutable_static_kind(),
                        "ty": adt
                    }
                ],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("scalar static fixture")
    }

    fn leaf_fun(def_id: u64, path: &[&str]) -> serde_json::Value {
        let ty = prebuilt_unit_ty();
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta_json(path, "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": ty},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": ty}
                ]},
                "body": [{"statements": [], "terminator": {"span": span_json(), "kind": "Return"}}]
            }}
        })
    }

    fn vtable_static_init(
        def_id: u64,
        path: &[&str],
        leaf_id: u64,
        vtable_ty: &serde_json::Value,
    ) -> serde_json::Value {
        let fn_ty = prebuilt_fn_ptr_ty();
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta_json(path, "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": vtable_ty},
            "src": {"GlobalInitializer": {"id": def_id}},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": vtable_ty},
                    {"index": 1, "name": null, "span": span_json(), "ty": fn_ty}
                ]},
                "body": [{
                    "statements": [
                        {"span": span_json(), "kind": {"Assign": [
                            {"kind": {"Local": 1}, "ty": fn_ty},
                            {"Use": [{"Const": prebuilt_fn_def_const(leaf_id, &fn_ty)}, "No"]}
                        ]}},
                        {"span": span_json(), "kind": {"Assign": [
                            {"kind": {"Local": 0}, "ty": vtable_ty},
                            {"Aggregate": [
                                {"Adt": [0, null]},
                                [{"Move": {"kind": {"Local": 1}, "ty": fn_ty}}]
                            ]}
                        ]}}
                    ],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        })
    }

    fn immutable_vtable_global(
        def_id: u64,
        path: &[&str],
        init: u64,
        vtable_ty: &serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta_json(path, "pub static TABLE: Vtable = Vtable { slot: leaf };", true),
            "global_kind": immutable_static_kind(),
            "ty": vtable_ty,
            "value": {"Value": [0, [{"Call": [{"kind": {"Fun": init}, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}, []]}, vtable_ty]]}
        })
    }

    fn vtable_fixture() -> Llbc {
        let vtable_ty = prebuilt_adt_ty(0);
        let fn_ty = prebuilt_fn_ptr_ty();
        let unit = prebuilt_unit_ty();
        let vtable_ref = serde_json::json!({"Ref": ["Erased", vtable_ty.clone(), "Shared"]});
        let field_attr = serde_json::json!({
            "attributes": [], "inline": null, "rename": null, "public": true
        });
        let vtable = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "Vtable"], "", true),
            "kind": {"Struct": [{
                "name": "slot",
                "ty": fn_ty,
                "attr_info": field_attr
            }]}
        });
        let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
        let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span_json(), "ty": ty});
        let slot_place = place(
            serde_json::json!({"Projection": [
                place(serde_json::json!({"Projection": [
                    place(serde_json::json!({"Local": 1}), &vtable_ref),
                    "Deref"
                ]}), &vtable_ty),
                {"Field": [null, 0]}
            ]}),
            &fn_ty,
        );
        let reader = reader_fun(&["fixture", "read_table"], 1);
        let field_reader = serde_json::json!({
            "def_id": 7,
            "item_meta": item_meta_json(&["fixture", "read_slot"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": fn_ty},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    local(0, &fn_ty)
                ]},
                "body": [{
                    "statements": [{
                        "span": span_json(),
                        "kind": {"Assign": [
                            place(serde_json::json!({"Local": 0}), &fn_ty),
                            {"Use": [{"Copy": place(
                                serde_json::json!({"Projection": [
                                    place(serde_json::json!({"Projection": [
                                        place(serde_json::json!({"Global": {"generics": empty_generics(), "id": 1}}), &vtable_ty),
                                        "Deref"
                                    ]}), &vtable_ty),
                                    {"Field": [null, 0]}
                                ]}),
                                &fn_ty
                            )}, "No"]}
                        ]}
                    }],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let caller = serde_json::json!({
            "def_id": 8,
            "item_meta": item_meta_json(&["fixture", "call_slot"], "", true),
            "signature": {"is_unsafe": false, "inputs": [vtable_ref.clone()], "output": unit},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 1, "locals": [
                    local(0, &unit),
                    local(1, &vtable_ref),
                    local(2, &fn_ty)
                ]},
                "body": [
                    {
                        "statements": [{
                            "span": span_json(),
                            "kind": {"Assign": [
                                place(serde_json::json!({"Local": 2}), &fn_ty),
                                {"Use": [{"Copy": slot_place}, "No"]}
                            ]}
                        }],
                        "terminator": {"span": span_json(), "kind": {"Call": {
                            "call": {
                                "func": {"Dynamic": {"Copy": place(
                                    serde_json::json!({"Local": 2}),
                                    &fn_ty
                                )}},
                                "args": [],
                                "dest": place(serde_json::json!({"Local": 0}), &unit)
                            },
                            "target": 1,
                            "on_unwind": 2
                        }}}
                    },
                    {"statements": [], "terminator": {"span": span_json(), "kind": "Return"}},
                    {"statements": [], "terminator": {"span": span_json(), "kind": "UnwindResume"}}
                ]
            }}
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [vtable],
                "fun_decls": [
                    reader,
                    null,
                    null,
                    vtable_static_init(3, &["fixture", "TABLE_A", "init"], 4, &vtable_ty),
                    leaf_fun(4, &["fixture", "leaf_a"]),
                    vtable_static_init(5, &["fixture", "TABLE_B", "init"], 6, &vtable_ty),
                    leaf_fun(6, &["fixture", "leaf_b"]),
                    field_reader,
                    caller
                ],
                "global_decls": [
                    null,
                    immutable_vtable_global(1, &["fixture", "TABLE_A"], 3, &vtable_ty),
                    immutable_vtable_global(2, &["fixture", "TABLE_B"], 5, &vtable_ty)
                ],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("vtable fixture parses")
    }

    fn op_kinds(graph: &crate::model::FunctionGraph) -> Vec<&OpKind> {
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .map(|op| &op.kind)
            .collect()
    }

    /// An immutable `static` Place::Global is the prebuilt constant
    /// (`bookkeeper.py immutablevalue`), not a residual call to the
    /// static's symbolic path.
    #[test]
    fn immutable_static_global_lowers_to_const_ref_addr() {
        let llbc = vtable_fixture();
        let graph = super::lower_function(&llbc, "read_table").expect("lower read_table");
        let addrs: Vec<i64> = op_kinds(&graph)
            .into_iter()
            .filter_map(|kind| match kind {
                OpKind::ConstRefAddr(addr) => Some(*addr),
                _ => None,
            })
            .collect();
        assert_eq!(addrs, vec![super::immutable_static_sentinel_addr(1)]);
        assert_eq!(
            nullary_calls_ending(&graph, "TABLE_A"),
            0,
            "immutable static must not residualize as FunctionPath: {graph:?}"
        );
    }

    /// A field of that static folds to the initializer's `FnDef`
    /// (`decoded_const_to_op` `__fn_const`).
    #[test]
    fn immutable_static_fnptr_field_folds_to_fn_const() {
        let llbc = vtable_fixture();
        let graph = super::lower_function(&llbc, "read_slot").expect("lower read_slot");
        let fn_consts: Vec<&[String]> = op_kinds(&graph)
            .into_iter()
            .filter_map(|kind| match kind {
                OpKind::Call { target, .. } => crate::model::fn_const_segments(target),
                _ => None,
            })
            .collect();
        assert!(
            fn_consts
                .iter()
                .any(|segments| segments.last().map(String::as_str) == Some("leaf_a")),
            "expected __fn_const of leaf_a, got {graph:?}"
        );
    }

    /// A call through a field of a type that only exists as immutable
    /// statics is `indirect_call` with the initializer `FnDef`s as
    /// `c_graphs` (`rpbc.py FunctionsPBCRepr` /
    /// `jtransform.py handle_regular_indirect_call`).
    #[test]
    fn immutable_static_fnptr_call_carries_closed_family() {
        let llbc = vtable_fixture();
        let graph = super::lower_function(&llbc, "call_slot").expect("lower call_slot");
        let families: Vec<Vec<String>> = op_kinds(&graph)
            .into_iter()
            .filter_map(|kind| match kind {
                OpKind::IndirectCall {
                    graphs: Some(graphs),
                    ..
                } => Some(
                    graphs
                        .iter()
                        .filter_map(|path| path.segments.last().cloned())
                        .collect(),
                ),
                _ => None,
            })
            .collect();
        assert_eq!(families.len(), 1, "one IndirectCall family: {graph:?}");
        let mut leaves = families[0].clone();
        leaves.sort();
        assert_eq!(leaves, vec!["leaf_a".to_string(), "leaf_b".to_string()]);
        let harvested = super::harvest_immutable_fields_from_immutable_statics(&llbc);
        let slot = harvested
            .get("Vtable")
            .or_else(|| harvested.get("fixture::Vtable"));
        assert_eq!(
            slot.map(|fields| fields
                .iter()
                .map(|(name, rank)| (name.as_str(), *rank))
                .collect::<Vec<_>>()),
            Some(vec![("slot", crate::model::ImmutableRank::Immutable)]),
            "harvested {harvested:?}"
        );
    }

    /// A Const `FnDef` dynamic call is `direct_call`
    /// (`jtransform.py handle_regular_indirect_call` when the funcbox
    /// is Const).
    #[test]
    fn const_fndef_dynamic_call_is_direct() {
        let fn_ty = prebuilt_fn_ptr_ty();
        let unit = prebuilt_unit_ty();
        let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
        let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span_json(), "ty": ty});
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "call_const"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": unit},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [local(0, &unit)]},
                "body": [
                    {
                        "statements": [],
                        "terminator": {"span": span_json(), "kind": {"Call": {
                            "call": {
                                "func": {"Dynamic": {"Const": prebuilt_fn_def_const(1, &fn_ty)}},
                                "args": [],
                                "dest": place(serde_json::json!({"Local": 0}), &unit)
                            },
                            "target": 1,
                            "on_unwind": 2
                        }}}
                    },
                    {"statements": [], "terminator": {"span": span_json(), "kind": "Return"}},
                    {"statements": [], "terminator": {"span": span_json(), "kind": "UnwindResume"}}
                ]
            }}
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [caller, leaf_fun(1, &["fixture", "leaf_direct"])],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("const FnDef fixture");
        let graph = super::lower_function(&llbc, "call_const").expect("lower call_const");
        let directs: Vec<_> = op_kinds(&graph)
            .into_iter()
            .filter_map(|kind| match kind {
                OpKind::Call {
                    target:
                        CallTarget::FunctionPath {
                            segments,
                            fun_decl_id,
                            ..
                        },
                    ..
                } if segments.last().map(String::as_str) == Some("leaf_direct") => {
                    Some(*fun_decl_id)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            directs,
            vec![Some(1)],
            "direct Call of the Const FnDef: {graph:?}"
        );
        assert_eq!(
            op_kinds(&graph)
                .iter()
                .filter(|kind| matches!(
                    kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().map(String::as_str) == Some("__dyn_call")
                ))
                .count(),
            0,
            "Const FnDef must not residualize as __dyn_call: {graph:?}"
        );
    }

    fn fieldless_enum_decl(def_id: u64, path: &[&str], n: usize) -> serde_json::Value {
        let variants: Vec<_> = (0..n)
            .map(|i| {
                serde_json::json!({
                    "name": format!("V{i}"),
                    "fields": [],
                    "discriminant": {"Unsigned": ["U8", i.to_string()]},
                })
            })
            .collect();
        let layouts: Vec<_> = (0..n)
            .map(|_| serde_json::json!({"field_offsets": []}))
            .collect();
        serde_json::json!({
            "def_id": def_id,
            "item_meta": item_meta_json(path, "", true),
            "kind": {"Enum": variants},
            "layout": [{
                "key": "test",
                "value": {
                    "size": 1,
                    "align": 1,
                    "variant_layouts": layouts,
                    "discriminator": {"Branch": {"offset": 0, "int_ty": {"Unsigned": "U8"}}}
                }
            }]
        })
    }

    fn option_of_enum_ty(option_id: u64, enum_id: u64) -> serde_json::Value {
        serde_json::json!({"Adt": {
            "id": option_id,
            "generics": {
                "regions": [],
                "types": [prebuilt_adt_ty(enum_id)],
                "const_generics": [],
                "trait_refs": []
            }
        }})
    }

    /// `Option<E>` over a dense fieldless E is E's scalar plus the
    /// reserved `None` tag (`tyref_to_field_layout_string`); it is not
    /// a boxed two-word aggregate.
    #[test]
    fn option_of_fieldless_enum_is_scalar_niche() {
        let option_decl = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["core", "option", "Option"], "", false),
            "kind": {"Enum": [
                {"name": "None", "fields": [], "discriminant": {"Unsigned": ["U8", "0"]}},
                {"name": "Some", "fields": [{
                    "name": null,
                    "is_positional": true,
                    "ty": prebuilt_adt_ty(1)
                }], "discriminant": {"Unsigned": ["U8", "1"]}}
            ]}
        });
        let disc_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "Isize"}}});
        let opt_ty = option_of_enum_ty(0, 1);
        let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
        let local = |i: u64, ty: &serde_json::Value| serde_json::json!({"index": i, "name": null, "span": span_json(), "ty": ty});
        let disc_fn = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "opt_disc"], "", true),
            "signature": {"is_unsafe": false, "inputs": [opt_ty.clone()], "output": disc_ty},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 1, "locals": [
                    local(0, &disc_ty),
                    local(1, &opt_ty)
                ]},
                "body": [{
                    "statements": [{
                        "span": span_json(),
                        "kind": {"Assign": [
                            place(serde_json::json!({"Local": 0}), &disc_ty),
                            {"Discriminant": place(serde_json::json!({"Local": 1}), &opt_ty)}
                        ]}
                    }],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [
                    option_decl.clone(),
                    fieldless_enum_decl(1, &["fixture", "Kind"], 3)
                ],
                "fun_decls": [disc_fn],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("option fixture");
        let opt_tyref: TyRef = serde_json::from_value(opt_ty).unwrap();
        assert_eq!(tyref_to_value_type(&opt_tyref, &llbc), ValueType::Int);
        let niche = super::tyref_option_fieldless_niche(&opt_tyref, &llbc)
            .expect("Option<fieldless> has a scalar niche");
        assert_eq!(niche.none_tag, 0xff);
        assert_eq!(niche.storage_ty, "u8");
        let graph = super::lower_function(&llbc, "opt_disc").expect("lower opt_disc");
        let kinds = op_kinds(&graph);
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, OpKind::ConstInt(0xff))),
            "None tag is the u8 storage max, got {graph:?}"
        );
        assert!(
            kinds.iter().any(|kind| matches!(
                kind,
                OpKind::BinOp { op, .. } if op == "ne"
            )),
            "discriminant of the niche Option is a scalar compare, got {graph:?}"
        );
        assert!(
            kinds.iter().all(|kind| !matches!(
                kind,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant"
            )),
            "must not box Option<fieldless> as an aggregate: {graph:?}"
        );
        let two = fieldless_enum_decl(1, &["fixture", "Kind"], 2);
        let two_file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [
                    option_decl,
                    two
                ],
                "fun_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let two_llbc =
            Llbc::from_slice(two_file.to_string().as_bytes()).expect("two-variant fixture");
        let two_niche =
            super::tyref_option_fieldless_niche(&opt_tyref, &two_llbc).expect("N=2 niche");
        assert_eq!(two_niche.none_tag, 2);
    }

    /// Interior mutability is a fact of the declared type tree
    /// (`UnsafeCell` / `Cell` / `sync::atomic`), not of the static's
    /// binding name. A struct of one integer with no cell in the tree
    /// is a prebuilt table; the same layout under an atomic/cell path
    /// is not.
    #[test]
    fn type_tree_interior_mutability_is_a_declared_fact() {
        let atomic = types_only_llbc(vec![u8_field_struct(
            0,
            &["core", "sync", "atomic", "AtomicBool"],
            "v",
        )]);
        assert!(super::type_has_interior_mutability(&atomic, 0));
        assert!(!super::type_is_prebuilt_table(&atomic, 0));

        let cell = types_only_llbc(vec![u8_field_struct(0, &["core", "cell", "Cell"], "value")]);
        assert!(super::type_has_interior_mutability(&cell, 0));
        assert!(!super::type_is_prebuilt_table(&cell, 0));

        let unsafecell = u8_field_struct(0, &["core", "cell", "UnsafeCell"], "value");
        let field_attr = serde_json::json!({
            "attributes": [], "inline": null, "rename": null, "public": true
        });
        let flag = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "Flag"], "", true),
            "kind": {"Struct": [{
                "name": "v",
                "ty": prebuilt_adt_ty(0),
                "attr_info": field_attr
            }]}
        });
        let nested = types_only_llbc(vec![unsafecell, flag]);
        assert!(super::type_has_interior_mutability(&nested, 1));
        assert!(!super::type_is_prebuilt_table(&nested, 1));

        let mask = types_only_llbc(vec![u8_field_struct(0, &["fixture", "Mask"], "bits")]);
        assert!(!super::type_has_interior_mutability(&mask, 0));
        assert!(super::type_is_prebuilt_table(&mask, 0));
    }

    /// `static TABLE: AtomicBool` is not a prebuilt constant value, but
    /// the binding still has an address to load through. A residual
    /// FunctionPath of the static's symbolic path is not that address.
    #[test]
    fn atomic_static_named_table_is_read_through_its_address() {
        let llbc = scalar_struct_static_llbc(
            &["core", "sync", "atomic", "AtomicBool"],
            &["fixture", "TABLE"],
            &["fixture", "read_table"],
        );
        let graph = super::lower_function(&llbc, "read_table").expect("lower read_table");
        let addrs: Vec<i64> = op_kinds(&graph)
            .into_iter()
            .filter_map(|kind| match kind {
                OpKind::ConstRefAddr(addr) => Some(*addr),
                _ => None,
            })
            .collect();
        assert_eq!(addrs, vec![super::immutable_static_sentinel_addr(1)]);
        assert_eq!(
            nullary_calls_ending(&graph, "TABLE"),
            0,
            "interior-mut static must not residualize as FunctionPath: {graph:?}"
        );
    }

    /// `static SYS_DEV_MODE: Mask` with no cell in the type tree is the
    /// prebuilt integer table. The binding name is the one that panicked
    /// when `AtomicBool` was folded; the type tree is what decides.
    #[test]
    fn integer_table_named_dev_mode_lowers_to_const_ref_addr() {
        let llbc = scalar_struct_static_llbc(
            &["fixture", "Mask"],
            &["fixture", "SYS_DEV_MODE"],
            &["fixture", "read_flag"],
        );
        let graph = super::lower_function(&llbc, "read_flag").expect("lower read_flag");
        let addrs: Vec<i64> = op_kinds(&graph)
            .into_iter()
            .filter_map(|kind| match kind {
                OpKind::ConstRefAddr(addr) => Some(*addr),
                _ => None,
            })
            .collect();
        assert_eq!(addrs, vec![super::immutable_static_sentinel_addr(1)]);
        assert_eq!(
            nullary_calls_ending(&graph, "SYS_DEV_MODE"),
            0,
            "prebuilt integer table must not residualize: {graph:?}"
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

    fn size_align_probe_llbc(type_decls: serde_json::Value) -> Llbc {
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [],
                "type_decls": type_decls,
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        Llbc::from_slice(doc.to_string().as_bytes()).expect("size-align fixture parses")
    }

    fn size_align_item_meta(leaf: &str) -> serde_json::Value {
        serde_json::json!({
            "name": [{"Ident": [leaf, 0]}],
            "span": span_json(),
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        })
    }

    #[test]
    fn size_align_of_thin_pointer_is_the_word() {
        let llbc = size_align_probe_llbc(serde_json::json!([]));
        let ty = serde_json::json!({"RawPtr": [usize_ty(), "Const"]});
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(super::size_align_of_tyexpr(&llbc, false, &ty), Some(word));
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(word));
    }

    #[test]
    fn size_align_of_pointer_to_tuple_is_the_word() {
        // The previous pointer lane required `builtin` to be absent, so a
        // pointer to a tuple declined. A thin pointer is one word
        // (`llmemory.sizeof`).
        let tuple = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": "Tuple",
            "generics": {"types": [usize_ty(), usize_ty()]}
        }});
        let llbc = size_align_probe_llbc(serde_json::json!([]));
        let ty = serde_json::json!({"RawPtr": [tuple, "Const"]});
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(super::size_align_of_tyexpr(&llbc, false, &ty), Some(word));
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(word));
    }

    #[test]
    fn align_of_tuple_is_max_of_field_aligns() {
        let llbc = size_align_probe_llbc(serde_json::json!([{
            "def_id": 0,
            "item_meta": size_align_item_meta("Tuple"),
            "kind": {"Struct": [
                {"name": null, "is_positional": true, "ty": usize_ty()},
                {"name": null, "is_positional": true, "ty": u32_ty()}
            ]}
        }]));
        let ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": "Tuple",
            "generics": {"types": [usize_ty(), u32_ty()]}
        }});
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(word));
        assert_eq!(
            super::size_align_of_tyexpr(&llbc, false, &ty),
            Some(word * 2)
        );
    }

    #[test]
    fn align_of_unresolved_struct_of_pointers_is_the_word() {
        // Charon records no chosen layout. Alignment is still the max of
        // the field types: a pointer is one word (`llmemory.sizeof`).
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let llbc = size_align_probe_llbc(serde_json::json!([{
            "def_id": 0,
            "item_meta": size_align_item_meta("Foreign"),
            "kind": {"Struct": [
                {"name": "p", "ty": ptr}
            ]}
        }]));
        let ty = serde_json::json!({"Adt": {"id": 0, "builtin": null, "generics": {"types": []}}});
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(word));
        assert_eq!(super::size_align_of_tyexpr(&llbc, false, &ty), Some(word));
    }

    #[test]
    fn align_of_generic_struct_instantiated_at_pointer_is_the_word() {
        // `GcEntries<K, V>` is a generic decl; `align_of::<GcEntries<P, P>>()`
        // names that decl with use-site args. Alignment is the max of the
        // substituted fields (`llmemory.sizeof`).
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let tvar = |index: u64| serde_json::json!({"TypeVar": {"Bound": [0, index]}});
        let llbc = size_align_probe_llbc(serde_json::json!([{
            "def_id": 0,
            "item_meta": size_align_item_meta("GcEntries"),
            "kind": {"Struct": [
                {"name": "length", "ty": usize_ty()},
                {"name": "key", "ty": tvar(0)},
                {"name": "value", "ty": tvar(1)}
            ]}
        }]));
        let ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "types": [ptr.clone(), ptr],
                "regions": [],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(word));
    }

    #[test]
    fn align_of_generic_struct_with_zst_array_field_is_the_word() {
        // `GcEntries<K, V> { length, items: [Entry<K, V>; 0] }`. The array
        // length is a const, not a type; alignment is still the element
        // alignment after substituting `K`/`V`.
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let tvar = |index: u64| serde_json::json!({"TypeVar": {"Bound": [0, index]}});
        let entry = serde_json::json!({"Adt": {
            "id": 1,
            "builtin": null,
            "generics": {
                "types": [tvar(0), tvar(1)],
                "regions": [],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let zst_array = serde_json::json!({"Array": [
            entry,
            [{"Integer": {"Unsigned": ["Usize", "0"]}}, usize_ty()],
            null
        ]});
        let llbc = size_align_probe_llbc(serde_json::json!([
            {
                "def_id": 0,
                "item_meta": size_align_item_meta("GcEntries"),
                "kind": {"Struct": [
                    {"name": "length", "ty": usize_ty()},
                    {"name": "items", "ty": zst_array}
                ]}
            },
            {
                "def_id": 1,
                "item_meta": size_align_item_meta("Entry"),
                "kind": {"Struct": [
                    {"name": "key", "ty": tvar(0)},
                    {"name": "value", "ty": tvar(1)}
                ]}
            }
        ]));
        let ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "types": [ptr.clone(), ptr],
                "regions": [],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(word));
    }

    #[test]
    fn align_of_instantiated_pointer_folds_to_the_word() {
        use crate::model::{CallTarget, OpKind};
        // A monomorphized `align_of::<T>()` stores `T` on the FunDecl
        // Instantiated segment (`ItemMeta::instantiation`); the call
        // site generics are empty.
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Const"]});
        let align_of = serde_json::json!({
            "def_id": 1,
            "item_meta": {
                "name": [
                    {"Ident": ["core", 0]},
                    {"Ident": ["mem", 0]},
                    {"Ident": ["align_of", 0]},
                    {"Instantiated": {
                        "params": [],
                        "skip_binder": {
                            "regions": [],
                            "types": [ptr],
                            "const_generics": [],
                            "trait_refs": []
                        }
                    }}
                ],
                "span": span_json(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": false
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": "Opaque"
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "read_align"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 1},
                                "generics": empty_generics()
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [caller, align_of],
                "type_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let graph = super::lower_function(&llbc, "read_align").expect("lower read_align");
        let word = crate::layout::target_word_size() as i64;
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter()
                .any(|op| matches!(&op.kind, OpKind::ConstInt(n) if *n == word)),
            "instantiated align_of of a pointer must fold to the word, ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|s| s == "align_of")
            )),
            "align_of must not residualize, ops={ops:?}"
        );
    }

    #[test]
    fn spec_body_align_of_generic_struct_folds_from_instantiation() {
        use crate::front::clause_spec::substituted_unstructured;
        use crate::model::{CallTarget, OpKind};
        // `alloc_entries<K, V>` calls `align_of::<GcEntries<K, V>>()`. A
        // `cachedgraph` copy (`specialize.py`) is a separate graph whose
        // types are concrete; the leftover TypeVars in that call must fold
        // through `size_align_const_from_tyexpr`.
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let tvar0 = serde_json::json!({"TypeVar": {"Bound": [0, 0]}});
        let wrapper_ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [tvar0],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let align_of = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["core", "mem", "align_of"], "", false),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": "Opaque"
        });
        let read_align = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "read_align"], "", true),
            "generics": {
                "regions": [],
                "types": [{"index": 0, "name": "T", "variance": "Invariant"}],
                "const_generics": [],
                "trait_clauses": []
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 1},
                                "generics": {
                                    "regions": [],
                                    "types": [wrapper_ty],
                                    "const_generics": [],
                                    "trait_refs": []
                                }
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [read_align, align_of],
                "type_decls": [{
                    "def_id": 0,
                    "item_meta": size_align_item_meta("GcEntries"),
                    "kind": {"Struct": [
                        {"name": "length", "ty": usize_ty()},
                        {"name": "key", "ty": {"TypeVar": {"Bound": [0, 0]}}}
                    ]}
                }],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let fd = llbc.fn_by_id(0).expect("read_align");
        let types = [ptr];
        let body = substituted_unstructured(fd, &llbc, &[], &types, &[])
            .expect("substituted unstructured body");
        let accum = super::AccumulatorFacts::build(&llbc, &body);
        let empty_set = std::collections::HashSet::new();
        let empty_gc_struct_ids = std::collections::HashSet::new();
        let empty_map = std::collections::HashMap::new();
        let root_state = super::RootStackState::new(&llbc);
        let root_stack = super::RootStackAnalyzer::new(&llbc, &root_state);
        let graph = super::lower_unstructured_with_static_addrs_and_attrs(
            &llbc,
            fd,
            &body,
            crate::HostStaticAddrs::default(),
            &[],
            &empty_map,
            &crate::front::semantic::StructFieldRegistry::default(),
            &empty_set,
            &empty_set,
            &empty_gc_struct_ids,
            super::no_immutable_fields(),
            &empty_set,
            accum.has_builder,
            &accum,
            &root_stack,
            None,
            true,
            None,
            Some((types.as_slice(), [].as_slice())),
        )
        .expect("lower spec body");
        let word = crate::layout::target_word_size() as i64;
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter()
                .any(|op| matches!(&op.kind, OpKind::ConstInt(n) if *n == word)),
            "spec-body align_of::<GcEntries<ptr>>() must fold to the word, ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|s| s == "align_of")
            )),
            "align_of must not residualize from a spec body, ops={ops:?}"
        );
    }

    #[test]
    fn align_of_codeobject_uses_the_layout_probe() {
        // `w_code_const` calls `align_of::<CodeObject<ConstantData>>()`.
        // Charon leaves that struct's `chosen` layout null and a field
        // is `C::Name` (`TraitType`). The probe is `llmemory.sizeof`.
        let llbc = size_align_probe_llbc(serde_json::json!([{
            "def_id": 0,
            "item_meta": item_meta_json(
                &["rustpython_compiler_core", "bytecode", "CodeObject"],
                "",
                false
            ),
            "kind": {"Struct": [
                {"name": "source_path", "ty": {"TraitType": [
                    {
                        "kind": {"Clause": {"Bound": [0, 0]}},
                        "trait_decl_ref": {
                            "regions": [],
                            "skip_binder": {"id": 0, "generics": empty_generics()}
                        }
                    },
                    0,
                    empty_generics()
                ]}}
            ]}
        }]));
        let ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": empty_generics()
        }});
        let expected = crate::codeobject_layout::exact_layout_for("bytecode::CodeObject")
            .and_then(|layout| layout.align)
            .expect("host CodeObject layout");
        assert_eq!(
            super::size_align_of_tyexpr(&llbc, true, &ty),
            Some(expected)
        );
        assert!(
            super::size_align_of_tyexpr(&llbc, false, &ty).is_some(),
            "size_of::<CodeObject>() must fold"
        );
    }

    #[test]
    fn size_of_empty_struct_is_zero() {
        // `gc_alloc_storage_box::<Global>()`: `alloc::alloc::Global` is a
        // ZST. Alignment 1, size 0 (`llmemory.sizeof`).
        let llbc = size_align_probe_llbc(serde_json::json!([{
            "def_id": 0,
            "item_meta": size_align_item_meta("Global"),
            "kind": {"Struct": []}
        }]));
        let ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": empty_generics()
        }});
        assert_eq!(super::size_align_of_tyexpr(&llbc, false, &ty), Some(0));
        assert_eq!(super::size_align_of_tyexpr(&llbc, true, &ty), Some(1));
    }

    #[test]
    fn w_code_const_align_of_codeobject_does_not_residualize() {
        use crate::model::{CallTarget, OpKind};
        // Real `w_code_const` shape: `align_of::<CodeObject<ConstantData>>()`,
        // FunDecl still `core::mem::align_of`, T on the call generics.
        let codeobject = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [{"Adt": {
                    "id": 1,
                    "builtin": null,
                    "generics": empty_generics()
                }}],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let align_of = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["core", "mem", "align_of"], "", false),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": "Opaque"
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "w_code_const"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 1},
                                "generics": {
                                    "regions": [],
                                    "types": [codeobject],
                                    "const_generics": [],
                                    "trait_refs": []
                                }
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [caller, align_of],
                "type_decls": [
                    {
                        "def_id": 0,
                        "item_meta": item_meta_json(
                            &["rustpython_compiler_core", "bytecode", "CodeObject"],
                            "",
                            false
                        ),
                        "kind": {"Struct": [
                            {"name": "flags", "ty": usize_ty()}
                        ]}
                    },
                    {
                        "def_id": 1,
                        "item_meta": item_meta_json(
                            &["rustpython_compiler_core", "bytecode", "ConstantData"],
                            "",
                            false
                        ),
                        "kind": {"Enum": []}
                    }
                ],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let graph = super::lower_function(&llbc, "w_code_const").expect("lower w_code_const");
        let expected = crate::codeobject_layout::exact_layout_for("bytecode::CodeObject")
            .and_then(|layout| layout.align)
            .expect("host CodeObject layout") as i64;
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter()
                .any(|op| matches!(&op.kind, OpKind::ConstInt(n) if *n == expected)),
            "w_code_const align_of::<CodeObject<_>>() must fold, ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().is_some_and(|s| s == "align_of")
            )),
            "align_of must not residualize from w_code_const, ops={ops:?}"
        );
    }

    #[test]
    fn size_align_of_struct_with_free_typevar_field_uses_owner_generics() {
        // Charon type_decls spell a struct parameter as `TypeVar::Free`
        // as well as `Bound[0, i]`. `llmemory.sizeof` of the instantiated
        // Adt must still read `generics.types[i]`.
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let ty = serde_json::json!({"Adt": {
            "id": 0,
            "builtin": null,
            "generics": {
                "regions": [],
                "types": [ptr],
                "const_generics": [],
                "trait_refs": []
            }
        }});
        let llbc = size_align_probe_llbc(serde_json::json!([{
            "def_id": 0,
            "item_meta": size_align_item_meta("GcEntries"),
            "kind": {"Struct": [
                {"name": "length", "ty": usize_ty()},
                {"name": "key", "ty": {"TypeVar": {"Free": 0}}}
            ]}
        }]));
        let word = crate::layout::target_word_size() as u64;
        assert_eq!(
            super::size_align_of_tyexpr(&llbc, true, &ty),
            Some(word),
            "align_of::<GcEntries<ptr>>() with Free typevar must be the word"
        );
    }

    #[test]
    fn nonspec_call_with_concrete_types_enqueues_cachedgraph() {
        // `dict_grow::<ObjectKey, …>` is an inherent generic: concrete
        // type arguments, empty `trait_refs`. `default_specialize` still
        // builds `FunctionDesc.cachedgraph`, so the call site names the
        // spec copy and the generic body (with `align_of::<GcEntries<K,V>>`)
        // is not the Regular candidate.
        use crate::front::clause_spec::SpecQueue;
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let alloc_like = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "alloc_like"], "", true),
            "generics": {
                "regions": [],
                "types": [{"index": 0, "name": "T", "variance": "Invariant"}],
                "const_generics": [],
                "trait_clauses": []
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "caller"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 1},
                                "generics": {
                                    "regions": [],
                                    "types": [ptr],
                                    "const_generics": [],
                                    "trait_refs": []
                                }
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [caller, alloc_like],
                "type_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let fd = llbc.fn_by_id(0).expect("caller");
        let body = fd.unstructured().expect("caller unstructured");
        let accum = super::AccumulatorFacts::build(&llbc, &body);
        let empty_set = std::collections::HashSet::new();
        let empty_gc_struct_ids = std::collections::HashSet::new();
        let empty_map = std::collections::HashMap::new();
        let root_state = super::RootStackState::new(&llbc);
        let root_stack = super::RootStackAnalyzer::new(&llbc, &root_state);
        let queue = std::cell::RefCell::new(SpecQueue::new());
        let graph = super::lower_unstructured_with_static_addrs_and_attrs(
            &llbc,
            fd,
            &body,
            crate::HostStaticAddrs::default(),
            &[],
            &empty_map,
            &crate::front::semantic::StructFieldRegistry::default(),
            &empty_set,
            &empty_set,
            &empty_gc_struct_ids,
            super::no_immutable_fields(),
            &empty_set,
            accum.has_builder,
            &accum,
            &root_stack,
            Some(&queue),
            false,
            None,
            None,
        )
        .expect("lower caller");
        let req = queue.borrow_mut().pop().expect("cachedgraph enqueued");
        assert!(
            req.leaf.starts_with("alloc_like__spec_"),
            "spec leaf, got {}",
            req.leaf
        );
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s.starts_with("alloc_like__spec_"))
            )),
            "call site must name the cachedgraph copy, ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s == "alloc_like")
            )),
            "unspecialized generic must not remain the Regular callee, ops={ops:?}"
        );
    }

    #[test]
    fn opaque_local_crate_generic_call_names_the_cachedgraph() {
        // `eval_loop_jit` calls `execute_opcode_step` as an opaque FunDecl
        // in the jit crate. `instance_leaf` still names the spec path so
        // `graphs_from` finds the copy the interpreter crate extracted.
        use crate::front::clause_spec::SpecQueue;
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let alloc_like = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["owner", "alloc_like"], "", true),
            "generics": {
                "regions": [],
                "types": [{"index": 0, "name": "T", "variance": "Invariant"}],
                "const_generics": [],
                "trait_clauses": []
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": "Opaque"
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "caller"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Fun": 1},
                                "generics": {
                                    "regions": [],
                                    "types": [ptr],
                                    "const_generics": [],
                                    "trait_refs": []
                                }
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [caller, alloc_like],
                "type_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let fd = llbc.fn_by_id(0).expect("caller");
        let body = fd.unstructured().expect("caller unstructured");
        let accum = super::AccumulatorFacts::build(&llbc, &body);
        let empty_set = std::collections::HashSet::new();
        let empty_gc_struct_ids = std::collections::HashSet::new();
        let empty_map = std::collections::HashMap::new();
        let root_state = super::RootStackState::new(&llbc);
        let root_stack = super::RootStackAnalyzer::new(&llbc, &root_state);
        let lower = |queue: &std::cell::RefCell<SpecQueue>| {
            super::lower_unstructured_with_static_addrs_and_attrs(
                &llbc,
                fd,
                &body,
                crate::HostStaticAddrs::default(),
                &[],
                &empty_map,
                &crate::front::semantic::StructFieldRegistry::default(),
                &empty_set,
                &empty_set,
                &empty_gc_struct_ids,
                super::no_immutable_fields(),
                &empty_set,
                accum.has_builder,
                &accum,
                &root_stack,
                Some(queue),
                false,
                None,
                None,
            )
            .expect("lower caller")
        };
        let foreign = std::cell::RefCell::new(SpecQueue::new());
        let graph = lower(&foreign);
        assert!(
            foreign.borrow_mut().pop().is_none(),
            "a foreign opaque generic stays the template path"
        );
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s == "alloc_like")
            )),
            "foreign opaque must keep the template callee, ops={ops:?}"
        );
        crate::local_crates::with_local_crate_root("owner", || {
            let queue = std::cell::RefCell::new(SpecQueue::new());
            let graph = lower(&queue);
            let req = queue.borrow_mut().pop().expect("cachedgraph enqueued");
            assert!(
                req.leaf.starts_with("alloc_like__spec_"),
                "spec leaf, got {}",
                req.leaf
            );
            assert_eq!(req.fn_path, "owner::alloc_like");
            let ops: Vec<_> = graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .collect();
            assert!(
                ops.iter().any(|op| matches!(
                    &op.kind,
                    crate::model::OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().is_some_and(|s| s.starts_with("alloc_like__spec_"))
                )),
                "local-crate opaque must name the cachedgraph copy, ops={ops:?}"
            );
        });
    }

    #[test]
    fn nonspec_trait_call_with_concrete_impl_enqueues_cachedgraph() {
        // `unwrapped_add` is a Trait method whose impl is generic in K, V.
        // A non-spec body that `CallKind::Trait`s a resolved `TraitImpl`
        // must still `cachedgraph` (`default_specialize`), or BFS compiles
        // the trait-declaration default body and residualizes
        // `align_of::<GcEntries<K, V>>()`.
        use crate::front::clause_spec::SpecQueue;
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let insert = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "insert"], "", true),
            "generics": {
                "regions": [],
                "types": [
                    {"index": 0, "name": "K", "variance": "Invariant"},
                    {"index": 1, "name": "V", "variance": "Invariant"}
                ],
                "const_generics": [],
                "trait_clauses": []
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let impl_ref = serde_json::json!({
            "kind": {"TraitImpl": {"id": 0, "generics": {
                "regions": [],
                "types": [ptr.clone(), ptr.clone()],
                "const_generics": [],
                "trait_refs": []
            }}}
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "caller"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Trait": [impl_ref, 0, 1]},
                                "generics": {
                                    "regions": [],
                                    "types": [],
                                    "const_generics": [],
                                    "trait_refs": []
                                }
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [caller, insert],
                "type_decls": [],
                "global_decls": [],
                "trait_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta_json(&["fixture", "Set"], "", true),
                    "methods": [{"name": "insert"}]
                }],
                "trait_impls": [{
                    "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                    "impl_trait": {"id": 0, "generics": {
                        "regions": [],
                        "types": [{"RawPtr": [u32_ty(), "Mut"]}, {"RawPtr": [u32_ty(), "Mut"]}],
                        "const_generics": [],
                        "trait_refs": []
                    }},
                    "implied_trait_refs": []
                }]
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let fd = llbc.fn_by_id(0).expect("caller");
        let body = fd.unstructured().expect("caller unstructured");
        let accum = super::AccumulatorFacts::build(&llbc, &body);
        let empty_set = std::collections::HashSet::new();
        let empty_gc_struct_ids = std::collections::HashSet::new();
        let empty_map = std::collections::HashMap::new();
        let root_state = super::RootStackState::new(&llbc);
        let root_stack = super::RootStackAnalyzer::new(&llbc, &root_state);
        let queue = std::cell::RefCell::new(SpecQueue::new());
        let graph = super::lower_unstructured_with_static_addrs_and_attrs(
            &llbc,
            fd,
            &body,
            crate::HostStaticAddrs::default(),
            &[],
            &empty_map,
            &crate::front::semantic::StructFieldRegistry::default(),
            &empty_set,
            &empty_set,
            &empty_gc_struct_ids,
            super::no_immutable_fields(),
            &empty_set,
            accum.has_builder,
            &accum,
            &root_stack,
            Some(&queue),
            false,
            None,
            None,
        )
        .expect("lower caller");
        let req = queue.borrow_mut().pop().expect("cachedgraph enqueued");
        assert!(
            req.leaf.starts_with("insert__spec_"),
            "spec leaf, got {}",
            req.leaf
        );
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s.starts_with("insert__spec_"))
            )),
            "trait call site must name the cachedgraph copy, ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s == "insert")
            )),
            "unspecialized trait default must not remain the callee, ops={ops:?}"
        );
    }

    #[test]
    fn nonspec_trait_call_without_override_enqueues_default_cachedgraph() {
        // `OpcodeStepExecutor::build_list` is a provided default; `impl for
        // PyFrame` does not override it. `default_specialize` still
        // `FunctionDesc.cachedgraph`s that default at `Self = PyFrame`, or
        // the call stays `[OpcodeStepExecutor, build_list]` and the inner
        // `opcode_build_list::<H>` residualizes with a symbolic fnaddr.
        use crate::front::clause_spec::SpecQueue;
        let ptr = serde_json::json!({"RawPtr": [u32_ty(), "Mut"]});
        let provided = serde_json::json!({
            "def_id": 1,
            "item_meta": item_meta_json(&["fixture", "provided"], "", true),
            "generics": {
                "regions": [],
                "types": [{"index": 0, "name": "Self", "variance": "Unknown"}],
                "const_generics": [],
                "trait_clauses": [{
                    "clause_id": 0,
                    "span": span_json(),
                    "origin": "TraitSelf",
                    "trait_": {"regions": [], "skip_binder": {"id": 0, "generics": {
                        "regions": [],
                        "types": [{"TypeVar": {"Bound": [0, 0]}}],
                        "const_generics": [],
                        "trait_refs": []
                    }}}
                }]
            },
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "src": {"TraitDefault": {"trait_ref": {"id": 0}, "item_id": 0}},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let impl_ref = serde_json::json!({
            "kind": {"TraitImpl": {"id": 0, "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }}},
            "trait_decl_ref": {"regions": [], "skip_binder": {"id": 0, "generics": {
                "regions": [],
                "types": [ptr.clone()],
                "const_generics": [],
                "trait_refs": []
            }}}
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta_json(&["fixture", "caller"], "", true),
            "signature": {"is_unsafe": false, "inputs": [], "output": usize_ty()},
            "body": {"Unstructured": {
                "span": span_json(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span_json(), "ty": usize_ty()}
                ]},
                "body": [{
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": {"Call": {
                        "call": {
                            "func": {"Regular": {
                                "kind": {"Trait": [impl_ref, 0]},
                                "generics": {
                                    "regions": [],
                                    "types": [],
                                    "const_generics": [],
                                    "trait_refs": []
                                }
                            }},
                            "args": [],
                            "dest": {"kind": {"Local": 0}, "ty": usize_ty()}
                        },
                        "target": 2,
                        "on_unwind": 1
                    }}}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "UnwindResume"}
                }, {
                    "statements": [],
                    "terminator": {"span": span_json(), "kind": "Return"}
                }]
            }}
        });
        let doc = serde_json::json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "fun_decls": [caller, provided],
                "type_decls": [],
                "global_decls": [],
                "trait_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta_json(&["fixture", "Set"], "", true),
                    "methods": [{
                        "params": {
                            "regions": [],
                            "types": [],
                            "const_generics": [],
                            "trait_clauses": []
                        },
                        "skip_binder": {
                            "name": "provided",
                            "default": {"id": 1, "generics": {
                                "regions": [],
                                "types": [],
                                "const_generics": [],
                                "trait_refs": []
                            }}
                        },
                        "kind": {"TraitMethod": [0, 0]}
                    }]
                }],
                "trait_impls": [{
                    "def_id": 0,
                    "methods": [],
                    "impl_trait": {"id": 0, "generics": {
                        "regions": [],
                        "types": [ptr],
                        "const_generics": [],
                        "trait_refs": []
                    }},
                    "implied_trait_refs": []
                }]
            }
        });
        let llbc = Llbc::from_slice(doc.to_string().as_bytes()).expect("fixture parses");
        let fd = llbc.fn_by_id(0).expect("caller");
        let body = fd.unstructured().expect("caller unstructured");
        let accum = super::AccumulatorFacts::build(&llbc, &body);
        let empty_set = std::collections::HashSet::new();
        let empty_gc_struct_ids = std::collections::HashSet::new();
        let empty_map = std::collections::HashMap::new();
        let root_state = super::RootStackState::new(&llbc);
        let root_stack = super::RootStackAnalyzer::new(&llbc, &root_state);
        let queue = std::cell::RefCell::new(SpecQueue::new());
        let graph = super::lower_unstructured_with_static_addrs_and_attrs(
            &llbc,
            fd,
            &body,
            crate::HostStaticAddrs::default(),
            &[],
            &empty_map,
            &crate::front::semantic::StructFieldRegistry::default(),
            &empty_set,
            &empty_set,
            &empty_gc_struct_ids,
            super::no_immutable_fields(),
            &empty_set,
            accum.has_builder,
            &accum,
            &root_stack,
            Some(&queue),
            false,
            None,
            None,
        )
        .expect("lower caller");
        let req = queue
            .borrow_mut()
            .pop()
            .expect("default cachedgraph enqueued");
        assert!(
            req.leaf.starts_with("provided__spec_"),
            "spec leaf, got {}",
            req.leaf
        );
        let ops: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s.starts_with("provided__spec_"))
            )),
            "trait default call site must name the cachedgraph copy, ops={ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().is_some_and(|s| s == "provided")
            )),
            "unspecialized trait default must not remain the callee, ops={ops:?}"
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
    let impl_seg = NameSeg::Other(serde_json::json!({"Impl": {"Trait": 1}}));
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
                        {"Use": [{"Copy": {
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
                            "args": [{"Copy": {"kind": {"Local": 1}, "ty": cell_ref}}],
                            "dest": {"kind": {"Local": 0}, "ty": option_fn}
                        },
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
                        {"Use": [{"Copy": {
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
                        {"Use": [{"Copy": {
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
                            "args": [{"Copy": {"kind": {"Local": 1}, "ty": cell_ref}}],
                            "dest": {"kind": {"Local": 0}, "ty": option_fn}
                        },
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
        "Assign": [place_local(2), {"Use": [{"Copy": field_place(1, 0)}, "No"]}]
    }))]);
    assert!(accepts(&reads, 1), "field reads only");

    let store = body_of(vec![stmt(serde_json::json!({
        "Assign": [field_place(1, 0), {"Use": [{"Copy": place_local(2)}, "No"]}]
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
            "Assign": [place_local(2), {"Use": [{"Copy": place_local(1)}, "No"]}]
        })),
        stmt(serde_json::json!({
            "Assign": [place_local(0), {"Use": [{"Copy": field_place(2, 0)}, "No"]}]
        })),
    ]);
    assert!(
        accepts(&copy_then_field, 1),
        "single-assignment copy then field read"
    );

    let copy_to_ret = body_of(vec![stmt(serde_json::json!({
        "Assign": [place_local(0), {"Use": [{"Copy": place_local(1)}, "No"]}]
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
