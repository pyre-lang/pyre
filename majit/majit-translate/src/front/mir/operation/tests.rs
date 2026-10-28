use super::{cast_call_segments, cast_kind_is_raw_ptr, int_binop_needs_ptr_to_int};

use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};

#[test]
fn int_cast_action_narrows_inside_the_64_bit_int_box() {
    // `int_or_float_decode_int`: `value as u32 as i32 as i64`. With a
    // 32-bit `usize` (wasm32) the first two steps still narrow inside
    // the Int box.
    assert_eq!(
        super::int_cast_action(64, false, 32, true),
        Some(super::IntCastAction::And(0xffff_ffff))
    );
    assert_eq!(
        super::int_cast_action(32, true, 32, false),
        Some(super::IntCastAction::SignExt(4))
    );
    assert_eq!(
        super::int_cast_action(32, false, 64, false),
        Some(super::IntCastAction::Alias)
    );
    assert_eq!(
        super::int_cast_action(32, true, 64, false),
        Some(super::IntCastAction::IntMask)
    );
    assert_eq!(
        super::int_cast_action(64, false, 64, true),
        Some(super::IntCastAction::RUint)
    );
    assert_eq!(super::int_cast_action(128, false, 64, false), None);
}

#[test]
fn numeric_bank_crossing_casts_preserve_the_builtin_owner() {
    assert_eq!(
        cast_call_segments(&ValueType::Int, &ValueType::Float),
        Some(vec!["__builtin__".into(), "float".into()])
    );
    assert_eq!(
        cast_call_segments(&ValueType::Float, &ValueType::Int),
        Some(vec!["__builtin__".into(), "int".into()])
    );
}

/// A `Shl` folded at the 64-bit carrier can be wider than the local it is
/// assigned to — Rust does not treat shifted-out bits as an overflow, so
/// `200u8 << 1` is a legal const equal to 144, not 400.  rustc truncates
/// to the destination type and so must the fold.
/// `test_flatten.py test_force_cast_ints` on a 64-bit word.
#[test]
fn int_to_int_cast_follows_the_force_cast_table() {
    use super::{IntToIntCast, int_to_int_cast};
    const SCHAR: (u64, bool) = (1, false);
    const UCHAR: (u64, bool) = (1, true);
    const SHORT: (u64, bool) = (2, false);
    const USHORT: (u64, bool) = (2, true);
    const LONG: (u64, bool) = (8, false);
    const ULONG: (u64, bool) = (8, true);
    let noop = Some(IntToIntCast::Noop);
    let and = |m| Some(IntToIntCast::And(m));
    let signext = |n| Some(IntToIntCast::Signext(n));
    for (from, to, expected) in [
        (SCHAR, SCHAR, noop.clone()),
        (SCHAR, UCHAR, and(255)),
        (SCHAR, SHORT, noop.clone()),
        (SCHAR, USHORT, and(65535)),
        (SCHAR, LONG, noop.clone()),
        (SCHAR, ULONG, noop.clone()),
        (UCHAR, SCHAR, signext(1)),
        (UCHAR, UCHAR, noop.clone()),
        (UCHAR, SHORT, noop.clone()),
        (UCHAR, USHORT, noop.clone()),
        (UCHAR, LONG, noop.clone()),
        (UCHAR, ULONG, noop.clone()),
        (SHORT, SCHAR, signext(1)),
        (SHORT, UCHAR, and(255)),
        (SHORT, SHORT, noop.clone()),
        (SHORT, USHORT, and(65535)),
        (SHORT, LONG, noop.clone()),
        (SHORT, ULONG, noop.clone()),
        (USHORT, SCHAR, signext(1)),
        (USHORT, UCHAR, and(255)),
        (USHORT, SHORT, signext(2)),
        (USHORT, USHORT, noop.clone()),
        (USHORT, LONG, noop.clone()),
        (USHORT, ULONG, noop.clone()),
        (LONG, SCHAR, signext(1)),
        (LONG, UCHAR, and(255)),
        (LONG, SHORT, signext(2)),
        (LONG, USHORT, and(65535)),
        (LONG, LONG, noop.clone()),
        (LONG, ULONG, noop.clone()),
        (ULONG, SCHAR, signext(1)),
        (ULONG, UCHAR, and(255)),
        (ULONG, SHORT, signext(2)),
        (ULONG, USHORT, and(65535)),
        (ULONG, LONG, noop.clone()),
        (ULONG, ULONG, noop.clone()),
        // 32-bit halves: `i64 as u32` masks, `i64 as i32` re-extends.
        (LONG, (4, true), and(0xffff_ffff)),
        (LONG, (4, false), signext(4)),
        ((4, false), (4, true), and(0xffff_ffff)),
    ] {
        assert_eq!(int_to_int_cast(from, to, 8), expected, "{from:?} -> {to:?}");
    }
    // A 128-bit operand is the longlong leg, not covered here.
    assert_eq!(int_to_int_cast((16, false), UCHAR, 8), None);
}

#[test]
fn cast_kind_raw_ptr_recognizes_atom_and_object_forms() {
    assert!(cast_kind_is_raw_ptr(&serde_json::json!("RawPtr")));
    assert!(cast_kind_is_raw_ptr(
        &serde_json::json!({"RawPtr": {"Deduplicated": 0}})
    ));
    assert!(!cast_kind_is_raw_ptr(&serde_json::json!("Unsize")));
    assert!(!cast_kind_is_raw_ptr(&serde_json::json!({"Scalar": []})));
}

#[test]
fn int_binop_needs_ptr_to_int_only_for_int_ops_with_a_ref() {
    let ptr = ValueType::Ref(None);
    let int = ValueType::Int;
    assert!(int_binop_needs_ptr_to_int("lt", Some(&ptr), Some(&ptr)));
    assert!(int_binop_needs_ptr_to_int("le", Some(&ptr), Some(&int)));
    assert!(int_binop_needs_ptr_to_int("gt", Some(&int), Some(&ptr)));
    assert!(int_binop_needs_ptr_to_int("ge", Some(&ptr), None));
    assert!(int_binop_needs_ptr_to_int("mod", Some(&ptr), Some(&int)));
    assert!(int_binop_needs_ptr_to_int(
        "floordiv",
        Some(&int),
        Some(&ptr)
    ));
    assert!(int_binop_needs_ptr_to_int("div", Some(&ptr), Some(&ptr)));
    assert!(!int_binop_needs_ptr_to_int("eq", Some(&ptr), Some(&ptr)));
    assert!(!int_binop_needs_ptr_to_int("ne", Some(&ptr), Some(&ptr)));
    assert!(!int_binop_needs_ptr_to_int("lt", Some(&int), Some(&int)));
    assert!(!int_binop_needs_ptr_to_int("mod", Some(&int), Some(&int)));
    assert!(!int_binop_needs_ptr_to_int("add", Some(&ptr), Some(&ptr)));
}

/// Byte `direct_ptradd` stamps `CCHARP`; `rffi.cast` recasts the
/// result to the entry Ptr (`ForceCastEntry` → `force_cast`).
#[test]
fn restore_offset_pointer_type_recasts_ccharp_to_entry_ptr() {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    let mut graph = FunctionGraph::new("test");
    let bb = graph.startblock;
    let ptr = graph
        .push_op_var(bb, OpKind::ConstRefAddr(0x1000), true)
        .expect("pointer value");
    let entry_ty = super::lltype_for_direct_ptradd_pointer(None, &LowLevelType::Signed);
    ptr.set_concretetype(Some(entry_ty.clone()));
    let count = graph
        .push_op_var(bb, OpKind::ConstInt(8), true)
        .expect("count");
    let ptradd =
        super::push_direct_ptradd(&mut graph, bb, ptr, count, &super::item_lltype_for_size(1));
    let restored =
        super::restore_offset_pointer_type(&mut graph, bb, Some(entry_ty.clone()), ptradd.clone());
    let ccharp = super::lltype_for_direct_ptradd_pointer(None, &LowLevelType::Char);
    let ptradd_op = graph
        .block(bb)
        .operations
        .iter()
        .find(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("direct_ptradd"),
            _ => false,
        })
        .expect("direct_ptradd");
    match &ptradd_op.kind {
        OpKind::Call { args, .. } => {
            let crate::model::LinkArg::Value(operand) = &args[0] else {
                panic!("pointer operand must be a variable, got {:?}", args[0]);
            };
            assert_eq!(
                operand.concretetype().as_ref(),
                Some(&ccharp),
                "direct_ptradd operand is CCHARP"
            );
        }
        other => panic!("expected direct_ptradd call, got {other:?}"),
    }
    assert_eq!(
        ptradd_op.result.as_ref(),
        Some(&ptradd),
        "direct_ptradd result variable"
    );
    assert_eq!(
        ptradd.concretetype().as_ref(),
        Some(&ccharp),
        "direct_ptradd result is CCHARP"
    );
    assert_eq!(
        restored.concretetype().as_ref(),
        Some(&entry_ty),
        "restored result is the entry CArrayPtr(Signed)"
    );
    let ptradd_idx = graph
        .block(bb)
        .operations
        .iter()
        .position(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str) == Some("direct_ptradd"),
            _ => false,
        })
        .expect("direct_ptradd index");
    let after = &graph.block(bb).operations[ptradd_idx + 1..];
    assert_eq!(
        after.len(),
        1,
        "one recast after direct_ptradd, got {after:?}"
    );
    match &after[0].kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.last().map(String::as_str) == Some("cast")
            && segments.iter().any(|s| s == "rffi") =>
        {
            assert_eq!(after[0].result.as_ref(), Some(&restored));
            assert_eq!(
                restored.concretetype().as_ref(),
                Some(&entry_ty),
                "rffi.cast result is the entry CArrayPtr(Signed)"
            );
            assert_eq!(args.len(), 2, "rffi.cast(PTRTYPE, ptr), got {args:?}");
            assert!(
                matches!(
                    &args[0],
                    crate::model::LinkArg::Const(c)
                        if matches!(
                            c.value,
                            crate::flowspace::model::ConstValue::LowLevelType(_)
                        )
                ),
                "PTRTYPE is the entry LowLevelType, got {:?}",
                args[0]
            );
            assert_eq!(args[1], crate::model::LinkArg::Value(ptradd));
        }
        other => panic!("expected rffi.cast after direct_ptradd, got {other:?}"),
    }
    let following: Vec<&str> = after
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().map(String::as_str),
            _ => None,
        })
        .collect();
    assert!(
        !following.iter().any(|name| *name == "cast_pointer"
            || *name == "cast_ptr_to_int"
            || *name == "cast_int_to_ptr"),
        "no cast_pointer / int-cast recast after ptradd, got {following:?}"
    );
}

#[test]
fn restore_offset_pointer_type_keeps_ptradd_when_entry_is_signed() {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    let mut graph = FunctionGraph::new("test");
    let bb = graph.startblock;
    let ptr = graph
        .push_op_var(bb, OpKind::ConstRefAddr(0x1000), true)
        .expect("pointer value");
    ptr.set_concretetype(Some(LowLevelType::Signed));
    let count = graph
        .push_op_var(bb, OpKind::ConstInt(8), true)
        .expect("count");
    let ptradd =
        super::push_direct_ptradd(&mut graph, bb, ptr, count, &super::item_lltype_for_size(1));
    let restored = super::restore_offset_pointer_type(
        &mut graph,
        bb,
        Some(LowLevelType::Signed),
        ptradd.clone(),
    );
    assert_eq!(restored, ptradd, "no pointer annotation to restore");
}

#[test]
fn restore_offset_pointer_type_keeps_ccharp_for_gc_entry() {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr, PtrTarget, Struct};
    let mut graph = FunctionGraph::new("test");
    let bb = graph.startblock;
    let ptr = graph
        .push_op_var(bb, OpKind::ConstRefAddr(0x1000), true)
        .expect("pointer value");
    let entry_ty = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(Struct::gc("S", vec![("x".into(), LowLevelType::Signed)])),
    }));
    ptr.set_concretetype(Some(entry_ty.clone()));
    let count = graph
        .push_op_var(bb, OpKind::ConstInt(8), true)
        .expect("count");
    let ptradd =
        super::push_direct_ptradd(&mut graph, bb, ptr, count, &super::item_lltype_for_size(1));
    let restored =
        super::restore_offset_pointer_type(&mut graph, bb, Some(entry_ty), ptradd.clone());
    let ccharp = super::lltype_for_direct_ptradd_pointer(None, &LowLevelType::Char);
    assert_eq!(restored, ptradd, "GC entry plus byte offset stays CCHARP");
    assert_eq!(
        restored.concretetype().as_ref(),
        Some(&ccharp),
        "result remains CCHARP"
    );
}
