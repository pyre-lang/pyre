use crate::model::{CallTarget, OpKind, ValueType};

use majit_charon_reader::Llbc;

use crate::front::mir::tests::{graph_ops, range_contains_fixture};

/// A specialized copy's leaf matches its template's tail.
#[test]
fn a_spec_leaf_matches_its_template_tail() {
    use crate::front::mir::fmt_path_ends_with;
    let segs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let spec = segs(&["lltype", "malloc_typed__spec_W_X_0123456789abcdef"]);
    assert!(fmt_path_ends_with(&spec, &["lltype", "malloc_typed"]));
    assert!(!fmt_path_ends_with(&spec, &["lltype", "malloc"]));
    let inner = segs(&["malloc_typed__spec_X_0123456789abcdef", "new"]);
    assert!(!fmt_path_ends_with(&inner, &["malloc_typed", "new"]));
    let identity = segs(&[
        "vec",
        "Vec",
        "<impl core::ops::index::Index<alloc::vec::Vec<v0,v2>,v1>>",
        "index",
    ]);
    assert!(fmt_path_ends_with(&identity, &["vec", "Vec", "index"]));
    let collect = segs(&[
        "iter",
        "traits",
        "collect",
        "<Impl>",
        "<impl core::iter::traits::collect::IntoIterator<core::ops::range::Range<v0>>>",
        "into_iter",
    ]);
    assert!(fmt_path_ends_with(
        &collect,
        &["iter", "traits", "collect", "<Impl>", "into_iter"]
    ));
}

#[test]
fn owner_method_path_accepts_trait_impl_identity() {
    use crate::front::mir::fmt_path_ends_with_owner_method;
    let segs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let peq = &["core", "cmp", "PartialEq"][..];
    let owner = &["non_null", "NonNull"][..];
    let inherent = segs(&["ptr", "non_null", "NonNull", "eq"]);
    let identity = segs(&[
        "ptr",
        "non_null",
        "NonNull",
        "<impl core::cmp::PartialEq<core::ptr::non_null::NonNull<T>>>",
        "eq",
    ]);
    let charon_impl = segs(&["ptr", "non_null", "NonNull", "<Impl>", "ne"]);
    let foreign_eq = segs(&["ptr", "non_null", "NonNull", "<impl crate::MyEq>", "eq"]);
    let application_eq = segs(&[
        "ptr",
        "non_null",
        "NonNull",
        "<impl application::cmp::PartialEq<core::ptr::non_null::NonNull<T>>>",
        "eq",
    ]);
    let partial_ord = segs(&[
        "ptr",
        "non_null",
        "NonNull",
        "<impl core::cmp::PartialOrd<core::ptr::non_null::NonNull<T>>>",
        "eq",
    ]);
    let option_ext = segs(&["option", "Option", "<impl local::Ext>", "ne"]);
    assert!(fmt_path_ends_with_owner_method(&inherent, owner, peq, "eq"));
    assert!(fmt_path_ends_with_owner_method(&identity, owner, peq, "eq"));
    assert!(!fmt_path_ends_with_owner_method(
        &charon_impl,
        owner,
        peq,
        "ne"
    ));
    assert!(!fmt_path_ends_with_owner_method(
        &identity, owner, peq, "ne"
    ));
    assert!(!fmt_path_ends_with_owner_method(
        &segs(&["option", "Option", "eq"]),
        owner,
        peq,
        "eq"
    ));
    assert!(!fmt_path_ends_with_owner_method(
        &foreign_eq,
        owner,
        peq,
        "eq"
    ));
    assert!(!fmt_path_ends_with_owner_method(
        &application_eq,
        owner,
        peq,
        "eq"
    ));
    assert!(!fmt_path_ends_with_owner_method(
        &partial_ord,
        owner,
        peq,
        "eq"
    ));
    assert!(!fmt_path_ends_with_owner_method(
        &option_ext,
        &["option", "Option"],
        peq,
        "ne"
    ));
}

#[test]
fn decode_packed_format_pieces_matches_real_llbc_templates() {
    use super::decode_packed_format_pieces;

    // The fixtures below are the verbatim `[u8; N]` pieces buffers
    // charon lowers for these handler graphs (captured from the real
    // pyre-interpreter.ullbc). Each asserts the reconstructed pieces +
    // per-placeholder argument indices, i.e. the original format string.

    // `format!("stack underflow during {}", context)`
    let (pieces, indices) = decode_packed_format_pieces(&[
        23, 115, 116, 97, 99, 107, 32, 117, 110, 100, 101, 114, 102, 108, 111, 119, 32, 100, 117,
        114, 105, 110, 103, 32, 192, 0,
    ])
    .unwrap();
    assert_eq!(
        pieces,
        vec!["stack underflow during ".to_string(), String::new()]
    );
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0]
    );

    // `format!("{} indices must be integers or slices, not {}", ..)`
    let (pieces, indices) = decode_packed_format_pieces(&[
        192, 41, 32, 105, 110, 100, 105, 99, 101, 115, 32, 109, 117, 115, 116, 32, 98, 101, 32,
        105, 110, 116, 101, 103, 101, 114, 115, 32, 111, 114, 32, 115, 108, 105, 99, 101, 115, 44,
        32, 110, 111, 116, 32, 192, 0,
    ])
    .unwrap();
    assert_eq!(
        pieces,
        vec![
            String::new(),
            " indices must be integers or slices, not ".to_string(),
            String::new(),
        ]
    );
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0, 1]
    );

    // `format!("'{}' object does not support item assignment", ..)`
    let (pieces, indices) = decode_packed_format_pieces(&[
        1, 39, 192, 41, 39, 32, 111, 98, 106, 101, 99, 116, 32, 100, 111, 101, 115, 32, 110, 111,
        116, 32, 115, 117, 112, 112, 111, 114, 116, 32, 105, 116, 101, 109, 32, 97, 115, 115, 105,
        103, 110, 109, 101, 110, 116, 0,
    ])
    .unwrap();
    assert_eq!(
        pieces,
        vec![
            "'".to_string(),
            "' object does not support item assignment".to_string(),
        ]
    );
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0]
    );

    // `format!("__init__() should return None, not '{}'", ..)`
    let (pieces, indices) = decode_packed_format_pieces(&[
        36, 95, 95, 105, 110, 105, 116, 95, 95, 40, 41, 32, 115, 104, 111, 117, 108, 100, 32, 114,
        101, 116, 117, 114, 110, 32, 78, 111, 110, 101, 44, 32, 110, 111, 116, 32, 39, 192, 1, 39,
        0,
    ])
    .unwrap();
    assert_eq!(
        pieces,
        vec![
            "__init__() should return None, not '".to_string(),
            "'".to_string(),
        ]
    );
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0]
    );

    // `format!("can only concatenate {a} (not \"{b}\") to {a}", ..)` —
    // the `add`-operand error tail (descroperation.rs), where the first
    // argument is reused by the trailing placeholder.  The reuse packs a
    // `0xC8` + little-endian u16 index (`200, 0, 0`) pointing back at
    // argument 0, so `indices == [0, 1, 0]` with two distinct arguments.
    let (pieces, indices) = decode_packed_format_pieces(&[
        21, 99, 97, 110, 32, 111, 110, 108, 121, 32, 99, 111, 110, 99, 97, 116, 101, 110, 97, 116,
        101, 32, 192, 7, 32, 40, 110, 111, 116, 32, 34, 192, 6, 34, 41, 32, 116, 111, 32, 200, 0,
        0, 0,
    ])
    .unwrap();
    assert_eq!(
        pieces,
        vec![
            "can only concatenate ".to_string(),
            " (not \"".to_string(),
            "\") to ".to_string(),
            String::new(),
        ]
    );
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0, 1, 0]
    );

    // A template naming arguments 1 and 3 explicitly (`0xC8` + u16 index
    // each): three sequential `0xC0` then two explicit reuses.
    let (pieces, indices) =
        decode_packed_format_pieces(&[192, 192, 192, 200, 1, 0, 200, 3, 0, 0]).unwrap();
    assert_eq!(pieces.len(), 6);
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0, 1, 2, 1, 3]
    );
}

#[test]
fn decode_packed_format_pieces_bails_and_handles_edges() {
    use super::decode_packed_format_pieces;

    // A flags-bearing placeholder truncated before its four-byte flags
    // field, and the reserved 0x80 long-literal marker without a length,
    // must bail.
    assert_eq!(decode_packed_format_pieces(&[0xC1, 0]), None);
    assert_eq!(decode_packed_format_pieces(&[0x80, 0]), None);

    // Verbatim `format!("\\x{c:02x}")` template from the production
    // LLBC.  0xC3 selects flags+width; 0x69000020 is space fill,
    // sign-aware zero-pad, width-present, unknown alignment.
    let (pieces, placeholders) =
        decode_packed_format_pieces(&[2, b'\\', b'x', 0xC3, 0x20, 0, 0, 0x69, 2, 0, 0]).unwrap();
    assert_eq!(pieces, ["\\x".to_string(), String::new()]);
    assert_eq!(placeholders.len(), 1);
    assert!(placeholders[0].is_lower_hex_byte_02());

    // A literal length that overruns the buffer bails.
    assert_eq!(decode_packed_format_pieces(&[5, 65, 66, 0]), None);

    // A 0 byte that is not the final byte bails (terminator is last).
    assert_eq!(decode_packed_format_pieces(&[0, 1, 65, 0]), None);

    // A buffer that runs out without the 0x00 terminator is truncated
    // or wrong — bail rather than accept a malformed template.
    assert_eq!(decode_packed_format_pieces(&[2, 104, 105]), None);
    assert_eq!(decode_packed_format_pieces(&[2, 104, 105, 192]), None);

    // A `0xC8` reuse token whose 2-byte little-endian index runs past
    // the buffer end bails rather than reading out of bounds.
    assert_eq!(decode_packed_format_pieces(&[0xC8, 0]), None);
    assert_eq!(decode_packed_format_pieces(&[0xC8, 0, 0]), None);

    // Literal-only template (no placeholders) → single piece, no args.
    let (pieces, indices) = decode_packed_format_pieces(&[2, 104, 105, 0]).unwrap();
    assert_eq!(pieces, vec!["hi".to_string()]);
    assert!(indices.is_empty());

    // Two consecutive literal segments accumulate into one piece
    // (how a >127-byte literal is split); one placeholder follows.
    let (pieces, indices) = decode_packed_format_pieces(&[1, 97, 1, 98, 192, 0]).unwrap();
    assert_eq!(pieces, vec!["ab".to_string(), String::new()]);
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0]
    );
}

#[test]
fn read_array_literal_then_decode_pieces_end_to_end() {
    use super::{decode_packed_format_pieces, read_array_literal_elements, resolve_const_int};
    use crate::flowspace::model::Variable;
    use crate::model::{
        CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, ValueType,
    };

    // Build the pieces array the way the front-end lowers it: an
    // `Array` ctor followed by `__pos_i` FieldWrites of ConstUInt
    // bytes. Template for `format!("hi{}")` → pieces ["hi", ""],
    // one placeholder = bytes [2, 'h', 'i', 0xC0, 0].
    let mut graph = FunctionGraph::new("arraylit");
    let a = graph.create_block();
    let arr = Variable::new();
    graph.block_mut(a).operations.push(SpaceOperation {
        result: Some(arr.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor("Array"),
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Array".to_string())),
        },
    });
    let bytes = [2u64, 104, 105, 0xC0, 0];
    for (i, b) in bytes.iter().enumerate() {
        let v = Variable::new();
        graph.block_mut(a).operations.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::ConstUInt(*b),
        });
        graph.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: arr.clone(),
                field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                value: crate::model::LinkArg::Value(v),
                ty: ValueType::Ref(None),
            },
        });
    }

    let elements = read_array_literal_elements(&graph, &arr).expect("array elements");
    assert_eq!(elements.len(), 5);
    let decoded: Vec<u8> = elements
        .iter()
        .map(|v| resolve_const_int(&graph, v).expect("const int") as u8)
        .collect();
    assert_eq!(decoded, vec![2, 104, 105, 0xC0, 0]);
    let (pieces, indices) = decode_packed_format_pieces(&decoded).unwrap();
    assert_eq!(pieces, vec!["hi".to_string(), String::new()]);
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0]
    );

    // A non-Array producer (plain ConstInt) is rejected.
    assert_eq!(read_array_literal_elements(&graph, &elements[0]), None);
}

/// Build the `alloc::fmt::format` block shape a fieldless-enum `{:?}`
/// chain reaches after the `Display` collapsers decline — the format op
/// as the block's last op, closed by a single goto to a continuation
/// that consumes the rendered String — and assert the rewrite splits it
/// into a discriminant switch: one arm per variant emitting the constant
/// `"pre{ident}post"` message forwarded through the format-result slot,
/// plus a default arm raising to the exceptblock.
#[test]
fn rewrite_debug_enum_fmt_site_splits_into_discriminant_switch() {
    use super::{DebugEnumFmtCollapse, rewrite_debug_enum_fmt_site};
    use crate::model::{CallTarget, ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType};

    let mut graph = FunctionGraph::new("debug_enum_fmt");
    let bf = graph.startblock;

    // Bf: the enum value is live (an op result here), and the terminal
    // `alloc::fmt::format` produces the rendered String.
    let func = graph.push_op_var(bf, OpKind::ConstInt(0), true).unwrap();
    let fmt_args = graph
        .push_op_var(
            bf,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor("Arguments"),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let formatted = graph
        .push_op_var(
            bf,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: ["alloc", "fmt", "format"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![fmt_args]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();

    // Continuation consumes the String plus an unrelated live carried
    // value, so the rewrite must thread `carried` through each arm.
    let carried = graph.push_op_var(bf, OpKind::ConstInt(7), true).unwrap();
    let (bcont, cont_args) = graph.create_block_with_arg_vars(2);
    graph.set_return(bcont, Some(cont_args[0].clone()));
    graph.block_mut(bf).exits = vec![
        Link::from_variables(&graph, vec![formatted.clone(), carried], bcont, None)
            .with_prevblock(bf),
    ];

    let site = DebugEnumFmtCollapse {
        format_result: formatted.clone(),
        enum_value: func.clone(),
        pieces: vec![
            "intrinsic function ".to_string(),
            " not implemented".to_string(),
        ],
        by_discr: vec![(0, "Invalid".to_string()), (1, "Print".to_string())],
        // This test drives `rewrite_debug_enum_fmt_site` (the switch
        // build) directly on a graph with no residual chain, so the
        // delete / re-thread plan is empty.
        link_rewrites: Vec::new(),
        dead_results: Vec::new(),
        dead_bases: Vec::new(),
    };
    assert!(rewrite_debug_enum_fmt_site(&mut graph, &site));

    // The residual `alloc::fmt::format` is gone; Bf now switches on the
    // enum value directly (a fieldless enum is `Int`-valued, so its
    // value IS the discriminant — no `__discriminant` field read).
    let has_format = graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
        matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if super::fmt_path_ends_with(segments, &["fmt", "format"])
        )
    });
    assert!(!has_format, "residual alloc::fmt::format removed");
    let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
    assert!(
        matches!(&bf_block.exitswitch, Some(ExitSwitch::Value(v)) if v.id() == func.id()),
        "switch selects the enum value directly"
    );
    let disc_read = bf_block.operations.iter().any(
        |op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__discriminant"),
    );
    assert!(
        !disc_read,
        "no __discriminant field read (enum value is the tag)"
    );

    // Two variant arms + a default arm.
    assert_eq!(bf_block.exits.len(), 3, "two variant arms + default");

    // Each variant arm emits its `__str_const` message and forwards it
    // as the format-result slot into the continuation.
    let str_consts: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.first().map(String::as_str) == Some("__str_const") => {
                segments.get(1).cloned()
            }
            _ => None,
        })
        .collect();
    assert!(str_consts.contains(&"intrinsic function Invalid not implemented".to_string()));
    assert!(str_consts.contains(&"intrinsic function Print not implemented".to_string()));

    // Exactly one arm raises (the default `AssertionError`).
    let raises = graph
        .blocks
        .iter()
        .filter(|b| b.exits.iter().any(|l| l.target == graph.exceptblock))
        .count();
    assert_eq!(raises, 1, "the default arm raises to exceptblock");

    // The continuation still receives a String in its first slot from
    // every variant arm (the message var), and the carried value in the
    // second slot.
    let arm_exits: Vec<&Link> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.exits)
        .filter(|l| l.target == bcont)
        .collect();
    assert_eq!(
        arm_exits.len(),
        2,
        "both variant arms reach the continuation"
    );
    for l in arm_exits {
        assert_eq!(l.args.len(), 2, "continuation arity preserved");
        assert!(matches!(l.args[0], LinkArg::Value(_)));
    }
}

/// `ssa.py` `SSA_to_SSI` `record_used_var(block.exitswitch)`: when the
/// enum value is defined in a predecessor, the switch block's
/// exitswitch names that block's `Variable.copy()` local, not the
/// producer identity (`Block.renamevariables`).
#[test]
fn rewrite_debug_enum_fmt_site_renames_exitswitch_to_block_local() {
    use super::{DebugEnumFmtCollapse, rewrite_debug_enum_fmt_site};
    use crate::model::{CallTarget, ExitSwitch, FunctionGraph, Link, OpKind, ValueType};

    let mut graph = FunctionGraph::new("debug_enum_fmt_exitswitch_local");
    let root = graph.startblock;
    let func = graph.push_op_var(root, OpKind::ConstInt(0), true).unwrap();
    let bf = graph.create_block();
    graph.set_goto(root, bf, vec![]);

    let fmt_args = graph
        .push_op_var(
            bf,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor("Arguments"),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let formatted = graph
        .push_op_var(
            bf,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: ["alloc", "fmt", "format"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![fmt_args]),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .unwrap();
    let carried = graph.push_op_var(bf, OpKind::ConstInt(7), true).unwrap();
    let (bcont, cont_args) = graph.create_block_with_arg_vars(2);
    graph.set_return(bcont, Some(cont_args[0].clone()));
    graph.block_mut(bf).exits = vec![
        Link::from_variables(&graph, vec![formatted.clone(), carried], bcont, None)
            .with_prevblock(bf),
    ];

    let site = DebugEnumFmtCollapse {
        format_result: formatted,
        enum_value: func.clone(),
        pieces: vec![
            "intrinsic function ".to_string(),
            " not implemented".to_string(),
        ],
        by_discr: vec![(0, "Invalid".to_string()), (1, "Print".to_string())],
        link_rewrites: Vec::new(),
        dead_results: Vec::new(),
        dead_bases: Vec::new(),
    };
    assert!(rewrite_debug_enum_fmt_site(&mut graph, &site));

    match &graph.block(bf).exitswitch {
        Some(ExitSwitch::Value(sw)) => {
            assert_ne!(
                sw.id(),
                func.id(),
                "exitswitch must not keep the producer identity"
            );
            assert!(
                graph.variable_defined_in_block(bf, sw),
                "exitswitch use must be defined in the switch block"
            );
        }
        other => panic!("expected Value exitswitch, got {other:?}"),
    }
}

#[test]
fn emit_fmt_concat_builds_interleaved_str_add_fold() {
    use super::{FmtArg, FmtArgKind, FmtChain, FmtPlaceholder, emit_fmt_concat};
    use crate::flowspace::model::Variable;
    use crate::model::{CallTarget, FunctionGraph, OpKind};

    // `format!("a{}b{}c", x, y)` → pieces ["a","b","c"], two args.
    let mut graph = FunctionGraph::new("concat");
    let bb = graph.create_block();
    let x = Variable::new();
    let y = Variable::new();
    let chain = FmtChain {
        pieces: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        args: vec![
            FmtArg {
                value: x.clone(),
                kind: FmtArgKind::Display,
            },
            FmtArg {
                value: y.clone(),
                kind: FmtArgKind::Display,
            },
        ],
        placeholders: vec![
            FmtPlaceholder {
                arg_index: 0,
                flags: None,
                width: None,
                precision: None,
                width_indirect: false,
                precision_indirect: false,
            },
            FmtPlaceholder {
                arg_index: 1,
                flags: None,
                width: None,
                precision: None,
                width_indirect: false,
                precision_indirect: false,
            },
        ],
    };
    let result = emit_fmt_concat(&mut graph, bb, &chain);

    let ops = &graph.blocks.iter().find(|b| b.id == bb).unwrap().operations;
    // 3 literal `__str_const`s + 4 `add`s = 7 ops.
    assert_eq!(ops.len(), 7);

    let str_const_text = |i: usize| -> String {
        match &ops[i].kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.first().map(String::as_str) == Some("__str_const") => segments[1].clone(),
            other => panic!("op[{i}] not a __str_const: {other:?}"),
        }
    };
    let add_operands = |i: usize| -> (u64, u64) {
        match &ops[i].kind {
            OpKind::BinOp { op, lhs, rhs, .. } if op == "add" => (lhs.id(), rhs.id()),
            other => panic!("op[{i}] not an add: {other:?}"),
        }
    };
    let result_id = |i: usize| ops[i].result.as_ref().unwrap().id();

    assert_eq!(str_const_text(0), "a");
    assert_eq!(str_const_text(2), "b");
    assert_eq!(str_const_text(5), "c");
    // Fold chain: ("a" + x) + "b", then (+ y) + "c".
    assert_eq!(add_operands(1), (result_id(0), x.id()));
    assert_eq!(add_operands(3), (result_id(1), result_id(2)));
    assert_eq!(add_operands(4), (result_id(3), y.id()));
    assert_eq!(add_operands(6), (result_id(4), result_id(5)));
    assert_eq!(result.id(), result_id(6));
}

/// Anchor the `format!` collapse to the real lowered IR of
/// `stack_underflow_error` (= `type_error(format!("stack underflow
/// during {context}"))`). Ignored by default (loads the real LLBC); run
/// with `cargo test -p majit-translate --lib
/// real_stack_underflow_fmt_chain -- --ignored --nocapture`.
///
/// This grades the collapse's OUTPUT rather than calling
/// [`extract_fmt_chain`] on the returned graph, because that call is not
/// reachable here: `collapse_fmt_chains` runs unconditionally inside the
/// lowering pipeline and consumes the `alloc::fmt::format` call, so no
/// entry point hands back a graph still carrying one. The two facts the
/// recognizer's `FmtChain` stated — the literal piece, and one argument
/// rendered by `Display` — survive as ops, so they are asserted on the
/// ops. [`extract_fmt_chain`] itself is graded on synthetic graphs by
/// `extract_fmt_chain_recovers_pieces_and_args_cross_block`.
///
/// The expected shape is read from the artefact:
/// `stack_underflow_error` lowers to five ops — the `context: Str` input,
/// `__str_const("stack underflow during ")`, `str(context)`, their `add`,
/// and the `error::PyError::type_error` call. Re-take it from a dump if
/// the message text or the template changes.
#[test]
#[ignore]
fn real_stack_underflow_fmt_chain_collapses_to_str_concat() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "stack_underflow_error").expect("lower stack_underflow_error");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .collect();

    // The chain is consumed: no `alloc::fmt::format` call survives.
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if super::fmt_path_ends_with(segments, &["fmt", "format"])
        )),
        "the fmt chain must not residualize as an alloc::fmt::format call"
    );

    // The template's literal piece became a `__str_const`.
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments
                == &[
                    "__str_const".to_string(),
                    "stack underflow during ".to_string(),
                ]
        )),
        "expected a __str_const carrying the template's literal piece"
    );

    // The one Display argument became a native `str` render, concatenated
    // onto that literal.
    let render = ops
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::UnaryOp { op: name, .. } if name == "str" => op.result.clone(),
            _ => None,
        })
        .expect("expected a `str` render of the Display argument");
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::BinOp { op: name, rhs, .. } if name == "add" && rhs == &render
        )),
        "the rendered argument must be concatenated onto the literal piece"
    );
}

/// Anchor the `Arguments::from_str_nonconst` alias to the real lowered IR
/// of `constructor_args` (the `_collections` deque-iterator `__new__`
/// helper), whose cold error branches build no-placeholder messages
/// (`type_error(format!("must be collections.deque, not {name}"))` renders
/// its one placeholder through the multi-arg chain; the arity guard's
/// `format!("…got {}", n)` and the descriptor-preamble
/// `concat!(…)`-message emit the const-string `from_str_nonconst`
/// constructor in this non-`const` body).  The identity alias must drop
/// every `fmt::Arguments::from_str_nonconst` extern — none may survive as
/// a residual Call to wall the rtyper.  Ignored by default (loads the real
/// LLBC).
#[test]
#[ignore]
fn arguments_from_str_nonconst_alias_real_constructor_args() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "constructor_args").expect("lower constructor_args");

    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["Arguments", "from_str_nonconst"])
            )
        })
        .count();
    assert_eq!(
        residual, 0,
        "no residual Arguments::from_str_nonconst extern survives the identity alias"
    );
}

/// Anchor the `next()` fold to the real lowered IR of
/// `call_function_impl_result` — the generic-dispatch slice for-loop
/// `for &arg in args { pin_root(arg) }` that 16 census heads funnel
/// through.  The unregistered `next()` returns an opaque `Ref` the MIR
/// recasts to `Option<*mut PyObject>` via a trailing
/// `__cast_instance_intrinsic` narrow, so the raw call is not the block's
/// last op; `peel_recast_chain` must strip the recast and fold anyway.
/// After the fold no residual `slice::iter::Iter::next` call survives
/// and the native `[__iter_next]` op is present.  Ignored by default
/// (loads the real LLBC).
#[test]
#[ignore]
fn iter_next_fold_real_call_function_impl_result() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "call_function_impl_result")
        .expect("lower call_function_impl_result");

    let count_call_leaf = |leaf: &[&str]| {
        graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if super::fmt_path_ends_with(segments, leaf)
                )
            })
            .count()
    };
    // The residual `slice::iter::Iter::next` dispatch wall is gone.
    assert_eq!(
        count_call_leaf(&["slice", "iter", "Iter", "next"]),
        0,
        "residual slice-iter next removed by the fold"
    );
    // The native `[__iter_next]` marker op is present.
    let iter_next_ops = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.first().is_some_and(|s| s == "__iter_next")
            )
        })
        .count();
    assert!(
        iter_next_ops >= 1,
        "at least one native [__iter_next] op emitted"
    );
    // The `rooted_args` reload is spelled as a `for i in 0..n` +
    // `Vec::push` loop (not `.map().collect()`), so no residual
    // `iter::adapters::map::Map::collect` dispatch wall survives — the
    // deeper co-wall this graph exposed once the slice-iter wall fell.
    assert_eq!(
        count_call_leaf(&["iter", "adapters", "map", "Map", "collect"]),
        0,
        "no residual Map::collect reload wall"
    );
}

/// Representative interpreter loops iterate `&[*mut PyObject]`, whose
/// `Option<&*mut PyObject>` is folded to the identity-null niche shape:
/// `(opt is null) == false`.  This is still RPython's `ptr_nonzero`
/// predicate and must feed the same `next` + StopIteration rewrite as the
/// aggregate Option shape above.  The four roots here fan out to most of
/// the census's `slice::Iter::next` failures.  Ignored by default because
/// it loads the real interpreter LLBC.
#[test]
#[ignore]
fn iter_next_fold_real_identity_null_niche() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    for function in [
        "pyre_interpreter::call::call_builtin_code_positional",
        "pyre_interpreter::runtime_ops::build_map_from_refs",
        "pyre_interpreter::display::py_repr_wtf8",
        "pyre_interpreter::baseobjspace::mutated",
    ] {
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|e| panic!("lower {function}: {e}"));
        let count_call_leaf = |leaf: &[&str]| {
            graph
                .blocks
                .iter()
                .flat_map(|b| b.operations.iter())
                .filter(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                            if super::fmt_path_ends_with(segments, leaf)
                    )
                })
                .count()
        };
        assert_eq!(
            count_call_leaf(&["slice", "iter", "Iter", "next"]),
            0,
            "{function}: identity-null niche loop must not retain residual Iter::next"
        );
        assert!(
            count_call_leaf(&["__iter_next"]) >= 1,
            "{function}: identity-null niche loop must emit a native next op"
        );
    }
}

/// PyPy's deque item operations call `decode_index4(w_index, self)`: the
/// post-`__index__` length read is an ordinary read from the deque owner,
/// not a niladic callback.  Keep the Rust interpreter spelling equally
/// direct so Charon does not introduce a synthetic `FnOnce::call_once`
/// graph between `deque_index` and `deque_len`.
#[test]
#[ignore]
fn deque_index_reads_its_owner_without_a_fnonce_adapter() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let function = "pyre_interpreter::module::_collections::deque_index";
    let graph =
        super::lower_function(&llbc, function).unwrap_or_else(|e| panic!("lower {function}: {e}"));
    assert_eq!(
        graph.block(graph.startblock).inputargs.len(),
        2,
        "deque_index should receive only (index, self_obj)"
    );
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
    assert!(
        calls.iter().any(|segments| {
            super::fmt_path_ends_with(segments, &["_collections", "deque_len"])
        }),
        "deque_index must read the live deque length directly"
    );
    assert!(
        calls
            .iter()
            .all(|segments| !super::fmt_path_ends_with(segments, &["FnOnce", "call_once"])),
        "deque_index must not manufacture a closure call for its owner read"
    );
}

/// PyPy spells `W_MemoryView.copy` and `descr_toreadonly` as two concrete
/// view constructions.  Their Rust counterparts must likewise reach the
/// buffer-view allocator directly, without a shared closure-valued derive
/// helper that inserts `FnOnce::call_once` into the translated closure.
#[test]
#[ignore]
fn memoryview_derivations_are_concrete_graphs() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    for function in [
        "pyre_interpreter::builtins::w_memoryview_copy_derived",
        "pyre_interpreter::builtins::w_memoryview_readonly_derived",
    ] {
        let graph = super::lower_function(&llbc, function)
            .unwrap_or_else(|e| panic!("lower {function}: {e}"));
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
        assert!(
            calls
                .iter()
                .any(|segments| { super::fmt_path_ends_with(segments, &["bufferview_alloc"]) }),
            "{function}: concrete derivation must allocate its view"
        );
        assert!(
            calls
                .iter()
                .all(|segments| { !super::fmt_path_ends_with(segments, &["FnOnce", "call_once"]) }),
            "{function}: concrete derivation must not call a closure adapter"
        );
    }
}

/// The field-bearing primary driver const must lower exactly like the
/// zero-field unpack driver: one opaque constant receiver for each marker,
/// never a residual call to the const item's symbolic path.  Ignored by
/// default because it loads the extracted pyre-jit artefact; run with
/// `cargo test -p majit-translate --lib
/// pypyjitdriver_named_const_is_a_marker_receiver_constant -- --ignored`.
#[test]
#[ignore]
fn pypyjitdriver_named_const_is_a_marker_receiver_constant() {
    use crate::model::{CallTarget, OpKind};

    let llbc = Llbc::load(crate::runtime_names::artifacts::JIT_ULLBC).expect("load pyre-jit LLBC");
    let graph = super::lower_function(&llbc, "eval_loop_jit").expect("lower eval_loop_jit");
    let driver_path = ["pyre_jit", "eval", "pypyjitdriver"];

    assert!(
        !graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if args.is_empty()
                    && super::fmt_path_ends_with(segments, &driver_path)
            )),
        "the driver const path must not survive as a nullary call"
    );

    let marker_receivers: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::Method {
                        name,
                        receiver_root: Some(root),
                        ..
                    },
                args,
                ..
            } if root == "PyPyJitDriver"
                && matches!(name.as_str(), "jit_merge_point" | "can_enter_jit") =>
            {
                args.first()
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        marker_receivers.len(),
        2,
        "eval_loop_jit must retain both source marker calls after MIR lowering"
    );
    for receiver in marker_receivers {
        let (block_id, op_index) = super::resolve_to_producer_op(&graph, receiver)
            .expect("marker receiver has one producer");
        let producer = &graph
            .blocks
            .iter()
            .find(|block| block.id == block_id)
            .expect("producer block")
            .operations[op_index];
        assert!(
            matches!(
                producer.kind,
                OpKind::ConstRefAddr(super::JITDRIVER_NAMEDCONST_SENTINEL_ADDR)
            ),
            "marker receiver producer must be the driver sentinel: {producer:?}"
        );
    }
}

/// `formatting::char_arg` narrows its BigInt to `u32`, not `i64`.  The
/// i64-specific fits/discriminant synthesis must decline and leave the
/// original `TryFrom<&BigInt>` call residual.  Ignored by default because
/// it loads the real interpreter LLBC.
#[test]
#[ignore]
fn bigint_i64_try_from_real_declines_u32_result() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "char_arg").expect("lower char_arg");

    let calls: Vec<&[String]> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.as_slice()),
            _ => None,
        })
        .collect();
    assert!(
        calls.iter().any(|segments| {
            super::fmt_path_ends_with(segments, &["bigint", "<Impl>", "try_from"])
                || super::fmt_path_ends_with(segments, &["<Impl>", "try_from"])
        }),
        "u32::try_from(&BigInt) must remain a residual FunctionPath call"
    );
    assert!(
        !calls.iter().any(|segments| {
            segments.last().map(String::as_str) == Some("jit_bigint_to_i64_fits")
        }),
        "u32 destination must not use the i64 fits/discriminant lowering"
    );
}

/// PyPy's `ObjSpace.c_int_w` performs the same signed 32-bit bounds
/// check explicitly. Anchor Rust's `i32::try_from(i64)` spelling in both
/// the interpreter helper and the int-or-float list strategy: neither may
/// retain an opaque core call, and both must keep a runtime Result tag.
#[test]
#[ignore]
fn i64_to_i32_try_from_real_sites_lower_to_checked_result() {
    use crate::model::{CallTarget, OpKind};

    for (relpath, fname) in [
        ("/../../target/llbc/pyre-interpreter.ullbc", "index_c_int_w"),
        (
            "/../../target/llbc/pyre-object.ullbc",
            "int_or_float_encode_int",
        ),
    ] {
        let path = format!("{}{relpath}", env!("CARGO_MANIFEST_DIR"));
        let llbc = Llbc::load(path).unwrap_or_else(|e| panic!("load {fname} LLBC: {e}"));
        let graph =
            super::lower_function(&llbc, fname).unwrap_or_else(|e| panic!("lower {fname}: {e:?}"));
        assert!(
                !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if super::fmt_path_ends_with(
                            segments,
                            &["convert", "num", "<Impl>", "try_from"],
                        ))
                }),
                "{fname}: checked narrowing must not remain an opaque core call"
            );
        // `or`, not a sum: `c_int_w` writes
        // `value < INT_MIN or value > INT_MAX` (`baseobjspace.py`).
        for expected in ["lt", "gt", "or"] {
            assert!(
                graph
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .any(|op| { matches!(&op.kind, OpKind::BinOp { op, .. } if op == expected) }),
                "{fname}: missing checked-narrowing {expected} op"
            );
        }
        assert!(
            graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::FieldWrite { field, .. }
                        if field.name == "__discriminant")
            }),
            "{fname}: Result discriminant must remain runtime data"
        );
    }
}

/// `w_range_iter` keeps the word iterator only when
/// `start + length * step` still fits i64 (`i64::try_from(i128)`).
/// That shell is the same checked Result `c_int_w` writes, with
/// `longlongmask` on the Ok payload (`rbuiltin.py`).
#[test]
fn w_range_iter_i128_try_from_lowers_to_checked_result() {
    use crate::model::{CallTarget, OpKind};

    let path = format!(
        "{}/../../target/llbc/pyre-object.ullbc",
        env!("CARGO_MANIFEST_DIR")
    );
    let llbc = Llbc::load(path).unwrap_or_else(|e| panic!("load w_range_iter LLBC: {e}"));
    let graph = super::lower_function(&llbc, "w_range_iter")
        .unwrap_or_else(|e| panic!("lower w_range_iter: {e:?}"));
    assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(
                        segments,
                        &["convert", "num", "<Impl>", "try_from"],
                    ))
            }),
            "w_range_iter: i128 try_from must not remain an opaque core call"
        );
    for expected in ["lt", "gt", "or"] {
        assert!(
            graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|op| { matches!(&op.kind, OpKind::BinOp { op, .. } if op == expected) }),
            "w_range_iter: missing checked-narrowing {expected} op"
        );
    }
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(&op.kind, OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("longlongmask"))
        }),
        "w_range_iter: Ok payload must go through longlongmask"
    );
}

#[test]
fn range_contains_lowers_as_method() {
    // Exclusive `Range` is a struct ADT whose FunDecl is Opaque.
    // `impl_method_owner` declines the Method hint for that shape, so
    // the call is a FunctionPath (`Bookkeeper.getdesc`).
    let llbc = range_contains_fixture();
    let graph = super::lower_function(&llbc, "range_contains").expect("lower Range::contains");
    let ops = graph_ops(&graph);
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if super::fmt_path_ends_with(segments, &["range", "Range", "contains"])
        )),
        "Range::contains must lower as FunctionPath; ops={ops:?}"
    );
}

#[test]
fn decode_packed_format_pieces_handles_long_literal_marker() {
    use super::decode_packed_format_pieces;

    // A literal of 128+ bytes cannot use the single length byte
    // (`L < 0x80`); it packs as `0x80` + little-endian u16 length +
    // the bytes.  A 130-byte piece (`0x0082`) after one placeholder,
    // the exact shape charon lowers for `space_int`'s deprecation
    // warning `format!("…(type {tp}).  The ability to…Python.")`.
    let tail = ").  The ability to return an instance of a strict subclass of int is deprecated, and may be removed in a future version of Python.";
    assert_eq!(tail.len(), 130);
    let mut bytes = vec![2u8, b'a', b'b', 0xC0]; // "ab" then a placeholder
    bytes.push(0x80); // long-literal marker
    bytes.push((tail.len() & 0xFF) as u8); // u16 LE lo
    bytes.push((tail.len() >> 8) as u8); // u16 LE hi
    bytes.extend_from_slice(tail.as_bytes());
    bytes.push(0x00); // terminator
    let (pieces, indices) = decode_packed_format_pieces(&bytes).unwrap();
    assert_eq!(pieces, vec!["ab".to_string(), tail.to_string()]);
    assert_eq!(
        indices.iter().map(|p| p.arg_index).collect::<Vec<_>>(),
        vec![0]
    );

    // A long-literal marker whose declared length overruns the buffer
    // bails (no OOB, fail-safe residual).
    assert_eq!(
        decode_packed_format_pieces(&[0x80, 0xFF, 0xFF, b'x', 0]),
        None
    );
    // A long-literal marker truncated before its 2 length bytes bails.
    assert_eq!(decode_packed_format_pieces(&[0x80, 0x05]), None);
}

/// Anchor the fmt-collapse to the real lowered IR of `space_int` /
/// `space_index` — both raise a deprecation `format!` whose 130-byte
/// trailing literal packs as the `0x80`+u16 long-literal token.  Before
/// the decoder handled that token the chain stayed residual as
/// `alloc::fmt::format` (unregisterable `fmt::rt::Argument::new_display`),
/// dropping ~50 census heads that reach `space_int` via `int_w` etc.
/// After: zero residual `alloc::fmt::format` and the native `str` +
/// `add` (`ll_strconcat`) expansion instead.  Ignored by default (loads
/// the ~440MB real LLBC); run with `cargo test -p majit-translate --lib
/// fmt_collapse_long_literal_real -- --ignored`.
#[test]
#[ignore]
fn fmt_collapse_long_literal_real_space_int() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    for fname in ["space_int", "space_index"] {
        let graph =
            super::lower_function(&llbc, fname).unwrap_or_else(|e| panic!("lower {fname}: {e:?}"));
        let fmt_calls = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if super::fmt_path_ends_with(segments, &["fmt", "format"])
                )
            })
            .count();
        assert_eq!(
            fmt_calls, 0,
            "{fname}: residual alloc::fmt::format after long-literal collapse"
        );
        let str_ops = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "str"))
            .count();
        assert!(
            str_ops >= 1,
            "{fname}: at least one native `str` op from the collapse"
        );
    }
}

/// Real-LLBC anchor for the `Vec<T>::index_mut` write-lift: the
/// arg-fill path `fill_user_function_args` writes its `filled_args`
/// slots via `Vec::index_mut`; after the fold there must be no residual
/// `["vec","Vec","index_mut"]` call and at least one native `ArrayWrite`
/// (setarrayitem).  `#[ignore]`d (loads the ~440MB real LLBC); run with
/// `cargo test -p majit-translate --lib vec_index_mut_fill_user_function_args_real
/// -- --ignored`.
#[test]
#[ignore]
fn vec_index_mut_fill_user_function_args_real() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "fill_user_function_args")
        .expect("lower fill_user_function_args");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["vec", "Vec", "index_mut"])
            )
        })
        .count();
    assert_eq!(
        residual, 0,
        "fill_user_function_args: residual vec::Vec::index_mut after the write-lift"
    );
    let array_writes = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(&op.kind, OpKind::ArrayWrite { .. }))
        .count();
    assert!(
        array_writes >= 1,
        "fill_user_function_args: expected at least one native ArrayWrite (setarrayitem)"
    );
}

/// Real-LLBC anchor for the unsigned `saturating_sub` clamp diamond:
/// `generic_alias_class_getitem`'s arg-count error path computes
/// `args.len().saturating_sub(1)`.  `core::intrinsics::saturating_sub`
/// (`rint.py` `rtype_sub_ovf` for signed; unsigned floors at zero) is
/// mapped in `lower_call`.  There must be no residual
/// `["num","<Impl>","saturating_sub"]` call and at least one `lt` guard
/// + one `sub` (the `if a < b { 0 } else { a - b }` diamond).  `#[ignore]`d
/// (loads the ~440MB real LLBC); run with
/// `cargo test -p majit-translate --lib saturating_sub_generic_alias_real
/// -- --ignored`.
#[test]
#[ignore]
fn saturating_sub_generic_alias_real() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "generic_alias_class_getitem")
        .expect("lower generic_alias_class_getitem");
    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["num", "<Impl>", "saturating_sub"])
            )
        })
        .count();
    assert_eq!(
        residual, 0,
        "generic_alias_class_getitem: residual saturating_sub after the clamp-diamond rewrite"
    );
    let lt_guards = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "lt"))
        .count();
    let subs = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "sub"))
        .count();
    assert!(
        lt_guards >= 1 && subs >= 1,
        "generic_alias_class_getitem: expected the a<b guard and the a-b subtraction \
             (lt={lt_guards}, sub={subs})"
    );
}

/// Real-LLBC anchor for the `FixedObjectArray::set_ref` setarrayitem
/// collapse: `set_locals_w` (the shared STORE_FAST-family leaf) calls
/// `self.locals_cells_stack_w.set_ref(index, value)`.  After the
/// whole-method lift there must be no residual
/// `["object_array","<Impl>","set_ref"]` call and no residual
/// `items_mut_ptr` / write-barrier accessor from the method body, and at
/// least one native `ArrayWrite` (the `setarrayitem_gc`).  `#[ignore]`d
/// (loads the ~440MB real LLBC); run with `cargo test -p majit-translate
/// --lib set_ref_set_locals_w_real -- --ignored`.
#[test]
#[ignore]
fn set_ref_set_locals_w_real() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "set_locals_w").expect("lower set_locals_w");
    let value_input_root = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .find_map(|op| match &op.kind {
            OpKind::Input {
                name, class_root, ..
            } if name == "value" => class_root.as_deref(),
            _ => None,
        });
    assert_eq!(
        value_input_root,
        Some("PyObject"),
        "PyObjectRef input must retain the W_Root-equivalent class identity"
    );
    let residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["object_array", "<Impl>", "set_ref"])
            )
        })
        .count();
    assert_eq!(
        residual, 0,
        "set_locals_w: residual set_ref call after the setarrayitem collapse"
    );
    let array_writes = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| matches!(&op.kind, OpKind::ArrayWrite { .. }))
        .count();
    assert!(
        array_writes >= 1,
        "set_locals_w: expected at least one native ArrayWrite (setarrayitem_gc)"
    );
    let typed_store = graph.blocks.iter().any(|block| {
        block.operations.windows(2).any(|ops| {
            let Some(cast_result) = ops[0].result.as_ref() else {
                return false;
            };
            crate::model::cast_instance_root(&ops[0].kind) == Some("PyObject")
                && matches!(
                    &ops[1].kind,
                    OpKind::ArrayWrite {
                        value: crate::model::LinkArg::Value(value),
                        array_type_id: Some(array_type_id),
                        ..
                    } if value == cast_result && array_type_id == super::OBJECT_REF_GCARRAY_TYPE_ID
                )
        })
    });
    assert!(
        typed_store,
        "set_locals_w: PyObjectRef store must preserve the W_Root external item repr"
    );
}

/// Real-LLBC anchor for the `we_are_jitted()` → `ConstBool(true)` front
/// fold: `lookup_in_type_where` branches on `if not we_are_jitted()` and,
/// in the interpreter arm, calls `_cached_lookup_where_name`, whose body
/// reads the residual-only `METHOD_CACHE` static.  After the front fold the
/// interpreter arm is JIT-dead and the annotator prunes it, but the fold
/// itself already removes the `we_are_jitted` call and collapses its
/// two-way branch to the JIT arm's unconditional goto — so the lowered
/// graph carries no `we_are_jitted` call.  `#[ignore]`d (loads the ~440MB
/// real LLBC); run with `cargo test -p majit-translate --lib
/// we_are_jitted_lookup_in_type_where_real -- --ignored`.
#[test]
#[ignore]
fn we_are_jitted_lookup_in_type_where_real() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "lookup_in_type_where").expect("lower lookup_in_type_where");
    let we_are_jitted_residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["jit", "we_are_jitted"])
            )
        })
        .count();
    assert_eq!(
        we_are_jitted_residual, 0,
        "lookup_in_type_where: residual we_are_jitted call after the front fold"
    );
    // The interpreter arm's `_cached_lookup_where_name` call (which reads
    // the residual-only `METHOD_CACHE` static) is JIT-dead once
    // `we_are_jitted()` folds to True.  The front fold collapses the branch
    // to the JIT arm's unconditional goto, so the interpreter arm becomes
    // unreachable and `simplify_lowered_graph`'s `clear_unreachable_blocks`
    // empties its operations — no `_cached_lookup_where_name` call survives.
    let cached_lookup_residual = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if super::fmt_path_ends_with(segments, &["baseobjspace", "_cached_lookup_where_name"])
                )
            })
            .count();
    assert_eq!(
        cached_lookup_residual, 0,
        "lookup_in_type_where: residual _cached_lookup_where_name (METHOD_CACHE reader) \
             after the we_are_jitted fold pruned the interpreter arm"
    );
}

/// Real-LLBC anchor for the fieldless-enum `Debug` `format!` collapse:
/// `call_intrinsic_1`'s unknown-intrinsic arm renders `func`
/// (`IntrinsicFunction1`, a fieldless enum) with `{:?}`, which lowers to
/// the `alloc::fmt::format` / `fmt::rt::Argument::new_debug` chain.  After
/// `collapse_debug_enum_fmt_chains` there must be no residual `new_debug`
/// (the graph-less extern that walls the rtyper) and the enum-discriminant
/// switch must be present.  `#[ignore]`d (loads the ~440MB real LLBC); run
/// with `cargo test -p majit-translate --lib call_intrinsic_1_debug_enum_fmt_real
/// -- --ignored`.
#[test]
#[ignore]
fn call_intrinsic_1_debug_enum_fmt_real() {
    use crate::model::{CallTarget, ExitSwitch, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "call_intrinsic_1").expect("lower call_intrinsic_1");
    let new_debug_residual = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["Argument", "new_debug"])
            )
        })
        .count();
    assert_eq!(
        new_debug_residual, 0,
        "call_intrinsic_1: residual fmt::rt::Argument::new_debug after the Debug-enum collapse"
    );
    let has_value_switch = graph
        .blocks
        .iter()
        .any(|b| matches!(b.exitswitch, Some(ExitSwitch::Value(_))));
    assert!(
        has_value_switch,
        "call_intrinsic_1: expected an enum-discriminant switch after the collapse"
    );
}

/// `<Option<&str> as PartialEq>::ne` on two niche nullable strings
/// (`opt != Some(c)`). `trait_default_call_segments` rewrites
/// `PartialEq::ne` to `["option", "Option", "ne"]`; the fold emits
/// string `BinOp("ne")` (`rstr.py` `AbstractStringRepr.rtype_ne` →
/// `ll_streq`) instead of leaving that FunctionPath residual.
#[test]
fn niche_option_str_ne_lowers_to_string_binop() {
    use crate::model::{CallTarget, OpKind};

    let span = serde_json::json!({"data": {"file_id": 0,
            "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = serde_json::json!({
        "regions": [], "types": [], "const_generics": [], "trait_refs": []
    });
    let str_ty = serde_json::json!({
        "Adt": {"builtin": "Str", "generics": generics}
    });
    let str_ref = serde_json::json!({"Ref": ["Erased", str_ty, "Shared"]});
    let option_ty = serde_json::json!({
        "Adt": {
            "id": 0,
            "generics": {
                "regions": [],
                "types": [str_ref],
                "const_generics": [],
                "trait_refs": []
            }
        }
    });
    let option_ref = serde_json::json!({"Ref": ["Erased", option_ty, "Shared"]});
    let bool_ty = serde_json::json!({"Scalar": "Bool"});
    let place = |id, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
    let file = serde_json::json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "option", "Option"]),
                "kind": "Opaque"
            }],
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["fixture", "cmp_option_str"]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [option_ref.clone(), option_ref.clone()],
                        "output": bool_ty.clone()
                    },
                    "body": {"Unstructured": {"span": span,
                        "locals": {"arg_count": 2, "locals": [
                            {"index": 0, "name": null, "span": span, "ty": bool_ty.clone()},
                            {"index": 1, "name": "lhs", "span": span, "ty": option_ref.clone()},
                            {"index": 2, "name": "rhs", "span": span, "ty": option_ref.clone()},
                            {"index": 3, "name": null, "span": span, "ty": bool_ty.clone()}
                        ]},
                        "body": [
                            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": {"Call": {
                                "call": {
                                    "func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                    "args": [
                                        {"Copy": place(1, &option_ref)},
                                        {"Copy": place(2, &option_ref)}
                                    ],
                                    "dest": place(3, &bool_ty)
                                },
                                "target": 1, "on_unwind": 2
                            }}}},
                            {"statements": [{
                                "span": span,
                                "kind": {"Assign": [
                                    place(0, &bool_ty),
                                    {"Use": [{"Copy": place(3, &bool_ty)}, "Yes"]}
                                ]}
                            }], "terminator": {"span": span, "kind": "Return"}},
                            {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                        ]
                    }}
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["core", "option", "Option", "ne"]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [option_ref.clone(), option_ref.clone()],
                        "output": bool_ty
                    },
                    "body": "Opaque"
                }
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("option-str fixture parses");
    let graph = super::lower_function(&llbc, "cmp_option_str").expect("lower cmp_option_str");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => super::fmt_path_ends_with_owner_method(
                    segments,
                    &["Option"],
                    &["core", "cmp", "PartialEq"],
                    "ne",
                ),
                _ => false,
            })
            .count(),
        0,
        "no residual `Option::ne` call survives the string-niche fold"
    );
    assert!(
        ops().any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne")),
        "the comparison lowers to a string `BinOp(ne)`"
    );
}

/// The `ne` half of the same fold, on the real lowered IR of
/// `getattr_str_impl` — `err.kind != PyErrorKind::AttributeError` over the
/// fieldless `PyErrorKind`.  `derive` generates only `eq`, so `!=` resolves
/// to `PartialEq`'s provided `ne` and reaches the lowering fully qualified
/// as a `FunctionPath` rather than as a `Method` on the receiver.  Left
/// residual it is an unregistered callee and the whole graph declines.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn fieldless_enum_ne_fold_real_getattr_str_impl() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::baseobjspace::getattr_str_impl")
        .expect("lower getattr_str_impl");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => super::fmt_path_ends_with(segments, &["PyErrorKind", "ne"]),
                _ => false,
            })
            .count(),
        0,
        "no residual `PyErrorKind::ne` call survives the fold"
    );
    assert!(
        ops().any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne")),
        "the discriminant comparison lowers to an int `BinOp(ne)`"
    );
}

/// Anchor the `&str != &str` fold to the comparison in PyPy's
/// `W_Super.getattribute`.  Charon resolves that Rust expression to the
/// generic `core::cmp::impls::<Impl>::ne`, but the translated graph must
/// contain RPython's string inequality operation rather than a residual
/// helper call.  Ignored by default (loads the real interpreter LLBC).
#[test]
#[ignore]
fn str_ne_fold_real_super_getattribute() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::baseobjspace::super_getattribute_str",
    )
    .expect("lower super_getattribute_str");
    let ops = || {
        graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
    };
    assert_eq!(
        ops()
            .filter(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if super::fmt_path_ends_with(segments, &["cmp", "impls", "<Impl>", "ne"])
            ))
            .count(),
        0,
        "no residual generic string `ne` helper survives the fold"
    );
    assert_eq!(
        ops()
            .filter(|operation| matches!(
                &operation.kind,
                OpKind::BinOp { op, .. } if op == "ne"
            ))
            .count(),
        1,
        "the `__class__` comparison lowers to one string `BinOp(ne)`"
    );
}

/// Anchor the string-reference `PartialEq` fold to the real lowered IR of
/// `is_w` — `w_str_get_wtf8(w_one) == w_str_get_wtf8(w_two)` over two
/// `&Wtf8`.  The comparison resolves to the blanket `&A == &B` impl in
/// `core::cmp::impls`, which must fold to `BinOp("eq")` like the direct
/// `<str as PartialEq>::eq` spelling; left residual it is the unlowered
/// helper every descent through `is_w` (`len(x)` via `len_w`) stops at.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn string_ref_eq_fold_real_is_w() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph =
        super::lower_function(&llbc, "pyre_interpreter::baseobjspace::is_w").expect("lower is_w");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if super::fmt_path_ends_with(segments, &["cmp", "impls", "<Impl>", "eq"])
            ))
            .count(),
        0,
        "no residual `core::cmp::impls::<Impl>::eq` call survives the fold"
    );
    assert!(
        ops().any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq")),
        "the `&Wtf8` comparison lowers to a `BinOp(eq)`"
    );
}

/// `<NonNull<T> as PartialEq>::ne` on the two type pointers in
/// `try_dispatch_binary_special` (`w_typ1 != w_typ2`). The operands are
/// the pointer words, so the comparison is `BinOp("ne")` and the
/// `NonNull::ne` call is gone. Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn niche_ptr_cmp_nonnull_ne_try_dispatch_binary_special() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::objspace::descroperation::try_dispatch_binary_special",
    )
    .expect("lower try_dispatch_binary_special");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => super::fmt_path_ends_with_owner_method(
                    segments,
                    &["NonNull"],
                    &["core", "cmp", "PartialEq"],
                    "ne",
                ),
                _ => false,
            })
            .count(),
        0,
        "no residual `NonNull::ne` call survives the fold"
    );
    assert!(
        ops().any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "ne")),
        "the address comparison lowers to a `BinOp(ne)`"
    );
}

/// `<Option<*mut PyObject> as PartialEq>::ne` in `try_hash_value`
/// (`Some(method) != base_hash`). Both words are niche nullable
/// pointers, so the `Option::ne` call is the same address `ne`.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn niche_ptr_cmp_option_ne_try_hash_value() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "pyre_interpreter::builtins::try_hash_value")
        .expect("lower try_hash_value");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert_eq!(
        ops()
            .filter(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => super::fmt_path_ends_with_owner_method(
                    segments,
                    &["Option"],
                    &["core", "cmp", "PartialEq"],
                    "ne",
                ),
                _ => false,
            })
            .count(),
        0,
        "no residual `Option::ne` call survives the address fold"
    );
}

/// `Option<&u8>` compares the `u8`, not the address. `bytes.get(1) !=
/// Some(&b'_')` in `shutdown_module_private_name` must stay a call.
/// Ignored by default (loads the real LLBC).
#[test]
#[ignore]
fn niche_ptr_cmp_option_ref_stays_shutdown_module_private_name() {
    use crate::model::{CallTarget, OpKind};

    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(
        &llbc,
        "pyre_interpreter::shutdown::shutdown_module_private_name",
    )
    .expect("lower shutdown_module_private_name");
    let ops = || graph.blocks.iter().flat_map(|b| b.operations.iter());
    assert!(
        ops().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => super::fmt_path_ends_with_owner_method(
                segments,
                &["Option"],
                &["core", "cmp", "PartialEq"],
                "ne",
            ),
            _ => false,
        }),
        "`Option<&u8>` inequality stays a residual `Option::ne` call"
    );
}

/// `rlib/jit.py isconstant` / `isvirtual` return `NonConstant(False)`, not
/// a bare `False`: the annotator must not read the result as a constant or
/// it folds every `if isconstant(x)` in the interpreter and deletes the
/// looked-inside arm before `jtransform` rewrites the call to
/// `*_isconstant`.  Pin the producer — both bodies must still route their
/// literal through `nonconst::non_constant`, which is resolved as a
/// registered external rather than a lifted graph.  Loads the majit-rlib
/// LLBC, ignored; run with `cargo test -p majit-translate --lib
/// jit_constancy_probes_return_non_constant -- --ignored`.
#[test]
#[ignore]
fn jit_constancy_probes_return_non_constant() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC;
    let llbc = Llbc::load(path).expect("load majit-rlib LLBC");
    for probe in ["isconstant", "isvirtual"] {
        let graph =
            super::lower_function(&llbc, probe).unwrap_or_else(|e| panic!("lower {probe}: {e}"));
        let non_constant_calls = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if super::fmt_path_ends_with(
                            segments,
                            &["nonconst", "non_constant"],
                        )
                )
            })
            .count();
        assert_eq!(
            non_constant_calls, 1,
            "{probe} must return its literal through nonconst::non_constant"
        );
    }
}

/// Every local UTF-8 accumulator in PyPy's repr helpers is a
/// `StringBuilder`; the large `py_repr_wtf8` dispatcher must therefore
/// eliminate the Rust `Wtf8Buf` mutation shell as well as the smaller
/// leaf helpers do.
#[test]
#[ignore]
fn py_repr_wtf8_lifts_owned_wtf8buf_appends() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "py_repr_wtf8").expect("lower py_repr_wtf8");
    // Prove every append/build receiver is rooted at a builder ctor.
    // Start from every phi as a candidate and remove any phi fed by a
    // non-candidate; this greatest fixed point admits loop-carried
    // builders while rejecting the old `to_wtf8_buf` identity value
    // (a plain SomeString producer) and every phi reachable from it.
    let builder_roots: std::collections::HashSet<u64> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| {
                matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments == &[crate::runtime_names::shims::STRINGBUILDER_NEW])
                .then(|| op.result.as_ref().expect("builder new result").id())
            })
            .collect();
    let mut builder_values = builder_roots.clone();
    builder_values.extend(
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.inputargs)
            .map(|input| input.id()),
    );
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            for (pos, input) in block.inputargs.iter().enumerate() {
                let incoming: Vec<_> = graph
                    .blocks
                    .iter()
                    .flat_map(|source| &source.exits)
                    .filter(|link| link.target == block.id)
                    .filter_map(|link| link.args.get(pos))
                    .collect();
                if (incoming.is_empty()
                    || incoming.iter().any(|arg| {
                        arg.as_variable()
                            .is_none_or(|value| !builder_values.contains(&value.id()))
                    }))
                    && !builder_roots.contains(&input.id())
                {
                    changed |= builder_values.remove(&input.id());
                }
            }
        }
        if !changed {
            break;
        }
    }
    let unproven_builder_receivers: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block.operations.iter().filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if matches!(segments.as_slice(), [marker]
                        if marker == crate::runtime_names::shims::STRINGBUILDER_APPEND
                            || marker == crate::runtime_names::shims::STRINGBUILDER_BUILD) =>
                {
                    args.first()
                        .filter(|recv| !builder_values.contains(&recv.id()))
                        .map(|recv| (block.id, segments.clone(), recv.clone()))
                }
                _ => None,
            })
        })
        .collect();
    assert!(
        unproven_builder_receivers.is_empty(),
        "builder operations must receive a value rooted at newstringbuilder: \
             {unproven_builder_receivers:#?}"
    );
    let residual: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| {
                block.operations.iter().filter_map(move |op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &["Wtf8Buf", "push_wtf8"])
                    .then_some((block.id, op))
                })
            })
            .collect();
    assert!(
        residual.is_empty(),
        "owned Wtf8Buf appends must lift to StringBuilder operations: {residual:#?}"
    );
    let residual_vec_indexes: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if super::fmt_path_ends_with(segments, &["vec", "Vec", "index"])
            )
        })
        .collect();
    assert!(
        residual_vec_indexes.is_empty(),
        "repr string lists must iterate without a fat-pointer Vec::index: \
             {residual_vec_indexes:#?}"
    );
    for block in &graph.blocks {
        let mut defined: std::collections::HashSet<u64> =
            block.inputargs.iter().map(|var| var.id()).collect();
        for op in &block.operations {
            for operand in crate::inline::op_variable_refs(&op.kind) {
                assert!(
                    defined.contains(&operand.id()),
                    "block {:?} uses undefined operand {:?} in {:?}",
                    block.id,
                    operand,
                    op.kind
                );
            }
            if let Some(result) = &op.result {
                defined.insert(result.id());
            }
        }
    }
}

/// PyPy's `_match_keywords` indexes its small argument lists with ordinary
/// `getitem`, while `Signature.find_argname` is `@jit.elidable`.  Pyre must
/// preserve both boundaries: the signature's host `Vec<&str>` stays behind
/// the pure residual call, and `keyword_names_w` / `keywords_w` remain
/// parallel one-word GC-pointer lists (represented by shadow-stack slot
/// indices in the hand-written root scaffold).  No fat-pointer
/// `Vec<Wtf8Buf>::index` may escape into the translated binder graph.
#[test]
#[ignore]
fn bind_kwargs_indexes_lower_to_array_reads() {
    use crate::model::{CallTarget, OpKind};
    let path = crate::runtime_names::artifacts::INTERPRETER_ULLBC;
    let llbc = Llbc::load(path).expect("load real LLBC");
    let hints = crate::front::llbc_hints::harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    assert!(
        hints
            .get("gateway::<Impl>::find_argname")
            .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
        "Signature.find_argname must retain PyPy's @jit.elidable marker"
    );
    let graph = super::lower_function(&llbc, "parse_builtin_signature")
        .expect("lower parse_builtin_signature");

    let residual_scalar_indexes: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &[
                    "core".to_string(),
                    "slice".to_string(),
                    "index".to_string(),
                    "<Impl>".to_string(),
                    "index".to_string(),
                ]
            )
        })
        .collect();
    assert_eq!(
        residual_scalar_indexes.len(),
        0,
        "scalar SliceIndex<usize> calls must become getitem operations"
    );
    let residual_vec_indexes: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if super::fmt_path_ends_with(segments, &["vec", "Vec", "index"])
            )
        })
        .collect();
    assert_eq!(
        residual_vec_indexes.len(),
        0,
        "keyword_names_w/keywords_w must lower through their one-word \
             rooted-list representation; a residual Vec::index blocks the \
             whole builtin-call chain: {residual_vec_indexes:#?}"
    );
    assert!(
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| { matches!(op.kind, OpKind::ArrayRead { .. }) }),
        "keyword matching should contain the lowered signature getitems"
    );
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["__getslice_range".to_string()]
            )
        }),
        "the proven *args tail Range should use the annotated getslice marker"
    );
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &[
                    "core".to_string(),
                    "convert".to_string(),
                    "num".to_string(),
                    "<Impl>".to_string(),
                    "from".to_string(),
                ]
            )
        }),
        "bool-to-usize offset conversion must not remain an opaque core call"
    );
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &[
                    "rpython".to_string(),
                    "rlib".to_string(),
                    "rarithmetic".to_string(),
                    "r_uint".to_string(),
                ]
            )
        }),
        "From<bool> for usize should use RPython's cast_bool_to_uint path"
    );
}

/// Real-LLBC anchors for three roots that spell one PyPy operation —
/// an ordinary list read — over three different host containers.
///
/// `combine_starargs_wrapped` lowers because its elements are object
/// pointers, one target word each. `dict_repr` must first flatten its
/// host-width key/value pairs into the same one-word rooted-list shape.
/// `bind_builtin_kwargs` remains the counter-example: being the same PyPy
/// shape is not what makes a read addressable, the element width is.
#[test]
#[ignore]
fn pypy_list_item_roots_have_no_residual_vec_index() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let residual_vec_indexes = |fname: &str| {
        let graph = super::lower_function(&llbc, fname)
            .unwrap_or_else(|err| panic!("lower {fname}: {err}"));
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if super::fmt_path_ends_with(segments, &["vec", "Vec", "index"])
                )
            })
            .count()
    };
    assert_eq!(
        residual_vec_indexes("combine_starargs_wrapped"),
        0,
        "argument-list concatenation indexes object pointers, one target \
             word each, so its getitems must not remain Vec::index"
    );
    assert_eq!(
        residual_vec_indexes("load_const_value"),
        0,
        "slice-constant handler values must cross the scalar SliceIndex boundary"
    );
    assert_eq!(
        residual_vec_indexes("dict_repr"),
        0,
        "dict_repr must flatten host-width pairs before translated reads"
    );
    // `bind_builtin_kwargs` indexes `names: &[&str]`. PyPy's
    // `Signature.argnames` is a list of `Ptr(STR)`, but `&str` is a
    // two-word fat pointer here, so a one-word getarrayitem would land
    // mid-element from the second name on. Lowering it requires a managed
    // string list, not a wider element gate.
    assert!(
        residual_vec_indexes("bind_builtin_kwargs") > 0,
        "a wider-than-one-word host element must keep its residual Vec::index"
    );
}

/// PyPy's rStringIO and buffered I/O implementations use ordinary
/// `buffer[start:end]` list slices. The Rust port spells those as a
/// `SliceIndex<Range<usize>>` call, which must recover RPython's direct
/// `getslice` operation without adding a non-upstream dominator proof.
#[test]
#[ignore]
fn pypy_buffer_ranges_lower_to_getslice() {
    use crate::model::{CallTarget, OpKind};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    for fname in ["read_bytes", "parse_length_prefixed_protocols"] {
        let graph = super::lower_function(&llbc, fname)
            .unwrap_or_else(|err| panic!("lower {fname}: {err}"));
        assert!(
                !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if super::fmt_path_ends_with(segments, &["core", "slice", "index", "<Impl>", "index"]))
                }),
                "{fname}: PyPy list slice must not remain an opaque Rust index"
            );
        assert!(
                graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                    matches!(&op.kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments == &["__getslice_range".to_string()])
                }),
                "{fname}: Range slice must use the RPython getslice marker"
            );
    }
}

#[test]
#[ignore]
fn check_len_result_uint_format_template_collapses() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    let graph = super::lower_function(&llbc, "_check_len_result").expect("lower _check_len_result");
    assert!(
        !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if super::fmt_path_ends_with(segments, &["Argument", "new_display"])
                    || super::is_arguments_new_path(segments)
                    || super::fmt_path_ends_with(segments, &["fmt", "format"])
            )
        }),
        "the unsigned `[u8; N]` template must collapse to native RPython string operations"
    );
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(&op.kind, OpKind::BinOp { op, result_ty: ValueType::Ref(None), .. }
                    if op == "add")
        }),
        "the formatted overflow message must retain the native string-concat expansion"
    );
}

/// PyPy's deque mutation marker is an object-resident `lock` identity:
/// `modified` stores None, `getlock` lazily stores a new fieldless object,
/// and `checklock` compares that field by identity.  The Rust source graph
/// must therefore expose ordinary field traffic, never the former
/// counter's `core::ptr::read_volatile` shell.
#[test]
#[ignore]
fn deque_mutation_marker_uses_pypy_lock_identity_fields() {
    use crate::model::{CallTarget, OpKind};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("load real LLBC");
    for name in [
        "pyre_interpreter::module::_collections::modified",
        // `checklock`'s comparison half — the graph that carries the
        // `self.lock` field read.  `checklock` itself only wraps it in
        // the `RuntimeError` raise.
        "pyre_interpreter::module::_collections::lock_valid",
    ] {
        let graph = super::lower_function(&llbc, name)
            .unwrap_or_else(|error| panic!("lower {name}: {error}"));
        assert!(
            !graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if super::fmt_path_ends_with(segments, &["ptr", "read_volatile"])
                            || super::fmt_path_ends_with(segments, &["ptr", "write_volatile"])
                    )
                }),
            "{name}: PyPy lock identity must not retain volatile-counter calls"
        );
        assert!(
            graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. } | OpKind::FieldWrite { field, .. }
                            if field.name == "lock"
                    )
                }),
            "{name}: expected an object-resident W_Deque.lock access"
        );
    }
}
