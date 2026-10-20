mod common;

use common::{INTERPRETER_LLBC, OBJECT_LLBC, load_llbc};
use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
    model::{CallTarget, OpKind},
};

/// `dunder_import_absolute_head` is the empty-fromlist arm that holds
/// `name.find('.')` and `name[:dotindex]`; `dunder_import` itself delegates
/// to it and carries neither call.
#[test]
fn dunder_import_lowers_rust_string_find_and_slices_to_rpython_ops() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    // The tuple decl is owned by pyre-object.
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs::default(),
        &["importing"],
        &["dunder_import_absolute_head"],
    )
    .expect("lower importing::dunder_import_absolute_head");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "dunder_import_absolute_head")
        .expect("dunder_import_absolute_head graph");

    assert_eq!(
        program
            .struct_fields
            .field_type("W_TupleObject", "wrappeditems"),
        Some("[*mut PyObject]"),
        "PyPy tuple storage must project as GcArray(OBJECTPTR), not ItemsBlock"
    );
    let data_layout = program.struct_fields.field_type("W_BytesObject", "data");
    assert!(
        matches!(
            data_layout,
            Some("*const BytesBlock") | Some("*const Utf8Str")
        ),
        "W_BytesObject.data is rstr.STR (`W_BytesObject._value`), a Ptr(STR); Charon spells the BytesBlock alias as Utf8Str, got {data_layout:?}"
    );

    let ops = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(ops.iter().any(|op| matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments == &["__getslice_rangeto".to_string()]
    )));
    assert!(ops.iter().any(|op| matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } if name == "find"
    )));
    assert!(!ops.iter().any(|op| matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments.starts_with(&["core".to_string(), "str".to_string()])
            && matches!(segments.last().map(String::as_str), Some("find" | "index"))
    )));
}

/// Tuple length is `ll_fixed_length` on `wrappeditems`, the fixed list.
/// It must not call `items_block_capacity`, which also receives the
/// resizable list's `ll_items` pointer.
#[test]
fn tuple_len_reads_the_fixed_list_not_the_items_block() {
    let llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["tupleobject"],
        &["w_tuple_len"],
    )
    .expect("lower tupleobject::w_tuple_len");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "w_tuple_len")
        .expect("w_tuple_len graph");
    let ops = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::ArrayLen {
                array_type_id: Some(id),
                nolength: false,
                ..
            } if id == "majit::object_ref_gcarray"
        )),
        "wrappeditems length is arraylen on the fixed list"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("items_block_capacity")
        )),
        "the fixed list must not enter items_block_capacity"
    );
}

/// Tuple item read is `getarrayitem_gc` on `wrappeditems`
/// (`rlist.py` `ll_getitem_fast` / `FixedSizeListRepr`). The
/// `items_block_items_base` accessor aliases to that array header.
#[test]
fn tuple_getitem_reads_the_fixed_list_not_the_items_block() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["tupleobject"],
        &["w_tuple_getitem_known"],
    )
    .expect("lower tupleobject::w_tuple_getitem_known");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "w_tuple_getitem_known")
        .expect("w_tuple_getitem_known graph");
    let ops = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::ArrayRead {
                array_type_id: Some(id),
                ..
            } if id == "majit::object_ref_gcarray"
        )),
        "wrappeditems item is getarrayitem on the fixed list"
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("items_block_items_base")
        )),
        "the fixed list must not enter items_block_items_base"
    );
}

/// `*(&[u8]).as_ptr().add(i)` in `ll_bytes_strcmp`
/// (`rstr.py` `AbstractStringRepr.ll_strcmp`) is `__string_byte_getitem`
/// (getitem + ord on the STR), not a `RawLoad`.
#[test]
fn bytes_strcmp_reads_u8_slice_as_array() {
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["descroperation"],
        &["ll_bytes_strcmp"],
    )
    .expect("lower descroperation::ll_bytes_strcmp");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "ll_bytes_strcmp")
        .expect("ll_bytes_strcmp graph");
    let ops = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("__string_byte_getitem")
        )),
        "byte slice item is getitem+ord on STR (`__string_byte_getitem`)"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::RawLoad { .. })),
        "byte slice item must not be RawLoad"
    );
}

/// `ll_list_obj_capacity` is `len(l.items)` on the resizable list's
/// `ItemsBlock` header. That header is not a fixed-list GcArray, so the
/// call stays.
#[test]
fn list_items_capacity_stays_on_the_items_block() {
    let llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["listobject"],
        &["ll_list_obj_capacity"],
    )
    .expect("lower listobject::ll_list_obj_capacity");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "ll_list_obj_capacity")
        .expect("ll_list_obj_capacity graph");
    let ops = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("items_block_capacity")
        )),
        "ItemsBlock capacity stays a call"
    );
    assert!(
        !ops.iter()
            .any(|op| matches!(&op.kind, OpKind::ArrayLen { .. })),
        "the items header is not ll_fixed_length"
    );
}
