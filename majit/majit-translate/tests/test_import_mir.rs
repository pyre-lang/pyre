mod common;

use std::collections::HashSet;

use common::{INTERPRETER_LLBC, OBJECT_LLBC, load_llbc};
use majit_charon_reader::Llbc;
use majit_translate::{
    CallPath, FunctionGraph, HostStaticAddrs,
    annotator::classdesc::register_struct_fields_with_layout,
    codewriter::{
        CodeWriter,
        call::CallControl,
        jtransform::{GraphTransformConfig, transform_graph},
    },
    flowspace::argument::Signature,
    front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
    model::{CallTarget, ConcreteType, OpKind},
    translator::rtyper::{legacy_annotator, legacy_resolve},
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

/// `W_BytesObject.descr_lt` compares the two `STR` payloads
/// (`rstr.py` `LLHelpers.ll_strcmp` / `stroruni.cmp`). The exact-bytes arms
/// read `&BYTES_TYPE`, and the rtyper declines a graph that reads a static
/// type object, which leaves a `stroruni.cmp` operand without `Ptr(STR)`.
/// So the arms call the `W_BytesObject.descr_*` methods, which own the
/// `ll_strcmp` / `ll_streq` call, and never a slice helper.
#[test]
fn bytes_ordering_leaves_ll_strcmp_to_w_bytes_object_methods() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs::default(),
        &["descroperation"],
        &["pin_free_builtin_ord", "compare_slot_rest"],
    )
    .expect("lower descroperation bytes-ordering arms");
    for name in ["pin_free_builtin_ord", "compare_slot_rest"] {
        let function = program
            .functions
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name} graph"));
        let ops = function
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect::<Vec<_>>();
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if matches!(
                    segments.last().map(String::as_str),
                    Some("jit_ll_strcmp" | "ll_bytes_strcmp" | "w_bytes_data")
                )
            )),
            "{name} must not call jit_ll_strcmp, ll_bytes_strcmp or w_bytes_data"
        );
    }
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

/// `PyCode.code_ptr` is a leaked `CodeObject` address. `history.getkind` of
/// a `Ptr` whose `TO._gckind == 'raw'` is `"int"`; a generic `*const ()`
/// rule would also bank `Function.code` / `PyFrame.pycode`.
#[test]
fn pycode_code_ptr_is_raw_int_kind() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs::default(),
        &["pycode", "baseobjspace"],
        &[
            "w_code_get_ptr",
            "w_code_addr2line",
            "w_code_firstlineno_raw",
            "w_code_filename_obj",
            "code_filename_bytes",
            "is_coroutine",
        ],
    )
    .expect(
        "lower pycode::w_code_get_ptr, w_code_addr2line, w_code_firstlineno_raw, w_code_filename_obj, code_filename_bytes, and baseobjspace::is_coroutine",
    );
    let spelling = program
        .struct_fields
        .field_type("PyCode", "code_ptr")
        .expect("PyCode.code_ptr harvested");
    assert!(
        spelling == "*const CodeObject" || spelling.starts_with("*const CodeObject<"),
        "PyCode.code_ptr spelling must be *const CodeObject, got {spelling}"
    );
    let w_code = program
        .functions
        .iter()
        .find(|f| f.name == "w_code_get_ptr")
        .expect("w_code_get_ptr graph");
    let code_ptr_result = w_code
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "code_ptr" => {
                Some(op.result.clone().expect("code_ptr FieldRead result"))
            }
            _ => None,
        })
        .expect("w_code_get_ptr reads PyCode.code_ptr");

    let filename = program
        .functions
        .iter()
        .find(|f| f.name == "w_code_filename_obj")
        .expect("w_code_filename_obj graph");
    assert!(
        filename
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "w_filename"
            )),
        "w_code_filename_obj reads PyCode.w_filename"
    );
    assert!(
        !filename
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "code_ptr"
            )),
        "w_code_filename_obj does not load code_ptr; w_filename is realized at construction"
    );

    // Same annotation as the two-phase prepass (`dual_gate_publish_concretetypes`).
    // A `*const CodeObject` value that is dereferenced must not annotate as
    // SomeInteger, or `getattr` of `flags` / `varnames` skips the graph.
    {
        let mut entries: Vec<_> = program.struct_field_attrs.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (qualified, fields) in entries {
            let layout_pairs: Option<Vec<(String, String)>> =
                program.struct_fields.fields.get(qualified).map(|rows| {
                    rows.iter()
                        .map(|row| (row.name.clone(), row.ty.clone()))
                        .collect()
                });
            register_struct_fields_with_layout(qualified, fields, layout_pairs.as_deref());
        }
    }
    let is_coroutine = program
        .functions
        .iter()
        .find(|f| f.name == "is_coroutine" && f.module_path == "baseobjspace")
        .expect("baseobjspace::is_coroutine graph");
    // Intern keys are `canonical_struct_name` of the crate-stripped TypeDecl
    // path (`bytecode::CodeObject`). Production publishes struct_origins
    // before annotation; do the same so a bare `CodeObject` intern lands
    // on that identity.
    majit_ir::descr::register_struct_origins(program.struct_origins.clone());
    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.set_enum_variant_by_discriminant(program.enum_variant_by_discriminant.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    // `_offset2lineno` is `dont_look_inside`; production registers it as a
    // signature-only residual stub (`collect_policy_opaque_fn_stubs_from_llbc`).
    // The filtered test lowering has no stub harvest, so install the i32
    // return here before the first dual-gate builds the registry.
    callcontrol.unsafe_fn_stubs.push((
        vec![
            "pyre_interpreter".into(),
            "pycode".into(),
            "_offset2lineno".into(),
        ],
        Signature::new(vec!["w_code".into(), "addrq".into()], None, None),
        Some("i64".into()),
    ));
    for func in &program.functions {
        let path = if func.module_path.is_empty() {
            CallPath::from_segments([func.name.as_str()])
        } else {
            CallPath::from_segments([func.module_path.as_str(), func.name.as_str()])
        };
        callcontrol.register_function_graph(path, func.graph().clone());
    }
    let mut cw = CodeWriter::new();
    let skip = cw
        .dual_gate_skip_reason(is_coroutine.graph(), &callcontrol)
        .expect("dual-gate of is_coroutine");
    if let Some(reason) = skip.as_deref() {
        assert!(
            !reason.contains("on Integer") && !reason.contains("CodeObject) ∪ Integer"),
            "is_coroutine must not annotate *const CodeObject as SomeInteger: {reason}"
        );
    }

    let addr2line = program
        .functions
        .iter()
        .find(|f| f.name == "w_code_addr2line")
        .expect("w_code_addr2line graph");
    cw.dual_gate_publish_concretetypes(w_code.graph(), &mut callcontrol, "w_code_get_ptr");
    cw.dual_gate_publish_concretetypes(addr2line.graph(), &mut callcontrol, "w_code_addr2line");

    let bk = callcontrol.bookkeeper().expect("dual-gate bookkeeper");
    let classdef = bk
        .getuniqueclassdef_for_struct_root("CodeObject")
        .expect("CodeObject classdef");
    let flavor = classdef
        .borrow()
        .classdesc
        .borrow()
        .pyobj
        .class_get("_alloc_flavor_");
    assert_eq!(
        flavor.as_ref().and_then(|v| v.as_text()).as_deref(),
        Some("raw"),
        "getgcflavor reads _alloc_flavor_ = raw off the CodeObject classdesc"
    );

    assert_eq!(
        FunctionGraph::concretetype_of(&code_ptr_result),
        ConcreteType::Signed,
        "FieldRead of PyCode.code_ptr has kind 'i'"
    );
    let rewritten = transform_graph(addr2line.graph(), &GraphTransformConfig::default());
    for op in rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
    {
        let dump = format!("{:?}", op.kind);
        assert!(
            !dump.contains("ptr_iszero") && !dump.contains("ptr_nonzero"),
            "w_code_addr2line must lower the null test as an int op: {dump}"
        );
    }

    fn run_legacy(graph: &FunctionGraph, raw: &HashSet<String>) {
        for var in graph.iter_variables() {
            *var.annotation.borrow_mut() = None;
            *var.concretetype.borrow_mut() = None;
        }
        legacy_annotator::annotate(graph);
        legacy_resolve::resolve_types(graph, raw);
    }

    let filename_bytes = program
        .functions
        .iter()
        .find(|f| f.name == "code_filename_bytes")
        .expect("code_filename_bytes graph");
    let filename_bytes_graph = filename_bytes.graph();
    let filename_bytes_code_ptr = filename_bytes_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "code_ptr" => {
                Some(op.result.clone().expect("code_ptr FieldRead result"))
            }
            _ => None,
        })
        .expect("code_filename_bytes reads PyCode.code_ptr");
    run_legacy(filename_bytes_graph, &program.raw_alloc_flavor_structs);
    assert_eq!(
        FunctionGraph::concretetype_of(&filename_bytes_code_ptr),
        ConcreteType::Signed,
        "legacy resolve of code_filename_bytes code_ptr FieldRead is Signed"
    );

    run_legacy(filename_bytes_graph, &HashSet::new());
    assert_eq!(
        FunctionGraph::concretetype_of(&filename_bytes_code_ptr),
        ConcreteType::GcRef,
        "legacy resolve with an empty raw set leaves code_ptr as GcRef"
    );
}
