mod common;

use std::collections::HashSet;

use common::{
    INTERPRETER_LLBC, MODULE_LLBC, OBJECT_LLBC, RLIB_LLBC,
    build_semantic_program_from_llbcs_with_static_addrs_and_function_names, load_llbc, lower_named,
    lower_named_with_static_addrs,
};
use majit_charon_reader::Llbc;
use majit_translate::{
    CallPath,
    ErrorCarrierSpec,
    FunctionGraph,
    HeuristicLayoutProvider,
    HostStaticAddrs,
    LayoutProvider,
    SemanticProgram,
    annotator::classdesc::register_struct_fields_with_layout,
    codewriter::{
        CodeWriter,
        call::CallControl,
        jitcode::JitCode,
        jtransform::{GraphTransformConfig, Transformer, transform_graph},
    },
    flowspace::argument::Signature,
    // build_semantic_* on shared artefacts goes through common.
    insns::{BC_INLINE_CALL_R_I, BC_INLINE_CALL_R_R},
    model::{
        ArrayGcKind, CallTarget, ConcreteType, ExitSwitch, LinkArg, OpKind, ValueType,
        VecFieldPart, getkind,
    },
    translator::rtyper::{legacy_annotator, legacy_resolve},
};

const ERROR_CARRIER: ErrorCarrierSpec<'static> = ErrorCarrierSpec {
    carrier_path: "pyre_interpreter::error::PyError",
    carrier_class: "",
    carrier_wrappers: &[],
    to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
    from_exc_object: Some(("PyError", "from_exc_object")),
};

/// `dunder_import_absolute_head` is the empty-fromlist arm that holds
/// `modulename.find('.')` and `modulename[:dotindex]` on a `Ptr(STR)`
/// (`rstr.py` `LLHelpers.ll_find_char` / `ll_stringslice_startstop`);
/// `dunder_import` itself delegates to it and carries neither call.
#[test]
fn dunder_import_lowers_str_find_and_slice_to_rpython_ops() {
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
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("ll_stringslice_startstop")
        )),
        "dotted head is rstr LLHelpers.ll_stringslice_startstop of the name STR"
    );
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "find"
        )),
        "dot search is the STR find method (rstr LLHelpers.ll_find_char)"
    );
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
                gckind: ArrayGcKind::Gc,
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
    let llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(llbc),
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
    cw.dual_gate_publish_concretetypes(
        w_code.graph(),
        &mut callcontrol,
        &CallPath::from_segments(["w_code_get_ptr"]),
    );
    cw.dual_gate_publish_concretetypes(
        addr2line.graph(),
        &mut callcontrol,
        &CallPath::from_segments(["w_code_addr2line"]),
    );

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
        "empty harvest cannot int-bank at the legacy layer"
    );
}

fn production_addrs() -> HostStaticAddrs<'static> {
    HostStaticAddrs {
        error_carrier: ERROR_CARRIER,
        ..Default::default()
    }
}

/// `utf8_error_handler` and `do_warn_explicit` each contain a `?` whose
/// Continue arm is `unreachable!()`. After simplify prunes that Panic arm,
/// the ControlFlow `__discriminant` read is gone; the front end must still
/// lower both graphs.
#[test]
fn utf8_error_handler_and_do_warn_explicit_lower_question_sites() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let addrs = production_addrs();
    for name in [
        "pyre_interpreter::typedef::utf8_error_handler",
        "pyre_interpreter::module::_warnings::do_warn_explicit",
    ] {
        match lower_named_with_static_addrs(interpreter_llbc, name, addrs) {
            Ok(graph) => {
                assert!(!graph.blocks.is_empty(), "{name} lowered to an empty graph");
            }
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    !msg.contains("unsupported MIR")
                        && !msg.contains("lacks the ControlFlow __discriminant read"),
                    "{name} failed to lower: {msg}"
                );
                panic!("{name} failed to lower: {msg}");
            }
        }
    }
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        addrs,
        &["typedef", "module::_warnings"],
        &["utf8_error_handler", "do_warn_explicit"],
    )
    .expect("lower utf8_error_handler and do_warn_explicit");
    for name in ["utf8_error_handler", "do_warn_explicit"] {
        let function = program
            .functions
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name} graph was not produced"));
        assert!(!function.graph().blocks.is_empty(), "{name} graph is empty");
    }
}

fn residual_function_paths(graph: &FunctionGraph) -> Vec<&[String]> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.as_slice()),
            _ => None,
        })
        .collect()
}

/// Default-spec `{:x}` (`sre_pattern_flag_tail`, `unknown_code_point_display`)
/// collapses to `'%x' % r_uint`; proven-u8 `{:02x}` (`unicode_decode_error_str`)
/// stays the nibble lookup. No `alloc::fmt::format` and no 16-nibble unroll.
#[test]
fn default_spec_lower_hex_collapses_to_mod_pct_x() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let names = [
        "sre_pattern_flag_tail",
        "unknown_code_point_display",
        "unicode_decode_error_str",
    ];
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        production_addrs(),
        &["display", "module::_sre::interp_sre", "type_methods"],
        &names,
    )
    .expect("lower hex-format production graphs");
    let mut found = Vec::new();
    let mut leftover_fmt = Vec::new();
    let mut missing_pct_x = Vec::new();
    let mut unrolled = Vec::new();
    for function in &program.functions {
        if !names.contains(&function.name.as_str()) {
            continue;
        }
        found.push(function.name.clone());
        let graph = function.graph();
        let paths = residual_function_paths(graph);
        if paths.iter().any(|segments| {
            segments.len() >= 2
                && segments[segments.len() - 2..] == ["fmt".to_string(), "format".to_string()]
        }) {
            leftover_fmt.push(function.name.clone());
        }
        let getslice = paths
            .iter()
            .filter(|segments| segments.last().map(String::as_str) == Some("__getslice_rangeto"))
            .count();
        if getslice > 2 {
            unrolled.push(format!("{}:{getslice}", function.name));
        }
        let has_pct_x_mod = graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                let OpKind::BinOp { op: name, lhs, .. } = &op.kind else {
                    return false;
                };
                if name != "mod" {
                    return false;
                }
                graph.blocks.iter().any(|b| {
                    b.operations.iter().any(|prod| {
                        prod.result.as_ref().map(|r| r.id()) == Some(lhs.id())
                            && matches!(
                                &prod.kind,
                                OpKind::Call {
                                    target: CallTarget::FunctionPath { segments, .. },
                                    ..
                                } if segments.as_slice()
                                    == ["__str_const".to_string(), "%x".to_string()]
                            )
                    })
                })
            })
        });
        if matches!(
            function.name.as_str(),
            "sre_pattern_flag_tail" | "unknown_code_point_display"
        ) && !has_pct_x_mod
        {
            missing_pct_x.push(function.name.clone());
        }
    }
    for name in names {
        assert!(
            found.iter().any(|n| n == name),
            "{name} was not lowered, got {found:?}"
        );
    }
    assert!(
        leftover_fmt.is_empty(),
        "alloc::fmt::format must be consumed, leftover {leftover_fmt:?}"
    );
    assert!(
        missing_pct_x.is_empty(),
        "default-spec {{:x}} must become BinOp mod of __str_const(\"%x\"), missing {missing_pct_x:?}"
    );
    assert!(
        unrolled.is_empty(),
        "hex collapse must not unroll digits: {unrolled:?}"
    );

    // `transform_graph_to_jitcode` folds `__str_const` to `ConstStr`
    // before jtransform. The leftover `'%x' % r_uint` must become
    // `jit_ll_int2hex`, not `int_mod` over a Ptr/ref.
    for function in &program.functions {
        if !matches!(
            function.name.as_str(),
            "sre_pattern_flag_tail" | "unknown_code_point_display"
        ) {
            continue;
        }
        let mut graph = function.graph().clone();
        for block in graph.blocks.iter_mut() {
            for op in block.operations.iter_mut() {
                let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const" {
                    op.kind = OpKind::ConstStr(segments[1].as_bytes().to_vec());
                }
            }
        }
        let rewritten = transform_graph(&graph, &GraphTransformConfig::default());
        for op in rewritten
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
        {
            let OpKind::BinOp {
                op: name, lhs, rhs, ..
            } = &op.kind
            else {
                continue;
            };
            if name != "mod" && name != "int_mod" {
                continue;
            }
            let lhs_ct = FunctionGraph::concretetype_of(lhs);
            let rhs_ct = FunctionGraph::concretetype_of(rhs);
            assert!(
                lhs_ct != ConcreteType::GcRef && rhs_ct != ConcreteType::GcRef,
                "{}: string-format mod must not survive jtransform as int_mod \
                 with a Ptr/ref operand ({name} lhs={lhs_ct:?} rhs={rhs_ct:?})",
                function.name
            );
        }
        let has_hex_residual = rewritten.graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::CallResidual {
                        result_kind: 'r',
                        args_i,
                        args_r,
                        args_f,
                        ..
                    } if !args_i.is_empty() && args_r.is_empty() && args_f.is_empty()
                )
            })
        });
        assert!(
            has_hex_residual,
            "{}: '%x' % n must lower to a jit_ll_int2hex residual after ConstStr fold",
            function.name
        );
    }
}

/// Scalar `Vec::index` is intercepted in `lower_call`; a `Range*` index is
/// rewritten by `front::slice_index` (`rtype_getitem` / `rtype_getslice`).
/// After #2470 the residual spelling is the trait-impl identity path
/// (`trait_impl_identity_path`), and the recognizer must still intercept it.
#[test]
fn oefmt_has_no_residual_vec_index() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        production_addrs(),
        &["error"],
        &["oefmt"],
    )
    .expect("lower pyre_interpreter::error::oefmt");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "oefmt")
        .expect("oefmt graph");
    let leftover: Vec<&[String]> = residual_function_paths(function.graph())
        .into_iter()
        .filter(|segments| {
            matches!(
                segments.last().map(String::as_str),
                Some("index" | "index_mut")
            ) && segments.iter().any(|seg| seg == "Vec")
        })
        .collect();
    assert!(
        leftover.is_empty(),
        "Vec index must be intercepted, leftover {leftover:#?}"
    );

    // Bytes arm: `Wtf8::from_bytes(value)` of `FmtArg::Bytes` is a window
    // pair (`TupleRepr.getitem` of SomeTuple([SomePtr(CCHARP), SomeInteger])),
    // rewritten to `rutf8.check_utf8` (`rutf8.py check_utf8` /
    // `unicodehelper.py str_decode_utf8`).
    let graph = function.graph();
    let paths = residual_function_paths(graph);
    assert!(
        !paths
            .iter()
            .any(|segments| segments.ends_with(&["Wtf8".to_string(), "from_bytes".to_string()])),
        "checked Wtf8::from_bytes must not remain a residual: {paths:#?}"
    );
    let mut check_utf8_window = false;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !segments.ends_with(&["rutf8".to_string(), "check_utf8".to_string()]) {
                continue;
            }
            if args.len() >= 3 {
                check_utf8_window = true;
                break;
            }
            for arg in args {
                let LinkArg::Value(var) = arg else {
                    continue;
                };
                for (kind, _) in root_producers(graph, var) {
                    if let OpKind::FieldRead { field, .. } = &kind
                        && matches!(
                            field.vec_part,
                            Some(VecFieldPart::FatData | VecFieldPart::FatLen)
                        )
                    {
                        check_utf8_window = true;
                    }
                }
            }
        }
    }
    assert!(
        check_utf8_window,
        "oefmt Bytes arm must call rutf8::check_utf8 with a window pair"
    );
}

/// `flowspace/model.py` `checkgraph`: a canraise block's exception link
/// (`c_last_exception` exits[1..]) must not carry the last, raising,
/// operation's result. That result is undefined on the exception edge;
/// the exception values travel in `last_exception` / `last_exc_value`
/// (`flowcontext.py` `guessexception`).
#[test]
fn raising_op_result_does_not_flow_into_exception_link() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        production_addrs(),
        &["error", "eval"],
        &["write_unraisable_first_line", "dict_merge"],
    )
    .expect("lower write_unraisable_first_line and dict_merge");
    let mut found = Vec::new();
    let mut violations = Vec::new();
    for function in &program.functions {
        if function.name != "write_unraisable_first_line" && function.name != "dict_merge" {
            continue;
        }
        found.push(format!(
            "{}::{}@{}",
            function.module_path,
            function.name,
            function.self_ty_root.as_deref().unwrap_or("-")
        ));
        let graph = function.graph();
        for (bi, block) in graph.blocks.iter().enumerate() {
            if !matches!(block.exitswitch, Some(ExitSwitch::LastException)) {
                continue;
            }
            let Some(last_op) = block.operations.last() else {
                continue;
            };
            let Some(last_res) = last_op.result.as_ref() else {
                continue;
            };
            for (ei, link) in block.exits.iter().enumerate().skip(1) {
                for (ai, arg) in link.args.iter().enumerate() {
                    if matches!(arg, LinkArg::Value(v) if v == last_res) {
                        violations.push(format!(
                            "{}::{} block {bi} last={:?} result={last_res:?} \
                             exc_exit[{ei}].args[{ai}]={arg:?}",
                            function.module_path, function.name, last_op.kind,
                        ));
                    }
                }
            }
        }
    }
    assert!(
        found
            .iter()
            .any(|s| s.contains("write_unraisable_first_line")),
        "write_unraisable_first_line was not lowered, got {found:?}"
    );
    assert!(
        found.iter().any(|s| s.contains("dict_merge")),
        "dict_merge was not lowered, got {found:?}"
    );
    assert!(
        violations.is_empty(),
        "raising operation result cannot flow into exception link: {violations:#?}"
    );
}

/// `oefmt` calls `expected_fmt_arg` with `'T'` and with
/// `formats[i].chars().next().unwrap()`. A Rust `char` is `UniChar`
/// (`builtin_unichr` / `IntegerRepr.rtype_unichr`); an int code point from
/// `Utf8StringIterator.next` (`rutf8.py`) is not. Call-site arguments
/// must be the `as char` unichr op, a char constant, a `FieldRead` of
/// `__pos_0`, a regular call of the extracted Option unwrap body (the
/// helper's Int result is already a UniChar; the None arm is implicit
/// AssertionError pruned by `remove_assertion_errors`),
/// or mergeinputargs unions Integer with UnicodeCodePoint.
#[test]
fn oefmt_expected_fmt_arg_args_come_from_unichr_or_char_const() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        production_addrs(),
        &["error"],
        &["oefmt"],
    )
    .expect("lower pyre_interpreter::error::oefmt");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "oefmt")
        .expect("oefmt graph");
    let graph = function.graph();

    let mut calls = 0;
    let mut bad = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("expected_fmt_arg") {
                continue;
            }
            calls += 1;
            let Some(arg) = args.first() else {
                bad.push(format!("block {bi} op {oi}: call has no argument"));
                continue;
            };
            let roots = root_producers(graph, arg);
            if roots.is_empty() {
                bad.push(format!(
                    "block {bi} op {oi}: arg {arg:?} has no producer; {op:?}"
                ));
                continue;
            }
            for (kind, var) in &roots {
                if !is_char_producing_op(kind) {
                    bad.push(format!(
                        "block {bi} op {oi}: arg {arg:?} root {var:?} is {kind:?}"
                    ));
                }
            }
        }
    }
    assert!(calls >= 2, "oefmt must call expected_fmt_arg, got {calls}");
    assert!(
        bad.is_empty(),
        "expected_fmt_arg args must be unichr or a char constant: {bad:#?}"
    );
}

fn is_char_producing_op(kind: &OpKind) -> bool {
    match kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments.len() == 1 && segments[0] == "__majit_unichr" => true,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            result_ty,
            ..
        } if *result_ty == ValueType::Int
            && segments.iter().any(|s| s == "option")
            && segments
                .last()
                .is_some_and(|s| s == "unwrap" || s.starts_with("unwrap__spec_")) =>
        {
            true
        }
        OpKind::Call {
            target:
                CallTarget::Method {
                    name,
                    resolved_path,
                    ..
                },
            result_ty,
            ..
        } if *result_ty == ValueType::Int
            && (name == "unwrap" || name.starts_with("unwrap__spec_"))
            && resolved_path
                .as_ref()
                .is_none_or(|p| p.segments.iter().any(|s| s == "option")) =>
        {
            true
        }
        OpKind::ConstInt(_) => true,
        OpKind::FieldRead { field, .. } if field.name == "__pos_0" => true,
        _ => false,
    }
}

fn root_producers(
    graph: &FunctionGraph,
    start: &majit_translate::flowspace::model::Variable,
) -> Vec<(OpKind, majit_translate::flowspace::model::Variable)> {
    use majit_translate::flowspace::model::Variable;
    let mut seen: HashSet<Variable> = HashSet::new();
    let mut stack = vec![start.clone()];
    let mut roots = Vec::new();
    while let Some(var) = stack.pop() {
        if !seen.insert(var.clone()) {
            continue;
        }
        let mut found_op = false;
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() == Some(&var) {
                    found_op = true;
                    roots.push((op.kind.clone(), var.clone()));
                }
            }
            for (slot, input) in block.inputargs.iter().enumerate() {
                if input != &var {
                    continue;
                }
                for pred in &graph.blocks {
                    for link in &pred.exits {
                        if link.target != block.id {
                            continue;
                        }
                        if let Some(LinkArg::Value(src)) = link.args.get(slot) {
                            stack.push(src.clone());
                        }
                    }
                }
            }
        }
        if !found_op && stack.is_empty() && roots.is_empty() {
            // Keep walking predecessor slots already queued.
        }
    }
    roots
}

/// `charpsize2str` is `vec![0u8; size]; copy_bytes(cp, out.as_mut_ptr(), size); out`.
/// `Vec<u8>::as_mut_ptr` of the byte list reads the items through
/// `rgc.nonmoving_raw_ptr_for_resizable_list`, whose result is `CCHARP`.
///
/// `str2charp` is `raw_malloc_or_memory_error(bytes)?` then
/// `copy_bytes(s.as_ptr(), array.cast(), …)`, so the dst is the `Ok`
/// payload of that Result. `raw_malloc` is its own graph (`libc::malloc`
/// then `p.cast()`); that identity cast is `cast_int_to_ptr(CCHARP)`
/// (`lltype.py` `ann_cast_int_to_ptr` / `rffi.cast`). The annotator carries
/// `SomePtr(CCHARP)` onto `copy_bytes` dst.
#[test]
fn charpsize2str_and_str2charp_copy_bytes_dst_is_ccharp() {
    use majit_translate::translator::rtyper::lltypesystem::rffi::CCHARP;

    let llbc = load_llbc(RLIB_LLBC);
    let charpsize = lower_named(llbc, "charpsize2str").expect("lower charpsize2str");
    let str2charp = lower_named(llbc, "str2charp").expect("lower str2charp");
    let raw_malloc = lower_native_raw_malloc(llbc);

    let cdst = copy_bytes_dst(&charpsize).expect("charpsize2str calls copy_bytes");
    let cdst_roots = root_producers(&charpsize, &cdst);
    assert!(
        cdst_roots.iter().any(|(kind, _)| {
            call_path_leaf(kind) == Some("__majit_nonmoving_raw_ptr_for_resizable_list")
        }),
        "charpsize2str copy_bytes dst roots at rgc.nonmoving_raw_ptr_for_resizable_list; cdst_roots={cdst_roots:?}"
    );
    assert!(
        !charpsize
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| call_path_leaf(&op.kind) == Some("ll_vec_as_ptr")),
        "charpsize2str does not go through ll_vec_as_ptr"
    );

    let dst = copy_bytes_dst(&str2charp).expect("str2charp calls copy_bytes");
    let dst_roots = root_producers(&str2charp, &dst);
    let ok_payload_base = dst_roots
        .iter()
        .find_map(|(kind, _)| match kind {
            OpKind::FieldRead { base, field, .. }
                if field.name == "__pos_0"
                    && field
                        .owner_root
                        .as_ref()
                        .is_some_and(|s| s.ends_with("::Ok") && s.contains("Result")) =>
            {
                Some(base)
            }
            _ => None,
        })
        .expect(&format!(
            "str2charp copy_bytes dst is the Ok payload of a Result; dst_roots={dst_roots:?}"
        ));
    let base_roots = root_producers(&str2charp, ok_payload_base);
    let result_arg = base_roots.iter().find_map(|(kind, _)| match kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.last().map(String::as_str) == Some("__cast_instance_intrinsic") => {
            args.first().and_then(LinkArg::as_variable).cloned()
        }
        _ => None,
    });
    // `rffi.py str2charp` copies into the raw allocation. A frontend
    // `cast_pointer` alias may narrow its Result carrier, but is not a
    // required producer when the call already carries that class.
    let result_arg = result_arg.unwrap_or_else(|| ok_payload_base.clone());
    let result_roots = root_producers(&str2charp, &result_arg);
    assert!(
        result_roots
            .iter()
            .any(|(kind, _)| call_path_leaf(kind) == Some("raw_malloc_or_memory_error")),
        "str2charp's Ok payload comes from raw_malloc_or_memory_error's Result; result_roots={result_roots:?}"
    );

    let ret = graph_return_value(&raw_malloc).expect("raw_malloc returns a value");
    let ret_roots = root_producers(&raw_malloc, &ret);
    let wrap = ret_roots.iter().find_map(|(kind, _)| match kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if segments.last().map(String::as_str) == Some("cast_int_to_ptr") && args.len() == 2 => {
            assert!(
                matches!(&args[0], LinkArg::Const(c)
                    if matches!(&c.value, majit_translate::flowspace::model::ConstValue::LowLevelType(ty)
                        if **ty == *CCHARP)),
                "rffi.str2charp allocation must be cast to CCHARP, got {:?}",
                args[0]
            );
            args.get(1).and_then(LinkArg::as_variable).cloned()
        }
        _ => None,
    });
    let wrap = wrap.expect(&format!(
        "raw_malloc return is 2-arg cast_int_to_ptr; ret_roots={ret_roots:?} leaves={:?}",
        call_leaves(&raw_malloc)
    ));
    let malloc_roots = root_producers(&raw_malloc, &wrap);
    assert!(
        malloc_roots
            .iter()
            .any(|(kind, _)| call_path_leaf(kind) == Some("malloc")),
        "cast_int_to_ptr addr roots at libc::malloc; malloc_roots={malloc_roots:?}"
    );

    for graph in [&charpsize, &str2charp, &raw_malloc] {
        for var in graph.iter_variables() {
            *var.annotation.borrow_mut() = None;
            *var.concretetype.borrow_mut() = None;
        }
        legacy_annotator::annotate(graph);
    }
}

fn call_path_leaf(kind: &OpKind) -> Option<&str> {
    match kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => segments.last().map(String::as_str),
        _ => None,
    }
}

fn call_leaves(graph: &FunctionGraph) -> Vec<&str> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| call_path_leaf(&op.kind))
        .collect()
}

fn graph_return_value(
    graph: &FunctionGraph,
) -> Option<majit_translate::flowspace::model::Variable> {
    graph
        .blocks
        .iter()
        .find(|block| block.id == graph.returnblock)
        .and_then(|block| block.inputargs.first().cloned())
}

/// Host `raw_malloc`: `libc::malloc` then `p.cast()` (`#[cfg(not(target_arch
/// = "wasm32"))]`). The wasm body uses `std::alloc::alloc`.
fn lower_native_raw_malloc(llbc: &'static Llbc) -> FunctionGraph {
    let graph = lower_named(llbc, "raw_malloc").expect("lower raw_malloc");
    assert_eq!(
        graph.name.as_str(),
        "majit_rlib::rffi::buffer::raw_malloc",
        "raw_malloc graph is the rffi buffer helper"
    );
    assert!(
        call_leaves(&graph).contains(&"malloc"),
        "native raw_malloc calls libc::malloc; leaves={:?}",
        call_leaves(&graph)
    );
    graph
}

fn copy_bytes_dst(graph: &FunctionGraph) -> Option<majit_translate::flowspace::model::Variable> {
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("copy_bytes") {
                continue;
            }
            return args.get(1).and_then(LinkArg::as_variable).cloned();
        }
    }
    None
}

/// Production `getcalldescr` panics in `w_code_qualname_obj` calling
/// residual `w_code_get_ptr`: FUNC.RESULT Int, caller cell GCREF.
/// A flavor-raw `*const CodeObject` result is `rclass.py getgcflavor`
/// raw `Ptr`, `history.getkind` int. Publish that cell.
#[test]
fn w_code_qualname_obj_get_ptr_call_result_is_raw_int() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["pycode", "gc_roots", "unicodeobject"],
        &[
            "w_code_qualname_obj",
            "w_code_get_ptr",
            "publish_code_slot_store_rooting",
            "push_roots",
            "pin_root",
            "root_scope_close",
            "w_str_new",
            "base",
            "get",
            "owner",
        ],
    )
    .expect("lower w_code_qualname_obj and residual get_ptr callees");

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
    majit_ir::descr::register_struct_origins(program.struct_origins.clone());

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.set_enum_variant_by_discriminant(program.enum_variant_by_discriminant.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    callcontrol.unsafe_fn_stubs = program.unsafe_fn_stubs.clone();
    callcontrol.foreign_opaque_method_externals = program.foreign_opaque_method_externals.clone();

    let crate_prefix = |module: &str| -> &'static str {
        match module {
            "gc_roots" | "unicodeobject" => "pyre_object",
            _ => "pyre_interpreter",
        }
    };
    let mut qualname_path = None;
    let mut get_ptr_path = None;
    let mut qualname_graph = None;
    let mut get_ptr_return_type = None;
    let mut get_ptr_hints = None;
    for func in &program.functions {
        let mut graph = func.graph().clone();
        if let Some(rt) = &func.return_type {
            graph = graph.with_return_type(rt.clone());
        }
        let mut paths = Vec::new();
        if func.module_path.is_empty() {
            paths.push(CallPath::from_segments([func.name.as_str()]));
        } else {
            paths.push(CallPath::from_segments([
                func.module_path.as_str(),
                func.name.as_str(),
            ]));
            paths.push(CallPath::from_segments([
                crate_prefix(&func.module_path),
                func.module_path.as_str(),
                func.name.as_str(),
            ]));
        }
        if let Some(owner) = func.self_ty_root.as_deref() {
            paths.push(CallPath::for_impl_method(owner, &func.name));
        }
        for path in paths {
            if func.name == "w_code_qualname_obj" && path.segments.len() == 3 {
                qualname_path = Some(path.clone());
                qualname_graph = Some(graph.clone());
            }
            if func.name == "w_code_get_ptr" && path.segments.len() == 3 {
                get_ptr_path = Some(path.clone());
                get_ptr_return_type = func.return_type.clone();
                get_ptr_hints = Some(func.hints.clone());
            }
            if func.hints.is_empty() {
                callcontrol.register_function_graph(path, graph.clone());
            } else {
                callcontrol.register_function_graph_with_hints(
                    path,
                    graph.clone(),
                    func.hints.clone(),
                );
            }
        }
    }
    let qualname_path = qualname_path.expect("pyre_interpreter::pycode::w_code_qualname_obj");
    let get_ptr_path = get_ptr_path.expect("pyre_interpreter::pycode::w_code_get_ptr");
    let qualname_graph = qualname_graph.expect("w_code_qualname_obj graph");
    callcontrol.add_candidate_graph(qualname_path.clone());

    let mut cw = CodeWriter::new();
    cw.run_two_phase_prepass(&callcontrol);
    let published =
        cw.dual_gate_publish_concretetypes(&qualname_graph, &mut callcontrol, &qualname_path);
    let is_match = published.is_some();

    let declared_kind = callcontrol.declared_return_kind(&get_ptr_path);
    let (result_ty, dest, dest_ll, dest_ann, dest_kind) = qualname_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                result_ty,
                ..
            } if segments.last().map(String::as_str) == Some("w_code_get_ptr") => {
                let dest = op.result.as_ref().expect("get_ptr dest").clone();
                let dest_ll = dest.concretetype();
                let dest_kind = dest_ll.as_ref().map(getkind);
                let dest_ann = dest.annotation.borrow().as_ref().map(|s| format!("{s:?}"));
                Some((result_ty.clone(), dest, dest_ll, dest_ann, dest_kind))
            }
            _ => None,
        })
        .expect("w_code_qualname_obj calls w_code_get_ptr");
    let (twin_ll, twin_ann, twin_kind) = published
        .as_ref()
        .and_then(|m| m.get(&dest))
        .map(|twin| {
            let twin_ll = twin.concretetype();
            let twin_kind = twin_ll.as_ref().map(getkind);
            let twin_ann = twin.annotation.borrow().as_ref().map(|s| format!("{s:?}"));
            (twin_ll, twin_ann, twin_kind)
        })
        .unwrap_or((None, Some("<dest not in twins>".into()), None));

    assert_eq!(
        dest_kind,
        Some(ConcreteType::Signed),
        "w_code_get_ptr dest must be getkind int (rclass.py getgcflavor raw). \
         match={is_match} declared_kind={declared_kind:?} \
         result_ty={result_ty:?} get_ptr_return_type={get_ptr_return_type:?} \
         get_ptr_hints={get_ptr_hints:?} harvest_codeobject={} \
         dest_ll={dest_ll:?} dest_ann={dest_ann:?} \
         twin_ll={twin_ll:?} twin_ann={twin_ann:?} twin_kind={twin_kind:?}",
        program
            .raw_alloc_flavor_structs
            .iter()
            .any(|k| k.contains("CodeObject")),
    );
}

/// Production `stack_base` inlines `code()` (`&CodeObject`). Front result_ty
/// is `Ref(Some("bytecode::CodeObject"))` and `code()`'s calldescr RESULT is
/// Int (`rclass.py getgcflavor` raw). Dual-gate of the caller must publish
/// that dest as Signed so assemble emits `inline_call_r_i`, matching the
/// callee's `INT_RETURN`.
#[test]
fn stack_base_code_call_dest_is_raw_int_after_dual_gate() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["pyframe", "pycode"],
        &[
            "stack_base",
            "code",
            "ncells",
            "w_code_get_ptr",
            "w_code_npure_cellvars",
        ],
    )
    .expect("lower stack_base, code, and residual get_ptr callees");

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
    majit_ir::descr::register_struct_origins(program.struct_origins.clone());

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.set_enum_variant_by_discriminant(program.enum_variant_by_discriminant.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    callcontrol.unsafe_fn_stubs = program.unsafe_fn_stubs.clone();
    callcontrol.foreign_opaque_method_externals = program.foreign_opaque_method_externals.clone();

    let mut stack_path = None;
    let mut code_path = None;
    let mut stack_graph = None;
    let mut code_graph = None;
    let mut code_return_type = None;
    let mut code_hints = None;
    let mut stack_candidates = Vec::new();
    let mut code_candidates = Vec::new();
    let mut seen_names: Vec<String> = Vec::new();
    for func in &program.functions {
        seen_names.push(format!(
            "{}::{} self={:?} hints={:?}",
            func.module_path, func.name, func.self_ty_root, func.hints
        ));
        let mut graph = func.graph().clone();
        if let Some(rt) = &func.return_type {
            graph = graph.with_return_type(rt.clone());
        }
        let mut paths = Vec::new();
        let mut module_segs: Vec<&str> = func
            .module_path
            .split("::")
            .filter(|s| !s.is_empty())
            .collect();
        if module_segs.is_empty() {
            paths.push(CallPath::from_segments([func.name.as_str()]));
        } else {
            module_segs.push(func.name.as_str());
            paths.push(CallPath::from_segments(module_segs.iter().copied()));
            let mut crate_segs = vec!["pyre_interpreter"];
            crate_segs.extend(module_segs.iter().copied());
            paths.push(CallPath::from_segments(crate_segs));
        }
        if let Some(owner) = func.self_ty_root.as_deref() {
            paths.push(CallPath::for_impl_method(owner, &func.name));
            if let Some(leaf) = owner.rsplit("::").next() {
                if leaf != owner {
                    paths.push(CallPath::for_impl_method(leaf, &func.name));
                }
            }
        }
        let is_pyframe = func
            .self_ty_root
            .as_deref()
            .is_some_and(|s| s.ends_with("PyFrame"));
        let is_stack = func.name == "stack_base" && is_pyframe;
        let is_code = func.name == "code" && is_pyframe;
        for path in paths {
            let production_stack =
                is_stack && path.canonical_key() == "pyre_interpreter::pyframe::<Impl>::stack_base";
            let production_code =
                is_code && path.canonical_key() == "pyre_interpreter::pyframe::<Impl>::code";
            if is_stack {
                stack_candidates.push(path.clone());
            }
            if is_code {
                code_candidates.push(path.clone());
            }
            if production_stack {
                stack_path = Some(path.clone());
                stack_graph = Some(graph.clone());
            }
            if production_code {
                code_path = Some(path.clone());
                code_graph = Some(graph.clone());
                code_return_type = func.return_type.clone();
                code_hints = Some(func.hints.clone());
            }
            if func.hints.is_empty() {
                callcontrol.register_function_graph(path, graph.clone());
            } else {
                callcontrol.register_function_graph_with_hints(
                    path,
                    graph.clone(),
                    func.hints.clone(),
                );
            }
        }
    }
    let stack_path = stack_path.unwrap_or_else(|| {
        panic!("pyre_interpreter::pyframe::<Impl>::stack_base; seen={seen_names:?}")
    });
    let code_path = code_path
        .unwrap_or_else(|| panic!("pyre_interpreter::pyframe::<Impl>::code; seen={seen_names:?}"));
    let stack_graph = stack_graph.expect("stack_base graph");
    let code_graph = code_graph.expect("code graph");
    for path in stack_candidates
        .iter()
        .chain(code_candidates.iter())
        .cloned()
    {
        callcontrol.add_candidate_graph(path);
    }
    for path in &code_candidates {
        let _ = callcontrol.get_jitcode(path);
    }

    let mut cw = CodeWriter::new();
    cw.run_two_phase_prepass(&callcontrol);
    let skip = cw
        .dual_gate_skip_reason(&stack_graph, &callcontrol)
        .expect("dual-gate skip reason of stack_base");
    let published = cw.dual_gate_publish_concretetypes(&stack_graph, &mut callcontrol, &stack_path);
    let is_match = published.is_some();

    let bk = callcontrol.bookkeeper().expect("dual-gate bookkeeper");
    let classdef = bk.getuniqueclassdef_for_struct_root("bytecode::CodeObject");
    let flavor = classdef.as_ref().ok().and_then(|cd| {
        cd.borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_get("_alloc_flavor_")
            .and_then(|v| v.as_text().map(str::to_string))
    });
    let classdef_name = classdef.as_ref().ok().map(|cd| cd.borrow().name.clone());
    let harvest_keys: Vec<_> = program
        .raw_alloc_flavor_structs
        .iter()
        .filter(|k| k.contains("CodeObject"))
        .cloned()
        .collect();

    fn target_is_code(target: &CallTarget) -> bool {
        match target {
            CallTarget::FunctionPath { segments, .. } => {
                segments.last().is_some_and(|s| s == "code")
            }
            CallTarget::Method { name, .. } => name == "code",
            _ => false,
        }
    }

    let (result_ty, dest, dest_ll, dest_ann, dest_kind, code_target) = stack_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target, result_ty, ..
            } if target_is_code(target) => {
                let dest = op.result.as_ref().expect("code() dest").clone();
                let dest_ll = dest.concretetype();
                let dest_kind = dest_ll.as_ref().map(getkind);
                let dest_ann = dest.annotation.borrow().as_ref().map(|s| format!("{s:?}"));
                Some((
                    result_ty.clone(),
                    dest,
                    dest_ll,
                    dest_ann,
                    dest_kind,
                    format!("{target}"),
                ))
            }
            _ => None,
        })
        .expect("stack_base calls code");
    let (twin_ll, twin_ann, twin_kind) = published
        .as_ref()
        .and_then(|m| m.get(&dest))
        .map(|twin| {
            let twin_ll = twin.concretetype();
            let twin_kind = twin_ll.as_ref().map(getkind);
            let twin_ann = twin.annotation.borrow().as_ref().map(|s| format!("{s:?}"));
            (twin_ll, twin_ann, twin_kind)
        })
        .unwrap_or((None, Some("<dest not in twins>".into()), None));

    let code_rb_kind =
        FunctionGraph::concretetype_of(&code_graph.blocks[code_graph.returnblock.0].inputargs[0]);
    cw.dual_gate_publish_concretetypes(&code_graph, &mut callcontrol, &code_path);
    let code_rb_kind_after =
        FunctionGraph::concretetype_of(&code_graph.blocks[code_graph.returnblock.0].inputargs[0]);
    let get_ptr_dump = code_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call {
                target, result_ty, ..
            } if matches!(
                target,
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().map(String::as_str) == Some("w_code_get_ptr")
            ) =>
            {
                let dest = op.result.as_ref()?;
                Some(format!(
                    "result_ty={result_ty:?} dest_kind={:?} dest_ann={:?}",
                    dest.concretetype().as_ref().map(getkind),
                    dest.annotation.borrow().as_ref().map(|s| format!("{s:?}")),
                ))
            }
            _ => None,
        });

    eprintln!(
        "stack_base dual_gate dump match={is_match} skip={skip:?} result_ty={result_ty:?} \
         target={code_target} code_return_type={code_return_type:?} code_hints={code_hints:?} \
         harvest_keys={harvest_keys:?} classdef_name={classdef_name:?} flavor={flavor:?} \
         dest_kind={dest_kind:?} dest_ll={dest_ll:?} dest_ann={dest_ann:?} \
         twin_ll={twin_ll:?} twin_ann={twin_ann:?} twin_kind={twin_kind:?} \
         code_rb_before={code_rb_kind:?} code_rb_after={code_rb_kind_after:?} \
         get_ptr={get_ptr_dump:?} code_candidates={code_candidates:?}"
    );
    assert_eq!(
        dest_kind,
        Some(ConcreteType::Signed),
        "stack_base call to code() dest must be getkind int (rclass.py getgcflavor raw). \
         match={is_match} skip={skip:?} result_ty={result_ty:?} target={code_target} \
         harvest_keys={harvest_keys:?} classdef_name={classdef_name:?} flavor={flavor:?} \
         dest_ll={dest_ll:?} dest_ann={dest_ann:?} \
         twin_ll={twin_ll:?} twin_ann={twin_ann:?} twin_kind={twin_kind:?}"
    );

    let field_dump: Vec<String> = stack_graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => Some(format!(
                "{}.{} ty={ty:?} declared_gc={:?} base={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.owner_declared_gc,
                FunctionGraph::concretetype_of(base)
            )),
            _ => None,
        })
        .collect();
    eprintln!("stack_base field dump {field_dump:?}");
    assert!(
        stack_graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "varnames"
                        && field.owner_declared_gc == Some(false)
                        && *ty == ValueType::Int
            )),
        "CodeObject.varnames must be Int with owner_declared_gc Raw (getgcflavor). fields={field_dump:?}"
    );

    let config = GraphTransformConfig::default();
    let probe_graph = majit_translate::model::copygraph(&stack_graph);
    let rewritten = Transformer::new(&config)
        .with_callcontrol(&mut callcontrol)
        .transform(&probe_graph);
    let inline_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::InlineCall { result_kind, .. } => Some(format!(
                "inline result_kind={result_kind} dest={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::CallResidual { result_kind, .. } => Some(format!(
                "residual result_kind={result_kind} dest={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            _ => None,
        })
        .collect();
    eprintln!("stack_base transformer dump {inline_dump:?}");
    let jitcode = std::sync::Arc::new(JitCode::new("stack_base"));
    cw.transform_graph_to_jitcode(
        &stack_graph,
        &stack_path,
        &mut callcontrol,
        &config,
        &jitcode,
        false,
        0,
    );
    let dest_kind_after = dest.concretetype().as_ref().map(getkind);
    let body = jitcode.body();
    let has_r_i = body.code.contains(&BC_INLINE_CALL_R_I);
    let has_r_r = body.code.contains(&BC_INLINE_CALL_R_R);
    eprintln!(
        "stack_base assemble dump dest_after={dest_kind_after:?} first={:?} \
         has_r_i={has_r_i} has_r_r={has_r_r} code={:?} resulttypes={:?}",
        body.code.first().copied(),
        body.code,
        body.resulttypes
    );
    assert_eq!(
        dest_kind_after,
        Some(ConcreteType::Signed),
        "promote_gc_field_bases must leave the flavor-raw code() dest Signed. \
         fields={field_dump:?} dest_after={dest_kind_after:?}"
    );
    assert!(
        has_r_i && !has_r_r,
        "stack_base must encode inline_call_r_i for code() and not inline_call_r_r. \
         match={is_match} skip={skip:?} dest_kind={dest_kind:?} dest_after={dest_kind_after:?} \
         twin_kind={twin_kind:?} flavor={flavor:?} classdef_name={classdef_name:?} \
         fields={field_dump:?} has_r_i={has_r_i} has_r_r={has_r_r} first={:?} \
         code={:?} resulttypes={:?}",
        body.code.first().copied(),
        body.code,
        body.resulttypes
    );
}

/// Install `CallControl` layouts the same way `lib.rs` does after
/// `derive_program_metadata`: heuristic rows, then `apply_exact_layout`
/// from `program.exact_layouts`.
fn install_semantic_layouts(callcontrol: &mut CallControl, program: &SemanticProgram) {
    majit_ir::descr::register_struct_ids(program.struct_ids.clone());
    majit_ir::descr::register_struct_origins(program.struct_origins.clone());
    let pair_fields: std::collections::HashMap<String, Vec<(String, String)>> =
        program.struct_fields.fields.clone().into();
    let provider = HeuristicLayoutProvider::from_struct_fields(
        &pair_fields,
        &program.known_struct_names,
        &program.immutable_fields,
        &program.raw_alloc_flavor_structs,
    );
    let mut names: Vec<_> = pair_fields.keys().cloned().collect();
    names.sort();
    for struct_name in names {
        let Some(sid) = majit_ir::descr::struct_id_for_name(&struct_name) else {
            continue;
        };
        let exact = program.exact_layouts.get(&sid);
        let layout = match exact {
            Some(exact) => {
                provider.get_struct_layout_exact(&struct_name, &exact.field_offsets, exact.size)
            }
            None => provider.get_struct_layout(&struct_name),
        };
        if let Some(layout) = layout {
            callcontrol.set_struct_layout(sid, layout);
        }
    }
}

fn dump_struct_field_rows(program: &SemanticProgram, needle: &str) -> Vec<String> {
    let mut keys: Vec<_> = program
        .struct_fields
        .fields
        .iter()
        .filter(|(k, _)| k.contains(needle))
        .map(|(k, rows)| {
            let cols: Vec<String> = rows
                .iter()
                .map(|row| format!("{}:{}", row.name, row.ty))
                .collect();
            format!("{k} => {cols:?}")
        })
        .collect();
    keys.sort();
    keys
}

fn dump_exact_layout(program: &SemanticProgram, spelling: &str) -> String {
    let sid = majit_ir::descr::struct_id_for_name(spelling)
        .unwrap_or_else(|| majit_ir::descr::StructId::from_canonical(spelling));
    match program.exact_layouts.get(&sid) {
        Some(exact) => {
            let mut offs: Vec<_> = exact
                .field_offsets
                .iter()
                .map(|(n, o)| format!("{n}@{o}"))
                .collect();
            offs.sort();
            format!(
                "sid={sid:?} size={:?} align={:?} offs={offs:?}",
                exact.size, exact.align
            )
        }
        None => format!("sid={sid:?} <missing>"),
    }
}

fn dump_row_size_census(cc: &CallControl, owners: &[&str]) {
    for spelling in owners {
        match cc.struct_layout_for(spelling) {
            None => eprintln!("row-size {spelling} <no layout>"),
            Some(layout) => {
                let mut rows = layout.fields.clone();
                rows.sort_by_key(|row| row.offset);
                for i in 0..rows.len() {
                    let row = &rows[i];
                    let gap = if i + 1 < rows.len() {
                        rows[i + 1].offset.saturating_sub(row.offset)
                    } else {
                        layout.size.saturating_sub(row.offset)
                    };
                    let ty = cc.field_type(spelling, &row.name).unwrap_or("?");
                    let bad = row.size > gap;
                    eprintln!(
                        "row-size {spelling}.{} ty={ty} offset={} size={} gap={gap} flag={:?} violate={bad}",
                        row.name, row.offset, row.size, row.flag
                    );
                }
            }
        }
    }
}

fn dump_cc_layout_rows(cc: &CallControl, spelling: &str) -> String {
    match cc.struct_layout_for(spelling) {
        Some(layout) => {
            let rows: Vec<String> = layout
                .fields
                .iter()
                .map(|row| {
                    format!(
                        "{}@{} size={} flag={:?} ty={:?}",
                        row.name, row.offset, row.size, row.flag, row.field_type
                    )
                })
                .collect();
            format!(
                "size={} align={} gckind={:?} n={} rows={rows:?}",
                layout.size,
                layout.align,
                layout.gckind,
                layout.fields.len()
            )
        }
        None => "<no layout>".to_string(),
    }
}

/// `code_getdocstring` reads `CodeObject.constants` as
/// `&(*code_ref).constants` (`FieldRead { taken_by_address: true }`)
/// and `flags` as the CodeFlags word. Both sit on a raw-flavor owner.
/// `varnames` is asserted on the `stack_base` graph, which reads it.
/// The front ty of an address-taken read is not the emitted op;
/// `jtransform.rewrite_op_getfield` turns it into the getsubstruct
/// address (`int_add` / Identity, kind `'i'`).
#[test]
fn code_getdocstring_constants_and_flags_are_raw() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["function", "pycode"],
        &[
            "code_getdocstring",
            "w_code_get_ptr",
            "w_code_const",
            "is_code",
            "function_get_code",
        ],
    )
    .expect("lower code_getdocstring");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "code_getdocstring")
        .expect("code_getdocstring graph");

    let field_dump: Vec<String> = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => Some(format!(
                "{}.{} ty={ty:?} declared_gc={:?} taken_by_address={} base={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.owner_declared_gc,
                field.taken_by_address,
                FunctionGraph::concretetype_of(base)
            )),
            _ => None,
        })
        .collect();
    eprintln!("code_getdocstring field dump {field_dump:?}");

    let mut saw_constants = false;
    let mut saw_flags = false;
    let mut constants_owner = None;
    for op in function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
    {
        let OpKind::FieldRead { field, ty, .. } = &op.kind else {
            continue;
        };
        let owner = field.owner_root.as_deref().unwrap_or("");
        let is_code_object = owner == "CodeObject" || owner.ends_with("::CodeObject");
        if !is_code_object {
            continue;
        }
        if field.name == "constants" {
            saw_constants = true;
            constants_owner = Some(owner.to_string());
            assert!(
                field.taken_by_address,
                "CodeObject.constants is `&(*code_ref).constants`; the front records \
                 FieldRead {{ taken_by_address: true }}. fields={field_dump:?}"
            );
            assert_eq!(
                field.owner_declared_gc,
                Some(false),
                "CodeObject.constants owner_declared_gc must be Raw. fields={field_dump:?}"
            );
        }
        if field.name == "flags" {
            saw_flags = true;
            assert_eq!(
                *ty,
                ValueType::Unsigned,
                "CodeObject.flags FieldRead must be the CodeFlags word. \
                 fields={field_dump:?}"
            );
            assert!(
                !field.taken_by_address,
                "CodeObject.flags is a value load of the CodeFlags word. \
                 fields={field_dump:?}"
            );
            assert_eq!(
                field.owner_declared_gc,
                Some(false),
                "CodeObject.flags owner_declared_gc must be Raw. fields={field_dump:?}"
            );
        }
    }
    assert!(
        saw_constants,
        "code_getdocstring must FieldRead CodeObject.constants. fields={field_dump:?}"
    );
    assert!(
        saw_flags,
        "code_getdocstring must FieldRead CodeObject.flags. fields={field_dump:?}"
    );

    let constants_ty = program
        .struct_fields
        .field_type("bytecode::CodeObject", "constants")
        .or_else(|| program.struct_fields.field_type("CodeObject", "constants"));
    let varnames_ty = program
        .struct_fields
        .field_type("bytecode::CodeObject", "varnames")
        .or_else(|| program.struct_fields.field_type("CodeObject", "varnames"));
    let flags_ty = program
        .struct_fields
        .field_type("bytecode::CodeObject", "flags")
        .or_else(|| program.struct_fields.field_type("CodeObject", "flags"));
    let known_constants: Vec<_> = program
        .known_struct_names
        .iter()
        .filter(|n| n.contains("Constants") || n.contains("CodeObject") || n.contains("CodeFlags"))
        .cloned()
        .collect();
    eprintln!(
        "code_getdocstring decl dump constants_ty={constants_ty:?} varnames_ty={varnames_ty:?} \
         flags_ty={flags_ty:?} known={known_constants:?}"
    );
    eprintln!(
        "code_getdocstring struct_fields CodeObject {}",
        dump_struct_field_rows(&program, "CodeObject").join(" | ")
    );
    eprintln!(
        "code_getdocstring struct_fields Constants {}",
        dump_struct_field_rows(&program, "Constants").join(" | ")
    );

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    install_semantic_layouts(&mut callcontrol, &program);

    eprintln!(
        "code_getdocstring exact CodeObject {}",
        dump_exact_layout(&program, "bytecode::CodeObject")
    );
    eprintln!(
        "code_getdocstring exact Constants {}",
        dump_exact_layout(&program, "bytecode::Constants")
    );
    for spelling in [
        "bytecode::CodeObject",
        "CodeObject",
        constants_owner.as_deref().unwrap_or("bytecode::CodeObject"),
        "bytecode::Constants",
        "Constants",
        "rustpython_compiler_core::bytecode::Constants",
    ] {
        eprintln!(
            "code_getdocstring layout {spelling} {}",
            dump_cc_layout_rows(&callcontrol, spelling)
        );
    }

    let word = majit_translate::layout::target_word_size();
    let exact = majit_translate::codeobject_layout::exact_layout_for("bytecode::CodeObject")
        .expect("native CodeObject exact layout");
    eprintln!(
        "code_getdocstring exact_layout_for size={:?} offs={:?}",
        exact.size,
        {
            let mut offs: Vec<_> = exact
                .field_offsets
                .iter()
                .map(|(n, o)| format!("{n}@{o}"))
                .collect();
            offs.sort();
            offs
        }
    );
    let layout = callcontrol
        .struct_layout_for("bytecode::CodeObject")
        .expect("bytecode::CodeObject layout");
    for name in [
        "source_path",
        "obj_name",
        "qualname",
        "constants",
        "varnames",
        "flags",
    ] {
        match layout.fields.iter().find(|row| row.name == name) {
            Some(row) => eprintln!(
                "code_getdocstring row {name} offset={} size={} flag={:?} wider={} word={word}",
                row.offset,
                row.size,
                row.flag,
                row.flag == majit_ir::descr::ArrayFlag::Struct && row.size > word,
            ),
            None => eprintln!("code_getdocstring row {name} <missing>"),
        }
    }
    let source_path = layout
        .fields
        .iter()
        .find(|row| row.name == "source_path")
        .expect("source_path row");
    assert_eq!(
        source_path.offset as u64, exact.field_offsets["source_path"],
        "source_path offset must match exact_layout_for"
    );
    assert_eq!(source_path.offset, 0);
    assert_eq!(
        source_path.size,
        3 * word,
        "source_path is the inline String/Vec<u8> header"
    );
    assert_eq!(source_path.flag, majit_ir::descr::ArrayFlag::Struct);
    for name in ["obj_name", "qualname", "constants", "varnames", "flags"] {
        let row = layout
            .fields
            .iter()
            .find(|row| row.name == name)
            .unwrap_or_else(|| panic!("{name} row"));
        assert_eq!(
            row.offset as u64, exact.field_offsets[name],
            "{name} offset must match exact_layout_for"
        );
    }

    let config = GraphTransformConfig::default();
    let probe_graph = majit_translate::model::copygraph(function.graph());
    let rewritten = Transformer::new(&config)
        .with_callcontrol(&mut callcontrol)
        .transform(&probe_graph);

    let rewrite_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, ty, .. } => Some(format!(
                "getfield {}.{} ty={ty:?} taken={} dest_kind={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.taken_by_address,
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "add" => Some(format!(
                "int_add result_ty={result_ty:?} dest_kind={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            _ => None,
        })
        .collect();
    eprintln!("code_getdocstring rewrite dump {rewrite_dump:?}");
    eprintln!("code_getdocstring rewrite notes {:?}", rewritten.notes);

    let leftover_raw_r: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                field, ty, pure, ..
            } if field.owner_declared_gc == Some(false)
                && matches!(ty, ValueType::Ref(_))
                && !*pure =>
            {
                Some(format!(
                    "{}.{} ty={ty:?} taken={}",
                    field.owner_root.as_deref().unwrap_or("?"),
                    field.name,
                    field.taken_by_address
                ))
            }
            _ => None,
        })
        .collect();
    assert!(
        leftover_raw_r.is_empty(),
        "jtransform must not leave getfield_raw_r. leftover={leftover_raw_r:?} \
         rewrite={rewrite_dump:?} fields={field_dump:?}"
    );

    let leftover_constants = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "constants"
                        && field
                            .owner_root
                            .as_deref()
                            .is_some_and(|o| o == "CodeObject" || o.ends_with("::CodeObject"))
            )
        });
    assert!(
        !leftover_constants,
        "CodeObject.constants must become the getsubstruct address, not a leftover getfield. \
         rewrite={rewrite_dump:?} fields={field_dump:?}"
    );

    let constants_addr = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "add" && *result_ty == ValueType::Int => {
                Some(FunctionGraph::concretetype_of(op.result.as_ref()?))
            }
            _ => None,
        });
    let constants_kind = function
        .graph()
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. }
                if field.name == "constants"
                    && field
                        .owner_root
                        .as_deref()
                        .is_some_and(|o| o == "CodeObject" || o.ends_with("::CodeObject")) =>
            {
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            }
            _ => None,
        });
    eprintln!(
        "code_getdocstring constants_addr={constants_addr:?} constants_kind={constants_kind:?}"
    );
    let addr_is_int = constants_addr == Some(ConcreteType::Signed);
    let dest_is_int = constants_kind == Some(ConcreteType::Signed);
    assert!(
        addr_is_int || dest_is_int,
        "constants getsubstruct address must be kind 'i' (int_add or Identity of the raw base). \
         addr={constants_addr:?} dest={constants_kind:?} rewrite={rewrite_dump:?}"
    );
}

/// `code.instructions` is the by-value `CodeUnits` header. After the
/// include it is FLAG_STRUCT, and jtransform emits the getsubstruct
/// address (`int_add` 72). `base_op` is the graph that FieldReads it
/// (`code.instructions.get`); the wrap that also reads `.len()` is a
/// virtualizable class with no jitdriver in this test.
#[test]
fn after_return_generator_index_instructions_is_substruct_address() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["pycode"],
        &[
            "base_op",
            "after_return_generator_index",
            "first_traceable_index",
        ],
    )
    .expect("lower base_op");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "base_op")
        .expect("base_op graph");

    let field_dump: Vec<String> = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => Some(format!(
                "{}.{} ty={ty:?} declared_gc={:?} taken_by_address={} base={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.owner_declared_gc,
                field.taken_by_address,
                FunctionGraph::concretetype_of(base)
            )),
            _ => None,
        })
        .collect();
    eprintln!("instructions field dump {field_dump:?}");

    let known_units: Vec<_> = program
        .known_struct_names
        .iter()
        .filter(|n| n.contains("CodeUnits") || n.contains("CodeFlags") || n.contains("CodeObject"))
        .cloned()
        .collect();
    eprintln!("instructions known {known_units:?}");
    eprintln!(
        "instructions struct_fields CodeUnits {}",
        dump_struct_field_rows(&program, "CodeUnits").join(" | ")
    );
    eprintln!(
        "instructions struct_fields CodeFlags {}",
        dump_struct_field_rows(&program, "CodeFlags").join(" | ")
    );

    let mut saw_instructions = false;
    for op in function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
    {
        let OpKind::FieldRead { field, .. } = &op.kind else {
            continue;
        };
        let owner = field.owner_root.as_deref().unwrap_or("");
        if field.name != "instructions"
            || !(owner == "CodeObject" || owner.ends_with("::CodeObject"))
        {
            continue;
        }
        saw_instructions = true;
        assert!(
            field.taken_by_address,
            "CodeObject.instructions is `&code.instructions`. fields={field_dump:?}"
        );
        assert_eq!(
            field.owner_declared_gc,
            Some(false),
            "CodeObject.instructions owner is raw. fields={field_dump:?}"
        );
    }
    assert!(
        saw_instructions,
        "must FieldRead CodeObject.instructions. fields={field_dump:?}"
    );

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    install_semantic_layouts(&mut callcontrol, &program);

    eprintln!(
        "instructions layout bytecode::CodeObject {}",
        dump_cc_layout_rows(&callcontrol, "bytecode::CodeObject")
    );
    eprintln!(
        "instructions layout bytecode::CodeUnits {}",
        dump_cc_layout_rows(&callcontrol, "bytecode::CodeUnits")
    );
    eprintln!(
        "instructions exact CodeUnits {}",
        dump_exact_layout(&program, "bytecode::CodeUnits")
    );

    let word = majit_translate::layout::target_word_size();
    let exact = majit_translate::codeobject_layout::exact_layout_for("bytecode::CodeObject")
        .expect("native CodeObject exact layout");
    let layout = callcontrol
        .struct_layout_for("bytecode::CodeObject")
        .expect("CodeObject layout");
    let instructions = layout
        .fields
        .iter()
        .find(|row| row.name == "instructions")
        .expect("instructions row");
    let locations = layout
        .fields
        .iter()
        .find(|row| row.name == "locations")
        .expect("locations row");
    eprintln!(
        "instructions row offset={} size={} flag={:?} locations@{} size={} CodeUnits_exact_size=48",
        instructions.offset, instructions.size, instructions.flag, locations.offset, locations.size
    );
    dump_row_size_census(
        &callcontrol,
        &[
            "bytecode::CodeObject",
            "CodeObject",
            "MapNode",
            "MapNodeAbstractAttribute",
            "MapNodeTerminatorView",
            "CellFamily",
            "QuasiImmut",
            "QuasiImmutLock",
            "LoopHeaderInfo",
            "WritableBuffer",
        ],
    );
    assert_eq!(
        instructions.offset as u64, exact.field_offsets["instructions"],
        "instructions offset must match exact_layout_for"
    );
    assert_eq!(instructions.offset, 72);
    assert_eq!(instructions.flag, majit_ir::descr::ArrayFlag::Struct);
    assert!(
        instructions.size > word,
        "instructions FLAG_STRUCT must be wider than a word so the leftover \
         rewrite takes the getsubstruct address. size={}",
        instructions.size
    );
    assert_eq!(
        locations.offset as u64, exact.field_offsets["locations"],
        "locations offset must stay at exact_layout_for"
    );
    assert_eq!(locations.offset, 72 + 6 * word);

    let config = GraphTransformConfig::default();
    let probe_graph = majit_translate::model::copygraph(function.graph());
    let rewritten = Transformer::new(&config)
        .with_callcontrol(&mut callcontrol)
        .transform(&probe_graph);

    let rewrite_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, ty, .. } => Some(format!(
                "getfield {}.{} ty={ty:?} taken={} dest_kind={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.taken_by_address,
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "add" => Some(format!(
                "int_add result_ty={result_ty:?} dest_kind={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            _ => None,
        })
        .collect();
    eprintln!("instructions rewrite dump {rewrite_dump:?}");
    eprintln!("instructions rewrite notes {:?}", rewritten.notes);

    let leftover_raw_r: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                field, ty, pure, ..
            } if field.owner_declared_gc == Some(false)
                && matches!(ty, ValueType::Ref(_))
                && !*pure =>
            {
                Some(format!(
                    "{}.{} ty={ty:?} taken={}",
                    field.owner_root.as_deref().unwrap_or("?"),
                    field.name,
                    field.taken_by_address
                ))
            }
            _ => None,
        })
        .collect();
    assert!(
        leftover_raw_r.is_empty(),
        "jtransform must not leave getfield_raw_r. leftover={leftover_raw_r:?} \
         rewrite={rewrite_dump:?} fields={field_dump:?}"
    );

    let leftover_instructions = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "instructions"
                        && field
                            .owner_root
                            .as_deref()
                            .is_some_and(|o| o == "CodeObject" || o.ends_with("::CodeObject"))
            )
        });
    assert!(
        !leftover_instructions,
        "CodeObject.instructions must become the getsubstruct address. rewrite={rewrite_dump:?}"
    );

    let addr = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .find_map(|op| match &op.kind {
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "add" && *result_ty == ValueType::Int => {
                Some(FunctionGraph::concretetype_of(op.result.as_ref()?))
            }
            _ => None,
        });
    eprintln!("instructions addr kind={addr:?}");
    assert_eq!(
        addr,
        Some(ConcreteType::Signed),
        "instructions getsubstruct address must be kind 'i'. rewrite={rewrite_dump:?}"
    );
}

/// `code.names[idx].as_ref()` is a borrow of a by-value String item
/// in the host `Box<[String]>` FatData region. The Index is a raw
/// `nolength` ArrayRead of the item address; jtransform rewrites it
/// as `getarraysubstruct` (`int_mul` 24 + `int_add`, kind `'i'`).
#[test]
fn execute_store_name_names_index_is_raw_item_address() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["pyopcode", "eval", "pyframe", "pycode"],
        &[
            "execute_store_name",
            "opcode_store_name",
            "store_name_value",
            "store_name",
            "get_w_locals",
            "getdebug_data",
            "get_or_create_w_locals",
            "w_code_getname_w_or_new",
            "w_code_getname_w",
        ],
    )
    .expect("lower execute_store_name");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "execute_store_name")
        .expect("execute_store_name graph");

    let array_dump: Vec<String> = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayRead {
                item_ty,
                array_type_id,
                nolength,
                taken_by_address,
                ..
            } => Some(format!(
                "ArrayRead item_ty={item_ty:?} array_type_id={array_type_id:?} \
                 nolength={nolength} taken_by_address={taken_by_address}"
            )),
            _ => None,
        })
        .collect();
    let mut saw_names = false;
    for op in function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
    {
        let OpKind::ArrayRead {
            array_type_id,
            nolength,
            taken_by_address,
            ..
        } = &op.kind
        else {
            continue;
        };
        let id = array_type_id.as_deref().unwrap_or("");
        if id != "Vec<String>" {
            continue;
        }
        saw_names = true;
        assert!(
            *nolength,
            "names FatData is a raw items region. dump={array_dump:?}"
        );
        assert!(
            *taken_by_address,
            "`code.names[idx].as_ref()` is the item address. dump={array_dump:?}"
        );
    }
    assert!(
        saw_names,
        "execute_store_name must ArrayRead Vec<String>. dump={array_dump:?}"
    );

    let saw_fat_len = function.graph().blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => {
                field.vec_part == Some(majit_translate::model::VecFieldPart::FatLen)
            }
            _ => false,
        })
    });
    assert!(
        saw_fat_len,
        "fat-box Index bounds-checks against the sibling FatLen part"
    );
    let saw_lt = function.graph().blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "lt"
            )
        })
    });
    assert!(saw_lt, "guard_item_index emits index < FatLen");
    let saw_bounds_panic = function.graph().blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.iter().any(|s| s.contains("ll_slice_bounds_panic")),
            _ => false,
        })
    });
    assert!(
        saw_bounds_panic,
        "out-of-bounds arm is ll_slice_bounds_panic"
    );

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    install_semantic_layouts(&mut callcontrol, &program);

    let config = GraphTransformConfig::default();
    let probe_graph = majit_translate::model::copygraph(function.graph());
    let rewritten = Transformer::new(&config)
        .with_callcontrol(&mut callcontrol)
        .transform(&probe_graph);

    let rewrite_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayRead {
                array_type_id,
                nolength,
                taken_by_address,
                item_ty,
                ..
            } => Some(format!(
                "ArrayRead id={array_type_id:?} nolength={nolength} \
                 taken={taken_by_address} item={item_ty:?}"
            )),
            OpKind::BinOp {
                op: bin, result_ty, ..
            } => Some(format!(
                "BinOp {bin} result_ty={result_ty:?} dest={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::ConstInt(n) => Some(format!("ConstInt {n}")),
            OpKind::FieldRead { field, ty, .. } => Some(format!(
                "getfield {}.{} ty={ty:?} dest={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            _ => None,
        })
        .collect();

    let word = 3 * majit_translate::layout::target_word_size() as i64;
    let saw_mul_24 = rewritten.graph.blocks.iter().any(|block| {
        let ops = &block.operations;
        ops.windows(2).any(|pair| {
            matches!(&pair[0].kind, OpKind::ConstInt(n) if *n == word)
                && matches!(
                    &pair[1].kind,
                    OpKind::BinOp { op, result_ty, .. }
                        if op == "int_mul" && *result_ty == ValueType::Int
                )
        })
    });
    assert!(
        saw_mul_24,
        "getarraysubstruct is int_mul by sizeof(String)=24. dump={rewrite_dump:?}"
    );
    let saw_int_add = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "int_add" && *result_ty == ValueType::Int => op
                .result
                .as_ref()
                .is_some_and(|v| FunctionGraph::concretetype_of(v) == ConcreteType::Signed),
            _ => false,
        })
    });
    assert!(
        saw_int_add,
        "getarraysubstruct is int_add kind 'i'. dump={rewrite_dump:?}"
    );
    let leftover_gc_item = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::ArrayRead { nolength, .. } => !nolength,
            _ => false,
        })
    });
    assert!(
        !leftover_gc_item,
        "no getarrayitem_gc after rewrite. dump={rewrite_dump:?}"
    );
    let leftover_raw_r = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead {
                field, ty, pure, ..
            } if field.owner_declared_gc == Some(false)
                && matches!(ty, ValueType::Ref(_))
                && !*pure
                && !field.taken_by_address =>
            {
                true
            }
            _ => false,
        })
    });
    assert!(
        !leftover_raw_r,
        "no getfield_raw_r leftover. dump={rewrite_dump:?}"
    );
}

/// `w_code_const` indexes `code_constants` (`from_raw_parts` of
/// `Constants` FatData / FatLen) as `&constants[idx]`. rustc does not
/// inline `code_constants`, so the Index is of that Call dest: a raw
/// `nolength` ArrayRead of `[ConstantData]` (headerless slice spelling)
/// with `taken_by_address`. jtransform rewrites it as `getarraysubstruct`
/// (`int_mul` sizeof(ConstantData) + `int_add`, kind `'i'`).
#[test]
fn w_code_const_constants_index_is_raw_item_address() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["pycode", "pyframe"],
        &[
            "w_code_const",
            "code_constants",
            "pyobject_from_constant",
            "realize_code_constant",
        ],
    )
    .expect("lower w_code_const");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "w_code_const")
        .expect("w_code_const graph");

    let ops_dump: Vec<String> = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayRead {
                item_ty,
                array_type_id,
                nolength,
                taken_by_address,
                ..
            } => Some(format!(
                "ArrayRead item_ty={item_ty:?} array_type_id={array_type_id:?} \
                 nolength={nolength} taken_by_address={taken_by_address}"
            )),
            OpKind::ArrayLen {
                array_type_id,
                nolength,
                ..
            } => Some(format!(
                "ArrayLen array_type_id={array_type_id:?} nolength={nolength}"
            )),
            OpKind::FieldRead { field, ty, .. } => Some(format!(
                "FieldRead {}.{} vec_part={:?} ty={ty:?} dest={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.vec_part,
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(format!("Call {}", segments.join("::"))),
            _ => None,
        })
        .collect();
    let array_dump = ops_dump.clone();
    let code_constants_dump: Vec<String> = program
        .functions
        .iter()
        .find(|f| f.name == "code_constants")
        .map(|f| {
            f.graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .map(|op| match &op.kind {
                    OpKind::ArrayRead {
                        item_ty,
                        array_type_id,
                        nolength,
                        taken_by_address,
                        ..
                    } => format!(
                        "ArrayRead item_ty={item_ty:?} array_type_id={array_type_id:?} \
                         nolength={nolength} taken_by_address={taken_by_address}"
                    ),
                    OpKind::ArrayLen {
                        array_type_id,
                        nolength,
                        ..
                    } => format!("ArrayLen array_type_id={array_type_id:?} nolength={nolength}"),
                    OpKind::FieldRead { field, ty, .. } => format!(
                        "FieldRead {}.{} vec_part={:?} ty={ty:?}",
                        field.owner_root.as_deref().unwrap_or("?"),
                        field.name,
                        field.vec_part
                    ),
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } => format!("Call {}", segments.join("::")),
                    other => format!("{other:?}"),
                })
                .collect()
        })
        .unwrap_or_else(|| vec!["<no code_constants graph>".to_string()]);
    let mut saw_constants = false;
    for op in function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
    {
        let OpKind::ArrayRead {
            array_type_id,
            nolength,
            taken_by_address,
            item_ty,
            ..
        } = &op.kind
        else {
            continue;
        };
        let id = array_type_id.as_deref().unwrap_or("");
        if id != "[ConstantData]" && id != "Vec<ConstantData>" {
            continue;
        }
        saw_constants = true;
        assert!(
            *nolength,
            "constants items region is headerless. dump={array_dump:?}"
        );
        assert!(
            *taken_by_address,
            "`&constants[idx]` is the item address. dump={array_dump:?}"
        );
        assert!(
            matches!(item_ty, ValueType::Ref(Some(name)) if name == "bytecode::ConstantData"),
            "borrowed ConstantData item. dump={array_dump:?}"
        );
    }
    assert!(
        saw_constants,
        "w_code_const must ArrayRead [ConstantData] or Vec<ConstantData>. \
         dump={array_dump:?} code_constants={code_constants_dump:?}"
    );

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    install_semantic_layouts(&mut callcontrol, &program);

    let item_bytes = callcontrol
        .struct_layout_for("ConstantData")
        .map(|layout| layout.size as i64)
        .expect("ConstantData Charon layout after --include");

    let config = GraphTransformConfig::default();
    let probe_graph = majit_translate::model::copygraph(function.graph());
    let rewritten = Transformer::new(&config)
        .with_callcontrol(&mut callcontrol)
        .transform(&probe_graph);

    let rewrite_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayRead {
                array_type_id,
                nolength,
                taken_by_address,
                item_ty,
                ..
            } => Some(format!(
                "ArrayRead id={array_type_id:?} nolength={nolength} \
                 taken={taken_by_address} item={item_ty:?}"
            )),
            OpKind::BinOp {
                op: bin, result_ty, ..
            } => Some(format!(
                "BinOp {bin} result_ty={result_ty:?} dest={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::ConstInt(n) => Some(format!("ConstInt {n}")),
            OpKind::FieldRead { field, ty, .. } => Some(format!(
                "getfield {}.{} ty={ty:?} dest={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            _ => None,
        })
        .collect();

    let saw_mul = rewritten.graph.blocks.iter().any(|block| {
        let ops = &block.operations;
        ops.windows(2).any(|pair| {
            matches!(&pair[0].kind, OpKind::ConstInt(n) if *n == item_bytes)
                && matches!(
                    &pair[1].kind,
                    OpKind::BinOp { op, result_ty, .. }
                        if op == "int_mul" && *result_ty == ValueType::Int
                )
        })
    });
    assert!(
        saw_mul,
        "getarraysubstruct is int_mul by sizeof(ConstantData)={item_bytes}. \
         dump={rewrite_dump:?}"
    );
    let saw_int_add = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "int_add" && *result_ty == ValueType::Int => op
                .result
                .as_ref()
                .is_some_and(|v| FunctionGraph::concretetype_of(v) == ConcreteType::Signed),
            _ => false,
        })
    });
    assert!(
        saw_int_add,
        "getarraysubstruct is int_add kind 'i'. dump={rewrite_dump:?}"
    );
    let leftover_gc_item = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::ArrayRead { nolength, .. } => !nolength,
            _ => false,
        })
    });
    assert!(
        !leftover_gc_item,
        "no getarrayitem_gc after rewrite. dump={rewrite_dump:?}"
    );
    let leftover_raw_r = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead {
                field, ty, pure, ..
            } if field.owner_declared_gc == Some(false)
                && matches!(ty, ValueType::Ref(_))
                && !*pure
                && !field.taken_by_address =>
            {
                true
            }
            _ => false,
        })
    });
    assert!(
        !leftover_raw_r,
        "no getfield_raw_r leftover. dump={rewrite_dump:?}"
    );
}

fn cc_nested_u8_array_id(id: &str) -> bool {
    let id = id.trim();
    id.starts_with("[[u8;1];") || id.starts_with("[[u8; 1];")
}

/// `interp_termios.tcgetattr` reads `t.cc[vmin][0] as i64`. `cc` is
/// `[[u8; 1]; NCCS]`: the outer item is ContainerType, so the Index is
/// `getarraysubstruct` (`int_mul` sizeof(`[u8;1]`) + `int_add`, kind `'i'`),
/// and the inner `[0]` is `getarrayitem_raw_i`.
#[test]
fn interp_termios_tcgetattr_cc_index_is_getarraysubstruct() {
    let llbc = load_llbc(MODULE_LLBC);
    let graph =
        lower_named(llbc, "interp_termios::tcgetattr").expect("lower interp_termios::tcgetattr");

    let array_dump: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayRead {
                item_ty,
                array_type_id,
                nolength,
                taken_by_address,
                ..
            } => Some(format!(
                "ArrayRead item_ty={item_ty:?} array_type_id={array_type_id:?} \
                 nolength={nolength} taken_by_address={taken_by_address}"
            )),
            _ => None,
        })
        .collect();

    let mut saw_outer = false;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::ArrayRead {
            array_type_id,
            nolength,
            taken_by_address,
            ..
        } = &op.kind
        else {
            continue;
        };
        let Some(id) = array_type_id.as_deref() else {
            continue;
        };
        if !cc_nested_u8_array_id(id) {
            continue;
        }
        saw_outer = true;
        assert!(
            *nolength,
            "cc items region is headerless. dump={array_dump:?}"
        );
        assert!(
            *taken_by_address,
            "`t.cc[vmin][0]` outer Index is the item address. dump={array_dump:?}"
        );
    }
    assert!(
        saw_outer,
        "tcgetattr must ArrayRead [[u8;1];NCCS]. dump={array_dump:?}"
    );

    let config = GraphTransformConfig::default();
    let rewritten = Transformer::new(&config).transform(&graph);
    let rewrite_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::ArrayRead {
                array_type_id,
                nolength,
                taken_by_address,
                item_ty,
                ..
            } => Some(format!(
                "ArrayRead id={array_type_id:?} nolength={nolength} \
                 taken={taken_by_address} item={item_ty:?}"
            )),
            OpKind::BinOp {
                op: bin, result_ty, ..
            } => Some(format!(
                "BinOp {bin} result_ty={result_ty:?} dest={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::ConstInt(n) => Some(format!("ConstInt {n}")),
            _ => None,
        })
        .collect();

    let saw_mul = rewritten.graph.blocks.iter().any(|block| {
        let ops = &block.operations;
        ops.windows(2).any(|pair| {
            matches!(&pair[0].kind, OpKind::ConstInt(1))
                && matches!(
                    &pair[1].kind,
                    OpKind::BinOp { op, result_ty, .. }
                        if op == "int_mul" && *result_ty == ValueType::Int
                )
        })
    });
    assert!(
        saw_mul,
        "getarraysubstruct is int_mul by sizeof([u8;1])=1. dump={rewrite_dump:?}"
    );
    let saw_int_add = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "int_add" && *result_ty == ValueType::Int => op
                .result
                .as_ref()
                .is_some_and(|v| FunctionGraph::concretetype_of(v) == ConcreteType::Signed),
            _ => false,
        })
    });
    assert!(
        saw_int_add,
        "getarraysubstruct is int_add kind 'i'. dump={rewrite_dump:?}"
    );
    let saw_inner = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::ArrayRead {
                    nolength: true,
                    taken_by_address: false,
                    item_ty: ValueType::Unsigned,
                    ..
                }
            )
        })
    });
    assert!(
        saw_inner,
        "inner [0] is getarrayitem_raw_i. dump={rewrite_dump:?}"
    );
    let leftover_raw_r = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::ArrayRead {
                    item_ty: ValueType::Ref(_),
                    taken_by_address: false,
                    ..
                }
            )
        })
    });
    assert!(
        !leftover_raw_r,
        "no getarrayitem_raw_r leftover. dump={rewrite_dump:?}"
    );
    let leftover_gc_item = rewritten.graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::ArrayRead { nolength, .. } => !nolength,
            _ => false,
        })
    });
    assert!(
        !leftover_gc_item,
        "no getarrayitem_gc after rewrite. dump={rewrite_dump:?}"
    );
}

/// `&(*code).source_path` is the inline String header. After the layout
/// row is FLAG_STRUCT 3 words, jtransform emits the getsubstruct address
/// (Identity at offset 0). The consumer is `String` equality.
#[test]
fn box_code_constant_inheriting_unit_source_path_is_substruct_address() {
    let interpreter_llbc = load_llbc(INTERPRETER_LLBC);
    let object_llbc = load_llbc(OBJECT_LLBC);
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[interpreter_llbc, object_llbc],
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
        &["pycode"],
        &[
            "box_code_constant_inheriting_unit",
            "box_code_constant_in_place",
            "code_unit_owner",
        ],
    )
    .expect("lower box_code_constant_inheriting_unit");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "box_code_constant_inheriting_unit")
        .expect("box_code_constant_inheriting_unit graph");

    let field_dump: Vec<String> = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => Some(format!(
                "{}.{} ty={ty:?} declared_gc={:?} taken_by_address={} base={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.owner_declared_gc,
                field.taken_by_address,
                FunctionGraph::concretetype_of(base)
            )),
            _ => None,
        })
        .collect();
    eprintln!("inheriting_unit field dump {field_dump:?}");

    let front_calls: Vec<String> = function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target,
                args,
                result_ty,
            } => {
                let arg_kinds: Vec<_> = args
                    .iter()
                    .map(|arg| match arg {
                        majit_translate::model::LinkArg::Value(v) => {
                            format!("{:?}", FunctionGraph::concretetype_of(v))
                        }
                        majit_translate::model::LinkArg::Const(_) => "const".to_string(),
                    })
                    .collect();
                Some(format!(
                    "call {target} result_ty={result_ty:?} args={arg_kinds:?}"
                ))
            }
            _ => None,
        })
        .collect();
    eprintln!("inheriting_unit front calls {front_calls:?}");

    let mut saw_source_path = false;
    for op in function
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
    {
        let OpKind::FieldRead { field, .. } = &op.kind else {
            continue;
        };
        let owner = field.owner_root.as_deref().unwrap_or("");
        if field.name != "source_path"
            || !(owner == "CodeObject" || owner.ends_with("::CodeObject"))
        {
            continue;
        }
        saw_source_path = true;
        assert!(
            field.taken_by_address,
            "CodeObject.source_path is `&(*code).source_path`. fields={field_dump:?}"
        );
        assert_eq!(
            field.owner_declared_gc,
            Some(false),
            "CodeObject.source_path owner is raw. fields={field_dump:?}"
        );
    }
    assert!(
        saw_source_path,
        "must FieldRead CodeObject.source_path. fields={field_dump:?}"
    );

    let mut callcontrol = CallControl::new();
    callcontrol.set_known_struct_names(program.known_struct_names.clone());
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.raw_alloc_flavor_structs = program.raw_alloc_flavor_structs.clone();
    install_semantic_layouts(&mut callcontrol, &program);

    let word = majit_translate::layout::target_word_size();
    let layout = callcontrol
        .struct_layout_for("bytecode::CodeObject")
        .expect("CodeObject layout");
    let source_path = layout
        .fields
        .iter()
        .find(|row| row.name == "source_path")
        .expect("source_path row");
    eprintln!(
        "inheriting_unit row source_path offset={} size={} flag={:?}",
        source_path.offset, source_path.size, source_path.flag
    );
    assert_eq!(source_path.offset, 0);
    assert_eq!(source_path.size, 3 * word);
    assert_eq!(source_path.flag, majit_ir::descr::ArrayFlag::Struct);

    let config = GraphTransformConfig::default();
    let probe_graph = majit_translate::model::copygraph(function.graph());
    let transform = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Transformer::new(&config)
            .with_callcontrol(&mut callcontrol)
            .transform(&probe_graph)
    }));
    let rewritten = match transform {
        Ok(rewritten) => rewritten,
        Err(payload) => {
            let msg = panic_payload_string(&payload);
            eprintln!("inheriting_unit transform panic {msg}");
            panic!("Task 2 STOP: jtransform panicked: {msg}");
        }
    };

    let rewrite_dump: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead { field, ty, .. } => Some(format!(
                "getfield {}.{} ty={ty:?} taken={} dest_kind={:?}",
                field.owner_root.as_deref().unwrap_or("?"),
                field.name,
                field.taken_by_address,
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::BinOp {
                op: bin, result_ty, ..
            } if bin == "add" => Some(format!(
                "int_add result_ty={result_ty:?} dest_kind={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::UnaryOp { op: un, .. } if un == "cast_ptr_to_int" => Some(format!(
                "cast_ptr_to_int dest_kind={:?}",
                op.result.as_ref().map(FunctionGraph::concretetype_of)
            )),
            OpKind::Call {
                target,
                args,
                result_ty,
            } => {
                let arg_kinds: Vec<_> = args
                    .iter()
                    .map(|arg| match arg {
                        majit_translate::model::LinkArg::Value(v) => {
                            format!("{:?}", FunctionGraph::concretetype_of(v))
                        }
                        majit_translate::model::LinkArg::Const(_) => "const".to_string(),
                    })
                    .collect();
                Some(format!(
                    "call {target} result_ty={result_ty:?} args={arg_kinds:?}"
                ))
            }
            OpKind::CallResidual {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => Some(format!(
                "residual {funcptr:?} arg_classes={} result_kind={result_kind} i={} r={} f={}",
                descriptor.arg_classes,
                args_i.len(),
                args_r.len(),
                args_f.len()
            )),
            OpKind::InlineCall {
                arg_classes,
                result_kind,
                args_i,
                args_r,
                args_f,
                ..
            } => Some(format!(
                "inline arg_classes={arg_classes} result_kind={result_kind} i={} r={} f={}",
                args_i.len(),
                args_r.len(),
                args_f.len()
            )),
            _ => None,
        })
        .collect();
    eprintln!("inheriting_unit rewrite dump {rewrite_dump:?}");
    eprintln!("inheriting_unit rewrite notes {:?}", rewritten.notes);

    let leftover_raw_r: Vec<String> = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::FieldRead {
                field, ty, pure, ..
            } if field.owner_declared_gc == Some(false)
                && matches!(ty, ValueType::Ref(_))
                && !*pure =>
            {
                Some(format!(
                    "{}.{} ty={ty:?} taken={}",
                    field.owner_root.as_deref().unwrap_or("?"),
                    field.name,
                    field.taken_by_address
                ))
            }
            _ => None,
        })
        .collect();
    assert!(
        leftover_raw_r.is_empty(),
        "source_path must not remain getfield_raw_r. leftover={leftover_raw_r:?} \
         rewrite={rewrite_dump:?} fields={field_dump:?}"
    );

    let leftover_source_path = rewritten
        .graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "source_path"
                        && field
                            .owner_root
                            .as_deref()
                            .is_some_and(|o| o == "CodeObject" || o.ends_with("::CodeObject"))
            )
        });
    assert!(
        !leftover_source_path,
        "CodeObject.source_path must become the getsubstruct address. rewrite={rewrite_dump:?}"
    );

    let mut unit_path = None;
    for func in &program.functions {
        let mut graph = func.graph().clone();
        if let Some(rt) = &func.return_type {
            graph = graph.with_return_type(rt.clone());
        }
        let mut module_segs: Vec<&str> = func
            .module_path
            .split("::")
            .filter(|s| !s.is_empty())
            .collect();
        let mut paths = Vec::new();
        if module_segs.is_empty() {
            paths.push(CallPath::from_segments([func.name.as_str()]));
        } else {
            module_segs.push(func.name.as_str());
            paths.push(CallPath::from_segments(module_segs.iter().copied()));
            let mut crate_segs = vec!["pyre_interpreter"];
            crate_segs.extend(module_segs.iter().copied());
            paths.push(CallPath::from_segments(crate_segs));
        }
        for path in paths {
            if func.name == "box_code_constant_inheriting_unit"
                && path
                    .canonical_key()
                    .contains("box_code_constant_inheriting_unit")
            {
                unit_path = Some(path.clone());
            }
            if func.hints.is_empty() {
                callcontrol.register_function_graph(path, graph.clone());
            } else {
                callcontrol.register_function_graph_with_hints(
                    path,
                    graph.clone(),
                    func.hints.clone(),
                );
            }
        }
    }
    let unit_path = unit_path.expect("inheriting_unit CallPath");
    callcontrol.add_candidate_graph(unit_path.clone());

    let mut cw = CodeWriter::new();
    cw.run_two_phase_prepass(&callcontrol);
    let jitcode = std::sync::Arc::new(JitCode::new("box_code_constant_inheriting_unit"));
    let assemble = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cw.transform_graph_to_jitcode(
            function.graph(),
            &unit_path,
            &mut callcontrol,
            &config,
            &jitcode,
            false,
            0,
        );
    }));
    match assemble {
        Ok(()) => eprintln!(
            "inheriting_unit assemble ok code={:?} resulttypes={:?}",
            jitcode.body().code,
            jitcode.body().resulttypes
        ),
        Err(payload) => {
            let msg = panic_payload_string(&payload);
            eprintln!("inheriting_unit assemble panic {msg}");
            panic!("Task 2 STOP: assemble/getcalldescr panicked: {msg}");
        }
    }
}

fn panic_payload_string(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        format!("{payload:?}")
    }
}
