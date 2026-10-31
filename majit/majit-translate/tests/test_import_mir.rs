mod common;

use std::collections::HashSet;

use common::{
    INTERPRETER_LLBC, OBJECT_LLBC, RLIB_LLBC, load_llbc, lower_named, lower_named_with_static_addrs,
};
use majit_charon_reader::Llbc;
use majit_translate::{
    CallPath, ErrorCarrierSpec, FunctionGraph, HostStaticAddrs,
    annotator::classdesc::register_struct_fields_with_layout,
    codewriter::{
        CodeWriter,
        call::CallControl,
        jtransform::{GraphTransformConfig, transform_graph},
    },
    flowspace::argument::Signature,
    front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
    model::{CallTarget, ConcreteType, ExitSwitch, LinkArg, OpKind, VecFieldPart},
    translator::rtyper::{legacy_annotator, legacy_resolve},
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

/// The carrier the production prepass names (`pyre-jit-trace/build/prepass.rs`).
/// Without it, `front::result_exc` leaves every `Result<T, PyError>` `?` as a
/// residual `branch` and these bodies appear to lower.
const ERROR_CARRIER: ErrorCarrierSpec<'static> = ErrorCarrierSpec {
    carrier_path: "pyre_interpreter::error::PyError",
    carrier_class: "",
    carrier_wrappers: &[],
    to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
    from_exc_object: Some(("PyError", "from_exc_object")),
};

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
/// must be the `as char` unichr op, a char constant, or a `FieldRead` of
/// `__pos_0` (the unwrap payload is the `__pos_0` read, already a UniChar),
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
        call_leaves(&graph).iter().any(|leaf| *leaf == "malloc"),
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
