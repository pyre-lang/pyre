//! `PyFrame::fast2locals` driven through the codewriter, against the shipped
//! interpreter LLBC.
//!
//! A graph containing a loop is normally residualized rather than looked
//! inside (`policy.py:50,61-62 contains_loop`), so nothing in the shipped
//! build exercises the front-through-assembler path on a looping graph that
//! reads the frame's locals array.  `fast2locals` is the smallest real graph
//! with that shape, and driving it here rather than waiting for a policy that
//! admits it is what surfaced the element-type defect below.

mod common;

use common::{interpreter_llbc, lower_context_for};
use majit_charon_reader::Llbc;
use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::front::mir::{LowerContext, lower_fun_decl_with_static_addrs};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
use majit_translate::{
    CallPath, ErrorCarrierSpec, GraphTransformConfig, HostStaticAddrs, VirtualizableFieldDescriptor,
};

fn lower_named(llbc: &Llbc, context: &LowerContext<'_>, leaf: &str) -> FunctionGraph {
    let suffix = format!("::{leaf}");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with(&suffix))
        .unwrap_or_else(|| panic!("{leaf} present in the shipped LLBC"));
    // The graph under test returns `Result<(), PyError>`, so the lowering has
    // to be told which `Result` is the fallible return before `result_exc` can
    // turn it into exception edges.  `majit-translate` names no carrier of its
    // own, and the production driver spells this same one in
    // `pyre-jit-trace/build/prepass.rs`.
    let static_addrs = HostStaticAddrs {
        error_carrier: ErrorCarrierSpec {
            carrier_path: "pyre_interpreter::error::PyError",
            carrier_class: "",
            carrier_wrappers: &[],
            to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
            from_exc_object: Some(("PyError", "from_exc_object")),
        },
        ..Default::default()
    };
    lower_fun_decl_with_static_addrs(context, fd, static_addrs)
        .unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"))
}

/// Lower `pyframe::<Impl>::fast2locals` out of the shipped LLBC, or `None`
/// when the artefact is absent so the tests degrade to a skip.
fn lower_fast2locals() -> Option<FunctionGraph> {
    let llbc = interpreter_llbc()?;
    let context = lower_context_for(llbc);
    Some(lower_named(llbc, &context, "fast2locals"))
}

fn call_leafs(graph: &FunctionGraph) -> Vec<String> {
    let mut leafs = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call { target, .. } = &op.kind
                && let Some(segs) = target.path_segments()
                && let Some(leaf) = segs.last()
            {
                leafs.push((*leaf).to_string());
            }
        }
    }
    leafs
}

/// `rlib/debug.py check_not_access_directly` is the identity after its
/// annotator assert has run — upstream's `specialize_call` is
/// `hop.inputarg(hop.args_r[0], arg=0)`. `identity_passthrough_alias` is the
/// port of that half, and this is the graph that carries the one call
/// (`typedef::type`, which fuses `space.type` and `W_Root.getclass`). Left
/// unfolded the marker would sit as a residual call on the type-lookup path.
#[test]
fn the_access_directly_marker_folds_out_of_the_type_lookup() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let context = lower_context_for(llbc);
    let graph = lower_named(llbc, &context, "typedef::type");
    assert!(
        call_leafs(&graph)
            .iter()
            .all(|leaf| leaf != "check_not_access_directly"),
        "the marker must be aliased to its argument, got {:?}",
        call_leafs(&graph)
    );
}

/// `GetSetProperty(PyFrame.fget_getdictscope)`: the wrapper carries the
/// residual force; `rewrite_op_jit_force_virtualizable` deletes it; the
/// method itself matches `pyframe.py fget_getdictscope`.
#[test]
fn f_locals_gateway_force_is_deleted_and_the_method_has_none() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let context = lower_context_for(llbc);
    let gateway = lower_named(
        llbc,
        &context,
        "__majit_wrap_descr_typecheck_fget_getdictscope",
    );
    let method = lower_named(llbc, &context, "fget_getdictscope");
    assert!(
        call_leafs(&gateway)
            .iter()
            .any(|leaf| leaf == "jit_force_virtualizable"),
        "gateway must spell the residual force as jit_force_virtualizable, got {:?}",
        call_leafs(&gateway)
    );
    assert!(
        call_leafs(&method)
            .iter()
            .all(|leaf| leaf != "jit_force_virtualizable"
                && leaf != "force_frame"
                && leaf != "force_frame_before_locals_read"),
        "fget_getdictscope must match pyframe.py (no hand-placed force), got {:?}",
        call_leafs(&method)
    );

    let transformed = majit_translate::transform_graph(&gateway, &config());
    assert!(
        call_leafs(&transformed.graph)
            .iter()
            .all(|leaf| leaf != "jit_force_virtualizable"),
        "rewrite_op_jit_force_virtualizable must delete the call, got {:?}",
        call_leafs(&transformed.graph)
    );
}

/// Every frame getset whose body reaches a REDIRECTED field is an interp2app
/// gateway on the same terms as `f_locals`: the untranslated interpreter keeps
/// the residual force, and the codewriter deletes it before lowering the field
/// access.
///
/// Keeping this as one census is deliberate.  The bug this guards was a
/// structural split where only `f_locals` published a candidate graph; a set
/// of independent happy-path tests would let the next manually-added frame
/// descriptor fall outside the family again.  Its other half is
/// [`the_gateways_outside_the_redirected_set_carry_no_force`], which closes the
/// complement so that neither list can grow by accident.
#[test]
fn every_redirected_frame_getter_carries_a_deletable_force() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let context = lower_context_for(llbc);
    for leaf in [
        "__majit_wrap_descr_typecheck_get_w_globals",
        "__majit_wrap_descr_typecheck_fget_f_lasti",
        "__majit_wrap_descr_typecheck_fget_f_lineno",
        "__majit_wrap_descr_typecheck_fset_f_lineno",
        "__majit_wrap_descr_typecheck_fget_f_trace",
        "__majit_wrap_descr_typecheck_fset_f_trace",
        "__majit_wrap_descr_typecheck_fdel_f_trace",
        "__majit_wrap_descr_typecheck_fget_f_trace_lines",
        "__majit_wrap_descr_typecheck_fset_f_trace_lines",
        "__majit_wrap_descr_typecheck_fget_f_trace_opcodes",
        "__majit_wrap_descr_typecheck_fset_f_trace_opcodes",
    ] {
        let gateway = lower_named(llbc, &context, leaf);
        let before = call_leafs(&gateway);
        assert!(
            before
                .iter()
                .any(|callee| callee == "jit_force_virtualizable"),
            "{leaf} must retain the residual force in the interpreter graph, got {before:?}",
        );
        let transformed = majit_translate::transform_graph(&gateway, &config());
        let after = call_leafs(&transformed.graph);
        assert!(
            after
                .iter()
                .all(|callee| callee != "jit_force_virtualizable"),
            "{leaf}: rewrite_op_jit_force_virtualizable left the marker in jitcode: {after:?}",
        );
    }
}

/// The complement of [`every_redirected_frame_getter_carries_a_deletable_force`]:
/// these four gateways carry no marker at all, and the reason is per gateway.
///
/// `rvirtualizable.py hook_access_field` injects only under
/// `if self.my_redirected_fields.get(cname.value)`, and `f_back`
/// (`f_backref`), `f_builtins` (`w_builtin`) and `get_generator`
/// (`f_generator_wref`) read no field in that set.  `f_code` reads `pycode`,
/// which IS in it, and is here for the second question a HAND-placed marker
/// owes and the injection never has to ask, because it fires at every access in
/// every graph rather than once per gateway: can the trace and the live frame
/// disagree about the field this body reads?  For `pycode` they cannot -- it is
/// written only at frame construction -- so the force reaches nothing but the
/// escape flush.  Each of the four says so at its own definition in
/// `pyframe.rs`.
#[test]
fn the_gateways_outside_the_redirected_set_carry_no_force() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    let context = lower_context_for(llbc);
    for leaf in [
        "__majit_wrap_descr_typecheck_fget_f_code",
        "__majit_wrap_descr_typecheck_fget_f_back",
        "__majit_wrap_descr_typecheck_fget_f_builtins",
        "__majit_wrap_descr_typecheck_get_generator",
    ] {
        let gateway = lower_named(llbc, &context, leaf);
        let before = call_leafs(&gateway);
        assert!(
            before
                .iter()
                .all(|callee| callee != "jit_force_virtualizable"),
            "{leaf} reads no field a trace can disagree about, so it must carry \
             no force: {before:?}",
        );
    }
}

/// The virtualizable configuration the production pipeline passes
/// (`pyre-jit-trace/src/virtualizable_spec.rs PYFRAME_VABLE_ARRAYS`).
fn config() -> GraphTransformConfig {
    GraphTransformConfig {
        vable_fields: ["last_instr", "pycode", "valuestackdepth", "debugdata"]
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                VirtualizableFieldDescriptor::new(name, Some("PyFrame".to_string()), index)
            })
            .collect(),
        vable_arrays: vec![VirtualizableFieldDescriptor::new(
            "locals_cells_stack_w",
            Some("PyFrame".to_string()),
            0,
        )],
        ..Default::default()
    }
}

/// The whole pipeline — jtransform, regalloc, flatten, assemble — runs to
/// completion on `fast2locals`.
///
/// `front::iter_next` typed the `range` iterator's element as a GC
/// reference, which put the loop counter in the ref register bank and made
/// every `array[i]` in the body a `getarrayitem_gc` with a ref index.
/// `assembler.rs` asserts that index kind, so it aborted the build outright
/// — asserting that this call returns at all is the assertion.
///
/// Production declines `fast2locals` in the rtyper's two-phase prepass and
/// types it with the legacy walker, so this test still drives a standalone
/// `CallControl` rather than the full portal set. `rewrite_op_jit_force_virtualizable`
/// still asserts a jitdriver vinfo once a handle is attached
/// (`jtransform.py`), so the eval-loop virtualizable is registered from
/// `config()` the same way `codewriter_vinfo_from_config` does in the
/// prepass.
#[test]
fn fast2locals_assembles() {
    let Some(graph) = lower_fast2locals() else {
        return;
    };
    // `co_flags()` reads the flags word. A `CodeFlags::bits` call must name a
    // translated compiler-core body rather than remain an unresolvable path
    // (`jtransform.py Transformer.handle_residual_call`).
    let llbc = interpreter_llbc().expect("interpreter LLBC");
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        match target {
            CallTarget::FunctionPath {
                segments,
                fun_decl_id,
                ..
            } if segments.last().map(String::as_str) == Some("bits") => {
                let id = fun_decl_id.unwrap_or_else(|| {
                    panic!("CodeFlags::bits FunctionPath has no fun_decl_id, segs={segments:?}")
                });
                let fd = llbc
                    .fn_by_id(id)
                    .unwrap_or_else(|| panic!("CodeFlags::bits fun_decl_id={id} missing"));
                assert!(
                    fd.has_unstructured_body(),
                    "CodeFlags::bits FunDecl {} is Opaque",
                    fd.item_meta.name_path()
                );
            }
            CallTarget::Method { name, .. } if name == "bits" => {
                panic!("CodeFlags::bits lowered as Method, got {target:?}");
            }
            _ => {}
        }
    }
    let path = CallPath::from_segments(["pyre_interpreter", "pyframe", "fast2locals"]);
    let cfg = config();
    let mut callcontrol = CallControl::new();
    let llbc = interpreter_llbc().expect("fast2locals_assembles already lowered from this LLBC");
    callcontrol.set_struct_fields(lower_context_for(llbc).struct_field_registry());
    callcontrol.set_exc_pytype_rows(vec![(
        "interp_exceptions::EXC_STOPITERATION_TYPE".to_string(),
        1,
    )]);
    callcontrol.register_function_graph(path.clone(), graph.clone());
    callcontrol.setup_jitdriver(
        CallPath::from_segments(["pyre_interpreter", "eval", "dispatch_bytecode"]),
        vec![],
        vec!["frame".into()],
        vec![],
        vec![],
        false,
        vec!["frame".into()],
        vec!["PyFrame".into()],
        CallPath::from_segments(["pyre_interpreter", "eval", "dispatch_bytecode"]),
    );
    callcontrol.make_virtualizable_infos(|_, vtype| {
        majit_translate::codewriter::call::codewriter_vinfo_from_config(vtype, &cfg)
    });
    let mut codewriter = CodeWriter::new();
    let jitcode = std::sync::Arc::new(JitCode::new("fast2locals"));
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &cfg,
        &jitcode,
        false,
        0,
    );
}

/// A `for i in a..b` loop yields an integer, not a GC reference.
///
/// `front::range_iter` reroutes the exclusive int `Range` onto the `range()`
/// builtin plus the `iter` bridge so the receiver selects upstream's
/// `RangeIteratorRepr`, whose `rtype_next` (`rrange.py ll_rangenext_*`)
/// yields a `Signed`.  The `[__iter_next]` marker `front::iter_next` folds
/// the `next` diamond into has to agree: the legacy type walker reads
/// `result_ty` straight off the op, and it is the tier `fast2locals` lands
/// in.
#[test]
fn a_range_loops_next_yields_an_int_element() {
    let Some(graph) = lower_fast2locals() else {
        return;
    };

    // The `range()` results, so an `[__iter_next]` can be attributed to a
    // range rather than to a slice/Vec iterator.
    let mut range_results = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
                && segments.len() == 1
                && segments[0] == "__majit_range"
                && let Some(result) = &op.result
            {
                range_results.push(result.clone());
            }
        }
    }
    assert!(
        !range_results.is_empty(),
        "`fast2locals` has two `for i in 0..n` loops; neither reached \
         `front::range_iter`, so this test no longer exercises anything"
    );

    // An iterator built over one of those ranges, threaded to its `next`.
    let iterators: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().is_some_and(|s| s == "iter")
                && args.first().is_some_and(|a| range_results.contains(a)) =>
            {
                op.result.clone()
            }
            _ => None,
        })
        .collect();
    assert!(
        !iterators.is_empty(),
        "no `iter` op over a `range()` result — the reroute changed shape"
    );

    let mut checked = 0;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            // `front::iter_next`'s marker spelling; the module is
            // crate-private, so match the reserved segment directly rather
            // than widening its visibility for a test.
            if segments.len() != 1 || segments[0] != "__iter_next" {
                continue;
            }
            // The iterator reaches `next` through the loop header's phi, so
            // match on the value that entered it rather than on the operand.
            if !args
                .first()
                .is_some_and(|a| iterators.contains(a) || reaches(&graph, a, &iterators))
            {
                continue;
            }
            checked += 1;
            assert_eq!(
                result_ty,
                &ValueType::Int,
                "the `range` iterator's `next` yields a GC reference, so the \
                 loop counter lands in the ref register bank",
            );
        }
    }
    assert!(
        checked > 0,
        "no `[__iter_next]` was attributed to a range iterator"
    );
}

/// Whether `var` is one of `sources`, following block inputargs back through
/// every predecessor's link argument in the matching slot.  The loop-carried
/// iterator arrives at `next` as a phi, not as the `iter` op's result.
fn reaches(
    graph: &FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
    sources: &[majit_translate::flowspace::model::Variable],
) -> bool {
    let mut visited = Vec::new();
    let mut stack = vec![var.clone()];
    while let Some(v) = stack.pop() {
        if visited.contains(&v) {
            continue;
        }
        if sources.contains(&v) {
            return true;
        }
        visited.push(v.clone());
        for b in &graph.blocks {
            let Some(slot) = b.inputargs.iter().position(|iv| iv == &v) else {
                continue;
            };
            for pb in &graph.blocks {
                for link in &pb.exits {
                    if link.target == b.id
                        && let Some(majit_translate::model::LinkArg::Value(src)) =
                            link.args.get(slot)
                    {
                        stack.push(src.clone());
                    }
                }
            }
        }
    }
    false
}
