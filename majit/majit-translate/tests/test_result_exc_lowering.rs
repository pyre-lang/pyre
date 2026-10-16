//! Result-of-PyError → exception-link lowering: production-LLBC
//! regression tests for `front::result_exc`.

mod common;

use common::{
    INTERPRETER_LLBC, MODULE_LLBC, load_llbc, lower_named_with_static_addrs, with_shared_lowering,
};
use majit_charon_reader::Llbc;
use majit_ir::descr::{ExtraEffect, OopSpecIndex};
use majit_ir::value::Type;
use majit_translate::annotator::bookkeeper::Bookkeeper;
use majit_translate::codewriter::call::{AnalysisCache, CallControl};
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::model::{CallTarget, ExitCase, ExitSwitch, LinkArg, OpKind, SpaceOperation};
use majit_translate::{CallPath, ErrorCarrierSpec, GraphTransformConfig, HostStaticAddrs};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Load `pyre-interpreter.ullbc` once and share it across every test.
///
/// The corpus is ~224MB on disk and its parsed `serde_json` form is
/// several GB resident.  Loading it per test means the four tests here
/// — run concurrently by the default test harness — each hold a full
/// parse resident at the same time, several times the runner's RAM on
/// the 16GB CI hosts; the resulting OOM/swap-thrash gets the job killed
/// (Linux), where a developer machine with more headroom only runs
/// slowly.  `Llbc` is read-only after `load`, so a single shared parse
/// behind `common::load_llbc` is sufficient, and `lower_function` only
/// borrows it.
/// The carrier the pass under test lowers, spelled the way the production
/// driver spells it (`pyre-jit-trace/build/prepass.rs`).  `majit-translate`
/// names no carrier of its own, so a test that expects `Result<T, PyError>`
/// to become exception links has to declare it.
const ERROR_CARRIER: ErrorCarrierSpec<'static> = ErrorCarrierSpec {
    carrier_path: "pyre_interpreter::error::PyError",
    carrier_class: "",
    carrier_wrappers: &[],
    to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
    from_exc_object: Some(("PyError", "from_exc_object")),
};

fn lower_function(
    llbc: &'static Llbc,
    function_name: &str,
) -> Result<majit_translate::model::FunctionGraph, majit_translate::front::mir::LowerError> {
    lower_named_with_static_addrs(
        llbc,
        function_name,
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
    )
}

/// [`lower_function`], then the codewriter's conversion of the carrier's
/// exception edges into the runtime exception value
/// (`codewriter::error_carrier_edges`): the raise paths the JitCode holds.
fn lower_function_to_runtime_edges(
    llbc: &'static Llbc,
    function_name: &str,
) -> Result<majit_translate::model::FunctionGraph, majit_translate::front::mir::LowerError> {
    let mut graph = lower_function(llbc, function_name)?;
    let hints = majit_translate::front::llbc_hints::harvest_exception_class_hints_from_llbcs(
        std::slice::from_ref(llbc),
    );
    majit_translate::codewriter::error_carrier_edges::lower_error_carrier_edges(
        &mut graph,
        &majit_translate::OwnedErrorCarrierSpec::own(ERROR_CARRIER),
        &|name| majit_translate::front::llbc_hints::exception_class_name_belongs(&hints, name),
    );
    Ok(graph)
}

fn interp() -> &'static Llbc {
    load_llbc(INTERPRETER_LLBC)
}

fn optional_module() -> &'static Llbc {
    load_llbc(MODULE_LLBC)
}

fn is_root_scope_close(op: &OpKind) -> bool {
    matches!(op,
        OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
            if segments.last().map(String::as_str) == Some("root_scope_close")
                || (segments.last().map(String::as_str) == Some("drop_in_place")
                    && segments.iter().any(|s| s == "RootScope")))
}

/// `PinnedResult::reload` reads `roots` / `base` off a raw stack struct and
/// matches `&mut PyResult`. `fielddescrof` now panics on a fabricated slot
/// (`descr.py get_field_descr`), so this graph must assemble with real
/// layout rows rather than `getfield_raw_r`.
#[test]
fn pinned_result_reload_assembles() {
    let name = "pyre_interpreter::executioncontext::<Impl>::reload";
    let graph = lower_function(interp(), name).expect("lower reload");
    let path = CallPath::from_segments(["pyre_interpreter", "executioncontext", "reload"]);
    let mut callcontrol = CallControl::new();
    callcontrol.set_struct_fields(with_shared_lowering(interp(), |ctx| {
        ctx.struct_field_registry()
    }));
    callcontrol.register_function_graph(path.clone(), graph.clone());
    let mut codewriter = CodeWriter::new();
    let jitcode = std::sync::Arc::new(JitCode::new("reload"));
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &GraphTransformConfig::default(),
        &jitcode,
        false,
        0,
    );
}

#[test]
fn trace_unit_result_declares_void_return() {
    // `call_result.is_err()` keeps a Result<(),PyError> disc-prep hop
    // (collapse of that is_err switch would drop the later `?`).
    // The callee rule still stamps FUNC.RESULT = void.
    let graph = lower_function(
        interp(),
        "pyre_interpreter::executioncontext::<Impl>::_trace",
    )
    .expect("lower _trace");
    {
        eprintln!("TRACE nblocks={}", graph.blocks.len());
        let mut dests: HashMap<majit_translate::flowspace::model::Variable, Vec<String>> =
            HashMap::new();
        let mut recast_roots: HashSet<String> = HashSet::new();
        let mut field_owners: HashSet<String> = HashSet::new();
        for (bi, block) in graph.blocks.iter().enumerate() {
            if let Some(op) = block.operations.get(1) {
                let dest = op
                    .result
                    .as_ref()
                    .map(|v| v.name())
                    .unwrap_or_else(|| "-".into());
                let kind = match &op.kind {
                    OpKind::FieldRead { field, .. } => {
                        format!(
                            "FieldRead {}::{}",
                            field.owner_root.as_deref().unwrap_or("?"),
                            field.name
                        )
                    }
                    OpKind::Call { target, .. } => format!("Call {target}"),
                    OpKind::Input {
                        name,
                        ty,
                        class_root,
                    } => {
                        format!("Input {name} ty={ty:?} class={class_root:?}")
                    }
                    other => format!("{other:?}"),
                };
                let kind = if kind.len() > 180 {
                    format!("{}…", &kind[..180])
                } else {
                    kind
                };
                eprintln!("TRACE bb{bi} op1 dest={dest} {kind}");
            }
            for (oi, op) in block.operations.iter().enumerate() {
                if let Some(dest) = &op.result {
                    dests
                        .entry(dest.clone())
                        .or_default()
                        .push(format!("bb{bi} op{oi}"));
                }
                if let OpKind::FieldRead { field, .. } = &op.kind {
                    if let Some(owner) = field.owner_root.as_deref() {
                        field_owners.insert(owner.to_string());
                    }
                    if field.name == "__discriminant" {
                        eprintln!(
                            "TRACE disc bb{bi} op{oi} dest={} owner={}",
                            op.result
                                .as_ref()
                                .map(|v| v.name())
                                .unwrap_or_else(|| "-".into()),
                            field.owner_root.as_deref().unwrap_or("?")
                        );
                        if oi > 0 {
                            let prev = &block.operations[oi - 1];
                            eprintln!(
                                "TRACE disc-prev bb{bi} op{} dest={} {prev:?}",
                                oi - 1,
                                prev.result
                                    .as_ref()
                                    .map(|v| v.name())
                                    .unwrap_or_else(|| "-".into())
                            );
                        }
                    }
                }
                if let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                    && segments.last().map(String::as_str) == Some("__cast_instance_intrinsic")
                {
                    for arg in args {
                        if let LinkArg::Const(c) = arg
                            && let Some(s) = c.value.as_pystr()
                        {
                            recast_roots.insert(s.to_string());
                        }
                    }
                    eprintln!(
                        "TRACE cast bb{bi} op{oi} dest={} args={args:?}",
                        op.result
                            .as_ref()
                            .map(|v| v.name())
                            .unwrap_or_else(|| "-".into())
                    );
                }
            }
        }
        for (dest, sites) in &dests {
            if sites.len() > 1 {
                eprintln!("TRACE dest-reuse {} {sites:?}", dest.name());
            }
        }
        let fields = with_shared_lowering(interp(), |ctx| ctx.struct_field_registry());
        let bk = Rc::new(Bookkeeper::new());
        bk.set_struct_fields(Rc::new(fields));
        bk.set_exception_carrier("error::PyErrorObject");
        bk.alias_exception_carrier_handle("pyre_interpreter::error::PyError");
        let mut roots: Vec<String> = recast_roots.into_iter().chain(field_owners).collect();
        roots.sort();
        roots.dedup();
        for root in &roots {
            let host = bk.intern_class_by_qualname(root);
            let projected = bk.project_struct_field_type(root);
            let transparent = bk.transparent_scalar_annotation(root);
            let classdef = bk
                .getuniqueclassdef_for_struct_root(root)
                .ok()
                .map(|cd| cd.borrow().name.clone());
            let proj = format!("{projected:?}");
            let proj = if proj.len() > 120 {
                format!("{}…", &proj[..120])
            } else {
                proj
            };
            eprintln!(
                "TRACE intern {root:?} host={} classdef={classdef:?} transparent={} proj={proj}",
                host.qualname(),
                transparent
                    .as_ref()
                    .map(|s| format!("{s:?}"))
                    .unwrap_or_else(|| "-".into())
            );
        }
    }
    assert_eq!(
        graph.return_type.as_deref(),
        Some("()"),
        "_trace: Result<(), PyError> callee must declare FUNC.RESULT = void"
    );
    let n_except = graph
        .blocks
        .iter()
        .flat_map(|b| &b.exits)
        .filter(|l| l.target == graph.exceptblock)
        .count();
    assert!(n_except > 0, "_trace must raise through exceptblock");
    for pred in graph.predecessors(graph.exceptblock) {
        for link in &graph.blocks[pred.0].exits {
            if link.target != graph.exceptblock {
                continue;
            }
            let Some(LinkArg::Value(evalue)) = link.args.get(1) else {
                continue;
            };
            let kind = graph.blocks.iter().find_map(|block| {
                block
                    .operations
                    .iter()
                    .find_map(|op| (op.result.as_ref() == Some(evalue)).then_some(&op.kind))
            });
            match kind {
                Some(OpKind::FieldRead { field, .. }) => {
                    let owner = field.owner_root.as_deref().unwrap_or("");
                    assert!(
                        !owner.ends_with("::Break"),
                        "_trace exceptblock pred {} raises Break.__pos_0 ({owner})",
                        pred.0
                    );
                }
                Some(OpKind::Call {
                    target:
                        CallTarget::SyntheticTransparentCtor {
                            name, owner_path, ..
                        },
                    ..
                }) => {
                    let owner = owner_path.join("::");
                    assert!(
                        !(name == "Err" && owner.contains("Result")),
                        "_trace exceptblock pred {} raises Result ADT {owner}::{name}",
                        pred.0
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn unit_result_callee_declares_void_return() {
    // A `Result<(), PyError>` scoped callee returns void after the
    // exception-link lowering, so `front::mir` stamps `return_type =
    // "()"` (`FUNC.RESULT = void`); the codewriter then collapses the
    // returnblock to a genuine void return post-annotation.  Without the
    // stamp the call descriptor would read `r` for the `Ref`-typed unit
    // `()` shell.  Covered: the callee-rule `Ok(())` (`store_local_value`)
    // and the tail-forward `f(...)?` (`store_fast` / `store_fast_store_fast`).
    for name in [
        "pyre_interpreter::eval::<Impl>::store_local_value",
        "pyre_interpreter::eval::<Impl>::send_value",
        "pyre_interpreter::pyopcode::OpcodeStepExecutor::store_fast",
        "pyre_interpreter::pyopcode::OpcodeStepExecutor::store_fast_store_fast",
        "pyre_interpreter::executioncontext::<Impl>::_trace",
    ] {
        let g = lower_function(interp(), name).expect("lower");
        assert_eq!(
            g.return_type.as_deref(),
            Some("()"),
            "{name}: Result<(), PyError> callee must declare FUNC.RESULT = void"
        );
    }

    // A non-unit `Result<T, PyError>` callee is not void-widened:
    // `pop_value` returns `Result<PyObjectRef, PyError>`, so it carries
    // no void stamp and keeps its single ref return variable.
    let pv = lower_function(interp(), "pyre_interpreter::eval::<Impl>::pop_value")
        .expect("lower pop_value");
    assert_ne!(
        pv.return_type.as_deref(),
        Some("()"),
        "pop_value returns a value, not void"
    );
    assert_eq!(
        pv.block(pv.returnblock).inputargs.len(),
        1,
        "pop_value returns a value; its returnblock keeps the return var"
    );
}

/// A wrapper whose every return is `return f(...)` — no `Ok`/`Err` shell of
/// its own — must still retype those calls to the payload.
///
/// `is_true` is that shape: `if …  { return is_true_slot(obj); }
/// is_true_lookup(obj)`.  Both callees are scoped `Result<bool, PyError>`
/// graphs the exception-link lowering transforms into plain `bool` returns,
/// so the value each call hands back is the payload — but `front::mir` types
/// every aggregate `Ref`, and a tail forward has no `__pos_0` read for the
/// diamond arm's `collapse_pos0_read` to narrow through.  Left `Ref`, the
/// disagreement lands on both sides of this graph: the calls emit
/// `inline_call_r_r` against callees that `int_return`, and the returnblock
/// inherits the `Ref` so `graph_result_kind` reports `r` to every `?`-site
/// caller whose own diamond narrowed to `bool`.
///
/// Assert the call ops themselves, not the return kind: the returnblock's
/// `ConcreteType` is coloured later by the rtyper, so this pass's output is
/// the declared `result_ty` on the two `Call`s.
#[test]
fn a_tail_forwarding_wrapper_retypes_its_calls_to_the_payload() {
    let graph =
        lower_function(interp(), "pyre_interpreter::baseobjspace::is_true").expect("lower is_true");
    let mut forwarded = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            let CallTarget::FunctionPath { segments, .. } = target else {
                continue;
            };
            let Some(leaf) = segments.last() else {
                continue;
            };
            if leaf != "is_true_slot" && leaf != "is_true_lookup" {
                continue;
            }
            forwarded += 1;
            assert_eq!(
                *result_ty,
                majit_translate::model::ValueType::Bool,
                "{leaf}: a tail-forwarded scoped callee hands back its `bool` \
                 payload, so the call must declare it -- not the `Result` shell \
                 (`Ref`, which is what this reads before the retype)"
            );
        }
    }
    assert_eq!(
        forwarded, 2,
        "is_true tail-forwards both is_true_slot and is_true_lookup"
    );
}

/// `contains` returns `Result<bool, PyError>`. A literal `Err` that shares
/// the root-scope continue with tail-forwarded `contains_slot` / `is_true`
/// must raise; projecting `Err.__pos_0` into that phi made the CFG return
/// kind `r` against `FUNC.RESULT=i` (`call.py` `get_jitcode_calldescr`).
#[test]
fn contains_err_does_not_join_the_bool_return() {
    use majit_translate::model::{ConcreteType, FunctionGraph, LinkArg, ValueType};
    let graph = lower_function(interp(), "pyre_interpreter::baseobjspace::contains")
        .unwrap_or_else(|e| panic!("lower contains: {e}"));
    let mut bool_returns = 0usize;
    for block in &graph.blocks {
        for link in &block.exits {
            if link.target != graph.returnblock {
                continue;
            }
            assert_eq!(link.args.len(), 1, "bool return has one arg");
            let LinkArg::Value(var) = &link.args[0] else {
                panic!("return arg is a value");
            };
            assert_ne!(
                FunctionGraph::concretetype_of(var),
                ConcreteType::GcRef,
                "CFG return kind must be the bool payload (`i`), not a GC ref"
            );
            let producer = graph.blocks.iter().find_map(|src| {
                src.operations
                    .iter()
                    .find(|op| op.result.as_ref() == Some(var))
            });
            if let Some(OpKind::FieldRead { field, .. }) = producer.map(|op| &op.kind) {
                let owner = field.owner_root.as_deref().unwrap_or("");
                assert!(
                    !(field.name == "__pos_0" && owner.ends_with("::Err")),
                    "Err payload reaches returnblock via {owner}"
                );
            }
            if let Some(OpKind::Call { result_ty, .. }) = producer.map(|op| &op.kind) {
                assert_eq!(result_ty, &ValueType::Bool, "tail-forwarded call is bool");
                bool_returns += 1;
            }
        }
    }
    assert!(
        bool_returns >= 1,
        "contains must return the bool payload of a scoped callee"
    );
}

/// The catch-and-rewrap fallback rebuilds the shell from the call's result,
/// which means that result IS the payload — so the call has to be retyped
/// like the diamond and tail-forward arms do.
///
/// `int_w` is `Result<i64, PyError>`; its consumers here match on the shell
/// by hand rather than through `?`, so the site takes `catch_and_rewrap`.
/// Left `Ref`, the call is emitted as `inline_call_*_r` against a callee that
/// `int_return`s (the returned value then has no destination in the int
/// bank), and the `Ok` shell the rewrite builds is handed a register the
/// caller never wrote.
#[test]
fn a_rewrapped_call_site_retypes_its_call_to_the_payload() {
    for name in [
        "pyre_interpreter::baseobjspace::getindex_w_index",
        "pyre_interpreter::baseobjspace::index_int_w_preserve_negative",
    ] {
        let graph = lower_function(interp(), name).expect("lower");
        let mut seen = 0usize;
        for block in &graph.blocks {
            for op in &block.operations {
                let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if segments.last().map(String::as_str) != Some("int_w") {
                    continue;
                }
                seen += 1;
                assert_eq!(
                    *result_ty,
                    majit_translate::model::ValueType::Int,
                    "{name}: the rewrap rebuilds `Ok(r)` from the call result, so \
                     the call declares int_w's `i64` payload -- not the `Result` \
                     shell (`Ref`, which is what this reads before the retype)"
                );
            }
        }
        assert!(seen >= 1, "{name} calls int_w");
    }
}

/// `getindex_w` is `try: int_w(...) except OperationError`, spelled as a
/// hand-written `match` on `with_roots!(w_index => int_w(w_index))`. The
/// restore hops sit between the call and the discriminant; collapsing
/// them must still drop the rebuilt `Result` shell so the rtyper sees
/// the unwrapped `int` / caught `PyError`.
#[test]
fn getindex_w_match_does_not_rebuild_a_result_shell() {
    let graph = lower_function(interp(), "pyre_interpreter::baseobjspace::getindex_w")
        .unwrap_or_else(|e| panic!("lower: {e}"));
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    let owner = field.owner_root.as_deref().unwrap_or("");
                    assert!(
                        !owner.contains("Result"),
                        "Result discriminant still read: {owner}"
                    );
                }
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } => {
                    let owner = owner_path.join("::");
                    assert!(
                        !owner.contains("Result"),
                        "Result shell ctor still built: {owner}"
                    );
                }
                _ => {}
            }
        }
    }
}

/// `int_w`'s hand-written `match` is `getindex_w`'s
/// `try/except OperationError`: the normal edge returns the `int`, and the
/// handler reads `PyError.kind`. The rebuilt `Result` shell must not survive
/// into the rtyper — its `__discriminant` read is the unbound arg that
/// phase B skips on `_check_len_result`.
#[test]
fn check_len_result_match_does_not_rebuild_a_result_shell() {
    let graph = lower_function(
        interp(),
        "pyre_interpreter::baseobjspace::_check_len_result",
    )
    .expect("lower");
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    let owner = field.owner_root.as_deref().unwrap_or("");
                    assert!(
                        !owner.contains("Result"),
                        "Result discriminant still read: {owner}"
                    );
                }
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } => {
                    let owner = owner_path.join("::");
                    assert!(
                        !owner.contains("Result"),
                        "Result shell ctor still built: {owner}"
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn payload_less_intermediate_result_does_not_decline_the_callee() {
    // `getitem_str` matches on `i64::try_from(&rbigint)`, whose `Err` wraps a
    // zero-sized error and so writes no `__pos_0`.  That shell is consumed in
    // the graph and never reaches `returnblock`; the callee's own
    // `Result<_, PyError>` returns must still lower.
    let graph = lower_function(interp(), "pyre_interpreter::baseobjspace::getitem_str")
        .expect("lower getitem_str");
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                ..
            } = &op.kind
            {
                let owner = owner_path.join("::");
                assert!(
                    !owner.contains("PyError"),
                    "PyError Result shell ctor still built: {owner}"
                );
            }
        }
    }
    // Its literal-message `IndexError` must fuse: a `PyError` aggregate left
    // in the graph hands the materialiser a `STR` word read as a `Wtf8Buf`.
    let (fused, _materialise, _ctors) =
        raise_path_calls("pyre_interpreter::baseobjspace::getitem_str");
    assert!(fused > 0, "getitem_str's literal IndexError must fuse");
}

#[test]
fn list_append_underflow_lowers_to_raise_links() {
    let llbc = interp();
    // `opcode_list_append`'s `depth == 0` arm returns
    // `Err(stack_underflow_error(..))` directly.
    let graph = lower_function_to_runtime_edges(llbc, "opcode_list_append")
        .expect("lower opcode_list_append");
    let mut result_ctors = 0usize;
    let mut to_exc_object_calls = 0usize;
    let mut except_links = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } if owner_path.last().map(String::as_str) == Some("Result") => {
                    result_ctors += 1;
                }
                // The raise site reaches the materialisation through the
                // published free function, not the method: its body must stay
                // opaque to the codewriter so the GC-root, exception-object
                // and WTF-8 machinery under it does not land in this graph.
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("pyerror_to_exc_object") => {
                    to_exc_object_calls += 1
                }
                _ => {}
            }
        }
        for link in &b.exits {
            if link.target == graph.exceptblock {
                except_links += 1;
            }
        }
    }
    assert_eq!(result_ctors, 0, "Result shells must be gone");
    assert_eq!(
        to_exc_object_calls, 1,
        "Err arm materialises the exception object"
    );
    assert!(except_links >= 1, "Err arm raises towards exceptblock");
    eprintln!(
        "opcode_list_append: to_exc_object={to_exc_object_calls} except_links={except_links}"
    );
}

#[test]
fn pop_value_caller_gets_lastexception_exits() {
    let llbc = interp();
    // The SFSF chain's free-fn body pops twice via `?`
    // (pyopcode.rs `opcode_store_fast_store_fast`).
    let graph = lower_function(llbc, "opcode_store_fast_store_fast").expect("lower caller");
    eprintln!("caller graph = {}", graph.name);
    let lastexc_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    let branch_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "branch")
        })
        .count();
    eprintln!("caller: lastexc_blocks={lastexc_blocks} branch_calls={branch_calls}");
    assert!(
        lastexc_blocks >= 1,
        "pop_value call sites get LastException exits"
    );
}

/// Count Result-shell ctors (`SyntheticTransparentCtor` with owner
/// `core::result::Result`) in a lowered graph.
fn count_result_ctors(graph: &majit_translate::model::FunctionGraph) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } if owner_path.last().is_some_and(|leaf| {
                    majit_ir::descr::strip_instantiation_suffix(majit_ir::descr::class_root_leaf(
                        leaf,
                    )) == "Result"
                })
            )
        })
        .count()
}

#[test]
fn count_result_ctors_counts_suffixed_shells() {
    use majit_translate::model::{FunctionGraph, ValueType};
    let mut graph = FunctionGraph::new("suffixed_result_shell");
    let entry = graph.startblock;
    graph
        .push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor_with_owner(
                    vec![
                        "core".into(),
                        "result".into(),
                        "Result<Tuple,PyError>".into(),
                    ],
                    "Ok",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .expect("ctor");
    assert_eq!(
        count_result_ctors(&graph),
        1,
        "Result<Tuple,PyError> is still a Result shell"
    );
}

#[test]
fn execute_wrapper_family_lowers_to_raise_links() {
    let llbc = interp();
    // The arm wrapper's `Ok(StepResult::Continue)` shell must be gone
    // and the `?` on the scoped `store_fast_store_fast` method must be
    // a LastException diamond.
    let graph = lower_function(
        llbc,
        "pyre_interpreter::pyopcode::execute_store_fast_store_fast",
    )
    .expect("lower wrapper");
    assert_eq!(
        count_result_ctors(&graph),
        0,
        "wrapper Result shells must be gone"
    );
    let lastexc_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    assert!(lastexc_blocks >= 1, "wrapper `?` gets LastException exits");
}

/// Facet A firing guard — the jd1 drain-loop `match next()` fusion.
///
/// `unpackiterable_portal`'s StopIteration drain loop is a
/// hand-written `match next() { Ok(w) => append, Err(e) => { let (stop, e)
/// = e.matches_stop_iteration_keep(); if stop { break } return Err(e) } }`.
/// Lowered naively it materialises a `Result` shell and leaves the PyError
/// predicate on its Err arm behind a discriminant switch. `try_fuse_drain_match`
/// (`front::result_exc`) replaces that shell with a `LastException`
/// exception edge catching the carrier (`except OperationError as e`) whose
/// handler runs the same predicate on the caught carrier.
///
/// The fusion is FAIL-SAFE: on any shape it does not recognise it silently
/// falls back to `catch_and_rewrap`, leaving the source predicate on the
/// shell's Err arm. That silent decline is invisible to the default
/// (non-jd1) drain path yet reintroduces the Result shell the jd1 walk cannot
/// consume. This lowers the real drain and asserts every predicate call sits
/// in a carrier handler reading the caught value, so a decline fails loud.
#[test]
fn unpackiterable_drain_match_fuses_to_kind_test() {
    let llbc = interp();
    let graph = lower_function(
        llbc,
        "pyre_interpreter::baseobjspace::unpackiterable_portal",
    )
    .expect("lower unpackiterable_portal");
    assert!(
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method {
                        name,
                        receiver_root: Some(root),
                        ..
                    },
                    ..
                } if name == "jit_merge_point" && root == "UnpackIterableJitDriver"
            )
        }),
        "jit_merge_point must stay a Method on the driver receiver"
    );

    // (block, caught value) for every `except OperationError as e` handler.
    let handlers: Vec<(usize, majit_translate::flowspace::model::Variable)> = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .filter(|link| link.exitcase == Some(ExitCase::ErrorCarrier))
        .filter_map(|link| {
            let caught = link.last_exc_value.as_ref()?.as_variable()?;
            let pos = link
                .args
                .iter()
                .position(|arg| arg.as_variable() == Some(caught))?;
            Some((
                link.target.0,
                graph.blocks[link.target.0].inputargs[pos].clone(),
            ))
        })
        .collect();
    let is_predicate = |op: &majit_translate::model::SpaceOperation| {
        let OpKind::Call { target, .. } = &op.kind else {
            return false;
        };
        let leaf = match target {
            CallTarget::Method { name, .. } => Some(name.as_str()),
            CallTarget::FunctionPath { segments, .. } => segments.last().map(String::as_str),
            _ => None,
        };
        matches!(
            leaf,
            Some("matches_stop_iteration") | Some("matches_stop_iteration_keep")
        )
    };
    // The handler runs the keep (or bool) predicate on the caught carrier
    // and may recast that carrier first. The predicate still reads that
    // carrier, not a second error value.
    let reads_caught = |block: usize, caught: &majit_translate::flowspace::model::Variable| {
        let mut images = vec![caught.clone()];
        for op in &graph.blocks[block].operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let (Some(result), Some(arg)) =
                (&op.result, args.first().and_then(|a| a.as_variable()))
            else {
                continue;
            };
            if !images.iter().any(|image| image == arg) {
                continue;
            }
            let leaf = match target {
                CallTarget::FunctionPath { segments, .. } => segments.last().map(String::as_str),
                _ => None,
            };
            if leaf == Some("__cast_instance_intrinsic") {
                images.push(result.clone());
            }
        }
        graph.blocks[block].operations.iter().any(|op| {
            is_predicate(op)
                && matches!(&op.kind, OpKind::Call { args, .. }
                    if args.first()
                        .and_then(|a| a.as_variable())
                        .is_some_and(|arg| images.iter().any(|image| image == arg)))
        })
    };
    let fused_predicates = handlers
        .iter()
        .filter(|(block, caught)| reads_caught(*block, caught))
        .count();
    assert!(
        fused_predicates >= 1,
        "drain fusion must run the StopIteration predicate on the caught carrier \
         (0 = recognizer silently declined to catch_and_rewrap → the \
         Result shell and source predicate remain on the jd1 walk)"
    );
    // Elimination signal: a predicate outside a carrier handler survives
    // only on the decline path.
    let stray_predicates = graph
        .blocks
        .iter()
        .enumerate()
        .filter(|(bi, _)| !handlers.iter().any(|(h, _)| h == bi))
        .flat_map(|(_, b)| b.operations.iter())
        .filter(|op| is_predicate(op))
        .count();
    assert_eq!(
        stray_predicates, 0,
        "the source Err-arm predicate must be gone after the drain fusion"
    );

    // keep returns `(bool, carrier)`; the bool switch may sit on the
    // handler or on its unique successor after the pair is unpacked.
    let reraise = handlers
        .iter()
        .find_map(|(block, _)| {
            let block = &graph.blocks[*block];
            block
                .exits
                .iter()
                .find(|link| link.exitcase == Some(ExitCase::Bool(false)))
                .or_else(|| {
                    let [link] = block.exits.as_slice() else {
                        return None;
                    };
                    graph.blocks[link.target.0]
                        .exits
                        .iter()
                        .find(|link| link.exitcase == Some(ExitCase::Bool(false)))
                })
        })
        .expect("fused predicate has a reraise edge");
    assert!(
        graph.blocks[reraise.target.0]
            .operations
            .iter()
            .any(|op| is_root_scope_close(&op.kind)),
        "drain reraise closes its RootScope"
    );

    let exc_kind_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("exc_kind_discriminant")
            )
        })
        .count();
    assert_eq!(
        exc_kind_calls, 0,
        "the fused handler must not reintroduce the flat exception-kind test"
    );

    // The fused next() call site carries a LastException exit.
    let lastexc_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    assert!(
        lastexc_blocks >= 1,
        "the drain next() site must become a LastException exception-edge"
    );

    // Reload livevars through the named `shadow_stack_get` residual, not a
    // `||` closure. A closure residual is `Method { name: "call",
    // receiver_root: Some("closure" | "closure#N") }`, which the codewriter
    // mints as `target:closure.call` — a hash `jit_trace_fnaddrs` cannot bind,
    // so `refuse_reachable_symbolic_residuals` aborts the jd1 walk.
    let closure_calls: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target:
                    CallTarget::Method {
                        name,
                        receiver_root: Some(receiver),
                        ..
                    },
                ..
            } if name == "call" && receiver.starts_with("closure") => {
                Some(format!("{receiver}.{name}"))
            }
            _ => None,
        })
        .collect();
    assert!(
        closure_calls.is_empty(),
        "unpackiterable_portal must not residualize `||` closures: {closure_calls:?}"
    );
    let shadow_stack_gets = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("shadow_stack_get")
            )
        })
        .count();
    assert!(
        shadow_stack_gets >= 1,
        "drain livevars must reload through named shadow_stack_get"
    );

    eprintln!(
        "drain fusion: fused_predicate={fused_predicates} \
         stray_predicate={stray_predicates} exc_kind_discriminant={exc_kind_calls} \
         lastexc_blocks={lastexc_blocks} shadow_stack_get={shadow_stack_gets}"
    );
}

fn call_target_leaf(target: &CallTarget) -> String {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().cloned().unwrap_or_default(),
        CallTarget::Method { name, .. } => name.clone(),
        other => format!("{other:?}"),
    }
}

/// `execute_opcode_step`'s `Instruction::ForIter` arm is `return execute_for_iter(...)`:
/// a scoped Result tail-forward.  That call must carry `LastException` so a
/// StopIteration from `iter_next` / `space.next` becomes `catch_exception`
/// (`flatten.py make_exception_link`) rather than the portal's abort arm.
#[test]
fn execute_for_iter_tail_forward_gets_lastexception_exits() {
    let graph = lower_function(interp(), "execute_opcode_step").expect("lower execute_opcode_step");
    let call_block = graph
        .blocks
        .iter()
        .find(|b| {
            b.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. }
                        if call_target_leaf(target) == "execute_for_iter"
                )
            })
        })
        .expect("execute_opcode_step calls execute_for_iter");
    assert!(
        matches!(call_block.exitswitch, Some(ExitSwitch::LastException)),
        "FOR_ITER tail-forward must get LastException exits"
    );
    assert!(
        call_block
            .exits
            .iter()
            .any(|link| link.last_exception.is_some() && link.last_exc_value.is_some()),
        "FOR_ITER tail-forward exception link carries last_exception/last_exc_value"
    );
}

#[test]
fn eval_loop_custom_match_gets_catch_and_rewrap() {
    let llbc = interp();
    let graph = lower_function(llbc, "pyre_interpreter::eval::eval_loop").expect("lower eval_loop");
    // The execute_opcode_step call block must carry LastException exits.
    let call_block = graph
        .blocks
        .iter()
        .find(|b| {
            b.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some("execute_opcode_step")
                )
            })
        })
        .expect("eval_loop calls execute_opcode_step");
    assert!(
        matches!(call_block.exitswitch, Some(ExitSwitch::LastException)),
        "custom-match call site gets catch-and-rewrap LastException exits"
    );
    // `except OperationError as e`: the call block catches the carrier.
    assert!(
        call_block
            .exits
            .iter()
            .any(|link| link.exitcase == Some(ExitCase::ErrorCarrier)),
        "custom-match call site catches the error carrier"
    );
    // At runtime the exception arm re-binds the caught value into the
    // PyError domain before rebuilding the Err shell.
    let graph = lower_function_to_runtime_edges(llbc, "pyre_interpreter::eval::eval_loop")
        .expect("lower eval_loop");
    let from_exc_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("from_exc_object")
            )
        })
        .count();
    assert!(
        from_exc_calls >= 1,
        "rewrap exception arm binds PyError::from_exc_object(last_exc_value)"
    );
}

/// Count the raise-path calls in `name`'s lowered graph: fused, unfused
/// materialisations, and surviving `PyError` constructors.
fn raise_path_calls(name: &str) -> (usize, usize, usize) {
    raise_path_calls_in(interp(), name)
}

/// The same count against a named artefact, for a wrapper whose module is not
/// in `pyre-interpreter`.
fn raise_path_calls_in(llbc: &'static Llbc, name: &str) -> (usize, usize, usize) {
    let graph = lower_function_to_runtime_edges(llbc, name)
        .unwrap_or_else(|e| panic!("lower {name}: {e:?}"));
    let (mut fused, mut materialise, mut ctors) = (0, 0, 0);
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &op.kind
            else {
                continue;
            };
            match segments.last().map(String::as_str) {
                Some("pyerror_type_error_to_exc_object")
                | Some("pyerror_zero_division_to_exc_object")
                | Some("pyerror_value_error_to_exc_object")
                | Some("pyerror_index_error_to_exc_object") => fused += 1,
                Some("pyerror_to_exc_object") => materialise += 1,
                Some(_) if segments.len() >= 2 && segments[segments.len() - 2] == "PyError" => {
                    ctors += 1
                }
                _ => {}
            }
        }
    }
    (fused, materialise, ctors)
}

#[test]
fn constant_message_raise_sites_fuse_their_constructor() {
    // `list_to_tuple_value` refuses a non-list with a literal message, which
    // is the whole of its raise path. That site must reach the published
    // `pyerror_type_error_to_exc_object`, leaving no `PyError` constructor
    // behind: the constructor is transparent, has no host symbol, and one of
    // them anywhere in the body refuses the whole descent.
    let (fused, materialise, ctors) =
        raise_path_calls("pyre_interpreter::opcode_ops::list_to_tuple_value");
    assert!(fused > 0, "the fusion must fire on a literal-message raise");
    assert_eq!(ctors, 0, "no PyError constructor may survive");
    assert_eq!(materialise, 0, "no unfused materialisation may survive");
}

#[test]
fn negative_shift_value_error_fuses_its_constructor() {
    // `descr_lshift` / `descr_rshift` raise `PyError::value_error` with the
    // literal "negative shift count" before `rbigint.lshift` / `rbigint.rshift`.
    // That constructor must become `pyerror_value_error_to_exc_object`. Leaving
    // the `PyError` aggregate in the graph makes the native materialiser read
    // the message word as a `Wtf8Buf` niche (empty message, or the shift count
    // as a huge length).
    for name in [
        "pyre_interpreter::objspace::descroperation::long_lshift",
        "pyre_interpreter::objspace::descroperation::long_rshift",
        "pyre_interpreter::objspace::descroperation::int_lshift",
        "pyre_interpreter::objspace::descroperation::int_rshift",
    ] {
        let (fused, _materialise, _ctors) = raise_path_calls(name);
        let graph = lower_function_to_runtime_edges(interp(), name).expect("lower");
        let leftover: Vec<String> = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("value_error") => {
                    Some(segments.join("::"))
                }
                _ => None,
            })
            .collect();
        assert!(
            fused > 0,
            "{name}: literal ValueError must fuse (fused={fused})"
        );
        assert!(
            leftover.is_empty(),
            "{name}: PyError::value_error must not survive: {leftover:?}"
        );
    }
}

#[test]
fn gateway_wrapper_refusals_all_residualize() {
    // A generated wrapper words none of its own refusals: the receiver test
    // and the two arity tests each call a `dont_look_inside` helper, whose
    // `Result` return carries the refusal out on its own exception link. So
    // the wrapper is left with no constructor to fuse and nothing to
    // materialise, and the formatting stays out of its JitCode.
    //
    // `receiver_mismatch` is the one that has to be asked for by name: it
    // reports a runtime receiver type, so unlike its two neighbours it could
    // never have been a literal, and left transparent it put one
    // materialisation in the wrapper per call site.
    // `_random` lives in `pyre-module`, so `__majit_wrap_random`'s graph is in
    // that artefact rather than the interpreter's.
    for (llbc, name) in [
        (optional_module(), "__majit_wrap_random"),
        (interp(), "__majit_wrap_getvalue"),
    ] {
        let (fused, materialise, ctors) = raise_path_calls_in(llbc, name);
        assert_eq!((fused, materialise, ctors), (0, 0, 0), "{name}");
        let graph = lower_function(llbc, name).expect("lower");
        let residuals = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if matches!(
                            segments.last().map(String::as_str),
                            Some("receiver_mismatch")
                                | Some("method_arity_failure")
                                | Some("method_noarg_failure")
                        )
                )
            })
            .count();
        assert!(residuals > 0, "{name}: the refusals must survive as calls");
    }
}

#[test]
fn exact_int_zero_division_raise_sites_fuse_their_constructor() {
    // `int_floordiv` and `int_mod` are the two exact-int operator bodies that
    // raise ZeroDivisionError.  Their shared literal message must reach the
    // fused materialiser so the generated descent carries a
    // `W_BaseException`, never an in-trace Rust `PyError` aggregate.
    for name in ["int_floordiv", "int_mod"] {
        let (fused, materialise, ctors) = raise_path_calls(name);
        assert!(fused > 0, "{name}: zero-division fusion must fire");
        assert_eq!(ctors, 0, "{name}: no PyError constructor may survive");
        assert_eq!(
            materialise, 0,
            "{name}: no unfused materialisation may survive"
        );
    }
}

#[test]
fn formatted_message_raise_sites_keep_the_two_call_form() {
    // `list_extend_value` words its non-iterable refusal with `format!`,
    // whose result is not the `box_str_constant` object the helper reads.
    // That site must keep the constructor plus `pyerror_to_exc_object`: the
    // fusion is additive and never replaces its own fallback.
    let (fused, materialise, ctors) = raise_path_calls("list_extend_value");
    assert_eq!(fused, 0, "a formatted message must not fuse");
    assert!(
        ctors > 0,
        "the fixture must still raise through a constructor"
    );
    assert_eq!(
        materialise, ctors,
        "every declined site keeps both halves of the pair"
    );
}

/// `list_extend_value` is `Ok(_) / Err(e) if pred(x) / Err(_)`. Drain fuse
/// treats the Err edge as the exception link and leaves the guard as
/// ordinary flow (`flowcontext.py` `FlowContext.guessexception`).
#[test]
fn list_extend_value_guarded_err_lowers() {
    let graph = lower_function(interp(), "pyre_interpreter::opcode_ops::list_extend_value")
        .unwrap_or_else(|e| panic!("lower list_extend_value: {e}"));
    assert_eq!(count_result_ctors(&graph), 0, "Result shells must be gone");
    let discs = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref().is_some_and(|o| o.contains("Result"))
            )
        })
        .count();
    assert_eq!(discs, 0, "the Result match is fused away");
    let lastexc = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    assert!(lastexc >= 1, "the call site is a LastException diamond");
    let raises = graph
        .blocks
        .iter()
        .filter(|b| b.exits.iter().any(|l| l.target == graph.exceptblock))
        .count();
    assert!(
        raises >= 2,
        "guarded re-raise and the new raise both reach exceptblock, got {raises}"
    );
}

#[test]
fn list_append_underflow_keeps_its_unfused_materialisation() {
    // `opcode_list_append` raises through
    // `shared_opcode::stack_underflow_error`, not a `PyError` constructor, so
    // there is nothing to fuse and the single materialisation call stands.
    assert_eq!(raise_path_calls("opcode_list_append"), (0, 1, 0));
}

/// Family C: the dual-gate slot on always-`Err` `__new__` wrappers is
/// the `result::Result<*mut PyObject,PyError>::Ok.__pos_0` extract, not the
/// returnblock inputarg.
/// The annotator never follows that arm (`links_followed`).  Indices
/// move when the wrapper changes crates, so the test looks for the
/// field rather than a frozen var number.
#[test]
fn wrap_new_always_err_ok_payload_is_result_fieldread() {
    for name in [
        "pyre_module::module::_hashlib::hash_state_class::__majit_wrap___new__",
        "pyre_module::module::_ssl::ssl_session_methods::__majit_wrap___new__",
    ] {
        let g =
            lower_function(optional_module(), name).unwrap_or_else(|e| panic!("lower {name}: {e}"));
        let field = g.blocks.iter().find_map(|block| {
            block.operations.iter().find_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref()
                            == Some("result::Result<*mut PyObject,PyError>::Ok") =>
                {
                    Some(field)
                }
                _ => None,
            })
        });
        assert!(
            field.is_some(),
            "{name}: family C slot must be the unfollowed Ok payload"
        );
    }
}

/// `finditem_str_named` / `load_attr_cached` / `store_attr_cached` each
/// collect a scoped Result call on the `if not we_are_jitted()`
/// interpreter arm.  The front folds `we_are_jitted()` to
/// `ConstBool(true)` and `fold_constant_exitswitch` clears that arm,
/// so the collected var has no producer; the live arm still lowers.
#[test]
fn finditem_str_named_and_attr_cached_lower() {
    let mut named = None;
    for name in [
        "pyre_interpreter::baseobjspace::finditem_str_named",
        "pyre_interpreter::eval::<Impl>::load_attr_cached",
        "pyre_interpreter::eval::<Impl>::store_attr_cached",
    ] {
        let graph = lower_function(interp(), name).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(
            count_result_ctors(&graph),
            0,
            "{name}: Result shells must be gone"
        );
        if name.ends_with("finditem_str_named") {
            named = Some(graph);
        }
    }
    let named = named.expect("finditem_str_named lowered");
    let leaves: Vec<String> = named
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            _ => None,
        })
        .collect();
    assert!(
        leaves.iter().any(|leaf| leaf == "finditem_str_generic"),
        "jitted-false arm is gone; the generic tail-forward stays: {leaves:?}"
    );
    assert!(
        leaves
            .iter()
            .all(|leaf| leaf != "finditem_str_shortcut_interp"),
        "we_are_jitted ConstBool(true) clears the interpreter shortcut: {leaves:?}"
    );
}
#[test]
fn a_payload_less_err_shell_of_an_inlined_callee_is_left_materialised() {
    // `from_utf8`'s `Err(Utf8Error)` is built field by field and consumed in
    // the body; it never reaches `returnblock`, so the callee rule skips it
    // like the other consumed intermediates.
    lower_function(
        interp(),
        "pyre_interpreter::baseobjspace::module_miss_error",
    )
    .expect("module_miss_error lowers under the carrier");
}

#[test]
fn a_dont_look_inside_by_value_adt_return_declares_its_class() {
    let graph = lower_function(interp(), "pyre_interpreter::call::take_call_error")
        .expect("take_call_error lowers");
    assert_eq!(
        graph.return_class_root.as_deref(),
        Some("Option<PyError>"),
        "Option<PyError> is rustc's pointer niche (`PyError` wraps NonNull); \
         dont_look_inside seeds the nullable payload stamp, not the tagged \
         Option enum class. rclass.py InstanceRepr.convert_const(None) is \
         null_instance."
    );
}

#[test]
fn a_formatd_residual_declares_a_string_and_keeps_its_host_formatter_call() {
    for path in [
        "pyre_interpreter::display::jit_format_float_repr_rstr",
        "pyre_interpreter::typedef::jit_format_complex_component_repr_rstr",
    ] {
        let graph = lower_function(interp(), path).expect("formatd residual lowers");
        assert!(
            graph.return_is_str,
            "{path}: the `*mut Utf8Str` result is the `SomeString` a stub returns"
        );
        let calls_itself = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.join("::") == path
                )
            });
        assert!(
            !calls_itself,
            "{path}: the wrapper must not be retargeted onto itself"
        );
    }
}

#[test]
fn result_map_of_some_builds_the_option_instead_of_a_fn_const() {
    let graph = lower_function(
        interp(),
        "pyre_interpreter::display::exception_kind_str_wtf8",
    )
    .expect("exception_kind_str_wtf8 lowers");
    let ops: Vec<&OpKind> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .map(|op| &op.kind)
        .collect();
    let fn_consts: Vec<_> = ops
        .iter()
        .filter_map(|kind| match kind {
            OpKind::Call { target, .. } => majit_translate::model::fn_const_segments(target),
            _ => None,
        })
        .collect();
    assert!(
        fn_consts.iter().all(|segments| {
            segments.last().map(String::as_str) == Some("w_none")
                && segments.iter().any(|s| s == "noneobject")
        }),
        "`.map(Some)` leaves no Some function-item define: {fn_consts:?}"
    );
    assert_eq!(
        fn_consts.len(),
        3,
        "three `unwrap_or_else(w_none)` keep the function item, got {fn_consts:?}"
    );
    let unwrap_or_else_calls = ops
        .iter()
        .filter(|kind| match kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => name == "unwrap_or_else",
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some("unwrap_or_else")
                    && segments.iter().any(|s| s == "option")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        unwrap_or_else_calls, 3,
        "caller keeps three regular calls to the core unwrap_or_else body, got {unwrap_or_else_calls}"
    );
    assert!(
        !ops.iter().any(|kind| matches!(kind,
            OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "map")),
        "`.map(Some)` leaves no residual Result::map"
    );
}

/// `ll_math.py ll_math_sqrt` raises `ValueError("math domain error")` in
/// the raising block. After `lower_carrier_raises` that is constructor +
/// `pyerror_to_exc_object` in the same block; fusion must rewrite it to
/// the published helper so `make_pyerror` is not a symbolic descent
/// blocker (`fuse_kind_ctor_raise`).
#[test]
fn ll_math_sqrt_fuses_literal_domain_error() {
    let name = "pyre_interpreter::objspace::descroperation::ll_math_sqrt";
    let (fused, materialise, ctors) = raise_path_calls(name);
    let graph = lower_function_to_runtime_edges(interp(), name).expect("lower");
    let leftover: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if matches!(
                segments.last().map(String::as_str),
                Some("value_error") | Some("make_pyerror") | Some("pyerror_to_exc_object")
            ) =>
            {
                Some(segments.join("::"))
            }
            _ => None,
        })
        .collect();
    assert!(
        fused > 0,
        "{name}: literal domain error must fuse (fused={fused})"
    );
    assert_eq!(ctors, 0, "{name}: no PyError constructor may survive");
    assert_eq!(
        materialise, 0,
        "{name}: no unfused materialisation may survive"
    );
    assert!(
        leftover.is_empty(),
        "{name}: domain-error constructor must not survive: {leftover:?}"
    );
}

fn call_path_leaf(op: &OpKind) -> Option<(String, String)> {
    match op {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => {
            let path = segments.join("::");
            let leaf = segments.last().cloned().unwrap_or_default();
            let unspecialized = leaf
                .split("__spec_")
                .next()
                .unwrap_or(leaf.as_str())
                .to_string();
            Some((path, unspecialized))
        }
        _ => None,
    }
}

fn leftover_symbolic_oefmt_helpers(graph: &majit_translate::model::FunctionGraph) -> Vec<String> {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| {
            let (path, unspecialized) = call_path_leaf(&op.kind)?;
            let hits = matches!(
                unspecialized.as_str(),
                "make_pyerror"
                    | "value_error"
                    | "oefmt"
                    | "oefmt_single_t"
                    | "oefmt_single_s"
                    | "push_roots"
                    | "kind_of_canonical_exc_class"
                    | "FMT_PTR"
                    | "FMT_LEN"
            ) || (unspecialized == "from" && path.contains("String"))
                || (unspecialized == "as_ptr" && path.contains("str"))
                || (unspecialized == "len" && path.contains("core::str"));
            hits.then_some(path)
        })
        .collect()
}

/// `math1`'s `except ValueError: raise oefmt(..., "%s", repr)` is looked
/// inside (`error.py oefmt` / `OpErrFmt.__init__`). `String::from`,
/// `make_pyerror`, and `oefmt_single_t` (`push_roots` / `str::as_ptr` /
/// `kind_of_canonical_exc_class`) are symbolic fnaddrs; any one after
/// `ll_math_sqrt`'s domain residual refuses the `__majit_wrap_math_sqrt`
/// descent (`summarize_body_blockers_with` `blocker_after_effect`).
#[test]
fn sqrt_translate_ll_error_has_no_string_from_or_make_pyerror() {
    let name = "sqrt_translate_ll_error";
    let graph = lower_function(optional_module(), name).expect("lower");
    let leftover = leftover_symbolic_oefmt_helpers(&graph);
    assert!(
        leftover.is_empty(),
        "{name}: OpErrFmt.__init__ must not leave oefmt_single_t / push_roots / str::as_ptr: {leftover:?}"
    );
}

/// `oefmt_one` is `get_operrcls2.OpErrFmt.__init__`: malloc + setfield.
/// Walking `oefmt_single_t` after `formatd` is the `push_roots` blocker
/// that residualises `math.sqrt` as `CallMayForceR`.
#[test]
fn oefmt_one_is_operrfmt_init_without_oefmt_single_t() {
    let name = "oefmt_one";
    let graph = lower_function(interp(), name).expect("lower");
    let leftover = leftover_symbolic_oefmt_helpers(&graph);
    assert!(
        leftover.is_empty(),
        "{name}: must tail-call make_pyerror_deferred_t, not oefmt_single_t: {leftover:?}"
    );
}

fn return_producer<'a>(
    graph: &'a majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    let mut current = var.clone();
    let mut seen = Vec::new();
    loop {
        if seen.iter().any(|found| found == &current) {
            return None;
        }
        seen.push(current.clone());
        if let Some(kind) = graph.blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find_map(|op| (op.result.as_ref() == Some(&current)).then_some(&op.kind))
        }) {
            match kind {
                OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                    current = operand.clone();
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("__cast_instance_intrinsic") => {
                    let Some(src) = args.first().and_then(LinkArg::as_variable) else {
                        return Some(kind);
                    };
                    current = src.clone();
                }
                other => return Some(other),
            }
            continue;
        }
        // A frame-exit cleanup block rebinds the returned word onto its
        // inputarg. One predecessor passing one value is that word.
        let src = single_inputarg_source(graph, &current)?;
        current = src;
    }
}

/// The value a single predecessor passes into `var` when `var` is an inputarg
/// of exactly one block. A merge, a missing link arg, or a second binding is
/// not one producer.
fn single_inputarg_source(
    graph: &majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<majit_translate::flowspace::model::Variable> {
    let mut source = None;
    for (bi, block) in graph.blocks.iter().enumerate() {
        let Some(pos) = block.inputargs.iter().position(|input| input == var) else {
            continue;
        };
        if source.is_some() {
            return None;
        }
        let mut incoming = Vec::new();
        for pred in &graph.blocks {
            for link in &pred.exits {
                if link.target.0 == bi {
                    incoming.push(link);
                }
            }
        }
        let [link] = incoming.as_slice() else {
            return None;
        };
        let src = link.args.get(pos)?.as_variable()?.clone();
        source = Some(src);
    }
    source
}

/// `space.index_w` returns the `Some` payload of `Option<Result<i64, PyError>>`.
/// That shell is `Ref`; the scalar callee must forward `Ok`'s `i64` instead.
/// `lower_function` does not stamp `FUNC.RESULT` (registration does), so this
/// asserts the CFG return, not `return_type`.
#[test]
fn space_index_w_returns_ok_i64() {
    use majit_translate::model::{LinkArg, ValueType};
    let g = lower_function(interp(), "pyre_interpreter::builtins::space_index_w")
        .unwrap_or_else(|e| panic!("lower: {e}"));
    let mut ok_returns = 0usize;
    for block in &g.blocks {
        for link in &block.exits {
            if link.target != g.returnblock {
                continue;
            }
            assert_eq!(link.args.len(), 1, "scalar return has one arg");
            let LinkArg::Value(var) = &link.args[0] else {
                panic!("return arg is a value");
            };
            let Some(OpKind::FieldRead { field, ty, .. }) = return_producer(&g, var) else {
                panic!("return {var:?} is not a field read");
            };
            let owner = field.owner_root.as_deref().unwrap_or("");
            assert_eq!(field.name, "__pos_0", "owner {owner}");
            assert!(
                owner.ends_with("::Ok") && owner.contains("Result<i64,PyError>"),
                "return owner {owner}"
            );
            assert!(
                !owner.ends_with("::Some"),
                "Option shell still reaches returnblock: {owner}"
            );
            assert_eq!(ty, &ValueType::Int, "Ok payload ty {ty:?}");
            ok_returns += 1;
        }
    }
    assert_eq!(
        ok_returns, 1,
        "joined as_index_value successes return Ok's i64"
    );
}

/// `getindex_w_index` is `space_index(index)?` followed by a `match` on
/// `with_roots!(index => int_w(w_index))`. Restore hops sit between the
/// rebuilt shells and the discriminant; collapsing them forwards `int_w`'s
/// `i64` to the Ok arm, so the function return is that call. The `?` on
/// `space_index` is an exception exit of that call (`shadow_stack_erase`
/// takes the `with_roots!` bracket out, so the callee raises instead of
/// returning a Result shell). Reminting it to `i64` would make the CFG
/// return `void` while `FUNC.RESULT` is `i` (`func_result_kind`,
/// `history.getkind`). The two `int_w` `Err` arms still raise through
/// `lower_error_carrier_edges` / `pyerror_to_exc_object`.
#[test]
fn getindex_w_index_from_residual_raises() {
    use majit_translate::model::{LinkArg, ValueType};
    let path = "pyre_interpreter::baseobjspace::getindex_w_index";
    let g =
        lower_function_to_runtime_edges(interp(), path).unwrap_or_else(|e| panic!("lower: {e}"));
    let mut reachable = vec![false; g.blocks.len()];
    let mut stack = vec![g.startblock.0];
    while let Some(block) = stack.pop() {
        if block >= reachable.len() || reachable[block] {
            continue;
        }
        reachable[block] = true;
        for link in &g.blocks[block].exits {
            stack.push(link.target.0);
        }
    }
    let mut ok_returns = 0usize;
    let mut exc_materialisers = 0usize;
    let mut space_index_raises = false;
    for (bi, block) in g.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        let space_index_here = block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("space_index")
            )
        });
        for op in &block.operations {
            if let OpKind::Call { target, .. } = &op.kind {
                match target {
                    CallTarget::Method { name, .. } if name == "from_residual" => {
                        panic!("reachable from_residual still returns a value");
                    }
                    CallTarget::FunctionPath { segments, .. }
                        if segments.last().map(String::as_str) == Some("pyerror_to_exc_object") =>
                    {
                        exc_materialisers += 1;
                    }
                    _ => {}
                }
            }
        }
        for link in &block.exits {
            if space_index_here && link.target == g.exceptblock {
                space_index_raises = true;
            }
            if link.target != g.returnblock {
                continue;
            }
            assert_eq!(link.args.len(), 1, "scalar return has one arg");
            let LinkArg::Value(var) = &link.args[0] else {
                panic!("return arg is a value");
            };
            match return_producer(&g, var) {
                Some(OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty,
                    ..
                }) if segments.last().map(String::as_str) == Some("int_w") => {
                    assert_eq!(*result_ty, ValueType::Int, "int_w result_ty {result_ty:?}");
                }
                other => panic!("return {var:?} is not int_w's i64: {other:?}"),
            }
            ok_returns += 1;
        }
    }
    assert_eq!(ok_returns, 1, "{path}");
    assert!(
        space_index_raises,
        "space_index ? is an exception exit of that call"
    );
    assert_eq!(
        exc_materialisers, 2,
        "two int_w Err arms raise through pyerror_to_exc_object"
    );
    let mut raised_overflow_rewrite = false;
    let mut raised_except_link = false;
    for (bi, block) in g.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        for op in &block.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("pyerror_to_exc_object") {
                continue;
            }
            let Some(arg) = args.first().and_then(LinkArg::as_variable) else {
                continue;
            };
            match return_producer(&g, arg) {
                Some(OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                }) if segments.last().map(String::as_str) == Some("new")
                    && segments.iter().any(|s| s == "PyError") =>
                {
                    raised_overflow_rewrite = true;
                }
                Some(OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                }) if segments.last().map(String::as_str) == Some("from_exc_object") => {
                    // `Err(e) => Err(e)` after the bracket is gone: int_w's
                    // exception is caught, wrapped, then raised again.
                    raised_except_link = true;
                }
                None => raised_except_link = true,
                other => panic!("pyerror_to_exc_object arg producer {other:?}"),
            }
        }
    }
    assert!(
        raised_overflow_rewrite,
        "OverflowError arm raises PyError::new"
    );
    assert!(
        raised_except_link,
        "an int_w Err arm raises the LastException payload"
    );
}

/// `eval_loop` uses `decode_instruction_forward(code, pc)?`. The callee's
/// error is `BytecodeCorruption` and the function returns `PyError` through
/// `impl From<BytecodeCorruption> for PyError`. `FromResidual::from_residual`
/// is `Err(From::from(e))`, so the raised carrier is that `from` result.
/// `lower_error_carrier_edges` wraps it in `pyerror_to_exc_object`.
#[test]
fn eval_loop_converts_bytecode_corruption_before_raising() {
    use majit_translate::model::LinkArg;
    let path = "pyre_interpreter::eval::eval_loop";
    let g =
        lower_function_to_runtime_edges(interp(), path).unwrap_or_else(|e| panic!("lower: {e}"));
    let mut reachable = vec![false; g.blocks.len()];
    let mut stack = vec![g.startblock.0];
    while let Some(block) = stack.pop() {
        if block >= reachable.len() || reachable[block] {
            continue;
        }
        reachable[block] = true;
        for link in &g.blocks[block].exits {
            stack.push(link.target.0);
        }
    }
    let mut converted = false;
    let mut saw_corruption = false;
    for (bi, block) in g.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        for op in &block.operations {
            if matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "from_residual"
            ) {
                panic!("reachable from_residual still returns a value at block {bi}");
            }
            if matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.owner_root.as_deref().is_some_and(|owner| {
                        owner.contains("BytecodeCorruption")
                    })
            ) {
                saw_corruption = true;
            }
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("pyerror_to_exc_object") {
                continue;
            }
            let Some(arg) = args.first().and_then(LinkArg::as_variable) else {
                continue;
            };
            let Some(OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args: from_args,
                ..
            }) = return_producer(&g, arg)
            else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("from")
                || !segments.iter().any(|seg| seg == "PyError")
                || !segments
                    .iter()
                    .any(|seg| seg.starts_with("<impl ") && seg.ends_with('>'))
            {
                continue;
            }
            assert!(
                !from_args.is_empty(),
                "BytecodeCorruption is a named empty class; From::from receives the instance"
            );
            converted = true;
        }
    }
    assert!(
        saw_corruption,
        "{path}: the residual owner must name BytecodeCorruption"
    );
    assert!(
        converted,
        "{path}: BytecodeCorruption from_residual must raise From::from"
    );
}

/// `encode_ascii` is `Result<Vec<u8>, PyError>`. The residual stub must
/// project the Ok payload as a byte list, not a classdef-less `ref`
/// that `List ∪ Instance` cannot merge at `encode_object`'s return.
#[test]
fn encode_ascii_residual_returns_a_byte_list() {
    let g = lower_function(interp(), "pyre_interpreter::codec_engine::encode_ascii")
        .expect("encode_ascii lowers");
    let root = g.return_container_root.as_deref().unwrap_or("");
    assert!(
        root.contains("Vec<u8>") || root.contains("[u8]"),
        "encode_ascii Result<Vec<u8>> stub must be a byte list, got {root:?}"
    );
}

/// `interp_sre.py FLAG_NAMES` is a prebuilt list. The NamedConst
/// `SRE_FLAG_NAMES: [&str; 9]` must fold to `__const_str_array`, not a
/// residual accessor Call the registry cannot bind.
#[test]
fn sre_flag_names_folds_to_a_str_list() {
    let g = lower_function(
        interp(),
        "pyre_interpreter::module::_sre::interp_sre::sre_pattern_flag_tail",
    )
    .expect("sre_pattern_flag_tail lowers");
    let mut residual = 0usize;
    let mut tables = Vec::new();
    for op in g.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } = &op.kind
        else {
            continue;
        };
        if segments.last().map(String::as_str) == Some("SRE_FLAG_NAMES") {
            residual += 1;
        }
        if segments.first().map(String::as_str) == Some("__const_str_array") {
            tables.push(segments.clone());
        }
    }
    assert_eq!(
        residual, 0,
        "sre_pattern_flag_tail left an SRE_FLAG_NAMES accessor"
    );
    assert!(
        tables.iter().any(|segments| {
            segments.len() == 10 && segments[1] == "re.TEMPLATE" && segments[9] == "re.ASCII"
        }),
        "sre_pattern_flag_tail must bake FLAG_NAMES, got {tables:?}"
    );
}

fn leftover_ok_or_else(graph: &majit_translate::model::FunctionGraph) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } if name == "ok_or_else" => true,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().map(String::as_str) == Some("ok_or_else") => true,
            _ => false,
        })
        .count()
}

fn assert_no_ok_or_else(graph: &majit_translate::model::FunctionGraph, name: &str) {
    assert_eq!(
        leftover_ok_or_else(graph),
        0,
        "{name} left a foreign Option::ok_or_else call"
    );
}

/// Tail-return `opt.ok_or_else(|| E)`: the residual Method is gone.
#[test]
fn lookup_dunder_import_ok_or_else_is_lowered() {
    let graph =
        lower_function(interp(), "lookup_dunder_import").expect("lookup_dunder_import lowers");
    assert_no_ok_or_else(&graph, "lookup_dunder_import");
}

/// Tail-return `opt.ok_or_else(named_fn)`: the function-item callable is a
/// direct `Call(FunctionPath)`, not a residual Method.
#[test]
fn try_new_buffer_ok_or_else_is_lowered() {
    let graph = lower_function(interp(), "try_new_buffer").expect("try_new_buffer lowers");
    assert_no_ok_or_else(&graph, "try_new_buffer");
}

/// `args.first().copied().ok_or_else(...)?` gateway: the foreign combinator
/// is gone even when the `?` diamond is not the private from_residual shape
/// the fused try-site requires.
#[test]
fn iobase_writelines_ok_or_else_is_lowered() {
    let graph = lower_function(interp(), "iobase_writelines").expect("iobase_writelines lowers");
    assert_no_ok_or_else(&graph, "iobase_writelines");
}

/// Tail-return niche `Option<&mut W_MMap>::ok_or_else`.
#[test]
fn mmap_this_ok_or_else_is_lowered() {
    let graph = lower_function(optional_module(), "mmap_this").expect("mmap_this lowers");
    assert_no_ok_or_else(&graph, "mmap_this");
}

/// Tail-return `queue_pop_front(...).ok_or_else(empty_error)`.
#[test]
fn simplequeue_get_ok_or_else_is_lowered() {
    let graph =
        lower_function(optional_module(), "simplequeue_get").expect("simplequeue_get lowers");
    assert_no_ok_or_else(&graph, "simplequeue_get");
}

/// `opt.ok_or_else(|| E).and_then(...)` — the Result is a combinator
/// consumer, not a `?` diamond.
#[test]
fn create_new_dict_ok_or_else_is_lowered() {
    let graph = lower_function(interp(), "create_new_dict").expect("create_new_dict lowers");
    assert_no_ok_or_else(&graph, "create_new_dict");
}

fn producer_op<'a>(
    graph: &'a majit_translate::model::FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a majit_translate::model::SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find(|op| op.result.as_ref() == Some(var))
}

fn value_type_is_payload(ty: &majit_translate::model::ValueType) -> bool {
    use majit_translate::model::ValueType;
    match ty {
        ValueType::Str | ValueType::StringBuilder => true,
        ValueType::Ref(Some(s)) => {
            s.contains("PyObject") || s.contains("string::String") || s.contains("&str")
        }
        _ => false,
    }
}

/// A merged continue whose one predecessor is a payload (PyObject / Str)
/// and another is unit `ConstNone` is the census `X ∪ Void` shape
/// (`annrpython.py` `follow_link`; `rnone.py` `NoneRepr` is only None).
fn payload_phi_joined_with_constnone(graph: &majit_translate::model::FunctionGraph) -> Vec<String> {
    use majit_translate::model::{LinkArg, OpKind};
    let mut hits = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        let mut preds = Vec::new();
        for src in &graph.blocks {
            for link in &src.exits {
                if link.target.0 == bi {
                    preds.push(link);
                }
            }
        }
        if preds.len() < 2 {
            continue;
        }
        for slot in 0..block.inputargs.len() {
            let mut saw_constnone = false;
            let mut saw_payload = false;
            for link in &preds {
                let Some(arg) = link.args.get(slot) else {
                    continue;
                };
                let LinkArg::Value(v) = arg else {
                    continue;
                };
                let Some(op) = producer_op(graph, v) else {
                    continue;
                };
                match &op.kind {
                    OpKind::ConstNone => saw_constnone = true,
                    OpKind::Call { result_ty, .. } | OpKind::IndirectCall { result_ty, .. }
                        if value_type_is_payload(result_ty) =>
                    {
                        saw_payload = true;
                    }
                    OpKind::FieldRead { ty, .. } if value_type_is_payload(ty) => {
                        saw_payload = true;
                    }
                    _ => {}
                }
            }
            if saw_constnone && saw_payload {
                hits.push(format!("bb{bi} slot{slot}"));
            }
        }
    }
    hits
}

fn assert_no_payload_constnone_phi(graph: &majit_translate::model::FunctionGraph, name: &str) {
    let hits = payload_phi_joined_with_constnone(graph);
    assert!(
        hits.is_empty(),
        "{name} merged a payload with ConstNone: {hits:?}"
    );
}

/// `Some(f) => f(..)` fn-pointer `Result<PyObjectRef, PyError>` merged with
/// the named `unpackiterable_portal` call. The hook is `Call(__dyn_call)`.
#[test]
fn unpack_portal_runner_dyn_call_return_carries_payload() {
    let graph =
        lower_function(interp(), "unpack_portal_runner").expect("unpack_portal_runner lowers");
    assert_no_payload_constnone_phi(&graph, "unpack_portal_runner");
    let rb = graph.returnblock.0;
    assert!(
        !graph.blocks[rb].inputargs.is_empty(),
        "unpack_portal_runner returnblock is void"
    );
}

/// Tail-return `opt.ok_or_else(err_fn)` as FunctionPath
/// `core::option::<Impl>::ok_or_else`.
#[test]
fn simplequeue_get_ok_or_else_return_carries_payload() {
    let graph =
        lower_function(optional_module(), "simplequeue_get").expect("simplequeue_get lowers");
    assert_no_ok_or_else(&graph, "simplequeue_get");
    assert_no_payload_constnone_phi(&graph, "simplequeue_get");
}

#[test]
fn get_attr_ok_or_else_return_carries_payload() {
    let graph = lower_function(optional_module(), "get_attr").expect("get_attr lowers");
    assert_no_ok_or_else(&graph, "get_attr");
    assert_no_payload_constnone_phi(&graph, "get_attr");
}

/// `args.get(1).copied().ok_or_else(...)?` in `struct::iter_unpack`: Regular
/// FunctionPath `core::option::<Impl>::ok_or_else` consumed by `?`. The
/// try-site rewrite fuses it (`exceptiontransform.py` `transform_completely`);
/// a leftover `from_residual` getattr / `ControlFlow` Break is the
/// annotate-skip that dual-recording the combinator used to leave.
#[test]
fn iter_unpack_ok_or_else_try_is_lowered() {
    let graph = lower_function(
        interp(),
        "pyre_interpreter::module::struct::<Impl>::iter_unpack",
    )
    .expect("iter_unpack lowers");
    assert_no_ok_or_else(&graph, "iter_unpack");
    assert_no_from_residual_break(&graph, "iter_unpack");
    assert_no_payload_constnone_phi(&graph, "iter_unpack");
}

fn leftover_from_residual_break(graph: &majit_translate::model::FunctionGraph) -> Vec<String> {
    let mut reachable = vec![false; graph.blocks.len()];
    let mut stack = vec![graph.startblock.0];
    while let Some(block) = stack.pop() {
        if block >= reachable.len() || reachable[block] {
            continue;
        }
        reachable[block] = true;
        for link in &graph.blocks[block].exits {
            stack.push(link.target.0);
        }
    }
    let mut hits = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        for op in &block.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "from_residual" => {
                    hits.push(format!("bb{bi} Call Method from_residual"));
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments.last().map(String::as_str) == Some("from_residual") => {
                    hits.push(format!("bb{bi} Call FunctionPath from_residual"));
                }
                OpKind::Call {
                    target:
                        CallTarget::SyntheticTransparentCtor {
                            name, owner_path, ..
                        },
                    ..
                } if name == "Break"
                    && owner_path.iter().any(|seg| seg.contains("ControlFlow")) =>
                {
                    hits.push(format!("bb{bi} ControlFlow Break ctor"));
                }
                OpKind::FieldRead { field, .. } if field.name == "from_residual" => {
                    hits.push(format!("bb{bi} getattr from_residual"));
                }
                _ => {}
            }
        }
    }
    hits
}

fn assert_no_from_residual_break(graph: &majit_translate::model::FunctionGraph, name: &str) {
    let hits = leftover_from_residual_break(graph);
    assert!(
        hits.is_empty(),
        "{name} left from_residual / ControlFlow Break: {hits:?}"
    );
}

/// `str_utf8_w(x)?` `Result<&str>` joined with an `Option<&str>` Some arm.
#[test]
fn reconfigure_encoding_join_carries_str() {
    let graph = lower_function(interp(), "reconfigure").expect("reconfigure lowers");
    assert_no_payload_constnone_phi(&graph, "reconfigure");
    let has_str_utf8 = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty: majit_translate::model::ValueType::Str,
                    ..
                } if segments.last().map(String::as_str) == Some("str_utf8_w")
            )
        })
    });
    assert!(has_str_utf8, "reconfigure lost reminted str_utf8_w");
}

fn path_from_graph_name(name: &str) -> CallPath {
    CallPath::from_segments(name.split("::").filter(|s| !s.is_empty()))
}

fn call_leaf(target: &CallTarget) -> String {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().cloned().unwrap_or_default(),
        CallTarget::Method { name, .. } => name.clone(),
        other => format!("{other:?}"),
    }
}

fn graph_summarize(graph: &majit_translate::model::FunctionGraph) -> String {
    let except_exits = graph
        .blocks
        .iter()
        .flat_map(|block| block.exits.iter())
        .filter(|link| link.target == graph.exceptblock)
        .count();
    let mut lines = vec![format!(
        "name={} return_type={:?} hints={:?} cannot_raise={} canraise={} except_exits={}",
        graph.name,
        graph.return_type,
        graph.hints,
        graph.func.cannot_raise_assertion,
        graph.func.canraise,
        except_exits,
    )];
    for block in &graph.blocks {
        let switch = match &block.exitswitch {
            Some(ExitSwitch::LastException) => " LastException".to_string(),
            Some(other) => format!(" {other:?}"),
            None => String::new(),
        };
        for op in &block.operations {
            if let OpKind::Call { target, .. } = &op.kind {
                lines.push(format!(
                    "  b{} call {}{switch}",
                    block.id.0,
                    call_leaf(target),
                ));
            }
        }
    }
    lines.join("\n")
}

fn extraeffect_of(
    cc: &CallControl,
    graph: &majit_translate::model::FunctionGraph,
    arg_types: Vec<Type>,
    result_type: Type,
) -> Result<ExtraEffect, String> {
    let path = path_from_graph_name(&graph.name);
    let target = CallTarget::from_call_path(&path);
    let op = SpaceOperation {
        result: None,
        kind: OpKind::Call {
            target,
            args: Vec::new(),
            result_ty: majit_translate::model::ValueType::Int,
        },
    };
    let mut cache = AnalysisCache::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cc.getcalldescr(
            &op,
            arg_types,
            result_type,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        )
    }));
    match result {
        Ok(descr) => Ok(descr.extra_info.extraeffect),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "getcalldescr panicked".to_string());
            Err(msg)
        }
    }
}

fn dump_insn_text(line: &str) -> &str {
    let text = line.trim();
    text.split_once(char::is_whitespace)
        .map(|(_, rest)| rest.trim())
        .unwrap_or(text)
}

/// True when some `catch_exception L` target residual-calls and `ref_return`s
/// without rebuilding a Result shell (`switch` on `__discriminant`).
fn catch_arm_ref_returns_residual(dump: &str) -> bool {
    let mut labels: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut current: Option<String> = None;
    let mut buf = String::new();
    let flush = |current: &mut Option<String>,
                 buf: &mut String,
                 labels: &mut std::collections::BTreeMap<String, String>| {
        if let Some(name) = current.take() {
            labels.insert(name, std::mem::take(buf));
        }
    };
    for line in dump.lines() {
        let text = dump_insn_text(line);
        if text == "---" {
            flush(&mut current, &mut buf, &mut labels);
            continue;
        }
        if let Some(name) = text.strip_suffix(':')
            && name.starts_with('L')
            && name[1..].chars().all(|c| c.is_ascii_digit())
        {
            flush(&mut current, &mut buf, &mut labels);
            current = Some(name.to_string());
            continue;
        }
        if current.is_some() {
            buf.push_str(text);
            buf.push('\n');
        }
    }
    flush(&mut current, &mut buf, &mut labels);
    dump.lines()
        .filter_map(|line| {
            dump_insn_text(line)
                .strip_prefix("catch_exception ")
                .map(str::trim)
        })
        .any(|target| {
            let Some(body) = labels.get(target) else {
                return false;
            };
            (body.contains("residual_call") || body.contains("pow_ovf2long"))
                && body.contains("ref_return")
                && !body.contains("switch")
                && !body.contains("reraise")
        })
}

fn jitcode_dump_of(
    cc: &mut CallControl,
    graph: &majit_translate::model::FunctionGraph,
) -> Result<String, String> {
    let path = path_from_graph_name(&graph.name);
    let jitcode = std::sync::Arc::new(JitCode::new(graph.name.clone()));
    let mut codewriter = CodeWriter::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        codewriter.transform_graph_to_jitcode(
            graph,
            &path,
            cc,
            &GraphTransformConfig::default(),
            &jitcode,
            false,
            0,
        );
    }));
    match result {
        Ok(()) => Ok(jitcode.dump()),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "transform_graph_to_jitcode panicked".to_string());
            Err(msg)
        }
    }
}

/// `int_pow` must lower `match int_pow_nomod { Ok => w_int_new, Err => pow_ovf2long }`
/// to descr_pow's `try: _pow except OverflowError: return _pow_ovf2long`.
/// The look_inside_iff trampoline residual is the raise channel of that try
/// (`ovfcheck` / `Result::Err`), so its calldescr is CanRaise and a
/// `catch_exception` targets the ovf2long arm.
#[test]
fn int_pow_trampoline_residual_is_canraise_with_catch_to_ovf2long() {
    let llbc = interp();
    let names = [
        "int_pow",
        "int_pow_nomod",
        "int_pow_nomod_trampoline",
        "_orig_int_pow_nomod",
        "__majit_call_target_int_pow_nomod",
        "__majit_call_target_int_pow_nomod_trampoline",
        "pow_ovf2long",
    ];
    let mut graphs = Vec::new();
    for name in names {
        match lower_function(llbc, name) {
            Ok(graph) => {
                eprintln!("=== graph {name} ===\n{}", graph_summarize(&graph));
                graphs.push((name, graph));
            }
            Err(err) => eprintln!("=== graph {name} MISSING: {err:?} ==="),
        }
    }

    let mut cc = CallControl::new();
    cc.set_struct_fields(with_shared_lowering(llbc, |ctx| {
        ctx.struct_field_registry()
    }));
    for (_, graph) in &graphs {
        cc.register_function_graph(path_from_graph_name(&graph.name), graph.clone());
    }

    let two_ints = vec![Type::Int, Type::Int];
    let two_refs = vec![Type::Ref, Type::Ref];
    for (name, graph) in &graphs {
        let (args, result) = if *name == "int_pow" {
            (two_refs.clone(), Type::Ref)
        } else if *name == "pow_ovf2long" {
            (two_ints.clone(), Type::Ref)
        } else {
            (two_ints.clone(), Type::Int)
        };
        match extraeffect_of(&cc, graph, args, result) {
            Ok(ee) => eprintln!("extraeffect {name} = {ee:?}"),
            Err(err) => eprintln!("extraeffect {name} FAIL: {err}"),
        }
    }

    let int_pow = graphs
        .iter()
        .find(|(name, _)| *name == "int_pow")
        .map(|(_, g)| g)
        .expect("int_pow lowers");
    let nomod_call = int_pow.blocks.iter().find(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if call_leaf(target).contains("int_pow_nomod")
            )
        })
    });
    if let Some(block) = nomod_call {
        assert!(
            matches!(block.exitswitch, Some(ExitSwitch::LastException)),
            "int_pow's int_pow_nomod call must be LastException (descr_pow try/except); \
             switch={:?} dump=\n{}",
            block.exitswitch,
            int_pow.dump()
        );
        let follow_call = |start: usize| -> String {
            let mut id = start;
            for _ in 0..int_pow.blocks.len() {
                let body = &int_pow.blocks[id];
                for op in &body.operations {
                    if let OpKind::Call { target, .. } = &op.kind {
                        return call_leaf(target);
                    }
                }
                match body.exits.as_slice() {
                    [link] if body.exitswitch.is_none() => id = link.target.0,
                    _ => break,
                }
            }
            format!("no-call-in-b{id}")
        };
        let exc = block
            .exits
            .iter()
            .find(|link| link.last_exception.is_some())
            .unwrap_or_else(|| panic!("nomod call has an exception exit: {}", int_pow.dump()));
        let exc_leaf = follow_call(exc.target.0);
        assert!(
            exc_leaf.contains("pow_ovf2long"),
            "exception edge must skip the Result match and reach pow_ovf2long, got {exc_leaf}: {}",
            int_pow.dump()
        );
        let ok = block
            .exits
            .iter()
            .find(|link| link.last_exception.is_none())
            .unwrap_or_else(|| panic!("nomod call has a normal exit: {}", int_pow.dump()));
        let ok_leaf = follow_call(ok.target.0);
        assert!(
            ok_leaf.contains("w_int_new"),
            "normal edge must skip the Result match and reach w_int_new, got {ok_leaf}: {}",
            int_pow.dump()
        );
    } else {
        panic!(
            "int_pow does not call int_pow_nomod; dump=\n{}",
            int_pow.dump()
        );
    }

    let trampoline_adapter = graphs
        .iter()
        .find(|(name, _)| *name == "__majit_call_target_int_pow_nomod_trampoline")
        .map(|(_, g)| g);
    if let Some(adapter) = trampoline_adapter {
        let ee = extraeffect_of(&cc, adapter, two_ints, Type::Int)
            .unwrap_or_else(|err| panic!("adapter getcalldescr: {err}"));
        assert!(
            ee > ExtraEffect::CannotRaise,
            "int_pow_nomod_trampoline adapter extraeffect must be CanRaise \
             (Result::Err is its raise channel), got {ee:?}"
        );
    }

    let mut int_pow_runtime = int_pow.clone();
    let hints = majit_translate::front::llbc_hints::harvest_exception_class_hints_from_llbcs(
        std::slice::from_ref(llbc),
    );
    majit_translate::codewriter::error_carrier_edges::lower_error_carrier_edges(
        &mut int_pow_runtime,
        &majit_translate::OwnedErrorCarrierSpec::own(ERROR_CARRIER),
        &|name| majit_translate::front::llbc_hints::exception_class_name_belongs(&hints, name),
    );
    let mut dump_cc = CallControl::new();
    dump_cc.set_struct_fields(with_shared_lowering(llbc, |ctx| {
        ctx.struct_field_registry()
    }));
    for (name, graph) in &graphs {
        if *name == "int_pow" {
            dump_cc.register_function_graph(
                path_from_graph_name(&int_pow_runtime.name),
                int_pow_runtime.clone(),
            );
        } else {
            dump_cc.register_function_graph(path_from_graph_name(&graph.name), graph.clone());
        }
    }
    match jitcode_dump_of(&mut dump_cc, &int_pow_runtime) {
        Ok(dump) => {
            eprintln!("=== int_pow jitcode ===\n{dump}");
            assert!(
                dump.contains("catch_exception"),
                "int_pow jitcode must catch the nomod overflow into pow_ovf2long: {dump}"
            );
            assert!(
                catch_arm_ref_returns_residual(&dump),
                "int_pow catch arm must residual-call the ovf2long recovery and ref_return \
                 (descr_pow except OverflowError: return _pow_ovf2long), got {dump}"
            );
        }
        Err(err) => panic!("int_pow jitcode FAIL: {err}"),
    }
    if let Some((_, wrapper)) = graphs.iter().find(|(name, _)| *name == "int_pow_nomod") {
        match jitcode_dump_of(&mut dump_cc, wrapper) {
            Ok(dump) => eprintln!("=== int_pow_nomod jitcode ===\n{dump}"),
            Err(err) => eprintln!("int_pow_nomod jitcode FAIL: {err}"),
        }
    }
    if let Some((_, trampoline)) = graphs
        .iter()
        .find(|(name, _)| *name == "int_pow_nomod_trampoline")
    {
        match jitcode_dump_of(&mut dump_cc, trampoline) {
            Ok(dump) => eprintln!("=== int_pow_nomod_trampoline jitcode ===\n{dump}"),
            Err(err) => eprintln!("int_pow_nomod_trampoline jitcode FAIL: {err}"),
        }
    }
}

/// `argument.py _match_signature` returns `ArgErr` or `OperationError`.
/// pyre keeps those as `MatchOutcome` (`parse_into_scope` catches `ArgErr`).
/// The carrier rewrite must not demand `From<MatchSignatureError> for PyError`
/// — that enum is the function's own error arm.
#[test]
fn match_signature_lowers_as_its_own_error() {
    lower_function(interp(), "match_signature")
        .unwrap_or_else(|err| panic!("match_signature must lower: {err}"));
}
