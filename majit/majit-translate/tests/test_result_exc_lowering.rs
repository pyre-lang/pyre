//! Result-of-PyError → exception-link lowering: production-LLBC
//! regression tests for `front::result_exc`.

mod common;

use common::{INTERPRETER_LLBC, MODULE_LLBC, load_llbc};
use majit_charon_reader::Llbc;
use majit_translate::codewriter::jtransform::{GraphTransformConfig, transform_graph};
use majit_translate::front::mir::{
    LowerContext, erased_root_bracket_guards, function_touches_root_stack,
    lower_fun_decl_with_static_addrs, lower_function_with_static_addrs, pin_roots_published_locals,
    shadow_stack_erase_status,
};
use majit_translate::model::{CallTarget, ExitCase, ExitSwitch, LinkArg, OpKind};
use majit_translate::{ErrorCarrierSpec, HostStaticAddrs};
use std::sync::{Mutex, MutexGuard};

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

fn interp_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lower_function_unlocked(
    llbc: &Llbc,
    function_name: &str,
) -> Result<majit_translate::model::FunctionGraph, majit_translate::front::mir::LowerError> {
    lower_function_with_static_addrs(
        llbc,
        function_name,
        HostStaticAddrs {
            error_carrier: ERROR_CARRIER,
            ..Default::default()
        },
    )
}

fn lower_function(
    llbc: &Llbc,
    function_name: &str,
) -> Result<majit_translate::model::FunctionGraph, majit_translate::front::mir::LowerError> {
    let _guard = interp_lock();
    lower_function_unlocked(llbc, function_name)
}

/// [`lower_function`], then the codewriter's conversion of the carrier's
/// exception edges into the runtime exception value
/// (`codewriter::error_carrier_edges`): the raise paths the JitCode holds.
fn lower_function_to_runtime_edges(
    llbc: &'static Llbc,
    function_name: &str,
) -> Result<majit_translate::model::FunctionGraph, majit_translate::front::mir::LowerError> {
    let _guard = interp_lock();
    let mut graph = lower_function_unlocked(llbc, function_name)?;
    majit_translate::codewriter::error_carrier_edges::lower_error_carrier_edges(
        &mut graph,
        &majit_translate::OwnedErrorCarrierSpec::own(ERROR_CARRIER),
    );
    Ok(graph)
}

fn interp() -> &'static Llbc {
    load_llbc(INTERPRETER_LLBC)
}

/// `Llbc::set_root_stack_effects` writes the shared OnceLock artefact.
/// Restore the empty table when the seeding test finishes, including on
/// panic, so a later test still sees the unpublished default.
struct ResetRootStackEffects<'a>(&'a Llbc);

impl Drop for ResetRootStackEffects<'_> {
    fn drop(&mut self) {
        self.0.set_root_stack_effects(Vec::new(), Vec::new());
    }
}

fn optional_module() -> &'static Llbc {
    load_llbc(MODULE_LLBC)
}

#[test]
fn pop_n_ok_payload_field_is_int() {
    let g =
        lower_function(interp(), "pyre_interpreter::shared_opcode::pop_n").expect("lower pop_n");
    let writes: Vec<(String, String, String)> = g
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite {
                field, ty, value, ..
            } if field.name == "__pos_0"
                && field
                    .owner_root
                    .as_deref()
                    .is_some_and(|o| o.contains("Result") && o.contains("Ok")) =>
            {
                Some((
                    field.owner_root.clone().unwrap_or_default(),
                    format!("{ty:?}"),
                    format!("{value:?}"),
                ))
            }
            _ => None,
        })
        .collect();
    for (owner, ty, value) in &writes {
        assert!(
            ty == "Int" || ty == "Unsigned",
            "pop_n {owner}.__pos_0 ty={ty} value={value}; \
             RustVecRepr.lowleveltype is raw Ptr, getkind int"
        );
    }
}

fn is_root_scope_close(op: &OpKind) -> bool {
    matches!(op,
        OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
            if segments.last().map(String::as_str) == Some("root_scope_close")
                || (segments.last().map(String::as_str) == Some("drop_in_place")
                    && segments.iter().any(|s| s == "RootScope")))
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
                    leaf.split_once('<').map_or(leaf.as_str(), |(base, _)| base) == "Result"
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
/// the `Result<*mut PyObject,PyError>::Ok.__pos_0` extract, not the
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
                            == Some("Result<*mut PyObject,PyError>::Ok") =>
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
        Some("core::option::Option<PyError>"),
        "the stub result is SomeInstance of the declared Option<PyError>"
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
        fn_consts.is_empty(),
        "`.map(Some)` leaves no function-item define: {fn_consts:?}"
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
        let Some(src) = single_inputarg_source(graph, &current) else {
            return None;
        };
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

/// `Lock.locked` is `*lock_state(&self.locked)`. `lock_state` is residual
/// and `MutexGuard::deref` returns `&bool`. That address is not the bool
/// `Rvalue::Ref` would have aliased, so the return is a `raw_load` and
/// `history.getkind` of the loaded word is `int`, matching `FUNC.RESULT`.
#[test]
fn lock_locked_returns_the_bool_word() {
    use majit_translate::model::{LinkArg, OpKind, ValueType};
    let path = "pyre_interpreter::module::thread::lock_class::<Impl>::locked";
    let g = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower: {e}"));
    let mut returns = 0usize;
    for block in &g.blocks {
        for link in &block.exits {
            if link.target != g.returnblock {
                continue;
            }
            assert_eq!(link.args.len(), 1, "bool return has one arg");
            let LinkArg::Value(var) = &link.args[0] else {
                panic!("return arg is a value");
            };
            match return_producer(&g, var) {
                Some(OpKind::RawLoad {
                    item_ty: ValueType::Int,
                    itemsize: 1,
                    ..
                }) => {}
                other => panic!("locked return must raw_load the bool, got {other:?}"),
            }
            returns += 1;
        }
    }
    assert_eq!(returns, 1, "{path}");
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
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "ok_or_else"
            )
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

/// `with_roots!(value, w_name => force(obj))` is a `pin_roots(&[value, w_name])`
/// run. Charon spells the Unsize as `Rvalue::Cast`; `RootBracketPlan` must
/// still answer the restore so the name local is not residual-swapped with
/// the value (`flowspace` `Constant` / `gctransform` `pin_src`).
#[test]
fn object_setattr_erases_with_roots_pin_run() {
    let path = "pyre_interpreter::baseobjspace::object_setattr";
    let llbc = interp();
    let fd = llbc
        .iter_fun_decls()
        .find(|fd| fd.item_meta.name_path() == path)
        .unwrap_or_else(|| panic!("{path} present in the shipped LLBC"));
    let body = fd
        .unstructured()
        .unwrap_or_else(|| panic!("{path} has an unstructured body"));
    // Single-artefact lowering has no linked table, so every foreign opaque
    // is charged.  The product driver publishes the dependency crates first
    // (`set_root_stack_effects`); seed the same shape so `force` is judged
    // from its own body, not from `core::ptr::is_null`.
    let _interp_guard = interp_lock();
    llbc.set_root_stack_effects(
        vec![
            "pyre_object".into(),
            "core".into(),
            "alloc".into(),
            "std".into(),
            "majit_rlib".into(),
        ],
        Vec::new(),
    );
    let _reset = ResetRootStackEffects(llbc);
    let erase_status = shadow_stack_erase_status(llbc, fd, &body);
    let force_path = "pyre_interpreter::module::_weakref::interp__weakref::force";
    let force_sensitive = llbc.is_stack_sensitive_fn(force_path);
    let erased = erased_root_bracket_guards(llbc, fd, &body);
    if !erased.contains(&7) {
        use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};
        let callee_of = |call: &majit_charon_reader::ullbc::CallPayload| -> String {
            match &call.func {
                CallFunc::Regular(reg) => match &reg.kind {
                    CallKind::Fun(FunId::Regular { id }) => llbc
                        .fn_by_id(*id)
                        .map(|f| f.item_meta.name_path())
                        .unwrap_or_else(|| format!("fun#{id}")),
                    other => format!("{other:?}"),
                },
                other => format!("{other:?}"),
            }
        };
        let mut dump = String::new();
        dump.push_str(&format!(
            "erase_status={erase_status:?} force_sensitive={force_sensitive} force_touches={:?} proxy_touches={:?} deref_touches={:?}\n",
            function_touches_root_stack(llbc, force_path),
            function_touches_root_stack(
                llbc,
                "pyre_interpreter::module::_weakref::interp__weakref::is_w_abstract_proxy",
            ),
            function_touches_root_stack(
                llbc,
                "pyre_interpreter::module::_weakref::interp__weakref::dereference",
            ),
        ));
        dump.push_str("--- object_setattr stack/root calls ---\n");
        for (bb_idx, bb) in body.body.iter().enumerate() {
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            let callee = callee_of(call);
            let leaf = callee.rsplit("::").next().unwrap_or("");
            if !(leaf == "push_roots"
                || leaf == "pin_roots"
                || leaf == "pin_root"
                || leaf == "get"
                || leaf == "force"
                || leaf == "is_true"
                || callee.contains("gc_roots"))
            {
                continue;
            }
            dump.push_str(&format!(
                "bb{bb_idx} CALL {callee} dest={:?} nargs={} sensitive={}\n",
                call.dest.kind,
                call.args.len(),
                llbc.is_stack_sensitive_fn(&callee)
            ));
        }
        if let Some(force_fd) = llbc
            .iter_fun_decls()
            .find(|fd| fd.item_meta.name_path() == force_path)
            && let Some(force_body) = force_fd.unstructured()
        {
            dump.push_str("--- force callees ---\n");
            for (bb_idx, bb) in force_body.body.iter().enumerate() {
                let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                    continue;
                };
                let callee = callee_of(call);
                let has_body = match &call.func {
                    CallFunc::Regular(reg) => match &reg.kind {
                        CallKind::Fun(FunId::Regular { id }) => {
                            llbc.fn_by_id(*id).and_then(|f| f.unstructured()).is_some()
                        }
                        _ => false,
                    },
                    _ => false,
                };
                dump.push_str(&format!(
                    "force bb{bb_idx} CALL {callee} nargs={} sensitive={} body={has_body} touches={:?}\n",
                    call.args.len(),
                    llbc.is_stack_sensitive_fn(&callee),
                    function_touches_root_stack(llbc, &callee)
                ));
            }
        }
        for helper in [
            "pyre_interpreter::module::_weakref::interp__weakref::is_w_abstract_proxy",
            "pyre_interpreter::module::_weakref::interp__weakref::dereference",
            "pyre_interpreter::module::_weakref::interp__weakref::weakref_obj_weak",
            "pyre_interpreter::typedef::type",
        ] {
            dump.push_str(&format!("--- {helper} callees ---\n"));
            if let Some(helper_fd) = llbc
                .iter_fun_decls()
                .find(|fd| fd.item_meta.name_path() == helper)
                && let Some(helper_body) = helper_fd.unstructured()
            {
                for (bb_idx, bb) in helper_body.body.iter().enumerate() {
                    let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                        continue;
                    };
                    let callee = callee_of(call);
                    let (has_body, is_local) = match &call.func {
                        CallFunc::Regular(reg) => match &reg.kind {
                            CallKind::Fun(FunId::Regular { id }) => llbc
                                .fn_by_id(*id)
                                .map(|f| (f.unstructured().is_some(), f.item_meta.is_local))
                                .unwrap_or((false, false)),
                            _ => (false, false),
                        },
                        _ => (false, false),
                    };
                    dump.push_str(&format!(
                        "  bb{bb_idx} CALL {callee} sensitive={} touches={:?} body={has_body} local={is_local}\n",
                        llbc.is_stack_sensitive_fn(&callee),
                        function_touches_root_stack(llbc, &callee),
                    ));
                }
            }
        }
        dump.push_str(&format!(
            "pin_slice_10={:?} pin_slice_8={:?} assigned_hint={:?}\n",
            pin_roots_published_locals(llbc, &body, 10),
            pin_roots_published_locals(llbc, &body, 8),
            pin_roots_published_locals(llbc, &body, 9),
        ));
        dump.push_str("--- bb0-bb10 terminators ---\n");
        for bb_idx in 0..11 {
            let Some(bb) = body.body.get(bb_idx) else {
                break;
            };
            match bb.term_ref(llbc) {
                Ok(TermKind::Call {
                    call,
                    target,
                    on_unwind,
                }) => {
                    dump.push_str(&format!(
                        "bb{bb_idx} CALL {} dest={:?} nargs={} target={target} unwind={on_unwind}\n",
                        callee_of(call),
                        call.dest.kind,
                        call.args.len(),
                    ));
                }
                Ok(TermKind::Drop {
                    place,
                    fn_ptr,
                    target,
                    on_unwind,
                }) => {
                    let drop_path = match &fn_ptr.kind {
                        CallKind::Fun(FunId::Regular { id }) => llbc
                            .fn_by_id(*id)
                            .map(|f| f.item_meta.name_path())
                            .unwrap_or_else(|| format!("fun#{id}")),
                        other => format!("{other:?}"),
                    };
                    dump.push_str(&format!(
                        "bb{bb_idx} DROP {drop_path} place={:?} touches={:?} target={target} unwind={on_unwind}\n",
                        place.kind,
                        function_touches_root_stack(llbc, &drop_path)
                    ));
                }
                Ok(other) => dump.push_str(&format!(
                    "bb{bb_idx} TERM {}\n",
                    format!("{other:?}").chars().take(180).collect::<String>()
                )),
                Err(_) => dump.push_str(&format!("bb{bb_idx} TERM-unparsed\n")),
            }
            for (si, stmt) in bb.statements.iter().enumerate() {
                match stmt.stmt_kind_ref() {
                    Ok(majit_charon_reader::ullbc::StmtKind::StorageLive(_))
                    | Ok(majit_charon_reader::ullbc::StmtKind::StorageDead(_))
                    | Ok(majit_charon_reader::ullbc::StmtKind::Borrowck(_)) => {}
                    Ok(majit_charon_reader::ullbc::StmtKind::Assign(place, _))
                        if matches!(
                            place.kind,
                            majit_charon_reader::ullbc::PlaceKind::Local(_)
                        ) => {}
                    other => dump.push_str(&format!("bb{bb_idx}.{si} NON-LOCAL {other:?}\n")),
                }
            }
        }
        panic!(
            "{path}: first with_roots (local 7) not erased; erased={erased:?}. MIR dump:\n{dump}"
        );
    }
    // Later brackets (`is_true` of `__abstractmethods__`, descriptor
    // lookup) may still residualize: those callees can pin. Lowering
    // the body must succeed once the first `with_roots!(value, w_name
    // => force(obj))` is answered from SSA.
    let graph = lower_function_unlocked(llbc, path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let mut promoted = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            let dump = match target {
                CallTarget::FunctionPath { segments, .. } => segments.join("::"),
                CallTarget::Method {
                    name,
                    resolved_path,
                    ..
                } => {
                    format!("{name} path={resolved_path:?}")
                }
                other => format!("{other:?}"),
            };
            if dump.contains("promoted_const") {
                promoted.push(dump);
            }
        }
    }
    if !promoted.is_empty() {
        use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, StmtKind, TermKind};
        let mut dump = format!("residual promoted_const: {}\n", promoted.join("\n"));
        dump.push_str("--- object_setattr calls to promoted_const ---\n");
        for (bb_idx, bb) in body.body.iter().enumerate() {
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            let callee = match &call.func {
                CallFunc::Regular(reg) => match &reg.kind {
                    CallKind::Fun(FunId::Regular { id }) => {
                        let path = llbc
                            .fn_by_id(*id)
                            .map(|f| f.item_meta.name_path())
                            .unwrap_or_else(|| format!("fun#{id}"));
                        let promoted = llbc.fn_by_id(*id).is_some_and(|f| {
                            f.item_meta.name.last().is_some_and(|seg| match seg {
                                majit_charon_reader::ullbc::NameSeg::Other(v) => {
                                    v.as_object()
                                        .and_then(|m| m.get("Builtin"))
                                        .and_then(serde_json::Value::as_array)
                                        .and_then(|arr| arr.first())
                                        .and_then(serde_json::Value::as_str)
                                        == Some("PromotedConst")
                                }
                                _ => false,
                            })
                        });
                        format!("{path} id={id} promoted={promoted} kind={:?}", reg.kind)
                    }
                    other => format!("kind={other:?}"),
                },
                other => format!("func={other:?}"),
            };
            if callee.contains("promoted_const") || callee.contains("PromotedConst") {
                dump.push_str(&format!("bb{bb_idx} {callee} nargs={}\n", call.args.len()));
            }
        }
        for fd in llbc.iter_fun_decls() {
            let p = fd.item_meta.name_path();
            if !(p.contains("object_setattr") && p.contains("promoted_const")) {
                continue;
            }
            dump.push_str(&format!("--- {p} ---\n"));
            let Some(body) = fd.unstructured() else {
                dump.push_str("no body\n");
                continue;
            };
            for (bb_idx, bb) in body.body.iter().enumerate() {
                for (si, stmt) in bb.statements.iter().enumerate() {
                    let kind = match stmt.stmt_kind_ref() {
                        Ok(
                            StmtKind::StorageLive(_)
                            | StmtKind::StorageDead(_)
                            | StmtKind::Borrowck(_),
                        ) => {
                            continue;
                        }
                        Ok(other) => format!("{other:?}"),
                        Err(_) => format!("unparsed {}", stmt.kind_value()),
                    };
                    dump.push_str(&format!(
                        "bb{bb_idx}.{si} {}\n",
                        kind.chars().take(400).collect::<String>()
                    ));
                }
                match bb.term_ref(llbc) {
                    Ok(TermKind::Call {
                        call,
                        target,
                        on_unwind,
                    }) => {
                        dump.push_str(&format!(
                            "bb{bb_idx} CALL nargs={} target={target} unwind={on_unwind} dest={:?}\n",
                            call.args.len(),
                            call.dest.kind
                        ));
                    }
                    Ok(other) => dump.push_str(&format!(
                        "bb{bb_idx} TERM {}\n",
                        format!("{other:?}").chars().take(300).collect::<String>()
                    )),
                    Err(_) => dump.push_str(&format!(
                        "bb{bb_idx} TERM-raw {}\n",
                        bb.terminator.kind_value()
                    )),
                }
            }
            dump.push_str(&format!(
                "term_kind_value={} const_meta={:?}\n",
                body.body[0].terminator.kind_value(),
                {
                    let mut found = None;
                    for stmt in &body.body[0].statements {
                        if let Ok(StmtKind::Assign(
                            _,
                            majit_charon_reader::ullbc::Rvalue::Use(
                                majit_charon_reader::ullbc::Operand::Const(v),
                                _,
                            ),
                        )) = stmt.stmt_kind_ref()
                        {
                            found = Some(format!(
                                "literal={:?} kind={:?}",
                                llbc.const_expr_literal(v)
                                    .map(|l| format!("{l}").chars().take(240).collect::<String>()),
                                llbc.const_expr_kind(v)
                                    .map(|k| format!("{k}").chars().take(240).collect::<String>()),
                            ));
                        }
                    }
                    found
                }
            ));
        }
        panic!("{path}: {dump}");
    }
    let mut method_pin_roots = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let dump = match target {
                CallTarget::FunctionPath { segments, .. } => segments.join("::"),
                CallTarget::Method {
                    name,
                    resolved_path,
                    ..
                } => format!("{name} path={resolved_path:?}"),
                other => format!("{other:?}"),
            };
            if dump.contains("RootScope") && dump.contains("pin_roots") {
                method_pin_roots.push(format!("{dump} nargs={}", args.len()));
            }
        }
    }
    assert!(
        method_pin_roots.is_empty(),
        "{path}: method pin_roots must retarget to free pin_roots:\n{}",
        method_pin_roots.join("\n")
    );
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let CallTarget::FunctionPath { segments, .. } = target else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("pin_roots") {
                continue;
            }
            assert_eq!(
                segments.as_slice(),
                ["pyre_object", "gc_roots", "pin_roots"],
                "{path}: leftover pin_roots must be the free helper: {}",
                segments.join("::")
            );
            assert_eq!(
                args.len(),
                1,
                "{path}: free pin_roots is one GcArray word, got {}",
                args.len()
            );
        }
    }
    let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    let mut array_news = Vec::new();
    for block in &transformed.blocks {
        for op in &block.operations {
            if let OpKind::NewArrayClear { array_type_id, .. } = &op.kind {
                array_news.push(array_type_id.clone());
            }
        }
    }
    assert!(
        array_news.iter().any(|id| {
            id.as_deref() == Some(majit_translate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID)
        }),
        "{path}: with_roots array must become NewArrayClear of object GcArray, got {array_news:?}"
    );
}

/// `get_and_call_function` publishes then normalizes. Method
/// `RootScope::publish` is two Ref words; the bound helper is the 3-word
/// pair ABI. The front retargets it to free `publish_roots` over the
/// length-prefixed GcArray (`rlist.py` `GcArray(OBJECTPTR)`), the same
/// split `pin_roots` already takes.
#[test]
fn get_and_call_function_publish_retargets_to_publish_roots() {
    let path = "pyre_interpreter::baseobjspace::get_and_call_function";
    let llbc = interp();
    let _interp_guard = interp_lock();
    llbc.set_root_stack_effects(
        vec![
            "pyre_object".into(),
            "core".into(),
            "alloc".into(),
            "std".into(),
            "majit_rlib".into(),
        ],
        Vec::new(),
    );
    let _reset = ResetRootStackEffects(llbc);
    let graph = lower_function_unlocked(llbc, path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let mut method_publish = Vec::new();
    let mut free_publish = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let dump = match target {
                CallTarget::FunctionPath { segments, .. } => segments.join("::"),
                CallTarget::Method {
                    name,
                    resolved_path,
                    ..
                } => format!("{name} path={resolved_path:?}"),
                other => format!("{other:?}"),
            };
            if dump.contains("RootScope") && dump.contains("publish") {
                method_publish.push(format!("{dump} nargs={}", args.len()));
            }
            if let CallTarget::FunctionPath { segments, .. } = target
                && segments.last().map(String::as_str) == Some("publish_roots")
            {
                free_publish.push(args.len());
                assert_eq!(
                    segments.as_slice(),
                    ["pyre_object", "gc_roots", "publish_roots"],
                    "{path}: leftover publish_roots must be the free helper: {}",
                    segments.join("::")
                );
                assert_eq!(
                    args.len(),
                    1,
                    "{path}: free publish_roots is one GcArray word, got {}",
                    args.len()
                );
            }
        }
    }
    assert!(
        method_publish.is_empty(),
        "{path}: method publish must retarget to free publish_roots:\n{}",
        method_publish.join("\n")
    );
    assert!(
        !free_publish.is_empty(),
        "{path}: expected a free publish_roots call after retarget"
    );
}

/// `call_explicit_args` / `call_kw` / `call_valuestack` do `args.reverse()`
/// on `Vec<PyObjectRef>`. rustc inlines that to `<[T]>::reverse`; the
/// FunDecl is opaque, so the front must retarget it to `ll_vec_reverse_*`
/// (`rlist.py ll_reverse`).
#[test]
fn call_explicit_args_reverse_is_ll_vec_reverse() {
    let path = "pyre_interpreter::eval::<Impl>::call_explicit_args";
    check_vec_reverse_helper(path);
    check_vec_reverse_helper("pyre_interpreter::eval::<Impl>::call_kw");
    check_vec_reverse_helper("pyre_interpreter::baseobjspace::call_valuestack");
}

/// `argument_factory` does `args.extend_from_slice(_arguments)` on
/// `&[PyObjectRef]`. The slice is a GcArray, not a pair; expand to
/// `(items, length)` and retarget to `ll_vec_extend_from_slice_r`
/// (`rlist.py ll_extend` / `rrustvec.rs` `VecOp::ExtendFromSlice`).
#[test]
fn argument_factory_extend_from_slice_is_ll_vec_extend() {
    let path = "pyre_interpreter::pyframe::<Impl>::argument_factory";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let mut leftover = Vec::new();
    let mut helpers = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            match target {
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().map(String::as_str) == Some("extend_from_slice") =>
                {
                    leftover.push(format!("{} nargs={}", segments.join("::"), args.len()));
                }
                CallTarget::FunctionPath { segments, .. }
                    if segments
                        .last()
                        .is_some_and(|leaf| leaf.starts_with("ll_vec_extend_from_slice")) =>
                {
                    helpers.push((segments.join("::"), args.len()));
                }
                _ => {}
            }
        }
    }
    assert!(
        leftover.is_empty(),
        "{path}: extend_from_slice must retarget to ll_vec_extend_from_slice_*:\n{}",
        leftover.join("\n")
    );
    assert!(
        helpers
            .iter()
            .any(|(name, n)| name.ends_with("ll_vec_extend_from_slice_r") && *n == 3),
        "{path}: expected 3-arg ll_vec_extend_from_slice_r, got {helpers:?}"
    );
}

fn check_vec_reverse_helper(path: &str) {
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let mut calls = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            calls.push(match target {
                CallTarget::FunctionPath {
                    segments,
                    fun_decl_id,
                    ..
                } => format!(
                    "FunctionPath {} nargs={} fun_decl_id={fun_decl_id:?}",
                    segments.join("::"),
                    args.len()
                ),
                CallTarget::Method {
                    name,
                    receiver_root,
                    resolved_path,
                    fun_decl_id,
                    ..
                } => format!(
                    "Method {name} root={receiver_root:?} path={resolved_path:?} nargs={} fun_decl_id={fun_decl_id:?}",
                    args.len()
                ),
                other => format!("{other:?} nargs={}", args.len()),
            });
        }
    }
    let dump = calls.join("\n");
    assert!(
        !dump.contains("core::slice::<Impl>::reverse"),
        "{path}: slice reverse must not residualize:\n{dump}"
    );
    assert!(
        dump.contains("ll_vec_reverse"),
        "{path}: expected ll_vec_reverse helper:\n{dump}"
    );
}

/// `_flat_pycall` passes `&[]` into `try_new_for_call_with_closure_and_globals_obj`.
/// That slice is the length-prefixed object GcArray (`rlist.py` `GcArray(OBJECTPTR)`);
/// `do_fixed_newlist_clear` allocates it with `new_array_clear(0)`.
#[test]
fn flat_pycall_empty_slice_is_a_cleared_object_gcarray() {
    let path = "pyre_interpreter::function::_flat_pycall";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    assert!(
        transformed
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::NewArrayClear {
                        array_type_id: Some(id),
                        ..
                    } if id.as_str() == majit_translate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID
                )
            }),
        "{path}: `&[]` must be new_array_clear(0) of the object GcArray"
    );
}

/// `readinto_impl` returns `Result<i64, PyError>`. `Ok(0)` is `CONST_0`
/// (`history.py`); coercing that payload to `ConstRefNull` made the CFG
/// return kind `r` against `FUNC.RESULT=i`.
#[test]
fn readinto_impl_ok_zero_stays_int() {
    let path = "buffered_random::<Impl>::readinto_impl";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    let mut null_returns = 0usize;
    for block in &transformed.blocks {
        for link in &block.exits {
            if link.target != transformed.returnblock {
                continue;
            }
            for arg in &link.args {
                let Some(var) = arg.as_variable() else {
                    continue;
                };
                if transformed
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .any(|op| {
                        op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstRefNull)
                    })
                {
                    null_returns += 1;
                }
            }
        }
    }
    assert_eq!(
        null_returns, 0,
        "{path}: Ok(0) must not return ConstRefNull"
    );
}

/// `classify_callable` returns `Result<CallableKind, PyError>`.
/// `CallableKind` is a fieldless enum (`Int`); `Ok(CallableKind::Builtin)`
/// is discriminant 0, `CONST_0`, not `CONST_NULL`.
#[test]
fn classify_callable_ok_kind_stays_int() {
    let path = "pyre_interpreter::runtime_ops::classify_callable";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    let mut null_returns = 0usize;
    for block in &transformed.blocks {
        for link in &block.exits {
            if link.target != transformed.returnblock {
                continue;
            }
            for arg in &link.args {
                let Some(var) = arg.as_variable() else {
                    continue;
                };
                if transformed
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .any(|op| {
                        op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstRefNull)
                    })
                {
                    null_returns += 1;
                }
            }
        }
    }
    assert_eq!(
        null_returns, 0,
        "{path}: Ok(CallableKind) must not return ConstRefNull"
    );
}

/// `prepare_frame_resume` reads `PyFrame.w_yielding_from` (a GCREF) and
/// stores `PY_NULL`. A null Ref is `ConstPtr(NULL)` (`history.py`
/// `CONST_NULL`); an Int 0 in that slot is what `make_equal_to` later
/// equates a `GetfieldGcR` to.
#[test]
fn yielding_from_null_store_is_const_ptr_null() {
    let path = "pyre_interpreter::eval::prepare_frame_resume";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    let writes: Vec<_> = transformed
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { field, .. } if field.name == "w_yielding_from"
            )
        })
        .map(|op| &op.kind)
        .collect();
    assert!(
        !writes.is_empty(),
        "{path}: expected a w_yielding_from store: {writes:?}"
    );
    for kind in &writes {
        let OpKind::FieldWrite { value, ty, .. } = kind else {
            continue;
        };
        assert!(
            matches!(ty, majit_translate::model::ValueType::Ref(_)),
            "{path}: w_yielding_from store kind must be Ref: {kind:?}"
        );
        assert!(
            !majit_translate::model::link_arg_is_int_zero(&transformed, value),
            "{path}: null Ref store must not keep Int 0: {kind:?}"
        );
    }
}

/// `RustVecRepr.lowleveltype` is a raw `Ptr` (`history.py` `getkind` `'int'`)
/// for every `Vec<T>`. `Result<Vec<u8>, PyError>::Ok.__pos_0` must use
/// that bank; a field left `Ref` while the header address is `Int` is
/// the `rewrite_op_setfield` kind mismatch in `read_all`.
#[test]
fn vec_u8_ok_payload_field_is_int() {
    let path = "pyre_interpreter::module::_io::interp_textio::<Impl>::read_all";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let writes: Vec<(String, String, String)> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite {
                field, ty, value, ..
            } if field.name == "__pos_0"
                && field.owner_root.as_deref().is_some_and(|o| {
                    o.contains("Result") && o.contains("Ok") && o.contains("Vec<u8>")
                }) =>
            {
                Some((
                    field.owner_root.clone().unwrap_or_default(),
                    format!("{ty:?}"),
                    format!("{value:?}"),
                ))
            }
            _ => None,
        })
        .collect();
    assert!(
        !writes.is_empty(),
        "{path}: expected Result<Vec<u8>>::Ok.__pos_0 FieldWrite"
    );
    for (owner, ty, value) in &writes {
        assert!(
            ty == "Int" || ty == "Unsigned",
            "{path} {owner}.__pos_0 ty={ty} value={value}; \
             RustVecRepr.lowleveltype is raw Ptr, getkind int"
        );
    }
    let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
}

/// `fileio_writebuf` returns `Result<(&mut [u8], PyObjectRef, bool), …>`.
/// `Tuple<[u8], …>::__pos_0` is the slice, whose `getkind` is `ref`.
/// Aliasing `Vec<u8>::as_mut_slice` to the Vec header (`RustVecRepr`,
/// kind `int`) stored an Int into that Ref field
/// (`jtransform.py` `rewrite_op_setfield`).
#[test]
fn fileio_writebuf_u8_slice_tuple_field_is_not_int_into_ref() {
    use majit_translate::model::{ConcreteType, FunctionGraph, LinkArg, ValueType};
    let path = "pyre_interpreter::builtins::fileio_writebuf";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let mut writes = 0usize;
    for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
        let OpKind::FieldWrite {
            field, ty, value, ..
        } = &op.kind
        else {
            continue;
        };
        if field.name != "__pos_0" {
            continue;
        }
        let Some(owner) = field.owner_root.as_deref() else {
            continue;
        };
        if !(owner.contains("Tuple") && owner.contains("[u8]")) {
            continue;
        }
        writes += 1;
        let value_is_int = match value {
            LinkArg::Value(var) => FunctionGraph::concretetype_of(var) == ConcreteType::Signed,
            _ => false,
        };
        assert!(
            !matches!(ty, ValueType::Ref(_) | ValueType::Str) || !value_is_int,
            "{path} {owner}.__pos_0 ty={ty:?} value={value:?}; \
             slice getkind is ref, Vec header getkind is int"
        );
    }
    assert!(
        writes >= 1,
        "{path}: expected Tuple<[u8],…>::__pos_0 FieldWrite"
    );
    let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
}

/// `_orig_parse_format` pushes `(Fmt, usize)` units. `Fmt` stores the
/// payload enum `Code`, so it is a class instance (`getkind` ref), not
/// a raw `Ptr`. `Tuple<Fmt,usize>::__pos_0` must use that bank;
/// `getfield_raw_r` is refused (`jtransform.py` `rewrite_op_getfield`).
#[test]
fn orig_parse_format_fmt_tuple_field_is_not_int_into_ref() {
    let path = "pyre_interpreter::module::struct::_orig_parse_format";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let writes: Vec<(String, String, String)> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::FieldWrite {
                field, ty, value, ..
            } if field.name == "__pos_0"
                && field
                    .owner_root
                    .as_deref()
                    .is_some_and(|o| o.contains("Tuple") && o.contains("Fmt")) =>
            {
                Some((
                    field.owner_root.clone().unwrap_or_default(),
                    format!("{ty:?}"),
                    format!("{value:?}"),
                ))
            }
            _ => None,
        })
        .collect();
    assert!(
        !writes.is_empty(),
        "{path}: expected Tuple<Fmt,usize>::__pos_0 FieldWrite"
    );
    for (owner, ty, value) in &writes {
        assert!(
            ty.starts_with("Ref"),
            "{path} {owner}.__pos_0 ty={ty} value={value}; \
             payload-enum field makes the struct a class instance"
        );
    }
    let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
}

fn call_targets_push_roots(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.iter().any(|s| s == "push_roots") && segments.iter().any(|s| s == "gc_roots")
        }
        _ => false,
    }
}

/// `push_roots` returns `RootScope`. Dependents see that type as Opaque
/// (`Ref`); `pin_root` / `publish` dispatch on the instance. FUNC.RESULT
/// and the call dest must share that bank (`call.py` `RESULT ==
/// FUNC.RESULT`). A raw-struct Int token is the `getcalldescr` wall.
#[test]
fn push_roots_result_is_ref_not_int() {
    use majit_translate::model::ValueType;
    let callers = [
        "pyre_interpreter::call::call_function_impl_result",
        "pyre_interpreter::call::call_function_impl",
        "pyre_interpreter::call::call_function_impl_raw",
    ];
    let mut calls = 0usize;
    for path in callers {
        let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
        for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
            let OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            if !call_targets_push_roots(target) {
                continue;
            }
            calls += 1;
            assert!(
                matches!(result_ty, ValueType::Ref(_)),
                "{path} push_roots result_ty={result_ty:?}; \
                 RootScope is the root-bracket instance, getkind ref"
            );
        }
        let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    }
    let callee = "pyre_object::gc_roots::push_roots";
    if let Ok(graph) = lower_function(interp(), callee) {
        let token = graph.return_type.as_deref().unwrap_or("");
        assert!(
            !(token == "i64"
                || token == "u64"
                || token == "isize"
                || token == "usize"
                || token.starts_with("raw:")),
            "{callee} return_type={token:?}; RootScope FUNC.RESULT is ref"
        );
        calls += 1;
        let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    }
    assert!(
        calls >= 1,
        "expected a push_roots call dest or the push_roots graph"
    );
}

fn call_targets_root_scope_new(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().is_some_and(|s| s == "new") && segments.iter().any(|s| s == "gc_roots")
        }
        _ => false,
    }
}

/// `RootScope::new` constructs the instance (`CFG r`). FUNC.RESULT must
/// share that bank (`call.py` `get_jitcode_calldescr`; `rtyper` `getrepr`
/// → `lowleveltype`; `getkind`). Peeling the guard through `PhantomData`
/// to `usize` stamps `i`.
#[test]
fn gc_roots_new_result_is_ref_not_int() {
    use majit_translate::model::ValueType;
    let callee = "pyre_object::gc_roots::<Impl>::new";
    let mut graphs = 0usize;
    if let Ok(graph) = lower_function(interp(), callee) {
        assert_root_scope_new_func_result(callee, &graph);
        graphs += 1;
        let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    }
    const OBJECT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-object.ullbc",
    );
    if std::path::Path::new(OBJECT).is_file() {
        let object = Llbc::load(OBJECT).expect("load pyre-object.ullbc");
        let graph =
            lower_function(&object, callee).unwrap_or_else(|e| panic!("lower {callee}: {e}"));
        assert_root_scope_new_func_result(callee, &graph);
        graphs += 1;
        let _transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    }
    let callers = [
        "pyre_object::gc_roots::push_roots",
        "pyre_interpreter::call::call_function_impl_result",
    ];
    for path in callers {
        let Ok(graph) = lower_function(interp(), path) else {
            continue;
        };
        for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
            let OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            if !call_targets_root_scope_new(target) {
                continue;
            }
            graphs += 1;
            assert!(
                matches!(result_ty, ValueType::Ref(_)),
                "{path} RootScope::new result_ty={result_ty:?}; \
                 RootScope is the root-bracket instance, getkind ref"
            );
        }
    }
    assert!(
        graphs >= 1,
        "expected RootScope::new FUNC.RESULT, a CFG graph, or a call dest"
    );
}

fn assert_root_scope_new_func_result(callee: &str, graph: &majit_translate::model::FunctionGraph) {
    let token = graph.return_type.as_deref().unwrap_or("");
    assert!(
        !(token == "i64"
            || token == "u64"
            || token == "isize"
            || token == "usize"
            || token.starts_with("raw:")),
        "{callee} return_type={token:?}; RootScope FUNC.RESULT is ref"
    );
}

/// A raw struct field read with a Ref result is `getfield_raw_r`, which
/// `jtransform.py rewrite_op_getfield` refuses unless the field is
/// immutable. `index_c_int_w::<Impl>::call_once` captures a callable
/// in a closure env; that owner is a class instance (`src: Closure`),
/// so the load is `getfield_gc_r`.
#[test]
fn index_c_int_w_call_once_has_no_mutable_getfield_raw_r() {
    use majit_translate::model::{ConcreteType, FunctionGraph, ValueType};
    let llbc = interp();
    let context = LowerContext::new(llbc);
    let static_addrs = HostStaticAddrs {
        error_carrier: ERROR_CARRIER,
        ..Default::default()
    };
    let mut raw_ref = Vec::new();
    let mut call_once = 0usize;
    for fd in llbc.iter_fun_decls() {
        let path = fd.item_meta.name_path();
        if !path.contains("index_c_int_w") || !path.ends_with("call_once") {
            continue;
        }
        let graph = lower_fun_decl_with_static_addrs(&context, fd, static_addrs)
            .unwrap_or_else(|e| panic!("lower {path}: {e}"));
        call_once += 1;
        let mut check = |label: &str, graph: &FunctionGraph| {
            for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
                let OpKind::FieldRead {
                    base,
                    field,
                    ty,
                    pure,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                let base_ct = FunctionGraph::concretetype_of(base);
                let result_ct = op.result.as_ref().map(FunctionGraph::concretetype_of);
                let result_is_ref = matches!(ty, ValueType::Ref(_) | ValueType::Str)
                    || result_ct == Some(ConcreteType::GcRef);
                let owner_raw =
                    field.owner_declared_gc == Some(false) || base_ct == ConcreteType::Signed;
                if owner_raw && result_is_ref && !*pure {
                    raw_ref.push(format!(
                        "{label} {path} {}.{} ty={ty:?} declared_gc={:?} base_ct={base_ct:?}",
                        field.owner_root.as_deref().unwrap_or("?"),
                        field.name,
                        field.owner_declared_gc,
                    ));
                }
            }
        };
        check("lower", &graph);
        let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
        check("xform", &transformed);
    }
    assert!(
        call_once >= 1,
        "expected index_c_int_w::<Impl>::call_once in interpreter LLBC"
    );
    assert!(raw_ref.is_empty(), "mutable getfield_raw_r: {raw_ref:?}");
}

/// `PositionCookie::unpack` returns `Result<Self, PyError>`. After
/// result_exc the success type is the cookie, a by-value raw struct
/// (`history.py` `getkind` of a raw `Ptr` is `'int'`). The CFG
/// constructor must use that bank so `func_result_kind` sees the same
/// colour as FUNC.RESULT (`call.py` `get_jitcode_calldescr`).
#[test]
fn unpack_raw_struct_func_result_matches_cfg() {
    use majit_translate::model::{ConcreteType, FunctionGraph};
    let llbc = interp();
    let context = LowerContext::new(llbc);
    let static_addrs = HostStaticAddrs {
        error_carrier: ERROR_CARRIER,
        ..Default::default()
    };
    let mut seen = 0usize;
    for fd in llbc.iter_fun_decls() {
        let path = fd.item_meta.name_path();
        if !path.contains("interp_textio") || !path.ends_with("unpack") {
            continue;
        }
        let graph = lower_fun_decl_with_static_addrs(&context, fd, static_addrs)
            .unwrap_or_else(|e| panic!("lower {path}: {e}"));
        seen += 1;
        let check = |label: &str, graph: &FunctionGraph| {
            let cfg = match graph.block(graph.returnblock).inputargs.first() {
                None => 'v',
                Some(arg) => match FunctionGraph::concretetype_of(arg) {
                    ConcreteType::Signed => 'i',
                    ConcreteType::GcRef => 'r',
                    ConcreteType::Float => 'f',
                    ConcreteType::Void => 'v',
                    ConcreteType::Unknown => panic!("{label} {path}: returnblock unresolved"),
                },
            };
            assert_eq!(
                cfg, 'i',
                "{label} {path}: raw struct CFG return is int, got {cfg}"
            );
            if let Some(token) = graph.return_type.as_deref() {
                assert!(
                    token == "i64" || token == "u64",
                    "{label} {path}: FUNC.RESULT={token:?}; raw struct getkind is int"
                );
            }
        };
        check("lower", &graph);
        let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
        check("xform", &transformed);
    }
    assert!(
        seen >= 1,
        "expected interp_textio::<Impl>::unpack in interpreter LLBC"
    );
}

/// `unpack_units` matches `fmt.code`. `Code` is a payload enum (`Ref`);
/// storing it makes `Fmt` a class instance so the read is
/// `getfield_gc_r`. A raw `Fmt` would be `getfield_raw_r`
/// (`jtransform.py` `rewrite_op_getfield`).
#[test]
fn unpack_units_has_no_mutable_getfield_raw_r() {
    use majit_translate::model::{ConcreteType, FunctionGraph, ValueType};
    let path = "pyre_interpreter::module::struct::unpack_units";
    let graph = lower_function(interp(), path).unwrap_or_else(|e| panic!("lower {path}: {e}"));
    let check = |label: &str, graph: &FunctionGraph| {
        let mut raw_ref = Vec::new();
        for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
            let OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
                ..
            } = &op.kind
            else {
                continue;
            };
            let base_ct = FunctionGraph::concretetype_of(base);
            let result_ct = op.result.as_ref().map(FunctionGraph::concretetype_of);
            let result_is_ref = matches!(ty, ValueType::Ref(_) | ValueType::Str)
                || result_ct == Some(ConcreteType::GcRef);
            let owner_raw =
                field.owner_declared_gc == Some(false) || base_ct == ConcreteType::Signed;
            if owner_raw && result_is_ref && !*pure {
                raw_ref.push(format!(
                    "{label} {path} {}.{} ty={ty:?} declared_gc={:?} base_ct={base_ct:?}",
                    field.owner_root.as_deref().unwrap_or("?"),
                    field.name,
                    field.owner_declared_gc,
                ));
            }
        }
        assert!(raw_ref.is_empty(), "mutable getfield_raw_r: {raw_ref:?}");
    };
    check("lower", &graph);
    let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
    check("xform", &transformed);
}

/// `stage_ioctl` builds `scoped_alloc_buffer` and returns
/// `Result<(i32, Vec<u8>, bool), PyError>`. A Ref store into a raw
/// (Int) owner is `setfield_raw_r`, which `jtransform.py`
/// `rewrite_op_setfield` refuses.
#[test]
fn stage_ioctl_has_no_setfield_raw_r() {
    use majit_translate::model::{ConcreteType, FunctionGraph, ValueType};
    let llbc = optional_module();
    let context = LowerContext::new(llbc);
    let static_addrs = HostStaticAddrs {
        error_carrier: ERROR_CARRIER,
        ..Default::default()
    };
    let mut raw_ref = Vec::new();
    let mut seen = 0usize;
    for fd in llbc.iter_fun_decls() {
        let path = fd.item_meta.name_path();
        if !path.contains("interp_fcntl") || !path.ends_with("stage_ioctl") {
            continue;
        }
        let graph = lower_fun_decl_with_static_addrs(&context, fd, static_addrs)
            .unwrap_or_else(|e| panic!("lower {path}: {e}"));
        seen += 1;
        let mut check = |label: &str, graph: &FunctionGraph| {
            for op in graph.blocks.iter().flat_map(|b| b.operations.iter()) {
                let OpKind::FieldWrite {
                    base,
                    field,
                    ty,
                    value,
                } = &op.kind
                else {
                    continue;
                };
                let base_ct = FunctionGraph::concretetype_of(base);
                let value_ct = match value {
                    LinkArg::Value(var) => format!("{:?}", FunctionGraph::concretetype_of(var)),
                    LinkArg::Const(_) => "const".to_string(),
                };
                let value_is_ref = match value {
                    LinkArg::Value(var) => {
                        FunctionGraph::concretetype_of(var) == ConcreteType::GcRef
                    }
                    LinkArg::Const(_) => matches!(ty, ValueType::Ref(_) | ValueType::Str),
                };
                let ty_is_ref = matches!(ty, ValueType::Ref(_) | ValueType::Str);
                let owner_raw =
                    field.owner_declared_gc == Some(false) || base_ct == ConcreteType::Signed;
                if owner_raw && (value_is_ref || ty_is_ref) {
                    raw_ref.push(format!(
                        "{label} {path} {}.{} ty={ty:?} declared_gc={:?} base_ct={base_ct:?} value={value_ct}",
                        field.owner_root.as_deref().unwrap_or("?"),
                        field.name,
                        field.owner_declared_gc,
                    ));
                }
            }
        };
        check("lower", &graph);
        let transformed = transform_graph(&graph, &GraphTransformConfig::default()).graph;
        check("xform", &transformed);
    }
    assert!(
        seen >= 1,
        "expected interp_fcntl::stage_ioctl in pyre-module LLBC"
    );
    assert!(raw_ref.is_empty(), "setfield_raw_r: {raw_ref:?}");
}
