//! Result-of-PyError → exception-link lowering: production-LLBC
//! regression tests for `front::result_exc`.

mod common;

use common::{INTERPRETER_LLBC, MODULE_LLBC, load_llbc, lower_named_with_static_addrs};
use majit_charon_reader::Llbc;
use majit_translate::model::{CallTarget, ExitCase, ExitSwitch, LinkArg, OpKind};
use majit_translate::{ErrorCarrierSpec, HostStaticAddrs};

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
    majit_translate::codewriter::error_carrier_edges::lower_error_carrier_edges(
        &mut graph,
        &majit_translate::OwnedErrorCarrierSpec::own(ERROR_CARRIER),
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
            "{path}: the `*mut BytesBlock` result is the `SomeString` a stub returns"
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
                || !segments.iter().any(|seg| seg.starts_with("<Impl#"))
            {
                continue;
            }
            assert!(
                from_args.is_empty(),
                "BytecodeCorruption is a void zero-sized type; From::from has no FUNC.ARGS slot"
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
