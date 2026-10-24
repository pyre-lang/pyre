//! Phase D acceptance anchor (lucky-growing-puzzle).
//!
//! Verifies that `CallControl::find_all_graphs` + `DefaultJitPolicy`
//! behave as RPython specifies when fed the pyre-interpreter source:
//!
//! 1. portal graph is in candidates (call.py
//!    `candidate_graphs = set(todo)`).
//! 2. Graphs reachable via `direct_call` from the portal — including
//!    every `opcode_*` helper — become candidates (call.py BFS).
//! 3. Graphs registered as `builtin_targets` are **not** followed —
//!    they stay at the residual-call boundary (call.py
//!    `getattr(targetgraph.func, 'oopspec')` → `builtin`, skipped in
//!    call.py `if kind != "regular": continue`).
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/call.py find_all_graphs`
//! - `rpython/jit/codewriter/call.py guess_call_kind`
//! - `rpython/jit/codewriter/policy.py look_inside_graph`
//!
//! The test constructs a **minimal synthetic** dispatch graph (no
//! pyre-interpreter source dependency) so the parity claim is isolated
//! to BFS + policy interaction. The broader `test_find_all_graphs`
//! already covers the integration path on the real handler corpus.

use majit_translate::{
    CallPath, call::CallControl, model::FunctionGraph, policy::DefaultJitPolicy,
};

/// Build a FunctionGraph with a single `direct_call` to `callee_path`
/// and no return value. This is the minimum the BFS needs to follow
/// an edge.
fn build_caller_graph(name: &str, callee_path: &CallPath) -> FunctionGraph {
    use majit_translate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

    let mut graph = FunctionGraph::new(name);
    let result_var = graph.alloc_value_var();
    graph
        .block_mut(graph.startblock)
        .operations
        .push(SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: callee_path.segments.clone(),
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: Vec::new(),
                result_ty: ValueType::Int,
            },
        });
    graph.set_return(graph.startblock, Some(result_var));
    graph
}

#[test]
fn find_all_graphs_follows_regular_edges_from_portal() {
    // RPython call.py:82 `if kind != "regular": continue` — a regular
    // direct_call edge whose callee has a registered graph must be
    // added to the candidate set.
    let portal_path = CallPath::from_segments(["portal"]);
    let callee_path = CallPath::from_segments(["callee"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &callee_path),
    );
    cc.register_function_graph(callee_path.clone(), FunctionGraph::new("callee"));
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&portal_path),
        "RPython call.py:65 portal must land in candidate_graphs"
    );
    assert!(
        cc.is_candidate(&callee_path),
        "RPython call.py:86-87 BFS must follow regular direct_call edges"
    );
}

#[test]
fn find_all_graphs_does_not_follow_builtin_targets() {
    // RPython call.py:132-133 `if hasattr(targetgraph.func, 'oopspec'):
    //   return 'builtin'` — and call.py:82 `if kind != "regular":
    //   continue`. A registered builtin target must stay at the
    //   residual-call boundary: reachable via the edge, but not added
    //   to the inline closure.
    let portal_path = CallPath::from_segments(["portal"]);
    let builtin_path = CallPath::from_segments(["ll_builtin"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &builtin_path),
    );
    // Register the builtin's graph so the edge has somewhere to land,
    // but give it an oopspec so it classifies as a builtin (call.py:135
    // `hasattr(targetgraph.func, 'oopspec')`) and BFS skips the regular
    // classification.
    cc.register_function_graph(builtin_path.clone(), FunctionGraph::new("ll_builtin"));
    cc.mark_oopspec(builtin_path.clone(), "ll_builtin(x)".to_string());
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&portal_path),
        "portal must always land in candidates"
    );
    assert!(
        !cc.is_candidate(&builtin_path),
        "RPython call.py:82 `kind != regular` — builtin targets stay \
         residual and must NOT be pulled into the candidate closure"
    );
}

#[test]
fn find_all_graphs_does_not_follow_portal_recursive_edges() {
    // RPython call.py `jitdriver_sd_from_portal_runner_ptr` —
    // a call to the portal itself classifies as `recursive`, which
    // call.py skips. The portal remains a candidate (via the seed)
    // but is not re-visited.
    let portal_path = CallPath::from_segments(["portal"]);

    let mut cc = CallControl::new();
    // Portal graph contains a self-call.
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &portal_path),
    );
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    // Reaching this assertion means BFS didn't infinite-loop on the
    // self-edge. The candidate set contains the portal by the seed
    // step, regardless of the recursive edge.
    assert!(cc.is_candidate(&portal_path));
}

/// Same as `build_caller_graph` but the single call is an `indirect_call`
/// site — a `dyn Trait` method call, before `rpbc::lower_indirect_calls`
/// rewrites it into `VtableMethodPtr` + `OpKind::IndirectCall`.  That
/// lowering runs inside `transform_graph_to_jitcode`, i.e. after
/// `find_all_graphs`, so this is the shape the BFS sees.
fn build_indirect_caller_graph(name: &str, trait_root: &str, method_name: &str) -> FunctionGraph {
    use majit_translate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

    let mut graph = FunctionGraph::new(name);
    let receiver = graph.alloc_value_var();
    let result_var = graph.alloc_value_var();
    graph
        .block_mut(graph.startblock)
        .operations
        .push(SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: CallTarget::indirect(trait_root, method_name),
                args: majit_translate::model::call_args(vec![receiver]),
                result_ty: ValueType::Int,
            },
        });
    graph.set_return(graph.startblock, Some(result_var));
    graph
}

#[test]
fn find_all_graphs_follows_every_member_of_an_indirect_call_family() {
    // RPython call.py walks `("direct_call", "indirect_call")`, and
    // call.py `graphs_from` returns the whole attached family for
    // an indirect_call, so every candidate member enters the closure.
    // Skipping the indirect branch leaves the family out of
    // `candidate_graphs`, which makes `graphs_from` return None at
    // jtransform time and downgrades the site to a bare residual call —
    // no `Live` prologue, no `GuardValue`.
    let portal_path = CallPath::from_segments(["portal"]);
    let alpha_path = CallPath::for_impl_method("Alpha", "step");
    let beta_path = CallPath::for_impl_method("Beta", "step");

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_indirect_caller_graph("portal", "Stepper", "step"),
    );
    cc.register_trait_method("step", Some("Stepper"), "Alpha", FunctionGraph::new("step"));
    cc.register_trait_method("step", Some("Stepper"), "Beta", FunctionGraph::new("step"));
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&alpha_path),
        "call.py:103-112 — every candidate member of the indirect_call \
         family must enter the closure, not just the first"
    );
    assert!(
        cc.is_candidate(&beta_path),
        "call.py:103-112 — every candidate member of the indirect_call \
         family must enter the closure, not just the first"
    );
}

#[test]
fn find_all_graphs_does_not_follow_close_stack_targets() {
    // RPython call.py:129-134 `_gctransformer_hint_close_stack_` →
    // 'residual', skipped by call.py. `get_jitcode` asserts a
    // close_stack graph never reaches it, so following the edge here
    // turns a residual classification into a panic later.
    let portal_path = CallPath::from_segments(["portal"]);
    let close_stack_path = CallPath::from_segments(["ll_close_stack"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &close_stack_path),
    );
    cc.register_function_graph(
        close_stack_path.clone(),
        FunctionGraph::new("ll_close_stack"),
    );
    cc.mark_close_stack(close_stack_path.clone());
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        !cc.is_candidate(&close_stack_path),
        "call.py:129-134 — a close_stack callee is residual and must not \
         enter the candidate closure"
    );
}

#[test]
fn find_all_graphs_leaves_unregistered_targets_as_residual() {
    // `call.py guess_call_kind`: `if getattr(funcobj, 'graph', None) is
    // None: return 'residual'`. A callee with no graph — an external
    // function, or in pyre a path with no lowered body (Rust stdlib, an
    // unextracted crate) — is residual whatever the policy says, and
    // `find_all_graphs` never makes it a candidate. This pins that contract
    // so a change that starts synthesising graphs for unregistered callees
    // (or otherwise short-circuits the residual fallback) surfaces loudly.
    //
    // It is not the module filter: `PyPyJitPolicy.look_inside_function`
    // (`pypy/module/pypyjit/policy.py`) rejects graph-bearing functions by
    // module, and pyre ports it as the `PyPyJitPolicy` the interpreter's
    // translation passes in.
    let portal_path = CallPath::from_segments(["portal"]);
    let unregistered_path = CallPath::from_segments(["external_helper"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &unregistered_path),
    );
    // NOTE: `unregistered_path` has no `register_function_graph` call — it
    // represents an opaque callee (Rust stdlib, unregistered module,
    // externally linked helper, etc.).
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&portal_path),
        "portal seeded by mark_portal must always be a candidate"
    );
    assert!(
        !cc.is_candidate(&unregistered_path),
        "unregistered direct_call targets have no graph, so they stay \
         residual and BFS must not pull them into candidate_graphs"
    );
}
