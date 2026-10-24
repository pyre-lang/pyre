//! Static registry of `Arc<JitCode>`s produced from
//! pyre-interpreter graphs.
//!
//! ## Positioning (TODO)
//!
//! Per parity rule #1 (`CLAUDE.md` majit ↔ RPython Parity Rules): this
//! module has **no RPython counterpart**. Upstream's
//! `rpython/jit/codewriter/codewriter.py make_jitcodes` is handed
//! `translator.graphs` — the rtyper has already materialised every graph
//! in process memory by the time codewriter runs. pyre cannot inherit
//! that assumption: Rust interpreter sources are on disk in a sibling crate
//! (`pyre/pyre-interpreter/`) and must become `FunctionGraph`s before
//! the codewriter can touch them.
//!
//! The adapter lives at the same logical boundary as the `front/` module
//! (Rust `syn::ItemFn` → `FunctionGraph`). Keeping the adapter here keeps
//! the parity layer (`codewriter/`) untouched: no new opnames, no new
//! `OpKind`, no new jitcode-keying schemas. The pipeline this module
//! drives is exactly the canonical
//! `analyze_multiple_pipeline_with_modules` (`crate::analyze_multiple_pipeline_with_modules`) —
//! i.e. the same entry point `rpython/jit/codewriter/codewriter.py
//! transform_func_to_jitcode` is wrapped by in the tests.
//!
//! ## What this module provides
//!
//! `with_all_jitcodes(|reg| …)` → closure access to the per-thread
//! pyre-interpreter JitCode registry keyed by `CallPath`. First call on
//! a thread performs the full pipeline via `analyze_multiple_pipeline_with_modules`;
//! subsequent calls are O(1) reads of a `thread_local!` `OnceCell`.
//! The closure form (rather than `&'static`) avoids forcing
//! `AllJitCodes: Sync`, which would in turn force the interior cells of
//! `Variable` to become thread-safe wrappers — a deviation from
//! RPython's single-thread annotator invariant.
//!
//! `AllJitCodes` is the Rust carrier that pairs upstream's
//! `CallControl.jitcodes` dict with `CallControl.all_jitcodes` list. It is
//! re-exported here as a convenience for downstream consumers that already
//! import `crate::generated`.
//!
//! ## Audience: TEST FIXTURE (not production)
//!
//! `all_jitcodes()` is consumed **only** by
//! majit-translate's own integration tests
//! (`test_phase_f_all_jitcodes.rs`,
//! `test_make_jitcodes_produces_graph_keyed_output.rs`). The pyre
//! runtime never imports this function.
//!
//! Production builds run a parallel pipeline at `pyre-jit-trace/build.rs`
//! that calls `analyze_multiple_pipeline_with_modules`
//! with `pyre_interpreter::jit_trace_fnaddrs()` and then bincode-embeds
//! the resulting `pipeline.jitcodes` into the `pyre-jit-trace` binary
//! (`$OUT_DIR/jitcodes.bin`). That separate path is what supplies
//! real `JitCode.fnaddr` values to the metainterp / blackhole runtime;
//! `all_jitcodes()` here intentionally retains the **symbolic fnaddr
//! fallback** because exercising the pipeline without `&fnaddr_bindings`
//! is what the unit tests need to assert (graph discovery, alloc-order
//! invariants, by-path keying).
//!
//! Production-side fnaddr resolution is done by
//! `pyre-jit-trace/build.rs`, so `generated::all_jitcodes` keeps its
//! symbolic-fallback contract as the deliberate test surface. It is not
//! routed through `fnaddr_bindings`. **Do not rewire this module
//! to thread `&fnaddr_bindings` — doing so would either (1) introduce a
//! `majit-translate` ⇒ `pyre-interpreter` dependency that breaks the
//! `majit/* ⊥ pyre/*` invariant, or (2) require parameterising the
//! `OnceCell` (forfeiting memoisation). Both are wrong shapes for what
//! is, in practice, a test fixture.**
//!
//! ## Why this wraps the full pipeline
//!
//! An earlier draft of this module ran its own narrow pipeline
//! (a hand-picked set of free functions + trait impl methods only, with empty
//! `StructFieldRegistry` / `known_struct_names`).
//! That shape is **narrower than `translator.graphs`** upstream assumes
//! and drops structural context:
//!
//! - Inherent impl methods (e.g. `PyFrame::push`, `PyFrame::pop`) are
//!   registered via `extract_inherent_impl_methods` in
//!   `analyze_pipeline_from_module_paths`; without them, direct_call targets
//!   like `self.pop()` cannot resolve to a concrete graph.
//! - `struct_fields` / `known_struct_names` carry array-type identity
//!   that `extract_trait_impls` consults; an empty context silently
//!   collapses those identities and the rtyped graph becomes
//!   syntax-only.
//!
//! Re-using `analyze_multiple_pipeline_with_modules` eliminates both gaps: the same
//! full-context registry the canonical analyzer consumes becomes this
//! module's input.
//!
//! ## What this module does NOT introduce
//!
//! - NOT a new key schema. The canonical key is `CallPath` (matching
//!   `CallControl.jitcodes`, which is `rpython/jit/codewriter/call.py
//!   self.jitcodes` keyed by graph identity).
//! - NOT a variant-keyed map. JitCode maps are keyed by `CallPath`, never by
//!   an opcode variant. The scope is every crate under `majit/*/src`, not
//!   this crate alone, and `test_epic_acceptance_greps` re-measures it on
//!   every run — read the selector list there rather than restating one
//!   here, since a recipe quoted beside its subject cannot disagree with
//!   the tree.
//! - NOT a new opname family. Every handler is transformed through the
//!   existing `CodeWriter::transform_graph_to_jitcode` without per-arm
//!   special cases.
//!
use std::cell::OnceCell;

pub use crate::codewriter::AllJitCodes;

/// This is the pyre-side equivalent of upstream's "reachable graph set"
/// consumed by `rpython/jit/codewriter/codewriter.py make_jitcodes`.
/// The manifest must cover every source module that defines a
/// function reachable by `direct_call` from a handler graph. pyre's
/// `analyze_multiple_pipeline_with_modules` resolves cross-file `direct_call`s
/// against the union of `function_graphs` from every module in this
/// list; a callee defined in a module absent from the manifest would be
/// emitted as a residual call (or panic during drain) even though
/// upstream treats it as inlinable graph.  The graph bodies themselves
/// come from the Charon-extracted workspace LLBC set, resolved at
/// pipeline run time — no source text is embedded.
///
/// Current roots:
/// - `pyopcode` (`pyre-interpreter/src/pyopcode.rs`) — freestanding
///   `opcode_*` handlers.
/// - `eval` (`pyre-interpreter/src/eval.rs`) — `PyFrame` trait impls
///   (LocalOpcodeHandler / SharedOpcodeHandler / ControlOpcodeHandler
///   / …).
/// - `pyframe` (`pyre-interpreter/src/pyframe.rs`) — inherent
///   `impl PyFrame` helpers (push / pop / peek / check_exc_match).
/// - `shared_opcode` (`pyre-interpreter/src/shared_opcode.rs`) —
///   freestanding `opcode_make_function`, `opcode_call`,
///   `opcode_build_{list,tuple,map}`, `opcode_store_subscr`,
///   `opcode_list_append`, `opcode_unpack_sequence`, `opcode_load_attr`,
///   `opcode_store_attr`. These are imported at `pyopcode.rs` and
///   called directly from default trait methods (pyopcode.rs).
///   Before their inclusion, `analyze_multiple_pipeline_with_modules` would report
///   them as unresolved `direct_call` targets.
/// - `eval` (`pyre-jit/src/eval.rs`) — portal runner `eval_loop_jit`
///   (pyre analogue of upstream `warmspot.py::portal_runner`) and
///   its resume/allocation helpers (`allocate_struct`,
///   `allocate_with_vtable`). Seeding this root lets
///   `find_all_graphs(portal, policy)` find the portal graph for
///   `setup_jitdriver`; opcode handlers become BFS callees, not
///   entry points.
///
/// Crate-stripped module paths, matching the form
/// `crate::module_path::module_path_from_source_file` emits, so
/// analyzer-side `struct_origins[bare_name] = module_path` and
/// `canonical_struct_name` produce the same canonical spelling
/// the runtime + production analyser pipeline produce.  Empty module
/// path (test fixtures that bypass module wiring) is reserved for
/// `parse_source`; here every entry carries its real module path.
const JIT_GRAPH_MODULES: &[&str] = &["pyopcode", "eval", "pyframe", "shared_opcode", "eval"];

thread_local! {
    /// Per-thread cache for the pyre-interpreter JitCode registry.
    ///
    /// The registry is `thread_local!` rather than a process-wide
    /// `static OnceLock<…>` because `AllJitCodes` transitively holds
    /// `Variable` graphs whose interior `RefCell` / `Cell` cells are
    /// !Sync — matching RPython's single-thread annotator invariant.
    /// All callers below this layer are single-thread test fixtures.
    static ALL_JITCODES: OnceCell<AllJitCodes> = const { OnceCell::new() };
}

/// Access the per-thread pyre-interpreter JitCode registry through a
/// closure.
///
/// First call on a given thread performs the full pipeline (see
/// [`build`]). Subsequent calls are O(1). A panic inside `build`
/// poisons the cell on that thread and every subsequent caller on the
/// same thread will panic too — by design, since a malformed handler
/// graph is a hard parity violation that should surface loudly.
///
/// The closure form (rather than `-> &'static AllJitCodes`) avoids the
/// `T: Sync` requirement that a process-wide static would impose;
/// `AllJitCodes` carries `Variable` graphs whose interior-mutability
/// cells are intentionally !Sync to match RPython's single-thread
/// annotator invariant.
pub fn with_all_jitcodes<R>(f: impl FnOnce(&AllJitCodes) -> R) -> R {
    ALL_JITCODES.with(|cell| f(cell.get_or_init(build)))
}

fn build() -> AllJitCodes {
    // Full canonical pipeline — the same entry point the
    // `test_analyze_pipeline_runs_canonical_graph_path` integration test
    // exercises. Builds a `SemanticProgram` from the LLBC modules
    // listed in `JIT_GRAPH_MODULES`, runs `analyze_program`,
    // collects trait impls + inherent impl methods with full
    // struct-field / return-type / known-struct context, wires up
    // jitdriver / portal / oopspec metadata, then calls
    // `grab_initial_jitcodes` + `drain_pending_graphs` through
    // `make_jitcodes`. The output mirrors RPython
    // `call.py self.jitcodes` (dict) + `call.py self.all_jitcodes`
    // (list).
    //
    // KNOWN DEVIATION: this path uses the
    // symbolic `JitCode.fnaddr` fallback at
    // `crate::call::symbolic_fnaddr_for_path`, NOT upstream's real
    // `getfunctionptr(graph)` surface (`rpython/jit/codewriter/
    // call.py`).
    //
    // The blocker is wider than "need a binding table":
    //
    // - Many graphs in `JIT_GRAPH_MODULES` are generic source-level
    //   functions, e.g. `pyopcode.rs` / `shared_opcode.rs`
    //   `opcode_*<H: ...>` helpers and trait default methods on
    //   `OpcodeStepExecutor`. A source graph like
    //   `opcode_load_const<H>` has no single concrete Rust fnaddr until a
    //   monomorphization is chosen (`<PyFrame as ...>`,
    //   trace-recorder handler, blackhole handler, ...).
    // - The binding-aware public entry points
    //   (`analyze_multiple_pipeline_with_*_fnaddr_bindings`) work for
    //   nongeneric helper surfaces whose concrete fnaddrs are known to a
    //   caller. `generated::with_all_jitcodes` is different: it caches one
    //   per-thread, monomorphization-neutral registry in a thread-local
    //   `OnceCell`, so it cannot pick one concrete instantiation without
    //   changing the meaning of the graph set it exposes.
    //
    // In short, this registry is parity-accurate for graph discovery and
    // JitCode bodies, but intentionally not for `fnaddr`. The symbolic
    // fallback is therefore part of the current API contract and is locked
    // down by the unit tests below.
    let result = crate::analyze_multiple_pipeline_with_modules(
        JIT_GRAPH_MODULES,
        &crate::AnalyzeConfig {
            pipeline: crate::PipelineConfig {
                transform: crate::GraphTransformConfig::default(),
                jit_drivers: vec![crate::JitDriverSpec {
                    portal: crate::CallPath::from_segments(["eval", "eval_loop_jit"]),
                    portal_runner: None,
                    portal_enter: None,
                    greens: vec![
                        "next_instr".to_string(),
                        "is_being_profiled".to_string(),
                        "pycode".to_string(),
                    ],
                    reds: vec!["frame".to_string(), "ec".to_string()],
                    green_kinds: Vec::new(),
                    red_kinds: Vec::new(),
                    autoreds: false,
                    virtualizables: vec!["frame".to_string()],
                    red_types: vec!["PyFrame".to_string(), "ExecutionContext".to_string()],
                    split_portal: false,
                }],
                register_trait_families: Vec::new(),
                helper_graphs: Vec::new(),
                builtin_graphs: Vec::new(),
            },
        },
        None,
        &|_, _| None,
        &[],
        crate::HostStaticAddrs {
            pytypes: &[("interp_exceptions::EXC_STOPITERATION_TYPE", 1)],
            ..crate::HostStaticAddrs::default()
        },
    );
    AllJitCodes {
        by_path: result.jitcodes_by_path,
        in_order: result.jitcodes,
    }
}
