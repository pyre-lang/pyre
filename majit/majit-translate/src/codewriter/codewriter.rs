//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-translate/src/codewriter/
//! ├── codewriter.py          →     ├── mod.rs (CodeWriter struct)
//! ├── jtransform.py          →     ├── jtransform.rs
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── call.py                →     └── (call.rs)
//! ```

use std::sync::Arc;

use crate::assembler::{Assembler, AssemblerExt};
use crate::call::CallControl;
use crate::jitcode::JitCode;
use crate::jtransform::GraphTransformConfig;
use crate::model::FunctionGraph;
use crate::parse::CallPath;

/// Output of `CodeWriter::make_jitcodes` — the populated JitCode registry
/// paired with its alloc-order index view.
///
/// Parity shape: RPython `rpython/jit/codewriter/call.py:87-88` holds two
/// fields on `CallControl` itself — `self.jitcodes = {}` (graph-keyed dict)
/// and `self.all_jitcodes = []` (alloc-order list). Pyre preserves both
/// fields on `CallControl` and additionally returns them as a single value
/// from `make_jitcodes`. TODO: upstream
/// callers (`warmspot.py:281-282`) read `self.all_jitcodes` off the
/// CallControl directly; pyre pairs the two views so that downstream
/// consumers (e.g. `crate::generated::all_jitcodes`) receive one value and
/// do not need to pass `CallControl` around to perform later lookups.
pub struct AllJitCodes {
    /// RPython `call.py CallControl.__init__ self.jitcodes = {}` (graph-keyed dict). Pyre's
    /// graph identity at this boundary is `CallPath`.
    pub by_path: indexmap::IndexMap<CallPath, Arc<JitCode>>,
    /// RPython `call.py:88 self.all_jitcodes = []` (alloc-order list). The
    /// `jitcode.index == i` invariant from `codewriter.py:80` is enforced
    /// by `CallControl::collect_jitcodes_in_alloc_order`.
    pub in_order: Vec<Arc<JitCode>>,
}

impl AllJitCodes {
    pub fn len(&self) -> usize {
        self.in_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.in_order.is_empty()
    }

    pub fn get(&self, path: &CallPath) -> Option<&Arc<JitCode>> {
        self.by_path.get(path)
    }
}

/// RPython: `codewriter.py::CodeWriter`.
///
/// Orchestrates the full JitCode generation pipeline:
///   annotate → rtype → jtransform → regalloc → flatten → liveness → assemble
///
/// RPython's CodeWriter owns both the Assembler and CallControl.
/// In majit, CallControl is passed by `&mut` reference to avoid
/// lifetime entanglement with the Transformer's borrows.
pub struct CodeWriter {
    /// RPython: `self.assembler = Assembler()` (codewriter.py).
    pub assembler: Assembler,
    /// RPython: `self.debug = True` (codewriter.py:18).
    pub debug: bool,
    /// Lazy program-wide `CallRegistry` for the
    /// MAJIT_RTYPER dual-gate (cf. `transform_graph_to_jitcode`).
    /// Built on first use from `callcontrol.function_graphs()` and
    /// reused across every dual-gated graph in the same CodeWriter
    /// run, mirroring upstream `Bookkeeper.descs` lifecycle (one
    /// descs map per `Translator` driving an `RPythonAnnotator`).
    real_rtyper_registry: std::cell::RefCell<
        Option<std::rc::Rc<crate::translator::rtyper::call_registry::CallRegistry>>,
    >,
}

/// `codewriter` as `MetaInterpStaticData.finish_setup` sees it: this
/// `CodeWriter` does not own its `CallControl`, so the pair stands in for
/// `codewriter.callcontrol`.
pub struct CodeWriterWithCallControl<'a> {
    pub codewriter: &'a CodeWriter,
    pub callcontrol: &'a CallControl,
}

impl majit_jitcode::codewriter::codewriter::CodeWriterSetup for CodeWriterWithCallControl<'_> {
    fn assembler(&self) -> &Assembler {
        &self.codewriter.assembler
    }

    fn virtualref_info(&self) -> Option<&std::sync::Arc<dyn crate::call::VirtualRefInfoHandle>> {
        self.callcontrol.virtualref_info.as_ref()
    }

    fn callinfocollection(&self) -> &majit_ir::CallInfoCollection {
        &self.callcontrol.callinfocollection
    }

    fn has_libffi_call(&self) -> bool {
        self.callcontrol.has_libffi_call
    }
}

impl CodeWriter {
    /// Pair this codewriter with the `CallControl` it drove, for
    /// `MetaInterpStaticData.finish_setup(codewriter)`.
    pub fn with_callcontrol<'a>(
        &'a self,
        callcontrol: &'a CallControl,
    ) -> CodeWriterWithCallControl<'a> {
        CodeWriterWithCallControl {
            codewriter: self,
            callcontrol,
        }
    }

    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py).
    ///
    /// `debug` mirrors the class-level default `debug = True`
    /// (`codewriter.py`). Upstream always produces per-jitcode
    /// diagnostic output (`log.dot()` in `print_ssa_repr`), so pyre
    /// matches by defaulting `debug: true`. Tests that want silent
    /// operation may flip the field after construction.
    pub fn new() -> Self {
        Self {
            assembler: Assembler::new(),
            debug: true,
            real_rtyper_registry: std::cell::RefCell::new(None),
        }
    }

    /// `codewriter.py CodeWriter.setup_vrefinfo(self, vrefinfo)`.
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// RPython's `CodeWriter` owns `self.callcontrol`; pyre keeps
    /// `CallControl` outside the `CodeWriter` (passed by `&mut` to
    /// `transform_graph_to_jitcode`), so the wrapper takes
    /// `callcontrol` as an explicit parameter and delegates to
    /// [`CallControl::setup_vrefinfo`].  The at-most-once assertion
    /// lives in the delegate.
    pub fn setup_vrefinfo(
        &self,
        callcontrol: &mut CallControl,
        vrefinfo: Arc<dyn crate::call::VirtualRefInfoHandle>,
    ) {
        callcontrol.setup_vrefinfo(vrefinfo);
    }

    /// Get-or-build the program-wide `CallRegistry` for the
    /// MAJIT_RTYPER dual-gate.  Builds once per CodeWriter run and
    /// caches; subsequent calls reuse the same `Rc`.  The registry
    /// is populated from `callcontrol.function_graphs()` — pyre's
    /// program-wide graph map, the production analog of
    /// `SemanticProgram.functions`.
    fn dual_gate_registry(
        &self,
        callcontrol: &CallControl,
    ) -> std::rc::Rc<crate::translator::rtyper::call_registry::CallRegistry> {
        if let Some((registry, bk)) = self
            .real_rtyper_registry
            .borrow()
            .as_ref()
            .map(|existing| (existing.clone(), existing.bookkeeper().clone()))
        {
            callcontrol.set_bookkeeper(bk);
            return registry;
        }
        // One bookkeeper per translation (`rtyper.annotator.bookkeeper`).
        // Reuse a bookkeeper already threaded onto this CallControl so
        // `InstanceRepr.convert_const` / `getuniqueclassdef(value.__class__)`
        // see the interned class; mint only when this session has none.
        let bk = match callcontrol.bookkeeper() {
            Some(existing) => existing,
            None => {
                let minted = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
                callcontrol.set_bookkeeper(minted.clone());
                minted
            }
        };
        let registry = std::rc::Rc::new(
            crate::translator::rtyper::call_registry::CallRegistry::new(bk),
        );
        // Thread the program-wide struct field shapes into the shared
        // bookkeeper so
        // `getuniqueclassdef_for_struct_root` / `project_struct_field_type`
        // can project a struct's fields onto its classdef when the
        // real-rtyper seed path resolves a `Ref(type_root)` to a class.
        registry.set_struct_fields(std::rc::Rc::new(callcontrol.struct_fields().clone()));
        // The carrier class is the program's `OperationError`: it subclasses
        // `Exception` (see `Bookkeeper::set_exception_carrier`). Mint it
        // before the struct-root prologue so first mint wins the base.
        let spec = callcontrol.error_carrier();
        let classdef_path = if spec.carrier_class.is_empty() {
            spec.carrier_path.as_str()
        } else {
            spec.carrier_class.as_str()
        };
        registry.set_exception_carrier(classdef_path);
        if !spec.carrier_class.is_empty() {
            registry.alias_exception_carrier_handle(&spec.carrier_path);
        } else {
            let _ = registry.bookkeeper().exception_carrier_class();
        }
        // Before populate: a `raw:<owner>` stub result asks the bookkeeper
        // for `SomePtr` while the unsafe-fn stubs are registered.
        registry.set_struct_layouts(callcontrol.struct_layouts_handle());
        // Enum `discriminant → variant` tables for the `__discriminant`
        // getattr's narrowing knowntypedata producer.
        registry.set_enum_variant_by_discriminant(std::rc::Rc::new(
            callcontrol.enum_variant_by_discriminant().clone(),
        ));
        // Trait → unique-impl owner map for the generic-receiver seed
        // path (`derive_subject_inputcells` resolves a bound-trait
        // `class_root` through it before the struct-root lookup).
        registry.set_trait_unique_impls(callcontrol.trait_unique_impls().clone());
        // Opt-in receiver-driven dispatch families (issue #346): mint
        // each registered trait's base ClassDef + impl subclasses BEFORE
        // `seed_struct_root_method_members` below, so the method seeding
        // attaches the impl methods onto the base-linked subclass and a
        // `dyn Trait` receiver getattr resolves the impl MethodDesc
        // family (attrfamily merge).  Empty for pyre production.
        registry.register_trait_families(callcontrol.trait_family_registrations());
        // PyPy's `Bookkeeper.compute_at_fixpoint` raises through to
        // the caller (`bookkeeper.py`); pyre's dual-gate
        // mirrors that propagation by routing the populate `TyperError`
        // through `is_known_unported`.  Known-unported categories
        // leave a partial registry (the per-graph dual-gate will
        // Skip-classify the affected callsites via the cascade
        // `"not registered in CallRegistry"`).  Unknown errors
        // panic immediately so parity bugs surface here rather than
        // silently shifting downstream behind a Skip mask.
        // Metadata-only stubs for the call-site spellings
        // `populate_call_registry_from_call_graphs` does not register: a
        // stub key is the crate-included `name_path()` verbatim, where the
        // `function_graphs` pass keys on the crate-stripped
        // `{module_path, name}` plus its alias fan-out (see
        // `CallControl::unsafe_fn_stubs`).  The specs are registered
        // INSIDE populate, between its Pass 1 (alias explosion) and Pass 2
        // (callee lift), so a safe-fn body lifted in Pass 2 that references
        // an unsafe callee (`is_cell`, `is_exception`, …) resolves it
        // instead of recording a "not registered" lift error.
        let populate_result =
            crate::translator::rtyper::cutover::populate_call_registry_from_call_graphs(
                callcontrol.function_graphs(),
                &callcontrol.unsafe_fn_stubs,
                &callcontrol.foreign_opaque_method_externals,
                &callcontrol.atomic_load_decls,
                &registry,
            );
        if let Err(err) = populate_result {
            let msg = format!("{err}");
            if !crate::translator::rtyper::cutover::is_known_unported(&msg) {
                panic!("populate_call_registry_from_call_graphs failed: {msg}");
            }
        }
        // Mirror classdesc.py:606-618 — methods live in the class
        // `__dict__`.  Install each registered `[.., Owner, method]`
        // entry's function host on the interned `Owner` class object so
        // `SomeInstance.getattr(method)` resolves to a bound MethodDesc
        // instead of blocking.  AFTER populate (which seeds the unsafe-fn
        // stubs internally) so the entry set is complete.
        registry.seed_struct_root_method_members();
        // Issue #346: register the just-seeded impl-subclass methods into
        // their ClassDef attr sources so a family-base / `dyn Trait`
        // receiver getattr resolves the impl MethodDesc family via the
        // attrfamily merge.  No-op for pyre production (no families).
        registry.populate_trait_family_base_attrs(callcontrol.trait_family_registrations());
        // RPython parity: `Translator.buildannotator()` /
        // `Translator.buildrtyper()` (`translator.py`) construct
        // exactly one annotator and one rtyper per Translator and assert
        // on re-entry.  Pyre mirrors that contract through
        // [`CallRegistry::ensure_session`]
        // (in `call_registry.rs`), which lazily builds a single
        // `(RPythonAnnotator, RPythonTyper)` pair on first use and
        // returns the cached pair on every subsequent
        // `specialize_legacy_graph_with_registry_returning_value_to_var`
        // call.  The
        // registry itself is cached on the `CodeWriter`
        // (`real_rtyper_registry`), so all per-graph subjects of one
        // CodeWriter share the same annotator + rtyper, matching
        // upstream's "one annotator + one rtyper per Translator"
        // semantics.
        *self.real_rtyper_registry.borrow_mut() = Some(registry.clone());
        registry
    }

    /// Two-phase rtyper prepass entry. Annotates the whole portal closure,
    /// THEN rtypes it, populating the shared registry's `two_phase()` cache so
    /// the subsequent per-graph drain reads cached types instead of re-running
    /// the fused per-graph path. The whole
    /// portal closure is annotated to a fixpoint before any block is rtyped,
    /// matching upstream `translator.annotate()` → `rtyper.specialize()` →
    /// `codewriter.make_jitcodes()` rather than the
    /// fused per-graph annotate+rtype.
    /// Must run AFTER `grab_initial_jitcodes` (so `candidate_graphs` is the
    /// portal closure) and BEFORE any `drain_pending_graphs` that publishes a
    /// covered graph — the production pipeline (`lib.rs`) calls this between the
    /// two.
    pub fn run_two_phase_prepass(&self, callcontrol: &CallControl) {
        let registry = self.dual_gate_registry(callcontrol);
        // Upstream `translator.graphs` is the annotate/rtype closure
        // (`driver.py` task_annotate / task_rtype_lltype) the codewriter
        // later consumes.  Builtin wrappers join `candidate_graphs` in
        // `find_all_graphs_bfs` when `helper_roots` is empty; union them
        // here so the two-phase subject set cannot close without the
        // PBC family `builtin_wrapper_indirect_graphs` names, even if a
        // caller ran the prepass against a narrower candidate set.
        let mut subjects = callcontrol.candidate_graphs().clone();
        for path in callcontrol.builtin_wrapper_indirect_graphs() {
            if callcontrol.has_function_graph(path) {
                subjects.insert(path.clone());
            }
        }
        crate::translator::rtyper::cutover::run_two_phase_prepass(
            &registry,
            &subjects,
            callcontrol.function_graphs(),
        );
        // The annotator is complete and no graph is built after it. The
        // pending bodies share their graphs with `function_graphs`, which
        // the codewriter mutates from here on.
        registry.release_source_graphs();
    }

    /// Port of `CodeWriter.transform_graph_to_jitcode`.
    ///
    /// Transforms a `FunctionGraph` into a `JitCode` through the upstream
    /// transform, register-allocation, flattening, liveness, and assembly
    /// pipeline.
    /// Upstream signature `(self, graph, jitcode, verbose, index)`. Pyre adds
    /// `path` / `callcontrol` / `config` as pyre-specific additions:
    ///   - `path`: graph identity surrogate (upstream uses `graph` object
    ///     identity; pyre uses `CallPath`).
    ///   - `callcontrol`: upstream has `self.callcontrol`; pyre passes
    ///     `&mut` due to Rust borrow-checker constraints (the Transformer
    ///     borrows callcontrol during `transform()`).
    ///   - `config`: pyre's `GraphTransformConfig` carries options that
    ///     upstream keeps in globals / command-line flags.
    ///
    /// Pyre performs annotation and rtyping before the upstream stages, calls
    /// `compute_liveness` from its assembler, and fills the call descriptors
    /// before committing the assembled body. The resulting stage order remains
    /// the one implemented by `CodeWriter.transform_graph_to_jitcode`.
    ///
    /// **Type-source contract (post graph-side concretetype migration)**
    ///
    /// `regalloc`/`flatten`/`assemble`/`liveness`/`format` all read
    /// kinds via `FunctionGraph::concretetype_of(&v)`, which routes
    /// straight to the backing `Variable.concretetype` cell carried
    /// inline by the `Variable` objects the IR references. RPython's
    /// `flowspace.model.Variable.concretetype` is the
    /// single source of truth for every value.  No type side-table
    /// parameter survives across stages: the post-rtyper merge
    /// `merge_synth_kinds_into_graph` stamps each synthetic
    /// Variable's `.concretetype` cell via
    /// `set_concretetype_of_inline`; before jtransform,
    /// `apply_from_flowspace_variables` copies lltypes from typed Variables
    /// in the `value_to_var` map so the rtyper's `Variable.concretetype` is
    /// what jtransform starts from.  Slots without a rtyper-bound Variable
    /// keep the synthetic canonical type the merge wrote.
    ///
    /// **Remaining structural divergence** — pyre's codewriter still
    /// consumes [`crate::model::FunctionGraph`] (a slot-indexed legacy
    /// IR) instead of [`crate::flowspace::model::FunctionGraph`] (the
    /// upstream `Variable`-based shape).  Operand identity in
    /// `SpaceOperation` payloads is already `Variable`;
    /// `FlatOp` operands are `Register` (regalloc color +
    /// kind).  What still ties the codewriter to the legacy IR is that
    /// it consumes [`crate::model::FunctionGraph`] and bridges to the
    /// rtyper's typed Variables through the identity-keyed
    /// `value_to_var` map. A graph-native typed IR would remove that bridge and
    /// let the codewriter consume the rtyper's `Variable` graph directly.
    ///
    /// Run the dual-gate type resolver against the program-wide
    /// `CallRegistry` and commit every resolved kind to the graph's backing
    /// `Variable.concretetype` cells. A Match returns the `LegacyToTyped` map so
    /// post-jtransform operands can be rebound to typed variables. A Skip runs
    /// the legacy annotator and resolver, commits their kinds, and returns
    /// `None`. `diag_label` identifies either outcome in diagnostics.
    pub fn dual_gate_publish_concretetypes(
        &mut self,
        graph: &FunctionGraph,
        callcontrol: &mut CallControl,
        diag_label: &str,
    ) -> Option<crate::translator::rtyper::flowspace_adapter::LegacyToTyped> {
        let dual_gate_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let registry = self.dual_gate_registry(callcontrol);
            // When the two-phase prepass populated the cache, read the cached
            // real types instead of re-running the fused per-graph real path
            // (which would re-rtype shared callees and re-trip the late-stage
            // Skip). The legacy-baseline comparison + fallback stay per-graph.
            if registry.two_phase_prepass_done() {
                crate::translator::rtyper::cutover::dual_gate_outcome_from_cache(
                    graph, &registry, diag_label,
                )
            } else {
                crate::translator::rtyper::cutover::dual_gate_check_with_registry(
                    graph,
                    &registry,
                    callcontrol.function_graphs(),
                )
            }
        }));
        let outcome = match dual_gate_outcome {
            Ok(result) => result,
            Err(payload) => {
                let msg = crate::annotator::model::panic_payload_text(&*payload)
                    .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
                if crate::translator::rtyper::cutover::is_known_unported(&msg) {
                    Ok(crate::translator::rtyper::cutover::DualGateOutcome::Skip(
                        format!("registry build panicked: {msg}"),
                    ))
                } else {
                    Err(format!("registry build panicked: {msg}"))
                }
            }
        };
        match outcome {
            Ok(crate::translator::rtyper::cutover::DualGateOutcome::Match {
                real_value_to_var,
                real_constants,
            }) => {
                // The accept arm, recorded so this gate's rows sum to a
                // denominator.  "N graphs Skipped" is not a finding on its
                // own — "N of M" is, and M has to come from the same run
                // rather than from a figure written down elsewhere.
                crate::decline::observe_accept(
                    crate::decline::gate::DUAL_GATE,
                    "match-real-rtyper (ACCEPT, not a decline)",
                    &graph.name,
                );
                // Commit each real-rtyper Variable's `concretetype`
                // (LowLevelType) onto its placeholder on the graph's
                // value table.  Mirrors RPython `rtyper.py setconcretetype v.concretetype = ...`
                // attribute aliasing; reads via `FunctionGraph::concretetype_of(&v)`
                // then match upstream `getkind(v.concretetype)`.
                for var in real_value_to_var.values() {
                    if let Some(lltype) = var.concretetype().as_ref() {
                        crate::model::FunctionGraph::set_concretetype_of_inline(
                            var,
                            crate::model::getkind(lltype),
                        );
                    }
                }
                // Hydrate the *legacy* Variables (the map keys) jtransform
                // reads through `FunctionGraph::concretetype_of` with the
                // real path's resolved kinds now, before jtransform's
                // `make_three_lists_from_vars` partitions residual-call args
                // by kind.  The commit loop above stamps the typed flowspace
                // Variables (the `.values()`), but a residual `dont_look_inside`
                // decode helper's argument list is partitioned off the legacy
                // key Variable's cell, so it has to carry the real kind
                // before jtransform partitions it.  This is the only copy:
                // jtransform's own retypes stand after it.
                crate::codewriter::type_state::apply_from_flowspace_variables(&real_value_to_var);
                crate::codewriter::type_state::apply_from_flowspace_constants(&real_constants);
                Some(real_value_to_var)
            }
            Ok(crate::translator::rtyper::cutover::DualGateOutcome::Skip(reason)) => {
                // A Skip is a decline: the real rtyper refused this graph
                // and the legacy walker types it instead.  This is the fork
                // that decides whether a host gets the real rtyper or the
                // flattening fallback, and it is keyed on substring-matching
                // a diagnostic string (`cutover::unported_category`) — so
                // WHICH arm matched is the finding, not that one did.  A
                // single "Skip" count cannot say whether a host is dominated
                // by registry misses or by unimplemented operations, and
                // those imply completely different work.
                //
                // The classification already exists at this point and was
                // being discarded; the outcome was visible only behind
                // `MAJIT_RTYPER_VERBOSE=1`.  Record the arm as the count key
                // and the graph as the subject, so the row carries both an
                // event count and a distinct-graph count.
                //
                // WHICH ROUTE the Skip took is decidable here, and the two
                // must not share a bucket.  `unported_category` classifies
                // a CAUGHT PANIC PAYLOAD; the arm above is the only
                // producer of that shape and it stamps a
                // `registry build panicked: ` prefix.  Every other Skip
                // arrives as a direct `Ok(DualGateOutcome::Skip(..))` and
                // never reaches that predicate at all — the two
                // populations are disjoint by construction, not
                // overlapping.
                //
                // So an unmatched direct Skip is NOT "unknown shape, the
                // arm list is incomplete".  It is "this route never
                // consults the arm list".  Those prescribe opposite next
                // actions — extend the arms, versus instrument a route
                // nobody had drawn — so the row is labelled for its
                // mechanism.  Same defect as a zero that two causes can
                // produce.
                const PANIC_PREFIX: &str = "registry build panicked: ";
                let class = if reason.starts_with(PANIC_PREFIX) {
                    crate::translator::rtyper::cutover::unported_category(&reason)
                        // Unreachable while the arm above is the sole
                        // producer (it only builds this shape when
                        // `is_known_unported` said yes), so a row here
                        // means that invariant broke.
                        .unwrap_or("panic-route-unmatched-by-any-arm")
                } else {
                    crate::translator::rtyper::cutover::non_arm_skip_category(&reason)
                        .unwrap_or("direct-skip (never reaches unported_category)")
                };
                crate::decline::record_reason(
                    crate::decline::gate::DUAL_GATE,
                    class,
                    &reason,
                    &graph.name,
                );
                if std::env::var_os("MAJIT_RTYPER_VERBOSE").is_some_and(|v| v == "1") {
                    eprintln!(
                        "[MAJIT_RTYPER skip] graph {diag_label:?} ({:?}): {reason}",
                        graph.name,
                    );
                }
                // Reset all annotation cells before re-running the
                // legacy walker.  The real-path attempt inside
                // `dual_gate_check_with_registry` may have partially
                // populated cells before panicking on a known-unported
                // shape (`SomeInstance.getattr` on classdef-less
                // instance, `Cannot find attribute`, etc.) — those
                // residue annotations propagate here and trip the
                // monotonicity assert at `legacy_annotator::setbinding
                // :158` (new value Integer does not contain previous
                // value Instance(classdef=None)).  PyPy parity:
                // `RPythonAnnotator.__init__` starts with `bindings =
                // {}`, equivalent to clearing every cell to `None`
                // before the iterative fixpoint loop.
                for var in graph.iter_variables() {
                    *var.annotation.borrow_mut() = None;
                }
                crate::translator::rtyper::legacy_annotator::annotate(graph);
                // `resolve_types` commits per-Variable concretetype
                // cells at the resolver boundary.  Skip arm has no
                // flowspace Variable surface — the legacy walker stays
                // the only source of types for these graphs.  Reads
                // annotations off `Variable.annotation` populated by
                // the preceding `annotate` post-loop publish.
                crate::translator::rtyper::legacy_resolve::resolve_types(graph);
                // Clear `Variable.annotation` after the Skip-arm
                // legacy walker has committed each cell's
                // `Variable.concretetype` via
                // `FunctionGraph::set_concretetype_of_inline`.
                // Downstream consumers (jtransform, regalloc, ...)
                // read `graph.concretetype(v)` — the persisted
                // `concretetype` cell — so clearing the intermediate
                // `annotation` is information-preserving.  Restores
                // the `flowspace_adapter::seed_variable` documented
                // invariant ("production `addpendingblock` flowin
                // path leaves `legacy_var.annotation` empty so the
                // fresh Variable starts unannotated and flowin
                // populates it via `setbinding`") for subsequent
                // dual-gate passes on the same graph; without it the
                // Skip arm's `legacy_annotator::annotate` writes
                // persist as residue and re-entering
                // `function_graph_to_flowspace::seed_variable` would
                // copy a wider legacy lift onto the fresh flowspace
                // Variable, tripping `bindinputargs::setbinding`
                // monotonicity when the real path's flowin computes a
                // narrower binding.
                for var in graph.iter_variables() {
                    *var.annotation.borrow_mut() = None;
                }
                None
            }
            Err(diff) => panic!(
                "MAJIT_RTYPER real-path failure on graph {diag_label:?} ({:?}): {diff}",
                graph.name,
            ),
        }
    }

    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub fn transform_graph_to_jitcode(
        &mut self,
        graph: &FunctionGraph,
        path: &crate::parse::CallPath,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
        jitcode: &std::sync::Arc<JitCode>,
        verbose: bool,
        index: usize,
    ) {
        // RPython: graph = copygraph(graph, shallowvars=True) (codewriter.py:38)
        // In Rust, Transformer.transform() already clones the graph.

        // Step 0: annotate + rtype (majit-specific)
        // RPython: types are already on Variable.concretetype from the rtyper.
        //
        // The legacy walker was narrowed to the Skip arm; the dual-gate
        // logic was lifted into the shared
        // [`Self::dual_gate_type_state`] helper so the parser-level
        // debug snapshot in `make_jitcodes`
        // (in `lib.rs`) can route through the same path.
        let canonical_diag = path.canonical_key().to_string();
        // `dual_gate_publish_concretetypes` commits every resolved
        // kind to each backing `Variable.concretetype` cell in both
        // arms, so downstream consumers read kinds via
        // `FunctionGraph::concretetype_of(&v)` directly.
        let real_value_to_var = crate::codewriter::transform_profile::time_phase(
            "step0_dual_gate_publish_concretetypes",
            || self.dual_gate_publish_concretetypes(graph, callcontrol, &canonical_diag),
        );

        // Indirect calls were lowered in place on the graphs `CallControl`
        // owns (`CallControl::lower_registered_indirect_calls`), before this
        // clone. That pass leaves a family with no registered impl as a
        // `CallTarget::Indirect` marker; the clone jtransform rewrites takes
        // it as the unknown family (`indirect_call` with `graphs=None`).
        let mut graph_owned = graph.clone();
        // The error carrier's `raise e` / `except OperationError as e`
        // edges, annotated as such, move to the runtime exception value
        // before jtransform reads them.
        crate::codewriter::error_carrier_edges::lower_error_carrier_edges(
            &mut graph_owned,
            &callcontrol.error_carrier().clone(),
        );
        crate::translator::rtyper::rpbc::lower_indirect_calls(&mut graph_owned, callcontrol);
        #[cfg(debug_assertions)]
        crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(&graph_owned);
        // Pre-jtransform rtyper fold of unit-variant ctors to singleton
        // instance constants (`rtyper/rpbc.py::SingleFrozenPBCRepr`).
        // Catches both the Match-arm and Skip-arm graphs; the
        // companion fold in `flowspace_adapter::
        // legacy_const_define_hlvalue` only reaches Match-arm graphs.
        let registry = self.dual_gate_registry(callcontrol);
        crate::translator::rtyper::unit_variant_fold::fold_unit_variant_ctors(
            &mut graph_owned,
            Some(registry.bookkeeper()),
        );
        // `resolve_types` (called upstream by the rtyper) already
        // commits each backing Variable's `concretetype` cell as it
        // resolves, so jtransform reads kinds via
        // `FunctionGraph::concretetype_of(&v)` (the upstream
        // `getkind(v.concretetype)` path) without a separate publish
        // step here.
        // Annotator monomorphization producer — for every Call(Method)
        // op in the graph, look up the receiver's annotated flowspace
        // Variable via `real_value_to_var`, resolve its concrete impl
        // type from `SomeInstance.classdef` → `MethodDesc.selfclassdef`,
        // and stamp the synthesized `CallPath` onto
        // `CallTarget::Method.resolved_path`. The resolver fast path in
        // `CallControl::resolve_method` / `resolve_method_impl_type`
        // then does `function_graphs.get(&path)` directly instead of
        // the receiver-root heuristic (mirrors `bookkeeper.py getmethoddesc
        // getmethoddesc` classdef-keyed dispatch). Ops whose receiver
        // carries no SomeInstance annotation are left untouched.
        if let Some(value_to_var) = real_value_to_var.as_ref() {
            crate::codewriter::transform_profile::time_phase(
                "step0c_stamp_classdef_hints_on_graph",
                || stamp_classdef_hints_on_graph(&mut graph_owned, value_to_var),
            );
        }
        // rstr.py `StringRepr.convert_const` resolves string literals into
        // prebuilt `Ptr(STR)` constants before the codewriter sees the
        // graph. Fold the synthetic `__str_const` calls the front lowers
        // literals to, so jtransform sees a constant rather than a call.
        crate::translator::rtyper::str_const_fold::fold_str_consts(&mut graph_owned);
        // String view construction is an identity in the model graph. When a
        // box call receives a proven literal, the interned `W_UnicodeObject`
        // is one Ref constant — not an rstr `Ptr(STR)`.
        crate::translator::rtyper::box_str_const_fold::fold_box_str_constants(&mut graph_owned);

        // RPython codewriter.py:37 `portal_jd =
        // self.callcontrol.jitdriver_sd_from_portal_graph(graph)` — look
        // up the JitDriverStaticData keyed by portal graph identity. In
        // pyre `CallPath` is the surrogate, so resolve via `path`. Store
        // only the index to avoid a second borrow against
        // `callcontrol` during Transformer construction; the
        // Transformer can re-dereference `jitdrivers_sd[portal_jd_index]`
        // on demand if it needs the full record.
        let portal_jd_index = callcontrol
            .jitdriver_sd_from_portal_graph(path)
            .map(|jd| jd.index);

        // Step 1: jtransform (codewriter.py:42)
        // RPython: transform_graph(graph, cpu, callcontrol, portal_jd)
        //
        // No `with_type_state(&type_state)` — `dual_gate_type_state`
        // already committed every kind to each backing Variable's
        // `concretetype` cell. `copygraph` shares that `Rc`, and
        // `Transformer::transform_owned` rewrites this graph in place
        // (`jtransform.py` `Transformer.transform`).
        //
        // `cpu.rtyper.exceptiondata.fn_exception_match` is cloned out
        // of the dual-gate session before the `&mut CallControl`
        // borrow, matching `transform_graph(graph, cpu, ...)`.
        let excmatch = self
            .dual_gate_registry(callcontrol)
            .session_if_started()
            .and_then(|(_, rt)| rt.exceptiondata().ok())
            .and_then(|ed| ed.fn_exception_match.borrow().clone());
        let mut rewritten =
            crate::codewriter::transform_profile::time_phase("step1_jtransform_transform", || {
                let mut transformer = crate::jtransform::Transformer::new(config)
                    .with_callcontrol(callcontrol)
                    .with_portal_jd(portal_jd_index)
                    .with_excmatch(excmatch.as_ref());
                transformer.transform_owned(graph_owned)
            });
        // Transformer is dropped here, releasing the &mut CallControl borrow.

        // RPython stores `.concretetype` on each Variable. Pyre merges
        // the pre-jtransform rtyper table with explicit result kinds
        // introduced by jtransform (`CallResidual.result_kind`, etc.)
        // without letting stale pre-rewrite entries override them.
        //
        // The merge precedence is `stamped > post_result > original`:
        //   - `original` = pre-jtransform `type_state` from the real
        //     RPythonTyper (already typed every pre-rewrite Variable).
        //   - `post_result` = `authoritative_result_types(rewritten)`
        //     reads the rewritten ops' declared `result_ty`/`result_
        //     kind` fields (per-op ground-truth).
        //   - `stamped` = `synth_kinds` jtransform produced for
        //     freshly-synthesised values.
        //
        // The legacy `resolve_types(rewritten_graph, annotations)`
        // walker that previously fed `merge_synth_kinds`'s `post_
        // resolve` lane is no longer called: every post-rewrite
        // Variable either reuses a pre-rewrite identity (covered by
        // `original`), is an op.result with declared kind (covered by
        // `post_result`), or is a synth jtransform value (covered by
        // `stamped`).  Block inputargs introduced by jtransform pick
        // up types via `original` (jtransform reuses the pre-rewrite
        // inputarg Variables when splicing new control flow).
        // Merge the four post-rtyper kind sources directly into
        // each backing `Variable.concretetype` (via
        // `FunctionGraph::set_concretetype_of_inline`) — pyre's
        // analogue of RPython's "rtyper finishes, every Variable
        // has `.concretetype` inline" handoff (`rtyper.py`).
        // Precedence stack
        // `stamped > post_result > post_resolve > original` is
        // preserved; the graph IS the merge target, no intermediate
        // type side-table survives the call.
        let post_result_types = crate::codewriter::transform_profile::time_phase(
            "step1b_authoritative_result_types",
            || crate::codewriter::type_state::authoritative_result_types(&rewritten.graph),
        );
        // `post_resolve` is intentionally empty here: jtransform now
        // writes resolved kinds straight to each backing
        // `Variable.concretetype` (see `Transformer::transform` →
        // `apply_to_graph`), so the legacy `resolve_rewritten_types`
        // walk is structurally dead in the production path.
        //
        // The rtyper's types were copied onto the legacy Variables once,
        // before jtransform (`dual_gate_publish_concretetypes`), the point
        // where `rtyper.py` leaves `v.concretetype` set.  They are not
        // copied again here: jtransform may retype a Variable it reuses
        // (`iter_lower` turns a range iterator's loop slot into the range's
        // integer stop), and a second copy would restore the pre-jtransform
        // type onto a slot that now holds a different kind of value.
        // Commit jtransform-induced op-result kinds (`result_ty` /
        // `result_kind` declarations) to each backing
        // `Variable.concretetype` cell.  Pre-jtransform kinds are
        // already on the graph cells (rtyper boundary via
        // `apply_to_graph` / `resolve_types`); this overlay handles the
        // post-jtransform op-result deltas.  It runs after the hydration
        // above because a rewritten declaration outranks the
        // pre-jtransform cell: an inlined `Result<bool, PyError>` call is
        // the post-exception `Bool` value even though the Rust aggregate
        // typed it as a `Ref` shell.  Precedence stack
        // `post_result > pre-jtransform` falls out of
        // `set_concretetype_of_inline`'s "preserve richer existing
        // iff getkind matches" semantic.
        for (var, kind) in &post_result_types {
            if !matches!(kind, crate::model::ConcreteType::Unknown) {
                crate::model::FunctionGraph::set_concretetype_of_inline(var, *kind);
            }
        }

        // Steps 2-4 (regalloc → flatten → liveness/assemble → calldescr →
        // commit) operate on the rewritten `crate::model::FunctionGraph` and
        // are shared verbatim with the opname-dispatch spine, so they live in
        // `finalize_rewritten_graph_to_jitcode`.
        self.finalize_rewritten_graph_to_jitcode(
            &mut rewritten.graph,
            path,
            callcontrol,
            jitcode,
            verbose,
            index,
        );
    }

    /// Shared assembler tail for both codewriter spines.
    ///
    /// Spine A (`transform_graph_to_jitcode`) reaches here after the rtyper
    /// dual-gate + `Transformer::transform` produced a rewritten rich-`OpKind`
    /// graph; Spine B (`transform_opname_graph_to_jitcode`) reaches here after
    /// `jtransform_opname::lower_graph` produced an equivalent rich-`OpKind`
    /// graph from the rtyper's opname `SpaceOperation`s.  From this point the
    /// pipeline is identical — Steps 2-4 (codewriter.py): regalloc →
    /// flatten → liveness/assemble → `get_jitcode_calldescr` → commit body →
    /// assign `jitcode.index`.  Each Variable's `.concretetype` cell is the
    /// kind source (the upstream `getkind(v.concretetype)` access).
    fn finalize_rewritten_graph_to_jitcode(
        &mut self,
        rewritten_graph: &mut crate::model::FunctionGraph,
        path: &crate::parse::CallPath,
        callcontrol: &mut CallControl,
        jitcode: &std::sync::Arc<JitCode>,
        verbose: bool,
        index: usize,
    ) {
        // Step 2: regalloc (codewriter.py:45-47)
        // RPython: for kind in KINDS: regallocs[kind] = perform_register_allocation(graph, kind)
        // Pyre reads each per-value kind via
        // `FunctionGraph::concretetype_of(&v)` (set up by `apply_to_graph`
        // / `apply_from_flowspace_variables` before jtransform), matching upstream's
        // `getkind(v.concretetype)` access.
        // Stamp canonical exceptblock kinds first so the rtyper-skip
        // path still gets `(etype=Int, evalue=Ref)`.
        crate::regalloc::augment_canonical_exceptblock_on_graph(rewritten_graph);
        // Void-widening (exceptiontransform parity).  A scoped
        // `Result<(), PyError>` callee declares `FUNC.RESULT = void`
        // (`return_type = "()"`, stamped in `front::mir`), but its CFG
        // still returns the unit `()` as a `Ref`-typed value because
        // every aggregate lowers to `Ref`.  Collapse the returnblock to a
        // genuine void return now — after the whole-program annotation
        // fixpoint, the same ordering `rpython/translator/exceptiontransform.py`
        // uses by running after rtyping — so flatten emits a void return
        // and `graph_result_kind` agrees with the declared `v`.  The
        // `declared==v && cfg==r` gate is exactly the unit-shell case; a
        // genuinely void CFG (`cfg==v`) needs no rewrite.
        if callcontrol.declared_return_kind(path) == Some('v')
            && graph_result_kind(rewritten_graph) == 'r'
        {
            crate::front::result_exc::widen_unit_return_to_void(rewritten_graph);
        }
        // Sweep the orphaned unit producers of any void-returning graph.
        // A void function's `()` value is built (a pure `ConstRefNull`
        // after `fold_unit_variant_ctors`) but the void return never
        // consumes it; without a dead-op pass it survives into flatten as
        // a value regalloc must colour, colliding a register with a live
        // parameter.  Run on every graph whose result kind is now void —
        // both the just-widened `declared='v' && cfg='r'` case and graphs
        // whose CFG already returns void (`cfg='v'`, e.g. `w_list_append`).
        // `rpython/translator/simplify.py remove_dead_links_and_setattrs`
        // runs after exceptiontransform for the same reason; `prune_dead_phis`
        // keeps raising/impure producers and only drops genuinely pure dead
        // shells.
        if graph_result_kind(rewritten_graph) == 'v' {
            crate::model::prune_dead_phis(rewritten_graph);
        }
        crate::model::remove_duplicate_inputargs(rewritten_graph);
        // `history.getkind(Ptr(GC))` is `"ref"`.  Stamp GC field and array
        // bases that still carry Signed so regalloc colours them in the Ref
        // bank.  `SSA_to_SSI` copies `concretetype` by value (`Variable::copy`),
        // so the stamp has to land on the single pre-SSI variable.  Stamping
        // afterwards updates only the copy that is the field base and leaves
        // the threaded copies in the other bank.
        super::type_state::promote_gc_field_bases(rewritten_graph, Some(callcontrol));
        // Re-establish SSI before register allocation.  RPython runs the
        // codewriter on graphs the translator already converted to SSI via
        // `SSA_to_SSI` (`simplify.py:1067`); `perform_register_allocation`
        // (`regalloc.rs`, ported from `rpython/jit/codewriter/regalloc.py`)
        // therefore derives cross-block liveness purely from link-threading
        // — a value lives across a block boundary only when it is passed
        // through an exit link's args into the successor's inputargs.  pyre's
        // rtyper/front-end produces these model graphs in global-SSA form:
        // a value defined in one block is referenced directly in a later
        // block (one block per former call) without being threaded along the
        // linear chain between them.  Without threading the colourer sees no
        // interference between two values simultaneously live across such a
        // chain (e.g. `w_list_append`'s `length` from `int_items.len` and
        // `capacity` from the backing block's `capacity` header, both live
        // into the `int_lt` capacity check) and aliases them to one register
        // — the body miscompiles to `int_lt(cap, cap)` and always takes the resize
        // path.  Run `SSA_to_SSI` (`model_ssa.rs`) here, after the dead-phi
        // / duplicate-inputarg cleanups and immediately before regalloc, so
        // every graph reaching the colourer is SSI.  Idempotent on graphs
        // already in SSI form (empty pending set), so non-split graphs are
        // unaffected.
        crate::model_ssa::ssa_to_ssi(rewritten_graph);
        // history.py getkind(Ptr(GC)) is "ref". Stamp GC field bases, then
        // align SSI phis and rebucket kind lists, then cast a GcRef link
        // that still feeds a Signed inputarg.
        super::type_state::promote_gc_field_bases(rewritten_graph, Some(callcontrol));
        super::type_state::align_gc_link_args(rewritten_graph);
        // Stamp address results before kind lists are split, so a cell
        // that moves to the int bank is not left inside a ref list.
        super::type_state::force_int_bank_cells(rewritten_graph);
        super::type_state::rebucket_kind_lists(rewritten_graph);
        super::type_state::coerce_cross_bank_links(rewritten_graph);
        let mut regallocs = crate::codewriter::transform_profile::time_phase(
            "step2_perform_all_register_allocations",
            || crate::regalloc::perform_all_register_allocations(rewritten_graph),
        );

        // Flatten the graph after register allocation.
        // Each Variable's `.concretetype` cell is the kind source
        // after type merge and hydration; flatten reads it via
        // `FunctionGraph::concretetype_of(&var)`.  `flatten_graph`
        // itself runs `enforce_input_args` so the
        // startblock inputarg colors land in the dense `0..N` prefix
        // of each kind, and the rotation persists into the assembler
        // assembly, matching the upstream invocation order.
        let mut ssarepr =
            crate::codewriter::transform_profile::time_phase("step3_flatten_graph", || {
                crate::flatten::flatten_graph(rewritten_graph, &mut regallocs)
            });

        // `assemble_with_callcontrol` computes liveness internally and returns
        // the body so call descriptors can be filled before `set_body` commits
        // the JitCode shell.
        let mut body = crate::codewriter::transform_profile::time_phase("step4_assemble", || {
            self.assembler
                .assemble_with_callcontrol(&mut ssarepr, &regallocs, Some(callcontrol))
        });

        // call.py `get_jitcode_calldescr`:
        //   FUNC = lltype.typeOf(fnptr).TO
        //   NON_VOID_ARGS = [ARG for ARG in FUNC.ARGS if ARG is not lltype.Void]
        //   calldescr = cpu.calldescrof(FUNC, NON_VOID_ARGS, FUNC.RESULT,
        //                               EffectInfo.MOST_GENERAL)
        // Arg kinds and the result kind come from the function type. The
        // flattened CFG return kind must agree; a disagreement is a hard
        // error in every build (`graph_result_kind` is that colour).
        {
            let arg_classes = match callcontrol.declared_non_void_arg_classes(path) {
                Some(classes) => classes,
                None => cfg_non_void_arg_classes(rewritten_graph),
            };
            let cfg_kind = graph_result_kind(rewritten_graph);
            let declared_kind = callcontrol.declared_return_kind(path);
            // A body that only aborts never assigns the return place
            // (`|| missing_hash_hook()`, `FnOnce::call_once` → `i64`).
            // `annrpython.py` `complete` binds that return var to
            // `s_ImpossibleValue`; it is not `FUNC.RESULT`. The call
            // descriptor still reads the function type
            // (`call.py` `get_jitcode_calldescr`).
            let cfg_kind = match declared_kind {
                Some(declared)
                    if declared != cfg_kind
                        && returnblock_predecessor_count(rewritten_graph) == 0 =>
                {
                    declared
                }
                _ => cfg_kind,
            };
            let result_type =
                func_result_kind(rewritten_graph.name.as_str(), declared_kind, cfg_kind);
            // `map_type_to_argclass(SingleFloat) == 'S'` while `getkind` is
            // `'int'`. The call descr keeps `'S'`; the bank check above
            // already required the int kind.
            let result_type = match callcontrol.declared_result_argclass(path) {
                Some('S') if result_type == 'i' => 'S',
                Some('S') => panic!(
                    "graph {} declared FUNC.RESULT=S but CFG return kind is {result_type}",
                    rewritten_graph.name
                ),
                _ => result_type,
            };
            body.calldescr = crate::jitcode::BhCallDescr::from_arg_classes(
                arg_classes,
                result_type,
                majit_ir::descr::EffectInfo::MOST_GENERAL,
            );
        }

        // Commit the body to the pre-allocated `Arc<JitCode>` shell.
        // RPython mutates the JitCode in place; pyre uses `OnceLock`
        // so that shells handed out earlier (e.g. into
        // `JitDriverStaticData.mainjitcode` by `grab_initial_jitcodes`)
        // see the same body without locking.
        jitcode.set_body(body);

        // RPython `codewriter.py jitcode.index = index` — assign the
        // dense position in `all_jitcodes[]` immediately after
        // `assembler.assemble(...)` (upstream line 67). The
        // `in_order[i].index == i` invariant asserted by
        // `CallControl::collect_jitcodes_in_alloc_order` (and mirrored in
        // `test_phase_f_all_jitcodes::all_jitcodes_indices_match_alloc_order`)
        // depends on the caller having passed `index = finished_jitcodes.len()`
        // BEFORE appending.
        jitcode.set_index(index);

        if self.debug {
            // RPython `codewriter.py` → `print_ssa_repr(ssarepr,
            // portal_jd, verbose)` → `log.dot()` (default, verbose=False)
            // or `print(format_assembler(ssarepr))` (verbose=True). Pyre
            // currently mirrors only the low-noise branch: one line per
            // jitcode with the name, analogous to upstream's udir
            // filename (`codewriter.py
            // dir.join(name+extra).write(format_assembler(ssarepr))`).
            // The `verbose` parameter is plumbed to match the upstream
            // signature; the high-verbosity branch lands when
            // `format_assembler` is ported.
            let _ = verbose;
            let _ = &ssarepr;
            // RPython `codewriter.py log.dot()` is unconditional —
            // the only gate is the `CodeWriter.debug` instance flag
            // (`codewriter.py CodeWriter debug = True`).  Keep that semantics:
            // a single `self.debug` gate, no additional `MAJIT_LOG`
            // dependency.
            eprintln!("[CodeWriter] {}", jitcode.name);
        }
    }

    /// Spine B entry: lower an rtyper opname helper graph and finalize it.
    ///
    /// The drain loop routes a path here (instead of through
    /// `transform_graph_to_jitcode`) when `CallControl::take_opname_graph`
    /// yields the registered `crate::flowspace::model::FunctionGraph`.
    /// `jtransform_opname::lower_graph` transduces it to an equivalent
    /// rich-`OpKind` `crate::model::FunctionGraph`, which then re-enters the
    /// shared `finalize_rewritten_graph_to_jitcode` tail.  None of the
    /// Spine-A front steps (dual-gate concretetype publish,
    /// `lower_indirect_calls`, unit-variant fold, classdef-hint stamping) run:
    /// the helper graph is already low-level and fully typed on each
    /// `Variable.concretetype` cell.
    fn transform_opname_graph_to_jitcode(
        &mut self,
        flow_graph: &crate::flowspace::model::FunctionGraph,
        path: &crate::parse::CallPath,
        callcontrol: &mut CallControl,
        jitcode: &std::sync::Arc<JitCode>,
        verbose: bool,
        index: usize,
    ) {
        // jtransform.py transform_graph starts with
        // constant_fold_ll_issubclass(graph, cpu) on the flowspace
        // graph. Spine B is the remaining entry that still holds
        // `direct_call` Constants.
        if let Some((_, rt)) = self.dual_gate_registry(callcontrol).session_if_started()
            && let Ok(ed) = rt.exceptiondata()
        {
            let matcher = ed.fn_exception_match.borrow();
            crate::jtransform::constant_fold_ll_issubclass_flowgraph(flow_graph, matcher.as_ref());
        }
        let mut lowered = crate::codewriter::jtransform_opname::lower_graph(flow_graph);
        self.finalize_rewritten_graph_to_jitcode(
            &mut lowered,
            path,
            callcontrol,
            jitcode,
            verbose,
            index,
        );
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py).
    ///
    /// Full pipeline: grab_initial_jitcodes → enum_pending_graphs loop → finished.
    pub fn make_jitcodes(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> AllJitCodes {
        // RPython: self.callcontrol.grab_initial_jitcodes() (codewriter.py:76)
        callcontrol.grab_initial_jitcodes();
        // Two-phase rtyper prepass: annotate the whole portal closure, THEN
        // rtype it, BEFORE
        // the per-graph drain — mirroring upstream `translator.annotate()` →
        // `rtyper.specialize()` → `make_jitcodes()`. This unions every caller of
        // a shared callee (e.g. `type_error`) before any callee is rtyped,
        // removing the `fixed_graphs` late-stage-modify Skip the fused per-graph
        // path suffers. The per-graph `dual_gate_publish_concretetypes` then
        // reads the cached result.
        self.run_two_phase_prepass(callcontrol);
        callcontrol.lower_registered_indirect_calls();
        self.make_jitcodes_pending(callcontrol, config)
    }

    /// Drain pending graphs and fill each `Arc<JitCode>` shell's body.
    ///
    /// RPython codewriter.py:79-84: the enum_pending_graphs loop. Pyre
    /// stores the allocated `Arc<JitCode>` shells inside
    /// `CallControl::jitcodes`; this loop pops one path at a time and
    /// commits its body via `JitCode::set_body`. The
    /// `all_jitcodes[i].index == i` invariant (RPython codewriter.py:80)
    /// is guaranteed by `CallControl::collect_jitcodes_in_alloc_order`.
    ///
    /// `verbose` threads through to `transform_graph_to_jitcode`'s debug
    /// branch; upstream `codewriter.py make_jitcodes(verbose=False)`
    /// treats `False` as the default, matching pyre's call site in
    /// `make_jitcodes` which does not expose the knob to callers yet.
    pub fn drain_pending_graphs(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) {
        // RPython: for graph, jitcode in self.callcontrol.enum_pending_graphs():
        //            self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
        //
        // RPython's enum_pending_graphs() pops from unfinished_graphs (LIFO).
        // During transform, new graphs may be discovered and added via
        // get_jitcode(). We pop one at a time to match RPython's yield semantics.
        let profile = std::env::var_os("MAJIT_PROFILE_DRAIN").is_some();
        let mut drain_count = 0usize;
        let drain_start = std::time::Instant::now();
        loop {
            let Some((path, jitcode)) = callcontrol.enum_pending_graphs() else {
                break;
            };
            let iter_start = std::time::Instant::now();
            // Spine B (opname-dispatch convergence): rtyper low-level helper
            // graphs are registered as opname `SpaceOperation`s and have no
            // rich-`OpKind` twin in `function_graphs`, so they are routed here
            // before the lookup below (which they intentionally miss).
            // `take_opname_graph` is empty for the rich-`OpKind` corpus, so
            // this branch is never taken for a Spine-A graph.
            if let Some(flow_graph) = callcontrol.take_opname_graph(&path) {
                // RPython `codewriter.py:80` index = len(all_jitcodes).
                let index = callcontrol.finished_jitcodes_len();
                self.transform_opname_graph_to_jitcode(
                    &flow_graph,
                    &path,
                    callcontrol,
                    &jitcode,
                    /* verbose = */ false,
                    index,
                );
                callcontrol.finish_jitcode(jitcode.clone());
                // Helper graphs are never portal graphs, so no
                // `jitdriver_sd` wiring is needed (Spine A handles that).
                if profile {
                    drain_count += 1;
                    let elapsed = iter_start.elapsed().as_secs_f64();
                    if elapsed >= 0.5 {
                        eprintln!(
                            "[MAJIT_PROFILE_DRAIN] graph #{:>3} {:>7.3}s name={}",
                            drain_count, elapsed, jitcode.name,
                        );
                    }
                }
                continue;
            }
            let Some(graph) = callcontrol
                .function_graphs()
                .get(&path)
                .map(|g| FunctionGraph::clone(&g))
            else {
                // RPython `enum_pending_graphs` (codewriter.py)
                // never yields a jitcode whose graph is missing —
                // `get_jitcode()` only allocates shells for paths that
                // already live in `function_graphs`. Pyre keeps that
                // invariant by routing
                // `handle_regular_call` through the qualified
                // `CallControl::target_to_path` (in `jtransform.rs`).
                // If this branch fires, a new producer has been added
                // that bypasses `target_to_path` or inserts under an
                // alias key. Producer-side bug, not an expected
                // runtime condition.
                panic!(
                    "drain_pending_graphs: jitcode shell has no matching graph — \
                     path={:?} idx={:?} name={:?}. Producer allocated a jitcode \
                     under a path that `function_graphs` does not contain. \
                     Fix the producer (prefer `CallControl::target_to_path`) \
                     instead of silently skipping.",
                    path.segments,
                    jitcode.try_index(),
                    jitcode.name,
                );
            };
            // RPython `codewriter.py:80` passes `index = len(all_jitcodes)`,
            // i.e. the slot this jitcode will occupy AFTER the append on
            // line 81. `finished_jitcodes_len()` is pyre's equivalent read.
            let index = callcontrol.finished_jitcodes_len();
            self.transform_graph_to_jitcode(
                &graph,
                &path,
                callcontrol,
                config,
                &jitcode,
                /* verbose = */ false,
                index,
            );
            // RPython `codewriter.py CodeWriter.make_jitcodes all_jitcodes.append(jitcode)`.
            // `transform_graph_to_jitcode` already set `jitcode.index`.
            callcontrol.finish_jitcode(jitcode.clone());

            // RPython call.py:148: jd.mainjitcode.jitdriver_sd = jd
            for jd in callcontrol.jitdrivers_sd() {
                if jd.portal_graph == path {
                    jitcode.set_jitdriver_sd(jd.index);
                }
            }
            if profile {
                drain_count += 1;
                let elapsed = iter_start.elapsed().as_secs_f64();
                if elapsed >= 0.5 {
                    eprintln!(
                        "[MAJIT_PROFILE_DRAIN] graph #{:>3} {:>7.3}s name={}",
                        drain_count, elapsed, jitcode.name,
                    );
                }
            }
        }
        if profile {
            eprintln!(
                "[MAJIT_PROFILE_DRAIN] DRAIN TOTAL {:>7.3}s  {} graphs",
                drain_start.elapsed().as_secs_f64(),
                drain_count,
            );
            crate::codewriter::transform_profile::dump_transform_phase_totals();
        }
    }

    /// Process all pending graphs and finalize.
    ///
    /// RPython codewriter.py:79-85: enum_pending_graphs loop + finished.
    pub fn make_jitcodes_pending(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> AllJitCodes {
        self.drain_pending_graphs(callcontrol, config);
        self.assembler.finished(&callcontrol.callinfocollection);
        let in_order = callcontrol.collect_jitcodes_in_alloc_order();
        let by_path = callcontrol.jitcodes().clone();
        AllJitCodes { by_path, in_order }
    }
}

impl Default for CodeWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Codewriter-time producer for `CallTarget::Method.resolved_path`.
///
/// Walks every unstamped Method op in `graph`, looks up the receiver's
/// annotated Variable via `value_to_var`, extracts
/// `SomeInstance.classdef` when present, resolves the impl_type name
/// from the MethodDesc's selfclassdef, builds
/// `CallPath::for_impl_method` and stamps it as `resolved_path`.
/// call.py `getfunctionptr(graph)` — consumer does
/// `function_graphs.get(&path)` directly.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn stamp_classdef_hints_on_graph(
    graph: &mut FunctionGraph,
    value_to_var: &crate::translator::rtyper::flowspace_adapter::LegacyToTyped,
) {
    use crate::annotator::model::SomeValue;
    use crate::model::{CallTarget, OpKind};
    let mut stamps: Vec<(usize, usize, String, String)> = Vec::new();
    for (b_idx, block) in graph.blocks.iter().enumerate() {
        for (o_idx, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target:
                    CallTarget::Method {
                        resolved_path: None,
                        fun_decl_id: None,
                        name: method_name,
                        ..
                    },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            let Some(receiver) = args.first() else {
                continue;
            };
            let Some(annotated_var) = value_to_var.get(receiver) else {
                continue;
            };
            let annotation = annotated_var.annotation.borrow();
            let Some(rc_someval) = annotation.as_ref() else {
                continue;
            };
            let SomeValue::Instance(inst) = &**rc_someval else {
                continue;
            };
            let Some(classdef_rc) = &inst.classdef else {
                continue;
            };
            // Resolve impl_type from the MethodDesc selfclassdef.
            // `getmethoddesc_for_attribute` returns the filtered desc
            // set (classdesc.py lookup_filter); in pyre's
            // closed world the receiver SomeInstance carries a single
            // concrete classdef, so the set binds to one selfclassdef
            // and `.next()` selects it. A multi-desc PBC set has no
            // representation in `CallTarget::Method` (carries one path)
            // — that polymorphic case is the §M3 annotator
            // monomorphization boundary, not reachable here.
            // Fallback to receiver classdef.name when:
            //   1. Bookkeeper weak-ref upgrade fails (fixture).
            //   2. getmethoddesc_for_attribute returns empty Vec.
            //   3. lookup_classdef cannot resolve selfclassdef key.
            let impl_type = if let Some(bk) = classdef_rc.borrow().bookkeeper.upgrade() {
                match bk
                    .getmethoddesc_for_attribute(classdef_rc, method_name)
                    .into_iter()
                    .next()
                {
                    Some(md) => {
                        let selfclassdef = md.borrow().selfclassdef;
                        match selfclassdef.and_then(|sc| bk.lookup_classdef(sc)) {
                            Some(cd) => cd.borrow().name.clone(),
                            None => classdef_rc.borrow().name.clone(),
                        }
                    }
                    None => classdef_rc.borrow().name.clone(),
                }
            } else {
                classdef_rc.borrow().name.clone()
            };
            stamps.push((b_idx, o_idx, method_name.clone(), impl_type));
        }
    }
    for (b_idx, o_idx, method_name, impl_type) in stamps {
        let normalized = impl_type.replace('.', "::");
        let path = crate::parse::CallPath::for_impl_method(&normalized, &method_name);
        let op = &mut graph.blocks[b_idx].operations[o_idx];
        if let OpKind::Call {
            target: CallTarget::Method { resolved_path, .. },
            ..
        } = &mut op.kind
        {
            *resolved_path = Some(path);
        }
    }
}

/// Non-void start-block inputarg kinds. Used only when the path has no
/// registered function type; `get_jitcode_calldescr` otherwise reads
/// `FUNC.ARGS`.
fn cfg_non_void_arg_classes(graph: &FunctionGraph) -> String {
    let start_block = graph.block(graph.startblock);
    let mut arg_classes = String::new();
    for (index, arg) in start_block.inputargs.iter().enumerate() {
        use crate::model::ConcreteType;
        let class = match FunctionGraph::concretetype_of(arg) {
            ConcreteType::Signed => 'i',
            ConcreteType::GcRef => 'r',
            ConcreteType::Float => 'f',
            ConcreteType::Void => continue,
            ConcreteType::Unknown => panic!(
                "{}: start-block inputarg {index} has no concrete type, \
                 so `getkind` has no kind to project",
                graph.name
            ),
        };
        arg_classes.push(class);
    }
    arg_classes
}

/// `call.py` `get_jitcode_calldescr` result kind is `FUNC.RESULT`.
/// When the declaration is present it must equal the CFG return colour.
fn func_result_kind(graph_name: &str, declared: Option<char>, cfg_kind: char) -> char {
    match declared {
        Some(declared) if declared == cfg_kind => declared,
        Some(declared) => panic!(
            "graph {graph_name} declared FUNC.RESULT={declared} but CFG return kind is {cfg_kind}"
        ),
        None => cfg_kind,
    }
}

/// Mirror of `FUNC.RESULT` in `rpython/jit/codewriter/call.py` `CallControl.get_jitcode_calldescr`.
///
/// Upstream reads `lltype.typeOf(fnptr).TO.RESULT` from the function
/// pointer type; the graph-level surface is
/// `flowspace/model.py` `graph.returnblock = Block([return_var])`,
/// where `return_var.concretetype` carries the same information.
/// Pyre reads `graph.concretetype(returnblock.inputargs[0])`,
/// which routes straight to the backing
/// [`crate::flowspace::model::Variable::concretetype`] cell carried
/// inline on the returnblock inputarg Variable.  The Variable IS the type source.
/// Links whose target is `returnblock`. Zero means the body never
/// returns: every path raises or aborts.
fn returnblock_predecessor_count(graph: &FunctionGraph) -> usize {
    let ret = graph.returnblock;
    graph
        .blocks
        .iter()
        .map(|block| block.exits.iter().filter(|link| link.target == ret).count())
        .sum()
}

fn graph_result_kind(graph: &FunctionGraph) -> char {
    let returnblock = graph.block(graph.returnblock);
    let Some(arg) = returnblock.inputargs.first() else {
        return 'v';
    };
    use crate::model::ConcreteType;
    match FunctionGraph::concretetype_of(arg) {
        ConcreteType::Signed => 'i',
        ConcreteType::GcRef => 'r',
        ConcreteType::Float => 'f',
        ConcreteType::Void => 'v',
        // Same reason as the inputarg projection above: `'v'` is the kind of
        // a function that returns nothing, so answering it for an unresolved
        // type tells every caller to discard a result the callee does
        // produce.  Never taken over the image — the four classified arms
        // answer 4096 `r`, 906 `i`, 552 `v` and 30 `f`.
        ConcreteType::Unknown => panic!(
            "{}: returnblock inputarg has no concrete type, so `getkind` has \
             no kind to project",
            graph.name
        ),
    }
}

#[cfg(test)]
mod result_kind_tests {
    use super::*;
    use crate::model::ConcreteType;

    /// The refusal above is never taken over the built image, so this is what
    /// keeps it armed: a graph still carrying the pre-rtyper `returnvar` has
    /// no `concretetype` cell set, which is the same `Unknown` an unresolved
    /// front-end type would present.
    #[test]
    #[should_panic(expected = "returnblock inputarg has no concrete type")]
    fn an_unresolved_return_type_is_refused() {
        let graph = FunctionGraph::new("unresolved_return");
        let _ = graph_result_kind(&graph);
    }

    /// Control for the above: the same shape with the cell set projects the
    /// kind rather than refusing, so the test pair separates "no kind" from
    /// "this graph shape".
    #[test]
    fn a_resolved_return_type_is_projected() {
        let mut graph = FunctionGraph::new("resolved_return");
        let var = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let returnblock = graph.returnblock;
        graph.block_mut(returnblock).inputargs = vec![var];
        assert_eq!(graph_result_kind(&graph), 'i');
    }

    #[test]
    #[should_panic(expected = "declared FUNC.RESULT=r but CFG return kind is i")]
    fn func_result_kind_disagreement_is_a_hard_error() {
        let _ = func_result_kind("unwrap_cell", Some('r'), 'i');
    }

    #[test]
    fn func_result_kind_uses_the_function_type_when_it_matches() {
        assert_eq!(func_result_kind("g", Some('r'), 'r'), 'r');
        assert_eq!(func_result_kind("g", None, 'i'), 'i');
    }

    /// `|| missing_hash_hook()` aborts. Nothing links to `returnblock`,
    /// so the synthetic return var is not `FUNC.RESULT`.
    #[test]
    fn an_aborting_body_has_no_return_predecessor() {
        let mut graph = FunctionGraph::new("call_once");
        let entry = graph.startblock;
        graph.set_raise_implicit(entry, "AssertionError");
        assert_eq!(returnblock_predecessor_count(&graph), 0);
        let ret = graph.alloc_value_var_with_type(ConcreteType::Signed);
        graph.set_return(entry, Some(ret));
        assert_eq!(returnblock_predecessor_count(&graph), 1);
    }
}

#[cfg(test)]
mod stamp_classdef_hints_tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::classdesc::{ClassDef, ClassDesc};
    use crate::annotator::description::ClassDefKey;
    use crate::annotator::model::{SomeInstance, SomeValue};
    use crate::flowspace::model::{HostObject, Variable};
    use crate::model::{CallTarget, OpKind, ValueType};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn make_classdef(name: &str) -> Rc<RefCell<ClassDef>> {
        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class(name, vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(&bk, pyobj, name.into())));
        ClassDef::new(&bk, &desc)
    }

    /// End-to-end producer test: a Call(Method) op whose receiver
    /// carries an annotated `SomeInstance(classdef=PyFrame)` gets a
    /// Producer stamps `resolved_path` on the Method op.
    #[test]
    fn stamp_classdef_hints_on_graph_stamps_resolved_path() {
        let classdef = make_classdef("PyFrame");

        // Synthetic flowspace Variable annotated SomeInstance(PyFrame).
        let recv_var = Variable::new();
        *recv_var.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ))));

        let mut graph = FunctionGraph::new("producer_test");
        let _result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("push_value", Some("H".to_string())),
                    args: crate::model::call_args(vec![recv_var.clone()]),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .expect("push_op_var should succeed");
        let mut value_to_var = std::collections::HashMap::new();
        value_to_var.insert(recv_var.clone(), recv_var.clone());

        stamp_classdef_hints_on_graph(&mut graph, &value_to_var);

        let op = &graph.blocks[graph.startblock.0].operations[0];
        let OpKind::Call { target, .. } = &op.kind else {
            panic!("expected Call op");
        };
        let expected_path = crate::parse::CallPath::for_impl_method("PyFrame", "push_value");
        assert_eq!(target.resolved_path(), Some(&expected_path));
    }

    /// Receivers whose annotation is `None` (annotator did not bind
    /// them) leave the hint untouched and do not poison the
    /// side-table. The string-keyed heuristic still runs for those
    /// sites until slices C4-C6 retire it.
    #[test]
    fn stamp_classdef_hints_on_graph_skips_unannotated_receiver() {
        let recv_var = Variable::new();
        // No annotation bound.
        let mut graph = FunctionGraph::new("producer_test");
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("push_value", Some("H".to_string())),
                    args: crate::model::call_args(vec![recv_var.clone()]),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .expect("push_op_var should succeed");
        let mut value_to_var = std::collections::HashMap::new();
        value_to_var.insert(recv_var.clone(), recv_var.clone());

        stamp_classdef_hints_on_graph(&mut graph, &value_to_var);

        let op = &graph.blocks[graph.startblock.0].operations[0];
        let OpKind::Call { target, .. } = &op.kind else {
            panic!("expected Call op");
        };
        assert_eq!(target.resolved_path(), None);
    }

    /// When the receiver classdef's `attrs[method_name].s_value` carries
    /// a `SomePBC` with a `MethodDesc`, the producer routes the receiver
    /// classdef + method name through `Bookkeeper.getmethoddesc_for_attribute`
    /// so the upstream-orthodox `bookkeeper.methoddescs` cache (per
    /// `bookkeeper.py getmethoddesc`) is populated. The cache
    /// is the real PyPy structure; pyre stores no separate
    /// MethodDescKey side-table on CallControl.
    #[test]
    fn stamp_classdef_hints_on_graph_primes_bookkeeper_methoddescs() {
        use crate::annotator::bookkeeper::MethodDescKey;
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::{DescEntry, FunctionDesc};
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;

        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class("PyFrame", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "PyFrame".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        bk.register_classdef(classdef.clone());
        let classdef_key = ClassDefKey::from_classdef(&classdef);

        // Seed the realistic upstream pattern: `attrs[push_value]`
        // carries an *unbound* MethodDesc (selfclassdef = None). PyPy
        // attaches the unbound carrier to the class dict at class-
        // definition time, then `bookkeeper.py` getdesc binds it
        // to the receiver on every access. The producer must perform
        // the same bind via `getmethoddesc_for_attribute`.
        let funcdesc = crate::annotator::description::FuncDescEntry::plain(Rc::new(RefCell::new(
            FunctionDesc::new(
                bk.clone(),
                None,
                "push_value",
                Signature::new(vec!["self".into(), "v".into()], None, None),
                None,
                None,
            ),
        )));
        let unbound = bk.getmethoddesc(
            &funcdesc,
            classdef_key,
            None,
            "push_value",
            std::collections::BTreeMap::new(),
        );
        assert_eq!(
            unbound.borrow().selfclassdef,
            None,
            "test fixture seeds an unbound MethodDesc",
        );
        let pbc = SomePBC::new([DescEntry::Method(unbound.clone())], false);
        let mut attr = Attribute::new("push_value");
        attr.s_value = SomeValue::PBC(pbc);
        classdef
            .borrow_mut()
            .attrs
            .insert("push_value".into(), attr);

        // Receiver Variable annotated SomeInstance(PyFrame).
        let recv_var = Variable::new();
        *recv_var.annotation.borrow_mut() = Some(Rc::new(SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        ))));

        let mut graph = FunctionGraph::new("producer_test");
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("push_value", Some("H".to_string())),
                    args: crate::model::call_args(vec![recv_var.clone()]),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .expect("push_op_var should succeed");
        let mut value_to_var = std::collections::HashMap::new();
        value_to_var.insert(recv_var.clone(), recv_var.clone());

        stamp_classdef_hints_on_graph(&mut graph, &value_to_var);

        // Producer routed the receiver classdef + method name through
        // `getmethoddesc_for_attribute`, which executed the upstream
        // bind path: `getmethoddesc(funcdesc, originclassdef,
        // selfclassdef=receiver_classdef, name)`. The cache now
        // contains the *bound* entry (selfclassdef = Some(receiver_key)).
        let bound_key = MethodDescKey {
            funcdesc_id: funcdesc.desc_key(),
            originclassdef: classdef_key,
            selfclassdef: Some(classdef_key),
            name: "push_value".into(),
            flags: Vec::new(),
        };
        let cached_bound = bk
            .methoddescs
            .borrow()
            .get(&bound_key)
            .cloned()
            .expect("producer should prime bookkeeper.methoddescs with the bound entry");
        assert_eq!(
            cached_bound.borrow().selfclassdef,
            Some(classdef_key),
            "primed entry must be bound to the receiver classdef",
        );
        assert!(
            !Rc::ptr_eq(&cached_bound, &unbound),
            "bound entry is a distinct MethodDesc rc from the seeded unbound carrier",
        );

        // resolved_path stamped directly on the op.
        let op = &graph.blocks[graph.startblock.0].operations[0];
        let OpKind::Call { target, .. } = &op.kind else {
            panic!("expected Call op");
        };
        let expected_path = crate::parse::CallPath::for_impl_method("PyFrame", "push_value");
        assert_eq!(target.resolved_path(), Some(&expected_path));
    }
}
