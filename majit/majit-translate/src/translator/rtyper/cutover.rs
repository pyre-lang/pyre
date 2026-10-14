//! `specialize_legacy_graph` cutover wrapper.
//!
//! Drives the real `RPythonTyper::specialize` against a pyre
//! `model::FunctionGraph` by way of the
//! [`crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace`]
//! adapter, then projects each per-`Variable` `LowLevelType` back to a
//! `ConcreteType` keyed by the legacy `Variable` identity.
//!
//! ## Why this file is in `translator/rtyper/`
//!
//! The legacy algorithms live alongside this file as
//! `translator/rtyper/legacy_{annotator,resolve,pipeline}.rs`.  This
//! cutover module is the bridge between them: the dual-gate
//! comparison and the production type-state path both call into the
//! entry points defined here.  The `flowspace_adapter`
//! sibling continues bridging pyre's surface-DSL
//! `model::FunctionGraph` to the RPython `flowspace::FunctionGraph`
//! shape the rtyper consumes, until pyre's
//! `parse → front → SemanticProgram` chain learns to emit
//! `flowspace::FunctionGraph` directly.
//!
//! ## Scope
//!
//! - Build a `FlowspaceAdapterOutput`, then run the ordinary
//!   `TranslationContext.buildflowgraph` simplification stage. Native MIR
//!   simplification is not a substitute for simplifying the lifted graph.
//! - Queue source-typed inputs with `addpendingblock` on the CallRegistry's
//!   shared annotator; `complete_pending_blocks` infers the reachable bodies.
//! - In the production two-phase path, finish annotation before specializing
//!   with the shared rtyper. Per-subject recovery and the fused compatibility
//!   path remain migration scaffolding, not the terminal #346 pipeline.
//! - Walk the `value_to_var` map and project each `Variable.concretetype`
//!   to `ConcreteType` (Signed/Float/GcRef/Void/Unknown).
//!
//! ## Dual-gate readiness
//!
//! Today this function still fails on graphs containing followups
//! that are not yet ported (`Call`, `FieldRead`, `ArrayRead`, ...).
//! The anchor corpus will surface which
//! followup is the next priority. `Ref`-typed operands now route
//! through `valuetype_to_someshell(Ref) → SomeInstance(classdef=None)`
//! (`codewriter/annotation_state.rs:69`), so the rtyper picks
//! `getinstancerepr(rtyper, None, Gc) → InstanceRepr::new_rootinstance
//! → Ptr(GcStruct(OBJECT))` and the projection collapses to
//! `ConcreteType::GcRef` matching the legacy resolver — the previous
//! `SomeObject` blocker is closed.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::codewriter::type_state::{ConcreteType, valuetype_to_concrete};
use crate::flowspace::argument::Signature;
use crate::flowspace::model::{
    AlwaysInline, Block, BlockRefExt, ConstValue, Constant, GraphFunc, Hlvalue, Link, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::model::FunctionGraph as LegacyGraph;
use crate::translator::rtyper::call_registry::{CallRegistry, FunctionEntry, FunctionPathKey};
use crate::translator::rtyper::call_registry::{
    exception_object_result_annotation, is_exception_object_materializer,
};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::flowspace_adapter::{
    FlowspaceAdapterOutput, LegacyToTyped, LegacyToTypedCandidates,
};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// The `graph.return_type` marker the front-end stamps on a
/// `dont_look_inside` callee that returns `*mut PyObject` (a
/// `PyObjectRef`).  `front::mir::output_type_is_objectptr` gates the
/// stamp (`front::mir` `SemanticFunctionHeader::new`), so this string only ever
/// labels a genuine object-pointer return; the residual prefill maps it
/// to `OBJECTPTR` so the residualized constructor reports a `Ref` result
/// rather than the `None`→`Void` default.  The literal doubles as the
/// readable Rust return type.
pub(crate) const OBJECTPTR_RETURN_TYPE: &str = "*mut PyObject";

/// Residual FUNC.RESULT token for the opaque root-stack word used by
/// RPython's GC transformer.  It is ref-kind like OBJECTPTR but retains the
/// `Ptr(GcOpaque("GCREF"))` low-level identity required for
/// `cast_opaque_ptr` back to StringRepr / InstanceRepr.
pub(crate) const GCREF_RETURN_TYPE: &str = "gcref";

/// Residual FUNC.RESULT token for an rstr `STR` result (`ValueType::Str`):
/// ref-kind, annotated `SomeString` and lowered `Ptr(STR)`.
pub(crate) const STR_RETURN_TYPE: &str = "str";

/// Project a post-`specialize` `LowLevelType` back to the legacy
/// `ConcreteType` bucket the codewriter consumes (Signed / Float /
/// GcRef / Void).
///
/// `Result`-routing wrapper around [`crate::model::getkind`] for
/// the `specialize_legacy_graph` callback channel.
///
/// `getkind` itself panics on the `NotImplementedError` cases
/// (longlonglong / longfloat / non-pointer aggregates / InteriorPtr
/// on a Variable's concretetype) — the upstream `history.py:62,70
/// raise NotImplementedError` shape.  Pyre's legacy specialize
/// callers want those failures routed through `Result<…, TyperError>`
/// instead so a single unported rtype path doesn't unwind the whole
/// `transform_graph_to_jitcode` driver; this wrapper catches the
/// `getkind: …not supported…` payload only and re-raises everything
/// else (so an assertion or logic bug inside `getkind` is NOT
/// silently rebranded as a missing-rtype error in the dual gate).
pub(crate) fn lowleveltype_to_concrete(ll: &LowLevelType) -> Result<ConcreteType, TyperError> {
    let ll_clone = ll.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::model::getkind(&ll_clone)
    }));
    match result {
        Ok(kind) => Ok(kind),
        Err(payload) => {
            // `crate::model::getkind` raises its NotImplementedError-
            // equivalent via `panic!("getkind: type … not supported …")`.
            // Match that exact prefix + substring; resume_unwind anything
            // else so we don't disguise real bugs as missing rtype ops.
            let msg = payload
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            if msg.starts_with("getkind:") && msg.contains("not supported") {
                Err(TyperError::missing_rtype_operation(format!(
                    "lowleveltype_to_concrete: type {ll:?} not supported \
                     (history.py:70 raise NotImplementedError)"
                )))
            } else {
                std::panic::resume_unwind(payload)
            }
        }
    }
}

// Seed every cached callee PyGraph's blocks into the annotator's
// `annotated`/`all_blocks` so the rtyper's specialize walk reaches
// them.

// There is no callee-block pre-seed.  The orthodox path
// `pycall -> recursivecall -> addpendingblock` from
// description.py / annrpython.py is reachable
// through the `simple_call_SomeObject` registration; the
// program-wide `compute_at_fixpoint` loop discovers callees from
// subject call sites without any pre-seed.

/// RAII guard that snapshots every live `Variable.annotation` cell
/// reachable from `iter_variables()` and restores it on `Drop`,
/// isolating the dual-gate baseline's `legacy_annotator::annotate`
/// writes from any subsequent reader on the same graph.  See the
/// `dual_gate_check` doc for the failure mode this prevents.
struct LegacyAnnotationGuard {
    snapshot: Vec<(
        crate::flowspace::model::Variable,
        Option<Rc<crate::annotator::model::SomeValue>>,
    )>,
}

impl LegacyAnnotationGuard {
    fn snapshot(graph: &LegacyGraph) -> Self {
        let snapshot = graph
            .iter_variables()
            .into_iter()
            .map(|var| {
                let ann = var.annotation.borrow().clone();
                (var, ann)
            })
            .collect();
        Self { snapshot }
    }
}

impl Drop for LegacyAnnotationGuard {
    fn drop(&mut self) {
        for (var, ann) in self.snapshot.drain(..) {
            *var.annotation.borrow_mut() = ann;
        }
    }
}

/// Run `specialize_legacy_graph` and diff the real path against the
/// legacy walker's `Variable.concretetype` view.
///
/// Returns `Err(message)` when:
///
/// - the real path errors out (typer error from an unported `OpKind`
///   arm), OR
/// - a legacy-known `Variable` is missing / `Unknown` / different in
///   the real path, OR
/// - the real path produced a definite kind for a `Variable` the legacy
///   resolver did not resolve.
///
/// Returns `Ok(())` only when the legacy `legacy_graph.concretetype`
/// view and the real-path `Variable.concretetype` projection agree on
/// every definite kind.
///
/// Today this entry survives only as a test helper: production
/// callers go through [`dual_gate_check_with_registry`], which dropped
/// per-graph divergence comparison once legacy was narrowed to the
/// Skip arm.  The legacy-baseline diff stays here so
/// anchor tests can keep validating the LL→Concrete projection.
///
/// `dual_gate_check_with_registry` wraps the baseline in a
/// [`LegacyAnnotationGuard`] so the dual-gate comparison is
/// side-effect-free on `legacy_graph.variable.annotation`.
/// Without the guard the baseline's `legacy_annotator::annotate` would
/// publish the wider legacy lift onto the graph's annotation cells
/// (annotate-write contract); a subsequent real-path pass over the same
/// `legacy_graph` would then see the residue at
/// `flowspace_adapter::seed_variable` (which copies
/// `legacy.variable(vid).annotation` onto the fresh flowspace
/// Variable) and trip `flowin`'s `setbinding: new value does not
/// contain old` monotonicity assertion when the wider legacy seed
/// later gets narrowed.  The guard's `Drop` restores the pre-baseline
/// `.annotation` slot contents on every exit path (success, panic,
/// or early `Err` return).
/// Outcome of a dual-gate run.  Distinguishes a real-path success
/// (`Match` — production consumes `real_state` authoritatively) from
/// a known-unported feature (`Skip(reason)` — production falls back
/// to the legacy walker for the affected graph).
///
/// `Match` carries the real path's `LegacyToTyped` map (each
/// `Variable.concretetype` cell set by `RPythonTyper::specialize`) so
/// production callers consume it directly.  The real path is the
/// authoritative producer; legacy is the fallback for
/// Skip-classified graphs.  The legacy-baseline diff inside
/// `dual_gate_check_with_registry` still runs while the transition is
/// active; `Match` means the real path succeeded and matched the legacy
/// baseline when that baseline could be produced. PyPy `codewriter.py transform_graph_to_jitcode`
/// consumes the rtyper-produced graph directly, with no dual-gate
/// equivalent; pyre's `Skip` arm is transitional scaffolding that
/// retires once every category in
/// `is_known_unported`'s table is implemented.
#[derive(Debug)]
pub(crate) enum DualGateOutcome {
    /// Real path completed without panicking and (when the legacy
    /// walker also succeeded as defensive baseline) every legacy-
    /// known `Variable` carried the same `ConcreteType` in the real
    /// path's projection.  Production consumes `real_state`
    /// authoritatively.  Per-`Variable` diff against the legacy
    /// baseline runs whenever the legacy walker can produce a
    /// state — divergence routes the graph through
    /// `Skip("dual-gate divergence: ...")` so the codewriter falls
    /// back to the legacy walker output for the affected graph.  PyPy
    /// `codewriter.py transform_graph_to_jitcode` consumes the rtyper-produced graph
    /// directly with no comparison stage; the legacy-baseline diff
    /// is pyre-only scaffolding that retires once the legacy walker
    /// itself retires.
    Match {
        /// `legacy Variable → typed flowspace::Variable`
        /// (`LegacyToTyped = HashMap<Variable, Variable>`) mapping built
        /// by the flowspace adapter.  Each typed Variable carries the
        /// `RPythonTyper`-set `concretetype` inline (`flowspace/
        /// model.py`), so codewriter callers copy that lltype onto
        /// the matching legacy Variable via
        /// [`crate::codewriter::type_state::apply_from_flowspace_variables`];
        /// `FunctionGraph::concretetype_of(&v)` then reads the legacy
        /// Variable's `concretetype` cell directly.
        real_value_to_var: LegacyToTyped,
        /// Constant representatives, including eliminated phi carriers.
        /// Publish only after the legacy comparison, alongside Variable types.
        real_constants: HashMap<Variable, LowLevelType>,
    },
    /// Real path failed on a known-unported feature — the gate
    /// cannot validate this graph yet but the failure is *not* a
    /// ConcreteType divergence.  Categories include:
    ///
    /// - `OpKind::Call::FunctionPath { segments, .. }` not in the
    ///   registry (cross-crate / primitive paths the production
    ///   walker doesn't reach yet).
    /// - `undefined operand` from cross-block locals
    ///   threading not yet covered.
    /// - `unimplemented operation` from a not-yet-ported rtyper op
    ///   (e.g. `direct_call` for graphs the rpbc port doesn't
    ///   cover).
    /// - `checkgraph` / flowspace consistency panics from the
    ///   adapter's downstream stages.
    Skip(String),
}

/// Production dual-gate — registry-aware entry.
///
/// Drives the real path via [`specialize_legacy_graph_with_registry_returning_value_to_var`]
/// against a pre-populated `CallRegistry` so graphs with
/// `OpKind::Call::FunctionPath` callsites resolve through the upstream
/// `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py`).  Production callers
/// (codewriter.rs `transform_graph_to_jitcode`) build the registry
/// once per `CallControl` (program-wide) and reuse across every
/// dual-gated graph.
///
/// Returns:
///
/// - `Ok(DualGateOutcome::Match)` when the real path succeeds.  The
///   real path's `LegacyToTyped` map (with each
///   `Variable.concretetype` cell populated) is the authoritative
///   source for production consumption.
/// - `Ok(DualGateOutcome::Skip(reason))` when the real path failed
///   on a known-unported feature (registry miss / adapter
///   invariant break / unimplemented rtyper op).  Callers fall back
///   to legacy walker output for the affected graph.
/// - `Err(message)` when the real path failed for an unrecognised
///   reason — surfaces upstream so the codewriter can re-classify
///   the panic if the message later turns out to be in the
///   known-unported table.
///
/// Defensive per-`Variable` diff against the legacy walker baseline
/// runs whenever both paths succeed.  Divergence is reported as
/// `Skip("dual-gate divergence: ...")` so the codewriter falls back
/// to the legacy walker output.  Anchor tests use [`dual_gate_check`]
/// directly for hand-built fixtures.
pub(crate) fn dual_gate_check_with_registry(
    legacy_graph: &LegacyGraph,
    call_registry: &CallRegistry,
    lift_sources: &crate::codewriter::call::GraphStore,
) -> Result<DualGateOutcome, String> {
    // Same panic-catch contract as `dual_gate_check` — the rtyper's
    // internal `genop`/`level` asserts surface as diagnostic panics
    // for unported pyre-front idioms; the gate uniformly returns a
    // stringified error so the env-flag wrapper can decide whether
    // to panic, log, or skip.
    //
    // Snapshot the fixed-graph keyset so failure repair can identify shared
    // callees newly fixed by this subject. Previously-annotated block cells
    // are journaled lazily by AddedBlocksGuard only when actually touched.
    let session_at_entry = SubjectSessionSnapshot::capture(call_registry);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drive_subject(legacy_graph, call_registry, true)
    }));
    let (graph, mut real_value_to_var, mut real_constants) = match result {
        Ok(Ok((graph, value_to_var, _, constants, _))) => (graph, value_to_var, constants),
        Ok(Err(e)) => {
            unpoison_failed_subject_callees(call_registry, &session_at_entry, lift_sources);
            let msg = format!("{e}");
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path failed: {msg}"));
        }
        Err(payload) => {
            unpoison_failed_subject_callees(call_registry, &session_at_entry, lift_sources);
            let msg = crate::annotator::model::panic_payload_text(&*payload)
                .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
            if is_known_unported(&msg) {
                return Ok(DualGateOutcome::Skip(msg));
            }
            return Err(format!("real path panicked: {msg}"));
        }
    };
    let _annotation_guard = LegacyAnnotationGuard::snapshot(legacy_graph);
    let baseline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::legacy_annotator::annotate(legacy_graph);
        super::legacy_resolve::resolve_types(legacy_graph);
    }));
    if let Err(payload) = baseline {
        let msg = crate::annotator::model::panic_payload_text(&*payload)
            .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
        return Err(format!(
            "dual-gate baseline panicked (legacy walker crashed before \
             comparison could run — comparison cannot validate the real \
             path's result): {msg}"
        ));
    }
    let followed_legacy = call_registry.session_if_started().map(|(annotator, _)| {
        annotator_followed_legacy_vars(&annotator, &graph, &real_value_to_var)
    });
    let excluded_legacy = followed_legacy
        .as_ref()
        .map(|followed| annotator_unfollowed_legacy_vars(&real_value_to_var, followed));
    if let Some(divergence) = compare_real_against_legacy(
        &real_value_to_var,
        &real_constants,
        legacy_graph,
        excluded_legacy.as_ref(),
    ) {
        return Ok(DualGateOutcome::Skip(format!(
            "dual-gate divergence: {divergence}"
        )));
    }
    // Compare unpaired vars first. Only if nothing was followed and
    // nothing unpaired diverged is the compared set empty — a Match
    // here would be a silent no-op.
    if followed_legacy
        .as_ref()
        .is_some_and(|followed| followed.is_empty())
    {
        return Ok(DualGateOutcome::Skip(
            "dual-gate compared set empty".to_string(),
        ));
    }
    // `follow_link` records `links_followed` only for an edge the
    // annotator walks (`annrpython.py`). Phase B still
    // `setconcretetype`s an unfollowed Variable to `Void` after
    // `force_return_var_annotation` seeds `s_ImpossibleValue`.
    // Publishing that twin would replace the legacy walker's kind
    // (`func_result_kind` of a return, `getcalldescr` NON_VOID_ARGS of
    // a residual call). A followed Variable typed `Void` still
    // diverges above and skips.
    keep_legacy_kind_when_unfollowed_void(
        &mut real_value_to_var,
        &mut real_constants,
        excluded_legacy.as_ref(),
    );
    Ok(DualGateOutcome::Match {
        real_value_to_var,
        real_constants,
    })
}

/// Repair shared callee state poisoned by a failed subject scope.
///
/// `specialize_more_blocks` mutates callee graphs in place: it
/// registers each graph in `annotator.fixed_graphs` at its first
/// specialized block (rtyper.py) and rewrites block operations to
/// LL form. When the subject then fails, `AddedBlocksGuard` evicts the
/// scope's blocks and clears their annotations — but the callee's
/// `FunctionDesc.cache` (and any calltable row built during the scope)
/// still holds the fixed + LL-rewritten + annotation-cleared graph.
/// Every later subject calling the same callee then dies at
/// `addpendingblock`'s fixed-graph safety check (annrpython.py
/// reads `arg.annotation`), or at `flowin`'s "unimplemented operation"
/// on the LL ops.
///
/// Upstream has no counterpart: RPython's `specialize()` runs once per
/// Translator and any failure is fatal. Pyre's per-subject dual-gate
/// continues past failed subjects, so a failed scope must leave shared
/// callee state as if the subject never ran. Each graph newly fixed
/// during the scope is swapped for a fresh re-lift from its
/// `GraphStore` source, and the rtyper/annotator records keyed to the
/// stale graph are dropped:
///
/// - `annotator.fixed_graphs` row (graph identity)
/// - `rtyper.already_seen` rows for the stale graph's blocks
/// - `translator.graphs` entry (appended on `cachedgraph` hit)
/// - `FunctionDesc.cache` values (patched to the fresh `PyGraph`)
/// - calltable rows holding the same `Rc<PyGraph>` (patched in place
///   so `total_calltable_size` and `get_concrete_calltable`'s size
///   invariant hold; the family's cached concrete calltable is
///   dropped so it rebuilds against the fresh graph)
#[derive(Default)]
struct SubjectSessionSnapshot {
    fixed_graphs: HashSet<crate::flowspace::model::GraphKey>,
}

impl SubjectSessionSnapshot {
    fn capture(call_registry: &CallRegistry) -> Self {
        Self::capture_fixed_only(call_registry)
    }

    fn capture_fixed_only(call_registry: &CallRegistry) -> Self {
        let Some((annotator, _)) = call_registry.session_if_started() else {
            return Self::default();
        };
        let fixed_graphs = annotator.fixed_graphs.borrow().keys().cloned().collect();
        Self { fixed_graphs }
    }
}

fn unpoison_failed_subject_callees(
    call_registry: &CallRegistry,
    session_at_entry: &SubjectSessionSnapshot,
    lift_sources: &crate::codewriter::call::GraphStore,
) {
    let Some((annotator, rtyper)) = call_registry.session_if_started() else {
        return;
    };
    let mut stale_graphs: Vec<(
        crate::flowspace::model::GraphKey,
        crate::flowspace::model::GraphRef,
    )> = annotator
        .fixed_graphs
        .borrow()
        .iter()
        .filter(|(k, _)| !session_at_entry.fixed_graphs.contains(*k))
        .map(|(k, g)| (k.clone(), g.clone()))
        .collect();
    // GraphKey is a pointer. Sort by graph name so the re-lift of
    // failed-subject callees is the same run to run.
    stale_graphs.sort_by(|a, b| {
        let ga = a.1.borrow();
        let gb = b.1.borrow();
        ga.name.cmp(&gb.name).then_with(|| {
            ga.func
                .as_ref()
                .map(|f| f.name.as_str())
                .cmp(&gb.func.as_ref().map(|f| f.name.as_str()))
        })
    });

    for (gkey, stale) in stale_graphs {
        annotator.fixed_graphs.borrow_mut().shift_remove(&gkey);
        let stale_blocks = stale.borrow().iterblocks();
        let stale_block_keys: HashSet<crate::flowspace::model::BlockKey> = stale_blocks
            .iter()
            .map(crate::flowspace::model::BlockKey::of)
            .collect();
        for block in &stale_blocks {
            let bkey = crate::flowspace::model::BlockKey::of(block);
            annotator.annotated.borrow_mut().shift_remove(&bkey);
            annotator.all_blocks.borrow_mut().shift_remove(&bkey);
            annotator.blocked_blocks.borrow_mut().shift_remove(&bkey);
            annotator.failed_blocks.borrow_mut().shift_remove(&bkey);
            rtyper.already_seen.borrow_mut().shift_remove(&bkey);
        }
        for generation in annotator.genpendingblocks.borrow_mut().iter_mut() {
            generation.retain(|bkey, _| !stale_block_keys.contains(bkey));
        }
        annotator.blocked_graphs.borrow_mut().shift_remove(&gkey);
        annotator
            .translator
            .graphs
            .borrow_mut()
            .retain(|g| !Rc::ptr_eq(g, &stale));
        let Some((key, entry)) = call_registry.find_entry_with_cached_graph(&stale) else {
            // The subject graph itself (lifted through the adapter, not
            // a registry callee) — the shared-map cleanup above is all
            // it needs.
            continue;
        };
        let fresh = call_registry
            .keys_for_entry(&key)
            .iter()
            .find_map(|k| {
                let path = crate::parse::CallPath::from_segments(k.segments().iter().cloned());
                let source = lift_sources.get(&path)?;
                let signature = lift_sources.signature(&path)?;
                Some((source, signature))
            })
            .and_then(|(source, signature)| {
                lift_callee_to_pygraph(&source, signature.clone(), call_registry).ok()
            });
        match fresh {
            Some(fresh) => {
                let fd = entry.function_desc.borrow();
                for pg in fd.cache.borrow_mut().values_mut() {
                    if Rc::ptr_eq(&pg.graph, &stale) {
                        *pg = fresh.clone();
                    }
                }
                if let Ok(family) = fd.base.getcallfamily() {
                    for table in family.borrow_mut().calltables.values_mut() {
                        for row in table.iter_mut() {
                            for pg in row.values_mut() {
                                if Rc::ptr_eq(&pg.graph, &stale) {
                                    *pg = fresh.clone();
                                }
                            }
                        }
                    }
                    rtyper
                        .concrete_calltables
                        .borrow_mut()
                        .remove(&(Rc::as_ptr(&family) as usize));
                }
            }
            None => {
                // No re-lift source — drop the poisoned cache rows and
                // record the condition so later callers surface a lift
                // error (Skip-classified) instead of the fixed-graph
                // safety panic.
                entry
                    .function_desc
                    .borrow()
                    .cache
                    .borrow_mut()
                    .retain(|_, pg| !Rc::ptr_eq(&pg.graph, &stale));
                entry.record_lift_error(
                    "callee graph poisoned by a failed subject scope; no re-lift source"
                        .to_string(),
                );
            }
        }
    }
}

/// Copy `getargs` annotations from `old` onto `stub`, aligned on the
/// trailing slots when the lifted startblock dropped a Void ZST `self`.
/// Unbound old slots become `SomeImpossibleValue` so `getrinputs` /
/// `bindingrepr` have a hole, not a KeyError (`LLCallTable.lookup`).
fn copy_stub_getargs_annotations(old: &Rc<PyGraph>, stub: &Rc<PyGraph>) {
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::Hlvalue;
    let old_args = old.graph.borrow().getargs();
    let stub_args = stub.graph.borrow().getargs();
    let offset = stub_args.len().saturating_sub(old_args.len());
    for (i, hl) in stub_args.iter().enumerate() {
        let Hlvalue::Variable(v) = hl else {
            continue;
        };
        let src = if i < offset {
            SomeValue::Impossible
        } else {
            match old_args.get(i - offset) {
                Some(Hlvalue::Variable(ov)) => ov
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| (**s).clone())
                    .unwrap_or(SomeValue::Impossible),
                _ => SomeValue::Impossible,
            }
        };
        v.annotation.replace(Some(Rc::new(src)));
    }
}

fn bind_unbound_stub_getargs_impossible(stub: &Rc<PyGraph>) {
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::Hlvalue;
    for hl in stub.graph.borrow().getargs() {
        let Hlvalue::Variable(v) = hl else {
            continue;
        };
        if v.annotation.borrow().is_none() {
            v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        }
    }
}

/// Prefill a skipped subject's default `FunctionDesc` graph with the same
/// signature-only stub `install_source_graph` uses for `@jit.dont_look_inside`.
///
/// `unpoison_failed_subject_callees` re-lifts the skipped body so a later
/// caller that `cachedgraph`s it walks the full graph and fails on a nested
/// wall (`rordereddict.py` `RDict.insert` → `pin_gcrefs`). Upstream never
/// re-flows a graph that failed to annotate; the callsite residualizes
/// (`policy.py look_inside_graph` / `ExtRegistryEntry.compute_result_annotation`).
///
/// Only `GraphCacheKey::None` and this funcdesc's family row are replaced.
/// Other specializations stay; replacing every cache value with one stub
/// made later callers of a working specialization fail as new skip ADDs.
fn residualize_skipped_subject(
    call_registry: &CallRegistry,
    path: &crate::parse::CallPath,
    lift_sources: &crate::codewriter::call::GraphStore,
) {
    let key = FunctionPathKey::from_segments(path.segments.iter().cloned());
    let Some(entry) = call_registry.lookup(&key) else {
        return;
    };
    let Some(legacy) = lift_sources.get(path) else {
        return;
    };
    let Some(signature) = lift_sources.signature(path) else {
        return;
    };
    let semantic_key = legacy
        .source_identity
        .as_deref()
        .map(|identity| FunctionPathKey::from_segments(identity.split("::")))
        .unwrap_or_else(|| key.clone());
    let result_shell = raw_ptr_token_annotation(call_registry, legacy.return_type.as_deref())
        .or_else(|| residual_stub_result_shell(&semantic_key, legacy.return_type.as_deref()))
        .map(|shell| declared_return_annotation(call_registry, legacy.as_ref()).unwrap_or(shell))
        .or_else(|| residual_return_shell(legacy.return_type.as_deref()));
    let Some(result_shell) = result_shell else {
        return;
    };
    let stub = build_stub_pygraph_with_result_shell(legacy.name.clone(), signature, result_shell);
    let rowkey = crate::annotator::description::FunctionDesc::rowkey(&entry.function_desc);
    let fd = entry.function_desc.borrow();
    let old_none = fd
        .cache
        .borrow()
        .get(&crate::annotator::description::GraphCacheKey::None)
        .cloned();
    if let Some(old) = old_none.as_ref() {
        copy_stub_getargs_annotations(old, &stub);
    } else {
        bind_unbound_stub_getargs_impossible(&stub);
    }
    fd.release_source_graph();
    fd.cache.borrow_mut().insert(
        crate::annotator::description::GraphCacheKey::None,
        stub.clone(),
    );
    *fd.lift_error.borrow_mut() = None;
    let Ok(family) = fd.base.getcallfamily() else {
        return;
    };
    replace_none_graph_in_calltables(&family, rowkey, old_none.as_ref(), &stub);
    if let Some((_, rtyper)) = call_registry.session_if_started() {
        rtyper
            .concrete_calltables
            .borrow_mut()
            .remove(&(Rc::as_ptr(&family) as usize));
    }
}

/// `CallFamily.calltable_lookup_row` compares rows by graph identity
/// (`description.py` `existing_row == row`; `FunctionGraph` has no
/// `__eq__`). `find_row` rebuilds the row with `desc.get_graph`, which
/// returns the live `cachedgraph` (`FunctionDesc.specialize`). The None
/// stub is only the default specialization; an AccessDirect copy stays
/// in `FunctionDesc.cache` under `GraphCacheKey::AccessDirect`
/// (`specialize.py` `cachedgraph`). Replacing every family-row graph
/// with the stub made that hop's rebuilt row miss.
fn replace_none_graph_in_calltables(
    family: &Rc<RefCell<crate::annotator::description::CallFamily>>,
    rowkey: crate::annotator::description::DescKey,
    old_none: Option<&Rc<PyGraph>>,
    stub: &Rc<PyGraph>,
) {
    let Some(old) = old_none else {
        return;
    };
    for table in family.borrow_mut().calltables.values_mut() {
        for row in table.iter_mut() {
            if let Some(graph) = row.get(&rowkey)
                && Rc::ptr_eq(graph, old)
            {
                row.insert(rowkey, stub.clone());
            }
        }
    }
}

/// Project the real path's resolved kind per legacy `Variable`, keyed
/// by Variable identity.  `value_to_var` already pairs each legacy
/// Variable with its rtyper-typed twin, and `constant_concretetypes`
/// carries the ground-truth lltype for const-define results, so no slot
/// projection is needed.  Errors from unsupported lltypes are treated as
/// missing entries — `specialize_legacy_graph` already propagates such a
/// failure before the dual-gate runs, so reaching here implies every
/// lltype is projectable.
///
/// Used by both [`dual_gate_check_with_registry`] (via
/// [`compare_real_against_legacy`]) and the [`dual_gate_check`]
/// anchor-test helper to diff the real path against the legacy walker.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn project_value_to_var(
    value_to_var: &LegacyToTyped,
    constant_concretetypes: &HashMap<Variable, LowLevelType>,
) -> HashMap<Variable, ConcreteType> {
    let mut real_state: HashMap<Variable, ConcreteType> = HashMap::new();
    for (legacy_var, typed_var) in value_to_var {
        if let Some(lltype) = typed_var.concretetype().as_ref()
            && let Ok(kind) = lowleveltype_to_concrete(lltype)
        {
            real_state.insert(legacy_var.clone(), kind);
        }
    }
    // `Constant.concretetype` is the ground truth for constant operands;
    // the const-define result Variable overrides the value_to_var entry.
    for (legacy_var, lltype) in constant_concretetypes {
        if let Ok(kind) = lowleveltype_to_concrete(lltype) {
            real_state.insert(legacy_var.clone(), kind);
        }
    }
    real_state
}

/// Select the typed block-local representative of each legacy MIR Variable.
///
/// RPython flowspace gives every block inputarg its own Variable definition,
/// while pyre's pre-adapter MIR can reuse one Variable identity for an op
/// result and the phi columns of several successor blocks. The adapter
/// therefore creates several orthodox typed Variables for one legacy key.
/// Simplification can remove the initially preferred producer while leaving a
/// successor representative live and rtyped. Repoint the projection map to a
/// live representative after Phase B; this is identity reconciliation, not a
/// kind backfill—the rtyper must have positively assigned the lltype.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn select_rtyped_representatives(
    value_to_var: &mut LegacyToTyped,
    candidates: &LegacyToTypedCandidates,
) -> Result<(), TyperError> {
    for (legacy, vars) in candidates {
        if value_to_var
            .get(legacy)
            .is_some_and(|var| var.concretetype.borrow().is_some())
        {
            continue;
        }
        let mut selected: Option<(Variable, ConcreteType)> = None;
        for candidate in vars {
            let Some(lltype) = candidate.concretetype.borrow().clone() else {
                continue;
            };
            let kind = lowleveltype_to_concrete(&lltype)?;
            if let Some((_, selected_kind)) = &selected {
                if *selected_kind != kind {
                    return Err(TyperError::message(format!(
                        "block-local representatives of one legacy Variable disagree: \
                         {selected_kind:?} vs {kind:?} for {legacy:?}"
                    )));
                }
            } else {
                selected = Some((candidate.clone(), kind));
            }
        }
        if let Some((selected, _)) = selected {
            value_to_var.insert(legacy.clone(), selected);
        }
    }
    Ok(())
}

/// Variables defined (inputarg or op result) in the reachable block
/// closure (DFS from `startblock` over `Block.exits`).  The real path
/// converts exactly this closure (flowspace `iterblocks()` parity —
/// an unreachable block cannot exist upstream), while legacy
/// `resolve_types` types every lowered block — including pruned
/// `SwitchInt` abort-default arms and the orphan `set_raise`
/// `[etype, evalue]` Link.args they carry.  A variable defined only
/// outside the closure is legacy-only by construction; diffing it
/// against the real path is a guaranteed false `real=Unknown`.
///
/// One further legacy-only class lives *inside* the structural closure:
/// a non-entry block inputarg that no reachable op or exitswitch reads.
/// Phase B's `transform_dead_op_vars` (simplify.py) drops such dead
/// merge inputargs from the real graph, so the real path never types
/// them, whereas the legacy walker types every block inputarg.  A value
/// only *forwarded* through `Link.args` (framestate reusing one Variable
/// identity for a branch-carried local — `checked_arith`'s dead `opt`
/// phi-thread into the overflow arm) is not a read; the dead-thread is
/// dropped and would otherwise diff as a false `real=Unknown`.  Entry
/// (`startblock`) inputargs are the function signature and are kept even
/// when unread, matching `transform_dead_op_vars`'s signature exemption.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn reachable_defined_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    // Id-keyed lookup, not the dense `blocks[id.0]` projection — block ids
    // need not be index-aligned (`flowspace_adapter::reachable_block_ids`).
    let by_id: std::collections::HashMap<crate::model::BlockId, &crate::model::Block> =
        graph.blocks.iter().map(|b| (b.id, b)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![graph.startblock];
    let mut reachable: Vec<&crate::model::Block> = Vec::new();
    // Vars actually consumed by an op or an exitswitch in the reachable
    // closure.  `Link.args` forwarding is deliberately excluded — a
    // forward-only value is a dead phi-thread, not a read.
    let mut read_vars: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    while let Some(bid) = stack.pop() {
        if !seen.insert(bid) {
            continue;
        }
        let Some(block) = by_id.get(&bid) else {
            continue;
        };
        reachable.push(block);
        for op in &block.operations {
            for v in crate::front::result_exc::op_operand_vars(&op.kind) {
                read_vars.insert(v);
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) => {
                read_vars.insert(v.clone());
            }
            // A fused guard (`jtransform optimize_goto_if_not`) reads its
            // comparison operands directly off the exitswitch, so those vars
            // are live even when no op consumes them.
            Some(crate::model::ExitSwitch::Fused { args, .. }) => {
                for v in args {
                    read_vars.insert(v.clone());
                }
            }
            Some(crate::model::ExitSwitch::LastException) | None => {}
        }
        for link in &block.exits {
            stack.push(link.target);
        }
    }
    let mut vars = std::collections::HashSet::new();
    for block in &reachable {
        // The entry block's inputargs are the function signature and the
        // return / except blocks' inputargs are the graph outputs;
        // `transform_dead_op_vars` keeps all three even when locally unread,
        // so the real path types them — never exclude them from the diff.
        let is_boundary = block.id == graph.startblock
            || block.id == graph.returnblock
            || block.id == graph.exceptblock;
        for ia in &block.inputargs {
            if is_boundary || read_vars.contains(ia) {
                vars.insert(ia.clone());
            }
        }
        for op in &block.operations {
            if let Some(r) = &op.result {
                vars.insert(r.clone());
            }
        }
    }
    vars
}

/// Variables the emit path *colors* — every position where flatten runs
/// `getcolor(v)` / the assembler runs `lookup_coloring(v)` on a value.
/// Upstream `regalloc.perform_register_allocation` colors only the three
/// value kinds (`int`/`ref`/`float`, `regalloc.py:6-8`); a `void` value
/// is never colored and is structurally filtered at each use site.  A
/// var appearing here therefore cannot legitimately be `Void`: it must
/// carry a colorable kind, so the `GcRef`→`Void` refinement accepted for
/// dropped-unit results (see [`collect_divergences`]) is unsound for it.
///
/// The colored positions:
/// - op operands (`serialize_op`/`flatten_list`, `flatten.py`),
/// - the block `exitswitch` (`goto_if_not` / `switch`,
///   `flatten.py/265`, including fused compare operands),
/// - operands forwarded on a Link into the `exceptblock` — the raise
///   value colored by `make_return`'s 2-arg arm (`flatten.py`).
///
/// A var forwarded only into the `returnblock` is *not* included: the
/// 1-arg return arm emits `void_return` without coloring when the kind
/// is void (`flatten.py`), which is exactly the sound dropped-unit
/// case the refinement targets.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn colored_operand_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    let mut operands = std::collections::HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            for v in crate::front::result_exc::op_operand_vars(&op.kind) {
                operands.insert(v);
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) => {
                operands.insert(v.clone());
            }
            Some(crate::model::ExitSwitch::Fused { args, .. }) => {
                for v in args {
                    operands.insert(v.clone());
                }
            }
            Some(crate::model::ExitSwitch::LastException) | None => {}
        }
        for link in &block.exits {
            if link.target != graph.exceptblock {
                continue;
            }
            // `flatten.py make_return`: the 2-arg exception link emits
            // `raise` coloring ONLY `args[1]` (the exception *value* / evalue);
            // `args[0]` (the exception *type* / etype) is not passed to
            // `getcolor`.  Mirror that — color the evalue operand alone, so a
            // `real=Void` refinement on the (uncolored) etype is not falsely
            // treated as a colored divergence.
            if let Some(crate::model::LinkArg::Value(v)) = link.args.get(1) {
                operands.insert(v.clone());
            }
        }
    }
    operands
}

/// Variables that appear more than once within a single block's
/// `inputargs` — duplicate phi columns.  The real path runs
/// `remove_duplicate_inputargs` (`model::remove_duplicate_inputargs`, the
/// `simplify.py remove_identical_vars_SSA` port) before rtyping:
/// when two inputarg columns of one block carry identical phi-args, they
/// collapse to one representative and the removed column's Variable is
/// unioned away, so `setup_block_entry` never types that identity.  The
/// legacy walker keeps both columns as the one Variable and types it, so
/// the dual-gate reads the merged-away column as `real=Unknown` — a false
/// divergence, since the value IS emitted through the surviving
/// representative (same kind).  This mirrors the dead-phi exemption
/// [`reachable_defined_vars`] already documents, for the dedup case.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn duplicate_inputarg_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    let mut dups = std::collections::HashSet::new();
    for block in &graph.blocks {
        let mut seen = std::collections::HashSet::new();
        for ia in &block.inputargs {
            if !seen.insert(ia.clone()) {
                dups.insert(ia.clone());
            }
        }
    }
    dups
}

/// Op-*result* Variables that no reachable op or exitswitch reads — their
/// only consumers are `Link.args` forwardings (dead phi-threads).  This is
/// the op-result analogue of the dead non-entry-block inputarg
/// [`reachable_defined_vars`] already excludes: the real path's
/// `transform_dead_op_vars` (`simplify.py`) drops the result binding of
/// such a value, so `setup_block_entry` never types it, whereas the legacy
/// walker types every op result.
///
/// The concrete hitters are element reads whose value is never observed: a
/// `swap_values` `getarrayitem(localsplus, idx)` on the two write-back
/// positions, or a `store_local_value` `getarrayitem` whose read half is
/// dead (the paired `setarrayitem` is the only live effect).  The read
/// value forwards through phi links into the returnblock but no op consumes
/// it, so `real=None` here is the dead-result artifact, not a kind
/// divergence.
///
/// Reads counted: op operands (`op_operand_vars`, which includes an
/// `ArrayWrite`/`FieldWrite` value operand — a written-back read is live and
/// excluded here) and the exitswitch (Value / fused compare).  `Link.args`
/// forwarding is deliberately NOT counted, matching `reachable_defined_vars`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn dead_op_result_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    use crate::model::{BlockId, ExitSwitch, LinkArg};

    // The legacy MIR can reuse one Variable identity as the producer and as
    // inputargs in multiple successor blocks. RPython flowspace cannot:
    // checkgraph requires each block inputarg to be a distinct definition.
    // Reproduce simplify.py:428-518 over block-local definition nodes rather
    // than collapsing those definitions into one global Variable key.
    let by_id: HashMap<BlockId, &crate::model::Block> =
        graph.blocks.iter().map(|block| (block.id, block)).collect();
    let mut reachable = HashSet::new();
    let mut pending = vec![graph.startblock];
    while let Some(id) = pending.pop() {
        if !reachable.insert(id) {
            continue;
        }
        if let Some(block) = by_id.get(&id) {
            pending.extend(block.exits.iter().map(|link| link.target));
        }
    }

    let mut next_node = 0usize;
    let mut input_nodes: HashMap<(BlockId, usize), usize> = HashMap::new();
    let mut entry_defs: HashMap<BlockId, HashMap<Variable, usize>> = HashMap::new();
    for block in graph
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id) && !block.dead)
    {
        let mut defs = HashMap::new();
        for (column, var) in block.inputargs.iter().enumerate() {
            let node = next_node;
            next_node += 1;
            input_nodes.insert((block.id, column), node);
            // remove_identical_vars_SSA runs before the dead-var pass; keeping
            // the first identity-equivalent column models its representative.
            defs.entry(var.clone()).or_insert(node);
        }
        entry_defs.insert(block.id, defs);
    }

    let mut dependencies: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut read_nodes = HashSet::new();
    let mut exit_defs: HashMap<BlockId, HashMap<Variable, usize>> = HashMap::new();
    let mut removable_results: HashMap<Variable, Vec<usize>> = HashMap::new();

    for block in graph
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id) && !block.dead)
    {
        let mut defs = entry_defs.get(&block.id).cloned().unwrap_or_default();
        if block.id == graph.startblock
            || block.id == graph.returnblock
            || block.id == graph.exceptblock
        {
            // Every terminal input column is observable. The legacy graph can
            // repeat one Variable identity in multiple columns (notably the
            // exceptblock's etype/evalue pair), while RPython SSA gives each
            // column its own definition node.
            read_nodes.extend(
                block
                    .inputargs
                    .iter()
                    .enumerate()
                    .filter_map(|(column, _)| input_nodes.get(&(block.id, column)).copied()),
            );
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for (index, op) in block.operations.iter().enumerate() {
            let operands: Vec<usize> = crate::front::result_exc::op_operand_vars(&op.kind)
                .into_iter()
                .filter_map(|var| defs.get(&var).copied())
                .collect();
            let removable = op_result_can_remove(&op.kind) && Some(index) != raising_op_idx;
            if !removable {
                read_nodes.extend(operands.iter().copied());
            }
            if let Some(result) = &op.result {
                let node = next_node;
                next_node += 1;
                if removable {
                    dependencies
                        .entry(node)
                        .or_default()
                        .extend(operands.iter().copied());
                    removable_results
                        .entry(result.clone())
                        .or_default()
                        .push(node);
                }
                defs.insert(result.clone(), node);
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                if let Some(node) = defs.get(var) {
                    read_nodes.insert(*node);
                }
            }
            Some(ExitSwitch::Fused { args, .. }) => {
                for var in args {
                    if let Some(node) = defs.get(var) {
                        read_nodes.insert(*node);
                    }
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        exit_defs.insert(block.id, defs);
    }

    // simplify.py:461-469: target inputargs depend on corresponding source
    // link args. Backward liveness through this relation keeps a producer iff
    // some live successor column ultimately consumes it.
    for block in graph
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id) && !block.dead)
    {
        let Some(source_defs) = exit_defs.get(&block.id) else {
            continue;
        };
        for link in &block.exits {
            for (column, arg) in link.args.iter().enumerate() {
                let LinkArg::Value(source_var) = arg else {
                    continue;
                };
                let Some(source_node) = source_defs.get(source_var).copied() else {
                    continue;
                };
                if let Some(target_node) = input_nodes.get(&(link.target, column)).copied() {
                    dependencies
                        .entry(target_node)
                        .or_default()
                        .insert(source_node);
                } else {
                    // Link leaves the analyzed closure: upstream marks both
                    // ends read rather than pruning an externally-used value.
                    read_nodes.insert(source_node);
                }
            }
        }
    }

    let mut work: Vec<usize> = read_nodes.iter().copied().collect();
    while let Some(node) = work.pop() {
        if let Some(previous) = dependencies.get(&node) {
            for &dependency in previous {
                if read_nodes.insert(dependency) {
                    work.push(dependency);
                }
            }
        }
    }

    removable_results
        .into_iter()
        .filter_map(|(var, nodes)| {
            nodes
                .iter()
                .all(|node| !read_nodes.contains(node))
                .then_some(var)
        })
        .collect()
}

/// Results of a synthetic `FieldRead("__discriminant")` that feed a block
/// exitswitch — the tag read a Rust `match` lowers to (`front/mir.rs:1184`
/// `Rvalue::Discriminant` → `FieldRead("__discriminant")`, then the switch
/// terminator reads that result as `ExitSwitch::Value`).
///
/// The legacy walker types this tag `Signed` (`ty: Int` at the producer,
/// `legacy_annotator.rs:273`).  The real (annotate→rtype) path lowers the
/// enclosing `Result`/`StepResult` through the uniform exception-transform
/// (`front/result_exc.rs`), which **rebuilds** the discriminant-switch block
/// with its own freshly-minted tag Variable; the projection back onto the
/// legacy graph (`project_value_to_var`) is keyed by Variable identity, so the
/// legacy tag Variable has no twin and reads `real=None` even though the real
/// graph types its own tag `Signed` and emits the switch from it.  This is the
/// same rebuilt-identity artifact as [`duplicate_inputarg_vars`] /
/// [`dead_op_result_vars`], scoped to the switch-feeding discriminant read.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn switch_discriminant_read_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    use crate::model::OpKind;
    // Vars a block exitswitch reads (Value or the fused-compare args).
    let mut switch_read: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    for block in &graph.blocks {
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(v)) => {
                switch_read.insert(v.clone());
            }
            Some(crate::model::ExitSwitch::Fused { args, .. }) => {
                for v in args {
                    switch_read.insert(v.clone());
                }
            }
            _ => {}
        }
    }
    let mut out = std::collections::HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let (Some(r), OpKind::FieldRead { field, .. }) = (&op.result, &op.kind)
                && field.name == "__discriminant"
                && switch_read.contains(r)
            {
                out.insert(r.clone());
            }
        }
    }
    out
}

/// Results of a variant-payload `FieldRead` whose *only* op consumer is the
/// value position of a `FieldWrite` — the extract-then-repack of an
/// identity `match` re-wrap (`match step { Return(v) => Ok(Return(v)),
/// CloseLoop { loop_header_pc } => Ok(CloseLoop { .. }), … }`,
/// `pyopcode.rs`).  Each arm reads the incoming variant's payload
/// (`FieldRead("__pos_0" | "loop_header_pc", owner = StepResult::Variant)`)
/// and immediately writes it into a freshly-built outgoing variant
/// (`FieldWrite(_, owner = StepResult<…>::Variant)`).  A tuple variant names
/// its storage by position, so a re-wrap that carries a value between
/// differently-shaped variants does not preserve the field name; field-name
/// equality is therefore not part of the identity relation.
///
/// The real path lowers the enclosing `Result` through the uniform
/// exception-transform (`front/result_exc.rs`), which recognises the `Ok(step)`
/// identity and **elides** the whole extract-repack, so the legacy read result
/// is never typed under its own Variable identity (`real=None`) — the same
/// rebuilt-block artifact as [`switch_discriminant_read_vars`], for the payload
/// projection rather than the tag.  Requiring the sole op-use to be the raw
/// value operand of a `FieldWrite` keeps this from over-reaching a payload the
/// arm actually consumes: any transform or call is another op-use and excludes
/// the candidate.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn repack_payload_read_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    use crate::model::OpKind;
    let mut read_results = std::collections::HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let (Some(r), OpKind::FieldRead { field, .. }) = (&op.result, &op.kind)
                && field.name != "__discriminant"
            {
                read_results.insert(r.clone());
            }
        }
    }
    if read_results.is_empty() {
        return std::collections::HashSet::new();
    }
    // Tally every op-use of each candidate: `repack` counts a use as the
    // `value` operand of a `FieldWrite`; `other` counts any other op-use. A
    // candidate survives only when it has ≥1 repack use and no other use.
    let mut repack_use = std::collections::HashSet::new();
    let mut other_use: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            // A `FieldWrite` value operand is the repack use; the base operand
            // is handled by the generic operand scan below.
            if let OpKind::FieldWrite {
                value: crate::model::LinkArg::Value(v),
                ..
            } = &op.kind
                && read_results.contains(v)
            {
                repack_use.insert(v.clone());
            }
            for used in crate::front::result_exc::op_operand_vars(&op.kind) {
                if !read_results.contains(&used) {
                    continue;
                }
                // Every position other than the FieldWrite's raw value is an
                // actual use that disqualifies the candidate.
                let is_repack_value = matches!(
                    &op.kind,
                    OpKind::FieldWrite { value: crate::model::LinkArg::Value(vv), .. }
                        if vv == &used
                );
                if !is_repack_value {
                    other_use.insert(used);
                }
            }
        }
    }
    read_results
        .into_iter()
        .filter(|v| repack_use.contains(v) && !other_use.contains(v))
        .collect()
}

/// Variant-carrier input Variables whose only observable work is the
/// extract→repack payload reads classified by
/// [`repack_payload_read_vars`].  The Result/StepResult exception transform
/// elides the entire identity arm, including this block-local carrier, so the
/// real flowspace graph has neither a typed representative for the carrier nor
/// for its payload reads.  This is the inputarg counterpart of the payload
/// result identity reconciliation above.
///
/// Keep the classification structural and narrow: every op-use of the carrier
/// must be as the base of an already-classified payload read, it must not feed
/// an exitswitch, and it must not be forwarded by a link.  A carrier used by
/// any real computation therefore remains a divergence.  This mirrors the
/// identity removal performed by the upstream-style flowspace simplification;
/// it does not assign a type or invent a parallel side table.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn repack_payload_carrier_vars(graph: &LegacyGraph) -> std::collections::HashSet<Variable> {
    use crate::model::{ExitSwitch, LinkArg, OpKind};

    let payloads = repack_payload_read_vars(graph);
    let block_inputargs: std::collections::HashSet<_> = graph
        .blocks
        .iter()
        .filter(|block| {
            block.id != graph.startblock
                && block.id != graph.returnblock
                && block.id != graph.exceptblock
        })
        .flat_map(|block| block.inputargs.iter().cloned())
        .collect();
    let mut carriers = std::collections::HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let (Some(result), OpKind::FieldRead { base, field, .. }) = (&op.result, &op.kind)
                && field.name != "__discriminant"
                && payloads.contains(result)
                && block_inputargs.contains(base)
            {
                carriers.insert(base.clone());
            }
        }
    }
    if carriers.is_empty() {
        return carriers;
    }

    let mut disqualified = std::collections::HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            for used in crate::front::result_exc::op_operand_vars(&op.kind) {
                if !carriers.contains(&used) {
                    continue;
                }
                let is_elided_payload_base = matches!(
                    (&op.result, &op.kind),
                    (Some(result), OpKind::FieldRead { base, field, .. })
                        if base == &used
                            && field.name != "__discriminant"
                            && payloads.contains(result)
                );
                if !is_elided_payload_base {
                    disqualified.insert(used);
                }
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) if carriers.contains(var) => {
                disqualified.insert(var.clone());
            }
            Some(ExitSwitch::Fused { args, .. }) => {
                disqualified.extend(args.iter().filter(|var| carriers.contains(*var)).cloned());
            }
            _ => {}
        }
        for link in &block.exits {
            disqualified.extend(link.args.iter().filter_map(|arg| match arg {
                LinkArg::Value(var) if carriers.contains(var) => Some(var.clone()),
                _ => None,
            }));
        }
    }
    carriers.retain(|var| !disqualified.contains(var));
    carriers
}

/// Whether an op's result may be dropped when unread, mirroring RPython's
/// `transform_dead_op_vars` `canremove(op, block)` gate (`simplify.py`):
/// only side-effect-free ops (`op.opname in CanRemove`) have their result
/// binding removed; a store, a call, a guard, or the block's raising op keeps
/// its result even when locally unread.  The upstream `CanRemove` set is the
/// pure reads/arithmetic/constructors (`getattr`/`getitem`/`len`/`lt`/`eq`/
/// `add`/`isinstance`/`newtuple`/… + `enum_ops_without_sideeffects`); the
/// pyre `OpKind` equivalents are the field/array/interior reads, the length
/// read, the arithmetic/comparison `BinOp`/`UnaryOp`, the `isinstance` test,
/// the immutable-field record, and the tuple/list constructors.  Every op
/// that mutates, calls, guards, forces, or aborts is excluded — its unread
/// result is NOT a dead-var-pass artifact, so a `real=None` on it is a genuine
/// typing divergence rather than a dropped binding.
fn op_result_can_remove(kind: &crate::model::OpKind) -> bool {
    use crate::model::OpKind;
    matches!(
        kind,
        OpKind::FieldRead { .. }
            | OpKind::ArrayRead { .. }
            | OpKind::ArrayLen { .. }
            | OpKind::InteriorFieldRead { .. }
            | OpKind::BinOp { .. }
            | OpKind::UnaryOp { .. }
            | OpKind::IsInstance { .. }
            | OpKind::VtableMethodPtr { .. }
            | OpKind::RecordQuasiImmutField { .. }
            | OpKind::NewTuple { .. }
            | OpKind::NewList { .. }
            | OpKind::GetSlice { .. }
            | OpKind::ConstInt(_)
            | OpKind::ConstFnAddr { .. }
            | OpKind::ConstUInt(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstFloat(_)
            | OpKind::ConstRef(_)
            | OpKind::ConstStr(_)
            | OpKind::ConstInternedStr(_)
            | OpKind::ConstRefNull
            | OpKind::ConstNone
            | OpKind::ConstRefAddr(_)
            | OpKind::ConstSymbolic { .. }
    )
}

/// Diff the real path's per-Variable kinds against the legacy walker's
/// kinds committed onto the same legacy Variables.  Walks
/// [`FunctionGraph::iter_variables`] for a deterministic, slot-free
/// traversal restricted to [`reachable_defined_vars`]; `value_to_var`
/// aligns the two paths by Variable identity, so the legacy kind is
/// read straight off each Variable's `concretetype` cell.  Returns
/// every divergence; the caller decides first-only versus all.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
#[cfg(test)]
fn collect_divergences(
    real_state: &HashMap<Variable, ConcreteType>,
    legacy_graph: &LegacyGraph,
) -> Vec<String> {
    collect_divergences_on(real_state, legacy_graph, None)
}

/// Same as [`collect_divergences`], optionally skipping variables the
/// real annotator never followed.
///
/// `excluded_legacy` is the complement of
/// [`annotator_followed_legacy_vars`]: `value_to_var` keys whose typed
/// twin sits only on a block `follow_link` / `follow_raise_link` never
/// recorded (`annrpython.py` `links_followed`). A structurally
/// reachable unfollowed arm (the `Result::Ok.__pos_0` extract on an
/// always-`Err` `__majit_wrap___new__`) is in this set. A legacy
/// variable with no typed twin is not: `value_to_var` omits Abort
/// results (`flowspace_adapter` `build_value_to_variable_map`), and
/// those must still compare as `real=Unknown`. An empty exclusion
/// compares every reachable variable.
fn collect_divergences_on(
    real_state: &HashMap<Variable, ConcreteType>,
    legacy_graph: &LegacyGraph,
    excluded_legacy: Option<&HashSet<Variable>>,
) -> Vec<String> {
    let reachable_vars = reachable_defined_vars(legacy_graph);
    let colored_operands = colored_operand_vars(legacy_graph);
    let duplicate_inputargs = duplicate_inputarg_vars(legacy_graph);
    let dead_op_results = dead_op_result_vars(legacy_graph);
    let switch_discriminant_reads = switch_discriminant_read_vars(legacy_graph);
    let repack_payload_reads = repack_payload_read_vars(legacy_graph);
    let repack_payload_carriers = repack_payload_carrier_vars(legacy_graph);
    let mut divergences = Vec::new();
    for (pos, var) in legacy_graph.iter_variables().iter().enumerate() {
        // `iterblocks()` parity: the real path annotates only the
        // startblock-reachable closure, so a value defined solely in an
        // unreachable legacy block has no real-path kind.  Comparing it
        // reads real=Unknown and reports a false divergence; skip it
        // exactly as the real path's `remove_dead_blocks` prune does.
        if !reachable_vars.contains(var) {
            continue;
        }
        if let Some(excluded) = excluded_legacy
            && excluded.contains(var)
        {
            continue;
        }
        let legacy_kind = LegacyGraph::concretetype_of(var);
        let real_present = real_state.get(var).copied();
        let real_kind = real_present.unwrap_or(ConcreteType::Unknown);
        // A kind the legacy walker resolved must match the real path;
        // conversely the real path must not resolve a kind for a value
        // the legacy walker left Unknown.
        //
        // Exception: the real path refining a boxed-pointer (`GcRef`)
        // slot to `Void` is sound, not a divergence.  On a `Match` the
        // real path's concretetype is what gets committed and emitted
        // (`codewriter.rs` commits `real_value_to_var`), and the real
        // (annotate→rtype) path is the authoritative parity path.  The
        // shape is the uniform `Result` exception-link lowering
        // extracting a `?`-propagated `Ok(())` as a unit: the legacy
        // walker keeps the boxed `Result` as `GcRef`, while the real path
        // types the extracted unit `Void`.  A `Void` value has no boxed
        // consumer — nothing reads the dropped `GcRef` slot as a pointer.
        // `real=Void` is a positive `()` result (the `None | Some("()")`
        // return-type arm), never the `Unknown` "real path did not type
        // it" default, so accepting it does not admit an untyped slot.
        //
        // Scoped to `legacy=GcRef` only: a `legacy=Signed` slot the real
        // path types `Void` is NOT accepted — the legacy walker's scalar
        // is a live integer the emit path still colors (proven by
        // `object_key_for_checked`'s `lookup_coloring` panic on a
        // `Some(Void)` value referenced under an `Int` regalloc class), so
        // that pairing is a genuine data-kind divergence, not a unit drop.
        //
        // Further excluded: any var the emit path *colors* (see
        // [`colored_operand_vars`]) cannot legitimately be `Void` —
        // upstream never color-allocates the void kind, so a `real=Void`
        // refinement on a colored operand hits the uncolored-Void
        // `getcolor` / `lookup_coloring` panic.  The "no boxed consumer"
        // premise holds only for a genuinely dropped unit (a value
        // forwarded solely into the `returnblock`, emitted as
        // `void_return` without coloring).  Keep the colored case a
        // divergence → Skip to legacy.
        let real_refines_gcref_to_void = legacy_kind == ConcreteType::GcRef
            && real_kind == ConcreteType::Void
            && !colored_operands.contains(var);
        // A `legacy=GcRef, real=Signed` pair is accepted: the real
        // (annotate→rtype) path is authoritative and it refined a genuine
        // integer.  The hitters are oparg-decoded values — a `LOAD_FAST`
        // index (`load_fast_var_num_to_index(var_num, op_arg)` consumed by
        // `idx < code_varnames_len`), an `IS_OP` / `CONVERT_VALUE` /
        // `BUILD_SLICE` discriminant read off an `Arg<T>` marker field, a
        // `JUMP_*` delta — all bare bytecode operands the rtyper types
        // `Signed` (matching upstream, where a bytecode oparg is a plain
        // int).  The legacy walker's `GcRef` is the pyre-only conservative
        // `Unknown → GcRef` backfill (`legacy_resolve.rs:374-378`), which has
        // no RPython analogue (upstream's rtyper never leaves a value
        // untyped, so it never defaults to ref); it is the divergent side.
        //
        // This pair once appeared unacceptable because accepting it crashed
        // `emit_list_of_kind` (`assembler.rs:2169`).  That crash was a phase
        // ordering artifact, not a real mistype: a residual `dont_look_inside`
        // decode helper's argument list is partitioned by kind at jtransform
        // time (`make_three_lists_from_vars`), and the real path's kind was
        // committed onto the typed flowspace Variable but not the legacy key
        // Variable jtransform reads — so the arg was partitioned `ref` while
        // later colored `int`.  Hydrating the legacy Variables before
        // jtransform (`dual_gate_publish_concretetypes` →
        // `apply_from_flowspace_variables`) keeps the partition and the
        // coloring consistent, and the crash does not recur.
        let real_refines_gcref_to_signed =
            legacy_kind == ConcreteType::GcRef && real_kind == ConcreteType::Signed;
        // Same conservative-legacy refinement for a genuine floating-point
        // value. `long_pow`'s negative-exponent arm extracts `f64` payloads
        // from generic Result shells; Charon's erased legacy walk falls back
        // Unknown→GcRef while the annotator/rtyper positively assigns Float.
        // Like the Signed case, publish commits the real kind before
        // jtransform partitions registers, so the float bank is authoritative.
        let real_refines_gcref_to_float =
            legacy_kind == ConcreteType::GcRef && real_kind == ConcreteType::Float;
        // A duplicate phi inputarg the real path's `remove_duplicate_inputargs`
        // merged away is untyped in the real graph (its column no longer
        // exists) while the legacy walker types the retained identity.  The
        // value is emitted through the surviving representative, so
        // `real=Unknown` here is a dedup artifact, not a kind divergence —
        // scoped to `legacy=GcRef` (the erased `<E>::Value` phi family) so a
        // genuine `Signed`/`Float` merged column is never silently accepted.
        let real_dropped_duplicate_inputarg = legacy_kind == ConcreteType::GcRef
            && real_present.is_none()
            && duplicate_inputargs.contains(var);
        // A dead op *result* — no op or exitswitch reads it, only `Link.args`
        // forwarding does, and its op is `canremove`-eligible
        // ([`op_result_can_remove`]) — is dropped by the real path's
        // `transform_dead_op_vars` (`simplify.py`), so it is untyped in
        // the real graph while the legacy walker types every op result.  The
        // hitters span both register banks: erased-element `getarrayitem`
        // reads never observed (`swap_values`'s write-back positions,
        // `store_local_value`'s dead read half), and dead scalar bounds
        // checks (`idx < code_varnames_len` `lt`, a fieldless-enum `getfield`)
        // whose branch the real path folded — `delete_*` / `import_*` /
        // `load_super_attr` / `w_bytes_getitem` / `bigint_external_bytes`.
        // `real=None` here is the dead-result artifact, not a kind divergence.
        // Not scoped by legacy kind: the `canremove` gate on
        // [`dead_op_result_vars`] already excludes every side-effecting /
        // call / store / guard op, so a `real=None` on a value in this set is
        // always the dropped binding — the legacy kind it carries (`GcRef` for
        // an erased element, `Signed` for a folded comparison) is irrelevant.
        // The acceptance stays constrained to the `real_present.is_none()`
        // (untyped) case: a *positively-typed* result whose kind conflicts
        // (the rejected `set_update` element mistyped `Signed`) is never in
        // this set, because a mistyped element is op-read (fed to `w_set_add`
        // / `w_list_append`) and so excluded by the `op_operand_vars` scan.
        let real_dropped_dead_op_result = real_present.is_none() && dead_op_results.contains(var);
        // A `match`-lowered `__discriminant` tag the real path's Result/StepResult
        // exc-transform rebuilt with a fresh switch Variable: the legacy tag
        // reads `real=None` (no twin under its identity) though the real graph
        // types and switches on its own tag.  See [`switch_discriminant_read_vars`].
        let real_rebuilt_switch_discriminant =
            real_present.is_none() && switch_discriminant_reads.contains(var);
        // A variant-payload read whose sole consumer is the identity re-wrap's
        // repacking `FieldWrite`: the real path's exc-transform elided the
        // extract-repack, so the legacy read is untyped.  See
        // [`repack_payload_read_vars`].
        let real_elided_repack_payload =
            real_present.is_none() && repack_payload_reads.contains(var);
        // The block-local variant carrier feeding only the elided payload
        // reads disappears with the same identity arm.  It has no typed twin,
        // but no surviving real-path operation consumes it either.
        let real_elided_repack_carrier =
            real_present.is_none() && repack_payload_carriers.contains(var);
        let diverges = if real_refines_gcref_to_void
            || real_refines_gcref_to_signed
            || real_refines_gcref_to_float
            || real_dropped_duplicate_inputarg
            || real_dropped_dead_op_result
            || real_rebuilt_switch_discriminant
            || real_elided_repack_payload
            || real_elided_repack_carrier
        {
            false
        } else if legacy_kind != ConcreteType::Unknown {
            real_kind != legacy_kind
        } else {
            real_present.is_some()
        };
        if diverges {
            let label = legacy_graph
                .value_name_for(var)
                .unwrap_or_else(|| format!("v{pos}"));
            divergences.push(format!(
                "{label}: legacy={legacy_kind:?}, real={real_kind:?}"
            ));
        }
    }
    divergences
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn compare_real_against_legacy(
    value_to_var: &LegacyToTyped,
    constants: &HashMap<Variable, LowLevelType>,
    legacy_graph: &LegacyGraph,
    excluded_legacy: Option<&HashSet<Variable>>,
) -> Option<String> {
    let real_state = project_value_to_var(value_to_var, constants);
    collect_divergences_on(&real_state, legacy_graph, excluded_legacy)
        .into_iter()
        .next()
}

/// Legacy Variables whose typed twins are defined on a block the annotator
/// recorded in `annotated` for `graph`.
///
/// `addpendingblock` writes that map only after `follow_link` /
/// `follow_raise_link` set `links_followed[link] = True`
/// (`annrpython.py`). The exceptblock is pre-seeded so always-raising
/// graphs still compare their exception pair. An unfollowed `Result::Ok`
/// successor is absent, so its payload extract is not in the compared set.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn annotator_followed_legacy_vars(
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    graph: &crate::flowspace::model::GraphRef,
    value_to_var: &LegacyToTyped,
) -> HashSet<Variable> {
    let annotated = annotator.annotated.borrow();
    let all_blocks = annotator.all_blocks.borrow();
    let mut typed_live: HashSet<Variable> = HashSet::new();
    for (bkey, owner) in annotated.iter() {
        let Some(owner_graph) = owner else {
            continue;
        };
        if !Rc::ptr_eq(owner_graph, graph) {
            continue;
        }
        let Some(block) = all_blocks.get(bkey) else {
            continue;
        };
        let blk = block.borrow();
        for input in &blk.inputargs {
            if let Hlvalue::Variable(v) = input {
                typed_live.insert(v.clone());
            }
        }
        for op in &blk.operations {
            if let Hlvalue::Variable(v) = &op.result {
                typed_live.insert(v.clone());
            }
        }
    }
    value_to_var
        .iter()
        .filter(|(_, typed)| typed_live.contains(*typed))
        .map(|(legacy, _)| legacy.clone())
        .collect()
}

/// `value_to_var` keys that are not in `followed`. The compared set is
/// every other reachable legacy variable, including those
/// `value_to_var` never paired.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn annotator_unfollowed_legacy_vars(
    value_to_var: &LegacyToTyped,
    followed: &HashSet<Variable>,
) -> HashSet<Variable> {
    value_to_var
        .keys()
        .filter(|legacy| !followed.contains(*legacy))
        .cloned()
        .collect()
}

/// Ordered (needles, category) table for [`unported_category`]. First match
/// wins; a row matches when every needle is contained in `msg`.
const UNPORTED_CATEGORIES: &[(&[&str], &str)] = &[
    (
        &["wtf8-strlen-unknown-receiver"],
        "wtf8-strlen-unknown-receiver",
    ),
    (&["not registered in CallRegistry"], "call-registry-miss"),
    (&["translate_op: undefined operand"], "undefined-operand"),
    (&["unimplemented operation"], "unimplemented-operation"),
    (
        &["variable ", " used before definition"],
        "used-before-definition",
    ),
    // `normalize_unary_op_name: pyre UnaryOp` and
    // `normalize_binop_name: pyre BinOp` are not Skip-classified:
    // the `not` / `deref` / `same_as` / `invert` surfaces are
    // desugared by the MIR front-end (rustc lowers `!`/`*` and the
    // `&&`/`||` short-circuits into MIR branches/calls before
    // lowering), and the bitwise `and` / `or` labels normalize to
    // the flowspace `and_` / `or_` registrations.  Synthetic
    // graphs that inject unported ops (anchor tests in
    // `cutover.rs::tests::anchor_unary_*_surfaces_*`) call
    // `specialize_legacy_graph` directly and never reach
    // `is_known_unported`, so the absence of these substring
    // matches does not affect them.  Any production reach surfaces
    // as a dual-gate divergence panic — the parity-correct outcome.
    // Field / method dispatch on a `SomeInstance(classdef=None)`
    // — pyre's `Ref` ValueType currently lifts to a classdef-less
    // SomeInstance, so `find_attribute`
    // (`rclass.py:556+find_attribute_or_None`) cannot route the
    // dispatch.  `InstanceRepr::rtype_getattr`
    // (`rclass.py`) routes through `getclsfield`, which
    // surfaces the upstream-orthodox `MissingRTypeAttribute(attr)`
    // when find_attribute returns None.  The `"no method ... on
    // Instance("` substring (`rmodel.rs:828` default
    // `rtype_getattr` find_method failure path) is not classified
    // here: every SomeInstance dispatch goes through the
    // InstanceRepr override, so the default fn never fires for
    // `Instance(...)`-shaped operands.  The MissingRTypeAttribute
    // entry stays until typed-Ref ClassDef projection lands and
    // field/method dispatch starts succeeding.
    (&["MissingRTypeAttribute"], "missing-rtype-attribute"),
    // Variable's `.annotation` slot empty at `bindingrepr`
    // lookup time — `ValueType::Unknown` has no
    // annotation-stage shell and `valuetype_to_someshell` returns
    // `None` for it (intentionally fail-loud for "annotation gap"
    // so producers know which slot missed seeding, see
    // `seed_variable` at `flowspace_adapter.rs:96-115`).  Closing
    // the gap means tightening the front-end / annotator
    // producers so every slot has a non-`Unknown` annotation; the
    // gate skips until then.  The mergeinputargs-side flavour of
    // this gap (raising subjects' `follow_raise_link` reaching an
    // unannotated exceptblock seed) was resolved 2026-05-31 via
    // `setbinding(Impossible)` pre-registration in
    // `specialize_legacy_graph_with_registry_returning_value_to_var`;
    // the remaining producer is the rtyper-side `bindingrepr`
    // gap on synthesized `current_sp`-like graphs.
    (
        &["KeyError: no binding for arg"],
        "bindingrepr-annotation-gap",
    ),
    // `cast_ptr_to_int` reached the typer with a non-`Ptr`
    // operand — the front-end could not lower a real pointer for
    // the source of an `expr as i64` cast.  The canonical hitter
    // is `stack_check::current_sp` (`&probe as *const usize as
    // usize`): taking the address of a stack local is a
    // target-specific raw-SP read that `front::mir` lowers with
    // the local's *value* (`Int(0)`) rather than its address, so
    // `rtype_cast_ptr_to_int`'s late `InstanceRepr→PtrRepr` swap
    // fallback finds `Signed` where it needs `Ptr(...)`.  The
    // legacy walker handles these graphs; skip until the
    // address-of-local lowering lands a typed pointer operand.
    (
        &["rtype_cast_ptr_to_int: operand concretetype must be Ptr"],
        "cast-ptr-to-int-non-ptr",
    ),
    // Unported per-annotation Repr families.  `rmodel.rs`
    // `rtyper_makerepr` fail-louds with this message shape for the
    // SomeValue kinds whose upstream Repr port has not landed
    // (rrange.py iterator reprs plus annotator/model.py SomeObject
    // and SomeProperty cases; rlist.py ListRepr, rdict.py DictRepr,
    // and rbytearray.py have since landed).  Reached once a graph
    // annotates such a value past the exc-edge and classdef walls;
    // the legacy walker keeps handling these graphs until each Repr
    // is ported.
    (&["rtyper_makerepr — port"], "unported-repr-family"),
    // TODO(annotator-fixpoint-fail-loud) — STRICT-PARITY REGRESSION
    // vs main / PyPy.  `bookkeeper.py compute_at_fixpoint` propagates fixpoint
    // exceptions uncaught and `annrpython.py consider_op` lets
    // `AnnotatorError` reach the caller; absorbing the four
    // patterns below is a deviation that hides real
    // annotator/rtyper parity gaps as "known unported".  Direct
    // removal breaks `pyre-jit-trace` `build.rs` at every reachable
    // hitter
    // (`make_green_key`, `Frame::load_fast`, `PyFrame::locals_w_mut`,
    // `<default methods of IterOpcodeHandler>::record_for_iter_guard`,
    // `pyjitpl_step::Cannot find attribute`); production cannot
    // compile until each underlying real-path gap is closed
    // (classdef-less SomeInstance dispatch + `CallRegistry::
    // ensure_session` coverage — every analyser body routes through
    // per-touch `arg_at` rather than an eager-prefix concrete
    // walk).  Until each gap-hitter is
    // ported, this Skip stays as documented divergence and the
    // codewriter falls back to the legacy walker for production.
    (&["compute_at_fixpoint failed"], "annotator-fixpoint-failed"),
    (
        &["complete_pending_blocks failed"],
        "annotator-pending-blocks-failed",
    ),
    (&["Cannot find attribute "], "cannot-find-attribute"),
    (&["AnnotatorError:"], "annotator-error"),
    // `rtyper.py convertvar` — no conversion path between
    // the two reprs.  The production hitter is the generic-ADT
    // payload attribute (`core.result.Result.Ok.__pos_0` etc.):
    // pyre collapses every `Result<T, E>` instantiation onto ONE
    // classdef, so the attribute's merged annotation unions
    // unrelated payload classes (unit-tuple placeholder `Adt`,
    // `core.option.Option.None`, …) and `pairtype(InstanceRepr,
    // InstanceRepr).convert_from_to` (rclass.py)
    // correctly finds no common base.  Upstream never faces this
    // shape — RPython has no generics, so each class attribute
    // carries a single annotated type.  Skip until
    // per-instantiation classdef specialization lands.
    (&["don't know how to convert from"], "convertvar-no-path"),
    // Annotation-stage flavour of the convertvar entry above
    // (`annrpython.py mergeinputargs` → `UnionError`): two
    // `SomeInstance`s with no common base meet at a block merge
    // because pyre collapses every generic-ADT instantiation onto
    // one classdef, so unrelated payload classes union at the phi.
    // Upstream propagates the UnionError uncaught; the dual gate
    // skips to the legacy walker until per-instantiation classdef
    // specialization lands.
    (
        &["cannot unify instances with no common base class"],
        "union-no-common-base",
    ),
    // The union fallback marker — no arm handles this pair.  Upstream
    // RAISES for the pairs that reach here: `pair(SomeObject, SomeObject)
    // .union()` raises (binaryop.py), `pair(SomePtr, SomeObject)`
    // raises (llannotation.py).  So a hit is a pyre PRODUCER
    // divergence — a boxed `*mut PyObject` lifted as `SomePtr` where
    // RPython carries `SomeInstance`, making a `SomeInstance ∪ SomePtr`
    // phi RPython never constructs — not a missing union handler.  Skip
    // to the legacy walker until the pointer values lift as instances
    // (typed-Ref ClassDef projection).
    (&["pair(s1, s2).union() found no arm"], "union-no-handler"),
    // `InstanceRepr.getfield` walking the `rbase` chain before the
    // repr's deferred `setup()` ran (upstream drains pending
    // setups via `call_all_setups`, rtyper.py, after every
    // specialized block; pyre's per-subject dual-gate can reach a
    // getfield on a freshly minted inner repr first).  Also covers
    // the `rbase missing — call setup() first` form: host-struct-
    // seeded receivers (`*mut PyObject` params with a registry
    // classdef) reach class-field reads through reprs minted
    // outside the `rtyper.getrepr`/`setup()` queue.  Skip until
    // the setup-ordering port lands.
    (&["rbase missing"], "repr-setup-ordering"),
    // `rpbc` calltable row lookup miss at rtype time — the
    // method-PBC came from the struct-root class-dict seeding
    // (`seed_struct_root_method_members`), whose `simple_call`
    // sites do not yet register call shapes into the CallFamily
    // the way `bookkeeper.pbc_call` does for ordinary descs
    // (pbc_call → getcallfamily row, bookkeeper.py).
    // Skip until the seeded-method family registration lands.
    (
        &["calltable row not found in CallFamily"],
        "calltable-row-missing",
    ),
    // A non-instance constant (e.g. a string literal) reaching
    // `InstanceRepr.convert_const` — the receiver field was typed
    // by the untyped FORCE_ATTRIBUTES shell (classdef-less
    // SomeInstance) because no registry-row projection covered
    // it, so a String-valued field rtypes as an instance (hitter:
    // PyError.msg via the type_error raise stubs).  Skip until
    // the typed-Ref field projection covers the remaining rows.
    (
        &["InstanceRepr.convert_const: expected HostObject or None"],
        "convert-const-non-instance",
    ),
    // `rmodel.py rtype_is_` — an `is` (pointer-identity)
    // comparison where one side is not a pointer repr.  The
    // production hitter is `py_type_check`'s `(*obj).ob_type ==
    // tp` chain when `tp` flowed from a `&STATIC` host address:
    // `HostStaticAddrs` annotates the static's address as a
    // constant `SomeInteger`, so the comparison pairs
    // `InstanceRepr` with `IntegerRepr`.  Upstream never faces
    // this — a prebuilt instance constant stays `SomeInstance`.
    // Skip until host static addresses annotate as typed
    // instance pointers.
    (
        &["is of instances of the non-pointers"],
        "rtype-is-non-pointer",
    ),
    (
        &["Call with CallTarget::Indirect", "rclass"],
        "indirect-call-pre-rtyper",
    ),
    // `dyn Trait` dispatch still enters the real-rtyper flowspace
    // adapter as pyre's pre-rtyper `CallTarget::Indirect` shape in
    // some registry-prefill paths.  The production codewriter's
    // jtransform path already runs `rpbc::lower_indirect_calls`
    // before consuming these graphs; the real-rtyper cutover path
    // cannot yet express the matching rclass/rpbc rewrite before
    // adapter input without leaking post-rtyper `VtableMethodPtr` /
    // `IndirectCall` ops into flowspace.  Treat this as a
    // known-unported real-path gap so the dual gate falls back to
    // the legacy type walker instead of panicking during registry
    // population.
    // `annrpython.py mergeinputargs` — an inputarg Variable
    // has no annotation. Happens when cross-block locals
    // threading misses a name in the predecessor link; the
    // annotator cannot merge `None` annotations.
    (&["inputarg lacks annotation"], "inputarg-lacks-annotation"),
    // `BrokenReprTyperError` (`rmodel.py`, ported at
    // `rmodel.rs:449-456`): a Repr whose `setup()` failed earlier
    // is re-requested and refuses to half-initialize.  In the
    // per-graph census a Skipped subject can leave a shared
    // session `ClassRepr` in the BROKEN state; the next graph
    // touching the same repr then surfaces this message.  It is a
    // secondary echo of the root failure (itself Skip-classified
    // above), not an independent cause — absorb it so one broken
    // repr does not fatal every later graph in the session.
    (&["cannot setup already failed Repr"], "broken-repr-echo"),
    // `AnnotatorError: immutablevalue(HostObject` is not
    // Skip-classified for `SyntheticTransparentCtor` (Ok/Err/Some/
    // None).  The adapter emits `HostObject::new_class(name, [])`
    // (`flowspace_adapter.rs:821 SyntheticTransparentCtor` arm),
    // routing through the existing `is_class()` arm in
    // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
    // (`bookkeeper.py` parity).  No production emit-site of
    // `HostObject::new_opaque` remains in `front/`/`translator/rtyper/`,
    // so any future `immutablevalue(HostObject` failure is a real
    // parity bug — let it surface as a dual-gate divergence panic.
    // TODO(post-rtyper-jtransform-variant-leak): retire this Skip
    // entry per upstream parity — `jit/codewriter/jtransform.py`
    // raises straight through on unexpected opnames.  The most
    // frequent reach is `OpKind::Abort` emitted by the surface-Rust
    // DSL front-end's `lower_expr` `stop_unsupported` /
    // `continue_with_unknown` helpers
    // when the surface DSL hits an unsupported expression — pyre
    // source like `execute_opcode_step` / `eval_loop_jit` carry such
    // placeholders.  Retiring this entry needs each Abort
    // emit-site at the front-end retired (closure
    // body, complex match arms, unsupported literals), then
    // every other post-rtyper variant (IndirectCall / ResidualCall
    // / Vable* / JitMergePoint / LoopHeader / InlineCall /
    // RecursiveCall) emitted from `rpbc.rs` / `rclass.rs` ahead
    // of the rtyper.  Until then the Skip absorbs the placeholder
    // leakage; without it `tests::test_codegen_output` /
    // `tests::test_recognition_report` and the analyse-pyre-source
    // pipeline panic at the first reachable Abort.
    (
        &["post-rtyper jtransform variant"],
        "post-rtyper-variant-leak",
    ),
    // Cross-block body `Input` whose `name` was not threaded
    // through `Link.args` / target `inputargs` by the predecessor.
    // RPython flowspace has no body-`Input` op — every cross-block
    // local reference goes via `flowcontext.py LOAD_FAST`
    // (which writes into `self.locals_w`) and the target block's
    // pre-allocated `inputargs[]`.  Pyre's body-`Input` emission
    // (the `OpKind::Input` ops `front::mir` pushes into each block)
    // is itself a TODO; when
    // cross-block locals threading misses a shape, the adapter
    // fails loud with this message instead of silently
    // fabricating a fresh Variable (which would hide an
    // SSA / alias-shape divergence from PyPy's flowspace).  Skip
    // until either cross-block locals threading covers every
    // shape, or the front-end's
    // body-`Input` emission is replaced by a Link.args / inputargs
    // threading pass that mirrors RPython.
    (
        &["adapter cross-block body Input"],
        "cross-block-body-input",
    ),
    // `SomeValue::noneify()` reached a `SomePtr` operand — a block
    // merge unioned a `_ptr` with `NoneType`.  The raise itself is
    // parity-correct: RPython's default `noneify` raises `UnionError`
    // (model.py:121-122) and `SomePtr` defines no override, while
    // `pair(SomePtr, SomeObject).union` raises too
    // (llannotation.py) — so this Skip catches a *correct*
    // raise, it does not paper over a wrong annotation rule.  The
    // only divergence is upstream of the merge: RPython spells a null
    // pointer as `lltype.nullptr(T)`, which annotates as a
    // (null-valued) `SomePtr`, so `pair(SomePtr, SomePtr).union`
    // (llannotation.py) keeps it in the pointer lattice and
    // `noneify` is never reached.  Pyre's `front::mir` still lowers
    // some Rust null/`Option<*T>` shapes to a `SomeNone` rather than
    // a typed null `SomePtr` (the same null-pointer-typing gap
    // tracked by the `LLAddress(Null)` exceptblock work), so the
    // merge surfaces `_ptr ∪ NoneType`.  CONVERGENCE: lower a typed
    // null pointer at the `front::mir` producer; removing this Skip
    // without that fix only hard-breaks on the parity-correct raise.
    // Until then the legacy walker keeps handling the graphs.
    (&["noneify() not supported"], "noneify-on-ptr"),
    // `flowspace_adapter::translate_op` rejected an UNFUSED GC malloc
    // `FunctionPath`. `fuse_boxing_alloc` is the general source-aggregate to
    // `NewWithVtable` lowering; a cluster whose owner/header/payload cannot be
    // proven stays fail-closed here instead of becoming a residual allocator
    // call with the wrong JIT semantics.
    (
        &["survived fuse_boxing_alloc unfused"],
        "malloc-typed-unfused",
    ),
    // `OrderedDictRepr::require_direct_compare_key` fail-closed gate
    // (rordereddict.rs) — `build_ll_dict_lookup_helper_graph`'s
    // `direct_compare_op` hardcodes `ptr_eq` for every `Ptr(_)` key
    // lltype, which is wrong for keys whose repr defines a real
    // structural equality (`get_ll_eq_function` returning `Some`, e.g.
    // `StringRepr::ll_streq`). int/bool/char/unichar keys return `None`
    // and proceed; every other key repr (str, instance) hits this
    // TyperError instead of silently miscompiling dict getitem/contains
    // to pointer-identity comparison. Skip-classify until the
    // call-based keyeq branch is ported (#140 DictRepr epic).
    (&["dict key eq function not wired"], "dict-key-eq-missing"),
];

/// Return true when `msg` matches one of the known-unported
/// categories the dual-gate currently treats as `Skip` rather than
/// `Err`.  Each category is its own future port; the predicate is
/// the central place to enumerate what's still unimplemented.
///
/// As ports land, entries here move out (the port closes the
/// category) and the surface of "unknown failure" shrinks.  When
/// the predicate matches nothing, the gate is at full coverage.
///
/// Each remaining entry is blocked on a specific unimplemented
/// feature; retiring an entry early is unsafe until that feature
/// lands:
///
/// | Substring                                  | Unimplemented feature                                                        |
/// |--------------------------------------------|------------------------------------------------------------------------------|
/// | `not registered in CallRegistry`       | Extern Rust helper registry walker.                                          |
/// | `translate_op: undefined operand`          | Adapter producer correctness.                                                |
/// | `unimplemented operation`                  | Per-opname rtyper handlers.                                                  |
/// | `variable used before definition`          | Cross-block locals threading.                                                |
/// | `MissingRTypeAttribute`                    | Typed-Ref → SomeInstance(ClassDef).                                          |
/// | `KeyError: no binding for arg`             | Rtyper bindingrepr gap on annotation-less Variable (e.g. `current_sp`).      |
/// | `compute_at_fixpoint failed`               | PBC dispatch / call-family coverage (per-call).                              |
/// | `post-rtyper jtransform variant`           | Per-variant emit-site retracing (rpbc / rclass / front-end).  Includes `OpKind::Abort` (pyre-only marker; retire every `Expr::ForLoop` / `stop_unsupported` / `continue_with_unknown` emit-site at the front-end). |
/// | `adapter cross-block body Input`           | Final emit-site retirement.                                                  |
/// | `noneify() not supported`                  | Front-end typed null-pointer lowering (`Option<*T>` / null → `SomePtr`, not `SomeNone`); the raise itself is parity-correct. |
///
/// The `complete_pending_blocks failed` / `Cannot find attribute `
/// blanket above also absorbs the classdef-less `SomeInstance`
/// container/iterator-protocol idiom: a Rust `&[T]` slice has no
/// class root (`tyref_class_root` returns `None`), so `.iter()`
/// yields a `SomeInstance(classdef=None)` and the iterator-protocol
/// `getattr("__iter__")` synthesis (`unaryop.rs` `OpKind::Iter` ×
/// classdef-less `SomeInstance`) finds no attribute.  The live
/// hitters are `pyre_object::listobject::{all_ints, all_floats,
/// boxed_from_ints}` (`items.iter().all(..)` / `.map(..)`).  This is
/// a deliberate non-port: the Rust slice-combinator idiom has no
/// RPython analogue (RPython iterates lists/ranges via explicit
/// `for`-loops, never raw slices, and has no `Iterator::all`/`any`
/// combinator), so degrading it to the legacy walker is the parity
/// envelope — closed only by giving slices a class root plus an
/// iterator Repr (the rlist iterator Repr extended past `SomeList`).
/// The sibling classdef-less hitters
/// (`getattr("deref"/"__len__"/"as_str"/"__getitem__")`) close via
/// foreign-value-type modeling, NOT ClassDef projection: the
/// receivers are foreign value types (a `Deref` wrapper / `Vec` /
/// `Wtf8` / a slice), and `deref`/`__len__`/`as_str`/`__getitem__`
/// are Rust trait methods, not `ClassDef.attrs` field rows.
/// `ClassDef::find_attribute` (`classdesc.rs`) reads only `attrs[..]`
/// field rows, so attaching a ClassDef to the receiver cannot
/// resolve a trait method.  Every classdef-less-instance hitter in
/// the prepass histogram accesses a foreign-value method
/// (`to_f64`/`deref`/`__iter__`/`len`/`as_str`/`to_i64`/`index`); none
/// reads an object struct field, so typed-Ref ClassDef projection
/// (the object-pointer epic) moves none of them.
///
/// PyPy `bookkeeper.py` propagates fixpoint failures uncaught;
/// pyre's dual-gate Skip defers exactly the categories enumerated
/// above. Once every category is implemented, every message returns
/// `None`, the legacy walker fallback at
/// `codewriter.rs::dual_gate_type_state` becomes dead code, and the
/// predicate retires entirely.
///
/// The returned tag identifies the matching fallback condition. Keeping the
/// category and the boolean decision in one function prevents diagnostic code
/// from maintaining a second, drifting copy of the substring table. Lowering
/// only tests whether the result is `Some`; the tags are stable identifiers
/// for the decline census (`crate::decline`).
pub(crate) fn unported_category(msg: &str) -> Option<&'static str> {
    for &(needles, category) in UNPORTED_CATEGORIES {
        if needles.iter().all(|n| msg.contains(n)) {
            return Some(category);
        }
    }
    // There is no `normalize_unary_op_name: pyre UnaryOp` Skip entry:
    // the 13 typed numeric / ptr / Unsigned casts route through
    // `simple_call(<host_callable>, v)` — reaching
    // `Repr.rtype_int/float/bool` or `BUILTIN_TYPER[lltype.cast_*]`
    // directly.  Only the `same_as` identity / source-type-unknown
    // fallback remains on the `OpKind::UnaryOp` route, dispatched by
    // `RPythonTyper::translate_operation` to `rbuiltin::rtype_same_as`
    // (verbatim port of `rtyper.py`).  `normalize_unary_op_name`
    // accepts `same_as` straight through; any residual fail-loud
    // surfaces as a dual-gate divergence panic — the parity-correct
    // outcome.
    None
}

/// Category for a Skip that never passes through [`unported_category`].
///
/// [`unported_category`] classifies a PANIC payload: the dual gate
/// catches an unwind and asks whether the failure is a known-unported
/// shape.  A Skip returned directly as `Ok(DualGateOutcome::Skip(..))`
/// never reaches that predicate, so a census that only broke down
/// `unported_category`'s arms would file every one of them under
/// "unclassified" — and on cel's closure the direct Skips are the ones
/// that fire.  The two populations are disjoint, which is why this is a
/// separate function rather than more arms: adding these to
/// `unported_category` would make `is_known_unported` answer `true` for
/// them and change what the panic handler does.
///
/// Census classification only.  No lowering decision reads it.
pub(crate) fn non_arm_skip_category(msg: &str) -> Option<&'static str> {
    if msg.contains("two-phase: AccessDirect graph was never a prepass subject") {
        return Some("two-phase-accessdirect-never-a-subject");
    }
    if msg.contains("two-phase: AccessDirect PyGraph missing from FunctionDesc cache") {
        return Some("two-phase-accessdirect-pygraph-missing");
    }
    if msg.contains("two-phase: graph was never a prepass subject") {
        return Some("two-phase-never-a-subject");
    }
    if msg.contains("two-phase: subject rtype-skipped in prepass") {
        return Some("two-phase-rtype-skipped");
    }
    if msg.contains("two-phase: subject annotate-skipped in prepass") {
        return Some("two-phase-annotate-skipped");
    }
    None
}

/// The registry dedupe key of `path` (see
/// [`populate_call_registry_from_call_graphs`]).
fn canonical_dedup_key(path: &crate::parse::CallPath) -> Vec<String> {
    let mut segs: Vec<String> = path.segments.to_vec();
    if segs
        .first()
        .map(|s| s == "crate" || crate::local_crates::is_local_crate_root(s))
        .unwrap_or(false)
    {
        segs.remove(0);
    }
    segs
}

/// Why a funcobj under the canonical path `canonical_strip` gets no
/// registry entry: its callsites resolve elsewhere.
fn registration_decline(canonical_strip: &[String]) -> Option<&'static str> {
    // `pyre_object::lltype::malloc[_typed/_stable]` are GC allocation intrinsics,
    // recognised as host builtins (annotator `malloc_typed_alloc`, HOST_ENV
    // `pyre_object.lltype` module). Their real bodies enter the host
    // allocator (`malloc_typed` first reads the un-flowable
    // `<T as GcType>::type_id()` trait accessor), so lifting either records
    // a poison lift-error that surfaces on the first boxing caller. Skip
    // registering it as a user function: with no registry entry, callsites
    // resolve to the HOST_ENV builtin (translate_op Layer-3b) instead of
    // this failed user-graph entry (Layer-1 `call_registry.lookup`).
    //
    // `fuse_boxing_alloc` now derives every payload from the registered
    // struct layout and rewrites a proven `malloc[_typed](T { ... })`
    // cluster to `NewWithVtable` plus setfields, matching
    // `jtransform.rewrite_op_malloc` (`jtransform.py`).  A surviving call
    // is still not a residual boundary: it means the frontend could not
    // prove the allocation header/type identity.  The fail-closed guard in
    // `flowspace_adapter::translate_op` therefore rejects every surviving
    // `malloc`, `malloc_typed`, `malloc_typed_managed`, or
    // `malloc_typed_stable` call instead of
    // emitting an executable symbolic fnaddr.
    if canonical_strip == ["lltype", "malloc"]
        || canonical_strip == ["lltype", "malloc_typed"]
        || canonical_strip == ["lltype", "malloc_typed_managed"]
        || canonical_strip == ["lltype", "malloc_typed_stable"]
    {
        return Some("skip-malloc-typed-intrinsic");
    }
    // `#[pyre_class]`'s generated allocation constructors
    // (`<Owner>::allocate(payload: Self)` / `allocate_stable(payload:
    // Self)`) build the object then call the non-numeric
    // `lltype::malloc_typed[_stable]`, which has no ported general
    // `malloc->new` lowering — lifting the body records a poison lift-error
    // that surfaces on the first constructing caller (`w_range_new`,
    // the iterator/`GenericAlias` `*_new` heads).  Skip registering the
    // constructor as a user function so its key is instead served by the
    // residual stub from `collect_marked_class_ctor_stubs_from_llbc` (seeded
    // through the `unsafe_fn_stubs` carrier), whose `*mut PyObject` return
    // shell lets the caller annotate.  The numeric boxes never reach here —
    // they call `malloc_typed` inline (`w_float_new`/`w_int_new`), not the
    // macro `allocate` wrapper — so this does not disturb
    // `fuse_boxing_alloc`.
    if matches!(
        canonical_strip.last().map(String::as_str),
        Some("allocate") | Some("allocate_stable")
    ) {
        return Some("skip-pyre-class-allocate-ctor");
    }
    // `pyre_object::lltype::malloc_raw` is the raw (non-GC) allocation
    // intrinsic (`lltype.malloc(T, flavor='raw')` parity), recognised as
    // a host builtin (annotator `malloc_raw_alloc`, HOST_ENV
    // `pyre_object.lltype` module).  Its body calls the analyser-less
    // `Box::new` / `Box::into_raw`, so lifting it records a poison
    // lift-error that surfaces on the first raw-allocating caller.  Skip
    // registering it as a user function for the same reason as
    // `malloc_typed`: callsites resolve to the HOST_ENV builtin
    // (translate_op Layer-3b) instead of this failed user-graph entry.
    if canonical_strip == ["lltype", "malloc_raw"] {
        return Some("skip-malloc-raw-intrinsic");
    }
    // `rlib/nonconst.py NonConstant` is an `ExtRegistryEntry`, not a
    // graph: its whole purpose is that the annotator must NOT see the
    // argument's constancy, and the identity body lifted from
    // `majit-rlib` would hand that constancy straight back.  With no
    // registry entry the callsite resolves through
    // `translate_op` Layer-3b to the `majit_rlib.nonconst` HOST_ENV
    // callable, whose analyzer answers `not_const(s_arg)` and whose
    // `rtype_non_constant` returns argument 0.
    if canonical_strip == ["nonconst", "non_constant"] {
        return Some("skip-nonconst-extregistry-entry");
    }
    // `rlib/jit.py ConditionalCallEntry` is an ExtRegistry object
    // (`_about_ = _jit_conditional_call, _jit_conditional_call_value`),
    // not a graph. The residual `majit_rlib::jit::conditional_call*`
    // bodies are the interpreter fallback (`if condition: function(*args)`
    // / `if value is NULL: function(*args)`). Lifting them lets the
    // annotator type `function(a)` as an ll funcptr call, which rejects a
    // high-level Instance argument. With no registry entry the callsite
    // reaches the HOST_ENV builtin (`translate_op` Layer-3b), whose
    // analyzer is `jit_conditional_call` / `jit_conditional_call_elidable`
    // and whose typer emits `jit_conditional_call` /
    // `jit_conditional_call_value`. Canonical strip is `["jit", leaf]`
    // after `canonical_dedup_key` drops the `majit_rlib` crate root.
    if is_jit_conditional_call_residual(canonical_strip) {
        return Some(SKIP_JIT_CONDITIONAL_CALL_EXTREGISTRY);
    }
    // `core::ptr::copy_nonoverlapping` is an opaque foreign decl. Layer-1
    // `call_registry.lookup` beats `host_env_callable`, so a registered
    // user graph is the callee: its empty body still `mergeinputargs`s
    // every pointee into one slot. Declining the registry entry lets the
    // callsite reach the `std.ptr.copy_nonoverlapping` host builtin
    // (`std_ptr_copy_nonoverlapping`). The leaf key is the alias
    // `free_function_alias_paths` registers from `SemanticFunction.name`.
    if canonical_strip == ["core", "ptr", "copy_nonoverlapping"]
        || canonical_strip == ["std", "ptr", "copy_nonoverlapping"]
        || canonical_strip == ["copy_nonoverlapping"]
    {
        return Some(SKIP_PTR_COPY_NONOVERLAPPING);
    }
    // `majit_gc::header::alloc_varsize_with_gc_header_immortal[_zeroed]`
    // / `dealloc_varsize_with_gc_header` are `malloc_varsize` operations
    // (`init_gc_object_immortal`), not graphs. Their bodies walk
    // `std::alloc::Layout` / `GcHeader::write`, which poisons every
    // string/bytes payload allocator that calls them. Skip the user-graph
    // entry so Layer-3b reaches the HOST_ENV builtin (`alloc_varsize_gc_header`).
    // `majit_gc` is not a local extract crate, so `canonical_dedup_key`
    // keeps the crate root; the crate-stripped and leaf-only aliases are
    // the same callable (`is_jit_conditional_call_residual` shape).
    if is_varsize_gc_header_alloc(canonical_strip) {
        return Some(SKIP_VARSIZE_GC_HEADER_ALLOC);
    }
    // `std::alloc::alloc` / `alloc_zeroed` (and `alloc::alloc` re-export)
    // are opaque foreign decls. A residual `ref` stub shells `*mut u8`
    // as classdef-less `SomeInstance`; `ann_malloc` annotates the
    // result as `SomePtr`. Declining the registry entry lets the
    // callsite reach the HOST_ENV builtin (`std_alloc_raw`). The
    // `alloc_zeroed` leaf is the alias `free_function_alias_paths`
    // registers from `SemanticFunction.name`. The `alloc` leaf is
    // left registered: too many other functions share that name.
    if is_std_alloc_raw(canonical_strip) {
        return Some(SKIP_STD_ALLOC_RAW);
    }

    None
}

/// [`registration_decline`] reason for `ptr::copy_nonoverlapping`.
const SKIP_PTR_COPY_NONOVERLAPPING: &str = "skip-ptr-copy-nonoverlapping";

/// [`registration_decline`] reason for `std::alloc::alloc` / `alloc_zeroed`.
const SKIP_STD_ALLOC_RAW: &str = "skip-std-alloc-raw";

/// `std::alloc::alloc` / `alloc_zeroed` and the `alloc::alloc` re-export.
/// The `alloc_zeroed` leaf is unique; the `alloc` leaf is not.
fn is_std_alloc_raw(canonical_strip: &[String]) -> bool {
    match canonical_strip {
        [crate_or_std, module, leaf]
            if (crate_or_std == "std" || crate_or_std == "alloc")
                && module == "alloc"
                && (leaf == "alloc" || leaf == "alloc_zeroed") =>
        {
            true
        }
        [leaf] if leaf == "alloc_zeroed" => true,
        _ => false,
    }
}

/// [`registration_decline`] reason for `rlib/jit.py ConditionalCallEntry`.
const SKIP_JIT_CONDITIONAL_CALL_EXTREGISTRY: &str = "skip-jit-conditional-call-extregistry";

/// [`registration_decline`] reason for `majit_gc::header` varsize immortal
/// allocators (`malloc_varsize` / `init_gc_object_immortal`).
const SKIP_VARSIZE_GC_HEADER_ALLOC: &str = "skip-varsize-gc-header-alloc";

/// Residual `majit_rlib::jit::conditional_call0..4` /
/// `conditional_call_elidable1`. `canonical_dedup_key` strips the crate
/// root only when `majit_rlib` is a seeded local crate, so both
/// `["jit", leaf]` and `["majit_rlib", "jit", leaf]` name the same
/// HOST_ENV / `BUILTIN_ANALYZERS` entry.
fn is_jit_conditional_call_residual(canonical_strip: &[String]) -> bool {
    let leaf = match canonical_strip {
        [crate_name, module, leaf] if crate_name == "majit_rlib" && module == "jit" => {
            leaf.as_str()
        }
        [module, leaf] if module == "jit" => leaf.as_str(),
        _ => return false,
    };
    matches!(
        leaf,
        "conditional_call0"
            | "conditional_call1"
            | "conditional_call2"
            | "conditional_call3"
            | "conditional_call4"
            | "conditional_call_elidable1"
    )
}

/// Residual `majit_gc::header::alloc_varsize_with_gc_header_immortal*` /
/// `dealloc_varsize_with_gc_header`. `canonical_dedup_key` keeps `majit_gc`
/// (not a local extract crate), so both `["majit_gc","header",leaf]` and
/// the crate-stripped `["header",leaf]` / leaf-only aliases name the same
/// HOST_ENV entry.
fn is_varsize_gc_header_alloc(canonical_strip: &[String]) -> bool {
    let leaf = match canonical_strip {
        [crate_name, module, leaf] if crate_name == "majit_gc" && module == "header" => {
            leaf.as_str()
        }
        [module, leaf] if module == "header" => leaf.as_str(),
        [leaf] => leaf.as_str(),
        _ => return false,
    };
    matches!(
        leaf,
        "alloc_varsize_with_gc_header_immortal"
            | "alloc_varsize_with_gc_header_immortal_zeroed"
            | "alloc_varsize_with_gc_header_immortal_zeroed_flag"
            | "dealloc_varsize_with_gc_header"
            | "varsize_immortal_layout"
    )
}

/// Publish what the source callable of `graph` declares about its result
/// on its registry entry.
fn publish_source_signature(graph: &LegacyGraph, entry: &FunctionEntry) {
    // GraphStore aliases are iteration-ordered, so the canonical entry
    // selected above can be keyed by the bare leaf even though the graph
    // retains its exact source callable identity.  RPython attaches
    // `_signature_` to that callable object, not to one spelling of its
    // import path (`bookkeeper.py getdesc`, `description.py:234-244`).  allow-line-citation
    // Recover the same fact from `FunctionGraph.source_identity` here;
    // this keeps a foreign same-leaf function distinct while ensuring
    // every alias of the real materializer shares the semantic result.
    if graph
        .source_identity
        .as_deref()
        .map(|identity| {
            is_exception_object_materializer(&FunctionPathKey::from_segments(identity.split("::")))
        })
        .unwrap_or(false)
    {
        entry.publish_exception_object_result_signature();
    }
}

/// Project the callee's declared LLBC fn-ptr signature onto its entry
/// before any body is lifted (see the pre-pass in
/// [`populate_call_registry_from_call_graphs`]).
fn seed_declared_funcptr_type(graph: &LegacyGraph, entry: &FunctionEntry) {
    if let Some(ft) = declared_funcptr_type_from_legacy(graph) {
        entry.set_declared_funcptr_type(ft);
    }
}

/// Give the registry entry of a funcobj the lowered body its default
/// graph is built from (Pass 2 of
/// [`populate_call_registry_from_call_graphs`]).
fn install_source_graph(
    registry: &Rc<CallRegistry>,
    key: &FunctionPathKey,
    graph: &LegacyGraph,
    entry: &FunctionEntry,
    signature: &Signature,
    body: &crate::codewriter::call::StoredBody,
) {
    // `@jit.dont_look_inside` (`rlib/jit.py`) callees: the JIT
    // pipeline never builds a jitcode for the body
    // (`policy.py look_inside_graph` returns False, so
    // `codewriter.py find_all_graphs` excludes the graph and the
    // callsite residualizes).  Pyre's annotator exists solely to
    // feed that jitcode pipeline, so lifting the body here is
    // upstream-dead work — and bodies marked opaque are typically
    // unliftable host plumbing (`OnceLock` init walks etc.) whose
    // lift failure would poison every caller.  Prefill the same
    // signature-only stub pygraph `register_unsafe_fn_stubs` uses
    // (the `ExtRegistryEntry.compute_result_annotation` shape) so
    // callers annotate the declared result and the codewriter emits
    // the residual call via the fn's registered C ABI address
    // (`pyre/jit_fnaddr.rs`).  The FUNC.RESULT token is the canonical
    // spelling `front::mir::dont_look_inside_return_token` stamps from
    // the callee's return `ValueType`; [`residual_return_shell`]
    // decodes it.  A token it declines falls through to the normal
    // lift.
    // specialize.memo evaluates the host callable, never its original
    // flow graph. MemoTable.finish installs the decision graph later.
    if entry.function_desc.borrow().is_memo() {
        return;
    }
    let residualize = graph.hints.iter().any(|h| h == "dont_look_inside")
        || graph.hints.iter().any(|h| h == "elidable");
    if residualize {
        // The materialisers' raw `*mut PyObject` return is only their
        // residual-call ABI.  Their FunctionDesc carries the orthodox
        // `_signature_`-style SomeInstance result, so the synthetic
        // annotator graph must carry that same result on its return link.
        // Leaving the generic OBJECTPTR/SomePtr shell here makes Phase B
        // insert a PtrRepr -> InstanceRepr conversion that upstream never
        // has, then the first failed specialization replaces the shared
        // stub and strands every later caller on an unbound return var.
        let semantic_key = graph
            .source_identity
            .as_deref()
            .map(|identity| FunctionPathKey::from_segments(identity.split("::")))
            .unwrap_or_else(|| key.clone());
        let result_shell = raw_ptr_token_annotation(registry, graph.return_type.as_deref())
            .or_else(|| residual_stub_result_shell(&semantic_key, graph.return_type.as_deref()))
            .map(|shell| declared_return_annotation(registry, graph).unwrap_or(shell));
        if let Some(result_shell) = result_shell {
            let stub = build_stub_pygraph_with_result_shell(
                graph.name.clone(),
                signature.clone(),
                result_shell,
            );
            entry.prefill_default_cache(stub);
            return;
        }
    }
    let func = entry
        .host_object
        .user_function()
        .expect("registered source callable")
        .clone();
    let body = body.clone();
    let signature = signature.clone();
    // The registry is the body's global namespace; the entry it owns
    // must not keep it alive.
    let globals = Rc::downgrade(registry);
    entry
        .function_desc
        .borrow()
        .set_source_graph(crate::annotator::description::SourceGraph::new(move || {
            let registry = globals
                .upgrade()
                .ok_or_else(|| "the call registry was dropped".to_string())?;
            let source = body
                .graph()
                .ok_or_else(|| "the funcobj's graph build produced no graph".to_string())?;
            lift_callee_to_pygraph_with_func(&source, signature, &registry, func)
                .map_err(|e| format!("{e}"))
        }));
}

/// Whether the dual gate should Skip this failure to the legacy walker.
///
/// The predicate is [`unported_category`] with its answer erased, so there is
/// one list of fallback conditions.
pub(crate) fn is_known_unported(msg: &str) -> bool {
    unported_category(msg).is_some()
}
/// Populate a `CallRegistry` from a
/// `HashMap<CallPath, FunctionGraph>` (the shape pyre's
/// `CallControl::function_graphs()` returns).
///
/// Mirrors [`populate_call_registry_from_program`] but keyed off
/// `CallPath` (parse layer's path type, used by `transform_graph_to_jitcode`'s
/// callcontrol map) instead of `SemanticProgram.functions`.  Every
/// callee referenced from any `OpKind::Call::FunctionPath` callsite
/// in the program lives in this map under the same path the front
/// end emitted when it canonicalised each call target,
/// so registry lookups during the lift step always hit.
///
/// Two-pass shape (same as the SemanticProgram walker — see that
/// helper's docstring for the recursivecall parity argument):
///
/// 1. `get_or_register` every entry's signature so HostObjects exist
///    before any callee body lifts.
/// 2. `lift_callee_to_pygraph` + `prefill_default_cache` per entry
///    so `cachedgraph` (`description.rs:1037-1039`) hits at the
///    rtyper's `direct_call`.
pub(crate) fn populate_call_registry_from_call_graphs(
    function_graphs: &crate::codewriter::call::GraphStore,
    unsafe_fn_stubs: &[(Vec<String>, Signature, Option<String>)],
    foreign_opaque_method_externals: &[(Vec<String>, Signature, crate::model::ValueType)],
    atomic_load_decls: &[crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl],
    registry: &Rc<CallRegistry>,
) -> Result<(), TyperError> {
    // Decline-census gate name for this function's registration skips.
    // A callable skipped here has no registry entry, so every callsite
    // resolves it as a host builtin or a residual stub rather than as a
    // user graph — silently, with the four reasons below indistinguishable
    // from one another at the callsite.
    use crate::decline::gate::CALL_REGISTRY as REGISTRY_GATE;
    // Dedupe by canonical path — RPython `Bookkeeper.getdesc(pyobj)`
    // (`bookkeeper.py`) returns the *same* FunctionDesc for
    // any reference to the same callable, keyed by `Constant(pyobj)`
    // identity. This port's `function_graphs: HashMap<CallPath, FunctionGraph>`
    // can carry the same callable under multiple keys (`lib.rs`
    // registers free functions under both their canonical path AND a
    // `crate::`-prefixed alias for callsite matching); without dedupe,
    // each path key would create a separate HostObject / FunctionDesc
    // pair, violating upstream's "create once, share thereafter"
    // contract.
    //
    // The dedupe key strips a leading crate-prefix segment so the alias
    // spellings registered by `lib.rs` for each free function (`[mod, foo]`,
    // `[crate, mod, foo]`, and `[loaded_crate, mod, foo]`) collapse to a
    // single canonical entry `[mod, foo]`. Stripping only `crate` (the
    // previous heuristic) left the external crate-name aliases as distinct canonical
    // entries with distinct `Rc<FunctionEntry>` HostObjects, and
    // `lookup_with_leaf_match`'s multi-match convergence check
    // (`all_same` on `host_object` Arc identity) then failed —
    // callsites like `["crate", "w_str_new"]` (a `use crate::w_str_new`
    // bare callsite from another module) found four matches with four
    // distinct hosts and rejected the cluster. Stripping all registered
    // prefixes preserves the rule that genuinely distinct
    // functions in different modules (`a::foo` vs `b::foo`) stay
    // separate (`a` / `b` aren't in the stripped set), and the
    // bare-name dedupe trap (using `FunctionGraph::name`) is still
    // avoided because we key on the post-strip segment sequence, not
    // on the bare leaf.
    let mut pending: Vec<(
        FunctionPathKey,
        Rc<LegacyGraph>,
        Rc<FunctionEntry>,
        Signature,
        crate::codewriter::call::StoredBody,
    )> = Vec::with_capacity(function_graphs.len());
    // `by_canonical_path` tracks the canonical `FunctionPathKey` of
    // the first-encountered alias for each canonical-stripped key.
    // Subsequent aliases of the same callable register an alias row
    // pointing at that canonical key — the registry's `entries` map
    // gets exactly one row per distinct callable, with `aliases`
    // carrying the indirection.
    let mut by_canonical_path: HashMap<Vec<String>, FunctionPathKey> = HashMap::new();
    for (path, graph) in function_graphs.iter_declared() {
        // A funcobj the front end declared gets its entry on first lookup
        // (`describe_declared_funcobj`).
        if function_graphs.declared_funcobj(&path).is_some() {
            continue;
        }
        // `{name}__AccessDirect_None` is `description.py cachedgraph`'s
        // alt_name for the `(AccessDirect, key)` copy of the same
        // FunctionDesc. Register it as an alias after this pass; inserting
        // it as canonical would make `CallRegistry.alias` panic and would
        // give the copy its own desc.
        if crate::codewriter::call::is_access_directly_path(&path) {
            continue;
        }
        let key = FunctionPathKey::from_segments(path.segments.iter().cloned());
        let canonical_strip = canonical_dedup_key(&path);
        if let Some(reason) = registration_decline(&canonical_strip) {
            crate::decline::record(REGISTRY_GATE, reason, format_args!("{path}"));
            continue;
        }
        let signature = crate::codewriter::call::StoreCore::signature_from_graph(&graph);
        let entry = if let Some(canonical_key) = by_canonical_path.get(&canonical_strip) {
            if canonical_key != &key {
                registry.alias(key.clone(), canonical_key);
            }
            registry
                .lookup(canonical_key)
                .expect("canonical entry registered")
        } else {
            let entry = registry.lookup(&key).unwrap_or_else(|| {
                registry.get_or_register_with_func(
                    key.clone(),
                    signature.clone(),
                    source_graph_func(&graph),
                )
            });
            by_canonical_path.insert(canonical_strip, key.clone());
            entry
        };
        publish_source_signature(&graph, &entry);
        let body = function_graphs
            .body(&path)
            .expect("a declared path names a funcobj");
        pending.push((key, graph, entry, signature, body));
    }
    // `function_graphs.iter_declared()` yields `CallPath.segments` order,
    // so `alloc_desc_key` and first-alias-wins already follow that key.
    // Sort pending as a defense for the lift/prefill passes below.
    pending.sort_by(|a, b| a.0.segments().cmp(b.0.segments()));
    // Point each AccessDirect path at the origin FunctionDesc
    // (`cachedgraph` stores both graphs on one desc). Skip when the
    // origin was never registered (a declined callable has no desc to
    // alias onto). `alias` panics if the AD key is already canonical,
    // which the skip in the loop above prevents.
    for (path, _) in function_graphs.iter_declared() {
        if !crate::codewriter::call::is_access_directly_path(&path) {
            continue;
        }
        let Some(origin) = crate::codewriter::call::access_directly_origin_path(&path) else {
            continue;
        };
        let origin_strip = canonical_dedup_key(&origin);
        let Some(canonical_key) = by_canonical_path.get(&origin_strip) else {
            continue;
        };
        let alias_key = FunctionPathKey::from_segments(path.segments.iter().cloned());
        if &alias_key != canonical_key {
            registry.alias(alias_key, canonical_key);
        }
    }
    // Register `unsafe fn` stubs between Pass 1 (alias explosion) and
    // Pass 2 (callee lift).  A stub carries the crate-included
    // `name_path()` spelling that the pass above does not register; without
    // it a safe-fn body lifted in Pass 2 that calls an unsafe callee
    // (`is_cell`, `is_exception`, …) records a "not registered" lift error
    // and the caller Skips.  Seeding here — not before Pass 1 — keeps Pass 1's
    // `registry.alias()` invariant ("alias key already a canonical entry")
    // intact: the alias explosion has already landed, and neither Pass 2
    // (lift) nor Pass 3 (class-member binding) calls `alias()`, so a
    // verbatim single-key stub cannot collide.  `register_unsafe_fn_stubs`
    // is non-overwriting (skips keys the `function_graphs` pass already
    // registered) and idempotent.
    register_unsafe_fn_stubs(registry, unsafe_fn_stubs);
    // `__deref_write(ptr, value)` is the MIR front end's opaque store
    // marker.  Give the annotator the same declared-external carrier as the
    // Rust-stdlib leaves below so one unfollowable store does not stop the
    // whole graph; the marker is absent from `CallControl::function_graphs`,
    // so the original call remains residual at the codewriter boundary.
    register_deref_write_external(registry);
    // Foreign Rust-stdlib externals (`core::*` / `std::*` / `fmt::*`)
    // the interpreter calls inline.  Declared opaque here — the rtyper
    // residualizes them rather than tracing into low-level plumbing, the
    // `register_external` / `@jit.dont_look_inside` analog.  Same
    // non-overwriting, between-passes seeding contract as the unsafe-fn
    // stubs above.
    register_foreign_stdlib_externals(registry);
    // Zero-arg Void no-effect externals (`icf_identity_token`). Same
    // between-passes seeding as the stdlib table; jtransform drops the
    // residual (`is_no_effect_external`) instead of emitting empty asm.
    register_no_effect_externals(registry);
    register_runtime_residual_externals(registry);
    // Opaque `f64` methods the front remaps to `ll_math::math_*` C
    // llexternals (`f64_method_llexternal`).  The `std::f64::<Impl>::floor`
    // rows in FOREIGN_STDLIB_EXTERNALS no longer match those callsites;
    // register the emitted crate path from the same table the front reads.
    register_ll_math_llexternals(registry);
    // Acquire-load accessors harvested by
    // `front::mir::collect_atomic_load_declined_fun_decls`.  The gate
    // in `ll_extaccessor` keeps only verified word-only readers
    // (`lowlevel_str_gc_type_id` and its three siblings).  Their
    // addresses are published in `pyre-interpreter` `jit_trace_fnaddrs`
    // (`pa0`), so the residual is a real call rather than a symbolic
    // hash.
    register_atomic_load_llexternals(registry, atomic_load_decls);
    // Foreign opaque-ADT method externals (`<BigInt as Add>::add`, …) the
    // LLBC collected.  `impl_method_owner` declines the Method hint for an
    // opaque owner, so these residualize as `FunctionPath` calls; declare
    // each external here with its faithful result shell.  Same
    // non-overwriting, between-passes seeding contract as the foreign-stdlib
    // and unsafe-fn stubs above.
    register_foreign_opaque_method_externals(registry, foreign_opaque_method_externals);
    // Pass 2 — give each *unique* registry entry the lowered body its
    // default graph is built from. Aliases share one `Rc<FunctionEntry>`,
    // so `Rc::as_ptr` identifies the unique entries.
    //
    // `bookkeeper.py getdesc(pyobj)` makes a `FunctionDesc` without
    // building its graph; `description.py FunctionDesc.cachedgraph` builds
    // it on the first miss (`buildgraph` -> `translator.buildflowgraph`), and
    // a construction error propagates from there. The body is installed on
    // the entry's `FunctionDesc` and lifted when the graph is first
    // observed; a failure is recorded on the entry and surfaces at that
    // `cachedgraph`.
    let mut lifted: HashSet<*const FunctionEntry> = HashSet::with_capacity(by_canonical_path.len());
    // Pre-pass — project each callee's declared LLBC fn-ptr signature onto
    // its entry BEFORE any body is lifted.  The fn-const materialisation
    // fallback (`flowspace_adapter::translate_op`) reads this to type an
    // address-taken JIT-dead method-table entry whose pygraph is lifted but
    // not yet rtyped: at populate time nothing is rtyped, so `getfunctionptr`
    // (which walks the callee's args for their `concretetype`, all `None`)
    // fails on every fn-const and, absent this, poisons every attr-lookup
    // caller.  Runs ahead of the lift loop so a caller lifted early still
    // sees a later callee's declared type.  A non-projectable signature
    // leaves the slot `None` — the fallback declines and the site fails
    // closed exactly as before.
    {
        let mut seeded: HashSet<*const FunctionEntry> =
            HashSet::with_capacity(by_canonical_path.len());
        for (_key, graph, entry, _signature, _body) in &pending {
            if !seeded.insert(Rc::as_ptr(entry)) {
                continue;
            }
            seed_declared_funcptr_type(graph, entry);
        }
    }
    for (key, graph, entry, signature, body) in &pending {
        let entry_ptr = Rc::as_ptr(entry);
        if !lifted.insert(entry_ptr) {
            continue;
        }
        install_source_graph(registry, key, graph, entry, signature, body);
    }
    // Pass 3 — make impl methods visible in their owner's class dict.
    // RPython's ClassDesc reads methods straight off the class object
    // (`classdesc.py find_source_for` → `cls.__dict__[name]`):
    // a Python class object already carries its functions as members
    // when annotation starts.  Pyre's struct-root class objects are
    // minted with no members (`Bookkeeper::intern_class_by_qualname`),
    // so a receiver method call — `CallTarget::Method` lowers to
    // `getattr(recv, name)` + `simple_call`
    // (`flowspace_adapter.rs:1357`) — found no attribute source and
    // blocked.  Registering each `[owner, method]` registry entry's
    // user-function HostObject as a class member completes the
    // analogue: `find_source_for` imports it into the classdict as a
    // Constant, and `s_get_value` binds it under the classdef
    // (`bind_callables_under` → MethodDesc) whose `cachedgraph` hits
    // the pass-2 prefill.  Owner names are gated on the struct-field
    // registry (module and trait leaves are never type roots), and a
    // same-named instance field wins — a function source for a field
    // attribute would union-conflict in `generalize_attr`.
    let bk = registry.bookkeeper();
    for (key, _graph, entry, _signature, _body) in &pending {
        // `CallPath::for_impl_method` splits the module-qualified owner
        // ("pyframe::PyFrame" → [pyframe, PyFrame, method]), so the
        // owner is the second-to-last segment.  Free-function paths
        // ([module, foo]) put a module there, and modules are never
        // type roots — the struct-field-registry gate filters them.
        let segs = key.segments();
        let [.., owner, method] = segs else {
            continue;
        };
        if !bk.is_known_struct_root(owner) || bk.struct_root_has_field(owner, method) {
            continue;
        }
        let class_host = bk.intern_class_by_qualname(owner);
        if class_host.class_get(method).is_none() {
            class_host.class_set(
                method.clone(),
                crate::flowspace::model::ConstValue::HostObject(entry.host_object.clone()),
            );
        }
    }
    let store = function_graphs.clone();
    let globals = Rc::downgrade(registry);
    registry.set_getdesc(Rc::new(move |key| {
        let registry = globals.upgrade()?;
        describe_declared_funcobj(&store, &registry, key)
    }));
    Ok(())
}

/// `bookkeeper.py getdesc` for a funcobj the front end declared: its
/// registry entry, made on the first lookup of the one path its call sites
/// name. `None` when `key` names no declared funcobj.
fn describe_declared_funcobj(
    store: &crate::codewriter::call::GraphStore,
    registry: &Rc<CallRegistry>,
    key: &FunctionPathKey,
) -> Option<Rc<FunctionEntry>> {
    use crate::decline::gate::CALL_REGISTRY as REGISTRY_GATE;
    let path = crate::parse::CallPath::from_segments(key.segments().iter().cloned());
    let graph = store.declared_funcobj(&path)?;
    if let Some(reason) = registration_decline(&canonical_dedup_key(&path)) {
        crate::decline::record(REGISTRY_GATE, reason, format_args!("{path}"));
        return None;
    }
    let signature = crate::codewriter::call::StoreCore::signature_from_graph(&graph);
    let entry = registry.get_or_register_with_func(
        key.clone(),
        signature.clone(),
        source_graph_func(&graph),
    );
    publish_source_signature(&graph, &entry);
    seed_declared_funcptr_type(&graph, &entry);
    let body = store.body(&path).expect("a declared path names a funcobj");
    install_source_graph(registry, key, &graph, &entry, &signature, &body);
    Some(entry)
}

// There is no program-wide `build_program_annotator` /
// `build_program_rtyper` / `ensure_program_specialize` pre-pass and no
// `ProgramSpecializeState` flag.  Per-session
// `specialize_legacy_graph_with_registry_returning_value_to_var`
// runs the
// orthodox flow directly: lift subject graph -> seed subject blocks
// -> compute_at_fixpoint (drives pycall->recursivecall to discover
// callees) -> RPythonTyper::specialize.

/// Lift a pyre `model::FunctionGraph` (a callee that may appear on a
/// `OpKind::Call::FunctionPath` callsite of some other graph) into a
/// `Rc<PyGraph>` suitable for pre-filling the callee's
/// `FunctionDesc.cache` (`description.rs:794`).
///
/// `cachedgraph` (`description.rs:1037-1039`) returns the cached
/// `Rc<PyGraph>` as soon as the lookup key matches, skipping the
/// `buildgraph` path that delegates to
/// `translator.buildflowgraph(pyobj, false)` — that delegation
/// requires a real Python `__code__` body which pyre's surface-DSL
/// callees do not have.  Pyre instead lifts the callee through the
/// adapter and wraps the resulting flowspace `FunctionGraph` in a
/// `PyGraph` here.
///
/// `signature` is the callee's authoritative parameter `Signature`
/// (the same one stored on the `FunctionDesc` in `CallRegistry`).
/// `defaults` is empty — pyre's surface DSL does not yet expose
/// per-parameter default values; when it does, this helper grows a
/// `defaults: Vec<Constant>` parameter mirroring upstream
/// `func.__defaults__`.
///
/// `nested_registry` carries the program-wide `CallRegistry`.
/// For leaf callees (no `OpKind::Call` ops in the body) the registry
/// is unused; nested callees recursively consult it as the adapter
/// processes the callee's own `OpKind::Call` ops.
pub(crate) fn lift_callee_to_pygraph(
    callee_graph: &LegacyGraph,
    signature: Signature,
    nested_registry: &CallRegistry,
) -> Result<Rc<PyGraph>, TyperError> {
    lift_callee_to_pygraph_with_func(
        callee_graph,
        signature,
        nested_registry,
        source_graph_func(callee_graph),
    )
}

fn lift_callee_to_pygraph_with_func(
    callee_graph: &LegacyGraph,
    signature: Signature,
    nested_registry: &CallRegistry,
    func: GraphFunc,
) -> Result<Rc<PyGraph>, TyperError> {
    // The adapter also returns `value_to_var` and `constant_concretetypes`
    // side maps, but they are not consumed here.
    // RPython parity: `Variable.concretetype` and `Constant.concretetype`
    // already carry the per-variable / per-constant LL type after
    // specialise; downstream readers must consult those fields directly
    // (`history.py` `same_constant`, `model.py` `Variable.
    // concretetype`).  The by-slot side map was a pyre-only divergence.
    //
    // Test fixtures that hand-roll minimal SSA shapes must seed
    // each `Variable.annotation` directly via
    // `legacy_annotator::setbinding(&var, ValueType::…)` before calling
    // this so `seed_variable` has type info to attach.
    let FlowspaceAdapterOutput { graph, .. } =
        crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace(
            callee_graph,
            nested_registry,
        )?;
    // The graph and FunctionDesc share the same source callable identity.
    // Synthesizing a second GraphFunc here drops native host evaluators and
    // splits function attributes from the object Bookkeeper.getdesc knows.
    graph.borrow_mut().func = Some(func.clone());
    // TranslationContext.buildflowgraph (translator.py) simplifies a freshly
    // flowed body before publishing it to FunctionDesc.cachedgraph. The native
    // source adapter replaces build_flow, not this stage. In particular its
    // method-call reconstruction leaves unused raw fnptr projections behind;
    // the ordinary dead-op pass removes those while retaining shared reads,
    // effects and raising_op. Do not suppress those reads by trait/name tests.
    crate::translator::simplify::simplify_graph(&graph.borrow(), None);
    let pygraph = Rc::new(PyGraph {
        graph,
        func,
        signature: RefCell::new(signature),
        defaults: RefCell::new(Some(Vec::new())),
        // Carried for the same reason `hints` is, three lines up: this is
        // one graph reached through two representations, so a flag set on
        // the legacy side has to survive the lift. The sibling stub
        // synthesizer below keeps `false` because it has no source graph to
        // read.
        access_directly: Cell::new(callee_graph.access_directly),
    });
    Ok(pygraph)
}

/// Source attributes are installed before Bookkeeper.newfuncdesc chooses
/// its policy, not merely on the separately lifted flow graph afterwards.
fn source_graph_func(graph: &LegacyGraph) -> GraphFunc {
    let mut func = GraphFunc::new(
        graph.name.clone(),
        Constant::new(ConstValue::Dict(HashMap::new())),
    );
    func._always_inline_ = if graph.hints.iter().any(|h| h == "always_inline") {
        AlwaysInline::True
    } else if graph.hints.iter().any(|h| h == "always_inline_try") {
        AlwaysInline::Try
    } else {
        AlwaysInline::Absent
    };
    func._dont_inline_ = graph.func.dont_inline;
    func.annspecialcase = graph
        .hints
        .iter()
        .find(|h| h.starts_with("specialize:"))
        .cloned();
    func
}

/// Synthesize a minimal flowed `PyGraph` for a
/// callee declared by a `LowLevelType` result rather than a
/// `SomeValue` shell — the static [`FOREIGN_STDLIB_EXTERNALS`] table.
/// The stub has a single Link from `startblock` to `returnblock`
/// carrying a pre-annotated `Variable` so the annotator's `flowin`
/// projects the callsite result to the declared return type without
/// touching the (non-modellable) body.
///
/// `name` becomes the synthetic graph's `FunctionGraph.name`.
/// `signature.argnames` are turned into named `Variable`s on the
/// startblock; `return_lltype` is projected through
/// [`crate::translator::rtyper::llannotation::lltype_to_annotation`]
/// to a `SomeValue` shell (no `const_box`) attached to a fresh
/// `Variable` carried on the Link to the returnblock.  Returns
/// `None` when the return type is a container
/// (`Func` / `Struct` / `Array` / `Opaque` / `ForwardReference` /
/// `FixedSizeArray`) or `Address` (no `SomeAddress` port yet) —
/// the caller skips registration for that fn and the original
/// "not registered in CallRegistry" Skip path covers it.
///
/// **Why pre-annotated Variable rather than `Constant`.** A
/// `Constant(default_value)` in the return Link routes through
/// `Bookkeeper::immutableconstant`, which lifts e.g.
/// `ConstValue::Bool(false)` to `SomeBool { const_box = Some(...) }`.
/// That constant slot leaks into the rtyper as a fold-eligible
/// "known false" annotation and can mis-specialise downstream code
/// that observes the callsite result.  Upstream `ExtRegistryEntry.
/// compute_result_annotation` (`extregistry.py`) returns a
/// `SomeXXX()` shell with no `const`; the pre-annotated Variable
/// here carries exactly that shape via `binding(arg)` reading
/// `v.annotation` directly (`annrpython.py`).
///
/// Mirrors how `description.py __init__` test fixtures build a
/// `FunctionDesc` with a minimal `PyGraph` body — the upstream
/// equivalent is `Translator._prebuilt_graphs[entry_point] = pygraph`
/// where `pygraph` is hand-constructed without going through
/// `build_flow`.  Pure constructor — no annotator / rtyper side
/// effects.
pub(crate) fn build_stub_pygraph_for_lltype(
    name: String,
    signature: Signature,
    return_lltype: LowLevelType,
) -> Option<Rc<PyGraph>> {
    let return_someval = default_someshell_for_lltype(&return_lltype)?;
    Some(build_stub_pygraph_with_result_shell(
        name,
        signature,
        return_someval,
    ))
}

/// Build an annotator-only stub PyGraph whose return Variable carries
/// `return_someval` directly.  Shared core of
/// [`build_stub_pygraph_for_lltype`] (which projects an lltype through
/// [`default_someshell_for_lltype`]) and the two shell-carrying paths —
/// the unsafe-fn / `dont_look_inside` residuals via
/// [`residual_return_shell`], and the foreign-opaque-method
/// registration built from the method's faithful result `ValueType`.
/// Both carry the classdef-less `SomeInstance` that an lltype cannot
/// express.  See [`build_stub_pygraph_for_lltype`] for the
/// annotator-only carrier contract.
fn build_stub_pygraph_with_result_shell(
    name: String,
    signature: Signature,
    return_someval: crate::annotator::model::SomeValue,
) -> Rc<PyGraph> {
    let inputargs: Vec<Hlvalue> = signature
        .argnames
        .iter()
        .map(|n| Hlvalue::Variable(Variable::named(n)))
        .collect();
    let startblock = Block::shared(inputargs);
    let func = GraphFunc::new(
        name.clone(),
        Constant::new(ConstValue::Dict(HashMap::new())),
    );
    let mut graph_inner = crate::flowspace::model::FunctionGraph::new(name, startblock.clone());
    graph_inner.func = Some(func.clone());
    let return_var = Variable::named("__unsafe_stub_result");
    *return_var.annotation.borrow_mut() = Some(Rc::new(return_someval));
    let return_hlvalue = Hlvalue::Variable(return_var);
    let link = Rc::new(RefCell::new(Link::new(
        vec![return_hlvalue],
        Some(graph_inner.returnblock.clone()),
        None,
    )));
    startblock.closeblock(vec![link]);
    Rc::new(PyGraph {
        graph: graph_inner.into_ref(),
        func,
        signature: RefCell::new(signature),
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
    })
}

/// Project a `LowLevelType` to a `SomeValue` shell suitable for
/// pre-annotating the stub-graph return Variable
/// ([`build_stub_pygraph_for_lltype`]).
///
/// Delegates to
/// [`crate::translator::rtyper::llannotation::lltype_to_annotation`]
/// for every primitive lltype `lltype_to_annotation` itself handles
/// (Void / Bool / Float family / Char / UniChar / integer family /
/// `Ptr(_)` / `InteriorPtr(_)`), so this helper inherits the
/// upstream `lltype_to_annotation` (`llannotation.py`) shape
/// — every returned `SomeXXX` has no `const_box` set, matching
/// `ExtRegistryEntry.compute_result_annotation` semantics.
///
/// Returns `None` for container types
/// (`Func` / `Struct` / `Array` / `FixedSizeArray` / `Opaque` /
/// `ForwardReference`) that `lltype_to_annotation` rejects.  The caller
/// treats `None` as "skip this fn"; the unported path then surfaces
/// the original "not registered" Skip.
///
/// `Address` maps to the untyped `SomeAddress` shell (`llmemory.py
/// SomeAddress`) — the annotation-stage projection a `*const T` / `*mut T`
/// raw pointer carries when its element type is not modeled, for a residual
/// whose result is consumed only as a residual-call argument.
pub(crate) fn default_someshell_for_lltype(
    lltype: &LowLevelType,
) -> Option<crate::annotator::model::SomeValue> {
    if lltype.is_container_type() {
        return None;
    }
    match lltype {
        LowLevelType::Address => Some(crate::annotator::model::SomeValue::Address(
            crate::annotator::model::SomeAddress::new(),
        )),
        _ => Some(crate::translator::rtyper::llannotation::lltype_to_annotation(lltype.clone())),
    }
}

/// Decode a residual callee's FUNC.RESULT token into the `SomeValue` shell
/// its stub graph returns, or `None` when the token is one this path cannot
/// model faithfully.
///
/// The token is the canonical spelling
/// [`crate::front::mir::dont_look_inside_return_token`] stamps from the
/// callee's return `ValueType`.  Scalar tokens decode to their primitive
/// lltype (`getkind` equals the legacy walker's
/// `valuetype_to_concrete(result_ty)`).
///
/// The `ref` token (a non-`*mut PyObject` heap result — e.g. a `*mut BigInt`,
/// or the one-word success payload of a `Result<_, PyError>`) carries a
/// classdef-less `SomeInstance` shell (`valuetype_to_someshell(Ref(None))`),
/// NOT a `SomePtr(GCREF)`.  A residual result is an annotation-stage value;
/// the annotation-stage projection of every `Ref` is the classdef-less
/// `SomeInstance` (`annotation_state.rs` doc-table), with `SomePtr` reserved
/// for a producer writing `Variable.annotation` at the rtyper stage.
/// Emitting a `SomePtr(GCREF)` here made the result collide with the
/// `SomeInstance(None)` an ordinary `Ref` merge partner carries (`_ptr ∪
/// <other>` UnionError in `mergeinputargs`, the union has no `SomePtr ∪
/// SomeInstance` arm).  The `SomeInstance(None)` shell unions cleanly (the
/// classdef-less widen arm) and is the same shell the sibling
/// [`register_foreign_opaque_method_externals`] path already gives a
/// `BigInt`-returning method residual.  Both spellings rtype to `getkind =
/// GcRef`, so the emitted `residual_call_*_r` opcode and C ABI are
/// byte-identical.
///
/// The `str` token (`STR_RETURN_TYPE`) is an rstr `STR` result — `ll_int2dec`,
/// a `W_UnicodeObject._utf8` accessor — and carries the `SomeString` shell
/// (`valuetype_to_someshell(Str)`), the annotation the callee body produces.
/// Projected through `ref`, it met the `SomeString` of a `StringBuilder.build()`
/// at the same `space.newutf8` argument as a `String ∪ Instance` UnionError.
///
/// The object-pointer marker (`OBJECTPTR_RETURN_TYPE`, stamped by
/// `SemanticFunctionHeader::new` for a `*mut PyObject`-returning opaque callee the
/// front-end left untokened) keeps the typed `OBJECTPTR` lltype so a caller
/// reading the returned object's fields rtypes against the real struct.
///
/// `i128` / `u128` are deliberately declined: they have no `getkind` at the
/// codewriter boundary (`history.getkind` rejects `SignedLongLongLong`, 16
/// bytes), so a caller annotating one would only reach a deeper wall.
pub(crate) fn residual_return_shell(
    token: Option<&str>,
) -> Option<crate::annotator::model::SomeValue> {
    if token == Some(GCREF_RETURN_TYPE) {
        return Some(
            crate::translator::rtyper::llannotation::lltype_to_annotation(
                crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
            ),
        );
    }
    if token == Some("ref") {
        return Some(
            crate::codewriter::annotation_state::valuetype_to_someshell(
                &crate::model::ValueType::Ref(None),
            )
            .expect("Ref(None) shells to a definite SomeInstance"),
        );
    }
    if token == Some(STR_RETURN_TYPE) {
        return crate::codewriter::annotation_state::valuetype_to_someshell(
            &crate::model::ValueType::Str,
        );
    }
    return_token_to_lltype(token).and_then(|ll| default_someshell_for_lltype(&ll))
}

/// Choose the annotation carried by a `dont_look_inside` stub's return link.
///
/// Normally the front-end's FUNC.RESULT token is the semantic annotation
/// source.  The two exception materialisers are the deliberate exception:
/// their `*mut PyObject` token describes only the residual trampoline ABI,
/// while their registered `_signature_` describes the interpreter-visible
/// exception instance.  The cached stub must agree with that signature, just
/// as an upstream flow graph constrained by `_signature_` does.
fn residual_stub_result_shell(
    function_key: &FunctionPathKey,
    token: Option<&str>,
) -> Option<crate::annotator::model::SomeValue> {
    if is_exception_object_materializer(function_key) {
        Some(exception_object_result_annotation())
    } else {
        residual_return_shell(token)
    }
}

/// The result a `_signature_` declares for a residual callee whose `ref`
/// token alone would shell it as a classdef-less instance: `SomeString` for
/// a low-level `STR` pointer (`FunctionGraph::return_is_str`), `SomeList`
/// for a pointer onto a Rust container (`FunctionGraph::return_container_root`,
/// the `GcArray` a field of that type already projects), else
/// `SomeInstance(classdef)` of a by-value ADT result
/// (`FunctionGraph::return_class_root`), whose payload reads
/// (`__discriminant`, `__pos_0`) the classdef-less shell cannot annotate.
///
/// A one-word-item `Vec` residual is `SomeRustVec` (`project_rust_vec`).
/// Its FUNC.RESULT token is `i64` (`history.getkind` of the raw header),
/// so the `ref` gate would miss it and the stub would be `SomeInteger` —
/// `Integer ∪ RustVec` at a `look_inside_iff` wrapper return
/// (`ll_vec_alloc_and_set_nonnull_r`).
/// `None` when the token is not `ref` (and not that `Vec` header) or the
/// graph declares neither.
fn declared_return_annotation(
    registry: &CallRegistry,
    graph: &LegacyGraph,
) -> Option<crate::annotator::model::SomeValue> {
    if let Some(spelling) = graph.return_container_root.as_deref() {
        let projected = registry.bookkeeper().project_struct_field_type(spelling);
        if matches!(projected, crate::annotator::model::SomeValue::RustVec(_)) {
            return Some(projected);
        }
    }
    if graph.return_type.as_deref() != Some("ref") {
        return None;
    }
    if graph.return_is_str {
        return crate::codewriter::annotation_state::valuetype_to_someshell(
            &crate::model::ValueType::Str,
        );
    }
    if let Some(spelling) = graph.return_container_root.as_deref() {
        let projected = registry.bookkeeper().project_struct_field_type(spelling);
        if !matches!(projected, crate::annotator::model::SomeValue::Impossible) {
            return Some(projected);
        }
    }
    let root = graph.return_class_root.as_deref()?;
    let bk = registry.bookkeeper();
    // A by-value Raw ADT is Ptr of that struct (`history.getkind`),
    // the same shell FieldRead and a `raw:<owner>` token already use.
    // SomeInstance here made mergeinputargs raise Instance(X) ∪ Ptr(X)
    // (`llannotation.py` pairtype(SomePtr, SomeObject).union).
    if let Some(cell) = bk.raw_struct_ptr_annotation(root) {
        return Some(cell);
    }
    let classdef = bk
        .getuniqueclassdef(&bk.intern_class_by_qualname(root))
        .ok()?;
    Some(crate::annotator::model::SomeValue::Instance(
        crate::annotator::model::SomeInstance::new(
            Some(classdef),
            false,
            std::collections::BTreeMap::new(),
        ),
    ))
}

/// Owner leaf of a `raw:<owner>` FUNC.RESULT token. `None` for every
/// other spelling, including a bare `raw:`.
pub(crate) fn raw_owner_return_root(token: &str) -> Option<&str> {
    let rest = token.strip_prefix("raw:")?;
    if rest.is_empty() { None } else { Some(rest) }
}

/// `SomePtr` for a `raw:<owner>` token when that owner's layout is registered.
/// `None` when the token is not a raw-pointer result or the layout is absent;
/// callers then keep [`residual_return_shell`]'s integer shell.
fn raw_ptr_token_annotation(
    registry: &CallRegistry,
    token: Option<&str>,
) -> Option<crate::annotator::model::SomeValue> {
    let root = raw_owner_return_root(token?)?;
    registry.bookkeeper().raw_struct_ptr_annotation(root)
}

/// Project a FUNC.RESULT token (the `return_type` string) to its
/// `LowLevelType`.  `None`/`"()"` → `Void`; `"ref"` and the `*mut PyObject`
/// token → `OBJECTPTR`; primitive tokens map directly; unrecognised → `None`
/// (decline).  `i128`/`u128` have no token spelling here and so decline for
/// the same `getkind`-at-the-codewriter reason [`residual_return_shell`]
/// documents.  `raw:<owner>` admits as `Signed` (the pointer's `getkind`);
/// the `SomePtr` shell is [`raw_ptr_token_annotation`].
fn return_token_to_lltype(token: Option<&str>) -> Option<LowLevelType> {
    match token {
        None | Some("()") => Some(LowLevelType::Void),
        Some("ref") => Some(crate::translator::rtyper::rclass::OBJECTPTR.clone()),
        Some(s) if s == GCREF_RETURN_TYPE => {
            Some(crate::translator::rtyper::lltypesystem::lltype::GCREF.clone())
        }
        Some(s) if s == STR_RETURN_TYPE => {
            Some(crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone())
        }
        Some("bool") => Some(LowLevelType::Bool),
        Some("i64") => Some(LowLevelType::Signed),
        // `raw:<owner>` is a pointer-to-Raw-T. The lltype here is only the
        // gate's admission shell (`getkind` Signed). The stub's annotation
        // is `SomePtr`, resolved once layouts are on the bookkeeper.
        Some(s) if raw_owner_return_root(s).is_some() => Some(LowLevelType::Signed),
        Some("u64") => Some(LowLevelType::Unsigned),
        Some("f64") => Some(LowLevelType::Float),
        Some(s) if s == OBJECTPTR_RETURN_TYPE => {
            Some(crate::translator::rtyper::rclass::OBJECTPTR.clone())
        }
        _ => None,
    }
}

/// Project a MIR `ValueType` to its rtyped `LowLevelType`, mirroring the
/// rtyper's lowering of the `valuetype_to_someshell` annotation shell:
/// `Str` → the string pointer, `Ref`/`State` → the erased object pointer
/// (`OBJECTPTR`, what a classdef-less `SomeInstance` rtypes to), `Unknown`
/// declines.  Used to build a callee's declared fn-ptr arg signature
/// (`getfunctionptr` / `FuncType.RESULT` from the return annotation).
pub(crate) fn valuetype_to_lltype(vt: &crate::model::ValueType) -> Option<LowLevelType> {
    use crate::model::ValueType;
    Some(match vt {
        ValueType::Int => LowLevelType::Signed,
        ValueType::Unsigned => LowLevelType::Unsigned,
        ValueType::Int128 => LowLevelType::SignedLongLongLong,
        ValueType::UInt128 => LowLevelType::UnsignedLongLongLong,
        ValueType::Bool => LowLevelType::Bool,
        ValueType::Float => LowLevelType::Float,
        ValueType::SingleFloat => LowLevelType::SingleFloat,
        ValueType::Str => crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone(),
        ValueType::StringBuilder => {
            crate::translator::rtyper::lltypesystem::rbuilder::STRINGBUILDERPTR.clone()
        }
        ValueType::Ref(_) | ValueType::State => {
            crate::translator::rtyper::rclass::OBJECTPTR.clone()
        }
        ValueType::Void => LowLevelType::Void,
        ValueType::Unknown => return None,
    })
}

/// Project a callee's declared fn-ptr `FuncType` from its `FunctionGraph`
/// startblock `Input` ValueTypes (in `inputargs` order) + return token,
/// mirroring `derive_subject_inputcells`'s startblock reading.  Returns
/// `None` if any arg lacks a startblock `Input` op or an unprojectable
/// ValueType, or the return token is unrecognised — the fn-const
/// materialisation fallback then declines (fail-closed).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn declared_funcptr_type_from_legacy(
    legacy: &LegacyGraph,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::FuncType> {
    use crate::model::OpKind;
    let startblock = legacy.blocks.iter().find(|b| b.id == legacy.startblock)?;
    let mut input_ty: HashMap<
        crate::flowspace::model::Variable,
        (&crate::model::ValueType, &Option<String>),
    > = HashMap::new();
    for op in &startblock.operations {
        if let (Some(result), OpKind::Input { ty, class_root, .. }) = (op.result.as_ref(), &op.kind)
        {
            input_ty.insert(result.clone(), (ty, class_root));
        }
    }
    let mut args = Vec::with_capacity(startblock.inputargs.len());
    for var in &startblock.inputargs {
        let (ty, class_root) = input_ty.get(var)?;
        args.push(if class_root.as_deref() == Some("GCREF") {
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone()
        } else {
            valuetype_to_lltype(ty)?
        });
    }
    // A `STR`-pointer result rtypes as `SomeString` (`declared_return_annotation`),
    // so its function type returns `Ptr(STR)` like the callable's typed
    // return variable (`rtyper.py getcallable`), not the token's `OBJECTPTR`.
    let result = if legacy.return_is_str && legacy.return_type.as_deref() == Some("ref") {
        crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone()
    } else {
        return_token_to_lltype(legacy.return_type.as_deref())?
    };
    Some(crate::translator::rtyper::lltypesystem::lltype::FuncType { args, result })
}

/// Register a batch of annotator-only residual
/// `(segments, signature, FUNC.RESULT token)` specs into `registry`.
/// Each entry is wrapped through [`residual_return_shell`] +
/// [`build_stub_pygraph_with_result_shell`]
/// + [`CallRegistry::register_callee`], so subsequent
///   `flowspace_adapter::translate_op` lookups via
///   `call_registry.lookup_with_leaf_match` find a registered entry and
///   the dual gate no longer Skips with "not registered in
///   CallRegistry" for these paths.
///
/// `specs` is the Charon/LLBC-sourced carrier assembled from unsafe path
/// aliases, `dont_look_inside` declarations, and marked constructors. Per-fn
/// failures ([`residual_return_shell`]
/// declines the token, or registry already has the same key
/// at a conflicting signature) propagate as silent skips — the
/// upstream "not registered" Skip path then absorbs that specific fn
/// while the rest of the batch lands.
///
/// Mirrors `populate_call_registry_from_call_graphs`'s "register and
/// prefill" contract but feeds from declarations rather than from the
/// `function_graphs` `GraphStore`.  The two key
/// differently, so both can hold the same fn; `CallControl::unsafe_fn_stubs`
/// carries the measured split.
///
/// **Annotator-only carrier — never reaches the codewriter.**
/// The stub graph carries a single return link holding a pre-annotated
/// Variable, suitable for `RPythonAnnotator` return-type inference via
/// `cachedgraph` (see `call_registry::prefill_default_cache`).  It
/// cannot become a compile candidate because it never lands in the
/// container the codewriter reads: `build_stub_pygraph_with_result_shell`
/// builds a `flowspace::model::FunctionGraph`, which
/// `PyreFunctionEntry::prefill_default_cache` parks in the entry's
/// `FunctionDesc.cache`, while `CallControl::find_all_graphs` and
/// `CallControl::target_to_path_and_graph` resolve candidates out of
/// `function_graphs` and hand `CodeWriter::transform_graph_to_jitcode` a
/// `crate::model::FunctionGraph`.  Nothing on this path can put a stub
/// there: this function takes only `&CallRegistry`, its caller holds
/// `function_graphs` by shared reference, and `GraphStore::insert` needs
/// `&mut self`.
pub(crate) fn register_unsafe_fn_stubs(
    registry: &CallRegistry,
    specs: &[(Vec<String>, Signature, Option<String>)],
) {
    use crate::decline::gate::CALL_REGISTRY as REGISTRY_GATE;
    for (segments, signature, return_token) in specs {
        // `collect_unsafe_fn_stubs_from_llbc` feeds opaque foreign decls
        // (`core::ptr::copy_nonoverlapping`, `std::alloc::alloc_zeroed`)
        // here, before they ever become a `SemanticFunction`. Declining
        // that path lets Layer-1 `call_registry.lookup` miss, so the
        // callsite reaches `host_env_callable`. The same carrier also
        // holds the residual `allocate` / `allocate_stable` stubs from
        // `collect_marked_class_ctor_stubs_from_llbc`.
        // `registration_decline` drops those as user graphs so this stub
        // can occupy the key; declining them here leaves the key empty.
        let path = crate::parse::CallPath::from_segments(segments.iter().cloned());
        if let Some(reason) = registration_decline(&canonical_dedup_key(&path))
            && (reason == SKIP_PTR_COPY_NONOVERLAPPING || reason == SKIP_STD_ALLOC_RAW)
        {
            crate::decline::record(REGISTRY_GATE, reason, format_args!("{path}"));
            continue;
        }
        let Some(result_shell) = raw_ptr_token_annotation(registry, return_token.as_deref())
            .or_else(|| residual_return_shell(return_token.as_deref()))
        else {
            continue;
        };
        let stub_pygraph = build_stub_pygraph_with_result_shell(
            segments.last().cloned().unwrap_or_default(),
            signature.clone(),
            result_shell,
        );
        let key = FunctionPathKey::from_segments(segments.iter().cloned());
        if registry.lookup(&key).is_some() {
            // Already registered via the function_graphs pass (e.g. a
            // safe fn with the same path).  Don't overwrite.
            continue;
        }
        registry.register_callee(key, signature.clone(), stub_pygraph);
    }
}

/// Foreign Rust-stdlib helper paths the interpreter source calls inline
/// (`core::*` / `std::*` / `alloc::*` / `fmt::*`) that the rtyper must
/// treat as opaque externals rather than trace into.
///
/// RPython's interpreter never traces into its low-level plumbing — C
/// helpers are declared with `register_external` / `rffi.llexternal`
/// (`extfunc.py` / `rffi.py`) so the annotator sees a
/// `SomeExternalFunction` with a *declared* result annotation and the
/// JIT residualizes the call (the same effect `@jit.dont_look_inside`
/// has on a graph).  pyre's interpreter is written in Rust, so the Rust
/// stdlib is pulled into the lifted call tree; the analog is to declare
/// these foreign paths external here.  Each entry mirrors a
/// `register_external(function, args, result)` declaration: a callee
/// path, the call-site arity, and the result lltype the residual call
/// produces.
///
/// The entries are recurring foreign paths the prepass reaches on cold /
/// opaque plumbing branches whose declared result lltype is faithful to
/// the Rust signature (the residual call really does produce that type):
///
/// - `core::mem::swap(&mut T, &mut T)` returns `()` — an in-place swap,
///   `Void` result.
/// - `core::cell::Cell::<T>::set(&self, val)` returns `()` — an
///   in-place store into the cell, `Void` result.
/// - `core::f64::<Impl>::is_infinite` / `core::f64::<Impl>::is_nan` /
///   `core::slice::<Impl>::is_empty` return `bool` — `Bool` result.
/// - `std::f64::<Impl>::floor` / `std::f64::<Impl>::powf` /
///   `core::f64::<Impl>::abs` return `f64` — `Float` result.
/// - `core::f64::<Impl>::NAN` / `core::f64::<Impl>::INFINITY` are the
///   associated `f64` constants, reached as zero-arg `FunctionPath`
///   accessors — `Float` result, empty arg list. These plus the `f64`
///   predicates above close the complex-division (`_Py_c_quot` /
///   `_Py_c_prod`) foreign-leaf cluster.
/// - `core::f64::<Impl>::to_bits` returns `u64` reinterpreted as `i64` —
///   `Signed` result.
///
/// `<[T]>::as_ptr` / `Vec::as_ptr` are NOT registered here: in the erased
/// array-pointer value-model the slice's data pointer is the receiver value,
/// so they fold to the receiver in the front end
/// (`Lowering::is_container_as_ptr_identity`, twin of the `as_slice` and
/// `NonNull::as_ptr` identity folds), leaving no residual to annotate.  Paths
/// whose faithful result is a non-scalar value the stub carrier still cannot
/// express (`alloc::fmt::format` → `String`,
/// `core::array::iter::<Impl>::into_iter` → an iterator with no Repr,
/// `core::num::<Impl>::checked_*` → `Option<i64>`) stay intentionally
/// absent: registering them with a placeholder `Void`/`Signed` result
/// mis-models the value and only migrates the failure to a deeper
/// annotation wall, so they stay residual at the "not registered" Skip
/// until their result type can be modeled.
const FOREIGN_STDLIB_EXTERNALS: &[(&[&str], &[&str], LowLevelType)] = &[
    (&["core", "mem", "swap"], &["x", "y"], LowLevelType::Void),
    // `mem::drop(x)` runs `x`'s destructor and returns `()`; the trace model
    // does not model destructors (the `Drop` terminator lowers to a
    // pass-through `Goto`), so a Void no-op stub is faithful.
    (&["core", "mem", "drop"], &["x"], LowLevelType::Void),
    // `mem::forget(x)` consumes `x` without running its destructor and returns
    // `()`; same Void no-op shape as `mem::drop`.
    (&["core", "mem", "forget"], &["x"], LowLevelType::Void),
    // `mem::size_of::<T>() -> usize`.  A concrete-type call is folded to a
    // `ConstInt` in the front end (`Lowering::lower_call` /
    // `size_align_const_from_tyexpr`); the residual reaching the rtyper is the
    // generic body (`gc_alloc_storage_box<T>`), whose `T` is an abstract type
    // parameter with no compile-time size.  Declare it opaque so the generic
    // body annotates — the monomorphized machine code the JIT actually traces
    // carries rustc's folded constant, so no residual call runs.  Both
    // spellings: the FunDecl name resolves `core`, the call site may write the
    // `std` re-export.
    (&["core", "mem", "size_of"], &[], LowLevelType::Unsigned),
    (&["std", "mem", "size_of"], &[], LowLevelType::Unsigned),
    // `handle_alloc_error(layout) -> !` aborts on OOM; it never returns and
    // its result is never consumed. The divergence is carried by the CFG, so
    // a Void stub lets the alloc helpers lift while the residual call aborts.
    (
        &["alloc", "alloc", "handle_alloc_error"],
        &["layout"],
        LowLevelType::Void,
    ),
    (
        &["sync", "atomic", "AtomicBool", "store"],
        &["self", "val", "order"],
        LowLevelType::Void,
    ),
    (
        &["sync", "atomic", "AtomicU8", "store"],
        &["self", "val", "order"],
        LowLevelType::Void,
    ),
    (
        &["cell", "Cell", "set"],
        &["self", "val"],
        LowLevelType::Void,
    ),
    // `Cell::get(&self) -> T` copies the cell.  The residual is the scalar
    // load; census uses are `Cell<i32>` (`cerrno` alt-errno), so `Signed`.
    (&["cell", "Cell", "get"], &["self"], LowLevelType::Signed),
    // `AtomicUsize` is layout-transparent over `usize`; `new(v)` yields that
    // inner unsigned word.
    (
        &["sync", "atomic", "AtomicUsize", "new"],
        &["v"],
        LowLevelType::Unsigned,
    ),
    (
        &["sync", "atomic", "AtomicUsize", "swap"],
        &["self", "val", "order"],
        LowLevelType::Unsigned,
    ),
    (
        &["sync", "atomic", "AtomicUsize", "store"],
        &["self", "val", "order"],
        LowLevelType::Void,
    ),
    // `AtomicPtr::swap` returns the previous `*mut T` as an untyped address.
    (
        &["sync", "atomic", "AtomicPtr", "swap"],
        &["self", "val", "order"],
        LowLevelType::Address,
    ),
    (
        &["sync", "atomic", "AtomicPtr", "new"],
        &["v"],
        LowLevelType::Address,
    ),
    (
        &["sync", "atomic", "AtomicI64", "new"],
        &["v"],
        LowLevelType::Signed,
    ),
    (
        &["core", "f64", "<Impl>", "is_infinite"],
        &["self"],
        LowLevelType::Bool,
    ),
    (
        &["core", "slice", "<Impl>", "is_empty"],
        &["self"],
        LowLevelType::Bool,
    ),
    // `<[T]>::contains(&self, x: &T) -> bool` scans the slice for an element
    // equal to `*x`.  Bool scalar result; the receiver is a slice value (same
    // shape `is_empty` already residualises) and `x` is a by-reference operand,
    // both passed through to the residual call.  The membership scan runs in
    // the residual, so no element `PartialEq` graph is demanded here.
    (
        &["core", "slice", "<Impl>", "contains"],
        &["self", "x"],
        LowLevelType::Bool,
    ),
    (
        &["std", "f64", "<Impl>", "floor"],
        &["self"],
        LowLevelType::Float,
    ),
    (
        &["std", "f64", "<Impl>", "ceil"],
        &["self"],
        LowLevelType::Float,
    ),
    // Fractional part. `complex_richcompare` uses it to test that a
    // float lane is integral before the integer compare. Same opaque
    // float residual as `floor` / `ceil`; the defining path is `core`.
    (
        &["std", "f64", "<Impl>", "fract"],
        &["self"],
        LowLevelType::Float,
    ),
    (
        &["core", "f64", "<Impl>", "fract"],
        &["self"],
        LowLevelType::Float,
    ),
    (
        &["std", "f64", "<Impl>", "powf"],
        &["self", "n"],
        LowLevelType::Float,
    ),
    (
        &["std", "f64", "<Impl>", "fract"],
        &["self"],
        LowLevelType::Float,
    ),
    (
        &["core", "f64", "<Impl>", "fract"],
        &["self"],
        LowLevelType::Float,
    ),
    (
        &["core", "f64", "<Impl>", "to_bits"],
        &["self"],
        LowLevelType::Signed,
    ),
    (
        &["core", "f64", "<Impl>", "abs"],
        &["self"],
        LowLevelType::Float,
    ),
    (
        &["core", "f64", "<Impl>", "is_nan"],
        &["self"],
        LowLevelType::Bool,
    ),
    (&["core", "f64", "<Impl>", "NAN"], &[], LowLevelType::Float),
    (
        &["core", "f64", "<Impl>", "INFINITY"],
        &[],
        LowLevelType::Float,
    ),
    // `Vec::set_len(&mut self, new_len)` is a foreign leaf (Charon does
    // not extract `alloc`). The residual mutates length in place and
    // returns `()`.
    (
        &["vec", "Vec", "set_len"],
        &["self", "new_len"],
        LowLevelType::Void,
    ),
];

/// Zero-arg Void externals with no effects and no graph.
///
/// Analog of `extfunc.py register_external(function, [], None,
/// sandboxsafe=True)` whose body is a no-op. Charon leaves
/// `majit_ir::icf::icf_identity_token` Foreign: the `#[oopspec]` /
/// `#[dont_look_inside]` macros insert it as empty `nomem` asm so LLVM
/// MergeFunctions cannot fold helper bodies. Upstream `isconstant` has
/// no such call (`rlib/jit.py` returns `NonConstant(False)`); the
/// annotator must see that body, and jtransform must drop the token
/// the way `rewrite_op_getfield` returns on `RESULT is lltype.Void`.
const NO_EFFECT_EXTERNALS: &[(&[&str], &[&str], LowLevelType)] = &[(
    &["majit_ir", "icf", "icf_identity_token"],
    &[],
    LowLevelType::Void,
)];

/// Residual runtime helpers the interpreter spells in source. Analog of
/// `extfunc.py register_external` / `@jit.dont_look_inside`: the annotator
/// sees an opaque stub so the caller graph can finish, and the codewriter
/// residualizes the real call. `set_gc_interp` arms a breaker bit;
/// `slot_from_cached` is the shadow-stack slot handle.
const RUNTIME_RESIDUAL_EXTERNALS: &[(&[&str], &[&str], LowLevelType)] = &[
    (
        &["majit_ir", "eval_breaker_word", "set_gc_interp"],
        &[],
        LowLevelType::Void,
    ),
    (
        &["majit_gc", "shadow_stack", "slot_from_cached"],
        &["raw"],
        LowLevelType::Address,
    ),
    (
        &["majit_gc", "shadow_stack", "cache_root_stack_slot"],
        &["cached"],
        LowLevelType::Void,
    ),
    (
        &["majit_gc", "shadow_stack", "shadow_stack_slot"],
        &[],
        LowLevelType::Address,
    ),
    // `GcAllocOutcome::allocated_or_abort` returns `Option<*mut u8>`.
    // The frontend may spell the enum type as the sole path prefix.
    (
        &["GcAllocOutcome", "allocated_or_abort"],
        &["self", "payload_size"],
        LowLevelType::Address,
    ),
    // `OnceLock<fn(...)>` for the `sys.exc_info` JIT-direct helper.
    (
        &["pyre_interpreter", "function", "SYS_EXC_INFO_DIRECT_FN"],
        &[],
        LowLevelType::Address,
    ),
];

/// True when `segments` is a [`NO_EFFECT_EXTERNALS`] path.
///
/// jtransform consults this instead of residualizing the empty asm:
/// `rewrite_op_direct_call` returns `Replace([])`, matching
/// `jtransform.py rewrite_op_getfield` `if RESULT is lltype.Void: return`.
pub(crate) fn is_no_effect_external(segments: &[String]) -> bool {
    NO_EFFECT_EXTERNALS.iter().any(|(path, _, _)| {
        path.len() == segments.len() && path.iter().zip(segments.iter()).all(|(a, b)| *a == b)
    })
}

/// Register the [`NO_EFFECT_EXTERNALS`] table into `registry`.
///
/// Same annotator-only, non-overwriting `register_opaque_external`
/// carrier as [`register_foreign_stdlib_externals`]. The codewriter
/// drops matching calls rather than residualizing them.
pub(crate) fn register_no_effect_externals(registry: &CallRegistry) {
    for (segments, argnames, return_lltype) in NO_EFFECT_EXTERNALS {
        register_opaque_external(registry, segments, argnames, return_lltype.clone());
    }
}

/// Register the [`RUNTIME_RESIDUAL_EXTERNALS`] table into `registry`.
pub(crate) fn register_runtime_residual_externals(registry: &CallRegistry) {
    for (segments, argnames, return_lltype) in RUNTIME_RESIDUAL_EXTERNALS {
        register_opaque_external(registry, segments, argnames, return_lltype.clone());
    }
    // `sys_exc_info_code` loads the process-global BuiltinCode pointer
    // (`function.py` `space._code_of_sys_exc_info`). Residual; the
    // result is a `PyObjectRef`.
    register_opaque_external(
        registry,
        &["pyre_interpreter", "function", "sys_exc_info_code"],
        &[],
        crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
    );
}

/// Register one opaque external through the annotator-only stub carrier.
/// The caller retains ownership of which table or synthetic path declares
/// the external; this helper keeps their registration semantics identical.
fn register_opaque_external(
    registry: &CallRegistry,
    segments: &[&str],
    argnames: &[&str],
    return_lltype: LowLevelType,
) {
    let signature = Signature::new(
        argnames.iter().map(|n| (*n).to_string()).collect(),
        None,
        None,
    );
    let Some(stub_pygraph) = build_stub_pygraph_for_lltype(
        segments.last().copied().unwrap_or_default().to_string(),
        signature.clone(),
        return_lltype,
    ) else {
        return;
    };
    let key = FunctionPathKey::from_segments(segments.iter().map(|s| s.to_string()));
    if registry.lookup(&key).is_some() {
        return;
    }
    registry.register_callee(key, signature, stub_pygraph);
}

/// Declare the opaque MIR deref-write marker to the annotator.
///
/// This is a two-parameter, `Void` external.  It is intentionally registered
/// only in the `CallRegistry`: the codewriter's graph registry has no body for
/// it and therefore sends the original `Call(__deref_write)` through
/// `handle_residual_call`, preserving the store as a carried side effect.
pub(crate) fn register_deref_write_external(registry: &CallRegistry) {
    register_opaque_external(
        registry,
        &["__deref_write"],
        &["ptr", "value"],
        LowLevelType::Void,
    );
}

/// Residual `core::num::<Impl>::saturating_{add,sub,mul}` stubs keyed by
/// receiver knowntype (`specialize.py specialize_argtype` /
/// `SomeInteger(knowntype=...)` / `rarithmetic.r_uint`).  Charon's
/// `<Impl>` path is shared across integer widths; a single Unsigned
/// stub would annotate `i64::saturating_mul` as `r_uint`.
fn register_saturating_arith_externals(registry: &CallRegistry) {
    use crate::translator::rtyper::call_registry::CORE_NUM_SATURATING_ARITH_LEAVES;
    for leaf in CORE_NUM_SATURATING_ARITH_LEAVES {
        for (knowntype, return_lltype) in [
            ("int", LowLevelType::Signed),
            ("r_uint", LowLevelType::Unsigned),
        ] {
            register_opaque_external(
                registry,
                &["core", "num", "<Impl>", knowntype, leaf],
                &["self", "rhs"],
                return_lltype,
            );
        }
    }
}

/// Register the [`FOREIGN_STDLIB_EXTERNALS`] table into `registry`.
///
/// Faithful analog of `extfuncregistry.py`'s `_register` table being
/// walked into the annotator's external registry: each foreign path is
/// declared as an opaque external with a fixed signature, registered
/// through the same opaque stub-pygraph carrier
/// [`register_unsafe_fn_stubs`] uses, so subsequent
/// `flowspace_adapter::translate_op` `FunctionPath` lookups resolve the
/// callee instead of raising "not registered in CallRegistry".  Like
/// `register_unsafe_fn_stubs` the registration is annotator-only and
/// non-overwriting: a path already registered by the `function_graphs`
/// or unsafe-fn pass wins, and the codewriter residualizes the call
/// through its fnaddr lowering (these foreign stubs are never present in
/// `CallControl::function_graphs`, so they never compile into JITCode).
pub(crate) fn register_foreign_stdlib_externals(registry: &CallRegistry) {
    for (segments, argnames, return_lltype) in FOREIGN_STDLIB_EXTERNALS {
        register_opaque_external(registry, segments, argnames, return_lltype.clone());
    }
    register_saturating_arith_externals(registry);
    // `<[T]>::to_vec` leftover the pair-slice rewrite did not retarget
    // (`front::mir` `is_slice_to_vec` only fires on an enabled pair
    // receiver). Analog of `extfunc.py register_external` returning a
    // list: the residual is an owned Vec, a GC pointer.
    register_opaque_external(
        registry,
        &["alloc", "slice", "<Impl>", "to_vec"],
        &["self"],
        crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
    );
}

/// Register every `f64_method_llexternal` name as `["ll_math", name]`.
///
/// The front rewrites opaque `f64::{floor,ceil,hypot,…}` calls to
/// `FunctionPath { segments: ["ll_math", "math_floor"] }` (the C
/// llexternal).  Walking `F64_METHOD_LLEXTERNALS` — the same table
/// `f64_method_llexternal` is — keeps that path resolvable without a
/// hand list at this call site.  The annotator result is `Float` for
/// every row: that is the C llexternal / front `result_ty`, matching
/// `register_external(..., [float], float)` even when the `ll_math_*`
/// wrapper returns `Result<f64, MathError>`.  Arity comes from the
/// table (unary vs `hypot`/`atan2`/`copysign`/`pow`).
pub(crate) fn register_ll_math_llexternals(registry: &CallRegistry) {
    use crate::translator::rtyper::lltypesystem::module::ll_math::F64_METHOD_LLEXTERNALS;
    for row in F64_METHOD_LLEXTERNALS {
        let argnames: &[&str] = match row.arity {
            1 => &["x"],
            2 => &["x", "y"],
            arity => panic!(
                "F64_METHOD_LLEXTERNALS[{}]: llexternal arity must be 1 or 2, got {arity}",
                row.method
            ),
        };
        register_opaque_external(
            registry,
            &["ll_math", row.name],
            argnames,
            LowLevelType::Float,
        );
    }
}

/// Register Acquire-load accessors whose body the MIR loop declined, as
/// `register_external` / CallRegistry stubs.
///
/// Walks [`crate::translator::rtyper::lltypesystem::module::ll_extaccessor::collect_atomic_load_llexternals`]:
/// only a verified word-only reader (zero-arg unsigned result,
/// ordered-load decline, no translatable body, path whose source is
/// exactly `AtomicU32::load(Acquire)`) is admitted.  Same
/// non-overwriting, annotator-only carrier as
/// [`register_ll_math_llexternals`].  The residual call is the real
/// function; the stub annotation is a non-const scalar, never a
/// translation-time constant.
pub(crate) fn register_atomic_load_llexternals(
    registry: &CallRegistry,
    decls: &[crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl],
) {
    use crate::translator::rtyper::lltypesystem::module::ll_extaccessor::collect_atomic_load_llexternals;
    let collected = collect_atomic_load_llexternals(decls);
    let _ = crate::translator::rtyper::extfuncregistry::register_atomic_load_accessor_externals(
        &collected,
    );
    for row in &collected {
        let argname_storage: Vec<String> = (0..row.arity()).map(|i| format!("arg{i}")).collect();
        let argnames: Vec<&str> = argname_storage.iter().map(String::as_str).collect();
        let segments: Vec<&str> = row.segments.iter().map(String::as_str).collect();
        register_opaque_external(registry, &segments, &argnames, row.result.clone());
    }
}

/// Register the foreign opaque-ADT method externals
/// [`crate::front::mir::collect_foreign_opaque_method_externals`] harvested
/// from the LLBC.
///
/// Faithful analog of `register_external` for a method on a foreign opaque
/// type: the JIT does not trace into `<BigInt as Add>::add` (the
/// interpreter's cold `@jit.dont_look_inside` overflow→long arm), it emits
/// a residual call.  [`crate::front::mir::Lowering::impl_method_owner`]
/// declines the `CallTarget::Method` hint for an opaque owner so the call
/// lowers as `CallTarget::FunctionPath`; registering each path here lets
/// the residual lookup resolve instead of raising "not registered in
/// CallRegistry" (and avoids the classdef-less `SomeInstance.getattr`
/// panic the Method form would surface).
///
/// The result shell comes from the per-method faithful `ValueType` the
/// collector read off the LLBC output signature — a `Ref(None)`
/// classdef-less `SomeInstance` for a `BigInt`-returning method (matching
/// `bigint_from`), a `SomeInstance(classdef)` of the interned enum-base
/// for a payload-enum return (`GcAllocOutcome::from_hook`), or a scalar
/// shell for an integer/float/bool return.  Same non-overwriting,
/// annotator-only-carrier contract as [`register_foreign_stdlib_externals`].
pub(crate) fn register_foreign_opaque_method_externals(
    registry: &CallRegistry,
    externals: &[(Vec<String>, Signature, crate::model::ValueType)],
) {
    for (segments, signature, result_ty) in externals {
        let Some(return_someval) = foreign_opaque_method_return_shell(registry, result_ty) else {
            continue;
        };
        let key = FunctionPathKey::from_segments(segments.iter().cloned());
        if registry.lookup(&key).is_some() {
            continue;
        }
        let stub_pygraph = build_stub_pygraph_with_result_shell(
            segments.last().cloned().unwrap_or_default(),
            signature.clone(),
            return_someval,
        );
        registry.register_callee(key, signature.clone(), stub_pygraph);
    }
}

/// Annotator shell for a harvested foreign-opaque residual result.
///
/// `valuetype_to_someshell(Ref)` always drops the classdef
/// (`annotation_state.rs`), which is the right shell for an opaque
/// `BigInt` return.  A payload-carrying enum (`GcAllocOutcome`) needs
/// the interned enum-base ClassDef so a caller `match` can getattr
/// `__discriminant` (`rclass.py` `InstanceRepr.rtype_getattr`).
/// [`declared_return_annotation`] is the wrong intern here: a raw-word
/// payload enum is a `raw_word_owner`, so that helper prefers
/// `raw_struct_ptr_annotation` (`SomePtr`) over `SomeInstance`.
fn foreign_opaque_method_return_shell(
    registry: &CallRegistry,
    result_ty: &crate::model::ValueType,
) -> Option<crate::annotator::model::SomeValue> {
    if let crate::model::ValueType::Ref(Some(root)) = result_ty {
        let bk = registry.bookkeeper();
        let host = bk.intern_class_by_qualname(root);
        if let Ok(classdef) = bk.getuniqueclassdef(&host) {
            return Some(crate::annotator::model::SomeValue::Instance(
                crate::annotator::model::SomeInstance::new(
                    Some(classdef),
                    false,
                    std::collections::BTreeMap::new(),
                ),
            ));
        }
    }
    crate::codewriter::annotation_state::valuetype_to_someshell(result_ty)
}

/// Test-only restricted entry: only graphs without `OpKind::Call::FunctionPath`
/// ops resolve through this path.  An empty `CallRegistry` is
/// constructed internally and shared with the annotator; any
/// `simple_call` op the adapter emits would surface a fail-loud
/// `TyperError` at `flowspace_adapter::translate_op`'s
/// `FunctionPath` arm because there is no `(HostObject, FunctionDesc)`
/// pair to resolve the callable to.
///
/// The upstream
/// `Constant(<function>) -> getdesc -> FunctionDesc` chain
/// (`bookkeeper.py`) is unreachable from this entry because
/// pyre's surface DSL has no Python callable object to resolve;
/// production callers that need free function calls must build a
/// pre-populated registry (one entry per reachable `FunctionPath`,
/// with `register_callee(key, signature, lifted_pygraph)`) and call
/// [`specialize_legacy_graph_with_registry_returning_value_to_var`] directly.  Until the
/// production walker that traverses `SemanticProgram.functions`
/// lands, this entry remains in-place for anchor tests and
/// dual-gate validation against graphs that have no `Call` ops.
#[cfg(test)]
pub fn specialize_legacy_graph(
    legacy: &LegacyGraph,
) -> Result<(LegacyToTyped, HashMap<Variable, LowLevelType>), TyperError> {
    let registry = crate::translator::rtyper::call_registry::CallRegistry::new(Rc::new(
        crate::annotator::bookkeeper::Bookkeeper::new(),
    ));
    specialize_legacy_graph_with_registry_returning_value_to_var(legacy, &registry)
}

/// `specialize_legacy_graph_with_registry` extended return shape that
/// also surfaces the [`LegacyToTyped`] map and the per-`Variable`
/// `Constant.concretetype` table produced by the flowspace adapter.
/// Codewriter callers consume `value_to_var` directly: the rtyper's
/// typed `Variable.concretetype` writes are copied onto the matching
/// legacy Variables by
/// [`crate::codewriter::type_state::apply_from_flowspace_variables`].
/// `constants` feeds [`project_value_to_var`] for the
/// dual-gate baseline comparison.
///
/// Test fixtures that hand-roll minimal SSA shapes without
/// production-shape `OpKind::Input { ty }` ops must seed
/// `legacy.variable(vid).annotation` directly via
/// `legacy_annotator::setbinding(&var, ValueType::...)` before calling
/// this so `seed_variable` has type info to attach.
#[allow(dead_code)] // flowspace adapter Constant.concretetype / value_to_var
pub fn specialize_legacy_graph_with_registry_returning_value_to_var(
    legacy: &LegacyGraph,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
) -> Result<(LegacyToTyped, HashMap<Variable, LowLevelType>), TyperError> {
    let (_graph, value_to_var, _value_to_var_candidates, constant_concretetypes, _) =
        drive_subject(legacy, call_registry, true)?;
    Ok((value_to_var, constant_concretetypes))
}

/// Annotate one legacy subject against the shared session annotator/rtyper,
/// optionally rtyping it in the same call.
///
/// `do_rtype = true` is the fused per-graph dual-gate path: annotate-half +
/// `specialize_more_blocks` + lltype validation, committing the subject's
/// blocks after rtype (the original behaviour, byte-identical).
///
/// `do_rtype = false` is the two-phase Phase-A annotate-only path: it runs the
/// annotate-half and commits the subject's blocks into the shared session
/// WITHOUT rtyping, so a later whole-program Phase-B `specialize_more_blocks`
/// rtypes every annotated block at once — after all callers of a shared callee
/// (e.g. `type_error`) have unioned its arg to a general `SomeString`, which is
/// what removes the `fixed_graphs` late-stage-modify Skip. Returns the lifted
/// flowspace `GraphRef` so the two-phase cache can record its `GraphKey`.
fn drive_subject(
    legacy: &LegacyGraph,
    call_registry: &crate::translator::rtyper::call_registry::CallRegistry,
    do_rtype: bool,
) -> Result<
    (
        crate::flowspace::model::GraphRef,
        LegacyToTyped,
        LegacyToTypedCandidates,
        HashMap<Variable, LowLevelType>,
        HashMap<Variable, Hlvalue>,
    ),
    TyperError,
> {
    // RPython parity path.
    //
    // Upstream `RPythonTyper.specialize` runs ONCE per `Translator`,
    // not per graph.  Pyre's per-graph dual-gate enters this function
    // once per graph (one per `transform_graph_to_jitcode`); the
    // upstream "specialize-once" semantics are reproduced by sharing
    // one annotator + rtyper across subjects through
    // `CallRegistry::ensure_session` and rtyping only the newly
    // added blocks each entry.  Already-cached PyGraphs hold
    // post-specialize LL ops; re-seeding them into the per-session
    // annotator would let `specialize_more_blocks` walk the LL ops a
    // second time and trip on `unimplemented operation: 'int_add'` /
    // `'direct_call'`.
    // ── Step 1 — adapter ─────────────────────────────────────────────
    let FlowspaceAdapterOutput {
        graph,
        mut value_to_var,
        value_to_var_candidates,
        mut constant_concretetypes,
        mut constant_hlvalues,
        ..
    } = crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace(
        legacy,
        call_registry,
    )?;

    // Same buildflowgraph stage as the cached-callee path above. Do not wait
    // until annotator.simplify: unused native projection preimages can block
    // annotation before that later cleanup is reachable.
    crate::translator::simplify::simplify_graph(&graph.borrow(), None);

    // ── Step 2 — annotator surface ────────────────────────────────
    //
    // The annotator + rtyper are
    // session-shared through `CallRegistry::ensure_session`,
    // mirroring RPython's "one annotator + one rtyper per Translator"
    // (`translator.py:69-83`).  Each per-graph subject seeds its
    // blocks into the shared annotator; subsequent
    // `specialize_more_blocks` rtype only the newly-added blocks.
    let (annotator, rtyper) = call_registry.ensure_session()?;
    // Scope an `added_blocks` tracker over this subject's drive of the
    // shared annotator so the post-drain blocked-annotation guard
    // (`raise_if_subject_blocked` below) can see exactly the blocks
    // seeded for this subject and roll them back if any stays in the
    // `None` (False) sentinel state. Mirrors `complete_helpers`'s
    // `saved = self.added_blocks; self.added_blocks = {}` prologue
    // (annrpython.py); the guard restores the previous tracker
    // when this function returns.
    let _subject_block_scope = annotator.enter_added_blocks_scope();
    // Queue `graph.startblock` onto the orthodox `addpendingblock`
    // queue, mirroring how callees enter through
    // `pycall -> recursivecall -> addpendingblock`
    // (`description.py`, `annrpython.py`).
    // `addpendingblock` (`annrpython.rs:1245-1302`) writes
    // `Variable.annotation` for each inputarg via
    // `bindinputargs.setbinding`, inserts `all_blocks[startblock]`
    // and `annotated[startblock] = None`, then
    // `schedulependingblock` queues the block.
    // `complete_pending_blocks` below drains the queue:
    // `processblock` flips `annotated[block] = Some(graph)`,
    // `flowin` walks every op and recurses into successors via
    // `process_link -> addpendingblock(target, inputs_s)`
    // (`annrpython.rs:1502/1690`).  Transitive blocks register
    // themselves into `all_blocks`/`annotated` through the same
    // bindinputargs path on first arrival.
    let subject_inputcells =
        crate::translator::rtyper::flowspace_adapter::derive_subject_inputcells(
            legacy,
            Some(call_registry.bookkeeper()),
        )?;
    let (startblock, exceptblock) = {
        let g = graph.borrow();
        (g.startblock.clone(), g.exceptblock.clone())
    };
    let subject_inputcells_opt: Vec<Option<crate::annotator::model::SomeValue>> =
        subject_inputcells.into_iter().map(Some).collect();
    annotator.addpendingblock(&graph, &startblock, &subject_inputcells_opt);
    // Pre-register the eagerly-created `exceptblock`
    // (`flowspace/model.py:21-25` parity) — non-raising subjects
    // never reach it through a `Link`, but `specialize_more_blocks`
    // must walk it so its inputargs receive an exception-typed
    // `concretetype`.  The rtyper's `setup_block_entry`
    // exception-block branch (`rtyper.rs:1915-1942`) writes
    // `Variable.concretetype = ExceptionData.lltype_of_exception_*`
    // without reading `Variable.annotation`, so the block needs no
    // `flowin`; it only needs to appear in `annotator.annotated`.
    //
    // Additionally, seed each inputarg with a `setbinding`-level
    // annotation (`Impossible` when none is already present) so that
    // a raising subject's `follow_raise_link` — which reaches the
    // same exceptblock via `addpendingblock` with `seen_before=true`
    // and routes to `mergeinputargs` — does not panic on the unbound
    // `seed_variable(legacy_v)` inputargs from
    // `flowspace_adapter.rs:1839`.  `setbinding` widens via the
    // lattice's `contains` check, so an `Integer`-already-annotated
    // slot stays `Integer`; a `None`-annotated slot becomes
    // `Impossible`, which any later `mergeinputargs` widens to the
    // real cell.  The `annotated[block] = Some(graph)` bare insert
    // (the next block) keeps the pre-fix `specialize_block` walk
    // semantic intact — the block is treated as already annotated
    // and never enters `complete_pending_blocks`.
    {
        let blk = exceptblock.borrow();
        let inputargs: Vec<crate::flowspace::model::Hlvalue> = blk.inputargs.clone();
        drop(blk);
        for a in &inputargs {
            if let crate::flowspace::model::Hlvalue::Variable(v) = a {
                let needs_seed = v.annotation.borrow().is_none();
                if needs_seed {
                    let mut tmp = v.clone();
                    annotator.setbinding(&mut tmp, crate::annotator::model::SomeValue::Impossible);
                }
            }
        }
    }
    {
        let bkey = crate::flowspace::model::BlockKey::of(&exceptblock);
        annotator
            .annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(graph.clone()));
        annotator
            .all_blocks
            .borrow_mut()
            .insert(bkey, exceptblock.clone());
    }

    // Callee blocks are seeded naturally with no explicit pre-seed:
    // `@op.simple_call.register(SomeObject)` (unaryop.py
    // parity) makes flowin dispatch the first
    // `simple_call(host_object_const, args)` op through `s_func.call(
    // argspec)` -> `SomePBC.call` -> `Bookkeeper.pbc_call` ->
    // `FunctionDesc.pycall` -> `annotator.recursivecall`, which calls
    // `addpendingblock(graph, startblock, inputcells)` for the
    // callee.

    // Addpendingblock conversion — drain the pending
    // queue so `flowin` walks every block reachable from
    // `graph.startblock` and writes `Variable.annotation` for each
    // op result.  Mirrors upstream `RPythonAnnotator.complete()`
    // (`annrpython.py`), which loops `complete_pending_blocks`
    // → policy hook → exit-on-empty.  Without this drain, the
    // `addpendingblock(startblock, inputcells)` queued just above
    // stays in `genpendingblocks`, `annotated[block]` remains the
    // `None` sentinel, and `specialize_block` (`rtyper.rs:1656`)
    // panics on "annotator.annotated[block] is False sentinel".
    annotator
        .complete_pending_blocks()
        .map_err(|err| TyperError::message(format!("complete_pending_blocks failed: {err}")))?;
    // Orthodox `complete()` blocked-annotation guard
    // (annrpython.py): a block that stayed in the `None` (False)
    // sentinel state after the fixpoint drain is permanently
    // `BlockedInference`. Upstream `complete()` raises here, before
    // `simplify`/`compute_at_fixpoint`. The dual-gate calls
    // `complete_pending_blocks` (the bare drain) instead of
    // `complete()`, so the guard is replicated here; the error reuses
    // the "complete_pending_blocks failed" prefix already recognised by
    // `is_known_unported`, routing the subject to the legacy walker.
    annotator
        .raise_if_subject_blocked()
        .map_err(|err| TyperError::message(format!("complete_pending_blocks failed: {err}")))?;

    // Populate per-callsite call-family / calltable state
    // by walking the seeded blocks' call_ops.  `compute_at_fixpoint`
    // (`bookkeeper.py`, pyre `bookkeeper.rs:627-648`) drains
    // `annotator.call_sites()` through `consider_call_site`
    // (`bookkeeper.py`, pyre `bookkeeper.rs:675`); each
    // `simple_call(callable_const, *args)` op resolves the callable
    // to a `SomePBC` (via `immutablevalue_hostobject` for the
    // pre-registered `HostObject::UserFunction`), then records the
    // call site in `bookkeeper.pbc_maximal_call_families` so the
    // rtyper's `FunctionRepr.call(hop)` (`rpbc.py`) finds the
    // matching call-family row at `find_row` time.
    //
    // The subject just seeded into this annotator is brand-new and its
    // `simple_call` ops have not been processed yet; this drain
    // processes the subject's (and any newly seeded callees')
    // call_sites so their rows land in `pbc_maximal_call_families`
    // before specialize looks them up.
    //
    // Errors propagate verbatim — upstream
    // `bookkeeper.py` runs the call-site walk without a
    // `try`/`except`, so a failed `consider_call_site` terminates
    // `simplify` and unwinds out of the annotator driver.  Pyre's
    // port surfaces the same condition through `?`-propagation
    // here rather than swallowing it at `bookkeeper.rs:627-648`.
    call_registry
        .bookkeeper()
        .compute_at_fixpoint()
        .map_err(|err| TyperError::message(format!("compute_at_fixpoint failed: {err}")))?;

    // ── Step 3 — incremental rtyper drive ──────────────────────────
    //
    // Lifecycle mapping vs upstream `RPythonTyper.specialize(self,
    // dont_simplify_again=False)` (rtyper.py):
    //
    //   1. `if not dont_simplify_again: self.annotator.simplify()`
    //      — pyre per-subject flow keeps the annotator-wide
    //        simplification and extra-pass work skipped: the subject was
    //        lifted via the adapter and its block list is already in
    //        canonical shape.  `transform_allocate` is applied per block
    //        immediately before specialization instead (see
    //        `specialize_block`).
    //   2. `self.exceptiondata.finish(self)` — hoisted into
    //      `CallRegistry::ensure_session` so it runs **once** at
    //      session start (idempotent — `getclassrepr` is cached on
    //      `rtyper.instance_reprs`).
    //   3. `self.already_seen = {}` — pyre keeps `already_seen`
    //      across per-subject calls so previously-rtyped subjects
    //      are not re-walked.
    //   4. `self.specialize_more_blocks()` — invoked here per
    //      subject; walks annotator.annotated\\already_seen.
    //   5. `self.exceptiondata.make_helpers(self)` — currently
    //      no-op (R4 / `MixLevelHelperAnnotator` not ported), see
    //      module header on `translator/rtyper/rclass.rs`.
    //   6. second `specialize_more_blocks()` for helpers — redundant
    //      while make_helpers is a no-op.
    //
    // Step 4 is the only per-subject step; steps 2/5/6 are
    // session-global one-shots, step 1/3 are pyre-incremental
    // adaptations.  Once R4 lands, `make_helpers` migrates to
    // `ensure_session` after `finish_exceptiondata` (single
    // session-prologue), keeping the per-subject body unchanged.
    //
    // In the two-phase path (`do_rtype = false`) the rtype-half is
    // deferred to the whole-program Phase-B `specialize_more_blocks`
    // (see `run_two_phase_prepass`); this call only annotates, leaving
    // the subject's blocks in `annotated` for that single later pass.
    if do_rtype {
        rtyper.specialize_more_blocks()?;

        // ── Step 4 — validate per-slot lltype projection ──────────────
        //
        // No eager per-slot side table is built here.  Each callable that
        // needs a kind view derives one on demand from `value_to_var` +
        // `constant_concretetypes` via
        // [`project_value_to_var`] — same `Variable.concretetype`
        // / `Constant.concretetype` ground truth, just routed at the
        // consumer instead of eagerly materialised here.  Run a
        // validation pass so unsupported lltypes still surface as a
        // [`TyperError`] at this boundary (a fail-loud `?` propagation).
        for var in value_to_var.values() {
            let cell = var.concretetype.borrow();
            if let Some(lltype) = cell.as_ref() {
                lowleveltype_to_concrete(lltype)?;
            }
        }
        for lltype in constant_concretetypes.values() {
            lowleveltype_to_concrete(lltype)?;
        }
    }

    // The subject specialized cleanly — keep its blocks in the shared
    // annotator (they are now `already_seen`). Every earlier `?` exit
    // above leaves `_subject_block_scope` uncommitted, so a Skipped
    // subject's partially-annotated blocks are evicted on drop and never
    // poison a later subject's session-global `specialize_more_blocks`.
    _subject_block_scope.commit();

    // Inert opname-reachability gauge over the rtyped flowspace graph this
    // subject specialized cleanly. The two-phase path reports after Phase B;
    // `do_rtype = false` is only Phase A and must not expose high-level
    // operations as a post-rtype lowering backlog.
    if do_rtype && crate::codewriter::jtransform_shadow::is_enabled() {
        crate::codewriter::jtransform_shadow::report_if_enabled(&graph.borrow());
    }

    if do_rtype {
        select_rtyped_representatives(&mut value_to_var, &value_to_var_candidates)?;
        reconcile_projection_aliases(
            legacy,
            &mut value_to_var,
            &mut constant_hlvalues,
            &mut constant_concretetypes,
        );
    }
    Ok((
        graph,
        value_to_var,
        value_to_var_candidates,
        constant_concretetypes,
        constant_hlvalues,
    ))
}

/// Whole-program two-phase rtyper prepass.
///
/// Mirrors upstream `translator.annotate()` → `rtyper.specialize()`
/// (`driver.py:325/345`): Phase A annotates EVERY graph in the portal closure
/// to fixpoint — sharing the session annotator so all callers of a shared
/// callee (e.g. `type_error`) union its arg to a general `SomeString` — THEN
/// Phase B rtypes every annotated block in one pass. The per-graph publish
/// ([`dual_gate_outcome_from_cache`]) reads the cached result instead of
/// re-running the fused per-graph path, so no callee is rtyped before all its
/// callers are annotated, removing the `fixed_graphs` late-stage-modify Skip.
///
/// Populates `call_registry.two_phase()`. The prepass is fully panic-contained
/// and ALWAYS sets `prepass_done = true`: `make_jitcodes` runs in a spawned
/// worker thread, so any escaping panic (the annrpython late-stage `panic!`, a
/// `call_all_setups` failure, an unported repr setup) would unwind the thread
/// and fail the build. The publish path ([`dual_gate_outcome_from_cache`]) is
/// session-independent — it reads cached `Variable`s and re-runs the legacy
/// walker, never the flowspace session — so a partial cache from an aborted
/// prepass still yields Match for the graphs that completed and a safe
/// legacy-walker Skip for the rest (zero regression, never the poisoned session).
pub(crate) fn run_two_phase_prepass(
    call_registry: &CallRegistry,
    candidate_graphs: &HashSet<crate::parse::CallPath>,
    function_graphs: &crate::codewriter::call::GraphStore,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_two_phase_prepass_inner(call_registry, candidate_graphs, function_graphs)
    }));
    call_registry.two_phase().prepass_done = true;
}

/// Text of a caught annotator or rtyper unwind.
///
/// `simple_call_SomeObject` raises `AnnotatorError` from
/// `compute_at_fixpoint`. `panic_payload_text` reads that object; a
/// string downcast misses it and the phase line cannot name the failure.
/// `binding` panics with a `String`. Capture the unwind origin so a
/// phase-B `KeyError` names the rtyper frame, not only the caller's block.
fn install_binding_backtrace_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let text = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or("");
        if text.contains("no binding for arg") {
            eprintln!(
                "[PREPASS binding-backtrace] {text}\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
        prev(info);
    }));
}

fn unwind_panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    crate::annotator::model::panic_payload_text(payload)
        .unwrap_or_else(|| "<non-string panic>".to_string())
}

/// Whether the `MAJIT_RTYPER_VERBOSE` de-aggregating census is enabled.
/// Matches the `== "1"` contract the codewriter's coverage gauge uses
/// (`codewriter.rs`), so a literal `MAJIT_RTYPER_VERBOSE=0` stays off.
/// Cached: the `Ref` shell projector consults this ~17 000 times per census.
pub(crate) fn rtyper_verbose_enabled() -> bool {
    thread_local! {
        static ON: bool =
            std::env::var_os("MAJIT_RTYPER_VERBOSE").is_some_and(|v| v == "1");
    }
    ON.with(|on| *on)
}

/// Map a captured prepass failure reason to its orthodox-disposition
/// bucket — the named [`is_known_unported`] category (or, for the
/// FunctionPath-not-registered family, the unregistered callee's nature)
/// whose port closes it. Turns the opaque "two-phase Skip" count into a
/// ranked, finite legacy-walker-deletion worklist: each bucket names a
/// concrete piece of work (port a Repr, lower a construct in `front::mir`,
/// residual-ize a helper, specialize a generic ADT).
/// Whether a `pair(s1, s2).union() found no arm: X ∪ Y` message names two
/// operands of different kinds.
///
/// The operands are rendered by `annotator::model`'s `union_operand_id`, which
/// spells a parameterised kind `Kind(detail)` and a bare one `Kind`, so the
/// kind is the text before the first `(`.  Two `Instance(..)`s of different
/// classes are the SAME kind — upstream's `pairtype(SomeInstance, SomeInstance)
/// .union` handles that pair, and its failure is a class-hierarchy question,
/// not a missing arm.
///
/// A message this cannot parse is reported as same-kind, so an unreadable one
/// lands in the bucket that asks for work rather than the one that closes the
/// question.
fn union_pair_is_cross_kind(reason: &str) -> bool {
    let Some((lhs, rhs)) = reason
        .rsplit_once("found no arm: ")
        .and_then(|(_, pair)| pair.split_once(" ∪ "))
    else {
        return false;
    };
    let kind = |s: &str| s.trim().split('(').next().unwrap_or("").trim().to_string();
    let (lhs, rhs) = (kind(lhs), kind(rhs.lines().next().unwrap_or(rhs)));
    !lhs.is_empty() && !rhs.is_empty() && lhs != rhs
}

fn classify_unported_reason(reason: &str) -> &'static str {
    // ── rtyper-stage named gaps (most specific first) ──
    if reason.contains("rtyper_makerepr — port") {
        "REPR-PORT (rlist/rdict/robject/iterator)"
    } else if reason.contains("cannot unify instances with no common base class")
        || reason.contains("don't know how to convert from")
    {
        "GENERIC-ADT-SPECIALIZE (per-instantiation classdef)"
    } else if reason.contains("rtype_cast_ptr_to_int")
        || reason.contains("is of instances of the non-pointers")
        || reason.contains("noneify() not supported")
    {
        "FRONTEND-TYPED-PTR (address-of-local / host-static / null)"
    } else if reason.contains("survived fuse_boxing_alloc unfused") {
        // A GC allocation reached the adapter without the frontend proving
        // its header/type identity and lowering it to `NewWithVtable`.
        "UNFUSED-MALLOC (unproven GC allocation header)"
    } else if reason.contains("dict key eq function not wired") {
        // `OrderedDictRepr::require_direct_compare_key` fail-closed gate —
        // a dict key repr with a custom `get_ll_eq_function` (str, instance)
        // reached getitem/contains before the call-based keyeq branch of
        // `build_ll_dict_lookup_helper_graph` was ported. #140 DictRepr epic.
        "DICT-KEY-EQ (str/instance dict keys)"
    } else if reason.contains("Call with CallTarget::Indirect") {
        "DYN-TRAIT-INDIRECT (lower_indirect_calls)"
    } else if reason.contains("pair(s1, s2).union() found no arm") {
        // Two dispositions share one message, and they call for opposite
        // work.  Upstream's union table is same-kind almost throughout, so a
        // cross-kind pair is one `pairtype(SomeObject, SomeObject).union`
        // raises for as well: nothing to port, the merge itself is the
        // defect.  A same-kind pair is the arm pyre has not written yet.
        if union_pair_is_cross_kind(reason) {
            "UNION-CROSS-KIND (a phi upstream never builds)"
        } else {
            "UNION-PAIR-PORT (same-kind arm not written)"
        }
    } else if reason.contains("rbase missing") {
        "REPR-SETUP-ORDER (call_all_setups)"
    } else if reason.contains("calltable row not found in CallFamily") {
        "RPBC-CALLFAMILY (seeded-method registration)"
    } else if reason.contains("InstanceRepr.convert_const") {
        "FIELD-PROJECTION (typed-Ref field rows)"
    } else if reason.contains("Cannot find attribute ") {
        // find_method has no arm for this method on a TYPED builtin repr
        // (`next` on SomeIterator, `len`/`swap`/`as_slice` on SomeList,
        // `get` on SomeInteger, …). A method-resolution gap on a typed
        // repr — NOT a classdef-less typed-Ref. (`next` on Iterator is the
        // iter_next vertical; the List/Integer/Dict/String rows are
        // find_method method-table gaps.)
        "METHOD-RESOLUTION-GAP (find_method on builtin repr)"
    } else if reason.contains("classdef-less instance") {
        // getattr/method on a SomeInstance seeded with classdef=None — the
        // receiver is a FOREIGN pointee (BigInt `to_f64`/`clone`, Wtf8Buf
        // `as_str`, Vec/slice `deref`/`len`/`iter`). Resolved by
        // residualizing/seeding the foreign type, NOT by ClassDef
        // projection: the real pyre structs are already registry-known, so
        // there is no missing struct to register here.
        "CLASSDEF-LESS-FOREIGN (foreign-type getattr)"
    } else if reason.contains("MissingRTypeAttribute") {
        // rtyper getclsfield walked rbase to the root class without finding
        // the attr (e.g. the `ob_type`-on-`PyType` wrong-resolution: PyType
        // has no ob_type field, so this is a mis-resolution, not a real
        // projectable field). A class-field walk miss, distinct from an
        // annotation-stage classdef-less instance.
        "GETCLSFIELD-MISS (rtyper class-field walk)"
    } else if reason.contains("KeyError: no binding for arg")
        || reason.contains("inputarg lacks annotation")
        || (reason.contains("variable ") && reason.contains(" used before definition"))
    {
        "ANNOTATION-SLOT-GAP (cross-block threading)"
    }
    // ── FunctionPath-not-registered family, split by callee nature ──
    else if reason.contains("not registered in CallRegistry")
        || reason.contains("translate_op")
        || reason.contains("cachedgraph")
    {
        if reason.contains("core::ptr")
            || reason.contains("from_raw_parts")
            || reason.contains("box_assume_init")
            || reason.contains("mut_ptr")
            || reason.contains("const_ptr")
        {
            "FRONTEND-RAWPTR (raw_store/raw_load op)"
        } else if reason.contains("GcType")
            || reason.contains("type_id")
            || reason.contains("__dyn_call")
        {
            "FRONTEND-VTABLE (gc_id / indirect_call)"
        } else if reason.contains("SHADOW_STACK")
            || reason.contains("TL_")
            || reason.contains("TYPEOBJECT_CACHE")
            || reason.contains("W_TYPE_TYPEOBJECT")
            || reason.contains("DETACHED_RECURSION")
            || reason.contains("PENDING_EXCEPTION")
        {
            "FRONTEND-THREADLOCAL/ONCELOCK (threadlocalref_get)"
        } else if reason.contains("into_iter")
            || reason.contains("reverse")
            || reason.contains("chars")
            || reason.contains("::iter")
        {
            "FRONTEND-ITERATOR (Vec-as-array / next protocol)"
        } else if reason.contains("bigint")
            || reason.contains("malachite")
            || reason.contains("indexmap")
            || reason.contains("IndexMap")
        {
            "RESIDUAL-FOREIGN (dont_look_inside + fnaddr)"
        } else if (reason.contains("bool") && reason.contains("then"))
            || reason.contains("filter")
            || reason.contains("FnMut")
            || reason.contains("FnOnce")
        {
            "RESIDUAL-CLOSURE (dont_look_inside)"
        } else {
            "FUNCPATH-OTHER (registry / residual)"
        }
    }
    // ── annotator-stage catch-alls ──
    else if reason.contains("UnionError") {
        "UNION-ERROR (generic-ADT phi / pair)"
    } else if reason.contains("setbinding: new value does not contain old") {
        // A binding moved backwards in the lattice: a variable's
        // `SomeInteger.knowntypedata` (or instance) annotation refined from
        // a generic to a per-instantiation classdef between fixpoint passes
        // (`Option` -> `Option<Result<*mut PyObject,PyError>>`), so
        // `contains` (union == self) is false. Same per-instantiation
        // classdef monotonicity as the mergeinputargs `UnionError` bucket —
        // fixed by the same generic-ADT union work, so it shares the label.
        "UNION-ERROR (generic-ADT phi / pair)"
    } else if reason.contains("Blocked block -- operation cannot succeed") {
        "BLOCKED-BLOCK (annotation dead-end downstream)"
    } else if reason.contains("compute_at_fixpoint failed")
        || reason.contains("complete_pending_blocks failed")
        || reason.contains("AnnotatorError")
    {
        "ANNOTATOR-FIXPOINT-OTHER (no inner cause surfaced)"
    } else {
        "UNCLASSIFIED"
    }
}

/// Print a sorted disposition histogram for a prepass phase's failures
/// (`MAJIT_RTYPER_VERBOSE`). The ranked buckets are the legacy-walker
/// deletion worklist; as each disposition's port lands its bucket shrinks.
fn emit_disposition_histogram(phase: &str, reasons: &[String]) {
    let mut counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for r in reasons {
        *counts.entry(classify_unported_reason(r)).or_insert(0) += 1;
    }
    let mut rows: Vec<(&'static str, usize)> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    eprintln!(
        "[PREPASS histogram {phase}] {} failures by orthodox disposition:",
        reasons.len()
    );
    for (cat, n) in rows {
        eprintln!("[PREPASS histogram {phase}]   {n:>4}  {cat}");
    }
}

// Exists to localise prepass nondeterminism (gh#1139).
fn emit_determinism_trace(phase: &str, index: usize, canonical_key: &str) {
    eprintln!(
        "[DTRACE] {phase} {index} {canonical_key} var={} const={} reflow={} block={} rf_list={} rf_sub={} rf_attr={} rf_pbc={} rf_notify={} widen={} nupd={} breuse={} greuse={} nhitreused={}",
        crate::flowspace::model::next_var_id(),
        crate::flowspace::model::next_constant_id(),
        crate::annotator::annrpython::reflow_count(),
        crate::annotator::annrpython::processblock_count(),
        crate::annotator::listdef::reflow_from_listitem_count(),
        crate::annotator::classdesc::reflow_from_subclass_count(),
        crate::annotator::bookkeeper::reflow_from_attr_count(),
        crate::annotator::bookkeeper::reflow_from_pbc_count(),
        crate::annotator::annrpython::reflow_from_notify_count(),
        crate::annotator::listdef::listitem_widen_count(),
        crate::annotator::listdef::listitem_notify_update_count(),
        crate::flowspace::model::block_addr_reuse_count(),
        crate::flowspace::model::graph_addr_reuse_count(),
        crate::annotator::annrpython::notify_hit_on_reused_count(),
    );
}

/// `warmspot.py WarmRunnerDesc.check_access_directly_sanity`: no graph
/// outside the JIT graph set is `access_directly`.
///
/// `specialize.py default_specialize` (`description.rs`) writes the flag on
/// the annotator's graphs, and upstream's codewriter reads it after the
/// annotator ran. Here `find_all_graphs` runs before the prepass annotates,
/// so the flag is first readable after Phase A. Upstream walks
/// `collect_called_graphs(entry_point_graph)`; the graphs the annotator
/// reached are the ones its `FunctionDesc`s cached. The JIT graph set is
/// taken per function: a candidate `{name}__AccessDirect_None` path maps
/// to that function's FunctionDesc (the `(AccessDirect, key)` PyGraph
/// lives on the same desc as the regular graph).
fn check_access_directly_sanity(
    call_registry: &CallRegistry,
    candidate_graphs: &HashSet<crate::parse::CallPath>,
) {
    let jit_descs: HashSet<*const RefCell<crate::annotator::description::FunctionDesc>> =
        candidate_graphs
            .iter()
            .filter_map(|path| {
                let origin = crate::codewriter::call::access_directly_origin_path(path)
                    .unwrap_or_else(|| path.clone());
                call_registry.lookup(&FunctionPathKey::from_segments(
                    origin.segments.iter().cloned(),
                ))
            })
            .map(|entry| Rc::as_ptr(&entry.function_desc))
            .collect();
    for entry in call_registry.bookkeeper().descs.borrow().values() {
        let crate::annotator::description::DescEntry::Func(func) = entry else {
            continue;
        };
        let desc = func.func();
        if jit_descs.contains(&Rc::as_ptr(&desc)) {
            continue;
        }
        let desc = desc.borrow();
        // `default_specialize` writes `(AccessDirect, key)` onto a callee
        // the annotator reached with flagged args. That callee is a JIT
        // graph even when `find_all_graphs` did not list it as a candidate
        // (`warmspot.py collect_called_graphs` is the annotator's reach
        // set, not only the codewriter's).
        if desc
            .cache
            .borrow()
            .keys()
            .any(crate::annotator::description::GraphCacheKey::is_access_direct)
        {
            continue;
        }
        for graph in desc.cache.borrow().values() {
            assert!(
                !graph.access_directly.get(),
                "access_directly on {}, which is outside the JIT graph set",
                desc.name
            );
        }
    }
}

fn run_two_phase_prepass_inner(
    call_registry: &CallRegistry,
    candidate_graphs: &HashSet<crate::parse::CallPath>,
    function_graphs: &crate::codewriter::call::GraphStore,
) {
    // Exists to localise prepass nondeterminism (gh#1139).
    let determinism_trace = crate::determinism_trace_enabled();
    if rtyper_verbose_enabled() {
        install_binding_backtrace_hook();
    }

    // Deterministic order (R3): candidate_graphs is a HashSet; iterating it
    // directly would make classdef numbering (and thus Match/Skip
    // classification) vary run-to-run. AccessDirect copies are first-class
    // subjects keyed by their derived `{name}__AccessDirect_None` CallPath
    // (`specialize.py default_specialize` / `cachedgraph`); they must not
    // reuse the regular graph's subject.
    let mut paths: Vec<&crate::parse::CallPath> = candidate_graphs
        .iter()
        .filter(|p| function_graphs.get(p).is_some())
        .collect();
    paths.sort_by_key(|p| p.canonical_key());

    // ── Phase A — annotate-all over the portal closure ───────────────
    let mut phase_a_reasons: Vec<String> = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        let Some(legacy) = function_graphs.get(path) else {
            continue;
        };
        let session_at_entry = SubjectSessionSnapshot::capture(call_registry);
        if determinism_trace {
            emit_determinism_trace("phaseA", index, &path.canonical_key());
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drive_subject(&legacy, call_registry, /* do_rtype = */ false)
        }));
        match attempt {
            Ok(Ok((
                graph,
                value_to_var,
                value_to_var_candidates,
                constant_concretetypes,
                constant_hlvalues,
            ))) => {
                let key = path.canonical_key();
                call_registry.two_phase().subjects.insert(
                    key,
                    crate::translator::rtyper::call_registry::TwoPhaseSubject {
                        graph: graph.clone(),
                        graph_key: crate::flowspace::model::GraphKey::of(&graph),
                        value_to_var,
                        value_to_var_candidates,
                        constant_concretetypes,
                        constant_hlvalues,
                    },
                );
                // Independent AccessDirect subjects never go through
                // `pycall` → `default_specialize`. Record the
                // `(AccessDirect, None)` PyGraph on the origin FunctionDesc
                // so dual_gate finds the annotator's cache key
                // (`specialize.py cachedgraph`).
                if crate::codewriter::call::is_access_directly_path(path) {
                    record_access_direct_pygraph_for_path(call_registry, path);
                }
            }
            ref other @ (Ok(Err(_)) | Err(_)) => {
                // The graph was a prepass subject: record the annotate-half
                // failure so publish classifies it as annotate-skipped, not
                // never-a-subject. Upstream `translator.graphs` contains every
                // graph the codewriter later sees (`driver.py` task_annotate
                // then task_rtype_lltype); a failed annotate is still a
                // subject. `downcast_ref::<String>` misses `AnnotatorError`
                // payloads (`panic_payload_text`).
                let reason = match other {
                    Ok(Err(e)) => format!("annotate Err: {e:?}"),
                    Err(p) => {
                        let msg = crate::annotator::model::panic_payload_text(&**p)
                            .unwrap_or_else(|| "<non-string panic>".to_string());
                        format!("annotate PANIC: {msg}")
                    }
                    Ok(Ok(_)) => unreachable!(),
                };
                {
                    let mut tp = call_registry.two_phase();
                    tp.annotate_skipped
                        .insert(path.canonical_key(), reason.clone());
                }
                if rtyper_verbose_enabled() {
                    eprintln!("[PREPASS phaseA fail] {:?}: {reason}", path);
                    phase_a_reasons.push(reason);
                    // MAJIT_RTYPER_FRONTIER: the adapter's whole call-wall set
                    // for this subject, not just the wall it stopped at. One
                    // line per wall, so the record splits without a banner.
                    for wall in crate::translator::rtyper::flowspace_adapter::take_frontier_walls()
                    {
                        eprintln!("[PREPASS phaseA wall] {:?}: {wall}", path);
                    }
                }
                // Annotate-half failed (or panicked): repair shared-callee state.
                // unpoison is itself contained — a panic here must not
                // abort the remaining Phase A graphs.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    unpoison_failed_subject_callees(
                        call_registry,
                        &session_at_entry,
                        function_graphs,
                    );
                    // `dont_look_inside` analog: a skipped subject is opaque
                    // to later callers (`install_source_graph` prefills the
                    // same signature stub). unpoison re-lifts the body, so
                    // a later subject that looks inside hits the next wall
                    // (`pin_gcrefs` not in CallRegistry) and is skipped too.
                    residualize_skipped_subject(call_registry, path, function_graphs);
                }));
            }
        }
    }

    check_access_directly_sanity(call_registry, candidate_graphs);

    if rtyper_verbose_enabled() {
        emit_disposition_histogram("phaseA", &phase_a_reasons);
        crate::codewriter::annotation_state::dump_classdef_less_ref_census();
    }

    // ── compute_at_fixpoint runs per-subject inside drive_subject's
    // annotate-half (cutover compute_at_fixpoint call); it is monotonic so
    // per-subject invocation reaches the same fixpoint (R2: the once-at-end
    // upstream shape — annotator.complete() — is otherwise a deferred
    // optimisation).

    // ── Inheritance-id barrier — number every classdef ONCE now, with the
    // complete Phase-A annotate-all classdef set and BEFORE Phase B rtypes
    // and materializes any vtable. Upstream runs this in
    // `RPythonTyper.specialize` via `perform_normalizations` before any vtable
    // materialisation (annrpython complete: `if block_subset is None:
    // perform_normalizations`); pyre's per-subject annotate defers
    // `annotator.complete()`, so do the inheritance-id half explicitly here.
    // With nothing yet baked, `assign_inheritance_ids`' global sort applies
    // unconditionally (`shifts_baked_id` is false), so a later subclass nests
    // into its parent bracket instead of dropping to the append-only fallback
    // once a subclassrange has been baked into the immutable jitcode const pool
    // during Phase B (#203 gap 9). The pass is idempotent without new
    // classdefs, so Phase B's per-subject numbering is then a no-op.
    // A failed session init must NOT fall through to Phase B: that would let
    // `run_phase_b_rtype_isolated` materialize vtables without the global
    // `assign_inheritance_ids` pass and reopen the stale minid/maxid contract
    // this barrier exists to hold. Surface and skip — Phase B is incremental,
    // so the next call retries with a live session.
    let (annotator, _rtyper) = match call_registry.ensure_session() {
        Ok(session) => session,
        Err(err) => {
            if rtyper_verbose_enabled() {
                eprintln!(
                    "[PREPASS inheritance-id barrier] ensure_session failed, \
                     skipping Phase B: {err:?}"
                );
            }
            return;
        }
    };
    crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&annotator);

    // Seed every annotated graph's return var (annrpython.py:258-261).
    // Per-subject annotate defers `annotator.complete()`, so its
    // return-var seeding never runs; an always-raising callee graph
    // (all paths reach the exceptblock, returnblock unreachable) would
    // otherwise enter Phase B with an unannotated return var and crash
    // `setconcretetype(getreturnvar())`. Run the seeding here, alongside
    // the other explicit stand-ins for the deferred complete().
    annotator.seed_all_annotated_return_vars();

    // ── Phase B — rtype-all with per-graph isolation ─────────────────
    // Writes its rtype_skipped set into the cache incrementally so partial
    // progress survives even if this call is cut short.
    run_phase_b_rtype_isolated(call_registry, function_graphs, determinism_trace, &paths);
}

/// Phase B: rtype every annotated block in one whole-program pass, tolerant of
/// per-graph rtype failure (the migration scaffold — upstream `specialize()` is
/// single-pass-fatal, rtyper.py). A block's `specialize_block` failure
/// records its owning graph into the cache's `rtype_skipped` set; that graph's
/// remaining blocks are excluded from later rounds (the publish then Skips it to
/// the legacy walker) and its half-fixed callees are repaired via
/// `unpoison_failed_subject_callees`. The skip set is written into the cache
/// incrementally so partial progress survives a cut-short call.
fn run_phase_b_rtype_isolated(
    call_registry: &CallRegistry,
    lift_sources: &crate::codewriter::call::GraphStore,
    determinism_trace: bool,
    paths: &[&crate::parse::CallPath],
) {
    use crate::flowspace::model::{BlockRef, GraphKey, GraphRef};
    let Ok((annotator, rtyper)) = call_registry.ensure_session() else {
        return;
    };
    let mut phase_b_reasons: Vec<String> = Vec::new();
    // Exists to localise prepass nondeterminism (gh#1139).
    let phase_b_trace_paths = determinism_trace.then(|| {
        let cache = call_registry.two_phase();
        paths
            .iter()
            .enumerate()
            .filter_map(|(index, path)| {
                let canonical_key = path.canonical_key();
                cache
                    .subjects
                    .get(&canonical_key)
                    .map(|subject| (subject.graph_key.clone(), (index, canonical_key)))
            })
            .collect::<Vec<_>>()
    });
    // Exists to localise prepass nondeterminism (gh#1139).
    let mut phase_b_traced = determinism_trace.then(Vec::new);

    // Upstream `RPythonTyper.specialize()` step 1 (rtyper.py):
    // `if not dont_simplify_again: self.annotator.simplify()`. pyre's
    // per-subject annotate-half (`drive_subject`) skips the annotator-wide
    // simplify, so a graph's constant-folded dead arms stay structurally
    // linked into Phase B. The annotator never followed those links — a
    // constant exitswitch makes the mismatched-exitcase arm carry
    // `s_ImpossibleValue` through `follow_link`, which returns before
    // `links_followed[link] = True` (annrpython.rs:1762-1768), leaving the
    // arm's target unannotated — but the surviving structural link reaches
    // `specialize_block` → `insert_link_conversions` → `bindingrepr`, which
    // then KeyErrors on the dead target's unbound inputargs.
    //
    // Run `transform_dead_code` (transform.py) once over the whole
    // fully-annotated block set here: it is the simplify pass that prunes
    // links absent from `links_followed`, folding a const switch back to a
    // plain goto. It is a no-op on graphs whose every exit was followed, so
    // healthy graphs are untouched. Per-block panic isolation:
    // `cutoff_alwaysraising_block`'s consistency asserts can fire on an
    // unrelated malformed always-raising block; isolating each block keeps
    // one such graph from aborting dead-code removal for the rest (that
    // graph then fails its own rtype below and Skips, unchanged).
    {
        use crate::flowspace::model::BlockKey;
        use crate::translator::transform::{
            cutoff_block_trace, fully_annotated_blocks, transform_dead_code, transform_dead_op_vars,
        };
        let annotated_blocks = fully_annotated_blocks(&annotator);

        if determinism_trace {
            use md5::{Digest, Md5};

            let graph_names: Vec<String> = {
                let annotated = annotator.annotated.borrow();
                annotated_blocks
                    .iter()
                    .map(|block| {
                        annotated
                            .get(&BlockKey::of(block))
                            .and_then(|graph| graph.as_ref())
                            .map(|graph| graph.borrow().name.clone())
                            .unwrap_or_else(|| "<unowned>".to_string())
                    })
                    .collect()
            };
            let mut digest = Md5::new();
            for name in &graph_names {
                digest.update((name.len() as u64).to_le_bytes());
                digest.update(name.as_bytes());
            }
            eprintln!(
                "[DTRACE-CUTSET] count={} md5={:x} first={:?}",
                graph_names.len(),
                digest.finalize(),
                &graph_names[..graph_names.len().min(20)],
            );
        }

        // Pass 1 — `transform_dead_code` (transform.py): prune the
        // unfollowed const-switch dead arms. Per-block panic isolation
        // (`cutoff_alwaysraising_block`'s consistency asserts can fire on an
        // unrelated malformed always-raising block).
        let mut cutoff_panics = 0usize;
        for block in &annotated_blocks {
            let trace = determinism_trace.then(|| cutoff_block_trace(&annotator, block));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                transform_dead_code(&annotator, std::slice::from_ref(block));
            }));
            if let Err(payload) = result {
                cutoff_panics += 1;
                if let Some(trace) = trace {
                    let payload = crate::annotator::model::panic_payload_text(&*payload)
                        .unwrap_or_else(|| "<non-string panic payload>".to_string());
                    eprintln!("[DTRACE-CUTPANIC] {trace} payload={payload:?}");
                }
            }
        }
        if determinism_trace {
            eprintln!("[DTRACE-CUTPANIC] total={cutoff_panics}");
        }

        // Pass 2 — `transform_dead_op_vars` (simplify.py, transform.py):
        // drop dead ops and, crucially, dead inputargs together with their
        // matching link args. This is the remaining half of upstream
        // `specialize()` step-1 `simplify()` that the per-subject flow also
        // skipped: removing a never-read merge inputarg clears the unbound
        // operand the rtyper would otherwise `binding`-KeyError on. Grouped
        // per owning graph and panic-isolated — the pass is whole-subgraph
        // dataflow (its link-arity assert can fire on a malformed graph), so a
        // single bad graph must not abort the drain for the rest; that graph
        // then fails its own rtype below and Skips, unchanged.
        let mut order: Vec<GraphKey> = Vec::new();
        let mut by_graph: HashMap<GraphKey, Vec<BlockRef>> = HashMap::new();
        {
            let annotated = annotator.annotated.borrow();
            for block in &annotated_blocks {
                if let Some(Some(g)) = annotated.get(&BlockKey::of(block)) {
                    let gk = GraphKey::of(g);
                    by_graph
                        .entry(gk.clone())
                        .or_insert_with(|| {
                            order.push(gk.clone());
                            Vec::new()
                        })
                        .push(block.clone());
                }
            }
        }
        for gk in &order {
            let group = &by_graph[gk];
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                transform_dead_op_vars(&annotator, group);
            }));
        }
    }

    loop {
        // Skip-tolerant repr setup: a single unported repr (e.g. DictRepr)
        // must not abort rtyping of every other annotated graph. The graph
        // that needs the skipped repr fails its own specialize_block below and
        // is Skip-classified (parity fallback to the legacy walker).
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rtyper.call_all_setups_skip_tolerant()
        }));
        // Snapshot the skip set (cache borrow drops before specialize_block).
        let skipped: HashSet<GraphKey> = call_registry.two_phase().rtype_skipped.clone();
        // annotated\already_seen, excluding blocks of already-skipped graphs.
        let pending: Vec<(BlockRef, Option<GraphRef>)> = {
            let annotated = annotator.annotated.borrow();
            let all_blocks = annotator.all_blocks.borrow();
            let already_seen = rtyper.already_seen.borrow();
            annotated
                .iter()
                .filter(|(bkey, _)| !already_seen.contains_key(*bkey))
                .filter_map(|(bkey, gopt)| {
                    if let Some(g) = gopt
                        && skipped.contains(&GraphKey::of(g))
                    {
                        return None;
                    }
                    all_blocks.get(bkey).map(|b| (b.clone(), gopt.clone()))
                })
                .collect()
        };
        if pending.is_empty() {
            break;
        }
        for (block, gopt) in pending {
            if let Some(g) = &gopt
                && skipped.contains(&GraphKey::of(g))
            {
                continue;
            }
            let session_at_entry = SubjectSessionSnapshot::capture_fixed_only(call_registry);
            if let (Some(g), Some(trace_paths), Some(traced)) =
                (&gopt, &phase_b_trace_paths, &mut phase_b_traced)
            {
                let graph_key = GraphKey::of(g);
                if !traced.contains(&graph_key) {
                    traced.push(graph_key.clone());
                    if let Some((_, (index, canonical_key))) = trace_paths
                        .iter()
                        .find(|(candidate, _)| candidate == &graph_key)
                    {
                        emit_determinism_trace("phaseB", *index, canonical_key);
                    }
                }
            }
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rtyper.specialize_block(&block)
            }));
            match res {
                Ok(Ok(())) => rtyper.mark_already_seen(&block),
                ref other @ (Ok(Err(_)) | Err(_)) => {
                    if rtyper_verbose_enabled() {
                        let reason = match other {
                            Ok(Err(e)) => format!("rtype Err: {e:?}"),
                            Err(p) => format!("rtype PANIC: {}", unwind_panic_text(&**p)),
                            Ok(Ok(())) => unreachable!(),
                        };
                        let gname = gopt
                            .as_ref()
                            .map(|g| g.borrow().name.clone())
                            .unwrap_or_else(|| "<none>".to_string());
                        eprintln!(
                            "[PREPASS phaseB fail] {:?} {gname}: {reason}",
                            gopt.as_ref().map(GraphKey::of)
                        );
                        // A panic from `bindingrepr` carries only the
                        // unbound Variable identity, unlike ordinary
                        // `TyperError::where_info`, which includes the
                        // current operation.  Emit this block's shallow
                        // contents in the explicitly verbose census so the
                        // producer/consumer edge can be located without a
                        // second instrumented build.
                        let failed_block = block.borrow();
                        eprintln!(
                            "[PREPASS phaseB block] {gname}: inputargs={:?} exitswitch={:?}",
                            failed_block.inputargs, failed_block.exitswitch
                        );
                        for op in &failed_block.operations {
                            eprintln!("[PREPASS phaseB op] {gname}: {op:?}");
                        }
                        for (ei, link) in failed_block.exits.iter().enumerate() {
                            let followed = annotator
                                .links_followed
                                .borrow()
                                .contains_key(&crate::flowspace::model::LinkKey::of(link));
                            eprintln!(
                                "[PREPASS phaseB exit] {gname}: i={ei} followed={followed} args={:?} case={:?}",
                                link.borrow().args,
                                link.borrow().exitcase
                            );
                        }
                        drop(failed_block);
                        if let Some(g) = &gopt {
                            let gb = g.borrow();
                            let ret = gb.getreturnvar();
                            eprintln!("[PREPASS phaseB returnvar] {gname}: {ret:?}");
                            let annotated = annotator.annotated.borrow();
                            let blocked = annotator.blocked_blocks.borrow();
                            let all_blocks = annotator.all_blocks.borrow();
                            let blocks = gb.iterblocks();
                            let mut n_ann_graph = 0usize;
                            let mut n_ann_none = 0usize;
                            let mut n_missing = 0usize;
                            let mut n_blocked = 0usize;
                            for b in &blocks {
                                let key = crate::flowspace::model::BlockKey::of(b);
                                match annotated.get(&key) {
                                    Some(Some(_)) => n_ann_graph += 1,
                                    Some(None) => n_ann_none += 1,
                                    None => n_missing += 1,
                                }
                                if blocked.contains_key(&key) {
                                    n_blocked += 1;
                                }
                                if !all_blocks.contains_key(&key) {
                                    eprintln!(
                                        "[PREPASS phaseB block-census] {gname}: reachable block missing from all_blocks"
                                    );
                                }
                            }
                            eprintln!(
                                "[PREPASS phaseB block-census] {gname}: reachable={} annotated-graph={} annotated-false={} missing={} blocked={} all_blocks={} annotated_map={}",
                                blocks.len(),
                                n_ann_graph,
                                n_ann_none,
                                n_missing,
                                n_blocked,
                                all_blocks.len(),
                                annotated.len()
                            );
                        }
                        phase_b_reasons.push(reason);
                    }
                    match &gopt {
                        Some(g) => {
                            // Skip this graph; the pending recompute excludes
                            // its blocks, guaranteeing progress (one graph
                            // removed per failure).
                            call_registry
                                .two_phase()
                                .rtype_skipped
                                .insert(GraphKey::of(g));
                        }
                        None => {
                            // No graph to attribute (None-sentinel block);
                            // mark it seen so the loop still progresses.
                            rtyper.mark_already_seen(&block);
                        }
                    }
                    // Repair half-fixed callees; contained so an unpoison panic
                    // can't escape Phase B.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        unpoison_failed_subject_callees(
                            call_registry,
                            &session_at_entry,
                            lift_sources,
                        );
                    }));
                    break;
                }
            }
        }
    }
    if rtyper_verbose_enabled() {
        emit_disposition_histogram("phaseB", &phase_b_reasons);
    }
    // Upstream's codewriter consumes graphs after `RPythonTyper.specialize`.
    // Report the same surface: Phase B has rewritten each retained GraphRef in
    // place, and `rtype_skipped` identifies graphs that did not specialize.
    if crate::codewriter::jtransform_shadow::is_enabled() {
        let tp = call_registry.two_phase();
        let mut subjects: Vec<_> = tp.subjects.values().collect();
        subjects.sort_by_key(|subject| subject.graph.borrow().name.clone());
        for subject in subjects {
            if !tp.rtype_skipped.contains(&subject.graph_key) {
                crate::codewriter::jtransform_shadow::report_if_enabled(&subject.graph.borrow());
            }
        }
    }
    // MixLevelHelperAnnotator drain (rtyper.py:238-241) is a no-op while R4 is
    // unported — specialize_block never queues helper graphs today. When R4
    // lands, take()+finish() rtyper.annmixlevel here.
}

/// Reconnect a legacy [`crate::model::OpKind::Hint`] result to the typed
/// representative of its operand when the real path removed the result
/// identity.
///
/// The flowspace adapter emits `same_as(value)` for a Hint
/// (`flowspace_adapter.rs`), and RPython's rtyper treats `same_as` as an
/// internal rename.  Its JIT codewriter then returns `None` from
/// `rewrite_op_hint` specifically to force `op.result` equal to `op.args[0]`
/// (`rpython/jit/codewriter/jtransform.py`).  The typed graph can
/// therefore contain only the operand representative while the legacy graph
/// still names the result.  Restoring that alias in the existing adapter map
/// preserves the one upstream value; it does not invent a type or suppress a
/// genuine kind conflict.  An already-typed result mapping wins; an adapter
/// placeholder whose `concretetype` is still empty is the erased identity and
/// is replaced.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn reconcile_elided_hint_results(legacy: &LegacyGraph, value_to_var: &mut LegacyToTyped) -> bool {
    let mut changed = false;
    for block in &legacy.blocks {
        for op in &block.operations {
            let crate::model::OpKind::Hint { value, .. } = &op.kind else {
                continue;
            };
            let Some(result) = &op.result else {
                continue;
            };
            if value_to_var
                .get(result)
                .is_some_and(|typed| typed.concretetype().is_some())
            {
                continue;
            }
            let Some(operand_twin) = value_to_var.get(value).cloned() else {
                continue;
            };
            if operand_twin.concretetype().is_none() {
                continue;
            }
            value_to_var.insert(result.clone(), operand_twin);
            changed = true;
        }
    }
    changed
}

/// Reconnect a block inputarg to the typed representative shared by all of
/// its incoming phi arguments when flowspace simplification removed the
/// inputarg identity.
///
/// RPython `remove_identical_vars_SSA` unions an input with its incoming
/// value when every `phi_arg` has the same union-find representative
/// (`rpython/translator/simplify.py`).  A one-predecessor phi is the
/// common case.  `join_blocks` performs the same rename while merging a
/// linear edge (`simplify.py:271-315`).  The adapter map predates those
/// mutations, so its original inputarg twin may remain untyped even though
/// every surviving operation was renamed to the predecessor's typed value.
///
/// Reproduce only that upstream identity relation in the projection map:
/// every incoming argument must be a Variable, already have a positively
/// typed representative, and resolve to the exact same representative.
/// Conflicting phi inputs and already-typed inputargs are left untouched.
/// Iterate to a fixpoint because upstream's union-find collapses chains of
/// single-predecessor blocks transitively.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn reconcile_elided_phi_inputargs(
    legacy: &LegacyGraph,
    value_to_var: &mut LegacyToTyped,
    constant_values: &mut HashMap<Variable, Hlvalue>,
    constant_types: &mut HashMap<Variable, LowLevelType>,
) -> bool {
    let mut changed = false;
    loop {
        let mut aliases = Vec::new();
        for target in &legacy.blocks {
            if target.id == legacy.startblock
                || target.id == legacy.returnblock
                || target.id == legacy.exceptblock
            {
                continue;
            }
            for (column, input) in target.inputargs.iter().enumerate() {
                if constant_values.contains_key(input)
                    || value_to_var
                        .get(input)
                        .is_some_and(|typed| typed.concretetype().is_some())
                {
                    continue;
                }
                let mut saw_incoming = false;
                let mut shared: Option<Hlvalue> = None;
                let mut shared_constant_type = None;
                let mut valid = true;
                for source in &legacy.blocks {
                    for link in source.exits.iter().filter(|link| link.target == target.id) {
                        saw_incoming = true;
                        let source = match link.args.get(column) {
                            Some(crate::model::LinkArg::Value(source_var)) => {
                                if let Some(Hlvalue::Constant(c)) = constant_values.get(source_var)
                                {
                                    constant_types
                                        .get(source_var)
                                        .cloned()
                                        .map(|ty| (Hlvalue::Constant(c.clone()), Some(ty)))
                                } else {
                                    value_to_var
                                        .get(source_var)
                                        .filter(|typed| typed.concretetype().is_some())
                                        .map(|typed| (Hlvalue::Variable(typed.clone()), None))
                                }
                            }
                            Some(crate::model::LinkArg::Const(c)) => c
                                .concretetype
                                .clone()
                                .map(|ty| (Hlvalue::Constant(c.clone()), Some(ty))),
                            _ => None,
                        };
                        let Some((source, constant_type)) = source else {
                            valid = false;
                            break;
                        };
                        if shared
                            .as_ref()
                            .is_some_and(|representative| representative != &source)
                            || (shared.is_some() && shared_constant_type != constant_type)
                        {
                            valid = false;
                            break;
                        }
                        if shared.is_none() {
                            shared = Some(source);
                            shared_constant_type = constant_type;
                        }
                    }
                    if !valid {
                        break;
                    }
                }
                if valid
                    && saw_incoming
                    && let Some(shared) = shared
                {
                    aliases.push((input.clone(), shared, shared_constant_type));
                }
            }
        }
        if aliases.is_empty() {
            break;
        }
        changed = true;
        for (input, shared, constant_type) in aliases {
            match shared {
                Hlvalue::Variable(shared) => {
                    value_to_var.insert(input, shared);
                }
                Hlvalue::Constant(c) => {
                    // simplify.py::remove_identical_vars_SSA uses the actual
                    // Constant representative, not an equal register bank.
                    // Retain it without changing the legacy baseline's cells.
                    constant_types.insert(input.clone(), constant_type.expect("typed constant"));
                    constant_values.insert(input, Hlvalue::Constant(c));
                }
            }
        }
    }
    changed
}

/// Backfill the low-level type of a call-result Variable the real path
/// left untyped, from the front-end `OpKind::Call { result_ty }` the
/// call site declared.
///
/// Charon does not monomorphize generics, so a call returning an erased
/// associated type — `<E as SharedOpcodeHandler>::Value` for the
/// generic opcode-executor family, whose two impls both bind it to a
/// gc-ref value (`PyObjectRef` / `FrontendOp`) — reaches the annotator
/// as a projection the rtyper never narrows to a concrete repr.  Its
/// result Variable's `concretetype` cell stays `None`, so
/// [`project_value_to_var`] omits it and [`collect_divergences`] reads
/// `real=Unknown` against the legacy walker's `GcRef` — the legacy
/// walker types the same slot from this identical `result_ty` at
/// `legacy_resolve::infer_concrete_from_op` (`OpKind::Call { result_ty }
/// => valuetype_to_concrete(result_ty)`), plus its final GcRef backfill.
///
/// Mirror that single rule into the real path so the two converge: for a
/// call result whose declared `result_ty` projects to `GcRef`, stamp the
/// canonical `GCREF` pointer onto the twin.
///
/// One equally erased scalar shape is admitted, but only at the producer
/// which proves it: `front::option_closure_select` replaces
/// `Option::unwrap_or_else` with the direct discriminant diamond RPython's
/// closure-free source would have built and synthesizes a closure
/// `call_once` in the `None` arm, and `front::iter_adapter` synthesizes
/// `call_mut` for `map(...).collect()`.  Charon retains the concrete
/// `FnOnce::Output` / `FnMut::Output` as that op's `result_ty` but does not
/// monomorphize it into the registered closure method, so the real rtyper
/// can default the live result to `Void`.  For that synthesized closure
/// call alone, carry a declared `Signed` or `Float` result onto the twin.
/// An ordinary scalar call is deliberately not
/// covered: its `Signed`/`Float` versus `Void` pairing remains a genuine kind
/// divergence and falls back to legacy.
///
/// Overrides both the untyped (`None`) twin and a twin the rtyper
/// collapsed to `Void`: a call the front-end declared pointer-returning
/// (`result_ty` projects to `GcRef`) is never legitimately void — a
/// genuinely void call carries `result_ty == Void` and is filtered by
/// the guard — so a `Void` twin is the same erased-`<E>::Value` artifact
/// as a `None` twin, reached when the rtyper defaulted the un-narrowed
/// projection to unit.  It would otherwise be colored by
/// `emit_call_result_arg` (the op's declared non-void `result_kind`
/// forces the `>X` result argcode, `assembler.rs:2226`) and panic in
/// `lookup_coloring`.  A twin the rtyper positively typed to another non-void
/// kind is left untouched — that is a real kind conflict → Skip.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn backfill_untyped_call_results(legacy: &LegacyGraph, value_to_var: &LegacyToTyped) {
    use crate::translator::rtyper::lltypesystem::lltype::GCREF;
    for block in &legacy.blocks {
        for op in &block.operations {
            let crate::model::OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            let declared_kind = valuetype_to_concrete(result_ty);
            let closure_scalar_call = matches!(
                target,
                crate::model::CallTarget::Method {
                    name,
                    receiver_root: Some(root),
                    ..
                } if (name == "call_once" || name == "call_mut")
                    && root.rsplit("::").next().is_some_and(
                        majit_charon_reader::ullbc::is_closure_leaf
                    )
            );
            let declared_lltype = match declared_kind {
                ConcreteType::GcRef => Some(GCREF.clone()),
                ConcreteType::Signed | ConcreteType::Float if closure_scalar_call => {
                    crate::model::concrete_to_canonical_lltype(declared_kind)
                }
                _ => None,
            };
            let Some(declared_lltype) = declared_lltype else {
                continue;
            };
            let Some(legacy_result) = &op.result else {
                continue;
            };
            let Some(typed) = value_to_var.get(legacy_result) else {
                continue;
            };
            let twin_kind = typed
                .concretetype()
                .map(|lltype| lowleveltype_to_concrete(&lltype).unwrap_or(ConcreteType::Unknown));
            if matches!(twin_kind, None | Some(ConcreteType::Void)) {
                typed.set_concretetype(Some(declared_lltype));
            }
        }
    }
}

/// Apply projection repairs in dependency order.
///
/// `same_as` results can feed phis and phi inputs can feed `same_as`.
/// Close both identity relations together, as the renaming performed by
/// simplify.py::remove_identical_vars_SSA and rtyping does in the canonical
/// graph. Each successful step publishes a positively typed representative;
/// already-typed mappings remain authoritative, so this terminates without
/// a retry limit or type guesses. Both direct and cached publication use it.
fn reconcile_projection_aliases(
    legacy: &LegacyGraph,
    value_to_var: &mut LegacyToTyped,
    constants: &mut HashMap<Variable, Hlvalue>,
    constant_types: &mut HashMap<Variable, LowLevelType>,
) {
    backfill_untyped_call_results(legacy, value_to_var);
    loop {
        let hint_changed = reconcile_elided_hint_results(legacy, value_to_var);
        let phi_changed =
            reconcile_elided_phi_inputargs(legacy, value_to_var, constants, constant_types);
        if !hint_changed && !phi_changed {
            break;
        }
    }
}

/// Whether `FunctionDesc.cache` holds a `(AccessDirect, key)` PyGraph for
/// the callable `diag_key` names. The AccessDirect copy shares that
/// FunctionDesc (`cachedgraph`); a miss means the annotator never
/// specialized a call as AccessDirect even if the codewriter cloned a
/// `{name}__AccessDirect_None` graph.
fn function_desc_has_access_direct_pygraph(call_registry: &CallRegistry, diag_key: &str) -> bool {
    let path = crate::parse::CallPath::from_segments(diag_key.split("::"));
    let origin = crate::codewriter::call::access_directly_origin_path(&path).unwrap_or(path);
    let Some(entry) = call_registry.lookup(&FunctionPathKey::from_segments(
        origin.segments.iter().cloned(),
    )) else {
        return false;
    };
    let desc = entry.function_desc.borrow();
    desc.cache
        .borrow()
        .keys()
        .any(crate::annotator::description::GraphCacheKey::is_access_direct)
}

/// `FunctionDesc.cachedgraph` the `(AccessDirect, None)` copy for an
/// independent AccessDirect two-phase subject.
fn record_access_direct_pygraph_for_path(
    call_registry: &CallRegistry,
    path: &crate::parse::CallPath,
) {
    let origin =
        crate::codewriter::call::access_directly_origin_path(path).unwrap_or_else(|| path.clone());
    let Some(entry) = call_registry.lookup(&FunctionPathKey::from_segments(
        origin.segments.iter().cloned(),
    )) else {
        return;
    };
    let _ = entry.function_desc.borrow().record_access_directly_graph();
}

/// Two-phase publish: derive the [`DualGateOutcome`] for `legacy` from the
/// prepass cache instead of re-running the real path. Mirrors
/// [`dual_gate_check_with_registry`]'s legacy-baseline comparison (the
/// migration safety oracle) but consumes the cached real types.
pub(crate) fn dual_gate_outcome_from_cache(
    legacy: &LegacyGraph,
    call_registry: &CallRegistry,
    diag_key: &str,
) -> Result<DualGateOutcome, String> {
    // Clone the cached real types out so the cache borrow drops before the
    // legacy baseline runs (the baseline never touches the cache).
    // Three causes reach the Skip below and they are not the same finding:
    // the prepass may never have had this graph as a subject at all, it
    // may have had it and annotate-skipped it, or it may have annotated
    // it and rtype-skipped it.  The first says the closure walk did not
    // reach the graph; the others say it did and the real rtyper refused
    // it.  Phase-A failures used to share the never-a-subject message, so
    // a census could not separate "never attempted" from "attempted and
    // declined" — the same conflation the decline census exists to remove.
    // Neither spelling matches any `unported_category` arm, so
    // classification is unchanged.  Lookup is by `diag_key` alone
    // (`CallPath::canonical_key()` at both prepass insert and publish);
    // graph names are not unique (`default`, `new`, `call`, and
    // `<Impl>::copy` shared across Int/Unicode/Bytes/Object dict
    // strategies) so a name scan can publish another graph's
    // concretetypes. `FunctionGraph::source_identity` is the unique
    // funcobj key.
    let cached = {
        let is_ad = crate::parse::CallPath::from_segments(diag_key.split("::"));
        let is_ad = crate::codewriter::call::is_access_directly_path(&is_ad);
        let tp = call_registry.two_phase();
        match tp.subjects.get(diag_key) {
            Some(subj) if !tp.rtype_skipped.contains(&subj.graph_key) => {
                // An AccessDirect candidate must publish the annotator's
                // `(AccessDirect, key)` PyGraph of the same FunctionDesc.
                // Falling back to the regular graph's subject would type
                // the copy as if it were the unflagged specialization.
                if is_ad && !function_desc_has_access_direct_pygraph(call_registry, diag_key) {
                    Err(
                        "two-phase: AccessDirect PyGraph missing from FunctionDesc cache"
                            .to_string(),
                    )
                } else {
                    Ok((
                        subj.graph.clone(),
                        subj.value_to_var.clone(),
                        subj.value_to_var_candidates.clone(),
                        subj.constant_concretetypes.clone(),
                        subj.constant_hlvalues.clone(),
                    ))
                }
            }
            Some(_) => Err("two-phase: subject rtype-skipped in prepass".to_string()),
            None => {
                if let Some(reason) = tp.annotate_skipped.get(diag_key) {
                    Err(format!(
                        "two-phase: subject annotate-skipped in prepass: {reason}"
                    ))
                } else if is_ad {
                    Err("two-phase: AccessDirect graph was never a prepass subject".to_string())
                } else {
                    Err("two-phase: graph was never a prepass subject".to_string())
                }
            }
        }
    };
    let (graph, mut value_to_var, value_to_var_candidates, mut constants, mut constant_values) =
        match cached {
            Ok(cached) => cached,
            Err(reason) => return Ok(DualGateOutcome::Skip(reason)),
        };
    select_rtyped_representatives(&mut value_to_var, &value_to_var_candidates)
        .map_err(|e| e.to_string())?;
    reconcile_projection_aliases(
        legacy,
        &mut value_to_var,
        &mut constant_values,
        &mut constants,
    );

    // Validate the cached lltypes project to concrete kinds (the Step-4 check
    // deferred from `drive_subject`, which did not rtype in Phase A).
    for var in value_to_var.values() {
        let cell = var.concretetype.borrow();
        if let Some(lltype) = cell.as_ref()
            && lowleveltype_to_concrete(lltype).is_err()
        {
            return Ok(DualGateOutcome::Skip(
                "two-phase: cached lltype does not project".to_string(),
            ));
        }
    }
    for lltype in constants.values() {
        if lowleveltype_to_concrete(lltype).is_err() {
            return Ok(DualGateOutcome::Skip(
                "two-phase: cached const lltype does not project".to_string(),
            ));
        }
    }

    // Legacy-baseline comparison oracle (the `dual_gate_check_with_registry`
    // pattern): the real path is trusted only when it agrees with the proven
    // legacy walker at `ConcreteType` granularity.
    let _annotation_guard = LegacyAnnotationGuard::snapshot(legacy);
    let baseline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::legacy_annotator::annotate(legacy);
        super::legacy_resolve::resolve_types(legacy);
    }));
    if let Err(payload) = baseline {
        let msg = crate::annotator::model::panic_payload_text(&*payload)
            .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
        return Err(format!(
            "two-phase baseline panicked (legacy walker crashed before comparison): {msg}"
        ));
    }
    let followed_legacy = call_registry
        .session_if_started()
        .map(|(annotator, _)| annotator_followed_legacy_vars(&annotator, &graph, &value_to_var));
    let excluded_legacy = followed_legacy
        .as_ref()
        .map(|followed| annotator_unfollowed_legacy_vars(&value_to_var, followed));
    if let Some(divergence) =
        compare_real_against_legacy(&value_to_var, &constants, legacy, excluded_legacy.as_ref())
    {
        return Ok(DualGateOutcome::Skip(format!(
            "two-phase divergence: {divergence}"
        )));
    }
    if followed_legacy
        .as_ref()
        .is_some_and(|followed| followed.is_empty())
    {
        return Ok(DualGateOutcome::Skip(
            "two-phase compared set empty".to_string(),
        ));
    }
    keep_legacy_kind_when_unfollowed_void(
        &mut value_to_var,
        &mut constants,
        excluded_legacy.as_ref(),
    );
    Ok(DualGateOutcome::Match {
        real_value_to_var: value_to_var,
        real_constants: constants,
    })
}

/// Drop an unfollowed Variable whose real concretetype is `Void` when the
/// legacy cell already has a kind. See the call in
/// [`dual_gate_check_with_registry`].
///
/// Phase B `setconcretetype`s `Void` on an unfollowed cell after
/// `s_ImpossibleValue` (`force_return_var_annotation`). The AccessDirect
/// copy is a fresh `copygraph` (`cachedgraph` copies before rtyping);
/// `emit_one_word_guard_release` already stamped the residual ABI word
/// Signed, and the skip-arm `IntegerRepr` analog keeps it. Publishing the
/// Void twin would make `getcalldescr` drop that actual (`call.py`
/// NON_VOID_ARGS) while `FUNC.ARGS` still has Int.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn keep_legacy_kind_when_unfollowed_void(
    value_to_var: &mut LegacyToTyped,
    constants: &mut HashMap<Variable, LowLevelType>,
    excluded_legacy: Option<&HashSet<Variable>>,
) {
    let Some(excluded) = excluded_legacy else {
        return;
    };
    let drop_keys: Vec<Variable> = value_to_var
        .iter()
        .filter_map(|(legacy, typed)| {
            if !excluded.contains(legacy) {
                return None;
            }
            let ll = typed.concretetype()?;
            let kind = lowleveltype_to_concrete(&ll).ok()?;
            if kind != crate::model::ConcreteType::Void {
                return None;
            }
            match LegacyGraph::concretetype_of(legacy) {
                crate::model::ConcreteType::Void | crate::model::ConcreteType::Unknown => None,
                crate::model::ConcreteType::Signed
                | crate::model::ConcreteType::GcRef
                | crate::model::ConcreteType::Float => Some(legacy.clone()),
            }
        })
        .collect();
    for key in drop_keys {
        value_to_var.remove(&key);
        constants.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::description::{CallFamily, CallTableRow, DescKey};
    use crate::flowspace::argument::CallShape;
    use crate::flowspace::model::BlockKey;
    use crate::model::{Block, BlockId, LinkArg, ValueType};
    use crate::translator::rtyper::legacy_annotator::setbinding;

    fn test_pygraph(name: &str) -> Rc<PyGraph> {
        use crate::flowspace::model::{Block as FlowBlock, GraphFunc, Hlvalue, Link, Variable};
        let arg = Hlvalue::Variable(Variable::named("x"));
        let startblock = FlowBlock::shared(vec![arg.clone()]);
        let graph = crate::flowspace::model::FunctionGraph::new(name, startblock.clone());
        let link = Rc::new(RefCell::new(Link::new(
            vec![arg],
            Some(graph.returnblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link]);
        Rc::new(PyGraph {
            graph: Rc::new(RefCell::new(graph)),
            func: GraphFunc::new(name, Constant::new(ConstValue::Dict(Default::default()))),
            signature: RefCell::new(Signature::new(vec!["x".to_string()], None, None)),
            defaults: RefCell::new(None),
            access_directly: Cell::new(false),
        })
    }

    #[test]
    fn residualize_replaces_only_the_none_calltable_row() {
        // `CallFamily.find_row` / `calltable_lookup_row` (`description.py`):
        // the AccessDirect cachedgraph stays a distinct row. Residualizing
        // the skipped None subject must not overwrite that row with the stub.
        let rowkey = DescKey(7);
        let none_graph = test_pygraph("call_module_helper");
        let access_direct = test_pygraph("call_module_helper__AccessDirect_None");
        let stub = test_pygraph("call_module_helper_stub");
        let shape = CallShape {
            shape_cnt: 4,
            shape_keys: Vec::new(),
            shape_star: false,
        };
        let mut none_row = CallTableRow::new();
        none_row.insert(rowkey, none_graph.clone());
        let mut ad_row = CallTableRow::new();
        ad_row.insert(rowkey, access_direct.clone());
        let family = Rc::new(RefCell::new(CallFamily::new(rowkey)));
        family
            .borrow_mut()
            .calltable_add_row(shape.clone(), none_row);
        family
            .borrow_mut()
            .calltable_add_row(shape.clone(), ad_row.clone());

        replace_none_graph_in_calltables(&family, rowkey, Some(&none_graph), &stub);

        let mut stub_row = CallTableRow::new();
        stub_row.insert(rowkey, stub.clone());
        let family = family.borrow();
        assert_eq!(
            family.calltable_lookup_row(&shape, &stub_row),
            Some(0),
            "None row becomes the residual stub"
        );
        assert_eq!(
            family.calltable_lookup_row(&shape, &ad_row),
            Some(1),
            "AccessDirect row keeps its cachedgraph"
        );
        let mut old_none_row = CallTableRow::new();
        old_none_row.insert(rowkey, none_graph);
        assert!(
            family.calltable_lookup_row(&shape, &old_none_row).is_none(),
            "the skipped None graph is no longer a family row"
        );
    }

    #[test]
    fn std_alloc_raw_is_declined_and_unregistered() {
        for segments in [
            vec!["std".into(), "alloc".into(), "alloc_zeroed".into()],
            vec!["std".into(), "alloc".into(), "alloc".into()],
            vec!["alloc".into(), "alloc".into(), "alloc_zeroed".into()],
            vec!["alloc".into(), "alloc".into(), "alloc".into()],
            vec!["alloc_zeroed".into()],
        ] {
            assert_eq!(
                registration_decline(&segments),
                Some(SKIP_STD_ALLOC_RAW),
                "{segments:?}"
            );
        }
        assert_eq!(
            registration_decline(&["alloc".into()]),
            None,
            "the alloc leaf is shared with other functions"
        );
        assert_eq!(
            registration_decline(&["std".into(), "alloc".into(), "dealloc".into()]),
            None
        );

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let mut graphs = crate::codewriter::call::GraphStore::default();
        for segments in [
            vec!["std", "alloc", "alloc_zeroed"],
            vec!["alloc", "alloc", "alloc_zeroed"],
            vec!["alloc_zeroed"],
        ] {
            let path = crate::parse::CallPath {
                segments: segments.iter().map(|s| (*s).to_string()).collect(),
            };
            let mut graph = LegacyGraph::new("alloc_zeroed");
            graph.set_return(graph.startblock.clone(), None);
            graphs.insert(path, graph);
        }
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        for segments in [
            &["std", "alloc", "alloc_zeroed"][..],
            &["alloc", "alloc", "alloc_zeroed"][..],
            &["alloc_zeroed"][..],
        ] {
            let key = FunctionPathKey::from_segments(segments.iter().copied());
            assert!(
                registry.lookup(&key).is_none(),
                "user graph still registered for {segments:?}"
            );
        }
        let host = crate::flowspace::model::host_env_callable(
            &["std", "alloc", "alloc_zeroed"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>(),
        )
        .expect("std.alloc.alloc_zeroed resolves in HOST_ENV");
        assert!(crate::annotator::builtin::is_registered(host.qualname()));
        let alloc_host = crate::flowspace::model::host_env_callable(
            &["alloc", "alloc", "alloc_zeroed"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>(),
        )
        .expect("alloc.alloc.alloc_zeroed resolves in HOST_ENV");
        assert_eq!(host, alloc_host);

        let stub_registry = CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        ));
        let mut specs = vec![(
            vec!["pyobject".to_string(), "is_none".to_string()],
            Signature::new(vec!["obj".to_string()], None, None),
            Some("bool".to_string()),
        )];
        for segments in [
            vec!["std", "alloc", "alloc_zeroed"],
            vec!["std", "alloc", "alloc"],
            vec!["alloc", "alloc", "alloc_zeroed"],
            vec!["alloc_zeroed"],
        ] {
            specs.push((
                segments.iter().map(|s| (*s).to_string()).collect(),
                Signature::new(vec!["layout".to_string()], None, None),
                Some("ref".to_string()),
            ));
        }
        register_unsafe_fn_stubs(&stub_registry, &specs);
        assert_eq!(
            stub_registry.len(),
            1,
            "declined alloc/alloc_zeroed specs must not be registered"
        );
        for segments in [
            &["std", "alloc", "alloc_zeroed"][..],
            &["std", "alloc", "alloc"][..],
            &["alloc", "alloc", "alloc_zeroed"][..],
            &["alloc_zeroed"][..],
        ] {
            assert!(
                stub_registry
                    .lookup(&FunctionPathKey::from_segments(segments.iter().copied()))
                    .is_none(),
                "unsafe stub still registered for {segments:?}"
            );
        }
    }

    #[test]
    fn copy_nonoverlapping_is_declined_and_unregistered() {
        for segments in [
            vec![
                "core".to_string(),
                "ptr".into(),
                "copy_nonoverlapping".into(),
            ],
            vec!["std".into(), "ptr".into(), "copy_nonoverlapping".into()],
            vec!["copy_nonoverlapping".into()],
        ] {
            assert_eq!(
                registration_decline(&segments),
                Some(SKIP_PTR_COPY_NONOVERLAPPING),
                "{segments:?}"
            );
        }
        assert_eq!(
            registration_decline(&["core".into(), "ptr".into(), "copy".into()]),
            None
        );

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let mut graphs = crate::codewriter::call::GraphStore::default();
        for segments in [
            vec!["core", "ptr", "copy_nonoverlapping"],
            vec!["std", "ptr", "copy_nonoverlapping"],
            vec!["copy_nonoverlapping"],
        ] {
            let path = crate::parse::CallPath {
                segments: segments.iter().map(|s| (*s).to_string()).collect(),
            };
            let mut graph = LegacyGraph::new("copy_nonoverlapping");
            graph.set_return(graph.startblock.clone(), None);
            graphs.insert(path, graph);
        }
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        for segments in [
            &["core", "ptr", "copy_nonoverlapping"][..],
            &["std", "ptr", "copy_nonoverlapping"][..],
            &["copy_nonoverlapping"][..],
        ] {
            let key = FunctionPathKey::from_segments(segments.iter().copied());
            assert!(
                registry.lookup(&key).is_none(),
                "user graph still registered for {segments:?}"
            );
        }
        let host = crate::flowspace::model::host_env_callable(
            &["core", "ptr", "copy_nonoverlapping"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>(),
        )
        .expect("core.ptr.copy_nonoverlapping resolves in HOST_ENV");
        assert!(crate::annotator::builtin::is_registered(host.qualname()));

        let stub_registry = CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        ));
        let mut specs = vec![(
            vec!["pyobject".to_string(), "is_none".to_string()],
            Signature::new(vec!["obj".to_string()], None, None),
            Some("bool".to_string()),
        )];
        for segments in [
            vec!["core", "ptr", "copy_nonoverlapping"],
            vec!["std", "ptr", "copy_nonoverlapping"],
            vec!["copy_nonoverlapping"],
        ] {
            specs.push((
                segments.iter().map(|s| (*s).to_string()).collect(),
                Signature::new(
                    vec!["dst".to_string(), "src".to_string(), "count".to_string()],
                    None,
                    None,
                ),
                None,
            ));
        }
        register_unsafe_fn_stubs(&stub_registry, &specs);
        assert_eq!(
            stub_registry.len(),
            1,
            "declined copy_nonoverlapping specs must not be registered"
        );
        assert!(
            stub_registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyobject".to_string(),
                    "is_none".to_string()
                ]))
                .is_some()
        );
        for segments in [
            &["core", "ptr", "copy_nonoverlapping"][..],
            &["std", "ptr", "copy_nonoverlapping"][..],
            &["copy_nonoverlapping"][..],
        ] {
            assert!(
                stub_registry
                    .lookup(&FunctionPathKey::from_segments(segments.iter().copied()))
                    .is_none(),
                "unsafe stub still registered for {segments:?}"
            );
        }
    }

    /// Residual `majit_rlib::jit::conditional_call*` bodies are
    /// `ConditionalCallEntry`, not graphs. Layer-1 would otherwise beat
    /// HOST_ENV and the annotator would walk `function(a)` as an ll
    /// funcptr call.
    #[test]
    fn conditional_call_residuals_are_declined_and_host_env_wins() {
        let leaves = [
            "conditional_call0",
            "conditional_call1",
            "conditional_call2",
            "conditional_call3",
            "conditional_call4",
            "conditional_call_elidable1",
        ];
        for leaf in leaves {
            assert_eq!(
                registration_decline(&["jit".into(), leaf.into()]),
                Some(SKIP_JIT_CONDITIONAL_CALL_EXTREGISTRY),
                "{leaf}"
            );
            assert_eq!(
                registration_decline(&["majit_rlib".into(), "jit".into(), leaf.into()]),
                Some(SKIP_JIT_CONDITIONAL_CALL_EXTREGISTRY),
                "majit_rlib::jit::{leaf}"
            );
        }
        assert_eq!(
            registration_decline(&["jit".into(), "we_are_jitted".into()]),
            None
        );
        assert_eq!(
            registration_decline(&["majit_rlib".into(), "jit".into(), "we_are_jitted".into()]),
            None
        );

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let mut graphs = crate::codewriter::call::GraphStore::default();
        for leaf in leaves {
            let path = crate::parse::CallPath::from_segments(["majit_rlib", "jit", leaf]);
            let mut graph = LegacyGraph::new(leaf);
            graph.set_return(graph.startblock, None);
            graphs.insert(path, graph);
        }
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        for leaf in leaves {
            let key = FunctionPathKey::from_segments(["majit_rlib", "jit", leaf]);
            assert!(
                registry.lookup(&key).is_none(),
                "user graph still registered for {leaf}"
            );
            let host = crate::flowspace::model::host_env_callable(
                &["majit_rlib", "jit", leaf]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|| panic!("HOST_ENV must resolve majit_rlib.jit.{leaf}"));
            assert!(crate::annotator::builtin::is_registered(host.qualname()));
        }
    }

    /// Residual `majit_gc::header` varsize immortal allocators are
    /// `malloc_varsize` operations, not graphs. Layer-1 would otherwise
    /// beat HOST_ENV and the annotator would walk `std::alloc::Layout`.
    #[test]
    fn varsize_gc_header_allocs_are_declined_and_host_env_wins() {
        let public_leaves = [
            "alloc_varsize_with_gc_header_immortal",
            "alloc_varsize_with_gc_header_immortal_zeroed",
            "dealloc_varsize_with_gc_header",
        ];
        let private_leaves = [
            "alloc_varsize_with_gc_header_immortal_zeroed_flag",
            "varsize_immortal_layout",
        ];
        for leaf in public_leaves.iter().chain(private_leaves.iter()) {
            assert_eq!(
                registration_decline(&["majit_gc".into(), "header".into(), (*leaf).into()]),
                Some(SKIP_VARSIZE_GC_HEADER_ALLOC),
                "{leaf}"
            );
            assert_eq!(
                registration_decline(&["header".into(), (*leaf).into()]),
                Some(SKIP_VARSIZE_GC_HEADER_ALLOC),
                "header::{leaf}"
            );
            assert_eq!(
                registration_decline(&[(*leaf).into()]),
                Some(SKIP_VARSIZE_GC_HEADER_ALLOC),
                "leaf {leaf}"
            );
        }
        assert_eq!(
            registration_decline(&["majit_gc".into(), "header".into(), "header_of".into()]),
            None
        );

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let mut graphs = crate::codewriter::call::GraphStore::default();
        for leaf in public_leaves {
            let path = crate::parse::CallPath::from_segments(["majit_gc", "header", leaf]);
            let mut graph = LegacyGraph::new(leaf);
            graph.set_return(graph.startblock, None);
            graphs.insert(path, graph);
        }
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        for leaf in public_leaves {
            let key = FunctionPathKey::from_segments(["majit_gc", "header", leaf]);
            assert!(
                registry.lookup(&key).is_none(),
                "user graph still registered for {leaf}"
            );
            let host = crate::flowspace::model::host_env_callable(
                &["majit_gc", "header", leaf]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|| panic!("HOST_ENV must resolve majit_gc.header.{leaf}"));
            assert!(crate::annotator::builtin::is_registered(host.qualname()));
            let stripped = crate::flowspace::model::host_env_callable(
                &["header", leaf]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|| panic!("HOST_ENV must resolve header.{leaf}"));
            assert!(crate::annotator::builtin::is_registered(
                stripped.qualname()
            ));
        }
    }

    /// Annotate + rtype a caller of `conditional_call_elidable1` with a
    /// high-level Instance callback argument and a fn-const. Walking the
    /// residual body would reject the Instance as an ll funcptr argument;
    /// the HOST_ENV builtin emits `jit_conditional_call_value`.
    #[test]
    fn conditional_call_elidable1_instance_arg_emits_value_llop() {
        let _lock = anchor_lock();
        use crate::flowspace::model::{ConstValue, Constant, GraphFunc, Hlvalue};
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, GCREF};

        let bookkeeper = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let registry = CallRegistry::new(bookkeeper);

        let callback_key = FunctionPathKey::from_segments(["dummy_compute"]);
        let ret = crate::flowspace::model::Variable::new();
        ret.annotation.replace(Some(std::rc::Rc::new(
            crate::annotator::model::SomeValue::Impossible,
        )));
        ret.set_concretetype(Some(GCREF.clone()));
        let startblock = crate::flowspace::model::Block::shared(vec![]);
        let func = GraphFunc::new(
            "dummy_compute",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        let pygraph = std::rc::Rc::new(crate::flowspace::pygraph::PyGraph {
            graph: std::rc::Rc::new(std::cell::RefCell::new(
                crate::flowspace::model::FunctionGraph::with_return_var(
                    "dummy_compute",
                    startblock,
                    Hlvalue::Variable(ret),
                ),
            )),
            func,
            signature: std::cell::RefCell::new(Signature::new(vec!["obj".into()], None, None)),
            defaults: std::cell::RefCell::new(Some(Vec::new())),
            access_directly: std::cell::Cell::new(false),
        });
        registry
            .register_callee(
                callback_key,
                Signature::new(vec!["obj".into()], None, None),
                pygraph,
            )
            .set_declared_funcptr_type(FuncType {
                args: vec![GCREF.clone()],
                result: GCREF.clone(),
            });

        let mut graph = LegacyGraph::new("get_index_storage");
        let vars = mint_vars(&mut graph, 5);
        let v_value = vars[1].clone();
        let v_arg = vars[2].clone();
        let v_fn = vars[3].clone();
        let v_out = vars[4].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1, 2]),
            operations: vec![
                crate::model::SpaceOperation {
                    result: Some(v_value.clone()),
                    kind: crate::model::OpKind::Input {
                        name: "value".into(),
                        ty: ValueType::Ref(None),
                        class_root: Some("GCREF".into()),
                    },
                },
                crate::model::SpaceOperation {
                    result: Some(v_arg.clone()),
                    kind: crate::model::OpKind::Input {
                        name: "arg".into(),
                        ty: ValueType::Ref(None),
                        class_root: None,
                    },
                },
                crate::model::SpaceOperation {
                    result: Some(v_fn.clone()),
                    kind: crate::model::OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec![
                                crate::model::FN_CONST_HEAD.into(),
                                "dummy_compute".into(),
                            ],
                            fun_decl_id: None,
                        },
                        args: vec![],
                        result_ty: ValueType::Int,
                    },
                },
                crate::model::SpaceOperation {
                    result: Some(v_out.clone()),
                    kind: crate::model::OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec![
                                "majit_rlib".into(),
                                "jit".into(),
                                "conditional_call_elidable1".into(),
                            ],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![
                            v_value.clone(),
                            v_fn.clone(),
                            v_arg.clone(),
                        ]),
                        result_ty: ValueType::Ref(None),
                    },
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v_out.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[4]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let (flow, _value_to_var, _, _, _) = drive_subject(&graph, &registry, true)
            .expect("instance-arg value helper must annotate and rtype through the builtin");
        let opnames: Vec<String> = flow
            .borrow()
            .iterblocks()
            .into_iter()
            .flat_map(|block| block.borrow().operations.clone())
            .map(|op| op.opname)
            .collect();
        assert!(
            opnames
                .iter()
                .any(|name| name == "jit_conditional_call_value"),
            "rtyper must emit jit_conditional_call_value, got {opnames:?}"
        );
    }

    /// A registry whose one funcobj, `owner::flagged`, has an annotator
    /// graph flagged `access_directly`.
    fn registry_with_a_flagged_graph() -> (std::rc::Rc<CallRegistry>, crate::parse::CallPath) {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let path = crate::parse::CallPath {
            segments: vec!["owner".into(), "flagged".into()],
        };
        let mut graph = LegacyGraph::new("flagged");
        graph.set_return(graph.startblock, None);
        graph.source_identity = Some("owner::flagged".into());
        let mut graphs = crate::codewriter::call::GraphStore::default();
        graphs.insert(path.clone(), graph);
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        let entry = registry
            .lookup(&FunctionPathKey::from_segments(["owner", "flagged"]))
            .unwrap();
        let pygraph = entry
            .function_desc
            .borrow()
            .cachedgraph(
                crate::annotator::description::GraphCacheKey::None,
                None,
                None,
            )
            .unwrap();
        pygraph.access_directly.set(true);
        (registry, path)
    }

    /// `warmspot.py check_access_directly_sanity`: a JIT graph may be
    /// `access_directly`.
    #[test]
    fn an_access_directly_jit_graph_passes_the_sanity_check() {
        let (registry, path) = registry_with_a_flagged_graph();
        check_access_directly_sanity(&registry, &HashSet::from([path]));
    }

    fn empty_return_graph(name: &str) -> LegacyGraph {
        let mut graph = LegacyGraph::new(name);
        graph.set_return(graph.startblock, None);
        graph
    }

    fn graph_calling_unregistered(name: &str) -> LegacyGraph {
        let mut graph = empty_return_graph(name);
        graph.block_mut(graph.startblock).operations.insert(
            0,
            crate::model::SpaceOperation {
                result: None,
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["no_such".into(), "callee".into()],
                        fun_decl_id: None,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Void,
                },
            },
        );
        graph
    }

    /// Phase A attempted the graph: a failure is annotate-skipped, not
    /// never-a-subject. Upstream `translator.graphs` still contains it
    /// (`driver.py` task_annotate).
    #[test]
    fn phase_a_failure_publishes_as_annotate_skipped_not_never_a_subject() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let path = crate::parse::CallPath {
            segments: vec!["owner".into(), "failing_subject".into()],
        };
        let graph = graph_calling_unregistered("failing_subject");
        let mut graphs = crate::codewriter::call::GraphStore::default();
        graphs.insert(path.clone(), graph.clone());
        run_two_phase_prepass(&registry, &HashSet::from([path.clone()]), &graphs);
        let outcome = dual_gate_outcome_from_cache(&graph, &registry, &path.canonical_key())
            .expect("publish reads the cache");
        let DualGateOutcome::Skip(reason) = outcome else {
            panic!("Phase A fail must Skip, got {outcome:?}");
        };
        assert!(
            reason.contains("annotate-skipped"),
            "Phase A fail must not look like a missed subject, got {reason}"
        );
        assert_eq!(
            non_arm_skip_category(&reason),
            Some("two-phase-annotate-skipped")
        );
    }

    /// A codewriter graph the prepass candidate set did not contain is
    /// never-a-subject.
    #[test]
    fn a_graph_absent_from_the_prepass_set_is_never_a_subject() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let path = crate::parse::CallPath {
            segments: vec!["owner".into(), "unreached".into()],
        };
        let graph = empty_return_graph("unreached");
        let mut graphs = crate::codewriter::call::GraphStore::default();
        graphs.insert(path.clone(), graph.clone());
        run_two_phase_prepass(&registry, &HashSet::new(), &graphs);
        let outcome = dual_gate_outcome_from_cache(&graph, &registry, &path.canonical_key())
            .expect("publish reads the cache");
        let DualGateOutcome::Skip(reason) = outcome else {
            panic!("unreached graph must Skip, got {outcome:?}");
        };
        assert_eq!(
            non_arm_skip_category(&reason),
            Some("two-phase-never-a-subject")
        );
    }

    /// A publish `diag_key` that is not the prepass `canonical_key` must
    /// not steal another graph's cached types through a name scan.
    #[test]
    fn dual_gate_cache_lookup_does_not_fall_back_to_graph_name() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let path = crate::parse::CallPath {
            segments: vec!["owner".into(), "trivial".into()],
        };
        let graph = empty_return_graph("trivial");
        let mut graphs = crate::codewriter::call::GraphStore::default();
        graphs.insert(path.clone(), graph.clone());
        run_two_phase_prepass(&registry, &HashSet::from([path.clone()]), &graphs);
        let outcome = dual_gate_outcome_from_cache(&graph, &registry, "not_the_canonical_key")
            .expect("publish reads the cache");
        let DualGateOutcome::Skip(reason) = outcome else {
            panic!("wrong diag_key must not Match, got {outcome:?}");
        };
        assert_eq!(
            non_arm_skip_category(&reason),
            Some("two-phase-never-a-subject")
        );
    }

    /// `warmspot.py check_access_directly_sanity`: a graph outside the JIT
    /// graph set may not.
    #[test]
    #[should_panic(expected = "access_directly on flagged, which is outside the JIT graph set")]
    fn an_access_directly_graph_outside_the_jit_set_fails_the_sanity_check() {
        let (registry, _) = registry_with_a_flagged_graph();
        check_access_directly_sanity(&registry, &HashSet::new());
    }

    #[test]
    fn registry_population_keeps_memo_source_policy_without_lifting_host_body() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let mut graph = LegacyGraph::new("memo_source");
        graph.source_identity = Some("owner::memo_source".into());
        graph.hints.push("specialize:memo".into());
        let mut graphs = crate::codewriter::call::GraphStore::default();
        graphs.insert(
            crate::parse::CallPath {
                segments: vec!["owner".into(), "memo_source".into()],
            },
            graph,
        );
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        let entry = registry
            .lookup(&FunctionPathKey::from_segments(["owner", "memo_source"]))
            .unwrap();
        assert!(entry.function_desc.borrow().is_memo());
        assert!(entry.function_desc.borrow().cache.borrow().is_empty());
        // Re-populating keeps the canonical callable instead of making a
        // second GraphFunc and silently erasing its specialization policy.
        let id = entry.host_object.user_function().unwrap().id;
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        assert_eq!(
            registry
                .lookup(&FunctionPathKey::from_segments(["owner", "memo_source"]))
                .unwrap()
                .host_object
                .user_function()
                .unwrap()
                .id,
            id
        );
        let err = entry
            .function_desc
            .borrow()
            .specialize(&mut Vec::new(), None)
            .unwrap_err();
        assert!(
            err.to_string().contains("no host_call hook registered"),
            "{err}"
        );
    }

    #[test]
    fn populated_memo_uses_the_registered_native_callable() {
        use crate::annotator::description::SpecializeResult;
        use crate::annotator::model::SomeObjectTrait;
        use crate::flowspace::model::HostCall;
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        let mut graph = LegacyGraph::new("native_memo");
        graph.source_identity = Some("owner::native_memo".into());
        graph.hints.push("specialize:memo".into());
        let mut func = source_graph_func(&graph);
        fn native_body() -> i64 {
            42
        }
        func.host_call = Some(HostCall(std::sync::Arc::new(|args| {
            assert!(args.is_empty());
            Ok(ConstValue::Int(native_body()))
        })));
        let key = FunctionPathKey::from_segments(["owner", "native_memo"]);
        let entry = registry.get_or_register_with_func(
            key,
            Signature::new(vec![], None, None),
            func.clone(),
        );
        let mut graphs = crate::codewriter::call::GraphStore::default();
        graphs.insert(
            crate::parse::CallPath {
                segments: vec!["owner".into(), "native_memo".into()],
            },
            graph,
        );
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();
        assert_eq!(entry.host_object.user_function().unwrap().id, func.id);
        let result = entry
            .function_desc
            .borrow()
            .specialize(&mut Vec::new(), None)
            .unwrap();
        let SpecializeResult::Annotation(value) = result else {
            panic!("memo annotation expected")
        };
        assert!(value.is_constant());
        assert!(matches!(
            *value,
            crate::annotator::model::SomeValue::Integer(_)
        ));
    }

    #[test]
    fn registered_memo_preserves_frozen_input_and_prebuilt_instance_result() {
        // specialize.py:memo feeds real host objects to the evaluator;
        // rclass.py:InstanceRepr.convert_const owns the resulting instance.
        use crate::annotator::description::SpecializeResult;
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{
            HostCall, HostGetAttrError, HostObject, NativeInstanceDict, host_getattr,
        };
        use crate::translator::rtyper::rtyper::RPythonTyper;
        use std::sync::Arc;

        struct NativeValue(i64);
        impl NativeInstanceDict for NativeValue {
            fn get(&self, name: &str) -> Result<ConstValue, HostGetAttrError> {
                if name == "value" {
                    Ok(ConstValue::Int(self.0))
                } else {
                    Err(HostGetAttrError::Missing)
                }
            }
            fn keys(&self) -> Result<Vec<String>, HostGetAttrError> {
                Ok(vec!["value".into()])
            }
        }

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = CallRegistry::new(ann.bookkeeper.clone());
        let frozen_class = HostObject::new_class("owner.FrozenKey", vec![]);
        frozen_class.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "owner.FrozenKey._freeze_",
                Arc::new(|_| Ok(ConstValue::Bool(true))),
            )),
        );
        let key = HostObject::new_native_instance(frozen_class, Arc::new(NativeValue(7)));
        let result = HostObject::new_native_instance(
            HostObject::new_class("owner.PreBuiltResult", vec![]),
            Arc::new(NativeValue(42)),
        );
        let expected_key = key.clone();
        let native_result = result.clone();
        let mut graph = LegacyGraph::new("prebuilt_memo");
        graph.hints.push("specialize:memo".into());
        let mut func = source_graph_func(&graph);
        func.host_call = Some(HostCall(Arc::new(move |args| {
            assert_eq!(args, &[ConstValue::HostObject(expected_key.clone())]);
            assert_eq!(host_getattr(&expected_key, "value"), Ok(ConstValue::Int(7)));
            Ok(ConstValue::HostObject(native_result.clone()))
        })));
        let entry = registry.get_or_register_with_func(
            FunctionPathKey::from_segments(["owner", "prebuilt_memo"]),
            Signature::new(vec!["key".into()], None, None),
            func,
        );
        let s_key = ann
            .bookkeeper
            .immutablevalue(&ConstValue::HostObject(key))
            .unwrap();
        assert!(matches!(s_key, SomeValue::PBC(_)));
        let SpecializeResult::Annotation(s_result) = entry
            .function_desc
            .borrow()
            .specialize(&mut vec![Some(s_key)], None)
            .unwrap()
        else {
            panic!("memo must first return its result annotation")
        };
        assert!(matches!(*s_result, SomeValue::Instance(_)));

        // Read the instance source through ClassDef's ordinary attribute path.
        // No manually seeded field annotation or readonly override: the native
        // object's enumerated source supplies both the cell and mutability.
        let cd = ann
            .bookkeeper
            .getuniqueclassdef(result.instance_class().unwrap())
            .unwrap();
        let field_annotation =
            crate::annotator::classdesc::ClassDef::find_attribute(&cd, "value").unwrap();
        assert!(matches!(field_annotation, SomeValue::Integer(_)));
        assert!(!cd.borrow().attrs["value"].readonly);

        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        let repr = rtyper.getrepr(&s_result).unwrap();
        rtyper.call_all_setups().unwrap();
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);
        let value = ConstValue::HostObject(result);
        let first = repr.convert_const(&value).unwrap();
        let second = repr.convert_const(&value).unwrap();
        let (ConstValue::LLPtr(first_ptr), ConstValue::LLPtr(second_ptr)) =
            (&first.value, &second.value)
        else {
            panic!("prebuilt conversion must produce typed low-level pointers")
        };
        assert!(first_ptr.nonzero());
        assert_eq!(first.concretetype.as_ref(), Some(repr.lowleveltype()));
        assert_eq!(
            first_ptr._hashable_identity(),
            second_ptr._hashable_identity()
        );
        let instance_repr = crate::translator::rtyper::rclass::getinstancerepr(
            &rtyper,
            Some(&cd),
            crate::translator::rtyper::rclass::Flavor::Gc,
        )
        .unwrap();
        let fields = instance_repr.fields();
        let (field, _) = fields.get("value").unwrap();
        let crate::translator::rtyper::lltypesystem::lltype::_ptr_obj::Struct(object) =
            first_ptr._obj().unwrap()
        else {
            panic!("typed prebuilt instance")
        };
        assert!(matches!(
            object._getattr(field).unwrap(),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(42)
        ));
    }

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        crate::model::Link::new_mixed(args, returnblock_id, None)
    }

    /// Mint `n` fresh values on `graph`, returned indexed `0..n` so
    /// fixtures can refer to inputargs / operands positionally.
    fn mint_vars(graph: &mut LegacyGraph, n: usize) -> Vec<crate::flowspace::model::Variable> {
        (0..n).map(|_| graph.alloc_value_var()).collect()
    }

    /// Read the concrete kind for a legacy graph `Variable` from the
    /// post-rtyper `LegacyToTyped` map's `Variable.concretetype` cell.
    /// The map is keyed by the legacy Variable's identity.
    fn kind_of_in(
        value_to_var: &LegacyToTyped,
        var: &crate::flowspace::model::Variable,
    ) -> ConcreteType {
        let typed = value_to_var
            .get(var)
            .unwrap_or_else(|| panic!("value missing from value_to_var"));
        let ll = typed
            .concretetype()
            .unwrap_or_else(|| panic!("Variable.concretetype not populated"));
        lowleveltype_to_concrete(&ll)
            .unwrap_or_else(|_| panic!("lltype does not project to ConcreteType"))
    }

    /// Project positional indices to held Variables for
    /// `Block { inputargs, .. }` literals.
    fn block_inputargs(
        vars: &[crate::flowspace::model::Variable],
        vids: &[usize],
    ) -> Vec<crate::flowspace::model::Variable> {
        vids.iter().map(|&i| vars[i].clone()).collect()
    }

    fn hint_alias_fixture() -> (LegacyGraph, Variable, Variable, LegacyToTyped) {
        let mut graph = LegacyGraph::new("hint_alias_fixture");
        let value = graph.alloc_value_var();
        let result = graph.alloc_value_var();
        let startblock = graph.startblock;
        graph
            .block_mut(startblock)
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(result.clone()),
                kind: crate::model::OpKind::Hint {
                    value: value.clone(),
                    kind: crate::hints::HintKind::Promote,
                },
            });
        let typed_value = Variable::new();
        typed_value.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        ));
        let value_to_var = HashMap::from([(value, typed_value.clone())]);
        (graph, result, typed_value, value_to_var)
    }

    #[test]
    fn reconcile_elided_hint_results_reuses_the_operand_representative() {
        let (graph, result, typed_value, mut value_to_var) = hint_alias_fixture();
        reconcile_elided_hint_results(&graph, &mut value_to_var);
        assert_eq!(
            value_to_var.get(&result),
            Some(&typed_value),
            "RPython's same_as elimination must preserve one representative for value and result"
        );
    }

    #[test]
    fn reconcile_elided_hint_results_replaces_an_untyped_placeholder() {
        let (graph, result, typed_value, mut value_to_var) = hint_alias_fixture();
        value_to_var.insert(result.clone(), Variable::new());
        reconcile_elided_hint_results(&graph, &mut value_to_var);
        assert_eq!(
            value_to_var.get(&result),
            Some(&typed_value),
            "an empty same_as result twin is the erased alias, not an authoritative type"
        );
    }

    #[test]
    fn reconcile_elided_hint_results_preserves_an_existing_result_mapping() {
        let (graph, result, _typed_value, mut value_to_var) = hint_alias_fixture();
        let typed_result = Variable::new();
        typed_result.set_concretetype(Some(LowLevelType::Signed));
        value_to_var.insert(result.clone(), typed_result.clone());
        reconcile_elided_hint_results(&graph, &mut value_to_var);
        assert_eq!(
            value_to_var.get(&result),
            Some(&typed_result),
            "an independently typed result is authoritative and must not be overwritten"
        );
    }

    #[test]
    fn reconcile_elided_phi_inputargs_follows_a_transitive_single_predecessor_chain() {
        let mut graph = LegacyGraph::new("phi_alias_chain");
        let source = graph.alloc_value_var();
        graph.block_mut(graph.startblock).inputargs = vec![source.clone()];
        let (middle_id, middle_args) = graph.create_block_with_arg_vars(1);
        let middle = middle_args[0].clone();
        let (tail_id, tail_args) = graph.create_block_with_arg_vars(1);
        let tail = tail_args[0].clone();
        graph.block_mut(graph.startblock).exits = vec![crate::model::Link::new_mixed(
            vec![LinkArg::Value(source.clone())],
            middle_id,
            None,
        )];
        graph.block_mut(middle_id).exits = vec![crate::model::Link::new_mixed(
            vec![LinkArg::Value(middle.clone())],
            tail_id,
            None,
        )];

        let typed_source = Variable::new();
        typed_source.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        ));
        let mut value_to_var = HashMap::from([
            (source, typed_source.clone()),
            (middle.clone(), Variable::new()),
            (tail.clone(), Variable::new()),
        ]);

        reconcile_elided_phi_inputargs(
            &graph,
            &mut value_to_var,
            &mut HashMap::new(),
            &mut HashMap::new(),
        );
        assert_eq!(value_to_var.get(&middle), Some(&typed_source));
        assert_eq!(value_to_var.get(&tail), Some(&typed_source));
    }

    #[test]
    fn projection_aliases_close_interleaved_phi_and_hint_chains() {
        let mut graph = LegacyGraph::new("phi_hint_chain");
        let source = graph.alloc_value_var();
        graph.block_mut(graph.startblock).inputargs = vec![source.clone()];
        let typed_source = Variable::new();
        typed_source.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        ));
        let mut variables = HashMap::from([(source.clone(), typed_source.clone())]);
        let mut predecessor = graph.startblock;
        let mut incoming = source;
        let mut aliases = Vec::new();
        for _ in 0..3 {
            let (block, inputs) = graph.create_block_with_arg_vars(1);
            graph.set_goto(predecessor, block, vec![incoming]);
            let input = inputs[0].clone();
            let result = graph.alloc_value_var();
            graph
                .block_mut(block)
                .operations
                .push(crate::model::SpaceOperation {
                    result: Some(result.clone()),
                    kind: crate::model::OpKind::Hint {
                        value: input.clone(),
                        kind: crate::hints::HintKind::Promote,
                    },
                });
            for alias in [input, result.clone()] {
                variables.insert(alias.clone(), Variable::new());
                aliases.push(alias);
            }
            predecessor = block;
            incoming = result;
        }
        reconcile_projection_aliases(
            &graph,
            &mut variables,
            &mut HashMap::new(),
            &mut HashMap::new(),
        );
        for alias in aliases {
            assert_eq!(variables.get(&alias), Some(&typed_source));
            assert!(alias.concretetype().is_none(), "do not seed the baseline");
        }
    }

    #[test]
    fn reconcile_elided_phi_inputargs_requires_one_shared_representative() {
        let mut graph = LegacyGraph::new("phi_alias_conflict");
        let left = graph.alloc_value_var();
        let right = graph.alloc_value_var();
        graph.block_mut(graph.startblock).inputargs = vec![left.clone(), right.clone()];
        let (target_id, target_args) = graph.create_block_with_arg_vars(1);
        let target = target_args[0].clone();
        graph.block_mut(graph.startblock).exits = vec![
            crate::model::Link::new_mixed(vec![LinkArg::Value(left.clone())], target_id, None),
            crate::model::Link::new_mixed(vec![LinkArg::Value(right.clone())], target_id, None),
        ];
        let typed_left = Variable::new();
        typed_left.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        ));
        let typed_right = Variable::new();
        typed_right.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone(),
        ));
        let untyped_target = Variable::new();
        let mut value_to_var = HashMap::from([
            (left, typed_left),
            (right, typed_right),
            (target.clone(), untyped_target.clone()),
        ]);

        reconcile_elided_phi_inputargs(
            &graph,
            &mut value_to_var,
            &mut HashMap::new(),
            &mut HashMap::new(),
        );
        assert_eq!(
            value_to_var.get(&target),
            Some(&untyped_target),
            "same kind is insufficient: upstream unions exact phi representatives"
        );
    }

    #[test]
    fn constant_phi_aliases_require_equal_values_and_publish_after_acceptance() {
        for (right, expected_alias) in [(1, true), (2, false)] {
            let mut graph = LegacyGraph::new("constant_phi");
            let literal = graph.alloc_value_var();
            let (join, inputs) = graph.create_block_with_arg_vars(1);
            let input = inputs[0].clone();
            let (tail, tail_inputs) = graph.create_block_with_arg_vars(1);
            let make_constant =
                |value| Constant::with_concretetype(ConstValue::Int(value), LowLevelType::Unsigned);
            let first = make_constant(1);
            graph.block_mut(graph.startblock).exits = vec![
                crate::model::Link::new_mixed(vec![LinkArg::Value(literal.clone())], join, None),
                crate::model::Link::new_mixed(
                    vec![LinkArg::Const(make_constant(right))],
                    join,
                    None,
                ),
            ];
            graph.block_mut(join).exits = vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(input.clone())],
                tail,
                None,
            )];
            let mut values = HashMap::from([(literal.clone(), Hlvalue::Constant(first))]);
            let mut types = HashMap::from([(literal, LowLevelType::Unsigned)]);
            let mut variables = HashMap::from([
                (input.clone(), Variable::new()),
                (tail_inputs[0].clone(), Variable::new()),
            ]);
            reconcile_elided_phi_inputargs(&graph, &mut variables, &mut values, &mut types);
            assert_eq!(types.contains_key(&input), expected_alias);
            assert_eq!(types.contains_key(&tail_inputs[0]), expected_alias);
            // Alias reconstruction must not contaminate the independent baseline.
            assert!(input.concretetype().is_none());
            assert!(tail_inputs[0].concretetype().is_none());
            if expected_alias {
                assert_eq!(types.get(&input), Some(&LowLevelType::Unsigned));
                crate::codewriter::type_state::apply_from_flowspace_constants(&types);
                assert_eq!(input.concretetype(), Some(LowLevelType::Unsigned));
                assert_eq!(tail_inputs[0].concretetype(), Some(LowLevelType::Unsigned));
            }
        }
    }

    fn backfill_call_result_fixture(
        target: crate::model::CallTarget,
        result_ty: ValueType,
    ) -> (LegacyGraph, Variable, LegacyToTyped) {
        let mut graph = LegacyGraph::new("backfill_call_result_fixture");
        let result = graph.alloc_value_var();
        graph
            .block_mut(graph.startblock)
            .operations
            .push(crate::model::SpaceOperation {
                result: Some(result.clone()),
                kind: crate::model::OpKind::Call {
                    target,
                    args: Vec::new(),
                    result_ty,
                },
            });
        let typed = Variable::new();
        typed.set_concretetype(Some(LowLevelType::Void));
        let value_to_var = HashMap::from([(result.clone(), typed)]);
        (graph, result, value_to_var)
    }

    #[test]
    fn backfill_untyped_call_results_carries_synthesized_closure_scalar_output() {
        for (result_ty, expected) in [
            (ValueType::Int, ConcreteType::Signed),
            (ValueType::Float, ConcreteType::Float),
        ] {
            for method in ["call_once", "call_mut"] {
                let (graph, result, value_to_var) = backfill_call_result_fixture(
                    crate::model::CallTarget::method(
                        method,
                        Some("module::function::closure#2".to_string()),
                    ),
                    result_ty.clone(),
                );
                backfill_untyped_call_results(&graph, &value_to_var);
                assert_eq!(
                    kind_of_in(&value_to_var, &result),
                    expected,
                    "{method}: the closure's declared output must survive Charon erasure"
                );
            }
        }
    }

    #[test]
    fn backfill_untyped_call_results_leaves_ordinary_scalar_conflict_visible() {
        let (graph, result, value_to_var) = backfill_call_result_fixture(
            crate::model::CallTarget::FunctionPath {
                segments: vec!["ordinary_scalar_call".to_string()],
                fun_decl_id: None,
            },
            ValueType::Int,
        );
        backfill_untyped_call_results(&graph, &value_to_var);
        assert_eq!(
            kind_of_in(&value_to_var, &result),
            ConcreteType::Void,
            "an ordinary Signed/Void conflict must still reach the dual-gate divergence"
        );
    }

    #[test]
    fn lowleveltype_to_concrete_signed_family_collapses_to_signed() {
        for ll in [
            LowLevelType::Signed,
            LowLevelType::Unsigned,
            LowLevelType::Bool,
            LowLevelType::Char,
            LowLevelType::UniChar,
            LowLevelType::SingleFloat,
            LowLevelType::Address,
        ] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                ConcreteType::Signed,
                "{ll:?} must project to Signed"
            );
        }
    }

    #[test]
    fn known_unported_classifies_indirect_call_adapter_invariant() {
        let msg = "translate_op: Call with CallTarget::Indirect at result=Some(v4) \
                   must be lowered to VtableMethodPtr + IndirectCall by rclass.rs before \
                   reaching the flowspace adapter";
        assert_eq!(
            unported_category(msg),
            Some("indirect-call-pre-rtyper"),
            "the indirect-call gap keeps its stable fallback category"
        );
        assert!(is_known_unported(msg));
    }

    #[test]
    fn known_unported_classifies_noneify_on_ptr_merge() {
        // A `_ptr ∪ NoneType` block merge reaches `SomeValue::noneify()`
        // on a `SomePtr` operand — a parity-correct raise (default
        // `noneify` raises, `SomePtr` has no override) the dual gate
        // defers to legacy until `front::mir` lowers a typed null pointer.
        let msg = "RPython noneify() not supported for this annotation";
        assert_eq!(
            unported_category(msg),
            Some("noneify-on-ptr"),
            "the typed-null-pointer gap keeps its stable fallback category"
        );
        assert!(is_known_unported(msg));
    }

    #[test]
    fn unknown_rtyper_failure_is_not_skip_classified() {
        let msg = "unexpected rtyper invariant failure";
        assert_eq!(unported_category(msg), None);
        assert!(!is_known_unported(msg));
    }

    #[test]
    fn unported_category_table_matches_former_contains_arms() {
        let cases: &[(&str, Option<&str>)] = &[
            (
                "helper not registered in CallRegistry",
                Some("call-registry-miss"),
            ),
            (
                "translate_op: undefined operand v2",
                Some("undefined-operand"),
            ),
            (
                "unimplemented operation: foo",
                Some("unimplemented-operation"),
            ),
            (
                "variable v3 used before definition",
                Some("used-before-definition"),
            ),
            (
                "MissingRTypeAttribute: bar",
                Some("missing-rtype-attribute"),
            ),
            (
                "compute_at_fixpoint failed: inner",
                Some("annotator-fixpoint-failed"),
            ),
            (
                "translate_op: Call with CallTarget::Indirect at result=Some(v4) \
                 must be lowered to VtableMethodPtr + IndirectCall by rclass.rs before \
                 reaching the flowspace adapter",
                Some("indirect-call-pre-rtyper"),
            ),
            (
                "RPython noneify() not supported for this annotation",
                Some("noneify-on-ptr"),
            ),
            (
                "dict key eq function not wired for StringRepr",
                Some("dict-key-eq-missing"),
            ),
            (
                "Call with CallTarget::Indirect must be rewritten by rclass",
                Some("indirect-call-pre-rtyper"),
            ),
            ("variable v3 is live", None),
        ];
        for &(msg, expected) in cases {
            assert_eq!(unported_category(msg), expected, "{msg}");
        }
        assert_ne!(
            unported_category("variable v3 is live"),
            Some("used-before-definition"),
            "a single needle of a conjunction row must not classify as that category",
        );
    }

    /// Callee-half twin of
    /// [`an_unclassified_translated_op_fails_the_build_instead_of_skipping`].
    /// `unported_category` may Skip a recorded call-path class
    /// (`call-registry-miss`).  A new class string must stay unclassified
    /// so it cannot hide as a skip.
    #[test]
    fn a_new_call_path_class_is_not_skip_classified() {
        let recorded = unported_category("not registered in CallRegistry");
        assert_eq!(recorded, Some("call-registry-miss"));
        let novel = "call-path-class: unexpected-new-taxonomy-entry";
        assert_eq!(
            unported_category(novel),
            None,
            "a call-path class outside the recorded taxonomy must fail the build"
        );
        assert!(!is_known_unported(novel));
    }

    /// Graph carrying one `UnaryOp` under `op`, both operand and result
    /// pre-bound `Int`, returning the result.  `op` decides whether
    /// `flowspace_adapter::normalize_unary_op_name` accepts the graph, so a
    /// name it does not register is the smallest unsupported translated op a
    /// fixture can build.
    fn one_unary_op_graph(name: &str, op: &str) -> LegacyGraph {
        let mut graph = LegacyGraph::new(name);
        let vars = mint_vars(&mut graph, 3);
        let operand = vars[1].clone();
        let result = vars[2].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(result.clone()),
                kind: crate::model::OpKind::UnaryOp {
                    op: op.to_string(),
                    operand: operand.clone(),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(result.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        setbinding(&operand, ValueType::Int);
        setbinding(&result, ValueType::Int);
        graph
    }

    /// The negative coverage the retirement plan asks for: introducing a
    /// translated op the pipeline does not support has to fail the build, not
    /// recover through the legacy walker.
    ///
    /// The fork is `dual_gate_check_with_registry`'s real-path arm — an error
    /// `is_known_unported` classifies becomes `DualGateOutcome::Skip` and the
    /// graph silently publishes the legacy walker's types, while an
    /// unclassified one becomes `Err` and reaches
    /// `dual_gate_publish_concretetypes`'s final `panic!`.  A new
    /// `unported_category` arm broad enough to swallow this shape would move
    /// the whole class from the first behaviour to the second without any
    /// other test noticing, which is what this pins.
    #[test]
    #[should_panic(expected = "MAJIT_RTYPER real-path failure")]
    fn an_unclassified_translated_op_fails_the_build_instead_of_skipping() {
        let _lock = anchor_lock();
        let graph = one_unary_op_graph(
            "negative_coverage_unsupported_unary",
            "__negative_coverage_unported_op__",
        );
        let mut codewriter = crate::codewriter::CodeWriter::new();
        let mut callcontrol = crate::codewriter::call::CallControl::new();
        codewriter.dual_gate_publish_concretetypes(&graph, &mut callcontrol, "negative_coverage");
    }

    /// The half of the test above that a `should_panic` cannot state: the
    /// adapter's own refusal text is the one that must stay unclassified.
    /// `normalize_unary_op_name` registers `pos` / `neg` / `invert` / `bool`
    /// and the ported `str`; anything else raises this message, and
    /// `unported_category` deliberately carries no arm for it.
    #[test]
    fn the_unary_op_adapter_refusal_carries_no_skip_category() {
        let msg = "normalize_unary_op_name: pyre UnaryOp \
                   `__negative_coverage_unported_op__` has no flowspace counterpart";
        assert_eq!(
            unported_category(msg),
            None,
            "an unsupported unary op must not be Skip-classified — a category here \
             turns a build failure into a silent legacy-walker recovery"
        );
        assert!(!is_known_unported(msg));
    }

    /// An annotator half that fails leaves the unwind carrying its
    /// `AnnotatorError`, not a rendering of it — the payload's type is what
    /// names the failing stage, the way the exception class does upstream.
    /// Every consumer therefore has to read it through
    /// `panic_payload_text`; a site that still hand-rolls
    /// `downcast_ref::<String>` sees "unrecognised" and turns a routine
    /// unported subject into a hard dual-gate failure.
    #[test]
    fn annotator_error_payload_routes_through_the_shared_reader() {
        use crate::annotator::model::{AnnotatorError, panic_payload_text};

        let payload = std::panic::catch_unwind(|| {
            std::panic::panic_any(
                AnnotatorError::new("calltable row not found in CallFamily for simple_call")
                    .in_stage("compute_at_fixpoint"),
            )
        })
        .expect_err("panic_any always unwinds");

        assert!(
            payload.downcast_ref::<String>().is_none(),
            "the payload is the error object, so the string downcast must miss"
        );

        let text = panic_payload_text(&*payload).expect("the object arm reads it");
        assert!(
            text.contains("compute_at_fixpoint failed"),
            "the entry point stays in the message: {text:?}"
        );
        assert_eq!(
            unported_category(&text),
            Some("annotator-fixpoint-failed"),
            "an annotator-stage failure keeps its Skip route: {text:?}"
        );
        assert!(is_known_unported(&text));
    }

    #[test]
    fn setbinding_backwards_binding_classifies_as_union_adt() {
        // `unary_invert`/`unary_negative`/`descroperation::{invert,neg}` fail
        // when a `SomeInteger.knowntypedata` inner annotation refines from a
        // generic `Option` to a per-instantiation `Option<Result<..>>`
        // classdef between passes — the same generic-ADT union gap that the
        // mergeinputargs `UnionError` failures carry, one lattice step later.
        let msg = "setbinding: new value does not contain old (Integer(SomeInteger { \
                   knowntypedata: Some({Int(1): {v: Instance(Option<Result<*mut \
                   PyObject,PyError>>::Some)}}) }) ⊄ Integer(SomeInteger { \
                   knowntypedata: Some({Int(1): {v: Instance(Option::Some)}}) }))";
        assert_eq!(
            classify_unported_reason(msg),
            "UNION-ERROR (generic-ADT phi / pair)",
            "backwards setbinding on per-instantiation classdef knowntypedata \
             shares the generic-ADT union disposition, not UNCLASSIFIED"
        );
    }

    /// A union fallback splits on operand kind, because the two halves call
    /// for opposite work and only one of them is a porting task.
    ///
    /// Upstream defines `union` on same-kind pairs almost throughout
    /// (`binaryop.py`: `pairtype(SomeInteger, SomeInteger)`,
    /// `(SomeList, SomeList)`, `(SomeInstance, SomeInstance)`, …), with the
    /// `SomeNone`/`SomeImpossibleValue` bridges and one cross-kind
    /// `(SomeUnicodeString, SomeInteger)` as the exceptions. A cross-kind pair
    /// therefore reaches `pairtype(SomeObject, SomeObject).union`, which
    /// raises — and `llannotation.py` spells that out for the ll level, with
    /// `pairtype(SomePtr, SomeObject).union` raising in its own body. There is
    /// no arm to port for those; the merge is the defect.
    #[test]
    fn a_cross_kind_union_fallback_is_not_reported_as_a_porting_gap() {
        let cross = "annotate PANIC: UnionError in mergeinputargs: UnionError: \
                     pair(s1, s2).union() found no arm: \
                     Instance(object_array::ItemsBlock) ∪ List";
        assert_eq!(
            classify_unported_reason(cross),
            "UNION-CROSS-KIND (a phi upstream never builds)",
            "a List meeting an Instance is a merge upstream raises on too, so \
             naming it a porting gap prescribes an arm RPython does not have"
        );

        // Same kind, different classes: `pairtype(SomeInstance, SomeInstance)`
        // exists upstream, so this one really is pyre's arm to write.
        let same = "pair(s1, s2).union() found no arm: \
                    Instance(option::Option<Wtf8Buf>::None) ∪ Instance(Exception)";
        assert_eq!(
            classify_unported_reason(same),
            "UNION-PAIR-PORT (same-kind arm not written)"
        );

        // A message the split cannot read must land in the bucket that asks
        // for work, never in the one that closes the question.
        assert_eq!(
            classify_unported_reason("pair(s1, s2).union() found no arm"),
            "UNION-PAIR-PORT (same-kind arm not written)"
        );
    }

    #[test]
    fn lowleveltype_to_concrete_float_family_collapses_to_float() {
        {
            let ll = LowLevelType::Float;
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                ConcreteType::Float,
                "{ll:?} must project to Float"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_longlong_follows_target_word_size() {
        // history.py:55-58 — `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed):
        //   if supports_longlong and TYPE is not lltype.LongFloat: return 'float'`
        let expected = if std::mem::size_of::<i64>() > std::mem::size_of::<isize>() {
            ConcreteType::Float
        } else {
            ConcreteType::Signed
        };
        for ll in [LowLevelType::SignedLongLong, LowLevelType::UnsignedLongLong] {
            assert_eq!(
                lowleveltype_to_concrete(&ll).expect("supported lltype"),
                expected,
                "{ll:?} must match history.getkind's sizeof(TYPE) > sizeof(Signed) branch"
            );
        }
    }

    #[test]
    fn lowleveltype_to_concrete_extended_floats_return_missing_rtype_err() {
        let cases = [
            ("SignedLongLongLong", LowLevelType::SignedLongLongLong),
            ("UnsignedLongLongLong", LowLevelType::UnsignedLongLongLong),
            ("LongFloat", LowLevelType::LongFloat),
        ];
        for (name, ty) in cases {
            let msg = format!("case {name}: has no history.getkind mapping");
            let err = lowleveltype_to_concrete(&ty).expect_err(&msg);
            assert!(err.is_missing_rtype_operation(), "case {name}");
        }
    }

    #[test]
    fn lowleveltype_to_concrete_void_passes_through() {
        assert_eq!(
            lowleveltype_to_concrete(&LowLevelType::Void).expect("supported lltype"),
            ConcreteType::Void
        );
    }

    /// Build a single-value passthrough graph (`start(v1) -> return(v1)`)
    /// whose `v1` is reachable, publish `legacy_kind` onto `v1`, and diff
    /// it against a `real_state` that types `v1` as `real_kind`.  Returns
    /// the divergence list `collect_divergences` produces.
    fn diff_single_var(legacy_kind: ConcreteType, real_kind: ConcreteType) -> Vec<String> {
        let mut graph = LegacyGraph::new("diff_single_var");
        let vars = mint_vars(&mut graph, 2);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        crate::model::FunctionGraph::set_concretetype_of_inline(&v1_var, legacy_kind);
        let mut real_state = HashMap::new();
        real_state.insert(v1_var, real_kind);
        collect_divergences(&real_state, &graph)
    }

    #[test]
    fn collect_divergences_accepts_gcref_refined_to_void() {
        // The real path refining a boxed-`Result` `GcRef` slot to the
        // `?`-extracted `Ok(())` unit (`Void`) is sound, not a
        // divergence: a `Void` value is dropped from register allocation
        // (`flatten.py:377,386`), so no consumer reads the slot as a
        // pointer.  `diff_single_var` forwards `v1` only into the
        // returnblock, whose 1-arg `make_return` arm emits `void_return`
        // without coloring — the sound dropped-unit case.
        assert!(
            diff_single_var(ConcreteType::GcRef, ConcreteType::Void).is_empty(),
            "legacy=GcRef, real=Void is a sound unit refinement, not a divergence"
        );
    }

    /// Build `start(v1) -> exceptblock(v1)` — `v1` is forwarded as the
    /// exceptblock raise operand, which `make_return`'s 2-arg arm colors
    /// unconditionally (`flatten.py`).  Publish `legacy_kind` onto
    /// `v1` and diff against a `real_state` typing it `real_kind`.
    fn diff_single_raise_operand(
        legacy_kind: ConcreteType,
        real_kind: ConcreteType,
    ) -> Vec<String> {
        let mut graph = LegacyGraph::new("diff_single_raise_operand");
        let vars = mint_vars(&mut graph, 2);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![
                    LinkArg::Value(v1_var.clone()),
                    LinkArg::Value(v1_var.clone()),
                ],
                graph.exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: block_inputargs(&vars, &[1, 1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, exceptblock];
        crate::model::FunctionGraph::set_concretetype_of_inline(&v1_var, legacy_kind);
        let mut real_state = HashMap::new();
        real_state.insert(v1_var, real_kind);
        collect_divergences(&real_state, &graph)
    }

    #[test]
    fn collect_divergences_rejects_gcref_to_void_on_raise_operand() {
        // A `GcRef`→`Void` refinement on the raise *evalue* (the exceptblock
        // link `args[1]`, the only exception operand `flatten.py:143` colors)
        // is NOT the sound dropped-unit case: a `Void` there hits the
        // uncolored-`Void` `getcolor` panic.  `diff_single_raise_operand`
        // forwards `v1` into both link positions, so it occupies `args[1]`.
        // Keep it a divergence → Skip.
        assert_eq!(
            diff_single_raise_operand(ConcreteType::GcRef, ConcreteType::Void).len(),
            1,
            "legacy=GcRef, real=Void on a colored raise evalue is a divergence, \
             not the accepted returnblock unit drop"
        );
    }

    #[test]
    fn colored_operand_vars_colors_only_the_raise_evalue() {
        // `flatten.py:139-143`: the 2-arg exception link emits `raise`
        // coloring ONLY `args[1]` (the evalue); `args[0]` (the etype) is not
        // passed to `getcolor`.  Build `start(etype, evalue) -> exceptblock`
        // with distinct operands and confirm only the evalue is colored.
        let mut graph = LegacyGraph::new("raise_operand_probe");
        let vars = mint_vars(&mut graph, 2);
        let etype = vars[0].clone();
        let evalue = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0, 1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![
                    LinkArg::Value(etype.clone()),
                    LinkArg::Value(evalue.clone()),
                ],
                graph.exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: block_inputargs(&vars, &[0, 1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, exceptblock];
        let colored = colored_operand_vars(&graph);
        assert!(
            colored.contains(&evalue),
            "the raise evalue (args[1]) is colored"
        );
        assert!(
            !colored.contains(&etype),
            "the raise etype (args[0]) is NOT colored"
        );
    }

    /// Build `start(v1) -> mid(v1, v1) -> return()` where `v1` is a
    /// within-block DUPLICATE inputarg of `mid` (read by a `same_as` op so
    /// it stays reachable), publish `legacy_kind` onto `v1`, and diff
    /// against a `real_state` that leaves it untyped — the shape
    /// `remove_duplicate_inputargs` collapses in the real path.
    fn diff_duplicate_inputarg(legacy_kind: ConcreteType) -> Vec<String> {
        let mut graph = LegacyGraph::new("diff_duplicate_inputarg");
        let vars = mint_vars(&mut graph, 3);
        let v1 = vars[1].clone();
        let mid_id = crate::model::BlockId(7);
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(v1.clone()), LinkArg::Value(v1.clone())],
                mid_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        // `mid` carries `v1` twice as inputargs (the duplicate column) and
        // reads it via a `FieldRead` base so `reachable_defined_vars` keeps
        // it (an unread inputarg would be dropped as a dead phi).
        let midblock = Block {
            id: mid_id,
            inputargs: block_inputargs(&vars, &[1, 1]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(vars[2].clone()),
                kind: crate::model::OpKind::FieldRead {
                    base: v1.clone(),
                    field: crate::model::FieldDescriptor {
                        name: "f".to_string(),
                        owner_root: None,
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Ref(None),
                    pure: false,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(vec![], graph.returnblock)],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, midblock, returnblock];
        crate::model::FunctionGraph::set_concretetype_of_inline(&v1, legacy_kind);
        // Real path merged the duplicate column away → `v1` untyped.
        let real_state = HashMap::new();
        collect_divergences(&real_state, &graph)
    }

    /// Build start → mid → sink → return where `mid` defines an `ArrayRead`
    /// result (`vars[2]`) that no op or exitswitch reads — it is only
    /// forwarded into an unused `sink` inputarg (a dead phi-thread).
    /// `dead_op_result_vars` classifies it; the real path leaves it untyped
    /// (`real=None`). The returnblock itself has no such input: upstream
    /// `transform_dead_op_vars` treats return/except inputargs as implicit
    /// reads because they are observable function results.
    fn diff_dead_op_result(legacy_kind: ConcreteType) -> Vec<String> {
        let mut graph = LegacyGraph::new("diff_dead_op_result");
        let vars = mint_vars(&mut graph, 3);
        let base = vars[0].clone();
        let index = vars[1].clone();
        let dead = vars[2].clone();
        let mid_id = crate::model::BlockId(7);
        let sink_id = crate::model::BlockId(8);
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0, 1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(base.clone()), LinkArg::Value(index.clone())],
                mid_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        // `mid` reads `base[index]` into `dead`, then forwards `dead` on the
        // link into the returnblock without any op consuming it.
        let midblock = Block {
            id: mid_id,
            inputargs: block_inputargs(&vars, &[0, 1]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(dead.clone()),
                kind: crate::model::OpKind::ArrayRead {
                    base: base.clone(),
                    index: index.clone(),
                    item_ty: ValueType::Ref(None),
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
            }],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(dead.clone())],
                sink_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let sinkblock = Block {
            id: sink_id,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(vec![], graph.returnblock)],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, midblock, sinkblock, returnblock];
        crate::model::FunctionGraph::set_concretetype_of_inline(&dead, legacy_kind);
        // Real path dropped the dead op result → `dead` untyped.
        let real_state = HashMap::new();
        collect_divergences(&real_state, &graph)
    }

    /// Same shape as [`diff_dead_op_result`], but `mid` defines the dead
    /// result through a side-effecting `Call` (a non-`canremove` op).  The
    /// real path cannot drop a call result — the call must still run for its
    /// effect — so `dead_op_result_vars` must NOT classify it, and the
    /// untyped `real=None` stays a divergence.
    fn diff_dead_call_result(legacy_kind: ConcreteType) -> Vec<String> {
        let mut graph = LegacyGraph::new("diff_dead_call_result");
        let vars = mint_vars(&mut graph, 3);
        let base = vars[0].clone();
        let index = vars[1].clone();
        let dead = vars[2].clone();
        let mid_id = crate::model::BlockId(7);
        let sink_id = crate::model::BlockId(8);
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0, 1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(base.clone()), LinkArg::Value(index.clone())],
                mid_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let midblock = Block {
            id: mid_id,
            inputargs: block_inputargs(&vars, &[0, 1]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(dead.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["side_effect".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![base.clone(), index.clone()]),
                    result_ty: ValueType::Ref(None),
                },
            }],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(dead.clone())],
                sink_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let sinkblock = Block {
            id: sink_id,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(vec![], graph.returnblock)],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, midblock, sinkblock, returnblock];
        crate::model::FunctionGraph::set_concretetype_of_inline(&dead, legacy_kind);
        let real_state = HashMap::new();
        collect_divergences(&real_state, &graph)
    }

    #[test]
    fn collect_divergences_rejects_dead_call_result_dropped() {
        // A dead result of a side-effecting `Call` is NOT `canremove`, so the
        // `canremove` gate on `dead_op_result_vars` excludes it: an untyped
        // `real=None` on such a var is a genuine divergence, not the
        // dead-var-pass artifact.  Holds for either legacy kind.
        assert_eq!(
            diff_dead_call_result(ConcreteType::Signed).len(),
            1,
            "a Signed dead Call result is not a canremove drop → still diverges"
        );
        assert_eq!(
            diff_dead_call_result(ConcreteType::GcRef).len(),
            1,
            "a GcRef dead Call result is not a canremove drop → still diverges"
        );
    }

    /// Build start → mid → {a, b} where `mid` reads `base.__discriminant`
    /// into `disc` and switches on it (`ExitSwitch::Value(disc)`).  The real
    /// path leaves `disc` untyped (`real=None`) — the rebuilt-switch identity
    /// artifact — so accepting it as non-divergent is the assertion.
    fn diff_switch_discriminant(legacy_kind: ConcreteType) -> Vec<String> {
        let mut graph = LegacyGraph::new("diff_switch_discriminant");
        let vars = mint_vars(&mut graph, 3);
        let base = vars[0].clone();
        let disc = vars[1].clone();
        let mid_id = crate::model::BlockId(7);
        let a_id = crate::model::BlockId(8);
        let b_id = crate::model::BlockId(9);
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(base.clone())],
                mid_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let read = crate::model::SpaceOperation {
            result: Some(disc.clone()),
            kind: crate::model::OpKind::FieldRead {
                base: base.clone(),
                field: crate::model::FieldDescriptor {
                    name: "__discriminant".to_string(),
                    owner_root: Some("pyopcode::StepResult".to_string()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                ty: ValueType::Int,
                pure: true,
            },
        };
        let midblock = Block {
            id: mid_id,
            inputargs: block_inputargs(&vars, &[0]),
            operations: vec![read],
            exitswitch: Some(crate::model::ExitSwitch::Value(disc.clone())),
            exits: vec![
                crate::model::Link::new_mixed(vec![], a_id, None),
                crate::model::Link::new_mixed(vec![], b_id, None),
            ],
            framestate: None,
            dead: false,
        };
        let mk_tail = |id| Block {
            id,
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(vec![], graph.returnblock)],
            framestate: None,
            dead: false,
        };
        let ablock = mk_tail(a_id);
        let bblock = mk_tail(b_id);
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, midblock, ablock, bblock, returnblock];
        crate::model::FunctionGraph::set_concretetype_of_inline(&disc, legacy_kind);
        let real_state = HashMap::new();
        collect_divergences(&real_state, &graph)
    }

    #[test]
    fn collect_divergences_accepts_switch_discriminant_rebuilt() {
        // A `__discriminant` read feeding a block exitswitch that the real
        // path's Result/StepResult exc-transform rebuilt with a fresh switch
        // Variable reads `real=None` under the legacy tag's identity — the
        // rebuilt-switch artifact, accepted for either legacy kind.
        assert!(
            diff_switch_discriminant(ConcreteType::Signed).is_empty(),
            "a Signed __discriminant switch read the real path rebuilt is not a \
             divergence"
        );
        assert!(
            diff_switch_discriminant(ConcreteType::GcRef).is_empty(),
            "a GcRef __discriminant switch read the real path rebuilt is not a \
             divergence"
        );
    }

    #[test]
    fn switch_discriminant_read_vars_requires_a_switch() {
        // The classifier only collects a `__discriminant` read whose result
        // feeds an exitswitch.  A forward-only discriminant read (no switch)
        // is NOT in the set — it would fall through to the dead-op gate
        // instead, so the rebuilt-switch acceptance never over-reaches to a
        // non-switch read.  Rebuild the two graphs and inspect the classifier
        // directly (via `collect_divergences`'s construction path is opaque,
        // so reconstruct the same graphs the diff helper builds).
        let build = |feeds_switch: bool| {
            let mut graph = LegacyGraph::new("classify_probe");
            let vars = mint_vars(&mut graph, 3);
            let base = vars[0].clone();
            let disc = vars[1].clone();
            let mid_id = crate::model::BlockId(7);
            let a_id = crate::model::BlockId(8);
            let read = crate::model::SpaceOperation {
                result: Some(disc.clone()),
                kind: crate::model::OpKind::FieldRead {
                    base: base.clone(),
                    field: crate::model::FieldDescriptor {
                        name: "__discriminant".to_string(),
                        owner_root: Some("pyopcode::StepResult".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Int,
                    pure: true,
                },
            };
            let (exitswitch, exits) = if feeds_switch {
                (
                    Some(crate::model::ExitSwitch::Value(disc.clone())),
                    vec![crate::model::Link::new_mixed(vec![], a_id, None)],
                )
            } else {
                (
                    None,
                    vec![crate::model::Link::new_mixed(
                        vec![LinkArg::Value(disc.clone())],
                        a_id,
                        None,
                    )],
                )
            };
            let midblock = Block {
                id: mid_id,
                inputargs: block_inputargs(&vars, &[0]),
                operations: vec![read],
                exitswitch,
                exits,
                framestate: None,
                dead: false,
            };
            graph.blocks = vec![midblock];
            (graph, disc)
        };
        let (switch_graph, disc_s) = build(true);
        assert!(
            switch_discriminant_read_vars(&switch_graph).contains(&disc_s),
            "a __discriminant read feeding a switch is classified"
        );
        let (fwd_graph, disc_f) = build(false);
        assert!(
            !switch_discriminant_read_vars(&fwd_graph).contains(&disc_f),
            "a forward-only __discriminant read is NOT classified"
        );
    }

    /// Build a single block that reads `base.field` into `pay` and, when
    /// `repack` is set, writes `pay` into `dst.payload` (the extract→repack
    /// of an identity `match` re-wrap).  With `repack=false` the read result
    /// is instead fed to a `Call` (a genuine consumer), so it is NOT a repack
    /// payload.
    fn build_payload_graph(repack: bool) -> (LegacyGraph, Variable, Variable) {
        let mut graph = LegacyGraph::new("payload_probe");
        let vars = mint_vars(&mut graph, 5);
        let source = vars[0].clone();
        let base = vars[1].clone();
        let pay = vars[2].clone();
        let dst = vars[3].clone();
        let mid_id = crate::model::BlockId(7);
        let field = crate::model::FieldDescriptor {
            name: "loop_header_pc".to_string(),
            owner_root: Some("StepResult::CloseLoop".to_string()),
            owner_id: None,
            base_is_deref: None,
            taken_by_address: false,
            inline_vec: false,
            vec_part: None,
            owner_declared_gc: None,
            host_index: None,
            scalar_word: None,
        };
        let read = crate::model::SpaceOperation {
            result: Some(pay.clone()),
            kind: crate::model::OpKind::FieldRead {
                base: base.clone(),
                field: field.clone(),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let consumer = if repack {
            crate::model::SpaceOperation {
                result: None,
                kind: crate::model::OpKind::FieldWrite {
                    base: dst.clone(),
                    field: crate::model::FieldDescriptor {
                        name: "payload".to_string(),
                        owner_root: Some(
                            "pyre_interpreter::pyopcode::StepResult<*mut PyObject>::CloseLoop"
                                .to_string(),
                        ),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: crate::model::LinkArg::Value(pay.clone()),
                    ty: ValueType::Ref(None),
                },
            }
        } else {
            crate::model::SpaceOperation {
                result: Some(dst.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["consume".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![pay.clone()]),
                    result_ty: ValueType::Int,
                },
            }
        };
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(source)],
                mid_id,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let midblock = Block {
            id: mid_id,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![read, consumer],
            exitswitch: None,
            exits: vec![link_to_returnblock(vec![], graph.returnblock)],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, midblock, returnblock];
        (graph, pay, base)
    }

    #[test]
    fn repack_payload_read_vars_classifies_cross_variant_writeback() {
        // A payload read whose sole consumer is a `FieldWrite` remains the raw
        // extract→repack value even when Rust's destination variant renames
        // the field; the same read fed to a Call is a genuine use.
        let (repack_graph, pay_r, _) = build_payload_graph(true);
        assert!(
            repack_payload_read_vars(&repack_graph).contains(&pay_r),
            "a payload read consumed only by a cross-variant FieldWrite is classified"
        );
        let (consumed_graph, pay_c, _) = build_payload_graph(false);
        assert!(
            !repack_payload_read_vars(&consumed_graph).contains(&pay_c),
            "a payload read fed to a real consumer is NOT classified"
        );
    }

    #[test]
    fn repack_payload_carrier_vars_require_only_elided_payload_reads() {
        let (repack_graph, _, carrier) = build_payload_graph(true);
        assert!(
            repack_payload_carrier_vars(&repack_graph).contains(&carrier),
            "the block-local carrier used only by identity-repack reads is classified"
        );

        let (mut consumed_graph, _, consumed_carrier) = build_payload_graph(true);
        let mid = consumed_graph
            .blocks
            .iter_mut()
            .find(|block| block.id == crate::model::BlockId(7))
            .expect("payload middle block");
        mid.operations.push(crate::model::SpaceOperation {
            result: None,
            kind: crate::model::OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["consume_carrier".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![consumed_carrier.clone()]),
                result_ty: ValueType::Void,
            },
        });
        assert!(
            !repack_payload_carrier_vars(&consumed_graph).contains(&consumed_carrier),
            "any non-repack use must keep the carrier as a real divergence"
        );
    }

    #[test]
    fn collect_divergences_accepts_elided_repack_carrier() {
        let (graph, _, carrier) = build_payload_graph(true);
        crate::model::FunctionGraph::set_concretetype_of_inline(&carrier, ConcreteType::GcRef);
        assert!(
            collect_divergences(&HashMap::new(), &graph).is_empty(),
            "an identity arm carrier removed with all its payload reads has no surviving real-path use"
        );
    }

    #[test]
    fn collect_divergences_accepts_gcref_dead_op_result_dropped() {
        // A `legacy=GcRef` op result no op reads (only link-forwarded) is
        // dropped by the real path's `transform_dead_op_vars`, reading
        // `real=Unknown` — the dead-result artifact for erased-element
        // `getarrayitem` reads (swap_values / store_local_value), not a
        // divergence.
        assert!(
            diff_dead_op_result(ConcreteType::GcRef).is_empty(),
            "a GcRef dead op result dropped by transform_dead_op_vars is not a \
             divergence"
        );
    }

    #[test]
    fn collect_divergences_accepts_signed_dead_op_result_dropped() {
        // A `legacy=Signed` dead op result of a `canremove` op (here the
        // `ArrayRead`) is likewise the `transform_dead_op_vars` drop artifact,
        // not a divergence: the `canremove` gate on `dead_op_result_vars`
        // already excludes every side-effecting op, so the dropped binding is
        // sound regardless of the kind the legacy walker assigned it (a folded
        // `idx < len` bounds check is Signed, a dead element read is GcRef).
        assert!(
            diff_dead_op_result(ConcreteType::Signed).is_empty(),
            "a Signed dead op result of a canremove op is the dead-var-pass \
             artifact, not a divergence"
        );
    }

    /// Family C: an always-`Err` wrapper keeps a structurally reachable
    /// `Result::Ok.__pos_0` extract whose block `follow_link` never
    /// recorded. The unfiltered compared set reports `real=Unknown`;
    /// restricting to followed vars does not, and a followed untyped
    /// var still diverges — this is not an Unknown-acceptance widening.
    #[test]
    fn collect_divergences_skips_unfollowed_ok_payload() {
        let mut graph = LegacyGraph::new("always_err_wrap_new");
        let vars = mint_vars(&mut graph, 4);
        let v_ok = vars[3].clone();
        let ok_id = BlockId(3);
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0]),
            operations: vec![],
            exitswitch: Some(crate::model::ExitSwitch::LastException),
            exits: vec![
                crate::model::Link::new_mixed(vec![], ok_id, None),
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::Value(vars[1].clone()),
                        LinkArg::Value(vars[2].clone()),
                    ],
                    graph.exceptblock,
                    None,
                ),
            ],
            framestate: None,
            dead: false,
        };
        let ok_block = Block {
            id: ok_id,
            inputargs: vec![],
            operations: vec![
                crate::model::SpaceOperation {
                    result: Some(v_ok.clone()),
                    kind: crate::model::OpKind::FieldRead {
                        base: vars[0].clone(),
                        field: crate::model::FieldDescriptor {
                            name: "__pos_0".to_string(),
                            owner_root: Some("Result::Ok".to_string()),
                            owner_id: None,
                            base_is_deref: Some(false),
                            taken_by_address: false,
                            inline_vec: false,
                            vec_part: None,
                            owner_declared_gc: None,
                            host_index: None,
                            scalar_word: None,
                        },
                        ty: ValueType::Ref(None),
                        pure: false,
                    },
                },
                crate::model::SpaceOperation {
                    result: None,
                    kind: crate::model::OpKind::Call {
                        target: crate::model::CallTarget::Method {
                            name: "is_null".to_string(),
                            receiver_root: Some("mut_ptr".to_string()),
                            resolved_path: None,
                            fun_decl_id: None,
                            branch_payloads: None,
                        },
                        args: crate::model::call_args(vec![v_ok.clone()]),
                        result_ty: ValueType::Bool,
                    },
                },
            ],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v_ok.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![v_ok.clone()],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: block_inputargs(&vars, &[1, 2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock, exceptblock, ok_block];
        crate::model::FunctionGraph::set_concretetype_of_inline(&v_ok, ConcreteType::GcRef);

        let unfiltered = collect_divergences(&HashMap::new(), &graph);
        assert!(
            unfiltered
                .iter()
                .any(|d| d.contains("legacy=GcRef, real=Unknown")),
            "unfiltered compared set must still see the untyped Ok payload: {unfiltered:?}"
        );

        let mut excluded = HashSet::new();
        excluded.insert(v_ok.clone());
        assert!(
            collect_divergences_on(&HashMap::new(), &graph, Some(&excluded)).is_empty(),
            "an unfollowed Ok payload is outside links_followed, not a kind mismatch"
        );

        let unpaired_hit = collect_divergences_on(&HashMap::new(), &graph, Some(&HashSet::new()));
        assert!(
            unpaired_hit
                .iter()
                .any(|d| d.contains("legacy=GcRef, real=Unknown")),
            "a legacy var with no typed twin must still diverge: {unpaired_hit:?}"
        );
    }

    #[test]
    fn dead_op_result_vars_excludes_the_raising_op() {
        // `canremove(op, block)` also excludes `block.raising_op`
        // (`simplify.py`): a `canremove`-classified op that is the last
        // op of a raising block (exitswitch `LastException`) keeps its result
        // even when unread, because the raise side effect is observable.  Two
        // graphs with an identical unread `ArrayRead` last op differ only in
        // whether the block raises; the read is classified dead in the
        // non-raising graph and NOT in the raising one.
        let build = |raises: bool| {
            let mut graph = LegacyGraph::new("raising_probe");
            let vars = mint_vars(&mut graph, 3);
            let base = vars[0].clone();
            let index = vars[1].clone();
            let dead = vars[2].clone();
            let read = crate::model::SpaceOperation {
                result: Some(dead.clone()),
                kind: crate::model::OpKind::ArrayRead {
                    base: base.clone(),
                    index: index.clone(),
                    item_ty: ValueType::Int,
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
            };
            let (exitswitch, exits) = if raises {
                (
                    Some(crate::model::ExitSwitch::LastException),
                    vec![
                        link_to_returnblock(vec![], graph.returnblock),
                        crate::model::Link::new_mixed(vec![], graph.exceptblock, None),
                    ],
                )
            } else {
                (None, vec![link_to_returnblock(vec![], graph.returnblock)])
            };
            let block = Block {
                id: graph.startblock,
                inputargs: block_inputargs(&vars, &[0, 1]),
                operations: vec![read],
                exitswitch,
                exits,
                framestate: None,
                dead: false,
            };
            graph.blocks = vec![block];
            (graph, dead)
        };
        let (nonraising, dead_n) = build(false);
        assert!(
            dead_op_result_vars(&nonraising).contains(&dead_n),
            "an unread canremove op result in a non-raising block is dead"
        );
        let (raising, dead_r) = build(true);
        assert!(
            !dead_op_result_vars(&raising).contains(&dead_r),
            "the raising op's result is NOT dead even when unread"
        );
    }

    #[test]
    fn dead_op_result_vars_propagates_liveness_through_block_local_phi_nodes() {
        let build = |consume_in_successor: bool| {
            let mut graph = LegacyGraph::new("phi_liveness_probe");
            let vars = mint_vars(&mut graph, 4);
            let input = vars[0].clone();
            let threaded = vars[1].clone();
            let call_result = vars[2].clone();
            let middle_id = BlockId(7);
            let producer = crate::model::SpaceOperation {
                result: Some(threaded.clone()),
                kind: crate::model::OpKind::BinOp {
                    op: "eq".to_string(),
                    lhs: input.clone(),
                    rhs: input.clone(),
                    result_ty: ValueType::Int,
                },
            };
            let start = Block {
                id: graph.startblock,
                inputargs: vec![input],
                operations: vec![producer],
                exitswitch: None,
                exits: vec![crate::model::Link::new_mixed(
                    vec![LinkArg::Value(threaded.clone())],
                    middle_id,
                    None,
                )],
                framestate: None,
                dead: false,
            };
            let operations = if consume_in_successor {
                vec![crate::model::SpaceOperation {
                    result: Some(call_result),
                    kind: crate::model::OpKind::Call {
                        target: crate::model::CallTarget::FunctionPath {
                            segments: vec!["consume".into()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![threaded.clone()]),
                        result_ty: ValueType::Int,
                    },
                }]
            } else {
                vec![]
            };
            let middle = Block {
                id: middle_id,
                // Deliberately reuse the legacy identity. The adapter turns
                // this into a distinct orthodox flowspace input Variable.
                inputargs: vec![threaded.clone()],
                operations,
                exitswitch: None,
                exits: vec![link_to_returnblock(vec![], graph.returnblock)],
                framestate: None,
                dead: false,
            };
            let returnblock = Block {
                id: graph.returnblock,
                inputargs: vec![],
                operations: vec![],
                exitswitch: None,
                exits: vec![],
                framestate: None,
                dead: false,
            };
            graph.blocks = vec![start, middle, returnblock];
            (graph, threaded)
        };

        let (dead_graph, dead) = build(false);
        assert!(
            dead_op_result_vars(&dead_graph).contains(&dead),
            "a pure producer forwarded only into an unused phi column is dead"
        );
        let (live_graph, live) = build(true);
        assert!(
            !dead_op_result_vars(&live_graph).contains(&live),
            "successor consumption must flow backward across the link and keep the producer live"
        );
    }

    #[test]
    fn collect_divergences_accepts_gcref_duplicate_inputarg_dropped() {
        // A `legacy=GcRef` duplicate phi inputarg the real path's
        // `remove_duplicate_inputargs` merged away reads `real=Unknown`, but
        // the value is emitted through the surviving representative — a dedup
        // artifact, not a divergence.
        assert!(
            diff_duplicate_inputarg(ConcreteType::GcRef).is_empty(),
            "a GcRef duplicate inputarg dropped by remove_duplicate_inputargs \
             is not a divergence"
        );
    }

    #[test]
    fn collect_divergences_rejects_signed_duplicate_inputarg_dropped() {
        // Scoped to `GcRef`: a `legacy=Signed` untyped duplicate is still a
        // divergence (the erased-`<E>::Value` phi family is GcRef-only; a
        // scalar merged column must not be silently accepted).
        assert_eq!(
            diff_duplicate_inputarg(ConcreteType::Signed).len(),
            1,
            "a Signed duplicate inputarg left untyped is still a divergence"
        );
    }

    #[test]
    fn collect_divergences_rejects_signed_refined_to_void() {
        // The refinement is GcRef-specific.  A `legacy=Signed` slot the
        // real path types `Void` is NOT a sound unit drop: the legacy
        // walker's scalar is a live integer the emit path still colors
        // (`object_key_for_checked` panicked in `lookup_coloring` on a
        // `Some(Void)` value referenced under an `Int` regalloc class), so
        // this pairing is a genuine data-kind divergence.
        assert_eq!(
            diff_single_var(ConcreteType::Signed, ConcreteType::Void).len(),
            1,
            "legacy=Signed, real=Void is a data-kind divergence, not the accepted GcRef unit drop"
        );
    }

    #[test]
    fn collect_divergences_accepts_gcref_refined_to_signed() {
        // A `(GcRef, Signed)` pair is an accepted refinement: the real path
        // resolved a genuine integer (an oparg-decoded index / discriminant /
        // delta) where the legacy walker left the pyre-only conservative
        // `Unknown → GcRef` backfill.  Real is authoritative, so the graph
        // lifts via real rather than Skipping to legacy.
        assert!(
            diff_single_var(ConcreteType::GcRef, ConcreteType::Signed).is_empty(),
            "legacy=GcRef, real=Signed is an accepted real refinement"
        );
    }

    #[test]
    fn collect_divergences_accepts_gcref_refined_to_float() {
        assert!(
            diff_single_var(ConcreteType::GcRef, ConcreteType::Float).is_empty(),
            "legacy=GcRef, real=Float is an accepted real refinement"
        );
    }

    #[test]
    fn select_rtyped_representatives_uses_live_block_local_peer() {
        let legacy = Variable::named("legacy");
        let removed_producer = Variable::named("producer");
        let live_phi = Variable::named("phi");
        live_phi.set_concretetype(Some(LowLevelType::Signed));
        let mut preferred = HashMap::from([(legacy.clone(), removed_producer.clone())]);
        let candidates =
            HashMap::from([(legacy.clone(), vec![removed_producer, live_phi.clone()])]);

        select_rtyped_representatives(&mut preferred, &candidates)
            .expect("same legacy value's live peer must reconcile");
        assert_eq!(
            preferred[&legacy], live_phi,
            "projection must point at the positively-rtyped block-local representative"
        );
    }

    #[test]
    fn collect_divergences_rejects_signed_vs_gcref() {
        // The inverse pairing is likewise a divergence.
        assert_eq!(
            diff_single_var(ConcreteType::Signed, ConcreteType::GcRef).len(),
            1,
            "legacy=Signed, real=GcRef is a real data-kind divergence"
        );
    }

    #[test]
    fn collect_divergences_rejects_void_refined_to_gcref() {
        // The refinement is one-directional: the legacy walker resolving
        // a real=Void slot to GcRef is still a divergence (the real path
        // dropped a value the legacy walker kept as a pointer).
        assert_eq!(
            diff_single_var(ConcreteType::Void, ConcreteType::GcRef).len(),
            1,
            "legacy=Void, real=GcRef is not the accepted refinement direction"
        );
    }

    /// Process-local guard serialising specialize-driving tests.
    /// The rtyper's lattice has process-global singletons that mutate
    /// during `setup()`; cargo's parallel test runner can interleave
    /// two specialize-driving tests so one observes the other's
    /// `setupstate::InProgress` and panics with "recursive invocation
    /// of Repr setup()".  Holding this `Mutex` serialises the runs.
    fn anchor_lock() -> parking_lot::MutexGuard<'static, ()> {
        static LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
        LOCK.lock()
    }

    #[test]
    fn specialize_legacy_graph_minimal_int_identity_resolves_signed() {
        let _lock = anchor_lock();
        // Smallest validation: identity-return graph carrying a single
        // Int-typed inputarg. Dual-gate's first anchor — proves
        // the adapter + annotator-surface seeding + specialize +
        // projection chain works end-to-end on a graph the rtyper can
        // resolve without any unported OpKind variants.
        let mut graph = LegacyGraph::new("identity_int");
        let vars = mint_vars(&mut graph, 2);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Int);
        let (value_to_var, _constants) =
            specialize_legacy_graph(&graph).expect("identity int graph must specialize");

        assert_eq!(
            kind_of_in(&value_to_var, &vars[1]),
            ConcreteType::Signed,
            "Int-typed inputarg must specialize to Signed via SomeInteger → IntegerRepr"
        );
    }

    #[test]
    fn specialize_legacy_graph_minimal_float_identity_resolves_float() {
        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("identity_float");
        let vars = mint_vars(&mut graph, 2);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Float);
        let (value_to_var, _constants) =
            specialize_legacy_graph(&graph).expect("identity float graph must specialize");

        assert_eq!(
            kind_of_in(&value_to_var, &vars[1]),
            ConcreteType::Float,
            "Float-typed inputarg must specialize to Float via SomeFloat → FloatRepr"
        );
    }

    /// `&S` where `S` is Raw and holds `&T` (`T` Raw with an `i64` field).
    /// The input annotates as `SomePtr(Ptr(Struct raw))`, the inner pointer
    /// field stays a pointer, and the two reads assemble as `getfield_raw_i`.
    #[test]
    fn raw_ptr_input_field_read_annotates_and_assembles_getfield_raw() {
        use crate::annotator::model::SomeValue;
        use crate::codewriter::assembler::{Assembler, AssemblerDescr, AssemblerExt};
        use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, OpKind};
        use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType, PtrTarget};
        use std::rc::Rc;

        let _lock = anchor_lock();
        let outer = "rawptr::S";
        let inner = "rawptr::T";
        let outer_id = majit_ir::descr::StructId::from_canonical(outer);
        let inner_id = majit_ir::descr::StructId::from_canonical(inner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (outer.to_string(), Some(outer_id)),
                (inner.to_string(), Some(inner_id)),
            ]));

        let mut fields = crate::front::StructFieldRegistry::default();
        fields
            .fields
            .insert(outer.to_string(), vec![("t".into(), format!("&{inner}"))]);
        fields
            .fields
            .insert(inner.to_string(), vec![("x".into(), "i64".into())]);

        let mut cc = crate::call::CallControl::new();
        let raw_layout = |size: usize, name: &str, offset: usize| crate::call::StructLayout {
            size,
            align: 8,
            gckind: GcKind::Raw,
            fields: vec![crate::call::StructFieldLayout {
                name: name.into(),
                offset,
                size: 8,
                flag: majit_ir::descr::ArrayFlag::Signed,
                field_type: majit_ir::value::Type::Int,
                rank: None,
            }],
            host: None,
            ll_struct: std::cell::RefCell::new(None),
            ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
        };
        cc.set_struct_layout(outer_id, raw_layout(8, "t", 0));
        cc.set_struct_layout(inner_id, raw_layout(16, "x", 8));

        let registry = crate::translator::rtyper::call_registry::CallRegistry::new(Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        ));
        registry.set_struct_fields(Rc::new(fields));
        registry.set_struct_layouts(cc.struct_layouts_handle());

        let mut graph = LegacyGraph::new("read_raw_ptr_field");
        let s = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "s".into(),
                    ty: ValueType::Int,
                    class_root: Some(outer.into()),
                },
                true,
            )
            .expect("input");
        graph.push_inputarg_var(graph.startblock, s.clone());
        let t = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: s.clone(),
                    field: FieldDescriptor::new("t", Some(outer.into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field t");
        let x = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: t.clone(),
                    field: FieldDescriptor::new("x", Some(inner.into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field x");
        graph.set_return(graph.startblock, Some(x.clone()));

        let cells = crate::translator::rtyper::flowspace_adapter::derive_subject_inputcells(
            &graph,
            Some(registry.bookkeeper()),
        )
        .expect("input cells");
        let SomeValue::Ptr(input_ptr) = &cells[0] else {
            panic!("raw pointer input must be SomePtr, got {:?}", cells[0]);
        };
        let PtrTarget::Struct(st) = &input_ptr.ll_ptrtype.TO else {
            panic!("input pointer target must be a Struct");
        };
        assert_eq!(st._name, outer);
        assert_eq!(st._gckind, GcKind::Raw);
        let LowLevelType::Ptr(t_ptr) = st._flds.get("t").expect("field t") else {
            panic!("S.t must be a pointer, got {:?}", st._flds.get("t"));
        };
        let PtrTarget::Struct(t_st) = &t_ptr.TO else {
            panic!("S.t must point at Struct T");
        };
        assert_eq!(t_st._name, inner);
        assert_eq!(t_st._gckind, GcKind::Raw);
        assert!(matches!(t_st._flds.get("x"), Some(LowLevelType::Signed)));

        let t_shell = raw_ptr_token_annotation(&registry, Some("raw:rawptr::T"))
            .expect("call result raw:T is SomePtr");
        assert!(
            matches!(t_shell, SomeValue::Ptr(_)),
            "raw: token shell is {t_shell:?}"
        );
        assert!(
            residual_return_shell(Some("raw:rawptr::T")).is_some(),
            "the front gate must admit a raw-pointer return token"
        );

        let (value_to_var, _constants) =
            specialize_legacy_graph_with_registry_returning_value_to_var(&graph, &registry)
                .expect("annotate and rtype the two raw field reads");
        let ann_of = |var: &crate::flowspace::model::Variable| {
            value_to_var
                .get(var)
                .expect("typed var")
                .annotation
                .borrow()
                .as_ref()
                .map(|cell| (**cell).clone())
        };
        assert!(
            matches!(ann_of(&t), Some(SomeValue::Ptr(_))),
            "(*s).t annotates as SomePtr, got {:?}",
            ann_of(&t)
        );
        assert!(
            matches!(ann_of(&x), Some(SomeValue::Integer(_))),
            "(*t).x annotates as SomeInteger, got {:?}",
            ann_of(&x)
        );

        for typed in value_to_var.values() {
            if let Some(ll) = typed.concretetype().as_ref() {
                LegacyGraph::set_concretetype_of_inline(typed, crate::model::getkind(ll));
            }
        }
        crate::codewriter::type_state::apply_from_flowspace_variables(&value_to_var);

        let out = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&graph);
        let mut graph = out.graph;
        crate::regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = crate::regalloc::perform_all_register_allocations(&graph);
        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        assert!(
            asm.insns.contains_key("getfield_raw_i/id>i"),
            "getfield_raw_i missing, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        let mut offsets = Vec::new();
        for descr in &asm.descrs {
            if let AssemblerDescr::Ready(ready) = descr
                && let crate::jitcode::BhDescr::Field { offset, .. } = ready.as_ref()
            {
                offsets.push(*offset);
            }
        }
        assert_eq!(
            offsets,
            vec![0, 8],
            "two getfield_raw_i: offsetof(S, t) then offsetof(T, x)"
        );
    }

    /// `fn f(e: &E) -> i64 { e.m() }` with `E` raw and
    /// `impl E { fn m(&self) -> i64 { self.x } }`.
    ///
    /// The call is `simple_call` of `m` with the pointer first, not
    /// `getattr` on the `SomePtr`. Assembly keeps that pointer in the
    /// int register the input occupied.
    #[test]
    fn raw_ptr_method_call_annotates_and_assembles_direct_call() {
        use crate::codewriter::assembler::{Assembler, AssemblerExt};
        use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{CallTarget, LinkArg, OpKind};
        use crate::translator::rtyper::lltypesystem::lltype::GcKind;

        let _lock = anchor_lock();
        let owner = "E";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (owner.to_string(), Some(owner_id)),
            ]));

        let llbc = raw_ptr_method_call_llbc();
        let caller = crate::front::mir::lower_function(&llbc, "f").expect("lower f");
        let mut method = crate::front::mir::lower_function(&llbc, "m").expect("lower m");
        assert!(
            method.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. } if field.name == "x"
                    )
                })
            }),
            "m must read self.x, ops={:?}",
            method
                .blocks
                .iter()
                .flat_map(|block| block.operations.iter())
                .map(|op| &op.kind)
                .collect::<Vec<_>>()
        );

        let input = caller
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match &op.kind {
                OpKind::Input { name, .. } if name == "e" => op.result.clone(),
                _ => None,
            })
            .expect("f's pointer input");
        let mut call_segments = None;
        let mut call_result = None;
        for op in caller
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
        {
            if let OpKind::Call { target, args, .. } = &op.kind {
                let CallTarget::FunctionPath { segments, .. } = target else {
                    panic!("raw-pointer method must be a direct call, got {target:?}");
                };
                assert_eq!(segments, &["E".to_string(), "m".to_string()]);
                assert_eq!(
                    args.first().and_then(LinkArg::as_variable),
                    Some(&input),
                    "the call's first argument is the pointer input"
                );
                call_segments = Some(segments.clone());
                call_result = op.result.clone();
            }
        }
        let call_segments = call_segments.expect("f calls m");
        let call_result = call_result.expect("call result");

        let mut fields = crate::front::StructFieldRegistry::default();
        fields
            .fields
            .insert(owner.to_string(), vec![("x".into(), "i64".into())]);
        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            owner_id,
            crate::call::StructLayout {
                size: 8,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![crate::call::StructFieldLayout {
                    name: "x".into(),
                    offset: 0,
                    size: 8,
                    flag: majit_ir::descr::ArrayFlag::Signed,
                    field_type: majit_ir::value::Type::Int,
                    rank: None,
                }],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        let registry = crate::translator::rtyper::call_registry::CallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        registry.set_struct_fields(std::rc::Rc::new(fields));
        registry.set_struct_layouts(cc.struct_layouts_handle());

        method.func.dont_inline = true;
        let signature = crate::flowspace::argument::Signature::new(vec!["self".into()], None, None);
        let leaf = crate::translator::rtyper::call_registry::CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        ));
        let pygraph = lift_callee_to_pygraph(&method, signature.clone(), &leaf).expect("m lifts");
        registry.register_callee(
            crate::translator::rtyper::call_registry::FunctionPathKey::from_segments(
                call_segments.iter().cloned(),
            ),
            signature,
            pygraph,
        );

        let (value_to_var, _constants) =
            specialize_legacy_graph_with_registry_returning_value_to_var(&caller, &registry)
                .expect("f and m annotate");
        let result_ann = value_to_var.get(&call_result).and_then(|typed| {
            typed
                .annotation
                .borrow()
                .as_ref()
                .map(|cell| (**cell).clone())
        });
        assert!(
            matches!(
                result_ann,
                Some(crate::annotator::model::SomeValue::Integer(_))
            ),
            "e.m() annotates as an int, got {result_ann:?}"
        );
        for typed in value_to_var.values() {
            if let Some(ll) = typed.concretetype().as_ref() {
                LegacyGraph::set_concretetype_of_inline(typed, crate::model::getkind(ll));
            }
        }
        crate::codewriter::type_state::apply_from_flowspace_variables(&value_to_var);

        let out = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&caller);
        let mut graph = out.graph;
        crate::regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = crate::regalloc::perform_all_register_allocations(&graph);
        // `history.getkind` Ptr(raw) → int: `&E` is SomePtr, so the
        // receiver lives in the int bank (`rewrite_op_getfield`
        // `_gckind=='raw'`).
        let int_regs = regallocs
            .get(&crate::flatten::RegKind::Int)
            .expect("int regalloc");
        let input_color = int_regs
            .color_for_variable(&input)
            .expect("raw pointer input is an i register");
        let mut saw_call = false;
        for block in &graph.blocks {
            for op in &block.operations {
                let args_i = match &op.kind {
                    OpKind::CallResidual { args_i, .. }
                    | OpKind::CallElidable { args_i, .. }
                    | OpKind::CallMayForce { args_i, .. }
                    | OpKind::InlineCall { args_i, .. } => args_i,
                    _ => continue,
                };
                let Some(arg0) = args_i.first() else {
                    continue;
                };
                let arg_color = int_regs
                    .color_for_variable(arg0)
                    .expect("call's first int arg is an i register");
                assert_eq!(
                    arg_color, input_color,
                    "call's first arg must be the input i register"
                );
                saw_call = true;
            }
        }
        assert!(
            saw_call,
            "f must assemble through a call op, graph={graph:?}"
        );

        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        assert!(
            asm.insns.keys().any(|key| key.contains("call")),
            "assembled call missing, keys={:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    fn raw_ptr_method_call_llbc() -> majit_charon_reader::Llbc {
        let span = || {
            serde_json::json!({
                "data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }
            })
        };
        let i64_ty = || serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
        let adt = || serde_json::json!({"Adt": {"id": 0, "generics": {"types": []}}});
        let ref_e = || {
            serde_json::json!({
                "Ref": [{"Erased": null}, adt(), "Shared"]
            })
        };
        let meta = |path: Vec<serde_json::Value>, is_local: bool| {
            serde_json::json!({
                "name": path,
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": is_local
            })
        };
        let ident = |name: &str| serde_json::json!({"Ident": [name, 0]});
        let impl_seg = serde_json::json!({
            "Impl": {
                "Ty": {
                    "skip_binder": {"Value": [0, adt()]},
                    "kind": "InherentImplBlock"
                }
            }
        });
        let place = |local: u64, ty: serde_json::Value| serde_json::json!({"kind": {"Local": local}, "ty": ty});
        let field_place = serde_json::json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(1, ref_e()), "Deref"]}, "ty": adt()},
                {"Field": [null, 0]}
            ]},
            "ty": i64_ty()
        });
        let method = serde_json::json!({
            "def_id": 1,
            "item_meta": meta(vec![ident("fixture"), ident("E"), impl_seg, ident("m")], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [ref_e()],
                "output": i64_ty()
            },
            "body": {"Unstructured": {
                "span": span(),
                "locals": {"arg_count": 1, "locals": [
                    {"index": 0, "name": null, "span": span(), "ty": i64_ty()},
                    {"index": 1, "name": "self", "span": span(), "ty": ref_e()}
                ]},
                "body": [{
                    "statements": [{
                        "span": span(),
                        "kind": {"Assign": [
                            place(0, i64_ty()),
                            {"Use": [{"Copy": field_place}, "No"]}
                        ]}
                    }],
                    "terminator": {"span": span(), "kind": "Return"}
                }]
            }}
        });
        let caller = serde_json::json!({
            "def_id": 0,
            "item_meta": meta(vec![ident("fixture"), ident("f")], true),
            "signature": {
                "is_unsafe": false,
                "inputs": [ref_e()],
                "output": i64_ty()
            },
            "body": {"Unstructured": {
                "span": span(),
                "locals": {"arg_count": 1, "locals": [
                    {"index": 0, "name": null, "span": span(), "ty": i64_ty()},
                    {"index": 1, "name": "e", "span": span(), "ty": ref_e()}
                ]},
                "body": [
                    {
                        "statements": [],
                        "terminator": {
                            "span": span(),
                            "kind": {"Call": {
                                "call": {
                                    "func": {"Regular": {
                                        "kind": {"Fun": 1},
                                        "generics": {
                                            "regions": [],
                                            "types": [],
                                            "const_generics": [],
                                            "trait_refs": []
                                        }
                                    }},
                                    "args": [{"Copy": place(1, ref_e())}],
                                    "dest": place(0, i64_ty())
                                },
                                "target": 2,
                                "on_unwind": 1
                            }}
                        }
                    },
                    {
                        "statements": [],
                        "terminator": {"span": span(), "kind": "UnwindResume"}
                    },
                    {
                        "statements": [],
                        "terminator": {"span": span(), "kind": "Return"}
                    }
                ]
            }}
        });
        let type_decl = serde_json::json!({
            "def_id": 0,
            "item_meta": meta(vec![ident("fixture"), ident("E")], true),
            "kind": {"Struct": [{
                "name": "x",
                "ty": i64_ty(),
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
            }]},
            "src": "Normal"
        });
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [type_decl],
                "fun_decls": [caller, method],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("raw pointer method fixture parses")
    }

    #[test]
    fn specialize_legacy_graph_ref_typed_inputarg_resolves_to_gcref() {
        let _lock = anchor_lock();
        // `valuetype_to_someshell(Ref)` lifts to
        // `SomeInstance(classdef=None)` instead of the illegal
        // `SomeObject` placeholder (`model.py` `SomeObject` is
        // abstract).  The rtyper routes through `getinstancerepr(rtyper,
        // None, Gc)` -> `InstanceRepr::new_rootinstance` ->
        // `Ptr(GcStruct(OBJECT))` and `lowleveltype_to_concrete`
        // collapses any GC pointer to `ConcreteType::GcRef`, matching
        // legacy `resolve_types(Ref) -> GcRef`.
        let mut graph = LegacyGraph::new("identity_ref");
        let vars = mint_vars(&mut graph, 2);
        let inputargs = block_inputargs(&vars, &[1]);
        let v1_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v1_var.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        setbinding(&v1_var, ValueType::Ref(None));
        let (value_to_var, _constants) = specialize_legacy_graph(&graph)
            .expect("Ref-typed inputarg must specialize via SomeInstance(classdef=None)");
        assert_eq!(
            kind_of_in(&value_to_var, &vars[1]),
            ConcreteType::GcRef,
            "Ref-typed inputarg must project to GcRef matching legacy"
        );
    }

    #[test]
    fn cutover_repeat_runs_transform_allocate_before_rtype() {
        let _lock = anchor_lock();
        let mut graph = LegacyGraph::new("cutover_repeat_alloc_and_set");
        let vars = mint_vars(&mut graph, 4);
        let fill = vars[1].clone();
        let count = vars[2].clone();
        let result = vars[3].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1, 2]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(result.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![fill.clone(), count.clone()]),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(result.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        setbinding(&fill, ValueType::Int);
        setbinding(&count, ValueType::Int);

        let registry = crate::translator::rtyper::call_registry::CallRegistry::new(Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        ));
        let (flow_graph, _, _, _, _) =
            drive_subject(&graph, &registry, /* do_rtype = */ false)
                .expect("repeat subject must annotate and transform");
        let transformed_ops: Vec<String> = flow_graph
            .borrow()
            .iterblocks()
            .into_iter()
            .flat_map(|block| block.borrow().operations.clone())
            .map(|op| op.opname)
            .collect();
        eprintln!(
            "[cutover_repeat_runs_transform_allocate_before_rtype] before={transformed_ops:?}"
        );
        assert!(
            transformed_ops.iter().any(|name| name == "mul"),
            "cutover must leave mul until per-block specialization, got {transformed_ops:?}"
        );
        assert!(
            transformed_ops.iter().all(|name| name != "alloc_and_set"),
            "cutover must not run graph transforms during annotation, got {transformed_ops:?}"
        );

        let (_, rtyper) = registry.ensure_session().expect("session must remain live");
        let startblock = flow_graph.borrow().startblock.clone();
        let direct_before = startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect::<Vec<_>>();
        eprintln!(
            "[cutover_repeat_runs_transform_allocate_before_rtype] direct_before={direct_before:?}"
        );
        assert!(
            direct_before.iter().any(|name| name == "mul"),
            "direct specialize_block fixture must start with mul: {direct_before:?}"
        );
        rtyper
            .specialize_block(&startblock)
            .expect("direct specialize_block must transform and rtype the start block");
        let direct_after = startblock
            .borrow()
            .operations
            .iter()
            .map(|op| format!("{op:?}"))
            .collect::<Vec<_>>();
        eprintln!(
            "[cutover_repeat_runs_transform_allocate_before_rtype] direct_after={direct_after:?}"
        );
        assert!(
            direct_after.iter().any(|op| op.contains("alloc_and_set")),
            "direct specialize_block must run transform_allocate before rtyping: {direct_after:?}"
        );
        rtyper.mark_already_seen(&startblock);
        rtyper
            .specialize_more_blocks()
            .expect("transformed repeat must rtype");
        let rtyped_output = flow_graph
            .borrow()
            .iterblocks()
            .into_iter()
            .flat_map(|block| block.borrow().operations.clone())
            .map(|op| format!("{op:?}"))
            .collect::<Vec<_>>();
        eprintln!("[cutover_repeat_runs_transform_allocate_before_rtype] after={rtyped_output:?}");
        assert!(
            rtyped_output.iter().any(|op| op.contains("alloc_and_set")),
            "rtyped repeat output must retain alloc_and_set identity: {rtyped_output:?}"
        );
        assert!(
            rtyped_output
                .iter()
                .all(|op| !op.contains("opname: \"mul\"")),
            "rtyped repeat output must contain no mul: {rtyped_output:?}"
        );
    }

    #[test]
    fn cutover_blocked_subject_block_is_excluded_from_per_block_transform() {
        let _lock = anchor_lock();
        // A blocked block is represented by `annotated[key] = None`, just as
        // in the subject drive. It must never enter the per-block transform
        // or rtyper walk; the subject guard aborts before specialization.
        let blocked = crate::flowspace::model::Block::shared(vec![]);
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&blocked), None);
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&blocked), blocked.clone());

        let raw_subject_blocks = vec![blocked];
        let annotated = ann.annotated.borrow();
        let subject_blocks: Vec<_> = raw_subject_blocks
            .into_iter()
            .filter(|block| matches!(annotated.get(&BlockKey::of(block)), Some(Some(_))))
            .collect();
        drop(annotated);
        crate::translator::transform::transform_allocate(&ann, &subject_blocks);
        eprintln!(
            "[cutover_blocked_subject_block_is_excluded_from_per_block_transform] {:?}",
            subject_blocks
                .iter()
                .map(|block| BlockKey::of(block).as_usize())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn lifted_dyn_dispatch_simplifies_only_the_consumed_slot_projection() {
        use crate::annotator::bookkeeper::Bookkeeper;

        // TranslationContext.buildflowgraph runs simplify_graph before the
        // annotator sees a body. A native dyn-call's consumed fnptr projection
        // must not survive as getattr(raw_vtable, "method_run"): the adapter
        // has reconstructed getattr(receiver, "run") + simple_call instead.
        // A separately returned slot remains live and must still be checked.
        for return_slot in [false, true] {
            let registry = CallRegistry::new(Rc::new(Bookkeeper::new()));
            let mut source = LegacyGraph::new("dyn_dispatch_preimage");
            let vars = mint_vars(&mut source, 4);
            setbinding(&vars[0], ValueType::Ref(None));
            setbinding(&vars[1], ValueType::Ref(None));
            let start = source.startblock;
            source.block_mut(start).inputargs = vars[..2].to_vec();
            source.block_mut(start).operations = vec![
                crate::model::SpaceOperation {
                    result: Some(vars[2].clone()),
                    kind: crate::model::OpKind::FieldRead {
                        base: vars[1].clone(),
                        field: crate::model::FieldDescriptor::new(
                            "method_run",
                            Some("{vtable}".into()),
                        ),
                        ty: ValueType::Int,
                        pure: false,
                    },
                },
                crate::model::SpaceOperation {
                    result: Some(vars[3].clone()),
                    kind: crate::model::OpKind::IndirectCall {
                        funcptr: vars[2].clone(),
                        args: vec![vars[0].clone()],
                        graphs: None,
                        family_key: Some(("Handler".into(), "run".into())),
                        result_ty: ValueType::Int,
                    },
                },
            ];
            source.set_return(start, Some(vars[if return_slot { 2 } else { 3 }].clone()));
            let lifted = lift_callee_to_pygraph(
                &source,
                Signature::new(vec!["receiver".into(), "vtable".into()], None, None),
                &registry,
            )
            .expect("dyn dispatch must lift");
            let graph = lifted.graph.borrow();
            crate::flowspace::model::checkgraph(&graph);
            let attr_names: Vec<String> = graph
                .iterblocks()
                .into_iter()
                .flat_map(|block| block.borrow().operations.clone())
                .filter(|op| op.opname == "getattr")
                .filter_map(|op| match &op.args[1] {
                    Hlvalue::Constant(c) => c.value.as_text().map(str::to_owned),
                    _ => None,
                })
                .collect();
            assert!(attr_names.iter().any(|name| name == "run"));
            assert_eq!(
                attr_names.iter().any(|name| name == "method_run"),
                return_slot,
                "only a separately consumed native slot is live"
            );
            assert_eq!(graph.startblock.borrow().inputargs.len(), 2);
            assert!(
                matches!(
                    source.block(start).operations[0].kind,
                    crate::model::OpKind::FieldRead { .. }
                ),
                "the native/legacy dispatch projection must remain intact"
            );
        }
    }

    /// Build a one-call caller graph naming `callee_segments`, with no body
    /// registered for the callee.
    fn caller_graph_calling(callee_segments: &[&str]) -> LegacyGraph {
        let mut graph = LegacyGraph::new("caller");
        let vars = mint_vars(&mut graph, 3);
        let arg = vars[0].clone();
        let result = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[0]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(result.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: callee_segments.iter().map(|s| s.to_string()).collect(),
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![arg]),
                    result_ty: ValueType::Ref(None),
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(result)],
                graph.returnblock,
            )],
            dead: false,
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            dead: false,
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];
        graph
    }

    /// A callee the JIT policy rejects (`policy.py _reject_function`:
    /// "explicitly elidable functions are always opaque") is reached only
    /// through its caller's lift, so the `function_graphs` residualize arm in
    /// [`populate_call_registry_from_call_graphs`] — which walks entries the
    /// MIR loop already lowered — never sees it.  Its stub therefore has to
    /// come from the declaration carrier
    /// (`front::mir::collect_policy_opaque_fn_stubs_from_llbc`), the
    /// `register_external` analog, or the caller fails its own lift with
    /// "not registered in CallRegistry" and every subject downstream of that
    /// caller inherits the failure.
    #[test]
    fn a_lazily_reached_opaque_callee_lifts_its_caller_with_the_declared_result() {
        use crate::annotator::model::SomeValue;
        use crate::codewriter::call::GraphStore;
        use crate::parse::CallPath;

        let callee = ["owner_crate", "opaque_leaf", "read_opaque_slot"];
        let mut graphs = GraphStore::default();
        graphs.insert(
            CallPath::from_segments(["owner_crate", "caller_mod", "caller"]),
            caller_graph_calling(&callee),
        );
        let caller_key = FunctionPathKey::from_segments(["owner_crate", "caller_mod", "caller"]);
        let callee_key = FunctionPathKey::from_segments(callee);

        // Without the declaration stub the caller cannot be annotated past
        // the call — the baseline this regression pins.
        let bare = std::rc::Rc::new(CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        )));
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &bare).unwrap();
        let bare_error = bare
            .lookup(&caller_key)
            .expect("caller registers regardless")
            .lift_error()
            .expect("an unregistered callee must fail the caller's lift");
        assert!(
            bare_error.contains("not registered in CallRegistry"),
            "{bare_error}"
        );
        assert!(bare.lookup(&callee_key).is_none());

        // The collector yields `(segments, signature, FUNC.RESULT token)` for
        // an opaque declaration whatever became of its body; the token is the
        // `*mut PyObject` this callee declares.
        let stubs = vec![(
            callee.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Signature::new(vec!["kind".to_string()], None, None),
            Some(OBJECTPTR_RETURN_TYPE.to_string()),
        )];
        let registry = std::rc::Rc::new(CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        )));
        populate_call_registry_from_call_graphs(&graphs, &stubs, &[], &[], &registry).unwrap();

        let caller = registry.lookup(&caller_key).expect("caller entry");
        assert_eq!(
            caller.lift_error(),
            None,
            "the declared residual must let the caller lift"
        );
        let callee_entry = registry.lookup(&callee_key).expect("callee entry");
        assert!(
            callee_entry.lift_error().is_none(),
            "a declaration stub has no body to fail on"
        );

        // The call stays residual, resolved to the stub's own callable.
        let caller_pygraph = caller
            .function_desc
            .borrow()
            .cache
            .borrow()
            .get(&crate::annotator::description::GraphCacheKey::None)
            .cloned()
            .expect("caller body lifted into the default cache");
        let callable = {
            let graph = caller_pygraph.graph.borrow();
            let start = graph.startblock.borrow();
            let op = start
                .operations
                .iter()
                .find(|op| op.opname == "simple_call")
                .expect("the opaque callee must remain a residual simple_call");
            match &op.args[0] {
                Hlvalue::Constant(c) => match &c.value {
                    ConstValue::HostObject(host) => host.clone(),
                    other => panic!("callable must be a HostObject, got {other:?}"),
                },
                other => panic!("args[0] must be the callable Constant, got {other:?}"),
            }
        };
        assert_eq!(
            callable, callee_entry.host_object,
            "the residual call must bind the stub's own callable"
        );

        // And it carries the callee's declared result, not a Void shell.
        let stub = callee_entry
            .function_desc
            .borrow()
            .cache
            .borrow()
            .get(&crate::annotator::description::GraphCacheKey::None)
            .cloned()
            .expect("the declaration stub is prefilled as the default cache entry");
        let graph = stub.graph.borrow();
        let start = graph.startblock.borrow();
        let link = start.exits[0].borrow();
        let Some(Hlvalue::Variable(ret)) = link.args[0].as_ref() else {
            panic!("stub return arg must be a pre-annotated Variable");
        };
        let annotation = ret.annotation.borrow();
        let annotation = annotation.as_ref().expect("stub return is pre-annotated");
        assert!(
            matches!(&**annotation, SomeValue::Ptr(_)),
            "the object-pointer token must annotate as the declared pointer, got {annotation:?}"
        );
    }

    #[test]
    fn cachedgraph_hit_registers_callee_graph_into_translator_graphs() {
        let _lock = anchor_lock();
        // Keystone linkage: after a real caller -> callee specialize,
        // the lifted callee `FunctionGraph` must live in the session
        // `translator.graphs`.  Upstream reaches that state because
        // `buildgraph -> buildflowgraph` appends every graph
        // (`translator.py`); pyre's lazy-lift prefills
        // `FunctionDesc.cache` via `lift_callee_to_pygraph` +
        // `prefill_default_cache`, bypassing `buildflowgraph`, so the
        // graph would otherwise never enter `translator.graphs`.
        // `FunctionDesc.cachedgraph`'s hit path restores the invariant
        // (description.rs:1058-1082).  This pins that the registration
        // actually fires through the shared-bookkeeper session and that
        // `funcobj.graph` (= the cached PyGraph's `graph`) is the exact
        // `Rc` now resolvable by the flowspace effect analyzers
        // (`canraise::RaiseAnalyzer` resolving `direct_call` through
        // `translator.graphs`, graphanalyze.rs) instead of falling to
        // `top_result()`.
        let mut graph = LegacyGraph::new("call_resolved");
        let vars = mint_vars(&mut graph, 4);
        let inputargs = block_inputargs(&vars, &[1, 2]);
        let v1_var = vars[1].clone();
        let v2_var = vars[2].clone();
        let v3_var = vars[3].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: Some(v3_var.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["foo".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![v1_var.clone(), v2_var.clone()]),
                    result_ty: ValueType::Unknown,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(v3_var.clone())],
                graph.returnblock,
            )],
            dead: false,
            framestate: None,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            dead: false,
            framestate: None,
        };
        graph.blocks = vec![startblock, returnblock];

        let bookkeeper = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let registry = crate::translator::rtyper::call_registry::CallRegistry::new(bookkeeper);

        // Leaf callee `fn foo(x: i64) -> i64 { x }` lifted through a
        // child registry; the entry's `FunctionDesc` (whose
        // `cachedgraph` is hit during the caller's specialize) is owned
        // by the session `registry` via `register_callee`, so its
        // `base.bookkeeper` is the bookkeeper `ensure_session` attaches
        // the annotator to.
        let mut callee_graph = LegacyGraph::new("foo");
        let callee_vars = mint_vars(&mut callee_graph, 13);
        let foo_inputargs = block_inputargs(&callee_vars, &[10, 11]);
        let foo_v10_var = callee_vars[10].clone();
        let foo_v11_var = callee_vars[11].clone();
        let foo_v12_var = callee_vars[12].clone();
        let foo_start = Block {
            id: callee_graph.startblock,
            inputargs: foo_inputargs,
            operations: vec![crate::model::SpaceOperation {
                result: Some(foo_v12_var.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath {
                        segments: vec!["__array_repeat".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![foo_v10_var.clone(), foo_v11_var.clone()]),
                    result_ty: ValueType::Ref(None),
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(foo_v12_var.clone())],
                callee_graph.returnblock,
            )],
            dead: false,
            framestate: None,
        };
        let foo_return = Block {
            id: callee_graph.returnblock,
            inputargs: block_inputargs(&callee_vars, &[12]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            dead: false,
            framestate: None,
        };
        callee_graph.blocks = vec![foo_start, foo_return];
        callee_graph.hints.push("always_inline".to_string());
        callee_graph.func.dont_inline = true;
        let leaf_registry = crate::translator::rtyper::call_registry::CallRegistry::new(
            std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new()),
        );
        setbinding(&foo_v10_var, ValueType::Int);
        setbinding(&foo_v11_var, ValueType::Int);
        let pygraph = lift_callee_to_pygraph(
            &callee_graph,
            crate::flowspace::argument::Signature::new(
                vec!["x".to_string(), "n".to_string()],
                None,
                None,
            ),
            &leaf_registry,
        )
        .expect("leaf callee must lift to PyGraph");
        assert_eq!(
            pygraph.func._always_inline_,
            AlwaysInline::True,
            "legacy always_inline hint must land on graph.func"
        );
        assert!(
            pygraph.func._dont_inline_,
            "legacy dont_inline carrier must land on graph.func"
        );
        assert!(
            pygraph
                .graph
                .borrow()
                .func
                .as_ref()
                .is_some_and(|func| func._always_inline_ == AlwaysInline::True),
            "legacy always_inline hint must land on the registered FunctionGraph.func"
        );
        assert!(
            pygraph
                .graph
                .borrow()
                .func
                .as_ref()
                .is_some_and(|func| func._dont_inline_),
            "legacy dont_inline carrier must land on the registered FunctionGraph.func"
        );
        // Capture the callee's flowspace graph `Rc` before the pygraph
        // is moved into `register_callee`; this is the identity that
        // `cachedgraph` clones into `translator.graphs` and that
        // `funcobj.graph` points at after rtyping.
        let callee_graph_ref = pygraph.graph.clone();
        registry.register_callee(
            crate::translator::rtyper::call_registry::FunctionPathKey::from_segments(["foo"]),
            crate::flowspace::argument::Signature::new(
                vec!["x".to_string(), "n".to_string()],
                None,
                None,
            ),
            pygraph,
        );

        setbinding(&v1_var, ValueType::Int);
        setbinding(&v2_var, ValueType::Int);
        specialize_legacy_graph_with_registry_returning_value_to_var(&graph, &registry)
            .expect("cache pre-fill must let the leaf Call resolve end-to-end");

        let callee_ops: Vec<String> = callee_graph_ref
            .borrow()
            .iterblocks()
            .into_iter()
            .flat_map(|block| block.borrow().operations.clone())
            .map(|op| format!("{op:?}"))
            .collect();
        eprintln!("[cachedgraph_hit_registers_callee_graph_into_translator_graphs] {callee_ops:?}");
        assert!(
            callee_ops.iter().any(|op| op.contains("alloc_and_set")),
            "callee-lifted repeat must transform to alloc_and_set: {callee_ops:?}"
        );
        assert!(
            callee_ops.iter().all(|op| !op.contains("opname: \"mul\"")),
            "callee-lifted repeat must contain no mul: {callee_ops:?}"
        );

        // `ensure_session` ran inside the specialize and wired the
        // annotator backlink onto the registry's shared bookkeeper
        // (`set_annotator`), so the diagnostic accessor must now
        // upgrade.
        let annotator = registry
            .bookkeeper()
            .try_annotator()
            .expect("specialize must have attached the session annotator");
        let graphs = annotator.translator.graphs.borrow();
        assert!(
            graphs
                .iter()
                .any(|g| std::rc::Rc::ptr_eq(g, &callee_graph_ref)),
            "the lifted callee graph must be registered into translator.graphs by the \
             cachedgraph hit path, so RaiseAnalyzer can resolve funcobj.graph through \
             translator.graphs instead of falling to top_result()"
        );
    }

    #[test]
    fn failed_subject_restores_previously_annotated_callee_that_lost_bindings() {
        let _lock = anchor_lock();
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::flowspace::model::{BlockKey, GraphKey};
        use crate::translator::rtyper::call_registry::{CallRegistry, FunctionPathKey};

        let mut source = LegacyGraph::new("shared_int_value");
        let vars = mint_vars(&mut source, 2);
        let value = vars[1].clone();
        source.blocks = vec![
            Block {
                id: source.startblock,
                inputargs: block_inputargs(&vars, &[1]),
                operations: vec![],
                exitswitch: None,
                exits: vec![link_to_returnblock(
                    vec![LinkArg::Value(value.clone())],
                    source.returnblock,
                )],
                framestate: None,
                dead: false,
            },
            Block {
                id: source.returnblock,
                inputargs: block_inputargs(&vars, &[1]),
                operations: vec![],
                exitswitch: None,
                exits: vec![],
                framestate: None,
                dead: false,
            },
        ];
        setbinding(&value, ValueType::Int);

        let registry = CallRegistry::new(Rc::new(Bookkeeper::new()));
        // `mint_vars` allocates unnamed value vars, so the startblock's single
        // inputarg has no source name and takes the positional `arg0`.
        let signature =
            crate::flowspace::argument::Signature::new(vec!["arg0".to_string()], None, None);
        let lifted = lift_callee_to_pygraph(&source, signature.clone(), &registry)
            .expect("shared callee must lift");
        let stale = lifted.graph.clone();
        let entry_key = FunctionPathKey::from_segments(["shared_int_value"]);
        registry.register_callee(entry_key.clone(), signature, lifted);
        let (annotator, _) = registry.ensure_session().expect("session must initialize");
        let stale_blocks = stale.borrow().iterblocks();
        for block in &stale_blocks {
            let bkey = BlockKey::of(block);
            annotator
                .annotated
                .borrow_mut()
                .insert(bkey.clone(), Some(stale.clone()));
            annotator
                .all_blocks
                .borrow_mut()
                .insert(bkey, block.clone());
        }
        // Model raise_if_subject_blocked's destructive reflow rollback:
        // one block that was healthy at subject entry is removed and all
        // definitions lose their annotations. No graph became fixed, so
        // the transaction log must restore the touched block without scanning
        // every annotated block in the session.
        let damaged = stale_blocks
            .first()
            .expect("identity graph has a start block")
            .clone();
        let damaged_key = BlockKey::of(&damaged);
        let inputcells: Vec<Option<crate::annotator::model::SomeValue>> = damaged
            .borrow()
            .inputargs
            .iter()
            .map(|value| match value {
                Hlvalue::Variable(var) => var
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|annotation| (**annotation).clone()),
                _ => None,
            })
            .collect();
        let scope = annotator.enter_added_blocks_scope();
        annotator.addpendingblock(&stale, &damaged, &inputcells);
        annotator.annotated.borrow_mut().shift_remove(&damaged_key);
        annotator.all_blocks.borrow_mut().shift_remove(&damaged_key);
        {
            let block = damaged.borrow();
            for value in block
                .inputargs
                .iter()
                .chain(block.operations.iter().map(|op| &op.result))
            {
                if let Hlvalue::Variable(var) = value {
                    *var.annotation.borrow_mut() = None;
                }
            }
        }
        drop(scope);

        assert!(
            registry.find_entry_with_cached_graph(&stale).is_some(),
            "an annotation-only rollback must retain the original FunctionDesc.cache graph"
        );
        assert!(
            stale_blocks.iter().all(|block| {
                let bkey = BlockKey::of(block);
                matches!(
                    annotator.annotated.borrow().get(&bkey),
                    Some(Some(graph)) if Rc::ptr_eq(graph, &stale)
                ) && matches!(
                    annotator.all_blocks.borrow().get(&bkey),
                    Some(current) if Rc::ptr_eq(current, block)
                )
            }),
            "all pre-existing annotator rows must be restored"
        );
        assert!(
            damaged.borrow().inputargs.iter().all(|value| match value {
                Hlvalue::Variable(var) => var.annotation.borrow().is_some(),
                _ => true,
            }),
            "the pre-subject Variable annotations must be restored"
        );
        assert!(
            !annotator
                .blocked_graphs
                .borrow()
                .contains_key(&GraphKey::of(&stale)),
            "discarded graph must not remain in blocked_graphs"
        );
    }

    // There is no program-wide annotator pre-pass to test: per-session
    // annotator construction inside
    // `specialize_legacy_graph_with_registry_returning_value_to_var`
    // is the only production lifecycle.

    // ─── unsafe-fn stub helpers ───
    #[test]
    fn default_someshell_for_lltype_integer_family_yields_someinteger_no_const() {
        use crate::annotator::model::SomeValue;
        for ll in [
            LowLevelType::Signed,
            LowLevelType::Unsigned,
            LowLevelType::SignedLongLong,
            LowLevelType::UnsignedLongLong,
        ] {
            let s = default_someshell_for_lltype(&ll)
                .unwrap_or_else(|| panic!("{ll:?} must project to a SomeValue"));
            match s {
                SomeValue::Integer(ref si) => assert!(
                    si.base.const_box.is_none(),
                    "{ll:?}: SomeInteger must carry no const_box"
                ),
                other => panic!("{ll:?}: expected SomeInteger, got {other:?}"),
            }
        }
    }

    #[test]
    fn default_someshell_for_lltype_bool_yields_somebool_no_const() {
        use crate::annotator::model::SomeValue;
        let s = default_someshell_for_lltype(&LowLevelType::Bool)
            .expect("Bool must project to SomeBool");
        match s {
            SomeValue::Bool(b) => assert!(
                b.base.const_box.is_none(),
                "SomeBool must carry no const_box (ExtRegistryEntry parity)"
            ),
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn default_someshell_for_lltype_void_yields_somenone() {
        use crate::annotator::model::SomeValue;
        let s = default_someshell_for_lltype(&LowLevelType::Void)
            .expect("Void must project to SomeNone");
        assert!(matches!(s, SomeValue::None_(_)), "got {s:?}");
    }

    #[test]
    fn default_someshell_for_lltype_float_family_yields_somefloat_no_const() {
        use crate::annotator::model::SomeValue;
        for ll in [
            LowLevelType::Float,
            LowLevelType::SingleFloat,
            LowLevelType::LongFloat,
        ] {
            let s = default_someshell_for_lltype(&ll)
                .unwrap_or_else(|| panic!("{ll:?} must project to a SomeValue"));
            let const_present = match &s {
                SomeValue::Float(f) => f.base.const_box.is_some(),
                SomeValue::SingleFloat(f) => f.base.const_box.is_some(),
                SomeValue::LongFloat(f) => f.base.const_box.is_some(),
                other => panic!("{ll:?}: expected float family, got {other:?}"),
            };
            assert!(!const_present, "{ll:?}: float shell must have no const_box");
        }
    }

    #[test]
    fn default_someshell_for_lltype_address_yields_someaddress() {
        use crate::annotator::model::SomeValue;
        // `Address` maps to the untyped `SomeAddress` shell (llmemory.py
        // `SomeAddress`) — the projection a raw `*const T` / `*mut T` result
        // takes.
        let s = default_someshell_for_lltype(&LowLevelType::Address)
            .expect("Address must project to SomeAddress");
        assert!(matches!(s, SomeValue::Address(_)), "got {s:?}");
    }

    #[test]
    fn exception_materializer_stub_uses_semantic_instance_result() {
        use crate::annotator::model::SomeValue;

        for name in [
            "pyerror_to_exc_object",
            "pyerror_type_error_to_exc_object",
            "pyerror_zero_division_to_exc_object",
            "pyerror_value_error_to_exc_object",
            "pyerror_index_error_to_exc_object",
        ] {
            let key = FunctionPathKey::from_segments(["pyre_interpreter", "error", name]);
            let shell = residual_stub_result_shell(&key, Some(OBJECTPTR_RETURN_TYPE))
                .expect("exception materializer must have a result shell");
            let SomeValue::Instance(instance) = shell else {
                panic!("{name}: raw pointer ABI must not leak into annotation")
            };
            assert!(instance.classdef.is_none());
            assert!(!instance.can_be_none);
        }

        let ordinary_key = FunctionPathKey::from_segments(["other", "pyerror_to_exc_object"]);
        let ordinary = residual_stub_result_shell(&ordinary_key, Some(OBJECTPTR_RETURN_TYPE))
            .expect("ordinary object-pointer residual must have a result shell");
        assert!(
            matches!(ordinary, SomeValue::Ptr(_)),
            "ordinary object-pointer residuals keep their typed Ptr shell"
        );
    }

    #[test]
    fn a_str_pointer_residual_declares_a_string_result() {
        use crate::annotator::model::SomeValue;
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = CallRegistry::new(ann.bookkeeper.clone());
        let mut graph = LegacyGraph::new("formatd");
        graph.return_type = Some("ref".into());
        assert_eq!(declared_return_annotation(&registry, &graph), None);
        graph.return_is_str = true;
        let declared = declared_return_annotation(&registry, &graph)
            .expect("a STR pointer result declares its annotation");
        let SomeValue::String(s) = declared else {
            panic!("a STR pointer result is SomeString, got {declared:?}")
        };
        assert!(!s.inner.can_be_none);
        let ft = declared_funcptr_type_from_legacy(&graph)
            .expect("a no-arg STR-returning graph has a declared fn type");
        assert_eq!(
            ft.result,
            crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone(),
            "the declared fn type returns what the SomeString result rtypes to"
        );
    }

    #[test]
    fn declared_return_annotation_raw_class_root_is_someptr() {
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::lltypesystem::lltype::{GcKind, PtrTarget};

        let owner = "rawret::Scope";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _ids =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (owner.to_string(), Some(owner_id)),
            ]));

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = CallRegistry::new(ann.bookkeeper.clone());
        let mut fields = crate::front::StructFieldRegistry::default();
        fields.fields.insert(
            owner.to_string(),
            vec![("save_point".to_string(), "usize".to_string())],
        );
        fields.raw_word_owners.insert(owner.to_string());
        registry.set_struct_fields(Rc::new(fields));

        let mut cc = crate::call::CallControl::new();
        cc.set_struct_layout(
            owner_id,
            crate::call::StructLayout {
                size: 8,
                align: 8,
                gckind: GcKind::Raw,
                fields: vec![crate::call::StructFieldLayout {
                    name: "save_point".into(),
                    offset: 0,
                    size: 8,
                    flag: majit_ir::descr::ArrayFlag::Unsigned,
                    field_type: majit_ir::value::Type::Int,
                    rank: None,
                }],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        registry.set_struct_layouts(cc.struct_layouts_handle());

        let mut graph = LegacyGraph::new("push_roots");
        graph.return_type = Some("ref".into());
        graph.return_class_root = Some(owner.into());
        let declared = declared_return_annotation(&registry, &graph)
            .expect("a by-value raw ADT result declares its annotation");
        let SomeValue::Ptr(ptr) = declared else {
            panic!("a raw class-root result is SomePtr, got {declared:?}");
        };
        let PtrTarget::Struct(st) = &ptr.ll_ptrtype.TO else {
            panic!("raw class-root SomePtr must point at a Struct");
        };
        assert_eq!(st._name, owner);
        assert_eq!(st._gckind, GcKind::Raw);
    }

    #[test]
    fn container_residuals_declare_a_list_result() {
        use crate::annotator::model::SomeValue;
        use crate::front::StructFieldRegistry;
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = CallRegistry::new(ann.bookkeeper.clone());
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Utf8LocElem".to_string(),
            Vec::<crate::front::semantic::FieldRow>::new(),
        );
        registry
            .bookkeeper()
            .set_struct_fields(std::rc::Rc::new(reg));
        let mut graph = LegacyGraph::new("w_str_compute_index_storage");
        graph.return_type = Some("ref".into());
        assert_eq!(declared_return_annotation(&registry, &graph), None);
        graph.return_container_root = Some("*mut Vec<Utf8LocElem>".into());
        let declared = declared_return_annotation(&registry, &graph)
            .expect("a Vec pointer result declares its annotation");
        let SomeValue::List(list) = declared else {
            panic!("a Vec pointer result is SomeList, got {declared:?}")
        };
        let SomeValue::Instance(item) = list.listdef.s_value() else {
            panic!(
                "Vec item must be the element instance, got {:?}",
                list.listdef.s_value()
            )
        };
        let classdef = item
            .classdef
            .expect("Utf8LocElem is a named struct, not a classdef-less shell");
        assert!(
            classdef.borrow().name.contains("Utf8LocElem"),
            "list item class is {}",
            classdef.borrow().name
        );

        let mut graph = LegacyGraph::new("bytes_like_data");
        graph.return_type = Some("ref".into());
        assert_eq!(declared_return_annotation(&registry, &graph), None);
        graph.return_container_root = Some("&[u8]".into());
        let declared = declared_return_annotation(&registry, &graph)
            .expect("a &[u8] residual declares a list");
        let SomeValue::List(list) = declared else {
            panic!("&[u8] residual is SomeList, got {declared:?}")
        };
        assert!(
            matches!(list.listdef.s_value(), SomeValue::Integer(_)),
            "u8 items are integers, got {:?}",
            list.listdef.s_value()
        );

        let mut graph = LegacyGraph::new("encode_ascii");
        graph.return_type = Some("ref".into());
        graph.return_container_root = Some("Vec<u8>".into());
        let declared = declared_return_annotation(&registry, &graph)
            .expect("a Vec<u8> residual declares a list");
        let SomeValue::List(list) = declared else {
            panic!("Vec<u8> residual is SomeList, got {declared:?}")
        };
        assert!(
            matches!(list.listdef.s_value(), SomeValue::Integer(_)),
            "u8 items are integers, got {:?}",
            list.listdef.s_value()
        );
    }

    #[test]
    fn one_word_vec_residual_on_the_i64_token_is_rustvec() {
        use crate::annotator::model::SomeValue;
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let registry = CallRegistry::new(ann.bookkeeper.clone());
        let mut graph = LegacyGraph::new("ll_vec_alloc_and_set_nonnull_r_trampoline");
        graph.return_type = Some("i64".into());
        assert_eq!(declared_return_annotation(&registry, &graph), None);
        graph.return_container_root = Some("Vec<*mut u8>".into());
        let declared = declared_return_annotation(&registry, &graph)
            .expect("a one-word Vec residual declares its annotation");
        let SomeValue::RustVec(vec) = declared else {
            panic!("a one-word Vec residual is SomeRustVec, got {declared:?}")
        };
        assert_eq!(
            vec.item_kind(),
            Some(majit_ir::rvec::VecItemKind::Ref),
            "Vec<*mut u8> items are managed references"
        );
    }

    #[test]
    fn build_stub_pygraph_carries_signature_and_links_to_returnblock() {
        use crate::annotator::model::SomeValue;
        let sig = Signature::new(vec!["obj".to_string()], None, None);
        let pygraph =
            build_stub_pygraph_for_lltype("is_none".to_string(), sig.clone(), LowLevelType::Bool)
                .expect("Bool return must produce a stub pygraph");
        assert_eq!(*pygraph.signature.borrow(), sig);
        let graph = pygraph.graph.borrow();
        assert_eq!(graph.name, "is_none");
        let start = graph.startblock.borrow();
        assert_eq!(
            start.inputargs.len(),
            1,
            "argname must map to a single inputarg"
        );
        assert_eq!(
            start.exits.len(),
            1,
            "stub must Link directly to returnblock"
        );
        let link = start.exits[0].borrow();
        assert_eq!(
            link.args.len(),
            1,
            "stub Link carries exactly the pre-annotated return Variable"
        );
        // The link's target must be the graph's returnblock.
        let link_target = link.target.as_ref().expect("Link must target a block");
        assert!(
            std::rc::Rc::ptr_eq(link_target, &graph.returnblock),
            "stub Link must target the graph's returnblock"
        );
        // The return arg must be a Variable carrying a const-free
        // SomeBool annotation (ExtRegistryEntry parity).
        let ret_arg = link.args[0]
            .as_ref()
            .expect("stub return arg must be Some(Hlvalue::Variable)");
        match ret_arg {
            Hlvalue::Variable(v) => {
                let ann = v.annotation.borrow();
                let s = ann.as_ref().expect("return Variable must be pre-annotated");
                match &**s {
                    SomeValue::Bool(b) => assert!(
                        b.base.const_box.is_none(),
                        "stub return SomeBool must not leak a const_box"
                    ),
                    other => panic!("expected SomeBool annotation, got {other:?}"),
                }
            }
            other => panic!("stub return must be Hlvalue::Variable, got {other:?}"),
        }
    }

    #[test]
    fn register_unsafe_fn_stubs_registers_each_spec_and_skips_compound_returns() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let registry = CallRegistry::new(bk);
        let specs = vec![
            (
                vec!["pyobject".to_string(), "is_none".to_string()],
                Signature::new(vec!["obj".to_string()], None, None),
                Some("bool".to_string()),
            ),
            (
                vec!["pyobject".to_string(), "is_int".to_string()],
                Signature::new(vec!["obj".to_string()], None, None),
                Some("bool".to_string()),
            ),
            (
                vec!["majit_gc".to_string(), "shadow_push".to_string()],
                Signature::new(vec!["obj".to_string()], None, None),
                Some("u64".to_string()),
            ),
            // A token `residual_return_shell` declines — 128-bit has no
            // `getkind` at the codewriter boundary — so
            // register_unsafe_fn_stubs must skip.
            (
                vec!["pyobject".to_string(), "unsupported".to_string()],
                Signature::new(vec!["x".to_string()], None, None),
                Some("i128".to_string()),
            ),
        ];
        register_unsafe_fn_stubs(&registry, &specs);
        assert_eq!(
            registry.len(),
            3,
            "undecodable-return spec must be skipped, others registered"
        );
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyobject".to_string(),
                    "is_none".to_string()
                ]))
                .is_some()
        );
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyobject".to_string(),
                    "is_int".to_string()
                ]))
                .is_some()
        );
    }

    #[test]
    fn register_unsafe_fn_stubs_keeps_allocate_ctors() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;
        let registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        let mut specs = vec![
            (
                vec!["pyobject".to_string(), "is_none".to_string()],
                Signature::new(vec!["obj".to_string()], None, None),
                Some("bool".to_string()),
            ),
            (
                vec![
                    "functional".to_string(),
                    "W_Range".to_string(),
                    "allocate_stable".to_string(),
                ],
                Signature::new(vec!["payload".to_string()], None, None),
                Some(OBJECTPTR_RETURN_TYPE.to_string()),
            ),
            (
                vec![
                    "nestedscope".to_string(),
                    "Cell".to_string(),
                    "allocate".to_string(),
                ],
                Signature::new(vec!["payload".to_string()], None, None),
                Some(OBJECTPTR_RETURN_TYPE.to_string()),
            ),
        ];
        for segments in [
            vec!["core", "ptr", "copy_nonoverlapping"],
            vec!["std", "ptr", "copy_nonoverlapping"],
            vec!["copy_nonoverlapping"],
        ] {
            specs.push((
                segments.iter().map(|s| (*s).to_string()).collect(),
                Signature::new(
                    vec!["dst".to_string(), "src".to_string(), "count".to_string()],
                    None,
                    None,
                ),
                None,
            ));
        }
        register_unsafe_fn_stubs(&registry, &specs);
        assert_eq!(
            registry.len(),
            3,
            "allocate ctor stubs stay registered; copy_nonoverlapping stays out"
        );
        for segments in [
            &["pyobject", "is_none"][..],
            &["functional", "W_Range", "allocate_stable"][..],
            &["nestedscope", "Cell", "allocate"][..],
        ] {
            let entry = registry
                .lookup(&FunctionPathKey::from_segments(segments.iter().copied()))
                .unwrap_or_else(|| panic!("{segments:?} must be registered"));
            assert!(
                !entry.function_desc.borrow().cache.borrow().is_empty(),
                "{segments:?}: annotator stub graph required"
            );
        }
        for segments in [
            &["core", "ptr", "copy_nonoverlapping"][..],
            &["std", "ptr", "copy_nonoverlapping"][..],
            &["copy_nonoverlapping"][..],
        ] {
            assert!(
                registry
                    .lookup(&FunctionPathKey::from_segments(segments.iter().copied()))
                    .is_none(),
                "unsafe stub still registered for {segments:?}"
            );
        }
    }

    #[test]
    fn runtime_residual_externals_register_in_the_call_registry() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;

        let registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_runtime_residual_externals(&registry);
        for segments in [
            ["majit_ir", "eval_breaker_word", "set_gc_interp"].as_slice(),
            ["majit_gc", "shadow_stack", "slot_from_cached"].as_slice(),
            ["majit_gc", "shadow_stack", "cache_root_stack_slot"].as_slice(),
            ["majit_gc", "shadow_stack", "shadow_stack_slot"].as_slice(),
        ] {
            assert!(
                registry
                    .lookup(&FunctionPathKey::from_segments(segments.iter().copied()))
                    .is_some(),
                "runtime residual {segments:?} must register"
            );
        }
        assert!(
            !is_no_effect_external(
                &["majit_ir", "eval_breaker_word", "set_gc_interp"]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect::<Vec<_>>()
            ),
            "set_gc_interp arms a breaker bit; jtransform must residualize it"
        );
    }

    #[test]
    fn foreign_stdlib_externals_registers_cell_get_and_atomic_scalar_rows() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;

        let registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_foreign_stdlib_externals(&registry);
        for (segments, argnames) in [
            (["cell", "Cell", "get"].as_slice(), ["self"].as_slice()),
            (
                ["sync", "atomic", "AtomicUsize", "new"].as_slice(),
                ["v"].as_slice(),
            ),
            (
                ["sync", "atomic", "AtomicPtr", "swap"].as_slice(),
                ["self", "val", "order"].as_slice(),
            ),
        ] {
            let entry = registry
                .lookup(&FunctionPathKey::from_segments(segments.iter().copied()))
                .unwrap_or_else(|| panic!("{segments:?} must be registered"));
            assert_eq!(
                entry.function_desc.borrow().signature.argnames,
                argnames
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect::<Vec<_>>(),
                "{segments:?}"
            );
            assert!(
                !entry.function_desc.borrow().cache.borrow().is_empty(),
                "{segments:?}: annotator stub graph required"
            );
        }
    }

    #[test]
    fn foreign_stdlib_externals_does_not_register_refcell_get_lookalike() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;

        let registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_foreign_stdlib_externals(&registry);
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments(["cell", "RefCell", "get"]))
                .is_none(),
            "RefCell::get is not Cell::get"
        );
    }

    fn saturating_mul_call_graph(
        result_ty: ValueType,
    ) -> (LegacyGraph, crate::flowspace::model::Variable) {
        let mut graph = LegacyGraph::new("saturating_mul_call");
        let vars = mint_vars(&mut graph, 4);
        let result = vars[3].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1, 2]),
            operations: vec![crate::model::SpaceOperation {
                result: Some(result.clone()),
                kind: crate::model::OpKind::Call {
                    target: crate::model::CallTarget::function_path([
                        "core",
                        "num",
                        "<Impl>",
                        "saturating_mul",
                    ]),
                    args: crate::model::call_args(vec![vars[1].clone(), vars[2].clone()]),
                    result_ty: result_ty.clone(),
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(result.clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: vec![result.clone()],
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        setbinding(&vars[1], result_ty.clone());
        setbinding(&vars[2], result_ty);
        (graph, result)
    }

    fn saturating_mul_result_is_unsigned(
        value_to_var: &LegacyToTyped,
        result: &crate::flowspace::model::Variable,
    ) -> bool {
        match value_to_var.get(result).and_then(|typed| {
            typed
                .annotation
                .borrow()
                .as_ref()
                .map(|cell| (**cell).clone())
        }) {
            Some(crate::annotator::model::SomeValue::Integer(i)) => i.unsigned,
            other => panic!("saturating_mul result must be SomeInteger, got {other:?}"),
        }
    }

    /// `i64::saturating_mul` must not reuse the `usize` residual
    /// (`specialize.py specialize_argtype` / `SomeInteger(knowntype=...)`
    /// / `rarithmetic.r_uint`).
    #[test]
    fn saturating_mul_signed_result_annotates_signed_usize_stays_unsigned() {
        let _lock = anchor_lock();
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;

        let keys = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_foreign_stdlib_externals(&keys);

        let shared = FunctionPathKey::from_segments(["core", "num", "<Impl>", "saturating_mul"]);
        let signed_key =
            FunctionPathKey::from_segments(["core", "num", "<Impl>", "int", "saturating_mul"]);
        let unsigned_key =
            FunctionPathKey::from_segments(["core", "num", "<Impl>", "r_uint", "saturating_mul"]);
        assert!(
            keys.lookup(&shared).is_none(),
            "untyped <Impl> saturating_mul must not mint a shared residual"
        );
        let signed_host = keys
            .lookup(&signed_key)
            .expect("int saturating_mul stub")
            .host_object
            .identity_id();
        let unsigned_host = keys
            .lookup(&unsigned_key)
            .expect("r_uint saturating_mul stub")
            .host_object
            .identity_id();
        assert_ne!(
            signed_host, unsigned_host,
            "i64 and usize saturating_mul must be distinct FunctionDescs"
        );

        let signed_registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_foreign_stdlib_externals(&signed_registry);
        let (signed_graph, signed_result) = saturating_mul_call_graph(ValueType::Int);
        let (signed_map, _) = specialize_legacy_graph_with_registry_returning_value_to_var(
            &signed_graph,
            &signed_registry,
        )
        .expect("i64 saturating_mul annotates");
        assert!(
            !saturating_mul_result_is_unsigned(&signed_map, &signed_result),
            "i64::saturating_mul result annotates signed"
        );

        let unsigned_registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_foreign_stdlib_externals(&unsigned_registry);
        let (unsigned_graph, unsigned_result) = saturating_mul_call_graph(ValueType::Unsigned);
        let (unsigned_map, _) = specialize_legacy_graph_with_registry_returning_value_to_var(
            &unsigned_graph,
            &unsigned_registry,
        )
        .expect("usize saturating_mul annotates");
        assert!(
            saturating_mul_result_is_unsigned(&unsigned_map, &unsigned_result),
            "usize::saturating_mul result stays unsigned"
        );
    }

    /// `rlib/jit.py isconstant`: `@oopspec` + `@specialize.call_location()`
    /// with body `return NonConstant(False)`. The Rust macro inserts
    /// `icf_identity_token` (empty `nomem` asm); Charon leaves that
    /// Foreign. Registering it as a zero-arg Void external
    /// (`register_external(..., [], None)`) lets the annotator see the
    /// same SomeBool-not-constant body upstream annotates, and
    /// jtransform drops the token the way `rewrite_op_getfield`
    /// returns on `RESULT is lltype.Void`.
    fn isconstant_oopspec_body_graph() -> LegacyGraph {
        let mut graph = LegacyGraph::new("isconstant");
        let value = graph.alloc_value_var();
        setbinding(&value, ValueType::Int);
        graph.block_mut(graph.startblock).inputargs = vec![value];
        graph.block_mut(graph.startblock).operations = vec![crate::model::SpaceOperation {
            result: None,
            kind: crate::model::OpKind::Call {
                target: crate::model::CallTarget::function_path([
                    "majit_ir",
                    "icf",
                    "icf_identity_token",
                ]),
                args: vec![],
                result_ty: ValueType::Void,
            },
        }];
        let false_var = graph
            .push_op_var(
                graph.startblock,
                crate::model::OpKind::ConstBool(false),
                true,
            )
            .unwrap();
        setbinding(&false_var, ValueType::Bool);
        let result = graph
            .push_op_var(
                graph.startblock,
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::function_path([
                        "majit_rlib",
                        "nonconst",
                        "non_constant",
                    ]),
                    args: crate::model::call_args(vec![false_var]),
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        setbinding(&result, ValueType::Bool);
        graph.set_return(graph.startblock, Some(result));
        graph
    }

    #[test]
    fn isconstant_oopspec_body_annotates_somebool_and_drops_icf_token() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::codewriter::call::GraphStore;
        use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{CallTarget, OpKind};
        use crate::parse::CallPath;

        let token = ["majit_ir", "icf", "icf_identity_token"];
        let isconstant = ["majit_rlib", "jit", "isconstant"];
        let translator = crate::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let source = isconstant_oopspec_body_graph();
        let mut graphs = GraphStore::default();
        graphs.insert(CallPath::from_segments(isconstant), source.clone());
        let registry = std::rc::Rc::new(CallRegistry::new(ann.bookkeeper.clone()));
        populate_call_registry_from_call_graphs(&graphs, &[], &[], &[], &registry).unwrap();

        let token_entry = registry
            .lookup(&FunctionPathKey::from_segments(token))
            .expect("icf_identity_token must be a registered no-effect external");
        assert!(
            token_entry
                .function_desc
                .borrow()
                .signature
                .argnames
                .is_empty(),
            "token signature is () -> None"
        );

        let isconstant_key = FunctionPathKey::from_segments(isconstant);
        let entry = registry.lookup(&isconstant_key).expect("isconstant entry");
        assert_eq!(
            entry.lift_error(),
            None,
            "oopspec body must lift once the token is an external"
        );
        let pygraph = entry
            .function_desc
            .borrow()
            .cachedgraph(
                crate::annotator::description::GraphCacheKey::None,
                None,
                None,
            )
            .expect("lifted isconstant body");
        ann.translator
            .graphs
            .borrow_mut()
            .push(std::rc::Rc::clone(&pygraph.graph));
        let annotated = ann
            .build_graph_types(
                &pygraph.graph,
                &[Some(SomeValue::Integer(SomeInteger::default()))],
                true,
            )
            .expect("isconstant body annotates");
        match annotated {
            Some(SomeValue::Bool(b)) => assert!(
                b.base.const_box.is_none(),
                "NonConstant(False) is a SomeBool that is not constant"
            ),
            other => panic!("expected SomeBool, got {other:?}"),
        }

        let transformed = Transformer::new(&GraphTransformConfig::default()).transform(&source);
        let ops = &transformed.graph.block(source.startblock).operations;
        assert!(
            ops.iter().all(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => segments.as_slice() != token,
                _ => !format!("{:?}", op.kind).contains("icf_identity_token"),
            }),
            "jtransform must drop the ICF token; ops={ops:?}"
        );
        assert!(
            ops.iter().all(|op| !matches!(
                &op.kind,
                OpKind::CallResidual {
                    result_kind: 'v',
                    args_i,
                    args_r,
                    args_f,
                    ..
                } if args_i.is_empty() && args_r.is_empty() && args_f.is_empty()
            )),
            "token must not residualize to an empty-asm call; ops={ops:?}"
        );
    }

    #[test]
    fn deref_write_external_has_two_parameter_void_stub() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;

        let registry = CallRegistry::new(std::rc::Rc::new(Bookkeeper::new()));
        register_deref_write_external(&registry);
        let entry = registry
            .lookup(&FunctionPathKey::from_segments(["__deref_write"]))
            .expect("opaque deref-write marker must be registered");
        assert_eq!(
            entry.function_desc.borrow().signature.argnames,
            vec!["ptr".to_string(), "value".to_string()]
        );
        assert!(
            !entry.function_desc.borrow().cache.borrow().is_empty(),
            "the annotator needs a cached stub graph for return inference"
        );
        assert_eq!(registry.len(), 1);
    }

    /// Pin the layering invariant
    /// that the synthetic stub-pygraph is annotator-only.  The
    /// `register_unsafe_fn_stubs` helper writes the stub into the
    /// `CallRegistry` cache but does NOT mutate any
    /// `CallControl::function_graphs` map; the codewriter's BFS walks
    /// `function_graphs.keys()` and resolves each call op's target via
    /// `target_to_path_and_graph` which requires the target to be
    /// present in `function_graphs`.  This test mirrors the
    /// `codewriter.rs:192-195` production call shape (registry +
    /// `callcontrol.unsafe_fn_stubs`) and asserts the path is reachable
    /// via the registry but not via `CallControl::function_graphs`.
    #[test]
    fn register_unsafe_fn_stubs_does_not_populate_callcontrol_function_graphs() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::codewriter::call::CallControl;
        use crate::translator::rtyper::call_registry::CallRegistry;
        let mut callcontrol = CallControl::new();
        callcontrol.unsafe_fn_stubs = vec![(
            vec!["pyobject".to_string(), "is_none".to_string()],
            Signature::new(vec!["obj".to_string()], None, None),
            Some("bool".to_string()),
        )];
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let registry = CallRegistry::new(bk);
        register_unsafe_fn_stubs(&registry, &callcontrol.unsafe_fn_stubs);
        // Registry side: the stub is present for `cachedgraph` lookups.
        let key = FunctionPathKey::from_segments(["pyobject".to_string(), "is_none".to_string()]);
        assert!(
            registry.lookup(&key).is_some(),
            "registry must carry the unsafe stub for annotator lookup"
        );
        // CallControl side: function_graphs is untouched, so
        // `find_all_graphs_for_tests` (which BFS-walks
        // `function_graphs.keys()`) cannot route to the stub.
        let cp = crate::parse::CallPath::from_segments(["pyobject", "is_none"]);
        assert!(
            !callcontrol.function_graphs().contains_key(&cp),
            "CallControl::function_graphs must NOT carry the unsafe stub path",
        );
        callcontrol.find_all_graphs_for_tests();
        assert!(
            !callcontrol.is_candidate(&cp),
            "find_all_graphs must not pick up an unsafe-stub-only path",
        );
    }

    #[test]
    fn register_unsafe_fn_stubs_does_not_overwrite_existing_entries() {
        // A path already registered via function_graphs (e.g. a safe fn
        // wearing the same segments) must NOT be overwritten by a stub.
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::rtyper::call_registry::CallRegistry;
        let bk = std::rc::Rc::new(Bookkeeper::new());
        let registry = CallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["pyobject".to_string(), "shared".to_string()]);
        let signature = Signature::new(vec!["x".to_string()], None, None);
        let existing = registry.get_or_register(key.clone(), signature.clone());
        let existing_host_id = existing.host_object.identity_id();
        let specs = vec![(
            vec!["pyobject".to_string(), "shared".to_string()],
            signature,
            Some("bool".to_string()),
        )];
        register_unsafe_fn_stubs(&registry, &specs);
        assert_eq!(registry.len(), 1, "no new entry expected");
        let after = registry.lookup(&key).expect("entry still present");
        assert_eq!(
            after.host_object.identity_id(),
            existing_host_id,
            "register_unsafe_fn_stubs must not replace an existing entry"
        );
    }

    #[test]
    fn build_stub_pygraph_returns_none_for_compound_lltype() {
        // Compound lltypes have no representable default; the helper
        // must return `None` so the caller skips registration.
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let func_ll = LowLevelType::Func(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::FuncType {
                args: vec![],
                result: LowLevelType::Void,
            },
        ));
        let result = build_stub_pygraph_for_lltype("synth".to_string(), sig, func_ll);
        assert!(result.is_none(), "Func lltype must surface as None");
    }

    #[test]
    fn call_registry_resolves_ll_math_math_floor_function_path() {
        // The front remaps opaque `f64::floor` to
        // `FunctionPath { segments: ["ll_math", "math_floor"] }`.  That
        // path must be in the CallRegistry after the same populate the
        // production builder runs; a miss is the
        // "not registered in CallRegistry, not in HOST_ENV" hard error.
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::lltypesystem::module::ll_math::F64_METHOD_LLEXTERNALS;
        let registry = std::rc::Rc::new(CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        )));
        populate_call_registry_from_call_graphs(
            &crate::codewriter::call::GraphStore::default(),
            &[],
            &[],
            &[],
            &registry,
        )
        .unwrap();
        let key = FunctionPathKey::from_segments(["ll_math", "math_floor"]);
        let floor = registry.lookup(&key).unwrap_or_else(|| {
            panic!(
                "FunctionPath {{ segments: [\"ll_math\", \"math_floor\"] }} \
                 is not registered in CallRegistry"
            )
        });
        assert_eq!(
            floor.function_desc.borrow().signature.argnames,
            ["x".to_string()]
        );
        for row in F64_METHOD_LLEXTERNALS {
            let entry = registry
                .lookup(&FunctionPathKey::from_segments(["ll_math", row.name]))
                .unwrap_or_else(|| {
                    panic!(
                        "FunctionPath {{ segments: [\"ll_math\", {:?}] }} \
                         is not registered in CallRegistry",
                        row.name
                    )
                });
            assert_eq!(
                entry.function_desc.borrow().signature.argnames.len(),
                row.arity,
                "{}",
                row.name
            );
            let stub = entry
                .function_desc
                .borrow()
                .cache
                .borrow()
                .get(&crate::annotator::description::GraphCacheKey::None)
                .cloned()
                .unwrap_or_else(|| panic!("{} stub is prefilled", row.name));
            let graph = stub.graph.borrow();
            let start = graph.startblock.borrow();
            let link = start.exits[0].borrow();
            let Some(Hlvalue::Variable(ret)) = link.args[0].as_ref() else {
                panic!(
                    "{} stub return arg must be a pre-annotated Variable",
                    row.name
                );
            };
            let annotation = ret.annotation.borrow();
            let annotation = annotation
                .as_ref()
                .unwrap_or_else(|| panic!("{} stub return is pre-annotated", row.name));
            assert!(
                matches!(&**annotation, SomeValue::Float(_)),
                "{} llexternal annotates as float (C result / front result_ty), \
                 got {annotation:?}; wrapper_raises={}",
                row.name,
                row.wrapper_raises
            );
        }
        let hypot = registry
            .lookup(&FunctionPathKey::from_segments(["ll_math", "math_hypot"]))
            .expect("math_hypot");
        assert_eq!(
            hypot.function_desc.borrow().signature.argnames,
            ["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn call_registry_resolves_atomic_load_accessor_function_path() {
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl;
        let registry = CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        ));
        let keep = DeclinedFunDecl {
            segments: vec![
                "pyre_object".into(),
                "lowlevel_string".into(),
                "lowlevel_str_gc_type_id".into(),
            ],
            arg_lltypes: vec![],
            result_lltype: LowLevelType::Unsigned,
            has_translatable_body: false,
            decline_reason: "unsupported MIR: atomic load ordering Acquire requires \
                 address-preserving ordered lowering"
                .into(),
        };
        let drop_ptr = DeclinedFunDecl {
            segments: vec![
                "pyre_object".into(),
                "typeobject".into(),
                "w_type_get_version_tag".into(),
            ],
            arg_lltypes: vec![LowLevelType::Func(Box::new(
                crate::translator::rtyper::lltypesystem::lltype::FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                },
            ))],
            result_lltype: LowLevelType::Unsigned,
            has_translatable_body: false,
            decline_reason: keep.decline_reason.clone(),
        };
        let drop_other = DeclinedFunDecl {
            segments: vec![
                "pyre_object".into(),
                "lowlevel_string".into(),
                "set_lowlevel_str_gc_type_id".into(),
            ],
            arg_lltypes: vec![LowLevelType::Unsigned],
            result_lltype: LowLevelType::Void,
            has_translatable_body: false,
            decline_reason: "declaration-has-no-unstructured-body".into(),
        };
        register_atomic_load_llexternals(&registry, &[keep, drop_ptr, drop_other]);
        let key = FunctionPathKey::from_segments([
            "pyre_object",
            "lowlevel_string",
            "lowlevel_str_gc_type_id",
        ]);
        let entry = registry.lookup(&key).unwrap_or_else(|| {
            panic!(
                "FunctionPath {{ segments: [\"pyre_object\", \"lowlevel_string\", \
                 \"lowlevel_str_gc_type_id\"] }} is not registered in CallRegistry"
            )
        });
        assert!(
            entry.function_desc.borrow().signature.argnames.is_empty(),
            "zero-arg accessor"
        );
        let stub = entry
            .function_desc
            .borrow()
            .cache
            .borrow()
            .get(&crate::annotator::description::GraphCacheKey::None)
            .cloned()
            .expect("accessor stub is prefilled");
        let graph = stub.graph.borrow();
        let start = graph.startblock.borrow();
        let link = start.exits[0].borrow();
        let Some(Hlvalue::Variable(ret)) = link.args[0].as_ref() else {
            panic!("accessor stub return arg must be a pre-annotated Variable");
        };
        let annotation = ret.annotation.borrow();
        let annotation = annotation
            .as_ref()
            .expect("accessor stub return is pre-annotated");
        match &**annotation {
            SomeValue::Integer(si) => {
                assert!(si.unsigned, "u32 GC type id shells as unsigned");
                assert!(
                    si.base.const_box.is_none(),
                    "GC type id is published at runtime; must not const-fold"
                );
            }
            other => panic!("expected unsigned SomeInteger, got {other:?}"),
        }
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyre_object",
                    "typeobject",
                    "w_type_get_version_tag"
                ]))
                .is_none(),
            "pointer-arg Acquire reader must not become an external"
        );
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyre_object",
                    "lowlevel_string",
                    "set_lowlevel_str_gc_type_id"
                ]))
                .is_none(),
            "a non-atomic-load decline must not become an external"
        );
    }

    fn atomic_load_llbc(ordering: Option<&str>, zero_arg: bool) -> majit_charon_reader::Llbc {
        use serde_json::{Value, json};
        let span = json!({"data": {"file_id": 0,
            "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
        let meta = |path: &[&str]| {
            json!({
                "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
                "span": span, "source_text": null, "is_local": true,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
            })
        };
        let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
        let adt = |id| json!({"Adt": {"id": id, "generics": generics}});
        let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
        // `AtomicU64` is the alias `Atomic<u64>`.
        let atomic = json!({"Adt": {"id": 0, "generics":
            {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
        let receiver = json!({"Ref": ["Erased", atomic, "Shared"]});
        let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
        let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
        let statements = ordering
            .map(|name| {
                let index = variants.iter().position(|v| *v == name).unwrap();
                json!([
                    {"span": span, "kind": {"Assign": [place(1, &receiver),
                        {"Aggregate": [{"Adt": [0, 0, null, generics]}, []]}]}},
                    {"span": span, "kind": {"Assign": [place(2, &adt(1)),
                        {"Aggregate": [{"Adt": [1, index, null, generics]}, []]}]}}
                ])
            })
            .unwrap_or_else(|| json!([]));
        let arg_count = if zero_arg { 0 } else { 1 };
        let inputs = if zero_arg {
            Vec::new()
        } else {
            vec![receiver.clone()]
        };
        let reader_path: &[&str] = if zero_arg {
            &["pyre_object", "lowlevel_string", "lowlevel_str_gc_type_id"]
        } else {
            &["fixture", "read_atomic"]
        };
        let file = json!({"charon_version": "0.1.201", "has_errors": false,
            "translated": {"crate_name": "fixture",
                "type_decls": [
                    {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                    {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                     "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                        "name": name, "fields": [],
                        "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                     })).collect::<Vec<_>>()}}
                ],
                "fun_decls": [
                    {"def_id": 0, "item_meta": meta(reader_path),
                     "signature": {"is_unsafe": false, "inputs": inputs, "output": word},
                     "body": {"Unstructured": {"span": span,
                        "locals": {"arg_count": arg_count, "locals": [
                            {"index": 0, "name": null, "span": span, "ty": word},
                            {"index": 1, "name": "slot", "span": span, "ty": receiver},
                            {"index": 2, "name": "ordering", "span": span, "ty": adt(1)}
                        ]},
                        "body": [
                            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                    "args": [{"Copy": place(1, &receiver)}, {"Copy": place(2, &adt(1))}],
                                    "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                            }}}},
                            {"statements": [], "terminator": {"span": span, "kind": "Return"}},
                            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
                        ]
                     }}},
                    {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                     "signature": {"is_unsafe": false, "inputs": [receiver, adt(1)], "output": word}, "body": "Opaque"}
                ], "global_decls": [], "trait_decls": [], "trait_impls": []
            }
        });
        majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("atomic fixture parses")
    }

    #[test]
    fn llexternal_path_declares_its_atomic_load_signature() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::lltypesystem::module::ll_extaccessor::{
            collect_atomic_load_llexternals, is_external_shaped_atomic_accessor,
        };
        let acquire = atomic_load_llbc(Some("Acquire"), true);
        let decls = crate::front::mir::build_semantic_program_from_llbc(&acquire)
            .expect("acquire fixture lowers")
            .atomic_load_decls;
        assert_eq!(decls.len(), 1);
        assert_eq!(
            decls[0].segments,
            ["pyre_object", "lowlevel_string", "lowlevel_str_gc_type_id"]
        );
        assert!(decls[0].arg_lltypes.is_empty());
        assert_eq!(decls[0].result_lltype, LowLevelType::Unsigned);
        assert!(!decls[0].has_translatable_body);
        assert!(decls[0].decline_reason.contains("llexternal"));
        assert!(is_external_shaped_atomic_accessor(&decls[0]));
        assert_eq!(collect_atomic_load_llexternals(&decls).len(), 1);

        // An unlisted path whose body does not lower declares nothing: a
        // lowering failure is not an `llexternal` declaration.
        let pointer = atomic_load_llbc(Some("Acquire"), false);
        let decls = crate::front::mir::build_semantic_program_from_llbc(&pointer)
            .expect("pointer fixture lowers")
            .atomic_load_decls;
        assert!(decls.is_empty());

        // The path is declared an external whatever its body loads: an
        // `rffi.llexternal` is a declaration, not a lowering outcome.
        let relaxed = atomic_load_llbc(Some("Relaxed"), true);
        let decls = crate::front::mir::build_semantic_program_from_llbc(&relaxed)
            .expect("relaxed fixture lowers")
            .atomic_load_decls;
        assert_eq!(decls.len(), 1);
        assert!(is_external_shaped_atomic_accessor(&decls[0]));
    }

    #[test]
    fn harvested_zero_arg_atomic_load_accessor_registers_as_extfunc_entry() {
        use crate::annotator::model::SomeValue;
        let acquire = atomic_load_llbc(Some("Acquire"), true);
        let decls = crate::front::mir::build_semantic_program_from_llbc(&acquire)
            .expect("acquire fixture lowers")
            .atomic_load_decls;
        let registry = std::rc::Rc::new(CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        )));
        populate_call_registry_from_call_graphs(
            &crate::codewriter::call::GraphStore::default(),
            &[],
            &[],
            &decls,
            &registry,
        )
        .unwrap();
        let key = FunctionPathKey::from_segments([
            "pyre_object",
            "lowlevel_string",
            "lowlevel_str_gc_type_id",
        ]);
        let entry = registry.lookup(&key).unwrap_or_else(|| {
            panic!(
                "harvested FunctionPath [pyre_object, lowlevel_string, \
                 lowlevel_str_gc_type_id] is not registered in CallRegistry"
            )
        });
        let stub = entry
            .function_desc
            .borrow()
            .cache
            .borrow()
            .get(&crate::annotator::description::GraphCacheKey::None)
            .cloned()
            .expect("accessor stub is prefilled");
        let graph = stub.graph.borrow();
        let start = graph.startblock.borrow();
        let link = start.exits[0].borrow();
        let Some(Hlvalue::Variable(ret)) = link.args[0].as_ref() else {
            panic!("accessor stub return arg must be a pre-annotated Variable");
        };
        let annotation = ret.annotation.borrow();
        let annotation = annotation
            .as_ref()
            .expect("accessor stub return is pre-annotated");
        match &**annotation {
            SomeValue::Integer(si) => {
                assert!(si.unsigned, "u64 GC type id shells as unsigned");
                assert!(
                    si.base.const_box.is_none(),
                    "GC type id is published at runtime; must not const-fold"
                );
            }
            other => panic!("expected unsigned SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn harvested_atomic_load_accessor_lowers_to_published_residual_call() {
        use crate::codewriter::call::is_symbolic_fnaddr;
        use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
        use crate::codewriter::type_state::ConcreteType;
        use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
        let acquire = atomic_load_llbc(Some("Acquire"), true);
        let decls = crate::front::mir::build_semantic_program_from_llbc(&acquire)
            .expect("acquire fixture lowers")
            .atomic_load_decls;
        let registry = std::rc::Rc::new(CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        )));
        populate_call_registry_from_call_graphs(
            &crate::codewriter::call::GraphStore::default(),
            &[],
            &[],
            &decls,
            &registry,
        )
        .unwrap();
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyre_object",
                    "lowlevel_string",
                    "lowlevel_str_gc_type_id",
                ]))
                .is_some(),
            "annotator registry has the harvested accessor"
        );

        let mut graph = FunctionGraph::new("caller");
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::function_path([
                        "pyre_object",
                        "lowlevel_string",
                        "lowlevel_str_gc_type_id",
                    ]),
                    args: vec![],
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);

        // Production binds this path from `jit_trace_fnaddrs` via
        // `register_macro_helper_trace_fnaddr`.  A non-zero, non-symbolic
        // stand-in is the same insert; `0` is refused by that entry point.
        const PUBLISHED: i64 = 0x00AB_CDEF;
        let mut callcontrol = crate::call::CallControl::new();
        callcontrol.register_macro_helper_trace_fnaddr(
            "pyre_object::lowlevel_string::lowlevel_str_gc_type_id",
            PUBLISHED,
        );
        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut callcontrol)
            .transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let fnaddr = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::ConstFnAddr { value, .. } => Some(*value),
                OpKind::ConstInt(addr) => Some(*addr),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("residual call materializes a funcptr constant; ops={ops:?}")
            });
        assert!(
            !is_symbolic_fnaddr(fnaddr),
            "published jit_trace_fnaddrs row must not bake a symbolic hash; fnaddr={fnaddr:#x}"
        );
        assert_eq!(fnaddr, PUBLISHED);
        assert!(
            ops.iter()
                .any(|op| matches!(op.kind, OpKind::CallResidual { .. })),
            "call site must residualize; got {ops:?}"
        );
    }

    #[test]
    fn populate_refuses_unverified_atomic_load_harvest() {
        use crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl;
        let decls = vec![DeclinedFunDecl {
            segments: vec!["pyre_object".into(), "gc_interp".into(), "safepoint".into()],
            arg_lltypes: vec![],
            result_lltype: LowLevelType::Void,
            has_translatable_body: false,
            decline_reason: "unsupported MIR: atomic load ordering Acquire requires \
                 address-preserving ordered lowering"
                .into(),
        }];
        let registry = std::rc::Rc::new(CallRegistry::new(std::rc::Rc::new(
            crate::annotator::bookkeeper::Bookkeeper::new(),
        )));
        populate_call_registry_from_call_graphs(
            &crate::codewriter::call::GraphStore::default(),
            &[],
            &[],
            &decls,
            &registry,
        )
        .unwrap();
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments([
                    "pyre_object",
                    "gc_interp",
                    "safepoint"
                ]))
                .is_none(),
            "safepoint contains an Acquire load but is not a word-only reader"
        );
    }

    /// An unfollowed return whose real twin is `Void` must not overwrite a
    /// legacy `Signed` cell. A followed return, and a session that never
    /// recorded `links_followed`, keeps the twin.
    #[test]
    fn unfollowed_void_return_keeps_the_legacy_kind() {
        use crate::model::ConcreteType;

        let legacy = LegacyGraph::new("unfollowed_void_return");
        let ret = legacy.block(legacy.returnblock).inputargs[0].clone();
        ret.set_concretetype(crate::model::concrete_to_canonical_lltype(
            ConcreteType::Signed,
        ));
        let typed = Variable::new();
        typed.set_concretetype(crate::model::concrete_to_canonical_lltype(
            ConcreteType::Void,
        ));
        let mut value_to_var = LegacyToTyped::new();
        value_to_var.insert(ret.clone(), typed);
        let mut constants = HashMap::new();
        constants.insert(ret.clone(), LowLevelType::Void);
        let excluded = HashSet::from([ret.clone()]);
        keep_legacy_kind_when_unfollowed_void(&mut value_to_var, &mut constants, Some(&excluded));
        assert!(
            !value_to_var.contains_key(&ret),
            "unfollowed void twin stays out of the publish map"
        );
        assert!(!constants.contains_key(&ret));
        assert_eq!(
            LegacyGraph::concretetype_of(&ret),
            ConcreteType::Signed,
            "legacy kind is the CFG return `history.getkind` reads"
        );

        let typed = Variable::new();
        typed.set_concretetype(crate::model::concrete_to_canonical_lltype(
            ConcreteType::Void,
        ));
        let mut followed_map = LegacyToTyped::new();
        followed_map.insert(ret.clone(), typed);
        let mut followed_constants = HashMap::new();
        keep_legacy_kind_when_unfollowed_void(
            &mut followed_map,
            &mut followed_constants,
            Some(&HashSet::new()),
        );
        assert!(
            followed_map.contains_key(&ret),
            "a followed return keeps its void twin"
        );

        let typed = Variable::new();
        typed.set_concretetype(crate::model::concrete_to_canonical_lltype(
            ConcreteType::Void,
        ));
        let mut unscoped = LegacyToTyped::new();
        unscoped.insert(ret.clone(), typed);
        let mut unscoped_constants = HashMap::new();
        keep_legacy_kind_when_unfollowed_void(&mut unscoped, &mut unscoped_constants, None);
        assert!(
            unscoped.contains_key(&ret),
            "no links_followed record leaves the twin in place"
        );
    }

    /// Residual-call actual (`emit_one_word_guard_release` Signed stamp)
    /// whose AccessDirect copy was unfollowed: Phase B Void must not
    /// publish over the skip-arm kind, or `getcalldescr` drops the actual.
    #[test]
    fn unfollowed_void_call_arg_keeps_the_legacy_kind() {
        use crate::model::ConcreteType;

        let arg = Variable::new();
        arg.set_concretetype(crate::model::concrete_to_canonical_lltype(
            ConcreteType::Signed,
        ));
        let typed = Variable::new();
        typed.set_concretetype(crate::model::concrete_to_canonical_lltype(
            ConcreteType::Void,
        ));
        let mut value_to_var = LegacyToTyped::new();
        value_to_var.insert(arg.clone(), typed);
        let mut constants = HashMap::new();
        constants.insert(arg.clone(), LowLevelType::Void);
        let excluded = HashSet::from([arg.clone()]);
        keep_legacy_kind_when_unfollowed_void(&mut value_to_var, &mut constants, Some(&excluded));
        assert!(
            !value_to_var.contains_key(&arg),
            "unfollowed void call-arg twin stays out of the publish map"
        );
        assert!(!constants.contains_key(&arg));
        assert_eq!(
            LegacyGraph::concretetype_of(&arg),
            ConcreteType::Signed,
            "legacy kind is the residual ABI word getcalldescr reads"
        );
    }
}
