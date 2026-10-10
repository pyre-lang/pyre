/// Loop unrolling pass (peel one iteration).
///
/// Detects loops ending with a `Jump` back-edge and peels one iteration:
/// the loop body is duplicated to create a "preamble" that executes once
/// before the main loop. Guards in the peeled preamble serve as initial
/// type checks, enabling downstream passes to remove redundant guards
/// from the main loop body.
///
/// The peeled structure looks like:
///
/// ```text
/// [peeled body]          ← first iteration (preamble), guards act as type checks
///   Label(...)           ← loop header
/// [original body]        ← main loop body
///   Jump(...)            ← back-edge to Label
/// ```
///
/// OpRefs in the peeled iteration are remapped to new positions so they
/// don't collide with the original ops.
use indexmap::{IndexMap, IndexSet};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, IndexMapExt, Op, OpCode, OpRc, OpRef, Type, Value};

use crate::history::TargetToken;
use crate::optimizeopt::{
    OptContext, Optimization, OptimizationResult, SnapshotBoxes, SnapshotFramePcs,
    SnapshotFrameSizes, next_snapshot_pos, snapshot_get, snapshot_insert,
};
use crate::resume::SnapshotBox;

/// `unroll.py:119-123`:
///
/// ```python
/// try:
///     info, _ = self.propagate_all_forward(trace, call_pure_results, flush=False)
/// except SpeculativeError:
///     raise InvalidLoop("Speculative heap access would be ill-typed")
/// ```
///
/// Catch a `SpeculativeError` panic raised by
/// `OptContext::protect_speculative_operation` and rethrow it as an
/// `InvalidLoop` — the existing catch_unwind sites at the pyjitpl
/// layer already convert that into a "skip retrace" outcome.
/// unroll.py:119-123 / 122 `except SpeculativeError: raise InvalidLoop`.
///
/// Speculative-fold failures are now recorded as a deferred `InvalidLoop`
/// signal on `OptContext` (see `protect_speculative_operation` /
/// `constant_fold`) and surfaced as `Err(InvalidLoop)` by the optimizer
/// driver, so there is no `SpeculativeError` panic left to catch.  This
/// wrapper is retained as a structural marker of the upstream conversion
/// point; it simply forwards the closure's result.
fn with_speculative_to_invalid_loop<R, F>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

fn is_trace_constant_ref(opref: OpRef, constants: &majit_ir::ConstMap<majit_ir::Value>) -> bool {
    if opref.is_none() {
        return false;
    }
    // history.py — Const variants are constants by Box class.
    // Const OpRefs carry the value directly; the surviving raw `constants`
    // map (empty in production) is also treated as Const. A guard minted by
    // `jump_to_existing_trace` (IntBound.make_guards, unroll.py) can
    // reference a fresh const-namespace OpRef whose raw is not yet a key in
    // `constants`; the variant check is authoritative, while the map check
    // additionally catches body-namespace OpRefs forwarded to a constant.
    if opref.is_constant() {
        return true;
    }
    constants.contains_key(&opref.raw())
}

fn is_trace_runtime_ref(opref: OpRef, constants: &majit_ir::ConstMap<majit_ir::Value>) -> bool {
    !opref.is_none() && !is_trace_constant_ref(opref, constants)
}

/// A body use-before-def whose `_forwarded` terminal is a non-null
/// ConstPtr is not a loop-carried runtime box. rewrite.py `used_boxes`
/// / `remove_constptr` drop Const; the flat OpRef model remints it as
/// an InputArg at a dead index, so assemble emits a SameAs of the
/// Const before the Label (defining that index) instead of appending
/// it to the header. Int/Float consts stay live-in: GuardTrue/False
/// postprocess may install those after emit, and baking them as the
/// first-iteration value is a 0-fill.
fn const_ref_replacement(ctx: &OptContext, opref: OpRef) -> Option<Operand> {
    let as_const_ref = |term: Operand| match term.const_value() {
        Some(Value::Ref(gcref)) if !gcref.is_null() => Some(term),
        _ => None,
    };
    if let Some(term) = as_const_ref(ctx.get_box_replacement_operand(opref)) {
        return Some(term);
    }
    // Compact remap keeps the InputArg variant when it repositions a
    // folded Ref producer (`with_raw`). The Const lives on the ResOp
    // at the same raw; `find_producer_op` is variant-aware and misses
    // the InputArgRef use.
    if matches!(opref, OpRef::InputArgRef(_)) {
        return as_const_ref(ctx.get_box_replacement_operand(OpRef::ref_op(opref.raw())));
    }
    None
}

fn callee_rca_virtual_state_summary(
    vs: &crate::optimizeopt::virtualstate::VirtualState,
) -> Vec<String> {
    fn walk(
        prefix: String,
        node: &crate::optimizeopt::virtualstate::VirtualStateInfoNode,
        out: &mut Vec<String>,
        seen: &mut crate::FxIndexSet<usize>,
    ) {
        let key = node as *const _ as usize;
        let kind = match &node.info {
            crate::optimizeopt::virtualstate::VirtualStateInfo::Constant(value) => {
                format!("Constant({value:?})")
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::Virtual { fields, .. } => {
                format!("Virtual(fields={})", fields.len())
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArray { items, .. } => {
                format!("VArray(items={})", items.len())
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VStruct { fields, .. } => {
                format!("VStruct(fields={})", fields.len())
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArrayStruct {
                element_fields,
                ..
            } => format!("VArrayStruct(elements={})", element_fields.len()),
            crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass { class_ptr } => {
                format!("KnownClass({class_ptr})")
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull => "NonNull".to_string(),
            crate::optimizeopt::virtualstate::VirtualStateInfo::IntBounded(bound) => {
                format!("IntBounded({bound:?})")
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(tp) => {
                format!("Unknown({tp:?})")
            }
        };
        out.push(format!(
            "{prefix}: {kind} pos={} notvirt={}",
            node.position.get(),
            node.position_in_notvirtuals.get()
        ));
        if !seen.insert(key) {
            return;
        }
        match &node.info {
            crate::optimizeopt::virtualstate::VirtualStateInfo::Virtual { fields, .. }
            | crate::optimizeopt::virtualstate::VirtualStateInfo::VStruct { fields, .. } => {
                for (idx, child) in fields {
                    walk(format!("{prefix}.{idx}"), child, out, seen);
                }
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArray { items, .. } => {
                for (idx, child) in items.iter().enumerate() {
                    if let Some(child) = child {
                        walk(format!("{prefix}[{idx}]"), child, out, seen);
                    }
                }
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArrayStruct {
                element_fields,
                ..
            } => {
                for (elem_idx, fields) in element_fields.iter().enumerate() {
                    for (field_idx, child) in fields {
                        if let Some(child) = child {
                            walk(
                                format!("{prefix}[{elem_idx}].{field_idx}"),
                                child,
                                out,
                                seen,
                            );
                        }
                    }
                }
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::Constant(_)
            | crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass { .. }
            | crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull
            | crate::optimizeopt::virtualstate::VirtualStateInfo::IntBounded(_)
            | crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(_) => {}
        }
    }

    let mut out = Vec::new();
    let mut seen = crate::FxIndexSet::default();
    for (idx, node) in vs.state.iter().enumerate() {
        walk(format!("state[{idx}]"), node, &mut out, &mut seen);
    }
    out
}

/// Root any GcRef payload reachable from a single `_forwarded` slot (the
/// `AbstractInputArg.forwarded` / `AbstractResOp.forwarded` host,
/// resoperation.py / :700). PyPy's Python GC walks `_forwarded`
/// transitively; pyre pins each GcRef on the shadow stack instead.
/// Used by `ExportedState::root_all_gcrefs` to keep
/// `PtrInfo::Constant` / `Const::Ref` payloads live across GC pauses.
/// (`PtrInfo::Instance.known_class` is an immortal vtable integer, not a
/// traced ref, so it is not rooted.)
fn root_forwarded_gcref(
    forwarded: &majit_ir::forwarding::Forwarded,
    info_constant_field: ExportedGcRefField,
    const_ref_field: ExportedGcRefField,
    dummy_key: Operand,
    rooted_refs: &mut Vec<(Operand, ExportedGcRefField, usize)>,
) {
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    if let majit_ir::forwarding::Forwarded::Info(OpInfo::Ptr(rc)) = forwarded {
        let info = rc.borrow();
        match &*info {
            PtrInfo::Constant(gcref) if !gcref.is_null() => {
                let ss_idx = majit_gc::shadow_stack::push(*gcref);
                rooted_refs.push((dummy_key.clone(), info_constant_field, ss_idx));
            }
            // PtrInfo::Instance.known_class is an immortal vtable integer
            // (ConstInt), never a traced ref — no rooting needed.
            _ => {}
        }
    } else if let Some(gcref) = forwarded.const_ref() {
        let ss_idx = majit_gc::shadow_stack::push(gcref);
        rooted_refs.push((dummy_key, const_ref_field, ss_idx));
    }
}

/// Mutate the `PtrInfo::Constant` payload of a `Forwarded::Info` cell in
/// place so any other handle sharing the `Rc<RefCell<PtrInfo>>` sees the
/// post-GC GcRef. Matches PyPy `_forwarded` Python object reference
/// semantics — the cell stays, only its content updates.
fn refresh_forwarded_ptrinfo_constant(
    forwarded: &majit_ir::forwarding::Forwarded,
    updated: majit_ir::GcRef,
) {
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    let rc = match forwarded {
        majit_ir::forwarding::Forwarded::Info(OpInfo::Ptr(rc))
            if matches!(&*rc.borrow(), PtrInfo::Constant(_)) =>
        {
            Some(rc.clone())
        }
        _ => None,
    };
    if let Some(rc) = rc {
        *rc.borrow_mut() = PtrInfo::Constant(updated);
    }
}

/// Overwrite a `Forwarded::Const(Const::Ref(_))` payload in place with the
/// post-GC GcRef. Matches PyPy `_forwarded` Python object reference
/// semantics — the chain terminal stays a Const, only its GcRef updates.
fn refresh_forwarded_const_ref(
    forwarded: &majit_ir::forwarding::Forwarded,
    updated: majit_ir::GcRef,
) {
    forwarded.refresh_const_ref(updated);
}

/// `pyjitpl.py compile_trace`'s `live_arg_boxes[num_green_args:]` —
/// `runtime_boxes` are the original boxes, InputArg and Op alike.
/// `generate_guards` reads them via `getint`/`getref_base` /
/// `get_runtime_field` (virtualstate.py, :493). A raw OpRef does
/// not carry `_resint`/`_resref`, so materialize each stamped value as
/// an inline Const — the same recovery `closing_jump_runtime_boxes`
/// already performs for bridges. InputArg values live on
/// `trace_inputarg_boxes`; ResOp values live on the recorded producer
/// (`set_concrete_at` → `Op::set_value`). Without the ResOp arm a
/// virtual W_IntObject JUMP slot arrives as `RefOp(n)` with no
/// readable intval, and `IntBounded` vs `Unknown(Int)` rejects the
/// peel even though the first-iteration heap value is in range.
fn fold_recorded_jump_args(
    args: Vec<OpRef>,
    inputarg_boxes: &[majit_ir::InputArgRc],
    recorded_ops: &[OpRc],
) -> Vec<OpRef> {
    args.into_iter()
        .map(|arg| {
            let inputarg_index = match arg {
                OpRef::InputArgInt(i) | OpRef::InputArgFloat(i) | OpRef::InputArgRef(i) => {
                    Some(i as usize)
                }
                _ => None,
            };
            if let Some(inputarg_box) = inputarg_index.and_then(|i| inputarg_boxes.get(i))
                && Some(inputarg_box.tp.get()) == arg.ty()
                && let Some(value) = inputarg_box.get_value()
                && !matches!(value, Value::Void)
            {
                return OpRef::const_inline_from_value(&value);
            }
            if let Some(op) = recorded_ops.iter().find(|op| op.pos().get() == arg)
                && let Some(value) = op.get_value()
                && !matches!(value, Value::Void)
            {
                return OpRef::const_inline_from_value(&value);
            }
            arg
        })
        .collect()
}

/// unroll.py: UnrollOptimizer — high-level loop optimization controller.
///
/// Wraps the streaming OptUnroll pass with RPython's UnrollOptimizer API:
/// - optimize_preamble: process and optimize the first iteration
/// - optimize_peeled_loop: optimize the main loop body
pub struct UnrollOptimizer {
    /// Backend capability propagated to both phase optimizers.
    pub supports_efficient_uint_mul_high: bool,
    /// The short preamble from the preamble optimization pass.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// history.py: JitCellToken.target_tokens — compiled versions of this loop.
    /// Each TargetToken has its own virtual state and short preamble.
    pub target_tokens: Vec<std::sync::Arc<TargetToken>>,
    /// `compile.py:355` — the `JitCellToken` this compilation's artifact will be
    /// installed under, when there is one. `compile_retrace` resolves it from
    /// `get_procedure_token(greenkey)` and the result is attached as a bridge to
    /// that same token (`compile.py:797-811`), so a close onto one of its
    /// `target_tokens` stays inside one code buffer.
    ///
    /// `None` for a fresh `compile_loop`, which mints its own token afterwards.
    /// `compile.py:287-289` expresses the same thing by *resetting*
    /// `jitcell_token.target_tokens` to `[start_descr]`, leaving nothing for
    /// `_jump_to_existing_trace` to match; pyre seeds the previous
    /// compilation's tokens instead, so the distinction has to be carried here.
    pub attach_jitcell_token_number: Option<u64>,
    /// history.py: JitCellToken.retraced_count — number of times this loop
    /// has been retraced. Compared against retrace_limit to prevent infinite
    /// retracing.
    pub retraced_count: u32,
    /// warmstate.py: retrace_limit parameter. When retraced_count reaches
    /// this limit, jump_to_preamble is forced instead of creating a new
    /// target token.
    pub retrace_limit: u32,
    /// warmstate.py: max_retrace_guards parameter. If a compiled trace has
    /// more guards than this, retracing is permanently disabled.
    pub max_retrace_guards: u32,
    /// compile.py: pre-imported ExportedState for compile_retrace.
    /// When set, Phase 1 (preamble) is skipped and Phase 2 uses this state
    /// directly, matching UnrolledLoopData.optimize → optimize_peeled_loop.
    pub imported_state: Option<ExportedState>,
    /// compile.py vs compile.py: `compile_loop` assembles
    /// `[start_label] + preamble_ops + ... + [label_op] + loop_ops`, but
    /// `compile_retrace` assembles `loop.operations + extra_same_as +
    /// [label_op] + loop_ops` — a retrace has NO start label, because its
    /// entry is the failing guard the resumekey attaches it to, not a fresh
    /// loop header. RPython splits this by having the two callers assemble
    /// the op list themselves; majit's optimizer returns it pre-assembled,
    /// so `compile_retrace` clears this flag instead.
    pub emit_start_label: bool,
    /// Phase 1's finalized ExportedState, retained across the Phase 2 run
    /// so the caller of `optimize_trace_*` can consult the renamed inputarg
    /// types that the optimizer decided on. RPython does not need this
    /// (Box.type is intrinsic), but majit's `InputArg.tp.get()` side table is
    /// otherwise disconnected from the optimizer's reduced LABEL.
    pub final_exported_state: Option<ExportedState>,
    /// Compact TargetToken LABEL source positions in the full Phase-1
    /// `end_args` vector. This is the small part of `ExportedState.end_args`
    /// needed by Cranelift's host-side direct LABEL entry after
    /// `final_exported_state` is reduced to type metadata.
    pub final_exported_label_source_positions: Option<Vec<usize>>,
    // RPython compile.py:278-284: Phase 1 results for retrace_needed.
    // In RPython, Phase 1 and Phase 2 are separate calls, so Phase 1
    // results are naturally accessible. In pyre, Phase 1 results are
    // returned via the phase1_out output parameter to the caller's
    // stack frame (survives Phase 2 panic).
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Passed through to Phase 1 and Phase 2 optimizers for
    /// store_final_boxes_in_guard snapshot-based fail_args rebuild.
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// Per-guard virtualref boxes from tracing-time snapshots
    /// (resume.py vref_array — _number_boxes consumes them
    /// after the virtualizable array).
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// `trace` in `optimize_preamble` / `optimize_peeled_loop`: the byte
    /// recorder whose `get_snapshot_iter` numbers each surviving guard.
    /// Each phase installs it over its own `TraceIterator._cache`.
    /// The caller keeps the recorder alive across `optimize_trace*`.
    pub(crate) snapshot_recorder: Option<*const crate::recorder::Trace>,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    /// Threaded through inner Optimizer instances for inline registration.
    pub all_descrs: std::sync::Arc<Vec<majit_ir::descr::DescrRef>>,
    /// compile.py:204-207 / heap.py:821-823 quasi-immutable deps collected
    /// by the phase optimizers for post-compile watcher registration.
    pub quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    /// RPython Box type parity: trace inputarg types from recorder.
    /// Each RPython Box carries its type; in majit OpRef is untyped u32.
    /// Propagated to Phase 1 and Phase 2 Optimizer.trace_inputargs
    /// so value_types covers inputarg OpRefs.
    pub trace_inputargs: Vec<majit_ir::OpRef>,
    /// compile.py `PreambleCompileData(trace, jumpargs, ...)` - the
    /// ORIGINAL history boxes behind `runtime_boxes`. `trace_inputargs`
    /// carries only each entry slot's position and type; these are the boxes
    /// themselves, whose observed value `virtualstate.py`'s `_generate_guards`
    /// / `_generate_guards_unkown` / `_generate_guards_nonnull` /
    /// `_generate_guards_knownclass` read un-forwarded off `runtime_boxes[i]`.
    pub trace_inputarg_boxes: Vec<majit_ir::InputArgRc>,
    /// Phase 1 emit ops (filtered to non-NONE pos, non-Void type) carried
    /// into Phase 2 so that `OptContext::op_at` resolves Phase 1 OpRefs
    /// directly via `op.type_` (history.py:220 box.type parity).
    phase1_emit_ops: Vec<majit_ir::OpRc>,
    /// RPython: same Optimizer instance across phases keeps patchguardop.
    /// In majit, separate instances — forward explicitly.
    phase1_patchguardop: Option<majit_ir::Op>,
    /// Cross-phase fresh OpRef high water (majit-specific companion to
    /// RPython's `TraceIterator._index`).
    ///
    /// In RPython each `TraceIterator.next()` allocates a fresh `cls()`
    /// ResOperation whose Python identity distinguishes Phase 1 from
    /// Phase 2 boxes; majit's `OpRef::from_raw(u32)` IS the identity, so Phase 2 must
    /// continue allocating *above* Phase 1's high water mark to keep the
    /// two phases' OpRef sets disjoint. After Phase 1 finishes,
    /// `next_global_opref` holds the smallest OpRef Phase 2 may emit; it
    /// is the `start_fresh` argument the next `TraceIterator::new` call
    /// (or bridge entry) should use. Initialized to 0.
    #[allow(dead_code)]
    pub(crate) next_global_opref: u32,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Maps oopspec indices to (calldescr, func_ptr) for generate_modified_call.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// compile.py:221 + optimizer.py:530: call_pure_results from tracing.
    /// Passed through to the inner Optimizer for cross-iteration CALL_PURE folding.
    pub call_pure_results: crate::optimizeopt::util::ArgsDict,
    /// `optimizer.cpu` (model.py `AbstractCPU`) backref, carried into
    /// the inner phase-1/phase-2 `Optimizer.cpu` at spawn time so
    /// `cpu.cls_of_box(runtime_box)` reads (virtualstate.py:601/:608/:620)
    /// and any future `bh_*` calls resolve to the same backend services.
    pub cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    /// `optimizer.py Optimizer.__init__` host slots the inner phase
    /// Optimizers need. `default_pipeline()` leaves them empty; pyjitpl
    /// copies them off `MetaInterp` before `optimize_trace_*`.
    pub string_length_resolver: Option<crate::optimizeopt::info::StringLengthResolver>,
    pub string_content_resolver: Option<crate::optimizeopt::info::StringContentResolver>,
    pub string_constant_alloc: Option<crate::optimizeopt::info::StringConstantAllocator>,
    pub vrefinfo: Option<crate::virtualref::VirtualRefInfo>,
    pub pureop_historylength: usize,
    /// Explicit `input_ops` seed for the phase optimizers, threaded from the
    /// compile caller. `Some(ops)` on the non-cut finish path =
    /// `preamble_data.base.operations()` — the recorder's `Rc<Op>` carrying
    /// the authoritative Phase-1 `_forwarded`. `Some(empty)` on the cut path:
    /// the cut trace's ops live in a remapped namespace, so no seed can
    /// resolve cut-op lookups anyway; an empty seed states that. `None`
    /// (retrace / fixtures) leaves `input_ops` empty.
    pub phase2_input_ops_seed: Option<Vec<majit_ir::OpRc>>,
    /// MetaInterp-owned compile snapshot root slot list. Stored as an address
    /// because the unroll optimizer owns the inner phase optimizers while the
    /// registered GC walker enters through MetaInterp.
    pub compile_snapshot_root_slots: Option<usize>,
    /// Address of `MetaInterp.compile_resume_memos`, forwarded to every
    /// `Optimizer` this unroll builds so its memo is rooted while in flight.
    pub compile_resume_memos_slot: Option<usize>,
    /// MetaInterp-owned slot that publishes the in-flight phase-2
    /// `Optimizer.short_preamble_producer` to the registered GC walker.
    pub compile_short_preamble_producer_slot: Option<usize>,
    /// Snapshot-root slots that must stay rooted across every phase's
    /// `replace_compile_snapshot_roots`. pyjitpl installs the caller-owned
    /// original snapshot maps here so a moving GC during unroll forwards their
    /// inline `ConstPtr`s in place — the `InvalidLoop` retry re-clones those
    /// originals and would otherwise read a stale pre-move gcref. Prepended to
    /// each phase's slot list rather than overwritten.
    pub persistent_snapshot_root_slots: Vec<usize>,
    /// `compile.py` `enable_opts` handed to `build_opt_chain` for each phase
    /// optimizer. Defaults to `ENABLE_ALL_OPTS`.
    pub enable_opts: Vec<String>,
}

/// Withdraws the address `publish_short_preamble_producer` installed in
/// `MetaInterp.compile_short_preamble_producer`.
pub(crate) struct PublishedShortPreambleProducer {
    slot: Option<usize>,
    previous: Option<usize>,
}

impl Drop for PublishedShortPreambleProducer {
    fn drop(&mut self) {
        if let Some(addr) = self.slot {
            // SAFETY: the same address pyjitpl installed for this compile, on
            // the same thread as the registered root walker. Restoring the
            // previous value preserves an outer publication and keeps the
            // walker from reading the optimizer local after it is dropped.
            unsafe {
                *(addr as *mut Option<usize>) = self.previous;
            }
        }
    }
}

/// Temporarily publishes the context storage while an extended short-preamble
/// builder is on loan from its optimizer. The two storage fields have the same
/// `Option<ExtendedShortPreambleBuilder>` type, which is required by the root
/// walker's cast.
struct ActiveShortPreambleProducerPublication {
    slot: Option<usize>,
    optimizer_producer_slot: usize,
}

impl ActiveShortPreambleProducerPublication {
    fn new(optimizer: &mut crate::optimizeopt::optimizer::Optimizer, ctx: &mut OptContext) -> Self {
        let optimizer_producer_slot = (&mut optimizer.short_preamble_producer
            as *mut Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>)
            as usize;
        let slot = optimizer.published_short_preamble_producer_slot;
        if let Some(addr) = slot {
            // SAFETY: `publish_short_preamble_producer` installed this
            // MetaInterp-owned slot for the current optimizer on the same
            // thread as the registered root walker.
            unsafe {
                *(addr as *mut Option<usize>) =
                    Some(ctx.active_short_preamble_producer_slot_addr());
            }
        }
        Self {
            slot,
            optimizer_producer_slot,
        }
    }
}

impl Drop for ActiveShortPreambleProducerPublication {
    fn drop(&mut self) {
        if let Some(addr) = self.slot {
            // SAFETY: the builder has returned to the optimizer before this
            // guard drops, so the published address again names its storage.
            unsafe {
                *(addr as *mut Option<usize>) = Some(self.optimizer_producer_slot);
            }
        }
    }
}

impl UnrollOptimizer {
    /// `optimizer.py Optimizer.__init__` fan-out onto a phase Optimizer
    /// spawned from `default_pipeline()`.
    fn pin_inner_optimizer(&self, opt: &mut crate::optimizeopt::optimizer::Optimizer) {
        opt.cpu = self.cpu.clone();
        opt.supports_efficient_uint_mul_high = self.supports_efficient_uint_mul_high;
        opt.set_pureop_historylength(self.pureop_historylength);
        if let Some(vrefinfo) = self.vrefinfo.clone() {
            opt.set_vrefinfo(vrefinfo);
        }
        opt.string_length_resolver = self.string_length_resolver.clone();
        opt.string_content_resolver = self.string_content_resolver.clone();
        opt.string_constant_alloc = self.string_constant_alloc.clone();
    }

    /// Supply the target tokens an earlier compile of this green key left
    /// behind.
    ///
    /// `unroll.py` declares `short_preamble_producer = None`;
    /// `import_state` sets its plain builder at `unroll.py`, and
    /// `finalize_short_preamble` replaces it with the extended builder at
    /// `unroll.py:298`. Thus one extended producer is live for an optimizer
    /// run, which is what `unroll.py:378`'s
    /// `assert isinstance(sb, ExtendedShortPreambleBuilder)` relies on.
    ///
    /// `inline_short_preamble` retains the `sb.target_token is target_token`
    /// discrimination at `unroll.py:376-385`; its descriptor-identity gate
    /// prevents an in-place setup against every matching candidate.
    pub fn seed_prior_target_tokens(&mut self, tokens: Vec<std::sync::Arc<TargetToken>>) {
        self.target_tokens = tokens;
    }

    /// Hand a phase optimizer its own snapshot banks without emptying
    /// `self`. compile_loop_body takes these fields back on InvalidLoop
    /// for `compile_simple_loop`; a `mem::take` into Phase 2 left that
    /// retry with no snapshot 0.
    fn clone_snapshots_onto(&self, opt: &mut crate::optimizeopt::optimizer::Optimizer) {
        opt.snapshot_boxes = self.snapshot_boxes.clone();
        opt.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
        opt.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
        opt.snapshot_vref_boxes = self.snapshot_vref_boxes.clone();
        opt.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
    }

    pub fn new() -> Self {
        UnrollOptimizer {
            supports_efficient_uint_mul_high: true,
            short_preamble: None,
            target_tokens: Vec::new(),
            attach_jitcell_token_number: None,
            retraced_count: 0,
            // unroll.py:215/265 reads
            // `warmrunnerdescr.memory_manager.{retrace_limit,max_retrace_guards}`.
            // Production callers (`pyjitpl.rs`'s `compile_loop_body` /
            // `compile_retrace`) override
            // these via `warm_state.retrace_limit()` /
            // `.max_retrace_guards()` before driving the optimizer,
            // matching the upstream MemoryManager hookup. These
            // fallback defaults (5/15) are only consulted by test
            // fixtures that bypass pyjitpl — they match the
            // `rpython/jit/metainterp/optimizeopt/test/test_util.py`
            // `n = 5` / `n = 15` test values.
            retrace_limit: 5,
            max_retrace_guards: 15,
            imported_state: None,
            emit_start_label: true,
            final_exported_state: None,
            final_exported_label_source_positions: None,
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: SnapshotFrameSizes::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: SnapshotFramePcs::new(),
            snapshot_recorder: None,
            all_descrs: std::sync::Arc::new(Vec::new()),
            quasi_immutable_deps: Vec::new(),
            trace_inputargs: Vec::new(),
            trace_inputarg_boxes: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_patchguardop: None,
            next_global_opref: 0,
            callinfocollection: None,
            call_pure_results: crate::optimizeopt::util::args_dict(),
            cpu: crate::cpu::default_cpu(),
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            vrefinfo: None,
            // Same default as `Optimizer::new` / `PARAMETERS.pureop_historylength`.
            // Tests construct this without a WarmState pin; 0 makes
            // `RecentPureOps` an empty ring and panics on the first CSE insert.
            pureop_historylength: crate::jit::PARAMETERS.pureop_historylength as usize,
            phase2_input_ops_seed: None,
            compile_snapshot_root_slots: None,
            compile_resume_memos_slot: None,
            compile_short_preamble_producer_slot: None,
            persistent_snapshot_root_slots: Vec::new(),
            enable_opts: crate::optimizeopt::optimizer::ENABLE_ALL_OPTS
                .split(':')
                .filter(|name| !name.is_empty())
                .map(String::from)
                .collect(),
        }
    }

    fn phase_optimizer(
        &self,
        vable: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
    ) -> crate::optimizeopt::optimizer::Optimizer {
        crate::optimizeopt::optimizer::Optimizer::build_opt_chain(&self.enable_opts, vable)
    }

    fn collect_snapshot_const_ptr_slots(maps: &mut [&mut SnapshotBoxes]) -> Vec<usize> {
        let mut slots = Vec::new();
        for map in maps {
            for boxes in map.iter_mut().flatten() {
                for sb in boxes {
                    if let majit_ir::OpRef::ConstPtr(gcref) = sb.opref
                        && !gcref.is_null()
                    {
                        slots.push((&mut sb.opref as *mut majit_ir::OpRef) as usize);
                    }
                }
            }
        }
        slots
    }

    fn replace_compile_snapshot_roots(&self, slots: Vec<usize>) {
        if let Some(addr) = self.compile_snapshot_root_slots {
            // SAFETY: pyjitpl installs this address from
            // `MetaInterp.compile_snapshot_refs` for the duration of one
            // compile. Unroll phases run on the same thread as the registered
            // root walker.
            let mut all = self.persistent_snapshot_root_slots.clone();
            all.extend(slots);
            unsafe {
                *(addr as *mut Vec<usize>) = all;
            }
        }
    }

    fn clear_compile_snapshot_roots(&self) {
        self.replace_compile_snapshot_roots(Vec::new());
    }

    /// Publish the address of `optimizer.short_preamble_producer` to the
    /// registered root walker, and restore the prior publication when the
    /// returned guard drops.
    ///
    /// The guard is not optional. The published address points into the
    /// caller's `Optimizer` local, which dies when the unroll call returns;
    /// `CompileSnapshotRootsGuard` clears the slot only when the enclosing
    /// compile entry returns, which is later. Between those two points a root
    /// walk would read the address of a dropped local. Bind the guard after
    /// the optimizer so it withdraws the address first.
    #[must_use]
    fn publish_short_preamble_producer(
        &self,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ) -> PublishedShortPreambleProducer {
        let mut previous = None;
        if let Some(addr) = self.compile_short_preamble_producer_slot {
            // SAFETY: pyjitpl installs this address from
            // `MetaInterp.compile_short_preamble_producer` for the duration
            // of one compile. Unroll phases run on the same thread as the
            // registered root walker.
            unsafe {
                let slot = &mut *(addr as *mut Option<usize>);
                previous = *slot;
                *slot = Some(
                    (&mut optimizer.short_preamble_producer
                        as *mut Option<
                            crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder,
                        >) as usize,
                );
            }
        }
        optimizer.published_short_preamble_producer_slot =
            self.compile_short_preamble_producer_slot;
        self.register_resume_memo(optimizer);
        PublishedShortPreambleProducer {
            slot: self.compile_short_preamble_producer_slot,
            previous,
        }
    }

    /// Hand `optimizer`'s resume memo to the metainterpreter's root area for
    /// the rest of this compile.
    ///
    /// optimizer.py gives every `Optimizer` its own `ResumeDataLoopMemo`,
    /// and `store_final_boxes_in_guard` fills that memo's `rd_consts` with live
    /// GCREFs as it numbers each guard. RPython reaches them through the
    /// optimizer the compiling frame holds; here the walker reaches only what
    /// the metainterpreter published, so each optimizer that numbers guards
    /// registers its own memo — `unroll.py optimize_preamble`'s as well as
    /// `optimize_peeled_loop`'s.
    ///
    /// The handle is `Weak`, so a dropped `Optimizer` needs no withdrawal:
    /// `MetaInterp::walk_rd_consts_refs` keeps only the ones that upgrade.
    fn register_resume_memo(&self, optimizer: &crate::optimizeopt::optimizer::Optimizer) {
        let Some(addr) = self.compile_resume_memos_slot else {
            return;
        };
        // SAFETY: pyjitpl installs this address from
        // `MetaInterp.compile_resume_memos` for the duration of one compile,
        // and `CompileSnapshotRootsGuard` empties the vector before the
        // compile returns. Unroll phases run on the same thread as the
        // registered root walker.
        unsafe {
            (*(addr as *mut Vec<crate::resume::LiveResumeMemo>))
                .push(std::rc::Rc::downgrade(&optimizer.resumedata_memo));
        }
    }

    /// Optimize the preamble (first iteration) of a loop trace.
    /// Returns the optimized preamble ops + the peeled loop ops.
    ///
    /// Upstream `unroll.py optimize_preamble` has no
    /// SpeculativeError catch — by construction the preamble's
    /// gcrefs are concrete runtime values from the recorded
    /// interpreter, so `cpu.protect_speculative_*` always passes.
    /// A SpeculativeError here indicates a genuine bug (concrete
    /// gcref failed type validation) and should propagate.
    pub fn optimize_preamble(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        // A fresh optimizer defaults the capability to present; carry the
        // backend's answer over as the two-phase path does, or these callers
        // get UINT_MUL_HIGH on a backend that has none.
        optimizer.supports_efficient_uint_mul_high = self.supports_efficient_uint_mul_high;
        optimizer.add_pass(Box::new(OptUnroll::new()));
        optimizer.propagate_all_forward(ops)
    }

    /// unroll.py `optimize_peeled_loop(trace)`
    /// Optimize the loop body AFTER preamble peeling.  The peeled
    /// preamble has already established the type/class/bounds
    /// information; this method optimizes the repeating body, with
    /// speculative imports that may fail type validation —
    /// `unroll.py:119-123 except SpeculativeError: raise InvalidLoop`.
    pub fn optimize_peeled_loop(&mut self, ops: &[Op]) -> Vec<Op> {
        let supports_efficient_uint_mul_high = self.supports_efficient_uint_mul_high;
        with_speculative_to_invalid_loop(|| {
            let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
            optimizer.supports_efficient_uint_mul_high = supports_efficient_uint_mul_high;
            optimizer.propagate_all_forward(ops)
        })
    }

    /// unroll.py: jump_to_preamble(cell_token, jump_op).
    ///
    /// Redirect the closing JUMP to the preamble entry token
    /// (`target_tokens[0]`, `virtual_state=None`). Upstream
    /// `copy_and_change` keeps the arglist; majit then reshapes it to
    /// `start_state.renamed_inputargs` so the assembler remap sees the
    /// same JUMP/LABEL arity `x86/regalloc.py`
    /// `_compute_hint_locations_from_descr` asserts for a compiled target.
    pub fn jump_to_preamble(
        body_ops: &[majit_ir::OpRc],
        preamble_target: &std::sync::Arc<TargetToken>,
        preamble_args: &[OpRef],
    ) -> Vec<majit_ir::OpRc> {
        assert!(
            preamble_target.attrs().virtual_state.is_none(),
            "jump_to_preamble expects the start/preamble target token"
        );
        let mut result = body_ops.to_vec();
        if let Some(idx) = result.iter().rposition(|op| op.opcode == OpCode::Jump) {
            // Re-clone before the descr write: the input Rc may still be
            // registered in the optimizer's producer stores.
            let mut jump = (*result[idx]).clone();
            jump.setdescr(preamble_target.as_jump_target_descr());
            apply_preamble_shape_to_jump(&mut jump, preamble_args);
            result[idx] = OpRc::new(jump);
        }
        result
    }

    fn ensure_preamble_target_token(&mut self) {
        if self
            .target_tokens
            .first()
            .is_some_and(|token| token.attrs().virtual_state.is_none())
        {
            return;
        }
        self.target_tokens.insert(0, TargetToken::new_preamble(0));
    }

    /// unroll.py: optimize_trace(trace, call_pure_results)
    /// Full trace optimization: peel → optimize preamble → optimize body.
    /// Returns the optimized peeled+body trace.
    pub fn optimize_trace(&mut self, ops: &[Op]) -> Vec<Op> {
        let result = self.optimize_preamble(ops);
        // After peeling, extract short preamble from the result.
        let sp = crate::optimizeopt::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// unroll.py: optimize_trace_with_constants
    /// Same as optimize_trace but with known constants.
    pub fn optimize_trace_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
    ) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.supports_efficient_uint_mul_high = self.supports_efficient_uint_mul_high;
        optimizer.add_pass(Box::new(OptUnroll::new()));
        let result = optimizer.optimize_with_constants(ops, constants);
        let sp = crate::optimizeopt::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// optimize_trace with constants AND explicit num_inputs.
    /// compile.py: compile_loop → optimize with preamble peeling.
    pub fn optimize_trace_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
    ) -> (Vec<majit_ir::OpRc>, usize) {
        let ops_rc: Vec<OpRc> = ops.iter().cloned().map(OpRc::new).collect();
        self.optimize_trace_with_constants_and_inputs_vable(&ops_rc, constants, num_inputs, None)
            .expect("optimize_trace_with_constants_and_inputs: unexpected InvalidLoop")
    }

    /// compile.py: compile_loop — 2-phase preamble peeling.
    /// compile.py:275-338: 2-phase preamble peeling (RPython parity).
    ///
    /// Phase 1 (optimize_preamble): full pipeline on trace → preamble_ops.
    /// export_state: capture the preamble's exported optimizer state.
    /// Phase 2 (optimize_peeled_loop): import_state + full pipeline → body_ops.
    /// Assembly: [preamble_no_jump] + Label(label_args) + [body_with_jump].
    pub(crate) fn optimize_trace_with_constants_and_inputs_vable(
        &mut self,
        ops: &[OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
    ) -> Result<(Vec<majit_ir::OpRc>, usize), crate::optimize::InvalidLoop> {
        self.optimize_trace_with_constants_and_inputs_vable_out(
            ops,
            constants,
            num_inputs,
            vable_config,
            None,
        )
    }

    /// Same as optimize_trace_with_constants_and_inputs_vable but with an
    /// output parameter for Phase 1 results. RPython compile.py:278-294
    /// parity: Phase 1 results (preamble_ops + exported_state) are written
    /// to `phase1_out` before Phase 2 starts. If Phase 2 panics, the caller
    /// still has the Phase 1 results for retrace_needed.
    pub(crate) fn optimize_trace_with_constants_and_inputs_vable_out(
        &mut self,
        ops: &[OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
        phase1_out: Option<&mut Option<(Vec<majit_ir::OpRc>, ExportedState)>>,
    ) -> Result<(Vec<majit_ir::OpRc>, usize), crate::optimize::InvalidLoop> {
        // compile.py: if imported_state is pre-set (compile_retrace path),
        // skip Phase 1 and go directly to Phase 2 with the imported state.
        let (mut exported_state, consts_p1, p1_ops) = if let Some(pre_imported) =
            self.imported_state.take()
        {
            // RPython uses object identity for Boxes, so a TraceIterator over a
            // retrace can never numerically collide with the already optimized
            // partial preamble. Majit's OpRef is the identity; when Phase 1 is
            // skipped, recover the partial preamble high-water from the imported
            // state before allocating Phase 2 input/result OpRefs.
            self.next_global_opref = self
                .next_global_opref
                .max(num_inputs as u32)
                .max(pre_imported.opref_high_water());
            // Retrace path: Phase 1 already done; the preamble producers
            // live in the imported state's `partial_trace_operations`
            // (export records them there from `phase1_emit_ops`, #104). The
            // non-skip path populates `self.phase1_emit_ops` from `opt_p1`;
            // restore the same pool here so Phase 2's producer lookup
            // (`new_operations ∪ phase1_emit_ops ∪ resop_refs`) can reach a
            // const-folded preamble producer that no other store carries.
            // RPython keeps box `_forwarded` reachable across the retrace;
            // this is the pyre analog. (Readers dedup, so double-coverage
            // with `resop_refs` is idempotent.)
            self.phase1_emit_ops = pre_imported.partial_trace_operations.clone();
            // RPython: same Optimizer persists patchguardop. Recover here.
            if self.phase1_patchguardop.is_none() {
                self.phase1_patchguardop = pre_imported.patchguardop.clone();
            }
            (pre_imported, constants.clone(), Vec::new())
        } else {
            // ── Phase 1: PreambleCompileData.optimize() ──
            // ── Phase 1: optimize_preamble (compile.py:275-276) ──
            let mut consts_p1 = constants.clone();
            let mut opt_p1 = self.phase_optimizer(vable_config.clone());
            self.register_resume_memo(&opt_p1);
            opt_p1.all_descrs = std::mem::take(&mut self.all_descrs);
            opt_p1.callinfocollection = self.callinfocollection.clone();
            self.pin_inner_optimizer(&mut opt_p1);
            opt_p1.trace_inputargs = self.trace_inputargs.clone();
            opt_p1.snapshot_boxes = self.snapshot_boxes.clone();
            opt_p1.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
            opt_p1.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
            opt_p1.snapshot_vref_boxes = self.snapshot_vref_boxes.clone();
            opt_p1.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
            self.replace_compile_snapshot_roots(Self::collect_snapshot_const_ptr_slots(&mut [
                &mut opt_p1.snapshot_boxes,
                &mut opt_p1.snapshot_vable_boxes,
                &mut opt_p1.snapshot_vref_boxes,
            ]));
            opt_p1.call_pure_results = self.call_pure_results.clone();
            // RPython optimize_preamble (unroll.py): flush=False.
            // JUMP/FINISH is NOT sent through the pass pipeline; it's
            // returned in info.jump_op for Phase 2 to consume.
            opt_p1.skip_flush = true;
            // RPython unroll.py `optimize_preamble` calls
            // `propagate_all_forward(trace.get_iter())`. `trace.get_iter()`
            // is a fresh `TraceIterator` whose `next()` produces a freshly
            // allocated `cls()` ResOperation for every visited op.
            //
            // Phase 1 routes the input ops through `TraceIterator::new`
            // with `start_fresh = 0`. The recorder emits ops at
            // monotonically increasing positions starting from
            // `num_inputs` (recorder.rs `record_op` uses `op_count` for
            // BOTH inputargs and ops, and ops follow inputargs), and
            // `TraceIterator::next` allocates fresh OpRefs from `_fresh`
            // which is also seeded at `num_inputs` after the inputarg
            // pre-seed loop. Both void and non-void ops advance `_fresh`
            // (see opencoder.rs::next), so the freshly produced OpRef
            // sequence is bit-identical to the input — this wrap is a
            // structural alignment with RPython's `trace.get_iter()`
            // call site, not a functional change.
            // opencoder.py `inputarg_from_tp(arg.type)` — fresh inputargs
            // are typed via `self.trace_inputargs`, the recorder-supplied
            // Box list (length must equal `num_inputs`). Derive the &[Type]
            // surface TraceIterator expects from each Box's variant tag
            // (resoperation.py/727/739).
            debug_assert_eq!(self.trace_inputargs.len(), num_inputs);
            let p1_inputarg_types: Vec<majit_ir::Type> = self
                .trace_inputargs
                .iter()
                .map(|op| op.ty().expect("inputarg OpRef must carry box.type"))
                .collect();
            // `TraceIterator` walks any `Borrow<Op>` slice, and `next` already
            // performs PyPy's `cls()` per-op fresh allocation (opencoder.py),
            // so the recorder ops feed it directly — a pre-wrap into
            // `Vec<OpRc>` would allocate a second copy of every op that
            // nothing reads (the iterator caches the op it mints, not its
            // source).
            let mut p1_iter = crate::opencoder::TraceIterator::new(
                ops,
                0,
                ops.len(),
                None,
                &p1_inputarg_types,
                0, // start_fresh = 0 — inputargs at [0..num_inputs)
            );
            let mut p1_ops_in: Vec<majit_ir::OpRc> = Vec::with_capacity(ops.len());
            while let Some(op) = p1_iter.next() {
                p1_ops_in.push(op);
            }
            let p1_iter_fresh_hw = p1_iter._fresh;
            opt_p1.trace_inputarg_boxes = p1_iter.inputargs.clone();
            if let Some(recorder) = self.snapshot_recorder {
                // SAFETY: the caller keeps the recorder alive across optimize.
                opt_p1.byte_bridge_resume = Some(crate::recorder::ByteBridgeResume::from_recorder(
                    unsafe { &*recorder },
                    std::mem::take(&mut p1_iter._cache),
                ));
            }
            // compile.py `PreambleCompileData(trace, jumpargs, ...)` —
            // the recorded JUMP arglist is the preamble's `runtime_boxes`
            // (live_arg_boxes captured at the merge point). Capture it into a
            // local here; `opt_p1.runtime_boxes` cannot carry it because
            // `setup()` clears that field at the start of every optimize run
            // (optimizer.rs `self.runtime_boxes.clear()`). Threaded into the
            // exported state below so the peeled-loop close reads it as
            // `state.runtime_boxes` (unroll.py:105 → :153/166) rather than the
            // peeled body's own jump args.
            let recorded_jump_args = fold_recorded_jump_args(
                ops.iter()
                    .rfind(|op| op.opcode == OpCode::Jump)
                    .map(|op| {
                        op.args_slice()
                            .iter()
                            .map(|a| match a.get_value() {
                                Some(value) if !matches!(value, Value::Void) => {
                                    OpRef::const_inline_from_value(&value)
                                }
                                _ => a.to_opref(),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                &self.trace_inputarg_boxes,
                ops,
            );
            // Hand opt_p1 the per-iter operand pool that p1_iter
            // allocated. trace.get_iter() per-call
            // inputarg_from_tp(...) / cls() — each phase optimizes against a
            // fresh Box identity set so _forwarded mutations cannot alias
            // across phases.
            //
            // Const operands are NOT cached: `get_box_replacement_operand_opt`
            // allocates a fresh const operand per call from `const_pool`
            // (`history.py` ConstInt(value) per-call-site parity).
            // opt_p1's entry path seeds `const_pool` from the shared
            // `constants` map (`optimizer.rs`'s
            // `lower_typed_constants_to_const_pool`).
            // unroll.py `optimize_preamble` has no
            // SpeculativeError catch — Phase 1 corresponds to the
            // preamble, whose gcrefs are concrete runtime values
            // from the recorded interpreter and never raise
            // SpeculativeError under correct construction.  A raise
            // here is a real bug and must propagate.
            // Phase 1's input ops are the same recorder ops the seed names,
            // so its `input_ops` can come from the seed too — only the read
            // source changes; `_forwarded` writes are untouched. No
            // forwarding yet at Phase 1 setup, so the seed is trivially the
            // authoritative source here.
            // The iterator's reminted ops carry the `inputarg_from_tp`
            // InputArg Rcs. Seed them so `ensure_inputarg_bindings`
            // reuses those objects instead of minting a second host.
            opt_p1.explicit_input_ops_seed = Some(p1_ops_in.clone());
            let p1_ops =
                opt_p1.run_optimize_from_inputs(&p1_ops_in, &mut consts_p1, num_inputs, false)?;
            merge_quasi_immutable_deps(
                &mut self.quasi_immutable_deps,
                &opt_p1.quasi_immutable_deps,
            );
            let p1_ni = opt_p1.final_num_inputs();

            match opt_p1.exported_loop_state.take() {
                Some(mut state) => {
                    // unroll.py `export_state(..., runtime_boxes, ...)` —
                    // carry the recorded JUMP args into ExportedState so the
                    // peeled-loop close passes them to generate_guards as
                    // `state.runtime_boxes` (unroll.py:153/166).
                    state.runtime_boxes = recorded_jump_args.clone();
                    self.final_exported_label_source_positions =
                        Some(state.label_source_positions.clone());
                    // end_arg_types is already populated by
                    // `Optimizer::optimize_with_constants_and_inputs_at`
                    // using the optimizer-visible `ctx.opref_type()` (see
                    // `optimizer.rs`). Overwriting it here with
                    // `opcode.result_type()` + `Type::Ref` default would
                    // retype Phase 1 inputarg OpRefs (absent from `p1_ops`)
                    // as `Ref`, which later feeds the cross-type forward
                    // assertion in `OptContext::make_equal_to`.
                    // RPython Phase 1 → Phase 2 heap cache transfer:
                    // RPython does NOT serialize heap cache in ExportedState.
                    // HeapOps in the short preamble are replayed during Phase 2's
                    // inline_short_preamble, populating the heap cache naturally
                    // through OptHeap. serialize_optheap is only for bridgeopt.
                    // opencoder.py _index parity: Phase 2's TraceIterator
                    // must allocate fresh boxes ABOVE Phase 1's high water
                    // mark so the two phases' OpRef namespaces are disjoint
                    // (RPython relies on Python identity to distinguish them;
                    // majit relies on disjoint integer ranges). The high
                    // water is `final_ctx.next_pos` after Phase 1 emit, with
                    // a floor of `num_inputs` for empty traces.
                    // Phase 1's emit-side high water (`final_ctx.next_pos`)
                    // reflects only the positions `ctx.reserve_pos` handed
                    // out; the per-iteration TraceIterator additionally
                    // allocates fresh OpRefs via its `_fresh` counter and
                    // those values can exceed `next_pos` for traces where
                    // Phase 1 dropped or folded many ops. Taking the max
                    // with `p1_iter_fresh_hw` keeps Phase 2's inputarg base
                    // strictly above every OpRef Phase 1 ever allocated,
                    // so Phase 2's own fresh OpRefs cannot collide with
                    // positions in `phase1_emit_ops` / typed snapshots.
                    self.next_global_opref = opt_p1
                        .final_ctx
                        .as_ref()
                        .map(|c| c.next_pos)
                        .unwrap_or(num_inputs as u32)
                        .max(num_inputs as u32)
                        .max(p1_iter_fresh_hw);
                    // RPython Box type parity: Phase 1's emit ops carry
                    // `op.type_` intrinsically; Phase 2's `op_at` reads it
                    // directly to resolve cross-phase OpRefs that appear as
                    // imported_label_args / fail_args / record_same_as
                    // sources (history.py:220 parity).
                    self.phase1_emit_ops = std::mem::take(&mut opt_p1.phase1_emit_ops);
                    // RPython: same Optimizer instance keeps patchguardop.
                    if self.phase1_patchguardop.is_none() {
                        self.phase1_patchguardop = opt_p1.patchguardop.clone();
                    }
                    state.patchguardop = opt_p1.patchguardop.clone();
                    self.all_descrs = std::mem::take(&mut opt_p1.all_descrs);
                    // Box.type parity: retain Phase 1's renamed_inputargs so
                    // the backend can read the reduced LABEL's declared types
                    // off each typed InputArg OpRef. Clone only the field we
                    // need to avoid keeping Phase 1 short preamble data alive
                    // longer than before.
                    let mut final_exported_state = ExportedState {
                        end_args: Vec::new(),
                        label_source_positions: state.label_source_positions.clone(),
                        next_iteration_args: Vec::new(),
                        end_arg_types: Vec::new(),
                        virtual_state: state.virtual_state.clone(),
                        exported_infos: indexmap::IndexMap::new(),
                        exported_short_boxes: Vec::new(),
                        const_short_boxes: Vec::new(),
                        short_boxes: Vec::new(),
                        short_preamble: None,
                        renamed_inputargs: state.renamed_inputargs.clone(),
                        short_inputargs: Vec::new(),
                        short_inputarg_refs: Vec::new(),
                        runtime_boxes: Vec::new(),
                        patchguardop: None,
                        quasi_immutable_deps: state.quasi_immutable_deps.clone(),
                        phase1_emit_high_water: self.next_global_opref,
                        partial_trace_inputargs: Vec::new(),
                        partial_trace_operations: Vec::new(),
                        short_box_producer_roots: Vec::new(),
                        rooted_refs: Vec::new(),
                        rooted_const_ptr_slots: Vec::new(),
                        shadow_stack_base: 0,
                    };
                    final_exported_state.root_all_gcrefs();
                    self.final_exported_state = Some(final_exported_state);
                    (state, consts_p1, p1_ops)
                }
                None => {
                    *constants = consts_p1;
                    // Take back the descriptor table grown during Phase 1's pass
                    // (`ensure_descr_index` appends at guard-resume serialization),
                    // mirroring the Some branch's restore above. Without this the
                    // non-peeled path drops those descriptors and publishes a stale
                    // `all_descrs` back over the global table.
                    self.all_descrs = std::mem::take(&mut opt_p1.all_descrs);
                    // RPython: compile_loop uses flush=True — terminal op
                    // (Finish/Jump) goes through the pass pipeline normally.
                    // majit: flush=False stores it in terminal_op; restore it
                    // here for non-peeled traces that return directly.
                    let mut ops = p1_ops;
                    if let Some(terminal) = opt_p1.terminal_op.take() {
                        ops.push(OpRc::new(terminal));
                    }
                    // `compile.py` `jitcell_token.target_tokens = [target_token]`
                    // / `:290` `jitcell_token.target_tokens = [start_descr]` —
                    // one preamble target token published unconditionally on
                    // whichever of the two paths ran. Those are the only two
                    // assignments of `target_tokens` upstream, and both are
                    // inside `compile_simple_loop` / `compile_loop`, so this
                    // says nothing about `compile_retrace`. Phase 2 is bypassed
                    // here (no exported_loop_state) but the loop still
                    // compiles, so mirror the registration so that
                    // `JitCellToken.target_tokens` is non-empty at the
                    // `has_compiled_targets` (`pyjitpl.py`) read site.
                    self.ensure_preamble_target_token();
                    let loop_arity = closing_loop_contract_arity(&ops, p1_ni);
                    self.clear_compile_snapshot_roots();
                    return Ok((ops, loop_arity));
                }
            }
        };
        self.clear_compile_snapshot_roots();
        // unroll.py end_args carry type via Box; export_state already
        // populated `exported_state.end_arg_types` from ctx.
        // RPython parity: Phase 2 needs patchguardop from Phase 1's
        // GuardFutureCondition (unroll.py:333). Extract before dropping opt_p1.
        let p1_patchguardop = exported_state.patchguardop.clone();

        self.ensure_preamble_target_token();
        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        let body_num_inputs = num_inputs;

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] preamble peeling: {} virtual(s), phase1 end_args={} p1_patchguardop={}",
                exported_state
                    .virtual_state
                    .state
                    .iter()
                    .filter(|s| s.is_virtual())
                    .count(),
                exported_state.end_args.len(),
                p1_patchguardop
                    .as_ref()
                    .map(|p| p.rd_resume_position())
                    .unwrap_or(-99),
            );
        }

        // opencoder.py:259-404 parity: Phase 2 uses the same ops as Phase 1.
        // RPython TraceIterator creates fresh Box objects per phase — each
        // iterator has its own _cache, so Phase 2 results never collide with
        // Phase 1. In majit, Phase 2 gets a separate OptContext, achieving
        // the same isolation.
        let mut consts_p2 = consts_p1.clone();

        let mut opt_p2 = self.phase_optimizer(vable_config.clone());
        // Bound after `opt_p2` so it drops first and withdraws the published
        // address while the optimizer it names is still alive.
        let _published_short_preamble_producer = self.publish_short_preamble_producer(&mut opt_p2);
        opt_p2.all_descrs = std::mem::take(&mut self.all_descrs);
        opt_p2.callinfocollection = self.callinfocollection.clone();
        self.pin_inner_optimizer(&mut opt_p2);
        // Phase 2 (peeled-loop pass) runtime value seed.
        // The remap is deferred until after the Phase 2 `TraceIterator`
        // builds its `_cache`, because both inputargs AND body op results
        // are reminted to fresh OpRefs (opencoder.py / :399-401).
        // Using the iterator's `_cache` as the lookup table is the
        // RPython-orthodox equivalent of Box identity parallelism between
        // `boxes` and `runtime_boxes` (virtualstate.py generate_guards): each
        // Phase 1 raw position maps to the same Phase 2 box the iterator
        // allocated for it.  See the remap block after `p2_cache` is
        // built below.
        // `trace_inputargs` stays a clone: it is re-read below at the Phase 2
        // TraceIterator setup (`p2_inputarg_types`) and by the earlier
        // `debug_assert_eq!` on its length.
        opt_p2.trace_inputargs = self.trace_inputargs.clone();
        // Clone, like Phase 1. compile_loop_body moves the recorded maps
        // onto this UnrollOptimizer and, on InvalidLoop, takes them back
        // for compile_simple_loop. Phase 2 used to `mem::take` them, so
        // a peel that raised InvalidLoop left the retry with empty
        // snapshot_boxes and `store_final_boxes_in_guard` panicked on
        // GuardIsnull resume_pos=0 (tuple_slice_index_rooting).
        // phase1_emit_ops is not read by that retry.
        opt_p2.phase1_emit_ops = std::mem::take(&mut self.phase1_emit_ops);
        self.clone_snapshots_onto(&mut opt_p2);
        // compile.py / optimizer.py share the attempt's args_dict itself.
        opt_p2.call_pure_results = self.call_pure_results.clone();
        // RPython: same Optimizer instance keeps patchguardop across phases.
        // Phase 1 processes GUARD_FUTURE_CONDITION (from close_loop_args_at)
        // which sets patchguardop. optimizer.py parity — no synthetic
        // fallback; the actual GFC provides rd_resume_position.
        opt_p2.patchguardop = self.phase1_patchguardop.clone();
        // gcreftracer.py parity: root GcRef values on the shadow stack.
        // RPython: single Python object — GC traces automatically.
        // Rust: LIFO shadow stack requires longer-lived roots at lower depth.
        //
        // Order: (1) phase1_out clone rooted first (survives beyond this
        // function for retrace — pyre-specific panic safety backup).
        // (2) original rooted second (lives until opt_p2 drops at Phase 2
        // end — shorter-lived, higher depth, dropped first).
        if let Some(out) = phase1_out {
            let mut backup = exported_state.clone();
            backup.root_all_gcrefs();
            *out = Some((p1_ops.clone(), backup));
        }
        exported_state.root_all_gcrefs();
        opt_p2.imported_loop_state = Some(exported_state);
        // Set imported_virtuals so Phase 2 intercepts GetfieldGcR(pool)
        // and sets up VirtualStruct PtrInfo for the imported head.
        // Virtual structure is derived from VirtualState (ExportedState).
        opt_p2.imported_virtuals =
            build_imported_virtuals_from_state(opt_p2.imported_loop_state.as_ref().unwrap());
        // RPython: propagate_all_forward(trace, flush=False) for Phase 2.
        // Don't flush lazy sets — virtuals remain virtual until JUMP handling.
        opt_p2.skip_flush = true;
        // RPython parity: Phase 2 DOES virtualize New(). Guard recovery uses
        // rd_virtuals (generated by finalize_guard_resume_data)
        // for virtual materialization on guard failure.
        // Previously disabled (set_phase2(true)) due to missing rd_virtuals;
        // now enabled after compile.rs rd_virtuals→rd_virtuals generation.
        // RPython parity: Phase 2 imports heap cache via short preamble
        // RPython: Phase 2 heap cache is populated by inline_short_preamble
        // replaying HeapOps through send_extra_operation.
        if crate::majit_log_enabled() {
            let gc_before = ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2 input: {} ops, {} guards, body_ni={}",
                ops.len(),
                gc_before,
                body_num_inputs
            );
        }

        // opencoder.py TraceIterator parity for Phase 2.
        //
        // RPython's `optimize_peeled_loop` calls `trace.get_iter()` which
        // constructs a FRESH TraceIterator whose `__init__` allocates new
        // Box objects for every inputarg (inputarg_from_tp) and whose
        // `next()` allocates new cls() ResOperation instances for every
        // emitted op. Each iteration over the same trace produces a
        // completely disjoint set of Python identities; the cache
        // `_cache[raw_position]` records the per-iteration fresh box so
        // later references resolve to the iteration-local identity.
        //
        // majit's `OpRef::from_raw(u32)` IS the identity, so "fresh per iteration"
        // means disjoint integer ranges. Phase 2 must not emit op results
        // at OpRefs that collide with Phase 1's emitted positions — doing
        // so reintroduces the box-identity collision that the reactive
        // check at `mod.rs::emit` (collision detection + forwarding
        // redirect) currently compensates for.
        //
        // This step (Commit D2 of the Box identity plan): run Phase 2
        // ops through TraceIterator with `start_index = next_global_opref`,
        // which gives BOTH inputargs AND op results fresh OpRefs in a
        // disjoint range. Phase 2 inputargs live at
        // `[next_global_opref..next_global_opref+body_num_inputs)` and
        // op results at `[next_global_opref+body_num_inputs..)`. This is
        // the RPython-literal model where each TraceIterator produces
        // freshly allocated InputArg and ResOp Python instances for
        // every iteration — `import_state`'s `assert source is not
        // target` (unroll.py:483) now holds by construction because the
        // Phase 2 source slot OpRefs are always distinct from any
        // Phase 1 end_arg OpRef.
        //
        // After optimization, Phase 2 output is post-translated back to
        // the shared `[0..body_num_inputs)` inputarg layout for the
        // final assembly (see the `shift_back` pass below). The
        // assembly and downstream consumers continue to assume the
        // shared-inputarg layout; only the Phase 2 optimizer internals
        // see the disjoint range.
        let phase2_inputarg_base = self.next_global_opref.max(body_num_inputs as u32);
        // opencoder.py:264 `inputarg_from_tp(arg.type)` — same per-arg types
        // as Phase 1 (Phase 2 walks the body half of the same trace).
        debug_assert_eq!(self.trace_inputargs.len(), body_num_inputs);
        let p2_inputarg_types: Vec<majit_ir::Type> = self
            .trace_inputargs
            .iter()
            .map(|op| op.ty().expect("inputarg OpRef must carry box.type"))
            .collect();
        // Fed straight to `TraceIterator` (a `Borrow<Op>` slice); `next` mints
        // the fresh per-op copy, so a pre-wrap would only duplicate it.
        let mut iter = crate::opencoder::TraceIterator::new(
            ops,
            0,
            ops.len(),
            None,
            &p2_inputarg_types,
            phase2_inputarg_base, // fresh inputargs at [phase2_inputarg_base..)
        );
        let mut p2_ops_in: Vec<majit_ir::OpRc> = Vec::with_capacity(ops.len());
        while let Some(op) = iter.next() {
            p2_ops_in.push(op);
        }
        let p2_high_water = iter._fresh;
        opt_p2.trace_inputarg_boxes = iter.inputargs.clone();
        let p2_cache = iter._cache;
        // opencoder.py `_get(self, i)` parity. `p2_cache[raw_pos]`
        // holds the fresh per-iteration box for every Phase 1 input/op
        // position that Phase 2's TraceIterator walks. snapshot_boxes /
        // snapshot_vable_boxes are populated during Phase 1 against
        // those same Phase 1 positions, so each entry must hit the
        // cache. A miss means the snapshot references a Phase 1 OpRef
        // that does not appear in Phase 2's input ops (e.g. a stale
        // reference to a Phase 1-DCE'd op). The previous
        // `unwrap_or(opref)` silently leaked such stale OpRefs into
        // Phase 2 namespace; production probes (nbody / fannkuch /
        // fib_loop / fib_recursive / spectral_norm) show zero misses,
        // so promote to a strict panic matching RPython's _get assert.
        // A miss has two shapes and they call for different fixes, so name
        // which one fired instead of leaving it to be inferred from
        // `cache_len`.  `TraceIterator::new` sizes the cache from the highest
        // raw position appearing as an op result, an argument or a failarg in
        // the range it walks -- snapshot feeds are not consulted -- so a
        // position ABOVE that is one no operation in the trace mentions at
        // all, while a position INSIDE it with an empty slot is one the range
        // covers and no operation produced.  The feed and the frame the entry
        // sits in are printed for the same reason: the three feeds are
        // remapped by one closure and the message otherwise cannot say which.
        let translate_opref = |feed: &str, frame: usize, opref: OpRef| -> OpRef {
            if opref.is_none() || opref.is_constant() {
                return opref;
            }
            p2_cache
                .get(opref.raw() as usize)
                .and_then(|slot| slot.as_ref())
                .map(|b| b.to_opref())
                .unwrap_or_else(|| {
                    let shape = if opref.raw() as usize >= p2_cache.len() {
                        "above every position the trace mentions"
                    } else {
                        "inside the range, no operation produced it"
                    };
                    // Counted here rather than beside the closure: the happy
                    // path walks every snapshot entry and must not pay two
                    // scans of the cache for a message it never prints.
                    let cache_filled = p2_cache.iter().filter(|slot| slot.is_some()).count();
                    let cache_high = p2_cache
                        .iter()
                        .rposition(std::option::Option::is_some)
                        .map_or(-1i64, |i| i as i64);
                    panic!(
                        "phase2 snapshot remap cache miss for {opref:?} \
                         in {feed}[{frame}] ({shape}) \
                         (cache_len={} filled={} highest_filled={} \
                          ops={} body_ni={} phase2_inputarg_base={})",
                        p2_cache.len(),
                        cache_filled,
                        cache_high,
                        ops.len(),
                        body_num_inputs,
                        phase2_inputarg_base,
                    )
                })
        };
        for (frame, boxes) in opt_p2.snapshot_boxes.iter_mut().enumerate() {
            for r in boxes.iter_mut().flatten() {
                *r = r.map_opref(|opref| translate_opref("snapshot_boxes", frame, opref));
            }
        }
        for (frame, boxes) in opt_p2.snapshot_vable_boxes.iter_mut().enumerate() {
            for r in boxes.iter_mut().flatten() {
                *r = r.map_opref(|opref| translate_opref("snapshot_vable_boxes", frame, opref));
            }
        }
        for (frame, boxes) in opt_p2.snapshot_vref_boxes.iter_mut().enumerate() {
            for r in boxes.iter_mut().flatten() {
                *r = r.map_opref(|opref| translate_opref("snapshot_vref_boxes", frame, opref));
            }
        }
        self.replace_compile_snapshot_roots(Self::collect_snapshot_const_ptr_slots(&mut [
            &mut opt_p2.snapshot_boxes,
            &mut opt_p2.snapshot_vable_boxes,
            &mut opt_p2.snapshot_vref_boxes,
        ]));
        if let Some(recorder) = self.snapshot_recorder {
            // SAFETY: the caller keeps the recorder alive across optimize.
            opt_p2.byte_bridge_resume = Some(crate::recorder::ByteBridgeResume::from_recorder(
                unsafe { &*recorder },
                p2_cache,
            ));
        }
        // Phase 1's emitted ops are already in Phase 1's emitted
        // namespace `[num_inputs..next_global_opref)`. Phase 2 body may
        // reference these via `imported_label_args`. They are NOT in the
        // `p2_cache` (only raw trace positions are), so leave
        // `phase1_emit_ops` untranslated.
        // Hand opt_p2 the per-iter operand pool that the Phase 2
        // iter allocated. Disjoint from opt_p1's pool — _forwarded mutations
        // recorded against Phase 1 boxes do not alias Phase 2 boxes for the
        // same OpRef raw index, fixing the Rc<Box> split-brain that broke
        // the first 77b attempt.
        //
        // Const operands: see opt_p1 plumb above — fresh per-call from
        // `const_pool`, no dedup.
        // unroll.py:119-123 — Phase 2 (peeled loop) raises
        // SpeculativeError on speculative-fold paths; convert
        // to InvalidLoop so the caller's catch handles it.
        // Seed Phase 2's `input_ops` from the recorder `Rc<Op>` (same `Rc`,
        // same Phase-1 `_forwarded`) so `input_ops` is the authoritative
        // producer store. A `Some(empty)` (cut path) leaves it empty.
        if let Some(seed) = &self.phase2_input_ops_seed {
            opt_p2.explicit_input_ops_seed = Some(seed.clone());
        }
        let p2_ops = with_speculative_to_invalid_loop(|| {
            opt_p2.optimize_with_constants_and_inputs_at(
                &p2_ops_in,
                &mut consts_p2,
                body_num_inputs,
                phase2_inputarg_base, // inputarg_base — Phase 2 inputargs at [phase2_inputarg_base..)
                p2_high_water,
                // Phase 2 ops are fresh-Rc `TraceIterator` wraps; with no
                // explicit seed threaded, `input_ops` stays empty and identity
                // comes from `bind_input_resops` / emit.
                false,
            )
        })?;
        merge_quasi_immutable_deps(&mut self.quasi_immutable_deps, &opt_p2.quasi_immutable_deps);
        self.clear_compile_snapshot_roots();
        // RPython optimizer.py:614-625 freezes op arguments during
        // `_emit_operation`; optimizer.py may then install a Const
        // forwarding on the result, but it never retroactively rewrites the
        // already-emitted op. Keep Phase 2 output in that emit-time shape here.
        // Post-translate Phase 2 output back to the shared-inputarg
        // layout expected by `assemble_peeled_trace_with_jump_args`.
        //
        // Phase 2's OptContext internally used disjoint inputarg OpRefs
        // at `[phase2_inputarg_base..phase2_inputarg_base+body_num_inputs)`
        // so the RPython `import_state` "source is not target"
        // (unroll.py) invariant held by construction. The final
        // assembled trace, however, uses shared body inputargs at
        // `[0..body_num_inputs)` so that the preamble and body share
        // the same inputarg slots — the `Label(label_args)` op at the
        // body boundary binds them to the preamble's end values, and
        // downstream consumers (Cranelift regalloc, resume data) see
        // a single consistent inputarg numbering.
        //
        // This shift only touches op args, fail_args, and the label /
        // source_slot vectors. Op positions (the emitted results) are
        // already in `[p2_high_water..)`, which is above both the
        // shared range `[0..body_num_inputs)` and the Phase 1 emitted
        // range `[num_inputs..next_global_opref)`, so they stay put.
        // D2 shift_back shim removed: the assembly now natively handles
        // disjoint Phase 2 inputarg OpRefs via the inputarg_base parameter.
        // Phase 2 inputarg OpRefs at [phase2_inputarg_base..+body_num_inputs)
        // flow directly into the assembly without translation.
        let body_terminal_op = opt_p2.terminal_op.clone();
        let p2_ni = opt_p2.final_num_inputs();
        self.all_descrs = std::mem::take(&mut opt_p2.all_descrs);
        let label_args = opt_p2
            .imported_label_args
            .clone()
            .expect("phase 2 missing import_state label_args");

        if crate::majit_log_enabled() {
            for op in &p2_ops {
                if op.opcode.is_guard() {
                    let rd_numb_len = op.resolved_rd_numb().map(|s| s.len()).unwrap_or(0);
                    if let Some(fa) = op.guard_fail_args() {
                        let fa_raw: Vec<String> = fa
                            .iter()
                            .map(|a| format!("OpRef::from_raw({})", a.to_opref().raw()))
                            .collect();
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=[{}]",
                            op.opcode,
                            op.pos().get(),
                            op.rd_resume_position(),
                            rd_numb_len,
                            fa_raw.join(", ")
                        );
                    } else {
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=<none>",
                            op.opcode,
                            op.pos().get(),
                            op.rd_resume_position(),
                            rd_numb_len,
                        );
                    }
                }
            }
            let nc = p2_ops
                .iter()
                .filter(|o| o.opcode == OpCode::New || o.opcode == OpCode::NewWithVtable)
                .count();
            let gc = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2: {} ops, {} New, {} guards, p2_ni={}",
                p2_ops.len(),
                nc,
                gc,
                p2_ni
            );
            for (i, op) in p2_ops.iter().enumerate() {
                eprintln!(
                    "[jit] p2[{i}]: {:?} pos={:?} args={:?}",
                    op.opcode,
                    op.pos().get(),
                    op.getarglist()
                );
            }
        }

        // ── unroll.py: finalize + jump_to_existing_trace ──
        let mut imported_short_aliases = opt_p2.imported_short_aliases.clone();
        // finalize_short_preamble: create TargetToken for this loop version
        // RPython parity: short preamble ops reference constant OpRefs from
        // the loop's constant pool. Pass consts_p1 so the ShortPreamble
        // captures (value, type) for each constant, enabling bridges to
        // re-register them in their own pool (RPython embeds Const objects
        // directly in op args; majit uses separate constant pool indices).
        // RPython parity: read back from the same ExportedState that
        // import_state used (Python reference semantics — one object).
        // exported_state was moved into opt_p2.imported_loop_state.
        // Extract needed fields before opt_p2 is borrowed mutably below.
        let (
            exported_vs,
            exported_end_args,
            exported_short_inputargs,
            exported_short_boxes_produced,
            exported_renamed_inputargs,
            exported_runtime_boxes,
        ) = {
            let es = opt_p2
                .imported_loop_state
                .as_ref()
                .expect("imported_loop_state must survive Phase 2");
            (
                es.virtual_state.clone(),
                es.end_args.clone(),
                es.short_inputargs.clone(),
                es.short_boxes.clone(),
                es.renamed_inputargs.clone(),
                es.runtime_boxes.clone(),
            )
        };
        // RPython unroll.py:124-141 performs an extra end-of-preamble forcing
        // pass on the closing JUMP args before finalize_short_preamble():
        //   for a in end_jump.getarglist():
        //       self.force_box_for_end_of_preamble(get_box_replacement(a))
        //   current_vs = self.optunroll.get_virtual_state(end_jump.getarglist())
        //   target_virtual_state = self.optunroll.pick_virtual_state(...)
        //   args = target_virtual_state.make_inputargs(..., force_boxes=True)
        //   for arg in args:
        //       self.force_box(arg)
        //
        // The optimizer-level imported_short_preamble snapshot is taken before
        // this step, so rebuild from the live Phase 2 context here instead of
        // trusting the stale copy.
        let body_jump_args: Vec<OpRef> = body_terminal_op
            .as_ref()
            .map(|jump| jump.args_slice().iter().map(|a| a.to_opref()).collect())
            .or_else(|| {
                p2_ops
                    .iter()
                    .rfind(|op| op.opcode == OpCode::Jump)
                    .map(|jump| jump.args_slice().iter().map(|a| a.to_opref()).collect())
            })
            .unwrap_or_default();
        let (imported_short_preamble_builder, rebuilt_imported_short_preamble) =
            if let Some(mut final_ctx) = opt_p2.final_ctx.take() {
                if !body_jump_args.is_empty() {
                    // unroll.py:126-127 parity:
                    //   for a in end_jump.getarglist():
                    //       self.force_box_for_end_of_preamble(get_box_replacement(a))
                    //
                    // Route through `force_box_for_end_of_preamble` (the
                    // per-box type-gating wrapper) rather than the inner
                    // dispatcher, matching optimizer.py:306-319.
                    let resolved_jump_args: Vec<majit_ir::operand::Operand> = body_jump_args
                        .iter()
                        .filter_map(|&arg| final_ctx.get_box_replacement_operand_opt(arg))
                        .collect();
                    for arg in &resolved_jump_args {
                        let _ = opt_p2.force_box_for_end_of_preamble(arg, &mut final_ctx);
                    }
                    let forced_jump_args: Vec<OpRef> = body_jump_args
                        .iter()
                        .map(|&arg| final_ctx.get_replacement_opref(arg))
                        .collect();
                    let current_vs = crate::optimizeopt::virtualstate::export_state(
                        &forced_jump_args,
                        &final_ctx,
                    );
                    let mut target_states: Vec<crate::optimizeopt::virtualstate::VirtualState> =
                        self.target_tokens
                            .iter()
                            .filter_map(|token| token.attrs().virtual_state.clone())
                            .collect();
                    let target_virtual_state = if let Some(idx) =
                        pick_virtual_state(&current_vs, &target_states, &mut final_ctx)
                    {
                        target_states.swap_remove(idx)
                    } else {
                        exported_vs.clone()
                    };
                    if let Ok(args) = target_virtual_state.make_inputargs(
                        &forced_jump_args,
                        &mut opt_p2,
                        &mut final_ctx,
                        true,
                    ) {
                        for arg_box in args {
                            if arg_box.is_none() {
                                continue;
                            }
                            let _ = opt_p2.force_box(&arg_box, &mut final_ctx);
                        }
                    }
                }
                let rebuilt = final_ctx.build_imported_short_preamble();
                let builder = final_ctx.imported_short_preamble_builder.clone();
                imported_short_aliases = final_ctx.used_imported_short_aliases();
                opt_p2.final_ctx = Some(final_ctx);
                (builder, rebuilt)
            } else {
                (
                    opt_p2.imported_short_preamble_builder.clone(),
                    opt_p2.imported_short_preamble.clone(),
                )
            };
        let mut initial_sp = rebuilt_imported_short_preamble.unwrap_or_else(|| {
            // RPython unroll.py:497-504 `for produced_op in
            // exported_state.short_boxes` parity: consume the eagerly-derived
            // ProducedShortOp list stored on ExportedState at export time.
            crate::optimizeopt::shortpreamble::build_short_preamble_from_produced_boxes(
                &exported_end_args,
                &exported_short_inputargs,
                &exported_short_boxes_produced,
            )
        });
        // Per-slot ORIGINAL box (what each renamed `short_inputargs[i]`
        // replaces), recovered from the produced `InputArg` short boxes via
        // `label_arg_idx` (shortpreamble.py:417 keys the info lookup by
        // `produced_op.short_op.res`, the original). Every label/virtual slot produces an InputArg
        // entry, except a duplicate slot (one box appears twice in
        // `label_args + virtuals`: the `potential_ops[box]` overwrite keys the
        // single entry at its LAST slot — shortpreamble.py:259, mirrored by
        // `live_slot` in add_short_input_arg — so the earlier dead slot stays
        // None) and a const-folded slot (dropped at export, never surviving
        // into Phase 2).
        let mut slot_to_original: Vec<Option<OpRef>> = vec![None; initial_sp.inputargs.len()];
        for (_, produced) in &exported_short_boxes_produced {
            if !matches!(
                produced.kind,
                crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
            ) {
                continue;
            }
            if let Some(slot) = produced.label_arg_idx
                && slot < slot_to_original.len()
            {
                slot_to_original[slot] = Some(produced.res.to_opref());
            }
        }
        // shortpreamble.py:416-425 parity: attach PtrInfo to each short
        // inputarg. RPython keys the info by the ORIGINAL res box
        // (`op = produced_op.short_op.res; info = exported_infos.get(op)`)
        // and forwards it onto the renamed `preamble_op`. The renamed short
        // inputarg carries no PtrInfo of its own, so the lookup MUST use the
        // original box, not the renamed one — otherwise a distinct renamed
        // identity yields None and the loop-carried info (e.g. KnownClass)
        // is dropped. A dead duplicate slot's box carries no info
        // (shortpreamble.py __init__), so its None is correct.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            let mut infos = Vec::with_capacity(initial_sp.inputargs.len());
            for (i, inputarg) in initial_sp.inputargs.iter().enumerate() {
                // Phase-2 info of the original box this short inputarg renames.
                let original = slot_to_original[i];
                let info = original
                    .and_then(|o| final_ctx.get_box_replacement_operand_opt(o))
                    .or_else(|| final_ctx.get_box_replacement_operand_opt(*inputarg))
                    .as_ref()
                    .and_then(|o| final_ctx.peek_ptr_info(o));
                infos.push(info);
            }
            initial_sp.inputarg_infos = infos;
        }
        let opt_unroll = OptUnroll::new();
        let (target_token, short_preamble_producer) = opt_unroll.finalize_short_preamble(
            self.target_tokens.len() as u64,
            exported_vs,
            initial_sp.clone(),
            imported_short_preamble_builder.as_ref(),
        );
        self.target_tokens.push(target_token);
        opt_p2.short_preamble_producer = short_preamble_producer;

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] finalize_short_preamble: target_tokens={}",
                self.target_tokens.len()
            );
        }

        // ── unroll.py: jump_to_existing_trace / close ladder ──
        // Try to match the body's JUMP virtual state to an existing target.
        // RPython: new_virtual_state = jump_to_existing_trace(end_jump, ...)
        //
        // RPython parity: never skip jump_to_existing_trace based on
        // guard count. RPython's unroll.py always attempts
        // jump_to_existing_trace regardless of body size.
        let p2_guard_count = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
        let skip_jump_to_existing = false;
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] post-finalize: entering jump_to_existing_trace section (p2_guards={}, skip={})",
                p2_guard_count, skip_jump_to_existing
            );
        }
        let mut body_ops = p2_ops;
        let mut redirected_tail_ops = Vec::new();
        let jump_was_redirected = {
            let body_jump_args: Vec<OpRef> = body_terminal_op
                .as_ref()
                .map(|jump| jump.args_slice().iter().map(|a| a.to_opref()).collect())
                .or_else(|| {
                    body_ops
                        .iter()
                        .rfind(|o| o.opcode == OpCode::Jump)
                        .map(|j| j.args_slice().iter().map(|a| a.to_opref()).collect())
                })
                .unwrap_or_default();
            let mut current_label_args = label_args.clone();
            // RPython Box parity: each used_box is a distinct Box even
            // when two virtuals share the same OpRef. Allocate fresh
            // OpRefs for duplicates so the LABEL carries independent slots.
            {
                let mut seen_used: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
                let mut next_fresh = current_label_args
                    .iter()
                    .copied()
                    .chain(initial_sp.used_boxes.iter().copied())
                    .map(|a| a.raw())
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1)
                    .max(body_num_inputs as u32 + 100);
                for ub in &initial_sp.used_boxes {
                    let ub = *ub;
                    if !seen_used.contains(&ub) {
                        seen_used.insert(ub);
                        current_label_args.push(ub);
                    } else {
                        // shortpreamble.py create_short_inputargs inputarg_from_tp parity:
                        // when the short preamble needs a fresh label slot,
                        // PyPy creates a new typed InputArg from the source
                        // box's `.type`. Carry that into duplicate used_box
                        // slots: typed parents produce a matching
                        // `InputArg{Int,Float,Ref}` alias.
                        //
                        // history.py:220 `box.type` is the authoritative
                        // source. Variant tag is the primary line-by-line
                        // equivalent (resoperation.py
                        // `AbstractValue.type`); legacy untyped paths fall
                        // through to Phase 2's `final_ctx.opref_type` and
                        // the recorder's `trace_inputargs`.
                        let tp = ub
                            .ty()
                            .or_else(|| {
                                opt_p2.final_ctx.as_ref().and_then(|ctx| ctx.opref_type(ub))
                            })
                            .or_else(|| {
                                let raw = ub.raw() as usize;
                                opt_p2.trace_inputargs.get(raw).and_then(|o| o.ty())
                            })
                            .unwrap_or_else(|| {
                                panic!(
                                    "duplicate short-preamble used_box {:?} has untyped variant; \
                                     RPython label/inputarg boxes carry Int/Ref/Float type \
                                     intrinsically (resoperation.py:1544 inputarg_from_tp)",
                                    ub
                                )
                            });
                        if matches!(tp, Type::Void) {
                            panic!(
                                "duplicate short-preamble used_box {:?} has void type; \
                                 RPython label/inputarg boxes must carry Int/Ref/Float \
                                 type (resoperation.py:1544 inputarg_from_tp)",
                                ub
                            );
                        }
                        let alias = OpRef::input_arg_typed(next_fresh, tp);
                        current_label_args.push(alias);
                        next_fresh += 1;
                    }
                }
            }
            let opt_unroll = OptUnroll::new();
            // Use Phase 2's final context for virtual state matching.
            let mut jump_ctx = opt_p2.final_ctx.take().unwrap_or_else(|| {
                // opencoder.py:259 inputarg_from_tp parity — Phase 2 inputargs
                // carry the same producer-side types as Phase 1's; fall back
                // to Ref when types are unavailable.
                let types: Vec<majit_ir::Type> = opt_p2
                    .trace_inputargs
                    .get(..body_num_inputs)
                    .map(|s| {
                        s.iter()
                            .map(|o| o.ty().unwrap_or(majit_ir::Type::Ref))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![majit_ir::Type::Ref; body_num_inputs]);
                crate::optimizeopt::OptContext::with_inputarg_types(32, &types)
            });

            // unroll.py: jump_to_existing_trace(force_boxes=False)
            // RPython: except InvalidLoop → jump_to_preamble immediately,
            // NO retry. The big comment at unroll.py:305-316 explains why
            // continuing after partial inlining is unsafe.
            // unroll.py:153/166 passes `state.runtime_boxes` (the preamble's
            // recorded JUMP args, exported at unroll.py:105) to
            // jump_to_existing_trace, NOT the peeled body's own jump arglist.
            // `runtime_boxes[i]` is read un-forwarded via `getint`/`getref_base`
            // (the box's own observed value), while `boxes[i]` (= forwarded body
            // jump args) drives the virtual-state match (virtualstate.py generate_guards
            // reads them as parallel-but-distinct lists).
            //
            // `exported_runtime_boxes` is the recorded JUMP arglist threaded
            // through ExportedState.runtime_boxes (set in the Phase-1 export
            // above, carried across a retrace import). Those Phase-1 OpRefs
            // resolve in Phase-2's jump_ctx via cross-phase find_producer_op
            // (phase1_emit_ops / input_ops) and carry the boxes' own observed
            // runtime values. Fall back to the body jump args only when the
            // channel is empty (a trace with no recorded JUMP), where the
            // generate_guards length assert would otherwise fail.
            let runtime_boxes = if exported_runtime_boxes.len() == body_jump_args.len() {
                exported_runtime_boxes.clone()
            } else {
                body_jump_args.clone()
            };
            let mut invalid_loop = false;
            let mut jumped = if skip_jump_to_existing {
                false
            } else {
                let did_jump = opt_unroll
                    .jump_to_existing_trace(
                        &body_jump_args,
                        Some(&current_label_args),
                        &mut self.target_tokens,
                        &mut opt_p2,
                        &mut jump_ctx,
                        false,
                        &runtime_boxes,
                    )
                    .is_none();
                if let Some(reason) = jump_ctx.take_invalid_loop() {
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] InvalidLoop during force_boxes=false: {}",
                            reason.0
                        );
                    }
                    // unroll.py:154-158: except InvalidLoop →
                    // jump_to_preamble, skip retry
                    invalid_loop = true;
                    false
                } else {
                    did_jump
                }
            };
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] jump_to_existing_trace(force_boxes=false) result: jumped={}, invalid_loop={}",
                    jumped, invalid_loop
                );
            }

            // unroll.py:154-158: on InvalidLoop, skip retry entirely
            if !jumped && !skip_jump_to_existing && !invalid_loop {
                // unroll.py:161-168: force_boxes=True, except InvalidLoop: pass
                let did_jump = opt_unroll
                    .jump_to_existing_trace(
                        &body_jump_args,
                        Some(&current_label_args),
                        &mut self.target_tokens,
                        &mut opt_p2,
                        &mut jump_ctx,
                        true,
                        &runtime_boxes,
                    )
                    .is_none();
                jumped = if let Some(reason) = jump_ctx.take_invalid_loop() {
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] InvalidLoop during force_boxes=true retry: {}",
                            reason.0
                        );
                    }
                    false // unroll.py:167-168: except InvalidLoop: pass
                } else {
                    did_jump
                };
            }
            if jumped && redirected_tail_ops.is_empty() {
                // Only take jump_ctx ops if we don't already have
                // a self-loop Jump from the retrace path.
                redirected_tail_ops = jump_ctx.take_new_operations();
                // A close onto the body token this compilation just minted is
                // always legal. A close onto any other token is legal exactly
                // when that token belongs to the artifact this compilation will
                // be installed under — compile.py:197's own test — because the
                // JUMP is emitted as an absolute branch to the target's
                // `ll_loop_code()` and only a same-artifact target shares the
                // buffer's lifetime and invalidation flag.
                //
                // `_jump_to_existing_trace` walks `jitcelltoken.target_tokens`
                // (unroll.py:171), so upstream's candidates are all owned by one
                // JitCellToken by construction. pyre seeds them from a green-key
                // side table that outlives a recompile, so the ownership has to
                // be re-checked here.
                let current_body_descr =
                    self.target_tokens.last().map(|t| t.as_jump_target_descr());
                let current_body_descr_idx = current_body_descr.as_ref().map(|d| d.index());
                let redirected_jump_descr = redirected_tail_ops
                    .iter()
                    .rfind(|o| o.opcode == OpCode::Jump)
                    .and_then(|o| o.getdescr());
                let redirected_jump_descr_idx = redirected_jump_descr.as_ref().map(|d| d.index());
                let redirected_jump_owner_num = redirected_jump_descr
                    .as_ref()
                    .and_then(|d| d.as_loop_target_descr())
                    .and_then(|d| d.original_jitcell_token_number());
                let closes_onto_own_body = redirected_jump_descr_idx == current_body_descr_idx;
                let closes_within_attached_artifact =
                    match (self.attach_jitcell_token_number, redirected_jump_owner_num) {
                        (Some(attach_num), Some(target_num)) => attach_num == target_num,
                        // No artifact to attach to yet: compile_loop mints its token
                        // after optimizing, so every seeded candidate still names the
                        // *previous* compilation. compile.py:288-289 keeps that case
                        // from arising at all by resetting the list.
                        _ => false,
                    };
                if !closes_onto_own_body && !closes_within_attached_artifact {
                    // Fall back to jump_to_preamble, matching the
                    // unreachable-target path in unroll.py:228.
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] jump_to_existing_trace: target {:?} (owner {:?}) is neither this body {:?} nor inside the attached artifact {:?}, falling back to preamble",
                            redirected_jump_descr_idx,
                            redirected_jump_owner_num,
                            current_body_descr_idx,
                            self.attach_jitcell_token_number,
                        );
                    }
                    redirected_tail_ops.clear();
                    jumped = false;
                }
            }
            opt_p2.final_ctx = Some(jump_ctx);
            jumped
        };

        let sp = self
            .target_tokens
            .last()
            .and_then(|target| target.attrs().short_preamble.clone())
            .unwrap_or(initial_sp);
        if !sp.is_empty() {
            self.short_preamble = Some(sp.clone());
        }
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] assembly_contract: label_args={:?} used_boxes={:?} jump_args={:?}",
                label_args, sp.used_boxes, sp.jump_args
            );
        }

        if !jump_was_redirected {
            // unroll.py: jump_to_preamble — body JUMP → preamble Label
            //
            // force_box_for_end_of_preamble (unroll.py:126-127) has already run
            // over the body JUMP's args above, so each arg is a forced box rather
            // than a virtual. That is a per-ARG property and says nothing about the
            // arg LIST matching the preamble Label's — on a retrace it does not, and
            // the arity check below is what catches it.
            let preamble_target = self
                .target_tokens
                .first()
                .expect("preamble target token must exist before jump_to_preamble")
                .clone();
            let preamble_arity = exported_renamed_inputargs.len();
            let body_jump_arity = body_terminal_op
                .as_ref()
                .map(|j| j.num_args())
                .or_else(|| {
                    body_ops
                        .iter()
                        .rev()
                        .find(|op| op.opcode == OpCode::Jump)
                        .map(|op| op.num_args())
                })
                .unwrap_or(0);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] jump_to_preamble: body_jump_args={} preamble_arity={} start_label_args={:?}",
                    body_jump_arity, preamble_arity, exported_renamed_inputargs,
                );
            }
            // Upstream has no arity check on this path at all, and the check
            // below is pyre's own. `unroll.py jump_to_preamble` only
            // asserts `target_tokens[0].virtual_state is None` and retargets the
            // JUMP with `copy_and_change`, keeping its args; `compile.py:334`'s
            // `assert jump_op.numargs() == loop_info.label_op.numargs()` is
            // guarded one line above by `if jump_op.getdescr() is
            // loop_info.label_op.getdescr()`, which is exactly what
            // `jump_to_preamble` has just stopped being true, and
            // `compile_retrace` checks nothing. Upstream needs no check because
            // its `target_tokens[0]` is the preamble start label carrying
            // `start_state.renamed_inputargs` — the same loop-carried positions
            // the body JUMP carries — so the retarget is sound by construction.
            //
            // A pyre RETRACE has no start label of its own (see
            // `emit_start_label`), so `target_tokens[0]` is the ORIGINAL loop's
            // start label: the portal's entry contract, one boxed Ref per frame
            // local. The optimized loop-carried set is wider and mixed, so the
            // retarget can land N values on an M-slot label and the target then
            // reads a raw integer as a Ref — measured as EXC_BAD_ACCESS on the
            // loop counter inside GuardClass.
            //
            // Give up, which is `jump_to_preamble`'s own escape (it lets
            // `send_extra_operation` raise `InvalidLoop`) and lands on
            // `compile_retrace`'s cancel. Scoped to the retrace: a
            // `compile_loop` unroll keeps its start label in the SAME artifact,
            // so its preamble target is local and this mismatch cannot arise.
            if !self.emit_start_label && body_jump_arity != preamble_arity {
                crate::mc_diag_bump(57);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] jump_to_preamble giveup: body JUMP args {body_jump_arity} \
                         != preamble LABEL args {preamble_arity}"
                    );
                }
                return Err(crate::optimize::InvalidLoop(
                    "jump_to_preamble: body JUMP arity != preamble LABEL arity",
                ));
            }
            if let Some(mut end_jump) = body_terminal_op {
                end_jump.setdescr(preamble_target.as_jump_target_descr());
                // Retrace mismatches already gave up above. A compile_loop
                // close onto the same-artifact start LABEL may still be one
                // box short of `exported_renamed_inputargs` (an entry-only
                // invariant). Pad/truncate here so consider_jump /
                // the assembler remap never sees JUMP 33 vs LABEL 34.
                // A missing *middle* slot shifts later JUMP boxes onto the
                // wrong LABEL kinds: an Int then lands in a Ref register and
                // GuardClass loads `[loop_counter + vtable_offset]`.
                // `unroll.py jump_to_preamble` keeps the arglist; give up
                // rather than compile that remap.
                if jump_and_preamble_slot_kinds_disagree(&end_jump, &exported_renamed_inputargs) {
                    crate::mc_diag_bump(57);
                    return Err(crate::optimize::InvalidLoop(
                        "jump_to_preamble: JUMP/LABEL slot kinds disagree",
                    ));
                }
                apply_preamble_shape_to_jump(&mut end_jump, &exported_renamed_inputargs);
                if let Some(mut final_ctx) = opt_p2.final_ctx.take() {
                    // unroll.py parity: jump_to_preamble retargets
                    // the live end_jump and routes it through
                    // send_extra_operation, preserving any force_box /
                    // partial-inline operations already appended to
                    // _newoperations.
                    //
                    // unroll.py:242 lets send_extra_operation raise
                    // InvalidLoop; this function returns Result, so `?`
                    // carries it out (final_ctx is abandoned with the
                    // discarded trace).
                    opt_p2.send_extra_operation(&OpRc::new(end_jump.clone()), &mut final_ctx)?;
                    let redirected_tail_ops: Vec<majit_ir::OpRc> = final_ctx.take_new_operations();
                    opt_p2.final_ctx = Some(final_ctx);
                    body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
                } else {
                    body_ops = replace_terminal_jump(&body_ops, end_jump);
                }
            } else {
                if let Some(end_jump) = body_ops.iter().rfind(|op| op.opcode == OpCode::Jump) {
                    if jump_and_preamble_slot_kinds_disagree(end_jump, &exported_renamed_inputargs)
                    {
                        crate::mc_diag_bump(57);
                        return Err(crate::optimize::InvalidLoop(
                            "jump_to_preamble: JUMP/LABEL slot kinds disagree",
                        ));
                    }
                }
                body_ops = Self::jump_to_preamble(
                    &body_ops,
                    &preamble_target,
                    &exported_renamed_inputargs,
                );
            }
            crate::debug::log_one(
                "jit-tracing",
                "jump_to_preamble: body JUMP retargeted to start descr",
            );
        } else if !redirected_tail_ops.is_empty() {
            body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
        } else {
            crate::debug::log_one(
                "jit-tracing",
                "jump_to_existing_trace: body JUMP → self-loop",
            );
        }

        // unroll.py: disable_retracing_if_max_retrace_guards
        if jump_was_redirected
            && Self::disable_retracing_if_max_retrace_guards(&body_ops, self.max_retrace_guards)
        {
            self.retraced_count = u32::MAX;
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] too many guards (>{}), disabling retracing",
                    self.max_retrace_guards
                );
            }
        }

        // ── Assembly (compile.py:310-338) ──
        let mut appended_label_args: Vec<OpRef> = Vec::new();
        let sp_used_boxes: Vec<OpRef> = sp.used_boxes.clone();
        let sp_jump_args: Vec<OpRef> = sp.jump_args.clone();
        let mut combined = assemble_peeled_trace_with_jump_args(
            &p1_ops,
            &body_ops,
            &label_args,
            &exported_renamed_inputargs,
            &sp_used_boxes,
            &sp_jump_args,
            p2_ni,
            phase2_inputarg_base,
            jump_was_redirected,
            &imported_short_aliases,
            &consts_p2,
            self.target_tokens
                .first()
                .filter(|_| self.emit_start_label)
                .map(|target| target.as_jump_target_descr()),
            self.target_tokens
                .last()
                .map(|target| target.as_jump_target_descr()),
            &exported_end_args,
            &mut appended_label_args,
            opt_p2
                .final_ctx
                .as_mut()
                .expect("assemble peeled trace requires Phase 2 OptContext for Box.type parity"),
        );
        // RPython Box parity: drop duplicate-position ops. In RPython
        // each Box is unique so collisions can't happen. Keep first.
        {
            let mut seen: crate::FxIndexSet<u32> = crate::FxIndexSet::default();
            combined.retain(|op| {
                if op.pos().get().is_none() || op.result_type() == Type::Void {
                    return true;
                }
                seen.insert(op.pos().get().raw())
            });
        }
        {
            let _s = crate::debug::scope("jit-log-opt-loop");
            if crate::debug::have_debug_prints() {
                crate::debug::debug_print("--- peeled trace (assembled) ---");
                for line in majit_ir::format_trace(&combined, &consts_p2).lines() {
                    crate::debug::debug_print(line);
                }
                let mut sorted_consts: Vec<_> =
                    consts_p2.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
                sorted_consts.sort_by_key(|(k, _)| *k);
                crate::debug::debug_print(&format!("consts_p2: {sorted_consts:?}"));
            }
        }
        // A LABEL arg the fallthrough scan appended is carried by the loop but
        // not by the short preamble, so `inline_short_preamble` cannot rebuild
        // it and every bridge closing onto this LABEL arrives one arg short.
        // Record the reconstruction recipe for the appends that have one: the
        // vable input layout is `[frame, static scalars.., array items..]`, so
        // an appended static-scalar slot is exactly `GetfieldGc*(frame, descr)`
        // — the load the preamble itself performed before the optimizer folded
        // it onto the seeded slot. Appends outside that window record nothing;
        // the LABEL/JUMP contract is `vable_label_arg_recipes`, and a token
        // with an append left uncovered is marked `label_tail_unrebuildable`.
        //
        // The list is all-or-nothing: the recipes rebuild a contiguous LABEL
        // tail, so one append with no recipe leaves the close short anyway and
        // a partial list would make it overshoot. It is also written on every
        // assembly, empty included, because this token can be recompiled with
        // a different body and a stale list would append args the LABEL no
        // longer has.
        if let Some(target) = self.target_tokens.last() {
            let recipes = vable_config
                .as_ref()
                .filter(|_| !appended_label_args.is_empty())
                .map(|config| {
                    let first = 1 + config.vable_input_offset;
                    appended_label_args
                        .iter()
                        .map(|&arg| {
                            let slot = arg
                                .is_input_arg()
                                .then(|| arg.raw() as usize)
                                .filter(|&raw| raw >= first)
                                .map(|raw| raw - first)
                                .filter(|&slot| slot < config.static_field_offsets.len())?;
                            let descr = config.static_field_descrs.get(slot).cloned()?;
                            Some((majit_ir::OpCode::getfield_for_type(arg.ty()?), descr))
                        })
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            let mut attrs = target.attrs();
            attrs.label_tail_unrebuildable = recipes.len() != appended_label_args.len();
            let replaced = !attrs.vable_label_arg_recipes.is_empty() || !recipes.is_empty();
            if replaced {
                attrs.vable_label_arg_recipes = recipes;
            }
            drop(attrs);
            if replaced {
                target.mark_minor_scan_pending();
            }
        }

        *constants = consts_p2;
        Ok((combined, p2_ni))
    }

    /// RPython compile.py uses the optimized loop state's inputargs contract,
    /// not a stale input counter. When majit falls back to the phase-1 trace,
    /// derive the live loop arity from the actual closing Label/Jump.
    pub fn closing_loop_contract_arity<T: AsRef<Op>>(ops: &[T], fallback: usize) -> usize {
        closing_loop_contract_arity(ops, fallback)
    }

    /// Count the guards in an optimized trace (for retrace_limit checks).
    pub fn count_guards<T: AsRef<Op>>(ops: &[T]) -> u32 {
        ops.iter()
            .filter(|op| op.as_ref().opcode.is_guard())
            .count() as u32
    }

    /// unroll.py: _map_args(mapping, arglist)
    /// Remap a list of OpRefs through a forwarding mapping.
    /// Constant OpRefs are left unchanged because they are not remapped.
    pub fn map_args(mapping: &crate::FxIndexMap<OpRef, OpRef>, args: &[OpRef]) -> Vec<OpRef> {
        args.iter()
            .map(|&arg| mapping.get(&arg).copied().unwrap_or(arg))
            .collect()
    }

    /// unroll.py: _check_no_forwarding(lsts)
    /// Debug assertion: verify no OpRef in the lists has been forwarded.
    pub fn check_no_forwarding(ctx: &crate::optimizeopt::OptContext, oprefs: &[OpRef]) -> bool {
        oprefs.iter().all(|&r| ctx.get_replacement_opref(r) == r)
    }

    /// unroll.py: disable_retracing_if_max_retrace_guards(ops, target_token)
    /// If the trace has too many guards, disable retracing for this location.
    /// Returns true if retracing was disabled.
    pub fn disable_retracing_if_max_retrace_guards<T: AsRef<Op>>(
        ops: &[T],
        max_retrace_guards: u32,
    ) -> bool {
        let guard_count = Self::count_guards(ops);
        guard_count > max_retrace_guards
    }

    /// unroll.py: get_virtual_state(args)
    /// Build a VirtualState from the optimizer's current knowledge about args.
    pub fn get_virtual_state(
        args: &[OpRef],
        ctx: &crate::optimizeopt::OptContext,
    ) -> crate::optimizeopt::virtualstate::VirtualState {
        crate::optimizeopt::virtualstate::export_state(args, ctx)
    }
}

fn closing_loop_contract_arity<T: AsRef<Op>>(ops: &[T], fallback: usize) -> usize {
    ops.iter()
        .map(|op| op.as_ref())
        .rev()
        .find_map(|op| match op.opcode {
            OpCode::Label | OpCode::Jump => Some(op.num_args()),
            _ => None,
        })
        .unwrap_or(fallback)
}

impl Default for UnrollOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod quasi_immutable_dep_tests {
    #[derive(Debug)]
    struct FakeHandle(std::sync::Arc<()>);

    impl majit_ir::QuasiImmutHandle for FakeHandle {
        fn is_current(&self) -> bool {
            true
        }

        fn register_loop_token(&self, _token: &std::sync::Arc<dyn majit_ir::QuasiImmutLoopToken>) {}

        fn instance_identity(&self) -> usize {
            std::sync::Arc::as_ptr(&self.0) as usize
        }
    }

    fn handle(instance: &std::sync::Arc<()>) -> std::sync::Arc<dyn majit_ir::QuasiImmutHandle> {
        std::sync::Arc::new(FakeHandle(instance.clone()))
    }

    /// `heap.py:821-823` keys `quasi_immutable_deps` by the `QuasiImmut`, so an
    /// instance is one entry however many reads found it. A handle is minted per
    /// read, so deduping on the handle instead lets two markers for one instance
    /// each register the loop again.
    #[test]
    fn two_handles_for_one_instance_merge_to_one_dep() {
        let instance = std::sync::Arc::new(());
        let first = handle(&instance);
        let second = handle(&instance);
        assert!(
            !std::sync::Arc::ptr_eq(&first, &second),
            "the two reads must mint distinct handles or this proves nothing",
        );

        let mut deps = vec![first];
        super::merge_quasi_immutable_deps(&mut deps, &[second]);
        assert_eq!(deps.len(), 1);

        super::merge_quasi_immutable_deps(&mut deps, &[handle(&std::sync::Arc::new(()))]);
        assert_eq!(deps.len(), 2, "a different instance is a different dep");
    }
}

pub(crate) fn merge_quasi_immutable_deps(
    dst: &mut Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    src: &[std::sync::Arc<dyn majit_ir::QuasiImmutHandle>],
) {
    for dep in src {
        if !dst
            .iter()
            .any(|seen| seen.instance_identity() == dep.instance_identity())
        {
            dst.push(dep.clone());
        }
    }
}

/// unroll.py: ExportedState — snapshot of optimizer state at the end of
/// the preamble, used to initialize the peeled loop body.
///
/// Contains the virtual state, short preamble boxes, arg mappings,
/// and exported infos needed to resume optimization after peeling.
///
/// gcreftracer.py parity: GcRef values in exported_infos are rooted on
/// the shadow stack. ExportedState persists between Phase 1 (preamble)
/// and Phase 2 (body), during which GC can run and move objects.
#[derive(Debug)]
pub struct ExportedState {
    /// Label args at the end of the preamble (after forcing).
    pub end_args: Vec<OpRef>,
    /// Positions in `end_args` that feed the compact non-virtual LABEL args.
    pub label_source_positions: Vec<usize>,
    /// Args for the next iteration (before forcing). unroll.py:467
    /// `next_iteration_args = end_args` — the SAME canonical box objects
    /// (producer-bound / const [`Operand`]s) used as `exported_infos` keys,
    /// so the import-state lookup is an identity hit.
    pub next_iteration_args: Vec<Operand>,
    /// Types of end_args as determined by Phase 1 optimization.
    /// Used by Phase 2 import_state to propagate unboxed types.
    pub end_arg_types: Vec<Type>,
    /// Virtual state at the loop boundary.
    pub virtual_state: crate::optimizeopt::virtualstate::VirtualState,
    /// unroll.py ExportedState.exported_infos — optimizer knowledge from preamble.
    /// Maps OpRef → info for all args including virtual field contents.
    ///
    /// RPython stores one of `PtrInfo` / `IntBound` / `FloatConstInfo` per box,
    /// dispatched via `isinstance` in `setinfo_from_preamble` (unroll.py).
    /// Majit uses the existing `OpInfo` enum (`info.rs`) as the discriminated
    /// union of these three cases. Keyed by box object identity ([`Operand`]
    /// `Eq` = producer / const-cell `Rc::ptr_eq`), matching RPython's plain
    /// box-keyed dict.
    pub exported_infos: indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
    /// RPython shortpreamble.py: produced short boxes in preamble order.
    /// This preserves the original preamble ops so the active path can build
    /// short preambles without re-extracting them from the peeled trace.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// Const-result heap short boxes. Kept separate from `short_boxes` so
    /// they are replayed on import but never feed carried short-preamble
    /// boxes, label args, or `used_boxes`.
    pub const_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// RPython unroll.py:478-479 `short_boxes = sb.create_short_boxes(...)`
    /// stored directly on ExportedState. Consumer sites
    /// (`build_imported_short_preamble`, `import_short_preamble_state`)
    /// iterate this list verbatim. Pyre derives this eagerly from
    /// `exported_short_boxes + label_args + short_inputargs` at export time
    /// (`shortpreamble.py ShortBoxes.create_short_boxes` parity).
    pub short_boxes: Vec<(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)>,
    /// Short preamble builder for bridge entry.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// Renamed inputargs from the preamble. Each OpRef is a typed
    /// `InputArg{Int,Float,Ref}` variant carrying its `.type` intrinsically
    /// (history.py:220), so consumers read the type via `OpRef::ty()` —
    /// RPython `info.renamed_inputargs` Box parity, no parallel type array.
    pub renamed_inputargs: Vec<OpRef>,
    /// Short inputargs for the short preamble — the renamed inputarg
    /// positions (shortpreamble.py:430 / unroll.py:480), shared with the
    /// renamed operands inside `short_boxes`.
    pub short_inputargs: Vec<OpRef>,
    /// Rooted `InputArgRc` carriers for `short_inputargs`, index-aligned.
    /// Keeping the strong Rc alive across the cross-peel export boundary
    /// preserves the renamed inputarg producers for consumers that need a
    /// bound operand view.
    pub short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// unroll.py: runtime_boxes — live values at the original jump point.
    /// Threaded into Phase 2 import as `runtime_boxes` for guard generation.
    /// Default `Vec::new()` until the export site populates it; callers
    /// that need it write the field directly after `ExportedState::new`.
    pub runtime_boxes: Vec<OpRef>,
    /// RPython parity: patchguardop from Phase 1's GuardFutureCondition.
    /// Phase 2's extra_guards (from virtualstate) need rd_resume_position
    /// from this patchguardop (unroll.py:333-336).
    pub patchguardop: Option<majit_ir::Op>,
    /// unroll.py:234 `exported_state.quasi_immutable_deps` — deps collected
    /// while optimizing the preamble, merged into retrace deps at
    /// compile.py:384-390.
    pub quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    /// `OptContext::next_pos` at end of Phase 1 — strict upper bound of
    /// every OpRef Phase 1 allocated, including intermediates folded /
    /// forwarded away. `reserve_pos_typed` skips `materialize_operand_at` on the
    /// zero-inputarg / retrace baselines (`optimizeopt/mod.rs`),
    /// so capturing `ctx.next_pos` at export is the only reliable
    /// floor. Phase 2 / retrace seed their TraceIterator namespace
    /// strictly above this watermark via `opref_high_water()` to keep
    /// the OpRef set disjoint — RPython's object-identity Boxes get
    /// disjointness from Python identity for free; pyre's numeric
    /// OpRefs need an explicit position floor.
    pub phase1_emit_high_water: u32,
    /// `partial_trace.inputargs` (compile.py:362, history.py:509).
    /// Keeps the preamble's `AbstractInputArg` instances alive so that
    /// `compile_retrace`'s `optimize_peeled_loop` reads `_forwarded` off
    /// the same objects (resoperation.py
    /// `AbstractInputArg._forwarded`). Each entry is `Rc::clone` cheap;
    /// the vec is bounded by `ExportedState` lifetime. Populated only by
    /// `export_state_with_bounds`.
    pub(crate) partial_trace_inputargs: Vec<majit_ir::InputArgRc>,
    /// `partial_trace.operations` (compile.py:362, history.py:528).
    /// Keeps the preamble's emit `AbstractResOp` instances alive — a
    /// follow-up `compile_retrace` spawns a fresh optimizer that only
    /// sees the imported `ExportedState`, so the OpRc identities the
    /// preamble mutated `_forwarded` on must travel through here
    /// (resoperation.py `_forwarded` host).
    pub(crate) partial_trace_operations: Vec<majit_ir::OpRc>,
    /// #173 producer-rooting: keeps the Phase-1 producer `Op` that each
    /// exported short-box `res` is bound to (`res.bound_op()`) alive into
    /// Phase 2. A short-box `res` carries only a `Weak<Op>`; the Phase-1
    /// OptContext that owns the strong `OpRc` drops at the peel boundary, so
    /// without this carry `res` (a `Weak<Op>`) upgrades to a dead producer at
    /// Phase 2 and its accessors panic. Populated at export from
    /// `res.bound_op()` while still alive; InputArg-kind res (`bound_inputarg`,
    /// not `bound_op`) contributes nothing here and is rooted via
    /// `short_inputarg_refs` instead.
    pub(crate) short_box_producer_roots: Vec<majit_ir::OpRc>,
    /// Shadow stack rooting for GcRef values in exported_infos.
    /// (operand key, field kind, shadow stack index). The key is the
    /// `exported_infos` operand key for `InfoPtrInfoConstant` entries; other
    /// field kinds carry the none/empty operand sentinel since they never key
    /// back into `exported_infos`.
    rooted_refs: Vec<(Operand, ExportedGcRefField, usize)>,
    /// Shadow stack slots for every inline `ConstPtr.value` reachable from
    /// this ExportedState's Rust object graph. RPython traces these fields as
    /// normal Const object attributes; pyre records the walk order and copies
    /// the forwarded values back in `refresh_from_gc`.
    rooted_const_ptr_slots: Vec<usize>,
    /// Shadow stack depth at creation. release_roots pops to here.
    shadow_stack_base: usize,
}

// unroll.py `exported_infos - a mapping from ops to infos, including inputargs`
// The per-entry info type is now `OpInfo` (`info.rs`), the discriminated
// union matching RPython's `PtrInfo | IntBound | FloatConstInfo` dispatched
// via `isinstance` in `setinfo_from_preamble` (unroll.py). The earlier
// majit-only bundle `ExportedValueInfo { constant, ptr_info, int_bound }`
// is removed as part of the Box identity plan Phase D.

/// Identifies which GcRef-bearing field inside an ExportedState
/// is rooted at a particular shadow stack slot.
#[derive(Clone, Copy, Debug)]
enum ExportedGcRefField {
    /// exported_infos[OpRef].ptr_info = PtrInfo::Constant(GcRef)
    InfoPtrInfoConstant,
    /// virtual_state.state[index] = Constant(Value::Ref)
    VirtualStateConstantRef(usize),
    /// `partial_trace_inputargs[index].forwarded = Info(OpInfo::Ptr(PtrInfo::Constant(_)))`.
    /// PyPy `InputArg._forwarded` host (resoperation.py:700).
    PartialTraceInputArgInfoPtrInfoConstant(usize),
    /// `partial_trace_inputargs[index].forwarded = Const(Const::Ref(_), _)`.
    PartialTraceInputArgConstRef(usize),
    /// `partial_trace_operations[index].forwarded = Info(OpInfo::Ptr(PtrInfo::Constant(_)))`.
    /// PyPy `AbstractResOp._forwarded` host (resoperation.py).
    PartialTraceOpInfoPtrInfoConstant(usize),
    /// `partial_trace_operations[index].forwarded = Const(Const::Ref(_), _)`.
    PartialTraceOpConstRef(usize),
}

impl ExportedState {
    /// unroll.py: ExportedState.__init__
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn new(
        end_args: Vec<OpRef>,
        label_source_positions: Vec<usize>,
        next_iteration_args: Vec<Operand>,
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        exported_infos: indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
        exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
        const_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
        renamed_inputargs: Vec<OpRef>,
        short_inputargs: Vec<OpRef>,
        short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    ) -> Self {
        // unroll.py:478-479 `sb.create_short_boxes(...)` parity: pyre
        // pre-derives the per-OpRef ProducedShortOp view and stores it
        // directly, matching RPython's `ExportedState.short_boxes`. The
        // label-arg → short-inputarg rename already happened at export time
        // inside `produce_arg` (shortpreamble.py/294), so this is a
        // plain GuardOverflow filter + transform of `exported_short_boxes`.
        let short_boxes =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &exported_short_boxes,
            );
        ExportedState {
            end_args,
            label_source_positions,
            // unroll.py:467 `next_iteration_args = end_args` — carry the literal
            // Phase-1 boxes (the same Rcs used as `exported_infos` keys) so the
            // import-state lookup is a ptr_eq hit. NOT `from_opref` (which would
            // mint fresh producer-less Rcs and sever the carry identity).
            next_iteration_args,
            end_arg_types: Vec::new(),
            virtual_state,
            exported_infos,
            exported_short_boxes,
            const_short_boxes,
            short_boxes,
            short_preamble: None,
            renamed_inputargs,
            short_inputargs,
            short_inputarg_refs,
            runtime_boxes: Vec::new(),
            patchguardop: None,
            quasi_immutable_deps: Vec::new(),
            phase1_emit_high_water: 0,
            partial_trace_inputargs: Vec::new(),
            partial_trace_operations: Vec::new(),
            short_box_producer_roots: Vec::new(),
            rooted_refs: Vec::new(),
            rooted_const_ptr_slots: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
        // gcreftracer.py parity: RPython ExportedState is a Python object
        // whose GcRef fields are automatically traced by the GC. In Rust,
        // root_all_gcrefs() must be called at each storage site in LIFO
        // order (longer-lived copy rooted first → lower shadow stack depth).
        // new() does NOT auto-root because the LIFO ordering depends on
        // the caller's storage pattern.
    }

    /// Visit every GC reference slot reachable from this exported optimizer
    /// state. This is the Rust equivalent of PyPy keeping `ExportedState`
    /// inside the GC-traced Python object graph: ConstPtr boxes, op args,
    /// short preamble state, virtual state, and `_forwarded` payloads all
    /// expose their actual mutable storage.
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_opref(opref: &mut OpRef, visitor: &mut dyn FnMut(&mut GcRef)) {
            if let OpRef::ConstPtr(gcref) = opref {
                visitor(gcref);
            }
        }

        fn visit_oprefs(refs: &mut [OpRef], visitor: &mut dyn FnMut(&mut GcRef)) {
            for r in refs {
                visit_opref(r, visitor);
            }
        }

        fn visit_operands(operands: &[Operand], visitor: &mut dyn FnMut(&mut GcRef)) {
            for o in operands {
                o.walk_const_ptr_refs(visitor);
            }
        }

        fn visit_op(op: &Op, visitor: &mut dyn FnMut(&mut GcRef)) {
            let mut pos = op.pos().get();
            visit_opref(&mut pos, visitor);
            op.pos().set(pos);
            op.walk_const_ptr_refs_mut(visitor);
        }

        fn visit_op_info(
            info: &mut crate::optimizeopt::info::OpInfo,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            if let crate::optimizeopt::info::OpInfo::Ptr(rc) = info {
                rc.borrow_mut().walk_const_ptr_refs_mut(visitor);
            }
        }

        fn visit_forwarded(
            forwarded: &mut majit_ir::forwarding::Forwarded,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            match forwarded {
                majit_ir::forwarding::Forwarded::Info(info) => visit_op_info(info, visitor),
                other => other.walk_const_ptr_refs(visitor),
            }
        }

        fn visit_preamble_op(
            entry: &mut crate::optimizeopt::shortpreamble::PreambleOp,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            visit_op(&entry.op, visitor);
            if let Some(source_op) = entry.source_op.as_ref() {
                visit_op(source_op, visitor);
            }
            entry.res.walk_const_ptr_refs(visitor);
            if let Some(source) = entry.same_as_source.as_ref() {
                source.walk_const_ptr_refs(visitor);
            }
        }

        fn visit_produced_short_op(
            produced: &mut crate::optimizeopt::shortpreamble::ProducedShortOp,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            visit_op(&produced.preamble_op, visitor);
            visit_op(&produced.source_op, visitor);
            produced.res.walk_const_ptr_refs(visitor);
            if let Some(source) = produced.same_as_source.as_ref() {
                source.walk_const_ptr_refs(visitor);
            }
        }

        visit_oprefs(&mut self.end_args, visitor);
        visit_operands(&self.next_iteration_args, visitor);
        self.virtual_state.walk_const_ptr_refs_mut(visitor);
        // Operand hashes by Rc pointer identity, so walking interior GcRefs
        // does not alter the key's hash — safe to borrow key immutably while
        // mutating values. Key GcRefs are walked through interior mutability
        // (Operand wraps Rc<RefCell<..>>).
        for (key, info) in self.exported_infos.iter_mut() {
            key.walk_const_ptr_refs(visitor);
            visit_op_info(info, visitor);
        }
        for entry in self
            .exported_short_boxes
            .iter_mut()
            .chain(self.const_short_boxes.iter_mut())
        {
            visit_preamble_op(entry, visitor);
        }
        for (key, produced) in &mut self.short_boxes {
            visit_opref(key, visitor);
            visit_produced_short_op(produced, visitor);
        }
        if let Some(short_preamble) = self.short_preamble.as_mut() {
            short_preamble.walk_const_ptr_refs_mut(visitor);
        }
        visit_oprefs(&mut self.renamed_inputargs, visitor);
        visit_oprefs(&mut self.short_inputargs, visitor);
        visit_oprefs(&mut self.runtime_boxes, visitor);
        if let Some(patchguardop) = self.patchguardop.as_ref() {
            visit_op(patchguardop, visitor);
        }
        for ia in &self.partial_trace_inputargs {
            visit_forwarded(&mut ia.forwarded.borrow_mut(), visitor);
        }
        for op in &self.partial_trace_operations {
            visit_op(op, visitor);
            visit_forwarded(&mut op.forwarded().borrow_mut(), visitor);
        }
    }

    pub fn has_shadow_roots(&self) -> bool {
        !self.rooted_refs.is_empty() || !self.rooted_const_ptr_slots.is_empty()
    }

    /// Smallest fresh OpRef that is guaranteed not to collide with any Box
    /// identity stored in this exported preamble state.
    ///
    /// RPython keeps these as object identities in `partial_trace` /
    /// `ExportedState`. Majit uses integer OpRefs, so compile_retrace must
    /// reconstruct this high-water mark before creating the fresh Phase 2
    /// TraceIterator namespace.
    fn opref_high_water(&self) -> u32 {
        let mut high = 0_u32;
        let mut visit = |opref: OpRef| {
            if !opref.is_none() && !opref.is_constant() {
                high = high.max(opref.raw().saturating_add(1));
            }
        };
        let visit_op = |op: &Op, visit: &mut dyn FnMut(OpRef)| {
            visit(op.pos().get());
            for arg in op.args_slice().iter() {
                visit(arg.to_opref());
            }
            if let Some(fail_args) = op.guard_fail_args() {
                for arg in fail_args.iter() {
                    visit(arg.to_opref());
                }
            }
        };
        for arg in &self.end_args {
            visit(*arg);
        }
        for arg in &self.next_iteration_args {
            visit(arg.to_opref());
        }
        for arg in &self.renamed_inputargs {
            visit(*arg);
        }
        for arg in &self.short_inputargs {
            visit(*arg);
        }
        // The original label/virtual positions that `short_inputargs` rename
        // are reached through the `exported_short_boxes` walk below: every
        // `ShortInputArg` entry carries its original as `preamble_op.res` and
        // as the `SameAs*` op's `pos` (visited via `visit_op`). Const-folded
        // slots never survive into Phase 2, so they need no high-water cover.
        for arg in &self.runtime_boxes {
            visit(*arg);
        }

        for (key, info) in &self.exported_infos {
            visit(key.to_opref());
            if let crate::optimizeopt::info::OpInfo::Ptr(rc) = info {
                let children = rc.borrow().visitor_walk_recursive();
                for child in children {
                    visit(child);
                }
            }
        }

        for preamble_op in self
            .exported_short_boxes
            .iter()
            .chain(self.const_short_boxes.iter())
        {
            visit_op(&preamble_op.op, &mut visit);
            if let Some(source_op) = preamble_op.source_op.as_ref() {
                visit_op(source_op, &mut visit);
            }
            visit(preamble_op.res.to_opref());
            if let Some(source) = preamble_op.same_as_source.as_ref() {
                visit(source.to_opref());
            }
        }

        if let Some(short_preamble) = &self.short_preamble {
            for r in short_preamble
                .inputargs
                .iter()
                .chain(short_preamble.used_boxes.iter())
                .chain(short_preamble.jump_args.iter())
            {
                visit(*r);
            }
            for short_op in &short_preamble.ops {
                visit_op(&short_op.op, &mut visit);
            }
            for ptr_info in short_preamble.inputarg_infos.iter().flatten() {
                for child in ptr_info.visitor_walk_recursive() {
                    visit(child);
                }
            }
        }

        // `phase1_emit_high_water` is `OptContext::next_pos` at end of
        // Phase 1 — covers every emit OpRef the recorder / optimizer
        // allocated, including intermediates forwarded/folded before
        // they could reach a structure-stored field. Phase 2 / retrace
        // must seed `start_fresh` strictly above this watermark to keep
        // the OpRef set disjoint.
        high = high.max(self.phase1_emit_high_water);
        // `partial_trace.inputargs` / `partial_trace.operations` cover
        // every preamble-pass `AbstractValue` whose `_forwarded` must
        // survive into `compile_retrace`; raise the high-water mark
        // above any position they reference so the retrace's fresh
        // OpRef namespace cannot collide with them.
        for ia in &self.partial_trace_inputargs {
            high = high.max(ia.index.saturating_add(1));
        }
        for op in &self.partial_trace_operations {
            let pos = op.pos().get();
            if !pos.is_none() && !pos.is_constant() {
                high = high.max(pos.raw().saturating_add(1));
            }
        }

        high
    }

    /// Push all GcRef values from exported_infos and virtual_state to
    /// shadow stack. gcreftracer.py parity: GC can run between Phase 1
    /// and Phase 2, and between Phase 1 and retrace.
    ///
    /// Must be called explicitly after construction — not auto-called in
    /// new(). This enables LIFO-correct rooting: root the longer-lived
    /// copy first (lower shadow stack depth), then the shorter-lived copy.
    pub fn root_all_gcrefs(&mut self) {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        use crate::optimizeopt::virtualstate::VirtualStateInfo;
        // Idempotency: a retrace can feed the same ExportedState back
        // through `imported_state` and call `root_all_gcrefs` again.
        // Without releasing the previous batch first, the earlier roots
        // get pinned (`release_roots` only pops to the most recent
        // `shadow_stack_base`).
        self.release_roots();
        self.shadow_stack_base = majit_gc::shadow_stack::depth();
        // ── exported_infos GcRef fields ──
        let mut keys: Vec<Operand> = self.exported_infos.keys().cloned().collect();
        // Sort for determinism via the key's resolved OpRef position. Const
        // variants (history.py:189-220) sort separately via the inline
        // payload — `.raw()` would panic on inline-Const OpRefs.
        keys.sort_by_key(|k| match k.to_opref() {
            OpRef::ConstInt(v) => (1u8, v as u64),
            OpRef::ConstFloat(v) => (1u8, v.to_bits()),
            OpRef::ConstPtr(v) => (1u8, v.0 as u64),
            other => (0u8, other.raw() as u64),
        });
        for key in keys {
            if let Some(OpInfo::Ptr(rc)) = self.exported_infos.get(&key) {
                let info = rc.borrow();
                match &*info {
                    // RPython ConstPtrInfo: GcRef constant stored directly in
                    // PtrInfo. Root the GcRef so GC keeps it alive.
                    PtrInfo::Constant(gcref) if !gcref.is_null() => {
                        let ss_idx = majit_gc::shadow_stack::push(*gcref);
                        self.rooted_refs.push((
                            key,
                            ExportedGcRefField::InfoPtrInfoConstant,
                            ss_idx,
                        ));
                    }
                    // InstancePtrInfo.known_class is an immortal vtable integer
                    // (ConstInt), never a traced ref — no rooting needed.
                    _ => {}
                }
            }
        }
        // ── virtual_state GcRef fields ──
        // VirtualStateInfo::KnownClass, Virtual{known_class}, Constant(Ref)
        // The rooted_refs `key` slot here just tags the field origin;
        // these entries dispatch on the field kind and never key back into
        // `exported_infos`, so use the `Operand::None` sentinel.
        let dummy_key = Operand::None;
        for (i, entry) in self.virtual_state.state.iter().enumerate() {
            match &entry.info {
                // KnownClass.class_ptr and Virtual.known_class are immortal
                // vtable integers (ConstInt), never traced refs — not rooted.
                VirtualStateInfo::Constant(Value::Ref(gcref)) if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs.push((
                        dummy_key.clone(),
                        ExportedGcRefField::VirtualStateConstantRef(i),
                        ss_idx,
                    ));
                }
                _ => {}
            }
        }
        // ── partial_trace `_forwarded` GcRef fields ──
        // `partial_trace.inputargs` / `partial_trace.operations`
        // (compile.py:362) keep every preamble-pass `AbstractValue`
        // instance alive; their `_forwarded` slots may carry GcRef
        // payloads (`PtrInfo::Constant` or `Const::Ref`; `Instance.known_class`
        // is an immortal vtable integer, never a traced ref). Root each so
        // `compile_retrace` chain walks observe live handles.
        for (i, ia) in self.partial_trace_inputargs.iter().enumerate() {
            let forwarded = ia.forwarded.borrow().clone();
            root_forwarded_gcref(
                &forwarded,
                ExportedGcRefField::PartialTraceInputArgInfoPtrInfoConstant(i),
                ExportedGcRefField::PartialTraceInputArgConstRef(i),
                dummy_key.clone(),
                &mut self.rooted_refs,
            );
        }
        for (i, op) in self.partial_trace_operations.iter().enumerate() {
            let forwarded = op.forwarded().borrow().clone();
            root_forwarded_gcref(
                &forwarded,
                ExportedGcRefField::PartialTraceOpInfoPtrInfoConstant(i),
                ExportedGcRefField::PartialTraceOpConstRef(i),
                dummy_key.clone(),
                &mut self.rooted_refs,
            );
        }

        let mut rooted_const_ptr_slots = Vec::new();
        self.walk_const_ptr_refs_mut(&mut |slot| {
            let ss_idx = majit_gc::shadow_stack::push(*slot);
            rooted_const_ptr_slots.push(ss_idx);
        });
        self.rooted_const_ptr_slots = rooted_const_ptr_slots;
    }

    /// Update GcRef values from shadow stack — GC may have moved objects.
    ///
    /// VirtualStateInfo top-level entries are stored as
    /// `Rc<VirtualStateInfoNode>` (so two aliased jump args can share a single
    /// `Rc`); GcRef updates have to replace the entire `Rc` with a fresh
    /// one because the inner enum is immutable.
    ///
    /// **Aliasing preservation**: RPython virtualstate.py
    /// `VirtualStateConstructor.create_state` caches by Python object
    /// identity, so a GC pause never breaks the "two aliased jump args
    /// share one VirtualStateInfo" invariant. Rust's `Rc<...>` is immutable
    /// after construction, so each per-slot GcRef refresh would otherwise
    /// allocate an independent new `Rc` for every shared slot, breaking
    /// the `Rc::as_ptr` dedup the walker relies on. Snapshot the original
    /// `Rc::as_ptr` per slot, group slots that
    /// originally shared an `Rc`, and after the GcRef updates re-clone a
    /// single canonical `Rc` into every slot of each group so the
    /// post-refresh tree preserves the pre-GC aliasing.
    pub fn refresh_from_gc(&mut self) {
        use crate::optimizeopt::info::PtrInfo;
        use crate::optimizeopt::virtualstate::{VirtualStateInfo, VirtualStateInfoNode};
        use std::rc::Rc;
        // virtualstate.py create_state cache parity: snapshot original Rc identities
        // so we can re-share post-update slots that aliased pre-GC.
        let original_ptrs: Vec<usize> = self
            .virtual_state
            .state
            .iter()
            .map(|rc| Rc::as_ptr(rc) as usize)
            .collect();
        let mut virtual_state_dirty = false;
        for &(ref key, ref field, ss_idx) in &self.rooted_refs {
            let updated = majit_gc::shadow_stack::get(ss_idx);
            match field {
                ExportedGcRefField::InfoPtrInfoConstant => {
                    if let Some(info) = self.exported_infos.get_mut(key) {
                        *info = crate::optimizeopt::info::OpInfo::ptr(PtrInfo::Constant(updated));
                    }
                }
                ExportedGcRefField::VirtualStateConstantRef(i) => {
                    if let Some(entry) = self.virtual_state.state.get_mut(*i) {
                        *entry = VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                            Value::Ref(updated),
                        ));
                        virtual_state_dirty = true;
                    }
                }
                ExportedGcRefField::PartialTraceInputArgInfoPtrInfoConstant(i) => {
                    if let Some(ia) = self.partial_trace_inputargs.get(*i) {
                        refresh_forwarded_ptrinfo_constant(&ia.forwarded.borrow(), updated);
                    }
                }
                ExportedGcRefField::PartialTraceInputArgConstRef(i) => {
                    if let Some(ia) = self.partial_trace_inputargs.get(*i) {
                        refresh_forwarded_const_ref(&ia.forwarded.borrow(), updated);
                    }
                }
                ExportedGcRefField::PartialTraceOpInfoPtrInfoConstant(i) => {
                    if let Some(op) = self.partial_trace_operations.get(*i) {
                        refresh_forwarded_ptrinfo_constant(&op.forwarded().borrow(), updated);
                    }
                }
                ExportedGcRefField::PartialTraceOpConstRef(i) => {
                    if let Some(op) = self.partial_trace_operations.get(*i) {
                        refresh_forwarded_const_ref(&op.forwarded().borrow(), updated);
                    }
                }
            }
        }
        if virtual_state_dirty {
            // Re-share slots that originally aliased: walk the snapshot
            // map and copy each group's first canonical Rc into every
            // peer slot, restoring the pre-GC `Rc::as_ptr` equivalences.
            let mut canonical_by_old: crate::FxIndexMap<usize, Rc<VirtualStateInfoNode>> =
                crate::FxIndexMap::default();
            for (slot_idx, &old_ptr) in original_ptrs.iter().enumerate() {
                let entry = canonical_by_old.entry_or_insert_with(old_ptr, || {
                    Rc::clone(&self.virtual_state.state[slot_idx])
                });
                if !Rc::ptr_eq(entry, &self.virtual_state.state[slot_idx]) {
                    self.virtual_state.state[slot_idx] = Rc::clone(entry);
                }
            }
            // Rc identities have shifted, so position cells must be
            // re-derived against the canonical state graph.
            self.virtual_state.enum_top_level();
        }
        let rooted_const_ptr_slots = self.rooted_const_ptr_slots.clone();
        let mut root_idx = 0usize;
        self.walk_const_ptr_refs_mut(&mut |slot| {
            if let Some(&ss_idx) = rooted_const_ptr_slots.get(root_idx) {
                *slot = majit_gc::shadow_stack::get(ss_idx);
            }
            root_idx = root_idx.saturating_add(1);
        });
        debug_assert_eq!(
            root_idx,
            rooted_const_ptr_slots.len(),
            "ExportedState ConstPtr root walk changed between root_all_gcrefs and refresh_from_gc"
        );
    }

    /// Release shadow stack roots.
    fn release_roots(&mut self) {
        if !self.rooted_refs.is_empty() || !self.rooted_const_ptr_slots.is_empty() {
            majit_gc::shadow_stack::pop_to(self.shadow_stack_base);
            self.rooted_refs.clear();
            self.rooted_const_ptr_slots.clear();
        }
    }

    /// unroll.py: final() — ExportedState is never final (loop continues).
    pub fn is_final(&self) -> bool {
        false
    }
}

impl Clone for ExportedState {
    /// Pure data clone — no shadow stack side effects.
    ///
    /// RPython has no clone (single Python object shared by reference).
    /// When a Rust clone is stored long-term (across potential GC points),
    /// the caller must call root_all_gcrefs() explicitly.
    fn clone(&self) -> Self {
        ExportedState {
            end_args: self.end_args.clone(),
            label_source_positions: self.label_source_positions.clone(),
            next_iteration_args: self.next_iteration_args.clone(),
            end_arg_types: self.end_arg_types.clone(),
            virtual_state: self.virtual_state.clone(),
            exported_infos: self.exported_infos.clone(),
            exported_short_boxes: self.exported_short_boxes.clone(),
            const_short_boxes: self.const_short_boxes.clone(),
            short_boxes: self.short_boxes.clone(),
            short_preamble: self.short_preamble.clone(),
            renamed_inputargs: self.renamed_inputargs.clone(),
            short_inputargs: self.short_inputargs.clone(),
            short_inputarg_refs: self.short_inputarg_refs.clone(),
            runtime_boxes: self.runtime_boxes.clone(),
            patchguardop: self.patchguardop.clone(),
            quasi_immutable_deps: self.quasi_immutable_deps.clone(),
            phase1_emit_high_water: self.phase1_emit_high_water,
            partial_trace_inputargs: self.partial_trace_inputargs.clone(),
            partial_trace_operations: self.partial_trace_operations.clone(),
            short_box_producer_roots: self.short_box_producer_roots.clone(),
            rooted_refs: Vec::new(),
            rooted_const_ptr_slots: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
    }
}

impl Drop for ExportedState {
    fn drop(&mut self) {
        self.release_roots();
    }
}

/// unroll.py: UnrollInfo(BasicLoopInfo) — return type from optimize_peeled_loop.
///
/// Carries the target_token, label_op, and extra_same_as needed to
/// finalize compilation after the peeled loop body is optimized.
#[derive(Clone, Debug)]
pub struct UnrollInfo {
    /// The target token for this loop's entry point.
    pub target_token: u64,
    /// Extra same_as ops added during finalization.
    pub extra_same_as: Vec<Op>,
    /// Quasi-immutable dependencies discovered during optimization
    /// (`optimizer.py:243` + `heap.py:821-823`). Vec-backed set with
    /// linear-scan dedup on instance identity.
    pub quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    /// Extra ops to insert before the label (from bridge inlining).
    pub extra_before_label: Vec<Op>,
}

impl UnrollInfo {
    /// unroll.py: final() — UnrollInfo is always final.
    pub fn is_final(&self) -> bool {
        true
    }
}

impl OptUnroll {
    /// unroll.py disable_retracing_if_max_retrace_guards.
    ///
    /// When the peeled body contains more guards than `max_retrace_guards`,
    /// set `retraced_count = u32::MAX` on the targeting JitCellToken so that
    /// future bridges jump to the preamble instead of requesting a retrace.
    pub fn disable_retracing_if_max_retrace_guards<T: AsRef<Op>>(
        ops: &[T],
        retraced_count: &mut u32,
        max_retrace_guards: u32,
    ) {
        let guard_count = ops
            .iter()
            .filter(|op| op.as_ref().opcode.is_guard())
            .count();
        if guard_count > max_retrace_guards as usize {
            *retraced_count = u32::MAX;
        }
    }

    /// unroll.py: export_state — capture optimizer state at end of preamble.
    ///
    /// After the preamble is optimized, snapshot:
    /// - end_args: forced versions of label args
    /// - virtual_state: abstract info for loop-carried values
    /// - short boxes: mapping of preamble ops to label args
    pub fn export_state(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> ExportedState {
        self.export_state_with_bounds(original_label_args, renamed_inputargs, optimizer, ctx, None)
    }

    /// unroll.py: export_state implementation.
    fn export_state_with_bounds(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        exported_int_bounds: Option<
            &crate::FxIndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
    ) -> ExportedState {
        // The preview's VS/inputarg evaluation, when it ran to completion.
        // `None` covers both "no preview ran" and the bridge arm that broke out
        // of the export before the VS capture; the latter still left
        // `ctx.preamble_end_args` behind, which the `end_args` fallback below
        // prefers over a recompute.
        let preview_args_state = ctx
            .preview_short_state
            .as_ref()
            .and_then(|preview| preview.args_state.clone());
        // unroll.py:454: end_args = [force_at_the_end_of_preamble(a) ...].
        // The preview path carried the exact Box objects used for this single
        // evaluation; their OpRefs are only the positional view of those boxes.
        let end_args: Vec<OpRef> = preview_args_state.as_ref().map_or_else(
            || {
                ctx.preamble_end_args.clone().unwrap_or_else(|| {
                    original_label_args
                        .iter()
                        .map(|&a| ctx.get_replacement_opref(a))
                        .collect()
                })
            },
            |(_, _, _, _, end_arg_boxes)| end_arg_boxes.iter().map(Operand::to_opref).collect(),
        );
        // unroll.py `virtual_state = self.get_virtual_state(end_args)`
        // — VS captured AFTER `force_box_for_end_of_preamble` and AFTER
        // `flush()`. The caller (`Optimizer::optimize_with_constants_and_inputs_at`)
        // already ran both passes before invoking us, so `end_args` is in
        // the same post-force, post-flush state RPython feeds in.
        let virtual_state = preview_args_state.as_ref().map_or_else(
            || crate::optimizeopt::virtualstate::export_state(&end_args, ctx),
            |(virtual_state, _, _, _, _)| virtual_state.clone(),
        );
        // unroll.py: infos = {}; for arg in end_args: _expand_info(arg, infos)
        let mut infos: indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo> =
            indexmap::IndexMap::new();
        // Resolve the ONE canonical box per end_arg up front: it is the
        // exported_infos key AND (unroll.py:467 next_iteration_args = end_args)
        // the carried import key, so they are the identical Rc and import_state's
        // lookup is an identity hit for const / inputarg / resop alike. Computing
        // it once keeps a single canonical Const cell per arg (Operand::Const
        // compares by cell identity, so re-resolving is tolerable but one cell
        // mirrors RPython's single Const box object).
        let end_arg_boxes: Vec<Operand> = preview_args_state.as_ref().map_or_else(
            || {
                end_args
                    .iter()
                    .map(|&a| match ctx.get_box_replacement_operand_opt(a) {
                        Some(o) => o,
                        // The None arm fires only for an unregistered ResOp
                        // position (Const / InputArg always resolve); #157
                        // drained those fires to zero. Materializing once keeps
                        // the export/import key identity-stable.
                        None => ctx.materialize_operand_at(a),
                    })
                    .collect()
            },
            |(_, _, _, _, end_arg_boxes)| end_arg_boxes.clone(),
        );
        for (arg, arg_box) in end_args.iter().zip(end_arg_boxes.iter()) {
            self.expand_info(*arg, arg_box, ctx, exported_int_bounds, &mut infos);
        }
        // unroll.py:462-463 `label_args, virtuals =
        //   virtual_state.make_inputargs_and_virtuals(end_args, self.optimizer)`.
        let (label_args, virtuals, label_source_positions) = if let Some((
            _,
            label_args,
            virtuals,
            label_source_positions,
            _,
        )) = preview_args_state
        {
            (label_args, virtuals, label_source_positions)
        } else {
            let (label_boxes, virtuals, label_source_positions) = virtual_state
                .make_inputargs_and_virtuals_with_source_positions(&end_args, optimizer, ctx, false)
                .expect("export_state make_inputargs_and_virtuals failed");
            (
                label_boxes.iter().map(|b| b.to_opref()).collect(),
                virtuals,
                label_source_positions,
            )
        };
        if crate::callee_rca_enabled() {
            eprintln!(
                "[callee-rca][export-state] original_label_args={:?} end_args={:?} \
                 label_source_positions={:?} label_args={:?} virtuals={:?} \
                 renamed_inputargs={:?} num_boxes={} entries={}",
                original_label_args,
                end_args,
                label_source_positions,
                label_args,
                virtuals,
                renamed_inputargs,
                virtual_state.num_boxes(),
                virtual_state.num_entries(),
            );
            for line in callee_rca_virtual_state_summary(&virtual_state) {
                eprintln!("[callee-rca][export-vs] {line}");
            }
        }
        // unroll.py: for arg in label_args: _expand_info(arg, infos)
        for &arg in &label_args {
            let arg_box = match ctx.get_box_replacement_operand_opt(arg) {
                Some(o) => o,
                // Same canonical-key fallback as end_arg_boxes above: an
                // unregistered ResOp position (drained to zero by #157) mints a
                // canonical registered synthetic, keeping the exported_infos
                // key identity-stable.
                None => ctx.materialize_operand_at(arg),
            };
            self.expand_info(arg, &arg_box, ctx, exported_int_bounds, &mut infos);
        }
        let mut short_args = label_args.to_vec();
        short_args.extend(virtuals);
        // unroll.py:466-480 `short_boxes` / `short_inputargs` — read the
        // ShortBoxes-derived lists off the published `PreviewShortState`. The
        // preview pass computed the inputargs from the same `label_args +
        // virtuals` (measured identical across the corpus); paths that never
        // ran the preview (test ExportedState setups) fall back to the local
        // recompute and carry no short boxes at all.
        //
        // Detach the publication before the `None` arm below takes `ctx`
        // mutably to mint its own renamed inputargs.
        let published = ctx.preview_short_state.as_ref().map(|preview| {
            (
                preview.short_inputargs.clone(),
                preview.short_inputarg_refs.clone(),
                preview.short_boxes.clone(),
            )
        });
        let (short_inputargs, short_inputarg_refs, exported_short_boxes) = match published {
            None => {
                // No preview pass ran (test ExportedState setups): mint the fresh
                // renamed InputArg positions directly, mirroring `add_short_input_arg`
                // (shortpreamble.py `OpHelpers.inputarg_from_tp(box.type)`).
                // Each is DISTINCT from its `short_args[i]` original so the
                // rename is a real rename, not an identity no-op. Build the
                // rooted `InputArgRc` pool alongside, matching the preview pass.
                let mut inputargs = Vec::with_capacity(short_args.len());
                let mut refs = Vec::with_capacity(short_args.len());
                for &a in &short_args {
                    let ty = a.ty().unwrap_or_else(|| {
                        panic!("short preamble inputarg {a:?} has no value type")
                    });
                    let pos = ctx.alloc_op_position_typed(ty).raw();
                    let ia = majit_ir::InputArg::from_type_rc(ty, pos);
                    inputargs.push(ia.opref());
                    refs.push(ia);
                }
                (inputargs, refs, Vec::new())
            }
            Some((inputargs, refs, short_boxes)) => {
                debug_assert_eq!(
                    inputargs.len(),
                    short_args.len(),
                    "preview-pass create_short_inputargs length diverged from the \
                     export-site label_args + virtuals recompute"
                );
                (inputargs, refs, short_boxes)
            }
        };
        let const_short_boxes = ctx.exported_const_short_boxes.clone();
        // #173 producer-rooting: capture each exported short-box's Phase-1
        // producer Op (`res.bound_op()`) while the Phase-1 ctx still owns the
        // strong `OpRc`, so `res`'s Weak still upgrades after the peel boundary
        // drops that ctx. InputArg-kind res is `bound_inputarg` (None here),
        // rooted via `short_inputarg_refs` instead.
        let short_box_producer_roots: Vec<majit_ir::OpRc> = exported_short_boxes
            .iter()
            .filter_map(|e| e.res.bound_op())
            .collect();
        // unroll.py:466-473 line-by-line:
        //
        //   sb = ShortBoxes()
        //   short_boxes = sb.create_short_boxes(self.optimizer,
        //       renamed_inputargs, label_args + virtuals)
        //   short_inputargs = sb.create_short_inputargs(label_args + virtuals)
        //   for produced_op in short_boxes:
        //       op = produced_op.short_op.res
        //       if not isinstance(op, Const):
        //           self._expand_info(op, infos)
        //
        // RPython expands info from the post-`create_short_boxes` list (the
        // same `short_boxes` that survives into `ExportedState.short_boxes`).
        // pyre's analog is `produced_short_boxes_from_exported_boxes` (which
        // applies the GuardOverflow filter); iterating the raw
        // `exported_short_boxes` here would expand info for entries (e.g.
        // standalone `GuardOverflow`) that PyPy never carries into
        // `short_boxes`, polluting the dict.
        let short_boxes_for_info =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &exported_short_boxes,
            );
        // unroll.py:481-484:
        //     for produced_op in short_boxes:
        //         op = produced_op.short_op.res
        //         if not isinstance(op, Const):
        //             self._expand_info(op, infos)
        // Key by the short_op result BOX (`produced_op.res`). Every
        // `produced_short_boxes_from_exported_boxes` invocation clones `res`
        // from the same `exported_short_boxes[k].res`
        // (`produced_short_boxes_from_exported_boxes` in `shortpreamble.rs`),
        // so the consumer-side
        // `initialize_imported_short_preamble_builder_from_short_boxes` lookup
        // re-derives the identical Rc and the box-identity lookup hits.
        for (_, produced_op) in &short_boxes_for_info {
            let op = produced_op.res.to_opref();
            if !op.is_constant() {
                // `expand_info` keys the operand-keyed `exported_infos`
                // (#158) on the producer-bound `res`, which is
                // `Rc::ptr_eq`-stable on the producer, so the import lookup
                // still hits.
                self.expand_info(op, &produced_op.res, ctx, exported_int_bounds, &mut infos);
            }
        }

        // RPython unroll.py:467: next_iteration_args = end_args (post-force).
        // Aliased boxes (same resolved OpRef) are handled by export_state's
        // create_state cache + make_inputargs' position_in_notvirtuals dedup.
        // RPython parity: store next_iteration_args in their post-forwarding
        // form so the cache key invariant established by `export_single_value`
        // (which uses `get_box_replacement` as the cache key, in `virtualstate.rs`)
        // survives the phase boundary. `import_state` later runs in a fresh
        // Phase 2 ctx that does NOT inherit Phase 1's forwarding map; if we
        // stored raw `end_args[i]` here, two top-level entries that shared
        // the same `VirtualStateInfoNode` Rc (because their post-resolution
        // targets coincided in Phase 1) would `make_equal_to(source(i), raw_i)`
        // to *different* raw OpRefs, and `make_inputargs` in Phase 2 would
        // see divergent forwarding for the supposedly-shared state slot.
        // RPython sidesteps this naturally because `box._forwarded` persists
        // across phases. virtualstate.py make_inputargs assumes shared state
        // ⇒ same forwarded target.
        // unroll.py:458/467 `end_args = [get_box_replacement(arg) ...]`;
        // `next_iteration_args = end_args`. Carry the SAME canonical box Rcs
        // already resolved as the exported_infos keys (`end_arg_boxes`), so the
        // import lookup is a ptr_eq hit. Each box's `.to_opref()` yields the
        // resolved (post-forwarding) position, so consumers that read `.to_opref()`
        // see the same value the prior `get_replacement_opref` form produced.
        let resolved_next_iteration_args: Vec<Operand> = end_arg_boxes;
        // `produced_short_boxes` is derived from
        // `exported_short_boxes` lazily at the consumer site
        // (`build_short_preamble_from_produced_boxes` in `import_state`)
        // via `produced_short_boxes_from_exported_boxes`
        // (in `shortpreamble.rs`). Storing both shapes on
        // `ExportedState` would risk drift across mutations.
        let mut state = ExportedState::new(
            label_args.clone(),
            label_source_positions,
            resolved_next_iteration_args,
            virtual_state,
            infos,
            exported_short_boxes,
            const_short_boxes,
            renamed_inputargs.to_vec(),
            short_inputargs,
            short_inputarg_refs,
        );
        // #173: install the Phase-1 producer roots captured above. Kept off the
        // positional `new` ctor (callers write fields post-construct).
        state.short_box_producer_roots = short_box_producer_roots;
        // unroll.py:234: carry deps collected while optimizing the exported
        // preamble state so compile.py:384-390 can merge them on retrace.
        state.quasi_immutable_deps = optimizer.quasi_immutable_deps.clone();
        // `OptContext::next_pos` is the strict upper bound on raw OpRefs
        // Phase 1 allocated, including intermediates folded / forwarded
        // away before any structure-stored field could observe them.
        // `reserve_pos_typed` skips `materialize_operand_at` on the zero-inputarg /
        // retrace baselines (`optimizeopt/mod.rs`), so capturing
        // `ctx.next_pos` at export is the only reliable floor for
        // `opref_high_water()` to feed retrace's `start_fresh`.
        state.phase1_emit_high_water = ctx.next_pos;
        // `partial_trace.inputargs` / `partial_trace.operations`
        // (compile.py:362) identity carriage: snapshot the InputArgRc /
        // OpRc lists the preamble pass mutated `_forwarded` on. A later
        // `compile_retrace` reads `Op.forwarded` / `InputArg.forwarded`
        // directly off the same objects (resoperation.py AbstractResOpOrInputArg / :700),
        // the PyPy-orthodox identity carry.
        //
        // Walk `ctx.inputargs` (the canonical inputarg-order OpRef list)
        // and pick up the `InputArgRc` for each inputarg's raw OpRef
        // position. `inputarg_refs[idx]` is the canonical strong-owner,
        // populated by either `with_inputarg_types`,
        // `ensure_inputarg_bindings`, or retrace prefix import; reading it
        // directly preserves any `_forwarded` state the Phase 1 passes wrote
        // on the original `InputArg` object (resoperation.py:700 `_forwarded`
        // host). Read it only when it actually matches the inputarg's type
        // and index; last-resort fresh allocation when it carries no matching
        // Rc (test fixtures / type-mismatch edge cases).
        state.partial_trace_inputargs = ctx
            .inputargs
            .iter()
            .map(|ia_opref| {
                let idx = ia_opref.raw() as usize;
                let want_ty = ia_opref.ty().unwrap_or(majit_ir::Type::Void);
                if let Some(rc) = ctx.inputarg_refs.get(&(idx as u32)).cloned()
                    && rc.tp.get() == want_ty
                    && rc.index == idx as u32
                {
                    return rc;
                }
                // `inputarg_refs[idx]` is the canonical `InputArgRc` for this
                // OpRef once `ensure_inputarg_bindings` / `materialize_operand_at`'s
                // InputArg placeholder arm have run (both write the canonical
                // `InputArgRc` matching the OpRef's type). Fall through to a
                // fresh `InputArg` allocation for the type-mismatch edge case.
                //
                // Reaching here means the slot was either absent or
                // type-mismatched. In production every canonical inputarg
                // is seeded by `ensure_inputarg_bindings`, so the slot must
                // be *present* (and we only fall through on a genuine type
                // mismatch); an absent slot would mean a fresh allocation
                // silently drops the Phase-1 `_forwarded` state. Trip a
                // future binding regression into a test failure here rather
                // than letting it lose state silently.
                debug_assert!(
                    ctx.inputarg_refs.contains_key(&(idx as u32)),
                    "partial_trace_inputargs: inputarg slot {idx} unpopulated at \
                     unroll close; ensure_inputarg_bindings must seed every canonical \
                     inputarg so this fresh allocation is only a type-mismatch repair",
                );
                majit_ir::InputArgRc::new(majit_ir::InputArg::from_type(want_ty, idx as u32))
            })
            .collect();
        state.partial_trace_operations = optimizer.phase1_emit_ops.clone();
        state
    }

    /// unroll.py: _expand_info
    fn expand_info(
        &self,
        arg: OpRef,
        arg_box: &Operand,
        ctx: &OptContext,
        exported_int_bounds: Option<
            &crate::FxIndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
        infos: &mut indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
    ) {
        // unroll.py `_expand_info`:
        //     if arg in infos:
        //         return
        //     if info:
        //         infos[arg] = info
        //         if info.is_virtual():
        //             self._expand_infos_from_virtual(info, infos)
        //
        // Keyed by `arg_box`, the ONE canonical Phase-1 box the caller resolved
        // for this arg (shared verbatim with the `next_iteration_args` carry so
        // the import lookup is a ptr_eq hit). The dual OpRef-key insert is gone:
        // a ptr-stable operand key tracks its position through the shared
        // set_position Cell across compaction, so there is no resolved-vs-original
        // OpRef drift left to bridge.
        if infos.contains_key(arg_box) {
            return;
        }
        let resolved = ctx.get_replacement_opref(arg);
        // RPython stores the entry only when `info` is truthy — a falsy
        // `info` (None) simply skips the insert, so downstream
        // `setinfo_from_preamble_list` at unroll.py sees "no entry"
        // and drops any inherited forwarded via `item.set_forwarded(None)`.
        let Some(info) = self.collect_exported_info(resolved, ctx, exported_int_bounds) else {
            return;
        };
        // `arg_box` is the canonical Phase-1 box, which can be position-only
        // (a virtual field box), so read its PtrInfo directly off the box —
        // exactly what `peek_ptr_info` does (`get_box_replacement(false)
        // .ptr_info()`, `self`-independent) — without the `from_opref`
        // position-only panic.
        let arg_pi = arg_box
            .get_box_replacement(false)
            .ptr_info()
            .map(|p| p.clone());
        let has_fields = matches!(
            arg_pi,
            Some(pi) if pi.is_virtual() || !pi.all_items().is_empty()
        );
        infos.insert(arg_box.clone(), info);
        if has_fields {
            self.expand_infos_from_virtual(resolved, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py: _expand_infos_from_virtual
    fn expand_infos_from_virtual(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<
            &crate::FxIndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
        infos: &mut indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
    ) {
        let opref_box = ctx.get_box_replacement_operand_opt(opref);
        // unroll.py `_expand_infos_from_virtual`:
        //     items = info.all_items()
        //     for item in items:
        //         if item is None: continue
        //         self._expand_info(item, infos)
        // Key each field by its raw `all_items()` box, NOT a re-resolved OpRef.
        // `peek_ptr_info` returns the SAME `Rc<RefCell<PtrInfo>>` cell whose
        // handle `collect_exported_info` stores as the `exported_infos` value
        // (its `ptr_info_handle()` read), so the import-side reader
        // (`setinfo_from_preamble_list`) walks the same `v.fields` and the
        // box-identity (`Rc::ptr_eq`) lookup hits.
        let Some(info) = opref_box
            .as_ref()
            .and_then(|b| b.get_box_replacement(false).ptr_info().map(|p| p.clone()))
        else {
            return;
        };
        for (_, entry) in info.all_items() {
            let field_box = entry.as_seen_operand();
            if field_box.to_opref().is_none() {
                continue;
            }
            self.expand_info(
                field_box.to_opref(),
                &field_box,
                ctx,
                exported_int_bounds,
                infos,
            );
        }
    }

    /// unroll.py: finalize_short_preamble — create a TargetToken
    /// and attach the short preamble to it. Called at the end of
    /// optimize_peeled_loop after the loop body is optimized.
    ///
    /// Returns the new TargetToken with virtual_state and short_preamble set.
    pub fn finalize_short_preamble(
        &self,
        token_id: u64,
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        short_preamble: crate::optimizeopt::shortpreamble::ShortPreamble,
        short_preamble_builder: Option<&crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    ) -> (
        std::sync::Arc<TargetToken>,
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    ) {
        let target_token = TargetToken::new_loop(token_id);
        {
            let mut attrs = target_token.attrs();
            attrs.virtual_state = Some(virtual_state);
            attrs.short_preamble = Some(short_preamble);
        }
        let short_preamble_producer = short_preamble_builder.map(|builder| {
            crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder::new(
                target_token.as_jump_target_descr(),
                builder,
            )
        });
        (target_token, short_preamble_producer)
    }

    /// unroll.py: _jump_to_existing_trace — check if any existing
    /// compiled trace (target_token) has a compatible virtual state.
    /// If so, generate extra guards, inline short preamble, and redirect jump.
    ///
    /// Returns None if jumped successfully, Some(virtual_state) otherwise.
    /// unroll.py: jump_to_existing_trace
    ///
    /// `runtime_boxes`: the concrete runtime values at the jump point
    /// (unroll.py:153/166/207/222), used by generate_guards to emit
    /// GUARD_VALUE when the runtime value matches a known constant. Always a
    /// position-aligned list — both the loop (`state.runtime_boxes`) and the
    /// bridge (`optimize_bridge`'s `runtime_boxes`) paths supply it.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn jump_to_existing_trace(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &[std::sync::Arc<TargetToken>],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: &[OpRef],
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        self.jump_to_existing_trace_with_vs(
            jump_args,
            current_label_args,
            target_tokens,
            optimizer,
            ctx,
            force_boxes,
            runtime_boxes,
            None,
        )
    }

    /// Like jump_to_existing_trace, but with an optional pre-computed
    /// virtual_state. Used by optimize_bridge where force_at_the_end_of_preamble
    /// may change forwarding chains after the virtual state was exported.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn jump_to_existing_trace_with_vs(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &[std::sync::Arc<TargetToken>],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        // unroll.py `with self.optimizer.cant_replace_guards():`
        // line-by-line — save current `can_replace_guards`, set False
        // for the guarded section, restore on exit. Nested scopes
        // preserve the outer False via the saved token. An InvalidLoop is
        // recorded as a deferred signal on `ctx` (no unwinding), so a plain
        // call + restore preserves the "restore on exit" contract.
        let guard = optimizer.cant_replace_guards(ctx);
        let result = self.jump_to_existing_trace_impl(
            jump_args,
            current_label_args,
            target_tokens,
            optimizer,
            ctx,
            force_boxes,
            runtime_boxes,
            pre_vs,
        );
        optimizer.restore_can_replace_guards(ctx, guard);
        result
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn jump_to_existing_trace_impl(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &[std::sync::Arc<TargetToken>],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        let mut virtual_state = pre_vs
            .unwrap_or_else(|| crate::optimizeopt::virtualstate::export_state(jump_args, ctx));
        let mut args: Vec<OpRef> = jump_args
            .iter()
            .map(|&a| ctx.get_replacement_opref(a))
            .collect();

        for (tt_idx, target_token) in target_tokens.iter().enumerate() {
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-tracing",
                    &format!("jump_to_existing trying target_token #{tt_idx}"),
                );
            }
            // Cloned out of the token's lock: the body below re-enters the
            // token (`short_preamble`, recipes) while this state is in use.
            let target_vs = match target_token.attrs().virtual_state.clone() {
                Some(vs) => vs,
                None => continue,
            };
            // A LABEL arg appended without a recipe is one no close can
            // supply: the JUMP would land short of the LABEL.
            if target_token.attrs().label_tail_unrebuildable {
                if crate::log_jtet_enabled() {
                    eprintln!(
                        "[jit][jte] target_token #{tt_idx} skipped: LABEL tail has no recipe",
                    );
                }
                continue;
            }

            // RPython unroll.py:333: patchguardop = self.optimizer.patchguardop
            // Ensure ctx.patchguardop is set before generate_guards so that
            // extra guards get a valid rd_resume_position (resume.py:397).
            if ctx.patchguardop.is_none() {
                ctx.patchguardop = optimizer.patchguardop.clone();
            }

            // RPython unroll.py:315 parity: try generate_guards directly
            // instead of gating on generalization_of. If guards can't be
            // generated (VirtualStatesCantMatch), skip this target.
            let extra_guards = match target_vs.generate_guards(
                &virtual_state,
                &args,
                runtime_boxes,
                ctx,
                force_boxes,
            ) {
                Ok(guards) => guards,
                Err(_) => {
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} generate_guards failed (force_boxes={force_boxes})",
                        );
                    }
                    continue;
                }
            };
            // unroll.py:333-338 parity: read patchguardop.rd_resume_position
            // and stamp it onto every extra guard. RPython has no fallback —
            // `optimize_GUARD_FUTURE_CONDITION` (rewrite.py / simplify.py) runs
            // unconditionally on the GUARD_FUTURE_CONDITION emitted at
            // `reached_loop_header` (pyjitpl.py), so by the time
            // `_jump_to_existing_trace` runs `self.optimizer.patchguardop` is
            // always populated.
            //
            // pyre emits the GFC on both closes: the own-loop close gets it
            // from `close_loop_args_at` and Phase 1 captures it into
            // `patchguardop`, which Phase 2 inherits; the merge-point crossing
            // that closes a bridge emits it in `jitcode_dispatch/mod.rs`. A
            // crossing that skipped that emit — a previously declined close, or
            // a walk that is not the authoritative top-level executor — reaches
            // `optimize_bridge` without one, and falls back to the stand-in it
            // synthesizes from one of the bridge's own body guards; a bridge
            // whose body carries no guard with a resume position leaves even
            // that unset. Upstream cannot reach that state, because
            // `reached_loop_header` emits the GFC before the closing JUMP of
            // every trace that reaches it, a bridge grown from a
            // ResumeGuardDescr included.
            //
            // Decline this target token rather than dereference a None, the
            // way the replayed-guard arm of `inline_short_preamble` does.  The
            // check sits here, BEFORE any `send_extra_operation`, because the
            // stamp loop below streams ops into the trace as it goes: bailing
            // out from inside it would leave a partially-guarded target
            // behind.  Declining costs this target token only; the caller
            // tries the next one and otherwise falls back to jump_to_preamble
            // (unroll.py:208-210), so the bridge still compiles.
            if !extra_guards.is_empty() && ctx.patchguardop.is_none() {
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit] jump_to_existing_trace: target_token #{tt_idx} needs \
                         {} extra guard(s) but the trace has no patchguardop — declining",
                        extra_guards.len(),
                    );
                }
                continue;
            }
            // Skip the read when `extra_guards` is empty — RPython's
            // `for guard in extra_guards.extra_guards` loop body is also
            // skipped in that case.
            for guard_req in &extra_guards {
                let emitted = guard_req.to_ops(&args, ctx);
                if emitted.is_empty() {
                    continue;
                }
                // unroll.py:333 reads `patchguardop` once per loop iteration.
                // Pull it eagerly so the stamp branch below stays cheap and
                // the invariant is asserted before any op streams out.
                let patch = ctx.patchguardop.as_ref().unwrap_or_else(|| {
                    panic!(
                        "unroll.py:333 invariant: the non-empty-extra_guards \
                         decline above must have skipped this target token \
                         (target_token #{}, {} extra guard(s), force_boxes={})",
                        tt_idx,
                        extra_guards.len(),
                        force_boxes,
                    )
                });
                let rd_resume_position = patch.rd_resume_position();
                for guard_op in emitted {
                    if crate::log_jtet_enabled() {
                        let arg_values: Vec<_> = guard_op
                            .args_slice()
                            .iter()
                            .map(|arg| {
                                let arg = arg.to_opref();
                                (
                                    arg,
                                    ctx.get_box_replacement_operand_opt(arg)
                                        .and_then(|cb| cb.const_value()),
                                )
                            })
                            .collect();
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} emit guard {:?} from {:?} args={:?}",
                            guard_op.opcode, guard_req, arg_values
                        );
                    }
                    // unroll.py:335-337 line-by-line:
                    //
                    //     if isinstance(guard, GuardResOp):
                    //         guard.rd_resume_position = patchguardop.rd_resume_position
                    //         guard.setdescr(compile.ResumeAtPositionDescr())
                    //     self.optimizer.send_extra_operation(guard)
                    //
                    // intutils.py IntBound.make_guards interleaves
                    // INT_GE/INT_LE/INT_AND (non-GuardResOp) with their
                    // GUARD_TRUE/GUARD_VALUE pairs; only the latter inherit
                    // resume metadata. Mirror the type filter via `is_guard()`.
                    if guard_op.opcode.is_guard() {
                        guard_op.set_rd_resume_position(rd_resume_position);
                        guard_op.setdescr(crate::optimizeopt::make_resume_at_position_descr());
                    }
                    // unroll.py:338 lets send_extra_operation raise InvalidLoop
                    // (only VirtualStatesCantMatch is caught around it). This
                    // function returns Option (flag convention), and
                    // propagate consumed the flag into the Err, so re-defer it
                    // for the caller's take_invalid_loop barrier and abort.
                    if let Err(e) = optimizer.send_extra_operation(&guard_op, ctx) {
                        ctx.signal_invalid_loop(e.0);
                        return None;
                    }
                }
            }

            // unroll.py:346-347: make_inputargs_and_virtuals
            // RPython: force_box emits New/SetfieldGc via emit_extra which
            // routes through passes AFTER Virtualize. The non-virtual PtrInfo
            // (Struct/Instance) on alloc_ref prevents re-absorption.
            let (target_boxes, virtuals) = match target_vs.make_inputargs_and_virtuals(
                &args,
                optimizer,
                ctx,
                force_boxes,
            ) {
                Ok(result) => result,
                Err(_) => {
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} make_inputargs failed (force_boxes={force_boxes})",
                        );
                    }
                    if force_boxes {
                        args = jump_args
                            .iter()
                            .map(|&a| ctx.get_replacement_opref(a))
                            .collect();
                        virtual_state = crate::optimizeopt::virtualstate::export_state(&args, ctx);
                    }
                    continue;
                }
            };
            // unroll.py `_jump_to_existing_trace` `short_jump_args = args + virtuals`:
            // the short preamble's label (`short[0].getarglist()`) carries one
            // inputarg per `args` entry AND one per `virtuals` entry, so
            // `inline_short_preamble`'s `len(short_inputargs) == len(jump_args)`
            // holds.
            //
            let mut short_jump_args = target_boxes.clone();
            short_jump_args.extend(virtuals.iter().map(|&v| {
                if v.is_none() || v.is_constant() {
                    Operand::from_opref(v)
                } else {
                    ctx.materialize_operand_at(v)
                }
            }));

            // Ensure jump_args carry PtrInfo from Phase 2 body.
            // RPython Box identity preserves info across forwarding.
            // In majit, forwarding target may lack PtrInfo — propagate
            // from the original label arg (before forwarding).
            if let Some(label) = current_label_args {
                for (i, jump_box) in short_jump_args.iter().enumerate() {
                    let resolved_has_info = ctx
                        .resolve_operand_operand_opt(jump_box)
                        .as_ref()
                        .is_some_and(|b| ctx.has_ptr_info(b));
                    if !resolved_has_info {
                        // Try label arg at same index
                        if let Some(&label_arg) = label.get(i) {
                            let label_box = ctx.get_box_replacement_operand_opt(label_arg);
                            if let Some(info) =
                                label_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
                            {
                                ctx.ensure_ptr_info_preserve_forwarding(jump_box.to_opref(), info);
                            }
                        }
                    }
                }
            }

            // unroll.py:353-356: inline short preamble
            let mut extra = Vec::new();
            // Bound outside the `if let`: the scrutinee's lock guard would
            // otherwise live through the body, which locks the token again.
            let short_preamble = target_token.attrs().short_preamble.clone();
            if let Some(sp) = short_preamble {
                let is_mine = optimizer
                    .short_preamble_producer
                    .as_ref()
                    .is_some_and(|sb| {
                        majit_ir::descr_identity(&sb.target_token)
                            == majit_ir::descr_identity(&target_token.as_jump_target_descr())
                    });
                // `descr_identity` identifies a descriptor allocation, not
                // Python `is`: same descr allocation means the same clone
                // family. That is sufficient here because TargetToken clones
                // share one Arc, `finalize_short_preamble` mints a fresh
                // LoopTargetDescr Arc per compile, and TargetToken carries no
                // producer; `unroll.py:376-385` performs the corresponding
                // identity discrimination.
                if is_mine && let Some(label_args) = current_label_args {
                    let mut builder = optimizer
                        .short_preamble_producer
                        .take()
                        .expect("identity gate requires a short-preamble producer");
                    // shortpreamble.py produce_arg / 311-341 parity:
                    // setup() returns false when an op references an
                    // unresolvable Phase 1 OpRef. Treat this exactly
                    // like RPython's "produce_arg returned None →
                    // add_op_to_short returned None" path: drop the
                    // peeled trace and let the unroll caller raise
                    // InvalidLoop, falling back to jump_to_preamble.
                    if !builder.setup(&sp, label_args, ctx) {
                        optimizer.short_preamble_producer = Some(builder);
                        // Drop the peeled trace and let the caller fall back
                        // to jump_to_preamble. Recorded as a deferred
                        // InvalidLoop signal (checked right after this
                        // returns) so no unwinding is needed.
                        ctx.signal_invalid_loop("short preamble has unresolvable Phase 1 args");
                        return None;
                    }
                    ctx.activate_short_preamble_producer(builder);
                    let publication = ActiveShortPreambleProducerPublication::new(optimizer, ctx);
                    extra = Self::inline_short_preamble(
                        &short_jump_args,
                        &target_boxes,
                        &sp,
                        optimizer,
                        ctx,
                    );
                    // Return the builder to its optimizer storage before
                    // anything reads it. Dropping `publication` below re-points
                    // the walker there, and `build_short_preamble_struct`
                    // must run with the builder rooted where the walker looks.
                    if let Some(builder) = ctx.take_active_short_preamble_producer() {
                        optimizer.short_preamble_producer = Some(builder);
                    }
                    drop(publication);
                    // history.py/268/314 — `Const{Int,Float,Ptr}.value`
                    // rides inline on the OpRef. Production no longer seeds
                    // `ctx.const_pool` (`merge_backend_constants_from_ctx`
                    // asserts the pool is empty at export), so the
                    // cross-compile `loop_constants` snapshot is no longer
                    // built: short-preamble ops embed the Const value
                    // directly in `op.args`, mirroring RPython's
                    // `shortpreamble.py` which has no parallel side table.
                    //
                    // Replay can abort mid-way and leave the builder partial,
                    // so publishing it would persist a short preamble later
                    // bridges and retraces consume.
                    if !ctx.has_pending_invalid_loop()
                        && let Some(builder) = optimizer.short_preamble_producer.as_ref()
                    {
                        target_token.attrs().short_preamble =
                            Some(builder.build_short_preamble_struct());
                        // `history.TargetToken.short_preamble` is a traced GC
                        // field upstream, so this replacement invokes the
                        // normal MiniMark write barrier.
                        target_token.mark_minor_scan_pending();
                    }
                } else {
                    extra = Self::inline_short_preamble(
                        &short_jump_args,
                        &target_boxes,
                        &sp,
                        optimizer,
                        ctx,
                    );
                    if crate::debug::have_debug_prints() {
                        crate::debug::log_one(
                            "jit-tracing",
                            &format!("jte-isp done, extra_len={}", extra.len()),
                        );
                    }
                }
                // The assembled LABEL's extra tail is `used_boxes` with
                // constants removed. An extended builder records a short
                // jump arg that tail never carried, and the bridge JUMP
                // is then one arg longer than the stored LABEL.
                let prefix = extra.len().min(sp.used_boxes.len());
                let mut kept = Vec::with_capacity(prefix);
                for i in 0..prefix {
                    if short_extra_is_carried(sp.used_boxes[i]) {
                        kept.push(extra[i].clone());
                    }
                }
                extra = kept;
            }

            // A short-preamble replay that hit an unresolvable Phase 1 arg or
            // a structurally incompatible mapping recorded a deferred
            // InvalidLoop signal above; abandon this jump attempt so the caller
            // falls back to jump_to_preamble.
            if ctx.has_pending_invalid_loop() {
                return None;
            }

            // unroll.py `_jump_to_existing_trace`: emit JUMP to target
            let mut jump_args = target_boxes;
            jump_args.extend(extra);
            // The target's LABEL carries one arg per recorded recipe beyond
            // `target_boxes + extra`; rebuild each from the frame this close
            // already carries (LABEL arg 0 and JUMP arg 0 are the standard
            // virtualizable) so the JUMP matches the LABEL the backend
            // regalloc asserts against. Without this the close lands short and
            // `compile_bridge` gives the bridge up.
            let recipes = target_token.attrs().vable_label_arg_recipes.clone();
            if !recipes.is_empty()
                && let Some(frame_operand) = jump_args.first().cloned()
            {
                for (opcode, descr) in recipes {
                    let tp = opcode.result_type();
                    let mut load = Op::new(opcode, std::slice::from_ref(&frame_operand));
                    load.setdescr(descr);
                    load.pos().set(ctx.reserve_pos_typed(tp));
                    let load_rc = OpRc::new(load);
                    if let Err(e) = optimizer.send_extra_operation(&load_rc, ctx) {
                        ctx.signal_invalid_loop(e.0);
                        return None;
                    }
                    jump_args.push(ctx.resolve_operand_operand(&Operand::from_bound_op(&load_rc)));
                }
            }
            let mut jump = Op::new(OpCode::Jump, &jump_args);
            jump.setdescr(target_token.as_jump_target_descr());
            // unroll.py:357 lets send_extra_operation raise InvalidLoop. This
            // function returns Option (flag convention); propagate consumed
            // the flag into the Err, so re-defer it for the caller's
            // take_invalid_loop barrier (the None return below then reads as an
            // aborted jump rather than a successful one).
            if let Err(e) = optimizer.send_extra_operation(&OpRc::new(jump), ctx) {
                ctx.signal_invalid_loop(e.0);
            }
            return None; // successfully jumped (or aborted via deferred InvalidLoop)
        }

        Some(virtual_state)
    }

    /// unroll.py: inline_short_preamble — replay short preamble ops
    /// to re-populate the optimizer's cache when entering from a bridge.
    ///
    /// Maps short preamble input args to the jump args, then emits
    /// each short preamble op with remapped arguments.
    pub fn inline_short_preamble(
        jump_args: &[Operand],
        args_no_virtuals: &[Operand],
        short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> Vec<Operand> {
        // history.py/268/314 — `Const{Int,Float,Ptr}.value` is inline on
        // the OpRef. All production short-preamble capture sites early-return
        // on Const OpRefs (`shortpreamble.rs`), so
        // `short_preamble.constants` is empty along every production export
        // and the bridge has no const-pool entries to replay through
        // `ctx.const_pool`. The export-side invariant at
        // `optimizer::merge_backend_constants_from_ctx` asserts the same
        // pool-empty contract; mirror it at the producer entry so any
        // re-introduction of const-pool seeding fails loudly here
        // rather than silently leaking into a backend that no longer
        // consumes `ctx.const_pool`.
        debug_assert!(
            short_preamble.constants.is_empty(),
            "inline_short_preamble: short_preamble.constants must be empty in production — \
             history.py:227/268/314 inline-Const is the single source of truth; \
             a non-empty entry indicates a stale legacy producer that would never reach \
             the backend (nothing exports ctx.const_pool)"
        );

        let mut mapping: crate::FxIndexMap<OpRef, Operand> = crate::FxIndexMap::default();

        // unroll.py `assert len(short_inputargs) == len(jump_args)` —
        // the mapping below is positional, so a length mismatch misaligns
        // every seeded pair (a recurring red maps to the wrong slot and the
        // back edge carries the loop-entry value as if it were invariant).
        // RPython guarantees equality by construction; reaching here with a
        // mismatch means the exported short inputargs cover a different
        // label shape than the target's make_inputargs output. Fall back to
        // jump_to_preamble via InvalidLoop instead of seeding a misaligned
        // mapping.
        if short_preamble.inputargs.len() != jump_args.len() {
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit] inline_short_preamble: short_inputargs len {} != jump_args len {} — InvalidLoop",
                    short_preamble.inputargs.len(),
                    jump_args.len()
                );
            }
            ctx.signal_invalid_loop(
                "inline_short_preamble: short_inputargs/jump_args arity mismatch",
            );
            return Vec::new();
        }
        for (i, short_inputarg) in short_preamble.inputargs.iter().enumerate() {
            let short_inputarg = *short_inputarg;
            if let Some(jump_arg) = jump_args.get(i) {
                mapping.insert(short_inputarg, jump_arg.clone());
                // RPython: jump_arg Box inherits info via identity.
                // In majit, propagate PtrInfo from short_inputarg (which
                // has info from Phase 1 export) to the resolved jump_arg.
                // shortpreamble.py __init__ parity: propagate PtrInfo from
                // Phase 1 export to jump_args so guards are redundant.
                let resolved_has_info = ctx
                    .resolve_operand_operand_opt(jump_arg)
                    .as_ref()
                    .is_some_and(|b| ctx.has_ptr_info(b));
                if !resolved_has_info {
                    let jump_box = ctx.resolve_operand_operand_opt(jump_arg);
                    let short_box = ctx.get_box_replacement_operand_opt(short_inputarg);
                    let info = jump_box
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .or_else(|| short_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)))
                        .or_else(|| {
                            short_preamble
                                .inputarg_infos
                                .get(i)
                                .and_then(|opt| opt.clone())
                        });
                    if let Some(info) = info {
                        ctx.ensure_ptr_info_preserve_forwarding(jump_arg.to_opref(), info);
                    }
                }
            }
        }

        // unroll.py inline_short_preamble: the only seed is
        // `mapping[short_inputargs[i]] = jump_args[i]`.

        let mut replay_index = 0;

        fn current_short_len(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
        ) -> usize {
            ctx.active_short_preamble_producer
                .as_ref()
                .map(|builder| builder.short_ops_len())
                .unwrap_or_else(|| short_preamble.ops.len())
        }

        fn current_short_op(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
            index: usize,
        ) -> Option<Op> {
            if let Some(builder) = ctx.active_short_preamble_producer.as_ref() {
                builder.short_op(index).cloned()
            } else {
                short_preamble.ops.get(index).map(|entry| entry.op.clone())
            }
        }

        fn current_short_jump_args(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
        ) -> Vec<OpRef> {
            ctx.active_short_preamble_producer
                .as_ref()
                .map(|builder| builder.jump_args().to_vec())
                .unwrap_or_else(|| short_preamble.jump_args.clone())
        }

        // unroll.py:398-427: fix-point loop, runs only once in almost all cases.
        // RPython uses `while 1:` until convergence, but in practice it should
        // converge in very few iterations. Add a safety cap to prevent hangs.
        let mut fixpoint_iter = 0u32;
        loop {
            fixpoint_iter += 1;
            if fixpoint_iter > 20 {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit][inline_short_preamble] fixpoint loop exceeded 20 iterations, breaking"
                    );
                }
                break;
            }
            // unroll.py:402: while i < len(short) - 1
            // Use LIVE length — newly added ops (from use_box during
            // send_extra_operation) are replayed in the same iteration.
            while replay_index < current_short_len(short_preamble, ctx) {
                let Some(sp_op) = current_short_op(short_preamble, ctx, replay_index) else {
                    break;
                };
                let mut new_op = sp_op.clone();
                // unroll.py: _map_args(mapping, sop.getarglist())
                // Const passes through unchanged, non-Const must be in mapping.
                for i in 0..new_op.num_args() {
                    let arg = new_op.arg(i);
                    // unroll.py:367: isinstance(box, Const) — true Const objects
                    // only (ConstInt/ConstPtr/ConstFloat). make_constant'd values
                    // are NOT Const objects — they are regular boxes with forwarded
                    // set to a Const, so they must go through the mapping.
                    if arg.is_constant()
                        || short_preamble
                            .constants
                            .iter()
                            .any(|(k, _)| *k == arg.to_opref().raw())
                    {
                        continue;
                    }
                    // unroll.py: _map_args — non-Const must be in mapping.
                    // RPython: mapping is complete (seeded from short_inputargs →
                    // jump_args, extended by mapping[sop] = op). Missing keys
                    // indicate a structural mismatch (e.g., cross-loop bridge
                    // with incompatible short preamble). Raise InvalidLoop.
                    match mapping.get(&arg.to_opref()) {
                        // unroll.py `_map_args`: mapping values are the replayed op
                        // objects.
                        Some(mapped) => new_op.setarg(i, mapped.clone()),
                        None => {
                            // RPython: _map_args raises KeyError for unmapped
                            // args. This is equivalent to InvalidLoop — the
                            // short preamble is structurally incompatible.
                            // Recorded as a deferred InvalidLoop signal that
                            // jump_to_existing_trace's caller observes, falling
                            // back to jump_to_preamble (unroll.py,
                            // 209-211).
                            if crate::optimizeopt::majit_log_enabled() {
                                eprintln!(
                                    "[jit] inline_short_preamble: unmapped arg {:?} in {:?} — InvalidLoop",
                                    arg, new_op.opcode
                                );
                            }
                            ctx.signal_invalid_loop(
                                "inline_short_preamble: unmapped arg in short preamble",
                            );
                            return Vec::new();
                        }
                    }
                }
                // unroll.py:405-414: unified guard/non-guard handling.
                // RPython: both guards and non-guards follow the same path:
                //   copy_and_change → mapping[sop] = op → send_extra_operation(op)
                if new_op.opcode.is_guard() {
                    // unroll.py:406-409: copy_and_change with ResumeAtPositionDescr.
                    // The fresh ResumeAtPositionDescr has an empty RdPayload, so
                    // the new guard reads None for every rd_* until
                    // store_final_boxes_in_guard repopulates them from the live
                    // snapshot.
                    new_op.setdescr(crate::optimizeopt::make_resume_at_position_descr());
                    new_op.clearfailargs();
                    new_op.clear_fail_arg_types();
                    // unroll.py:409: op.rd_resume_position =
                    // patchguardop.rd_resume_position — an unconditional
                    // dereference, so upstream has no branch for an absent
                    // patchguardop.
                    //
                    // `new_op` is a clone of the short-preamble entry, which is
                    // itself a whole-op clone of a guard from the PREAMBLE
                    // trace, so it arrives carrying that trace's
                    // rd_resume_position. Skipping the stamp does not leave the
                    // guard unstamped — it leaves it stamped with a coordinate
                    // from another trace's numbering, which
                    // `store_final_boxes_in_guard` then resolves against this
                    // one. Decline the inlining instead, exactly as the unmapped
                    // -arg arm above does: the caller falls back to
                    // jump_to_preamble and the trace still compiles.
                    let Some(patch_pos) = ctx.patchguardop.as_ref().map(|p| p.rd_resume_position())
                    else {
                        if crate::optimizeopt::majit_log_enabled() {
                            eprintln!(
                                "[jit] inline_short_preamble: no patchguardop for replayed \
                                 {:?} — InvalidLoop",
                                new_op.opcode
                            );
                        }
                        ctx.signal_invalid_loop(
                            "inline_short_preamble: replayed guard with no patchguardop",
                        );
                        return Vec::new();
                    };
                    new_op.set_rd_resume_position(patch_pos);
                    // history.py/268/314 — Const values ride inline
                    // on the OpRef (ConstInt/ConstFloat/
                    // ConstPtr). No pool replay needed.
                    debug_assert!(short_preamble.constants.is_empty());
                } else if let Some(fail_args) = new_op.fail_args_mut() {
                    for arg in fail_args.iter_mut() {
                        if let Some(mapped) = mapping.get(&arg.to_opref()) {
                            // Measured dead (PYRE_REMAP_PROBE 2026-06-11: 0
                            // fires across check.py corpus + lib tests) —
                            // only guards carry fail_args and guards take the
                            // clearfailargs arm above, matching unroll.py:
                            // 405-414 which has no fail_args handling on the
                            // non-guard path. Rewrite kept as a release
                            // safety net.
                            debug_assert!(
                                false,
                                "non-guard short-preamble op carried fail_args: {:?}",
                                new_op.opcode
                            );
                            *arg = mapped.clone();
                        }
                    }
                }
                let new_ref = ctx.alloc_op_position_typed(new_op.result_type());
                new_op.pos().set(new_ref);
                // unroll.py `inline_short_preamble`: mapping[sop] = op, then
                // send_extra_operation(op). The mapping write happens before
                // send_extra_operation.
                let new_rc = OpRc::new(new_op);
                mapping.insert(sp_op.pos().get(), Operand::from_bound_op(&new_rc));
                replay_index += 1;
                // unroll.py:414 lets send_extra_operation raise InvalidLoop.
                // This function returns Vec (flag convention, as the arity /
                // unmapped-arg signals above); propagate consumed the flag into
                // the Err, so re-defer it for the caller's take_invalid_loop
                // barrier and bail to jump_to_preamble.
                if let Err(e) = optimizer.send_extra_operation(&new_rc, ctx) {
                    ctx.signal_invalid_loop(e.0);
                    return Vec::new();
                }
            }

            // unroll.py:417-434: force all except virtuals.
            loop {
                let short_jump_args = current_short_jump_args(short_preamble, ctx);
                let num_short_jump_args = short_jump_args.len();
                // unroll.py `_map_args(mapping, args)`: Const passes
                // through unchanged, non-Const requires mapping.
                // An unmapped non-Const arg is the same KeyError upstream's
                // `_map_args` raises, and the other two sites in this function
                // already answer it by declining to InvalidLoop rather than
                // substituting the preamble's box. Answer it the same way here
                // instead of aborting the process.
                let mut mapped_jump_args: Vec<Operand> = Vec::with_capacity(num_short_jump_args);
                for jump_arg in short_jump_args.iter() {
                    let mapped = if jump_arg.is_constant() {
                        Operand::from_opref(*jump_arg)
                    } else {
                        let Some(mapped) = mapping.get(jump_arg) else {
                            if crate::optimizeopt::majit_log_enabled() {
                                eprintln!(
                                    "[jit] inline_short_preamble: unmapped short jump arg \
                                     {jump_arg:?} in the force fix-point — InvalidLoop"
                                );
                            }
                            ctx.signal_invalid_loop(
                                "inline_short_preamble: unmapped jump arg in force fix-point",
                            );
                            return Vec::new();
                        };
                        mapped.clone()
                    };
                    mapped_jump_args.push(mapped);
                }
                // unroll.py `inline_short_preamble`: force all except virtuals.
                for arg_box in args_no_virtuals.iter() {
                    if arg_box.is_none() {
                        continue;
                    }
                    let _ = optimizer.force_box(arg_box, ctx);
                }
                for arg_box in mapped_jump_args.iter() {
                    let _ = optimizer.force_box(arg_box, ctx);
                }
                // unroll.py:425-434: if short_jump_args did not grow we are
                // done. If force_box grew it via add_preamble_op, only re-force
                // when every new entry is a Const or already in `mapping`;
                // otherwise break and let the outer replay loop register the
                // producer first, since the top-of-loop _map_args above would
                // otherwise read an unmapped arg.
                let grown = current_short_jump_args(short_preamble, ctx);
                if grown.len() == num_short_jump_args {
                    break;
                }
                let all_new_mapped = grown[num_short_jump_args..]
                    .iter()
                    .all(|arg| arg.is_constant() || mapping.contains_key(arg));
                if !all_new_mapped {
                    break;
                }
            }
            // unroll.py:424
            // flush may raise InvalidLoop via send_extra_operation; this
            // function returns Vec (flag convention), so re-defer the
            // consumed flag for the caller's take_invalid_loop barrier and
            // bail to jump_to_preamble.
            if let Err(e) = optimizer.flush(ctx) {
                ctx.signal_invalid_loop(e.0);
                return Vec::new();
            }
            // unroll.py:426: done unless "short" has grown again
            if replay_index == current_short_len(short_preamble, ctx) {
                break;
            }
        }

        // unroll.py:438-439 `return [get_box_replacement(box) for box in
        // self._map_args(mapping, short_jump_args)]`.
        //
        // `_map_args` (unroll.py) is STRICT for non-Const boxes:
        // `box = mapping[box]` raises KeyError when the short op that produces
        // this jump arg was never replayed into the peeled body. Substituting
        // the unmapped arg instead hands the JUMP the PREAMBLE's box — the
        // loop-entry value — so the back edge re-supplies the entry value every
        // iteration and a field the body mutates reads as loop-invariant. Treat
        // it as the InvalidLoop that KeyError amounts to and fall back to
        // jump_to_preamble; the loop still compiles, it just declines to inline
        // this short preamble.
        let final_jump_args = current_short_jump_args(short_preamble, ctx);
        let mut mapped_args = Vec::with_capacity(final_jump_args.len());
        for jump_arg in final_jump_args {
            if jump_arg.is_constant() {
                mapped_args.push(Operand::from_opref(jump_arg));
                continue;
            }
            let Some(mapped) = mapping.get(&jump_arg) else {
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit] inline_short_preamble: unmapped short jump arg {jump_arg:?} \
                         (short op never replayed into the peeled body) — InvalidLoop"
                    );
                }
                ctx.signal_invalid_loop(
                    "inline_short_preamble: unmapped short jump arg at end of inlining",
                );
                return Vec::new();
            };
            mapped_args.push(ctx.resolve_operand_operand(mapped));
        }
        mapped_args
    }

    /// unroll.py import_state — line-by-line port.
    ///
    /// ```python
    /// def import_state(self, targetargs, exported_state):
    ///     assert len(exported_state.next_iteration_args) == len(targetargs)
    ///     for i, target in enumerate(exported_state.next_iteration_args):
    ///         source = targetargs[i]
    ///         assert source is not target
    ///         source.set_forwarded(target)
    ///         info = exported_state.exported_infos.get(target, None)
    ///         if info is not None:
    ///             self.optimizer.setinfo_from_preamble(source, info,
    ///                                             exported_state.exported_infos)
    ///     label_args = exported_state.virtual_state.make_inputargs(
    ///         targetargs, self.optimizer)
    ///     self.short_preamble_producer = ShortPreambleBuilder(
    ///         label_args, exported_state.short_boxes,
    ///         exported_state.short_inputargs, exported_state.exported_infos,
    ///         self.optimizer)
    ///     for produced_op in exported_state.short_boxes:
    ///         produced_op.produce_op(self, exported_state.exported_infos)
    ///     return label_args
    /// ```
    ///
    /// In RPython the short-preamble setup is done by constructing
    /// `ShortPreambleBuilder` with `label_args` and looping over
    /// `exported_state.short_boxes` calling `produce_op`.  In majit the
    /// equivalent is `import_short_preamble_ops` (a serialized form of
    /// the per-PreambleOp `produce_op` body) plus
    /// `initialize_imported_short_preamble_builder_from_exported_ops`
    /// (a serialized form of the `ShortPreambleBuilder` constructor).
    /// Both consume the same `short_args = label_args + virtuals` slot
    /// space that `export_state` used to build `ExportedShortOp` (see
    /// `collect_exported_short_ops`); the slot indices are the
    /// majit-only stand-in for RPython's Box-keyed `produced_short_boxes`
    /// dict.
    pub fn import_state(
        &self,
        targetargs: &[OpRef],
        exported_state: &ExportedState,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        // assert len(exported_state.next_iteration_args) == len(targetargs)
        assert_eq!(
            exported_state.next_iteration_args.len(),
            targetargs.len(),
            "import_state: next_iteration_args mismatch"
        );
        // for i, target in enumerate(exported_state.next_iteration_args):
        for (i, carried) in exported_state.next_iteration_args.iter().enumerate() {
            // `carried` is the literal Phase-1 box (the same Rc used as the
            // exported_infos key); `.to_opref()` is its resolved position, used
            // only for the forwarding plumbing below.
            let target = carried.to_opref();
            // source = targetargs[i]
            let source = targetargs[i];
            // source.set_forwarded(target)
            // `source` is `targetargs[i]`, produced by the caller's
            // cross-slot resolution as either a materialized inputarg or a
            // `reserve_virtual_box`-minted alias — both resolve without
            // minting here.
            let b_source = ctx
                .get_box_replacement_operand_opt(source)
                .expect("import_state source must have a materialized operand slot");
            // `target` is the literal Phase-1 Box carried by ExportedState.
            // Forward to that exact Operand, not to a Phase-2 box recovered
            // from its numeric position: unroll.py:497 writes
            // `source.set_forwarded(target)`, preserving box identity across
            // the phase boundary. Re-materializing by OpRef collapses the
            // source/replay namespaces when their raw positions coincide.
            // The carried Box arrives as a private `Rc` that
            // `find_producer_op` cannot reach, so register it as the host for
            // its position first: otherwise every chain forwarded here ends on
            // a producerless position and guard resume numbering has no
            // producer to bind.
            let b_target = carried.clone();
            // unroll.py:496 checks Box identity. `Operand` equality is Rc
            // pointer identity, while raw OpRef positions may legitimately
            // coincide across this phase boundary.
            debug_assert!(
                b_source != b_target,
                "import_state: source is target (unroll.py:496 `assert source is not target`)"
            );
            ctx.register_carried_host(&b_target);
            ctx.make_equal_to(&b_source, &b_target);
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-optimizer",
                    &format!("import_state_map[{i}]: source={source:?} target={target:?}"),
                );
            }
            // info = exported_state.exported_infos.get(target, None)
            // Look up by the carried Phase-1 box (the same Rc the exporter used
            // as the key) — a ptr_eq hit, with no Phase-2 re-resolution that would
            // mint a fresh (non-ptr_eq) Const/InputArg box.
            if let Some(info) = exported_state.exported_infos.get(carried) {
                //     self.optimizer.setinfo_from_preamble(source, info,
                //                                     exported_state.exported_infos)
                self.setinfo_from_preamble(source, info, &exported_state.exported_infos, ctx);
            }
        }
        // label_args = exported_state.virtual_state.make_inputargs(
        //     targetargs, self.optimizer)
        let label_args = match exported_state
            .virtual_state
            .make_inputargs(targetargs, optimizer, ctx, false)
        {
            Ok(args) => args.iter().map(|b| b.to_opref()).collect(),
            // unroll.py:483 `raise InvalidLoop`: the imported virtual state is
            // incompatible. Recorded as a deferred signal (checked by the
            // caller) so the loop is abandoned without unwinding.
            Err(_) => {
                ctx.signal_invalid_loop("Cannot import state, virtual states don't match");
                return Vec::new();
            }
        };
        if crate::callee_rca_enabled() {
            let next_iteration_args: Vec<_> = exported_state
                .next_iteration_args
                .iter()
                .map(|arg| arg.to_opref())
                .collect();
            eprintln!(
                "[callee-rca][import-state] targetargs={:?} next_iteration_args={:?} \
                 label_args={:?} num_boxes={} entries={}",
                targetargs,
                next_iteration_args,
                label_args,
                exported_state.virtual_state.num_boxes(),
                exported_state.virtual_state.num_entries(),
            );
            for line in callee_rca_virtual_state_summary(&exported_state.virtual_state) {
                eprintln!("[callee-rca][import-vs] {line}");
            }
        }
        // The short-preamble replay part of unroll.py:496-502 is performed
        // by import_short_preamble_state after majit's split-out
        // Optimizer::install_imported_virtuals has completed. RPython's
        // virtual_state.make_inputargs installs virtual PtrInfo before
        // produced_op.produce_op; doing it earlier here replays short ops
        // against incomplete virtual state.
        // return label_args
        label_args
    }

    pub fn import_short_preamble_state(
        &self,
        targetargs: &[OpRef],
        label_args: &[OpRef],
        exported_state: &ExportedState,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) {
        // self.short_preamble_producer = ShortPreambleBuilder(
        //     label_args, exported_state.short_boxes,
        //     exported_state.short_inputargs, exported_state.exported_infos,
        //     self.optimizer)
        //
        // majit's `ShortPreambleBuilder` constructor is the
        // `initialize_imported_short_preamble_builder_from_exported_ops`
        // call below. The majit serialization keys short ops by slot index
        // over the combined `label_args + virtuals` slot space (the same
        // space `collect_exported_short_ops` built against), so we compute
        // the virtuals tail inline before calling the initializer.
        let virtuals: Vec<OpRef> = exported_state
            .virtual_state
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| targetargs.get(i).copied())
            .collect();
        let mut short_args = label_args.to_vec();
        short_args.extend(virtuals);
        // unroll.py:486-489 ShortPreambleBuilder constructor parity.
        //
        // RPython already has all Box identities before this point, so the
        // ShortPreambleBuilder can be constructed before the produce_op loop.
        // Majit precomputes the equivalent Phase-2 result OpRefs first, then
        // shares that map with both the builder constructor and produce_op.
        // history.py:220 Box `.type` is fixed at construction.  RPython
        // assigns a Box of the correct kind from `produced_op.short_op.res`
        // before `produce_op` runs; majit must allocate the result OpRef
        // with the typed allocator so `opref_type` resolution doesn't
        // depend on a downstream type-registration patch.
        let mut result_map: indexmap::IndexMap<OpRef, OpRef, rustc_hash::FxBuildHasher> =
            indexmap::IndexMap::with_hasher(rustc_hash::FxBuildHasher);
        for (source, produced) in &exported_state.short_boxes {
            let result_type = produced.preamble_op.result_type();
            let result = match produced.kind {
                crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
                | crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant => {
                    Some(ctx.alloc_op_position_typed(result_type))
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::Heap => {
                    match produced.preamble_op.opcode {
                        OpCode::GetfieldGcI
                        | OpCode::GetfieldGcR
                        | OpCode::GetfieldGcF
                        | OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => Some(ctx.alloc_op_position_typed(result_type)),
                        _ => None,
                    }
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
                | crate::optimizeopt::shortpreamble::PreambleOpKind::Guard => None,
            };
            if let Some(result) = result {
                // shortpreamble.py:327: `self.res` is a Box object that
                // exists from import time. The freshly allocated
                // body-visible result slots (the `alloc_op_position_typed`
                // arms above) have no producer yet; mint their canonical
                // `SameAs*` stand-in here so a later `get_box_replacement`
                // resolves them instead of fabricating a position-only box.
                if ctx.get_box_replacement_operand_opt(result).is_none() {
                    ctx.mint_box_at(result);
                }
                result_map.insert(*source, result);
            }
        }
        let from_short_boxes = ctx.initialize_imported_short_preamble_builder_from_short_boxes(
            &short_args,
            &exported_state.short_inputargs,
            &exported_state.short_boxes,
            &result_map,
            &exported_state.exported_infos,
        );
        debug_assert!(
            from_short_boxes,
            "initialize_imported_short_preamble_builder_from_short_boxes returned false: \
             a loop-invariant function pointer was not a constant int"
        );
        // unroll.py `import_state`:
        //
        //   for produced_op in exported_state.short_boxes:
        //       produced_op.produce_op(self, exported_state.exported_infos)
        for (_, produced) in &exported_state.short_boxes {
            let produced_result =
                produced.produce_op(ctx, optimizer, &exported_state.exported_infos, &result_map);
            debug_assert!(
                produced_result.is_some()
                    || matches!(
                        produced.kind,
                        crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
                            | crate::optimizeopt::shortpreamble::PreambleOpKind::Guard
                    )
                    || !result_map.contains_key(&produced.preamble_op.pos().get()),
                "ProducedShortOp::produce_op failed for source {:?} kind {:?}",
                produced.preamble_op.pos().get(),
                produced.kind
            );
        }

        let const_short_boxes =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &exported_state.const_short_boxes,
            );
        // shortpreamble.py:417-421: Const results use optimizer.getinfo(op),
        // and that info is forwarded on the replay operation itself before
        // HeapOp.produce_op installs the PreambleOp in the heap cache.
        for (_, produced) in &const_short_boxes {
            let info = self.collect_exported_info(produced.res.to_opref(), ctx, None);
            if let Some(info) = &info {
                ctx.set_preamble_forwarded_info(produced.preamble_op.pos().get(), info);
            }
        }
        // unroll.py:512-513 covers both groups upstream, because
        // `create_short_boxes` returns one list. pyre keeps the const heap short
        // boxes on a separate channel so they cannot reach `used_boxes`
        // (unroll.py:33-36 never registers a const PreambleOp in
        // `potential_extra_ops`), so they are replayed here explicitly.
        for (_, produced) in &const_short_boxes {
            let _ =
                produced.produce_op(ctx, optimizer, &exported_state.exported_infos, &result_map);
        }
    }

    /// unroll.py `_expand_info` + `unroll.py`
    /// `ExportedState.exported_infos` entry producer. Returns the single
    /// `OpInfo` variant RPython's dict would hold for this box, or `None`
    /// for the RPython `if info:` falsy fall-through at unroll.py — the
    /// caller must treat `None` as "no entry in the dict" so downstream
    /// `setinfo_from_preamble_list` takes the `item.set_forwarded(None)`
    /// branch at unroll.py:49.
    ///
    ///   - Ref-typed live box with PtrInfo → `Some(OpInfo::Ptr(...))`.
    ///   - Constant OpRef (value-carrying) → `Some(OpInfo::Ptr(PtrInfo::Constant))`
    ///     for Refs, `Some(OpInfo::FloatConstInfo(FloatConstInfo))` for floats,
    ///     `Some(OpInfo::IntBound(IntBound::from_constant(v)))` for ints
    ///     (mirroring RPython's `ConstPtrInfo` / `FloatConstInfo` /
    ///     `IntBound` dispatch).
    ///   - Int-typed box with a preamble-exported `IntBound` →
    ///     `Some(OpInfo::IntBound(...))`.
    ///   - No info → `None`. `OpInfo::Unknown` must not appear in
    ///     `ExportedState.exported_infos`.
    fn collect_exported_info(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<
            &crate::FxIndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
    ) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::{FloatConstInfo, OpInfo, PtrInfo};
        let resolved = ctx.get_replacement_opref(opref);
        // unroll.py `_expand_info` calls `self.optimizer.getinfo(arg)`
        // which itself runs `get_box_replacement` first, so a non-constant
        // OpRef forwarded to a Const surfaces the corresponding constant
        // info class (ConstPtrInfo / FloatConstInfo / IntBound from_constant).
        let synthesize_const_info = |value: Value| -> Option<OpInfo> {
            match value {
                // ConstPtrInfo parity: RPython stores Ref constants as
                // ConstPtrInfo (a `PtrInfo` subclass). `setinfo_from_preamble`
                // at unroll.py:65-68 dispatches through `is_constant()`.
                Value::Ref(gcref) => Some(OpInfo::ptr(PtrInfo::Constant(gcref))),
                // FloatConstInfo parity: unroll.py:97-98 handles
                // `isinstance(preamble_info, info.FloatConstInfo)` with
                // `op.set_forwarded(preamble_info._const)`.
                Value::Float(f) => Some(OpInfo::FloatConstInfo(FloatConstInfo::new(f))),
                // Int constants: RPython uses IntBound with lower==upper.
                Value::Int(v) => Some(OpInfo::int_bound(
                    crate::optimizeopt::intutils::IntBound::from_constant(v),
                )),
                Value::Void => None,
            }
        };
        if resolved.is_constant()
            && let Some(value) = ctx
                .get_box_replacement_operand_opt(resolved)
                .and_then(|cb| cb.const_value())
        {
            return synthesize_const_info(value);
        }
        // make_constant mirrors optimizer.py as `Forwarded::Const(constval)`.
        // The walker has advanced to the constbox terminal — surface RPython's
        // ConstPtrInfo / FloatConstInfo / IntBound dispatch via const_value().
        let resolved_box = ctx.get_box_replacement_operand_opt(opref);
        if let Some(b) = resolved_box.as_ref()
            && b.is_constant()
            && let Some(value) = b.const_value()
        {
            return synthesize_const_info(value);
        }
        // unroll.py _expand_info uses self.optimizer.getinfo(arg) which
        // dispatches by op.type ('r' → getptrinfo, 'i' → getintbound). The Rust
        // port stores int bounds in a separate table populated earlier by
        // `OptIntBounds::export_arg_int_bounds`, which already filters by
        // `opref_type(resolved) == Some(Int)`. We rely on that filter so the
        // lookup here cannot pull a bound for a ref/float box.
        if let Some(handle) = resolved_box.as_ref().and_then(|b| b.ptr_info_handle()) {
            // RPython object identity: re-export the same Rc handle so
            // downstream `setinfo_from_preamble` sees the live cell, not
            // a snapshot. Matches PyPy `_forwarded` reference passing.
            return Some(OpInfo::Ptr(handle));
        }
        // unroll.py `_expand_info` calls `self.optimizer.getinfo(arg)`
        // for EVERY exported value with no jump-arg restriction, so an int-typed
        // value's bound is read straight off its `_forwarded` slot
        // (`getintbound`). Read the live bound from the still-alive Phase-1
        // optimizer context; this recovers the `[0, mask]` bound of a masked
        // short-preamble / loop-invariant pure result that is not a jump arg and
        // therefore never entered the `exported_int_bounds` side table. Mirror
        // `export_arg_int_bounds` (intbounds.rs): skip Const/non-Int values and
        // unbounded bounds.
        if let Some(box_op) = resolved_box.as_ref()
            && !resolved.is_constant()
            && matches!(ctx.opref_type(resolved), Some(majit_ir::Type::Int))
            && let Some(bound) = ctx.peek_intbound_box(box_op)
            && !bound.is_unbounded()
        {
            return Some(OpInfo::int_bound(bound));
        }
        // Fallback: read from `exported_int_bounds`
        // side table.  RPython's `IntBound` flows through
        // `OptInfo.IntBound` on the Box itself
        // (`optimizeopt/info.py:580 IntBoundInfo`), so successive peeling
        // iterations see the bound without an explicit hand-off.
        // pyre's flat-OpRef `OptContext` is rebuilt per round so the
        // preamble's bound must be exported by
        // `intbounds.rs::export_arg_int_bounds` and re-imported here.
        // Convergence: extend `setinfo_from_preamble_item` (`mod.rs`)
        // to attach `OpInfo::IntBound` alongside `OpInfo::Ptr` so this
        // branch becomes redundant and the side-table parameter
        // disappears.
        if let Some(bound) = exported_int_bounds.and_then(|bounds| {
            // Same Phase-1 ctx as the export producer, so the canonical box for
            // `opref` is the memoized `Rc` the bound was keyed under (ptr_eq).
            ctx.get_box_replacement_operand_opt(opref)
                .and_then(|o| bounds.get(&o).cloned())
        }) {
            return Some(OpInfo::int_bound(bound));
        }
        None
    }

    /// unroll.py `setinfo_from_preamble(op, preamble_info, exported_infos)`.
    ///
    /// Thin forwarding wrapper onto `OptContext::setinfo_from_preamble_item`
    /// (mod.rs), which is the shared RPython-literal dispatcher used both by
    /// this top-level import and by the recursive virtual-field walker at
    /// `setinfo_from_preamble_list`. Keeping a single implementation prevents
    /// the two call sites from drifting: any shortcut in one must also apply
    /// to the other or the early-return semantics at unroll.py:54-58 are
    /// bypassed for part of the import path.
    fn setinfo_from_preamble(
        &self,
        opref: OpRef,
        info: &crate::optimizeopt::info::OpInfo,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        ctx: &mut OptContext,
    ) {
        let op = ctx.materialize_operand_at(opref);
        ctx.setinfo_from_preamble_item(&op, info, exported_infos);
    }
}

/// unroll.py: export_state — module-level entry point.
pub(crate) fn export_state(
    jump_args: &[OpRef],
    renamed_inputargs: &[OpRef],
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
    exported_int_bounds: Option<
        &crate::FxIndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
    >,
) -> ExportedState {
    OptUnroll::new().export_state_with_bounds(
        jump_args,
        renamed_inputargs,
        optimizer,
        ctx,
        exported_int_bounds,
    )
}

/// unroll.py import_state — module-level entry point.
#[allow(dead_code)]
pub(crate) fn import_state(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    OptUnroll::new().import_state(targetargs, exported_state, optimizer, ctx)
}

#[allow(dead_code)]
pub(crate) fn import_short_preamble_state(
    targetargs: &[OpRef],
    label_args: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) {
    OptUnroll::new().import_short_preamble_state(
        targetargs,
        label_args,
        exported_state,
        optimizer,
        ctx,
    )
}

/// RPython unroll.py `import_state(targetargs, exported_state)`
/// canonical end-to-end orchestrator.  Bundles the three majit-side
/// sub-steps that mirror the RPython single-function flow:
///
/// 1. `OptUnroll::import_state` — `unroll.py` forwarding +
///    `virtual_state.make_inputargs` to compute label_args.
/// 2. `Optimizer::install_imported_virtuals` — majit-only counterpart
///    to RPython's inline virtualizable PtrInfo installation that
///    happens during `make_inputargs` itself.  Split out for borrow
///    reasons (`Optimizer` access to `imported_virtuals` / `imported_loop_state`).
/// 3. `OptUnroll::import_short_preamble_state` — `unroll.py`
///    `ShortPreambleBuilder` construction + `produce_op` loop.
///
/// Production callers should use this end-to-end form.  Test callers
/// that need to inspect intermediate state can still invoke the
/// individual functions above.
pub(crate) fn import_state_full(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    let label_args = OptUnroll::new().import_state(targetargs, exported_state, optimizer, ctx);
    // import_state records a deferred InvalidLoop signal on `ctx` when the
    // virtual states don't match; stop here (the caller observes the signal and
    // abandons the loop) rather than replaying short preamble ops against an
    // incomplete import.
    if ctx.has_pending_invalid_loop() {
        return label_args;
    }
    optimizer.imported_label_args = Some(label_args.clone());
    if !optimizer.imported_virtuals.is_empty() {
        optimizer.install_imported_virtuals(ctx);
    }
    OptUnroll::new().import_short_preamble_state(
        targetargs,
        &label_args,
        exported_state,
        optimizer,
        ctx,
    );
    label_args
}

// unroll.py: pick_virtual_state(my_vs, label_vs, target_tokens)
// Given the current virtual state and available target tokens, find a
// compatible target to jump to. RPython: import_state + _generate_virtual.
// ── RPython-parity helper functions for 2-phase preamble peeling ──

/// Derive ImportedVirtual entries from ExportedState's VirtualState.
/// Virtual structure is obtained from the VirtualState snapshot.
fn build_imported_virtuals_from_state(
    state: &ExportedState,
) -> Vec<crate::optimizeopt::optimizer::ImportedVirtual> {
    use crate::optimizeopt::virtualstate::{VirtualStateInfo, VirtualStateInfoNode};

    /// virtualstate.py AbstractVirtualStructStateInfo.generate_guards
    /// parity: walk `fielddescrs` in parent-local slot order, looking up
    /// the matching field state via descr.get_index() (= field_idx in pyre).
    fn ordered_fields(
        owner: &majit_ir::DescrRef,
        field_descrs: &[majit_ir::DescrRef],
        fields: &[(u32, std::rc::Rc<VirtualStateInfoNode>)],
    ) -> Vec<(majit_ir::DescrRef, VirtualStateInfo)> {
        field_descrs
            .iter()
            .filter_map(|field_descr| {
                let field_idx = field_descr.as_field_descr().map(|fd| {
                    crate::optimizeopt::virtualize::parent_list_slot_in(Some(owner), fd)
                })?;
                fields
                    .iter()
                    .find(|(idx, _)| *idx == field_idx)
                    .map(|(_, field_value)| (field_descr.clone(), field_value.info.clone()))
            })
            .collect()
    }

    let mut result = Vec::new();
    for (idx, info) in state.virtual_state.state.iter().enumerate() {
        match &info.info {
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                fields,
                field_descrs,
                ..
            } => {
                result.push(crate::optimizeopt::optimizer::ImportedVirtual {
                    inputarg_index: idx,
                    size_descr: descr.clone(),
                    kind: crate::optimizeopt::optimizer::ImportedVirtualKind::Instance {
                        known_class: *known_class,
                    },
                    fields: ordered_fields(descr, field_descrs, fields),
                    head_load_descr_index: None,
                });
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                result.push(crate::optimizeopt::optimizer::ImportedVirtual {
                    inputarg_index: idx,
                    size_descr: descr.clone(),
                    kind: crate::optimizeopt::optimizer::ImportedVirtualKind::Struct,
                    fields: ordered_fields(descr, field_descrs, fields),
                    head_load_descr_index: None,
                });
            }
            _ => {}
        }
    }
    result
}

/// compile.py:310-338: [preamble_no_jump] + Label(label_args) + [body_with_jump]
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "this parity-test helper exposes every segment and mapping of RPython's peeled-trace assembly independently so failures identify the exact upstream phase input"
)]
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    body_num_inputs: usize,
    jump_was_redirected: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    constants: &majit_ir::ConstMap<majit_ir::Value>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
) -> Vec<majit_ir::OpRc> {
    let mut ctx = assemble_test_context(p1_ops, p2_ops, body_num_inputs);
    let p1_ops_rc: Vec<majit_ir::OpRc> = p1_ops.iter().map(|op| OpRc::new(op.clone())).collect();
    let p2_ops_rc: Vec<majit_ir::OpRc> = p2_ops.iter().map(|op| OpRc::new(op.clone())).collect();
    assemble_peeled_trace_with_jump_args(
        &p1_ops_rc,
        &p2_ops_rc,
        label_args,
        start_label_args,
        extra_label_args,
        extra_label_args,
        body_num_inputs,
        0, // inputarg_base — tests/simple cases use shared namespace
        jump_was_redirected,
        imported_short_aliases,
        constants,
        start_label_descr,
        loop_label_descr,
        &[], // no p1_end_args for simple assembly
        &mut Vec::new(),
        &mut ctx,
    )
}

#[cfg(test)]
fn assemble_test_context(p1_ops: &[Op], p2_ops: &[Op], body_num_inputs: usize) -> OptContext {
    // Test helper — caller provides only the body inputarg count, so seed
    // every slot as Ref (matches the common loop-body shape; fixtures that
    // need typed inputargs construct ctx directly).
    //
    // Type lookup now resolves through `op.pos.ty()` (variant
    // tag, post-Slice-P5/P6) at priority 0 of `opref_type` and
    // `op.type_` at priority 2 — no longer through a `value_types`
    // side table.
    let types = vec![Type::Ref; body_num_inputs];
    OptContext::with_inputarg_types(p1_ops.len() + p2_ops.len(), &types)
}

/// shortpreamble.py:436-439 alias-side `extra_same_as` emission.
///
/// Compound alternates (`invented_name=true`) record their SameAs source in
/// `imported_short_aliases`. Emit one `SameAs(alias.same_as_source)` op at
/// `pos=alias.result` per entry so the body's reference to the alias result
/// has a defining op.
fn emit_alias_same_as_for_imports(
    result: &mut Vec<majit_ir::OpRc>,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
) {
    for alias in imported_short_aliases {
        let mut op = Op::new(
            alias.same_as_opcode,
            std::slice::from_ref(&alias.same_as_source),
        );
        op.pos().set(alias.result);
        result.push(OpRc::new(op));
    }
}

/// Phase 2 may forward a preamble-defined Box to a fresh body-visible Box
/// that the LABEL carries. First fall-through only has the preamble source;
/// the flat OpRef model needs an explicit SameAs so that local is written
/// before the join. Resume and JUMP write it themselves.
///
/// `stream_defs` is the values the assembled preamble actually writes
/// (op results already in `result`, plus start-label args). Phase-2
/// inputarg slots are not in that stream; SameAs from one of those
/// makes dynasm/cranelift `loc()` an `InputArgRef` the compiled loop
/// never received.
fn push_fallthrough_same_as(
    fallthrough_aliases: &mut Vec<Op>,
    stream_defs: &crate::FxIndexSet<OpRef>,
    arg: OpRef,
    constants: &majit_ir::ConstMap<majit_ir::Value>,
    ctx: &mut crate::optimizeopt::OptContext,
) {
    if !is_trace_runtime_ref(arg, constants) || stream_defs.contains(&arg) {
        return;
    }
    let Some(source) = stream_defs
        .iter()
        .copied()
        .find(|&source| source != arg && ctx.get_replacement_opref(source) == arg)
    else {
        return;
    };
    let tp = ctx
        .opref_type(arg)
        .or_else(|| ctx.opref_type(source))
        .or_else(|| arg.ty())
        .unwrap_or_else(|| {
            panic!(
                "assemble_peeled_trace_with_jump_args: cannot type \
                 fallthrough SameAs alias arg={:?} source={:?}; \
                 ctx.opref_type(arg), ctx.opref_type(source), and \
                 arg.ty() all returned None",
                arg, source
            )
        });
    if tp == Type::Void {
        return;
    }
    let arg_source = ctx.materialize_operand_at(source);
    let mut same_as = Op::new(OpCode::same_as_for_type(tp), &[arg_source]);
    same_as.pos().set(arg);
    fallthrough_aliases.push(same_as);
}

#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
fn assemble_peeled_trace_with_jump_args(
    p1_ops: &[majit_ir::OpRc],
    p2_ops: &[majit_ir::OpRc],
    label_args: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    extra_jump_args: &[OpRef],
    body_num_inputs: usize,
    inputarg_base: u32,
    jump_was_redirected: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    constants: &majit_ir::ConstMap<majit_ir::Value>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
    _p1_end_args: &[OpRef],
    appended_label_args: &mut Vec<OpRef>,
    ctx: &mut crate::optimizeopt::OptContext,
) -> Vec<majit_ir::OpRc> {
    let mut result =
        Vec::with_capacity(p1_ops.len() + p2_ops.len() + 1 + imported_short_aliases.len());
    let mut filtered_extra_label_args = Vec::new();
    let mut filtered_extra_jump_args = Vec::new();
    // RPython Box parity (compile.py:327, shortpreamble.py:436-439):
    //
    //   loop.operations = ([start_label] + preamble_ops + loop_info.extra_same_as +
    //                      loop_info.extra_before_label + [loop_info.label_op] + loop_ops)
    //
    //   op = preamble_op.op.get_box_replacement()
    //   if preamble_op.invented_name:
    //       self.extra_same_as.append(op)
    //   self.used_boxes.append(op)
    //   self.short_preamble_jump.append(preamble_op.preamble_op)
    //
    // `used_boxes` is appended to `label_op.arglist()` in its entirety, with
    // NO deduplication against the base label args. RPython's Box identity
    // makes every used_box a distinct Python object; even two boxes that
    // happen to carry the same runtime value are separate slots in the
    // label. In majit's flat OpRef model we carry this literally — do not
    // drop entries just because their OpRef coincides with a base label
    // arg, because that silently corrupts the loop arity (the matching
    // `extra_jump_args[idx]` value that the JUMP was supposed to deliver
    // into that slot disappears along with it).
    //
    // The only entries that legitimately drop out here are literal Const
    // boxes. `used_boxes` / `short_preamble_jump` are RPython Box lists;
    // a stale backend-constants entry must not make us drop a live runtime
    // Box from the LABEL/JUMP contract (e.g. an imported short-preamble
    // HeapField result that still has optimizer constant knowledge attached).
    for (idx, &label_arg) in extra_label_args.iter().enumerate() {
        let jump_arg = extra_jump_args.get(idx).copied().unwrap_or(label_arg);
        if !short_extra_is_carried(label_arg) {
            continue;
        }
        filtered_extra_label_args.push(label_arg);
        filtered_extra_jump_args.push(jump_arg);
    }
    let next_free_pos = |mut next: u32| {
        next = next.max(inputarg_base + body_num_inputs as u32);
        while constants.contains_key(&next) {
            next += 1;
        }
        next
    };

    // Retrace (`compile_retrace`) sets `emit_start_label = false` and
    // passes no start descr. Remap and `stream_defs` must then use the
    // loop LABEL that is actually emitted; `start_label_args` would be
    // producerless and `stream_defs` would hide the hole.
    let emit_start_label = start_label_descr.is_some();
    if let Some(start_label_descr) = start_label_descr {
        let mut start_label_args_box_operand: Vec<majit_ir::operand::Operand> =
            Vec::with_capacity(start_label_args.len());
        for a in start_label_args {
            start_label_args_box_operand.push(ctx.materialize_operand_at(*a));
        }
        let mut start_label = Op::new(OpCode::Label, &start_label_args_box_operand);
        start_label.pos().set(OpRef::NONE);
        start_label.setdescr(start_label_descr);
        result.push(OpRc::new(start_label));
    }

    // Preamble: everything except Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump {
            break;
        }
        result.push(OpRc::new((**op).clone()));
    }

    // max_pos must account for ALL p1_ops positions, including SameAs
    // ops AFTER the Jump that weren't copied into result. These positions
    // are referenced by the body Label args and must not be reused.
    let result_max = result
        .iter()
        .map(|op| op.pos().get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(inputarg_base + body_num_inputs as u32);
    let p1_all_max = p1_ops
        .iter()
        .map(|op| op.pos().get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut max_pos = result_max.max(p1_all_max);
    max_pos = next_free_pos(max_pos.saturating_add(1));
    emit_alias_same_as_for_imports(&mut result, imported_short_aliases);

    // Label position
    let label_pos = next_free_pos(max_pos);
    // compile.py:327-328 splices `label_op` verbatim: the label arg list produced
    // by import_state (make_inputargs) IS the loop-header contract, and the jump
    // args (assemble_jump below) enumerate the SAME VirtualState positionally, so
    // the two must stay arity-aligned. Constant virtual-state slots were already
    // dropped at enum time (virtualstate.rs NotVirtual `position_in_notvirtuals`
    // only assigned to non-LEVEL_CONSTANT leaves), so every arg present here is a
    // live carried position. Do NOT re-derive the slot set from the post-hoc
    // backend-constants map: a phase-2 guard postprocess may const-forward a
    // still-live loop-carried box (e.g. the exhaust-guard's `remaining` proving
    // `<=0 ∧ >=0 ⇒ ==0`), and dropping that label slot while the jump keeps
    // rebinding it desyncs the label/jump contract and orphans the head guard's
    // operand. RPython performs no such filter.
    let mut full_label_args: Vec<OpRef> = label_args.to_vec();
    debug_assert!(
        full_label_args.iter().all(|arg| !arg.is_constant()),
        "base label arg is an inline-Const OpRef; LEVEL_CONSTANT virtual-state \
         slots must be dropped at make_inputargs enum time, not at assembly"
    );

    // Collect preamble-defined OpRefs BEFORE adding extra label args,
    // so we can filter out virtual remnants (removed New ops).
    //
    // resoperation.py InputArgInt/727/739 InputArg{Int,Ref,Float}: preamble-defined
    // inputargs carry `box.type` intrinsically (history.py:220). Mint
    // typed variants from `inputarg_types` so this set's OpRefs match the
    // typed mints used at trace start / Phase 2 import under variant-aware
    // Eq.
    let preamble_defs: crate::FxIndexSet<OpRef> = {
        let mut s: crate::FxIndexSet<OpRef> = (0..body_num_inputs)
            .map(|i| {
                let pos = inputarg_base + i as u32;
                // history.py:220 box.type / resoperation.py InputArgInt/727/739  allow-line-citation
                // InputArgInt/Float/Ref invariant: every label / inputarg
                // carries a value-box class. RPython has no InputArgVoid;
                // a missing type at this site is a structural bookkeeping
                // bug. `inputarg_type_at_strict` panics on miss with the
                // RPython-citation message rather than recovering into
                // VoidOp / a guessed default.
                let tp = ctx.inputarg_type_at_strict(i);
                OpRef::input_arg_typed(pos, tp)
            })
            .collect();
        for op in &result {
            if !op.pos().get().is_none()
                && op.opcode != OpCode::Jump
                && op.result_type() != Type::Void
            {
                s.insert(op.pos().get());
            }
        }
        s
    };

    // Append non-constant extra label args. The caller (unroll pass)
    // determines which extra values the loop header needs; virtual
    // remnants have already been filtered by the caller.
    //
    // Advance max_pos past every position already in use before allocating
    // fresh SameAs positions. In RPython, Box identity prevents collisions;
    // in the flat OpRef model, the assembly-allocated SameAs position must
    // not overlap base label_args, filtered_extra_label_args, or p2_ops
    // positions/args/fail_args — each of those may be referenced by the
    // body. A single colliding OpRef aliases two distinct values and
    // corrupts the loop.
    for &la in &full_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.raw().saturating_add(1));
        }
    }
    for &la in &filtered_extra_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.raw().saturating_add(1));
        }
    }
    for op in p2_ops.iter() {
        if is_trace_runtime_ref(op.pos().get(), constants) {
            max_pos = max_pos.max(op.pos().get().raw().saturating_add(1));
        }
        for arg in op.args_slice().iter() {
            if is_trace_runtime_ref(arg.to_opref(), constants) {
                max_pos = max_pos.max(arg.to_opref().raw().saturating_add(1));
            }
        }
        if let Some(fa) = op.guard_fail_args() {
            for arg in fa.iter() {
                if is_trace_runtime_ref(arg.to_opref(), constants) {
                    max_pos = max_pos.max(arg.to_opref().raw().saturating_add(1));
                }
            }
        }
    }
    let extra_label_start_idx = full_label_args.len();
    full_label_args.extend(filtered_extra_label_args.iter().copied());

    // RPython compile.py parity: after the loop label, only the loop-header
    // contract is live. When `splice_redirected_tail` glues a redirected
    // tail onto the body, the spliced output may mention OpRefs whose
    // defining op was removed (the section between the splice point and
    // the original Jump). Carry such "body use-before-def" references
    // through the label so the assembled body doesn't contain dangling
    // references. With Phase 2's disjoint OpRef namespace, body op args
    // are pre-resolved through ctx.get_box_replacement, so the carried
    // OpRef can be appended directly to full_label_args — the JUMP's
    // mapped_base_args path picks up the corresponding fresh value on the
    // next iteration. The filter only needs to skip filtered_extra_jump_args.
    let mut carried_source_slots: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
    carried_source_slots.extend(filtered_extra_jump_args.iter().copied());
    // `label_set` tracks which OpRefs are already carried by the label so
    // that the body-use-before-def pass doesn't add the same OpRef twice
    // (which would inflate label arity without adding new information).
    // RPython's Box identity makes this a correctness filter: two
    // references to the same Box collapse into one live-in slot. This
    // is NOT the Issue 1 dedup — which drops distinct Boxes that happen
    // to share an OpRef — it is RPython parity: the same Box appears
    // once in the label arglist.
    let mut label_set: crate::FxIndexSet<OpRef> = full_label_args.iter().copied().collect();
    let mut fallthrough_aliases = Vec::new();
    let remap_dsts: &[OpRef] = if emit_start_label {
        start_label_args
    } else {
        full_label_args.as_slice()
    };
    // Values available BEFORE the loop LABEL. The start LABEL's args are
    // live only when that LABEL is actually emitted; the loop LABEL's
    // args are destinations of that LABEL, not producers. Seeding them
    // here drops the fallthrough SameAs that binds a preamble box onto
    // a remapped LABEL slot.
    let mut stream_defs: crate::FxIndexSet<OpRef> = if emit_start_label {
        start_label_args
            .iter()
            .copied()
            .filter(|a| is_trace_runtime_ref(*a, constants))
            .collect()
    } else {
        crate::FxIndexSet::default()
    };
    for op in &result {
        if !op.pos().get().is_none() && op.opcode != OpCode::Jump && op.result_type() != Type::Void
        {
            stream_defs.insert(op.pos().get());
        }
    }
    // Phase 2 optimizer inputargs live at `[inputarg_base, inputarg_base+n)`.
    // The assembled trace shares `[0, n)` with the preamble (`shift_back`
    // used to do this translation; assembly now owns it). A Phase-2
    // InputArg that is not rewritten here is appended as a LABEL live-in
    // with no producer (`InputArgRef(99)` with token inputs `[0, 1]`).
    let mut phase2_input_remap: rustc_hash::FxHashMap<OpRef, OpRef> =
        rustc_hash::FxHashMap::default();
    if inputarg_base > 0 {
        for i in 0..body_num_inputs {
            let tp = ctx.inputarg_type_at_strict(i);
            let src = OpRef::input_arg_typed(inputarg_base + i as u32, tp);
            let dst = remap_dsts
                .get(i)
                .copied()
                .unwrap_or_else(|| OpRef::input_arg_typed(i as u32, tp));
            if src != dst {
                phase2_input_remap.insert(src, dst);
            }
        }
    }
    {
        let mut seen_body_defs = crate::FxIndexSet::default();
        for op in p2_ops {
            // compile.py assembles the loop LABEL from `label_op` plus
            // short-preamble `used_boxes`; the terminal JUMP's target-local
            // payload is not part of that contract. In particular,
            // jump_to_preamble retargets the body JUMP to the preamble start
            // label, so carrying its extra args into the loop LABEL mutates
            // the body arity in a way upstream never does.
            if op.opcode == OpCode::Jump {
                continue;
            }
            let mut consider_arg = |arg: OpRef| {
                if !is_trace_runtime_ref(arg, constants) {
                    return; // skip NONE and constants
                }
                if label_set.contains(&arg)
                    || carried_source_slots.contains(&arg)
                    || seen_body_defs.contains(&arg)
                    || phase2_input_remap.contains_key(&arg)
                {
                    return;
                }
                // Folded ConstPtr: define the reminted index in the
                // preamble rather than carrying it on the header.
                // `forget_optimization_info` then clears `_forwarded`,
                // so the backend only sees the SameAs + Const.
                if let Some(const_op) = const_ref_replacement(ctx, arg) {
                    let tp = const_op.type_();
                    if tp != Type::Void {
                        let mut same_as = Op::new(OpCode::same_as_for_type(tp), &[const_op]);
                        same_as.pos().set(arg);
                        fallthrough_aliases.push(same_as);
                    }
                    return;
                }
                push_fallthrough_same_as(
                    &mut fallthrough_aliases,
                    &stream_defs,
                    arg,
                    constants,
                    ctx,
                );
                full_label_args.push(arg);
                appended_label_args.push(arg);
                label_set.insert(arg);
            };
            for a in op.args_slice().iter() {
                consider_arg(a.to_opref());
            }
            op.visit_failarg_oprefs(&mut consider_arg);
            if op.result_type() != Type::Void && !op.pos().get().is_none() {
                seen_body_defs.insert(op.pos().get());
            }
        }
    }

    // compile.py emit_op / get_box_replacement: a residual virtualizable
    // slot on the body LABEL is the same Box as `loop.inputargs[i]`. A
    // reminted producerless RefOp loses that identity — recover it when
    // the arg still forwards to an InputArg so `patch_new_loop` can
    // rewrite the slot to the GETARRAYITEM it installs on that inputarg.
    for a in &mut full_label_args {
        let resolved = ctx.get_replacement_opref(*a);
        if resolved.is_input_arg() {
            *a = resolved;
        }
    }

    // SameAs for every LABEL arg that is a forwarded preamble box, not
    // only extras discovered from a body use-before-def. Base label_args
    // sit in `label_set` from the start, so the scan above never saw them.
    for &arg in &full_label_args {
        if arg.is_none() {
            continue;
        }
        if fallthrough_aliases.iter().any(|op| op.pos().get() == arg) {
            continue;
        }
        push_fallthrough_same_as(&mut fallthrough_aliases, &stream_defs, arg, constants, ctx);
        if fallthrough_aliases.iter().any(|op| op.pos().get() == arg) {
            continue;
        }
        if let Some(const_op) = const_ref_replacement(ctx, arg) {
            let tp = const_op.type_();
            if tp != Type::Void {
                let mut same_as = Op::new(OpCode::same_as_for_type(tp), &[const_op]);
                same_as.pos().set(arg);
                fallthrough_aliases.push(same_as);
            }
        }
    }

    let mut full_label_args_box_operand: Vec<majit_ir::operand::Operand> =
        Vec::with_capacity(full_label_args.len());
    for a in &full_label_args {
        full_label_args_box_operand.push(ctx.materialize_operand_at(*a));
    }
    let mut label_op = Op::new(OpCode::Label, &full_label_args_box_operand);
    // resoperation.py AbstractResOp.type = 'v' default — Label has no
    // result Box, so its OpRef position carries the Void tag rather than
    // a stray Int tag. `op_index` filters Void ops so this OpRef never
    // shadows a real Box-bearing op at the same raw position, and
    // variant-aware Hash matches when downstream consumers compare
    // label_op.pos against Op-keyed maps.
    label_op
        .pos()
        .set(OpRef::op_typed(label_pos, label_op.result_type()));
    if let Some(ref d) = loop_label_descr {
        label_op.setdescr(d.clone());
    }
    result.extend(fallthrough_aliases.into_iter().map(OpRc::new));
    result.push(OpRc::new(label_op));

    // Body: 2-pass remap (inputarg refs -> label args, body results -> fresh boxes)
    let max_label_arg_pos = full_label_args
        .iter()
        .map(|a| a.raw())
        .max()
        .unwrap_or(label_pos);
    let max_emitted_pos = result
        .iter()
        .map(|op| op.pos().get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(label_pos);
    // Fresh body positions must be higher than ALL existing positions:
    // already-emitted preamble/imported-short ops, label, label args, AND
    // Phase 2 op positions (which may be higher than label positions due
    // to Phase 2 remap or redirect ops).
    //
    // RPython's distinct Box identities prevent an imported short-preamble
    // replay result from colliding with a freshly emitted body result. In
    // majit's flat OpRef model we must keep the body allocator above every
    // position already emitted into `result`.
    let max_p2_pos = p2_ops
        .iter()
        .map(|op| op.pos().get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut next_body_pos = next_free_pos(
        max_emitted_pos
            .max(label_pos)
            .max(max_label_arg_pos)
            .max(max_p2_pos)
            .saturating_add(1),
    );
    // Keyed lookup only (`get`/`insert`, never iterated — codegen order comes
    // from the `p2_ops` walk below), so a hash map keeps each per-op remap O(1)
    // instead of the IndexMap linear `get_index_of` scan on long peeled traces.
    let mut body_result_remap: rustc_hash::FxHashMap<OpRef, OpRef> =
        rustc_hash::FxHashMap::default();
    let visible_before_label: crate::FxIndexSet<OpRef> = full_label_args
        .iter()
        .copied()
        .chain(preamble_defs.iter().copied())
        .collect();

    let mut assembly_alias_remap: rustc_hash::FxHashMap<OpRef, OpRef> = phase2_input_remap;
    for (i, &source_slot) in filtered_extra_label_args.iter().enumerate() {
        if source_slot.is_none() {
            continue;
        }
        let Some(&extended_label_arg) = full_label_args.get(extra_label_start_idx + i) else {
            continue;
        };
        if let Some(&jump_source) = filtered_extra_jump_args.get(i)
            && !jump_source.is_none()
            && jump_source != source_slot
        {
            let b_js = ctx.materialize_operand_at(jump_source);
            let b_ela = match ctx.get_box_replacement_operand_opt(extended_label_arg) {
                Some(b) => b,
                None => ctx.materialize_operand_at(extended_label_arg),
            };
            ctx.make_equal_to(&b_js, &b_ela);
            assembly_alias_remap.insert(jump_source, extended_label_arg);
        }
    }
    for op in p2_ops.iter() {
        // Only map non-Void ops that actually produce a result.
        // Void ops (SetfieldGc, guards, Jump) don't define values at
        // their position — mapping them creates phantom OpRefs.
        if op.pos().get().raw() != u32::MAX && op.result_type() != Type::Void {
            // history.py:220 box.type / resoperation.py/589/615 IntOp /
            // RefOp / FloatOp.type — the fresh result Box inherits the
            // producing op's type tag so downstream readers (`opref_type`
            // typed-first arm + variant-aware HashMap/HashSet lookups)
            // see the correct `box.type` instead of a default-int guess.
            let fresh = OpRef::op_typed(next_body_pos, op.result_type());
            next_body_pos = next_free_pos(next_body_pos.saturating_add(1));
            body_result_remap.insert(op.pos().get(), fresh);
        }
    }

    let mut seen_body_defs = crate::FxIndexSet::default();
    let mut current_inner_label_index: Option<usize> = None;
    let mut defs_since_inner_label: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
    // Combined-trace position → emitted clone, so remap hits bind to the
    // body clone producer instead of re-minting a position-only box (SSA:
    // a remapped arg's target clone was pushed in an earlier iteration).
    let mut emitted_at: rustc_hash::FxHashMap<OpRef, majit_ir::OpRc> =
        rustc_hash::FxHashMap::default();
    for (op_idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = (**op).clone();
        // compile.py never snapshots every body's arglist. Only the Label
        // arm extends this vec; a Vec per p2 op was a 32 B malloc on
        // every getfield/setfield (regex and/or 0.15 class).
        let mut original_args: Vec<OpRef> = if op.opcode == OpCode::Label {
            op.args_slice().iter().map(|a| a.to_opref()).collect()
        } else {
            Vec::new()
        };
        if let Some(&mapped_pos) = body_result_remap.get(&op.pos().get()) {
            new_op.pos().set(mapped_pos);
        }
        // Body op args were already resolved at emit time by
        // optimizer.py / Optimizer::emit_operation. Do not walk
        // forwarding chains again here: postprocess_GUARD_TRUE/FALSE may have
        // installed Const forwarding after the guard was emitted, and PyPy keeps
        // the guard's original runtime argument.
        let remap_body_arg = |arg: OpRef,
                              assembly_alias_remap: &rustc_hash::FxHashMap<OpRef, OpRef>,
                              body_result_remap: &rustc_hash::FxHashMap<OpRef, OpRef>,
                              seen_body_defs: &crate::FxIndexSet<OpRef>,
                              visible_before_label: &crate::FxIndexSet<OpRef>|
         -> OpRef {
            if let Some(&mapped) = assembly_alias_remap.get(&arg) {
                return mapped;
            }
            if let Some(&mapped) = body_result_remap.get(&arg)
                && (seen_body_defs.contains(&arg) || !visible_before_label.contains(&arg))
            {
                return mapped;
            }
            arg
        };
        // optimizer.py force_box loop pattern:
        //   for i in range(op.numargs()): op.setarg(i, ...)
        // Only remap hits are rewritten; a miss keeps the existing operand
        // (a bound operand stays live-tracking instead of degrading to a
        // position-only re-mint of the same position).
        for i in 0..new_op.num_args() {
            let arg = new_op.arg(i).to_opref();
            let mapped = remap_body_arg(
                arg,
                &assembly_alias_remap,
                &body_result_remap,
                &seen_body_defs,
                &visible_before_label,
            );
            if mapped != arg {
                // A remap hit whose target clone was already pushed binds to
                // that producer; otherwise route through the canonical
                // "box always exists" materializer (parity with the start_label
                // / jump_source / extended_label_arg sites above) so the arg
                // carries a bound `Operand::Op`/`InputArg` instead of a
                // position-only box. `materialize_operand_at(mapped).to_opref() ==
                // mapped`, so the rewritten arg is OpRef-identical.
                let boxed = match emitted_at.get(&mapped) {
                    Some(rc) => majit_ir::operand::Operand::from_bound_op(rc),
                    None => ctx.materialize_operand_at(mapped),
                };
                new_op.setarg(i, boxed);
            }
        }
        if new_op.opcode == OpCode::Label {
            let mut seen_after_label_defs = crate::FxIndexSet::default();
            let mut extra_inner_sources = Vec::new();
            let mut extra_inner_set = crate::FxIndexSet::default();
            let label_arg_set: crate::FxIndexSet<OpRef> = original_args
                .iter()
                .copied()
                .filter(|arg| !arg.is_none())
                .collect();
            for later_op in p2_ops.iter().skip(op_idx + 1) {
                let mut consider_arg = |arg: OpRef| {
                    // unroll.py `_map_args` passes Const through unchanged
                    // — inline-Const args (history.py/268/314) carry their
                    // value on the OpRef itself, so they are never label-args.
                    // `arg.is_constant()` short-circuits before `.raw()` (which
                    // panics on inline-Const variants).
                    if arg.is_none()
                        || arg.is_constant()
                        || constants.contains_key(&arg.raw())
                        || label_arg_set.contains(&arg)
                        || extra_inner_set.contains(&arg)
                        || seen_after_label_defs.contains(&arg)
                    {
                        return;
                    }
                    // optimizer.py freezes op args at emit time;
                    // walking ctx.get_box_replacement here would follow Const
                    // forwarding that postprocess installed AFTER the body op
                    // was emitted (corrupting already-emitted trace). The
                    let resolved_arg = assembly_alias_remap.get(&arg).copied().unwrap_or(arg);
                    let available_before_label = visible_before_label.contains(&arg)
                        || visible_before_label.contains(&resolved_arg)
                        || seen_body_defs.contains(&arg);
                    if available_before_label {
                        extra_inner_set.insert(arg);
                        extra_inner_sources.push(arg);
                    }
                };
                for a in later_op.args_slice().iter() {
                    consider_arg(a.to_opref());
                }
                later_op.visit_failarg_oprefs(&mut consider_arg);
                if later_op.result_type() != Type::Void && !later_op.pos().get().is_none() {
                    seen_after_label_defs.insert(later_op.pos().get());
                }
            }
            // RPython Box parity: do not dedup the inner-label extension
            // against its existing args. Each source_arg collected from
            // the later body is a distinct RPython Box; adding it as a
            // separate slot matches `label_op.initarglist(label_op.getarglist()
            // + sb.used_boxes)` (unroll.py:300) where RPython never filters
            // by value or by box coincidence. The outer collection above
            // already dedups by box identity (`extra_inner_set.contains`),
            // so each source_arg appears at most once — which is the
            // RPython-parity behavior for Box-keyed live-in sets.
            // unroll.py:301 label_op.initarglist(label_op.getarglist() +
            //                                    sb.used_boxes)
            let mut extended_args: smallvec::SmallVec<[OpRef; 3]> = new_op
                .getarglist_copy()
                .iter()
                .map(|a| a.to_opref())
                .collect();
            for &source_arg in &extra_inner_sources {
                // optimizer.py:614-625 freeze: do not follow ctx forwarding
                // chains here; postprocess Const forwarding on body ops would
                // otherwise leak into the extended Label's arg list, turning
                // a runtime inputarg slot into a Const reference.
                let resolved = assembly_alias_remap
                    .get(&source_arg)
                    .copied()
                    .unwrap_or(source_arg);
                let mapped_arg = remap_body_arg(
                    resolved,
                    &assembly_alias_remap,
                    &body_result_remap,
                    &seen_body_defs,
                    &visible_before_label,
                );
                extended_args.push(mapped_arg);
                original_args.push(source_arg);
            }
            let mut extended_args_box: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                smallvec::SmallVec::with_capacity(extended_args.len());
            for a in &extended_args {
                extended_args_box.push(ctx.materialize_operand_at(*a));
            }
            new_op.initarglist(extended_args_box);
        }
        if new_op.opcode == OpCode::Jump {
            // unroll.py jump_to_preamble sends the live JUMP after
            // force_box / send_extra_operation rewrites. Body args reference
            // the trace inputarg slots OpRef(0)..OpRef(start_label_args.len());
            // remap those positional refs to start_label_args[i].
            let mapped_base_args: Vec<OpRef> = new_op
                .args_slice()
                .iter()
                .map(|arg| {
                    let arg = arg.to_opref();
                    if jump_was_redirected {
                        return arg;
                    }
                    // unroll.py `_map_args` passes Const through unchanged.
                    // Inline-Const variants (history.py/268/314) panic on
                    // `.raw()`; the inline payload IS the value and has no
                    // positional remap target.
                    if arg.is_constant() {
                        return arg;
                    }
                    if let Some(&mapped) = assembly_alias_remap.get(&arg) {
                        return mapped;
                    }
                    let i = if arg.is_input_arg() && arg.raw() >= inputarg_base {
                        (arg.raw() - inputarg_base) as usize
                    } else {
                        arg.raw() as usize
                    };
                    start_label_args.get(i).copied().unwrap_or(arg)
                })
                .collect();
            let target_label_args: Vec<OpRef> = current_inner_label_index
                .and_then(|label_idx| {
                    result
                        .get(label_idx)
                        .map(|op| op.args_slice().iter().map(|a| a.to_opref()).collect())
                })
                .unwrap_or_else(|| full_label_args.clone());
            let jump_target_descr_idx = new_op.getdescr().map(|descr| descr.index());
            let local_label_descr_idx = match current_inner_label_index {
                Some(label_idx) => result
                    .get(label_idx)
                    .and_then(|op| op.getdescr())
                    .map(|descr| descr.index()),
                None => loop_label_descr.as_ref().map(|descr| descr.index()),
            };
            let jump_targets_local_label = matches!(
                (jump_target_descr_idx, local_label_descr_idx),
                (Some(jump_idx), Some(label_idx)) if jump_idx == label_idx
            );
            let target_base_len = if current_inner_label_index.is_some() {
                op.num_args()
            } else {
                label_args.len()
            };
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] assemble_jump: inner_label={:?} redirected={} targets_local={} jump_target_descr={:?} local_label_descr={:?} original_args={:?} mapped_base_args={:?} label_args={:?} filtered_extra_jump_args={:?}",
                    current_inner_label_index,
                    jump_was_redirected,
                    jump_targets_local_label,
                    jump_target_descr_idx,
                    local_label_descr_idx,
                    original_args,
                    mapped_base_args,
                    label_args,
                    filtered_extra_jump_args,
                );
            }
            let mut jump_args = mapped_base_args;
            if jump_was_redirected && jump_targets_local_label {
                // compile.py:334: same-label closes require equal JUMP and LABEL
                // arity. A JUMP redirected to another token already carries the
                // arity produced by that token's virtual state (unroll.py:346-357)
                // and must pass through unchanged.
                if jump_args.len() > target_label_args.len() {
                    jump_args.truncate(target_label_args.len());
                }
                // Pad if JUMP is shorter than the target label.
                while jump_args.len() < target_label_args.len() {
                    let extra_idx = jump_args.len().saturating_sub(target_base_len);
                    // shortpreamble.py add_preamble_op
                    // ExtendedShortPreambleBuilder.add_preamble_op:
                    //
                    //   self.label_args.append(op)
                    //   self.jump_args.append(preamble_op.preamble_op)
                    //
                    // The extra LABEL slot receives the value produced by
                    // replaying the short-preamble op on the back edge.  This
                    // is equally true when the target is an inner/self LABEL;
                    // passing `target_label_args[jump_args.len()]` there feeds
                    // the LABEL's own input box back to itself and freezes the
                    // entry value instead of re-reading a mutable field/array.
                    let extra_arg = filtered_extra_jump_args
                        .get(extra_idx)
                        .copied()
                        .unwrap_or(target_label_args[jump_args.len()]);
                    jump_args.push(extra_arg);
                }
            }
            // unroll.py-style bulk replace: jump arity is finalized here.
            let mut jump_args_box: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                smallvec::SmallVec::with_capacity(jump_args.len());
            for a in &jump_args {
                jump_args_box.push(ctx.materialize_operand_at(*a));
            }
            new_op.initarglist(jump_args_box);
        }
        // RPython resume.py parity: fail_args capture guard-point state
        // (the snapshot), not the body's final state. body_result_remap
        // applies only to values body-defined AND not visible before the
        // label, so snapshot refs to label args stay intact.
        if let Some(fa) = new_op.fail_args_mut() {
            for a in fa.iter_mut() {
                let current = a.to_opref();
                if let Some(&mapped) = assembly_alias_remap.get(&current) {
                    *a = match emitted_at.get(&mapped) {
                        Some(rc) => majit_ir::operand::Operand::from_bound_op(rc),
                        None => ctx.materialize_operand_at(mapped),
                    };
                    continue;
                }
                if let Some(&mapped) = body_result_remap.get(&current)
                    && seen_body_defs.contains(&current)
                    && !visible_before_label.contains(&current)
                {
                    *a = match emitted_at.get(&mapped) {
                        Some(rc) => majit_ir::operand::Operand::from_bound_op(rc),
                        None => majit_ir::operand::Operand::from_opref(mapped),
                    };
                }
            }
        }
        // Jump args fill existing label slots. The `full_label_args` scan
        // skips Jump so those args are not appended as extra live-ins.
        if new_op.opcode != OpCode::Jump
            && let Some(label_idx) = current_inner_label_index
        {
            let mut extra_live_args = Vec::new();
            let label_args: smallvec::SmallVec<[OpRef; 3]> = result[label_idx]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect();
            let mut consider_arg = |arg: OpRef| {
                // unroll.py `_map_args` passes Const through; inline-Const
                // (history.py:227/268/314) carries its value on the OpRef and
                // is never an inner-label-extension candidate. Short-circuit
                // before `.raw()` (panics on inline-Const variants).
                if arg.is_none()
                    || arg.is_constant()
                    || constants.contains_key(&arg.raw())
                    || defs_since_inner_label.contains(&arg)
                    || label_args.contains(&arg)
                    || extra_live_args.contains(&arg)
                {
                    return;
                }
                extra_live_args.push(arg);
            };
            for a in new_op.args_slice().iter() {
                consider_arg(a.to_opref());
            }
            new_op.visit_failarg_oprefs(&mut consider_arg);
            if !extra_live_args.is_empty() {
                let existing: crate::FxIndexSet<OpRef> = result[label_idx]
                    .args_slice()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect();
                let mut new_args: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                    (0..result[label_idx].num_args())
                        .map(|i| result[label_idx].arg(i))
                        .collect();
                new_args.extend(
                    extra_live_args
                        .into_iter()
                        .filter(|arg| !existing.contains(arg))
                        .map(|arg| ctx.materialize_operand_at(arg)),
                );
                result[label_idx].initarglist(new_args);
            }
        }
        // RPython parity: each guard in the assembled trace owns a
        // distinct ResumeGuardDescr with a globally unique fail_index.
        // optimizeopt::store_final_boxes_in_guard (`optimizeopt/mod.rs`)
        // installs a fresh ResumeGuardDescr on every optimizer-routed
        // guard, and Optimizer::_copy_resume_data_from /
        // OptContext::emit_guard_operation share-branch call
        // Descr::clone_descr to allocate a fresh fail_index for the
        // sharing-path guard. Both phases of the unroll optimizer
        // therefore emit body guards with unique descrs already; no
        // post-process re-stamping is needed.
        let new_rc = OpRc::new(new_op);
        if new_rc.result_type() != Type::Void && !new_rc.pos().get().is_none() {
            emitted_at.insert(new_rc.pos().get(), new_rc.clone());
        }
        result.push(new_rc);
        if op.opcode == OpCode::Label {
            current_inner_label_index = Some(result.len() - 1);
            defs_since_inner_label.clear();
        }
        if op.result_type() != Type::Void && !op.pos().get().is_none() {
            seen_body_defs.insert(op.pos().get());
            defs_since_inner_label.insert(result.last().unwrap().pos().get());
        }
    }

    if crate::callee_rca_enabled() {
        let labels: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == OpCode::Label)
            .map(|(idx, op)| {
                (
                    idx,
                    op.pos().get(),
                    op.args_slice()
                        .iter()
                        .map(|a| a.to_opref())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let jumps: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == OpCode::Jump)
            .map(|(idx, op)| {
                (
                    idx,
                    op.pos().get(),
                    op.args_slice()
                        .iter()
                        .map(|a| a.to_opref())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        eprintln!(
            "[callee-rca][assembled-loop] labels={:?} jumps={:?}",
            labels, jumps
        );
    }

    result
}

fn splice_redirected_tail(
    body_ops: &[majit_ir::OpRc],
    redirected_tail_ops: &[majit_ir::OpRc],
) -> Vec<majit_ir::OpRc> {
    let mut result = Vec::with_capacity(body_ops.len() + redirected_tail_ops.len());
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.extend_from_slice(redirected_tail_ops);
    result
}

fn replace_terminal_jump(body_ops: &[majit_ir::OpRc], jump_op: Op) -> Vec<majit_ir::OpRc> {
    let mut result = Vec::with_capacity(body_ops.len() + 1);
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.push(OpRc::new(jump_op));
    result
}

/// True when a positional JUMP box and the start LABEL box at the same
/// index carry different `AbstractValue.type` tags.
///
/// `unroll.py jump_to_preamble` keeps the JUMP arglist. Pad/truncate is
/// only sound for an extra trailing invariant of the same kind; a kind
/// disagreement means a missing middle slot, and the assembler would
/// remap an Int into a Ref (`x86/assembler.py` `_cmp_guard_class`).
fn jump_and_preamble_slot_kinds_disagree(jump: &Op, preamble_args: &[OpRef]) -> bool {
    jump.args_slice()
        .iter()
        .map(|a| a.to_opref())
        .zip(preamble_args.iter().copied())
        .any(
            |(jump_arg, preamble_arg)| match (jump_arg.ty(), preamble_arg.ty()) {
                (Some(jump_ty), Some(preamble_ty)) => jump_ty != preamble_ty,
                _ => false,
            },
        )
}

/// Match a jump_to_preamble JUMP to the start LABEL's
/// `ExportedState.renamed_inputargs`. Extra LABEL slots are
/// loop-invariant entry boxes: pad with the preamble arg at that
/// index so `x86/regalloc.py` `consider_jump` identity-remaps them.
/// A longer JUMP is truncated to the LABEL contract
/// (`_compute_hint_locations_from_descr` asserts equal length).
fn reshape_jump_args_for_preamble(jump_args: &mut Vec<OpRef>, preamble_args: &[OpRef]) {
    if jump_args.len() > preamble_args.len() {
        jump_args.truncate(preamble_args.len());
    }
    while jump_args.len() < preamble_args.len() {
        jump_args.push(preamble_args[jump_args.len()]);
    }
}

fn apply_preamble_shape_to_jump(jump: &mut Op, preamble_args: &[OpRef]) {
    let mut jump_args: Vec<OpRef> = jump.args_slice().iter().map(|a| a.to_opref()).collect();
    let old_len = jump_args.len();
    reshape_jump_args_for_preamble(&mut jump_args, preamble_args);
    if jump_args.len() == old_len {
        return;
    }
    let mut boxes: Vec<Operand> = jump.args_slice().iter().cloned().collect();
    if jump_args.len() < old_len {
        boxes.truncate(jump_args.len());
    } else {
        for extra in &jump_args[old_len..] {
            boxes.push(Operand::bound_from_opref(*extra));
        }
    }
    jump.initarglist(boxes);
}

fn pick_virtual_state(
    my_vs: &crate::optimizeopt::virtualstate::VirtualState,
    target_states: &[crate::optimizeopt::virtualstate::VirtualState],
    ctx: &mut OptContext,
) -> Option<usize> {
    for (i, target_vs) in target_states.iter().enumerate() {
        if target_vs.generalization_of(my_vs, ctx) {
            return Some(i);
        }
    }
    None
}

pub struct OptUnroll {
    /// Buffer of ops received before the Jump back-edge.
    buffer: Vec<Op>,
    /// Whether a Jump was already seen (avoid double-unrolling).
    seen_jump: bool,
}

impl OptUnroll {
    pub fn new() -> Self {
        OptUnroll {
            buffer: Vec::new(),
            seen_jump: false,
        }
    }

    /// Peel one iteration of the loop body.
    ///
    /// Given the buffered ops (everything before the Jump), this emits:
    /// 1. The peeled (duplicated) body with remapped OpRefs
    /// 2. A Label op marking the loop header
    /// 3. The original body ops
    ///
    /// The caller is responsible for emitting the final Jump.
    /// Emit the peeled iteration, the loop Label, and the body iteration
    /// in that order, returning the `original_pos -> body_pos` map the
    /// caller uses to rewrite the back-edge Jump's args.
    ///
    /// Positions are allocated through `reserve_pos` (which honors the
    /// `inputarg_base + num_inputs` floor) instead of being computed
    /// arithmetically from `new_operations.len()`. The arithmetic form
    /// is fine for production traces (inputargs occupy a disjoint low
    /// range) but collides with the 1024-slot `trace_inputargs`
    /// stubs that the unit-test harnesses seed. Using `reserve_pos`
    /// closes that collision and lets the Box.type invariant enforce
    /// itself uniformly at `emit()` / `emit_extra()` /
    /// `propagate_from_pass_range`.
    fn peel_iteration(
        &self,
        jump_op: &Op,
        ctx: &mut OptContext,
    ) -> crate::FxIndexMap<OpRef, OpRef> {
        // First pass: reserve peeled-iteration positions, tagged with each
        // source op's result type ( `OpRef.ty()`
        // matches RPython's `box.type` at allocation time).
        let peeled_positions: Vec<OpRef> = self
            .buffer
            .iter()
            .map(|op| ctx.reserve_pos_typed(op.result_type()))
            .collect();
        let mut ref_map: crate::FxIndexMap<OpRef, OpRef> = crate::FxIndexMap::default();
        for (op, &new_pos) in self.buffer.iter().zip(peeled_positions.iter()) {
            ref_map.insert(op.pos().get(), new_pos);
        }

        // Emit peeled iteration with remapped refs.
        for (op, &new_pos) in self.buffer.iter().zip(peeled_positions.iter()) {
            let mut peeled = op.clone();
            peeled.pos().set(new_pos);
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..peeled.num_args() {
                let arg = peeled.arg(i);
                if let Some(&new_ref) = ref_map.get(&arg.to_opref()) {
                    peeled.setarg(i, remapped_producer_operand(ctx, new_ref));
                }
                // Args referencing ops outside the buffer (e.g., input args)
                // are kept as-is.
            }
            if let Some(fa) = peeled.fail_args_mut() {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = ref_map.get(&arg.to_opref()) {
                        *arg = remapped_producer_operand(ctx, new_ref);
                    }
                }
            }
            if peeled.opcode.is_guard() {
                clone_guard_snapshot_remapped(ctx, &mut peeled, &ref_map);
                // opencoder.py:391-401 parity: trace iteration strips guard
                // descrs, keeping only rd_resume_position. optimizer.py then
                // invents a fresh opcode-appropriate descr at emission time.
                peeled.cleardescr();
            }
            ctx.emit(peeled);
        }

        // Emit Label between peeled and original body.
        // The Label's args match the Jump's args, forming the loop header.
        let label_pos = ctx.reserve_pos_typed(OpCode::Label.result_type());
        let jump_label_args: Vec<majit_ir::operand::Operand> =
            (0..jump_op.num_args()).map(|i| jump_op.arg(i)).collect();
        let mut label_op = Op::new(OpCode::Label, &jump_label_args);
        label_op.pos().set(label_pos);
        ctx.emit(label_op);

        // Reserve body positions, tagged per source op ( ).
        let body_positions: Vec<OpRef> = self
            .buffer
            .iter()
            .map(|op| ctx.reserve_pos_typed(op.result_type()))
            .collect();
        let mut orig_ref_map: crate::FxIndexMap<OpRef, OpRef> = crate::FxIndexMap::default();
        for (op, &new_pos) in self.buffer.iter().zip(body_positions.iter()) {
            orig_ref_map.insert(op.pos().get(), new_pos);
        }

        // Emit original body ops with remapped positions and refs.
        for (op, &new_pos) in self.buffer.iter().zip(body_positions.iter()) {
            let mut body_op = op.clone();
            body_op.pos().set(new_pos);
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..body_op.num_args() {
                let arg = body_op.arg(i);
                if let Some(&new_ref) = orig_ref_map.get(&arg.to_opref()) {
                    body_op.setarg(i, remapped_producer_operand(ctx, new_ref));
                }
            }
            if let Some(fa) = body_op.fail_args_mut() {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = orig_ref_map.get(&arg.to_opref()) {
                        *arg = remapped_producer_operand(ctx, new_ref);
                    }
                }
            }
            if body_op.opcode.is_guard() {
                clone_guard_snapshot_remapped(ctx, &mut body_op, &orig_ref_map);
                // Same opencoder.py guard-descr stripping as the peeled copy.
                // ResumeAtPositionDescr is inline_short_preamble-only
                // (unroll.py:406-409), not normal peeled body guard state.
                body_op.cleardescr();
            }
            ctx.emit(body_op);
        }
        orig_ref_map
    }
}

/// Resolve a remapped peel position to its canonical producer operand,
/// mirroring the import-path binding (`get_box_replacement_operand_opt`, else
/// `materialize_operand_at`). The peeled / body
/// producer emitted at `new_ref` precedes its consumers' arg writes (SSA
/// def-before-use), so the read resolves to the bound `Op`; the
/// `materialize_operand_at` arm mints a registered stand-in that the producer's
/// later `emit` catches up (mod.rs forward-reference path), never a
/// position-only operand.
fn remapped_producer_operand(ctx: &mut OptContext, new_ref: OpRef) -> Operand {
    match ctx.get_box_replacement_operand_opt(new_ref) {
        Some(o) => o,
        None => ctx.materialize_operand_at(new_ref),
    }
}

fn fresh_snapshot_key(ctx: &OptContext) -> i32 {
    next_snapshot_pos(&ctx.snapshot_boxes)
}

fn remap_snapshot_boxes(
    boxes: &[SnapshotBox],
    ref_map: &crate::FxIndexMap<OpRef, OpRef>,
) -> crate::optimizeopt::SnapshotBoxList {
    boxes
        .iter()
        .map(|boxref| boxref.map_opref(|opref| ref_map.get(&opref).copied().unwrap_or(opref)))
        .collect()
}

fn clone_guard_snapshot_remapped(
    ctx: &mut OptContext,
    guard: &mut Op,
    ref_map: &crate::FxIndexMap<OpRef, OpRef>,
) {
    let old_pos = guard.rd_resume_position();
    if old_pos < 0 {
        return;
    }
    let Some(snapshot_boxes) = snapshot_get(&ctx.snapshot_boxes, old_pos).cloned() else {
        return;
    };

    // RPython TraceIterator decodes snapshot box references against the
    // current iteration's box cache.  Majit keeps snapshots in side tables,
    // so cloned guards need a cloned snapshot table entry with the same
    // raw-box -> cloned-box mapping used for op args.
    let new_pos = fresh_snapshot_key(ctx);
    snapshot_insert(
        &mut ctx.snapshot_boxes,
        new_pos,
        remap_snapshot_boxes(&snapshot_boxes, ref_map),
    );
    if let Some(vable_boxes) = snapshot_get(&ctx.snapshot_vable_boxes, old_pos).cloned() {
        snapshot_insert(
            &mut ctx.snapshot_vable_boxes,
            new_pos,
            remap_snapshot_boxes(&vable_boxes, ref_map),
        );
    }
    if let Some(vref_boxes) = snapshot_get(&ctx.snapshot_vref_boxes, old_pos).cloned() {
        snapshot_insert(
            &mut ctx.snapshot_vref_boxes,
            new_pos,
            remap_snapshot_boxes(&vref_boxes, ref_map),
        );
    }
    ctx.snapshot_frame_pcs.copy_run(old_pos, new_pos);
    ctx.snapshot_frame_sizes.copy_run(old_pos, new_pos);
    guard.set_rd_resume_position(new_pos);
}

impl Default for OptUnroll {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptUnroll {
    fn propagate_forward(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // Only peel once per trace, and only for Jump (back-edge).
        if op.opcode == OpCode::Jump && !self.seen_jump {
            self.seen_jump = true;

            if self.buffer.is_empty() {
                // Empty loop body, nothing to peel.
                return OptimizationResult::Emit(op.clone());
            }

            // Perform the peeling: emit peeled body + Label + original body.
            // The returned map forwards original `buffer` positions to
            // their freshly-allocated body positions so we can rewrite
            // the back-edge Jump's args without re-deriving the layout.
            let orig_ref_map = self.peel_iteration(op, ctx);

            // Emit the final Jump with remapped args pointing to the
            // body iteration's ops.
            let mut jump = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..jump.num_args() {
                let arg = jump.arg(i);
                if let Some(&new_ref) = orig_ref_map.get(&arg.to_opref()) {
                    jump.setarg(i, remapped_producer_operand(ctx, new_ref));
                }
            }
            // Reserve the Jump's own position so it lands above any
            // inputarg range and above every body op allocated by
            // peel_iteration. Jump is Void-typed.
            jump.pos().set(ctx.reserve_pos_typed(jump.result_type()));
            return OptimizationResult::Emit(jump);
        }

        // For non-Jump ops (or after we've already unrolled), buffer them.
        if !self.seen_jump {
            self.buffer.push(op.clone());
            return OptimizationResult::Remove;
        }

        // After unrolling, pass everything through.
        OptimizationResult::Emit(op.clone())
    }

    fn setup(&mut self) {
        self.buffer.clear();
        self.seen_jump = false;
    }

    fn name(&self) -> &'static str {
        "unroll"
    }
}

/// A literal Const short-preamble extra is not a LABEL or JUMP slot.
/// The loop assembler and the bridge close both consult this, so a
/// constant virtualizable field cannot make the bridge JUMP one arg
/// longer than the loop LABEL.
fn short_extra_is_carried(arg: OpRef) -> bool {
    !arg.is_constant()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::{rooted_inputarg_operand, rooted_resop_operand};
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::GcRef;
    use majit_ir::operand::Operand;

    #[test]
    fn phase2_snapshot_handoff_keeps_recorded_snapshot_zero() {
        // compile_loop_body moves recorded maps onto UnrollOptimizer and
        // takes them back on InvalidLoop. clone_snapshots_onto must not
        // empty that field (a take left GuardIsnull with no snapshot).
        let mut u = UnrollOptimizer::new();
        let mut boxes = crate::optimizeopt::SnapshotBoxes::new();
        crate::optimizeopt::snapshot_insert(
            &mut boxes,
            0,
            crate::optimizeopt::SnapshotBoxList::default(),
        );
        u.snapshot_boxes = boxes;
        let mut phase2 = Optimizer::default_pipeline();
        u.clone_snapshots_onto(&mut phase2);
        assert!(
            crate::optimizeopt::snapshot_contains(&u.snapshot_boxes, 0),
            "UnrollOptimizer must keep snapshot 0 for the InvalidLoop retry"
        );
        assert!(crate::optimizeopt::snapshot_contains(
            &phase2.snapshot_boxes,
            0
        ));
    }

    /// Publish a `PreviewShortState` the way the preview pass does at its
    /// group site (`optimizer.rs`), so `export_state` reads carried short
    /// boxes instead of recomputing.
    ///
    /// `args_state` stays `None`: these fixtures own no `VirtualState`, so the
    /// export site recomputes it from `end_args` — the reader path a preview
    /// that broke out before the VS capture also takes.
    fn publish_preview_short_state(
        ctx: &mut crate::optimizeopt::OptContext,
        short_inputargs: Vec<OpRef>,
        short_inputarg_refs: Vec<majit_ir::InputArgRc>,
        short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    ) {
        // The production publisher gets this for free from
        // `create_short_inputarg_refs`, which carries its own `debug_assert_eq!`
        // against `self.short_inputargs`.  A fixture builds the two vectors by
        // hand, so restate it here: a refs vector shorter than `short_inputargs`
        // under-roots, and the unrooted renamed `InputArg`'s `Weak` then fails
        // to upgrade past the peel boundary — silently, with no assert between.
        debug_assert_eq!(
            short_inputarg_refs.len(),
            short_inputargs.len(),
            "short_inputarg_refs must root every short_inputargs entry",
        );
        ctx.preview_short_state = Some(crate::optimizeopt::PreviewShortState {
            short_inputargs,
            short_inputarg_refs,
            short_boxes,
            args_state: None,
        });
    }

    /// Mint the fresh renamed short-preamble inputargs for `types`, exactly as
    /// `add_short_input_arg` does (shortpreamble.py:257
    /// `OpHelpers.inputarg_from_tp(box.type)`), for fixtures that carry short
    /// boxes but build no ShortBoxes object of their own.
    fn mint_short_inputargs(
        ctx: &mut crate::optimizeopt::OptContext,
        types: &[Type],
    ) -> (Vec<OpRef>, Vec<majit_ir::InputArgRc>) {
        let mut inputargs = Vec::with_capacity(types.len());
        let mut refs = Vec::with_capacity(types.len());
        for &ty in types {
            let pos = ctx.alloc_op_position_typed(ty).raw();
            let ia = majit_ir::InputArg::from_type_rc(ty, pos);
            inputargs.push(ia.opref());
            refs.push(ia);
        }
        (inputargs, refs)
    }

    /// Assign sequential positions to ops starting from `base`.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos()
                .set(OpRef::op_typed(base + i as u32, op.opcode.result_type()));
        }
    }

    #[test]
    fn test_exported_state_high_water_covers_retrace_namespace() {
        let exported = ExportedState::new(
            vec![OpRef::int_op(52)],
            vec![0],
            vec![
                rooted_resop_operand(Type::Int, 109),
                Operand::from_opref(OpRef::const_int(3)),
            ],
            crate::optimizeopt::virtualstate::VirtualState::new(Vec::new()),
            indexmap::IndexMap::new(),
            Vec::new(),
            Vec::new(),
            vec![OpRef::int_op(14)],
            vec![OpRef::int_op(23)],
            // short_inputarg_refs: this fixture stores plain OpRef positions,
            // so no parallel InputArgRc pool is needed here.
            Vec::new(),
        );

        assert_eq!(exported.opref_high_water(), 110);
    }

    fn run_unroll_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        // See `run_heap_opt` in heap.rs for the 1024-slot Ref seed
        // rationale — the preamble exporter needs an intrinsic type
        // for every renamed inputarg, which production derives from
        // the recorder's trace_inputargs.
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![majit_ir::Type::Ref; 1024]);
        // Production trace iteration gets rd_resume_position from
        // opencoder.py:399-401.  Direct unit-test guards must seed the
        // corresponding snapshot explicitly before store_final_boxes_in_guard.
        let (ops, snapshots) = super::super::seed_guard_snapshots_with(ops, |guard| {
            // Direct unroll tests use guard brackets as the explicit active
            // snapshot so remapping can verify the TraceIterator cache
            // semantics that RPython gets from opencoder.py.
            guard
                .guard_fail_args()
                .map(|fail_args| fail_args.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_default()
        });
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024)
    }

    #[test]
    fn test_no_jump_no_unroll() {
        // Without a Jump back-edge, the pass just buffers and nothing is emitted.
        // (In practice, traces always end with Jump or Finish.)
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Finish, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // IntAdd gets buffered, Finish is not a Jump so it gets buffered too.
        // Nothing is emitted because there's no Jump to trigger peeling.
        // The buffered ops are lost (which is correct: no loop = no unrolling).
        assert!(
            result.is_empty(),
            "no Jump means no loop to unroll, ops are buffered but never emitted"
        );
    }

    #[test]
    fn test_empty_loop_body() {
        // Jump with no prior ops: nothing to peel.
        let mut ops = vec![Op::new(OpCode::Jump, &[])];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Jump);
    }

    #[test]
    fn test_jump_to_preamble_preserves_jump_args() {
        let body_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                op.pos().set(OpRef::int_op(2));
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 2),
                    rooted_resop_operand(Type::Int, 50),
                ],
            ),
        ];
        let preamble_target = TargetToken::new_preamble(7);

        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(OpRc::new).collect();
        let preamble_args = [OpRef::int_op(0), OpRef::int_op(2), OpRef::int_op(50)];
        let result = UnrollOptimizer::jump_to_preamble(&body_ops, &preamble_target, &preamble_args);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(
            result[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(0), OpRef::int_op(2), OpRef::int_op(50)]
        );
        assert_eq!(
            result[1].getdescr().map(|descr| descr.repr()),
            Some("TargetToken(start:7)".to_string())
        );
    }

    #[test]
    fn test_reshape_jump_args_for_preamble_pads_missing_invariants() {
        let mut jump_args = vec![OpRef::int_op(0)];
        reshape_jump_args_for_preamble(&mut jump_args, &[OpRef::int_op(0), OpRef::int_op(1)]);
        assert_eq!(jump_args.as_slice(), &[OpRef::int_op(0), OpRef::int_op(1)]);
    }

    #[test]
    fn test_jump_to_preamble_pads_a_short_body_jump() {
        let body_ops = vec![Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)])];
        let preamble_target = TargetToken::new_preamble(7);
        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(OpRc::new).collect();
        let preamble_args = [OpRef::int_op(0), OpRef::int_op(1)];
        let result = UnrollOptimizer::jump_to_preamble(&body_ops, &preamble_target, &preamble_args);
        assert_eq!(result[0].opcode, OpCode::Jump);
        assert_eq!(
            result[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(0), OpRef::int_op(1)]
        );
        assert_eq!(
            result[0].getdescr().map(|descr| descr.repr()),
            Some("TargetToken(start:7)".to_string())
        );
    }

    #[test]
    fn test_reshape_jump_args_for_preamble_truncates_extra_slots() {
        let mut jump_args = vec![OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)];
        reshape_jump_args_for_preamble(&mut jump_args, &[OpRef::int_op(10), OpRef::int_op(11)]);
        assert_eq!(jump_args.as_slice(), &[OpRef::int_op(0), OpRef::int_op(1)]);
    }

    #[test]
    fn test_jump_and_preamble_slot_kinds_disagree_on_shifted_ref() {
        let jump = Op::new(
            OpCode::Jump,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
        );
        assert!(jump_and_preamble_slot_kinds_disagree(
            &jump,
            &[OpRef::int_op(0), OpRef::ref_op(1)],
        ));
        assert!(!jump_and_preamble_slot_kinds_disagree(
            &jump,
            &[OpRef::int_op(0), OpRef::int_op(1), OpRef::ref_op(2)],
        ));
    }

    #[test]
    fn test_replace_terminal_jump_appends_when_body_prefix_has_no_jump() {
        let body_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(2));
            op
        }];
        let mut jump = Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]);
        jump.setdescr(TargetToken::new_preamble(7).as_jump_target_descr());

        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(OpRc::new).collect();
        let result = replace_terminal_jump(&body_ops, jump);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(
            result[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(2)]
        );
        assert_eq!(
            result[1].getdescr().map(|descr| descr.repr()),
            Some("TargetToken(start:7)".to_string())
        );
    }

    #[test]
    fn test_ensure_preamble_target_token_inserts_start_descr_first() {
        let mut unroll = UnrollOptimizer::new();
        let mut regular = TargetToken::new_loop(3);
        regular.attrs().virtual_state =
            Some(crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(majit_ir::Type::Int),
            ]));
        unroll.target_tokens.push(regular);

        unroll.ensure_preamble_target_token();

        assert_eq!(unroll.target_tokens.len(), 2);
        assert!(unroll.target_tokens[0].is_preamble_target);
        assert!(unroll.target_tokens[0].attrs().virtual_state.is_none());
        assert_eq!(unroll.target_tokens[1].token_id, 3);
    }

    #[test]
    fn test_pick_virtual_state_uses_target_generalization_direction() {
        // NonNull is Ref-typed; target Unknown must also be Ref to match.
        // RPython: NotVirtualStateInfoPtr(LEVEL_UNKNOWN) generalizes
        // NotVirtualStateInfoPtr(LEVEL_NONNULL).
        let my_vs = crate::optimizeopt::virtualstate::VirtualState::new(vec![
            crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull,
        ]);
        let target_states = vec![
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(majit_ir::Type::Ref),
            ]),
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass {
                    class_ptr: 0x1234,
                },
            ]),
        ];

        let mut ctx = OptContext::new(128);
        assert_eq!(
            pick_virtual_state(&my_vs, &target_states, &mut ctx),
            Some(0)
        );
    }

    #[test]
    fn test_exported_state_walks_inline_const_ptr_slots() {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        use crate::optimizeopt::shortpreamble::{
            PreambleOp, PreambleOpKind, ProducedShortOp, ShortPreamble, ShortPreambleOp,
        };
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};

        let old = GcRef(0x1111_0000);
        let new = GcRef(0x2222_0000);
        let old_ref = OpRef::const_ptr(old);
        let new_ref = OpRef::const_ptr(new);
        let mut exported_infos = indexmap::IndexMap::new();
        exported_infos.insert(
            Operand::from_opref(old_ref),
            OpInfo::ptr(PtrInfo::Constant(old)),
        );
        let mut constants = majit_ir::ConstMap::default();
        constants.insert(0, majit_ir::Const::Ref(old));

        let mut state = ExportedState::new(
            vec![old_ref],
            Vec::new(),
            vec![Operand::from_opref(old_ref)],
            VirtualState::new(vec![VirtualStateInfo::Constant(Value::Ref(old))]),
            exported_infos,
            vec![PreambleOp {
                op: OpRc::new(Op::new(OpCode::SameAsR, &[Operand::from_opref(old_ref)])),
                source_op: None,
                res: Operand::bound_from_opref(old_ref),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: Some(Operand::from_opref(old_ref)),
            }],
            // The const-result heap entries ride their own channel, and the
            // walk reaches them by chaining that channel onto
            // `exported_short_boxes`. An inline ConstPtr receiver here has to
            // be forwarded by the same visitor.
            vec![PreambleOp {
                op: OpRc::new(Op::with_descr(
                    OpCode::GetfieldGcI,
                    &[Operand::from_opref(old_ref)],
                    majit_ir::descr::make_field_descr_full(0, 0, 8, Type::Int, true),
                )),
                source_op: None,
                res: Operand::from_opref(OpRef::const_int(7)),
                kind: PreambleOpKind::Heap,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
            vec![old_ref],
            vec![old_ref],
            // short_inputarg_refs: `old_ref` is a ConstPtr inline position.
            Vec::new(),
        );
        state.short_boxes.push((
            old_ref,
            ProducedShortOp {
                kind: PreambleOpKind::Pure,
                res: Operand::from_opref(old_ref),
                preamble_op: OpRc::new(Op::new(OpCode::SameAsR, &[Operand::from_opref(old_ref)])),
                source_op: OpRc::new(Op::new(OpCode::SameAsR, &[Operand::from_opref(old_ref)])),
                invented_name: false,
                same_as_source: Some(Operand::from_opref(old_ref)),
                label_arg_idx: None,
            },
        ));
        state.short_preamble = Some(ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: Op::new(OpCode::GuardNonnull, &[Operand::from_opref(old_ref)]),
                arg_mapping: Vec::new(),
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![old_ref],
            used_boxes: vec![old_ref],
            jump_args: vec![old_ref],
            exported_state: Some(VirtualState::new(vec![VirtualStateInfo::KnownClass {
                class_ptr: old.as_usize() as i64,
            }])),
            constants,
            inputarg_infos: vec![Some(PtrInfo::Constant(old))],
        });
        state.runtime_boxes.push(old_ref);
        state.patchguardop = Some(Op::new(
            OpCode::GuardNonnull,
            &[Operand::from_opref(old_ref)],
        ));

        state.walk_const_ptr_refs_mut(&mut |slot| {
            if *slot == old {
                *slot = new;
            }
        });

        assert_eq!(state.end_args[0], new_ref);
        assert_eq!(state.next_iteration_args[0].to_opref(), new_ref);
        assert_eq!(state.renamed_inputargs[0], new_ref);
        assert_eq!(state.short_inputargs[0], new_ref);
        assert_eq!(state.runtime_boxes[0], new_ref);
        assert!(state.exported_infos.keys().any(|k| k.to_opref() == new_ref));
        assert_eq!(state.exported_short_boxes[0].op.arg(0).to_opref(), new_ref);
        assert_eq!(
            state.exported_short_boxes[0]
                .same_as_source
                .as_ref()
                .map(|b| b.to_opref()),
            Some(new_ref)
        );
        assert_eq!(state.const_short_boxes[0].op.arg(0).to_opref(), new_ref);
        let produced = state
            .short_boxes
            .iter()
            .find_map(|(key, produced)| (*key == new_ref).then_some(produced))
            .expect("short_boxes key must be forwarded");
        assert_eq!(produced.preamble_op.arg(0).to_opref(), new_ref);
        assert_eq!(
            produced.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(new_ref)
        );
        assert_eq!(
            state.patchguardop.as_ref().map(|op| op.arg(0).to_opref()),
            Some(new_ref)
        );
        match &state.virtual_state.state[0].info {
            VirtualStateInfo::Constant(Value::Ref(gcref)) => assert_eq!(*gcref, new),
            other => panic!("unexpected virtual state after walk: {other:?}"),
        }
        let short = state.short_preamble.as_ref().unwrap();
        assert_eq!(short.ops[0].op.arg(0).to_opref(), new_ref);
        assert_eq!(short.inputargs[0], new_ref);
        assert_eq!(short.used_boxes[0], new_ref);
        assert_eq!(short.jump_args[0], new_ref);
        assert_eq!(short.constants.get(&0), Some(&majit_ir::Const::Ref(new)));
    }

    #[test]
    fn test_simple_loop_peeled() {
        // A simple loop: one add op, then Jump.
        // Expected output: peeled_add, Label, original_add, Jump
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(
            result.len(),
            4,
            "expected: peeled_add, Label, original_add, Jump"
        );
        assert_eq!(result[0].opcode, OpCode::IntAdd); // peeled
        assert_eq!(result[1].opcode, OpCode::Label);
        assert_eq!(result[2].opcode, OpCode::IntAdd); // original body
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    #[test]
    fn test_peeled_ops_have_different_positions() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntSub,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 2 peeled + Label + 2 original + Jump = 6
        assert_eq!(result.len(), 6);

        // All positions should be unique.
        let positions: Vec<OpRef> = result.iter().map(|op| op.pos().get()).collect();
        for (i, pos) in positions.iter().enumerate() {
            for (j, other) in positions.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        pos, other,
                        "positions at index {} and {} should differ",
                        i, j
                    );
                }
            }
        }
    }

    #[test]
    fn test_internal_refs_remapped_in_peeled_copy() {
        // op0: v0 = IntAdd(v100, v101)  -- uses input args
        // op1: v1 = IntMul(v0, v101)    -- uses result of op0
        // Jump()
        //
        // After peeling:
        // peeled_v0 = IntAdd(v100, v101)     -- input refs unchanged
        // peeled_v1 = IntMul(peeled_v0, v101) -- v0 remapped to peeled_v0
        // Label()
        // body_v0 = IntAdd(v100, v101)
        // body_v1 = IntMul(body_v0, v101)    -- v0 remapped to body_v0
        // Jump()
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ), // references op0
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 6); // 2 peeled + Label + 2 body + Jump

        // Peeled iteration:
        let peeled_add = &result[0];
        let peeled_mul = &result[1];
        assert_eq!(peeled_add.opcode, OpCode::IntAdd);
        assert_eq!(peeled_mul.opcode, OpCode::IntMul);
        // peeled_mul should reference peeled_add's position, not original op0.
        assert_eq!(peeled_mul.arg(0).to_opref(), peeled_add.pos().get());
        // Second arg (input ref) should be unchanged.
        assert_eq!(peeled_mul.arg(1).to_opref(), OpRef::int_op(101));

        // Original body:
        let body_add = &result[3];
        let body_mul = &result[4];
        assert_eq!(body_add.opcode, OpCode::IntAdd);
        assert_eq!(body_mul.opcode, OpCode::IntMul);
        // body_mul should reference body_add's position.
        assert_eq!(body_mul.arg(0).to_opref(), body_add.pos().get());
        assert_eq!(body_mul.arg(1).to_opref(), OpRef::int_op(101));
    }

    #[test]
    fn test_external_refs_preserved() {
        // Refs to ops outside the buffer (input arguments) should not be remapped.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Peeled add should still reference v100 and v101.
        assert_eq!(result[0].arg(0).to_opref(), OpRef::int_op(100));
        assert_eq!(result[0].arg(1).to_opref(), OpRef::int_op(101));

        // Body add should also reference v100 and v101.
        assert_eq!(result[2].arg(0).to_opref(), OpRef::int_op(100));
        assert_eq!(result[2].arg(1).to_opref(), OpRef::int_op(101));
    }

    #[test]
    fn test_guards_duplicated_in_peel() {
        // Guards in the preamble serve as type checks.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 100)]),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_guard, peeled_add, Label, body_guard, body_add, Jump
        assert_eq!(result.len(), 6);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(
            guard_count, 2,
            "guard should appear in both peeled and body"
        );
    }

    #[test]
    fn test_guard_fail_args_remapped() {
        // Guards with fail_args should have those refs remapped too.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            {
                let mut guard = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 100)]);
                guard.setfailargs(vec![rooted_resop_operand(Type::Int, 0)].into()); // refs op0
                guard
            },
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Check peeled guard's fail_args.
        let peeled_guard = &result[1];
        assert_eq!(peeled_guard.opcode, OpCode::GuardTrue);
        let peeled_add_pos = result[0].pos().get();
        assert_eq!(
            peeled_guard.guard_fail_args().unwrap()[0].to_opref(),
            peeled_add_pos,
            "peeled guard's fail_args should reference peeled add"
        );

        // Check body guard's fail_args.
        let body_guard = &result[4]; // after Label (idx 3) and body_add (idx 3)
        assert_eq!(body_guard.opcode, OpCode::GuardTrue);
        let body_add_pos = result[3].pos().get();
        assert_eq!(
            body_guard.guard_fail_args().unwrap()[0].to_opref(),
            body_add_pos,
            "body guard's fail_args should reference body add"
        );
    }

    #[test]
    fn test_jump_args_remapped_to_body() {
        // Jump args should reference the body's ops, not the original positions.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]), // carries v0 (the add result)
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_add, Label, body_add, Jump
        assert_eq!(result.len(), 4);

        let jump = result.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);

        let body_add_pos = result[2].pos().get();
        assert_eq!(
            jump.arg(0).to_opref(),
            body_add_pos,
            "Jump arg should reference body add, not original"
        );
    }

    #[test]
    fn test_label_args_match_jump_args() {
        // The Label should carry the same args as the Jump.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 100),
                ],
            ),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let label = result.iter().find(|o| o.opcode == OpCode::Label).unwrap();
        let jump_args = ops.last().unwrap().getarglist_copy();
        assert_eq!(
            label
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            jump_args.iter().map(|a| a.to_opref()).collect::<Vec<_>>(),
            "Label args should match original Jump args"
        );
    }

    #[test]
    fn test_multi_op_loop() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntSub,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 2)]),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 4 peeled + Label + 4 body + Jump = 10
        assert_eq!(result.len(), 10);

        // Verify structure: peeled body, then Label, then body, then Jump.
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::IntSub);
        assert_eq!(result[2].opcode, OpCode::IntMul);
        assert_eq!(result[3].opcode, OpCode::GuardTrue);
        assert_eq!(result[4].opcode, OpCode::Label);
        assert_eq!(result[5].opcode, OpCode::IntAdd);
        assert_eq!(result[6].opcode, OpCode::IntSub);
        assert_eq!(result[7].opcode, OpCode::IntMul);
        assert_eq!(result[8].opcode, OpCode::GuardTrue);
        assert_eq!(result[9].opcode, OpCode::Jump);
    }

    #[test]
    fn test_setup_resets_state() {
        let mut pass = OptUnroll::new();

        // Simulate some state.
        pass.buffer.push(Op::new(
            OpCode::IntAdd,
            &[rooted_resop_operand(Type::Int, 0)],
        ));
        pass.seen_jump = true;

        pass.setup();

        assert!(pass.buffer.is_empty());
        assert!(!pass.seen_jump);
    }

    #[test]
    fn test_unroll_standalone_optimizer() {
        // Run the unroll pass through the optimizer infrastructure.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![majit_ir::Type::Ref; 1024]);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        // Expect: peeled_add, peeled_guard, Label, body_add, body_guard, Jump = 6
        assert_eq!(result.len(), 6);

        // All ops should have valid (non-NONE) positions.
        for op in &result {
            assert!(
                !op.pos().get().is_none(),
                "op {:?} should have a valid pos",
                op.opcode
            );
        }
    }

    #[test]
    fn test_unroll_mints_fresh_guard_descrs() {
        // RPython parity: normal trace guards do not keep their encoded descr.
        //
        // opencoder.py:391 strips guard descrs to None during trace iteration;
        // optimizer.py:724-729 invents a fresh opcode-specific descr at guard
        // emission. ResumeAtPositionDescr is only for inline_short_preamble().
        let descr = crate::compile::make_resume_guard_descr_typed(Vec::new());
        let original_index = descr.index();
        let mut ops = vec![
            Op::with_descr(
                OpCode::GuardTrue,
                &[rooted_resop_operand(Type::Int, 100)],
                descr.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let guards: Vec<&Op> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .collect();
        assert_eq!(guards.len(), 2);
        for guard in &guards {
            assert!(guard.has_descr(), "guard should have a descriptor");
            assert!(
                guard.getdescr().unwrap().is_resume_guard(),
                "optimizer.py `store_final_boxes_in_guard`: descr must be a ResumeGuardDescr subtype"
            );
        }
        let peel_index = guards[0].getdescr().unwrap().index();
        let body_index = guards[1].getdescr().unwrap().index();
        assert_ne!(peel_index, original_index);
        assert_ne!(body_index, original_index);
        assert_ne!(
            peel_index, body_index,
            "peeled and body guards must own distinct freshly invented descrs"
        );
        assert!(!guards[0].getdescr().unwrap().is_resume_at_position());
        assert!(!guards[1].getdescr().unwrap().is_resume_at_position());
    }

    #[test]
    fn test_chain_of_refs_correctly_remapped() {
        // v0 = IntAdd(v100, v101)
        // v1 = IntAdd(v0, v100)
        // v2 = IntAdd(v1, v0)
        // Jump(v2)
        //
        // In the peeled copy, all internal refs must point to peeled positions.
        // In the body copy, all internal refs must point to body positions.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 100),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 1),
                    rooted_resop_operand(Type::Int, 0),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 3 peeled + Label + 3 body + Jump = 8
        assert_eq!(result.len(), 8);

        // Peeled iteration refs:
        let p0 = result[0].pos().get();
        let p1 = result[1].pos().get();
        let _p2 = result[2].pos().get();
        assert_eq!(
            result[1].arg(0).to_opref(),
            p0,
            "peeled v1 should ref peeled v0"
        );
        assert_eq!(
            result[2].arg(0).to_opref(),
            p1,
            "peeled v2 should ref peeled v1"
        );
        assert_eq!(
            result[2].arg(1).to_opref(),
            p0,
            "peeled v2 should ref peeled v0"
        );

        // Body refs:
        let b0 = result[4].pos().get();
        let b1 = result[5].pos().get();
        let b2 = result[6].pos().get();
        assert_eq!(
            result[5].arg(0).to_opref(),
            b0,
            "body v1 should ref body v0"
        );
        assert_eq!(
            result[6].arg(0).to_opref(),
            b1,
            "body v2 should ref body v1"
        );
        assert_eq!(
            result[6].arg(1).to_opref(),
            b0,
            "body v2 should ref body v0"
        );

        // Jump should reference body v2.
        let jump = &result[7];
        assert_eq!(jump.arg(0).to_opref(), b2, "Jump should ref body v2");
    }

    #[test]
    fn test_unroll_optimizer_optimize_trace() {
        let mut unroll_opt = UnrollOptimizer::new();
        // IntAdd operates on Int-typed inputs — seed the inner phase1/2
        // optimizers' trace_inputargs via UnrollOptimizer so the
        // intbounds pass sees Int on the two inputargs.
        unroll_opt.trace_inputargs =
            majit_ir::OpRef::inputarg_refs(&[majit_ir::Type::Int, majit_ir::Type::Int]);
        // Use optimize_trace_with_constants_and_inputs to properly set
        // num_inputs so input args don't collide with op positions. Args
        // address inputarg slots via `InputArg*` OpRef variants so the
        // operand shape (`with_inputarg_types` plants the inputarg operands)
        // and the orthodox `_forwarded` mirror agree on the namespace.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_inputarg_operand(Type::Int, 0),
                    rooted_inputarg_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_inputarg_operand(Type::Int, 0)]),
        ];
        assign_positions(&mut ops, 2);
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let (result, _) =
            unroll_opt.optimize_trace_with_constants_and_inputs(&ops, &mut constants, 2);
        // The optimizer processes the trace; result should not be empty
        assert!(!result.is_empty(), "optimize_trace should produce output");
    }

    #[test]
    fn unroll_keeps_retry_source_operations_unchanged() {
        for reject in [false, true] {
            let input = rooted_inputarg_operand(Type::Int, 0);
            let mut recorded = vec![Op::new(OpCode::IntAdd, &[input.clone(), input.clone()])];
            if reject {
                recorded.push(Op::new(
                    OpCode::GuardTrue,
                    &[Operand::const_from_value(Value::Int(0))],
                ));
            }
            recorded.push(Op::new(OpCode::Jump, &[input]));
            assign_positions(&mut recorded, 1);
            let canonical: Vec<_> = recorded.into_iter().map(OpRc::new).collect();
            let source: Vec<_> = canonical.iter().map(|op| (**op).clone()).collect();
            let backup = source.clone();
            let mut unroll = UnrollOptimizer::new();
            unroll.trace_inputargs = majit_ir::OpRef::inputarg_refs(&[Type::Int]);
            unroll.phase2_input_ops_seed = Some(canonical.clone());
            let result = unroll.optimize_trace_with_constants_and_inputs_vable_out(
                &canonical,
                &mut majit_ir::ConstMap::default(),
                1,
                None,
                None,
            );
            assert_eq!(result.is_err(), reject);
            // opencoder.py TraceIterator.next creates each phase's operation;
            // neither successful peeling nor InvalidLoop writes to its source.
            for (op, saved) in source.iter().zip(&backup) {
                assert_eq!(op.opcode, saved.opcode);
                assert_eq!(op.pos().get(), saved.pos().get());
                assert_eq!(op.rd_resume_position(), saved.rd_resume_position());
                assert_eq!(op.get_value(), saved.get_value());
                assert!(matches!(
                    op.forwarded().borrow(),
                    majit_ir::forwarding::Forwarded::None
                ));
                assert_eq!(op.num_args(), saved.num_args());
                for i in 0..op.num_args() {
                    assert!(op.arg(i).same_box(&saved.arg(i)));
                }
            }
        }
    }

    #[test]
    fn test_recorded_jump_inputarg_keeps_observed_runtime_value() {
        let mut unroll_opt = UnrollOptimizer::new();
        unroll_opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&[Type::Int]);
        let inputarg_box = majit_ir::InputArg::new_int_rc(0);
        inputarg_box.set_value(Value::Int(73));
        unroll_opt.trace_inputarg_boxes = vec![inputarg_box];

        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_inputarg_operand(Type::Int, 0),
                    rooted_inputarg_operand(Type::Int, 0),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_inputarg_operand(Type::Int, 0)]),
        ];
        assign_positions(&mut ops, 1);
        let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
        let mut constants = majit_ir::ConstMap::default();
        let mut phase1_out = None;

        unroll_opt
            .optimize_trace_with_constants_and_inputs_vable_out(
                &ops,
                &mut constants,
                1,
                None,
                Some(&mut phase1_out),
            )
            .expect("tiny loop should optimize");
        let state = phase1_out.expect("Phase 1 should export loop state").1;

        assert_eq!(
            state.runtime_boxes[0],
            OpRef::const_int(73),
            "recorded closing-JUMP entry InputArg must retain observed runtime value",
        );
    }

    #[test]
    fn test_recorded_jump_resop_keeps_observed_runtime_value() {
        // closing_jump_runtime_boxes already lifts a stamped ResOp into an
        // inline Const for bridges; fold_recorded_jump_args must do the
        // same for the peeled-loop runtime_boxes channel.
        let mut add = Op::new(
            OpCode::IntAdd,
            &[
                rooted_inputarg_operand(Type::Int, 0),
                rooted_inputarg_operand(Type::Int, 0),
            ],
        );
        add.pos().set(OpRef::int_op(1));
        add.set_value(Value::Int(73));
        let folded = fold_recorded_jump_args(vec![OpRef::int_op(1)], &[], &[OpRc::new(add)]);
        assert_eq!(
            folded[0],
            OpRef::const_int(73),
            "recorded closing-JUMP ResOp must retain observed runtime value",
        );
    }

    #[test]
    fn test_unroll_optimizer_count_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::GuardNonnull, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        assert_eq!(UnrollOptimizer::count_guards(&ops), 2);
    }

    #[test]
    fn test_exported_state_reimports_widened_intbounds() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        let mut exported_bounds: crate::FxIndexMap<majit_ir::operand::Operand, IntBound> =
            crate::FxIndexMap::default();
        // Bind the export-input position at its source (a forced end-arg is a
        // bound box in production); virtualstate.py create_state
        // receives real AbstractValues.
        ctx.materialize_operand_at(OpRef::int_op(21));
        // Key by the canonical box identity the consumer resolves `int_op(21)`
        // to, so the operand-keyed lookup hits by `Rc::ptr_eq`.
        let box21 = ctx
            .get_box_replacement_operand_opt(OpRef::int_op(21))
            .expect("int_op(21) bound to a box");
        exported_bounds.insert(box21, IntBound::bounded(10, 20));

        let exported = export_state(
            &[OpRef::int_op(21)],
            &[],
            &mut optimizer,
            &mut ctx,
            Some(&exported_bounds),
        );

        assert_eq!(
            match exported
                .exported_infos
                .iter()
                .find(|(k, _)| k.to_opref() == OpRef::int_op(21))
                .map(|(_, v)| v)
                .unwrap()
            {
                crate::optimizeopt::info::OpInfo::IntBound(b) => {
                    let b = b.borrow();
                    Some((b.lower, b.upper))
                }
                _ => None,
            },
            Some((10, 20))
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Int]);
        let label_args = import_state(
            &[OpRef::input_arg_int(0)],
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        assert_eq!(label_args, vec![OpRef::int_op(21)]);
        // unroll.py:93-96: IntBound IS imported with widen() and stored
        // directly on the box's _forwarded slot via setintbound.
        // widen() relaxes bounds: lower < MININT/2 → MININT, upper > MAXINT/2 → MAXINT.
        // For [10, 20], both are within MININT/2..MAXINT/2 so widen() preserves them.
        let imported_bound = {
            // import_state forwards the Phase-2 source Box to the literal
            // carried Phase-1 Box. Follow that identity edge instead of
            // fabricating a new box from the carried box's numeric position.
            let __mb = ctx2
                .materialize_operand_at(OpRef::input_arg_int(0))
                .get_box_replacement(false);
            ctx2.getintbound_handle(&__mb).borrow().clone()
        };
        assert_eq!((imported_bound.lower, imported_bound.upper), (10, 20));
    }

    #[test]
    fn test_exported_state_reads_live_bound_for_non_jumparg() {
        // unroll.py `_expand_info` calls `self.optimizer.getinfo(arg)`
        // for EVERY exported value with no jump-arg restriction. A masked pure
        // result (`IntAnd(x, mask)`) that is a short-preamble / loop-invariant
        // value is not a jump arg, so its `[0, mask]` IntBound never enters the
        // `exported_int_bounds` side table (populated only from
        // `jump.getarglist()`). The live bound must still be exported by reading
        // it directly from the Phase-1 optimizer context.
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use crate::optimizeopt::intutils::IntBound;
        const MASK: i64 = 0xFFFF_FFFF;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        // Bind the export-input position at its source (a forced end-arg is a
        // bound box in production).
        ctx.materialize_operand_at(OpRef::int_op(21));
        let box21 = ctx
            .get_box_replacement_operand_opt(OpRef::int_op(21))
            .expect("int_op(21) bound to a box");
        // The bound lives on the box's `_forwarded` slot (as it does after
        // `IntAnd` narrows it), but it is NOT registered in the side table —
        // exactly the state of a non-jump-arg short-box result.
        ctx.setintbound(&box21, &IntBound::bounded(0, MASK));

        // `None` side table → current code has no way to recover the bound and
        // drops it; the live-bound read must surface it.
        let exported = export_state(&[OpRef::int_op(21)], &[], &mut optimizer, &mut ctx, None);

        assert_eq!(
            match exported
                .exported_infos
                .iter()
                .find(|(k, _)| k.to_opref() == OpRef::int_op(21))
                .map(|(_, v)| v)
            {
                Some(crate::optimizeopt::info::OpInfo::IntBound(b)) => {
                    let b = b.borrow();
                    Some((b.lower, b.upper))
                }
                _ => None,
            },
            Some((0, MASK)),
            "live [0, MASK] bound on a non-jump-arg short-box result must be exported"
        );
    }

    #[test]
    fn test_short_box_dependency_preserves_mask_bound_for_loop_close() {
        // RPython keeps the replay op's `_forwarded` info separate from the
        // body Box's optimizer info. A preamble-computed masked pure value can
        // therefore be used by another short-box replay op without losing the
        // live `[0, MASK]` IntBound needed by virtualstate.py:491-492 to avoid
        // loop-close re-establishment guards.
        use crate::optimizeopt::intutils::IntBound;
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};
        const MASK: i64 = 0xFFFF_FFFF;

        let masked = OpRef::int_op(58);
        let dependent = OpRef::int_op(59);
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(128, &[Type::Int; 128]);

        let mask_op = {
            let mut op = Op::new(
                OpCode::IntAnd,
                &[
                    rooted_inputarg_operand(Type::Int, 8),
                    Operand::from_opref(OpRef::const_int(MASK)),
                ],
            );
            op.pos().set(masked);
            OpRc::new(op)
        };
        let dep_op = {
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    Operand::from_bound_op(&mask_op),
                    Operand::from_opref(OpRef::const_int(2160)),
                ],
            );
            op.pos().set(dependent);
            OpRc::new(op)
        };
        let mask_box = ctx.materialize_operand_at(masked);
        let dep_box = ctx.materialize_operand_at(dependent);
        ctx.setintbound(&mask_box, &IntBound::bounded(0, MASK));
        ctx.imported_short_preamble_builder = Some(ShortPreambleBuilder::new(
            &[masked, dependent],
            &[
                (
                    mask_box.clone(),
                    ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        res: mask_box.clone(),
                        preamble_op: mask_op.clone(),
                        source_op: mask_op.clone(),
                        invented_name: false,
                        same_as_source: None,
                        label_arg_idx: Some(0),
                    },
                ),
                (
                    dep_box.clone(),
                    ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        res: dep_box.clone(),
                        preamble_op: dep_op.clone(),
                        source_op: dep_op.clone(),
                        invented_name: false,
                        same_as_source: None,
                        label_arg_idx: Some(1),
                    },
                ),
            ],
            &[OpRef::input_arg_int(8)],
        ));

        let mask_pop = crate::optimizeopt::info::PreambleOp {
            op: mask_box.clone(),
            invented_name: false,
            preamble_op: mask_op.clone(),
            same_as_source: None,
        };
        let dep_pop = crate::optimizeopt::info::PreambleOp {
            op: dep_box,
            invented_name: false,
            preamble_op: dep_op,
            same_as_source: None,
        };
        assert_eq!(ctx.force_op_from_preamble_op(&mask_pop).to_opref(), masked);
        assert_eq!(
            ctx.force_op_from_preamble_op(&dep_pop).to_opref(),
            dependent
        );

        let target_vs = VirtualState::new(vec![VirtualStateInfo::IntBounded(IntBound::bounded(
            0, MASK,
        ))]);
        let incoming_vs = crate::optimizeopt::virtualstate::export_state(&[masked], &ctx);
        let runtime_box = ctx.make_constant_int(0);
        let guard_reqs = target_vs
            .generate_guards(&incoming_vs, &[masked], &[runtime_box], &mut ctx, false)
            .expect("masked short-box state should match target loop state");
        let mut emitted = Vec::new();
        for req in &guard_reqs {
            emitted.extend(req.to_ops(&[masked], &mut ctx));
        }

        assert!(
            emitted
                .iter()
                .all(|op| !matches!(op.opcode, OpCode::IntGe | OpCode::IntLe | OpCode::GuardTrue)),
            "loop close must not emit IntBound re-establishment guards for masked short-box arg: {:?}",
            emitted.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_export_state_uses_forced_end_args_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Forced end-of-preamble args are bound boxes in production
        // (force_box_for_end_of_preamble); bind the fixture's export-input
        // position at its source so every value reaching export_single_value
        // has a canonical box (virtualstate.py create_state receives
        // real AbstractValues, never bare positions).
        ctx.materialize_operand_at(OpRef::int_op(21));
        ctx.preamble_end_args = Some(vec![OpRef::int_op(21)]);

        let exported = export_state(&[OpRef::int_op(0)], &[], &mut optimizer, &mut ctx, None);

        assert_eq!(exported.end_args.clone(), vec![OpRef::int_op(21)]);
    }

    #[test]
    fn test_exported_state_reimports_short_heap_field_facts() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        // optimizer.py ensure_ptr_info_arg0 parity: every FieldDescr
        // must resolve its parent SizeDescr so that the routing between
        // StructPtrInfo and InstancePtrInfo can ask `is_object()`.
        // SimpleFieldDescr stores the parent as Weak<DescrRef>, so the
        // test must keep the parent Arc alive until import_state has run.
        let parent = majit_ir::descr::make_size_descr(16);
        let field_descr = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(0, 0, 8, majit_ir::Type::Int, false)
                .with_signed(true)
                .with_parent_descr(parent.clone(), 0),
        ) as majit_ir::DescrRef;
        // shortpreamble.py:257/285: the export-time rename mints a fresh
        // renamed short_inputarg per label/virtual slot, and exported short
        // ops carry the renamed box in their args (not the original label
        // arg). Seed the two slot boxes and use slot 0 — the GETFIELD
        // receiver, whose original is int_op(10).
        // Bound short_inputarg boxes rooted by an index-aligned `InputArgRc`
        // pool, mirroring production (the preview pass publishes
        // `short_inputargs` and `short_inputarg_refs` together inside one
        // `PreviewShortState`); the context channel stores their positions.
        let (si0, ia0) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si1, ia1) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let source_receiver = ctx.materialize_operand_at(OpRef::int_op(10));
        publish_preview_short_state(
            &mut ctx,
            vec![si0.to_opref(), si1.to_opref()],
            vec![ia0, ia1],
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcI,
                        std::slice::from_ref(&si0),
                        field_descr.clone(),
                    );
                    op.pos().set(OpRef::int_op(11));
                    OpRc::new(op)
                },
                source_op: Some({
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcI,
                        std::slice::from_ref(&source_receiver),
                        field_descr.clone(),
                    );
                    op.pos().set(OpRef::int_op(11));
                    OpRc::new(op)
                }),
                res: rooted_resop_operand(Type::Int, 11),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            }],
        );
        // Bind export-input positions at their source: the GETFIELD receiver
        // and result are bound boxes in production (label arg / ProducedShortOp.res
        // = materialize_operand_at, `PreambleOp::add_op_to_short`).
        // virtualstate.py:711-720
        // create_state receives real AbstractValues, never bare positions.
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // The fixture references both slots via `int_op(N)` (Int variant
        // tag), so targetargs must agree under variant-aware OpRef Eq —
        // a Ref-typed targetarg would trip the Box.type cross-type
        // forward check in `make_equal_to`. Heap-field-cache mechanics are
        // exercised independent of base type.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(
            &targetargs,
            &label_args,
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        assert_eq!(label_args, vec![OpRef::int_op(10), OpRef::int_op(11)]);
        // RPython PreambleOp parity: PreambleOp stored in PtrInfo._fields.
        // No imported_short_fields for heap fields — PtrInfo is the single
        // source of truth, matching RPython's HeapOp.produce_op → opinfo.setfield.
        let obj_box = ctx2.get_box_replacement_operand_opt(targetargs[0]).unwrap();
        let pop = ctx2
            .with_ptr_info_mut(&obj_box, |info| {
                info.take_preamble_field(crate::optimizeopt::heap::OptHeap::field_slot_index(
                    &field_descr,
                ))
            })
            .flatten();
        assert!(pop.is_some(), "PreambleOp must be in PtrInfo._fields");
        let pop = pop.unwrap();
        assert_eq!(pop.op.to_opref(), OpRef::int_op(11)); // Phase 1 source — pop.op
        assert_eq!(
            pop.preamble_op.arg(0).to_opref(),
            si0.to_opref(),
            "heap replay must retain the renamed short-inputarg receiver"
        );
        drop(parent);
    }

    /// shortpreamble.py:272-280 replays the `const_short_boxes` half of
    /// `create_short_boxes` through the same `produce_op` loop as the
    /// `produced_ops` half. pyre keeps those entries on their own channel so
    /// they cannot reach `used_boxes`, so the replay is driven explicitly at
    /// the end of `import_short_preamble_state`; this covers that channel
    /// end-to-end, the way the sibling test covers `exported_short_boxes`.
    #[test]
    fn test_exported_state_reimports_const_short_heap_field_facts() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        // optimizer.py ensure_ptr_info_arg0 parity: keep the parent Arc
        // alive until import_state has run, as SimpleFieldDescr holds it Weak.
        let parent = majit_ir::descr::make_size_descr(16);
        let field_descr = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(0, 0, 8, majit_ir::Type::Int, false)
                .with_signed(true)
                .with_parent_descr(parent.clone(), 0),
        ) as majit_ir::DescrRef;
        let (si0, ia0) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si1, ia1) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let source_receiver = ctx.materialize_operand_at(OpRef::int_op(10));
        // The const half rides its own channel, so the published preview
        // carries the renamed short-inputargs and no `short_boxes` — this is
        // the one fixture where the two halves are separable.
        publish_preview_short_state(
            &mut ctx,
            vec![si0.to_opref(), si1.to_opref()],
            vec![ia0, ia1],
            Vec::new(),
        );
        // shortpreamble.py:274-279: the entry the const half carries is the
        // heap read whose result folded to a Const — `short_op.res` stays that
        // Const while `copy_and_change` mints a fresh replay position.
        ctx.exported_const_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcI,
                        std::slice::from_ref(&si0),
                        field_descr.clone(),
                    );
                    op.pos().set(OpRef::int_op(11));
                    OpRc::new(op)
                },
                source_op: Some({
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcI,
                        std::slice::from_ref(&source_receiver),
                        field_descr.clone(),
                    );
                    op.pos().set(OpRef::int_op(11));
                    OpRc::new(op)
                }),
                res: Operand::from_opref(OpRef::const_int(7)),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                // A Const result occupies no label slot: `lookup_label_arg`
                // has nothing to report for it.
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            });
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        assert_eq!(
            exported.const_short_boxes.len(),
            1,
            "the const channel must survive export"
        );

        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(
            &targetargs,
            &label_args,
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        // shortpreamble.py:72-76: the replayed entry lands in the receiver's
        // `_fields`, and for a const entry `PreambleOp(self.res, ...)` keeps
        // the Const as the body-visible Box.
        let obj_box = ctx2.get_box_replacement_operand_opt(targetargs[0]).unwrap();
        let pop = ctx2
            .with_ptr_info_mut(&obj_box, |info| {
                info.take_preamble_field(crate::optimizeopt::heap::OptHeap::field_slot_index(
                    &field_descr,
                ))
            })
            .flatten();
        assert!(
            pop.is_some(),
            "const short box must reach PtrInfo._fields on import"
        );
        let pop = pop.unwrap();
        assert_eq!(pop.op.to_opref(), OpRef::const_int(7));
        // The const entries are not builder keys, so the replay is the
        // exported `preamble_op` itself: its receiver is the renamed
        // `short_inputargs[0]` that `produce_arg` returned at export, the box
        // `inline_short_preamble` maps to the jump arg.
        assert_eq!(
            pop.preamble_op.arg(0).to_opref(),
            exported.short_inputargs[0]
        );
        drop(parent);
    }

    #[test]
    fn test_import_state_reimports_short_ref_constant_identity() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        let ptr = GcRef(0x1234_5678);
        let field_descr = majit_ir::descr::make_field_descr_full(88, 0, 8, Type::Int, true);
        // ConstPtr.value inline (history.py): the producer seeds the
        // inline variant that carries the pointer directly; the consumer
        // reads it back without any pool lookup.
        let ptr_box = ctx.materialize_operand_at(OpRef::const_ptr(ptr));
        ctx.seed_constant(&ptr_box, Value::Ref(ptr));
        // Two Int label args (`int_op(12)` / `int_op(11)`) and no virtuals, so
        // the preview's `create_short_inputargs` mints two Int renames.
        let (short_inputargs, short_inputarg_refs) =
            mint_short_inputargs(&mut ctx, &[Type::Int, Type::Int]);
        // `produce_pure` copies `res.bound_op()` args. That producer is the
        // Phase-1 getfield, whose receiver is the const pointer.
        let const_arg = Operand::from_opref(OpRef::const_ptr(ptr));
        let getfield = {
            let mut op = Op::with_descr(
                OpCode::GetfieldGcI,
                &[const_arg.clone()],
                field_descr.clone(),
            );
            op.pos().set(OpRef::int_op(11));
            OpRc::new(op)
        };
        publish_preview_short_state(
            &mut ctx,
            short_inputargs,
            short_inputarg_refs,
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                op: getfield.clone(),
                source_op: None,
                res: Operand::from_bound_op(&getfield),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            }],
        );
        // Bind export-input positions at their source (label arg /
        // ProducedShortOp.res = materialize_operand_at,
        // `PreambleOp::add_op_to_short`);
        // virtualstate.py create_state receives real AbstractValues.
        ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(12), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // Both label args resolve to Int op slots (12 / 11 = GetfieldGcI base / result).
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(
            &targetargs,
            &label_args,
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        assert_eq!(label_args, vec![OpRef::int_op(12), OpRef::int_op(11)]);
        // history.py ConstPtr.value inline: the imported constant
        // lands at `OpRef::ConstPtr(ptr)`, carrying the pointer
        // directly.
        let fresh_const = OpRef::const_ptr(ptr);
        assert_eq!(ctx2.get_constant(fresh_const), Some(Value::Ref(ptr)));
        // A Pure short box gets a fresh replay box (`alloc_op_position_typed`).
        // `label_arg_idx` does not reuse `short_args[slot]`.
        assert_eq!(ctx2.imported_short_pure_ops.len(), 1);
        let result = ctx2.imported_short_pure_ops[0].result;
        assert_ne!(result, OpRef::int_op(11));
        let expected = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx2,
            OpCode::GetfieldGcI,
            Some(field_descr.clone()),
            vec![const_arg],
            result,
            OpRef::int_op(11),
            false,
            None,
        );
        assert_eq!(ctx2.imported_short_pure_ops, vec![expected]);
    }

    #[test]
    fn test_import_short_loopinvariant_uses_producer_const_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        // ConstInt.value inline (history.py): the producer seeds the
        // inline variant carrying the func address directly; the consumer
        // reads it back without any pool lookup.
        let func_ptr = 0xCAFE;
        let func = OpRef::const_int(func_ptr);
        let func_box = ctx.materialize_operand_at(func);
        ctx.seed_constant(&func_box, Value::Int(func_ptr));
        // Two Int label args (`int_op(10)` / `int_op(11)`) and no virtuals, so
        // the preview's `create_short_inputargs` mints two Int renames.
        let (short_inputargs, short_inputarg_refs) =
            mint_short_inputargs(&mut ctx, &[Type::Int, Type::Int]);
        let call = {
            let mut op = Op::new(OpCode::CallLoopinvariantI, &[Operand::from_opref(func)]);
            op.pos().set(OpRef::int_op(11));
            OpRc::new(op)
        };
        publish_preview_short_state(
            &mut ctx,
            short_inputargs,
            short_inputarg_refs,
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                // `LoopInvariantOp.produce_op` reads `self.res.getarg(0)`.
                res: Operand::from_bound_op(&call),
                op: call,
                source_op: None,
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            }],
        );

        // Bind export-input positions at their source (label arg /
        // ProducedShortOp.res = materialize_operand_at,
        // `PreambleOp::add_op_to_short`);
        // virtualstate.py create_state receives real AbstractValues.
        ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // CallLoopinvariantI produces an Int at int_op(11); slot 10 is the
        // Int func-address constant base, so both label args are Int.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(
            &targetargs,
            &label_args,
            &exported,
            &mut optimizer,
            &mut ctx2,
        );

        assert_eq!(
            ctx2.imported_loop_invariant_results
                .iter()
                .find(|(k, _)| *k == func_ptr)
                .map(|(_, v)| *v),
            Some(OpRef::int_op(11))
        );
        // history.py ConstInt.value inline — the imported func
        // address lands at `OpRef::ConstInt(func_ptr)`, which
        // carries the value directly.
        assert_eq!(
            ctx2.get_constant(OpRef::const_int(func_ptr)),
            Some(Value::Int(func_ptr))
        );
    }

    #[test]
    fn test_import_short_loopinvariant_result_uses_short_inputarg_slot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        // history.py:227 — the CallLoopinvariant func address is an inline
        // `ConstInt` arg; the value rides on the OpRef and the import path
        // keys `imported_loop_invariant_results` by that value.
        let func_ptr = 0xBEEF;
        let func = OpRef::const_int(func_ptr);
        let source = OpRef::int_op(11);
        let source_box = rooted_resop_operand(Type::Int, 11);
        let phase2_result = OpRef::int_op(3);
        let call = {
            let mut op = Op::new(OpCode::CallLoopinvariantI, &[Operand::from_opref(func)]);
            op.pos().set(source);
            OpRc::new(op)
        };
        let exported = ExportedState::new(
            vec![source],
            vec![0],
            vec![source_box.clone()],
            crate::optimizeopt::virtualstate::VirtualState::new(Vec::new()),
            indexmap::IndexMap::new(),
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                // `LoopInvariantOp.produce_op` reads `self.res.getarg(0)`.
                res: Operand::from_bound_op(&call),
                op: call,
                source_op: None,
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant,
                label_arg_idx: Some(0),
                invented_name: false,
                same_as_source: None,
            }],
            Vec::new(),
            Vec::new(),
            vec![source_box.to_opref()],
            // short_inputarg_refs: this fixture stores plain OpRef positions.
            Vec::new(),
        );
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            8,
            &[Type::Int, Type::Int, Type::Int, Type::Int],
        );
        let func_box = ctx.materialize_operand_at(func);
        ctx.seed_constant(&func_box, Value::Int(func_ptr));

        import_short_preamble_state(
            &[OpRef::int_op(0)],
            &[phase2_result],
            &exported,
            &mut optimizer,
            &mut ctx,
        );

        // imported_loop_invariant_results stores the
        // Phase 1 source directly (RPython `shortpreamble.py:120 op = self.res`).
        let _ = phase2_result;
        assert_eq!(
            ctx.imported_loop_invariant_results
                .iter()
                .find(|(k, _)| *k == func_ptr)
                .map(|(_, v)| *v),
            Some(source)
        );
    }

    /// Invariant: `force_op_from_preamble_op` follows RPython
    /// `unroll.py:26-39` line-by-line. It calls `use_box` (updates
    /// `self.short`) and seeds `potential_extra_ops`. It does NOT touch
    /// `used_boxes` / `short_preamble_jump`. Those grow only via the
    /// orthodox `force_box` → `potential_extra_ops.pop` →
    /// `add_preamble_op` (`shortpreamble.py`) path.
    #[test]
    fn test_force_op_from_preamble_orthodox_does_not_record_used_boxes() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            10,
            &[Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)],
            &[OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(
                        OpCode::IntAdd,
                        &[
                            rooted_resop_operand(Type::Int, 0),
                            rooted_resop_operand(Type::Int, 1),
                        ],
                    );
                    op.pos().set(OpRef::int_op(20));
                    OpRc::new(op)
                },
                source_op: None,
                res: rooted_resop_operand(Type::Int, 20),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );

        let src20 = ctx.materialize_operand_at(OpRef::int_op(20));
        let produced = ctx
            .imported_short_preamble_builder
            .as_ref()
            .unwrap()
            .produced_short_op(&src20)
            .unwrap();
        let pop = crate::optimizeopt::info::PreambleOp {
            op: src20.clone(),
            invented_name: produced.invented_name,
            same_as_source: produced.same_as_source.clone(),
            preamble_op: produced.preamble_op,
        };
        let forced = ctx.force_op_from_preamble_op(&pop).to_opref();
        assert_eq!(forced, OpRef::int_op(20));

        // RPython `unroll.py:32` `use_box` populates `self.short`.
        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        // RPython `unroll.py:34-37` seeds `potential_extra_ops` so a later
        // `force_box` will run `add_preamble_op` (shortpreamble.py).
        assert!(
            ctx.has_potential_extra_op(&src20),
            "force_op_from_preamble_op must seed potential_extra_ops"
        );
        // RPython parity: `force_op_from_preamble` does NOT call
        // `add_preamble_op`. `used_boxes` / `short_preamble_jump` /
        // `extra_same_as` stay empty until `force_box` pops the entry.
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(&src20, &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        // After force_box: orthodox `add_preamble_op` (shortpreamble.py)
        // populated all three lists in lock-step.
        assert_eq!(sp.used_boxes.clone(), vec![OpRef::int_op(20)]);
        assert_eq!(sp.jump_args.clone(), vec![OpRef::int_op(20)]);
        assert!(
            !ctx.has_potential_extra_op(&src20),
            "force_box must consume the potential_extra_ops entry"
        );
    }

    /// Invariant (Heap variant): same orthodox boundary as the Pure
    /// case, with the producer's `make_equal_to(source, value)` forwarding
    /// installed (`shortpreamble.rs::produce_heap_field`).
    /// `force_op_from_preamble_op` returns `preamble_source` (RPython
    /// `unroll.py UnrollOptimizer.force_op_from_preamble return preamble_op.op` ≡ `self.res`); the producer's
    /// `get_box_replacement(source)` lookup is consumed inside `force_box`
    /// (`shortpreamble.py:436 op = preamble_op.op.get_box_replacement()`)
    /// when it runs `add_preamble_op`.
    #[test]
    fn test_force_op_from_preamble_returns_source_for_heap_variant() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            32,
            &[Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.initialize_imported_short_preamble_builder(
            &[
                OpRef::ref_op(0),
                OpRef::ref_op(1),
                OpRef::ref_op(2),
                OpRef::ref_op(3),
            ],
            &[
                OpRef::ref_op(10),
                OpRef::ref_op(11),
                OpRef::ref_op(12),
                OpRef::ref_op(13),
            ],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcR,
                        &[rooted_resop_operand(Type::Ref, 3)],
                        majit_ir::descr::make_field_descr_full(56, 0, 8, Type::Ref, false),
                    );
                    op.pos().set(OpRef::ref_op(19));
                    OpRc::new(op)
                },
                source_op: None,
                res: rooted_resop_operand(Type::Ref, 19),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        let src19 = ctx.materialize_operand_at(OpRef::ref_op(19));
        let produced = ctx
            .imported_short_preamble_builder
            .as_ref()
            .unwrap()
            .produced_short_op(&src19)
            .unwrap();
        let b_src = ctx.materialize_operand_at(OpRef::ref_op(19));
        let pop = crate::optimizeopt::info::PreambleOp {
            op: b_src.clone(),
            invented_name: produced.invented_name,
            same_as_source: produced.same_as_source.clone(),
            preamble_op: produced.preamble_op,
        };
        let forced = ctx.force_op_from_preamble_op(&pop).to_opref();
        // RPython `unroll.py UnrollOptimizer.force_op_from_preamble return preamble_op.op` ≡ self.res.
        // pyre's Phase 1 source IS self.res for the imported short box.
        assert_eq!(forced, OpRef::ref_op(19));

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        // RPython parity: orthodox boundary — used_boxes / jump_args stay
        // empty until force_box runs add_preamble_op.
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());
        // `force_op_from_preamble_op` keys potential_extra_ops by
        // preamble_op.op == short_op.res, the exact body-visible Box.
        assert!(
            ctx.has_potential_extra_op(&b_src),
            "force_op_from_preamble_op must seed potential_extra_ops by the body-visible box"
        );
        let slot14 = ctx.materialize_operand_at(OpRef::ref_op(14));
        assert!(
            !ctx.has_potential_extra_op(&slot14),
            "the unrelated replay/body slot must not carry the potential_extra_ops entry"
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(&b_src, &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        // shortpreamble.py:436 `op = preamble_op.op.get_box_replacement()`.
        assert_eq!(sp.used_boxes.clone(), vec![OpRef::ref_op(19)]);
        assert_eq!(sp.jump_args.clone(), vec![OpRef::ref_op(19)]);
        assert!(
            !ctx.has_potential_extra_op(&b_src),
            "force_box must consume the potential_extra_ops entry"
        );
    }

    #[test]
    fn test_exported_state_reimports_invented_short_alias_metadata() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(6, 0);
        // shortpreamble.py:257/285: exported short ops carry the renamed
        // short_inputargs in their args. Seed three slot boxes (label args
        // 12/13/14) and rename the IntAdd operands to slots 0/1. The
        // `same_as_source` alias is a ProducedShortOp field, not an op arg,
        // so it keeps its original (int_op(14)).
        // Bound short_inputarg boxes rooted by an index-aligned `InputArgRc`
        // pool, matching production (the preview pass publishes
        // `short_inputargs` and `short_inputarg_refs` together inside one
        // `PreviewShortState`); the context channel stores their positions.
        let (si0, ia0) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si1, ia1) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si2, ia2) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        publish_preview_short_state(
            &mut ctx,
            vec![si0.to_opref(), si1.to_opref(), si2.to_opref()],
            vec![ia0, ia1, ia2],
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[si0.clone(), si1.clone()]);
                    op.pos().set(OpRef::int_op(30));
                    OpRc::new(op)
                },
                source_op: None,
                res: rooted_resop_operand(Type::Int, 30),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: true,
                same_as_source: Some(rooted_resop_operand(Type::Int, 14)),
            }],
        );
        // Bind export-input positions at their source (IntAdd operands /
        // same-as alias are bound boxes in production); virtualstate.py:711-720
        // create_state receives real AbstractValues, never bare positions.
        ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.materialize_operand_at(OpRef::int_op(13));
        ctx.materialize_operand_at(OpRef::int_op(14));

        let exported = export_state(
            &[OpRef::int_op(12), OpRef::int_op(13), OpRef::int_op(14)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // IntAdd producer at int_op(30) consumes Int op slots, so the
        // imported targetargs must be Int-typed too — generous pool keeps
        // typed `OpRef::input_arg_int(K)` handles for any K the test walks.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(6, &vec![Type::Int; 1024]);
        let targetargs = [
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_int(2),
        ];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(
            &targetargs,
            &label_args,
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        let imported_result = ctx2.imported_short_pure_ops[0].result;
        assert_ne!(imported_result, OpRef::int_op(30));
        let pop = ctx2.imported_short_pure_ops[0].pop.clone();
        let forced = ctx2.force_op_from_preamble_op(&pop).to_opref();
        // force_op_from_preamble may return the imported position (not necessarily 30)
        assert_eq!(ctx2.imported_short_pure_ops.len(), 1);
        // RPython parity: extra_same_as is populated lazily by add_preamble_op
        // (called from optimizer.force_box's potential_extra_ops.pop path).
        // shortpreamble.py record the SameAs Box at use-box time, not
        // at produce_op time — verify the lazy population fires.
        let forced_box = ctx2
            .get_box_replacement_operand_opt(forced)
            .unwrap_or_else(|| Operand::bound_from_opref(forced));
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(&forced_box, &mut ctx2);
        let aliases = ctx2.used_imported_short_aliases();
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].same_as_source.to_opref(), OpRef::int_op(14));
        assert_eq!(aliases[0].same_as_opcode, OpCode::SameAsI);
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_snapshot_every_body_arglist() {
        // compile.py never builds a Vec of every body's args. Nine
        // setfields + a JUMP (regex-shaped) must still assemble.
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(3));
            op
        }];
        let mut p2_ops: Vec<Op> = (0..9)
            .map(|i| {
                let mut op = Op::new(
                    OpCode::SetfieldGc,
                    &[
                        rooted_resop_operand(Type::Ref, 10),
                        rooted_resop_operand(Type::Int, 3 + i),
                    ],
                );
                op.pos().set(OpRef::void_op(20 + i as u32));
                op
            })
            .collect();
        p2_ops.push(Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 3)]));
        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(3)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &majit_ir::ConstMap::default(),
            None,
            None,
        );
        let setfields = combined
            .iter()
            .filter(|op| op.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(setfields, 9, "all body setfields must survive assembly");
        assert!(
            combined.iter().any(|op| op.opcode == OpCode::Jump),
            "the closing JUMP must survive without a per-op arglist Vec"
        );
    }

    #[test]
    fn test_assemble_peeled_trace_emits_extra_same_as_before_label() {
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntMul,
                    &[
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 0),
                    ],
                );
                op.pos().set(OpRef::int_op(1));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 50)]),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::int_op(50),
                same_as_source: rooted_resop_operand(Type::Int, 10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &majit_ir::ConstMap::default(),
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].opcode, OpCode::SameAsI);
        assert_eq!(
            combined[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10)]
        );
        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[3].opcode, OpCode::IntMul);
        assert_eq!(combined[3].arg(0).to_opref(), combined[1].pos().get());
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].arg(0).to_opref(), combined[1].pos().get());
    }

    #[test]
    fn test_assemble_peeled_trace_emits_same_as_for_remapped_label_arg() {
        // The LABEL contract is a Phase-2 forwarded box (50), not the
        // preamble producer (3). Fall-through only has 3; emit SameAs.
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 0),
                    ],
                );
                op.pos().set(OpRef::int_op(4));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 4)]),
        ];
        let p1_ops_rc: Vec<majit_ir::OpRc> = p1_ops
            .iter()
            .map(|op| majit_ir::OpRc::new(op.clone()))
            .collect();
        let p2_ops_rc: Vec<majit_ir::OpRc> = p2_ops
            .iter()
            .map(|op| majit_ir::OpRc::new(op.clone()))
            .collect();
        let mut ctx = assemble_test_context(&p1_ops, &p2_ops, 1);
        let src = ctx.materialize_operand_at(OpRef::int_op(3));
        let dst = ctx.materialize_operand_at(OpRef::int_op(50));
        ctx.make_equal_to(&src, &dst);

        let combined = assemble_peeled_trace_with_jump_args(
            &p1_ops_rc,
            &p2_ops_rc,
            &[OpRef::int_op(50)],
            &[OpRef::int_op(0)],
            &[],
            &[],
            1,
            0,
            true,
            &[],
            &majit_ir::ConstMap::default(),
            None,
            None,
            &[],
            &mut Vec::new(),
            &mut ctx,
        );

        assert_eq!(combined[0].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].opcode, OpCode::SameAsI);
        assert_eq!(combined[1].pos().get(), OpRef::int_op(50));
        assert_eq!(
            combined[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(3)]
        );
        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[2].arg(0).to_opref(), OpRef::int_op(50));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_eq!(combined[3].arg(0).to_opref(), OpRef::int_op(50));
    }

    #[test]
    fn test_assemble_peeled_trace_skips_same_as_from_a_phase2_inputarg() {
        // A forwarded Phase-2 inputarg slot is not in the assembled
        // preamble. SameAs from it would hand dynasm an InputArgRef
        // the compiled loop never received.
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![Op::new(
            OpCode::Jump,
            &[rooted_resop_operand(Type::Int, 50)],
        )];
        let p1_ops_rc: Vec<majit_ir::OpRc> = p1_ops
            .iter()
            .map(|op| majit_ir::OpRc::new(op.clone()))
            .collect();
        let p2_ops_rc: Vec<majit_ir::OpRc> = p2_ops
            .iter()
            .map(|op| majit_ir::OpRc::new(op.clone()))
            .collect();
        let mut ctx = assemble_test_context(&p1_ops, &p2_ops, 1);
        let src = ctx.materialize_operand_at(OpRef::input_arg_int(100));
        let dst = ctx.materialize_operand_at(OpRef::int_op(50));
        ctx.make_equal_to(&src, &dst);

        let combined = assemble_peeled_trace_with_jump_args(
            &p1_ops_rc,
            &p2_ops_rc,
            &[OpRef::int_op(50)],
            &[OpRef::int_op(0)],
            &[],
            &[],
            1,
            100,
            true,
            &[],
            &majit_ir::ConstMap::default(),
            None,
            None,
            &[],
            &mut Vec::new(),
            &mut ctx,
        );

        assert!(
            combined.iter().all(|op| !matches!(
                op.opcode,
                OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
            )),
            "SameAs must not be sourced from a Phase-2 inputarg absent from the preamble stream"
        );
        assert_eq!(combined[1].opcode, OpCode::Label);
    }

    #[test]
    fn test_assemble_peeled_trace_resolves_residual_label_refop_to_inputarg() {
        // compile.py emit_op / get_box_replacement: a reminted residual
        // slot on the body LABEL is the same Box as the expanded inputarg.
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![Op::new(
            OpCode::Jump,
            &[rooted_resop_operand(Type::Ref, 85)],
        )];
        let p1_ops_rc: Vec<majit_ir::OpRc> = p1_ops
            .iter()
            .map(|op| majit_ir::OpRc::new(op.clone()))
            .collect();
        let p2_ops_rc: Vec<majit_ir::OpRc> = p2_ops
            .iter()
            .map(|op| majit_ir::OpRc::new(op.clone()))
            .collect();
        let mut ctx = assemble_test_context(&p1_ops, &p2_ops, 1);
        let reminted = ctx.materialize_operand_at(OpRef::ref_op(85));
        let input = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.make_equal_to(&reminted, &input);

        let combined = assemble_peeled_trace_with_jump_args(
            &p1_ops_rc,
            &p2_ops_rc,
            &[OpRef::ref_op(85)],
            &[OpRef::input_arg_ref(0)],
            &[],
            &[],
            1,
            0,
            true,
            &[],
            &majit_ir::ConstMap::default(),
            None,
            None,
            &[],
            &mut Vec::new(),
            &mut ctx,
        );

        let label = combined
            .iter()
            .find(|op| op.opcode == OpCode::Label)
            .expect("assembled peel has a body LABEL");
        assert_eq!(
            label
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(0)],
            "residual LABEL RefOp must recover the forwarded InputArg identity"
        );
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_label_arg_until_body_redef() {
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(11));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntGe,
                    &[
                        rooted_resop_operand(Type::Int, 11),
                        Operand::from_opref(OpRef::const_int(2)),
                    ],
                );
                op.pos().set(OpRef::int_op(4));
                op
            },
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 11),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos().set(OpRef::int_op(11));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 11)]),
        ];

        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(11)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(
            combined[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(11)]
        );
        assert_eq!(combined[2].opcode, OpCode::IntGe);
        assert_eq!(combined[2].arg(0).to_opref(), OpRef::int_op(11));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_eq!(combined[3].arg(0).to_opref(), OpRef::int_op(11));
        assert_ne!(combined[3].pos().get(), OpRef::int_op(11));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].arg(0).to_opref(), combined[3].pos().get());
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_preamble_box_over_body_collision() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::GetfieldGcR, &[rooted_resop_operand(Type::Int, 3)]);
            op.pos().set(OpRef::int_op(19));
            op.setdescr(majit_ir::descr::make_field_descr_full(
                56,
                0,
                8,
                Type::Ref,
                false,
            ));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::SetfieldGc,
                    &[
                        rooted_resop_operand(Type::Int, 25),
                        rooted_resop_operand(Type::Int, 19),
                    ],
                );
                op.setdescr(majit_ir::descr::make_field_descr_full(
                    57,
                    8,
                    8,
                    Type::Ref,
                    false,
                ));
                op
            },
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos().set(OpRef::int_op(19));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 19)]),
        ];

        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(0)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].arg(1).to_opref(), OpRef::int_op(19));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_ne!(combined[3].pos().get(), OpRef::int_op(19));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].arg(0).to_opref(), combined[3].pos().get());
    }

    #[test]
    fn test_assemble_peeled_trace_extends_label_with_used_boxes() {
        // Production caller passes body ops whose args have already been
        // resolved through ctx.get_box_replacement (`optimizer.rs`'s
        // `propagate_from_pass_range` forwarding-resolve pass). We mirror that here by writing the
        // resolved label_arg (OpRef::int_op(10)) directly into the body op rather
        // than the raw inputarg position OpRef::int_op(0).
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos().set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntMul,
                    &[
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 10),
                    ],
                );
                op.pos().set(OpRef::int_op(1));
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 10),
                    rooted_resop_operand(Type::Int, 50),
                ],
            ),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[OpRef::int_op(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::int_op(50),
                same_as_source: rooted_resop_operand(Type::Int, 10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &majit_ir::ConstMap::default(),
            None,
            None,
        );

        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(
            combined[2]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), combined[1].pos().get()]
        );
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(
            combined[4]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), combined[1].pos().get()]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_keeps_used_box_with_inline_constant_operand() {
        // assemble_peeled_trace must preserve an inline-Const operand
        // (history.py `ConstInt.value`) untouched as the GuardValue
        // immediate, and the box-namespace entries in the constants
        // snapshot must round-trip unchanged.
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::SameAsI, &[rooted_resop_operand(Type::Int, 37)]);
            op.pos().set(OpRef::void_op(857));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::GuardValue,
                    &[
                        rooted_resop_operand(Type::Void, 857),
                        Operand::from_opref(OpRef::const_int(2)),
                    ],
                );
                op.setfailargs(vec![rooted_resop_operand(Type::Void, 857)].into());
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 10),
                    rooted_resop_operand(Type::Int, 853),
                    rooted_resop_operand(Type::Void, 857),
                    rooted_resop_operand(Type::Int, 850),
                ],
            ),
        ];
        let constants =
            majit_ir::ConstMap::from_iter([(OpRef::void_op(857).raw(), majit_ir::Value::Int(2))]);

        let mut ctx = assemble_test_context(&p1_ops, &p2_ops, 1);
        let p1_ops_rc: Vec<majit_ir::OpRc> =
            p1_ops.iter().map(|op| OpRc::new(op.clone())).collect();
        let p2_ops_rc: Vec<majit_ir::OpRc> =
            p2_ops.iter().map(|op| OpRc::new(op.clone())).collect();
        let combined = assemble_peeled_trace_with_jump_args(
            &p1_ops_rc,
            &p2_ops_rc,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[OpRef::int_op(22), OpRef::void_op(857), OpRef::int_op(150)],
            &[OpRef::int_op(853), OpRef::void_op(857), OpRef::int_op(850)],
            1,
            0,
            true,
            &[],
            &constants,
            None,
            None,
            &[],
            &mut Vec::new(),
            &mut ctx,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(
            combined[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[
                OpRef::int_op(10),
                OpRef::int_op(22),
                OpRef::void_op(857),
                OpRef::int_op(150)
            ]
        );
        assert_eq!(combined[2].opcode, OpCode::GuardValue);
        assert_eq!(
            combined[2]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::void_op(857), OpRef::const_int(2)]
        );
        assert_eq!(combined[3].opcode, OpCode::Jump);
        assert_eq!(
            combined[3]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[
                OpRef::int_op(10),
                OpRef::int_op(22),
                OpRef::void_op(857),
                OpRef::int_op(150)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_remaps_extra_label_source_slots() {
        // Production caller pre-resolves body args through
        // ctx.get_box_replacement; OpRef::int_op(0) (Phase 2 inputarg slot) is
        // pre-replaced with OpRef::int_op(10) (the corresponding label_arg).
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GetfieldGcI, &[rooted_resop_operand(Type::Int, 50)]);
                op.pos().set(OpRef::int_op(1));
                op.setdescr(majit_ir::descr::make_field_descr_full(
                    0,
                    0,
                    8,
                    majit_ir::Type::Int,
                    true,
                ));
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 10),
                    rooted_resop_operand(Type::Int, 50),
                ],
            ),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[OpRef::int_op(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::int_op(50),
                same_as_source: rooted_resop_operand(Type::Int, 10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &majit_ir::ConstMap::default(),
            None,
            None,
        );

        let label_idx = combined
            .iter()
            .position(|op| op.opcode == OpCode::Label)
            .expect("label");
        let label = &combined[label_idx];
        let extra_label_arg = label.arg(1);
        assert_eq!(
            label
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), extra_label_arg.to_opref()]
        );
        let body_getfield = &combined[label_idx + 1];
        assert_eq!(body_getfield.opcode, OpCode::GetfieldGcI);
        assert_eq!(
            body_getfield
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![extra_label_arg.to_opref()]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_carries_body_value_used_before_local_def() {
        // Production caller pre-resolves body args through
        // ctx.get_box_replacement (`optimizer.rs`'s
        // `propagate_from_pass_range`), so OpRef::int_op(0) (Phase 2 inputarg slot) would already be replaced
        // with OpRef::int_op(10) (the label_arg for that slot) by the time
        // it reaches the assembler. We mirror that here.
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 64)]);
                op.setfailargs(vec![rooted_resop_operand(Type::Int, 64)].into());
                op
            },
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 10),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos().set(OpRef::int_op(64));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 64)]),
        ];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), OpRef::int_op(64)]
        );
        assert_eq!(combined[1].opcode, OpCode::GuardTrue);
        assert_eq!(
            combined[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(64)]
        );
        assert_eq!(
            combined[1]
                .guard_fail_args()
                .expect("guard fail args")
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(64)]
        );
        assert_eq!(combined[2].opcode, OpCode::IntAdd);
        assert_ne!(combined[2].pos().get(), OpRef::int_op(64));
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_route_jump_by_preamble_descr() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let p2_ops = vec![{
            let mut jump = Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                    rooted_resop_operand(Type::Int, 2),
                    rooted_resop_operand(Type::Int, 3),
                    rooted_resop_operand(Type::Int, 4),
                ],
            );
            jump.setdescr(start_descr.clone());
            jump
        }];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[
                OpRef::int_op(10),
                OpRef::int_op(11),
                OpRef::int_op(12),
                OpRef::int_op(13),
                OpRef::int_op(14),
            ],
            &[OpRef::int_op(100), OpRef::int_op(101), OpRef::int_op(102)],
            &[OpRef::int_op(13), OpRef::int_op(14)],
            5,
            false,
            &[],
            &majit_ir::ConstMap::default(),
            Some(start_descr),
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::Jump);
        // RPython parity: unroll.py `jump_to_preamble` retargets the
        // Jump before compile.py assembles the trace, and this assembly helper
        // must not derive a different arg contract by inspecting that descr.
        // The caller is responsible for constructing the preamble-shaped Jump.
        assert_eq!(
            combined[2]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[
                OpRef::int_op(100),
                OpRef::int_op(101),
                OpRef::int_op(102),
                OpRef::int_op(3),
                OpRef::int_op(4)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_preamble_jump_arity_when_body_label_is_shorter() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                op.pos().set(OpRef::int_op(2));
                op
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                jump.setdescr(start_descr.clone());
                jump
            },
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(0)],
            &[OpRef::int_op(100), OpRef::int_op(101)],
            &[],
            2,
            false,
            &[],
            &majit_ir::ConstMap::default(),
            Some(start_descr),
            Some(loop_descr),
        );

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(100), OpRef::int_op(101)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_extend_body_label_with_preamble_jump_only_args() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 10),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos().set(OpRef::int_op(20));
                op
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        rooted_resop_operand(Type::Int, 10),
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 60),
                    ],
                );
                jump.setdescr(start_descr.clone());
                jump
            },
        ];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(100), OpRef::int_op(101), OpRef::int_op(102)],
            &[],
            1,
            false,
            &[],
            &constants,
            Some(start_descr),
            Some(loop_descr),
        );

        let body_label = combined
            .iter()
            .find(|op| {
                op.opcode == OpCode::Label
                    && op.getdescr().map(|descr| descr.repr()) == Some("TargetToken(1)".to_string())
            })
            .expect("body label");
        assert_eq!(
            body_label
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10)]
        );

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), OpRef::int_op(50), OpRef::int_op(60)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_slots_for_new_body_positions() {
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::New, &[]);
                op.pos().set(OpRef::ref_op(1));
                op
            },
            Op::new(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 1),
                    rooted_resop_operand(Type::Int, 0),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        let constants = majit_ir::ConstMap::from_iter([
            (2_u32, majit_ir::Value::Int(606)),
            (4_u32, majit_ir::Value::Int(611)),
        ]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::New);
        assert_ne!(combined[1].pos().get(), OpRef::int_op(2));
        assert_ne!(combined[1].pos().get(), OpRef::int_op(4));
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].arg(0).to_opref(), combined[1].pos().get());
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_extra_label_args() {
        // Production caller pre-resolves the Jump's body inputarg ref
        // (OpRef::int_op(0)) to label_args[0] = OpRef::int_op(10) before it reaches the
        // assembler.
        //
        // `extra_label_args` holds one literal Const entry and one Box.
        // The filter keeps Boxes (they still back runtime slots) and drops
        // literal Const entries (their value is emitted inline at use
        // sites). Mirrors `assemble_peeled_trace_with_jump_args`'s
        // `label_arg.is_constant()` predicate.
        let const_extra = OpRef::const_int(606);
        let p2_ops = vec![Op::new(
            OpCode::Jump,
            &[rooted_resop_operand(Type::Int, 10)],
        )];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[const_extra, OpRef::int_op(8)],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), OpRef::int_op(8)]
        );
    }

    #[test]
    fn loop_label_and_bridge_jump_drop_the_same_constant_extra() {
        let extras = [
            OpRef::const_int(16),
            OpRef::int_op(8),
            OpRef::const_int(0),
            OpRef::int_op(9),
        ];
        let base = [OpRef::int_op(10)];
        let p2_ops = vec![Op::new(
            OpCode::Jump,
            &[rooted_resop_operand(Type::Int, 10)],
        )];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &base,
            &[OpRef::int_op(0)],
            &extras,
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );
        let label_args: Vec<_> = combined[0]
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect();
        let mut jump_args = base.to_vec();
        jump_args.extend(
            extras
                .into_iter()
                .filter(|arg| short_extra_is_carried(*arg)),
        );
        assert_eq!(
            jump_args, label_args,
            "a bridge JUMP and its loop LABEL carry the same boxes"
        );
    }

    #[test]
    fn test_assemble_peeled_trace_passes_through_resolved_body_inputs() {
        // Production caller pre-resolves Phase 2 body op args through
        // ctx.get_box_replacement (`optimizer.rs`'s
        // `propagate_from_pass_range`), so by the time
        // they reach assemble_peeled_trace they already match label_args.
        // The assembler is therefore a passthrough for inputarg references
        // — no source_slot input_remap needed. This test verifies that
        // pre-resolved body args survive intact.
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 200),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos().set(OpRef::int_op(20));
                op
            },
            {
                // The JUMP closes onto this trace's own LABEL, which is what
                // makes the arity coercion below applicable.
                let mut jump = Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 200)]);
                jump.setdescr(loop_descr.clone());
                jump
            },
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(200), OpRef::int_op(300)],
            &[OpRef::int_op(0)],
            &[],
            6,
            true,
            &[],
            &constants,
            None,
            Some(loop_descr),
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(200), OpRef::int_op(300)]
        );
        assert_eq!(combined[1].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].arg(0).to_opref(), OpRef::int_op(200));
        assert_eq!(combined[2].opcode, OpCode::Jump);
        assert_eq!(
            combined[2]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(200), OpRef::int_op(300)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_rewrites_phase2_inputargs_to_start_label() {
        // Phase 2 optimizer inputargs sit at `[inputarg_base, inputarg_base+n)`.
        // Assembly must rewrite them onto the shared start-label slots;
        // leaving `InputArg(base)` in the LABEL/JUMP is an unbound local.
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let base = 99u32;
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::SameAsR, &[rooted_inputarg_operand(Type::Ref, base)]);
                op.pos().set(OpRef::ref_op(200));
                op
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        rooted_inputarg_operand(Type::Ref, base),
                        rooted_inputarg_operand(Type::Ref, base + 1),
                    ],
                );
                jump.setdescr(loop_descr.clone());
                jump
            },
        ];
        let mut ctx = assemble_test_context(&[], &p2_ops, 2);
        let p2_ops_rc: Vec<majit_ir::OpRc> =
            p2_ops.iter().cloned().map(majit_ir::OpRc::new).collect();
        let start = [OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)];
        let combined = assemble_peeled_trace_with_jump_args(
            &[],
            &p2_ops_rc,
            &start,
            &start,
            &[],
            &[],
            2,
            base,
            false,
            &[],
            &constants,
            None,
            Some(loop_descr),
            &[],
            &mut Vec::new(),
            &mut ctx,
        );
        let label = combined
            .iter()
            .find(|op| op.opcode == OpCode::Label)
            .expect("assembled loop LABEL");
        let label_args: Vec<OpRef> = label.args_slice().iter().map(|a| a.to_opref()).collect();
        assert!(
            !label_args
                .iter()
                .any(|a| a.is_input_arg() && a.raw() >= base),
            "Phase-2 InputArg({base}+i) must not remain on the LABEL: {label_args:?}"
        );
        let jump = combined
            .iter()
            .find(|op| op.opcode == OpCode::Jump)
            .expect("assembled JUMP");
        let jump_args: Vec<OpRef> = jump.args_slice().iter().map(|a| a.to_opref()).collect();
        assert_eq!(jump_args, start.to_vec());
    }

    #[test]
    fn test_assemble_retrace_remaps_phase2_inputs_to_emitted_label() {
        // Retrace emits no start LABEL. Phase-2 InputArg(base+i) must land
        // on the loop LABEL that is emitted, not on the unemitted portal
        // start-label slots.
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let base = 99u32;
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::SameAsR, &[rooted_inputarg_operand(Type::Ref, base)]);
                op.pos().set(OpRef::ref_op(200));
                op
            },
            {
                let mut guard = Op::new(
                    OpCode::GuardTrue,
                    &[rooted_inputarg_operand(Type::Ref, base)],
                );
                guard.setfailargs(vec![rooted_inputarg_operand(Type::Ref, base)].into());
                guard
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        rooted_inputarg_operand(Type::Ref, base),
                        rooted_inputarg_operand(Type::Ref, base + 1),
                    ],
                );
                jump.setdescr(loop_descr.clone());
                jump
            },
        ];
        let mut ctx = assemble_test_context(&[], &p2_ops, 2);
        let p2_ops_rc: Vec<majit_ir::OpRc> =
            p2_ops.iter().cloned().map(majit_ir::OpRc::new).collect();
        let start = [OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)];
        let loop_label = [OpRef::ref_op(10), OpRef::ref_op(11)];
        let combined = assemble_peeled_trace_with_jump_args(
            &[],
            &p2_ops_rc,
            &loop_label,
            &start,
            &[],
            &[],
            2,
            base,
            false,
            &[],
            &constants,
            None,
            Some(loop_descr),
            &[],
            &mut Vec::new(),
            &mut ctx,
        );
        let same = combined
            .iter()
            .find(|op| op.opcode == OpCode::SameAsR)
            .expect("assembled SameAsR");
        assert_eq!(
            same.arg(0).to_opref(),
            OpRef::ref_op(10),
            "retrace must remap Phase-2 InputArg onto the emitted loop LABEL"
        );
        let guard = combined
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("assembled guard");
        let fa = guard.getfailargs().expect("guard failargs");
        assert_eq!(
            fa[0].to_opref(),
            OpRef::ref_op(10),
            "retrace failarg must use the emitted loop LABEL slot"
        );
        assert!(
            combined
                .iter()
                .filter(|op| op.opcode == OpCode::Label)
                .count()
                == 1,
            "retrace must emit only the loop LABEL"
        );
    }

    /// A JUMP redirected onto ANOTHER trace's target token already carries the
    /// arity that token's virtual state produced (unroll.py:346-357), so the
    /// assembler must pass it through. compile.py:334 asserts equal JUMP and
    /// LABEL arity for the same-label close only; applying that coercion to an
    /// external close cut the trailing arg and left the backend comparing a
    /// short JUMP against the target LABEL.
    #[test]
    fn test_assemble_peeled_trace_keeps_external_target_jump_arity() {
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let external_descr = TargetToken::new_loop(2).as_jump_target_descr();
        let p2_ops = vec![{
            let mut jump = Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 200),
                    rooted_resop_operand(Type::Int, 300),
                    Operand::from_opref(OpRef::const_int(1)),
                ],
            );
            jump.setdescr(external_descr);
            jump
        }];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(200), OpRef::int_op(300)],
            &[OpRef::int_op(0)],
            &[],
            6,
            true,
            &[],
            &majit_ir::ConstMap::default(),
            None,
            Some(loop_descr),
        );

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(200), OpRef::int_op(300), OpRef::const_int(1)]
        );
    }

    #[test]
    fn test_splice_redirected_tail_replaces_terminal_jump() {
        let body_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                op.pos().set(OpRef::void_op(3));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        let redirected_tail = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Void, 3)]);
                op.setfailargs(vec![rooted_resop_operand(Type::Void, 3)].into());
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Void, 3),
                    rooted_resop_operand(Type::Int, 4),
                ],
            ),
        ];

        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(OpRc::new).collect();
        let redirected_tail: Vec<majit_ir::OpRc> =
            redirected_tail.into_iter().map(OpRc::new).collect();
        let spliced = splice_redirected_tail(&body_ops, &redirected_tail);
        assert_eq!(spliced.len(), 3);
        assert_eq!(spliced[0].opcode, OpCode::IntAdd);
        assert_eq!(spliced[1].opcode, OpCode::GuardTrue);
        assert_eq!(spliced[2].opcode, OpCode::Jump);
        assert_eq!(
            spliced[2]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::void_op(3), OpRef::int_op(4)]
        );
    }

    #[test]
    fn test_closing_loop_contract_arity_uses_actual_jump_contract() {
        let ops = vec![Op::new(
            OpCode::Jump,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
                rooted_resop_operand(Type::Int, 2),
            ],
        )];

        assert_eq!(closing_loop_contract_arity(&ops, 5), 3);
    }

    /// unroll.py:409 stamps every replayed short-preamble guard with
    /// `patchguardop.rd_resume_position`, unconditionally. A short-preamble
    /// entry is a whole-op clone of a guard from the preamble trace, so it
    /// arrives carrying that trace's coordinate; leaving it in place when no
    /// patchguardop exists hands `store_final_boxes_in_guard` a position from
    /// another trace's numbering. Decline the inlining instead.
    #[test]
    fn inline_short_preamble_declines_a_replayed_guard_with_no_patchguardop() {
        use crate::optimizeopt::shortpreamble::{ShortPreamble, ShortPreambleOp};

        let mut optimizer = Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(2, 0);

        let short_input_operand = rooted_resop_operand(Type::Int, 11);
        let short_input = short_input_operand.to_opref();
        let jump_arg = OpRef::int_op(12);
        let jump_arg_box = ctx.materialize_operand_at(jump_arg);

        // A guard carrying the PREAMBLE trace's resume coordinate.
        let mut guard = Op::new(OpCode::GuardTrue, &[short_input_operand]);
        guard.pos().set(OpRef::void_op(13));
        guard.set_rd_resume_position(4242);

        let mut short_preamble = ShortPreamble::empty();
        short_preamble.inputargs = vec![short_input];
        short_preamble.ops = vec![ShortPreambleOp {
            op: guard,
            arg_mapping: Vec::new(),
            fail_arg_mapping: Vec::new(),
        }];

        assert!(ctx.patchguardop.is_none());
        let extra = OptUnroll::inline_short_preamble(
            &[jump_arg_box.clone()],
            &[jump_arg_box],
            &short_preamble,
            &mut optimizer,
            &mut ctx,
        );

        assert!(extra.is_empty(), "declined inlining returns no extra args");
        assert!(
            ctx.take_invalid_loop().is_some(),
            "the caller must see an InvalidLoop and fall back to jump_to_preamble"
        );
    }

    /// unroll.py:333 dereferences `self.optimizer.patchguardop` before stamping
    /// the guards `generate_guards` produced, which upstream can do because
    /// `reached_loop_header` emits a GUARD_FUTURE_CONDITION before the closing
    /// JUMP of every trace that reaches it — a bridge included
    /// (pyjitpl.py). pyre records that GFC only on the loop path, so
    /// a bridge arrives here with whatever `optimize_bridge` could synthesize
    /// from its own body guards, and a bridge with no such guard arrives with
    /// none. Skip the target token instead of dereferencing.
    #[test]
    fn jump_to_existing_trace_declines_a_target_needing_guards_with_no_patchguardop() {
        use crate::history::TargetToken;
        use crate::optimizeopt::intutils::IntBound;
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};

        let mut optimizer = Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(1, 0);

        let arg_operand = rooted_resop_operand(Type::Int, 11);
        let arg = arg_operand.to_opref();

        // The incoming state knows nothing statically; the target demands a
        // narrow bound and the concrete runtime value falls inside it, which is
        // virtualstate.py:493-498's runtime fallback — it manufactures
        // INT_GE/INT_LE + GUARD_TRUE pairs (intutils.py make_guards
        // `IntBound.make_guards`) rather than rejecting the target.
        let incoming = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        let runtime_box = OpRef::const_int(5);
        let mut target_tokens = vec![TargetToken::new_loop(1)];
        target_tokens[0].attrs().virtual_state =
            Some(VirtualState::new(vec![VirtualStateInfo::IntBounded(
                IntBound::bounded(0, 10),
            )]));

        let unroll = OptUnroll::default();
        assert!(ctx.patchguardop.is_none());
        assert!(optimizer.patchguardop.is_none());
        let vs = unroll.jump_to_existing_trace_with_vs(
            &[arg],
            None,
            &mut target_tokens,
            &mut optimizer,
            &mut ctx,
            false,
            &[runtime_box],
            Some(incoming),
        );

        // unroll.py:207-211 reads the result as "None means a target matched and
        // the jump was retargeted"; a returned virtual state means none did, and
        // `optimize_bridge` falls back to jump_to_preamble. The point of the test
        // is that the declined target reaches that fallback instead of
        // dereferencing the absent patchguardop.
        assert!(
            vs.is_some(),
            "the only target token needed extra guards it could not stamp, so it \
             must be declined and the caller left to jump_to_preamble"
        );
    }

    #[test]
    fn jump_to_existing_trace_skips_a_target_whose_label_tail_has_no_recipe() {
        use crate::history::TargetToken;
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};

        let mut optimizer = Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(1, 0);
        let arg = rooted_resop_operand(Type::Int, 11).to_opref();

        // The virtual states match exactly, so the only reason to decline is
        // the LABEL slot no close can deliver.
        let incoming = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        let mut target_tokens = vec![TargetToken::new_loop(1)];
        target_tokens[0].attrs().virtual_state =
            Some(VirtualState::new(vec![VirtualStateInfo::Unknown(
                Type::Int,
            )]));
        target_tokens[0].attrs().label_tail_unrebuildable = true;

        let vs = OptUnroll::default().jump_to_existing_trace_with_vs(
            &[arg],
            None,
            &mut target_tokens,
            &mut optimizer,
            &mut ctx,
            false,
            &[OpRef::const_int(5)],
            Some(incoming),
        );
        assert!(
            vs.is_some(),
            "a JUMP to this target would be short of its LABEL, so the caller \
             must be left to jump_to_preamble"
        );
    }

    /// A loop-invariant `CALL_PURE` on a label arg the jump passes back
    /// unchanged, an immutable getfield of its result, and `int_add_ovf` on
    /// that field. `PureOp.add_op_to_short` records the call as `CALL_PURE`,
    /// `inline_short_preamble` sends it through the optimizer, and
    /// `HeapOp.produce_op` / `CachedField._getfield` share the resolved box,
    /// so the peeled body (the ops after the loop label, including the
    /// jump tail) contains none of the three.
    #[test]
    fn peeled_loop_reuses_invariant_call_pure_immutable_getfield_and_ovf() {
        use majit_ir::descr::{make_call_descr, make_simple_descr_group};
        use majit_ir::{ArrayFlag, EffectInfo, ExtraEffect, OopSpecIndex, Type, Value};

        let passed = rooted_inputarg_operand(Type::Int, 0);
        let idx = rooted_inputarg_operand(Type::Int, 1);
        let func = Operand::const_from_value(Value::Int(0x1000));
        let one = Operand::const_from_value(Value::Int(1));
        let thousand = Operand::const_from_value(Value::Int(1000));
        let mut call = Op::new(OpCode::CallPureR, &[func, passed.clone(), one]);
        call.setdescr(make_call_descr(
            vec![Type::Int, Type::Int, Type::Int],
            Type::Ref,
            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None),
        ));
        let group = make_simple_descr_group(
            1,
            16,
            1,
            0,
            &[majit_ir::descr::SimpleFieldDescrSpec {
                is_class_word: Some(false),
                index: 1,
                field_key: "intval".to_string(),
                name: "intval".to_string(),
                offset: 8,
                field_size: 8,
                field_type: Type::Int,
                is_immutable: true,
                is_quasi_immutable: false,
                flag: ArrayFlag::Signed,
                virtualizable: false,
                index_in_parent: 0,
            }],
        );
        let descr = group.field_descrs[0].clone();
        let getfield = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rooted_resop_operand(Type::Ref, 2)],
            descr,
        );
        let ovf = Op::new(
            OpCode::IntAddOvf,
            &[rooted_resop_operand(Type::Int, 3), thousand],
        );
        let guard = Op::new(OpCode::GuardNoOverflow, &[]);
        let inc = Op::new(
            OpCode::IntAdd,
            &[idx, Operand::const_from_value(Value::Int(1))],
        );
        let mut ops = vec![call, getfield, ovf, guard, inc];
        assign_positions(&mut ops, 2);
        let jump = Op::new(OpCode::Jump, &[passed, rooted_resop_operand(Type::Int, 6)]);
        ops.push(jump);

        let (ops, snapshots) = crate::optimizeopt::seed_empty_guard_snapshots(&ops);
        let mut unroll = UnrollOptimizer::new();
        unroll.trace_inputargs = OpRef::inputarg_refs(&[Type::Int, Type::Int]);
        unroll.snapshot_boxes = snapshots;
        let (result, _) = unroll.optimize_trace_with_constants_and_inputs(
            &ops,
            &mut majit_ir::ConstMap::default(),
            2,
        );
        let label_at = result
            .iter()
            .rposition(|op| op.opcode == OpCode::Label)
            .expect("peeled loop has a label");
        let body = &result[label_at + 1..];
        let bad: Vec<_> = body
            .iter()
            .filter(|op| {
                op.opcode.is_call()
                    || op.opcode == OpCode::GetfieldGcI
                    || op.opcode == OpCode::IntAddOvf
                    || op.opcode == OpCode::GuardNoOverflow
            })
            .map(|op| op.opcode)
            .collect();
        assert!(
            bad.is_empty(),
            "peeled body must drop the invariant call, getfield and ovf, got {bad:?}"
        );
    }

    /// `PureOp.add_op_to_short` rewrites a demoted CALL to `CALL_PURE` before
    /// the short preamble is replayed. `inline_short_preamble` then sends
    /// that op through the optimizer, so a jump onto a target whose short
    /// preamble contains the call does not append the plain CALL.
    #[test]
    fn short_preamble_records_call_as_call_pure() {
        use crate::optimizeopt::shortpreamble::{PreambleOpKind, ShortBoxes};
        use majit_ir::descr::make_call_descr;
        use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex, Type, Value};

        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Int]);
        let passed = ctx.materialize_operand_at(OpRef::input_arg_int(0));
        let func = Operand::const_from_value(Value::Int(0x1000));
        let mut call = Op::new(OpCode::CallR, &[func, passed]);
        call.pos().set(OpRef::ref_op(2));
        call.setdescr(make_call_descr(
            vec![Type::Int, Type::Int],
            Type::Ref,
            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None),
        ));
        let i0 = OpRef::input_arg_int(0);
        let mut sb = ShortBoxes::with_label_args(&[i0]);
        sb.add_pure_op(&mut ctx, &OpRc::new(call));
        let short_boxes = sb.create_short_boxes(&mut ctx, &[i0], &[Type::Int]);
        let pure = short_boxes
            .iter()
            .find(|produced| produced.kind == PreambleOpKind::Pure)
            .expect("the call is a short-preamble pure op");
        assert_eq!(
            pure.preamble_op.opcode,
            OpCode::CallPureR,
            "PureOp.add_op_to_short must record CALL_PURE, not the demoted CALL"
        );
    }

    /// virtualstate.py `VirtualStateInfo._generalization_of_structpart` matches
    /// two virtual instances by `known_class.same_constant` and fielddescr
    /// identity, not by the allocation's SizeDescr object. A guard-failure
    /// bridge whose `new_with_vtable` carries a different SizeDescr allocation
    /// of the same class must still jump to the peeled target, and must not
    /// force the virtual (`NewWithVtable` in the bridge).
    #[test]
    fn bridge_virtual_of_the_same_class_jumps_to_the_peeled_target() {
        use crate::history::TargetToken;
        use crate::optimizeopt::info::AbstractVirtualPtrInfo;
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};
        use std::sync::Arc;

        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::ptr_info::{PtrInfo, VirtualInfo};

        let vtable = 0x1000i64;
        let size_of = |index: u32| {
            Arc::new(SimpleSizeDescr::with_vtable(index, 16, 7, vtable as usize)) as DescrRef
        };
        let target_descr = size_of(1);
        let bridge_descr = size_of(2);
        assert_ne!(
            majit_ir::descr_identity(&target_descr),
            majit_ir::descr_identity(&bridge_descr)
        );

        let virtual_shape = |descr: DescrRef| VirtualStateInfo::Virtual {
            descr,
            known_class: Some(vtable),
            ob_type_descr: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
        };
        let target_vs = VirtualState::new(vec![virtual_shape(target_descr.clone())]);
        let incoming = VirtualState::new(vec![virtual_shape(bridge_descr.clone())]);

        let mut optimizer = Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(1, 0);
        let arg = OpRef::ref_op(11);
        let arg_box = ctx.materialize_operand_at(arg);
        ctx.set_ptr_info(
            &arg_box,
            PtrInfo::Virtual(VirtualInfo {
                descr: bridge_descr,
                known_class: Some(vtable),
                ob_type_descr: None,
                fields: Default::default(),
                last_guard_pos: -1,
                avpi: AbstractVirtualPtrInfo::new(),
            }),
        );

        let mut target_tokens = vec![TargetToken::new_loop(1)];
        let peeled = target_tokens[0].as_jump_target_descr();
        target_tokens[0].attrs().virtual_state = Some(target_vs);

        let unroll = OptUnroll::default();
        let vs = unroll.jump_to_existing_trace_with_vs(
            &[arg],
            None,
            &mut target_tokens,
            &mut optimizer,
            &mut ctx,
            false,
            &[arg],
            Some(incoming),
        );
        assert!(
            vs.is_none(),
            "same virtual class must jump to the peeled target, not fall through \
             to the preamble"
        );
        let jump = ctx
            .new_operations
            .last()
            .expect("a successful match emits the retargeted JUMP");
        assert_eq!(jump.opcode, OpCode::Jump);
        let jump_descr = jump.getdescr().expect("JUMP carries the target descr");
        assert_eq!(
            majit_ir::descr_identity(&jump_descr),
            majit_ir::descr_identity(&peeled)
        );
        assert!(
            ctx.new_operations
                .iter()
                .all(|op| op.opcode != OpCode::NewWithVtable),
            "matching virtuals stay virtual; the bridge must not materialize them"
        );
    }
}
