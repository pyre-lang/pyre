use crate::optimizeopt::OptContext;
use crate::optimizeopt::{
    earlyforce::OptEarlyForce,
    heap::OptHeap,
    intbounds::OptIntBounds,
    pure::OptPure,
    rewrite::OptRewrite,
    simplify::OptSimplify,
    virtualize::{OptVirtualize, VirtualizableConfig},
    vstring::OptString,
};

/// `optimizeopt/__init__.py` `ALL_OPTS`. `unroll` is listed and is not
/// instantiated (`opt is None`).
pub const ALL_OPTS: &[(&str, bool)] = &[
    ("intbounds", true),
    ("rewrite", true),
    ("virtualize", true),
    ("string", true),
    ("pure", true),
    ("earlyforce", true),
    ("heap", true),
    ("unroll", false),
];

/// `rlib/jit.py` `ENABLE_ALL_OPTS`.
pub const ENABLE_ALL_OPTS: &str = "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll";
/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.
use indexmap::{IndexMap, IndexSet};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, Op, OpCode, OpRc, OpRef, Type};
use std::sync::Arc;

use crate::optimizeopt::info::{PtrInfo, PtrInfoExt};
use crate::optimizeopt::intutils::IntBound;
use crate::optimizeopt::{SnapshotBoxes, SnapshotFramePcs, SnapshotFrameSizes};

/// optimizer.py OptimizationResult: result of an optimization pass.
#[derive(Debug)]
pub enum OptimizationResult {
    /// Emit this operation (possibly modified).
    Emit(Op),
    /// Replace with a different operation; continue with the next pass.
    Replace(Op),
    /// optimizer.py `send_extra_operation(newop, opt=None)` — re-dispatch
    /// the new op from the first optimization, dropping the original.
    /// autogenintrules.py:54-55 uses this pattern for every rewrite-style
    /// rule so that chained OptIntBounds rules (add_zero, int_is_zero, …)
    /// fire on the rewritten op.
    Restart(Op),
    /// Remove the operation entirely.
    Remove,
    /// Pass the operation to the next pass unchanged.
    PassOn,
    /// rewrite.py:406 — a guard was proven to always fail; abort the trace.
    /// RPython raises `InvalidLoop`; pyre threads it as a value (the driver
    /// converts it to `Err(InvalidLoop)` at the pass barrier) so it works
    /// under `panic=abort`.  Carries the abandon reason for diagnostics.
    InvalidLoop(&'static str),
}

/// An optimization pass.
///
/// optimizer.py: Optimization base class.
pub trait Optimization {
    /// Process an operation. Called for each operation in the trace.
    fn propagate_forward(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult;

    /// optimizer.py propagate_postprocess — called AFTER the op has been
    /// emitted through all passes and added to new_operations. Runs in
    /// REVERSE pass order. RPython uses this for bounds propagation
    /// (intbounds.py postprocess_GUARD_TRUE) and heap cache updates
    /// (heap.py postprocess_GETFIELD_GC_I).
    fn propagate_postprocess(&mut self, _op: &Op, _op_rc: &majit_ir::OpRc, _ctx: &mut OptContext) {}

    /// optimizer.py have_postprocess
    fn have_postprocess(&self) -> bool {
        false
    }

    /// optimizer.py have_postprocess_op(opnum)
    fn have_postprocess_op(&self, _opcode: OpCode) -> bool {
        self.have_postprocess()
    }

    /// Called once before optimization starts.
    fn setup(&mut self) {}

    /// Drop state that `setup` keeps for an intra-compile phase-1 → phase-2
    /// handoff (`OptPure.extra_call_pure` / `preamble_pure_ops`) so a recycled
    /// optimizer cannot leak it into the next `compile_bridge`.
    ///
    /// RPython `BridgeCompileData.optimize` constructs a new
    /// `UnrollOptimizer` per compile; nursery allocation is cheap there.
    fn reset_between_compiles(&mut self) {
        self.setup();
    }

    /// Called after all operations have been processed.
    fn flush(&mut self, _ctx: &mut OptContext) {}

    /// Mark this pass as Phase 2 (loop body). Phase 2 should not fully
    /// virtualize New() ops because guard recovery_layout is not yet
    /// populated. Default: no-op.
    fn set_phase2(&mut self, _phase2: bool) {}

    /// warmstate.py: pureop_historylength.
    /// Only OptPure consumes this; other passes ignore it.
    fn set_pureop_historylength(&mut self, _limit: usize) {}

    /// `metainterp_sd.virtualref_info` parity hook. `OptVirtualize` reads all
    /// three descriptors; `OptHeap.force_from_effectinfo` reads
    /// `descr_forced` for forcing calls.
    fn set_vrefinfo(&mut self, _vrefinfo: crate::virtualref::VirtualRefInfo) {}

    /// optimizer.py propagate_all_forward(trace, call_pure_results, flush).
    /// Only OptPure consumes this; other passes ignore it.
    fn set_call_pure_results(&mut self, _results: &crate::optimizeopt::util::ArgsDict) {}

    /// Name of this pass (for debugging).
    fn name(&self) -> &'static str;

    /// optimizer.py:557 parity hook — drain this pass's accumulated
    /// `Counters.*` bumps into `staticdata.profiler` and reset the
    /// internal accumulators.
    ///
    /// Each pass that records its own `Counters.*` bumps
    /// (vector.py:139/146 OPT_VECTORIZE_TRY/OPT_VECTORIZED, heap.py
    /// HEAPCACHED_OPS, ...) overrides this; the default impl does
    /// nothing for passes that have no counters of their own.
    /// `Optimizer::update_counters` calls this on every pass after
    /// each `propagate_all_forward` exit.
    fn drain_profiler_counters(&mut self, _profiler: &crate::jitprof::JitProfiler) {}

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Contribute operations to the short preamble builder.
    /// Called after preamble optimization to collect ops that bridges need to replay.
    /// RPython passes `optimizer` for PtrInfo access. We pass `ctx`.
    fn produce_potential_short_preamble_ops(
        &self,
        _sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &mut OptContext,
    ) {
        // Default: no contribution
    }

    /// heap.py serialize_optheap — export struct field triples. The
    /// `available_boxes` filter (heap.py:836,845) is applied in bridgeopt.
    fn export_cached_fields(
        &self,
        _ctx: &mut OptContext,
    ) -> Vec<(OpRef, majit_ir::DescrRef, OpRef)> {
        Vec::new()
    }

    /// heap.py deserialize_optheap — import struct fields.
    fn import_cached_fields(
        &mut self,
        _entries: &[(OpRef, majit_ir::DescrRef, OpRef)],
        _ctx: &mut OptContext,
    ) {
    }

    /// heap.py serialize_optheap — export array item triples. The
    /// `available_boxes` filter (heap.py:855,866) is applied in bridgeopt.
    fn export_cached_arrayitems(
        &self,
        _ctx: &mut OptContext,
    ) -> Vec<(OpRef, i64, majit_ir::DescrRef, OpRef)> {
        Vec::new()
    }

    /// heap.py deserialize_optheap — import array item triples.
    fn import_cached_arrayitems(
        &mut self,
        _entries: &[(OpRef, i64, majit_ir::DescrRef, OpRef)],
        _ctx: &mut OptContext,
    ) {
    }

    /// shortpreamble.py `HeapOp.produce_op` passes the owning
    /// `CachedField` / `ArrayCachedItem` into `PtrInfo.setfield/setitem`, which
    /// immediately calls `register_info`.  This hook preserves that ownership
    /// while the Rust import driver dispatches across optimizer passes.
    fn register_preamble_cached_field(
        &mut self,
        _struct_box: &majit_ir::operand::Operand,
        _descr: &majit_ir::DescrRef,
    ) {
    }

    fn register_preamble_cached_arrayitem(
        &mut self,
        _array_box: &majit_ir::operand::Operand,
        _descr: &majit_ir::DescrRef,
        _index: i64,
    ) {
    }

    /// rewrite.py serialize_optrewrite
    fn serialize_optrewrite(&self) -> Vec<(i64, OpRef)> {
        Vec::new()
    }

    /// rewrite.py deserialize_optrewrite
    fn deserialize_optrewrite(&mut self, _entries: &[(i64, OpRef)]) {}

    /// shortpreamble.py: PureOp.produce_op / LoopInvariantOp.produce_op
    /// Transfer imported PreambleOp entries from OptContext to this pass.
    /// RPython calls `opt.optimizer.optpure` directly during produce_op.
    /// In majit, the Optimization trait mediates this transfer.
    fn install_preamble_pure_ops(&mut self, _ctx: &mut OptContext) {}

    /// RPython unroll.py: exported_infos also carries widened IntBound knowledge.
    fn export_arg_int_bounds(
        &self,
        _args: &[OpRef],
        _ctx: &OptContext,
    ) -> indexmap::IndexMap<majit_ir::operand::Operand, IntBound> {
        indexmap::IndexMap::new()
    }

    /// optimizer.py: is_virtual(opref)
    /// Whether an opref refers to a virtual object (for this pass).
    fn is_virtual(&self, _opref: OpRef) -> bool {
        false
    }

    /// RPython optimizer.py: emitting_operation(op)
    /// Called before any operation is emitted to the output, regardless of
    /// which pass emits it. This enables passes like OptHeap to force lazy
    /// sets before guards, even when the guard is emitted by an earlier pass.
    ///
    /// `self_pass_idx` is this pass's own index in the optimizer pipeline.
    /// RPython uses `self.next_optimization` to route lazy-set emissions
    /// starting AFTER the current pass. In majit, pass this index to
    /// `emit_extra` to achieve the same behavior.
    fn emitting_operation(&mut self, _op: &Op, _ctx: &mut OptContext, _self_pass_idx: usize) {}
}

/// optimizer.py `have_postprocess(cls)`.
///
/// RPython asks the class object whether `propagate_postprocess` was
/// overridden. In Rust each pass exposes the same answer through the
/// `Optimization::have_postprocess` method.
pub fn have_postprocess<T: Optimization + ?Sized>(opt: &T) -> bool {
    opt.have_postprocess()
}

/// optimizer.py `CantReplaceGuards`.
///
/// PyPy stores the optimizer reference on the context manager.  Pyre cannot
/// keep a mutable borrow of `Optimizer` across the guarded section, so this
/// object carries the saved `oldval` and `Optimizer::restore_can_replace_guards`
/// applies it at scope exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CantReplaceGuards {
    oldval: bool,
}

impl CantReplaceGuards {
    fn new(oldval: bool) -> Self {
        Self { oldval }
    }
}

/// optimizer.py `LoopInfo`.
pub trait LoopInfo {
    fn label_op(&self) -> Option<&Op> {
        None
    }

    fn r#final(&self) -> bool;

    fn post_loop_compilation(
        &mut self,
        _loop_ops: &[majit_ir::OpRc],
        _jitdriver_sd: (),
        _metainterp: (),
        _jitcell_token: (),
    ) {
    }
}

/// optimizer.py `BasicLoopInfo`.
#[derive(Clone, Debug)]
pub struct BasicLoopInfo {
    pub inputargs: Vec<OpRef>,
    pub quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    pub jump_op: Option<Op>,
    pub extra_same_as: Vec<Op>,
    pub extra_before_label: Vec<Op>,
    pub label_op: Option<Op>,
}

impl BasicLoopInfo {
    pub fn new(
        inputargs: Vec<OpRef>,
        quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
        jump_op: Option<Op>,
    ) -> Self {
        Self {
            inputargs,
            quasi_immutable_deps,
            jump_op,
            extra_same_as: Vec::new(),
            extra_before_label: Vec::new(),
            label_op: None,
        }
    }
}

impl LoopInfo for BasicLoopInfo {
    fn label_op(&self) -> Option<&Op> {
        self.label_op.as_ref()
    }

    fn r#final(&self) -> bool {
        true
    }
}

/// bridgeopt.py deserialize_optimizer_knowledge parity: data needed to call
/// deserialize_optimizer_knowledge after optimizer setup.
pub(crate) struct PendingBridgeRd {
    /// compile.py `ResumeGuardDescr` storage handle. The bridge
    /// deserializer borrows `storage.rd_numb` / `storage.rd_consts()`
    /// off the same Arc the guard owns — no owned clones, and every
    /// reader (runtime + GC walker) observes the same pool.
    pub storage: std::sync::Arc<crate::resume::ResumeStorage>,
    pub frontend_boxes: Vec<i64>,
    pub liveboxes: Vec<OpRef>,
    pub livebox_types: Vec<Type>,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    pub all_descrs: Arc<Vec<majit_ir::descr::DescrRef>>,
    /// `optimizer.cpu` (model.py AbstractCPU) — carried through the
    /// bridge into the retrace `Optimizer.cpu` slot. RPython never sees a
    /// `None` here: `optimizer.cpu` is set at `Optimizer.__init__` time.
    pub cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
}

/// The optimizer: chains passes and runs them over a trace.
///
/// RPython optimizer.py: Optimizer class with pass chain and shared state.
pub struct Optimizer {
    /// Backend capability threaded into each optimization context.
    pub supports_efficient_uint_mul_high: bool,
    passes: Vec<Box<dyn Optimization>>,
    pub pureop_historylength: usize,
    /// Final num_inputs after optimization (may increase if virtualizable
    /// adds virtual input args).
    final_num_inputs: usize,
    /// Cache of CALL_PURE results from previous traces.
    /// optimizer.py: `call_pure_results` — maps constant values
    /// (via get_constant_box) → result value, carried across
    /// loop iterations so the optimizer can constant-fold repeated
    /// pure calls. RPython uses value-based equality for keys.
    pub call_pure_results: crate::optimizeopt::util::ArgsDict,
    /// optimizer.py: `_last_guard_op` — tracks the last emitted guard
    /// for guard sharing and descriptor fusion.
    ///
    /// Stored as the donor's index in `ctx.new_operations` (set by
    /// `emit_guard_operation` just before `ctx.emit` pushes the op, so
    /// the predicted position equals `ctx.new_operations.len()` at
    /// assignment time). `_copy_resume_data_from` looks up the donor
    /// via this index and stamps a `ResumeGuardCopiedDescr(prev)`
    /// (or `ResumeGuardCopiedExcDescr(prev)` for exception guards) on
    /// the sharer, matching `compile.py:832
    /// ResumeGuardCopiedDescr(prev)` parity.
    last_guard_op_idx: Option<usize>,
    /// optimizer.py:241/304/632-634 `replaces_guard` — maps an emitted
    /// guard op to its replacement. RPython keys this dict by the guard
    /// `op` object itself (object identity); pyre keys by the guard op's
    /// canonical producer `Operand` (compared by `Rc::ptr_eq`), the
    /// box-identity analog of `op is op`. Every key is resolved through
    /// `ctx.resolve_to_operand(op.pos)` so insert and lookup agree on the
    /// canonical producer. Guard ops are never Const, so the key is always a
    /// ptr-stable ResOp producer.
    replaces_guard: crate::FxIndexMap<majit_ir::operand::Operand, Op>,
    /// optimizer.py: `pendingfields` — heap fields that need to be
    /// written back before the next guard (lazy set forcing).
    pendingfields: Vec<Op>,
    /// optimizer.py: `can_replace_guards` — flag to enable/disable guard sharing.
    can_replace_guards: bool,
    /// optimizer.py: `quasi_immutable_deps` — quasi-immutable field
    /// dependencies, `dict[QuasiImmut -> None]` upstream. Each entry is the
    /// instance the recording resolved; after compilation each one gets the
    /// loop's invalidation flag registered on it.
    pub quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    /// RPython unroll.py: import_state — virtual structures to inject at Phase 2 start.
    /// Maps the original loop-carried input slot to a recursive abstract
    /// description of the virtual's field values.
    pub(crate) imported_virtuals: Vec<ImportedVirtual>,
    /// optimizer.py `self.inputargs = inputargs` parity.
    /// Typed InputArg OpRefs (InputArgInt/InputArgRef/InputArgFloat)
    /// carrying `box.type` (history.py:220) intrinsically via variant tag.
    pub trace_inputargs: Vec<OpRef>,
    /// unroll.py / compile.py parity: original live values at the jump
    /// point, threaded into export_state as `runtime_boxes`. Dormant
    /// until the export/import pair starts reading it.
    pub runtime_boxes: Vec<OpRef>,
    /// RPython unroll.py: export_state — exported optimizer facts at the end
    /// of the preamble, adapted to majit's slot-based inputarg model.
    pub exported_loop_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// RPython unroll.py: import_state — exported facts to re-apply onto the
    /// next optimizer instance before phase 2 body optimization starts.
    pub imported_loop_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// Invented SameAs aliases imported from short-preamble export/import.
    pub imported_short_aliases: Vec<crate::optimizeopt::ImportedShortAlias>,
    /// Builder-derived short preamble actually used by phase 2.
    pub imported_short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: short_preamble_producer after import_state.
    /// Preserved so finalize_short_preamble can create the live extended
    /// producer for the target token currently being compiled.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// RPython `unroll.py:298` `self.short_preamble_producer = ...`.
    ///
    /// `imported_short_preamble_builder` above ports `unroll.py:507`.
    /// Upstream stage-switches one slot in time, overwriting the imported
    /// plain builder with this extended builder. Pyre keeps both stages live
    /// as distinct Rust types; relocating this field to `Optimizer` changes
    /// the extended builder's home without merging those stages.
    pub short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    /// MetaInterp's `Option<usize>` publication slot for this producer. While
    /// the builder is on loan to OptContext, the slot is re-pointed at
    /// `OptContext.active_short_preamble_producer` so the root walker always
    /// follows the builder's current home.
    pub(crate) published_short_preamble_producer_slot: Option<usize>,
    /// RPython unroll.py: `label_args = import_state(...)`.
    /// The peeled loop's LABEL must use these args, not the phase-1 end_args.
    pub imported_label_args: Option<Vec<OpRef>>,
    /// simplify.py: patchguardop recorded from GUARD_FUTURE_CONDITION.
    pub patchguardop: Option<Op>,
    /// RPython: propagate_all_forward(trace, flush=False) for Phase 2.
    /// When true, skip flush() at end of optimization.
    pub skip_flush: bool,
    /// RPython optimizer.py: last_op/info.jump_op when flush=False.
    /// Phase-2 loop compilation keeps the terminal JUMP/FINISH outside the
    /// returned body ops and lets unroll/compile own it explicitly.
    pub terminal_op: Option<Op>,
    /// Preserved final context after optimization, for jump_to_existing_trace.
    pub final_ctx: Option<OptContext>,
    /// Recycled `OptContext` maps from the previous compile. Taken at the
    /// start of the next `optimize_with_constants_and_inputs_at` so
    /// `resop_refs` / `new_operations` / `emitted_operations` keep their
    /// capacity (`optimizer.py Optimizer.__init__` builds those maps once).
    recycled_ctx: Option<Box<OptContext>>,
    /// RPython Box identity: generation epoch for Phase 2 ops.
    /// Phase 1 JUMP arg OpRef indices to pre-tag as gen=0.
    /// bridgeopt.py deserialize_optimizer_knowledge: pending bridge resume data for deserialization
    /// after setup(). RPython calls deserialize_optimizer_knowledge after the
    /// optimizer is constructed.
    pending_bridge_rd: Option<PendingBridgeRd>,
    /// unroll.py `optimize_bridge` vs unroll.py
    /// `optimize_preamble`: the bridge path's `propagate_all_forward`
    /// (unroll.py:193) does NOT build the short-preamble/export preview, and
    /// its virtual-state matching is deferred to `jump_to_existing_trace`
    /// (unroll.py:207), which catches `VirtualStatesCantMatch` and falls back
    /// to `jump_to_preamble` (unroll.py). pyre folds the preview export
    /// into the shared `optimize_with_constants_and_inputs_at`; this flag marks
    /// the bridge call so the preview's `make_inputargs_and_virtuals` mismatch
    /// is NON-fatal (exported_loop_state = None) instead of escaping as an
    /// InvalidLoop that discards the bridge (the pi guard-9 infinite-deopt hang).
    building_bridge: bool,
    /// Whether this call is the one `compile.py`'s `SimpleCompileData.optimize`
    /// makes. That one runs `Optimizer.optimize_loop`, whose result is a
    /// `BasicLoopInfo` and the operations; the short-preamble export and the
    /// `ExportedState` it produces belong to `UnrollOptimizer.optimize_preamble`
    /// alone, and the force-then-flush-then-virtual-state that opens
    /// `UnrollOptimizer.export_state` runs only there. pyre folds that preview into the shared
    /// `optimize_with_constants_and_inputs_at`, so a simple compile needs this
    /// flag to reach the same shape: no forcing of the closing JUMP's arguments
    /// for a preamble that is not being built, and no exported state, which no
    /// reader of a simple loop's result consults.
    pub simple_compile: bool,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    /// Taken from MIStaticData at optimizer construction, returned after.
    /// descr.py setup_descrs: descriptors get descr_index assigned inline during
    /// collect_optimizer_knowledge_for_resume().
    pub all_descrs: Arc<Vec<DescrRef>>,
    /// optimizer.py: constant_fold allocator for compile-time object creation.
    pub constant_fold_alloc: Option<crate::optimizeopt::ConstantFoldAllocFn>,
    /// info.py `ConstPtrInfo.getstrlen1(mode)` runtime hook —
    /// propagated to `OptContext::string_length_resolver` at the start of
    /// each `optimize_with_constants_and_inputs_at` run so
    /// `EnsuredPtrInfo::Constant.getlenbound(Some(mode))` can return an
    /// exact constant string length when the host runtime supplies one.
    ///
    /// `Arc` rather than `Box` because the resolver is shared across
    /// multiple optimizer runs (propagation is a `clone()`, not a `take()`).
    pub string_length_resolver: Option<crate::optimizeopt::info::StringLengthResolver>,
    /// info.py ConstPtrInfo._unpack_str(mode) — runtime hook.
    pub string_content_resolver: Option<crate::optimizeopt::info::StringContentResolver>,
    /// history.py get_const_ptr_for_string(s) — runtime hook.
    pub string_constant_alloc: Option<crate::optimizeopt::info::StringConstantAllocator>,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Propagated to OptContext for generate_modified_call (vstring.py).
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// optimizer.py is_call_pure_pure_canraise — resume.ResumeDataLoopMemo.
    /// Shared constant pool + box numbering cache across all guards in a loop.
    pub resumedata_memo: std::rc::Rc<std::cell::RefCell<crate::resume::ResumeDataLoopMemo>>,
    /// resume.py parity: per-guard snapshots from tracing time.
    /// Maps rd_resume_position → flattened OpRef boxes from the snapshot.
    /// Propagated to OptContext for store_final_boxes_in_guard.
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    /// Propagated to OptContext for store_final_boxes_in_guard multi-frame encoding.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// resume.py:399-402 `minimum_virtualizable_size`, propagated to
    /// OptContext for the `memo.number()` call in
    /// `store_final_boxes_in_guard`.
    ///
    /// `-1` disables the check (`resume.py`); a non-negative value
    /// arms it. `virtualizable.py minimum_size()` is
    /// `num_static_fields`, and the comparison is `>` rather than `>=`
    /// because the virtualizable identity occupies one array entry of its
    /// own on top of the static fields.
    pub minimum_virtualizable_size: i64,
    /// Per-guard virtualref boxes from tracing-time snapshots.
    /// resume.py _number_boxes reads vref_array as a separate
    /// section. opencoder.py create_top_snapshot records vref_boxes
    /// alongside vable_boxes.
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// Byte-mode bridge only. `store_final_boxes_in_guard` numbers one
    /// guard from `SnapshotIterator` through this source. Loop compile and
    /// the `Vec<Op>` recorder leave it `None` and use `snapshot_boxes`.
    pub(crate) byte_bridge_resume: Option<crate::recorder::ByteBridgeResume>,
    /// Phase 1 emit ops carried into Phase 2's lookup surface (6).
    ///
    /// Mirror of `OptContext.phase1_emit_ops`; populated at the end of
    /// `optimize_with_constants_and_inputs_at` from `ctx.new_operations`,
    /// and propagated into the Phase 2 `OptContext` via `setup_optimizations`.
    /// `OptContext::op_at` resolves Phase 1 emit OpRefs through `op.type_`
    /// directly — single source of truth for cross-phase Box.type
    /// (history.py:220 parity).
    pub phase1_emit_ops: Vec<majit_ir::OpRc>,
    /// jitprof.Counters.OPT_OPS / OPT_GUARDS / OPT_GUARDS_SHARED accumulators.
    ///
    /// RPython optimizer.py:626/629/673 calls
    /// `self.metainterp_sd.profiler.count(...)` directly inside
    /// `_emit_operation` / `emit_guard_operation`.  Pyre's `Optimizer`
    /// does not hold a reference to `MetaInterpStaticData.profiler`, so
    /// we mirror the same deferred-fold pattern already in use for
    /// `nvirtuals` / `nvholes` / `nvreused`: accumulate per-counter here
    /// and let the caller fold them into `JitStatsCounters` via
    /// `update_counters` after each `propagate_all_forward` exit.
    pub(crate) opt_ops_emitted: usize,
    pub(crate) opt_guards_emitted: usize,
    pub(crate) opt_guards_shared_emitted: usize,
    /// `optimizer.cpu` (`rpython/jit/backend/model.py AbstractCPU`)
    /// backref.  Hosts `cls_of_box(box)` (model.py) and other
    /// backend services every Optimization sub-class reaches via
    /// `self.optimizer.cpu.<method>()`.  Propagated to `OptContext.cpu`
    /// at `setup_optimizations` time.
    pub cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    /// One-shot explicit `input_ops` seed for the next
    /// `optimize_with_constants_and_inputs_at` run. When `Some`, the
    /// canonical producer `Rc<Op>` slice is used directly as
    /// `find_producer_op`'s lowest-priority store. Set on the Phase 2
    /// optimizer to the recorder's `Vec<OpRc>` — the same `Rc`, so the
    /// Phase-1 `_forwarded` it carries is the authoritative one. Consumed
    /// (`take`) by the run.
    pub explicit_input_ops_seed: Option<Vec<majit_ir::OpRc>>,
    /// optimizer.py `self.inputargs`: the box objects of the trace being
    /// optimized (`trace.get_iter().inputargs`), not only their positions.
    /// A box that only a guard snapshot names is still this object, so
    /// `import_state` forwards the same box `ResumeDataLoopMemo.number`
    /// reads through `_cache`. Consumed by the run.
    pub(crate) trace_inputarg_boxes: Vec<majit_ir::InputArgRc>,
}

/// Lower a typed-`Value` constants pool into the dense
/// `IndexMap<u32, Const>` shape consumed by pyre-side guard metadata
/// builders, CompiledTrace storage, and the backend's
/// `set_constants_pool` boundary.
///
/// history.py/261/307 `ConstInt/ConstFloat/ConstPtr` are the only
/// constant classes — `Value::Void` panics rather than fabricate a
/// nonexistent `ConstVoid`.
pub(crate) fn lower_typed_constants_to_const_pool(
    constants: &majit_ir::ConstMap<majit_ir::Value>,
) -> majit_ir::ConstMap<majit_ir::Const> {
    let mut pool = majit_ir::ConstMap::default();
    for (&k, v) in constants {
        pool.insert(k, v.to_const());
    }
    pool
}

/// RPython unroll.py: import_state virtual info for Phase 2.
/// Tells OptVirtualize that an inputarg is a virtual object.
#[derive(Clone, Debug)]
pub(crate) struct ImportedVirtual {
    /// Inputarg index that holds this virtual.
    pub(crate) inputarg_index: usize,
    /// Size descriptor for the virtual's New().
    pub(crate) size_descr: majit_ir::DescrRef,
    /// Whether this imported virtual is an instance or a plain struct.
    pub(crate) kind: ImportedVirtualKind,
    /// Fields: (field_descr, exported abstract info for the field value).
    pub(crate) fields: Vec<(
        majit_ir::DescrRef,
        crate::optimizeopt::virtualstate::VirtualStateInfo,
    )>,
    /// Descr index of the GetfieldGcR(pool) that loads this head.
    /// OptVirtualize forwards this load result to the virtual head.
    pub(crate) head_load_descr_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub(crate) enum ImportedVirtualKind {
    Instance { known_class: Option<i64> },
    Struct,
}

impl Optimizer {
    pub(crate) fn register_preamble_cached_field(
        &mut self,
        struct_box: &majit_ir::operand::Operand,
        descr: &majit_ir::DescrRef,
    ) {
        for pass in &mut self.passes {
            pass.register_preamble_cached_field(struct_box, descr);
        }
    }

    pub(crate) fn register_preamble_cached_arrayitem(
        &mut self,
        array_box: &majit_ir::operand::Operand,
        descr: &majit_ir::DescrRef,
        index: i64,
    ) {
        for pass in &mut self.passes {
            pass.register_preamble_cached_arrayitem(array_box, descr, index);
        }
    }

    fn is_constant_placeholder_op(op: &Op, ctx: &OptContext) -> bool {
        if !matches!(
            op.opcode,
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
        ) {
            return false;
        }
        let Some(forwarded) = ctx.read_forwarded(op.pos().get()) else {
            return false;
        };
        if !forwarded.is_const() {
            return false;
        }
        op.num_args() == 0 || op.args_slice().iter().all(|arg| arg.is_none())
    }

    fn import_virtual_state_value(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        ctx: &mut OptContext,
    ) -> Operand {
        // virtualstate.py make_inputargs parity: each VSI leaf
        // realizes a Box whose `box.type` matches the variant. Tag the
        // OpRef variant tag at allocation time so `opref.ty()` resolves
        // priority-0 in `opref_type` / `OptBoxEnv::get_type`.
        use crate::optimizeopt::virtualstate::VirtualStateInfo;
        let tp = match info {
            VirtualStateInfo::Constant(value) => value.get_type(),
            VirtualStateInfo::Virtual { .. }
            | VirtualStateInfo::VArray { .. }
            | VirtualStateInfo::VStruct { .. }
            | VirtualStateInfo::VArrayStruct { .. }
            | VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull => majit_ir::Type::Ref,
            VirtualStateInfo::IntBounded(_) => majit_ir::Type::Int,
            // virtualstate.py not_virtual leaves are int/ref/float —
            // void-result ops are not value boxes (resoperation.py:260),
            // so allocating an InputArg cell tagged Void here would mint
            // an `OpRef::VoidOp` for what should be a value box.
            VirtualStateInfo::Unknown(majit_ir::Type::Void) => panic!(
                "import_virtual_state_value: Unknown(Void) leaf — VirtualState \
                 leaves are int/ref/float (virtualstate.py:655)"
            ),
            VirtualStateInfo::Unknown(tp) => *tp,
        };
        // Producer-less leaf: mint the box up front and route any forwarded
        // write through it (the `Unknown` arm of
        // `apply_imported_virtual_state` is a no-op, but the position must
        // still bind its canonical host at allocation — a bare
        // position resolves to `None`, so every later resolution would mint
        // a fresh position-only box, splitting `same_box`-keyed export-cache
        // keys and missing the `in_progress` cycle guard).
        let (opref, box_) = ctx.reserve_virtual_box(tp);
        if crate::debug::have_debug_prints() {
            crate::debug::log_one(
                "jit-optimizer",
                &format!("import_virtual_state_value {opref:?} <= {info:?}"),
            );
        }
        Self::apply_imported_virtual_state(info, &box_, ctx);
        box_
    }

    fn apply_imported_virtual_state(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        box_: &Operand,
        ctx: &mut OptContext,
    ) {
        use crate::optimizeopt::virtualstate::VirtualStateInfo;

        // `op.set_forwarded(info)` per optimizer.py — each arm writes the
        // imported state through the caller-provided bound box: fresh
        // virtual-state leaves arrive minted via `reserve_virtual_box`,
        // imported label-arg leaves arrive resolved via
        // `get_box_replacement_operand_opt`. Constants take the value-only path via
        // `make_constant_box`.
        match info {
            VirtualStateInfo::Constant(value) => {
                // `box_` is a caller-provided bound box (reserve_virtual_box /
                // get_box_replacement_operand_opt), so the Operand lowering is panic-free.
                ctx.make_constant_box(box_, *value);
            }
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                ob_type_descr,
                fields,
                field_descrs,
            } => {
                let mut imported_fields = majit_ir::ptr_info::VirtualFieldList::new();
                for (field_idx, field_info) in fields {
                    let field_ref = Self::import_virtual_state_value(field_info, ctx);
                    // ob_type (offset 0) class pointers — the exporter at
                    // `virtualstate.rs`'s `export_single_value` already
                    // encodes these as
                    // `Value::Ref(GcRef)`, so the imported box's position is
                    // already a typed `RefOp` variant and the box carries a
                    // `Value::Ref(class_gcref)` const forwarding step on its
                    // `_forwarded` slot
                    // (via `make_constant` in the `VirtualStateInfo::Constant`
                    // arm of `apply_imported_virtual_state`). The typed
                    // variant tag + Ref-typed const forwarding are
                    // the authoritative shape; no extra type marker is
                    // needed at the import side.
                    let _ = (field_descrs, known_class, field_idx);
                    imported_fields.push((*field_idx, field_ref));
                }
                let _ = field_descrs; // descr.all_fielddescrs() is authoritative
                ctx.set_ptr_info(
                    box_,
                    crate::optimizeopt::info::PtrInfo::Virtual(
                        crate::optimizeopt::info::VirtualInfo {
                            descr: descr.clone(),
                            known_class: *known_class,
                            ob_type_descr: ob_type_descr.clone(),
                            fields: imported_fields,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
            }
            VirtualStateInfo::VArray { descr, items, .. } => {
                let imported_items = items
                    .iter()
                    .map(|item_info| match item_info {
                        Some(item_info) => Self::import_virtual_state_value(item_info, ctx),
                        // virtualstate.py:272-280: absent fieldstate remains
                        // an unwritten virtual-array slot on import.
                        None => Operand::None,
                    })
                    .collect();
                ctx.set_ptr_info(
                    box_,
                    crate::optimizeopt::info::PtrInfo::VirtualArray(
                        crate::optimizeopt::info::VirtualArrayInfo {
                            descr: descr.clone(),
                            clear: false,
                            items: imported_items,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                let mut imported_fields = majit_ir::ptr_info::VirtualFieldList::new();
                for (field_idx, field_info) in fields {
                    let field_ref = Self::import_virtual_state_value(field_info, ctx);
                    imported_fields.push((*field_idx, field_ref));
                }
                let _ = field_descrs; // descr.all_fielddescrs() is authoritative
                ctx.set_ptr_info(
                    box_,
                    crate::optimizeopt::info::PtrInfo::VirtualStruct(
                        crate::optimizeopt::info::VirtualStructInfo {
                            descr: descr.clone(),
                            fields: imported_fields,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
            }
            VirtualStateInfo::VArrayStruct {
                descr,
                fielddescrs,
                element_fields,
            } => {
                let imported_elements = element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_info)| {
                                let field = match field_info {
                                    Some(field_info) => {
                                        Self::import_virtual_state_value(field_info, ctx)
                                    }
                                    // virtualstate.py _enum: retain the
                                    // dense unwritten element-field slot.
                                    None => Operand::None,
                                };
                                (*field_idx, field)
                            })
                            .collect()
                    })
                    .collect();
                ctx.set_ptr_info(
                    box_,
                    crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(
                        crate::optimizeopt::info::ArrayStructInfo {
                            descr: descr.clone(),
                            fielddescrs: fielddescrs.clone(),
                            element_fields: imported_elements,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
            }
            VirtualStateInfo::KnownClass { class_ptr } => {
                ctx.set_ptr_info(
                    box_,
                    crate::optimizeopt::info::PtrInfo::known_class(*class_ptr, true),
                );
            }
            VirtualStateInfo::NonNull => {
                ctx.set_ptr_info(box_, crate::optimizeopt::info::PtrInfo::nonnull());
            }
            VirtualStateInfo::IntBounded(bound) => {
                // RPython parity: imported preamble bounds become the box's
                // forwarded IntBound directly (optimizer.py setintbound
                // setintbound). No separate "imported" or "lower-only" maps.
                let widened = bound.widen();
                ctx.setintbound(box_, &widened);
            }
            VirtualStateInfo::Unknown(_tp) => {
                // virtualstate.py make_inputargs parity: each
                // NotVirtualStateInfo leaf is realized as an InputArg* whose
                // `Box.type` is intrinsic. 5: type lookup now resolves
                // through the variant tag of `opref` (typed via
                // `OpRef::input_arg_typed` / `op_typed` upstream) at priority
                // 0 of `opref_type`; the side-table seed is dead.
            }
        }
    }

    #[allow(deprecated)] // Phase 2 import_state forwards synthetic positions for imported virtual fields.
    pub(crate) fn install_imported_virtuals(&self, ctx: &mut OptContext) {
        // virtualstate.py make_inputargs + 627-634 _enum parity:
        // label_args are laid out by recursive _enum traversal where each
        // NotVirtualStateInfo leaf gets a position_in_notvirtuals slot.
        // Virtual states don't consume slots directly — their non-virtual
        // FIELD sub-states do, interleaved with top-level non-virtual states.
        // Traverse ALL states in order, advancing label_slot for each leaf.
        let imported_label_args = self
            .imported_label_args
            .as_ref()
            .expect("install_imported_virtuals requires imported_label_args");
        let mut label_slot = 0usize;
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!(
                "[jit] install_virt: label_slot={} label_args_len={} label_args={:?}",
                label_slot,
                imported_label_args.len(),
                imported_label_args
            );
        }
        // RPython Box identity parity: Two-pass approach.
        //
        // Pass 1: process all fields with import_virtual_state_from_label_args.
        //   Constant fields get alloc_op_position. Leaf fields get LABEL arg OpRefs.
        //   Record which leaf field positions need SameAs (skip_flush_mode = Phase 2).
        //
        // Pass 2: allocate SameAs positions and create ops for recorded leaves.
        //   This happens AFTER all Constant allocs, preventing reserve_pos collision
        //   with Phase 1 virtual head positions.
        //
        // Without SameAs, the assembly's body_result_remap maps fail_args entries
        // from LABEL arg positions to body op fresh positions, causing virtual fields
        // to get body IntAddOvf results instead of LABEL values.
        struct VirtualEntry {
            head: OpRef,
            size_descr: majit_ir::DescrRef,
            fields: Vec<(u32, OpRef)>,
            kind: ImportedVirtualKind,
            head_load_descr_index: Option<u32>,
        }
        let mut entries: Vec<VirtualEntry> = Vec::new();
        // Track leaf field positions that need SameAs (position, entry_idx, field_idx).
        let mut same_as_targets: Vec<(OpRef, usize, usize)> = Vec::new();

        // RPython parity: traverse ALL states in order (not just virtuals).
        // Non-virtual states advance label_slot without creating entries.
        // Virtual states create entries with fields from label_args.
        // Build a map from inputarg_index to imported_virtual for virtual lookup.
        let mut iv_map: crate::FxIndexMap<usize, &ImportedVirtual> = crate::FxIndexMap::default();
        for iv in &self.imported_virtuals {
            iv_map.insert(iv.inputarg_index, iv);
        }
        let all_states = self
            .imported_loop_state
            .as_ref()
            .map(|s| &s.virtual_state.state[..])
            .unwrap_or(&[]);
        // virtualstate.py `enum` parity: a single visited map
        // tracked via `Rc::as_ptr` dedups shared subtrees across all
        // top-level state entries so `label_slot` advances by the same
        // number of leaves as `imported_label_args.len()`.
        // The map value caches the imported Phase 2 box for the first
        // visit so subsequent revisits resolve to the same box (mirroring
        // RPython's setinfo_from_preamble.get_forwarded sharing).
        let mut walk_visited: crate::FxIndexMap<usize, Operand> = crate::FxIndexMap::default();
        for (state_idx, state_info) in all_states.iter().enumerate() {
            if let Some(iv) = iv_map.get(&state_idx).copied() {
                // Virtual state: process fields recursively, consuming slots
                // for non-virtual leaf fields.
                //
                // Top-level Rc dedup: if two top-level state entries are
                // aliased (same Rc), the second visit should not advance
                // label_slot or allocate a new VirtualEntry. Mark the
                // top-level Rc identity as visited so subsequent revisits
                // (including nested-field references back to the same Rc)
                // short-circuit through walk_visited.
                let top_key = std::rc::Rc::as_ptr(state_info) as usize;
                if walk_visited.contains_key(&top_key) {
                    continue;
                }
                walk_visited.insert(top_key, Operand::None);
                //
                // `iv.inputarg_index` is the virtualizable state's index
                // within `next_iteration_args`, which corresponds to
                // Phase 2 inputarg slot i (not to a raw OpRef). Shift
                // by `ctx.inputarg_base` so the OpRef we resolve points
                // to Phase 2's fresh inputarg slot in the disjoint
                // `[inputarg_base..inputarg_base+num_inputs)` range
                // (matches RPython's TraceIterator fresh-InputArg model).
                // For Phase 1 / standalone runs `inputarg_base == 0`, so
                // the shift is a no-op and the legacy raw-position path
                // still works.
                //
                // resoperation.py InputArgInt/727/739 InputArg{Int,Ref,Float}: mint a
                // typed variant from `inputarg_types` so the OpRef carries
                // RPython `box.type` (history.py:220) and matches the
                // typed `OpRef::input_arg_typed` minted at trace start
                // (pyre/pyre-jit-trace/src/trace.rs) under variant-aware Eq.
                // opencoder.py inputarg_from_tp(arg.type) parity:
                // every inputarg always carries `box.type` (history.py:220);
                // RPython has no InputArgVoid class. Strict accessor
                // panics on missing/Void → exposes structural bookkeeping
                // bugs instead of silently minting an InputArgVoid-shaped
                // VoidOp variant.
                let pos = ctx.inputarg_base + iv.inputarg_index as u32;
                let raw =
                    OpRef::input_arg_typed(pos, ctx.inputarg_type_at_strict(iv.inputarg_index));
                let virtual_head = ctx.get_replacement_opref(raw);
                let virtual_head_box = ctx
                    .get_box_replacement_operand_opt(raw)
                    .unwrap_or(Operand::None);
                walk_visited.insert(top_key, virtual_head_box);
                let mut fields = Vec::new();
                for (descr, field_info) in &iv.fields {
                    let field_box = Self::import_virtual_state_from_label_args(
                        field_info,
                        imported_label_args,
                        &mut label_slot,
                        ctx,
                        &mut walk_visited,
                    );
                    let field_ref = field_box.to_opref();
                    let field_idx = fields.len();
                    let field_is_virtual = ctx
                        .get_box_replacement_operand_opt(field_ref)
                        .as_ref()
                        .is_some_and(|b| ctx.is_virtual(b));
                    let field_is_const = ctx
                        .get_box_replacement_operand_opt(field_ref)
                        .and_then(|cb| cb.const_value())
                        .is_some();
                    if ctx.skip_flush_mode
                        && !field_ref.is_none()
                        && !field_is_const
                        && !field_is_virtual
                    {
                        same_as_targets.push((field_ref, entries.len(), field_idx));
                    }
                    let field_idx = descr
                        .as_field_descr()
                        .map(|fd| {
                            crate::optimizeopt::virtualize::parent_list_slot_in(
                                Some(&iv.size_descr),
                                fd,
                            )
                        })
                        .unwrap_or_else(|| descr.index());
                    fields.push((field_idx, field_ref));
                    let _ = descr; // descr threading handled via SizeDescr.all_fielddescrs()
                }
                entries.push(VirtualEntry {
                    head: virtual_head,
                    size_descr: iv.size_descr.clone(),
                    fields,
                    kind: iv.kind.clone(),
                    head_load_descr_index: iv.head_load_descr_index,
                });
            } else {
                // Non-virtual state: advance label_slot to match RPython's
                // position_in_notvirtuals enumeration order. Top-level Rc
                // dedup is handled inside count_forced_boxes_for_entry_static.
                let count = crate::optimizeopt::virtualstate::VirtualState::count_forced_boxes_for_entry_static(
                    state_info,
                    &mut walk_visited,
                );
                label_slot += count;
            }
        }

        // Pass 2: allocate SameAs ops for leaf field values.
        // Advance next_pos past all virtual head positions to prevent
        // reserve_pos from returning a position that's already used
        // as a virtual head (allocated during import_state).
        for entry in &entries {
            if !entry.head.is_none()
                && ctx
                    .get_box_replacement_operand_opt(entry.head)
                    .and_then(|cb| cb.const_value())
                    .is_none()
            {
                ctx.next_pos = ctx.next_pos.max(entry.head.raw() + 1);
            }
        }
        for (label_arg, entry_idx, field_idx) in &same_as_targets {
            // RPython parity: Box.type carries the source type intrinsically.
            // value_types[label_arg] is whatever the optimizer's emit() last
            // recorded for that OpRef. A Type::Void here would be an upstream
            // bookkeeping bug (a guard's pos collided with a Box position):
            // letting it through reaches `same_as_for_type(Void)`, which is
            // `unreachable!` and panics with a clear message.
            let tp = ctx
                .opref_type(*label_arg)
                .expect("imported virtual leaf missing box.type");
            let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
            let arg0 = ctx.materialize_operand_at(*label_arg);
            let mut op = majit_ir::Op::new(same_as_op, std::slice::from_ref(&arg0));
            op.pos().set(ctx.reserve_pos_typed(tp));
            let fresh = op.pos().get();
            // Op.type_ carries `tp` intrinsically (resoperation.py:1693
            // SAME_AS_*.type parity); the immediate push below makes
            // op_at(fresh) the authoritative type source. No
            // `value_types` write needed (5).
            let op_rc = OpRc::new(op);
            ctx.emitted_operations
                .insert(majit_ir::operand::Operand::from_bound_op(&op_rc));
            ctx.push_new_operation(op_rc);
            // Update the field to reference the SameAs result.
            entries[*entry_idx].fields[*field_idx].1 = fresh;
        }

        // Install PtrInfo for each virtual.
        // unroll.py:55: if op.get_forwarded() is not None: return
        // Skip heads that already have PtrInfo (duplicate entries from
        // aliased JUMP args sharing the same VirtualState position).
        // Keyed by the virtual head's producer identity (`Operand` ptr_eq).
        // A head is normally a NEW (virtual-alloc) ResOp with a producer, so
        // `get_box_replacement_operand_opt` returns the memoized bound box and two
        // entries sharing a head dedupe. A producer-less head resolves to
        // None and is never dedupable (matching the prior fresh-unbound-box
        // behaviour); its `set_ptr_info` is skipped below anyway.
        let mut installed_heads: crate::FxIndexSet<majit_ir::operand::Operand> =
            crate::FxIndexSet::default();
        for entry in entries {
            let head_box = ctx.get_box_replacement_operand_opt(entry.head);
            if let Some(hk) = &head_box {
                let head_key = hk.clone();
                if installed_heads.contains(&head_key) {
                    continue;
                }
                installed_heads.insert(head_key);
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] install_imported_virtual head={:?} fields={:?}",
                    entry.head, entry.fields
                );
            }
            match &entry.kind {
                ImportedVirtualKind::Instance { known_class } => {
                    if let Some(b) = &head_box {
                        let fields: majit_ir::ptr_info::VirtualFieldList = entry
                            .fields
                            .iter()
                            .map(|(i, r)| (*i, ctx.materialize_operand_at(*r)))
                            .collect();
                        ctx.set_ptr_info(
                            b,
                            crate::optimizeopt::info::PtrInfo::Virtual(
                                crate::optimizeopt::info::VirtualInfo {
                                    descr: entry.size_descr,
                                    known_class: *known_class,
                                    ob_type_descr: None,
                                    fields,
                                    last_guard_pos: -1,
                                    avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                                },
                            ),
                        );
                    }
                }
                ImportedVirtualKind::Struct => {
                    if let Some(b) = &head_box {
                        let fields: majit_ir::ptr_info::VirtualFieldList = entry
                            .fields
                            .iter()
                            .map(|(i, r)| (*i, ctx.materialize_operand_at(*r)))
                            .collect();
                        ctx.set_ptr_info(
                            b,
                            crate::optimizeopt::info::PtrInfo::VirtualStruct(
                                crate::optimizeopt::info::VirtualStructInfo {
                                    descr: entry.size_descr,
                                    fields,
                                    last_guard_pos: -1,
                                    avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                                },
                            ),
                        );
                    }
                }
            }
            // RPython parity: imported virtual heads are NOT looked up via
            // a side table. inline_short_preamble replays the getfield ops
            // through send_extra_operation, which populates OptHeap's cache.
            // The body's getfield then folds naturally.
            let _ = entry.head_load_descr_index; // consumed by shortpreamble replay
        }
    }

    /// virtualstate.py `VirtualStateConstructor.create_state` cache
    /// parity for the import side: dedup nested `Rc<VirtualStateInfo>`
    /// references via pointer identity, returning the previously imported
    /// Phase 2 box on revisits so shared substates collapse onto a
    /// single Phase 2 box (matching RPython's setinfo_from_preamble
    /// `if op.get_forwarded() is not None: return` semantics).
    fn import_virtual_state_from_label_args_recurse(
        rc: &std::rc::Rc<crate::optimizeopt::virtualstate::VirtualStateInfoNode>,
        imported_label_args: &[OpRef],
        label_slot: &mut usize,
        ctx: &mut OptContext,
        walk_visited: &mut crate::FxIndexMap<usize, Operand>,
    ) -> Operand {
        let key = std::rc::Rc::as_ptr(rc) as usize;
        if let Some(cached) = walk_visited.get(&key) {
            return cached.clone();
        }
        // Insert a placeholder NONE to break cycles, then overwrite with
        // the real box once import_virtual_state_from_label_args returns.
        walk_visited.insert(key, Operand::None);
        let opref = Self::import_virtual_state_from_label_args(
            &rc.info,
            imported_label_args,
            label_slot,
            ctx,
            walk_visited,
        );
        walk_visited.insert(key, opref.clone());
        opref
    }

    fn import_virtual_state_from_label_args(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        imported_label_args: &[OpRef],
        label_slot: &mut usize,
        ctx: &mut OptContext,
        walk_visited: &mut crate::FxIndexMap<usize, Operand>,
    ) -> Operand {
        use crate::optimizeopt::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(_) => Self::import_virtual_state_value(info, ctx),
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                ob_type_descr,
                fields,
                field_descrs,
            } => {
                // unroll.py:454 Box carries its type. Virtual heads are
                // Ref-typed. PtrInfo presence alone cannot stand in for
                // box.type because PtrInfo can also describe int-typed
                // raw pointers (info.py RawBufferPtrInfo +
                // getrawptrinfo()). Bind the head's canonical host at
                // allocation so the PtrInfo write below lands
                // unconditionally — a bare position resolves to `None` and
                // would silently drop the imported virtual-ness.
                let (_, head_box) = ctx.reserve_virtual_box(majit_ir::Type::Ref);
                let imported_fields: majit_ir::ptr_info::VirtualFieldList = fields
                    .iter()
                    .map(|(field_idx, field_info)| {
                        let field_ref = Self::import_virtual_state_from_label_args_recurse(
                            field_info,
                            imported_label_args,
                            label_slot,
                            ctx,
                            walk_visited,
                        );
                        // ob_type (offset 0) class pointers are already Ref-typed
                        // by `import_virtual_state_value` (typed `RefOp` variant
                        // tag) + a Ref-typed `Forwarded::Const`
                        // (`Value::Ref(class_gcref)`) from `make_constant`. The
                        // Ref-typed const forwarding is the authoritative shape.
                        let _ = (field_descrs, known_class, field_idx);
                        (*field_idx, field_ref)
                    })
                    .collect();
                let _ = field_descrs; // descr.all_fielddescrs() is authoritative
                ctx.set_ptr_info(
                    &head_box,
                    crate::optimizeopt::info::PtrInfo::Virtual(
                        crate::optimizeopt::info::VirtualInfo {
                            descr: descr.clone(),
                            known_class: *known_class,
                            ob_type_descr: ob_type_descr.clone(),
                            fields: imported_fields,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
                head_box
            }
            VirtualStateInfo::VArray { descr, items, .. } => {
                // unroll.py:454 Box carries its type. VArray heads are
                // Ref-typed. Bound at allocation via `reserve_virtual_box`
                // so the `set_ptr_info` write below cannot land on a bare
                // position and silently drop the imported virtual-ness.
                let (_, head_box) = ctx.reserve_virtual_box(majit_ir::Type::Ref);
                let imported_items = items
                    .iter()
                    .map(|item_info| match item_info {
                        Some(item_info) => {
                            let item_ref = Self::import_virtual_state_from_label_args_recurse(
                                item_info,
                                imported_label_args,
                                label_slot,
                                ctx,
                                walk_visited,
                            );
                            item_ref
                        }
                        // virtualstate.py:272-280: absent fieldstate remains
                        // an unwritten virtual-array slot on import.
                        None => Operand::None,
                    })
                    .collect();
                ctx.set_ptr_info(
                    &head_box,
                    crate::optimizeopt::info::PtrInfo::VirtualArray(
                        crate::optimizeopt::info::VirtualArrayInfo {
                            descr: descr.clone(),
                            clear: false,
                            items: imported_items,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
                head_box
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                // unroll.py:454 Box carries its type. VStruct heads are
                // Ref-typed. Bound at allocation via `reserve_virtual_box`
                // so the `set_ptr_info` write below cannot land on a bare
                // position and silently drop the imported virtual-ness.
                let (_, head_box) = ctx.reserve_virtual_box(majit_ir::Type::Ref);
                let imported_fields = fields
                    .iter()
                    .map(|(field_idx, field_info)| {
                        let r = Self::import_virtual_state_from_label_args_recurse(
                            field_info,
                            imported_label_args,
                            label_slot,
                            ctx,
                            walk_visited,
                        );
                        (*field_idx, r)
                    })
                    .collect();
                let _ = field_descrs; // descr.all_fielddescrs() is authoritative
                ctx.set_ptr_info(
                    &head_box,
                    crate::optimizeopt::info::PtrInfo::VirtualStruct(
                        crate::optimizeopt::info::VirtualStructInfo {
                            descr: descr.clone(),
                            fields: imported_fields,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
                head_box
            }
            VirtualStateInfo::VArrayStruct {
                descr,
                fielddescrs,
                element_fields,
            } => {
                // unroll.py:454 Box carries its type. VArrayStruct heads
                // are Ref-typed. Bound at allocation via
                // `reserve_virtual_box` so the `set_ptr_info` write below
                // cannot land on a bare position and silently drop the
                // imported virtual-ness.
                let (_, head_box) = ctx.reserve_virtual_box(majit_ir::Type::Ref);
                let imported_elements = element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_info)| {
                                let field = match field_info {
                                    Some(field_info) => {
                                        let field_ref =
                                            Self::import_virtual_state_from_label_args_recurse(
                                                field_info,
                                                imported_label_args,
                                                label_slot,
                                                ctx,
                                                walk_visited,
                                            );
                                        field_ref
                                    }
                                    // virtualstate.py _enum: retain the
                                    // dense unwritten element-field slot.
                                    None => Operand::None,
                                };
                                (*field_idx, field)
                            })
                            .collect()
                    })
                    .collect();
                ctx.set_ptr_info(
                    &head_box,
                    crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(
                        crate::optimizeopt::info::ArrayStructInfo {
                            descr: descr.clone(),
                            fielddescrs: fielddescrs.clone(),
                            element_fields: imported_elements,
                            last_guard_pos: -1,
                            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                        },
                    ),
                );
                head_box
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {
                let opref = imported_label_args
                    .get(*label_slot)
                    .copied()
                    .unwrap_or_else(|| {
                        if crate::majit_log_enabled() {
                            eprintln!(
                                "[jit] MISS: label_slot={} len={}",
                                *label_slot,
                                imported_label_args.len()
                            );
                        }
                        OpRef::NONE
                    });
                *label_slot += 1;
                // Imported label-arg leaf (KnownClass / NonNull / IntBounded
                // / Unknown): resolve the already-materialized box for this
                // label slot and write the imported state through it. An
                // unresolvable slot (`OpRef::NONE`, e.g. the MISS fallback
                // above) yields `None` and the write no-ops — matching the
                // prior `materialize_operand_at(OpRef::NONE) -> None` behavior.
                let resolved = ctx.get_box_replacement_operand_opt(opref);
                if let Some(box_) = &resolved {
                    Self::apply_imported_virtual_state(info, box_, ctx);
                }
                resolved.unwrap_or(Operand::None)
            }
        }
    }

    pub fn new() -> Self {
        Optimizer {
            supports_efficient_uint_mul_high: true,
            passes: Vec::new(),
            pureop_historylength: crate::jit::PARAMETERS.pureop_historylength as usize,
            final_num_inputs: 0,
            call_pure_results: crate::optimizeopt::util::ArgsDict::default(),
            last_guard_op_idx: None,
            replaces_guard: crate::FxIndexMap::default(),
            pendingfields: Vec::new(),
            can_replace_guards: true,
            quasi_immutable_deps: Vec::new(),
            imported_virtuals: Vec::new(),
            trace_inputargs: Vec::new(),
            runtime_boxes: Vec::new(),
            exported_loop_state: None,
            imported_loop_state: None,
            imported_short_aliases: Vec::new(),
            imported_short_preamble: None,
            imported_short_preamble_builder: None,
            short_preamble_producer: None,
            published_short_preamble_producer_slot: None,
            imported_label_args: None,
            patchguardop: None,
            skip_flush: false,
            terminal_op: None,
            final_ctx: None,
            recycled_ctx: None,
            pending_bridge_rd: None,
            building_bridge: false,
            simple_compile: false,
            all_descrs: Arc::new(Vec::new()),
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            callinfocollection: None,
            resumedata_memo: std::rc::Rc::new(std::cell::RefCell::new(
                crate::resume::ResumeDataLoopMemo::new(),
            )),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: SnapshotFrameSizes::new(),
            snapshot_vable_boxes: Vec::new(),
            // resume.py:401-402: `-1` for a jitdriver with no virtualizable.
            minimum_virtualizable_size: -1,
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: SnapshotFramePcs::new(),
            byte_bridge_resume: None,
            phase1_emit_ops: Vec::new(),
            opt_ops_emitted: 0,
            opt_guards_emitted: 0,
            opt_guards_shared_emitted: 0,
            cpu: crate::cpu::default_cpu(),
            explicit_input_ops_seed: None,
            trace_inputarg_boxes: Vec::new(),
        }
    }

    /// Keep the pass boxes and `ResumeDataLoopMemo` scratch, drop the
    /// per-compile graph. RPython `BridgeCompileData.optimize` constructs a
    /// new `UnrollOptimizer` per compile; nursery allocation is cheap there.
    pub fn recycle_for_next_compile(&mut self) {
        self.final_num_inputs = 0;
        self.call_pure_results = crate::optimizeopt::util::ArgsDict::default();
        self.last_guard_op_idx = None;
        self.replaces_guard.clear();
        self.pendingfields.clear();
        self.can_replace_guards = true;
        self.quasi_immutable_deps.clear();
        self.imported_virtuals.clear();
        self.trace_inputargs.clear();
        self.runtime_boxes.clear();
        self.exported_loop_state = None;
        self.imported_loop_state = None;
        self.imported_short_aliases.clear();
        self.imported_short_preamble = None;
        self.imported_short_preamble_builder = None;
        self.short_preamble_producer = None;
        self.published_short_preamble_producer_slot = None;
        self.imported_label_args = None;
        self.patchguardop = None;
        self.skip_flush = false;
        self.terminal_op = None;
        if let Some(ctx) = self.final_ctx.take() {
            self.stash_recycled_ctx(ctx);
        }
        self.pending_bridge_rd = None;
        self.building_bridge = false;
        self.simple_compile = false;
        self.all_descrs = Arc::new(Vec::new());
        self.snapshot_boxes = Vec::new();
        self.snapshot_frame_sizes.clear();
        self.snapshot_vable_boxes = Vec::new();
        self.snapshot_vref_boxes = Vec::new();
        self.snapshot_frame_pcs.clear();
        self.byte_bridge_resume = None;
        self.phase1_emit_ops.clear();
        self.opt_ops_emitted = 0;
        self.opt_guards_emitted = 0;
        self.opt_guards_shared_emitted = 0;
        self.explicit_input_ops_seed = None;
        self.trace_inputarg_boxes.clear();
        self.resumedata_memo.borrow_mut().recycle_for_next_compile();
        for pass in &mut self.passes {
            pass.reset_between_compiles();
        }
    }

    fn stash_recycled_ctx(&mut self, mut ctx: OptContext) {
        ctx.reset_keep_capacity(0, 0, 0, 0);
        self.recycled_ctx = Some(Box::new(ctx));
    }

    fn take_opt_context(
        &mut self,
        estimated_ops: usize,
        num_inputs: usize,
        inputarg_base: u32,
        start_next_pos: u32,
    ) -> OptContext {
        if let Some(mut ctx) = self.recycled_ctx.take() {
            ctx.reset_keep_capacity(estimated_ops, num_inputs, inputarg_base, start_next_pos);
            return *ctx;
        }
        OptContext::with_num_inputs_and_start_pos(
            estimated_ops,
            num_inputs,
            inputarg_base,
            start_next_pos,
        )
    }

    /// Record a CALL_PURE result for cross-iteration constant folding.
    /// RPython pyjitpl.py:3572: `arg_consts = [executor.constant_from_op(a) ...]`
    /// RPython optimizer.py: `get_constant_box(arg)` → value-based keys.
    pub fn record_call_pure_result(&mut self, args: Vec<majit_ir::Value>, value: majit_ir::Value) {
        self.call_pure_results.insert(args, value);
    }

    /// Look up a previously recorded CALL_PURE result.
    pub fn get_call_pure_result(&self, args: &[majit_ir::Value]) -> Option<majit_ir::Value> {
        self.call_pure_results.get(args)
    }

    /// bridgeopt.py: deserialize_optimizer_knowledge
    /// bridgeopt.py optimizer.optheap.deserialize_optheap(result_struct, result_array)
    pub fn import_heap_knowledge(
        &mut self,
        result_struct: &[(OpRef, majit_ir::DescrRef, OpRef)],
        result_array: &[(OpRef, i64, majit_ir::DescrRef, OpRef)],
        ctx: &mut OptContext,
    ) {
        if !result_struct.is_empty() {
            for pass in &mut self.passes {
                pass.import_cached_fields(result_struct, ctx);
            }
        }
        if !result_array.is_empty() {
            for pass in &mut self.passes {
                pass.import_cached_arrayitems(result_array, ctx);
            }
        }
    }

    /// bridgeopt.py:184-185 optimizer.optrewrite.deserialize_optrewrite(result_loopinvariant)
    pub fn import_loopinvariant_knowledge(&mut self, result_loopinvariant: &[(i64, OpRef)]) {
        for pass in &mut self.passes {
            pass.deserialize_optrewrite(result_loopinvariant);
        }
    }

    /// Get the final num_inputs after optimization.
    /// May be larger than the original if virtualizable added virtual input args.
    pub fn final_num_inputs(&self) -> usize {
        self.final_num_inputs
    }

    /// optimizer.py: getlastop() — index of the last emitted guard.
    /// Resolved through `ctx.new_operations[idx]` by callers.
    pub fn last_guard_op_idx(&self) -> Option<usize> {
        self.last_guard_op_idx
    }

    /// optimizer.py replace_guard: `self.replaces_guard[op] = value.last_guard_pos`.
    /// Record that a guard at the given position should be replaced when the
    /// future condition is realized. PyPy keys `replaces_guard` by the raw `op`
    /// object identity, and `_emit_operation` (optimizer.py) looks it up by
    /// the raw `orig_op` — both before `get_box_replacement`. `resolve_to_operand`
    /// yields that producer box (chain root, before `_forwarded`), so insert and
    /// the emit-time lookup compare the same raw box.
    /// (The method keeps its historical name; the body performs replace_guard's
    /// insert.)
    pub fn notice_guard_future_condition(
        &mut self,
        ctx: &OptContext,
        guard_pos: OpRef,
        replacement: Op,
    ) {
        if let Some(op) = ctx.resolve_to_operand(guard_pos) {
            self.replaces_guard.insert(op, replacement);
        }
    }

    /// optimizer.py: replace_guard_op(old_op_pos, new_op)
    /// Replace a previously emitted guard with a new one.
    ///
    /// optimizer.py: `self._emittedoperations[new_op] = None` —
    /// the new guard takes over the emit identity, so it must enter
    /// the emit set even though it was substituted post-hoc rather
    /// than directly emitted via `_emit_operation`.
    pub fn replace_guard_op(&mut self, ctx: &mut OptContext, old_pos: OpRef, new_guard: Op) {
        let new_pos = new_guard.pos().get();
        // replaces_guard is keyed by the raw `op` identity (optimizer.py:307),
        // so resolve to the producer box without following `_forwarded`.
        if let Some(op) = ctx.resolve_to_operand(old_pos) {
            self.replaces_guard.insert(op, new_guard);
        }
        // optimizer.py `self._emittedoperations[new_op] = None` — new_op is
        // the canonical (get_box_replacement'd) emitted op, so this insert stays
        // canonical, matching the emit-set keying in `_emit_operation`.
        let new_op = ctx.get_box_replacement_operand(new_pos);
        ctx.emitted_operations.insert(new_op);
    }

    /// optimizer.py `as_operation(op, required_opnum=-1)`:
    ///
    /// ```python
    /// def as_operation(self, op, required_opnum=-1):
    ///     if isinstance(op, AbstractResOp):
    ///         if required_opnum != -1 and op.opnum != required_opnum:
    ///             return None
    ///         if op in self._emittedoperations:
    ///             return op
    ///     return None
    /// ```
    ///
    /// Returns `Some(opref)` iff the opref refers to an actually-emitted
    /// op (in `ctx.emitted_operations`) and its opcode matches
    /// `required_opnum` (or `required_opnum` is `None` meaning "any
    /// opcode"). Callers verify identity before reasoning about
    /// emit-bound metadata.
    pub fn as_operation(
        &self,
        opref: OpRef,
        required_opnum: Option<majit_ir::OpCode>,
        ctx: &OptContext,
    ) -> Option<OpRef> {
        if let Some(required) = required_opnum
            && ctx.op_at(opref).map(|op| op.opcode) != Some(required)
        {
            return None;
        }
        // optimizer.py `if op in self._emittedoperations` keys by the op's
        // own (raw) identity, not its forwarded replacement. `resolve_to_operand`
        // is that producer box (chain root, before `_forwarded`); `ctx.emit_rc`
        // records the same op with `Operand::from_bound_op`.
        // A constant is never an emitted op; short-circuit before resolving so
        // resolving never mints a throwaway Const key (which would harmlessly
        // miss anyway, since no Const is ever inserted into the emit set).
        if opref.is_constant() {
            return None;
        }
        match ctx.resolve_to_operand(opref) {
            Some(op) if ctx.emitted_operations.contains(&op) => Some(opref),
            _ => None,
        }
    }

    // RPython optimizer.py store_final_boxes_in_guard and
    // optimizer.py emit_guard_operation are implemented inside
    // emit_operation: _copy_resume_data_from, store_final_boxes_in_guard,
    // force_box on fail_args, and store_final_boxes_in_guard in ctx.emit().

    /// optimizer.py: add_pending_field(op)
    /// Queue a SETFIELD_GC to be emitted before the next guard.
    pub fn add_pending_field(&mut self, op: Op) {
        self.pendingfields.push(op);
    }

    /// optimizer.py: flush_pendingfields(ctx)
    /// Emit all pending field writes.
    pub fn flush_pendingfields(&mut self, ctx: &mut OptContext) {
        let pending = std::mem::take(&mut self.pendingfields);
        for op in pending {
            ctx.emit(op);
        }
    }

    /// optimizer.py: has_pending_fields()
    pub fn has_pending_fields(&self) -> bool {
        !self.pendingfields.is_empty()
    }

    /// optimizer.py: num_pending_fields()
    pub fn num_pending_fields(&self) -> usize {
        self.pendingfields.len()
    }

    /// optimizer.py + :899-909 `cant_replace_guards()` returns a
    /// `CantReplaceGuards` context manager whose `__enter__` saves
    /// `self.optimizer.can_replace_guards` into `self.oldval` and sets
    /// the flag to False; `__exit__` restores from `self.oldval`.
    ///
    /// pyre returns the `CantReplaceGuards` token carrying the previously-set
    /// value. Callers pair this with a manual restore. This matches the
    /// upstream save-old-then-set-false semantics exactly — including the
    /// nested case where an outer scope has already set the flag to False and
    /// the inner restore must preserve that.
    ///
    /// ```text
    /// let guard = optimizer.cant_replace_guards(ctx);
    /// // ... guarded section ...
    /// optimizer.restore_can_replace_guards(ctx, guard);
    /// ```
    pub fn cant_replace_guards(&mut self, ctx: &mut OptContext) -> CantReplaceGuards {
        debug_assert_eq!(self.can_replace_guards, ctx.can_replace_guards);
        let oldval = self.can_replace_guards;
        self.can_replace_guards = false;
        ctx.can_replace_guards = false;
        CantReplaceGuards::new(oldval)
    }

    /// Pair with `cant_replace_guards` — restores the saved oldval.
    /// Matches `CantReplaceGuards.__exit__` (optimizer.py).
    pub fn restore_can_replace_guards(&mut self, ctx: &mut OptContext, guard: CantReplaceGuards) {
        self.can_replace_guards = guard.oldval;
        ctx.can_replace_guards = guard.oldval;
    }

    /// `optimizer.py:243` + `heap.py:821-823`
    /// `self.quasi_immutable_deps[qmutdescr.qmut] = None`. Vec-backed
    /// set with linear-scan dedup.
    pub fn add_quasi_immutable_dep(&mut self, dep: std::sync::Arc<dyn majit_ir::QuasiImmutHandle>) {
        if !self
            .quasi_immutable_deps
            .iter()
            .any(|seen| seen.instance_identity() == dep.instance_identity())
        {
            self.quasi_immutable_deps.push(dep);
        }
    }

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Collect short preamble ops from all passes.
    pub fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        for pass in &self.passes {
            pass.produce_potential_short_preamble_ops(sb, ctx);
        }
    }

    /// Publish `ctx.preview_short_state` from the post-force jump args.
    ///
    /// unroll.py `export_state` (463-488): virtual_state, label args / virtuals,
    /// ShortBoxes, `produce_potential_short_preamble_ops`, produced ops, the
    /// PreambleOp list. The loop path calls this at the site the preview used
    /// to run inline; a bridge calls it only in `optimize_bridge`'s retrace
    /// arm, immediately before the late `unroll::export_state`.
    fn publish_preview_short_state(
        &mut self,
        mut ctx: &mut OptContext,
        resolved_args: &[OpRef],
    ) -> Result<(), crate::optimize::InvalidLoop> {
        // unroll.py `virtual_state = self.get_virtual_state(end_args)`.
        // VS is captured AFTER force + flush so its `Virtual` /
        // `VStruct` entries match the `info.is_virtual()` predicate
        // that `enum_forced_boxes` asserts. Virtuals that were
        // forced into concrete instances by `force_at_the_end_of_preamble`
        // come back as `NotVirtualStateInfo` here, exactly as they
        // do in RPython.
        let post_force_args: Vec<OpRef> = resolved_args
            .iter()
            .map(|&a| {
                let resolved = ctx.get_replacement_opref(a);
                // bind-at-alloc: `export_state` below keys its
                // `ExportCache` by these resolved positions. A producer-less
                // value-bearing position resolves to a throwaway `from_opref`
                // operand, so each `bound_from_opref` fallback returns a
                // distinct Rc (ptr-Eq-unstable) and `export_single_value` logs it as an
                // unbound export key. Bind the canonical `_forwarded` host
                // once via `materialize_operand_at` (a `SameAs*` synthetic
                // in `resop_refs`) so the
                // export key resolves to one ptr-stable host — the identity
                // the #188 `OpRef`→operand `ExportCache` rekey requires. The
                // returned `OpRef` is unchanged (synthetic and orphan share
                // the position), so the exported state is byte-identical.
                if !resolved.is_none()
                    && !resolved.is_constant()
                    && ctx.get_box_replacement_operand_opt(resolved).is_none()
                {
                    ctx.materialize_operand_at(resolved);
                }
                resolved
            })
            .collect();
        // RPython's `end_args` are Box objects and the same objects are
        // carried into `next_iteration_args`.  Preserve those exact
        // operands before ShortBoxes production mutates forwarding;
        // retaining only their OpRef positions loses Box identity.
        let preview_end_arg_boxes: Vec<majit_ir::operand::Operand> = post_force_args
            .iter()
            .map(|&arg| {
                ctx.get_box_replacement_operand_opt(arg)
                    .unwrap_or_else(|| ctx.materialize_operand_at(arg))
            })
            .collect();
        let preview_virtual_state =
            crate::optimizeopt::virtualstate::export_state_operands(&preview_end_arg_boxes, &ctx);
        let vs_args = &post_force_args;
        // virtualstate.py:687-689 / unroll.py:154-158: a virtual-state
        // mismatch here raises `VirtualStatesCantMatch` and the outer
        // `compile_loop_body` catches it as an `InvalidLoop` to skip
        // jump-to-existing and either retrace or fall back to the
        // interpretive path. Propagate via the `InvalidLoop` panic
        // payload so the existing wrapper
        // (`unroll.rs`'s
        // `optimize_trace_with_constants_and_inputs_vable_out`) catches
        // and reroutes instead of crashing the worker thread.
        let (preview_label_args, preview_virtuals, preview_label_source_positions) =
            match preview_virtual_state
                .make_inputargs_and_virtuals_with_source_positions(vs_args, self, &mut ctx, false)
            {
                Ok(pair) => {
                    let (label_boxes, virtuals, positions) = pair;
                    (
                        label_boxes.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
                        virtuals,
                        positions,
                    )
                }
                Err(_) => {
                    // unroll.py:193,207-210: on the BRIDGE path the
                    // short-preamble/export preview does not exist — VS
                    // matching is deferred to `jump_to_existing_trace`,
                    // which catches `VirtualStatesCantMatch` and falls back
                    // to `jump_to_preamble`. Do not surface the preview
                    // mismatch as a fatal InvalidLoop; leave
                    // exported_loop_state = None (the flush + force above
                    // already committed the bridge's heap writebacks into
                    // `ctx.new_operations`) and let `optimize_bridge`'s own
                    // ladder route the bridge to the always-matching
                    // preamble target (unroll.py:238-242). Only the
                    // loop/peeled-loop path (optimize_peeled_loop
                    // unroll.py:135-145) keeps this fatal.
                    //
                    // Neither arm is reachable today: the preview exports
                    // its state from `post_force_args` and re-matches that
                    // same list, so every `state[i]` was derived from
                    // `args[i]` and the walk is self-consistent. Probed
                    // with five virtual-carrying fixtures (escaping tuple,
                    // escaping instance, aliased list, varying-length
                    // array, nested virtual), two of which do compile
                    // bridges — zero hits, as with pyre/bench and
                    // pyre/extra_tests. Upstream matches
                    // against a *different* loop's stored state in
                    // `jump_to_existing_trace` (unroll.py);
                    // `export_state_re_matched_against_its_own_args_cannot_fail`
                    // (virtualstate.rs) pins the self-match, so moving the
                    // preview to the upstream shape breaks that test and
                    // flags this branch as newly live.
                    if ctx.building_bridge {
                        if crate::bridge_debug_enabled() {
                            eprintln!(
                                "[bridgeB] preview virtual-state mismatch — leaving the export empty for the jump_to_existing_trace ladder"
                            );
                        }
                        return Ok(());
                    }
                    return Err(crate::optimize::InvalidLoop(
                        "preview virtual state mismatch (VirtualStatesCantMatch)",
                    ));
                }
            };
        let mut preview_short_args = preview_label_args.clone();
        preview_short_args.extend_from_slice(&preview_virtuals);
        let mut short_boxes =
            crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&preview_short_args);
        for &arg in &preview_short_args {
            // RPython shortpreamble.py:255-259 parity: each label arg
            // is `box.type`, where Box objects intrinsically carry one
            // of i / r / f. There is no `void` Box because Box always
            // wraps a runtime value. Pyre recovers the same type from
            // the typed OpRef variant or the trace's inputarg/op metadata.
            let raw_type = ctx
            .opref_type(arg)
            .unwrap_or_else(|| {
                panic!(
                    "preview short arg missing box.type: arg={arg:?} preview_short_args={preview_short_args:?}"
                )
            });
            if raw_type == majit_ir::Type::Void {
                // shortpreamble.py reads `box.type` from a
                // value Box and emits same_as_i/r/f. RPython has no
                // Void value Box here; reaching Void means Rust-side
                // OpRef/type bookkeeping lost Box identity and must not
                // silently drop the short-preamble inputarg.
                panic!(
                    "preview short arg {arg:?} resolved to Type::Void; \
                 short preamble inputargs must be int/ref/float value boxes \
                 (shortpreamble.py:255-259)"
                );
            }
            short_boxes.add_short_input_arg(&mut ctx, arg, raw_type);
        }
        self.produce_potential_short_preamble_ops(&mut short_boxes, &mut ctx);
        let produced = short_boxes.produced_ops(&mut ctx);
        // shortpreamble.py:272-281 — the `for short_op in
        // self.const_short_boxes:` half of `create_short_boxes`, which
        // upstream runs on every unroll with no condition on it.
        let produced_const = short_boxes.produced_const_ops(&mut ctx);
        // unroll.py:480 `short_inputargs = sb.create_short_inputargs(
        // label_args + virtuals)` — read off the ShortBoxes object and
        // carry to export_state through the ctx channel, together with
        // the rooted InputArgRc pool (index-aligned, so the renamed
        // boxes stay bound to live `InputArg`s across the export
        // boundary instead of shedding to position-only boxes) and the
        // short boxes below. Bound to locals here and published as one
        // `PreviewShortState` at the group site.
        let short_inputargs = short_boxes.create_short_inputargs(&preview_short_args);
        let short_inputarg_refs = short_boxes.create_short_inputarg_refs();
        // unroll.py computes virtual_state, label_args and virtuals
        // once, then builds ShortBoxes from that exact result. Keep
        // the same single evaluation across majit's split preview /
        // export implementation: producing heap facts may mutate Box
        // forwarding, so recomputing hereafter is observably different.
        let args_state = Some((
            preview_virtual_state.clone(),
            preview_label_args.clone(),
            preview_virtuals.clone(),
            preview_label_source_positions,
            preview_end_arg_boxes,
        ));
        // Single-object carry: each exported entry keeps the preview
        // ProducedShortOp's replay Rc, so the pos/arg canonicalization
        // below lands on the object that dep-replay operands reference
        // (upstream exports the ResOperation objects themselves,
        // unroll.py:478-487). The per-entry rewrites require each entry
        // to own a distinct Rc.
        #[cfg(debug_assertions)]
        {
            let mut seen: Vec<*const majit_ir::Op> = Vec::with_capacity(produced.len());
            for (_, p) in &produced {
                let ptr = OpRc::as_ptr(&p.preamble_op);
                debug_assert!(
                    !seen.contains(&ptr),
                    "exported short boxes share a replay OpRc at {:?}",
                    p.preamble_op.pos().get()
                );
                seen.push(ptr);
            }
        }
        let mut convert_produced =
            |result: OpRef,
             produced: crate::optimizeopt::shortpreamble::ProducedShortOp,
             const_group: bool| {
                let canonical_result = ctx.get_replacement_opref(result);
                let replay_result = if const_group {
                    None
                } else {
                    // pure.py `produce_potential_short_preamble_ops` walks
                    // `_newoperations`, not their replacements.  In particular,
                    // `postprocess_GUARD_TRUE` can forward an already-emitted
                    // pure comparison to CONST_1; PyPy still exports that
                    // comparison Box and `PureOp.produce_op` installs it in the
                    // next iteration's pure cache.  Keep the original result
                    // identity for that case.  Using the inline Const as the
                    // replay position would collapse distinct comparison Boxes
                    // that happen to prove the same value.
                    Some(exported_short_box_replay_result(
                        result,
                        canonical_result,
                        &produced.kind,
                    )?)
                };
                let preamble_op = produced.preamble_op.clone();
                // shortpreamble.py keeps two boxes: `short_op.res` (body) and
                // `preamble_op` (replay). `add_op_to_short` already minted the
                // replay result. `exported_short_box_replay_result` only
                // decides whether a forwarded-to-Const entry is kept; it must
                // not be written onto `preamble_op.pos`.
                if replay_result.is_none() && !const_group {
                    return None;
                }
                // optimizer.py force_box loop parity.
                //
                // Resolve POSITIONALLY when a producer is registered at
                // this slot: replay-op args carry the dep replay handle
                // (produce_arg, shortpreamble.py) whose forwarded slot
                // is empty, so only the body producer registered at the
                // same position carries the Phase-1 forwarding to the
                // canonical end box this export boundary needs.
                //
                // When positional resolution finds NO producer, the carried
                // handle is unforwarded and resolves to itself
                // (resoperation.py:57-68): keep the handle OBJECT instead of
                // re-minting a producer-less position-only box, so its
                // identity (and the `Operand::Op`/`InputArg` shed) survives
                // the export. The encoded OpRef is identical either way
                // (`from_opref(arg.to_opref()) == arg.to_opref()`).
                for i in 0..preamble_op.num_args() {
                    let arg = preamble_op.arg(i);
                    // The `OpRef::none()` sentinel has no producer box
                    // (`materialize_operand_at` doc) — routing it through the
                    // producer lookup is meaningless and trips the
                    // `get_box_replacement` "box must exist" debug tripwire.
                    if arg.is_none() {
                        continue;
                    }
                    let resolved = ctx
                        .heal_arg_to_canonical(&arg)
                        .or_else(|| ctx.resolve_operand_operand_opt(&arg))
                        .unwrap_or_else(|| arg.clone());
                    preamble_op.setarg(i, resolved);
                }
                if let Some(mut fail_args) = preamble_op.getfailargs() {
                    for arg in fail_args.iter_mut() {
                        if arg.is_none() {
                            continue;
                        }
                        *arg = ctx.get_box_replacement_operand(arg.to_opref());
                    }
                    preamble_op.setfailargs(fail_args);
                }
                // An InputArg label arg whose canonical result forwards away
                // is absent from `short_boxes.label_args`, so a re-lookup of
                // `canonical_result` returns None and the per-slot original
                // is lost. `produced.label_arg_idx` is the slot stamped at
                // `add_short_input_arg` (the slot the renamed
                // `short_inputargs[i]` pairs with — consumed by
                // `slot_to_original`). Non-InputArg entries keep the same
                // stamped field; `inline_short_preamble` does not read it.
                Some(crate::optimizeopt::shortpreamble::PreambleOp {
                    op: preamble_op,
                    source_op: Some(produced.source_op.clone()),
                    // short_op.res travels with the entry as the
                    // exported `PreambleOp.res` operand; the preview
                    // ProducedShortOp already carries the bound producer
                    // / const operand, so it moves across unchanged.
                    res: produced.res.clone(),
                    kind: produced.kind,
                    label_arg_idx: produced.label_arg_idx,
                    invented_name: produced.invented_name,
                    same_as_source: produced.same_as_source.clone(),
                })
            };
        let exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp> = produced
            .into_iter()
            .filter_map(|(result, produced)| convert_produced(result, produced, false))
            .collect();
        let exported_const_short_boxes = produced_const
            .into_iter()
            .filter_map(|produced| convert_produced(produced.res.to_opref(), produced, true))
            .collect();
        ctx.exported_const_short_boxes = exported_const_short_boxes;
        if crate::majit_log_enabled() {
            for entry in &exported_short_boxes {
                // Print args / same_as_source as OpRefs, not via the
                // Operands' derived Debug: a bound InputArg/Op carries a
                // `forwarded` slot whose Debug walks the whole abstract-value
                // graph (fields, descrs, nested ops), dumping tens of MB per
                // box and stalling the run.
                let arg_oprefs: Vec<OpRef> =
                    entry.op.args_slice().iter().map(|a| a.to_opref()).collect();
                eprintln!(
                    "[jit] exported_short_box: kind={:?} pos={:?} opcode={:?} args={:?} descr_idx={:?} invented={} same_as_source={:?}",
                    entry.kind,
                    entry.op.pos().get(),
                    entry.op.opcode,
                    arg_oprefs,
                    entry.op.getdescr().map(|d| d.index()),
                    entry.invented_name,
                    entry.same_as_source.as_ref().map(|o| o.to_opref()),
                );
            }
        }
        // Publish the preview's one evaluation as a single object: the
        // forcing above (and its side effects) stays exactly where it
        // is, only the publication is grouped, so a reader can no
        // longer see one vector from this evaluation next to another
        // that was never written.  Alignment is narrower than that —
        // `short_inputargs` / `short_inputarg_refs` share an index
        // space, `exported_short_boxes` does not.
        ctx.preview_short_state = Some(crate::optimizeopt::PreviewShortState {
            short_inputargs,
            short_inputarg_refs,
            short_boxes: exported_short_boxes,
            args_state,
        });
        Ok(())
    }

    // Pre-tag Phase 1 JUMP arg OpRefs as generation 0.

    // Lock JUMP arg OpRefs so make_equal_to won't forward them.

    /// optimizer.py:557 parity:
    ///
    /// ```python
    /// self.resumedata_memo.update_counters(self.metainterp_sd.profiler)
    /// ```
    ///
    /// RPython invokes `update_counters` at the tail of every
    /// `propagate_all_forward` call, folding accumulated
    /// NVIRTUALS / NVHOLES / NVREUSED from `resumedata_memo` into the
    /// static-data profiler. In pyre the optimizer does not hold a
    /// reference to `MetaInterp.stats`, so this is a thin accessor the
    /// caller invokes after each optimize_loop / optimize_bridge /
    /// propagate_all_forward exit to fold counters into the JIT stats.
    ///
    /// Also folds OPT_OPS / OPT_GUARDS / OPT_GUARDS_SHARED accumulated
    /// inside `_emit_operation` / `emit_guard_operation`
    /// (optimizer.py/629/673-674).  RPython publishes these
    /// directly via `self.metainterp_sd.profiler.count(...)`; pyre's
    /// `Optimizer` carries no `metainterp_sd` reference and therefore
    /// piggybacks on the same deferred-fold pattern as the resumedata
    /// memo counters.
    pub fn update_counters(&mut self, profiler: &crate::jitprof::JitProfiler) {
        self.resumedata_memo.borrow().update_counters(profiler);
        profiler.count(crate::pyjitpl::counters::OPT_OPS, self.opt_ops_emitted);
        profiler.count(
            crate::pyjitpl::counters::OPT_GUARDS,
            self.opt_guards_emitted,
        );
        profiler.count(
            crate::pyjitpl::counters::OPT_GUARDS_SHARED,
            self.opt_guards_shared_emitted,
        );
        self.opt_ops_emitted = 0;
        self.opt_guards_emitted = 0;
        self.opt_guards_shared_emitted = 0;
        // Drain per-pass `Counters.*` accumulators (e.g.
        // VectorizingOptimizer's OPT_VECTORIZE_TRY / OPT_VECTORIZED).
        for pass in &mut self.passes {
            pass.drain_profiler_counters(profiler);
        }
    }

    /// optimizer.py: flush()
    /// Flush all passes' postponed state.
    pub fn flush(&mut self, ctx: &mut OptContext) -> Result<(), crate::optimize::InvalidLoop> {
        for pass_idx in 0..self.passes.len() {
            // heap.flush() resolves the "next_optimization" emit_extra
            // target through `ctx.current_pass_idx`. Set it to the
            // currently-flushing pass so any ops force_all_lazy_sets
            // emits land at `(pass_idx + 1, op)` and skip the flushing
            // pass on the subsequent drain — RPython parity with
            // `self.next_optimization`.
            ctx.current_pass_idx = pass_idx;
            let pass = &mut self.passes[pass_idx];
            pass.flush(ctx);
            // RPython Optimization.emit_extra() routes newly forced ops to
            // optimizer.send_extra_operation(op, self.next_optimization).
            // During flush we must preserve that contract: each pass's flush
            // output is processed only by downstream passes, never by the
            // flushing pass again. send_extra_operation (via the drain) may
            // raise InvalidLoop; propagate it like RPython's flush.
            self.drain_extra_operations_from(pass_idx + 1, ctx)?;
        }
        Ok(())
    }

    /// Build a short preamble from an optimized trace's preamble section.
    /// Convenience method that combines extract + produce.
    pub fn build_short_preamble(
        optimized_ops: &[Op],
    ) -> crate::optimizeopt::shortpreamble::ShortPreamble {
        crate::optimizeopt::shortpreamble::extract_short_preamble(optimized_ops)
    }

    /// optimizer.py: send_extra_operation(op, ctx)
    /// Send an extra operation through the pass chain as if it were
    /// a new operation from the trace. Used by passes that need to
    /// inject additional operations.
    pub fn send_extra_operation(
        &mut self,
        op_rc: &OpRc,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        // Register the producer for op_rc.pos before dispatch so a pass that
        // folds it via make_equal_to(from_bound_op(op_rc), ..) writes the
        // forwarding onto a host find_producer_op can reach (the normal trace
        // path registers via bind_input_resops; emit_extra does the same).
        ctx.register_extra_producer(op_rc);
        self.propagate_from_pass(0, op_rc, ctx)
    }

    /// RPython optimizer.py: emit_extra(op, emit=False) parity.
    /// Route an operation through passes starting AFTER `after_pass_idx`,
    /// matching RPython's `send_extra_operation(op, self.next_optimization)`.
    pub fn send_extra_operation_after(
        &mut self,
        after_pass_idx: usize,
        op: &Op,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        let op_rc = OpRc::new(op.clone());
        ctx.register_extra_producer(&op_rc);
        self.propagate_from_pass(after_pass_idx + 1, &op_rc, ctx)
    }

    /// optimizer.py: force_box — force a virtual to be materialized.
    /// Also pops from potential_extra_ops (optimizer.py).
    ///
    /// Body refs route through the preamble source directly, so the prior
    /// reverse-lookup (`imported_short_source`) 3rd key is no longer needed.
    /// Mirrors force_box_inline (mod.rs) contract.
    /// `op = get_box_replacement(op)` walks the caller-held box.
    pub fn force_box(
        &mut self,
        op: &majit_ir::operand::Operand,
        ctx: &mut OptContext,
    ) -> majit_ir::operand::Operand {
        let opref = op.to_opref();
        // optimizer.py: op = get_box_replacement(op)
        if op.is_constant() {
            return op.clone();
        }
        let resolved_box = op.get_box_replacement(false);
        let resolved_op = Some(resolved_box);
        let resolved = resolved_op.as_ref().map_or(opref, |op| op.to_opref());
        // optimizer.py:351-359: potential_extra_ops.pop(op) → sb.add_preamble_op.
        // The pool is keyed by the pure op's result Box. When that result
        // folded to an inline Const, the Const can never be a pool key (the
        // value is reproduced by inlining at use sites, not by a produced
        // short box), so `op in potential_extra_ops` is always false and no
        // short-preamble op is recorded. Skip the recording for const-resolved
        // results — otherwise the Const reaches `used_boxes` and the carried
        // label slot trips `OpRef::raw()` in unroll.rs.
        if !resolved.is_constant() {
            let tracked = resolved_op
                .as_ref()
                .and_then(|resolved_box| ctx.take_potential_extra_op(resolved_box))
                .or_else(|| {
                    if opref == resolved {
                        None
                    } else {
                        ctx.take_potential_extra_op(op)
                    }
                });
            if let Some(preamble_op) = tracked {
                // shortpreamble.py:434 `op = preamble_op.op.get_box_replacement()`
                // — the resolved Box itself is handed to the builder.
                // shortpreamble.py:434 `op = preamble_op.op.get_box_replacement()`
                // — walk the box's own `_forwarded` chain (total; identity on a
                // miss), object-native rather than positional.
                let resolved_for_pop = preamble_op.op.get_box_replacement(false);
                if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                    builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
                } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                    builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
                }
            }
        }
        let resolved_box = resolved_op.expect("get_box_replacement is total");
        // `optimizer.py Optimizer.force_box` returns a Const unchanged after
        // the potential-extra-op lookup: Const carries no PtrInfo and cannot
        // be virtual.
        if resolved_box.is_constant() {
            return resolved_box;
        }
        // optimizer.py:361-362: if op.type == 'i' and info.is_constant():
        //     return ConstInt(info.get_constant_int())
        // A forced operand whose IntBound is already constant materializes as a
        // ConstInt before the virtual-force branch. Read the bound without
        // installing one (peek), so a plain int box keeps flowing unchanged.
        if resolved_box.type_() == majit_ir::Type::Int
            && let Some(bound) = ctx.peek_intbound_box(&resolved_box)
            && bound.is_constant()
        {
            return majit_ir::operand::Operand::const_(majit_ir::Const::Int(
                bound.get_constant_int(),
            ));
        }
        if ctx.is_virtual(&resolved_box) {
            // Virtualizable represents an existing heap object with tracked
            // fields — not a deferred allocation. force_box must not take
            // its PtrInfo. RPython parity: Virtualizable is never a "true"
            // virtual (no allocation to emit); it just tracks field state
            // for the standard frame. Calling force_box on it would destroy
            // the tracked state via take_ptr_info.
            if ctx.is_virtualizable(&resolved_box) {
                return resolved_box;
            }
            // RPython: info.force_box() sets _is_virtual=False in-place.
            // Take ownership so the Virtual PtrInfo is removed. force_box_impl
            // installs a non-virtual (Instance/Struct) at the alloc_ref.
            let mut info = ctx.take_ptr_info(&resolved_box).unwrap();
            let forced = info.force_box(&resolved_box, ctx);
            return ctx.resolve_operand_operand(&forced);
        }
        resolved_box
    }

    /// `optimizer.py` `force_box_for_end_of_preamble(box)`.
    ///
    /// ```python
    /// def force_box_for_end_of_preamble(self, box):
    ///     if box.type == 'r':
    ///         info = getptrinfo(box)
    ///         if info is not None and info.is_virtual():
    ///             rec = {}
    ///             return info.force_at_the_end_of_preamble(box, self.optearlyforce, rec)
    ///         return box
    ///     if box.type == 'i':
    ///         info = getrawptrinfo(box)
    ///         if info is not None:
    ///             return info.force_at_the_end_of_preamble(box, self.optearlyforce, None)
    ///     return box
    /// ```
    ///
    /// Wraps `force_at_the_end_of_preamble` with the per-type gating
    /// RPython inlines at the call site.  Ref boxes only recurse when
    /// a virtual `PtrInfo` is present; Int boxes only recurse when a
    /// raw-ptr info is present (pyre stores that under
    /// `PtrInfo::VirtualRawBuffer` in the same info table); Float
    /// boxes are returned unchanged.  Callers that previously invoked
    /// `force_at_the_end_of_preamble` directly should route through
    /// this wrapper for RPython structural parity (unroll.py:126-127).
    pub fn force_box_for_end_of_preamble(
        &mut self,
        box_: &majit_ir::operand::Operand,
        ctx: &mut OptContext,
    ) -> majit_ir::operand::Operand {
        let resolved = ctx.resolve_operand_operand(box_);
        match resolved.type_() {
            // optimizer.py:307-313 — `box.type == 'r'` path.
            majit_ir::Type::Ref => {
                if ctx.is_virtual(&resolved) {
                    return self.force_at_the_end_of_preamble(&resolved, ctx);
                }
                box_.clone()
            }
            // optimizer.py:314-318 — `box.type == 'i'` path.
            //
            // pyre tracks raw-ptr info on Int boxes via
            // `PtrInfo::VirtualRawBuffer` in the same `ptr_info` table
            // that Ref boxes use; there is no separate `getrawptrinfo`
            // registry, so the presence check fires on any PtrInfo
            // attached to an Int-typed box.
            majit_ir::Type::Int => {
                if ctx.has_ptr_info(&resolved) {
                    return self.force_at_the_end_of_preamble(&resolved, ctx);
                }
                box_.clone()
            }
            // optimizer.py:319 — fall-through `return box`.
            _ => box_.clone(),
        }
    }

    /// optimizer.py: force_at_the_end_of_preamble(box)
    ///
    /// The exported loop state should record the boxes that survive the end of
    /// the preamble after virtuals have been forced into a loop-carried shape.
    pub fn force_at_the_end_of_preamble(
        &mut self,
        box_: &majit_ir::operand::Operand,
        ctx: &mut OptContext,
    ) -> majit_ir::operand::Operand {
        let mut rec: crate::FxIndexSet<majit_ir::operand::Operand> = crate::FxIndexSet::default();
        self.force_at_the_end_of_preamble_rec(box_, ctx, &mut rec)
    }

    fn force_at_the_end_of_preamble_rec(
        &mut self,
        box_: &majit_ir::operand::Operand,
        ctx: &mut OptContext,
        rec: &mut crate::FxIndexSet<majit_ir::operand::Operand>,
    ) -> majit_ir::operand::Operand {
        let resolved = ctx.resolve_operand_operand(box_);
        let Some(mut info) = ctx.peek_ptr_info(&resolved) else {
            return resolved;
        };

        // `info.py` `PtrInfo.force_at_the_end_of_preamble` base:
        //
        // ```python
        // def force_at_the_end_of_preamble(self, op, optforce, rec):
        //     if not self.is_virtual():
        //         return get_box_replacement(op)
        //     return self._force_at_the_end_of_preamble(op, optforce, rec)
        // ```
        //
        // RPython gates every dispatch on `self.is_virtual()`.  pyre's
        // variant-tag match below would otherwise re-force a non-virtual
        // `VirtualRawSlice` (post-`_force_elements`, `parent = OpRef::NONE`)
        // every time this method is re-entered — matching info.py
        // `return self.parent is not None` requires the same is_virtual
        // gate here as in the RPython base.
        if !info.is_virtual() {
            return resolved;
        }

        // RPython info.py: InstancePtrInfo, StructPtrInfo, ArrayPtrInfo all
        // override _force_at_the_end_of_preamble to keep the virtual alive
        // and recurse into fields. AbstractRawPtrInfo uses the base
        // _force_at_the_end_of_preamble → force_box() (materialization).
        if matches!(
            info,
            crate::optimizeopt::info::PtrInfo::Virtual(_)
                | crate::optimizeopt::info::PtrInfo::VirtualStruct(_)
                | crate::optimizeopt::info::PtrInfo::VirtualArray(_)
                | crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(_)
        ) {
            // info.py:231 `rec[self] = None` keys the recursion guard by the
            // virtual's PtrInfo object identity; in pyre one virtual head box
            // <-> one PtrInfo, so key by the canonical resolved box.
            if rec.contains(&resolved) {
                return resolved;
            }
            rec.insert(resolved.clone());
            info.force_at_the_end_of_preamble(|child| {
                self.force_at_the_end_of_preamble_rec(&child, ctx, rec)
            });
            ctx.set_ptr_info(&resolved, info);
            return resolved;
        }

        // `AbstractVirtualPtrInfo._force_at_the_end_of_preamble`
        // without override is `force_box()`.
        // `StrPtrInfo` (`vstring.py`) and `AbstractRawPtrInfo`
        // (`RawBufferPtrInfo` / `RawSlicePtrInfo`) all inherit that
        // base.  `VirtualStateConstructor` has no `visit_vstr*`
        // (`walkvirtual.py` raises `NotImplementedError`), so a
        // virtual string that reaches export is a missed force.
        // optimizer.py routes through `optforce = self.optearlyforce`.
        if matches!(
            info,
            crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_)
                | crate::optimizeopt::info::PtrInfo::Str(_)
        ) {
            let saved = ctx.current_pass_idx;
            ctx.current_pass_idx = ctx.optearlyforce_idx;
            let result = self.force_box(&resolved, ctx);
            ctx.current_pass_idx = saved;
            return result;
        }

        resolved
    }

    /// optimizer.py: protect_speculative_operation(op, ctx)
    /// When constant-folding a pure operation, verify that the folded
    /// constant doesn't cause a memory safety issue (e.g., null deref in
    /// getfield). If the result would be invalid, don't fold.
    pub fn protect_speculative_operation(&self, op: &Op, ctx: &OptContext) -> bool {
        // For now, conservative: only allow folding on arithmetic ops.
        // getfield/getarrayitem on constant null pointer would crash.
        match op.opcode {
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF => {
                // Check arg(0) is not null constant.
                if let Some(0) = ctx.get_constant_int_box(&op.arg(0).get_box_replacement(false)) {
                    return false; // would deref null
                }
                true
            }
            _ => true,
        }
    }

    /// optimizer.py: getlastop() — return the last emitted non-guard operation.
    pub fn getlastop<'a>(&self, ctx: &'a OptContext) -> Option<&'a Op> {
        ctx.new_operations.last().map(|rc| rc.as_ref())
    }

    /// optimizer.py: get_count_of_ops()
    /// Count operations emitted so far.
    pub fn get_count_of_ops(ctx: &OptContext) -> usize {
        ctx.new_operations.len()
    }

    /// optimizer.py: get_count_of_guards()
    /// Count guards emitted so far.
    pub fn get_count_of_guards(ctx: &OptContext) -> usize {
        ctx.new_operations
            .iter()
            .filter(|op| op.opcode.is_guard())
            .count()
    }

    /// logger.py:33 `Logger.log_loop(inputargs, operations, ...)`, called once
    /// a trace's operation list is final — compile.py:474, :557, :562.
    ///
    /// Dumps the operation list, not just a count. A count separates a compiled
    /// body from a compiled nothing; it cannot separate a working body from a
    /// dead one, because deadness is a property of shape. A body that opens by
    /// failing a guard has no back edge and runs zero iterations while
    /// reporting a healthy-looking op total.
    ///
    /// `label` names which compile path produced the trace, since they differ
    /// in whether the body was unrolled and whether it is a bridge. Bridges
    /// must be logged too: they take trace ids from the same counter the loops
    /// draw from, so omitting them leaves gaps that make a dumped trace
    /// impossible to line up with the `trace=` field of a guard-failure log.
    pub fn log_optimized_trace<V, T, C>(label: &str, ops: &[T], constants: &C)
    where
        V: std::fmt::Debug,
        T: AsRef<Op>,
        C: majit_ir::resoperation::ConstLookup<V>,
    {
        if !crate::log_opt_enabled() {
            return;
        }
        eprintln!(
            "[MAJIT] optimized trace ({}): {} ops, {} guards",
            label,
            ops.len(),
            ops.iter()
                .filter(|op| op.as_ref().opcode.is_guard())
                .count()
        );
        eprint!("{}", majit_ir::format_trace(ops, constants));
    }

    /// optimizer.py `getnullness(op)` parity (line-by-line port).
    ///
    /// Delegates to `OptContext::getnullness`, which implements the
    /// upstream `op.type == 'r' or is_raw_ptr(op)` dispatch and the
    /// `getintbound(op).getnullness()` fallback for `'i'`-typed Boxes.
    /// Returns one of `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py).
    ///
    /// Takes `&mut OptContext` to mirror the upstream `getintbound`
    /// lazy-install side effect (optimizer.py).
    pub fn getnullness(ctx: &mut OptContext, opref: OpRef) -> i8 {
        // optimizer.py `getnullness` reads `getptrinfo` /
        // `getintbound` of an existing box; the `'r'` arm uses
        // `getptrinfo(create=False)`, so an absent info yields `INFO_UNKNOWN`
        // without minting. Resolve the producer box and delegate — a
        // producer-less position has no info and returns `INFO_UNKNOWN`, never
        // a fresh stand-in (a nullness query must not pollute the producer
        // registry). `ctx.getnullness` performs the per-type `getintbound`
        // lazy-install on the resolved box.
        let Some(b) = ctx.get_box_replacement_operand_opt(opref) else {
            return crate::optimizeopt::INFO_UNKNOWN;
        };
        ctx.getnullness(&b)
    }

    /// optimizer.py: make_constant_class(op, class_const, update_last_guard)
    ///
    /// Sets known class on PtrInfo, preserving last_guard_pos from any
    /// existing info. When `update_last_guard` is false (RECORD_EXACT_CLASS),
    /// the guard position is NOT updated — preserving guard strengthening
    /// opportunities for subsequent GUARD_CLASS ops.
    ///
    /// PyPy `InstancePtrInfo(descr, known_class, is_virtual)` is one class
    /// (info.py) that covers both virtual and non-virtual instances via
    /// the `is_virtual` flag. The Rust port splits that into two PtrInfo
    /// enum variants — `Instance` (is_virtual=False) and `Virtual`
    /// (is_virtual=True) — for ergonomic dispatch. `isinstance(opinfo,
    /// InstancePtrInfo)` at optimizer.py is therefore true for *either*
    /// Rust variant, so both must update `_known_class` in place rather
    /// than being overwritten by a fresh known-class info (which would
    /// drop the `Virtual` fields / descr / cached_vinfo state).
    pub fn make_constant_class(
        ctx: &mut OptContext,
        op: &Operand,
        class_value: i64,
        update_last_guard: bool,
    ) {
        // optimizer.py: op = op.get_box_replacement()
        let resolved = op.get_box_replacement(false);
        // optimizer.py:139: opinfo = op.get_forwarded()
        // RPython's InstancePtrInfo covers both virtual and non-virtual
        // instances (`is_virtual` is a field). Rust splits those states into
        // PtrInfo::Instance and PtrInfo::Virtual, so both arms must preserve
        // the existing object info and only update `_known_class`.
        let updated_existing = ctx
            .with_ptr_info_mut(&resolved, |info| match info {
                PtrInfo::Instance(iinfo) => {
                    iinfo.known_class = Some(class_value);
                    true
                }
                PtrInfo::Virtual(vinfo) => {
                    vinfo.known_class = Some(class_value);
                    true
                }
                _ => false,
            })
            .unwrap_or(false);
        if !updated_existing {
            // optimizer.py:142-148: preserve last_guard_pos from old info.
            // operand-direct read mirrors `info.py get_last_guard_pos`
            // — drops the `last_guard_pos(opref)` bridge.
            let old_guard_pos = resolved
                .ptr_info()
                .and_then(|p| p.get_last_guard_pos())
                .map(|p| p as i32)
                .unwrap_or(-1);
            let mut new_info = PtrInfo::known_class(class_value, true);
            new_info.set_last_guard_pos(old_guard_pos);
            ctx.set_ptr_info(&resolved, new_info);
        }
        // optimizer.py:150-151: if update_last_guard: mark_last_guard
        if update_last_guard {
            ctx.mark_last_guard(&resolved);
        }
    }

    /// optimizer.py `is_call_pure_pure_canraise(op)`.
    ///
    /// ```text
    /// if not rop.is_call_pure(op.getopnum()):
    ///     return False
    /// effectinfo = op.getdescr().get_extra_info()
    /// return effectinfo.check_can_raise(ignore_memoryerror=True)
    /// ```
    ///
    /// The `is_call_pure` gate is what makes the answer `False` for every
    /// ordinary operation: only a CALL_PURE is formally side-effect free yet
    /// still able to raise, so only a CALL_PURE can break guard resume-data
    /// sharing while passing `has_no_side_effect`.  `MemoryError`-only
    /// effects are ignored.
    pub fn is_call_pure_pure_canraise(op: &Op) -> bool {
        if !op.opcode.is_call_pure() {
            return false;
        }
        op.with_call_descr(|cd| cd.get_extra_info().check_can_raise(true))
            .unwrap_or(false)
    }

    /// Add an optimization pass to the chain.
    pub fn add_pass(&mut self, mut pass: Box<dyn Optimization>) {
        pass.set_pureop_historylength(self.pureop_historylength);
        self.passes.push(pass);
    }

    pub fn set_pureop_historylength(&mut self, limit: usize) {
        self.pureop_historylength = limit;
        for pass in &mut self.passes {
            pass.set_pureop_historylength(limit);
        }
    }

    /// `virtualize.py:140` fan-out: publish the live `VirtualRefInfo`
    /// (`MetaInterp.virtualref_info` / RPython
    /// `metainterp_sd.virtualref_info`) to every pass that consumes it
    /// (`OptVirtualize` and `OptHeap`). Other passes' `set_vrefinfo` is a no-op.
    pub fn set_vrefinfo(&mut self, vrefinfo: crate::virtualref::VirtualRefInfo) {
        for pass in &mut self.passes {
            pass.set_vrefinfo(vrefinfo.clone());
        }
    }

    /// Mark all passes as Phase 2 (loop body).
    pub fn set_phase2(&mut self, phase2: bool) {
        for pass in &mut self.passes {
            pass.set_phase2(phase2);
        }
    }

    /// Run all optimization passes over a list of operations.
    ///
    /// Returns the optimized operation list.
    /// optimizer.py: propagate_all_forward(trace, call_pure_results, flush)
    pub fn propagate_all_forward(&mut self, ops: &[Op]) -> Vec<Op> {
        self.optimize_with_constants(ops, &mut majit_ir::ConstMap::default())
    }

    /// Run all optimization passes, with known constants pre-populated.
    ///
    /// `constants` maps OpRef indices to typed `Value` payloads. RPython
    /// keeps typed `ConstInt/ConstFloat/ConstPtr` boxes in the optimized
    /// trace (history.py/261/307 — every Const pins `box.type`);
    /// majit mirrors that by carrying `Value` directly here, so the
    /// type is intrinsic to each pool entry.
    ///
    /// After optimization, newly-discovered constants (from constant folding)
    /// are written back into the map so the backend can resolve them.
    pub fn optimize_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
    ) -> Vec<Op> {
        self.optimize_with_constants_and_inputs(ops, constants, 0)
    }

    /// Like `optimize_with_constants`, but also takes `num_inputs` so that
    /// ops emitted by the optimizer (e.g. from force_virtual) get pos values
    /// that don't collide with input argument variable indices.
    /// Phase 1 / standalone entry point. Computes `start_next_pos` from
    /// the input ops' max OpRef so newly emitted ops never collide with
    /// the trace's existing positional layout. Phase 2 and bridge callers
    /// should use [`Self::optimize_with_constants_and_inputs_at`] with an
    /// explicit `inputarg_base` to allocate disjoint OpRefs.
    pub fn optimize_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
    ) -> Vec<Op> {
        // `_at` traffics in `OpRc`; this `&[Op]` overload wraps each op in a
        // fresh `Rc` (the #62 boundary-conversion pattern). The fresh wraps
        // are not the canonical producers, so `input_ops` is seeded as empty
        // here (`input_ops_from_ops = false`); the canonical stores
        // (`bind_input_resops` / emit) carry identity instead.
        //
        // `TraceIterator::next` caches the minted object and later args
        // resolve through that cache (`_untag`). Fixtures assert the
        // recorded positions, so this entry keeps each op's `pos` and only
        // rebinds a later arg / fail_arg onto the fresh `Rc` of the op that
        // owns that position. Void results stay uncached.
        let ops_rc = Self::rebind_wrapped_trace_ops(ops);
        // This `&[Op]` convenience overload backs unit-test fixtures and the
        // legacy `propagate_all_forward` / `optimize_trace_with_constants`
        // helpers, not the production JIT path (which uses the `_oprc` /
        // `_vable_out` / `optimize_bridge` `Result` entries).  An `InvalidLoop`
        // here means a fixture built a contradictory trace, which should fail
        // loudly rather than be silently swallowed.
        self.run_optimize_from_inputs(&ops_rc, constants, num_inputs, false)
            .expect("optimize_with_constants_and_inputs: unexpected InvalidLoop")
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect()
    }

    /// Wrap each recorded `Op` in a fresh `OpRc` and rebind later args the
    /// way `TraceIterator::_cache` does: a non-void result is the object
    /// later positions name. Positions stay as the fixture recorded them.
    fn rebind_wrapped_trace_ops(ops: &[Op]) -> Vec<majit_ir::OpRc> {
        let mut wrapped: Vec<majit_ir::OpRc> = Vec::with_capacity(ops.len());
        // First InputArg at each index. Later args at that index reuse it,
        // the same way `TraceIterator` seeds `_cache` once per input.
        let mut input_cache: Vec<(u32, Operand)> = Vec::new();
        for op in ops {
            let rc = OpRc::new(op.clone());
            for i in 0..rc.num_args() {
                let arg = rc.arg(i);
                if let Some(rebound) = Self::rebound_arg_to_wrapped(&arg, &wrapped)
                    .or_else(|| Self::rebound_inputarg(&arg, &input_cache))
                {
                    rc.setarg(i, rebound);
                }
                Self::remember_inputarg(&rc.arg(i), &mut input_cache);
            }
            rc.map_failargs_in_place(|arg| {
                if let Some(rebound) = Self::rebound_arg_to_wrapped(arg, &wrapped)
                    .or_else(|| Self::rebound_inputarg(arg, &input_cache))
                {
                    *arg = rebound;
                }
                Self::remember_inputarg(arg, &mut input_cache);
            });
            wrapped.push(rc);
        }
        wrapped
    }

    fn rebound_inputarg(arg: &Operand, cache: &[(u32, Operand)]) -> Option<Operand> {
        let ia = arg.bound_inputarg()?;
        cache
            .iter()
            .rev()
            .find(|(idx, _)| *idx == ia.index)
            .map(|(_, cached)| cached.clone())
    }

    fn remember_inputarg(arg: &Operand, cache: &mut Vec<(u32, Operand)>) {
        let Some(ia) = arg.bound_inputarg() else {
            return;
        };
        if cache.iter().any(|(idx, _)| *idx == ia.index) {
            return;
        }
        cache.push((ia.index, arg.clone()));
    }

    /// Last non-void wrapped op whose recorded position is `arg`'s.
    /// Constants and the absent slot pass through (`TraceIterator::_untag`).
    fn rebound_arg_to_wrapped(arg: &Operand, prior: &[majit_ir::OpRc]) -> Option<Operand> {
        if arg.is_constant() || arg.is_none() {
            return None;
        }
        let opref = arg.to_opref();
        let prod = prior.iter().rev().find(|op| {
            let pos = op.pos().get();
            pos == opref && !pos.is_none() && op.opcode.result_type() != majit_ir::Type::Void
        })?;
        Some(Operand::from_bound_op(prod))
    }

    /// `OpRc`-threading entry for callers that hold the canonical
    /// `Rc<Op>` slice the recorder produced (e.g. `TreeLoop.ops` at the
    /// loop-finish / simple-loop sites). Passing those canonical ops lets
    /// `input_ops` be seeded from them directly (`input_ops_from_ops =
    /// true`), so `find_producer_op`'s lowest-priority store is populated
    /// without any snapshot read.
    ///
    /// The caller transfers its argument slots: emission resolves each
    /// argument in place through its forwarding chain
    /// (`optimizer.py _emit_operation`), so on return an op's
    /// `getarglist` reads the resolved operands, not the recorded ones. A
    /// caller that still needs the recorded arguments must read them before
    /// the call.
    pub fn optimize_with_constants_and_inputs_oprc(
        &mut self,
        ops: &[majit_ir::OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
    ) -> Result<Vec<majit_ir::OpRc>, crate::optimize::InvalidLoop> {
        self.run_optimize_from_inputs(ops, constants, num_inputs, true)
    }

    pub(crate) fn run_optimize_from_inputs(
        &mut self,
        ops: &[majit_ir::OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        input_ops_from_ops: bool,
    ) -> Result<Vec<majit_ir::OpRc>, crate::optimize::InvalidLoop> {
        // Ensure new ops get positions beyond all original trace positions.
        // Original ops keep their tracer-assigned positions; new ops (constants,
        // force materializations) must not collide with them.
        let max_pos = ops
            .iter()
            .map(|op| op.pos().get())
            .filter(|op| !op.is_none() && !op.is_constant())
            .map(|op| op.raw())
            .max()
            .unwrap_or(0);
        let start_next_pos = (max_pos + 1).max(num_inputs as u32);
        self.optimize_with_constants_and_inputs_at(
            ops,
            constants,
            num_inputs,
            0,
            start_next_pos,
            input_ops_from_ops,
        )
    }

    /// opencoder.py:271 _index parity entry point.
    ///
    /// Optimizes a slice of ops whose `pos`/`args` reference OpRefs in a
    /// shifted namespace `[inputarg_base, …)`. The first fresh OpRef the
    /// optimizer assigns to a non-void result is `OpRef::int_op(start_next_pos)`.
    /// Phase 2 / bridges pass `inputarg_base = parent_high_water` and
    /// `start_next_pos = parent_high_water + num_inputs` so the iteration's
    /// OpRefs are disjoint from any parent trace's emitted ops.
    pub fn optimize_with_constants_and_inputs_at(
        &mut self,
        ops: &[majit_ir::OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        inputarg_base: u32,
        start_next_pos: u32,
        input_ops_from_ops: bool,
    ) -> Result<Vec<majit_ir::OpRc>, crate::optimize::InvalidLoop> {
        use majit_ir::OpRef;
        // Test-only auto-seed of `trace_inputargs` from the variant
        // tags of any InputArg*/IntOp/FloatOp/RefOp OpRef that references
        // a slot index in `[0, num_inputs)`. Production callers populate
        // the list from the recorder's InputArg{Int,Ref,Float} objects
        // via `setup_optimizations` (optimizer.py:34 self.inputargs).
        //
        // Falls back to `Type::Ref` for slots that no arg references —
        // RPython never sees that case (every InputArg flows through some
        // op), but unit fixtures can omit references to unused slots.
        #[cfg(test)]
        if self.trace_inputargs.is_empty() && num_inputs > 0 {
            // RPython InputArg variants are Int/Ref/Float only
            // (resoperation.py InputArgInt/727/739) — Void inputargs do not exist.
            // VoidOp args at inputarg slots leave the slot at the Ref
            // fallback because the actual inputarg-side Box would have
            // been Int/Ref/Float per producer-side typing.
            let mut types = vec![majit_ir::Type::Ref; num_inputs];
            for op in ops.iter() {
                for arg in op.args_slice().iter() {
                    // Const args (inline-value or legacy idx) never reference an
                    // inputarg slot; only InputArg*/IntOp/FloatOp/RefOp do.
                    if arg.is_constant() {
                        continue;
                    }
                    let idx = arg.to_opref().raw() as usize;
                    if idx < num_inputs {
                        match arg.to_opref().ty() {
                            Some(majit_ir::Type::Void) | None => {}
                            Some(tp) => types[idx] = tp,
                        }
                    }
                }
            }
            self.trace_inputargs = OpRef::inputarg_refs(&types);
        }
        self.imported_label_args = None;
        self.terminal_op = None;
        self.runtime_boxes.clear();
        // RPython parity: each optimizer run is a fresh Optimizer instance.
        // In pyre we reuse the same Optimizer, so clear per-run state.
        self.last_guard_op_idx = None;
        let mut ctx = self.take_opt_context(ops.len(), num_inputs, inputarg_base, start_next_pos);
        ctx.skip_flush_mode = self.skip_flush;
        ctx.building_bridge = self.building_bridge;
        ctx.constant_fold_alloc = self.constant_fold_alloc.take();
        // Seed `input_ops` with the input ops' producers, whose operands
        // carry the InputArg objects `ensure_inputarg_bindings` reuses.
        // When the caller threads the canonical `Rc<Op>` slice
        // (`input_ops_from_ops`, e.g. `TreeLoop.ops` at the loop-finish /
        // simple-loop sites), take them directly. Otherwise (fresh-Rc `&[Op]`
        // boundary wraps, bridges, Phase 2, fixtures) the threaded ops are not
        // the producers, so the store stays empty or uses the explicit seed.
        ctx.input_ops = if let Some(seed) = self.explicit_input_ops_seed.take() {
            // Phase 2: the caller threaded the recorder `Rc<Op>` slice
            // (`preamble_data.base.operations()`) — the same `Rc` whose
            // Phase-1 `_forwarded` the canonical resolvers observe.
            seed.into_iter()
                .filter(|op| {
                    let p = op.pos().get();
                    !p.is_none() && !p.is_constant()
                })
                .collect()
        } else if input_ops_from_ops {
            ops.iter()
                .filter(|op| {
                    let p = op.pos().get();
                    !p.is_none() && !p.is_constant()
                })
                .cloned()
                .collect()
        } else {
            // No explicit seed and no threaded canonical ops. Every entry now
            // seeds `input_ops` explicitly: the loop-finish, retrace, and retry
            // paths thread their producers (or an empty seed on the cut path),
            // and the bridge seeds empty. The remaining callers that reach here
            // (`propagate_all_forward`, `optimize_with_constants`, unit
            // fixtures) rely on the canonical stores (`seed_boxes_canonical` /
            // `bind_input_resops` / emit), so the store is empty here.
            Vec::new()
        };
        ctx.string_length_resolver = self.string_length_resolver.clone();
        ctx.string_content_resolver = self.string_content_resolver.clone();
        ctx.string_constant_alloc = self.string_constant_alloc.clone();
        ctx.callinfocollection = self.callinfocollection.clone();
        // optimizer.py `Optimizer.__init__` owns one ResumeDataLoopMemo and
        // `store_final_boxes_in_guard` reuses it for every emitted guard.
        // OptContext is only a Rust borrow-splitting adaptation, so share the
        // same object rather than constructing a memo per guard.
        ctx.resumedata_memo = std::rc::Rc::clone(&self.resumedata_memo);
        // virtualstate.py `GenerateGuardState.__init__: self.cpu =
        // optimizer.cpu`. Propagate the runtime typeptr-read hook so
        // virtualstate match (KnownClass arms) can fall back to
        // `cpu.cls_of_box(runtime_box)` when the optimizer-tracked
        // PtrInfo has no `known_class` recorded.
        ctx.cpu = self.cpu.clone();
        ctx.supports_efficient_uint_mul_high = self.supports_efficient_uint_mul_high;
        // RPython resume.py parity: Phase 2 optimizer needs imported_label_args
        // to resolve NONE positions in fail_args inherited from Phase 1.
        ctx.imported_virtuals = self.imported_virtuals.clone();
        ctx.imported_label_args = self.imported_label_args.clone();

        // RPython Box type parity: in RPython each Box carries its type
        // intrinsically (InputArgInt, InputArgRef, etc.). In majit, OpRef
        // is an untyped u32, so the cross-phase carries below thread the
        // Phase 1-side type information into Phase 2's `opref_type` chain.
        // 1. Phase 1 → Phase 2 carry: `phase1_emit_ops` below; `op_at`
        //    resolves cross-phase OpRefs through `op.type_` directly
        //    (history.py:220 box.type parity).
        // optimizer.py `self.inputargs = inputargs` parity.
        ctx.inputargs = self.trace_inputargs.clone();
        for ia in std::mem::take(&mut self.trace_inputarg_boxes) {
            ctx.inputarg_refs.entry(ia.index).or_insert(ia);
        }
        // The `&[Op]` wrap carries the fixture's InputArg objects. Install
        // those before `ensure_inputarg_bindings` mints a second host per
        // slot (`or_insert` / same-type keep).
        for op in ops {
            for arg in op.args_slice() {
                if let Some(ia) = arg.bound_inputarg() {
                    ctx.inputarg_refs.entry(ia.index).or_insert(ia);
                }
            }
            if let Some(failargs) = op.guard_fail_args() {
                for arg in failargs {
                    if let Some(ia) = arg.bound_inputarg() {
                        ctx.inputarg_refs.entry(ia.index).or_insert(ia);
                    }
                }
            }
        }
        // Bind inputarg hosts so `make_equal_to` routes InputArg-targeted
        // chain steps through `Forwarded::InputArg(_)` (the orphan-box
        // forwarding fallback has been retired). Phase 2 enters with a
        // fresh per-iteration inputarg set whose TreeLoop-owned strong
        // `InputArgRc`s were dropped, so re-bind them here.
        ctx.ensure_inputarg_bindings();
        // Bind every input op's resop operand so chain-walker
        // terminals are guaranteed bound before any `&self` reader
        // (e.g. `OptIntBounds::getintbound_box`) reaches a
        // `set_forwarded_*` write. `TraceIterator::next()`
        // (`opencoder.rs`) plants unbound resop slots; without this
        // pre-pass, `get_box_replacement_operand_opt(&self)` could return an
        // unbound terminal that fails `write_forwarded`'s bound-
        // precondition assert.
        ctx.bind_input_resops(ops);
        // unroll.py:188 — the bridge iterator's fresh inputargs must enter
        // optimization without forwarding.  This is deliberately a
        // debug-only invariant, matching the upstream assertion without
        // adding release-build work.
        debug_assert!(
            !self.building_bridge
                || crate::optimizeopt::unroll::UnrollOptimizer::check_no_forwarding(
                    &ctx,
                    &ctx.inputargs,
                )
        );
        // Phase 1 emit ops: single source of truth for cross-phase OpRef →
        // `op.type_` lookup (history.py:220 parity).
        ctx.phase1_emit_ops = std::mem::take(&mut self.phase1_emit_ops);
        // Single write site for the OptContext field: rebuild its O(1)
        // producer index in lockstep (never mutated afterwards).
        ctx.rebuild_phase1_emit_ops_index();
        // 3. (removed) 5: transformed trace ops carry `op.type_`
        //    intrinsically (resoperation.py:1693 parity); the pipeline
        //    emits each op into `new_operations` before moving to the
        //    next, so `OptContext::op_at` resolves the type of any
        //    earlier-emitted op without going through `value_types`.
        //    The fan-in into `value_types` here was redundant book-
        //    keeping that only fed the `next_pos` bump below — replaced
        //    with a direct max over the input slice.

        // Bump reserve_pos cursor past every seeded position.
        //
        // `reserve_pos` only skips over `constants`. Phase 2 / retrace
        // contexts inherit Phase 1's positions through
        // input `ops` slice (post-fold transformed positions); the
        // emit-side high water can sit above the caller-provided
        // `start_next_pos` whenever the transformed `ops` slice ends
        // below that mark. Without this bump, a later `get_or_make_const`
        // / `emit` would hand out an already-seeded position and fire
        // the Box.type retype invariant.
        let max_input = ops
            .iter()
            .filter(|op| !op.pos().get().is_none() && op.result_type() != majit_ir::Type::Void)
            .map(|op| op.pos().get().raw())
            .max()
            .unwrap_or(0);
        let max_snapshot = self
            .snapshot_boxes
            .iter()
            .chain(self.snapshot_vable_boxes.iter())
            .chain(self.snapshot_vref_boxes.iter())
            .flatten()
            .flat_map(|boxes| boxes.iter().map(|boxref| boxref.opref()))
            .filter(|opref| !opref.is_none() && !opref.is_constant())
            .map(|opref| opref.raw())
            .max()
            .into_iter()
            .chain(
                self.byte_bridge_resume
                    .as_ref()
                    .and_then(|feed| feed.max_box_position()),
            )
            .max()
            .unwrap_or(0);
        let next_after = max_input.max(max_snapshot).saturating_add(1);
        ctx.next_pos = ctx.next_pos.max(next_after);

        // optimizer.py force_op_from_preamble patchguardop parity: propagate to Phase 2
        // OptContext so copy_and_change guards (unroll.py:409) can get
        // rd_resume_position before GUARD_FUTURE_CONDITION is re-encountered.
        if ctx.patchguardop.is_none() {
            ctx.patchguardop = self.patchguardop.clone();
        }
        // optimizer.py:294: patchguardop is set by GUARD_FUTURE_CONDITION
        // during optimization (rewrite.rs / simplify.rs). No synthetic
        // fallback — RPython relies solely on the actual GFC from tracing.

        // RPython resume.py parity: pass snapshot_boxes to OptContext so
        // emit() can call store_final_boxes_in_guard inline at each guard
        // emission (not post-assembly).
        ctx.snapshot_boxes = std::mem::take(&mut self.snapshot_boxes);
        ctx.snapshot_frame_sizes = std::mem::take(&mut self.snapshot_frame_sizes);
        ctx.snapshot_vable_boxes = std::mem::take(&mut self.snapshot_vable_boxes);
        ctx.minimum_virtualizable_size = self.minimum_virtualizable_size;
        ctx.snapshot_vref_boxes = std::mem::take(&mut self.snapshot_vref_boxes);
        ctx.snapshot_frame_pcs = std::mem::take(&mut self.snapshot_frame_pcs);
        ctx.byte_bridge_resume = self.byte_bridge_resume.take();

        // Pre-populate known constants so passes can see them.
        //
        // history.py/261/307: `ConstInt/ConstFloat/ConstPtr` pin
        // `Box.type` at construction. The `Value` payload carries the
        // box class intrinsically, so the OpRef variant tag is recovered
        // directly from the `Value`'s type tag without any external
        // `constant_types` side table.
        //
        // The `constants` map keys inline-value-at-op-position slots: each
        // `idx` mints `OpRef::*_op(idx)` matching the `Value` variant.
        for (&idx, value) in constants.iter() {
            let opref = OptContext::op_ref_for_value(idx, value);
            // seed_constant takes the canonical `_forwarded` host; resolve
            // the body OpRef to its producing `Op` / `InputArg` box first.
            let op_ = ctx.materialize_operand_at(opref);
            ctx.seed_constant(&op_, *value);
        }

        // Setup all passes
        for pass in &mut self.passes {
            pass.setup();
        }

        // earlyforce.py:32: self.optimizer.optearlyforce = self
        // Find the EarlyForce pass index so force paths can route
        // emitted operations starting from earlyforce.next (= heap).
        for (idx, pass) in self.passes.iter().enumerate() {
            if pass.name() == "earlyforce" {
                ctx.optearlyforce_idx = idx;
                break;
            }
        }

        // bridgeopt.py deserialize_optimizer_knowledge: apply pending bridge knowledge AFTER setup.
        // RPython calls deserialize_optimizer_knowledge before propagate_all_forward
        // but after the optimizer is constructed (setup already done at __init__).
        if let Some(prd) = self.pending_bridge_rd.take() {
            crate::optimizeopt::bridgeopt::deserialize_optimizer_knowledge(
                prd.storage.rd_numb().expect("rd_numb"),
                prd.storage.rd_consts().unwrap_or(&[]),
                &prd.frontend_boxes,
                &prd.liveboxes,
                &prd.livebox_types,
                &prd.all_descrs,
                prd.cpu.clone(),
                self,
                &mut ctx,
            );
        }

        // optimizer.py: pass call_pure_results to OptPure so it can
        // constant-fold repeated pure calls across loop iterations.
        for pass in &mut self.passes {
            pass.set_call_pure_results(&self.call_pure_results);
        }

        // Take ownership of imported_loop_state for the duration of
        // this block: import_state needs `&mut self` (force_box dispatch
        // through the optimizer pass), so we cannot keep an immutable
        // borrow of `self.imported_loop_state` simultaneously. The state
        // is restored at the end of the block.
        let imported_loop_state_taken = self.imported_loop_state.take();
        if let Some(ref exported_state) = imported_loop_state_taken {
            // Every `end_arg` OpRef from the exported state is typed via
            // `OpRef::input_arg_typed` /
            // `op_typed`, so its variant tag carries Box.type and the
            // side-table refresh is dead.
            // opencoder.py + unroll.py:479-504 parity: RPython's  allow-line-citation
            // TraceIterator allocates fresh InputArg Box objects for
            // each iteration, and `import_state` asserts
            // `source is not target` (unroll.py) because the fresh
            // Phase 2 inputargs never coincide with any Phase 1 emitted
            // Box. majit encodes "fresh per iteration" via the
            // `ctx.inputarg_base` offset — Phase 2's source slots live
            // at `[inputarg_base..inputarg_base+n)`, disjoint from
            // Phase 1 emitted OpRefs `[num_inputs..next_global_opref)`.
            //
            // With disjoint namespaces, the `source == target` edge
            // case RPython avoids by construction cannot occur here
            // either: any `target` coming from `next_iteration_args`
            // is either a constant, a Phase 1 emitted OpRef, or a
            // Phase 1 inputarg OpRef (in `[0..num_inputs)`) — all
            // outside the Phase 2 `source` range. The cross-slot
            // collision guard below is preserved for the Phase 1 /
            // standalone path where `inputarg_base == 0` and `source`
            // can coincide with another slot's `target`.
            let inputarg_base = ctx.inputarg_base;
            let nia = &exported_state.next_iteration_args;
            let n = nia.len();
            // `next_iteration_args` must not be longer than the trace's
            // inputargs: every loop-back value needs a backing inputarg
            // slot to forward to. RPython guarantees this by construction:
            // `reached_loop_header` (pyjitpl.py) builds
            // `live_arg_boxes = reds + virtualizable_boxes[:-1]` for BOTH
            // the merge-point registration and the closing JUMP, so the
            // two shapes always match. Pyre's full-body-walk
            // `jit_merge_point` handler now mirrors that exactly
            // (jitcode_dispatch.rs: reds rebound to `sym.frame` /
            // `sym.execution_context`, then `append_virtualizable_boxes`
            // + `remove_consts_and_duplicates`), and the historical
            // reds-only `[frame, ec]` seeding that made a cross-loop cut
            // declare 2 inputargs against a full-shape JUMP is gone —
            // verified unreachable across the bench suite (fannkuch /
            // nbody, which used to trip it, at default and raised
            // trace_eagerness).
            //
            // The guard stays as a tripwire: a future shape regression
            // must decline through `InvalidLoop` — the designed
            // compile-failure channel `compile_loop_body` catches to fall
            // back to the interpreter — instead of letting
            // `inputarg_type_at_strict` hard-panic the worker thread.
            if (0..n).any(|i| ctx.inputarg_type_at(i).is_none()) {
                return Err(crate::optimize::InvalidLoop(
                    "next_iteration_args longer than inputargs (full-body-walk \
                     cross-loop cut over a forced heap virtual)",
                ));
            }
            // resoperation.py InputArgInt/727/739 InputArg{Int,Ref,Float}: mint typed
            // variants for each Phase 2 source slot from `inputarg_types`
            // (history.py:220 box.type). Variant-aware Eq requires the
            // OpRef minted here to match the typed `OpRef::input_arg_typed`
            // emitted at trace start (pyre/pyre-jit-trace/src/trace.rs).
            let typed_inputargs: Vec<OpRef> = (0..n)
                .map(|i| {
                    // opencoder.py:259 inputarg_from_tp parity — strict
                    // box.type lookup, no InputArgVoid fallback.
                    let pos = inputarg_base + i as u32;
                    OpRef::input_arg_typed(pos, ctx.inputarg_type_at_strict(i))
                })
                .collect();
            let source_set: crate::FxIndexSet<OpRef> = typed_inputargs.iter().copied().collect();
            let targetargs: Vec<OpRef> = (0..n)
                .map(|i| {
                    let source = typed_inputargs[i];
                    let target = nia[i].to_opref();
                    // Constants don't participate in forwarding.
                    if ctx.get_box_replacement_operand_opt(target).and_then(|cb| cb.const_value()).is_some() {
                        return source;
                    }
                    // Cross-slot collision: target is another slot's source.
                    // Allocate fresh to avoid forwarding overwrite.
                    if target != source && source_set.contains(&target) {
                        // The alias inherits `box.type` from `source`. RPython
                        // Box always carries `box.type` (history.py:220), so
                        // a missing type here is an unrecoverable invariant
                        // violation — panic in release as well as debug.
                        let tp = ctx.opref_type(source).unwrap_or_else(|| {
                            panic!(
                                "cross-slot collision: source {:?} has no resolvable box.type (RPython invariant violated)",
                                source,
                            )
                        });
                        // `fresh` is a producer-less resop: mint its bound
                        // box up front. `source` is a Phase 2 inputarg whose
                        // slot `ensure_inputarg_bindings` materialized above,
                        // so it resolves without minting.
                        let (fresh, b_fresh) = ctx.reserve_virtual_box(tp);
                        let b_source = ctx
                            .get_box_replacement_operand_opt(source)
                            .expect("Phase 2 source inputarg must have a materialized operand slot");
                        ctx.make_equal_to(&b_source, &b_fresh);
                        fresh
                    } else {
                        source
                    }
                })
                .collect();
            // unroll.py import_state: forwarding + make_inputargs
            // + install_imported_virtuals + ShortPreambleBuilder/produce_op.
            let _label_args = crate::optimizeopt::unroll::import_state_full(
                &targetargs,
                exported_state,
                self,
                &mut ctx,
            );
            // unroll.py:483 `except InvalidLoop`: an incompatible imported
            // virtual state recorded a deferred signal during import; abandon
            // the loop before running Phase 2 against an incomplete import.
            if let Some(e) = ctx.take_invalid_loop() {
                return Err(e);
            }
            if crate::optimizeopt::majit_log_enabled() {
                if let Some((base, ref sa)) = ctx.imported_virtual_args {
                    eprintln!(
                        "[jit] virtual_args from import_state: base={} total={} virtuals={:?}",
                        base,
                        sa.len(),
                        &sa[base..]
                    );
                }
                if !self.imported_virtuals.is_empty() {
                    for slot in 0..ctx.num_inputs as usize {
                        let raw_pos = ctx.inputarg_base + slot as u32;
                        // history.py:220 box.type invariant: every InputArg
                        // carries a `.type`. opencoder.py:259
                        // `inputarg_from_tp(arg.type)` always types its
                        // result; missing type is a bookkeeping bug, even
                        // on this debug-log path.
                        let tp = ctx.inputarg_type_at_strict(slot);
                        let raw = OpRef::input_arg_typed(raw_pos, tp);
                        eprintln!(
                            "[jit] import_state_resolved: raw={raw:?} resolved={:?}",
                            ctx.get_replacement_opref(raw)
                        );
                    }
                }
            }
        }
        // Restore the temporarily-taken imported_loop_state.
        self.imported_loop_state = imported_loop_state_taken;

        // RPython shortpreamble.py: PureOp.produce_op stores PreambleOp
        // directly in opt.optimizer.optpure. In majit, imported short pure ops
        // are first collected in ctx.imported_short_pure_ops, then transferred
        // to the OptPure pass here (matching RPython's produce_op timing).
        if !ctx.imported_short_pure_ops.is_empty() {
            for pass in &mut self.passes {
                pass.install_preamble_pure_ops(&mut ctx);
            }
        }

        // RPython optimizer.py new_const_item: JUMP/FINISH always breaks the main
        // loop. flush() is called before JUMP is processed.
        let mut last_op = None;
        for op in ops {
            if op.opcode == OpCode::Jump || op.opcode == OpCode::Finish {
                last_op = Some((**op).clone());
                break;
            }
            self.propagate_one(op, &mut ctx)?;
        }

        // RPython: flush() before JUMP processing (export_state calls flush
        // before get_virtual_state). Phase 2 skips flush.
        if !self.skip_flush {
            self.flush(&mut ctx)?;
        }

        // RPython unroll.py:454-457:
        //     end_args = [self.optimizer.force_box_for_end_of_preamble(a)
        //                 for a in original_label_args]
        //     self.optimizer.flush()
        //     virtual_state = self.get_virtual_state(end_args)
        //
        // The virtual state is captured AFTER force_box_for_end_of_preamble
        // and AFTER flush, so its `Virtual` / `VStruct` entries correspond
        // exactly to the boxes whose `info.is_virtual()` is still true at
        // that moment (e.g. virtualizables that the force pass intentionally
        // leaves virtual). This invariant is what `make_inputargs` /
        // `make_inputargs_and_virtuals` rely on when they call
        // `enum_forced_boxes` — the validating walker raises
        // `VirtualStatesCantMatch` if a `Virtual` entry's resolved opref
        // is not actually virtual.
        //
        // Capture is performed below, *after* the terminal_op force pass
        // and `flush()` have run inside the `exported_loop_state` map
        // closure, mirroring RPython's order. Keep the raw jump args here:
        // unroll.py:452-455 passes `original_label_args` directly to
        // `force_box_for_end_of_preamble`, and optimizer.py returns
        // the original box unchanged when no force is required.
        let pre_jump_resolved_args = last_op
            .as_ref()
            .filter(|op| op.opcode == OpCode::Jump)
            .map(|jump_op| jump_op.getarglist_copy());

        // RPython optimizer.py (_propagate_all_forward):
        //     if flush:
        //         self.flush()
        //         if last_op:
        //             self.send_extra_operation(last_op)
        //
        // flush=False is used ONLY by optimize_preamble (Phase 1) and
        // optimize_peeled_loop (Phase 2) — terminal_op is returned in
        // BasicLoopInfo.jump_op for unroll.rs to handle.
        //
        // flush=True (default) is used by optimize_loop (compile_loop) and
        // finish_and_compile — terminal op is sent through passes and ends
        // up in new_operations naturally.
        if let Some(mut terminal_op) = last_op {
            // RPython unroll.py:126-127 parity:
            //   for a in end_jump.getarglist():
            //       self.force_box_for_end_of_preamble(get_box_replacement(a))
            //
            // Call force_at_the_end_of_preamble on ALL args (not just
            // Ref-mismatch ones). RPython's force_box_for_end_of_preamble
            // handles both Ref virtuals (type 'r') and raw-ptr-int (type 'i'):
            //   - Ref virtual: recursively process fields, keep virtual alive
            //   - Raw ptr int: force via optearlyforce
            //   - Non-virtual: return get_box_replacement(op) unchanged
            //
            // Additionally, in majit OpRef forwarding is typeless, so a Ref
            // virtual can forward to a Float (SameAsF). When that happens,
            // force_box materializes the virtual to preserve Ref type.
            let inputargs = self.trace_inputargs.clone();
            // Phase 1: resolve and call force_at_the_end_of_preamble on all args
            let resolved_args: Vec<OpRef> = terminal_op
                .args_slice()
                .iter()
                .map(|arg| arg.get_box_replacement(false).to_opref())
                .collect();
            for arg in terminal_op.args_slice().iter() {
                self.force_box_for_end_of_preamble(&arg.get_box_replacement(false), &mut ctx);
            }
            // Phase 2: re-resolve after forcing (force may have changed forwarding)
            // and fix Ref→non-Ref type crossings by force_box
            let mut force_needed: Vec<usize> = Vec::new();
            // optimizer.py force_box loop parity:
            //   for i in range(op.numargs()): op.setarg(i, force_box(...))
            //
            // `trace_inputargs` names the positions this JUMP writes into only
            // on the unroll paths, where unroll.py:454 builds `end_args` from
            // `original_label_args`: the preamble closes back to the loop start,
            // and the peeled body closes back to the LABEL made from those same
            // args. A bridge's JUMP targets a different trace's label while
            // `trace_inputargs` holds the guard's failargs — two unrelated
            // lists, so a positional Ref check reads arbitrary positions. There
            // the type contract belongs to virtual-state matching
            // (virtualstate.py `generate_guards`), and what applies here
            // is the plain force_box loop.
            let inputargs_are_the_jump_target = !self.building_bridge;
            for i in 0..terminal_op.num_args() {
                let arg = terminal_op.arg(i);
                let resolved = ctx.resolve_operand_operand(&arg).to_opref();
                let expected_ref = inputargs_are_the_jump_target
                    && i < inputargs.len()
                    && inputargs[i].ty() == Some(majit_ir::Type::Ref);
                // setup_optimizations seeds `trace_inputargs` into
                // ctx.inputargs (optimizer.py:34), and `opref_type`
                // consults it via the inputarg-slot fallback after the
                // op/value_types chain. PtrInfo presence is an additional
                // Ref-only side channel for inputargs not in `new_operations`.
                let resolved_has_ptr_info = ctx
                    .resolve_operand_operand_opt(&arg)
                    .as_ref()
                    .is_some_and(|b| ctx.has_ptr_info(b));
                let resolved_is_ref =
                    ctx.opref_type(resolved) == Some(majit_ir::Type::Ref) || resolved_has_ptr_info;
                if expected_ref
                    && !resolved_is_ref
                    && ctx
                        .get_box_replacement_operand_opt(resolved)
                        .and_then(|cb| cb.const_value())
                        .is_none()
                {
                    let arg_is_virtual = ctx
                        .resolve_operand_operand_opt(&arg)
                        .as_ref()
                        .is_some_and(|b| ctx.is_virtual(b));
                    if arg_is_virtual {
                        force_needed.push(i);
                    } else {
                        // RPython Box parity: a ref-typed box never forwards to
                        // an int/float box. majit's flat OpRef forwarding can
                        // collapse a boxed W_Int/W_Float local to its raw payload,
                        // but end-of-preamble JUMP args must preserve the ref box
                        // identity at ref-typed positions.
                        //
                        // Keep the original ref-typed arg instead of substituting
                        // the non-ref replacement. This matches the upstream
                        // force_box_for_end_of_preamble contract more closely than
                        // allowing Ref -> Float/Int type substitution at the JUMP.
                        if crate::majit_log_enabled() {
                            eprintln!(
                                "[jump-type] arg {i}: label position is Ref, replacement is not; \
                                 keeping the original arg"
                            );
                        }
                    }
                } else {
                    let arg = ctx.materialize_operand_at(resolved);
                    terminal_op.setarg(i, arg);
                }
            }
            for i in force_needed {
                let original = terminal_op.arg(i);
                let b_forced = self.force_box(&original, &mut ctx);
                terminal_op.setarg(i, b_forced);
            }
            if self.skip_flush {
                // flush=False: store for caller to consume.
                self.terminal_op = Some(terminal_op);
            } else {
                // flush=True: send through passes (optimizer.py).
                self.send_extra_operation(&OpRc::new(terminal_op.clone()), &mut ctx)?;
            }
        }

        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // in guard fail_args. Phase 2 guards may inherit NONE from Phase 1
        // virtualization — rescan resolves these using imported_virtuals.
        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // RPython parity: store_final_boxes_in_guard in ctx.emit() handles
        // virtual tagging inline at each guard emit. No post-pass rescan.

        // RPython Box type parity: each Box carries its type intrinsically
        // (resoperation.py `opclasses[opnum].type` for ResOps,
        // history.py:220 `InputArg{Int,Ref,Float}.type` for inputargs).
        // Phase 1 emit ops carry their own `op.type_`; Phase 1 inputarg slot
        // OpRefs are resolved through `OptContext::inputarg_type` (which
        // falls back to the shared `inputarg_types` Vec for low OpRefs in
        // `[0, num_inputs)`). ExportedState carries boundary input types
        // explicitly, matching RPython's typed Box objects without a
        // phase-wide OpRef→Type side table.
        self.phase1_emit_ops.clear();
        for op in &ctx.new_operations {
            if !op.pos().get().is_none() && op.type_ != majit_ir::Type::Void {
                self.phase1_emit_ops.push(op.clone());
            }
        }
        // PyPy parity: a folded `ResOperation` keeps its Python identity
        // through `_forwarded` links (resoperation.py); the
        // `optimize_peeled_loop` (compile.py) chain walk reaches it
        // via `partial_trace.operations`.
        //
        // pyre's per-iter `TraceIterator::next()` (`opencoder.rs`)
        // pushes a fresh resop operand slot for every visited op
        // BEFORE the optimizer pipeline decides whether to emit. Two
        // categories of slot escape the `new_operations` carry above
        // and need explicit handling so retrace sees the same
        // `Forwarded::Op(_)` chain targets Phase 1 installed:
        //
        //   - Unbound orphan: the pipeline folded/dropped the op so
        //     `ctx.emit` never ran. Synthesize a `SameAs` stand-in
        //     and bind the slot.
        //   - Synthetic-bound: a forward reference reached `materialize_operand_at`
        //     first; `materialize_operand_at` minted a `SameAs` stand-in into
        //     `ctx.resop_refs[idx]` and `bind_op`'ed it. When `emit`
        //     never arrived to upgrade the binding to a real producer,
        //     the stand-in itself is the chain target carrier.
        //
        // In both cases the stand-in OpRc must travel into
        // `phase1_emit_ops` (and from there into
        // `ExportedState.partial_trace_operations`) so retrace reads it as
        // part of the partial trace.
        // `live_synthetics` is the incrementally-maintained set of synthetic
        // stand-ins (mint_synthetic_resop / bind_input_resops) whose position
        // was never superseded by an `emit` — the box-bound-to-synthetic
        // producers that must travel into `phase1_emit_ops`. Positions that
        // were never synthesized carry no producer, so nothing else needs
        // draining here.
        self.phase1_emit_ops
            .extend(ctx.live_synthetics.iter().cloned());
        // Transfer exported virtual state from context to optimizer
        // RPython BasicLoopInfo: quasi_immutable_deps collected during optimization
        self.quasi_immutable_deps = std::mem::take(&mut ctx.quasi_immutable_deps);
        self.imported_short_aliases = ctx.used_imported_short_aliases();
        self.imported_short_preamble = ctx.build_imported_short_preamble();
        self.imported_short_preamble_builder = ctx.imported_short_preamble_builder.clone();
        // optimizer.py:294: patchguardop is set exclusively by
        // GUARD_FUTURE_CONDITION during optimization. No fallback.
        self.patchguardop = ctx.patchguardop.clone();
        // JUMP location: in new_operations (flush=True path where JUMP was
        // processed through passes), or terminal_op (skip_flush path).
        let jump = ctx
            .new_operations
            .iter()
            .rfind(|op| op.opcode == OpCode::Jump)
            .cloned()
            .or_else(|| {
                self.terminal_op
                    .clone()
                    .filter(|op| op.opcode == OpCode::Jump)
                    .map(OpRc::new)
            });
        let mut loop_info = BasicLoopInfo::new(
            self.trace_inputargs.clone(),
            self.quasi_immutable_deps.clone(),
            jump.as_ref().map(|op| (**op).clone()),
        );
        // RPython compile.py:327 `loop.operations = ([start_label] + preamble_ops
        // + loop_info.extra_same_as + loop_info.extra_before_label + [label_op]
        // + loop_ops)`: alias SameAs ops allocated during the preamble
        // export dedup belong to `loop_info.extra_same_as`, NOT to the
        // preamble body. Keep them in a side vector so they land at the
        // spliced parity position below instead of appearing in
        // `ctx.new_operations` past the terminator.
        // `match` (not `jump.map(|jump| ...)`) so the `return Err(InvalidLoop)`
        // for the preview virtual-state mismatch below propagates out of
        // `optimize_with_constants_and_inputs_at`, not just the closure.
        let building_bridge = self.building_bridge;
        // `Optimizer.optimize_loop` (optimizer.py) is the whole of
        // `SimpleCompileData.optimize`; there is no `optimize_preamble` behind
        // it to export for. See [`Self::simple_compile`].
        self.exported_loop_state = match jump.filter(|_| !self.simple_compile) {
            Some(jump) => 'export: {
                // RPython unroll.py:454-457 order:
                //   end_args = [force_box_for_end_of_preamble(a) for a ...]
                //   self.optimizer.flush()
                //   virtual_state = self.get_virtual_state(end_args)
                // — VS captured AFTER force + flush.
                let original_jump_args: Vec<OpRef> = pre_jump_resolved_args
                    .clone()
                    .map(|v| v.iter().map(|b| b.to_opref()).collect())
                    .unwrap_or_else(|| {
                        jump.args_slice()
                            .iter()
                            .map(|a| a.get_box_replacement(false).to_opref())
                            .collect()
                    });
                let mut resolved_args = original_jump_args.clone();
                // Dedup: two cases require SameAs allocation at end of preamble.
                //
                // Case A — duplicate args: when two JUMP slots reference the same
                // OpRef (e.g. b and t in fib_loop after b = t aliasing), create a
                // fresh SameAs for the second occurrence so each slot has a
                // distinct identity.
                //
                // Case B — preamble inputarg position used by a non-self slot:
                // when JUMP slot j carries OpRef::int_op(k) and k < num_inputs, k != j,
                // the export hands Phase 2 a `next_iteration_args[j] = OpRef::int_op(k)`
                // entry that collides with body inputarg slot k's own source
                // position. import_state forwards both `OpRef::int_op(j) → OpRef::int_op(k)` and
                // `OpRef::int_op(k) → nia[k]`, and get_box_replacement walks the chain
                // through OpRef::int_op(k), making body inputarg j resolve to nia[k]
                // instead of OpRef::int_op(k). RPython avoids this with fresh-Box identity
                // per phase; majit's flat OpRef space needs an explicit SameAs
                // alias so nia[j] points outside the body inputarg position range.
                //
                // A bridge needs the aliases too: its retrace arm exports from
                // `ctx.preamble_end_args`, which is built from these args.
                {
                    let mut seen: crate::FxIndexSet<OpRef> = crate::FxIndexSet::default();
                    // RPython parity: positions already holding an emitted op
                    // are phase 1 results, not body inputarg sources. Only
                    // the UNUSED positions in 0..num_inputs correspond to
                    // trace inputargs (`InputArgRef/Int/Float` in RPython).
                    let emitted_positions: crate::FxIndexSet<OpRef> = ctx
                        .new_operations
                        .iter()
                        .map(|op| op.pos().get())
                        .filter(|p| !p.is_none())
                        .collect();
                    let original_args = resolved_args.clone();
                    for (slot_idx, arg) in resolved_args.iter_mut().enumerate() {
                        if ctx
                            .get_box_replacement_operand_opt(*arg)
                            .and_then(|cb| cb.const_value())
                            .is_some()
                            || *arg == OpRef::NONE
                        {
                            continue;
                        }
                        let is_dup = !seen.insert(*arg);
                        // Case B fires only when slot k (where `k = arg.raw()`, the
                        // position the current slot points at) holds a DIFFERENT
                        // OpRef. If slot k self-forwards (original_args[k] ==
                        // OpRef::int_op(k)), Phase 2 sets `forwarding[OpRef::int_op(k)] = OpRef::int_op(k)`
                        // which is a no-op in make_equal_to — no chain forms, so no
                        // aliasing is needed.
                        let target_slot = arg.raw() as usize;
                        let target_slot_self_forwards = original_args
                            .get(target_slot)
                            .is_none_or(|other| *other == *arg);
                        let is_cross_inputarg = target_slot < num_inputs
                            && target_slot != slot_idx
                            && !emitted_positions.contains(arg)
                            && !target_slot_self_forwards;
                        if !is_dup && !is_cross_inputarg {
                            continue;
                        }
                        let orig = *arg;
                        // history.py `record_same_as(box)` reads
                        // `box.type` directly to pick `same_as_i/r/f` — there is
                        // no guess-on-miss path in RPython. Match strict parity by
                        // requiring `opref_type` to resolve; a None here is a
                        // bookkeeping bug, not a recoverable case.
                        let arg_type = ctx.opref_type(orig).expect(
                            "propagate_from_pass_range SameAs: source OpRef missing Box.type",
                        );
                        let same_as = OpCode::same_as_for_type(arg_type);
                        let fresh = ctx.alloc_op_position_typed(arg_type);
                        let arg0 = ctx.materialize_operand_at(orig);
                        let mut op = Op::new(same_as, std::slice::from_ref(&arg0));
                        op.pos().set(fresh);
                        // unroll.py:146 + compile.py:327 parity: accumulate the
                        // alias op in `extra_same_as` and splice it between the
                        // preamble body and the label at final assembly. Emitting
                        // directly into `ctx.new_operations` would push the op
                        // past the already-sent terminal JUMP and force the
                        // loop-tail relocation workaround below.
                        loop_info.extra_same_as.push(op);
                        let orig_box = ctx.get_box_replacement_operand_opt(orig);
                        if let Some(info) = orig_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
                            let fresh_info = match info {
                                crate::optimizeopt::info::PtrInfo::Virtual(mut vinfo) => {
                                    for field in &mut vinfo.fields {
                                        let orig_field = field.1.to_opref();
                                        // RPython Box type parity: the alias
                                        // inherits `box.type` from the original
                                        // field. RPython Box always carries
                                        // `box.type` (history.py:220), so a
                                        // missing type here is an unrecoverable
                                        // invariant violation — panic in release
                                        // as well as debug.
                                        let tp = ctx.opref_type(orig_field).unwrap_or_else(|| {
                                        panic!(
                                            "virtual-field alias: orig_field {:?} has no resolvable box.type (RPython invariant violated)",
                                            orig_field,
                                        )
                                    });
                                        // Producer-less alias resop: mint its
                                        // bound box up front and forward it to the
                                        // original field box.
                                        let (ff, b_ff) = ctx.reserve_virtual_box(tp);
                                        let b_orig = ctx.get_box_replacement_operand(orig_field);
                                        ctx.make_equal_to(&b_ff, &b_orig);
                                        let _ = ff;
                                        field.1 = b_ff;
                                    }
                                    crate::optimizeopt::info::PtrInfo::Virtual(vinfo)
                                }
                                other => other,
                            };
                            if let Some(b) = ctx.get_box_replacement_operand_opt(fresh) {
                                ctx.set_ptr_info(&b, fresh_info);
                            }
                        }
                        *arg = fresh;
                        // After allocating a fresh alias for an inputarg position,
                        // update `seen` so the next slot referencing the same
                        // original OpRef goes through the duplicate path and
                        // gets its own fresh alias too.
                        if is_cross_inputarg {
                            seen.insert(fresh);
                        }
                    }
                }
                // unroll.py:203 self.flush() — force lazy sets before
                // producing short preamble ops (heap.py:53 invariant).
                self.flush(&mut ctx)?;

                // Now force all resolved (and dedup'd) args.
                // unroll.py:454 `end_args = [force_box_for_end_of_preamble(a)
                // for a in original_label_args]`.
                ctx.preamble_end_args = Some(
                    resolved_args
                        .iter()
                        .map(|&arg| match ctx.get_box_replacement_operand_opt(arg) {
                            Some(b) => self.force_box_for_end_of_preamble(&b, &mut ctx).to_opref(),
                            None => arg,
                        })
                        .collect(),
                );
                // unroll.py: `optimize_bridge` calls `export_state`
                // only in its retrace arm, after `jump_to_existing_trace`
                // found no match. Steps (1) and (2) stay eager so that arm
                // can read `ctx.preamble_end_args`; step (3) is published
                // there immediately before the late export.
                if building_bridge {
                    break 'export None;
                }
                // unroll.py `optimize_peeled_loop` exports nothing: only the
                // preamble's `export_state` feeds the peeled loop.
                if self.imported_loop_state.is_some() {
                    break 'export None;
                }
                self.publish_preview_short_state(&mut ctx, &resolved_args)?;
                let jump_arglist_oprefs: Vec<OpRef> =
                    jump.args_slice().iter().map(|a| a.to_opref()).collect();
                let exported_int_bounds =
                    self.collect_exported_int_bounds(&jump_arglist_oprefs, &mut ctx);
                // RPython unroll.py:186-193 + compile.py: `info.renamed_inputargs`
                // are the fresh per-iteration boxes from `trace.get_iter()`. They
                // live in this run's iteration namespace, not the original
                // frontend's. In pyre this maps to `[inputarg_base..inputarg_base
                // + num_inputs)`: `inputarg_base = 0` for top-level loops
                // (compile_loop / compile_retrace) where the frontend already
                // owns `[0, num_inputs)`, and `inputarg_base = bridge_inputarg_base`
                // for bridges, where `prepare_bridge_trace_for_optimizer`
                // in pyjitpl.rs shifts
                // the iteration into a disjoint range.  Use `num_inputs` (the
                // external loop-entry contract count) rather than `ctx.num_inputs`
                // (which may be widened by virtualizable expansion).
                // resoperation.py InputArgInt/727/739 InputArg{Int,Ref,Float}: each
                // renamed inputarg's Box carries `.type` intrinsically
                // (history.py:220). Mint typed variants from `inputarg_types`
                // so the exported state's OpRefs match what Phase 2 will see
                // under variant-aware Eq.
                let renamed_inputargs: Vec<OpRef> = (0..num_inputs)
                    .map(|i| {
                        // opencoder.py:259 inputarg_from_tp parity — strict
                        // box.type lookup, no InputArgVoid fallback.
                        let pos = ctx.inputarg_base + i as u32;
                        OpRef::input_arg_typed(pos, ctx.inputarg_type_at_strict(i))
                    })
                    .collect();
                Some(crate::optimizeopt::unroll::export_state(
                    &original_jump_args,
                    &renamed_inputargs,
                    self,
                    &mut ctx,
                    Some(&exported_int_bounds),
                ))
            }
            None => None,
        };
        // RPython parity: propagate patchguardop to ExportedState so Phase 2
        // can use it for extra_guards from virtualstate (unroll.py:333-336).
        if let Some(ref mut es) = self.exported_loop_state {
            es.patchguardop = self.patchguardop.clone();
        }

        // RPython export_state() flushes force artifacts into the preamble
        // before building the exported loop state. If the loop header needs
        // additional inputargs, the corresponding SETFIELD/SETARRAYITEM must
        // remain in the trace rather than being silently discarded.
        // final_num_inputs = original inputs + virtual inputs added by passes.
        // Pipeline passes (e.g., virtualizable) may bump ctx.num_inputs by
        // adding synthetic input slots. The delta is the count of virtual
        // inputargs the optimizer needs to materialize at trace entry.
        let num_virtual_inputs = (ctx.num_inputs as usize).saturating_sub(num_inputs);
        self.final_num_inputs = num_inputs + num_virtual_inputs;

        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // RPython parity: store_final_boxes_in_guard handles virtual tagging
        // and rd_numb production inline. No post-pass rescan needed.

        // Finalize cascade removed. PyPy walks args ONCE at op-emit
        // time via `_emit_operation:614-625 force_box`; never re-walks. The
        // cited `compile.py:emit_op:403-423` is a virtualizable-only patcher
        // (5 callers all inside `patch_new_loop_to_load_virtualizable_fields`),
        // not a general finalize pass. Pyre's emit-time arg walking happens
        // in two PyPy-parity locations:
        //   1. `propagate_from_pass_range` — incoming op args
        //      resolved via `ctx.get_box_replacement` BEFORE pass dispatch.
        //   2. `Optimizer::emit_operation` — `force_box` on every
        //      arg unconditionally (PyPy `optimizer.py` parity).
        // After emit, args are frozen on the op. Postprocess setting
        // `box._forwarded = Const` (e.g., `make_constant` from
        // `optimize_GUARD_FALSE`) must NOT retroactively rewrite already-
        // emitted ops' args — doing so converts a runtime-typed GuardFalse
        // into `GuardFalse Const(0)` whose machine code is always-success
        // (cmp 0,0; jz), causing infinite loops in compiled traces.

        // RPython keeps constants as Const boxes, not SameAs placeholder ops in
        // the final trace. Drop constant-only SameAs placeholders before the
        // backend sees the trace; their OpRefs remain available through the
        // constants table.
        // Manual filter (instead of `.retain`) because the predicate
        // borrows `ctx` while `ctx.new_operations` would be borrowed
        // mutably by `retain`.
        let keep: bit_set::BitSet = ctx
            .new_operations
            .iter()
            .enumerate()
            .filter(|(_, op)| !Self::is_constant_placeholder_op(op, &ctx))
            .map(|(i, _)| i)
            .collect();
        let mut idx = 0usize;
        ctx.new_operations.retain(|_| {
            let kept = keep.contains(idx);
            idx += 1;
            kept
        });

        // Drain remaining extra ops. send_extra_operation may raise
        // InvalidLoop; propagate it.
        self.drain_extra_operations_from(0, &mut ctx)?;
        // Second finalize cascade also removed, on the same rationale:
        // each emit_extra-queued op runs through `propagate_from_pass_range`
        // which resolves its incoming args at input time via
        // `ctx.get_box_replacement`, and through `Optimizer::emit_operation`
        // which `force_box`es every arg. No retroactive walking of pre-existing
        // new_operations is needed.
        // RPython: no equivalent filter — Box identity guarantees all
        // references are valid. The retain filter was a majit safety net
        // but incorrectly dropped valid ops (e.g., IntAddOvf referencing
        // inputarg positions beyond final_num_inputs in bridge traces).

        // `compile.py:327` order:
        //   `[start_label] + preamble_ops + extra_same_as +
        //    extra_before_label + [label_op] + loop_ops`.
        // The SameAs aliases emitted in the JUMP-arg dedup loop above
        // (`shortpreamble.py add_preamble_op extra_same_as`) and any `flush()` /
        // `drain_extra_operations_from` ops that followed are structurally
        // part of `extra_same_as`; upstream splices that list BETWEEN
        // `preamble_ops` and the label, so in pyre's single-phase flow
        // they must sit before the preamble-terminating JUMP. Move the
        // last JUMP to the tail to restore that ordering — running AFTER
        // all emits but BEFORE the remap pass so positions stay
        // sequential in trace order.
        if let Some(jump_idx) = ctx
            .new_operations
            .iter()
            .rposition(|op| op.opcode == OpCode::Jump)
            && jump_idx + 1 < ctx.new_operations.len()
        {
            let jump_op = ctx.new_operations.remove(jump_idx);
            ctx.new_operations.push(jump_op);
        }
        // `new_operations_index` intentionally not maintained across this
        // finalization tail (jump reorder + the position remap below): it
        // feeds `find_producer_op`, which is only queried during the forward
        // propagate/emit phase that precedes this point. The context is
        // cleared (`clear_newoperations`) before any reuse re-enters that
        // phase.

        // Remap ALL positions: virtual inputs go to num_inputs..final_num_inputs,
        // This ensures no position collisions between input block params and ops.
        if num_virtual_inputs > 0 {
            let fni = self.final_num_inputs as u32;
            let mut remap: crate::FxIndexMap<u32, u32> = crate::FxIndexMap::default();

            // Virtual input positions: optimizer used num_inputs+k, backend needs num_inputs+k
            for k in 0..num_virtual_inputs {
                let opt_pos = (num_inputs + k) as u32;
                let be_pos = (num_inputs + k) as u32;
                if opt_pos != be_pos {
                    remap.insert(opt_pos, be_pos);
                }
            }

            // Op positions: reassign ALL ops to start from final_num_inputs.
            for (new_idx, op) in ctx.new_operations.iter_mut().enumerate() {
                let new_pos = fni + new_idx as u32;
                if !op.pos().get().is_none() {
                    remap.insert(op.pos().get().raw(), new_pos);
                    op.pos().set(op.pos().get().with_raw(new_pos));
                }
            }

            // Constant-folded operations that were removed from the trace still
            // have their box._forwarded = constbox forwarding step from
            // make_constant/seed_constant. If we leave them at their old
            // positions, they can collide with the freshly compacted op
            // positions above (for example old constant v71 vs new live op v71),
            // and the backend will resolve the live op as the stale constant.
            // Give every such constant-only opref a fresh slot after the last
            // live op, mirroring RPython's separate constant identity.
            // Scan the canonical `_forwarded` hosts for constant-folded ops
            // still at their pre-compact position. At this
            // point set_position has not run, so a folded op's `Op.pos` still
            // holds the old raw index; live emitted ops carry `Forwarded::Op`/
            // info (not Const) so the forwarded filter excludes them, and the
            // `remap.contains_key` guard dedups a position reached through more
            // than one store.
            let mut next_const_pos = fni + ctx.new_operations.len() as u32;
            // Const-folded producers whose `Op.pos` needs aligning to the
            // post-compact slot. Captured here (op + assigned slot) so the
            // production `Op.pos` update below runs without re-walking the
            // stores. The `remap.contains_key` guard makes each old
            // position trigger exactly once, so a producer reachable from
            // more than one canonical store is captured a single time.
            let mut const_remaps: Vec<(majit_ir::OpRc, u32)> = Vec::new();
            let mut consider_const = |op: &majit_ir::OpRc| {
                let old_idx = op.pos().get().raw();
                if remap.contains_key(&old_idx) || old_idx < num_inputs as u32 {
                    return;
                }
                if !op.forwarded().borrow().is_const() {
                    return;
                }
                remap.insert(old_idx, next_const_pos);
                const_remaps.push((op.clone(), next_const_pos));
                next_const_pos += 1;
            };
            for op in &ctx.new_operations {
                consider_const(op);
            }
            for op in &ctx.phase1_emit_ops {
                consider_const(op);
            }
            for op in ctx.resop_refs.values() {
                consider_const(op);
            }
            // Align each const-folded producer's canonical `Op.pos` to its
            // post-compact slot. The `new_operations` loop above already
            // remapped emitted live ops; removed constant-folded ops (their
            // box carries `Forwarded::Const`) had their `Op.pos` left at the
            // pre-compact value. Readers off the canonical `Op.pos`
            // (`merge_backend_constants`) need the post-compact position.
            // Non-const synthetics never entered `remap`, so they keep their
            // position; only const-folded producers are repositioned here.
            for (op, new_pos) in &const_remaps {
                op.pos().set(op.pos().get().with_raw(*new_pos));
            }

            // Apply remap to all args and fail_args. Const operands carry
            // their value inline (history.py:227/268/314); position-remap
            // applies only to body-namespace OpRefs (InputArg* / *Op).
            for op in &ctx.new_operations {
                for i in 0..op.num_args() {
                    // A bound operand (`Operand::Op` / `InputArg`) live-tracks
                    // its producer's `op.pos`, already mutated to the new dense
                    // slot above, so it needs no snapshot rewrite. Only
                    // position-only `Operand::Box` operands carry a stale
                    // pre-remap position the table must rewrite.
                    if op.arg_is_bound(i) {
                        continue;
                    }
                    let arg = op.arg(i).to_opref();
                    if arg.is_constant() {
                        continue;
                    }
                    // A non-const operand reaching const-compact is always
                    // producer-bound (skipped by `arg_is_bound` above), so a
                    // position-only operand at a remapped slot is unreachable
                    // (PYRE_REMAP_PROBE 2026-06-11: 0 fires across check.py
                    // corpus + lib tests). The #9 grind retires the
                    // position-only rewrite; a hard guard replaces it so any
                    // regression panics instead of silently minting a stale
                    // position-only box.
                    if remap.contains_key(&arg.raw()) {
                        unreachable!(
                            "position-only non-const operand hit const-compact remap: {arg:?}"
                        );
                    }
                }
                if let Some(fail_args) = op.guard_fail_args() {
                    for arg in fail_args.iter() {
                        // Same rule as the args loop above: a bound failarg
                        // live-tracks its producer's already-remapped
                        // `op.pos`; re-remapping would double-map. Only
                        // position-only `Operand::Box` snapshots carry a
                        // stale pre-remap position.
                        if arg.is_bound() {
                            continue;
                        }
                        let arg_opref = arg.to_opref();
                        // Unreachable on the same evidence: a non-const
                        // failarg is always producer-bound (PYRE_REMAP_PROBE
                        // 2026-06-11: 0 fires).
                        if !arg_opref.is_constant() && remap.contains_key(&arg_opref.raw()) {
                            unreachable!(
                                "position-only failarg hit const-compact remap: {arg_opref:?}"
                            );
                        }
                    }
                }
            }

            // Constants no longer need remapping — every optimizeopt
            // consumer reads a const-folded op's value off its canonical
            // `_forwarded` host (`Forwarded::Const`), and the const-remap
            // loop above already aligned each such op's `Op.pos` to its
            // post-compact slot. The flat `OptContext.constants` Vec backing
            // has been retired; const values live entirely on the forwarding
            // chain.

            // Remap exported_loop_state OpRefs so Phase 2 sees post-remap
            // positions. Without this, Phase 2's import_boxes maps to
            // pre-remap positions that no longer exist.
            if let Some(ref mut state) = self.exported_loop_state {
                // unroll.py:463 — end_args/infos may contain Const; Const
                // is not a body position so it must not be remapped.
                let remap_opref = |opref: &mut OpRef| {
                    if opref.is_constant() {
                        return;
                    }
                    if let Some(&new_pos) = remap.get(&opref.raw()) {
                        *opref = opref.with_raw(new_pos);
                    }
                };
                // next_iteration_args carries the canonical Phase-1 operands
                // that double as exported_infos keys. They need NO remap: a
                // bound operand live-tracks its producer's `op.pos` (already
                // moved to the new dense slot by the main loop above), a Const
                // carries its value inline (no body position), and Operand has
                // no position-only form — so every carried identity (the
                // exported_infos key) is preserved untouched.
                for arg in &mut state.end_args {
                    remap_opref(arg);
                }
                for arg in &mut state.renamed_inputargs {
                    remap_opref(arg);
                }
                for arg in &mut state.short_inputargs {
                    remap_opref(arg);
                }
                // exported_infos is keyed by box identity (Rc::ptr_eq), so its keys
                // need NO position remap — the lookup matches by Rc, not OpRef. The
                // keys are the same Rcs as next_iteration_args (for end_args) /
                // bound producer boxes (for label/short-box args), whose positions
                // already moved through the shared set_position Cell above; a
                // mem::take + re-insert under a remapped OpRef would only re-mint
                // fresh non-ptr_eq keys and break the carry.
                // Remap exported short boxes, including the const-only replay
                // channel whose fresh operation positions remain live in state.
                for entry in state
                    .exported_short_boxes
                    .iter_mut()
                    .chain(state.const_short_boxes.iter_mut())
                {
                    // Cell::get() returns a copy; the previous
                    // `remap_opref(&mut entry.op.pos().get())` mutated that
                    // temporary and never wrote back.  Read into a local,
                    // remap, then `set(...)` to persist the new OpRef on
                    // the Cell.
                    let mut new_pos = entry.op.pos().get();
                    remap_opref(&mut new_pos);
                    entry.op.pos().set(new_pos);
                    for i in 0..entry.op.num_args() {
                        // Bound operands live-track their producer's already
                        // remapped `op.pos` (the main loop set it above); only
                        // position-only `Operand::Box` snapshots need the table
                        // rewrite. Re-remapping a bound operand would double-map.
                        if entry.op.arg_is_bound(i) {
                            continue;
                        }
                        let mut arg = entry.op.arg(i).to_opref();
                        let pre = arg;
                        remap_opref(&mut arg);
                        // A position-only exported-short-box operand is never
                        // remapped (PYRE_REMAP_PROBE 2026-06-11: 0 fires), so
                        // its frozen position stays valid and needs no rewrite.
                        // A hard guard replaces the position-only re-mint (#9):
                        // any regression panics instead of silently leaving a
                        // stale position.
                        assert!(
                            arg == pre,
                            "position-only exported-short-box arg remapped: {pre:?}"
                        );
                    }
                    if let Some(fa) = entry.op.guard_fail_args() {
                        for arg in fa.iter() {
                            // Bound failargs live-track the producer's
                            // already-remapped pos (same rule as the args
                            // loop above); re-remapping would double-map.
                            if arg.is_bound() {
                                continue;
                            }
                            let mut arg_opref = arg.to_opref();
                            let pre = arg_opref;
                            remap_opref(&mut arg_opref);
                            // Never remapped (PYRE_REMAP_PROBE 2026-06-11:
                            // 0 fires) — frozen position
                            // stays valid, no rewrite; a hard guard replaces the
                            // position-only re-mint (#9).
                            assert!(
                                arg_opref == pre,
                                "position-only exported-short-box failarg remapped: {pre:?}"
                            );
                        }
                    }
                    // same_as_source is a producer-bound operand (or None):
                    // it live-tracks its producer's already-remapped `Op.pos`
                    // through the carried `Rc<Op>`, so there is no separate
                    // position Cell to rewrite. The former bound-box refresh
                    // copied that same `op.pos`; the position-only else-branch
                    // was dead since #173 roots the producer.
                    // res: a producer-bound / const operand live-tracks its
                    // producer through the carried handle — the producer's
                    // `Op.pos` was already remapped above, so there is no
                    // separate position to rewrite (same as `same_as_source`).
                }
            }
        }

        // Preserve final context for jump_to_existing_trace.
        let mut ops = ctx.take_new_operations();

        // RPython compile.py:327 final loop assembly:
        //   loop.operations = ([start_label] + preamble_ops
        //       + loop_info.extra_same_as + loop_info.extra_before_label
        //       + [label_op] + loop_ops)
        //
        // pyre does not split the preamble / body boundary at this layer —
        // both simple-loop and Phase-1-of-unroll consumers receive a single
        // `ops` vector that already carries the terminal JUMP. Matching the
        // RPython ordering means splicing the alias `extra_same_as` ops
        // just ahead of that terminator so they execute at end of preamble
        // and never appear past the Jump. With the export dedup (the
        // `'export:` block's Case A / Case B arms) accumulating into
        // `loop_info.extra_same_as` instead of `ctx.new_operations`, no
        // `ctx.emit`-after-terminator cleanup is needed.
        if !loop_info.extra_same_as.is_empty() || !loop_info.extra_before_label.is_empty() {
            let term_idx = ops
                .iter()
                .position(|op| op.opcode == OpCode::Jump || op.opcode == OpCode::Finish)
                .unwrap_or(ops.len());
            let extra_same_as_len = loop_info.extra_same_as.len();
            for (offset, op) in loop_info.extra_same_as.into_iter().enumerate() {
                ops.insert(term_idx + offset, OpRc::new(op));
            }
            let before_label_idx = term_idx + extra_same_as_len;
            for (offset, op) in loop_info.extra_before_label.into_iter().enumerate() {
                ops.insert(before_label_idx + offset, OpRc::new(op));
            }
        }
        // resume.py:411-417 parity: store_final_boxes_in_guard
        // (`optimizeopt/mod.rs`) already replaces TAGCONST/TAGVIRTUAL fail_args
        // entries with OpRef::NONE via the snapshot-driven numbering pass
        // (`liveboxes = [None] * n; liveboxes[i] = box for TAGBOX`).
        // No additional const filtering is needed here.
        if crate::debug::have_debug_prints() {
            let cmf_count = ops.iter().filter(|o| o.opcode.is_call_may_force()).count();
            let gnf_count = ops
                .iter()
                .filter(|o| matches!(o.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2))
                .count();
            let _s = crate::debug::scope("jit-optimizer");
            crate::debug::debug_print(&format!(
                "final ops: total={} call_may_force={cmf_count} guard_not_forced={gnf_count}",
                ops.len(),
            ));
            if cmf_count == 0 && gnf_count > 0 {
                for (i, op) in ops.iter().enumerate() {
                    crate::debug::debug_print(&format!(
                        "idx={i} {:?} pos={:?}",
                        op.opcode,
                        op.pos().get()
                    ));
                }
            }
        }
        self.final_ctx = Some(ctx);
        Ok(ops)
    }

    /// compile.py `SimpleCompileData.optimize` → `Optimizer.optimize_loop`.
    ///
    /// The caller has already done `trace.get_iter()` (`prepare_bridge_trace`).
    /// `resumestorage` arrives as `pending_bridge_rd` and is deserialized
    /// inside `optimize_with_constants_and_inputs_at`, matching
    /// `optimize_loop`'s `if resumestorage: deserialize_optimizer_knowledge`.
    /// `BasicLoopInfo.final()` is always True, so there is no retrace.
    pub(crate) fn optimize_loop(
        &mut self,
        ops: &[majit_ir::OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        pending_bridge_rd: Option<PendingBridgeRd>,
        inputarg_base: u32,
    ) -> Result<Vec<majit_ir::OpRc>, crate::optimize::InvalidLoop> {
        self.simple_compile = true;
        self.pending_bridge_rd = pending_bridge_rd;
        // Bridge ops are a fresh TraceIterator's reminted slots; producer
        // lookup runs off `resop_refs` after `bind_input_resops`.
        self.explicit_input_ops_seed = Some(Vec::new());
        let max_op_pos = ops
            .iter()
            .filter_map(|op| {
                if op.pos().get().is_none() || op.pos().get().is_constant() {
                    None
                } else {
                    Some(op.pos().get().raw())
                }
            })
            .max();
        let start_next_pos = max_op_pos
            .map(|p| p + 1)
            .unwrap_or(inputarg_base + num_inputs as u32)
            .max(inputarg_base + num_inputs as u32);
        self.optimize_with_constants_and_inputs_at(
            ops,
            constants,
            num_inputs,
            inputarg_base,
            start_next_pos,
            false,
        )
    }

    /// unroll.py: optimize_bridge()
    ///
    /// Optimizes a bridge trace and redirects its terminal JUMP to the
    /// appropriate loop body target token, falling back to the preamble
    /// when no match is found.
    ///
    /// `retraced_count` / `retrace_limit`: RPython history.py
    /// JitCellToken.retraced_count tracking. When retrace_limit > 0 and
    /// no existing trace matches, export_state creates a new specialization.
    /// Default retrace_limit = 0 (disabled, warmstate.py PARAMETERS).
    ///
    /// Returns `(optimized_ops, retrace_requested)`. When retrace is
    /// requested, this method increments `retraced_count` itself
    /// (`unroll.py` `cell_token.retraced_count += 1`). The caller writes
    /// the updated count back onto the entered JitCellToken.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub(crate) fn optimize_bridge(
        &mut self,
        ops: &[majit_ir::OpRc],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        front_target_tokens: &mut [crate::history::TargetToken],
        runtime_boxes: &[OpRef],
        inline_short_preamble: bool,
        retraced_count: &mut u32,
        retrace_limit: u32,
        pending_bridge_rd: Option<PendingBridgeRd>,
        _loop_num_inputs: Option<usize>,
        // Disjoint OpRef namespace for bridge inputargs. RPython `opencoder.py TraceIterator
        // TraceIterator.__init__` allocates fresh `InputArg` Python
        // objects per iteration so bridges carry Python `is` identity
        // distinct from the parent loop's boxes. Pyre's flat
        // `OpRef::int_op(u32)` lacks identity, so `compile_bridge` calls
        // `prepare_bridge_trace_for_optimizer` (pyjitpl.rs) which
        // walks the recorded ops through a fresh `TraceIterator` with
        // `start_fresh = bridge_inputarg_base`, allocating OpRefs in
        // `[bridge_inputarg_base..)`. This signature carries the same
        // base into `optimize_with_constants_and_inputs_at` so step 3
        // seeds inputarg types at the shifted slots.
        bridge_inputarg_base: u32,
    ) -> Result<(Vec<majit_ir::OpRc>, bool), crate::optimize::InvalidLoop> {
        // bridgeopt.py: deserialize_optimizer_knowledge
        // Store as pending — setup() inside optimize_with_constants_and_inputs
        // clears pass state, so we apply AFTER setup.
        self.pending_bridge_rd = pending_bridge_rd;
        // unroll.py:193: info, ops = self.propagate_all_forward(trace, ...)
        // Bridge ops use a disjoint OpRef
        // namespace `[bridge_inputarg_base..)` (set by
        // `prepare_bridge_trace_for_optimizer`). Drive the optimizer
        // through the shifted entry point so `inputarg_base` /
        // `start_next_pos` align with the prepared ops. Compute
        // `start_next_pos` from the maximum raw `op.pos` so the
        // optimizer's emit cursor does not collide with the bridge's
        // own already-allocated result slots.
        let max_op_pos = ops
            .iter()
            .filter_map(|op| {
                if op.pos().get().is_none() || op.pos().get().is_constant() {
                    None
                } else {
                    Some(op.pos().get().raw())
                }
            })
            .max();
        let start_next_pos = max_op_pos
            .map(|p| p + 1)
            .unwrap_or(bridge_inputarg_base + num_inputs as u32)
            .max(bridge_inputarg_base + num_inputs as u32);
        // Bridge ops are a fresh `TraceIterator`'s planted UNBOUND resop slots
        // (`bind_input_resops` binds them later, after `input_ops` is built),
        // so the input-ops seed is empty; producer lookup runs off `resop_refs`
        // (populated by `bind_input_resops`).
        self.explicit_input_ops_seed = Some(Vec::new());
        // Bridge retarget: hold the closing JUMP (skip_flush) so
        // try_jump_to_existing_trace can redirect it to an existing loop
        // target (unroll.py jump_to_existing_trace) instead of
        // jump_to_preamble's boxed entry. Fires when the bridge ends in a
        // JUMP, the trace carries a peeled short preamble, and more than one
        // front target token exists.
        //
        // Also require a synthesizable patchguardop: a bridge carries no
        // GUARD_FUTURE_CONDITION, so `self.patchguardop` is only populated
        // from one of the bridge's own body guards (synthesized below). A
        // bridge that closes into an existing loop with no body guard
        // (interpreter-entry / straight-line prologue) would retarget with
        // `patchguardop` left None and hit the unroll.py:333 invariant in
        // `unroll.rs`'s `jump_to_existing_trace_impl` the
        // moment virtual-state matching emits an extra guard. Keep those on
        // the jump_to_preamble path. The predicate matches the synthesis
        // filter below, so retarget runs only when patchguardop is gettable.
        let has_body_guard = ops
            .iter()
            .any(|op| op.opcode.is_guard() && op.rd_resume_position() >= 0);
        let retarget_close_jump = ops.last().is_some_and(|op| op.opcode == OpCode::Jump)
            && inline_short_preamble
            && front_target_tokens.len() > 1
            && has_body_guard;
        let skip_flush_saved = self.skip_flush;
        self.skip_flush = retarget_close_jump;
        // unroll.py `optimize_bridge` propagate_all_forward builds NO
        // short-preamble/export preview; step (3) runs only in the retrace
        // arm below. `building_bridge` still marks this call so a preview
        // VirtualStatesCantMatch (if the retrace arm publishes) is non-fatal
        // rather than an InvalidLoop that discards the bridge.
        let building_bridge_saved = self.building_bridge;
        self.building_bridge = true;
        let optimized_ops = self.optimize_with_constants_and_inputs_at(
            ops,
            constants,
            num_inputs,
            bridge_inputarg_base,
            start_next_pos,
            false,
        );
        self.building_bridge = building_bridge_saved;
        self.skip_flush = skip_flush_saved;
        let mut optimized_ops = optimized_ops?;

        // `reached_loop_header` emits the GUARD_FUTURE_CONDITION before both
        // of its closes, so upstream always has a `patchguardop`:
        // `_jump_to_existing_trace` dereferences
        // `self.optimizer.patchguardop` unconditionally for the extra guards
        // (unroll.py) and passes it to `inline_short_preamble`, which
        // dereferences it again per replayed guard (unroll.py:409).
        // Pyre emits it on the merge-point crossing that closes a bridge
        // (`jitcode_dispatch/mod.rs`) and on the own-loop close
        // (`close_loop_args_at`); a crossing that skipped the emit — a
        // previously declined close, or a walk that is not the authoritative
        // top-level executor — still reaches here without one. Synthesize it
        // from the bridge's own last body guard (highest resume position,
        // closest to the close); that guard's resume coordinate is mid-trace
        // where the merge point's is the loop header, so it is a fallback, not
        // an equivalent.
        //
        // Gated only on there being such a guard, NOT on `retarget_close_jump`.
        // Every path that reaches `try_jump_to_existing_trace` below can read
        // patchguardop, and `retarget_close_jump` is strictly narrower than
        // that: it also demands `ops.last()` be a JUMP, while the close is
        // decided from the post-optimization `terminal_op`. Leaving the gate
        // there let a bridge inline a short preamble with no patchguardop,
        // where the replayed guards keep the `rd_resume_position` cloned off
        // the PREAMBLE's guard and resolve their snapshot against this
        // trace's numbering. Setting it when nothing reads it is inert.
        if self.patchguardop.is_none()
            && let Some(g) = ops
                .iter()
                .filter(|o| o.opcode.is_guard() && o.rd_resume_position() >= 0)
                .max_by_key(|o| o.rd_resume_position())
        {
            self.patchguardop = Some((**g).clone());
        }

        // unroll.py optimize_bridge: `jump_op = info.jump_op`.
        // `propagate_all_forward(..., flush=False)` parks JUMP on
        // `info.jump_op`. Pyre's skip_flush=false path instead sends it
        // through the passes into `new_operations`; pull it back out so
        // `jump_to_preamble` can rewrite `descr=cell_token.target_tokens[0]`.
        let mut terminal_jump = self
            .terminal_op
            .take()
            .filter(|op| op.opcode == OpCode::Jump);
        if terminal_jump.is_none()
            && let Some(idx) = optimized_ops
                .iter()
                .rposition(|op| op.opcode == OpCode::Jump)
        {
            terminal_jump = Some((*optimized_ops.remove(idx)).clone());
        }

        if optimized_ops.len() < 120 && crate::smallir_enabled() {
            eprintln!(
                "@@@SMALLIR BRIDGE total={} has_jump={} front_targets={}",
                optimized_ops.len(),
                terminal_jump.is_some() as i32,
                front_target_tokens.len(),
            );
            for (i, op) in optimized_ops.iter().enumerate() {
                eprintln!(
                    "@@@SMALLIR   B[{i}] {:?} pos={:?}",
                    op.opcode,
                    op.pos().get()
                );
            }
        }

        // A FINISH-only trace belongs on SimpleCompileData.optimize_loop
        // (`compile.py compile_trace` `ends_with_jump=False`).
        let Some(terminal_jump) = terminal_jump else {
            return Err(crate::optimize::InvalidLoop(
                "optimize_bridge requires a JUMP; FINISH uses optimize_loop",
            ));
        };
        let jump_args: Vec<OpRef> = terminal_jump
            .args_slice()
            .iter()
            .map(|a| a.to_opref())
            .collect();

        // unroll.py: not inline_short_preamble → jump_to_preamble
        // RPython calls send_extra_operation(jump_op) which forces virtuals
        // through the full pass chain. No explicit flush()/force_box() needed.
        if !inline_short_preamble || front_target_tokens.len() <= 1 {
            // unroll.py `cell_token = jump_op.getdescr()`: the jump-to
            // jitcell is the one the recorded close JUMP points to, which is
            // also where `front_target_tokens` comes from (`compile_bridge`
            // resolves it off the JUMP target, not the bridge origin).
            // `assert cell_token.target_tokens`: require a target.
            if !front_target_tokens.is_empty() {
                let mut ctx = self.final_ctx.take().unwrap_or_else(|| {
                    // opencoder.py:259 inputarg_from_tp parity — seed inputarg
                    // operands with the producer-side types when available; the
                    // Ref fallback covers the rare case the recorder hasn't
                    // populated `trace_inputargs` (e.g. test-only entry
                    // paths) and matches the historical Type::Void behaviour
                    // for ptr-handle slots.
                    let ni = self.final_num_inputs();
                    let types: Vec<majit_ir::Type> = self
                        .trace_inputargs
                        .get(..ni)
                        .map(|s| {
                            s.iter()
                                .map(|o| o.ty().unwrap_or(majit_ir::Type::Ref))
                                .collect()
                        })
                        .unwrap_or_else(|| vec![majit_ir::Type::Ref; ni]);
                    OptContext::with_inputarg_types(32, &types)
                });
                let result = self.jump_to_preamble(
                    &terminal_jump,
                    front_target_tokens,
                    optimized_ops,
                    &mut ctx,
                );
                self.final_ctx = Some(ctx);
                return result;
            }
            return Ok((optimized_ops, false));
        }

        // unroll.py:203: self.flush()
        let mut ctx = self.final_ctx.take().unwrap_or_else(|| {
            // opencoder.py:259 inputarg_from_tp parity — same fallback shape
            // as the inline_short_preamble path above.
            let ni = self.final_num_inputs();
            let types: Vec<majit_ir::Type> = self
                .trace_inputargs
                .get(..ni)
                .map(|s| {
                    s.iter()
                        .map(|o| o.ty().unwrap_or(majit_ir::Type::Ref))
                        .collect()
                })
                .unwrap_or_else(|| vec![majit_ir::Type::Ref; ni]);
            OptContext::with_inputarg_types(32, &types)
        });

        // unroll.py:148-158 `_optimize_unrolled_loop` ordering:
        //
        //     self.optimizer.flush()
        //     self.optimizer.optimize_force_at_the_end_of_preamble()
        //     ...
        //     virtual_state = self._jump_to_existing_trace(jump_op, ...)
        //
        // The flush+force pair runs BEFORE `_jump_to_existing_trace`; the
        // VS is captured inside `_jump_to_existing_trace` (unroll.py
        // `get_virtual_state(jump_op.getarglist())`) on the now-forwarded
        // jump args. force_at_the_end_of_preamble (info.py) recurses
        // into virtual fields but does NOT force the top-level virtuals,
        // so the top-level VS shape is preserved. We therefore omit any
        // pre-flush snapshot and let try_jump_to_existing_trace compute
        // VS internally — matching RPython 1:1.
        ctx.skip_flush_mode = retarget_close_jump;
        self.flush(&mut ctx)?;

        // unroll.py:204-205: force_at_the_end_of_preamble for each jump arg
        let saved_pass_idx = ctx.current_pass_idx;
        ctx.current_pass_idx = ctx.optearlyforce_idx;
        for arg in terminal_jump.args_slice().iter() {
            let _ = self.force_box_for_end_of_preamble(arg, &mut ctx);
        }
        ctx.current_pass_idx = saved_pass_idx;

        // A failed match attempt's operations stay in `new_operations`.
        // unroll.py `jump_to_existing_trace` states the choice
        // outright — "leaving the bogus operations at the end of the trace is
        // not great, but should be safe: at worst, they just always do a bit
        // of stuff and then fail" — and its only stopgap is
        // `cant_replace_guards`, so the guards may not be hoisted earlier but
        // do stay. `optimize_bridge` never trims either: `jump_to_preamble`
        // appends the JUMP onto the buffer as it stands (unroll.py:210-211).
        //
        // Trimming would be unsound, because it drops the buffer but not the
        // optimizer state those operations installed: `propagate_postprocess`
        // (rewrite.rs) runs `make_constant_arg` for an emitted GuardTrue /
        // GuardFalse / GuardValue, and GuardClass / GuardNonNull install
        // PtrInfo. The forwarding outlives the deletion, so the optimizer goes
        // on believing a fact that no operation in the compiled bridge
        // establishes — for the JUMP re-sent below through
        // `send_extra_operation` -> force_box -> get_box_replacement, for the
        // `force_boxes=true` retry, and for the int bounds any later
        // comparison is folded against.
        //
        // Not hypothetical: `generate_guards` emits an `IntLe` + `GuardTrue`
        // pair for a length relation, and the GuardTrue's postprocess pins the
        // comparison result to 1. Trimming deleted both while keeping that
        // pin.

        // unroll.py: jump_to_existing_trace(force_boxes=False)
        // RPython iterates ALL target_tokens; preamble (virtual_state=None)
        // is skipped inside jump_to_existing_trace (unroll.py).
        let opt_unroll = crate::optimizeopt::unroll::OptUnroll::new();
        let vs = match Self::try_jump_to_existing_trace(
            &opt_unroll,
            &jump_args,
            front_target_tokens,
            self,
            &mut ctx,
            false,
            runtime_boxes,
            None,
        ) {
            Ok(vs) => vs,
            // unroll.py: except InvalidLoop → jump_to_preamble
            // RPython: self.jump_to_preamble → send_extra_operation
            Err(_) => {
                if !front_target_tokens.is_empty() {
                    let result = self.jump_to_preamble(
                        &terminal_jump,
                        front_target_tokens,
                        optimized_ops,
                        &mut ctx,
                    );
                    self.final_ctx = Some(ctx);
                    return result;
                }
                let mut result = optimized_ops;
                result.append(&mut ctx.new_operations);
                self.final_ctx = Some(ctx);
                return Ok((result, false));
            }
        };

        // unroll.py:212-213: vs is None → matched, JUMP redirected
        if vs.is_none() {
            let mut result = optimized_ops;
            result.append(&mut ctx.new_operations);
            self.final_ctx = Some(ctx);
            return Ok((result, false));
        }

        // unroll.py:214-218: retrace check
        if crate::debug::have_debug_prints() {
            crate::debug::log_one(
                "jit-tracing",
                &format!(
                    "bridge-retrace-check retraced_count={retraced_count} retrace_limit={retrace_limit} jump_args={} force_boxes=false",
                    jump_args.len(),
                ),
            );
        }
        if *retraced_count < retrace_limit {
            // unroll.py `cell_token.retraced_count += 1` before export.
            *retraced_count += 1;
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-tracing",
                    &format!("Retracing ({}/{retrace_limit})", retraced_count),
                );
            }
            // unroll.py `exported_state = self.optunroll.export_state(
            //     info.jump_op.getarglist(), info.inputargs, runtime_boxes)`
            // after the failed virtual-state match. Reusing a JUMP-time
            // `exported_loop_state` (or leaving it None) skips that late
            // export; `retrace_needed` then starts without a state.
            let renamed_inputargs: Vec<OpRef> = (0..num_inputs)
                .map(|i| {
                    let pos = ctx.inputarg_base + i as u32;
                    OpRef::input_arg_typed(pos, ctx.inputarg_type_at_strict(i))
                })
                .collect();
            let exported_int_bounds = self.collect_exported_int_bounds(&jump_args, &mut ctx);
            // unroll.py `export_state` (463-488) builds the short-preamble
            // preview internally; the eager path skipped it for this bridge.
            // `ctx.preamble_end_args` is the forced list from step (2).
            let preview_args = ctx
                .preamble_end_args
                .clone()
                .unwrap_or_else(|| jump_args.clone());
            // A mismatch is the empty-export outcome the eager bridge path
            // used to take (`break 'export None`); this late export then
            // recomputes from `ctx.preamble_end_args`.
            let _ = self.publish_preview_short_state(&mut ctx, &preview_args);
            let mut state = crate::optimizeopt::unroll::export_state(
                &jump_args,
                &renamed_inputargs,
                self,
                &mut ctx,
                Some(&exported_int_bounds),
            );
            state.runtime_boxes = runtime_boxes.to_vec();
            // unroll.py `exported_state.quasi_immutable_deps =
            // self.quasi_immutable_deps`
            crate::optimizeopt::unroll::merge_quasi_immutable_deps(
                &mut state.quasi_immutable_deps,
                &self.quasi_immutable_deps,
            );
            crate::optimizeopt::unroll::merge_quasi_immutable_deps(
                &mut state.quasi_immutable_deps,
                &ctx.quasi_immutable_deps,
            );
            state.patchguardop = self.patchguardop.clone();
            self.exported_loop_state = Some(state);
            // unroll.py:234-236 returns `self._newoperations` whole, so the
            // retrace is built on the flush + end-of-preamble force + failed
            // match guards above, not on the body alone. `retarget_close_jump`
            // set `self.skip_flush`, which suppressed the in-body flush, so
            // OptHeap's deferred setfield_gc stores are exactly what the
            // `self.flush` above emitted — dropping this buffer would erase a
            // heap store the bridge body performed.
            let mut result = optimized_ops;
            result.append(&mut ctx.new_operations);
            // unroll.py `_clean_optimization_info(self._newoperations)`
            Self::clean_optimization_info(&result);
            self.final_ctx = Some(ctx);
            return Ok((result, true));
        }

        // unroll.py:220-227: retrace limit reached, try force_boxes=True.
        // Matches `_optimize_unrolled_loop`'s second call to
        // `_jump_to_existing_trace(..., force_boxes=True)` (unroll.py);
        // VS is recomputed inside that call from the current (post-force)
        // jump_op.getarglist() — no pre-snapshot is reused.
        let vs2 = match Self::try_jump_to_existing_trace(
            &opt_unroll,
            &jump_args,
            front_target_tokens,
            self,
            &mut ctx,
            true,
            runtime_boxes,
            None,
        ) {
            Ok(vs) => vs,
            // unroll.py:224-225: except InvalidLoop: pass
            // vs (from first attempt) is still not None → falls through
            // to jump_to_preamble below.
            Err(_) => vs,
        };

        // unroll.py:226-227: vs is None → matched with forced boxes
        if vs2.is_none() {
            let mut result = optimized_ops;
            result.append(&mut ctx.new_operations);
            self.final_ctx = Some(ctx);
            return Ok((result, false));
        }

        // unroll.py: jump_to_preamble → send_extra_operation.
        // RPython sends the JUMP through the full optimization chain so that
        // force_box materializes virtuals and potential_extra_ops are consumed.
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!(
                "[jit] Retrace count reached, jumping to preamble (retraced_count={} retrace_limit={})",
                retraced_count, retrace_limit,
            );
        }
        let result = if !front_target_tokens.is_empty() {
            self.jump_to_preamble(&terminal_jump, front_target_tokens, optimized_ops, &mut ctx)
        } else {
            let mut result = optimized_ops;
            result.append(&mut ctx.new_operations);
            Ok((result, false))
        };
        self.final_ctx = Some(ctx);
        result
    }

    /// unroll.py `UnrollOptimizer.jump_to_preamble`:
    /// `jump_op.copy_and_change(rop.JUMP, descr=cell_token.target_tokens[0])`
    /// then `send_extra_operation`. The recorded JUMP descr is the
    /// `JitCellToken`; this rewrites it to the preamble TargetToken.
    fn jump_to_preamble(
        &mut self,
        terminal_jump: &Op,
        front_target_tokens: &[crate::history::TargetToken],
        mut optimized_ops: Vec<majit_ir::OpRc>,
        ctx: &mut OptContext,
    ) -> Result<(Vec<majit_ir::OpRc>, bool), crate::optimize::InvalidLoop> {
        // unroll.py: `assert cell_token.target_tokens[0].virtual_state is None`
        if front_target_tokens
            .first()
            .is_some_and(|token| token.virtual_state.is_some())
        {
            return Err(crate::optimize::InvalidLoop(
                "jump_to_preamble: target_tokens[0].virtual_state is not None",
            ));
        }
        let preamble = front_target_tokens
            .first()
            .map(|token| token.as_jump_target_descr());
        let jump_op = terminal_jump.copy_and_change(OpCode::Jump, None, Some(preamble));
        // A compiled preamble already stored `_x86_arglocs`.
        // `_compute_hint_locations_from_descr` asserts
        // `len(arglocs) == jump_op.numargs()`; a short JUMP would panic
        // in the assembler. Give up like `unroll.py` `InvalidLoop` so
        // the bridge is cancelled instead.
        if let Some(target) = jump_op
            .getdescr()
            .as_ref()
            .and_then(|d| d.as_loop_target_descr())
        {
            let n = target.target_arglocs().len();
            if n > 0 && jump_op.num_args() != n {
                return Err(crate::optimize::InvalidLoop(
                    "jump_to_preamble: JUMP arity != compiled preamble LABEL arity",
                ));
            }
        }
        self.send_extra_operation(&OpRc::new(jump_op.clone()), ctx)?;
        optimized_ops.append(&mut ctx.new_operations);
        Ok((optimized_ops, false))
    }

    /// Wrapper: call jump_to_existing_trace, catch only InvalidLoop panics.
    /// Returns Ok(vs) on normal return, Err(_) on InvalidLoop.
    /// Non-InvalidLoop panics are re-raised.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn try_jump_to_existing_trace(
        opt_unroll: &crate::optimizeopt::unroll::OptUnroll,
        jump_args: &[OpRef],
        front_target_tokens: &mut [crate::history::TargetToken],
        optimizer: &mut Self,
        ctx: &mut OptContext,
        force_boxes: bool,
        pre_opt_jump_args: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Result<Option<crate::optimizeopt::virtualstate::VirtualState>, ()> {
        let vs = opt_unroll.jump_to_existing_trace_with_vs(
            jump_args,
            None,
            front_target_tokens,
            optimizer,
            ctx,
            force_boxes,
            pre_opt_jump_args,
            pre_vs,
        );
        // unroll.py:209-210 / 224-225 `except InvalidLoop`: a short-preamble
        // replay that could not resolve its args records a deferred InvalidLoop
        // signal on `ctx`; surface it as `Err(())` so the caller falls back to
        // jump_to_preamble.
        if ctx.take_invalid_loop().is_some() {
            Err(())
        } else {
            Ok(vs)
        }
    }

    /// TODO: aggregate per-pass `export_arg_int_bounds`
    /// HashMaps for hand-off to the next peeling iteration.  RPython
    /// preserves `IntBound` across iterations through Box-stable
    /// `OptInfo.IntBound` forwarding; pyre's flat-OpRef rebuilds
    /// `OptContext` per round, so the import path
    /// (`unroll.rs:setinfo_from_preamble`) must consult this side
    /// table.  See sibling cite at `intbounds.rs::export_arg_int_bounds`.
    fn collect_exported_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &mut OptContext,
    ) -> crate::FxIndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound> {
        let mut exported = crate::FxIndexMap::default();
        for pass in &self.passes {
            // Each pass resolves through the same `ctx`, so a box for one
            // canonical position is memoized to a single `Rc` — entries across
            // passes dedup by `Rc::ptr_eq`.
            for (box_key, bound) in pass.export_arg_int_bounds(args, ctx).iter() {
                exported.insert(box_key.clone(), bound.clone());
            }
        }
        exported
    }

    /// optimizer.py `_clean_optimization_info(lst)` — drop `_forwarded`
    /// on each exported op so the next optimizer (the retrace body) does
    /// not inherit preamble forwarding.
    fn clean_optimization_info(ops: &[majit_ir::OpRc]) {
        use majit_ir::forwarding::ForwardingHost;
        for op in ops {
            if !matches!(op.get_forwarded(), majit_ir::forwarding::Forwarded::None) {
                op.clear_forwarded();
            }
        }
    }

    /// Send one operation through the pass chain.
    ///
    /// NOTE: Do NOT add `make_equal_to(original_pos, new_pos)` here.
    /// The Emit variant's position tracking is handled by each pass
    /// and OptContext. Adding automatic replacement mapping here
    /// causes spurious forwarding that breaks heap/guard tests.
    fn propagate_one(
        &mut self,
        op: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        self.propagate_from_pass(0, op, ctx)
    }

    fn drain_extra_operations_from(
        &mut self,
        _start_pass: usize,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        if ctx.extra_operations_after.is_empty() {
            return Ok(());
        }
        // RPython `send_extra_operation` walks the list in place. Take the
        // queued deque instead of copying each entry into a fresh one.
        let pending = std::mem::take(&mut ctx.extra_operations_after);
        // The level is parked on the context so `flush_queued_producer` can
        // pull one entry out of it; this function runs nested inside itself
        // (`propagate_from_pass_range` drains after every pass), and only the
        // innermost level may absorb what the current propagate queues.
        ctx.extra_pending.push(pending);
        let result = self.drain_innermost_pending(ctx);
        let unfinished = ctx
            .extra_pending
            .pop()
            .expect("the active extra-operation drain level must remain installed");
        if result.is_err() {
            // `InvalidLoop` unwinds the recursive Rust drain in place of
            // RPython's ordinary exception unwinding.  Keep both operations
            // emitted by the failing propagation and the untouched tail from
            // this level available to the caller's unwind.  Each outer level
            // appends its own untouched tail in turn, so no parked producer is
            // silently discarded and a reused context never observes a
            // half-drained scheduling state.
            ctx.extra_operations_after.extend(unfinished);
        } else {
            debug_assert!(unfinished.is_empty());
        }
        result
    }

    fn drain_innermost_pending(
        &mut self,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        let end_pass = self.extra_operation_end_pass();
        let level = ctx.extra_pending.len() - 1;
        while let Some((from_pass, op)) = ctx.extra_pending[level].pop_front() {
            self.propagate_from_pass_range(from_pass, end_pass, &op, ctx)?;
            while let Some((start, op)) = ctx.extra_operations_after.pop_front() {
                ctx.extra_pending[level].push_front((start, op));
            }
        }
        Ok(())
    }

    /// `optimizer.py:623-625` forces every argument before the operation is
    /// appended, and `info.py force_box` emits the allocation as it
    /// clears the virtual flag — so a forced box's definition is always already
    /// in `_newoperations`. Pyre's `force_box` only *queues* the allocation
    /// when it runs from a pass rather than from final emission, which leaves a
    /// window where the box reads as non-virtual (so `force_box` is a no-op on
    /// it) while its `NEW_WITH_VTABLE` is still parked. Emitting a store in that
    /// window puts the use ahead of the definition.
    ///
    /// Close the window by propagating that one parked definition here, ahead
    /// of the operation that reads it. Everything else keeps its queued order.
    fn flush_queued_producer(
        &mut self,
        opref: OpRef,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        if opref.is_none() || opref.is_constant() {
            return Ok(());
        }
        let queued = ctx
            .extra_operations_after
            .iter()
            .position(|(_, op)| op.pos().get() == opref)
            .and_then(|at| ctx.extra_operations_after.remove(at))
            .or_else(|| {
                ctx.extra_pending.iter_mut().rev().find_map(|level| {
                    level
                        .iter()
                        .position(|(_, op)| op.pos().get() == opref)
                        .and_then(|at| level.remove(at))
                })
            });
        let Some((from_pass, op)) = queued else {
            return Ok(());
        };
        let end_pass = self.extra_operation_end_pass();
        self.propagate_from_pass_range(from_pass, end_pass, &op, ctx)
    }

    fn propagate_from_pass(
        &mut self,
        start_pass: usize,
        op: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        self.propagate_from_pass_range(start_pass, self.passes.len(), op, ctx)
    }

    fn extra_operation_end_pass(&self) -> usize {
        let mut end = self.passes.len();
        while end > 0 && self.passes[end - 1].name() == "unroll" {
            end -= 1;
        }
        end
    }

    fn propagate_from_pass_range(
        &mut self,
        start_pass: usize,
        end_pass: usize,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        // Box.type lives intrinsically on `OpRef.ty()` (variant
        // tag, history.py:220 + resoperation.py:1693 parity) and on
        // `Op.type_` once the op lands in `new_operations`, so an external
        // type side-table refresh is redundant.

        // Resolve forwarded arguments. PyPy `_emit_operation`
        // (optimizer.py) walks args via force_box at the entry to
        // emission; pyre's pre-pass walk via `ctx.get_box_replacement` is
        // the structural analog (force_box is invoked separately at
        // `Optimizer::emit_operation` for the post-pass refresh).
        // fail_args are NOT walked here — PyPy snapshots them once at
        // `store_final_boxes_in_guard` time via the numbering layer (which
        // encodes Const entries as TAGCONST in rd_numb, leaving fail_args
        // as TAGBOX-only references). Re-walking fail_args at pass entry
        // would be a pyre-only layer that could substitute Const refs
        // post-`make_equal_to(_, const_target)` and de-sync from the
        // numbering snapshot.
        // optimizer.py force_box loop parity: resolve each arg
        // through its forwarding chain. Store the CANONICAL terminal box
        // (carrying the live _forwarded chain), not a fresh from_opref box,
        // so passes can read PtrInfo / IntBound / known-class directly off
        // op.arg(i) — matching RPython where get_box_replacement returns the
        // Box object itself. A from_opref box is unbound and drops the chain.
        //
        // `optimizer.py _emit_operation` writes the resolution back
        // with `op.setarg(i, arg)` — into the operation's own slot, because
        // upstream's operation IS the box the passes forward. Written back
        // onto `op_rc` here for the same reason. Two things make that sound:
        //
        //  - No pass reads `op_rc`'s arguments. A pass reaches `op_rc` only as
        //    `Operand::from_bound_op(op_rc)`, the `make_equal_to` forwarding
        //    host, so what the second parameter carries is identity, which
        //    resolving in place does not change; the argument view every pass
        //    reads is `current_op` below.
        //  - The caller has handed its argument slots over. The unroll and
        //    bridge entries walk `TraceIterator`, whose `next` mints a fresh
        //    operation per step, so those ops are the pass's own outright;
        //    `optimize_with_constants_and_inputs_oprc` threads the recorder's
        //    own ops and documents the transfer on its own signature.
        for i in 0..op_rc.num_args() {
            // `get_box_replacement` is the consumer walk (`optimizer.py`
            // `get_box_replacement(op.getarg(i))`). A slot that already holds
            // its chain terminal keeps it: `setarg` of the same box changes
            // nothing. The old resolver stays in the debug witness only:
            // compute the new terminal first so the witness's
            // `materialize_operand_at` cannot heal before the comparison.
            let replaced = {
                let arg = &op_rc.args_slice()[i];
                arg.with_box_replacement(|terminal| {
                    (!terminal.same_box(arg)).then(|| terminal.clone())
                })
            };
            #[cfg(debug_assertions)]
            {
                let arg = op_rc.arg(i);
                let resolved = replaced.clone().unwrap_or_else(|| arg.clone());
                let old = match ctx.resolve_operand_operand_opt(&arg) {
                    Some(b) => b,
                    None => {
                        let argref = arg.to_opref();
                        if argref.is_none() {
                            arg.clone()
                        } else {
                            ctx.materialize_operand_at(argref)
                                .get_box_replacement(false)
                        }
                    }
                };
                debug_assert!(
                    resolved.same_box(&old),
                    "propagate_from_pass_range {:?} arg {i}: new {:?} old {:?}",
                    op_rc.opcode,
                    resolved.to_opref(),
                    old.to_opref()
                );
            }
            if let Some(resolved) = replaced {
                op_rc.setarg(i, resolved);
            }
        }

        // Borrowed until a pass replaces the operation; `Replace` and
        // `Restart` mint their own, and only those need an owned copy.
        let mut current_op: std::borrow::Cow<'_, Op> = std::borrow::Cow::Borrowed(op_rc);

        // optimizer.py: optimize_SAME_AS_I/R/F → make_equal_to(op, arg0)
        // SameAs ops are absorbed into forwarding, never emitted.
        if matches!(
            current_op.opcode,
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
        ) {
            let new = current_op.arg(0).to_opref();
            let b_old = Operand::from_bound_op(op_rc);
            let b_new = ctx.get_box_replacement_operand(new);
            ctx.make_equal_to(&b_old, &b_new);
            return Ok(());
        }

        // optimizer.py:570-589 parity: collect pass indices that need
        // postprocess callbacks. After emission, invoke them in reverse
        // order — matching RPython's OptimizationResult.callback() chain.
        let mut postprocess_passes: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();

        // Track whether any pass replaced the op. Only the untouched
        // pass-through reaches final emission structurally identical to the
        // recorder input op, so only then may emit reuse that input op as the
        // producer (one box per value); a Replace mints a distinct op that is
        // not the operand-wrapped input op.
        let mut replaced = false;

        for pass_idx in start_pass..end_pass {
            ctx.current_pass_idx = pass_idx;
            let result = {
                let pass = &mut self.passes[pass_idx];
                pass.propagate_forward(&current_op, op_rc, ctx)
            };
            // A leaf site deep in the pass may have recorded a deferred
            // `InvalidLoop` (e.g. `get_const_info_mut_box`) while still
            // returning a benign result. Abort before acting on that result.
            if let Some(e) = ctx.take_invalid_loop() {
                return Err(e);
            }
            self.drain_extra_operations_from(pass_idx + 1, ctx)?;
            match result {
                OptimizationResult::Emit(op) => {
                    // optimizer.py:576-581: collect postprocess for this pass
                    // if it has postprocess for this opnum.
                    if self.passes[pass_idx].have_postprocess_op(op.opcode) {
                        postprocess_passes.push(pass_idx);
                    }
                    // heap.py optimize_SETFIELD_GC ends in emit(op) of the
                    // same ResOperation. Reuse that Rc instead of cls().
                    let emit_rc = if op.opcode == op_rc.opcode {
                        OptContext::stamp_emitted_op(&op, op_rc);
                        op_rc.clone()
                    } else {
                        let created = OpRc::new(op.clone());
                        // Opcode change mints a second object at this position.
                        // `replace_op_with` links the walked-in op before emit.
                        OptContext::link_replaced_producer(op_rc, &created);
                        created
                    };
                    self.emit_operation_inner(emit_rc.clone(), ctx, false)?;
                    // optimizer.py:585-589: invoke postprocess callbacks
                    // in reverse order after emission.
                    for &pp_idx in postprocess_passes.iter().rev() {
                        self.passes[pp_idx].propagate_postprocess(&op, &emit_rc, ctx);
                    }
                    self.drain_pending_finish_guard_postprocess(ctx);
                    return Ok(());
                }
                OptimizationResult::Replace(op) => {
                    debug_assert!(
                        !(current_op.opcode.is_guard()
                            && op.opcode.is_guard()
                            && current_op.rd_resume_position() >= 0
                            && op.rd_resume_position() < 0),
                        "Replace dropped rd_resume_position: {:?} -> {:?}",
                        current_op.opcode,
                        op.opcode,
                    );
                    // optimizer.py: pass emitted (replace ≈ emit
                    // with modified op). Collect if has postprocess.
                    if self.passes[pass_idx].have_postprocess_op(op.opcode) {
                        postprocess_passes.push(pass_idx);
                    }
                    // 5: replace's new op carries `Op.type_` from
                    // construction (1) and a typed `op.pos` so
                    // downstream `op_at` lookups resolve it directly
                    // without a side-table refresh.
                    current_op = std::borrow::Cow::Owned(op);
                    replaced = true;
                }
                OptimizationResult::Restart(op) => {
                    // optimizer.py `send_extra_operation(newop, opt=None)`:
                    // re-dispatch from `first_optimization`, dropping the
                    // original op (the rule that returned Restart already
                    // skipped its own emit). autogenintrules.py uses this
                    // for every rewrite rule so that chained OptIntBounds
                    // rules (add_zero, int_is_zero, reassoc) re-fire on the
                    // rewritten op.
                    debug_assert!(
                        !(current_op.opcode.is_guard()
                            && op.opcode.is_guard()
                            && current_op.rd_resume_position() >= 0
                            && op.rd_resume_position() < 0),
                        "Restart dropped rd_resume_position: {:?} -> {:?}",
                        current_op.opcode,
                        op.opcode,
                    );
                    // 5: Restart's new op carries `Op.type_` from
                    // construction; no side-table refresh needed.
                    let restart_op_rc = OpRc::new(op);
                    // replace_op_with parity: the rewrite supersedes the
                    // original as the producer at its position so the
                    // re-dispatch reads/writes one canonical `_forwarded` host
                    // and the bound it accumulates survives emit's catch-up.
                    ctx.supersede_restart_producer(&restart_op_rc);
                    self.propagate_from_pass_range(0, end_pass, &restart_op_rc, ctx)?;
                    // Run any postprocess callbacks accumulated in the outer
                    // chain (passes that already returned PassOn for the
                    // original op before the Restart-returning pass fired).
                    // RPython's send_extra_operation does not unwind these —
                    // they belong to the original chain.
                    for &pp_idx in postprocess_passes.iter().rev() {
                        self.passes[pp_idx].propagate_postprocess(&current_op, op_rc, ctx);
                    }
                    self.drain_pending_finish_guard_postprocess(ctx);
                    return Ok(());
                }
                OptimizationResult::Remove => {
                    ctx.pending_mark_last_guard = None;
                    ctx.pending_guard_class_postprocess = None;
                    // optimizer.py `last_emitted_operation = REMOVED`
                    // — broadcast the removal so subsequent passes
                    // (e.g. `optimize_GUARD_NO_EXCEPTION`,
                    // rewrite.py:712-718) can observe the drop.
                    ctx.last_op_removed = true;
                    // optimizer.py:600-616: a removal ends the WALK
                    // (`op = None; break`) but not the callbacks — the
                    // `opt_results` collected before it still run, in reverse.
                    // Same loop as the Restart arm above.
                    for &pp_idx in postprocess_passes.iter().rev() {
                        self.passes[pp_idx].propagate_postprocess(&current_op, op_rc, ctx);
                    }
                    self.drain_pending_finish_guard_postprocess(ctx);
                    return Ok(());
                }
                OptimizationResult::PassOn => {
                    // optimizer.py: PASS_OP_ON path.
                    // RPython's emit() returns PASS_OP_ON if no postprocess,
                    // OptimizationResult if postprocess needed.
                    // Collect if this pass has postprocess for this opcode.
                    if self.passes[pass_idx].have_postprocess_op(current_op.opcode) {
                        postprocess_passes.push(pass_idx);
                    }
                }
                // rewrite.py:406 — guard proven to always fail
                OptimizationResult::InvalidLoop(msg) => {
                    return Err(crate::optimize::InvalidLoop(msg));
                }
            }
        }

        // If no pass handled it, emit as-is. An unreplaced pass-through is the
        // recorder input op verbatim (args re-resolved), so emit may append
        // that same Rc — no second ResOperation().
        let emitted_rc = if !replaced {
            self.emit_operation_inner(op_rc.clone(), ctx, true)?;
            op_rc.clone()
        } else {
            let created = OpRc::new((*current_op).clone());
            OptContext::link_replaced_producer(op_rc, &created);
            self.emit_operation_inner(created.clone(), ctx, false)?;
            created
        };
        // Postprocess in reverse order after emission.
        for &pp_idx in postprocess_passes.iter().rev() {
            self.passes[pp_idx].propagate_postprocess(&current_op, &emitted_rc, ctx);
        }
        // The FINISH reaches its emit down this path — `optimize_FINISH`
        // returns PassOn — so this is the drain that postprocess_FINISH
        // actually goes through.
        self.drain_pending_finish_guard_postprocess(ctx);
        Ok(())
    }

    /// optimizer.py: _emit_operation — emit with guard tracking.
    ///
    /// When emitting a guard, check replaces_guard to see if this guard
    /// should replace a previously emitted one (guard strengthening).
    /// Also track last_guard_op for consecutive guard descriptor sharing.
    /// RPython optimizer.py: _emit_operation calls force_box(arg)
    /// on every arg before final emission. In majit, this forces any remaining
    /// virtual args that weren't caught by pass-level handlers.
    #[cfg(test)]
    fn emit_operation(
        &mut self,
        op: Op,
        replaced: &majit_ir::OpRc,
        ctx: &mut OptContext,
        reuse: bool,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        let created = OpRc::new(op);
        // `Replace` mints a second object. Link the walked-in producer to it
        // before emission (`replace_op_with`).
        OptContext::link_replaced_producer(replaced, &created);
        self.emit_operation_inner(created, ctx, reuse)
    }

    fn emit_operation_inner(
        &mut self,
        mut op: majit_ir::OpRc,
        ctx: &mut OptContext,
        reuse: bool,
    ) -> Result<(), crate::optimize::InvalidLoop> {
        // RPython optimizer.py: _emit_operation is on the Optimizer (last
        // "pass" in the chain). Any force_box called here should emit directly,
        // matching RPython's Optimizer.emit_extra which just calls self.emit(op).
        let saved_in_final_emission = ctx.in_final_emission;
        ctx.in_final_emission = true;

        // RPython optimizer.py: emitting_operation callback — notify all passes
        // before any op is emitted. This is how OptHeap forces lazy sets before
        // guards even when the guard is emitted by an earlier pass.
        for (idx, pass) in self.passes.iter_mut().enumerate() {
            pass.emitting_operation(&op, ctx, idx);
        }
        // RPython emit_extra(op, emit=False) parity: drain operations
        // queued by emitting_operation (e.g., heap's force_lazy_set)
        // BEFORE the current op is emitted, preserving correct ordering.
        {
            let end_pass = self.extra_operation_end_pass();
            while let Some((start, queued_op)) = ctx.extra_operations_after.pop_front() {
                self.propagate_from_pass_range(start, end_pass, &queued_op, ctx)?;
            }
        }
        // A pass `emitting_operation` / queued drain may have deferred an
        // `InvalidLoop`; abort before emitting on inconsistent state. On the
        // abort path the context is discarded, so `in_final_emission` need not
        // be restored.
        if let Some(e) = ctx.take_invalid_loop() {
            return Err(e);
        }

        // optimizer.py `_emit_operation`: `arg = self.force_box(op.getarg(i));
        // op.setarg(i, arg)`. The forced box is the chain terminal carrying its
        // `_forwarded` info, so emitted ops and get_producing_op consumers can
        // read info off op.arg(i).
        for i in 0..op.num_args() {
            let arg = self.force_box(&op.arg(i), ctx);
            self.flush_queued_producer(arg.to_opref(), ctx)?;
            op.setarg(i, arg);
        }
        // force_box may force a virtual whose materialization defers an
        // `InvalidLoop`; abort before the emit / `expect` sites below.
        if let Some(e) = ctx.take_invalid_loop() {
            return Err(e);
        }

        // optimizer.py:626: self.metainterp_sd.profiler.count(Counters.OPT_OPS).
        // Pyre defers the fold into JitStatsCounters via update_counters
        // because `Optimizer` holds no `metainterp_sd.profiler` reference
        // (see the `opt_ops_emitted` field doc).
        self.opt_ops_emitted = self.opt_ops_emitted.saturating_add(1);
        if op.opcode.is_guard() {
            // optimizer.py:629: profiler.count(Counters.OPT_GUARDS).
            self.opt_guards_emitted = self.opt_guards_emitted.saturating_add(1);
            // optimizer.py:630-631: pendingfields = self.pendingfields; self.pendingfields = None
            // — captured into ctx.pending_for_guard by the heap pass via emitting_operation.
            // emit_guard_operation reads ctx.pending_for_guard inside store_final_boxes_in_guard
            // and clears it.

            // optimizer.py: replaces_guard check BEFORE emit_guard_operation.
            // optimizer.py `orig_op in self.replaces_guard` keys by the raw
            // `orig_op` identity (before get_box_replacement), so resolve to the
            // producer box without following `_forwarded`.
            // optimizer.py `if self.replaces_guard and orig_op in
            // self.replaces_guard`.
            if self.can_replace_guards
                && !self.replaces_guard.is_empty()
                && let Some(replacement) = ctx
                    .resolve_to_operand(op.pos().get())
                    .and_then(|op_key| self.replaces_guard.swap_remove(&op_key))
            {
                let target_pos = replacement.pos().get().raw() as usize;
                if target_pos < ctx.new_operations.len() {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[opt] guard replacement op={:?} pos={:?} target_index={} len={}",
                            op.opcode,
                            op.pos().get(),
                            target_pos,
                            ctx.new_operations.len()
                        );
                    }
                    // optimizer.py replace_guard_op:
                    //   old_descr = old_op.getdescr()
                    //   new_descr = new_op.getdescr()
                    //   new_descr.copy_all_attributes_from(old_descr)
                    // Inherit the slot's resume payload onto the new op's
                    // descr so the new guard carries old's rd_* — without
                    // this, `_newoperations[target_pos]` would land with
                    // the new descr's empty rd_* defaults.  In-place
                    // mutation preserves new_descr's identity (fail_index
                    // / status / subtype tag).
                    //
                    // RPython performs both getdescr() calls and the
                    // copy_all_attributes_from unconditionally; an old/new
                    // descr missing here means the optimizer gave us a
                    // replacement guard without resume payload, which
                    // would silently overwrite the slot.  Match RPython
                    // by panicking instead of skipping.
                    let old_descr = ctx.new_operations[target_pos].getdescr().expect(
                        "optimizer.py:716 old_descr = old_op.getdescr(): \
                                 replaced guard slot has no descr",
                    );
                    let new_descr = op.getdescr().expect(
                        "optimizer.py:717 new_descr = new_op.getdescr(): \
                             replacement guard has no descr",
                    );
                    crate::compile::copy_all_attributes_from(&new_descr, &old_descr);
                    ctx.replace_new_operation(target_pos, op.clone());
                    ctx.in_final_emission = saved_in_final_emission;
                    return Ok(());
                }
            }

            // optimizer.py: op = self.emit_guard_operation(op, pendingfields)
            if let Some(newop) = self.emit_guard_operation(&op, ctx) {
                let created = OpRc::new(newop);
                // Guard resume sharing copies into a second object.
                OptContext::link_replaced_producer(&op, &created);
                op = created;
            }
            // emit_guard_operation may defer an `InvalidLoop` (e.g. a pending
            // SETARRAYITEM index that is not a non-negative constant).
            if let Some(e) = ctx.take_invalid_loop() {
                return Err(e);
            }
        } else {
            // optimizer.py:639-644: preserve last_guard_op for guard chaining
            // unless the op has side effects or is a call_pure that can raise.
            let preserves_chain = (op.opcode.has_no_side_effect()
                || op.opcode.is_guard()
                || op.opcode.is_jit_debug()
                || op.opcode.is_ovf())
                && !Self::is_call_pure_pure_canraise(&op);
            if !preserves_chain {
                self.last_guard_op_idx = None;
            }
        }
        // optimizer.py:598-602:
        //     if rop.returns_bool_result(op.opnum):
        //         self.getintbound(op).make_bool()
        // The `make_bool` IntBound write runs post-emit (below), once the
        // op's box is bound; only the type assertion is checked here.
        if op.opcode.returns_bool() {
            assert_eq!(
                op.result_type(),
                majit_ir::Type::Int,
                "returns_bool op must have int result: {:?} pos={:?} args={:?}",
                op.opcode,
                op.pos().get(),
                op.getarglist()
            );
        }
        // `optimizer.py _emit_operation`'s `_newoperations.append(op)` appends the operation
        // itself. Captured up front so the emit can move `op` in rather than
        // hand it a copy; everything the post-emit block reads off `op` is
        // this opcode and the result type it implies (asserted equal by
        // `debug_assert_box_type_invariant`).
        let op_opcode = op.opcode;
        let op_result_type = op.result_type();
        let _ = reuse;
        let emitted_box = majit_ir::operand::Operand::from_bound_op(&op);
        let emitted = ctx.emit_rc(op);
        // optimizer.py `self._emittedoperations[op] = None` — `ctx.emit_rc`
        // records the appended op so `as_operation` can later confirm it
        // is in the emit set.
        let emitted_box = emitted_box.get_box_replacement(false);
        debug_assert!(
            ctx.get_box_replacement_operand(emitted)
                .same_box(&emitted_box),
            "emitted op at {emitted:?} is not its position's producer"
        );
        // optimizer.py `_emit_operation` clears the REMOVED
        // sentinel on each successful emit. Cross-pass readers
        // (rewrite.py `optimize_GUARD_NO_EXCEPTION`) see the
        // flag transition from true (prior Remove) → false (this
        // emit).
        ctx.last_op_removed = false;
        // optimizer.py: returns_bool_result → getintbound(op).make_bool().
        // Run here (post-emit) so the now-bound op box carries the IntBound
        // write; returns_bool ops are Int-typed (asserted above).
        if op_opcode.returns_bool() {
            ctx.with_intbound_mut(&emitted_box, |bound| bound.make_bool());
        }
        // optimizer.py: after emit, promote IntBound→Const.
        //   op = self.get_box_replacement(op)
        //   if op.type == 'i':
        //       opinfo = op.get_forwarded()  # IntBound
        //       if opinfo is not None and opinfo.is_constant():
        //           op.set_forwarded(ConstInt(opinfo.get_constant_int()))
        if op_result_type == majit_ir::Type::Int
            && let Some(bound) = ctx.peek_intbound_box(&emitted_box)
            && bound.is_constant()
        {
            let const_val = bound.get_constant_int();
            ctx.make_constant_box(&emitted_box, majit_ir::Value::Int(const_val));
        }
        if crate::majit_log_enabled()
            && matches!(
                op_opcode,
                OpCode::CallMayForceI
                    | OpCode::CallMayForceR
                    | OpCode::CallMayForceF
                    | OpCode::CallMayForceN
                    | OpCode::GuardNotForced
                    | OpCode::GuardNotForced2
            )
        {
            eprintln!(
                "[opt] emit {:?} pos={:?} len={}",
                op_opcode,
                emitted,
                ctx.new_operations.len()
            );
        }
        // optimizer.py: run deferred postprocess after emit.
        // RPython calls OptimizationResult.callback() → propagate_postprocess.
        // rewrite.py: postprocess_GUARD_NONNULL → mark_last_guard
        if let Some(opref) = ctx.pending_mark_last_guard.take()
            && let Some(b) = ctx.get_box_replacement_operand_opt(opref)
        {
            ctx.mark_last_guard(&b);
        }
        if let Some(pp) = ctx.pending_guard_class_postprocess.take() {
            // rewrite.py postprocess_GUARD_CLASS:
            //   update_last_guard = not old_guard or isinstance(descr, ResumeAtPositionDescr)
            //   make_constant_class(arg0, expectedclassbox, update_last_guard)
            //
            // Delegate to `Optimizer::make_constant_class` rather than
            // inlining a fresh `PtrInfo::known_class` overwrite — the
            // helper preserves existing `Instance` / `Virtual` PtrInfo
            // (fields, descr, cached_vinfo, virtual state) and only
            // refreshes `_known_class`, matching upstream
            // `optimizer.py` where `isinstance(opinfo,
            // InstancePtrInfo)` mutates in place.
            // optimizer.py `make_constant_class` always updates
            // `_forwarded` — `materialize_operand_at` materializes the Box so the
            // write is never silently skipped. Same materializer feeds
            // the `last_guard_pos` read; `info.py
            // get_last_guard_pos` reads the PtrInfo field (None if no
            // PtrInfo, mapped to -1 = "no guard recorded").
            let pp_obj_box = ctx.get_box_replacement_operand_opt(pp.obj);
            let old_guard_pos = pp_obj_box
                .as_ref()
                .and_then(|b| ctx.last_guard_pos(b))
                .map(|p| p as i32)
                .unwrap_or(-1);
            let update_last_guard =
                old_guard_pos < 0 || ctx.is_resume_at_position_guard(old_guard_pos);
            if let Some(b) = &pp_obj_box {
                Self::make_constant_class(ctx, b, pp.class_val, update_last_guard);
            }
        }
        ctx.in_final_emission = saved_in_final_emission;
        // A post-emit `make_constant_box` / postprocess may have deferred an
        // `InvalidLoop`; surface it so the driver abandons the trace.
        if let Some(e) = ctx.take_invalid_loop() {
            return Err(e);
        }
        Ok(())
    }

    /// optimizer.py emit_guard_operation
    ///
    /// Manages the guard sharing chain (`_last_guard_op`) and dispatches to
    /// `_copy_resume_data_from` (descrless follow-up guard, e.g.
    /// `GUARD_NO_EXCEPTION` after a `CALL_MAY_FORCE`) or to
    /// `store_final_boxes_in_guard` for fresh guards.
    fn emit_guard_operation(&mut self, op: &Op, ctx: &mut OptContext) -> Option<Op> {
        let opcode = op.opcode;
        // optimizer.py `_emit_operation`.  A side-effecting op emitted straight into
        // `new_operations` (a forced virtual's NEW/SETFIELD_GC, a flushed lazy
        // set) never reaches this function's `emit_operation` caller, so the
        // reset it owes the chain is recorded on the context.  RPython gets
        // this for free: `emit_extra` -> `emit` -> `_emit_operation` runs the
        // reset whoever emitted the op.
        if ctx.take_guard_chain_broken() {
            self.last_guard_op_idx = None;
        }
        // optimizer.py:661-664: GUARD_NO_EXCEPTION / GUARD_EXCEPTION can only
        // share resume data with a preceding GUARD_NOT_FORCED. Anything else
        // breaks the chain.  GUARD_NOT_FORCED_2 is intentionally excluded:
        // upstream emits it only at finish() (pyjitpl.py:3236) where no
        // GUARD_(NO_)EXCEPTION can follow, so reuse would create an
        // unreachable sharing edge.
        if (opcode == OpCode::GuardNoException || opcode == OpCode::GuardException)
            && self
                .last_guard_op_idx
                .and_then(|idx| ctx.new_operations.get(idx))
                .is_some_and(|last| last.opcode != OpCode::GuardNotForced)
        {
            self.last_guard_op_idx = None;
        }
        // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share resume data.
        if opcode == OpCode::GuardAlwaysFails {
            self.last_guard_op_idx = None;
        }

        // optimizer.py `emit_guard_operation`: `if (self._last_guard_op and
        // guard_op.getdescr() is None):` — a descrless guard reuses the
        // previous guard's resume data instead of numbering its own.  The
        // shared guard then resumes at the donor's position and the
        // intervening operations re-execute; `_emit_operation` keeps the
        // chain alive only across ops that are side-effect free, guards,
        // jit_debug or ovf, which is what makes that re-execution sound.
        // A recorded guard carries its own `rd_resume_position` and still
        // shares upstream — that position is simply left unused, so it must
        // not gate the test.
        // compile.py invent_fail_descr_for_op: GUARD_NOT_FORCED /
        // GUARD_NOT_FORCED_2 must always mint a fresh ResumeGuardForcedDescr
        // (`assert copied_from_descr is None`).  They are never on the
        // sharing chain.  Mirrors the OptContext path in
        // `optimizeopt/mod.rs`'s `emit_guard_operation`.
        // `op.rd_resume_position() < 0` is a pyre precondition with no
        // upstream counterpart, and it is load-bearing.  Upstream shares
        // whenever `self._last_guard_op and guard_op.getdescr() is None`,
        // because a sharing guard resumes at the donor's position and the
        // ops in between merely re-execute.  Pyre resumes through a
        // *recorded snapshot* named by `rd_resume_position`, and rewinding a
        // guard that owns one replays the recorded bytecode range, not just
        // the residual ops: side effects already committed in that range run
        // a second time.  Measured on `bench/synth/short_circuit_side_effects`
        // (0903): dropping this conjunct over-counts a helper by 2 and moves
        // the total, and takes 131 check.py fixtures that are green on main
        // red — 12 wrong answers and 7 crashes.  Only a guard the optimizer
        // invented (no snapshot of its own, e.g. GUARD_NO_EXCEPTION after a
        // call — upstream's own motivating case) can safely inherit one.
        let shared = !op.has_descr()
            && op.rd_resume_position() < 0
            && self.last_guard_op_idx.is_some()
            && opcode != OpCode::GuardNotForced
            && opcode != OpCode::GuardNotForced2;
        if shared {
            // optimizer.py:673-674:
            //   self.metainterp_sd.profiler.count_ops(
            //       opnum, jitprof.Counters.OPT_GUARDS_SHARED)
            self.opt_guards_shared_emitted = self.opt_guards_shared_emitted.saturating_add(1);
            let replaced = self._copy_resume_data_from(op, ctx);
            if opcode == OpCode::GuardException {
                self.last_guard_op_idx = None;
            }
            return replaced;
        } else {
            // optimizer.py:630-631 + resume.py:428-445 + 520-558:
            // RPython passes `pendingfields` INTO finish() so
            // `_add_pending_fields` tags target_tagged / value_tagged
            // during the same numbering pass that builds rd_numb.
            // Pyre builds the entries here with UNASSIGNED placeholders
            // and hands them to `store_final_boxes_in_guard` as the
            // `pending_setfields` parameter — finish() tags them and
            // the descr's `set_rd_pendingfields` receives the tagged
            // slice (compile.py:855 `_attrs_`).
            let pending = std::mem::take(&mut ctx.pending_for_guard);
            // resume.py:428-445 + 520-558: pending fields are passed
            // INTO finish() so `_add_pending_fields` can tag
            // target_tagged / value_tagged during the same numbering
            // pass that builds rd_numb.  We build the UNASSIGNED
            // entries here, hand them to `store_final_boxes_in_guard`,
            // and the descr's `set_rd_pendingfields` receives them
            // post-tagging — descr is the single SOT (compile.py:855
            // `_attrs_`).
            let pending_for_finish: Vec<majit_ir::GuardPendingFieldEntry> = if !pending.is_empty() {
                pending
                    .into_iter()
                    .map(|pf_op| {
                        let (target, value, item_index) = if pf_op.opcode == OpCode::SetarrayitemGc
                        {
                            // resume.py _add_pending_fields:
                            //   boxindex = op.getarg(1).get_box_replacement()
                            //   itemindex = boxindex.getint()       # → Const required
                            //   if itemindex < 0: raise TagOverflow
                            //   if itemindex > 2147483647: raise TagOverflow
                            // RPython's `getint()` extracts the
                            // tagged Const integer; non-Const args
                            // never reach `_add_pending_fields`
                            // because the heap pass only stages
                            // setarrayitem ops with a constant
                            // index. Fall back to InvalidLoop
                            // (caught in `pyjitpl.rs`'s
                            // `compile_loop_body`) on
                            // either invariant violation rather
                            // than silently coercing to 0.
                            let boxindex = ctx.resolve_operand_operand(&pf_op.arg(1));
                            let idx = match boxindex.const_int() {
                                Some(v) if (0..=i32::MAX as i64).contains(&v) => v,
                                // Defer the abort; the caller checks the signal
                                // after `emit_guard_operation` returns and
                                // abandons the trace before the bogus entry is
                                // used.
                                _ => {
                                    ctx.signal_invalid_loop(
                                        "_add_pending_fields: SETARRAYITEM_GC index \
                                             must be a non-negative Const i32 (TagOverflow)",
                                    );
                                    0
                                }
                            };
                            (pf_op.arg(0), pf_op.arg(2), idx as i32)
                        } else {
                            (pf_op.arg(0), pf_op.arg(1), -1i32)
                        };
                        majit_ir::GuardPendingFieldEntry {
                            descr: pf_op.getdescr(),
                            item_index,
                            target: ctx.resolve_operand_operand(&target).to_opref(),
                            value: ctx.resolve_operand_operand(&value).to_opref(),
                            target_tagged: majit_ir::resumedata::UNASSIGNED,
                            value_tagged: majit_ir::resumedata::UNASSIGNED,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // resume.py _add_optimizer_sections + bridgeopt.py serialize_optimizer_knowledge:
            // RPython collects optimizer knowledge INSIDE
            // store_final_boxes_in_guard → finish() → serialize_optimizer_knowledge.
            // Rust adaptation: collect BEFORE the call (borrow checker) and pass
            // as parameter. available_boxes filtering happens inside
            // memo.finish() using liveboxes ∩ liveboxes_from_env.
            let failarg_refs: Vec<OpRef> = op
                .guard_fail_args()
                .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_default();
            let knowledge_for_resume =
                self.collect_optimizer_knowledge_for_resume(ctx, &failarg_refs);
            let knowledge = if knowledge_for_resume.is_empty() {
                None
            } else {
                Some(knowledge_for_resume)
            };
            // optimizer.py: store_final_boxes_in_guard.
            // resume.py self._add_pending_fields(pending_setfields) —
            // tags target_tagged/value_tagged in place during finish();
            // the descr's set_rd_pendingfields receives the tagged slice
            // post-finish (mod.rs::store_final_boxes_in_guard).
            Self::store_final_boxes_in_guard(op, ctx, knowledge, pending_for_finish);
            // optimizer.py: force_box on each fail_arg for unrolling.
            if let Some(fa) = op.guard_fail_args() {
                let fargs: smallvec::SmallVec<[majit_ir::operand::Operand; 8]> =
                    fa.iter().cloned().collect();
                for farg in &fargs {
                    if !farg.is_none() {
                        self.force_box(farg, ctx);
                    }
                }
            }
            // optimizer.py (called from store_final_boxes_in_guard):
            // GUARD_VALUE → bool replacement. We invoke it here so descr is
            // already set when _maybe_replace_guard_value reads it.
            let replaced = if op.opcode == OpCode::GuardValue {
                Self::_maybe_replace_guard_value_if_changed(op, ctx)
            } else {
                None
            };
            // optimizer.py: update last_guard_op only on the fresh-guard path.
            // The op has not yet been pushed to `ctx.new_operations` (the
            // surrounding `emit_operation` does so at `ctx.emit(op.clone())`
            // shortly after we return), so the predicted donor index is
            // exactly `ctx.new_operations.len()` at this point.
            self.last_guard_op_idx = Some(ctx.new_operations.len());
            // optimizer.py: GUARD_EXCEPTION breaks the chain.
            if opcode == OpCode::GuardException {
                self.last_guard_op_idx = None;
            }
            return replaced;
        }
    }

    /// optimizer.py _copy_resume_data_from
    ///
    /// Inherits descr / fail_args from `_last_guard_op` for the follow-up
    /// descrless guard, then runs `_maybe_replace_guard_value` if the
    /// inheriting op is a `GUARD_VALUE`.
    ///
    /// `compile.py ResumeGuardCopiedDescr(prev)` parity: stamp
    /// the sharer's descr as a `ResumeGuardCopiedDescr` (or
    /// `ResumeGuardCopiedExcDescr` for exception guards) whose
    /// `prev` references the donor's `ResumeGuardDescr`.  Readers
    /// go through `FailDescr::rd_*()` which chases `prev` automatically
    /// (compile.py `get_resumestorage(): return prev`).
    fn _copy_resume_data_from(&mut self, op: &Op, ctx: &mut OptContext) -> Option<Op> {
        let donor_idx = self
            .last_guard_op_idx
            .expect("_copy_resume_data_from requires last_guard_op_idx");
        let last = ctx
            .new_operations
            .get(donor_idx)
            .expect("last_guard_op_idx must point inside ctx.new_operations");
        // compile.py `invent_fail_descr_for_op`: the sharing
        // path mints `ResumeGuardCopiedDescr(prev)` (or
        // `ResumeGuardCopiedExcDescr(prev)` for GUARD_EXCEPTION /
        // GUARD_NO_EXCEPTION) rather than cloning the donor's
        // ResumeGuardDescr — `get_resumestorage()` (compile.py)
        // routes reads back to `prev`.  GUARD_NOT_FORCED never
        // reaches this path (compile.py:921 `assert copied_from_descr
        // is None`), so no Forced-copied variant is needed.
        //
        // optimizer.py:691 `assert isinstance(last_descr,
        // compile.ResumeGuardDescr)` — the donor must be a
        // head-of-chain `ResumeGuardDescr` (never None, never a
        // `ResumeGuardCopiedDescr`).  Mirror the assertion so the
        // sharer never silently ends up with `descr=None` (which
        // would make `op.descr.fail_descr().rd_*()` reads fall back
        // to empty slices) or a two-hop prev.
        let last_descr = last
            .getdescr()
            .expect("optimizer.py:691 last_guard_op.getdescr() must exist");
        assert!(
            last_descr.is_resume_guard(),
            "optimizer.py:691 assert isinstance(last_descr, ResumeGuardDescr): \
             donor descr is not a ResumeGuardDescr"
        );
        op.setdescr({
            use majit_ir::OpCode;
            match op.opcode {
                OpCode::GuardException | OpCode::GuardNoException => {
                    crate::compile::make_resume_guard_copied_exc_descr(last_descr.clone())
                }
                _ => crate::compile::make_resume_guard_copied_descr(last_descr.clone()),
            }
        });
        // optimizer.py _copy_resume_data_from: guard_op.setfailargs(last_guard_op.getfailargs())
        match last.guard_fail_args() {
            Some(fa) => op.setfailargs(fa.iter().cloned().collect()),
            None => op.clearfailargs(),
        }
        op.set_rd_resume_position(last.rd_resume_position());
        // bridgeopt.py parity: the class-knowledge bitfield baked into
        // rd_numb is indexed by the donor's per-livebox type layout.
        // `deserialize_optimizer_knowledge` reads that bitfield using the
        // types it receives as `livebox_types`, which the bridge site
        // sources from the shared guard's `fail_arg_types`. Dropping the
        // copy here leaves the shared guard with `fail_arg_types=None`,
        // the bridge code falls back to the bridge tracer's (unboxed)
        // inputarg types, and the Ref count disagrees with the serializer
        // → rd_numb over-read.
        if let Some(types) = last.get_fail_arg_types() {
            op.set_fail_arg_types(types.to_vec());
        } else {
            op.clear_fail_arg_types();
        }
        // ResumeGuardCopiedDescr(prev) parity (compile.py
        // `get_resumestorage(): return prev`): the descr-side `prev`
        // pointer (set by `make_resume_guard_copied_descr` above)
        // routes resume reads to the donor's RdPayload — no op-index
        // chase pointer needed.
        let _ = donor_idx;
        // optimizer.py:698-699: if guard_op.opnum == GUARD_VALUE: ...
        if op.opcode == OpCode::GuardValue {
            return Self::_maybe_replace_guard_value_if_changed(op, ctx);
        }
        None
    }

    /// optimizer.py store_final_boxes_in_guard
    ///
    /// Resolve fail_args through get_box_replacement and delegate to
    /// finalize_guard_resume_data for snapshot-based virtual encoding
    /// (rd_numb, rd_consts, rd_virtuals).
    ///
    /// resume.py finish() is unconditional: every guard reaching
    /// here must have a snapshot. We delegate without any guard condition
    /// — finalize_guard_resume_data already handles the missing-snapshot
    /// case internally (silent return for guards without rd_resume_position
    /// or snapshot_boxes entry).
    fn store_final_boxes_in_guard(
        op: &Op,
        ctx: &mut OptContext,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        // optimizer.py is_call_pure_pure_canraise + resume.py finish:
        // RPython finish() handles virtuals without forcing.
        // _number_boxes tags virtual fail_args as TAGVIRTUAL,
        // _number_virtuals builds rd_virtuals from PtrInfo.
        //
        // PyPy parity: fail_args are canonicalized through
        // `get_box_replacement(a, True)` in the compile/backend patching
        // helpers. In RPython the second argument is `not_const=True`:
        // the chain walk stops before stepping into a Const target, so
        // the guard fail_arg keeps the runtime box identity here. Const
        // entries are encoded by `resume.py _number_boxes` as TAGCONST
        // in rd_numb during numbering, and the liveboxes returned by
        // `finish()` / `descr.store_final_boxes` remain TAGBOX-only for
        // backend regalloc.
        if let Some(fail_args) = op.fail_args_mut() {
            for fa_idx in 0..fail_args.len() {
                if !fail_args[fa_idx].is_none()
                    && let Some(resolved) =
                        ctx.get_box_replacement_not_const_operand(&fail_args[fa_idx])
                {
                    fail_args[fa_idx] = resolved;
                }
            }
        }
        ctx.finalize_guard_resume_data(op, knowledge, pending_setfields);
    }

    /// bridgeopt.py serialize_optimizer_knowledge parity:
    /// Collect UNFILTERED optimizer knowledge from passes for rd_numb
    /// serialization. available_boxes filtering happens inside
    /// memo.finish() during the actual serialization, matching RPython's
    /// flow where serialize_optheap/serialize_optrewrite receive
    /// available_boxes computed from liveboxes ∩ liveboxes_from_env.
    ///
    /// descr.py:28: v.descr_index = len(all_descrs)
    /// Assigns a sequential descr_index if not already assigned, and
    /// appends to self.all_descrs. Returns the assigned index.
    fn ensure_descr_index(&mut self, descr: &DescrRef) -> i32 {
        let idx = descr.get_descr_index();
        if idx >= 0 {
            return idx;
        }
        let new_idx = self.all_descrs.len() as i32;
        descr.set_descr_index(new_idx);
        Arc::make_mut(&mut self.all_descrs).push(descr.clone());
        // descr.py:47: assert len(all_descrs) < 2**15
        assert!(
            self.all_descrs.len() < (1 << 15),
            "too many descriptors: {}",
            self.all_descrs.len()
        );
        new_idx
    }

    /// virtualize.py postprocess_FINISH, Optimizer half.
    ///
    /// `OptVirtualize::propagate_postprocess` stashed the GUARD_NOT_FORCED_2 it
    /// took off the FINISH; the finalization needs `store_final_boxes_in_guard`
    /// and the knowledge `collect_optimizer_knowledge_for_resume` gathers, both
    /// of which live here.  Running after the FINISH's own emit is the point:
    /// that emit force_box'd the FINISH args, so a return box that was virtual
    /// is materialized by the time it is numbered.
    ///
    /// The guard goes straight into `new_operations` rather than back through
    /// the pass chain, matching `_newoperations.insert` — it has already been
    /// through every pass once, on its way to being stashed.
    fn drain_pending_finish_guard_postprocess(&mut self, ctx: &mut OptContext) {
        let Some(guard_op) = ctx.pending_finish_guard_postprocess.take() else {
            return;
        };
        // virtualize.py:87 store_final_boxes_in_guard(guard_op, []) — the
        // pendingfields argument is the empty list.
        let knowledge_for_resume = self.collect_optimizer_knowledge_for_resume(ctx, &[]);
        let knowledge = if knowledge_for_resume.is_empty() {
            None
        } else {
            Some(knowledge_for_resume)
        };
        Self::store_final_boxes_in_guard(&guard_op, ctx, knowledge, Vec::new());
        // virtualize.py:88-90 `i = len(_newoperations) - 1; assert i >= 0;
        // insert(i, guard_op)` — the FINISH this postprocess belongs to is the
        // last element, so the guard lands immediately in front of it.
        let Some(i) = ctx.new_operations.len().checked_sub(1) else {
            debug_assert!(false, "virtualize.py:89 assert i >= 0");
            return;
        };
        ctx.new_operations.insert(i, OpRc::new(guard_op));
        // `new_operations_index` maps position -> op with last-occurrence-wins
        // semantics, which an insert in the middle cannot maintain
        // incrementally.
        ctx.rebuild_new_operations_index();
    }

    fn collect_optimizer_knowledge_for_resume(
        &mut self,
        ctx: &mut OptContext,
        failargs: &[OpRef],
    ) -> crate::resume::OptimizerKnowledgeForResume {
        let mut heap_fields_raw = Vec::new();
        let mut heap_arrayitems_raw = Vec::new();
        // Guard resume: export all cached fields. The available_boxes filter
        // (serialize_optheap, bridgeopt.py) is applied later in bridgeopt once
        // the live-box set is known; this raw export accepts every cached field.
        for pass in &self.passes {
            let fields = pass.export_cached_fields(ctx);
            let items = pass.export_cached_arrayitems(ctx);
            if !fields.is_empty() || !items.is_empty() {
                heap_fields_raw = fields;
                heap_arrayitems_raw = items;
                break;
            }
        }
        let mut loopinvariant_results = Vec::new();
        for pass in &self.passes {
            for (func_ptr, result) in pass.serialize_optrewrite() {
                let replaced = ctx.get_replacement_opref(result);
                loopinvariant_results.push((func_ptr, replaced));
            }
        }

        // descr.py: assign descr_index inline, matching setup_descrs().
        // heap.py:828: descriptors with get_descr_index() == -1 after
        // ensure_descr_index should not occur — ensure_descr_index always
        // assigns. The filter_map here mirrors the RPython path where
        // heap.py:828 skips descriptors not in the CPU cache.
        let heap_fields: Vec<(OpRef, i32, OpRef)> = heap_fields_raw
            .into_iter()
            .map(|(obj, descr, val)| {
                let idx = self.ensure_descr_index(&descr);
                (obj, idx, val)
            })
            .collect();
        let heap_arrayitems: Vec<(OpRef, i64, i32, OpRef)> = heap_arrayitems_raw
            .into_iter()
            .map(|(obj, index, descr, val)| {
                let idx = self.ensure_descr_index(&descr);
                (obj, index, idx, val)
            })
            .collect();

        // bridgeopt.py:74-88: known_classes bitfield is now computed
        // directly at serialization time in resume.rs via env.has_known_class(),
        // matching RPython's per-livebox getptrinfo(box).get_known_class(cpu).

        crate::resume::OptimizerKnowledgeForResume {
            heap_fields,
            heap_arrayitems,
            loopinvariant_results,
        }
    }

    /// optimizer.py _maybe_replace_guard_value
    ///
    /// ```text
    /// def _maybe_replace_guard_value(self, op, descr):
    ///     if op.getarg(0).type == 'i':
    ///         b = self.getintbound(op.getarg(0))
    ///         if b.is_bool():
    ///             # Hack: turn guard_value(bool) into guard_true/guard_false.
    ///             constvalue = op.getarg(1).getint()
    ///             if constvalue == 0:
    ///                 opnum = rop.GUARD_FALSE
    ///             elif constvalue == 1:
    ///                 opnum = rop.GUARD_TRUE
    ///             else:
    ///                 # Issue #3128: rare cases — give up rather than crash.
    ///                 return op
    ///             newop = self.replace_op_with(op, opnum, [op.getarg(0)], descr)
    ///             return newop
    ///     return op
    /// ```
    fn _maybe_replace_guard_value_if_changed(op: &Op, ctx: &mut OptContext) -> Option<Op> {
        // optimizer.py:755: if op.getarg(0).type == 'i'
        let arg0 = op.arg(0);
        if ctx.opref_type(arg0.to_opref()) != Some(majit_ir::Type::Int) {
            return None;
        }
        // optimizer.py: b = self.getintbound(op.getarg(0)); if b.is_bool()
        let b = {
            let b = ctx.resolve_operand_operand(&arg0);
            ctx.getintbound_handle(&b).borrow().clone()
        };
        if !b.is_bool() {
            return None;
        }
        // optimizer.py:762: constvalue = op.getarg(1).getint()
        let constvalue = op.arg(1).get_box_replacement(false).const_int()?;
        // optimizer.py:763-775: 0 → GUARD_FALSE, 1 → GUARD_TRUE, else give up.
        let new_opcode = match constvalue {
            0 => OpCode::GuardFalse,
            1 => OpCode::GuardTrue,
            _ => return None,
        };
        // optimizer.py: replace_op_with(op, opnum, [op.getarg(0)], descr)
        let mut newop = Op::new(new_opcode, &[arg0]);
        newop.pos().set(op.pos().get());
        if let Some(d) = op.getdescr() {
            newop.setdescr(d);
        }
        match op.guard_fail_args() {
            Some(fa) => newop.setfailargs(fa.iter().cloned().collect()),
            None => newop.clearfailargs(),
        }
        match op.get_fail_arg_types() {
            Some(types) => newop.set_fail_arg_types(types.to_vec()),
            None => newop.clear_fail_arg_types(),
        }
        // compile.py _attrs_ live on the descr; Arc-clone of
        // op.descr above shares the donor's RdPayload, so newop's
        // FailDescr::rd_* readers see the same data.
        newop.set_rd_resume_position(op.rd_resume_position());
        Some(newop)
    }
}

impl Optimizer {
    /// `optimizeopt/__init__.py` `build_opt_chain`. Passes are the names in
    /// `enable_opts`, in `ALL_OPTS` order. `unroll` is not a pass.
    /// `OptSimplify` is appended when `rewrite`, `virtualize`, `heap`, or
    /// `pure` is absent.
    pub(crate) fn build_opt_chain(
        enable_opts: &[String],
        vable: Option<VirtualizableConfig>,
    ) -> Self {
        let enabled = |name: &str| enable_opts.iter().any(|opt| opt == name);
        let mut opt = Self::new();
        if let Some(config) = vable.as_ref() {
            // resume.py:399-402 `minimum_virtualizable_size` from
            // `virtualizable_info.minimum_size()` (`num_static_fields`).
            // The size is a property of the driver, not of whether the
            // `virtualize` pass is in the chain.
            opt.minimum_virtualizable_size = config.static_field_offsets.len() as i64;
        }
        for (name, instantiate) in ALL_OPTS {
            if !instantiate || !enabled(name) {
                continue;
            }
            let pass: Box<dyn Optimization> = match *name {
                "intbounds" => Box::new(OptIntBounds::new()),
                "rewrite" => Box::new(OptRewrite::new()),
                "virtualize" => {
                    if let Some(config) = vable.clone() {
                        Box::new(OptVirtualize::with_virtualizable(config))
                    } else {
                        Box::new(OptVirtualize::new())
                    }
                }
                "string" => Box::new(OptString::new()),
                "pure" => Box::new(OptPure::new()),
                "earlyforce" => Box::new(OptEarlyForce::new()),
                "heap" => Box::new(OptHeap::new()),
                _ => continue,
            };
            opt.add_pass(pass);
        }
        if !enabled("rewrite") || !enabled("virtualize") || !enabled("heap") || !enabled("pure") {
            opt.add_pass(Box::new(OptSimplify::new()));
        }
        opt
    }

    /// Create an optimizer with the standard pass pipeline.
    /// `ENABLE_ALL_OPTS` (`unroll` is handled by `UnrollOptimizer`).
    pub fn default_pipeline() -> Self {
        Self::build_opt_chain(&default_enable_opt_names(), None)
    }

    /// Create an optimizer with virtualizable config for frame field tracking.
    ///
    /// Deviation, deliberate: RPython gates `minimum_virtualizable_size` on
    /// the STATIC per-jitdriver `virtualizable_info`, while this gates on the
    /// PER-TRACE config, which exists only when `has_virtualizable_boxes()`
    /// held at `make_optimizer`. Ours is therefore strictly narrower — it
    /// cannot fire on a trace that never installed a shadow. That direction
    /// is the safe one (no false positives), but it does mean a trace whose
    /// `virtualizable_boxes` is `None` while the driver's `vinfo` is `Some`
    /// still slips past here and is caught only by the reader's
    /// `assert!(vable_size > 0)` in `resume.rs::consume_vable_info`.
    #[allow(dead_code)] // optimizeopt/__init__.py build_opt_chain virtualizable
    pub(crate) fn default_pipeline_with_virtualizable(config: VirtualizableConfig) -> Self {
        Self::build_opt_chain(&default_enable_opt_names(), Some(config))
    }

    /// Number of passes in this optimizer.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }

    /// Names of the installed passes, in chain order.
    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name()).collect()
    }
}

fn default_enable_opt_names() -> Vec<String> {
    ENABLE_ALL_OPTS
        .split(':')
        .filter(|name| !name.is_empty())
        .map(String::from)
        .collect()
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

/// Select the replay identity for a non-const-channel short box.
///
/// PyPy keys `ShortBoxes.potential_ops` by the original result Box even when
/// that Box was subsequently forwarded to a Const by guard postprocessing.
/// Such an entry is useful only for `PureOp`: it seeds the next iteration's
/// pure cache without becoming a carried constant slot.
fn exported_short_box_replay_result(
    result: OpRef,
    canonical_result: OpRef,
    kind: &crate::optimizeopt::shortpreamble::PreambleOpKind,
) -> Option<OpRef> {
    if !canonical_result.is_constant() {
        return Some(canonical_result);
    }
    (*kind == crate::optimizeopt::shortpreamble::PreambleOpKind::Pure).then_some(result)
}

// OptimizerBoxEnv removed: was only used by store_final_boxes_in_guard.
// store_final_boxes_in_guard in mod.rs defines InlineBoxEnv for the same purpose.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_resop_operand;
    use majit_ir::Type;
    use majit_ir::descr::make_size_descr;
    use majit_ir::descr::{CallDescr, EffectInfo, ExtraEffect, OopSpecIndex};
    use majit_ir::operand::Operand;
    use majit_ir::{DescrRef, OpCode, OpRef};
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    #[test]
    fn guarded_pure_constant_keeps_its_original_short_box_identity() {
        use crate::optimizeopt::shortpreamble::PreambleOpKind;

        let original = OpRef::int_op(17);
        let proven_true = OpRef::const_int(1);
        assert_eq!(
            exported_short_box_replay_result(original, proven_true, &PreambleOpKind::Pure),
            Some(original),
            "pure.py produce_potential_short_preamble_ops exports the emitted \
             comparison Box even after postprocess_GUARD_TRUE forwards it to CONST_1",
        );
        assert_eq!(
            exported_short_box_replay_result(original, proven_true, &PreambleOpKind::Heap),
            None,
            "only PureOp can seed a cache without carrying a constant result slot",
        );
        let loop_value = OpRef::int_op(23);
        assert_eq!(
            exported_short_box_replay_result(original, loop_value, &PreambleOpKind::Pure),
            Some(loop_value),
        );
    }

    /// `OpRc`-threading analogue of [`super::super::seed_empty_guard_snapshots`]
    /// for fixtures built with [`TraceBuilder`]: assigns each guard a fresh
    /// resume position in place (`set_rd_resume_position`, mutable
    /// behind the `Rc`) and inserts an empty active-frame snapshot, so the
    /// canonical producer `Rc<Op>` identity survives for the
    /// `optimize_with_constants_and_inputs_oprc` driver.
    fn seed_guard_snapshots_with_oprc<F>(
        ops: &[majit_ir::OpRc],
        mut snapshot_for_guard: F,
    ) -> crate::optimizeopt::SnapshotBoxes
    where
        F: FnMut(&Op) -> Vec<OpRef>,
    {
        use crate::resume::SnapshotBox;
        let mut snapshots: crate::optimizeopt::SnapshotBoxes = Vec::new();
        let mut next_resume_pos = 0i32;
        for op in ops.iter().filter(|op| op.opcode.is_guard()) {
            let snapshot_boxes = snapshot_for_guard(op);
            let resume_pos = if op.rd_resume_position() >= 0
                && !crate::optimizeopt::snapshot_contains(&snapshots, op.rd_resume_position())
            {
                op.rd_resume_position()
            } else {
                while crate::optimizeopt::snapshot_contains(&snapshots, next_resume_pos) {
                    next_resume_pos += 1;
                }
                let resume_pos = next_resume_pos;
                next_resume_pos += 1;
                resume_pos
            };
            op.set_rd_resume_position(resume_pos);
            crate::optimizeopt::snapshot_insert(
                &mut snapshots,
                resume_pos,
                snapshot_boxes.into_iter().map(SnapshotBox::from).collect(),
            );
        }
        snapshots
    }

    fn seed_empty_guard_snapshots_oprc(
        ops: &[majit_ir::OpRc],
    ) -> crate::optimizeopt::SnapshotBoxes {
        seed_guard_snapshots_with_oprc(ops, |_| Vec::new())
    }

    /// A trivial pass that removes INT_ADD(x, 0) -> x
    #[allow(dead_code)]
    struct AddZeroElimination;

    impl Optimization for AddZeroElimination {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                // Check if second arg is constant 0
                if let Some(0) = ctx.get_constant_int_box(&op.arg(1).get_box_replacement(false)) {
                    // Replace with first arg
                    let old = op.pos().get();
                    let new = op.arg(0).to_opref();
                    let b_old = ctx.materialize_operand_at(old);
                    let b_new = ctx.materialize_operand_at(new);
                    ctx.make_equal_to(&b_old, &b_new);
                    return OptimizationResult::Remove;
                }
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "add_zero_elim"
        }
    }

    struct AddVirtualInputsOnce {
        added: bool,
    }

    struct RemoveAsConstant {
        target: OpRef,
        value: i64,
    }

    impl Optimization for AddVirtualInputsOnce {
        fn propagate_forward(
            &mut self,
            _op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if !self.added {
                ctx.num_inputs += 2;
                self.added = true;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "add_virtual_inputs_once"
        }
    }

    impl Optimization for RemoveAsConstant {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if op.pos().get() == self.target {
                let b = majit_ir::operand::Operand::from_bound_op(_op_rc);
                ctx.make_constant_box(&b, majit_ir::Value::Int(self.value));
                return OptimizationResult::Remove;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "remove_as_constant"
        }
    }

    struct FlushCounter {
        hits: Rc<Cell<usize>>,
    }

    impl Optimization for FlushCounter {
        fn propagate_forward(
            &mut self,
            _op: &Op,
            _op_rc: &majit_ir::OpRc,
            _ctx: &mut OptContext,
        ) -> OptimizationResult {
            OptimizationResult::PassOn
        }

        fn flush(&mut self, _ctx: &mut OptContext) {
            self.hits.set(self.hits.get() + 1);
        }

        fn name(&self) -> &'static str {
            "flush_counter"
        }
    }

    #[allow(dead_code)]
    struct RemoveAsTypedConstant {
        target: OpRef,
        value: majit_ir::Value,
    }

    impl Optimization for RemoveAsTypedConstant {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if op.pos().get() == self.target {
                let b = ctx.materialize_operand_at(op.pos().get());
                ctx.make_constant_box(&b, self.value);
                return OptimizationResult::Remove;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "remove_as_typed_constant"
        }
    }

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
    }

    impl majit_ir::Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
    }

    fn test_parent_descr() -> majit_ir::DescrRef {
        std::sync::Arc::new(TestSizeDescr { index: 0xFFFF_0000 })
    }

    #[derive(Debug)]
    struct TestDescr(u32);

    impl majit_ir::Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
        fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
            Some(self)
        }
    }

    impl majit_ir::FieldDescr for TestDescr {
        fn get_parent_descr(&self) -> Option<majit_ir::DescrRef> {
            Some(test_parent_descr())
        }
        fn offset(&self) -> usize {
            self.0 as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    #[derive(Debug)]
    struct TestCallDescr {
        idx: u32,
        effect: EffectInfo,
        result_type: majit_ir::Type,
    }

    impl majit_ir::Descr for TestCallDescr {
        fn index(&self) -> u32 {
            self.idx
        }

        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[majit_ir::Type] {
            &[]
        }

        fn result_type(&self) -> majit_ir::Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn get_extra_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    fn call_may_force_descr(idx: u32, result_type: majit_ir::Type) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            },
            result_type,
        })
    }

    struct QueueForceLikeExtraOps {
        queued: bool,
        field_descr: majit_ir::DescrRef,
    }

    impl Optimization for QueueForceLikeExtraOps {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntAdd {
                self.queued = true;

                // The IntAdd args are the only objects at those positions.
                ctx.seed_boxes_canonical(&[op.arg(0), op.arg(1)]);
                let alloc = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let alloc_box = ctx.materialize_operand_at(alloc);
                let value = op.arg(0);
                let mut set = Op::new(OpCode::SetfieldGc, &[alloc_box, value]);
                set.setdescr(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_force_like_extra_ops"
        }
    }

    struct QueueRestartCandidate {
        queued: bool,
    }

    impl Optimization for QueueRestartCandidate {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntMul {
                self.queued = true;
                ctx.emit_extra(
                    ctx.current_pass_idx,
                    Op::new(OpCode::IntAdd, &[op.arg(0), op.arg(1)]),
                );
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_restart_candidate"
        }
    }

    struct CountRestartedIntSub {
        hits: Rc<Cell<usize>>,
    }

    impl Optimization for CountRestartedIntSub {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            _ctx: &mut OptContext,
        ) -> OptimizationResult {
            if op.opcode == OpCode::IntSub {
                self.hits.set(self.hits.get() + 1);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "count_restarted_int_sub"
        }
    }

    struct RestartIntAddAsSub;

    impl Optimization for RestartIntAddAsSub {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            _ctx: &mut OptContext,
        ) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                let mut restarted = Op::new(OpCode::IntSub, &[op.arg(0), op.arg(1)]);
                restarted.pos().set(op.pos().get());
                return OptimizationResult::Restart(restarted);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "restart_int_add_as_sub"
        }
    }

    struct QueueExtraThenInvalidate;

    impl Optimization for QueueExtraThenInvalidate {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                ctx.emit_extra(
                    ctx.current_pass_idx,
                    Op::new(OpCode::IntMul, &[op.arg(0), op.arg(1)]),
                );
                ctx.signal_invalid_loop("test invalid loop while draining extras");
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_extra_then_invalidate"
        }
    }

    #[test]
    fn test_optimizer_passthrough() {
        let mut opt = Optimizer::new();
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
        )];
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn invalid_loop_preserves_queued_and_unprocessed_extra_operations() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueExtraThenInvalidate));
        let mut ctx = OptContext::new(2);
        let lhs = rooted_resop_operand(Type::Int, 0);
        let rhs = rooted_resop_operand(Type::Int, 1);
        let add = OpRc::new(Op::new(OpCode::IntAdd, &[lhs.clone(), rhs.clone()]));
        let sub = OpRc::new(Op::new(OpCode::IntSub, &[lhs, rhs]));
        // Result positions stay off the arg boxes. `emit_extra_at` would
        // otherwise number a pos-less IntAdd as int_op(0), the same OpRef
        // as `lhs`.
        add.pos().set(OpRef::int_op(2));
        sub.pos().set(OpRef::int_op(3));
        ctx.emit_extra_at(0, add);
        ctx.emit_extra_at(0, sub);

        let result = opt.drain_extra_operations_from(0, &mut ctx);

        assert!(result.is_err());
        assert!(ctx.extra_pending.is_empty());
        let queued: Vec<_> = ctx
            .extra_operations_after
            .iter()
            .map(|(_, op)| op.opcode)
            .collect();
        assert_eq!(queued, [OpCode::IntMul, OpCode::IntSub]);
    }

    #[test]
    fn test_restart_from_extra_operation_rediscovers_first_pass() {
        let hits = Rc::new(Cell::new(0));
        let mut opt = Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&[majit_ir::Type::Int; 8]);
        opt.add_pass(Box::new(QueueRestartCandidate { queued: false }));
        opt.add_pass(Box::new(RestartIntAddAsSub));
        opt.add_pass(Box::new(CountRestartedIntSub { hits: hits.clone() }));

        let mut ops = vec![Op::new(
            OpCode::IntMul,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
        )];
        ops[0].pos().set(OpRef::int_op(2));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 2);

        assert_eq!(
            hits.get(),
            1,
            "Restart must re-dispatch from first_optimization, as optimizer.py:567 does"
        );
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntSub),
            "restarted extra operation should be emitted as IntSub: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_keeps_call_may_force_pairs_alive_when_results_are_used() {
        let field_group = majit_ir::descr::make_simple_descr_group(
            91,
            16,
            1,
            0,
            &[majit_ir::descr::SimpleFieldDescrSpec {
                is_class_word: Some(false),
                index: 91,
                field_key: "CallResult.field".to_string(),
                name: "CallResult.field".to_string(),
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                is_immutable: true,
                is_quasi_immutable: false,
                flag: majit_ir::ArrayFlag::Signed,
                virtualizable: false,
                index_in_parent: 0,
            }],
        );
        let field_descr = field_group.field_descrs[0].clone() as DescrRef;
        let call_descr_a = call_may_force_descr(81, majit_ir::Type::Ref);
        let call_descr_b = call_may_force_descr(82, majit_ir::Type::Ref);
        let mut ops = vec![
            Op::with_descr(
                OpCode::CallMayForceR,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
                call_descr_a,
            ),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_resop_operand(Type::Ref, 3)],
                field_descr.clone(),
            ),
            Op::with_descr(
                OpCode::CallMayForceR,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 2),
                ],
                call_descr_b,
            ),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[rooted_resop_operand(Type::Ref, 6)],
                field_descr,
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 5),
                    rooted_resop_operand(Type::Int, 8),
                ],
            ),
            Op::new(OpCode::Finish, &[rooted_resop_operand(Type::Int, 9)]),
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos()
                .set(OpRef::op_typed((idx as u32) + 3, op.opcode.result_type()));
        }

        let mut opt = Optimizer::default_pipeline();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 3);

        let call_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .count();
        let guard_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(
            call_count, 2,
            "optimized trace lost CallMayForceR ops: {result:?}"
        );
        assert_eq!(
            guard_count, 2,
            "optimized trace lost GuardNotForced ops: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_keeps_call_may_force_when_guard_fail_args_reference_results() {
        let field_group = majit_ir::descr::make_simple_descr_group(
            101,
            24,
            2,
            0,
            &[
                majit_ir::descr::SimpleFieldDescrSpec {
                    is_class_word: Some(false),
                    index: 101,
                    field_key: "CallResult.type".to_string(),
                    name: "CallResult.type".to_string(),
                    offset: 0,
                    field_size: 8,
                    field_type: Type::Int,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    flag: majit_ir::ArrayFlag::Signed,
                    virtualizable: false,
                    index_in_parent: 0,
                },
                majit_ir::descr::SimpleFieldDescrSpec {
                    is_class_word: Some(false),
                    index: 102,
                    field_key: "CallResult.value".to_string(),
                    name: "CallResult.value".to_string(),
                    offset: 8,
                    field_size: 8,
                    field_type: Type::Int,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    flag: majit_ir::ArrayFlag::Signed,
                    virtualizable: false,
                    index_in_parent: 1,
                },
            ],
        );
        let field_descr = field_group.field_descrs[0].clone() as DescrRef;
        // Distinct field so `get_a_val` is not CSE-folded into `get_a_type`:
        // it stays a live producer at its own position, which guard_b's fail
        // args reference (a fail arg pointing at a *folded* getfield would make
        // the position-only synthetic resolve to itself while the OpRef store
        // forwards to the survivor — the `resolve_operand_operand` divergence tripwire,
        // `materialize_operand_at`; binding to the real folded producer needs the oprc
        // driver, blocked here by CallMayForceR's void result position).
        let field_descr_b = field_group.field_descrs[1].clone() as DescrRef;
        let call_descr_a = call_may_force_descr(83, majit_ir::Type::Ref);
        let call_descr_b = call_may_force_descr(84, majit_ir::Type::Ref);
        let guard_types_a = vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
        ];
        let guard_types_b = vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
        ];

        // Position-only op-args / fail-args replaced by the `rooted_resop_operand`
        // drop-in (a bound ResOp operand whose synthetic producer is rooted in
        // the thread-local pool; sheds to `Operand::Op`, `to_opref`s to the same
        // `(type, position)` so position-keyed resolution is unchanged).
        // CallMayForceR's result is consumed as a Ref by the getfields/fail-args,
        // so its result refs are `Type::Ref` at the call position; the resume-only
        // free fail-vars (2000..3003) and the dangling positions stay `Type::Int`
        // exactly as the fixture wired them. Detached fail-arg synthetics resolve
        // to themselves (the `same_box` arm of `resolve_operand_operand`),
        // deferring to the OpRef
        // store — no `Operand::Box`.
        let mut call_a = Op::with_descr(
            OpCode::CallMayForceR,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
            call_descr_a,
        );
        let mut guard_a = Op::with_descr(
            OpCode::GuardNotForced,
            &[],
            crate::compile::make_resume_guard_forced_descr_typed(guard_types_a.clone()),
        );
        guard_a.setfailargs(
            vec![
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 2000),
                rooted_resop_operand(Type::Int, 2001),
                rooted_resop_operand(Type::Int, 3),
                rooted_resop_operand(Type::Int, 3000),
                rooted_resop_operand(Type::Int, 3001),
                rooted_resop_operand(Type::Int, 4),
            ]
            .into(),
        );
        guard_a.set_fail_arg_types(guard_types_a);
        let get_a_type = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rooted_resop_operand(Type::Ref, 3)],
            field_descr.clone(),
        );
        let get_a_val = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rooted_resop_operand(Type::Ref, 3)],
            field_descr_b.clone(),
        );
        let mut call_b = Op::with_descr(
            OpCode::CallMayForceR,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 2),
            ],
            call_descr_b,
        );
        let mut guard_b = Op::with_descr(
            OpCode::GuardNotForced,
            &[],
            crate::compile::make_resume_guard_forced_descr_typed(guard_types_b.clone()),
        );
        guard_b.setfailargs(
            vec![
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 2002),
                rooted_resop_operand(Type::Int, 2003),
                rooted_resop_operand(Type::Int, 3),
                rooted_resop_operand(Type::Int, 6),
                rooted_resop_operand(Type::Int, 3002),
                rooted_resop_operand(Type::Int, 3003),
                rooted_resop_operand(Type::Int, 7),
            ]
            .into(),
        );
        guard_b.set_fail_arg_types(guard_types_b);
        let get_b_type = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rooted_resop_operand(Type::Ref, 7)],
            field_descr.clone(),
        );
        let get_b_val = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rooted_resop_operand(Type::Ref, 7)],
            field_descr,
        );
        // Second add arg references a live int getfield result (int_op(9) =
        // get_b_type) rather than the original fixture's dangling position 8
        // (the void guard_b): a bound synthetic at a void position has no
        // forwardable result box, so IntBounds' getintbound would write
        // forwarded onto an unbound box (forwarding.rs). The add result is
        // unused either way — this only keeps its operands resolvable.
        let add = Op::new(
            OpCode::IntAdd,
            &[
                rooted_resop_operand(Type::Int, 5),
                rooted_resop_operand(Type::Int, 9),
            ],
        );
        let finish = Op::new(OpCode::Finish, &[rooted_resop_operand(Type::Int, 9)]);

        let mut ops = vec![
            call_a.clone(),
            guard_a,
            get_a_type,
            get_a_val,
            call_b.clone(),
            guard_b,
            get_b_type,
            get_b_val,
            add,
            finish,
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos()
                .set(OpRef::op_typed((idx as u32) + 3, op.opcode.result_type()));
        }
        call_a.pos().set(ops[0].pos().get());
        call_b.pos().set(ops[4].pos().get());

        let mut opt = Optimizer::default_pipeline();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 3);

        let call_positions: indexmap::IndexSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .map(|op| op.pos().get())
            .collect();
        assert!(
            call_positions.contains(&call_a.pos().get())
                && call_positions.contains(&call_b.pos().get()),
            "optimized trace lost CallMayForceR producer(s): {result:?}"
        );
        let guarded = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(
            guarded, 2,
            "optimized trace lost GuardNotForced ops: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_has_7_passes() {
        // ENABLE_ALL_OPTS minus `unroll` (not instantiated) and without
        // OptSimplify, which is appended only when a core pass is absent.
        let opt = Optimizer::default_pipeline();
        assert_eq!(opt.num_passes(), 7);
        assert_eq!(
            opt.pass_names(),
            [
                "intbounds",
                "rewrite",
                "virtualize",
                "string",
                "pure",
                "earlyforce",
                "heap",
            ]
        );
    }

    /// `optimizeopt/test/test_optimizeopt.py` `test_build_opt_chain`.
    #[test]
    fn test_build_opt_chain_follows_all_opts_order() {
        let names = |spec: &str| {
            let enable_opts: Vec<String> = spec
                .split(':')
                .filter(|name| !name.is_empty())
                .map(String::from)
                .collect();
            Optimizer::build_opt_chain(&enable_opts, None).pass_names()
        };
        assert_eq!(names(""), ["simplify"]);
        assert_eq!(names("heap:intbounds"), ["intbounds", "heap", "simplify"]);
        assert_eq!(names("unroll"), ["simplify"]);
        assert_eq!(names("aaa:bbb"), ["simplify"]);
        assert!(
            !names("intbounds:rewrite:virtualize:string:pure:earlyforce")
                .iter()
                .any(|name| *name == "heap"),
            "a chain built without heap has no heap pass",
        );
    }

    #[test]
    fn test_default_pipeline_processes_trace() {
        use crate::history::test_support::TraceBuilder;
        let mut opt = Optimizer::default_pipeline();
        // A simple trace: two INT_ADD with identical args. The Pure pass (CSE)
        // should eliminate the duplicate. The two IntAdd reference the SAME
        // header input boxes so the resolved-OpRef CSE key matches.
        let mut b = TraceBuilder::new();
        let x = b.input(Type::Int, 0);
        let y = b.input(Type::Int, 1);
        b.op(OpCode::IntAdd, &[x.clone(), y.clone()]);
        b.op(OpCode::IntAdd, &[x, y]);
        b.op(OpCode::Jump, &[]);
        let (ops, inputs) = b.build();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                num_inputs,
            )
            .expect("test: unexpected InvalidLoop");
        // The duplicate INT_ADD should be eliminated by CSE (OptPure).
        let add_count = result.iter().filter(|o| o.opcode == OpCode::IntAdd).count();
        assert_eq!(add_count, 1, "CSE should eliminate duplicate INT_ADD");
        // Jump should still be present.
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    #[test]
    fn test_remaps_all_op_positions_when_virtual_inputs_are_added() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));

        let mut ops = vec![
            Op::new(OpCode::GetfieldRawI, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::GetfieldRawI, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::GetfieldRawI, &[rooted_resop_operand(Type::Int, 4)]),
            Op::new(
                OpCode::IntGt,
                &[
                    rooted_resop_operand(Type::Int, 5),
                    Operand::const_from_value(majit_ir::Value::Int(27)),
                ],
            ),
        ];
        ops[0].pos().set(OpRef::int_op(3));
        ops[1].pos().set(OpRef::int_op(4));
        ops[2].pos().set(OpRef::int_op(5));
        ops[3].pos().set(OpRef::int_op(6));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 3);

        let positions: Vec<_> = result.iter().map(|op| op.pos().get()).collect();
        assert_eq!(
            positions,
            vec![
                OpRef::int_op(5),
                OpRef::int_op(6),
                OpRef::int_op(7),
                OpRef::int_op(8)
            ]
        );
        assert_eq!(result[2].arg(0).to_opref(), OpRef::int_op(6));
        assert_eq!(result[3].arg(0).to_opref(), OpRef::int_op(7));
    }

    #[test]
    fn test_remaps_removed_constants_away_from_compacted_live_ops() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));
        opt.add_pass(Box::new(RemoveAsConstant {
            target: OpRef::int_op(5),
            value: 123,
        }));

        let mut ops = vec![
            Op::new(OpCode::GetfieldRawI, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::GetfieldRawI, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::GetfieldRawI, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(
                OpCode::IntGt,
                &[
                    rooted_resop_operand(Type::Int, 3),
                    Operand::const_from_value(majit_ir::Value::Int(27)),
                ],
            ),
        ];
        ops[0].pos().set(OpRef::int_op(3));
        ops[1].pos().set(OpRef::int_op(4));
        ops[2].pos().set(OpRef::int_op(5));
        ops[3].pos().set(OpRef::int_op(6));
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        assert_eq!(result[0].pos().get(), OpRef::int_op(5));
        assert_eq!(result[1].pos().get(), OpRef::int_op(6));
        assert_eq!(result[2].pos().get(), OpRef::int_op(7));
        assert_eq!(result[2].arg(0).to_opref(), OpRef::int_op(5));
    }

    #[test]
    fn test_skip_flush_keeps_terminal_jump_out_of_result_ops() {
        let mut opt = Optimizer::new();
        opt.skip_flush = true;

        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]),
        ];
        ops[0].pos().set(OpRef::int_op(2));
        ops[1].pos().set(OpRef::void_op(3));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        let terminal = opt
            .terminal_op
            .as_ref()
            .expect("skip_flush should preserve terminal jump");
        assert_eq!(terminal.opcode, OpCode::Jump);
        assert_eq!(
            terminal
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(2)]
        );
    }

    #[test]
    fn test_optimize_loop_keeps_finish_and_does_not_export() {
        // compile.py SimpleCompileData.optimize → Optimizer.optimize_loop:
        // a DoneWithThisFrame FINISH is final (BasicLoopInfo.final is True)
        // and must stay in the op list. optimize_bridge would park it in
        // terminal_op / try jump matching.
        let mut opt = Optimizer::default_pipeline();
        let mut finish = Op::new(OpCode::Finish, &[rooted_resop_operand(Type::Int, 0)]);
        finish.pos().set(OpRef::void_op(1));
        let ops = vec![OpRc::new(finish)];
        opt.trace_inputargs = OpRef::inputarg_refs(&[Type::Int]);
        let mut constants = majit_ir::ConstMap::default();
        let result = opt
            .optimize_loop(&ops, &mut constants, 1, None, 0)
            .expect("optimize_loop must accept a FINISH-only trace");
        assert!(
            opt.simple_compile,
            "SimpleCompileData.optimize sets Optimizer.simple_compile"
        );
        assert!(
            opt.exported_loop_state.is_none(),
            "BasicLoopInfo.final is True; optimize_loop must not export for retrace"
        );
        assert!(
            result.last().is_some_and(|op| op.opcode == OpCode::Finish),
            "FINISH must remain in the optimized ops, got {result:?}"
        );
    }

    /// A guard snapshot can name an inputarg that no operation reads.
    /// `ResumeDataLoopMemo.number` resolves it through the trace's own box
    /// (`TraceIterator._cache`), so the optimizer must bind that same object
    /// rather than mint a second one that `import_state` would forward
    /// instead.
    #[test]
    fn trace_inputarg_boxes_bind_an_inputarg_no_operation_reads() {
        use crate::history::test_support::TraceBuilder;
        let mut opt = Optimizer::default_pipeline();
        let mut b = TraceBuilder::new();
        let x = b.input(Type::Int, 0);
        let y = b.input(Type::Int, 1);
        b.op(OpCode::Finish, std::slice::from_ref(&x));
        let (ops, inputs) = b.build();
        let x_box = x.bound_inputarg().expect("x is an inputarg");
        let y_box = y.bound_inputarg().expect("y is an inputarg");
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.trace_inputarg_boxes = vec![x_box.clone(), y_box.clone()];
        opt.snapshot_boxes = seed_empty_guard_snapshots_oprc(&ops);
        opt.optimize_with_constants_and_inputs_oprc(
            &ops,
            &mut majit_ir::ConstMap::default(),
            inputs.len(),
        )
        .expect("test: unexpected InvalidLoop");
        let ctx = opt.final_ctx.as_ref().expect("optimize stores its context");
        let bound = ctx.inputarg_refs.get(&1).expect("y is bound");
        assert!(
            majit_ir::InputArgRc::ptr_eq(bound, &y_box),
            "the unread inputarg must stay the trace's own box"
        );
    }

    #[test]
    fn test_get_count_of_ops_and_guards() {
        use crate::history::test_support::TraceBuilder;
        let mut opt = Optimizer::default_pipeline();
        // This test only exercises the ops/guard counter; each inputarg
        // is read by at least one Int-shape consumer (GuardTrue, IntAdd).
        // optimizer.py:691 `assert isinstance(last_descr, ResumeGuardDescr)`:
        // the first GuardTrue becomes the sharing donor for the
        // descrless GuardNonnull below — give it a real descr so
        // OptContext::emit_guard_operation finds a valid donor.
        let mut b = TraceBuilder::new();
        let x = b.input(Type::Int, 0);
        let y = b.input(Type::Int, 1);
        b.op_with_descr(
            OpCode::GuardTrue,
            std::slice::from_ref(&x),
            crate::compile::make_resume_guard_descr_typed(Vec::new()),
        );
        b.op(OpCode::IntAdd, &[x.clone(), y]);
        b.op(OpCode::GuardNonnull, &[x]);
        b.op(OpCode::Finish, &[]);
        let (ops, inputs) = b.build();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        let num_inputs = inputs.len();
        opt.snapshot_boxes = seed_empty_guard_snapshots_oprc(&ops);
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                num_inputs,
            )
            .expect("test: unexpected InvalidLoop");
        let ctx = OptContext::new(result.len());
        // Just verify the counting methods work
        assert_eq!(Optimizer::get_count_of_ops(&ctx), 0); // empty ctx
    }

    #[test]
    fn test_ref_guard_value_is_not_bool_specialized() {
        let mut opt = Optimizer::new();
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::New, &[]),
            Op::new(
                OpCode::GuardValue,
                &[
                    rooted_resop_operand(Type::Ref, 0),
                    rooted_resop_operand(Type::Ref, 1),
                ],
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        ops[0].pos().set(OpRef::ref_op(0));
        ops[1].pos().set(OpRef::ref_op(1));

        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 0);

        assert!(
            result.iter().any(|op| op.opcode == OpCode::GuardValue),
            "ref-typed GuardValue should remain GuardValue"
        );
        assert!(
            !result
                .iter()
                .any(|op| op.opcode == OpCode::GuardTrue || op.opcode == OpCode::GuardFalse),
            "ref-typed GuardValue must not be rewritten as a bool guard"
        );
    }

    #[test]
    fn test_flush_invokes_all_passes() {
        let hits = Rc::new(Cell::new(0));
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(FlushCounter { hits: hits.clone() }));
        opt.add_pass(Box::new(FlushCounter { hits: hits.clone() }));

        let mut ctx = OptContext::new(0);
        opt.flush(&mut ctx).unwrap();

        assert_eq!(hits.get(), 2);
    }

    #[test]
    fn test_extra_ops_do_not_flow_into_unroll_buffer() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(1)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));
        opt.add_pass(Box::new(crate::optimizeopt::unroll::OptUnroll::new()));

        let v0 = rooted_resop_operand(Type::Int, 0);
        let v1 = rooted_resop_operand(Type::Int, 1);
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[v0.clone(), v1.clone()]),
            Op::new(OpCode::Jump, &[v0, v1]),
        ];
        ops[0].pos().set(OpRef::int_op(2));
        ops[1].pos().set(OpRef::void_op(3));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        // force_all_lazy_setfields emits lazy SetfieldGc before JUMP.
        let new_count = result.iter().filter(|op| op.opcode == OpCode::New).count();
        assert!(
            new_count > 0,
            "force-like extra ops should still emit a New; got {:?}",
            result
        );
        // SetfieldGc is emitted by force_all_lazy_setfields at JUMP
        let setfield_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(
            setfield_count, 1,
            "lazy SetfieldGc should be emitted at JUMP; got {:?}",
            result
        );
    }

    struct QueueTwoForceLikePairs {
        queued: bool,
        field_descr: majit_ir::DescrRef,
    }

    impl Optimization for QueueTwoForceLikePairs {
        fn propagate_forward(
            &mut self,
            op: &Op,
            _op_rc: &majit_ir::OpRc,
            ctx: &mut OptContext,
        ) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntAdd {
                self.queued = true;

                ctx.seed_boxes_canonical(&[op.arg(0), op.arg(1)]);
                let alloc_a = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let alloc_a_box = ctx.materialize_operand_at(alloc_a);
                let value_a = op.arg(0);
                let mut set_a = Op::new(OpCode::SetfieldGc, &[alloc_a_box, value_a]);
                set_a.setdescr(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set_a);

                let alloc_b = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let alloc_b_box = ctx.materialize_operand_at(alloc_b);
                let value_b = op.arg(1);
                let mut set_b = Op::new(OpCode::SetfieldGc, &[alloc_b_box, value_b]);
                set_b.setdescr(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set_b);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_two_force_like_pairs"
        }
    }

    #[test]
    fn test_force_like_extra_ops_preserve_new_before_matching_setfield() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueTwoForceLikePairs {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(7)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let v0 = rooted_resop_operand(Type::Int, 0);
        let v1 = rooted_resop_operand(Type::Int, 1);
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[v0.clone(), v1.clone()]),
            Op::new(OpCode::Jump, &[v0, v1]),
        ];
        ops[0].pos().set(OpRef::int_op(2));
        ops[1].pos().set(OpRef::void_op(3));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        for set_op in result.iter().filter(|op| op.opcode == OpCode::SetfieldGc) {
            let alloc_ref = set_op.arg(0);
            let new_idx = result
                .iter()
                .position(|op| op.opcode == OpCode::New && op.pos().get() == alloc_ref.to_opref())
                .unwrap_or_else(|| panic!("missing New for {alloc_ref:?} in {result:?}"));
            let set_idx = result
                .iter()
                .position(|op| op.pos().get() == set_op.pos().get())
                .unwrap_or_else(|| {
                    panic!(
                        "missing setfield pos {:?} in {:?}",
                        set_op.pos().get(),
                        result
                    )
                });
            assert!(
                new_idx < set_idx,
                "matching New must appear before SetfieldGc; got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_remap_keeps_force_like_new_positions_out_of_constant_map() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(9)),
        }));
        opt.add_pass(Box::new(RemoveAsConstant {
            target: OpRef::int_op(2),
            value: 472,
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let v0 = rooted_resop_operand(Type::Int, 0);
        let v1 = rooted_resop_operand(Type::Int, 1);
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[v0.clone(), v1.clone()]),
            Op::new(OpCode::Jump, &[v0, v1]),
        ];
        ops[0].pos().set(OpRef::int_op(2));
        ops[1].pos().set(OpRef::void_op(3));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: indexmap::IndexSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos().get().raw())
            .collect();
        assert!(
            !new_positions.is_empty(),
            "expected force-like New op in optimized trace; got {:?}",
            result
        );
        for pos in new_positions.iter() {
            assert!(
                !constants.contains_key(pos),
                "live New position v{pos} must not collide with exported int constant map {:?}; trace {:?}",
                constants,
                result
            );
        }
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::SetfieldGc)
                .all(|op| new_positions.contains(&op.arg(0).to_opref().raw())),
            "SetfieldGc targets must remain emitted New refs; got {:?}",
            result
        );
    }

    #[test]
    #[ignore = "test fixture uses FieldDescr without parent SizeDescr, so ensure_ptr_info_arg0 panics before rd_numb encoding is exercised"]
    // Pre-existing test-setup bug (independent of rd_virtuals shape).
    // The fix needs three coupled changes that exceed an in-session port:
    //   (1) a parent-aware FieldDescr (this file's `TestDescr` fits) returning
    //       a parent_descr so virtualize.rs's `optimize_setfield_gc` doesn't
    //       panic;
    //   (2) a SizeDescr that exposes an `all_field_descrs` array containing
    //       that field, so virtualize.rs:`init_fields` can index into the
    //       VirtualInfo (mirrors virtualize.rs:`TestParentSizeDescr`); and
    //   (3) re-link the two with a Weak parent reference so init_fields can
    //       look up the field's offset back to the size descr.
    // virtualize.rs already has all three primitives in its private `tests`
    // module — the cleanest port is to lift TestParentSizeDescr +
    // TestParentFieldDescr into a shared `mod test_descrs;` and reuse here.
    fn test_optimizer_encodes_direct_virtual_guard_fail_args_as_rd_numb() {
        let mut opt = Optimizer::default_pipeline();
        let size_descr = make_size_descr(16);
        let field_descr = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);

        let mut guard = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 10)]);
        guard.setfailargs(vec![rooted_resop_operand(Type::Int, 0)].into());
        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 11),
                ],
                field_descr,
            ),
            guard,
            Op::new(OpCode::Jump, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos()
                .set(OpRef::op_typed(i as u32, op.opcode.result_type()));
        }

        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);
        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");

        // resume.py:411-417 parity: rd_numb + rd_virtuals from
        // ResumeDataVirtualAdder.finish().
        assert!(
            guard.resolved_rd_numb().is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );
        assert!(
            guard.resolved_rd_virtuals().is_some(),
            "virtual structure should be encoded into rd_virtuals tree"
        );
        let fail_args = guard
            .guard_fail_args()
            .expect("guard should keep fail args");
        // resume.py parity: liveboxes is TAGBOX-only.  The virtual
        // p0 is encoded into rd_virtuals; only its int field (OpRef::int_op(11))
        // survives in liveboxes.
        assert!(
            fail_args.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fail_args
        );
        assert!(
            fail_args.iter().any(|a| a.to_opref() == OpRef::int_op(11)),
            "virtual's field value (OpRef::int_op(11)) should appear in liveboxes; got {:?}",
            fail_args
        );
    }

    #[test]
    fn test_call_pure_results() {
        use majit_ir::Value;
        let mut opt = Optimizer::new();
        opt.record_call_pure_result(vec![Value::Int(10), Value::Int(20)], Value::Int(42));
        assert_eq!(
            opt.get_call_pure_result(&[Value::Int(10), Value::Int(20)]),
            Some(Value::Int(42))
        );
        assert_eq!(
            opt.get_call_pure_result(&[Value::Int(10), Value::Int(99)]),
            None
        );
    }

    #[test]
    fn test_phase_carry_holds_emit_only() {
        // 6: phase1_emit_ops is rebuilt at end-of-phase from
        // `ctx.new_operations` filtered by non-NONE pos and non-Void type
        // (resoperation.py:1693 parity). Phase 1 inputarg slot OpRefs are
        // resolved from Phase 2 through `OptContext::inputarg_type`
        // (history.py:220 parity) against the shared `inputarg_types`
        // Vec, so they are NOT carried here.
        let mut opt = Optimizer::new();
        opt.trace_inputargs = vec![
            OpRef::void_op(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_ref(2),
        ];
        opt.phase1_emit_ops.push(OpRc::new(majit_ir::Op::new(
            majit_ir::OpCode::SameAsI,
            &[rooted_resop_operand(Type::Int, 50)],
        )));

        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let result = opt
            .optimize_with_constants_and_inputs_at(&[], &mut constants, 3, 0, 0, false)
            .expect("empty trace must not produce InvalidLoop");

        assert!(result.is_empty());
        // Empty trace, no emitted ops — carry must be empty. Inputarg
        // slots are reachable through `inputarg_type`, not via the carry.
        assert!(opt.phase1_emit_ops.is_empty());
    }

    #[test]
    fn test_inputarg_type_resolves_phase1_slots_from_phase2_context() {
        // From a Phase-2-like context (`inputarg_base > 0`)
        // `OptContext::inputarg_type` must resolve low OpRefs in
        // `[0, num_inputs)` as Phase 1 inputarg slot lookups against the
        // shared `inputarg_types` Vec (history.py:220 parity for
        // `box.type` reads through `imported_label_args`).
        let mut ctx = OptContext::with_num_inputs_and_start_pos(8, 3, 100, 103);
        ctx.inputargs = OpRef::inputarg_refs(&[Type::Int, Type::Ref, Type::Float]);
        // Phase 2's own inputargs at [100..103) — own range still resolves.
        assert_eq!(ctx.inputarg_type(OpRef::int_op(100)), Some(Type::Int));
        assert_eq!(ctx.inputarg_type(OpRef::int_op(101)), Some(Type::Ref));
        assert_eq!(ctx.inputarg_type(OpRef::int_op(102)), Some(Type::Float));
        // Phase 1's inputarg slot OpRefs at [0..3) resolve through the
        // low-range fallback (Phase 2 inputarg_base=100 > 0).
        assert_eq!(ctx.inputarg_type(OpRef::int_op(0)), Some(Type::Int));
        assert_eq!(ctx.inputarg_type(OpRef::int_op(1)), Some(Type::Ref));
        assert_eq!(ctx.inputarg_type(OpRef::int_op(2)), Some(Type::Float));
        // Mid-range Phase 1 emit positions (`[num_inputs..inputarg_base)`)
        // are NOT inputargs — `inputarg_type` returns None and the
        // chain falls through to value_types / op.type_.
        assert_eq!(ctx.inputarg_type(OpRef::int_op(50)), None);
        assert_eq!(ctx.inputarg_type(OpRef::int_op(99)), None);
    }

    #[test]
    fn test_inputarg_type_phase1_no_fallback() {
        // In Phase 1 (`inputarg_base == 0`) the low-range fallback must
        // NOT trigger — only the canonical own-range path applies.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Ref, Type::Ref]);
        ctx.inputargs = OpRef::inputarg_refs(&[Type::Int, Type::Ref, Type::Float]);
        // OpRefs in [0..3) resolve through the own range (inputarg_base=0).
        assert_eq!(ctx.inputarg_type(OpRef::int_op(0)), Some(Type::Int));
        assert_eq!(ctx.inputarg_type(OpRef::int_op(2)), Some(Type::Float));
        // Out of range — None.
        assert_eq!(ctx.inputarg_type(OpRef::int_op(3)), None);
        assert_eq!(ctx.inputarg_type(OpRef::int_op(50)), None);
    }

    #[test]
    fn test_is_call_pure_pure_canraise_false_for_ops_that_are_not_call_pure() {
        // optimizer.py `is_call_pure_pure_canraise` opens with
        // `if not rop.is_call_pure(op.getopnum()): return False`
        // is the gate the whole guard-sharing chain rests on: `_emit_operation`
        // keeps `_last_guard_op` alive across side-effect-free operations only
        // while this answers False for them.  An ordinary op carries no call
        // descr, and answering True there retires the chain on every operation.
        for opcode in [
            OpCode::IntAdd,
            OpCode::GetfieldGcI,
            OpCode::ForceToken,
            OpCode::DebugMergePoint,
        ] {
            let op = Op::new(opcode, &[]);
            assert!(
                !Optimizer::is_call_pure_pure_canraise(&op),
                "{opcode:?} is not a CALL_PURE, so optimizer.py:733 returns False"
            );
        }
    }

    #[test]
    fn test_is_call_pure_pure_canraise_false_for_a_plain_call() {
        // A non-pure CALL is excluded by the same gate; it breaks the chain
        // through `has_no_side_effect` instead.
        let op = Op::new(OpCode::CallI, &[]);
        assert!(!Optimizer::is_call_pure_pure_canraise(&op));
    }

    #[test]
    fn test_is_call_pure_pure_canraise_ignores_memoryerror_only() {
        let mut op = Op::new(
            OpCode::CallPureI,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
        );
        op.setdescr(Arc::new(TestCallDescr {
            idx: 400,
            effect: EffectInfo::new(ExtraEffect::ElidableOrMemoryError, OopSpecIndex::None),
            result_type: majit_ir::Type::Int,
        }));
        assert!(
            !Optimizer::is_call_pure_pure_canraise(&op),
            "optimizer.py:705-711 ignores MemoryError-only effects"
        );
    }

    #[test]
    fn test_is_call_pure_pure_canraise_true_for_other_raising_effects() {
        let mut op = Op::new(
            OpCode::CallPureI,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
        );
        op.setdescr(Arc::new(TestCallDescr {
            idx: 401,
            effect: EffectInfo::new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None),
            result_type: majit_ir::Type::Int,
        }));
        assert!(Optimizer::is_call_pure_pure_canraise(&op));
    }

    #[test]
    fn test_protect_speculative_operation() {
        let opt = Optimizer::new();
        let ctx = OptContext::new(10);

        // Arithmetic ops are always safe
        let add_op = Op::new(
            OpCode::IntAdd,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
            ],
        );
        assert!(opt.protect_speculative_operation(&add_op, &ctx));

        // Getfield on unknown arg is safe (not constant null)
        let get_op = Op::new(OpCode::GetfieldGcI, &[rooted_resop_operand(Type::Int, 0)]);
        assert!(opt.protect_speculative_operation(&get_op, &ctx));
    }

    /// optimizer.py `getnullness(op)` parity test.
    ///
    /// Returns the upstream INFO_NULL / INFO_NONNULL / INFO_UNKNOWN
    /// integer constants (info.py).
    #[test]
    fn test_getnullness() {
        use crate::optimizeopt::{INFO_NONNULL, INFO_NULL, INFO_UNKNOWN};
        let mut ctx = OptContext::new(10);
        // Unknown integer → INFO_UNKNOWN.
        assert_eq!(
            Optimizer::getnullness(&mut ctx, OpRef::int_op(0)),
            INFO_UNKNOWN
        );
        // Known nonzero integer → INFO_NONNULL.
        let b = ctx.materialize_operand_at(OpRef::int_op(1));
        ctx.make_constant_box(&b, majit_ir::Value::Int(42));
        assert_eq!(
            Optimizer::getnullness(&mut ctx, OpRef::int_op(1)),
            INFO_NONNULL
        );
        // Known zero integer → INFO_NULL.
        let b = ctx.materialize_operand_at(OpRef::int_op(2));
        ctx.make_constant_box(&b, majit_ir::Value::Int(0));
        assert_eq!(
            Optimizer::getnullness(&mut ctx, OpRef::int_op(2)),
            INFO_NULL
        );
    }

    #[test]
    fn test_guard_replacement_flag() {
        let mut opt = Optimizer::new();
        let mut ctx = OptContext::new(0);
        assert!(opt.can_replace_guards);
        let guard = opt.cant_replace_guards(&mut ctx);
        assert!(!opt.can_replace_guards);
        assert!(!ctx.can_replace_guards);
        opt.restore_can_replace_guards(&mut ctx, guard);
        assert!(opt.can_replace_guards);
        assert!(ctx.can_replace_guards);
    }

    #[test]
    fn test_force_at_the_end_of_preamble_recurses_virtual_fields() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        // Generous Ref-typed inputarg pool — slot 10/11 are opaque opref handles
        // for the test, every slot uses Ref to match the producer-side shape
        // (`inputarg_from_tp` per opencoder.py:259 with all Ref args).
        let mut ctx = OptContext::with_inputarg_types(32, &vec![Type::Ref; 1024]);
        let b10 = ctx.materialize_operand_at(OpRef::int_op(10));
        let b11 = ctx.materialize_operand_at(OpRef::int_op(11));
        ctx.set_ptr_info(
            &b10,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, b11.clone())].into(),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
        let b20 = ctx.materialize_operand_at(OpRef::int_op(20));
        ctx.make_equal_to(&b11, &b20);
        let b20 = ctx.materialize_operand_at(OpRef::int_op(20));
        ctx.set_ptr_info(
            &b20,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: Default::default(),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let mut opt = Optimizer::new();
        let b10 = ctx.get_box_replacement_operand(OpRef::int_op(10));
        let result = opt.force_box_for_end_of_preamble(&b10, &mut ctx).to_opref();

        // The virtual is forced to a concrete allocation; the returned ref
        // is the allocation's position, which ctx.get_box_replacement(OpRef::int_op(10))
        // should resolve to.
        assert_eq!(result, ctx.get_replacement_opref(OpRef::int_op(10)));
        // After forcing, the struct's ptr_info reflects that field 1
        // (originally OpRef::int_op(11), forwarded to OpRef::int_op(20)) has been recursively forced.
        let result_box = ctx.get_box_replacement_operand_opt(result);
        match result_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            Some(PtrInfo::VirtualStruct(info)) => {
                // The inner virtual (OpRef::int_op(20)) was also forced; its allocation
                // ref is whatever force_box assigned.
                assert_eq!(info.fields.len(), 1);
                assert_eq!(info.fields[0].0, 1);
            }
            // After full forcing the info might become NonNull or similar
            Some(PtrInfo::NonNull { .. }) => {}
            other => panic!("expected virtual struct or non-null after forcing, got {other:?}"),
        }
    }

    #[test]
    fn test_emit_operation_materializes_virtual_args_directly() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let test_descr = std::sync::Arc::new(TestDescr(0));
        let field_descr_typed: std::sync::Arc<dyn majit_ir::FieldDescr> = test_descr;
        // Bake the field descriptor into the SizeDescr so the descr-derived
        // `all_fielddescrs_from_descr` view (info.rs) returns it for force_box.
        let descr: DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleSizeDescr::new(0, 16, 0)
                .with_all_fielddescrs(vec![field_descr_typed]),
        );
        let mut ctx = OptContext::with_inputarg_types(16, &[Type::Ref]);
        // The virtual is the result of its NEW, which force_box emits.
        let new_rc = OpRc::new(Op::with_descr(OpCode::New, &[], descr.clone()));
        new_rc.pos().set(OpRef::ref_op(10));
        ctx.register_extra_producer(&new_rc);
        let b10 = ctx.materialize_operand_at(OpRef::ref_op(10));
        // The field value box is materialized THROUGH the context so it binds
        // to the context's producer host; force_box's `resolve_operand_operand` then
        // resolves it to a bound box (sheds to Operand::Op), not a fresh
        // position-only `from_opref` box.
        let field_value = ctx.materialize_operand_at(OpRef::int_op(11));
        ctx.set_ptr_info(
            &b10,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(0, field_value)].into(),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let mut opt = Optimizer::new();
        let op = Op::new(
            OpCode::GuardNonnull,
            &[ctx.materialize_operand_at(OpRef::ref_op(10))],
        );
        let (mut seeded_ops, snapshots) =
            super::super::seed_empty_guard_snapshots(std::slice::from_ref(&op));
        ctx.snapshot_boxes = snapshots;
        let op = seeded_ops.pop().unwrap();
        let prior = OpRc::new(op.clone());
        let _ = opt.emit_operation(op, &prior, &mut ctx, false);

        assert!(!ctx.in_final_emission);

        // The whole emission, in order. The three membership checks this
        // replaces could see neither the count nor the ordering, and ordering
        // is the property under test: forcing the virtual must emit its New and
        // SetfieldGc BEFORE the guard that consumes it.
        //
        // info.py: force_box emits the ORIGINAL box op, so the forced
        // GuardNonnull keeps arg(0) = OpRef::ref_op(10) (the virtual's original
        // identity). force_box_impl preserves `new_op.pos = opref`, which is
        // why New lands at RefOp(10) rather than at a fresh position.
        let emitted: Vec<(OpCode, Vec<OpRef>, bool)> = ctx
            .new_operations
            .iter()
            .map(|op| {
                (
                    op.opcode,
                    (0..op.num_args()).map(|i| op.arg(i).to_opref()).collect(),
                    op.has_descr(),
                )
            })
            .collect();
        assert_eq!(
            emitted,
            [
                (OpCode::New, vec![], true),
                (
                    OpCode::SetfieldGc,
                    vec![OpRef::ref_op(10), OpRef::int_op(11)],
                    true
                ),
                (OpCode::GuardNonnull, vec![OpRef::ref_op(10)], true),
            ]
        );
    }

    #[test]
    fn test_emit_operation_forces_imported_short_guard_args() {
        let mut opt = Optimizer::new();
        let mut ctx = OptContext::with_inputarg_types(16, &[Type::Ref]);

        let mut preamble_op = Op::new(
            OpCode::IntGe,
            &[
                rooted_resop_operand(Type::Int, 3),
                rooted_resop_operand(Type::Int, 10_000),
            ],
        );
        preamble_op.pos().set(OpRef::int_op(14));
        let b = ctx.materialize_operand_at(OpRef::int_op(10_000));
        ctx.make_constant_box(&b, majit_ir::Value::Int(0));
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::int_op(0)],
            &[OpRef::int_op(0)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: OpRc::new(preamble_op.clone()),
                source_op: None,
                res: rooted_resop_operand(Type::Int, 14),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        let b14 = ctx.materialize_operand_at(OpRef::int_op(14));
        ctx.set_potential_extra_op(
            b14.clone(),
            crate::optimizeopt::info::PreambleOp {
                op: b14.clone(),
                invented_name: false,
                preamble_op: {
                    let mut op = majit_ir::Op::new(
                        majit_ir::OpCode::SameAsI,
                        &[rooted_resop_operand(Type::Int, 14)],
                    );
                    op.pos().set(OpRef::op_typed(14, op.result_type()));
                    OpRc::new(op)
                },
                same_as_source: None,
            },
        );

        let mut guard = Op::new(OpCode::GuardTrue, &[b14.clone()]);
        guard.pos().set(OpRef::op_typed(15, guard.result_type()));
        let (mut seeded_ops, snapshots) =
            super::super::seed_empty_guard_snapshots(std::slice::from_ref(&guard));
        ctx.snapshot_boxes = snapshots;
        let op = seeded_ops.pop().unwrap();
        let prior = OpRc::new(op.clone());
        let _ = opt.emit_operation(op, &prior, &mut ctx, false);

        let sp = ctx
            .build_imported_short_preamble()
            .expect("forcing imported short guard arg should build short preamble");
        assert_eq!(sp.used_boxes, vec![OpRef::int_op(14)]);
        assert_eq!(sp.jump_args, vec![OpRef::int_op(14)]);
    }

    #[test]
    fn test_resumedata_memo_encodes_rd_numb_on_guard() {
        use crate::history::test_support::TraceBuilder;
        let mut opt = Optimizer::default_pipeline();
        // OptIntBound (`mod.rs`'s `getintbound_handle`) requires IntAdd's args to be
        // Type::Int — the two header inputs are Int.
        let mut b = TraceBuilder::new();
        let x = b.input(Type::Int, 0);
        let y = b.input(Type::Int, 1);
        let x_ref = x.to_opref();
        let y_ref = y.to_opref();
        b.op(OpCode::IntAdd, &[x.clone(), y.clone()]);
        b.op(OpCode::GuardTrue, std::slice::from_ref(&x));
        b.op(OpCode::Finish, &[]);
        let (ops, inputs) = b.build();
        // The GuardTrue (ops[1]) keeps both header inputs as fail args; bind
        // them to the same canonical InputArg boxes the header threads.
        ops[1].setfailargs(vec![x.clone(), y.clone()].into());
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        let num_inputs = inputs.len();
        opt.snapshot_boxes = seed_guard_snapshots_with_oprc(&ops, |_| vec![x_ref, y_ref]);
        let result = opt
            .optimize_with_constants_and_inputs_oprc(
                &ops,
                &mut majit_ir::ConstMap::default(),
                num_inputs,
            )
            .expect("test: unexpected InvalidLoop");

        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");
        assert!(
            guard.resolved_rd_numb().is_some(),
            "resumedata_memo should set rd_numb on guard"
        );
        assert!(
            guard.resolved_rd_consts().is_some(),
            "resumedata_memo should set rd_consts on guard"
        );
    }

    #[test]
    fn update_counters_folds_opt_ops_opt_guards_opt_guards_shared_into_profiler() {
        // optimizer.py/629/673-674: every emit_operation bumps OPT_OPS,
        // each guard bumps OPT_GUARDS additionally, and the sharing path
        // bumps OPT_GUARDS_SHARED.  Pyre defers the fold via
        // update_counters; assert the accumulators land in the matching
        // `JitProfiler` atomics and reset after the fold so a second
        // call doesn't double-count.
        let mut opt = Optimizer::new();
        opt.opt_ops_emitted = 5;
        opt.opt_guards_emitted = 2;
        opt.opt_guards_shared_emitted = 1;
        let prof = crate::jitprof::JitProfiler::default();
        opt.update_counters(&prof);
        let snap = prof.snapshot();
        assert_eq!(snap.opt_ops, 5);
        assert_eq!(snap.opt_guards, 2);
        assert_eq!(snap.opt_guards_shared, 1);
        // Accumulators reset so a second update doesn't re-add.
        assert_eq!(opt.opt_ops_emitted, 0);
        assert_eq!(opt.opt_guards_emitted, 0);
        assert_eq!(opt.opt_guards_shared_emitted, 0);
        opt.update_counters(&prof);
        let snap = prof.snapshot();
        assert_eq!(snap.opt_ops, 5);
        assert_eq!(snap.opt_guards, 2);
        assert_eq!(snap.opt_guards_shared, 1);
    }

    /// bind-at-alloc totality: an `Unknown` virtual-state leaf
    /// must bind its canonical `_forwarded` host at allocation. A bare
    /// position resolves to `None`, so every later resolution mints a fresh
    /// position-only box — splitting `same_box`-keyed export-cache keys and
    /// missing the `in_progress` cycle guard (the #228 residual).
    #[test]
    fn import_unknown_leaf_resolves_to_bound_box() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(8, 0, 0, 50);
        let info = crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(Type::Ref);
        let box_ = Optimizer::import_virtual_state_value(&info, &mut ctx);
        assert!(
            ctx.get_box_replacement_operand_opt(box_.to_opref())
                .is_some(),
            "Unknown-leaf import must bind the canonical host at allocation"
        );
    }

    /// Preamble `int_ge(a, b)`, `guard_true`, `guard_value(a, C)` proves
    /// `int_ge(C, b)`. `RecentPureOps.lookup2` compares
    /// `get_box_replacement` of the cached args at lookup time, and
    /// `OptPure.produce_potential_short_preamble_ops` exports that pure op
    /// with the replaced args, so the peeled body's `int_ge(C, b)` /
    /// `guard_true` are removed (`optimize_GUARD_TRUE`).
    #[test]
    fn peeled_loop_drops_int_ge_guard_after_guard_value_const() {
        use crate::history::test_support::rooted_inputarg_operand;
        use crate::optimizeopt::unroll::UnrollOptimizer;

        let a = rooted_inputarg_operand(Type::Int, 0);
        let b_arg = rooted_inputarg_operand(Type::Int, 1);
        let four = Operand::const_from_value(majit_ir::Value::Int(4));
        let mut cmp = Op::new(OpCode::IntGe, &[a.clone(), b_arg.clone()]);
        cmp.pos().set(OpRef::int_op(2));
        let mut guard_true = Op::with_descr(
            OpCode::GuardTrue,
            &[rooted_resop_operand(Type::Int, 2)],
            crate::compile::make_resume_guard_descr_typed(Vec::new()),
        );
        guard_true.pos().set(OpRef::void_op(3));
        let mut guard_value = Op::with_descr(
            OpCode::GuardValue,
            &[a.clone(), four],
            crate::compile::make_resume_guard_descr_typed(Vec::new()),
        );
        guard_value.pos().set(OpRef::void_op(4));
        let mut jump = Op::new(OpCode::Jump, &[a, b_arg]);
        jump.pos().set(OpRef::void_op(5));
        let ops = vec![cmp, guard_true, guard_value, jump];
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
        // `shortpreamble.py` `PureOp.add_op_to_short` calls `produce_arg`
        // on the op's own args. `guard_value` runs after `int_ge` was
        // emitted, and `produce_arg` does not walk `get_box_replacement`,
        // so the loop header still re-executes `int_ge(Const(4), b)` and
        // its `guard_true`.
        let body: Vec<_> = result[label_at + 1..].iter().map(|op| op.opcode).collect();
        assert_eq!(
            body,
            vec![OpCode::IntGe, OpCode::GuardTrue, OpCode::Jump],
            "full={:?}",
            result
                .iter()
                .map(|op| {
                    let args: Vec<_> = op.getarglist().iter().map(|a| a.to_opref()).collect();
                    format!("{:?} {:?}", op.opcode, args)
                })
                .collect::<Vec<_>>()
        );
        assert!(
            result[label_at + 1].arg(0).is_constant(),
            "int_ge's first arg is the guard_value constant"
        );
    }

    /// Imported virtual heads must carry their `PtrInfo::Virtual`. The
    /// label-args import used to allocate a bare head position and write
    /// the info through `get_box_replacement_operand_opt(..)` guarded by `if let
    /// Some(..)` — always `None` for a bare position, silently dropping
    /// the virtual-ness of the imported state.
    #[test]
    fn import_label_args_virtual_head_installs_ptr_info() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(8, 0, 0, 50);
        let info = crate::optimizeopt::virtualstate::VirtualStateInfo::Virtual {
            descr: make_size_descr(16),
            known_class: None,
            ob_type_descr: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
        };
        let mut walk_visited = crate::FxIndexMap::default();
        let mut label_slot = 0usize;
        let head = Optimizer::import_virtual_state_from_label_args(
            &info,
            &[],
            &mut label_slot,
            &mut ctx,
            &mut walk_visited,
        );
        let b = ctx
            .get_box_replacement_operand_opt(head.to_opref())
            .expect("virtual head must bind the canonical host at allocation");
        assert!(
            matches!(
                ctx.peek_ptr_info(&b),
                Some(crate::optimizeopt::info::PtrInfo::Virtual(_))
            ),
            "imported virtual head must carry PtrInfo::Virtual"
        );
    }
}
