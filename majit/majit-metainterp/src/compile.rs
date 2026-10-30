//! Trace compilation helpers.
//!
//! Mirrors RPython's `compile.py`: guard metadata building, exit layout
//! management, backend layout merging, trace post-processing (unboxing),
//! and the ResumeGuard-descriptor class hierarchy
//! (`compile.py AbstractResumeGuardDescr` →
//! `ResumeGuardDescr` → `{ResumeAtPositionDescr, ResumeGuardForcedDescr,
//! ResumeGuardExcDescr, CompileLoopVersionDescr}`,
//! `ResumeGuardCopiedDescr` / `ResumeGuardCopiedExcDescr`,
//! `invent_fail_descr_for_op`).  Pyre wraps upstream's direct attribute
//! assignments in `UnsafeCell` so the optimizer can mutate the descr
//! through `Arc<dyn Descr>` shared ownership; the only structural
//! difference vs upstream is the use of cells, not a separate module.

use indexmap::{IndexMap, IndexSet};
use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use smallvec::smallvec;

use majit_backend::{
    Backend, BackendError, CompiledLoopToken, CompiledTraceInfo, DeadFrame, FailDescrLayout,
    JitCellToken,
};
use majit_ir::forwarding::ForwardingHost;
use majit_ir::operand::Operand;
use majit_ir::{
    AccumInfo, Const, DescrRef, FailDescr, GcRef, GreenKey, GuardPendingFieldEntry,
    HandleFailClass, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, RdVirtualInfo, Type, Value,
};

use crate::blackhole::ExceptionState;
use crate::history::TreeLoop;
use crate::pyjitpl::StoredExitLayout;
use crate::resume::ResumeStorage;
use crate::trace_ctx::{MergePoint, TraceCtx};

/// `compile.py` `make_jitcell_token(jitdriver_sd)`.
///
/// ```python
/// def make_jitcell_token(jitdriver_sd):
///     jitcell_token = JitCellToken()
///     jitcell_token.outermost_jitdriver_sd = jitdriver_sd
///     return jitcell_token
/// ```
///
/// The token is created unnumbered (`history.py JitCellToken.number = -1`).
/// `compile.py send_loop_to_backend` assigns
/// `metainterp_sd.jitlog.trace_id` immediately before `cpu.compile_loop`.
/// `compile_tmp_callback` leaves the number unassigned.
///
/// The `outermost_jitdriver_index: Option<usize>` is the pyre analog of
/// RPython's `outermost_jitdriver_sd` attribute (pyre stores a slot
/// index into `metainterp_sd.jitdrivers_sd` instead of a direct
/// reference, since the descriptor lives in another crate).
///
/// Returns the canonical `Arc<JitCellToken>` immediately.  Callers can
/// mutate the token before it is cloned with `Arc::get_mut`, then pass
/// the same Arc through `compile_loop` (`compile.py`) →
/// `attach_procedure_to_interp` (`compile.py`) →
/// `MemoryManager.keep_loop_alive`.
pub fn make_jitcell_token(jd_index: Option<usize>) -> Arc<JitCellToken> {
    let mut token = JitCellToken::new(JitCellToken::NUMBER_UNASSIGNED);
    token.outermost_jitdriver_index = jd_index;
    Arc::new(token)
}

/// `compile.py:180-181` `wref = weakref.ref(original_jitcell_token);
/// clt.loop_token_wref = wref` parity.  The token's late-written fields
/// (`green_key`, `inputarg_types`, `compiled`, `compiled_loop_token`) are
/// interior-mutable, so `configure_loop_token_for_driver` and the backend's
/// `compile_loop` operate through `&JitCellToken`; call this once those
/// writes have settled, before the token is published into `compiled_loops`
/// / `attach_procedure_with_redirect`.
pub fn wire_clt_loop_token_wref(token: &Arc<JitCellToken>) {
    if let Some(clt) = token.compiled_loop_token() {
        clt.set_loop_token_wref(Arc::downgrade(token));
    }
}

/// Resolve the type of an OpRef in guard fail_args.
/// OpRef::NONE is a virtual slot placeholder (null GC ref).
/// history.py Box.type is intrinsic on the Box itself
/// (`Const{Int,Float,Ptr}.type`, `InputArg{Int,Float,Ref}.type`, and the
/// `{Int,Float,Ref}Op` mixins), read off the OpRef variant tag
/// (`OpRef::ty()` in `resoperation.rs`). A fail_arg carries its own type regardless of
/// trace position, so no position-keyed side table is needed.
#[cfg(test)]
fn fail_arg_type(opref: &OpRef) -> Type {
    if *opref == OpRef::NONE {
        return Type::Ref;
    }
    opref.ty().unwrap_or(Type::Ref)
}

/// Storage for one exit's type list.
///
/// Inline up to a width that costs nothing. `Type` is one byte, so sixteen of
/// them occupy exactly the bytes a heap vector spends on its pointer and
/// capacity anyway: both spellings are 24 bytes. What the inline storage buys
/// is that the steady exit — a FINISH returning one word, or a guard with a
/// handful of fail arguments — carries its types without asking the allocator,
/// on a path taken once per entry into compiled code. A wider exit spills to
/// the heap and pays exactly what it paid before.
pub type ExitTypes = smallvec::SmallVec<[Type; 16]>;

/// Inline width of the two decoded-exit buffers below.
///
/// Chosen from the exit arities a compiled entry actually returns, not from
/// the widest one possible. Two shapes account for the steady path: a FINISH
/// handing back the traced function's single result, and a guard handing back
/// a frame's worth of live values — a handful, not a frame's full extent,
/// because the exit carries only what the guard's fail arguments name. Eight
/// covers both.
///
/// The width is bounded from above by the exits that do *not* fit. They exist,
/// but they sit far above eight rather than just past it, so nothing is gained
/// by inching this number up: the next size that would capture them costs more
/// per entry, on every entry, than the allocations it saves on the few.
const EXIT_VALUE_INLINE: usize = 8;

/// Storage for one exit's decoded values, as machine words.
///
/// Written by the raw run path, and by the guard-failure and detailed-run arms
/// that ask [`raw_exit_values`] for one. Holding it inline keeps those from
/// asking the allocator for a buffer they free before the next entry; an exit
/// wider than [`EXIT_VALUE_INLINE`] spills to the heap and pays exactly what it
/// paid before.
pub type ExitRawValues = smallvec::SmallVec<[i64; EXIT_VALUE_INLINE]>;

/// Storage for one exit's decoded values, typed — see [`ExitRawValues`].
///
/// A `Value` is two words, so this array is the more expensive half of the pair
/// and is what bounds [`EXIT_VALUE_INLINE`] from above. It is also the only
/// half a [`CompileResult`] carries, because the other is a projection of it.
pub type ExitValues = smallvec::SmallVec<[Value; EXIT_VALUE_INLINE]>;

/// Project decoded exit slots back to the machine words they were read as.
///
/// [`Value::as_raw_i64`] is exact for every slot type a fail-arg list can hold,
/// so this recovers what a second list built beside [`ExitValues`] would have
/// held. Its callers are the guard-failure and detailed-run arms, which then
/// own the buffer; the steady finish exit calls nothing here.
///
/// Upstream has no list of this shape to be faithful to: a final descriptor
/// reads the single slot its own type calls for straight off the deadframe
/// (`DoneWithThisFrameDescrInt` / `Ref` / `Float`, each with its own
/// `get_result`), so a machine-word vector covering every slot is a Rust-side
/// convenience. It therefore belongs with the two readers that want one, not on
/// the result every entry returns.
pub fn raw_exit_values(typed: &[Value]) -> ExitRawValues {
    typed.iter().map(Value::as_raw_i64).collect()
}

/// Static exit metadata for a compiled guard or finish point.
#[derive(Debug, Clone)]
pub struct CompiledExitLayout {
    /// compile.py:186 rd_loop_token: the green_key of the compiled loop
    /// that owns this guard. Used by handle_fail to find the owning
    /// compiled entry without scanning all entries.
    pub rd_loop_token: u64,
    pub trace_id: u64,
    pub fail_index: u32,
    pub source_op_index: Option<usize>,
    pub exit_types: ExitTypes,
    pub is_finish: bool,
    /// `compile.py ExitFrameWithExceptionDescrRef`
    /// vs `compile.py DoneWithThisFrameDescrRef`: distinguishes
    /// the exception-propagation FINISH so the synthesis fallback at
    /// `make_finish_fail_descr_typed` routes a `[Type::Ref]` exit to the
    /// correct `_DoneWithThisFrameDescr` subclass.
    pub is_exception_exit: bool,
    /// compile.py `ResumeGuardDescr` storage handle — shared
    /// pool with rd_numb / rd_consts / rd_virtuals / rd_pendingfields.
    pub storage: Option<std::sync::Arc<crate::resume::ResumeStorage>>,
}

/// `compile.py AbstractResumeGuardDescr.handle_fail`: the failing exit is
/// described by the descr the deadframe named and by nothing else —
/// `ResumeGuardDescr.get_resumestorage()` for the `rd_*` payload,
/// `fail_arg_types` for the slot types, `rd_loop_token` for the owning
/// loop. No frontend record is consulted: `send_bridge_to_backend` keeps
/// none for a bridge, and a loop's guard answers the same way.
///
/// `storage` is `None` for a descr with no resume payload (the
/// `_DoneWithThisFrameDescr` family and `ExitFrameWithExceptionDescrRef`).
pub fn exit_layout_for_descr(descr_arc: &majit_ir::DescrRef) -> CompiledExitLayout {
    let descr = descr_arc
        .as_fail_descr()
        .expect("a compiled exit's descr always implements FailDescr");
    CompiledExitLayout {
        rd_loop_token: majit_backend::descr_owning_green_key(descr).unwrap_or(0),
        trace_id: descr.trace_id(),
        fail_index: descr.fail_index_per_trace(),
        source_op_index: descr.source_op_index(),
        exit_types: ExitTypes::from_slice(descr.fail_arg_types()),
        is_finish: descr.is_finish(),
        is_exception_exit: descr.is_exit_frame_with_exception(),
        storage: crate::resume::get_resumestorage(descr_arc),
    }
}

/// The pc the outermost frame of `descr`'s resume data resumes at.
/// `resume.py ResumeDataVirtualAdder.number` writes each frame as
/// `jitcode_index, pc, values...` after the vable and vref sections, and
/// `rebuild_from_resumedata` / `blackhole_from_resumedata` set every
/// rebuilt frame at its pc. `None` for a descr with no resume payload or
/// no frame section.
pub fn guard_resume_pc(descr: &dyn majit_ir::FailDescr) -> Option<u64> {
    let rd_numb = descr.rd_numb_arc()?;
    crate::resume::outermost_frame_pc(rd_numb.as_ref()).map(|pc| pc as u64)
}

impl CompiledExitLayout {
    /// Whether the collector traces exit slot `slot`.
    ///
    /// The rule `llsupport/assembler.py compute_gcmap` applies: it skips the
    /// hole a virtual leaves (`if arg is None: continue`), marks every
    /// remaining `REF`-typed failarg, and narrows nothing else.  A force
    /// token is marked like any other, because `resoperation.py
    /// FORCE_TOKEN/0/r` is REF upstream as well — it returns the jitframe,
    /// itself a GC object that moves — and both emitted gcmaps
    /// (`guard_gcmap_from_faillocs`, `collect_guards`) mark it.
    ///
    /// The skip has nothing to do here: an exit layout reaches this point
    /// with every fail arg bound, so the type test is the whole rule.
    ///
    /// Do not narrow it by force-token position: that would stop rooting a
    /// live jitframe pointer.
    pub fn is_traced_ref_slot(&self, slot: usize) -> bool {
        self.exit_types.get(slot) == Some(&Type::Ref)
    }
}

/// Typed result from running compiled code.
pub struct CompileResult<M> {
    /// The exit slots this run left through.
    ///
    /// Populated on a FINISH and on a back-edge JUMP (`fail_index ==
    /// u32::MAX`). A guard-failure leaves this empty: `compile.py
    /// ResumeGuardDescr.handle_fail` does not copy the deadframe out, and
    /// `resume.py ResumeDataDirectReader.decode_int` reads named slots via
    /// `self.cpu.get_int_value(self.deadframe, num)`.
    pub typed_values: ExitValues,
    /// Optional frontend snapshot taken *before* `execute_token`. Warm entry
    /// and restore rebuild from live state (`JitState::build_meta`); this
    /// field is `None` on that path. A re-lookup after the run would read
    /// whatever the compiled-loop index holds by then, and compiled code
    /// re-enters the driver through residual calls, so the entry can be
    /// recompiled or dropped mid-run.
    ///
    /// The hold is a shared reference, not a copy: upstream keeps the
    /// `loop_token` object itself alive across `execute_assembler`, and an
    /// `Arc` gives the same "this exact metadata, for the whole run" guarantee
    /// without paying a struct copy per entry.
    ///
    /// Carried only for the caller that asked for it. The detailed runs read
    /// it back off the result; the warm entry in `jitdriver` already holds
    /// the same snapshot across the run and never reads this field, so it
    /// hands in `None` and pays no refcount pair for the round trip.
    pub meta: Option<std::sync::Arc<M>>,
    pub fail_index: u32,
    pub trace_id: u64,
    /// `cpu.get_latest_descr(deadframe)` (`history.py:125`) — the
    /// backend-side guard descr recovered from the failing frame.  Pyre
    /// currently has split metainterp/backend descr objects, so this is
    /// the runtime descr carrying the same `rd_loop_token_clt` /
    /// `fail_index_per_trace` identity used for bridge routing.
    ///
    /// Carried for a guard exit only. Every reader of it is a bridge or
    /// blackhole consumer, and a final descr resumes nothing, so the finish
    /// arm reads the descr off the frame by reference and never takes the
    /// shared hold on it.
    pub descr_arc: Option<std::sync::Arc<dyn majit_ir::Descr>>,
    pub is_finish: bool,
    /// compile.py ExitFrameWithExceptionDescrRef parity:
    /// true when the FINISH descriptor was
    /// `sd.exit_frame_with_exception_descr_ref` (emitted via
    /// `pyjitpl.py compile_exit_frame_with_exception`).
    /// jitdriver routes this to `jitexc.ExitFrameWithExceptionRef`.
    pub is_exit_frame_with_exception: bool,
    /// The failing guard's exit description, and `None` on a FINISH exit.
    ///
    /// Held behind a pointer because this result is returned BY VALUE from
    /// every compiled entry, so an inline layout is bytes moved on each one.
    /// Every reader (`jitdriver`'s two run loops and pyre's drain loop) opens
    /// the layout only past its own FINISH and `fail_index == u32::MAX` arms,
    /// so a call that returns through either was copying a layout nothing read.
    ///
    /// The allocation lands where the layout is built, which the FINISH arm now
    /// returns before reaching. A FINISH exit stores one null word here, so the
    /// steady entry path's allocation count is unchanged.
    pub exit_layout: Option<Box<CompiledExitLayout>>,
    /// `compile.py handle_fail` reads `resumedescr.rd_loop_token` once and
    /// hands the loop it names to everything downstream. The green key of the
    /// `JitCellToken` the failing descr belongs to, recovered by walking
    /// `descr.rd_loop_token_clt() -> clt.upgrade_loop_token()`.
    ///
    /// `None` when the walk finds no live owner (memmgr eviction) and on a
    /// finish or back-edge JUMP exit, neither of which has a guard identity to
    /// recover: the layout fallback and `must_compile` are the only readers,
    /// and neither exit reaches either. Readers fall back to the key of the
    /// entry they came in through.
    pub rd_loop_token: Option<u64>,
    pub savedata: Option<GcRef>,
    /// `llmodel.py` deadframe from `cpu.force` / `execute_token`.
    /// Kept until `AllVirtuals.show` so `jf_savedata` stays rooted the
    /// way upstream traces it on the live jitframe. `None` on FINISH
    /// and JUMP exits that never hide a cache.
    pub deadframe: Option<DeadFrame>,
    pub exception: ExceptionState,
    /// compile.py: ResumeGuardDescr.status read at guard failure.
    pub status: u64,
    /// `compile.py must_compile` reads the failing GUARD_VALUE operand off the
    /// deadframe with `cpu.get_value_direct(deadframe, tp, index)`. The
    /// deadframe's live range ends inside the run, so the read happens where
    /// `status` is read and the word travels out on this result, keeping it
    /// per-failure the way upstream's deadframe read is.
    pub guard_value_operand: Option<i64>,
}

/// Raw (lightweight) result from running compiled code.
pub struct RawCompileResult<M> {
    pub values: ExitRawValues,
    pub typed_values: ExitValues,
    /// Pre-run metadata snapshot — see `CompileResult::meta`.
    /// `None` when the run entered on the procedure token without a
    /// `compiled_loops` row; restore rebuilds from live state.
    pub meta: Option<std::sync::Arc<M>>,
    pub fail_index: u32,
    pub trace_id: u64,
    /// `cpu.get_latest_descr(deadframe)` (`history.py:125`,
    /// `compile.py:701`) — runtime descr Arc owning this exit.  Always
    /// set: routed through `Backend::get_latest_descr_arc` via
    /// `RawExecResult::descr_arc`, so FINISH / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` singletons return their global
    /// Arc identity instead of `None`.  Bridge consumers
    /// (`start_bridge_tracing`, `_trace_and_compile_from_bridge`) read
    /// `rd_loop_token_clt` / `fail_index_per_trace` directly from this.
    pub descr_arc: std::sync::Arc<dyn majit_ir::Descr>,
    pub is_finish: bool,
    /// compile.py ExitFrameWithExceptionDescrRef parity —
    /// mirrors `CompileResult::is_exit_frame_with_exception`.
    pub is_exit_frame_with_exception: bool,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
    /// compile.py: ResumeGuardDescr.status read at guard failure.
    pub status: u64,
    /// Failing GUARD_VALUE operand read while the raw frame is live.
    pub guard_value_operand: Option<i64>,
}

/// Terminal exit layout for a FINISH or JUMP op.
#[derive(Debug, Clone)]
pub struct CompiledTerminalExitLayout {
    pub op_index: usize,
    pub exit_layout: CompiledExitLayout,
}

/// Full trace compilation layout with all exits.
#[derive(Debug, Clone)]
pub struct CompiledTraceLayout {
    pub trace_id: u64,
    pub exit_layouts: Vec<CompiledExitLayout>,
    pub terminal_exit_layouts: Vec<CompiledTerminalExitLayout>,
}

/// Artifacts extracted from a backend DeadFrame.
#[derive(Debug, Clone)]
pub struct DeadFrameArtifacts {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
}

/// `compile.py` `class CompileData(object)`.
///
/// `optimize_trace` is the compile.py method: write `MARK_TRACE`, run
/// `logger_noopt` when `log_noopt` is set, run subclass `optimize()`, then
/// `forget_optimization_info`. `build_opt_chain` stays at the flattened call
/// site in `pyjitpl.rs` because the optimizer borrows `MetaInterp`, backend
/// state, constant pools, and snapshot side tables. Call sites still pass the
/// same trace/runtime/resume/call-pure/opts state that RPython would store on
/// the corresponding object.
pub struct CompileData<'a> {
    pub trace: &'a TreeLoop,
    /// compile.py `CompileData.log_noopt`; `UnrolledLoopData` clears it.
    pub log_noopt: bool,
}

impl<'a> CompileData<'a> {
    pub fn new(trace: &'a TreeLoop) -> Self {
        Self {
            trace,
            log_noopt: true,
        }
    }

    /// compile.py `CompileData.forget_optimization_info`:
    /// `for arg in self.trace.inputargs: arg.set_forwarded(None)`.
    pub fn forget_optimization_info(&self) {
        for arg in self.inputargs() {
            arg.clear_forwarded();
        }
    }

    /// compile.py `CompileData.optimize_trace`: `log_trace(MARK_TRACE)`,
    /// `logger_noopt.log_loop_from_trace` when `log_noopt` is set, run the
    /// subclass `optimize()` body, then `forget_optimization_info`.
    ///
    /// `constants` is handed through to `optimize` so the noopt dump can read
    /// it before the optimizer takes it mutably.
    pub fn optimize_trace<V: std::fmt::Debug, C: majit_ir::resoperation::ConstLookup<V>, T, E>(
        &self,
        tid: u64,
        constants: &mut C,
        optimize: impl FnOnce(&mut C) -> Result<T, E>,
    ) -> Result<T, E> {
        crate::rjitlog::write_trace(
            crate::rjitlog::MARK_TRACE,
            tid,
            self.inputargs(),
            self.operations(),
        );
        if self.log_noopt {
            crate::logger::log_loop_from_trace(self.operations(), &*constants);
        }
        let result = optimize(constants);
        self.forget_optimization_info();
        result
    }

    pub fn inputargs(&self) -> &'a [majit_ir::InputArgRc] {
        &self.trace.inputargs
    }

    pub fn operations(&self) -> &'a [majit_ir::OpRc] {
        &self.trace.ops
    }

    pub fn snapshots(&self) -> &'a [crate::recorder::Snapshot] {
        &self.trace.snapshots
    }
}

/// `compile.py` `class PreambleCompileData(CompileData)`.
pub struct PreambleCompileData<'a> {
    pub base: CompileData<'a>,
    #[allow(dead_code)]
    pub runtime_boxes: &'a [OpRef],
    pub call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
    #[allow(dead_code)]
    pub enable_opts: &'a [String],
}

impl<'a> PreambleCompileData<'a> {
    pub fn new(
        trace: &'a TreeLoop,
        runtime_boxes: &'a [OpRef],
        call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
        enable_opts: &'a [String],
    ) -> Self {
        Self {
            base: CompileData::new(trace),
            runtime_boxes,
            call_pure_results,
            enable_opts,
        }
    }
}

/// `compile.py` `class SimpleCompileData(CompileData)`.
pub struct SimpleCompileData<'a> {
    pub base: CompileData<'a>,
    #[allow(dead_code)]
    pub resumestorage: Option<&'a ResumeStorage>,
    pub call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
    #[allow(dead_code)]
    pub enable_opts: &'a [String],
}

impl<'a> SimpleCompileData<'a> {
    pub fn new(
        trace: &'a TreeLoop,
        resumestorage: Option<&'a ResumeStorage>,
        call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
        enable_opts: &'a [String],
    ) -> Self {
        Self {
            base: CompileData::new(trace),
            resumestorage,
            call_pure_results,
            enable_opts,
        }
    }

    /// compile.py `CompileData.optimize_trace` + `SimpleCompileData.optimize`.
    pub fn optimize_trace<V: std::fmt::Debug, C: majit_ir::resoperation::ConstLookup<V>, T, E>(
        &self,
        tid: u64,
        constants: &mut C,
        optimize: impl FnOnce(&Self, &mut C) -> Result<T, E>,
    ) -> Result<T, E> {
        self.base
            .optimize_trace(tid, constants, |constants| optimize(self, constants))
    }
}

/// `compile.py` `class BridgeCompileData(CompileData)`.
pub struct BridgeCompileData<'a> {
    #[allow(dead_code)]
    pub base: CompileData<'a>,
    pub runtime_boxes: &'a [OpRef],
    #[allow(dead_code)]
    pub resumestorage: Option<&'a ResumeStorage>,
    #[allow(dead_code)]
    pub call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
    #[allow(dead_code)]
    pub inline_short_preamble: bool,
    #[allow(dead_code)]
    pub enable_opts: &'a [String],
}

impl<'a> BridgeCompileData<'a> {
    pub fn new(
        trace: &'a TreeLoop,
        runtime_boxes: &'a [OpRef],
        resumestorage: Option<&'a ResumeStorage>,
        call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
        inline_short_preamble: bool,
        enable_opts: &'a [String],
    ) -> Self {
        Self {
            base: CompileData::new(trace),
            runtime_boxes,
            resumestorage,
            call_pure_results,
            inline_short_preamble,
            enable_opts,
        }
    }

    /// compile.py `CompileData.optimize_trace` + `BridgeCompileData.optimize`.
    pub fn optimize_trace<V: std::fmt::Debug, C: majit_ir::resoperation::ConstLookup<V>, T, E>(
        &self,
        tid: u64,
        constants: &mut C,
        optimize: impl FnOnce(&Self, &mut C) -> Result<T, E>,
    ) -> Result<T, E> {
        self.base
            .optimize_trace(tid, constants, |constants| optimize(self, constants))
    }
}

/// `compile.py` `class UnrolledLoopData(CompileData)`.
#[allow(dead_code)]
pub struct UnrolledLoopData<'a> {
    pub base: CompileData<'a>,
    #[allow(dead_code)]
    pub celltoken: &'a Arc<JitCellToken>,
    #[allow(dead_code)]
    pub state: &'a crate::optimizeopt::unroll::ExportedState,
    #[allow(dead_code)]
    pub call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
    #[allow(dead_code)]
    pub enable_opts: &'a [String],
}

#[allow(dead_code)]
impl<'a> UnrolledLoopData<'a> {
    pub fn new(
        trace: &'a TreeLoop,
        celltoken: &'a Arc<JitCellToken>,
        state: &'a crate::optimizeopt::unroll::ExportedState,
        call_pure_results: &'a crate::optimizeopt::util::ArgsDict,
        enable_opts: &'a [String],
    ) -> Self {
        Self {
            base: CompileData {
                log_noopt: false,
                ..CompileData::new(trace)
            },
            celltoken,
            state,
            call_pure_results,
            enable_opts,
        }
    }
}

/// Pyre-only: stamp the per-trace `fail_index` and the trace-op index onto
/// every guard's metainterp `ResumeGuardDescr`, so `(trace_id, fail_index)`
/// readers resolve through the descr Arc directly.  The counter walks guards
/// and FINISH in trace order — the same numbering the assemblers'
/// `fail_descrs` positions use.  Readers compare
/// against `fail_index_per_trace()` (this slot), not `fail_index()` (the
/// global `alloc_fail_index()` id).
///
/// `ops` is the optimized frontend trace.  The backend GC rewriter inserts
/// operations before code generation, so the assembler's prepared-op index
/// is not an index into this slice; the descr is stamped from the frontend
/// op, matching RPython where the `ResumeGuardDescr` stays attached to the
/// live ResOperation rather than carrying an index into a rewritten array.
///
/// Non-resume FailDescrs are skipped: their `set_fail_index_per_trace`
/// panics by default.
pub(crate) fn stamp_guard_descr_trace_positions<T: AsRef<majit_ir::Op>>(ops: &[T]) {
    let mut fail_index = 0u32;
    for (op_idx, op) in ops.iter().enumerate() {
        let op = op.as_ref();
        let is_guard = op.opcode.is_guard();
        if !is_guard && op.opcode != OpCode::Finish {
            continue;
        }
        if is_guard
            && let Some(descr) = op.getdescr()
            && (descr.is_resume_guard() || descr.is_resume_guard_copied())
            && let Some(fd) = descr.as_fail_descr()
        {
            fd.set_fail_index_per_trace(fail_index);
            fd.set_source_op_index(op_idx);
        }
        fail_index += 1;
    }
}

/// history.py Box.type is on the box. Prefer the descr's `fail_arg_types`
/// slice in place — a `to_vec()` per guard was an 8 B / 16 B class on
/// the regex and/or timed row (`build_guard_metadata`).
#[cfg(test)]
fn exit_types_for_guard_or_finish<A: AsRef<InputArg>>(
    op: &Op,
    is_finish: bool,
    inputargs: &[A],
) -> ExitTypes {
    let mut out = ExitTypes::new();
    let want = if is_finish {
        Some(op.num_args())
    } else {
        op.guard_fail_args().map(|fa| fa.len())
    };
    if op.with_fail_descr(|fd| {
        let types = fd.fail_arg_types();
        if want.is_none_or(|n| types.len() == n) {
            out.extend_from_slice(types);
            true
        } else {
            false
        }
    }) == Some(true)
    {
        return out;
    }
    let finish_arg_type = |b: &Operand| -> Type { b.to_opref().ty().unwrap_or(Type::Int) };
    if is_finish {
        out.extend(op.args_slice().iter().map(finish_arg_type));
        return out;
    }
    if let Some(fail_args) = op.guard_fail_args() {
        // Copy types first: `guard_fail_args` and the extra type slice
        // cannot be borrowed together (both extras).
        let mut fa_types = ExitTypes::new();
        let had_types = op
            .with_fail_arg_types(|types| {
                fa_types.extend_from_slice(types);
            })
            .is_some();
        out.extend(fail_args.iter().enumerate().map(|(i, opref)| {
            if had_types && let Some(&tp) = fa_types.get(i) {
                return tp;
            }
            fail_arg_type(&opref.to_opref())
        }));
        return out;
    }
    if op
        .with_fail_arg_types(|types| {
            out.extend_from_slice(types);
        })
        .is_some()
    {
        return out;
    }
    out.extend(inputargs.iter().map(|arg| arg.as_ref().tp.get()));
    out
}

/// Build guard metadata for a compiled trace.
///
/// The backend numbers every guard and finish in a single exit table, so this
/// helper mirrors that numbering and records the source op index plus the
/// guard's own descr for blackhole fallback. Resume payload stays on the
/// descr (`ResumeGuardDescr.get_resumestorage`); this map does not copy it.
pub(crate) fn build_guard_metadata<T: AsRef<majit_ir::Op>, A: AsRef<InputArg>>(
    _inputargs: &[A],
    ops: &[T],
    _pc: u64,
    _frame_value_count_fn: Option<fn(i32, i32) -> usize>,
) -> crate::FxIndexMap<u32, StoredExitLayout> {
    let mut exit_layouts: crate::FxIndexMap<u32, StoredExitLayout> = Default::default();
    let mut fail_index = 0u32;
    stamp_guard_descr_trace_positions(ops);
    for (op_idx, op) in ops.iter().enumerate() {
        let op = op.as_ref();
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        if !is_guard && !is_finish {
            continue;
        }
        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: Some(op_idx),
                descr: op.getdescr(),
                op_arg_types_for_jump: None,
            },
        );
        fail_index += 1;
    }
    exit_layouts
}

pub(crate) fn merge_backend_exit_layouts<T: AsRef<majit_ir::Op>>(
    exit_layouts: &mut crate::FxIndexMap<u32, StoredExitLayout>,
    backend_layouts: &[FailDescrLayout],
    ops: &[T],
) {
    for layout in backend_layouts {
        // Pre-resolve the source-op `descr` for backend-only entries so
        // `entry.descr` matches what `build_guard_metadata` would have
        // primed had the frontend seen this exit.  When the source op
        // is no longer reachable (frontend trace evicted but backend
        // exit layout persists across previous-token fallback) the
        // backend's `fail_arg_types` is the canonical typed-data
        // carrier; mint a fresh `ResumeGuardDescr` (`make_fail_descr_typed`) carrying that vector
        // so descr-side readers stay populated.  Branch on
        // `layout.is_finish` so a backend-only FINISH entry synthesizes
        // a `_DoneWithThisFrameDescr`-flavored handle (is_finish=true).
        // Without the branch, `pyjitpl.rs`'s `resolve_is_finish` reading
        // `descr.is_finish()`
        // returns false on the synthesized guard descr and breaks
        // PyPy's `DoneWithThisFrameDescr` /
        // `ExitFrameWithExceptionDescrRef` identity check
        // (compile.py:658-662 / compile.py:701-784).
        let descr_from_op = layout
            .source_op_index
            .and_then(|idx| ops.get(idx))
            .and_then(|op| op.as_ref().getdescr())
            .or_else(|| {
                Some(if layout.is_finish {
                    make_finish_fail_descr_typed(
                        layout.fail_arg_types.clone(),
                        layout.is_exception_exit,
                    )
                } else {
                    make_fail_descr_typed(layout.fail_arg_types.clone())
                })
            });
        let entry: &mut StoredExitLayout =
            exit_layouts
                .entry(layout.fail_index)
                .or_insert_with(|| StoredExitLayout {
                    source_op_index: layout.source_op_index,
                    descr: descr_from_op.clone(),
                    op_arg_types_for_jump: None,
                });
        entry.source_op_index = layout.source_op_index;
        // Backfill descr if the entry was inserted before `op.descr` was
        // available (or the op walk produced a fresh handle). Keeps
        // descr-side identity in sync with `build_guard_metadata`'s
        // priority-1 source.
        if entry.descr.is_none() {
            entry.descr = descr_from_op;
        }
    }
}

pub(crate) fn normalize_closing_jump_args(
    ops: Vec<majit_ir::OpRc>,
    constants: &majit_ir::ConstMap<majit_ir::Value>,
    num_inputs: usize,
) -> Vec<majit_ir::OpRc> {
    let Some((label_args, label_descr)) = ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Label)
        .map(|op| (op.getarglist(), op.getdescr()))
    else {
        return ops;
    };

    let defined: crate::FxIndexSet<OpRef> = ops
        .iter()
        .filter(|op| op.result_type() != majit_ir::Type::Void && !op.pos().get().is_none())
        .map(|op| op.pos().get())
        .collect();

    let Some(jump) = ops.iter().rfind(|op| op.opcode == OpCode::Jump) else {
        return ops;
    };

    // The substitution below is positional — jump slot `idx` is repaired from
    // label slot `idx` — so it holds only while the JUMP closes onto THIS
    // LABEL. `compile_retrace` also reaches here with a JUMP retargeted at
    // another TargetToken (`unroll.py/171 jump_to_preamble`), and that
    // JUMP already carries the argument order the target token's virtual
    // state produced (`unroll.py:346-357`, which
    // `assemble_peeled_trace_with_jump_args` passes through untouched for
    // exactly this reason). Slot `idx` then names a different live value in
    // each list, and rewriting it lands the trailing label's value in an
    // unrelated target slot.
    if jump.getdescr().as_ref().map(majit_ir::descr_identity)
        != label_descr.as_ref().map(majit_ir::descr_identity)
    {
        return ops;
    }

    // optimizer.py:651-652 setarg loop parity.
    for idx in 0..jump.num_args() {
        if idx >= label_args.len() {
            break;
        }
        let arg = jump.arg(idx);
        // history.py Const* are values, not body-namespace OpRefs
        // — they never need closing-jump normalization.
        if arg.is_constant() {
            continue;
        }
        if constants.contains_key(&arg.to_opref().raw()) {
            continue;
        }
        if (arg.to_opref().raw() as usize) < num_inputs {
            continue;
        }
        if defined.contains(&arg.to_opref()) {
            continue;
        }
        jump.setarg(idx, label_args[idx].clone());
    }

    ops
}

/// `rpython/jit/metainterp/compile.py patch_new_loop_to_load_virtualizable_fields`
/// `patch_new_loop_to_load_virtualizable_fields`.
///
/// ```python
/// def patch_new_loop_to_load_virtualizable_fields(loop, jitdriver_sd, vable):
///     vinfo = jitdriver_sd.virtualizable_info
///     extra_ops = []
///     inputargs = loop.inputargs
///     vable_box = inputargs[jitdriver_sd.index_of_virtualizable]
///     i = jitdriver_sd.num_red_args
///     loop.inputargs = inputargs[:i]
///     for descr in vinfo.static_field_descrs:
///         assert i < len(inputargs)
///         box = inputargs[i]
///         opnum = OpHelpers.getfield_for_descr(descr)
///         emit_op(extra_ops,
///                 ResOperation(opnum, [vable_box], descr=descr))
///         box.set_forwarded(extra_ops[-1])
///         i += 1
///     arrayindex = 0
///     for descr in vinfo.array_field_descrs:
///         arraylen = vinfo.get_array_length(vable, arrayindex)
///         arrayop = ResOperation(rop.GETFIELD_GC_R, [vable_box], descr=descr)
///         emit_op(extra_ops, arrayop)
///         arraydescr = vinfo.array_descrs[arrayindex]
///         assert i + arraylen <= len(inputargs)
///         for index in range(arraylen):
///             opnum = OpHelpers.getarrayitem_for_descr(arraydescr)
///             box = inputargs[i]
///             emit_op(extra_ops,
///                 ResOperation(opnum,
///                              [arrayop, ConstInt(index)],
///                              descr=arraydescr))
///             i += 1
///             box.set_forwarded(extra_ops[-1])
///         arrayindex += 1
///     assert i == len(inputargs)
///     for op in loop.operations:
///         emit_op(extra_ops, op)
///     loop.operations = extra_ops
/// ```
///
/// Called from `send_loop_to_backend` (compile.py) after the loop
/// has been optimized but before it is handed to the CPU backend. The
/// virtualizable's static and array fields ride through the optimizer as
/// expanded trace inputargs; this function strips them and reconstructs
/// each field at loop entry with a `GETFIELD_GC` / `GETARRAYITEM_GC` op
/// so the compiled loop's `len(inputargs) == entry_prefix_len` matches
/// `execute_token`'s `clt._debug_nbargs` and CA's `op.args.len()`.
///
/// `entry_prefix_len` is `compile.py i = jitdriver_sd.num_red_args`: the
/// number of leading inputargs that survive the truncation, and the position
/// the field reconstruction starts from. Upstream can spell it as the red-arg
/// count because upstream's entry contract IS the red list — `warmstate.py execute_assembler
/// execute_assembler` is called with the reds, and the virtualizable is one of
/// them. A caller whose entry contract is not a red list (one flat slot per
/// declared state field, with the virtualizable occupying a single slot among
/// them) has a different width and must state it here rather than let this
/// helper re-derive one from a red count that describes a different model. It
/// is a hard error for the two to be confused: truncating at the wrong point
/// silently reinterprets live entry values as virtualizable fields.
///
/// `index_of_virtualizable` is `compile.py` — an index INTO that entry
/// prefix, so it is expressed in whichever of the two models the caller used.
///
/// `vable_array_lengths` mirrors RPython's `vinfo.get_array_length(vable, i)`
/// reads: one length per array field (in `vinfo.array_fields` order), taken
/// from the concrete virtualizable at trace-start time.
///
/// SOUNDNESS INVARIANT — baking the lengths in.
/// `compile.py:443` reads each array length off the concrete virtualizable at
/// compile time and burns that many `GETARRAYITEM` ops into the prologue, so
/// the compiled entry is only valid for a virtualizable whose arrays still
/// have those lengths. Nothing re-checks it later: the entry's compatibility
/// test covers the FIXED array fields only, and reads their length off the
/// live object rather than comparing it against a recorded one. What makes
/// the baking safe is that the array lengths are a function of the trace's
/// GREEN key — upstream the frame's array sizes come from the code object,
/// which is green — so a virtualizable with different lengths cannot reach
/// this compiled entry in the first place; it keys to a different trace. A
/// caller whose array lengths can vary while the greens stay fixed must NOT
/// route through this helper: it would hand such an entry a prologue that
/// reads a stale length.
///
/// The assertion at the baking site below is what that invariant buys in
/// code: `compile.py assert i == len(inputargs)` requires the baked
/// lengths to account for EXACTLY the inputargs the tracer expanded, so a
/// length that has moved since trace-start fails the compile instead of
/// producing a prologue that reads the wrong number of elements. It cannot
/// catch a length that changes AFTER the compile — no assertion here can,
/// which is why the green-key argument above has to hold.
pub fn patch_new_loop_to_load_virtualizable_fields(
    ops: &mut Vec<majit_ir::OpRc>,
    inputargs: &mut Vec<InputArgRc>,
    vinfo: &crate::virtualizable::VirtualizableInfo,
    vable_array_lengths: &[usize],
    entry_prefix_len: usize,
    index_of_virtualizable: usize,
    _constants: &mut majit_ir::ConstMap<majit_ir::Value>,
) {
    // `compile.py:425-461` redirects each stripped entry inputarg at its own
    // `_forwarded` slot — `box.set_forwarded(extra_ops[-1])` — and `emit_op`
    // walks those slots through `get_box_replacement` as it copies the body.
    // The caller has already run `forget_optimization_info` over the ops and
    // inputargs, so the only chains here are the ones this function writes.
    use majit_ir::{Op, OpCode, OpRef, descr::ArrayFlag};

    /// `compile.py emit_op`: append `op` to `extra_ops`, first replacing
    /// every argument by what it is forwarded to. An op whose arguments
    /// change, and every guard, is appended as a copy that the original
    /// forwards to; a guard's fail_args take `get_box_replacement(a, True)`.
    /// The copy takes the original's position: the original leaves the
    /// operation list, so the position still names exactly one producer.
    fn emit_op(extra_ops: &mut Vec<majit_ir::OpRc>, op: &majit_ir::OpRc) {
        let copy_of = || {
            let copy = op.copy_and_change(op.opcode, None, None);
            copy.pos().set(op.pos().get());
            copy
        };
        let mut copy: Option<Op> = None;
        for i in 0..op.num_args() {
            let orig_arg = op.arg(i);
            let arg = orig_arg.get_box_replacement(false);
            if arg != orig_arg {
                copy.get_or_insert_with(copy_of).setarg(i, arg);
            }
        }
        if op.opcode.is_guard() {
            let copy = copy.get_or_insert_with(copy_of);
            if let Some(fail_args) = copy.fail_args_mut() {
                for arg in fail_args.iter_mut() {
                    *arg = arg.get_box_replacement(true);
                }
            }
        }
        match copy {
            Some(copy) => {
                let copy = OpRc::new(copy);
                Operand::from_bound_op(op).set_forwarded_op(&copy);
                extra_ops.push(copy);
            }
            None => extra_ops.push(op.clone()),
        }
    }

    /// `box.set_forwarded(extra_ops[-1])`.
    fn forward_to_last(box_: &InputArgRc, extra_ops: &[majit_ir::OpRc]) {
        let last = extra_ops
            .last()
            .expect("emit_op appended the replacement op");
        Operand::from_bound_inputarg(box_).set_forwarded_op(last);
    }

    assert!(
        index_of_virtualizable < entry_prefix_len,
        "virtualizable must live inside the entry prefix (pyjitpl.py:3589 index_of_virtualizable < num_red_args)"
    );
    if inputargs.len() <= entry_prefix_len {
        // Already reduced or no virtualizable expansion in the trace.
        return;
    }

    // compile.py:429-430 — vable_box = inputargs[index_of_virtualizable].
    let expanded_inputargs: Vec<majit_ir::InputArgRc> = std::mem::take(inputargs);
    let vable_box = Operand::from_bound_inputarg(&expanded_inputargs[index_of_virtualizable]);

    // compile.py keeps Box identities disjoint automatically; in the flat
    // OpRef model the new field-load ops take positions above every runtime
    // ref already reachable from the trace.
    let mut max_runtime_ref = 0u32;
    let mut consider = |opref: majit_ir::OpRef| {
        if !opref.is_none() && !opref.is_constant() {
            max_runtime_ref = max_runtime_ref.max(opref.raw());
        }
    };
    for op in ops.iter() {
        consider(op.pos().get());
        for b in op.args_slice().iter() {
            consider(b.to_opref());
        }
        op.visit_failarg_oprefs(&mut consider);
    }
    for ia in expanded_inputargs.iter() {
        consider(ia.opref());
    }
    let mut next_opref = max_runtime_ref + 1;
    fn fresh_pos(next_opref: &mut u32, tp: Type) -> OpRef {
        let pos = OpRef::op_typed(*next_opref, tp);
        *next_opref += 1;
        pos
    }

    let mut extra_ops: Vec<majit_ir::OpRc> =
        Vec::with_capacity(ops.len() + expanded_inputargs.len() - entry_prefix_len);
    let mut i = entry_prefix_len;

    // compile.py:431-432 — i = jitdriver_sd.num_red_args; loop.inputargs =
    // inputargs[:i].  `entry_prefix_len` is that `i`, stated by the caller
    // because the two entry models it can be in disagree about the number;
    // see the SOUNDNESS INVARIANT on this function for what the reconstruction
    // below is allowed to assume about the array lengths it bakes in.
    inputargs.extend_from_slice(&expanded_inputargs[..i]);

    // compile.py:433-440 — GETFIELD_GC per static field.
    let static_descrs = vinfo.static_field_descrs();
    for (fi, field) in vinfo.static_fields.iter().enumerate() {
        assert!(
            i < expanded_inputargs.len(),
            "static field {fi} exceeds inputargs ({} <= {})",
            i,
            expanded_inputargs.len()
        );
        let descr = static_descrs
            .get(fi)
            .cloned()
            .expect("static_field_descrs must be populated by set_parent_descr");
        let opcode = match field.field_type {
            Type::Int => OpCode::GetfieldGcI,
            Type::Ref => OpCode::GetfieldGcR,
            Type::Float => OpCode::GetfieldGcF,
            Type::Void => panic!("virtualizable static field {fi} has Void type"),
        };
        let mut op = Op::new(opcode, std::slice::from_ref(&vable_box));
        op.pos().set(fresh_pos(&mut next_opref, field.field_type));
        op.setdescr(descr);
        emit_op(&mut extra_ops, &OpRc::new(op));
        forward_to_last(&expanded_inputargs[i], &extra_ops);
        i += 1;
    }

    // compile.py:441-457 — GETFIELD_GC_R (array ptr) + GETARRAYITEM_GC per element.
    let array_descrs_list = vinfo.array_field_descrs();
    for (ai, array_field_descr) in array_descrs_list.iter().enumerate() {
        // compile.py `arraylen = vinfo.get_array_length(vable, arrayindex)`
        // — the length is BAKED into the compiled prologue here, as a fixed
        // count of `GETARRAYITEM` ops. No later check re-reads it: the entry
        // compatibility test is built from the fixed array fields and takes
        // their length off the live object, so it cannot notice that a
        // virtualizable array has grown or shrunk since this compile. The
        // baking is sound only because the array lengths are a function of
        // this trace's GREENS, so a virtualizable with other lengths keys to
        // a different trace and never reaches this entry. See the SOUNDNESS
        // INVARIANT on this function.
        let array_len = vable_array_lengths.get(ai).copied().unwrap_or(0);
        // GETFIELD_GC_R(vable_box, array_field_descr) → array pointer (Ref-typed).
        let mut arr_load = Op::new(OpCode::GetfieldGcR, std::slice::from_ref(&vable_box));
        arr_load.pos().set(fresh_pos(&mut next_opref, Type::Ref));
        arr_load.setdescr(array_field_descr.clone());
        emit_op(&mut extra_ops, &OpRc::new(arr_load));
        let array_box = Operand::from_bound_op(extra_ops.last().expect("emit_op appended arrayop"));

        let array_descr = vinfo
            .array_descrs
            .get(ai)
            .cloned()
            .expect("VirtualizableInfo.array_descrs must cover every array_field");
        assert!(
            i + array_len <= expanded_inputargs.len(),
            "array {ai} length {array_len} would overrun inputargs (i={i}, len={})",
            expanded_inputargs.len()
        );
        let array_info = &vinfo.array_fields[ai];
        let item_opcode = match array_info.item_type {
            Type::Int => OpCode::GetarrayitemGcI,
            Type::Ref => OpCode::GetarrayitemGcR,
            Type::Float => OpCode::GetarrayitemGcF,
            Type::Void => panic!("virtualizable array {ai} has Void item_type"),
        };
        let (item_opcode, item_descr, item_base) = match array_info.storage {
            crate::virtualizable::VableArrayStorage::DirectPointer => {
                (item_opcode, array_descr, array_box)
            }
            crate::virtualizable::VableArrayStorage::EmbeddedArray { ptr_offset } => {
                // TODO (heap layout divergence):
                // RPython's `vinfo.array_field_descrs` points at a real
                // GC array field; `compile.py:445 ResOperation(GETFIELD_GC_R)`
                // returns the array Box and `:451 ResOperation(GETARRAYITEM_GC_*)`
                // reads the items directly. Pyre's `FixedObjectArray` is
                // a `{ ptr: *T, len: usize }` container struct, so we
                // first emit `GetfieldGcI` to project the backing-storage
                // pointer (via `ptr_offset`), then use `GetarrayitemRaw*`
                // (raw because the items live behind a non-GC pointer).
                // The `make_array_descr(0, item_size, item_type)` mirrors
                // RPython's `array_descrs[arrayindex]` shape; only the
                // base-pointer indirection step is added. Convergence
                // would require switching pyre's `FixedObjectArray` to
                // RPython's flat GC-array layout — out of scope.
                let mut ptr_load = Op::new(OpCode::GetfieldGcI, std::slice::from_ref(&array_box));
                ptr_load.pos().set(fresh_pos(&mut next_opref, Type::Int));
                ptr_load.setdescr(majit_ir::descr::make_field_descr(
                    ptr_offset,
                    std::mem::size_of::<usize>(),
                    Type::Int,
                    ArrayFlag::Unsigned,
                ));
                emit_op(&mut extra_ops, &OpRc::new(ptr_load));
                let ptr_box =
                    Operand::from_bound_op(extra_ops.last().expect("emit_op appended ptr load"));

                let raw_opcode = match array_info.item_type {
                    Type::Int => OpCode::GetarrayitemRawI,
                    Type::Ref => OpCode::GetarrayitemRawR,
                    Type::Float => OpCode::GetarrayitemRawF,
                    Type::Void => unreachable!(),
                };
                let raw_descr = majit_ir::descr::make_array_descr(
                    0,
                    crate::virtualizable::item_size_for_type(array_info.item_type),
                    array_info.item_type,
                );
                (raw_opcode, raw_descr, ptr_box)
            }
        };
        for index in 0..array_len {
            // compile.py — ConstInt(index) for the array subscript.
            let mut elem_op = Op::new(
                item_opcode,
                &[
                    item_base.clone(),
                    Operand::from_opref(OpRef::const_int(index as i64)),
                ],
            );
            elem_op
                .pos()
                .set(fresh_pos(&mut next_opref, array_info.item_type));
            elem_op.setdescr(item_descr.clone());
            emit_op(&mut extra_ops, &OpRc::new(elem_op));
            forward_to_last(&expanded_inputargs[i], &extra_ops);
            i += 1;
        }
    }

    // compile.py `assert i == len(inputargs)` when every expanded inputarg is
    // a red or a virtualizable field. A cut may also carry a snapshot box
    // promoted to an entry inputarg past that span (`CutTrace` keeps pre-cut
    // boxes as inputargs). That tail is not another array element — consuming
    // it would bake a length the greens do not describe — so it stays on the
    // loop entry. `i` past the end is still a broken expansion.
    assert!(
        i <= expanded_inputargs.len(),
        "compile.py:458 assert i == len(inputargs) failed ({i} != {})",
        expanded_inputargs.len()
    );
    if i < expanded_inputargs.len() {
        inputargs.extend_from_slice(&expanded_inputargs[i..]);
    }

    // compile.py:459-460 — `for op in loop.operations: emit_op(extra_ops, op)`.
    for op in std::mem::take(ops).iter() {
        emit_op(&mut extra_ops, op);
    }
    *ops = extra_ops;
}

/// RPython dependency.py requires GUARD_(NO_)OVERFLOW to be scheduled only
/// when there is a live preceding INT_*_OVF operation to consume.
/// intbounds.py optimize_GUARD_OVERFLOW: optimizer raises InvalidLoop for stray overflow
/// guards. This function is a post-optimization safety net: if any stray
/// guard survived, strip it to prevent backend panic.
pub(crate) fn strip_stray_overflow_guards(ops: Vec<majit_ir::OpRc>) -> Vec<majit_ir::OpRc> {
    use majit_ir::OpCode;

    let mut pending_ovf = false;
    let mut result = Vec::with_capacity(ops.len());
    for op in ops {
        match op.opcode {
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf => {
                pending_ovf = true;
                result.push(op);
            }
            OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                if pending_ovf {
                    result.push(op);
                }
                // else: stray guard — strip it (intbounds.py optimize_GUARD_OVERFLOW InvalidLoop
                // should have caught it; this is a safety net).
                pending_ovf = false;
            }
            OpCode::Label | OpCode::Jump | OpCode::Finish => {
                pending_ovf = false;
                result.push(op);
            }
            _ => {
                result.push(op);
            }
        }
    }
    result
}

// `rpython/jit/metainterp/compile.py _DoneWithThisFrameDescr` — finish/propagate descrs.
//
// These are ported as backend-agnostic `FailDescr` impls in the
// shared backend crate so `compile_tmp_callback` and
// `finish_setup` can reference the same singletons RPython does.
//
// `pyjitpl.py` `compile.make_and_attach_done_descrs([self, cpu])` —
// RPython attaches a *single* `DoneWithThisFrameDescr*` object to both
// `MetaInterpStaticData` and the CPU so the FINISH descr pointer the
// backend observes is the same Arc the metainterp reads back in
// `handle_fail`. pyre mirrors the same shape through the
// `DescrContainer` trait implemented on both `MetaInterpStaticData`
// (pyjitpl.rs) and `dyn Backend` (majit-backend/finish_descrs.rs),
// so `MetaInterp::new` installs a single `Arc` on both
// halves; `attach_descrs_to_cpu` forwards the clones to the backend.

// `compile.py, 1092-1099` `_DoneWithThisFrameDescr` family /
// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr`: the
// class definitions live in `majit-backend::finish_descrs` so backend
// codegen and synthetic-exit paths can construct them directly without
// depending on `majit-metainterp`.  Re-exported here for callers
// reaching them through `compile::DoneWithThisFrameDescr*` etc.
pub use majit_backend::{
    DoneWithThisFrameDescrFloat, DoneWithThisFrameDescrInt, DoneWithThisFrameDescrRef,
    DoneWithThisFrameDescrVoid, ExitFrameWithExceptionDescrRef, PropagateExceptionDescr,
};

pub use majit_backend::DescrContainer;

/// Compatibility entry point for `compile.py` `make_and_attach_done_descrs`.
/// The shared backend implementation preserves the same descriptor identities.
pub fn make_and_attach_done_descrs(targets: &mut [&mut dyn DescrContainer]) {
    majit_backend::make_and_attach_done_descrs(targets);
}

/// `rpython/jit/metainterp/compile.py` `compile_tmp_callback`.
///
/// Make a `JitCellToken` that corresponds to assembler code that just
/// calls back the interpreter.  Used temporarily: a fully compiled
/// version of the code may end up replacing it via
/// `redirect_call_assembler`.
///
/// The RPython-orthodox approach has **no separate "pending target
/// registry"**: every `JitCellToken` points at a real compiled body.
/// For an unfinished callee the body is a 3-op stub —
/// `CALL portal_runner_adr(funcbox, *greenboxes, *inputargs)` →
/// `GUARD_NO_EXCEPTION` → `FINISH` — which bounces control back into
/// the interpreter. Once the real trace compiles,
/// `redirect_call_assembler` (`x86/assembler.py`) in-place patches
/// `_ll_function_addr` so callers reach the real loop transparently.
///
/// # Wiring status
///
/// Recursive `CALL_ASSEMBLER` paths route unfinished callees through
/// `warmstate::get_assembler_token` (`warmstate.py`), which installs
/// this synthesised cell with `tmp=true` (`set_procedure_token(token, true)`).
/// Backend address resolution reads the descr-carried `Arc<JitCellToken>` and
/// its `_ll_function_addr`; backend registries that remain are metadata /
/// helper registries, not bodyless pending targets.
///
/// # Parameters
///
/// `jitdriver_sd` must have `portal_runner_adr`, `portal_calldescr`,
/// `portal_finishtoken`, and `propagate_exc_descr` populated.  The
/// first three are set by `warmspot.py:1010-1017`; the last two by
/// `pyjitpl.py:2279-2281` (see
/// `MetaInterpStaticData::finish_setup_descrs_for_jitdrivers`).
///
/// `memory_manager` is `compile.py`'s `memory_manager=None`
/// parameter and carries the same "for tests" meaning: pass `Some` from every
/// production path, because the warm cell that receives this token keeps only
/// a weak handle to it and `alive_loops` is the sole strong owner
/// (`memmgr.py:9-14`).
pub fn compile_tmp_callback(
    backend: &mut dyn Backend,
    jitdriver_sd: &crate::jitdriver::JitDriverStaticData,
    green_key: u64,
    greenboxes: &[Value],
    red_arg_types: &[Type],
    memory_manager: Option<&mut crate::memmgr::MemoryManager>,
) -> Result<Arc<JitCellToken>, BackendError> {
    // Invariant: every `JitDriverStaticData` reaching
    // `compile_tmp_callback` must have `portal_runner_adr` AND `portal_calldescr`
    // populated. `portal_runner_adr == 0` is the "attribute absent" sentinel
    // upstream (`warmspot.py:1010-1012` sets the address before any tmp_callback
    // can fire); `portal_calldescr.is_none()` means
    // `MetaInterpStaticData::finish_setup_descrs_for_jitdrivers` (pyjitpl.rs:
    // 12336-12338, mirroring `pyjitpl.py` + `warmspot.py:1013-1017`)  allow-line-citation
    // never ran for this driver, so `funcbox` would dereference a null portal
    // address and the resulting tmp callback would jump to 0x0. `debug_assert!`
    // catches the misuse in dev/test builds (the bench harness runs in dev
    // profile so violations surface in `pyre/check.py`); release builds opt
    // out for the same hot-path reason upstream avoids per-call asserts.
    debug_assert!(
        jitdriver_sd.portal_runner_adr != 0,
        "compile_tmp_callback: jitdriver_sd.portal_runner_adr is 0 — \
         warmspot.py:1010-1012 must populate portal_runner_adr before tmp_callback \
         can build a real funcbox"
    );
    debug_assert!(
        jitdriver_sd.portal_calldescr.is_some(),
        "compile_tmp_callback: jitdriver_sd.portal_calldescr is None — \
         MetaInterpStaticData::register_jitdriver_sd must have run \
         finish_setup_descrs_for_jitdrivers (pyjitpl.py:2274-2281 + \
         warmspot.py:1013-1017) so the CALL_* descr is available"
    );
    // The caller-supplied `red_arg_types` ↔ `jd.red_args_types` consistency
    // check lives below the upstream length assertion (`compile.py:1113`)
    // adjacent to the InputArg loop so the two signal-pairs (length /
    // typed-shape) sit together. See the `debug_assert_eq!` block at
    // `compile.py:1113` parity below.
    // `compile.py` `jitcell_token = make_jitcell_token(jitdriver_sd)`.
    // Pyre adaptation: the token carries `green_key` (enabling cell lookup
    // on later CALL_ASSEMBLER) and `virtualizable_arg_index` (cached from
    // `jitdriver_sd.index_of_virtualizable`, matching the fields populated
    // on real-loop tokens at `compile_loop`).
    //
    // `compile.py` `jitcell_token.outermost_jitdriver_sd = jitdriver_sd`
    // is set inside `make_jitcell_token`.
    //
    // `compile.py compile_tmp_callback` leaves `JitCellToken.number == -1`.
    let jitcell_token = make_jitcell_token(jitdriver_sd.index);
    // compile.py: `jl.tmp_callback(jitcell_token)` after make_jitcell_token.
    crate::rjitlog::tmp_callback(
        std::sync::Arc::as_ptr(&jitcell_token) as *const () as u64,
        jitcell_token.number() as u64,
    );
    // `green_key` / `virtualizable_arg_index` are interior-mutable (`Cell`),
    // so they are written through the shared `Arc` directly.
    jitcell_token.green_key.set(green_key);
    // Red-arg space, not entry space: this token's signature is `nb_red_args`
    // wide (asserted below), and `direct_assembler_call` selects the vable out
    // of a list of exactly that width. A driver whose compiled entry is a flat
    // state-field prefix numbers its virtualizable in the entry instead, and
    // that number would name the wrong red — or a red that is not there.
    jitcell_token
        .virtualizable_arg_index
        .set(jitdriver_sd.red_arg_virtualizable_index());
    //
    // `compile.py` `nb_red_args = jitdriver_sd.num_red_args`.
    let nb_red_args = jitdriver_sd.num_red_args();
    // `compile.py` `assert len(redargtypes) == nb_red_args`.
    assert_eq!(
        red_arg_types.len(),
        nb_red_args,
        "compile_tmp_callback: red_arg_types length mismatch",
    );
    // Contract: caller-passed `red_arg_types`
    // must match `jd.red_args_types` — upstream's `compile.py:1107-1124`
    // reads `redargtypes` from `jitdriver_sd.red_args_types` directly so
    // the tmp-callback signature is owned by the jd, not by the call
    // site. Pyre still threads `red_arg_types` through the parameter
    // list while runtime callers derive kinds from CALL_ASSEMBLER args
    // (see pyjitpl.rs); this assertion locks the invariant so a
    // later cutover can drop the parameter without
    // a silent semantic shift.
    debug_assert_eq!(
        red_arg_types,
        jitdriver_sd.red_arg_types_as_ir_types().as_slice(),
        "compile_tmp_callback: caller-provided red_arg_types must match \
         jd.red_args_types — warmspot.py:664 makes the jd the source of truth"
    );
    // `compile.py:1114-1124` build `inputargs`:
    //     for kind in redargtypes:
    //         if kind == history.INT:   box = InputArgInt()
    //         elif kind == history.REF: box = InputArgRef()
    //         elif kind == history.FLOAT: box = InputArgFloat()
    //         ...
    //         inputargs.append(box)
    let inputargs: Vec<majit_ir::InputArgRc> = red_arg_types
        .iter()
        .enumerate()
        .map(|(i, kind)| match kind {
            Type::Int => InputArg::new_int_rc(i as u32),
            Type::Ref => InputArg::new_ref_rc(i as u32),
            Type::Float => InputArg::new_float_rc(i as u32),
            Type::Void => panic!("compile_tmp_callback: void red arg is invalid"),
        })
        .collect();
    let num_inputs = inputargs.len() as u32;
    //
    // `compile.py:1125-1126`
    //     k = jitdriver_sd.portal_runner_adr
    //     funcbox = history.ConstInt(adr2int(k))
    //
    // `compile.py` `callargs = [funcbox] + greenboxes + inputargs`.
    //
    // pyre layout: the CALL op's `args` slots reference `OpRef` numbers.
    // InputArgs occupy `0..num_inputs` and are minted via
    // `InputArg::opref()` so they carry RPython-parity InputArg{Int,
    // Float,Ref} variants (resoperation.py InputArgInt/727/739). Constants
    // (funcbox + greens) carry their value inline on the OpRef
    // variant (history.py/268/314 `Const{Int,Float,Ptr}.value`).
    // `compile.py:1126` funcbox = ConstInt(adr2int(k)) — `ConstInt.value`
    // inline, carried directly on the OpRef variant.
    let funcbox_ref = OpRef::const_int(jitdriver_sd.portal_runner_adr);
    // Green boxes follow in declaration order.
    let mut callargs_box: Vec<Operand> = Vec::with_capacity(1 + greenboxes.len() + inputargs.len());
    callargs_box.push(Operand::from_opref(funcbox_ref));
    for gb in greenboxes.iter() {
        // history.py/268/314 Const{Int,Float,Ptr}.value inline.
        let g_ref = match *gb {
            Value::Int(v) => OpRef::const_int(v),
            Value::Ref(r) => OpRef::const_ptr(r),
            Value::Float(f) => OpRef::const_float(f),
            Value::Void => panic!("compile_tmp_callback: void greenbox"),
        };
        callargs_box.push(Operand::from_opref(g_ref));
    }
    // Red args — bound to the inputarg objects themselves
    // (resoperation.py InputArgInt/727/739 InputArg{Int,Float,Ref}).
    for ia in inputargs.iter() {
        callargs_box.push(Operand::from_bound_inputarg(ia));
    }
    //
    let portal_calldescr = jitdriver_sd
        .portal_calldescr
        .as_ref()
        .expect("compile_tmp_callback: jd.portal_calldescr not set")
        .clone();
    let portal_finishtoken = jitdriver_sd
        .portal_finishtoken
        .as_ref()
        .expect("compile_tmp_callback: jd.portal_finishtoken not set")
        .clone();
    let propagate_exc_descr = jitdriver_sd
        .propagate_exc_descr
        .as_ref()
        .expect("compile_tmp_callback: jd.propagate_exc_descr not set")
        .clone();
    //
    // `compile.py` `jd = jitdriver_sd`.
    // `compile.py` `opnum = OpHelpers.call_for_descr(jd.portal_calldescr)`.
    let call_opcode = OpCode::call_for_type(jitdriver_sd.result_type);
    // `compile.py:1132` `call_op = ResOperation(opnum, callargs,
    // descr=jd.portal_calldescr)`.
    let call_op = OpRc::new(Op::with_descr(call_opcode, &callargs_box, portal_calldescr));
    //
    // `compile.py` `if call_op.type != 'v': finishargs = [call_op]
    // else: finishargs = []`.
    //
    // A void CALL leaves no result OpRef — match `Op::default_pos()` /
    // `OpRef::NONE` so dynasm/cranelift backends that only emit a store
    // when `op.pos != NONE` (e.g. `x86/assembler.rs` CALL handler) don't
    // produce a bogus result slot.
    let finishargs_box: Vec<Operand> = if jitdriver_sd.result_type == Type::Void {
        Vec::new()
    } else {
        // The CALL writes to the first free OpRef after inputargs.
        // resoperation.py IntOp/FloatOp/RefOp mixin: the result
        // box of a typed CALL is a typed ResOp variant.
        let call_result_ref = OpRef::op_typed(num_inputs, jitdriver_sd.result_type);
        call_op.pos().set(call_result_ref);
        vec![Operand::from_bound_op(&call_op)]
    };
    //
    // `compile.py:1138-1144` operations = [call_op,
    //   GUARD_NO_EXCEPTION(descr=faildescr),
    //   FINISH(finishargs, descr=jd.portal_finishtoken)].
    let mut guard_op = Op::with_descr(OpCode::GuardNoException, &[], propagate_exc_descr);
    // `compile.py` `operations[1].setfailargs([])` — no fail args.
    guard_op.setfailargs(smallvec![]);
    let finish_op = Op::with_descr(OpCode::Finish, &finishargs_box, portal_finishtoken);
    let operations: Vec<majit_ir::OpRc> = vec![call_op, OpRc::new(guard_op), OpRc::new(finish_op)];
    //
    // `compile.py:1145` `operations = get_deep_immutable_oplist(operations)` —
    // pyre has no immutable-list transformation.
    //
    // `compile.py` `cpu.compile_loop(inputargs, operations, jitcell_token,
    // log=False)`.
    // Inline-Const carries each Const value directly on its OpRef
    // variant (history.py:227/268/314), so the backend pool is left
    // empty for `compile_tmp_callback`.
    backend.set_constants_pool(majit_ir::ConstMap::default());
    // compile.py `cpu.compile_loop(inputargs, operations, jitcell_token)`
    // — the same boxes, after forget_optimization_info.
    backend.compile_loop(&inputargs, &operations, &jitcell_token)?;
    // `compile.py:180-181` wire wref now that all `Arc::get_mut` writes
    // have settled.  `compile_tmp_callback` doesn't go through
    // `record_loop_or_bridge` (the tmp callback is a synthetic
    // 3-instruction loop, not a real trace), so wref wiring happens
    // directly here.
    wire_clt_loop_token_wref(&jitcell_token);
    //
    // `compile.py:1148-1149` `if memory_manager is not None:
    //   memory_manager.keep_loop_alive(jitcell_token)`.  The cell this token
    // is about to be installed on holds a WEAK handle
    // (`warmstate.py:188/:208-210`), so `alive_loops` is the only strong owner
    // there will ever be: skipping this call leaves the token with no owner at
    // all past the caller's temporary, and the cell answers
    // `get_procedure_token() == None` — "never compiled" — instead of failing
    // loudly.  `Option` mirrors upstream's `memory_manager=None  # for tests`
    // default.
    if let Some(memory_manager) = memory_manager {
        memory_manager.keep_loop_alive(&jitcell_token);
    }
    //
    // `compile.py:1150` `return jitcell_token`.
    // `compile.py` record_loop_or_bridge: the tmp-callback loop is a
    // real compiled loop even though MetaInterp never inserts it into
    // compiled_loops. Register it with the backend so `find_descr_by_ptr`
    // can still walk its fail_descrs on cross-token guard resolution.
    backend.track_compiled_token(Arc::clone(&jitcell_token));
    Ok(jitcell_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::make_fail_descr_with_index;
    use crate::history::test_support::rooted_inputarg_operand;
    use crate::resume::{ResumeDataLoopMemo, SimpleBoxEnv, Snapshot, SnapshotFrame};
    use majit_ir::{ArrayFlag, Op, OpCode, OpRef};

    /// `normalize_closing_jump_args` repairs a JUMP slot from the LABEL slot
    /// at the same index, which only names the same live value while the JUMP
    /// closes onto that LABEL. `compile_retrace` also feeds it a trace whose
    /// JUMP was retargeted at another TargetToken, and there the two lists are
    /// ordered by two different virtual states.
    #[test]
    fn closing_jump_to_a_foreign_target_token_keeps_its_own_args() {
        use crate::history::test_support::rooted_resop_operand;
        use majit_ir::OpRc;

        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let build = |label_descr: majit_ir::DescrRef, jump_descr: majit_ir::DescrRef| {
            let mut label = Op::new(
                OpCode::Label,
                &[
                    rooted_resop_operand(Type::Ref, 20),
                    rooted_resop_operand(Type::Ref, 21),
                    rooted_resop_operand(Type::Ref, 22),
                ],
            );
            label.setdescr(label_descr);
            let mut jump = Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Ref, 50),
                    rooted_resop_operand(Type::Ref, 51),
                    rooted_resop_operand(Type::Ref, 52),
                ],
            );
            jump.setdescr(jump_descr);
            vec![OpRc::new(label), OpRc::new(jump)]
        };
        let jump_args = |ops: &[OpRc]| -> Vec<OpRef> {
            ops.iter()
                .rfind(|op| op.opcode == OpCode::Jump)
                .map(|op| op.args_slice().iter().map(|a| a.to_opref()).collect())
                .unwrap()
        };

        // Same token object: the JUMP feeds this LABEL slot for slot, so an
        // arg with no definition in the trace is repaired from it.
        let same_token = majit_ir::make_loop_target_descr(2, false);
        let same = normalize_closing_jump_args(
            build(same_token.clone(), same_token.clone()),
            &constants,
            8,
        );
        assert_eq!(
            jump_args(&same),
            vec![OpRef::ref_op(20), OpRef::ref_op(21), OpRef::ref_op(22)]
        );

        // Foreign token: the JUMP already carries that token's argument order.
        let local = majit_ir::make_loop_target_descr(2, false);
        let foreign = normalize_closing_jump_args(
            build(local.clone(), majit_ir::make_loop_target_descr(1, false)),
            &constants,
            8,
        );
        assert_eq!(
            jump_args(&foreign),
            vec![OpRef::ref_op(50), OpRef::ref_op(51), OpRef::ref_op(52)]
        );

        // Same `Descr::index`, distinct objects: still a foreign JUMP.
        let same_index_distinct = normalize_closing_jump_args(
            build(local, majit_ir::make_loop_target_descr(2, false)),
            &constants,
            8,
        );
        assert_eq!(
            jump_args(&same_index_distinct),
            vec![OpRef::ref_op(50), OpRef::ref_op(51), OpRef::ref_op(52)]
        );
    }

    /// The deadframe rooting in `handle_fail` reaches the layout, not the
    /// descr it came from, so the layout answers from its own types.  A
    /// force-token slot is traced like any other ref — `compute_gcmap`
    /// marks every REF failarg it does not skip, and it skips only the hole a
    /// virtual leaves.
    #[test]
    fn exit_layout_traces_every_ref_slot_including_force_tokens() {
        let layout = CompiledExitLayout {
            rd_loop_token: 0,
            trace_id: 0,
            fail_index: 0,
            source_op_index: None,
            exit_types: ExitTypes::from_slice(&[Type::Ref, Type::Int, Type::Ref]),
            is_finish: false,
            is_exception_exit: false,
            storage: None,
        };

        assert!(layout.is_traced_ref_slot(0));
        assert!(!layout.is_traced_ref_slot(1));
        assert!(layout.is_traced_ref_slot(2));
        assert!(!layout.is_traced_ref_slot(3));
    }

    /// Frame value count for the `build_guard_metadata` fixtures below, each
    /// of which numbers one box per frame. Passed as the driver-scoped count
    /// so the decode does not fall back to the process-global
    /// `set_frame_value_count_fn`, which a test running in parallel may have
    /// installed for its own jitcodes.
    fn one_box_per_frame(_jitcode_index: i32, _pc: i32) -> usize {
        1
    }

    // history.py ConstInt.value inline — SimpleBoxEnv.get_const
    // reads inline-Const directly without the legacy raw-u32 side table.
    #[test]
    fn test_build_guard_metadata_keeps_vable_array_out_of_frame_slots() {
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(0, Type::Ref);
        env.types.insert(1, Type::Int);

        let snapshot = Snapshot {
            vable_array: vec![
                OpRef::input_arg_ref(0).into(),
                OpRef::const_int(8).into(),
                OpRef::const_int(777).into(), // code object payload
                OpRef::const_int(2).into(),
                OpRef::const_int(999).into(), // namespace payload
            ],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 8,
                py_pc: 8,
                boxes: vec![OpRef::input_arg_int(1).into()],
            }],
        };
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();
        // An unused extra pool slot: a fresh ResumeDataLoopMemo would
        // not count it. The descr's own rd_consts length is the pool
        // (compile.py storage.rd_consts).
        let mut rd_consts = memo.consts().to_vec();
        rd_consts.push(majit_ir::Const::Int(123_456_789));
        let rd_consts_len = rd_consts.len();

        let inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_int_rc(1)];
        let mut guard = Op::new(OpCode::GuardTrue, &[rooted_inputarg_operand(Type::Int, 1)]);
        let descr = crate::compile::make_resume_guard_descr_typed(vec![Type::Ref, Type::Int]);
        if let Some(fd) = descr.as_fail_descr() {
            fd.set_rd_numb_arc(Some(rd_numb));
            fd.set_rd_consts(Some(rd_consts));
        }
        guard.setdescr(descr);
        guard.setfailargs(smallvec::smallvec![
            rooted_inputarg_operand(Type::Ref, 0),
            rooted_inputarg_operand(Type::Int, 1)
        ]);
        guard.set_fail_arg_types(vec![Type::Ref, Type::Int]);

        let exit_layouts = build_guard_metadata(&inputargs, &[guard], 8, Some(one_box_per_frame));
        let exit = exit_layouts.get(&0).expect("guard exit layout");
        let descr = exit.descr.as_ref().expect("descr");
        let storage = crate::resume::get_resumestorage(descr).expect("get_resumestorage");
        let rd_numb = storage.rd_numb().expect("rd_numb");
        let rd_consts = storage.rd_consts().expect("rd_consts");
        assert_eq!(
            rd_consts.len(),
            rd_consts_len,
            "descr keeps the guard rd_consts pool; it does not re-number into a fresh memo"
        );

        let fvc: fn(i32, i32) -> usize = one_box_per_frame;
        let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> = Some(&fvc);
        let (_nfa, _vable, _vref, frames) = majit_ir::resumedata::rebuild_from_numbering(
            rd_numb,
            rd_consts,
            &[Type::Ref, Type::Int],
            fvc_ref,
            0,
        );
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].values.len(), 1);
        match &frames[0].values[0] {
            majit_ir::resumedata::RebuiltValue::Box(idx, ty) => {
                assert_eq!(*idx, 1);
                assert_eq!(*ty, Type::Int);
            }
            other => panic!("expected Box fail_arg 1, got {other:?}"),
        }
    }

    #[test]
    fn test_build_guard_metadata_prefers_explicit_fail_arg_types_over_stale_inputarg_types() {
        let inputargs = vec![
            InputArg::new_ref_rc(0),
            InputArg::new_ref_rc(1),
            InputArg::new_ref_rc(2),
            InputArg::new_ref_rc(3),
        ];
        let mut guard = Op::new(OpCode::GuardTrue, &[rooted_inputarg_operand(Type::Ref, 0)]);
        let fail_arg_types = vec![Type::Ref, Type::Ref, Type::Int, Type::Int];
        let descr = make_fail_descr_with_index(0, fail_arg_types.len());
        descr
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(fail_arg_types.clone());
        guard.setdescr(descr);
        guard.setfailargs(smallvec::smallvec![
            rooted_inputarg_operand(Type::Ref, 0),
            rooted_inputarg_operand(Type::Ref, 1),
            rooted_inputarg_operand(Type::Ref, 2),
            rooted_inputarg_operand(Type::Ref, 3)
        ]);
        guard.set_fail_arg_types(fail_arg_types);

        let exit_layouts = build_guard_metadata(&inputargs, &[guard], 0, None);
        let exit = exit_layouts.get(&0).expect("guard exit layout");

        assert_eq!(
            exit.resolve_exit_types(),
            &[Type::Ref, Type::Ref, Type::Int, Type::Int][..]
        );
    }

    #[test]
    fn test_exit_types_for_guard_reads_descr_slice() {
        // history.py Box.type lives on the box / descr. A heap `to_vec`
        // per guard was an 8 B / 16 B class on the regex and/or row.
        let value = rooted_inputarg_operand(Type::Int, 0);
        let descr = make_fail_descr_typed(vec![Type::Int, Type::Ref]);
        let mut guard = Op::with_descr(OpCode::GuardTrue, std::slice::from_ref(&value), descr);
        guard.setfailargs(smallvec::smallvec![
            rooted_inputarg_operand(Type::Int, 0),
            rooted_inputarg_operand(Type::Ref, 1),
        ]);
        let types = exit_types_for_guard_or_finish::<InputArgRc>(&guard, false, &[]);
        assert_eq!(&types[..], &[Type::Int, Type::Ref]);
    }

    #[test]
    fn test_build_guard_metadata_does_not_mint_derived_frame_vecs() {
        // resume.py walks `rd_numb` on the descr. StoredExitLayout no longer
        // projects parallel frame_pcs / frame_slot_counts / slot_sources Vecs.
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(0, Type::Int);
        let snapshot = Snapshot {
            vable_array: vec![],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 4,
                py_pc: 4,
                boxes: vec![OpRef::input_arg_int(0).into()],
            }],
        };
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();
        let rd_consts = memo.consts().to_vec();

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = Op::new(OpCode::GuardTrue, &[rooted_inputarg_operand(Type::Int, 0)]);
        let descr = crate::compile::make_resume_guard_descr_typed(vec![Type::Int]);
        if let Some(fd) = descr.as_fail_descr() {
            fd.set_rd_numb_arc(Some(rd_numb));
            fd.set_rd_consts(Some(rd_consts));
        }
        guard.setdescr(descr);
        guard.setfailargs(smallvec::smallvec![rooted_inputarg_operand(Type::Int, 0)]);
        guard.set_fail_arg_types(vec![Type::Int]);

        let exit_layouts = build_guard_metadata(&inputargs, &[guard], 4, Some(one_box_per_frame));
        let exit = exit_layouts.get(&0).expect("guard exit layout");
        let descr = exit.descr.as_ref().expect("descr");
        let storage = crate::resume::get_resumestorage(descr).expect("get_resumestorage");
        assert!(storage.rd_numb().is_some());
        assert_eq!(exit.source_op_index, Some(0));
    }

    #[test]
    fn test_build_guard_metadata_restamps_frontend_source_op_index() {
        let inputargs = vec![InputArg::new_int_rc(0)];
        let value = rooted_inputarg_operand(Type::Int, 0);
        let prefix = Op::new(OpCode::SameAsI, std::slice::from_ref(&value));
        let descr = make_fail_descr_with_index(0, 1);
        let fd = descr.as_fail_descr().unwrap();
        // The GC-rewritten backend trace can insert operations before a
        // guard and temporarily stamp its prepared-op index on the shared
        // descr.  That index is not valid in the retained frontend trace.
        fd.set_source_op_index(99);
        let mut guard = Op::with_descr(
            OpCode::GuardTrue,
            std::slice::from_ref(&value),
            descr.clone(),
        );
        guard.setfailargs(smallvec::smallvec![value]);
        guard.set_fail_arg_types(vec![Type::Int]);

        let exit_layouts = build_guard_metadata(&inputargs, &[prefix, guard], 0, None);
        let exit = exit_layouts.get(&0).expect("guard exit layout");

        assert_eq!(exit.source_op_index, Some(1));
        assert_eq!(fd.source_op_index(), Some(1));
    }

    #[test]
    fn test_patch_new_loop_reemits_ops_through_forwarded_results() {
        let mut vinfo = crate::virtualizable::VirtualizableInfo::new(0);
        vinfo.add_field("obj", Type::Ref, 8);
        vinfo.set_parent_descr(majit_ir::descr::make_size_descr(16));

        // op0 produces a ResOp result at ref_op(10); the Label and getfield
        // consumers bind that result (from_bound_op) instead of a position-only
        // box, so patch_new_loop's forwarding rewrites them through op identity.
        let mut inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_ref_rc(1)];
        let ia = |k: usize| majit_ir::operand::Operand::from_bound_inputarg(&inputargs[k]);
        let op0: majit_ir::OpRc = {
            let mut op = Op::new(OpCode::SameAsR, &[ia(1)]);
            op.pos().set(OpRef::ref_op(10));
            OpRc::new(op)
        };
        let op0_result = majit_ir::operand::Operand::from_bound_op(&op0);
        let op0_copy_source = op0.clone();
        let op1: majit_ir::OpRc = OpRc::new(Op::new(OpCode::Label, &[ia(0), op0_result.clone()]));
        let op2: majit_ir::OpRc = {
            let mut op = Op::new(OpCode::GetfieldGcI, &[op0_result]);
            op.pos().set(OpRef::int_op(11));
            op.setdescr(majit_ir::descr::make_field_descr(
                16,
                8,
                Type::Int,
                ArrayFlag::Signed,
            ));
            OpRc::new(op)
        };
        let mut ops: Vec<majit_ir::OpRc> = vec![op0, op1, op2];
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        patch_new_loop_to_load_virtualizable_fields(
            &mut ops,
            &mut inputargs,
            &vinfo,
            &[],
            1,
            0,
            &mut constants,
        );

        assert_eq!(
            inputargs.iter().map(|a| a.opref()).collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(0)]
        );
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        let vable_field = ops[0].pos().get();

        assert_eq!(ops[1].opcode, OpCode::SameAsR);
        assert_eq!(
            ops[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![vable_field]
        );
        // The copy stands in for op0 at op0's own position.
        let forwarded_same_as = ops[1].pos().get();
        assert_eq!(forwarded_same_as, OpRef::ref_op(10));
        assert!(!majit_ir::OpRc::ptr_eq(&ops[1], &op0_copy_source));

        assert_eq!(ops[2].opcode, OpCode::Label);
        assert_eq!(
            ops[2]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(0), forwarded_same_as]
        );

        assert_eq!(ops[3].opcode, OpCode::GetfieldGcI);
        assert_eq!(
            ops[3]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![forwarded_same_as]
        );
    }

    #[test]
    fn test_patch_new_loop_reads_embedded_array_items_from_backing_storage() {
        let mut vinfo = crate::virtualizable::VirtualizableInfo::new(0);
        vinfo.add_embedded_array_field(
            "locals_cells_stack_w",
            Type::Ref,
            8,
            0,
            8,
            0,
            majit_ir::descr::make_array_descr(0, 8, Type::Ref),
        );
        vinfo.set_parent_descr(majit_ir::descr::make_size_descr(16));

        let mut inputargs = vec![
            InputArg::new_ref_rc(0),
            InputArg::new_ref_rc(1),
            InputArg::new_ref_rc(2),
        ];
        let ops = vec![Op::new(
            OpCode::Label,
            &inputargs
                .iter()
                .map(majit_ir::operand::Operand::from_bound_inputarg)
                .collect::<Vec<_>>(),
        )];
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        let mut ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();
        patch_new_loop_to_load_virtualizable_fields(
            &mut ops,
            &mut inputargs,
            &vinfo,
            &[2],
            1,
            0,
            &mut constants,
        );

        assert_eq!(
            inputargs.iter().map(|a| a.opref()).collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(0)]
        );
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcI);
        assert_eq!(
            ops[1]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![ops[0].pos().get()]
        );
        assert_eq!(ops[2].opcode, OpCode::GetarrayitemRawR);
        assert_eq!(ops[2].arg(0).to_opref(), ops[1].pos().get());
        assert_eq!(ops[3].opcode, OpCode::GetarrayitemRawR);
        assert_eq!(ops[3].arg(0).to_opref(), ops[1].pos().get());
        assert_eq!(ops[4].opcode, OpCode::Label);
        assert_eq!(
            ops[4]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![
                OpRef::input_arg_ref(0),
                ops[2].pos().get(),
                ops[3].pos().get()
            ]
        );
    }

    /// Residual body-LABEL `RefOp`s that still forward to an expanded
    /// inputarg are the same Box as `loop.inputargs[i]`. `emit_op` must
    /// rewrite them to the GETARRAYITEM, not leave a producerless hole.
    #[test]
    fn test_patch_new_loop_rewrites_residual_label_refop_forwarded_to_inputarg() {
        use crate::history::test_support::rooted_resop_operand;

        let mut vinfo = crate::virtualizable::VirtualizableInfo::new(0);
        vinfo.add_embedded_array_field(
            "locals_cells_stack_w",
            Type::Ref,
            8,
            0,
            8,
            0,
            majit_ir::descr::make_array_descr(0, 8, Type::Ref),
        );
        vinfo.set_parent_descr(majit_ir::descr::make_size_descr(16));

        let mut inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_ref_rc(1)];
        let reminted = rooted_resop_operand(Type::Ref, 85);
        reminted.set_forwarded_inputarg(&inputargs[1]);

        let mut ops: Vec<majit_ir::OpRc> = vec![OpRc::new(Op::new(
            OpCode::Label,
            &[
                majit_ir::operand::Operand::from_bound_inputarg(&inputargs[0]),
                reminted,
            ],
        ))];
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();

        patch_new_loop_to_load_virtualizable_fields(
            &mut ops,
            &mut inputargs,
            &vinfo,
            &[1],
            1,
            0,
            &mut constants,
        );

        assert_eq!(
            inputargs.iter().map(|a| a.opref()).collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(0)]
        );
        let label = ops.iter().find(|op| op.opcode == OpCode::Label).unwrap();
        let getitem = ops
            .iter()
            .find(|op| op.opcode == OpCode::GetarrayitemRawR)
            .unwrap();
        assert_eq!(
            label
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(0), getitem.pos().get()]
        );
    }

    /// The entry prefix is the caller's number, not `num_red_args`.
    ///
    /// A flat entry contract puts the virtualizable at some interior slot with
    /// other live entry values around it — here one int scalar at slot 0, the
    /// virtualizable at slot 1, and the array elements following from slot 2.
    /// `compile.py:431-432` must truncate at 2 and start reconstructing there;
    /// a helper that re-derived the point from a red count (this driver
    /// declares one red, the whole state) would cut at 1 and reinterpret the
    /// virtualizable itself as the first array element.
    #[test]
    fn test_patch_new_loop_truncates_at_the_callers_entry_prefix_not_a_red_count() {
        let mut vinfo = crate::virtualizable::VirtualizableInfo::new(0);
        vinfo.add_embedded_array_field(
            "regs",
            Type::Int,
            8,
            0,
            8,
            0,
            majit_ir::descr::make_array_descr(0, 8, Type::Int),
        );
        vinfo.set_parent_descr(majit_ir::descr::make_size_descr(16));

        let mut inputargs = vec![
            InputArg::new_int_rc(0),
            InputArg::new_ref_rc(1),
            InputArg::new_int_rc(2),
            InputArg::new_int_rc(3),
        ];
        let ops = vec![Op::new(
            OpCode::Label,
            &inputargs
                .iter()
                .map(majit_ir::operand::Operand::from_bound_inputarg)
                .collect::<Vec<_>>(),
        )];
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let mut ops: Vec<majit_ir::OpRc> = ops.into_iter().map(OpRc::new).collect();

        patch_new_loop_to_load_virtualizable_fields(
            &mut ops,
            &mut inputargs,
            &vinfo,
            &[2],
            2,
            1,
            &mut constants,
        );

        // The scalar at slot 0 survives the truncation alongside the
        // virtualizable; only the two array elements are reconstructed.
        assert_eq!(
            inputargs.iter().map(|a| a.opref()).collect::<Vec<_>>(),
            vec![OpRef::input_arg_int(0), OpRef::input_arg_ref(1)]
        );
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        // The array load reads through the virtualizable at slot 1, not the
        // int scalar at slot 0.
        assert_eq!(ops[0].arg(0).to_opref(), OpRef::input_arg_ref(1));
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[2].opcode, OpCode::GetarrayitemRawI);
        assert_eq!(ops[3].opcode, OpCode::GetarrayitemRawI);
        assert_eq!(ops[4].opcode, OpCode::Label);
        assert_eq!(
            ops[4]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![
                OpRef::input_arg_int(0),
                OpRef::input_arg_ref(1),
                ops[2].pos().get(),
                ops[3].pos().get()
            ]
        );
    }

    /// A declared flat contract is what every consumer of the entry shape must
    /// see — both the width and the virtualizable's position inside it.
    #[test]
    fn test_flat_entry_contract_overrides_the_red_list_spelling() {
        use crate::jitdriver::{FlatEntryContract, JitDriverStaticData};

        // The merge-point payload schema: the whole state is a single red, so
        // a red-list lookup would answer "slot 0" for the virtualizable.
        let mut sd = JitDriverStaticData::with_virtualizable(
            vec![("pc", Type::Int), ("program", Type::Ref)],
            vec![("state", Type::Ref)],
            Some("state"),
        );
        assert_eq!(sd.virtualizable_arg_index(), Some(0));
        assert_eq!(sd.num_reds(), 1);

        sd.flat_entry = Some(FlatEntryContract {
            len: 2,
            index_of_virtualizable: 1,
        });
        assert_eq!(sd.flat_entry_contract().map(|f| f.len), Some(2));
        assert_eq!(sd.virtualizable_arg_index(), Some(1));
        // The red count is untouched — it still describes the payload, which
        // is why it cannot double as the entry width.
        assert_eq!(sd.num_reds(), 1);
    }
}
/// Re-export the backend-owned resume payload under its historical
/// `compile::RdPayload` path. [`copy_all_attributes_from`] shares the payload's
/// `Arc`-backed sections, matching `ResumeGuardDescr.copy_all_attributes_from`
/// without changing the descriptor identity stored on a guard operation.
pub use majit_backend::RdPayload;

fn push_vector_info(head: &mut Option<Box<AccumInfo>>, mut info: AccumInfo) {
    info.prev = head.take();
    *head = Some(Box::new(info));
}

fn flatten_vector_info(head: Option<&AccumInfo>) -> Vec<AccumInfo> {
    let mut result = Vec::new();
    let mut current = head;
    while let Some(info) = current {
        result.push(info.clone());
        current = info.prev.as_deref();
    }
    result
}

/// `compile.py self.rd_vector_info = other.rd_vector_info.clone()`
/// rebuild helper: takes the donor's flattened chain (head at index 0)
/// and assembles the equivalent linked-list head suitable for writing
/// through `vector_info: UnsafeCell<Option<Box<AccumInfo>>>`.
fn build_vector_info_chain(chain: Vec<AccumInfo>) -> Option<Box<AccumInfo>> {
    let mut current: Option<Box<AccumInfo>> = None;
    for mut info in chain.into_iter().rev() {
        info.prev = current;
        current = Some(Box::new(info));
    }
    current
}

/// Allocate the next unique fail_index.
///
/// `compile.py AbstractResumeGuardDescr` has no integer identity: the
/// descr object is the guard. The backend still keys exits on an integer,
/// so both mint sites (`ResumeGuardDescr` construction here and
/// `alloc_fail_index` in `resume_guard_descr`) share that one counter.
/// A second counter starting at 1 collides and a lookup by integer
/// attaches the bridge to the wrong guard.
fn alloc_fail_index() -> u32 {
    majit_backend::alloc_fail_index()
}

// `compile.py AbstractResumeGuardDescr` status-bit constants.
//
// Status packs three pieces in one `u64`:
//   - bit 0          : `ST_BUSY_FLAG` (set during retrace; clear once done).
//   - bits 1..3      : `ST_TYPE_MASK` — `TY_NONE` / `TY_INT` / `TY_REF` /
//                      `TY_FLOAT`, set by `make_a_counter_per_value` to
//                      distinguish guard_value-by-int / -by-ref / -by-float.
//   - bits 3..end    : jitcounter hash (when TY_NONE) or backend value-slot
//                      index (when TY_INT/REF/FLOAT), read as
//                      `status >> STATUS_SHIFT` masked with
//                      `STATUS_SHIFT_MASK` (`compile.py
//                      AbstractResumeGuardDescr.ST_SHIFT` /
//                      `ST_SHIFT_MASK`).
//
// What the value-slot index NAMES is backend-specific: dynasm and cranelift
// record a deadframe slot, which only a caller still holding the deadframe can
// read, so it arrives at `must_compile_with_values` as `guard_value_operand`;
// wasm records a fail-argument position. See the same note on
// `guard_value_counter_slot` in majit-backend's `resume_guard_descr.rs`, which
// this block mirrors.
pub(crate) const STATUS_BUSY_FLAG: u64 = 0x01;
#[allow(dead_code)]
pub(crate) const STATUS_TYPE_MASK: u64 = 0x06;
pub(crate) const STATUS_SHIFT: u32 = 3;
pub(crate) const STATUS_SHIFT_MASK: u64 = !((1u64 << STATUS_SHIFT) - 1);
#[allow(dead_code)]
pub(crate) const STATUS_TY_NONE: u64 = 0x00;
#[allow(dead_code)]
pub(crate) const STATUS_TY_INT: u64 = 0x02;
#[allow(dead_code)]
pub(crate) const STATUS_TY_REF: u64 = 0x04;
#[allow(dead_code)]
pub(crate) const STATUS_TY_FLOAT: u64 = 0x06;

// MetaFailDescr removed: pyre-introduced placeholder for non-resume
// FailDescr is no longer needed.  All factories
// (`make_fail_descr_typed`, `make_fail_descr`, `make_fail_descr_with_index`)
// now route through `make_resume_guard_descr_typed` to produce a
// `ResumeGuardDescr` (compile.py), matching PyPy's class hierarchy
// where placeholder guard descrs are the same `ResumeGuardDescr`
// instances backfilled by `store_final_boxes_in_guard` (compile.py:869).

// ResumeGuardDescr moved to majit-backend::resume_guard_descr (Phase
// C-1 cascade endpoint).  Re-exported here for caller compatibility;
// all impl Descr / impl FailDescr live in majit-backend along with
// the helpers (alloc_fail_index, push_vector_info, etc.) and status
// constants.
pub use majit_backend::ResumeGuardDescr;

/// Create a FailDescr for `num_live` integer values with an auto-assigned
/// unique fail_index.
///
/// Each call produces a distinct fail_index so the backend can identify
/// which guard failed.
pub fn make_fail_descr(num_live: usize) -> DescrRef {
    make_resume_guard_descr_typed(vec![Type::Int; num_live])
}

/// Create a FailDescr with an explicit fail_index. Tests only — see
/// `compile.rs::tests` for the invocation that needs a fixed fail_index
/// to align against a synthesised bridge descr.  The result is a
/// `ResumeGuardDescr` (PyPy `compile.py`) with the
/// `Descr::index()` global fail_index set to the requested value
/// instead of `alloc_fail_index()`.
#[cfg(test)]
pub fn make_fail_descr_with_index(fail_index: u32, num_live: usize) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index,
        types: UnsafeCell::new(vec![Type::Int; num_live]),
        payload: RdPayload::empty(),
        vector_info: UnsafeCell::new(None),
        adr_jump_offset: UnsafeCell::new(0),
        rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
        status: AtomicU64::new(0),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
        bridge_declined_terminally: AtomicBool::new(false),
        source_op_index: UnsafeCell::new(None),
        back_edge_poll: std::sync::atomic::AtomicBool::new(false),
        trace_info: AtomicPtr::new(std::ptr::null_mut()),
        external_jump_target: OnceLock::new(),
        bridge: std::sync::OnceLock::new(),
    })
}

/// Create a guard FailDescr with explicit types and auto-assigned fail_index.
///
/// `compile.py ResumeGuardDescr` — pyre routes the
/// non-finish guard-placeholder factory through the same
/// `ResumeGuardDescr` constructor `make_resume_guard_descr_typed` uses;
/// the result's `is_resume_guard()` returns true and the empty
/// `payload` is filled later by `store_final_boxes_in_guard`
/// (`compile.py:869`).  This collapses pyre's earlier
/// `MetaFailDescr` placeholder into the PyPy class.
pub fn make_fail_descr_typed(types: Vec<Type>) -> DescrRef {
    make_resume_guard_descr_typed(types)
}

/// FINISH-flavored variant of [`make_fail_descr_typed`]. Used by the
/// backend-only FINISH fallback in `merge_backend_exit_layouts` when
/// the FINISH op has been evicted, and by FINISH-singleton fallbacks in
/// tests that bypass `MetaInterp::new` (so the resulting descr's
/// `is_finish()` matches `compile.py ExitFrameWithExceptionDescrRef`
/// / `pyjitpl.py compile_done_with_this_frame` semantics).
///
/// `compile.py` `_DoneWithThisFrameDescr` family — return the
/// class-distinct singleton matching the result-type signature.
///
/// `is_exception_exit` discriminates `compile.py:658-662
/// ExitFrameWithExceptionDescrRef` (the exception-propagation FINISH
/// that raises `jitexc.ExitFrameWithExceptionRef`) from the
/// normal-result `compile.py DoneWithThisFrameDescrRef`; both
/// carry a single `Type::Ref` slot, so the type list alone cannot
/// distinguish them.
pub fn make_finish_fail_descr_typed(types: Vec<Type>, is_exception_exit: bool) -> DescrRef {
    if is_exception_exit {
        return Arc::new(majit_backend::ExitFrameWithExceptionDescrRef::new());
    }
    match types.as_slice() {
        [] => Arc::new(DoneWithThisFrameDescrVoid::new()),
        [Type::Float] => Arc::new(DoneWithThisFrameDescrFloat::new()),
        [Type::Ref] => Arc::new(DoneWithThisFrameDescrRef::new()),
        [Type::Int] => Arc::new(DoneWithThisFrameDescrInt::new()),
        // Multi-result FINISH is a Pyre extension; preserve the terminal
        // layout even though PyPy's finish-descr class family has only
        // 0/1-result concrete subclasses.  Route through a Vec<Type>-keyed
        // cache so two callers with structurally equal type lists share
        // one Arc (matching the singleton semantics of `compile.py:626-656`).
        _ => majit_backend::get_or_attach_done_with_this_frame_descr_multi(types),
    }
}

/// compile.py `ResumeGuardDescr` parity: a fresh guard descr
/// carrying the post-numbering `fail_arg_types`. Used by
/// `store_final_boxes_in_guard` to replace the tracer-stamped descr
/// (whose `types` reflect the pre-numbering snapshot)
/// with a descr whose `fail_arg_types()` matches `op.fail_arg_types`
/// exactly.
///
/// `payload` is initialized empty here; `store_final_boxes_in_guard`
/// in `optimizeopt/mod.rs` fills `rd_numb / rd_consts / rd_virtuals
/// / rd_pendingfields` post-numbering through the descr-side
/// `set_rd_*` setters (compile.py:855 `_attrs_`).  The legacy
/// `ResumeData` field is kept only for tests that still mint synthetic
/// guards; production reads route through `payload`.
pub fn make_resume_guard_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        payload: RdPayload::empty(),
        vector_info: UnsafeCell::new(None),
        adr_jump_offset: UnsafeCell::new(0),
        rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
        status: AtomicU64::new(0),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
        bridge_declined_terminally: AtomicBool::new(false),
        source_op_index: UnsafeCell::new(None),
        back_edge_poll: std::sync::atomic::AtomicBool::new(false),
        trace_info: AtomicPtr::new(std::ptr::null_mut()),
        external_jump_target: OnceLock::new(),
        bridge: std::sync::OnceLock::new(),
    })
}

/// The `ResumeGuardDescr` inside a descr that either is one or is one of its
/// tag-only newtypes (`ResumeGuardForcedDescr` / `ResumeGuardExcDescr`).
///
/// `ResumeGuardForcedDescr::as_any` returns the wrapper so `is_guard_forced`
/// dispatch can downcast the subtype; `ResumeGuardExcDescr::as_any` returns
/// the inner descr. Readers that need the base resume payload
/// (`rd_numb`, `rd_loop_token_clt`) go through this unwrap — including
/// Cranelift `cranelift_resumedata_deopt`.
pub fn resume_guard_descr(descr: &DescrRef) -> Option<&ResumeGuardDescr> {
    resume_guard_from_any(descr.as_any()?)
}

/// `compile.py` `assert isinstance(self, ResumeGuardDescr)` then field
/// reads. `ResumeGuardForcedDescr::as_any` returns the wrapper; the
/// other tag-only newtypes hand out the inner descr.
pub fn resume_guard_from_fail(fd: &dyn FailDescr) -> Option<&ResumeGuardDescr> {
    resume_guard_from_any(fd.as_any()?)
}

/// The `ResumeGuardDescr` inside a descr that either is one or is one of its
/// tag-only newtypes.
fn resume_guard_from_any(any: &dyn std::any::Any) -> Option<&ResumeGuardDescr> {
    if let Some(plain) = any.downcast_ref::<ResumeGuardDescr>() {
        return Some(plain);
    }
    if let Some(forced) = any.downcast_ref::<ResumeGuardForcedDescr>() {
        return Some(&forced.inner);
    }
    any.downcast_ref::<ResumeGuardExcDescr>()
        .map(|exc| &exc.inner)
}

/// compile.py: ResumeAtPositionDescr(ResumeGuardDescr) — subclass
/// with no additional fields or method overrides. Type tag only.
///
/// In RPython, ResumeAtPositionDescr inherits all of ResumeGuardDescr's
/// fields (rd_numb, rd_consts, rd_virtuals, rd_pendingfields) and its
/// clone() method (which calls copy_all_attributes_from). The only
/// difference is the type tag used by compile_trace to decide
/// inline_short_preamble.
///
/// We model this as a newtype wrapping ResumeGuardDescr so that
/// clone_descr() produces a plain ResumeGuardDescr with resume data
/// preserved — matching RPython's inherited clone() behavior exactly.
#[derive(Debug)]
pub struct ResumeAtPositionDescr {
    inner: ResumeGuardDescr,
}

// Safety: same as ResumeGuardDescr (single-threaded JIT).
unsafe impl Send for ResumeAtPositionDescr {}
unsafe impl Sync for ResumeAtPositionDescr {}

impl majit_ir::Descr for ResumeAtPositionDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        // Hand out the inner ResumeGuardDescr so consumers that want to
        // read meta-side cells (e.g. recovery_layout) can downcast
        // uniformly across the subclass family — matching RPython's
        // `compile.py` `class ResumeAtPositionDescr(ResumeGuardDescr)`
        // shape where attribute access goes through the base.
        Some(&self.inner)
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_resume_at_position(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    // compile.py: inherited ResumeGuardDescr.clone() →
    // plain ResumeGuardDescr with copy_all_attributes_from(self).
    // Marker lost, resume data preserved.
    fn clone_descr(&self) -> Option<DescrRef> {
        self.inner.clone_descr()
    }
}

impl FailDescr for ResumeAtPositionDescr {
    fn handle_fail_class(&self) -> HandleFailClass {
        HandleFailClass::ResumeGuard
    }
    fn handle_fail_is_plain_resume_guard(&self) -> bool {
        true
    }
    fn fail_index(&self) -> u32 {
        // Per-trace key (see ResumeGuardDescr::fail_index).
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<majit_ir::NumberingRef>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn visit_rd_numb(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        self.inner.payload.visit_rd_numb(visitor);
    }
    fn rd_numb_ref(&self) -> Option<&majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_ref()
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.inner.payload.rd_consts_arc()
    }
    fn rd_consts_pool(&self) -> Option<&majit_ir::SharedConstPool> {
        self.inner.payload.rd_consts_pool()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<majit_ir::SharedConstPool>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn std::any::Any> {
        self.inner.payload.shaped_virtuals_any()
    }
    fn cache_shaped_virtuals(
        &self,
        value: Box<dyn std::any::Any>,
    ) -> Result<(), Box<dyn std::any::Any>> {
        self.inner.payload.cache_shaped_virtuals(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.inner.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.inner.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.inner.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        unsafe { *self.inner.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.inner.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.inner
            .status
            .fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.inner
            .status
            .fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.inner
            .status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.inner.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
    fn source_op_index(&self) -> Option<usize> {
        FailDescr::source_op_index(&self.inner)
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        FailDescr::set_source_op_index(&self.inner, source_op_index);
    }
    fn trace_info_any(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        FailDescr::trace_info_any(&self.inner)
    }
    fn set_trace_info_any(&self, info: Arc<dyn std::any::Any + Send + Sync>) {
        FailDescr::set_trace_info_any(&self.inner, info);
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        FailDescr::bridge_cache_addrs(&self.inner)
    }
    fn bridge_code_ptr(&self) -> usize {
        FailDescr::bridge_code_ptr(&self.inner)
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        FailDescr::store_bridge_caches(&self.inner, code_ptr, body_ptr);
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        FailDescr::bridge_dispatch_load(&self.inner)
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        FailDescr::bridge_dispatch_swap(&self.inner, new_ptr, drop_fn)
    }
    fn is_external_jump(&self) -> bool {
        FailDescr::is_external_jump(&self.inner)
    }
    fn target_descr(&self) -> Option<DescrRef> {
        FailDescr::target_descr(&self.inner)
    }
    fn set_external_jump_target(&self, target: DescrRef) {
        FailDescr::set_external_jump_target(&self.inner, target);
    }
}

/// Create a ResumeAtPositionDescr with auto-assigned fail_index, the
/// supplied `types`, and empty resume data.
pub fn make_resume_at_position_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeAtPositionDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            bridge_declined_terminally: AtomicBool::new(false),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: std::sync::atomic::AtomicBool::new(false),
            trace_info: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target: OnceLock::new(),
            bridge: std::sync::OnceLock::new(),
        },
    })
}

/// Create a ResumeAtPositionDescr with auto-assigned fail_index and
/// empty resume data + empty types. The optimizer's
/// `store_final_boxes_in_guard` mutates `types` in place via
/// `FailDescr::set_fail_arg_types` (preserving subtype + fail_index).
pub fn make_resume_at_position_descr() -> DescrRef {
    make_resume_at_position_descr_typed(Vec::new())
}

/// compile.py `AllVirtuals` — llopaque instance hidden in
/// `jf_savedata` so `handle_fail` can fish the virtuals
/// `handle_async_forcing` just materialized.
pub const ALL_VIRTUALS_VTABLE: usize = 0x414C_5652; // "ALVR"

const ALL_VIRTUALS_GC_TYPE_ID_UNSET: u32 = u32::MAX;
static ALL_VIRTUALS_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(ALL_VIRTUALS_GC_TYPE_ID_UNSET);

/// Publish the leaf type used by [`AllVirtuals::hide`].
pub fn set_all_virtuals_gc_type_id(type_id: u32) {
    ALL_VIRTUALS_GC_TYPE_ID.store(type_id, Ordering::Relaxed);
}

/// compile.py `class AllVirtuals(object)` with `llopaque = True`.
#[repr(C)]
pub struct AllVirtuals {
    pub super_: crate::virtualref::ObjectHeader,
    n_ptrs: usize,
    ptrs: *mut i64,
    n_ints: usize,
    ints: *mut i64,
}

impl AllVirtuals {
    /// compile.py `AllVirtuals(cache)` + `hide()`.
    pub fn hide(cache: (Vec<i64>, Vec<i64>)) -> majit_ir::GcRef {
        let (mut ptrs, mut ints) = cache;
        ptrs.shrink_to_fit();
        ints.shrink_to_fit();
        let n_ptrs = ptrs.len();
        let n_ints = ints.len();
        let ptrs = Box::into_raw(ptrs.into_boxed_slice()) as *mut i64;
        let ints = Box::into_raw(ints.into_boxed_slice()) as *mut i64;
        let value = AllVirtuals {
            super_: crate::virtualref::ObjectHeader {
                typeptr: ALL_VIRTUALS_VTABLE,
            },
            n_ptrs,
            ptrs,
            n_ints,
            ints,
        };
        let type_id = ALL_VIRTUALS_GC_TYPE_ID.load(Ordering::Relaxed);
        if type_id == ALL_VIRTUALS_GC_TYPE_ID_UNSET {
            // Tests that never publish a leaf type keep the object via
            // this leak; production registers the type and owns it
            // through the deadframe `jf_savedata` GCREF
            // (`compile.py AllVirtuals.hide` / `cpu.set_savedata_ref`).
            return majit_ir::GcRef(Box::into_raw(Box::new(value)) as usize);
        }
        let obj = majit_gc::alloc_oldgen_typed(type_id, std::mem::size_of::<AllVirtuals>());
        assert!(!obj.is_null(), "AllVirtuals old-gen allocation failed");
        // Upstream the instance is born young, so storing `cache` needs no
        // barrier. This one is born old and `ptrs` can name nursery objects
        // (`bh_newstr` allocates through `alloc_varsize_typed`), so remember it
        // before the store; a minor collection then traces it through
        // `custom_trace`.
        majit_gc::gc_write_barrier_managed(obj);
        unsafe { std::ptr::write(obj.0 as *mut AllVirtuals, value) };
        obj
    }

    /// Trace the off-heap `ptrs` slice as GCREF slots.
    ///
    /// # Safety
    /// `obj_addr` is an `AllVirtuals` payload the collector is tracing.
    pub unsafe fn custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
        let obj = obj_addr as *mut AllVirtuals;
        unsafe {
            let n = (*obj).n_ptrs;
            let ptrs = (*obj).ptrs;
            for i in 0..n {
                f(ptrs.add(i) as *mut majit_ir::GcRef);
            }
        }
    }

    /// Drop the off-heap `ptrs` / `ints` slices.
    ///
    /// # Safety
    /// `obj_addr` is an `AllVirtuals` payload about to be reclaimed,
    /// allocated by [`AllVirtuals::hide`].
    pub unsafe fn destructor(obj_addr: usize) {
        let obj = obj_addr as *mut AllVirtuals;
        unsafe {
            let _ptrs = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                (*obj).ptrs,
                (*obj).n_ptrs,
            ));
            let _ints = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                (*obj).ints,
                (*obj).n_ints,
            ));
        }
    }

    /// compile.py `AllVirtuals.show(gcref)`.
    pub fn show(gcref: majit_ir::GcRef) -> Option<(Vec<i64>, Vec<i64>)> {
        if gcref.is_null() {
            return None;
        }
        let obj = gcref.0 as *const AllVirtuals;
        if obj.is_null() {
            return None;
        }
        unsafe {
            if (*obj).super_.typeptr != ALL_VIRTUALS_VTABLE {
                return None;
            }
            let ptrs = if (*obj).n_ptrs == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts((*obj).ptrs, (*obj).n_ptrs).to_vec()
            };
            let ints = if (*obj).n_ints == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts((*obj).ints, (*obj).n_ints).to_vec()
            };
            Some((ptrs, ints))
        }
    }
}

/// compile.py: ResumeGuardForcedDescr(ResumeGuardDescr) — subtype
/// minted by `invent_fail_descr_for_op` for `GUARD_NOT_FORCED` /
/// `GUARD_NOT_FORCED_2`. Upstream attaches `metainterp_sd` /
/// `jitdriver_sd` via `_init` (compile.py) so
/// `handle_async_forcing` (compile.py) can call back into resume
/// during a residual call.
///
/// PYRE-ADAPTATION: pyre's forced-guard handling currently routes
/// through opcode checks (`pyjitpl.rs` GUARD_NOT_FORCED chain),
/// not the descr's `handle_fail`, so this subtype is tag-only.
/// `is_guard_forced()` returns true so descr-keyed dispatch can
/// migrate later without reshaping the optimizer call site.
#[derive(Debug)]
pub struct ResumeGuardForcedDescr {
    inner: ResumeGuardDescr,
}

unsafe impl Send for ResumeGuardForcedDescr {}
unsafe impl Sync for ResumeGuardForcedDescr {}

impl majit_ir::Descr for ResumeGuardForcedDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_guard_forced(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py ResumeGuardDescr.clone() — `ResumeGuardForcedDescr`
    /// inherits the base implementation (no override at compile.py:939+),
    /// so cloning produces a plain `ResumeGuardDescr` with resume attributes
    /// copied over via `copy_all_attributes_from`. The Forced subtype tag
    /// is intentionally dropped.
    fn clone_descr(&self) -> Option<DescrRef> {
        self.inner.clone_descr()
    }
}

impl ResumeGuardForcedDescr {
    /// compile.py `ResumeGuardForcedDescr.force_now(cpu, token)`.
    ///
    /// Upstream wraps `cpu.force` + `handle_async_forcing` in
    /// `_stack_criticalcode_start/stop`. The body is supplied by the
    /// caller (`MetaInterp::force_virtualizable_token_with_allocator`)
    /// because this descr does not own `metainterp_sd` / `jitdriver_sd`.
    pub fn force_now(body: impl FnOnce()) {
        let _cc = crate::CriticalCodeGuard::enter();
        body();
    }
}

impl FailDescr for ResumeGuardForcedDescr {
    fn handle_fail_class(&self) -> HandleFailClass {
        HandleFailClass::Forced
    }
    fn fail_index(&self) -> u32 {
        // Per-trace key (see ResumeGuardDescr::fail_index).
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<majit_ir::NumberingRef>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn visit_rd_numb(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        self.inner.payload.visit_rd_numb(visitor);
    }
    fn rd_numb_ref(&self) -> Option<&majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_ref()
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.inner.payload.rd_consts_arc()
    }
    fn rd_consts_pool(&self) -> Option<&majit_ir::SharedConstPool> {
        self.inner.payload.rd_consts_pool()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<majit_ir::SharedConstPool>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn std::any::Any> {
        self.inner.payload.shaped_virtuals_any()
    }
    fn cache_shaped_virtuals(
        &self,
        value: Box<dyn std::any::Any>,
    ) -> Result<(), Box<dyn std::any::Any>> {
        self.inner.payload.cache_shaped_virtuals(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.inner.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.inner.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.inner.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        unsafe { *self.inner.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.inner.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.inner
            .status
            .fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.inner
            .status
            .fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.inner
            .status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.inner.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
    fn source_op_index(&self) -> Option<usize> {
        FailDescr::source_op_index(&self.inner)
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        FailDescr::set_source_op_index(&self.inner, source_op_index);
    }
    fn trace_info_any(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        FailDescr::trace_info_any(&self.inner)
    }
    fn set_trace_info_any(&self, info: Arc<dyn std::any::Any + Send + Sync>) {
        FailDescr::set_trace_info_any(&self.inner, info);
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        FailDescr::bridge_cache_addrs(&self.inner)
    }
    fn bridge_code_ptr(&self) -> usize {
        FailDescr::bridge_code_ptr(&self.inner)
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        FailDescr::store_bridge_caches(&self.inner, code_ptr, body_ptr);
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        FailDescr::bridge_dispatch_load(&self.inner)
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        FailDescr::bridge_dispatch_swap(&self.inner, new_ptr, drop_fn)
    }
    fn is_external_jump(&self) -> bool {
        FailDescr::is_external_jump(&self.inner)
    }
    fn target_descr(&self) -> Option<DescrRef> {
        FailDescr::target_descr(&self.inner)
    }
    fn set_external_jump_target(&self, target: DescrRef) {
        FailDescr::set_external_jump_target(&self.inner, target);
    }
}

/// Create a ResumeGuardForcedDescr with auto-assigned fail_index, the
/// supplied `types`, and empty resume data.
pub fn make_resume_guard_forced_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardForcedDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            bridge_declined_terminally: AtomicBool::new(false),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: std::sync::atomic::AtomicBool::new(false),
            trace_info: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target: OnceLock::new(),
            bridge: std::sync::OnceLock::new(),
        },
    })
}

/// compile.py: ResumeGuardExcDescr(ResumeGuardDescr) — subtype
/// minted by `invent_fail_descr_for_op` for `GUARD_EXCEPTION` /
/// `GUARD_NO_EXCEPTION`. Upstream uses `pass` to make it a tag-only
/// subclass; `handle_fail` routes the exception path off this tag.
#[derive(Debug)]
pub struct ResumeGuardExcDescr {
    inner: ResumeGuardDescr,
}

unsafe impl Send for ResumeGuardExcDescr {}
unsafe impl Sync for ResumeGuardExcDescr {}

impl majit_ir::Descr for ResumeGuardExcDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        // Hand out the inner ResumeGuardDescr — see ResumeAtPositionDescr.
        Some(&self.inner)
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_guard_exc(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py `class ResumeGuardExcDescr(ResumeGuardDescr): pass`
    /// — no clone() override, so inheriting compile.py
    /// `ResumeGuardDescr.clone()` produces a plain `ResumeGuardDescr` with
    /// resume attributes copied via `copy_all_attributes_from`. The Exc
    /// subtype tag is intentionally dropped.
    fn clone_descr(&self) -> Option<DescrRef> {
        self.inner.clone_descr()
    }
}

impl FailDescr for ResumeGuardExcDescr {
    fn handle_fail_class(&self) -> HandleFailClass {
        HandleFailClass::Exc
    }
    fn fail_index(&self) -> u32 {
        // Per-trace key (see ResumeGuardDescr::fail_index).
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<majit_ir::NumberingRef>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn visit_rd_numb(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        self.inner.payload.visit_rd_numb(visitor);
    }
    fn rd_numb_ref(&self) -> Option<&majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_ref()
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.inner.payload.rd_consts_arc()
    }
    fn rd_consts_pool(&self) -> Option<&majit_ir::SharedConstPool> {
        self.inner.payload.rd_consts_pool()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<majit_ir::SharedConstPool>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn std::any::Any> {
        self.inner.payload.shaped_virtuals_any()
    }
    fn cache_shaped_virtuals(
        &self,
        value: Box<dyn std::any::Any>,
    ) -> Result<(), Box<dyn std::any::Any>> {
        self.inner.payload.cache_shaped_virtuals(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.inner.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.inner.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.inner.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        unsafe { *self.inner.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.inner.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.inner
            .status
            .fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.inner
            .status
            .fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.inner
            .status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.inner.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
    fn source_op_index(&self) -> Option<usize> {
        FailDescr::source_op_index(&self.inner)
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        FailDescr::set_source_op_index(&self.inner, source_op_index);
    }
    fn trace_info_any(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        FailDescr::trace_info_any(&self.inner)
    }
    fn set_trace_info_any(&self, info: Arc<dyn std::any::Any + Send + Sync>) {
        FailDescr::set_trace_info_any(&self.inner, info);
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        FailDescr::bridge_cache_addrs(&self.inner)
    }
    fn bridge_code_ptr(&self) -> usize {
        FailDescr::bridge_code_ptr(&self.inner)
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        FailDescr::store_bridge_caches(&self.inner, code_ptr, body_ptr);
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        FailDescr::bridge_dispatch_load(&self.inner)
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        FailDescr::bridge_dispatch_swap(&self.inner, new_ptr, drop_fn)
    }
    fn is_external_jump(&self) -> bool {
        FailDescr::is_external_jump(&self.inner)
    }
    fn target_descr(&self) -> Option<DescrRef> {
        FailDescr::target_descr(&self.inner)
    }
    fn set_external_jump_target(&self, target: DescrRef) {
        FailDescr::set_external_jump_target(&self.inner, target);
    }
}

/// Create a ResumeGuardExcDescr with auto-assigned fail_index, the
/// supplied `types`, and empty resume data.
pub fn make_resume_guard_exc_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardExcDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            payload: RdPayload::empty(),
            vector_info: UnsafeCell::new(None),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            bridge_declined_terminally: AtomicBool::new(false),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: std::sync::atomic::AtomicBool::new(false),
            trace_info: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target: OnceLock::new(),
            bridge: std::sync::OnceLock::new(),
        },
    })
}

/// compile.py: `ResumeGuardCopiedDescr(prev)` —
/// shared-resume subtype minted by `invent_fail_descr_for_op` when
/// `_copy_resume_data_from` shares a donor guard's resume data.
/// `get_resumestorage()` (compile.py) returns the donor
/// `ResumeGuardDescr` so reads chase through to the original
/// `rd_numb` / `rd_consts` / `rd_virtuals` / `rd_pendingfields`.
///
/// Reads route through `prev_descr()` (compile.py:849
/// `get_resumestorage(): return prev`); every `rd_*` getter on
/// `FailDescr` chases the donor stored in `prev` so a copied descr
/// has no owned resume payload of its own.
#[derive(Debug)]
pub struct ResumeGuardCopiedDescr {
    fail_index: u32,
    /// compile.py: `assert isinstance(prev, ResumeGuardDescr)`.
    /// pyre keeps the donor as a `DescrRef` so chained sharing
    /// (`prev.prev` etc.) can be walked uniformly through
    /// `prev_descr()` until a non-copied descr is reached.
    ///
    /// `compile.py ResumeGuardCopiedDescr.copy_all_attributes_from`
    /// mutates `self.prev = other.prev` in place, preserving the
    /// receiver's identity (`fail_index` / status).  Pyre wraps it in
    /// `UnsafeCell` so the optimizer-side helper can swap the donor
    /// pointer through `&self` without minting a new Arc — same
    /// single-threaded contract used for the `rd_*` cells.
    prev: UnsafeCell<DescrRef>,
    /// history.py:125 `_attrs_ = ('adr_jump_offset', 'rd_locs',
    /// 'rd_loop_token', 'rd_vector_info')` — `rd_vector_info` lives on
    /// `AbstractFailDescr` itself, not on the resume storage.  Copied
    /// descrs share their donor's resume payload via `prev`, but each
    /// guard owns its own vector-info chain (history.py:143
    /// `attach_vector_info` writes `self.rd_vector_info`).
    vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// `history.py` `AbstractFailDescr._attrs_` — copied descrs
    /// share their donor's *resume* payload via `prev` (`compile.py:849
    /// get_resumestorage(): return prev`), but `adr_jump_offset` is
    /// written per-fail at codegen time (`assembler.py:849`), so each
    /// copied descr owns its own slot.  Same scoping as `vector_info`
    /// above (history.py:143 writes `self.rd_vector_info`, not
    /// `prev.rd_vector_info`).
    adr_jump_offset: UnsafeCell<usize>,
    /// `history.py` `_attrs_` `rd_locs` — same per-fail scoping
    /// as `adr_jump_offset`.
    rd_locs: UnsafeCell<majit_ir::RdLocs>,
    /// `compile.py` `AbstractResumeGuardDescr._attrs_` `status` —
    /// each copied descr carries its own status (the copied receiver is
    /// retraced independently of the donor).
    status: AtomicU64,
    /// `compile.py` `descr.rd_loop_token = clt`.  Copied descrs are
    /// stamped per-guard by the same `record_loop_or_bridge` walker
    /// (`compile.py isinstance(descr, ResumeDescr)` covers
    /// `ResumeGuardCopiedDescr` — `_attrs_` lists `rd_loop_token`
    /// directly on `AbstractFailDescr` at history.py, owned by
    /// the receiver, not chased through `prev`).
    rd_loop_token_clt: UnsafeCell<Option<std::sync::Arc<CompiledLoopToken>>>,
    /// Pyre-only owning-trace identifier — same role as on
    /// `ResumeGuardDescr`. Stamped by `record_loop_or_bridge`
    /// (`compile.py:185-186` walker) since copied descrs are equally
    /// owned by exactly one trace.
    trace_id: AtomicU64,
    /// Pyre-only per-trace fail-index — same role as on
    /// `ResumeGuardDescr`. Stamped by `build_guard_metadata`.
    fail_index_per_trace: AtomicU32,
    /// Pyre-only per-emission slot: codegen-time trace-op index.
    /// Classified per-emission alongside `history.py:132
    /// AbstractFailDescr._attrs_` `rd_locs` / `adr_jump_offset`
    /// (`assembler.py:279` writes onto each emitted faildescr directly,
    /// never chasing `prev`).  Owned per copied descr so multiple
    /// copies of a single donor (optimizer.py:691 / optimizeopt/mod.rs)
    /// do not clobber each other's op indices.
    source_op_index: UnsafeCell<Option<usize>>,
    /// Pyre-only per-emission slot: GC-root classification for force-
    /// token producer slots.  PyPy bakes the equivalent GC map into
    /// machine code via `assembler.py:write_failure_recovery_description`
    /// per emission; cranelift has no inline encoding so the slot list
    /// lives on the descr.  Same per-emission scoping as
    /// `source_op_index` / `rd_locs` — owned per copied descr.
    /// Pyre-only per-emission slot: this guard is the eval-breaker word's
    /// back-edge poll.  Same per-emission scoping as `source_op_index`; a
    /// copy guards the same back edge as its donor, so the optimizer stamps
    /// each copy as it emits it.
    back_edge_poll: std::sync::atomic::AtomicBool,
    /// Pyre-only per-emission `CompiledTraceInfo` cell.  Same shape
    /// and atomic-swap discipline as
    /// `ResumeGuardDescr::trace_info`.  Per-emission because
    /// `record_loop_or_bridge` (compile.py) stamps the
    /// loop-token-equivalent metadata onto each emitted descr
    /// individually; the cell is reclaimed in `Drop`.
    trace_info: std::sync::atomic::AtomicPtr<CompiledTraceInfo>,
    /// Per-emission cranelift bridge cells; see
    /// `ResumeGuardDescr::bridge`.  Per-emission because each copied descr
    /// can have its own bridge attached (`compile.py handle_fail` applies
    /// to both ResumeGuardDescr and ResumeGuardCopiedDescr).
    bridge: std::sync::OnceLock<Box<majit_backend::BridgeDispatchCells>>,
    /// Pyre-only per-emission cranelift cross-loop JUMP target slot.
    /// Mirrors `ResumeGuardDescr::external_jump_target`; per-emission
    /// because each copied descr can be the JUMP exit for an
    /// independently retraced loop-version peel, even though the
    /// donor (`prev`) carries shared resume payload.  Membership
    /// (`OnceLock.get().is_some()`) is the `is_external_jump`
    /// predicate.  Write-once at codegen finalisation.
    pub external_jump_target: std::sync::OnceLock<DescrRef>,
}

unsafe impl Send for ResumeGuardCopiedDescr {}
unsafe impl Sync for ResumeGuardCopiedDescr {}

impl Drop for ResumeGuardCopiedDescr {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        // Reclaim any published `Arc<CompiledTraceInfo>`; mirrors
        // `ResumeGuardDescr::drop`.  Swap-to-null first so a concurrent
        // reader either bumps the strong count on the live pointer or
        // observes null.
        let ptr = self.trace_info.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            // Safety: produced by `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info_any`.
            unsafe { drop(Arc::from_raw(ptr as *const CompiledTraceInfo)) };
        }
    }
}

impl ResumeGuardCopiedDescr {
    /// Read the current `prev` Arc.
    fn prev(&self) -> &DescrRef {
        // Safety: single-threaded JIT, no concurrent writers.
        unsafe { &*self.prev.get() }
    }
    /// `compile.py self.prev = other.prev` — overwrite the donor
    /// pointer in place. Identity (`fail_index` / subtype tag) stays.
    fn set_prev(&self, prev: DescrRef) {
        // Safety: single-threaded JIT, no concurrent readers.
        unsafe { *self.prev.get() = prev }
    }

    /// Mirror `ResumeGuardDescr::set_external_jump_target` (-Tβ8):
    /// publish the cross-loop JUMP target `DescrRef` into the
    /// write-once slot.  Each copied descr carries its own slot so
    /// distinct loop-version peels sharing a donor (compile.py:840
    /// `prev`) can each be the JUMP exit for their independent peel.
    pub fn set_external_jump_target(&self, target: DescrRef) {
        self.external_jump_target
            .set(target)
            .expect("external_jump_target already published");
    }
}

impl majit_ir::Descr for ResumeGuardCopiedDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_resume_guard_copied(&self) -> bool {
        true
    }
    fn prev_descr(&self) -> Option<DescrRef> {
        Some(self.prev().clone())
    }
    fn set_prev_descr(&self, prev: DescrRef) {
        self.set_prev(prev);
    }
    /// compile.py: `clone()` constructs a fresh
    /// `ResumeGuardCopiedDescr(self.prev)` — identity on `prev` is
    /// preserved (Arc share), only `fail_index` is fresh.
    fn clone_descr(&self) -> Option<DescrRef> {
        // history.py `rd_vector_info = None` is the class default;
        // `clone()` does not copy it (compile.py only forwards
        // `prev`).  Mint a fresh empty chain.
        Some(Arc::new(ResumeGuardCopiedDescr {
            fail_index: alloc_fail_index(),
            prev: UnsafeCell::new(self.prev().clone()),
            vector_info: UnsafeCell::new(None),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: std::sync::atomic::AtomicBool::new(false),
            trace_info: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            bridge: std::sync::OnceLock::new(),
            external_jump_target: std::sync::OnceLock::new(),
        }))
    }
}

impl FailDescr for ResumeGuardCopiedDescr {
    fn handle_fail_class(&self) -> HandleFailClass {
        HandleFailClass::Copied
    }
    fn fail_index(&self) -> u32 {
        // Per-trace key (see ResumeGuardDescr::fail_index).  Global id
        // remains available via Descr::index() / get_descr_index().
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    /// compile.py `get_resumestorage(): return prev`: reads chase
    /// to the donor.  The `fail_arg_types` slot is shared too —
    /// upstream stores the type list on the donor `ResumeGuardDescr`,
    /// which `prev` references.
    fn fail_arg_types(&self) -> &[Type] {
        self.prev()
            .as_fail_descr()
            .map(|fd| fd.fail_arg_types())
            .unwrap_or(&[])
    }
    /// `_copy_resume_data_from` does not call
    /// `store_final_boxes_in_guard`, so the optimizer never invokes
    /// `set_fail_arg_types` on a copied descr.  Match RPython's
    /// implicit invariant by panicking — a setter that wrote
    /// through to `prev` would silently mutate the donor's type
    /// vector and is never the desired behavior.
    fn set_fail_arg_types(&self, _types: Vec<Type>) {
        panic!(
            "set_fail_arg_types invoked on a ResumeGuardCopiedDescr — \
             RPython optimizer.py:724 only allows ResumeGuardDescr; \
             copied descrs share their donor's type vector via prev"
        );
    }
    /// history.py `AbstractFailDescr.attach_vector_info`: writes
    /// `self.rd_vector_info`, never `self.prev`.  `prev` is for resume
    /// storage only (compile.py `get_resumestorage`); vector info
    /// lives on the copied descr itself.
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }

    // compile.py `get_resumestorage(): return prev` — every rd_*
    // read chases through to the donor descr.  Setters panic for the
    // same reason `set_fail_arg_types` does: `_copy_resume_data_from`
    // never finalizes a copied descr; mutation must go through the
    // donor's own `ResumeGuardDescr`.
    fn rd_numb(&self) -> Option<&[u8]> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_numb())
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_numb_arc())
    }
    fn visit_rd_numb(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        if let Some(fd) = self.prev().as_fail_descr() {
            fd.visit_rd_numb(visitor);
        }
    }
    fn rd_numb_ref(&self) -> Option<&majit_ir::NumberingRef> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_numb_ref())
    }
    fn set_rd_numb(&self, _value: Option<Vec<u8>>) {
        panic!(
            "set_rd_numb invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_consts())
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_consts_arc())
    }
    fn rd_consts_pool(&self) -> Option<&majit_ir::SharedConstPool> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_consts_pool())
    }
    fn set_rd_consts(&self, _value: Option<Vec<Const>>) {
        panic!(
            "set_rd_consts invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.prev().as_fail_descr().and_then(|fd| fd.rd_virtuals())
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_virtuals_arc())
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn std::any::Any> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.shaped_virtuals_any())
    }
    fn cache_shaped_virtuals(
        &self,
        value: Box<dyn std::any::Any>,
    ) -> Result<(), Box<dyn std::any::Any>> {
        match self.prev().as_fail_descr() {
            Some(fd) => fd.cache_shaped_virtuals(value),
            None => Err(value),
        }
    }
    fn set_rd_virtuals(&self, _value: Option<Vec<Rc<RdVirtualInfo>>>) {
        panic!(
            "set_rd_virtuals invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_pendingfields())
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.prev()
            .as_fail_descr()
            .and_then(|fd| fd.rd_pendingfields_arc())
    }
    fn set_rd_pendingfields(&self, _value: Option<Vec<GuardPendingFieldEntry>>) {
        panic!(
            "set_rd_pendingfields invoked on a ResumeGuardCopiedDescr — \
             upstream optimizer.py:728 only finalizes ResumeGuardDescr"
        );
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.adr_jump_offset.get() = offset };
    }
    /// `store_info_on_descr` writes `guardtok.faildescr.rd_locs`
    /// (`llsupport/assembler.py`), so a backend that emits machine code for
    /// this guard leaves its own physical positions here.  The frontend's
    /// identity-with-holes layout is resume data instead, and
    /// `_copy_resume_data_from` never runs `store_final_boxes_in_guard` on a
    /// copied descr, so it is written only on the donor.  Read through to the
    /// donor when no backend has written one, the same way `fail_arg_types`,
    /// `rd_numb` and `rd_consts` chase `get_resumestorage(): return prev`
    /// (`compile.py`).  Without it the hole mask is empty on every
    /// backend that keeps failargs in their logical slots, and both consumers
    /// silently take their unmasked branch.
    fn rd_locs(&self) -> &[u16] {
        let own = unsafe { &*self.rd_locs.get() };
        if !own.is_empty() {
            return own;
        }
        self.prev()
            .as_fail_descr()
            .map(|fd| fd.rd_locs())
            .unwrap_or(&[])
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        unsafe { *self.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.status.fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.status.fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.rd_loop_token_clt.get() = Some(typed) };
    }
    /// Per-emission `source_op_index` (see field comment).  Owned per
    /// copied descr — each copy in `optimizeopt/mod.rs`'s `emit_guard_operation`
    /// records its own trace-op origin, matching how
    /// `assembler.py:279` writes `rd_locs` onto each emitted descr
    /// directly.
    fn source_op_index(&self) -> Option<usize> {
        unsafe { *self.source_op_index.get() }
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        unsafe { *self.source_op_index.get() = Some(source_op_index) };
    }
    /// Per-emission, and deliberately NOT
    /// chased through `prev`: the classification describes this guard's own
    /// condition chain, which a sharer does not inherit from its donor.  A
    /// copied descr always answers `false`, and that is correct twice over.
    /// A poll guard can never be the *sharer*: sharing requires
    /// `!op.has_descr() && op.rd_resume_position < 0` (optimizer.rs
    /// `emit_guard_operation`, mirrored in optimizeopt/mod.rs), whereas the
    /// poll is emitted by `close_loop_args_at` through `generate_guard`,
    /// which captures resume data and so always carries a resume position.
    /// And when a poll guard is the *donor*, the sharer is some descrless
    /// optimizer-created follow-up guard that is not itself a poll, so
    /// reading through `prev` would misreport it as one.
    fn is_back_edge_poll(&self) -> bool {
        self.back_edge_poll
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set_back_edge_poll(&self) {
        self.back_edge_poll
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// Per-emission `trace_info` (see field comment).  Same
    /// atomic-swap discipline as `ResumeGuardDescr::set_trace_info` —
    /// owning Arc reclaimed by `Drop`.
    fn trace_info_any(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        let ptr = self.trace_info.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // Safety: produced by `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info_any`.
            unsafe {
                Arc::increment_strong_count(ptr as *const CompiledTraceInfo);
                let arc = Arc::from_raw(ptr as *const CompiledTraceInfo);
                Some(arc as Arc<dyn std::any::Any + Send + Sync>)
            }
        }
    }
    fn set_trace_info_any(&self, info: Arc<dyn std::any::Any + Send + Sync>) {
        let typed: Arc<CompiledTraceInfo> = info
            .downcast::<CompiledTraceInfo>()
            .expect("set_trace_info_any expected Arc<CompiledTraceInfo>");
        let new_ptr = Arc::into_raw(typed) as *mut CompiledTraceInfo;
        let old_ptr = self.trace_info.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            unsafe { drop(Arc::from_raw(old_ptr as *const CompiledTraceInfo)) };
        }
    }
    /// Per-emission cranelift bridge cells; same contract as
    /// `ResumeGuardDescr`.
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        Some(
            self.bridge
                .get_or_init(majit_backend::BridgeDispatchCells::new)
                .cache_addrs(),
        )
    }
    fn bridge_code_ptr(&self) -> usize {
        self.bridge.get().map_or(0, |cells| cells.code_ptr())
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        self.bridge
            .get_or_init(majit_backend::BridgeDispatchCells::new)
            .store_caches(code_ptr, body_ptr)
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        self.bridge
            .get()
            .map_or(std::ptr::null_mut(), |cells| cells.dispatch_load())
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        self.bridge
            .get_or_init(majit_backend::BridgeDispatchCells::new)
            .dispatch_swap(new_ptr, drop_fn)
    }

    /// Mirror `ResumeGuardDescr::is_external_jump`
    /// (`resume_guard_descr.rs`): membership in the per-emission
    /// `external_jump_target` slot IS the cross-loop-JUMP predicate.
    fn is_external_jump(&self) -> bool {
        self.external_jump_target.get().is_some()
    }

    /// Mirror `ResumeGuardDescr::target_descr` (resume_guard_descr.rs):
    /// when this copied descr is the synthesised cross-loop JUMP exit,
    /// surface the target `DescrRef` the dispatcher re-enters via.
    fn target_descr(&self) -> Option<DescrRef> {
        self.external_jump_target.get().cloned()
    }
    fn set_external_jump_target(&self, target: DescrRef) {
        ResumeGuardCopiedDescr::set_external_jump_target(self, target);
    }
}

/// compile.py: `class ResumeGuardCopiedExcDescr(ResumeGuardCopiedDescr): pass`
/// — exception variant of the shared-resume descr, minted by
/// `invent_fail_descr_for_op` for `GUARD_EXCEPTION` /
/// `GUARD_NO_EXCEPTION` on the sharing path.
#[derive(Debug)]
pub struct ResumeGuardCopiedExcDescr {
    inner: ResumeGuardCopiedDescr,
}

unsafe impl Send for ResumeGuardCopiedExcDescr {}
unsafe impl Sync for ResumeGuardCopiedExcDescr {}

impl majit_ir::Descr for ResumeGuardCopiedExcDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_resume_guard_copied(&self) -> bool {
        true
    }
    fn is_guard_exc(&self) -> bool {
        true
    }
    fn prev_descr(&self) -> Option<DescrRef> {
        Some(self.inner.prev().clone())
    }
    fn set_prev_descr(&self, prev: DescrRef) {
        self.inner.set_prev(prev);
    }
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(ResumeGuardCopiedExcDescr {
            inner: ResumeGuardCopiedDescr {
                fail_index: alloc_fail_index(),
                prev: UnsafeCell::new(self.inner.prev().clone()),
                vector_info: UnsafeCell::new(None),
                adr_jump_offset: UnsafeCell::new(0),
                rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
                status: AtomicU64::new(0),
                rd_loop_token_clt: UnsafeCell::new(None),
                trace_id: AtomicU64::new(0),
                fail_index_per_trace: AtomicU32::new(0),
                source_op_index: UnsafeCell::new(None),
                back_edge_poll: std::sync::atomic::AtomicBool::new(false),
                trace_info: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
                bridge: std::sync::OnceLock::new(),
                external_jump_target: std::sync::OnceLock::new(),
            },
        }))
    }
}

impl FailDescr for ResumeGuardCopiedExcDescr {
    fn handle_fail_class(&self) -> HandleFailClass {
        HandleFailClass::Exc
    }
    fn fail_index(&self) -> u32 {
        // Per-trace key (see ResumeGuardDescr::fail_index).
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        self.inner.fail_arg_types()
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        self.inner.set_fail_arg_types(types)
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        self.inner.attach_vector_info(info)
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        self.inner.vector_info()
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        self.inner.replace_vector_info(chain)
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.inner.rd_numb_arc()
    }
    fn visit_rd_numb(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        self.inner.visit_rd_numb(visitor);
    }
    fn rd_numb_ref(&self) -> Option<&majit_ir::NumberingRef> {
        self.inner.rd_numb_ref()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.set_rd_numb(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.inner.rd_consts_arc()
    }
    fn rd_consts_pool(&self) -> Option<&majit_ir::SharedConstPool> {
        self.inner.rd_consts_pool()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.set_rd_consts(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.set_rd_virtuals(value)
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn std::any::Any> {
        FailDescr::shaped_virtuals_any(&self.inner)
    }
    fn cache_shaped_virtuals(
        &self,
        value: Box<dyn std::any::Any>,
    ) -> Result<(), Box<dyn std::any::Any>> {
        FailDescr::cache_shaped_virtuals(&self.inner, value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.set_rd_pendingfields(value)
    }
    fn adr_jump_offset(&self) -> usize {
        self.inner.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.inner.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.inner.rd_locs()
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        self.inner.set_rd_locs(locs);
    }
    fn get_status(&self) -> u64 {
        self.inner.get_status()
    }
    fn start_compiling(&self) {
        self.inner.start_compiling();
    }
    fn done_compiling(&self) {
        self.inner.done_compiling();
    }
    fn store_hash(&self, hash: u64) {
        self.inner.store_hash(hash);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        self.inner.make_a_counter_per_value(index, type_tag);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        self.inner.rd_loop_token_clt()
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        self.inner.set_rd_loop_token_clt(clt)
    }
    fn source_op_index(&self) -> Option<usize> {
        self.inner.source_op_index()
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        self.inner.set_source_op_index(source_op_index);
    }
    fn is_back_edge_poll(&self) -> bool {
        self.inner.is_back_edge_poll()
    }
    fn set_back_edge_poll(&self) {
        self.inner.set_back_edge_poll();
    }
    fn trace_info_any(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.inner.trace_info_any()
    }
    fn set_trace_info_any(&self, info: Arc<dyn std::any::Any + Send + Sync>) {
        self.inner.set_trace_info_any(info);
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        self.inner.bridge_cache_addrs()
    }
    fn bridge_code_ptr(&self) -> usize {
        self.inner.bridge_code_ptr()
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        self.inner.store_bridge_caches(code_ptr, body_ptr);
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        self.inner.bridge_dispatch_load()
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        self.inner.bridge_dispatch_swap(new_ptr, drop_fn)
    }
    fn is_external_jump(&self) -> bool {
        FailDescr::is_external_jump(&self.inner)
    }
    fn target_descr(&self) -> Option<DescrRef> {
        FailDescr::target_descr(&self.inner)
    }
    fn set_external_jump_target(&self, target: DescrRef) {
        FailDescr::set_external_jump_target(&self.inner, target);
    }
}

/// Mint a `ResumeGuardCopiedDescr` whose `get_resumestorage()` chases
/// back to `prev`.  `prev` must already carry the donor's
/// `fail_arg_types` (RPython invariant: copied descrs share the
/// donor's type vector via `get_resumestorage`).
///
/// compile.py:835-838 `ResumeGuardCopiedDescr.__init__`:
///   `assert isinstance(prev, ResumeGuardDescr)` —
/// the donor must be a head-of-chain ResumeGuardDescr (or its
/// subclasses), never another ResumeGuardCopiedDescr.  Two-hop
/// chasing would be silent in pyre (`prev.prev` returns the head
/// anyway) but masks real bugs in the optimizer's sharing path.
pub fn make_resume_guard_copied_descr(prev: DescrRef) -> DescrRef {
    // compile.py:837-838 ResumeGuardCopiedDescr.__init__:
    //   assert isinstance(prev, ResumeGuardDescr)
    // The donor must be a head-of-chain ResumeGuardDescr (or any
    // subclass thereof: ResumeAtPositionDescr / ResumeGuardForcedDescr /
    // ResumeGuardExcDescr / CompileLoopVersionDescr).  Reject siblings
    // (ResumeGuardCopiedDescr itself) and unrelated FailDescr subtypes
    // (`SimpleFailDescr`) — the descr-side rd_* readers chase `prev` at
    // resume time and would observe garbage if prev cannot carry resume
    // data.
    assert!(
        prev.is_resume_guard(),
        "compile.py:838 assert isinstance(prev, ResumeGuardDescr): \
         donor must be a ResumeGuardDescr subclass (got descr_index={:?})",
        prev.index()
    );
    Arc::new(ResumeGuardCopiedDescr {
        fail_index: alloc_fail_index(),
        prev: UnsafeCell::new(prev),
        vector_info: UnsafeCell::new(None),
        adr_jump_offset: UnsafeCell::new(0),
        rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
        status: AtomicU64::new(0),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
        source_op_index: UnsafeCell::new(None),
        back_edge_poll: std::sync::atomic::AtomicBool::new(false),
        trace_info: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        bridge: std::sync::OnceLock::new(),
        external_jump_target: std::sync::OnceLock::new(),
    })
}

/// Mint a `ResumeGuardCopiedExcDescr` for the GUARD_EXCEPTION /
/// GUARD_NO_EXCEPTION sharing path.
///
/// compile.py `ResumeGuardCopiedExcDescr` inherits
/// `ResumeGuardCopiedDescr.__init__`, so the same
/// `isinstance(prev, ResumeGuardDescr)` invariant applies.
pub fn make_resume_guard_copied_exc_descr(prev: DescrRef) -> DescrRef {
    // compile.py `class ResumeGuardCopiedExcDescr(...)` inherits
    // `ResumeGuardCopiedDescr.__init__`; same `isinstance(prev,
    // ResumeGuardDescr)` invariant.
    assert!(
        prev.is_resume_guard(),
        "compile.py:838 assert isinstance(prev, ResumeGuardDescr): \
         ResumeGuardCopiedExcDescr donor must be a ResumeGuardDescr \
         subclass (got descr_index={:?})",
        prev.index()
    );
    Arc::new(ResumeGuardCopiedExcDescr {
        inner: ResumeGuardCopiedDescr {
            fail_index: alloc_fail_index(),
            prev: UnsafeCell::new(prev),
            vector_info: UnsafeCell::new(None),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: std::sync::atomic::AtomicBool::new(false),
            trace_info: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            bridge: std::sync::OnceLock::new(),
            external_jump_target: std::sync::OnceLock::new(),
        },
    })
}

/// `compile.py AbstractResumeGuardDescr.get_resumestorage`: the descr that
/// actually holds this guard's resume data.
///
/// `ResumeGuardDescr.get_resumestorage` returns self (compile.py);
/// `ResumeGuardCopiedDescr`'s returns `prev` (compile.py), the donor
/// guard whose resume data it shares. The distinction is load-bearing wherever
/// upstream tests the *storage* rather than the guard: `handle_guard_failure`
/// takes `key = resumedescr.get_resumestorage()` and arms
/// `seen_loop_header_for_jdindex` from `isinstance(key,
/// ResumeAtPositionDescr)`, so a copied guard sharing a short-preamble donor
/// counts as one. `compile_trace` deliberately does not resolve — it tests
/// `resumekey` itself (compile.py:1040) — so callers must pick.
pub fn get_resumestorage(descr: &DescrRef) -> DescrRef {
    if descr.is_resume_guard_copied() {
        descr
            .prev_descr()
            .expect("compile.py:849-850 ResumeGuardCopiedDescr.get_resumestorage requires prev")
    } else {
        descr.clone()
    }
}

/// `compile.py ResumeGuardDescr.copy_all_attributes_from` +
/// `compile.py ResumeGuardCopiedDescr.copy_all_attributes_from`
/// dispatched on the receiver's variant.  Mutates `my_descr` in place;
/// the receiver's identity (`fail_index` / status / subtype tag) is
/// always preserved.
///
/// Used by `optimizer.py replace_guard_op` and
/// `guard.py inhert_attributes` — both call
/// `new_descr.copy_all_attributes_from(old_descr)` on a guard whose
/// own descr is already in place.
///
/// `Plain` self (default `ResumeGuardDescr`, including subclasses
/// `ResumeGuardExcDescr` / `ResumeAtPositionDescr` / `CompileLoopVersionDescr`
/// / `ResumeGuardForcedDescr`): copy `rd_numb` / `rd_consts` / `rd_virtuals`
/// / `rd_pendingfields` / `rd_vector_info` from the donor onto self via
/// descr-side setters.  `donor.rd_*()` chases through
/// `ResumeGuardCopiedDescr.prev` automatically (`compile.py other =
/// other.get_resumestorage()`).
///
/// `Copied` self (`ResumeGuardCopiedDescr` / `ResumeGuardCopiedExcDescr`):
/// `compile.py` overwrites `self.prev = other.prev` in place.
/// Pyre stores `prev` in `UnsafeCell<DescrRef>` and exposes
/// `set_prev_descr(&self, prev)` so the swap preserves the receiver's
/// `fail_index` and subtype tag — observable from guard failure tables
/// / status / bridge attachment.
///
/// Panics if either descr lacks a `FailDescr`, or if `Copied`-self path
/// is hit but the donor is not itself a `ResumeGuardCopiedDescr`
/// (matches RPython's `compile.py:841 assert isinstance(other,
/// ResumeGuardCopiedDescr)`).
pub fn copy_all_attributes_from(my_descr: &DescrRef, donor_descr: &DescrRef) {
    if my_descr.is_resume_guard_copied() {
        // compile.py ResumeGuardCopiedDescr.copy_all_attributes_from:
        //     assert isinstance(other, ResumeGuardCopiedDescr)
        //     self.prev = other.prev
        let donor_prev = donor_descr
            .prev_descr()
            .expect("compile.py:841 other must be a ResumeGuardCopiedDescr with a prev");
        my_descr.set_prev_descr(donor_prev);
    } else {
        // compile.py ResumeGuardDescr.copy_all_attributes_from:
        //     other = other.get_resumestorage()
        //     assert isinstance(other, ResumeGuardDescr)
        //     self.rd_consts = other.rd_consts
        //     self.rd_pendingfields = other.rd_pendingfields
        //     self.rd_virtuals = other.rd_virtuals
        //     self.rd_numb = other.rd_numb
        //     # we don't copy status
        //     if other.rd_vector_info:
        //         self.rd_vector_info = other.rd_vector_info.clone()
        // compile.py `other = other.get_resumestorage()`: copied
        // donors route reads through `prev`. Resolve the chain so we
        // never deep-copy from a copied descr's empty payload.
        let resolved_donor = get_resumestorage(donor_descr);
        // compile.py `assert isinstance(other, ResumeGuardDescr)` —
        // post-resolution donor must be a ResumeGuardDescr (or subclass).
        assert!(
            resolved_donor.is_resume_guard(),
            "compile.py:863 copy_all_attributes_from: \
             resolved donor must be a ResumeGuardDescr (got descr_index={:?})",
            resolved_donor.index()
        );
        let my_fd = my_descr
            .as_fail_descr()
            .expect("copy_all_attributes_from: my_descr must be a FailDescr");
        let donor_fd = resolved_donor
            .as_fail_descr()
            .expect("copy_all_attributes_from: donor must be a FailDescr after get_resumestorage");
        // RPython compile.py:864-867 does reference-share (`self.rd_consts
        // = other.rd_consts` etc.).  Pyre stores rd_* as `Arc<[T]>` so
        // the share is a single refcount bump per slot.  Reads still
        // get `&[T]` slices via the Deref impl; swap-replacement
        // through `set_rd_*` (Vec input) just builds a fresh Arc.
        my_fd.set_rd_numb_arc(donor_fd.rd_numb_arc());
        my_fd.set_rd_consts_arc(donor_fd.rd_consts_arc());
        my_fd.set_rd_virtuals_arc(donor_fd.rd_virtuals_arc());
        my_fd.set_rd_pendingfields_arc(donor_fd.rd_pendingfields_arc());
        // `AbstractFailDescr.rd_locs` is a backend slot upstream, written
        // only by `store_info_on_descr`, which every upstream backend runs —
        // so `copy_all_attributes_from` has nothing to carry there.  Pyre also
        // seeds it in `store_final_boxes_in_guard` with the resume numbering's
        // identity-with-holes layout, which cranelift and wasm keep because
        // they leave failargs in their logical slots.  That seed belongs to
        // the payload copied just above: the known-class bitfield inside
        // `rd_numb` holds one bit per non-hole Ref livebox, and
        // `deserialize_optimizer_knowledge` can only find those bits by
        // masking its fail args with the same holes.
        //
        // Every caller hands a descr the optimizer minted for a guard put in
        // place of an already-emitted one — `replace_guard_op`, the
        // GUARD_VALUE and GUARD_CLASS strengthening arms of
        // `optimize_guard_value` / `replace_old_guard_with_guard_class`, and
        // the vectorizer's `inhert_attributes` — and such a descr never
        // reaches `store_final_boxes_in_guard`, so it has no layout of its own.
        my_fd.set_rd_locs(donor_fd.rd_locs().to_vec().into());
        // compile.py — chain.clone() preserves the donor's
        // (already-flattened) accumulator chain on self, identity-stable.
        let donor_chain = donor_fd.vector_info();
        if !donor_chain.is_empty() {
            my_fd.replace_vector_info(donor_chain);
        }
    }
}

/// compile.py: CompileLoopVersionDescr(ResumeGuardDescr)
///
/// A guard descriptor for loop-version guards. These guards must never
/// fail at runtime — they exist only to mark where a specialized loop
/// version should be compiled and stitched.
///
/// Modeled as a newtype wrapping `ResumeGuardDescr` so the subclass
/// inherits the full `_attrs_` slot set (adr_jump_offset, rd_locs,
/// status, rd_loop_token, plus the pyre-side per-emission cells:
/// source_op_index, trace_info,
/// external_jump_target, bridge_*) the cranelift codegen path reads
/// off every guard descr at `collect_guards` / dispatch emission.
/// Same shape as `ResumeAtPositionDescr` (compile.py).
#[derive(Debug)]
pub struct CompileLoopVersionDescr {
    inner: ResumeGuardDescr,
}

// Safety: same as ResumeGuardDescr (single-threaded JIT).
unsafe impl Send for CompileLoopVersionDescr {}
unsafe impl Sync for CompileLoopVersionDescr {}

impl majit_ir::Descr for CompileLoopVersionDescr {
    fn index(&self) -> u32 {
        self.inner.fail_index
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        // Hand out the inner ResumeGuardDescr — see ResumeAtPositionDescr
        // in this file.  Uniform downcast across the subclass family.
        Some(&self.inner)
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn as_fail_descr_arc(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<dyn FailDescr>> {
        Some(self)
    }
    fn is_loop_version(&self) -> bool {
        true
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py: CompileLoopVersionDescr.clone() — overrides
    /// the inherited `ResumeGuardDescr.clone()` to mint a fresh
    /// `CompileLoopVersionDescr` (preserving the marker).  Resume data
    /// and types are copied via the base clone's `copy_all_attributes_from`
    /// shape; the per-emission cells reset to defaults.
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(CompileLoopVersionDescr {
            inner: ResumeGuardDescr {
                fail_index: alloc_fail_index(),
                types: UnsafeCell::new(unsafe { (&*self.inner.types.get()).clone() }),
                payload: self.inner.payload.deep_clone(),
                vector_info: UnsafeCell::new(unsafe { (&*self.inner.vector_info.get()).clone() }),
                adr_jump_offset: UnsafeCell::new(0),
                rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
                status: AtomicU64::new(0),
                rd_loop_token_clt: UnsafeCell::new(None),
                trace_id: AtomicU64::new(0),
                fail_index_per_trace: AtomicU32::new(0),
                bridge_declined_terminally: AtomicBool::new(false),
                source_op_index: UnsafeCell::new(None),
                back_edge_poll: std::sync::atomic::AtomicBool::new(false),
                trace_info: AtomicPtr::new(std::ptr::null_mut()),
                external_jump_target: OnceLock::new(),
                bridge: std::sync::OnceLock::new(),
            },
        }))
    }
}

impl FailDescr for CompileLoopVersionDescr {
    fn fail_index(&self) -> u32 {
        // Per-trace key (see ResumeGuardDescr::fail_index).  Global id
        // remains available via Descr::index() / get_descr_index().
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.inner.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.inner.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.inner.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.inner
            .fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.inner.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.inner.types.get() = types }
    }
    /// compile.py:899-900
    fn exits_early(&self) -> bool {
        true
    }
    /// compile.py:902-903
    fn loop_version(&self) -> bool {
        true
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.inner.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.inner.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.inner.vector_info.get() = build_vector_info_chain(chain) }
    }
    fn rd_numb(&self) -> Option<&[u8]> {
        self.inner.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.inner.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<majit_ir::NumberingRef>) {
        self.inner.payload.set_rd_numb_arc(value)
    }
    fn visit_rd_numb(&self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        self.inner.payload.visit_rd_numb(visitor);
    }
    fn rd_numb_ref(&self) -> Option<&majit_ir::NumberingRef> {
        self.inner.payload.rd_numb_ref()
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.inner.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<majit_ir::SharedConstPool>> {
        self.inner.payload.rd_consts_arc()
    }
    fn rd_consts_pool(&self) -> Option<&majit_ir::SharedConstPool> {
        self.inner.payload.rd_consts_pool()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.inner.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<majit_ir::SharedConstPool>>) {
        self.inner.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.inner.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.inner.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.inner.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.inner.payload.set_rd_virtuals_arc(value)
    }
    fn shaped_virtuals_any(&self) -> Option<&dyn std::any::Any> {
        self.inner.payload.shaped_virtuals_any()
    }
    fn cache_shaped_virtuals(
        &self,
        value: Box<dyn std::any::Any>,
    ) -> Result<(), Box<dyn std::any::Any>> {
        self.inner.payload.cache_shaped_virtuals(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.inner.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.inner.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.inner.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.inner.payload.set_rd_pendingfields_arc(value)
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.inner.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.inner.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.inner.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: majit_ir::RdLocs) {
        unsafe { *self.inner.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.inner.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.inner
            .status
            .fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.inner
            .status
            .fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.inner
            .status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.inner.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        let cell = unsafe { &*self.inner.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn std::any::Any)
    }
    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.inner.rd_loop_token_clt.get() = Some(typed) };
    }
    fn source_op_index(&self) -> Option<usize> {
        FailDescr::source_op_index(&self.inner)
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        FailDescr::set_source_op_index(&self.inner, source_op_index);
    }
    fn trace_info_any(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        FailDescr::trace_info_any(&self.inner)
    }
    fn set_trace_info_any(&self, info: Arc<dyn std::any::Any + Send + Sync>) {
        FailDescr::set_trace_info_any(&self.inner, info);
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        FailDescr::bridge_cache_addrs(&self.inner)
    }
    fn bridge_code_ptr(&self) -> usize {
        FailDescr::bridge_code_ptr(&self.inner)
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        FailDescr::store_bridge_caches(&self.inner, code_ptr, body_ptr);
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        FailDescr::bridge_dispatch_load(&self.inner)
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        FailDescr::bridge_dispatch_swap(&self.inner, new_ptr, drop_fn)
    }
    fn is_external_jump(&self) -> bool {
        FailDescr::is_external_jump(&self.inner)
    }
    fn target_descr(&self) -> Option<DescrRef> {
        FailDescr::target_descr(&self.inner)
    }
    fn set_external_jump_target(&self, target: DescrRef) {
        FailDescr::set_external_jump_target(&self.inner, target);
    }
}

fn make_compile_loop_version_descr_with_payload(types: Vec<Type>, payload: RdPayload) -> DescrRef {
    Arc::new(CompileLoopVersionDescr {
        inner: ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(types),
            payload,
            vector_info: UnsafeCell::new(None),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            bridge_declined_terminally: AtomicBool::new(false),
            source_op_index: UnsafeCell::new(None),
            back_edge_poll: std::sync::atomic::AtomicBool::new(false),
            trace_info: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target: OnceLock::new(),
            bridge: std::sync::OnceLock::new(),
        },
    })
}

/// compile.py: a fresh CompileLoopVersionDescr with no copied
/// resume payload. Used by vector.py:588-591 when the early-exit guard
/// has no donor descr to copy from.
pub fn make_compile_loop_version_descr_typed(types: Vec<Type>) -> DescrRef {
    make_compile_loop_version_descr_with_payload(types, RdPayload::empty())
}

pub fn make_compile_loop_version_descr() -> DescrRef {
    make_compile_loop_version_descr_typed(Vec::new())
}

/// guard.py:89-91:
///   descr = CompileLoopVersionDescr()
///   descr.copy_all_attributes_from(self.op.getdescr())
///   descr.rd_vector_info = None
///
/// Creates a fresh CompileLoopVersionDescr.  rd_* (compile.py
/// `_attrs_`) are reference-shared from the source descr via the
/// `_arc` getters — same semantics as `copy_all_attributes_from`
/// (compile.py:861-867).  `rd_vector_info` is reset to None per
/// guard.py:91.  The descr also carries fail_arg types from the
/// source so the backend layout matches the donor guard.
///
/// Panics if source_op has no descr or the resolved donor (after
/// `get_resumestorage()`) is not a `ResumeGuardDescr` — matching
/// RPython's invariant at compile.py:861-863
/// (`other = other.get_resumestorage(); assert isinstance(other,
/// ResumeGuardDescr)`).
pub fn make_compile_loop_version_descr_from(source_op: &majit_ir::Op) -> DescrRef {
    let src_descr = source_op
        .getdescr()
        .expect("guard.py:90: self.op.getdescr() must exist");
    // compile.py `other = other.get_resumestorage()`: if the source
    // is a `ResumeGuardCopiedDescr`, resolve to its `prev` so we read
    // resume data from the canonical donor.  ResumeGuardDescr's
    // `get_resumestorage` returns self, so direct sources pass through
    // unchanged.
    let resolved_descr = get_resumestorage(&src_descr);
    // compile.py `assert isinstance(other, ResumeGuardDescr)`:
    // reject non-resume FailDescr (e.g. `SimpleFailDescr`) that would
    // otherwise yield an empty rd_* payload on the loop-version descr.
    assert!(
        resolved_descr.is_resume_guard(),
        "compile.py:863 assert isinstance(other, ResumeGuardDescr): \
         loop-version donor descr_index={} is not a ResumeGuardDescr \
         subclass",
        resolved_descr.index()
    );
    let src_fd = resolved_descr
        .as_fail_descr()
        .expect("compile.py:863 ResumeGuardDescr is also a FailDescr");
    let types = src_fd.fail_arg_types().to_vec();
    // compile.py copy_all_attributes_from copies rd_*; mirror
    // RPython's reference-share by reusing the donor's `Arc<[T]>`
    // slots — `Arc::clone` only bumps a refcount.
    let payload = RdPayload::from_arcs(
        src_fd.rd_numb_arc(),
        src_fd.rd_consts_arc(),
        src_fd.rd_virtuals_arc(),
        src_fd.rd_pendingfields_arc(),
    );
    make_compile_loop_version_descr_with_payload(types, payload)
}

// Guard resume payload stays on the descr (`ResumeGuardDescr.get_resumestorage`).
// `build_guard_metadata` records source_op_index and that descr; it does not
// project a parallel resume/recovery layout.

//
// These are the **compile role** of `TraceCtx`, mirroring RPython's
// `pyjitpl.py` merge-point bookkeeping (`current_merge_points`,
// `portal_trace_positions`) and `compile.py compile_loop` /
// `compile_bridge` consumption of merge-point state.
//
// `MergePoint` itself lives in `trace_ctx.rs` alongside the
// `current_merge_points` field that owns it — matching RPython where
// `MetaInterp` (pyjitpl.py) owns both the struct and the list.

impl TraceCtx {
    /// pyjitpl.py reverse `same_greenkey` scan.
    ///
    /// `pyjitpl.py` `assert len(original_boxes) == len(live_arg_boxes)`.
    /// Primary traces seed the start boxes via
    /// [`Self::seed_compile_and_run_once_merge_point`]
    /// (`pyjitpl.py` `_compile_and_run_once` `current_merge_points = [(original_boxes, start)]`);
    /// later header visits append the same jitdriver red+vable shape.
    /// `TraceCtx::new` does not seed; bridges stay empty.
    pub fn has_merge_point_with_shape_assert(&self, key: &GreenKey, live_args_len: usize) -> bool {
        self.has_merge_point_same_greenkey(key, live_args_len)
    }

    /// pyjitpl.py `same_greenkey(original_boxes, live_arg_boxes, num_green_args)`.
    ///
    /// Live greens compare with `GreenKey::same_constant` (`pyjitpl.py`
    /// `same_greenkey` / `Const.same_constant`).
    pub fn has_merge_point_same_greenkey(&self, live: &GreenKey, live_args_len: usize) -> bool {
        // pyjitpl.py reverse scan:
        //   for j in range(len(self.current_merge_points) - 1, -1, -1):
        //       original_boxes, start = self.current_merge_points[j]
        //       assert len(original_boxes) == len(live_arg_boxes)
        //       if same_greenkey(...):
        //           ...
        for mp in self.current_merge_points.iter().rev() {
            // pyjitpl.py `assert len(original_boxes) == len(live_arg_boxes)`.
            // Seed and every later header visit push the same red+vable
            // length as the header's live args.
            assert_eq!(
                mp.green_boxes.len(),
                live_args_len,
                "same_greenkey: merge points in one trace share the jitdriver shape"
            );
            if mp.green_key_typed.same_constant(live) {
                return true;
            }
        }
        false
    }

    /// pyjitpl.py:3029-3030 — record a loop header visit with position
    /// and live variable snapshot.
    ///
    /// RPython allows multiple merge points with the same green key
    /// (representing different loop iterations or inlining depths).
    /// Always appends; has_merge_point checks if any match exists.
    ///
    /// `key_typed` is the green key `key` is the `JitCell.get_uhash` of
    /// (warmstate.py `def get_uhash(*greenargs)`). It travels with the
    /// hash because the trace-segmenting consumers of this entry
    /// (`mark_force_finish_tracing`, `disable_noninlinable_function`) write
    /// flags on a cell through `ensure_jit_cell_at_key(greenkey)`.
    ///
    /// `key` and `key_typed.get_uhash()` are NOT required to be equal. `key` is
    /// a cell key, which is the bucket hash only while that hash is unclaimed;
    /// once a second cell chains into the bucket the later one is minted a
    /// distinct number. The invariant is that `bucket_of(key)` equals
    /// `key_typed.get_uhash()`, and that is a warmstate question this type
    /// cannot ask.
    pub fn add_merge_point_with_key(
        &mut self,
        key: u64,
        key_typed: GreenKey,
        green_boxes: Vec<crate::trace_ctx::GreenBox>,
        header_pc: usize,
    ) {
        // Use the TraceCtx-level position so `snapshot_data_len` is
        // `len(_snapshot_data)` in byte mode and `Vec<Snapshot>` length
        // on the structured recorder.
        let position = self.get_trace_position();
        // `green_boxes` describes the virtualizable that is live right now, so
        // pair the snapshot with its address: `compile.py:510` reads the frame
        // out of the same list it hands to the backend as `loop.inputargs`, and
        // the trace-level pointer has moved on by the time a cut selects this
        // entry.
        let vable_ptr = self.standard_virtualizable_ptr().unwrap_or(0);
        self.current_merge_points.push(MergePoint {
            green_key: key,
            green_key_typed: key_typed,
            position,
            green_boxes,
            header_pc,
            vable_ptr,
        });
    }

    /// pyjitpl.py:2908 — bridge traces start with empty merge points.
    pub fn clear_merge_points(&mut self) {
        self.current_merge_points.clear();
    }

    /// pyjitpl.py `_compile_and_run_once`:
    /// `self.current_merge_points = [(original_boxes, (0, 0, 0, 0, 0))]`.
    ///
    /// Primary traces seed the start boxes so the first matching header
    /// visit closes. Bridges must not call this — `_handle_guard_failure`
    /// starts empty. The walker path seeds itself and does not go through
    /// `setup_tracing`.
    pub fn seed_compile_and_run_once_merge_point(&mut self) {
        let green_boxes = crate::trace_ctx::GreenBox::from_recorder_inputargs(&self.recorder);
        let key = self.green_key;
        let key_typed = self.green_key_values().cloned().expect(
            "a primary trace starts with its greens: pyjitpl.py _compile_and_run_once original_boxes",
        );
        let header_pc = self.header_pc;
        self.add_merge_point_with_key(key, key_typed, green_boxes, header_pc);
    }

    /// `current_merge_points[0]` is the outermost loop header's greenkey —
    /// pyjitpl.py:2791, :2827 and :2842 all spell it
    /// `self.current_merge_points[0][0][:jd_sd.num_green_args]`, in
    /// `aborted_tracing` / `blackhole_if_trace_too_long` /
    /// `prepare_trace_segmenting` respectively.  `None` distinguishes "tracing
    /// a bridge" (empty merge-points list) from "tracing a loop body".
    ///
    /// Returns the key in both forms it is carried in, together rather than
    /// through two accessors: the callers that only move a counter
    /// (`trace_next_iteration`) need the hash, the callers that install flags
    /// on a cell (`mark_force_finish_tracing`, `disable_noninlinable_function`)
    /// need the typed key so the flag lands on the cell `JitCell(*greenargs)`
    /// installed.
    pub fn current_merge_points_first_green_key_pair(&self) -> Option<(u64, GreenKey)> {
        self.current_merge_points
            .first()
            .map(|mp| (mp.green_key, mp.green_key_typed.clone()))
    }

    /// Hash-only form of [`Self::current_merge_points_first_green_key_pair`],
    /// for a caller that only tests whether a loop header exists or feeds a
    /// counter-only consumer. A caller that installs cell flags must take the
    /// pair instead.
    pub fn current_merge_points_first_greenkey(&self) -> Option<u64> {
        self.current_merge_points.first().map(|mp| mp.green_key)
    }

    /// pyjitpl.py `same_greenkey`, most recent hit.
    ///
    /// Live greens compare with `GreenKey::same_constant` (`pyjitpl.py`
    /// `same_greenkey` / `Const.same_constant`). A side `header_pc` is not
    /// part of the compare: `same_greenkey` matches the boxes, and requiring
    /// the trace-start `header_pc` missed a point recorded under the merge
    /// point's own pc (and the other way round). `header_pc` on
    /// [`MergePoint`] stays the guest pc of that visit, for the cut.
    pub fn find_merge_point_same_greenkey(&self, live: &GreenKey) -> Option<&MergePoint> {
        self.current_merge_points
            .iter()
            .rev()
            .find(|mp| mp.green_key_typed.same_constant(live))
    }

    /// See [`TraceCtx::merge_point_resumed`] — pyjitpl.py:1570/1577
    /// `saved_pc`. Read-and-clear: the skip applies to exactly one visit.
    pub fn take_merge_point_resumed(&mut self) -> bool {
        std::mem::take(&mut self.merge_point_resumed)
    }

    /// Both halves of pyjitpl.py:1571/1574 `saved_pc`, for a close that did
    /// NOT end the trace and so has to hand the walk back.
    ///
    /// The two are one operation upstream and must stay one here: a re-entry
    /// that latches the skip without handing over the pc rewinds the walk to
    /// wherever its segment began while its state stands at the merge point
    /// just reached, and the instructions in between are re-executed against
    /// state belonging to neither. Both driver re-entries call this so neither
    /// can acquire one half without the other.
    ///
    /// `close_green_pc` is `None` for a close that named no merge-point pc, in
    /// which case there is nothing to correct and the walk keeps the pc it was
    /// called with.
    pub fn resume_walk_after_close(&mut self) {
        self.walk_resume_pc = self.close_green_pc.and_then(|p| usize::try_from(p).ok());
        self.merge_point_resumed = true;
    }

    /// [`Self::find_merge_point_same_greenkey`] for a live typed key.
    /// `header_pc` is not part of `same_greenkey`; callers that still pass
    /// the trace-start pc must not have that pc reject a point recorded
    /// under the loop's own pc.
    pub fn has_merge_point_at(&self, key: &GreenKey, _header_pc: usize) -> bool {
        self.find_merge_point_same_greenkey(key).is_some()
    }

    /// `pyjitpl.py same_greenkey(original_boxes, live_arg_boxes,
    /// num_green_args)` — the loop header the trace is closing ON, read from
    /// the greens it closes WITH.
    ///
    /// The guest pc is `close_green_pc`. It is not the trace-start
    /// `header_pc`, which a close on another loop leaves stale.
    /// `same_greenkey` always reads the closing boxes; slot 56 counts a
    /// close that never recorded a pc, where the session header is the
    /// only value left.
    pub fn close_header_pc(&self) -> usize {
        // A negative pc is not a jitcode position. `as usize` would wrap it
        // into a huge header and file the merge point on the wrong loop.
        if let Some(pc) = self.close_green_pc.and_then(|pc| usize::try_from(pc).ok()) {
            return pc;
        }
        crate::mc_diag_bump(56);
        self.header_pc
    }

    /// `compile.py compile_retrace(..., start)` — the merge point the
    /// `same_greenkey` scan already selected. Look up by that `start`
    /// position, not by `(green_key, header_pc)`: `header_pc` is the
    /// last registered header and a close on another loop leaves it
    /// stale (`close_header_pc`).
    pub fn merge_point_at_start(
        &self,
        start: crate::recorder::TracePosition,
    ) -> Option<&MergePoint> {
        self.current_merge_points
            .iter()
            .rev()
            .find(|mp| mp.position == start)
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.inline_frames.len()
    }

    /// Update the green key for this trace.
    ///
    /// RPython pyjitpl.py reached_loop_header(): when func-entry tracing
    /// hits a back-edge, the loop must be registered under the back-edge's
    /// green key, not the function-entry key.
    pub fn set_green_key(&mut self, key: u64, raw: (usize, usize)) {
        self.green_key = key;
        self.green_key_raw = raw;
        crate::loop_census::note_key(key, raw);
    }

    /// Record the structured greenkey for the root trace. Called once
    /// at trace start to seed `green_key_raw` and `root_green_key_raw`
    /// from the tracer-side `(code_ptr, pc)`. Subsequent back-edge
    /// retargeting flows through [`Self::set_green_key`].
    pub fn set_root_green_key_raw(&mut self, raw: (usize, usize)) {
        self.green_key_raw = raw;
        self.root_green_key_raw = raw;
        crate::loop_census::note_key(self.green_key, raw);
    }

    /// pyjitpl.py:1396-1401 element-wise greenkey comparison against
    /// the current trace's greenkey and each inline-frame greenkey.
    pub fn is_tracing_key(&self, target: (usize, usize)) -> bool {
        self.green_key_raw == target
            || self.root_green_key_raw == target
            || self.inline_frames.contains(&target)
    }

    /// pyjitpl.py:1390-1402 recursion counting only walks portal
    /// frames already pushed on `framestack`; the root trace entry is
    /// not counted unless it has become an actual inline frame.
    pub fn has_inline_frame_for(&self, target: (usize, usize)) -> bool {
        self.inline_frames.contains(&target)
    }

    /// pyjitpl.py `_opimpl_recursive_call` element-wise walk:
    ///
    /// ```python
    /// count = 0
    /// for f in self.metainterp.framestack:
    ///     if f.jitcode is not portal_code: continue
    ///     gk = f.greenkey
    ///     for i in range(len(gk)):
    ///         if not gk[i].same_constant(greenboxes[i]): break
    ///     else: count += 1
    /// ```
    ///
    /// Pyre's greenkey is `(code_ptr, pc)` — a fixed-arity pair — so
    /// tuple equality reproduces the element-wise `same_constant`
    /// result without an intermediate hash that could falsely collide.
    ///
    /// Only inlined portal frames count here. The root frame is
    /// created without a `greenkey` in upstream `initialize_state_from_start`
    /// / `newframe(mainjitcode)`, so counting `root_green_key_raw` would
    /// make self-recursion hit `max_unroll_recursion` one level early.
    pub fn recursive_depth(&self, target: (usize, usize)) -> usize {
        self.inline_frames.iter().filter(|&&k| k == target).count()
    }

    /// Push an inline frame (entering a callee).
    /// Returns false if the max inline depth has been exceeded.
    /// `callee_raw` is the structured `(code_ptr, pc)` greenkey, stored
    /// in `inline_frames` so `recursive_depth` / `is_tracing_key` can
    /// walk it element-wise (pyjitpl.py:1396-1401 parity).
    pub(crate) fn push_inline_frame(&mut self, callee_raw: (usize, usize), max_depth: u32) -> bool {
        if (self.inline_frames.len() as u32) >= max_depth {
            return false;
        }
        self.inline_frames.push(callee_raw);
        true
    }

    /// Pop an inline frame (returning from a callee).
    pub(crate) fn pop_inline_frame(&mut self) {
        self.inline_frames.pop();
    }
}

#[cfg(test)]
mod same_greenkey_tests {
    use super::*;
    use majit_ir::GreenType;

    /// pyjitpl.py `same_greenkey` / history.py `ConstFloat.same_constant`:
    /// bit identity, so Float(+0.0) does not match Float(-0.0).
    #[test]
    fn find_merge_point_same_greenkey_uses_const_float_bit_identity() {
        let mut ctx = TraceCtx::for_test_types(&[]);
        let pos = GreenKey::with_types(vec![0.0f64.to_bits() as i64], vec![GreenType::Float]);
        let neg = GreenKey::with_types(vec![(-0.0f64).to_bits() as i64], vec![GreenType::Float]);
        ctx.add_merge_point_with_key(pos.get_uhash(), pos.clone(), Vec::new(), 0);
        assert!(
            ctx.find_merge_point_same_greenkey(&pos).is_some(),
            "same_greenkey matches ConstFloat(+0.0) against itself"
        );
        assert!(
            ctx.find_merge_point_same_greenkey(&neg).is_none(),
            "ConstFloat.same_constant distinguishes +0.0 from -0.0"
        );
    }
}

#[cfg(test)]
mod fail_descr_tests {
    use super::*;

    #[test]
    fn all_virtuals_hide_show_round_trips_the_cache() {
        let hidden = AllVirtuals::hide((vec![1, 2], vec![3]));
        assert_eq!(AllVirtuals::show(hidden), Some((vec![1, 2], vec![3])));
        assert_eq!(AllVirtuals::show(majit_ir::GcRef(0)), None);
    }

    #[test]
    fn resume_guard_descr_unwraps_forced_and_exc_wrappers() {
        let forced = make_resume_guard_forced_descr_typed(vec![Type::Int]);
        let forced_inner =
            resume_guard_descr(&forced).expect("Forced as_any is the wrapper, not the base");
        assert_eq!(forced_inner.fail_arg_types(), &[Type::Int]);

        let exc = make_resume_guard_exc_descr_typed(vec![Type::Ref]);
        let exc_inner = resume_guard_descr(&exc).expect("Exc as_any is the inner base descr");
        assert_eq!(exc_inner.fail_arg_types(), &[Type::Ref]);

        let plain = make_resume_guard_descr_typed(vec![Type::Float]);
        let plain_inner = resume_guard_descr(&plain).expect("plain ResumeGuardDescr unwraps");
        assert_eq!(plain_inner.fail_arg_types(), &[Type::Float]);
    }

    #[test]
    fn force_now_runs_the_body() {
        let ran = std::cell::Cell::new(false);
        ResumeGuardForcedDescr::force_now(|| ran.set(true));
        assert!(
            ran.get(),
            "compile.py force_now runs cpu.force + handle_async"
        );
    }

    #[test]
    fn test_attach_vector_info_builds_prev_chain() {
        let descr = make_fail_descr(2);
        let fail_descr = descr.as_fail_descr().unwrap();
        fail_descr.attach_vector_info(AccumInfo {
            prev: None,
            failargs_pos: 0,
            variable: majit_ir::OpRef::int_op(10),
            location: majit_ir::OpRef::int_op(20),
            accum_operation: '+',
            scalar: majit_ir::OpRef::NONE,
        });
        fail_descr.attach_vector_info(AccumInfo {
            prev: None,
            failargs_pos: 1,
            variable: majit_ir::OpRef::int_op(11),
            location: majit_ir::OpRef::int_op(21),
            accum_operation: '*',
            scalar: majit_ir::OpRef::NONE,
        });

        let vector_info = fail_descr.vector_info();
        assert_eq!(vector_info.len(), 2);
        assert_eq!(vector_info[0].failargs_pos, 1);
        assert_eq!(vector_info[1].failargs_pos, 0);
        assert_eq!(
            vector_info[0].prev.as_ref().map(|info| info.failargs_pos),
            Some(0)
        );
        assert!(vector_info[1].prev.is_none());

        let cloned = descr.clone_descr().unwrap();
        let cloned_vector_info = cloned.as_fail_descr().unwrap().vector_info();
        assert_eq!(cloned_vector_info.len(), 2);
        assert_eq!(cloned_vector_info[0].failargs_pos, 1);
        assert_eq!(
            cloned_vector_info[0]
                .prev
                .as_ref()
                .map(|info| info.failargs_pos),
            Some(0)
        );
    }

    #[test]
    fn test_fail_index_one_counter_across_modules() {
        // `alloc_fail_index` (`resume_guard_descr`) and `compile.rs`'s
        // mint both call that one counter. Two independent counters that
        // both start at 1 hand the same integer to two guards.
        let from_descr = majit_backend::alloc_fail_index();
        let from_compile = alloc_fail_index();
        assert_ne!(from_descr, from_compile);
        let from_descr_again = majit_backend::alloc_fail_index();
        assert_ne!(from_compile, from_descr_again);
        assert_ne!(from_descr, from_descr_again);
    }

    #[test]
    fn test_fail_descr_unique_indices() {
        // The fail-index counter is `alloc_fail_index` in
        // `resume_guard_descr`, shared by every test that allocates a
        // `FailDescr`. cargo test runs tests in parallel, so resetting
        // the counter here would race against concurrent allocations in
        // unrelated tests and let two descrs share the same fail_index.
        // The check below only asserts pairwise uniqueness, so the
        // starting value of the counter is irrelevant.
        let d1 = make_fail_descr(2);
        let d2 = make_fail_descr(3);
        let d3 = make_fail_descr(1);

        let fi1 = d1.index();
        let fi2 = d2.index();
        let fi3 = d3.index();

        // All indices must be unique
        assert_ne!(fi1, fi2);
        assert_ne!(fi2, fi3);
        assert_ne!(fi1, fi3);
    }

    #[test]
    fn test_fail_descr_with_explicit_index() {
        let d = make_fail_descr_with_index(42, 3);
        assert_eq!(d.index(), 42);
        assert_eq!(d.as_fail_descr().unwrap().fail_arg_types().len(), 3);
    }

    #[test]
    fn test_fail_descr_typed() {
        let types = vec![Type::Int, Type::Ref, Type::Float];
        let d = make_fail_descr_typed(types.clone());
        assert_eq!(d.as_fail_descr().unwrap().fail_arg_types(), &types);
    }

    /// `compile.py ResumeGuardDescr.store_final_boxes` parity:
    /// `store_final_boxes_in_guard` mutates types in place, preserving
    /// the descr's `Arc` identity, `fail_index`, and concrete subtype.
    #[test]
    fn test_set_fail_arg_types_preserves_identity_and_subtype() {
        // ResumeAtPositionDescr (unroll extra_guards path).
        let descr = make_resume_at_position_descr_typed(vec![Type::Int]);
        let original_fail_index = descr.index();
        let original_ptr = Arc::as_ptr(&descr);
        assert!(descr.is_resume_at_position());

        descr
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Ref, Type::Float]);

        // Identity preserved: same Arc, same fail_index, same subtype tag.
        assert_eq!(Arc::as_ptr(&descr), original_ptr);
        assert_eq!(descr.index(), original_fail_index);
        assert!(descr.is_resume_at_position());
        // Types updated.
        assert_eq!(
            descr.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Ref, Type::Float]
        );

        // CompileLoopVersionDescr.
        let lv = Arc::new(CompileLoopVersionDescr {
            inner: ResumeGuardDescr {
                fail_index: alloc_fail_index(),
                types: UnsafeCell::new(vec![Type::Int]),
                payload: RdPayload::empty(),
                vector_info: UnsafeCell::new(None),
                adr_jump_offset: UnsafeCell::new(0),
                rd_locs: UnsafeCell::new(majit_ir::RdLocs::new()),
                status: AtomicU64::new(0),
                rd_loop_token_clt: UnsafeCell::new(None),
                trace_id: AtomicU64::new(0),
                fail_index_per_trace: AtomicU32::new(0),
                bridge_declined_terminally: AtomicBool::new(false),
                source_op_index: UnsafeCell::new(None),
                back_edge_poll: std::sync::atomic::AtomicBool::new(false),
                trace_info: AtomicPtr::new(std::ptr::null_mut()),
                external_jump_target: OnceLock::new(),
                bridge: std::sync::OnceLock::new(),
            },
        }) as DescrRef;
        let lv_fi = lv.index();
        assert!(lv.as_fail_descr().unwrap().loop_version());

        lv.as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Ref]);

        assert_eq!(lv.index(), lv_fi);
        assert!(lv.as_fail_descr().unwrap().loop_version());
        assert_eq!(lv.as_fail_descr().unwrap().fail_arg_types(), &[Type::Ref]);

        // Plain `ResumeGuardDescr` factory.
        let plain = make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
        let plain_fi = plain.index();
        assert!(!plain.is_resume_at_position());

        plain
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Float]);

        assert_eq!(plain.index(), plain_fi);
        assert!(!plain.is_resume_at_position());
        assert!(!plain.is_guard_forced());
        assert!(!plain.is_guard_exc());
        assert!(
            plain
                .as_fail_descr()
                .unwrap()
                .handle_fail_is_plain_resume_guard()
        );
        assert_eq!(
            plain.as_fail_descr().unwrap().handle_fail_class(),
            HandleFailClass::ResumeGuard
        );
        assert_eq!(
            plain.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );

        // ResumeGuardForcedDescr — `is_guard_forced()` survives
        // `set_fail_arg_types`, identity preserved.
        let forced = make_resume_guard_forced_descr_typed(vec![Type::Int]);
        let forced_fi = forced.index();
        let forced_ptr = Arc::as_ptr(&forced);
        assert!(forced.is_guard_forced());
        assert!(!forced.is_guard_exc());
        assert!(
            !forced
                .as_fail_descr()
                .unwrap()
                .handle_fail_is_plain_resume_guard()
        );
        assert_eq!(
            forced.as_fail_descr().unwrap().handle_fail_class(),
            HandleFailClass::Forced
        );

        forced
            .as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Ref]);

        assert_eq!(Arc::as_ptr(&forced), forced_ptr);
        assert_eq!(forced.index(), forced_fi);
        assert!(forced.is_guard_forced());
        assert_eq!(
            forced.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Ref]
        );

        // ResumeGuardExcDescr — `is_guard_exc()` survives, identity preserved.
        let exc = make_resume_guard_exc_descr_typed(vec![Type::Ref, Type::Int]);
        let exc_fi = exc.index();
        let exc_ptr = Arc::as_ptr(&exc);
        assert!(exc.is_guard_exc());
        assert!(!exc.is_guard_forced());
        assert!(
            !exc.as_fail_descr()
                .unwrap()
                .handle_fail_is_plain_resume_guard()
        );
        assert_eq!(
            exc.as_fail_descr().unwrap().handle_fail_class(),
            HandleFailClass::Exc
        );

        exc.as_fail_descr()
            .unwrap()
            .set_fail_arg_types(vec![Type::Float]);

        assert_eq!(Arc::as_ptr(&exc), exc_ptr);
        assert_eq!(exc.index(), exc_fi);
        assert!(exc.is_guard_exc());
        assert_eq!(
            exc.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );

        // compile.py ResumeGuardDescr.clone() returns a plain
        // ResumeGuardDescr — both `ResumeGuardForcedDescr` (compile.py+,
        // no clone override) and `ResumeGuardExcDescr` (compile.py
        // `pass`) inherit this base implementation, so the subtype tag is
        // intentionally dropped on clone. Resume attributes / fail_arg_types
        // are copied; fail_index is fresh.
        let forced_clone = forced.clone_descr().unwrap();
        assert!(!forced_clone.is_guard_forced());
        assert!(!forced_clone.is_guard_exc());
        assert_ne!(forced_clone.index(), forced_fi);
        assert_eq!(
            forced_clone.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Ref]
        );

        let exc_clone = exc.clone_descr().unwrap();
        assert!(!exc_clone.is_guard_exc());
        assert!(!exc_clone.is_guard_forced());
        assert_ne!(exc_clone.index(), exc_fi);
        assert_eq!(
            exc_clone.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );
    }

    /// `handle_guard_failure` arms `seen_loop_header_for_jdindex` from
    /// `isinstance(key, ResumeAtPositionDescr)` where `key =
    /// resumedescr.get_resumestorage()` (pyjitpl.py / :2941), so a copied
    /// guard sharing a short-preamble donor answers yes even though the copied
    /// descr is not itself a `ResumeAtPositionDescr`.  One hop reaches the
    /// donor for every copied descr; the sibling test below pins the
    /// constructor assert that rules a copied-over-copied chain out.
    #[test]
    fn get_resumestorage_reports_a_copied_guards_resume_at_position_donor() {
        let donor = make_resume_at_position_descr_typed(vec![Type::Int]);
        assert!(donor.is_resume_at_position());

        let copied = make_resume_guard_copied_descr(donor.clone());
        assert!(
            !copied.is_resume_at_position(),
            "compile.py:832 ResumeGuardCopiedDescr is a sibling of \
             ResumeGuardDescr, not a ResumeAtPositionDescr"
        );
        assert!(
            get_resumestorage(&copied).is_resume_at_position(),
            "compile.py:848-851 get_resumestorage() resolves to prev"
        );

        // A plain resume guard is its own storage (compile.py:883-884).
        let plain = make_resume_guard_descr_typed(vec![Type::Int]);
        assert_eq!(Arc::as_ptr(&get_resumestorage(&plain)), Arc::as_ptr(&plain));
        assert!(!get_resumestorage(&plain).is_resume_at_position());
    }

    /// `get_resumestorage` hops to `prev` exactly once because a chain of
    /// copied descrs cannot exist: `compile.py
    /// ResumeGuardCopiedDescr.__init__` asserts `isinstance(prev,
    /// ResumeGuardDescr)`, and `ResumeGuardCopiedDescr` is a sibling of that
    /// class rather than a subclass, so it can never be a donor.  The
    /// constructor carries that assert, which is what makes the single hop in
    /// `get_resumestorage` complete rather than a truncated walk.
    #[test]
    #[should_panic(expected = "assert isinstance(prev, ResumeGuardDescr)")]
    fn a_copied_descr_is_refused_as_a_copied_descrs_donor() {
        let donor = make_resume_at_position_descr_typed(vec![Type::Int]);
        let copied = make_resume_guard_copied_descr(donor);
        assert!(copied.is_resume_guard_copied() && !copied.is_resume_guard());
        let _chained = make_resume_guard_copied_descr(copied);
    }

    /// `get_resumestorage()` chases to `prev`, `fail_arg_types`
    /// shares the donor's vector, `is_resume_guard_copied()` flags
    /// the subtype, and the exc variant additionally reports
    /// `is_guard_exc() = true` while tracking the same `prev`.
    #[test]
    fn test_resume_guard_copied_descr_delegates_to_prev() {
        // Plain copied descr over a ResumeGuardDescr donor.
        let donor = make_resume_guard_descr_typed(vec![Type::Int, Type::Ref]);
        let donor_fi = donor.index();
        let donor_ptr = Arc::as_ptr(&donor);

        let copied = make_resume_guard_copied_descr(donor.clone());
        assert!(copied.is_resume_guard_copied());
        assert!(!copied.is_guard_exc());
        assert!(!copied.is_guard_forced());
        assert_ne!(copied.index(), donor_fi);
        assert_eq!(
            copied.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Int, Type::Ref]
        );
        // get_resumestorage() chases to prev — same Arc ptr.
        let prev = copied.prev_descr().unwrap();
        assert_eq!(Arc::as_ptr(&prev), donor_ptr);

        // clone_descr() preserves prev identity, allocates fresh fail_index.
        let copied_clone = copied.clone_descr().unwrap();
        assert!(copied_clone.is_resume_guard_copied());
        assert_eq!(Arc::as_ptr(&copied_clone.prev_descr().unwrap()), donor_ptr);
        assert_ne!(copied_clone.index(), copied.index());

        // Exc-copied subtype carries the same prev and additionally
        // reports is_guard_exc() = true.
        let exc_donor = make_resume_guard_exc_descr_typed(vec![Type::Float]);
        let exc_donor_ptr = Arc::as_ptr(&exc_donor);
        let copied_exc = make_resume_guard_copied_exc_descr(exc_donor);
        assert!(copied_exc.is_resume_guard_copied());
        assert!(copied_exc.is_guard_exc());
        assert!(!copied_exc.is_guard_forced());
        assert_eq!(
            copied_exc.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Float]
        );
        assert_eq!(
            Arc::as_ptr(&copied_exc.prev_descr().unwrap()),
            exc_donor_ptr
        );

        // set_fail_arg_types must NOT mutate the donor through a copied
        // descr — `_copy_resume_data_from` never calls
        // store_final_boxes_in_guard, so any invocation indicates an
        // upstream invariant violation.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copied
                .as_fail_descr()
                .unwrap()
                .set_fail_arg_types(vec![Type::Float])
        }));
        assert!(
            result.is_err(),
            "set_fail_arg_types on ResumeGuardCopiedDescr must panic"
        );
        // Donor's types unchanged.
        assert_eq!(
            donor.as_fail_descr().unwrap().fail_arg_types(),
            &[Type::Int, Type::Ref]
        );
    }
}
