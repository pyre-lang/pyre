//! Force-before-writeback (FBW) walker state: the mutable side channels
//! the tracer consults while deciding whether an opcode can be folded,
//! inlined, or must fall back to a residual call.
//!
//! **Parity:** pyre-specific — the FBW (full-body-walk) live tracer has no
//! `rpython/jit/metainterp/` file counterpart. PyPy's parity-faithful
//! `MIFrame` interpret loop lives in `majit-metainterp/pyjitpl.rs` and is
//! retired as the production path; this is walker-local state with no
//! upstream analogue.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Groups the
//! `PYRE_FBW_*` feature gates, the store / append / for-iter journals and
//! their rollback machinery, the executed-effect and residual counters,
//! the finish-payload channel, and the abort-resume carriers. These are
//! thread-local / walker-scoped helpers with no opcode dispatch of their
//! own; the dispatch arms in `mod.rs` call into them.

use super::*;

/// Maximum inline depth the multiframe guard-snapshot path
/// (`walker_capture_multi_frame_inline_snapshot`) unrolls a straight-line
/// value-returning callee CHAIN before folding to the `CALL_ASSEMBLER` tail.
/// Default 7: a deep value-returning chain (`b→c→…→h`) stays in compiled code
/// and each extra inlined level removes a residual call — measured ~2.0–2.3×
/// on `bench/synth/inline_chain_depth_typeflip.py` with no regression
/// elsewhere.  The number predates that file: it was taken on the separate
/// per-depth fixtures later consolidated into it, so re-measure before
/// leaning on it.
///
/// A callee that raises inline is capped separately — at THREE distinct
/// multiframe levels, keyed off `callee_body_contains_raise` — because its
/// unwind crosses suspended intermediate frames through the cross-frame
/// bridge (gh#343 / gh#467).  Recursive / duplicate raising used to sit at
/// 2 while the carrier treated a middle `SubRaise` as a drain abort; the
/// drain now continues `finishframe_exception` there, so that bound matches
/// value-returning recursion (`max_unroll_recursion` only).
///
/// A self-recursive callee is bounded instead by
/// [`fbw_inline_recursion_count`] against `max_unroll_recursion`, mirroring
/// `opimpl_recursive_call` (`pyjitpl.py`) folding the recursive call
/// straight to `CALL_ASSEMBLER` past the bound rather than continuing to unroll
/// the call tree.
/// (The depth-≥2 blackhole-resume crash that previously blocked this path was a
/// GC-rooting gap in the nested `run()` chain, fixed by rooting the whole
/// pending `nextblackholeinterp` chain across `run()`; the bound is now a
/// performance, not a soundness, question.)
pub(crate) fn fbw_max_multiframe_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("PYRE_FBW_MULTIFRAME_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(7)
            .clamp(1, 7)
    })
}

/// Recursion depth of `w_code` on the walk's framestack —
/// `opimpl_recursive_call` (`pyjitpl.py:1390-1402`) counting the portal frames
/// already on `MetaInterp.framestack` before comparing against
/// `max_unroll_recursion`.
///
/// Upstream compares the whole greenkey element-wise (`pyjitpl.py`
/// `gk[i].same_constant(greenboxes[i])`) over `(next_instr, is_being_profiled,
/// bytecode)` (`interp_jit.py get_printable_location`); matching `w_code` alone is that comparison,
/// because every entry this scan can see carries `next_instr == 0`:
///
/// * `MIFrame.setup` (`pyjitpl.py`) assigns `greenkey` once and never
///   updates it, so a frame's greenkey stays its ENTRY greens however far its
///   pc has since advanced;
/// * every [`InlineFrame`] is pushed by `InlineFrameGuard::enter` at a callee
///   entry. Fresh forward calls carry that greenkey, while
///   `resume.rebuild_from_resumedata` calls `MetaInterp.newframe(jitcode)`
///   without one, so bridge-reconstructed frames are deliberately skipped.
///
/// The root frame is deliberately absent from this framestack, which is also
/// what upstream's greenkey compare achieves: the root's greenkey holds the
/// merge-point pc, so it fails `same_constant` against a call's `next_instr ==
/// 0` greens and is not counted either.
pub(crate) fn fbw_inline_recursion_count<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    w_code: usize,
) -> usize {
    inline_recursion_count(&ctx.session.borrow().framestack, w_code)
}

fn inline_recursion_count(frames: &[InlineFrame], w_code: usize) -> usize {
    frames
        .iter()
        .filter(|frame| frame.recursion_greenkey && frame.w_code == w_code)
        .count()
}

/// Whether the callee portal is already represented by any live inline
/// frame, independently of `MIFrame.greenkey`.
///
/// This is not the upstream recursive-unroll count above. It only keeps
/// pyre's extra non-recursive call-chain valve from applying to a recursive
/// portal. PyPy `_opimpl_recursive_call` has no such generic framestack cap;
/// in particular a frame rebuilt by `resume.rebuild_from_resumedata` has a
/// `None` greenkey (so it contributes zero to the max-unroll count) but a call
/// back into the same portal is still recursive and must not acquire a second
/// local bound.
pub(crate) fn fbw_recursive_portal_present<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    w_code: usize,
) -> bool {
    recursive_portal_present(&ctx.session.borrow().framestack, w_code)
}

fn recursive_portal_present(frames: &[InlineFrame], w_code: usize) -> bool {
    frames.iter().any(|frame| frame.w_code == w_code)
}

/// True when two live frames share a `w_code`.  Used with
/// [`fbw_effective_multiframe_depth`] so a raising leaf through two
/// copies of the same intermediate is classified as recursive/duplicate
/// (then bounded only by `max_unroll_recursion`) rather than as a
/// distinct three-frame chain.
pub(crate) fn framestack_has_duplicate_w_code(frames: &[InlineFrame]) -> bool {
    let mut seen = Vec::with_capacity(frames.len());
    for frame in frames {
        if frame.w_code == 0 {
            continue;
        }
        if seen.contains(&frame.w_code) {
            return true;
        }
        seen.push(frame.w_code);
    }
    false
}

/// Total inline-stack bound for a callee after its same-greenkey recursion
/// count has already been checked against the live `max_unroll_recursion`.
///
/// PyPy `_opimpl_recursive_call` has no second, total-framestack depth gate:
/// once a recursive portal is admitted, `memory_manager.max_unroll_recursion`
/// is the sole value-returning recursion bound.  FBW's generic multiframe cap
/// is a local cost valve for non-recursive call chains and must not silently
/// turn an upstream value of 7 into an effective 6 when an ambient inline
/// frame is present.
///
/// Raising chains keep a shallower local bound on a *distinct* call chain
/// because the carrier unwind crosses suspended frames.  `perform_call`
/// (`pyjitpl.py`) has no such cap: it always pushes a new `MIFrame`.  A
/// distinct `shape → mid → leaf` chain compiles the exception-edge bridge
/// the way `finishframe_exception` does (`ChangeFrame` into the catching
/// middle), so that shape is admitted to depth 3.  Recursive / duplicate
/// raising is bounded only by `max_unroll_recursion`, the same as
/// value-returning recursion: a middle `SubRaise` after `finishframe` is
/// `finishframe_exception` over the remaining shallower frames.
pub(crate) fn fbw_effective_multiframe_depth(
    contains_raise: bool,
    recursive_or_duplicate: bool,
) -> usize {
    if contains_raise {
        if recursive_or_duplicate {
            usize::MAX
        } else {
            3
        }
    } else if recursive_or_duplicate {
        usize::MAX
    } else {
        fbw_max_multiframe_depth()
    }
}

#[cfg(test)]
mod recursion_depth_policy_tests {
    use super::*;

    fn frame(w_code: usize, recursion_greenkey: bool) -> InlineFrame {
        InlineFrame {
            is_portal: false,
            w_code,
            recursion_greenkey,
            call_id: 0,
            debug_merge_point_py_pc: None,
            parents: Vec::new(),
            entry_executed_effects: 0,
            live: None,
        }
    }

    #[test]
    fn resumed_frames_without_greenkeys_do_not_count_as_recursive_unrolls() {
        let frames = [frame(42, true), frame(42, false), frame(7, true)];
        assert_eq!(inline_recursion_count(&frames, 42), 1);
        assert!(recursive_portal_present(&frames, 42));
    }

    #[test]
    fn value_returning_recursion_is_bounded_only_by_max_unroll_recursion() {
        assert_eq!(fbw_effective_multiframe_depth(false, true), usize::MAX);
    }

    #[test]
    fn raising_recursion_is_bounded_only_by_max_unroll_recursion() {
        assert_eq!(fbw_effective_multiframe_depth(true, true), usize::MAX);
    }

    #[test]
    fn distinct_raising_chain_admits_the_three_frame_exception_edge() {
        assert_eq!(fbw_effective_multiframe_depth(true, false), 3);
    }

    #[test]
    fn duplicate_w_code_on_stack_is_the_selfrec_unroll_shape() {
        let frames = [frame(7, true), frame(7, true), frame(9, false)];
        assert!(framestack_has_duplicate_w_code(&frames));
        assert!(!framestack_has_duplicate_w_code(&[
            frame(7, true),
            frame(9, false)
        ]));
    }
}

/// The innermost inline level's strict-fold frame register (`u16::MAX` when
/// inactive / no inline level).
pub(crate) fn fbw_strict_fold_frame_reg<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> u16 {
    ctx.frame_state
        .borrow()
        .callee_shadow
        .as_ref()
        .map_or(u16::MAX, |shadow| shadow.fold_frame_reg)
}

thread_local! {
    /// Discriminates a top-level [`DispatchOutcome::Terminate`] payload:
    /// `true` when it is an uncaught raise (`fbw_terminate_with_raise`),
    /// so [`crate::trace::full_body_walk_trace`] builds a
    /// `TraceAction::Finish { exit_with_exception: true }`
    /// (`compile_exit_frame_with_exception`) rather than a value-return
    /// FINISH.  A dedicated flag rather than the `FBW_FINISH_CONCRETE::Raise`
    /// marker because the latter is null-guarded for GC-rooting and so is
    /// absent when the raised exception has no concrete Ref.  Reset at the
    /// start of every walk.
    static FBW_FINISH_IS_EXCEPTION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    /// The terminal disposition a top-level walk produced, set for a
    /// loop-free portal exit (`DispatchOutcome::Terminate`).  Unlike
    /// the symbolic `OpRef` the compile consumer records into the trace,
    /// this holds the value the walk *concretely* computed.
    ///
    /// A function trace that fully unrolls to `done_with_this_frame`
    /// executed every residual call concretely (consuming side-effecting
    /// callees like a tokenizer's `get`), so re-running the freshly
    /// compiled trace for the SAME invocation (`ContinueRunningNormally`)
    /// would re-read the already-mutated heap and deopt.  The portal
    /// instead returns this captured value directly (no replay); the
    /// compiled trace serves only subsequent invocations.  See the
    /// consume site in `eval.rs` (`maybe_compile_and_run` portal exit).
    ///
    /// `ConcreteValue::Ref` payloads hold a nursery-resident object across
    /// the post-walk compile (which allocates), so the slot is GC-rooted
    /// via [`fbw_finish_concrete_root_walker`].  `None` for ungated /
    /// loop-closing / float (no concrete float shadow bank) walks → the
    /// portal degrades to the legacy `ContinueRunningNormally` replay.
    ///
    /// This root exists because of the *outliving*, not because a walk-time
    /// concrete is otherwise unrooted.  Every value the walk computes is
    /// stamped onto its frontend op (`set_opref_concrete` →
    /// `history.py` `*FrontendOp(pos, value)`), and
    /// `MetaInterp::walk_active_trace_refs` forwards every recorder
    /// `Op`/`InputArg` `value` cell holding a `Value::Ref` — so for the
    /// duration of the walk the recorder IS the root set.  The compile path
    /// then does `self.tracing.take()`, dropping that recorder while this
    /// stash is still live, which is exactly the window no op-graph slot
    /// covers.  A walk-scoped side list of the same values would be a second
    /// source of truth for a window the op graph already owns.
    static FBW_FINISH_CONCRETE: std::cell::Cell<Option<FinishConcrete>> =
        const { std::cell::Cell::new(None) };

    /// `(frame, last_instr_before)` for the concrete half of
    /// [`fbw_publish_exit_last_instr`], so a walk that does not commit can put
    /// the field back.
    ///
    /// The publish fires when the walk reaches the exit; whether that exit is
    /// kept is decided afterwards, in the walk-end epilogue.  A declined walk
    /// returns to a replay that resumes the frame from the pre-walk state, and
    /// `last_instr` is exactly what that resume reads
    /// (`PyFrame::next_instr` = `last_instr + 1`), so an exit coordinate left
    /// behind would restart the frame PAST its own return or raise.  The
    /// recorded `setfield_vable_i` half needs no undo — it only reaches a frame
    /// on a compiled run.
    ///
    /// Only the first publish of a walk is recorded PER FRAME, so the restore
    /// targets the value each frame carried when the walk began rather than an
    /// intermediate one.  More than one frame can be displaced in a walk — the
    /// recording frame's per-opcode publication, and the portal frame's
    /// coordinate when a positive-depth `_getframe` fold lands on it — so this
    /// holds one entry each rather than one entry overall.  Cleared with the
    /// store journal at the start of every walk.
    static FBW_EXIT_LAST_INSTR_UNDO: std::cell::RefCell<Vec<(usize, isize)>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// The `locals_cells_stack_w` image a walk-time `f_locals` fold displaced,
    /// per frame it mirrored into.
    ///
    /// The fold answers `frame.f_locals` by writing the virtualizable shadow
    /// into the live array so the `FrameLocalsProxy` it hands out reads current
    /// values (`virtualizable.py write_boxes`, which `pyframe.py fast2locals`
    /// reaches through `force`).  That write lands whether or not the walk
    /// commits, and a walk that does not commit replays the frame from its
    /// pre-walk instruction — the replay would then re-derive these locals on
    /// top of walk-time values instead of the ones the frame entered with.
    ///
    /// Kept separate from the residual force's `ESCAPE_FLUSH_UNDO` even though
    /// both snapshot the same array: that capture is consumed by the tail
    /// restore in `try_execute_residual_call_via_executor`, which runs after
    /// EVERY non-forcing residual, so a mirror recorded there would be reverted
    /// by the next call in the same walk.  This one is consumed only by the
    /// walk-end legs, beside [`fbw_exit_last_instr_rollback`].
    ///
    /// First write per frame wins, for the same reason as the coordinate undo
    /// above.  Rolled back AFTER `restore_escape_flush_undo`, so where both are
    /// armed for one frame the older (pre-fold) image is the one that stands.
    static FBW_LOCALS_MIRROR_UNDO: std::cell::RefCell<Vec<FbwLocalsMirrorUndo>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// `(frame, valuestackdepth_before)` for the walk-time stores that sync a
    /// live frame's operand-stack depth.
    ///
    /// The sibling of [`FBW_EXIT_LAST_INSTR_UNDO`], and armed for the same
    /// reason: `store_live_frame_static_int` writes field 2 straight into the
    /// concrete frame, so a walk that does not commit leaves the depth the walk
    /// modelled rather than the depth the frame entered with.  It stayed
    /// unjournalled while every frame that field was written on was one the
    /// walk had CREATED — a fresh callee frame a decline simply discards.  A
    /// resumed generator's frame is not fresh: it is the durable frame the
    /// generator will next be resumed from, so its depth has to come back with
    /// `last_instr` and the locals image.
    ///
    /// First write per frame wins, as with both siblings.
    static FBW_FRAME_VSD_UNDO: std::cell::RefCell<Vec<(usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };

    /// `(frame, slot, value, concrete)` for the most recent operand-stack Ref
    /// store a generator-resume sub-walk made into its own suspended frame.
    ///
    /// The yield interception needs the value the body is suspending WITH, and
    /// `emit_abort_permanent!` has already put it there: the marker is preceded
    /// by one `setarrayitem_vable_r` per live operand-stack slot, ascending, so
    /// the last of them writes the top of stack — which for `YIELD_VALUE` is
    /// exactly the value the opcode pops.  Latched at the store rather than
    /// re-read off the frame at the marker because the interception wants the
    /// OpRef as well as the object, and only the store site holds both.
    ///
    /// One entry, last write wins, carrying the frame it was made on and the
    /// slot it landed in.  A nested resume that overwrote it names a different
    /// frame, and the interception then declines rather than taking another
    /// generator's value.
    static FBW_GENERATOR_YIELD_TOS: std::cell::Cell<Option<GeneratorStackStore>> =
        const { std::cell::Cell::new(None) };

    /// Armed by the bridge tracer (`call_jit::trace_and_compile_from_bridge`)
    /// before a single-frame, direct-return-capable guard-failure walk.  When
    /// set, the `run_perfn_walk` epilogue lets a bridge `Terminate` walk keep
    /// the no-replay shortcut — commit the store journal and keep the
    /// finish-concrete stash — so the caller hands the captured result forward
    /// as `DoneWithThisFrame` instead of rewinding to the guard pc and
    /// re-interpreting the region (which would double every eagerly executed
    /// residual side effect, #177).  Only the bridge tracer sets it, and only
    /// when the resume is single-frame; the general guard path consumes the
    /// kept stash as a terminal `BridgeResolution`, and the CALL_ASSEMBLER
    /// callback hands it to its back-to-back blackhole hook, so a committed
    /// journal never strands into a guard-state re-run.  Cleared after
    /// every bridge walk.
    static FBW_BRIDGE_NOREPLAY_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Set when this walk concretely executed a residual call that is not
    /// provably side-effect-free. Such a residual may have committed a heap
    /// effect outside the FBW journals; later exit handling must not replay it.
    static FBW_EXECUTED_NONPURE_RESIDUAL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    /// Set when this walk concretely executed a non-provably-pure residual that
    /// is NOT the self-recursive `CALL_ASSEMBLER` fold target — a foreign body
    /// write (`events.append(n)`).  A self-recursive fold ahead of which such a
    /// residual ran declines, since folding would leave the walk uncommittable
    /// and the interpreter would replay the executed mutation.
    static FBW_EXECUTED_BODY_RESIDUAL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Arm/disarm the bridge `Terminate` no-replay shortcut for the next walk
/// (see [`FBW_BRIDGE_NOREPLAY_ARMED`]).  The bridge tracer sets it before
/// the walk and clears it after.
pub fn fbw_bridge_noreplay_arm(armed: bool) {
    FBW_BRIDGE_NOREPLAY_ARMED.with(|c| c.set(armed));
}

/// Whether the bridge `Terminate` no-replay shortcut is armed for the
/// current walk (read by the `run_perfn_walk` epilogue predicate).
pub(crate) fn fbw_bridge_noreplay_armed() -> bool {
    FBW_BRIDGE_NOREPLAY_ARMED.with(|c| c.get())
}

/// Record that the current walk concretely executed a residual which could
/// have committed non-journaled heap state.
pub(crate) fn fbw_mark_executed_nonpure_residual() {
    FBW_EXECUTED_NONPURE_RESIDUAL.with(|c| c.set(true));
}

/// Clear the executed-residual latch at a walk boundary.
pub(crate) fn fbw_executed_nonpure_residual_reset() {
    FBW_EXECUTED_NONPURE_RESIDUAL.with(|c| c.set(false));
}

/// Record a foreign (non self-recursive) non-pure residual concrete execution.
pub(crate) fn fbw_mark_executed_body_residual() {
    FBW_EXECUTED_BODY_RESIDUAL.with(|c| c.set(true));
}

/// Whether a foreign non-pure residual has concretely executed this walk.
pub(crate) fn fbw_executed_body_residual() -> bool {
    FBW_EXECUTED_BODY_RESIDUAL.with(|c| c.get())
}

/// Clear the foreign-body-residual latch at a walk boundary.
pub(crate) fn fbw_executed_body_residual_reset() {
    FBW_EXECUTED_BODY_RESIDUAL.with(|c| c.set(false));
}

/// Whether the `CALL_ASSEMBLER` fold's own concrete-stamp executor is on the
/// stack (see [`SELFREC_CA_FOLD_ACTIVE`]).
pub(crate) fn selfrec_ca_fold_active() -> bool {
    SELFREC_CA_FOLD_ACTIVE.with(|c| c.get())
}

/// Whether `PYRE_FBW_DEBUG_ABORT` is set.  When on, `full_body_walk_trace`
/// prints the structured reason (the `DispatchError` variant or the
/// non-loop-closing `DispatchOutcome`) for every walk that maps to
/// `TraceAction::Abort` / `AbortPermanent`.  The metainterp's own
/// "abort trace at key={} (permanent={})" log (`pyjitpl.rs`) only
/// reports the key and permanence; the walker-side reason is otherwise
/// swallowed.  Default OFF → no output, zero production effect.
pub fn fbw_debug_abort_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_DEBUG_ABORT").is_some())
}

/// Whether a may-force residual's own fastlocals writes are adopted into the
/// walk (`adopt_residual_locals_writes`).  On unless
/// `PYRE_FBW_NO_ADOPT_RESIDUAL_LOCALS` is set; the switch exists so the two
/// sides can be measured on one binary, since the shape that loses the write
/// takes thousands of iterations to reach and cannot be reproduced by reading
/// `f_locals` (that read forces, and the force publishes the stale box).
pub fn fbw_import_residual_locals_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_NO_ADOPT_RESIDUAL_LOCALS").is_none())
}

/// Clear any stashed Finish payload before a walk begins.  Also clears the
/// concrete-return cell so a stale value from a prior aborted walk cannot leak
/// into this one.
pub(crate) fn fbw_finish_payload_reset() {
    FBW_FINISH_IS_EXCEPTION.with(|c| c.set(false));
    FBW_FINISH_CONCRETE.with(|c| c.set(None));
}

/// Stash the concrete return value of a top-level value-returning
/// `*_return` arm (see [`FBW_FINISH_CONCRETE`]).
pub(crate) fn fbw_finish_concrete_set(value: ConcreteValue) {
    FBW_FINISH_CONCRETE.with(|c| c.set(Some(FinishConcrete::Return(value))));
}

/// Stash the concrete exception object of a top-level uncaught raise.
pub(crate) fn fbw_finish_raise_set(value: ConcreteValue) {
    FBW_FINISH_CONCRETE.with(|c| c.set(Some(FinishConcrete::Raise(value))));
}

/// Peek at the stashed terminal disposition without consuming it (the
/// `run_perfn_walk` epilogue uses this to decide whether to commit the
/// store journal and keep the no-replay shortcut; the CALL_ASSEMBLER
/// bridge callback uses it to leave a kept stash in its rooted cell for
/// the back-to-back blackhole hook).
pub fn fbw_finish_concrete_peek() -> Option<FinishConcrete> {
    FBW_FINISH_CONCRETE.with(|c| c.get())
}

/// Clear the stashed terminal disposition.  The `run_perfn_walk`
/// epilogue calls this when the no-replay shortcut is declined (not a
/// `Terminate` walk, or an unjournaled effect only the replay applies) so
/// the portal degrades to `ContinueRunningNormally`; the CALL_ASSEMBLER
/// blackhole hook calls it so a kept stash that cannot be consumed does
/// not leak into a later portal take.
pub fn fbw_finish_concrete_reset() {
    FBW_FINISH_CONCRETE.with(|c| c.set(None));
}

/// Consume the stashed terminal disposition at the portal exit.
pub fn fbw_finish_concrete_take() -> Option<FinishConcrete> {
    FBW_FINISH_CONCRETE.with(|c| c.take())
}

/// `framework.py root_walker.walk_roots` parity for the concrete terminal
/// value: a `Ref` payload holds a nursery-resident object across the
/// post-walk compile (which allocates and may trigger a minor collection
/// that moves nursery objects), so the slot is forwarded as a root.
/// Registered once via `register_extra_root_walker` at JIT init, mirroring
/// [`fbw_store_journal_root_walker`].
pub fn fbw_finish_concrete_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let data = capture_fbw_finish_concrete_root_area();
    unsafe { fbw_finish_concrete_root_walker_area(data, visitor) };
}

pub fn capture_fbw_finish_concrete_root_area() -> *const () {
    FBW_FINISH_CONCRETE.with(|value| value as *const _ as *const ())
}

/// Record that `op` is a walker-built inline exception (construction fold).
pub(crate) fn fbw_built_exc_insert(op: OpRef) {
    FBW_BUILT_EXC.with(|s| {
        s.borrow_mut().insert(op);
    });
}

/// Consume (remove) `op` from the walker-built-exception set.  Returns
/// `true` if it was present — i.e. the raised value was built inline by
/// [`try_walker_trace_exception_new`].  Removed (not just read) so a
/// second raise of the same object (whose `w_context` is now stamped)
/// takes the residual path. This set tracks symbolic freshness only; the
/// exception's concrete value remains attached to its recorder box.
pub(crate) fn fbw_built_exc_take(op: OpRef) -> bool {
    FBW_BUILT_EXC.with(|s| s.borrow_mut().remove(&op))
}

/// Record that a raise lowering already emitted the `__context__` chaining for
/// `op`, so the in-trace catch compensation leaves it alone.
pub(crate) fn fbw_context_chained_insert(op: OpRef) {
    FBW_CONTEXT_CHAINED.with(|s| {
        s.borrow_mut().insert(op);
    });
}

/// Whether a raise lowering already chained `op`'s `__context__`.
///
/// Read, never consumed: the handler that catches the exception is downstream
/// of the raise in the same trace, and a second stamp there would both
/// duplicate the store and hand the exception to a call — forcing the
/// allocation the optimizer had folded away.
pub(crate) fn fbw_context_chained_contains(op: OpRef) -> bool {
    FBW_CONTEXT_CHAINED.with(|s| s.borrow().contains(&op))
}

/// Clear the store journal and residual-call census before a walk begins.
pub(crate) fn fbw_store_journal_reset() {
    FBW_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_CELL_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_NAMESPACE_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_NAMESPACE_STORE_ROLLED_BACK.with(|c| c.set(false));
    FBW_SYS_EXC_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_UNJOURNALED_VALUE_UNAVAILABLE.with(|c| c.set(false));
    FBW_UNJOURNALED_SYMBOLIC.with(|c| c.set(false));
    FBW_EXECUTED_RESIDUAL_VOID.with(|c| c.set(0));
    FBW_EXECUTED_RESIDUAL_MAYFORCE.with(|c| c.set(0));
    FBW_EXECUTED_RESIDUAL_PLAIN.with(|c| c.set(0));
    // gh#467: reset the executed-effect odometer and any stale
    // forward-flush carrier a prior aborted walk latched.
    FBW_EXECUTED_EFFECT_COUNT.with(|c| c.set(0));
    FBW_OPCODE_ENTRY_EFFECTS.with(|c| c.set(None));
    FBW_QMUT_ABORT_STACK.with(|c| *c.borrow_mut() = None);
    FBW_BRANCH_ABORT_STACK.with(|c| *c.borrow_mut() = None);
    FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS.with(|c| c.set(None));
    FBW_ABORT_CALL_RESUME.with(|c| *c.borrow_mut() = None);
    // #57 Option C: drop any in-flight FOR_ITER items a prior aborted walk
    // left undelivered (its live frame already consumed the delivery), so a
    // stale item cannot be re-delivered by this walk's abort.  This also
    // clears the per-entry body-effect signal so a prior walk's committed
    // mutation cannot block this walk's delivery.
    FBW_FORITER_INFLIGHT.with(|c| c.borrow_mut().clear());
    // ...and with them the census key a take parked for a deliver site that
    // never reported, so the next walk's outcome cannot land on that row.
    FORITER_INFLIGHT_PENDING.with(|slot| slot.set(None));
    // Drop any inline-built-exception OpRef keys a
    // prior aborted walk recorded, so they cannot match a same-numbered
    // OpRef minted by this walk's recorder.
    FBW_BUILT_EXC.with(|s| s.borrow_mut().clear());
    FBW_CONTEXT_CHAINED.with(|s| s.borrow_mut().clear());
    // Drop any unbalanced PUSH_EXC_INFO prev saves a
    // prior aborted walk left (an exception that propagated out without its
    // POP_EXCEPT restore), so a stale saved-prev cannot be popped by an
    // unrelated POP_EXCEPT in this walk.
    FBW_EXC_PREV.with(|s| s.borrow_mut().clear());
    FBW_EXC_PENDING_PUSH_SET.with(|c| c.set(false));
    FBW_EXIT_LAST_INSTR_UNDO.with(|c| c.borrow_mut().clear());
    FBW_LOCALS_MIRROR_UNDO.with(|c| c.borrow_mut().clear());
    FBW_FRAME_VSD_UNDO.with(|c| c.borrow_mut().clear());
    FBW_GENERATOR_YIELD_TOS.with(|c| c.set(None));
}

/// The address a journalled frame lives at NOW.
///
/// A journal entry names its frame by raw address, and a JIT-created frame can
/// be nursery-resident: a minor collection between the eager write and the
/// walk-end restore drags it out and leaves a forwarding stub at the recorded
/// address, so restoring through that address would write the abandoned copy
/// and leave the live frame at its walk-time state.  `PyFrame::live_mut`
/// carries the same reload for the interpreter's own field writes.
///
/// Applied at both ends: at record time so entries key on one identity, and at
/// restore time so an entry recorded before the move still lands.  One
/// application suffices — the drag-out promotes the frame out of the nursery,
/// and nothing outside it moves (`gc_current_object_address`).
fn live_frame_addr(frame: usize) -> usize {
    pyre_object::gc_hook::try_gc_current_object_address(frame as pyre_object::gc_hook::GCREF)
        as usize
}

/// Record the `last_instr` an eager walk-time write is about to displace, so
/// [`fbw_exit_last_instr_rollback`] can put it back when the walk does not
/// commit.  First write per frame wins: that is the coordinate the frame
/// carried when the walk began.
pub(crate) fn fbw_note_last_instr_undo(frame: usize) {
    if frame == 0 {
        return;
    }
    let frame = live_frame_addr(frame);
    FBW_EXIT_LAST_INSTR_UNDO.with(|c| {
        let mut undo = c.borrow_mut();
        if undo.iter().any(|(f, _)| *f == frame) {
            return;
        }
        // SAFETY: the caller holds a live `PyFrame` at this address, and
        // `frame_layout` pins `last_instr` to this offset with a compile-time
        // assertion against the interpreter's own constant.
        let before =
            unsafe { *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *const isize) };
        undo.push((frame, before));
    });
}

/// Put `last_instr` back for a walk that did not commit its end state, so the
/// replay resumes where the frame stood before the walk.  Runs beside
/// [`fbw_store_journal_rollback`] on every non-committed exit; the commit side
/// just drops the undo ([`fbw_exit_last_instr_commit`]).
pub(crate) fn fbw_exit_last_instr_rollback() {
    let undo = FBW_EXIT_LAST_INSTR_UNDO.with(|c| std::mem::take(&mut *c.borrow_mut()));
    // Newest first, so a frame displaced twice — which the per-frame guard
    // above already prevents — could never end on the later value.
    for (frame, before) in undo.into_iter().rev() {
        let frame = live_frame_addr(frame);
        // SAFETY: every frame a publish wrote is live for the whole walk, and
        // `frame_layout` pins `last_instr` to this offset with a compile-time
        // assertion against the interpreter's constant.
        unsafe {
            *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize) = before;
        }
    }
}

/// Drop the undo: the walk's end state is kept, so the published exit
/// coordinate is the one the frame should carry.
pub(crate) fn fbw_exit_last_instr_commit() {
    FBW_EXIT_LAST_INSTR_UNDO.with(|c| c.borrow_mut().clear());
}

/// One frame's pre-fold `locals_cells_stack_w` image.  See
/// [`FBW_LOCALS_MIRROR_UNDO`].
pub(crate) struct FbwLocalsMirrorUndo {
    frame: usize,
    pub(crate) slots: Vec<pyre_object::PyObjectRef>,
}

/// Record the locals image a walk-time `f_locals` mirror is about to
/// overwrite.  `nlocals` is the region the mirror writes — the operand stack
/// above it is deliberately left out, so the rollback cannot revert a stack
/// slot some other walk-time write owns.  First write per frame wins: that is
/// the image the frame carried when the walk began.
pub(crate) fn fbw_note_locals_mirror_undo(frame: usize, nlocals: usize) {
    if frame == 0 {
        return;
    }
    let frame = live_frame_addr(frame);
    FBW_LOCALS_MIRROR_UNDO.with(|c| {
        let mut undo = c.borrow_mut();
        if undo.iter().any(|entry| entry.frame == frame) {
            return;
        }
        // SAFETY: the caller holds a live `PyFrame` at this address for the
        // rest of the walk — it is the frame whose `f_locals` it just folded.
        let pf = unsafe { &*(frame as *const pyre_interpreter::PyFrame) };
        let live = pyre_interpreter::locals_w!(pf).as_slice();
        let slots = live[..nlocals.min(live.len())].to_vec();
        undo.push(FbwLocalsMirrorUndo { frame, slots });
    });
}

/// Non-commit epilogue: put each mirrored frame's pre-fold locals back, so the
/// replay re-derives them from the frame's entry state instead of compounding
/// onto the walk's.  Newest first, matching the store journal.
pub(crate) fn fbw_locals_mirror_rollback() {
    let undo = FBW_LOCALS_MIRROR_UNDO.with(|c| std::mem::take(&mut *c.borrow_mut()));
    for entry in undo.into_iter().rev() {
        let frame = live_frame_addr(entry.frame);
        // SAFETY: as in the capture — the frame is live for the whole walk.
        unsafe {
            let pf = &mut *(frame as *mut pyre_interpreter::PyFrame);
            let arr_ptr = pf.locals_cells_stack_w;
            let dst = pyre_interpreter::locals_w_mut!(pf);
            // The array cannot be reallocated mid-walk, but clamp anyway: a
            // short write is recoverable, a write past the end is not.
            let n = entry.slots.len().min(dst.as_slice().len());
            for (i, &value) in entry.slots.iter().take(n).enumerate() {
                dst[i] = value;
            }
            // A restored value can be nursery-young while the frame and its
            // array are old-gen, and nothing re-traces it unless an owner is in
            // the remembered set.  The forward write arms this per store
            // because boxing allocates between them; the restore writes values
            // that are already boxed, so nothing can collect mid-loop and one
            // arming after it covers every slot.
            crate::state::frame_array_write_barrier(frame as *mut u8, arr_ptr);
        }
    }
}

/// Drop the undo: the walk's end state is kept, so the mirrored locals are the
/// ones the frame should carry (the resumed interpreter reads its fastlocals
/// straight out of that array).
pub(crate) fn fbw_locals_mirror_commit() {
    FBW_LOCALS_MIRROR_UNDO.with(|c| c.borrow_mut().clear());
}

/// Record the `valuestackdepth` a walk-time sync is about to displace, so
/// [`fbw_frame_vsd_rollback`] can put it back when the walk does not commit.
/// First write per frame wins: that is the depth the frame carried when the
/// walk began.
pub(crate) fn fbw_note_frame_vsd_undo(frame: usize) {
    if frame == 0 {
        return;
    }
    let frame = live_frame_addr(frame);
    FBW_FRAME_VSD_UNDO.with(|c| {
        let mut undo = c.borrow_mut();
        if undo.iter().any(|(f, _)| *f == frame) {
            return;
        }
        let Some(before) = crate::state::concrete_stack_depth(frame) else {
            return;
        };
        undo.push((frame, before));
    });
}

/// Put `valuestackdepth` back for a walk that did not commit, so the replay
/// resumes on the operand stack the frame carried before the walk.  Runs beside
/// [`fbw_exit_last_instr_rollback`]; the commit side just drops the undo.
pub(crate) fn fbw_frame_vsd_rollback() {
    let undo = FBW_FRAME_VSD_UNDO.with(|c| std::mem::take(&mut *c.borrow_mut()));
    // Newest first, matching both siblings.
    for (frame, before) in undo.into_iter().rev() {
        crate::state::set_concrete_stack_depth(live_frame_addr(frame), before);
    }
}

/// Drop the undo: the walk's end state is kept, so the synced depth is the one
/// the frame should carry.
pub(crate) fn fbw_frame_vsd_commit() {
    FBW_FRAME_VSD_UNDO.with(|c| c.borrow_mut().clear());
}

/// One operand-stack Ref store made into a resumed generator's own frame.
#[derive(Clone, Copy)]
pub(crate) struct GeneratorStackStore {
    /// The suspended frame the store landed on.
    pub(crate) frame: usize,
    /// Absolute `locals_cells_stack_w` slot it wrote.
    pub(crate) slot: usize,
    /// The stored value.
    pub(crate) value: OpRef,
}

/// Record an operand-stack Ref store a generator-resume sub-walk made into its
/// own suspended frame.  See [`FBW_GENERATOR_YIELD_TOS`].
pub(crate) fn fbw_note_generator_stack_store(store: GeneratorStackStore) {
    FBW_GENERATOR_YIELD_TOS.with(|c| c.set(Some(store)));
}

/// The latched store, if the last one was made on `frame` and landed on `slot`.
pub(crate) fn fbw_generator_stack_store_at(
    frame: usize,
    slot: usize,
) -> Option<GeneratorStackStore> {
    FBW_GENERATOR_YIELD_TOS
        .with(|c| c.get())
        .filter(|store| store.frame == frame && store.slot == slot)
}

/// Arm all three frame undos, called by each walk-time store into a frame the
/// walk did not create, before that store lands.
///
/// The three are otherwise armed separately, each by the site that makes its
/// own kind of write, because every frame those sites reach was built by the
/// walk writing it: a decline discards the frame, so only the one field the
/// caller reads back needs an undo.  A resumed generator's frame is the object
/// the generator will next be resumed from, so a declined walk owes it back
/// whole — locals, operand stack, depth and coordinate alike.
///
/// Arming at the store rather than at walk entry keeps the same snapshot: each
/// note is first-write-per-frame wins, so whichever store gets there first
/// records the state the frame carried before the walk touched it.
///
/// The locals note takes the whole `locals_cells_stack_w` array rather than the
/// `nlocals` prefix its other caller passes.  That caller excludes the operand
/// stack deliberately, so a rollback cannot revert a stack slot another
/// walk-time write owns; here there is no other writer — the frame is suspended
/// and this walk is the only thing running on it — and the operand stack is
/// exactly where a resumed generator's yield state lives.
pub(crate) fn fbw_arm_durable_frame_undo(frame: usize) {
    if frame == 0 {
        return;
    }
    fbw_note_last_instr_undo(frame);
    fbw_note_frame_vsd_undo(frame);
    let whole_array = crate::state::concrete_frame_array_len(frame).unwrap_or(0);
    fbw_note_locals_mirror_undo(frame, whole_array);
}

/// Put one resumed generator frame back without taking the parent walk's
/// undo notes.  The three `*_rollback` helpers `take()` the whole TLS list,
/// which would also revert the caller's `last_instr` / locals while the
/// `FOR_ITER` walk is still running.
pub(crate) fn fbw_durable_frame_rollback_one(frame: usize) {
    if frame == 0 {
        return;
    }
    let frame = live_frame_addr(frame);
    FBW_EXIT_LAST_INSTR_UNDO.with(|c| {
        let mut undo = c.borrow_mut();
        if let Some(i) = undo.iter().position(|(f, _)| *f == frame) {
            let (_, before) = undo.remove(i);
            unsafe {
                *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize) = before;
            }
        }
    });
    FBW_FRAME_VSD_UNDO.with(|c| {
        let mut undo = c.borrow_mut();
        if let Some(i) = undo.iter().position(|(f, _)| *f == frame) {
            let (_, before) = undo.remove(i);
            crate::state::set_concrete_stack_depth(frame, before);
        }
    });
    FBW_LOCALS_MIRROR_UNDO.with(|c| {
        let mut undo = c.borrow_mut();
        if let Some(i) = undo.iter().position(|entry| entry.frame == frame) {
            let entry = undo.remove(i);
            unsafe {
                let pf = &mut *(frame as *mut pyre_interpreter::PyFrame);
                let arr_ptr = pf.locals_cells_stack_w;
                let dst = pyre_interpreter::locals_w_mut!(pf);
                let n = entry.slots.len().min(dst.as_slice().len());
                for (j, &value) in entry.slots.iter().take(n).enumerate() {
                    dst[j] = value;
                }
                crate::state::frame_array_write_barrier(frame as *mut u8, arr_ptr);
            }
        }
    });
}

/// GC: a pre-fold image can be the ONLY reference to a value the mirror
/// displaced, so the root area forwards them (see
/// [`capture_fbw_store_journal_root_area`]).
pub(crate) fn locals_mirror_undo_cell_ptr() -> *const std::cell::RefCell<Vec<FbwLocalsMirrorUndo>> {
    FBW_LOCALS_MIRROR_UNDO.with(|c| c as *const _)
}

/// Record the element a walked eager list store displaces, for rollback
/// when the walk does not commit its end state.
pub(crate) fn fbw_store_journal_push(
    list: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
    displaced: pyre_object::PyObjectRef,
) {
    fbw_store_journal_root(list, key, displaced);
    // gh#467: a journaled store still mutated the heap this iteration; the
    // forward-flush gate counts it so a callee sub-walk that appends/setitems
    // cannot be committed-then-re-executed (a double).
    fbw_bump_executed_effect("store_journal");
}

/// Root `[list, key, displaced]` so a collecting setitem sub-walk forwards
/// the original element.  Does not bump the effect odometer; the caller
/// bumps on a successful store or pops on decline.
pub(crate) fn fbw_store_journal_root(
    list: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
    displaced: pyre_object::PyObjectRef,
) {
    FBW_STORE_JOURNAL.with(|j| j.borrow_mut().push([list, key, displaced]));
}

pub(crate) fn fbw_store_journal_pop() {
    FBW_STORE_JOURNAL.with(|j| {
        let _ = j.borrow_mut().pop();
    });
}

/// Record the live length a walked eager list append grew past, for the
/// length rewind when the walk does not commit its end state.  `list` must
/// be an Integer-strategy list whose backing array had spare capacity (the
/// append's gate), so the rewind is allocation-free.
// Consumed by the #171 `list.append` orthodox descent
// (`try_walker_orthodox_list_append`).
pub(crate) fn fbw_list_journal_push_append(
    list: pyre_object::PyObjectRef,
    length_before: usize,
    allocated_before: isize,
) {
    FBW_LIST_EFFECT_JOURNAL.with(|j| {
        j.borrow_mut().push(FbwListEffect::Append {
            list,
            length_before,
            allocated_before,
        })
    });
    // gh#467: see `fbw_store_journal_push`.
    fbw_bump_executed_effect("list_append");
}

/// Record an eagerly executed Integer-strategy end-pop and its boxed displaced
/// item so a non-commit walk can restore both the length and element even if
/// the list changes strategy before rollback.
pub(crate) fn fbw_list_journal_push_pop_end(
    list: pyre_object::PyObjectRef,
    length_before: usize,
    w_item: pyre_object::PyObjectRef,
) {
    FBW_LIST_EFFECT_JOURNAL.with(|j| {
        j.borrow_mut().push(FbwListEffect::PopEnd {
            list,
            length_before,
            w_item,
        })
    });
    fbw_bump_executed_effect("list_pop_end");
}

/// Record an Empty-list first append whose eager execution promoted the list
/// to a typed strategy, for strategy restore when the walk does not commit.
#[allow(dead_code)]
pub(crate) fn fbw_append_promote_journal_push(list: pyre_object::PyObjectRef) {
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().push(list));
}

pub(crate) fn fbw_append_promote_journal_len() -> usize {
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow().len())
}

/// The list recorded at `index`. The journal is a GC root, so this is the
/// forwarded pointer after `w_list_switch_to_strategy_for` moves the wrapper.
pub(crate) fn fbw_append_promote_journal_at(index: usize) -> Option<pyre_object::PyObjectRef> {
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow().get(index).copied())
}

/// Undo the newest Empty-to-typed promotion this attempt pushed.
///
/// `orthodox_list_append_commit` pushes that entry only after the strategy
/// switch. A guard snapshot can fail earlier, and the list can move during
/// the switch, so the rollback pops the journal's own pointer instead of
/// asserting it still equals the pre-commit reference.
pub(crate) fn fbw_append_promote_journal_rollback_newest() {
    let Some(list) = FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().pop()) else {
        return;
    };
    unsafe { pyre_object::listobject::w_list_clear(list) };
}

/// Shrink a list a declined append sub-walk already grew, when that growth
/// never reached `fbw_list_journal_push_append`.
///
/// The success path of `orthodox_list_append_commit` re-reads the length so
/// it does not store twice. A resume decline returns before that re-read, and
/// the residual fall-through appends again. Absolute `set_len` matches one
/// arm of `fbw_store_journal_rollback`.
pub(crate) fn fbw_rewind_unjournaled_list_append(
    list: pyre_object::PyObjectRef,
    length_before: usize,
    allocated_before: isize,
) {
    unsafe {
        let current = pyre_object::w_list_len(list);
        if current <= length_before {
            return;
        }
        let list_ref = &mut *(list as *mut pyre_object::listobject::W_ListObject);
        match list_ref.strategy {
            pyre_object::listobject::ListStrategy::Object => {
                for index in length_before..current {
                    pyre_object::listobject::ll_list_obj_setitem_fast(
                        list_ref,
                        index,
                        pyre_object::pyobject::PY_NULL,
                    );
                }
                pyre_object::listobject::ll_list_obj_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::Integer => {
                pyre_object::listobject::ll_list_int_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::IntOrFloat => {
                pyre_object::listobject::w_list_int_or_float_set_len(list, length_before);
            }
            pyre_object::listobject::ListStrategy::Float => {
                pyre_object::listobject::ll_list_float_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::Ascii => {
                // The ItemsBlock is scanned over capacity, same as Object
                // storage. Null each vacated `STR` slot before shrinking.
                for index in length_before..current {
                    pyre_object::listobject::ll_list_ascii_setitem_fast(
                        list_ref,
                        index,
                        std::ptr::null(),
                    );
                }
                pyre_object::listobject::ll_list_ascii_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::Empty
            | pyre_object::listobject::ListStrategy::Size
            | pyre_object::listobject::ListStrategy::SimpleRange
            | pyre_object::listobject::ListStrategy::Range
            | pyre_object::listobject::ListStrategy::Bytes => {
                crate::trace::fbw_diag::bump(crate::trace::fbw_diag::STORE_JOURNAL_ROLLBACK_FAILED);
                return;
            }
        }
        pyre_object::listobject::w_list_set_allocated(list, allocated_before);
    }
}

/// Record the `intvalue` a walked eager `IntMutableCell` store displaces,
/// for the in-place restore when the walk does not commit its end state.
// Consumed by the StoreName/StoreGlobal cell fold
// (`try_walker_orthodox_write_cell`).
pub(crate) fn fbw_cell_store_journal_push(cell: pyre_object::PyObjectRef, intvalue_before: i64) {
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-cell-journal] push cell=0x{:x} before={intvalue_before}",
            cell as usize
        );
    }
    FBW_CELL_STORE_JOURNAL.with(|j| {
        j.borrow_mut().push(FbwCellStore::Int {
            cell,
            before: intvalue_before,
        })
    });
    // gh#467: see `fbw_store_journal_push`.
    fbw_bump_executed_effect("cell_store_journal");
}

/// Record the `w_value` a walked eager `ObjectMutableCell` store displaces.
pub(crate) fn fbw_obj_cell_store_journal_push(
    cell: pyre_object::PyObjectRef,
    w_value_before: pyre_object::PyObjectRef,
) {
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-cell-journal] push-obj cell=0x{:x} before=0x{:x}",
            cell as usize, w_value_before as usize
        );
    }
    FBW_CELL_STORE_JOURNAL.with(|j| {
        j.borrow_mut().push(FbwCellStore::Obj {
            cell,
            before: w_value_before,
        })
    });
    fbw_bump_executed_effect("cell_store_journal");
}

/// Record the word a walked `setfield_gc` / `setarrayitem_gc` is about to
/// overwrite in an object the walk did not allocate
/// ([`FbwCellStore::Gc`]).
pub(crate) fn fbw_gc_store_journal_push(
    obj: pyre_object::PyObjectRef,
    offset: usize,
    size: usize,
    before: Value,
    op_count: u32,
) {
    let managed = pyre_object::gc_hook::try_gc_owns_object(obj as pyre_object::gc_hook::GCREF);
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-gc-store-journal] push obj=0x{:x} offset={offset} size={size} before={before:?} managed={managed}",
            obj as usize
        );
    }
    FBW_CELL_STORE_JOURNAL.with(|j| {
        j.borrow_mut().push(FbwCellStore::Gc {
            obj,
            offset,
            size,
            before,
            managed,
            op_count,
        })
    });
    fbw_bump_executed_effect("gc_store_journal");
}

/// [`majit_metainterp::trace_ctx::TraceCtx::cut_observer`] for the walker:
/// undo every journaled [`FbwCellStore::Gc`] store whose op the cut to `pos`
/// discarded, newest first.
///
/// A descent that declines is cut back and its call handed to a residual,
/// which executes the callee's body from the top.  The stores the descent had
/// already executed are in that body, so leaving them in place makes the
/// residual read its own output (`length = length + 1` twice).  The odometer
/// comes back with them: what is undone here was not applied.
pub(crate) fn fbw_gc_store_journal_cut(pos: &majit_metainterp::recorder::TracePosition) {
    let mut undone = 0usize;
    FBW_CELL_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        let mut i = entries.len();
        while i > 0 {
            i -= 1;
            if matches!(entries[i], FbwCellStore::Gc { op_count, .. } if op_count > pos._count) {
                undo_cell_store_entry(entries.remove(i));
                undone += 1;
            }
        }
    });
    fbw_unbump_executed_effects(undone);
}

/// Keep, without an undo entry, the journaled stores into `obj` pushed since
/// `mark`.
///
/// A FOR_ITER consume is the one walked effect a non-committing walk does not
/// put back: the consumed item is delivered to the frame instead
/// ([`FBW_FORITER_INFLIGHT`]), which is only right while the iterator stays
/// advanced.  The cursor stores the descended `next` body executed are that
/// advance, so they leave the journal — and the odometer, which the delivery
/// gate reads as "the body applied something" — the way the residual
/// `for_iter_next` never entered them.
pub(crate) fn fbw_gc_store_journal_keep_since(
    mark: FbwEffectJournalMark,
    obj: pyre_object::PyObjectRef,
) {
    let mut kept = 0usize;
    FBW_CELL_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        let mut i = entries.len();
        while i > mark.cells {
            i -= 1;
            if matches!(entries[i], FbwCellStore::Gc { obj: stored, .. } if stored == obj) {
                entries.remove(i);
                kept += 1;
            }
        }
    });
    fbw_unbump_executed_effects(kept);
}

/// Write `value` into the `size`-byte word at `obj + offset`, the store
/// `bh_setfield_gc_*` / `bh_setarrayitem_gc_*` perform.  `false` for a width
/// the value's bank has no store for.
///
/// # Safety
/// `obj + offset` names a live `size`-byte field or array item of `obj`.
pub(crate) unsafe fn fbw_gc_store_word(
    obj: pyre_object::PyObjectRef,
    offset: usize,
    size: usize,
    value: Value,
    managed: bool,
) -> bool {
    let addr = unsafe { (obj as *mut u8).add(offset) };
    match value {
        Value::Int(v) if matches!(size, 1 | 2 | 4 | 8) => unsafe { raw_store_int(addr, size, v) },
        Value::Float(v) if size == 8 => unsafe { addr.cast::<f64>().write_unaligned(v) },
        Value::Ref(r) if size == std::mem::size_of::<usize>() => {
            unsafe { addr.cast::<usize>().write_unaligned(r.0) };
            if managed {
                pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
            }
        }
        _ => return false,
    }
    true
}

/// Record the namespace binding a `*_NAME` / `*_GLOBAL` residual is about to
/// overwrite, so a walk that hands its region back to the replay can put it
/// back ([`FBW_NAMESPACE_STORE_JOURNAL`]).  `displaced` is null when the name is
/// unbound at store time — a fresh binding, which the rollback removes.
///
/// The residual itself bumps the executed-effect odometer, so unlike
/// [`fbw_cell_store_journal_push`] — which stands in for the residual its fold
/// elided — this push must not bump it a second time.
pub(crate) fn fbw_namespace_store_journal_push(
    namespace: pyre_object::PyObjectRef,
    name: pyre_object::PyObjectRef,
    displaced: pyre_object::PyObjectRef,
    loop_var: bool,
    written: Option<pyre_object::PyObjectRef>,
) {
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-namespace-journal] push ns=0x{:x} name=0x{:x} displaced=0x{:x} \
             loop_var={loop_var}",
            namespace as usize, name as usize, displaced as usize,
        );
    }
    FBW_NAMESPACE_STORE_JOURNAL.with(|j| {
        j.borrow_mut().push(FbwNamespaceStore {
            namespace,
            name,
            displaced,
            loop_var,
            written,
        })
    });
}

/// Whether this walk's rollback undid a namespace binding that an in-flight
/// FOR_ITER delivery would not re-apply for itself.
pub(crate) fn fbw_namespace_store_rolled_back() -> bool {
    FBW_NAMESPACE_STORE_ROLLED_BACK.with(|c| c.get())
}

/// Raise the flag [`fbw_namespace_store_rolled_back`] reads.  Its own name
/// because the delivery refusal is the whole point of the rollback recording
/// anything, and a test can drive the refusal without a live namespace dict.
fn fbw_namespace_store_mark_rolled_back() {
    FBW_NAMESPACE_STORE_ROLLED_BACK.with(|c| c.set(true));
}

/// Current value of an exact-str dict entry, or `None` when reading it would
/// run a stored key's Python `__eq__`. A null pointer inside `Some` is a miss.
fn store_subscr_current_if_callback_free(
    namespace: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
) -> Option<pyre_object::PyObjectRef> {
    unsafe {
        if pyre_object::is_module_dict(namespace) {
            if pyre_object::w_module_dict_is_object_strategy(namespace) {
                return None;
            }
            let key_text = pyre_object::unicodeobject::w_str_get_value_opt(key)?;
            return Some(
                pyre_object::w_module_dict_getitem_str(namespace, key_text)
                    .unwrap_or(std::ptr::null_mut()),
            );
        }
        if pyre_object::w_dict_is_empty_strategy(namespace) {
            return Some(std::ptr::null_mut());
        }
        pyre_object::w_dict_lookup_str_keyed(namespace, key, std::ptr::null_mut())
    }
}

/// Non-commit epilogue for the namespace bindings the walked region wrote:
/// restore each displaced value — or unbind the name that had none — in reverse
/// push order, so the replay re-derives the region's bindings from the pre-walk
/// namespace instead of reading the walk's own.
fn fbw_namespace_store_journal_rollback() {
    FBW_NAMESPACE_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(entry) = entries.pop() {
            // A `STORE_SUBSCR` entry is shared. Restore it only while it is
            // still the object this walk wrote; a later store from another
            // thread stays in place.
            if let Some(written) = entry.written {
                match store_subscr_current_if_callback_free(entry.namespace, entry.name) {
                    Some(current) if current == written => {}
                    _ => continue,
                }
            }
            if !entry.loop_var {
                fbw_namespace_store_mark_rolled_back();
            }
            // SAFETY: the push gated `name` on the residual's own str name
            // operand and `namespace` on a live dict the frame still holds.
            unsafe {
                let name = pyre_object::unicodeobject::w_str_get_wtf8(entry.name);
                if entry.displaced.is_null() {
                    pyre_object::dictmultiobject::w_dict_delitem_wtf8_no_proxy(
                        entry.namespace,
                        name,
                    );
                } else {
                    pyre_object::dictmultiobject::w_dict_setitem_wtf8_no_proxy(
                        entry.namespace,
                        name,
                        entry.displaced,
                    );
                }
            }
        }
    });
}

/// Record the `sys_exc_value` a walked eager `set_current_exception`
/// displaces, for the in-place restore when the walk does not commit its
/// end state.  Pushed by [`try_walker_lower_exc_info_residual`] before it
/// applies the concrete store.
pub(crate) fn fbw_sys_exc_journal_push(displaced: pyre_object::PyObjectRef) {
    FBW_SYS_EXC_JOURNAL.with(|j| j.borrow_mut().push(displaced));
}

/// Read an exception's concrete traceback head before a bridge-entry recording
/// helper runs.  `None` means the concrete is not an exception and therefore
/// cannot receive a host-side attach.
pub(crate) fn fbw_traceback_journal_head(
    exception: pyre_object::PyObjectRef,
) -> Option<pyre_object::PyObjectRef> {
    if exception.is_null() || unsafe { !pyre_object::is_exception(exception) } {
        return None;
    }
    Some(unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) })
}

/// Journal the node a bridge-entry recording helper concretely prepended.
/// An unchanged head means the helper declined the host attach.
pub(crate) fn fbw_traceback_journal_push_if_attached(
    exception: pyre_object::PyObjectRef,
    previous_head: Option<pyre_object::PyObjectRef>,
) {
    let Some(previous_head) = previous_head else {
        return;
    };
    let node = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) };
    if node == previous_head {
        return;
    }
    debug_assert!(!node.is_null());
    debug_assert!(unsafe { pyre_interpreter::pytraceback::is_pytraceback(node) });
    debug_assert_eq!(
        unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_w_next(node) },
        previous_head
    );
    // A carrier sub-walk that keeps its journals across the root continuation
    // attaches once per entered handler, so the log grows rather than asserting
    // a single attach; the rollback unwinds the entries in reverse push order.
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| j.borrow_mut().push((exception, node)));
}

/// Commit-path epilogue: the walk's eager stores and appends stand; drop
/// the undo logs.
pub(crate) fn fbw_store_journal_commit() {
    FBW_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_CELL_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    // A committed walk's end state IS the region's result, so the namespace
    // bindings it wrote are the ones the frame goes on with.
    FBW_NAMESPACE_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    // A committed walk keeps its eager `sys_exc_value` store (the compiled
    // trace or the adopted end state carries the same exception state), so
    // drop the undo log without re-applying it.
    FBW_SYS_EXC_JOURNAL.with(|j| j.borrow_mut().clear());
    // The compiled trace owns the recorded attach and the authoritative walk
    // already applied this iteration's concrete node, so keep it in place.
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    // #57 Option C: a committed walk's end-flush adopts the advanced
    // iterator + the body that consumed it (counted once), so the in-flight
    // items must NOT also be delivered — drop the stash (and with it the
    // per-entry body-effect signals).
    FBW_FORITER_INFLIGHT.with(|c| c.borrow_mut().clear());
}

/// Record a bridge/retrace recording walk's range-iterator cursor before its
/// eager advance, so the abort path can restore it ([`FBW_BRIDGE_ITER_JOURNAL`]).
/// Called from the range FOR_ITER specialization ONLY while `is_bridge_trace`.
pub(crate) fn fbw_bridge_iter_journal_push(
    iter: pyre_object::PyObjectRef,
    pre_current: i64,
    pre_remaining: i64,
) {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| {
        j.borrow_mut().push(BridgeIterJournalEntry::Range {
            iter,
            pre_current,
            pre_remaining,
        })
    });
}

/// Record list-iterator state before a bridge walk mutates it.
pub(crate) fn fbw_bridge_list_iter_journal_push(
    iter: pyre_object::PyObjectRef,
    pre_seq: pyre_object::PyObjectRef,
    pre_index: isize,
) {
    fbw_bridge_cursor_journal_push(BridgeIterKind::List, iter, pre_seq, pre_index);
}

/// Record tuple-iterator state before a bridge walk mutates it.
pub(crate) fn fbw_bridge_tuple_iter_journal_push(
    iter: pyre_object::PyObjectRef,
    pre_seq: pyre_object::PyObjectRef,
    pre_index: isize,
) {
    fbw_bridge_cursor_journal_push(BridgeIterKind::Tuple, iter, pre_seq, pre_index);
}

fn fbw_bridge_cursor_journal_push(
    kind: BridgeIterKind,
    iter: pyre_object::PyObjectRef,
    pre_seq: pyre_object::PyObjectRef,
    pre_index: isize,
) {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| {
        j.borrow_mut().push(BridgeIterJournalEntry::Cursor {
            kind,
            iter,
            pre_seq,
            pre_index,
        })
    });
}

/// Whether [`fbw_bridge_iter_journal_capture`] can record `iter`.
///
/// List, list-reverse, and tuple iterators move only a cursor.  A generator,
/// `map`, dict/set iterator, itertools iterator, or user `__next__` does not:
/// nothing is recorded, and entry replay would call `__next__` again.
/// `try_walker_specialize_for_iter_next` folds a machine-word `range` before
/// this residual and journals that fold's own advance.
pub(crate) fn fbw_bridge_iter_journalable(iter: pyre_object::PyObjectRef) -> bool {
    if iter.is_null() {
        return false;
    }
    unsafe {
        pyre_object::is_list_iter(iter)
            || pyre_object::is_list_reverse_iter(iter)
            || pyre_object::is_tuple_iter(iter)
    }
}

/// Snapshot `iter`'s cursor.  Returns whether a cursor entry was pushed.
pub(crate) fn fbw_bridge_iter_journal_capture(iter: pyre_object::PyObjectRef) -> bool {
    if !fbw_bridge_iter_journalable(iter) {
        return false;
    }
    unsafe {
        let kind = if pyre_object::is_list_iter(iter) {
            BridgeIterKind::List
        } else if pyre_object::is_list_reverse_iter(iter) {
            BridgeIterKind::ListReverse
        } else {
            BridgeIterKind::Tuple
        };
        let (pre_seq, pre_index) = match kind {
            BridgeIterKind::List => (
                pyre_object::w_list_iter_seq(iter),
                pyre_object::w_list_iter_index(iter),
            ),
            BridgeIterKind::ListReverse => (
                pyre_object::w_list_reverse_iter_seq(iter),
                pyre_object::w_list_reverse_iter_index(iter),
            ),
            BridgeIterKind::Tuple => (
                pyre_object::w_tuple_iter_seq(iter),
                pyre_object::w_tuple_iter_index(iter),
            ),
        };
        fbw_bridge_cursor_journal_push(kind, iter, pre_seq, pre_index);
    }
    true
}

/// Non-commit epilogue for a bridge/retrace recording walk: restore each
/// iterator to the state it held before the walk mutated it, in reverse push
/// order.  The interpreter resume then re-consumes the item the aborted
/// recording had taken, so the iteration is executed exactly once.
pub(crate) fn fbw_bridge_iter_journal_rollback() {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(entry) = entries.pop() {
            match entry {
                BridgeIterJournalEntry::Range {
                    iter,
                    pre_current,
                    pre_remaining,
                } => unsafe {
                    pyre_object::functional::w_range_iter_set_cursor(
                        iter,
                        pre_current,
                        pre_remaining,
                    );
                },
                BridgeIterJournalEntry::Cursor {
                    kind,
                    iter,
                    pre_seq,
                    pre_index,
                } => unsafe {
                    let (set_seq, set_index): (
                        unsafe fn(pyre_object::PyObjectRef, pyre_object::PyObjectRef),
                        unsafe fn(pyre_object::PyObjectRef, isize),
                    ) = match kind {
                        BridgeIterKind::List => (
                            pyre_object::w_list_iter_set_seq,
                            pyre_object::w_list_iter_set_index,
                        ),
                        BridgeIterKind::ListReverse => (
                            pyre_object::w_list_reverse_iter_set_seq,
                            pyre_object::w_list_reverse_iter_set_index,
                        ),
                        BridgeIterKind::Tuple => (
                            pyre_object::w_tuple_iter_set_seq,
                            pyre_object::w_tuple_iter_set_index,
                        ),
                    };
                    set_seq(iter, pre_seq);
                    set_index(iter, pre_index);
                },
            }
        }
    });
}

/// Commit epilogue: a committed bridge recording keeps its advanced cursor
/// (the compiled bridge adopts it as the authoritative continuation), so drop
/// the undo log without restoring.
pub(crate) fn fbw_bridge_iter_journal_clear() {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| j.borrow_mut().clear());
}

/// Record the in-flight FOR_ITER continuation (#57 Option C): the consumed
/// item the `for_iter_next` residual produced and its FOR_ITER body coordinate.
/// Called from the residual
/// executor's success arm when the helper is [`RuntimeHelperKind::ForIterNext`]
/// and it produced a non-null item (a null item is the exhaustion arm — no
/// body runs, nothing to deliver).  The stack mirrors loop nesting: a consume
/// of a DIFFERENT (deeper) FOR_ITER pushes a new entry on top of the loops
/// that enclose it, while a re-consume of a FOR_ITER ALREADY on the stack is
/// that exact JitCode/opcode advancing to its next iteration — every entry
/// above it belongs to nested loops that have run to completion inside the
/// prior body, so they are popped, and the loop's own entry is replaced (a
/// fresh body-effect window).  The outer loop's in-flight item is thus no
/// longer destroyed by an inner consume, and a completed inner loop leaves no
/// stale entry.
pub(crate) fn fbw_foriter_inflight_capture(
    item: pyre_object::PyObjectRef,
    body: InflightForiterBody,
    consume_journaled: bool,
) {
    FBW_FORITER_INFLIGHT.with(|c| {
        let mut stack = c.borrow_mut();
        // The "body effect since consume" window restarts at each consume:
        // only effects committed after THIS consume can double on a re-run of
        // THIS iteration's body (Finding #1).  A fresh entry starts clear.
        // `consume_journaled` is the cursor snapshot from
        // [`fbw_bridge_iter_journal_capture`] (or a range/zip fold's own push).
        // An unjournaled consume is not replay-safe: `convert_and_run_from_pyjitpl`
        // has to continue the frame, because entry replay calls `__next__` again.
        let entry = InflightForiter {
            item,
            body,
            body_effect_since_consume: false,
            body_completed: false,
            consume_journaled,
        };
        let Some(_body_pc) = inflight_foriter_body_pc(body) else {
            // An unresolvable native coordinate cannot identify an existing
            // loop. Keep this item as a distinct entry; later consumers also
            // refuse it conservatively instead of guessing a Python pc.
            stack.push(entry);
            return;
        };
        match stack.iter().position(|e| e.body == body) {
            Some(at) => {
                stack.truncate(at + 1);
                stack[at] = entry;
            }
            None => stack.push(entry),
        }
    });
}

/// Whether an in-flight FOR_ITER item is currently captured (a consume ran
/// this iteration and no commit/abort has cleared it yet).  Sampled by the
/// residual executor to decide whether a non-elidable concrete mutation
/// counts as a body effect committed after the consume (Finding #1).
pub(crate) fn fbw_foriter_inflight_active() -> bool {
    FBW_FORITER_INFLIGHT.with(|c| !c.borrow().is_empty())
}

/// An in-flight consume whose cursor [`fbw_bridge_iter_journal_capture`] did
/// not record.  Entry replay would call `__next__` again;
/// `convert_and_run_from_pyjitpl` continues the frame instead.
pub(crate) fn fbw_foriter_unjournaled_consume() -> bool {
    FBW_FORITER_INFLIGHT.with(|c| c.borrow().iter().any(|entry| !entry.consume_journaled))
}

/// Mark the in-flight entry for `body` body-completed: a NEW
/// `for_iter_next` attempt is being dispatched for the same FOR_ITER, so the
/// prior consumed item's body has run to completion (the walk is back at the
/// header).  Called from the residual dispatch BEFORE the call executes so an
/// attempt that aborts mid-way (a kept-stack guard on the exhaustion arm)
/// still leaves the completion recorded; a successful attempt replaces the
/// entry with a fresh one anyway ([`fbw_foriter_inflight_capture`]).
pub(crate) fn fbw_foriter_inflight_mark_attempt(body: InflightForiterBody) {
    FBW_FORITER_INFLIGHT.with(|c| {
        let Some(_body_pc) = inflight_foriter_body_pc(body) else {
            return;
        };
        if let Some(entry) = c.borrow_mut().iter_mut().find(|e| e.body == body) {
            entry.body_completed = true;
        }
    });
}

/// Flag that a non-elidable concrete residual committed an irreversible heap
/// mutation after the in-flight FOR_ITER consume (Finding #1, R1).  A mutation
/// committed while several FOR_ITER items are in flight is "after" every one
/// of them — re-running ANY of their bodies on delivery re-applies it — so
/// mark every active entry.
pub(crate) fn fbw_mark_foriter_body_effect_since_consume() {
    FBW_FORITER_INFLIGHT.with(|c| {
        for entry in c.borrow_mut().iter_mut() {
            entry.body_effect_since_consume = true;
        }
    });
}

/// Drop every in-flight FOR_ITER entry (#32 S2): a committed branch-flush has
/// adopted the walk's end state and owns the iteration count, so no item may be
/// delivered afterward.
pub fn fbw_foriter_inflight_clear() {
    FBW_FORITER_INFLIGHT.with(|c| c.borrow_mut().clear());
}

/// #32 S2 deliver selector for the branch-flush leg.  Returns
/// `Some((item, body_pc))` to push at the body ONLY when `resume_py_pc` is the
/// header of a FOR_ITER whose consumed item is in flight (`body_pc ==
/// resume_py_pc + 1`, and the opcode there really is a FOR_ITER) — Shape A, the
/// abort parked on the FOR_ITER before its body ran, so the item is not yet on
/// the flushed header stack and must be delivered.  Returns `None` when the
/// resume pc is not such a header, or when the matching entry carries a
/// body-effect signal (the R1 never-double guard: re-running the body would
/// re-apply an irreversible mutation).  Read-only — the caller drops the stash
/// via [`fbw_foriter_inflight_clear`] only after the flush commits, so a
/// declined flush leaves the in-flight items intact for the legacy deliver.
pub fn fbw_foriter_inflight_take_for_resume(
    frame: usize,
    resume_py_pc: usize,
) -> Option<(pyre_object::PyObjectRef, usize)> {
    let body_pc = resume_py_pc + 1;
    if !foriter_header_at(frame, resume_py_pc) {
        return None;
    }
    FBW_FORITER_INFLIGHT.with(|c| {
        let stack = c.borrow();
        let at = stack
            .iter()
            .position(|e| foriter_body_matches_frame(e.body, frame, body_pc))?;
        // R1 never-double guard (cross-checks #33): an irreversible body effect
        // committed since this consume means re-running the body on delivery
        // would double it — refuse delivery.  A body-COMPLETED entry (the walk
        // re-reached the consume, so this item's body already ran) must never
        // be delivered either — that is the header-flush-without-delivery
        // shape ([`fbw_foriter_inflight_completed_at_resume`]).  Also refuse if
        // either journal is non-empty or an unjournaled effect stands. Unlike
        // legacy replay delivery, this branch adopts the walk's end state, so
        // recorded-but-unexecuted residual work also makes it unsafe to flush.
        if stack[at].body_effect_since_consume
            || stack[at].body_completed
            || fbw_store_journal_len() != 0
            || FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len()) != 0
            || FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len()) != 0
            || fbw_has_unjournaled_effect()
        {
            return None;
        }
        Some((stack[at].item, body_pc))
    })
}

/// #493 selector for the header-flush-without-delivery shape: `resume_py_pc`
/// is a FOR_ITER header whose in-flight entry is body-COMPLETED — the abort
/// fired during the NEXT consume attempt (a kept-stack guard on the FOR_ITER
/// arms after the `for_iter_next` residual), so the consumed item's body
/// already ran during the walk.  The walk end state at the header is then the
/// complete post-body state: the flush adopts it WITHOUT delivering the item
/// and the interpreter re-attempts the consume against the advanced iterator.
/// Refuses when an effect committed since the consume (re-attempting the
/// consume could re-apply the failed attempt's effect) — same signals as the
/// delivery selector above.
pub fn fbw_foriter_inflight_completed_at_resume(frame: usize, resume_py_pc: usize) -> bool {
    let body_pc = resume_py_pc + 1;
    if !foriter_header_at(frame, resume_py_pc) {
        return false;
    }
    FBW_FORITER_INFLIGHT.with(|c| {
        let stack = c.borrow();
        let Some(at) = stack
            .iter()
            .position(|e| foriter_body_matches_frame(e.body, frame, body_pc))
        else {
            return false;
        };
        stack[at].body_completed
            && !stack[at].body_effect_since_consume
            && fbw_store_journal_len() == 0
            && FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len()) == 0
            && FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len()) == 0
            && !fbw_has_unjournaled_effect()
    })
}

/// Whether a body effect committed since the most-recent in-flight FOR_ITER
/// consume (Finding #1, R1) — the top entry, the one [`fbw_foriter_inflight_take`]
/// delivers.
pub(crate) fn fbw_foriter_body_effect_since_consume() -> bool {
    FBW_FORITER_INFLIGHT
        .with(|c| c.borrow().last().map(|e| e.body_effect_since_consume))
        .unwrap_or(false)
}

/// Resolved `body_pc` of the most-recent (top) in-flight FOR_ITER entry — the
/// FOR_ITER continue-arm fallthrough, i.e. the pc of the loop-variable store
/// that binds the just-consumed item.  `None` when no item is in flight or its
/// coordinate is an unresolvable native pc.
pub(crate) fn fbw_foriter_inflight_top_body_pc() -> Option<usize> {
    FBW_FORITER_INFLIGHT.with(|c| {
        c.borrow()
            .last()
            .and_then(|e| inflight_foriter_body_pc(e.body))
    })
}

/// Whether ANY R1 committed-body-effect signal is currently present: the
/// body-effect-since-consume flag or either journal non-empty. These are the
/// exact signals
/// [`fbw_foriter_inflight_take`] consults to REFUSE delivery, and `take`
/// leaves them untouched.  Exposed for the deliver-path loud-failure
/// debug-assert (#57 Finding #3): a successful take (delivery) while any
/// signal stands would be a silent double, so the deliver site asserts this
/// is `false` in debug builds. An unjournaled effect is deliberately NOT a
/// member: it is recorded-but-unexecuted work that only this legacy replay
/// will apply, so it requires delivery rather than making delivery unsafe.
pub fn fbw_foriter_any_body_effect_signal() -> bool {
    fbw_foriter_body_effect_since_consume()
        || fbw_store_journal_len() != 0
        || FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len()) != 0
        || FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len()) != 0
}

/// Take the in-flight FOR_ITER continuation for delivery on a trace abort
/// (#57 Option C).  Returns `(consumed_item, body_pc)` and clears the stash
/// so it is delivered at most once.
///
/// R1 (double-apply guard): delivery resumes the live frame at the FOR_ITER
/// body, so any body op that ALREADY ran concretely during the aborted walk
/// would be re-applied.  C may DELIVER only when it can PROVE no body effect
/// committed for the in-flight iteration — then re-running the body cannot
/// double. Two signal classes cover every committed body effect:
///
/// * `fbw_foriter_body_effect_since_consume()` — a non-elidable concrete
///   residual mutated the heap OUTSIDE the journals after the consume (a dict
///   `store_subscr_fn`, an unmodeled container method).  Irreversible: the
///   mutation already stands on the live heap, so a body re-run would double
///   it (Finding #1).
/// * either journal non-empty (`FBW_STORE_JOURNAL` list setitem /
///   `FBW_LIST_EFFECT_JOURNAL` list append/pop). On the production abort path
///   `fbw_store_journal_rollback` empties these BEFORE this take, so this is
///   normally false here; the check is a belt-and-suspenders refusal in case
///   a future caller takes before the rollback.
/// Any signal set → refuse delivery (drop the stash → the legacy bypass keeps
/// the prior drop-on-abort behaviour for that shape, never a double).
/// `fbw_has_unjournaled_effect()` is the converse: the residual was recorded
/// but NOT executed, and only the legacy replay applies it. It prevents an
/// end-state commit, but it must not suppress delivery of the consumed item
/// or that replay starts at the next iteration and drops this one.
/// `for_mutate` aborts BEFORE the append's effect, so both signal classes are
/// clear at the abort point — the clean continuation case.
/// Which stash entry a take removes, given the indices that matched the parked
/// frame.  `None` only for an empty stash.
///
/// `foriter_body_matches_frame` decides on the frame's CODE and body pc, not on
/// the frame itself — a frame address cannot be the key, since a minor
/// collection relocates the frame between the capture and the take.  Two
/// activations of one code (a recursive call, or one function inlined at two
/// levels under different JitCode identities) therefore answer that match
/// identically, and reaching past the top for one of them would deliver another
/// activation's item into this frame.  So the match decides only while it is
/// UNAMBIGUOUS; a second candidate falls back to the legacy top entry, whose
/// header check then refuses and drops exactly as it did before.  Wrong-frame
/// delivery is a wrong VALUE and the drop is only a loss, so never trading one
/// for the other is what makes reaching past the top safe at all.
fn inflight_take_index(len: usize, mut matching: impl Iterator<Item = usize>) -> Option<usize> {
    match (matching.next(), matching.next()) {
        (Some(only), None) => Some(only),
        _ => len.checked_sub(1),
    }
}

/// The census key for one stash entry.  The live frame the legacy delivery
/// site passes is the portal frame, but an inline sub-walk's consumed item
/// belongs to its own JitCode, so an entry that names one is keyed by THAT
/// code; a legacy Python-pc entry, and an unresolvable identity (a negative
/// index), fall back to the live frame's.  The resolution borrows the
/// metainterp state, so callers must ask only while the census is collecting.
fn inflight_census_key(
    fallback_code_ptr: usize,
    body: InflightForiterBody,
) -> Option<(usize, usize)> {
    let body_pc = inflight_foriter_body_pc(body)?;
    let code_ptr = match body {
        InflightForiterBody::Jit { jitcode_index, .. } => {
            crate::state::raw_code_for_jitcode_index(jitcode_index)
                .map_or(fallback_code_ptr, |code| code as usize)
        }
        _ => fallback_code_ptr,
    };
    Some((code_ptr, body_pc))
}

thread_local! {
    /// The census key of the entry [`fbw_foriter_inflight_take`] last handed
    /// out.  The take is the only place that can resolve that key, but only
    /// the deliver site knows whether the push happened, so the key waits here
    /// for whichever of [`fbw_foriter_report_delivered`] /
    /// [`fbw_foriter_report_refused_header`] reports.  It holds no object
    /// reference, so it is not a GC root area the way the stash itself is.
    static FORITER_INFLIGHT_PENDING: std::cell::Cell<Option<(usize, usize)>> =
        const { std::cell::Cell::new(None) };
    /// `consume_journaled` of the entry [`fbw_foriter_inflight_take`] last
    /// handed out.  [`fbw_foriter_report_refused_header`] reads it: a journaled
    /// cursor that [`fbw_bridge_iter_journal_rollback`] restored is re-read,
    /// not lost.
    static FORITER_INFLIGHT_PENDING_JOURNALED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

fn census_report_pending(outcome: super::ForiterInflightOutcome) {
    if !super::foriter_inflight_census_enabled() {
        return;
    }
    if let Some((code_ptr, body_pc)) = FORITER_INFLIGHT_PENDING.with(|slot| slot.take()) {
        super::census_record_foriter_inflight(code_ptr, body_pc, outcome);
    }
}

/// Report that the item the take handed out was pushed and the frame
/// repositioned to its body pc — the one outcome that costs no iteration.
pub fn fbw_foriter_report_delivered() {
    // The delivered item owns the advanced cursor.  Drop the pre-advance
    // snapshot so a later refuse cannot roll this consume back.
    fbw_bridge_iter_journal_clear();
    FORITER_INFLIGHT_PENDING_JOURNALED.with(|slot| slot.set(false));
    census_report_pending(super::ForiterInflightOutcome::Delivered);
}

/// Report that the take's item was dropped after all, because the live frame
/// was not parked at the header its body pc names.  That is one lost iteration,
/// so it bumps the gated tally as well as the census column.
pub fn fbw_foriter_report_refused_header() {
    // The frame is parked at the pc the walk entered from, which is before
    // everything the walk executed, so the resume re-reaches this FOR_ITER.
    // Restoring the cursor makes it re-consume the item the walk took; without
    // it the resume reads the NEXT one and the iteration is lost.  The R1
    // refusal below restores the cursor on the same reasoning, but only on its
    // no-committed-effect half: re-consuming re-runs the body, which would
    // double an effect that already stands on the live heap.
    //
    // A journaled cursor that this rollback (or the bridge epilogue's earlier
    // one) put back is re-read by the resume, not lost.  An unjournaled consume
    // has nothing to restore, so its item is dropped.
    let journaled = FORITER_INFLIGHT_PENDING_JOURNALED.with(|slot| slot.replace(false));
    fbw_bridge_iter_journal_rollback();
    if !journaled {
        crate::trace::fbw_diag::record_foriter_item_dropped();
    }
    census_report_pending(super::ForiterInflightOutcome::RefusedHeader);
}

pub fn fbw_foriter_inflight_take(
    fallback_code_ptr: usize,
    frame: usize,
    resume_py_pc: usize,
) -> Option<(pyre_object::PyObjectRef, usize)> {
    // Take the entry that belongs to the header the live frame is PARKED at,
    // then drop the rest: one take still delivers at most one item and leaves
    // nothing for a subsequent deliver call.
    //
    // Selecting the most-recent entry instead loses an iteration whenever two
    // FOR_ITERs of one frame are in flight at once, which needs no nesting of
    // frames: an outer `list` loop captures through the residual leg
    // (`residual_call.rs`, `ForIterNext`) while a `range` body inside it
    // captures through the specialised leg (`specialize.rs`, `w_range_iter_next`).
    // `deliver_inflight_foriter_item` can only push at the header the frame is
    // parked on, so a most-recent entry naming the INNER body pc fails its
    // header check and is refused — and the `clear` below has already destroyed
    // the outer entry, which that same frame COULD have taken. Neither item is
    // delivered and the outer iteration is lost. Matching the frame first
    // delivers the deliverable one; the entries that no frame can accept keep
    // the single-slot drop.
    //
    let census_on = super::foriter_inflight_census_enabled();
    let (stash, dropped) = FBW_FORITER_INFLIGHT.with(|c| {
        let mut stack = c.borrow_mut();
        let Some(at) = inflight_take_index(
            stack.len(),
            stack
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    foriter_body_matches_frame(entry.body, frame, resume_py_pc + 1)
                })
                .map(|(at, _)| at),
        ) else {
            return (None, Vec::new());
        };
        let picked = stack.remove(at);
        // Everything still standing is about to be destroyed by the clear, and
        // each one is an iteration whose consumed item nothing will hand back.
        // Name them before they go: a census that reports only what the take
        // RETURNED cannot see this column at all.
        let dropped = if census_on {
            stack.iter().map(|entry| entry.body).collect()
        } else {
            Vec::new()
        };
        stack.clear();
        (Some(picked), dropped)
    });
    let stash = stash?;
    let body_effect = stash.body_effect_since_consume;
    let Some(body_pc) = inflight_foriter_body_pc(stash.body) else {
        // The entry is already out of the stash and its coordinate cannot be
        // resolved to a Python body pc, so nothing downstream can push it: the
        // iteration it belongs to is lost the same way the two refusals below
        // lose one.
        crate::trace::fbw_diag::record_foriter_item_dropped();
        return None;
    };
    for body in dropped {
        if let Some((code_ptr, dropped_pc)) = inflight_census_key(fallback_code_ptr, body) {
            super::census_record_foriter_inflight(
                code_ptr,
                dropped_pc,
                super::ForiterInflightOutcome::DroppedUnmatched,
            );
        }
    }
    let key = if census_on {
        inflight_census_key(fallback_code_ptr, stash.body)
    } else {
        None
    };
    let store_len = fbw_store_journal_len();
    let append_len = FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len());
    let cell_store_len = FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len());
    let unjournaled = fbw_has_unjournaled_effect();
    // A rolled-back namespace binding the delivery cannot re-apply: the
    // reposition runs the FOR_ITER body, not the region ahead of it, so a
    // binding written before the consume would stay undone.
    let namespace_rolled_back = fbw_namespace_store_rolled_back();
    let committed_effect = body_effect || store_len != 0 || append_len != 0 || cell_store_len != 0;
    if committed_effect || namespace_rolled_back {
        // Same cursor restore as `fbw_foriter_report_refused_header`, and only
        // on the half that has committed nothing.  A root walk's non-commit
        // epilogue leaves the journal in place so a delivery can push the
        // consumed item; refusing here without putting the cursor back loses
        // that iteration (`fbw_foriter_item_dropped`).  That is the shape the
        // rollback was added for: a `GotoIfNotValueNotConcrete` abort on the
        // first body opcode, whose only standing signal is a namespace binding
        // that was itself rolled back, so re-consuming replays a body that has
        // run to none of its stores.
        //
        // `committed_effect` is the opposite case and keeps drop-on-abort: an
        // effect already stands on the live heap for this iteration, and a
        // rewound cursor would re-run the body and double it.
        if !committed_effect {
            fbw_bridge_iter_journal_rollback();
        }
        crate::trace::fbw_diag::record_foriter_item_dropped();
        if let Some((code_ptr, body_pc)) = key {
            super::census_record_foriter_inflight(
                code_ptr,
                body_pc,
                super::ForiterInflightOutcome::RefusedEffect,
            );
        }
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] deliver REFUSED (body effect committed since consume) body_pc={} \
                 body_effect={body_effect} store_journal_len={store_len} \
                 append_journal_len={append_len} unjournaled={unjournaled} \
                 namespace_rolled_back={namespace_rolled_back} \
                 — restored the iterator cursor so resume re-consumes",
                body_pc
            );
        }
        return None;
    }
    // NOT recorded as delivered here: the deliver site still refuses on its
    // header check, and counting the take as a delivery is what let a refusal
    // read as a success. Park the key for whichever of the two reports runs.
    FORITER_INFLIGHT_PENDING.with(|slot| slot.set(key));
    FORITER_INFLIGHT_PENDING_JOURNALED.with(|slot| slot.set(stash.consume_journaled));
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-foriter] deliver item=0x{:x} body_pc={} store_journal_len={store_len} \
             unjournaled={unjournaled}",
            stash.item as usize, body_pc,
        );
    }
    Some((stash.item, body_pc))
}

#[cfg(test)]
mod foriter_delivery_tests {
    use super::*;

    /// The census columns, for a test that just drove one outcome.  Reading
    /// them is why `foriter_inflight_census_enabled` is unconditionally true
    /// under `cfg(test)`: no corpus fixture reaches a refusal, so these are
    /// the only place the refusal columns are ever non-zero.
    fn row(body_pc: usize) -> super::super::ForiterInflightCensusCounts {
        super::super::census_foriter_inflight_row(0, body_pc)
    }

    fn reset() {
        fbw_store_journal_reset();
        super::super::census_foriter_inflight_reset();
    }

    /// The gated `fbw_foriter_item_dropped` tally.  It is one process-wide
    /// atomic while the census columns are thread-local, so a test asserts the
    /// DIRECTION of its own bump rather than an absolute value — another test
    /// on another thread may bump it in between.  What is being proven here is
    /// the wiring: the census column and the tally move on the same line.
    fn dropped_tally() -> u64 {
        crate::trace::fbw_diag::get(crate::trace::fbw_diag::FORITER_ITEM_DROPPED)
    }

    /// A take that hands an item out has NOT delivered it: the deliver site
    /// still refuses on its header check.  Counting the take as the delivery
    /// is what made a refusal read as a success.
    #[test]
    fn the_take_alone_records_no_outcome() {
        reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);

        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), Some((item, 34)));

        let r = row(34);
        assert_eq!(
            (
                r.delivered,
                r.refused_effect,
                r.refused_header,
                r.dropped_unmatched
            ),
            (0, 0, 0, 0),
            "the take is not an outcome"
        );
        reset();
    }

    #[test]
    fn a_push_at_the_header_is_the_only_delivery() {
        reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);
        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), Some((item, 34)));

        fbw_foriter_report_delivered();

        assert_eq!(row(34).delivered, 1);
        assert_eq!(row(34).refused_header, 0);
        reset();
    }

    /// The conflation this census was fixed for: the deliver site refuses when
    /// the live frame is parked somewhere else, and that refusal used to be
    /// indistinguishable from a delivery.
    #[test]
    fn a_header_refusal_is_not_a_delivery() {
        reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);
        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), Some((item, 34)));

        let before = dropped_tally();
        fbw_foriter_report_refused_header();

        assert_eq!(row(34).refused_header, 1);
        assert_eq!(row(34).delivered, 0, "a refusal must not read as a success");
        assert!(
            dropped_tally() > before,
            "a header refusal loses the iteration and must reach the gated tally"
        );
        reset();
    }

    /// R1 gets its own column, because it is the only refusal that is about an
    /// effect: the other two are about where the frame is parked.
    #[test]
    fn an_r1_refusal_is_its_own_column() {
        reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);
        fbw_mark_foriter_body_effect_since_consume();

        let before = dropped_tally();
        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), None);

        let r = row(34);
        assert_eq!((r.refused_effect, r.refused_header, r.delivered), (1, 0, 0));
        assert!(
            dropped_tally() > before,
            "the R1 refusal loses the iteration and must reach the gated tally"
        );
        reset();
    }

    /// The take clears every entry it did not pick, and each one is an
    /// iteration whose consumed item nothing will hand back.  A census that
    /// reports only what the take RETURNED cannot see them at all, so they are
    /// named before the clear.
    #[test]
    fn the_entries_the_clear_destroys_are_named() {
        reset();
        let outer = 1usize as pyre_object::PyObjectRef;
        let inner = 2usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(outer, InflightForiterBody::Py(6), false);
        fbw_foriter_inflight_capture(inner, InflightForiterBody::Py(35), false);

        // Parked at the OUTER header, so the inner entry is the one destroyed.
        assert_eq!(fbw_foriter_inflight_take(0, 0, 5), Some((outer, 6)));

        assert_eq!(row(35).dropped_unmatched, 1, "the inner entry is gone");
        assert_eq!(
            row(6).dropped_unmatched,
            0,
            "the taken entry is not dropped"
        );
        reset();
    }

    #[test]
    fn recorded_but_unexecuted_residual_keeps_consumed_item_for_replay() {
        fbw_store_journal_reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);
        fbw_mark_unjournaled_effect(ResidualDecline::Symbolic);

        assert!(!fbw_foriter_any_body_effect_signal());
        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), Some((item, 34)));

        fbw_store_journal_reset();
    }

    /// A `PureUnfolded` decline is the one cause that must leave the walk
    /// unmarked: the call it stands for applies no effect, so there is no
    /// pending write a replay could be the only carrier of.  Marking it would
    /// shut every no-replay walk-end road and leave replay — which re-applies
    /// the already-executed effects — as the only exit.  Guard both flag
    /// components, not just the aggregate, so a future rewrite of the `match`
    /// cannot set one of them and still read false through some other path.
    #[test]
    fn a_pure_unfolded_decline_leaves_the_walk_unmarked() {
        fbw_store_journal_reset();
        assert!(!fbw_has_unjournaled_effect());

        fbw_mark_unjournaled_effect(ResidualDecline::PureUnfolded);

        assert_eq!(fbw_unjournaled_kinds(), (false, false));
        assert!(!fbw_has_unjournaled_effect());

        fbw_store_journal_reset();
    }

    /// Two FOR_ITERs of one frame can be in flight at once (an outer `list`
    /// loop through the residual leg, a `range` body inside it through the
    /// specialised leg). The deliver site can only push at the header the
    /// frame is parked on, so the take must select on that coordinate: taking
    /// the most-recent entry returns a body pc this frame will refuse AND
    /// discards the entry it could have delivered, losing the outer iteration.
    #[test]
    fn take_selects_the_entry_the_parked_frame_can_accept_not_the_most_recent() {
        fbw_store_journal_reset();
        let outer = 1usize as pyre_object::PyObjectRef;
        let inner = 2usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(outer, InflightForiterBody::Py(6), false);
        fbw_foriter_inflight_capture(inner, InflightForiterBody::Py(35), false);

        // Parked at the OUTER header (pc 5), whose fallthrough body pc is 6.
        assert_eq!(fbw_foriter_inflight_take(0, 0, 5), Some((outer, 6)));

        fbw_store_journal_reset();
    }

    /// With no entry the parked frame can accept, the single-slot most-recent
    /// behaviour stands: the deliver site refuses on its header check exactly
    /// as before, so this change never widens what gets pushed.
    #[test]
    fn take_falls_back_to_the_most_recent_entry_when_none_matches_the_frame() {
        fbw_store_journal_reset();
        let older = 1usize as pyre_object::PyObjectRef;
        let newer = 2usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(older, InflightForiterBody::Py(6), false);
        fbw_foriter_inflight_capture(newer, InflightForiterBody::Py(35), false);

        assert_eq!(fbw_foriter_inflight_take(0, 0, 99), Some((newer, 35)));

        fbw_store_journal_reset();
    }

    /// The rule the take selects by. Driven directly because the ambiguous case
    /// cannot be built through `fbw_foriter_inflight_capture`: it replaces an
    /// entry carrying the same `body`, so two candidates need two distinct
    /// JitCode identities over one `CodeObject` — reachable in a recursive
    /// inline, not constructible here.
    #[test]
    fn an_ambiguous_frame_match_keeps_the_legacy_top_entry() {
        assert_eq!(
            inflight_take_index(3, [1usize].into_iter()),
            Some(1),
            "one candidate is the frame's entry"
        );
        // Both candidates sit BELOW the top on purpose: picking the last match
        // (index 1) is what a select-by-recency rule does, so only the
        // ambiguity refusal answers 2 here.
        assert_eq!(
            inflight_take_index(3, [0usize, 1].into_iter()),
            Some(2),
            "two candidates cannot be told apart, so the legacy top stands"
        );
        assert_eq!(
            inflight_take_index(2, std::iter::empty()),
            Some(1),
            "no candidate keeps the legacy top"
        );
        assert_eq!(
            inflight_take_index(0, std::iter::empty()),
            None,
            "an empty stash has nothing to take"
        );
    }

    /// The second refusal reason, beside an executed body effect: a rollback
    /// that put a namespace binding back.  Delivery repositions the frame to
    /// the body pc, so every statement between the walk entry and that pc is
    /// skipped and never rewrites the binding the rollback just restored —
    /// only a replay from the walk entry does.  The loop variable's own store
    /// is exempt at push time, so it never raises this flag.
    #[test]
    fn a_rolled_back_namespace_binding_refuses_consumed_item_delivery() {
        fbw_store_journal_reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);
        fbw_namespace_store_mark_rolled_back();

        assert!(fbw_namespace_store_rolled_back());
        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), None);

        fbw_store_journal_reset();
        assert!(
            !fbw_namespace_store_rolled_back(),
            "the reset clears the flag, so it cannot refuse the NEXT walk"
        );
    }

    #[test]
    fn executed_body_effect_still_refuses_consumed_item_delivery() {
        fbw_store_journal_reset();
        let item = 1usize as pyre_object::PyObjectRef;
        fbw_foriter_inflight_capture(item, InflightForiterBody::Py(34), false);
        fbw_mark_foriter_body_effect_since_consume();

        assert!(fbw_foriter_any_body_effect_signal());
        assert_eq!(fbw_foriter_inflight_take(0, 0, 33), None);

        fbw_store_journal_reset();
    }

    #[test]
    fn effect_journal_rollback_since_keeps_an_earlier_append() {
        use pyre_object::listobject::{w_list_can_append_without_realloc, w_list_len};
        use pyre_object::{w_int_new, w_list_append};

        fbw_store_journal_reset();
        let list =
            pyre_object::listobject::w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe { w_list_append(list, w_int_new(4)) };
        let len0 = unsafe { w_list_len(list) };
        let alloc0 = unsafe { pyre_object::listobject::w_list_allocated(list) };
        assert!(unsafe { w_list_can_append_without_realloc(list) });

        fbw_list_journal_push_append(list, len0, alloc0);
        unsafe { w_list_append(list, w_int_new(5)) };
        let mark = fbw_effect_journal_mark();
        let len1 = unsafe { w_list_len(list) };
        let alloc1 = unsafe { pyre_object::listobject::w_list_allocated(list) };
        fbw_list_journal_push_append(list, len1, alloc1);
        unsafe { w_list_append(list, w_int_new(6)) };
        assert_eq!(unsafe { w_list_len(list) }, len0 + 2);
        assert_eq!(fbw_executed_effect_count(), 2);

        fbw_effect_journal_rollback_since(mark);
        assert_eq!(unsafe { w_list_len(list) }, len0 + 1);
        assert_eq!(fbw_executed_effect_count(), 1);
        assert_eq!(
            fbw_journaled_effect_lens().1,
            1,
            "the append journaled before the mark stays"
        );

        fbw_store_journal_rollback();
        assert_eq!(unsafe { w_list_len(list) }, len0);
    }
}

/// Lengths of the journals [`fbw_binop_rewind_refuse_commit`] leaves in
/// place, sampled before a descent that may later decline to a residual.
#[derive(Clone, Copy)]
pub(crate) struct FbwEffectJournalMark {
    stores: usize,
    list_effects: usize,
    promotes: usize,
    cells: usize,
}

pub(crate) fn fbw_effect_journal_mark() -> FbwEffectJournalMark {
    FbwEffectJournalMark {
        stores: fbw_store_journal_len(),
        list_effects: FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len()),
        promotes: fbw_append_promote_journal_len(),
        cells: FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len()),
    }
}

fn fbw_unbump_executed_effects(n: usize) {
    if n == 0 {
        return;
    }
    FBW_EXECUTED_EFFECT_COUNT.with(|c| {
        let count = c.get().saturating_sub(n);
        c.set(count);
        if fbw_debug_abort_enabled() {
            eprintln!("[fbw-effect-unbump] n={n} count={count}");
        }
    });
}

fn undo_store_journal_entry(
    list: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
    displaced: pyre_object::PyObjectRef,
) {
    let restored = unsafe {
        let index = pyre_object::w_int_get_value(key);
        pyre_object::w_list_setitem(list, index, displaced)
    };
    if !restored {
        // Only reachable when another eagerly executed residual
        // shrank the list after the store — a shape the replay
        // already cannot undo (the residual re-runs).  Surface it
        // under the debug gate instead of corrupting silently.
        crate::trace::fbw_diag::bump(crate::trace::fbw_diag::STORE_JOURNAL_ROLLBACK_FAILED);
        if fbw_debug_abort_enabled() {
            eprintln!("[fbw-store-journal] rollback failed (index out of bounds)");
        }
    }
}

fn undo_list_effect_entry(entry: FbwListEffect) {
    unsafe {
        let (list, length_before, allocated_before) = match entry {
            FbwListEffect::Append {
                list,
                length_before,
                allocated_before,
            } => (list, length_before, allocated_before),
            FbwListEffect::PopEnd {
                list,
                length_before,
                w_item,
            } => {
                let list_ref = &mut *(list as *mut pyre_object::listobject::W_ListObject);
                match list_ref.strategy {
                    pyre_object::listobject::ListStrategy::Integer => {
                        pyre_object::listobject::ll_list_int_set_len(list_ref, length_before);
                        pyre_object::listobject::ll_list_int_setitem_fast(
                            list_ref,
                            length_before - 1,
                            pyre_object::w_int_get_value(w_item),
                        );
                    }
                    pyre_object::listobject::ListStrategy::Object => {
                        // A later ordinary append of a non-int can switch
                        // the popped Integer list to Object storage, seeded
                        // only from the post-pop prefix. Restore into that
                        // live block rather than the discarded int block.
                        pyre_object::listobject::ll_list_obj_set_len(list_ref, length_before);
                        pyre_object::listobject::ll_list_obj_setitem_fast(
                            list_ref,
                            length_before - 1,
                            w_item,
                        );
                    }
                    pyre_object::listobject::ListStrategy::IntOrFloat => {
                        pyre_object::listobject::w_list_int_or_float_set_len(list, length_before);
                        if !pyre_object::listobject::w_list_int_or_float_setitem(
                            list,
                            length_before - 1,
                            w_item,
                        ) {
                            crate::trace::fbw_diag::bump(
                                crate::trace::fbw_diag::STORE_JOURNAL_ROLLBACK_FAILED,
                            );
                        }
                    }
                    pyre_object::listobject::ListStrategy::Float
                    | pyre_object::listobject::ListStrategy::Empty
                    | pyre_object::listobject::ListStrategy::Size
                    | pyre_object::listobject::ListStrategy::SimpleRange
                    | pyre_object::listobject::ListStrategy::Range
                    | pyre_object::listobject::ListStrategy::Bytes
                    | pyre_object::listobject::ListStrategy::Ascii => {
                        crate::trace::fbw_diag::bump(
                            crate::trace::fbw_diag::STORE_JOURNAL_ROLLBACK_FAILED,
                        );
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-list-effect-journal] PopEnd rollback failed (invalid strategy)"
                            );
                        }
                    }
                }
                return;
            }
        };
        let list_ref = &mut *(list as *mut pyre_object::listobject::W_ListObject);
        match list_ref.strategy {
            pyre_object::listobject::ListStrategy::Object => {
                // The appended element is a GC ptr and the items block is
                // scanned over [0..capacity], so null the vacated slot
                // before shrinking (ll_pop_default: ll_setitem_fast(index,
                // ll_null_item) then _ll_resize_le) — otherwise the slot at
                // `length_before` holds a stale ref past the logical length.
                pyre_object::listobject::ll_list_obj_setitem_fast(
                    list_ref,
                    length_before,
                    pyre_object::pyobject::PY_NULL,
                );
                pyre_object::listobject::ll_list_obj_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::Integer => {
                pyre_object::listobject::ll_list_int_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::IntOrFloat => {
                pyre_object::listobject::w_list_int_or_float_set_len(list, length_before);
            }
            // Float items are non-ptr f64 scalars (no stale GC ref to
            // clear, unlike the Object slot), so rewinding the length
            // field suffices.
            pyre_object::listobject::ListStrategy::Float => {
                pyre_object::listobject::ll_list_float_set_len(list_ref, length_before);
            }
            pyre_object::listobject::ListStrategy::Ascii => {
                // Erased `STR` slots are GC pointers in an ItemsBlock scanned
                // over capacity. Null the vacated slot, then shrink the live
                // length (`ll_pop_default`: `ll_setitem_fast` of the null item,
                // then the length store).
                pyre_object::listobject::ll_list_ascii_setitem_fast(
                    list_ref,
                    length_before,
                    std::ptr::null(),
                );
                pyre_object::listobject::ll_list_ascii_set_len(list_ref, length_before);
            }
            // Empty never enters the append journal (no spare-capacity
            // fold path records it); nothing to rewind.
            pyre_object::listobject::ListStrategy::Empty => {}
            pyre_object::listobject::ListStrategy::Size => {}
            // BaseRangeListStrategy append materialises before the
            // append, so compact range storage is never journalled.
            pyre_object::listobject::ListStrategy::SimpleRange => {}
            pyre_object::listobject::ListStrategy::Range => {}
            // Bytes append does not enter this journal until the
            // walker has a BytesBlock store emitter.
            pyre_object::listobject::ListStrategy::Bytes => {}
        }
        pyre_object::listobject::w_list_set_allocated(list, allocated_before);
    }
}

fn undo_cell_store_entry(entry: FbwCellStore) {
    unsafe {
        match entry {
            FbwCellStore::Int { cell, before } => {
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-cell-journal] rollback cell=0x{:x} {} -> {before}",
                        cell as usize,
                        (*(cell as *const pyre_object::celldict::IntMutableCell)).intvalue
                    );
                }
                (*(cell as *mut pyre_object::celldict::IntMutableCell)).intvalue = before;
            }
            FbwCellStore::Obj { cell, before } => {
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-cell-journal] rollback-obj cell=0x{:x} -> 0x{:x}",
                        cell as usize, before as usize
                    );
                }
                // A minor collection can run between the speculative
                // store's barrier and rollback, so restoring a young
                // `before` needs its own barrier.
                pyre_object::celldict::object_mutable_cell_write_barrier(
                    cell as pyre_object::gc_hook::GCREF,
                );
                (*(cell as *mut pyre_object::celldict::ObjectMutableCell)).w_value = before;
            }
            FbwCellStore::Gc {
                obj,
                offset,
                size,
                before,
                managed,
                ..
            } => {
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-gc-store-journal] rollback obj=0x{:x} offset={offset} -> {before:?}",
                        obj as usize
                    );
                }
                let _ = fbw_gc_store_word(obj, offset, size, before, managed);
            }
        }
    }
}

/// Undo journaled effects pushed after `mark`.
///
/// `fbw_binop_rewind_refuse_commit` does not refuse `store_journal`,
/// `list_append`, `list_pop_end`, or `cell_store_journal`. A later refusal
/// still returns from the `BINARY_OP` / `COMPARE_OP` descent, and that
/// descent's caller falls through to `binary_value_from_tag` (or the
/// compare residual). The outer walk can then `fbw_store_journal_commit`,
/// which drops the undo log and keeps the eager write, so the residual
/// applies it a second time. Rewinding only the suffix leaves an effect
/// the enclosing frame journaled before this descent.
pub(crate) fn fbw_effect_journal_rollback_since(mark: FbwEffectJournalMark) {
    let mut bumped = 0usize;
    FBW_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while entries.len() > mark.stores {
            let Some([list, key, displaced]) = entries.pop() else {
                break;
            };
            bumped += 1;
            undo_store_journal_entry(list, key, displaced);
        }
    });
    FBW_LIST_EFFECT_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while entries.len() > mark.list_effects {
            let Some(entry) = entries.pop() else {
                break;
            };
            bumped += 1;
            undo_list_effect_entry(entry);
        }
    });
    // The length rewind above already shrank a promoted list back to 0.
    // `w_list_clear` restores Empty strategy for the promotions this
    // descent pushed. An earlier promotion stays on the log.
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while entries.len() > mark.promotes {
            if let Some(list) = entries.pop() {
                unsafe { pyre_object::listobject::w_list_clear(list) };
            }
        }
    });
    FBW_CELL_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while entries.len() > mark.cells {
            let Some(entry) = entries.pop() else {
                break;
            };
            bumped += 1;
            undo_cell_store_entry(entry);
        }
    });
    // Each popped store, list effect, and cell push bumped the odometer
    // once. A promotion push does not. The residual that re-executes the
    // dunder bumps for itself.
    fbw_unbump_executed_effects(bumped);
}

/// Non-commit epilogue: restore each displaced element in reverse push
/// order so the legacy replay re-executes against the pre-walk heap.
/// `w_list_setitem` allocates nothing on the restore (the displaced value
/// is already boxed and strategy-matching), so entries cannot move
/// mid-rollback.
///
/// Stores are restored BEFORE appends are rewound: a store's key was
/// in-bounds at store time and stays in-bounds at the walk's final
/// (max) length, so every restore lands while the list is still grown;
/// shrinking first could push a restore index past the length and drop it.
pub(crate) fn fbw_store_journal_rollback() {
    fbw_namespace_store_journal_rollback();
    FBW_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some([list, key, displaced]) = entries.pop() {
            undo_store_journal_entry(list, key, displaced);
        }
    });
    // Undo eager list effects in reverse push order. Append entries record
    // only spare-capacity growth, so no allocation needs undoing; a later
    // strategy switch preserves their logical length. Dispatch every entry
    // against the strategy that is live at rollback time.
    FBW_LIST_EFFECT_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(entry) = entries.pop() {
            undo_list_effect_entry(entry);
        }
    });
    // The length rewind above already shrank the list back to length 0.
    // `w_list_clear` additionally restores the Empty strategy and drops the
    // typed backing block, completing the undo of the Empty-to-typed switch
    // that the length journal alone cannot undo.
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(list) = entries.pop() {
            unsafe {
                pyre_object::listobject::w_list_clear(list);
            }
        }
    });
    // Restore each eagerly stored `IntMutableCell`'s prior `intvalue` in
    // reverse push order (raw i64 write; allocation-free, cells immovable).
    FBW_CELL_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(entry) = entries.pop() {
            undo_cell_store_entry(entry);
        }
    });
    // Restore `sys_exc_value` to its pre-walk value.  Replaying in reverse
    // push order makes the LAST write the value read at walk entry (the
    // first eager store's displaced prior), so an aborted walk leaves the
    // live per-thread EC exactly as the legacy replay-from-start expects —
    // in particular an exception that propagated OUT of an except-handler
    // (walk aborted before its POP_EXCEPT restore) no longer leaks the
    // caught exception into the next frame's `__context__`.
    FBW_SYS_EXC_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(displaced) = entries.pop() {
            pyre_interpreter::eval::set_current_exception(displaced);
        }
    });
    // Splice every journaled bridge-entry node back out of its exception's
    // chain.  The node is not necessarily still the head: concrete execution
    // after the attach can prepend further nodes to the same exception (a
    // handler that re-raises, or another raising callee), and dropping the undo
    // in that case would leave the speculative node behind for the replay to
    // record the catching frame on top of — a duplicated frame in the reported
    // traceback.  Newer heads are preserved; only the journaled link is
    // removed, in reverse push order so the chain collapses to its pre-walk
    // shape.
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some((exception, node)) = entries.pop() {
            let next = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_w_next(node) };
            let head =
                unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) };
            if head == node {
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_traceback(exception, next);
                }
                continue;
            }
            let mut prev = head;
            while !prev.is_null() && unsafe { pyre_interpreter::pytraceback::is_pytraceback(prev) }
            {
                let curr = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_w_next(prev) };
                if curr == node {
                    // The removed node sat between `prev` and `next`, so the
                    // shortened chain cannot reach `prev` again and the
                    // setter's loop check always passes.
                    let _ = unsafe {
                        pyre_interpreter::pytraceback::w_pytraceback_set_w_next(prev, next)
                    };
                    break;
                }
                prev = curr;
            }
        }
    });
}

/// Current journal length (commit-point diagnostics).
pub(crate) fn fbw_store_journal_len() -> usize {
    FBW_STORE_JOURNAL.with(|j| j.borrow().len())
}

/// The three journals whose pushes bump [`fbw_bump_executed_effect`], as
/// `(stores, appends, cell_stores)`.  Read at the walk-end epilogue before the
/// commit/rollback, their sum is the number of executed effects
/// [`fbw_store_journal_rollback`] is about to undo.  The traceback journal is
/// absent because its push bumps nothing.
pub(crate) fn fbw_journaled_effect_lens() -> (usize, usize, usize) {
    (
        fbw_store_journal_len(),
        FBW_LIST_EFFECT_JOURNAL.with(|j| j.borrow().len()),
        FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len()),
    )
}

/// Mark the walk as carrying a recorded-but-unexecuted side effect only
/// the legacy replay applies.
///
/// The mark decides which walk-end roads stay open, and a walk that also
/// executed an unrecoverable effect has none left, so the useful question at
/// a stuck walk is which residual took this branch.  Six call sites forward a
/// cause here and the flag itself keeps no provenance; `PYRE_UNJOURNALED_SITE`
/// names the caller, in the shape `PYRE_LB_SITE` uses for the callee-inline
/// declines.
fn unjournaled_site_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_UNJOURNALED_SITE").is_some())
}

#[track_caller]
pub(crate) fn fbw_mark_unjournaled_effect(cause: ResidualDecline) {
    if unjournaled_site_enabled() {
        let loc = std::panic::Location::caller();
        // Only a `Symbolic` decline records a site, so read the cell only for
        // that cause: on any other one it still holds whichever earlier
        // symbolic decline happened to run last, and reporting that would name
        // an unrelated residual as this mark's origin.
        let declined_at = match cause {
            ResidualDecline::Symbolic => super::residual_call::last_declined_symbolic_site()
                .map_or_else(
                    || "?".to_owned(),
                    |(l, op)| format!("{}:{} opcode={op:?}", l.file(), l.line()),
                ),
            _ => "n/a".to_owned(),
        };
        eprintln!(
            "[unjournaled-site] {}:{} cause={cause:?} executed={} declined_at={declined_at}",
            loc.file(),
            loc.line(),
            fbw_executed_effect_count(),
        );
    }
    match cause {
        ResidualDecline::ValueUnavailable => {
            FBW_UNJOURNALED_VALUE_UNAVAILABLE.with(|c| c.set(true));
        }
        ResidualDecline::Symbolic => FBW_UNJOURNALED_SYMBOLIC.with(|c| c.set(true)),
        // Nothing to mark: an elidable call applies no effect, so there is no
        // pending write for a replay to be the only carrier of.  Setting the
        // flag here would be read as "this walk still owes an effect" and shut
        // every no-replay walk-end road, leaving a walk that had already
        // executed unrecoverable effects with replay as its only exit — which
        // re-applies them.
        ResidualDecline::PureUnfolded => {}
    }
}

/// gh#467 executed-effect odometer read (see [`FBW_EXECUTED_EFFECT_COUNT`]).
pub(crate) fn fbw_executed_effect_count() -> usize {
    FBW_EXECUTED_EFFECT_COUNT.with(|c| c.get())
}

pub(crate) fn fbw_structural_abort_opcode_is_effect_free(pc: usize) -> bool {
    FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS.with(|c| c.get() == Some((pc, 0)))
}

/// Latch the resume coordinate + operand-stack mirror for an
/// `ABORT_FORCE_QUASIIMMUT` abort (see [`FBW_QMUT_ABORT_STACK`]).
pub(crate) fn fbw_qmut_abort_stack_latch(py_pc: usize, stack: Vec<OpRef>) {
    FBW_QMUT_ABORT_STACK.with(|c| *c.borrow_mut() = Some((py_pc, stack)));
}

/// Take the latch above.  `None` when the abort could not offer a mirror (a
/// sub-walk, a bridge walk, or an invalidated mirror), which leaves the flush
/// leg to decline and the legacy replay to stand.
pub(crate) fn fbw_qmut_abort_stack_take() -> Option<(usize, Vec<OpRef>)> {
    FBW_QMUT_ABORT_STACK.with(|c| c.borrow_mut().take())
}

/// Preserve the complete MIFrame-equivalent operand stack for a kept-stack
/// branch abort.  The caller has already proved the mirror valid and trims it
/// to the depth on entry to `py_pc`.
pub(crate) fn fbw_branch_abort_stack_latch(py_pc: usize, stack: Vec<OpRef>) {
    FBW_BRANCH_ABORT_STACK.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.is_none() {
            *slot = Some((py_pc, stack));
        }
    });
}

/// Consume the kept-stack branch-abort carrier above.
pub(crate) fn fbw_branch_abort_stack_take() -> Option<(usize, Vec<OpRef>)> {
    FBW_BRANCH_ABORT_STACK.with(|c| c.borrow_mut().take())
}

/// Record the executed-effect odometer as the walk enters Python opcode
/// `py_pc` (see [`FBW_OPCODE_ENTRY_EFFECTS`]).
pub(crate) fn fbw_note_opcode_entry_effects(py_pc: usize) {
    FBW_OPCODE_ENTRY_EFFECTS.with(|c| c.set(Some((py_pc, fbw_executed_effect_count()))));
}

/// The odometer sampled on entry to `py_pc`, for a leg that resumes the
/// interpreter there.  `None` — no sample, or one taken at a different opcode —
/// leaves the rewind unproven, which [`walk_end_resume_provable`] declines.
///
/// [`walk_end_resume_provable`]: crate::trace::walk_end_resume_provable
pub(crate) fn fbw_opcode_entry_effects_at(py_pc: usize) -> Option<usize> {
    FBW_OPCODE_ENTRY_EFFECTS.with(|c| match c.get() {
        Some((sampled_py_pc, effects)) if sampled_py_pc == py_pc => Some(effects),
        _ => None,
    })
}

/// gh#467 bump the executed-effect odometer (see [`FBW_EXECUTED_EFFECT_COUNT`]).
///
/// `site` names the bumping journal or fold.  The forward-flush gate reads only
/// the count, so a refusal reports `effects>0` without saying which of the six
/// bumps moved it; the tag is what turns that into an attributable reading.
pub(crate) fn fbw_bump_executed_effect(site: &'static str) {
    FBW_EXECUTED_EFFECT_COUNT.with(|c| {
        let count = c.get() + 1;
        c.set(count);
        if fbw_debug_abort_enabled() {
            eprintln!("[fbw-effect-bump] site={site} count={count}");
        }
    });
}

pub(crate) fn fbw_set_midbody_abort_resume(payload: MidBodyPayload) {
    FBW_ABORT_CALL_RESUME.with(|c| *c.borrow_mut() = Some(InlineAbortCarrier::MidBody(payload)));
}

pub(crate) fn fbw_abort_carrier_clone() -> Option<InlineAbortCarrier> {
    FBW_ABORT_CALL_RESUME.with(|c| c.borrow().clone())
}

pub(crate) fn fbw_abort_carrier_set_return(value: pyre_object::PyObjectRef) {
    FBW_ABORT_CALL_RESUME.with(|c| {
        if let Some(InlineAbortCarrier::MidBody(payload)) = c.borrow_mut().as_mut() {
            payload.return_value = value;
        }
    });
}

pub(crate) fn fbw_abort_carrier_clear() {
    FBW_ABORT_CALL_RESUME.with(|c| *c.borrow_mut() = None);
}

/// A declined residual call (`try_execute_residual_call_via_executor`
/// returned `None`) reached during a multiframe-inlined callee sub-walk
/// (the framestack is non-empty) cannot fall back to the walk-end
/// legacy replay.  The replay re-enters the freshly compiled loop from the
/// recorded entry state while sibling concretely-executed heap mutations of
/// the SAME iteration (the enclosing loop's `i = i + 1`) have already
/// applied, so the first compiled iteration is half-applied — one
/// iteration's contribution silently dropped (#68 depth-2 multiframe:
/// `s = s + outer(i)` lands short by exactly `outer(N+1)` because the
/// nested callee's residual never ran). Decline the enclosing trace to
/// interpretation instead. At top level (no active inline)
/// the unjournaled-effect / legacy-replay path is sound, so only abort when
/// nested.
thread_local! {
    /// Marks the self-recursive `CALL_ASSEMBLER` fold's concrete-stamp
    /// executor call. RPython `do_residual_call` executes the recorded
    /// residual at any framestack depth (`pyjitpl.py`), while
    /// pyre's nested-residual decline below is a local protection for
    /// FOREIGN unjournaled residuals.
    pub(crate) static SELFREC_CA_FOLD_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The bounded `str(exc)` / `repr(exc)` descriptor inline may retain an
    /// interior residual such as `repr(self.args)`. The caller's original
    /// iteration already supplied the concrete result, while the compiled
    /// trace executes that residual once on later iterations, so the generic
    /// nested-replay decline does not apply to this resolved descriptor path.
    pub(crate) static EXCEPTION_STRING_INLINE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Code keys of the callees [`fbw_decline_inline_callee`] named.  The
    /// inline callsite declines them from then on, so the call residualizes
    /// and the enclosing trace never re-enters the identical abort.
    ///
    /// This is `disable_noninlinable_function` (`warmstate.py`, which sets
    /// `JC_DONT_TRACE_HERE` = "do not inline calls to this function"): upstream
    /// answers an abort attributable to one inlined callee by denying THAT
    /// callee and retracing the enclosing loop, not by penalising the loop
    /// (`pyjitpl.py:2818-2828`).  Like the upstream flag the set has no removal
    /// path — the hazard is a static property of the callee's `CodeObject`.
    ///
    /// Keying on the `CodeObject` alone is the full key for this decision, not
    /// a truncation of one.  The flag is consumed by `can_inline_callable`
    /// (`warmstate.py`), whose only caller is `_opimpl_recursive_call`
    /// (`pyjitpl.py`) — it passes the CALLEE's green args, and a
    /// callee reached through a CALL is always entered at its own entry, so the
    /// `next_instr` component is constant and `pycode` carries the whole
    /// decision.  The same holds here: the deny is recorded and queried for an
    /// inline frame pushed at a CALL boundary (`inline_call.rs`), never for a
    /// mid-body resume.
    ///
    /// Per-thread for the same reason as [`crate::trace::fbw_declined`]'s
    /// `FBW_DECLINED_KEYS` and `RANGE_FORITER_DEMOTED`: pyre's walk state is
    /// per-thread, and an inline hazard observed while tracing is a property of
    /// the tracing thread's framestack.  Sharing one memo while its siblings
    /// stay per-thread would be the inconsistency.
    ///
    /// NOT yet ported: `warmstate.py` also treats the flag as "please
    /// trace from here as soon as possible" — a denied cell that never had a
    /// procedure token reaches `bound_reached` immediately, so the callee gets
    /// its own trace instead of staying a plain residual forever.  Since
    /// residual calls re-enter the JIT the callee does reach its own threshold,
    /// just on the ordinary counter rather than at once.
    ///
    /// The stored key is `PyCode.code_ptr`, not the movable wrapper address.
    /// Top-level `CodeObject` bodies are permanent `Box::into_raw` owners and
    /// nested bodies borrow from that permanent graph (`pycode.rs`), so this
    /// process-lifetime memo cannot accidentally deny a later wrapper which
    /// reused a collected PyCode address.  The warm-state cell still receives
    /// the wrapper green below; this raw owner key is only for this memo.
    static FBW_HAZARDOUS_INLINE_DENY: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

pub(crate) fn fbw_hazardous_inline_denied(callee_code_key: usize) -> bool {
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(callee_code_key as pyre_object::PyObjectRef) as usize
    };
    raw_code != 0 && FBW_HAZARDOUS_INLINE_DENY.with(|c| c.borrow().contains(&raw_code))
}

fn fbw_deny_hazardous_inline(callee_code_key: usize) {
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(callee_code_key as pyre_object::PyObjectRef) as usize
    };
    if raw_code != 0 {
        FBW_HAZARDOUS_INLINE_DENY.with(|c| {
            c.borrow_mut().insert(raw_code);
        });
    }
}

pub(crate) fn fbw_abort_nested_unjournaled_residual<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
    cause: Option<ResidualDecline>,
) -> Result<(), DispatchError> {
    // `cause` is `None` when the residual is about to EXECUTE and `Some` when
    // it declined and was left recorded.  A `PureUnfolded` decline records an
    // elidable call the fold could not answer: it applies no effect, so it
    // leaves the sub-walk's "commits nothing" promise intact and there is
    // nothing for a resume to double — the same reason
    // `fbw_mark_unjournaled_effect` sets no flag for it.  The other two causes
    // leave a recorded-but-unexecuted write only the legacy replay applies,
    // which is what this decline exists to refuse.
    if matches!(cause, Some(ResidualDecline::PureUnfolded)) {
        return Ok(());
    }
    // Under the current portal-runner ABI the inline frame is a real red
    // frame in the captured chain and aborts resume forward, so the residual
    // is executed and recorded exactly as `do_residual_call` does.
    // The one remaining miscompile, a self-recursive callee whose hot
    // `CALL_ASSEMBLER` recursion-bridge frame the residual trampoline cannot
    // retain, is the `CALL_ASSEMBLER` fold itself and is exempted by
    // `SELFREC_CA_FOLD_ACTIVE` where it is folded.
    //
    // What is left is the rewind region, which has no forward resume of its
    // own: the abort of a `BINARY_OP` / `COMPARE_OP` dunder answering
    // `NotImplemented` rewinds to the operator, so that region refuses every
    // unjournaled commit, this one included.
    fbw_binop_rewind_refuse_commit(ctx, pc, None)
}

/// Refuse the innermost inline callee: deny it for the rest of this thread's
/// tracing and build the decline the walk returns.
///
/// Shared by the declined `locals()` expansion in `specialize.rs` and the
/// inline-call refuse sites in `inline_call.rs`, which answer the same
/// question — this callee must not be inlined HERE — from an op the walk
/// actually reached and from a frame-payload gate.
///
/// Denying is what stops the decline being a property of the framestack, which
/// the next attempt rebuilds identically: without it the abort recurs
/// byte-for-byte until the enclosing location is retired, so the loop never
/// compiles at all.
///
/// The trigger has no upstream counterpart, and the citation this used to
/// carry named a different question.  `disable_noninlinable_function`
/// (`warmstate.py` `WarmEnterState.disable_noninlinable_function`) has exactly
/// one caller, `pyjitpl.py` `MetaInterp.blackhole_if_trace_too_long`, and that
/// one asks a SIZE question — `history.length() > trace_limit`, or
/// `trace_tag_overflow()` — and then denies the greenkey
/// `MetaInterp.find_biggest_function` names, which need not be the callee at
/// hand.  A vable escape denies nothing: `MetaInterp.vable_after_residual_call`
/// raises `SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)` and
/// `MetaInterp.run_blackhole_interp_to_cancel_tracing` hands that to
/// `blackhole.py` `convert_and_run_from_pyjitpl`, which carries the frame
/// FORWARD.  Reaching the upstream API from this trigger is therefore an
/// extension of it, not a port of it.
///
/// For the caller that refuses at a declined `locals()` expansion, the reason
/// upstream never arrives here is that the escape does not happen:
/// `pyframe.py` `PyFrame.fast2locals` is `@jit.unroll_safe`, so the tracer
/// descends into it rather than leaving a residual, and inside a graph it
/// looks into, `jtransform.py` `Transformer.rewrite_op_jit_force_virtualizable`
/// erases the force outright.  What stands in for that here is
/// `try_walker_specialize_builtin_locals_in_callee_expand`, and this refusal
/// covers exactly the shapes that expansion declines.  Widening it until the
/// refusal is unreachable is the convergence path; the refusal is not the
/// answer.
///
/// `callee_code_key` is `None` when no callee could be named, in which case the
/// decline still unwinds but nothing is remembered.
pub(crate) fn fbw_decline_inline_callee<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
    callee_code_key: Option<usize>,
) -> DispatchError {
    if let Some(callee_code_key) = callee_code_key {
        fbw_deny_hazardous_inline(callee_code_key);
        // The walk's own profiling green, not the ambient one: `make_green_key`
        // builds the `u64` key at a site that holds no frame, and a walk-internal
        // caller names it from `WalkSession`, seeded once per walk from the
        // concrete frame.  Read before `try_driver_pair` so the session borrow
        // is released across the driver call.
        let is_being_profiled = ctx.session.borrow().is_being_profiled;
        // `fbw_deny_hazardous_inline` writes a thread-local set that only
        // this walker reads, so the deny stayed invisible to the warm
        // state: `dont_trace_here` counted zero on every fixture that
        // reaches here.  `warmstate.py`
        // `WarmEnterState.disable_noninlinable_function` is the call that sets
        // `JC_DONT_TRACE_HERE` on the callee's cell, and that bit is what
        // `can_inline_callable` reads, so the deny is spelled through it
        // rather than beside it.  The mechanism is upstream's; the trigger is
        // not, and the doc above says which.  The recursion-bound deny in
        // `inline_call.rs` already calls it for the callee it names; this one
        // names a callee the same way and had no reason not to.  Pass the
        // typed key so the verdict lands on the same comparekey-bearing cell
        // consulted at the next call.
        if let Some((driver, _)) = crate::driver::try_driver_pair() {
            let key = crate::driver::make_green_key_typed(
                callee_code_key as *const (),
                0,
                is_being_profiled,
            );
            driver
                .meta_interp_mut()
                .warm_state_for_driver(crate::state::PyreJitState::PYPYJIT_JD_INDEX)
                .disable_noninlinable_function_for_key(&key);
        }
    }
    // A seeded inline already has the callee `PyFrame`, and
    // `run_blackhole_interp_to_cancel_tracing` (`pyjitpl.py`) converts the
    // live framestack in place.  Unseeded stays residual-not-abort — the
    // enclosing CALL was never entered.
    //
    // `blackhole_required` still names whether the aborting MIFrame has
    // already applied an effect, so an unjournaled FOR_ITER consume converts
    // instead of dropping the item.  `convert_and_run_from_pyjitpl` copies
    // every `MIFrame` independently (`blackhole.py`).
    let blackhole_required = {
        let session = ctx.session.borrow();
        session
            .framestack
            .last()
            .is_some_and(|frame| fbw_executed_effect_count() != frame.entry_executed_effects)
            || (fbw_foriter_inflight_active()
                && (fbw_foriter_any_body_effect_signal() || fbw_foriter_unjournaled_consume()))
    };
    DispatchError::callee_inline_abort(pc, blackhole_required)
}

/// The code key of the callee whose body the active sub-walk is recording, for
/// the refusal to name.  This is the innermost live `MIFrame` — the same frame
/// [`fbw_decline_inline_callee`]'s `blackhole_required` is computed from.
pub(crate) fn fbw_innermost_inline_callee_key<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<usize> {
    ctx.session.borrow().last_inline().map(|f| f.w_code)
}

/// Whether the walk recorded an effect outside the journal's reach.
pub(crate) fn fbw_has_unjournaled_effect() -> bool {
    let (value_unavailable, symbolic) = fbw_unjournaled_kinds();
    value_unavailable || symbolic
}

pub(crate) fn fbw_unjournaled_kinds() -> (bool, bool) {
    (
        FBW_UNJOURNALED_VALUE_UNAVAILABLE.with(|c| c.get()),
        FBW_UNJOURNALED_SYMBOLIC.with(|c| c.get()),
    )
}

pub(crate) fn fbw_count_executed_residual(is_void: bool, is_may_force: bool) {
    let counter = if is_void {
        &FBW_EXECUTED_RESIDUAL_VOID
    } else if is_may_force {
        &FBW_EXECUTED_RESIDUAL_MAYFORCE
    } else {
        &FBW_EXECUTED_RESIDUAL_PLAIN
    };
    counter.with(|c| c.set(c.get().wrapping_add(1)));
}

pub(crate) fn fbw_executed_residual_counts() -> (u32, u32, u32) {
    (
        FBW_EXECUTED_RESIDUAL_VOID.with(|c| c.get()),
        FBW_EXECUTED_RESIDUAL_MAYFORCE.with(|c| c.get()),
        FBW_EXECUTED_RESIDUAL_PLAIN.with(|c| c.get()),
    )
}

/// `framework.py root_walker.walk_roots` parity for the store and append
/// journals: the entries hold nursery-resident refs across the rest of the
/// walk (residual calls allocate, and a minor collection moves nursery
/// objects), so every ref slot is forwarded as a root.  Registered once via
/// `majit_gc::shadow_stack::register_extra_root_walker` at JIT init.
pub fn fbw_store_journal_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let data = capture_fbw_store_journal_root_area();
    unsafe { fbw_store_journal_root_walker_area(data, visitor) };
}

pub fn capture_fbw_store_journal_root_area() -> *const () {
    FBW_STORE_JOURNAL_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// FBW-native port of [`crate::state::ensure_boxed_for_ca`] that operates
/// purely on the [`TraceCtx`] (no borrowed `MIFrame`).  A portal-exit
/// FINISH must carry `Type::Ref` (`pyjitpl.py` REF result_type);
/// if the optimizer left the return value unboxed as Int/Float, re-box it
/// (`wrapint` / `wrapfloat` = `NewWithVtable` + `SetfieldGc`).  `value_type`
/// here is `ctx.get_opref_type(value).unwrap_or(Type::Ref)`, the exact body
/// of `MIFrame::value_type` minus the borrow.
pub(crate) fn fbw_ensure_boxed_for_ca<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    value: OpRef,
) -> Result<OpRef, DispatchError> {
    let ty = if value.is_none() {
        Type::Ref
    } else {
        ctx.trace_ctx.get_opref_type(value).unwrap_or(Type::Ref)
    };
    let _ = op_pc;
    let boxed = match ty {
        Type::Int => crate::state::wrapint(ctx.trace_ctx, value),
        Type::Float => crate::state::wrapfloat(ctx.trace_ctx, value),
        Type::Ref | Type::Void => value,
    };
    Ok(boxed)
}

/// FBW-native port of `MIFrame::store_token_in_vable` (`pyjitpl.py`).
/// Records `FORCE_TOKEN` + `SETFIELD_GC(vbox, token, vable_token_descr)`
/// via `store_token_in_vable_setfield` and, when that fires, the
/// `GUARD_NOT_FORCED_2` with resumedata captured through the walker's own
/// single-frame snapshot machinery (`walker_capture_snapshot_for_last_guard`)
/// — the same resume coordinate (`entry_py_pc` / `outer_active_boxes`) every
/// other FBW guard uses, since pyre's blackhole can only re-enter the outer
/// Python opcode boundary.  No-op when there is no standard virtualizable.
pub(crate) fn fbw_store_token_in_vable<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.store_token_in_vable_setfield() {
        ctx.trace_ctx.record_guard(OpCode::GuardNotForced2, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    Ok(())
}

/// `pyopcode.py handle_operation_error`'s `frame_finished_execution = True`
/// on the portal frame, recorded as `getfield flags; int_or
/// FLAG_FRAME_FINISHED; setfield flags` the way `RETURN_VALUE`'s lowered
/// store and `walker_ec_leave`'s exception sibling record it for an
/// inlined callee.  The trace carries the bit itself, so a compiled
/// `exit_frame_with_exception` run leaves the frame clearable even when no
/// interpreter re-delivery follows (a `CALL_ASSEMBLER` entry).  No-op
/// without a standard virtualizable, like [`fbw_publish_exit_last_instr`].
pub(crate) fn fbw_record_top_level_frame_finished<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
) {
    let Some(vbox) = ctx.trace_ctx.standard_virtualizable_box() else {
        return;
    };
    let flags_descr = crate::descr::pyframe_flags_descr();
    let live_flags = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, vbox, flags_descr.clone());
    let finished_bit = ctx
        .trace_ctx
        .const_int(i64::from(pyre_interpreter::PyFrame::FLAG_FRAME_FINISHED));
    let new_flags = ctx
        .trace_ctx
        .record_op(OpCode::IntOr, &[live_flags, finished_bit]);
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[vbox, new_flags], flags_descr.clone());
    ctx.trace_ctx
        .heapcache_setfield_cached(vbox, flags_descr.index(), new_flags);
}

/// Shared top-level finish path for the three value-returning arms
/// (`ref_return` / `int_return` / `float_return`).  Re-boxes `result` to
/// `Type::Ref`, publishes the return coordinate into `last_instr`, stores the
/// virtualizable token through `FORCE_TOKEN` + `GUARD_NOT_FORCED_2`, and stashes
/// the finish payload for `full_body_walk_trace`.  Deliberately
/// does NOT record the `FINISH` op: under the gate the compile consumer
/// (`finish_and_compile` -> `recorder.finish`, mod.rs) records it from
/// `finish_args`, so recording it here too would double it.
///
/// Returns the FINISH operand (`N_aryOp._args`) and stores the same pair
/// on the walk session so `walk_session_roots` can forward a `ConstPtr`.
pub(crate) fn fbw_terminate_with_finish<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    result: OpRef,
    op_pc: usize,
) -> Result<(OpRef, Type), DispatchError> {
    let finish_value = fbw_ensure_boxed_for_ca(ctx, op_pc, result)?;
    fbw_publish_exit_last_instr(ctx, op_pc);
    fbw_store_token_in_vable(ctx, op_pc)?;
    let payload = (finish_value, Type::Ref);
    ctx.session.borrow_mut().finish_payload = Some(payload);
    Ok(payload)
}

/// Void variant of [`fbw_terminate_with_finish`] for the top-level
/// `void_return/` portal exit (`compile_done_with_this_frame`'s VOID
/// branch, pyjitpl.py).  Publishes the return coordinate and stores the
/// virtualizable back the same way, then stashes a `Type::Void`-marked payload so
/// [`crate::trace::full_body_walk_trace`] builds a `TraceAction::Finish`
/// with no args (`done_with_this_frame_descr_from_types(&[])` resolves the
/// void descr).  Like the value path it does NOT record the `FINISH` op —
/// the compile consumer records it from the empty `finish_args`.
pub(crate) fn fbw_terminate_void_with_finish<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(OpRef, Type), DispatchError> {
    fbw_publish_exit_last_instr(ctx, op_pc);
    fbw_store_token_in_vable(ctx, op_pc)?;
    let payload = (OpRef::NONE, Type::Void);
    ctx.session.borrow_mut().finish_payload = Some(payload);
    Ok(payload)
}

/// Publish the exiting instruction's Python coordinate into the standard
/// virtualizable's `last_instr` slot before the top-level frame exits.
///
/// Compiled code never runs the per-opcode interpreter store, so a frame
/// entered through the function-entry portal still carries the `-1`
/// initialization sentinel and a loop-entry frame carries the loop header.
/// `offset2lineno` answers `-1` with the code object's first line — the `def`
/// line — so whatever reads the field afterwards reports a line the frame was
/// not executing.
///
/// Both frame exits publish, because both leave a reader behind.  On the
/// uncaught raise, `handle_exception` (pyre-interpreter) stamps the traceback
/// node from `frame.last_instr` and keys the exception-table lookup on the
/// same field.  On the normal return, a traceback the frame handed out
/// outlives it, and `tb_frame.f_lineno` resolves through `offset2lineno` on
/// this field — the return is the last coordinate the frame ever reached.
///
/// Upstream never faces either shape: the portal is entered only from a
/// backward jump (`can_enter_jit`, `interp_jit.py`), so a loop-free function
/// is never compiled as one, and the frame's `dispatch` loop runs inside the
/// traced portal where every opcode writes `last_instr` (`pyopcode.py`).
/// pyre's function-entry portal reaches the field from the interpreter after
/// the trace has finished, so the coordinate has to be published before it
/// does.
///
/// The store reaches the frame on a compiled run the way
/// `gen_store_back_in_vable`'s does; the shadow mirror keeps the walker's own
/// virtualizable view in step with it.
pub(crate) fn fbw_publish_exit_last_instr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) {
    let (recording_frame_ptr, jitcode_index) = {
        let session = ctx.session.borrow();
        (session.recording_frame_ptr, session.recording_jitcode_index)
    };
    // `dispatch_bytecode` stores the exact opcode before executing it.  A
    // JitCode position can sit in a later emission island owned by that same
    // opcode (notably an exception-edge bridge), where the containing/floor
    // coordinate still names the preceding instruction.  Use the same
    // exact-first resolution as `record_caught_blackhole_traceback`; otherwise
    // the traceback records the raising opcode while GUARD_NOT_FORCED_2 saves
    // the preceding one and a later force rolls `frame.last_instr` backwards.
    let Some(py_pc) =
        crate::py_coord::exact_py_pc_for_jitcode_pc_public(jitcode_index, opcode_position as i32)
            .or_else(|| {
                crate::py_coord::containing_py_pc_for_jitcode_pc_public(
                    jitcode_index,
                    opcode_position as i32,
                )
            })
    else {
        return;
    };
    let Some(vbox) = ctx.trace_ctx.standard_virtualizable_box() else {
        return;
    };
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return;
    };
    let Some(field_index) = info.static_field_index_by_name("last_instr") else {
        return;
    };
    let value = ctx.trace_ctx.const_int(i64::from(py_pc));
    // Record under the PARENT-STRUCT field descr, the resolution
    // `vable_setfield` applies through `vable_static_record_descr`, not the
    // vinfo's own `static_field_descrs[i]`.  `virtualizable.py:71` builds the
    // vinfo descrs with `cpu.fielddescrof(VTYPE, name)`, so upstream's vinfo
    // descr and the descr an ordinary `setfield_gc` on that field carries are
    // the same object; pyre keeps the two numberings apart, so a store left on
    // the vinfo descr reaches the optimizer as a location of its own.  A frame
    // that catches gets `last_instr` written twice at one offset — the raise
    // coordinate by the traceback node, the return coordinate here — and under
    // two descrs neither store supersedes the other, so which one survives to
    // the frame is decided by the order the deferred stores are flushed in
    // rather than the order they were recorded in.
    let descr = info.static_field_struct_descr(field_index);
    ctx.trace_ctx.vable_setfield_descr(vbox, value, descr);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "last_instr",
        value,
        Value::Int(i64::from(py_pc)),
    );
    // The recorded store has to have a concrete counterpart.  Upstream's
    // tracing IS the interpreter, so its per-opcode `last_instr` write
    // (`pyopcode.py`) lands in the real frame on the very iteration the trace
    // is recorded from; the walker only records ops, so without this the
    // recording iteration is the one iteration that still reports the stale
    // sentinel — a single wrong answer in the middle of a survey.
    // `recording_frame_ptr` is the LIVE frame, not `virtualizable_heap_ptr`'s
    // trace-stepping snapshot: the snapshot's storage is released when tracing
    // ends, so a store there reaches nothing the interpreter goes on to read.
    // Unlike the recorded store, this one lands whether or not the walk goes on
    // to commit, so it is journaled: a declined walk resumes the frame from its
    // pre-walk state and reads this very field to find the next instruction.
    if recording_frame_ptr != 0 {
        // SAFETY: the recording frame is the live `PyFrame` this walk steps,
        // and `frame_layout` pins `last_instr` to this offset with a
        // compile-time assertion against the interpreter's own constant.
        let slot =
            (recording_frame_ptr + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize;
        fbw_note_last_instr_undo(recording_frame_ptr);
        unsafe {
            *slot = py_pc as isize;
        }
    }
}

/// Exception variant of [`fbw_terminate_with_finish`] for the top-level
/// uncaught raise (`compile_exit_frame_with_exception`, pyjitpl.py).
/// Stashes the exception box (`exc`, already a `Type::Ref`) as an
/// `is_exception` payload and, when the raised exception has a concrete Ref,
/// the concrete disposition for the GC root walker / no-replay portal.  Like
/// the value path it does NOT record the `FINISH` op — [`crate::trace::
/// full_body_walk_trace`]'s Terminate arm builds
/// `TraceAction::Finish { exit_with_exception: true }` and the compile
/// consumer records it once against `exit_frame_with_exception_descr`.
pub(crate) fn fbw_terminate_with_raise<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    exc: OpRef,
    exc_concrete: ConcreteValue,
) -> (OpRef, Type) {
    let payload = (exc, Type::Ref);
    ctx.session.borrow_mut().finish_payload = Some(payload);
    FBW_FINISH_IS_EXCEPTION.with(|c| c.set(true));
    if let ConcreteValue::Ref(p) = exc_concrete {
        if !p.is_null() {
            fbw_finish_raise_set(exc_concrete);
        }
    }
    payload
}

/// Whether the terminating payload is a top-level uncaught raise
/// (see [`fbw_terminate_with_raise`]).  Read by the Terminate arm;
/// reset at the start of every walk.
pub(crate) fn fbw_finish_is_exception() -> bool {
    FBW_FINISH_IS_EXCEPTION.with(|c| c.get())
}

/// Map an `abort_permanent` marker's jitcode pc back to the Python opcode
/// the interpreter must resume at.  `emit_abort_permanent` anchors the graph
/// marker at that opcode's `py_pc` and stores `last_instr = py_pc - 1`; the
/// full-body walk flushes the frame there instead of replaying the walked
/// region.  Returns None when the sym's jitcode / `code_ptr` is unavailable
/// (no resume coordinate derivable → legacy replay).
///
/// `Instruction::LoadFastCheck`'s null arm is a dead-end block emitted after
/// the rest of the body.  `py_floor_by_jit_pc` keys each Python PC to its
/// first jitcode offset, so that late block floors onto whichever opcode last
/// opened a segment — the back-edge jump — and resuming there skips the
/// check.  `abort_permanent_py_pc_by_jit_pc` is the marker's own inverse
/// (`loop_body_abort_permanent_pc` resolves the same table).  The floor
/// lookup remains for a skeleton jitcode that carries no marker inverse.
pub(crate) fn fbw_abort_resume_py_pc<Sym: WalkSym>(
    sym: &Sym,
    abort_jit_pc: usize,
) -> Option<usize> {
    if sym.jitcode().is_null() {
        return None;
    }
    // SAFETY: read-only access to the sym's immutable jitcode layout, live
    // for the walk that produced `abort_jit_pc`.
    let jc = unsafe { &*sym.jitcode() };
    if jc.payload.code_ptr.is_null() {
        return None;
    }
    let table = &jc.payload.metadata.abort_permanent_py_pc_by_jit_pc;
    if let Ok(idx) = table.binary_search_by_key(&(abort_jit_pc as u32), |&(off, _)| off) {
        return Some(table[idx].1 as usize);
    }
    Some(
        crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, abort_jit_pc)
            as usize,
    )
}

/// Resolve a jitcode `i` operand to the immutable constant it names.  `None`
/// when it indexes a live int register instead of a constant pool slot.
pub(crate) fn body_int_operand_constant(
    ireg: u8,
    num_regs_i: usize,
    constants_i: &[i64],
) -> Option<i64> {
    (ireg as usize)
        .checked_sub(num_regs_i)
        .and_then(|index| constants_i.get(index))
        .copied()
}

pub(crate) fn fbw_callee_body_has_binary_op_residual(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(op) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if op.opname.starts_with("residual_call")
            && residual_call_descr_index_in_body(body_code, &op)
                .and_then(|index| callee_descr_refs.get(index))
                .and_then(|descr| descr.as_call_descr())
                .is_some_and(|descr| {
                    descr.get_extra_info().runtime_helper == majit_ir::RuntimeHelperKind::BinaryOp
                })
        {
            return true;
        }
        // Flatten lowered BINARY to `inline_call_ir_r` of
        // `binary_value_from_tag` (tag + two refs).  Without this the
        // root-bridge self-rec admission (`bridge_rec_root_selfrec`)
        // never sees a BinaryOp and the nested recursive CALL aborts
        // the enclosing bridge (`selfrec_bridge_nontail_promote`).
        if op.opname.starts_with("inline_call_ir_r") {
            let i_len = body_code.get(op.pc + 3).copied().unwrap_or(0);
            let r_len_pc = op.pc + 3 + 1 + i_len as usize;
            if i_len == 1 && body_code.get(r_len_pc) == Some(&2) {
                return true;
            }
        }
        pc = op.next_pc;
    }
    false
}

/// Refuse a commit the enclosing `BINARY_OP` / `COMPARE_OP` rewind cannot undo.
///
/// A dunder inlined on that entry's rewind has exactly one exit, the
/// record-time cut, and the cut is legal only while the region has applied
/// nothing: a `NotImplemented` result has to go back to the binary protocol,
/// and every route back re-executes the body.  So the answer has to be given
/// BEFORE the commit, which is what this is; by the time the odometer has
/// moved, the write that had to be prevented has happened.
///
/// The odometer's sites are what this has to cover, and they split in two.
/// The unjournaled pair is refused here — `residual`, through
/// [`fbw_abort_nested_unjournaled_residual`], and `store_attr_direct`, whose
/// specialization performs the write inside the resolver and reaches no other
/// hook.  The journaled four — `store_journal`, `list_append`,
/// `list_pop_end`, `cell_store_journal` — need no refusal: they move the
/// odometer. A descent that then returns to the residual rewinds that suffix
/// with [`fbw_effect_journal_rollback_since`]. A descent that aborts the
/// walk is undone by [`fbw_store_journal_rollback`] instead.
pub(crate) fn fbw_binop_rewind_refuse_commit<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
    receiver: Option<OpRef>,
) -> Result<(), DispatchError> {
    let mut session = ctx.session.borrow_mut();
    if session.binop_rewind_depth == 0 {
        return Ok(());
    }
    // A write into an object the region itself allocated is not a write the
    // cut has to undo — see [`WalkSession::binop_rewind_fresh_from`].  The
    // position is the region's starting `_index`: value boxes use that
    // sequence (`OpRef::RefOp` `raw()`), and `_count` would refuse a fresh
    // allocation after any void op.  The heap cache answers the rest, since
    // `new_object` raises `HF_IS_UNESCAPED` on the allocation and `escape_box`
    // lowers it, so an object since handed to something outside the region
    // stops answering here.  A constant or an input arg is neither; matching
    // the `*Op` variants keeps `raw()` off the inline-`Const` variants it
    // panics on.
    if let Some(obj) = receiver
        && let Some(from) = session.binop_rewind_fresh_from
        && matches!(
            obj,
            OpRef::IntOp(_) | OpRef::FloatOp(_) | OpRef::RefOp(_) | OpRef::VoidOp(_)
        )
        && obj.raw() >= from
        && ctx.trace_ctx.heap_cache().is_unescaped(obj)
    {
        return Ok(());
    }
    session.binop_rewind_refused = true;
    Err(DispatchError::callee_inline_unsupported(pc))
}

/// Raises a [`fbw_binop_rewind_refuse_commit`] region for one descent.
pub(crate) struct BinopRewindInlineGuard<'s> {
    session: &'s std::cell::RefCell<WalkSession>,
    /// Whether this guard is the one that raised the region, as opposed to a
    /// nested entry inside one already standing.
    outermost: bool,
}

impl<'s> BinopRewindInlineGuard<'s> {
    /// `from` is the recorder position the region starts at — see
    /// [`WalkSession::binop_rewind_fresh_from`].  A nested entry keeps the
    /// outermost one's, so an allocation of the enclosing region stays exempt
    /// while the inner descent runs.
    pub(crate) fn enter(session: &'s std::cell::RefCell<WalkSession>, from: u32) -> Self {
        let mut s = session.borrow_mut();
        let outermost = s.binop_rewind_depth == 0;
        // The outermost entry starts from a clean slate, so a flag left behind
        // by a descent that unwound for some other reason cannot be read as
        // this one's refusal.
        if outermost {
            s.binop_rewind_refused = false;
            s.binop_rewind_fresh_from = Some(from);
        }
        s.binop_rewind_depth += 1;
        drop(s);
        Self { session, outermost }
    }

    /// Whether a commit was refused while this guard was in force.
    ///
    /// A guard nested inside another region does NOT clear the flag: the
    /// enclosing entry's rewind is exposed by the same commit, and consuming
    /// the flag here would leave it looking at a descent that declined for no
    /// stated reason and continuing past it.  Only the entry that raised the
    /// region takes it.
    pub(crate) fn refused(&self) -> bool {
        let mut s = self.session.borrow_mut();
        let refused = s.binop_rewind_refused;
        if refused && self.outermost {
            s.binop_rewind_refused = false;
        }
        refused
    }
}

impl Drop for BinopRewindInlineGuard<'_> {
    fn drop(&mut self) {
        let mut s = self.session.borrow_mut();
        s.binop_rewind_depth -= 1;
        if s.binop_rewind_depth == 0 {
            s.binop_rewind_fresh_from = None;
        }
    }
}
