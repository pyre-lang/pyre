//! Guard resume-data snapshot capture.
//!
//! **Parity:** trace-side counterpart of `resume.py`'s `capture_resumedata`
//! (implementation mirror `majit-metainterp/resume.rs`).
//!
//! Builds the resume-data snapshot attached to the last recorded guard
//! (the `capture_resumedata` analogue): the single-frame path
//! (`walker_capture_snapshot_for_last_guard*`) and the multi-frame inline
//! path (`walker_capture_multi_frame_inline_snapshot`) plus the
//! inline-caller-frame computation helpers they rely on.

use super::*;

/// The JitCode `op_pc` is an offset into — the body currently being walked.
///
/// `fbw_mode.snapshot_sym.jitcode()` is the eval-loop portal. Pairing that
/// identity with a per-function or inlined-callee offset invents a coordinate
/// `can_decode_live_vars` rejects (`resume.py` `jitcodes[jitcode_pos]`).
/// Inline callees already name their body through [`InlineCalleeConsts`]; a
/// top-level per-function walk stamps [`WalkSession::recording_jitcode_index`].
fn walked_snapshot_jitcode<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<crate::state::JitCode> {
    let index = if let Some(consts) = ctx.inline_callee_consts {
        consts.jitcode_index
    } else {
        let rec = ctx.session.borrow().recording_jitcode_index;
        if rec >= 0 {
            rec
        } else {
            let snapshot_sym = ctx.fbw_mode.snapshot_sym;
            if snapshot_sym.is_null() {
                return None;
            }
            let sym = unsafe { &*snapshot_sym };
            if sym.jitcode().is_null() {
                return None;
            }
            unsafe { (*sym.jitcode()).index as i32 }
        }
    };
    if index < 0 {
        return None;
    }
    let payload = crate::state::pyjitcode_for_jitcode_index(index)?;
    Some(crate::state::JitCode { index, payload })
}

/// Exact `jtransform.py handle_residual_call` trailing `-live-` marker.
///
/// The codewriter emits this immediately after every may-force/can-raise
/// residual.  Runtime per-Code metadata normally carries the same coordinate;
/// a terminal opcode immediately after the call (`RAISE_VARARGS`, for example)
/// may have no first-insn pivot, so retain the bytecode-local identity as the
/// authoritative fallback rather than rewinding to the call.
fn trailing_live_marker(payload: &crate::PyJitCode, call_pc: usize) -> Option<usize> {
    let call = crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), call_pc)?;
    let marker =
        crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), call.next_pc)?;
    (marker.opname == "live").then_some(marker.pc)
}

/// The `-live-` directly before `op_pc`: the liveness a plain guard reads
/// (`get_list_of_active_boxes`, `pc = self.pc - SIZE_LIVE_OP`).
/// The marker must be a decodable `-live-` startpoint, not an operand byte equal to `op_live`.
pub(crate) fn preceding_live_marker(payload: &crate::PyJitCode, op_pc: usize) -> Option<usize> {
    const SIZE_LIVE_OP: usize = majit_jitcode::liveness::OFFSET_SIZE + 1;
    let pc = op_pc.checked_sub(SIZE_LIVE_OP)?;
    let marker = crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), pc)?;
    (marker.opname == "live"
        && marker.next_pc == op_pc
        && payload
            .jitcode
            .can_decode_live_vars(pc, crate::state::op_live()))
    .then_some(pc)
}

/// The last `-live-` at or before `op_pc`.
fn live_marker_at_or_before(payload: &crate::PyJitCode, op_pc: usize) -> Option<usize> {
    crate::jitcode_runtime::decoded_ops(payload.jitcode.code.as_slice())
        .take_while(|op| op.pc <= op_pc)
        .filter(|op| op.opname == "live")
        .last()
        .map(|op| op.pc)
}

/// Whether a `jit_merge_point` sits in `[op_pc, marker)`.
fn merge_point_between(payload: &crate::PyJitCode, op_pc: usize, marker: usize) -> bool {
    crate::jitcode_runtime::decoded_ops(payload.jitcode.code.as_slice())
        .skip_while(|op| op.pc < op_pc)
        .take_while(|op| op.pc < marker)
        .any(|op| op.opname == "jit_merge_point")
}

/// Resume coordinate for a plain (non-after-residual) guard.
///
/// `op_pc` is the capture coordinate. Branch guards pass the jump's own
/// `-live-` (`guarded_branch_core` capture_pc = `live_before_jit_pc`;
/// `pyjitpl.py` `opimpl_goto_if_not` → `generate_guard(resumepc=orgpc)`).
/// That word is already the `get_list_of_active_boxes` coordinate
/// (`pc = self.pc - SIZE_LIVE_OP` when `self.pc` is the goto). Subtracting
/// `SIZE_LIVE_OP` from the `-live-` itself looks at a non-adjacent byte;
/// the opcode-start twin then re-executes the residual before the jump
/// (`bhimpl_goto_if_not` must re-run only the jump). When `op_pc` is still
/// the instruction after a `-live-`, [`preceding_live_marker`] recovers it.
/// The opcode-start marker of a loop-header floor is past its `jit_merge_point`
/// (`jtransform.py handle_jit_marker__jit_merge_point` op2 `-live-`), the
/// point the opcode body re-executes from.
pub(crate) fn plain_guard_resume_jitcode_pc(payload: &crate::PyJitCode, op_pc: usize) -> i32 {
    let op_live = crate::state::op_live();
    if payload.jitcode.can_decode_live_vars(op_pc, op_live) {
        op_pc as i32
    } else {
        preceding_live_marker(payload, op_pc)
            .or_else(|| payload.resume_marker_for_jitcode_pc(op_pc))
            .map(|m| {
                jit_merge_point_pc_in_containing_floor(payload, op_pc)
                    .filter(|&merge_pc| m < merge_pc && merge_pc < op_pc)
                    .and_then(|merge_pc| resume_pc_past_merge_point(payload, merge_pc))
                    .unwrap_or(m) as i32
            })
            .unwrap_or(op_pc as i32)
    }
}

/// Select the resume marker for an after-residual-call guard.  The bytecode's
/// immediate trailing `-live-` is the RPython authority; metadata twins are
/// compatibility fallbacks for incomplete/fixture bodies.
pub(crate) fn after_residual_guard_marker(
    payload: &crate::PyJitCode,
    op_pc: usize,
    marker_call_jit_pc: Option<usize>,
) -> Option<usize> {
    trailing_live_marker(payload, op_pc).or_else(|| {
        marker_call_jit_pc
            .and_then(|call_jit_pc| payload.after_residual_call_resume_for_jitcode_pc(call_jit_pc))
            .or_else(|| payload.after_residual_marker_for_jitcode_pc(op_pc))
            .or_else(|| payload.after_residual_call_resume_for_jitcode_pc(op_pc))
    })
}

/// Read the Python PC paired with a native resume word.  The codewriter builds
/// this table from Python-PC keys, including the same forward trivia skip as
/// `trivia_normalized_py_pc_for_jitcode_pc`; capture deliberately never derives
/// it from the word.
fn forward_snapshot_py_pc(jitcode_index: u32, pc: u32) -> Result<u32, DispatchError> {
    if pc == majit_ir::resumedata::NO_JITCODE_PC as u32 {
        return Ok(u32::MAX);
    }
    let payload = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
        .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: pc as usize })?;
    // A level with no Python code object has no Python pc, and so no entry in
    // a table keyed by one. Helper bodies and `descr_call`'s tail
    // (`crate::ctor_continuation`) are such levels; `u32::MAX` is the same
    // "no coordinate" answer the sentinel arm above gives, and what every
    // reader of this field already treats as absent.
    if payload.code_ptr.is_null() {
        return Ok(u32::MAX);
    }
    payload
        .resume_position_for_jitcode_pc(pc as usize)
        .map(|(_, py_pc)| py_pc)
        .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: pc as usize })
}

pub(crate) fn vstack_box_for_snapshot(value: OpRef) -> Option<OpRef> {
    (value != OpRef::NONE).then_some(value)
}

/// `generate_guard` (`pyjitpl.py`) keys `after_residual_call`
/// on the guard opcode itself: `GUARD_EXCEPTION` / `GUARD_NO_EXCEPTION` /
/// `GUARD_NOT_FORCED` / `GUARD_ALWAYS_FAILS` resume *after* the residual
/// call; every other guard resumes at its own opcode.  The call already
/// executed in compiled code and consumed its Python stack operands;
/// resuming at the call's own opcode would re-execute it from a
/// coordinate whose stack no longer holds those operands,
/// dropping/duplicating the side effect (e.g. an in-place `a[i] = a[j]`
/// swap losing one store at a guard-failure transition).  #124/#281.
pub(crate) fn walker_capture_snapshot_for_last_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    walker_capture_snapshot_for_last_guard_scoped(ctx, op_pc, GuardCaptureScope::default())
}

/// `jit_merge_point` inside the Python-opcode floor that contains `jit_pc`.
///
/// Loop-header py_pcs lower `emit_portal_is_being_profiled_read` at the floor
/// and `jit_merge_point` later in the same `py_floor_by_jit_pc` region.
pub(super) fn jit_merge_point_pc_in_containing_floor(
    payload: &crate::PyJitCode,
    jit_pc: usize,
) -> Option<usize> {
    let (floor_start, _) = crate::pyjitcode::floor_segment_for_jitcode_pc(
        &payload.metadata.py_floor_by_jit_pc,
        jit_pc,
    )?;
    let region_end = payload
        .metadata
        .py_floor_by_jit_pc
        .iter()
        .find(|&&(off, _)| off as usize > floor_start)
        .map(|&(off, _)| off as usize)
        .unwrap_or(payload.jitcode.code.len());
    let mut pc = floor_start;
    while pc < region_end {
        let op = crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), pc)?;
        if op.key.starts_with("jit_merge_point") {
            return Some(op.pc);
        }
        if op.next_pc <= pc {
            break;
        }
        pc = op.next_pc;
    }
    None
}

/// True when `op_pc` sits in a loop-header floor *before* that floor's
/// `jit_merge_point`: the `PyFrame.get_is_being_profiled` diamond
/// (`emit_portal_is_being_profiled_read`) and the green promotes that
/// `interp_jit.py PyFrame.dispatch` evaluates ahead of `jit_merge_point`.
pub(super) fn ptr_iszero_is_before_merge_in_floor<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> bool {
    walked_snapshot_jitcode(ctx)
        .and_then(|jc| {
            jit_merge_point_pc_in_containing_floor(&jc.payload, op_pc)
                .map(|merge_pc| op_pc < merge_pc)
        })
        .unwrap_or(false)
}

/// Byte after `jit_merge_point`: the op2 `-live-` when present, otherwise
/// `op.next_pc` (`jtransform.py handle_jit_marker__jit_merge_point`).
pub(super) fn resume_pc_past_merge_point(
    payload: &crate::PyJitCode,
    merge_pc: usize,
) -> Option<usize> {
    trailing_live_marker(payload, merge_pc).or_else(|| {
        crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), merge_pc)
            .map(|op| op.next_pc)
    })
}

/// Resume pc for a synthesized per-opcode cut: op2 `-live-` after a
/// same-floor `jit_merge_point` (`pyjitpl.py generate_guard`
/// `after_residual_call=True`), else the cut itself.
pub(crate) fn segmented_python_opcode_resume_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> usize {
    walked_snapshot_jitcode(ctx)
        .and_then(|jc| {
            jit_merge_point_pc_in_containing_floor(&jc.payload, op_pc)
                .and_then(|merge_pc| resume_pc_past_merge_point(&jc.payload, merge_pc))
        })
        .unwrap_or(op_pc)
}

/// Capture the `GUARD_ALWAYS_FAILS` used by a synthesized per-Python-opcode
/// debug merge point.
///
/// Upstream records that guard in the interpreter dispatch JitCode, after its
/// real `jit_merge_point`; `after_residual_call=True` advances past that marker
/// and lands on the Python opcode the dispatch loop is about to execute.
/// Pyre's per-CodeObject JitCode has no such marker. Its synthesized DMP is
/// already sitting at the first operation of that Python opcode, so applying
/// the generic `GUARD_ALWAYS_FAILS` advance would skip the opcode entirely.
/// A loop-header floor still contains the portal `jit_merge_point`; resume at
/// that marker's op2 `-live-` so the blackhole does not re-enter the preamble.
pub(crate) fn walker_capture_segmented_python_opcode_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    let resume_pc = segmented_python_opcode_resume_pc(ctx, op_pc);
    if resume_pc != op_pc {
        return walker_capture_snapshot_for_last_guard_impl(
            ctx,
            op_pc,
            false,
            GuardCaptureScope {
                carried_resume_jit_pc: Some(resume_pc),
                liveness_jit_pc: Some(resume_pc),
                ..GuardCaptureScope::default()
            },
        );
    }
    walker_capture_snapshot_for_last_guard_impl(ctx, op_pc, false, GuardCaptureScope::default())
}

/// JitCode pc the segmented merge-point guard resumes at: the `-live-`
/// immediately after `jit_merge_point` (`jtransform.py
/// handle_jit_marker__jit_merge_point` op2). Falls back to the first byte
/// after the merge op when that marker is missing.
pub(crate) fn segmented_merge_point_resume_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    merge_pc: usize,
) -> usize {
    walked_snapshot_jitcode(ctx)
        .and_then(|jc| resume_pc_past_merge_point(&jc.payload, merge_pc))
        .unwrap_or(merge_pc)
}

/// Capture the `GUARD_ALWAYS_FAILS` recorded at a portal `jit_merge_point`.
///
/// `pyjitpl.py generate_guard` with `after_residual_call=True` snapshots
/// `self.pc` after the merge-point handler has decoded the op
/// (`jtransform.py handle_jit_marker__jit_merge_point` op2 `-live-`). Carry
/// that following marker verbatim so the bridge walk enters at
/// `MIFrame.setup_resume_at_op`'s stored pc.
pub(crate) fn walker_capture_segmented_merge_point_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    merge_pc: usize,
) -> Result<(), DispatchError> {
    let resume_pc = segmented_merge_point_resume_pc(ctx, merge_pc);
    walker_capture_snapshot_for_last_guard_impl(
        ctx,
        merge_pc,
        false,
        GuardCaptureScope {
            carried_resume_jit_pc: Some(resume_pc),
            liveness_jit_pc: Some(resume_pc),
            ..GuardCaptureScope::default()
        },
    )
}

pub(crate) fn walker_capture_snapshot_for_last_guard_scoped<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    scope: GuardCaptureScope,
) -> Result<(), DispatchError> {
    // `pyjitpl.py generate_guard` keys `after_residual_call` on the
    // opcode (`GUARD_EXCEPTION` included). Shared `dispatch_bytecode`
    // residuals capture at the `goto L_shared` instead, and that
    // startpoint must not take the after-residual advance
    // (`GuardCaptureScope::preserve_vable_last_instr`).
    let after_residual_call = if scope.preserve_vable_last_instr {
        false
    } else {
        matches!(
            ctx.trace_ctx.last_guard_opcode(),
            Some(
                OpCode::GuardException
                    | OpCode::GuardNoException
                    | OpCode::GuardNotForced
                    | OpCode::GuardAlwaysFails
            )
        )
    };
    let scope = GuardCaptureScope {
        residual_call_catch_resume: scope.residual_call_catch_resume || after_residual_call,
        ..scope
    };
    walker_capture_snapshot_for_last_guard_impl(ctx, op_pc, after_residual_call, scope)
}

fn inline_snapshot_has_complete_parent_chain(n_parents: usize, n_callees: usize) -> bool {
    n_parents > 0 && n_parents == n_callees
}

/// Whether a guard emitted at the current inline depth resumes at its own
/// callee coordinate instead of collapsing to the caller's CALL boundary.
///
/// The multi-frame snapshot below fires only when the paused-caller chain
/// covers the full inline depth — one parent per active inlined callee.  A
/// shorter chain falls through to the single-frame collapse, whose resume
/// re-executes the entire call, so a guard emitted under it re-runs every side
/// effect the inline region sequenced before it.  A fold that must not be
/// re-run consults this before emitting its guards.
pub(crate) fn walker_inline_guard_resumes_in_callee<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> bool {
    let session = ctx.session.borrow();
    let n_parents = session
        .framestack
        .iter()
        .filter(|frame| !frame.parents.is_empty())
        .count();
    inline_snapshot_has_complete_parent_chain(n_parents, session.inline_depth())
}

/// `capture_resumedata(after_residual_call=True)` (`pyjitpl.py`) reads
/// the MIFrame registers as they stand after the residual call.  The
/// capture below advances the operand-stack mirror to that same post-call
/// boundary so the guard's virtualizable image sees the residual result;
/// that is the transition the following walk step would make.  The walk
/// may still emit `-live-` markers owned by the retired opcode; those
/// report the earlier py_pc and would look like a backward layout switch.
/// `step_vstack_mirror` ignores that marker for GET_ITER / LOAD_SPECIAL
/// while their two pushed slots are still empty in the shadow, so the
/// residual result stays in the live mirror. After those pushes fill the
/// shadow, the same marker reseeds from it. Other retired opcodes reseed.
pub(crate) fn walker_capture_snapshot_for_last_guard_impl<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    after_residual_call: bool,
    scope: GuardCaptureScope,
) -> Result<(), DispatchError> {
    // A guard whose resume snapshot cannot be built must abort the trace,
    // not panic.  `build_vable_snapshot_boxes` requires every virtualizable
    // box (including the identity at `[-1]`) to carry `OpRef::ty()`; a
    // deeper inlined / recursive frame can leave the identity untyped.
    // Surface a typed abort rather than tripping the
    // invariant panic (the multi-frame vable snapshot is unported).
    if !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::GuardSnapshotVableUntyped { pc: op_pc });
    }
    // Snapshot semantics for walker-emitted guards
    // (`pyjitpl.py generate_guard` + `capture_resumedata`):
    //
    // `capture_resumedata` numbers one resume section per `MIFrame` on
    // `metainterp.framestack` (`resume.py` `ResumeDataLoopMemo.number`).
    // Helper jitcodes are real `MIFrame`s: `blackhole_from_resumedata`
    // re-enters each section, `_setup_return_value_*` writes the helper
    // result into the caller's result register (`code[position-1]`), and
    // `_resume_mainloop` continues the caller. A guard inside a
    // transparent helper therefore numbers the full stack
    // outermost-first: paused Python levels, the caller paused past its
    // CALL (`get_list_of_active_boxes(in_a_call=True)`), one section per
    // intermediate paused helper, and the innermost helper at the
    // guard's own jitcode coordinate (`plain_guard_resume_jitcode_pc` /
    // `after_residual_guard_marker`). Deopt runs the helper from the
    // guard to its return; nothing before the guard runs twice.
    //
    // `opencoder.py create_top_snapshot` writes vable_array +
    // vref_array on the top snapshot. Feed the trace-time vable/vref
    // shadow through. Empty when no virtualizable / virtualref is live,
    // matching the upstream 0-length-array shape. The build is deferred
    // until after the resume py_pc is resolved and the `last_instr`
    // vable scalar is published, so the snapshot carries this guard's
    // coordinate (the walker never crosses `set_orgpc`, so the scalar
    // is otherwise stale at the loop-header pc — see the publish below).

    // Full-body walk (Phase 7): `op_pc` is an offset into the body being
    // walked (`recording_jitcode_index` / inlined callee), which is the
    // per-CodeObject jitcode for a portal-shaped Python function. Map it
    // back to the containing Python opcode (the blackhole resumes and
    // re-executes that opcode — `orgpc` parity) and read liveness from
    // the live walk register banks at that pc. `snapshot_sym.jitcode()` is
    // the eval-loop portal and does not own this offset.
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if let Some(result) = scope.finished_callee_result {
        return walker_capture_finished_callee_snapshot(ctx, op_pc, result);
    }
    // Inline sub-walk: the guard's `op_pc` is a *callee* coordinate that
    // does not exist in the outer (`full_body_sym`) jitcode's py_pc→jitcode
    // tables.
    // Skip the full-body mapping and fall through to the caller-boundary
    // capture below, which resumes at the CALL site (re-execute the
    // call on deopt — see `fbw_mode.inline_subwalk`).
    let inline_subwalk = ctx.fbw_mode.inline_subwalk;
    // #68 multi-frame inline guard: a guard emitted inside an inlined callee
    // sub-walk with paused caller frames on the walk framestack resumes
    // BOTH the callee (at its own pc) and the caller(s) (at the CALL return
    // point), instead of collapsing to the caller boundary (re-execute).  Only
    // the forward-branch inline path
    // populates the chain; straight-line callees keep the empty chain + the
    // single-frame collapse below.
    if inline_subwalk {
        let parent_frames = {
            let session = ctx.session.borrow();
            session
                .framestack
                .iter()
                .flat_map(|frame| frame.parents.iter().cloned())
                .collect::<Vec<_>>()
        };
        // Fire the multi-frame snapshot only when the paused-caller chain
        // covers the FULL current inline depth. A nested straight-line callee
        // inlined under a multiframe ancestor (e.g. `add3` inside a multiframe
        // `mix`) pushes NO parent frame, so its own guards see a SHORTER chain
        // than the callee depth — fall through to the single-frame collapse
        // (the strict callee's resume-at-CALL behavior) rather than emit a
        // chain that skips the intermediate frame.
        //
        // Numbered helper levels number their callers plus the helper
        // sections (`capture_resumedata`). The Python levels below them take
        // the same full-chain rule (`numbered_helper_guard_resume`).
        let helper_chain = if ctx.fbw_mode.transparent_helper_subwalk {
            ctx.session.borrow().helper_chain()
        } else {
            HelperChain::NoHelper
        };
        if helper_chain == HelperChain::Unresumable {
            return Err(DispatchError::callee_inline_unsupported(op_pc));
        }
        if helper_chain == HelperChain::Numbered {
            match numbered_helper_guard_resume(
                ctx,
                op_pc,
                after_residual_call,
                scope,
                parent_frames,
            )? {
                NumberedHelperGuard::Frames(frames) => {
                    return walker_capture_transparent_helper_snapshot(
                        ctx,
                        op_pc,
                        frames,
                        scope.guard_stamp,
                    );
                }
                // Fall through to the root portal's CALL boundary below.
                NumberedHelperGuard::CallerBoundary => {}
            }
        } else if walker_inline_guard_resumes_in_callee(ctx) {
            // A STRICT straight-line callee (gh#420) whose own frame is not
            // MF-snapshot-able (a kept operand-stack temp the sub-walk does not
            // mirror) propagates the `Unsupported` error the same as the branch
            // path: the enclosing inline declines to a residual call rather than
            // resuming through the single-frame collapse, whose caller-boundary
            // re-execute both mis-sizes the resumed frame (a decode/`LOAD_FAST`
            // out-of-bounds) and re-applies the callee's committed side effect.
            // The caller names the guard (`GuardFromEnd` when the guard is
            // not the last op). Overwriting that with `LastOp` attaches the
            // snapshot to a later op and leaves the guard at `resume_pos == -1`.
            let guard_stamp = scope.guard_stamp;
            return walker_capture_multi_frame_inline_snapshot(
                ctx,
                op_pc,
                after_residual_call,
                parent_frames,
                scope,
                guard_stamp,
            );
        }
    }
    if !inline_subwalk && !full_body_sym.is_null() {
        // SAFETY: the pointer is set only for the lifetime of the
        // full-body `dispatch_via_miframe` (the guard restores it on
        // exit); the `PyreSym` outlives the walk.  Read-only access to
        // immutable layout fields (jitcode / color maps / frame / ec) —
        // the walk's mutable register file lives in `ctx.registers_*`
        // (fresh `top_regs`), not in `sym.registers_*`.
        let sym = unsafe { &*full_body_sym };
        if !sym.jitcode().is_null() {
            let walked_owned = walked_snapshot_jitcode(ctx).unwrap_or_else(|| {
                let jc = unsafe { &*sym.jitcode() };
                crate::state::JitCode {
                    index: jc.index,
                    payload: std::sync::Arc::clone(&jc.payload),
                }
            });
            // Set when an after-residual-call guard's residual call sits inside
            // a try-block (its per-CodeObject jitcode emitted a post-call
            // catch); the JitCode resume coordinate then selects that catch.
            let mut marker_call_jit_pc: Option<usize> = None;
            let (py_pc, jitcode_index, num_instrs) = unsafe {
                let jc = &walked_owned;
                // Forward py twin first (#73 phase-3): equals the containing
                // coordinate plus trivia normalization by construction; the
                // containing lookup survives for the empty-twin class.
                let mut py = jc
                    .payload
                    .forward_py_pc_for_jitcode_pc(op_pc)
                    .unwrap_or_else(|| {
                        crate::py_coord::note_empty_twin_fallback(
                            "capture_seam",
                            jc.index,
                            op_pc as i32,
                        );
                        crate::py_coord::containing_py_pc_for_jitcode_pc(
                            &jc.payload.metadata,
                            op_pc,
                        )
                    });
                // A resolved py must be a real opcode, never Python trivia
                // (e.g. a branch target whose block lowers `NOT_TAKEN`): the
                // interpreter resumes branches at `semantic_fallthrough_pc` /
                // `jump_target_forward`, both of which forward-skip trivia.
                // Land on the same real opcode so the resume reader's BACKWARD
                // trivia backtrack (call_jit.rs) is a no-op — otherwise a
                // `NOT_TAKEN` py_pc backtracks to the preceding branch
                // opcode, whose block-entry liveness differs from the
                // target's and desyncs the snapshot box-count.  The twin is
                // already trivia-normalized, so the skip below is an identity
                // on that path and normalizes only the inversion fallback.
                if !jc.payload.code_ptr.is_null() {
                    let code = &*jc.payload.code_ptr;
                    py = skip_python_trivia_forward(code, py as usize) as u32;
                    // Entry-resume capture: the carried coordinate is the
                    // authority; take its forward-py twin (the same pairing
                    // the decoder resolves) instead of the inversion above,
                    // so the liveness/entry windows key exactly where the
                    // decode will.
                    if let Some(carried) = scope.carried_resume_jit_pc {
                        if let Some(fwd) = jc.payload.forward_py_pc_for_jitcode_pc(carried) {
                            py = fwd;
                        }
                    }
                    // after_residual_call=True (`pyjitpl.py`): the
                    // may-force call already executed in compiled code and
                    // consumed its Python stack operands. Resume at the NEXT
                    // executable opcode so the blackhole continues past the call
                    // (re-executing from the call's coordinate would drop/dup the
                    // side effect, e.g. an in-place list swap store). The
                    // fallthrough resume routes a raise through the next opcode's
                    // own `catch_exception` (still inside the same try-block) and
                    // the bridge-decline path, which handles every sequential
                    // residual call whose NEXT opcode shares the try.
                    //
                    // A residual whose CALL pc is itself directly covered by an
                    // enclosing exception-table handler needs its OWN catch to
                    // receive the raise: its fallthrough may leave the covered
                    // region (e.g. FOR_ITER-next's fallthrough is the continue-arm
                    // body, reached only on a NON-null item, which carries no
                    // catch for the call's OWN raise). When the guard capture
                    // requested catch-resume
                    // (`GuardCaptureScope::residual_call_catch_resume`) and the
                    // call's CALL pc is covered by the code's exception table,
                    // carry the CALL jitcode offset so the blackhole resumes at
                    // the call's OWN catch and routes the raise to the enclosing
                    // handler instead of escaping the frame.
                    if after_residual_call {
                        let call_py_pc = py;
                        let exact_after_marker = jc
                            .payload
                            .after_residual_marker_for_jitcode_pc(op_pc)
                            .or_else(|| jc.payload.after_residual_call_resume_for_jitcode_pc(op_pc))
                            .or_else(|| trailing_live_marker(&jc.payload, op_pc));
                        py = exact_after_marker
                            .and_then(|marker| jc.payload.forward_py_pc_for_jitcode_pc(marker))
                            .unwrap_or_else(|| {
                                crate::pyjitpl::semantic_fallthrough_pc(code, py as usize) as u32
                            });
                        let call_pc_has_catch = pyre_interpreter::pycode::lookup_exceptiontable(
                            &code.exceptiontable,
                            call_py_pc * 2,
                        )
                        .is_some();
                        if scope.residual_call_catch_resume
                            && call_pc_has_catch
                            && jc
                                .payload
                                .after_residual_call_resume_for_jitcode_pc(op_pc)
                                .is_some()
                        {
                            marker_call_jit_pc = Some(op_pc);
                        }
                    }
                }
                (
                    py,
                    jc.index as u32,
                    jc.payload.metadata.n_py_instrs as usize,
                )
            };
            // #67/#124: synthetic loop-close guard pc overshoots past the last
            // Python opcode → resume at the trace's entry py (loop header).
            let loop_close_overshoot = py_pc as usize >= num_instrs;
            let py_pc = if loop_close_overshoot {
                ctx.entry_py_pc()
            } else {
                py_pc
            };
            // The three resume-depth consumers below (vstack reconcile,
            // valuestackdepth publication, and vable stack sync) share one
            // JitCode-PC twin. Plain guards key their trivia twin at `op_pc`;
            // after-residual guards key the same family at their post-call
            // marker. A loop-close overshoot falls back to entry_py_pc and has
            // no corresponding op_pc-derived twin, so it retains the legacy
            // Python-PC lookup below.
            let resume_depth_twin: Option<u16> = unsafe {
                let jc = &walked_owned;
                if loop_close_overshoot {
                    None
                } else if scope.carried_resume_jit_pc.is_some() {
                    // Entry-resume capture: the depth twins are keyed for
                    // op-START coordinates; keying them with an
                    // already-advanced resume coordinate reads the depth of
                    // the coordinate's KEY opcode (the call, with its
                    // operands still on the stack) — an over-count whose
                    // published `valuestackdepth` makes the resume read
                    // garbage slots as Refs.  Fall back to the raw static
                    // liveness at the resume py (the forward twin above),
                    // exactly what the decode derives.
                    None
                } else if after_residual_call {
                    jc.payload
                        .after_residual_marker_for_jitcode_pc(op_pc)
                        .or_else(|| jc.payload.after_residual_call_resume_for_jitcode_pc(op_pc))
                        .or_else(|| trailing_live_marker(&jc.payload, op_pc))
                        .and_then(|marker| jc.payload.depth_trivia_for_jitcode_pc(marker))
                } else {
                    jc.payload.depth_trivia_for_jitcode_pc(op_pc)
                }
            };
            // `capture_resumedata(after_residual_call=True)` snapshots the
            // trailing `-live-`, after the residual result has replaced the
            // Python opcode's consumed operands (pyjitpl.py,
            // opencoder.py).  The walk-level stack mirror normally
            // applies that replacement only when `step_vstack_mirror` reaches
            // the next Python opcode.  A guard emitted by the residual itself
            // captures before that step, so advance the mirror here to the
            // same post-call boundary before it supplies the vable snapshot.
            // This is the same transition the following walk step would make;
            // it merely makes the guard's resume image observe it at the
            // required point.  The live walk keeps that post-call mirror so a
            // later FOR_ITER snapshot still sees GET_ITER's iterator.
            //
            // `py_pc == vstack_cur_pypc` is not that transition: the walk has
            // not left the opcode the mirror already holds, so the following
            // step makes no boundary at all — `step_vstack_mirror` returns on
            // the same equality. Reconciling anyway replays that opcode's
            // stack effect a second time, and the permutation classes are not
            // idempotent: a `SWAP` applied twice is the identity, so the two
            // slots it exchanged keep their pre-swap order for the rest of the
            // walk and every later guard snapshot publishes them crossed.
            if after_residual_call && ctx.vstack_valid && py_pc != ctx.vstack_cur_pypc {
                let jc = &walked_owned;
                let code_ptr = jc.payload.code_ptr;
                if !code_ptr.is_null() {
                    let resume_depth = match resume_depth_twin {
                        Some(depth) => depth as usize,
                        None => crate::liveness::liveness_for(code_ptr)
                            .depth_at_py_pc()
                            .get(py_pc as usize)
                            .copied()
                            .unwrap_or(0) as usize,
                    };
                    let code = unsafe { &*code_ptr };
                    reconcile_vstack_at_boundary(ctx, code, py_pc, resume_depth, true);
                }
            }
            // Publish `last_instr = py_pc - 1` to the vable static shadow
            // before snapshotting.  The walker walks JitCode and never
            // crosses `set_orgpc`, so the `last_instr` scalar in
            // `virtualizable_boxes` keeps whatever the trace seed or the
            // previous `close_loop_args_at` override wrote (the loop-header
            // pc).  The blackhole / vable-sync resume reads this scalar into
            // `frame.last_instr`; a stale loop-header value makes a mid-body
            // guard resume at the loop header instead of the guard's own
            // opcode, re-running loop iterations and corrupting the result.
            // Mirror of `MIFrame::publish_last_instr_to_vable`.
            // `store_token_in_vable` emits GUARD_NOT_FORCED_2 only after the
            // exit path has stored the frame's final virtualizable image.
            // Unlike an ordinary guard it does not resume by re-executing the
            // current opcode: async forcing materializes that already-final
            // image.  Keep the `last_instr`/`valuestackdepth` values published
            // by `fbw_publish_exit_last_instr` instead of applying the generic
            // `resume py_pc - 1` rewrite below.  This is the direct shape of
            // PyPy `MetaInterp.store_token_in_vable` followed by
            // `generate_guard(GUARD_NOT_FORCED_2)`.
            let mut async_force_vable_boxes = None;
            let mut saved_vable_scalars = None;
            if sym.owns_virtualizable_shadow()
                && ctx.trace_ctx.last_guard_opcode() != Some(OpCode::GuardNotForced2)
            {
                saved_vable_scalars = Some(crate::trace_opcode::save_vable_resume_scalars(
                    ctx.trace_ctx,
                ));
                if !scope.preserve_vable_last_instr {
                    let last_instr_value = py_pc as i64 - 1;
                    let last_instr_op = ctx.trace_ctx.const_int(last_instr_value);
                    crate::trace_opcode::mirror_vable_static_to_boxes_nosync(
                        ctx.trace_ctx,
                        "last_instr",
                        last_instr_op,
                        Value::Int(last_instr_value),
                    );
                }
                // `ResumeGuardForcedDescr.handle_async_forcing` consumes the
                // virtualizable section WHILE the residual is running. Its
                // result belongs to the MIFrame's post-call registers, but
                // the interpreter has not pushed it yet. Preserve the stack
                // before the post-call depth/operand overlay below, otherwise
                // forcing reads an unwritten result home as a GC Ref.
                // Do publish last_instr first: the walker does not maintain
                // that scalar at each opcode, and callers inspecting this
                // frame must see the current call, not the last merge point.
                let guard_opcode = ctx.trace_ctx.last_op_opcode();
                // Publish `valuestackdepth` for THIS guard's resume
                // coordinate the same way `last_instr` is published above.
                // `sym.valuestackdepth` is NOT usable: the walker never
                // crosses `set_orgpc`, so that scalar keeps the loop-entry
                // inputarg the trace was seeded with (a loop-invariant the
                // optimizer folds to a constant = the loop-header depth).  A
                // guard that resumes at a different depth — the `while`
                // condition branch, or a may-force call whose
                // `semantic_fallthrough_pc` lands on an opcode that keeps an
                // operand-stack temp (`#124`) — then carries the wrong depth
                // into its snapshot.  The resume reader writes this scalar
                // into `frame.valuestackdepth`; an under-count hides the kept
                // operand (the interpreter resumes thinking the stack is
                // shallower than it is), and `setup_bridge_sym` derives
                // `stack_only = valuestackdepth - nlocals` from it
                // (`state.rs` Part 1), so a wrong count desyncs the bridge's
                // operand-stack slots.  Compute the depth at the resume py_pc
                // the SAME way the encoder (`collect_outer_active_boxes`)
                // derives `valid_stack_only` — `nlocals +
                // depth_at_py_pc[py_pc]` — so the published scalar stays
                // symmetric with the active-box layout.  Fall back to
                // `sym.valuestackdepth` only when liveness is unavailable.
                // The publish is snapshot-only like `last_instr`.
                let vsd_value = unsafe {
                    let jc = &walked_owned;
                    if jc.payload.code_ptr.is_null() {
                        sym.valuestackdepth() as i64
                    } else {
                        match resume_depth_twin {
                            Some(d) => (sym.nlocals() + d as usize) as i64,
                            None => crate::liveness::liveness_for(jc.payload.code_ptr)
                                .depth_at_py_pc()
                                .get(py_pc as usize)
                                .copied()
                                .map(|d| (sym.nlocals() + d as usize) as i64)
                                .unwrap_or(sym.valuestackdepth() as i64),
                        }
                    }
                };
                let vsd_op = ctx.trace_ctx.const_int(vsd_value);
                crate::trace_opcode::mirror_vable_static_to_boxes_nosync(
                    ctx.trace_ctx,
                    "valuestackdepth",
                    vsd_op,
                    Value::Int(vsd_value),
                );
                // Capture the force image AFTER last_instr and vsd so a
                // GUARD_NOT_FORCED residual that resumes past the CALL does
                // not carry CALL-entry vsd (args still on the stack).
                if matches!(guard_opcode, Some(OpCode::GuardNotForced)) {
                    async_force_vable_boxes =
                        Some(ctx.trace_ctx.build_snapshot_vable_vref_boxes().0);
                }
            }
            // Sync the live operand-stack registers into the vable shadow for
            // THIS snapshot only.  The walker keeps the operand stack in the
            // walk register file (`ctx.registers_r`); the vable shadow array
            // syncs the operand stack only at merge points
            // (`close_loop_args_at`) and is otherwise NULL for the slots a
            // mid-opcode guard resumes before (the two operands a
            // value-guarded `BINARY_OP` / `BINARY_SUBSCR` pops).  The
            // blackhole rebuilds those from the frame snapshot
            // (`collect_outer_active_boxes`), but the bridge re-trace reads
            // them back from the resumed vable shadow — without this overlay it
            // constant-folds NULL operands into the residual call
            // (`CallR(binop, null, null, ...)`) and the compiled bridge then
            // dereferences a null operand.  Overlay the live operands, build
            // the snapshot, then restore the shadow so this transient write
            // never leaks into a later op or merge-point sync.
            let mut stack_sync: Vec<(usize, OpRef)> = if sym.owns_virtualizable_shadow() {
                let depth = unsafe {
                    let jc = &walked_owned;
                    if jc.payload.code_ptr.is_null() {
                        0usize
                    } else {
                        match resume_depth_twin {
                            Some(depth) => depth as usize,
                            None => crate::liveness::liveness_for(jc.payload.code_ptr)
                                .depth_at_py_pc()
                                .get(py_pc as usize)
                                .copied()
                                .unwrap_or(0) as usize,
                        }
                    }
                };
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                let nlocals = sym.nlocals();
                // #73: the walk-level operand-stack box mirror
                // (`ctx.vstack_boxes[s]`, maintained per-op by
                // `step_vstack_mirror`) is the analog of PyPy
                // `MIFrame.registers_r` and the sole kept-stack source at a
                // branch guard.  The retired flat `stack_slot_color_map` read
                // (`registers_r[color]` plus the #420 edge-move recovery) was
                // unreliable in loop traces: loop-carried slots are renamed to
                // inputarg colors and chordal coloring reuses colors, so a
                // static-color read returned a stale / reused box (the #424
                // merge-color staleness).  Dropping it was proven redundant —
                // force-off byte-identical on 37/169 corpus benches on both
                // backends — so a slot the mirror does not cover (invalid
                // mirror, slot beyond the mirror, or an Int-bank temp the
                // Ref-only mirror leaves NONE) is simply omitted; resume
                // re-materializes it rather than reading the flat color. A
                // ConstPtr(NULL) is not a hole: it is CALL's live
                // `self_or_null` operand and must overwrite a stale shadow
                // slot in the capture-only overlay.
                (0..depth)
                    .filter_map(|s| {
                        let v = if ctx.vstack_valid {
                            ctx.frame_state
                                .borrow()
                                .vstack_boxes
                                .get(s)
                                .copied()
                                .unwrap_or(OpRef::NONE)
                        } else {
                            OpRef::NONE
                        };
                        vstack_box_for_snapshot(v).map(|v| (nvs + nlocals + s, v))
                    })
                    .collect()
            } else {
                Vec::new()
            };
            // `MetaInterp.capture_resumedata` (`pyjitpl.py`) temporarily sets
            // `MIFrame.pc` to `resumepc` before `get_list_of_active_boxes`
            // and restores the tracing pc afterwards. A branch guard's
            // `resumepc` is the jump's own `-live-` (`generate_guard(resumepc=orgpc)`);
            // blackhole `bhimpl_goto_if_not` re-evaluates the restored
            // condition. Specialization and after-residual guards keep the
            // marker construction below so encoder banks and decoder
            // liveness stay on one coordinate.
            let guard_jitcode_pc: i32 = if let Some(carried) = scope.carried_resume_jit_pc {
                // Bridge-entry flavor guard: re-carry the source guard's own
                // resume word verbatim (see `GuardCaptureScope`).  It is a
                // decodable `-live-` startpoint by construction — the runtime
                // resolved it to reach this walk.
                carried as i32
            } else if !after_residual_call
                && matches!(
                    ctx.trace_ctx.last_guard_opcode(),
                    Some(
                        OpCode::GuardValue
                            | OpCode::GuardClass
                            | OpCode::GuardTrue
                            | OpCode::GuardFalse
                            | OpCode::GuardIsnull
                            | OpCode::GuardNonnull
                    )
                )
            {
                // #366: carry the `-live-` marker offset, NOT the raw guard
                // `op_pc`. Reached on the specialization guards
                // (`GuardValue`/`GuardClass`), the compare-fused branch
                // guards (`GuardTrue`/`GuardFalse`), and the pointer-nullity
                // pair (`GuardIsnull`/`GuardNonnull`) from
                // `_establish_nullity` (`opimpl_goto_if_not_ptr_iszero` /
                // `opimpl_goto_if_not_ptr_nonzero`). For every guard here
                // the codewrite-time twin resolves the `-live-` marker,
                // keeping the encoder reg-bank window and decoder liveness
                // symmetric.  It is a valid startpoint
                // (`can_decode_live_vars` holds).
                // The `after_residual_call` family is excluded — it routes
                // through the separate post-call catch-marker twin, so the
                // ordinary marker would name a different resume point.
                // Every guard-capture point is emitted after a `-live-` marker;
                // its populated codewrite-time twin is therefore total here.
                // All of these source the resume coordinate from the walk
                // cursor's per-op `-live-` BEFORE (`ctx.live_before_jit_pc`).
                // That is `generate_guard(resumepc=orgpc)` (`pyjitpl.py
                // opimpl_goto_if_not` / `_establish_nullity`); blackhole
                // `bhimpl_goto_if_not` / `bhimpl_goto_if_not_ptr_iszero`
                // re-evaluates the restored condition. `op_pc` is already
                // that `-live-` for a branch capture; the opcode-start twin
                // is only the fallback when neither the capture pc nor the
                // adjacent preceding `-live-` is a decode startpoint.
                let live_before_decodable = ctx.live_before_jit_pc != usize::MAX
                    && unsafe {
                        (&walked_owned)
                            .payload
                            .jitcode
                            .can_decode_live_vars(ctx.live_before_jit_pc, crate::state::op_live())
                    };
                if live_before_decodable {
                    ctx.live_before_jit_pc as i32
                } else {
                    plain_guard_resume_jitcode_pc(&walked_owned.payload, op_pc)
                }
            } else if after_residual_call {
                // #366: extend the direct-pc carry to the after-residual-call
                // guard family (`GuardException`/`GuardNoException`/
                // `GuardNotForced`/`GuardAlwaysFails` — exactly what the
                // `after_residual_call` bool tracks, `walker_capture_snapshot_for_last_guard`).
                // These resume AFTER the residual call at the marker the
                // decoder already resolves for this guard, so carrying it
                // directly makes decode consult the carried word instead of a
                // fallback `py_pc → jitcode` resume translation.
                //
                // Which marker the decoder resolves depends on the sub-case
                // captured at the CALL pc:
                //   * `Some(call_jit_pc)` — the residual call is in a try-block
                //     (FOR_ITER-next catch resume): carry that same post-call
                //     catch `-live-` offset.
                //   * `None` — plain sequential residual call resuming at the
                //     next opcode's start marker: decode routes the plain word
                //     through the resume-translation, so carry
                //     the ordinary post-call marker.
                // Both offsets are the SAME physical post-call `-live-` insn by
                // codewriter construction (the plain fallthrough re-key and the
                // catch-predecessor anchor resolve to one marker), so the
                // carried value is identical to the decoder's fallback by
                // construction — encoder reg-bank window and decoder liveness
                // stay symmetric, and both pass `can_decode_live_vars` (a
                // `-live-` marker). Fall back to the sentinel on a map miss.
                let marker = unsafe {
                    let jc = &walked_owned;
                    // RPython `pyjitpl.py capture_resumedata(
                    // after_residual_call=True)` snapshots the `-live-`
                    // immediately following the residual call.  Read that
                    // structural marker first for every after-call guard.  In
                    // a try-block it sits before `catch_exception/L`, so its
                    // liveness includes both the normal continuation and the
                    // handler; a Python-PC fallthrough twin may instead name a
                    // later normal-path marker and lose handler-only locals.
                    after_residual_guard_marker(&jc.payload, op_pc, marker_call_jit_pc)
                };
                match marker {
                    Some(jp) => jp as i32,
                    None => majit_ir::resumedata::NO_JITCODE_PC,
                }
            } else {
                // The remaining nonbranch guards (outside the
                // arm-2 allow-list, not after-residual) carry the same
                // block-head marker as arm-2, sourced from the jitcode-keyed
                // twin at the guard's own `op_pc`.
                // A guard recorded before its opcode's `jit_merge_point` (the
                // `is_being_profiled` diamond, `promote_greens`, the merge
                // itself) has a block-head marker AFTER it: the op2 `-live-`
                // the opcode body resumes from.  Resuming there would skip the
                // merge point, so `bhimpl_jit_merge_point` would never hand
                // the opcode back to the interpreter.  Such a guard resumes at
                // the `-live-` before it, `generate_guard(resumepc=orgpc)`
                // (`pyjitpl.py`).
                let payload = unsafe { &(&walked_owned).payload };
                match payload.resume_marker_for_jitcode_pc(op_pc) {
                    Some(jp) if merge_point_between(payload, op_pc, jp) => {
                        live_marker_at_or_before(payload, op_pc).map_or(jp as i32, |m| m as i32)
                    }
                    Some(jp) => jp as i32,
                    None => majit_ir::resumedata::NO_JITCODE_PC,
                }
            };
            let liveness_py_pc = py_pc;
            let payload = &walked_owned.payload;
            let resolved = payload
                .resolve_resume_pc_with_jitcode_pc(guard_jitcode_pc, crate::state::op_live());
            let Some(resolved_offset) = resolved else {
                if let Some(saved) = saved_vable_scalars.take() {
                    crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved);
                }
                return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
            };
            // A residual result is installed in the active register bank before
            // the exception guard captures resume data
            // (`rpython/jit/metainterp/pyjitpl.py`).  Project every
            // live Ref register of the owning frame through the resume
            // position's color-to-slot map so the separately encoded
            // virtualizable array read by `virtualizable.py` carries the
            // same locals and operand-stack values as the frame section.  Inner
            // frames have no ownership of this shadow and are handled by the
            // multi-frame path above.
            if after_residual_call && sym.owns_virtualizable_shadow() {
                let maps = crate::state::bridge_semantic_maps_at_with_jitcode_pc(
                    jitcode_index as i32,
                    Some(liveness_py_pc as i32),
                    guard_jitcode_pc,
                );
                let banks = crate::state::frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
                    jitcode_index as i32,
                    guard_jitcode_pc,
                );
                let semantic_limit = sym.nlocals() + maps.stack_depth_at_pc;
                let nlocals = sym.nlocals();
                // The residual call's result lands on the operand stack, so
                // only stack slots need reconciling here.  A local's slot is
                // maintained in lockstep by the shadow, while the per-PC
                // color-to-slot map is a per-program-point label rather than a
                // binding: the register allocator can reuse a local's color
                // for unrelated SSA temps, so projecting a register into a
                // local's slot can publish an unrelated value.
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                for &(bank, color, slot) in &maps.pcdep_entries {
                    if bank != 1
                        || (slot as usize) < nlocals
                        || slot as usize >= semantic_limit
                        || !banks.ref_.contains(&(color as u32))
                    {
                        continue;
                    }
                    let Some(value) = ctx.registers_r.get(color as usize) else {
                        continue;
                    };
                    if value == OpRef::NONE || value.ty() != Some(majit_ir::Type::Ref) {
                        continue;
                    }
                    let vable_index = nvs + slot as usize;
                    if let Some((_, current)) = stack_sync
                        .iter_mut()
                        .find(|(index, _)| *index == vable_index)
                    {
                        *current = value;
                    } else {
                        stack_sync.push((vable_index, value));
                    }
                }
            }
            let saved_shadow: Vec<(usize, Option<OpRef>)> = stack_sync
                .iter()
                .map(|&(idx, _)| (idx, ctx.trace_ctx.virtualizable_box_at(idx)))
                .collect();
            for &(idx, value) in &stack_sync {
                ctx.trace_ctx.set_virtualizable_box_at(idx, value);
            }
            let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
            let vable_boxes = async_force_vable_boxes.unwrap_or(vable_boxes);
            if let Some(saved) = saved_vable_scalars.take() {
                crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved);
            }
            for (idx, old) in saved_shadow {
                if let Some(old) = old {
                    ctx.trace_ctx.set_virtualizable_box_at(idx, old);
                }
            }
            let (entry_jitcode_pc, entry_twin, entry_caller) = if let Some(live_pc) =
                scope.liveness_jit_pc
            {
                // Resume and liveness share this later `-live-`.  The opcode
                // floor would snapshot the landing, whose live set omits the
                // caught exception a PUSH_EXC_INFO prev-nullity guard must
                // carry (`pyjitpl.py` `get_list_of_active_boxes` at the
                // interpreter pc that already bound `last_exc_value`).
                (
                    live_pc as i32,
                    OuterActiveBoxesEntryTwin::Plain,
                    "guard_snapshot_carried_live",
                )
            } else if after_residual_call && guard_jitcode_pc != majit_ir::resumedata::NO_JITCODE_PC
            {
                // `get_list_of_active_boxes(after_residual_call=True)` reads
                // `registers_X[index]` at the trailing `-live-` (`self.pc`).
                // `make_result_of_lastop` / `do_residual_call` /
                // `execute_varargs` wrote the call result into that
                // register before `generate_guard(GUARD_NOT_FORCED)` and
                // `handle_possible_exception`. Key pcdep at that same
                // trailing `-live-` (banks already use `guard_jitcode_pc`)
                // so `collect_outer_active_boxes` prefers the just-written
                // walk register. The fallthrough opcode floor's color→slot
                // map can name a reused pre-call slot; the virtualizable
                // overlay then substitutes that slot's shadow and the
                // GUARD_NOT_FORCED failargs omit the call result.
                (
                    guard_jitcode_pc,
                    OuterActiveBoxesEntryTwin::Plain,
                    "guard_snapshot_after_residual",
                )
            } else {
                let entry = unsafe {
                    crate::py_coord::first_floor_boundary_for_py(
                        &walked_owned.payload.metadata,
                        liveness_py_pc,
                    )
                    .map(|(pc, _)| pc)
                };
                match entry {
                    Some(pc) => (
                        pc as i32,
                        OuterActiveBoxesEntryTwin::Plain,
                        "guard_snapshot_fallthrough",
                    ),
                    None => (
                        resolved_offset as i32,
                        OuterActiveBoxesEntryTwin::Trivia,
                        "guard_snapshot_fallthrough",
                    ),
                }
            };
            let active = collect_outer_active_boxes(
                sym,
                ctx.trace_ctx,
                ctx.registers_i,
                ctx.registers_r,
                ctx.registers_f,
                jitcode_index,
                guard_jitcode_pc,
                entry_jitcode_pc,
                entry_twin,
                entry_caller,
                None,
            );
            let pc_word = resolved_offset as u32;
            let forward_py_pc = forward_snapshot_py_pc(jitcode_index, pc_word)?;
            publish_single_frame_snapshot(
                ctx,
                scope.guard_stamp,
                &active,
                jitcode_index,
                pc_word,
                forward_py_pc,
                &vable_boxes,
                &vref_boxes,
                "fbw",
                op_pc,
            );
            return Ok(());
        }
    }

    // Per-opcode arm path: `op_pc` is arm-local, so its snapshot must use
    // the carried static outer JitCode coordinate; an absent coordinate
    // declines instead of reconstructing one from the Python pc.
    //
    // A collapsed helper or inlined callee carries the parent resume
    // coordinate on `outer_resume_marker_jit_pc`. On a loop-header opcode
    // that is the op2 `-live-` after `jit_merge_point` (`jtransform.py
    // handle_jit_marker__jit_merge_point`). Active boxes were collected
    // at that same pc.
    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    let arm_word = ctx
        .outer_resume_marker_jit_pc
        .map(|m| m as i32)
        .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
    let Some(arm_pc_word) =
        crate::state::pyjitcode_for_jitcode_index(ctx.outer_jitcode_index as i32)
            .and_then(|payload| {
                let resolved =
                    payload.resolve_resume_pc_with_jitcode_pc(arm_word, crate::state::op_live());
                resolved
            })
            .map(|offset| offset as u32)
    else {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    };
    let arm_py_pc = forward_snapshot_py_pc(ctx.outer_jitcode_index, arm_pc_word)?;
    let active = std::mem::take(&mut ctx.frame_state.borrow_mut().outer_active_boxes);
    publish_single_frame_snapshot(
        ctx,
        scope.guard_stamp,
        &active,
        ctx.outer_jitcode_index,
        arm_pc_word,
        arm_py_pc,
        &vable_boxes,
        &vref_boxes,
        "carried",
        op_pc,
    );
    ctx.frame_state.borrow_mut().outer_active_boxes = active;
    Ok(())
}

/// `PYRE_GUARD_RESUME_PC=1` prints the coordinate every walker-emitted guard
/// resumes at.  A guard whose printed `py_pc` is not the opcode it was emitted
/// under re-executes the wrong bytecode on deopt, which reads as a livelock or
/// a corrupted local rather than as a crash, so the coordinate is otherwise
/// invisible.
fn guard_resume_pc_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PYRE_GUARD_RESUME_PC").as_deref() == Ok("1"))
}

/// Attach a freshly built single-frame snapshot to the guard this capture is
/// for. `guard_stamp` picks the recorder-side target: the default stamps the
/// last recorded op (the guard IS that op), while a guard that records further
/// ops after itself — the `_nonstandard_virtualizable` promote, whose
/// `emit_force_virtualizable` tail records GETFIELD_GC / PTR_NE / COND_CALL —
/// or shares an opcode with another guard must stamp the corresponding guard
/// op instead.
#[allow(clippy::too_many_arguments)]
fn publish_single_frame_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    guard_stamp: GuardStampTarget,
    active: &[OpRef],
    jitcode_index: u32,
    pc: u32,
    py_pc: u32,
    vable_boxes: &[majit_metainterp::recorder::SnapshotTagged],
    vref_boxes: &[majit_metainterp::recorder::SnapshotTagged],
    origin: &str,
    op_pc: usize,
) {
    if guard_resume_pc_probe_enabled() {
        let opcode = match guard_stamp {
            GuardStampTarget::LastOp => ctx.trace_ctx.last_op_opcode(),
        };
        eprintln!(
            "[guard-pc] {origin} {opcode:?} op_pc={op_pc} jc={jitcode_index} \
             jit_pc={pc} py_pc={py_pc} active={}",
            active.len(),
        );
    }
    match guard_stamp {
        GuardStampTarget::LastOp => ctx
            .trace_ctx
            .capture_snapshot_for_last_guard_with_vable_vref(
                active,
                jitcode_index,
                pc,
                py_pc,
                vable_boxes,
                vref_boxes,
            ),
    }
}

/// Build the paused caller frame for a multi-frame inline snapshot (#68),
/// computed at the inline CALL site where the caller's live register banks
/// (`ctx.registers_*`) are still in scope.  `call_jit_pc` is the CALL op's
/// jitcode pc in the caller.  Returns a named decline reason
/// when the caller frame is not snapshot-able for this first slice: missing
/// liveness / resume tables, or no result on the operand stack at the return
/// point.
///
/// The caller resumes at the CALL's return point (fallthrough) with the
/// not-yet-produced call-result slot nulled — `get_list_of_active_boxes(
/// in_a_call=true)` parity (`trace_opcode.rs`). Reuses
/// [`collect_outer_active_boxes`] (the caller owns the portal virtualizable)
/// after temporarily nulling the result slot's register.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InlineCallerFrameDecline {
    Unavailable,
}

/// Name the `Unavailable` site for the `[fbw-census]` tally, then return it.
/// Report-only: gated on `fbw_debug_abort_enabled`, so the cost on a build
/// without the flag is one predictable branch on an already-cold path.
fn unavail(site: &'static str) -> InlineCallerFrameDecline {
    if fbw_debug_abort_enabled() {
        crate::jitcode_dispatch::census_record(site);
    }
    InlineCallerFrameDecline::Unavailable
}

pub(crate) fn concrete_ref_for_color<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    color: usize,
) -> Option<pyre_object::PyObjectRef> {
    let paused = paused_python_live(ctx);
    let paused_concrete = paused
        .as_ref()
        .map(|live| live.frame_state().borrow().concrete_registers_r.clone());
    let paused_regs = paused.as_ref().map(|live| live.registers_r.clone());
    let concrete = match &paused_concrete {
        Some(concretes) => concretes.get(color).copied(),
        None => ctx
            .frame_state
            .borrow()
            .concrete_registers_r
            .get(color)
            .copied(),
    };
    if let Some(ConcreteValue::Ref(ptr)) = concrete {
        if !ptr.is_null() {
            return Some(ptr);
        }
    }
    let opref = match &paused_regs {
        Some(regs) => regs.get(color)?,
        None => ctx.registers_r.get(color)?,
    };
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Ref(r)) if !r.is_null() => Some(r.as_usize() as pyre_object::PyObjectRef),
        _ => None,
    }
}

/// Resolve a recorded box to the concrete ref it stands for.
///
/// `GcRef::NO_CONCRETE` is the sentinel for "no runtime value is known for
/// this box" (`value.rs:51-59`), and `heapcache_ops` stamps it on a box whose
/// load could not be replayed, so it never answers a slot.  A stamped NULL is
/// judged by what carries it:
///
/// * A box [`null_ref_is_a_value`] accepts carries the operand's own null.
///   Answering unresolved for one left every paused-caller image built over
///   that stack one slot short and `capture_root_parent_resume_stack`
///   declined it.
/// * Any other box may read back `Ref(0)` while the walk still holds the
///   operand symbolically — a deferred `LOAD_ATTR name + NULL|self` pair is
///   the shape that reaches here.  Answer unresolved, as
///   [`concrete_ref_for_color`] and the virtualizable-shadow fallback in
///   [`collect_call_stack_overrides`] already do, so the slot stays ABSENT and
///   the outer-call flush declines to the legacy replay.  Publishing it
///   instead writes a null into a live operand-stack slot, and the interpreter
///   resumed there dispatches through it: `out.append(f(i))` in an `except`
///   handler faulted in `classify_callable` on the method-load slot this used
///   to hand back as a resolved value.
///
/// The one operand whose correct value IS null whatever carries it — a
/// `CALL`'s own `null_or_self` — is named and supplied separately by
/// [`collect_call_stack_overrides`].
/// Whether a NULL carried by `opref` is the operand's VALUE rather than a slot
/// whose value is simply unknown.
///
/// An input argument is bound from the real frame when the loop is entered and
/// rebound from the fail args at every guard failure, so a NULL stamped on one
/// is what that operand holds — RPython's `MIFrame.registers_r` entry for a
/// loop input box holds the same null.  `LOAD_FAST_AND_CLEAR` saves an unbound
/// local as exactly that null and leaves it on the operand stack below an
/// inlined comprehension for the whole loop, so a merge point inside the
/// comprehension closes over one.
///
/// Any other box may read back `Ref(0)` while the walk still holds the operand
/// symbolically, which is a different fact and answers false here.
pub(crate) fn null_ref_is_a_value(opref: OpRef) -> bool {
    matches!(opref, OpRef::InputArgRef(_))
}

pub(crate) fn concrete_ref_for_opref<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    opref: OpRef,
) -> Option<pyre_object::PyObjectRef> {
    // RPython history.py defines CONST_NULL as a real
    // `ConstPtr(lltype.nullptr(llmemory.GCREF.TO))`.  Preserve that typed
    // constant before consulting the runtime-concrete table: an inline
    // ConstPtr(NULL) is positive proof that this operand-stack slot contains
    // Python's call sentinel.  LOAD_SPECIAL records exactly that constant for
    // its `self_or_null` half, including the `__exit__` pair retained below a
    // nested CALL inside a `with` body.
    if let OpRef::ConstPtr(index) = opref {
        return Some(
            majit_ir::const_ptr_table::resolve(index).as_usize() as pyre_object::PyObjectRef
        );
    }
    let null_is_a_value = null_ref_is_a_value(opref);
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Ref(r))
            if r != majit_ir::GcRef::NO_CONCRETE && (null_is_a_value || !r.is_null()) =>
        {
            Some(r.as_usize() as pyre_object::PyObjectRef)
        }
        _ => None,
    }
}

pub(crate) fn collect_call_stack_overrides<Sym: WalkSym>(
    caller_sym: &Sym,
    ctx: &WalkContext<'_, '_, Sym>,
    call_jitcode_pc: usize,
) -> Option<Vec<(usize, pyre_object::PyObjectRef)>> {
    if caller_sym.jitcode().is_null() {
        return None;
    }
    let (nlocals, depth, pcdep_entries, const_ref_slots) = unsafe {
        let jc = &*caller_sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            return None;
        }
        // The caller CALL key is a plain JitCode→Python inversion, so use the
        // certified predecessor twins rather than the marker/trivia flavor
        // reserved for the CALL fallthrough below.
        let depth = jc
            .payload
            .depth_for_jitcode_pc_pred(call_jitcode_pc)
            .unwrap_or(0) as usize;
        let pcdep = jc
            .payload
            .pcdep_for_jitcode_pc(call_jitcode_pc)
            .unwrap_or_default();
        // Same predecessor coordinate as `pcdep_for_jitcode_pc`.
        // `pcdep_slot_var_resume` records Variables only; `const_ref_slots_at_pc`
        // (via `resolve_const_ref_slot`) holds the Ref constants beside it.
        let consts = jc
            .payload
            .const_ref_slots_for_jitcode_pc(call_jitcode_pc)
            .unwrap_or_default();
        (caller_sym.nlocals(), depth, pcdep, consts)
    };
    match assemble_call_stack_overrides(
        caller_sym,
        ctx,
        call_jitcode_pc,
        nlocals,
        depth,
        &pcdep_entries,
        &const_ref_slots,
    ) {
        Ok(overrides) => return Some(overrides),
        Err(AssembleFail::NoShape) => {
            if fbw_debug_abort_enabled() {
                crate::jitcode_dispatch::census_record("CallStack::NoOperandShape");
            }
            return None;
        }
        Err(AssembleFail::Proof(gap)) => {
            if fbw_debug_abort_enabled() {
                crate::jitcode_dispatch::census_record("CallStack::ProofSlotUnresolved");
                let depth = gap.depth;
                let stack_end = gap.stack_end;
                let proof_slot = gap.proof_slot;
                let vstack_valid = ctx.vstack_valid;
                let vstack_depth = ctx.vstack_depth;
                let pcdep_color = pcdep_entries.iter().find_map(|&(bank, color, slot)| {
                    (bank == 1 && slot as usize == proof_slot).then_some(color)
                });
                let reg = pcdep_color.and_then(|color| ctx.registers_r.get(color as usize));
                let const_has_proof = const_ref_slots
                    .iter()
                    .any(|&(slot, raw)| slot as usize == proof_slot && raw != 0);
                let const_n = const_ref_slots.len();
                let pcdep_n = pcdep_entries.len();
                let stack_pcdep: Vec<(u8, u16, u16)> = pcdep_entries
                    .iter()
                    .copied()
                    .filter(|&(bank, _, slot)| {
                        bank == 1 && (slot as usize) >= nlocals && (slot as usize) < stack_end
                    })
                    .collect();
                let vstack_box = (vstack_valid && vstack_depth == depth && depth > 0).then(|| {
                    ctx.frame_state
                        .borrow()
                        .vstack_boxes
                        .get(depth - 1)
                        .copied()
                });
                eprintln!(
                    "[call-overrides-proof] pc={call_jitcode_pc} nlocals={nlocals} \
                     depth={depth} stack_end={stack_end} proof_slot={proof_slot} \
                     vstack_valid={vstack_valid} vstack_depth={vstack_depth} \
                     vstack_box={vstack_box:?} pcdep_color={pcdep_color:?} reg={reg:?} \
                     const_has_proof={const_has_proof} const_n={const_n} pcdep_n={pcdep_n} \
                     stack_pcdep={stack_pcdep:?}"
                );
            }
            None
        }
    }
}

enum AssembleFail {
    NoShape,
    Proof(ProofGap),
}

struct ProofGap {
    depth: usize,
    stack_end: usize,
    proof_slot: usize,
}

/// Record `value` for `slot`.
///
/// A null already stored is the hole `emit_popvalue_ref` leaves in the vstack
/// mirror while `pcdep_slot_var_resume` / `const_ref_slots_at_pc` still name
/// the opcode-entry owner. With `replace_null`, that owner replaces the hole.
/// A resolved object stays, and a null never replaces one: the virtualizable
/// shadow must not overwrite `PUSH_NULL`'s sentinel.
fn note_stack_override(
    overrides: &mut Vec<(usize, pyre_object::PyObjectRef)>,
    slot: usize,
    value: pyre_object::PyObjectRef,
    replace_null: bool,
) {
    if let Some(existing) = overrides.iter_mut().find(|(present, _)| *present == slot) {
        if replace_null && existing.1.is_null() && !value.is_null() {
            existing.1 = value;
        }
        return;
    }
    overrides.push((slot, value));
}

fn assemble_call_stack_overrides<Sym: WalkSym>(
    caller_sym: &Sym,
    ctx: &WalkContext<'_, '_, Sym>,
    call_jitcode_pc: usize,
    nlocals: usize,
    depth: usize,
    pcdep_entries: &[(u8, u16, u16)],
    const_ref_slots: &[(u16, i64)],
) -> Result<Vec<(usize, pyre_object::PyObjectRef)>, AssembleFail> {
    let stack_end = nlocals + depth;
    if depth == 0 {
        return Ok(Vec::new());
    }
    let mut overrides = Vec::new();
    if ctx.vstack_valid
        && ctx.vstack_depth == depth
        && ctx.frame_state.borrow().vstack_boxes.len() >= depth
    {
        for d in 0..depth {
            let slot = nlocals + d;
            if let Some(value) =
                concrete_ref_for_opref(ctx, ctx.frame_state.borrow().vstack_boxes[d])
            {
                overrides.push((slot, value));
            }
        }
    }
    if pcdep_entries.is_empty() {
        for slot in nlocals..stack_end {
            if overrides.iter().any(|&(present, _)| present == slot) {
                continue;
            }
            if let Some(value) = concrete_ref_for_color(ctx, slot) {
                overrides.push((slot, value));
            }
        }
    } else {
        for &(bank, color, slot) in pcdep_entries {
            let slot = slot as usize;
            if bank == 1 && slot >= nlocals && slot < stack_end {
                if let Some(value) = concrete_ref_for_color(ctx, color as usize) {
                    // The vstack mirror can already hold the NULL
                    // `emit_popvalue_ref` wrote for this slot. That hole is
                    // not the opcode-entry owner `pcdep_slot_var_resume`
                    // recorded, and `CallStack::ProofSlotUnresolved` declines
                    // `load_attr_fn` on `Attrs.cm(i)` when the hole wins.
                    note_stack_override(&mut overrides, slot, value, true);
                }
            }
        }
    }
    // Use the virtualizable shadow only after the live register/vstack
    // sources, and only for slots those sources left unresolved.  A mid-opcode
    // shadow stack slot can be NULL because it was not mirrored, while the
    // corresponding live color still holds the value — so a NULL reaching this
    // fallback is an UNRESOLVED slot, not a proven null (see the per-slot note
    // below), and is dropped rather than committed.
    let base = ctx
        .trace_ctx
        .virtualizable_info()
        .map(|info| info.num_static_extra_boxes)
        .unwrap_or(0);
    for slot in nlocals..stack_end {
        if overrides.iter().any(|&(present, _)| present == slot) {
            continue;
        }
        if let Some((_opref, Value::Ref(value))) = ctx.trace_ctx.virtualizable_entry_at(base + slot)
        {
            // Only a positively-resolved (non-null) shadow value is a faithful
            // stack slot.  The live vstack/color sources above emit a genuine
            // null-or-self sentinel only where they hold a box for the slot at
            // all — the `PUSH_NULL` position has none, which is what the
            // CALL operand resolution below exists to cover.  A slot that
            // reaches this shadow fallback with a NULL Ref is one the walk could
            // not resolve — e.g. an
            // unmaterialized `LOAD_CONST` operand whose concrete value was
            // never mirrored — not a real null.  Leaving it ABSENT makes the
            // outer-call flush validation decline, so the legacy replay
            // rebuilds the frame from its start state instead of resuming the
            // interpreter over a NULL where a live object belongs.
            if value.as_usize() != 0 {
                overrides.push((slot, value.as_usize() as pyre_object::PyObjectRef));
            }
        }
    }
    // `pcdep_slot_var_resume` records Variables only. A Ref constant on the
    // entry stack is in `const_ref_slots_at_pc` (`resolve_const_ref_slot`).
    // A null hole already stored for the slot is the same mid-opcode clear
    // as above; the entry constant replaces it. A zero raw is the cleared
    // slot, not a value.
    for &(slot, raw) in const_ref_slots {
        let slot = slot as usize;
        if slot < nlocals || slot >= stack_end || raw == 0 {
            continue;
        }
        note_stack_override(&mut overrides, slot, raw as pyre_object::PyObjectRef, true);
    }
    // A CALL's `[callable, null_or_self, arg0 .. arg_{argc-1}]` operands end at
    // `stack_end`.  The null_or_self slot is the one stack slot no source above
    // can speak for, so synthesize its known null and require the callable
    // immediately below it to prove that reconstruction reached the operand
    // region.  Every other resume shape synthesizes nothing and proves itself
    // with the deepest operand it consumes.
    let Some(operand_slots) = caller_operand_slots(caller_sym, call_jitcode_pc, stack_end) else {
        return Err(AssembleFail::NoShape);
    };
    let (sentinel_slots, proof_slot) = match operand_slots {
        CallerOperandSlots::Call {
            null_or_self,
            callable,
        } => ([Some(null_or_self), None], callable),
        // `self_or_null` and `kwargs_or_null` are the checked null sentinels.
        // A resolved receiver or mapping already in `overrides` stays; only a
        // slot no source named is filled with null.
        CallerOperandSlots::CallFunctionEx {
            self_or_null,
            kwargs,
            callable,
        } => ([Some(self_or_null), Some(kwargs)], callable),
        CallerOperandSlots::Operands { deepest } => ([None, None], deepest),
    };
    for sentinel_slot in sentinel_slots.into_iter().flatten() {
        if sentinel_slot >= nlocals
            && !overrides
                .iter()
                .any(|&(present, _)| present == sentinel_slot)
        {
            overrides.push((
                sentinel_slot,
                std::ptr::null_mut::<u8>() as pyre_object::PyObjectRef,
            ));
        }
    }
    // Ref(0) is valid only for the synthesized null_or_self sentinel.  A null
    // proof value means the sparse vstack/color reconstruction is incomplete;
    // decline instead of publishing an invalid operand region.
    let proof_value = overrides
        .iter()
        .find_map(|&(slot, value)| (slot == proof_slot).then_some(value));
    // A missing callable stays unresolved. `outer_active_boxes` is ordered by
    // liveness color, so its first non-null entry is not the CALL's callable.
    if !matches!(proof_value, Some(value) if !value.is_null()) {
        return Err(AssembleFail::Proof(ProofGap {
            depth,
            stack_end,
            proof_slot,
        }));
    }
    // Report-only: a slot left absent here is what makes the outer-call flush
    // decline, and the consumer can only say "not capturable". Name each source
    // that passed on it so the gap is attributable to one of them.
    if fbw_debug_abort_enabled() {
        for slot in nlocals..stack_end {
            if overrides.iter().any(|&(present, _)| present == slot) {
                continue;
            }
            let vstack_box = (ctx.vstack_valid
                && ctx.vstack_depth == depth
                && ctx.frame_state.borrow().vstack_boxes.len() >= depth)
                .then(|| ctx.frame_state.borrow().vstack_boxes[slot - nlocals]);
            let vstack_concrete =
                vstack_box.map(|opref| concrete_ref_for_opref(ctx, opref).is_some());
            let pcdep_color = pcdep_entries
                .iter()
                .find_map(|&(bank, color, s)| (bank == 1 && s as usize == slot).then_some(color));
            let shadow = ctx.trace_ctx.virtualizable_entry_at(base + slot).map(
                |(_opref, value)| match value {
                    Value::Ref(r) => r.as_usize(),
                    _ => usize::MAX,
                },
            );
            eprintln!(
                "[call-overrides-absent] slot={slot} nlocals={nlocals} depth={depth} \
                 vstack_box={vstack_box:?} vstack_concrete={vstack_concrete:?} \
                 pcdep_color={pcdep_color:?} shadow={shadow:?} \
                 proof_slot={proof_slot}"
            );
        }
    }
    Ok(overrides)
}

enum CallerOperandSlots {
    Call {
        null_or_self: usize,
        callable: usize,
    },
    /// `call_function_ex` pops `kwargs_or_null`, `starargs`, `self_or_null`,
    /// `callable`. The two null-capable slots are synthesized when unresolved;
    /// `starargs` is an ordinary operand and is not.
    CallFunctionEx {
        self_or_null: usize,
        kwargs: usize,
        callable: usize,
    },
    /// A shape with no slot to synthesize: the deepest of the operands it
    /// consumes is the proof slot.
    Operands { deepest: usize },
}

/// Absolute frame slots proving the operand region at `call_jitcode_pc`, for a
/// caller whose operand stack ends at `stack_end`. The CALL forms name
/// their synthetic `null_or_self` sentinel and callable proof; every other
/// shape names only how many operands it consumes, and the deepest of those
/// plays the role the callable plays for CALL. `None` keeps the conservative
/// decline for every resume shape not listed.
///
/// The non-CALL shapes are what an inlined dunder needs: the entry admits a
/// body on the strength of an abort resuming forward past it, and with no arm
/// here that reconstruction answered `None`, so the resume fell back to
/// replaying the loop from its entry. `LOAD_ATTR` / `STORE_ATTR` are the
/// attribute-access half of that — a property accessor, a `__getattr__` hook
/// or a descriptor `__get__` body is reached from one of them, never from a
/// Python-level CALL.  `FORMAT_WITH_SPEC` is the same half for `__format__`:
/// the walker route that inlines a user `__format__` body
/// (`try_walker_inline_format`) reaches this with the f-string opcode as the
/// caller instruction, and with no arm here every one of those bodies passed
/// each of that route's own screens and then lost the caller image, so the
/// body stayed a residual call.  FORMAT_SIMPLE has no such route yet (its
/// residual carries no spec operand to hand the two-parameter dunder), so it
/// is not listed.  `POP_JUMP_IF_FALSE` / `POP_JUMP_IF_TRUE` are the truth-test
/// half: `opcode_pop_jump_if` pops one value and calls `truth_value`, and
/// `try_walker_inline_truth_bool` inlines an overriding `__bool__` from that
/// residual.  With no arm `CallStack::NoOperandShape` left the body a residual
/// `Truth`.
///
/// CALL_KW is here for the opposite reason: it IS a Python-level call, and
/// without an arm the seeded inline had no caller image for one, so every
/// keyword call stayed a residual `CallMayForce` with its arguments built on
/// the heap once per execution. `CALL_FUNCTION_EX` is the same call: its
/// entry stack is `[callable, self_or_null, starargs, kwargs_or_null]`
/// (`call_function_ex`), and with no arm `CallStack::NoOperandShape` left
/// `add(*args)` a residual `bh_call_function_ex_fn`.
fn caller_operand_slots<Sym: WalkSym>(
    caller_sym: &Sym,
    call_jitcode_pc: usize,
    stack_end: usize,
) -> Option<CallerOperandSlots> {
    let jc = unsafe { caller_sym.jitcode().as_ref()? };
    let code = unsafe { (jc.payload.code_ptr as *const pyre_interpreter::CodeObject).as_ref()? };
    // The floor pivot names the block head. A helper entered from a CALL
    // whose jit pc still sits in that block must use the exact owning
    // instruction, or `LOAD_GLOBAL` is what the operand shape sees.
    let py_pc = crate::pyjitcode::exact_py_pc_for_jitcode_pc(
        &jc.payload.metadata.py_exact_by_jit_pc,
        call_jitcode_pc,
    )
    .map(|pc| pc as usize)
    .unwrap_or_else(|| {
        crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, call_jitcode_pc)
            as usize
    });
    let (instruction, op_arg) = pyre_interpreter::decode_instruction_at(code, py_pc)?;
    // `[callable, null_or_self, arg0 .. arg_{argc-1}]` for CALL, and the
    // keyword-name tuple one slot above that block for CALL_KW: `call_kw`
    // (`eval.rs`) pops the names first and then the identical argument block,
    // so the two shapes differ only in where the stack ends above the
    // arguments, and both name the same synthetic sentinel and callable proof.
    let call_shape = match instruction {
        pyre_interpreter::Instruction::Call { argc } => Some((argc.get(op_arg) as usize, 1usize)),
        pyre_interpreter::Instruction::CallKw { argc } => Some((argc.get(op_arg) as usize, 2usize)),
        _ => None,
    };
    if let Some((argc, slots_above_args)) = call_shape {
        let null_or_self = stack_end.checked_sub(argc + slots_above_args)?;
        return Some(CallerOperandSlots::Call {
            null_or_self,
            callable: null_or_self.checked_sub(1)?,
        });
    }
    if matches!(instruction, pyre_interpreter::Instruction::CallFunctionEx) {
        // Entry order, bottom to top: callable, self_or_null, starargs,
        // kwargs_or_null. `stack_end` is one past kwargs.
        let kwargs = stack_end.checked_sub(1)?;
        let self_or_null = stack_end.checked_sub(3)?;
        let callable = self_or_null.checked_sub(1)?;
        return Some(CallerOperandSlots::CallFunctionEx {
            self_or_null,
            kwargs,
            callable,
        });
    }
    let operand_count = match instruction {
        // `[iterable]` for GET_ITER, `[iterator]` for FOR_ITER, and `[owner]`
        // for the attribute read whose descriptor body is the callee.
        // `stack_end` is `nlocals + depth_at_py_pc` (`depth_for_jitcode_pc_pred`),
        // the depth on entry to the opcode. LOAD_ATTR consumes that one owner
        // either way. The method form's extra `self_or_null` is a result
        // (`load_method` pushes it after the attribute, net +1 in
        // `stack_effects`), not an input. Counting it named the local below
        // the owner, and `collect_call_stack_overrides` declined
        // (`CallStack::ProofSlotUnresolved`) on `Attrs.cm(i)`.
        pyre_interpreter::Instruction::GetIter { .. } => 1,
        pyre_interpreter::Instruction::ForIter { .. } => 2,
        pyre_interpreter::Instruction::LoadAttr { .. } => 1,
        // `[value]`. `opcode_pop_jump_if` pops that one value and then calls
        // `truth_value`. `stack_end` is the depth on entry, so the value is
        // `stack_end - 1`. A second slot named the local below it and
        // `collect_call_stack_overrides` declined
        // (`CallStack::ProofSlotUnresolved`).
        pyre_interpreter::Instruction::PopJumpIfFalse { .. }
        | pyre_interpreter::Instruction::PopJumpIfTrue { .. } => 1,
        // `[lhs, rhs]`, and `[value, owner]` for the attribute store whose
        // setter body is the callee — `store_attr_cached` pops the owner
        // first, so `value` is the deeper of the two.
        //
        // `FORMAT_WITH_SPEC` has the same shape: `format_with_spec`
        // (`eval.rs`) pops the spec and then the value, so `value` — the
        // receiver whose `__format__` is dispatched — is the deeper one.
        pyre_interpreter::Instruction::BinaryOp { .. }
        | pyre_interpreter::Instruction::CompareOp { .. }
        | pyre_interpreter::Instruction::StoreAttr { .. }
        | pyre_interpreter::Instruction::FormatWithSpec => 2,
        // `[obj, start, stop]`. `binary_slice` pops stop, start, then obj, so
        // `obj` is the deepest of the three. `try_walker_orthodox_binary_slice`
        // inlines each bound's `__index__` from this residual; with no arm
        // `CallStack::NoOperandShape` left the body a residual `BinarySlice`.
        pyre_interpreter::Instruction::BinarySlice => 3,
        other => {
            if fbw_debug_abort_enabled() {
                let jit_op = crate::jitcode_runtime::decode_op_at(
                    jc.payload.jitcode.code.as_slice(),
                    call_jitcode_pc,
                )
                .map(|op| op.key)
                .unwrap_or("");
                eprintln!(
                    "[caller-operand-shape-absent] instruction={other:?} py_pc={py_pc} jit_pc={call_jitcode_pc} jit_op={jit_op}"
                );
            }
            return None;
        }
    };
    Some(CallerOperandSlots::Operands {
        deepest: stack_end.checked_sub(operand_count)?,
    })
}

/// Name the live color that refused a caller image.
///
/// The liveness pass below DEMANDS a concrete for every color live at the
/// resume pc and answers `None` for the whole image when one is missing.  That
/// `None` surfaces far downstream as `[s2-build-decline] ... parent.blackhole
/// None (capture missing)`, which cannot say which bank, which color, or
/// whether the shadow was merely too short — and which is printed only when a
/// multi-frame build was actually attempted, so counting it undercounts the
/// refusals tenfold.  Reporting at the refusal itself, a sweep of 524 fixtures
/// found 21 declines across 10 fixtures, every one of them a ref color whose
/// shadow held the untracked `Null` sentinel.  Gated like the rest of the
/// walk's build reporting.
fn report_caller_image_decline<T: std::fmt::Debug>(
    jitcode_index: u32,
    call_jit_pc: usize,
    bank: char,
    color: usize,
    shadow_len: usize,
    got: Option<T>,
) {
    if !fbw_debug_abort_enabled() {
        return;
    }
    let why = if color >= shadow_len {
        "color past the end of the walk's shadow"
    } else {
        "shadow holds a different concrete kind"
    };
    eprintln!(
        "[fbw-blackhole] caller image DECLINED jitcode_index={jitcode_index} \
         call_jit_pc={call_jit_pc} bank={bank} live_color={color} \
         shadow_len={shadow_len} ({why}) got={got:?}"
    );
}

/// Ref-bank addendum to [`report_caller_image_decline`]: name the box the
/// declining color holds and whether the recorded producer could still supply
/// its value.
///
/// The innermost-frame fill (`residual_call.rs`, `build_single_frame_miframe`'s
/// `live.ref_` loop) meets the same situation and does not decline on it. It
/// carries two refinements this site lacks: a live color whose register holds
/// no box at all is SKIPPED, because a `-live-` set is the union over the paths
/// INTO its coordinate and the walk took one of them; and a color whose shadow
/// is the untracked `Null` sentinel is recovered through
/// `TraceCtx::recover_ref_value`, which replays a `GetfieldGcR` producer chain.
///
/// Which of the two would have applied — or neither — is what separates a
/// decline that is a missing refinement from one where the value is genuinely
/// lost, so print it rather than infer it.
fn report_caller_image_ref_box<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>, color: usize) {
    if !fbw_debug_abort_enabled() {
        return;
    }
    let opref = ctx.registers_r.get(color);
    let box_is_none = opref.map(|o| o.is_none()).unwrap_or(true);
    let recovered = opref
        .filter(|o| !o.is_none())
        .and_then(|o| ctx.trace_ctx.recover_ref_value(o, 8))
        .is_some();
    eprintln!(
        "[fbw-blackhole] caller image ref detail live_color={color} box={opref:?} \
         box_is_none={box_is_none} recoverable={recovered}"
    );
}

/// Python `MIFrame` paused under a transparent helper. The helper's own
/// banks are a different jitcode; `framestack[-1].live` is the caller
/// (`pyjitpl.py` keeps that frame while the looked-inside body runs).
fn paused_python_live<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<crate::jitcode_dispatch::LiveFrameRegs> {
    if !ctx.fbw_mode.transparent_helper_subwalk {
        return None;
    }
    ctx.session
        .borrow()
        .framestack
        .last()
        .and_then(|frame| frame.paused_live())
}

/// Re-read a paused caller's blackhole image from its rooted register shadow.
///
/// `compute_inline_caller_frame` copies `getref_base` (`blackhole.py`
/// `_copy_data_from_miframe` reads `MIFrame.registers_r`) into a Rust
/// `Vec`. Seeding the callee then records ops (`history.py` `record` →
/// `_record_op`) and a minor can run before the image is a
/// `walk_session_roots` root. The collector updates the box and
/// `walk_frame_state_roots` updates `concrete_registers_r`; it does not
/// update that `Vec`. Publishing the copy writes the pre-move address.
pub(in crate::jitcode_dispatch) fn refresh_paused_parent_blackhole(parent: &mut InlineParentFrame) {
    let Some(state) = parent.frame_state.clone() else {
        return;
    };
    let data = state.borrow();
    let Some(blackhole) = parent.blackhole.as_mut() else {
        return;
    };
    for (color, value) in &mut blackhole.ref_values {
        if let Some(ConcreteValue::Ref(live)) = data.concrete_registers_r.get(*color).copied()
            && !live.is_null()
        {
            *value = live;
        }
    }
}

/// Re-read the paused caller's concrete image immediately before
/// [`super::InlineFrameGuard`] publishes it. See
/// [`refresh_paused_parent_blackhole`].
pub(in crate::jitcode_dispatch) fn refresh_paused_parent_concretes<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    parent: &mut InlineParentFrame,
) {
    refresh_paused_parent_blackhole(parent);
    // Nested callers leave `call_stack_overrides` empty: their operand
    // stack is reconstructed from the sym-less banks, and filling it here
    // would publish slots the original capture deliberately omitted.
    // A top-level caller has no inlined frame on the session yet.
    if ctx.session.borrow().last_inline().is_some() {
        return;
    }
    let Some(call_pc) = parent.call_jitcode_pc else {
        return;
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return;
    }
    let caller_sym = unsafe { &*sym_ptr };
    if let Some(overrides) = collect_call_stack_overrides(caller_sym, ctx, call_pc) {
        parent.call_stack_overrides = overrides;
    }
}

/// Re-read every parent image a helper level holds immediately before it
/// is published. Guards and array stores between `helper_subwalk_level`
/// and `enter_helper_level` can minor-collect, and the level is not a
/// `walk_session_roots` root until then.
pub(in crate::jitcode_dispatch) fn refresh_helper_level<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    level: &mut HelperLevel,
) {
    for parent in &mut level.parents {
        refresh_paused_parent_concretes(ctx, parent);
    }
    if let HelperCallEntry::Frame(entry) = &mut level.call_entry {
        refresh_paused_parent_concretes(ctx, entry);
    }
}

/// Forwarded addresses only: `walk_session_roots` already traces a
/// pending level, and `push_frame` has no `WalkContext`.
pub(in crate::jitcode_dispatch) fn refresh_helper_level_blackhole(level: &mut HelperLevel) {
    for parent in &mut level.parents {
        refresh_paused_parent_blackhole(parent);
    }
    if let HelperCallEntry::Frame(entry) = &mut level.call_entry {
        refresh_paused_parent_blackhole(entry);
    }
}

fn capture_inline_parent_blackhole<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode_index: u32,
    call_jit_pc: usize,
) -> Option<InlineParentBlackhole> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)?;
    let call = decode_op_at(&pjc.jitcode.code, call_jit_pc)?;
    let result_bank = call.result_argcode();
    let result_color = (result_bank != 'v')
        .then(|| pjc.jitcode.code.get(call.next_pc.checked_sub(1)?).copied())
        .flatten()
        .map(usize::from);
    let live = crate::state::frame_liveness_reg_indices_by_bank_at(
        jitcode_index as i32,
        i32::try_from(call.next_pc).ok()?,
    );
    // `_copy_data_from_miframe` (`blackhole.py`) copies the WHOLE
    // bank — `range(num_regs_i/r/f())` — filtering only on "the MIFrame has a
    // box here", never on liveness.  Each bank below therefore runs twice: the
    // liveness pass, and then a sweep that seeds every remaining color the walk
    // happens to have a concrete for.
    //
    // The liveness pass DEMANDS a concrete for every color live at `resume_pc`
    // that HAS a box, and refuses the whole image when it cannot produce one.
    // Upstream's bank is allocated to `num_regs_*()` and its only skip is a
    // `None` / `MissingValue` box, so a color whose box is genuinely absent is
    // the one case that may go unset here too: a `-live-` set is the union over
    // the paths INTO its coordinate and this walk took one of them.  Everything
    // else — a color past the end of the walk's shorter bank, or a real box
    // whose concrete is the untracked sentinel or the wrong type — is a bank
    // this walk cannot answer for, and the sweep below re-tests the same
    // condition, so skipping would leave a live color unset and hand the
    // interpreter a NULL in a live slot.
    //
    // Seeding only the live subset is unsound the moment the drive leaves the
    // straight line: the blackhole runs on to a `jit_merge_point` whose own
    // live set is generally LARGER (a loop-carried value defined in the
    // prologue and not rewritten in the body is dead at a mid-body pc but live
    // at the header).  Those colors read back NULL, and a NULL in a live
    // operand-stack slot faults the interpreter.  The innermost frame already
    // gets this treatment; a caller frame is no different.
    //
    // The sweep cannot introduce a stale value: a color the drive rewrites is
    // overwritten before any read, and one it does not rewrite still holds what
    // the walk observed here, which is what a merge reached without an
    // intervening definition expects.  Colors with no concrete stay unset,
    // exactly as an absent upstream box does.  Nor can a swept color turn a
    // usable image into a declined one: `num_regs_*()` is upstream's own bound
    // and never exceeds the `num_regs_and_consts_*()` bank the replay indexes,
    // so the replay's out-of-range refusal stays out of reach.
    // How many colors each sweep adds beyond the live set is the one number
    // that says whether the caller banks were actually sparse here.  Report-only,
    // gated on `fbw_debug_abort_enabled` like the rest of the walk's build
    // reporting, so a build without the flag pays three counter bumps and one
    // predictable branch.
    let mut swept = [0usize; 3];
    let paused = paused_python_live(ctx);
    let paused_regs_i = paused.as_ref().map(|live| live.registers_i.clone());
    let paused_regs_r = paused.as_ref().map(|live| live.registers_r.clone());
    let paused_regs_f = paused.as_ref().map(|live| live.registers_f.clone());
    let paused_concrete_r = paused
        .as_ref()
        .map(|live| live.frame_state().borrow().concrete_registers_r.clone());
    let registers_i = paused_regs_i.as_ref().unwrap_or(ctx.registers_i);
    let registers_r = paused_regs_r.as_ref().unwrap_or(ctx.registers_r);
    let registers_f = paused_regs_f.as_ref().unwrap_or(ctx.registers_f);
    let mut int_values = Vec::with_capacity(pjc.jitcode.num_regs_i());
    let mut int_seeded = vec![false; pjc.jitcode.num_regs_i()];
    for &color in &live.int {
        let color = color as usize;
        if result_bank == 'i' && result_color == Some(color) {
            continue;
        }
        let got = if paused_regs_i.is_some() {
            registers_i
                .get(color)
                .and_then(|op| match ctx.trace_ctx.concrete_of_opref(op) {
                    Some(majit_ir::Value::Int(value)) => Some(ConcreteValue::Int(value)),
                    _ => None,
                })
        } else {
            ctx.concrete_registers_i.get(color).copied()
        };
        let value = match got {
            Some(ConcreteValue::Int(value)) => value,
            // No box at this color: `_copy_data_from_miframe` leaves the
            // register unset rather than refusing the image.
            _ if registers_i.get(color) == Some(OpRef::NONE) => continue,
            // A real box (or a color past the walk's shorter bank) the shadow
            // cannot answer for.  The sweep re-tests the same condition, so
            // skipping would leave this live color NULL.
            _ => {
                report_caller_image_decline(
                    jitcode_index,
                    call_jit_pc,
                    'i',
                    color,
                    if paused_regs_i.is_some() {
                        registers_i.len()
                    } else {
                        ctx.concrete_registers_i.len()
                    },
                    got,
                );
                return None;
            }
        };
        int_values.push((color, value));
        if let Some(seeded) = int_seeded.get_mut(color) {
            *seeded = true;
        }
    }
    for color in 0..pjc.jitcode.num_regs_i() {
        if int_seeded[color] || (result_bank == 'i' && result_color == Some(color)) {
            continue;
        }
        let concrete = if paused_regs_i.is_some() {
            registers_i
                .get(color)
                .and_then(|op| match ctx.trace_ctx.concrete_of_opref(op) {
                    Some(majit_ir::Value::Int(value)) => Some(ConcreteValue::Int(value)),
                    _ => None,
                })
        } else {
            ctx.concrete_registers_i.get(color).copied()
        };
        if let Some(ConcreteValue::Int(value)) = concrete {
            int_values.push((color, value));
            swept[0] += 1;
        }
    }

    let mut ref_values = Vec::with_capacity(pjc.jitcode.num_regs_r());
    let mut ref_seeded = vec![false; pjc.jitcode.num_regs_r()];
    for &color in &live.ref_ {
        let color = color as usize;
        if result_bank == 'r' && result_color == Some(color) {
            continue;
        }
        // `concrete_registers_r` is the color-indexed shadow of `registers_r`:
        // it is allocated to `num_regs_r + constants_r.len()` and every walker
        // handler writes `concrete_registers_r[dst]` in lock-step with
        // `registers_r[dst]`.  `semantic_ref_slot_for_reg_color` maps a color
        // to a `locals_cells_stack_w` slot, which is what
        // `restore_guard_failure_values` needs to call `set_local_at` /
        // `set_stack_at` on a concrete PyFrame — a different index space.
        // Reading the shadow through it stamped whatever register happened to
        // live at the slot's number.
        let got = match &paused_concrete_r {
            Some(concretes) => concretes.get(color).copied(),
            None => ctx
                .frame_state
                .borrow()
                .concrete_registers_r
                .get(color)
                .copied(),
        };
        let ref_shadow_len = paused_concrete_r
            .as_ref()
            .map(|concretes| concretes.len())
            .unwrap_or_else(|| ctx.frame_state.borrow().concrete_registers_r.len());
        let value = match got {
            Some(ConcreteValue::Ref(value)) => value,
            // The shadow holds `ConcreteValue::Null`, the walker's UNTRACKED
            // sentinel rather than a proven Python null: `write_ref_reg` stamps
            // it for every op whose result the walk records without observing
            // (field reads, residual calls).  The innermost-frame fill
            // (`residual_call.rs`, `build_single_frame_miframe`'s `live.ref_`
            // loop) meets the same two situations and answers them without
            // refusing the image; this site is the one copy that never received
            // those answers.
            _ => {
                let opref = registers_r.get(color);
                match opref {
                    // No box at this color at all.  A `-live-` set is the union
                    // over the paths INTO its coordinate and the walk took one
                    // of them, so the color can be live at the coordinate and
                    // undefined on the path actually walked.  Leave it unset,
                    // exactly as `_copy_data_from_miframe`
                    // (`blackhole.py`) leaves a `None` box unset —
                    // refusing here discards the whole image over a register
                    // nothing on this path will read.
                    Some(o) if o.is_none() => continue,
                    // The shadow never held this color's concrete, but the
                    // recorded producer still can: `recover_ref_value` walks a
                    // `GetfieldGcR` chain back to a value the walk did observe
                    // and rereads the field.
                    Some(o) => match ctx.trace_ctx.recover_ref_value(o, 8) {
                        Some(majit_ir::Value::Ref(gc)) => gc.0 as pyre_object::PyObjectRef,
                        _ => {
                            // A real box whose concrete cannot be recovered
                            // is not an absent `None`.  `_copy_data_from_miframe`
                            // skips only missing boxes; `build_single_frame_miframe`
                            // declines this same unrecoverable live color.
                            report_caller_image_decline(
                                jitcode_index,
                                call_jit_pc,
                                'r',
                                color,
                                ctx.frame_state.borrow().concrete_registers_r.len(),
                                got,
                            );
                            report_caller_image_ref_box(ctx, color);
                            return None;
                        }
                    },
                    None => {
                        // Color past the walk's `registers_r` bank, which
                        // upstream's always-allocated `num_regs_r()` bank has
                        // no counterpart for.  That is a bank this walk cannot
                        // answer for, not an absent box, so refuse.
                        report_caller_image_decline(
                            jitcode_index,
                            call_jit_pc,
                            'r',
                            color,
                            ref_shadow_len,
                            got,
                        );
                        report_caller_image_ref_box(ctx, color);
                        return None;
                    }
                }
            }
        };
        ref_values.push((color, value));
        if let Some(seeded) = ref_seeded.get_mut(color) {
            *seeded = true;
        }
    }
    for color in 0..pjc.jitcode.num_regs_r() {
        if ref_seeded[color] || (result_bank == 'r' && result_color == Some(color)) {
            continue;
        }
        // `ConcreteValue::Null` is the walker's "unknown" sentinel rather than a
        // proven Python null, so it seeds nothing.
        let concrete = match &paused_concrete_r {
            Some(concretes) => concretes.get(color).copied(),
            None => ctx
                .frame_state
                .borrow()
                .concrete_registers_r
                .get(color)
                .copied(),
        };
        if let Some(ConcreteValue::Ref(value)) = concrete {
            ref_values.push((color, value));
            swept[1] += 1;
        }
    }

    let mut float_values = Vec::with_capacity(pjc.jitcode.num_regs_f());
    let mut float_seeded = vec![false; pjc.jitcode.num_regs_f()];
    for &color in &live.float {
        let color = color as usize;
        if result_bank == 'f' && result_color == Some(color) {
            continue;
        }
        let got = registers_f.get(color);
        let opref = match got {
            // `_copy_data_from_miframe` skips a missing float box.
            Some(opref) if opref == OpRef::NONE => continue,
            Some(opref) => opref,
            // Color past the walk's `registers_f` bank: the same refusal as
            // the two banks above.
            None => {
                report_caller_image_decline(
                    jitcode_index,
                    call_jit_pc,
                    'f',
                    color,
                    registers_f.len(),
                    got,
                );
                return None;
            }
        };
        float_values.push((color, opref));
        if let Some(seeded) = float_seeded.get_mut(color) {
            *seeded = true;
        }
    }
    // Floats keep their OpRefs and resolve at force time, and the consumer
    // DECLINES the whole image on one that is not a stamped Float — so unlike
    // the two concrete banks, the sweep resolves here and carries only the
    // colors that already have a Float.  A swept color is opportunistic; it
    // must not be able to turn a usable image into a declined one.
    for color in 0..pjc.jitcode.num_regs_f() {
        if float_seeded[color] || (result_bank == 'f' && result_color == Some(color)) {
            continue;
        }
        let Some(opref) = registers_f.get(color) else {
            continue;
        };
        if opref == OpRef::NONE {
            continue;
        }
        if matches!(
            ctx.trace_ctx.concrete_of_opref(opref),
            Some(majit_ir::Value::Float(_))
        ) {
            float_values.push((color, opref));
            swept[2] += 1;
        }
    }

    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-blackhole] caller image jitcode_index={jitcode_index} \
             resume_pc={} banks={}/{}/{} swept={}/{}/{}",
            call.next_pc,
            int_values.len(),
            ref_values.len(),
            float_values.len(),
            swept[0],
            swept[1],
            swept[2],
        );
    }

    Some(InlineParentBlackhole {
        resume_pc: call.next_pc,
        int_values,
        ref_values,
        float_values,
    })
}

/// Return the exact JitCode coordinate at which a paused caller continues
/// after an inlined residual call.
///
/// RPython's `MIFrame.pc` remains immediately after the call instruction
/// while the callee runs.  Preserve that shape here: blackhole return setup
/// walks backward from this coordinate to recover the call's destination
/// register.  Advancing to a later semantic-fallthrough marker skips the
/// intervening `live` / virtualizable synchronization instructions and makes
/// that backward walk read an unrelated operand byte.
pub(crate) fn inline_call_return_marker(
    pjc: &crate::pyjitcode::PyJitCode,
    call_jit_pc: usize,
) -> Option<usize> {
    let marker =
        crate::jitcode_runtime::decode_op_at(pjc.jitcode.code.as_slice(), call_jit_pc)?.next_pc;
    pjc.jitcode
        .can_decode_live_vars(marker, crate::state::op_live())
        .then_some(marker)
}

/// `get_list_of_active_boxes(in_a_call=True)` (`pyjitpl.py`): the call's
/// result register `self.bytecode[self.pc - 1]` is not defined yet, so it
/// reads as `CONST_FALSE` / `CONST_NULL` / `CONST_FZERO` in its
/// `_result_argcode` bank. `banks` is `[int, ref, float]`. Returns the box
/// the walk keeps there, for [`restore_in_a_call_result`].
fn clear_in_a_call_result(
    trace_ctx: &mut TraceCtx,
    banks: [&RegisterBank; 3],
    result_register: Option<(char, usize)>,
) -> Option<(usize, usize, OpRef)> {
    let (bank, color) = result_register?;
    let (index, cleared) = match bank {
        'i' => (0, trace_ctx.const_int(0)),
        'r' => (1, trace_ctx.const_ref(pyre_object::PY_NULL as i64)),
        'f' => (2, trace_ctx.const_float(0)),
        _ => return None,
    };
    let saved = banks[index].get(color)?;
    banks[index].set(color, cleared);
    Some((index, color, saved))
}

/// Put back the box [`clear_in_a_call_result`] displaced: the walk, unlike
/// the paused `MIFrame`, keeps reading the caller's registers.
fn restore_in_a_call_result(banks: [&RegisterBank; 3], saved: Option<(usize, usize, OpRef)>) {
    if let Some((index, color, saved)) = saved {
        banks[index].set(color, saved);
    }
}

pub(crate) fn compute_inline_caller_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    // #68 nested multiframe: when an inlined callee is already active
    // (the framestack has a top level), the immediate caller of THIS
    // call is that intermediate callee (a sym-less sub-jitcode), not the
    // top-level `fbw_mode.snapshot_sym`.  Compute its paused frame from that
    // jitcode (index / liveness / resume tables via its `PyJitCode`), reading the
    // boxes from the caller's live register banks (`ctx.registers_*`, which ARE
    // the intermediate callee's banks here) via the sym-less
    // `collect_callee_active_boxes`.  The stack is empty for a top-level
    // caller, falling through to the `fbw_mode.snapshot_sym` path below.
    let caller_code = ctx.session.borrow().last_inline().map(|frame| frame.w_code);
    if let Some(caller_code) = caller_code {
        return compute_nested_inline_caller_frame(ctx, call_jit_pc, caller_code);
    }
    let caller_sym_ptr = ctx.fbw_mode.snapshot_sym;
    if caller_sym_ptr.is_null() {
        return Err(unavail("Unavail::Top/NoSym"));
    }
    // SAFETY: set for the lifetime of the top-level `dispatch_via_miframe`;
    // read-only access to immutable layout fields.
    let caller_sym = unsafe { &*caller_sym_ptr };
    if caller_sym.jitcode().is_null() {
        return Err(unavail("Unavail::Top/NoJitCode"));
    }
    let (jitcode_index, fallthrough_py_pc, resume_marker_jit_pc, code_ptr, call_is_void) = unsafe {
        let jc = &*caller_sym.jitcode();
        if jc.payload.code_ptr.is_null() || !jc.payload.is_populated() {
            return Err(unavail("Unavail::Top/NotPopulated"));
        }
        let call_is_void =
            crate::jitcode_runtime::decode_op_at(jc.payload.jitcode.code.as_slice(), call_jit_pc)
                .is_some_and(|op| op.result_argcode() == 'v');
        // #73 phase-3 (twin-first): the forward `after_residual_fallthrough`
        // twin carries the containing-then-fallthrough coordinate; the
        // containing lookup survives for the empty-twin class and as the audit
        // oracle.
        let legacy_fallthrough = || unsafe {
            let call_py =
                crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, call_jit_pc)
                    as usize;
            let code = &*jc.payload.code_ptr;
            crate::pyjitpl::semantic_fallthrough_pc(code, call_py) as u32
        };
        let fallthrough = match jc
            .payload
            .after_residual_fallthrough_py_pc_for_jitcode_pc(call_jit_pc)
        {
            Some(ft) => {
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        ft,
                        legacy_fallthrough(),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: inline-caller fallthrough-py twin diverged at jit_pc {call_jit_pc}"
                    );
                }
                ft
            }
            None => {
                crate::py_coord::note_empty_twin_fallback(
                    "inline_caller_fallthrough",
                    jc.index,
                    call_jit_pc as i32,
                );
                legacy_fallthrough()
            }
        };
        (
            jc.index as u32,
            fallthrough,
            inline_call_return_marker(&jc.payload, call_jit_pc),
            jc.payload.code_ptr,
            call_is_void,
        )
    };
    // The call result is the top operand-stack slot only when it remains live
    // at the return point; a call whose value is immediately discarded can
    // resume with depth zero.
    //
    // `None` is "this source has no entry for the coordinate", which is not
    // the same statement as a source answering zero — see the decline below.
    let depth_read: Option<usize> = match resume_marker_jit_pc {
        Some(marker) => unsafe { &(*caller_sym.jitcode()).payload }
            .depth_trivia_for_jitcode_pc(marker)
            .map(|depth| depth as usize),
        // A missing marker has no fallthrough-native key. Preserve the
        // existing Python-keyed path rather than declining a formerly valid
        // caller frame.
        None => {
            let payload = unsafe { &(*caller_sym.jitcode()).payload };
            let raw = || unsafe {
                crate::liveness::liveness_for(code_ptr)
                    .depth_at_py_pc()
                    .get(fallthrough_py_pc as usize)
                    .map(|&depth| depth as usize)
            };
            if payload.depth_after_residual_populated() {
                let depth = payload
                    .depth_after_residual_for_jitcode_pc(call_jit_pc)
                    .map(|depth| depth as usize);
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        depth.unwrap_or(0),
                        raw().unwrap_or(0),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: inline-caller after-residual depth twin diverged at jit_pc {call_jit_pc} (py {fallthrough_py_pc})"
                    );
                }
                depth
            } else {
                raw()
            }
        }
    };
    let depth = depth_read.unwrap_or(0);
    // A source with no entry for this coordinate defaults to zero, and that
    // default is the only zero this cannot trust. A source that ANSWERS zero
    // states an empty operand stack, which is what a call leaving no result
    // behind resumes with — `STORE_ATTR`'s setter body is that shape, popping
    // both operands and pushing nothing. Keep declining the defaulted zero.
    if depth == 0 && !call_is_void && depth_read.is_none() {
        return Err(unavail("Unavail::Top/DepthZero"));
    }
    let call_stack_overrides = collect_call_stack_overrides(caller_sym, ctx, call_jit_pc)
        .ok_or_else(|| unavail("Unavail::Top/CallStack"))?;
    let result_register = super::residual_call::parent_call_result_color(
        unsafe { &(*caller_sym.jitcode()).payload },
        call_jit_pc,
    );
    let paused = paused_python_live(ctx);
    let paused_regs_r = paused.as_ref().map(|live| live.registers_r.clone());
    let paused_regs_i = paused.as_ref().map(|live| live.registers_i.clone());
    let paused_regs_f = paused.as_ref().map(|live| live.registers_f.clone());
    let registers_r = paused_regs_r.as_ref().unwrap_or(ctx.registers_r);
    let registers_i = paused_regs_i.as_ref().unwrap_or(ctx.registers_i);
    let registers_f = paused_regs_f.as_ref().unwrap_or(ctx.registers_f);
    let caller_state = paused.as_ref().map(|live| live.frame_state().clone());
    let banks = [registers_i, registers_r, registers_f];
    let saved = clear_in_a_call_result(ctx.trace_ctx, banks, result_register);
    // Keep the caller at the immediate post-call `-live-`, matching the
    // paused `MIFrame.pc` used by RPython and the blackhole return ABI.
    let caller_liveness_word = match resume_marker_jit_pc {
        Some(m) => m as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let boxes = collect_outer_active_boxes(
        caller_sym,
        ctx.trace_ctx,
        registers_i,
        registers_r,
        registers_f,
        jitcode_index,
        caller_liveness_word,
        caller_liveness_word,
        OuterActiveBoxesEntryTwin::Trivia,
        "inline_caller",
        None,
    );
    restore_in_a_call_result(banks, saved);
    Ok(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: Some(call_jit_pc),
        call_pc: Some(call_jit_pc),
        resume_pc: resume_marker_jit_pc,
        call_stack_overrides,
        blackhole: capture_inline_parent_blackhole(ctx, jitcode_index, call_jit_pc),
        resume_coord: ParentResumeCoord::CallFallthrough(call_jit_pc),
        resume_marker_jit_pc,
        boxes,
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
        result_slot: CallResultSlot::Unknown,
    }
    .attach_live_caller(
        registers_r,
        registers_i,
        registers_f,
        caller_state.as_ref().unwrap_or(&ctx.frame_state),
    )
    .with_caller_py_pc(jitcode_index, call_jit_pc))
}

/// Paused-caller-frame computation for a NESTED multiframe inline (#68): the
/// immediate caller is an intermediate inlined callee (`caller_code`), a
/// sym-less sub-jitcode whose live boxes are in `ctx.registers_*` (this `ctx`
/// IS that callee's sub-walk).  Mirror of the top-level
/// [`compute_inline_caller_frame`] body but keyed on `caller_code`'s own
/// `PyJitCode` instead of `fbw_mode.snapshot_sym`, and reading boxes via the
/// sym-less [`collect_callee_active_boxes`] (no portal-vable shadow to fold).
pub(crate) fn compute_nested_inline_caller_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
    caller_code: usize,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    let jitcode_index = crate::state::ensure_jitcode_index(caller_code as *const ())
        .ok_or_else(|| unavail("Unavail::Nested/NoJitcodeIndex"))? as u32;
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
        .ok_or_else(|| unavail("Unavail::Nested/NoPjc"))?;
    if !pjc.is_populated() || pjc.code_ptr.is_null() {
        return Err(unavail("Unavail::Nested/NotPopulated"));
    }
    let resume_marker_jit_pc = inline_call_return_marker(&pjc, call_jit_pc);
    let call_is_void =
        crate::jitcode_runtime::decode_op_at(pjc.jitcode.code.as_slice(), call_jit_pc)
            .is_some_and(|op| op.result_argcode() == 'v');
    let legacy_fallthrough_py_pc = || unsafe {
        let call_py =
            crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, call_jit_pc) as usize;
        let code = &*pjc.code_ptr;
        crate::pyjitpl::semantic_fallthrough_pc(code, call_py) as u32
    };
    // The call result is the top operand-stack slot only when it remains live
    // at the return point; a call whose value is immediately discarded can
    // resume with depth zero.
    //
    // `None` is "this source has no entry for the coordinate", which is not
    // the same statement as a source answering zero — see the decline below.
    let depth_read: Option<usize> = match resume_marker_jit_pc {
        Some(marker) => pjc
            .depth_trivia_for_jitcode_pc(marker)
            .map(|depth| depth as usize),
        None => {
            let raw = || {
                let fallthrough_py_pc = legacy_fallthrough_py_pc();
                unsafe {
                    crate::liveness::liveness_for(pjc.code_ptr)
                        .depth_at_py_pc()
                        .get(fallthrough_py_pc as usize)
                        .map(|&depth| depth as usize)
                }
            };
            if pjc.depth_after_residual_populated() {
                let depth = pjc
                    .depth_after_residual_for_jitcode_pc(call_jit_pc)
                    .map(|depth| depth as usize);
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        depth.unwrap_or(0),
                        raw().unwrap_or(0),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: nested inline-caller after-residual depth twin diverged at jit_pc {call_jit_pc}"
                    );
                }
                depth
            } else {
                raw()
            }
        }
    };
    let depth = depth_read.unwrap_or(0);
    // Only the defaulted zero is untrusted; a source answering zero states an
    // empty operand stack — see the top-level sibling.
    if depth == 0 && !call_is_void && depth_read.is_none() {
        return Err(unavail("Unavail::Nested/DepthZero"));
    }
    let result_register = super::residual_call::parent_call_result_color(&pjc, call_jit_pc);
    let banks = [ctx.registers_i, ctx.registers_r, ctx.registers_f];
    let saved = clear_in_a_call_result(ctx.trace_ctx, banks, result_register);
    // Keep the caller at the immediate post-call `-live-`, matching the
    // paused `MIFrame.pc` used by RPython and the blackhole return ABI.
    // Without a marker there is no coordinate to encode against, so the
    // sentinel declines the caller frame.
    let caller_liveness_word = match resume_marker_jit_pc {
        Some(m) => m as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let boxes = collect_callee_active_boxes(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        jitcode_index,
        call_jit_pc,
        caller_liveness_word,
    );
    restore_in_a_call_result(banks, saved);
    let boxes = match boxes {
        Ok(b) => b,
        Err(_) => return Err(unavail("Unavail::Nested/BoxesErr")),
    };
    // #343 depth-2 operand flush.  `emit_new_pyframe_inline_with_params` seeds
    // this inlined-callee frame's `locals_cells_stack_w` virtual array with LOCALS
    // only; its live operand-stack slots live in the register file and are never
    // written back into the frame array.  When a guard fires inside a NESTED
    // inline (this frame is the paused MIDDLE caller), the multi-frame snapshot
    // captures this frame virtual and numbers its array — an unwritten operand
    // slot numbers to `NULLREF`, which `reconstruct_inline_recipe`
    // (state.rs, semantic-slot-ordered `arr[k]` read) decodes to a
    // concrete-NULL Ref box.  The reconstructed middle would then consume that
    // hole at its next residual.  Materialize the live
    // operand boxes into the frame array now — at the point we pause this frame to
    // enter the nested inline — mirroring the locals seed at `helpers.rs`
    // (`virtualizable.py` write_boxes writes the frame's field boxes into
    // its array when it is forced).  Only NON-pending operand slots are written;
    // the top slot is the not-yet-produced nested-call result, delivered on resume
    // (`pending_result_abs_slot`), so the decoder skips it and so do we.  Emitting
    // here (before the nested callee's body walk) — not at capture — places the
    // stores ahead of every in-callee guard, so `_number_virtuals` reads the
    // populated array when it numbers those guards' resume data.
    // A void call leaves no pending result, so every operand slot is a
    // sibling (`get_list_of_active_boxes` clears only the `_result_argcode`
    // bank's register).
    let result_slots = usize::from(!call_is_void);
    if resume_marker_jit_pc.is_some() {
        if caller_liveness_word != majit_ir::resumedata::NO_JITCODE_PC && depth > result_slots {
            let (frame_reg, _) = crate::state::portal_red_regs_at(jitcode_index as i32);
            let Some(frame_red) = (frame_reg != u16::MAX)
                .then(|| ctx.registers_r.get(frame_reg as usize))
                .flatten()
                .filter(|&r| r != OpRef::NONE)
            else {
                return Err(unavail("Unavail::Nested/NoFrameRed"));
            };
            // `gen_store_back_in_vable` reads the array with
            // `GETFIELD_GC_R`; the heapcache answers it when it can.
            {
                let locals_array =
                    crate::state::frame_locals_cells_stack_array(ctx.trace_ctx, frame_red);
                // The operand stack starts behind the cell and free variable
                // slots (`PyFrame._stack_start`), the same base
                // `reconstruct_inline_recipe` decodes the array with.
                let code = unsafe { &*pjc.code_ptr };
                let nlocals = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
                // `depth` is the operand-stack depth at the CALL return point;
                // a call with a result leaves it pending in the top slot.
                let pending_top_slot = (!call_is_void).then(|| nlocals + depth - 1);
                // No Python coordinate here. This slot used to carry
                // `resume_marker_jit_pc`, which `inline_call_return_marker`
                // mints as `decode_op_at(..).next_pc` — a JitCode byte offset,
                // not a Python PC. It gates the block above and nothing else.
                let maps = crate::state::bridge_semantic_maps_at_with_jitcode_pc(
                    jitcode_index as i32,
                    None,
                    caller_liveness_word,
                );
                let array_descr = crate::state::pyobject_gcarray_descr();
                // Heapcache array-element key = the virtualizable info's array
                // item descr, matching the seed store at `helpers.rs`.
                let item_descr_index = ctx
                    .trace_ctx
                    .virtualizable_info()
                    .map(|info| info.array_item_descr(0).index())
                    .unwrap_or_else(|| array_descr.index());
                // Live Ref colors at the CALL return coordinate, same enumeration
                // as `collect_callee_active_boxes`.
                let banks = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
                    jitcode_index as i32,
                    caller_liveness_word,
                );
                for &color in &banks.ref_ {
                    let Some(operand) = ctx.registers_r.get(color as usize) else {
                        continue;
                    };
                    if operand == OpRef::NONE {
                        continue;
                    }
                    // `gen_store_back_in_vable` stores every array position;
                    // a COPY shape maps one register to several of them.
                    for slot in crate::state::semantic_ref_slots_for_reg_color(
                        nlocals,
                        depth,
                        &maps.pcdep_entries,
                        color as usize,
                    ) {
                        // Locals are already seeded; the pending result slot is
                        // delivered on resume — only sibling operands need the flush.
                        if slot < nlocals || Some(slot) == pending_top_slot {
                            continue;
                        }
                        let idx = ctx.trace_ctx.const_int(slot as i64);
                        // pyjitpl.py `gen_store_back_in_vable`
                        // escape flush uses the standard recorded store path.
                        ctx.trace_ctx
                            .profiler()
                            .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
                        ctx.trace_ctx.profiler().count_ops(
                            OpCode::SetarrayitemGc,
                            majit_metainterp::counters::RECORDED_OPS,
                        );
                        ctx.trace_ctx.record_op_with_descr(
                            OpCode::SetarrayitemGc,
                            &[locals_array, idx, operand],
                            array_descr.clone(),
                        );
                        ctx.trace_ctx.heapcache_setarrayitem(
                            locals_array,
                            idx,
                            item_descr_index,
                            operand,
                        );
                    }
                }
            }
        }
    }
    Ok(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: Some(call_jit_pc),
        call_pc: Some(call_jit_pc),
        resume_pc: resume_marker_jit_pc,
        call_stack_overrides: Vec::new(),
        blackhole: capture_inline_parent_blackhole(ctx, jitcode_index, call_jit_pc),
        resume_coord: ParentResumeCoord::CallFallthrough(call_jit_pc),
        resume_marker_jit_pc,
        boxes,
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
        result_slot: CallResultSlot::Unknown,
    }
    .attach_live_caller(
        ctx.registers_r,
        ctx.registers_i,
        ctx.registers_f,
        &ctx.frame_state,
    )
    .with_caller_py_pc(jitcode_index, call_jit_pc))
}

/// Fill [`InlineParentFrame::result_slot`] from the post-call `-live-`
/// (`call_result_slot`, `blackhole.py` `_setup_return_value_*`).
fn with_call_result_slot(mut frame: InlineParentFrame) -> InlineParentFrame {
    if let Some(pc) = frame.resume_marker_jit_pc {
        frame.result_slot = super::call_result_slot(frame.jitcode_index, pc, frame.boxes.len());
    }
    frame
}

/// Pause the enclosing helper at its CALL into a nested helper.
///
/// `capture_inline_parent_blackhole` would snapshot the Python banks
/// (`paused_python_live`); this caller is the helper's own jitcode, so the
/// live helper banks go on `attach_live_caller` and `blackhole` stays empty.
/// `build_parent_miframe` rebuilds that `MIFrame` from those banks.
fn compute_paused_helper_caller_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    let Some(jitcode_index) = ctx.fbw_mode.transparent_helper_jitcode_index else {
        return Err(unavail("Unavail::PausedHelper/NoIndex"));
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32) else {
        return Err(unavail("Unavail::PausedHelper/NoPjc"));
    };
    if pjc.jitcode.code.is_empty() {
        return Err(unavail("Unavail::PausedHelper/Empty"));
    }
    let resume_marker_jit_pc = inline_call_return_marker(&pjc, call_jit_pc)
        .ok_or_else(|| unavail("Unavail::PausedHelper/NoMarker"))?;
    let result_register = super::residual_call::parent_call_result_color(&pjc, call_jit_pc);
    let banks = [ctx.registers_i, ctx.registers_r, ctx.registers_f];
    let saved = clear_in_a_call_result(ctx.trace_ctx, banks, result_register);
    let boxes = collect_callee_active_boxes(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        jitcode_index as u32,
        call_jit_pc,
        resume_marker_jit_pc as i32,
    );
    restore_in_a_call_result(banks, saved);
    let boxes = match boxes {
        Ok(boxes) => boxes,
        Err(_) => return Err(unavail("Unavail::PausedHelper/BoxesErr")),
    };
    Ok(with_call_result_slot(
        InlineParentFrame {
            jitcode_index: jitcode_index as u32,
            call_jitcode_pc: Some(call_jit_pc),
            call_pc: Some(call_jit_pc),
            resume_pc: Some(resume_marker_jit_pc),
            call_stack_overrides: Vec::new(),
            blackhole: None,
            resume_coord: ParentResumeCoord::CallFallthrough(call_jit_pc),
            resume_marker_jit_pc: Some(resume_marker_jit_pc),
            boxes,
            registers_r: None,
            registers_i: None,
            registers_f: None,
            frame_state: None,
            caller_py_pc: None,
            result_slot: CallResultSlot::Unknown,
        }
        .attach_live_caller(
            ctx.registers_r,
            ctx.registers_i,
            ctx.registers_f,
            &ctx.frame_state,
        ),
    ))
}

/// A resume section `rebuild_from_resumedata` can turn back into a Python
/// frame. The codeless slot 0 (and constructor tails) have no `CodeObject`.
fn jitcode_index_names_python_code(jitcode_index: u32) -> bool {
    crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
        .is_some_and(|pjc| pjc.is_populated() && !pjc.code_ptr.is_null())
}

/// Capture the current inlined Python frame at a translated helper call's
/// entry boundary, for a helper whose return the walk converts before it
/// becomes the call's result ([`HelperReturn::Converted`]). No `MIFrame`
/// holds that conversion, so a guard inside the helper rebuilds its Python
/// caller at the CALL and re-executes the call, while the already-paused
/// outer frames remain below it. The caller still needs its own concrete
/// blackhole image: without it the multi-frame conversion declines and the
/// helper-local abort pc can escape into an outer frame's coordinate space.
pub(crate) fn compute_inline_helper_call_entry_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    // `w_code == 0` is a helper already on the framestack. It has no PyCode.
    // `rebuild_from_resumedata` does not look one up either: it indexes the
    // jitcode `MetaInterp.newframe` is already executing, which this frame's
    // `InlineParentFrame.jitcode_index` holds (the resume word captured at the
    // helper CALL). `ensure_jitcode_index` on that null code binds the
    // codeless slot 0, and a bridge walk then enters the helper body at pc 0.
    // A deeper helper with an empty parent list walks out to that same frame.
    // A real PyCode still takes the lookup below, unless that lookup is the
    // codeless slot — a bridge callee's `w_code` is the live wrapper, and the
    // jitcode `newframe` was given is `InlineCalleeConsts.jitcode_index`.
    let (held_helper_parent, helper_level, stored_jitcode_index) = {
        let session = ctx.session.borrow();
        let helper_level = session.last_inline().is_some_and(|frame| frame.w_code == 0);
        let mut held = None;
        let mut stored = None;
        if helper_level {
            for frame in session.framestack.iter().rev() {
                if frame.is_portal {
                    break;
                }
                if let Some(parent) = frame
                    .parents
                    .iter()
                    .rev()
                    .find(|parent| jitcode_index_names_python_code(parent.jitcode_index))
                {
                    held = Some(parent.clone());
                    stored = Some(parent.jitcode_index);
                    break;
                }
                if frame.w_code != 0 {
                    break;
                }
            }
        }
        if stored.is_none() {
            stored = ctx
                .inline_callee_consts
                .map(|consts| consts.jitcode_index)
                .filter(|index| *index >= 0 && jitcode_index_names_python_code(*index as u32))
                .map(|index| index as u32);
        }
        (held, helper_level, stored)
    };
    if let Some(held) = held_helper_parent {
        return Ok(held);
    }
    let caller_code = ctx.session.borrow().last_inline().map(|frame| frame.w_code);
    let (jitcode_index, pjc) = if helper_level || caller_code.is_some() {
        let looked_up = caller_code.and_then(|caller_code| {
            if caller_code == 0 {
                return None;
            }
            let index = crate::state::ensure_jitcode_index(caller_code as *const ())? as u32;
            jitcode_index_names_python_code(index).then_some(index)
        });
        // Helper level and a bridge callee both already know the jitcode
        // `newframe` is executing. The `w_code` lookup answers the codeless
        // slot there; keep the stored index.
        let jitcode_index = if helper_level {
            stored_jitcode_index.or(looked_up)
        } else {
            looked_up.or(stored_jitcode_index)
        }
        .ok_or_else(|| unavail("Unavail::Helper/NoJitcodeIndex"))?;
        let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
            .ok_or_else(|| unavail("Unavail::Helper/NoPjc"))?;
        (jitcode_index, pjc)
    } else {
        let caller_sym_ptr = ctx.fbw_mode.snapshot_sym;
        if caller_sym_ptr.is_null() {
            return Err(unavail("Unavail::Helper/NoSym"));
        }
        // SAFETY: `snapshot_sym` is installed for the lifetime of the
        // full-body walk and only its immutable JitCode payload is read here.
        let caller_sym = unsafe { &*caller_sym_ptr };
        if caller_sym.jitcode().is_null() {
            return Err(unavail("Unavail::Helper/NoJitCode"));
        }
        let jc = unsafe { &*caller_sym.jitcode() };
        (jc.index as u32, jc.payload.clone())
    };
    if !pjc.is_populated() || pjc.code_ptr.is_null() {
        return Err(unavail("Unavail::Helper/NotPopulated"));
    }
    // Same parent resume as `run_orthodox_helper_subwalk`: the op2 `-live-`
    // after `jit_merge_point` on a loop-header opcode
    // (`jtransform.py handle_jit_marker__jit_merge_point`).
    let resume_marker_jit_pc = super::specialize::collapsed_parent_resume_marker(&pjc, call_jit_pc)
        .ok_or_else(|| unavail("Unavail::Helper/NoResumeMarker"))?;
    let boxes = if caller_code.is_some() {
        collect_callee_active_boxes(
            ctx.registers_i,
            ctx.registers_r,
            ctx.registers_f,
            jitcode_index,
            call_jit_pc,
            resume_marker_jit_pc as i32,
        )
        .map_err(|_| unavail("Unavail::Helper/BoxesErr"))?
    } else {
        let vstack_boxes = ctx.frame_state.borrow().vstack_boxes.clone();
        let vstack = ctx.vstack_valid.then_some(vstack_boxes.as_slice());
        let caller_sym_ptr = ctx.fbw_mode.snapshot_sym;
        // SAFETY: checked non-null above and valid for this full-body walk.
        let caller_sym = unsafe { &*caller_sym_ptr };
        collect_outer_active_boxes(
            caller_sym,
            ctx.trace_ctx,
            ctx.registers_i,
            ctx.registers_r,
            ctx.registers_f,
            jitcode_index,
            resume_marker_jit_pc as i32,
            call_jit_pc as i32,
            OuterActiveBoxesEntryTwin::Plain,
            "builtin_wrapper_call_entry",
            vstack,
        )
    };
    Ok(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: None,
        call_pc: Some(call_jit_pc),
        resume_pc: Some(resume_marker_jit_pc),
        call_stack_overrides: Vec::new(),
        blackhole: capture_inline_parent_blackhole(ctx, jitcode_index, call_jit_pc),
        resume_coord: ParentResumeCoord::Backxlat(resume_marker_jit_pc),
        resume_marker_jit_pc: Some(resume_marker_jit_pc),
        boxes,
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
        result_slot: CallResultSlot::Unknown,
    }
    .attach_live_caller(
        ctx.registers_r,
        ctx.registers_i,
        ctx.registers_f,
        &ctx.frame_state,
    )
    .with_caller_py_pc(jitcode_index, call_jit_pc))
}

/// A transparent helper level to push on `framestack` for a helper sub-walk.
pub(crate) struct HelperLevel {
    pub(crate) parents: Vec<InlineParentFrame>,
    pub(crate) resume: HelperResume,
    pub(crate) call_entry: HelperCallEntry,
    /// Length of the argument array the helper is entered with, when the
    /// call site knows it (a generated gateway wrapper).
    pub(crate) entry_array_len: Option<usize>,
    /// Recorder `_index` at the CALL that entered this level, stamped by
    /// [`enter_helper_level`] from `Trace.get_trace_position`.
    pub(crate) entry_trace_index: u32,
}

impl HelperLevel {
    fn section(caller: InlineParentFrame, call_entry: HelperCallEntry) -> Self {
        HelperLevel {
            parents: vec![caller],
            resume: HelperResume::Section,
            call_entry,
            entry_array_len: None,
            entry_trace_index: 0,
        }
    }

    fn nested_section(caller: InlineParentFrame) -> Self {
        Self::section(caller, HelperCallEntry::Unavailable)
    }

    fn unresumable() -> Self {
        HelperLevel {
            parents: Vec::new(),
            resume: HelperResume::Unresumable,
            call_entry: HelperCallEntry::Unavailable,
            entry_array_len: None,
            entry_trace_index: 0,
        }
    }
}

/// The helper level to push when the current walk enters a helper at
/// `call_jit_pc`, or `None` when its guards keep the enclosing resume.
///
/// `capture_resumedata` numbers one section per `MIFrame`: the caller paused
/// past its call (`get_list_of_active_boxes(in_a_call=True)`), then the
/// helper. That holds only when every helper level below is numbered too and
/// the helper's return is the call's result. An inlined Python caller that
/// cannot pause past the call (or whose call result is converted) resumes at
/// the CALL entry instead; an `Err` there declines the descent. A helper under
/// a numbered helper that cannot be numbered is pushed as
/// [`HelperResume::Unresumable`] so its guards are refused rather than
/// numbered at the enclosing helper's position.
pub(crate) fn helper_subwalk_level<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
    ret: HelperReturn,
) -> Result<Option<HelperLevel>, InlineCallerFrameDecline> {
    if ctx.fbw_mode.transparent_helper_subwalk {
        let chain = ctx.session.borrow().helper_chain();
        return Ok(match chain {
            HelperChain::Numbered => Some(
                (ret == HelperReturn::CallResult)
                    .then(|| compute_paused_helper_caller_frame(ctx, call_jit_pc).ok())
                    .flatten()
                    .map_or_else(HelperLevel::unresumable, HelperLevel::nested_section),
            ),
            HelperChain::NoHelper | HelperChain::Collapsed | HelperChain::Unresumable => None,
        });
    }
    if ret == HelperReturn::CallResult
        && let Ok(caller) = compute_inline_caller_frame(ctx, call_jit_pc)
    {
        let call_entry = if ctx.fbw_mode.inline_subwalk {
            compute_inline_helper_call_entry_frame(ctx, call_jit_pc)
                .map_or(HelperCallEntry::Unavailable, HelperCallEntry::Frame)
        } else {
            HelperCallEntry::CallerBoundary
        };
        return Ok(Some(HelperLevel::section(
            with_call_result_slot(caller),
            call_entry,
        )));
    }
    if !ctx.fbw_mode.inline_subwalk {
        // The root portal resumes at its CALL boundary (`entry_py_pc`).
        return Ok(None);
    }
    compute_inline_helper_call_entry_frame(ctx, call_jit_pc).map(|caller| {
        Some(HelperLevel {
            parents: vec![caller],
            resume: HelperResume::CallEntry,
            call_entry: HelperCallEntry::Unavailable,
            entry_array_len: None,
            entry_trace_index: 0,
        })
    })
}

/// How a guard inside a numbered helper chain resumes.
enum NumberedHelperGuard {
    /// The frames to number, outermost-first.
    Frames(Vec<InlineParentFrame>),
    /// The root portal's CALL boundary.
    CallerBoundary,
}

/// Resume of a guard at `op_pc` inside a numbered helper chain:
/// `parent_frames` plus the innermost helper section at the guard.
///
/// The blackhole runs every helper section from its resume pc to its return,
/// and `reject_unresolved_call` bails at a residual call it cannot resolve;
/// a helper section has no Python coordinate the interpreter could continue
/// from. When a section can reach one
/// ([`helper_resume_reaches_blocker`]), the guard resumes at the CALL that
/// entered the outermost helper level instead ([`HelperCallEntry`]), which
/// re-runs that call. That is refused once a heap-writing residual (live
/// heap or a user frame, or an unjournaled `ForIterNext`) or a journaled
/// live-heap effect (a list pop or append, a store) has executed since the
/// level was entered, since the re-run would apply it twice; the refusal is
/// the one an unlowered helper gets, so the descent is cut back to a residual
/// call when it applied nothing and the trace aborts otherwise. A residual
/// the executed-effect odometer counts only for its void result (an
/// allocation, `make_pyerror`) keeps the CALL-entry resume.
fn numbered_helper_guard_resume<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    after_residual_call: bool,
    scope: GuardCaptureScope,
    mut parent_frames: Vec<InlineParentFrame>,
) -> Result<NumberedHelperGuard, DispatchError> {
    parent_frames.push(helper_innermost_resume_frame(
        ctx,
        op_pc,
        after_residual_call,
        scope,
    )?);
    let runs_from = |frame: &InlineParentFrame, entry_array_len: Option<usize>| {
        frame.resume_marker_jit_pc.is_some_and(|pc| {
            !helper_resume_reaches_blocker(frame.jitcode_index as usize, pc, entry_array_len)
        })
    };
    // `capture_resumedata` numbers every frame on `framestack`. A Python
    // inline level below the helper levels that holds no paused-caller image
    // (a straight-line callee) has no section here, so numbering would skip
    // it. Such a guard takes the caller-boundary collapse that
    // `walker_inline_guard_resumes_in_callee` gives the same Python level.
    let python_chain_complete = {
        let session = ctx.session.borrow();
        session
            .framestack
            .iter()
            .rev()
            .skip_while(|frame| frame.is_helper_level())
            .filter(|frame| !frame.is_portal)
            .all(|frame| !frame.parents.is_empty())
    };
    let runnable = python_chain_complete && {
        let session = ctx.session.borrow();
        // The trailing helper levels own the last sections: level `j`'s is
        // the paused parent level `j + 1` holds, and the innermost level's
        // is the one pushed above.
        let levels: Vec<&InlineFrame> = session
            .framestack
            .iter()
            .rev()
            .take_while(|frame| frame.is_helper_level())
            .collect();
        let split = parent_frames.len().saturating_sub(levels.len());
        let (outer, sections) = parent_frames.split_at(split);
        outer.iter().all(|frame| {
            !crate::state::jitcode_is_transparent_helper(frame.jitcode_index as i32)
                || runs_from(frame, None)
        }) && sections
            .iter()
            .zip(levels.iter().rev())
            .all(|(frame, level)| runs_from(frame, level.helper_entry_array_len))
    };
    if runnable {
        return Ok(NumberedHelperGuard::Frames(parent_frames));
    }
    let refused = DispatchError::OrthodoxSubWalkTraceUnsupported {
        pc: op_pc,
        symbolic: 0,
    };
    let session = ctx.session.borrow();
    let helper_levels = session
        .framestack
        .iter()
        .rev()
        .take_while(|frame| frame.is_helper_level())
        .count();
    let outer_index = session.framestack.len() - helper_levels;
    let Some(outer) = session.framestack.get(outer_index) else {
        return Err(refused);
    };
    let heap_moved = fbw_heap_writing_effect_count() != outer.entry_heap_writing_effects;
    let journal_delta = fbw_journaled_effect_delta_since(outer.entry_journal_mark);
    if heap_moved || journal_delta != 0 {
        return Err(refused);
    }
    if !python_chain_complete {
        return Ok(NumberedHelperGuard::CallerBoundary);
    }
    match &outer.helper_call_entry {
        HelperCallEntry::Unavailable => Err(refused),
        HelperCallEntry::CallerBoundary => Ok(NumberedHelperGuard::CallerBoundary),
        HelperCallEntry::Frame(entry) => {
            let mut frames: Vec<InlineParentFrame> = session.framestack[..outer_index]
                .iter()
                .flat_map(|frame| frame.parents.iter().cloned())
                .collect();
            frames.push(entry.clone());
            Ok(NumberedHelperGuard::Frames(frames))
        }
    }
}

/// Innermost helper section at the guard's own jitcode coordinate.
///
/// Same coordinate rule as the Python path: `carried_resume_jit_pc`, else
/// `after_residual_guard_marker` / `live_after_jit_pc`, else
/// `plain_guard_resume_jitcode_pc`. Box count must equal
/// `frame_value_count_at`.
fn helper_innermost_resume_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    after_residual_call: bool,
    scope: GuardCaptureScope,
) -> Result<InlineParentFrame, DispatchError> {
    let Some(jitcode_index) = ctx.fbw_mode.transparent_helper_jitcode_index else {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32) else {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    };
    if pjc.jitcode.code.is_empty() {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    }
    let helper_jitcode_pc: usize = if let Some(carried) = scope.carried_resume_jit_pc {
        carried
    } else if after_residual_call {
        after_residual_guard_marker(&pjc, op_pc, None)
            .or_else(|| {
                (ctx.live_after_jit_pc != usize::MAX
                    && pjc
                        .jitcode
                        .can_decode_live_vars(ctx.live_after_jit_pc, crate::state::op_live()))
                .then_some(ctx.live_after_jit_pc)
            })
            .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc })?
    } else {
        let pc = plain_guard_resume_jitcode_pc(&pjc, op_pc);
        if pc < 0 {
            return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
        }
        pc as usize
    };
    if !pjc
        .jitcode
        .can_decode_live_vars(helper_jitcode_pc, crate::state::op_live())
    {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    }
    let boxes = collect_callee_active_boxes(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        jitcode_index as u32,
        op_pc,
        helper_jitcode_pc as i32,
    )?;
    let expected =
        crate::state::frame_value_count_at(jitcode_index as i32, helper_jitcode_pc as i32);
    if boxes.len() != expected {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    }
    Ok(InlineParentFrame {
        jitcode_index: jitcode_index as u32,
        call_jitcode_pc: None,
        call_pc: None,
        resume_pc: Some(helper_jitcode_pc),
        call_stack_overrides: Vec::new(),
        blackhole: None,
        resume_coord: ParentResumeCoord::Backxlat(helper_jitcode_pc),
        resume_marker_jit_pc: Some(helper_jitcode_pc),
        boxes,
        registers_r: None,
        registers_i: None,
        registers_f: None,
        frame_state: None,
        caller_py_pc: None,
        result_slot: CallResultSlot::Unknown,
    })
}

fn publish_outermost_parent_vable_scalars<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    parent_frames: &[InlineParentFrame],
    op_pc: usize,
) -> Result<crate::trace_opcode::SavedVableResumeScalars, DispatchError> {
    let empty = crate::trace_opcode::SavedVableResumeScalars::default();
    let Some(outer) = parent_frames.first() else {
        return Err(DispatchError::callee_inline_unsupported(op_pc));
    };
    let caller_sym_ptr = ctx.fbw_mode.snapshot_sym;
    if caller_sym_ptr.is_null() {
        return Ok(empty);
    }
    let caller_sym = unsafe { &*caller_sym_ptr };
    if !caller_sym.owns_virtualizable_shadow() || caller_sym.jitcode().is_null() {
        return Ok(empty);
    }

    let saved = crate::trace_opcode::save_vable_resume_scalars(ctx.trace_ctx);
    let resume_py_pc = resolve_parent_resume_py_pc(outer)
        .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc })?;
    let last_instr_value = resume_py_pc as i64 - 1;
    let last_instr_op = ctx.trace_ctx.const_int(last_instr_value);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "last_instr",
        last_instr_op,
        Value::Int(last_instr_value),
    );
    let vsd_value = unsafe {
        let jc = &*caller_sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            caller_sym.valuestackdepth() as i64
        } else {
            let raw = || {
                crate::liveness::liveness_for(jc.payload.code_ptr)
                    .depth_at_py_pc()
                    .get(resume_py_pc as usize)
                    .copied()
            };
            let twin: Option<Option<u16>> = if outer.jitcode_index != jc.index as u32 {
                None
            } else {
                match outer.resume_coord {
                    ParentResumeCoord::Backxlat(jitcode_pc) => jc
                        .payload
                        .depth_trivia_populated()
                        .then(|| jc.payload.depth_trivia_for_jitcode_pc(jitcode_pc)),
                    ParentResumeCoord::CallFallthrough(call_jit_pc) => jc
                        .payload
                        .depth_after_residual_populated()
                        .then(|| jc.payload.depth_after_residual_for_jitcode_pc(call_jit_pc)),
                }
            };
            let depth = match twin {
                Some(d) => {
                    if pcmap_afterresidual_audit_enabled() {
                        assert_eq!(
                            d,
                            raw(),
                            "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: multiframe vsd depth twin diverged (py {resume_py_pc})"
                        );
                    }
                    d
                }
                None => raw(),
            };
            match depth {
                Some(d) => (caller_sym.nlocals() + d as usize) as i64,
                None => caller_sym.valuestackdepth() as i64,
            }
        }
    };
    let vsd_op = ctx.trace_ctx.const_int(vsd_value);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "valuestackdepth",
        vsd_op,
        Value::Int(vsd_value),
    );
    Ok(saved)
}

/// Live boxes for one paused parent at the pc the snapshot will publish.
///
/// `ResumeDataLoopMemo.number` writes `jitcode_index, pc, [tagged...]` and
/// `rebuild_from_numbering` splits that stream with `frame_value_count_at`
/// at the same pc. `frame.boxes` is the image the caller-frame constructor
/// collected with the pending call result nulled
/// (`MIFrame.get_list_of_active_boxes`, `in_a_call`); the paused banks hold
/// the restored register, so they are not re-read here. An image whose
/// length disagrees with the decoder's count is refused, so a parent cannot
/// donate a tagged value into the next frame header.
fn boxes_matching_resume_liveness(
    frame: &InlineParentFrame,
    pc_word: u32,
    op_pc: usize,
) -> Result<Vec<OpRef>, DispatchError> {
    let expected = crate::state::frame_value_count_at(frame.jitcode_index as i32, pc_word as i32);
    // No box is invented for a slot the image does not hold: a section of
    // the wrong length cannot be numbered, so the guard is not recorded.
    if frame.boxes.len() != expected {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    }
    Ok(frame.boxes.clone())
}

/// Number a helper-guard snapshot: paused callers outermost-first, then
/// the helper itself when the caller appended that innermost section.
///
/// `parent_frames` is the whole framestack's paused levels flattened in
/// order, so a level that ran between two frames arrives already at its own
/// depth — `descr_call`'s tail, held on the `__init__` frame that owns it,
/// reaches here as `[caller, tail, __init__]` without this path inserting
/// anything. Nothing here may re-position a level relative to the innermost
/// callee; a helper descent is not that callee's depth. The outermost
/// section must name a Python `CodeObject`: a helper-only numbering would
/// make the helper the vable portal frame (`rebuild_from_resumedata`).
/// The framestack a guard captures once `finishframe` popped the callee: the
/// paused callers, the innermost with `result` in the slot
/// `make_result_of_lastop` writes (`pyjitpl.py do_residual_call`, which
/// records `GUARD_NOT_FORCED` after it). A deopt resumes in the caller past
/// its call instead of re-entering the callee the call already ran.
fn walker_capture_finished_callee_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    result: OpRef,
) -> Result<(), DispatchError> {
    let mut parent_frames = finished_callee_parent_frames(ctx)
        .ok_or(DispatchError::callee_inline_unsupported(op_pc))?;
    if let Some(caller) = parent_frames.last_mut()
        && let CallResultSlot::At(idx) = caller.result_slot
    {
        caller.boxes[idx] = result;
    }
    walker_capture_transparent_helper_snapshot(ctx, op_pc, parent_frames, GuardStampTarget::LastOp)
}

/// The paused callers [`walker_capture_finished_callee_snapshot`] captures,
/// or `None` when the innermost one does not say where its call's result
/// goes.
pub(crate) fn finished_callee_parent_frames<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<Vec<InlineParentFrame>> {
    let parent_frames = {
        let session = ctx.session.borrow();
        session
            .framestack
            .iter()
            .flat_map(|frame| frame.parents.iter().cloned())
            .collect::<Vec<_>>()
    };
    let caller = parent_frames.last()?;
    if caller.result_slot == CallResultSlot::Unknown {
        if super::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-finished-callee] result slot unknown: jitcode={} marker={:?}",
                caller.jitcode_index, caller.resume_marker_jit_pc
            );
        }
        return None;
    }
    if !ctx.trace_ctx.vable_snapshot_buildable() {
        return None;
    }
    Some(parent_frames)
}

fn walker_capture_transparent_helper_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    parent_frames: Vec<InlineParentFrame>,
    guard_stamp: GuardStampTarget,
) -> Result<(), DispatchError> {
    if parent_frames.is_empty() || !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::callee_inline_unsupported(op_pc));
    }
    // A helper-only chain would publish the helper as the jitdriver
    // virtualizable (`consume_vable_info`). Keep that on a Python portal.
    // A Python framestack with no helper levels — including a test skeleton
    // whose `code_ptr` is null — is the same path as main and does not
    // apply this. `walker_capture_snapshot_for_last_guard_impl` already
    // treats `!transparent_helper_subwalk` as `HelperChain::NoHelper`.
    let helper_chain = if ctx.fbw_mode.transparent_helper_subwalk {
        ctx.session.borrow().helper_chain()
    } else {
        HelperChain::NoHelper
    };
    if helper_chain != HelperChain::NoHelper
        && !parent_frames.first().is_some_and(|outer| {
            crate::state::pyjitcode_for_jitcode_index(outer.jitcode_index as i32)
                .is_some_and(|pjc| !pjc.code_ptr.is_null())
        })
    {
        return Err(DispatchError::callee_inline_unsupported(op_pc));
    }
    let saved_vable = publish_outermost_parent_vable_scalars(ctx, &parent_frames, op_pc)?;
    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved_vable);
    // `capture_resumedata` numbers every `MetaInterp.framestack` level in one
    // `ResumeDataLoopMemo` pass (`resume.py` `number`). Each section's tagged
    // values are the live boxes at that frame's published pc, so the length
    // agrees with `get_current_position_info` / `frame_value_count_at` when
    // cranelift rebuilds the numbering. A stored `in_a_call` image whose
    // length differs from that window refuses the capture.
    let mut captured: Vec<(u32, u32, u32, Vec<OpRef>)> = Vec::with_capacity(parent_frames.len());
    for frame in &parent_frames {
        let word = frame
            .resume_marker_jit_pc
            .map(|marker| marker as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let pc_word = crate::state::pyjitcode_for_jitcode_index(frame.jitcode_index as i32)
            .and_then(|payload| {
                payload.resolve_resume_pc_with_jitcode_pc(word, crate::state::op_live())
            })
            .map(|offset| offset as u32)
            .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc })?;
        let py_pc = forward_snapshot_py_pc(frame.jitcode_index, pc_word)?;
        let boxes = boxes_matching_resume_liveness(frame, pc_word, op_pc)?;
        captured.push((frame.jitcode_index, pc_word, py_pc, boxes));
    }
    let frames: Vec<(u32, u32, u32, &[OpRef])> = captured
        .iter()
        .map(|(jitcode_index, pc_word, py_pc, boxes)| {
            (*jitcode_index, *pc_word, *py_pc, boxes.as_slice())
        })
        .collect();
    match guard_stamp {
        GuardStampTarget::LastOp => ctx
            .trace_ctx
            .capture_snapshot_for_last_guard_multi_frame_with_vable_vref(
                &frames,
                &vable_boxes,
                &vref_boxes,
            ),
    }
    Ok(())
}

/// A post-return guard on an inlined operator dunder's result.
///
/// `descroperation.py` checks the box AFTER the call, in the caller
/// level (`format` `isinstance_w`, `len` `_check_len_result`, `is_true`
/// bool-ness). The compiled pin must resume in `OperatorTail` with the
/// result as the tail's pending return (`_setup_return_value_r`), not at
/// the caller's CALL / FORMAT_WITH_SPEC `op.pc` — that coordinate re-runs
/// the dunder. Every guard on the dunder result before the destination
/// write goes through this helper.
pub(crate) fn walker_emit_operator_tail_result_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    tail: crate::operator_continuation::OperatorTail,
    caller: &InlineParentFrame,
    result: OpRef,
    opcode: OpCode,
    args: &[OpRef],
    receiver: Option<OpRef>,
) -> Result<(), DispatchError> {
    let subject = args.first().copied();
    if subject.is_some_and(|arg| arg.is_constant()) {
        if opcode == OpCode::GuardClass
            && let Some(subject) = subject
        {
            ctx.trace_ctx.heap_cache_mut().class_now_known(subject);
        }
        return Ok(());
    }
    // `MIFrame.opimpl_guard_class` records nothing when `is_class_known`.
    // A prior pin already proved `ob_type`, so repeating `GuardClass` is
    // dead weight and would mark the class known before a later typed
    // unbox, suppressing that helper's tagged-pointer test.
    if opcode == OpCode::GuardClass
        && let Some(subject) = subject
        && ctx.trace_ctx.heap_cache().is_class_known(subject)
    {
        return Ok(());
    }
    stamp_guard_value_concrete(ctx.trace_ctx, opcode, args);
    ctx.trace_ctx.record_guard(opcode, args, 0);
    if opcode == OpCode::GuardClass
        && let Some(subject) = subject
    {
        ctx.trace_ctx.heap_cache_mut().class_now_known(subject);
    }
    census_record("OperatorTailResultGuard");
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[operator-tail-result-guard] tail={} opcode={opcode:?}",
            tail.name()
        );
    }
    let mut caller = caller.clone();
    refresh_paused_parent_blackhole(&mut caller);
    let mut tail_frame = super::operator_continuation_parent_frame(tail, receiver)
        .ok_or(DispatchError::callee_inline_abort(op_pc, true))?;
    // r0 is the dunder result. Format keeps the receiver in boxes[1].
    tail_frame.boxes[0] = result;
    // `capture_resumedata` walks every `MetaInterp.framestack` entry.
    // `walker_capture_transparent_helper_snapshot` takes the same
    // outermost-first chain: enclosing paused parents, then the immediate
    // caller, then the operator tail. When FORMAT_WITH_SPEC / len / bool
    // sits inside an already-inlined Python callee, dropping the portal
    // and other ancestors would resume without that callee's caller.
    let mut parent_frames = {
        let session = ctx.session.borrow();
        session
            .framestack
            .iter()
            .flat_map(|frame| frame.parents.iter().cloned())
            .collect::<Vec<_>>()
    };
    parent_frames.push(caller);
    parent_frames.push(tail_frame);
    walker_capture_transparent_helper_snapshot(ctx, op_pc, parent_frames, GuardStampTarget::LastOp)
}

/// Emit a multi-frame inline guard snapshot (#68): the inlined callee's OWN
/// (top/innermost) frame built from the live sub-walk register banks, plus the
/// pre-computed paused caller frame(s) on the walk framestack. Frame
/// order is OUTERMOST-FIRST (`recorder.rs` / `build_resumed_frames`
/// `eval.rs`): the parent chain followed by the callee top frame.  The
/// stale doc on `capture_snapshot_for_last_guard_multi_frame_with_vable_vref`
/// claiming `frames[0]=top` is wrong — the function writes frames verbatim.
pub(crate) fn walker_capture_multi_frame_inline_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    callee_op_pc: usize,
    after_residual_call: bool,
    parent_frames: Vec<InlineParentFrame>,
    scope: GuardCaptureScope,
    // The `_nonstandard_virtualizable` promote guard is not the last recorded op
    // (`emit_force_virtualizable` records GETFIELD_GC / PTR_NE / COND_CALL after
    // it), so its caller stamps the resume position on the last *guard* op.
    // Straight-line / branch inline guards ARE the last op and pass `LastOp`.
    guard_stamp: GuardStampTarget,
) -> Result<(), DispatchError> {
    if ctx.fbw_mode.transparent_helper_subwalk {
        return walker_capture_transparent_helper_snapshot(
            ctx,
            callee_op_pc,
            parent_frames,
            guard_stamp,
        );
    }
    if !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::GuardSnapshotVableUntyped { pc: callee_op_pc });
    }
    // Callee (top/innermost) frame: map the guard's jitcode pc to the callee's
    // own Python pc, read liveness from the live sub-walk register banks.
    let callee_w_code = ctx
        .session
        .borrow()
        .framestack
        .last()
        .map(|frame| frame.w_code);
    let Some(callee_w_code) = callee_w_code else {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    };
    // A continuation level has no Python code object. Its jitcode index is the
    // one the sub-walk was entered with (`InlineCalleeConsts`).
    let callee_jitcode_index = if callee_w_code == 0 {
        ctx.inline_callee_consts
            .as_ref()
            .map(|consts| consts.jitcode_index)
            .filter(|index| *index >= 0)
    } else {
        crate::state::ensure_jitcode_index(callee_w_code as *const ())
    };
    let Some(callee_jitcode_index) = callee_jitcode_index else {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_jitcode_index(callee_jitcode_index) else {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    };
    if callee_pjc.jitcode.code.is_empty()
        || (!callee_pjc.code_ptr.is_null() && !callee_pjc.is_populated())
    {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    }
    // Mirror of the single-frame path: the callee (top) frame always carries
    // the resume-marker twin for its requested resume coordinate. For guards,
    // the callee payload supplies the resume-marker twin: after-residual
    // guards use the fallthrough twin and retain the sentinel on a miss,
    // while plain guards retain the raw `callee_op_pc` on a miss. A branch
    // guard's `callee_op_pc` is already the jump's `-live-`
    // (`generate_guard(resumepc=orgpc)`). Computed before box collection so
    // encoder liveness and decoder resume use the same coordinate.
    // `GuardCaptureScope::carried_resume_jit_pc` is already the post-call
    // `-live-` (`pyjitpl.py` `generate_guard` / `capture_resumedata`,
    // `after_residual_call=True`, `resumepc=-1` leaves `MIFrame.pc` after
    // the call). `resume_marker_for_jitcode_pc` is the opcode-start
    // `-live-`; resuming there re-executes the call from the `in_a_call`
    // image whose operands `get_list_of_active_boxes` has cleared.
    let callee_jitcode_pc: i32 = if let Some(carried) = scope.carried_resume_jit_pc {
        carried as i32
    } else {
        match after_residual_call {
            true => after_residual_guard_marker(&callee_pjc, callee_op_pc, None)
                .or_else(|| {
                    // Fallback only, for the same reason as the single-frame path:
                    // the sticky cursor names this frame's op only while no other
                    // residual has passed, so it may not displace the twin.
                    (ctx.live_after_jit_pc != usize::MAX
                        && callee_pjc
                            .jitcode
                            .can_decode_live_vars(ctx.live_after_jit_pc, crate::state::op_live()))
                    .then_some(ctx.live_after_jit_pc)
                })
                .map(|m| m as i32)
                .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC),
            // `get_list_of_active_boxes` (`pyjitpl.py`): a plain guard reads
            // `pc = self.pc - SIZE_LIVE_OP`. A branch guard's `callee_op_pc`
            // is already that `-live-` (`generate_guard(resumepc=orgpc)`);
            // [`plain_guard_resume_jitcode_pc`] keeps it instead of walking
            // three bytes earlier onto the opcode-start twin.
            false => plain_guard_resume_jitcode_pc(&callee_pjc, callee_op_pc),
        }
    };
    let mf_diag = fbw_mf_diag_enabled();
    let recipe_resultcolor_audit = pcmap_recipe_resultcolor_audit_enabled();
    if (mf_diag || recipe_resultcolor_audit) && !callee_pjc.code_ptr.is_null() {
        let callee_py_pc = unsafe {
            let code = &*callee_pjc.code_ptr;
            let mut py = crate::py_coord::containing_py_pc_for_jitcode_pc(
                &callee_pjc.metadata,
                callee_op_pc,
            );
            py = skip_python_trivia_forward(code, py as usize) as u32;
            if after_residual_call {
                py = crate::pyjitpl::semantic_fallthrough_pc(code, py as usize) as u32;
            }
            py
        };
        if recipe_resultcolor_audit {
            // The snapshot selects this native resume coordinate below. For
            // non-branch captures, its marker construction already applies
            // the same trivia (and, after a residual call, semantic
            // fallthrough) transform as the diagnostic inversion above.
            let (site, native_marker) = match after_residual_call {
                true => (
                    "mf_callee_inversion_after_residual",
                    callee_pjc.after_residual_marker_for_jitcode_pc(callee_op_pc),
                ),
                false => (
                    "mf_callee_inversion_plain",
                    callee_pjc.resume_marker_for_jitcode_pc(callee_op_pc),
                ),
            };
            pcmap_recipe_resultcolor_audit_probe(site, "fire");
            let verdict = match native_marker {
                Some(marker) => {
                    let native_py = crate::py_coord::containing_py_pc_for_jitcode_pc(
                        &callee_pjc.metadata,
                        marker,
                    ) as usize;
                    if native_py == callee_py_pc as usize {
                        "eq"
                    } else {
                        "di"
                    }
                }
                None => "native_miss",
            };
            pcmap_recipe_resultcolor_audit_probe(site, verdict);
        }
        if mf_diag {
            let pcdep = callee_pjc
                .pcdep_for_jitcode_pc(callee_op_pc)
                .unwrap_or_default();
            let depth = callee_pjc
                .depth_for_jitcode_pc_pred(callee_op_pc)
                .unwrap_or(0);
            // Read the coordinate the box collection below actually uses, not
            // the raw `callee_op_pc`; and read it through the
            // decline-reporting form, since the lossy query maps an
            // unresolvable coordinate and a genuinely empty window onto the
            // same all-empty result.
            let resolved = crate::state::try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
                callee_jitcode_index as i32,
                callee_jitcode_pc,
            );
            let window = if resolved.is_some() {
                "resolved"
            } else {
                "UNRESOLVED"
            };
            let banks = resolved.unwrap_or_default();
            eprintln!(
                "[fbw-mf-diag] callee jc={callee_jitcode_index} op_pc={callee_op_pc} \
             py_pc={callee_py_pc} after_residual={after_residual_call} depth={depth} \
             pcdep_color_slots={pcdep:?}"
            );
            // The color→slot inversion must be read at the same coordinate as
            // the banks above; `pcdep` at the raw `callee_op_pc` can name a
            // different set, which is what makes a live color look unmapped.
            let coord_pcdep = usize::try_from(callee_jitcode_pc)
                .ok()
                .and_then(|coord| callee_pjc.pcdep_for_jitcode_pc(coord))
                .unwrap_or_default();
            eprintln!(
                "[fbw-mf-diag]   live banks: i={:?} r={:?} f={:?} coord={callee_jitcode_pc} \
                 window={window} coord_pcdep={coord_pcdep:?}",
                banks.int, banks.ref_, banks.float,
            );
            for &c in &banks.ref_ {
                eprintln!(
                    "[fbw-mf-diag]   regs_r[{c}] = {:?}",
                    ctx.registers_r.get(c as usize)
                );
            }
            for &c in &banks.int {
                eprintln!(
                    "[fbw-mf-diag]   regs_i[{c}] = {:?}",
                    ctx.registers_i.get(c as usize)
                );
            }
            if !callee_pjc.code_ptr.is_null() {
                let code = unsafe { &*callee_pjc.code_ptr };
                let lo = callee_py_pc.saturating_sub(3);
                for py in lo..callee_py_pc + 5 {
                    if let Some((instr, arg)) =
                        pyre_interpreter::decode_instruction_at(code, py as usize)
                    {
                        let mark = if py == callee_py_pc {
                            " <== resume"
                        } else {
                            ""
                        };
                        eprintln!("[fbw-mf-diag]   py{py}: {instr:?} arg={arg:?}{mark}");
                    }
                }
            }
        }
    }
    // `MIFrame.registers_r` is the live color bank, but a reconstructed
    // carrier frame also owns the semantic locals/stack image from which that
    // bank was rebuilt.  A post-call liveness marker can conservatively carry
    // a color whose current bank value is NONE (adjacent `-live-` union); map
    // that color back to this callee's OWN frame slot and source the box from
    // its frame-local shadow.  This is the inline-frame counterpart of
    // `collect_outer_active_boxes`' virtualizable-slot recovery, and preserves
    // the one-red-frame-per-MIFrame shape instead of falling back to the root.
    let mut recovered_regs_r = ctx.registers_r.to_vec();
    let stack_base = if callee_pjc.code_ptr.is_null() {
        0
    } else {
        let code = unsafe { &*callee_pjc.code_ptr };
        crate::state::callee_layout_for_call_assembler(code).0
    };
    let maps =
        crate::state::bridge_semantic_maps_from_jitcode_pc(callee_jitcode_index, callee_jitcode_pc);
    if recovered_regs_r.iter().any(|value| value.is_none()) && !callee_pjc.code_ptr.is_null() {
        if let Some(shadow) = ctx.frame_state.borrow().callee_shadow.as_ref() {
            for (color, value) in recovered_regs_r.iter_mut().enumerate() {
                if !value.is_none() {
                    continue;
                }
                let Some(slot) = crate::state::semantic_ref_slot_for_reg_color(
                    stack_base,
                    maps.stack_depth_at_pc,
                    &maps.pcdep_entries,
                    color,
                ) else {
                    continue;
                };
                if let Some(&slot_value) = shadow.opref.get(&(slot as i64)) {
                    *value = slot_value;
                }
            }
        }
        // A dense fallthrough marker can name the NEXT Python opcode's result
        // color before that opcode has executed.  RPython's sparse stream has
        // no opcode-entry marker at this seam; the translated Ref bank slot is
        // still its empty/null value and the next opcode overwrites it before
        // any read.  Admit exactly that metadata-proven result color as NULL;
        // every other missing live color still declines below.
        if after_residual_call {
            let marker_result = usize::try_from(callee_jitcode_pc)
                .ok()
                .and_then(|coord| callee_pjc.result_color_trivia_for_jitcode_pc(coord))
                .filter(|&color| color != u16::MAX);
            let null_ref = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
            seed_missing_fallthrough_result(&mut recovered_regs_r, marker_result, null_ref);
        }
    }
    let callee_boxes = collect_callee_active_boxes(
        ctx.registers_i,
        &recovered_regs_r,
        ctx.registers_f,
        callee_jitcode_index as u32,
        callee_op_pc,
        callee_jitcode_pc,
    )?;

    // Publish the OUTERMOST caller's vable scalars for its resume coordinate so
    // the resume reader restores the caller's `PyFrame` at the CALL return
    // point rather than the stale loop-header seed the walker never crosses
    // `set_orgpc` to update.
    let saved_vable = if !parent_frames.is_empty() {
        Some(publish_outermost_parent_vable_scalars(
            ctx,
            &parent_frames,
            callee_op_pc,
        )?)
    } else {
        None
    };

    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    if let Some(saved) = saved_vable {
        crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved);
    }

    // Frame tuples, OUTERMOST-FIRST: the paused caller chain, then the callee
    // top frame last (innermost).
    let mut frames: Vec<(u32, u32, u32, &[OpRef])> = Vec::with_capacity(parent_frames.len() + 1);
    for pf in &parent_frames {
        let pf_word = pf
            .resume_marker_jit_pc
            .map(|m| m as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let Some(pf_pc_word) = crate::state::pyjitcode_for_jitcode_index(pf.jitcode_index as i32)
            .and_then(|payload| {
                let resolved =
                    payload.resolve_resume_pc_with_jitcode_pc(pf_word, crate::state::op_live());
                resolved
            })
            .map(|offset| offset as u32)
        else {
            return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: callee_op_pc });
        };
        let pf_py_pc = forward_snapshot_py_pc(pf.jitcode_index, pf_pc_word)?;
        frames.push((pf.jitcode_index, pf_pc_word, pf_py_pc, pf.boxes.as_slice()));
    }
    // `descr_call`'s tail needs no case of its own here: the constructor
    // inline records it among `__init__`'s level's `parents`, so the loop
    // above already emitted it — at its own depth, however deep the chain
    // below it goes.  See `crate::ctor_continuation`.
    let callee_py_pc =
        forward_snapshot_py_pc(callee_jitcode_index as u32, callee_jitcode_pc as u32)?;
    frames.push((
        callee_jitcode_index as u32,
        callee_jitcode_pc as u32,
        callee_py_pc,
        callee_boxes.as_slice(),
    ));

    match guard_stamp {
        GuardStampTarget::LastOp => ctx
            .trace_ctx
            .capture_snapshot_for_last_guard_multi_frame_with_vable_vref(
                &frames,
                &vable_boxes,
                &vref_boxes,
            ),
    }
    Ok(())
}

/// Seed only the not-yet-defined result color carried by pyre's dense
/// fallthrough marker.  Returns whether a hole was filled.
pub(crate) fn seed_missing_fallthrough_result(
    registers_r: &mut [OpRef],
    marker_result: Option<u16>,
    null_ref: OpRef,
) -> bool {
    let Some(dst) = marker_result
        .filter(|&color| color != u16::MAX)
        .map(usize::from)
    else {
        return false;
    };
    let Some(slot) = registers_r.get_mut(dst) else {
        return false;
    };
    if !slot.is_none() {
        return false;
    }
    *slot = null_ref;
    true
}
