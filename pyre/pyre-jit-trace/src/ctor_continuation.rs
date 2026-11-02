//! `typeobject.py descr_call`'s JIT-visible tail, as a resume level of its own.
//!
//! Upstream, a guard inside an inlined `__init__` resumes into a framestack
//! that still holds `descr_call`'s own frame: `capture_resumedata`
//! (`pyjitpl.py`) hands the whole `self.framestack` to the trace, and
//! `convert_and_run_from_pyjitpl` (`blackhole.py`) copies every frame into a
//! chained blackhole interpreter.  `descr_call` is an ordinary graph there, so
//! the three things it does after the call —
//!
//!   1. discard `__init__`'s result,
//!   2. raise `TypeError` unless that result was `None`,
//!   3. return `w_newobject`,
//!
//! — are just its jitcode, and the blackhole runs them with no special case.
//!
//! Pyre cannot reuse that graph, and this module is the ADAPTATION that
//! stands in for it.  The counterpart of `descr_call` is
//! `pyre-interpreter/src/call.rs type_descr_call_impl`, and two independent
//! things keep it from playing the level:
//!
//!   1. Pyre never *enters* it.  `try_walker_inline_type_call`
//!      (`jitcode_dispatch/inline_call.rs`) recognises `C(...)` while decoding
//!      the CALL and synthesises the body itself — it resolves `__new__` and
//!      `__init__` off the type, emits the allocation through
//!      `helpers::emit_instance_inline`, and inlines only `__init__` as a user
//!      call.  No pc inside `type_descr_call_impl` is ever current, so no
//!      resume coordinate can name one.  This is the same divergence
//!      `find_all_graphs_bfs` already documents for the builtin gateways:
//!      pyre's opcode walker lowers a Python CALL directly instead of through
//!      the source-level dispatch graph.
//!   2. The graph is not in the frozen table either.  The build-time
//!      `find_all_graphs` BFS from the `eval::eval_loop_jit` portal stops
//!      above it — `call_function_impl_raw` is present, its callee
//!      `call_function_impl_result` is not — so the whole type-call spine
//!      below that cut is absent.
//!
//! CONVERGENCE PATH: (2) is a seed away — `find_all_graphs_bfs` already takes
//! explicit extra seeds — but fixing it alone changes nothing, because (1) is
//! what decides that the graph is never run.  Converging means retiring the
//! opcode-level recogniser and inlining `type_descr_call_impl`'s generated
//! jitcode as an ordinary callee frame, the way upstream traces through
//! `descr_call`.  Then `__init__` is a frame inside a frame, the discard and
//! the None check are that graph's own jitcode, and this module deletes.
//! Until then the stand-in is deliberately the smallest thing that can hold a
//! resume coordinate: the placeholder call, two `-live-` anchors, the None
//! check, the after-call `-live-` (`handle_residual_call`), and `ref_return`,
//! entered only by resume, never compiled.
//!
//! Without such a level the constructor route cannot be seeded at all.  A
//! blackhole level's return kind is decided by the `*_return` op the callee
//! executes, so a seeded `__init__` returns `Ref` and `_setup_return_value_r`
//! (`blackhole.py`) writes its `None` into the caller's call-result register —
//! the register that has to hold the instance.  That is why the constructor
//! inline used to pin itself to the caller boundary instead.
//!
//! The level is recorded the way upstream's is: as one of the paused levels
//! on `__init__`'s framestack entry (`InlineFrame::parents`, outermost-first),
//! not as a value the snapshot splices in at a fixed offset.  That is what
//! keeps it at its own depth when `__init__` itself inlines a callee — the
//! chain becomes `caller -> tail -> __init__ -> callee`, which is exactly what
//! `capture_resumedata` would hand over upstream.
//!
//! So this module builds the tail, and only the tail, as its own jitcode:
//!
//! ```text
//!   0:          inline_call_r_r <placeholder>, self -> init_result
//!   resume_pc:  -live-                       [r0 live, r1 pending]
//!   assembler:  -live-                       [r0 live, r1 live]
//!               residual_call_r_v  bh_check_init_returned_none(init_result)
//!               -live-                       [r0 live]
//!               ref_return         instance
//! ```
//!
//! `resume_pc` is the in-call coordinate: a guard inside `__init__` keeps this
//! level paused with `r1` pending so `_setup_return_value_r` fills it when
//! `__init__` returns. The assembler coordinate is the same tail after
//! `CALL_ASSEMBLER` has already produced that result (`do_residual_call`
//! writes dest before `GUARD_NOT_FORCED`): both registers are live, and a
//! force resumes here so the None check still runs.
//!
//! The level is only ever *resumed*, never entered from the top:
//! `blackhole_from_resumedata` sets each level's `position` from its resume
//! section, and `run_this_frame` dispatches forward from there.  The
//! `inline_call` before the anchor therefore never decodes — it exists because
//! `_setup_return_value_r` reads `code[position-1]`, the call's result
//! register. Its callee operand is a placeholder for the same reason: the
//! instruction is never executed, and the real callee varies per class while
//! this jitcode is shared.

use crate::PyJitCode;
use majit_metainterp::jitcode::{JitCallArg, JitCodeBuilder};

/// The instance `__new__` produced, supplied by this level's resume section.
/// `descr_call`'s `w_newobject`.
pub(crate) const INSTANCE_REG: u16 = 0;

/// Where the inlined `__init__`'s return lands when it leaves the blackhole
/// (`_setup_return_value_r` → `call_result_reg`).  `descr_call`'s `w_result`.
/// Never carried by the resume section: `get_list_of_active_boxes(in_a_call=
/// True)` (`pyjitpl.py`) clears a caller's pending call-result slot, and this
/// register is exactly that slot.
pub(crate) const INIT_RESULT_REG: u16 = 1;

/// Callee operand of the never-decoded `inline_call`.  See the module doc:
/// `_setup_return_value_r` only reads the last operand as the dest register.
const PLACEHOLDER_CALLEE: u16 = 0;

/// `typeobject.py descr_call`: `if not space.is_w(w_result, space.w_None):
/// raise oefmt(space.w_TypeError, "__init__() should return None")`.
///
/// Returns nothing, matching the `'v'` result the calldescr declares —
/// `bh_call_v_dispatch` (`majit-backend/src/call_stub.rs`) transmutes the
/// address to a `()`-returning `extern "C"` fn, so any other Rust return type
/// would be a signature mismatch.  A refusal publishes through
/// `ResidualError::publish_residual` (`jit_publish_residual_error`):
/// `BH_LAST_EXC_VALUE` for the blackhole handler and the compiled
/// `GUARD_NO_EXCEPTION` / `GUARD_EXCEPTION` cell.  The walker records this
/// helper after a loop-callee `CALL_ASSEMBLER` so a non-None `__init__`
/// result raises as `descr_call`'s outcome instead of aborting into an
/// entry replay.
pub extern "C" fn bh_check_init_returned_none(result: pyre_object::PyObjectRef) {
    if let Err(err) = pyre_interpreter::call::check_init_returned_none(result) {
        use majit_ir::helper_fnaddr::ResidualError;
        err.publish_residual();
    }
}

thread_local! {
    static LEVEL: std::cell::OnceCell<Option<(i32, usize, usize)>> =
        const { std::cell::OnceCell::new() };
}

/// `(jitcode index, in-call resume pc, assembler resume pc)` of the shared
/// tail, built on first use.
///
/// Thread-local because `MetaInterpStaticData` is: the index it hands back
/// names a slot in this thread's `jitcodes`.
fn level() -> Option<(i32, usize, usize)> {
    LEVEL.with(|cell| *cell.get_or_init(build))
}

/// The jitcode index of the tail, or `None` when it could not be built.
pub(crate) fn jitcode_index() -> Option<i32> {
    level().map(|(index, _, _)| index)
}

/// Whether `index` names the tail, WITHOUT building it.
///
/// A reader that only asks "is this resumed level the tail?" must not be the
/// thing that mints it: [`level`] installs a jitcode into
/// `MetaInterpStaticData.jitcodes`, so calling it from a decode path would
/// append that slot in every process that ever decodes a resume section,
/// shifting the index space for programs that never inline a constructor.
pub(crate) fn is_installed_level(index: i32) -> bool {
    LEVEL.with(|cell| matches!(cell.get(), Some(&Some((installed, _, _))) if installed == index))
}

/// The byte offset the tail's in-call resume section must name.
pub(crate) fn resume_pc() -> Option<usize> {
    level().map(|(_, pc, _)| pc)
}

/// The byte offset a constructor `CALL_ASSEMBLER` `GUARD_NOT_FORCED` names:
/// `__init__` has already returned, so `INIT_RESULT_REG` is live.
pub(crate) fn assembler_resume_pc() -> Option<usize> {
    level().map(|(_, _, pc)| pc)
}

fn build() -> Option<(i32, usize, usize)> {
    let mut builder = JitCodeBuilder::new();
    builder.set_name("descr_call_tail");

    // Never decoded; present so `_setup_return_value_r` can read
    // `code[position-1]` as the dest register. See the module doc.
    builder.inline_call_r_r(
        PLACEHOLDER_CALLEE,
        &[(INSTANCE_REG, INSTANCE_REG)],
        Some(INIT_RESULT_REG),
    );

    let resume_pc = builder.current_pos();
    // `-live-` operands are 2-byte offsets into the ONE shared
    // `MetaInterpStaticData.liveness_info` pool (`pyjitpl.py`), so the
    // triple is interned there rather than in a private assembler buffer.
    let liveness_offset = crate::state::intern_liveness(&[], &[INSTANCE_REG as u8], &[])?;
    let live_patch = builder.live_placeholder();
    builder.patch_live_offset(live_patch, liveness_offset);

    // After `CALL_ASSEMBLER`, `_setup_return_value_r` has already run: both
    // the instance and `__init__`'s result are live, and a force resumes here
    // so `bh_check_init_returned_none` still executes. `bhimpl_live` is a
    // marker skip, so an in-call resume that falls through this second
    // `-live-` keeps the result `_setup_return_value_r` just wrote.
    let assembler_resume_pc = builder.current_pos();
    let assembler_liveness =
        crate::state::intern_liveness(&[], &[INSTANCE_REG as u8, INIT_RESULT_REG as u8], &[])?;
    let assembler_live_patch = builder.live_placeholder();
    builder.patch_live_offset(assembler_live_patch, assembler_liveness);

    let funcptr = pyre_interpreter::residual_word_addr!(1, bh_check_init_returned_none) as i64;
    let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr {
        // One `Ref` argument, no result: the same signature
        // `bh_check_init_returned_none` is declared with.
        arg_classes: "r".to_string(),
        result_type: 'v',
        ..Default::default()
    };
    builder.residual_call_void_canonical_typed_args(
        funcptr,
        &[JitCallArg {
            kind: majit_metainterp::jitcode::JitArgKind::Ref,
            reg: INIT_RESULT_REG,
        }],
        calldescr,
    );

    // `handle_residual_call` emits `[op1, SpaceOperation('-live-', [], None)]`
    // whenever `calldescr_canraise(calldescr)`. `bh_check_init_returned_none`
    // can raise TypeError, so this is the after-call resume coordinate
    // (`after_residual_guard_marker`). Live refs are the instance only:
    // `INIT_RESULT_REG` is dead after the check; `ref_return` reads
    // `INSTANCE_REG`.
    let after_call_liveness = crate::state::intern_liveness(&[], &[INSTANCE_REG as u8], &[])?;
    let after_call_live_patch = builder.live_placeholder();
    builder.patch_live_offset(after_call_live_patch, after_call_liveness);

    // `descr_call` returns `w_newobject`, not `__init__`'s result.
    builder.ref_return(INSTANCE_REG);

    // `try_finish` carries the builder's own `startpoints`, which every emit
    // helper above has already populated through `start_instr`.  The set is
    // every instruction start, not just the addressable ones: `run_inner`
    // asserts membership on EVERY dispatched position under
    // `jit_strict_mode`, so narrowing it would panic at the first op after
    // the resume anchor.
    let jitcode = builder.try_finish()?;

    let payload = std::sync::Arc::new(PyJitCode::from_core_degenerate(
        std::sync::Arc::new(jitcode),
        std::ptr::null(),
        /* has_abort */ false,
    ));
    let index = crate::state::install_codeless_jitcode(payload);
    Some((index, resume_pc, assembler_resume_pc))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The anchor must sit immediately behind the complete `inline_call`, so
    /// `_setup_return_value_r` can read `code[position-1]` as the dest.
    #[test]
    fn resume_anchor_sits_behind_the_inline_call_return_tail() {
        let Some((index, resume_pc, _)) = level() else {
            return;
        };
        let payload = crate::state::pyjitcode_for_jitcode_index(index)
            .expect("the tail was just installed at this index");
        let code = payload.jitcode.code.as_slice();
        assert!(
            resume_pc >= 3,
            "no room for the call header and its return register"
        );
        assert_eq!(
            code[resume_pc - 1] as u16,
            INIT_RESULT_REG,
            "the last operand is the ref result register",
        );
        assert_eq!(
            code[resume_pc],
            crate::state::op_live(),
            "the resume position must be the `-live-` anchor itself",
        );
        let startpoints = payload
            .jitcode
            .startpoints
            .as_ref()
            .expect("the builder records one startpoint per emitted instruction");
        assert!(
            startpoints.contains(&resume_pc),
            "the anchor must be addressable as a resume coordinate",
        );
        // `run_inner` asserts membership on every dispatched position, not
        // only the entry one, so the tail's ops need theirs too: `inline_call`,
        // in-call `-live-`, assembler `-live-`, `residual_call_r_v`,
        // after-call `-live-`, `ref_return`.
        assert_eq!(
            startpoints.len(),
            6,
            "startpoints must cover every op in the tail, got {startpoints:?}",
        );
    }

    /// The generic paused-level loop resolves every parent's resume offset
    /// through `resolve_resume_pc_with_jitcode_pc`, so the tail must answer it
    /// like any other level — otherwise recording the tail as an ordinary
    /// parent aborts the guard with `GuardResumeCoordinateUnavailable`.
    #[test]
    fn the_anchor_resolves_as_an_ordinary_parent_resume_coordinate() {
        let Some((index, resume_pc, _)) = level() else {
            return;
        };
        let payload = crate::state::pyjitcode_for_jitcode_index(index)
            .expect("the tail was just installed at this index");
        assert!(
            payload
                .jitcode
                .can_decode_live_vars(resume_pc, crate::state::op_live()),
            "the anchor must decode its live vars",
        );
        assert_eq!(
            payload.resolve_resume_pc_with_jitcode_pc(resume_pc as i32, crate::state::op_live()),
            Some(resume_pc),
            "the parent loop must resolve the anchor to itself",
        );
    }

    /// `handle_residual_call` trails a raising residual with `-live-`. The
    /// None check is such a residual, and `ref_return` then reads only the
    /// instance, so that marker's live refs are `[INSTANCE_REG]`.
    #[test]
    fn residual_call_is_followed_by_a_live_naming_the_instance() {
        let Some((index, _, _)) = level() else {
            return;
        };
        let payload = crate::state::pyjitcode_for_jitcode_index(index)
            .expect("the tail was just installed at this index");
        let ops: Vec<_> =
            crate::jitcode_runtime::decoded_ops(payload.jitcode.code.as_slice()).collect();
        let residual_idx = ops
            .iter()
            .position(|op| op.opname.starts_with("residual_call"))
            .expect("the tail emits the None-check residual");
        let after = ops
            .get(residual_idx + 1)
            .expect("the residual has a following op");
        assert_eq!(
            after.opname, "live",
            "handle_residual_call emits a trailing `-live-`",
        );
        let startpoints = payload
            .jitcode
            .startpoints
            .as_ref()
            .expect("the builder records one startpoint per emitted instruction");
        assert!(
            startpoints.contains(&after.pc),
            "the after-call `-live-` must be addressable as a resume coordinate",
        );
        assert!(
            payload
                .jitcode
                .can_decode_live_vars(after.pc, crate::state::op_live()),
            "the after-call `-live-` must decode its live vars",
        );
        let live = crate::state::try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
            index,
            after.pc as i32,
        )
        .expect("the after-call `-live-` must decode");
        assert_eq!(
            live.int,
            Vec::<u32>::new(),
            "no int is live after the check"
        );
        assert_eq!(
            live.ref_,
            vec![INSTANCE_REG as u32],
            "only INSTANCE_REG is live after the check",
        );
        assert_eq!(
            live.float,
            Vec::<u32>::new(),
            "no float is live after the check",
        );
        assert_eq!(
            crate::state::frame_value_count_at(index, after.pc as i32),
            1,
            "only INSTANCE_REG is live after the check",
        );
    }

    /// A constructor `CALL_ASSEMBLER` force resumes with `__init__`'s result
    /// already in `INIT_RESULT_REG`, so that coordinate's `-live-` names both
    /// ref registers.
    #[test]
    fn assembler_resume_names_the_instance_and_the_init_result() {
        let Some((index, _, assembler_pc)) = level() else {
            return;
        };
        let payload = crate::state::pyjitcode_for_jitcode_index(index)
            .expect("the tail was just installed at this index");
        assert!(
            payload
                .jitcode
                .can_decode_live_vars(assembler_pc, crate::state::op_live()),
            "the assembler anchor must decode its live vars",
        );
        assert_eq!(
            crate::state::frame_value_count_at(index, assembler_pc as i32),
            2,
            "instance and __init__ result are both live after CALL_ASSEMBLER",
        );
    }

    /// A non-`None` `__init__` result raises through the channel the blackhole
    /// tests after a residual call, and a `None` result leaves it clear.
    #[test]
    fn the_none_check_publishes_through_the_blackhole_exception_channel() {
        let cell = &majit_metainterp::blackhole::BH_LAST_EXC_VALUE;
        cell.with(|c| c.set(0));
        bh_check_init_returned_none(pyre_object::w_none());
        assert_eq!(cell.with(|c| c.get()), 0, "a None result must not raise");

        bh_check_init_returned_none(pyre_object::w_int_new(1));
        assert_ne!(
            cell.with(|c| c.get()),
            0,
            "a non-None result must publish the TypeError where \
             `handler_residual_call_r_v` reads it",
        );
        cell.with(|c| c.set(0));
    }
}
