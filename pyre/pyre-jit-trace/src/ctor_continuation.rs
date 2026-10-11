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
//! resume coordinate: three ops, entered only by resume, never compiled.
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
//!               residual_call_r_v  bh_check_init_returned_none(init_result)
//!               ref_return         instance
//! ```
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
/// would be a signature mismatch.  The exception reaches the blackhole through
/// `BH_LAST_EXC_VALUE`, which `handler_residual_call_r_v` zeroes before the
/// call and tests after it.  The compiled-code channel (`store_jit_exception`)
/// is deliberately not written: this jitcode is a resume coordinate, never a
/// compilation unit.
pub extern "C" fn bh_check_init_returned_none(result: pyre_object::PyObjectRef) {
    if let Err(mut err) = pyre_interpreter::call::check_init_returned_none(result) {
        let exc_obj = err.to_exc_object();
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
    }
}

thread_local! {
    static LEVEL: std::cell::OnceCell<Option<(i32, usize)>> =
        const { std::cell::OnceCell::new() };
}

/// `(jitcode index, resume pc)` of the shared tail, built on first use.
///
/// Thread-local because `MetaInterpStaticData` is: the index it hands back
/// names a slot in this thread's `jitcodes`.
fn level() -> Option<(i32, usize)> {
    LEVEL.with(|cell| *cell.get_or_init(build))
}

/// The jitcode index of the tail, or `None` when it could not be built.
pub(crate) fn jitcode_index() -> Option<i32> {
    level().map(|(index, _)| index)
}

/// Whether `index` names the tail, WITHOUT building it.
///
/// A reader that only asks "is this resumed level the tail?" must not be the
/// thing that mints it: [`level`] installs a jitcode into
/// `MetaInterpStaticData.jitcodes`, so calling it from a decode path would
/// append that slot in every process that ever decodes a resume section,
/// shifting the index space for programs that never inline a constructor.
pub(crate) fn is_installed_level(index: i32) -> bool {
    LEVEL.with(|cell| matches!(cell.get(), Some(&Some((installed, _))) if installed == index))
}

/// The byte offset the tail's resume section must name.
pub(crate) fn resume_pc() -> Option<usize> {
    level().map(|(_, pc)| pc)
}

fn build() -> Option<(i32, usize)> {
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

    let funcptr = bh_check_init_returned_none as *const () as i64;
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
    Some((index, resume_pc))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The anchor must sit immediately behind the complete `inline_call`, so
    /// `_setup_return_value_r` can read `code[position-1]` as the dest.
    #[test]
    fn resume_anchor_sits_behind_the_inline_call_return_tail() {
        let Some((index, resume_pc)) = level() else {
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
        // only the entry one, so the tail's three remaining ops need theirs
        // too: `inline_call`, `-live-`, `residual_call_r_v`, `ref_return`.
        assert_eq!(
            startpoints.len(),
            4,
            "startpoints must cover every op in the tail, got {startpoints:?}",
        );
    }

    /// The generic paused-level loop resolves every parent's resume offset
    /// through `resolve_resume_pc_with_jitcode_pc`, so the tail must answer it
    /// like any other level — otherwise recording the tail as an ordinary
    /// parent aborts the guard with `GuardResumeCoordinateUnavailable`.
    #[test]
    fn the_anchor_resolves_as_an_ordinary_parent_resume_coordinate() {
        let Some((index, resume_pc)) = level() else {
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
