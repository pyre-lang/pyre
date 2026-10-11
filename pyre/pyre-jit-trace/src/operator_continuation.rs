//! The JIT-visible TAIL of an operator that post-processes a dunder's result,
//! as a resume level of its own.
//!
//! `crate::ctor_continuation` solves this for `descr_call`; read its module
//! doc first, because the mechanism, the register discipline and the
//! "entered only by resume" contract are identical.  This module generalises
//! it from one operator to a table of them.
//!
//! The defect it answers: a fold that inlines an app-level dunder under an
//! operator is sound only while the operator's result IS the callee's return
//! value.  `a[k]` through `__getitem__`, `GET_ITER` through `__iter__` and a
//! property's `fget` are identity, so they need nothing.  `len(o)` is not —
//! `operation.py len` is
//!
//!   1. `w_res = space._len(w_obj)`        (the `__len__` call),
//!   2. `w_index = space.index(w_res)`,
//!   3. `space._check_len_result(w_index)` (>= 0, and it fits a machine word),
//!   4. answer `space.newint` of that machine length.
//!
//! The fold emits 2-4 into the TRACE, and a guard failure inside the inlined
//! `__len__` never reaches them: `BhFrame::call_result_reg` is
//! `code[position - 1]`, so `_setup_return_value_r` (`blackhole.py`) writes
//! the callee's raw return straight into the result register of the CALLER's
//! `len` residual, and the caller resumes at `call.next_pc` — past the whole
//! operator.  `len(o)` then answers `-1` instead of raising `ValueError`, and
//! `True` instead of `1`.
//!
//! Upstream has no such gap because `len` is an ordinary graph on the
//! framestack, so `capture_resumedata` (`pyjitpl.py`) hands it over with
//! everything else and the chained blackhole simply runs steps 2-4.  Pyre's
//! walker recognises the operator at the opcode instead of entering that
//! graph, exactly as `try_walker_inline_type_call` recognises `C(...)`, so
//! the tail has to be spelled out:
//!
//! ```text
//!   0:          inline_call_r_r <placeholder> -> dunder_result
//!   resume_pc:  -live-                        [nothing live, r0 pending]
//!               residual_call_r_r  bh_len_tail(dunder_result) -> answer
//!               -live-                        [nothing live]
//!               ref_return         answer
//! ```
//!
//! The second `-live-` is the resume coordinate of the residual's exception
//! guard. Nothing is live there: a failing guard carries the exception, and
//! `ref_return` runs only on the trace that passed the guard.
//!
//! Pushed as a paused parent level under the caller (`InlineFrame::parents`,
//! outermost-first), so the chain is `caller -> tail -> __len__ -> ...` and
//! the tail keeps its own depth when the dunder body inlines a callee of its
//! own.  The never-decoded `inline_call` exists only so `code[position - 1]`
//! names the register the finished dunder's return lands in; its callee
//! operand is a placeholder because the instruction never executes and the
//! real callee varies per class while this jitcode is shared.
//!
//! The tail's own `ref_return` then feeds the CALLER's `call_result_reg` —
//! the `len` residual's destination — with the operator's finished answer,
//! which is what makes the deopt observationally identical to never having
//! inlined anything.
//!
//! CONVERGENCE PATH: the same as `ctor_continuation`'s.  Entering
//! `builtins::builtin_len`'s generated jitcode as an ordinary callee frame,
//! with the `__len__` call inlined at the `get_and_call_function` inside it,
//! makes steps 2-4 that graph's own jitcode and deletes this module.  The
//! blocker is documented on `try_walker_orthodox_descent`: a mid-descent
//! decline does not rewind an effect the sub-walk already executed.

use crate::PyJitCode;
use majit_metainterp::jitcode::{JitCallArg, JitCodeBuilder};

/// Which operator tail a level plays.
///
/// One variant per `(operator, result bank)` pair, because the level's
/// closing `*_return` has to match the bank of the caller's own call-result
/// register — `_setup_return_value_i` and `_setup_return_value_r` write
/// different banks of the same frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OperatorTail {
    /// `descroperation.py len`: `space.index` then `_check_len_result` over
    /// what `__len__` returned, answering the `space.index` box.
    Len,
    /// `descroperation.py is_true` after `__bool__`: the bool object becomes
    /// the int `opcode_ops::truth_value` leaves in the Truth residual.
    Truth,
    /// `descroperation.py format` after `__format__`: the returned box must
    /// be a `str` (or subclass); anything else is a TypeError, and a str
    /// subclass keeps its identity.
    Format,
}

impl OperatorTail {
    const ALL: [OperatorTail; 3] = [OperatorTail::Len, OperatorTail::Truth, OperatorTail::Format];

    fn slot(self) -> usize {
        match self {
            OperatorTail::Len => 0,
            OperatorTail::Truth => 1,
            OperatorTail::Format => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            OperatorTail::Len => "len_tail",
            OperatorTail::Truth => "truth_bool_tail",
            OperatorTail::Format => "format_tail",
        }
    }

    /// The host function the level's residual call runs, and the result bank
    /// its `*_return` and the caller's call-result register share.
    fn tail_call(self) -> (i64, char) {
        match self {
            OperatorTail::Len => (bh_len_tail as *const () as i64, 'r'),
            OperatorTail::Truth => (bh_truth_bool_tail as *const () as i64, 'i'),
            OperatorTail::Format => (bh_format_tail as *const () as i64, 'r'),
        }
    }
}

/// Where the inlined dunder's return lands when it leaves the blackhole
/// (`_setup_return_value_r` → `call_result_reg`).  Never carried by the
/// resume section: `get_list_of_active_boxes(in_a_call=True)` (`pyjitpl.py`)
/// clears a caller's pending call-result slot, and this register is exactly
/// that slot.
const DUNDER_RESULT_REG: u16 = 0;

/// The operator's finished answer, in whichever bank [`OperatorTail::tail_call`]
/// names.
const ANSWER_REG: u16 = 0;

/// Callee operand of the never-decoded `inline_call`.  See the module doc:
/// `_setup_return_value_r` only reads the last operand as the dest register.
const PLACEHOLDER_CALLEE: u16 = 0;

/// `operation.py len` after `_len`: `space.index` on what `__len__` returned,
/// then `_check_len_result` on that, and the machine length re-boxed.
///
/// The re-box is the operator's, not a convenience: `builtins.rs builtin_len`
/// answers `w_int_new(space.len_w(obj))`, so `len()` is an exact `int` even
/// where `__len__` returned a `bool`.
///
/// A refusal publishes through `ResidualError::publish_residual`
/// (`jit_publish_residual_error`): `BH_LAST_EXC_VALUE` for the blackhole
/// handler and the compiled `GUARD_NO_EXCEPTION` cell. The return is the
/// residual ABI's zero; the handler does not read it.
pub extern "C" fn bh_len_tail(w_res: pyre_object::PyObjectRef) -> i64 {
    match pyre_interpreter::baseobjspace::len_result_tail(w_res) {
        Ok(length) => pyre_object::w_int_new(length) as i64,
        Err(err) => {
            use majit_ir::helper_fnaddr::ResidualError;
            err.publish_residual();
            0
        }
    }
}

/// `descroperation.py is_true` after `get_and_call_function` of `__bool__`.
///
/// `w_False` and `w_True` are the only bool instances, so they become 0 and
/// 1. Any other box is the TypeError `bool_must_return_bool` builds. The
/// Truth residual's result bank is int (`opcode_ops::truth_value`); a guard
/// inside the inlined `__bool__` resumes here so that int, not the raw bool
/// object, is what the caller's result register receives.
pub extern "C" fn bh_truth_bool_tail(w_res: pyre_object::PyObjectRef) -> i64 {
    if !w_res.is_null() && unsafe { pyre_object::is_bool(w_res) } {
        return unsafe { pyre_object::w_bool_get_value(w_res) } as i64;
    }
    pyre_interpreter::bool_must_return_bool_jit_abi(w_res)
}

/// `descroperation.py format` after `get_and_call_function` of `__format__`.
///
/// `isinstance_w(w_res, w_unicode)` keeps a str subclass's identity; any
/// other box is the TypeError `call_format_dispatch_w` builds. A guard
/// inside the inlined `__format__` resumes here so that check, not the raw
/// dunder return, is what the FORMAT_WITH_SPEC result register receives.
pub extern "C" fn bh_format_tail(w_res: pyre_object::PyObjectRef) -> i64 {
    if !w_res.is_null() && unsafe { pyre_object::is_str(w_res) } {
        return w_res as i64;
    }
    let err = pyre_interpreter::PyError::type_error(format!(
        "__format__ must return a str, not {}",
        pyre_interpreter::type_methods::arg_type_name(w_res),
    ));
    use majit_ir::helper_fnaddr::ResidualError;
    err.publish_residual();
    0
}

thread_local! {
    static LEVELS: std::cell::RefCell<[Option<Option<(i32, usize)>>; OperatorTail::ALL.len()]> =
        const { std::cell::RefCell::new([None; OperatorTail::ALL.len()]) };
}

/// `(jitcode index, resume pc)` of `tail`'s shared level, built on first use.
///
/// Thread-local because `MetaInterpStaticData` is: the index it hands back
/// names a slot in this thread's `jitcodes`.
fn level(tail: OperatorTail) -> Option<(i32, usize)> {
    if let Some(cached) = LEVELS.with(|cell| cell.borrow()[tail.slot()]) {
        return cached;
    }
    // Built outside the borrow: `install_codeless_jitcode` reaches
    // `METAINTERP_SD`, and a nested borrow of this cell from there would
    // panic.
    let built = build(tail);
    LEVELS.with(|cell| cell.borrow_mut()[tail.slot()] = Some(built));
    built
}

/// The jitcode index of `tail`'s level, or `None` when it could not be built.
pub(crate) fn jitcode_index(tail: OperatorTail) -> Option<i32> {
    level(tail).map(|(index, _)| index)
}

/// The byte offset `tail`'s resume section must name.
pub(crate) fn resume_pc(tail: OperatorTail) -> Option<usize> {
    level(tail).map(|(_, pc)| pc)
}

/// Whether `index` names an installed level, WITHOUT building one.
///
/// A reader that only asks "is this resumed level a tail?" must not be the
/// thing that mints it: [`level`] installs a jitcode into
/// `MetaInterpStaticData.jitcodes`, so calling it from a decode path would
/// append that slot in every process that ever decodes a resume section,
/// shifting the index space for programs that never inline a dunder.
pub(crate) fn is_installed_level(index: i32) -> bool {
    LEVELS.with(|cell| {
        cell.borrow()
            .iter()
            .any(|slot| matches!(slot, Some(Some((installed, _))) if *installed == index))
    })
}

fn build(tail: OperatorTail) -> Option<(i32, usize)> {
    let mut builder = JitCodeBuilder::new();
    builder.set_name(tail.name());

    // Never decoded; present so `_setup_return_value_r` can read
    // `code[position-1]` as the dest register. See the module doc.
    builder.inline_call_r_r(PLACEHOLDER_CALLEE, &[], Some(DUNDER_RESULT_REG));

    let resume_pc = builder.current_pos();
    // `-live-` operands are 2-byte offsets into the ONE shared
    // `MetaInterpStaticData.liveness_info` pool (`pyjitpl.py`), so the triple
    // is interned there rather than in a private assembler buffer.  Nothing
    // is live: the level's whole state is the pending call-result slot, which
    // a resume section never carries.
    let liveness_offset = crate::state::intern_liveness(&[], &[], &[])?;
    let live_patch = builder.live_placeholder();
    builder.patch_live_offset(live_patch, liveness_offset);

    let (funcptr, result_type) = tail.tail_call();
    let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr {
        // One `Ref` argument — the box the dunder returned — and the result
        // bank the caller's own call-result register lives in.
        arg_classes: "r".to_string(),
        result_type,
        ..Default::default()
    };
    let args = [JitCallArg {
        kind: majit_metainterp::jitcode::JitArgKind::Ref,
        reg: DUNDER_RESULT_REG,
    }];
    match result_type {
        'r' => {
            builder.residual_call_ref_canonical_typed_args(funcptr, &args, calldescr, ANSWER_REG);
            let after_call = builder.live_placeholder();
            builder.patch_live_offset(after_call, liveness_offset);
            builder.ref_return(ANSWER_REG);
        }
        'i' => {
            builder.residual_call_int_canonical_typed_args(funcptr, &args, calldescr, ANSWER_REG);
            let after_call = builder.live_placeholder();
            builder.patch_live_offset(after_call, liveness_offset);
            builder.int_return(ANSWER_REG);
        }
        _ => return None,
    }

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
    fn every_anchor_sits_behind_its_inline_call_return_tail() {
        for tail in OperatorTail::ALL {
            let Some((index, resume_pc)) = level(tail) else {
                continue;
            };
            let payload = crate::state::pyjitcode_for_jitcode_index(index)
                .expect("the tail was just installed at this index");
            let code = payload.jitcode.code.as_slice();
            assert!(
                resume_pc >= 3,
                "{tail:?}: no room for the call header and its return register",
            );
            assert_eq!(
                code[resume_pc - 1] as u16,
                DUNDER_RESULT_REG,
                "{tail:?}: the last operand is the dunder's result register",
            );
            assert_eq!(
                code[resume_pc],
                crate::state::op_live(),
                "{tail:?}: the resume position must be the `-live-` anchor itself",
            );
            let startpoints = payload
                .jitcode
                .startpoints
                .as_ref()
                .expect("the builder records one startpoint per emitted instruction");
            assert!(
                startpoints.contains(&resume_pc),
                "{tail:?}: the anchor must be addressable as a resume coordinate",
            );
            // `run_inner` asserts membership on every dispatched position,
            // not only the entry one: `inline_call`, `-live-`,
            // `residual_call`, the guard's `-live-`, the return.
            assert_eq!(
                startpoints.len(),
                5,
                "{tail:?}: startpoints must cover every op, got {startpoints:?}",
            );
        }
    }

    /// The generic paused-level loop resolves every parent's resume offset
    /// through `resolve_resume_pc_with_jitcode_pc`, so a tail must answer it
    /// like any other level — otherwise recording it as an ordinary parent
    /// aborts the guard with `GuardResumeCoordinateUnavailable`.
    #[test]
    fn every_anchor_resolves_as_an_ordinary_parent_resume_coordinate() {
        for tail in OperatorTail::ALL {
            let Some((index, resume_pc)) = level(tail) else {
                continue;
            };
            let payload = crate::state::pyjitcode_for_jitcode_index(index)
                .expect("the tail was just installed at this index");
            assert!(
                payload
                    .jitcode
                    .can_decode_live_vars(resume_pc, crate::state::op_live()),
                "{tail:?}: the anchor must decode its live vars",
            );
            assert_eq!(
                payload
                    .resolve_resume_pc_with_jitcode_pc(resume_pc as i32, crate::state::op_live()),
                Some(resume_pc),
                "{tail:?}: the parent loop must resolve the anchor to itself",
            );
        }
    }

    /// `len`'s two checks run on the resume path, and the `ValueError` a
    /// negative length owes reaches the channel the blackhole tests after a
    /// residual call.
    #[test]
    fn the_len_tail_applies_check_len_result() {
        let cell = &majit_metainterp::blackhole::BH_LAST_EXC_VALUE;
        cell.with(|c| c.set(0));
        let answer = bh_len_tail(pyre_object::w_bool_from(true));
        assert_eq!(cell.with(|c| c.get()), 0, "a valid length must not raise");
        let answer = answer as pyre_object::PyObjectRef;
        assert!(
            unsafe { !pyre_object::is_bool(answer) },
            "`len()` re-boxes the machine length, so a `__len__` that answered              `True` still gives an exact int",
        );
        assert_eq!(
            unsafe { pyre_object::w_int_get_value(answer) },
            1,
            "the answer is the checked length",
        );

        bh_len_tail(pyre_object::w_int_new(-1));
        assert_ne!(
            cell.with(|c| c.get()),
            0,
            "a negative length must publish the ValueError where \
             `handler_residual_call_r_r` reads it",
        );
        cell.with(|c| c.set(0));
    }

    /// `__bool__` answering `True` / `False` is 1 / 0. Anything else publishes
    /// the TypeError the Truth residual's exception guard reads.
    #[test]
    fn the_truth_bool_tail_requires_a_bool() {
        let cell = &majit_metainterp::blackhole::BH_LAST_EXC_VALUE;
        cell.with(|c| c.set(0));
        assert_eq!(bh_truth_bool_tail(pyre_object::w_bool_from(true)), 1);
        assert_eq!(bh_truth_bool_tail(pyre_object::w_bool_from(false)), 0);
        assert_eq!(cell.with(|c| c.get()), 0, "a bool must not raise");

        assert_eq!(bh_truth_bool_tail(pyre_object::w_none()), 0);
        assert_ne!(
            cell.with(|c| c.get()),
            0,
            "a non-bool must publish the TypeError where the residual's \
             exception guard reads it",
        );
        cell.with(|c| c.set(0));
    }

    /// `__format__` answering a str is that box. Anything else publishes the
    /// TypeError the FORMAT_WITH_SPEC residual's exception guard reads.
    #[test]
    fn the_format_tail_requires_a_str() {
        let cell = &majit_metainterp::blackhole::BH_LAST_EXC_VALUE;
        cell.with(|c| c.set(0));
        let answer = bh_format_tail(pyre_object::w_str_new("s"));
        assert_eq!(cell.with(|c| c.get()), 0, "a str must not raise");
        let answer = answer as pyre_object::PyObjectRef;
        assert!(
            unsafe { pyre_object::is_str(answer) },
            "`format` hands the str box on, so a subclass keeps its type",
        );

        assert_eq!(bh_format_tail(pyre_object::w_int_new(42)), 0);
        assert_ne!(
            cell.with(|c| c.get()),
            0,
            "a non-str must publish the TypeError where the residual's \
             exception guard reads it",
        );
        cell.with(|c| c.set(0));
    }
}
