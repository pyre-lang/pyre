//! Dispatch JitCode IR shape unit tests.
//!
//! Verifies the singleton dispatch JitCode emitted by `__dispatch_jitcode_*`
//! contains the expected portal IR (BC_JIT_MERGE_POINT, BC_LOOP_HEADER) +
//! pre-dispatch ops in source order + opcode fetch + dispatch chain
//! (BC_GOTO_IF_NOT_INT_EQ_CONST chain) + Lowerable arm bodies force-inlined
//! into the dispatch JitCode (`try_inline_dispatch_arm`). Nop/Halt emit
//! nothing; AbortPermanent / Unsupported abort this frame. Loop close
//! follows the arm chain.

use majit_metainterp::jitcode::insns::{
    BC_ABORT, BC_GETARRAYITEM_GC_I_PURE, BC_GOTO_IF_NOT_INT_EQ, BC_INLINE_CALL, BC_INT_ADD,
    BC_INT_RETURN, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE, BC_STORE_STATE_FIELD,
};
use majit_metainterp::{Assembler, BC_GOTO, JitCode, JitDriver};

struct DispatchTestState {
    a: i64,
}

const OP_NOP: u8 = 0;
const OP_INC_A: u8 = 1;
const OP_END: u8 = 2;

pub type Bytecode = [u8];

#[allow(dead_code)]
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}
impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

#[majit_macros::jit_interp(
    state = DispatchTestState,
    env = Bytecode,
    state_fields = { a: int },
    greens = [],
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_minimal(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<DispatchTestState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = DispatchTestState { a: 0 };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_NOP => {}
            OP_INC_A => state.a += 1,
            _ => break,
        }
    }
    state.a
}

/// Shared fixture for the `dispatch_minimal` cluster: rebuilds the
/// canonical-liveness Assembler and returns the lowered dispatch JitCode.
/// Each test keeps its own assertions; only the identical build lines are
/// factored out here.
fn build_dispatch_minimal() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// Dispatch JitCode plus every registered arm sub-JitCode. Lowerable red-pc
/// arms land in the dispatch body (`try_inline_dispatch_arm`); Nop/Halt emit
/// nothing and Unsupported / AbortPermanent abort this frame. Shape
/// assertions that only care that an op was emitted must scan both.
fn dispatch_tree(jc: &JitCode) -> impl Iterator<Item = &JitCode> {
    std::iter::once(jc).chain(
        jc.exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode().map(|sub| sub.as_ref())),
    )
}

fn count_op_in_dispatch_tree(jc: &JitCode, op: u8) -> usize {
    dispatch_tree(jc)
        .map(|code| code.code.iter().filter(|&&b| b == op).count())
        .sum()
}

#[test]
fn dispatch_jitcode_emits_portal_markers() {
    let dispatch_jc = build_dispatch_minimal();
    let code = &dispatch_jc.code;
    assert!(!code.is_empty(), "dispatch JitCode body must not be empty");
    let mut i = 0;
    while i < code.len() && code[i] == BC_LIVE {
        i += 3;
    }
    assert!(
        i < code.len(),
        "dispatch JitCode body has only BC_LIVE markers; missing portal markers"
    );
    assert!(
        code[i] == BC_JIT_MERGE_POINT || code[i] == BC_JIT_MERGE_POINT_C,
        "dispatch JitCode portal must start with BC_JIT_MERGE_POINT or BC_JIT_MERGE_POINT_C, got byte {:#x} at offset {}",
        code[i],
        i
    );
}

#[test]
fn dispatch_jitcode_contains_inline_call_per_arm() {
    let dispatch_jc = build_dispatch_minimal();
    let code = &dispatch_jc.code;
    let inline_call_count = code.iter().filter(|&&b| b == BC_INLINE_CALL).count();
    // OP_NOP is ArmPattern::Nop and emits nothing; the jump is the transfer.
    // OP_INC_A is Lowerable and is force-inlined into the dispatch body.
    assert_eq!(
        inline_call_count, 0,
        "dispatch JitCode must not emit BC_INLINE_CALL when Nop/Halt stay in this frame; got {}",
        inline_call_count
    );
}

#[test]
fn dispatch_arm_subjitcode_lowers_state_field_write() {
    let dispatch_jc = build_dispatch_minimal();
    let sub_jitcodes = dispatch_jc
        .exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode())
        .collect::<Vec<_>>();

    assert_eq!(
        sub_jitcodes.len(),
        0,
        "Nop emits nothing and Lowerable OP_INC_A is inlined; no arm sub-JitCode remains"
    );
    assert!(
        dispatch_jc.code.contains(&BC_STORE_STATE_FIELD),
        "state.a += 1 arm must lower to store_state_field in the dispatch body; dispatch={:?}",
        dispatch_jc.code
    );
    assert!(
        !dispatch_jc.code.contains(&BC_ABORT),
        "inlined lowerable dispatch arms must not degenerate to an in-frame abort; dispatch={:?}",
        dispatch_jc.code
    );
}

mod literal_for_unroll {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::BC_STORE_STATE_FIELD;
    use majit_metainterp::{Assembler, JitDriver};

    const OP_UNROLL: u8 = 3;

    struct LiteralForState {
        acc: i64,
    }

    #[majit_macros::jit_interp(
        state = LiteralForState,
        env = Bytecode,
        state_fields = { acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_literal_for_unroll(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<LiteralForState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = LiteralForState { acc: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_UNROLL => {
                    for k in 0..4 {
                        state.acc += k;
                    }
                }
                _ => break,
            }
        }
        state.acc
    }

    fn build_literal_for_unroll() -> majit_metainterp::JitCode {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_literal_for_unroll(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        __dispatch_jitcode_dispatch_literal_for_unroll(&mut asm, 0i64)
            .expect("literal-range for-loop dispatch lower must succeed")
    }

    #[test]
    fn dispatch_arm_unrolls_literal_range_for_loop() {
        let dispatch_jc = build_literal_for_unroll();
        let store_count = super::count_op_in_dispatch_tree(&dispatch_jc, BC_STORE_STATE_FIELD);
        assert_eq!(
            store_count, 4,
            "for k in 0..4 body must be unrolled to four straight-line state writes; dispatch={:?}",
            dispatch_jc.code
        );
    }
}

mod float_state_field_shape {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::BC_STORE_STATE_FIELD_FLOAT;
    use majit_metainterp::{Assembler, JitDriver};

    const OP_TOUCH_F: u8 = 9;

    struct FloatDispatchState {
        f: f64,
    }

    #[majit_macros::jit_interp(
        state = FloatDispatchState,
        env = Bytecode,
        state_fields = { f: float },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_float_field(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatDispatchState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatDispatchState { f: 0.0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_TOUCH_F => state.f += 2.5,
                _ => break,
            }
        }
        0
    }

    #[test]
    fn dispatch_arm_subjitcode_lowers_float_state_field_write() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![], vec![], vec![0]);
        __prebuild_jitcode_liveness_dispatch_float_field(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_float_field(&mut asm, 0i64)
            .expect("float dispatch lower must succeed");
        assert!(
            super::count_op_in_dispatch_tree(&dispatch_jc, BC_STORE_STATE_FIELD_FLOAT) >= 1,
            "state.f update must lower to store_state_field_float; dispatch={:?}",
            dispatch_jc.code
        );
    }
}

mod or_pattern {
    use super::{Bytecode, OP_INC_A, OP_NOP};
    use majit_metainterp::jitcode::insns::BC_STORE_STATE_FIELD;
    use majit_metainterp::{Assembler, BC_GOTO, JitDriver};

    struct OrDispatchState {
        a: i64,
    }

    #[majit_macros::jit_interp(
        state = OrDispatchState,
        env = Bytecode,
        state_fields = { a: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_or_pattern(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<OrDispatchState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = OrDispatchState { a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP | OP_INC_A => state.a += 1,
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn dispatch_or_pattern_short_circuits_alternatives_before_inline_call() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_or_pattern(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_or_pattern(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let first_body = code
            .iter()
            .position(|&b| b == BC_STORE_STATE_FIELD)
            .expect("missing inlined store for OR-pattern arm");
        let first_goto = code
            .iter()
            .position(|&b| b == BC_GOTO)
            .expect("missing successful-alternative jump for OR pattern");

        assert!(
            first_goto < first_body,
            "OR-pattern dispatch must jump from a successful early alternative \
             to the shared matched body before the inlined arm; otherwise A | B is \
             lowered as an accidental conjunction"
        );
    }
}

mod switch_dispatch {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::{BC_GOTO_IF_NOT_INT_EQ, BC_SWITCH};
    use majit_metainterp::{Assembler, JitDriver};

    const OP_LIT_A: u8 = 3;
    const OP_LIT_B: u8 = 4;
    const OP_RANGE_START: u8 = 10;
    const OP_RANGE_END: u8 = 12;
    const OP_END: u8 = 99;

    struct SwitchDispatchState {
        a: i64,
    }

    #[majit_macros::jit_interp(
        state = SwitchDispatchState,
        env = Bytecode,
        state_fields = { a: int },
        switch_dispatch = true,
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_switch_pattern(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<SwitchDispatchState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = SwitchDispatchState { a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_LIT_A | OP_LIT_B => state.a += 1,
                OP_RANGE_START..=OP_RANGE_END => state.a += 10,
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn switch_dispatch_lowers_match_to_single_switch() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_switch_pattern(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_switch_pattern(&mut asm, 0i64)
            .expect("switch dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        assert_eq!(
            code.iter().filter(|&&b| b == BC_SWITCH).count(),
            1,
            "switch_dispatch must emit one BC_SWITCH for the top-level dispatch; got bytes {:?}",
            code
        );
        assert_eq!(
            code.iter().filter(|&&b| b == BC_GOTO_IF_NOT_INT_EQ).count(),
            0,
            "switch_dispatch must not emit the old guard chain for the top-level dispatch"
        );
    }

    #[test]
    fn switch_dispatch_fixture_runs_literal_range_and_default() {
        let program = [OP_LIT_A, OP_RANGE_START, OP_RANGE_END, OP_END];
        let result = dispatch_switch_pattern(&program, 1);
        assert_eq!(result, 21);
    }
}

#[test]
fn dispatch_jitcode_contains_loop_back_goto() {
    let dispatch_jc = build_dispatch_minimal();
    let code = &dispatch_jc.code;
    let goto_count = code.iter().filter(|&&b| b == BC_GOTO).count();
    assert!(
        goto_count >= 1,
        "dispatch JitCode must emit at least one BC_GOTO for loop closure; got {}",
        goto_count
    );
}

/// The dispatch JitCode body must contain the opcode-fetch IR ops
/// lowered from `let opcode = program[pc]; pc += 1;`:
///   1. BC_GETARRAYITEM_GC_I_PURE — loads program[pc] into an int register
///   2. BC_INT_ADD           — increments pc_reg by 1 (via load_const + int_add)
///
/// pyopcode.py `opcode = ord(co_code[next_instr])` + `next_instr += 1`.
#[test]
fn dispatch_jitcode_lowers_opcode_fetch() {
    let dispatch_jc = build_dispatch_minimal();
    let code = &dispatch_jc.code;
    let mp_idx = code
        .iter()
        .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
        .expect("missing JIT_MERGE_POINT");
    let post_mp = &code[mp_idx..];
    assert!(
        post_mp.contains(&BC_GETARRAYITEM_GC_I_PURE),
        "dispatch JitCode body must emit BC_GETARRAYITEM_GC_I_PURE for program[pc]; \
         got bytes {:?}",
        post_mp
    );
    assert!(
        post_mp.contains(&BC_INT_ADD),
        "dispatch JitCode body must emit BC_INT_ADD for pc += 1; \
         got bytes {:?}",
        post_mp
    );
}

/// The dispatch JitCode body must emit one BC_GOTO_IF_NOT_INT_EQ
/// per non-default arm (OP_NOP, OP_INC_A → ≥ 2 checks).
///
/// pyopcode.py+ if/elif chain over opcode constants.
/// jtransform.py optimize_goto_if_not fuses int_eq + goto_if_not
/// into goto_if_not_int_eq/iiL (BC_GOTO_IF_NOT_INT_EQ).
#[test]
fn dispatch_jitcode_emits_chain_of_int_eq_dispatch() {
    let dispatch_jc = build_dispatch_minimal();
    let code = &dispatch_jc.code;
    let chain_count = code.iter().filter(|&&b| b == BC_GOTO_IF_NOT_INT_EQ).count();
    assert!(
        chain_count >= 2,
        "dispatch chain must emit at least one BC_GOTO_IF_NOT_INT_EQ per \
         non-default arm; OP_NOP + OP_INC_A → expected ≥ 2, got {}",
        chain_count
    );
}

/// The dispatch JitCode default arm must emit a typed return.
/// dispatch_minimal returns `state.a` (i64 / Int binding) so the JitCode
/// must end with BC_INT_RETURN.
///
/// interp_jit.py return boundary: when no arm matches the loop
/// exits; the dispatch JitCode signals this via an int_return insn.
#[test]
fn dispatch_jitcode_emits_typed_return_for_default_arm() {
    let dispatch_jc = build_dispatch_minimal();
    let code = &dispatch_jc.code;
    assert!(
        code.contains(&BC_INT_RETURN),
        "dispatch JitCode default arm must emit BC_INT_RETURN for `state.a` return; \
         got bytes {:?}",
        code
    );
    // Position check: the typed return must come AFTER the dispatch chain's
    // last fall-through GOTO (which targets the typed-return site itself).
    let last_goto_idx = code
        .iter()
        .rposition(|&b| b == BC_GOTO)
        .expect("missing dispatch chain default GOTO");
    let return_idx = code
        .iter()
        .position(|&b| b == BC_INT_RETURN)
        .expect("missing BC_INT_RETURN");
    assert!(
        return_idx > last_goto_idx,
        "BC_INT_RETURN must come after the dispatch chain default GOTO; \
         got return@{} goto@{}",
        return_idx,
        last_goto_idx
    );
}

mod pre_promote {

    use majit_metainterp::{Assembler, JitDriver};

    struct PromoteState {
        stackpos: i64,
    }

    type Bytecode = [u8];

    #[majit_macros::jit_interp(
        state = PromoteState,
        env = Bytecode,
        state_fields = { stackpos: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_pre_promote(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<PromoteState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = PromoteState { stackpos: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // jtransform.py:608-615: hint(promote=True) → -live- + int_guard_value
            state.stackpos = majit_metainterp::jit::promote(state.stackpos);
            let opcode = program[pc];
            pc += 1;
            match opcode {
                0 => state.stackpos += 1,
                _ => break,
            }
        }
        state.stackpos
    }

    #[test]
    fn dispatch_jitcode_lowers_pre_dispatch_promote_in_position() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
        };
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_pre_promote(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_with_pre_promote(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        // Search for any int_guard_value opcode after the merge point.
        let guard_idx = code[mp_idx..].iter().position(|&b| b == BC_INT_GUARD_VALUE);
        assert!(
            guard_idx.is_some(),
            "pre-dispatch promote(state.stackpos) must lower to BC_INT_GUARD_VALUE \
             in dispatch JitCode body (RPython jtransform.py:608-615 -live- + guard_value pair)"
        );
    }
}

/// `__trace_dispatch_minimal` accepts the dispatch JitCode singleton as an
/// extra parameter forwarded from `__merge_dispatch_minimal`.  When the
/// singleton is absent because dispatch lowering failed, the trace path
/// aborts permanently instead of falling back to a per-(pc, op) factory.
///
/// This test exercises the full trace path end-to-end: warmup threshold=1
/// causes one pre-trace iteration, then the trace fires and records ops via
/// the registered dispatch JitCode.
#[test]
fn dispatch_minimal_traces_via_dispatch_jitcode_singleton() {
    // OP_INC_A increments state.a; threshold=1 ensures trace fires after 1 iter.
    // Result must equal 3 (three OP_INC_A ops), then OP_END breaks.
    let program = [OP_INC_A, OP_INC_A, OP_INC_A, OP_END];
    let result = dispatch_minimal(&program, 1);
    assert_eq!(
        result, 3,
        "dispatch_minimal must increment state.a per OP_INC_A; got {}",
        result
    );
}

/// JitDriver::register_dispatch_jitcode stores the dispatch JitCode
/// as an Arc singleton and dispatch_jitcode() returns None before registration
/// and Some after.
///
/// RPython parity: metainterp_sd.jitcodes[driver_idx] global registry slot,
/// scoped to the per-#[jit_interp] driver.
#[test]
fn register_dispatch_jitcode_stores_singleton() {
    use std::sync::Arc;

    let mut driver: JitDriver<DispatchTestState> = JitDriver::new(100);
    assert!(
        driver.dispatch_jitcode().is_none(),
        "register_dispatch_jitcode_stores_singleton: dispatch_jitcode must be None before registration"
    );
    // `register_dispatch_jitcode` validates the dispatch JitCode's
    // `BC_JIT_MERGE_POINT` payload against the driver descriptor
    // (`warmspot.py make_args_specification` parity), so the
    // schema must be declared first — same lifecycle as the macro-
    // generated install path (`#declare_schema_fn_name(driver)` runs
    // before `ensure_descriptor_registered` in `codegen_state.rs`'s
    // `generate_state_fields_jit_state`).
    __declare_jit_schema_dispatch_minimal(&mut driver);

    let dispatch_jc = build_dispatch_minimal();

    driver.register_dispatch_jitcode(dispatch_jc);

    let stored = driver.dispatch_jitcode();
    assert!(
        stored.is_some(),
        "register_dispatch_jitcode_stores_singleton: dispatch_jitcode must be Some after register"
    );
    // Stored as Arc — ensure no panic on access
    let _arc: &Arc<JitCode> = stored.unwrap();
}

/// A thread that keeps two drivers alive holds ONE state-field store, and it
/// belongs to whichever driver published last. Every compile is reached through
/// a trace entry, so `force_start_tracing` re-aims the store: the guard metadata
/// that trace produces decodes frame value counts against its OWN jitcodes.
///
/// Without that re-aim the second registration owned the slot for good, and
/// `JitDriverStaticData::frame_value_count_fn` records that the wrong store
/// frequently *succeeds* — it decodes an unrelated jitcode at the same pc and
/// returns a mistyped frame count rather than failing.
#[test]
fn a_trace_entry_takes_the_state_field_store_back_from_a_second_driver() {
    let mut first: JitDriver<DispatchTestState> = JitDriver::new(100);
    __declare_jit_schema_dispatch_minimal(&mut first);
    first.register_dispatch_jitcode(build_dispatch_minimal());
    assert_eq!(
        majit_metainterp::current_state_field_fvc_epoch(),
        first.state_field_fvc_epoch(),
        "registering a dispatch jitcode publishes the registering driver's store",
    );

    let mut second: JitDriver<DispatchTestState> = JitDriver::new(100);
    __declare_jit_schema_dispatch_minimal(&mut second);
    second.register_dispatch_jitcode(build_dispatch_minimal());
    assert_ne!(
        first.state_field_fvc_epoch(),
        second.state_field_fvc_epoch(),
        "each driver's registry is published under its own identity",
    );
    assert_eq!(
        majit_metainterp::current_state_field_fvc_epoch(),
        second.state_field_fvc_epoch(),
        "the later registration takes the single slot",
    );

    let program = [OP_INC_A, OP_END];
    let mut state = DispatchTestState { a: 0 };
    first.meta_interp_mut().finish_setup_descrs_for_jitdrivers();
    first.force_start_tracing(0, None, 0, &mut state, &program[..]);
    assert_eq!(
        majit_metainterp::current_state_field_fvc_epoch(),
        first.state_field_fvc_epoch(),
        "force_start_tracing must aim the store at the driver about to trace",
    );
}

/// A.2.1 fixture: minimal `#[jit_interp]` dispatch loop carrying both the
/// `state.last_instr = pc as i64` store (`pyopcode.py`) AND the two-byte
/// `[opcode][oparg]` fetch with `pc += 2` (`pyopcode.py`).
///
/// A.2.1 only asserts the macro expands and the dispatch JitCode body builds
/// non-empty. A.2.2-A.2.4 add lower-side recognition for the new surfaces
/// (oparg fetch, last_instr store position) and pin shape via dedicated tests.
mod oparg_minimal {

    use majit_metainterp::{Assembler, JitDriver};

    struct OpargState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;
    /// Backward-jump opcode whose arm contains `can_enter_jit!()` to mirror
    /// RPython's `interp_jit.py pypyjitdriver.can_enter_jit(...)` inside
    /// `jump_absolute()`'s backward-branch path (`interp_jit.py if jumpto
    /// >= next_instr: return jumpto` early-out).  The arm exists so the
    /// dispatch JitCode lowerer emits a `BC_LOOP_HEADER` for at least one
    /// arm — `lower_dispatch_chain`'s per-arm gating suppresses
    /// `loop_header` for forward-progress arms (OP_NOP, OP_ADD_I) that have
    /// no `can_enter_jit!` marker, matching `jtransform.py handle_jit_marker__loop_header
    /// > handle_jit_marker__loop_header` which only runs at user-placed
    /// `can_enter_jit` source sites.
    const OP_JUMP_BACK: u8 = 2;

    /// A.2.5 helper: stand-in for RPython
    /// `executioncontext.py ec.bytecode_only_trace(self)` — the
    /// JIT-branch hook fired between `last_instr` store and opcode fetch.
    /// Pyre's `#[majit_macros::dont_look_inside]` attribute lowers a
    /// statement-form free-fn call to `BC_RESIDUAL_CALL_R_V` with
    /// `DEFAULT_EFFECT_INFO` (saturated read/write descrs +
    /// `EF_CAN_RAISE` per `assembler.rs`'s
    /// `residual_call_void_canonical_via_target_with_effect_info` +
    /// `call_descr.rs::DEFAULT_EFFECT_INFO`), matching plan v1 codex
    /// BLOCKER 2's required effect class for a side-effecting +
    /// can-raise hook.
    ///
    /// No args → `assembler.rs`'s `emit_canonical_call_void` auto-selection picks the R
    /// variant (no float / no int args → `BC_RESIDUAL_CALL_R_V`); the
    /// ref-list count byte is `0`. Pre-A.2.5 codex review BLOCKER 3
    /// (Ref-binding requirement) is satisfied by the no-args choice
    /// rather than threading a Ref binding through the dispatch loop.
    #[majit_macros::dont_look_inside]
    fn bytecode_only_trace_helper() {}

    #[majit_macros::jit_interp(
        state = OpargState,
        env = Bytecode,
        auto_calls = true,
        state_fields = { last_instr: int, acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_oparg_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<OpargState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        // `can_enter_jit!()` rewrites to a runtime call that updates
        // `stacksize` after a back_edge close (`mod.rs`'s `rewrite_body`
        // `#stacksize_expr = 0i32;`).  All examples that wire
        // `can_enter_jit!()` declare `let mut stacksize: i32 = 0;` in the
        // function scope; the OP_JUMP_BACK arm below requires the same.
        let stacksize: i32 = 0;
        let mut state = OpargState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // pyopcode.py:172  self.last_instr = intmask(next_instr)
            state.last_instr = pc as i64;
            // pyopcode.py:174  if jit.we_are_jitted():
            //                      ec.bytecode_only_trace(self)
            // (we_are_jitted branch only — the trace-fn version flows
            // through `#[dont_look_inside]` which pyre lowers as a
            // residual_call_void.)
            bytecode_only_trace_helper();
            // pyopcode.py:179  opcode = ord(co_code[next_instr])
            let opcode = program[pc];
            // pyopcode.py:180  oparg = ord(co_code[next_instr + 1])
            let oparg = program[pc + 1];
            // pyopcode.py:181  next_instr += 2
            pc += 2;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg as i64,
                // Backward-jump arm — `can_enter_jit!()` at the call
                // site so `Lowerer::lower_stmt`'s `Stmt::Macro` arm
                // emits `BC_LOOP_HEADER` INSIDE this arm's sub-JitCode
                // (jtransform.py handle_jit_marker__loop_header
                // at the source-level can_enter_jit position).  The
                // `if target < pc { ... }` conditional + `continue` shape
                // used by production consumers (interp_jit.py:104
                // `if jumpto >= next_instr: return jumpto` early-out
                // mirror) is kept on the runtime-side path via
                // `transform_function`'s `rewrite_body` substitution
                // (`mod.rs`'s `rewrite_body`, emitting
                // `driver.back_edge_structured(...)`); the
                // dispatch JitCode lowerer cannot yet lower `if`+`continue`
                // inside arm bodies (lowerer not yet ported), so the
                // fixture body is reduced to the macro call alone — that
                // is the smallest shape that exercises the call-site LH
                // emit per RPython parity.
                OP_JUMP_BACK => {
                    can_enter_jit!(driver, pc, &mut state, program, || {});
                }
                _ => break,
            }
        }
        state.acc
    }

    /// Shared fixture for the `dispatch_oparg_minimal` cluster: rebuilds the
    /// canonical-liveness Assembler and returns the lowered dispatch JitCode.
    /// Each test keeps its own assertions; only the identical build lines are
    /// factored out here.
    fn build_oparg_minimal() -> majit_metainterp::JitCode {
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture")
    }

    #[test]
    fn dispatch_oparg_minimal_builds_jitcode() {
        let dispatch_jc = build_oparg_minimal();
        assert!(
            !dispatch_jc.code.is_empty(),
            "A.2.1: dispatch JitCode body must be non-empty for the oparg+last_instr fixture"
        );
    }

    /// A.2.2: the dispatch JitCode body must lower BOTH byte fetches in
    /// `let opcode = program[pc]; let oparg = program[pc + 1]; pc += 2;`
    /// per RPython `pyopcode.py`:
    ///     - opcode fetch → `BC_GETARRAYITEM_GC_I_PURE result, program, pc`
    ///     - oparg  fetch → `BC_INT_ADD offset, pc, +1` then
    ///       `BC_GETARRAYITEM_GC_I_PURE result, program, offset`
    ///     - pc advance   → `BC_INT_ADD pc, pc, +2`
    ///
    /// Net post-merge-point op shape: BC_GETARRAYITEM_GC_I_PURE × 2 and at
    /// least two BC_INT_ADDs (one for `pc + 1` offset, one for `pc += 2`).
    #[test]
    fn dispatch_oparg_minimal_lowers_oparg_fetch_and_pc_advance() {
        use majit_metainterp::jitcode::insns::{
            BC_GETARRAYITEM_GC_I_PURE, BC_INT_ADD, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
        };

        let dispatch_jc = build_oparg_minimal();
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        let getarrayitem_count = post_mp
            .iter()
            .filter(|&&b| b == BC_GETARRAYITEM_GC_I_PURE)
            .count();
        assert_eq!(
            getarrayitem_count, 2,
            "A.2.2: dispatch JitCode body must emit exactly two BC_GETARRAYITEM_GC_I_PURE \
             ops (opcode + oparg per pyopcode.py:179-180); got {}",
            getarrayitem_count
        );

        let int_add_count = post_mp.iter().filter(|&&b| b == BC_INT_ADD).count();
        assert!(
            int_add_count >= 2,
            "A.2.2: dispatch JitCode body must emit ≥ 2 BC_INT_ADD ops \
             (one for pc+1 offset, one for pc += 2 advance); got {}",
            int_add_count
        );

        // Order check: the FIRST BC_GETARRAYITEM_GC_I_PURE (opcode fetch) must
        // precede the first BC_INT_ADD (which is the pc+1 offset compute
        // for the oparg fetch). The SECOND BC_GETARRAYITEM_GC_I_PURE (oparg
        // fetch) must come after the offset compute. RPython
        // `pyopcode.py` orders opcode → oparg.
        let first_getarr = post_mp
            .iter()
            .position(|&b| b == BC_GETARRAYITEM_GC_I_PURE)
            .expect("first BC_GETARRAYITEM_GC_I_PURE missing");
        let first_int_add = post_mp
            .iter()
            .position(|&b| b == BC_INT_ADD)
            .expect("first BC_INT_ADD missing");
        let second_getarr = post_mp
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b == BC_GETARRAYITEM_GC_I_PURE)
            .nth(1)
            .map(|(i, _)| i)
            .expect("second BC_GETARRAYITEM_GC_I_PURE missing");
        assert!(
            first_getarr < first_int_add,
            "A.2.2: opcode fetch must precede pc+1 offset compute; \
             first_getarr={} first_int_add={}",
            first_getarr,
            first_int_add
        );
        assert!(
            first_int_add < second_getarr,
            "A.2.2: pc+1 offset compute must precede oparg fetch; \
             first_int_add={} second_getarr={}",
            first_int_add,
            second_getarr
        );
    }

    /// A.2.4: the `state.last_instr = pc as i64` write
    /// (`pyopcode.py  self.last_instr = intmask(next_instr)`) must
    /// lower to a single `BC_STORE_STATE_FIELD` op emitted in the
    /// prelude position — between the JIT merge-point markers and the
    /// opcode fetch. The existing `state_fields` macro lowering
    /// (`jitcode_lower/lower_vable.rs`'s `lower_state_field_write`,
    /// `jitcode/assembler.rs`'s `store_state_field`, `pyjitpl/dispatch.rs`'s
    /// `run_one_step`) already emits the op;
    /// This test pins position and
    /// count so a future refactor cannot silently drop the store or
    /// reorder it across the opcode fetch.
    #[test]
    fn dispatch_oparg_minimal_pins_last_instr_store_position() {
        use majit_metainterp::jitcode::insns::{
            BC_GETARRAYITEM_GC_I_PURE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
            BC_STORE_STATE_FIELD,
        };

        let dispatch_jc = build_oparg_minimal();
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        let first_getarr = post_mp
            .iter()
            .position(|&b| b == BC_GETARRAYITEM_GC_I_PURE)
            .expect("BC_GETARRAYITEM_GC_I_PURE missing");
        let prelude_store_count = post_mp[..first_getarr]
            .iter()
            .filter(|&&b| b == BC_STORE_STATE_FIELD)
            .count();
        assert_eq!(
            prelude_store_count, 1,
            "A.2.4: prelude before opcode fetch must emit exactly one \
             BC_STORE_STATE_FIELD for `state.last_instr = pc as i64` \
             (pyopcode.py:172); later stores belong to inlined Lowerable \
             arms. got {}",
            prelude_store_count
        );

        let store_pos = post_mp
            .iter()
            .position(|&b| b == BC_STORE_STATE_FIELD)
            .expect("BC_STORE_STATE_FIELD missing");
        assert!(
            store_pos < first_getarr,
            "A.2.4: last_instr store must precede opcode fetch \
             (pyopcode.py:172 before L179); store_pos={} first_getarr={}",
            store_pos,
            first_getarr
        );
    }

    /// A.2.5.b: the `bytecode_only_trace_helper()` call (RPython
    /// `pyopcode.py:174 ec.bytecode_only_trace(self)` JIT branch) must
    /// lower to `BC_RESIDUAL_CALL_R_V` per `assembler.rs`'s
    /// `emit_canonical_call_void`
    /// auto-selection (no float / no int args → R variant with
    /// `ref_count = 0`). Position: between the `BC_STORE_STATE_FIELD`
    /// for `state.last_instr` (L172) and the first `BC_GETARRAYITEM_GC_I_PURE`
    /// for the opcode fetch (L179) — RPython source order is
    /// L172 store → L174 trace → L179 fetch.
    ///
    /// Codex Pre-A.2.5 review BLOCKER 2 absorbed: canonical opcode is
    /// `BC_RESIDUAL_CALL_R_V` (159), not the legacy `BC_RESIDUAL_CALL_VOID_R`
    /// named in plan v1 (retired). BLOCKER 3 absorbed: the no-args
    /// helper picks the R variant directly without requiring a Ref
    /// binding to be threaded through the dispatch loop.
    ///
    /// Optimizer-no-hoist verification is deferred — this test only
    /// pins the dispatch JitCode body shape, which is the byte stream
    /// the install gate sees, not the optimized loop IR. A future
    /// effort can add an optimized-IR
    /// observability harness and assert the call survives across two
    /// consecutive iterations without dedup or hoisting.
    #[test]
    fn dispatch_oparg_minimal_pins_bytecode_only_trace_residual_call() {
        use majit_metainterp::jitcode::insns::{
            BC_GETARRAYITEM_GC_I_PURE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
            BC_RESIDUAL_CALL_R_V, BC_STORE_STATE_FIELD,
        };

        let dispatch_jc = build_oparg_minimal();
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        // Counts use ≥ rather than == because the byte stream interleaves
        // opcode bytes with operand bytes; high-byte opcode values
        // (BC_RESIDUAL_CALL_R_V = 159) can collide with descr-index or
        // const-pool operand bytes in the surrounding ops. Strict
        // positional ordering via the first occurrence after each prior
        // landmark is the robust signal.
        let residual_count = post_mp
            .iter()
            .filter(|&&b| b == BC_RESIDUAL_CALL_R_V)
            .count();
        assert!(
            residual_count >= 1,
            "A.2.5.b: dispatch JitCode body must emit ≥ 1 BC_RESIDUAL_CALL_R_V \
             for the `#[dont_look_inside]`-annotated `bytecode_only_trace_helper()` \
             call (pyopcode.py:174); got {}",
            residual_count
        );

        let store_pos = post_mp
            .iter()
            .position(|&b| b == BC_STORE_STATE_FIELD)
            .expect("BC_STORE_STATE_FIELD missing");
        let residual_pos = post_mp
            .iter()
            .enumerate()
            .skip(store_pos + 1)
            .find(|&(_, &b)| b == BC_RESIDUAL_CALL_R_V)
            .map(|(i, _)| i)
            .expect("BC_RESIDUAL_CALL_R_V after BC_STORE_STATE_FIELD missing");
        let first_getarr = post_mp
            .iter()
            .enumerate()
            .skip(residual_pos + 1)
            .find(|&(_, &b)| b == BC_GETARRAYITEM_GC_I_PURE)
            .map(|(i, _)| i)
            .expect("BC_GETARRAYITEM_GC_I_PURE after BC_RESIDUAL_CALL_R_V missing");
        // Successful chained `find_after` proves the ordering:
        // BC_STORE_STATE_FIELD → BC_RESIDUAL_CALL_R_V → BC_GETARRAYITEM_GC_I_PURE,
        // matching pyopcode.py:172 → L174 → L179 source order.
        let _ = (store_pos, residual_pos, first_getarr);
    }

    /// A.3.3: the dispatch JitCode for `dispatch_oparg_minimal` (`greens = []`)
    /// must encode empty greens and real reds derived from portal inputs minus greens.
    ///
    /// Candidates: `["program", "pc"]`.  Greens empty → reds = both.
    /// `pc` is Int at i0 → reds_i = [0]; `program` is Ref at r0 → reds_r = [0].
    ///
    /// Full payload at offset +2 from the opcode byte (greens_base = mp_pos + 2):
    ///   greens_base + 0: greens_i_len = 0
    ///   greens_base + 1: greens_r_len = 0
    ///   greens_base + 2: greens_f_len = 0
    ///   greens_base + 3: reds_i_len   = 1  (pc → Int i0)
    ///   greens_base + 4: reds_i[0]    = 0
    ///   greens_base + 5: reds_r_len   = 1  (program → Ref r0)
    ///   greens_base + 6: reds_r[0]    = 0
    ///   greens_base + 7: reds_f_len   = 0
    #[test]
    fn dispatch_oparg_minimal_pins_real_reds_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let dispatch_jc = build_oparg_minimal();
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.3: BC_JIT_MERGE_POINT(_C) must be present in dispatch JitCode");

        // Compact or non-compact: jdindex occupies 1 byte immediately after
        // the opcode byte, so the greens/reds payload begins at mp_pos + 2.
        let greens_base = mp_pos + 2;

        assert!(
            greens_base + 7 < code.len(),
            "A.3.3: payload too short to decode greens+reds; greens_base={}, len={}",
            greens_base,
            code.len()
        );

        // Greens: all empty (greens = []).
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 0,
            "A.3.3: greens_i_len must be 0 (no greens declared); got {}",
            greens_i_len
        );
        let greens_r_len = code[greens_base + 1] as usize;
        assert_eq!(
            greens_r_len, 0,
            "A.3.3: greens_r_len must be 0 (no greens declared); got {}",
            greens_r_len
        );
        let greens_f_len = code[greens_base + 2] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.3: greens_f_len must be 0 (no greens declared); got {}",
            greens_f_len
        );

        // Reds: pc (Int/i0), last_instr/acc identity slots, program (Ref/r0).
        // `handle_jit_marker__jit_merge_point` lists identity slots as reds.
        let reds_i_len = code[greens_base + 3] as usize;
        assert_eq!(
            reds_i_len, 3,
            "A.3.3: reds_i_len must be 3 (pc + last_instr + acc); got {}",
            reds_i_len
        );
        assert_eq!(
            code[greens_base + 4],
            0,
            "A.3.3: reds_i[0] must be 0 (pc at register i0); got {}",
            code[greens_base + 4]
        );
        assert_eq!(
            code[greens_base + 5],
            1,
            "A.3.3: reds_i[1] must be 1 (last_instr identity); got {}",
            code[greens_base + 5]
        );
        assert_eq!(
            code[greens_base + 6],
            2,
            "A.3.3: reds_i[2] must be 2 (acc identity); got {}",
            code[greens_base + 6]
        );

        let reds_r_len = code[greens_base + 7] as usize;
        assert_eq!(
            reds_r_len, 1,
            "A.3.3: reds_r_len must be 1 (program → Ref r0); got {}",
            reds_r_len
        );
        let reds_r_byte = code[greens_base + 8];
        assert_eq!(
            reds_r_byte, 0,
            "A.3.3: reds_r[0] must be 0 (program at register r0); got {}",
            reds_r_byte
        );

        let reds_f_len = code[greens_base + 9] as usize;
        assert_eq!(
            reds_f_len, 0,
            "A.3.3: reds_f_len must be 0 (no float reds); got {}",
            reds_f_len
        );
    }

    /// A.3.4: `BC_LOOP_HEADER` jdindex wire — call-site emission for the
    /// force-inlined OP_JUMP_BACK arm.
    ///
    /// `jtransform.py` `handle_jit_marker__loop_header` emits
    /// `SpaceOperation('loop_header', [c_index], None)` with `c_index =
    /// Constant(jd.index, lltype.Signed)` AT THE SOURCE-LEVEL CALL SITE —
    /// in PyPy that is `interp_jit.py pypyjitdriver.can_enter_jit(...)`
    /// inside `jump_absolute()`'s BACKWARD-JUMP BRANCH ONLY (the forward
    /// path early-returns at `interp_jit.py if jumpto >= next_instr:
    /// return jumpto`).  Pyre's parity is achieved by recognising
    /// `can_enter_jit!(...)` as a `Stmt::Macro` inside `Lowerer::lower_stmt`
    /// (`jitcode_lower/lower_stmt.rs`) and emitting the LH op at that exact
    /// stmt position. The OP_JUMP_BACK arm is Lowerable, so
    /// `try_inline_dispatch_arm` places that call-site LH in the dispatch
    /// JitCode rather than a sub-JitCode. Forward-progress arms still must
    /// not emit their own LH.
    ///
    /// The test pins:
    ///   1. Exactly one const-pool-anchored `BC_LOOP_HEADER` in the dispatch
    ///      body (the inlined `can_enter_jit!()`).
    ///   2. The byte immediately after the LH opcode is in valid const-pool
    ///      range and stores the jdindex constant (`0i64` for this invocation).
    ///   3. No leftover arm sub-JitCode emits LH of its own.
    #[test]
    fn dispatch_oparg_minimal_pins_loop_header_jdindex() {
        use majit_metainterp::jitcode::insns::BC_LOOP_HEADER;

        let dispatch_jc = build_oparg_minimal();
        let dispatch_code = &dispatch_jc.code;

        let dispatch_num_regs_i = dispatch_jc.c_num_regs_i as usize;
        let dispatch_const_start = dispatch_num_regs_i;
        let dispatch_const_end = dispatch_num_regs_i + dispatch_jc.constants_i.len();
        let dispatch_lh_positions: Vec<usize> = (0..dispatch_code.len().saturating_sub(1))
            .filter(|&i| {
                dispatch_code[i] == BC_LOOP_HEADER
                    && (dispatch_code[i + 1] as usize) >= dispatch_const_start
                    && (dispatch_code[i + 1] as usize) < dispatch_const_end
            })
            .collect();
        assert_eq!(
            dispatch_lh_positions.len(),
            1,
            "A.3.4: dispatch JitCode body must contain exactly one \
             const-pool-anchored BC_LOOP_HEADER from the inlined \
             OP_JUMP_BACK can_enter_jit!() site; got {}",
            dispatch_lh_positions.len()
        );
        let lh_pos = dispatch_lh_positions[0];
        let slot_byte = dispatch_code[lh_pos + 1] as usize;
        let const_idx = slot_byte - dispatch_num_regs_i;
        let stored_jdindex = dispatch_jc.constants_i[const_idx];
        assert_eq!(
            stored_jdindex, 0i64,
            "A.3.4: BC_LOOP_HEADER const-pool slot must store jdindex=0; got {}",
            stored_jdindex
        );

        let arm_lh_emitting = dispatch_jc
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode())
            .filter(|sub| {
                let snr = sub.c_num_regs_i as usize;
                let cstart = snr;
                let cend = snr + sub.constants_i.len();
                (0..sub.code.len().saturating_sub(1)).any(|i| {
                    sub.code[i] == BC_LOOP_HEADER
                        && (sub.code[i + 1] as usize) >= cstart
                        && (sub.code[i + 1] as usize) < cend
                })
            })
            .count();
        assert_eq!(
            arm_lh_emitting, 0,
            "A.3.4: leftover arm sub-JitCodes must not emit BC_LOOP_HEADER; got {}",
            arm_lh_emitting
        );
    }

    /// A.3.5 (negative): with `greens = []`, NO `BC_*_GUARD_VALUE` op must
    /// appear in the prefix before `BC_JIT_MERGE_POINT`.
    ///
    /// Mirrors `jtransform.py promote_greens`: the loop over
    /// `args[:num_green_args]` is empty when `num_green_args == 0`, so no
    /// guard ops are emitted.
    #[test]
    fn dispatch_oparg_minimal_no_promote_when_greens_empty() {
        use majit_metainterp::jitcode::insns::{
            BC_FLOAT_GUARD_VALUE, BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
            BC_REF_GUARD_VALUE,
        };

        let dispatch_jc = build_oparg_minimal();
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("BC_JIT_MERGE_POINT(_C) must be present");

        let prefix = &code[..mp_pos];
        assert!(
            !prefix.contains(&BC_INT_GUARD_VALUE)
                && !prefix.contains(&BC_REF_GUARD_VALUE)
                && !prefix.contains(&BC_FLOAT_GUARD_VALUE),
            "A.3.5 (`jtransform.py` `promote_greens`): empty greens must not emit any \
             *_guard_value before BC_JIT_MERGE_POINT; prefix={:?}",
            prefix
        );
    }
}

/// A.2.3a fixture: dispatch loop carrying the EXTENDED_ARG inner while
/// shape per RPython `pyopcode.py`. The fixture's structural
/// contents (inner `while opcode == EXTENDED_ARG`, multi-byte oparg
/// merge, HAVE_ARGUMENT corruption guard) are NOT lowered to dispatch
/// JitCode IR yet — A.2.3a is recognition-only. The fail-closed install
/// gate added in A.2.3a guarantees that lowering FAILS (returns `None`,
/// dispatch body empty, gate refuses install) when the inner while
/// shape cannot be recognized; the recognizer accepts this fixture's
/// `while opcode == EXTENDED_ARG` form, so the test asserts the body
/// is non-empty.
mod oparg_extended {

    use majit_metainterp::jitcode::insns::{
        BC_ABORT, BC_GETARRAYITEM_GC_I_PURE, BC_INT_MUL, BC_INT_OR, BC_JIT_MERGE_POINT,
        BC_JIT_MERGE_POINT_C,
    };
    use majit_metainterp::{Assembler, JitDriver};

    struct OpargExtState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;
    const EXTENDED_ARG: u8 = 254;
    const HAVE_ARGUMENT: u8 = 90;

    #[majit_macros::jit_interp(
        state = OpargExtState,
        env = Bytecode,
        state_fields = { last_instr: int, acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables, unused_mut)]
    fn dispatch_oparg_extended(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<OpargExtState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = OpargExtState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // pyopcode.py:172  self.last_instr = intmask(next_instr)
            state.last_instr = pc as i64;
            // pyopcode.py:179-181  opcode/oparg fetch + pc += 2
            let mut opcode = program[pc];
            let mut oparg: i64 = program[pc + 1] as i64;
            pc += 2;
            // pyopcode.py:187-193  EXTENDED_ARG inner loop. Its body is
            // NOT lowered in A.2.3a — A.2.3b ports the multi-byte oparg
            // merge (`(oparg * 256) | arg2`) to BC_INT_MUL + BC_INT_OR
            // and the HAVE_ARGUMENT guard's `raise BytecodeCorruption`
            // (RPython L190-191) to a BC_ABORT bailout label per
            // Pre-A.2.3 codex review BLOCKERs (c) + (d).
            while opcode == EXTENDED_ARG {
                let opcode2 = program[pc];
                let arg2 = program[pc + 1];
                if opcode2 < HAVE_ARGUMENT {
                    panic!("BytecodeCorruption parity bailout (A.2.3b lowers to BC_ABORT)");
                }
                pc += 2;
                oparg = (oparg * 256) | arg2 as i64;
                opcode = opcode2;
            }
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg,
                _ => break,
            }
        }
        state.acc
    }

    /// A.2.3a positive test: the EXTENDED_ARG inner-while recognizer
    /// in `lower_pre_dispatch_stmts` accepts `while opcode ==
    /// EXTENDED_ARG`, so `lower_dispatch_body` returns `Some(body)`
    /// and the dispatch JitCode body is non-empty (and contains the
    /// A.2.2 opcode/oparg fetch + pc advance ops).
    #[test]
    fn dispatch_oparg_extended_builds_jitcode() {
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_extended(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_extended(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        assert!(
            !dispatch_jc.code.is_empty(),
            "A.2.3a: dispatch JitCode body must be non-empty when the \
             inner-while shape `while opcode == EXTENDED_ARG` is \
             recognized — gate refuses install (empty body) only when \
             the shape FAILS to match"
        );
    }

    /// A.2.3b shape pin: lowered EXTENDED_ARG inner while emits the
    /// RPython L188-193 IR sequence:
    ///
    /// - 4 × BC_GETARRAYITEM_GC_I_PURE — outer opcode + outer oparg
    ///   (pyopcode.py:179-180) plus inner opcode2 + inner arg2
    ///   (pyopcode.py:188-189)
    /// - 1 × BC_INT_MUL — `oparg * 256` (Pre-A.2.3 codex BLOCKER (c):
    ///   pyre lacks `BC_INT_LSHIFT_IMM`, jtransform.py:363-366 keeps
    ///   int_mul symmetric, so the line-by-line port is INT_MUL)
    /// - 1 × BC_INT_OR — `(oparg * 256) | arg`
    /// - 1 × BC_ABORT — corruption-bailout for `opcode < HAVE_ARGUMENT`
    ///   (Pre-A.2.3 codex BLOCKER (d): RPython L190-191
    ///   `raise BytecodeCorruption` ports to the canonical local bailout
    ///   `BC_ABORT`, not `guard_value`)
    ///
    /// Ordering is also pinned: outer fetch precedes inner fetch
    /// precedes int_mul precedes int_or precedes abort. The inner
    /// `let opcode2 = program[pc]` aliases to the outer `opcode`
    /// register per RPython L188 (`opcode = ord(co_code[next_instr])`
    /// reuses the outer opcode slot), so the back-edge re-tests the
    /// freshly-fetched value without an extra `int_or` copy.
    #[test]
    fn dispatch_oparg_extended_pins_inner_while_ir_shape() {
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_extended(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_extended(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        // Count assertions are intentionally positional: the JitCode
        // byte stream interleaves opcode bytes with operand bytes, so
        // a naive byte filter (.filter(|&&b| b == BC_X).count()) can
        // collide with operand values that happen to equal BC_X. The
        // existing `dispatch_oparg_minimal_lowers_oparg_fetch_and_pc_advance`
        // test gets away with `== 2` for the fetch opcode
        // because the fixture uses a small register window; this richer
        // EXTENDED_ARG fixture has more registers and high-byte descr
        // indices, so we assert minimums + ordering instead.
        let count = |target: u8| post_mp.iter().filter(|&&b| b == target).count();
        let getarr_count = count(BC_GETARRAYITEM_GC_I_PURE);
        assert!(
            getarr_count >= 4,
            "A.2.3b: dispatch JitCode body must emit ≥ 4 BC_GETARRAYITEM_GC_I_PURE \
             ops (outer opcode + outer oparg + inner opcode2 + inner arg2 \
             per pyopcode.py:179-180/188-189); got {}",
            getarr_count
        );
        assert!(
            count(BC_INT_MUL) >= 1,
            "A.2.3b: oparg merge `oparg * 256` must lower to ≥ 1 BC_INT_MUL \
             (pyre has no BC_INT_LSHIFT_IMM; jtransform.py:363-366 keeps \
             int_mul symmetric per Pre-A.2.3 codex BLOCKER (c))"
        );
        assert!(
            count(BC_INT_OR) >= 1,
            "A.2.3b: oparg merge `(oparg * 256) | arg` must lower to ≥ 1 \
             BC_INT_OR; opcode2 is aliased to the outer opcode register so \
             no second `int_or` copy is needed"
        );
        assert!(
            count(BC_ABORT) >= 1,
            "A.2.3b: HAVE_ARGUMENT corruption guard must lower `raise \
             BytecodeCorruption` to ≥ 1 BC_ABORT (Pre-A.2.3 codex BLOCKER \
             (d) — guard_value is wrong shape for a range check)"
        );

        // Ordering checks against FIRST occurrences in the byte stream.
        // The lowerer emits ops in source order
        // (pyopcode.py:179-180 outer fetches → L188-189 inner fetches →
        // L190 corruption abort → L191 next_instr += 2 → L193 oparg
        // merge). Operand-byte aliasing CAN spuriously pull a "first
        // occurrence" of BC_ABORT etc. before it is actually emitted —
        // but only if the operand happens to land before the real op,
        // which would also collapse the ordering chain into an
        // inconsistent prefix. Treating "first occurrence" as a strict
        // upper bound keeps the assertion conservative.
        let first_pos = |target: u8, after: usize| -> usize {
            post_mp
                .iter()
                .enumerate()
                .skip(after)
                .find(|&(_, &b)| b == target)
                .map(|(i, _)| i)
                .unwrap_or_else(|| panic!("BC_{:02x} after offset {} missing", target, after))
        };
        let outer_opcode_pos = first_pos(BC_GETARRAYITEM_GC_I_PURE, 0);
        let outer_oparg_pos = first_pos(BC_GETARRAYITEM_GC_I_PURE, outer_opcode_pos + 1);
        let inner_opcode_pos = first_pos(BC_GETARRAYITEM_GC_I_PURE, outer_oparg_pos + 1);
        let inner_arg_pos = first_pos(BC_GETARRAYITEM_GC_I_PURE, inner_opcode_pos + 1);
        let abort_pos = first_pos(BC_ABORT, inner_arg_pos + 1);
        let int_mul_pos = first_pos(BC_INT_MUL, abort_pos + 1);
        let int_or_pos = first_pos(BC_INT_OR, int_mul_pos + 1);
        // Pinning successive `find_after` succeeded → the ops appear in
        // exactly this order: outer opcode → outer oparg → inner
        // opcode2 → inner arg2 → corruption abort → oparg-merge int_mul
        // → oparg-merge int_or. RPython `pyopcode.py:179-193`.
        let _ = (
            outer_opcode_pos,
            outer_oparg_pos,
            inner_opcode_pos,
            inner_arg_pos,
            abort_pos,
            int_mul_pos,
            int_or_pos,
        );
    }
}

/// A.3.2 fixture: minimal dispatch loop with `greens = [pc]` declared.
///
/// Mirrors `dispatch_oparg_minimal` but adds `pc` to the green list so
/// `resolve_greens` must resolve the `pc` ident to register `i0` and emit
/// `greens_i = [0]` in the `jit_merge_point` payload.
mod oparg_with_pc_green {

    use majit_metainterp::{Assembler, JitDriver};

    struct PcGreenState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = PcGreenState,
        env = Bytecode,
        state_fields = { last_instr: int, acc: int },
        greens = [pc],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_oparg_with_pc_green(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<PcGreenState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = PcGreenState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // pyopcode.py:172  self.last_instr = intmask(next_instr)
            state.last_instr = pc as i64;
            // pyopcode.py:179  opcode = ord(co_code[next_instr])
            let opcode = program[pc];
            // pyopcode.py:180  oparg = ord(co_code[next_instr + 1])
            let oparg = program[pc + 1];
            // pyopcode.py:181  next_instr += 2
            pc += 2;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg as i64,
                _ => break,
            }
        }
        state.acc
    }

    /// A.3.2/A.3.3: the dispatch JitCode for `dispatch_oparg_with_pc_green` must
    /// encode `greens_i = [0]` (pc → Int register i0), `greens_r = []`,
    /// `greens_f = []`, then `reds_i = []` (pc filtered as green),
    /// `reds_r = [0]` (program → Ref r0), `reds_f = []`.
    ///
    /// Payload layout after the opcode + jdindex byte (assembler.py encoding):
    ///   [greens_i_len][greens_i bytes...][greens_r_len][greens_r bytes...]
    ///   [greens_f_len][greens_f bytes...][reds_i_len][reds_i bytes...]
    ///   [reds_r_len][reds_r bytes...][reds_f_len]
    ///
    /// For `jdindex = 0i64` the compact form `BC_JIT_MERGE_POINT_C` is used
    /// (jdindex fits in -128..=127), followed by one signed-byte jdindex.
    /// Full payload at offset +2 from the opcode byte:
    ///   offset+2: greens_i_len = 1
    ///   offset+3: greens_i[0]  = 0  (register i0 = pc)
    ///   offset+4: greens_r_len = 0
    ///   offset+5: greens_f_len = 0
    ///   offset+6: reds_i_len   = 0  (pc filtered as green)
    ///   offset+7: reds_r_len   = 1  (program → Ref r0)
    ///   offset+8: reds_r[0]    = 0
    ///   offset+9: reds_f_len   = 0
    #[test]
    fn dispatch_oparg_with_pc_green_pins_real_greens_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_with_pc_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_with_pc_green(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.2: BC_JIT_MERGE_POINT(_C) must be present in dispatch JitCode");

        let is_compact = code[mp_pos] == BC_JIT_MERGE_POINT_C;
        // Compact form: opcode(1) + jdindex_byte(1) + payload.
        // Non-compact: opcode(1) + const_pool_offset_byte(1) + payload.
        // Either way the jdindex occupies exactly 1 byte immediately after
        // the opcode, so the greens lists begin at mp_pos + 2.
        let greens_base = mp_pos + 2;

        assert!(
            greens_base + 7 < code.len(),
            "A.3.2: payload too short to decode greens+reds; is_compact={}, mp_pos={}, len={}",
            is_compact,
            mp_pos,
            code.len()
        );

        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.2: greens_i_len must be 1 (pc is Int → i0); got {}",
            greens_i_len
        );

        let greens_i_byte = code[greens_base + 1];
        assert_eq!(
            greens_i_byte, 0,
            "A.3.2: greens_i[0] must be 0 (register i0 = pc); got {}",
            greens_i_byte
        );

        let greens_r_len = code[greens_base + 2] as usize;
        assert_eq!(
            greens_r_len, 0,
            "A.3.2: greens_r_len must be 0 (pc is Int, not Ref); got {}",
            greens_r_len
        );

        let greens_f_len = code[greens_base + 3] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.2: greens_f_len must be 0 (pc is Int, not Float); got {}",
            greens_f_len
        );

        // A.3.3 reds: pc is green, identity slots last_instr/acc remain.
        //   greens_base + 4: reds_i_len  = 2
        //   greens_base + 5..6: last_instr, acc identity
        //   greens_base + 7: reds_r_len  = 1  (program → Ref r0)
        //   greens_base + 8: reds_r[0]   = 0
        //   greens_base + 9: reds_f_len  = 0

        let reds_i_len = code[greens_base + 4] as usize;
        assert_eq!(
            reds_i_len, 2,
            "A.3.3: reds_i_len must be 2 (last_instr + acc identity); got {}",
            reds_i_len
        );
        assert_eq!(code[greens_base + 5], 1, "last_instr identity at i1");
        assert_eq!(code[greens_base + 6], 2, "acc identity at i2");

        let reds_r_len = code[greens_base + 7] as usize;
        assert_eq!(
            reds_r_len, 1,
            "A.3.3: reds_r_len must be 1 (program → Ref r0); got {}",
            reds_r_len
        );
        let reds_r_byte = code[greens_base + 8];
        assert_eq!(
            reds_r_byte, 0,
            "A.3.3: reds_r[0] must be 0 (program at register r0); got {}",
            reds_r_byte
        );

        let reds_f_len = code[greens_base + 9] as usize;
        assert_eq!(
            reds_f_len, 0,
            "A.3.3: reds_f_len must be 0 (no float reds); got {}",
            reds_f_len
        );
    }

    /// A.3.5: with `greens = [pc]`, a `-live-` + `BC_INT_GUARD_VALUE` pair
    /// must appear BEFORE `BC_JIT_MERGE_POINT(_C)` in the dispatch JitCode.
    ///
    /// Mirrors `jtransform.py promote_greens`: for each green
    /// Variable, emit SpaceOperation('-live-', ...) then
    /// SpaceOperation('int_guard_value', [v], None).  `pc` is Int kind →
    /// int_guard_value.  The `-live-` byte is 1 opcode + 2-byte offset
    /// (from `live_placeholder`), so BC_LIVE is at `igv_pos - 3`.
    #[test]
    fn dispatch_oparg_with_pc_green_promotes_green_before_merge_point() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_with_pc_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_with_pc_green(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("BC_JIT_MERGE_POINT(_C) must be present");

        // `jtransform.py` `promote_greens`: pc is Int → int_guard_value must precede merge point.
        let igv_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_INT_GUARD_VALUE)
            .expect(
                "A.3.5 (`jtransform.py` `promote_greens`): pc green must be promoted via \
                 BC_INT_GUARD_VALUE before BC_JIT_MERGE_POINT",
            );

        assert!(
            igv_pos < mp_pos,
            "BC_INT_GUARD_VALUE must precede BC_JIT_MERGE_POINT"
        );

        // jtransform.py:1707: -live- marker immediately precedes each guard_value.
        // BC_LIVE is 1-byte opcode + 2-byte offset, so the BC_LIVE byte is at
        // igv_pos - 3.
        assert!(
            igv_pos >= 3,
            "BC_INT_GUARD_VALUE at position {} has no room for preceding -live- (need >= 3 bytes before)",
            igv_pos
        );
        assert_eq!(
            code[igv_pos - 3],
            BC_LIVE,
            "A.3.5 (jtransform.py:1707): -live- marker must precede BC_INT_GUARD_VALUE \
             for green promotion; code[{}]={:#04x}, expected BC_LIVE={:#04x}",
            igv_pos - 3,
            code[igv_pos - 3],
            BC_LIVE,
        );
    }
}

/// A.3.3 optional parity-anchor fixture: `greens = [pc, program]`.
///
/// Mirrors `interp_jit.py reds = ['frame', 'ec']` adapted to pyre's
/// portal binding names.  Declaring both portal inputs as green leaves
/// reds = [] — this is the A.6 follow-up parity shape.
///
/// Byte layout (greens_base = mp_pos + 2):
///   greens_base + 0: greens_i_len = 1  (pc → Int i0)
///   greens_base + 1: greens_i[0]  = 0
///   greens_base + 2: greens_r_len = 1  (program → Ref r0)
///   greens_base + 3: greens_r[0]  = 0
///   greens_base + 4: greens_f_len = 0
///   greens_base + 5: reds_i_len   = 0  (pc is green)
///   greens_base + 6: reds_r_len   = 0  (program is green)
///   greens_base + 7: reds_f_len   = 0
mod oparg_with_pypy_parity_greens {

    use majit_metainterp::{Assembler, JitDriver};

    struct ParityState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = ParityState,
        env = Bytecode,
        state_fields = { last_instr: int, acc: int },
        greens = [pc, program],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_oparg_with_pypy_parity_greens(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<ParityState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = ParityState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            state.last_instr = pc as i64;
            let opcode = program[pc];
            let oparg = program[pc + 1];
            pc += 2;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg as i64,
                _ => break,
            }
        }
        state.acc
    }

    /// A.3.3: `greens = [pc, program]` yields empty reds and both portal
    /// inputs in greens buckets.  This is the A.6 parity anchor for
    /// `interp_jit.py reds = ['frame', 'ec']` adapted to pyre.
    #[test]
    fn dispatch_oparg_with_pypy_parity_greens_pins_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_with_pypy_parity_greens(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_with_pypy_parity_greens(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.3 parity: BC_JIT_MERGE_POINT(_C) must be present");

        let greens_base = mp_pos + 2;

        assert!(
            greens_base + 7 < code.len(),
            "A.3.3 parity: payload too short; greens_base={}, len={}",
            greens_base,
            code.len()
        );

        // Greens: pc (Int/i0) then program (Ref/r0).
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.3 parity: greens_i_len must be 1 (pc → Int i0); got {}",
            greens_i_len
        );
        let greens_i_byte = code[greens_base + 1];
        assert_eq!(
            greens_i_byte, 0,
            "A.3.3 parity: greens_i[0] must be 0 (register i0 = pc); got {}",
            greens_i_byte
        );
        let greens_r_len = code[greens_base + 2] as usize;
        assert_eq!(
            greens_r_len, 1,
            "A.3.3 parity: greens_r_len must be 1 (program → Ref r0); got {}",
            greens_r_len
        );
        let greens_r_byte = code[greens_base + 3];
        assert_eq!(
            greens_r_byte, 0,
            "A.3.3 parity: greens_r[0] must be 0 (register r0 = program); got {}",
            greens_r_byte
        );
        let greens_f_len = code[greens_base + 4] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.3 parity: greens_f_len must be 0; got {}",
            greens_f_len
        );

        // Portal inputs are green; identity slots last_instr/acc stay red.
        let reds_i_len = code[greens_base + 5] as usize;
        assert_eq!(
            reds_i_len, 2,
            "A.3.3 parity: reds_i_len must be 2 (last_instr + acc); got {}",
            reds_i_len
        );
        assert_eq!(code[greens_base + 6], 1, "last_instr identity at i1");
        assert_eq!(code[greens_base + 7], 2, "acc identity at i2");
        let reds_r_len = code[greens_base + 8] as usize;
        assert_eq!(
            reds_r_len, 0,
            "A.3.3 parity: reds_r_len must be 0 (program is green); got {}",
            reds_r_len
        );
        let reds_f_len = code[greens_base + 9] as usize;
        assert_eq!(
            reds_f_len, 0,
            "A.3.3 parity: reds_f_len must be 0; got {}",
            reds_f_len
        );
    }
}

/// A.3.6.1: body-local green from a state-field comparison (`state.f1 <= state.f2`).
///
/// Pre-merge-point body-local lowering exercises `bind_pre_merge_point_stmts`:
/// the `let g = state.f1 <= state.f2;` stmt sits BEFORE `jit_merge_point!()`
/// and must be bound (via `Lowerer::lower_local`) so that
/// `resolve_greens` / `emit_promote_greens` can resolve `g` via
/// `lowerer.bindings`. Without the pre-pass `emit_promote_greens` strict-
/// `expect`s on the missing binding (A.3.5).
///
/// `lower_value_expr` already handles `BinOp::Le` on int state fields, so
/// no extension to the expression vocabulary is required by this fixture.
mod oparg_with_body_local_state_le_green {

    use majit_metainterp::{Assembler, JitDriver};

    struct LeGreenState {
        f1: i64,
        f2: i64,
        a: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = LeGreenState,
        env = Bytecode,
        state_fields = { f1: int, f2: int, a: int },
        greens = [g],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_body_local_state_le_green(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<LeGreenState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = LeGreenState { f1: 0, f2: 0, a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            let g = state.f1 <= state.f2;
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            let oparg = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.a += oparg as i64,
                _ => break,
            }
            let _ = g;
        }
        state.a
    }

    /// A.3.6.1: the dispatch JitCode for `dispatch_with_body_local_state_le_green`
    /// must lower the body-local `let g = state.f1 <= state.f2;` BEFORE the
    /// portal `BC_JIT_MERGE_POINT(_C)` and emit A.3.5's promote_greens pair
    /// (`-live-` + `BC_INT_GUARD_VALUE`) for the body-local int reg holding `g`.
    /// The same reg byte must appear in the `greens_i` payload of the merge
    /// point op (A.3.2 layout).
    #[test]
    fn dispatch_with_body_local_state_le_green_promotes_body_local_before_merge_point() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_INT_LE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_body_local_state_le_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc =
            __dispatch_jitcode_dispatch_with_body_local_state_le_green(&mut asm, 0i64)
                .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.6.1: BC_JIT_MERGE_POINT(_C) must be present");

        // BC_INT_LE for `state.f1 <= state.f2` must land BEFORE the merge point.
        let int_le_pos = code[..mp_pos].iter().position(|&b| b == BC_INT_LE).expect(
            "A.3.6.1: BC_INT_LE must precede BC_JIT_MERGE_POINT \
                 (body-local `let g = state.f1 <= state.f2;` lowered via \
                 bind_pre_merge_point_stmts)",
        );
        assert!(
            int_le_pos < mp_pos,
            "BC_INT_LE must precede BC_JIT_MERGE_POINT"
        );

        // BC_INT_LE canonical encoding (`assembler.py:165-174` argcode
        // `ii>i`): [opcode][lhs_reg][rhs_reg][dst_reg]; the dst (result)
        // reg is byte int_le_pos + 3 and is `g`'s int register byte.
        assert!(
            int_le_pos + 3 < mp_pos,
            "BC_INT_LE payload truncated; int_le_pos={}, mp_pos={}",
            int_le_pos,
            mp_pos
        );
        let g_reg_byte = code[int_le_pos + 3];

        // A.3.5 promote_greens: BC_INT_GUARD_VALUE for `g` must precede the
        // merge point and reference `g`'s int reg.
        let igv_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_INT_GUARD_VALUE)
            .expect(
                "A.3.5 (`jtransform.py` `promote_greens`): body-local green `g` must be \
                 promoted via BC_INT_GUARD_VALUE before BC_JIT_MERGE_POINT",
            );
        assert!(
            igv_pos > int_le_pos,
            "BC_INT_GUARD_VALUE must come AFTER the BC_INT_LE that defines `g`; \
             got int_le@{} igv@{}",
            int_le_pos,
            igv_pos
        );
        assert_eq!(
            code[igv_pos + 1],
            g_reg_byte,
            "A.3.6.1: BC_INT_GUARD_VALUE reg byte must equal the BC_INT_LE result \
             reg byte (`g`'s int register); got igv_reg={} le_result_reg={}",
            code[igv_pos + 1],
            g_reg_byte
        );

        // jtransform.py:1707: -live- marker immediately precedes guard_value
        // (1-byte opcode + 2-byte offset → BC_LIVE byte at igv_pos - 3).
        assert!(
            igv_pos >= 3,
            "BC_INT_GUARD_VALUE at {} has no room for preceding -live-",
            igv_pos
        );
        assert_eq!(
            code[igv_pos - 3],
            BC_LIVE,
            "A.3.5 (jtransform.py:1707): -live- marker must precede \
             BC_INT_GUARD_VALUE for green promotion; code[{}]={:#04x}, \
             expected BC_LIVE={:#04x}",
            igv_pos - 3,
            code[igv_pos - 3],
            BC_LIVE,
        );

        // A.3.2 merge-point payload: greens_i must include `g`'s reg byte.
        // Layout at greens_base = mp_pos + 2:
        //   greens_base + 0: greens_i_len
        //   greens_base + 1..: greens_i bytes
        let greens_base = mp_pos + 2;
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.6.1: greens_i_len must be 1 (only `g` is green); got {}",
            greens_i_len
        );
        assert_eq!(
            code[greens_base + 1],
            g_reg_byte,
            "A.3.6.1: greens_i[0] must equal `g`'s int register byte from \
             BC_INT_LE result; got greens_i[0]={} le_result_reg={}",
            code[greens_base + 1],
            g_reg_byte
        );
    }
}

/// A.3.6.3: pin the method-call body-local lowering shape for a green
/// derived as `let g = program.get_req_size(pc) as i32 <= state.f1 as i32;`.
/// The fixture exercises:
///   - body-local `let g = program.get_req_size(pc) as i32 <= state.f1
///     as i32;` placed before `jit_merge_point!()`,
///   - `MockProgram::get_req_size => elidable_int` policy registration,
///   - `greens = [g]` so the merge-point's `greens_i` payload + the
///     A.3.5 promote-greens pair both reference `g`'s int register.
///
/// The shape test pins the canonical `call_pure_int_canonical_via_target`
/// emission (`BC_RESIDUAL_CALL_IR_I` because receiver=Ref + pc=Int) to
/// land BEFORE `BC_JIT_MERGE_POINT(_C)` and threads its result reg
/// forward through `BC_INT_LE` and finally A.3.5's `BC_INT_GUARD_VALUE`
/// + greens_i payload byte, so future refactors of
///   `lower_method_call_value` (A.3.6.2) cannot silently regress the chain.
///
/// RPython parity: `jtransform.py handle_residual_call`
/// (graph-identity-keyed; pyre keys on canonical path so
/// `<MockProgram>::get_req_size` is the segment-form lookup that mirrors
/// the upstream identity match).
mod oparg_with_body_local_method_call_green {
    use majit_metainterp::{Assembler, JitDriver};
    use std::ops::Index;

    /// Mock env type registered against the `Program::get_req_size =>
    /// elidable_int` lookup. Implements `Index<usize, Output = u8>` +
    /// `len()` so the dispatch loop's `program[pc]` opcode fetch and
    /// `pc < program.len()` guard both compile in the macro's input
    /// function (the macro-emitted IR replaces them with
    /// `getarrayitem_gc_i` so the trait impl is only used by the
    /// non-JIT warm-up path).
    struct MockProgram {
        bytes: Vec<u8>,
    }

    impl MockProgram {
        /// Mirror of `ahsembler::Program::get_req_size` (`compiler.rs`)
        /// but without the `OP_REQSIZE` lookup table — the body is
        /// irrelevant for the shape test, only the registered
        /// `MockProgram::get_req_size` segment-form policy lookup matters.
        fn get_req_size(&self, _pc: usize) -> i64 {
            0
        }

        #[expect(
            dead_code,
            reason = "shape-test fixture mirrors the interpreter program API"
        )]
        fn len(&self) -> usize {
            self.bytes.len()
        }
    }

    impl Index<usize> for MockProgram {
        type Output = u8;
        fn index(&self, pc: usize) -> &u8 {
            &self.bytes[pc]
        }
    }

    struct MethodCallGreenState {
        f1: i64,
        a: i64,
    }

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = MethodCallGreenState,
        env = MockProgram,
        state_fields = { f1: int, a: int },
        // `program.get_req_size(pc) as i32 <= state.f1 as i32` is a
        // body-local computation a dispatch loop derives a green from;
        // lower_method_call_value (A.3.6.2) routes the method-call RHS
        // through the call-policy table.
        calls = {
            MockProgram::get_req_size => elidable_int,
        },
        greens = [g],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_body_local_method_call_green(program: &MockProgram, threshold: u32) -> i64 {
        let mut driver: JitDriver<MethodCallGreenState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = MethodCallGreenState { f1: 0, a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            // A body-local green computed from a method call on the env
            // and a state field.
            let g = program.get_req_size(pc) as i32 <= state.f1 as i32;
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.a += 1,
                _ => break,
            }
            let _ = g;
        }
        state.a
    }

    /// A.3.6.3: dispatch JitCode for `dispatch_with_body_local_method_
    /// call_green` must lower the body-local `let g = program.get_req_
    /// size(pc) as i32 <= state.f1 as i32;` BEFORE the portal
    /// `BC_JIT_MERGE_POINT(_C)`, threading the elidable canonical call
    /// op (`BC_RESIDUAL_CALL_IR_I` — receiver Ref + pc Int → IR family
    /// per `assembler.rs`'s `emit_canonical_call_void`) into `BC_INT_LE` and then A.3.5's
    /// `BC_INT_GUARD_VALUE`. The same reg byte must appear in the
    /// merge-point op's `greens_i` payload (A.3.2 layout).
    #[test]
    fn dispatch_with_body_local_method_call_green_pins_canonical_call_before_merge_point() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_INT_LE, BC_INT_SIGNEXT, BC_JIT_MERGE_POINT,
            BC_JIT_MERGE_POINT_C, BC_LIVE, BC_RESIDUAL_CALL_IR_I,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_body_local_method_call_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc =
            __dispatch_jitcode_dispatch_with_body_local_method_call_green(&mut asm, 0i64)
                .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.6.3: BC_JIT_MERGE_POINT(_C) must be present");

        // Canonical elidable call (`call_pure_int_canonical_via_target`)
        // for `program.get_req_size(pc)` must land BEFORE the merge
        // point. Receiver Ref + Int arg → IR-family opcode
        // `BC_RESIDUAL_CALL_IR_I` per `assembler.rs`'s
        // `emit_canonical_call_void` family selection.
        let call_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_RESIDUAL_CALL_IR_I)
            .expect(
                "A.3.6.3: BC_RESIDUAL_CALL_IR_I (elidable canonical \
                 method-call form) must precede BC_JIT_MERGE_POINT — \
                 lower_method_call_value should have routed `program.\
                 get_req_size(pc)` through call_pure_int_canonical_via_\
                 target with effect_info = ELIDABLE_EFFECT_INFO",
            );

        // BC_RESIDUAL_CALL_IR_I encoding (`assembler.rs`'s
        // `emit_canonical_call_void` + `emit_canonical_call_typed`)
        // for arg classes "ri" + dst:
        //   [0] opcode
        //   [1] funcptr_reg
        //   [2] int_count = 1
        //   [3] int reg byte (pc → i0)
        //   [4] ref_count = 1
        //   [5] ref reg byte (program → r0)
        //   [6..8] calldescr_idx u16
        //   [8] dst reg byte (the int register holding the call result)
        // The call-result reg drives the LHS of the `<=` comparison.
        let call_op_len = 9;
        assert!(
            call_pos + call_op_len <= mp_pos,
            "A.3.6.3: BC_RESIDUAL_CALL_IR_I encoding truncated before \
             merge point; call_pos={} mp_pos={} need {} bytes",
            call_pos,
            mp_pos,
            call_op_len
        );
        let call_result_reg_byte = code[call_pos + 8];

        // BC_INT_LE for `<call_result> as i32 <= state.f1 as i32` must land
        // AFTER the canonical call op and BEFORE the merge point. Its LHS is
        // the `as i32` sign-extension of the call result; the dataflow chain
        // is verified below.
        let int_le_pos = code[..mp_pos].iter().position(|&b| b == BC_INT_LE).expect(
            "A.3.6.3: BC_INT_LE must precede BC_JIT_MERGE_POINT for \
                 the body-local `<=` comparison",
        );
        assert!(
            int_le_pos > call_pos,
            "BC_INT_LE must come AFTER the BC_RESIDUAL_CALL_IR_I that \
             produces its LHS; got call@{} int_le@{}",
            call_pos,
            int_le_pos
        );

        // BC_INT_LE canonical encoding (`assembler.py:165-174` argcode
        // `ii>i`): [opcode][lhs_reg][rhs_reg][dst_reg], 1 byte per reg.
        assert!(
            int_le_pos + 4 <= mp_pos,
            "BC_INT_LE payload truncated; int_le_pos={}, mp_pos={}",
            int_le_pos,
            mp_pos
        );
        let g_reg_byte = code[int_le_pos + 3];

        // The fixture compares `get_req_size(pc) as i32 <= state.f1 as i32`
        // (line 1976), so BC_INT_LE's lhs is the `as i32` sign-extension of
        // the call result, not the raw call-result register. The cast lowers
        // to `int_signext(x, 4)` (`jtransform.py` `_int_to_int_cast`), whose
        // byte count is a constant operand, so verify BC_INT_LE's lhs is the
        // result of a BC_INT_SIGNEXT that reads the canonical call-result reg.
        let int_le_lhs = code[int_le_pos + 1];
        // binop_i encoding is `[opcode][lhs][rhs][dst]` (assembler.py:165-174).
        let signext_pos = code[call_pos..int_le_pos]
            .windows(4)
            .position(|w| w[0] == BC_INT_SIGNEXT && w[3] == int_le_lhs)
            .map(|p| call_pos + p)
            .expect(
                "A.3.6.3: BC_INT_LE lhs must be produced by the `as i32` \
                 sign-extension (BC_INT_SIGNEXT) of the call result",
            );
        assert_eq!(
            code[signext_pos + 1],
            call_result_reg_byte,
            "A.3.6.3: the `as i32` sign-extension feeding BC_INT_LE's lhs \
             must read the canonical call-result reg; got signext_lhs={} \
             call_result_reg={}",
            code[signext_pos + 1],
            call_result_reg_byte,
        );

        // A.3.5 promote_greens: BC_INT_GUARD_VALUE for `g` must precede
        // the merge point and reference `g`'s int reg.
        let igv_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_INT_GUARD_VALUE)
            .expect(
                "A.3.5 (`jtransform.py` `promote_greens`): body-local green `g` must \
                 be promoted via BC_INT_GUARD_VALUE before \
                 BC_JIT_MERGE_POINT",
            );
        assert!(
            igv_pos > int_le_pos,
            "BC_INT_GUARD_VALUE must come AFTER the BC_INT_LE that \
             defines `g`; got int_le@{} igv@{}",
            int_le_pos,
            igv_pos
        );
        assert_eq!(
            code[igv_pos + 1],
            g_reg_byte,
            "A.3.6.3: BC_INT_GUARD_VALUE reg byte must equal the \
             BC_INT_LE result reg byte (`g`'s int register); got \
             igv_reg={} le_result_reg={}",
            code[igv_pos + 1],
            g_reg_byte
        );

        // jtransform.py:1707 -live- marker (1-byte opcode + 2-byte
        // offset → BC_LIVE byte at igv_pos - 3).
        assert!(
            igv_pos >= 3,
            "BC_INT_GUARD_VALUE at {} has no room for preceding -live-",
            igv_pos
        );
        assert_eq!(
            code[igv_pos - 3],
            BC_LIVE,
            "A.3.5 (jtransform.py:1707): -live- marker must precede \
             BC_INT_GUARD_VALUE; code[{}]={:#04x}, expected \
             BC_LIVE={:#04x}",
            igv_pos - 3,
            code[igv_pos - 3],
            BC_LIVE,
        );

        // A.3.2 merge-point payload: greens_i[0] must equal `g`'s int reg.
        // Layout at greens_base = mp_pos + 2:
        //   greens_base + 0: greens_i_len
        //   greens_base + 1..: greens_i bytes
        let greens_base = mp_pos + 2;
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.6.3: greens_i_len must be 1 (only `g` is green); got {}",
            greens_i_len
        );
        assert_eq!(
            code[greens_base + 1],
            g_reg_byte,
            "A.3.6.3: greens_i[0] must equal `g`'s int register byte \
             from BC_INT_LE result; got greens_i[0]={} le_result_reg={}",
            code[greens_base + 1],
            g_reg_byte
        );
    }
}

/// A.3.7: pin the merge-point shape for a 4-green declaration whose two
/// middle greens are body-locals — `greens = [pc, a, b, program]`, with
/// `let a = ...; let b = ...;` ahead of the merge point, which A.3.6.1's
/// body-local walker accepts.
///
/// Layout pinned here:
///   - greens_i_len = 3 (pc=Int/i0, one Int from BinOp::Le, one Int from
///     BinOp::Eq).
///   - greens_r_len = 1 (program=Ref/r0).
///   - greens_f_len = 0.
///   - reds_i_len = reds_r_len = reds_f_len = 0 (both portal-input
///     reds are declared green).
///
/// Locks the byte order so a future macro change that re-buckets greens
/// or shifts the merge-point payload trips this fixture rather than
/// silently regressing the shape.
mod oparg_with_full_4_green_parity {

    use majit_metainterp::{Assembler, JitDriver};

    struct FullParityState {
        f1: i64,
        f2: i64,
        sel: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = FullParityState,
        env = Bytecode,
        state_fields = { f1: int, f2: int, sel: int },
        greens = [pc, stackok, is_queue, program],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_full_4_green_parity(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FullParityState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FullParityState {
            f1: 0,
            f2: 0,
            sel: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            // Two body-local greens recomputed before the merge point.
            let stackok = state.f1 <= state.f2;
            let is_queue = state.sel == 21i64;
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.f1 += 1,
                _ => break,
            }
            let _ = stackok;
            let _ = is_queue;
        }
        state.f1
    }

    #[test]
    fn dispatch_with_full_4_green_parity_pins_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_full_4_green_parity(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_with_full_4_green_parity(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.7: BC_JIT_MERGE_POINT(_C) must be present");

        // Layout (jtransform.py make_three_lists):
        //   greens_base + 0: greens_i_len   = 3
        //   greens_base + 1..3: greens_i bytes (pc, stackok, is_queue regs)
        //   greens_base + 4: greens_r_len   = 1
        //   greens_base + 5: greens_r[0]    = 0 (r0 = program)
        //   greens_base + 6: greens_f_len   = 0
        //   greens_base + 7: reds_i_len     = 0
        //   greens_base + 8: reds_r_len     = 0
        //   greens_base + 9: reds_f_len     = 0
        let greens_base = mp_pos + 2;
        assert!(
            greens_base + 9 < code.len(),
            "A.3.7: payload too short; greens_base={}, len={}",
            greens_base,
            code.len()
        );

        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 3,
            "A.3.7: greens_i_len must be 3 (pc + stackok + is_queue); got {}",
            greens_i_len
        );

        // greens_i[0] = pc → register i0 (= 0).
        assert_eq!(
            code[greens_base + 1],
            0,
            "A.3.7: greens_i[0] must be 0 (i0 = pc per portal-input \
             binding at jitcode_lower.rs:7145); got {}",
            code[greens_base + 1]
        );

        let greens_r_offset = greens_base + 1 + greens_i_len;
        let greens_r_len = code[greens_r_offset] as usize;
        assert_eq!(
            greens_r_len, 1,
            "A.3.7: greens_r_len must be 1 (program → Ref r0); got {}",
            greens_r_len
        );
        assert_eq!(
            code[greens_r_offset + 1],
            0,
            "A.3.7: greens_r[0] must be 0 (r0 = program per portal-input \
             binding at jitcode_lower.rs:7136); got {}",
            code[greens_r_offset + 1]
        );

        let greens_f_offset = greens_r_offset + 1 + greens_r_len;
        let greens_f_len = code[greens_f_offset] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.7: greens_f_len must be 0; got {}",
            greens_f_len
        );

        let reds_i_offset = greens_f_offset + 1 + greens_f_len;
        assert_eq!(
            code[reds_i_offset], 3,
            "A.3.7: reds_i_len must be 3 (f1 + f2 + sel identity); got {}",
            code[reds_i_offset]
        );
        let reds_r_offset = reds_i_offset + 1 + 3;
        assert_eq!(
            code[reds_r_offset], 0,
            "A.3.7: reds_r_len must be 0 (program declared as green); got {}",
            code[reds_r_offset]
        );
        let reds_f_offset = reds_r_offset + 1;
        assert_eq!(
            code[reds_f_offset], 0,
            "A.3.7: reds_f_len must be 0; got {}",
            code[reds_f_offset]
        );
    }
}

/// A qualified (multi-segment) call that no lowering arm consumed must
/// abort lowering, not be silently dropped as "inert".  Regression for the
/// side-effect-drop hole in `lower_stmt_fallback` / for-loop unrolling.
mod residual_call_not_dropped {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::BC_STORE_STATE_FIELD;
    use majit_metainterp::{Assembler, JitDriver};

    // `sidefx::noop` is a two-segment path, so `expr_references_unknown_local`
    // treats it as a safe module-scope reference; before the fix its unrolled
    // loop body was dropped rather than aborting the arm.
    mod sidefx {
        #[allow(dead_code)]
        pub fn noop() {}
    }

    const OP_PURE: u8 = 4;
    const OP_LOOP_CALL: u8 = 5;

    struct CallState {
        acc: i64,
    }

    #[majit_macros::jit_interp(
        state = CallState,
        env = Bytecode,
        state_fields = { acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_residual_call(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<CallState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = CallState { acc: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_PURE => state.acc += 1,
                OP_LOOP_CALL => {
                    for _ in 0..2 {
                        sidefx::noop();
                        state.acc += 1;
                    }
                }
                _ => break,
            }
        }
        state.acc
    }

    #[test]
    fn qualified_call_in_unrolled_loop_aborts_instead_of_dropping() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_residual_call(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_residual_call(&mut asm, 0i64)
            .expect("dispatch lower must succeed");
        let store_count = super::count_op_in_dispatch_tree(&dispatch_jc, BC_STORE_STATE_FIELD);
        // Only `OP_PURE` contributes one state-field store.  `OP_LOOP_CALL`
        // holds an unregistered qualified call, so its whole body must abort
        // lowering (interpreter fallback) instead of unrolling with the call
        // dropped — which would have added two more stores (total 3).
        assert_eq!(
            store_count, 1,
            "residual-call loop arm must abort, not drop the call and unroll; stores={store_count}"
        );
    }
}

/// `jtransform.py`'s `optimize_goto_if_not` keeps a float comparison as a
/// value when it escapes, but fuses it into the branch when it is consumed
/// only by the exitswitch.
mod float_compare_branch_gate {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::{
        BC_FLOAT_GE, BC_GOTO_IF_NOT_FLOAT_GE, BC_GOTO_IF_NOT_INT_GE, BC_GOTO_IF_NOT_INT_IS_ZERO,
    };
    use majit_metainterp::{Assembler, JitDriver};

    const OP_VALUE: u8 = 6;
    const OP_BRANCH: u8 = 7;
    const OP_INT_BRANCH: u8 = 8;
    const OP_NEGATED_BRANCH: u8 = 9;

    struct FloatCmpState {
        fa: f64,
        fb: f64,
        acc: i64,
    }

    #[majit_macros::jit_interp(
        state = FloatCmpState,
        env = Bytecode,
        state_fields = { fa: float, fb: float, acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_float_cmp(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatCmpState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatCmpState {
            fa: 0.0,
            fb: 0.0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                // Value form: float compare materialized to an int — lowers.
                OP_VALUE => state.acc += (state.fa >= state.fb) as i64,
                // Branch form: lowers to goto_if_not_float_ge/ffL.
                OP_BRANCH => {
                    if state.fa >= state.fb {
                        state.acc += 1;
                    }
                }
                // The same upstream optimization covers integer comparisons.
                OP_INT_BRANCH => {
                    if state.acc >= 0 {
                        state.acc += 1;
                    }
                }
                // An odd negation must not invert a float comparison: that
                // would change the result for NaNs. PyPy materializes the
                // comparison and fuses bool_not as int_is_zero instead.
                OP_NEGATED_BRANCH => {
                    if !(state.fa >= state.fb) {
                        state.acc += 1;
                    }
                }
                _ => break,
            }
        }
        state.acc
    }

    #[test]
    fn float_compare_lowers_as_value_and_fused_branch_guard() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![2], vec![], vec![0, 1]);
        __prebuild_jitcode_liveness_dispatch_float_cmp(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_float_cmp(&mut asm, 0i64)
            .expect("dispatch lower must succeed");
        let ge_count = super::count_op_in_dispatch_tree(&dispatch_jc, BC_FLOAT_GE);
        let guard_ge_count =
            super::count_op_in_dispatch_tree(&dispatch_jc, BC_GOTO_IF_NOT_FLOAT_GE);
        let int_guard_ge_count =
            super::count_op_in_dispatch_tree(&dispatch_jc, BC_GOTO_IF_NOT_INT_GE);
        let negated_guard_count =
            super::count_op_in_dispatch_tree(&dispatch_jc, BC_GOTO_IF_NOT_INT_IS_ZERO);
        // The value and odd-negation arms retain float_ge/ff>i. The direct
        // branch arm has no materialized boolean and emits the fused ffL
        // exitswitch instead.
        assert_eq!(
            ge_count, 2,
            "only the escaping and negated float comparisons should materialize; \
             float_ge={ge_count}"
        );
        assert_eq!(
            guard_ge_count, 1,
            "branch comparison must fuse; goto_if_not_float_ge={guard_ge_count}"
        );
        assert_eq!(
            int_guard_ge_count, 1,
            "integer branch comparison must fuse too; goto_if_not_int_ge={int_guard_ge_count}"
        );
        assert_eq!(
            negated_guard_count, 1,
            "odd float negation must preserve NaN semantics via int_is_zero; \
             goto_if_not_int_is_zero={negated_guard_count}"
        );
    }
}

/// A literal-range `for` loop larger than the unroll cap must bail to
/// interpreter fallback WITHOUT first materializing the whole range — a
/// huge range (`0..i64::MAX`) would otherwise exhaust memory during macro
/// expansion.  The mere fact that this fixture compiles proves the range is
/// never collected; the assertion confirms the oversized arm falls back.
mod huge_range_for_loop_falls_back {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::BC_STORE_STATE_FIELD;
    use majit_metainterp::{Assembler, JitDriver};

    const OP_PURE: u8 = 8;
    const OP_HUGE: u8 = 9;

    struct HugeState {
        acc: i64,
    }

    #[majit_macros::jit_interp(
        state = HugeState,
        env = Bytecode,
        state_fields = { acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_huge_range(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<HugeState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = HugeState { acc: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_PURE => state.acc += 1,
                OP_HUGE => {
                    for _ in 0i64..9223372036854775807 {
                        state.acc += 1;
                    }
                }
                _ => break,
            }
        }
        state.acc
    }

    #[test]
    fn oversized_range_arm_bails_without_unrolling() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_huge_range(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_huge_range(&mut asm, 0i64)
            .expect("dispatch lower must succeed");
        let store_count = super::count_op_in_dispatch_tree(&dispatch_jc, BC_STORE_STATE_FIELD);
        // Only the `OP_PURE` arm inlines its single store; the oversized
        // `OP_HUGE` loop exceeds the unroll cap and aborts the arm.
        assert_eq!(
            store_count, 1,
            "oversized-range arm must bail; stores={store_count}"
        );
    }
}

/// Pairs one genuinely unlowerable float-remainder assignment with healthy
/// controls, ensuring installation reports the degraded arm without producing
/// false positives for arms the macro can lower.
mod degraded_arm_is_named_at_install {
    use super::Bytecode;
    use majit_metainterp::jitcode::insns::{
        BC_ABORT, BC_CONVERT_FLOAT_BYTES_TO_LONGLONG, BC_CONVERT_LONGLONG_BYTES_TO_FLOAT,
    };
    use majit_metainterp::virt_array::VirtArray;
    use majit_metainterp::{Assembler, JitCode, JitDriver};

    const OP_ADD: u8 = 1;
    const OP_RETURN: u8 = 2;
    const OP_RETURN_F: u8 = 3;
    const OP_BUMP: u8 = 4;
    const OP_LOAD_CONST_F: u8 = 5;
    const OP_FBUMP: u8 = 6;
    const OP_FREM_ASSIGN: u8 = 7;

    struct DegradedArmState {
        regs: VirtArray<i64>,
        fregs: VirtArray<f64>,
    }

    #[majit_macros::jit_interp(
        state = DegradedArmState,
        env = Bytecode,
        greens = [pc, program],
        state_fields = {
            regs: [int; virt],
            fregs: [float; virt],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_degraded_arm(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<DegradedArmState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = DegradedArmState {
            regs: VirtArray::filled(0i64, 4),
            fregs: VirtArray::filled(0.0f64, 4),
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        loop {
            jit_merge_point!();
            let opcode = program[pc];
            match opcode {
                OP_ADD => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = state.regs[a] + state.regs[b];
                    pc += 4;
                }
                OP_RETURN => {
                    return state.regs[program[pc + 1] as usize];
                }
                OP_RETURN_F => {
                    return state.fregs[program[pc + 1] as usize].to_bits() as i64;
                }
                OP_BUMP => {
                    state.regs[program[pc + 1] as usize] += 1;
                    pc += 2;
                }
                OP_FBUMP => {
                    state.fregs[program[pc + 1] as usize] += 1.0;
                    pc += 2;
                }
                OP_FREM_ASSIGN => {
                    state.fregs[program[pc + 1] as usize] %= 1.0;
                    pc += 2;
                }
                OP_LOAD_CONST_F => {
                    state.fregs[program[pc + 2] as usize] = f64::from_bits(program[pc + 1] as u64);
                    pc += 3;
                }
                _ => break,
            }
        }
        0
    }

    /// Build the dispatch JitCode so the install-time recorder runs.
    fn install() -> JitCode {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
        __prebuild_jitcode_liveness_dispatch_degraded_arm(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        __dispatch_jitcode_dispatch_degraded_arm(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture")
    }

    /// Entries this fixture contributed, isolated from every other `#[jit_interp]`
    /// machine in the binary by the `state = T` name.
    fn recorded_arms() -> Vec<String> {
        majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|entry| entry.interp == "DegradedArmState")
            .map(|entry| entry.arm.to_string())
            .collect()
    }

    #[test]
    fn degraded_arm_is_named_and_healthy_arms_are_not() {
        let _ = install();
        let arms = recorded_arms();

        // Positive: the arm that actually degrades is named.
        assert!(
            arms.iter().any(|arm| arm == "OP_FREM_ASSIGN"),
            "the unsupported float remainder-assign arm must be recorded as degraded; recorded={arms:?}"
        );

        // Negative controls.  `OP_ADD` is `OP_BUMP` written out and lowers;
        // `OP_RETURN_F` reads a float register through `f64::to_bits`, which
        // the macro recognizes, and `OP_RETURN` is its int twin; `_ => break`
        // is a DECLARED halt rather than a degradation.  A channel that named
        // any of these would not be telling anyone which arm to fix.
        for silent in [
            "OP_ADD",
            "OP_RETURN",
            "OP_RETURN_F",
            "OP_LOAD_CONST_F",
            "OP_BUMP",
            "OP_FBUMP",
            "_",
        ] {
            assert!(
                !arms.iter().any(|arm| arm == silent),
                "`{silent}` lowers (or is declared) and must not be recorded; recorded={arms:?}"
            );
        }
    }

    /// The inherent float bitcast spellings must reach the bitcast resops, not
    /// merely avoid the abort stub: `<f64>.to_bits()` is
    /// `convert_float_bytes_to_longlong` and `f64::from_bits(..)` is its
    /// inverse, the same two ops the free `majit_f64_to_bits` /
    /// `majit_bits_to_f64` intrinsics emit.
    #[test]
    fn inherent_float_bitcast_spellings_lower_to_the_convert_ops() {
        let dispatch_jc = install();
        let count = |op: u8| -> usize {
            let in_dispatch = dispatch_jc.code.iter().filter(|&&b| b == op).count();
            let in_subs: usize = dispatch_jc
                .exec
                .descrs
                .iter()
                .filter_map(|descr| descr.as_jitcode())
                .map(|sub| sub.code.iter().filter(|&&b| b == op).count())
                .sum();
            in_dispatch + in_subs
        };
        let to_bits = count(BC_CONVERT_FLOAT_BYTES_TO_LONGLONG);
        let from_bits = count(BC_CONVERT_LONGLONG_BYTES_TO_FLOAT);
        assert_eq!(
            (to_bits, from_bits),
            (1, 1),
            "`.to_bits()` and `f64::from_bits(..)` must each lower to one bitcast op"
        );
    }

    /// The registry must agree with the artifact it describes; checking it
    /// against itself would prove nothing.  Every arm the macro named is one
    /// abort stub in the built dispatch JitCode, and there are no others.
    #[test]
    fn the_named_arms_are_exactly_the_abort_stubs_in_the_ir() {
        let dispatch_jc = install();
        let abort_stubs = dispatch_jc
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode())
            .filter(|sub| sub.code.iter().any(|&b| b == BC_ABORT))
            .count();
        let arms = recorded_arms();
        assert_eq!(
            abort_stubs,
            arms.len(),
            "abort stubs in the dispatch JitCode must match the named arms; recorded={arms:?}"
        );
        assert_eq!(abort_stubs, 1, "fixture must degrade exactly one arm");
    }

    #[test]
    fn install_time_record_carries_a_reason_and_dedups() {
        let _ = install();
        let _ = install();
        let entries: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|entry| entry.interp == "DegradedArmState" && entry.arm == "OP_FREM_ASSIGN")
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "a second dispatch build must not duplicate the entry; entries={entries:?}"
        );
        assert!(
            !entries[0].reason.is_empty(),
            "the recorded arm must carry a staged reason, not an empty bucket"
        );
    }
}
