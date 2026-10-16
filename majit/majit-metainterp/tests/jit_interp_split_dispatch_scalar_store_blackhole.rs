//! A split-dispatch int scalar store must survive the abort blackhole.
//!
//! `split_dispatch` lowers a pure forward-advancing arm to a sub-JitCode
//! called with `inline_call_*`. The arm body `state.acc = state.acc + K`
//! emits `getfield_vable_i` / `setfield_vable_i` inside that callee
//! (`jtransform.py` `rewrite_op_getfield` / `rewrite_op_setfield` on the
//! virtualizable; `pyjitpl.py` `opimpl_setfield_vable_*` writes
//! `metainterp.virtualizable_boxes`, shared by every frame of the portal).
//! `acc` is a vable field (`state_fields` has no `[..; virt]` array, but
//! `split_dispatch` takes the same state-as-vable path).
//!
//! The abort path is `pyjitpl.py` `run_blackhole_interp_to_cancel_tracing`
//! → `JitDriver::run_pending_abort_blackhole` → `blackhole.py`
//! `convert_and_run_from_pyjitpl`. `warmspot.py` `handle_jitexception`
//! resumes at the merge point the chain reaches. `trace_limit` 2 is the
//! same cut `abort_blackhole_virt_array` uses. The unrolled `pad` writes
//! keep that abort in the portal so the blackhole — not the tracer, not
//! compiled code — executes the arm through `interpret_unresolved_inline_call`.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use majit_metainterp::jitcode::insns::{
    BC_INLINE_CALL, BC_INLINE_CALL_IR_I, BC_INLINE_CALL_IR_V, BC_SETFIELD_VABLE_I,
};
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const OP_ADD: u8 = 1;
const OP_LOOP: u8 = 2;
const ADD_BY: i64 = 7;
const N: i64 = 40;

struct ScalarAcc {
    acc: i64,
    ticks: i64,
    pad: i64,
}

static ABORTS: AtomicU32 = AtomicU32::new(0);
static ABORT_TOO_LONG: AtomicU64 = AtomicU64::new(0);
static COMPILES: AtomicU32 = AtomicU32::new(0);

#[majit_macros::jit_interp(
    state = ScalarAcc,
    env = Bytecode,
    split_dispatch = true,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int, pad: int },
)]
#[allow(unused_assignments, unused_variables)]
fn scalar_acc_loop(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<ScalarAcc> = JitDriver::new(threshold);
    driver
        .meta_interp_mut()
        .warm_state_for_driver(0)
        .set_param("trace_limit", 2);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_trace_abort(|_gk, _permanent| {
        ABORTS.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = ScalarAcc {
        acc: 0,
        ticks,
        pad: 0,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        // Unrolled portal-side identity stores. `trace_limit` 2 is checked
        // only at interpret-loop safe points, so two pad writes are not
        // enough to abort before the tracer inlines OP_ADD. Sixteen writes
        // keep `convert_and_run_from_pyjitpl` in the portal until it hits
        // the arm `inline_call`.
        for _ in 0..16 {
            state.pad = state.pad + 1i64;
        }
        let opcode = program[pc];
        match opcode {
            OP_ADD => {
                state.acc = state.acc + ADD_BY;
                pc += 1;
            }
            OP_LOOP => {
                state.ticks = state.ticks - 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
                pc += 1;
            }
            _ => break,
        }
    }
    ABORT_TOO_LONG.store(driver.abort_diag(0), Ordering::Relaxed);
    state.acc
}

fn program() -> Vec<u8> {
    vec![OP_ADD, OP_LOOP]
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_scalar_acc_loop(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_scalar_acc_loop(&mut asm, 0i64).expect("dispatch lower must succeed")
}

fn dispatch_subs(jc: &JitCode) -> impl Iterator<Item = &JitCode> {
    jc.exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode().map(|sub| sub.as_ref()))
}

#[test]
fn split_dispatch_add_arm_stores_acc_in_a_sub_jitcode() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("ScalarAcc");

    let dispatch_has_inline = dispatch_jc.code.contains(&BC_INLINE_CALL)
        || dispatch_jc.code.contains(&BC_INLINE_CALL_IR_I)
        || dispatch_jc.code.contains(&BC_INLINE_CALL_IR_V);
    assert!(
        dispatch_has_inline,
        "split_dispatch must emit an arm inline_call; dispatch={:?}",
        dispatch_jc.code
    );
    assert!(
        dispatch_jc.code.contains(&BC_SETFIELD_VABLE_I),
        "pre-dispatch pad stores must land in the portal JitCode so the \
         abort sits before the arm inline_call; dispatch={:?}",
        dispatch_jc.code
    );

    let writers: Vec<&JitCode> = dispatch_subs(&dispatch_jc)
        .filter(|code| code.code.contains(&BC_SETFIELD_VABLE_I))
        .collect();
    assert!(
        !writers.is_empty(),
        "the OP_ADD arm sub-JitCode must emit setfield_vable_i; dispatch={:?} subs={:?}",
        dispatch_jc.code,
        dispatch_subs(&dispatch_jc)
            .map(|c| &c.code)
            .collect::<Vec<_>>()
    );
}

#[test]
fn abort_blackhole_split_arm_store_matches_the_interpreter() {
    let program = program();
    let expected = ADD_BY * N;

    let interp = scalar_acc_loop(&program, u32::MAX, N);
    assert_eq!(
        interp, expected,
        "interpreter adds {ADD_BY} once per loop trip"
    );

    ABORTS.store(0, Ordering::Relaxed);
    ABORT_TOO_LONG.store(0, Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let traced = scalar_acc_loop(&program, 3, N);

    assert_eq!(
        COMPILES.load(Ordering::Relaxed),
        0,
        "trace_limit 2 must abort before a loop compiles, so the arm runs in \
         the blackhole rather than compiled code"
    );
    assert!(
        ABORTS.load(Ordering::Relaxed) >= 1 || ABORT_TOO_LONG.load(Ordering::Relaxed) >= 1,
        "TraceCtx::is_too_long must abort so run_pending_abort_blackhole \
         reaches interpret_unresolved_inline_call; aborts={} abort_too_long={}",
        ABORTS.load(Ordering::Relaxed),
        ABORT_TOO_LONG.load(Ordering::Relaxed),
    );
    assert_eq!(
        traced, interp,
        "split-arm setfield_vable_i through interpret_unresolved_inline_call \
         must match the interpreter: traced acc={traced} interp acc={interp}"
    );
}
