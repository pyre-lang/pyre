//! A residual call that sits after a red-conditional branch inside one opcode
//! stops running once a bridge is compiled for that branch.
//!
//! The interpreter shape is the smallest one that can hold the defect: a single
//! scalar state field, one opcode whose body is `if <red condition> { <call> }`,
//! and a back edge. Nothing here is exotic — no virtualizable array, no virtual
//! object, no exception path — so any `jit_interp` interpreter with a
//! conditionally-effectful opcode is in range.
//!
//! What the run must produce is arithmetic, not a crash. `OP_PUSH` hands the
//! displaced word to `chain_push` on every trip past `CAP`, so after `N` pushes
//! the chain has seen exactly `N - CAP` words and their sum is fixed. Both are
//! computed twice: once with the threshold out of reach, which fixes the
//! answer, and once warm.
//!
//! It measured 202 of 1992 before the fix, with `N = 2000` and `CAP = 8`: the
//! 202 are the trips before the branch earned a bridge, and from the bridge
//! onwards the call never ran again. `MAJIT_NO_BRIDGE=1` and
//! `MAJIT_MAX_BRIDGES=0` both made it pass, which placed the loss in bridge
//! tracing rather than in the guard, the recovery layout or the blackhole
//! resume.
//!
//! The mechanism is a position mismatch. The failing guard's resume state is
//! mid-opcode — its failargs carry `sp` ALREADY incremented, so the point it
//! describes is after `state.sp = state.sp + 1` and before the call.
//! `JitDriver::start_bridge_tracing` seeds the bridge with
//! `state.build_meta(resume_pc, env)`, and `resume_pc` is a green pc, which can
//! only ever name an opcode BOUNDARY. So the bridge begins at the next opcode
//! and the tail of the one that failed — here the call, and the store after it
//! — is never recorded. `resume_in_blackhole` does not have this problem
//! because it resumes at a jitcode position rather than a pc, which is why
//! turning bridges off repairs the count.
use core::sync::atomic::{AtomicU32, Ordering};

pub type Bytecode = [u8];

/// The depth past which a push displaces a word. Small, so the branch flips
/// early and stays taken for the rest of the run.
const CAP: usize = 8;

const OP_PUSH: u8 = 1;
const OP_BACK: u8 = 2;
const OP_RET: u8 = 3;

static COMPILES: AtomicU32 = AtomicU32::new(0);
/// Bridges compiled. The counts below hold whether or not a bridge is ever
/// built — declining to bridge keeps them right too — so without this the
/// fixture cannot tell the repair from the refusal.
static BRIDGES: AtomicU32 = AtomicU32::new(0);

/// Counts and sums what it is handed. A count alone cannot tell a lost word
/// from a duplicated one; the sum can.
#[repr(C)]
struct Chain {
    size: i64,
    sum: i64,
}

extern "C" fn chain_push(chain: usize, value: i64) {
    let chain = unsafe { &mut *(chain as *mut Chain) };
    chain.size += 1;
    chain.sum += value;
}

struct PlainStack {
    /// The one word the machine keeps; everything below it has gone to the
    /// chain.
    top: i64,
    sp: usize,
    counter: i64,
    chain: usize,
}

#[majit_macros::jit_interp(
    state = PlainStack,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        top: int,
        sp: int(usize),
        counter: int,
        chain: ref(Chain),
    },
    calls = {
        chain_push => residual_void,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, iterations: i64, threshold: u32, chain: usize) -> i64 {
    let mut driver: majit_metainterp::JitDriver<PlainStack> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _before, _after, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_compile_bridge(|_green_key, _descr_id, _num_ops| {
        BRIDGES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = PlainStack {
        top: 0i64,
        sp: 0usize,
        counter: iterations,
        chain,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            OP_PUSH => {
                // A red, and a different one each trip, so a word that never
                // reaches the chain is missing from the sum rather than hidden
                // by a repeat.
                let value = state.counter;
                state.sp = state.sp + 1;
                if state.sp > CAP {
                    // The displaced word. This call and the store below are
                    // the same opcode's work as the test above them, so a
                    // guard between them owes its resume all three.
                    chain_push(state.chain, state.top);
                }
                state.top = value;
            }
            OP_BACK => {
                state.counter = state.counter - 1;
                if state.counter != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_RET => break,
            _ => unreachable!(),
        }
    }

    state.sp as i64
}

fn run(program: &[u8], iterations: i64, threshold: u32) -> (i64, i64, i64) {
    let mut chain = Chain { size: 0, sum: 0 };
    let height = mainloop(
        program,
        iterations,
        threshold,
        &mut chain as *mut Chain as usize,
    );
    (height, chain.size, chain.sum)
}

#[test]

fn a_residual_call_after_a_mid_opcode_guard_survives_a_bridge() {
    let program = vec![OP_PUSH, OP_BACK, OP_RET];
    // Deep enough that the branch is taken for far longer than the
    // trace-eagerness window it needs to earn a bridge.
    const ITERATIONS: i64 = 2000;

    COMPILES.store(0, Ordering::Relaxed);
    BRIDGES.store(0, Ordering::Relaxed);
    // Unreachable threshold, so this run fixes the answer without the JIT.
    let cold = run(&program, ITERATIONS, u32::MAX);
    assert_eq!(
        COMPILES.load(Ordering::Relaxed),
        0,
        "the cold run must not have compiled anything"
    );
    assert_eq!(
        cold.1,
        ITERATIONS - CAP as i64,
        "cold run: every push past the first {CAP} displaces one word"
    );

    let warm = run(&program, ITERATIONS, 3);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the loop did not compile"
    );
    assert_eq!(
        warm.1, cold.1,
        "a conditional residual call inside an opcode ran a different NUMBER of \
         times compiled than interpreted"
    );
    assert_eq!(
        warm.2, cold.2,
        "a conditional residual call inside an opcode was handed different WORDS \
         compiled than interpreted"
    );
    assert_eq!(warm.0, cold.0, "compiled height");
    assert!(
        BRIDGES.load(Ordering::Relaxed) >= 1,
        "the mid-opcode guard sourced no bridge, so the counts above agree for \
         the wrong reason: they say the tail still ran, not that it ran from \
         inside a bridge"
    );
}
