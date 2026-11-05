//! Ensures a `break` arm leaves post-loop work to native execution.
//!
//! The fixture gives the trailing expression an observable state mutation, so
//! routing a halt through that expression during tracing would apply the
//! epilogue twice.

use std::sync::atomic::{AtomicUsize, Ordering};

use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

/// `acc += cnt`.
const OP_ADD: u8 = 1;
/// `cnt -= 1`.
const OP_DEC: u8 = 2;
/// `[OP_BACK, target]` — jump to `target` while `cnt > 0`, else fall past.
const OP_BACK: u8 = 3;
/// Bare `break` arm, the spelling that classifies as `ArmPattern::Halt`.
const OP_END: u8 = 4;

/// What the post-loop expression adds. A round number, so a doubled
/// application is unmistakable in the failure message rather than reading like
/// an off-by-one somewhere else.
const EPILOGUE_BONUS: i64 = 100;

/// Loops majit compiled during the most recent [`run`] — evidence the JIT tier
/// was alive for it. Process-global, so every read is bracketed by
/// [`PROBE_LOCK`]; a load taken outside that window observes another test's
/// compile.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

struct HaltPostLoopState {
    acc: i64,
    cnt: i64,
}

#[majit_macros::jit_interp(
    state = HaltPostLoopState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        acc: int,
        cnt: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_halt_post_loop(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<HaltPostLoopState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = HaltPostLoopState { acc: 0, cnt: n };
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
            OP_ADD => {
                state.acc += state.cnt;
            }
            OP_DEC => {
                state.cnt -= 1;
            }
            OP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.cnt > 0 {
                    if target < pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            _ => break,
        }
    }
    // The post-loop expression, shaped like tlc's: a trailing `if` whose taken
    // branch STORES to a scalar state field before yielding the value. Running
    // it twice adds `EPILOGUE_BONUS` twice.
    if state.cnt < 0 {
        0
    } else {
        state.acc += EPILOGUE_BONUS;
        state.acc
    }
}

/// `[OP_ADD, OP_DEC, OP_BACK, 0, OP_END]`.
fn program() -> Vec<u8> {
    vec![OP_ADD, OP_DEC, OP_BACK, 0, OP_END]
}

/// What the same program computes with no JIT tier involved at all, post-loop
/// expression applied exactly once.
fn interpret(n: i64) -> i64 {
    let (mut acc, mut cnt) = (0i64, n);
    loop {
        acc += cnt;
        cnt -= 1;
        if cnt > 0 {
            continue;
        }
        break;
    }
    if cnt < 0 { 0 } else { acc + EPILOGUE_BONUS }
}

/// Run one `(threshold, n)` pair, returning the answer and the number of loops
/// compiled during it. The counter is reset *and* read under [`PROBE_LOCK`];
/// reading after the guard drops would reintroduce exactly the race the lock
/// exists to remove.
fn run(threshold: u32, n: i64) -> (i64, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    let got = dispatch_halt_post_loop(&program(), threshold, n);
    (got, COMPILES.load(Ordering::Relaxed))
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_halt_post_loop(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_halt_post_loop(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// The fixture is only meaningful while the dispatch actually lowers.
#[test]
fn dispatch_lowers() {
    let _ = install();
}

/// The post-loop expression must be applied exactly once.
///
/// Each pair satisfies `n == threshold + 1`, the only relation under which the
/// walk is still recording when the loop exits — the fixture gives the
/// trailing expression an observable state mutation, so a halt routed through
/// it during tracing would apply the epilogue twice. The compile
/// assertion is not decoration: at any other `n` this program answers correctly
/// even with the defect present, so a fixture that quietly stopped tracing
/// would pass while testing nothing.
#[test]
fn a_halt_arm_does_not_run_the_post_loop_expression() {
    for (threshold, n) in [(2u32, 3i64), (4, 5), (8, 9)] {
        let (got, compiles) = run(threshold, n);
        assert!(
            compiles >= 1,
            "threshold={threshold} n={n}: no loop compiled, so the walk never \
             reached the `break` and this case tests nothing"
        );
        assert_eq!(
            got,
            interpret(n),
            "threshold={threshold} n={n}: the `break` arm diverted through \
             `default_label`, so the walk ran the post-loop expression and the \
             write-back carried its store into native state, which then ran it \
             a second time"
        );
    }
}

/// The answer must be right at every trip count, not only at the three the
/// test above singles out. This sweep spans both regimes — the `CloseLoop`
/// walks that the whole example corpus exercises, and the three `Finish` walks
/// — so it also guards against a fix that traded one regime for the other.
#[test]
fn every_threshold_and_trip_count_answers_correctly() {
    for threshold in [2u32, 4, 8] {
        for n in 2i64..30 {
            let (got, _) = run(threshold, n);
            assert_eq!(got, interpret(n), "threshold={threshold} n={n}");
        }
    }
}
