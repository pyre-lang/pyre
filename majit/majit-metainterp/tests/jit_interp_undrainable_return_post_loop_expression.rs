//! Ensures a portal whose return type carries no finish projection leaves its
//! post-loop epilogue to native execution.
//!
//! `finish_return_for` recognises the integer and float widths and nothing
//! else, and both drains that keep a lowered epilogue from running a second
//! time — `FinishReturn::drain` after the back edge and
//! `FinishReturn::drain_single_pass` at the single-executor close — are
//! emitted only when it returns `Some`. A `bool`-returning portal gets neither,
//! so lowering its epilogue would publish a FINISH nothing drains: the close
//! breaks the native loop and the suffix runs again, applying its state
//! mutation twice.
//!
//! The sibling `jit_interp_halt_arm_post_loop_expression` covers the same
//! epilogue-once property for the `i64` return that does have a projection.

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

/// Loops majit compiled during the most recent [`run`] — evidence the JIT tier
/// was alive for it. Process-global, so every read is bracketed by
/// [`PROBE_LOCK`]; a load taken outside that window observes another test's
/// compile.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

struct UndrainablePostLoopState {
    acc: i64,
    cnt: i64,
    /// How many times the epilogue has run. The portal's answer is this
    /// counter's post-increment value compared against one, so a doubled
    /// epilogue is what the returned `bool` reports rather than something a
    /// separate observer has to reach in and read.
    ran: i64,
}

#[majit_macros::jit_interp(
    state = UndrainablePostLoopState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        acc: int,
        cnt: int,
        ran: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_undrainable_post_loop(program: &Bytecode, threshold: u32, n: i64) -> bool {
    let mut driver: JitDriver<UndrainablePostLoopState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_, _, _, _| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = UndrainablePostLoopState {
        acc: 0,
        cnt: n,
        ran: 0,
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
    // Same shape as the `i64` sibling's epilogue — a scalar state store ahead
    // of the value the portal yields — with the store made self-reporting so
    // the `bool` return is itself the once-or-twice evidence.
    state.ran += 1;
    state.ran == 1
}

/// `[OP_ADD, OP_DEC, OP_BACK, 0, OP_END]`.
fn program() -> Vec<u8> {
    vec![OP_ADD, OP_DEC, OP_BACK, 0, OP_END]
}

/// Run one `(threshold, n)` pair, returning the answer and the number of loops
/// compiled during it. The counter is reset *and* read under [`PROBE_LOCK`];
/// reading after the guard drops would reintroduce exactly the race the lock
/// exists to remove.
fn run(threshold: u32, n: i64) -> (bool, usize) {
    let _guard = PROBE_LOCK.lock();
    COMPILES.store(0, Ordering::Relaxed);
    let got = dispatch_undrainable_post_loop(&program(), threshold, n);
    (got, COMPILES.load(Ordering::Relaxed))
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_undrainable_post_loop(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_undrainable_post_loop(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// The fixture is only meaningful while the dispatch actually lowers.
#[test]
fn dispatch_lowers() {
    let _ = install();
}

/// The post-loop epilogue must be applied exactly once.
///
/// Each pair satisfies `n == threshold + 1`, the only relation under which the
/// walk is still recording when the loop exits, so the `Finish` regime — the
/// one whose drain a `bool` return does not have — is the regime under test.
/// The compile assertion is not decoration: with no loop compiled the walk
/// never reaches the `break`, and the case would pass while testing nothing.
#[test]
fn an_undrainable_return_does_not_run_the_post_loop_epilogue_twice() {
    for (threshold, n) in [(2u32, 3i64), (4, 5), (8, 9)] {
        let (got, compiles) = run(threshold, n);
        assert!(
            compiles >= 1,
            "threshold={threshold} n={n}: no loop compiled, so the walk never \
             reached the `break` and this case tests nothing"
        );
        assert!(
            got,
            "threshold={threshold} n={n}: the epilogue ran more than once — the \
             walk lowered it and published a FINISH, and a `bool` return has no \
             drain to consume it, so the close broke the native loop and the \
             suffix ran again"
        );
    }
}

/// The answer must hold at every trip count, not only at the three the test
/// above singles out. This sweep spans both regimes — the `CloseLoop` walks and
/// the `Finish` walks — so it also guards against a fix that traded one regime
/// for the other.
#[test]
fn every_threshold_and_trip_count_runs_the_epilogue_once() {
    for threshold in [2u32, 4, 8] {
        for n in 2i64..30 {
            let (got, _) = run(threshold, n);
            assert!(got, "threshold={threshold} n={n}");
        }
    }
}
