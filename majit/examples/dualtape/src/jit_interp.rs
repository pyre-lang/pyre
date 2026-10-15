/// JIT-enabled two-tape interpreter using `#[jit_interp]` and two
/// virtualizable integer arrays.
///
/// The program counter and bytecode are green inputs. Each tape contributes
/// its pointer, length, and symbolic elements to the loop state.
use majit_metainterp::virt_array::VirtArray;
pub type Bytecode = [u8];

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Hot loops majit compiled, and the optimized op count of the last one.
/// A count alone cannot distinguish a useful body from an empty or segmented
/// trace; [`LAST_HAS_JUMP`] and [`LAST_ALWAYS_FAILS`] record that shape.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);
pub static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);
/// Shape of the last compiled body, for `jit_tier_shape_gate`.
///
/// The op count cannot tell an empty dispatch from a segmented runaway (see
/// that gate). These two can: they are `LoopBodyShape`'s fields, recorded off
/// the same hook.
///
/// The count was once documented as sufficient on its own; the shape gate
/// refuted that. One integer represents at least three states: `1` is an empty
/// dispatch whose whole body is `Finish()`, `5` is a segmented runaway, and a
/// useful loop has some other count. The count therefore reports a body's size,
/// while these flags report its shape; `COMPILES > 0` alone proves neither.
pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

const TAPE_SIZE: usize = 8;
const DEFAULT_THRESHOLD: u32 = 3;

struct DualState {
    pa: i64,
    a: VirtArray<i64>,
    pb: i64,
    b: VirtArray<i64>,
}

/// Uses split dispatch to exercise two virtualizable tapes and scalar tape
/// pointers through lowering and bridge setup.
#[majit_macros::jit_interp(
    state = DualState,
    env = Bytecode,
    split_dispatch = true,
    greens = [pc, program],
    state_fields = {
        pa: int,
        a: [int; virt],
        pb: int,
        b: [int; virt],
    },
)]
fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<DualState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_green_key, _ops_before, ops_after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        LAST_HAS_JUMP.store(shape.has_jump, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(shape.has_always_fails, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = DualState {
        pa: 0,
        a: VirtArray::filled(0i64, TAPE_SIZE),
        pb: 0,
        b: VirtArray::filled(0i64, TAPE_SIZE),
    };

    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        if pc >= program.len() {
            break;
        }
        // Keep observer/replay until `OP_LOOP_END` lowers to a sub-JitCode.
        // Single-executor would otherwise stop at that abort stub before the
        // remaining program executes. `jit_tier_liveness_gate` checks the
        // current non-compiling behavior; `jit_tier_shape_gate` is ignored
        // until a useful loop body can compile.
        jit_merge_point!();
        let ch = program[pc];

        match ch {
            b'>' => {
                state.pa += 1;
                pc += 1;
            }
            b'<' => {
                state.pa -= 1;
                pc += 1;
            }
            b'+' => {
                state.a[state.pa as usize] += 1;
                pc += 1;
            }
            b'-' => {
                state.a[state.pa as usize] -= 1;
                pc += 1;
            }
            b'}' => {
                state.pb += 1;
                pc += 1;
            }
            b'{' => {
                state.pb -= 1;
                pc += 1;
            }
            b'*' => {
                state.b[state.pb as usize] += 1;
                pc += 1;
            }
            b'/' => {
                state.b[state.pb as usize] -= 1;
                pc += 1;
            }
            b'[' => {
                if state.a[state.pa as usize] == 0 {
                    let mut need: i32 = 1;
                    let mut p = pc + 1;
                    while need > 0 {
                        if program[p] == b']' {
                            need -= 1;
                        } else if program[p] == b'[' {
                            need += 1;
                        }
                        p += 1;
                    }
                    pc = p;
                } else {
                    pc += 1;
                }
            }
            b']' if state.a[state.pa as usize] != 0 => {
                // rpython/jit/tl/braininterp.py
                // `BrainInterpreter.interp_char`: keep the matching-bracket
                // scan in the opcode body. Pulling it into a Rust helper turns
                // an ordinary translated loop into an opaque slice-argument
                // call and degrades this dispatch arm to an abort JitCode.
                let mut need: i32 = 1;
                let mut target = pc - 1;
                while need > 0 {
                    if program[target] == b']' {
                        need += 1;
                    } else if program[target] == b'[' {
                        need -= 1;
                    }
                    if need > 0 {
                        target -= 1;
                    }
                }
                if target < pc {
                    can_enter_jit!(driver, target, &mut state, program, || {});
                }
                pc = target;
                continue;
            }
            _ => {
                pc += 1;
            }
        }
    }

    state.a.iter().sum::<i64>() + state.b.iter().sum::<i64>()
}

pub struct JitDualInterp {
    threshold: u32,
}

impl JitDualInterp {
    pub fn new() -> Self {
        JitDualInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, code: &[u8]) -> i64 {
        mainloop(code, self.threshold)
    }
}

impl Default for JitDualInterp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// `COMPILES` / `LAST_OPS_AFTER` are process-global, so under the default
    /// parallel libtest runner any concurrent `run` lands inside
    /// `jit_tier_liveness_gate`'s store/run/load window and the gate reads
    /// another test's compile. The lock therefore covers *every* test that can
    /// compile, not just the gate. `run_locked` and the gate are the only ways
    /// a test may enter `JitDualInterp::run`; neither may call the other, since
    /// this is a plain mutex and re-entering it on one thread deadlocks.
    static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn run_locked(code: &[u8]) -> i64 {
        let _guard = PROBE_LOCK.lock();
        JitDualInterp::new().run(code)
    }

    fn check(code: &[u8]) {
        let expected = interp::interpret(code);
        let got = run_locked(code);
        assert_eq!(
            got, expected,
            "JIT result {got} != interp {expected} for {code:?}"
        );
    }

    #[test]
    fn jit_matches_interp_dual_loop() {
        // a[0]=10; loop 10x mutating both tapes. Runs hot enough to compile
        // the inner loop, then guard-fails on exit and reconstructs both tapes.
        check(b"++++++++++[->+<*}*{]");
    }

    #[test]
    fn jit_matches_interp_only_tape_b_in_body() {
        // Loop counts on tape a but the body only touches tape b.
        check(b"+++++++[-*}*{]");
    }

    #[test]
    fn jit_matches_interp_wider_elements() {
        // Spread writes across several cells of each tape so more than one
        // element box per array is live across the loop header.
        check(b"++++++++[->+>+<<*}*}*{{]");
    }

    #[test]
    fn jit_matches_interp_no_loop() {
        check(b"+++>+<*}*");
    }

    #[test]
    fn jit_matches_interp_zero_cell_bracket() {
        let mut program = vec![b'+'; 1001];
        program.extend_from_slice(b"[->[*]<]");
        // Absolute pin BESIDE the differential one. `check()` alone compares two
        // readers that would agree if both were wrong; 0 is the answer the
        // program has, derived from the drain: `a` empties and nothing is ever
        // banked into `b`.
        assert_eq!(
            interp::interpret(&program),
            0,
            "fixture changed: the zero-cell scan no longer drains to 0, so the \
             differential below no longer covers the branch this test exists for"
        );
        check(&program);
    }

    /// A program whose answer IS the number of loop passes.
    ///
    /// `('+' * n) + "[-*}*{]"` charges `a[0]` to `n`; each pass spends one unit
    /// of `a[0]` and banks one unit into each of `b[0]` and `b[1]`. `a` drains
    /// to zero, so `mainloop`'s trailing `sum(a) + sum(b)` is `2 * passes` — an
    /// exact count, not a modular one: the tape cells are `i64` and nothing
    /// here folds them.
    ///
    /// The two banked cells are what make the count readable. A one-for-one
    /// body (`[-*]`) answers `n` after `n` passes AND `n` after zero passes,
    /// because the drain and the deposit cancel — the assertion would hold on a
    /// loop that never ran. Depositing twice per pass separates them: `n`
    /// passes answer `2n`, zero passes answer `n`, `n + 1` passes answer
    /// `2n + 1`, `n - 1` passes answer `2n - 1`.
    ///
    /// `check()` above cannot settle this: it compares the JIT against
    /// `interp::interpret`, and both run the same program, so a duplicated
    /// iteration of the *compiled* loop is invisible to it. Only an ABSOLUTE
    /// count catches that.
    ///
    /// Two trip counts of different parity, because a peeled first iteration
    /// plus an even/odd body count is exactly the shape an off-by-one hides in.
    #[test]
    fn trip_count_gate() {
        for n in [1001i64, 1002] {
            let mut program = vec![b'+'; n as usize];
            program.extend_from_slice(b"[-*}*{]");
            let got = run_locked(&program);
            assert_eq!(
                got,
                2 * n,
                "sum after ('+' * {n}) + \"[-*}}*{{]\" is {got}, so the dispatch loop \
                 banked {got} units rather than {} — the pass count is off, which is \
                 what a merge-point exit that re-runs or drops a pass the walk \
                 already executed looks like",
                2 * n
            );
            println!("[trip-count] ('+' * {n}) + \"[-*}}*{{]\" = {got} — exactly {n} passes");
        }
    }

    #[test]
    fn jit_tier_liveness_gate() {
        let _guard = PROBE_LOCK.lock();
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);

        let mut program = vec![b'+'; 1001];
        program.extend_from_slice(b"[-*}*{]");
        let got = JitDualInterp::new().run(&program);
        assert_eq!(got, 2002, "fixture changed; the liveness reading is moot");

        let compiles = COMPILES.load(Ordering::Relaxed);
        let ops_after = LAST_OPS_AFTER.load(Ordering::Relaxed);
        println!("[tier] COMPILES={compiles} LAST_OPS_AFTER={ops_after}");

        assert!(
            compiles > 0,
            "nothing compiled at all, so LAST_OPS_AFTER says nothing either"
        );
    }

    /// WHICH dispatch arms lowered to an abort stub, and WHY.
    ///
    /// Split out of `jit_tier_liveness_gate` when that gate's `compiles > 0`
    /// was suspended as non-discriminating. This is the half that still does
    /// work, and it is the half that fired when the arm set moved. It is also
    /// independent of the suspension's cause: `record_degraded_dispatch_arm`
    /// runs when the dispatch JitCode is INSTALLED, not when a trace walks into
    /// a stub, so these readings survive a tier that compiles nothing at all.
    #[test]
    fn jit_tier_degraded_arm_gate() {
        let _guard = PROBE_LOCK.lock();

        let mut program = vec![b'+'; 1001];
        program.extend_from_slice(b"[-*}*{]");
        let got = JitDualInterp::new().run(&program);
        assert_eq!(got, 2002, "fixture changed; the arm reading is moot");

        // Every opcode body now lowers. In particular, both bracket scans stay
        // in the translated body like RPython `BrainInterpreter.interp_char`;
        // byte literals and loop-carried local updates no longer turn either
        // arm into an abort stub.
        let mut degraded: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "DualState")
            .collect();
        degraded.sort_unstable_by_key(|a| a.arm);
        let degraded_arms: Vec<&str> = degraded.iter().map(|a| a.arm).collect();
        println!("[tier] degraded_arms={degraded_arms:?}");
        assert!(
            degraded_arms.is_empty(),
            "a dualtape opcode body still installs an abort stub: {degraded:?}"
        );
    }

    #[test]
    fn jit_tier_shape_gate() {
        let _guard = PROBE_LOCK.lock();
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);

        let mut program = vec![b'+'; 1001];
        program.extend_from_slice(b"[-*}*{]");
        let got = JitDualInterp::new().run(&program);
        assert_eq!(got, 2002, "fixture changed; the liveness reading is moot");

        let compiles = COMPILES.load(Ordering::Relaxed);
        let ops_after = LAST_OPS_AFTER.load(Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape {
            has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
            has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
        };
        // Both booleans in the failure output, not only the rendered reason.
        // A single human-readable string is a lossy encoding of a compound
        // state, and the loss is always the discrimination — which is the exact
        // defect this gate replaces.
        assert!(
            shape.closes_a_loop(),
            "compiled body does not close a loop: has_jump={} has_always_fails={} \
             ({}). COMPILES={compiles}, ops_after={ops_after}",
            shape.has_jump,
            shape.has_always_fails,
            shape.why_not().unwrap_or("closes a loop"),
        );
    }
}
