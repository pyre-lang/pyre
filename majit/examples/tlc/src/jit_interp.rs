/// JIT-enabled TLC interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tlc.py JitDriver(greens=['pc','code'], reds=['frame','pool']).
/// TLC's Frame has a plain list stack (no `_virtualizable_`); we use
/// `state_fields = { stackpos: int, stack: [int; virt] }` to mirror the
/// virtualizable-stack shape used by tl.py/tla.py for the integer-only trace.
///
/// Greens: [pc, program]
/// Reds:   [stackpos, stack]
///
/// `greens = [pc, program]` is load-bearing, not documentation. With the list
/// left empty, every arm that reads an operand out of `program` — PUSH, PICK,
/// PUT, BR, BR_COND, PUSHARG — failed to lower and was emitted as an abort stub,
/// so every trace aborted and `Traces compiled` stayed 0 for the whole suite:
/// the JIT compiled nothing and the tests were green only because the legacy
/// merge point discards the walk and lets the native loop answer. Declaring the
/// greens leaves PUSHARG as the only degraded arm; `ROLL` joined it there until
/// the `VirtArray` spelling gave that arm a base pointer to lower against. The
/// set is pinned by `degraded_dispatch_arms` in the test below, which is what
/// this sentence must agree with.
///
/// Only integer-stack opcodes are traced. Object opcodes (NIL, CONS, CAR, CDR,
/// NEW, GETATTR, SETATTR, SEND) cause guard failure in RPython and are absent
/// from this function, matching that behavior.
use crate::interp::{self, ConstantPool};
use majit_metainterp::virt_array::VirtArray;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ── State ──

pub type Bytecode = [u8];

/// Hot loops majit compiled. The only positive evidence the JIT tier is alive:
/// a green suite, byte-identical output and an exact absolute trip count are all
/// satisfied by the interpreter answering alone. This example ran 22/22 green at
/// `Traces compiled: 0` for the whole suite before `greens` was declared.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Ops in the last compiled loop body after optimization.
///
/// `COMPILES > 0` is necessary but NOT sufficient: an entirely empty dispatch
/// still compiles a trace — one whose whole optimized body is `Finish()`, i.e.
/// `ops_after == 1`. A compile counter counts TRACES, not WORK. This is the
/// term that separates a compiled loop from a compiled nothing.
pub static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

struct TlcState {
    stackpos: i64,
    stack: VirtArray<i64>,
}

/// Rotates the live stack through a residual call.
///
/// This mirrors the RPython TLC list operations. The helper receives the live
/// virtualizable object, so its may-force effect synchronizes redirected array
/// fields before the residual mutation.
#[majit_macros::jit_may_force]
extern "C" fn tlc_roll(state: &mut TlcState, r: i64) {
    let stack = &mut state.stack[..state.stackpos as usize];
    let len = stack.len();
    if r < -1 {
        // Move top element to position len+r (counted from bottom).
        let i = len as i64 + r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[len - 1];
        for j in (i..len - 1).rev() {
            stack[j + 1] = stack[j];
        }
        stack[i] = elem;
    } else if r > 1 {
        // Move element at position len-r to top.
        let i = len as i64 - r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[i];
        for j in i..len - 1 {
            stack[j] = stack[j + 1];
        }
        stack[len - 1] = elem;
    }
}

// ── Opcodes ──

const NOP: u8 = interp::NOP;
const PUSH: u8 = interp::PUSH;
const POP: u8 = interp::POP;
const SWAP: u8 = interp::SWAP;
const ROLL: u8 = interp::ROLL;
const PICK: u8 = interp::PICK;
const PUT: u8 = interp::PUT;
const ADD: u8 = interp::ADD;
const SUB: u8 = interp::SUB;
const MUL: u8 = interp::MUL;
// DIV not traced: IntObj.div() in tlc.py uses Python 2 floor division (//),
// which differs from Rust's truncating division for negative operands.
const EQ: u8 = interp::EQ;
const NE: u8 = interp::NE;
const LT: u8 = interp::LT;
const LE: u8 = interp::LE;
const GT: u8 = interp::GT;
const GE: u8 = interp::GE;
const BR: u8 = interp::BR;
const BR_COND: u8 = interp::BR_COND;
const RETURN: u8 = interp::RETURN;
const PUSHARG: u8 = interp::PUSHARG;

const DEFAULT_THRESHOLD: u32 = 3;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlcState,
    env = Bytecode,
    greens = [pc, program],
    auto_calls = true,
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlcState> =
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
    let stacksize: i32 = 0;
    // The size is `tl.py interp`'s: it builds its stack as `Stack(len(code))`,
    // and `Stack.__init__` fills `[0] * size` once, because the same
    // `_virtualizable_ = ['stackpos', 'stack[*]']` this state field ports needs
    // a length the entry can rebuild from. tlc.py's own `self.stack = []`
    // grows, but tlc.py's stack is not the virtualizable one.
    //
    // A fixed size is a bound, not a proof: a back-edge over a net-pushing
    // region can outrun `program.len()`, and the write then panics on the
    // slice rather than growing. Upstream carries the same exposure for the
    // same reason, and this example runs fixture bytecode, so the size is left
    // as upstream spells it.
    let mut state = TlcState {
        stackpos: 0,
        stack: VirtArray::filled(0i64, program.len()),
    };

    // RPython warmspot.py:281-289 — `make_jitcodes(); finish_setup(codewriter)`
    // surface for state-field JIT.  Publishes the canonical
    // `(live_i, live_r, live_f)` triple into `staticdata.liveness_info`
    // before the first `jit_merge_point!()` so that
    // `MIFrame::get_list_of_active_boxes` can decode the
    // macro-emitted `live/<offset>` placeholders.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        // The `; state` single-pass form. Reaching it took two fixes, because
        // this file's `PUSHARG` arm is an abort stub (see `jit_tier_is_alive`)
        // and its post-loop expression stores:
        //
        // 1. A degraded-stub abort resumed at `opcode_pc + 1` — the shared
        //    prologue advance below, not the instruction width — so the
        //    aborting opcode applied nothing and was then skipped. Fixed by
        //    resuming at the opcode's own boundary; the regression test is
        //    `jit_interp_degraded_stub_abort_resume.rs`.
        // 2. The walk ran this function's trailing expression, whose `stackpos`
        //    store the write-back then pushed into native `state` for the
        //    post-loop code to apply a second time — `fibo(7)` answered 8, the
        //    entry one below the right one, instead of 13. Fixed by keeping a
        //    *storing* trailing expression out of the walk; the regression test
        //    is `jit_interp_halt_arm_post_loop_expression.rs`.
        //
        // The `ROLL` arm lowers since the `VirtArray` spelling gave it a base
        // pointer for its `[int; virt]` state-field array; the degraded-arm
        // pin below is what states the remaining set.
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos += 1;
            }
            POP => {
                state.stackpos -= 1;
            }
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos -= 1;
            }
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos -= 1;
            }
            MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos -= 1;
            }
            EQ => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b == a { 1 } else { 0 };
                state.stackpos -= 1;
            }
            NE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b != a { 1 } else { 0 };
                state.stackpos -= 1;
            }
            LT => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b < a { 1 } else { 0 };
                state.stackpos -= 1;
            }
            LE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b <= a { 1 } else { 0 };
                state.stackpos -= 1;
            }
            GT => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b > a { 1 } else { 0 };
                state.stackpos -= 1;
            }
            GE => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = if b >= a { 1 } else { 0 };
                state.stackpos -= 1;
            }
            ROLL => {
                let r = program[pc] as i8 as i64;
                tlc_roll(&mut state, r);
                pc += 1;
            }
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos += 1;
            }
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.stackpos -= 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[(state.stackpos as usize) - i] = v;
            }
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos += 1;
            }
            BR_COND => {
                let target = ((pc as i64) + program[pc] as i8 as i64 + 1) as usize;
                let next_pc = pc + 1;
                state.stackpos -= 1;
                let jump = state.stack[state.stackpos as usize] != 0;
                if jump {
                    if target < next_pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
                pc = next_pc;
            }
            BR => {
                let target = ((pc as i64) + program[pc] as i8 as i64 + 1) as usize;
                let next_pc = pc + 1;
                if target < next_pc {
                    can_enter_jit!(driver, target, &mut state, program, || {});
                }
                pc = target;
                continue;
            }
            // Bare `break` bodies, never `{ …; break }`: `classify.rs`
            // `is_break_expr` requires the arm body to be exactly `break`, so a
            // composite body classifies `Lowerable` and its tail `break` reaches
            // `lower_stmt_fallback`, which guards an enclosed `return` but not an
            // enclosed `break` — the statement is inert and is silently dropped,
            // leaving the lowered arm to fall through to the dispatch back-edge.
            RETURN => break,
            _ => break,
        }
    }

    // Reads `state` and never `pc`, which is what the single-executor merge
    // point will need when it lands: its `break` precedes the `pc` handoff, so
    // `pc` there still names the position the walk started from. `stackpos` is a
    // scalar state field and `stack` a virtualizable array field, so the
    // walk-final values arrive through `writeback_scalar_state_fields` /
    // `writeback_virt_array_state_fields` — no `ret` field is needed here.
    if state.stackpos == 0 {
        0
    } else {
        state.stackpos -= 1;
        state.stack[state.stackpos as usize]
    }
}

// ── Public wrapper matching the old API ──

pub struct JitTlcInterp {
    threshold: u32,
}

impl Default for JitTlcInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitTlcInterp {
    pub fn new() -> Self {
        JitTlcInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Run the TLC interpreter with JIT support.
    /// Only traces integer-only loops; unknown opcodes cause loop exit.
    pub fn run(&mut self, code: &[u8], inputarg: i64, _pool: &ConstantPool) -> i64 {
        mainloop(code, inputarg, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;
    use majit_metainterp::{RefusalKind, refusal_kind};

    /// Fibonacci using ROLL -- pure integer loop, good JIT candidate.
    fn fibo_bytecode(pool: &mut ConstantPool) -> Vec<u8> {
        interp::compile(
            include_str!("../../../../rpython/jit/tl/fibo.tlc.src"),
            pool,
        )
    }

    /// [`COMPILES`] is process-global, so under the default parallel libtest
    /// runner a concurrent run lands inside [`compile_probe`]'s store/run/load
    /// window and the probe reads someone else's compile. The lock therefore
    /// covers *every* call that can compile — [`run_jit`] and [`compile_probe`]
    /// are the only two ways a test may enter the JIT, and neither may call the
    /// other (a plain mutex re-entered on one thread deadlocks).
    static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// For tests that assert only on the result. They still compile, so they
    /// must not run inside the probe's window. See [`PROBE_LOCK`].
    fn run_jit(bc: &[u8], arg: i64, pool: &ConstantPool) -> i64 {
        let _guard = PROBE_LOCK.lock();
        let mut jit = JitTlcInterp::new();
        jit.run(bc, arg, pool)
    }

    /// Run with both counters reset, returning `(result, compiles, ops_after)`.
    ///
    /// [`LAST_OPS_AFTER`] is read here rather than at the call site, and reset
    /// here rather than nowhere. Both counters are process-global, so both need
    /// the same treatment [`PROBE_LOCK`] exists to give [`COMPILES`]: a load
    /// taken after the guard drops can observe a concurrent test's compile, and
    /// a counter that is never stored to zero retains whatever the last compile
    /// anywhere in the process left behind. Unreset, a zero from this probe is
    /// indistinguishable from an inherited value.
    fn compile_probe(
        bc: &[u8],
        arg: i64,
        pool: &ConstantPool,
    ) -> (i64, usize, usize, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock();
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let mut jit = JitTlcInterp::new();
        let got = jit.run(bc, arg, pool);
        (
            got,
            COMPILES.load(Ordering::Relaxed),
            LAST_OPS_AFTER.load(Ordering::Relaxed),
            majit_metainterp::LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        )
    }

    #[test]
    fn jit_fibo_7() {
        let mut pool = ConstantPool::new();
        let bc = fibo_bytecode(&mut pool);
        assert_eq!(run_jit(&bc, 7, &pool), 13);
    }

    #[test]
    fn jit_tier_is_alive() {
        let mut pool = ConstantPool::new();
        let bc = countdown_bytecode(&mut pool);
        let (got, compiles, ops_after, shape) = compile_probe(&bc, 100, &pool);
        // The body actually closes a loop — see `LoopBodyShape`. A compile
        // count and an op count together still accept a body that bails out on
        // its first pass; this is the term that does not. Sound HERE because
        // this fixture loops: on a straight-line subject a `Jump`-less body is
        // the right answer, not a defect.
        assert!(
            shape.closes_a_loop(),
            "compiled {ops_after} ops but the body {} ({shape:?})",
            shape.why_not().unwrap_or("closes a loop")
        );
        assert_eq!(got, 0, "countdown(100) must still answer 0");

        let mut tlc_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "TlcState")
            .collect();
        tlc_arms.sort_unstable_by_key(|a| a.arm);
        let degraded: Vec<&str> = tlc_arms.iter().map(|a| a.arm).collect();
        assert_eq!(
            degraded,
            ["PUSHARG"],
            "the degraded-arm set moved. A NEW name means an arm silently \
             stopped lowering and every trace reaching it now aborts; a MISSING \
             name means that arm lowers again, so the abort that blocks the \
             `; state` conversion may be gone — re-check the merge point"
        );

        let causes: Vec<(&str, RefusalKind)> = tlc_arms
            .iter()
            .map(|a| (a.arm, refusal_kind(a.reason)))
            .collect();
        for a in &tlc_arms {
            eprintln!(
                "[cause] {} {:?} — {}",
                a.arm,
                refusal_kind(a.reason),
                a.reason
            );
        }
        assert_eq!(
            causes,
            [("PUSHARG", RefusalKind::UnlowerableStmt)],
            "an arm still degrades but a different mechanism is refusing it. \
             `RefusalKind::Unclassified` on either side means majit grew a \
             refusal family the classifier does not know — add it in \
             `majit-metainterp`, do not re-record this pin"
        );

        assert!(
            tlc_arms[0].reason.contains("inputarg"),
            "PUSHARG's refusal no longer names the loop-external input it \
             stores: {}",
            tlc_arms[0].reason
        );

        // Zero-vs-nonzero is the property; a later change that legitimately
        // mints more than one artifact is not this regression.
        assert!(
            compiles >= 1,
            "countdown(100) compiled {compiles} loops — the JIT tier is inert and \
             the interpreter is answering alone, which every other assertion in \
             this file would still pass"
        );
        assert_eq!(
            ops_after, 10,
            "compiled loop body is {ops_after} ops, not the pinned 10 — a value \
             of 1 means the body is a bare `Finish()`, i.e. a dispatch that \
             lowered nothing at all"
        );
        println!(
            "[tier-alive] countdown(100) = {got}, compiled {compiles} loop(s) of \
             {ops_after} ops, degraded = {degraded:?}"
        );
    }

    #[test]
    fn jit_fibo_matches_interp() {
        let mut pool = ConstantPool::new();
        let bc = fibo_bytecode(&mut pool);
        for n in [1, 2, 3, 5, 7, 10, 15] {
            let expected = interp::interp(&bc, 0, n, &pool);
            let got = run_jit(&bc, n, &pool);
            assert_eq!(got, expected, "fibo mismatch for n={n}");
        }
    }

    /// Simple integer countdown loop (no object ops).
    fn countdown_bytecode(pool: &mut ConstantPool) -> Vec<u8> {
        interp::compile(
            "
            PUSHARG         # [n]
        loop:
            PUSH 1
            SUB             # [n-1]
            PICK 0          # [n-1, n-1]
            BR_COND loop    # [n-1] if n-1 != 0
            RETURN
        ",
            pool,
        )
    }

    #[test]
    fn jit_countdown() {
        let mut pool = ConstantPool::new();
        let bc = countdown_bytecode(&mut pool);
        assert_eq!(run_jit(&bc, 100, &pool), 0);
    }

    #[test]
    fn jit_operand_less_degraded_arm_runs_every_pass() {
        for n in [100i64, 101] {
            let mut pool = ConstantPool::new();
            let bc = interp::compile(
                &format!(
                    "
                PUSH 0          # [acc]
            loop:
                PUSHARG         # [acc, 1]
                ADD             # [acc+1]
                PICK 0          # [acc, acc]
                PUSH {n}        # [acc, acc, n]
                LT              # [acc, acc < n]
                BR_COND loop    # [acc]
                RETURN
            "
                ),
                &mut pool,
            );
            let got = run_jit(&bc, 1, &pool);
            assert_eq!(
                got, n,
                "count-by-PUSHARG returned {got}, not {n} — a degraded arm's \
                 abort dropped the opcode, so a pass ran without its increment"
            );
        }
    }

    /// Absolute trip-count gate on the JIT path.
    ///
    /// The accumulator gains 1 once per pass and the loop runs while it is still
    /// below `n`, so the returned value names the number of passes exactly: `n`
    /// passes answer `n`, and one extra pass answers `n + 1`. `LT` (rather than
    /// `NE`) is what makes the overshoot observable instead of divergent.
    ///
    /// Agreement with `interp::interp` would not settle this — the two run the
    /// same program — and neither would a byte-identical before/after output
    /// comparison: a duplicated iteration of the *compiled* loop is invisible to
    /// any check that does not assert an absolute count.
    ///
    /// Two lengths of different parity, because a peeled first iteration plus an
    /// even/odd body count is exactly the shape an off-by-one hides in.
    #[test]
    fn jit_trip_count_gate() {
        let mut pool = ConstantPool::new();
        let bc = interp::compile(
            "
            PUSHARG         # [n]
            PUSH 0          # [n, acc]
        loop:
            PUSH 1          # [n, acc, 1]
            ADD             # [n, acc+1]
            PICK 0          # [n, acc, acc]
            PICK 2          # [n, acc, acc, n]
            LT              # [n, acc, acc < n]
            BR_COND loop    # [n, acc]
            RETURN
        ",
            &mut pool,
        );
        for n in [1001i64, 1002] {
            let got = run_jit(&bc, n, &pool);
            assert_eq!(
                got, n,
                "count_to({n}) = {got}, so the loop ran {got} passes rather than \
                 {n} — an off-by-one trip count is the signature of a terminal \
                 arm whose exit the trace dropped"
            );
            println!("[trip-count] count_to({n}) = {got} — exactly {n} passes");
        }

        // Non-vacuity: the same loop with the accumulator seeded at `n` instead
        // of 0 runs one pass, overshoots to `n + 1`, and `LT` lets it out —
        // exactly what a duplicated compiled iteration would do at the end. So
        // the assertion above is one the program can fail, not one pinned to `n`
        // by its own exit condition.
        let overshoot = interp::compile(
            "
            PUSHARG         # [n]
            PUSHARG         # [n, acc=n]
        loop:
            PUSH 1
            ADD             # [n, acc+1]
            PICK 0
            PICK 2
            LT              # [n, acc, acc < n]
            BR_COND loop
            RETURN
        ",
            &mut pool,
        );
        let got = run_jit(&overshoot, 1001, &pool);
        assert_eq!(
            got, 1002,
            "the loop cannot leave above `n`, so the gate above cannot \
             distinguish 1001 passes from 1002"
        );
    }

    #[test]
    fn jit_sum() {
        let mut pool = ConstantPool::new();
        let bc = interp::compile(
            "
            PUSH 0          # [acc=0]
            PUSHARG         # [acc, n]
        loop:
            PICK 0          # [acc, n, n]
            BR_COND body
            POP
            RETURN
        body:
            SWAP            # [n, acc]
            PICK 1          # [n, acc, n]
            ADD             # [n, acc+n]
            SWAP            # [acc+n, n]
            PUSH 1
            SUB             # [acc, n-1]
            PUSH 1
            BR_COND loop
        ",
            &mut pool,
        );
        assert_eq!(run_jit(&bc, 10, &pool), 55);
        assert_eq!(run_jit(&bc, 100, &pool), 5050);
    }
}
