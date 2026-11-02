/// JIT-enabled TL interpreter via `#[jit_interp]` with `state_fields`.
///
/// RPython parity: tl.py JitDriver(greens=['pc','code'], reds=['inputarg','stack'],
/// virtualizables=['stack']). Stack._virtualizable_ = ['stackpos', 'stack[*]']
/// at tl.py maps directly to `state_fields = { stackpos: int, stack: [int; virt] }`.
///
/// Greens: [pc, code]
/// Reds:   [inputarg, stackpos, stack]  (inputarg is a function parameter — red by nature)
use majit_metainterp::jit::promote;
use majit_metainterp::virt_array::VirtArray;

/// Hot loops majit compiled. The only positive evidence the JIT tier is alive:
/// a green suite, agreement with `interp::interpret` and an exact absolute trip
/// count are all satisfied by an interpreter answering alone.
///
/// Also the control/probe counter for `spike_portal_call_compile_through`.
pub static COMPILES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Optimized operation count for the most recently compiled loop. A compile
/// count alone cannot distinguish a real loop body from an empty `Finish`.
pub static LAST_OPS_AFTER: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Stack rotation — @dont_look_inside in RPython (tl.py).
///
/// Operates on the live portion of the stack `stack[0..stackpos]`.
/// The JIT does not trace into this function; it emits a residual CALL.
///
/// The residual is a MAY-FORCE call, not a plain can-raise one.
/// `call.py getcalldescr` consults `virtualizable_analyzer` BEFORE
/// `_canraise`, and `roll` writes `self.stack[...]`, a field `tl.py`
/// declares in `Stack._virtualizable_ = ['stackpos', 'stack[*]']` — so the
/// analyzer picks `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` there. `@dont_look_inside`
/// only clears `_jit_look_inside_` (`rlib/jit.py`); it does not change the
/// effect row. pyre runs no analyzer over this helper, so the row is declared:
/// `#[dont_look_inside]` would assert `EF_CAN_RAISE` with an empty write set
/// this raw-pointer mutation of the virtualizable array has not earned.
#[cfg(test)]
pub static ROLL_CALLS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Rotates the live stack through a residual call.
///
/// As in `tl.py::Stack.roll`, the helper receives the virtualizable object,
/// not a raw pointer to one redirected field. The may-force call therefore
/// synchronizes the virtualizable before the helper mutates its live array.
#[majit_macros::jit_may_force]
extern "C" fn storage_roll(state: &mut TlState, r: i64) {
    #[cfg(test)]
    ROLL_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let stack = &mut state.stack[..state.stackpos as usize];
    let len = stack.len();
    if r < -1 {
        // tl.py:45-55
        let i = len as i64 + r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let n = len - 1;
        let elem = stack[n];
        for j in (i..n).rev() {
            stack[j + 1] = stack[j];
        }
        stack[i] = elem;
    } else if r > 1 {
        // tl.py:56-65
        let i = len as i64 - r;
        assert!(i >= 0, "IndexError in ROLL");
        let i = i as usize;
        let elem = stack[i];
        for j in i..len - 1 {
            stack[j] = stack[j + 1];
        }
        let n = len - 1;
        stack[n] = elem;
    }
}

// ── State ──

pub type Bytecode = [u8];

/// tl.py Stack object. `_virtualizable_ = ['stackpos', 'stack[*]']`.
/// tl.py `Stack(size)` — `size` is the bytecode length; the caller
/// (`interp_eval`) passes `len(code)`. See tl.py.
struct TlState {
    stackpos: i64,
    stack: VirtArray<i64>,
}

// ── Opcodes ──

const NOP: u8 = 1;
const PUSH: u8 = 2;
const POP: u8 = 3;
const SWAP: u8 = 4;
const ROLL: u8 = 5;
const PICK: u8 = 6;
const PUT: u8 = 7;
const ADD: u8 = 8;
const SUB: u8 = 9;
const MUL: u8 = 10;
const DIV: u8 = 11;
const EQ: u8 = 12;
const NE: u8 = 13;
const LT: u8 = 14;
const LE: u8 = 15;
const GT: u8 = 16;
const GE: u8 = 17;
const BR_COND: u8 = 18;
const BR_COND_STK: u8 = 19;
const CALL: u8 = 20;
const RETURN: u8 = 21;
const PUSHARG: u8 = 22;

// ── JIT mainloop ──

#[majit_macros::jit_interp(
    state = TlState,
    env = Bytecode,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
    recursive_entry = crate::interp::interpret_recursive,
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TlState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    // Count compiled loops, and record the size of the last compiled body.
    driver.set_on_compile_loop(|_green_key, _ops_before, ops_after, opcodes| {
        COMPILES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, core::sync::atomic::Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        LAST_HAS_JUMP.store(shape.has_jump, core::sync::atomic::Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(
            shape.has_always_fails,
            core::sync::atomic::Ordering::Relaxed,
        );
    });
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = TlState {
        stackpos: 0,
        stack: VirtArray::filled(0i64, program.len()),
    };

    // RPython warmspot.py:281-289 canonical-liveness install hook.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        // tl.py:88  stack.stackpos = promote(stack.stackpos)
        state.stackpos = promote(state.stackpos);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            NOP => {}
            // tl.py:94-96
            PUSH => {
                let value = program[pc] as i8 as i64;
                pc += 1;
                state.stack[state.stackpos as usize] = value;
                state.stackpos += 1;
            }
            // tl.py:98-99
            POP => {
                state.stackpos -= 1;
            }
            // tl.py:101-104
            SWAP => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 1) as usize] = b;
                state.stack[(state.stackpos - 2) as usize] = a;
            }
            // tl.py  Stack.roll() is @dont_look_inside
            ROLL => {
                let r = program[pc] as i8 as i64;
                storage_roll(&mut state, r);
                pc += 1;
            }
            // tl.py  Stack.pick(i): duplicate stack[stackpos - i - 1]
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos += 1;
            }
            // tl.py  Stack.put(i): pop and store at stackpos - i - 1
            PUT => {
                let i = program[pc] as usize;
                pc += 1;
                state.stackpos -= 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[(state.stackpos as usize) - i] = v;
            }
            // tl.py:119-121
            ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos -= 1;
            }
            // tl.py:123-125
            SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos -= 1;
            }
            // tl.py:127-129
            MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos -= 1;
            }
            // tl.py:131-133
            DIV => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b / a;
                state.stackpos -= 1;
            }
            // tl.py:135-157 — inline comparisons (no helper functions)
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
            // tl.py:159-165
            BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                state.stackpos -= 1;
                let jump = state.stack[state.stackpos as usize] != 0;
                if jump {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py:167-172
            BR_COND_STK => {
                state.stackpos -= 1;
                let offset = state.stack[state.stackpos as usize];
                state.stackpos -= 1;
                let cond = state.stack[state.stackpos as usize];
                if cond != 0 {
                    let target = (pc as i64 + offset) as usize;
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            // tl.py — `res = interp(code, pc + offset)`, a recursive
            // portal re-entry.  Greens in declaration order [pc, program];
            // the concrete fallback is `interpret_recursive` (declared via
            // `recursive_entry`), the JIT path emits BC_RECURSIVE_CALL_INT.
            CALL => {
                let offset = program[pc] as i8 as i64;
                pc += 1;
                let target = (pc as i64 + offset) as usize;
                let res = recursive_portal_call!(driver, target, program);
                state.stack[state.stackpos as usize] = res;
                state.stackpos += 1;
            }
            // tl.py:180-181
            //
            // A bare `break` body, never `{ …; break }`: `classify.rs`
            // `is_break_expr` requires the arm body to be exactly `break`, so a
            // composite body classifies `Lowerable` and its tail `break` reaches
            // `lower_stmt_fallback`, which guards an enclosed `return` but not an
            // enclosed `break` — the statement is judged inert and silently
            // dropped, leaving the lowered arm to fall through to the dispatch
            // back-edge and run one extra iteration.
            RETURN => break,
            // tl.py:183-184
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos += 1;
            }
            _ => {}
        }
    }

    // Reads `state` and never `pc`, which is what the single-executor merge
    // point will need when it lands: its `break` precedes the `pc = __sp_pc`
    // handoff, so `pc` there still names the position the walk started from.
    // `stackpos` is a scalar state field and `stack` a virtualizable array
    // field, so the walk-final values arrive through
    // `writeback_scalar_state_fields` / `writeback_virt_array_state_fields` —
    // no `ret` field is needed here.
    state.stackpos -= 1;
    state.stack[state.stackpos as usize]
}

// ── Public wrapper matching the old API ──

pub struct JitTlInterp {
    threshold: u32,
}

impl Default for JitTlInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitTlInterp {
    pub fn new() -> Self {
        JitTlInterp { threshold: 3 }
    }

    pub fn run(&mut self, bytecode: &[u8], inputarg: i64) -> i64 {
        mainloop(bytecode, inputarg, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;
    use core::sync::atomic::Ordering;

    /// sum(N) = 1 + 2 + ... + N
    fn sum_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0,       // acc = 0
            PUSHARG, // counter = N
            // loop (offset 3):
            PICK, 0, // dup counter
            BR_COND, 2,      // if counter != 0, skip to body (offset 9)
            POP,    // pop counter
            RETURN, // body (offset 9):
            SWAP,   // [counter, acc]
            PICK, 1,    // [counter, acc, counter]
            ADD,  // [counter, acc+counter]
            SWAP, // [acc+counter, counter]
            PUSH, 1, SUB, // [acc, counter-1]
            PUSH, 1, BR_COND, 238, // -18: jump to loop (offset 3)
        ]
    }

    /// [`COMPILES`] is process-global, so under the default parallel libtest
    /// runner a concurrent `run` lands inside [`compile_probe`]'s
    /// store/run/load window and the probe reads someone else's compile. The
    /// lock therefore covers *every* call that can compile, not just the
    /// probe's own — [`run_jit`] and [`compile_probe`] are the only two ways a
    /// test may enter the JIT, and neither may call the other (a plain mutex
    /// re-entered on one thread deadlocks).
    static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// For tests that assert only on the result. They still compile, so they
    /// must not run inside the probe's window. See [`PROBE_LOCK`].
    fn run_jit(bc: &[u8], inputarg: i64) -> i64 {
        let _guard = PROBE_LOCK.lock();
        JitTlInterp::new().run(bc, inputarg)
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
        inputarg: i64,
    ) -> (i64, usize, usize, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock();
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let got = JitTlInterp::new().run(bc, inputarg);
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

    /// Run with [`ROLL_CALLS`] reset, returning `(result, roll_calls)`.
    ///
    /// [`ROLL_CALLS`] is process-global for the same reason [`COMPILES`] is, and
    /// any concurrently running test whose program issues `ROLL` adds to it. It
    /// therefore needs [`PROBE_LOCK`] over its own store/run/load window too.
    fn run_jit_counting_rolls(bc: &[u8], inputarg: i64) -> (i64, u32) {
        let _guard = PROBE_LOCK.lock();
        ROLL_CALLS.store(0, Ordering::Relaxed);
        let got = JitTlInterp::new().run(bc, inputarg);
        (got, ROLL_CALLS.load(Ordering::Relaxed))
    }

    use majit_metainterp::{RefusalKind, refusal_kind};

    #[test]
    fn jit_tier_is_alive() {
        // 500 * 501 / 2. `sum` is a weaker trip-count oracle than
        // `trip_count_bytecode` (its terminal pass adds a counter of 0, so a
        // duplicated final iteration leaves the sum unchanged), but it is a
        // loop the tier actually compiles, and tier liveness is what this test
        // is for.
        let (got, compiles, ops_after, shape) = compile_probe(&sum_bytecode(), 500);
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
        assert_eq!(got, 125_250, "sum(500) must still answer 125250");

        let tl_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "TlState")
            .collect();

        let mut degraded: Vec<&str> = tl_arms.iter().map(|a| a.arm).collect();
        degraded.sort_unstable();
        assert_eq!(
            degraded,
            ["PUSHARG"],
            "the degraded-arm set moved. A NEW name means an arm silently \
             stopped lowering and every trace reaching it now aborts; a MISSING \
             name means that arm lowers again, so a loop that could not trace \
             before may now — re-check which subjects compile"
        );

        let mut causes: Vec<(&str, RefusalKind)> = tl_arms
            .iter()
            .map(|a| (a.arm, refusal_kind(a.reason)))
            .collect();
        causes.sort_unstable();
        assert_eq!(
            causes,
            [("PUSHARG", RefusalKind::UnlowerableStmt)],
            "a degraded arm's CAUSE moved while its name did not. That is the \
             signal the set above structurally cannot carry: the arm still \
             degrades, so part 3 is unchanged, but a different mechanism is \
             refusing it now — which is what a routing or lowering change looks \
             like when it half-works"
        );

        // Same mechanism, different offending statement, is a third way to move
        // while parts 3 and 4 both hold. Substrings, not whole reasons: the
        // macro's stringification spacing (`state.stack [ ... ]`) is an artifact
        // of token rendering and is not a property worth pinning.
        let reason_of = |arm: &str| -> &'static str {
            tl_arms
                .iter()
                .find(|a| a.arm == arm)
                .expect("part 3 already pinned this arm as present")
                .reason
        };
        assert!(
            reason_of("PUSHARG").contains("state.stack"),
            "PUSHARG's refusal no longer names the `state.stack` write as the \
             unlowerable statement: {}",
            reason_of("PUSHARG")
        );

        // Zero-vs-nonzero is the property; a later change that legitimately
        // mints more than one artifact is not this regression.
        assert!(
            compiles >= 1,
            "sum(500) compiled {compiles} loops — the JIT tier is inert and the \
             interpreter is answering alone, which every other assertion in \
             this file would still pass"
        );
        assert_eq!(
            ops_after, 12,
            "compiled loop body is {ops_after} ops, not the pinned 12 — a value \
             of 1 means the body is a bare `Finish()`, i.e. a dispatch that \
             lowered nothing at all"
        );
        println!(
            "[tier-alive] sum(500) = {got}, compiled {compiles} loop(s) of {ops_after} ops, degraded {degraded:?}"
        );
    }

    #[test]
    fn jit_sum() {
        let bc = sum_bytecode();
        let cases = [("5", 5, 15), ("100", 100, 5050)];
        for (name, n, want) in cases {
            assert_eq!(run_jit(&bc, n), want, "case {name}");
        }
    }

    #[test]
    fn jit_matches_interp() {
        let bc = sum_bytecode();
        for a in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    /// Hot loop that runs a residual `storage_roll` (ROLL, @dont_look_inside)
    /// twice per iteration. An accumulator stays on the stack throughout so
    /// RETURN always has a value; the two ROLLs cancel (rotate + rotate back)
    /// so the final acc equals the interpreter's. Counting actual
    /// `storage_roll` invocations detects walk-vs-native double-execution of
    /// the residual during tracing.
    ///
    /// Stack discipline (acc kept at bottom, counter counts down from N):
    ///   PUSH 0             [0]              acc
    ///   PUSHARG            [0, N]           counter
    /// loop @ off 3:
    ///   PICK 0             [acc, c, c]      dup counter
    ///   BR_COND 2          [acc, c]         if c!=0 -> body(off 9), pops dup
    ///   POP                [acc]            c==0: drop counter
    ///   RETURN                              return acc
    /// body @ off 9:                         [acc, c]
    ///   ROLL 2             [c, acc]         residual (rotate 2)
    ///   ROLL 254(=-2)      [acc, c]         residual (rotate back)
    ///   PUSH 1 SUB         [acc, c-1]       decrement counter
    ///   PUSH 1 BR_COND -N  jump to loop
    fn roll_loop_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0,       // 0: [0]
            PUSHARG, // 2: [0, N]
            // loop @ off 3:
            PICK, 0, // 3: [acc, c, c]
            BR_COND, 2,      // 5: if c!=0 -> body off 9 (pops dup); else fall through
            POP,    // 7: [acc]
            RETURN, // 8: return acc
            // body @ off 9:
            ROLL, 2, // 9: [c, acc] residual
            ROLL, 254, // 11: r=-2 -> [acc, c] residual
            PUSH, 1, SUB, // 13: [acc, c-1]
            PUSH, 1, // 16: [acc, c-1, 1]
            BR_COND, 239, // 18: offset byte @19, target=19+(-17)+1=3 -> loop header
        ]
    }

    #[test]
    fn jit_residual_not_double_executed() {
        let bc = roll_loop_bytecode();
        // Two trip counts of different parity: an off-by-one that only shows on
        // one parity cannot hide behind the other.
        for n in [20i64, 21] {
            // First confirm the program is well-formed and the JIT result matches
            // the interpreter (the two ROLLs cancel, so acc == 0).
            let expected = interp::interpret(&bc, n);
            let (got, jit_rolls) = run_jit_counting_rolls(&bc, n);
            assert_eq!(got, expected, "JIT result diverged from interp at n={n}");

            // Two ROLLs per iteration; N iterations before the counter hits 0.
            let expected_rolls = (n as u32) * 2;
            assert_eq!(
                jit_rolls, expected_rolls,
                "residual storage_roll executed {jit_rolls}× at n={n} but the \
                 program has exactly {expected_rolls} ROLLs — a walk-vs-native \
                 double-execution would inflate this count"
            );
        }
    }

    /// The first residual ROLL's result is consumed before the inverse ROLL.
    /// This prevents a pair that merely cancels in the heap from hiding a
    /// missing post-call virtualizable reload in the trace.
    #[test]
    fn jit_residual_result_is_visible_to_following_ops() {
        let bc = vec![
            PUSH, 1, PUSHARG, PICK, 0, BR_COND, 2, POP, RETURN, ROLL, 2, PICK, 0, ADD, ROLL, 254,
            PUSH, 1, SUB, PUSH, 1, BR_COND, 236,
        ];
        let n = 8;
        let expected = interp::interpret(&bc, n);
        assert_eq!(expected, 1 << n, "fixture computes 2**n");
        assert_eq!(run_jit(&bc, n), expected);
    }

    fn trip_count_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0, // 0: [0]
            // loop @ 2:
            PUSH, 1,   // 2: [acc, 1]
            ADD, // 4: [acc+1]
            PICK, 0,       // 5: [acc, acc]
            PUSHARG, // 7: [acc, acc, n]
            LT,      // 8: [acc, acc<n]
            BR_COND, 247,    // 9: offset byte @10 -> target 2, a back edge
            RETURN, // 11: [acc]
        ]
    }

    /// The same loop seeded at `n` instead of 0, so it leaves one pass above
    /// `n` and answers `n + 1`. Without this the gate below would be asserting
    /// on a value no reachable program can overshoot, and could not fail.
    fn trip_count_overshoot_bytecode() -> Vec<u8> {
        vec![
            PUSHARG, // 0: [acc=n]
            // loop @ 1:
            PUSH, 1,   // 1: [acc, 1]
            ADD, // 3: [acc+1]
            PICK, 0,       // 4: [acc, acc]
            PUSHARG, // 6: [acc, acc, n]
            LT,      // 7: [acc, acc<n]
            BR_COND, 247,    // 8: offset byte @9 -> target 1, a back edge
            RETURN, // 10: [acc]
        ]
    }

    #[test]
    fn jit_trip_count_gate() {
        let bc = trip_count_bytecode();
        for n in [1001i64, 1002] {
            let got = run_jit(&bc, n);
            assert_eq!(
                got, n,
                "the accumulator gains 1 per pass, so the loop ran {got} passes \
                 rather than {n}"
            );
            assert_eq!(
                interp::interpret(&bc, n),
                n,
                "interpreter disagrees with the expected pass count at n={n}"
            );
        }

        // Non-vacuity, in the same test: the overshoot the gate asserts against
        // is representable by this program shape and is actually produced.
        let over = trip_count_overshoot_bytecode();
        for n in [1001i64, 1002] {
            assert_eq!(
                run_jit(&over, n),
                n + 1,
                "the seeded variant must answer n+1, otherwise the gate above \
                 asserts on a value nothing can move"
            );
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![PUSH, 42, RETURN];
        assert_eq!(run_jit(&prog, 0), 42);
    }

    /// A hot loop whose body issues a recursive `CALL` to a constant-returning
    /// subroutine — exercises `recursive_portal_call!` → BC_RECURSIVE_CALL_INT
    /// end to end.  The loop runs `N` times, each iteration adds the
    /// subroutine's result (3) to the accumulator, so `interpret(prog, N) ==
    /// 3 * N`.  The JIT must match the interpreter on every input.
    fn call_loop_bytecode() -> Vec<u8> {
        vec![
            PUSHARG, // counter = N                       [counter]
            PUSH, 0, // acc = 0                           [counter, acc]
            // loop (offset 3):
            SWAP, // [acc, counter]
            PICK, 0, // dup counter   [acc, counter, counter]
            BR_COND, 2, // pop top; if counter != 0 → body (offset 10)
            // exit (counter == 0):  [acc, counter]
            POP,    // [acc]
            RETURN, // return acc
            // body (offset 10):     [acc, counter]
            SWAP, // [counter, acc]
            CALL, 10,   // call subroutine (offset 23) → [counter, acc, 3]
            ADD,  // [counter, acc+3]
            SWAP, // [acc+3, counter]
            PUSH, 1, SUB,  // counter -= 1   [acc+3, counter-1]
            SWAP, // [counter-1, acc+3]   (loop-top stack shape)
            PUSH, 1, BR_COND, 236, // -20: jump back to loop (offset 3)
            // subroutine (offset 23): fresh stack, returns 3
            PUSH, 3, RETURN,
        ]
    }

    /// [`call_loop_bytecode`]'s twin with the recursive `CALL` replaced by the
    /// inert `PUSH 3` that returns the same value the subroutine would.
    ///
    /// The two differ in **exactly two bytes** — `CALL, 10` at offsets 11-12
    /// becomes `PUSH, 3` — so the program length, every jump offset, the loop
    /// header, the stack shape at each point and the final result are all
    /// identical. The subroutine bytes at offset 23 stay in place and become
    /// unreachable, which is what keeps the offsets aligned.
    ///
    /// That makes this the arm that turns "call_loop compiles nothing" from an
    /// observation into evidence: it holds the loop *shape* fixed and varies
    /// only the opcode under suspicion.
    fn call_loop_inert_twin_bytecode() -> Vec<u8> {
        let mut bc = call_loop_bytecode();
        // Offsets 11-12: `CALL, 10` -> `PUSH, 3`.
        assert_eq!(
            (bc[11], bc[12]),
            (CALL, 10),
            "twin patches the wrong offset: call_loop_bytecode has been edited",
        );
        bc[11] = PUSH;
        bc[12] = 3;
        bc
    }

    /// A recursive portal call in a hot loop body blocks tracing, and it is the
    /// `CALL` itself that does it — not the loop shape and not a degraded arm.
    ///
    /// Three arms, because a two-arm reading of `sum` against `call_loop` proves
    /// nothing: those are different programs with different loops, so a
    /// difference in compile count has many available explanations. The twin is
    /// the arm whose outcome is *known* to differ by one opcode.
    ///
    /// | arm | program | compiles |
    /// |---|---|---|
    /// | control | `sum(500)` — a loop known to trace | 1 |
    /// | twin | `call_loop` with `CALL` -> `PUSH 3` | 1 |
    /// | probe | `call_loop` | **0** |
    ///
    /// All three compute their expected value, so the interpreter is answering
    /// correctly throughout and the difference is purely in the JIT tier.
    ///
    /// The twin's `compiles >= 1` is the load-bearing assertion. Without it,
    /// the probe's zero is equally explained by "this loop shape cannot trace",
    /// and that is exactly the confusion `jit_trip_count_gate` fell into — there
    /// a `PUSHARG` *inside* the loop was the cause, and the loop looked innocent.
    ///
    /// `recursive_portal_call!` appears in **no other example crate**, so this
    /// is the only place in the corpus where the portal-call path is exercised
    /// at all. Its being untraceable is therefore invisible everywhere else.
    ///
    /// The probe's `0` is pinned deliberately. It records a gap, not a desired
    /// property: when the portal call starts compiling this assertion fails,
    /// which is the intended signal to come back and re-measure rather than a
    /// regression. Do not relax it to `>= 0`, which would assert nothing.
    ///
    /// The gap it records is now traced end to end. `decide_recursive_inline`
    /// answers `Inline` here (`max_unroll_recursion` defaults to 7, so the
    /// depth gates do not bite), but `exec_recursive_call` can only serve an
    /// `Inline` where `Runtime::portal_jitcode` is `Some`, and the
    /// `ClosureRuntimeWithResolver` this crate traces under leaves it `None` —
    /// only two `dispatch.rs` test fixtures override it. So the decision falls
    /// through to `TraceAction::Abort`, and nothing compiles.
    ///
    /// Redirecting that fallthrough into `exec_recursive_call_assembler` — the
    /// `do_recursive_call(assembler_call=True)` shape, and the repair this test
    /// invites — was tried and measured, and it does not work from this crate:
    /// the seam resolves its token through `get_assembler_token` →
    /// `compile_tmp_callback`, which needs `jitdriver_sd.portal_runner_adr`.
    /// This crate installs its driver through the `#[jit_interp]` macro, which
    /// has no hook for one, so the field stays 0.  The production recursive
    /// target resolver therefore declines before `compile_tmp_callback` rather
    /// than synthesizing a callback whose funcbox would contain address zero.
    /// Wiring a `tl` portal runner remains the prerequisite for compiling this
    /// edge, and it is what makes this probe worth re-measuring.
    #[test]
    fn jit_portal_call_in_loop_body_blocks_tracing() {
        let (control_got, control, ..) = compile_probe(&sum_bytecode(), 500);
        assert_eq!(control_got, 125250, "control still sums 1..=500");
        assert!(
            control >= 1,
            "control loop compiled nothing — the JIT tier is dead and this \
             experiment cannot discriminate anything (compiles={control})",
        );

        let twin_bc = call_loop_inert_twin_bytecode();
        let (twin_got, twin, ..) = compile_probe(&twin_bc, 500);
        assert_eq!(
            twin_got,
            interp::interpret(&twin_bc, 500),
            "twin agrees with the interpreter",
        );
        assert_eq!(twin_got, 1500, "twin computes the same 3*N as call_loop");
        assert!(
            twin >= 1,
            "the loop SHAPE compiles when the only change is CALL -> PUSH 3; \
             if this fails the probe's zero is not attributable to the portal \
             call (compiles={twin})",
        );

        let probe_bc = call_loop_bytecode();
        let (probe_got, probe, ..) = compile_probe(&probe_bc, 500);
        assert_eq!(probe_got, 1500, "leaf-call loop still computes 3*N");
        assert_eq!(
            probe, 0,
            "a recursive portal call in the loop body is expected to block \
             tracing today; if this now compiles, the portal-call path has \
             started working — re-measure and update this gate",
        );
    }

    #[test]
    fn jit_recursive_call_matches_interp() {
        let bc = call_loop_bytecode();
        // Sanity: the interpreter computes 3 * N.
        assert_eq!(interp::interpret(&bc, 4), 12);
        for a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "recursive-call mismatch for a={a}");
        }
    }

    #[test]
    fn recursive_fresh_entry_reds_layout() {
        use majit_metainterp::{JitCodeSym as _, JitState as _};
        let program = sum_bytecode();
        let caller = TlState {
            stackpos: 7,
            stack: VirtArray::from_slice(&[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0]),
        };
        let meta = caller.build_meta(0, program.as_slice());
        let mut sym = <TlState as majit_metainterp::JitState>::create_sym(&meta, 0);
        // Seeds `sym.stack_len_value` from the caller's live capacity.
        caller.initialize_sym(&mut sym, &meta);
        let (values, owner) = sym
            .recursive_fresh_entry_reds()
            .expect("ref-scalar-free state-field interp must support portal entry");
        // `stackpos` sits on the virtualizable with `stack`, so the red
        // block is the one identity. Fresh stackpos is zeroed on the
        // owner below, not carried as a red.
        assert_eq!(values.len(), 1, "the one vable identity");
        match values[0] {
            majit_ir::Value::Ref(majit_ir::GcRef(p)) => {
                assert_ne!(p, 0, "fresh vable base must be non-null");
                assert_ne!(
                    p, &caller as *const TlState as usize,
                    "fresh base must differ from the caller's state",
                );
            }
            ref other => panic!("slot 0 must be the vable identity Ref, got {other:?}"),
        }
        let fresh = owner
            .downcast_ref::<TlState>()
            .expect("the owner is the fresh state the reds name");
        assert_eq!(
            fresh.stack.len(),
            caller.stack.len(),
            "fresh stack re-allocated at the caller's captured capacity",
        );
        assert_eq!(fresh.stackpos, 0, "fresh frame starts empty");
    }

    #[test]
    fn recursive_fresh_alloc_free_roundtrip() {
        use majit_metainterp::{JitCodeSym as _, JitState as _};
        let program = sum_bytecode();
        let caller = TlState {
            stackpos: 3,
            stack: VirtArray::from_slice(&[9, 8, 7, 0, 0, 0, 0, 0]),
        };
        let meta = caller.build_meta(0, program.as_slice());
        let mut sym = <TlState as majit_metainterp::JitState>::create_sym(&meta, 0);
        caller.initialize_sym(&mut sym, &meta);
        let (alloc_fp, free_fp) = sym
            .recursive_fresh_alloc_free_targets()
            .expect("single-virt-array state-field interp must support portal alloc/free");
        let alloc: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(alloc_fp) };
        let free: extern "C" fn(i64) = unsafe { core::mem::transmute(free_fp) };

        let cap: i64 = 12;
        let raw = alloc(cap);
        assert_ne!(raw, 0, "fresh alloc must return a non-null pointer");
        assert_ne!(
            raw as usize, &caller as *const TlState as usize,
            "fresh state must differ from the caller's state",
        );
        unsafe {
            let fresh = &*(raw as *const TlState);
            assert_eq!(fresh.stackpos, 0, "fresh stackpos zeroed");
            assert_eq!(
                fresh.stack.len(),
                cap as usize,
                "fresh stack sized at the requested capacity",
            );
            assert!(
                fresh.stack.iter().all(|&x| x == 0),
                "fresh stack zero-initialised",
            );
        }
        // Must reclaim the Box::into_raw allocation without double-free / crash.
        free(raw);
        // A null free is a no-op (the dispatcher never frees a null, but the
        // compiled guard-fail path must tolerate it).
        free(0);
    }

    #[test]
    fn jit_various_sizes() {
        let bc = sum_bytecode();
        for a in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_bridge_exercise() {
        let bc = sum_bytecode();
        for a in [3, 5, 10, 20, 50, 100] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    /// A loop whose body branches on `counter > 50` (a *forward* BR_COND, so
    /// it is an in-trace guard, not a back-edge).  Traced while `counter > 50`,
    /// it guard-fails on every iteration once `counter <= 50` — driving the
    /// state-field blackhole forward-resume path (the `!should_bridge` arm of
    /// `back_edge_internal`) rather than the clean `is_finish` loop-exit.
    /// The computed value is irrelevant; the assertion is that the JIT result
    /// equals the plain interpreter result across the guard-failure divergence.
    fn divergent_branch_bytecode() -> Vec<u8> {
        vec![
            PUSH, 0,       // [0] acc = 0
            PUSHARG, // [2] counter = N            stack = [acc, counter]
            // loop header (offset 3):
            PICK, 0, //       [3] dup counter
            BR_COND, 2,      //    [5] if counter != 0 -> body(9); else fall through
            POP,    //      [7] pop counter
            RETURN, //      [8] return acc
            // body (offset 9):
            PICK, 0, //       [9]  dup counter            [acc, ctr, ctr]
            PUSH, 50, //      [11] push 50                [acc, ctr, ctr, 50]
            GT, //      [13] ctr > 50 ?             [acc, ctr, (ctr>50)]
            BR_COND, 5, //    [14] if ctr>50 -> skip_extra(21)
            // not-taken (ctr <= 50): acc += 1 (the divergent path)
            SWAP, //          [16] [ctr, acc]
            PUSH, 1,    //       [17] [ctr, acc, 1]
            ADD,  //          [19] [ctr, acc+1]
            SWAP, //          [20] [acc+1, ctr]
            // skip_extra (offset 21): common tail — acc += counter; counter -= 1
            SWAP, //          [21] [ctr, acc]
            PICK, 1,    //       [22] [ctr, acc, ctr]
            ADD,  //          [24] [ctr, acc+ctr]
            SWAP, //          [25] [acc+ctr, ctr]
            PUSH, 1,   //       [26] [acc+ctr, ctr, 1]
            SUB, //          [28] [acc+ctr, ctr-1]
            PUSH, 1, //       [29] push 1 (unconditional back-jump cond)
            BR_COND, 226, //  [31] -30: jump to loop header(3)
        ]
    }

    #[test]
    fn jit_divergent_branch_matches_interp() {
        let bc = divergent_branch_bytecode();
        // N spans both sides of the `counter > 50` split so the traced
        // (`> 50`) path guard-fails for the lower half of every run.
        for a in [3, 5, 49, 50, 51, 60, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let got = run_jit(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }
}
