/// JIT-enabled tinyframe interpreter — auto-generated tracing via `#[jit_interp]` + `state_fields`.
///
/// Greens: [pc, bytecode]
/// Reds:   [regs]  (tracked via state_fields)
use crate::interp::{ADD, JUMP_IF_ABOVE, LOAD, RETURN};
use majit_metainterp::embed::Census;
use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

struct TinyFrameState {
    regs: VirtArray<i64>,
    /// What `RETURN` hands back.
    ///
    /// The `; state` merge point leaves the loop through `break` before it
    /// assigns the walk's resume pc, so after the loop `pc` still names the
    /// position the walk started from and `program[pc + 1]` — the operand
    /// saying which register holds the result — cannot be read there. The
    /// result has to arrive in `state`.
    ret: i64,
}

const DEFAULT_THRESHOLD: u32 = 3;

#[majit_macros::jit_interp(
    state = TinyFrameState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
        ret: int,
    },
)]
fn mainloop(
    program: &Bytecode,
    num_regs: usize,
    init_regs: &[(usize, i64)],
    threshold: u32,
) -> i64 {
    let mut driver: majit_metainterp::JitDriver<TinyFrameState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    // Every counter the tier gate reads, plus the last body's op count and
    // shape, off the driver callbacks. `Census::begin` is what opens a window
    // over them.
    Census::install(&mut driver);
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = TinyFrameState {
        regs: VirtArray::filled(0, num_regs),
        ret: 0,
    };
    for &(r, v) in init_regs {
        state.regs[r] = v;
    }

    // RPython warmspot.py: make_jitcodes() before pyjitpl.py finish_setup().
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        // `; state` selects the single-executor close: the walk's final state is
        // transferred into `state` here and the native loop resumes at the close
        // pc, instead of discarding the walk outcome and re-running the circuit
        // the walk already executed.
        // With `TINYFRAME_NATIVE_PROBE=1`, every pass of the NATIVE loop prints
        // its own pc and the state it is about to hand to the merge point. This
        // sits BEFORE the merge point on purpose: the merge point is where a
        // walk takes over and leaves the loop, so a probe after it never prints
        // the pass the walk consumed — the one pass whose state is the question.
        if std::env::var_os("TINYFRAME_NATIVE_PROBE").is_some() {
            eprintln!(
                "[native-mp] pc={pc} regs={:?} ret={}",
                state.regs, state.ret
            );
        }
        jit_merge_point!(driver, program, pc; state);
        if pc == 0 {
            can_enter_jit!(driver, pc, &mut state, program, || {});
        }
        let opcode = program[pc];

        match opcode {
            LOAD => {
                let val = program[pc + 1] as i64;
                let reg = program[pc + 2] as usize;
                state.regs[reg] = val;
                pc += 3;
            }
            ADD => {
                let r1 = program[pc + 1] as usize;
                let r2 = program[pc + 2] as usize;
                let r3 = program[pc + 3] as usize;
                state.regs[r3] = state.regs[r1] + state.regs[r2];
                pc += 4;
            }
            JUMP_IF_ABOVE => {
                let r1 = program[pc + 1] as usize;
                let r2 = program[pc + 2] as usize;
                let tgt = program[pc + 3] as usize;
                let v1 = state.regs[r1];
                let v2 = state.regs[r2];
                let jump = v1 > v2;
                if jump {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            // Stores into `ret` and then leaves through an in-arm `return`,
            // never `{ store; break }`: `classify.rs` `is_break_expr` requires
            // the arm body to be exactly `break`, so a composite body classifies
            // `Lowerable` and its tail `break` reaches `lower_stmt_fallback`,
            // which guards an enclosed `return` but not an enclosed `break` —
            // the statement is inert and is silently dropped, leaving the
            // lowered arm to fall through to the dispatch back-edge.
            RETURN => {
                let r = program[pc + 1] as usize;
                state.ret = state.regs[r];
                return state.ret;
            }
            // The panic sits in the arm rather than below the loop. The loop
            // has two ways out — this dispatch and the merge point's own
            // `break` on a walk that reached a terminal return — so falling out
            // of it does not identify a bad opcode; only reaching this arm
            // does.
            _ => panic!("fell off end of code"),
        }
    }
    // Reached only when the merge point broke out on a walk that already ran the
    // terminal opcode, so the result is whatever that opcode parked in `ret`.
    state.ret
}

// -- Public wrapper matching the old API --

pub struct JitTinyFrameInterp {
    threshold: u32,
}

impl Default for JitTinyFrameInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitTinyFrameInterp {
    pub fn new() -> Self {
        JitTinyFrameInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// Run a tinyframe Code with initial integer register values.
    pub fn run(&mut self, code: &crate::interp::Code, init_regs: &[(usize, i64)]) -> i64 {
        mainloop(&code.code, code.regno, init_regs, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    /// For tests that assert only on the result. They still compile, so they
    /// must not run inside another test's census window — the window is what
    /// makes [`compile_probe`]'s numbers this run's rather than the process's.
    ///
    /// [`Census::begin`] is the only lock here, and it is not reentrant:
    /// [`run_jit`] and [`compile_probe`] are the only two ways a test may enter
    /// the JIT, and neither may call the other.
    fn run_jit(code: &interp::Code, init_regs: &[(usize, i64)]) -> i64 {
        let _census = Census::begin();
        JitTinyFrameInterp::new().run(code, init_regs)
    }

    /// How many unroll-free fallback compiles the tier gate currently sees —
    /// see the block at its assertion. `MC_DIAG` slot 73.
    const EXPECT_UNPEELED: u64 = 0;

    /// Run inside a census window, returning `(result, counts, unpeeled)`.
    ///
    /// The window is what separates this run's compiles from every other test's:
    /// the counters behind it are process-global, so an absolute read carries
    /// whatever the rest of the binary left behind, and a zero from it would be
    /// indistinguishable from an inherited value.
    fn compile_probe(
        code: &interp::Code,
        init_regs: &[(usize, i64)],
    ) -> (i64, majit_metainterp::embed::CensusCounts, u64) {
        let census = Census::begin();
        // Slot 73 is process-global and CUMULATIVE with no reset of its own, so
        // it is read as a delta across this run and inside the census window,
        // for the same reason the counters are.
        let unpeeled_before = majit_metainterp::mc_diag(73);
        let got = JitTinyFrameInterp::new().run(code, init_regs);
        let unpeeled = majit_metainterp::mc_diag(73) - unpeeled_before;
        (got, census.counts(), unpeeled)
    }

    /// A real loop body was compiled — the one property no assertion on a
    /// *result* can establish.
    ///
    /// This certifies that the JIT tier produced a non-empty compiled body. It
    /// is silent on whether that body's resume data is well-formed: a trace can
    /// carry a malformed promote snapshot and still be counted here.
    ///
    /// All three parts are needed and none implies another:
    ///
    /// 1. `loops_compiled` non-zero over the window. A green suite, agreement
    ///    with `interp::Frame::interpret` and even an exact absolute trip count
    ///    are all satisfied by the interpreter answering alone.
    /// 2. `last_ops_after` pinned by equality. `compiles >= 1` is necessary but
    ///    NOT sufficient: an entirely empty dispatch still compiles a trace —
    ///    one whose whole optimized body is `Finish()`, i.e. `ops_after == 1`.
    ///    A compile counter counts TRACES, not WORK, and every inequality a
    ///    real loop satisfies that degenerate body satisfies too.
    /// 3. No degraded dispatch arm. An arm whose body did not lower is an abort
    ///    stub, so any trace reaching it aborts. The assertion names the arm,
    ///    which an abort count cannot: `trace action at pc=N -> Abort` reports
    ///    the trace-START pc, not the arm that caused it.
    ///
    /// The arm registry is process-wide, so it is asked about this machine's
    /// `state = TinyFrameState`, and read *after* a run because nothing
    /// installs the dispatch JitCode until the interpreter is entered.
    ///
    /// The subject is the `jit_matches_interp` program seeded through
    /// `init_regs`, so the result assertion is itself an absolute trip count:
    /// r0 counts up by 1 per pass while r2 > r0, so `run(.., [(2, N)])` returns
    /// the number of passes. One compiled artifact carries both properties.
    #[test]
    fn jit_tier_is_alive() {
        const N: i64 = 1001;
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );
        let (got, counts, unpeeled) = compile_probe(&code, &[(2, N)]);
        let (compiles, ops_after) = (counts.loops_compiled, counts.last_ops_after);
        let shape = counts.last_loop_body_shape;
        // `unpeeled` counts loops this run compiled through the unroll-free
        // fallback: the unrolled compile raised `InvalidLoop`, and the retry
        // WITHOUT the peel succeeded. The retry succeeding is what makes a
        // relapse silent — measured on this crate with the `identity_input_ref`
        // repair present and absent, one variable:
        //
        //   INVARIANT:     `compiles`, `closes_a_loop()` and the result read
        //                  the same in both arms, so not one of them notices.
        //   DISCRIMINATING: `unpeeled` 0 vs 1, and `ops_after` 9 vs 4.
        //
        // So the `ops_after` pin below is the other half of this one: they are
        // two arms of one defect, not two independent measurements.
        //
        // Not a target value in either direction. A rise means a subject lost
        // its peel and the unroll-free fallback is live on it again; the
        // response is to find the subject, not to raise the pin.
        assert_eq!(
            unpeeled, EXPECT_UNPEELED,
            "expected {EXPECT_UNPEELED} unroll-free fallback compile(s) out of \
             {compiles}, saw {unpeeled}"
        );
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
        assert_eq!(
            got, N,
            "count_to({N}) = {got}, so the loop the tier assertions below \
             describe ran {got} passes rather than {N}"
        );

        // Stronger than filtering `degraded_dispatch_arms()` by hand: an empty
        // list is also what a portal that was never installed produces, and
        // this settles which of the two happened.
        majit_metainterp::assert_no_degraded_dispatch_arms("TinyFrameState");

        // Zero-vs-nonzero is the property; a later change that legitimately
        // mints more than one artifact is not this regression.
        assert!(
            compiles >= 1,
            "count_to({N}) compiled {compiles} loops — the JIT tier is inert and \
             the interpreter is answering alone, which every other assertion in \
             this file would still pass"
        );
        // 9 is the peeled body: `Label`/`IntAdd`/`IntGt`/`GuardTrue` twice plus
        // the closing `Jump`. A drop to 4 is the unpeeled fallback body — read
        // it together with the `unpeeled` pin above, which moves with it.
        assert_eq!(
            ops_after, 9,
            "compiled loop body is {ops_after} ops, not the pinned 9 — 4 is the \
             unpeeled fallback body and a value of 1 means the body is a bare \
             `Finish()`, i.e. a dispatch that lowered nothing at all"
        );
        println!(
            "[tier-alive] count_to({N}) = {got}, compiled {compiles} loop(s) of {ops_after} ops, 0 degraded arms"
        );
    }

    #[test]
    fn jit_loop_count_to_40() {
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 40 => r2
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );
        let result = run_jit(&code, &[]);
        assert_eq!(result, 40);
    }

    #[test]
    fn jit_loop_sum() {
        // loop.tf: sum from 1 to N
        let code = interp::compile(
            "
        main:
        LOAD 0 => r1
        LOAD 1 => r2
        @add
        ADD r2 r1 => r1
        JUMP_IF_ABOVE r0 r1 @add
        RETURN r1
        ",
        );
        let result = run_jit(&code, &[(0, 40)]);
        assert_eq!(result, 40);
    }

    /// The failing fixture's program: straight-line, no back edge anywhere.
    /// `r1` is written at pc 0 and read by the `ADD` at pc 6.
    ///
    ///   pc 0   LOAD 111 => r1
    ///   pc 3   LOAD 222 => r2
    ///   pc 6   ADD r1 r2 => r0
    ///   pc 10  RETURN r0
    fn straight_line_program() -> interp::Code {
        interp::compile(
            "
        main:
        LOAD 111 => r1
        LOAD 222 => r2
        ADD r1 r2 => r0
        RETURN r0
        ",
        )
    }

    #[test]
    fn jit_trace_reads_input_written_after_the_arming_pc() {
        let _census = Census::begin();
        let code = straight_line_program();
        // Threshold 1: the entry door arms on its single hit at pc 0.
        let got = mainloop(&code.code, code.regno, &[], 1);
        assert_eq!(
            got, 333,
            "expected 111 + 222; a result of 222 means the walk read r1 as 0 — \
             its value at the arming pc 0 — instead of the 111 native stored \
             there before the walk headed at pc 3"
        );
    }

    fn pre_loop_read_program() -> interp::Code {
        interp::compile(
            "
        main:
        LOAD 111 => r3
        LOAD 1 => r1
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        ADD r0 r3 => r0
        RETURN r0
        ",
        )
    }

    #[test]
    fn jit_finish_trace_reads_pre_loop_input() {
        let code = pre_loop_read_program();
        let got = run_jit(&code, &[(2, 4)]);
        assert_eq!(
            got, 115,
            "expected 4 + 111 from the walk; a result of 4 would mean a \
             Finish-terminated trace read r3 as 0 even with no arming-pc skew"
        );
    }

    #[test]
    fn jit_matches_interp() {
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );

        for n in [5, 10, 20, 40] {
            // Interpreter
            let mut frame = interp::Frame::new(&code);
            frame.registers[2] = Some(interp::Object::Int(n));
            let interp_result = frame.interpret(&code).as_int();

            // JIT
            let jit_result = run_jit(&code, &[(2, n)]);

            assert_eq!(jit_result, interp_result, "count_to({n}) mismatch");
        }
    }
}
