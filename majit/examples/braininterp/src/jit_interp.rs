/// JIT-enabled Brainfuck interpreter using `#[jit_interp]` and virtualizable
/// state fields.
///
/// The program counter and bytecode are green inputs. The pointer and tape are
/// red state, with tape cells represented as symbolic values while tracing.
/// A backward `]` branch identifies the loop header.
use majit_metainterp::virt_array::VirtArray;
pub type Bytecode = [u8];

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Hot loops majit compiled.
///
/// Until this counter existed, this crate had **no** observable for the JIT
/// tier at all — no compile count, no `ops_after`, no degraded-arm read. Its
/// two tests compare JIT output against the interpreter, which an interpreter
/// answering alone satisfies perfectly. That is not hypothetical here: this
/// example ran its whole suite green while `Total ops recorded` went 9610 -> 0
/// under an experiment that declared `greens` (see the header).
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Ops in the last compiled loop body after optimization.
///
/// `COMPILES > 0` is necessary but NOT sufficient: an entirely empty dispatch
/// still compiles a trace — one whose whole optimized body is `Finish()`, i.e.
/// `ops_after == 1`. A compile counter counts TRACES, not WORK.
///
/// On THIS crate it does not separate a compiled loop from a compiled
/// nothing either, and it never could: see [`LAST_OPS_BEFORE`].
pub static LAST_OPS_AFTER: AtomicUsize = AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: AtomicBool = AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: AtomicBool = AtomicBool::new(false);

/// Operations recorded before optimization for the most recently compiled
/// loop. Tests compare this with [`LAST_OPS_AFTER`] to distinguish a useful
/// loop body from a segmented trace-limit terminator.
pub static LAST_OPS_BEFORE: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(test))]
const TAPE_SIZE: usize = 30000;
#[cfg(test)]
const TAPE_SIZE: usize = 128;
const DEFAULT_THRESHOLD: u32 = 3;

struct BfState {
    pointer: i64,
    tape: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = BfState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        pointer: int,
        tape: [int; virt],
    },
)]
fn mainloop(program: &Bytecode, threshold: u32) -> String {
    let mut driver: majit_metainterp::JitDriver<BfState> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_green_key, ops_before, ops_after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        LAST_OPS_BEFORE.store(ops_before, Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        LAST_HAS_JUMP.store(shape.has_jump, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(shape.has_always_fails, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = BfState {
        pointer: 0,
        tape: VirtArray::filled(0i64, TAPE_SIZE),
    };
    let mut output = String::new();

    // RPython warmspot.py: make_jitcodes() before pyjitpl.py finish_setup().
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
        // Keep the observer/replay form until the backward `]` arm lowers to a
        // sub-JitCode. In the single-executor form that arm is an abort stub,
        // so tracing repeats the loop-header guards until trace segmentation
        // instead of compiling a useful loop body.
        jit_merge_point!();
        let ch = program[pc];

        match ch {
            b'>' => {
                state.pointer += 1;
                pc += 1;
            }
            b'<' => {
                state.pointer -= 1;
                pc += 1;
            }
            b'+' => {
                state.tape[state.pointer as usize] += 1;
                pc += 1;
            }
            b'-' => {
                state.tape[state.pointer as usize] -= 1;
                pc += 1;
            }
            b'.' => break,
            b',' => break,
            b'[' => {
                if state.tape[state.pointer as usize] == 0 {
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
            b']' if state.tape[state.pointer as usize] != 0 => {
                // RPython `BrainInterpreter.interp_char` keeps this scan in
                // the opcode body, so the translator sees the loop and its
                // loop-carried locals instead of an opaque slice helper.
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

    // Handle I/O after breaking out of the traced loop.
    // The trace aborts on '.' and ','; the interpreter handles them here
    // and re-enters the loop.
    while pc < program.len() {
        let ch = program[pc];
        if ch == b'.' {
            output.push(state.tape[state.pointer as usize] as u8 as char);
            pc += 1;
        } else if ch == b',' {
            state.tape[state.pointer as usize] = 0;
            pc += 1;
        } else {
            // Re-enter the traced mainloop for the remaining program.
            let remaining_output = mainloop_resume(program, &mut state, &mut driver, pc, threshold);
            output.push_str(&remaining_output);
            break;
        }
    }

    output
}

/// Resume mainloop execution from a given pc, reusing existing state and driver.
fn mainloop_resume(
    program: &Bytecode,
    state: &mut BfState,
    _driver: &mut majit_metainterp::JitDriver<BfState>,
    mut pc: usize,
    _threshold: u32,
) -> String {
    let mut output = String::new();

    while pc < program.len() {
        let ch = program[pc];
        if ch == b'>' {
            state.pointer += 1;
            pc += 1;
        } else if ch == b'<' {
            state.pointer -= 1;
            pc += 1;
        } else if ch == b'+' {
            state.tape[state.pointer as usize] += 1;
            pc += 1;
        } else if ch == b'-' {
            state.tape[state.pointer as usize] -= 1;
            pc += 1;
        } else if ch == b'.' {
            output.push(state.tape[state.pointer as usize] as u8 as char);
            pc += 1;
        } else if ch == b',' {
            state.tape[state.pointer as usize] = 0;
            pc += 1;
        } else if ch == b'[' {
            if state.tape[state.pointer as usize] == 0 {
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
        } else if ch == b']' {
            if state.tape[state.pointer as usize] != 0 {
                let target = find_matching_open(program, pc);
                pc = target;
            } else {
                pc += 1;
            }
        } else {
            pc += 1;
        }
    }

    output
}

/// Find the matching '[' for a ']' at the given position.
fn find_matching_open(code: &[u8], close_pos: usize) -> usize {
    let mut need: i32 = 1;
    let mut p = close_pos - 1;
    while need > 0 {
        if code[p] == b']' {
            need += 1;
        } else if code[p] == b'[' {
            need -= 1;
        }
        if need > 0 {
            p -= 1;
        }
    }
    p
}

pub struct JitBrainInterp {
    threshold: u32,
}

impl Default for JitBrainInterp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitBrainInterp {
    pub fn new() -> Self {
        JitBrainInterp {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn run(&mut self, code: &[u8]) -> String {
        mainloop(code, self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::interp;
    static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// Run `prog` and return only its output.
    ///
    /// Not an alias for `JitBrainInterp::new().run()`: the lock is the point.
    /// Neither this nor [`compile_probe`] may call the other — the mutex is
    /// plain, and re-entering it on one thread deadlocks.
    fn run_jit(prog: &[u8]) -> String {
        let _guard = PROBE_LOCK.lock();
        JitBrainInterp::new().run(prog)
    }

    /// Run `prog` with the tier counters reset, returning
    /// `(output, compiles, ops_before, ops_after)`.
    fn compile_probe(
        prog: &[u8],
    ) -> (String, usize, usize, usize, majit_metainterp::LoopBodyShape) {
        let _guard = PROBE_LOCK.lock();
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_BEFORE.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let got = JitBrainInterp::new().run(prog);
        (
            got,
            COMPILES.load(Ordering::Relaxed),
            LAST_OPS_BEFORE.load(Ordering::Relaxed),
            LAST_OPS_AFTER.load(Ordering::Relaxed),
            majit_metainterp::LoopBodyShape {
                has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
                has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
            },
        )
    }

    fn multiply_gate_bf(a: u8, b: u8) -> Vec<u8> {
        let mut prog = Vec::new();
        for _ in 0..a {
            prog.push(b'+');
        }
        prog.push(b'[');
        prog.push(b'>');
        for _ in 0..b {
            prog.push(b'+');
        }
        prog.extend_from_slice(b"<-]");
        prog.extend_from_slice(b">.");
        prog
    }

    #[test]
    fn jit_tier_closes_the_brainfuck_loop() {
        const A: u8 = 7;
        const B: u8 = 3;
        let prog = multiply_gate_bf(A, B);
        let (got, compiles, ops_before, ops_after, shape) = compile_probe(&prog);

        // The interpreter must still answer correctly whatever the tier does.
        assert_eq!(
            got.chars().count(),
            1,
            "multiply({A}*{B}) printed {got:?}; the gate expects exactly one cell"
        );
        let cell = got.chars().next().unwrap() as u32;
        assert_eq!(
            cell,
            u32::from(A) * u32::from(B),
            "multiply({A}*{B}) left {cell} in cell 1, so the loop ran {} passes \
             rather than {A}",
            cell / u32::from(B)
        );

        let bf_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "BfState")
            .collect();
        let degraded: Vec<&str> = bf_arms.iter().map(|a| a.arm).collect();
        assert!(
            degraded.is_empty(),
            "a braininterp opcode body still installs an abort stub: {bf_arms:?}"
        );
        assert!(
            compiles > 0,
            "the bracket arms lower, but the hot Brainfuck loop compiled nothing"
        );
        assert!(
            shape.closes_a_loop(),
            "the compiled body does not close a loop: {}",
            shape.why_not().unwrap_or("closes a loop")
        );

        println!(
            "[tier-live] multiply({A}*{B}) = {cell}, {compiles} loops compiled, \
             {ops_before} ops recorded, \
             {ops_after} after opt, degraded {degraded:?}"
        );
    }

    #[test]
    fn jit_matches_interp_hot_loops() {
        let code = b"+++++[->+<]>++++++++[<++++++++>-]<.";
        let expected = interp::interpret(code);
        let got = run_jit(code);
        assert_eq!(got, "h");
        assert_eq!(got, expected);
    }

    #[test]
    fn jit_no_loop() {
        let output = run_jit(b"+++");
        assert_eq!(output.len(), 0);
    }
}
