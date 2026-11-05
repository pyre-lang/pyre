//! A virtualizable too large to number must abandon the compile, not compile a
//! guard that has no resume data.
//!
//! `resume.py` raises `TagOverflow` once a livebox index no longer fits
//! the tag's 13 value bits, and `optimizer.py:761-766
//! store_final_boxes_in_guard` answers it with `raise compile.giveup()` — the
//! whole compilation is abandoned and the interpreter keeps running from state
//! it never handed over. Every element of a virtualizable array is a distinct
//! livebox (`resume.py _number_boxes`), so a long enough declared
//! length reaches that limit on its own.
//!
//! What makes this a correctness test rather than a capacity one: the failure
//! is silent. A guard emitted without resume data still compiles, and the
//! damage appears only when it later fails and the deopt rebuilds interpreter
//! state out of an empty numbering. The interpreter then resumes with values
//! that are not the ones it had, so the machine's own arithmetic — not any JIT
//! statistic — is what witnesses it.
use core::sync::atomic::{AtomicUsize, Ordering};

use majit_metainterp::virt_array::VirtArray;

pub type Bytecode = [u8];

/// Past the 13 value bits `resume.py:100-103` tags a box index with.
const SLOTS: usize = 8192;

const OP_PUSH1: u8 = 1;
const OP_PUSH2: u8 = 2;
const OP_ADD: u8 = 3;
const OP_DRAIN: u8 = 4;
const OP_BACK: u8 = 5;
const OP_RET: u8 = 6;

pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

struct StackMachine {
    stack: VirtArray<i64>,
    sp: usize,
    counter: i64,
    acc: i64,
}

#[majit_macros::jit_interp(
    state = StackMachine,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        stack: [int; virt],
        sp: int(usize),
        counter: int,
        acc: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Bytecode, iterations: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<StackMachine> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _before, _after, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = StackMachine {
        stack: VirtArray::filled(0i64, SLOTS),
        sp: 0usize,
        counter: iterations,
        acc: 0i64,
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
            OP_PUSH1 => {
                state.stack[state.sp] = 1;
                state.sp += 1;
            }
            OP_PUSH2 => {
                state.stack[state.sp] = 2;
                state.sp += 1;
            }
            OP_ADD => {
                let top = state.stack[state.sp - 1];
                state.sp -= 1;
                state.stack[state.sp - 1] += top;
            }
            OP_DRAIN => {
                state.sp -= 1;
                state.acc += state.stack[state.sp];
            }
            OP_BACK => {
                state.counter -= 1;
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
    state.acc
}

/// Each iteration pushes 1 and 2, adds them, and drains the sum, so the answer
/// counts the iterations that actually ran. Losing the compile is allowed;
/// losing iterations is not.
#[test]
fn a_virtualizable_too_large_to_number_still_interprets_correctly() {
    let program = vec![OP_PUSH1, OP_PUSH2, OP_ADD, OP_DRAIN, OP_BACK, OP_RET];
    // Kept small on purpose: every compile attempt at this declared length pays
    // the numbering cost, and the trace is re-attempted after each giveup, so the
    // iteration count is what sets the test's runtime. The unfixed tree loses all
    // but 5 iterations, so a few hundred is already a wide margin.
    const ITERATIONS: i64 = 300;

    // The cold run never reaches the threshold, so it fixes the answer without
    // the JIT having any say in it.
    let cold = mainloop(&program, ITERATIONS, u32::MAX);
    assert_eq!(cold, ITERATIONS * 3, "cold interpretation");

    let warm = mainloop(&program, ITERATIONS, 3);
    assert_eq!(
        warm,
        cold,
        "a virtualizable that cannot be numbered must abandon the compile and \
         keep interpreting; instead {} of {ITERATIONS} iterations survived",
        warm / 3
    );
}
