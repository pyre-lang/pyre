//! Two `#[jit_interp]` machines must be able to share one module.
//!
//! Every item the macro emits lands in the CALLER's module. Five of them used
//! to carry fixed names (`__JitMeta`, `__JitSym`, `__jit_loop_carried_boxes`,
//! `__majit_recursive_fresh_alloc`, `__majit_recursive_fresh_free`) while five
//! others were already suffixed with the annotated function's name, so a second
//! machine in the same module failed with E0428 "defined multiple times". Every
//! machine in the tree worked around it by living in its own `mod`.
//!
//! THIS FIXTURE NEEDS BOTH SHAPES, and a green run with only one proves
//! nothing about the other two idents. `__majit_recursive_fresh_alloc` /
//! `_free` are emitted ONLY when `supports_fresh_alloc` holds — zero ref
//! scalars, no opaque carriers, no fixed arrays, and exactly one `[int; virt]`
//! array (`codegen_state.rs` `let supports_fresh_alloc`). So:
//!   * `CounterState` is scalars-only  -> supports_fresh_alloc = FALSE (3 idents)
//!   * `StackState`   has one virt array -> supports_fresh_alloc = TRUE (5 idents)
//!
//! Adding a machine here without checking which side of that predicate it falls
//! on re-opens the hole this fixture exists to close.
//!
//! CEILING, deliberately not tested because it cannot be fixed by naming:
//! two machines over the SAME state type still collide on
//! `impl JitState for #state_type` (E0119). Unique idents buy co-residence only
//! for machines whose `state` types differ.

use majit_metainterp::virt_array::VirtArray;
pub type Bytecode = [u8];

const C_ADD: u8 = 1; // acc += n; n -= 1
const C_BR_COND: u8 = 2; // [C_BR_COND, off]: if n != 0 jump
const C_RETURN: u8 = 3;

const PUSH: u8 = 2;
const POP: u8 = 3;
const SWAP: u8 = 4;
const PICK: u8 = 6;
const ADD: u8 = 8;
const SUB: u8 = 9;
const BR_COND: u8 = 18;
const RETURN: u8 = 21;
const PUSHARG: u8 = 22;

// Scalars only: supports_fresh_alloc is FALSE, so this machine alone would
// leave `__majit_recursive_fresh_alloc` / `_free` unexercised.
struct CounterState {
    acc: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = CounterState,
    env = Bytecode,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        acc: int,
        n: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop_counter(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<CounterState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = CounterState {
        acc: 0,
        n: inputarg,
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
            C_ADD => {
                state.acc += state.n;
                state.n -= 1;
            }
            C_BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                if state.n != 0 {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            C_RETURN => break,
            _ => {}
        }
    }

    state.acc
}

/// `acc = n + (n-1) + ... + 1`, a hot loop over two int scalars.
fn counter_program() -> Vec<u8> {
    vec![
        C_ADD, // 0
        C_BR_COND, 253,      // 1  (-3 -> back to 0)
        C_RETURN, // 3
    ]
}

// One `[int; virt]` array and no ref scalars/opaques/fixed arrays: this is the
// shape that turns `supports_fresh_alloc` ON, so the two extern "C" idents are
// emitted here and nowhere else in this file.
struct StackState {
    stackpos: i64,
    stack: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = StackState,
    env = Bytecode,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop_stack(program: &Bytecode, inputarg: i64, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<StackState> =
        majit_metainterp::JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = StackState {
        stackpos: 0,
        stack: VirtArray::filled(0i64, program.len()),
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
            PICK => {
                let i = program[pc] as usize;
                pc += 1;
                let v = state.stack[(state.stackpos as usize) - i - 1];
                state.stack[state.stackpos as usize] = v;
                state.stackpos += 1;
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
            RETURN => break,
            PUSHARG => {
                state.stack[state.stackpos as usize] = inputarg;
                state.stackpos += 1;
            }
            _ => {}
        }
    }

    state.stackpos -= 1;
    state.stack[state.stackpos as usize]
}

/// `sum(N) = N + (N-1) + ... + 1`, a hot loop over the virt stack.
fn sum_program() -> Vec<u8> {
    vec![
        PUSH, 0,       // 0
        PUSHARG, // 2
        PICK, 0, // 3  (loop header)
        BR_COND, 2,      // 5  -> body @9
        POP,    // 7
        RETURN, // 8
        SWAP,   // 9
        PICK, 1,    // 10
        ADD,  // 12
        SWAP, // 13
        PUSH, 1, SUB, // 14
        PUSH, 1, // 17
        BR_COND, 238, // 19 -> back to the header @3
    ]
}

/// The compile-time assertion is that this file builds at all: before the five
/// idents were suffixed, two machines in one module were an E0428. The runtime
/// assertions confirm each machine still computes its own answer, i.e. that
/// renaming did not cross the two expansions' wires.
#[test]
fn two_machines_share_one_module() {
    let n = 10i64;
    let expected = n * (n + 1) / 2; // 55

    assert_eq!(mainloop_counter(&counter_program(), n, 4), expected);
    assert_eq!(mainloop_stack(&sum_program(), n, 4), expected);
}
