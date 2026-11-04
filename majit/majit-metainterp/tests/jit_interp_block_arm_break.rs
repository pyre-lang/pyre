//! Ensures a terminal dispatch arm spelled `{ <work>; break }` does not lose
//! its control transfer during lowering.
//!
//! A block arm is lowered statement by statement, so an unsupported terminal
//! `break` must degrade the arm instead of being discarded as inert.

use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

/// `regs[0] -= 1`, advance.
const OP_DEC: u8 = 1;
/// Back edge: jump to 0 while `regs[0] != 0`, else fall past.
const OP_BACK: u8 = 2;
/// The arm under test: a *block* body whose tail is `break`.
const OP_FIN: u8 = 3;
/// Reached only if `OP_FIN`'s `break` was dropped. Bumps the same counter,
/// which is what turns the dropped transfer into a wrong number rather than a
/// silent no-op.
const OP_TICK: u8 = 4;
/// Bare `break` arm — `ArmPattern::Halt`, the spelling that always worked —
/// so a dropped `break` terminates the program instead of spinning.
const OP_END: u8 = 5;

struct BlockBreakState {
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = BlockBreakState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_block_break(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<BlockBreakState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = BlockBreakState {
        regs: VirtArray::filled(0i64, 2),
    };
    state.regs[0] = program[program.len() - 1] as i64;
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    loop {
        jit_merge_point!();
        let opcode = program[pc];
        match opcode {
            OP_DEC => {
                state.regs[0] -= 1;
                pc += 1;
            }
            OP_BACK => {
                if state.regs[0] != 0 {
                    pc = 0;
                    continue;
                } else {
                    pc += 1;
                }
            }
            // The shape under test. A block body, not a bare `break`.
            OP_FIN => {
                state.regs[1] += 1;
                pc += 1;
                break;
            }
            OP_TICK => {
                state.regs[1] += 1;
                pc += 1;
            }
            _ => break,
        }
    }
    state.regs[1]
}

/// `[OP_DEC, OP_BACK, OP_FIN, OP_TICK, OP_END, n]` — the trailing byte seeds
/// `regs[0]`, so the loop runs `n` times before reaching `OP_FIN`.
fn program_for(n: u8) -> Vec<u8> {
    vec![OP_DEC, OP_BACK, OP_FIN, OP_TICK, OP_END, n]
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_block_break(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_block_break(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// Confirms that degradation preserves the arm's interpreted result instead
/// of silently dropping the terminal `break`.
#[test]
fn a_degraded_terminal_arm_still_answers_correctly() {
    for n in [17u8, 18u8] {
        let ticks = dispatch_block_break(&program_for(n), 4);
        assert_eq!(
            ticks, 1,
            "n={n}: OP_FIN must run exactly once however the arm is lowered"
        );
    }
}

/// The refusal must be *visible*, not merely correct.
///
/// Before the guard the arm lowered and silently misbehaved; after it, the arm
/// cannot lower and is recorded. Asserting the name pins that the arm reached
/// the fallback rather than being fixed somewhere upstream — if a later change
/// lowers a block `break` properly, this fires and should be deleted, not
/// relaxed.
#[test]
fn the_block_break_arm_is_recorded_as_degraded() {
    let _ = install();
    let arms: Vec<String> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == "BlockBreakState")
        .map(|e| e.arm.to_string())
        .collect();
    assert!(
        arms.iter().any(|a| a == "OP_FIN"),
        "OP_FIN must be recorded as degraded: its block body ends in `break`, \
         which cannot be lowered in place, so the arm degrades to an abort stub \
         and runs interpreted; recorded={arms:?}"
    );
}
