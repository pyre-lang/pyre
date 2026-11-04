//! Identifies which part of `state.regs[<computed>] += 1` prevents sub-JitCode
//! lowering.
//!
//! The fixture varies compound assignment, computed indexing, and same-slot
//! read-modify-write independently against a common control.

use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const OP_CTRL_ADD: u8 = 1;
const OP_COMPOUND_LETIDX: u8 = 2;
const OP_RMW_LETIDX: u8 = 3;
const OP_PLAIN_COMPUTEDIDX: u8 = 4;
const OP_COMPOUND_COMPUTEDIDX: u8 = 5;

// Named as the `jit_interp` state type; the macro lowers the interpreter body
// into jitcode and never runs it, so nothing constructs the struct.
#[expect(dead_code)]
struct CompoundAssignState {
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = CompoundAssignState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_compound_assign(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<CompoundAssignState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = CompoundAssignState {
        regs: VirtArray::filled(0i64, 4),
    };
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
            // Control: three distinct let-bound indices, no compound operator,
            // no same-slot read-modify-write. Known to lower.
            OP_CTRL_ADD => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] + state.regs[b];
                pc += 4;
            }
            // Ingredient 1 alone: compound operator, index is a `let`-bound
            // local. (Also same-slot RMW, implicitly — `+=` cannot avoid it.)
            OP_COMPOUND_LETIDX => {
                let i = program[pc + 1] as usize;
                state.regs[i] += 1;
                pc += 2;
            }
            // Ingredient 3 alone: same-slot read-modify-write written out,
            // index is a `let`-bound local, no compound operator.
            OP_RMW_LETIDX => {
                let i = program[pc + 1] as usize;
                state.regs[i] = state.regs[i] + 1;
                pc += 2;
            }
            // Ingredient 2 alone: computed index in lvalue position, plain
            // assignment, no read of the slot being written.
            OP_PLAIN_COMPUTEDIDX => {
                state.regs[program[pc + 1] as usize] = 1;
                pc += 2;
            }
            OP_COMPOUND_COMPUTEDIDX => {
                state.regs[program[pc + 1] as usize] += 1;
                pc += 2;
            }
            _ => break,
        }
    }
    0
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_compound_assign(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_compound_assign(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// A grouping module. It is no longer a requirement: every item `#[jit_interp]`
/// emits is now suffixed with the annotated function's name, so machines over
/// DIFFERENT state types co-reside (`jit_interp_two_machines_one_module.rs`).
/// Two machines over the SAME state type still collide on `impl JitState`.
mod float_control {
    use super::Bytecode;
    use majit_metainterp::virt_array::VirtArray;
    use majit_metainterp::{Assembler, JitCode, JitDriver};

    const OP_FADD_ASSIGN: u8 = 1;
    const OP_FREM_ASSIGN: u8 = 2;

    /// Float compound assignments supported by `opcode_for_assign_binop_f`
    /// lower normally; remainder is deliberately unsupported and supplies
    /// the positive control for the degraded-arm recorder.
    // Named as the `jit_interp` state type; the macro lowers the interpreter body
    // into jitcode and never runs it, so nothing constructs the struct.
    #[expect(dead_code)]
    struct FloatCompoundState {
        fregs: VirtArray<f64>,
    }

    #[majit_macros::jit_interp(
    state = FloatCompoundState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        fregs: [float; virt],
    },
)]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_float_compound(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatCompoundState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatCompoundState {
            fregs: VirtArray::filled(0.0f64, 4),
        };
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
                OP_FADD_ASSIGN => {
                    let i = program[pc + 1] as usize;
                    state.fregs[i] += 1.0;
                    pc += 2;
                }
                OP_FREM_ASSIGN => {
                    let i = program[pc + 1] as usize;
                    state.fregs[i] %= 1.0;
                    pc += 2;
                }
                _ => break,
            }
        }
        0
    }

    pub(super) fn install_float() -> JitCode {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
        __prebuild_jitcode_liveness_dispatch_float_compound(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        __dispatch_jitcode_dispatch_float_compound(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture")
    }
}

use float_control::install_float;

fn recorded_arms() -> Vec<String> {
    majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|entry| entry.interp == "CompoundAssignState")
        .map(|entry| entry.arm.to_string())
        .collect()
}

/// Post-fix: every spelling lowers, including both compound-assign forms.
///
/// Before `lower_vable_array_update` existed, this same fixture recorded
/// `["OP_COMPOUND_COMPUTEDIDX", "OP_COMPOUND_LETIDX"]` — the two arms carrying
/// the compound operator, and only those. That measurement is what identified
/// the operator as the blocker and cleared the computed index and the
/// read-modify-write; the assertion below is its post-fix half.
#[test]
fn every_compound_assign_spelling_lowers() {
    let _ = install();
    let mut arms = recorded_arms();
    arms.sort();

    assert!(
        arms.is_empty(),
        "no arm in this fixture may degrade; recorded={arms:?}",
    );
}

/// Guard the guard: `every_compound_assign_spelling_lowers` asserts an empty
/// set, which a channel that records nothing at all would also satisfy. This
/// pins that the recorder is live in this binary by driving it with an arm
/// that genuinely cannot lower: float remainder assignment is not in
/// `opcode_for_assign_binop_f`, while float addition assignment must lower.
#[test]
fn the_recorder_is_live_in_this_binary() {
    let _ = install_float();
    let arms: Vec<String> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == "FloatCompoundState")
        .map(|e| e.arm.to_string())
        .collect();
    assert!(
        arms.iter().any(|a| a == "OP_FREM_ASSIGN"),
        "unsupported float remainder-assign must be recorded, \
         otherwise the empty set above asserts nothing; recorded={arms:?}",
    );
    assert!(
        !arms.iter().any(|a| a == "OP_FADD_ASSIGN"),
        "supported float addition-assign must lower; recorded={arms:?}",
    );
}
