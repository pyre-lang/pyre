//! Verifies that a degraded-stub abort resumes at the aborting opcode's own
//! boundary.
//!
//! The subject uses a multi-byte degraded opcode after a shared prologue
//! advance, so resuming at the following byte would decode an operand as a new
//! opcode and produce a wrong result.

use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

/// `regs[0] -= 1`, advance one byte.
const OP_DEC: u8 = 1;
/// `[OP_BUMP, k]` — `regs[1] += k` through a helper the lowerer cannot express,
/// so this arm degrades to an abort stub. Two bytes wide, and its operand is
/// spelled `1` so a late resume decodes it as [`OP_DEC`].
const OP_BUMP: u8 = 2;
/// `[OP_BACK, target]` — jump to `target` while `regs[0] > 0`, else fall past.
const OP_BACK: u8 = 3;
/// Bare `break` arm, the spelling that classifies as `ArmPattern::Halt`.
const OP_END: u8 = 4;

/// Mutates the register array through its raw base pointer, a shape the lowerer
/// cannot express and therefore emits as a degraded stub.
///
/// # Safety
/// `base` points to the two-element register array owned by this fixture.
fn bump_via_ptr(base: usize, idx: i64, by: i64) {
    let p = base as *mut i64;
    unsafe {
        *p.add(idx as usize) += by;
    }
}

struct StubResumeState {
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = StubResumeState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_stub_resume(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<StubResumeState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = StubResumeState {
        regs: VirtArray::filled(0i64, 2),
    };
    state.regs[0] = n;
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        // The shared prologue advance. It is what makes the abort resume
        // position wrong: `i0` is advanced HERE, before the arm runs, so a
        // stub abort inside the arm leaves `i0` one past an opcode that
        // applied nothing. Each arm then advances again over its own operands.
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_DEC => {
                state.regs[0] -= 1;
            }
            OP_BUMP => {
                let k = program[pc] as i64;
                pc += 1;
                bump_via_ptr(state.regs.as_mut_ptr() as usize, 1, k);
            }
            OP_BACK => {
                let target = program[pc] as usize;
                pc += 1;
                if state.regs[0] > 0 {
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
    state.regs[1]
}

/// `[OP_BUMP, 1, OP_DEC, OP_BACK, 0, OP_END]`.
///
/// Each iteration bumps `regs[1]` by one and decrements `regs[0]` by one, so a
/// clean run of `n` iterations answers exactly `n`.
fn program() -> Vec<u8> {
    vec![OP_BUMP, 1, OP_DEC, OP_BACK, 0, OP_END]
}

/// What the same program computes with no JIT tier involved at all.
fn interpret(n: i64) -> i64 {
    let (mut a, mut b) = (n, 0i64);
    while a > 0 {
        b += 1;
        a -= 1;
    }
    b
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_stub_resume(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_stub_resume(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// The fixture is only meaningful while `OP_BUMP` actually degrades.
///
/// If a later lowering change teaches the macro to express the helper call,
/// this arm stops being a stub, the abort under test never happens, and the
/// resume assertion below silently stops testing anything. Fail here instead,
/// loudly, so the fixture cannot rot into an oracle that cannot fail.
#[test]
fn op_bump_is_a_degraded_arm() {
    let _ = install();
    let arms: Vec<String> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == "StubResumeState")
        .map(|e| e.arm.to_string())
        .collect();
    assert!(
        arms.iter().any(|a| a == "OP_BUMP"),
        "OP_BUMP must lower to an abort stub for this fixture to exercise the \
         degraded-stub resume path at all; recorded={arms:?}"
    );
}

/// The opcode the stub aborted in must run exactly once, not zero times.
///
/// A threshold low enough to trace, over enough iterations to reach the abort
/// and continue past it. The answer is compared against the JIT-free
/// computation rather than a literal, so the fixture states the property
/// (`the JIT must not change the answer`) rather than a constant.
#[test]
fn a_degraded_stub_abort_reruns_its_own_opcode() {
    for n in [12i64, 20i64] {
        let got = dispatch_stub_resume(&program(), 4, n);
        assert_eq!(
            got,
            interpret(n),
            "n={n}: a degraded-stub abort resumed one byte past OP_BUMP, so the \
             bump was skipped and its operand byte was decoded as OP_DEC"
        );
    }
}
