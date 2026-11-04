//! Verifies that an arm with multiple lowering blockers records every refusal
//! in encounter order and that refusal classification handles each member.

use majit_metainterp::virt_array::VirtArray;
use majit_metainterp::{
    Assembler, JitCode, JitDriver, REFUSAL_SEPARATOR, RefusalKind, refusal_kind, refusal_kinds,
};

pub type Bytecode = [u8];

/// `regs[0] -= 1`, advance.
const OP_DEC: u8 = 1;
/// Back edge: jump to 0 while `regs[0] != 0`, else fall past.
const OP_BACK: u8 = 2;
/// The arm under test. Two blockers, deliberately ordered so the FIRST one's
/// family sits LATER in `refusal_kind_of_one`'s `contains` chain than the
/// SECOND one's.
const OP_REALLOC_THEN_BREAK: u8 = 3;

struct AccumState {
    regs: VirtArray<i64>,
}

#[majit_macros::jit_interp(
    state = AccumState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_accum(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<AccumState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = AccumState {
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
            OP_REALLOC_THEN_BREAK => {
                // Blocker 1 — whole-array reallocation has no lowering.
                // `UnlowerableStmt`, the chain's 4th test.
                state.regs = VirtArray::filled(0i64, 2);
                // Blocker 2 — an enclosed `break` cannot be lowered in place.
                // `EnclosedBreakContinue`, the chain's 2nd test.
                if state.regs[0] == 0 {
                    break;
                }
                // Blocker 3 — `pc` is green; a mid-arm write cannot be
                // carried back out of this lowering path.
                pc += 1;
            }
            _ => break,
        }
    }
    state.regs[1]
}

/// `[OP_DEC, OP_BACK, OP_REALLOC_THEN_BREAK, n]` — the trailing byte seeds
/// `regs[0]`, so the loop runs `n` times before reaching the subject arm.
fn program_for(n: u8) -> Vec<u8> {
    vec![OP_DEC, OP_BACK, OP_REALLOC_THEN_BREAK, n]
}

fn install() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![0]);
    __prebuild_jitcode_liveness_dispatch_accum(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_accum(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture")
}

/// The subject arm's recorded `reason`, taken live from the registry rather
/// than copied into a literal, so it cannot go stale against the macro.
fn subject_reason() -> String {
    let _ = install();
    majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .find(|e| e.interp == "AccumState" && e.arm == "OP_REALLOC_THEN_BREAK")
        .map(|e| e.reason.to_string())
        .expect("OP_REALLOC_THEN_BREAK must be recorded as degraded")
}

/// The recorded reason contains both blockers in the order lowering encounters
/// them.
#[test]
fn both_blockers_are_reported() {
    let reason = subject_reason();
    let kinds = refusal_kinds(&reason);
    assert_eq!(
        kinds,
        vec![
            RefusalKind::UnlowerableStmt,
            RefusalKind::EnclosedBreakContinue,
            RefusalKind::GreenWriteback,
            RefusalKind::EnclosedBreakContinue
        ],
        "the arm's blockers must all be reported, reallocation first \
         (statement order), then the enclosing `if`, the green `pc` write, \
         and the `break` inside the `if`; reason={reason:?}"
    );
    assert!(
        reason.contains("state.regs =") && reason.contains("filled") && reason.contains("break"),
        "each member must still name its own offending statement; reason={reason:?}"
    );
}

/// Property 2: `refusal_kind` reports the OUTERMOST refusal, which it can only
/// do by splitting first.
///
/// The negative control is executable rather than asserted: collapsing the
/// separator turns the accumulated reason into a single segment, which is
/// exactly what an un-split classifier would see. If the two answers agreed,
/// this subject could not discriminate and the test would be worthless — so the
/// disagreement is asserted too.
#[test]
fn classifying_the_head_requires_splitting_first() {
    let reason = subject_reason();
    let unsplit = reason.replace(REFUSAL_SEPARATOR, " ");
    assert_eq!(
        refusal_kind(&unsplit),
        RefusalKind::EnclosedBreakContinue,
        "control: seen as one segment, the ordered `contains` chain reaches \
         `encloses a `break`` (2nd test) before `cannot express` (4th), so it \
         answers with the arm's SECOND blocker; unsplit={unsplit:?}"
    );
    assert_eq!(
        refusal_kind(&reason),
        RefusalKind::UnlowerableStmt,
        "subject: split first, the head is the reallocation — the blocker \
         lowering actually stopped at; reason={reason:?}"
    );
    assert_ne!(
        refusal_kind(&unsplit),
        refusal_kind(&reason),
        "this subject exists to make the two disagree; if they ever agree it \
         has stopped discriminating and must be re-chosen, not relaxed"
    );
}

/// The degraded arm runs interpreted. Liveness only — the refusal-order and
/// classification assertions live in this file's other tests; this one is not
/// an oracle for the refusal and must not be cited as one.
#[test]
fn the_degraded_arm_runs_without_panicking() {
    let _ = dispatch_accum(&program_for(3), 4);
}
