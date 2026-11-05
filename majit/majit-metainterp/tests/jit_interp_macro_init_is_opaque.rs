//! A `let` whose initialiser is a macro invocation.
//!
//! `lower_local`'s last resort treats an initialiser it could not lower as a
//! compile-time constant: it emits the original `let x = <init>;` verbatim
//! into the surrounding `__builder` block and loads `x as i64` as a constant.
//! That contract holds only while `<init>` names nothing the generated scope
//! lacks, and the guard on it — `expr_touches_storage` →
//! `expr_references_unknown_local` — is what checks that: a bare lowercase
//! identifier is read as a user local and the fallback is refused.
//!
//! The walk had no `Expr::Macro` arm, so a macro invocation fell to its
//! catch-all and reported no reference at all. It cannot know that: the
//! tokens are opaque to it. `let bump = pick!(sel);` therefore passed the
//! guard and put `match sel { .. }` inside `__dispatch_jitcode_*`, whose
//! parameters are the assembler and the driver index:
//!
//! ```text
//! error[E0425]: cannot find value `sel` in this scope
//!   --> tests/...:19:26
//!    |
//! 19 |         let bump = pick!(sel);
//!    |                          ^^^ not found in this scope
//! ```
//!
//! pointing at a line where `sel` plainly is in scope. This file compiling at
//! all is the gate; the assertions below say the refusal is the fail-closed
//! one rather than a silent drop.

use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const OP_USES: u8 = 1;
const OP_PLAIN: u8 = 2;
const OP_TICK: u8 = 3;
const PROGRAM: [u8; 3] = [OP_USES, OP_PLAIN, OP_TICK];

const TICKS: i64 = 40;
const PLAIN_WORK: i64 = 3;

/// Opaque to the lowerer, and it names the portal's own parameter.
macro_rules! pick {
    ($sel:expr) => {
        match $sel {
            0i64 => 5i64,
            _ => 7i64,
        }
    };
}

struct PreState {
    acc: i64,
    ticks: i64,
}
struct ArmState {
    acc: i64,
    ticks: i64,
}

/// The macro-initialised `let` sits ahead of the merge point, the region
/// `bind_pre_merge_point_stmts` walks.
#[majit_macros::jit_interp(
    state = PreState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_pre(program: &Bytecode, threshold: u32, ticks: i64, sel: i64) -> i64 {
    let mut driver: JitDriver<PreState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = PreState { acc: 0, ticks };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        let bump = pick!(sel);
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_USES => {
                state.acc += bump;
            }
            OP_PLAIN => {
                state.acc += PLAIN_WORK;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.acc
}

/// The same initialiser inside a dispatch arm, where the refusal produces an
/// abort stub rather than a silent skip.
#[majit_macros::jit_interp(
    state = ArmState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_arm(program: &Bytecode, threshold: u32, ticks: i64, sel: i64) -> i64 {
    let mut driver: JitDriver<ArmState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = ArmState { acc: 0, ticks };
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
            OP_USES => {
                let n = pick!(sel);
                state.acc += n;
            }
            OP_PLAIN => {
                state.acc += PLAIN_WORK;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.acc
}

fn degraded_of(interp: &str) -> Vec<&'static str> {
    let mut arms: Vec<&'static str> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == interp)
        .map(|e| e.arm)
        .collect();
    arms.sort_unstable();
    arms.dedup();
    arms
}

fn arms_of(interp: &str) -> usize {
    majit_metainterp::dispatch_arm_census()
        .into_iter()
        .find(|e| e.interp == interp)
        .unwrap_or_else(|| panic!("no dispatch-arm census for `{interp}`: no portal was installed"))
        .arms
}

/// `sel = 0` picks 5, so each pass adds 5 + `PLAIN_WORK`.
fn expected() -> i64 {
    TICKS * (5 + PLAIN_WORK)
}

#[test]
fn a_macro_initialiser_is_refused_rather_than_spliced() {
    for (name, cold, warm) in [
        (
            "PreState",
            dispatch_pre(&PROGRAM, u32::MAX, TICKS, 0),
            dispatch_pre(&PROGRAM, 8, TICKS, 0),
        ),
        (
            "ArmState",
            dispatch_arm(&PROGRAM, u32::MAX, TICKS, 0),
            dispatch_arm(&PROGRAM, 8, TICKS, 0),
        ),
    ] {
        assert_eq!(cold, expected(), "{name}: the cold arm states the answer");
        assert_eq!(
            warm, cold,
            "{name}: the compiled loop disagreed with the interpreter",
        );
        assert_eq!(
            arms_of(name),
            3,
            "{name}: the portal did not lower this machine's three opcodes",
        );
        assert_eq!(
            degraded_of(name),
            vec!["OP_USES"],
            "{name}: the arm reading the macro-initialised binding must be \
             recorded as refused — a silent skip would leave the registry \
             empty and the answer would still be right, because the \
             interpreter computes it either way",
        );
    }
}
