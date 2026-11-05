//! Where a dispatch match's default arm sends the trace.
//!
//! The unmatched-opcode edge used to be one thing: the label bound at the
//! portal function's trailing return. That is right for `_ => break`, which is
//! what almost every machine in this corpus writes, and it was applied to the
//! other spellings too. `_ => {}` means "fall out of the match and run the next
//! iteration" and `_ => { .. }` means "run this"; both got the return instead,
//! so the walk reported a finished frame, `take_single_pass_finish` broke the
//! native loop, and the portal returned a partial answer.
//!
//! Nothing reported it. `degraded_dispatch_arms()` was empty and the arm
//! census was intact, because from the lowerer's side nothing had refused —
//! the arm was simply never emitted. Measured here before the fix:
//!
//! | default arm        | cold  | warm | compiled body     |
//! |--------------------|-------|------|-------------------|
//! | `_ => {}`          |  1200 |   27 | `[IntAdd, Finish]` |
//! | `_ => { acc += N }`| 41200 |  827 | `[IntAdd, Finish]` |
//!
//! So the answer is the subject, and the loop shape is what keeps the answer
//! honest: an interpreter that never entered the JIT agrees with itself too.
//!
//! Each machine puts an opcode no arm matches into its program. Without one
//! the default edge is never walked and none of this can appear.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const OP_WORK: u8 = 1;
const OP_TICK: u8 = 2;
/// Matched by no arm in any machine below.
const OP_UNKNOWN: u8 = 9;

const TICKS: i64 = 400;
const WORK: i64 = 3;
const DEFAULT_WORK: i64 = 100;

/// One recording slot per machine: these run in the same process and the
/// compile hook is global to a driver, not to a test.
struct Recorder {
    compiles: AtomicUsize,
    body: Mutex<Vec<OpCode>>,
}

impl Recorder {
    const fn new() -> Self {
        Self {
            compiles: AtomicUsize::new(0),
            body: Mutex::new(Vec::new()),
        }
    }
    fn compiled_loop(&self) -> Vec<OpCode> {
        assert!(
            self.compiles.load(Ordering::Relaxed) > 0,
            "nothing compiled, so the warm answer was the interpreter's alone \
             and every claim below is vacuous",
        );
        self.body.lock().clone()
    }
}

static SKIP: Recorder = Recorder::new();
static WORKED: Recorder = Recorder::new();
static EXITED: Recorder = Recorder::new();
static SWITCHED: Recorder = Recorder::new();

struct SkipState {
    acc: i64,
    ticks: i64,
}
struct WorkState {
    acc: i64,
    ticks: i64,
}
struct ExitState {
    acc: i64,
    ticks: i64,
}
struct SwitchState {
    acc: i64,
    ticks: i64,
}

/// `_ => {}` — go round again.
#[majit_macros::jit_interp(
    state = SkipState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_skip(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<SkipState> = JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        SKIP.compiles.fetch_add(1, Ordering::Relaxed);
        *SKIP.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = SkipState { acc: 0, ticks };
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
            OP_WORK => {
                state.acc += WORK;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => {}
        }
    }
    state.acc
}

/// `_ => { .. }` — an arm like any other, with no opcode test in front of it.
#[majit_macros::jit_interp(
    state = WorkState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_work(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<WorkState> = JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        WORKED.compiles.fetch_add(1, Ordering::Relaxed);
        *WORKED.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = WorkState { acc: 0, ticks };
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
            OP_WORK => {
                state.acc += WORK;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => {
                state.acc += DEFAULT_WORK;
            }
        }
    }
    state.acc
}

/// The control: `_ => break`, the spelling 55 of this corpus's default arms
/// use. It keeps the trailing-return edge it always had, and its program has
/// no unmatched opcode — reaching one would end the loop on the first pass and
/// there would be nothing hot to compile.
#[majit_macros::jit_interp(
    state = ExitState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_exit(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<ExitState> = JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        EXITED.compiles.fetch_add(1, Ordering::Relaxed);
        *EXITED.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = ExitState { acc: 0, ticks };
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
            OP_WORK => {
                state.acc += WORK;
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

/// The same default-arm body under `switch_dispatch`, where the switch's own
/// default is what reaches it rather than a chain of missed guards. The corpus
/// has exactly one `switch_dispatch` machine and its default arm is
/// `_ => break`, so this combination had no coverage at all.
#[majit_macros::jit_interp(
    state = SwitchState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
    switch_dispatch = true,
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_switch(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<SwitchState> = JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        SWITCHED.compiles.fetch_add(1, Ordering::Relaxed);
        *SWITCHED.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = SwitchState { acc: 0, ticks };
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
            OP_WORK => {
                state.acc += WORK;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            _ => {
                state.acc += DEFAULT_WORK;
            }
        }
    }
    state.acc
}

fn closes_a_loop(body: &[OpCode]) -> bool {
    body.contains(&OpCode::Jump) && !body.contains(&OpCode::Finish)
}

#[test]
fn an_empty_default_arm_goes_round_again() {
    let program: &Bytecode = &[OP_WORK, OP_UNKNOWN, OP_TICK];
    let cold = dispatch_skip(program, u32::MAX, TICKS);
    assert_eq!(cold, TICKS * WORK, "the cold arm states the answer");

    let warm = dispatch_skip(program, 8, TICKS);
    assert_eq!(
        warm, cold,
        "the portal returned early: a warm answer of one compiled pass past \
         the threshold is the default edge taking the function's typed return",
    );
    let body = SKIP.compiled_loop();
    assert!(
        closes_a_loop(&body),
        "the default arm ended the trace instead of continuing it: {body:?}",
    );
}

#[test]
fn a_default_arm_with_a_body_runs_it() {
    let program: &Bytecode = &[OP_WORK, OP_UNKNOWN, OP_TICK];
    let cold = dispatch_work(program, u32::MAX, TICKS);
    assert_eq!(cold, TICKS * (WORK + DEFAULT_WORK));

    let warm = dispatch_work(program, 8, TICKS);
    assert_eq!(
        warm, cold,
        "the default arm's body is missing from the trace"
    );

    let body = WORKED.compiled_loop();
    assert!(closes_a_loop(&body), "{body:?}");
    // One `IntAdd` per arm that ran, plus `sub_const_canonicalize` rewriting
    // `ticks - 1` to `int_add(ticks, -1)`. With the default arm dropped the
    // peeled body carried one acc add, and the answer above could not tell
    // you which.
    let adds = body.iter().filter(|op| **op == OpCode::IntAdd).count();
    assert_eq!(
        adds, 6,
        "two acc adds and the rewritten tick sub per pass, peeled, so six: {body:?}",
    );
    // The default arm is emitted, so the census counts it: an arm is counted
    // exactly when the chain emits a body for it.
    let census = majit_metainterp::dispatch_arm_census();
    let counted = census
        .iter()
        .find(|e| e.interp == "WorkState")
        .unwrap_or_else(|| panic!("the portal must record its arm count; {census:?}"));
    assert_eq!(counted.arms, 3, "OP_WORK, OP_TICK and `_`; {census:?}");
    majit_metainterp::assert_no_degraded_dispatch_arms("WorkState");
}

/// The control. `_ => break` keeps the trailing-return edge, so this machine
/// must be untouched by the two above.
#[test]
fn a_breaking_default_arm_keeps_the_return_edge() {
    let program: &Bytecode = &[OP_WORK, OP_TICK];
    let cold = dispatch_exit(program, u32::MAX, TICKS);
    assert_eq!(cold, TICKS * WORK);
    assert_eq!(dispatch_exit(program, 8, TICKS), cold);

    let body = EXITED.compiled_loop();
    assert!(closes_a_loop(&body), "{body:?}");
    let census = majit_metainterp::dispatch_arm_census();
    let counted = census
        .iter()
        .find(|e| e.interp == "ExitState")
        .unwrap_or_else(|| panic!("the portal must record its arm count; {census:?}"));
    assert_eq!(
        counted.arms, 2,
        "OP_WORK and OP_TICK; a `break` default is still the exit edge and \
         emits no body, so it is still not counted; {census:?}",
    );
}

/// `switch_dispatch`: the switch's default target is the arm body, not the
/// portal's return.
#[test]
fn a_switch_dispatch_default_arm_runs_its_body_too() {
    let program: &Bytecode = &[OP_WORK, OP_UNKNOWN, OP_TICK];
    let cold = dispatch_switch(program, u32::MAX, TICKS);
    assert_eq!(cold, TICKS * (WORK + DEFAULT_WORK));
    assert_eq!(
        dispatch_switch(program, 8, TICKS),
        cold,
        "the switch's default edge took the portal's return",
    );
    let body = SWITCHED.compiled_loop();
    assert!(closes_a_loop(&body), "{body:?}");
    majit_metainterp::assert_no_degraded_dispatch_arms("SwitchState");
}
