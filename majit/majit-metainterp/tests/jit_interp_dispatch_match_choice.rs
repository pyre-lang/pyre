//! Which `match` in a portal function is the dispatch.
//!
//! `find_dispatch_match` chose the one with the most arms, over every match
//! anywhere in the function body, on the reasoning that a dispatch has many
//! opcode arms and a setup match only a few. Arm count is not a property of
//! being the dispatch. A five-arm `match` beside a three-opcode dispatch takes
//! its place, and then two things happen at once: `classify_arms` reads that
//! match's arms as the opcodes, and the two pre-dispatch walkers — which find
//! their loop by asking which one holds the dispatch match — find no loop and
//! lower nothing. The portal compiles a loop that runs none of the
//! interpreter, and answers with it.
//!
//! Measured here before the fix, on `dispatch_setup_before`:
//! cold 1200, warm 24. 24 is the eight interpreted iterations that reach the
//! threshold; every iteration after that ran the compiled loop, which does
//! nothing. It compiles clean — no degraded arm, no diagnostic, no warning.
//!
//! The arm census is the direct reading: two opcodes means `arms == 2`, and
//! the five-arm setup match would make it 4. The answer is the consequence.
//! Both are asserted, because a machine that stopped lowering altogether
//! would agree with the interpreter by being it.
//!
//! Every machine below is the same interpreter. Only where the setup match
//! sits changes: nowhere, before the loop, inside the loop ahead of the merge
//! point, and after the loop.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;

pub type Bytecode = [u8];

const OP_WORK: u8 = 1;
const OP_TICK: u8 = 2;
const PROGRAM: [u8; 2] = [OP_WORK, OP_TICK];

const TICKS: i64 = 400;
const WORK: i64 = 3;

/// One recording slot per machine: these share a process, and the compile
/// hook belongs to a driver rather than to a test.
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
             and every claim about the trace is vacuous",
        );
        self.body.lock().clone()
    }
}

static BARE: Recorder = Recorder::new();
static BEFORE: Recorder = Recorder::new();
static IN_LOOP: Recorder = Recorder::new();
static AFTER: Recorder = Recorder::new();

struct BareState {
    acc: i64,
    ticks: i64,
}
struct BeforeState {
    acc: i64,
    ticks: i64,
}
struct InLoopState {
    acc: i64,
    ticks: i64,
}
struct AfterState {
    acc: i64,
    ticks: i64,
}

/// The control: no competing match at all.
#[majit_macros::jit_interp(
    state = BareState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_bare(program: &Bytecode, threshold: u32, ticks: i64, sel: i64) -> i64 {
    let mut driver: JitDriver<BareState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        BARE.compiles.fetch_add(1, Ordering::Relaxed);
        *BARE.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = BareState { acc: 0, ticks };
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

/// The setup match ahead of the portal loop. This is the shape that broke.
#[majit_macros::jit_interp(
    state = BeforeState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_setup_before(program: &Bytecode, threshold: u32, ticks: i64, sel: i64) -> i64 {
    let mut driver: JitDriver<BeforeState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        BEFORE.compiles.fetch_add(1, Ordering::Relaxed);
        *BEFORE.body.lock() = opcodes.to_vec();
    });
    let label = match sel {
        0i64 => "zero",
        1i64 => "one",
        2i64 => "two",
        3i64 => "three",
        _ => "many",
    };
    let _ = label;
    let mut pc: usize = 0;
    let mut state = BeforeState { acc: 0, ticks };
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

/// Inside the loop, ahead of the merge point — the region
/// `bind_pre_merge_point_stmts` walks.
#[majit_macros::jit_interp(
    state = InLoopState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_setup_in_loop(program: &Bytecode, threshold: u32, ticks: i64, sel: i64) -> i64 {
    let mut driver: JitDriver<InLoopState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        IN_LOOP.compiles.fetch_add(1, Ordering::Relaxed);
        *IN_LOOP.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = InLoopState { acc: 0, ticks };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        let label = match sel {
            0i64 => "zero",
            1i64 => "one",
            2i64 => "two",
            3i64 => "three",
            _ => "many",
        };
        let _ = label;
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

/// Behind the loop: the finder walked the whole function body, so a match
/// after the portal competed too.
#[majit_macros::jit_interp(
    state = AfterState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_setup_after(program: &Bytecode, threshold: u32, ticks: i64, sel: i64) -> i64 {
    let mut driver: JitDriver<AfterState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        AFTER.compiles.fetch_add(1, Ordering::Relaxed);
        *AFTER.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = AfterState { acc: 0, ticks };
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
    let label = match sel {
        0i64 => "zero",
        1i64 => "one",
        2i64 => "two",
        3i64 => "three",
        _ => "many",
    };
    let _ = label;
    state.acc
}

/// What the interpreter alone computes: one `WORK` per tick.
const EXPECTED: i64 = TICKS * WORK;

fn arms_of(interp: &str) -> usize {
    majit_metainterp::dispatch_arm_census()
        .into_iter()
        .find(|e| e.interp == interp)
        .unwrap_or_else(|| panic!("no dispatch-arm census for `{interp}`: no portal was installed"))
        .arms
}

fn check(name: &'static str, cold: i64, warm: i64, rec: &Recorder) {
    assert_eq!(cold, EXPECTED, "{name}: the cold arm states the answer");
    assert_eq!(
        warm, cold,
        "{name}: the compiled loop disagreed with the interpreter",
    );
    let body = rec.compiled_loop();
    assert!(
        body.contains(&OpCode::IntAdd),
        "{name}: the compiled loop carries no arithmetic, so it runs none of \
         the interpreter: {body:#?}",
    );
    assert_eq!(
        arms_of(name),
        2,
        "{name}: the portal lowered a different match's arms as its opcodes; \
         this machine has two, and its five-arm setup match has four besides \
         the default",
    );
    majit_metainterp::assert_no_degraded_dispatch_arms(name);
}

#[test]
fn a_setup_match_does_not_displace_the_dispatch() {
    // The control first: if it fails, the three below measure the fixture.
    check(
        "BareState",
        dispatch_bare(&PROGRAM, u32::MAX, TICKS, 0),
        dispatch_bare(&PROGRAM, 8, TICKS, 0),
        &BARE,
    );
    check(
        "BeforeState",
        dispatch_setup_before(&PROGRAM, u32::MAX, TICKS, 0),
        dispatch_setup_before(&PROGRAM, 8, TICKS, 0),
        &BEFORE,
    );
    check(
        "InLoopState",
        dispatch_setup_in_loop(&PROGRAM, u32::MAX, TICKS, 0),
        dispatch_setup_in_loop(&PROGRAM, 8, TICKS, 0),
        &IN_LOOP,
    );
    check(
        "AfterState",
        dispatch_setup_after(&PROGRAM, u32::MAX, TICKS, 0),
        dispatch_setup_after(&PROGRAM, 8, TICKS, 0),
        &AFTER,
    );
}
