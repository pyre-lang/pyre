//! A `#[jit_interp]` portal with no opcode `match`.
//!
//! `marked.py` is a `while i < len(s)` around `jit_merge_point` and the
//! body; `warmspot.rewrite_can_enter_jit` inserts the header tick when
//! the source has no `can_enter_jit`. The three-opcode PROGRAM tax existed
//! only because the macro refused a loop without a dispatch `match`.

use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;
use parking_lot::Mutex;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 400;

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
}

static REC: Recorder = Recorder::new();
/// Serializes every test that touches `REC`, the same way `PROBE_LOCK`
/// does in `shortcircuit.rs`. `cargo test` runs these `#[test]`s on
/// multiple threads by default.
static REC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct CountState {
    acc: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = CountState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn matchless_count(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<CountState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        REC.compiles.fetch_add(1, Ordering::Relaxed);
        *REC.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = CountState { acc: 0, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.acc += 1i64;
        state.pos += 1i64;
    }
    state.acc
}

#[test]
fn a_matchless_portal_compiles_the_while_body() {
    let _guard = REC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    REC.compiles.store(0, Ordering::Relaxed);
    REC.body.lock().clear();
    let cold = matchless_count(&PROGRAM, u32::MAX, N);
    let warm = matchless_count(&PROGRAM, 3, N);
    assert_eq!(cold, N, "the interpreter should count once per step");
    assert_eq!(
        warm, cold,
        "the compiled matchless loop disagreed with the interpreter",
    );
    assert!(
        REC.compiles.load(Ordering::Relaxed) > 0,
        "the matchless portal never compiled a loop",
    );
    let body = REC.body.lock().clone();
    assert!(
        body.contains(&OpCode::IntAdd),
        "the compiled loop carries no increment, so it runs none of the \
         while body: {body:#?}",
    );
    majit_metainterp::assert_no_degraded_dispatch_arms("CountState");
}

/// A post-merge local `match` is not an opcode dispatch. Selecting it as
/// one would skip matchless lowering and compile a loop with no body.
struct LocalMatchState {
    acc: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = LocalMatchState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn matchless_with_local_match(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<LocalMatchState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        REC.compiles.fetch_add(1, Ordering::Relaxed);
        *REC.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = LocalMatchState { acc: 0, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        match state.acc {
            0 => state.acc += 2i64,
            _ => state.acc += 1i64,
        }
        state.pos += 1i64;
    }
    state.acc
}

#[test]
fn a_matchless_portal_with_a_local_match_still_runs_the_body() {
    let _guard = REC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    REC.compiles.store(0, Ordering::Relaxed);
    REC.body.lock().clear();
    let cold = matchless_with_local_match(&PROGRAM, u32::MAX, N);
    let warm = matchless_with_local_match(&PROGRAM, 3, N);
    assert_eq!(
        warm, cold,
        "a post-merge local match was treated as opcode dispatch"
    );
    assert!(
        REC.compiles.load(Ordering::Relaxed) > 0,
        "the matchless portal with a local match never compiled a loop",
    );
    let body = REC.body.lock().clone();
    assert!(
        body.contains(&OpCode::IntAdd),
        "the compiled loop carries no increment: {body:#?}",
    );
}

/// A `continue` in the matchless body must re-check the while condition.
/// Jumping to the merge point would skip that test and keep iterating
/// after `pos` has reached `n`.
struct ContinueState {
    acc: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = ContinueState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { acc: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn matchless_with_continue(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<ContinueState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        REC.compiles.fetch_add(1, Ordering::Relaxed);
        *REC.body.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = ContinueState { acc: 0, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        if state.pos % 2i64 == 0i64 {
            state.pos += 1i64;
            continue;
        }
        state.acc += 1i64;
        state.pos += 1i64;
    }
    state.acc
}

#[test]
fn a_matchless_portal_continue_rechecks_the_while_condition() {
    let _guard = REC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    REC.compiles.store(0, Ordering::Relaxed);
    REC.body.lock().clear();
    let cold = matchless_with_continue(&PROGRAM, u32::MAX, N);
    let warm = matchless_with_continue(&PROGRAM, 3, N);
    assert_eq!(
        warm, cold,
        "continue skipped the while condition after warmup"
    );
    assert!(
        REC.compiles.load(Ordering::Relaxed) > 0,
        "the matchless portal with continue never compiled a loop",
    );
}
