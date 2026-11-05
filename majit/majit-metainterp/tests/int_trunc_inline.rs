//! `support.py` `_ll_2_int_mod` / `_ll_2_int_floordiv` are inlined into the
//! trace (`inline_calls_to`): `int.py_mod` / `int.py_div` plus the
//! truncation adjustment. A nonnegative power-of-two divisor folds the
//! call away (`rewrite.py` `_optimize_CALL_INT_PY_MOD` /
//! `_optimize_CALL_INT_PY_DIV`). A varying divisor keeps the oopspec call
//! and still truncates toward zero.

use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;
use parking_lot::Mutex;

pub type Bytecode = [u8];

const PROGRAM: [u8; 1] = [0];
const N: i64 = 40;
const THRESHOLD: u32 = 3;

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

fn is_call(op: &OpCode) -> bool {
    format!("{op:?}").starts_with("Call")
}

fn install_body(rec: &Recorder, opcodes: &[OpCode]) {
    rec.compiles.fetch_add(1, Ordering::Relaxed);
    *rec.body.lock() = opcodes.to_vec();
}

struct ModPowState {
    sum: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = ModPowState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { sum: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn mod_pow2(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<ModPowState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        install_body(&MOD_POW, opcodes);
    });
    let mut pc: usize = 0;
    let mut state = ModPowState { sum: 0, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.sum += state.pos % 2;
        state.pos += 1i64;
    }
    state.sum
}

static MOD_POW: Recorder = Recorder::new();
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct DivPowState {
    sum: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = DivPowState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { sum: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn div_pow2(program: &Bytecode, threshold: u32, n: i64) -> i64 {
    let mut driver: JitDriver<DivPowState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        install_body(&DIV_POW, opcodes);
    });
    let mut pc: usize = 0;
    let mut state = DivPowState { sum: 0, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.sum += state.pos / 4;
        state.pos += 1i64;
    }
    state.sum
}

static DIV_POW: Recorder = Recorder::new();

struct ModVarState {
    sum: i64,
    a: i64,
    b: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = ModVarState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { sum: int, a: int, b: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn mod_var(program: &Bytecode, threshold: u32, n: i64, a: i64, b: i64) -> i64 {
    let mut driver: JitDriver<ModVarState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        install_body(&MOD_VAR, opcodes);
    });
    let mut pc: usize = 0;
    let mut state = ModVarState {
        sum: 0,
        a,
        b,
        pos: 0,
        n,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        // `b` advances, so the divisor is not a constant and
        // `_optimize_CALL_INT_PY_MOD` leaves the oopspec call in place.
        state.sum += state.a % state.b;
        state.b += 1i64;
        state.pos += 1i64;
    }
    state.sum
}

static MOD_VAR: Recorder = Recorder::new();

struct DivVarState {
    sum: i64,
    a: i64,
    b: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = DivVarState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { sum: int, a: int, b: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn div_var(program: &Bytecode, threshold: u32, n: i64, a: i64, b: i64) -> i64 {
    let mut driver: JitDriver<DivVarState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        install_body(&DIV_VAR, opcodes);
    });
    let mut pc: usize = 0;
    let mut state = DivVarState {
        sum: 0,
        a,
        b,
        pos: 0,
        n,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.sum += state.a / state.b;
        state.b += 1i64;
        state.pos += 1i64;
    }
    state.sum
}

static DIV_VAR: Recorder = Recorder::new();

struct ModZeroState {
    a: i64,
    b: i64,
    pos: i64,
    n: i64,
}

#[majit_macros::jit_interp(
    state = ModZeroState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { a: int, b: int, pos: int, n: int },
)]
#[allow(unused_assignments, unused_variables)]
fn mod_zero(program: &Bytecode, threshold: u32, n: i64, a: i64, b: i64) -> i64 {
    let mut driver: JitDriver<ModZeroState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = ModZeroState { a, b, pos: 0, n };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while state.pos < state.n {
        can_enter_jit!(driver, 0usize, &mut state, program, || {});
        jit_merge_point!(driver, program, pc; state);
        state.a %= state.b;
        state.pos += 1i64;
    }
    state.a
}

fn rust_mod_sum(a: i64, mut b: i64, n: i64) -> i64 {
    let mut sum = 0i64;
    for _ in 0..n {
        sum += a % b;
        b += 1;
    }
    sum
}

fn rust_div_sum(a: i64, mut b: i64, n: i64) -> i64 {
    let mut sum = 0i64;
    for _ in 0..n {
        sum += a / b;
        b += 1;
    }
    sum
}

#[test]
fn nonnegative_power_of_two_mod_and_div_leave_no_call() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    MOD_POW.compiles.store(0, Ordering::Relaxed);
    DIV_POW.compiles.store(0, Ordering::Relaxed);
    let mod_cold = mod_pow2(&PROGRAM, u32::MAX, N);
    let mod_warm = mod_pow2(&PROGRAM, THRESHOLD, N);
    let div_cold = div_pow2(&PROGRAM, u32::MAX, N);
    let div_warm = div_pow2(&PROGRAM, THRESHOLD, N);
    let mod_expect: i64 = (0..N).map(|i| i % 2).sum();
    let div_expect: i64 = (0..N).map(|i| i / 4).sum();
    assert_eq!(mod_cold, mod_expect);
    assert_eq!(mod_warm, mod_cold);
    assert_eq!(div_cold, div_expect);
    assert_eq!(div_warm, div_cold);
    assert!(MOD_POW.compiles.load(Ordering::Relaxed) > 0);
    assert!(DIV_POW.compiles.load(Ordering::Relaxed) > 0);
    let mod_body = MOD_POW.body.lock().clone();
    let div_body = DIV_POW.body.lock().clone();
    assert!(
        mod_body.contains(&OpCode::IntAnd) && mod_body.iter().all(|op| !is_call(op)),
        "a % 2 on a nonnegative int must fold to int_and with no call: {mod_body:?}"
    );
    assert!(
        div_body.contains(&OpCode::IntRshift) && div_body.iter().all(|op| !is_call(op)),
        "a / 4 on a nonnegative int must fold to int_rshift with no call: {div_body:?}"
    );
}

#[test]
fn variable_divisor_keeps_py_call_and_truncates_toward_zero() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    MOD_VAR.compiles.store(0, Ordering::Relaxed);
    DIV_VAR.compiles.store(0, Ordering::Relaxed);
    // First step is the pinned pair: -7 % 3 == -1, -7 / 2 == -3.
    assert_eq!(-7 % 3, -1);
    assert_eq!(-7 / 2, -3);
    let mod_expect = rust_mod_sum(-7, 3, N);
    let div_expect = rust_div_sum(-7, 2, N);
    let mod_cold = mod_var(&PROGRAM, u32::MAX, N, -7, 3);
    let mod_warm = mod_var(&PROGRAM, THRESHOLD, N, -7, 3);
    let div_cold = div_var(&PROGRAM, u32::MAX, N, -7, 2);
    let div_warm = div_var(&PROGRAM, THRESHOLD, N, -7, 2);
    assert_eq!(mod_cold, mod_expect);
    assert_eq!(
        mod_warm, mod_cold,
        "traced a % b disagreed with C truncation"
    );
    assert_eq!(div_cold, div_expect);
    assert_eq!(
        div_warm, div_cold,
        "traced a / b disagreed with C truncation"
    );
    assert!(MOD_VAR.compiles.load(Ordering::Relaxed) > 0);
    assert!(DIV_VAR.compiles.load(Ordering::Relaxed) > 0);
    let mod_body = MOD_VAR.body.lock().clone();
    let div_body = DIV_VAR.body.lock().clone();
    // OptPure demotes an unfolded elidable call to `CallI`; the oopspec
    // stayed on the descr long enough for the power-of-two case to fold.
    // The adjustment (`int_neg` / `int_sub`) is what distinguishes the
    // inlined `_ll_2_int_*` body from a residual helper call.
    assert!(
        mod_body.iter().any(is_call)
            && mod_body.contains(&OpCode::IntNeg)
            && mod_body.contains(&OpCode::IntSub),
        "a varying divisor must keep the int.py_mod call plus the truncation adjustment: {mod_body:?}"
    );
    assert!(
        div_body.iter().any(is_call)
            && div_body.contains(&OpCode::IntAdd)
            && div_body.contains(&OpCode::IntNe),
        "a varying divisor must keep the int.py_div call plus the truncation adjustment: {div_body:?}"
    );
}

#[test]
fn zero_divisor_still_panics() {
    // Rust `%` / `/` panic on a zero divisor. The inlined body reaches the
    // same panic inside `ll_int_py_mod` / `ll_int_py_div` (`wrapping_rem` /
    // `wrapping_div`), which is what `blackhole::_ll_2_int_mod` /
    // `_ll_2_int_floordiv` do. Those helpers are `extern "C"`, so the panic
    // aborts rather than unwinding; the interpreter `%` is the catchable one.
    let interp = std::panic::catch_unwind(|| mod_zero(&PROGRAM, u32::MAX, 1, 7, 0));
    assert!(interp.is_err(), "interpreter `%` by zero must still panic");
}
