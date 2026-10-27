//! Integration tests: trace recording -> optimization -> Cranelift compilation -> execution.
//!
//! These tests exercise the full majit pipeline from end to end.

use std::sync::Arc;

use majit_backend::{
    Backend, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout, JitCellToken,
};
use majit_backend_cranelift::{CraneliftBackend, force_token_to_dead_frame, jit_exc_raise};
use majit_ir::test_support::{RecordedTrace, Trace};
use majit_ir::{
    ArrayDescr, Descr, DescrRef, FieldDescr, GcRef, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef,
    Type, Value,
};

/// The recorder's inputarg identities, cloned as handles (same boxes).
fn inputargs_view(t: &RecordedTrace) -> Vec<InputArgRc> {
    t.inputargs.clone()
}

fn magic_numbers(m: i64) -> (u64, u32) {
    debug_assert!(m >= 3);
    debug_assert!(m & (m - 1) != 0, "m must not be a power of two");

    let m_u = m as u64;
    let mut i: u32 = 1;
    while (1u64 << (i + 1)) < m_u {
        i += 1;
    }

    let high_word_dividend = 1u64 << i;
    let mut quotient: u64 = 0;
    for bit in (0..64).rev() {
        let t = quotient + (1u64 << bit);
        let (_, high) = full_mul_u64(t, m_u);
        if high < high_word_dividend {
            quotient = t;
        }
    }

    (quotient + 1, i)
}

#[inline]
fn full_mul_u64(a: u64, b: u64) -> (u64, u64) {
    let result = (a as u128) * (b as u128);
    (result as u64, (result >> 64) as u64)
}

// Test helpers

/// Test helper — produces a real `ResumeGuardDescr` (PyPy
/// `compile.py ResumeGuardDescr` family) so that
/// `recovery_layout` storage routes through the meta-side slot.
/// The meta slot is the sole storage; earlier
/// test scaffolds returned a custom `TestFailDescr` that the
/// downcast couldn't reach.  `_index` parameter is preserved for
/// call-site clarity but discarded — every guard descr gets a
/// fresh global fail_index inside `make_resume_guard_descr_typed`.
fn make_descr(_index: u32) -> DescrRef {
    majit_backend::make_resume_guard_descr_typed(Vec::new())
}

use majit_ir::forwarding::bound_operand_from_opref as rb;

/// Stamp the guard's recovery list. `runner_test.py` uses
/// `op.setfailargs(...)` the same way `compile.py`
/// `ResumeGuardDescr.store_final_boxes` does after numbering.
fn set_guard_failargs(trace: &RecordedTrace, guard: OpRef, args: &[OpRef]) {
    let op = trace
        .ops
        .iter()
        .find(|op| op.pos().get() == guard)
        .expect("recorded guard");
    op.setfailargs(args.iter().copied().map(rb).collect());
}

// Test 1: Simple arithmetic (trace -> optimize -> compile -> execute)

#[test]
fn test_simple_arithmetic() {
    // Record: input(i) -> result = i + CONST_1 -> finish(result)
    // Execute with i=41, expect 42.
    let mut rec = Trace::new();
    let i0 = rec.record_input_arg(Type::Int);

    let const_one = OpRef::const_int(1);
    let result = rec.record_op(OpCode::IntAdd, &[i0, const_one]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    // Compile directly without optimizer (RPython test_compile_linear_loop parity)
    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(0);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // Execute
    let frame = backend.execute_token(&token, &[Value::Int(41)]);
    assert_eq!(backend.get_int_value(&frame, 0), 42);
}

// Test 2: Sum loop (trace -> optimize -> compile -> execute)

#[test]
fn test_sum_loop() {
    // Record a loop: input(i, sum) -> sum2 = sum + i -> i2 = i - 1
    //   -> cmp = i2 > 0 -> guard_true(cmp) -> jump(i2, sum2)
    // Execute with i=100, sum=0. Guard fails when i2=0, at which point
    // the accumulated sum in the frame should be 5050.
    let mut rec = Trace::new();
    let i = rec.record_input_arg(Type::Int);
    let sum = rec.record_input_arg(Type::Int);

    let const_one = OpRef::const_int(1);
    let const_zero = OpRef::const_int(0);

    let sum2 = rec.record_op(OpCode::IntAdd, &[sum, i]);
    let i2 = rec.record_op(OpCode::IntSub, &[i, const_one]);
    let cmp = rec.record_op(OpCode::IntGt, &[i2, const_zero]);
    let guard = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(make_descr(0)));
    rec.close_loop(&[i2, sum2]);
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[i, sum]);

    // Optimize

    // Compile
    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(1);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // Execute: i=100, sum=0
    let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(0)]);

    // The loop runs: i=100..1, guard fails when i2=0.
    // At guard failure, the saved frame contains the current input arg values.
    // When i2 becomes 0, we've done sum = 100+99+...+1 = 5050,
    // but the guard fires *before* the jump, so the saved values are i=1, sum=4950+100=5050? No.
    //
    // Let's trace carefully:
    //   Iteration 1: i=100, sum=0 -> sum2=100, i2=99, cmp=true -> jump(99, 100)
    //   Iteration 2: i=99, sum=100 -> sum2=199, i2=98, cmp=true -> jump(98, 199)
    //   ...
    //   Iteration 99: i=2, sum=4949 -> sum2=4951, i2=1, cmp=true -> jump(1, 4951)
    //   Iteration 100: i=1, sum=4951 -> sum2=4952, i2=0, cmp=false -> guard_true fails
    //
    // Wait: 100+99+...+2 = 5050 - 1 = 5049. And then sum2 = 4951 + 1 = 4952? Let me recalculate.
    //   sum after iter 1: 0 + 100 = 100
    //   sum after iter 2: 100 + 99 = 199
    //   ...
    //   sum after iter k: sum of (100, 99, ..., 100-k+1)
    //   After 99 iters: sum = 100+99+...+2 = (100*101/2) - 1 = 5050 - 1 = 5049
    //   Wait: 100+99+...+2 = sum(1..100) - 1 = 5050 - 1 = 5049.
    //   The jump args are (i2=1, sum2=5049).
    //   Iteration 100: i=1, sum=5049 -> sum2=5049+1=5050, i2=0, cmp=false -> guard fails
    //
    // At guard failure, the backend saves fail_args=[i, sum], not (i2, sum2).
    //
    // The sum2 and i2 have NOT been written back to the input vars yet (that happens at Jump).
    assert_eq!(backend.get_int_value(&frame, 0), 1); // i at guard failure
    assert_eq!(backend.get_int_value(&frame, 1), 5049); // sum at guard failure

    // The total sum 1+2+...+100 = 5050.
    // At guard failure we have i=1, sum=5049. The "missing" addition is sum + i = 5049 + 1 = 5050.
    let final_i = backend.get_int_value(&frame, 0);
    let final_sum = backend.get_int_value(&frame, 1);
    assert_eq!(final_sum + final_i, 5050);
}

// Test 3: Constant folding through pipeline

// Test 4: Guard failure path

#[test]
fn test_guard_failure_path() {
    // Record: input(x) -> cmp = x > 0 -> guard_true(cmp)
    //   -> result = x * 2 -> finish(result)
    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);

    let const_zero = OpRef::const_int(0);
    let const_two = OpRef::const_int(2);

    let cmp = rec.record_op(OpCode::IntGt, &[x, const_zero]);
    let guard = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(make_descr(0)));
    let result = rec.record_op(OpCode::IntMul, &[x, const_two]);
    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[x]);

    // Optimize

    // Compile
    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(3);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // Execute with x=5: guard passes, result = 5 * 2 = 10
    let frame = backend.execute_token(&token, &[Value::Int(5)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 10);

    // Execute with x=-1: guard fails, DeadFrame has input arg values
    let frame = backend.execute_token(&token, &[Value::Int(-1)]);
    let descr = backend.get_latest_descr(&frame);
    // fail_index=0 means the guard_true failed (first guard/finish in the trace)
    assert_eq!(descr.fail_index(), 0);
    // Guard failure saves input args: x=-1
    assert_eq!(backend.get_int_value(&frame, 0), -1);

    // Execute with x=0: 0 > 0 is false, guard fails
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    let descr = backend.get_latest_descr(&frame);
    assert_eq!(descr.fail_index(), 0);
    assert_eq!(backend.get_int_value(&frame, 0), 0);
}

// Test 5: Multiple passes working together (CSE + algebraic simplification)

// Test 6: Bridge compilation end-to-end

#[test]
fn test_bridge_end_to_end() {
    // Main trace: sum loop counting down from N.
    //   input(i, sum) -> sum2 = sum + i -> i2 = i - 1
    //   -> cmp = i2 > 0 -> guard_true(cmp) -> jump(i2, sum2)
    //
    // When guard fails (i2 <= 0), compile a bridge that doubles the sum.
    let mut rec = Trace::new();
    let i = rec.record_input_arg(Type::Int);
    let sum = rec.record_input_arg(Type::Int);

    let const_one = OpRef::const_int(1);
    let const_zero = OpRef::const_int(0);

    let sum2 = rec.record_op(OpCode::IntAdd, &[sum, i]);
    let i2 = rec.record_op(OpCode::IntSub, &[i, const_one]);
    let cmp = rec.record_op(OpCode::IntGt, &[i2, const_zero]);
    let guard = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(make_descr(0)));
    rec.close_loop(&[i2, sum2]);
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[i, sum]);

    // Optimize

    // Compile main loop
    let mut backend = CraneliftBackend::new();

    // Pin the loop trace id to match the test's TestFailDescr defaults
    // (`trace_id() == 0`).  In production, metainterp's
    // `record_loop_or_bridge` (compile.py line-by-line counterpart)
    // stamps the metainterp ResumeGuardDescr's trace_id to match the
    // backend allocation; this integration test bypasses metainterp, so
    // we instead pin the backend's `next_trace_id` to align both sides
    // explicitly.  Without this, `find_fail_descr_in_fail_descrs`
    // compares the backend descr's `trace_id` field (allocated by
    // `compile_loop`, default 1) against the bridge fail descr's
    // `trace_id()` trait method (which forwards through the test's
    // TestFailDescr meta_descr, default 0).
    backend.set_next_trace_id(0);
    let token = JitCellToken::new(10);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // Execute without bridge: i=5, sum=0
    // Loop counts 5,4,3,2,1 -> guard fails when i2=0
    // At guard failure we get: i=1, sum=5049? No, with small N=5:
    //   iter1: i=5,sum=0 -> sum2=5, i2=4, pass -> jump(4,5)
    //   iter2: i=4,sum=5 -> sum2=9, i2=3, pass -> jump(3,9)
    //   iter3: i=3,sum=9 -> sum2=12, i2=2, pass -> jump(2,12)
    //   iter4: i=2,sum=12 -> sum2=14, i2=1, pass -> jump(1,14)
    //   iter5: i=1,sum=14 -> sum2=15, i2=0, fail
    // Guard saves input args (i=1, sum=14)
    let frame = backend.execute_token(&token, &[Value::Int(5), Value::Int(0)]);
    let (source_trace_id, source_fail_index_per_trace) = {
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        (descr.trace_id(), descr.fail_index_per_trace())
    };
    assert_eq!(backend.get_int_value(&frame, 0), 1); // i
    assert_eq!(backend.get_int_value(&frame, 1), 14); // sum

    // Now compile a bridge for the guard failure.
    // Bridge takes (i, sum) and returns sum * 2.
    let mut bridge_rec = Trace::new();
    let bi = bridge_rec.record_input_arg(Type::Int);
    let bsum = bridge_rec.record_input_arg(Type::Int);
    let _ = bi; // bridge ignores i

    let bridge_const_two = OpRef::const_int(2);
    let result = bridge_rec.record_op(OpCode::IntMul, &[bsum, bridge_const_two]);
    bridge_rec.finish(&[result], make_descr(1));
    let bridge_trace = bridge_rec.get_trace();

    // Mint a `ResumeGuardDescr` that mirrors the source guard's identity.
    // Bridge compilation looks up the parent guard by
    // `FailDescr::{trace_id, fail_index_per_trace}` from the descr passed
    // by the metainterp (`pyjitpl.rs`).
    let bridge_fail_descr =
        majit_backend::make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
    {
        let fd = bridge_fail_descr.as_fail_descr().unwrap();
        fd.set_trace_id(source_trace_id);
        fd.set_fail_index_per_trace(source_fail_index_per_trace);
    }
    let bridge_fail_descr_ref = bridge_fail_descr.as_fail_descr().unwrap();

    let bridge_info = backend
        .compile_bridge(
            bridge_fail_descr_ref,
            &inputargs_view(&bridge_trace),
            &bridge_trace.ops,
            &token,
            None,
        )
        .expect("bridge compilation should succeed");
    assert!(bridge_info.code_addr != 0);

    // Execute with bridge: i=5, sum=0
    // Loop runs same as before, guard fails with i=1, sum=14
    // Bridge runs: sum * 2 = 14 * 2 = 28
    let frame = backend.execute_token(&token, &[Value::Int(5), Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 28);

    // Also test with different initial values: i=3, sum=0
    //   iter1: i=3,sum=0 -> sum2=3, i2=2, pass -> jump(2,3)
    //   iter2: i=2,sum=3 -> sum2=5, i2=1, pass -> jump(1,5)
    //   iter3: i=1,sum=5 -> sum2=6, i2=0, fail
    // Guard saves i=1, sum=5 -> bridge: 5 * 2 = 10
    let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 10);
}

// Helpers for intdiv pipeline tests

/// Floor division (towards negative infinity).
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r ^ b) < 0) { d - 1 } else { d }
}

/// Floor modulo: a - floor_div(a, b) * b.
fn floor_mod(a: i64, b: i64) -> i64 {
    a - floor_div(a, b) * b
}

/// Build a magic-number division trace directly (no optimizer), compile it,
/// and return (backend, token).
///
/// The trace implements: result = floor_div(input, m)
///   t = input >> 63
///   nt = input ^ t
///   mul = UINT_MUL_HIGH(nt, k)
///   sh = UINT_RSHIFT(mul, i)
///   result = sh ^ t
///   finish(result)
fn build_magic_div_trace(m: i64, token_id: u64) -> (CraneliftBackend, JitCellToken) {
    let (k, i) = magic_numbers(m);

    // Record the magic-number division sequence using Trace
    // so that OpRef indexing is handled correctly.
    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);

    let const_k = OpRef::const_int(k as i64);
    let const_i = OpRef::const_int(i as i64);
    let const_63 = OpRef::const_int(63);

    // t = x >> 63
    let t = rec.record_op(OpCode::IntRshift, &[x, const_63]);
    // nt = x ^ t
    let nt = rec.record_op(OpCode::IntXor, &[x, t]);
    // mul = UINT_MUL_HIGH(nt, k)
    let mul = rec.record_op(OpCode::UintMulHigh, &[nt, const_k]);
    // sh = UINT_RSHIFT(mul, i)
    let sh = rec.record_op(OpCode::UintRshift, &[mul, const_i]);
    // result = sh ^ t
    let result = rec.record_op(OpCode::IntXor, &[sh, t]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(token_id as i64);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    (backend, token)
}

/// Build a magic-number modulo trace: result = n - (n // m) * m.
fn build_magic_mod_trace(m: i64, token_id: u64) -> (CraneliftBackend, JitCellToken) {
    let (k, i) = magic_numbers(m);

    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);

    let const_k = OpRef::const_int(k as i64);
    let const_i = OpRef::const_int(i as i64);
    let const_63 = OpRef::const_int(63);
    let const_m = OpRef::const_int(m);

    // Division: floor_div(x, m)
    let t = rec.record_op(OpCode::IntRshift, &[x, const_63]);
    let nt = rec.record_op(OpCode::IntXor, &[x, t]);
    let mul = rec.record_op(OpCode::UintMulHigh, &[nt, const_k]);
    let sh = rec.record_op(OpCode::UintRshift, &[mul, const_i]);
    let div = rec.record_op(OpCode::IntXor, &[sh, t]);
    // Modulo: x - div * m
    let product = rec.record_op(OpCode::IntMul, &[div, const_m]);
    let remainder = rec.record_op(OpCode::IntSub, &[x, product]);
    rec.finish(&[remainder], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(token_id as i64);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    (backend, token)
}

/// Build a power-of-2 division trace:
///   sign = x >> 63
///   correction = sign & (divisor - 1)
///   adjusted = x + correction
///   result = adjusted >> shift
///   finish(result)
fn build_power_of_two_div_trace(divisor: i64, token_id: u64) -> (CraneliftBackend, JitCellToken) {
    assert!(divisor > 1 && divisor.count_ones() == 1);
    let shift = divisor.trailing_zeros();

    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);

    let const_63 = OpRef::const_int(63);
    let const_mask = OpRef::const_int(divisor - 1);
    let const_shift = OpRef::const_int(shift as i64);

    let sign = rec.record_op(OpCode::IntRshift, &[x, const_63]);
    let correction = rec.record_op(OpCode::IntAnd, &[sign, const_mask]);
    let adjusted = rec.record_op(OpCode::IntAdd, &[x, correction]);
    let result = rec.record_op(OpCode::IntRshift, &[adjusted, const_shift]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(token_id as i64);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    (backend, token)
}

// Test 7: IntFloorDiv magic-number pipeline (divisor = 7)
//
// The magic-number algorithm produces floor division (towards -inf).

#[test]
fn test_intdiv_magic_number_pipeline() {
    let (backend, token) = build_magic_div_trace(7, 100);

    let test_cases: &[(i64, i64)] = &[
        (0, 0),
        (1, 0),
        (6, 0),
        (7, 1),
        (14, 2),
        (100, 14),
        (-1, -1),
        (-6, -1),
        (-7, -1),
        (-8, -2),
        (-100, -15),
        (i64::MAX, floor_div(i64::MAX, 7)),
        (i64::MIN + 1, floor_div(i64::MIN + 1, 7)),
    ];

    for &(input, expected) in test_cases {
        let frame = backend.execute_token(&token, &[Value::Int(input)]);
        let actual = backend.get_int_value(&frame, 0);
        assert_eq!(
            actual, expected,
            "IntFloorDiv: {input} // 7 = expected {expected}, got {actual}"
        );
    }
}

// Test 8: IntMod magic-number pipeline (divisor = 7)

#[test]
fn test_intmod_magic_number_pipeline() {
    let (backend, token) = build_magic_mod_trace(7, 101);

    let test_cases: &[(i64, i64)] = &[
        (0, 0),
        (1, 1),
        (6, 6),
        (7, 0),
        (14, 0),
        (100, floor_mod(100, 7)),
        (-1, floor_mod(-1, 7)),
        (-7, 0),
        (-8, floor_mod(-8, 7)),
        (-100, floor_mod(-100, 7)),
        (i64::MAX, floor_mod(i64::MAX, 7)),
        (i64::MIN + 1, floor_mod(i64::MIN + 1, 7)),
    ];

    for &(input, expected) in test_cases {
        let frame = backend.execute_token(&token, &[Value::Int(input)]);
        let actual = backend.get_int_value(&frame, 0);
        assert_eq!(
            actual, expected,
            "IntMod: {input} % 7 = expected {expected}, got {actual}"
        );
    }
}

// Test 9: IntFloorDiv power-of-2 pipeline (divisor = 8)
//
// The power-of-2 strength reduction uses the Hacker's Delight formula:
//   result = (x + ((x >> 63) & (2^n - 1))) >> n
// This produces truncation division (towards zero), matching Cranelift sdiv.

#[test]
fn test_intdiv_power_of_two_pipeline() {
    let (backend, token) = build_power_of_two_div_trace(8, 102);

    let test_cases: &[(i64, i64)] = &[
        (0, 0),
        (1, 0),
        (7, 0),
        (8, 1),
        (16, 2),
        (100, 12),
        (-1, 0),     // truncation: -1/8 = 0
        (-7, 0),     // truncation: -7/8 = 0
        (-8, -1),    // exact: -8/8 = -1
        (-9, -1),    // truncation: -9/8 = -1
        (-100, -12), // truncation: -100/8 = -12
        (i64::MAX, i64::MAX / 8),
        (i64::MIN + 1, (i64::MIN + 1) / 8),
    ];

    for &(input, expected) in test_cases {
        let frame = backend.execute_token(&token, &[Value::Int(input)]);
        let actual = backend.get_int_value(&frame, 0);
        assert_eq!(
            actual, expected,
            "IntFloorDiv(pow2): {input} / 8 = expected {expected}, got {actual}"
        );
    }
}

// Test 10: IntFloorDiv magic-number various divisors

#[test]
fn test_intdiv_various_divisors() {
    // Keep one small odd divisor, one even composite divisor, and one
    // larger odd divisor. The per-divisor code generation path is the
    // same, so a representative set is enough here.
    for (tid, divisor) in [3i64, 10, 127].iter().enumerate() {
        let (backend, token) = build_magic_div_trace(*divisor, 200 + tid as u64);

        for input in [
            0i64,
            1,
            *divisor - 1,
            *divisor,
            *divisor + 1,
            999,
            -1,
            -*divisor,
            -999,
        ] {
            let expected = floor_div(input, *divisor);
            let frame = backend.execute_token(&token, &[Value::Int(input)]);
            let actual = backend.get_int_value(&frame, 0);
            assert_eq!(
                actual, expected,
                "IntFloorDiv: {input} // {divisor} = expected {expected}, got {actual}"
            );
        }
    }
}

// Test 11: VecIntAdd native SIMD (pack + add + unpack)

#[test]
fn test_vec_int_add_simd() {
    // input(a, b, c, d)
    // vec0 = VecI()
    // vec1 = VecPackI(vec0, a, 0, 2)
    // vec2 = VecPackI(vec1, b, 1, 2)
    // vec3 = VecI()
    // vec4 = VecPackI(vec3, c, 0, 2)
    // vec5 = VecPackI(vec4, d, 1, 2)
    // vec6 = VecIntAdd(vec2, vec5)
    // r0 = VecUnpackI(vec6, 0, 2)
    // r1 = VecUnpackI(vec6, 1, 2)
    // finish(r0, r1)
    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef::const_int(0);
    let const_1 = OpRef::const_int(1);
    let const_2 = OpRef::const_int(2);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    let vec6 = rec.record_op(OpCode::VecIntAdd, &[vec2, vec5]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(300);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // a=10, b=20, c=3, d=7 -> r0 = 10+3 = 13, r1 = 20+7 = 27
    let frame = backend.execute_token(
        &token,
        &[Value::Int(10), Value::Int(20), Value::Int(3), Value::Int(7)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 13);
    assert_eq!(backend.get_int_value(&frame, 1), 27);

    // Negative values: a=-5, b=100, c=15, d=-200
    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(-5),
            Value::Int(100),
            Value::Int(15),
            Value::Int(-200),
        ],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 10); // -5 + 15
    assert_eq!(backend.get_int_value(&frame, 1), -100); // 100 + (-200)
}

// Test 12: VecIntSub native SIMD

#[test]
fn test_vec_int_sub_simd() {
    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef::const_int(0);
    let const_1 = OpRef::const_int(1);
    let const_2 = OpRef::const_int(2);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    let vec6 = rec.record_op(OpCode::VecIntSub, &[vec2, vec5]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(301);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // a=10, b=20, c=3, d=7 -> r0 = 10-3 = 7, r1 = 20-7 = 13
    let frame = backend.execute_token(
        &token,
        &[Value::Int(10), Value::Int(20), Value::Int(3), Value::Int(7)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    assert_eq!(backend.get_int_value(&frame, 1), 13);
}

// Test 13: VecIntMul native SIMD

#[test]
fn test_vec_int_mul_simd() {
    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef::const_int(0);
    let const_1 = OpRef::const_int(1);
    let const_2 = OpRef::const_int(2);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    let vec6 = rec.record_op(OpCode::VecIntMul, &[vec2, vec5]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(302);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // a=5, b=6, c=7, d=8 -> r0 = 5*7 = 35, r1 = 6*8 = 48
    let frame = backend.execute_token(
        &token,
        &[Value::Int(5), Value::Int(6), Value::Int(7), Value::Int(8)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 35);
    assert_eq!(backend.get_int_value(&frame, 1), 48);
}

// Test 14: VecExpandI + VecIntAdd (broadcast + vector add)

#[test]
fn test_vec_expand_add_simd() {
    // input(a, b, s)
    // vec_ab = pack(a, b)
    // vec_s = VecExpandI(s)       -- broadcast s to both lanes
    // vec_r = VecIntAdd(vec_ab, vec_s)
    // r0 = unpack(vec_r, 0)
    // r1 = unpack(vec_r, 1)
    // finish(r0, r1)
    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let s = rec.record_input_arg(Type::Int);

    let const_0 = OpRef::const_int(0);
    let const_1 = OpRef::const_int(1);
    let const_2 = OpRef::const_int(2);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec_s = rec.record_op(OpCode::VecExpandI, &[s]);
    let vec_r = rec.record_op(OpCode::VecIntAdd, &[vec2, vec_s]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec_r, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec_r, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(303);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // a=10, b=20, s=100 -> r0 = 10+100 = 110, r1 = 20+100 = 120
    let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(20), Value::Int(100)]);
    assert_eq!(backend.get_int_value(&frame, 0), 110);
    assert_eq!(backend.get_int_value(&frame, 1), 120);
}

// Test 15: VecFloatAdd native SIMD

#[test]
fn test_vec_float_add_simd() {
    // We use CastIntToFloat to convert i64 inputs to f64, then pack into
    // F64X2 vectors, add, unpack, and CastFloatToInt to get back i64.
    // This avoids type mismatch in Finish fail_arg_types.
    //
    // Use VecPackI/VecUnpackI with VecFloatAdd: the VecFloatAdd reads
    // I64X2 operands and reinterprets them as F64X2 internally.
    // Instead, use Int-typed inputs, CastIntToFloat for each, then pack
    // into F64X2 and unpack as Float, then CastFloatToInt.
    //
    // Simpler: just use Int paths throughout, packing f64 bit patterns
    // into I64X2 via VecPackI, then use VecFloatAdd (which handles the
    // bitcast internally), then VecUnpackI to get scalar i64 back.
    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef::const_int(0);
    let const_1 = OpRef::const_int(1);
    let const_2 = OpRef::const_int(2);

    // Pack f64 bit patterns into I64X2 vectors
    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    // VecFloatAdd operates on the I64X2 as if they contain f64 bit patterns
    let vec6 = rec.record_op(OpCode::VecFloatAdd, &[vec2, vec5]);
    // Unpack using VecUnpackI to get scalar i64 (containing f64 bits)
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(304);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // Pass f64 values as i64 bit patterns
    let a_f = 1.5f64;
    let b_f = 2.5f64;
    let c_f = 3.0f64;
    let d_f = 4.0f64;

    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(a_f.to_bits() as i64),
            Value::Int(b_f.to_bits() as i64),
            Value::Int(c_f.to_bits() as i64),
            Value::Int(d_f.to_bits() as i64),
        ],
    );
    let r0_bits = backend.get_int_value(&frame, 0) as u64;
    let r1_bits = backend.get_int_value(&frame, 1) as u64;
    assert_eq!(f64::from_bits(r0_bits), 4.5); // 1.5 + 3.0
    assert_eq!(f64::from_bits(r1_bits), 6.5); // 2.5 + 4.0
}

// Test 16: Chained VecIntAdd + VecIntMul (vector add then multiply)

#[test]
fn test_vec_chained_add_mul_simd() {
    // input(a, b, c, d, e, f)
    // vec_ab = pack(a, b)
    // vec_cd = pack(c, d)
    // vec_ef = pack(e, f)
    // vec_sum = VecIntAdd(vec_ab, vec_cd)
    // vec_result = VecIntMul(vec_sum, vec_ef)
    // r0 = unpack(vec_result, 0)
    // r1 = unpack(vec_result, 1)
    // finish(r0, r1)
    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);
    let e = rec.record_input_arg(Type::Int);
    let f = rec.record_input_arg(Type::Int);

    let const_0 = OpRef::const_int(0);
    let const_1 = OpRef::const_int(1);
    let const_2 = OpRef::const_int(2);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec_a = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec_ab = rec.record_op(OpCode::VecPackI, &[vec_a, b, const_1, const_2]);

    let vec1 = rec.record_op(OpCode::VecI, &[]);
    let vec_c = rec.record_op(OpCode::VecPackI, &[vec1, c, const_0, const_2]);
    let vec_cd = rec.record_op(OpCode::VecPackI, &[vec_c, d, const_1, const_2]);

    let vec2 = rec.record_op(OpCode::VecI, &[]);
    let vec_e = rec.record_op(OpCode::VecPackI, &[vec2, e, const_0, const_2]);
    let vec_ef = rec.record_op(OpCode::VecPackI, &[vec_e, f, const_1, const_2]);

    let vec_sum = rec.record_op(OpCode::VecIntAdd, &[vec_ab, vec_cd]);
    let vec_result = rec.record_op(OpCode::VecIntMul, &[vec_sum, vec_ef]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec_result, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec_result, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(305);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // a=2, b=3, c=4, d=5, e=10, f=20
    // sum = (2+4, 3+5) = (6, 8)
    // result = (6*10, 8*20) = (60, 160)
    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5),
            Value::Int(10),
            Value::Int(20),
        ],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 60);
    assert_eq!(backend.get_int_value(&frame, 1), 160);
}

// Stress tests: multi-pass optimizer pipeline integration

/// Minimal field descriptor for optimizer-level tests.
#[derive(Debug)]
struct TestFieldDescr {
    idx: u32,
    immutable: bool,
}

impl Descr for TestFieldDescr {
    fn index(&self) -> u32 {
        self.idx
    }
    fn is_always_pure(&self) -> bool {
        self.immutable
    }
}

/// Minimal call descriptor for optimizer-level tests.
#[derive(Debug)]
struct TestCallDescr {
    idx: u32,
    effect: majit_ir::EffectInfo,
    arg_types: Vec<Type>,
    result_type: Type,
}

impl Descr for TestCallDescr {
    fn index(&self) -> u32 {
        self.idx
    }
    fn as_call_descr(&self) -> Option<&dyn majit_ir::CallDescr> {
        Some(self)
    }
}

impl majit_ir::CallDescr for TestCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        8
    }
    fn get_extra_info(&self) -> &majit_ir::EffectInfo {
        &self.effect
    }
}

#[expect(
    dead_code,
    reason = "optimizer descriptor fixture helper retained for coverage parity"
)]
fn field_descr(idx: u32) -> DescrRef {
    Arc::new(TestFieldDescr {
        idx,
        immutable: false,
    })
}

#[expect(
    dead_code,
    reason = "optimizer descriptor fixture helper retained for coverage parity"
)]
fn immutable_field_descr(idx: u32) -> DescrRef {
    Arc::new(TestFieldDescr {
        idx,
        immutable: true,
    })
}

#[expect(
    dead_code,
    reason = "optimizer descriptor fixture helper retained for coverage parity"
)]
fn call_descr_can_raise(idx: u32) -> DescrRef {
    Arc::new(TestCallDescr {
        idx,
        effect: majit_ir::EffectInfo {
            extraeffect: majit_ir::ExtraEffect::CanRaise,
            oopspecindex: majit_ir::OopSpecIndex::None,
            ..Default::default()
        },
        arg_types: vec![Type::Int],
        result_type: Type::Void,
    })
}

/// Assign sequential positions to ops starting at `base`.
fn assign_positions(ops: &mut [Op], base: u32) {
    for (i, op) in ops.iter_mut().enumerate() {
        let pos = base + i as u32;
        op.pos().set(OpRef::op_typed(pos, op.result_type()));
    }
}

// Stress Test 1: Virtual Object + Guard Elimination + Constant Folding
//
// Trace (optimizer-level):
//   p0 = new_with_vtable(descr=size0)
//   setfield_gc(p0, CONST_42, descr=field0)
//   i1 = getfield_gc_i(p0, descr=field0)
//   i2 = int_add(i1, CONST_8)
//   i3 = int_gt(i2, CONST_0)
//   guard_true(i3)
//   finish(i2)
//
// Expected after full pipeline:
//   - new_with_vtable eliminated (virtualized by OptVirtualize)
//   - setfield_gc eliminated (virtual, no escape)
//   - getfield_gc_i folded to CONST_42 (virtual field read)
//   - int_add(42, 8) = 50 (constant folded by OptPure/OptRewrite)
//   - int_gt(50, 0) = true (constant folded by IntBounds)
//   - guard_true(true) eliminated (guard on known-true condition)
//   - finish(50)

// Stress Test 2: IntDiv magic numbers + IntBounds guard elimination
//
// Part A: Optimizer level - verify redundant guard removal.
// Part B: Execution level - verify magic number division correctness
//         (separate trace without redundant guards to avoid fail_index conflicts).

// Stress Test 3: Loop with CSE + Guard Deduplication
//
// Loop trace:
//   label(i, acc)
//   sq = int_mul(i, i)        -- (first computation)
//   acc2 = int_add(acc, sq)
//   i2   = int_sub(i, CONST_1)
//   cmp1 = int_gt(i2, CONST_0)
//   guard_true(cmp1)
//   cmp2 = int_gt(i2, CONST_0)   -- CSE of cmp1
//   guard_true(cmp2)              -- duplicate guard, should be removed
//   sq2  = int_mul(i, i)         -- CSE of sq
//   result = int_add(acc2, sq2)   -- after CSE: acc2 + sq
//   jump(i2, result)
//
// Expected after optimization:
//   - cmp2 eliminated by CSE (OptPure)
//   - second guard_true eliminated (duplicate folded by OptRewrite)
//   - sq2 eliminated by CSE (OptPure), forwarded to sq

// Stress Test 4: Heap Cache + Green Field (Immutable) Optimization
//
// Trace (optimizer-level):
//   i0 = getfield_gc_i(p_input, descr=immutable_d0)
//   call_n(some_func, i0, descr=call_d)
//   i1 = getfield_gc_i(p_input, descr=immutable_d0)  -- should be cached
//   i2 = getfield_gc_i(p_input, descr=mutable_d1)
//   call_n(some_func, i2, descr=call_d)
//   i3 = getfield_gc_i(p_input, descr=mutable_d1)    -- re-emitted (invalidated)
//   i4 = int_add(i0, i1)     -- after CSE: i1 -> i0, so i4 = i0 + i0
//   i5 = int_add(i4, i3)
//   finish(i5)
//
// Expected after full pipeline:
//   - Second immutable getfield (i1) eliminated by OptHeap green field cache
//   - Second mutable getfield (i3) re-emitted (cache invalidated by call)
//   - i1 forwarded to i0, so i4 becomes int_add(i0, i0)

// Stress Test 5: String Virtualization
//
// Trace (optimizer-level):
//   s0 = newstr(CONST_3)
//   strsetitem(s0, CONST_0, CONST_65)   -- 'A'
//   strsetitem(s0, CONST_1, CONST_66)   -- 'B'
//   strsetitem(s0, CONST_2, CONST_67)   -- 'C'
//   i0 = strgetitem(s0, CONST_1)        -- should resolve to CONST_66
//   i1 = strlen(s0)                     -- should resolve to CONST_3
//   i2 = int_add(i0, i1)               -- 66 + 3 = 69
//   finish(i2)
//
// Expected after full pipeline:
//   - newstr eliminated (virtualized by OptString)
//   - all strsetitem eliminated (writing to virtual)
//   - strgetitem folded to constant 66
//   - strlen folded to constant 3
//   - int_add(66, 3) folded to 69
//   - Trace reduces to Finish(69)

// Stress Test 6: Combined loop with guard dedup, CSE, and IntBounds
//
// This test verifies that the full pipeline handles a realistic loop trace
// with interleaved optimizations: IntBounds narrows ranges, Pure deduplicates,
// Guard removes redundant guards, and the result compiles and runs correctly.
//
// Loop trace:
//   input(x, acc)
//   // Guard x > 0 (provides IntBounds: x >= 1)
//   cmp0 = int_gt(x, CONST_0)
//   guard_true(cmp0)
//   // Computation: acc += x * x
//   sq = int_mul(x, x)
//   acc2 = int_add(acc, sq)
//   // Decrement
//   x2 = int_sub(x, CONST_1)
//   // Guard x2 >= 0 (IntBounds knows x >= 1, so x2 >= 0 is known)
//   cmp1 = int_ge(x2, CONST_0)
//   guard_true(cmp1)           -- should be eliminated by IntBounds
//   // Guard x > 0 again (redundant)
//   cmp2 = int_gt(x, CONST_0)
//   guard_true(cmp2)           -- CSE + guard dedup should eliminate
//   // Continue check
//   cmp3 = int_gt(x2, CONST_0)
//   guard_true(cmp3)
//   jump(x2, acc2)
//
// Expected:
//   - cmp1/guard eliminated (IntBounds: x>=1 implies x-1>=0)
//   - cmp2/guard eliminated (CSE of cmp0 + guard dedup)

// Stress Test 7: Full pipeline with arithmetic chains and multiple CSE
//
// This test builds a longer arithmetic chain with several CSE opportunities,
// compiles, and verifies execution through the Cranelift backend.
//
// Trace:
//   input(a, b)
//   c = a + b
//   d = a * b
//   e = a + b           -- CSE -> c
//   f = c + d
//   g = a * b           -- CSE -> d
//   h = f + g           -- after CSE: f + d
//   i_val = c + d       -- CSE -> f
//   result = h + i_val  -- after CSE: h + f
//   finish(result)
//
// After CSE: e->c, g->d, i_val->f
//   c = a + b
//   d = a * b
//   f = c + d
//   h = f + d
//   result = h + f
//   finish(result)

// Threadlocal parity tests (RPython: test_threadlocal.py)

// Test: ThreadlocalrefGet compiles and reads back a value set via the shim
//
// Mirrors RPython's test_threadlocalref_get: set a TLS slot, then compile
// a trace that reads it via ThreadlocalrefGet, verify the value matches.

#[test]
fn test_threadlocalref_get_basic() {
    use majit_backend_cranelift::compiler::jit_threadlocalref_set;

    // Set slot 0 (offset=0) to 0x544C (same magic value as RPython test).
    jit_threadlocalref_set(0, 0x544C);

    // Build trace: ThreadlocalrefGet(offset=0) -> finish(result)
    let mut rec = Trace::new();
    let _dummy = rec.record_input_arg(Type::Int); // need at least one input

    let const_offset = OpRef::const_int(0); // offset = 0 bytes
    let result = rec.record_op(OpCode::ThreadlocalrefGet, &[const_offset]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(500);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    // ThreadlocalrefGet returns a Ref type (pointer-sized).
    let got = backend.get_ref_value(&frame, 0);
    assert_eq!(
        got.0 as i64, 0x544C,
        "ThreadlocalrefGet should read back 0x544C"
    );
}

// Test: ThreadlocalrefGet reads different slots at different offsets
//
// Mirrors RPython's test_threadlocalref_get_char: verify that distinct
// offsets map to distinct slots. Offsets are in bytes (divided by 8
// internally to get slot index).

#[test]
fn test_threadlocalref_get_multiple_slots() {
    use majit_backend_cranelift::compiler::jit_threadlocalref_set;

    // Set slot 0 (offset 0) and slot 1 (offset 8) to different values.
    jit_threadlocalref_set(0, 0xAAAA);
    jit_threadlocalref_set(8, 0xBBBB);

    // Build trace that reads both slots and adds them:
    //   r0 = ThreadlocalrefGet(offset=0)
    //   r1 = ThreadlocalrefGet(offset=8)
    //   result = r0 + r1
    //   finish(result)
    let mut rec = Trace::new();
    let _dummy = rec.record_input_arg(Type::Int);

    let const_off0 = OpRef::const_int(0);
    let const_off8 = OpRef::const_int(8);

    let r0 = rec.record_op(OpCode::ThreadlocalrefGet, &[const_off0]);
    let r1 = rec.record_op(OpCode::ThreadlocalrefGet, &[const_off8]);
    let result = rec.record_op(OpCode::IntAdd, &[r0, r1]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(501);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        0xAAAA + 0xBBBB,
        "Sum of two TLS slots should be 0xAAAA + 0xBBBB"
    );
}

// Test: ThreadlocalrefGet set-then-read roundtrip
//
// Verifies that writing a value with jit_threadlocalref_set and reading
// it back via compiled ThreadlocalrefGet produces the same value, for
// several test values including zero and negative numbers.

#[test]
fn test_threadlocalref_set_and_read_roundtrip() {
    use majit_backend_cranelift::compiler::jit_threadlocalref_set;

    // Build trace once: ThreadlocalrefGet(offset=16) -> finish(result)
    let mut rec = Trace::new();
    let _dummy = rec.record_input_arg(Type::Int);

    let const_offset = OpRef::const_int(16); // offset 16 -> slot index 2
    let result = rec.record_op(OpCode::ThreadlocalrefGet, &[const_offset]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(502);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    for value in [0i64, 1, -1, 0x544C, i64::MAX, i64::MIN] {
        jit_threadlocalref_set(16, value);
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        // ThreadlocalrefGet returns a Ref type (pointer-sized).
        let got = backend.get_ref_value(&frame, 0);
        assert_eq!(got.0 as i64, value, "roundtrip failed for value {value}");
    }
}

// Test: Different threads see independent TLS values
//
// Mirrors RPython's implicit thread isolation: each thread has its own
// JIT_THREADLOCAL_SLOTS (thread_local!). Verify that writing on one
// thread does not affect reads on another.

#[test]
fn test_threadlocalref_thread_isolation() {
    use majit_backend_cranelift::compiler::jit_threadlocalref_set;
    use std::sync::{Arc as StdArc, Barrier};
    use std::thread;

    // Set the main-thread slot.
    jit_threadlocalref_set(0, 0x1111);

    let barrier = StdArc::new(Barrier::new(2));
    let b2 = barrier.clone();

    let child = thread::spawn(move || {
        // Child thread: its TLS slot 0 should default to 0 (not 0x1111).
        jit_threadlocalref_set(0, 0x2222);
        b2.wait(); // synchronize so main can check its own value
        // Read back to confirm child's slot is 0x2222.
        // We can't easily run compiled code on the child thread
        // (JitCellToken/backend not Send), but we can verify via the shim.
        let base = majit_backend_cranelift::compiler::jit_threadlocalref_base();

        if base.is_null() { 0 } else { unsafe { *base } }
    });

    barrier.wait();
    // Main thread's slot 0 should still be 0x1111.
    let main_base = majit_backend_cranelift::compiler::jit_threadlocalref_base();
    let main_val = if main_base.is_null() {
        0
    } else {
        unsafe { *main_base }
    };
    assert_eq!(main_val, 0x1111, "main thread TLS slot should be unchanged");

    let child_val = child.join().unwrap();
    assert_eq!(child_val, 0x2222, "child thread should see its own value");
}

// FFI call (CallReleaseGil) end-to-end parity tests
// (rpython/jit/metainterp/test/test_fficall.py)
//
// Verifies that CallReleaseGilI compiles and executes correctly, with the
// GIL release/reacquire shims called around the actual foreign function.

/// Simple extern "C" function: adds two i64 values.
extern "C" fn ffi_add(a: i64, b: i64) -> i64 {
    a + b
}

/// extern "C" function returning a constant to verify void-arg calls.
extern "C" fn ffi_constant() -> i64 {
    42
}

fn call_descr_release_gil_i(idx: u32, arg_types: Vec<Type>) -> DescrRef {
    Arc::new(TestCallDescr {
        idx,
        effect: majit_ir::EffectInfo {
            extraeffect: majit_ir::ExtraEffect::CanRaise,
            oopspecindex: majit_ir::OopSpecIndex::None,
            ..Default::default()
        },
        arg_types,
        result_type: Type::Int,
    })
}

fn call_descr_release_gil_n(idx: u32, arg_types: Vec<Type>) -> DescrRef {
    Arc::new(TestCallDescr {
        idx,
        effect: majit_ir::EffectInfo {
            extraeffect: majit_ir::ExtraEffect::CanRaise,
            oopspecindex: majit_ir::OopSpecIndex::None,
            ..Default::default()
        },
        arg_types,
        result_type: Type::Void,
    })
}

#[test]
fn test_call_release_gil_i_compiles_and_executes() {
    // Parity with test_fficall._run: CallReleaseGilI compiles and calls an
    // extern "C" fn, returning the correct result through the release-gil path.
    // RPython: CallReleaseGilI must be followed by GuardNotForced.
    //
    // Trace: input(a, b) -> result = call_release_gil_i(saveerr, ffi_add, a, b)
    //        -> guard_not_forced(fail_args=[a, b, result]) -> finish(result)
    let cd = call_descr_release_gil_i(60, vec![Type::Int, Type::Int]);

    let mut rec = Trace::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);

    let saveerr = OpRef::const_int(0);
    let fn_ptr = OpRef::const_int(ffi_add as *const () as usize as i64);

    let result = rec.record_op_with_descr(OpCode::CallReleaseGilI, &[saveerr, fn_ptr, a, b], cd);
    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(0)));
    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[a, b, result]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(600);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("CallReleaseGilI should compile");

    // Execute with a=10, b=32 -> expected result 42
    let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "ffi_add(10, 32) should return 42"
    );

    // Execute with different values
    let frame = backend.execute_token(&token, &[Value::Int(-5), Value::Int(5)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        0,
        "ffi_add(-5, 5) should return 0"
    );
}

#[test]
fn test_call_release_gil_i_no_args() {
    // CallReleaseGilI with no arguments (like ffi_constant() -> 42).
    // RPython: CallReleaseGilI must be followed by GuardNotForced.
    let cd = call_descr_release_gil_i(61, vec![]);

    let mut rec = Trace::new();
    let dummy = rec.record_input_arg(Type::Int); // need at least one input
    let saveerr = OpRef::const_int(0);
    let fn_ptr = OpRef::const_int(ffi_constant as *const () as usize as i64);

    let result = rec.record_op_with_descr(OpCode::CallReleaseGilI, &[saveerr, fn_ptr], cd);
    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(0)));
    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[dummy, result]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(601);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("CallReleaseGilI with no args should compile");

    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "ffi_constant() should return 42"
    );
}

/// Extern "C" sink: stores the value so we can verify the call happened.
static FFI_SINK_VALUE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

extern "C" fn ffi_sink(val: i64) {
    FFI_SINK_VALUE.store(val, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn test_call_release_gil_n_void_return() {
    // Parity with test_fficall: CallReleaseGilN for void-returning FFI calls.
    // RPython: CallReleaseGilN must be followed by GuardNotForced.
    let cd = call_descr_release_gil_n(62, vec![Type::Int]);

    let mut rec = Trace::new();
    let input = rec.record_input_arg(Type::Int);
    let saveerr = OpRef::const_int(0);
    let fn_ptr = OpRef::const_int(ffi_sink as *const () as usize as i64);

    rec.record_op_with_descr(OpCode::CallReleaseGilN, &[saveerr, fn_ptr, input], cd);
    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(0)));
    rec.finish(&[input], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[input]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(602);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("CallReleaseGilN should compile");

    FFI_SINK_VALUE.store(0, std::sync::atomic::Ordering::SeqCst);
    let _frame = backend.execute_token(&token, &[Value::Int(12345)]);
    assert_eq!(
        FFI_SINK_VALUE.load(std::sync::atomic::Ordering::SeqCst),
        12345,
        "ffi_sink should have been called with 12345"
    );
}

#[test]
fn test_call_release_gil_result_flows_through_trace() {
    // Verify that the result of CallReleaseGilI can be used by subsequent ops.
    // RPython: CallReleaseGilI must be followed by GuardNotForced.
    // Trace: input(x) -> tmp = call_release_gil_i(saveerr, ffi_add, x, 10)
    //        -> guard_not_forced -> result = int_add(tmp, 5) -> finish(result)
    let cd = call_descr_release_gil_i(63, vec![Type::Int, Type::Int]);

    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);
    let saveerr = OpRef::const_int(0);
    let fn_ptr = OpRef::const_int(ffi_add as *const () as usize as i64);
    let const_10 = OpRef::const_int(10);
    let const_5 = OpRef::const_int(5);

    let tmp =
        rec.record_op_with_descr(OpCode::CallReleaseGilI, &[saveerr, fn_ptr, x, const_10], cd);
    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(0)));
    let result = rec.record_op(OpCode::IntAdd, &[tmp, const_5]);
    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[x, tmp]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(603);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("trace with CallReleaseGilI + IntAdd should compile");

    // x=7 -> ffi_add(7, 10)=17 -> 17+5=22
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        22,
        "ffi_add(7,10)+5 should be 22"
    );
}

#[test]
fn test_call_release_gil_hooks_are_callable() {
    // Parity with GIL release/reacquire semantics: verify that set_gil_hooks
    // is callable and the shim infrastructure exists. Since OnceLock can only
    // be set once per process, we just verify the API is available.
    //
    // The actual hook invocation is tested implicitly by the CallReleaseGilI
    // tests above (the shim functions are always called; they just check
    // whether a hook was installed).
    use majit_backend_cranelift::set_gil_hooks;

    // set_gil_hooks uses OnceLock, so it may fail silently if already set
    // by another test. That's fine; we just verify the function is callable.
    set_gil_hooks(|| {}, || {});
}

// Raw memory test helpers

#[derive(Debug)]
struct RawArrayDescr {
    item_size: usize,
    item_type: Type,
    signed: bool,
}

impl Descr for RawArrayDescr {
    fn index(&self) -> u32 {
        0
    }
    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(self)
    }
}

impl ArrayDescr for RawArrayDescr {
    fn base_size(&self) -> usize {
        0
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        0
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.signed
    }
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        None
    }
}

fn raw_descr_int(item_size: usize) -> DescrRef {
    Arc::new(RawArrayDescr {
        item_size,
        item_type: Type::Int,
        signed: true,
    })
}

fn raw_descr_float() -> DescrRef {
    Arc::new(RawArrayDescr {
        item_size: 8,
        item_type: Type::Float,
        signed: false,
    })
}

// Test: RawStore + RawLoadI roundtrip (integer)
//
// Mirrors RPython's test_raw_storage_int: allocate raw memory, store a value,
// load it back, verify the result.

#[test]
fn test_raw_store_load_int_roundtrip() {
    let ad = raw_descr_int(8);
    let const_offset = OpRef::const_int(0); // offset 0

    let mut rec = Trace::new();
    let r0 = rec.record_input_arg(Type::Ref);
    let i0 = rec.record_input_arg(Type::Int);

    rec.record_op_with_descr(OpCode::RawStore, &[r0, const_offset, i0], ad.clone());
    let loaded = rec.record_op_with_descr(OpCode::RawLoadI, &[r0, const_offset], ad.clone());
    rec.finish(&[loaded], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(600);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("raw int roundtrip compilation should succeed");

    // Allocate a buffer and execute
    let mut buf = vec![0u8; 16];
    let ptr = buf.as_mut_ptr() as usize;

    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(12345)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        12345,
        "raw_store then raw_load_i should roundtrip the integer value"
    );
}

// Test: RawStore + RawLoadF roundtrip (float)
//
// Mirrors RPython's test_raw_storage_float.

#[test]
fn test_raw_store_load_float_roundtrip() {
    let ad = raw_descr_float();
    let const_offset = OpRef::const_int(0);

    let mut rec = Trace::new();
    let r0 = rec.record_input_arg(Type::Ref);
    let f0 = rec.record_input_arg(Type::Float);

    rec.record_op_with_descr(OpCode::RawStore, &[r0, const_offset, f0], ad.clone());
    let loaded = rec.record_op_with_descr(OpCode::RawLoadF, &[r0, const_offset], ad.clone());
    rec.finish(&[loaded], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(601);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("raw float roundtrip compilation should succeed");

    let mut buf = vec![0u8; 16];
    let ptr = buf.as_mut_ptr() as usize;

    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Float(3.25)]);
    assert!(
        (backend.get_float_value(&frame, 0) - 3.25).abs() < 1e-10,
        "raw_store then raw_load_f should roundtrip the float value"
    );
}

// Test: RawStore + RawLoadI at an odd byte offset
//
// Mirrors RawMemTests.test_raw_storage_int: store and load a Signed at
// offset 3, which is not naturally aligned for an 8-byte item.

#[test]
fn test_raw_store_load_int_odd_offset() {
    let ad = raw_descr_int(8);
    let const_offset = OpRef::const_int(3);

    let mut rec = Trace::new();
    let r0 = rec.record_input_arg(Type::Ref);
    let i0 = rec.record_input_arg(Type::Int);

    rec.record_op_with_descr(OpCode::RawStore, &[r0, const_offset, i0], ad.clone());
    let loaded = rec.record_op_with_descr(OpCode::RawLoadI, &[r0, const_offset], ad.clone());
    rec.finish(&[loaded], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(606);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("raw int odd-offset roundtrip compilation should succeed");

    let mut buf = vec![0u8; 16];
    let ptr = buf.as_mut_ptr() as usize;

    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(24)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        24,
        "raw_store then raw_load_i at offset 3 should roundtrip the integer value"
    );
}

// Test: RawStore + RawLoadF at an odd byte offset
//
// Mirrors RawMemTests.test_raw_storage_float (offset 4 there). Offset 3
// is the same unaligned contract for an 8-byte float.

#[test]
fn test_raw_store_load_float_odd_offset() {
    let ad = raw_descr_float();
    let const_offset = OpRef::const_int(3);

    let mut rec = Trace::new();
    let r0 = rec.record_input_arg(Type::Ref);
    let f0 = rec.record_input_arg(Type::Float);

    rec.record_op_with_descr(OpCode::RawStore, &[r0, const_offset, f0], ad.clone());
    let loaded = rec.record_op_with_descr(OpCode::RawLoadF, &[r0, const_offset], ad.clone());
    rec.finish(&[loaded], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(607);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("raw float odd-offset roundtrip compilation should succeed");

    let mut buf = vec![0u8; 16];
    let ptr = buf.as_mut_ptr() as usize;

    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Float(2.4e15)]);
    assert_eq!(
        backend.get_float_value(&frame, 0),
        2.4e15,
        "raw_store then raw_load_f at offset 3 should roundtrip the float value"
    );
}

// Test: Raw ops at different offsets don't interfere
//
// Store two different integers at different offsets, then load both back.
// Verifies that the Cranelift codegen correctly handles distinct offsets.

#[test]
fn test_raw_ops_different_offsets_no_interference() {
    let ad = raw_descr_int(8);
    let off0 = OpRef::const_int(0);
    let off8 = OpRef::const_int(8);

    let mut rec = Trace::new();
    let r0 = rec.record_input_arg(Type::Ref);
    let i0 = rec.record_input_arg(Type::Int);
    let i1 = rec.record_input_arg(Type::Int);

    // Store val1 at offset 0, val2 at offset 8
    rec.record_op_with_descr(OpCode::RawStore, &[r0, off0, i0], ad.clone());
    rec.record_op_with_descr(OpCode::RawStore, &[r0, off8, i1], ad.clone());
    // Load from each offset
    let l0 = rec.record_op_with_descr(OpCode::RawLoadI, &[r0, off0], ad.clone());
    let l1 = rec.record_op_with_descr(OpCode::RawLoadI, &[r0, off8], ad.clone());
    // Add the two loaded values together
    let sum = rec.record_op(OpCode::IntAdd, &[l0, l1]);
    rec.finish(&[sum], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(602);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("multi-offset raw ops should compile");

    let mut buf = vec![0u8; 32];
    let ptr = buf.as_mut_ptr() as usize;

    let frame = backend.execute_token(
        &token,
        &[Value::Ref(GcRef(ptr)), Value::Int(100), Value::Int(200)],
    );
    assert_eq!(
        backend.get_int_value(&frame, 0),
        300,
        "100 at offset 0 + 200 at offset 8 should yield 300"
    );
}

// Test: RawLoadI with unsigned 1-byte item (zero extension)
//
// Mirrors RPython's test_raw_storage_byte: store 0xFF in a 1-byte item,
// load it as unsigned (should get 255, not -1).

#[test]
fn test_raw_load_unsigned_byte() {
    let ad: DescrRef = Arc::new(RawArrayDescr {
        item_size: 1,
        item_type: Type::Int,
        signed: false,
    });
    let const_offset = OpRef::const_int(0);

    let mut rec = Trace::new();
    let r0 = rec.record_input_arg(Type::Ref);

    let loaded = rec.record_op_with_descr(OpCode::RawLoadI, &[r0, const_offset], ad);
    rec.finish(&[loaded], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(603);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("unsigned byte raw load should compile");

    let mut data = vec![0xFFu8];
    let ptr = data.as_mut_ptr() as usize;

    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        255,
        "unsigned 1-byte load of 0xFF should yield 255"
    );
}

// FFI forcing and guard_not_forced parity tests
// (rpython/jit/metainterp/test/test_fficall.py: test_guard_not_forced_fails)
//
// Verifies CallMayForceI + GuardNotForced and CallReleaseGilI + GuardNoException
// compile and execute correctly, covering the forced path and exception
// propagation semantics.

fn call_descr_may_force_i(idx: u32, arg_types: Vec<Type>) -> DescrRef {
    Arc::new(TestCallDescr {
        idx,
        effect: majit_ir::EffectInfo {
            extraeffect: majit_ir::ExtraEffect::CanRaise,
            oopspecindex: majit_ir::OopSpecIndex::None,
            ..Default::default()
        },
        arg_types,
        result_type: Type::Int,
    })
}

/// FFI function that never forces: returns a + b.
extern "C" fn ffi_add_no_force(force_token: i64, a: i64) -> i64 {
    let _ = force_token; // not used — no forcing
    a * 2
}

/// FFI function that conditionally forces the frame.
/// When flag != 0, forces via force_token_to_dead_frame.
extern "C" fn ffi_maybe_force(force_token: i64, flag: i64) -> i64 {
    if flag != 0 {
        let _deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
    }
    flag * 2
}

/// OBJECTPTR layout: typeptr at offset 0 (pyjitpl.py raise_if_successful).
#[repr(C)]
struct FakeExcObject {
    typeptr: usize,
}

/// FFI function that sets a pending exception via jit_exc_raise.
/// Allocates a fake OBJECTPTR so typeptr dereference works.
extern "C" fn ffi_raise_exception(val: i64) -> i64 {
    if val != 0 {
        let obj = Box::leak(Box::new(FakeExcObject { typeptr: 0xEE }));
        jit_exc_raise(obj as *const FakeExcObject as usize as i64);
    }
    val
}

#[test]
fn test_call_release_gil_with_guard_not_forced() {
    // Parity with test_fficall.test_guard_not_forced_fails (non-forced path).
    //
    // Trace: input(x) -> force_token -> result = call_may_force_i(fn, token, x)
    //        -> guard_not_forced(fail_args=[x, result]) -> finish(result)
    //
    // When the called function does NOT force, GuardNotForced passes and
    // Finish returns the call result.
    let cd = call_descr_may_force_i(70, vec![Type::Ref, Type::Int]);

    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);

    // ForceToken produces a Ref-typed value (the current frame handle)
    let token_ref = rec.record_op(OpCode::ForceToken, &[]);

    // fn_ptr is stored as a constant
    let fn_ptr = OpRef::const_int(ffi_add_no_force as *const () as usize as i64);

    // CallMayForceI: arg(0)=fn_ptr, rest=[force_token, x]
    let result = rec.record_op_with_descr(OpCode::CallMayForceI, &[fn_ptr, token_ref, x], cd);

    // GuardNotForced with fail_args — exits here if the callee forced
    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(0)));

    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[x, result]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(700);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("CallMayForceI + GuardNotForced should compile");

    // Not forced: guard passes, finish returns result = x * 2
    let frame = backend.execute_token(&token, &[Value::Int(21)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "should reach Finish"
    );
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "ffi_add_no_force(token, 21) should return 42"
    );

    // Another input
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        0,
        "ffi_add_no_force(token, 0) should return 0"
    );
}

#[test]
fn test_call_may_force_with_forcing_semantics() {
    // Parity with test_fficall.test_guard_not_forced_fails (forced path).
    //
    // Trace: input(flag) -> force_token -> result = call_may_force_i(fn, token, flag)
    //        -> guard_not_forced(fail_args=[flag, result]) -> finish(result)
    //
    // When flag != 0 the callee forces via force_token_to_dead_frame,
    // so GuardNotForced fails and we exit through the guard's fail path.
    let cd = call_descr_may_force_i(71, vec![Type::Ref, Type::Int]);

    let mut rec = Trace::new();
    let flag = rec.record_input_arg(Type::Int);

    let token_ref = rec.record_op(OpCode::ForceToken, &[]);
    let fn_ptr = OpRef::const_int(ffi_maybe_force as *const () as usize as i64);

    let result = rec.record_op_with_descr(OpCode::CallMayForceI, &[fn_ptr, token_ref, flag], cd);

    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(0)));

    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[flag, result]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(701);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("CallMayForceI + GuardNotForced (forced) should compile");

    // flag=0 -> not forced -> reaches Finish
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "flag=0: should reach Finish"
    );
    assert_eq!(
        backend.get_int_value(&frame, 0),
        0,
        "flag=0: result = 0*2 = 0"
    );

    // flag=1 -> forced -> exits via GuardNotForced (descr 0)
    let frame = backend.execute_token(&token, &[Value::Int(1)]);
    assert_eq!(
        backend.get_latest_descr(&frame).fail_index(),
        0,
        "flag=1: should exit via GuardNotForced (descr 0)"
    );
    // fail_args are [flag, result]; result is a preview snapshot taken
    // before the call (placeholder zero for forward-defined values).
    assert_eq!(
        backend.get_int_value(&frame, 0),
        1,
        "flag=1: fail_args[0] = flag = 1"
    );
}

#[test]
fn test_ffi_call_exception_propagation() {
    // Parity with test_fficall exception semantics.
    // RPython: CallReleaseGilI -> GuardNotForced -> GuardNoException
    //
    // Trace: input(val) -> result = call_release_gil_i(saveerr, ffi_raise_exception, val)
    //        -> guard_not_forced(fail_args=[val, result])
    //        -> guard_no_exception(fail_args=[result]) -> finish(result)
    //
    // When val != 0, ffi_raise_exception sets a pending exception via
    // jit_exc_raise, so GuardNoException fails and exits through the guard.
    let cd = call_descr_release_gil_i(72, vec![Type::Int]);

    let mut rec = Trace::new();
    let val = rec.record_input_arg(Type::Int);

    let saveerr = OpRef::const_int(0);
    let fn_ptr = OpRef::const_int(ffi_raise_exception as *const () as usize as i64);
    let result = rec.record_op_with_descr(OpCode::CallReleaseGilI, &[saveerr, fn_ptr, val], cd);

    // GuardNotForced: required immediately after CallReleaseGil
    let not_forced = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(2)));
    // GuardNoException: exits if jit_exc_get_value() != 0
    let no_exc = rec.record_guard(OpCode::GuardNoException, &[], Some(make_descr(0)));

    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, not_forced, &[val, result]);
    set_guard_failargs(&trace, no_exc, &[result]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(702);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("CallReleaseGilI + GuardNotForced + GuardNoException should compile");

    // collect_guards: GuardNotForced=0, GuardNoException=1, Finish=2
    // val=0 -> no exception -> reaches Finish
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "val=0: should reach Finish"
    );
    assert_eq!(backend.get_int_value(&frame, 0), 0, "val=0: result = 0");

    // val=99 -> exception raised -> exits via GuardNoException (fail_index=1)
    let frame = backend.execute_token(&token, &[Value::Int(99)]);
    assert_eq!(
        backend.get_latest_descr(&frame).fail_index(),
        1,
        "val=99: should exit via GuardNoException (fail_index=1)"
    );
    assert_eq!(
        backend.get_int_value(&frame, 0),
        99,
        "val=99: fail_args[0] = result = 99"
    );
}

// Frame-stack metadata integration tests

// Test: Guard failure preserves frame-stack metadata in describe_deadframe

#[test]
fn test_compiled_guard_failure_preserves_frame_stack_metadata() {
    // Trace: input(x) -> result = x + 5 -> cmp = result < 100
    //        -> guard_true(cmp) -> finish(result)
    // Execute with x=50: guard passes, finish returns 55.
    // Execute with x=200: result=205, 205 < 100 is false, guard fails.
    // Verify the DeadFrame carries frame_stack metadata.
    let mut rec = Trace::new();
    let x = rec.record_input_arg(Type::Int);

    let const_5 = OpRef::const_int(5);
    let const_100 = OpRef::const_int(100);

    let result = rec.record_op(OpCode::IntAdd, &[x, const_5]);
    let cmp = rec.record_op(OpCode::IntLt, &[result, const_100]);
    let guard = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(make_descr(0)));
    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[x]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(900);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // x=50: guard passes (55 < 100), reaches Finish
    let frame = backend.execute_token(&token, &[Value::Int(50)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish(), "x=50 should reach Finish");
    assert_eq!(backend.get_int_value(&frame, 0), 55);

    // x=200: guard fails (205 < 100 is false)
    let frame = backend.execute_token(&token, &[Value::Int(200)]);
    let descr = backend.get_latest_descr(&frame);
    assert_eq!(descr.fail_index(), 0, "x=200 should fail guard");
    // Guard saves input args: x=200
    assert_eq!(backend.get_int_value(&frame, 0), 200);

    // Backend's `describe_deadframe` no longer caches the recovery
    // layout — `frame_stack`/`recovery_layout` come from the
    // metainterp's `StoredExitLayout.recovery_layout` (consumed via
    // `trace_layout_ref.recovery_layout` at `pyjitpl.rs`).
    // Without a pyre-jit boot here, the backend reports None; this
    // matches the dynasm contract (see `runner.rs` describe_deadframe
    // default impl returning recovery_layout=None).
    let layout = backend
        .describe_deadframe(&frame)
        .expect("describe_deadframe should return a layout");
    assert_eq!(layout.fail_index, 0);
    assert!(
        layout.frame_stack.is_none(),
        "backend returns None; metainterp owns recovery layout"
    );
}

// Test: Bridge guard failure carries frame-stack metadata

#[test]
fn test_compiled_bridge_guard_failure_has_frame_stack() {
    // Main loop: input(i, sum) -> sum2 = sum + i -> i2 = i - 1
    //            -> cmp = i2 > 0 -> guard_true(cmp) -> jump(i2, sum2)
    let mut rec = Trace::new();
    let i = rec.record_input_arg(Type::Int);
    let sum = rec.record_input_arg(Type::Int);

    let const_one = OpRef::const_int(1);
    let const_zero = OpRef::const_int(0);

    let sum2 = rec.record_op(OpCode::IntAdd, &[sum, i]);
    let i2 = rec.record_op(OpCode::IntSub, &[i, const_one]);
    let cmp = rec.record_op(OpCode::IntGt, &[i2, const_zero]);
    let guard = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(make_descr(0)));
    rec.close_loop(&[i2, sum2]);
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[i, sum]);

    let mut backend = CraneliftBackend::new();

    backend.set_next_trace_id(910);
    backend.set_next_header_pc(1000);
    let token = JitCellToken::new(902);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("main loop compilation should succeed");

    // Set up recovery layout on the guard with 2 frames so the bridge
    // inherits a caller prefix. compile_bridge pops the last frame from the
    // source guard's recovery layout to form the caller prefix, then the
    // bridge's own frame is appended, giving 2 frames total.
    let source_layout = ExitRecoveryLayout {
        vable_array: vec![],
        vref_array: vec![],
        frames: vec![
            ExitFrameLayout {
                trace_id: Some(909),
                header_pc: Some(500),
                source_guard: None,
                pc: 500,
                jitcode_index: 0,
                slots: vec![ExitValueSourceLayout::Constant(0, Type::Int)],
                slot_types: Some(vec![Type::Int]),
            },
            ExitFrameLayout {
                trace_id: Some(910),
                header_pc: Some(1000),
                source_guard: Some((909, 0)),
                pc: 1000,
                jitcode_index: 0,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(0),
                    ExitValueSourceLayout::ExitValue(1),
                ],
                slot_types: Some(vec![Type::Int, Type::Int]),
            },
        ],
        virtual_layouts: vec![],
        pending_field_layouts: vec![],
    };
    // Backend no longer caches recovery_layout; the source
    // layout flows through `compile_bridge`'s explicit
    // `caller_recovery_layout` parameter (QQ-7 contract).

    // Bridge: takes (i, sum), checks sum > 0, returns sum * 2
    let mut bridge_rec = Trace::new();
    let _bi = bridge_rec.record_input_arg(Type::Int);
    let bsum = bridge_rec.record_input_arg(Type::Int);

    let bridge_const_zero = OpRef::const_int(0);
    let bridge_const_two = OpRef::const_int(2);

    let bcmp = bridge_rec.record_op(OpCode::IntGt, &[bsum, bridge_const_zero]);
    let bridge_guard = bridge_rec.record_guard(OpCode::GuardTrue, &[bcmp], Some(make_descr(10)));
    let bresult = bridge_rec.record_op(OpCode::IntMul, &[bsum, bridge_const_two]);
    bridge_rec.finish(&[bresult], make_descr(11));
    let bridge_trace = bridge_rec.get_trace();
    set_guard_failargs(&bridge_trace, bridge_guard, &[_bi, bsum]);

    // Mint a `ResumeGuardDescr` that mirrors the source guard's identity
    // (per-trace fail_index = 0, trace_id = 910). Bridge compilation
    // looks up the parent guard by `FailDescr::{trace_id,
    // fail_index_per_trace}` from the descr passed by the metainterp.
    let bridge_fail_descr =
        majit_backend::make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
    {
        let fd = bridge_fail_descr.as_fail_descr().unwrap();
        fd.set_trace_id(910);
        fd.set_fail_index_per_trace(0);
    }
    let bridge_fail_descr_ref = bridge_fail_descr.as_fail_descr().unwrap();

    backend.set_next_trace_id(911);
    backend.set_next_header_pc(2000);
    let _bridge_info = backend
        .compile_bridge(
            bridge_fail_descr_ref,
            &inputargs_view(&bridge_trace),
            &bridge_trace.ops,
            &token,
            Some(&source_layout),
        )
        .expect("bridge compilation should succeed");

    // Execute: i=3, sum=0 -> loop runs 3 iters -> guard fails with i=1, sum=5
    // Bridge runs: sum=5 > 0 passes, returns 5 * 2 = 10
    let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 10);

    // Now test bridge guard failure: i=3, sum=-100
    //   iter1: i=3,sum=-100 -> sum2=-97, i2=2, pass -> jump(2,-97)
    //   iter2: i=2,sum=-97 -> sum2=-95, i2=1, pass -> jump(1,-95)
    //   iter3: i=1,sum=-95 -> sum2=-94, i2=0, fail -> bridge
    // Bridge: sum=-95, -95 > 0 is false -> bridge guard fails
    let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(-100)]);
    let descr = backend.get_latest_descr(&frame);
    // Bridge guard failure should have a fail_index from the bridge
    assert!(!descr.is_finish(), "bridge guard should fail, not finish");

    // See test_compiled_guard_failure_preserves_frame_stack_metadata.
    let layout = backend
        .describe_deadframe(&frame)
        .expect("bridge guard failure should produce a layout");
    assert!(
        layout.frame_stack.is_none(),
        "backend returns None; metainterp owns recovery layout"
    );
}

// Test: CallAssemblerI callee guard failure propagates frame_stack

#[test]
fn test_call_assembler_callee_guard_failure_frame_stack() {
    // Compile a callee trace with a guard that fails:
    //   input(x) -> cmp = x > 10 -> guard_true(cmp) -> finish(x)
    let callee_inputargs = vec![InputArg::new_int_rc(0)];
    let mut callee_ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        Op::new(
            OpCode::IntGt,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(10))],
        ),
        Op::with_descr(OpCode::GuardTrue, &[rb(OpRef::int_op(1))], make_descr(0)),
        Op::with_descr(
            OpCode::Finish,
            &[rb(OpRef::input_arg_int(0))],
            make_descr(1),
        ),
    ];
    assign_positions(&mut callee_ops, 0);
    callee_ops[2].setfailargs(vec![rb(OpRef::input_arg_int(0))].into());
    let callee_ops_rc: Vec<OpRc> = callee_ops.into_iter().map(OpRc::new).collect();

    let mut backend = CraneliftBackend::new();

    backend.set_next_trace_id(920);
    backend.set_next_header_pc(3000);
    let callee_token = JitCellToken::new(903);
    backend
        .compile_loop(&callee_inputargs, &callee_ops_rc, &callee_token)
        .expect("callee compilation should succeed");

    // x=20: guard passes, finish
    let frame = backend.execute_token(&callee_token, &[Value::Int(20)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 20);

    // x=5: guard fails (5 > 10 is false)
    let frame = backend.execute_token(&callee_token, &[Value::Int(5)]);
    let descr = backend.get_latest_descr(&frame);
    assert_eq!(descr.fail_index(), 0);
    assert_eq!(backend.get_int_value(&frame, 0), 5);

    // See test_compiled_guard_failure_preserves_frame_stack_metadata.
    let layout = backend
        .describe_deadframe(&frame)
        .expect("callee guard failure should have a layout");
    assert!(
        layout.frame_stack.is_none(),
        "backend returns None; metainterp owns recovery layout"
    );
}

// Test: Frame-stack slot_types match fail_arg_types for mixed Int+Float

#[test]
fn test_frame_stack_slot_types_match_fail_arg_types() {
    // Trace with mixed Int and Float fail args:
    //   input(x_int, x_float) -> cmp = x_int > 0 -> guard_true(cmp, fail_args=[x_int, x_float])
    //   -> finish(x_int)
    let mut rec = Trace::new();
    let x_int = rec.record_input_arg(Type::Int);
    let x_float = rec.record_input_arg(Type::Float);

    let const_0 = OpRef::const_int(0);

    let cmp = rec.record_op(OpCode::IntGt, &[x_int, const_0]);
    let guard = rec.record_guard(OpCode::GuardTrue, &[cmp], Some(make_descr(0)));
    rec.finish(&[x_int], make_descr(1));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[x_int, x_float]);

    let mut backend = CraneliftBackend::new();

    backend.set_next_trace_id(930);
    backend.set_next_header_pc(5000);
    let token = JitCellToken::new(904);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("compilation should succeed");

    // x_int=10, x_float=3.14: guard passes, finish returns 10
    let frame = backend.execute_token(&token, &[Value::Int(10), Value::Float(3.25)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 10);

    // x_int=-5, x_float=2.5: guard fails (-5 > 0 is false)
    let frame = backend.execute_token(&token, &[Value::Int(-5), Value::Float(2.5)]);
    let descr = backend.get_latest_descr(&frame);
    assert_eq!(descr.fail_index(), 0);
    assert_eq!(backend.get_int_value(&frame, 0), -5);
    assert!(
        (backend.get_float_value(&frame, 1) - 2.5).abs() < 1e-10,
        "float fail arg should be preserved"
    );

    // fail_arg_types still flow through `FailDescr` directly;
    // frame_stack metadata is now metainterp-owned (see
    // test_compiled_guard_failure_preserves_frame_stack_metadata).
    let layout = backend
        .describe_deadframe(&frame)
        .expect("mixed-type guard failure should have a layout");
    assert_eq!(
        layout.fail_arg_types,
        vec![Type::Int, Type::Float],
        "fail_arg_types should be [Int, Float]"
    );
    assert!(
        layout.frame_stack.is_none(),
        "backend returns None; metainterp owns recovery layout"
    );
}

// Test: FFI exchange buffer pattern
//
// Parity with rpython/jit/metainterp/test/test_fficall.py lines 240-290.
// Simulates the libffi exchange buffer protocol:
//   1. RawStore arguments to buffer at known offsets (exchange_args)
//   2. CallReleaseGilI (the FFI callee reads from buffer, writes result)
//   3. RawLoadI result from buffer at exchange_result offset
//
// The buffer layout follows RPython's CIF description convention:
//   offset 16 = first argument slot
//   offset 32 = result slot

/// FFI function that reads from an exchange buffer and writes back.
/// Simulates `fake_call_impl_any` from test_fficall.py:
///   reads arg at offset 16, computes arg * 2, writes result at offset 32.
extern "C" fn ffi_exchange_buffer_fn(buf_ptr: i64) -> i64 {
    let buf = buf_ptr as *mut u8;
    unsafe {
        let arg = *(buf.add(16) as *const i64);
        let result = arg * 2;
        *(buf.add(32) as *mut i64) = result;
    }
    0 // return value unused; result is in the buffer
}

#[test]
fn test_ffi_exchange_buffer_pattern() {
    // Simulate FFI exchange buffer:
    // 1. RawStore arguments to buffer at known offsets
    // 2. CallReleaseGilI (the FFI call reads from buffer)
    // 3. RawLoadI result from buffer
    // This is the pattern RPython uses for libffi calls.

    let ad = raw_descr_int(8);
    let cd = call_descr_release_gil_i(80, vec![Type::Ref]);

    // Constants: offset_16 = 16 (exchange_args[0]), offset_32 = 32 (exchange_result)
    let off_arg = OpRef::const_int(16); // offset 16
    let off_result = OpRef::const_int(32); // offset 32
    let saveerr = OpRef::const_int(0); // CALL_RELEASE_GIL saveerr flag
    let fn_ptr = OpRef::const_int(ffi_exchange_buffer_fn as *const () as usize as i64);

    let mut rec = Trace::new();
    // Inputs: r0 = exchange buffer pointer, i0 = argument value
    let r0 = rec.record_input_arg(Type::Ref);
    let i0 = rec.record_input_arg(Type::Int);

    // Step 1: Store argument into buffer at offset 16 (exchange_args[0])
    rec.record_op_with_descr(OpCode::RawStore, &[r0, off_arg, i0], ad.clone());

    // Step 2: Call the FFI function with buffer pointer
    let _call_result =
        rec.record_op_with_descr(OpCode::CallReleaseGilI, &[saveerr, fn_ptr, r0], cd);
    // RPython: CallReleaseGilI must be followed by GuardNotForced
    let guard = rec.record_guard(OpCode::GuardNotForced, &[], Some(make_descr(1)));

    // Step 3: Load result from buffer at offset 32 (exchange_result)
    let loaded = rec.record_op_with_descr(OpCode::RawLoadI, &[r0, off_result], ad);

    rec.finish(&[loaded], make_descr(0));
    let trace = rec.get_trace();
    set_guard_failargs(&trace, guard, &[r0, i0]);

    let mut backend = CraneliftBackend::new();

    let token = JitCellToken::new(950);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .expect("FFI exchange buffer pattern should compile");

    // Allocate a 48-byte exchange buffer (matching RPython's exbuf allocation)
    let mut exbuf = vec![0u8; 48];
    let ptr = exbuf.as_mut_ptr() as usize;

    // Execute with arg=25: store 25 at offset 16, FFI computes 25*2=50, load from offset 32
    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(25)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        50,
        "exchange buffer: arg=25 -> result=50 (25*2)"
    );

    // Verify the buffer contents directly
    let result_in_buf = unsafe { *(exbuf.as_ptr().add(32) as *const i64) };
    assert_eq!(result_in_buf, 50, "buffer at offset 32 should contain 50");

    // Execute with another value: arg=0
    exbuf.fill(0);
    let ptr = exbuf.as_mut_ptr() as usize;
    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(0)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        0,
        "exchange buffer: arg=0 -> result=0"
    );

    // Execute with negative value: arg=-10
    exbuf.fill(0);
    let ptr = exbuf.as_mut_ptr() as usize;
    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(-10)]);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        -20,
        "exchange buffer: arg=-10 -> result=-20"
    );
}

/// `x86/assembler.py redirect_call_assembler`: overlay JMP at the old
/// `_ll_function_addr` and leave the slot pointing at that overlay site.
#[test]
fn test_redirect_call_assembler_overlays_old_entry() {
    fn compile_add(backend: &mut CraneliftBackend, token: &JitCellToken, addend: i64) {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let k = OpRef::const_int(addend);
        let result = rec.record_op(OpCode::IntAdd, &[i0, k]);
        rec.finish(&[result], make_descr(0));
        let trace = rec.get_trace();
        backend
            .compile_loop(&inputargs_view(&trace), &trace.ops, token)
            .expect("compilation should succeed");
    }

    let mut backend = CraneliftBackend::new();
    let old = JitCellToken::new(501);
    let new = JitCellToken::new(502);
    compile_add(&mut backend, &old, 1);
    compile_add(&mut backend, &new, 100);
    let old_addr = old.ll_function_addr();
    let new_addr = new.ll_function_addr();
    assert!(old_addr != 0 && new_addr != 0 && old_addr != new_addr);

    backend
        .redirect_call_assembler(&old, &new)
        .expect("redirect");

    // Slot stays the overlay site (`assembler.py` `oldadr`).
    assert_eq!(old.ll_function_addr(), old_addr);

    #[cfg(target_arch = "x86_64")]
    {
        let first = unsafe { std::ptr::read(old_addr as *const u8) };
        assert!(first == 0xE9 || first == 0x49, "JMP rel32 or MOV r11,imm64");
    }
    #[cfg(target_arch = "aarch64")]
    {
        let word = unsafe { std::ptr::read(old_addr as *const u32) };
        // B imm26 or MOVZ x16 (far veneer).
        assert!(
            word & 0xFC00_0000 == 0x1400_0000 || word & 0xFFC0_001F == 0xD280_0010,
            "B or MOVZ x16, got {word:#010x}"
        );
    }

    let frame = backend.execute_token(&old, &[Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 100);

    let newer = JitCellToken::new(503);
    compile_add(&mut backend, &newer, 7);
    backend
        .redirect_call_assembler(&old, &newer)
        .expect("second redirect of the same token");
    assert_eq!(old.ll_function_addr(), old_addr);
    let frame = backend.execute_token(&old, &[Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 7);
}

/// `runner_test.py test_redirect_call_assembler`: a CALL_ASSEMBLER compiled
/// against one token still reaches a body installed later by
/// `redirect_call_assembler`, including a second overlay of that token.
#[test]
fn test_call_assembler_follows_redirect_after_caller_compiled() {
    fn compile_add(backend: &mut CraneliftBackend, token: &JitCellToken, addend: i64) {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let k = OpRef::const_int(addend);
        let result = rec.record_op(OpCode::IntAdd, &[i0, k]);
        rec.finish(&[result], make_descr(0));
        let trace = rec.get_trace();
        backend
            .compile_loop(&inputargs_view(&trace), &trace.ops, token)
            .expect("callee compilation should succeed");
    }

    fn make_call_assembler_backend() -> CraneliftBackend {
        use majit_backend_cranelift::{JitFrameLayoutInfo, register_jitframe_layout};
        use majit_gc::GcAllocator;
        use majit_gc::collector::{GcConfig, MiniMarkGC};
        use majit_gc::trace::TypeInfo;

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        gc.set_jitframe_type_id(tid);
        let backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        register_jitframe_layout(JitFrameLayoutInfo {
            jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
                jitframe_tid: tid,
                jitframe_fixed_size: majit_backend::jitframe::JITFRAME_FIXED_SIZE,
                jf_frame_info_ofs: majit_backend::jitframe::JF_FRAME_INFO_OFS,
                jf_descr_ofs: majit_backend::jitframe::JF_DESCR_OFS,
                jf_force_descr_ofs: majit_backend::jitframe::JF_FORCE_DESCR_OFS,
                jf_savedata_ofs: majit_backend::jitframe::JF_SAVEDATA_OFS,
                jf_guard_exc_ofs: majit_backend::jitframe::JF_GUARD_EXC_OFS,
                jf_forward_ofs: majit_backend::jitframe::JF_FORWARD_OFS,
                jf_frame_ofs: majit_backend::jitframe::JF_FRAME_OFS,
                jf_frame_baseitemofs: majit_backend::jitframe::FIRST_ITEM_OFFSET,
                jf_frame_lengthofs: majit_backend::jitframe::JF_FRAME_OFS
                    + majit_backend::jitframe::LENGTHOFS,
                sign_size: majit_backend::jitframe::SIGN_SIZE,
                jf_frame_itemsize: majit_backend::jitframe::SIGN_SIZE,
            }),
        });
        backend
    }

    let mut backend = make_call_assembler_backend();
    let callee = Arc::new(JitCellToken::new(1_500_600));
    compile_add(&mut backend, callee.as_ref(), 1);

    let mut caller_rec = Trace::new();
    let i0 = caller_rec.record_input_arg(Type::Int);
    let ca = caller_rec.record_op_with_descr(
        OpCode::CallAssemblerI,
        &[i0],
        Arc::new(majit_backend_cranelift::CallAssemblerDescr::new(
            Arc::clone(&callee),
            vec![Type::Int],
            Type::Int,
        )),
    );
    caller_rec.finish(&[ca], make_descr(1));
    let caller_trace = caller_rec.get_trace();
    let caller = JitCellToken::new(1_500_601);
    backend
        .compile_loop(&inputargs_view(&caller_trace), &caller_trace.ops, &caller)
        .expect("caller compilation should succeed");

    let frame = backend.execute_token(&caller, &[Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 1);

    let replacement = JitCellToken::new(1_500_602);
    compile_add(&mut backend, &replacement, 100);
    let overlay = callee.ll_function_addr();
    backend
        .redirect_call_assembler(callee.as_ref(), &replacement)
        .expect("redirect after caller compiled");
    assert_eq!(callee.ll_function_addr(), overlay);
    let frame = backend.execute_token(&caller, &[Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 100);

    let second = JitCellToken::new(1_500_603);
    compile_add(&mut backend, &second, 7);
    backend
        .redirect_call_assembler(callee.as_ref(), &second)
        .expect("second redirect of the same token");
    assert_eq!(callee.ll_function_addr(), overlay);
    let frame = backend.execute_token(&caller, &[Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 7);
}
