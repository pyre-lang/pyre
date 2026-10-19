//! x86 CAST_FLOAT_TO_SINGLEFLOAT / CAST_SINGLEFLOAT_TO_FLOAT through the
//! backend public API. Inputs are arguments so the casts are not folded.
//! Only x86/runner.py sets `supports_singlefloats = True`; AArch64 keeps
//! model.py's `False` and never sees these casts.
#![cfg(target_arch = "x86_64")]

use majit_backend::{Backend, JitCellToken};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::{InputArg, Op, OpCode, OpRc, OpRef, Type, Value};

fn float_to_singlefloat_bits(backend: &mut DynasmBackend, token_id: u64, input: f64) -> i64 {
    let token = JitCellToken::new(token_id as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let f0 = inputargs[0].opref();
    let cast = Op::new(OpCode::CastFloatToSinglefloat, &[rb(f0)]);
    cast.pos().set(OpRef::int_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
    let ops = vec![OpRc::new(cast), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CAST_FLOAT_TO_SINGLEFLOAT: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Float(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_int_value(&frame, 0)
}

fn singlefloat_bits_to_float(backend: &mut DynasmBackend, token_id: u64, bits: i64) -> f64 {
    let token = JitCellToken::new(token_id as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let cast = Op::new(OpCode::CastSinglefloatToFloat, &[rb(i0)]);
    cast.pos().set(OpRef::float_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Float]);
    finish.setfailargs(vec![rb(OpRef::float_op(1))].into());
    let ops = vec![OpRc::new(cast), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CAST_SINGLEFLOAT_TO_FLOAT: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(bits)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_float_value(&frame, 0)
}

fn chained_cast(backend: &mut DynasmBackend, token_id: u64, input: f64) -> f64 {
    let token = JitCellToken::new(token_id as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let f0 = inputargs[0].opref();
    let to_sf = Op::new(OpCode::CastFloatToSinglefloat, &[rb(f0)]);
    to_sf.pos().set(OpRef::int_op(1));
    let to_f = Op::new(OpCode::CastSinglefloatToFloat, &[rb(OpRef::int_op(1))]);
    to_f.pos().set(OpRef::float_op(2));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(2))]);
    finish.pos().set(OpRef::void_op(3));
    finish.set_fail_arg_types(vec![Type::Float]);
    finish.setfailargs(vec![rb(OpRef::float_op(2))].into());
    let ops = vec![OpRc::new(to_sf), OpRc::new(to_f), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile chained singlefloat cast: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Float(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_float_value(&frame, 0)
}

#[test]
fn singlefloat_casts_round_trip_bits() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token_id = 1u64;
    let cases = [1.5f64, 0.1, -0.0];
    for input in cases {
        let expect_bits = i64::from((input as f32).to_bits());
        let got_bits = float_to_singlefloat_bits(&mut backend, token_id, input);
        token_id += 1;
        assert_eq!(got_bits, expect_bits, "f32 bits of {input:?}");

        let got = singlefloat_bits_to_float(&mut backend, token_id, expect_bits);
        token_id += 1;
        let expect = f64::from(input as f32);
        assert_eq!(
            got.to_bits(),
            expect.to_bits(),
            "CAST_SINGLEFLOAT_TO_FLOAT of {expect_bits:#x}"
        );
    }
    assert_eq!(
        float_to_singlefloat_bits(&mut backend, token_id, 0.1),
        i64::from(0.1f32.to_bits()),
        "0.1 must round to 0.1f32 bits"
    );
    token_id += 1;
    let minus_zero_bits = float_to_singlefloat_bits(&mut backend, token_id, -0.0);
    token_id += 1;
    assert_eq!(
        minus_zero_bits,
        i64::from((-0.0f32).to_bits()),
        "-0.0 must keep the f32 sign bit"
    );
    let from_point_one_bits =
        singlefloat_bits_to_float(&mut backend, token_id, i64::from(0.1f32.to_bits()));
    token_id += 1;
    assert_eq!(
        from_point_one_bits.to_bits(),
        (0.1f32 as f64).to_bits(),
        "0.1f32 bits expand to 0.1f32 as f64"
    );
    assert_ne!(
        from_point_one_bits.to_bits(),
        0.1f64.to_bits(),
        "0.1 must take the f32 rounding, not pass the f64 through"
    );

    let chained = chained_cast(&mut backend, token_id, 0.1);
    assert_eq!(
        chained.to_bits(),
        (0.1f32 as f64).to_bits(),
        "chained cast of 0.1"
    );
    assert_ne!(chained.to_bits(), 0.1f64.to_bits());
}
