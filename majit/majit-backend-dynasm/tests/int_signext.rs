//! Host-arch `int_signext` against `support.py int_signext` (byte count 1/2/4).

mod common;
use majit_backend::{Backend, JitCellToken};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::{InputArg, Op, OpCode, OpRc, OpRef, Type, Value};

fn signext(value: i64, numbytes: i64) -> i64 {
    let shift = 64 - numbytes * 8;
    (value << shift) >> shift
}

fn compiled_signext(backend: &mut DynasmBackend, token_id: u64, numbytes: i64, value: i64) -> i64 {
    let token = JitCellToken::new(token_id as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let sign = Op::new(
        OpCode::IntSignext,
        &[rb(i0), rb(OpRef::const_int(numbytes))],
    );
    sign.pos().set(OpRef::int_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
    let ops = vec![OpRc::new(sign), OpRc::new(finish)];
    common::rebind_trace_arg_boxes(&inputargs, &ops);
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile int_signext({numbytes}): {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(value)]);
    backend.get_int_value(&frame, 0)
}

#[test]
fn int_signext_matches_support() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let values = [
        0x80,
        0x7f,
        0xff,
        0x8000,
        0xffff,
        0x8000_0000,
        0xffff_ffff,
        0x1234_5680,
    ];
    let mut token_id = 1u64;
    for numbytes in [1i64, 2, 4] {
        for value in values {
            let got = compiled_signext(&mut backend, token_id, numbytes, value);
            token_id += 1;
            assert_eq!(
                got,
                signext(value, numbytes),
                "int_signext({value:#x}, {numbytes})"
            );
        }
    }
}
