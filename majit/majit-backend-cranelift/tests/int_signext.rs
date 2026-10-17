//! `int_signext` byte-count lowering against `support.py int_signext`.

use majit_backend::{Backend, JitCellToken};
use majit_backend_cranelift::CraneliftBackend;
use majit_ir::test_support::{RecordedTrace, Trace};
use majit_ir::{DescrRef, InputArgRc, OpCode, OpRef, Type, Value};

fn signext(value: i64, numbytes: i64) -> i64 {
    let shift = 64 - numbytes * 8;
    (value << shift) >> shift
}

fn inputargs_view(trace: &RecordedTrace) -> Vec<InputArgRc> {
    trace.inputargs.clone()
}

fn make_descr(_index: u32) -> DescrRef {
    majit_backend::make_resume_guard_descr_typed(Vec::new())
}

fn compiled_signext(
    backend: &mut CraneliftBackend,
    token_id: u64,
    numbytes: i64,
    value: i64,
) -> i64 {
    let mut rec = Trace::new();
    let i0 = rec.record_input_arg(Type::Int);
    let result = rec.record_op(OpCode::IntSignext, &[i0, OpRef::const_int(numbytes)]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();
    let token = JitCellToken::new(token_id as i64);
    backend
        .compile_loop(&inputargs_view(&trace), &trace.ops, &token)
        .unwrap_or_else(|err| panic!("compile int_signext({numbytes}): {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(value)]);
    backend.get_int_value(&frame, 0)
}

#[test]
fn int_signext_matches_support() {
    let mut backend = CraneliftBackend::new();
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
