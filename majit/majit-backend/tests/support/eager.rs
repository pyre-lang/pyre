//! Shared native-backend conformance tests. No recorder, JitDriver or warmup.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use majit_backend::eager::{ArgumentError, CompiledIr};
use majit_backend::{Backend, BackendError, JitCellToken};
use majit_ir::descr::make_finish_descr;
use majit_ir::operand::Operand;
use majit_ir::{ConstMap, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};

fn token() -> Arc<JitCellToken> {
    static NEXT: AtomicU64 = AtomicU64::new(1_000_000);
    Arc::new(JitCellToken::new(
        NEXT.fetch_add(1, Ordering::Relaxed) as i64
    ))
}

fn backend() -> Box<dyn Backend> {
    let mut backend = super::new_backend();
    // The embedding, not CompiledIr, owns AbstractCPU setup and the
    // compile.make_and_attach_done_descrs / propagate-exception attachments.
    majit_backend::make_and_attach_done_descrs(&mut [&mut *backend]);
    backend.set_propagate_exception_descr(Arc::new(majit_backend::PropagateExceptionDescr::new()));
    backend.setup_once();
    backend
}

fn op(opcode: OpCode, args: &[Operand], pos: u32) -> OpRc {
    let op = Op::new(opcode, args);
    op.pos().set(OpRef::op_typed(pos, opcode.result_type()));
    OpRc::new(op)
}

fn finish(args: &[Operand], types: Vec<Type>) -> OpRc {
    OpRc::new(Op::with_descr(
        OpCode::Finish,
        args,
        make_finish_descr(90, types),
    ))
}

#[test]
fn compiles_before_first_execution_and_reuses_code_for_new_inputs() {
    let mut backend = backend();
    let tracker = Arc::clone(backend.cpu_tracker());
    let before = tracker.total_compiled_loops.load(Ordering::Relaxed);
    let x = InputArg::from_type_rc(Type::Int, 10);
    let y = InputArg::from_type_rc(Type::Int, 20);
    let add = op(
        OpCode::IntAdd,
        &[
            Operand::from_bound_inputarg(&x),
            Operand::from_bound_inputarg(&y),
        ],
        30,
    );
    let operations = vec![
        add.clone(),
        finish(&[Operand::from_bound_op(&add)], vec![Type::Int]),
    ];
    let inputs = vec![x.clone(), y.clone()];
    let token = token();
    let mut code = unsafe {
        CompiledIr::compile(
            &mut *backend,
            token.clone(),
            &inputs,
            &operations,
            ConstMap::default(),
        )
    }
    .unwrap();
    assert_eq!(
        tracker.total_compiled_loops.load(Ordering::Relaxed),
        before + 1
    );
    assert!(code.asm_info().code_size > 0);
    assert_eq!(code.input_types(), &[Type::Int, Type::Int]);
    // Compilation does not need the frontend op graph to survive.
    drop(operations);
    drop(inputs);
    for (a, b) in [(4, 5), (-19, 3), (i64::MAX, 1)] {
        let result = unsafe { code.execute(&[Value::Int(a), Value::Int(b)]) }.unwrap();
        assert!(result.is_finish);
        assert_eq!(result.typed_outputs, vec![Value::Int(a.wrapping_add(b))]);
    }
    assert_eq!(
        tracker.total_compiled_loops.load(Ordering::Relaxed),
        before + 1
    );
    assert!(matches!(
        unsafe { code.execute(&[]) },
        Err(ArgumentError::Arity {
            expected: 2,
            actual: 0
        })
    ));
    assert!(matches!(
        unsafe { code.execute(&[Value::Float(1.0), Value::Int(2)]) },
        Err(ArgumentError::Type {
            index: 0,
            expected: Type::Int,
            actual: Type::Float
        })
    ));
    drop(code);

    // Same CPU/token still works through the pre-existing PyPy backend API.
    let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
    assert_eq!(backend.get_int_value(&frame, 0), 42);
    drop(frame);
    assert!(matches!(
        unsafe { CompiledIr::compile(&mut *backend, token, &[], &[], ConstMap::default()) },
        Err(BackendError::CompilationFailed(_))
    ));
}

#[test]
fn guard_failure_is_returned_without_fallback_or_recompilation() {
    let mut backend = backend();
    let tracker = Arc::clone(backend.cpu_tracker());
    let x = InputArg::from_type_rc(Type::Int, 0);
    let arg = Operand::from_bound_inputarg(&x);
    let compare = op(
        OpCode::IntGt,
        &[arg.clone(), Operand::from_opref(OpRef::const_int(0))],
        1,
    );
    // Native guard assembly writes AbstractFailDescr.rd_locs. Use the real
    // shared guard descriptor, not SimpleFailDescr (which lacks that slot).
    let descr = majit_backend::make_resume_guard_descr_typed(vec![Type::Int]);
    let fail_index = descr.as_fail_descr().unwrap().fail_index();
    let guard = Op::with_descr(
        OpCode::GuardTrue,
        &[Operand::from_bound_op(&compare)],
        descr.clone(),
    );
    guard.setfailargs([arg.clone()].into_iter().collect());
    let operations = vec![compare, OpRc::new(guard), finish(&[arg], vec![Type::Int])];
    let compiled_token = token();
    let mut code = unsafe {
        CompiledIr::compile(
            &mut *backend,
            compiled_token.clone(),
            &[x.clone()],
            &operations,
            ConstMap::default(),
        )
    }
    .unwrap();
    let compiled = tracker.total_compiled_loops.load(Ordering::Relaxed);
    for input in [7, -3, 0, 12] {
        let result = unsafe { code.execute(&[Value::Int(input)]) }.unwrap();
        assert_eq!(result.is_finish, input > 0);
        assert_eq!(result.typed_outputs, vec![Value::Int(input)]);
        if input <= 0 {
            assert!(Arc::ptr_eq(&result.descr_arc, &descr));
            assert_eq!(result.fail_index, fail_index);
            let owner = majit_backend::descr_owning_jct(result.descr_arc.as_fail_descr().unwrap())
                .expect(
                    "a guard must resolve its owner through the ordinary PyPy descriptor chain",
                );
            assert!(Arc::ptr_eq(&owner, &compiled_token));
        }
    }
    assert_eq!(
        tracker.total_compiled_loops.load(Ordering::Relaxed),
        compiled
    );
}

#[test]
fn mixed_types_and_void_results_preserve_the_backend_contract() {
    let mut backend = backend();
    let int = InputArg::from_type_rc(Type::Int, 0);
    let float = InputArg::from_type_rc(Type::Float, 1);
    let reference = InputArg::from_type_rc(Type::Ref, 2);
    let inputs = vec![int.clone(), float.clone(), reference.clone()];
    let args = [&int, &float, &reference].map(Operand::from_bound_inputarg);
    let values = [
        Value::Int(42),
        Value::Float(-0.0),
        Value::Ref(majit_ir::GcRef::NULL),
    ];
    // FINISH has one language result; exercise every bank with a mixed
    // argument list without requiring a backend-specific multi-return ABI.
    for (arg, value) in args.iter().zip(values) {
        let operations = vec![finish(std::slice::from_ref(arg), vec![value.get_type()])];
        let mut code = unsafe {
            CompiledIr::compile(
                &mut *backend,
                token(),
                &inputs,
                &operations,
                ConstMap::default(),
            )
        }
        .unwrap();
        let result = unsafe { code.execute(&values) }.unwrap();
        assert!(result.is_finish);
        assert_eq!(result.typed_outputs, vec![value]);
    }

    let mut code = unsafe {
        CompiledIr::compile(
            &mut *backend,
            token(),
            &[],
            &[finish(&[], vec![])],
            ConstMap::default(),
        )
    }
    .unwrap();
    let result = unsafe { code.execute(&[]) }.unwrap();
    assert!(result.is_finish);
    assert!(result.typed_outputs.is_empty());
}

#[test]
fn empty_submission_does_not_reach_codegen() {
    let mut backend = backend();
    let before = backend
        .cpu_tracker()
        .total_compiled_loops
        .load(Ordering::Relaxed);
    let result =
        unsafe { CompiledIr::compile(&mut *backend, token(), &[], &[], ConstMap::default()) };
    assert!(matches!(result, Err(BackendError::CompilationFailed(_))));
    drop(result);
    assert_eq!(
        backend
            .cpu_tracker()
            .total_compiled_loops
            .load(Ordering::Relaxed),
        before
    );
}

#[test]
fn host_function_runs_only_on_explicit_execution() {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    extern "C" fn twice(value: i64) -> i64 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        value.wrapping_mul(2)
    }

    let mut backend = backend();
    let x = InputArg::new_int_rc(0);
    let descr = majit_ir::make_call_descr(
        vec![Type::Int],
        Type::Int,
        majit_ir::EffectInfo::const_new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    let call = OpRc::new(Op::with_descr(
        OpCode::CallI,
        &[
            Operand::from_opref(OpRef::const_int(twice as *const () as usize as i64)),
            Operand::from_bound_inputarg(&x),
        ],
        descr,
    ));
    call.pos().set(OpRef::int_op(1));
    let operations = vec![
        call.clone(),
        finish(&[Operand::from_bound_op(&call)], vec![Type::Int]),
    ];
    let mut code = unsafe {
        CompiledIr::compile(
            &mut *backend,
            token(),
            &[x.clone()],
            &operations,
            ConstMap::default(),
        )
    }
    .unwrap();
    assert_eq!(
        CALLS.load(Ordering::Relaxed),
        0,
        "compilation must not call the host function"
    );
    // Bad arguments must not execute either.
    assert!(unsafe { code.execute(&[Value::Float(3.0)]) }.is_err());
    assert_eq!(CALLS.load(Ordering::Relaxed), 0);
    for (count, input) in [3, 21].into_iter().enumerate() {
        let result = unsafe { code.execute(&[Value::Int(input)]) }.unwrap();
        assert!(result.is_finish);
        assert_eq!(result.typed_outputs, vec![Value::Int(input * 2)]);
        assert_eq!(CALLS.load(Ordering::Relaxed), count as u64 + 1);
    }
}
