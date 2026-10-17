use parking_lot::Mutex;
/// Basic test: compile and execute a simple int_add loop via dynasm backend.
///
/// Trace: i0 = input
///   label(i0)
///   i1 = int_add(i0, 1)
///   i2 = int_lt(i1, 10)
///   guard_true(i2)  [fail_args: i1]
///   jump(i1)        → label
///   finish(i1)      [on guard failure]
use std::sync::LazyLock;

use majit_backend::{Backend, JitCellToken, STATUS_TYPE_MASK, make_resume_guard_descr_typed};
use majit_ir::{
    GcRef, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type, Value, make_array_descr,
    make_loop_target_descr,
};

use majit_backend_dynasm::runner::DynasmBackend;

static EXCEPTION_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

use majit_ir::forwarding::bound_operand_from_opref as rb;

#[test]
fn test_just_finish() {
    // Simplest possible trace: just FINISH with no args
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    let inputargs = vec![];

    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos().set(OpRef::void_op(0));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());

    let ops = vec![finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
}

#[test]
fn guard_not_forced_2_arms_returned_force_token_and_preserves_failargs() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(101);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let force_token = Op::new(OpCode::ForceToken, &[]);
    force_token.pos().set(OpRef::ref_op(1));

    let guard = Op::new(OpCode::GuardNotForced2, &[]);
    guard.pos().set(OpRef::void_op(2));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int]);
    guard.setfailargs(vec![rb(i0)].into());

    let finish = Op::new(OpCode::Finish, &[rb(OpRef::ref_op(1))]);
    finish.pos().set(OpRef::void_op(3));
    finish.set_fail_arg_types(vec![Type::Ref]);
    finish.setfailargs(vec![rb(OpRef::ref_op(1))].into());

    let ops: Vec<OpRc> = vec![force_token, guard, finish]
        .into_iter()
        .map(OpRc::new)
        .collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .expect("compile loop with GUARD_NOT_FORCED_2");

    let returned = backend.execute_token(&token, &[Value::Int(42)]);
    let raw_token = backend.get_ref_value(&returned, 0);
    assert!(backend.is_force_token_armed(raw_token));
    let forced = backend.force(raw_token).expect("armed force token");
    assert_eq!(backend.get_int_value(&forced, 0), 42);
}

#[test]
fn test_simple_int_add() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    // Simple trace: i1 = int_add(i0, CONST_1)
    // finish(i1)  [fail_arg_types: [Int], fail_args: [i1]]
    // history.py ConstInt.value inline.
    let const_1 = OpRef::const_int(1);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(const_1)]);
    add_op.pos().set(OpRef::int_op(1)); // result is OpRef::int_op(1)

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops = vec![add_op, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    // Compile
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Execute with input i0 = 42
    let args = vec![Value::Int(42)];
    let frame = backend.execute_token(&token, &args);

    // Check result
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());

    let result_val = backend.get_int_value(&frame, 0);
    assert_eq!(result_val, 43, "42 + 1 should be 43");
}

#[test]
fn test_finish_infers_int_type_when_explicit_types_are_empty() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(11);

    let const_1 = OpRef::const_int(1);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(const_1)]);
    add_op.pos().set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops = vec![add_op, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(descr.fail_arg_types(), &[Type::Int]);
    assert_eq!(backend.get_int_value(&frame, 0), 43);
}

#[test]
fn test_float_add() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    let i0 = OpRef::input_arg_float(0); // input: f64
    // history.py ConstFloat.value inline.
    let const_half = OpRef::const_float(0.5);

    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];

    let add_op = Op::new(OpCode::FloatAdd, &[rb(i0), rb(const_half)]);
    add_op.pos().set(OpRef::float_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(1))].into());

    let ops = vec![add_op, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Execute with input = 1.5
    let args = vec![Value::Float(1.5)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());

    let result_val = backend.get_float_value(&frame, 0);
    assert!(
        (result_val - 2.0).abs() < 1e-10,
        "1.5 + 0.5 should be 2.0, got {}",
        result_val
    );
}

/// `genop_float_neg` flips the sign bit. Subtracting from zero instead passes
/// every magnitude and fails only on zero, where `(+0) - (+0)` is `+0` — so
/// the assertions compare bit patterns rather than values.
#[test]
fn test_float_neg_keeps_the_sign_of_zero() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    let i0 = OpRef::input_arg_float(0);
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];

    let neg_op = Op::new(OpCode::FloatNeg, &[rb(i0)]);
    neg_op.pos().set(OpRef::float_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(1))].into());

    let ops = vec![neg_op, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    for (input, expected) in [(0.0f64, -0.0f64), (-0.0, 0.0), (1.5, -1.5), (-2.25, 2.25)] {
        let frame = backend.execute_token(&token, &[Value::Float(input)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());

        let got = backend.get_float_value(&frame, 0);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "-({input}) should be {expected}, got {got}"
        );
    }
}

#[test]
fn test_setarrayitem_raw_float_roundtrip() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(23);

    let const_index = OpRef::const_int(3);

    let array_descr = make_array_descr(0, 8, Type::Float);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Float, 1),
    ];
    let base = inputargs[0].opref();
    let value = inputargs[1].opref();

    let set_op = Op::new(
        OpCode::SetarrayitemRaw,
        &[rb(base), rb(const_index), rb(value)],
    );
    set_op.pos().set(OpRef::void_op(2));
    set_op.setdescr(array_descr.clone());

    let get_op = Op::new(OpCode::GetarrayitemRawF, &[rb(base), rb(const_index)]);
    get_op.pos().set(OpRef::float_op(3));
    get_op.setdescr(array_descr);

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(3))]);
    finish_op.pos().set(OpRef::void_op(4));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(3))].into());

    let ops = vec![set_op, get_op, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let mut items = vec![0.0f64; 8];
    let base_ref = GcRef(items.as_mut_ptr() as usize);
    let frame = backend.execute_token(&token, &[Value::Ref(base_ref), Value::Float(1.25)]);

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(items[3], 1.25);
    assert_eq!(backend.get_float_value(&frame, 0), 1.25);
}

#[test]
fn test_setarrayitem_raw_float_roundtrip_with_variable_index() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(24);

    let array_descr = make_array_descr(0, 8, Type::Float);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Float, 2),
    ];
    let base = inputargs[0].opref();
    let index = inputargs[1].opref();
    let value = inputargs[2].opref();

    let set_op = Op::new(OpCode::SetarrayitemRaw, &[rb(base), rb(index), rb(value)]);
    set_op.pos().set(OpRef::void_op(3));
    set_op.setdescr(array_descr.clone());

    let get_op = Op::new(OpCode::GetarrayitemRawF, &[rb(base), rb(index)]);
    get_op.pos().set(OpRef::float_op(4));
    get_op.setdescr(array_descr);

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(4))]);
    finish_op.pos().set(OpRef::void_op(5));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(4))].into());

    let ops = vec![set_op, get_op, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let mut items = vec![0.0f64; 8];
    let base_ref = GcRef(items.as_mut_ptr() as usize);
    let frame = backend.execute_token(
        &token,
        &[Value::Ref(base_ref), Value::Int(5), Value::Float(-2.5)],
    );

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(items[5], -2.5);
    assert_eq!(backend.get_float_value(&frame, 0), -2.5);
}

#[test]
fn test_guard_and_loop() {
    // Trace: loop that adds 1 until >= 5, then guard fails
    // i0 = input
    // label(i0)
    // i1 = int_add(i0, CONST_1)
    // i2 = int_lt(i1, CONST_5)
    // guard_true(i2)   [fail_args: i1]
    // jump(i1)
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let label_op = Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]);
    label_op.pos().set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let add_op = Op::new(
        OpCode::IntAdd,
        &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
    );
    add_op.pos().set(OpRef::int_op(1));

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(5))],
    );
    lt_op.pos().set(OpRef::int_op(2));

    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard_op.pos().set(OpRef::void_op(3));
    guard_op.set_fail_arg_types(vec![Type::Int]);
    guard_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let jump_op = Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]);
    jump_op.pos().set(OpRef::void_op(4));
    jump_op.setdescr(loop_descr);

    let ops = vec![label_op, add_op, lt_op, guard_op, jump_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Start with 0: should loop 0→1→2→3→4→5, guard fails at 5
    let args = vec![Value::Int(0)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should be guard failure, not finish");

    // fail_args = [OpRef::int_op(1)], so fail_arg index 0 = the IntAdd result.
    let result_val = backend.get_int_value(&frame, 0);
    assert_eq!(result_val, 5, "loop should stop at 5, fail_arg[0]");
}

/// `reshape_jump_args_for_preamble` pads a short body JUMP with the
/// preamble LABEL's extra entry box. The assembler remap then sees
/// equal arity (`x86/regalloc.py` `_compute_hint_locations_from_descr`)
/// and identity-moves the invariant slot.
#[test]
fn a_jump_that_carries_a_label_invariant_slot_compiles() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(11);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let counter = inputargs[0].opref();
    let invariant = inputargs[1].opref();
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let label_op = Op::new(OpCode::Label, &[rb(counter), rb(invariant)]);
    label_op.pos().set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let add_op = Op::new(OpCode::IntAdd, &[rb(counter), rb(OpRef::const_int(1))]);
    add_op.pos().set(OpRef::int_op(1));

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(5))],
    );
    lt_op.pos().set(OpRef::int_op(2));

    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard_op.pos().set(OpRef::void_op(3));
    guard_op.set_fail_arg_types(vec![Type::Int, Type::Int]);
    guard_op.setfailargs(vec![rb(OpRef::int_op(1)), rb(invariant)].into());

    let jump_op = Op::new(OpCode::Jump, &[rb(OpRef::int_op(1)), rb(invariant)]);
    jump_op.pos().set(OpRef::void_op(4));
    jump_op.setdescr(loop_descr);

    let ops_rc: Vec<OpRc> = vec![
        OpRc::new(label_op),
        OpRc::new(add_op),
        OpRc::new(lt_op),
        OpRc::new(guard_op),
        OpRc::new(jump_op),
    ];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(99)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should be guard failure, not finish");
    assert_eq!(backend.get_int_value(&frame, 0), 5);
    assert_eq!(backend.get_int_value(&frame, 1), 99);
}

/// The assembler remap is `x86/regalloc.py` `consider_jump`'s 1:1 walk;
/// a JUMP shorter than its compiled LABEL is the arity bug, not a
/// silent truncate.
#[test]
#[should_panic(expected = "JUMP args (1)")]
fn a_jump_shorter_than_its_label_is_rejected() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(12);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let counter = inputargs[0].opref();
    let invariant = inputargs[1].opref();
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let label_op = Op::new(OpCode::Label, &[rb(counter), rb(invariant)]);
    label_op.pos().set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let add_op = Op::new(OpCode::IntAdd, &[rb(counter), rb(OpRef::const_int(1))]);
    add_op.pos().set(OpRef::int_op(1));

    let jump_op = Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]);
    jump_op.pos().set(OpRef::void_op(2));
    jump_op.setdescr(loop_descr);

    let ops_rc: Vec<OpRc> = vec![OpRc::new(label_op), OpRc::new(add_op), OpRc::new(jump_op)];
    let _ = backend.compile_loop(&inputargs, &ops_rc, &token);
}

/// `regalloc.py consider_guard_value`: the counter index names the
/// guard operand's deadframe slot, independent of the guard's fail arguments.
#[test]
fn guard_value_gets_a_per_value_counter_when_its_operand_is_not_a_failarg() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(2);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let value = inputargs[0].opref();
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let label_op = Op::new(OpCode::Label, &[rb(value)]);
    label_op.pos().set(OpRef::void_op(0));
    label_op.setdescr(loop_descr.clone());

    let guard_descr = make_resume_guard_descr_typed(vec![]);
    let guard_op = Op::new(OpCode::GuardValue, &[rb(value), rb(OpRef::const_int(7))]);
    guard_op.pos().set(OpRef::void_op(1));
    guard_op.set_fail_arg_types(vec![]);
    guard_op.setfailargs(vec![].into());
    guard_op.setdescr(guard_descr.clone());

    let jump_op = Op::new(OpCode::Jump, &[rb(value)]);
    jump_op.pos().set(OpRef::void_op(2));
    jump_op.setdescr(loop_descr);

    let ops_rc: Vec<OpRc> = vec![OpRc::new(label_op), OpRc::new(guard_op), OpRc::new(jump_op)];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let guard_fd = guard_descr
        .as_fail_descr()
        .expect("guard descr is FailDescr");
    let status = guard_fd.get_status();
    assert_ne!(
        status & STATUS_TYPE_MASK,
        0,
        "GUARD_VALUE must have a per-value counter when arg0 is not a failarg"
    );
    let slot = majit_backend::guard_value_counter_slot(guard_fd)
        .expect("GUARD_VALUE must record a deadframe counter slot");
    assert!(
        slot < 64,
        "GUARD_VALUE deadframe slot is implausible: {slot}"
    );

    // 99 != 7, so the guard fails on the first check.  `compile.py must_compile`
    // hashes the failing operand, so the recorded slot has to name the word
    // that holds it — here a register the guard carries no fail argument for.
    let frame = backend.execute_token(&token, &[Value::Int(99)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should be guard failure, not finish");
    assert_eq!(
        backend.get_value_direct(&frame, slot),
        99,
        "the deadframe slot recorded for the GUARD_VALUE must hold its operand"
    );
}

/// Loop over `(f0, f1, i2)` guarded by `GUARD_VALUE(f0, rhs)`, then a counter
/// that exits through `GUARD_TRUE(i < 5)`. Returns `(exited_at_counter, i)`:
/// a counter exit means every `GUARD_VALUE` passed, `i == 0` from the first
/// `GUARD_VALUE` means it failed on entry.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn run_float_guard_value_loop(rhs: OpRef, f0: f64, f1: f64) -> (bool, i64) {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(3);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Float, 0),
        InputArg::from_type_rc(Type::Float, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let a = OpRef::input_arg_float(0);
    let b = OpRef::input_arg_float(1);
    let i = OpRef::input_arg_int(2);
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let label_op = Op::new(OpCode::Label, &[rb(a), rb(b), rb(i)]);
    label_op.pos().set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let value_descr = make_resume_guard_descr_typed(vec![Type::Float, Type::Int]);
    let value_guard = Op::new(OpCode::GuardValue, &[rb(a), rb(rhs)]);
    value_guard.pos().set(OpRef::void_op(1));
    value_guard.set_fail_arg_types(vec![Type::Float, Type::Int]);
    value_guard.setfailargs(vec![rb(a), rb(i)].into());
    value_guard.setdescr(value_descr.clone());

    let inc_op = Op::new(OpCode::IntAdd, &[rb(i), rb(OpRef::const_int(1))]);
    inc_op.pos().set(OpRef::int_op(2));

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::int_op(2)), rb(OpRef::const_int(5))],
    );
    lt_op.pos().set(OpRef::int_op(3));

    let counter_descr = make_resume_guard_descr_typed(vec![Type::Float, Type::Int]);
    let counter_guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(3))]);
    counter_guard.pos().set(OpRef::void_op(4));
    counter_guard.set_fail_arg_types(vec![Type::Float, Type::Int]);
    counter_guard.setfailargs(vec![rb(a), rb(OpRef::int_op(2))].into());
    counter_guard.setdescr(counter_descr.clone());

    let jump_op = Op::new(OpCode::Jump, &[rb(a), rb(b), rb(OpRef::int_op(2))]);
    jump_op.pos().set(OpRef::void_op(5));
    jump_op.setdescr(loop_descr);

    let ops_rc: Vec<OpRc> = vec![label_op, value_guard, inc_op, lt_op, counter_guard, jump_op]
        .into_iter()
        .map(OpRc::new)
        .collect();
    backend
        .compile_loop(&inputargs, &ops_rc, &token)
        .expect("compile loop with a float GUARD_VALUE");

    let frame = backend.execute_token(&token, &[Value::Float(f0), Value::Float(f1), Value::Int(0)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "the loop exits only through a guard");
    let counter_fd = counter_descr
        .as_fail_descr()
        .expect("GUARD_TRUE descr is FailDescr");
    let exited_at_counter = std::ptr::addr_eq(descr, counter_fd);
    assert_eq!(backend.get_float_value(&frame, 0), f0);
    (exited_at_counter, backend.get_int_value(&frame, 1))
}

/// FLOAT `GUARD_VALUE` against a `ConstFloat`.
/// x86 `genop_guard_guard_value` compares with `UCOMISD`;
/// aarch64 `emit_op_guard_value` compares with `FCMP_dd`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn guard_value_on_a_float_against_a_const_float() {
    let k = OpRef::const_float(2.5);
    assert_eq!(run_float_guard_value_loop(k, 2.5, 0.0), (true, 5));
    assert_eq!(run_float_guard_value_loop(k, 1.0, 0.0), (false, 0));
}

/// FLOAT `GUARD_VALUE` with both operands in xmm / vfp registers
/// (`UCOMISD` / `FCMP_dd`).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn guard_value_on_a_float_against_a_float_box() {
    let b = OpRef::input_arg_float(1);
    assert_eq!(run_float_guard_value_loop(b, -3.75, -3.75), (true, 5));
    assert_eq!(run_float_guard_value_loop(b, -3.75, 3.75), (false, 0));
}

/// `CVTTSD2SI_rx` / `CVTSD2SS_xx` of a `ConstFloat` operand.
/// `execute_cast_const` does not fold non-finite or out-of-range values,
/// so the constant reaches `consider_cast_float_to_int`.
/// `genop_finish` stores only its first argument, so the four results
/// come back as failargs of `GUARD_ALWAYS_FAILS`.
/// `genop_cast_float_to_singlefloat` does `CVTSD2SS` into the low 32 bits
/// of the xmm temp, then `MOVD32` into the int.
#[cfg(target_arch = "x86_64")]
#[test]
fn cast_const_float_uses_cvttsd2si_and_cvtsd2ss() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(48);

    let inf = Op::new(
        OpCode::CastFloatToInt,
        &[rb(OpRef::const_float(f64::INFINITY))],
    );
    inf.pos().set(OpRef::int_op(1));
    let nan = Op::new(OpCode::CastFloatToInt, &[rb(OpRef::const_float(f64::NAN))]);
    nan.pos().set(OpRef::int_op(2));
    let finite = Op::new(OpCode::CastFloatToInt, &[rb(OpRef::const_float(3.75))]);
    finite.pos().set(OpRef::int_op(3));
    let single = Op::new(
        OpCode::CastFloatToSinglefloat,
        &[rb(OpRef::const_float(1.5))],
    );
    single.pos().set(OpRef::int_op(4));

    let fail_descr = make_resume_guard_descr_typed(vec![Type::Int; 4]);
    let guard = Op::new(OpCode::GuardAlwaysFails, &[]);
    guard.pos().set(OpRef::void_op(5));
    guard.set_fail_arg_types(vec![Type::Int; 4]);
    guard.setfailargs(
        vec![
            rb(OpRef::int_op(1)),
            rb(OpRef::int_op(2)),
            rb(OpRef::int_op(3)),
            rb(OpRef::int_op(4)),
        ]
        .into(),
    );
    guard.setdescr(fail_descr);

    let finish = Op::new(OpCode::Finish, &[]);
    finish.pos().set(OpRef::void_op(6));
    finish.set_fail_arg_types(vec![]);
    finish.setfailargs(vec![].into());

    let ops_rc: Vec<OpRc> = vec![inf, nan, finite, single, guard, finish]
        .into_iter()
        .map(OpRc::new)
        .collect();
    let inputargs: Vec<InputArgRc> = vec![];
    backend
        .compile_loop(&inputargs, &ops_rc, &token)
        .expect("compile casts of ConstFloat");
    let frame = backend.execute_token(&token, &[]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "GUARD_ALWAYS_FAILS should side-exit with the cast results"
    );
    assert_eq!(backend.get_int_value(&frame, 0), i64::MIN);
    assert_eq!(backend.get_int_value(&frame, 1), i64::MIN);
    assert_eq!(backend.get_int_value(&frame, 2), 3);
    assert_eq!(
        backend.get_int_value(&frame, 3),
        i64::from(1.5f32.to_bits())
    );
}

/// More float loop arguments than xmm registers, rotated by one at every
/// `JUMP`. Some of them live in frame slots, so `remap_frame_layout` breaks
/// a cycle that runs through float frame slots with `regalloc_push` /
/// `regalloc_pop`.
#[test]
fn float_frame_slots_rotated_across_jump() {
    const N: u32 = 24;
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(7);

    let mut inputargs: Vec<InputArgRc> = (0..N)
        .map(|k| InputArg::from_type_rc(Type::Float, k))
        .collect();
    inputargs.push(InputArg::from_type_rc(Type::Int, N));
    let floats: Vec<OpRef> = (0..N).map(OpRef::input_arg_float).collect();
    let counter = OpRef::input_arg_int(N);
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let mut label_args: Vec<_> = floats.iter().map(|&f| rb(f)).collect();
    label_args.push(rb(counter));
    let label_op = Op::new(OpCode::Label, &label_args);
    label_op.pos().set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let inc_op = Op::new(OpCode::IntAdd, &[rb(counter), rb(OpRef::const_int(1))]);
    inc_op.pos().set(OpRef::int_op(101));
    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::int_op(101)), rb(OpRef::const_int(3))],
    );
    lt_op.pos().set(OpRef::int_op(102));

    let mut fail_types = vec![Type::Float; N as usize];
    fail_types.push(Type::Int);
    let mut fail_args: Vec<_> = floats.iter().map(|&f| rb(f)).collect();
    fail_args.push(rb(OpRef::int_op(101)));
    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(102))]);
    guard_op.pos().set(OpRef::void_op(103));
    guard_op.set_fail_arg_types(fail_types);
    guard_op.setfailargs(fail_args.into());

    let mut jump_args: Vec<_> = (1..N)
        .chain(std::iter::once(0))
        .map(|k| rb(floats[k as usize]))
        .collect();
    jump_args.push(rb(OpRef::int_op(101)));
    let jump_op = Op::new(OpCode::Jump, &jump_args);
    jump_op.pos().set(OpRef::void_op(104));
    jump_op.setdescr(loop_descr);

    let ops_rc: Vec<OpRc> = vec![label_op, inc_op, lt_op, guard_op, jump_op]
        .into_iter()
        .map(OpRc::new)
        .collect();
    backend
        .compile_loop(&inputargs, &ops_rc, &token)
        .expect("compile a loop rotating float arguments");

    let mut args: Vec<Value> = (0..N).map(|k| Value::Float(k as f64 + 0.5)).collect();
    args.push(Value::Int(0));
    let frame = backend.execute_token(&token, &args);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, N as usize), 3);
    // Two JUMPs ran before the guard failed on the third pass.
    for k in 0..N {
        let expected = ((k + 2) % N) as f64 + 0.5;
        assert_eq!(
            backend.get_float_value(&frame, k as usize),
            expected,
            "float argument {k}"
        );
    }
}

#[test]
fn test_float_loop_carried_across_jump() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Float, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let label_op = Op::new(
        OpCode::Label,
        &[rb(OpRef::input_arg_float(0)), rb(OpRef::input_arg_int(1))],
    );
    label_op.pos().set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::input_arg_int(1)), rb(OpRef::const_int(5))],
    );
    lt_op.pos().set(OpRef::int_op(2));

    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard_op.pos().set(OpRef::void_op(3));
    guard_op.set_fail_arg_types(vec![Type::Float, Type::Int]);
    guard_op.setfailargs(vec![rb(OpRef::input_arg_float(0)), rb(OpRef::input_arg_int(1))].into());

    let cast_op = Op::new(OpCode::CastIntToFloat, &[rb(OpRef::input_arg_int(1))]);
    cast_op.pos().set(OpRef::float_op(4));

    let mul_op = Op::new(
        OpCode::FloatMul,
        &[rb(OpRef::float_op(4)), rb(OpRef::const_float(0.5))],
    );
    mul_op.pos().set(OpRef::float_op(5));

    let add_op = Op::new(
        OpCode::FloatAdd,
        &[rb(OpRef::input_arg_float(0)), rb(OpRef::float_op(5))],
    );
    add_op.pos().set(OpRef::float_op(6));

    let inc_op = Op::new(
        OpCode::IntAdd,
        &[rb(OpRef::input_arg_int(1)), rb(OpRef::const_int(1))],
    );
    inc_op.pos().set(OpRef::int_op(7));

    let jump_op = Op::new(
        OpCode::Jump,
        &[rb(OpRef::float_op(6)), rb(OpRef::int_op(7))],
    );
    jump_op.pos().set(OpRef::void_op(8));
    jump_op.setdescr(loop_descr);

    let ops = vec![
        label_op, lt_op, guard_op, cast_op, mul_op, add_op, inc_op, jump_op,
    ];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let args = vec![Value::Float(0.0), Value::Int(0)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should exit via guard failure");

    let sum = backend.get_float_value(&frame, 0);
    let index = backend.get_int_value(&frame, 1);
    assert!(
        (sum - 5.0).abs() < 1e-10,
        "expected carried float sum to be 5.0, got {}",
        sum
    );
    assert_eq!(index, 5, "expected guard failure at i=5");
}

#[test]
fn test_gc_typeinfo_guards_use_dynasm_emit() {
    let mut gc = majit_gc::collector::MiniMarkGC::new();
    let root_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let child_tid = gc.register_type(majit_gc::TypeInfo::object_subclass(16, root_tid));
    let root_vtable: usize = 0x1234_5000;
    let child_vtable: usize = 0x1234_6000;
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_vtable, root_tid);
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, child_vtable, child_tid);
    let child_obj = gc.alloc_with_type(child_tid, 16);

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    // jitframe.py — a collector with a type table carries JITFRAME before install.
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    backend.set_gc_allocator(Box::new(gc));
    let token = JitCellToken::new(41);

    let const_child_tid = OpRef::const_int(child_tid as i64);
    let const_root_vtable = OpRef::const_int(root_vtable as i64);

    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let i0 = inputargs[0].opref();

    let guard_gc_type = Op::new(OpCode::GuardGcType, &[rb(i0), rb(const_child_tid)]);
    guard_gc_type.pos().set(OpRef::void_op(1));
    guard_gc_type.set_fail_arg_types(vec![]);
    guard_gc_type.setfailargs(vec![].into());

    let guard_is_object = Op::new(OpCode::GuardIsObject, &[rb(i0)]);
    guard_is_object.pos().set(OpRef::void_op(2));
    guard_is_object.set_fail_arg_types(vec![]);
    guard_is_object.setfailargs(vec![].into());

    let guard_subclass = Op::new(OpCode::GuardSubclass, &[rb(i0), rb(const_root_vtable)]);
    guard_subclass.pos().set(OpRef::void_op(3));
    guard_subclass.set_fail_arg_types(vec![]);
    guard_subclass.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos().set(OpRef::void_op(4));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());

    let ops = vec![guard_gc_type, guard_is_object, guard_subclass, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Ref(child_obj)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish(), "all GC type-info guards should pass");
}

#[test]
fn test_gc_typeinfo_guards_side_exit_on_mismatch() {
    {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let root_tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let child_tid = gc.register_type(majit_gc::TypeInfo::object_subclass(16, root_tid));
        let child_vtable: usize = 0x1235_6000;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, child_vtable, child_tid);
        let root_obj = gc.alloc_with_type(root_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        // jitframe.py — a collector with a type table carries JITFRAME before install.
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        backend.set_gc_allocator(Box::new(gc));
        let token = JitCellToken::new(45);

        let const_child_tid = OpRef::const_int(child_tid as i64);

        let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
        let i0 = inputargs[0].opref();
        let guard_gc_type = Op::new(OpCode::GuardGcType, &[rb(i0), rb(const_child_tid)]);
        guard_gc_type.pos().set(OpRef::void_op(1));
        guard_gc_type.set_fail_arg_types(vec![Type::Ref]);
        guard_gc_type.setfailargs(vec![rb(i0)].into());
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos().set(OpRef::void_op(2));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<OpRc> = vec![guard_gc_type, finish_op]
            .into_iter()
            .map(OpRc::new)
            .collect();
        let result = backend.compile_loop(&inputargs, &ops_rc, &token);
        assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
        let frame = backend.execute_token(&token, &[Value::Ref(root_obj)]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "GUARD_GC_TYPE should side-exit on typeid mismatch"
        );
        assert_eq!(backend.get_ref_value(&frame, 0), root_obj);
    }

    {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let raw_tid = gc.register_type(majit_gc::TypeInfo::simple(16));
        let raw_obj = gc.alloc_with_type(raw_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        // jitframe.py — a collector with a type table carries JITFRAME before install.
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        backend.set_gc_allocator(Box::new(gc));
        let token = JitCellToken::new(46);

        let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
        let i0 = inputargs[0].opref();
        let guard_is_object = Op::new(OpCode::GuardIsObject, &[rb(i0)]);
        guard_is_object.pos().set(OpRef::void_op(1));
        guard_is_object.set_fail_arg_types(vec![Type::Ref]);
        guard_is_object.setfailargs(vec![rb(i0)].into());
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos().set(OpRef::void_op(2));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<OpRc> = vec![guard_is_object, finish_op]
            .into_iter()
            .map(OpRc::new)
            .collect();
        let result = backend.compile_loop(&inputargs, &ops_rc, &token);
        assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
        let frame = backend.execute_token(&token, &[Value::Ref(raw_obj)]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "GUARD_IS_OBJECT should side-exit for non-rclass object types"
        );
        assert_eq!(backend.get_ref_value(&frame, 0), raw_obj);
    }

    {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let root_a_tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let root_b_tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let root_a_vtable: usize = 0x1236_5000;
        let root_b_vtable: usize = 0x1236_7000;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_a_vtable, root_a_tid);
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_b_vtable, root_b_tid);
        let root_b_obj = gc.alloc_with_type(root_b_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        // jitframe.py — a collector with a type table carries JITFRAME before install.
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
        backend.set_gc_allocator(Box::new(gc));
        let token = JitCellToken::new(47);

        let const_root_a_vtable = OpRef::const_int(root_a_vtable as i64);

        let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
        let i0 = inputargs[0].opref();
        let guard_subclass = Op::new(OpCode::GuardSubclass, &[rb(i0), rb(const_root_a_vtable)]);
        guard_subclass.pos().set(OpRef::void_op(1));
        guard_subclass.set_fail_arg_types(vec![Type::Ref]);
        guard_subclass.setfailargs(vec![rb(i0)].into());
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos().set(OpRef::void_op(2));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<OpRc> = vec![guard_subclass, finish_op]
            .into_iter()
            .map(OpRc::new)
            .collect();
        let result = backend.compile_loop(&inputargs, &ops_rc, &token);
        assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
        let frame = backend.execute_token(&token, &[Value::Ref(root_b_obj)]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "GUARD_SUBCLASS should side-exit outside the expected subclass range"
        );
        assert_eq!(backend.get_ref_value(&frame, 0), root_b_obj);
    }
}

#[test]
fn guard_subclass_vtable_range_without_type_info() {
    #[repr(C)]
    struct Class {
        subclassrange_min: i64,
        subclassrange_max: i64,
    }
    #[repr(C)]
    struct Obj {
        typeptr: *const Class,
    }
    let parent = Class {
        subclassrange_min: 0,
        subclassrange_max: 4,
    };
    let child = Class {
        subclassrange_min: 2,
        subclassrange_max: 3,
    };
    let other = Class {
        subclassrange_min: 10,
        subclassrange_max: 11,
    };
    let mut child_obj = Obj { typeptr: &child };
    let mut other_obj = Obj { typeptr: &other };

    let prev = majit_backend::cpu_subclassrange_min_offset();
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    backend.set_vtable_offset(Some(0));
    backend.set_subclassrange_min_offset(Some(0));

    let const_parent = OpRef::const_int(&parent as *const Class as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let i0 = inputargs[0].opref();
    let guard_subclass = Op::new(OpCode::GuardSubclass, &[rb(i0), rb(const_parent)]);
    guard_subclass.pos().set(OpRef::void_op(1));
    guard_subclass.set_fail_arg_types(vec![Type::Ref]);
    guard_subclass.setfailargs(vec![rb(i0)].into());
    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());
    let ops_rc: Vec<OpRc> = vec![guard_subclass, finish_op]
        .into_iter()
        .map(OpRc::new)
        .collect();

    let pass = JitCellToken::new(90);
    let result = backend.compile_loop(&inputargs, &ops_rc, &pass);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
    let child_ref = GcRef(&mut child_obj as *mut Obj as usize);
    let frame = backend.execute_token(&pass, &[Value::Ref(child_ref)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "a subclass must pass GUARD_SUBCLASS"
    );

    let fail = JitCellToken::new(91);
    let result = backend.compile_loop(&inputargs, &ops_rc, &fail);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
    let other_ref = GcRef(&mut other_obj as *mut Obj as usize);
    let frame = backend.execute_token(&fail, &[Value::Ref(other_ref)]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "an unrelated class must fail GUARD_SUBCLASS"
    );
    backend.set_subclassrange_min_offset(prev);
}

#[test]
fn test_exception_guards_use_dynasm_emit() {
    let _guard = EXCEPTION_TEST_LOCK.lock();
    majit_backend_dynasm::jit_exc_clear();

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(42);

    let expected_class = 0x5151_0000_i64;
    let const_expected_class = OpRef::const_int(expected_class);

    let inputargs = vec![];

    let guard_exception = Op::new(OpCode::GuardException, &[rb(const_expected_class)]);
    guard_exception.pos().set(OpRef::ref_op(0));
    guard_exception.set_fail_arg_types(vec![]);
    guard_exception.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::ref_op(0))]);
    finish_op.pos().set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![Type::Ref]);
    finish_op.setfailargs(vec![rb(OpRef::ref_op(0))].into());

    let ops = vec![guard_exception, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let mut exc_obj = vec![expected_class as usize, 0usize];
    let exc_ref = GcRef(exc_obj.as_mut_ptr() as usize);
    majit_backend_dynasm::jit_exc_raise(exc_ref.0 as i64);

    let frame = backend.execute_token(&token, &[]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish(), "matching GUARD_EXCEPTION should pass");
    assert_eq!(backend.get_ref_value(&frame, 0), exc_ref);
    assert!(!majit_backend_dynasm::jit_exc_is_pending());

    majit_backend_dynasm::jit_exc_clear();
}

/// A mismatching `GUARD_EXCEPTION` leaves through the `exc=True`
/// `_build_failure_recovery` stub, which moves `pos_exc_value` into
/// `jf_guard_exc` and clears both cells.
#[test]
fn guard_exception_mismatch_saves_the_value_in_jf_guard_exc() {
    let _guard = EXCEPTION_TEST_LOCK.lock();
    majit_backend_dynasm::jit_exc_clear();

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(45);

    let guard_exception = Op::new(OpCode::GuardException, &[rb(OpRef::const_int(0x5151_0000))]);
    guard_exception.pos().set(OpRef::ref_op(0));
    guard_exception.set_fail_arg_types(vec![]);
    guard_exception.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::ref_op(0))]);
    finish_op.pos().set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![Type::Ref]);
    finish_op.setfailargs(vec![rb(OpRef::ref_op(0))].into());

    let ops_rc: Vec<OpRc> = vec![OpRc::new(guard_exception), OpRc::new(finish_op)];
    backend
        .compile_loop(&[], &ops_rc, &token)
        .expect("compile loop with GUARD_EXCEPTION");

    let mut exc_obj = vec![0x6262_0000_usize, 0usize];
    let exc_ref = GcRef(exc_obj.as_mut_ptr() as usize);
    majit_backend_dynasm::jit_exc_raise(exc_ref.0 as i64);

    let frame = backend.execute_token(&token, &[]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "a different class must fail GUARD_EXCEPTION"
    );
    assert_eq!(backend.grab_exc_value(&frame), exc_ref);
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
    assert_eq!(majit_backend_dynasm::jit_exc_class_raw(), 0);

    majit_backend_dynasm::jit_exc_clear();
}

/// `_restore_exception` writes `pos_exc_value`, then `pos_exception`, from
/// registers, from immediates that fit 32 bits, and from wider immediates.
#[test]
fn restore_exception_writes_both_cells() {
    let _guard = EXCEPTION_TEST_LOCK.lock();

    let run = |class: OpRef, value: OpRef, inputargs: &[InputArgRc], args: &[Value]| {
        majit_backend_dynasm::jit_exc_clear();
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let token = JitCellToken::new(46);

        let restore = Op::new(OpCode::RestoreException, &[rb(class), rb(value)]);
        restore.pos().set(OpRef::void_op(10));
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos().set(OpRef::void_op(11));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<OpRc> = vec![OpRc::new(restore), OpRc::new(finish_op)];
        backend
            .compile_loop(inputargs, &ops_rc, &token)
            .expect("compile loop with RESTORE_EXCEPTION");
        let frame = backend.execute_token(&token, args);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let got = (
            majit_backend_dynasm::jit_exc_class_raw(),
            majit_backend_dynasm::jit_exc_value_peek(),
        );
        majit_backend_dynasm::jit_exc_clear();
        got
    };

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let mut exc_obj = vec![0x7171_0000_usize, 0usize];
    let exc_ref = GcRef(exc_obj.as_mut_ptr() as usize);
    assert_eq!(
        run(
            OpRef::input_arg_int(0),
            OpRef::input_arg_ref(1),
            &inputargs,
            &[Value::Int(0x7171_0000), Value::Ref(exc_ref)],
        ),
        (0x7171_0000, exc_ref.0 as i64)
    );

    let wide = 0x1234_5678_9abc_i64;
    assert_eq!(
        run(
            OpRef::const_int(0x7272_0000),
            OpRef::const_int(wide),
            &[],
            &[]
        ),
        (0x7272_0000, wide)
    );
}

#[test]
fn test_guard_no_exception_and_always_fails_emit_side_exits() {
    let _guard = EXCEPTION_TEST_LOCK.lock();
    majit_backend_dynasm::jit_exc_clear();

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(43);

    let inputargs = vec![];
    let guard_no_exception = Op::new(OpCode::GuardNoException, &[]);
    guard_no_exception.pos().set(OpRef::void_op(0));
    guard_no_exception.set_fail_arg_types(vec![]);
    guard_no_exception.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos().set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());

    let ops = vec![guard_no_exception, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "GUARD_NO_EXCEPTION should pass with clear exception state"
    );

    let mut exc_obj = vec![0x6161_0000_usize, 0usize];
    majit_backend_dynasm::jit_exc_raise(exc_obj.as_mut_ptr() as i64);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "GUARD_NO_EXCEPTION should side-exit with pending exception"
    );
    majit_backend_dynasm::jit_exc_clear();

    let mut always_backend = DynasmBackend::new();
    always_backend.attach_default_test_descrs();
    let always_token = JitCellToken::new(44);
    let guard_always_fails = Op::new(OpCode::GuardAlwaysFails, &[]);
    guard_always_fails.pos().set(OpRef::void_op(0));
    guard_always_fails.set_fail_arg_types(vec![]);
    guard_always_fails.setfailargs(vec![].into());
    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos().set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());
    let ops = vec![guard_always_fails, finish_op];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let result = always_backend.compile_loop(&[], &ops_rc, &always_token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
    let frame = always_backend.execute_token(&always_token, &[]);
    assert!(
        !always_backend.get_latest_descr(&frame).is_finish(),
        "GUARD_ALWAYS_FAILS should side-exit unconditionally"
    );
}

#[test]
fn test_int_binop_wide_immediate_is_not_truncated() {
    // An immediate that does not fit in 32 bits must not use the imm32
    // form: the encoder would truncate it and the CPU sign-extend the low
    // half, so `x & 0xFFFF_FFFF_FFFF` degenerated to `x & -1` (a no-op).
    // regloc.py:456-464 routes such immediates through the scratch register.
    let mask = 0xFFFF_FFFF_FFFFi64;
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(45);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let and_op = Op::new(OpCode::IntAnd, &[rb(i0), rb(OpRef::const_int(mask))]);
    and_op.pos().set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops_rc: Vec<OpRc> = vec![OpRc::new(and_op), OpRc::new(finish_op)];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Int(-1)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        mask,
        "-1 & 0xFFFF_FFFF_FFFF must keep only the low 48 bits"
    );
}

#[test]
fn test_int_add_wide_immediate_is_not_truncated() {
    // Twin of the AND test for the LEA-form `int_add` emitter: a 2^32
    // addend truncated to imm32 would add 0.
    let addend = 1i64 << 32;
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(46);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(addend))]);
    add_op.pos().set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops_rc: Vec<OpRc> = vec![OpRc::new(add_op), OpRc::new(finish_op)];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        addend + 7,
        "7 + 2^32 must not truncate the immediate"
    );
}
