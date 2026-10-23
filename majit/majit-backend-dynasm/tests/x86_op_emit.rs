//! Execution checks for the x86 emitters ported onto the upstream sequences:
//! `genop_int_and`, `genop_int_force_ge_zero`, `genop_float_neg`,
//! `genop_float_abs`, `_cmp_guard_gc_type`, `genop_guard_guard_is_object`,
//! `genop_guard_guard_subclass`, `_binaryop`, `_cmpop_float`,
//! `_store_force_index`, `store_force_descr`, `genop_guard_guard_no_exception`,
//! `cond_call` / `CondCallSlowPath`, `generate_quick_failure`,
//! `_build_failure_recovery`, `malloc_cond`,
//! `malloc_cond_varsize_frame`, `malloc_cond_varsize`,
//! `_binaryop_or_lea`, `flush_cc`, `addr_add_const`, `genop_finish`,
//! `closing_jump`, `genop_discard_gc_store_indexed` / `save_into_mem`,
//! `consider_gc_store`,
//! `test_location`, `AddressLoc`, `genop_cast_float_to_singlefloat`,
//! `genop_cast_singlefloat_to_float`.

#![cfg(target_arch = "x86_64")]

use std::arch::naked_asm;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use majit_backend::{Backend, JitCellToken, make_resume_guard_descr_typed};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_gc::GcFlags;
use majit_gc::collector::MiniMarkGC;
use majit_gc::header::{GcHeader, header_of};
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::operand::Operand;
use majit_ir::{
    CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, GcRef, InputArg, InputArgRc,
    LoopTokenDescr, OopSpecIndex, Op, OpCode, OpRc, OpRef, Type, Value, make_loop_target_descr,
};

fn compile_and_run_int(opcode: OpCode, extra: Option<i64>, input: i64, token_id: u64) -> i64 {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(token_id as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let mut args = vec![rb(i0)];
    if let Some(imm) = extra {
        args.push(rb(OpRef::const_int(imm)));
    }
    let body = Op::new(opcode, &args);
    body.pos().set(OpRef::int_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_int_value(&frame, 0)
}

fn compile_float(opcode: OpCode, token_id: u64) -> (DynasmBackend, JitCellToken) {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(token_id as i64);
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let i0 = inputargs[0].opref();
    let body = Op::new(opcode, &[rb(i0)]);
    body.pos().set(OpRef::float_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Float]);
    finish.setfailargs(vec![rb(OpRef::float_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    (backend, token)
}

fn float_bits(backend: &mut DynasmBackend, token: &JitCellToken, input: f64) -> u64 {
    let frame = backend.execute_token(token, &[Value::Float(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_float_value(&frame, 0).to_bits()
}

struct GcWorld {
    backend: DynasmBackend,
    child_tid: i64,
    root_vtable: usize,
    child: GcRef,
    raw: GcRef,
    other: GcRef,
}

fn gc_world(token_id: u64) -> (GcWorld, JitCellToken) {
    let mut gc = majit_gc::collector::MiniMarkGC::new();
    let root_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let child_tid = gc.register_type(majit_gc::TypeInfo::object_subclass(16, root_tid));
    let other_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let raw_tid = gc.register_type(majit_gc::TypeInfo::simple(16));
    let root_vtable: usize = 0x1240_5000 + token_id as usize;
    let other_vtable: usize = 0x1240_7000 + token_id as usize;
    let child_vtable: usize = 0x1240_6000 + token_id as usize;
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_vtable, root_tid);
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, other_vtable, other_tid);
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, child_vtable, child_tid);
    let child = gc.alloc_with_type(child_tid, 16);
    let raw = gc.alloc_with_type(raw_tid, 16);
    let other = gc.alloc_with_type(other_tid, 16);

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    backend.set_gc_allocator(Box::new(gc));
    let token = JitCellToken::new(token_id as i64);
    (
        GcWorld {
            backend,
            child_tid: i64::from(child_tid),
            root_vtable,
            child,
            raw,
            other,
        },
        token,
    )
}

fn guard_finished(world: &mut GcWorld, token: &JitCellToken, ops: Vec<Op>, obj: GcRef) -> bool {
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    world
        .backend
        .compile_loop(&inputargs, &ops_rc, token)
        .unwrap_or_else(|err| panic!("compile guard: {err:?}"));
    let frame = world.backend.execute_token(token, &[Value::Ref(obj)]);
    world.backend.get_latest_descr(&frame).is_finish()
}

fn void_guard(opcode: OpCode, args: &[Operand], pos: u32) -> Op {
    let op = Op::new(opcode, args);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![]);
    op.setfailargs(vec![].into());
    op
}

fn finish(pos: u32) -> Op {
    let op = Op::new(OpCode::Finish, &[]);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![]);
    op.setfailargs(vec![].into());
    op
}

#[test]
fn int_and_low32_mask_zero_extends() {
    // genop_int_and: (1<<32)-1 is MOV32, which keeps the low 32 bits.
    let mask = (1i64 << 32) - 1;
    assert_eq!(
        compile_and_run_int(OpCode::IntAnd, Some(mask), -1, 81),
        mask
    );
    assert_eq!(
        compile_and_run_int(OpCode::IntAnd, Some(mask), 0x1_0000_0005, 82),
        5
    );
    // A negative immediate that does fit stays AND, so -1 is unchanged.
    assert_eq!(compile_and_run_int(OpCode::IntAnd, Some(-1), -1, 83), -1);
}

/// `load_effective_addr` with a constant base: `AddressLoc` folds the
/// `ImmedLoc` into the static offset, directly when the sum fits a disp32 and
/// through `_addr_as_reg_offset` when it does not.
#[test]
fn load_effective_address_folds_a_constant_base() {
    for (token_id, base) in [(96i64, 0x1000i64), (97, 0x1234_5678_0000)] {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let token = JitCellToken::new(token_id);
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let i0 = inputargs[0].opref();
        let lea = Op::new(
            OpCode::LoadEffectiveAddress,
            &[
                rb(OpRef::const_int(base)),
                rb(i0),
                rb(OpRef::const_int(16)),
                rb(OpRef::const_int(3)),
            ],
        );
        lea.pos().set(OpRef::int_op(1));
        let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
        finish.pos().set(OpRef::void_op(2));
        finish.set_fail_arg_types(vec![Type::Int]);
        finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
        let ops = vec![OpRc::new(lea), OpRc::new(finish)];
        backend
            .compile_loop(&inputargs, &ops, &token)
            .expect("compile LOAD_EFFECTIVE_ADDRESS");
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), base + (5 << 3) + 16);
    }
}

#[test]
fn int_force_ge_zero_clamps_negatives() {
    // genop_int_force_ge_zero: negatives become 0, non-negatives pass through.
    let cases = [(-7, 0), (0, 0), (9, 9), (i64::MIN, 0), (i64::MAX, i64::MAX)];
    let mut token_id = 90u64;
    for (input, expected) in cases {
        let got = compile_and_run_int(OpCode::IntForceGeZero, None, input, token_id);
        token_id += 1;
        assert_eq!(got, expected, "int_force_ge_zero({input})");
    }
}

#[test]
fn float_neg_and_abs_keep_zero_and_nan_payload() {
    let pos_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0001);
    let (mut neg_backend, neg_token) = compile_float(OpCode::FloatNeg, 100);
    let neg_cases = [
        (0.0, (-0.0f64).to_bits()),
        (-0.0, 0.0f64.to_bits()),
        (1.5, (-1.5f64).to_bits()),
        (-2.25, 2.25f64.to_bits()),
        (pos_nan, neg_nan.to_bits()),
        (neg_nan, pos_nan.to_bits()),
    ];
    for (input, expected) in neg_cases {
        assert_eq!(
            float_bits(&mut neg_backend, &neg_token, input),
            expected,
            "float_neg({input:?})"
        );
    }

    let (mut abs_backend, abs_token) = compile_float(OpCode::FloatAbs, 101);
    let abs_cases = [
        (0.0, 0.0f64.to_bits()),
        (-0.0, 0.0f64.to_bits()),
        (-2.25, 2.25f64.to_bits()),
        (2.25, 2.25f64.to_bits()),
        (pos_nan, pos_nan.to_bits()),
        (neg_nan, pos_nan.to_bits()),
    ];
    for (input, expected) in abs_cases {
        assert_eq!(
            float_bits(&mut abs_backend, &abs_token, input),
            expected,
            "float_abs({input:?})"
        );
    }
}

#[test]
fn guard_gc_type_matches_header_typeid() {
    let (mut world, token) = gc_world(110);
    let i0 = OpRef::input_arg_ref(0);
    let expect = OpRef::const_int(world.child_tid);
    let ops = vec![
        void_guard(OpCode::GuardGcType, &[rb(i0), rb(expect)], 1),
        finish(2),
    ];
    let child = world.child;
    assert!(
        guard_finished(&mut world, &token, ops, child),
        "GUARD_GC_TYPE should pass for the object's typeid"
    );

    let (mut world, token) = gc_world(111);
    let i0 = OpRef::input_arg_ref(0);
    let expect = OpRef::const_int(world.child_tid);
    let ops = vec![
        void_guard(OpCode::GuardGcType, &[rb(i0), rb(expect)], 1),
        finish(2),
    ];
    let other = world.other;
    assert!(
        !guard_finished(&mut world, &token, ops, other),
        "GUARD_GC_TYPE should side-exit on a different typeid"
    );
}

#[test]
fn guard_is_object_tests_infobits_flag() {
    let (mut world, token) = gc_world(120);
    let i0 = OpRef::input_arg_ref(0);
    let ops = vec![void_guard(OpCode::GuardIsObject, &[rb(i0)], 1), finish(2)];
    let child = world.child;
    assert!(
        guard_finished(&mut world, &token, ops, child),
        "GUARD_IS_OBJECT should pass for an object type"
    );

    let (mut world, token) = gc_world(121);
    let i0 = OpRef::input_arg_ref(0);
    let ops = vec![void_guard(OpCode::GuardIsObject, &[rb(i0)], 1), finish(2)];
    let raw = world.raw;
    assert!(
        !guard_finished(&mut world, &token, ops, raw),
        "GUARD_IS_OBJECT should side-exit for a non-object type"
    );
}

#[test]
fn guard_subclass_typeid_range() {
    let (mut world, token) = gc_world(130);
    let i0 = OpRef::input_arg_ref(0);
    let classptr = OpRef::const_int(world.root_vtable as i64);
    let ops = vec![
        void_guard(OpCode::GuardSubclass, &[rb(i0), rb(classptr)], 1),
        finish(2),
    ];
    let child = world.child;
    assert!(
        guard_finished(&mut world, &token, ops, child),
        "GUARD_SUBCLASS should pass for a subclass"
    );

    let (mut world, token) = gc_world(131);
    let i0 = OpRef::input_arg_ref(0);
    let classptr = OpRef::const_int(world.root_vtable as i64);
    let ops = vec![
        void_guard(OpCode::GuardSubclass, &[rb(i0), rb(classptr)], 1),
        finish(2),
    ];
    let other = world.other;
    assert!(
        !guard_finished(&mut world, &token, ops, other),
        "GUARD_SUBCLASS should side-exit for an unrelated object"
    );
}

fn next_token_id() -> i64 {
    static NEXT: AtomicU64 = AtomicU64::new(0x00D4_0001);
    NEXT.fetch_add(1, Ordering::Relaxed) as i64
}

fn fresh_backend() -> DynasmBackend {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    backend
}

fn assert_float_bits(got: f64, expected: f64, what: &str) {
    if expected.is_nan() {
        assert!(got.is_nan(), "{what}: expected NaN, got {got}");
    } else {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "{what}: got {got}, expected {expected}"
        );
    }
}

fn host_float_binop(opcode: OpCode, lhs: f64, rhs: f64) -> f64 {
    match opcode {
        OpCode::FloatAdd => lhs + rhs,
        OpCode::FloatSub => lhs - rhs,
        OpCode::FloatMul => lhs * rhs,
        OpCode::FloatTrueDiv => lhs / rhs,
        _ => unreachable!("{opcode:?} is not a float binop"),
    }
}

/// `_binaryop` / `_consider_float_op`: one side is a constant (`ConstFloatLoc`
/// or, when it is the accumulator, the register `force_result_in_reg` filled).
fn compile_float_binop(
    opcode: OpCode,
    const_on_left: bool,
    konst: f64,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let i0 = inputargs[0].opref();
    let c = OpRef::const_float(konst);
    let args = if const_on_left {
        [rb(c), rb(i0)]
    } else {
        [rb(i0), rb(c)]
    };
    let body = Op::new(opcode, &args);
    body.pos().set(OpRef::float_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Float]);
    finish.setfailargs(vec![rb(OpRef::float_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| {
            panic!("compile {opcode:?} const_on_left={const_on_left} konst={konst}: {err:?}")
        });
    (backend, token)
}

fn run_float_binop(backend: &mut DynasmBackend, token: &JitCellToken, input: f64) -> f64 {
    let frame = backend.execute_token(token, &[Value::Float(input)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "float binop input={input}"
    );
    backend.get_float_value(&frame, 0)
}

#[test]
fn float_binop_reads_const_operand_and_nan() {
    // `_binaryop("ADDSD"|"SUBSD"|"MULSD"|"DIVSD")` keeps the non-accumulator
    // operand in place, including a constant pool load and a NaN input.
    let finite = [
        (OpCode::FloatAdd, 0.5, [1.5, 0.0, -0.5]),
        (OpCode::FloatSub, 2.0, [5.0, 2.0, 0.0]),
        (OpCode::FloatMul, 2.0, [3.0, -1.5, 0.0]),
        (OpCode::FloatTrueDiv, 2.0, [6.0, -4.0, 2.0]),
    ];
    for (opcode, konst, inputs) in finite {
        for const_on_left in [false, true] {
            let (mut backend, token) = compile_float_binop(opcode, const_on_left, konst);
            for input in inputs {
                let (lhs, rhs) = if const_on_left {
                    (konst, input)
                } else {
                    (input, konst)
                };
                let got = run_float_binop(&mut backend, &token, input);
                assert_float_bits(
                    got,
                    host_float_binop(opcode, lhs, rhs),
                    &format!("{opcode:?} const_on_left={const_on_left} {lhs} . {rhs}"),
                );
            }
            let nan_input = f64::from_bits(0x7ff8_0000_0000_0001);
            let (lhs, rhs) = if const_on_left {
                (konst, nan_input)
            } else {
                (nan_input, konst)
            };
            let got = run_float_binop(&mut backend, &token, nan_input);
            assert_float_bits(
                got,
                host_float_binop(opcode, lhs, rhs),
                &format!("{opcode:?} NaN input const_on_left={const_on_left}"),
            );
        }
    }
    let nan_const = f64::from_bits(0x7ff8_0000_0000_0001);
    for opcode in [
        OpCode::FloatAdd,
        OpCode::FloatSub,
        OpCode::FloatMul,
        OpCode::FloatTrueDiv,
    ] {
        for const_on_left in [false, true] {
            let (mut backend, token) = compile_float_binop(opcode, const_on_left, nan_const);
            let input = 1.5;
            let (lhs, rhs) = if const_on_left {
                (nan_const, input)
            } else {
                (input, nan_const)
            };
            let got = run_float_binop(&mut backend, &token, input);
            assert_float_bits(
                got,
                host_float_binop(opcode, lhs, rhs),
                &format!("{opcode:?} NaN const const_on_left={const_on_left}"),
            );
        }
    }
}

fn host_float_cmp(opcode: OpCode, lhs: f64, rhs: f64) -> i64 {
    let pass = match opcode {
        OpCode::FloatLt => lhs < rhs,
        OpCode::FloatLe => lhs <= rhs,
        OpCode::FloatEq => lhs == rhs,
        OpCode::FloatNe => lhs != rhs,
        OpCode::FloatGt => lhs > rhs,
        OpCode::FloatGe => lhs >= rhs,
        _ => unreachable!("{opcode:?} is not a float compare"),
    };
    i64::from(pass)
}

/// `_cmpop_float` / `_consider_float_cmp`: the constant stays a memory
/// operand and the other side is forced into a register.
fn compile_float_cmp(
    opcode: OpCode,
    const_on_left: bool,
    konst: f64,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let i0 = inputargs[0].opref();
    let c = OpRef::const_float(konst);
    let args = if const_on_left {
        [rb(c), rb(i0)]
    } else {
        [rb(i0), rb(c)]
    };
    let body = Op::new(opcode, &args);
    body.pos().set(OpRef::int_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| {
            panic!("compile {opcode:?} const_on_left={const_on_left} konst={konst}: {err:?}")
        });
    (backend, token)
}

fn run_float_cmp(backend: &mut DynasmBackend, token: &JitCellToken, input: f64) -> i64 {
    let frame = backend.execute_token(token, &[Value::Float(input)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "float cmp input={input}"
    );
    backend.get_int_value(&frame, 0)
}

#[test]
fn float_cmp_reads_const_operand_and_nan() {
    // `_cmpop_float`: unordered compares are false for LT/LE/EQ/GT/GE and
    // true for NE. `_if_parity_clear_zero_and_carry` is what makes B/E/BE
    // reject NaN. Both operand orders exercise `_consider_float_cmp`.
    let opcodes = [
        OpCode::FloatLt,
        OpCode::FloatLe,
        OpCode::FloatEq,
        OpCode::FloatNe,
        OpCode::FloatGt,
        OpCode::FloatGe,
    ];
    let inputs = [-1.0, 0.0, 1.0, 2.0, f64::NAN, -0.0];
    for opcode in opcodes {
        for const_on_left in [false, true] {
            for konst in [1.0, f64::NAN] {
                let (mut backend, token) = compile_float_cmp(opcode, const_on_left, konst);
                for input in inputs {
                    let (lhs, rhs) = if const_on_left {
                        (konst, input)
                    } else {
                        (input, konst)
                    };
                    let got = run_float_cmp(&mut backend, &token, input);
                    assert_eq!(
                        got,
                        host_float_cmp(opcode, lhs, rhs),
                        "{opcode:?} const_on_left={const_on_left} {lhs} ? {rhs}"
                    );
                }
            }
        }
    }
}

/// `_if_parity_clear_zero_and_carry`: `emit_forward_jump('NP')` is a
/// two-byte `JNP rel8` over `CMP_ri(ebp, 0)`, whose immediate is imm8.
#[test]
fn float_cmp_parity_jump_is_rel8() {
    let (_backend, token) = compile_float_cmp(OpCode::FloatEq, false, 1.0);
    assert_has(
        compiled_bytes(&token),
        &[0x7B, 0x04, 0x48, 0x83, 0xFD, 0x00],
        "jnp rel8; cmp rbp, 0",
    );
}

/// `genop_guard_guard_nonnull_class`: `CMP obj, 1` then a `JB rel8`
/// (`emit_forward_jump('B')`) over `_cmp_guard_class` to the guard's `JNE`.
#[test]
fn guard_nonnull_class_forward_jump_is_rel8() {
    #[repr(C)]
    struct Obj {
        typeptr: usize,
    }
    let class_a = 0x5A5A_0000usize;
    let class_b = 0x5B5B_0000usize;
    let mut obj_a = Obj { typeptr: class_a };
    let mut obj_b = Obj { typeptr: class_b };
    let mut backend = fresh_backend();
    backend.set_vtable_offset(Some(0));
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let p0 = inputargs[0].opref();
    let guard = Op::new(
        OpCode::GuardNonnullClass,
        &[rb(p0), rb(OpRef::const_int(class_a as i64))],
    );
    guard.pos().set(OpRef::void_op(1));
    guard.setdescr(make_resume_guard_descr_typed(vec![]));
    guard.setfailargs(vec![].into());
    let finish = Op::new(OpCode::Finish, &[]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![]);
    finish.setfailargs(vec![].into());
    let ops = vec![OpRc::new(guard), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile GUARD_NONNULL_CLASS: {err:?}"));
    let code = compiled_bytes(&token);
    assert!(
        code.windows(4)
            .any(|w| w[0] == 0x83 && w[1] & 0xF8 == 0xF8 && w[2] == 0x01 && w[3] == 0x72),
        "cmp obj, 1; jb rel8"
    );
    let run = |backend: &mut DynasmBackend, ptr: usize| {
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        backend.get_latest_descr(&frame).is_finish()
    };
    assert!(run(&mut backend, &mut obj_a as *mut Obj as usize));
    assert!(!run(&mut backend, &mut obj_b as *mut Obj as usize));
    assert!(!run(&mut backend, 0));
}

// Test-only bridge into a `CALL_MAY_FORCE` helper. The helper is `extern "C"`
// and has no Rust argument for the backend that owns the force token.
thread_local! {
    static FORCE_BACKEND: Cell<*const DynasmBackend> = const { Cell::new(std::ptr::null()) };
    static FORCE_SEEN: Cell<Option<i64>> = const { Cell::new(None) };
}

struct ClearForceBackend;

impl Drop for ClearForceBackend {
    fn drop(&mut self) {
        FORCE_BACKEND.with(|cell| cell.set(std::ptr::null()));
    }
}

extern "C" fn force_and_record(force_token: i64) {
    let ptr = FORCE_BACKEND.with(Cell::get);
    if ptr.is_null() {
        FORCE_SEEN.with(|cell| cell.set(Some(i64::MIN)));
        return;
    }
    let backend = unsafe { &*ptr };
    let recorded = catch_unwind(AssertUnwindSafe(|| {
        match backend.force(GcRef(force_token as usize)) {
            Some(frame) => backend.get_int_value(&frame, 0),
            None => -2,
        }
    }));
    let value = match recorded {
        Ok(value) => value,
        Err(_) => -3,
    };
    FORCE_SEEN.with(|cell| cell.set(Some(value)));
}

extern "C" fn force_nop(_force_token: i64) {}

#[derive(Debug)]
struct MayForceCallDescr {
    arg_types: Vec<Type>,
}

impl Descr for MayForceCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }

    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for MayForceCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        Type::Void
    }

    fn result_size(&self) -> usize {
        0
    }

    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
}

/// `_store_force_index` then `genop_guard_guard_not_forced`.
fn compile_may_force(helper: extern "C" fn(i64)) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let force_token = Op::new(OpCode::ForceToken, &[]);
    force_token.pos().set(OpRef::ref_op(1));

    let call = Op::new(
        OpCode::CallMayForceN,
        &[
            rb(OpRef::const_int(helper as usize as i64)),
            rb(OpRef::ref_op(1)),
        ],
    );
    call.pos().set(OpRef::void_op(2));
    call.setdescr(Arc::new(MayForceCallDescr {
        arg_types: vec![Type::Ref],
    }) as DescrRef);

    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.pos().set(OpRef::void_op(3));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int]);
    guard.setfailargs(vec![rb(i0)].into());

    let finish = Op::new(OpCode::Finish, &[rb(i0)]);
    finish.pos().set(OpRef::void_op(4));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(i0)].into());

    let ops: Vec<OpRc> = vec![force_token, call, guard, finish]
        .into_iter()
        .map(OpRc::new)
        .collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_MAY_FORCE: {err:?}"));
    (backend, token)
}

fn run_may_force(helper: extern "C" fn(i64)) -> (bool, i64, Option<i64>) {
    FORCE_SEEN.with(|cell| cell.set(None));
    let (backend, token) = compile_may_force(helper);
    FORCE_BACKEND.with(|cell| cell.set(&backend as *const DynasmBackend));
    let _clear = ClearForceBackend;
    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let finished = backend.get_latest_descr(&frame).is_finish();
    let after = backend.get_int_value(&frame, 0);
    let during = FORCE_SEEN.with(Cell::get);
    (finished, after, during)
}

#[test]
fn may_force_guard_not_forced_reads_failarg() {
    // `_store_force_index` publishes `jf_force_descr` and leaves `jf_descr`
    // alone. A helper that does not force falls through `GUARD_NOT_FORCED`
    // (`CMP [jf_descr], 0`). A helper that forces copies that descr into
    // `jf_descr`, so the guard fails and the live fail arg is readable both
    // during `force` and on the side exit. That failure stub loads and
    // clears the process-wide exception cells.
    let _lock = lock_exception_cells();
    let (finished, after, during) = run_may_force(force_nop);
    assert!(during.is_none(), "nop helper must not force");
    assert!(finished, "GUARD_NOT_FORCED must pass when nothing forces");
    assert_eq!(after, 42);

    let (finished, after, during) = run_may_force(force_and_record);
    assert_eq!(during, Some(42), "force() must observe the live fail arg");
    assert!(!finished, "GUARD_NOT_FORCED must fail after force()");
    assert_eq!(after, 42, "side exit must keep the live fail arg");
}

#[test]
fn guard_not_forced_2_constant_failarg_is_hole() {
    // `store_force_descr` → `store_info_on_descr`: an absent fail arg is
    // None → 0xFFFF, and `get_int_value` of that hole is 0. Failargs must
    // not contain Const (`compute_vars_longevity`). The live fail arg is
    // still the spilled input, readable after `force`.
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let force_token = Op::new(OpCode::ForceToken, &[]);
    force_token.pos().set(OpRef::ref_op(1));

    let guard = Op::new(OpCode::GuardNotForced2, &[]);
    guard.pos().set(OpRef::void_op(2));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int, Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int, Type::Int]);
    guard.setfailargs(vec![rb(i0), Operand::none()].into());

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
        .expect("compile GUARD_NOT_FORCED_2");

    let returned = backend.execute_token(&token, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&returned).is_finish());
    let raw_token = backend.get_ref_value(&returned, 0);
    assert!(backend.is_force_token_armed(raw_token));
    let forced = backend.force(raw_token).expect("armed force token");
    let descr = backend.get_latest_descr(&forced);
    assert!(!descr.is_finish());
    assert_eq!(descr.rd_locs().get(1), Some(&0xFFFF));
    assert_eq!(backend.get_int_value(&forced, 0), 42);
    assert_eq!(backend.get_int_value(&forced, 1), 0);
}

/// `JIT_EXC_VALUE` / `JIT_EXC_TYPE` (`pos_exc_value` / `pos_exception`).
/// The cells are process-wide and the JIT clears them with plain stores,
/// so every test that pre-raises or whose executed code reaches them
/// shares this lock.
static EXCEPTION_CELLS_LOCK: Mutex<()> = Mutex::new(());

fn lock_exception_cells() -> std::sync::MutexGuard<'static, ()> {
    EXCEPTION_CELLS_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

static EXC_OBJ: [i64; 2] = [0x5151_0D11, 0];

struct ClearExc;

impl Drop for ClearExc {
    fn drop(&mut self) {
        majit_backend_dynasm::jit_exc_clear();
    }
}

extern "C" fn cond_call_raise() {
    majit_backend_dynasm::jit_exc_raise(EXC_OBJ.as_ptr() as i64);
}

extern "C" fn cond_call_quiet() {}

/// `genop_guard_guard_no_exception`: `COND_CALL` + `GUARD_NO_EXCEPTION`.
fn compile_cond_call_no_exception(
    predicate: i64,
    func: extern "C" fn(),
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CondCallN,
        &[
            rb(OpRef::const_int(predicate)),
            rb(OpRef::const_int(func as usize as i64)),
        ],
    );
    call.pos().set(OpRef::void_op(0));
    let guard = Op::new(OpCode::GuardNoException, &[]);
    guard.pos().set(OpRef::void_op(1));
    guard.set_fail_arg_types(vec![]);
    guard.setfailargs(vec![].into());
    let finish = Op::new(OpCode::Finish, &[]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![]);
    finish.setfailargs(vec![].into());
    let ops = vec![OpRc::new(call), OpRc::new(guard), OpRc::new(finish)];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile COND_CALL + GUARD_NO_EXCEPTION: {err:?}"));
    (backend, token)
}

#[test]
fn cond_call_guard_no_exception_checks_the_call_path() {
    // `genop_guard_guard_no_exception` / `generate_guard_no_exception`: the
    // fast path of a false `COND_CALL` emits no exception check, so a
    // pre-raised exception is still pending at `FINISH`. The call path runs
    // `CMP [pos_exception], 0` and, on failure, `grab_exc_value` reads the
    // value the recovery stub stashed.
    let _lock = lock_exception_cells();
    let _clear = ClearExc;
    majit_backend_dynasm::jit_exc_clear();

    let (backend, token) = compile_cond_call_no_exception(0, cond_call_raise);
    majit_backend_dynasm::jit_exc_raise(EXC_OBJ.as_ptr() as i64);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "predicate 0 must skip the call and the exception check"
    );
    assert!(
        majit_backend_dynasm::jit_exc_is_pending(),
        "the don't-call edge must leave the pre-raised exception pending"
    );
    majit_backend_dynasm::jit_exc_clear();

    let (backend, token) = compile_cond_call_no_exception(1, cond_call_raise);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "a raising call must fail GUARD_NO_EXCEPTION"
    );
    assert_eq!(
        backend.grab_exc_value(&frame),
        GcRef(EXC_OBJ.as_ptr() as usize)
    );
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
    majit_backend_dynasm::jit_exc_clear();

    let (backend, token) = compile_cond_call_no_exception(1, cond_call_quiet);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "a quiet call must pass GUARD_NO_EXCEPTION"
    );
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
}

static FUSED_HITS: AtomicUsize = AtomicUsize::new(0);
static UNFUSED_HITS: AtomicUsize = AtomicUsize::new(0);
static VALUE_I_SKIP_HITS: AtomicUsize = AtomicUsize::new(0);
static VALUE_I_TAKE_HITS: AtomicUsize = AtomicUsize::new(0);
static VALUE_R_SKIP_HITS: AtomicUsize = AtomicUsize::new(0);
static VALUE_R_TAKE_HITS: AtomicUsize = AtomicUsize::new(0);

/// Clobber every register the cond-call helper is supposed to restore.
/// A normal `extern "C"` callee would save the callee-saved set itself, so
/// the live value would survive even when `cond_call_slowpath` did not.
macro_rules! clobber_hits {
    ($name:ident, $hits:ident) => {
        #[unsafe(naked)]
        extern "C" fn $name() {
            naked_asm!(
                "lock inc qword ptr [rip + {hits}]",
                "xor eax, eax",
                "xor ecx, ecx",
                "xor edx, edx",
                "xor ebx, ebx",
                "xor esi, esi",
                "xor edi, edi",
                "xor r8, r8",
                "xor r9, r9",
                "xor r10, r10",
                "xor r11, r11",
                "xor r12, r12",
                "xor r14, r14",
                "xor r15, r15",
                "pxor xmm0, xmm0",
                "pxor xmm1, xmm1",
                "pxor xmm2, xmm2",
                "pxor xmm3, xmm3",
                "pxor xmm4, xmm4",
                "pxor xmm5, xmm5",
                "ret",
                hits = sym $hits,
            );
        }
    };
}

clobber_hits!(cond_call_fused_hit, FUSED_HITS);
clobber_hits!(cond_call_unfused_hit, UNFUSED_HITS);

extern "C" fn cond_call_value_i_skip() -> i64 {
    VALUE_I_SKIP_HITS.fetch_add(1, Ordering::SeqCst);
    0x5A5A_5A5A_5A5A_5A5A_u64 as i64
}

extern "C" fn cond_call_value_i_take() -> i64 {
    VALUE_I_TAKE_HITS.fetch_add(1, Ordering::SeqCst);
    0x5A5A_5A5A_5A5A_5A5A_u64 as i64
}

extern "C" fn cond_call_value_r_skip() -> i64 {
    VALUE_R_SKIP_HITS.fetch_add(1, Ordering::SeqCst);
    0x5A5A_5A5A_5A5A_5A5A_u64 as i64
}

extern "C" fn cond_call_value_r_take() -> i64 {
    VALUE_R_TAKE_HITS.fetch_add(1, Ordering::SeqCst);
    0x5A5A_5A5A_5A5A_5A5A_u64 as i64
}

fn compile_fused_cond_call(func: i64) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let cmp = Op::new(OpCode::IntEq, &[rb(i0), rb(OpRef::const_int(1))]);
    cmp.pos().set(OpRef::int_op(1));
    let call = Op::new(
        OpCode::CondCallN,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(func))],
    );
    call.pos().set(OpRef::void_op(0));
    let ops = vec![
        OpRc::new(cmp),
        OpRc::new(call),
        OpRc::new(finish_of(i0, Type::Int, 2)),
    ];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile fused COND_CALL: {err:?}"));
    (backend, token)
}

fn compile_unfused_cond_call(
    predicate: i64,
    func: i64,
    finish_float: bool,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Float, 1),
    ];
    let i0 = inputargs[0].opref();
    let f0 = inputargs[1].opref();
    let call = Op::new(
        OpCode::CondCallN,
        &[rb(OpRef::const_int(predicate)), rb(OpRef::const_int(func))],
    );
    call.pos().set(OpRef::void_op(0));
    let finish = if finish_float {
        finish_of(f0, Type::Float, 2)
    } else {
        finish_of(i0, Type::Int, 2)
    };
    let ops = vec![OpRc::new(call), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile unfused COND_CALL: {err:?}"));
    (backend, token)
}

fn compile_cond_call_value(opcode: OpCode, func: i64, ty: Type) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(ty, 0)];
    let input = inputargs[0].opref();
    let result = match ty {
        Type::Int => OpRef::int_op(1),
        Type::Ref => OpRef::ref_op(1),
        other => panic!("cond_call value result type {other:?}"),
    };
    let call = Op::new(opcode, &[rb(input), rb(OpRef::const_int(func))]);
    call.pos().set(result);
    call.setdescr(call_descr(vec![], ty, true, 8));
    let ops = vec![OpRc::new(call), OpRc::new(finish_of(result, ty, 2))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    (backend, token)
}

#[test]
fn cond_call_fused_compare_calls_only_when_true() {
    // `next_op_can_accept_cc` fuses `IntEq` into `CondCallN`. The fast path
    // is one not-taken `jcc` of that cc. False does not call. True calls,
    // and the input survives the callee's clobber.
    let func = cond_call_fused_hit as *const () as usize as i64;
    FUSED_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_fused_cond_call(func);
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 0);
    assert_eq!(FUSED_HITS.load(Ordering::SeqCst), 0);

    FUSED_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_fused_cond_call(func);
    let frame = backend.execute_token(&token, &[Value::Int(1)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 1);
    assert_eq!(FUSED_HITS.load(Ordering::SeqCst), 1);
}

#[test]
fn cond_call_unfused_preserves_a_live_value() {
    // Constant predicate: `load_condition_into_cc` emits `test` and `CC_NE`.
    // A live int and a live float survive both edges. The callee clobbers
    // the managed registers, including the callee-saved ones.
    let func = cond_call_unfused_hit as *const () as usize as i64;
    let live_int = 0x1111_2222_3333_4444i64;
    let live_float = -2.5f64;

    UNFUSED_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_unfused_cond_call(0, func, false);
    let frame = backend.execute_token(&token, &[Value::Int(live_int), Value::Float(live_float)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), live_int);
    assert_eq!(UNFUSED_HITS.load(Ordering::SeqCst), 0);

    UNFUSED_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_unfused_cond_call(1, func, false);
    let frame = backend.execute_token(&token, &[Value::Int(live_int), Value::Float(live_float)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), live_int);
    assert_eq!(UNFUSED_HITS.load(Ordering::SeqCst), 1);

    UNFUSED_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_unfused_cond_call(0, func, true);
    let frame = backend.execute_token(&token, &[Value::Int(live_int), Value::Float(live_float)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_float_bits(
        backend.get_float_value(&frame, 0),
        live_float,
        "unfused don't-call float",
    );
    assert_eq!(UNFUSED_HITS.load(Ordering::SeqCst), 0);

    UNFUSED_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_unfused_cond_call(1, func, true);
    let frame = backend.execute_token(&token, &[Value::Int(live_int), Value::Float(live_float)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_float_bits(
        backend.get_float_value(&frame, 0),
        live_float,
        "unfused call float",
    );
    assert_eq!(UNFUSED_HITS.load(Ordering::SeqCst), 1);
}

#[test]
fn cond_call_value_nonzero_skips_the_call() {
    // `cond_call` tests `resloc` and takes the slow path on `CC_E`.
    let live_int = 0x1111_2222_3333_4444i64;
    VALUE_I_SKIP_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_cond_call_value(
        OpCode::CondCallValueI,
        cond_call_value_i_skip as *const () as usize as i64,
        Type::Int,
    );
    let frame = backend.execute_token(&token, &[Value::Int(live_int)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), live_int);
    assert_eq!(VALUE_I_SKIP_HITS.load(Ordering::SeqCst), 0);

    let live_ref = GcRef(0x1234);
    VALUE_R_SKIP_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_cond_call_value(
        OpCode::CondCallValueR,
        cond_call_value_r_skip as *const () as usize as i64,
        Type::Ref,
    );
    let frame = backend.execute_token(&token, &[Value::Ref(live_ref)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_ref_value(&frame, 0), live_ref);
    assert_eq!(VALUE_R_SKIP_HITS.load(Ordering::SeqCst), 0);
}

#[test]
fn cond_call_value_zero_writes_the_helper_result() {
    let helper_bits = 0x5A5A_5A5A_5A5A_5A5A_u64 as i64;
    VALUE_I_TAKE_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_cond_call_value(
        OpCode::CondCallValueI,
        cond_call_value_i_take as *const () as usize as i64,
        Type::Int,
    );
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), helper_bits);
    assert_eq!(VALUE_I_TAKE_HITS.load(Ordering::SeqCst), 1);

    VALUE_R_TAKE_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_cond_call_value(
        OpCode::CondCallValueR,
        cond_call_value_r_take as *const () as usize as i64,
        Type::Ref,
    );
    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0))]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_ref_value(&frame, 0),
        GcRef(helper_bits as usize)
    );
    assert_eq!(VALUE_R_TAKE_HITS.load(Ordering::SeqCst), 1);
}

static CALLER_SAVED_HITS: AtomicUsize = AtomicUsize::new(0);
static CALLEE_SAVED_HITS: AtomicUsize = AtomicUsize::new(0);

clobber_hits!(cond_call_caller_saved_hit, CALLER_SAVED_HITS);
clobber_hits!(cond_call_callee_saved_hit, CALLEE_SAVED_HITS);

/// One `INT_ADD` whose result is the only value live across `COND_CALL`.
/// No float op, so `xrm.reg_bindings` is empty. The result is the first
/// GPR allocation, which is in `save_around_call_regs`.
fn compile_caller_saved_cond_call(func: i64) -> (DynasmBackend, JitCellToken, usize) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let add = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(1))]);
    add.pos().set(OpRef::int_op(1));
    let call = Op::new(
        OpCode::CondCallN,
        &[rb(OpRef::const_int(1)), rb(OpRef::const_int(func))],
    );
    call.pos().set(OpRef::void_op(0));
    let ops = vec![
        OpRc::new(add),
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    let info = backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile caller-saved COND_CALL: {err:?}"));
    (backend, token, info.code_size)
}

#[test]
fn cond_call_only_caller_saved_lives_survive() {
    // No live float. The only live core register is the add result, and
    // that register is in `save_around_call_regs`. `cond_call` selects
    // `cond_call_slowpath[1]` (`callee_only`, no floats).
    let func = cond_call_caller_saved_hit as *const () as usize as i64;
    CALLER_SAVED_HITS.store(0, Ordering::SeqCst);
    let (backend, token, code_size) = compile_caller_saved_cond_call(func);
    assert_eq!(
        cond_call_slowpath_index(loop_code(&token, code_size), func as u64),
        1
    );
    let frame = backend.execute_token(&token, &[Value::Int(40)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 41);
    assert_eq!(CALLER_SAVED_HITS.load(Ordering::SeqCst), 1);
}

fn compile_callee_saved_cond_call(func: i64) -> (DynasmBackend, JitCellToken, usize) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let mut ops = Vec::new();
    let mut live = Vec::new();
    for n in 1..=4 {
        let add = Op::new(
            OpCode::IntAdd,
            &[rb(i0), rb(OpRef::const_int(i64::from(n)))],
        );
        add.pos().set(OpRef::int_op(n));
        ops.push(add);
        live.push(rb(OpRef::int_op(n)));
    }
    let call = Op::new(
        OpCode::CondCallN,
        &[rb(OpRef::const_int(1)), rb(OpRef::const_int(func))],
    );
    call.pos().set(OpRef::void_op(0));
    ops.push(call);
    // `genop_finish` writes only arglocs[0] into frame slot 0, and the
    // finish descr leaves `rd_locs` empty. A failing guard records every
    // live register there, which is what `get_int_value` reads.
    let eq = Op::new(
        OpCode::IntEq,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(0))],
    );
    eq.pos().set(OpRef::int_op(5));
    ops.push(eq);
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(5))]);
    guard.pos().set(OpRef::void_op(6));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int; 4]));
    guard.set_fail_arg_types(vec![Type::Int; 4]);
    guard.setfailargs(live.into());
    ops.push(guard);
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    let info = backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile callee-saved COND_CALL: {err:?}"));
    (backend, token, info.code_size)
}

#[test]
fn cond_call_live_callee_saved_survives() {
    // Four live adds. `ebx` is outside `save_around_call_regs`, so
    // `callee_only` is false. No live float. `cond_call` selects
    // `cond_call_slowpath[0]` (every core register, no floats). The
    // naked callee xors `ebx`/`esi`/`edi`/`r12`/`r14`/`r15`.
    let func = cond_call_callee_saved_hit as *const () as usize as i64;
    CALLEE_SAVED_HITS.store(0, Ordering::SeqCst);
    let (backend, token, code_size) = compile_callee_saved_cond_call(func);
    assert_eq!(
        cond_call_slowpath_index(loop_code(&token, code_size), func as u64),
        0
    );
    let frame = backend.execute_token(&token, &[Value::Int(10)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(CALLEE_SAVED_HITS.load(Ordering::SeqCst), 1);
    for (index, expected) in [11, 12, 13, 14].into_iter().enumerate() {
        assert_eq!(
            backend.get_int_value(&frame, index),
            expected,
            "live add {index}"
        );
    }
}

const LIVE_ADD_COUNT: u32 = 16;
const EXTRA_ADD_NS: [u32; 3] = [14, 15, 16];
const COND_CALL_EXTRA_HELPER_RESULT: i64 = 0x5A5A_5A5A_5A5A_5A5A_u64 as i64;

// One set of cells per helper, so the two tests running in parallel never
// read or reset each other's.
static COND_CALL_EXTRA0: AtomicI64 = AtomicI64::new(0);
static COND_CALL_EXTRA1: AtomicI64 = AtomicI64::new(0);
static COND_CALL_EXTRA2: AtomicI64 = AtomicI64::new(0);
static COND_CALL_EXTRA_HITS: AtomicUsize = AtomicUsize::new(0);
static COND_CALL_VALUE_EXTRA0: AtomicI64 = AtomicI64::new(0);
static COND_CALL_VALUE_EXTRA1: AtomicI64 = AtomicI64::new(0);
static COND_CALL_VALUE_EXTRA2: AtomicI64 = AtomicI64::new(0);
static COND_CALL_VALUE_EXTRA_HITS: AtomicUsize = AtomicUsize::new(0);
static EAX_LIVE_HITS: AtomicUsize = AtomicUsize::new(0);

/// The cells one clobber helper writes: its three recorded extras and its
/// hit count.
struct ExtraCells {
    extras: [&'static AtomicI64; 3],
    hits: &'static AtomicUsize,
}

const COND_CALL_EXTRAS: ExtraCells = ExtraCells {
    extras: [&COND_CALL_EXTRA0, &COND_CALL_EXTRA1, &COND_CALL_EXTRA2],
    hits: &COND_CALL_EXTRA_HITS,
};

const COND_CALL_VALUE_EXTRAS: ExtraCells = ExtraCells {
    extras: [
        &COND_CALL_VALUE_EXTRA0,
        &COND_CALL_VALUE_EXTRA1,
        &COND_CALL_VALUE_EXTRA2,
    ],
    hits: &COND_CALL_VALUE_EXTRA_HITS,
};

impl ExtraCells {
    fn reset(&self) {
        for cell in self.extras {
            cell.store(0, Ordering::SeqCst);
        }
        self.hits.store(0, Ordering::SeqCst);
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    fn assert_recorded(&self, expected: [i64; 3]) {
        for (cell, expected) in self.extras.iter().zip(expected) {
            assert_eq!(cell.load(Ordering::SeqCst), expected);
        }
    }
}

fn live_add_ops(base: OpRef) -> (Vec<Op>, Vec<Operand>) {
    let mut ops = Vec::new();
    let mut live = Vec::new();
    for n in 1..=LIVE_ADD_COUNT {
        let add = Op::new(
            OpCode::IntAdd,
            &[rb(base), rb(OpRef::const_int(i64::from(n)))],
        );
        add.pos().set(OpRef::int_op(n));
        ops.push(add);
        live.push(rb(OpRef::int_op(n)));
    }
    (ops, live)
}

fn extra_operands() -> [Operand; 3] {
    EXTRA_ADD_NS.map(|n| rb(OpRef::int_op(n)))
}

fn extra_expected(base: i64) -> [i64; 3] {
    EXTRA_ADD_NS.map(|n| base + i64::from(n))
}

fn failing_guard_with_lives(
    lives: Vec<Operand>,
    types: Vec<Type>,
    pos_eq: u32,
    pos_guard: u32,
) -> [Op; 2] {
    let eq = Op::new(
        OpCode::IntEq,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(0))],
    );
    eq.pos().set(OpRef::int_op(pos_eq));
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(pos_eq))]);
    guard.pos().set(OpRef::void_op(pos_guard));
    guard.setdescr(make_resume_guard_descr_typed(types.clone()));
    guard.set_fail_arg_types(types);
    guard.setfailargs(lives.into());
    [eq, guard]
}

fn assert_live_adds(
    backend: &DynasmBackend,
    frame: &majit_backend::DeadFrame,
    base: i64,
    offset: usize,
) {
    for n in 1..=LIVE_ADD_COUNT {
        assert_eq!(
            backend.get_int_value(frame, offset + (n as usize - 1)),
            base + i64::from(n),
            "live add {n}"
        );
    }
}

/// Record the three extra arguments, then clobber every managed register.
/// Extra args live in non-ABI registers / frame slots, so the taken path
/// must remap them (`CondCallSlowPath.generate_body`).
#[cfg(not(target_os = "windows"))]
#[unsafe(naked)]
extern "C" fn cond_call_extra_clobber() {
    naked_asm!(
        "lock inc qword ptr [rip + {hits}]",
        "mov qword ptr [rip + {e0}], rdi",
        "mov qword ptr [rip + {e1}], rsi",
        "mov qword ptr [rip + {e2}], rdx",
        "xor eax, eax",
        "xor ecx, ecx",
        "xor edx, edx",
        "xor ebx, ebx",
        "xor esi, esi",
        "xor edi, edi",
        "xor r8, r8",
        "xor r9, r9",
        "xor r10, r10",
        "xor r11, r11",
        "xor r12, r12",
        "xor r14, r14",
        "xor r15, r15",
        "ret",
        hits = sym COND_CALL_EXTRA_HITS,
        e0 = sym COND_CALL_EXTRA0,
        e1 = sym COND_CALL_EXTRA1,
        e2 = sym COND_CALL_EXTRA2,
    );
}

#[cfg(target_os = "windows")]
#[unsafe(naked)]
extern "C" fn cond_call_extra_clobber() {
    naked_asm!(
        "lock inc qword ptr [rip + {hits}]",
        "mov qword ptr [rip + {e0}], rcx",
        "mov qword ptr [rip + {e1}], rdx",
        "mov qword ptr [rip + {e2}], r8",
        "xor eax, eax",
        "xor ecx, ecx",
        "xor edx, edx",
        "xor ebx, ebx",
        "xor esi, esi",
        "xor edi, edi",
        "xor r8, r8",
        "xor r9, r9",
        "xor r10, r10",
        "xor r11, r11",
        "xor r12, r12",
        "xor r14, r14",
        "xor r15, r15",
        "ret",
        hits = sym COND_CALL_EXTRA_HITS,
        e0 = sym COND_CALL_EXTRA0,
        e1 = sym COND_CALL_EXTRA1,
        e2 = sym COND_CALL_EXTRA2,
    );
}

#[cfg(not(target_os = "windows"))]
#[unsafe(naked)]
extern "C" fn cond_call_value_extra_clobber() {
    naked_asm!(
        "lock inc qword ptr [rip + {hits}]",
        "mov qword ptr [rip + {e0}], rdi",
        "mov qword ptr [rip + {e1}], rsi",
        "mov qword ptr [rip + {e2}], rdx",
        "xor ecx, ecx",
        "xor edx, edx",
        "xor ebx, ebx",
        "xor esi, esi",
        "xor edi, edi",
        "xor r8, r8",
        "xor r9, r9",
        "xor r10, r10",
        "xor r11, r11",
        "xor r12, r12",
        "xor r14, r14",
        "xor r15, r15",
        "mov rax, 0x5A5A5A5A5A5A5A5A",
        "ret",
        hits = sym COND_CALL_VALUE_EXTRA_HITS,
        e0 = sym COND_CALL_VALUE_EXTRA0,
        e1 = sym COND_CALL_VALUE_EXTRA1,
        e2 = sym COND_CALL_VALUE_EXTRA2,
    );
}

#[cfg(target_os = "windows")]
#[unsafe(naked)]
extern "C" fn cond_call_value_extra_clobber() {
    naked_asm!(
        "lock inc qword ptr [rip + {hits}]",
        "mov qword ptr [rip + {e0}], rcx",
        "mov qword ptr [rip + {e1}], rdx",
        "mov qword ptr [rip + {e2}], r8",
        "xor ecx, ecx",
        "xor edx, edx",
        "xor ebx, ebx",
        "xor esi, esi",
        "xor edi, edi",
        "xor r8, r8",
        "xor r9, r9",
        "xor r10, r10",
        "xor r11, r11",
        "xor r12, r12",
        "xor r14, r14",
        "xor r15, r15",
        "mov rax, 0x5A5A5A5A5A5A5A5A",
        "ret",
        hits = sym COND_CALL_VALUE_EXTRA_HITS,
        e0 = sym COND_CALL_VALUE_EXTRA0,
        e1 = sym COND_CALL_VALUE_EXTRA1,
        e2 = sym COND_CALL_VALUE_EXTRA2,
    );
}

clobber_hits!(cond_call_eax_live_hit, EAX_LIVE_HITS);

fn compile_cond_call_three_extras(predicate: i64, func: i64) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let base = inputargs[0].opref();
    let (mut ops, lives) = live_add_ops(base);
    let [e0, e1, e2] = extra_operands();
    let call = Op::new(
        OpCode::CondCallN,
        &[
            rb(OpRef::const_int(predicate)),
            rb(OpRef::const_int(func)),
            e0,
            e1,
            e2,
        ],
    );
    call.pos().set(OpRef::void_op(0));
    ops.push(call);
    let types = vec![Type::Int; LIVE_ADD_COUNT as usize];
    ops.extend(failing_guard_with_lives(
        lives,
        types,
        LIVE_ADD_COUNT + 1,
        LIVE_ADD_COUNT + 2,
    ));
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile COND_CALL extras: {err:?}"));
    (backend, token)
}

fn compile_cond_call_value_three_extras(func: i64) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let value = inputargs[0].opref();
    let base = inputargs[1].opref();
    let (mut ops, mut lives) = live_add_ops(base);
    let [e0, e1, e2] = extra_operands();
    let result = OpRef::int_op(LIVE_ADD_COUNT + 1);
    let call = Op::new(
        OpCode::CondCallValueI,
        &[rb(value), rb(OpRef::const_int(func)), e0, e1, e2],
    );
    call.pos().set(result);
    call.setdescr(call_descr(
        vec![Type::Int, Type::Int, Type::Int],
        Type::Int,
        true,
        8,
    ));
    ops.push(call);
    let mut failargs = vec![rb(result)];
    failargs.append(&mut lives);
    let types = vec![Type::Int; LIVE_ADD_COUNT as usize + 1];
    ops.extend(failing_guard_with_lives(
        failargs,
        types,
        LIVE_ADD_COUNT + 2,
        LIVE_ADD_COUNT + 3,
    ));
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile COND_CALL_VALUE extras: {err:?}"));
    (backend, token)
}

#[test]
fn cond_call_three_extras_remap_on_taken_path() {
    // Sixteen live adds fill the GPR pool, so extras 14..=16 sit in frame
    // slots. `consider_cond_call` keeps those locations; the taken path
    // remaps them into `cond_call_register_arguments`.
    let func = cond_call_extra_clobber as *const () as usize as i64;
    let base = 10i64;
    let expected = extra_expected(base);

    COND_CALL_EXTRAS.reset();
    let (backend, token) = compile_cond_call_three_extras(0, func);
    let frame = backend.execute_token(&token, &[Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_EXTRAS.hits(), 0);
    assert_live_adds(&backend, &frame, base, 0);

    COND_CALL_EXTRAS.reset();
    let (backend, token) = compile_cond_call_three_extras(1, func);
    let frame = backend.execute_token(&token, &[Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_EXTRAS.hits(), 1);
    COND_CALL_EXTRAS.assert_recorded(expected);
    assert_live_adds(&backend, &frame, base, 0);
}

#[test]
fn cond_call_value_i_three_extras_taken_and_not_taken() {
    // Same spilled extras as the void form. Not-taken keeps `args[0]`;
    // taken remaps extras and writes the helper word into `resloc`.
    let func = cond_call_value_extra_clobber as *const () as usize as i64;
    let base = 10i64;
    let expected = extra_expected(base);
    let live_value = 0x1111_2222_3333_4444i64;

    COND_CALL_VALUE_EXTRAS.reset();
    let (backend, token) = compile_cond_call_value_three_extras(func);
    let frame = backend.execute_token(&token, &[Value::Int(live_value), Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_VALUE_EXTRAS.hits(), 0);
    assert_eq!(backend.get_int_value(&frame, 0), live_value);
    assert_live_adds(&backend, &frame, base, 1);

    COND_CALL_VALUE_EXTRAS.reset();
    let (backend, token) = compile_cond_call_value_three_extras(func);
    let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_VALUE_EXTRAS.hits(), 1);
    COND_CALL_VALUE_EXTRAS.assert_recorded(expected);
    assert_eq!(
        backend.get_int_value(&frame, 0),
        COND_CALL_EXTRA_HELPER_RESULT
    );
    assert_live_adds(&backend, &frame, base, 1);
}

extern "C" fn eax_live_return() -> i64 {
    0x1111_2222_3333_4444
}

fn compile_eax_live_across_cond_call(func: i64) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CallI,
        &[rb(OpRef::const_int(
            eax_live_return as *const () as usize as i64,
        ))],
    );
    call.pos().set(OpRef::int_op(1));
    call.setdescr(call_descr(vec![], Type::Int, true, 8));
    let cond = Op::new(
        OpCode::CondCallN,
        &[rb(OpRef::const_int(1)), rb(OpRef::const_int(func))],
    );
    cond.pos().set(OpRef::void_op(0));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(cond),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile eax-live COND_CALL: {err:?}"));
    (backend, token)
}

#[test]
fn cond_call_taken_restores_live_eax() {
    // `after_call` leaves the CALL_I result in eax. That value is not the
    // cond-call result (`resloc` is None). `CondCallSlowPath.generate_body`
    // saves eax because it is in `should_be_saved`, then restores it after
    // the helper (`restore_eax`).
    let func = cond_call_eax_live_hit as *const () as usize as i64;
    EAX_LIVE_HITS.store(0, Ordering::SeqCst);
    let (backend, token) = compile_eax_live_across_cond_call(func);
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(EAX_LIVE_HITS.load(Ordering::SeqCst), 1);
    assert_eq!(backend.get_int_value(&frame, 0), 0x1111_2222_3333_4444);
}

fn loop_code(token: &JitCellToken, code_size: usize) -> &[u8] {
    unsafe { std::slice::from_raw_parts(token.ll_raw_start() as *const u8, code_size) }
}

/// `generate_cond_call_slowpath` loads the helper with `_load_scratch` and
/// `call r11`. Its `push_gcmap`, before the func immediate, has loaded the
/// gcmap into r11 the same way, so the helper is either a `mov r11, imm64`
/// or an `lea r11, [r11+d]` off that gcmap. Track r11 from the gcmap load.
fn cond_call_slowpath_index(code: &[u8], func: u64) -> usize {
    let needle = func.to_le_bytes();
    let at = code
        .windows(8)
        .position(|window| window == needle)
        .unwrap_or_else(|| panic!("cond_call func immediate is missing"));
    let gcmap_load = code[..at]
        .windows(2)
        .rposition(|window| window == [0x49, 0xBB])
        .unwrap_or_else(|| panic!("cond_call gcmap load is missing"));
    let tail = &code[gcmap_load..];
    let mut r11: Option<u64> = None;
    let mut i = 0;
    while i + 3 <= tail.len() {
        if tail[i..].starts_with(&[0x49, 0xBB]) && i + 10 <= tail.len() {
            r11 = Some(u64::from_le_bytes(tail[i + 2..i + 10].try_into().unwrap()));
            i += 10;
            continue;
        }
        if let Some(value) = r11 {
            if tail[i..].starts_with(&[0x4D, 0x8D, 0x5B]) && i + 4 <= tail.len() {
                r11 = Some(value.wrapping_add(tail[i + 3] as i8 as i64 as u64));
                i += 4;
                continue;
            }
            if tail[i..].starts_with(&[0x4D, 0x8D, 0x9B]) && i + 7 <= tail.len() {
                let d = i32::from_le_bytes(tail[i + 3..i + 7].try_into().unwrap());
                r11 = Some(value.wrapping_add(d as i64 as u64));
                i += 7;
                continue;
            }
            if tail[i..].starts_with(&[0x41, 0xFF, 0xD3]) {
                return classify_cond_call_helper(value);
            }
        }
        i += 1;
    }
    panic!("cond_call_slowpath call is missing after the func immediate");
}

/// `_build_cond_call_slowpath` prologue: GPR `mov [rbp+slot], reg`, then
/// XMM stores when `supports_floats`, then `sub rsp, align`. `ebx` is in
/// `all_regs` and not in `save_around_call_regs`, so a store of it means
/// `callee_only` is false.
fn classify_cond_call_helper(addr: u64) -> usize {
    let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, 192) };
    let mut saves_ebx = false;
    let mut saves_xmm = false;
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i] == 0x48 && bytes[i + 1] == 0x83 && bytes[i + 2] == 0xEC {
            break;
        }
        if bytes[i] == 0xF2 {
            saves_xmm = true;
            let modrm_at = if bytes[i + 1] == 0x0F { i + 3 } else { i + 4 };
            i = modrm_at + 1 + disp_after_modrm(bytes[modrm_at]);
            continue;
        }
        if (bytes[i] == 0x48 || bytes[i] == 0x4C) && bytes[i + 1] == 0x89 {
            let modrm = bytes[i + 2];
            let reg = ((bytes[i] & 0x04) << 1) | ((modrm >> 3) & 7);
            if reg == 3 {
                saves_ebx = true;
            }
            i += 3 + disp_after_modrm(modrm);
            continue;
        }
        panic!(
            "cond_call helper prologue byte {:#x} at {i} (addr {addr:#x})",
            bytes[i]
        );
    }
    usize::from(saves_xmm) * 2 + usize::from(!saves_ebx)
}

fn disp_after_modrm(modrm: u8) -> usize {
    match modrm >> 6 {
        1 => 1,
        2 => 4,
        _ => 0,
    }
}

/// `INT_ADD`, `FLOAT_NEG`, and the input int stay live as fail-args.
/// `exc` is `GUARD_NO_EXCEPTION`; otherwise `INT_EQ` / `GUARD_TRUE` fails.
/// Failargs are not constants (`compute_vars_longevity`).
fn compile_guard_fail_with_float(exc: bool) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Float, 1),
    ];
    let i0 = inputargs[0].opref();
    let f0 = inputargs[1].opref();
    let add = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(1))]);
    add.pos().set(OpRef::int_op(1));
    let neg = Op::new(OpCode::FloatNeg, &[rb(f0)]);
    neg.pos().set(OpRef::float_op(2));
    let mut ops = vec![add, neg];
    let guard = if exc {
        let guard = Op::new(OpCode::GuardNoException, &[]);
        guard.pos().set(OpRef::void_op(3));
        guard
    } else {
        let eq = Op::new(OpCode::IntEq, &[rb(i0), rb(OpRef::const_int(1))]);
        eq.pos().set(OpRef::int_op(3));
        ops.push(eq);
        let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(3))]);
        guard.pos().set(OpRef::void_op(4));
        guard
    };
    guard.setdescr(make_resume_guard_descr_typed(vec![
        Type::Int,
        Type::Float,
        Type::Int,
    ]));
    guard.set_fail_arg_types(vec![Type::Int, Type::Float, Type::Int]);
    guard.setfailargs(vec![rb(OpRef::int_op(1)), rb(OpRef::float_op(2)), rb(i0)].into());
    ops.push(guard);
    ops.push(finish(5));
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile guard failure exc={exc}: {err:?}"));
    (backend, token)
}

fn assert_guard_failargs(backend: &DynasmBackend, frame: &majit_backend::DeadFrame) {
    assert!(!backend.get_latest_descr(frame).is_finish());
    assert_eq!(backend.get_int_value(frame, 0), 8);
    assert_float_bits(
        backend.get_float_value(frame, 1),
        2.5,
        "guard float failarg",
    );
    assert_eq!(backend.get_int_value(frame, 2), 7);
}

#[test]
fn failure_recovery_reads_failargs_without_an_exception() {
    // `GUARD_TRUE` fails. No pending exception, and a float fail-arg, so
    // `store_info_on_descr` selects `failure_recovery_code[2]`
    // (`exc=false`, `withfloats=true`).
    let _lock = lock_exception_cells();
    let _clear = ClearExc;
    majit_backend_dynasm::jit_exc_clear();
    let (backend, token) = compile_guard_fail_with_float(false);
    let frame = backend.execute_token(&token, &[Value::Int(7), Value::Float(-2.5)]);
    assert_guard_failargs(&backend, &frame);
    assert_eq!(backend.grab_exc_value(&frame), GcRef(0));
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
}

#[test]
fn failure_recovery_captures_a_pending_exception() {
    // `GUARD_NO_EXCEPTION` with a float fail-arg selects
    // `failure_recovery_code[3]` (`exc=true`, `withfloats=true`).
    // `_build_failure_recovery` stores `pos_exc_value` in `jf_guard_exc`
    // and clears both cells.
    let _lock = lock_exception_cells();
    let _clear = ClearExc;
    majit_backend_dynasm::jit_exc_clear();
    majit_backend_dynasm::jit_exc_raise(EXC_OBJ.as_ptr() as i64);
    let (backend, token) = compile_guard_fail_with_float(true);
    let frame = backend.execute_token(&token, &[Value::Int(7), Value::Float(-2.5)]);
    assert_guard_failargs(&backend, &frame);
    assert_eq!(
        backend.grab_exc_value(&frame),
        GcRef(EXC_OBJ.as_ptr() as usize)
    );
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
}

fn no_collect_effect() -> EffectInfo {
    let mut effect = EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None);
    effect.can_collect = false;
    effect
}

fn call_descr(arg_types: Vec<Type>, result: Type, signed: bool, size: usize) -> DescrRef {
    majit_ir::descr::make_call_descr_full(0, arg_types, result, signed, size, no_collect_effect())
}

fn finish_of(result: OpRef, ty: Type, pos: u32) -> Op {
    let op = Op::new(OpCode::Finish, &[rb(result)]);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![ty]);
    op.setfailargs(vec![rb(result)].into());
    op
}

fn compile_call(
    opcode: OpCode,
    func: i64,
    real_args: &[Operand],
    arg_types: Vec<Type>,
    result: OpRef,
    result_type: Type,
    signed: bool,
    size: usize,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let mut args = vec![rb(OpRef::const_int(func))];
    args.extend_from_slice(real_args);
    let call = Op::new(opcode, &args);
    call.pos().set(result);
    call.setdescr(call_descr(arg_types, result_type, signed, size));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(result, result_type, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    (backend, token)
}

extern "C" fn ret_i8() -> i8 {
    -42
}
extern "C" fn ret_u8() -> u8 {
    0xFE
}
extern "C" fn ret_i16() -> i16 {
    -300
}
extern "C" fn ret_u16() -> u16 {
    0xFFFE
}
extern "C" fn ret_i32() -> i32 {
    -2
}
extern "C" fn ret_u32() -> u32 {
    0xFFFF_FFFE
}

#[test]
fn call_narrow_results_are_sign_or_zero_extended() {
    // `CallBuilderX86.load_result` / `load_from_mem`: MOVSX/MOVZX/MOV32
    // on eax. A 32-bit return leaves the high half of rax clear, so a
    // missing extension is visible for every width below a word.
    let runs: Vec<(i64, bool, usize, i64)> = vec![
        (ret_i8 as usize as i64, true, 1, -42),
        (ret_u8 as usize as i64, false, 1, 0xFE),
        (ret_i16 as usize as i64, true, 2, -300),
        (ret_u16 as usize as i64, false, 2, 0xFFFE),
        (ret_i32 as usize as i64, true, 4, -2),
        (ret_u32 as usize as i64, false, 4, 0xFFFF_FFFE),
    ];
    for (func, signed, size, expected) in runs {
        let (backend, token) = compile_call(
            OpCode::CallI,
            func,
            &[],
            vec![],
            OpRef::int_op(1),
            Type::Int,
            signed,
            size,
        );
        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(
            backend.get_int_value(&frame, 0),
            expected,
            "size={size} signed={signed}"
        );
    }
}

static REF_BYTE: u8 = 0x5A;

extern "C" fn ref_static() -> *mut u8 {
    &REF_BYTE as *const u8 as *mut u8
}

#[test]
fn call_ref_result_is_the_pointer_word() {
    // A pointer-word ref result stays in eax.
    let (backend, token) = compile_call(
        OpCode::CallR,
        ref_static as usize as i64,
        &[],
        vec![],
        OpRef::ref_op(1),
        Type::Ref,
        false,
        8,
    );
    let frame = backend.execute_token(&token, &[]);
    assert_eq!(
        backend.get_ref_value(&frame, 0),
        GcRef(&REF_BYTE as *const u8 as usize)
    );
}

extern "C" fn neg_float(x: f64) -> f64 {
    -x
}

#[test]
fn call_float_result_is_read_from_xmm0() {
    // `load_result` does not spill xmm0. FINISH reads the register
    // `after_call` bound.
    let (backend, token) = compile_call(
        OpCode::CallF,
        neg_float as usize as i64,
        &[rb(OpRef::const_float(1.5))],
        vec![Type::Float],
        OpRef::float_op(1),
        Type::Float,
        false,
        8,
    );
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_float_bits(backend.get_float_value(&frame, 0), -1.5, "call float");
}

extern "C" fn singlefloat_then_double(a: f32, b: f64) -> f64 {
    f64::from(a) * 1000.0 + b
}

#[test]
fn call_singlefloat_movd_does_not_clobber_a_double_source() {
    // `CallBuilder64.prepare_arguments` remaps the xmm (double) arguments
    // before the singlefloat MOVD32s. Signature `('S', 'f')`: the double is
    // the previous call's result, so it sits in xmm0 and goes to xmm1,
    // while the singlefloat goes to xmm0. Installing the singlefloat first
    // would overwrite the double before the remap reads it.
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let neg = Op::new(
        OpCode::CallF,
        &[
            rb(OpRef::const_int(neg_float as usize as i64)),
            rb(inputargs[0].opref()),
        ],
    );
    neg.pos().set(OpRef::float_op(1));
    neg.setdescr(call_descr(vec![Type::Float], Type::Float, false, 8));
    let call = Op::new(
        OpCode::CallF,
        &[
            rb(OpRef::const_int(singlefloat_then_double as usize as i64)),
            rb(OpRef::const_int(i64::from(1.5f32.to_bits()))),
            rb(OpRef::float_op(1)),
        ],
    );
    call.pos().set(OpRef::float_op(2));
    call.setdescr(majit_ir::descr::make_call_descr_full_with_classes(
        0,
        "Sf".to_string(),
        vec![Type::Int, Type::Float],
        Type::Float,
        'f',
        false,
        8,
        no_collect_effect(),
    ));
    let ops = vec![
        OpRc::new(neg),
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::float_op(2), Type::Float, 3)),
    ];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile ('S', 'f') call: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Float(2.0)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_float_bits(
        backend.get_float_value(&frame, 0),
        1500.0 - 2.0,
        "('S', 'f') call",
    );
}

#[allow(clippy::too_many_arguments)]
extern "C" fn mixed_stack_args(
    a0: f64,
    b1: i64,
    a2: f64,
    b3: i64,
    a4: f64,
    b5: i64,
    a6: f64,
    b7: i64,
    a8: f64,
    b9: i64,
    a10: f64,
    b11: i64,
    a12: f64,
    b13: i64,
    a14: f64,
    b15: i64,
    a16: f64,
    a17: f64,
) -> i64 {
    let floats = [(0, a0), (2, a2), (4, a4), (6, a6), (8, a8)]
        .into_iter()
        .chain([(10, a10), (12, a12), (14, a14), (16, a16), (17, a17)]);
    let ints = [(1, b1), (3, b3), (5, b5), (7, b7), (9, b9)]
        .into_iter()
        .chain([(11, b11), (13, b13), (15, b15)]);
    floats
        .map(|(k, v)| (k, (v * 4.0) as i64))
        .chain(ints)
        .fold(0i64, |acc, (k, v)| {
            acc.wrapping_add((k + 1i64).wrapping_mul(v))
        })
}

#[test]
fn call_stack_arguments_from_frame_slots_and_constants() {
    // 10 doubles and 8 ints: past the argument registers on both ABIs, so
    // `prepare_arguments` hands out `RawEspLoc`s for frame-slot inputargs,
    // `ConstFloatLoc`s and a 64-bit immediate alike. `jump.py _move` stages
    // each memory source through the class's scratch register, except a
    // `ConstFloatLoc`, which `regalloc_immedmem2mem` stores as two `MOV32_si`.
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let mut inputargs: Vec<InputArgRc> = (0..6)
        .map(|i| InputArg::from_type_rc(Type::Float, i))
        .collect();
    inputargs.extend((6..10).map(|i| InputArg::from_type_rc(Type::Int, i)));
    let arg = |i: usize| rb(inputargs[i].opref());
    let big = 0x1_0000_0003i64;
    let args = [
        rb(OpRef::const_int(mixed_stack_args as usize as i64)),
        arg(0),
        arg(6),
        rb(OpRef::const_float(1.25)),
        rb(OpRef::const_int(7)),
        arg(1),
        arg(7),
        arg(2),
        rb(OpRef::const_int(big)),
        rb(OpRef::const_float(2.5)),
        arg(8),
        arg(3),
        arg(6),
        arg(0),
        rb(OpRef::const_int(-5)),
        arg(4),
        arg(9),
        rb(OpRef::const_float(-0.75)),
        arg(5),
    ];
    let types: Vec<Type> = args[1..]
        .iter()
        .enumerate()
        .map(|(k, _)| match k {
            1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 => Type::Int,
            _ => Type::Float,
        })
        .collect();
    let call = Op::new(OpCode::CallI, &args);
    call.pos().set(OpRef::int_op(10));
    call.setdescr(call_descr(types, Type::Int, true, 8));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::int_op(10), Type::Int, 11)),
    ];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile 18-argument call: {err:?}"));
    let (f, i) = (
        [0.5, 1.75, -2.25, 3.0, 4.5, -1.5],
        [11, -13, 1i64 << 40, 17],
    );
    let values: Vec<Value> = f
        .iter()
        .map(|&v| Value::Float(v))
        .chain(i.iter().map(|&v| Value::Int(v)))
        .collect();
    let frame = backend.execute_token(&token, &values);
    assert!(backend.get_latest_descr(&frame).is_finish());
    let expected = mixed_stack_args(
        f[0], i[0], 1.25, 7, f[1], i[1], f[2], big, 2.5, i[2], f[3], i[0], f[0], -5, f[4], i[3],
        -0.75, f[5],
    );
    assert_eq!(backend.get_int_value(&frame, 0), expected);
}

fn finish_const_float_bits(value: f64, token_id: i64) -> u64 {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(token_id);
    let finish = finish_of(OpRef::const_float(value), Type::Float, 0);
    let ops = vec![OpRc::new(finish)];
    let inputargs: Vec<InputArgRc> = vec![];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile FINISH(ConstFloat({value})): {err:?}"));
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_float_value(&frame, 0).to_bits()
}

#[test]
fn finish_of_a_const_float_stores_its_bits() {
    // `consider_finish`'s `make_sure_var_in_reg` is a `FloatImmedLoc`
    // for a ConstFloat; `genop_finish`'s `save_into_mem` stores it with
    // `MOV`: `MOV_bi` when the bits fit imm32, else `_load_scratch` and
    // `MOV_br`.
    assert_eq!(finish_const_float_bits(1.5, 0x5f10), 1.5f64.to_bits());
    let small = f64::from_bits(0x1234_5678);
    assert_eq!(finish_const_float_bits(small, 0x5f11), 0x1234_5678);
    assert_eq!(finish_const_float_bits(-0.0, 0x5f12), (-0.0f64).to_bits());
}

extern "C" fn threadlocalref_get_helper_must_not_run(_offset: i64) -> i64 {
    -1
}

/// `_consider_threadlocalref_get` / `threadlocalref_get`: `MOV_rs` of
/// `THREADLOCAL_OFS`, then `load_from_mem` at the offset. The helper address
/// in `getarg(0)` is never called. Returns the finished frame.
fn compile_threadlocalref_get(
    backend: &mut DynasmBackend,
    opcode: OpCode,
    result: OpRef,
    result_type: Type,
) -> majit_backend::DeadFrame {
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        opcode,
        &[
            rb(OpRef::const_int(
                threadlocalref_get_helper_must_not_run as usize as i64,
            )),
            rb(OpRef::const_int(8)),
        ],
    );
    call.pos().set(result);
    let effect = EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::ThreadlocalrefGet);
    call.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        vec![Type::Int],
        result_type,
        result_type == Type::Int,
        8,
        effect,
    ));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(result, result_type, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile threadlocalref_get {opcode:?}: {err:?}"));
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    frame
}

#[test]
fn threadlocalref_get_loads_inline_from_threadlocal_ofs() {
    majit_backend_dynasm::jit_threadlocalref_set(8, 0x1234_5678_9abc);
    let mut backend = fresh_backend();
    let frame =
        compile_threadlocalref_get(&mut backend, OpCode::CallI, OpRef::int_op(1), Type::Int);
    assert_eq!(backend.get_int_value(&frame, 0), 0x1234_5678_9abc);
}

#[test]
fn threadlocalref_get_with_a_ref_result_loads_inline() {
    // `_consider_threadlocalref_get` takes either result kind; a CALL_R
    // reaches the same `threadlocalref_get`, not the ordinary call emitter.
    majit_backend_dynasm::jit_threadlocalref_set(8, 0x1234_5678_9abc);
    let mut backend = fresh_backend();
    let frame =
        compile_threadlocalref_get(&mut backend, OpCode::CallR, OpRef::ref_op(1), Type::Ref);
    assert_eq!(
        backend.get_ref_value(&frame, 0).as_usize(),
        0x1234_5678_9abc
    );
}

#[derive(Debug)]
struct AssemblerLoopDescr {
    token: Arc<JitCellToken>,
}

impl Descr for AssemblerLoopDescr {
    fn as_loop_token_descr(&self) -> Option<&dyn LoopTokenDescr> {
        Some(self)
    }
}

impl LoopTokenDescr for AssemblerLoopDescr {
    fn loop_token_number(&self) -> i64 {
        self.token.number()
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.token)
    }
}

fn compile_identity_loop(backend: &mut DynasmBackend, ty: Type) -> Arc<JitCellToken> {
    let token = Arc::new(JitCellToken::new(next_token_id()));
    let inputargs = vec![InputArg::from_type_rc(ty, 0)];
    let i0 = inputargs[0].opref();
    let ops = vec![OpRc::new(finish_of(i0, ty, 1))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile identity {ty:?}: {err:?}"));
    assert_ne!(token.ll_function_addr(), 0, "callee entry was not baked");
    token
}

/// `handle_call_assembler` reads `jitframe_info` and allocates the callee
/// frame itself. The layout is process-wide and set-once; the tid has to
/// be the one this thread's collector registered.
fn backend_with_call_assembler_layout() -> DynasmBackend {
    use majit_backend_dynasm::jitframe::{
        FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FORWARD_OFS, JF_FRAME_INFO_OFS,
        JF_FRAME_OFS, JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS, JITFRAME_FIXED_SIZE, LENGTHOFS, SIGN_SIZE,
    };
    let mut gc = majit_gc::collector::MiniMarkGC::new();
    let jitframe_tid = gc.register_type(majit_backend_dynasm::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    backend.set_gc_allocator(Box::new(gc));
    majit_backend_dynasm::register_jitframe_layout(majit_backend_dynasm::JitFrameLayoutInfo {
        jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
            jitframe_tid,
            jitframe_fixed_size: JITFRAME_FIXED_SIZE,
            jf_frame_info_ofs: JF_FRAME_INFO_OFS,
            jf_descr_ofs: JF_DESCR_OFS,
            jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
            jf_savedata_ofs: JF_SAVEDATA_OFS,
            jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
            jf_forward_ofs: JF_FORWARD_OFS,
            jf_frame_ofs: JF_FRAME_OFS,
            jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
            jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
            sign_size: SIGN_SIZE,
            jf_frame_itemsize: SIGN_SIZE,
        }),
    });
    backend
}

#[test]
fn call_assembler_returns_int_and_float_from_the_dead_frame() {
    // `call_assembler` / `_call_assembler_load_result`: the fast path
    // loads value index 0. Same backend so the done-descr compare hits.
    // `handle_call_assembler` turns the value argument into the callee
    // frame; the callee is registered on this thread first.
    let mut backend = backend_with_call_assembler_layout();
    let int_callee = compile_identity_loop(&mut backend, Type::Int);
    let float_callee = compile_identity_loop(&mut backend, Type::Float);

    let int_inputs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = int_inputs[0].opref();
    let int_token = JitCellToken::new(next_token_id());
    let int_call = Op::new(OpCode::CallAssemblerI, &[rb(i0)]);
    int_call.pos().set(OpRef::int_op(1));
    int_call.setdescr(Arc::new(AssemblerLoopDescr {
        token: Arc::clone(&int_callee),
    }) as DescrRef);
    let int_ops = vec![
        OpRc::new(int_call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&int_inputs, &int_ops, &int_token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER_I: {err:?}"));
    let int_frame = backend.execute_token(&int_token, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&int_frame).is_finish());
    assert_eq!(backend.get_int_value(&int_frame, 0), 42);

    let float_inputs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let f0 = float_inputs[0].opref();
    let float_token = JitCellToken::new(next_token_id());
    let float_call = Op::new(OpCode::CallAssemblerF, &[rb(f0)]);
    float_call.pos().set(OpRef::float_op(1));
    float_call.setdescr(Arc::new(AssemblerLoopDescr {
        token: Arc::clone(&float_callee),
    }) as DescrRef);
    let float_ops = vec![
        OpRc::new(float_call),
        OpRc::new(finish_of(OpRef::float_op(1), Type::Float, 2)),
    ];
    backend
        .compile_loop(&float_inputs, &float_ops, &float_token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER_F: {err:?}"));
    let float_frame = backend.execute_token(&float_token, &[Value::Float(-2.5)]);
    assert!(backend.get_latest_descr(&float_frame).is_finish());
    assert_float_bits(
        backend.get_float_value(&float_frame, 0),
        -2.5,
        "call_assembler float",
    );
    let _keep = (int_callee, float_callee);
}

extern "C" fn call_assembler_return_const(_force_token: i64) -> i64 {
    7
}

extern "C" fn force_caller_and_return(force_token: i64) -> i64 {
    let ptr = FORCE_BACKEND.with(Cell::get);
    if ptr.is_null() {
        FORCE_SEEN.with(|cell| cell.set(Some(i64::MIN)));
        return 7;
    }
    let backend = unsafe { &*ptr };
    let recorded = catch_unwind(AssertUnwindSafe(|| {
        match backend.force(GcRef(force_token as usize)) {
            Some(frame) => backend.get_int_value(&frame, 0),
            None => -2,
        }
    }));
    let value = match recorded {
        Ok(value) => value,
        Err(_) => -3,
    };
    FORCE_SEEN.with(|cell| cell.set(Some(value)));
    7
}

fn compile_call_assembler_force_callee(
    backend: &mut DynasmBackend,
    helper: extern "C" fn(i64) -> i64,
) -> Arc<JitCellToken> {
    let token = Arc::new(JitCellToken::new(next_token_id()));
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let token_arg = inputargs[0].opref();
    let call = Op::new(
        OpCode::CallI,
        &[rb(OpRef::const_int(helper as usize as i64)), rb(token_arg)],
    );
    call.pos().set(OpRef::int_op(1));
    call.setdescr(call_descr(vec![Type::Ref], Type::Int, true, 8));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER callee: {err:?}"));
    assert_ne!(token.ll_function_addr(), 0, "callee entry was not baked");
    token
}

fn compile_call_assembler_force_caller(
    backend: &mut DynasmBackend,
    callee: &Arc<JitCellToken>,
) -> JitCellToken {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let token = JitCellToken::new(next_token_id());
    let force_token = Op::new(OpCode::ForceToken, &[]);
    force_token.pos().set(OpRef::ref_op(1));
    let call = Op::new(OpCode::CallAssemblerI, &[rb(OpRef::ref_op(1))]);
    call.pos().set(OpRef::int_op(2));
    call.setdescr(Arc::new(AssemblerLoopDescr {
        token: Arc::clone(callee),
    }) as DescrRef);
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.pos().set(OpRef::void_op(3));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int, Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int, Type::Int]);
    guard.setfailargs(vec![rb(i0), rb(OpRef::int_op(2))].into());
    let ops: Vec<OpRc> = vec![
        force_token,
        call,
        guard,
        finish_of(OpRef::int_op(2), Type::Int, 4),
    ]
    .into_iter()
    .map(OpRc::new)
    .collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER caller: {err:?}"));
    token
}

#[test]
fn call_assembler_async_force_reads_spilled_failarg() {
    // `call_assembler` leaves the result register's jitframe home alone.
    // `AbstractLLCPU.force` only copies `jf_force_descr` into `jf_descr`.
    // The live fail arg spilled by `before_call(SAVE_ALL_REGS)` is what
    // `force` can read while the callee is still inside the call. Once the
    // callee returns, `_call_assembler_load_result` has the result and
    // `GUARD_NOT_FORCED` fails into the recovery stub, which saves it.
    let _lock = lock_exception_cells();
    let mut backend = backend_with_call_assembler_layout();
    let quiet = compile_call_assembler_force_callee(&mut backend, call_assembler_return_const);
    let forcing = compile_call_assembler_force_callee(&mut backend, force_caller_and_return);
    let quiet_caller = compile_call_assembler_force_caller(&mut backend, &quiet);
    let forcing_caller = compile_call_assembler_force_caller(&mut backend, &forcing);

    let quiet_frame = backend.execute_token(&quiet_caller, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&quiet_frame).is_finish());
    assert_eq!(backend.get_int_value(&quiet_frame, 0), 7);

    FORCE_SEEN.with(|cell| cell.set(None));
    FORCE_BACKEND.with(|cell| cell.set(&backend as *const DynasmBackend));
    let _clear = ClearForceBackend;
    let forced = backend.execute_token(&forcing_caller, &[Value::Int(42)]);
    let during = FORCE_SEEN.with(Cell::get);
    assert_eq!(
        during,
        Some(42),
        "force() must observe the spilled fail arg"
    );
    assert!(
        !backend.get_latest_descr(&forced).is_finish(),
        "GUARD_NOT_FORCED must fail after force()"
    );
    assert_eq!(backend.get_int_value(&forced, 0), 42);
    assert_eq!(backend.get_int_value(&forced, 1), 7);
    let _keep = (quiet, forcing);
}

fn errno_ptr() -> *mut i32 {
    #[cfg(target_os = "windows")]
    unsafe {
        unsafe extern "C" {
            fn _errno() -> *mut i32;
        }
        _errno()
    }
    #[cfg(not(target_os = "windows"))]
    unsafe {
        unsafe extern "C" {
            fn __errno_location() -> *mut i32;
        }
        __errno_location()
    }
}

extern "C" fn swap_errno(new_value: i64) -> i64 {
    let slot = errno_ptr();
    let old = unsafe { *slot } as i64;
    unsafe {
        *slot = new_value as i32;
    }
    old
}

fn errno_save_flags(base: i64) -> i64 {
    // `RFFI_READSAVED_LASTERROR`: Win64 spills the used argument
    // registers around `SetLastError`. Errno itself is unchanged.
    #[cfg(target_os = "windows")]
    {
        base | 16
    }
    #[cfg(not(target_os = "windows"))]
    {
        base
    }
}

fn run_errno_swap(flags: i64, new_value: i64) -> i64 {
    let (backend, token) = compile_release_gil(flags, new_value);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "save_err={flags:#x}"
    );
    backend.get_int_value(&frame, 0)
}

fn compile_release_gil(flags: i64, new_value: i64) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CallReleaseGilI,
        &[
            rb(OpRef::const_int(flags)),
            rb(OpRef::const_int(swap_errno as usize as i64)),
            rb(OpRef::const_int(new_value)),
        ],
    );
    call.pos().set(OpRef::int_op(1));
    call.setdescr(call_descr(vec![Type::Int], Type::Int, true, 8));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_RELEASE_GIL: {err:?}"));
    (backend, token)
}

#[test]
fn call_release_gil_round_trips_errno() {
    // `write_real_errno` / `read_real_errno`: zero, restore the saved
    // copy, then read that copy back. The thread-local container
    // survives across the three calls.
    unsafe {
        *errno_ptr() = 11;
    }
    assert_eq!(run_errno_swap(errno_save_flags(5), 77), 0);
    unsafe {
        *errno_ptr() = 5;
    }
    assert_eq!(run_errno_swap(errno_save_flags(3), 9), 77);
    unsafe {
        *errno_ptr() = 3;
    }
    assert_eq!(run_errno_swap(errno_save_flags(2), 0), 9);
}

#[cfg(target_os = "windows")]
extern "C" fn set_last_error_and_return(code: i64) -> i64 {
    unsafe { majit_rlib::rwin32::_SetLastError(code as u32) };
    code + 1000
}

#[cfg(target_os = "windows")]
#[test]
fn call_release_gil_saves_last_error() {
    // `read_real_errno` with `RFFI_SAVE_LASTERROR`: after
    // `restore_stack_pointer`, `GetLastError` runs below a fresh shadow
    // area and rax survives around it.
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CallReleaseGilI,
        &[
            rb(OpRef::const_int(majit_jitcode::rffi::RFFI_SAVE_LASTERROR)),
            rb(OpRef::const_int(set_last_error_and_return as usize as i64)),
            rb(OpRef::const_int(123)),
        ],
    );
    call.pos().set(OpRef::int_op(1));
    call.setdescr(call_descr(vec![Type::Int], Type::Int, true, 8));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_RELEASE_GIL: {err:?}"));
    majit_rlib::rwin32::SetLastError_saved(0);
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 1123);
    assert_eq!(majit_rlib::rwin32::GetLastError_saved(), 123);
}

#[cfg(target_os = "windows")]
extern "C" fn set_last_error_and_return_float(code: i64) -> f64 {
    unsafe { majit_rlib::rwin32::_SetLastError(code as u32) };
    code as f64 + 0.25
}

#[cfg(target_os = "windows")]
#[test]
fn call_release_gil_saves_last_error_float_result() {
    // `CallBuilder64.save_result_value` for a FLOAT result is one
    // `MOVSD [rsp], xmm0` (and `restore_result_value` the reverse) around
    // `GetLastError`; rax is not saved.
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CallReleaseGilF,
        &[
            rb(OpRef::const_int(majit_jitcode::rffi::RFFI_SAVE_LASTERROR)),
            rb(OpRef::const_int(
                set_last_error_and_return_float as usize as i64,
            )),
            rb(OpRef::const_int(77)),
        ],
    );
    call.pos().set(OpRef::float_op(1));
    call.setdescr(call_descr(vec![Type::Int], Type::Float, false, 8));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::float_op(1), Type::Float, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_RELEASE_GIL_F: {err:?}"));
    let code = compiled_bytes(&token);
    assert_has(code, &[0xF2, 0x0F, 0x11, 0x04, 0x24], "movsd [rsp], xmm0");
    assert_has(code, &[0xF2, 0x0F, 0x10, 0x04, 0x24], "movsd xmm0, [rsp]");
    assert!(
        !code.windows(4).any(|w| w == [0x48, 0x89, 0x04, 0x24]),
        "a float result must not save rax"
    );
    majit_rlib::rwin32::SetLastError_saved(0);
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_float_value(&frame, 0), 77.25);
    assert_eq!(majit_rlib::rwin32::GetLastError_saved(), 77);
}

fn op_at(opcode: OpCode, args: &[OpRef], pos: OpRef) -> Op {
    let operands: Vec<_> = args.iter().copied().map(rb).collect();
    let op = Op::new(opcode, &operands);
    op.pos().set(pos);
    op
}

fn wb_backend(mut gc: MiniMarkGC) -> DynasmBackend {
    let mut backend = DynasmBackend::new();
    // `setup_once` inside `attach_default_test_descrs` runs before MiniMark
    // is installed, so the helpers are built again at `compile_loop`.
    backend.set_gc_allocator(Box::new(majit_backend::jitframe::HostHeapGc));
    backend.attach_default_test_descrs();
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    backend.set_gc_allocator(Box::new(gc));
    backend
}

/// `TEST8 [base+byteofs], mask` immediately followed by a not-taken near
/// `JNZ` (`0F 85`). `byteofs` is `-4` (`WriteBarrierDescr::extract_flag_byte`).
fn wb_fastpath_site(code: &[u8], mask: u8) -> (usize, Vec<u8>) {
    let mut hits = Vec::new();
    let mut taken_jz = 0usize;
    for i in 0..code.len() {
        if code[i] != 0xF6 {
            continue;
        }
        for len in 3..=8 {
            let end = i + len;
            if end + 6 > code.len() {
                continue;
            }
            if code[end - 1] == mask && code[end - 2] == 0xFC && code[end] == 0x0F {
                match code[end + 1] {
                    0x85 => hits.push((i, code[i..end + 6].to_vec())),
                    0x84 => taken_jz += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(
        taken_jz, 0,
        "write barrier fast path is still a taken JZ over an inlined body"
    );
    assert_eq!(
        hits.len(),
        1,
        "write-barrier fast paths in the trace: {hits:x?}"
    );
    hits.pop().unwrap()
}

/// `call r11` (`41 FF D3`) is the helper in `WriteBarrierSlowPath.generate_body`.
/// It must sit out of line, after the fast-path window.
fn assert_helper_call_out_of_line(code: &[u8], site_at: usize, site: &[u8]) {
    let call = [0x41, 0xFF, 0xD3];
    assert!(
        !site.windows(3).any(|window| window == call),
        "call r11 must not sit inside the fast path"
    );
    let site_end = site_at + site.len();
    let call_at = code[site_end..]
        .windows(3)
        .position(|window| window == call)
        .map(|off| site_end + off)
        .unwrap_or_else(|| {
            let offs: Vec<usize> = code
                .windows(2)
                .enumerate()
                .filter(|(_, window)| *window == [0xFF, 0xD3])
                .map(|(index, _)| index)
                .collect();
            panic!("call r11 ({call:02X?}) must follow the fast path; FF D3 at {offs:?}")
        });
    assert!(
        call_at - site_end > 16,
        "helper call at {call_at} is only {} bytes after the fast path at {site_at}",
        call_at - site_end
    );
}

fn compiled_bytes(token: &JitCellToken) -> &[u8] {
    let compiled = token
        .compiled
        .get()
        .expect("compile_loop stores CompiledCode")
        .downcast_ref::<majit_backend_dynasm::x86::assembler::CompiledCode>()
        .expect("x86 CompiledCode");
    &compiled.buffer
}

fn run_wb_trace(
    backend: &mut DynasmBackend,
    ops: Vec<Op>,
    inputargs: Vec<InputArgRc>,
    values: &[Value],
    mask: u8,
) -> majit_backend::DeadFrame {
    let token = JitCellToken::new(next_token_id());
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile write barrier: {err:?}"));
    let bytes = compiled_bytes(&token);
    let (at, site) = wb_fastpath_site(bytes, mask);
    assert_helper_call_out_of_line(bytes, at, &site);
    let frame = backend.execute_token(&token, values);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "the barrier trace must finish"
    );
    frame
}

/// Flag byte clear: `TEST8`/`Jcc` falls through, `remember_young_pointer`
/// does not run, and the store completes.
#[test]
fn write_barrier_flag_clear_falls_through() {
    let mut gc = MiniMarkGC::new();
    let tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let old = majit_gc::GcAllocator::alloc_oldgen_typed(&mut gc, tid, 16);
    let young = gc.alloc_with_type(tid, 16);
    assert!(
        !gc.is_in_nursery(old.0),
        "alloc_oldgen_typed must birth an old object"
    );
    assert!(
        gc.is_in_nursery(young.0),
        "a small alloc_with_type is nursery"
    );
    unsafe {
        let hdr = &mut *header_of(old.0);
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        hdr.clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        hdr.set_flag(GcFlags::GCFLAG_NO_HEAP_PTRS);
    }
    let mut backend = wb_backend(gc);

    let base = OpRef::input_arg_ref(0);
    let value = OpRef::input_arg_ref(1);
    let loaded = OpRef::ref_op(3);
    let ops = vec![
        op_at(
            OpCode::GcStore,
            &[base, OpRef::const_int(0), value, OpRef::const_int(8)],
            OpRef::void_op(1),
        ),
        op_at(OpCode::CondCallGcWb, &[base], OpRef::void_op(2)),
        op_at(
            OpCode::GcLoadR,
            &[base, OpRef::const_int(0), OpRef::const_int(8)],
            loaded,
        ),
        finish_of(loaded, Type::Ref, 4),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let mask = majit_gc::WriteBarrierDescr::for_current_gc().jit_wb_if_flag_singlebyte;
    let frame = run_wb_trace(
        &mut backend,
        ops,
        inputargs,
        &[Value::Ref(old), Value::Ref(young)],
        mask,
    );
    assert_eq!(backend.get_ref_value(&frame, 0), young);
    assert_eq!(unsafe { *(old.0 as *const usize) }, young.0);
    let hdr = unsafe { &*header_of(old.0) };
    assert!(
        !hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
        "the fast path must not set TRACK_YOUNG_PTRS"
    );
    assert!(
        hdr.has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS),
        "remember_young_pointer clears NO_HEAP_PTRS; the fast path must not call it"
    );
}

/// Flag set, cards not set: the slow path calls the helper once for this object.
#[test]
fn write_barrier_flag_set_calls_helper() {
    let mut gc = MiniMarkGC::new();
    let tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let old = majit_gc::GcAllocator::alloc_oldgen_typed(&mut gc, tid, 16);
    let young = gc.alloc_with_type(tid, 16);
    assert!(gc.is_in_nursery(young.0));
    unsafe {
        let hdr = &mut *header_of(old.0);
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(!hdr.has_flag(GcFlags::GCFLAG_CARDS_SET));
        hdr.set_flag(GcFlags::GCFLAG_NO_HEAP_PTRS);
    }
    let mut backend = wb_backend(gc);

    let base = OpRef::input_arg_ref(0);
    let value = OpRef::input_arg_ref(1);
    let loaded = OpRef::ref_op(3);
    let ops = vec![
        op_at(
            OpCode::GcStore,
            &[base, OpRef::const_int(0), value, OpRef::const_int(8)],
            OpRef::void_op(1),
        ),
        op_at(OpCode::CondCallGcWb, &[base], OpRef::void_op(2)),
        op_at(
            OpCode::GcLoadR,
            &[base, OpRef::const_int(0), OpRef::const_int(8)],
            loaded,
        ),
        finish_of(loaded, Type::Ref, 4),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let mask = majit_gc::WriteBarrierDescr::for_current_gc().jit_wb_if_flag_singlebyte;
    let frame = run_wb_trace(
        &mut backend,
        ops,
        inputargs,
        &[Value::Ref(old), Value::Ref(young)],
        mask,
    );
    assert_eq!(backend.get_ref_value(&frame, 0), young);
    assert_eq!(unsafe { *(old.0 as *const usize) }, young.0);
    let hdr = unsafe { &*header_of(old.0) };
    assert!(
        !hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
        "remember_young_pointer must have recorded the object"
    );
    assert!(
        !hdr.has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS),
        "remember_young_pointer clears NO_HEAP_PTRS when it records the object"
    );
}

/// `GCFLAG_CARDS_SET` takes the card arm: the helper is not called, and
/// exactly one card bit is set.
#[test]
fn write_barrier_cards_set_marks_one_bit() {
    const LENGTH: usize = 17024;
    let mut gc = MiniMarkGC::new();
    let item_size = std::mem::size_of::<GcRef>();
    let array_tid = gc.register_type(majit_gc::TypeInfo::varsize(
        8,
        item_size,
        0,
        true,
        Vec::new(),
    ));
    let young_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let total_size = GcHeader::SIZE + 8 + item_size * LENGTH;
    let array = gc.alloc_in_oldgen_with_cards(array_tid, total_size, LENGTH, true);
    unsafe { *(array.0 as *mut usize) = LENGTH };
    let young = gc.alloc_with_type(young_tid, 16);
    assert!(gc.is_in_nursery(young.0));
    unsafe {
        let hdr = &mut *header_of(array.0);
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_HAS_CARDS));
        // `jit_remember_young_pointer_from_array` does not clear TRACK when
        // HAS_CARDS is set, so drop HAS_CARDS: a helper call then falls
        // through to `remember_young_pointer` and clears TRACK. The card
        // bytes stay in front of the header. `dirty_cards` ignores an object
        // without HAS_CARDS, so the bit is read directly.
        hdr.clear_flag(GcFlags::GCFLAG_HAS_CARDS);
        hdr.set_flag(GcFlags::GCFLAG_CARDS_SET);
    }
    let shift = majit_gc::WriteBarrierDescr::for_current_gc().jit_wb_card_page_shift;
    let card_bytes = (LENGTH + (8 << shift) - 1) >> (shift as usize + 3);
    for i in 0..card_bytes {
        unsafe {
            *((array.0 - GcHeader::SIZE - 1 - i) as *mut u8) = 0;
        }
    }
    let mut backend = wb_backend(gc);

    let index: i64 = 1152;
    let base = OpRef::input_arg_ref(0);
    let index_ref = OpRef::input_arg_int(1);
    let value = OpRef::input_arg_ref(2);
    let loaded = OpRef::ref_op(4);
    let scale = OpRef::const_int(item_size as i64);
    let item_ofs = OpRef::const_int(8);
    let size = OpRef::const_int(8);
    let ops = vec![
        op_at(
            OpCode::GcStoreIndexed,
            &[base, index_ref, value, scale, item_ofs, size],
            OpRef::void_op(1),
        ),
        op_at(
            OpCode::CondCallGcWbArray,
            &[base, index_ref],
            OpRef::void_op(2),
        ),
        op_at(
            OpCode::GcLoadIndexedR,
            &[base, index_ref, scale, item_ofs, size],
            loaded,
        ),
        finish_of(loaded, Type::Ref, 5),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Ref, 2),
    ];
    let wb = majit_gc::WriteBarrierDescr::for_current_gc();
    let frame = run_wb_trace(
        &mut backend,
        ops,
        inputargs,
        &[Value::Ref(array), Value::Int(index), Value::Ref(young)],
        wb.jit_wb_if_flag_singlebyte | 0x80,
    );
    assert_eq!(backend.get_ref_value(&frame, 0), young);
    let slot = array.0 + 8 + (index as usize) * item_size;
    assert_eq!(unsafe { *(slot as *const usize) }, young.0);
    let hdr = unsafe { &*header_of(array.0) };
    assert!(
        hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
        "the card arm must not call the helper"
    );
    assert!(!hdr.has_flag(GcFlags::GCFLAG_HAS_CARDS));
    let card = (index as usize) >> shift;
    let mut bits = 0usize;
    for i in 0..card_bytes {
        let byte = unsafe { *((array.0 - GcHeader::SIZE - 1 - i) as *const u8) };
        bits += byte.count_ones() as usize;
        if i == card >> 3 {
            assert_eq!(byte, 1 << (card & 7), "card {card} byte {i}");
        }
    }
    assert_eq!(bits, 1, "exactly one card bit");
}

/// `genop_discard_check_memory_error`: `TEST reg, reg` plus a not-taken
/// `jz` (`Conditions['Z']`) to the in-buffer trampoline. A null value
/// reaches `propagate_exception_path`; a non-null value falls through.
fn compile_check_memory_error() -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let check = Op::new(OpCode::CheckMemoryError, &[rb(i0)]);
    check.pos().set(OpRef::void_op(1));
    let ops = vec![OpRc::new(check), OpRc::new(finish_of(i0, Type::Int, 2))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CHECK_MEMORY_ERROR: {err:?}"));
    let code = compiled_bytes(&token);
    let (jz, jnz) = count_test_rr_jcc(code);
    assert_eq!(jz, 1, "fast path is one not-taken jz");
    assert_eq!(
        jnz, 0,
        "CHECK_MEMORY_ERROR must not jnz over an inlined body"
    );
    (backend, token)
}

fn count_test_rr_jcc(code: &[u8]) -> (usize, usize) {
    let mut jz = 0usize;
    let mut jnz = 0usize;
    let mut i = 0;
    while i + 3 < code.len() {
        let start = i;
        let (rex, op_at) = if (0x40..0x50).contains(&code[i]) {
            (code[i], i + 1)
        } else if i > 0 && (0x40..0x50).contains(&code[i - 1]) {
            i += 1;
            continue;
        } else {
            (0, i)
        };
        if op_at + 1 >= code.len() || code[op_at] != 0x85 {
            i = start + 1;
            continue;
        }
        let modrm = code[op_at + 1];
        let reg = (((rex >> 2) & 1) << 3) | ((modrm >> 3) & 7);
        let rm = ((rex & 1) << 3) | (modrm & 7);
        if modrm & 0xC0 != 0xC0 || reg != rm {
            i = start + 1;
            continue;
        }
        let jcc = op_at + 2;
        if jcc + 1 < code.len() && code[jcc] == 0x0F {
            match code[jcc + 1] {
                0x84 => jz += 1,
                0x85 => jnz += 1,
                _ => {}
            }
        }
        i = jcc;
    }
    (jz, jnz)
}

fn assert_propagate(backend: &DynasmBackend, frame: &majit_backend::DeadFrame) {
    assert!(
        !backend.get_latest_descr(frame).is_finish(),
        "propagate must not finish"
    );
    let descr = backend.get_latest_descr_arc(frame);
    assert!(
        descr
            .as_any()
            .is_some_and(|any| any.is::<majit_backend::PropagateExceptionDescr>()),
        "latest descr is PropagateExceptionDescr"
    );
}

fn shadow_top() -> usize {
    let addr = majit_gc::shadow_stack::get_root_stack_top_addr();
    unsafe { *(addr as *const usize) }
}

#[test]
fn check_memory_error_nonzero_falls_through() {
    let (backend, token) = compile_check_memory_error();
    let top = shadow_top();
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    assert_eq!(shadow_top(), top, "finish pushes and pops the shadow stack");
}

#[test]
fn check_memory_error_null_propagates() {
    let (backend, token) = compile_check_memory_error();
    // `generate_propagate_error_64` clears the process-wide exception cells.
    let _lock = lock_exception_cells();
    let top = shadow_top();
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert_propagate(&backend, &frame);
    assert!(
        !majit_backend_dynasm::jit_exc_is_pending(),
        "generate_propagate_error_64 clears both exception cells"
    );
    assert_eq!(
        shadow_top(),
        top,
        "the propagate footer pops the prologue push"
    );
}

const BRIDGE_LIVE_INTS: u32 = 48;

thread_local! {
    static SEEN_FRAME: Cell<usize> = const { Cell::new(0) };
    static STACK_CALLS: Cell<u32> = const { Cell::new(0) };
    static STACK_MODE: Cell<u8> = const { Cell::new(0) };
}

/// `insert_stack_check` cells. `register_stack_check_addresses` keeps the
/// raw addresses for the process, so they have to outlive every compile.
static STACK_END: AtomicUsize = AtomicUsize::new(0);
static STACK_LENGTH: AtomicUsize = AtomicUsize::new(usize::MAX);

struct StackProbeGuard;

impl Drop for StackProbeGuard {
    fn drop(&mut self) {
        STACK_LENGTH.store(usize::MAX, Ordering::Release);
        STACK_END.store(0, Ordering::Release);
        STACK_MODE.with(|mode| mode.set(0));
    }
}

extern "C" fn stack_check_probe(_current: usize) -> u8 {
    STACK_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    match STACK_MODE.with(|mode| mode.get()) {
        2 => {
            majit_backend_dynasm::jit_exc_raise(EXC_OBJ.as_ptr() as i64);
            1
        }
        1 => 1,
        _ => 0,
    }
}

extern "C" fn snapshot_shadow_frame() -> i64 {
    let top = shadow_top();
    let jf = if top >= 8 {
        unsafe { *((top - 8) as *const usize) }
    } else {
        0
    };
    SEEN_FRAME.with(|cell| cell.set(jf));
    0
}

fn live_int_adds(i0: OpRef, count: u32) -> (Vec<Op>, Vec<Operand>) {
    let mut ops = Vec::with_capacity(count as usize);
    let mut finished = vec![rb(i0)];
    for n in 0..count {
        let pos = 10 + n;
        let add = Op::new(
            OpCode::IntAdd,
            &[rb(i0), rb(OpRef::const_int(i64::from(n) + 1))],
        );
        add.pos().set(OpRef::int_op(pos));
        ops.push(add);
        finished.push(rb(OpRef::int_op(pos)));
    }
    (ops, finished)
}

fn finish_many(args: &[Operand], pos: u32) -> Op {
    let op = Op::new(OpCode::Finish, args);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![Type::Int; args.len()]);
    op.setfailargs(args.to_vec().into());
    op
}

fn tip_addr(frame: &majit_backend::DeadFrame) -> usize {
    if let Some(libc) = frame.as_libc_jitframe() {
        libc.frame_addr()
    } else if let Some(jf) = frame.as_jitframe() {
        jf.jf_gcref().0
    } else {
        panic!("deadframe has no jitframe");
    }
}

fn frame_info_words(token: &JitCellToken) -> (isize, isize) {
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let info = clt.frame_info.lock();
    (info.depth(), info.size())
}

fn store_frame_info(token: &JitCellToken, depth: isize, size: isize) {
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let info = clt.frame_info.lock();
    // Shrink depth before size so a reader cannot observe a deep depth
    // with a short allocation.
    info.jfi_frame_depth.store(depth, Ordering::Relaxed);
    info.jfi_frame_size.store(size, Ordering::Relaxed);
}

fn bridge_code(token: &JitCellToken) -> (Vec<u8>, usize) {
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let blocks = clt.asmmemmgr_blocks.lock();
    for block in blocks.iter().rev() {
        if let Some(code) =
            block.downcast_ref::<majit_backend_dynasm::x86::assembler::CompiledCode>()
        {
            if code.source_guard.is_some() {
                return (
                    code.buffer.to_vec(),
                    code.frame_depth.load(Ordering::Acquire),
                );
            }
        }
    }
    panic!("bridge CompiledCode missing");
}

/// `_check_frame_depth`: `CMP` immediate then not-taken `jl`. Both
/// `0xffffff` sites are patched to the bridge depth.
fn assert_frame_depth_sites(code: &[u8], depth: usize) {
    // `cmp qword [rbp+disp32], imm32` (`48 81 /7`) then `jl`.
    let cmp_at = unique_bytes(code, &[0x48, 0x81, 0xBD, 0x38, 0x00, 0x00, 0x00]);
    let mov_at = unique_bytes(code, &[0x48, 0xC7, 0x44, 0x24, 0x08]);
    let cmp_imm = u32::from_le_bytes(code[cmp_at + 7..cmp_at + 11].try_into().unwrap());
    let mov_imm = u32::from_le_bytes(code[mov_at + 5..mov_at + 9].try_into().unwrap());
    assert_eq!(
        cmp_imm, depth as u32,
        "cmp immediate {cmp_imm:#x} depth {depth}"
    );
    assert_eq!(mov_imm, cmp_imm, "mov_si immediate {mov_imm:#x}");
    assert_ne!(cmp_imm, 0x00ff_ffff, "placeholder left in the bridge");
    assert_eq!(
        &code[cmp_at + 11..cmp_at + 13],
        &[0x0F, 0x8C],
        "fast path is jl, not jge"
    );
}

fn unique_bytes(code: &[u8], needle: &[u8]) -> usize {
    let mut found = None;
    for (index, window) in code.windows(needle.len()).enumerate() {
        if window == needle {
            assert!(found.is_none(), "{needle:02x?} appears twice");
            found = Some(index);
        }
    }
    found.unwrap_or_else(|| panic!("{needle:02x?} missing"))
}

/// Loop finish always lists `i0` first, then `live_across` int adds, so
/// those adds stay live across the failing guard and widen the loop frame.
fn compile_guarded_loop(live_across: u32) -> (DynasmBackend, JitCellToken, Vec<InputArgRc>) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let call = Op::new(
        OpCode::CallI,
        &[rb(OpRef::const_int(
            snapshot_shadow_frame as *const () as usize as i64,
        ))],
    );
    call.pos().set(OpRef::int_op(1));
    call.setdescr(call_descr(vec![], Type::Int, true, 8));
    let eq = Op::new(OpCode::IntEq, &[rb(i0), rb(OpRef::const_int(1))]);
    eq.pos().set(OpRef::int_op(2));
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard.pos().set(OpRef::void_op(3));
    guard.set_fail_arg_types(vec![Type::Int]);
    guard.setfailargs(vec![rb(i0)].into());
    let label = Op::new(OpCode::Label, &[rb(i0)]);
    label.pos().set(OpRef::void_op(0));
    let (adds, finished) = live_int_adds(i0, live_across);
    let mut ops = vec![label, call];
    ops.extend(adds);
    ops.push(eq);
    ops.push(guard);
    ops.push(finish_many(&finished, 4));
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile guarded loop: {err:?}"));
    (backend, token, inputargs)
}

fn bridge_with_live_ints(i0: OpRef, count: u32) -> Vec<Op> {
    let (adds, finished) = live_int_adds(i0, count);
    let label = Op::new(OpCode::Label, &[rb(i0)]);
    label.pos().set(OpRef::void_op(0));
    let mut ops = Vec::with_capacity(adds.len() + 2);
    ops.push(label);
    ops.extend(adds);
    ops.push(finish_many(&finished, 100));
    ops
}

fn attach_bridge(
    backend: &mut DynasmBackend,
    token: &JitCellToken,
    inputargs: &[InputArgRc],
    bridge_ops: Vec<Op>,
) -> (Vec<u8>, usize) {
    SEEN_FRAME.with(|cell| cell.set(0));
    let failed = backend.execute_token(token, &[Value::Int(7)]);
    assert!(!backend.get_latest_descr(&failed).is_finish());
    assert_eq!(backend.get_int_value(&failed, 0), 7);
    let guard_descr = backend.get_latest_descr_arc(&failed);
    assert_ne!(guard_descr.as_fail_descr().unwrap().adr_jump_offset(), 0);
    drop(failed);
    SEEN_FRAME.with(|cell| cell.set(0));
    let bridge_ops: Vec<OpRc> = bridge_ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_bridge(
            guard_descr.as_fail_descr().unwrap(),
            inputargs,
            &bridge_ops,
            token,
            None,
        )
        .unwrap_or_else(|err| panic!("compile bridge: {err:?}"));
    bridge_code(token)
}

fn run_bridged(backend: &mut DynasmBackend, token: &JitCellToken) -> (usize, usize) {
    SEEN_FRAME.with(|cell| cell.set(0));
    let frame = backend.execute_token(token, &[Value::Int(7)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    let seen = SEEN_FRAME.with(|cell| cell.get());
    assert_ne!(seen, 0, "CallI must read the shadow-stack frame");
    let tip = tip_addr(&frame);
    (seen, tip)
}

/// `_call_footer` restores the callee-saves `_call_header` spilled.
/// Win64 reserves 32 bytes before the direct `call`: the prologue writes
/// `THREADLOCAL_OFS` and `SHADOWSTORE2_OFS` / `SHADOWSTORE3_OFS` into
/// that shadow.
fn call_entry_check_spills(token: &JitCellToken, depth: usize, input: i64) {
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let bytes = majit_backend::jitframe::JitFrame::alloc_size(depth);
    let head = majit_backend::llmodel::take_or_alloc_parked_entry_frame(token, bytes);
    let info = clt.frame_info.data_ptr();
    unsafe {
        majit_backend::jitframe::JitFrame::init(head, info, depth);
        majit_backend::llmodel::set_int_value(
            head,
            majit_backend_dynasm::arch::JITFRAME_FIXED_SIZE,
            input as isize,
        );
    }
    struct FreeHead<'a> {
        token: &'a JitCellToken,
        head: *mut majit_backend::jitframe::JitFrame,
    }
    impl Drop for FreeHead<'_> {
        fn drop(&mut self) {
            if self.head.is_null() {
                return;
            }
            majit_backend::llmodel::park_or_free_done_entry_frame(
                self.token,
                self.head,
                std::ptr::null_mut(),
                false,
            );
            self.head = std::ptr::null_mut();
        }
    }
    let _free = FreeHead { token, head };
    let entry = token.ll_function_addr();
    let head_addr = head as usize;
    // LLVM reserves rbx as an asm operand on this target. Save and restore
    // the prologue spills around the call and read them back from memory.
    // slots: saved rbx, saved rsi (Win64) or r12, saved r12, then the
    // values those registers hold after `_call_footer`.
    let mut slots = [0u64; 6];
    let slots_ptr = slots.as_mut_ptr() as usize;
    unsafe {
        #[cfg(target_os = "windows")]
        std::arch::asm!(
            "mov rax, rbx",
            "mov [r15], rax",
            "mov rax, rsi",
            "mov [r15 + 8], rax",
            "mov rax, r12",
            "mov [r15 + 16], rax",
            "mov rbx, 0x1111111111111111",
            "mov rsi, 0x2222222222222222",
            "mov r12, 0x3333333333333333",
            "mov rcx, r13",
            "xor edx, edx",
            "sub rsp, 32",
            "call r14",
            "add rsp, 32",
            "mov rax, rbx",
            "mov [r15 + 24], rax",
            "mov rax, rsi",
            "mov [r15 + 32], rax",
            "mov rax, r12",
            "mov [r15 + 40], rax",
            "mov rax, [r15]",
            "mov rbx, rax",
            "mov rax, [r15 + 8]",
            "mov rsi, rax",
            "mov rax, [r15 + 16]",
            "mov r12, rax",
            in("r13") head_addr,
            in("r14") entry,
            in("r15") slots_ptr,
            clobber_abi("win64"),
        );
        #[cfg(not(target_os = "windows"))]
        std::arch::asm!(
            "mov rax, rbx",
            "mov [r15], rax",
            "mov rax, r12",
            "mov [r15 + 8], rax",
            "mov rbx, 0x1111111111111111",
            "mov r12, 0x3333333333333333",
            "mov rdi, r13",
            "xor esi, esi",
            "call r14",
            "mov rax, rbx",
            "mov [r15 + 24], rax",
            "mov rax, r12",
            "mov [r15 + 32], rax",
            "mov rax, [r15]",
            "mov rbx, rax",
            "mov rax, [r15 + 8]",
            "mov r12, rax",
            in("r13") head_addr,
            in("r14") entry,
            in("r15") slots_ptr,
            clobber_abi("sysv64"),
        );
    }
    assert_eq!(slots[3], 0x1111_1111_1111_1111, "rbx prologue spill");
    #[cfg(target_os = "windows")]
    {
        assert_eq!(slots[4], 0x2222_2222_2222_2222, "rsi");
        assert_eq!(slots[5], 0x3333_3333_3333_3333, "r12");
    }
    #[cfg(not(target_os = "windows"))]
    assert_eq!(slots[4], 0x3333_3333_3333_3333, "r12");
}

#[test]
fn bridge_frame_depth_reallocates_when_short() {
    let (mut backend, token, inputargs) = compile_guarded_loop(0);
    let (loop_depth, loop_size) = frame_info_words(&token);
    let i0 = inputargs[0].opref();
    let (code, bridge_frame_depth) = attach_bridge(
        &mut backend,
        &token,
        &inputargs,
        bridge_with_live_ints(i0, BRIDGE_LIVE_INTS),
    );
    assert_frame_depth_sites(&code, bridge_frame_depth);
    let (bridge_depth, _) = frame_info_words(&token);
    assert!(
        bridge_depth > loop_depth,
        "bridge frame {bridge_depth} must exceed the loop frame {loop_depth}"
    );
    // `execute_token` sizes from `jfi_frame_depth`. Put the loop's depth
    // back so the bridge's `jl` is taken and `realloc_frame` runs.
    // `build_frame_realloc_slowpath` clears both exception cells for that call.
    store_frame_info(&token, loop_depth, loop_size);
    let _lock = lock_exception_cells();
    let (seen, tip) = run_bridged(&mut backend, &token);
    assert_ne!(seen, tip, "realloc_frame forwards the jitframe");
    store_frame_info(&token, loop_depth, loop_size);
    call_entry_check_spills(&token, loop_depth as usize, 7);
}

#[test]
fn bridge_frame_depth_skips_when_deep() {
    let (mut backend, token, inputargs) = compile_guarded_loop(BRIDGE_LIVE_INTS);
    let (loop_depth, _) = frame_info_words(&token);
    let i0 = inputargs[0].opref();
    let (code, bridge_frame_depth) = attach_bridge(
        &mut backend,
        &token,
        &inputargs,
        bridge_with_live_ints(i0, 0),
    );
    assert_frame_depth_sites(&code, bridge_frame_depth);
    assert!(
        loop_depth >= bridge_frame_depth as isize,
        "loop frame {loop_depth} must cover the bridge frame {bridge_frame_depth}"
    );
    let (seen, tip) = run_bridged(&mut backend, &token);
    assert_eq!(
        seen, tip,
        "jl is not taken when the entry frame is already deep enough"
    );
    call_entry_check_spills(&token, loop_depth as usize, 7);
}

/// Enter a finish loop from inline asm and check every callee-saved GPR
/// `_call_header` / `_call_footer` own. Win64's set is ebx, esi, edi, r12,
/// r14, r15. r13 is not in that set (`arch.py` "never use r13") and the
/// trace does not write it, so the caller's value survives without a home.
/// The 32-byte shadow is inside the `sub rsp, 128` area.
fn call_finish_preserves_callee_saves(token: &JitCellToken, depth: usize, input: i64) {
    #[cfg(target_os = "windows")]
    {
        // Old r13 home: `(FRAME_FIXED_SIZE - 1) * WORD` == 0xA0.
        // `mov [rsp+0xA0], r13` / `mov r13, [rsp+0xA0]`.
        let code = compiled_bytes(token);
        assert!(
            !code.windows(8).any(|window| {
                window == [0x4C, 0x89, 0xAC, 0x24, 0xA0, 0x00, 0x00, 0x00]
                    || window == [0x4C, 0x8B, 0xAC, 0x24, 0xA0, 0x00, 0x00, 0x00]
            }),
            "Win64 frame must not save or restore r13"
        );
    }
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let bytes = majit_backend::jitframe::JitFrame::alloc_size(depth);
    let head = majit_backend::llmodel::take_or_alloc_parked_entry_frame(token, bytes);
    let info = clt.frame_info.data_ptr();
    unsafe {
        majit_backend::jitframe::JitFrame::init(head, info, depth);
        majit_backend::llmodel::set_int_value(head, 0, -1);
        majit_backend::llmodel::set_int_value(
            head,
            majit_backend_dynasm::arch::JITFRAME_FIXED_SIZE,
            input as isize,
        );
    }
    struct FreeHead<'a> {
        token: &'a JitCellToken,
        head: *mut majit_backend::jitframe::JitFrame,
    }
    impl Drop for FreeHead<'_> {
        fn drop(&mut self) {
            if self.head.is_null() {
                return;
            }
            majit_backend::llmodel::park_or_free_done_entry_frame(
                self.token,
                self.head,
                std::ptr::null_mut(),
                false,
            );
            self.head = std::ptr::null_mut();
        }
    }
    let _free = FreeHead { token, head };
    let entry = token.ll_function_addr();
    let head_addr = head as usize;
    #[cfg(target_os = "windows")]
    let mut got = [0u64; 9];
    #[cfg(not(target_os = "windows"))]
    let mut got = [0u64; 7];
    let got_ptr = got.as_mut_ptr() as usize;
    unsafe {
        #[cfg(target_os = "windows")]
        std::arch::asm!(
            "sub rsp, 128",
            "mov [rsp + 32], r13",
            "mov [rsp + 40], r14",
            "mov [rsp + 48], r15",
            "mov rax, rbx",
            "mov [rsp + 64], rax",
            "mov rax, rsi",
            "mov [rsp + 72], rax",
            "mov rax, rdi",
            "mov [rsp + 80], rax",
            "mov rax, rbp",
            "mov [rsp + 88], rax",
            "mov rax, r12",
            "mov [rsp + 96], rax",
            "mov rax, r13",
            "mov [rsp + 104], rax",
            "mov rax, r14",
            "mov [rsp + 112], rax",
            "mov rax, r15",
            "mov [rsp + 120], rax",
            "mov rbx, 0x1111111111111111",
            "mov rsi, 0x2222222222222222",
            "mov rdi, 0x3333333333333333",
            "mov rbp, 0x4444444444444444",
            "mov r12, 0x5555555555555555",
            "mov r13, 0x6666666666666666",
            "mov r14, 0x7777777777777777",
            "mov r15, 0x8888888888888888",
            "mov rcx, [rsp + 32]",
            "xor edx, edx",
            "mov rax, [rsp + 40]",
            "call rax",
            "mov r11, [rsp + 48]",
            "mov [r11], rbx",
            "mov [r11 + 8], rsi",
            "mov [r11 + 16], rdi",
            "mov [r11 + 24], rbp",
            "mov [r11 + 32], r12",
            "mov [r11 + 40], r13",
            "mov [r11 + 48], r14",
            "mov [r11 + 56], r15",
            "mov [r11 + 64], rax",
            "mov rax, [rsp + 64]",
            "mov rbx, rax",
            "mov rax, [rsp + 72]",
            "mov rsi, rax",
            "mov rax, [rsp + 80]",
            "mov rdi, rax",
            "mov rax, [rsp + 88]",
            "mov rbp, rax",
            "mov rax, [rsp + 96]",
            "mov r12, rax",
            "mov rax, [rsp + 104]",
            "mov r13, rax",
            "mov rax, [rsp + 112]",
            "mov r14, rax",
            "mov rax, [rsp + 120]",
            "mov r15, rax",
            "add rsp, 128",
            in("r13") head_addr,
            in("r14") entry,
            in("r15") got_ptr,
            clobber_abi("win64"),
        );
        #[cfg(not(target_os = "windows"))]
        std::arch::asm!(
            "sub rsp, 128",
            "mov [rsp + 32], r13",
            "mov [rsp + 40], r14",
            "mov [rsp + 48], r15",
            "mov rax, rbx",
            "mov [rsp + 64], rax",
            "mov rax, rbp",
            "mov [rsp + 72], rax",
            "mov rax, r12",
            "mov [rsp + 80], rax",
            "mov rax, r13",
            "mov [rsp + 88], rax",
            "mov rax, r14",
            "mov [rsp + 96], rax",
            "mov rax, r15",
            "mov [rsp + 104], rax",
            "mov rbx, 0x1111111111111111",
            "mov rbp, 0x4444444444444444",
            "mov r12, 0x5555555555555555",
            "mov r13, 0x6666666666666666",
            "mov r14, 0x7777777777777777",
            "mov r15, 0x8888888888888888",
            "mov rdi, [rsp + 32]",
            "xor esi, esi",
            "mov rax, [rsp + 40]",
            "call rax",
            "mov r11, [rsp + 48]",
            "mov [r11], rbx",
            "mov [r11 + 8], rbp",
            "mov [r11 + 16], r12",
            "mov [r11 + 24], r13",
            "mov [r11 + 32], r14",
            "mov [r11 + 40], r15",
            "mov [r11 + 48], rax",
            "mov rax, [rsp + 64]",
            "mov rbx, rax",
            "mov rax, [rsp + 72]",
            "mov rbp, rax",
            "mov rax, [rsp + 80]",
            "mov r12, rax",
            "mov rax, [rsp + 88]",
            "mov r13, rax",
            "mov rax, [rsp + 96]",
            "mov r14, rax",
            "mov rax, [rsp + 104]",
            "mov r15, rax",
            "add rsp, 128",
            in("r13") head_addr,
            in("r14") entry,
            in("r15") got_ptr,
            clobber_abi("sysv64"),
        );
    }
    #[cfg(target_os = "windows")]
    {
        assert_eq!(got[0], 0x1111_1111_1111_1111, "rbx");
        assert_eq!(got[1], 0x2222_2222_2222_2222, "rsi");
        assert_eq!(got[2], 0x3333_3333_3333_3333, "rdi");
        assert_eq!(got[3], 0x4444_4444_4444_4444, "rbp");
        assert_eq!(got[4], 0x5555_5555_5555_5555, "r12");
        assert_eq!(got[5], 0x6666_6666_6666_6666, "r13");
        assert_eq!(got[6], 0x7777_7777_7777_7777, "r14");
        assert_eq!(got[7], 0x8888_8888_8888_8888, "r15");
        assert_eq!(got[8] as usize, head_addr, "footer returns the jitframe");
    }
    #[cfg(not(target_os = "windows"))]
    {
        assert_eq!(got[0], 0x1111_1111_1111_1111, "rbx");
        assert_eq!(got[1], 0x4444_4444_4444_4444, "rbp");
        assert_eq!(got[2], 0x5555_5555_5555_5555, "r12");
        assert_eq!(got[3], 0x6666_6666_6666_6666, "r13");
        assert_eq!(got[4], 0x7777_7777_7777_7777, "r14");
        assert_eq!(got[5], 0x8888_8888_8888_8888, "r15");
        assert_eq!(got[6] as usize, head_addr, "footer returns the jitframe");
    }
    assert_eq!(
        majit_backend::llmodel::done_int_slot0(head),
        input,
        "finish stores the result at jf_frame[0]"
    );
}

#[test]
fn finish_loop_preserves_callee_saves() {
    let mut backend = fresh_backend();
    let token = compile_input_finish(&mut backend);
    let (depth, _) = frame_info_words(&token);
    call_finish_preserves_callee_saves(&token, depth as usize, 41);
}

fn compile_input_finish(backend: &mut DynasmBackend) -> JitCellToken {
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let ops = vec![OpRc::new(finish_of(i0, Type::Int, 1))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile finish loop: {err:?}"));
    token
}

/// `_call_header_shadowstack` pushes through `ebx`; `_call_footer` does
/// `MOV eax, ebp` and then `_call_footer_shadowstack`, which is
/// `SUB_ji8(rst, 16)` or `MOV ebx, rst; SUB [ebx], 16`.
#[test]
fn shadowstack_header_and_footer_go_through_ebx() {
    let mut backend = fresh_backend();
    let token = compile_input_finish(&mut backend);
    let code = compiled_bytes(&token);
    let header = [
        &[0x48, 0xC7, 0x03, 0x01, 0x00, 0x00, 0x00][..],
        &[0x48, 0x89, 0x6B, 0x08],
        &[0x48, 0x83, 0xC3, 0x10],
    ]
    .concat();
    assert_has(code, &header, "MOV [ebx], 1; MOV [ebx+8], ebp; ADD ebx, 16");
    let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    let mut footer = vec![0x48, 0x89, 0xE8];
    if i32::try_from(rst).is_ok() {
        footer.extend_from_slice(&[0x48, 0x83, 0x2C, 0x25]);
        footer.extend_from_slice(&(rst as i32).to_le_bytes());
        footer.push(0x10);
    } else {
        footer.extend_from_slice(&[0x48, 0xBB]);
        footer.extend_from_slice(&rst.to_le_bytes());
        footer.extend_from_slice(&[0x48, 0x83, 0x2B, 0x10]);
    }
    assert_has(code, &footer, "MOV eax, ebp; then SUB [rootstacktop], 16");
}

fn count_bytes(code: &[u8], needle: &[u8]) -> usize {
    code.windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}

fn assert_stack_probe(code: &[u8]) {
    // `sub rax, rsp` is `48 29 E0` (r/m, reg), then `ja`.
    assert_eq!(
        count_bytes(code, &[0x48, 0x29, 0xE0]),
        1,
        "one `sub rax, rsp`"
    );
    let at = unique_bytes(code, &[0x48, 0x29, 0xE0]);
    let end = (at + 3 + 32).min(code.len());
    let window = &code[at + 3..end];
    let jcc = window
        .windows(2)
        .find(|bytes| bytes[0] == 0x0F && (bytes[1] == 0x86 || bytes[1] == 0x87));
    assert_eq!(jcc, Some(&[0x0F, 0x87][..]), "not-taken ja, not jbe");
}

/// `_call_header_with_stack_check`. `STACK_CHECK_ADDRS` is a process-wide
/// `OnceLock`, so the unregistered probe and the three slow-path behaviours
/// share this one test.
#[test]
fn stack_check_slowpath_once_per_process() {
    STACK_END.store(0, Ordering::Release);
    STACK_LENGTH.store(usize::MAX, Ordering::Release);
    let _restore = StackProbeGuard;
    let mut backend = fresh_backend();
    if majit_backend_dynasm::stack_check_addresses().is_none() {
        let token = compile_input_finish(&mut backend);
        assert_eq!(
            count_bytes(compiled_bytes(&token), &[0x48, 0x29, 0xE0]),
            0,
            "no probe before insert_stack_check"
        );
    }
    majit_backend_dynasm::register_stack_check_addresses(
        STACK_END.as_ptr() as usize,
        STACK_LENGTH.as_ptr() as usize,
        stack_check_probe as *const () as usize,
    );
    let installed = majit_backend_dynasm::stack_check_addresses().expect("stack check registered");
    assert_eq!(
        installed.slowpath_addr,
        stack_check_probe as *const () as usize
    );
    assert_eq!(installed.end_adr, STACK_END.as_ptr() as usize);
    assert_eq!(installed.length_adr, STACK_LENGTH.as_ptr() as usize);

    // Same CPU: the first compile left the helper uncached, so this
    // compile must retry `ensure_stack_check_slowpath`.
    let token = compile_input_finish(&mut backend);
    assert_stack_probe(compiled_bytes(&token));
    // `_call_header_shadowstack` (`mov [rbx+8], rbp`) precedes the probe.
    assert!(
        unique_bytes(compiled_bytes(&token), &[0x48, 0x89, 0x6B, 0x08])
            < unique_bytes(compiled_bytes(&token), &[0x48, 0x29, 0xE0]),
        "the stack check follows the shadow-stack push"
    );

    let _lock = lock_exception_cells();
    let _clear = ClearExc;
    STACK_END.store(0, Ordering::Release);
    STACK_LENGTH.store(0, Ordering::Release);
    STACK_MODE.with(|mode| mode.set(2));
    STACK_CALLS.with(|calls| calls.set(0));
    let before = shadow_top();
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert_propagate(&backend, &frame);
    assert!(
        !majit_backend_dynasm::jit_exc_is_pending(),
        "propagate_exception_path clears both exception cells"
    );
    assert_eq!(
        shadow_top(),
        before,
        "the probe runs after the shadow-stack push, and          propagate_exception_path's _call_footer pops it"
    );
    assert!(
        STACK_CALLS.with(|calls| calls.get()) >= 1,
        "the ja was taken"
    );
    drop(frame);
    STACK_MODE.with(|mode| mode.set(0));
    majit_backend_dynasm::jit_exc_clear();

    // Returning 1 without publishing must fall through: the helper tests
    // `pos_exception`, not `al`.
    STACK_MODE.with(|mode| mode.set(1));
    STACK_CALLS.with(|calls| calls.set(0));
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    assert!(STACK_CALLS.with(|calls| calls.get()) >= 1);
    drop(frame);
    STACK_MODE.with(|mode| mode.set(0));
    STACK_LENGTH.store(usize::MAX, Ordering::Release);

    STACK_CALLS.with(|calls| calls.set(0));
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    assert_eq!(STACK_CALLS.with(|calls| calls.get()), 0);
}

fn nursery_backend(nursery_size: usize, threshold: usize) -> DynasmBackend {
    let gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        nursery_size,
        large_object_threshold: threshold,
        ..majit_gc::collector::GcConfig::default()
    });
    wb_backend(gc)
}

fn finish_ref(obj: OpRef, pos: u32) -> Op {
    // `genop_finish` publishes one result at `jf_frame[0]`.
    let op = Op::new(OpCode::Finish, &[rb(obj)]);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![Type::Ref]);
    op.setfailargs(vec![rb(obj)].into());
    op
}

fn compile_ops(
    backend: &mut DynasmBackend,
    inputargs: &[InputArgRc],
    ops: Vec<Op>,
) -> JitCellToken {
    let token = JitCellToken::new(next_token_id());
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile malloc nursery: {err:?}"));
    token
}

fn finished_ref(backend: &DynasmBackend, frame: &majit_backend::DeadFrame) -> majit_ir::GcRef {
    assert!(
        backend.get_latest_descr(frame).is_finish(),
        "malloc nursery trace must finish"
    );
    backend.get_ref_value(frame, 0)
}

fn store_sum(obj: OpRef, sum: OpRef, pos: u32) -> Op {
    op_at(
        OpCode::GcStore,
        &[obj, OpRef::const_int(0), sum, OpRef::const_int(8)],
        OpRef::void_op(pos),
    )
}

/// `malloc_cond` when the bump fits. The live sum is stored into the object
/// after the bump, so it has to survive the allocation.
#[test]
fn call_malloc_nursery_fits_and_keeps_live_int() {
    let mut backend = nursery_backend(1 << 20, 1 << 20);
    let a = OpRef::input_arg_int(0);
    let b = OpRef::input_arg_int(1);
    let sum = OpRef::int_op(2);
    let obj = OpRef::ref_op(3);
    let ops = vec![
        op_at(OpCode::IntAdd, &[a, b], sum),
        op_at(OpCode::CallMallocNursery, &[OpRef::const_int(64)], obj),
        store_sum(obj, sum, 4),
        finish_ref(obj, 5),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(22)]);
    let ptr = finished_ref(&backend, &frame);
    assert!(!ptr.is_null());
    assert!(
        majit_gc::can_move(ptr),
        "a 64-byte bump stays in the nursery"
    );
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 42);
}

/// `malloc_cond` when the bump passes `nursery_top`. The slow path calls
/// `malloc_slowpath_fixed`; the live sum is stored afterwards.
#[test]
fn call_malloc_nursery_overflow_keeps_live_int() {
    let mut backend = nursery_backend(4096, 1 << 20);
    let a = OpRef::input_arg_int(0);
    let b = OpRef::input_arg_int(1);
    let sum = OpRef::int_op(2);
    let obj = OpRef::ref_op(3);
    let ops = vec![
        op_at(OpCode::IntAdd, &[a, b], sum),
        op_at(OpCode::CallMallocNursery, &[OpRef::const_int(8192)], obj),
        store_sum(obj, sum, 4),
        finish_ref(obj, 5),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(22)]);
    let ptr = finished_ref(&backend, &frame);
    assert!(!ptr.is_null());
    assert!(
        !majit_gc::can_move(ptr),
        "8192 bytes do not fit a 4096-byte nursery"
    );
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 42);
}

/// `malloc_cond_varsize`: a short array takes the bump; a length past
/// `maxlength` calls `malloc_slowpath_varsize`. The live sum is stored at
/// the first item.
#[test]
fn call_malloc_nursery_varsize_small_and_large() {
    let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        nursery_size: 1 << 20,
        large_object_threshold: 4096,
        ..majit_gc::collector::GcConfig::default()
    });
    let array_tid = gc.register_type(majit_gc::TypeInfo::varsize(8, 8, 0, false, Vec::new()));
    let mut backend = wb_backend(gc);
    let a = OpRef::input_arg_int(0);
    let b = OpRef::input_arg_int(1);
    let len = OpRef::input_arg_int(2);
    let sum = OpRef::int_op(3);
    let obj = OpRef::ref_op(4);
    let alloc = op_at(
        OpCode::CallMallocNurseryVarsize,
        &[OpRef::const_int(0), OpRef::const_int(8), len],
        obj,
    );
    alloc.setdescr(Arc::new(majit_ir::descr::SimpleArrayDescr::new(
        0,
        8,
        8,
        array_tid,
        Type::Int,
    )));
    let ops = vec![
        op_at(OpCode::IntAdd, &[a, b], sum),
        alloc,
        op_at(
            OpCode::GcStore,
            &[obj, OpRef::const_int(0), len, OpRef::const_int(8)],
            OpRef::void_op(5),
        ),
        op_at(
            OpCode::GcStore,
            &[obj, OpRef::const_int(8), sum, OpRef::const_int(8)],
            OpRef::void_op(6),
        ),
        finish_ref(obj, 7),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);

    let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(22), Value::Int(3)]);
    let ptr = finished_ref(&backend, &frame);
    assert!(!ptr.is_null());
    assert!(majit_gc::can_move(ptr));
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, array_tid);
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 3);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, 42);
    drop(frame);

    let large = 140_000;
    let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(22), Value::Int(large)]);
    let ptr = finished_ref(&backend, &frame);
    assert!(!ptr.is_null());
    assert!(!majit_gc::can_move(ptr));
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, array_tid);
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, large);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, 42);
}

/// `malloc_cond_varsize` with `FLAG_STR` / `FLAG_UNICODE`: a length past
/// `maxlength` calls `malloc_slowpath_str` / `_unicode`, whose helper
/// (`malloc_str(length)`) stamps the tid and the length itself.
fn run_varsize_string_kind(kind: i64, base_size: usize, itemsize: usize, length: i64) {
    let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        nursery_size: 1 << 20,
        large_object_threshold: 4096,
        ..majit_gc::collector::GcConfig::default()
    });
    let tid = gc.register_type(majit_gc::TypeInfo::varsize(
        base_size,
        itemsize,
        8,
        false,
        Vec::new(),
    ));
    let mut backend = wb_backend(gc);
    let len = OpRef::input_arg_int(0);
    let obj = OpRef::ref_op(1);
    let alloc = op_at(
        OpCode::CallMallocNurseryVarsize,
        &[
            OpRef::const_int(kind),
            OpRef::const_int(itemsize as i64),
            len,
        ],
        obj,
    );
    alloc.setdescr(Arc::new(majit_ir::descr::SimpleArrayDescr::new(
        0,
        base_size,
        itemsize,
        tid,
        Type::Int,
    )));
    let ops = vec![alloc, finish_ref(obj, 2)];
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Int(length)]);
    let ptr = finished_ref(&backend, &frame);
    assert!(!ptr.is_null());
    assert!(!majit_gc::can_move(ptr));
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, tid);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, length);
}

#[test]
fn call_malloc_nursery_varsize_str_and_unicode_slowpaths() {
    run_varsize_string_kind(1, majit_backend::BUILTIN_STR_TOKEN_BASE_SIZE, 1, 140_000);
    run_varsize_string_kind(2, 16, 4, 140_000);
}

/// `consider_call_malloc_nursery_varsize`: the length is an `INT_ADD`
/// result, so it lives in a register. When that box dies at the alloc,
/// `spill_or_move_registers_before_call` must not run — it would unbind
/// the length with no stack sync. When a later store keeps it alive, the
/// spill is legal and `loc` still sees the value.
fn run_varsize_from_add(length_lives: bool, len_a: i64, len_b: i64) -> (majit_ir::GcRef, u32, i64) {
    let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        nursery_size: 1 << 20,
        large_object_threshold: 4096,
        ..majit_gc::collector::GcConfig::default()
    });
    let array_tid = gc.register_type(majit_gc::TypeInfo::varsize(8, 8, 0, false, Vec::new()));
    let mut backend = wb_backend(gc);
    let a = OpRef::input_arg_int(0);
    let b = OpRef::input_arg_int(1);
    let n = OpRef::input_arg_int(2);
    let k = OpRef::input_arg_int(3);
    let sum = OpRef::int_op(4);
    let len = OpRef::int_op(5);
    let obj = OpRef::ref_op(6);
    let alloc = op_at(
        OpCode::CallMallocNurseryVarsize,
        &[OpRef::const_int(0), OpRef::const_int(8), len],
        obj,
    );
    alloc.setdescr(Arc::new(majit_ir::descr::SimpleArrayDescr::new(
        0,
        8,
        8,
        array_tid,
        Type::Int,
    )));
    let mut ops = vec![
        op_at(OpCode::IntAdd, &[a, b], sum),
        op_at(OpCode::IntAdd, &[n, k], len),
        alloc,
    ];
    let mut pos = 7u32;
    if length_lives {
        ops.push(gc_store(obj, 0, len, 8, pos));
        pos += 1;
    }
    ops.push(gc_store(obj, 8, sum, 8, pos));
    ops.push(finish_ref(obj, pos + 1));
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
        InputArg::from_type_rc(Type::Int, 3),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(20),
            Value::Int(22),
            Value::Int(len_a),
            Value::Int(len_b),
        ],
    );
    let ptr = finished_ref(&backend, &frame);
    (ptr, array_tid, len_a.wrapping_add(len_b))
}

#[test]
fn call_malloc_nursery_varsize_length_dies_and_stays() {
    let (ptr, array_tid, _len) = run_varsize_from_add(false, 1, 2);
    assert!(!ptr.is_null());
    assert!(majit_gc::can_move(ptr), "length 3 stays in the nursery");
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, array_tid);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, 42);

    let (ptr, array_tid, len) = run_varsize_from_add(false, 100_000, 40_000);
    assert!(!ptr.is_null());
    assert!(
        !majit_gc::can_move(ptr),
        "length {len} must take the non-moving path"
    );
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, array_tid);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, 42);

    let (ptr, array_tid, len) = run_varsize_from_add(true, 1, 2);
    assert!(!ptr.is_null());
    assert!(majit_gc::can_move(ptr));
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, array_tid);
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, len);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, 42);

    let (ptr, array_tid, len) = run_varsize_from_add(true, 100_000, 40_000);
    assert!(!ptr.is_null());
    assert!(!majit_gc::can_move(ptr));
    assert_eq!(unsafe { (*header_of(ptr.0)).type_id() }, array_tid);
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, len);
    assert_eq!(unsafe { *((ptr.0 + 8) as *const i64) }, 42);
}

fn run_varsize_frame(nursery_size: usize, alloc_size: i64) -> majit_ir::GcRef {
    let mut backend = nursery_backend(nursery_size, 1 << 20);
    let size = OpRef::input_arg_int(0);
    let a = OpRef::input_arg_int(1);
    let b = OpRef::input_arg_int(2);
    let sum = OpRef::int_op(3);
    let obj = OpRef::ref_op(4);
    let ops = vec![
        op_at(OpCode::IntAdd, &[a, b], sum),
        op_at(OpCode::CallMallocNurseryVarsizeFrame, &[size], obj),
        store_sum(obj, sum, 5),
        finish_ref(obj, 6),
    ];
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(
        &token,
        &[Value::Int(alloc_size), Value::Int(20), Value::Int(22)],
    );
    finished_ref(&backend, &frame)
}

/// `malloc_cond_varsize_frame` bump, and the same op when the size does not fit.
#[test]
fn call_malloc_nursery_varsize_frame_fit_and_overflow() {
    let ptr = run_varsize_frame(1 << 20, 32);
    assert!(!ptr.is_null());
    assert!(majit_gc::can_move(ptr));
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 42);

    let ptr = run_varsize_frame(4096, 8192);
    assert!(!ptr.is_null());
    assert!(!majit_gc::can_move(ptr));
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 42);
}

fn finish_int(value: OpRef, pos: u32) -> Op {
    let op = Op::new(OpCode::Finish, &[rb(value)]);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![Type::Int]);
    op.setfailargs(vec![rb(value)].into());
    op
}

/// `TypeInfo::simple` is registered before the jitframe type so a zero
/// header is not traced as a frame. `tid` is that simple type.
fn nursery_backend_with_simple(
    nursery_size: usize,
    threshold: usize,
    payload: usize,
) -> (DynasmBackend, u32) {
    let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        nursery_size,
        large_object_threshold: threshold,
        ..majit_gc::collector::GcConfig::default()
    });
    let tid = gc.register_type(majit_gc::TypeInfo::simple(payload));
    (wb_backend(gc), tid)
}

fn gc_store(base: OpRef, ofs: i64, value: OpRef, size: i64, pos: u32) -> Op {
    op_at(
        OpCode::GcStore,
        &[base, OpRef::const_int(ofs), value, OpRef::const_int(size)],
        OpRef::void_op(pos),
    )
}

/// `malloc_cond` overflow. The young object stays live across
/// `MallocCondSlowPath`. The collection promotes it and the register
/// must hold the updated pointer, with the marker still in the first word.
#[test]
fn call_malloc_nursery_slowpath_updates_live_ref() {
    let (mut backend, tid) = nursery_backend_with_simple(4096, 1 << 20, 64);
    let young = OpRef::ref_op(1);
    let ops = vec![
        op_at(OpCode::CallMallocNursery, &[OpRef::const_int(2048)], young),
        gc_store(
            young,
            -(GcHeader::SIZE as i64),
            OpRef::const_int(i64::from(tid)),
            8,
            3,
        ),
        gc_store(young, 0, OpRef::const_int(0x5A5A), 8, 4),
        op_at(
            OpCode::CallMallocNursery,
            &[OpRef::const_int(3000)],
            OpRef::ref_op(5),
        ),
        finish_ref(young, 6),
    ];
    let token = compile_ops(&mut backend, &[], ops);
    let frame = backend.execute_token(&token, &[]);
    let ptr = finished_ref(&backend, &frame);
    assert!(
        !majit_gc::can_move(ptr),
        "live ref {:#x} was not updated out of the nursery",
        ptr.0
    );
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 0x5A5A);
    assert!(
        !unsafe { (*header_of(ptr.0)).has_flag(GcFlags::GCFLAG_HAS_SHADOW) },
        "promoted object kept HAS_SHADOW"
    );
}

/// Header word after the fast path and a 4-byte type-id store.
/// `FORWARDED_MARKER` was planted at that slot. The high half stays set
/// until `malloc_cond_varsize_frame` clears the word.
#[test]
fn call_malloc_nursery_varsize_frame_fast_path_header_word() {
    let (mut backend, tid) = nursery_backend_with_simple(1 << 20, 1 << 20, 64);
    let size = OpRef::input_arg_int(0);
    let pad = OpRef::ref_op(1);
    let young = OpRef::ref_op(2);
    let hdr = OpRef::int_op(3);
    let ops = vec![
        op_at(OpCode::CallMallocNursery, &[OpRef::const_int(64)], pad),
        gc_store(
            pad,
            56,
            OpRef::const_int(majit_gc::header::FORWARDED_MARKER as i64),
            8,
            4,
        ),
        op_at(OpCode::CallMallocNurseryVarsizeFrame, &[size], young),
        gc_store(
            young,
            -(GcHeader::SIZE as i64),
            OpRef::const_int(i64::from(tid)),
            (majit_gc::header::TYPE_ID_BITS / 8) as i64,
            5,
        ),
        op_at(
            OpCode::GcLoadI,
            &[
                young,
                OpRef::const_int(-(GcHeader::SIZE as i64)),
                OpRef::const_int(8),
            ],
            hdr,
        ),
        finish_int(hdr, 6),
    ];
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Int(64)]);
    let word = backend.get_int_value(&frame, 0) as u64;
    assert_eq!(
        word >> 32,
        0,
        "recycled header flags survived the fast path: {word:#x}"
    );
    assert_eq!(word as u32, tid);
}

/// `malloc_cond_varsize_frame` fast path, then a 4-byte type-id store
/// (`gen_initialize_tid_keep_flags`). The slot is planted with
/// `FORWARDED_MARKER`. That word's flag half includes `GCFLAG_PINNED`
/// and `GCFLAG_HAS_SHADOW`, so a nursery-overflowing `malloc_cond`
/// leaves the live register pointing at the nursery slot. Clearing the
/// header word lets the collection promote the object and rewrite it.
#[test]
fn call_malloc_nursery_varsize_frame_live_ref_across_recycled_header() {
    let (mut backend, tid) = nursery_backend_with_simple(4096, 1 << 20, 64);
    let size = OpRef::input_arg_int(0);
    let pad = OpRef::ref_op(1);
    let young = OpRef::ref_op(2);
    let header_plant = majit_gc::header::FORWARDED_MARKER as i64;
    let ops = vec![
        op_at(OpCode::CallMallocNursery, &[OpRef::const_int(64)], pad),
        gc_store(pad, 56, OpRef::const_int(header_plant), 8, 4),
        op_at(OpCode::CallMallocNurseryVarsizeFrame, &[size], young),
        gc_store(
            young,
            -(GcHeader::SIZE as i64),
            OpRef::const_int(i64::from(tid)),
            (majit_gc::header::TYPE_ID_BITS / 8) as i64,
            5,
        ),
        gc_store(young, 0, OpRef::const_int(0x5A5A), 8, 6),
        op_at(
            OpCode::CallMallocNursery,
            &[OpRef::const_int(4000)],
            OpRef::ref_op(7),
        ),
        finish_ref(young, 8),
    ];
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Int(64)]);
    let ptr = finished_ref(&backend, &frame);
    assert!(
        !majit_gc::can_move(ptr),
        "live ref {:#x} was not updated out of the nursery",
        ptr.0
    );
    assert_eq!(unsafe { *(ptr.0 as *const i64) }, 0x5A5A);
    let hdr = unsafe { *header_of(ptr.0) };
    assert!(
        !hdr.has_flag(GcFlags::GCFLAG_HAS_SHADOW),
        "recycled header kept HAS_SHADOW: {:#x}",
        hdr.tid_and_flags
    );
}

const ZERO_ARRAY_BASEOFS: usize = 16;
const ZERO_ARRAY_UNROLL_LIMIT: i64 = 16 * 8;

fn zero_array_descr() -> DescrRef {
    Arc::new(majit_ir::descr::SimpleArrayDescr::new(
        0,
        ZERO_ARRAY_BASEOFS,
        1,
        1,
        Type::Int,
    ))
}

fn run_zero_array(nbytes: i64, start: i64, start_in_reg: bool) {
    run_zero_array_at(nbytes, start, start_in_reg, false, 0);
}

/// `base_const` passes the array as an immediate base. `bias` moves the
/// base down and an immediate start up by the same amount, so the stores
/// land on the same bytes through a larger displacement.
fn run_zero_array_at(nbytes: i64, start: i64, start_in_reg: bool, base_const: bool, bias: i64) {
    let prefix = ZERO_ARRAY_BASEOFS + start as usize;
    let total = prefix + nbytes as usize + 8;
    let mut buf = vec![0xAAu8; total];
    for slot in buf.iter_mut().take(prefix + nbytes as usize).skip(prefix) {
        *slot = 0xBB;
    }
    let ptr = buf.as_mut_ptr() as usize;
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let base_value = (ptr as i64).wrapping_sub(bias);
    let mut inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let mut zero_args = vec![
        rb(OpRef::input_arg_ref(0)),
        rb(OpRef::const_int(start + bias)),
        rb(OpRef::const_int(nbytes)),
        rb(OpRef::const_int(1)),
        rb(OpRef::const_int(1)),
    ];
    if base_const {
        zero_args[0] = rb(OpRef::const_int(base_value));
    }
    if start_in_reg {
        inputargs.push(InputArg::from_type_rc(Type::Int, 1));
        zero_args[1] = rb(OpRef::input_arg_int(1));
    }
    let zero = Op::new(OpCode::ZeroArray, &zero_args);
    zero.pos().set(OpRef::void_op(1));
    zero.setdescr(zero_array_descr());
    let ops = vec![OpRc::new(zero), OpRc::new(finish(2))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile ZERO_ARRAY nbytes={nbytes}: {err:?}"));
    let mut values = vec![Value::Ref(GcRef(base_value as usize))];
    if start_in_reg {
        values.push(Value::Int(start + bias));
    }
    let frame = backend.execute_token(&token, &values);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "ZERO_ARRAY nbytes={nbytes} must finish"
    );
    assert!(
        buf[..prefix].iter().all(|&b| b == 0xAA),
        "bytes before the range must stay set, nbytes={nbytes}: {:02x?}",
        &buf[..prefix]
    );
    assert!(
        buf[prefix..prefix + nbytes as usize]
            .iter()
            .all(|&b| b == 0),
        "ZERO_ARRAY must clear nbytes={nbytes}: {:02x?}",
        &buf[prefix..prefix + nbytes as usize]
    );
    assert!(
        buf[prefix + nbytes as usize..].iter().all(|&b| b == 0xAA),
        "bytes after the range must stay set, nbytes={nbytes}: {:02x?}",
        &buf[prefix + nbytes as usize..]
    );
}

#[test]
fn zero_array_unroll_and_memset_leave_neighbors() {
    for nbytes in [0, 8, 16, 24, 136, ZERO_ARRAY_UNROLL_LIMIT + 16] {
        run_zero_array(nbytes, 0, false);
    }
    run_zero_array(16, 4, true);
}

#[test]
fn zero_array_immediate_base() {
    // `addr_add` with an `ImmedLoc` base: inline stores and the memset
    // destination, with an immediate and a register start.
    for nbytes in [24, ZERO_ARRAY_UNROLL_LIMIT + 16] {
        run_zero_array_at(nbytes, 0, false, true, 0);
        run_zero_array_at(nbytes, 4, true, true, 0);
    }
}

#[test]
fn zero_array_displacement_crosses_i32() {
    // The first 16-byte store sits just below `i32::MAX`; the second is
    // above it and must go through the scratch register
    // (`_fix_static_offset_64_m`), not wrap to a negative displacement.
    let bias = i64::from(i32::MAX) - 8 - ZERO_ARRAY_BASEOFS as i64;
    run_zero_array_at(32, 0, false, false, bias);
    run_zero_array_at(ZERO_ARRAY_UNROLL_LIMIT + 16, 0, false, false, bias);
}

/// `consider_zero_array`'s memset() path with a variable length:
/// `LEA` of the destination, `LEA` of the byte length when the length
/// scale is 8, and `simple_call_no_collect`. A value computed before the
/// call and the length itself must survive it.
fn run_zero_array_variable_length(
    items: i64,
    start: i64,
    len_itemsize: i64,
    use_length_after: bool,
) {
    const BASEOFS: usize = 16;
    let total = BASEOFS + ((start + items) as usize + 2) * 8;
    let mut buf = vec![0xAAu8; total];
    let ptr = buf.as_mut_ptr() as usize;
    let descr: DescrRef = Arc::new(majit_ir::descr::SimpleArrayDescr::new(
        0,
        BASEOFS,
        8,
        1,
        Type::Int,
    ));
    let length = if len_itemsize == 1 { items * 8 } else { items };
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let x = OpRef::int_op(3);
    let add = Op::new(
        OpCode::IntAdd,
        &[rb(OpRef::input_arg_int(1)), rb(OpRef::const_int(1000))],
    );
    add.pos().set(x);
    let zero = Op::new(
        OpCode::ZeroArray,
        &[
            rb(OpRef::input_arg_ref(0)),
            rb(OpRef::input_arg_int(1)),
            rb(OpRef::input_arg_int(2)),
            rb(OpRef::const_int(8)),
            rb(OpRef::const_int(len_itemsize)),
        ],
    );
    zero.pos().set(OpRef::void_op(4));
    zero.setdescr(descr);
    let mut ops = vec![OpRc::new(add), OpRc::new(zero)];
    let result = if use_length_after {
        let y = OpRef::int_op(5);
        let sum = Op::new(OpCode::IntAdd, &[rb(x), rb(OpRef::input_arg_int(2))]);
        sum.pos().set(y);
        ops.push(OpRc::new(sum));
        y
    } else {
        x
    };
    ops.push(OpRc::new(finish_of(result, Type::Int, 6)));
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile variable ZERO_ARRAY: {err:?}"));
    let frame = backend.execute_token(
        &token,
        &[
            Value::Ref(GcRef(ptr)),
            Value::Int(start),
            Value::Int(length),
        ],
    );
    let expected = start + 1000 + if use_length_after { length } else { 0 };
    assert_eq!(backend.get_int_value(&frame, 0), expected);
    let lo = BASEOFS + start as usize * 8;
    let hi = lo + items as usize * 8;
    assert!(buf[..lo].iter().all(|&b| b == 0xAA), "{:02x?}", &buf[..lo]);
    assert!(buf[lo..hi].iter().all(|&b| b == 0), "{:02x?}", &buf[lo..hi]);
    assert!(buf[hi..].iter().all(|&b| b == 0xAA), "{:02x?}", &buf[hi..]);
}

#[test]
fn zero_array_memset_variable_length() {
    for len_itemsize in [1, 8] {
        for use_length_after in [false, true] {
            run_zero_array_variable_length(40, 3, len_itemsize, use_length_after);
            run_zero_array_variable_length(0, 2, len_itemsize, use_length_after);
        }
    }
}

static CALL_ENTRY_RSP: AtomicUsize = AtomicUsize::new(0);

fn assert_call_entry_aligned() {
    let rsp = CALL_ENTRY_RSP.load(Ordering::SeqCst);
    assert_ne!(rsp, 0, "callee must record entry rsp");
    assert_eq!(
        (rsp + 8) % 16,
        0,
        "rsp at entry minus the return address must be 16-byte aligned, rsp={rsp:#x}"
    );
}

macro_rules! aligned_int_callee {
    ($entry:ident, $real:ident $(, $arg:ident: $ty:ty = $expect:expr)*) => {
        #[unsafe(naked)]
        extern "C" fn $entry() {
            naked_asm!(
                "mov [rip + {seen}], rsp",
                "jmp {real}",
                seen = sym CALL_ENTRY_RSP,
                real = sym $real,
            );
        }
        extern "C" fn $real($($arg: $ty),*) -> i64 {
            let ok = true $(&& $arg == $expect)*;
            ok as i64
        }
    };
}

aligned_int_callee!(call_align_0, call_align_0_real);
aligned_int_callee!(
    call_align_4,
    call_align_4_real,
    a0: i64 = 1,
    a1: i64 = 2,
    a2: i64 = 3,
    a3: i64 = 4
);
aligned_int_callee!(
    call_align_6,
    call_align_6_real,
    a0: i64 = 1,
    a1: i64 = 2,
    a2: i64 = 3,
    a3: i64 = 4,
    a4: i64 = 5,
    a5: i64 = 6
);
aligned_int_callee!(
    call_align_7,
    call_align_7_real,
    a0: i64 = 1,
    a1: i64 = 2,
    a2: i64 = 3,
    a3: i64 = 4,
    a4: i64 = 5,
    a5: i64 = 6,
    a6: i64 = 7
);
aligned_int_callee!(
    call_align_9,
    call_align_9_real,
    a0: i64 = 1,
    a1: i64 = 2,
    a2: i64 = 3,
    a3: i64 = 4,
    a4: i64 = 5,
    a5: i64 = 6,
    a6: i64 = 7,
    a7: i64 = 8,
    a8: i64 = 9
);
aligned_int_callee!(
    call_align_19,
    call_align_19_real,
    a0: i64 = 1,
    a1: i64 = 2,
    a2: i64 = 3,
    a3: i64 = 4,
    a4: i64 = 5,
    a5: i64 = 6,
    a6: i64 = 7,
    a7: i64 = 8,
    a8: i64 = 9,
    a9: i64 = 10,
    a10: i64 = 11,
    a11: i64 = 12,
    a12: i64 = 13,
    a13: i64 = 14,
    a14: i64 = 15,
    a15: i64 = 16,
    a16: i64 = 17,
    a17: i64 = 18,
    a18: i64 = 19
);

#[unsafe(naked)]
extern "C" fn call_align_float() {
    naked_asm!(
        "mov [rip + {seen}], rsp",
        "jmp {real}",
        seen = sym CALL_ENTRY_RSP,
        real = sym call_align_float_real,
    );
}

extern "C" fn call_align_float_real(a: f64, b: f64) -> i64 {
    (a == 1.5 && b == 2.25) as i64
}

fn run_aligned_call(func: i64, args: &[Operand], types: Vec<Type>) {
    CALL_ENTRY_RSP.store(0, Ordering::SeqCst);
    let (backend, token) = compile_call(
        OpCode::CallI,
        func,
        args,
        types,
        OpRef::int_op(1),
        Type::Int,
        true,
        8,
    );
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        1,
        "callee must see every argument"
    );
    assert_call_entry_aligned();
}

fn int_args(n: i64) -> (Vec<Operand>, Vec<Type>) {
    let args = (1..=n).map(|i| rb(OpRef::const_int(i))).collect();
    (args, vec![Type::Int; n as usize])
}

#[test]
fn call_subtract_esp_aligned_passes_args() {
    let (args, types) = int_args(0);
    run_aligned_call(call_align_0 as usize as i64, &args, types);
    let (args, types) = int_args(4);
    run_aligned_call(call_align_4 as usize as i64, &args, types);
    let (args, types) = int_args(6);
    run_aligned_call(call_align_6 as usize as i64, &args, types);
    let (args, types) = int_args(7);
    run_aligned_call(call_align_7 as usize as i64, &args, types);
    let (args, types) = int_args(9);
    run_aligned_call(call_align_9 as usize as i64, &args, types);
    // 19 ints: SysV puts 13 words on the stack, Win64's `on_stack` is 19.
    // Both exceed `PASS_ON_MY_FRAME`, so the call subtracts.
    let (args, types) = int_args(19);
    run_aligned_call(call_align_19 as usize as i64, &args, types);
    run_aligned_call(
        call_align_float as usize as i64,
        &[rb(OpRef::const_float(1.5)), rb(OpRef::const_float(2.25))],
        vec![Type::Float, Type::Float],
    );
}

fn compile_add_regs(lhs: i64, rhs: i64) -> i64 {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let add = Op::new(
        OpCode::IntAdd,
        &[rb(inputargs[0].opref()), rb(inputargs[1].opref())],
    );
    add.pos().set(OpRef::int_op(1));
    let ops = vec![
        OpRc::new(add),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile reg/reg INT_ADD: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(lhs), Value::Int(rhs)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_int_value(&frame, 0)
}

/// LHS stays live past the add, and the immediate fits a disp32, so
/// `_consider_lea` puts the sum in a register other than the LHS.
/// A failing guard publishes both; `genop_finish` only stores its first arg.
/// Returns `(sum, lhs)`.
fn compile_add_fresh(imm: i64, input: i64) -> (i64, i64) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let add = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(imm))]);
    add.pos().set(OpRef::int_op(1));
    // `input` is never 0, so this compare fails and the guard exits.
    let eq = Op::new(OpCode::IntEq, &[rb(i0), rb(OpRef::const_int(0))]);
    eq.pos().set(OpRef::int_op(2));
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard.pos().set(OpRef::void_op(3));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int, Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int, Type::Int]);
    guard.setfailargs(vec![rb(OpRef::int_op(1)), rb(i0)].into());
    let ops = vec![
        OpRc::new(add),
        OpRc::new(eq),
        OpRc::new(guard),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 4)),
    ];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile fresh INT_ADD: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(input)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    (
        backend.get_int_value(&frame, 0),
        backend.get_int_value(&frame, 1),
    )
}

/// `_binaryop_or_lea`: ADD when the result reuses the first operand's
/// register, LEA when the sum is a fresh register. Both wrap.
#[test]
fn int_add_reuses_lhs_or_leas_into_a_fresh_reg() {
    assert_eq!(compile_add_regs(20, 22), 42);
    assert_eq!(compile_add_regs(i64::MAX, 1), i64::MIN);
    let (sum, saved) = compile_add_fresh(3, 10);
    assert_eq!((sum, saved), (13, 10));
    let (sum, saved) = compile_add_fresh(1, i64::MAX);
    assert_eq!((sum, saved), (i64::MIN, i64::MAX));
}

/// `flush_cc` materialises every comparison condition into a register.
#[test]
fn flush_cc_materializes_every_condition() {
    let cases = [
        (OpCode::IntLt, -2, -1, 1),
        (OpCode::IntLt, 2, 2, 0),
        (OpCode::IntLe, 2, 2, 1),
        (OpCode::IntLe, 3, 2, 0),
        (OpCode::IntGt, 3, 2, 1),
        (OpCode::IntGt, 2, 2, 0),
        (OpCode::IntGe, 2, 2, 1),
        (OpCode::IntGe, 1, 2, 0),
        (OpCode::IntEq, 5, 5, 1),
        (OpCode::IntEq, 5, 6, 0),
        (OpCode::IntNe, 5, 6, 1),
        (OpCode::IntNe, 5, 5, 0),
        (OpCode::UintLt, 1, -1, 1),
        (OpCode::UintLt, -1, 1, 0),
        (OpCode::UintLe, -1, -1, 1),
        (OpCode::UintLe, -1, 1, 0),
        (OpCode::UintGt, -1, 1, 1),
        (OpCode::UintGt, 1, -1, 0),
        (OpCode::UintGe, -1, -1, 1),
        (OpCode::UintGe, 1, -1, 0),
    ];
    for (op, input, rhs, expected) in cases {
        let got = compile_and_run_int(op, Some(rhs), input, next_token_id() as u64);
        assert_eq!(got, expected, "{op:?}({input}, {rhs})");
    }
}

fn imm_base_encoded(code: &[u8], addr: i64) -> bool {
    let bytes = (addr as u64).to_le_bytes();
    if code.windows(8).any(|window| window == bytes) {
        return true;
    }
    if (-2147483648..=2147483647).contains(&addr)
        && code.windows(4).any(|window| window == &bytes[..4])
    {
        return true;
    }
    // `_addr_as_reg_offset` reuses r11: an earlier imm64 plus a disp32.
    code.windows(8).any(|window| {
        let base = i64::from_le_bytes(window.try_into().unwrap());
        let delta = addr.wrapping_sub(base);
        (-2147483648..=2147483647).contains(&delta)
            && code
                .windows(4)
                .any(|disp| disp == (delta as i32).to_le_bytes())
    })
}

/// `addr_add_const` of an immediate base: flag clear falls through, flag
/// set pushes that immediate and calls the helper.
#[test]
fn write_barrier_immediate_base_checks_the_flag_byte() {
    for clear_flag in [true, false] {
        let mut gc = MiniMarkGC::new();
        let tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let old = majit_gc::GcAllocator::alloc_oldgen_typed(&mut gc, tid, 16);
        unsafe {
            let hdr = &mut *header_of(old.0);
            hdr.set_flag(GcFlags::GCFLAG_NO_HEAP_PTRS);
            if clear_flag {
                hdr.clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            } else {
                assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
            }
        }
        let mut backend = wb_backend(gc);
        let addr = old.0 as i64;
        let ops = vec![
            op_at(
                OpCode::CondCallGcWb,
                &[OpRef::const_int(addr)],
                OpRef::void_op(1),
            ),
            finish(2),
        ];
        let token = JitCellToken::new(next_token_id());
        let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
        backend
            .compile_loop(&[], &ops, &token)
            .unwrap_or_else(|err| panic!("compile immediate write barrier: {err:?}"));
        let byteofs = majit_gc::WriteBarrierDescr::for_current_gc().jit_wb_if_flag_byteofs;
        let flag_addr = addr.wrapping_add(i64::from(byteofs));
        assert!(
            imm_base_encoded(compiled_bytes(&token), flag_addr),
            "TEST8 must encode the immediate flag address {flag_addr:#x}"
        );
        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let hdr = unsafe { &*header_of(old.0) };
        assert!(!hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert_eq!(
            hdr.has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS),
            clear_flag,
            "helper runs only when the flag byte is set (clear_flag={clear_flag})"
        );
    }
}

/// `genop_finish` stores a ref and a float, and the rip-relative descr
/// word still classifies the exit.
#[test]
fn finish_returns_ref_and_float() {
    let mut gc = MiniMarkGC::new();
    let tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let obj = gc.alloc_with_type(tid, 16);
    let mut backend = wb_backend(gc);
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let ops = vec![OpRc::new(finish_of(inputargs[0].opref(), Type::Ref, 1))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile FINISH ref: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Ref(obj)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_ref_value(&frame, 0), obj);

    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let ops = vec![OpRc::new(finish_of(inputargs[0].opref(), Type::Float, 1))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile FINISH float: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Float(-2.25)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_float_value(&frame, 0), -2.25);
}

/// `closing_jump` to another loop's relocated token runs that loop.
/// A target that was never compiled stays below `MIN_RELOCATED_JUMP_TARGET`
/// and `check_unrelocated_jump_target` fails the compile.
#[test]
fn jump_to_another_loop_target_token() {
    let mut backend = fresh_backend();
    let unresolved = make_loop_target_descr(next_token_id() as u64, false);
    let jump = Op::new(OpCode::Jump, &[]);
    jump.pos().set(OpRef::void_op(1));
    jump.setdescr(unresolved);
    let token = JitCellToken::new(next_token_id());
    let err = backend
        .compile_loop(&[], &[OpRc::new(jump)], &token)
        .expect_err("unrelocated JUMP must fail to compile");
    assert!(
        err.to_string()
            .contains("is not a relocated executable address"),
        "{err}"
    );

    let descr = make_loop_target_descr(next_token_id() as u64, false);
    let target = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let label = Op::new(OpCode::Label, &[rb(i0)]);
    label.pos().set(OpRef::void_op(1));
    label.setdescr(descr.clone());
    let ops = vec![OpRc::new(label), OpRc::new(finish_of(i0, Type::Int, 2))];
    backend
        .compile_loop(&inputargs, &ops, &target)
        .unwrap_or_else(|err| panic!("compile jump target: {err:?}"));

    let source = JitCellToken::new(next_token_id());
    let jump = Op::new(OpCode::Jump, &[rb(i0)]);
    jump.pos().set(OpRef::void_op(1));
    jump.setdescr(descr);
    backend
        .compile_loop(&inputargs, &[OpRc::new(jump)], &source)
        .unwrap_or_else(|err| panic!("compile JUMP: {err:?}"));
    let frame = backend.execute_token(&source, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 42);
    // Keep the target buffer mapped across the jump.
    drop(target);
}

fn read_bytes(ptr: *const u8, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len) };
    out
}

fn read_i64_unaligned(ptr: *const u8) -> i64 {
    i64::from_le_bytes(read_bytes(ptr, 8).try_into().unwrap())
}

/// `save_into_mem`: constants 0 and -1 fit every width (`MOV*_ai`);
/// `0x7fff_ffff` fits 4 and 8; `0x1_0000_0000` fits none and is staged
/// through the scratch register. The index is an input, not a constant.
#[test]
fn gc_store_indexed_immediate_values_round_trip_memory() {
    let values = [0i64, -1, 0x7fff_ffff, 0x1_0000_0000];
    let sizes = [1i64, 2, 4, 8];
    let factors = [1i64, 8];
    let mut slots = Vec::new();
    let mut slot = 0i64;
    for &value in &values {
        for &size in &sizes {
            for &factor in &factors {
                slots.push((slot * 16, factor, size, value));
                slot += 1;
            }
        }
    }
    let bytes = (slot as usize) * 16 + 16;
    let mut storage = vec![0xA5u8; bytes];
    let base_ptr = storage.as_mut_ptr();

    let mut backend = fresh_backend();
    let base = OpRef::input_arg_ref(0);
    let index = OpRef::input_arg_int(1);
    let mut ops = Vec::new();
    for (n, &(static_ofs, factor, size, value)) in slots.iter().enumerate() {
        ops.push(op_at(
            OpCode::GcStoreIndexed,
            &[
                base,
                index,
                OpRef::const_int(value),
                OpRef::const_int(factor),
                OpRef::const_int(static_ofs),
                OpRef::const_int(size),
            ],
            OpRef::void_op(n as u32 + 1),
        ));
    }
    ops.push(finish_int(index, slots.len() as u32 + 1));
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(
        &token,
        &[Value::Ref(GcRef(base_ptr as usize)), Value::Int(1)],
    );
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 1);

    for &(static_ofs, factor, size, value) in &slots {
        let at = unsafe { base_ptr.add((static_ofs + factor) as usize) };
        let size = size as usize;
        assert_eq!(
            unsafe { *at.sub(1) },
            0xA5,
            "byte before value {value:#x} size {size} factor {factor}"
        );
        assert_eq!(
            read_bytes(at, size),
            &value.to_le_bytes()[..size],
            "value {value:#x} size {size} factor {factor} ofs {static_ofs}"
        );
        assert_eq!(
            unsafe { *at.add(size) },
            0xA5,
            "byte after value {value:#x} size {size} factor {factor}"
        );
    }
}

/// GC_STORE_INDEXED with a `ConstInt` index, end to end: `cpu_simplify_scale`
/// folds `index * factor + offset` into a GC_STORE offset. Every value lives
/// in a register, so the size-1 stores read `lowest8bits` of `rsi`/`rdi`/
/// `r8`.. as well as `rax`..`rbx`. The backend's own handling of an
/// `ImmedLoc` index is covered by the `x86::assembler` unit tests.
#[test]
fn gc_store_indexed_constant_index_stores_registers() {
    const N: usize = 12;
    let mut storage = vec![0xA5u8; 512];
    let base_ptr = storage.as_mut_ptr();

    let mut backend = fresh_backend();
    let base = OpRef::input_arg_ref(0);
    let mut inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let mut ops = Vec::new();
    let mut slots = Vec::new();
    for k in 0..N {
        inputargs.push(InputArg::from_type_rc(Type::Int, k as u32 + 1));
        let value = OpRef::input_arg_int(k as u32 + 1);
        // index * factor + offset: (3 * 8 + 8) for the word, then one byte
        // slot per value past it.
        let (index, factor, offset, size) = if k == 0 {
            (3i64, 8i64, 8i64, 8i64)
        } else {
            (k as i64, 2, 40 + k as i64, 1)
        };
        slots.push((index * factor + offset, size));
        ops.push(op_at(
            OpCode::GcStoreIndexed,
            &[
                base,
                OpRef::const_int(index),
                value,
                OpRef::const_int(factor),
                OpRef::const_int(offset),
                OpRef::const_int(size),
            ],
            OpRef::void_op(k as u32 + N as u32 + 1),
        ));
    }
    // Keep every value live across the stores.
    let mut sum = OpRef::input_arg_int(1);
    let mut pos = 2 * N as u32 + 1;
    for k in 1..N {
        let next = OpRef::int_op(pos);
        ops.push(op_at(
            OpCode::IntAdd,
            &[sum, OpRef::input_arg_int(k as u32 + 1)],
            next,
        ));
        sum = next;
        pos += 1;
    }
    ops.push(finish_int(sum, pos));
    let token = compile_ops(&mut backend, &inputargs, ops);

    let values: Vec<i64> = (0..N as i64)
        .map(|k| {
            if k == 0 {
                0x0123_4567_89ab_cdef
            } else {
                0x1100 + k
            }
        })
        .collect();
    let mut args = vec![Value::Ref(GcRef(base_ptr as usize))];
    args.extend(values.iter().map(|&v| Value::Int(v)));
    let frame = backend.execute_token(&token, &args);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        values.iter().copied().fold(0i64, i64::wrapping_add)
    );

    for (k, &(at, size)) in slots.iter().enumerate() {
        let at = at as usize;
        let size = size as usize;
        assert_eq!(
            read_bytes(unsafe { base_ptr.add(at) }, size),
            &values[k].to_le_bytes()[..size],
            "value {k} at {at} size {size}"
        );
    }
    // Nothing else was written: the word at 32..40, then bytes 43, 46, ...
    let written: Vec<usize> = slots
        .iter()
        .flat_map(|&(at, size)| (at as usize)..(at as usize + size as usize))
        .collect();
    for (i, &byte) in storage.iter().enumerate() {
        if !written.contains(&i) {
            assert_eq!(byte, 0xA5, "stray write at {i}");
        }
    }
}

/// GC_LOAD_INDEXED_I with a `ConstInt` index, end to end, for each width and
/// sign: `emit_gc_load_or_indexed` folds the index into a GC_LOAD offset.
#[test]
fn gc_load_indexed_constant_index_reads_memory() {
    let mut storage = [0u8; 64];
    for (i, byte) in storage.iter_mut().enumerate() {
        *byte = 0x80 | i as u8;
    }
    let base_ptr = storage.as_mut_ptr();
    // (index, factor, offset, nsize)
    let cases = [
        (2i64, 8i64, 0i64, 8i64),
        (5, 4, 4, -4),
        (5, 4, 4, 4),
        (3, 2, 1, -2),
        (3, 2, 1, 2),
        (7, 1, 9, -1),
        (7, 1, 9, 1),
    ];
    for (n, &(index, factor, offset, nsize)) in cases.iter().enumerate() {
        let mut backend = fresh_backend();
        let base = OpRef::input_arg_ref(0);
        let loaded = OpRef::int_op(1);
        let ops = vec![
            op_at(
                OpCode::GcLoadIndexedI,
                &[
                    base,
                    OpRef::const_int(index),
                    OpRef::const_int(factor),
                    OpRef::const_int(offset),
                    OpRef::const_int(nsize),
                ],
                loaded,
            ),
            finish_int(loaded, 2),
        ];
        let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
        let token = compile_ops(&mut backend, &inputargs, ops);
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(base_ptr as usize))]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let at = (index * factor + offset) as usize;
        let size = nsize.unsigned_abs() as usize;
        let mut raw = [0u8; 8];
        raw[..size].copy_from_slice(&storage[at..at + size]);
        let unsigned = i64::from_le_bytes(raw);
        let expected = if nsize < 0 {
            let shift = 64 - 8 * size as u32;
            (unsigned << shift) >> shift
        } else {
            unsigned
        };
        assert_eq!(
            backend.get_int_value(&frame, 0),
            expected,
            "case {n}: index {index} factor {factor} offset {offset} nsize {nsize}"
        );
    }
}

extern "C" fn spilled_guard_nop() {}

/// `test_location` on a frame slot: `CMP [rbp+disp], 0`.
///
/// The first results land in callee-saved registers, which a
/// non-collecting call leaves alone. `GUARDED` is past that pool, so
/// `before_call` spills it once every callee-saved register is live.
fn spilled_guard_trace(opcode: OpCode, input: i64) -> (bool, i64, Vec<u8>) {
    const LIVE: u32 = 13;
    const GUARDED: u32 = 7;
    let mut backend = fresh_backend();
    let base = OpRef::input_arg_int(0);
    let mut ops = Vec::new();
    let mut lives = vec![rb(OpRef::int_op(GUARDED))];
    for n in 1..=LIVE {
        ops.push(op_at(
            OpCode::IntAdd,
            &[base, OpRef::const_int(i64::from(n))],
            OpRef::int_op(n),
        ));
        if n != GUARDED {
            lives.push(rb(OpRef::int_op(n)));
        }
    }
    let func = spilled_guard_nop as *const () as usize as i64;
    let call = op_at(OpCode::CallN, &[OpRef::const_int(func)], OpRef::void_op(20));
    call.setdescr(call_descr(vec![], Type::Void, false, 0));
    ops.push(call);
    let guarded = OpRef::int_op(GUARDED);
    let guard = Op::new(opcode, &[rb(guarded)]);
    guard.pos().set(OpRef::void_op(21));
    let types = vec![Type::Int; LIVE as usize];
    guard.setdescr(make_resume_guard_descr_typed(types.clone()));
    guard.set_fail_arg_types(types);
    guard.setfailargs(lives.into());
    ops.push(guard);
    ops.push(finish_int(guarded, 22));
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let code = compiled_bytes(&token).to_vec();
    let frame = backend.execute_token(&token, &[Value::Int(input)]);
    let finished = backend.get_latest_descr(&frame).is_finish();
    let value = backend.get_int_value(&frame, 0);
    (finished, value, code)
}

fn has_cmp_frame_imm0(code: &[u8]) -> bool {
    // `cmp_bi(..., 0)`: `48 83 7D disp8 00` or `48 83 BD disp32 00`.
    code.windows(5)
        .any(|w| w[0] == 0x48 && w[1] == 0x83 && w[2] == 0x7D && w[4] == 0x00)
        || code
            .windows(8)
            .any(|w| w[0] == 0x48 && w[1] == 0x83 && w[2] == 0xBD && w[7] == 0x00)
}

#[test]
fn spilled_guard_true_and_false_read_frame() {
    // The guarded box is input + 7.
    let (pass, value, code) = spilled_guard_trace(OpCode::GuardTrue, 0);
    assert!(pass, "GUARD_TRUE of 7 must finish");
    assert_eq!(value, 7);
    assert!(
        has_cmp_frame_imm0(&code),
        "unfused guard must CMP the spilled box, not TEST a register"
    );

    let (pass, value, _) = spilled_guard_trace(OpCode::GuardTrue, -7);
    assert!(!pass, "GUARD_TRUE of 0 must fail");
    assert_eq!(value, 0);

    let (pass, value, code) = spilled_guard_trace(OpCode::GuardFalse, -7);
    assert!(pass, "GUARD_FALSE of 0 must finish");
    assert_eq!(value, 0);
    assert!(has_cmp_frame_imm0(&code));

    let (pass, value, _) = spilled_guard_trace(OpCode::GuardFalse, 0);
    assert!(!pass, "GUARD_FALSE of 7 must fail");
    assert_eq!(value, 7);
}

/// x86 displacements are the full `fits_in_32bits` range. A 12-bit mask
/// would send 4096 and 0x10000 to offset 0, which stays poisoned.
#[test]
fn gc_load_store_negative_and_large_displacements() {
    let offsets = [-8i64, 4096, 0x10000];
    let values = [0x1111i64, 0x2222, 0x3333];
    let mut storage = vec![0xA5u8; 8 + 0x10000 + 8];
    let base_ptr = unsafe { storage.as_mut_ptr().add(8) };

    let mut backend = fresh_backend();
    let base = OpRef::input_arg_ref(0);
    let mut ops = Vec::new();
    for (n, (&ofs, &value)) in offsets.iter().zip(values.iter()).enumerate() {
        ops.push(gc_store(
            base,
            ofs,
            OpRef::const_int(value),
            8,
            n as u32 + 1,
        ));
    }
    let loaded = [OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)];
    for (n, (&ofs, dst)) in offsets.iter().zip(loaded).enumerate() {
        ops.push(op_at(
            OpCode::GcLoadI,
            &[base, OpRef::const_int(ofs), OpRef::const_int(8)],
            dst,
        ));
        let _ = n;
    }
    let sum = OpRef::int_op(13);
    let total = OpRef::int_op(14);
    ops.push(op_at(OpCode::IntAdd, &[loaded[0], loaded[1]], sum));
    ops.push(op_at(OpCode::IntAdd, &[sum, loaded[2]], total));
    ops.push(finish_int(total, 20));
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Ref(GcRef(base_ptr as usize))]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), values.iter().sum::<i64>());
    for (&ofs, &value) in offsets.iter().zip(values.iter()) {
        let at = unsafe { base_ptr.offset(ofs as isize) };
        assert_eq!(read_i64_unaligned(at), value, "offset {ofs}");
    }
    assert_eq!(
        read_bytes(base_ptr, 8),
        [0xA5u8; 8],
        "offset 0 must stay untouched"
    );
}

fn singlefloat_bits(input: f64) -> i64 {
    let mut backend = fresh_backend();
    let src = OpRef::input_arg_float(0);
    let bits = OpRef::int_op(1);
    let ops = vec![
        op_at(OpCode::CastFloatToSinglefloat, &[src], bits),
        finish_int(bits, 2),
    ];
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Float(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_int_value(&frame, 0)
}

fn singlefloat_roundtrip(input: f64) -> f64 {
    let mut backend = fresh_backend();
    let src = OpRef::input_arg_float(0);
    let bits = OpRef::int_op(1);
    let back = OpRef::float_op(2);
    let ops = vec![
        op_at(OpCode::CastFloatToSinglefloat, &[src], bits),
        op_at(OpCode::CastSinglefloatToFloat, &[bits], back),
        finish_of(back, Type::Float, 3),
    ];
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(&token, &[Value::Float(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_float_value(&frame, 0)
}

/// `genop_cast_float_to_singlefloat` / `genop_cast_singlefloat_to_float`.
/// The int result holds the f32 bits in the low 32. 0.1 rounds; -0.0
/// keeps its sign bit.
#[test]
fn singlefloat_casts_round_trip_bits() {
    let cases = [1.5f64, 0.1, -0.0];
    for input in cases {
        let expect_bits = i64::from((input as f32).to_bits());
        assert_eq!(
            singlefloat_bits(input),
            expect_bits,
            "f32 bits of {input:?}"
        );
        let got = singlefloat_roundtrip(input);
        let expect = f64::from(input as f32);
        assert_eq!(
            got.to_bits(),
            expect.to_bits(),
            "round-trip {input:?} -> {got:?}"
        );
    }
    assert_ne!(
        singlefloat_roundtrip(0.1).to_bits(),
        0.1f64.to_bits(),
        "0.1 must take the f32 rounding, not pass the f64 through"
    );
}

/// `mov r11, imm; call r11` (`_load_scratch` + `CALL_r(X86_64_SCRATCH_REG)`),
/// matching `rx86::mov_ri` into r11.
fn r11_imm_call(addr: i64) -> Vec<u8> {
    let mut bytes = Vec::new();
    if (0..=0xFFFF_FFFF).contains(&addr) {
        // `MOV_riu32` of r11: REX.B, `B8+3`, imm32.
        bytes.extend_from_slice(&[0x41, 0xBB]);
        bytes.extend_from_slice(&(addr as u32).to_le_bytes());
    } else if (i32::MIN as i64..=i32::MAX as i64).contains(&addr) {
        // `MOV_ri32` of r11: REX.W|REX.B, `C7 /0`, imm32.
        bytes.extend_from_slice(&[0x49, 0xC7, 0xC3]);
        bytes.extend_from_slice(&(addr as i32).to_le_bytes());
    } else {
        // `MOV_ri64` of r11: REX.W|REX.B, `B8+3`, imm64.
        bytes.extend_from_slice(&[0x49, 0xBB]);
        bytes.extend_from_slice(&addr.to_le_bytes());
    }
    bytes.extend_from_slice(&[0x41, 0xFF, 0xD3]);
    bytes
}

/// The length of a `push_gcmap` at the start of `code`, or 0: `MOV_bi` of
/// `jf_gcmap`, or `_load_scratch` (`MOV_ri64` / `LEA`) then `MOV_br`.
fn push_gcmap_len(code: &[u8]) -> usize {
    let ofs = majit_backend::jitframe::JF_GCMAP_OFS as u8;
    let store = [0x4C, 0x89, 0x5D, ofs];
    let load = match code {
        [0x48, 0xC7, 0x45, o, ..] if *o == ofs => return 8,
        [0x49, 0xBB, ..] => 10,
        [0x4D, 0x8D, 0x5B, ..] => 4,
        [0x4D, 0x8D, 0x9B, ..] => 7,
        _ => return 0,
    };
    if code.len() >= load + store.len() && code[load..load + store.len()] == store {
        load + store.len()
    } else {
        0
    }
}

/// Thread-local load, the gcmap push `AbstractCallBuilder.emit` places
/// after `prepare_arguments`, then the callee call. `_load_scratch` is
/// `MOV_ri` when r11 is unknown and `LEA r11, [r11+disp]` when a cached
/// scratch value is near the target.
fn threadlocal_then_scratch_call(code: &[u8], addr: i64) -> bool {
    let threadlocal = threadlocal_arg1_load();
    let exact = r11_imm_call(addr);
    (0..code.len()).any(|i| {
        let Some(rest) = code[i..].strip_prefix(threadlocal.as_slice()) else {
            return false;
        };
        let rest = &rest[push_gcmap_len(rest)..];
        rest.starts_with(&exact)
            || matches!(
                rest,
                [0x4D, 0x8D, 0x5B, _, 0x41, 0xFF, 0xD3, ..]
                    | [0x4D, 0x8D, 0x9B, _, _, _, _, 0x41, 0xFF, 0xD3, ..]
            )
    })
}

/// The values r11 holds at each `CALL r11` (`41 FF D3`), following the
/// `_load_scratch` sequence in front of it: `MOV_ri` (`41 BB imm32`,
/// `49 C7 C3 imm32`, `49 BB imm64`) and `LEA r11, [r11+disp8/disp32]`.
/// A `LEA` with no known r11 before it yields no target.
fn r11_call_targets(code: &[u8]) -> Vec<i64> {
    let mut r11: Option<i64> = None;
    let mut targets = Vec::new();
    let mut i = 0;
    while i < code.len() {
        let rest = &code[i..];
        let read32 = |at: usize| i32::from_le_bytes(rest[at..at + 4].try_into().unwrap());
        match rest {
            [0x49, 0xBB, ..] if rest.len() >= 10 => {
                r11 = Some(i64::from_le_bytes(rest[2..10].try_into().unwrap()));
                i += 10;
            }
            [0x41, 0xBB, ..] if rest.len() >= 6 => {
                r11 = Some(i64::from(read32(2) as u32));
                i += 6;
            }
            [0x49, 0xC7, 0xC3, ..] if rest.len() >= 7 => {
                r11 = Some(i64::from(read32(3)));
                i += 7;
            }
            [0x4D, 0x8D, 0x5B, d, ..] => {
                r11 = r11.map(|v| v.wrapping_add(i64::from(*d as i8)));
                i += 4;
            }
            [0x4D, 0x8D, 0x9B, ..] if rest.len() >= 7 => {
                r11 = r11.map(|v| v.wrapping_add(i64::from(read32(3))));
                i += 7;
            }
            [0x41, 0xFF, 0xD3, ..] => {
                targets.extend(r11.take());
                i += 3;
            }
            _ => i += 1,
        }
    }
    targets
}

/// `mov ARGUMENTS_GPR[1], [rsp + THREADLOCAL_OFS]`.
fn threadlocal_arg1_load() -> Vec<u8> {
    use majit_backend_dynasm::x86::arch::THREADLOCAL_OFS;
    use majit_backend_dynasm::x86::callbuilder::ARGUMENTS_GPR;
    let reg = ARGUMENTS_GPR[1].value;
    let modrm = 0x80 | ((reg & 7) << 3) | 0x04;
    let mut bytes = vec![0x48, 0x8B, modrm, 0x24];
    bytes.extend_from_slice(&(THREADLOCAL_OFS as i32).to_le_bytes());
    bytes
}

#[test]
fn call_assembler_frame_goes_straight_into_arg0() {
    // `_call_assembler_emit_call` → `simple_call(addr, [argloc, threadlocal_loc])`.
    // `handle_call_assembler` leaves the callee frame in ecx. On Win64 that
    // register is `ARGUMENTS_GPR[0]`, so the remap is a self-move and the
    // bytes before the callee call are only the thread-local load.
    let mut backend = backend_with_call_assembler_layout();
    let int_callee = compile_identity_loop(&mut backend, Type::Int);
    let int_inputs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = int_inputs[0].opref();
    let int_token = JitCellToken::new(next_token_id());
    let int_call = Op::new(OpCode::CallAssemblerI, &[rb(i0)]);
    int_call.pos().set(OpRef::int_op(1));
    int_call.setdescr(Arc::new(AssemblerLoopDescr {
        token: Arc::clone(&int_callee),
    }) as DescrRef);
    let int_ops = vec![
        OpRc::new(int_call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&int_inputs, &int_ops, &int_token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER_I: {err:?}"));
    let int_frame = backend.execute_token(&int_token, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&int_frame).is_finish());
    assert_eq!(backend.get_int_value(&int_frame, 0), 42);

    let code = compiled_bytes(&int_token);
    let addr =
        majit_backend_dynasm::x86::callbuilder::follow_jump(int_callee.ll_function_addr() as i64);
    let threadlocal = threadlocal_arg1_load();
    assert!(
        threadlocal_then_scratch_call(code, addr),
        "thread-local load must sit directly against the callee call"
    );
    for hop in [[0x48, 0x89, 0xC8], [0x48, 0x89, 0xC1], [0x48, 0x89, 0xC9]] {
        let mut prefixed = hop.to_vec();
        prefixed.extend_from_slice(&threadlocal);
        assert!(
            !code
                .windows(prefixed.len())
                .any(|window| window == prefixed.as_slice()),
            "frame must not hop through rax ahead of the thread-local load: {hop:02x?}"
        );
    }
    let _keep = int_callee;
}

extern "C" fn indirect_const() -> i64 {
    99
}

extern "C" fn indirect_id(x: i64) -> i64 {
    x
}

extern "C" fn indirect_sum4(a: i64, b: i64, c: i64, d: i64) -> i64 {
    a + b + c + d
}

extern "C" fn read_errno_and_stamp(stamp: i64) -> i64 {
    let old = unsafe { *errno_ptr() } as i64;
    unsafe {
        *errno_ptr() = (stamp as i32).wrapping_add(1);
    }
    #[cfg(target_os = "windows")]
    unsafe {
        majit_rlib::rwin32::_SetLastError((stamp as u32).wrapping_add(3));
    }
    old.wrapping_add(stamp)
}

fn indirect_errno_flags() -> i64 {
    let flags = majit_jitcode::rffi::RFFI_READSAVED_ERRNO | majit_jitcode::rffi::RFFI_SAVE_ERRNO;
    #[cfg(target_os = "windows")]
    {
        flags
            | majit_jitcode::rffi::RFFI_READSAVED_LASTERROR
            | majit_jitcode::rffi::RFFI_SAVE_LASTERROR
    }
    #[cfg(not(target_os = "windows"))]
    {
        flags
    }
}

/// `mov r13, rax` / `mov rax, r13`: the per-call park around `write_real_errno`.
fn assert_no_rax_fnloc_park(code: &[u8]) {
    assert!(
        !code.windows(3).any(|window| window == [0x49, 0x89, 0xC5]),
        "mov r13, rax must not park the call target"
    );
    assert!(
        !code.windows(3).any(|window| window == [0x4C, 0x89, 0xE8]),
        "mov rax, r13 must not reload the call target"
    );
}

fn assert_has(code: &[u8], needle: &[u8], what: &str) {
    assert!(
        code.windows(needle.len()).any(|window| window == needle),
        "{what} missing: {needle:02x?}"
    );
}

/// The residual call followed by its result store `mov [rbp+disp8], rax`
/// (`48 89 45`). The pair pins the call site, so a `mov rax, fnloc` /
/// `call rax` shape fails it. Scanning the whole buffer for `FF D0` instead
/// also matches across instruction boundaries: the
/// `_call_header_with_stack_check` slow path, emitted while another test has
/// registered the stack-check addresses, ends in a `jmp rel32` whose last byte
/// is `FF`, and the descr word after the code can start with `D0`.
fn assert_call_site(code: &[u8], call: &[u8], what: &str) {
    let mut site = call.to_vec();
    site.extend_from_slice(&[0x48, 0x89, 0x45]);
    assert_has(code, &site, what);
}

/// `CALL reg` (`FF /2`). `r8`–`r15` take a REX.B prefix.
fn call_reg_bytes(reg: u8) -> Vec<u8> {
    if reg < 8 {
        vec![0xFF, 0xD0 + reg]
    } else {
        vec![0x41, 0xFF, 0xD0 + (reg - 8)]
    }
}

fn arg_gpr_call(index: usize) -> Vec<u8> {
    call_reg_bytes(majit_backend_dynasm::x86::callbuilder::ARGUMENTS_GPR[index].value)
}

/// `CALL [rbp+ofs]` for input 0 (`get_ebp_ofs` / `CALL_b`), the shape a
/// frame fnloc must not keep.
fn call_input0_frame() -> Vec<u8> {
    let offset = majit_backend_dynasm::regalloc::get_ebp_ofs(
        majit_backend::jitframe::FIRST_ITEM_OFFSET as i32,
        0,
    );
    let mut bytes = vec![0xFF];
    if (-128..=127).contains(&offset) {
        bytes.push(0x55);
        bytes.push(offset as u8);
    } else {
        bytes.push(0x95);
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    bytes
}

/// `func_in_reg` is `IntAdd(input, 0)`, which `_consider_lea` leaves in a
/// register. Otherwise the input stays a frame slot.
fn compile_indirect_call(
    func_in_reg: bool,
    release_flags: Option<i64>,
    real_args: &[Operand],
    arg_types: Vec<Type>,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let (add, func) = if func_in_reg {
        let add = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(0))]);
        add.pos().set(OpRef::int_op(1));
        (Some(add), OpRef::int_op(1))
    } else {
        (None, i0)
    };
    let result = OpRef::int_op(2);
    let mut args = Vec::new();
    if let Some(flags) = release_flags {
        args.push(rb(OpRef::const_int(flags)));
    }
    args.push(rb(func));
    args.extend_from_slice(real_args);
    let opcode = if release_flags.is_some() {
        OpCode::CallReleaseGilI
    } else {
        OpCode::CallI
    };
    let call = Op::new(opcode, &args);
    call.pos().set(result);
    call.setdescr(call_descr(arg_types, Type::Int, true, 8));
    let mut ops = Vec::new();
    if let Some(add) = add {
        ops.push(OpRc::new(add));
    }
    ops.push(OpRc::new(call));
    ops.push(OpRc::new(finish_of(result, Type::Int, 3)));
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile indirect call: {err:?}"));
    (backend, token)
}

#[test]
fn residual_call_register_fnloc_is_call_reg() {
    // No integer arguments: `prepare_arguments` places the fnloc in
    // `ARGUMENTS_GPR[0]` (ecx on Win64). The `IntAdd` result already lives
    // there, so the call is `call rcx` with no `mov rax`.
    let (backend, token) = compile_indirect_call(true, None, &[], vec![]);
    let frame = backend.execute_token(&token, &[Value::Int(indirect_const as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 99);
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    assert_call_site(code, &arg_gpr_call(0), "call ARGUMENTS_GPR[0]");

    // One integer argument takes `ARGUMENTS_GPR[0]`, so the fnloc moves to
    // `ARGUMENTS_GPR[1]` and the call is that register.
    let (backend, token) =
        compile_indirect_call(true, None, &[rb(OpRef::const_int(42))], vec![Type::Int]);
    let frame = backend.execute_token(&token, &[Value::Int(indirect_id as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 42);
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    assert_call_site(code, &arg_gpr_call(1), "call ARGUMENTS_GPR[1]");

    // Four integer arguments fill the Win64 GPRs, so the fnloc spills to
    // `[rsp+32]`. SysV still has `r8` free (`ARGUMENTS_GPR[4]`).
    let (backend, token) = compile_indirect_call(
        true,
        None,
        &[
            rb(OpRef::const_int(1)),
            rb(OpRef::const_int(2)),
            rb(OpRef::const_int(3)),
            rb(OpRef::const_int(4)),
        ],
        vec![Type::Int; 4],
    );
    let frame = backend.execute_token(&token, &[Value::Int(indirect_sum4 as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 10);
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    #[cfg(target_os = "windows")]
    let spilled = vec![0xFF, 0x54, 0x24, 0x20];
    #[cfg(not(target_os = "windows"))]
    let spilled = arg_gpr_call(4);
    assert_call_site(code, &spilled, "call spilled or fifth GPR fnloc");
}

#[test]
fn residual_call_frame_fnloc_moves_into_an_argument_register() {
    // The function pointer is an untouched input, so its location is the
    // frame slot. `CallBuilderX86.__init__` appends it to the arguments all
    // the same, and `prepare_arguments` moves it into the next free
    // argument register: the call is that register, never `CALL [rbp+ofs]`.
    let (backend, token) = compile_indirect_call(false, None, &[], vec![]);
    let frame = backend.execute_token(&token, &[Value::Int(indirect_const as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 99);
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    assert_call_site(code, &arg_gpr_call(0), "call ARGUMENTS_GPR[0]");
    assert!(
        !code
            .windows(call_input0_frame().len())
            .any(|window| window == call_input0_frame()),
        "the frame fnloc must not stay CALL [rbp+ofs]"
    );

    let (backend, token) =
        compile_indirect_call(false, None, &[rb(OpRef::const_int(42))], vec![Type::Int]);
    let frame = backend.execute_token(&token, &[Value::Int(indirect_id as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 42);
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    assert_call_site(code, &arg_gpr_call(1), "call ARGUMENTS_GPR[1]");
}

#[test]
fn residual_call_fnloc_survives_errno_and_lasterror() {
    // `write_real_errno` / `read_real_errno`, including Win64
    // `RFFI_READSAVED_LASTERROR` / `RFFI_SAVE_LASTERROR`. The fnloc is not
    // sitting in rax across that sequence.
    let stamp = 100;
    let flags = indirect_errno_flags();
    unsafe {
        *errno_ptr() = 7;
    }
    majit_rlib::rposix::set_saved_errno(41);
    #[cfg(target_os = "windows")]
    majit_rlib::rwin32::SetLastError_saved(41);

    let (backend, token) = compile_indirect_call(
        true,
        Some(flags),
        &[rb(OpRef::const_int(stamp))],
        vec![Type::Int],
    );
    let frame = backend.execute_token(&token, &[Value::Int(read_errno_and_stamp as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 41 + stamp);
    assert_eq!(majit_rlib::rposix::get_saved_errno(), (stamp as i32) + 1);
    #[cfg(target_os = "windows")]
    assert_eq!(
        majit_rlib::rwin32::GetLastError_saved(),
        i64::from(stamp as u32 + 3)
    );
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    assert_has(
        code,
        &arg_gpr_call(1),
        "call ARGUMENTS_GPR[1] after write_real_errno",
    );

    unsafe {
        *errno_ptr() = 7;
    }
    majit_rlib::rposix::set_saved_errno(41);
    #[cfg(target_os = "windows")]
    majit_rlib::rwin32::SetLastError_saved(41);
    let (backend, token) = compile_indirect_call(
        false,
        Some(flags),
        &[rb(OpRef::const_int(stamp))],
        vec![Type::Int],
    );
    let frame = backend.execute_token(&token, &[Value::Int(read_errno_and_stamp as usize as i64)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 41 + stamp);
    assert_eq!(majit_rlib::rposix::get_saved_errno(), (stamp as i32) + 1);
    #[cfg(target_os = "windows")]
    assert_eq!(
        majit_rlib::rwin32::GetLastError_saved(),
        i64::from(stamp as u32 + 3)
    );
    let code = compiled_bytes(&token);
    assert_no_rax_fnloc_park(code);
    assert_has(
        code,
        &arg_gpr_call(1),
        "frame fnloc called through ARGUMENTS_GPR[1] after write_real_errno",
    );
}

fn e9_rel32(from: u64, to: u64) -> [u8; 5] {
    let rel = (to as i64).wrapping_sub(from.wrapping_add(5) as i64) as i32;
    let mut bytes = [0xE9, 0, 0, 0, 0];
    bytes[1..].copy_from_slice(&rel.to_le_bytes());
    bytes
}

fn mov_eax_imm32_ret(value: i32) -> [u8; 6] {
    let mut bytes = [0xB8, 0, 0, 0, 0, 0xC3];
    bytes[1..5].copy_from_slice(&value.to_le_bytes());
    bytes
}

fn alloc_rwx(size: usize) -> *mut u8 {
    #[cfg(target_os = "windows")]
    unsafe {
        unsafe extern "system" {
            fn VirtualAlloc(
                addr: *mut core::ffi::c_void,
                size: usize,
                alloc_type: u32,
                protect: u32,
            ) -> *mut u8;
        }
        let page = VirtualAlloc(core::ptr::null_mut(), size, 0x1000 | 0x2000, 0x40);
        assert!(!page.is_null(), "VirtualAlloc");
        page
    }
    #[cfg(not(target_os = "windows"))]
    unsafe {
        unsafe extern "C" {
            fn mmap(
                addr: *mut core::ffi::c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut u8;
        }
        #[cfg(target_os = "linux")]
        const MAP_ANON: i32 = 0x20;
        #[cfg(target_os = "macos")]
        const MAP_ANON: i32 = 0x1000;
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        const MAP_ANON: i32 = 0x20;
        let page = mmap(core::ptr::null_mut(), size, 1 | 2 | 4, 2 | MAP_ANON, -1, 0);
        assert!(!page.is_null() && page as usize != usize::MAX, "mmap");
        page
    }
}

fn publish_code(page: *mut u8, bytes: &[u8]) {
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), page, bytes.len());
    }
    #[cfg(target_os = "windows")]
    unsafe {
        unsafe extern "system" {
            fn FlushInstructionCache(
                process: *mut core::ffi::c_void,
                addr: *const u8,
                size: usize,
            ) -> i32;
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
        }
        FlushInstructionCache(GetCurrentProcess(), page, bytes.len());
    }
}

fn run_immediate_call(func: i64) -> (i64, Vec<u8>) {
    let (backend, token) = compile_call(
        OpCode::CallI,
        func,
        &[],
        vec![],
        OpRef::int_op(1),
        Type::Int,
        true,
        8,
    );
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    (
        backend.get_int_value(&frame, 0),
        compiled_bytes(&token).to_vec(),
    )
}

#[test]
fn follow_jump_calls_the_final_e9_target() {
    // Two leading `E9 rel32` jumps, then `mov eax, 123; ret`.
    // `follow_jump` calls the body, not either thunk.
    let page = alloc_rwx(4096);
    let base = page as u64;
    let mut bytes = vec![0x90u8; 64];
    bytes[0..5].copy_from_slice(&e9_rel32(base, base + 16));
    bytes[16..21].copy_from_slice(&e9_rel32(base + 16, base + 32));
    bytes[32..38].copy_from_slice(&mov_eax_imm32_ret(123));
    publish_code(page, &bytes);

    let thunk = base as i64;
    let mid = base.wrapping_add(16) as i64;
    let final_addr = base.wrapping_add(32) as i64;
    assert_eq!(
        majit_backend_dynasm::x86::callbuilder::follow_jump(thunk),
        final_addr
    );
    let (got, code) = run_immediate_call(thunk);
    assert_eq!(got, 123);
    let targets = r11_call_targets(&code);
    assert!(
        targets.contains(&final_addr),
        "call followed E9 target missing: {targets:x?}"
    );
    assert!(
        !targets.contains(&thunk),
        "call must not use the first thunk"
    );
    assert!(!targets.contains(&mid), "call must not stop after one E9");
}

#[test]
fn follow_jump_stops_on_a_non_e9_prefix() {
    // `EB 05` skips an `E9` to the wrong body and lands on an `E9` to
    // `mov eax, 123; ret`. Only a leading `E9` is followed, so the
    // immediate is the thunk itself.
    let page = alloc_rwx(4096);
    let base = page as u64;
    let mut bytes = vec![0x90u8; 64];
    bytes[0] = 0xEB;
    bytes[1] = 0x05;
    bytes[2..7].copy_from_slice(&e9_rel32(base + 2, base + 48));
    bytes[7..12].copy_from_slice(&e9_rel32(base + 7, base + 32));
    bytes[32..38].copy_from_slice(&mov_eax_imm32_ret(123));
    bytes[48..54].copy_from_slice(&mov_eax_imm32_ret(456));
    publish_code(page, &bytes);

    let thunk = base as i64;
    let wrong = base.wrapping_add(48) as i64;
    let real = base.wrapping_add(32) as i64;
    assert_eq!(
        majit_backend_dynasm::x86::callbuilder::follow_jump(thunk),
        thunk
    );
    let (got, code) = run_immediate_call(thunk);
    assert_eq!(got, 123);
    let targets = r11_call_targets(&code);
    assert!(
        targets.contains(&thunk),
        "call keeps the non-E9 thunk: {targets:x?}"
    );
    assert!(
        !targets.contains(&wrong),
        "the skipped E9 must not become the call target"
    );
    assert!(
        !targets.contains(&real),
        "the E9 after EB must not become the call target"
    );
}

/// `consider_uint_mul_high` under register pressure: every pair of fourteen
/// live values feeds `UINT_MUL_HIGH`, so some multiplicand already sits in
/// EDX when the result claims it. All fourteen stay live past the multiply.
#[test]
fn uint_mul_high_with_every_operand_register() {
    const N: usize = 14;
    let inputs: Vec<i64> = (0..N as i64)
        .map(|k| (0x9E37_79B9_7F4A_7C15u64 as i64).wrapping_mul(k + 3))
        .collect();
    for p in 0..N {
        for q in 0..N {
            let mut backend = DynasmBackend::new();
            backend.attach_default_test_descrs();
            let token = JitCellToken::new(next_token_id());
            let inputargs: Vec<InputArgRc> = (0..N as u32)
                .map(|k| InputArg::from_type_rc(Type::Int, k))
                .collect();
            let mut ops = Vec::new();
            let adds: Vec<OpRef> = (0..N as u32).map(|k| OpRef::int_op(100 + k)).collect();
            for k in 0..N {
                let add = Op::new(
                    OpCode::IntAdd,
                    &[rb(inputargs[k].opref()), rb(OpRef::const_int(1))],
                );
                add.pos().set(adds[k]);
                ops.push(add);
            }
            let mul = Op::new(OpCode::UintMulHigh, &[rb(adds[p]), rb(adds[q])]);
            mul.pos().set(OpRef::int_op(200));
            ops.push(mul);
            // `r < i64::MIN` never holds, so the guard always leaves with
            // the result and every multiplicand as fail arguments.
            let never = Op::new(
                OpCode::IntLt,
                &[rb(OpRef::int_op(200)), rb(OpRef::const_int(i64::MIN))],
            );
            never.pos().set(OpRef::int_op(201));
            ops.push(never);
            let mut fail_args = vec![rb(OpRef::int_op(200))];
            fail_args.extend(adds.iter().map(|&a| rb(a)));
            let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(201))]);
            guard.pos().set(OpRef::void_op(202));
            guard.set_fail_arg_types(vec![Type::Int; N + 1]);
            guard.setfailargs(fail_args.into());
            ops.push(guard);
            let finish = Op::new(OpCode::Finish, &[]);
            finish.pos().set(OpRef::void_op(203));
            finish.set_fail_arg_types(vec![]);
            finish.setfailargs(vec![].into());
            ops.push(finish);
            let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
            backend
                .compile_loop(&inputargs, &ops, &token)
                .unwrap_or_else(|err| panic!("compile uint_mul_high({p}, {q}): {err:?}"));
            let args: Vec<Value> = inputs.iter().map(|&v| Value::Int(v)).collect();
            let frame = backend.execute_token(&token, &args);
            assert!(!backend.get_latest_descr(&frame).is_finish());
            let a = inputs[p].wrapping_add(1) as u64 as u128;
            let b = inputs[q].wrapping_add(1) as u64 as u128;
            assert_eq!(
                backend.get_int_value(&frame, 0),
                ((a * b) >> 64) as u64 as i64,
                "uint_mul_high(a{p}, a{q})"
            );
            for k in 0..N {
                assert_eq!(
                    backend.get_int_value(&frame, k + 1),
                    inputs[k].wrapping_add(1),
                    "a{k} after uint_mul_high(a{p}, a{q})"
                );
            }
        }
    }
}

/// `consider_increment_debug_counter` / `genop_discard_increment_debug_counter`:
/// the argument is an immediate address and the op is one `INC` of that word.
#[test]
fn increment_debug_counter_increments_the_word() {
    static COUNTER: AtomicI64 = AtomicI64::new(0);
    let addr = &COUNTER as *const AtomicI64 as i64;
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(911);
    let inputargs: Vec<InputArgRc> = vec![];
    let mut ops = Vec::new();
    for pos in 1..=2 {
        let op = Op::new(OpCode::IncrementDebugCounter, &[rb(OpRef::const_int(addr))]);
        op.pos().set(OpRef::void_op(pos));
        ops.push(op);
    }
    ops.push(finish(3));
    let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile increment_debug_counter: {err:?}"));
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COUNTER.load(Ordering::SeqCst), 2);
}

/// x86/regalloc.py `oplist = [RegAlloc.not_implemented_op] * rop._LAST`: an
/// operation the optimizer removes before the backend is refused, not
/// silently dropped.
#[test]
fn removed_operation_reaching_the_backend_is_not_implemented() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(912);
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let r0 = inputargs[0].opref();
    let op = Op::new(OpCode::AssertNotNone, &[rb(r0)]);
    op.pos().set(OpRef::void_op(1));
    let ops: Vec<OpRc> = vec![op, finish(2)].into_iter().map(OpRc::new).collect();
    let payload = catch_unwind(AssertUnwindSafe(|| {
        backend.compile_loop(&inputargs, &ops, &token)
    }))
    .expect_err("ASSERT_NOT_NONE must not compile");
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("not implemented operation: AssertNotNone"),
        "panic message: {msg}"
    );
}

/// x86/regalloc.py `_consider_gc_load` / `consider_gc_store` take the offset
/// through `rm.make_sure_var_in_reg`: a variable offset sits in a register,
/// and a constant one that does not fit 32 bits is left to the location
/// builder (`_fix_static_offset_64_m`), also when the stored value is a
/// 64-bit immediate.
#[test]
fn gc_load_store_take_register_and_wide_offsets() {
    const WIDE: i64 = 1 << 33;
    const BIG: i64 = 0x1234_5678_9abc_def0;
    let mut buf = [0i64; 4];
    let base = buf.as_mut_ptr() as usize;
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
        InputArg::from_type_rc(Type::Int, 2),
        InputArg::from_type_rc(Type::Int, 3),
    ];
    let obj = inputargs[0].opref();
    let shifted = inputargs[1].opref();
    let ofs = inputargs[2].opref();
    let value = inputargs[3].opref();
    let loaded = OpRef::int_op(5);
    let ops = vec![
        op_at(
            OpCode::GcStore,
            &[obj, ofs, value, OpRef::const_int(8)],
            OpRef::void_op(4),
        ),
        op_at(
            OpCode::GcLoadI,
            &[shifted, OpRef::const_int(WIDE + 8), OpRef::const_int(8)],
            loaded,
        ),
        gc_store(shifted, WIDE + 16, OpRef::const_int(BIG), 8, 6),
        finish_int(loaded, 7),
    ];
    let token = compile_ops(&mut backend, &inputargs, ops);
    let frame = backend.execute_token(
        &token,
        &[
            Value::Ref(GcRef(base)),
            Value::Ref(GcRef(base.wrapping_sub(WIDE as usize))),
            Value::Int(8),
            Value::Int(77),
        ],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 77);
    assert_eq!(buf, [0, 77, BIG, 0]);
}
