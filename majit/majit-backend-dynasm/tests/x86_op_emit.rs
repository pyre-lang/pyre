//! Execution checks for the x86 emitters ported onto the upstream sequences:
//! `genop_int_and`, `genop_int_force_ge_zero`, `genop_float_neg`,
//! `genop_float_abs`, `_cmp_guard_gc_type`, `genop_guard_guard_is_object`,
//! `genop_guard_guard_subclass`, `_binaryop`, `_cmpop_float`,
//! `_store_force_index`, `store_force_descr`, `genop_guard_guard_no_exception`,
//! `cond_call` / `CondCallSlowPath`, `generate_quick_failure`,
//! `_build_failure_recovery`, `malloc_cond`,
//! `malloc_cond_varsize_frame`, `malloc_cond_varsize`,
//! `_binaryop_or_lea`, `flush_cc`, `addr_add_const`, `genop_finish`,
//! `closing_jump`.

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
    let token = JitCellToken::new(token_id);
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
    let token = JitCellToken::new(token_id);
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
    let token = JitCellToken::new(token_id);
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

fn next_token_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0x00D4_0001);
    NEXT.fetch_add(1, Ordering::Relaxed)
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

static COND_CALL_EXTRA0: AtomicI64 = AtomicI64::new(0);
static COND_CALL_EXTRA1: AtomicI64 = AtomicI64::new(0);
static COND_CALL_EXTRA2: AtomicI64 = AtomicI64::new(0);
static COND_CALL_EXTRA_HITS: AtomicUsize = AtomicUsize::new(0);
static COND_CALL_VALUE_EXTRA_HITS: AtomicUsize = AtomicUsize::new(0);
static EAX_LIVE_HITS: AtomicUsize = AtomicUsize::new(0);

fn reset_cond_call_extras() {
    COND_CALL_EXTRA0.store(0, Ordering::SeqCst);
    COND_CALL_EXTRA1.store(0, Ordering::SeqCst);
    COND_CALL_EXTRA2.store(0, Ordering::SeqCst);
    COND_CALL_EXTRA_HITS.store(0, Ordering::SeqCst);
    COND_CALL_VALUE_EXTRA_HITS.store(0, Ordering::SeqCst);
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

fn assert_recorded_extras(expected: [i64; 3]) {
    assert_eq!(COND_CALL_EXTRA0.load(Ordering::SeqCst), expected[0]);
    assert_eq!(COND_CALL_EXTRA1.load(Ordering::SeqCst), expected[1]);
    assert_eq!(COND_CALL_EXTRA2.load(Ordering::SeqCst), expected[2]);
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
        e0 = sym COND_CALL_EXTRA0,
        e1 = sym COND_CALL_EXTRA1,
        e2 = sym COND_CALL_EXTRA2,
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
        e0 = sym COND_CALL_EXTRA0,
        e1 = sym COND_CALL_EXTRA1,
        e2 = sym COND_CALL_EXTRA2,
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

    reset_cond_call_extras();
    let (backend, token) = compile_cond_call_three_extras(0, func);
    let frame = backend.execute_token(&token, &[Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_EXTRA_HITS.load(Ordering::SeqCst), 0);
    assert_live_adds(&backend, &frame, base, 0);

    reset_cond_call_extras();
    let (backend, token) = compile_cond_call_three_extras(1, func);
    let frame = backend.execute_token(&token, &[Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_EXTRA_HITS.load(Ordering::SeqCst), 1);
    assert_recorded_extras(expected);
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

    reset_cond_call_extras();
    let (backend, token) = compile_cond_call_value_three_extras(func);
    let frame = backend.execute_token(&token, &[Value::Int(live_value), Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_VALUE_EXTRA_HITS.load(Ordering::SeqCst), 0);
    assert_eq!(backend.get_int_value(&frame, 0), live_value);
    assert_live_adds(&backend, &frame, base, 1);

    reset_cond_call_extras();
    let (backend, token) = compile_cond_call_value_three_extras(func);
    let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(base)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(COND_CALL_VALUE_EXTRA_HITS.load(Ordering::SeqCst), 1);
    assert_recorded_extras(expected);
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

/// `generate_cond_call_slowpath` loads the helper with `mov r11, imm64`
/// and `call r11`. `push_gcmap` forgets the scratch cache first, so
/// `load_scratch` is that `movabs`, not an `lea`.
fn cond_call_slowpath_index(code: &[u8], func: u64) -> usize {
    let needle = func.to_le_bytes();
    let at = code
        .windows(8)
        .position(|window| window == needle)
        .unwrap_or_else(|| panic!("cond_call func immediate is missing"));
    let tail = &code[at..];
    let mut i = 0;
    while i + 13 <= tail.len() {
        if tail[i] == 0x49
            && tail[i + 1] == 0xBB
            && tail[i + 10] == 0x41
            && tail[i + 11] == 0xFF
            && tail[i + 12] == 0xD3
        {
            let helper = u64::from_le_bytes(tail[i + 2..i + 10].try_into().unwrap());
            return classify_cond_call_helper(helper);
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
    fn loop_token_number(&self) -> u64 {
        self.token.number
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
            &[],
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

/// `[rsp+8]` is a prologue spill (`_call_header` has no `PASS_ON_MY_FRAME`
/// scratch). The realloc body parks that word; these callee-saves come
/// back only when the park is restored before the footer.
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
    // slots: saved rbx, saved [rsp+8] reg, saved r12, got rbx, got [rsp+8], got r12.
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
            "call r14",
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
        assert_eq!(slots[4], 0x2222_2222_2222_2222, "rsi at [rsp+8]");
        assert_eq!(slots[5], 0x3333_3333_3333_3333, "r12");
    }
    #[cfg(not(target_os = "windows"))]
    assert_eq!(slots[4], 0x3333_3333_3333_3333, "r12 at [rsp+8]");
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

fn shadow_words() -> (usize, usize, usize) {
    let top = shadow_top();
    if top == 0 {
        return (0, 0, 0);
    }
    unsafe { (top, *(top as *const usize), *((top + 8) as *const usize)) }
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

    let _lock = lock_exception_cells();
    let _clear = ClearExc;
    STACK_END.store(0, Ordering::Release);
    STACK_LENGTH.store(0, Ordering::Release);
    STACK_MODE.with(|mode| mode.set(2));
    STACK_CALLS.with(|calls| calls.set(0));
    let before = shadow_words();
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert_propagate(&backend, &frame);
    assert!(
        !majit_backend_dynasm::jit_exc_is_pending(),
        "the overflow footer clears both exception cells"
    );
    assert_eq!(
        shadow_words(),
        before,
        "overflow runs before gen_shadowstack_header and must not pop"
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
        let got = compile_and_run_int(op, Some(rhs), input, next_token_id());
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
    let unresolved = make_loop_target_descr(next_token_id(), false);
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

    let descr = make_loop_target_descr(next_token_id(), false);
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
