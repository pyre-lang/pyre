//! The deadframe accessors and the raw exec path must report the same slot
//! values.
//!
//! `execute_token_ints_raw` decodes `rd_locs` a second time, on its own copy of
//! the returned frame's slots, without ever building a deadframe. That makes
//! the two decoders an oracle for each other: for every fail-arg index `i`, the
//! `typed_outputs[i]` the raw path reports and the value
//! `get_{int,float,ref}_value(deadframe, i)` reads back must be the same value.
//! `llmodel.py get_int_value` is the single logical accessor both are
//! standing in for, so a disagreement means one of them is answering in the
//! wrong index space.
//!
//! The corpus below has to contain at least one guard whose `rd_locs` is NOT
//! the identity permutation, and `deadframe_reads_agree_with_raw_exec` asserts
//! that it does. On an identity-only corpus the logical fail-arg index and the
//! jitframe slot index coincide, so every reader agrees no matter which of the
//! two it is actually indexing by — the comparison would pass with the
//! `rd_locs` decode removed entirely, and would be no control at all.
mod common;
use majit_backend::{Backend, JitCellToken};
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::{InputArg, Op, OpCode, OpRc, OpRef, Type, Value, make_loop_target_descr};

use majit_backend_dynasm::runner::DynasmBackend;

/// One compiled trace plus the arguments to enter it with.
struct Case {
    name: &'static str,
    backend: DynasmBackend,
    token: JitCellToken,
    args: Vec<Value>,
}

/// The same arguments in the `i64` spelling `execute_token_ints_raw` takes.
/// Both entry points stage arguments identically — one `set_int_value` per
/// argument at `input_slot(i)`, floats as their bit pattern — so this is the
/// same entry, not an approximation of it.
fn raw_args(args: &[Value]) -> Vec<i64> {
    args.iter()
        .map(|v| match v {
            Value::Int(i) => *i,
            Value::Float(f) => f.to_bits() as i64,
            Value::Ref(r) => r.0 as i64,
            Value::Void => 0,
        })
        .collect()
}

/// Run `case` down both paths and assert every fail-arg index reads the same.
/// Returns the exit's `rd_locs` so the caller can check what the corpus covers.
fn check_case(case: &Case) -> Vec<u16> {
    let name = case.name;
    let raw = case
        .backend
        .execute_token_ints_raw(&case.token, &raw_args(&case.args));
    let frame = case.backend.execute_token(&case.token, &case.args);
    let descr = case.backend.get_latest_descr(&frame);

    // Two separate runs of the same deterministic trace on the same arguments.
    // If they left through different exits the value comparison below would be
    // meaningless, so establish that first.
    assert_eq!(
        (raw.trace_id, raw.fail_index, raw.is_finish),
        (
            descr.trace_id(),
            descr.fail_index_per_trace(),
            descr.is_finish()
        ),
        "{name}: the raw run and the deadframe run left through different exits"
    );

    let rd_locs = descr.rd_locs().to_vec();
    for (i, expected) in raw.typed_outputs.iter().enumerate() {
        match expected {
            Value::Int(v) => assert_eq!(
                case.backend.get_int_value(&frame, i),
                *v,
                "{name}: fail arg {i} (rd_locs={rd_locs:?})"
            ),
            Value::Float(v) => assert_eq!(
                case.backend.get_float_value(&frame, i).to_bits(),
                v.to_bits(),
                "{name}: fail arg {i} (rd_locs={rd_locs:?})"
            ),
            Value::Ref(v) => assert_eq!(
                case.backend.get_ref_value(&frame, i),
                *v,
                "{name}: fail arg {i} (rd_locs={rd_locs:?})"
            ),
            // The raw path replaces a `Type::Void` fail arg's value with
            // `Value::Void` unconditionally (`resume.py:411-417` reconstructs
            // that slot from the resume snapshot instead), so there is no value
            // to compare unless the descr also maps it nowhere — and that case
            // is the boundary reading asserted below.
            Value::Void => {
                if rd_locs.get(i) == Some(&0xFFFF) {
                    assert_eq!(
                        case.backend.get_int_value(&frame, i),
                        0,
                        "{name}: unmapped fail arg {i} must read 0"
                    );
                }
            }
        }
    }

    // An index past the frame's own slot count is answered, not trapped: it
    // reads 0, the same as an unmapped `rd_locs` entry.
    assert_eq!(
        case.backend.get_int_value(&frame, 1_000_000),
        0,
        "{name}: an index past the frame's slots must read 0"
    );
    assert_eq!(
        case.backend.get_float_value(&frame, 1_000_000).to_bits(),
        0,
        "{name}: an index past the frame's slots must read 0.0"
    );
    assert!(
        case.backend.get_ref_value(&frame, 1_000_000).is_null(),
        "{name}: an index past the frame's slots must read NULL"
    );

    rd_locs
}

/// `finish(i0 + 1)` — one int fail arg, no guard.
fn finish_one_int() -> Case {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    let add_op = Op::new(
        OpCode::IntAdd,
        &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
    );
    add_op.pos().set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops: Vec<OpRc> = vec![add_op, finish_op].into_iter().map(OpRc::new).collect();
    common::rebind_trace_arg_boxes(&inputargs, &ops);
    backend
        .compile_loop(&inputargs, &ops, &token)
        .expect("compile_loop");

    Case {
        name: "finish_one_int",
        backend,
        token,
        args: vec![Value::Int(42)],
    }
}

/// A counting loop whose `guard_true` carries one int fail arg.
fn guard_one_int() -> Case {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(2);

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

    let ops: Vec<OpRc> = vec![label_op, add_op, lt_op, guard_op, jump_op]
        .into_iter()
        .map(OpRc::new)
        .collect();
    common::rebind_trace_arg_boxes(&inputargs, &ops);
    backend
        .compile_loop(&inputargs, &ops, &token)
        .expect("compile_loop");

    Case {
        name: "guard_one_int",
        backend,
        token,
        args: vec![Value::Int(0)],
    }
}

/// A loop carrying a float accumulator beside its int counter, guarded on the
/// counter. Both fail args come out SPILLED — `faillocs` reads
/// `[Frame(position: 0), Frame(position: 1)]` — and a spilled arg encodes as
/// `position + JITFRAME_FIXED_SIZE` (`aarch64/assembler.rs:4029`), so this
/// guard's `rd_locs` is `[24, 25]`: the non-identity permutation the corpus
/// needs. An accessor that skipped the `rd_locs` decode would read frame slots
/// 0 and 1 here and get neither value.
///
/// The 24 is `aarch64/arch.rs`'s `JITFRAME_FIXED_SIZE`, which counts SLOTS.
/// A second constant of the same name in `majit-backend`'s `jitframe.rs` is
/// `size_of::<JitFrame>()` and counts BYTES (56); the encoder uses the arch
/// one, and the two are not interchangeable.
fn guard_float_and_int() -> Case {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(3);

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

    let ops: Vec<OpRc> = vec![
        label_op, lt_op, guard_op, cast_op, mul_op, add_op, inc_op, jump_op,
    ]
    .into_iter()
    .map(OpRc::new)
    .collect();
    common::rebind_trace_arg_boxes(&inputargs, &ops);
    backend
        .compile_loop(&inputargs, &ops, &token)
        .expect("compile_loop");

    Case {
        name: "guard_float_and_int",
        backend,
        token,
        args: vec![Value::Float(0.0), Value::Int(0)],
    }
}

/// A guard whose middle fail arg is an absent slot. `locs_for_fail_args` gives
/// an absent operand no location at all, and the encoder writes 0xFFFF for it
/// (`llsupport/assembler.py store_info_on_descr`), which is the
/// unmapped entry `_decode_pos` has to answer 0 for.
fn guard_with_hole() -> Case {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(4);

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

    let dbl_op = Op::new(
        OpCode::IntMul,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(2))],
    );
    dbl_op.pos().set(OpRef::int_op(2));

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(5))],
    );
    lt_op.pos().set(OpRef::int_op(3));

    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(3))]);
    guard_op.pos().set(OpRef::void_op(4));
    guard_op.set_fail_arg_types(vec![Type::Int, Type::Void, Type::Int]);
    guard_op.setfailargs(vec![rb(OpRef::int_op(1)), rb(OpRef::None), rb(OpRef::int_op(2))].into());

    let jump_op = Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]);
    jump_op.pos().set(OpRef::void_op(5));
    jump_op.setdescr(loop_descr);

    let ops: Vec<OpRc> = vec![label_op, add_op, dbl_op, lt_op, guard_op, jump_op]
        .into_iter()
        .map(OpRc::new)
        .collect();
    common::rebind_trace_arg_boxes(&inputargs, &ops);
    backend
        .compile_loop(&inputargs, &ops, &token)
        .expect("compile_loop");

    Case {
        name: "guard_with_hole",
        backend,
        token,
        args: vec![Value::Int(0)],
    }
}

#[test]
fn deadframe_reads_agree_with_raw_exec() {
    let cases = [
        finish_one_int(),
        guard_one_int(),
        guard_float_and_int(),
        guard_with_hole(),
    ];

    let mut saw_non_identity = false;
    for case in &cases {
        let rd_locs = check_case(case);
        let identity = rd_locs
            .iter()
            .enumerate()
            .all(|(i, &slot)| slot as usize == i);
        eprintln!("{}: rd_locs={rd_locs:?} identity={identity}", case.name);
        if !rd_locs.is_empty() && !identity {
            saw_non_identity = true;
        }
    }

    assert!(
        saw_non_identity,
        "no exit in the corpus mapped a fail arg to a slot other than its own \
         index, so nothing here distinguishes a reader that decodes `rd_locs` \
         from one that indexes the frame directly. Add a trace whose guard \
         spills or reorders its fail args before trusting this test."
    );
}

#[test]
fn unmapped_fail_arg_reads_zero() {
    let case = guard_with_hole();
    let frame = case.backend.execute_token(&case.token, &case.args);
    let rd_locs = case.backend.get_latest_descr(&frame).rd_locs().to_vec();

    let hole = rd_locs
        .iter()
        .position(|&slot| slot == 0xFFFF)
        .unwrap_or_else(|| panic!("no 0xFFFF entry in rd_locs={rd_locs:?}"));

    assert_eq!(
        case.backend.get_int_value(&frame, hole),
        0,
        "fail arg {hole} is unmapped (rd_locs={rd_locs:?}) and must read 0"
    );
    // The mapped neighbours still read their own values, so the zero above is
    // the sentinel being honoured and not the whole descr reading as empty.
    assert_eq!(
        case.backend.get_int_value(&frame, 0),
        5,
        "mapped fail arg 0 (rd_locs={rd_locs:?})"
    );
    assert_eq!(
        case.backend.get_int_value(&frame, 2),
        10,
        "mapped fail arg 2 (rd_locs={rd_locs:?})"
    );
}
