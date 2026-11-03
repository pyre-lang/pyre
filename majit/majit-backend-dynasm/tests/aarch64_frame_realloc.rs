//! Runtime check that aarch64 `_check_frame_depth` `BL`s the shared
//! `_frame_realloc_slowpath` and that a deeper bridge reallocates the
//! in-flight JITFRAME.
#![cfg(target_arch = "aarch64")]

mod common;

use std::sync::atomic::Ordering;

use majit_backend::{Backend, JitCellToken};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::{ForwardingHost, bound_operand_from_opref as rb};
use majit_ir::operand::Operand;
use majit_ir::{InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type, Value};

const BRIDGE_LIVE_INTS: u32 = 48;

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

fn frame_info_words(token: &JitCellToken) -> (isize, isize) {
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let info = clt.frame_info.lock();
    (info.depth(), info.size())
}

fn store_frame_info(token: &JitCellToken, depth: isize, size: isize) {
    let clt = token.compiled_loop_token().expect("compiled loop token");
    let info = clt.frame_info.lock();
    info.jfi_frame_depth.store(depth, Ordering::Relaxed);
    info.jfi_frame_size.store(size, Ordering::Relaxed);
}

fn compile_guarded_loop() -> (DynasmBackend, JitCellToken, Vec<InputArgRc>) {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(0xA64F_0001);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let eq = Op::new(OpCode::IntEq, &[rb(i0), rb(OpRef::const_int(1))]);
    eq.pos().set(OpRef::int_op(1));
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(1))]);
    guard.pos().set(OpRef::void_op(2));
    guard.set_fail_arg_types(vec![Type::Int]);
    guard.setfailargs(vec![rb(i0)].into());
    let finish = finish_many(&[rb(i0)], 3);
    let ops: Vec<OpRc> = vec![eq, guard, finish].into_iter().map(OpRc::new).collect();
    common::rebind_trace_arg_boxes(&inputargs, &ops);
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile guarded loop: {err:?}"));
    (backend, token, inputargs)
}

fn bridge_with_live_ints(i0: OpRef, count: u32) -> Vec<Op> {
    let (adds, finished) = live_int_adds(i0, count);
    // `genop_finish` writes only arglocs[0] and leaves `rd_locs` empty, so
    // `get_int_value` of later finish args is not the add results. A failing
    // guard records every failarg in `rd_locs`.
    let eq = Op::new(OpCode::IntEq, &[rb(i0), rb(OpRef::const_int(1))]);
    eq.pos().set(OpRef::int_op(200));
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(200))]);
    guard.pos().set(OpRef::void_op(201));
    guard.set_fail_arg_types(vec![Type::Int; finished.len()]);
    guard.setfailargs(finished.into());
    let finish = finish_many(&[rb(i0)], 202);
    let mut ops = Vec::with_capacity(adds.len() + 3);
    ops.extend(adds);
    ops.push(eq);
    ops.push(guard);
    ops.push(finish);
    ops
}

/// Loop with a small frame, bridge with many live ints. Shrinking
/// `jfi_frame_depth` back to the loop's depth forces `_check_frame_depth`
/// onto `_frame_realloc_slowpath`; the finish values survive the copy.
#[test]
fn bridge_with_a_larger_frame_reallocates_and_returns_its_exit_values() {
    let (mut backend, token, inputargs) = compile_guarded_loop();
    let (loop_depth, loop_size) = frame_info_words(&token);
    let i0 = inputargs[0].opref();

    let failed = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(!backend.get_latest_descr(&failed).is_finish());
    assert_eq!(backend.get_int_value(&failed, 0), 7);
    let guard_descr = backend.get_latest_descr_arc(&failed);
    assert_ne!(guard_descr.as_fail_descr().unwrap().adr_jump_offset(), 0);
    drop(failed);

    let bridge_ops: Vec<OpRc> = bridge_with_live_ints(i0, BRIDGE_LIVE_INTS)
        .into_iter()
        .map(OpRc::new)
        .collect();
    for ia in &inputargs {
        ia.clear_forwarded();
    }
    common::rebind_trace_arg_boxes(&inputargs, &bridge_ops);
    backend
        .compile_bridge(
            guard_descr.as_fail_descr().unwrap(),
            &inputargs,
            &bridge_ops,
            &token,
        )
        .unwrap_or_else(|err| panic!("compile bridge: {err:?}"));

    let (bridge_depth, _) = frame_info_words(&token);
    assert!(
        bridge_depth > loop_depth,
        "bridge frame {bridge_depth} must exceed the loop frame {loop_depth}"
    );

    // `execute_token` sizes from `jfi_frame_depth`. Put the loop's depth
    // back so the bridge's `b.ge` is not taken and `realloc_frame` runs.
    store_frame_info(&token, loop_depth, loop_size);
    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    for n in 0..BRIDGE_LIVE_INTS {
        assert_eq!(
            backend.get_int_value(&frame, 1 + n as usize),
            7 + i64::from(n) + 1,
            "bridge failarg {}",
            1 + n
        );
    }
    let (after_depth, _) = frame_info_words(&token);
    assert_eq!(
        after_depth, bridge_depth,
        "realloc_frame must widen jfi_frame_depth from {loop_depth} to {bridge_depth}"
    );
}
