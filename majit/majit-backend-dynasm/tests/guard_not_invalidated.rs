//! `GUARD_NOT_INVALIDATED` is the one guard that tests nothing at run time.
//!
//! `x86/assembler.py genop_guard_guard_not_invalidated` records the guard's
//! position and emits no test, and `x86/runner.py invalidate_loop` /
//! `aarch64/runner.py invalidate_loop` later write a branch to the guard's recovery stub over
//! the recorded position.  So the contract these tests pin is behavioural, not
//! structural: the same entry point runs to completion before
//! `invalidate_loop`, and takes the guard after it, with nothing in the trace
//! changed in between.
//!
//! `aarch64/runner.py invalidate_loop`'s docstring is the second half of it: "afterwards, if
//! one such guard fails often enough, it has a bridge attached to it; it is
//! possible then to re-call invalidate_loop() on the same looptoken, which must
//! invalidate all newer GUARD_NOT_INVALIDATED, but not the old one that already
//! has a bridge attached to it".  The list is emptied by the walk, so a second
//! call has nothing to write.

use majit_backend::{Backend, JitCellToken, make_resume_guard_descr_typed};
#[cfg(target_arch = "x86_64")]
use majit_ir::make_loop_target_descr;
use majit_ir::{InputArg, Op, OpCode, OpRc, OpRef, Type, Value};

use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;

/// `guard_not_invalidated() [i0]` / `i1 = int_add(i0, 1)` / `finish(i1)`.
///
/// The guard carries `i0` as its one fail argument and the finish carries `i1`,
/// so the two exits are distinguishable by value alone: 42 means the guard was
/// taken, 43 means it was not.
fn compile_guarded_add(backend: &mut DynasmBackend, token: &JitCellToken) -> majit_ir::DescrRef {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let guard_descr = make_resume_guard_descr_typed(vec![Type::Int]);
    let guard_op = Op::new(OpCode::GuardNotInvalidated, &[]);
    guard_op.pos().set(OpRef::void_op(0));
    guard_op.set_fail_arg_types(vec![Type::Int]);
    guard_op.setfailargs(vec![rb(i0)].into());
    guard_op.setdescr(guard_descr.clone());

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(1))]);
    add_op.pos().set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops_rc: Vec<OpRc> = vec![OpRc::new(guard_op), OpRc::new(add_op), OpRc::new(finish_op)];
    let result = backend.compile_loop(&inputargs, &ops_rc, token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
    guard_descr
}

#[test]
fn the_guard_is_inert_until_invalidate_loop_writes_the_branch() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);
    compile_guarded_add(&mut backend, &token);

    // Before: the guard site holds the placeholder the emitter left, so control
    // falls through it into the trace body.
    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(
        descr.is_finish(),
        "an un-invalidated GUARD_NOT_INVALIDATED must not be reachable as an exit"
    );
    assert_eq!(backend.get_int_value(&frame, 0), 43);

    // `quasiimmut.py QuasiImmut.invalidate`: `looptoken.invalidated = True;
    // cpu.invalidate_loop(looptoken)`.
    backend.invalidate_loop(&token);

    // After: the very same entry point takes the guard.  Nothing about the
    // trace changed — only the bytes at the recorded position.
    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(
        !descr.is_finish(),
        "an invalidated GUARD_NOT_INVALIDATED must exit through its recovery stub"
    );
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "the deadframe must hold the guard's fail argument, not the finish's"
    );
}

/// `runner.py invalidate_loop`'s trailing
/// `looptoken.compiled_loop_token.invalidate_positions = []` —
/// the walk consumes the list, so calling it again writes nothing.  The already
/// written branch stays written.
#[test]
fn a_second_invalidation_has_nothing_left_to_write() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(2);
    compile_guarded_add(&mut backend, &token);

    backend.invalidate_loop(&token);
    backend.invalidate_loop(&token);

    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
}

/// A trace with two of them.  Both positions are recorded and both are written,
/// and the first one reached is the one that exits — the guards are ordinary
/// trace positions, not a single per-trace switch.
#[test]
fn every_recorded_position_in_a_trace_is_written() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(3);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let first = Op::new(OpCode::GuardNotInvalidated, &[]);
    first.pos().set(OpRef::void_op(0));
    first.set_fail_arg_types(vec![Type::Int]);
    first.setfailargs(vec![rb(i0)].into());
    first.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(1))]);
    add_op.pos().set(OpRef::int_op(1));

    let second = Op::new(OpCode::GuardNotInvalidated, &[]);
    second.pos().set(OpRef::void_op(2));
    second.set_fail_arg_types(vec![Type::Int]);
    second.setfailargs(vec![rb(OpRef::int_op(1))].into());
    second.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos().set(OpRef::void_op(3));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops_rc: Vec<OpRc> = vec![
        OpRc::new(first),
        OpRc::new(add_op),
        OpRc::new(second),
        OpRc::new(finish_op),
    ];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    assert!(
        backend
            .get_latest_descr(&backend.execute_token(&token, &[Value::Int(42)]))
            .is_finish()
    );

    backend.invalidate_loop(&token);

    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "the first guard is the one reached, so its fail argument is the one saved"
    );
}

/// `genop_guard_guard_not_invalidated` emits zero bytes. `invalidate_loop`
/// writes `JMP rel32` over the five bytes that already followed the site.
#[cfg(target_arch = "x86_64")]
#[test]
fn the_x86_site_emits_nothing_until_invalidated() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(4);
    let guard_descr = compile_guarded_add(&mut backend, &token);

    let entry = token.ll_function_addr();
    assert_ne!(entry, 0);
    let before = unsafe { std::slice::from_raw_parts(entry as *const u8, 256).to_vec() };

    backend.invalidate_loop(&token);
    let after = unsafe { std::slice::from_raw_parts(entry as *const u8, 256) };
    let site = (0..before.len())
        .find(|&i| before[i] != after[i])
        .expect("invalidate_loop writes the branch");
    assert!(site + 5 <= before.len());
    // Only the site itself is checked: an address immediate elsewhere in the
    // window can contain an 0xE9 byte.
    assert_ne!(
        before[site], 0xE9,
        "the unpatched trace has no JMP rel32 at the guard"
    );
    assert_eq!(after[site], 0xE9, "invalidate_loop writes JMP rel32");
    assert_eq!(
        &before[site + 5..],
        &after[site + 5..],
        "only the five bytes at the site change"
    );
    // `patch_pending_failure_recoveries`: `adr_jump_offset` is the raw
    // address of the JMP's 4-byte target field, `pos + 1`.
    let adr_jump_offset = guard_descr
        .as_fail_descr()
        .expect("guard descr")
        .adr_jump_offset();
    assert_eq!(
        adr_jump_offset,
        entry + site + 1,
        "adr_jump_offset names the JMP rel32 field"
    );
    let rel = i32::from_le_bytes(after[site + 1..site + 5].try_into().unwrap());
    let target = (entry + site + 5) as i64 + rel as i64;
    assert!(
        target as usize > entry + site + 5,
        "JMP rel32 targets the recovery stub after the body"
    );
}

/// `consider_guard_not_invalidated` /
/// `ensure_next_label_is_at_least_at_position(n + 5)`: a label bound on the
/// next op is pushed at least five bytes past the guard.
#[cfg(target_arch = "x86_64")]
#[test]
fn a_label_immediately_after_the_guard_is_at_least_five_bytes_later() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(5);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let loop_descr = make_loop_target_descr(token.number() as u64, false);

    let guard_descr = make_resume_guard_descr_typed(vec![Type::Int]);
    let guard_op = Op::new(OpCode::GuardNotInvalidated, &[]);
    guard_op.pos().set(OpRef::void_op(0));
    guard_op.set_fail_arg_types(vec![Type::Int]);
    guard_op.setfailargs(vec![rb(i0)].into());
    guard_op.setdescr(guard_descr.clone());

    let label_op = Op::new(OpCode::Label, &[rb(i0)]);
    label_op.pos().set(OpRef::void_op(1));
    label_op.setdescr(loop_descr.clone());

    let finish_op = Op::new(OpCode::Finish, &[rb(i0)]);
    finish_op.pos().set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(i0)].into());

    let ops_rc: Vec<OpRc> = vec![
        OpRc::new(guard_op),
        OpRc::new(label_op),
        OpRc::new(finish_op),
    ];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let entry = token.ll_function_addr();
    let before = unsafe { std::slice::from_raw_parts(entry as *const u8, 256).to_vec() };
    backend.invalidate_loop(&token);
    let after = unsafe { std::slice::from_raw_parts(entry as *const u8, 256) };
    let site = (0..before.len())
        .find(|&i| before[i] != after[i])
        .expect("invalidate_loop writes the branch");
    let label = loop_descr
        .as_loop_target_descr()
        .expect("label descr")
        .ll_loop_code();
    assert!(
        label >= entry + site + 5,
        "label {label:#x} must be at least 5 bytes after the guard at {:#x}",
        entry + site
    );
}
