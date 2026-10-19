//! End-to-end live-wire check: `#[elidable_cannot_raise]` attachment
//! → policy byte 19 → `TraceCtx::call_typed_with_effect_pure` → trace
//! is patched to `CallPureI` (or fully cut to `Const` when every
//! argument is a constant).
//!
//! `pyjitpl.py MIFrame.execute_varargs(opnum, argboxes, descr, exc=False, pure=True)`
//! + `pyjitpl.py record_result_of_call_pure` — the live wire
//!   must not break when observed from outside the metainterp crate.
//!
//! Infrastructure unit tests already exist at:
//!  - `majit-metainterp/src/pyjitpl.rs`
//!    `record_result_of_call_pure_all_const_args_truncates_and_returns_const`
//!  - `majit-metainterp/src/optimizeopt/optimizer.rs`
//!    `test_call_pure_results`
//!  - `majit-metainterp/src/optimizeopt/pure.rs` `test_extra_call_pure` /
//!    `test_known_result_call_pure_lookup`
//!  - `majit-macros/tests/macros.rs` `test_elidable_cannot_raise_function`
//!    `#[elidable_cannot_raise]` policy byte
//!
//! This file verifies that the unit tests above keep working when
//! bundled into the same compilation unit.

use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex, OpCode, Type, Value};
use majit_macros::elidable_cannot_raise;
use majit_metainterp::{BackEdgeAction, MetaInterp};

/// Canary helper.  `rlib/jit.py @jit.elidable` parity.
///
/// Deterministic, no side effects, no raise → `EF_ELIDABLE_CANNOT_RAISE`
/// (`call.py getcalldescr` else branch, policy byte 19).
#[elidable_cannot_raise]
fn elidable_canary_mul(x: i64, y: i64) -> i64 {
    x.wrapping_mul(y) ^ 0x5eed
}

#[test]
fn elidable_canary_macro_advertises_extern_c_trampoline() {
    // `call_policy_byte.rs`'s `INT_ELIDABLE_CANNOT_RAISE = 19`.
    // Also confirms the 4-tuple's trace_target slot points at the
    // macro-emitted `extern "C" fn(i64, i64) -> i64` wrapper — PyPy
    // `getfunctionptr` (`call.py get_jitcode_calldescr`) parity.
    let (policy, _, trace_target, concrete_target, _, _) =
        __majit_call_policy_elidable_canary_mul();
    assert_eq!(
        policy, 19u8,
        "`#[elidable_cannot_raise]` int helper must advertise byte 19"
    );
    assert_eq!(
        trace_target, __majit_call_target_elidable_canary_mul as *const (),
        "policy 4-tuple trace_target must point to the macro-generated extern \"C\" wrapper",
    );
    assert!(!concrete_target.is_null());
}

#[test]
fn elidable_canary_traces_to_call_pure_i_when_args_not_all_const() {
    // pyjitpl.py:3570-3579 — one inputarg + one Const argument →
    // all_const=false → cut from patch_pos and re-emit CALL → CALL_PURE.

    let mut meta = MetaInterp::<()>::new(0);
    // pyjitpl.py:2273-2283 — fixtures that bypass JitDriver::register_descriptor
    // must seed propagate_exception_descr before the first trace start.
    meta.finish_setup_descrs_for_jitdrivers();
    let live_x: i64 = 7;
    let action = meta.force_start_tracing(0, (0, 0), None, None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    // EffectInfo: ElidableCannotRaise + OopSpecIndex::None
    // (= `EffectInfo::new(0, 0)` in `effectinfo.py` parity).
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);

    // inputarg slot 0 = first live value (Type::Int).  `force_start_tracing`'s
    // record_input_arg allocates the Int slot first.
    let live_arg = majit_ir::OpRef::input_arg_int(0);
    let const_y: i64 = 11;
    let const_arg = meta.trace_ctx().expect("active trace").const_int(const_y);

    // Use the macro ABI trampoline directly as the trace function
    // pointer — `extern "C" fn(i64, i64) -> i64`.  The user fn
    // (`fn(i64, i64) -> i64`) has a Rust-ABI raw function pointer with
    // no guarantee that its calling convention matches the C ABI the
    // JIT calls through.  Only the macro wrapper carries the exact C
    // ABI signature (`majit-macros/src/lib.rs`'s `emit_helper_call_target_fn`).
    let trace_fn = __majit_call_target_elidable_canary_mul;
    let func_ptr = trace_fn as *const ();
    let concrete_result = trace_fn(live_x, const_y);

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[live_arg, const_arg],
        &[Type::Int, Type::Int],
        Type::Int,
        effect,
        // pyjitpl.py _build_allboxes parity: funcbox value first,
        // then per-arg concrete values.
        &[
            Value::Int(func_ptr as usize as i64),
            Value::Int(live_x),
            Value::Int(const_y),
        ],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();

    // pyjitpl.py record_result_of_call_pure — original CallI cut, CallPureI re-recorded.
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "trace must contain CallPureI after record_result_of_call_pure patch; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut, not coexist with CallPureI",
    );
    // Result must remain a live OpRef, not a Const.
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const path must not return a Const result; resbox={:?}",
        resbox,
    );
}

#[test]
fn elidable_canary_all_const_args_fold_to_const_and_cut_call() {
    // pyjitpl.py:3568-3569 — every argument Const → CALL cut →
    // ConstInt(resvalue) returned.

    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let action = meta.force_start_tracing(0, (0, 0), None, None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    let trace_fn = __majit_call_target_elidable_canary_mul;
    let func_ptr = trace_fn as *const ();
    let const_x: i64 = 3;
    let const_y: i64 = 4;
    let concrete_result = trace_fn(const_x, const_y);

    let (a, b) = {
        let ctx = meta.trace_ctx().expect("active trace");
        (ctx.const_int(const_x), ctx.const_int(const_y))
    };

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[a, b],
        &[Type::Int, Type::Int],
        Type::Int,
        effect,
        &[
            Value::Int(func_ptr as usize as i64),
            Value::Int(const_x),
            Value::Int(const_y),
        ],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    // Neither CallI nor CallPureI may remain in the trace.
    assert!(
        ops.iter()
            .all(|op| op.opcode != OpCode::CallI && op.opcode != OpCode::CallPureI),
        "all-const elidable call must be fully cut, not patched to CallPureI; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert_eq!(
        ctx.constants_get_value(resbox),
        Some(Value::Int(concrete_result)),
        "all-const fold must return ConstInt(concrete_result)",
    );
}
