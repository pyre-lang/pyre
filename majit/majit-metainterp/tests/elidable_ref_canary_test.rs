//! REF (pointer) analog of `elidable_canary_test.rs`: proves the
//! pointer-returning `#[elidable]` → `CallPureR` → constant-fold path is
//! live, end-to-end, observed from outside the metainterp crate.
//!
//! `resoperation.py RefOp.type = 'r'` — the result Box and the
//! function-pointer arg of a `CALL_PURE_R` are Ref-typed.  This file is
//! the Ref counterpart to the Int canary's `CALL_PURE_I` checks.
//!
//! `pyjitpl.py record_result_of_call_pure` patches a CALL into
//! a CALL_PURE (`call_pure_for_type(Type::Ref) == CallPureR`,
//! resoperation.py), and folds to a `ConstPtr` (history.py)
//! when every argument is constant — *not* a `ConstInt` of the same
//! numeric value (history.py ConstInt vs :307/:314 ConstPtr pin
//! distinct Box types at construction; `history.rs`'s
//! `cond_call_value_ref_typed` documents the aliasing hazard the OpRef
//! enum prevents structurally).
//!
//! ── Policy-byte vs fold-path subtlety ───────────────────────────────
//!
//! Plain `#[elidable]` advertises a *can-raise* policy byte
//! (`REF_ELIDABLE = 21`, `call.py elif cr:`).  The fold machinery
//! exercised by checks 2-3 (`call_typed_with_effect_pure` →
//! `record_result_of_call_pure` → `execute_pure_call`) is the
//! `MIFrame.execute_varargs(pure=True)` path, which is reached ONLY
//! for `EF_ELIDABLE_CANNOT_RAISE`.  Both `call_typed_with_effect_pure`
//! (`history.rs` debug_assert) and `execute_pure_call`
//! (`executor.rs`'s `elidable_can_raise_panics_debug_assertion`
//! `should_panic` guard) hard-reject
//! `EF_ELIDABLE_CAN_RAISE` by design.  So the fold-path checks thread
//! `ExtraEffect::ElidableCannotRaise`, exactly as the Int canary does;
//! a `EF_ELIDABLE_CAN_RAISE` ref helper instead records
//! `CALL_PURE_R + GUARD_NO_EXCEPTION` through the residual-call walker
//! (`jitcode_dispatch/residual_call.rs`'s
//! `select_residual_call_opcode`), a distinct
//! mechanism not under test here.  Check 1 asserts the can-raise policy
//! byte; checks 2-3 assert the cannot-raise fold — these are two
//! separate facts about the same REF helper family.

use majit_ir::{EffectInfo, ExtraEffect, GcRef, OopSpecIndex, OpCode, OpRef, Type, Value};
use majit_macros::elidable;
use majit_metainterp::{BackEdgeAction, MetaInterp};

/// Pointer-returning canary helper.  `*mut u8` return →
/// `helper_call_kind_for_type` → `HelperCallKind::Ref`
/// (`majit-macros/src/lib.rs`).  Plain `#[elidable]` →
/// `EF_ELIDABLE_CAN_RAISE` → `REF_ELIDABLE = 21`
/// (`call_policy_byte.rs`).
#[elidable]
fn elidable_ref_canary(x: i64) -> *mut u8 {
    (x as usize) as *mut u8
}

#[test]
fn elidable_ref_canary_macro_advertises_ref_policy_byte_21() {
    // `call_policy_byte.rs`'s `REF_ELIDABLE = 21`.  Must NOT collapse to
    // `UNSUPPORTED` (0): the `*mut u8` return is classified as a Ref
    // helper (`HelperCallKind::Ref`), and plain `#[elidable]` is the
    // can-raise variant (`call.py:297`).  Also confirms the 6-tuple's
    // trace_target slot points at the macro-emitted `extern "C"` wrapper
    // — `getfunctionptr` (`call.py get_jitcode_calldescr`) parity.
    let (policy, _, trace_target, concrete_target, _, _) =
        __majit_call_policy_elidable_ref_canary();
    assert_eq!(
        policy, 21u8,
        "`#[elidable]` ref helper must advertise REF_ELIDABLE byte 21, not UNSUPPORTED (0)"
    );
    assert_eq!(
        trace_target, __majit_call_target_elidable_ref_canary as *const (),
        "policy 6-tuple trace_target must point to the macro-generated extern \"C\" wrapper",
    );
    assert!(!concrete_target.is_null());
}

#[test]
fn elidable_ref_canary_traces_to_call_pure_r_when_args_not_all_const() {
    // pyjitpl.py:3570-3579 — one inputarg (Ref) + the funcbox → the
    // user-arg is non-const → all_const=false → CALL_R cut, re-emit as
    // CALL_PURE_R (resoperation.py call_pure_for_type(Ref)).

    let mut meta = MetaInterp::<()>::new(0);
    // pyjitpl.py:2273-2283 — seed propagate_exception_descr before the
    // first trace start (fixtures bypass register_descriptor).
    meta.finish_setup_descrs_for_jitdrivers();

    // Live Ref inputarg: seed the trace with a Ref start value so slot 0
    // is a Ref bank input (force_start_tracing's record_input_arg).
    let live_p: usize = 0xBEEF_0000;
    let action = meta.force_start_tracing(0, (0, 0), None, None, &[Value::Ref(GcRef(live_p))]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    // Fold path requires EF_ELIDABLE_CANNOT_RAISE:
    // `MIFrame.execute_varargs(pure=True)` is reached only for that policy.
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);

    // inputarg slot 0 = first live value, Ref-typed (resoperation.py InputArgRef
    // InputArgRef).
    let live_arg = OpRef::input_arg_ref(0);

    // Drive through the macro `extern "C"` trampoline — the only symbol
    // carrying the exact C ABI the JIT calls through
    // (`majit-macros/src/lib.rs`).
    let trace_fn = __majit_call_target_elidable_ref_canary;
    let func_ptr = trace_fn as *const ();
    let concrete_result = trace_fn(live_p as i64) as usize;

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallR,
        func_ptr,
        &[live_arg],
        &[Type::Ref],
        Type::Ref,
        effect,
        // _build_allboxes (pyjitpl.py): funcbox concrete value
        // first (always Int — the raw fn pointer), then per-arg concrete
        // values.  The user arg is Ref-typed.
        &[
            Value::Int(func_ptr as usize as i64),
            Value::Ref(GcRef(live_p)),
        ],
        Value::Ref(GcRef(concrete_result)),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    let opcodes: Vec<_> = ops.iter().map(|op| op.opcode).collect();

    // pyjitpl.py record_result_of_call_pure — original CallR cut, CallPureR re-recorded.
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureR),
        "trace must contain CallPureR after record_result_of_call_pure patch; got opcodes {opcodes:?}",
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallR),
        "original CallR must be cut, not coexist with CallPureR; got opcodes {opcodes:?}",
    );
    // Result must remain a live (non-const) Ref OpRef.
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const path must not return a Const result; resbox={resbox:?}",
    );
}

#[test]
fn elidable_ref_canary_all_const_args_fold_to_const_ptr_not_const_int() {
    // pyjitpl.py:3566-3569 — every arg Const → CALL cut → ConstPtr
    // returned (record_result_of_call_pure: `Value::Ref(r) =>
    // OpRef::const_ptr(r)`).  Critically the folded constant must be a
    // ConstPtr / `Value::Ref`, NOT a ConstInt / `Value::Int` of the same
    // numeric value (`history.rs`'s `cond_call_value_ref_typed` aliasing hazard).

    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let action = meta.force_start_tracing(0, (0, 0), None, None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    let trace_fn = __majit_call_target_elidable_ref_canary;
    let func_ptr = trace_fn as *const ();
    let const_x: i64 = 0x1234;
    let concrete_result = trace_fn(const_x) as usize;

    // A const Ref arg — `const_ref(value)` builds `OpRef::const_ptr(...)`
    // (history.py ConstPtr), distinct from a ConstInt of the same
    // bits.
    let const_ref_arg = meta.trace_ctx().expect("active trace").const_ref(const_x);

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallR,
        func_ptr,
        &[const_ref_arg],
        &[Type::Ref],
        Type::Ref,
        effect,
        &[
            Value::Int(func_ptr as usize as i64),
            Value::Ref(GcRef(const_x as usize)),
        ],
        Value::Ref(GcRef(concrete_result)),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    let opcodes: Vec<_> = ops.iter().map(|op| op.opcode).collect();

    // Neither CallR nor CallPureR may remain — the all-const call is cut.
    assert!(
        ops.iter()
            .all(|op| op.opcode != OpCode::CallR && op.opcode != OpCode::CallPureR),
        "all-const elidable ref call must be fully cut, not patched to CallPureR; got opcodes {opcodes:?}",
    );

    let folded = ctx.constants_get_value(resbox);
    // The folded constant MUST be Ref-typed (ConstPtr), value-matching
    // the concrete pointer.
    assert_eq!(
        folded,
        Some(Value::Ref(GcRef(concrete_result))),
        "all-const ref fold must return ConstPtr(concrete_result), got {folded:?}",
    );
    // Critically: NOT a ConstInt of the same numeric value.
    assert!(
        !matches!(folded, Some(Value::Int(_))),
        "folded constant aliased to Value::Int — ConstPtr/ConstInt distinction lost (history.rs:3150)",
    );
    // Type must be Ref via the typed OpRef variant.
    assert_eq!(
        resbox.ty(),
        Some(Type::Ref),
        "folded constant OpRef must carry Ref type (resoperation.py:638 RefOp.type='r'); resbox={resbox:?}",
    );
}
