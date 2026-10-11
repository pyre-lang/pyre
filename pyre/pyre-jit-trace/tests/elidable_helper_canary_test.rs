//! `#[elidable_cannot_raise]` production helper canary — proves that
//! the first helper attached on the pyre-jit-trace side
//! (`jit_int_in_small_cache_range`) fires through the elidable path
//! when traced.
//!
//! The metainterp-side fixture (`majit-metainterp/tests/elidable_canary_test.rs`)
//! showed the infrastructure live wire; this test goes further and
//! shows that **a macro-attached helper inside a production crate
//! (pyre-jit-trace) is patched to `CallPureI` on a real trace call**.
//!
//! Every trace function pointer routes through the macro-emitted
//! `extern "C" fn(_, _) -> i64` ABI trampoline (`__majit_call_target_*`).
//! PyPy `getfunctionptr` (`call.py get_jitcode_calldescr`) parity — the function
//! pointer the trace records must match (down to ABI) the function the
//! JIT actually calls.
//!
//! `pyjitpl.py MIFrame.execute_varargs(opnum, argboxes, descr,
//! exc=False, pure=True)` + `pyjitpl.py record_result_of_call_pure`
//! still work when driven from a production helper.

use majit_ir::{GcRef, OpCode, OpRef, Type, Value};
use majit_metainterp::{BackEdgeAction, MetaInterp};
use pyre_jit_trace::helpers::{
    __majit_call_policy_jit_int_in_small_cache_range,
    __majit_call_target_jit_int_in_small_cache_range,
    emit_trace_call_int_typed_elidable_cannot_raise, emit_trace_call_ref_typed,
    emit_trace_call_ref_typed_elidable_cannot_raise, jit_instance_getdictvalue,
    jit_lookup_where_with_method_cache,
};
use pyre_object::{__majit_call_policy_int_bit_count, __majit_call_target_int_bit_count};

/// Confirms that the macro emits `__majit_call_policy_*` /
/// `__majit_call_target_*` with the user fn's visibility and that the
/// 4-tuple's trace_target slot points at that wrapper.  PyPy
/// `getfunctionptr` parity, observed from outside the helper's crate.
#[test]
fn elidable_helper_macro_advertises_extern_c_trampoline() {
    let (policy, _, trace_target, concrete_target, _, _) =
        __majit_call_policy_jit_int_in_small_cache_range();
    assert_eq!(policy, 19u8);
    assert_eq!(
        trace_target, __majit_call_target_jit_int_in_small_cache_range as *const (),
        "policy 4-tuple trace_target must point to the macro-generated extern \"C\" wrapper",
    );
    assert!(!concrete_target.is_null());
}

#[test]
fn elidable_helper_traces_to_call_pure_i_when_args_not_all_const() {
    // One inputarg + one Const argument → all_const=false →
    // record_result_of_call_pure patches CallI to CallPureI.
    //
    // Strictly speaking `jit_int_in_small_cache_range` is a 1-arg
    // helper, so a single-inputarg scenario is enough to demonstrate
    // the live wire.

    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let live_x: i64 = 7;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let live_arg = OpRef::input_arg_int(0);
    // Use the macro ABI trampoline directly as the trace function
    // pointer — `extern "C" fn(i64) -> i64`.  The user fn's
    // `pub fn(i64) -> bool` signature may not share calling convention
    // with the C ABI, so feeding it straight into a `Type::Int` (i64)
    // target is unsafe.  Only the macro wrapper covers the bool→i64
    // conversion (1/0) in `majit-macros`'s `helper_return_to_i64`.
    let trace_fn = __majit_call_target_jit_int_in_small_cache_range;
    let func_ptr =
        pyre_interpreter::residual_word_addr!(1, __majit_call_target_jit_int_in_small_cache_range);
    let concrete_result = trace_fn(live_x);
    let effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::ElidableCannotRaise,
        majit_ir::OopSpecIndex::None,
    );

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[live_arg],
        &[Type::Int],
        Type::Int,
        effect,
        &[Value::Int(func_ptr as usize as i64), Value::Int(live_x)],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();

    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "trace must contain CallPureI after record_result_of_call_pure patch; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const result must remain a live OpRef",
    );
}

/// Confirms that the `emit_trace_call_int_typed_elidable_cannot_raise`
/// production emitter is auto-patched to `CallPureI` in the trace.
/// Unlike `helpers.rs::emit_trace_call_int_typed`'s conservative
/// `default_effect_info` call, this wrapper threads an explicit
/// elidable EffectInfo through `record_result_of_call_pure`.
#[test]
fn emit_trace_call_int_typed_elidable_cannot_raise_routes_to_call_pure_i() {
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let live_x: i64 = 0x_dead_beef;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let live_arg = OpRef::input_arg_int(0);
    let trace_fn = __majit_call_target_jit_int_in_small_cache_range;
    let func_ptr =
        pyre_interpreter::residual_word_addr!(1, __majit_call_target_jit_int_in_small_cache_range);
    let concrete_result = trace_fn(live_x);

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_int_typed_elidable_cannot_raise(
            ctx,
            func_ptr,
            &[live_arg],
            &[Type::Int],
            &[Value::Int(func_ptr as usize as i64), Value::Int(live_x)],
            Value::Int(concrete_result),
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "production emitter must record CallPureI; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const result must remain a live OpRef",
    );
}

/// Confirms that `pyre-object::int_bit_count` (port of RPython
/// `intobject.py _bit_count`) is patched to `CallPureI` when traced.
/// Together with `jit_int_in_small_cache_range` (pyre-jit-trace side
/// attachment), this proves that the pyre-object side fires through the
/// same `record_result_of_call_pure` path.
#[test]
fn elidable_int_bit_count_macro_advertises_extern_c_trampoline_and_traces_call_pure_i() {
    let (policy, _, trace_target, _, _, _) = __majit_call_policy_int_bit_count();
    assert_eq!(policy, 19u8);
    assert_eq!(trace_target, __majit_call_target_int_bit_count as *const (),);

    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let live_x: i64 = 0x_0bad_cafe_dead_beef;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let live_arg = OpRef::input_arg_int(0);
    let trace_fn = __majit_call_target_int_bit_count;
    let func_ptr = pyre_interpreter::residual_word_addr!(1, __majit_call_target_int_bit_count);
    let concrete_result = trace_fn(live_x);
    let effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::ElidableCannotRaise,
        majit_ir::OopSpecIndex::None,
    );

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[live_arg],
        &[Type::Int],
        Type::Int,
        effect,
        &[Value::Int(func_ptr as usize as i64), Value::Int(live_x)],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "int_bit_count must record CallPureI via record_result_of_call_pure",
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut"
    );
    assert!(ctx.constants_get_value(resbox).is_none());
}

#[test]
fn elidable_helper_all_const_args_fold_to_const_and_cut_call() {
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let action = meta.force_start_tracing(0, (0, 0), None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let const_x: i64 = 3;
    let trace_fn = __majit_call_target_jit_int_in_small_cache_range;
    let func_ptr =
        pyre_interpreter::residual_word_addr!(1, __majit_call_target_jit_int_in_small_cache_range);
    let concrete_result = trace_fn(const_x);
    let effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::ElidableCannotRaise,
        majit_ir::OopSpecIndex::None,
    );

    let const_arg = meta.trace_ctx().expect("active trace").const_int(const_x);

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[const_arg],
        &[Type::Int],
        Type::Int,
        effect,
        &[Value::Int(func_ptr as usize as i64), Value::Int(const_x)],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    assert!(
        ops.iter()
            .all(|op| op.opcode != OpCode::CallI && op.opcode != OpCode::CallPureI),
        "all-const elidable call must be fully cut; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert_eq!(
        ctx.constants_get_value(resbox),
        Some(Value::Int(concrete_result)),
        "all-const fold must return ConstInt(concrete_result)",
    );
}

// ── getattr-inline: synthetic-pointer canary ──────────────────────────
//
// Proves the PRODUCTION Ref emitter
// `emit_trace_call_ref_typed_elidable_cannot_raise` records `CallPureR`
// for the exact `(Ref, Ref, Int) -> Ref` argument shape of
// `_pure_lookup_where_with_method_cache` (baseobjspace.rs, `@elidable`
// — parity with typeobject.py:516).  The metainterp-side
// `elidable_ref_canary_test.rs` proved the generic Ref fold via
// `call_typed_with_effect_pure` directly with a one-Int-arg helper; this
// goes further and exercises the production wrapper the getattr inline will
// call from `trace_load_attr` / `MIFrame::load_method`, with the real
// two-Ref-args + one-Int-arg shape.
//
// The recorder records but does not execute the call (pyjitpl.py
// `MIFrame.execute_varargs`), using the supplied `concrete_result`, so a
// synthetic same-shape `extern "C"` pointer stands in for the still-private
// real lookup helper; the recordable-wrapper canary below wires the real one.

/// `(w_type, w_name, version_tag) -> w_descr`-shaped synthetic helper.
extern "C" fn lookup_where_shape_canary(_w_type: i64, _w_name: i64, _version_tag: i64) -> i64 {
    0x5EED_1000
}

#[test]
fn emit_ref_lookup_shape_routes_to_call_pure_r_when_type_not_const() {
    // The hot-loop fold promotes w_type so all three args become const and
    // the call folds entirely (next test).  Before the version_tag guard
    // hoists it, w_type is a live (non-const) Ref — that is the shape that
    // exercises the CALL_R -> CALL_PURE_R patch
    // (record_result_of_call_pure, pyjitpl.py).
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();

    // Seed a live Ref inputarg in slot 0 = w_type (the receiver type).
    let live_type: usize = 0x710E_0000;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Ref(GcRef(live_type))]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let func_ptr = pyre_interpreter::residual_word_addr!(3, lookup_where_shape_canary);
    let w_name: usize = 0x4A3E_0000;
    let version_tag: i64 = 0x5234;
    let concrete_result = lookup_where_shape_canary(live_type as i64, w_name as i64, version_tag);

    let w_type_arg = OpRef::input_arg_ref(0);
    let (w_name_arg, version_arg) = {
        let ctx = meta.trace_ctx().expect("active trace");
        (ctx.const_ref(w_name as i64), ctx.const_int(version_tag))
    };

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_ref_typed_elidable_cannot_raise(
            ctx,
            func_ptr,
            &[w_type_arg, w_name_arg, version_arg],
            &[Type::Ref, Type::Ref, Type::Int],
            // _build_allboxes (pyjitpl.py): funcbox concrete value
            // first (the raw fn pointer, always Int), then per-arg concretes.
            &[
                Value::Int(func_ptr as usize as i64),
                Value::Ref(GcRef(live_type)),
                Value::Ref(GcRef(w_name)),
                Value::Int(version_tag),
            ],
            Value::Ref(GcRef(concrete_result as usize)),
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    let opcodes: Vec<_> = ops.iter().map(|op| op.opcode).collect();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureR),
        "production Ref emitter must record CallPureR for the (Ref,Ref,Int) lookup shape; got {opcodes:?}",
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallR),
        "original CallR must be cut, not coexist with CallPureR; got {opcodes:?}",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const path must keep a live Ref OpRef result",
    );
}

#[test]
fn emit_ref_lookup_shape_all_const_folds_to_const_ptr() {
    // Hot-loop shape: w_type promoted, w_name interned, version_tag guarded
    // -> all three args const -> the call is fully cut and the result folds
    // to a ConstPtr (Value::Ref), NOT a ConstInt of the same bits
    // (a `ConstInt` slot aliases any int constant of the same raw value —
    // the hazard `history.rs` pins on its typed-constant recorders).  This is
    // the cross-iteration fold the getattr inline delivers.
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let action = meta.force_start_tracing(0, (0, 0), None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let func_ptr = pyre_interpreter::residual_word_addr!(3, lookup_where_shape_canary);
    let w_type: usize = 0x710E_1111;
    let w_name: usize = 0x4A3E_1111;
    let version_tag: i64 = 0x5235;
    let concrete_result = lookup_where_shape_canary(w_type as i64, w_name as i64, version_tag);

    let (w_type_arg, w_name_arg, version_arg) = {
        let ctx = meta.trace_ctx().expect("active trace");
        (
            ctx.const_ref(w_type as i64),
            ctx.const_ref(w_name as i64),
            ctx.const_int(version_tag),
        )
    };

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_ref_typed_elidable_cannot_raise(
            ctx,
            func_ptr,
            &[w_type_arg, w_name_arg, version_arg],
            &[Type::Ref, Type::Ref, Type::Int],
            &[
                Value::Int(func_ptr as usize as i64),
                Value::Ref(GcRef(w_type)),
                Value::Ref(GcRef(w_name)),
                Value::Int(version_tag),
            ],
            Value::Ref(GcRef(concrete_result as usize)),
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    assert!(
        ops.iter()
            .all(|op| op.opcode != OpCode::CallR && op.opcode != OpCode::CallPureR),
        "all-const elidable lookup call must be fully cut; got {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert_eq!(
        ctx.constants_get_value(resbox),
        Some(Value::Ref(GcRef(concrete_result as usize))),
        "all-const fold must return ConstPtr(concrete_result), not ConstInt",
    );
}

// ── getattr-inline: REAL lookup helper, recordable wrapper ────────────
//
// The canary above proved the fold mechanism for the `(Ref, Ref, Int) -> Ref`
// shape with a synthetic same-shape pointer.  This one wires the REAL recordable
// surface for the lookup: `jit_lookup_where_with_method_cache`
// (helpers.rs), a plain `extern "C"` i64-ABI wrapper — like
// `jit_namespace_cell_lookup` — that calls the now-`pub`
// `_pure_lookup_where_with_method_cache` (baseobjspace.rs, `@elidable`,
// typeobject.py:516).
//
// The `#[elidable]` macro does NOT emit a usable trampoline for that helper:
// its `PyObjectRef`-aliased args are not recognised as pointers by the
// macro's type matcher (helper_arg_from_i64), so the policy degrades to
// UNSUPPORTED.  As with every other pyre trace helper, the foldable surface
// is therefore this i64 wrapper, recorded with an explicit
// `ElidableCannotRaise` effect (the raw-ptr return genuinely cannot raise).
// The getattr inline records it from `trace_load_attr` /
// `MIFrame::load_method` behind a promoted-`version_tag` guard, with
// `jit_getattr` as the deopt fallback.
//
// The recorder records but does not execute the call (pyjitpl.py),
// so a synthetic concrete result stands in — the real wrapper would
// dereference `w_name` and walk the MRO, which needs a live objspace.

#[test]
fn real_lookup_wrapper_records_call_pure_r_when_type_not_const() {
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();

    // Live (non-const) w_type Ref in slot 0 — the shape before the
    // version_tag guard promotes the receiver type to a constant.
    let live_type: usize = 0x710E_2222;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Ref(GcRef(live_type))]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let func_ptr = pyre_interpreter::residual_word_addr!(3, jit_lookup_where_with_method_cache);
    let w_name: usize = 0x4A3E_2222;
    let version_tag: i64 = 0x5236;
    // Synthetic descriptor pointer; the wrapper is NOT invoked here.
    let concrete_result: usize = 0xDE5C_2222;

    let w_type_arg = OpRef::input_arg_ref(0);
    let (w_name_arg, version_arg) = {
        let ctx = meta.trace_ctx().expect("active trace");
        (ctx.const_ref(w_name as i64), ctx.const_int(version_tag))
    };

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_ref_typed_elidable_cannot_raise(
            ctx,
            func_ptr,
            &[w_type_arg, w_name_arg, version_arg],
            &[Type::Ref, Type::Ref, Type::Int],
            &[
                Value::Int(func_ptr as usize as i64),
                Value::Ref(GcRef(live_type)),
                Value::Ref(GcRef(w_name)),
                Value::Int(version_tag),
            ],
            Value::Ref(GcRef(concrete_result)),
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    let opcodes: Vec<_> = ops.iter().map(|op| op.opcode).collect();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureR),
        "real lookup wrapper must record CallPureR; got {opcodes:?}",
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallR),
        "original CallR must be cut; got {opcodes:?}",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const path keeps a live Ref OpRef result",
    );
}

#[test]
fn real_lookup_wrapper_all_const_folds_to_const_ptr() {
    // Hot-loop shape: w_type promoted, w_name interned, version_tag guarded
    // -> all three args const -> the lookup call is fully cut and folds to a
    // ConstPtr (Value::Ref), the cross-iteration fold the getattr inline delivers.
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let action = meta.force_start_tracing(0, (0, 0), None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let func_ptr = pyre_interpreter::residual_word_addr!(3, jit_lookup_where_with_method_cache);
    let w_type: usize = 0x710E_3333;
    let w_name: usize = 0x4A3E_3333;
    let version_tag: i64 = 0x5237;
    let concrete_result: usize = 0xDE5C_3333;

    let (w_type_arg, w_name_arg, version_arg) = {
        let ctx = meta.trace_ctx().expect("active trace");
        (
            ctx.const_ref(w_type as i64),
            ctx.const_ref(w_name as i64),
            ctx.const_int(version_tag),
        )
    };

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_ref_typed_elidable_cannot_raise(
            ctx,
            func_ptr,
            &[w_type_arg, w_name_arg, version_arg],
            &[Type::Ref, Type::Ref, Type::Int],
            &[
                Value::Int(func_ptr as usize as i64),
                Value::Ref(GcRef(w_type)),
                Value::Ref(GcRef(w_name)),
                Value::Int(version_tag),
            ],
            Value::Ref(GcRef(concrete_result)),
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    assert!(
        ops.iter()
            .all(|op| op.opcode != OpCode::CallR && op.opcode != OpCode::CallPureR),
        "all-const lookup call must be fully cut; got {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert_eq!(
        ctx.constants_get_value(resbox),
        Some(Value::Ref(GcRef(concrete_result))),
        "all-const fold must return ConstPtr(concrete_result), not ConstInt",
    );
}

// ── getattr-inline: instance-dict shadow read residual ────────────────
//
// `jit_instance_getdictvalue` (helpers.rs) wraps `instance_node_getdictvalue`
// (mapdict.rs, `getdictvalue` mapdict.py).  The LOAD_METHOD fast
// path (callmethod.py) reads it after the type lookup to confirm no
// instance attribute shadows the class method.  Unlike the type lookup, it is
// NOT pure — the instance dict mutates — so it is recorded as a normal
// residual `CallR` (emit_trace_call_ref_typed, default_effect_info), guarded
// `null` per iteration rather than folded across iterations.  These two tests
// pin (1) the null-guard ABI contract and (2) that the residual call survives
// even with all-const args (the contrast with the foldable lookup above).

#[test]
fn instance_getdictvalue_wrapper_null_receiver_returns_py_null() {
    // ABI contract: a null receiver / name returns PY_NULL (no deref).
    assert_eq!(
        jit_instance_getdictvalue(pyre_object::PY_NULL, pyre_object::PY_NULL),
        pyre_object::PY_NULL,
        "null receiver must short-circuit to PY_NULL",
    );
    assert_eq!(
        jit_instance_getdictvalue(
            pyre_object::PY_NULL,
            0x4A3E_4444 as pyre_object::PyObjectRef
        ),
        pyre_object::PY_NULL,
        "null receiver must short-circuit to PY_NULL even with a non-null name",
    );
}

#[test]
fn instance_getdictvalue_records_residual_call_r_not_pure() {
    // Even with all-const args the instance-dict read stays a live residual
    // CallR (NOT CallPureR, NOT folded to a Const) — the per-iteration
    // shadowing check the LOAD_METHOD fast path needs.
    let mut meta = MetaInterp::<()>::new(0);
    meta.finish_setup_descrs_for_jitdrivers();
    let action = meta.force_start_tracing(0, (0, 0), None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let func_ptr = pyre_interpreter::residual_word_addr!(2, jit_instance_getdictvalue);
    let w_obj: usize = 0x1457_5555;
    let w_name: usize = 0x4A3E_5555;

    let (w_obj_arg, w_name_arg) = {
        let ctx = meta.trace_ctx().expect("active trace");
        (ctx.const_ref(w_obj as i64), ctx.const_ref(w_name as i64))
    };

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_ref_typed(
            ctx,
            func_ptr,
            &[w_obj_arg, w_name_arg],
            &[Type::Ref, Type::Ref],
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    ctx.ensure_ops_materialized();
    let ops = ctx.ops();
    let opcodes: Vec<_> = ops.iter().map(|op| op.opcode).collect();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallR),
        "instance-dict read must record a residual CallR; got {opcodes:?}",
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallPureR),
        "instance-dict read must NOT be a foldable CallPureR; got {opcodes:?}",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "residual call result must stay a live OpRef even with all-const args",
    );
}
