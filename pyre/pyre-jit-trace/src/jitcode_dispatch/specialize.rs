//! Per-opcode specialization strategies: the `try_walker_*` entry points
//! that recognize a specializable shape (int/long/float arithmetic and
//! comparisons, attribute and method loads, container builds and subscript,
//! list-append, exception construction/raise, for-iter, slice, and the
//! module/name cell folds) and either fold or record a specialized trace,
//! returning `None` to fall through to the generic path.
//!
//! **Parity:** pyre-local trace-time folding. PyPy defers most
//! specialization to `optimizeopt/` (a separate later pass); pyre folds
//! during the walk instead. The fast-path shapes still mirror the
//! `opimpl_*` fast paths and `blackhole.py`'s `bhimpl_*` folds.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. The shared walker
//! primitives these build on (unbox/box, guard emission, operand reads)
//! stay in `mod.rs`; the specialization opname arms stay in `handle` and
//! call into these entry points.

use super::*;
use rustpython_wtf8::Wtf8;

/// Replace an authentically executed builtin raise with ordinary trace
/// allocations.  The exception's stored args are the authority for both
/// wording and arity; keeping their objects as rooted trace constants avoids
/// re-deriving messages while still letting escape analysis virtualize the
/// exception and its args list when a handler discards them.
fn walker_recorded_builtin_raise_is_supported(
    exc: pyre_object::PyObjectRef,
    expected_kind: pyre_object::interp_exceptions::ExcKind,
) -> bool {
    if exc.is_null()
        || unsafe { !pyre_object::is_exception(exc) }
        || unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) } != expected_kind
    {
        return false;
    }
    let args_storage = unsafe { pyre_object::interp_exceptions::w_exception_get_args_storage(exc) };
    if args_storage.is_null() {
        return true;
    }
    let args_len = unsafe { pyre_object::interp_exceptions::rlist_len(args_storage) };
    (0..args_len).all(|index| {
        let arg = unsafe { pyre_object::interp_exceptions::rlist_getitem(args_storage, index) };
        !arg.is_null()
            && unsafe { pyre_object::is_str(arg) && pyre_object::is_exact_builtin_instance(arg) }
    })
}

/// `GETFIELD_GC_R(ec, sys_exc_value)` then `SETFIELD_GC(exc, active,
/// w_context)`, plus the same write on the concrete exception so the
/// authoritative walk observes `__context__`.
fn walker_chain_exception_context<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ec: OpRef,
    raised: OpRef,
    exc: pyre_object::PyObjectRef,
    kind: pyre_object::interp_exceptions::ExcKind,
    user: bool,
) {
    let active = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[raised, active],
        crate::descr::w_exception_context_descr_for(kind, user),
    );
    fbw_context_chained_insert(raised);
    let active_concrete = pyre_interpreter::eval::get_current_exception();
    if !active_concrete.is_null() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exc, active_concrete);
        }
    }
}

fn walker_emit_recorded_builtin_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ec: OpRef,
    exc: pyre_object::PyObjectRef,
    expected_kind: pyre_object::interp_exceptions::ExcKind,
) -> DispatchOutcome {
    debug_assert!(walker_recorded_builtin_raise_is_supported(
        exc,
        expected_kind
    ));
    let _roots = pyre_object::gc_roots::push_roots();
    // Read every pinned value back out of its slot instead of reusing the
    // local that was handed to `pin_root`.  `pin_root` normalizes the address
    // it publishes once a second mutator has existed (`gc_roots.rs`
    // `RootScope::pin_root`), so past that point the caller's copy can still
    // name the pre-forwarding object while the slot names the live one — and
    // these values are baked into the trace as `ConstPtr`s, which outlive the
    // walk.  Same shape as the zip/tuple concrete-shadow build below.
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let exc = pyre_object::gc_roots::pin_root(exc);
    // The typeptr is a static. Capture it before later allocations move `exc`.
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc) };
    let exc_type = unsafe { (*exc).ob_type } as *const _ as i64;
    let args_storage = unsafe {
        pyre_object::interp_exceptions::w_exception_get_args_storage(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
        )
    };
    let args_storage_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args_storage);
    let args_len = unsafe {
        pyre_object::interp_exceptions::rlist_len(pyre_object::gc_roots::shadow_stack_get(
            args_storage_slot,
        ))
    };
    let mut concrete_args = Vec::with_capacity(args_len);
    for index in 0..args_len {
        let arg = unsafe {
            pyre_object::interp_exceptions::rlist_getitem(
                pyre_object::gc_roots::shadow_stack_get(args_storage_slot),
                index,
            )
        };
        let arg_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(arg);
        concrete_args.push(unsafe { pyre_object::gc_roots::shadow_stack_get(arg_slot) });
    }
    let args = concrete_args
        .iter()
        .map(|&arg| ctx.trace_ctx.const_ref(arg as i64))
        .collect::<Vec<_>>();
    let args_list = crate::helpers::emit_rlist_inline(ctx.trace_ctx, &args);

    let class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(expected_kind);
    let class = ctx.trace_ctx.const_ref(class as i64);
    let raised = crate::helpers::emit_exception_new_inline(
        ctx.trace_ctx,
        expected_kind,
        class,
        args_list,
        user,
    );
    ctx.trace_ctx.heap_cache_mut().class_now_known(raised);
    ctx.trace_ctx
        .set_opref_concrete(raised, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    fbw_built_exc_insert(raised);
    walker_chain_exception_context(ctx, ec, raised, exc, expected_kind, user);

    fbw_count_executed_residual(false, true);
    let exc_concrete = ConcreteValue::Ref(exc);
    ctx.set_last_exc_value(raised, exc_concrete);
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));
    DispatchOutcome::SubRaise {
        exc: raised,
        exc_concrete,
    }
}

const TRUTH_VALUE_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::opcode_ops::truth_value",
    commit_label: "truth_value_commit",
    call_site_label: "truth_value_call_site",
    decline_tag: "TRUTH-VALUE-SUBWALK",
};

/// Truth residual: walk `opcode_ops::truth_value` for an exact builtin.
///
/// `is_true` sends an exact builtin to `is_true_slot` and every other
/// object to `is_true_lookup` (`dont_look_inside`). A sub-walk that reaches
/// that call and then declines has already run the Python, so this descent
/// stays exact-builtin-only. An overriding `__bool__` is inlined by
/// `try_walker_inline_truth_bool` instead of folding the payload. The
/// jitcode returns the raw bool in the int bank.
pub(crate) fn try_walker_orthodox_truth<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'i' {
        return Ok(None);
    }
    let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
        return Ok(None);
    };
    if unsafe { !pyre_object::is_exact_builtin_instance(obj) } {
        return Ok(None);
    }
    try_walker_orthodox_descent(
        ctx,
        op_pc,
        &[],
        &[(operand, obj)],
        &[],
        dst,
        dst_bank,
        &TRUTH_VALUE_DESCENT,
    )
}

/// The `W_LongObject.value` payload of a concrete long, read the way the folds
/// that pass a payload to an `rbigint` helper need it.
///
/// # Safety
/// `obj` must be a live concrete `W_LongObject` from the walker shadow.
#[allow(dead_code)] // longobject.py W_LongObject.value
unsafe fn long_payload_of(obj: pyre_object::PyObjectRef) -> i64 {
    unsafe { *((obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET) as *const i64) }
}

/// `longobject.py _make_descr_cmp`'s `isinstance(self, W_LongObject)` plus
/// the `self.num` field read.  Exact `w_class` first implies the LONG
/// vtable, so `walker_guard_class` then skips the redundant `GuardClass`.
/// `W_IntObject.intval` sits at the same offset as `W_LongObject.value`.
#[allow(dead_code)] // longobject.py _make_descr_cmp
fn walker_guard_long_and_read_payload<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    boxed: OpRef,
    expected_class: pyre_object::PyObjectRef,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(obj) = walker_concrete_ref_object(ctx, boxed) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_long(obj) } {
        return Ok(None);
    }
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && !unsafe { pyre_object::tagged_int::is_tagged_int(obj) }
    {
        let lowbit = crate::helpers::emit_tag_lowbit_test(ctx.trace_ctx, boxed, false);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[lowbit])?;
    }
    walker_guard_exact_w_class(ctx, op_pc, boxed, expected_class)?;
    walker_guard_class(ctx, op_pc, boxed, long_type_addr)?;
    let payload = unsafe { long_payload_of(obj) };
    let field =
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, boxed, crate::descr::long_value_descr());
    ctx.trace_ctx.set_opref_concrete(
        field,
        majit_ir::Value::Ref(majit_ir::GcRef(payload as usize)),
    );
    Ok(Some(field))
}

/// Record the `getfield_gc_r` that reads a long operand's `value` payload.
/// A box the same trace built with [`crate::helpers::emit_box_long_inline`]
/// answers this out of the heap cache, so the read costs nothing and the box
/// keeps no reason to escape.
#[allow(dead_code)] // longobject.py getfield of value
fn walker_read_long_payload<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    boxed: OpRef,
    concrete_payload: i64,
) -> OpRef {
    let payload = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[boxed],
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        payload,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_payload as usize)),
    );
    payload
}

/// rint.py `_ovf_zer` guards for a machine-int division: `int_eq(rhs,0)` →
/// `guard_false` plus `(lhs==INT_MIN)&(rhs==-1)` → `guard_false`.  Both must
/// precede the elidable `ll_int_py_div` / `ll_int_py_mod` call so a re-used
/// trace bails before the helper's `wrapping_div` / `wrapping_rem` returns a
/// wrap value.  A `divmod` site shares one guard pair across both halves.
fn walker_emit_int_div_domain_guards<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
) -> Result<(), DispatchError> {
    let rhs_zero = walker_int_eq_const(ctx, rhs_raw, 0, (rb == 0) as i64);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[rhs_zero])?;
    let lhs_is_min = walker_int_eq_const(ctx, lhs_raw, i64::MIN, (la == i64::MIN) as i64);
    let rhs_is_neg_one = walker_int_eq_const(ctx, rhs_raw, -1, (rb == -1) as i64);
    let ovf_both = ctx
        .trace_ctx
        .record_op(OpCode::IntAnd, &[lhs_is_min, rhs_is_neg_one]);
    ctx.trace_ctx.set_opref_concrete(
        ovf_both,
        majit_ir::Value::Int(((la == i64::MIN) as i64) & ((rb == -1) as i64)),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[ovf_both])
}

/// After a successful `//` / `%` dest-write, pin the same `_ovf_zer`
/// pair `try_emit_exact_int_binop` records so a later zero divisor deopts
/// before the compiled dest is stored into a Python local (`checksum +=`).
pub(crate) fn walker_guard_int_div_domain_if_exact<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<(), DispatchError> {
    if r_args.len() != 2 {
        return Ok(());
    }
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(());
    };
    unsafe {
        for obj in [lhs_obj, rhs_obj] {
            if !pyre_object::is_int(obj) || !pyre_object::is_exact_builtin_instance(obj) {
                return Ok(());
            }
        }
    }
    let la = unsafe { pyre_object::w_int_get_value(lhs_obj) };
    let rb = unsafe { pyre_object::w_int_get_value(rhs_obj) };
    // `acc //= 0` on an except bridge: `int_eq(0, 0)` is ConstInt(1) and
    // `GUARD_FALSE` of that is `InvalidLoop` ("proven to always fail").
    // The raise is the domain check; do not emit it as a never-taken guard.
    if rb == 0 {
        return Ok(());
    }
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let lhs_raw = walker_unbox_int(ctx, op_pc, lhs, int_type_addr)?;
    let rhs_raw = walker_unbox_int(ctx, op_pc, rhs, int_type_addr)?;
    // Pin the boxed divisor only when the unbox is already a ConstInt.
    // `int_eq(unbox(rhs), 0)` folds away in that case, so a later
    // `divisor = 7 if ... else 0` (`flip_floor`) would skip the zer
    // check.  A red GetfieldGc unbox keeps the check live; pinning the
    // box identity there retraces every new divisor (`check(depth-2)`
    // in `selfrec_tail_exception_unwind`).
    //
    // A `NewWithVtable` box is unescaped and the optimizer virtualizes
    // it; `GUARD_VALUE` on that box is `promote of a virtual`
    // (`optimizeopt` `optimize_GUARD_VALUE`).
    if rhs_raw.is_constant() && !rhs.is_constant() && !ctx.trace_ctx.heap_cache().is_unescaped(rhs)
    {
        let expected = ctx.trace_ctx.const_ref(rhs_obj as i64);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[rhs, expected])?;
    }
    walker_emit_int_div_domain_guards(ctx, op_pc, lhs_raw, rhs_raw, la, rb)
}

/// jtransform.py `OS_INT_PY_DIV` / `OS_INT_PY_MOD` elidable residual call
/// (`call_typed_with_effect_pure` → `CallI` patched via
/// `record_result_of_call_pure`), returning the result op and its recorded
/// value.  The caller must have emitted
/// [`walker_emit_int_div_domain_guards`] over the same operand pair first.
fn walker_emit_int_py_div_or_mod<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
    is_div: bool,
) -> (OpRef, i64) {
    let (func_ptr, effect_info, concrete_result) = if is_div {
        (
            majit_metainterp::blackhole::ll_int_py_div as *const (),
            majit_metainterp::INT_PY_DIV_EFFECT_INFO,
            majit_metainterp::blackhole::ll_int_py_div(la, rb),
        )
    } else {
        (
            majit_metainterp::blackhole::ll_int_py_mod as *const (),
            majit_metainterp::INT_PY_MOD_EFFECT_INFO,
            majit_metainterp::blackhole::ll_int_py_mod(la, rb),
        )
    };
    let r = ctx.trace_ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[lhs_raw, rhs_raw],
        &[majit_ir::Type::Int, majit_ir::Type::Int],
        majit_ir::Type::Int,
        effect_info,
        &[
            majit_ir::Value::Int(func_ptr as usize as i64),
            majit_ir::Value::Int(la),
            majit_ir::Value::Int(rb),
        ],
        majit_ir::Value::Int(concrete_result),
    );
    ctx.trace_ctx
        .set_opref_concrete(r, majit_ir::Value::Int(concrete_result));
    (r, concrete_result)
}

/// Exact-int `//` / `%` by a zero divisor, recorded as the interpreter's raise
/// rather than as the descent's materialiser call.
///
/// `binary_value_from_tag`'s `int_floordiv` / `int_mod` bodies build their
/// `ZeroDivisionError` through `pyerror_zero_division_to_exc_object`, a
/// published `dont_look_inside` materialiser (`front/result_exc.rs`
/// `FUSED_KIND_CTORS`).  A descent therefore records the instance as the result
/// of an opaque call, and an opaque call's result is a concrete object:
/// `OptVirtualize` can fold away neither it, nor the `PyTraceback` the raise
/// links onto it, nor the `sys_exc_value` save/restore around the handler.  A
/// `try: x // 0 / except ZeroDivisionError:` loop then materialises all of them
/// on every iteration.  `walker_emit_recorded_builtin_raise` records the same
/// construction as generated ops, which the optimizer removes when nothing
/// observes the instance.
///
/// The divisor test is a `GUARD_TRUE(int_eq(divisor, 0))`, so a later non-zero
/// divisor side-exits to a bridge that takes the dividing arm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_binary_op_int_zero_div<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    if !matches!(
        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
        Some(
            BinaryOperator::FloorDivide
                | BinaryOperator::InplaceFloorDivide
                | BinaryOperator::Remainder
                | BinaryOperator::InplaceRemainder
        )
    ) {
        return Ok(None);
    }
    let lhs = r_args[0];
    let rhs = r_args[1];
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, lhs),
        walker_concrete_ref_object(ctx, rhs),
    ) else {
        return Ok(None);
    };
    // `is_int` reads `ob_type`, which an `int` subclass shares, and
    // `walker_numeric_builtin_class` answers with the canonical `int` for one.
    unsafe {
        for obj in [lhs_obj, rhs_obj] {
            if !pyre_object::is_int(obj)
                || pyre_object::is_bool(obj)
                || !pyre_object::is_exact_builtin_instance(obj)
            {
                return Ok(None);
            }
        }
        if pyre_object::w_int_get_value(rhs_obj) != 0 {
            return Ok(None);
        }
    }
    // The raising arm needs the helper-produced exception as its concrete
    // shadow, but records no helper call in the trace.  Execute the live
    // `allboxes` (`executor.execute_residual_call`), not the shadow objects
    // used for the exactness gate: a stale zero in the rhs shadow must not
    // invent a `ZeroDivisionError` for `n % 2`.
    // Native `sdiv` by zero returns 0 on some backends and the helper
    // then dest-writes NULL instead of `Err`.  A live zero divisor is
    // still the raising arm.
    let exc_i64 = match walker_execute_may_force_boxed_outcome(ctx, allboxes, call_descr) {
        Some(Err(exc)) => exc,
        Some(Ok(0)) => {
            let mut err = pyre_interpreter::PyError::zero_division("division by zero");
            err.to_exc_object() as i64
        }
        // `None` means the call was never executed (non-constant callee,
        // symbolic fnaddr, or an argument without a concrete), not that it
        // divided by zero.
        Some(Ok(_)) | None => return Ok(None),
    };
    // The helper publishes through both the blackhole cell (drained by
    // `execute_residual_call`) and the backend exception cells.  The latter
    // belong to compiled execution; drain the trace-time publish before the
    // walk continues into the Python handler.
    if let Some(cb) = crate::callbacks::try_get() {
        (cb.drain_backend_jit_exc)();
    }
    let exc = exc_i64 as usize as pyre_object::PyObjectRef;
    let kind = pyre_object::interp_exceptions::ExcKind::ZeroDivisionError;
    if !walker_recorded_builtin_raise_is_supported(exc, kind) {
        return Ok(None);
    }
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // Commit to the raising arm only after every decline.
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let _lhs_raw = walker_unbox_int(ctx, op_pc, lhs, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, lhs, walker_numeric_builtin_class(lhs_obj))?;
    let rhs_raw = walker_unbox_int(ctx, op_pc, rhs, int_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, rhs, walker_numeric_builtin_class(rhs_obj))?;
    let rhs_zero = walker_int_eq_const(ctx, rhs_raw, 0, 1);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[rhs_zero])?;
    Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpecialisedPairKind {
    Int,
    Float,
    Object,
}

/// Identify the three classes produced by
/// `specialisedtupleobject.py makespecialisedtuple2`.
pub(crate) fn specialised_pair_kind(
    seq_type: *const pyre_object::pyobject::PyType,
) -> Option<SpecialisedPairKind> {
    use pyre_object::specialisedtupleobject::{
        SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    };
    if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_II_TYPE) {
        Some(SpecialisedPairKind::Int)
    } else if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_FF_TYPE) {
        Some(SpecialisedPairKind::Float)
    } else if std::ptr::eq(seq_type, &SPECIALISED_TUPLE_OO_TYPE) {
        Some(SpecialisedPairKind::Object)
    } else {
        None
    }
}

/// FBW fold of the UNPACK_SEQUENCE two-residual lowering (`unpack_sequence_fn`
/// validator + per-index `unpack_item_fn` reader emitted by the codewriter
/// UNPACK_SEQUENCE arm) for an arity-2 specialised tuple: guard the
/// specialisation's class once, then read `value0` / `value1` directly instead
/// of leaving three opaque `CALL_MAY_FORCE` residuals in the loop.
///
/// `objspace.py fixedview` reaches `tolist()` for every
/// `W_AbstractTupleObject`, and `specialisedtupleobject.py tolist`
/// unrolls over `_immutable_fields_` value slots, so upstream traces the whole
/// unpack inline and the optimizer virtualizes the pair away. Both arity-2
/// layouts are covered here because `makespecialisedtuple2`
/// (`specialisedtupleobject.py`) never falls back to a plain tuple:
///   * `ii` — `value0`/`value1` are inline machine ints, so the read is
///     `getfield_gc_pure_i` + `wrapint` and the items stay unboxed through the
///     downstream BINARY_OP int fold (the walker analogue of the retired
///     MIFrame `W_SpecialisedTupleObject_ii` reads);
///   * `ff` — the same shape with `getfield_gc_pure_f` + `wrapfloat`; this is
///     the representation `zip` produces for a pair of exact floats;
///   * `oo` — `wraps[i]` for an object slot is the identity
///     (`specialisedtupleobject.py`), so the `getfield_gc_r` result is
///     already the item. This is the layout a `divmod(long, long)` result pair
///     takes, since neither half satisfies `is_plain_int1`.
///
/// Returns `Ok(Some(()))` when folded (the caller returns `Continue`);
/// `Ok(None)` to fall through to the opaque residual record, which stays
/// correct for any other shape — so a non-foldable sequence is not declined.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_unpack<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    i_args: &[OpRef],
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let (Some(&int_arg), Some(&seq)) = (i_args.first(), r_args.first()) else {
        return Ok(None);
    };
    let Some(majit_ir::Value::Int(int_val)) = ctx.trace_ctx.box_value(int_arg) else {
        return Ok(None);
    };
    let Some(concrete_seq) = walker_concrete_ref_object(ctx, seq) else {
        return Ok(None);
    };
    // `objspace.py:507-541 StdObjSpace.{unpackiterable,fixedview}` takes an
    // exact tuple straight to its immutable `wrappeditems` list; and
    // `pyopcode.py UNPACK_SEQUENCE` calls `fixedview_unroll`, so a
    // constant item count exposes each tuple item directly to the trace.
    // Preserve that shape for pyre's split `UnpackSequence` / `UnpackItem`
    // helpers.  In particular, `zip` produces an ordinary array-backed tuple
    // each iteration; leaving these helpers residual makes the three-item
    // comprehension trace execute one validation call plus one call per
    // projected item.
    let tuple_type = &pyre_object::pyobject::TUPLE_TYPE as *const pyre_object::pyobject::PyType;
    let canonical_tuple_class = pyre_object::pyobject::get_instantiate(unsafe { &*tuple_type });
    if unsafe {
        std::ptr::eq((*concrete_seq).ob_type, tuple_type)
            && std::ptr::eq((*concrete_seq).w_class, canonical_tuple_class)
    } {
        let concrete_len = unsafe { pyre_object::w_tuple_len(concrete_seq) };
        if int_val < 0 {
            return Ok(None);
        }
        walker_guard_exact_instance(ctx, op_pc, seq, tuple_type as i64, canonical_tuple_class)?;
        let items = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq,
            crate::descr::tuple_wrappeditems_descr(),
        );
        match helper {
            majit_ir::RuntimeHelperKind::UnpackSequence => {
                if int_val as usize != concrete_len {
                    return Ok(None);
                }
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    items,
                    crate::state::pyobject_gcarray_descr(),
                );
                let expected = ctx.trace_ctx.const_int(int_val);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, expected],
                )?;
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
                return Ok(Some(()));
            }
            majit_ir::RuntimeHelperKind::UnpackItem => {
                let index = int_val as usize;
                if index >= concrete_len {
                    return Ok(None);
                }
                // `index < concrete_len` only holds for the tuple recorded
                // here. A trace can enter between the `UnpackSequence` helper
                // and this one — a bridge resumes mid-unpack — so this arm
                // cannot rely on that helper's length guard being in the same
                // trace. Emit it; when it is present the optimizer folds this
                // one away, so a whole unpack still guards the length once.
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    items,
                    crate::state::pyobject_gcarray_descr(),
                );
                let expected = ctx.trace_ctx.const_int(concrete_len as i64);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardValue,
                    &[length, expected],
                )?;
                let index_op = ctx.trace_ctx.const_int(int_val);
                let item = crate::state::trace_items_block_getitem_value_pure(
                    ctx.trace_ctx,
                    items,
                    index_op,
                );
                let concrete_item = unsafe {
                    pyre_object::w_tuple_getitem(concrete_seq, int_val)
                        .unwrap_or(pyre_object::PY_NULL)
                };
                ctx.trace_ctx.set_opref_concrete(
                    item,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_item as usize)),
                );
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
                return Ok(Some(()));
            }
            _ => {}
        }
    }
    // Both arity-2 specialisations fold; any other shape (a plain tuple, a
    // list, or a non-canonical tuple) falls through to the opaque residual
    // (correct, slower).
    let seq_type = unsafe { (*concrete_seq).ob_type };
    let Some(pair_kind) = specialised_pair_kind(seq_type) else {
        return Ok(None);
    };
    let spec_type = seq_type;
    match helper {
        majit_ir::RuntimeHelperKind::UnpackSequence => {
            // Either specialisation is always arity 2, so the class guard
            // subsumes the exact-length check `unpack_sequence_fn` performs.
            if int_val != 2 {
                return Ok(None);
            }
            walker_guard_specialised_pair_class(ctx, op_pc, seq, spec_type)?;
            // Pass `seq` through as the validated tuple; the per-index
            // `unpack_item_fn` reads below fold off it.
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, seq)?;
            Ok(Some(()))
        }
        majit_ir::RuntimeHelperKind::UnpackItem => {
            if !(0..2).contains(&int_val) {
                return Ok(None);
            }
            // Normally the partner `unpack_sequence_fn` fold already guarded
            // the class (its validated-tuple passthrough reg == `seq`), in
            // which case this is a no-op; guard here too so a fold that only
            // catches the item reads still proves the layout it loads from.
            walker_guard_specialised_pair_class(ctx, op_pc, seq, spec_type)?;
            let Some(item) = walker_emit_specialised_pair_item(
                ctx, op_pc, seq, pair_kind, int_val, allboxes, call_descr,
            )?
            else {
                return Ok(None);
            };
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, item)?;
            Ok(Some(()))
        }
        _ => Ok(None),
    }
}

/// The tuple `descr_getargs` would build from an exception's `args_w` list.
/// Built only to read the representation `newtuple` picks off it, so the caller
/// keeps the shape and drops the tuple.
unsafe fn args_tuple_shape_probe(stored: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    let len = unsafe { pyre_object::interp_exceptions::rlist_len(stored) };
    let items = (0..len)
        .map(|index| unsafe { pyre_object::interp_exceptions::rlist_getitem(stored, index) })
        .collect();
    pyre_object::w_tuple_new(items)
}

/// `guard_class(seq, spec)` for one of the arity-2 tuple specialisations,
/// emitted once per traced `seq` — the heap cache turns every later fold on the
/// same register into a no-op, the way upstream's optimizer keeps a single
/// class guard for a value it already proved.
fn walker_guard_specialised_pair_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq: OpRef,
    spec_type: *const pyre_object::pyobject::PyType,
) -> Result<(), DispatchError> {
    walker_guard_fold_class_if_unknown(ctx, op_pc, seq, spec_type as i64)
}

/// Read slot `index` (0 or 1) of an arity-2 tuple specialisation whose class
/// the caller has already guarded, applying that slot's `wraps[i]`
/// (`specialisedtupleobject.py`, and `:134-142 getitem`, which unrolls
/// `iter_n` to the matching `value%s`).
///
/// `Ok(None)` declines: the `ii` / `ff` slots need the authentic box for its
/// identity, and that execution can fail.
///
/// The `ff` arm currently has no producer to serve. Upstream builds `Cls_ff`
/// from `makespecialisedtuple2` (`specialisedtupleobject.py`) and from
/// `specialized_zip_2_lists` (`:230`); pyre does not port the latter, and
/// `w_tuple_new` (`tupleobject.rs`) sends a plain-float pair to `Cls_oo`
/// instead so that `(x, x)` keeps the exact `x` object. It is kept because it
/// is the layout upstream reads, not because a trace reaches it today.
#[allow(clippy::too_many_arguments)]
fn walker_emit_specialised_pair_item<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq: OpRef,
    pair_kind: SpecialisedPairKind,
    index: i64,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
) -> Result<Option<OpRef>, DispatchError> {
    let first = index == 0;
    if pair_kind == SpecialisedPairKind::Object {
        let descr = if first {
            crate::descr::specialised_tuple_oo_value0_descr()
        } else {
            crate::descr::specialised_tuple_oo_value1_descr()
        };
        // `wraps[i]` is the identity for an object slot, so the field read is
        // the whole item — no re-boxing, and no `may_force` execution needed
        // to recover a box identity.
        return Ok(Some(crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            seq,
            descr,
        )));
    }
    // Authentic boxed element supplies the concrete shadow / identity while
    // the emitted field read and transparent wrapper replace the residual call
    // in machine code. Fall through if execution raises or cannot provide that
    // box.
    let Some(elem_ptr) = walker_execute_may_force_boxed(ctx, allboxes, call_descr) else {
        return Ok(None);
    };
    if pair_kind == SpecialisedPairKind::Float {
        let descr = if first {
            crate::descr::specialised_tuple_ff_value0_descr()
        } else {
            crate::descr::specialised_tuple_ff_value1_descr()
        };
        let raw = majit_metainterp::box_trace::getfield_gc_f_pureornot(ctx.trace_ctx, seq, descr);
        let elem = unsafe { pyre_object::w_float_get_value(elem_ptr as pyre_object::PyObjectRef) };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Float(elem));
        let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(elem_ptr as usize)),
        );
        return Ok(Some(boxed));
    }
    // `ii`: preserve authentic small-int caching / identity.
    let descr = if first {
        crate::descr::specialised_tuple_ii_value0_descr()
    } else {
        crate::descr::specialised_tuple_ii_value1_descr()
    };
    let raw = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, seq, descr);
    let elem = unsafe { pyre_object::w_int_get_value(elem_ptr as pyre_object::PyObjectRef) };
    let boxed = walker_box_int(ctx, op_pc, raw, elem)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(elem, elem_ptr as i64));
    Ok(Some(boxed))
}

/// One hop of a `while tb is not None: names.append(tb.tb_frame.f_code.co_name);
/// tb = tb.tb_next` traceback walk.
///
/// Each of these is a `GetSetProperty` whose getter body is a slot read on a
/// receiver [`walker_specialize_traceback_walk_field`] pins by class:
/// `pytraceback.py descr_get_next` / `descr_get_tb_frame` /
/// `descr_get_tb_lineno` / `descr_get_tb_lasti` and `pyframe.py fget_code`.
/// None of them dispatches anywhere or can raise.  Left residual, every hop of
/// the walk costs a forcing call — measured at 207 ns per `tb_lineno` read
/// against 0 for the folded `tb_next` — which is what makes each traceback
/// fixture dominated by the walk rather than by the raise.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TracebackWalkField {
    /// `tb.tb_next` — the chain link; a null slot is the terminator and
    /// surfaces as `None`.
    TbNext,
    /// `tb.tb_frame` — the node's frame.  Unlike its two siblings the getter
    /// ALSO runs `mark_as_escaped()`; see the escape emit.
    TbFrame,
    /// `frame.f_code` — `fget_f_code` is `self.pycode as PyObjectRef`.
    FCode,
    /// `tb.tb_lineno` — the line the node froze at.  The getter resolves the
    /// sentinel out of `w_code` and `lasti`; `record_application_traceback`
    /// stamps the real line instead, so a recorded node reads as the slot and
    /// only a hand-constructed one has to resolve.  The fold covers the stamped
    /// case and declines the other.
    ///
    /// `tb_lasti` is deliberately absent: it is the one traceback slot the
    /// walker has no reason to reach, since nothing on the walk consumes it.
    TbLineno,
    /// `code.co_name` — `code_get_field` answers it with `w_code_name_obj`,
    /// which realizes the string once and retains it on the code object, so
    /// every later read is the retained slot.
    CoName,
    /// `code.co_firstlineno` — the `co_firstlineno_raw` slot, reboxed.  The
    /// other code fields are deliberately absent: they read the host
    /// `CodeObject` behind `code_ptr` rather than a slot on the `PyCode`, so
    /// folding one means a raw load through a second indirection.
    CoFirstlineno,
}

/// Which walk hop, if any, this `(receiver, attribute)` pair is.
fn traceback_walk_field(
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
) -> Option<TracebackWalkField> {
    let ob_type = unsafe { (*concrete_obj).ob_type };
    if std::ptr::eq(ob_type, &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE) {
        return match name {
            "tb_next" => Some(TracebackWalkField::TbNext),
            "tb_frame" => Some(TracebackWalkField::TbFrame),
            "tb_lineno" => Some(TracebackWalkField::TbLineno),
            _ => None,
        };
    }
    if std::ptr::eq(ob_type, &pyre_interpreter::pyframe::FRAME_TYPE) && name == "f_code" {
        return Some(TracebackWalkField::FCode);
    }
    if std::ptr::eq(ob_type, &pyre_interpreter::pycode::CODE_TYPE) {
        return match name {
            "co_name" => Some(TracebackWalkField::CoName),
            "co_firstlineno" => Some(TracebackWalkField::CoFirstlineno),
            _ => None,
        };
    }
    None
}

/// Runtime half of the optimized-frame `f_locals` getter.  The proxy owns the
/// exact frame passed to it; reading or mutating the proxy later goes through
/// that frame's existing synchronization path.
extern "C" fn jit_inline_frame_locals_proxy_new(
    frame: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    pyre_interpreter::pyframe::frame_locals_proxy::new(frame)
}

/// The frame and its `locals_cells_stack_w` were both allocated in this trace,
/// and `heapcache` still calls both current (`new` / `new_array`,
/// `saw_allocation`).
///
/// Those flags die at `reset_keep_likely_virtuals`, so a residual between the
/// allocation and this read declines.  While they hold, the array is the
/// `NewArrayClear` this trace stored into the frame: nothing unpublished sits
/// beside it.  A frame the trace did not allocate — the catching portal — keeps
/// its except-bound name in the virtualizable, and `jit_force_virtualizable`
/// is what publishes that name.
fn frame_locals_heap_is_trace_allocation<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame: OpRef,
) -> bool {
    if frame.is_constant() || !ctx.trace_ctx.heap_cache().saw_allocation(frame) {
        return false;
    }
    let index = crate::descr::pyframe_locals_cells_stack_descr().index();
    let Some(array) = ctx.trace_ctx.heapcache_getfield_cached(frame, index) else {
        return false;
    };
    !array.is_constant() && ctx.trace_ctx.heap_cache().saw_allocation(array)
}

/// Prove the receiving code object still owns its host `CodeObject`, the
/// `require_code` check every code-field getter runs before reading a slot.
fn walker_guard_code_ptr_present<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    let code_ptr = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        obj,
        crate::descr::pycode_code_ptr_descr(),
    );
    let live = unsafe { pyre_interpreter::w_code_get_ptr(concrete_obj) } as i64;
    ctx.trace_ctx
        .set_opref_concrete(code_ptr, majit_ir::Value::Int(live));
    let zero = ctx.trace_ctx.const_int(0);
    let absent = ctx.trace_ctx.record_op(OpCode::IntEq, &[code_ptr, zero]);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[absent])
}

/// The line `w_pytraceback_get_lineno` resolves for a node whose `lineno`,
/// `lasti` and `w_code` slots are all trace constants in the heap cache —
/// the node `emit_traceback_node` built in this trace, whose stores are the
/// cached values.  `None` when any slot is not a known constant, when the
/// stored `lineno` is not the sentinel, or when `tb_lasti` names no line (the
/// getter answers `None` there, which only the residual reproduces).
fn walker_traceback_lineno_from_trace_constants<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    obj: OpRef,
) -> Option<i64> {
    let mut cached_const = |index: usize| {
        let descr = crate::descr::pytraceback_field_descr(index);
        let value = ctx
            .trace_ctx
            .heapcache_getfield_cached(obj, descr.index())?;
        ctx.trace_ctx.const_value(value)
    };
    // `emit_traceback_node`'s field order: 1 = `lasti`, 3 = `lineno`,
    // 4 = `w_code`.
    if cached_const(3)? != pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED {
        return None;
    }
    let lasti = cached_const(1)?;
    let w_code = cached_const(4)? as pyre_object::PyObjectRef;
    if w_code.is_null() || !unsafe { pyre_interpreter::pycode::is_code(w_code) } {
        return None;
    }
    let lineno = unsafe { pyre_interpreter::pycode::w_code_addr2line(w_code, lasti) };
    (lineno >= 0).then_some(lineno as i64)
}

/// Emit one traceback-walk hop as a guarded inline field read instead of the
/// opaque `getattr_fn` residual.
///
/// Returns `None` (fall through to the residual) BEFORE recording any guard for
/// every shape it cannot settle — an uncacheable `version_tag`, or a null slot
/// on a hop whose null is not a documented value.  A bail-out after a guard
/// would leave the caller reading the attribute as already pinned.
fn walker_specialize_traceback_walk_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    field: TracebackWalkField,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::pyframe::PyFrame;

    let receiver_type = match field {
        TracebackWalkField::FCode => &pyre_interpreter::pyframe::FRAME_TYPE,
        TracebackWalkField::CoName | TracebackWalkField::CoFirstlineno => {
            &pyre_interpreter::pycode::CODE_TYPE
        }
        _ => &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE,
    };
    let descr = match field {
        TracebackWalkField::TbNext => crate::descr::pytraceback_w_next_descr(),
        TracebackWalkField::TbFrame => crate::descr::pytraceback_frame_descr(),
        TracebackWalkField::FCode => crate::descr::pyframe_code_descr(),
        TracebackWalkField::TbLineno => crate::descr::pytraceback_lineno_descr(),
        TracebackWalkField::CoName => crate::descr::pycode_w_name_descr(),
        TracebackWalkField::CoFirstlineno => crate::descr::pycode_co_firstlineno_descr(),
    };
    let w_type = pyre_interpreter::typedef::gettypeobject(receiver_type);
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return Ok(None);
    }
    // The slot guard pins the receiver's `w_class` against `w_type`.  A frame
    // built before `init_typeobjects` carries a null `w_class`, which would
    // make that guard fail on its first execution, so decline instead of
    // recording a doomed trace.
    if unsafe { (*concrete_obj).w_class } != w_type {
        return Ok(None);
    }

    // Every code-field getter resolves the host `CodeObject` first
    // (`code_get_field` -> `require_code`) and raises when it is absent, so a
    // code fold owes that check.  It is a slot on the receiver, so the trace
    // proves it the same way — a read plus a non-null guard — rather than
    // trusting the record-time object.
    let code_receiver = matches!(
        field,
        TracebackWalkField::CoName | TracebackWalkField::CoFirstlineno
    );
    if code_receiver
        && unsafe { pyre_interpreter::w_code_get_ptr(concrete_obj) }
            .cast::<u8>()
            .is_null()
    {
        return Ok(None);
    }

    if field == TracebackWalkField::CoFirstlineno {
        let live =
            i64::from(unsafe { pyre_interpreter::pycode::w_code_firstlineno_raw(concrete_obj) });
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        walker_guard_code_ptr_present(ctx, op_pc, obj, concrete_obj)?;
        let raw_value = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, descr);
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Int(live));
        // Reboxed for the same reason `TbLineno` is: the getter hands back a
        // Python int, and the boxed op is a heap `NewWithVtable`.
        let boxed = walker_box_int(ctx, op_pc, raw_value, live)?;
        let live_ptr = pyre_object::w_int_new(live) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
        return Ok(Some(()));
    }

    if field == TracebackWalkField::TbLineno {
        let live =
            unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_lineno_raw(concrete_obj) };
        // A slot holding the sentinel is not the getter's value — the getter
        // resolves it out of `w_code` and `lasti`.  A node this trace built
        // (`emit_traceback_node`) carries all three as trace constants, and
        // resolving an immutable code object's line table at a constant
        // offset is the value the getter's resolution produces, so that node
        // folds to the resolved line.  Any other sentinel node — recorded by
        // the interpreter, or handed `-1` through `TracebackType(...)` — has
        // slots this fold would have to pin, so it declines before recording
        // anything and the getter runs as a residual.
        if live == pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED {
            let Some(resolved) = walker_traceback_lineno_from_trace_constants(ctx, obj) else {
                return Ok(None);
            };
            walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
            let raw_value = ctx.trace_ctx.const_int(resolved);
            let boxed = walker_box_int(ctx, op_pc, raw_value, resolved)?;
            let live_ptr = pyre_object::w_int_new(resolved) as i64;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(resolved, live_ptr));
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
            return Ok(Some(()));
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let raw_value = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, obj, descr);
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Int(live));
        let not_computed = ctx
            .trace_ctx
            .const_int(pyre_interpreter::pytraceback::LINENO_NOT_COMPUTED);
        let is_not_computed = ctx
            .trace_ctx
            .record_op(OpCode::IntEq, &[raw_value, not_computed]);
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[is_not_computed])?;
        // The getter returns a Python int, so the raw slot is reboxed the way
        // the unboxed mapdict read does; the boxed op is a heap `NewWithVtable`
        // so its concrete has to be a heap pointer too.
        let boxed = walker_box_int(ctx, op_pc, raw_value, live)?;
        let live_ptr = pyre_object::w_int_new(live) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
        return Ok(Some(()));
    }

    let stored = match field {
        TracebackWalkField::TbNext => unsafe {
            pyre_interpreter::pytraceback::w_pytraceback_get_w_next(concrete_obj)
        },
        TracebackWalkField::TbFrame => {
            (unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(concrete_obj) })
                as pyre_object::PyObjectRef
        }
        // `w_name` is realized on first demand, so an unread code object
        // carries a null here; that declines below and the residual realizes
        // it for the next attempt.
        TracebackWalkField::CoName => unsafe {
            (*(concrete_obj as *const pyre_interpreter::pycode::PyCode)).w_name
        },
        _ => (unsafe { (*(concrete_obj as *const PyFrame)).pycode }) as pyre_object::PyObjectRef,
    };
    // Only `tb_next` has a null with a defined meaning.  A null frame is a
    // torn-down traceback and a null `pycode` a half-built frame; both are
    // answered by a `sys.namespace` stub or `None` the residual owns.
    if stored.is_null() && field != TracebackWalkField::TbNext {
        return Ok(None);
    }
    walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
    if code_receiver {
        walker_guard_code_ptr_present(ctx, op_pc, obj, concrete_obj)?;
    }
    let raw_value = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, descr);
    let value = if stored.is_null() {
        // End of the chain.  A nullity test is `pyjitpl.py
        // _establish_nullity`'s GUARD_ISNULL plus a `replace_box` onto the null
        // constant `constant_from_op` gives it — not a promote.  The
        // distinction is load-bearing: `compile.py
        // make_a_counter_per_value` keys a GUARD_VALUE's jitcounter on the
        // *failing value*, and this slot holds a different PyTraceback on every
        // walk, so no one value here ever reaches `trace_eagerness` and the
        // continuation for a non-null link never gets a bridge.
        // Stamped ahead of the guard because `stamp_guard_value_concrete` only
        // does it for a GUARD_VALUE, and the snapshot the guard captures reads
        // the slot's concrete.
        ctx.trace_ctx
            .set_opref_concrete(raw_value, majit_ir::Value::Ref(majit_ir::GcRef(0)));
        walker_guard_stamped_isnull(ctx, op_pc, raw_value)?;
        ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
    } else {
        walker_guard_stamped_nonnull(ctx, op_pc, raw_value, stored)?;
        raw_value
    };

    if field == TracebackWalkField::TbFrame {
        // `descr_get_tb_frame` also runs `frame.mark_as_escaped()`
        // (`pyframe.py mark_as_escaped`): the reference it hands out has to
        // keep the frame materialised.  `set_escaped` ORs `FLAG_ESCAPED` into
        // the `flags` byte, so the trace reads that byte, sets the bit, and
        // stores it back.
        //
        // The bit has to be set BY THE TRACE, not only stamped now: the trace
        // is reused, and each replay walks a different traceback naming a
        // different frame, so a trace-time-only mark would leave every later
        // frame unmarked.  The concrete write below is the one the
        // authoritative walk's residual executor would have performed, applied
        // here for the same reason `try_walker_lower_exc_info_residual` applies
        // its own.
        let flags_descr = crate::descr::pyframe_flags_descr();
        let live_flags =
            crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, raw_value, flags_descr.clone());
        let escaped_bit = ctx.trace_ctx.const_int(i64::from(PyFrame::FLAG_ESCAPED));
        let new_flags = ctx
            .trace_ctx
            .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[raw_value, new_flags],
            flags_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(raw_value, flags_descr.index(), new_flags);
        unsafe { (*(stored as *mut PyFrame)).mark_as_escaped() };
    }

    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// Write the standard virtualizable's locals region into the array a folded
/// `f_locals` hands out.
///
/// `pyframe.py fast2locals` — the body behind `getdictscope`, and so behind
/// `f_locals` — is `@jit.unroll_safe`, and the `locals_cells_stack_w[i]` reads
/// it unrolls are `getarrayitem_vable_r` against the virtualizable BOXES.  The
/// getter therefore neither forces the virtualizable nor reads its array, which
/// is what makes folding it legitimate at all.
///
/// pyre answers `f_locals` with the 3.14 `FrameLocalsProxy`, which reads the
/// frame's array lazily instead of copying out of it at the call, so the values
/// have to be IN that array by the time the proxy is handed out.
/// `pyjitpl.py synchronize_virtualizable` (`virtualizable.py write_boxes`) is
/// the write-back that puts them there.  Upstream runs it against the
/// recording-time virtualizable after every vable store; both halves are
/// needed here, because the values have to be in the array for the walk's own
/// read of the proxy AND for the compiled run's.  So this mirrors the region
/// onto the concrete frame and emits the same store into the trace.  Without
/// it the residual getter's read barrier was the only thing writing the region
/// out, and folding the getter silently dropped every local the traced body
/// had assigned.
///
/// Only the locals/cells region is written back.  `write_boxes` covers the
/// whole array, but the operand-stack region above `nlocals` is not reachable
/// through the proxy and its shadow slots read NULL outside a merge point
/// (see [`crate::state::flush_locals_region_to_frame`]), so writing those back
/// would destroy the values the walk is holding.
///
/// A slot the shadow cannot answer declines the whole write-back, and with it
/// the fold, leaving the residual force in place.  The validation pass runs
/// before the first emission, so a decline emits nothing.
pub(crate) fn walker_write_back_standard_frame_locals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame_op: OpRef,
    concrete_frame: usize,
) -> bool {
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return false;
    };
    let base = info.num_static_extra_boxes;
    let Some(nlocals) = crate::state::concrete_nlocals(concrete_frame) else {
        return false;
    };
    // `Value::Void` is the shadow's "no concrete half" sentinel rather than an
    // unbound local, so a slot carrying it cannot be written back.
    let mut slots = Vec::with_capacity(nlocals);
    for slot in 0..nlocals {
        match ctx.trace_ctx.virtualizable_entry_at(base + slot) {
            Some((_, majit_ir::Value::Void)) | None => return false,
            Some((value, _)) => slots.push((slot as i64, value)),
        }
    }
    // The mirror below writes the live frame's locals array, and a walk that
    // does not commit replays from its pre-walk instruction — so the pre-walk
    // values have to be recoverable.  Journal them against the walk's own
    // non-commit epilogue rather than the escape-flush capture: that capture is
    // consumed by every non-forcing residual (`try_execute_residual_call_via_
    // executor`'s tail restore), which would revert this mirror mid-walk and
    // leave a live `FrameLocalsProxy` reading pre-fold values.
    crate::jitcode_dispatch::fbw_note_locals_mirror_undo(concrete_frame, nlocals);
    if !crate::state::flush_locals_region_to_frame(ctx.trace_ctx, concrete_frame) {
        // All-or-nothing decline: nothing was written.  The journal entry is
        // harmless — restoring the values still in place is a no-op — and the
        // first-per-frame rule means dropping it could discard a real one.
        return false;
    }
    ctx.trace_ctx
        .vable_array_region_write_back(frame_op, 0, &slots)
}

/// Publish every local the proxy can observe.
///
/// `fast2locals` reads `locals_cells_stack_w[i]` for every varname. A shadow
/// slot whose concrete half is `Void` still has a box; omitting it drops the
/// name. Slots the shadow does not carry are read off this frame object.
///
/// Boxing an `Int`/`Float` slot allocates and can move a nursery frame, so
/// each store continues from the address the previous one returned, and the
/// frame's current address is returned to the caller.
fn walker_publish_complete_frame_locals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    frame_op: OpRef,
    concrete_frame: usize,
    source: &ReceiverTraceLocals,
) -> usize {
    let mut concrete_frame = concrete_frame;
    let Some(nlocals) = crate::state::concrete_nlocals(concrete_frame) else {
        return concrete_frame;
    };
    let info = ctx.trace_ctx.virtualizable_info().cloned();
    let mut unread: Vec<usize> = Vec::new();
    let inline_slots = match source {
        ReceiverTraceLocals::Inline(slots) => Some(slots.as_slice()),
        ReceiverTraceLocals::Portal => None,
    };
    if let Some(slots) = inline_slots {
        let mut trace_slots = Vec::with_capacity(nlocals);
        for slot in 0..nlocals {
            match slots.iter().find(|(index, _, _)| *index == slot as i64) {
                Some((_, opref, value)) if !opref.is_none() => {
                    let concrete = match *value {
                        majit_ir::Value::Void => ctx
                            .trace_ctx
                            .concrete_of_opref(*opref)
                            .unwrap_or(majit_ir::Value::Void),
                        majit_ir::Value::Ref(gc) if gc == majit_ir::GcRef::NO_CONCRETE => {
                            ctx.trace_ctx.concrete_of_opref(*opref).unwrap_or(*value)
                        }
                        other => other,
                    };
                    if let Some(frame_now) =
                        crate::state::store_frame_local_value(concrete_frame, slot, &concrete)
                    {
                        concrete_frame = frame_now;
                    }
                    trace_slots.push((slot as i64, *opref));
                }
                _ => unread.push(slot),
            }
        }
        if !trace_slots.is_empty() {
            ctx.trace_ctx
                .vable_array_region_write_back(frame_op, 0, &trace_slots);
        }
    } else if let Some(info) = info.as_ref() {
        let base = info.num_static_extra_boxes;
        let mut trace_slots = Vec::with_capacity(nlocals);
        for slot in 0..nlocals {
            match ctx.trace_ctx.virtualizable_entry_at(base + slot) {
                Some((opref, value)) if !opref.is_none() => {
                    let concrete = match value {
                        majit_ir::Value::Void => ctx
                            .trace_ctx
                            .concrete_of_opref(opref)
                            .unwrap_or(majit_ir::Value::Void),
                        majit_ir::Value::Ref(gc) if gc == majit_ir::GcRef::NO_CONCRETE => {
                            ctx.trace_ctx.concrete_of_opref(opref).unwrap_or(value)
                        }
                        other => other,
                    };
                    // A `Void` or null half is not a value to write over the
                    // live frame. The slot stays a read of that frame.
                    let real = match concrete {
                        majit_ir::Value::Ref(gc) => {
                            gc != majit_ir::GcRef::NO_CONCRETE && gc.as_usize() != 0
                        }
                        majit_ir::Value::Int(_) | majit_ir::Value::Float(_) => true,
                        majit_ir::Value::Void => false,
                    };
                    if real {
                        if let Some(frame_now) =
                            crate::state::store_frame_local_value(concrete_frame, slot, &concrete)
                        {
                            concrete_frame = frame_now;
                        }
                        trace_slots.push((slot as i64, opref));
                    } else {
                        unread.push(slot);
                    }
                }
                _ => unread.push(slot),
            }
        }
        crate::jitcode_dispatch::fbw_note_locals_mirror_undo(concrete_frame, nlocals);
        if !trace_slots.is_empty() {
            ctx.trace_ctx
                .vable_array_region_write_back(frame_op, 0, &trace_slots);
        }
    } else {
        unread.extend(0..nlocals);
    }
    if unread.is_empty() {
        return concrete_frame;
    }
    let Some(info) = info else {
        return concrete_frame;
    };
    if info.array_fields.is_empty() {
        return concrete_frame;
    }
    let field = info.array_pointer_field_descr(0);
    let adescr = info.array_item_descr(0);
    let array = ctx.trace_ctx.vable_getfield_ref_descr(frame_op, field);
    for slot in unread {
        let _ = ctx
            .trace_ctx
            .read_gc_array_item_ref(array, slot as i64, adescr.clone());
    }
    concrete_frame
}

/// The frame box and EXECUTING Python pc of a frame receiver the walk owns,
/// or `None` for one it does not.
///
/// `pyjitpl.py` keeps one MIFrame per inlined call and each carries its own
/// coordinate, so this resolves per level exactly as
/// [`LiveLastInstrGuard::enter`] retargets its publication: inside an inline
/// sub-walk the callee's own jitcode pc resolved through the callee's
/// metadata, at the portal the walk's `vstack_cur_pypc`.
///
/// The virtualizable boxes are NOT a source here, which a probe against the
/// residual's own answers settled rather than an argument: at three portal
/// sites the boxes' `last_instr` entry read 110/165/185 against executing
/// pcs of 115/170/197, and at an inlined-callee site it read 232 -- the
/// caller's CALL boundary -- against the callee's own 22.  They describe the
/// PORTAL frame and carry whichever of the field's two conventions their last
/// writer left.
///
/// A receiver that is not this level's own frame declines, which is what keeps
/// a suspended generator's frame, a traceback node's frame and a caller's
/// frame read from inside a callee on the residual getter that reads the heap.
/// The box and concrete address of the frame the walk is executing.
///
/// Two sources, chosen by whether a sub-walk is active, because they describe
/// different frames: the portal's virtualizable describes the PORTAL, so an
/// inlined callee has to answer from its own shadow or it reports its caller's.
fn walker_executing_frame_box<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(OpRef, usize)> {
    let inline_frame = current_inline_concrete_frame();
    if inline_frame != 0 {
        let state = ctx.frame_state.borrow();
        let shadow = state.callee_shadow.as_ref()?;
        if shadow.concrete_frame != inline_frame || shadow.frame_box == OpRef::NONE {
            return None;
        }
        return Some((shadow.frame_box, inline_frame));
    }
    if ctx.fbw_mode.inline_subwalk {
        return None;
    }
    let vable_box = ctx.trace_ctx.standard_virtualizable_box()?;
    let vable_ptr = ctx.trace_ctx.standard_virtualizable_ptr()?;
    Some((vable_box, vable_ptr))
}

/// Concrete immediate caller of the inlined level now executing, and the red
/// box that names it: the standard virtualizable when that caller is the
/// portal, otherwise the virtual box `walker_ec_enter` published for the
/// ancestor.
fn walker_immediate_inline_caller_box<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(OpRef, usize)> {
    let inline = current_inline_concrete_frame();
    if inline == 0 {
        return None;
    }
    let raw = unsafe { (*(inline as *const pyre_interpreter::PyFrame)).f_backref };
    if raw.is_null() {
        return None;
    }
    let caller_ptr =
        if unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(raw as *const u8) } {
            let referent = unsafe {
                majit_metainterp::virtualref::vref_forced(raw as *const u8)
                    as *mut pyre_interpreter::PyFrame
            };
            if referent.is_null() {
                return None;
            }
            referent as usize
        } else {
            raw as usize
        };
    if let (Some(vable_box), Some(vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) && vable_ptr == caller_ptr
    {
        return Some((vable_box, caller_ptr));
    }
    let caller_box = ctx
        .trace_ctx
        .virtualref_virtual_for_object_ptr(caller_ptr)?;
    Some((caller_box, caller_ptr))
}

/// A paused inlined ancestor stores the CALL that entered its child on
/// `InlineParentFrame.caller_py_pc`. Walk `framestack` the way
/// `MetaInterp.replace_box` walks `MIFrame`s.
fn walker_paused_ancestor_py_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    landing_ptr: usize,
) -> Option<(OpRef, u32)> {
    let session = ctx.session.borrow();
    for frame in &session.framestack {
        for parent in &frame.parents {
            let Some(py_pc) = parent.caller_py_pc else {
                continue;
            };
            let Some(ptr) = parent.paused_concrete_frame() else {
                continue;
            };
            if ptr != landing_ptr {
                continue;
            }
            let red = ctx
                .trace_ctx
                .virtualref_virtual_for_object_ptr(ptr)
                .or_else(|| {
                    (ctx.trace_ctx.standard_virtualizable_ptr() == Some(ptr))
                        .then(|| ctx.trace_ctx.standard_virtualizable_box())
                        .flatten()
                })?;
            return Some((red, py_pc));
        }
    }
    None
}

fn walker_frame_executing_py_pc<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    concrete_obj: pyre_object::PyObjectRef,
    op_pc: usize,
) -> Option<(OpRef, u32)> {
    // A traceback can expose an inline frame after exception propagation has
    // finished it while the enclosing sub-walk guard is still live.  PyPy no
    // longer has an executing MIFrame coordinate for that object: `f_lasti`
    // reads the value `handle_operation_error` left on the finished frame.
    if unsafe { &*(concrete_obj as *const pyre_interpreter::PyFrame) }.frame_finished_execution() {
        return None;
    }
    // The portal stays at the outermost CALL (`inline_caller_py_pc`).
    // An intermediate inlined caller stays at the CALL that entered THIS
    // level (`immediate_inline_caller_py_pc`), which nested inlines do not
    // inherit.
    if let (Some(vable_box), Some(vable_ptr), Some(caller_py_pc)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
        ctx.fbw_mode.inline_caller_py_pc,
    ) && vable_ptr == concrete_obj as usize
    {
        return Some((vable_box, caller_py_pc));
    }
    if let Some(caller_py_pc) = ctx.fbw_mode.immediate_inline_caller_py_pc
        && let Some((caller_box, caller_ptr)) = walker_immediate_inline_caller_box(ctx)
        && caller_ptr == concrete_obj as usize
    {
        return Some((caller_box, caller_py_pc));
    }
    if let Some(found) = walker_paused_ancestor_py_pc(ctx, concrete_obj as usize) {
        return Some(found);
    }
    let (frame_box, frame_ptr) = walker_executing_frame_box(ctx)?;
    if frame_ptr != concrete_obj as usize {
        return None;
    }
    if current_inline_concrete_frame() != 0 {
        return Some((frame_box, residual_call::inline_callee_py_pc(ctx, op_pc)?));
    }
    // An inline sub-walk with no concrete callee frame has no level-local
    // coordinate to answer with: `vstack_cur_pypc` is the outer walk's mirror
    // and a sub-walk never advances it.
    if !ctx.vstack_valid {
        return None;
    }
    Some((frame_box, ctx.vstack_cur_pypc))
}

/// Pin a frame-attribute receiver to the frame type's getset.
///
/// Its class, its `w_class` and the frame type's `version_tag` are guarded, so
/// rebinding the getset on the type revokes the loop instead of the fold
/// outliving the descriptor that produced it.  This is the half of
/// [`walker_prove_owned_frame_pc`] that does not ask whether the receiver is
/// the frame the walk is executing: `f_back` records a field on the object,
/// and `pyframe.py get_f_back` derefs `f_backref` without forcing, so a
/// receiver that merely *is* a frame still owes these guards.
fn walker_guard_frame_attr_receiver<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    let w_type = pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE);
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 || unsafe { (*concrete_obj).w_class } != w_type {
        return Ok(false);
    }
    walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
    Ok(true)
}

/// Prove the receiver IS the frame the walk is executing, and answer that
/// frame's executing pc.
///
/// Shared by the two owned-frame getter folds, which answer a coordinate the
/// walk holds rather than the one the frame's own field records and therefore
/// owe the same proof about the object in hand.
///
/// The receiver is pinned two ways.  [`walker_guard_frame_attr_receiver`]
/// guards the type.  And when the receiver arrives in a box other than the
/// frame's own — a local the loop hoisted the frame into — a `ptr_eq` against
/// that box is guarded, so a later entry holding a different frame side-exits
/// to the residual rather than reading this trace's coordinate.  That second
/// pin is only meaningful once [`walker_frame_executing_py_pc`] has named the
/// frame this walk is executing.
fn walker_prove_owned_frame_pc<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
) -> Result<Option<u32>, DispatchError> {
    let Some((frame_box, py_pc)) = walker_frame_executing_py_pc(ctx, concrete_obj, op_pc) else {
        return Ok(None);
    };
    if !walker_guard_frame_attr_receiver(ctx, op_pc, obj, concrete_obj)? {
        return Ok(None);
    }
    if obj != frame_box {
        walker_guard_stamped_ptr_eq(ctx, op_pc, obj, frame_box)?;
    }
    Ok(Some(py_pc))
}

/// `pyframe.py fget_f_lasti` — `return self.last_instr`, loop-free and
/// carrying no hint, so `policy.py look_inside_graph` admits it, `jtransform.py
/// rewrite_op_jit_force_virtualizable` deletes the force
/// `rvirtualizable.py hook_access_field` injects, and `pyjitpl.py
/// opimpl_getfield_vable_i` answers the field out of `virtualizable_boxes`.
/// The box there is a `ConstInt`, because the only writer is the bytecode
/// dispatch's `_opimpl_setfield_vable` of the pc it is about to run — which is
/// why upstream answers the read for less than a loop that does not perform
/// it.  Nothing forces, so the generic reader's residual boundary buys nothing
/// and this emits the same constant.
///
/// The constant owes two coordinates the residual boundary hides.
/// `last_instr` is an instruction-unit index here while the getset reports the
/// byte offset (`typedef.rs` returns `fget_f_lasti() * 2`), so the emission
/// carries the factor; without it a `dis` consumer's `f_lasti // 2` lands on
/// half the instruction index.  And the field has two writers on two
/// conventions — `flush_walk_end_state_to_frame` stores the resume coordinate
/// `pc - 1`, `LiveLastInstrGuard::enter_frame` stores the executing pc
/// unshifted — and a getter owes the executing one, which is what
/// [`walker_frame_executing_py_pc`] resolves.
///
/// `last_instr` travels as half of a pair — `capture_frame_scalars` records it
/// beside `valuestackdepth` because the interpreter derives its next opcode
/// from `last_instr + 1` and reads the operand stack at `valuestackdepth`, so
/// a consumer restoring one of them owes the other.  This emission assumes
/// nothing about `valuestackdepth` and is entitled to: it is a pure read that
/// never reaches the frame at all.  The pc it answers with comes from the
/// walk's own trace-time coordinate, not from the frame's field, so no state
/// is captured, none is restored, and the pair is never split.  Writing
/// `last_instr` from here would incur that obligation, which is the second
/// reason this path never does.
///
/// The receiver is pinned by [`walker_prove_owned_frame_pc`].
fn try_walker_specialize_frame_lasti<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(py_pc) = walker_prove_owned_frame_pc(ctx, op_pc, obj, concrete_obj)? else {
        return Ok(None);
    };
    let value = py_pc as i64 * 2;
    let raw = ctx.trace_ctx.const_int(value);
    let boxed = walker_box_int(ctx, op_pc, raw, value)?;
    // `walker_box_int` emits a heap `NewWithVtable`, so the recording-time
    // shadow has to be a heap `W_IntObject` too — the same pairing
    // [`box_int_concrete`] makes for a residual whose result arrived tagged.
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(
            pyre_object::intobject::w_int_new_unique(value) as usize,
        )),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Runtime half of the owned-frame `f_lineno` getter: `pyframe.py
/// fget_f_lineno` with the executing `last_instr` supplied by its caller
/// instead of read back off the frame.
extern "C" fn jit_frame_f_lineno_at(
    frame: *const pyre_interpreter::PyFrame,
    last_instr: i64,
) -> pyre_object::PyObjectRef {
    unsafe { &*frame }.f_lineno_at(last_instr as isize)
}

/// `pyframe.py fget_f_lineno` — the line the frame is currently executing.
///
/// Unlike its `f_lasti` sibling this is **not** a constant, and upstream's
/// compiled shape is not one either.  `policy.py look_inside_graph` admits
/// `fget_f_lineno` and `pyjitpl.py opimpl_getfield_vable_r` answers the
/// `debugdata` test out of `virtualizable_boxes`, but the decode underneath —
/// `pytraceback.py offset2lineno` walking the line table — stays a residual
/// call.  One non-forcing call is the shape to emit.
///
/// What this removes is the FORCE, not the call.  The generic reader
/// residualizes `space.getattr` as a single `CALL_MAY_FORCE`, and a may-force
/// boundary materializes the virtualizable — which is the only reason the
/// getter could read `last_instr` off the frame at all, since that field is
/// virtualizable and a compiled loop keeps the live coordinate in its own
/// state.  Handing the leaf the coordinate the walk already holds
/// ([`walker_prove_owned_frame_pc`]) removes that reason, leaving a leaf call
/// that names the frame without forcing it.
///
/// The leaf is [`PyFrame::f_lineno_at`], i.e. the getter body whole.  Its
/// `f_trace` test, `-1` sentinel and `first_line_number` fallback are one
/// decision, and `w_f_trace` can be armed while the loop is already compiled,
/// so that decision belongs at run time rather than baked into the trace.
///
/// Measured on a 200k-iteration read against a same-shape loop that does not
/// read the frame, best of 5: the read costs 0.0274s through the residual and
/// 0.0069s through this emission, against 0.0084s on CPython 3.14.6 and
/// 0.0227s on pypy3.  What removing the boundary is worth is counted rather
/// than inferred — the optimized trace loses half its `CALL_MAY_FORCE` (16 ->
/// 8) and half its `GuardNotForced` (16 -> 8) — and the two arms report the
/// same `loops_compiled`, `loops_aborted` and `guard_failures`, so the
/// difference is the emission and not one arm compiling less.
///
/// The call states an empty `EffectInfo` descr set, which is not a claim that
/// the leaf reads nothing: `make_call_descr_sized` panics on a non-empty raw
/// descr set minted after `compute_bitstrings`, and `finish_setup_descrs` runs
/// before any trace does, so every trace-time residual states the empty set and
/// `extraeffect` carries what is claimed.
///
/// The empty set is inert because no trace op names a field the leaf reads, and
/// `force_from_effectinfo` forces only descrs already in `cached_fields`.  The
/// reads are `PyFrame.pycode` and `PyFrame.debugdata`, then
/// `FrameDebugData.w_f_trace` and the code object's `linetable` and
/// `first_line_number`; `pyframe_debugdata_descr` has no emitter, `w_f_trace`
/// has no descr at all, and neither vable slot has a writer.  A later fold that
/// gives one of them a descr and caches it owes this call an explicit op, the
/// way `ResolveExceptionContext` records its own `SetfieldGc` rather than
/// naming `w_context` in a write set it cannot carry.
///
/// `pycode` and `debugdata` are read off the frame in memory rather than
/// through `virtualizable_entry_at`, where the neighbouring `locals()` fold
/// reads `debugdata`.  Neither slot has a `setfield_vable` writer
/// (`virtualizable_spec.rs` names the ones that do), so memory holds the live
/// value while compiled code runs, while the recording-time shadow's
/// `debugdata` is a `clone_debugdata_ptr` copy of it.  Memory answers the same
/// at both times; the shadow does not.
///
/// The receiver is pinned by [`walker_prove_owned_frame_pc`].
fn try_walker_specialize_frame_lineno<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(py_pc) = walker_prove_owned_frame_pc(ctx, op_pc, obj, concrete_obj)? else {
        return Ok(None);
    };
    let pc = ctx.trace_ctx.const_int(py_pc as i64);
    let value = ctx.trace_ctx.call_ref_typed_with_effect(
        jit_frame_f_lineno_at as *const (),
        &[obj, pc],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    // The recording-time shadow comes from the same entry point the leaf calls,
    // so it carries the getter's own small-int caching rather than a second
    // rendering of it.
    let concrete = unsafe {
        (*(concrete_obj as *const pyre_interpreter::PyFrame)).f_lineno_at(py_pc as isize)
    };
    ctx.trace_ctx.set_opref_concrete(
        value,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// The two operands `builtins.rs super_operands_from_frame` reads off the
/// frame, resolved as SSA values: `localsplus[0]` and the `__class__` freevar
/// cell.
///
/// Which channel holds them is the frame's own: an inlined callee owns a
/// [`CalleeLocalsShadow`], and everything else reads the standard
/// virtualizable, the same split
/// `try_walker_specialize_builtin_locals_in_callee` draws.
///
/// Either way the entries are already there.  The inline seeds the shadow with
/// the argument operands and with the live closure-cell reads
/// (`function_closure_descr`, then the items block) it threaded into the new
/// callee frame; the portal's boxes are seeded from the frame it entered on.
/// Reading them records no op.
fn walker_bare_super_frame_slots<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(OpRef, OpRef, bool)> {
    if let Some(shadow) = ctx.frame_state.borrow().callee_shadow.as_ref() {
        // `u16::MAX` is the strict fresh-frame fold switched off and a `NONE`
        // frame box is a frame register that was never seeded; in neither case
        // is the shadow the authority for this level's slots.
        if shadow.fold_frame_reg == u16::MAX || shadow.frame_box.is_none() {
            return None;
        }
        // SAFETY: the code object outlives the walk that resolved it; read-only.
        let code = unsafe { shadow.code_ptr.as_ref()? };
        let layout = pyre_interpreter::builtins::bare_super_frame_layout(code)?;
        let class_slot = layout.class_slot? as i64;
        let slot_op = |slot: i64| -> Option<OpRef> {
            let op = shadow.opref.get(&slot).copied()?;
            // Only an entry recorded through THIS level's frame register
            // describes this frame -- the same per-frame isolation the
            // own-frame vable read applies.
            (shadow.concrete.get(&slot)?.frame_reg == shadow.fold_frame_reg).then_some(op)
        };
        return Some((slot_op(0)?, slot_op(class_slot)?, layout.self_is_cell));
    }
    // A sub-walk that owns no shadow walks a frame the standard virtualizable
    // does not name, and a trace has exactly one of those.
    if ctx.fbw_mode.inline_subwalk || current_inline_concrete_frame() != 0 {
        return None;
    }
    // The frame `builtin_super`'s zero-argument tail reads is
    // `ExecutionContext::gettopframe()`.  Require it to BE the standard
    // virtualizable, so a hidden frame -- or any deeper one reached through the
    // backref chain -- declines rather than answering for someone else's slots.
    let vable_ptr = ctx.trace_ctx.standard_virtualizable_ptr()?;
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return None;
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != vable_ptr {
        return None;
    }
    // SAFETY: the frame is the live standard virtualizable; read-only.
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(&*frame) };
    let code = unsafe { code_ptr.as_ref()? };
    let layout = pyre_interpreter::builtins::bare_super_frame_layout(code)?;
    let class_slot = layout.class_slot?;
    // `locals_cells_stack_w` is PyFrame's only virtualizable array
    // (`virtualizable_gen.rs arrays`), so array index 0 names it.
    let info = ctx.trace_ctx.virtualizable_info()?;
    let lengths = ctx.trace_ctx.virtualizable_array_lengths()?;
    if info.num_arrays() != 1 || lengths.first().copied().unwrap_or(0) <= class_slot {
        return None;
    }
    // The value comes from the SHADOW, never from the frame's heap array: an
    // unsynchronized virtualizable's array holds whatever the frame last wrote
    // out, which is the staleness the read barrier's `force_now` repairs before
    // the residual reads it.  The shadow already holds the repaired value.
    let self_op = ctx
        .trace_ctx
        .virtualizable_entry_at(info.get_index_in_array(0, 0, lengths))?
        .0;
    let cell_op = ctx
        .trace_ctx
        .virtualizable_entry_at(info.get_index_in_array(0, class_slot, lengths))?
        .0;
    Some((self_op, cell_op, layout.self_is_cell))
}

/// The walker's holder for
/// [`pyre_interpreter::baseobjspace::super_check_apparent_fast_path`] -- the
/// third arm of `descriptor.py:_super_check`, admitted for the receivers whose
/// `space.getattr(obj, '__class__')` is the ordinary class-attribute read.
///
/// The predicate is the interpreter's so that both engines answer the same
/// receivers; what stays here is the three pins it reports, which are exactly
/// the guards an ordinary `LOAD_ATTR __class__` emits.  Super construction can
/// therefore consume the answer without inventing a stronger receiver-class
/// equality.
#[derive(Clone, Copy)]
pub(crate) struct ApparentSuperClass {
    pub(crate) objtype: pyre_object::PyObjectRef,
    receiver_type: pyre_object::PyObjectRef,
    receiver_version_tag: u64,
    receiver_map: pyre_interpreter::objspace::std::mapdict::MapRef,
}

pub(crate) fn walker_apparent_super_class(
    start_type: pyre_object::PyObjectRef,
    obj: pyre_object::PyObjectRef,
) -> Option<ApparentSuperClass> {
    let (receiver_type, receiver_version_tag, receiver_map, objtype) =
        unsafe { pyre_interpreter::baseobjspace::super_check_apparent_fast_path(start_type, obj) }?;
    Some(ApparentSuperClass {
        objtype,
        receiver_type,
        receiver_version_tag,
        receiver_map,
    })
}

/// Emit the proof corresponding to an ordinary traced
/// `obj.__class__` class-attribute read and return the promoted apparent type.
pub(crate) fn walker_guard_apparent_super_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj_op: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    apparent: ApparentSuperClass,
) -> Result<OpRef, DispatchError> {
    walker_guard_mapdict_instance_shape(
        ctx,
        op_pc,
        obj_op,
        concrete_obj,
        apparent.receiver_type,
        apparent.receiver_version_tag,
        apparent.receiver_map,
    )?;
    let objtype_const = ctx.trace_ctx.const_ref(apparent.objtype as i64);
    // `_super_check`'s final `issubtype_w(w_type, w_starttype)` is an
    // elidable MRO query on promoted types.  The receiver-type version pin
    // above protects which object `__class__` returns; this second pin
    // protects the returned class's ancestry.
    walker_pin_type_version_tag(ctx, op_pc, objtype_const)?;
    Ok(objtype_const)
}

/// Zero-argument `super()` folded to the proxy itself rather than re-routed to
/// a may-force residual.
///
/// [`try_walker_specialize_bare_super_call`] moves the frame force onto a
/// channel the walker can see, which is what keeps the loop from aborting; it
/// does not remove it.  What is left is a `MOST_GENERAL` call that publishes a
/// vref for the frame, wipes the trace's heap-field cache and is re-checked by
/// two guards, once per iteration.  Measured over 2,000,000 iterations:
/// `su = super(); su.m(x)` ran ~76ns each in an inlined callee and ~62 with
/// the loop in its own frame, against ~1.8 for `su = super(C, self); su.m(x)`.
///
/// The residual reads two frame slots and nothing else, and the walk holds
/// both as SSA values already ([`walker_bare_super_frame_slots`]), so the whole
/// call becomes the same `New` + `SetfieldGc` the two-argument spelling emits.
///
/// The class comes out of the `__class__` cell as a baked constant under the
/// `CellFamily.ever_mutated` quasi-immutable rather than a live read per
/// iteration: the cell a class body fills is written once, before any method
/// of that class can run, and `w_cell_set` marks the family the moment a
/// second write happens -- which retires this trace.  A cell that has already
/// been rebound declines here and keeps the residual.  A method whose own
/// `self` is a cellvar takes the same guarded live `Cell.contents` read as
/// LOAD_DEREF before `_super_check` sees the receiver.
pub(crate) fn try_walker_specialize_bare_super_virtual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `super()` with no user arguments arrives as `[callable, null_or_self]`.
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some((concrete_callable, _)) = plain_builtin_call_concretes(ctx, code, op, r_args, 0)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_super_type(concrete_callable) {
        return Ok(None);
    }
    let Some((raw_self_op, class_cell_op, self_is_cell)) = walker_bare_super_frame_slots(ctx)
    else {
        return Ok(None);
    };
    let Some(concrete_raw_self) = walker_concrete_ref_object(ctx, raw_self_op) else {
        return Ok(None);
    };
    let (concrete_self, self_cell_ref) = if self_is_cell {
        if !unsafe { pyre_object::is_cell(concrete_raw_self) } {
            return Ok(None);
        }
        let family = unsafe { pyre_object::w_cell_family(concrete_raw_self) };
        if family.is_null() || unsafe { (*family).ever_mutated.get() } {
            return Ok(None);
        }
        let contents = unsafe { pyre_object::w_cell_get(concrete_raw_self) };
        if contents.is_null() {
            return Ok(None);
        }
        (contents, Some(majit_ir::GcRef(concrete_raw_self as usize)))
    } else {
        (concrete_raw_self, None)
    };
    let Some(concrete_cell) = walker_concrete_ref_object(ctx, class_cell_op) else {
        return Ok(None);
    };
    // `fast2locals` falls back to the raw slot when it does not hold a cell.
    // That shape is unreachable for an OPTIMIZED frame past its
    // `COPY_FREE_VARS` prologue, and modelling it would need a second arm with
    // its own guard.
    if !unsafe { pyre_object::is_cell(concrete_cell) } {
        return Ok(None);
    }
    let family = unsafe { pyre_object::w_cell_family(concrete_cell) };
    if family.is_null() || unsafe { (*family).ever_mutated.get() } {
        return Ok(None);
    }
    let concrete_cls = unsafe { pyre_object::w_cell_get(concrete_cell) };
    if concrete_cls.is_null() || !unsafe { pyre_object::is_type(concrete_cls) } {
        return Ok(None);
    }
    // `descriptor.py:28-30` -- `None` builds the UNBOUND proxy, whose `w_self`
    // is null and whose attribute reads take a different arm entirely.
    if unsafe { pyre_object::is_none(concrete_self) } {
        return Ok(None);
    }
    let python_free =
        pyre_interpreter::builtins::super_check_python_free(concrete_cls, concrete_self);
    let apparent = if python_free.is_none() {
        walker_apparent_super_class(concrete_cls, concrete_self)
    } else {
        None
    };
    let Some(objtype) = python_free.or_else(|| apparent.map(|answer| answer.objtype)) else {
        return Ok(None);
    };
    let class_mode = python_free.is_some()
        && unsafe { pyre_object::is_type(concrete_self) }
        && std::ptr::eq(objtype, concrete_self);
    // The first two `_super_check` arms derive instance mode from the live
    // `w_class` slot.  The third derives it from the separately guarded
    // `__class__` lookup and must not impose this equality: transparent proxy
    // objects exist precisely because the two classes differ.
    if apparent.is_none()
        && !class_mode
        && !std::ptr::eq(objtype, unsafe { (*concrete_self).w_class })
    {
        return Ok(None);
    }

    // `_get_self_location`'s cellvar arm is the ordinary red-cell LOAD_DEREF
    // shape.  All of that helper's decline conditions were proved above (and
    // the inline-resume condition at entry), so once it emits no later
    // optional branch can abandon a partially-written fold.
    let self_op = if let Some(cell_ref) = self_cell_ref {
        let Some(value) =
            residual_call::try_walker_read_deref_cell(ctx, op.pc, raw_self_op, cell_ref)?
        else {
            return Ok(None);
        };
        value
    } else {
        raw_self_op
    };

    // Which callable `super` names is baked into the emitted body.
    walker_guard_stamped_ref(ctx, op.pc, r_args[0], concrete_callable)?;
    let cell_type = &pyre_object::nestedscope::CELL_TYPE as *const _ as i64;
    walker_guard_stamped_class(ctx, op.pc, class_cell_op, cell_type)?;
    let owner = ctx.trace_ctx.const_ref(family as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::cell_family_ever_mutated_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op.pc)?;
    let cls_op = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        class_cell_op,
        crate::descr::cell_contents_descr(),
    );
    if !matches!(
        ctx.trace_ctx.box_value(cls_op),
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
    ) {
        ctx.trace_ctx.set_opref_concrete(
            cls_op,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_cls as usize)),
        );
    }
    let proxy_op = if let Some(apparent) = apparent {
        walker_emit_apparent_super_proxy(
            ctx,
            op.pc,
            cls_op,
            self_op,
            concrete_cls,
            concrete_self,
            apparent,
        )?
    } else {
        walker_emit_super_proxy(
            ctx,
            op.pc,
            cls_op,
            self_op,
            concrete_cls,
            concrete_self,
            objtype,
            class_mode,
        )?
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', proxy_op)?;
    Ok(Some(()))
}

/// Zero-argument `super()` reached as a call, i.e. the `LOAD_GLOBAL super` +
/// `CALL` spelling a name binding produces rather than `LOAD_SUPER_ATTR`.
///
/// A substitution, not a fold: the call stays a may-force residual, still
/// guarded and still concrete-executed by the shared tail in
/// `dispatch_residual_call_iRd_kind`, so the force and exception guards, the
/// heapcache invalidation and the dst writeback are the ones the generic path
/// already emits.  What changes is the target.
///
/// The generic `bh_call_fn` this replaces reaches `builtin_super`'s
/// zero-argument tail, whose `ExecutionContext::gettopframe()` runs
/// `force_frame` INSIDE an opaque residual; that clears
/// `TOKEN_TRACING_RESCALL` and `tracing_after_residual_call` reads it back as
/// `VableEscapedDuringResidualCall`.  The frame `gettopframe` answers with is
/// the one being traced, so that residual always escapes, the walk always
/// aborts, and after `MAX_TRACE_ABORT_COUNT` of them the merge point is
/// stamped `JC_DONT_TRACE_HERE` for the rest of the process.
///
/// [`crate::helpers::jit_bare_super_from_frame`] is the same `descriptor.py
/// _super_from_frame` half `bh_load_super_attr_fn` calls, and it takes the
/// frame as an operand.  Naming it directly is what lets the name-bound
/// spelling take the route `LOAD_SUPER_ATTR` already takes: there is nothing
/// for the callee to rediscover, so nothing forces the way `gettopframe` does.
///
/// Every receiver is admitted, including the ones `_super_check` can only
/// settle by asking Python.  Restricting it to the settled half is what an
/// earlier version did, and the decline is what cost the abort — the fall-back
/// was the escaping residual, so a `__class__` property was paid for with the
/// whole trace.  The shared tail is what makes the wider admission safe: it
/// brackets the call with `vable_and_vrefs_before_residual_call`, transcribes a
/// raise into `last_exc_value` the way `execute_raised` does, and counts the
/// call on the executed-effect odometer (through
/// [`majit_ir::RuntimeHelperKind::BareSuperFromFrame`]) so a nested abort
/// cannot rewind past a property that has already run.  Executing at record
/// time and THEN declining is the one shape that would double it, and it is
/// what site E of `bare_super_frame_escape.py` pins.
///
/// The virtual fold ([`try_walker_specialize_bare_super_virtual`]) runs ahead
/// of this and answers without any call at all where it can; this is what its
/// declines fall through to.
pub(crate) fn try_walker_specialize_bare_super_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
) -> Result<Option<DirectResidualSubst>, DispatchError> {
    // `super()` with no user arguments arrives as `[callable, null_or_self]`.
    let Some((concrete_callable, _)) = plain_builtin_call_concretes(ctx, code, op, r_args, 0)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_super_type(concrete_callable) {
        return Ok(None);
    }
    let Some((frame_box, _frame_ptr)) = walker_executing_frame_box(ctx) else {
        return Ok(None);
    };

    // ── tentative commit ──
    // Pin the callable the way the constructor folds do, so rebinding the
    // global `super` side-exits instead of keeping this route.
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let funcptr = ctx
        .trace_ctx
        .const_int(crate::helpers::jit_bare_super_from_frame as *const () as i64);
    Ok(Some(DirectResidualSubst {
        funcptr,
        descr: bare_super_from_frame_descr(),
        allboxes: vec![funcptr, frame_box],
    }))
}

/// The descr the zero-argument `super()` substitution installs: `(Ref) -> Ref`,
/// `MOST_GENERAL`, tagged [`majit_ir::RuntimeHelperKind::BareSuperFromFrame`].
///
/// `MOST_GENERAL`, not a constructed `EffectInfo`: the leaf is the whole entry
/// point, whose `_super_check` arm can run a `__class__` property, so no
/// read/write descr set describes it and an empty one would claim the opposite,
/// letting the optimizer keep cached fields across the call.  `RandomEffects`
/// outranks `ForcesVirtualOrVirtualizable`, which is what keeps
/// `select_residual_call_opcode` on the may-force branch and the trailing
/// `GuardNotForced` meaningful.
///
/// The helper tag on top of it is the executed-effect odometer's only reader
/// here — see the variant's own documentation.
pub(crate) fn bare_super_from_frame_descr() -> DescrRef {
    majit_metainterp::make_call_descr_with_effect(
        &[Type::Ref],
        Type::Ref,
        majit_ir::EffectInfo {
            runtime_helper: majit_ir::RuntimeHelperKind::BareSuperFromFrame,
            ..majit_ir::EffectInfo::MOST_GENERAL
        },
    )
}

/// `mapdict.py LOAD_ATTR_caching` full-body-walker fast path for a
/// plain (non-method) instance attribute.  When the concrete receiver is a
/// monomorphic instance whose attribute resolves to a boxed plain storage slot
/// or an unboxed integer/float slot, emit the guarded read PyPy compiles
/// LOAD_ATTR to under the JIT —
///   * `guard_class(obj, concrete_layout)` — the receiver keeps the exact
///     layout vtable whose mapdict-carrier prefix was proved at trace time (so
///     the `map`/`storage` reads below are valid; `mapdict.py` `if map is not
///     None:` also filters non-carriers at trace time).
///   * `guard_value(getfield_gc_i(w_type, version_tag), C_version_tag)` — pins
///     the class lookup result so a later descriptor or `__getattribute__`
///     mutation deopts on trace re-entry.
///   * `guard_value(getfield_gc_i(obj, map), C_map)` — `jit.promote(self.map)`
///     (`mapdict.py`); pins the exact instance shape so `find_map_attr`
///     const-folds `storageindex` to a green constant.
///   * boxed: `getfield_gc_r(obj, storage)` +
///     `getarrayitem_gc_r(block, C_index)` for
///     `mapdict.py _mapdict_read_storage`;
///   * unboxed int/float: a non-forcing typed read plus `wrapint`/`wrapfloat`,
///     matching `_prim_direct_read` (mapdict.py).
/// — instead of the opaque `getattr_fn` `CALL_MAY_FORCE` MRO-walk residual.
///
/// Returns `Some(())` after writing the dst; `None` (fall through to the
/// residual) for every shape [`load_attr_fast_path`] declines: non-instance
/// receiver, missing map, custom `__getattribute__`, uncacheable `version_tag`,
/// a data-descriptor / `INVALID` classification, or an attribute not on this
/// instance's map.  The map `guard_value` proves the attribute is present on
/// this shape, so a successful fold provably cannot raise `AttributeError` —
/// dropping the residual's exception guard is sound even in a handler-bearing
/// body (same reasoning as the LoadGlobal fold).
///
/// `name` is the already-resolved attribute name, so the fold serves both the
/// `LOAD_ATTR` residual — whose caller reads it out of the jitcode's own
/// `co_names` — and the `getattr(obj, "name")` builtin, whose name arrives as a
/// constant string operand.  Both spell one `space.getattr`, so they must reach
/// the same read.
pub(crate) fn try_walker_specialize_load_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    use pyre_interpreter::pyframe::PyFrame;

    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    // The receiver must be a concrete instance for the map/storageindex
    // resolution below; a non-concrete or non-instance receiver declines.
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    // CPython 3.14 exposes an optimized frame's locals as a fresh
    // `FrameLocalsProxy`.  Constructing that proxy does not read fast locals;
    // its operations synchronize through the frame when they are actually
    // used.  Keep the exact per-MIFrame red receiver as the proxy owner instead
    // of residualizing the getter, whose explicit read barrier would force a
    // live virtualizable while an inline MIFrame is still active.
    //
    // There are two identities the walker can prove here: the current inline
    // callee's shadow frame, whose locals region the walk flushes itself at the
    // escape, or the portal frame.  `descr_get_tb_frame` hands the portal
    // back as a getfield result, a red box other than
    // `standard_virtualizable_box`; the check below proves that alias is the
    // standard virtualizable.  `fast2locals` is `@jit.unroll_safe` and reads
    // `locals_cells_stack_w` from the virtualizable boxes, so the proxy read
    // does not force.  The dropped force was also what wrote the portal's
    // locals region out, so the fold writes that region itself.
    // `pyjitpl.py MIFrame._nonstandard_virtualizable`: a box that is not
    // `virtualizable_boxes[-1]` but points at it is still the standard
    // virtualizable.  The check records `PTR_EQ` + `implement_guard_value`
    // and, when the pointers match, `replace_box`s the alias onto the
    // standard box.  `f_locals` then takes the existing standard-frame arm.
    // A failed identity falls through to `emit_force_virtualizable` and this
    // fold declines, the same as a receiver that was never the portal frame.
    let mut obj = obj;
    if name == "f_locals"
        && ctx.trace_ctx.standard_virtualizable_ptr() == Some(concrete_obj as usize)
        && ctx
            .trace_ctx
            .standard_virtualizable_box()
            .is_some_and(|standard| standard != obj)
        && let Some(info) = ctx.trace_ctx.virtualizable_info().cloned()
    {
        // `locals_cells_stack_w` is the virtualizable array `f_locals` reads,
        // so its descr carries the active vinfo (`vinfo is fielddescr.get_vinfo()`).
        let fielddescr = info.array_pointer_field_descr(0);
        let nonstandard = vable_ops::with_replace_frames(ctx, |ctx| {
            vable_ops::walker_nonstandard_virtualizable(ctx, op_pc, obj, &fielddescr)
        })?;
        if !nonstandard && let Some(standard) = ctx.trace_ctx.standard_virtualizable_box() {
            obj = standard;
        }
    }
    let concrete_addr = concrete_obj as usize;

    // `f_locals` on a frame this trace still owns reads that frame's shadow
    // (portal virtualizable or the inline level's own slots), including a
    // `Void` concrete half, and publishes it before the proxy exists.
    // `receiver_trace_locals` is also `None` for a finished frame this trace
    // allocated. `frame_locals_heap_is_trace_allocation` is that case:
    // `fget_getdictscope` only allocates the proxy, because the array stores
    // already in the trace are the locals. Any other optimized frame stays on
    // `bh_load_attr_fn`, whose `jit_force_virtualizable` publishes the
    // virtualizable — including an except-bound name the heap array does not
    // hold yet.
    if name == "f_locals"
        && unsafe { (*concrete_obj).ob_type } == &pyre_interpreter::pyframe::FRAME_TYPE
        && unsafe {
            (*(concrete_obj as *const pyre_interpreter::PyFrame))
                .code()
                .flags
                .contains(pyre_interpreter::CodeFlags::OPTIMIZED)
        }
    {
        let w_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE);
        let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
        if version_tag == 0 || unsafe { (*concrete_obj).w_class } != w_type {
            return Ok(None);
        }
        let concrete_frame = if let Some(source) = ctx.receiver_trace_locals(obj, concrete_addr) {
            walker_publish_complete_frame_locals(ctx, obj, concrete_addr, &source)
                as pyre_object::PyObjectRef
        } else if frame_locals_heap_is_trace_allocation(ctx, obj) {
            concrete_obj
        } else {
            return Ok(None);
        };
        let concrete_proxy = pyre_interpreter::pyframe::frame_locals_proxy::new(concrete_frame);
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_frame, w_type, version_tag)?;
        let proxy = ctx.trace_ctx.call_ref_typed_with_effect(
            jit_inline_frame_locals_proxy_new as *const (),
            &[obj],
            &[majit_ir::Type::Ref],
            // The proxy's later reads observe `locals_cells_stack_w`. An
            // effect with an empty read set lets the optimizer drop the
            // slot stores published just above, and the name those stores
            // carried disappears.
            majit_ir::EffectInfo::MOST_GENERAL,
        );
        ctx.trace_ctx.set_opref_concrete(
            proxy,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_proxy as usize)),
        );
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, proxy)?;
        return Ok(Some(()));
    }
    if name == "f_lasti"
        && unsafe { (*concrete_obj).ob_type } == &pyre_interpreter::pyframe::FRAME_TYPE
        && spec_gate(SpecFold::FrameLasti, || {
            try_walker_specialize_frame_lasti(ctx, op_pc, obj, concrete_obj, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok(Some(()));
    }
    if name == "f_lineno"
        && unsafe { (*concrete_obj).ob_type } == &pyre_interpreter::pyframe::FRAME_TYPE
        && spec_gate(SpecFold::FrameLineno, || {
            try_walker_specialize_frame_lineno(ctx, op_pc, obj, concrete_obj, dst, dst_bank)
        })?
        .is_some()
    {
        return Ok(Some(()));
    }
    // `mapdict.py` resolution, returning the fold ingredients (the
    // read is left to the caller so it can be folded to a guarded inline read).
    if let Some((w_type, version_tag, map, storageindex, attr)) =
        unsafe { pyre_interpreter::objspace::std::mapdict::load_attr_fast_path(concrete_obj, name) }
    {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;

        // mapdict.py `PlainAttribute._pure_direct_read`, the `@jit.elidable`
        // read `AbstractAttribute.read` picks when the attribute has never been
        // written since it was added and both it and the receiver are green.
        // Without it the two loads below stay in the trace and are carried
        // around the loop as label arguments; the elidable read answers from
        // the recorded value under the `ever_mutated?` quasi-immutable alone,
        // which `write` and `delete` both set and whose sweep invalidates the
        // trace.
        if obj.is_constant()
            && let Some(()) = spec_gate(SpecFold::LoadAttrPureRead, || {
                let Some(plain) = (unsafe {
                    pyre_interpreter::objspace::std::mapdict::pure_direct_read_attr(attr)
                }) else {
                    return Ok(None);
                };
                walker_pin_plain_ever_mutated(ctx, op_pc, plain)?;
                let w_value = unsafe {
                    pyre_interpreter::objspace::std::mapdict::read_boxed_storage(
                        concrete_obj,
                        storageindex,
                    )
                };
                // The same forwarding the class-attribute arm below relies on:
                // `remove_constptrs_in` rewrites the recorded `ConstPtr` to a
                // `LoadFromGcTable` at emit, so a nursery value folds too.
                let value = ctx.trace_ctx.const_ref(w_value as i64);
                write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
                Ok(Some(()))
            })?
        {
            return Ok(Some(()));
        }

        // getfield_gc_r(obj, storage) + getarrayitem_gc_r(block, C_storageindex):
        // the inline value read (`mapdict.py`).  `storageindex` is a green
        // constant (the map guard pinned it); `trace_mapdict_storage_getitem`
        // stamps the dst's concrete shadow from the live block slot.
        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
            crate::descr::mapdict_storage_descr(concrete_obj)
        });
        let idx_const = ctx.trace_ctx.const_int(storageindex as i64);
        let value = crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, idx_const);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // `object.__class__` is a data descriptor, so the slot fold above
    // declines it. `class_descr_fast_path` is that getter: `space.type(w_obj)`,
    // guarded like any other mapdict attribute read.
    if name == "__class__"
        && let Some((w_type, version_tag, map)) =
            unsafe { pyre_interpreter::objspace::std::mapdict::class_descr_fast_path(concrete_obj) }
    {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;
        let value = ctx.trace_ctx.const_ref(w_type as i64);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // Class attribute that object.__getattribute__ returns unchanged
    // (`Object.descr__getattribute__` / `class_attr_fast_path`).  The
    // instance-slot fold above needs a mapdict storage index; a name that
    // lives only on the type has none and would otherwise residualize
    // `space.getattr`.
    if let Some((w_type, version_tag, map, w_value)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::class_attr_fast_path(concrete_obj, name)
    } {
        // Movability does not decide this fold.  A recorded `ConstPtr` is
        // forwarded — `remove_constptrs_in` rewrites it to a `LoadFromGcTable`
        // at emit and `gcreftracer` keeps the table slot current at run — and
        // the window before that, while `write_residual_call_result_to_dst`
        // holds the `OpRef` in the walker's `registers_r`, is what
        // `InlineRegisterBankGuard` (`miframe_registers`) walks.  So a class
        // attribute allocated in the nursery folds like any other object.
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;
        let value = ctx.trace_ctx.const_ref(w_value as i64);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    if let Some(walk_field) = traceback_walk_field(concrete_obj, name) {
        if let Some(()) = walker_specialize_traceback_walk_field(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            walk_field,
            dst,
            dst_bank,
        )? {
            return Ok(Some(()));
        }
    }

    // A user attribute on an exception instance. `mapdict.py:1483-1490
    // LOAD_ATTR_caching` declines this receiver upstream too — an exception is
    // not a `MapdictStorageMixin` and `_get_mapdict_map` answers None
    // (`baseobjspace.py`) — and it reaches its speed by inlining
    // `getdictvalue -> MapDictStrategy.getitem_str -> AbstractAttribute.read`
    // (`mapdict.py`, `:442-444`) instead. The attribute lives in the
    // `newdict(instance=True)` dictionary, two hops out: `w_dict` ->
    // `W_DictObject.dstorage` -> the fake carrier that holds the map.
    //
    // `w_exception_get_kind` and `w_exception_peek_dict` both cast straight to
    // `W_BaseException`, so the `is_exception` test is load-bearing: this
    // function runs for every `LOAD_ATTR` receiver that reached it.
    let exc_dict = unsafe {
        pyre_object::is_exception(concrete_obj)
            .then(|| pyre_object::interp_exceptions::w_exception_peek_dict(concrete_obj))
            .filter(|dict| !dict.is_null())
    };
    if let Some(dict) = exc_dict
        && let Some((w_type, _version_tag, carrier, map, storageindex, unboxed)) = unsafe {
            pyre_interpreter::objspace::std::mapdict::instance_dict_attr_fast_path(
                concrete_obj,
                dict,
                name,
            )
        }
        // An unboxed float slot keeps the `f64` bit pattern in the same
        // longlong block as an int, so folding it needs a bits-to-float
        // reinterpret the trace has no operation for; leave it on the residual.
        && !matches!(
            unboxed,
            Some((pyre_interpreter::objspace::std::mapdict::UnboxType::Float, _))
        )
    {
        let kind = unsafe { pyre_object::w_exception_get_kind(concrete_obj) };
        let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete_obj) };
        walker_pin_stamped_instance_class(ctx, op_pc, obj, concrete_obj, w_type)?;

        let dict_op = walker_record_getfield_gc_r_uncached(
            ctx,
            obj,
            crate::descr::w_exception_dict_descr_for(kind, user),
        );
        walker_guard_stamped_nonnull(ctx, op_pc, dict_op, dict)?;

        // `instance_dict_attr_fast_path` declines a dictionary that is not
        // `MapDictStrategy`-backed, and the carrier read below is out of bounds
        // on a devolved one, so pin the strategy before dereferencing
        // `dstorage`.
        walker_guard_stamped_dict_strategy(
            ctx,
            op_pc,
            dict_op,
            &pyre_interpreter::objspace::std::mapdict::MAP_DICT_STRATEGY_REF as *const _ as i64,
        )?;

        let carrier_op =
            walker_record_getfield_gc_r_uncached(ctx, dict_op, crate::descr::dict_dstorage_descr());
        ctx.trace_ctx.set_opref_concrete(
            carrier_op,
            majit_ir::Value::Ref(majit_ir::GcRef(carrier as usize)),
        );
        let map_op = walker_record_getfield_gc_i_uncached(ctx, carrier_op, unsafe {
            crate::descr::mapdict_map_descr(carrier)
        });
        walker_guard_stamped_int(ctx, op_pc, map_op, map as i64)?;

        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, carrier_op, unsafe {
            crate::descr::mapdict_storage_descr(carrier)
        });
        let index = ctx.trace_ctx.const_int(storageindex as i64);
        let slot = crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, index);
        let value = match unboxed {
            None => slot,
            // `_prim_direct_read` (mapdict.py): the storage slot holds
            // the shared longlong block, and the value is `items[listindex]`.
            // Keeping the boxing in the trace lets an immediate integer
            // consumer virtualize it away.
            Some((_, listindex)) => {
                let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
                let live = unsafe {
                    pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
                        carrier,
                        storageindex,
                        listindex,
                    )
                };
                let raw = crate::state::trace_int_block_getitem_value(
                    ctx.trace_ctx,
                    slot,
                    listindex_const,
                );
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(live));
                let boxed = walker_box_int(ctx, op_pc, raw, live)?;
                let live_ptr = pyre_object::w_int_new(live) as i64;
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
                boxed
            }
        };
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    if let Some((slot, kind, w_type, version_tag, stored)) = unsafe {
        pyre_interpreter::baseobjspace::exception_attr_slot_fold(concrete_obj, name, false)
    } {
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args && stored.is_null() {
            return Ok(None);
        }
        // `descr_getargs` copies `args_w` through `newtuple`, which picks the
        // tuple representation from the arity and the element types.  Ask
        // `newtuple` itself which shape this read produces, rather than second-
        // guessing its dispatch, and settle it here — before any guard is
        // recorded, so a decline stays clean (a bail-out after the class pin
        // would leave the caller reading this attribute as already guarded).
        //
        // Only the shape crosses into the emit below; the probe tuple is
        // dropped rather than held, since nothing roots it across the guards.
        let args_specialised_oo = if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args
        {
            let ob_type = unsafe { (*args_tuple_shape_probe(stored)).ob_type };
            if std::ptr::eq(
                ob_type,
                &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
            ) {
                true
            } else if std::ptr::eq(ob_type, &pyre_object::TUPLE_TYPE) {
                false
            } else {
                // The unboxed arity-2 specialisations (`Cls_ii` / `Cls_ff`)
                // hold machine values in their inline fields, which this copy
                // has no unboxed operand for.  Only an Object-strategy `args_w`
                // holding exactly two plain ints or two plain floats gets here.
                return Ok(None);
            }
        } else {
            false
        };
        let traceback_frame = if slot
            == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback
        {
            let frame = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(stored) };
            // `mark_traceback_escaped` leaves a torn-down traceback alone.
            // Decline before recording the receiver guards when the
            // authoritative read sees that shape; compiled replays guard
            // the frame load below and side-exit to the same residual path.
            if frame.is_null() {
                return Ok(None);
            }
            Some(frame)
        } else {
            None
        };
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete_obj) };
        let raw_value = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            obj,
            crate::descr::w_exception_attr_slot_descr_for(kind, slot, user),
        );
        walker_guard_stamped_nonnull(ctx, op_pc, raw_value, stored)?;
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback {
            // The fold replaces `descr_gettraceback`, whose read marks the
            // traceback's frame escaped so `ExecutionContext::leave` forces
            // its vref.  `descr_settraceback` admits only None or PyTraceback,
            // so the non-null slot is already type-safe without a class guard.
            // Read the node's frame, require the non-null case handled by the
            // traced path, and mirror `PyFrame.mark_as_escaped` directly.
            let frame_ref = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                raw_value,
                crate::descr::pytraceback_frame_descr(),
            );
            let concrete_frame = traceback_frame.expect("traceback fold has no frame");
            walker_guard_stamped_nonnull(
                ctx,
                op_pc,
                frame_ref,
                concrete_frame as pyre_object::PyObjectRef,
            )?;
            let flags_descr = crate::descr::pyframe_flags_descr();
            let live_flags =
                crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, frame_ref, flags_descr.clone());
            let escaped_bit = ctx.trace_ctx.const_int(i64::from(PyFrame::FLAG_ESCAPED));
            let new_flags = ctx
                .trace_ctx
                .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
            ctx.trace_ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[frame_ref, new_flags],
                flags_descr.clone(),
            );
            ctx.trace_ctx
                .heapcache_setfield_cached(frame_ref, flags_descr.index(), new_flags);

            // The walk is the authoritative execution path, so mark its
            // concrete frame now as well as on every compiled re-execution.
            unsafe { pyre_interpreter::pytraceback::mark_traceback_escaped(stored) };
        }
        let value = if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
            let len = unsafe { pyre_object::interp_exceptions::rlist_len(stored) };
            let length = crate::state::opimpl_arraylen_gc(
                ctx.trace_ctx,
                raw_value,
                crate::state::pyobject_gcarray_descr(),
            );
            walker_guard_stamped_len(ctx, op_pc, length, len as i64)?;
            let mut items = Vec::with_capacity(len);
            let mut concrete_items = Vec::with_capacity(len);
            for index in 0..len {
                let index_op = ctx.trace_ctx.const_int(index as i64);
                items.push(crate::state::trace_items_block_getitem_value(
                    ctx.trace_ctx,
                    raw_value,
                    index_op,
                ));
                concrete_items
                    .push(unsafe { pyre_object::interp_exceptions::rlist_getitem(stored, index) });
            }
            // Emit the representation `newtuple` picks, settled above.  Emitting
            // the array-backed shape for an arity the runtime specialises leaves
            // the trace disagreeing with its own record-time concrete, and the
            // `except <tuple>:` match fold — which dispatches on that concrete's
            // layout — then guards for a shape this trace never builds, so the
            // loop aborts instead of compiling.
            let tuple = if args_specialised_oo {
                crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, items[0], items[1])
            } else {
                crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items)
            };
            let concrete_tuple = pyre_object::w_tuple_new(concrete_items);
            ctx.trace_ctx.set_opref_concrete(
                tuple,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
            );
            tuple
        } else {
            raw_value
        };
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // Module attribute (`math.sqrt`): the receiver is an exact module and the
    // name is present in its dict.  Fold the module-dict read to a
    // `QUASIIMMUT_FIELD(dict, slot)` version guard + elidable cell lookup —
    // celldict.py `_getdictvalue_no_unwrapping_pure` (`@jit.elidable_promote`) —
    // so a hot `math.sqrt(x)` loop drops its per-iteration LOAD_ATTR may-force
    // residual and the `math.sqrt` callable becomes a trace constant.  A rebind
    // of the attribute bumps the module dict `version` and fails the guard.
    // All resolution below is read-only; a missing / non-canonical
    // shape falls through to the residual with no IR emitted.  An exact
    // `module` `w_class` excludes a module subclass with a custom
    // `__getattribute__`; a module-level PEP 562 `__getattr__` is irrelevant
    // because the name is present (the dict lookup wins before `__getattr__`).
    // A data descriptor on the module type (e.g. `__dict__`) outranks a
    // same-named dict entry in generic getattr, so decline when the name
    // resolves to one — the descriptor result, not the dict cell, is what a
    // read returns.
    if unsafe { pyre_object::is_module(concrete_obj) }
        && std::ptr::eq(
            unsafe { (*concrete_obj).w_class },
            pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::MODULE_TYPE),
        )
        && !unsafe {
            pyre_interpreter::baseobjspace::type_lookup_is_data_descr((*concrete_obj).w_class, name)
        }
    {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(concrete_obj) };
        if !w_dict.is_null() {
            if let Some(slot) = crate::state::module_dict_cell_slot_direct(w_dict, name) {
                if let Some(stored) = crate::state::module_dict_cell_value_direct(w_dict, slot) {
                    if !stored.is_null() {
                        // The first binding of a module name is the raw object
                        // (`write_cell` StoreBare), which is nursery-born for a
                        // function.  `emit_namespace_cell_fold` records that
                        // GCREF as a `ConstPtr` the active-trace walk, resume
                        // pools and gcref table forward; movability does not
                        // decide the fold.  Same as `emit_module_dict_cell_fold`.
                        // Pin the receiver to THIS module so the baked dict
                        // address is correct: a constant receiver is already
                        // pinned; a non-constant one gets a `guard_value`.
                        walker_guard_fold_callable(ctx, op_pc, obj, concrete_obj)?;
                        // `guard_frame_globals=false`: the receiver pin above
                        // (not a frame-globals-identity guard) proves the dict.
                        if !emit_namespace_cell_fold(
                            ctx, op_pc, dst, dst_bank, w_dict, slot, stored, false, true,
                        )? {
                            // Nothing was written to `dst`, so the residual
                            // still owes the load.
                            return Ok(None);
                        }
                        return Ok(Some(()));
                    }
                }
            }
        }
    }

    let Some((w_type, version_tag, map, storageindex, listindex, unbox_type, attr)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::load_attr_unboxed_fast_path(concrete_obj, name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, concrete_obj, w_type, version_tag, map)?;
    let terminator = unsafe { (*map).terminator() };
    let term = unsafe { (*terminator).as_terminator() as *const _ };
    walker_pin_terminator_allow_unboxing(ctx, op_pc, term)?;

    // mapdict.py `UnboxedPlainAttribute._pure_direct_read`:
    // `self._box(self._pure_unboxed_read(obj))` — only the raw longlong read is
    // `@jit.elidable`, and the boxing stays visible to the trace, so an
    // immediate numeric consumer still unwraps the constant. Taken when the
    // attribute has never been written since it was added and the receiver is
    // green, under the `ever_mutated?` quasi-immutable the elidable read owes.
    let pure_raw = if obj.is_constant() {
        spec_gate(SpecFold::LoadAttrPureRead, || {
            Ok::<_, DispatchError>(unsafe {
                pyre_interpreter::objspace::std::mapdict::pure_direct_read_attr(attr)
            })
        })?
    } else {
        None
    };

    // `_prim_direct_read` (mapdict.py) as the three loads it is: the
    // instance's storage block, the slot holding this attribute's raw list,
    // and the item.  Both coordinates are green constants the map guard pins,
    // so the two leading loads are loop-invariant over any stretch that cannot
    // force and the heap cache folds them away; keeping the boxing in the
    // trace lets an immediate numeric consumer virtualize that too.  A
    // residual reading the same three words instead would take the receiver as
    // a `Ref` argument, which forces an instance the trace allocated.
    let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
    let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
    let live = unsafe {
        pyre_interpreter::objspace::std::mapdict::read_unboxed_storage_raw(
            concrete_obj,
            storageindex,
            listindex,
        )
    };
    let raw_read = |ctx: &mut WalkContext<'_, '_, Sym>| -> Result<OpRef, DispatchError> {
        if let Some(plain) = pure_raw {
            walker_pin_plain_ever_mutated(ctx, op_pc, plain)?;
            return Ok(match unbox_type {
                pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                    ctx.trace_ctx.const_int(live)
                }
                pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                    ctx.trace_ctx.const_float(live)
                }
            });
        }
        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
            crate::descr::mapdict_storage_descr(concrete_obj)
        });
        let slot =
            crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, storageindex_const);
        Ok(match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                crate::state::trace_int_block_getitem_value(ctx.trace_ctx, slot, listindex_const)
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                crate::state::trace_float_block_getitem_value(ctx.trace_ctx, slot, listindex_const)
            }
        })
    };
    let raw = raw_read(ctx)?;
    let boxed = match unbox_type {
        pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(live));
            let boxed = walker_box_int(ctx, op_pc, raw, live)?;
            // The `wrapint` op is a heap box, so its concrete must be a heap ptr too:
            // box the raw longlong through the same `w_int_new` the unboxed read uses
            // (mapdict.py `_box`); `box_int_concrete` re-homes a tagged small
            // int to a fresh heap `W_IntObject` so op(NewWithVtable) == concrete(heap).
            // Without this stamp the boxed result carries no concrete, so a downstream
            // eager void residual (e.g. the STORE_ATTR that writes `self.value`) cannot
            // resolve its value arg and the walk aborts `ResidualCallArgUnbound`.
            let live_ptr = pyre_object::w_int_new(live) as i64;
            ctx.trace_ctx
                .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
            boxed
        }
        pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
            let live_f = f64::from_bits(live as u64);
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(live_f));
            let boxed = crate::state::wrapfloat(ctx.trace_ctx, raw);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::w_float_new(live_f) as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// The receiver-layout-specific half of the LOAD_METHOD fold: which field the
/// walker guards to keep "no instance attribute shadows this method" true for
/// the life of the trace.
///
/// `load_method_fast_path` proved the property once, at record time; this is
/// what re-proves it on every execution.  One variant per layout whose
/// non-allocating dictionary peek that predicate covers.
enum ShadowGuard {
    /// A `W_ObjectObject` receiver: pin the mapdict map, so adding
    /// `obj.<name>` grows the map chain and side-exits.
    InstanceMap(*const u8),
    /// A `W_BaseException` receiver: pin `w_dict` at null, so the lazy
    /// allocation `e.<name> = ...` performs side-exits. Carries the kind
    /// because each `ExcKind` has its own descr group; the `_getusercls`
    /// bit is read from the concrete typeptr at the emit.
    ExceptionDictIsNull(pyre_object::interp_exceptions::ExcKind),
}

/// Which [`ShadowGuard`] proves that no instance attribute shadows a type
/// lookup on this receiver.  `None` is a layout with no such guard — including
/// a devolved instance — and declines the fold before anything is emitted.
///
/// # Safety
/// `concrete_obj` must be a valid, non-null object pointer.
unsafe fn walker_classify_shadow_guard(
    concrete_obj: pyre_object::PyObjectRef,
) -> Option<ShadowGuard> {
    unsafe {
        if pyre_object::is_instance(concrete_obj) {
            let map = (*(concrete_obj as *const pyre_object::W_ObjectObject)).map;
            if map == 0 {
                return None;
            }
            // A devolved instance holds its attributes in a dictionary and
            // keeps the same map across a later `e.<name> = ...`, so pinning
            // the map would not observe the shadow the assignment installs.
            // `W_ObjectObject.map` is stored as a raw word; the map layer
            // owns the node type.
            if pyre_interpreter::objspace::std::mapdict::map_is_devolved(map as *const _) {
                return None;
            }
            Some(ShadowGuard::InstanceMap(map as *const u8))
        } else if pyre_object::is_exception(concrete_obj) {
            Some(ShadowGuard::ExceptionDictIsNull(
                pyre_object::w_exception_get_kind(concrete_obj),
            ))
        } else {
            None
        }
    }
}

/// Re-prove the shadowing precondition: growing an instance attribute named
/// like the method must side-exit before the constant descriptor is reused.
/// mapdict.py LOAD_ATTR caching does this by pinning the map; an exception
/// has no map, and pins the still-unallocated `w_dict` slot instead.
fn walker_emit_shadow_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    shadow: ShadowGuard,
) -> Result<(), DispatchError> {
    let (slot_op, slot_const, shadow_guard) = match shadow {
        ShadowGuard::InstanceMap(map) => (
            walker_record_getfield_gc_i_uncached(ctx, obj, unsafe {
                crate::descr::mapdict_map_descr(concrete_obj)
            }),
            ctx.trace_ctx.const_int(map as i64),
            OpCode::GuardValue,
        ),
        // Pinning `w_dict` at null is a nullity test, and `pyjitpl.py
        // _establish_nullity` proves one with GUARD_ISNULL.  As a GUARD_VALUE
        // the guard's jitcounter keys on the *failing* value
        // (`compile.py make_a_counter_per_value`), which here is
        // whatever dictionary the assignment just allocated — a fresh address
        // every time, so no one value reaches `trace_eagerness` and the
        // has-a-dictionary continuation never gets a bridge.
        ShadowGuard::ExceptionDictIsNull(kind) => {
            let user =
                unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete_obj) };
            (
                walker_record_getfield_gc_r_uncached(
                    ctx,
                    obj,
                    crate::descr::w_exception_dict_descr_for(kind, user),
                ),
                ctx.trace_ctx.const_ref(0),
                OpCode::GuardIsnull,
            )
        }
    };
    // GUARD_ISNULL carries only the pointer; the null constant is still what
    // the box is replaced with, the way `_establish_nullity` does it.  The
    // concrete is stamped here because `stamp_guard_value_concrete` reads it
    // off a GUARD_VALUE's expected operand, which GUARD_ISNULL does not carry.
    if shadow_guard == OpCode::GuardIsnull {
        ctx.trace_ctx
            .set_opref_concrete(slot_op, majit_ir::Value::Ref(majit_ir::GcRef(0)));
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, shadow_guard, &[slot_op])?;
    } else {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, shadow_guard, &[slot_op, slot_const])?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(slot_op, slot_const);
    Ok(())
}

/// `callmethod.py LOAD_METHOD` method-cache fold for the
/// codewriter's method-form `LOAD_ATTR` residual.  The safety oracle is the
/// interpreter's `load_method_fast_path`: it declines custom
/// `__getattribute__`, uncacheable types, non-function descriptors, and
/// shadowing instance attributes.  On success the
/// walker emits the guards that keep that decision stable, then writes
/// `w_descr` as a green constant so the following `CALL` can use the existing
/// constant-callee inline path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_method_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, _version_tag, w_descr)) =
        (unsafe { pyre_interpreter::load_method_fast_path(concrete_obj, &name) })
    else {
        let cell = unsafe { pyre_interpreter::load_method_cell_fast_path(concrete_obj, &name) };
        return walker_fold_load_method_cell(ctx, op_pc, obj, concrete_obj, cell, dst, dst_bank);
    };
    if unsafe { resolve_inlinable_callee(w_descr) }.is_none() {
        return Ok(None);
    }
    // `space.type` reaches an exception's class through the kind registry when
    // the generic stub is still installed, and the `w_class` guard below can
    // only pin a class the slot actually holds.
    if !std::ptr::eq(unsafe { (*concrete_obj).w_class }, w_type) {
        return Ok(None);
    }
    // `load_method_fast_path` admits only the layouts the helper answers for;
    // keep the two in step so a new layout there cannot reach an emit that has
    // no shadowing guard for it.
    let Some(shadow) = (unsafe { walker_classify_shadow_guard(concrete_obj) }) else {
        return Ok(None);
    };

    // guard_class(obj, ob_type): pins the payload layout, so the `w_class` and
    // shadowing-slot reads below name the fields they were recorded against.
    // Pin the Python-level receiver class (`w_class`) exactly.  This is the
    // per-frame method namespace anchor: a subclass with the same instance
    // payload vtable side-exits instead of reusing the caller's method.
    // typeobject.py `promote(self.version_tag())`: class mutation or method
    // reassignment bumps `_version_tag`, so the old `w_descr` side-exits.
    walker_pin_stamped_instance_class(ctx, op_pc, obj, concrete_obj, w_type)?;

    walker_emit_shadow_guard(ctx, op_pc, obj, concrete_obj, shadow)?;

    let method_const = ctx.trace_ctx.const_ref(w_descr as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_const)?;
    Ok(Some(()))
}

/// `load_method_cell_fast_path`: the namespace entry is an `ObjectMutableCell`.
/// Pin the cell pointer under `_version_tag` and `getfield` `w_value`, so an
/// in-place method store stays visible without a new trace.
fn walker_fold_load_method_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    cell_hit: Option<(pyre_object::PyObjectRef, u64, pyre_object::PyObjectRef)>,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some((w_type, _version_tag, cell)) = cell_hit else {
        return Ok(None);
    };
    if !object_mutable_cell_payload_is_guardable(cell) {
        return Ok(None);
    }
    if !std::ptr::eq(unsafe { (*concrete_obj).w_class }, w_type) {
        return Ok(None);
    }
    let Some(shadow) = (unsafe { walker_classify_shadow_guard(concrete_obj) }) else {
        return Ok(None);
    };
    walker_pin_stamped_instance_class(ctx, op_pc, obj, concrete_obj, w_type)?;
    walker_emit_shadow_guard(ctx, op_pc, obj, concrete_obj, shadow)?;
    // Do not stamp the payload.  The following CALL must invoke whatever
    // `w_value` holds, not the function that was there at record time.
    let value = walker_read_object_mutable_cell_stamped(ctx, cell, false);
    // `load_method_cell_fast_path` admitted this name because the payload's
    // type carries `flag_method_descriptor`, which is what decides the
    // `(method, self)` pair the paired self-fold writes.  An in-place rebind
    // can replace the method with a `property`; the guard is what makes that
    // side-exit instead of binding a receiver to it.
    walker_guard_object_mutable_cell_payload(ctx, op_pc, value, cell)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// `LOAD_METHOD` classmethod fold for a type receiver (`Type.cmethod(...)`).
/// The safety oracle is [`pyre_interpreter::classmethod_on_type_fast_path`]: it
/// declines a custom metaclass, a metatype-defined name, an uncacheable type,
/// and any non-`classmethod` descriptor.  On success the walker pins the exact
/// type, its version tag, and the descriptor's `w_function?` slot, then writes
/// the classmethod's `__func__` as a green constant.  Because the method-load result is the plain `__func__` (not
/// a bound `Method`), the paired [`try_walker_fold_load_method_self`] runs
/// `compute_load_method_bound`, whose `is_type` + `is_exact_classmethod` arm
/// binds the type as `cls` — the same exactness this oracle applies, so the two
/// agree on a wrapper subclass.  The following `CALL` inlines `__func__(cls, ...)` — the
/// instance-method shape with the class in the receiver slot.
///
/// Carries the inline-depth restriction
/// [`try_walker_specialize_load_bound_method_attr`] documents: under the
/// single-frame collapse a fold guard inside an inlined callee sub-walk
/// resumes at the caller's CALL, re-running side effects.  The `getattr`
/// residual resumes past the call, so declining there re-runs nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_classmethod_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    if name.contains("__") {
        return Ok(None);
    }
    let Some((w_type, version_tag, w_descr, w_func)) =
        (unsafe { pyre_interpreter::classmethod_on_type_fast_path(concrete_obj, &name) })
    else {
        return Ok(None);
    };
    if unsafe { resolve_inlinable_callee(w_func) }.is_none() {
        return Ok(None);
    }

    // Pin the exact class.  The receiver IS the type, so a single GuardValue
    // anchors both the metaclass (exact `type`, via `is_type`) and the MRO the
    // classmethod lookup walks; the version tag below covers method reassignment.
    // typeobject.py `promote(self.version_tag())`: class mutation or rebinding
    // the attribute to a different descriptor in the class or any base bumps
    // `_version_tag`, so the pinned `__func__` side-exits.
    walker_guard_stamped_type_version(ctx, op_pc, obj, w_type)?;

    // What the version tag does NOT reach: re-initialising the classmethod in
    // place leaves the class dict, the descriptor's address, and every version
    // tag alone while replacing the callable this fold is about to bake.
    // `function.py:720` declares that slot `w_function?` for exactly this, and
    // `w_classmethod_set_func` forces the invalidation.
    walker_pin_descriptor_slot(
        ctx,
        op_pc,
        w_descr,
        crate::descr::classmethod_w_function_quasi_descr(),
    )?;

    let func_const = ctx.trace_ctx.const_ref(w_func as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_const)?;
    Ok(Some(()))
}

fn walker_read_object_mutable_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    cell: pyre_object::PyObjectRef,
) -> OpRef {
    walker_read_object_mutable_cell_stamped(ctx, cell, true)
}

/// `stamp` is false when a later op must not treat the payload as a green
/// constant.  A method call inlines whatever concrete it sees and guards that
/// identity; an in-place cell write would then fail the guard on every
/// iteration instead of calling the function the `getfield` just read.
fn walker_read_object_mutable_cell_stamped<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    cell: pyre_object::PyObjectRef,
    stamp: bool,
) -> OpRef {
    let cell_op = ctx.trace_ctx.const_ref(cell as i64);
    let value = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        cell_op,
        crate::descr::object_mutable_cell_value_descr(),
    );
    if stamp {
        let live = unsafe { (*(cell as *const pyre_object::celldict::ObjectMutableCell)).w_value };
        ctx.trace_ctx
            .set_opref_concrete(value, majit_ir::Value::Ref(majit_ir::GcRef(live as usize)));
    }
    value
}

/// Guard the class of a payload [`walker_read_object_mutable_cell`] just read.
///
/// The `getfield` keeps an in-place rebind visible, which is the whole point of
/// reading the cell; what it does not keep is the admission the oracle computed
/// against `type(w_value)` while recording -- `flag_method_descriptor` for a
/// method load, "no `__get__`, not a heaptype" for a plain type attribute.  An
/// in-place write is the one namespace change `_version_tag` does not report,
/// so a rebind from a function to a `property` would otherwise reach code that
/// already decided the descriptor protocol does not run.  This is the class
/// check the dispatch the fold replaced records anyway: `space.get` resolves
/// `__get__` on `type(w_descr)`, and the same-class rebind the live read exists
/// for passes it.
///
/// Asked before the fold emits anything: a tagged int has no `ob_type`, so its
/// payload cannot carry the class guard and the fold declines instead.
fn object_mutable_cell_payload_is_guardable(cell: pyre_object::PyObjectRef) -> bool {
    let live = unsafe { (*(cell as *const pyre_object::celldict::ObjectMutableCell)).w_value };
    !live.is_null()
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && unsafe { pyre_object::tagged_int::is_tagged_int(live) })
}

fn walker_guard_object_mutable_cell_payload<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    value: OpRef,
    cell: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    let live = unsafe { (*(cell as *const pyre_object::celldict::ObjectMutableCell)).w_value };
    let physical_type = unsafe { (*live).ob_type } as i64;
    let type_const = ctx.trace_ctx.const_int(physical_type);
    walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardClass, &[value, type_const])?;
    ctx.trace_ctx.heap_cache_mut().class_now_known(value);
    Ok(())
}

fn walker_read_int_mutable_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    cell: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let cell_op = ctx.trace_ctx.const_ref(cell as i64);
    let raw = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        cell_op,
        crate::descr::int_mutable_cell_value_descr(),
    );
    let live = unsafe { (*(cell as *const pyre_object::celldict::IntMutableCell)).intvalue };
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Int(live));
    let boxed = walker_box_int(ctx, op_pc, raw, live)?;
    let live_ptr = pyre_object::w_int_new(live) as i64;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(live, live_ptr));
    Ok(boxed)
}

/// `Cls.__name__` — `descr_getattribute`'s metatype data-descriptor arm for
/// the `type.__name__` getset.
///
/// [`pyre_interpreter::type_name_obj_fast_path`] admits only a class whose
/// metaclass is exactly `type`, which is the case the getset cannot be
/// replaced.  The slot is `typeobject.py name?`: `descr_set__name__` stores
/// a new string without `mutated()`, so the pin is this field rather than
/// `_version_tag?`.  A null slot (not materialized yet) declines; filling
/// it in allocates.
fn walker_fold_type_name<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if name != "__name__" {
        return Ok(None);
    }
    let Some((_metatype, _)) = (unsafe { pyre_interpreter::type_name_obj_fast_path(concrete_obj) })
    else {
        return Ok(None);
    };
    let w_type_const = walker_guard_stamped_ref(ctx, op_pc, obj, concrete_obj)?;
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        w_type_const,
        crate::descr::type_name_obj_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    // `record_quasiimmut_field` installs the watcher before it captures
    // `constantfieldbox` (`quasiimmut.py QuasiImmutDescr.__init__`). The
    // eligibility peek is only the non-null test; baking it would keep a
    // pre-rename pointer after `invalidate_then_store` completed with no
    // watcher. Re-read after the record, the same order as the `getfield`
    // that follows `record_quasiimmut_field` on the rewritten load.
    let w_name = unsafe { pyre_object::typeobject::w_type_peek_name_obj(concrete_obj) };
    if w_name.is_null() {
        return Ok(None);
    }
    let name_const = ctx.trace_ctx.const_ref(w_name as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, 'r', name_const)?;
    Ok(Some(()))
}

/// `LOAD_ATTR` of a type attribute stored in a `MutableCell`.  The cell
/// pointer is constant under the type's `_version_tag`; the payload is a
/// `getfield`, so an in-place write stays visible.
fn walker_fold_type_attr_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((w_type, _version_tag, cell)) =
        (unsafe { pyre_interpreter::type_attr_cell_fast_path(concrete_obj, Wtf8::new(name)) })
    else {
        return Ok(None);
    };
    if !unsafe { pyre_object::celldict::is_int_mutable_cell(cell) }
        && !object_mutable_cell_payload_is_guardable(cell)
    {
        return Ok(None);
    }
    walker_guard_stamped_type_version(ctx, op_pc, obj, w_type)?;
    let value = if unsafe { pyre_object::celldict::is_int_mutable_cell(cell) } {
        // An `IntMutableCell` only ever holds an int: `write_cell`'s in-place
        // arm stores `intval`, so the payload cannot change shape and the
        // boxing below is the whole of the binding.
        walker_read_int_mutable_cell(ctx, op_pc, cell)?
    } else {
        let value = walker_read_object_mutable_cell(ctx, cell);
        // `type_attr_cell_fast_path` admitted this name because the payload's
        // type has no `__get__` -- the arm where `get` returns the value
        // unchanged.  An in-place rebind can put a descriptor there.
        // `GuardClass` on `ob_type` is the vtable.  User instances share
        // one (`typedef.py _getusercls` of `W_ObjectObject`), so a rebind
        // to a descriptor of another class still passes it.  `space.get`
        // finds `__get__` on `w_class`, and that guard resumes at this
        // LOAD_ATTR.
        walker_guard_object_mutable_cell_payload(ctx, op_pc, value, cell)?;
        let live = unsafe { (*(cell as *const pyre_object::celldict::ObjectMutableCell)).w_value };
        let live_class = unsafe { (*live).w_class };
        if !live_class.is_null() {
            walker_pin_instance_w_class(ctx, op_pc, value, live_class)?;
        }
        // A heap type can grow `__get__` without this receiver's version tag
        // moving. Pin the payload type so that edit invalidates the trace.
        if let Some(value_type) = pyre_interpreter::typedef::r#type(live)
            && unsafe { pyre_object::w_type_is_heaptype(value_type.as_ptr()) }
        {
            let type_const = ctx.trace_ctx.const_ref(value_type.as_ptr() as i64);
            walker_pin_type_version_tag(ctx, op_pc, type_const)?;
        }
        value
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, 'r', value)?;
    Ok(Some(()))
}

/// `LOAD_ATTR` of a slot wrapper stored on a type whose metaclass is `type`.
///
/// `bind_slot_wrapper` returns the wrapper when the instance is null, which
/// is the value `float.__add__` (and every sibling slot) has on the type.
/// The wrapper type has `__get__`, so `type_attr_value_fast_path` declines it.
/// A cell-backed entry is left alone: `lookup_in_type` would unwrap a payload
/// `write_cell` can replace without moving the version tag. The metaclass
/// test is pointer equality with `typedef::w_type`; a metaclass that supplies
/// its own `__getattribute__` stays on the residual. A same-named metatype
/// entry preempts the class slot only when it is a data descriptor
/// (`descr_getattribute`, the same predicate as `type_attr_value_fast_path`).
/// `object.__lt__` sits on `type`'s MRO and is not one, so the class slot
/// still folds.
fn walker_fold_slot_wrapper_on_type<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    name: &str,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if !unsafe { pyre_object::is_type(concrete_obj) } {
        return Ok(None);
    }
    let w_type = pyre_interpreter::typedef::w_type();
    let metatype = unsafe { (*concrete_obj).w_class };
    if w_type.is_null() || !std::ptr::eq(metatype, w_type) {
        return Ok(None);
    }
    if unsafe { pyre_object::typeobject::w_type_get_version_tag(concrete_obj) } == 0 {
        return Ok(None);
    }
    // The cell fold owns an entry `type_attr_cell_fast_path` admits. A slot
    // wrapper's type has `__get__`, so that fast path declines it;
    // `type_attr_is_cell_backed` is what still sees the cell.
    if unsafe { pyre_interpreter::type_attr_cell_fast_path(concrete_obj, Wtf8::new(name)) }
        .is_some()
        || unsafe { type_attr_is_cell_backed(concrete_obj, name) }
    {
        return Ok(None);
    }
    // Presence is not enough: `type`'s MRO includes `object.__lt__` /
    // `object.__eq__`, and those lose to the class's own slot.
    if unsafe { pyre_interpreter::baseobjspace::type_lookup_is_data_descr(metatype, name) } {
        return Ok(None);
    }
    let Some(value) = (unsafe { pyre_interpreter::lookup_in_type(concrete_obj, name) }) else {
        return Ok(None);
    };
    // `SLOT_WRAPPER_TYPE` is not subclassable. Pointer equality on `ob_type`
    // is the exact test; `is_slot_wrapper` walks `py_type_check`.
    if value.is_null()
        || !unsafe { std::ptr::eq((*value).ob_type, &pyre_interpreter::SLOT_WRAPPER_TYPE) }
    {
        return Ok(None);
    }
    walker_guard_stamped_type_version(ctx, op_pc, obj, concrete_obj)?;
    let value_const = ctx.trace_ctx.const_ref(value as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// Fold `LOAD_ATTR` on a type receiver when
/// [`pyre_interpreter::type_attr_value_fast_path`] resolves
/// `typeobject.py` `getattribute`'s `space.get(w_value, w_None, self)` to a
/// value the trace can name.  The exact
/// receiver, its version tag, and whatever the binding arm depends on are
/// pinned before the value is written as a green constant.  [`pyre_interpreter::mutated`] recursively invalidates
/// subclasses, so the one receiver pin covers reassignment or deletion on any
/// base class as well.
///
/// A name the metatype answers with a data descriptor is refused by
/// [`pyre_interpreter::type_attr_value_fast_path`]. `__name__` is that case
/// and is read live by [`walker_fold_type_name`]. A slot wrapper on the type
/// is folded by [`walker_fold_slot_wrapper_on_type`] before that. Every other
/// refused name falls through to the cell fold.
///
/// The name needs no operand guard: the codewriter baked its `co_names` index
/// into the residual.  This read-only, present-attribute fold cannot raise, so
/// unlike the classmethod method-load fold it is safe inside an inlined callee
/// sub-walk; resuming past it cannot repeat a side effect.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_type_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, _version_tag, w_value, binding)) = (unsafe {
        pyre_interpreter::type_attr_value_fast_path(concrete_obj, Wtf8::new(name.as_str()))
    }) else {
        if walker_fold_slot_wrapper_on_type(ctx, op_pc, obj, concrete_obj, name.as_str(), dst)?
            .is_some()
        {
            return Ok(Some(()));
        }
        if walker_fold_type_name(ctx, op_pc, obj, concrete_obj, name.as_str(), dst)?.is_some() {
            return Ok(Some(()));
        }
        return walker_fold_type_attr_cell(ctx, op_pc, obj, concrete_obj, name.as_str(), dst);
    };

    walker_guard_stamped_type_version(ctx, op_pc, obj, w_type)?;
    walker_pin_type_attr_binding(ctx, op_pc, binding)?;

    let value_const = ctx.trace_ctx.const_ref(w_value as i64);
    write_residual_call_result_to_dst(ctx, op_pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// Fold the `getattr` residual for a receiver whose name resolves to a
/// function or method descriptor on its type — the `lst.append`
/// shape [`try_walker_specialize_load_method_attr`] declines because upstream
/// restricts its `[w_descr, w_obj]` push to `flag_method_descriptor` types,
/// and the bare `g = o.m` read, which has no `load_method_self` after it to
/// reach that push at all.
///
/// Both engines materialise a `Method` here (`space.getattr`), but PyPy traces
/// *through* that `getattr` — it is ordinary RPython — so the type lookup folds
/// to a constant under `guard_class` + the version pin and the `Method` itself
/// virtualizes away: its optimized LOAD_METHOD emits no ops at all in a
/// steady-state loop (`pypy/objspace/std/callmethod.py`). pyre's `getattr`
/// is an opaque `CALL_MAY_FORCE` residual, which additionally drags a
/// `GUARD_NOT_FORCED` (forcing the virtualizable frame) and a `GUARD_NO_EXCEPTION`
/// through every iteration. This fold reproduces PyPy's shape directly:
///
///   guard_class(obj, ob_type)
///   guard_value(getfield(obj, w_class), the type)
///   guard_value(getfield(the type, version_tag), the tag)
///   guard_value(getfield(obj, map), the map)   — a dict-bearing receiver only
///   new_with_vtable(Method) + setfield(w_function/w_self/w_class/header)
///
/// The guards make `lookup_in_type` constant exactly as the version-tag promote
/// does upstream, and the emitted `Method` is dead once the following `CALL`
/// folds — the append fold reads `w_function` / `w_self` straight back off it.
///
/// Returns `None` (fall through to the residual, SAFE) for every shape
/// [`pyre_interpreter::baseobjspace::bound_method_attr_fast_path`] declines.
///
/// Inside an inlined callee sub-walk the fold is restricted to a depth whose
/// guards resume at their own callee coordinate
/// ([`walker_inline_guard_resumes_in_callee`]).  Under the single-frame
/// collapse the reason [`try_walker_orthodox_list_append`] documents applies: a
/// guard resumes at the caller's CALL boundary, so a failure re-runs the callee
/// from its entry and doubles any side effect it sequenced before this
/// `LOAD_ATTR`. The residual resumes past the call instead, so declining there
/// re-runs nothing extra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_bound_method_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_descr, owes_shadow_guard)) = (unsafe {
        pyre_interpreter::baseobjspace::bound_method_attr_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    // Classified before anything is emitted: a receiver whose shadowing
    // precondition has no guard must decline, not emit half a fold.
    let shadow = if owes_shadow_guard {
        let Some(shadow) = (unsafe { walker_classify_shadow_guard(concrete_obj) }) else {
            return Ok(None);
        };
        Some(shadow)
    } else {
        None
    };

    let Some(header) = super_attr_method_header(w_descr) else {
        return Ok(None);
    };
    walker_emit_constant_descr_bound_method(
        ctx,
        op_pc,
        obj,
        concrete_obj,
        w_type,
        w_descr,
        header,
        shadow,
        dst,
        dst_bank,
        None,
    )?;
    Ok(Some(()))
}

/// LOAD_SPECIAL's `__enter__` / `__exit__` lookup, folded to the shape PyPy's
/// `BEFORE_WITH` has for free: `space.lookup` is ordinary RPython there, so a
/// promoted type makes the descriptor constant and the bound method it builds
/// virtualizes into the `CALL` that immediately consumes it.  pyre's
/// `load_special` is an opaque `CALL_MAY_FORCE` residual instead, which drags a
/// `GUARD_NOT_FORCED` (forcing the virtualizable frame) and a
/// `GUARD_NO_EXCEPTION` through every `with`, and materialises the context
/// manager because it escapes into the call.
///
/// The lookup reads the type only, so the guards are just the receiver's class
/// and the type's version tag — no instance-dict shadowing guard, which is what
/// [`try_walker_specialize_load_bound_method_attr`] additionally needs.
///
/// Returns `None` (fall through to the residual, SAFE) for every shape
/// [`pyre_interpreter::baseobjspace::load_special_fast_path`] declines,
/// including the `async with` discriminants.
///
/// Inside an inlined callee sub-walk the fold carries the depth restriction
/// [`try_walker_specialize_load_bound_method_attr`] documents.
pub(crate) fn try_walker_specialize_load_special_method<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    method_kind: i64,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    // The codewriter emits the raw `SpecialMethod` oparg; `async with` reaches
    // `emit_abort_permanent` there and never records this residual, so the two
    // synchronous discriminants are the whole domain.
    let name = match u8::try_from(method_kind)
        .ok()
        .and_then(|kind| pyre_interpreter::bytecode::SpecialMethod::try_from(kind).ok())
    {
        Some(pyre_interpreter::bytecode::SpecialMethod::Enter) => "__enter__",
        Some(pyre_interpreter::bytecode::SpecialMethod::Exit) => "__exit__",
        _ => return Ok(None),
    };
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some((w_type, _version_tag, w_descr, attr_cell)) =
        (unsafe { pyre_interpreter::baseobjspace::load_special_fast_path(concrete_obj, name) })
    else {
        return Ok(None);
    };

    let Some(header) = super_attr_method_header(w_descr) else {
        return Ok(None);
    };
    let cell_guard = (!attr_cell.is_null()).then_some((attr_cell, w_descr));
    walker_emit_constant_descr_bound_method(
        ctx,
        op_pc,
        obj,
        concrete_obj,
        w_type,
        w_descr,
        header,
        None,
        dst,
        dst_bank,
        cell_guard,
    )?;
    Ok(Some(()))
}

/// Emit the guards and the inline `Method` a constant-descriptor bind reduces
/// to, given a `w_descr` some caller has already proven binds through
/// `w_method_new(w_descr, obj, w_type)`, with the descriptor's header stamp.
///
/// Three guards make that reduction reproducible: `guard_class` on the physical
/// layout the `w_class` read needs, `guard_value` on the Python-level class so
/// a subclass reaching the same layout side-exits, and the type version tag
/// (`typeobject.py promote(self.version_tag())`) so rebinding the name on the
/// type retires the trace.  An optional instance-dict shadowing guard is the
/// fourth when the caller looked the name up through `LOAD_ATTR`;
/// `LOAD_SPECIAL` is type-only (`baseobjspace.py lookup`) and passes `None`.
/// The `Method` is then built inline rather than called for, which is what
/// lets the consuming `CALL` virtualize it away.
///
/// Shared by `LOAD_ATTR`, builtin `getattr`, and `LOAD_SPECIAL`'s type-only
/// `__enter__` / `__exit__` lookup, each with its own lookup preconditions.
#[allow(clippy::too_many_arguments)]
fn walker_emit_constant_descr_bound_method<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
    w_descr: pyre_object::PyObjectRef,
    header: (bool, pyre_object::PyObjectRef),
    shadow: Option<ShadowGuard>,
    dst: usize,
    dst_bank: char,
    attr_cell: Option<(pyre_object::PyObjectRef, pyre_object::PyObjectRef)>,
) -> Result<(), DispatchError> {
    let w_type_const = walker_pin_stamped_instance_class(ctx, op_pc, obj, concrete_obj, w_type)?;
    // The version-tag pin does not cover an in-place cell write.  Same
    // getfield and `guard_value` as `ExceptionInlineReceiverGuard`'s attr_cell.
    if let Some((cell, expected)) = attr_cell {
        super::walker_promote_object_mutable_cell(ctx, op_pc, cell, expected)?;
    }

    if let Some(shadow) = shadow {
        walker_emit_shadow_guard(ctx, op_pc, obj, concrete_obj, shadow)?;
    }

    // `baseobjspace::get` binds method descriptors through
    // `builtin_bound_method_new`, including its w_class / w_module stores.
    // Share the descriptor binding body with super; only lookup differs.
    let method_op = walker_emit_super_attr_binding(
        ctx,
        op_pc,
        obj,
        concrete_obj,
        w_type,
        w_type_const,
        w_descr,
        SuperAttrBinding::Method {
            w_function: w_descr,
            header,
            bind_to_class: false,
            slot_pin: None,
        },
    )?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, method_op)
}

/// Fold `bh_load_method_self_fn(obj, attr, code, name_idx)` once both the
/// receiver and the attribute are concrete.  The method-attribute fold above
/// already guards class, type version, and instance map.  `callmethod.py
/// LOAD_METHOD` decides both halves under one test (`f.pushvalue(w_descr);
/// f.pushvalue(w_obj)`); this residual consumes that same
/// `load_method_fast_path` verdict before re-deriving through
/// `compute_load_method_bound`, so the two walker folds cannot disagree.
/// A plain instance-method bind writes the original red receiver box, not a
/// baked `ConstRef`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_fold_load_method_self<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    attr: OpRef,
    _attr_reg: usize,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(concrete_attr) = walker_concrete_ref_object(ctx, attr) else {
        return Ok(None);
    };
    // `compute_load_method_bound` answers PY_NULL for an already-bound method
    // without inspecting anything else, so pinning the attribute's class is
    // the whole precondition.  Left as a residual this is a second per-iteration
    // call on top of the `getattr` one (`lst.append(x)` pays both).
    if unsafe { pyre_object::is_method(concrete_attr) } {
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        let class_pinned = attr.is_constant() || ctx.trace_ctx.heap_cache().is_class_known(attr);
        if !class_pinned {
            // Under the single-frame collapse a guard here would resume at the
            // caller's CALL, re-running whatever that callee already did;
            // leave those to the residual (which resumes past the call).
            if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
                return Ok(None);
            }
            walker_guard_stamped_class(ctx, op_pc, attr, method_type_addr)?;
        }
        let null_const = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, null_const)?;
        return Ok(Some(()));
    }
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    // Same oracle the attribute fold asked: when it still answers the
    // descriptor this residual was paired with, bind the receiver.
    if let Some((_, _, w_descr)) =
        unsafe { pyre_interpreter::baseobjspace::load_method_fast_path(concrete_obj, &name) }
    {
        if std::ptr::eq(w_descr, concrete_attr) {
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, obj)?;
            return Ok(Some(()));
        }
    }
    if let Some((_, _, cell)) =
        unsafe { pyre_interpreter::load_method_cell_fast_path(concrete_obj, &name) }
    {
        let w_descr = unsafe { pyre_object::celldict::unwrap_cell(cell) };
        if std::ptr::eq(w_descr, concrete_attr) {
            write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, obj)?;
            return Ok(Some(()));
        }
    }
    let bound =
        pyre_interpreter::eval::compute_load_method_bound(concrete_obj, concrete_attr, &name);
    let bound_op = if std::ptr::eq(bound, concrete_obj) {
        obj
    } else if bound == pyre_object::PY_NULL {
        // The fallback arm pushes the receiver slot unconditionally empty
        // (`callmethod.py LOAD_METHOD` `f.pushvalue_none()`), including when
        // an instance attribute shadows the method.  Baking the constant
        // keeps that arm folded rather than paying a residual for it.
        ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64)
    } else {
        return Ok(None);
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, bound_op)?;
    Ok(Some(()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_load_super_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    global_super: OpRef,
    self_obj: OpRef,
    cls: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_super) = walker_concrete_ref_object(ctx, global_super) else {
        return Ok(None);
    };
    // Only the builtin `super` resolves through `W_Super.getattribute`; a
    // rebound global names some other callable entirely.
    if !pyre_interpreter::builtins::is_builtin_super_type(concrete_super) {
        return Ok(None);
    }
    let Some(concrete_cls) = walker_concrete_ref_object(ctx, cls) else {
        return Ok(None);
    };
    let Some(concrete_self) = walker_concrete_ref_object(ctx, self_obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let python_free = unsafe {
        pyre_interpreter::baseobjspace::super_attr_fast_path(concrete_cls, concrete_self, &name)
    };
    let apparent = if python_free.is_none() {
        walker_apparent_super_class(concrete_cls, concrete_self)
    } else {
        None
    };
    let (objtype, w_descr, class_mode) =
        if let Some((objtype, _, w_descr, class_mode)) = python_free {
            (objtype, w_descr, class_mode)
        } else if let Some(apparent) = apparent {
            let Some((_, w_descr, class_mode)) = (unsafe {
                pyre_interpreter::baseobjspace::super_attr_proxy_fast_path(
                    concrete_cls,
                    apparent.objtype,
                    concrete_self,
                    &name,
                )
            }) else {
                return Ok(None);
            };
            (apparent.objtype, w_descr, class_mode)
        } else {
            return Ok(None);
        };
    let Some(binding) = super_attr_binding(w_descr, concrete_self, class_mode) else {
        return Ok(None);
    };

    // Which callable `super` names and which class the walk starts after are
    // both baked into the emitted body, so both are pinned.
    walker_guard_stamped_ref_unless_const(ctx, op_pc, global_super, concrete_super)?;
    let value_op = if let Some(apparent) = apparent {
        walker_emit_apparent_super_attr_result(
            ctx,
            op_pc,
            self_obj,
            cls,
            concrete_self,
            concrete_cls,
            apparent,
            w_descr,
            binding,
        )?
    } else {
        walker_emit_super_attr_result(
            ctx,
            op_pc,
            self_obj,
            cls,
            concrete_self,
            concrete_cls,
            objtype,
            w_descr,
            class_mode,
            binding,
        )?
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value_op)?;
    Ok(Some(()))
}

/// A Python-free result of binding the descriptor found by the `super` MRO
/// suffix walk.  Descriptor identity is protected by the receiver type's
/// version tag; `slot_pin` additionally protects an in-place replacement of a
/// staticmethod/classmethod wrapper's `_immutable_fields_` callable slot.
enum SuperAttrBinding {
    Constant {
        value: pyre_object::PyObjectRef,
        slot_pin: Option<majit_ir::DescrRef>,
    },
    Method {
        w_function: pyre_object::PyObjectRef,
        header: (bool, pyre_object::PyObjectRef),
        bind_to_class: bool,
        slot_pin: Option<majit_ir::DescrRef>,
    },
}

/// Classify the exact descriptor shapes for which PyPy's
/// `space.get(w_descr, descr_obj, objtype)` runs no Python.
fn super_attr_binding(
    w_descr: pyre_object::PyObjectRef,
    concrete_self: pyre_object::PyObjectRef,
    class_mode: bool,
) -> Option<SuperAttrBinding> {
    let descr_ob_type = unsafe { (*w_descr).ob_type };
    if std::ptr::eq(descr_ob_type, &pyre_interpreter::FUNCTION_TYPE as *const _)
        || std::ptr::eq(
            descr_ob_type,
            &pyre_interpreter::METHOD_DESCRIPTOR_TYPE as *const _,
        )
    {
        if class_mode {
            return Some(SuperAttrBinding::Constant {
                value: w_descr,
                slot_pin: None,
            });
        }
        return Some(SuperAttrBinding::Method {
            w_function: w_descr,
            header: super_attr_method_header(w_descr)?,
            bind_to_class: false,
            slot_pin: None,
        });
    }
    if unsafe { pyre_object::function::is_exact_staticmethod(w_descr) } {
        let mut value = unsafe { pyre_object::function::w_staticmethod_get_func(w_descr) };
        if value.is_null() {
            value = pyre_object::w_none();
        }
        return Some(SuperAttrBinding::Constant {
            value,
            slot_pin: Some(crate::descr::staticmethod_w_function_quasi_descr()),
        });
    }
    if unsafe { pyre_object::function::is_exact_classmethod(w_descr) } {
        let w_function = unsafe { pyre_object::function::w_classmethod_get_func(w_descr) };
        if w_function.is_null() {
            return None;
        }
        let header = pyre_object::get_instantiate(&pyre_object::function::METHOD_TYPE);
        if header.is_null() {
            return None;
        }
        return Some(SuperAttrBinding::Method {
            w_function,
            header: (false, header),
            bind_to_class: true,
            slot_pin: Some(crate::descr::classmethod_w_function_quasi_descr()),
        });
    }
    // `get`'s slot-wrapper arm, which class mode never reaches: there the
    // descriptor comes back unchanged.  Its instance check is a precondition of
    // the binding rather than part of it, so it is settled here against the
    // receiver whose class the emitted guards pin; a receiver it rejects
    // declines and raises in the interpreter.
    if !class_mode
        && unsafe {
            pyre_interpreter::baseobjspace::super_attr_slot_wrapper_binds(w_descr, concrete_self)
        }
    {
        return Some(SuperAttrBinding::Method {
            w_function: w_descr,
            header: super_attr_method_header(w_descr)?,
            bind_to_class: false,
            slot_pin: None,
        });
    }
    if unsafe {
        pyre_interpreter::baseobjspace::super_attr_returns_descr_unchanged(w_descr, class_mode)
    } {
        return Some(SuperAttrBinding::Constant {
            value: w_descr,
            slot_pin: None,
        });
    }
    None
}

/// The two words that separate the `Method` `get` builds for the descriptor
/// typedefs the `super` fold binds.
///
/// A `function` binds through `w_method_new`, which leaves `w_module` null and
/// lets the allocation's own header stand.  The other two arms bind through
/// `restamped_bound_method_new`, which is that same call followed by two
/// stores: the Python-visible class becomes `builtin_function_or_method` for a
/// `method_descriptor` and `method-wrapper` for a slot wrapper, and `w_module`
/// becomes `None`.  The payload is identical in all three, so one emission
/// serves them once these two words are chosen.
///
/// `None` when the chosen type object is not registered, which is resolved
/// here — before the caller emits anything — so the decline leaves the trace
/// untouched.
fn super_attr_method_header(
    w_descr: pyre_object::PyObjectRef,
) -> Option<(bool, pyre_object::PyObjectRef)> {
    let restamped_class = if unsafe { pyre_interpreter::is_method_descriptor(w_descr) } {
        Some(&pyre_interpreter::BUILTIN_FUNCTION_TYPE)
    } else if unsafe { pyre_interpreter::is_slot_wrapper(w_descr) } {
        Some(&pyre_interpreter::METHOD_WRAPPER_TYPE)
    } else {
        None
    };
    let header = match restamped_class {
        Some(ty) => pyre_interpreter::typedef::gettypeobject(ty),
        None => pyre_object::get_instantiate(&pyre_object::function::METHOD_TYPE),
    };
    (!header.is_null()).then_some((restamped_class.is_some(), header))
}

/// The body `super(cls, self).name` compiles to, once
/// `baseobjspace.rs super_attr_fast_path` has settled which class the MRO
/// suffix walk answers with (`objtype`) and what it finds there (`w_descr`).
///
/// Emitting starts here: every operand this needs is already proved, so a
/// caller that declines does so with the trace untouched.
///
/// Shared by the two spellings that reach the same lookup — `LOAD_SUPER_ATTR`,
/// and an attribute load on a proxy an earlier op built
/// ([`try_walker_specialize_load_attr_on_super`]).  Which callable `super`
/// names is the caller's question, because the two prove it differently: the
/// opcode form pins the global it loaded, while the proxy form has a
/// `GuardClass` on the proxy itself, which no other type can pass.
#[allow(clippy::too_many_arguments)]
fn walker_emit_super_attr_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    self_obj: OpRef,
    cls: OpRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    w_descr: pyre_object::PyObjectRef,
    class_mode: bool,
    binding: SuperAttrBinding,
) -> Result<OpRef, DispatchError> {
    let objtype_const = walker_emit_super_attr_lookup_guards(
        ctx,
        op_pc,
        self_obj,
        cls,
        concrete_self,
        concrete_cls,
        objtype,
        class_mode,
    )?;

    walker_emit_super_attr_binding(
        ctx,
        op_pc,
        self_obj,
        concrete_self,
        objtype,
        objtype_const,
        w_descr,
        binding,
    )
}

/// Bind the descriptor found by the MRO suffix walk after the caller has
/// emitted the particular `_super_check` proof that supplied `objtype`.
fn walker_emit_super_attr_binding<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    self_obj: OpRef,
    concrete_self: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    objtype_const: OpRef,
    w_descr: pyre_object::PyObjectRef,
    binding: SuperAttrBinding,
) -> Result<OpRef, DispatchError> {
    let slot_pin = match &binding {
        SuperAttrBinding::Constant { slot_pin, .. } | SuperAttrBinding::Method { slot_pin, .. } => {
            slot_pin.clone()
        }
    };
    if let Some(field) = slot_pin {
        walker_pin_descriptor_slot(ctx, op_pc, w_descr, field)?;
    }

    let (w_function, method_header, bind_to_class) = match binding {
        SuperAttrBinding::Constant { value, .. } => {
            return Ok(ctx.trace_ctx.const_ref(value as i64));
        }
        SuperAttrBinding::Method {
            w_function,
            header,
            bind_to_class,
            ..
        } => (w_function, header, bind_to_class),
    };

    // `get(w_descr, self, objtype)` is `w_method_new(w_descr, self, objtype)`
    // plus the header stamp its allocation performs (`ob_type` comes from the
    // NewWithVtable's size descr).  [`super_attr_method_header`] has already
    // picked the header class, and its flag says whether the two extra stores
    // `builtin_bound_method_new` performs are owed on top.
    let (restamps_header, header_w_class_obj) = method_header;
    let func_const = ctx.trace_ctx.const_ref(w_function as i64);
    let header_w_class = ctx.trace_ctx.const_ref(header_w_class_obj as i64);
    let bound_self = if bind_to_class {
        objtype_const
    } else {
        self_obj
    };
    let method_op = crate::helpers::emit_bound_method_inline(
        ctx.trace_ctx,
        func_const,
        bound_self,
        objtype_const,
        header_w_class,
    );
    if restamps_header {
        // `w_method_new` leaves `w_module` null and a virtual reads an
        // unwritten field as null, so only the restamping arms owe the slot a
        // store.
        let module_descr = crate::descr::method_w_module_descr();
        let module_index = module_descr.index();
        let none_const = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[method_op, none_const],
            module_descr,
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(method_op, module_index, none_const);
    }
    // The physical layout is `Method` either way: `restamped_bound_method_new`
    // restamps the Python-visible `w_class`, not `ob_type`.
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    ctx.trace_ctx.heap_cache_mut().class_now_known(method_op);
    // The concrete bound method the walker's own execution must observe; a
    // fresh `Method` per evaluation is what `getattribute` produces anyway, so
    // the trace allocating its own is not an identity divergence.
    let concrete_bound_self = if bind_to_class {
        objtype
    } else {
        concrete_self
    };
    let bound = if restamps_header {
        pyre_interpreter::restamped_bound_method_new(
            w_function,
            concrete_bound_self,
            objtype,
            header_w_class_obj,
        )
    } else {
        pyre_object::w_method_new(w_function, concrete_bound_self, objtype)
    };
    ctx.trace_ctx.set_opref_concrete(
        method_op,
        majit_ir::Value::Ref(majit_ir::GcRef(bound as usize)),
    );
    Ok(method_op)
}

/// Attribute-binding twin for `_super_check`'s apparent-`__class__` arm.
/// `walker_emit_super_attr_lookup_guards` assumes `self.w_class == objtype`;
/// the mapdict lookup is the proof here instead, while the MRO suffix binding
/// after that point is identical.
#[allow(clippy::too_many_arguments)]
fn walker_emit_apparent_super_attr_result<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    self_obj: OpRef,
    cls: OpRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
    apparent: ApparentSuperClass,
    w_descr: pyre_object::PyObjectRef,
    binding: SuperAttrBinding,
) -> Result<OpRef, DispatchError> {
    walker_guard_stamped_ref_unless_is(ctx, op_pc, cls, concrete_cls)?;
    let objtype_const =
        walker_guard_apparent_super_class(ctx, op_pc, self_obj, concrete_self, apparent)?;
    walker_emit_super_attr_binding(
        ctx,
        op_pc,
        self_obj,
        concrete_self,
        apparent.objtype,
        objtype_const,
        w_descr,
        binding,
    )
}

/// Emit the guards that make a recording-time `super` MRO suffix answer valid
/// on every execution.  Kept separate from result binding so a Python property
/// getter can enter the ordinary inline-call path after the same lookup guards.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walker_emit_super_attr_lookup_guards<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    self_obj: OpRef,
    cls: OpRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    class_mode: bool,
) -> Result<OpRef, DispatchError> {
    // The class the walk starts after is baked into the emitted body.
    // A virtual proxy's `super_type` field is already that constant; a
    // second GUARD_VALUE of the GETFIELD box is a tautology.
    walker_guard_stamped_ref_unless_is(ctx, op_pc, cls, concrete_cls)?;

    let objtype_const = ctx.trace_ctx.const_ref(objtype as i64);
    if class_mode {
        // `_super_check`'s first arm: the receiver is the class whose MRO is
        // walked.  Pin that class object itself; its `w_class` is a metaclass
        // and is not the namespace anchor this lookup uses.
        walker_guard_stamped_ref_unless_const(ctx, op_pc, self_obj, objtype)?;
    } else {
        // guard_class(self, ob_type): the physical layout the `w_class` read
        // below needs.
        let phys_type = unsafe { (*concrete_self).ob_type } as i64;
        walker_guard_stamped_class(ctx, op_pc, self_obj, phys_type)?;

        // Pin the Python-level class exactly: a subclass reaching the same
        // physical layout has its own MRO suffix after `cls`, and the
        // slot-wrapper binding was settled against this class.  It also decides
        // instance mode against class mode, whose arm above pins the receiver
        // to `objtype` itself.
        //
        // In the opcode spelling this constant IS `objtype`, because
        // `super_attr_fast_path` declines a receiver whose `w_class` is
        // anything else -- there this guard is what proves the class the walk
        // used.  A proxy proves that from its own `w_objtype` field instead,
        // and its receiver's class may be unrelated.
        //
        // The cached pin: `two_arg_super_call` / `bare_super_virtual` already
        // ran this on the same receiver when they emitted the virtual proxy,
        // and a second uncached GETFIELD + GUARD_VALUE was the extra name-bound
        // cost.  `walker_pin_instance_w_class` reuses the heapcache box.
        walker_pin_instance_w_class(ctx, op_pc, self_obj, unsafe { (*concrete_self).w_class })?;
    }

    // typeobject.py `promote(self.version_tag())`.  Every class the suffix
    // walk reads is an ancestor of `objtype` and `mutated()` recurses into
    // subclasses, so this one tag covers a dict store or a `__bases__`
    // reassignment anywhere in that suffix.
    walker_pin_type_version_tag(ctx, op_pc, objtype_const)?;
    Ok(objtype_const)
}

/// `descriptor.py W_Super.getattribute` for a proxy the trace already holds —
/// the `su.name` half of the `su = super(...); su.name(...)` spelling, which
/// `LOAD_SUPER_ATTR` never sees because the name binding split the two.
///
/// Left alone this is an opaque `getattr_fn` MRO walk per iteration, and being
/// may-force it also wipes the trace's heap-field cache.  What replaces it is
/// the same body [`try_walker_specialize_load_super_attr`] emits: the two
/// operands come out of the proxy instead of off the stack.
///
/// Reading them is free where it matters.  When the proxy is the virtual
/// [`try_walker_specialize_two_arg_super_call`] emitted, `opimpl_getfield_gc_r`
/// answers from that emission's own `SetfieldGc` cache and no op is recorded at
/// all -- which is also what lets the allocation die, since a virtual whose
/// every read is answered has nothing left to materialise for.
///
/// `GuardClass(su, SUPER_TYPE)` matches the exact builtin layout.
/// `allocate_instance` stamps `SUPER_USER_TYPE` on a subclass
/// (`typedef.py` `_getusercls`), so the guard already excludes it. The
/// `w_class` pin stays because `walker_exact_builtin_class` records the
/// canonical Python class, and a walker-emitted proxy is virtual and
/// already carries it.
pub(crate) fn try_walker_specialize_load_attr_on_super<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some(concrete_proxy) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::descriptor::is_super(concrete_proxy) } {
        return Ok(None);
    }
    let Some(proxy_w_class) = (unsafe { walker_exact_builtin_class(concrete_proxy) }) else {
        return Ok(None);
    };
    let concrete_cls = unsafe { pyre_object::descriptor::w_super_get_type(concrete_proxy) };
    let concrete_self = unsafe { pyre_object::descriptor::w_super_get_obj(concrete_proxy) };
    let objtype = unsafe { pyre_object::descriptor::w_super_get_obj_type(concrete_proxy) };
    // `super_attr_proxy_fast_path` refuses a null receiver (the unbound
    // `super(C)` proxy), `__class__` / `__dict__`, an uncacheable type and a
    // name no MRO suffix answers -- every shape this must not emit.
    let Some((_version_tag, w_descr, class_mode)) = (unsafe {
        pyre_interpreter::baseobjspace::super_attr_proxy_fast_path(
            concrete_cls,
            objtype,
            concrete_self,
            name,
        )
    }) else {
        return Ok(None);
    };
    let Some(binding) = super_attr_binding(w_descr, concrete_self, class_mode) else {
        return Ok(None);
    };

    let (self_op, cls_op) = walker_guard_and_read_super_proxy(
        ctx,
        op_pc,
        obj,
        proxy_w_class,
        concrete_self,
        concrete_cls,
        objtype,
    )?;
    let value_op = walker_emit_super_attr_result(
        ctx,
        op_pc,
        self_op,
        cls_op,
        concrete_self,
        concrete_cls,
        objtype,
        w_descr,
        class_mode,
        binding,
    )?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value_op)?;
    Ok(Some(()))
}

/// Guard an already-built `super` proxy as the exact builtin implementation
/// and expose its three live lookup operands to the trace.
///
/// Both the Python-free result fold and the property-getter inline use this
/// prefix.  Descriptor classification happens before entry, so no caller can
/// decline after these guards merely because the selected descriptor needs a
/// different binding path.
///
/// `w_objtype` is guarded here and the two returned operands are not, because
/// the callers pin those through
/// [`walker_emit_super_attr_lookup_guards`], which the opcode spelling shares.
pub(crate) fn walker_guard_and_read_super_proxy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    proxy_w_class: pyre_object::PyObjectRef,
    concrete_self: pyre_object::PyObjectRef,
    concrete_cls: pyre_object::PyObjectRef,
    concrete_objtype: pyre_object::PyObjectRef,
) -> Result<(OpRef, OpRef), DispatchError> {
    let super_type_addr = &pyre_object::descriptor::SUPER_TYPE as *const _ as i64;
    walker_guard_stamped_class(ctx, op_pc, obj, super_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, obj, proxy_w_class)?;
    let cls_op = walker_read_super_field(
        ctx,
        obj,
        crate::descr::super_start_type_descr(),
        concrete_cls,
    );
    let self_op = walker_read_super_field(ctx, obj, crate::descr::super_obj_descr(), concrete_self);
    // `w_objtype` is the class the suffix walk answers with, and on a proxy it
    // is a stored word rather than something derived from the receiver, so it
    // is pinned where it is read.  This is the whole of the proof for the
    // receivers `_super_check` needs Python to settle: their `w_class` says
    // nothing about the class the proxy resolved.
    let objtype_op = walker_read_super_field(
        ctx,
        obj,
        crate::descr::super_obj_type_descr(),
        concrete_objtype,
    );
    // A virtual proxy answers this field from its own SetfieldGc cache as
    // the constant `emit_super_proxy` stored.  A second GUARD_VALUE of that
    // word is a tautology; `walker_ref_box_is` sees the forwarded constant
    // even when the opref itself is still the GETFIELD box.
    walker_guard_stamped_ref_unless_is(ctx, op_pc, objtype_op, concrete_objtype)?;
    Ok((self_op, cls_op))
}

/// One `W_Super` field read, with the recording-time value attached when the
/// read did not already carry one.
///
/// A virtual proxy answers out of its own `SetfieldGc` cache and the operand
/// comes back already concrete; a materialised one records a `GETFIELD_GC_R`
/// whose live load may be absent, and the walk cannot continue on a box with
/// no concrete half.
fn walker_read_super_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    proxy: OpRef,
    descr: majit_ir::DescrRef,
    concrete: pyre_object::PyObjectRef,
) -> OpRef {
    let op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, proxy, descr);
    if !matches!(
        ctx.trace_ctx.box_value(op),
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
    ) {
        ctx.trace_ctx
            .set_opref_concrete(op, majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)));
    }
    op
}

/// Two-argument `super(cls, obj)` reached as a call — the spelling a name
/// binding produces, and the one `LOAD_SUPER_ATTR` does not fuse away.
///
/// `try_walker_specialize_bare_super_call` is its zero-argument sibling and
/// re-routes rather than removes, because zero-argument `super()` reads the
/// frame and the frame read has to happen on a channel the walker can see.
/// Two arguments read nothing: `descriptor.py super_init_impl` validates the
/// pair and stores three words, so the whole call is an allocation and the
/// emission is that allocation spelled out.
///
/// Removing the CALL removes more than the call.  `bh_call_fn` is may-force,
/// so the walk publishes a vref for the executing frame ahead of it
/// (`ForceToken`, a `NewWithVtable(VRef)`, a store into
/// `ExecutionContext.topframeref`) and re-checks `GuardNotForced` /
/// `GuardNoException` after -- 9 ops around one that allocates 4 words.  With
/// the proxy emitted as `New` + `SetfieldGc` and its reads answered
/// ([`try_walker_specialize_load_attr_on_super`]), the optimizer drops the
/// allocation entirely for a proxy that never escapes.
///
/// Only the pair `_super_check` settles by walking installed MROs is folded:
/// its third arm asks for `__class__`, which a property answers with arbitrary
/// Python, and the walk executes its own emissions concretely, so a fold that
/// reached that arm would run user code at recording time and then repeat it
/// under the residual on a decline.  The first arm is the class-method case:
/// it stores the class itself as both `w_objtype` and `w_self`; the proxy
/// emission below pins that class object directly rather than reading its
/// metaclass out of `w_class`.
pub(crate) fn try_walker_specialize_two_arg_super_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // `super(cls, obj)` arrives as `[callable, null_or_self, cls, obj]` — the
    // same `bh_call_fn` operand list the zero-argument sibling reads, with the
    // two user arguments after the bound-receiver slot.
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        return Ok(None);
    }
    let Some((concrete_callable, [concrete_cls, concrete_obj])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 2)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_super_type(concrete_callable) {
        return Ok(None);
    }
    // `descriptor.py:28-30` — `None` builds the UNBOUND proxy, whose `w_self`
    // is null and whose attribute reads take a different arm entirely.
    if unsafe { pyre_object::is_none(concrete_obj) } {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_type(concrete_cls) } {
        return Ok(None);
    }
    let python_free =
        pyre_interpreter::builtins::super_check_python_free(concrete_cls, concrete_obj);
    let apparent = if python_free.is_none() {
        walker_apparent_super_class(concrete_cls, concrete_obj)
    } else {
        None
    };
    let Some(objtype) = python_free.or_else(|| apparent.map(|answer| answer.objtype)) else {
        return Ok(None);
    };
    let class_mode = python_free.is_some()
        && unsafe { pyre_object::is_type(concrete_obj) }
        && std::ptr::eq(objtype, concrete_obj);
    // In instance mode the receiver's class is read back out of the object
    // below, so the two must be the same word: an exception instance carrying
    // the generic stub resolves its class through the kind registry instead.
    // In class mode `_super_check` returns the receiver class itself and its
    // `w_class` is the metaclass, so identity of `obj_op` is the guard instead.
    if apparent.is_none()
        && !class_mode
        && !std::ptr::eq(objtype, unsafe { (*concrete_obj).w_class })
    {
        return Ok(None);
    }

    let cls_op = r_args[2];
    let obj_op = r_args[3];
    // Which callable `super` names is baked into the emitted body.
    walker_guard_stamped_ref(ctx, op.pc, r_args[0], concrete_callable)?;
    let proxy_op = if let Some(apparent) = apparent {
        walker_emit_apparent_super_proxy(
            ctx,
            op.pc,
            cls_op,
            obj_op,
            concrete_cls,
            concrete_obj,
            apparent,
        )?
    } else {
        walker_emit_super_proxy(
            ctx,
            op.pc,
            cls_op,
            obj_op,
            concrete_cls,
            concrete_obj,
            objtype,
            class_mode,
        )?
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', proxy_op)?;
    Ok(Some(()))
}

/// The proxy `descriptor.py super_init_impl` stores, emitted as a virtual.
///
/// Shared by the two spellings that reach it with a settled pair: the explicit
/// `super(cls, obj)` call, and the zero-argument one whose operands come out of
/// the callee's own frame slots
/// ([`try_walker_specialize_bare_super_virtual`]).  How each proves its two
/// operands is the caller's question; from here the guards and the emission are
/// the same.
///
/// Emitting starts here, so a caller that declines does so with the trace
/// untouched.
fn walker_emit_super_proxy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    cls_op: OpRef,
    obj_op: OpRef,
    concrete_cls: pyre_object::PyObjectRef,
    concrete_obj: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
    class_mode: bool,
) -> Result<OpRef, DispatchError> {
    let cls_const = walker_guard_stamped_ref_unless_is(ctx, op_pc, cls_op, concrete_cls)?;
    let objtype_const = ctx.trace_ctx.const_ref(objtype as i64);
    if class_mode {
        // descriptor.py `_super_check`'s first arm returns `w_obj_or_type`
        // itself.  Pin that class object: guarding its physical TYPE_TYPE
        // layout would admit every class and reading `w_class` would produce
        // the metaclass, neither of which protects the baked MRO root.
        walker_guard_stamped_ref_unless_const(ctx, op_pc, obj_op, objtype)?;
    } else {
        // guard_class(obj, ob_type): the physical layout the `w_class` read
        // below needs.
        let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
        walker_guard_stamped_class(ctx, op_pc, obj_op, phys_type)?;
        // `_super_check`'s answer is baked, so pin the receiver's exact Python
        // class.  An exception instance carrying the generic stub may resolve
        // its class through the kind registry instead and was declined above.
        // Cached: the following `load_attr_on_super` lookup re-reads this
        // slot, and an uncached GETFIELD made that a second per-iteration
        // GUARD_VALUE of the same word.
        walker_pin_instance_w_class(ctx, op_pc, obj_op, objtype)?;
    }
    // A `__bases__` reassignment anywhere in the selected class's ancestry
    // bumps this tag and can make `issubtype_w(objtype, cls)` stop holding.
    walker_pin_type_version_tag(ctx, op_pc, objtype_const)?;
    walker_emit_super_proxy_storage(
        ctx,
        cls_const,
        objtype_const,
        obj_op,
        concrete_cls,
        concrete_obj,
        objtype,
    )
}

/// The `_super_check` third-arm twin of [`walker_emit_super_proxy`].  The
/// ordinary emitter pins `obj.w_class == objtype`, which is the proof supplied
/// by the normal second arm and exactly the condition an apparent-class proxy
/// violates.  Here the traced `obj.__class__` read supplies its own mapdict
/// proof, then both paths share the literal `W_Super` allocation below.
fn walker_emit_apparent_super_proxy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    cls_op: OpRef,
    obj_op: OpRef,
    concrete_cls: pyre_object::PyObjectRef,
    concrete_obj: pyre_object::PyObjectRef,
    apparent: ApparentSuperClass,
) -> Result<OpRef, DispatchError> {
    let cls_const = walker_guard_stamped_ref_unless_is(ctx, op_pc, cls_op, concrete_cls)?;
    let objtype_const =
        walker_guard_apparent_super_class(ctx, op_pc, obj_op, concrete_obj, apparent)?;
    walker_emit_super_proxy_storage(
        ctx,
        cls_const,
        objtype_const,
        obj_op,
        concrete_cls,
        concrete_obj,
        apparent.objtype,
    )
}

/// The field-for-field `W_Super` allocation shared after `_super_check`'s
/// different proof arms have produced and protected `objtype`.
fn walker_emit_super_proxy_storage<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    cls_const: OpRef,
    objtype_const: OpRef,
    obj_op: OpRef,
    concrete_cls: pyre_object::PyObjectRef,
    concrete_obj: pyre_object::PyObjectRef,
    objtype: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_object::descriptor::SUPER_TYPE) as i64);
    let proxy_op = crate::helpers::emit_super_inline(
        ctx.trace_ctx,
        cls_const,
        objtype_const,
        obj_op,
        header_w_class,
    );
    let super_type_addr = &pyre_object::descriptor::SUPER_TYPE as *const _ as i64;
    ctx.trace_ctx.heap_cache_mut().class_now_known(proxy_op);
    // The concrete proxy the walker's own execution must observe.  Built last:
    // it allocates, and every address baked above is read before it runs.
    let proxy = pyre_object::descriptor::w_super_new(
        concrete_cls,
        objtype,
        concrete_obj,
        pyre_object::PY_NULL,
    );
    ctx.trace_ctx.set_opref_concrete(
        proxy_op,
        majit_ir::Value::Ref(majit_ir::GcRef(proxy as usize)),
    );
    Ok(proxy_op)
}

/// Fold `super_attr_unwrap(raw, which)` — the LOAD_SUPER_ATTR method form's
/// `[func, self_or_null]` split — once `raw` is concrete.  The interpreter
/// spells the same decision inline (`is_method` then `w_method_get_func` /
/// `w_method_get_self`, else `(raw, PY_NULL)`), so left as a residual it is a
/// second and third per-iteration call on top of the attribute one, and it
/// FORCES the `Method` [`try_walker_specialize_load_super_attr`] emits
/// instead of letting it virtualize away.
pub(crate) fn try_walker_fold_super_attr_unwrap<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
    which: i64,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(concrete_raw) = walker_concrete_ref_object(ctx, raw) else {
        return Ok(None);
    };
    // A POSITIVE class pin, which decides `is_method` in BOTH directions: a
    // `raw` that is a `Method` on one iteration and a plain function on the
    // next side-exits rather than re-running the baked arm.
    if !raw.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(raw) {
        // Under the single-frame collapse a guard here would resume at the
        // caller's CALL, re-running whatever that callee already did.
        if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
            return Ok(None);
        }
        let phys_type = unsafe { (*concrete_raw).ob_type } as i64;
        walker_guard_stamped_class(ctx, op_pc, raw, phys_type)?;
    }
    let value = if unsafe { pyre_object::is_method(concrete_raw) } {
        let (descr, concrete) = if which == 0 {
            (crate::descr::method_w_function_descr(), unsafe {
                pyre_object::w_method_get_func(concrete_raw)
            })
        } else {
            (crate::descr::method_w_self_descr(), unsafe {
                pyre_object::w_method_get_self(concrete_raw)
            })
        };
        // The CACHED read, not the uncached one.  The producing fold primes
        // both fields through `heapcache_setfield_cached`, so this resolves to
        // the value it stored and records no op at all.  An uncached
        // `GETFIELD_GC_R` hands the following CALL a callable slot with no
        // concrete ref, which declines an inline the residual this replaces
        // did not — measured as `[inline-decline] why=callable slot carries no
        // concrete ref` and a slower loop than leaving the residual alone.
        let op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, raw, descr);
        let resolved = matches!(
            ctx.trace_ctx.box_value(op),
            Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE
        );
        if !resolved {
            // A `Method` this fold did not build reaches the cache empty; the
            // read is still the one the helper performs, so give the walker
            // the value it would have returned.
            ctx.trace_ctx
                .set_opref_concrete(op, majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)));
        }
        op
    } else if which == 0 {
        raw
    } else {
        // `PY_NULL` is the correct `self` slot for a non-`Method` attribute;
        // it flows into the following CALL's checked `null_or_self` operand.
        ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64)
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// How the add-transition fold pins the stored value's type, resolved before
/// any guard is emitted so a decline falls cleanly to the residual.
enum StoreAttrAddValuePin {
    Boxed,
    /// Fresh unboxed int slot; `Some` pins a heap operand's canonical
    /// `w_class`, `None` means tagged (the unbox's tag guard is the pin).
    UnboxedInt(Option<pyre_object::PyObjectRef>),
}

/// `None` (unpinnable `w_class`, or a float pick) keeps the residual.
fn store_attr_add_value_pin(
    add: &pyre_interpreter::objspace::std::mapdict::StoreAttrAdd,
    concrete_value: pyre_object::PyObjectRef,
) -> Option<StoreAttrAddValuePin> {
    match add.unbox_type {
        None => Some(StoreAttrAddValuePin::Boxed),
        Some(pyre_interpreter::objspace::std::mapdict::UnboxType::Int) => {
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && unsafe { pyre_object::tagged_int::is_tagged_int(concrete_value) }
            {
                return Some(StoreAttrAddValuePin::UnboxedInt(None));
            }
            unsafe { walker_exact_builtin_class(concrete_value) }
                .map(|canonical| StoreAttrAddValuePin::UnboxedInt(Some(canonical)))
        }
        // `store_attr_add_fast_path` never resolves a float pick; defensive.
        Some(pyre_interpreter::objspace::std::mapdict::UnboxType::Float) => None,
    }
}

pub(crate) fn try_walker_specialize_store_attr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    value: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let (Some(concrete_obj), Some(concrete_value)) = (
        walker_concrete_ref_object(ctx, obj),
        walker_concrete_ref_object(ctx, value),
    ) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    if let Some((w_type, version_tag, map, storageindex, listindex, unbox_type, attr)) = unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_unboxed_fast_path(concrete_obj, &name)
    } {
        match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                // Match mapdict.py `_direct_write` exactly, through the same
                // predicate the interpreter's own store uses: `is_int` reads
                // `ob_type`, which an `int` subclass shares, so unboxing on it
                // would take the raw payload and lose `w_class`.
                if !unsafe {
                    pyre_interpreter::objspace::std::mapdict::is_unboxable_int(concrete_value)
                } {
                    return Ok(None);
                }
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                // Match mapdict.py `_direct_write` exactly: subclasses and
                // NaNs convert the slot to boxed storage.
                if !unsafe {
                    pyre_interpreter::objspace::std::mapdict::is_unboxable_float(concrete_value)
                } {
                    return Ok(None);
                }
            }
        }

        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            w_type,
            version_tag,
            map,
        )?;
        unsafe { pyre_interpreter::objspace::std::mapdict::mark_attr_ever_mutated(attr) };
        let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
        let listindex_const = ctx.trace_ctx.const_int(listindex as i64);
        // A same-type update writes ONE raw slot.  The map-transition fold
        // above needs `is_unescaped` because it publishes `map` and `storage`
        // as an unlocked pair a concurrent mutator could tear; this write
        // publishes neither, so it holds for an escaped receiver too.
        // Recording it as `setarrayitem_gc` rather than a residual also puts
        // it in the same heap cache the unboxed read uses, so a read of the
        // slot the trace just wrote answers from the trace.
        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
            crate::descr::mapdict_storage_descr(concrete_obj)
        });
        let slot =
            crate::state::trace_mapdict_storage_getitem(ctx.trace_ctx, block, storageindex_const);
        let raw_live = match unbox_type {
            pyre_interpreter::objspace::std::mapdict::UnboxType::Int => {
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                let raw = walker_unbox_int_exact(
                    ctx,
                    op_pc,
                    value,
                    int_type_addr,
                    crate::descr::int_intval_descr(),
                    walker_numeric_builtin_class(concrete_value),
                )?;
                crate::state::trace_int_block_setitem_value(
                    ctx.trace_ctx,
                    slot,
                    listindex_const,
                    raw,
                );
                unsafe { pyre_object::w_int_get_value(concrete_value) }
            }
            pyre_interpreter::objspace::std::mapdict::UnboxType::Float => {
                let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
                let raw = walker_unbox_float(ctx, op_pc, value, float_type_addr)?;
                // A subclass shares the builtin's `ob_type`, which is all the unbox
                // guard proves; the operand gate read `w_class`, so pin that too.
                walker_guard_exact_w_class(
                    ctx,
                    op_pc,
                    value,
                    walker_numeric_builtin_class(concrete_value),
                )?;
                let live_f = unsafe { pyre_object::w_float_get_value(concrete_value) };
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(live_f));
                walker_guard_float_not_nan(ctx, op_pc, raw)?;
                crate::state::trace_float_block_setitem_value(
                    ctx.trace_ctx,
                    slot,
                    listindex_const,
                    raw,
                );
                live_f.to_bits() as i64
            }
        };
        // The walk is the authoritative execution path, so apply the write
        // now (`_direct_write`'s same-type arm, mapdict.py); the ops above
        // reproduce it in compiled code.
        unsafe {
            pyre_interpreter::objspace::std::mapdict::write_unboxed_storage_raw(
                concrete_obj,
                storageindex,
                listindex,
                raw_live,
            )
        };
        return Ok(Some(()));
    }

    if let Some((slot, kind, w_type, version_tag, _stored)) = unsafe {
        pyre_interpreter::baseobjspace::exception_attr_slot_fold(concrete_obj, &name, true)
    } {
        if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
            let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
            let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
            if !unsafe {
                std::ptr::eq((*concrete_value).ob_type, tuple_type)
                    && std::ptr::eq((*concrete_value).w_class, canonical_tuple_class)
            } {
                return Ok(None);
            }
        }
        walker_guard_exception_attr_slot(ctx, op_pc, obj, concrete_obj, w_type, version_tag)?;
        let (stored_value, concrete_stored) =
            if slot == pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args {
                let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::PyType;
                let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
                if !unsafe {
                    std::ptr::eq((*concrete_value).ob_type, tuple_type)
                        && std::ptr::eq((*concrete_value).w_class, canonical_tuple_class)
                } {
                    return Ok(None);
                }
                let tuple_type_addr = tuple_type as i64;
                walker_guard_stamped_class(ctx, op_pc, value, tuple_type_addr)?;
                walker_guard_exact_w_class(ctx, op_pc, value, canonical_tuple_class)?;
                let block = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    value,
                    crate::descr::tuple_wrappeditems_descr(),
                );
                let len = unsafe { pyre_object::w_tuple_len(concrete_value) };
                let length = crate::state::opimpl_arraylen_gc(
                    ctx.trace_ctx,
                    block,
                    crate::state::pyobject_gcarray_descr(),
                );
                walker_guard_stamped_len(ctx, op_pc, length, len as i64)?;
                let mut items = Vec::with_capacity(len);
                let mut concrete_items = Vec::with_capacity(len);
                for index in 0..len {
                    let index_op = ctx.trace_ctx.const_int(index as i64);
                    items.push(crate::state::trace_items_block_getitem_value(
                        ctx.trace_ctx,
                        block,
                        index_op,
                    ));
                    concrete_items.push(
                        unsafe { pyre_object::w_tuple_getitem(concrete_value, index as i64) }
                            .unwrap_or(pyre_object::PY_NULL),
                    );
                }
                let list = crate::helpers::emit_rlist_inline(ctx.trace_ctx, &items);
                let concrete_list = pyre_object::interp_exceptions::rlist_new(concrete_items);
                ctx.trace_ctx.set_opref_concrete(
                    list,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_list as usize)),
                );
                (list, concrete_list)
            } else {
                (value, concrete_value)
            };
        let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete_obj) };
        let field_descr = crate::descr::w_exception_attr_slot_descr_for(kind, slot, user);
        let field_index = field_descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[obj, stored_value], field_descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(obj, field_index, stored_value);
        // The walk is the authoritative execution path.  Apply the same raw
        // slot writer now so interpreter execution after a side exit observes
        // the store; the writer supplies the host-side remembered-set barrier.
        // Compiled SetfieldGc reference stores receive CondCallGcWb from
        // majit-gc's rewrite pass, consumed by both dynasm and cranelift.
        unsafe {
            match slot {
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Args => {
                    pyre_object::interp_exceptions::w_exception_set_args(
                        concrete_obj,
                        concrete_stored,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Context
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Cause
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Traceback
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::Name
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::AttrObj
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeObject
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeStart
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEnd
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeReason
                | pyre_interpreter::baseobjspace::ExceptionAttrSlot::UnicodeEncoding => {
                    // `exception_attr_slot_fold` declines these for stores, so
                    // the store fold never reaches here.
                    unreachable!("load-only exception slots fold on load only")
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Code => {
                    pyre_object::interp_exceptions::w_exception_set_code(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Errno => {
                    pyre_object::interp_exceptions::w_exception_set_errno(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Strerror => {
                    pyre_object::interp_exceptions::w_exception_set_strerror(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename => {
                    pyre_object::interp_exceptions::w_exception_set_filename(
                        concrete_obj,
                        concrete_value,
                    )
                }
                pyre_interpreter::baseobjspace::ExceptionAttrSlot::Filename2 => {
                    pyre_object::interp_exceptions::w_exception_set_filename2(
                        concrete_obj,
                        concrete_value,
                    )
                }
            }
        }
        return Ok(Some(()));
    }

    // The attribute is not in the map yet: fold the `map -> PlainAttribute`
    // transition and the grow-by-one storage rewrite into trace ops instead of
    // leaving the generic `setattr` residual, which would force the receiver.
    //
    // Only for a receiver this trace allocated and has not let escape.  The
    // emitted transition is a pair of raw field stores, so unlike the
    // interpreter's it does not hold the striped `instance_lock`, and unlike
    // the single-slot in-place write it publishes two fields: a concurrent
    // mutator of the same instance could pair one thread's `map` with
    // another's `storage`.  `is_unescaped` is what rules that out — no other
    // thread has a reference yet.  It costs almost nothing, because the fold's
    // payoff is exactly the unescaped case: an escaped receiver is one the
    // optimizer cannot remove anyway.
    if ctx.trace_ctx.heap_cache().is_unescaped(obj)
        && let Some(add) = unsafe {
            pyre_interpreter::objspace::std::mapdict::store_attr_add_fast_path(
                concrete_obj,
                &name,
                concrete_value,
            )
        }
        && let Some(value_pin) = store_attr_add_value_pin(&add, concrete_value)
    {
        walker_guard_mapdict_instance_shape(
            ctx,
            op_pc,
            obj,
            concrete_obj,
            add.w_type,
            add.version_tag,
            add.map,
        )?;
        walker_pin_holder_typ(ctx, op_pc, add.holder)?;
        // This marker is not redundant with the instance-map GuardValue. The
        // fold bakes `add.new_map`, the transition target, as a green
        // `const_int`, while the guard pins `add.map`, the instance map before
        // the transition. `holder_pick_attr` can replace `holder.attr` with a
        // fresh `PlainAttribute` without changing any instance's current map;
        // this marker protects the baked target.
        walker_pin_holder_attr(ctx, op_pc, add.holder)?;
        // General rule: plant the `allow_unboxing` marker only where the fold
        // read the flag as true. Marking a false read re-arms a permanently
        // dead field and lets later writes invalidate without bound.
        if add.picked_unbox.is_some() {
            let terminator = unsafe { (*add.map).terminator() };
            let term = unsafe { (*terminator).as_terminator() as *const _ };
            walker_pin_terminator_allow_unboxing(ctx, op_pc, term)?;
        }
        let new_map_const = ctx.trace_ctx.const_int(add.new_map as i64);
        let map_descr = unsafe { crate::descr::mapdict_map_descr(concrete_obj) };
        let storage_descr = unsafe { crate::descr::mapdict_storage_descr(concrete_obj) };
        match value_pin {
            StoreAttrAddValuePin::Boxed => {
                crate::helpers::emit_mapdict_add_attr_inline(
                    ctx.trace_ctx,
                    obj,
                    add.storageindex,
                    new_map_const,
                    value,
                    map_descr,
                    storage_descr,
                );
            }
            StoreAttrAddValuePin::UnboxedInt(canonical) => {
                // Only an exactly-`int` runtime value may unbox: the tag
                // guard pins a tagged operand, the `w_class` pin a heap one
                // (a subclass shares `W_IntObject`'s `ob_type`).
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                let raw = if let Some(canonical) = canonical {
                    walker_unbox_int_exact(
                        ctx,
                        op_pc,
                        value,
                        int_type_addr,
                        crate::descr::int_intval_descr(),
                        canonical,
                    )?
                } else {
                    walker_unbox_int(ctx, op_pc, value, int_type_addr)?
                };
                crate::helpers::emit_mapdict_add_unboxed_attr_inline(
                    ctx.trace_ctx,
                    obj,
                    add.storageindex,
                    new_map_const,
                    raw,
                    map_descr,
                    storage_descr,
                );
            }
        }
        // The walk is the authoritative execution path, so apply the resolved
        // transition now; the emitted operations reproduce it in compiled code.
        unsafe {
            pyre_interpreter::objspace::std::mapdict::store_attr_add_commit(
                concrete_obj,
                &add,
                concrete_value,
            )
        };
        return Ok(Some(()));
    }

    let Some((w_type, version_tag, map, storageindex, attr)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::store_attr_boxed_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    walker_guard_mapdict_instance_shape(ctx, op_pc, obj, concrete_obj, w_type, version_tag, map)?;
    unsafe { pyre_interpreter::objspace::std::mapdict::mark_attr_ever_mutated(attr) };
    let storageindex_const = ctx.trace_ctx.const_int(storageindex as i64);
    // mapdict.py `PlainAttribute._direct_write` →
    // `obj._mapdict_write_storage(self.storageindex, w_value)`: one
    // `setarrayitem_gc` into the storage list the guarded map sizes.  Like the
    // unboxed same-type update above, it publishes no `map`/`storage` pair, so
    // it holds for an escaped receiver too, and the reference store gets its
    // `cond_call_gc_wb` from the GC rewrite pass.
    let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, obj, unsafe {
        crate::descr::mapdict_storage_descr(concrete_obj)
    });
    crate::state::trace_mapdict_storage_setitem(ctx.trace_ctx, block, storageindex_const, value);
    // The walk is the authoritative execution path, so apply the write now;
    // the ops above reproduce it in compiled code.
    unsafe {
        pyre_interpreter::objspace::std::mapdict::write_boxed_storage(
            concrete_obj,
            storageindex,
            concrete_value,
        )
    };
    Ok(Some(()))
}

/// #171: FBW virtualization of a non-escaping BUILD_LIST.
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_LIST to `new_array_clear`
/// + per-index `setarrayitem_gc` + a `newlist_from_array` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::NewlistFromArray`]) whose single
/// r-arg is the already-built backing array.  Decompose that residual into
/// the virtualizable `opimpl_newlist` shape (`pyjitpl.py`) —
/// `new_with_vtable` + `new_array` + `setarrayitem_gc` + `setfield_gc` —
/// so the optimizer folds the whole list (wrapper + block) when it never
/// escapes and the array build + residual DCE.
///
/// The element boxes are recovered from the backing array (its const length
/// from `heapcache.arraylen`, then per-index element shadows via
/// `heapcache_getarrayitem`), NOT from residual args.  The storage strategy
/// is chosen from the concrete element shadows exactly as
/// `list_strategy_for` / `w_list_new` does at runtime, so the traced object
/// matches the strategy the blackhole rebuilds on deopt:
///   * `list_strategy_for` → Integer AND every element an exact
///     `W_IntObject` → Integer (`int_items` typed block, elements unboxed
///     via `walker_unbox_int`);
///   * → Float → Float (`float_items` typed block, strict `W_FloatObject`
///     elements only, so exact-type by construction);
///   * → Object → Object (boxed refs into an `ItemsBlock`).
///
/// Returns `Ok(None)` to fall through to the opaque residual (always
/// byte-correct) for any shape it cannot reproduce faithfully: empty list
/// (Empty strategy), a non-const / unrecoverable array length, an element
/// without a concrete Ref shadow, or an Integer-strategy list that carries a
/// tagged immediate (which has no `&INT_TYPE`/`&LONG_TYPE` header for the
/// unbox guard). A fits-in-word `W_LongObject` is accepted: `is_plain_int1`
/// covers it and `walker_unbox_long` supplies the `&LONG_TYPE` + `_fits_int`
/// guarded extraction.
pub(crate) fn try_walker_specialize_newlist<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];

    // Const backing-array length (`new_array_clear(Const(len))` seeded
    // `heapcache.arraylen` — a cleared array has every slot set, so read the
    // length directly rather than probing getarrayitem until a miss).  Empty
    // list → Empty strategy: decline (the residual reproduces it).
    let len = {
        let Some(len_op) = ctx.trace_ctx.heap_cache().arraylen(arr) else {
            return Ok(None);
        };
        match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 1 => n as usize,
            _ => return Ok(None),
        }
    };

    // Recover the element boxes from the array heap-cache (the values the
    // BUILD_LIST `setarrayitem_gc` ops stored); a cache miss (clobbered array)
    // bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let mut items: Vec<OpRef> = Vec::with_capacity(len);
    for i in 0..len {
        let Some(elem) =
            ctx.trace_ctx
                .heapcache_getarrayitem(arr, OpRef::ConstInt(i as i64), descr_idx)
        else {
            return Ok(None);
        };
        items.push(elem);
    }

    // Concrete element objects (needed to classify the strategy and extract
    // the payloads before any allocation).  An element without a concrete Ref
    // shadow declines to the residual.
    let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for &it in &items {
        let Some(obj) = walker_concrete_ref_object(ctx, it) else {
            return Ok(None);
        };
        concretes.push(obj);
    }

    // Strategy the runtime `w_list_new` would pick — the source of truth for
    // the concrete shadow, so the traced storage matches on deopt.
    let strategy = pyre_object::listobject::list_strategy_for(&concretes);
    use pyre_object::listobject::ListStrategy;

    // Pre-extract the machine payloads BEFORE `build_list_from_refs` allocates
    // (a minor collection there could move the boxed elements, so the raw
    // pointers must not be dereferenced afterwards).
    enum Emit {
        // Per element: `(unboxed i64, is_fits_long)`.  `is_fits_long` selects
        // `walker_unbox_long` (`&LONG_TYPE` + `_fits_int` guard) over the
        // plain `walker_unbox_int`.
        Int(Vec<(i64, bool)>),
        Float(Vec<f64>),
        Object,
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let emit = match strategy {
        ListStrategy::Integer => {
            // `IntegerListStrategy.is_correct_type` is `is_plain_int1`, which
            // accepts an exact `W_IntObject` or a fits-in-word `W_LongObject`;
            // both store the unboxed i64 (`plain_int_w`). A tagged immediate
            // has no header for the unbox guard, so decline it to the residual
            // (correct for any element).
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                if pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(p)
                {
                    return Ok(None);
                }
                if !unsafe { pyre_object::is_plain_int1(p) } {
                    return Ok(None);
                }
                let is_fits_long = unsafe { pyre_object::pyobject::is_long(p) };
                let val = if is_fits_long {
                    pyre_object::longobject::jit_w_long_toint(p)
                } else {
                    unsafe { pyre_object::w_int_get_value(p) }
                };
                vals.push((val, is_fits_long));
            }
            Emit::Int(vals)
        }
        ListStrategy::Float => {
            // `list_strategy_for` admits only exact, non-NaN floats here.  Its
            // subclass term is enforced on replay by pinning each element's
            // `w_class`; `is_plain_float_strict` also admits the null spelling
            // of "exact float", which no pin can express, so decline such an
            // element rather than emit a guard it would fail itself.
            let mut vals = Vec::with_capacity(len);
            for &p in &concretes {
                if unsafe { walker_exact_builtin_class(p) }.is_none() {
                    return Ok(None);
                }
                vals.push(unsafe { pyre_object::w_float_get_value(p) });
            }
            Emit::Float(vals)
        }
        ListStrategy::Object => Emit::Object,
        // The interpreter stores this as encoded signed-longlong values.
        // The walker does not yet have an encoded numeric payload variant;
        // leave construction to the ordinary residual instead of emitting an
        // Integer array whose values would have the wrong representation.
        ListStrategy::IntOrFloat => return Ok(None),
        // The generic residual constructs the erased rpython-string array.
        // The walker has no BytesBlock payload emitter yet.
        ListStrategy::Bytes => return Ok(None),
        // The generic residual constructs AsciiListStrategy's erased UTF-8
        // storage; the walker has no raw UnicodeValueStorage emitter yet.
        ListStrategy::Ascii => return Ok(None),
        // Empty is impossible here (len >= 1); decline defensively. Range
        // storage is built only by the interpreter-internal `make_range_list`
        // seam and has no walker-native erased-tuple emitter yet.
        ListStrategy::Empty
        | ListStrategy::Size
        | ListStrategy::SimpleRange
        | ListStrategy::Range => return Ok(None),
    };

    // Concrete shadow: a fresh list built from the element shadows
    // (`w_list_new` parity — picks the same strategy). A new allocation with
    // no heap mutation, safe during the walk like `wrapint`.
    let result_concrete = pyre_interpreter::build_list_from_refs(&concretes);
    if result_concrete.is_null() {
        return Ok(None);
    }

    // emit the virtualizable decomposed newlist (walker-native)
    let list_op = match emit {
        Emit::Int(vals) => {
            let int_type_addr = int_ty as i64;
            let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &(v, is_fits_long)) in items.iter().zip(vals.iter()) {
                let exact_class = walker_concrete_ref_object(ctx, it)
                    .map(walker_numeric_builtin_class)
                    .unwrap_or(pyre_object::PY_NULL);
                let raw = if is_fits_long {
                    walker_guard_exact_w_class(ctx, op_pc, it, exact_class)?;
                    walker_unbox_long(ctx, op_pc, it, long_type_addr)?
                } else {
                    walker_unbox_int_exact(
                        ctx,
                        op_pc,
                        it,
                        int_type_addr,
                        crate::descr::int_intval_descr(),
                        exact_class,
                    )?
                };
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Int(v));
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                &mut *ctx.trace_ctx,
                &raws,
                crate::state::int_gcarray_descr(),
                crate::descr::list_int_items_len_descr(),
                crate::descr::list_int_items_block_descr(),
                ListStrategy::Integer,
            )
        }
        Emit::Float(vals) => {
            let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
            let mut raws: Vec<OpRef> = Vec::with_capacity(len);
            for (&it, &v) in items.iter().zip(vals.iter()) {
                if let Some(obj) = walker_concrete_ref_object(ctx, it) {
                    walker_guard_exact_w_class(ctx, op_pc, it, walker_numeric_builtin_class(obj))?;
                }
                let raw = walker_unbox_float(ctx, op_pc, it, float_type_addr)?;
                ctx.trace_ctx
                    .set_opref_concrete(raw, majit_ir::Value::Float(v));
                // `walker_unbox_float` guards `ob_type` only, which a float
                // SUBCLASS instance shares; pin `w_class` so it side-exits
                // instead of being unboxed into Float storage the interpreter
                // would have declined (`all_floats` is strict).
                walker_guard_exact_w_class(
                    ctx,
                    op_pc,
                    it,
                    pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
                )?;
                walker_guard_float_not_nan(ctx, op_pc, raw)?;
                raws.push(raw);
            }
            crate::helpers::emit_typed_list_inline(
                &mut *ctx.trace_ctx,
                &raws,
                crate::state::float_gcarray_descr(),
                crate::descr::list_float_items_len_descr(),
                crate::descr::list_float_items_block_descr(),
                ListStrategy::Float,
            )
        }
        Emit::Object => crate::helpers::emit_object_list_inline(&mut *ctx.trace_ctx, &items),
    };

    ctx.trace_ctx.set_opref_concrete(
        list_op,
        majit_ir::Value::Ref(majit_ir::GcRef(result_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, list_op)?;
    Ok(Some(()))
}

/// FBW virtualization of the array-backed BUILD_TUPLE — the arities
/// `makespecialisedtuple2` does not claim.  Sibling of
/// [`try_walker_specialize_newtuple`] (arity-2 plain-int `spec_ii`) and
/// [`try_walker_specialize_newlist`], reached only after the `spec_ii` fold
/// declines, so that path stays byte-identical.
///
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_TUPLE to `new_array_clear` +
/// per-index `setarrayitem_gc` + a `newtuple_from_array` residual.  Re-emit the
/// canonical `W_TupleObject` shape walker-native (`new_with_vtable` +
/// `w_class` / `wrappeditems` `setfield_gc` over a fresh items block), reading
/// the elements straight out of the array heap-cache so the array build keeps
/// no consumer and DCEs.  A tuple that never escapes then folds away entirely,
/// and one that does escape materializes from the same fields the residual
/// would have written.
///
/// Arity 2 is `makespecialisedtuple2` territory (`Cls_ii` / `Cls_ff` /
/// `Cls_oo`, `specialisedtupleobject.py`): the runtime never builds an
/// array-backed tuple there, so emitting one would diverge from what the
/// blackhole rebuilds on deopt.  Declined here — the `spec_ii` fold owns the
/// int-int case and the residual owns the rest.  The empty tuple is declined
/// too (no element to recover a length from).
///
/// Lifting that decline is not a trace-local question: the trace stays
/// self-consistent, but a side exit hands a real pair — inline `value0` /
/// `value1`, no `wrappeditems` block — to whatever consumer the trace picked
/// for the canonical layout, and
/// [`try_walker_orthodox_subscr_tuple_item`] then reads a field that is
/// not there.
///
/// Returns `Ok(Some(()))` when folded; `Ok(None)` falls through to the opaque
/// residual, which stays correct for any shape — a non-const array length or
/// an element without a concrete Ref shadow is not declined, just not folded.
pub(crate) fn try_walker_specialize_newtuple_object<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];
    // Const backing-array length (`new_array_clear(Const(len))` seeded
    // `heapcache.arraylen`; a cleared array has every slot set, so read the
    // length directly rather than probing getarrayitem until a miss).
    let len = {
        let Some(len_op) = ctx.trace_ctx.heap_cache().arraylen(arr) else {
            return Ok(None);
        };
        match len_op.inline_const_to_value() {
            Some(majit_ir::Value::Int(n)) if n >= 1 => n as usize,
            _ => return Ok(None),
        }
    };
    if len == 2 {
        return Ok(None);
    }

    // Element boxes the BUILD_TUPLE `setarrayitem_gc` ops stored; a cache miss
    // (clobbered array / non-const index) bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let mut items: Vec<OpRef> = Vec::with_capacity(len);
    for i in 0..len {
        let Some(elem) =
            ctx.trace_ctx
                .heapcache_getarrayitem(arr, OpRef::ConstInt(i as i64), descr_idx)
        else {
            return Ok(None);
        };
        items.push(elem);
    }
    let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for &it in &items {
        let Some(obj) = walker_concrete_ref_object(ctx, it) else {
            return Ok(None);
        };
        concretes.push(obj);
    }

    // Concrete shadow: a fresh array-backed tuple from the element shadows
    // (`w_tuple_new` parity for every arity but 2). A new allocation with no
    // heap mutation, safe during the walk like `wrapint`.  Built before the
    // emit so a failure leaves no orphan ops in the trace.
    let result_concrete = pyre_object::w_tuple_new_array_backed(concretes);
    if result_concrete.is_null() {
        return Ok(None);
    }

    let tuple_op = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items);
    ctx.trace_ctx.set_opref_concrete(
        tuple_op,
        majit_ir::Value::Ref(majit_ir::GcRef(result_concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple_op)?;
    Ok(Some(()))
}

const SPECIALISED_TUPLE_II_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::specialisedtupleobject::w_specialised_tuple_ii_new",
    commit_label: "specialised_tuple_ii_commit",
    call_site_label: "specialised_tuple_ii_call_site",
    decline_tag: "TUPLE-II-SUBWALK",
};

/// #195 / #73: FBW virtualization of an arity-2 plain-int BUILD_TUPLE.
/// `lower_tuple_build_hlop_to_insn` lowers BUILD_TUPLE to `new_array_clear`
/// + per-index `setarrayitem_gc` + a `newtuple_from_array` residual
/// (oopspec [`majit_ir::RuntimeHelperKind::NewtupleFromArray`]).  When both
/// backing-array elements are concrete plain ints, descend
/// [`SPECIALISED_TUPLE_II_DESCENT`].  `fuse_boxing_alloc` records that
/// constructor as `new_with_vtable` plus the `value0` / `value1` stores.
/// The elements come straight out of the array heap-cache, so the array
/// build keeps no consumer and DCEs.  The partner
/// [`try_walker_specialize_unpack`] then folds those stores, collapsing
/// build→unpack to a pure-int loop.
///
/// Returns `Ok(Some(()))` when folded (the caller returns `Continue`);
/// `Ok(None)` to fall through to the opaque residual, which stays correct
/// for any other shape (object tuple, arity ≠ 2, out-of-range long, tagged
/// immediate, cache miss) — so a non-foldable build is not declined.
pub(crate) fn try_walker_specialize_newtuple<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }
    let arr = r_args[0];
    // Read the two backing-array element boxes out of the heap-cache (the
    // values the BUILD_TUPLE `setarrayitem_gc` ops stored); a cache miss
    // (non-const index / clobbered array) bails to the opaque residual.
    let descr_idx = crate::state::pyobject_gcarray_descr().index();
    let (Some(e0), Some(e1)) = (
        ctx.trace_ctx
            .heapcache_getarrayitem(arr, OpRef::ConstInt(0), descr_idx),
        ctx.trace_ctx
            .heapcache_getarrayitem(arr, OpRef::ConstInt(1), descr_idx),
    ) else {
        return Ok(None);
    };
    // Arity must be exactly 2 (the only specialised int tuple).  A BUILD_TUPLE
    // array sets every index before `newtuple_from_array`, so a cached element
    // at index 2 means arity ≥ 3 → fall through to the residual (a wrongly
    // built arity-2 spec_ii would length-mismatch the arity-N unpack).
    if ctx
        .trace_ctx
        .heapcache_getarrayitem(arr, OpRef::ConstInt(2), descr_idx)
        .is_some()
    {
        return Ok(None);
    }
    let (Some(c0), Some(c1)) = (
        walker_concrete_ref_object(ctx, e0),
        walker_concrete_ref_object(ctx, e1),
    ) else {
        return Ok(None);
    };
    // The arity-2 int specialised tuple `Cls_ii` (`makespecialisedtuple2`,
    // specialisedtupleobject.py) is built when both elements pass
    // `is_plain_int1` — an exact `W_IntObject` or a fits-in-word
    // `W_LongObject`; the stored payload is `plain_int_w` of each.  A tagged
    // immediate has no real header for the unbox guard, so decline it to the
    // residual (correct for any shape).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(c0)
            || pyre_object::tagged_int::is_tagged_int(c1))
    {
        return Ok(None);
    }
    let int_ty = &pyre_object::pyobject::INT_TYPE as *const pyre_object::pyobject::PyType;
    let both_plain_int =
        unsafe { pyre_object::is_plain_int1(c0) && pyre_object::is_plain_int1(c1) };
    if !both_plain_int {
        return Ok(None);
    }
    let c0_long = unsafe { pyre_object::pyobject::is_long(c0) };
    let c1_long = unsafe { pyre_object::pyobject::is_long(c1) };
    // Concrete element int payloads (`plain_int_w`: `W_IntObject`'s `intval`
    // or a fits-int `W_LongObject`'s `toint()`).
    let v0 = if c0_long {
        pyre_object::longobject::jit_w_long_toint(c0)
    } else {
        unsafe { pyre_object::w_int_get_value(c0) }
    };
    let v1 = if c1_long {
        pyre_object::longobject::jit_w_long_toint(c1)
    } else {
        unsafe { pyre_object::w_int_get_value(c1) }
    };

    let Some(jc) = crate::jitcode_runtime::pathed_jitcode_cached(SPECIALISED_TUPLE_II_DESCENT.path)
    else {
        return Ok(None);
    };
    if jc.calldescr.arg_classes != "ii" || jc.calldescr.result_type != 'r' {
        return Ok(None);
    }

    // Paired `w_class` guard per element so a runtime int subclass sharing
    // the public `int` `w_class` side-exits, then the plain-int payload unbox.
    // A fits-int `W_LongObject` also carries the public `int` `w_class`
    // (`is_plain_int1`), so the same guard covers it; the payload extraction
    // switches to `walker_unbox_long` (`&LONG_TYPE` + `_fits_int`).
    let pre_body = ctx.trace_ctx.get_trace_position();
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    walker_guard_exact_w_class(ctx, op_pc, e0, int_typeobj)?;
    walker_guard_exact_w_class(ctx, op_pc, e1, int_typeobj)?;
    let int_type_addr = int_ty as i64;
    let long_type_addr = &pyre_object::pyobject::LONG_TYPE as *const _ as i64;
    let raw0 = if c0_long {
        walker_unbox_long(ctx, op_pc, e0, long_type_addr)?
    } else {
        walker_unbox_int_typed(
            ctx,
            op_pc,
            e0,
            int_type_addr,
            crate::descr::int_intval_descr(),
        )?
    };
    let raw1 = if c1_long {
        walker_unbox_long(ctx, op_pc, e1, long_type_addr)?
    } else {
        walker_unbox_int_typed(
            ctx,
            op_pc,
            e1,
            int_type_addr,
            crate::descr::int_intval_descr(),
        )?
    };
    ctx.trace_ctx
        .set_opref_concrete(raw0, majit_ir::Value::Int(v0));
    ctx.trace_ctx
        .set_opref_concrete(raw1, majit_ir::Value::Int(v1));
    let mut produced = None;
    let outcome = try_walker_orthodox_descent_ex(
        ctx,
        op_pc,
        &[(raw0, v0), (raw1, v1)],
        &[],
        &[],
        dst,
        dst_bank,
        &SPECIALISED_TUPLE_II_DESCENT,
        Some(&mut produced),
        false, // payloads, not a BinaryOperator tag
    )?;
    if !matches!(outcome, Some(DispatchOutcome::Continue)) {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_body);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    // `setfield_gc` cached `value0` / `value1`. Stamp the pair when the walk
    // published no concrete, so a star-call unpack still sees this allocation.
    let Some(tuple) = produced else {
        return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc });
    };
    if !matches!(
        walker_concrete_ref_object(ctx, tuple),
        Some(obj) if !obj.is_null()
    ) {
        let tuple_ptr = pyre_object::specialisedtupleobject::w_specialised_tuple_ii_new(v0, v1);
        if tuple_ptr.is_null() {
            return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op_pc });
        }
        ctx.trace_ctx.set_opref_concrete(
            tuple,
            majit_ir::Value::Ref(majit_ir::GcRef(tuple_ptr as usize)),
        );
    }
    Ok(Some(()))
}

/// Walker-native fold of the CHECK_EXC_MATCH
/// residual (`compare_value_from_tag(exc, match_type, op_tag=10)`,
/// `call_jit.rs`). Computes the match concretely from
/// `type(exc)` and `match_type` and emit a `const_ref` of the immortal
/// TRUE/FALSE bool singleton, eliding the opaque may-force compare (and, since
/// that singleton is a constant, the immediately-following `is_true`
/// truth-extract residual).  With the exception's constructor + raise
/// already virtualized by their own folds, folding the match to a constant
/// lets the whole exception de-escape and DCE.
///
/// Soundness — the fold result depends only on `(type(exc), match_type)`:
///   * `exc` (`r_args[0]`) is the in-trace inline-built virtual exception
///     whose kind/vtable are baked into the `NewWithVtable`, so its class
///     cannot differ at runtime — no guard needed.  (A `GuardClass` is
///     emitted defensively when the heapcache does not already know its
///     class, e.g. a non-construct-fold exc reaching here.)
///   * `match_type` (`r_args[1]`) is a runtime value (typically a
///     `LOAD_GLOBAL` of the handler class), so a `GuardValue` pins its
///     identity — a reassigned handler global side-exits and re-traces
///     instead of running the wrong handler.  (Stricter than the trait,
///     which elides this guard.)
///
/// Declines (`None` → generic residual) when either operand lacks a
/// concrete shadow, or `match_type` is not a valid exception class /
/// tuple (the residual then raises the correct `TypeError`).
/// Pin the elements of a tuple `except` clause's match target, returning
/// whether the target was a tuple layout this could read through.
///
/// `w_tuple_new` picks the layout by arity: two object elements become a
/// `W_SpecialisedTupleObject_oo` holding them in inline immutable `value0` /
/// `value1` slots, everything else an array-backed `W_TupleObject` behind
/// `wrappeditems`. Both are read here with the same guarded-load shape the
/// other tuple folds use, and each element is pinned to the class seen while
/// tracing. The `_oo` loads are pure (immutable fields), so a tuple built from
/// constants leaves nothing behind after optimization.
///
/// A match target that is neither layout — a bare class, the int/float
/// specialisations, or a tuple subclass whose `w_class` diverges — returns
/// `false` so the caller pins the object identity instead. That is correct for
/// a target that is loaded rather than built, which is the only case where the
/// identity can hold across iterations.
fn walker_guard_exc_match_tuple_items<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    match_op: OpRef,
    match_type: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    let ob_type = unsafe { (*(match_type as *const pyre_object::pyobject::PyObject)).ob_type };
    let spec_oo = &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE
        as *const pyre_object::pyobject::PyType;
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::pyobject::PyType;

    // Either layout may carry a subclass `w_class`, and the element reads below
    // are only the whole target when it is a plain tuple.
    let canonical_tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    if !std::ptr::eq(
        unsafe { (*(match_type as *const pyre_object::pyobject::PyObject)).w_class },
        canonical_tuple_class,
    ) {
        return Ok(false);
    }

    let mut items: Vec<(OpRef, pyre_object::PyObjectRef)> = Vec::new();
    if std::ptr::eq(ob_type, spec_oo) {
        walker_guard_stamped_class(ctx, op_pc, match_op, spec_oo as i64)?;
        for index in 0..2usize {
            let descr = if index == 0 {
                crate::descr::specialised_tuple_oo_value0_descr()
            } else {
                crate::descr::specialised_tuple_oo_value1_descr()
            };
            let item = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, match_op, descr);
            let concrete = unsafe {
                pyre_object::specialisedtupleobject::w_specialised_tuple_oo_getvalue(
                    match_type, index,
                )
            };
            items.push((item, concrete));
        }
    } else if std::ptr::eq(ob_type, tuple_type) {
        // Read every element before recording anything: a bail-out after a
        // guard has been emitted would leave the target's class pinned, and the
        // caller reads that as "already guarded" and drops its own pin.
        let len = unsafe { pyre_object::w_tuple_len(match_type) };
        let mut concretes: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
        for index in 0..len {
            let Some(concrete) =
                (unsafe { pyre_object::w_tuple_getitem(match_type, index as i64) })
            else {
                return Ok(false);
            };
            concretes.push(concrete);
        }
        walker_guard_stamped_class(ctx, op_pc, match_op, tuple_type as i64)?;
        walker_guard_exact_w_class(ctx, op_pc, match_op, canonical_tuple_class)?;
        let block = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            match_op,
            crate::descr::tuple_wrappeditems_descr(),
        );
        let length = crate::state::opimpl_arraylen_gc(
            ctx.trace_ctx,
            block,
            crate::state::pyobject_gcarray_descr(),
        );
        walker_guard_stamped_len(ctx, op_pc, length, len as i64)?;
        for (index, concrete) in concretes.into_iter().enumerate() {
            let index_op = ctx.trace_ctx.const_int(index as i64);
            let item =
                crate::state::trace_items_block_getitem_value(ctx.trace_ctx, block, index_op);
            items.push((item, concrete));
        }
    } else {
        return Ok(false);
    }

    for (item, concrete) in items {
        let expected = walker_guard_stamped_ref_unless_const(ctx, op_pc, item, concrete)?;
        if unsafe { pyre_object::is_type(concrete) } {
            walker_pin_type_version_tag(ctx, op_pc, expected)?;
        }
    }
    Ok(true)
}

pub(crate) fn try_walker_fold_check_exc_match<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let exc_op = r_args[0];
    let match_op = r_args[1];
    let (Some(exc), Some(match_type)) = (
        walker_concrete_ref_object(ctx, exc_op),
        walker_concrete_ref_object(ctx, match_op),
    ) else {
        return Ok(None);
    };
    // `validate_check_exc_match_class` gates `except <non-exception>:`
    // (raising `TypeError`); on a validity error decline so the residual
    // reproduces the raise instead of baking a wrong bool into the trace.
    if pyre_interpreter::eval::validate_check_exc_match_class(match_type).is_err() {
        return Ok(None);
    }
    // The answer depends on `space.type(exc)`, and `typedef::type` reaches an
    // exception's class through the kind registry whenever the `w_class` slot
    // still holds the generic stub. The guard below can only pin a class the
    // slot itself holds, so decline the registry answer rather than emit a
    // guard that pins something else.
    let Some(exc_class) = pyre_interpreter::typedef::r#type(exc) else {
        return Ok(None);
    };
    let exc_class = exc_class.as_ptr();
    // `typedef::type` falls back to the kind registry when `w_class` is
    // still the generic BaseException stub (`w_exception_new` internals).
    // The match walked that registry class; pin the kind field so a
    // different ExcKind cannot reuse this fold. A GuardValue on the
    // registry class would pin a pointer the slot does not hold and fail
    // at runtime. Pin `w_class` as well: `_getusercls` shares one `ob_type`
    // across `_new_exception` classes, so `class MyError(ValueError)` and
    // `ValueError` share the layout and `ExcKind` but not `w_class`.
    let pin_kind_instead_of_w_class = !std::ptr::eq(unsafe { (*exc).w_class }, exc_class);
    // `eval::check_exc_match_against` = `exception_match(type(exc), match)`
    // (eval.rs), walking the exception class MRO and accepting a tuple of
    // classes. Inlined here.
    let matched = pyre_interpreter::eval::check_exc_match_against(exc, match_type);

    // commit to the fold: emit IR (no further declines)
    // Pin `match_type` so a runtime divergence (a reassigned handler global)
    // side-exits rather than running the wrong handler.
    //
    // A tuple clause is pinned through its ELEMENTS. `except (A, B):` lowers to
    // `BUILD_TUPLE`, which allocates a fresh tuple on every visit, so an
    // identity guard on the container is unsatisfiable — it fails once per
    // visit and the loop collapses into side exits and bridge churn. The
    // elements are what the match actually reads and what a rebinding would
    // change, so guarding them is both sound and stable across the
    // re-allocation.
    //
    // A known class does not stand in for the pin: every class object shares
    // the one `type` layout, so `is_class_known` on the clause operand says
    // only that it is a class and leaves which class free to change.
    if !match_op.is_constant()
        && !walker_guard_exc_match_tuple_items(ctx, op_pc, match_op, match_type)?
    {
        walker_guard_fold_callable(ctx, op_pc, match_op, match_type)?;
        if unsafe { pyre_object::is_type(match_type) } {
            let expected = ctx.trace_ctx.const_ref(match_type as i64);
            walker_pin_type_version_tag(ctx, op_pc, expected)?;
        }
    }
    // Pin the exception's Python-level class, the value the match walked the
    // MRO of. `GuardClass` alone cannot do it: `_getusercls` shares one
    // `ob_type` across every `_new_exception` class of a realbase, so
    // `ValueError`, `KeyError` and `class E(Exception)` share a layout and
    // one's recorded answer would replay for the other. The `w_class` pin
    // is what splits them. The layout guard still comes first — it is what
    // makes the `w_class` read below name the field it was recorded against.
    if !exc_op.is_constant() {
        let exc_layout =
            unsafe { (*(exc as *const pyre_object::pyobject::PyObject)).ob_type } as i64;
        walker_guard_stamped_class(ctx, op_pc, exc_op, exc_layout)?;
        if pin_kind_instead_of_w_class {
            let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) };
            let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc) };
            let (_, kind_descr, _, _) = crate::descr::w_exception_descrs_for(kind, user);
            let kind_op = walker_record_getfield_gc_i_uncached(ctx, exc_op, kind_descr);
            walker_guard_stamped_int(ctx, op_pc, kind_op, kind as u8 as i64)?;
            walker_guard_exact_w_class(ctx, op_pc, exc_op, unsafe { (*exc).w_class })?;
        } else {
            let w_class_op =
                walker_record_getfield_gc_r_uncached(ctx, exc_op, crate::descr::w_class_descr());
            walker_guard_stamped_ref(ctx, op_pc, w_class_op, exc_class)?;
        }
    }
    // `A.__bases__ = (B,)` changes `exception_match` without touching
    // the instance's `w_class`.  The isinstance fold pins the same
    // quasi-immutable tag (`walker_pin_type_version_tag`).
    let exc_class_const = ctx.trace_ctx.const_ref(exc_class as i64);
    walker_pin_type_version_tag(ctx, op_pc, exc_class_const)?;

    // The match is a constant at trace time: emit the immortal bool singleton
    // as a `const_ref`.  The following `is_true` (the `except` clause's
    // `POP_JUMP_IF_FALSE`) reads that constant W_Bool.
    let result_obj = pyre_object::w_bool_from(matched);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, const_bool)?;
    Ok(Some(()))
}

/// Trace `space.newbool` as its directional truth guard and prebuilt result.
/// `baseobjspace.py` chooses `w_True` or `w_False`, the prebuilt
/// singletons from `boolobject.py:79-80`; `pyjitpl.py opimpl_goto_if_not` records the  allow-line-citation
/// matching `GUARD_TRUE` / `GUARD_FALSE`, and `pyjitpl.py:525-526` replaces
/// the truth box with the promoted constant.
///
/// The guard is unconditional, because `newbool`'s `if b:` is: it is plain
/// RPython carrying no `@jit` hint (`baseobjspace.py` is only
/// `@signature`, and `boolobject.py` has none), so the tracer resolves it the
/// one way it observed and pins that with a guard no matter who consumes the
/// result.  This used to be restricted to a box that "decides one branch and
/// nothing else", on the reasoning that guarding an escaping bool pins a value
/// the trace would otherwise carry unconstrained.  Upstream grants no such
/// exemption, and the transform that does read like one —
/// `jtransform.py optimize_goto_if_not` — is a different thing: it fuses a
/// compare into a block's exitswitch, and `:205-211` makes it *refuse* when the
/// boolean has any other consumer.  It never decides whether `newbool` guards.
///
/// The store that publishes the result survives the guard: it mirrors the box
/// into the operand-stack slot the guard's own resume image describes, and
/// swapping the prebuilt singleton in for a recorded call result leaves it
/// storing the same value.
///
/// A constant truth takes the singleton with no guard and no resume image, the
/// way `generate_guard` (`pyjitpl.py`) returns early for a `Const` box.
pub(crate) fn walker_newbool_guarded<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    truth: OpRef,
    observed: bool,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if truth.is_constant() {
        return Ok(Some(walker_const_bool(ctx, observed)));
    }
    // No resume image, no guard: emitting one without a snapshot to resume
    // into would leave the bail with nowhere to land.  That is the only thing
    // that keeps a caller on the residual box.
    if ctx.fbw_mode.snapshot_sym.is_null() || dst_bank != 'r' {
        return Ok(None);
    }
    let guard = if observed {
        OpCode::GuardTrue
    } else {
        OpCode::GuardFalse
    };
    ctx.trace_ctx.record_guard(guard, &[truth], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    Ok(Some(walker_const_bool(ctx, observed)))
}

/// The prebuilt `w_True` / `w_False` singleton (`boolobject.py:79-80`) as a
/// trace constant carrying its own concrete shadow.
fn walker_const_bool<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>, observed: bool) -> OpRef {
    let result_obj = pyre_object::w_bool_from(observed);
    let const_bool = ctx.trace_ctx.const_ref(result_obj as i64);
    ctx.trace_ctx.set_opref_concrete(
        const_bool,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    const_bool
}

/// MAKE_FUNCTION inline emission: replace the
/// `jit_make_function_from_globals(globals, code)` residual with the
/// `NewWithVtable` + `SetfieldGc` set `function.py Function.__init__`
/// performs, so a `def` in a loop body virtualizes away instead of allocating a
/// `Function` per iteration.
///
/// Everything the constructor stores is loop-invariant when `code` is a
/// constant and `globals` names one fixed dict. A constant globals operand
/// is that dict. A non-constant one is accepted only when it is the
/// `GetfieldGcR` of `PyCode.w_globals` whose quasi-immutable dependency is
/// already recorded, so the dict stays the one `__builtins__` was read
/// from. `code` comes from a `LOAD_CONST`, and the remaining slots are
/// derived from the operands:
///
/// * `name` — `function.py:51 self.name = code.co_name`, a pointer into the
///   `Box::into_raw`'d `CodeObject`, which is never rewritten in place nor
///   freed.  This is the same pointer the residual stores
///   (`function_new_from_code` borrows it too), so a materialized function is
///   indistinguishable from an interpreted one.
/// * `w_name` / `w_qualname` — the code object's single realized `co_name` and
///   `co_qualname`, shared by every function built from it.
/// * `w_builtins` — CPython 3.14 `_PyEval_BuiltinsFromGlobals`, frozen at
///   construction from `globals['__builtins__']`.  Only the allocation-free
///   shape is reproduced: `__builtins__` naming a module, reduced to its dict.
///   An absent or non-module `__builtins__` routes
///   `pick_builtin_obj_checked` through `w_module_new_aliasing_dict` / the
///   default-module build, which mint a fresh object per call and so cannot be
///   baked — those decline to the residual.
///
/// Soundness for a quasi `PyCode.w_globals` read is that marker: it keeps
/// `globals` equal to the dict whose `__builtins__` was baked. The module
/// dict's `version?` still revokes the loop when that name is rebound,
/// exactly as it does for a shadowing insert under the LOAD_GLOBAL cell
/// fold.  Nothing watches the code object's `co_name` / `co_qualname`
/// because neither is mutable in place — `code.replace()` clones first and
/// yields a different code object, which is a different constant.
///
/// Declines (each falls through to the residual, which stays correct): a
/// non-constant `code` operand, a globals operand that is neither a
/// constant nor that quasi `PyCode.w_globals` read, a globals operand with
/// no concrete module dict, a non-`PyCode` or bodyless code object, an
/// unbakeable `__builtins__`, and any baked pointer the collector may
/// relocate.
pub(crate) fn try_walker_specialize_make_function<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 {
        return Ok(None);
    }
    let (globals_live, code_op) = (r_args[0], r_args[1]);
    if !code_op.is_constant() {
        return Ok(None);
    }
    let Some(w_code) = walker_concrete_ref_object(ctx, code_op) else {
        return Ok(None);
    };
    // A `BuiltinCode`-backed carrier is immortal and boxes its name through
    // `malloc_raw`; `is_code` admits only the `PyCode` shape this reproduces.
    if !unsafe { pyre_interpreter::is_code(w_code) } {
        return Ok(None);
    }
    let code_roots = pyre_object::gc_roots::push_roots();
    let code_slot = code_roots.base();
    let w_code = code_roots.pin_root(w_code);
    let code_ptr =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code) } as *const pyre_interpreter::CodeObject;
    if code_ptr.is_null() {
        return Ok(None);
    }

    // Realizing the name/qualname are the fold's collection points (they
    // allocate once per code object and hit the cache afterwards), so they run
    // here. `CodeObject` is a permanent `Box`, while its ordinary `PyCode`
    // wrapper is kept in the shadow stack and reloaded between the two calls.
    let w_name = unsafe { pyre_interpreter::pycode::w_code_name_obj(w_code) };
    if w_name.is_null() {
        return Ok(None);
    }
    let w_qualname =
        unsafe { pyre_interpreter::pycode::w_code_qualname_obj(code_roots.get(code_slot)) };
    if w_qualname.is_null() {
        return Ok(None);
    }
    let Some(w_globals) = walker_concrete_ref_object(ctx, globals_live) else {
        return Ok(None);
    };
    // Restrict to a module namespace before probing it directly, so the slot
    // read below walks the same storage `pick_builtin_obj_checked`'s
    // `finditem_str` does.  This is also what `walker_pin_namespace_version`
    // needs, but that one emits IR, so it runs only once the fold commits.
    if unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(w_globals) }.is_null()
    {
        return Ok(None);
    }
    // `function_new_impl`'s `w_builtins` derivation, restricted to the branch
    // that allocates nothing and therefore answers the same object each run.
    let w_builtins_module = unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }
        .unwrap_or(pyre_object::PY_NULL);
    if w_builtins_module.is_null() || !unsafe { pyre_object::is_module(w_builtins_module) } {
        return Ok(None);
    }
    let w_builtins = unsafe { pyre_object::w_module_get_w_dict(w_builtins_module) };
    if w_builtins.is_null() {
        return Ok(None);
    }
    // `w_builtins` / `w_name` / `w_qualname` are recorded as `ConstPtr`
    // arguments of the MAKE_FUNCTION body below.  Those slots are forwarded
    // by `walk_const_ptr_refs` and loaded from the gcref table at run;
    // movability does not decide the fold.
    // `__builtins__` was read from the dict observed on this iteration.
    // A mutable globals box is not that dict for the rest of the trace.
    if !crate::state::globals_read_keeps_recorded_namespace(ctx.trace_ctx, globals_live) {
        return Ok(None);
    }

    // commit to the fold: emit IR (no further declines)
    // `globals['__builtins__']` may be rebound after this function is built,
    // and a later iteration must then see the new mapping.  Pinning the
    // namespace `version?` revokes the loop instead.
    walker_pin_namespace_version(ctx, op_pc, w_globals)?;
    let header_w_class = ctx
        .trace_ctx
        .const_ref(pyre_object::get_instantiate(&pyre_interpreter::FUNCTION_TYPE) as i64);
    // `function.py can_change_code = True` for a plain `def`.
    let can_change_code = ctx.trace_ctx.const_int(1);
    let name = ctx
        .trace_ctx
        .const_ref(unsafe { &(*code_ptr).obj_name } as *const String as i64);
    let w_name_const = ctx.trace_ctx.const_ref(w_name as i64);
    let w_builtins_const = ctx.trace_ctx.const_ref(w_builtins as i64);
    let w_qualname_const = ctx.trace_ctx.const_ref(w_qualname as i64);
    let func_op = crate::helpers::emit_make_function_inline(
        ctx.trace_ctx,
        header_w_class,
        code_op,
        can_change_code,
        name,
        w_name_const,
        globals_live,
        w_builtins_const,
        w_qualname_const,
    );
    ctx.trace_ctx.heap_cache_mut().class_now_known(func_op);
    // Tracing is execution: build the concrete function the rest of the walk
    // observes.  A fresh `Function` per evaluation is what MAKE_FUNCTION
    // produces anyway, so the trace allocating its own is not an identity
    // divergence.
    let func = pyre_interpreter::runtime_ops::make_function_from_code_obj_with_globals_obj(
        w_code, w_globals,
    );
    ctx.trace_ctx.set_opref_concrete(
        func_op,
        majit_ir::Value::Ref(majit_ir::GcRef(func as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, func_op)?;
    Ok(Some(()))
}

/// #62: walker-native speculative specialization for the `BINARY_SUBSCR`
/// helper residual_call (oopspec `BinaryOp`, op_tag `Subscr`).  Ports
/// the former subscription/list-strategy path for the object-, int-,
/// float- and ascii-storage list strategies with a
/// non-negative concrete index: `guard_class LIST` + `guard_value(strategy)`
/// + unbox index + `IntLt` bounds guard, then the strategy-specific element
/// load — `getarrayitem_gc_r` against the `Ptr(GcArray(OBJECTPTR))` items
/// block for object storage (the element is a boxed Ref read directly), or a
/// raw-array getitem + `wrapint` / `wrapfloat` rebox for int/float storage,
/// or the same items-block load followed by `AsciiListStrategy.wrap` for
/// ascii storage (the element there is the shared `_utf8` payload, so the
/// wrap allocates only the header).
/// The authentic boxed result is taken from the same `execute_may_force_call`
/// path the generic leg uses.
///
/// A canonical Unicode-strategy dict with an exact-str key and a concrete hit
/// records the `rordereddict.py dict.lookup` oopspec producer: exact dict/key
/// guards, a strategy guard, elidable `rstr.ll_strhash`, `dict.lookup`, a
/// non-negative guard on the returned entry index, then a guarded value read.
///
/// Tuples, dict misses, empty-strategy lists, negative indices, and every
/// other operand shape fall through to the generic `CallMayForce` record
/// (`Ok(None)`), preserving Python `__getitem__` semantics.
pub(crate) fn try_walker_specialize_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let (Some(list_obj), Some(key_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
    ) else {
        return Ok(None);
    };

    // Exact `FrameLocalsProxy`: descend `__getitem__`. No fold row — the
    // reader is the body (`fast2locals` / `locals_plus_value`, both
    // `@jit.unroll_safe`). A non-exact proxy or a failed descent keeps
    // the generic residual below.
    if unsafe {
        pyre_interpreter::pyframe::frame_locals_proxy::is_frame_locals_proxy(list_obj)
            && walker_exact_builtin_class(list_obj).is_some()
    } {
        if let Some(hit) = try_walker_orthodox_frame_locals_getitem(
            ctx, op_pc, list_op, key_op, list_obj, key_obj, dst, dst_bank,
        )? {
            return Ok(Some(hit));
        }
    }

    if let Some(hit) = walker_probe_exact_dict_hit(list_obj, key_obj)? {
        return walker_emit_exact_dict_hit(
            ctx, op_pc, list_op, key_op, list_obj, hit, dst, dst_bank,
        );
    }

    // #171/#11 Approach C: canonical array-backed `W_TupleObject[i]`.  Two
    // gates, both required:
    //   * `ob_type == &TUPLE_TYPE` (tupleobject.py / tupleobject.rs) —
    //     NOT `is_tuple()` (which also accepts the three
    //     SPECIALISED_TUPLE_{II,FF,OO} variants).  Specialised tuples store
    //     `value0`/`value1` inline with no `wrappeditems` block, so a
    //     `getfield(wrappeditems)` on one yields garbage.
    //   * `w_class == canonical tuple` — a tuple SUBCLASS instance carries
    //     `TUPLE_USER_TYPE` and may override `__getitem__`; `baseobjspace::getitem` honours that
    //     override (subclass_special_override) so the pure `wrappeditems[i]`
    //     load must NOT be taken for it.
    // A failing gate falls to the generic residual.  The paired runtime
    // `guard_class(&TUPLE_TYPE)` + exact `w_class` guard (in
    // `try_walker_orthodox_subscr_tuple_item`) deopt any later non-canonical
    // tuple or subclass instance flowing in.
    let tuple_canonical = unsafe {
        std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::TUPLE_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
            )
    };
    // Serve the canonical layout and the arity-2 specialisations from the
    // real reader.
    if tuple_canonical || specialised_pair_kind(unsafe { (*list_obj).ob_type }).is_some() {
        if let Some(hit) = spec_gate(SpecFold::SubscrTupleDescent, || {
            try_walker_orthodox_subscr_tuple_item(
                ctx, op_pc, list_op, key_op, list_obj, key_obj, dst, dst_bank,
            )
        })? {
            return Ok(Some(hit));
        }
    }

    if tuple_canonical && unsafe { pyre_object::is_slice(key_obj) } {
        return spec_gate(SpecFold::SubscrTupleSlice2, || {
            try_walker_orthodox_subscr_tuple_slice(
                ctx, op_pc, list_op, key_op, list_obj, key_obj, dst, dst_bank,
            )
        });
    }

    // A `str` receiver descends `getitem_str` (`descr_getitem`'s scalar arm).
    // `is_exact_type` only checks the shared payload `ob_type`; a str subclass
    // carries that same value and distinguishes itself through `w_class`.
    // Admit exactly the shape the descent's guards pin, so recording a
    // subclass cannot manufacture a guard which its own concrete operand
    // already fails.  Every other key shape, slices included, keeps the
    // generic residual.
    if unsafe { pyre_object::is_str(list_obj) && walker_exact_builtin_class(list_obj).is_some() } {
        return try_walker_orthodox_str_getitem(
            ctx, op_pc, list_op, key_op, list_obj, key_obj, dst, dst_bank,
        );
    }

    // Exact `bytes`: `stringmethods.descr_getitem` / `_getitem_result`
    // (`strgetitem` + `newint`).  No fold row — the reader is the body.
    if unsafe {
        pyre_object::bytesobject::is_bytes(list_obj)
            && walker_exact_builtin_class(list_obj).is_some()
    } {
        if try_walker_orthodox_bytes_getitem(
            ctx, op_pc, list_op, key_op, list_obj, key_obj, dst, dst_bank,
        )?
        .is_some()
        {
            return Ok(Some(()));
        }
    }

    // The `dict.lookup` gate.  Both `w_class` checks are load-bearing: a dict
    // SUBCLASS shares `ob_type == &DICT_TYPE` but retags `w_class` and reaches
    // `__missing__` on a miss, and a str SUBCLASS key may override `__hash__` /
    // `__eq__`, so neither may take the exact-str probe.  The strategy check is
    // what makes the probe non-raising: `UnicodeDictStrategy` hands the dict to
    // `ObjectDictStrategy` the moment a non-exact-str key is stored, so while
    // it holds, every stored key is an exact str and the comparisons are WTF-8
    // byte equality (`dictmultiobject.py+` `r_dict(unicode_eq,
    // unicode_hash)`).
    let canonical_dict = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    let dict_unicode = !canonical_dict.is_null()
        && unsafe {
            std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*list_obj).w_class, canonical_dict)
                && pyre_object::dictmultiobject::w_dict_get_strategy(list_obj).strategy_kind()
                    == pyre_object::dictmultiobject::StrategyKind::Unicode
        };
    let canonical_str = if dict_unicode {
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE)
    } else {
        std::ptr::null_mut()
    };
    let dict_unicode_hit = !canonical_str.is_null()
        && unsafe {
            std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::STR_TYPE)
                && std::ptr::eq((*key_obj).w_class, canonical_str)
        };
    if dict_unicode_hit {
        let hash = unsafe { pyre_object::dictmultiobject::w_dict_unicode_key_hash(key_obj) };
        let index = unsafe {
            pyre_object::dictmultiobject::w_dict_unicode_lookup_index(list_obj, key_obj, hash, 0)
        };
        if index < 0 {
            return Ok(None);
        }

        let Some(boxed_result_i64) = walker_execute_may_force_boxed(ctx, allboxes, call_descr)
        else {
            return Ok(None);
        };

        walker_guard_exact_instance(
            ctx,
            op_pc,
            list_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
            canonical_dict,
        )?;
        let strategy = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            list_op,
            crate::descr::dict_strategy_word_descr(),
        );
        walker_guard_fold_int(
            ctx,
            op_pc,
            strategy,
            &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64,
        )?;
        walker_guard_exact_str(ctx, op_pc, key_op)?;

        let hash_effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::ElidableCannotRaise,
            majit_ir::OopSpecIndex::None,
        );
        // Both residuals bind the macro-emitted `__majit_call_target_*`
        // trampoline rather than the raw fn: the wasm backend derives a
        // residual's `call_indirect` type from the descr alone — `(i64 x n) ->
        // i64` — so a raw `*mut PyObject` argument, `i32` on wasm32, traps
        // `indirect call type mismatch`. The trampoline takes and returns the
        // uniform machine word everywhere, and is the address `jit_fnaddr`
        // registers for these paths.
        let hash_fn = {
            let f: extern "C" fn(i64) -> i64 =
                pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_key_hash;
            f as *const ()
        };
        let hash_op = ctx.trace_ctx.call_typed_with_effect_pure(
            OpCode::CallI,
            hash_fn,
            &[key_op],
            &[majit_ir::Type::Ref],
            majit_ir::Type::Int,
            hash_effect,
            &[
                majit_ir::Value::Int(hash_fn as i64),
                majit_ir::Value::Ref(majit_ir::GcRef(key_obj as usize)),
            ],
            majit_ir::Value::Int(hash),
        );

        let mut lookup_effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::DictLookup,
        );
        lookup_effect.extradescrs = Some(vec![
            crate::descr::dict_lookup_namespace_descr(),
            crate::descr::dict_lookup_entries_array_descr(),
        ]);
        let lookup_flag = ctx.trace_ctx.const_int(0);
        let lookup_fn: extern "C" fn(i64, i64, i64, i64) -> i64 =
            pyre_object::dictmultiobject::__majit_call_target_w_dict_unicode_lookup_index;
        let index_op = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            lookup_fn as *const (),
            &[list_op, key_op, hash_op, lookup_flag],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
            ],
            majit_ir::Type::Int,
            lookup_effect,
        );
        ctx.trace_ctx
            .set_opref_concrete(index_op, majit_ir::Value::Int(index));

        let zero = ctx.trace_ctx.const_int(0);
        let nonneg = ctx.trace_ctx.record_op(OpCode::IntGe, &[index_op, zero]);
        ctx.trace_ctx
            .set_opref_concrete(nonneg, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonneg])?;

        let value = ctx.trace_ctx.call_ref_typed_with_effect(
            crate::helpers::jit_dict_value_at as *const (),
            &[list_op, index_op, key_op, hash_op],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
            ],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        walker_guard_stamped_nonnull(
            ctx,
            op_pc,
            value,
            boxed_result_i64 as pyre_object::PyObjectRef,
        )?;
        write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
        return Ok(Some(()));
    }

    // Gate: EXACT list, non-negative int index in bounds, object-, int-,
    // float-, int-or-float-, ascii- or bytes-storage.  A bool index (`is_int` accepts `W_BoolObject`)
    // is fine:
    // bool shares int's `intval`, so the walk unboxes it through `&BOOL_TYPE`.
    // A list SUBCLASS instance shares `ob_type == &LIST_TYPE`
    // but retags `w_class` and may override `__getitem__`; `is_exact_list`
    // excludes it so it falls to the generic residual (which honours the
    // override) instead of this direct-storage load.
    let (sid, index) = unsafe {
        if !pyre_object::is_exact_list(list_obj) || !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        if index as usize >= pyre_object::w_list_len(list_obj) {
            return Ok(None);
        }
        let sid = if pyre_object::w_list_uses_int_storage(list_obj) {
            1i64
        } else if pyre_object::w_list_uses_float_storage(list_obj) {
            2i64
        } else if pyre_object::listobject::w_list_uses_int_or_float_storage(list_obj) {
            pyre_object::listobject::ListStrategy::IntOrFloat as i64
        } else if pyre_object::w_list_uses_object_storage(list_obj) {
            0i64
        } else if pyre_object::listobject::w_list_uses_ascii_storage(list_obj) {
            pyre_object::listobject::ListStrategy::Ascii as i64
        } else if pyre_object::listobject::w_list_strategy(list_obj)
            == pyre_object::listobject::ListStrategy::Bytes
        {
            pyre_object::listobject::ListStrategy::Bytes as i64
        } else {
            // Empty-strategy list: no concrete element to read.
            return Ok(None);
        };
        (sid, index)
    };

    // Walk `w_list_getitem_inner`. A missing body or an unfinished walk
    // leaves the generic residual.
    let Some(boxed) = try_walker_orthodox_list_getitem(
        ctx, op_pc, list_op, key_op, list_obj, key_obj, sid, index,
    )?
    else {
        return Ok(None);
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// Descend `baseobjspace::getitem_tuple` for an exact tuple and an exact
/// slice. The slice arm is `tuple_descr_getslice` (`descr_getitem` →
/// `_getslice`): `slice.indices`, then the source items, then `newtuple`.
fn try_walker_orthodox_subscr_tuple_slice<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    tuple_op: OpRef,
    slice_op: OpRef,
    tuple_obj: pyre_object::PyObjectRef,
    slice_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    let slice_typeobj = pyre_object::get_instantiate(&pyre_object::SLICE_TYPE);
    if unsafe { !std::ptr::eq((*slice_obj).w_class, slice_typeobj) } {
        return Ok(None);
    }
    // `slice_unpack` raises on a zero step and runs `__index__` for any
    // other component. Decline before the guards so that work stays on
    // the residual call.
    let plain = |w: pyre_object::PyObjectRef| unsafe {
        pyre_object::is_none(w)
            || (pyre_object::is_int(w)
                && pyre_object::is_exact_type(w, &pyre_object::pyobject::INT_TYPE))
    };
    let (w_start, w_stop, w_step) = unsafe {
        (
            pyre_object::w_slice_get_start(slice_obj),
            pyre_object::w_slice_get_stop(slice_obj),
            pyre_object::w_slice_get_step(slice_obj),
        )
    };
    if !plain(w_start) || !plain(w_stop) || !plain(w_step) {
        return Ok(None);
    }
    if unsafe { !pyre_object::is_none(w_step) && pyre_object::w_int_get_value(w_step) == 0 } {
        return Ok(None);
    }
    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode_cached(
        "pyre_interpreter::baseobjspace::getitem_tuple",
    ) else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        tuple_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
    )?;
    let tuple_type_addr = &pyre_object::TUPLE_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, tuple_op, tuple_type_addr)?;
    let slice_type_addr = &pyre_object::SLICE_TYPE as *const _ as i64;
    walker_guard_exact_instance(ctx, op_pc, slice_op, slice_type_addr, slice_typeobj)?;
    ctx.trace_ctx.set_opref_concrete(
        tuple_op,
        majit_ir::Value::Ref(majit_ir::GcRef(tuple_obj as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        slice_op,
        majit_ir::Value::Ref(majit_ir::GcRef(slice_obj as usize)),
    );
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "subscr_tuple_slice_commit",
        "getitem_tuple_call_site",
        &[],
        &[],
        &[tuple_op, slice_op],
        &[ConcreteValue::Ref(tuple_obj), ConcreteValue::Ref(slice_obj)],
        &[],
    );
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] SUBSCR-TUPLE-SLICE-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Descend `w_tuple_getitem`'s compiled body for a tuple subscript whose
/// receiver class and item index are both known at trace time, instead of
/// re-emitting that body's length test and field reads by hand.
///
/// This is the orthodox shape, and it replaced the hand-written
/// `subscr_specialised_pair` reader that stood in for it:
/// upstream's `getitem` is an ordinary graph the tracer inlines
/// (`specialisedtupleobject.py`, whose `getitem` unrolls `iter_n` to the
/// matching `value%s`), and the callee's `ob_type` chain folds against the
/// pinned class down to the one specialisation this trace saw.
///
/// The trace-time range check here is a decline gate, not the trace's safety
/// argument: it keeps an out-of-range subscript on the generic residual instead
/// of tracing a raising path.  What holds for the *next* receiver is the
/// callee's own length test, which is why this enters at `w_tuple_getitem`
/// rather than at the `_known` reader it wraps -- the reader is documented
/// "known-in-bounds" and carries no test to record.
fn try_walker_orthodox_subscr_tuple_item<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // The canonical layout, whose caller has already checked the exact class,
    // and the arity-2 specialisations.
    let spec_type = unsafe { (*seq_obj).ob_type };
    let canonical = std::ptr::eq(spec_type, &pyre_object::pyobject::TUPLE_TYPE);
    if !canonical && specialised_pair_kind(spec_type).is_none() {
        return Ok(None);
    }
    // Exact int keys only: a slice, a bool or an int subclass reaches a
    // different objspace path, and the callee takes a machine index.
    if !unsafe { pyre_object::is_int(key_obj) } {
        return Ok(None);
    }
    let raw_key = unsafe { pyre_object::w_int_get_value(key_obj) };
    let len = unsafe { pyre_object::tupleobject::w_tuple_len(seq_obj) } as i64;
    let index = if raw_key < 0 { raw_key + len } else { raw_key };
    if !(0..len).contains(&index) {
        return Ok(None);
    }

    // Resolve every possible decline before recording a guard.
    let Some(prep) = prepare_orthodox_descent(ctx, op_pc, &TUPLE_GETITEM_DESCENT) else {
        return Ok(None);
    };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // A specialisation carries its own `ob_type`, so its class guard is the
    // whole precondition, and its length is 2 by construction.  The canonical
    // layout also pins the exact class, since a subclass instance may
    // override `__getitem__`.
    if canonical {
        walker_guard_exact_w_class(
            ctx,
            op_pc,
            seq_op,
            pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::TUPLE_TYPE),
        )?;
    }
    walker_guard_specialised_pair_class(ctx, op_pc, seq_op, spec_type)?;

    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let key_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(key_index, majit_ir::Value::Int(raw_key));
    let index_arg = if canonical {
        key_index
    } else {
        // Freeze the key: the two slots are separate fields, so the callee's
        // `match idx` folds to one of them only against a constant.
        let index_arg = ctx.trace_ctx.const_int(raw_key);
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[key_index, index_arg])?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(key_index, index_arg);
        index_arg
    };
    // The class guard and the frozen-key `GuardValue` carry snapshots.
    // An unsupported body cuts back through them; leaving the snapshots
    // would name boxes the later remap has dropped.
    let walked = run_prepared_orthodox_descent(
        ctx,
        op_pc,
        prep,
        &[(index_arg, raw_key)],
        &[(seq_op, seq_obj)],
        &[],
        dst,
        dst_bank,
        &TUPLE_GETITEM_DESCENT,
        None,
        false, // index payload, not a BinaryOperator tag
    )?;
    orthodox_descent_unit(ctx, op_pc, walked, Some(pre_fold_pos))
}

const TUPLE_GETITEM_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::tupleobject::w_tuple_getitem",
    commit_label: "subscr_tuple_item_commit",
    call_site_label: "w_tuple_getitem_known_call_site",
    decline_tag: "SUBSCR-TUPLE-SUBWALK",
};

/// An operator's descent: which body to enter and how the trace names the
/// site.
struct HelperDescent {
    /// Canonical path of the operator's own body -- `descroperation::pos`,
    /// `opcode_ops::binary_value_from_tag` -- not a split of it.
    path: &'static str,
    commit_label: &'static str,
    call_site_label: &'static str,
    decline_tag: &'static str,
}

/// `argument.py` `_match_signature` `space.newdict(kwargs=True)` —
/// `dictmultiobject.py allocate_and_init_instance` kwargs branch.
const W_DICT_NEW_KWARGS_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::dictmultiobject::w_dict_new_kwargs",
    commit_label: "w_dict_new_kwargs_commit",
    call_site_label: "w_dict_new_kwargs_call_site",
    decline_tag: "KWARGS-DICT-NEW-SUBWALK",
};

/// `argument.py` `_match_signature` `space.setitem(w_kwds, w_key, w_value)`
/// — `dictmultiobject.py W_DictMultiObject.setitem`.
const W_DICT_STORE_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::dictmultiobject::w_dict_store",
    commit_label: "w_dict_store_commit",
    call_site_label: "w_dict_store_call_site",
    decline_tag: "KWARGS-DICT-SETITEM-SUBWALK",
};

/// The `BINARY_OP` helper itself: the operator tag is a trace-time constant,
/// so its `match` folds and only the selected operator's body is traced.
const BINARY_OP_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::opcode_ops::binary_value_from_tag",
    commit_label: "binary_op_commit",
    call_site_label: "binary_op_call_site",
    decline_tag: "BINARY-OP-SUBWALK",
};

/// BINARY_SLICE residual: walk `runtime_ops::binary_slice_values_inner`.
///
/// Flatten lowers the opcode to `bh_binary_slice_fn`, a MayForce pointer
/// with no jitcode of its own. The inner opens a RootScope and batch-
/// publishes the three operands; `shadow_stack_erase` scalar-replaces that
/// bracket (`w_range_iter_one_arg_new`), so the walk never sees `push_roots`.
/// Integer-list copy walks `ll_listslice_inner` → `ll_listslice_new_int_list`
/// (`rlist.py ll_listslice_startstop`): `newlist(length)` becomes `new_array`,
/// `list.int_items` a getfield, `list.ll_arraycopy` `OS_ARRAYCOPY` as CallN.
/// `w_list_adopt_int_items` is residual (`dont_look_inside`), the
/// `w_tuple_adopt_fixed_items` / `w_int_gc_alloc` collector-heap boundary.
/// The body's `is_list` / `is_str` / `is_tuple` tests are the guards; a
/// custom `__index__` is `eval_slice_index` (`sliceobject.py`
/// `_eval_slice_index`). Bytes / bytearray / user `__getitem__` stay in
/// `binary_slice_getitem_fallback` (`dont_look_inside`). Not a spec-fold row.
const BINARY_SLICE_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::runtime_ops::binary_slice_values_inner",
    commit_label: "binary_slice_commit",
    call_site_label: "binary_slice_call_site",
    decline_tag: "BINARY-SLICE-SUBWALK",
};

/// Walk `binary_slice_values_inner` for `obj[start:stop]`. Declines when
/// the body is missing or the walk does not finish; the residual stays.
///
/// Bounds stay the interpreter operands. `eval_slice_index` /
/// `space.getindex_w` (`sliceobject.py` `_eval_slice_index`) is the
/// body's own conversion; a pre-descent `try_walker_inline_index` would
/// feed the helper an exact int and skip that control flow.
pub(crate) fn try_walker_orthodox_binary_slice<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 3 {
        return Ok(None);
    }
    let Some(obj) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    let Some(start) = walker_concrete_ref_object(ctx, r_args[1]) else {
        return Ok(None);
    };
    let Some(stop) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return Ok(None);
    };
    try_walker_orthodox_descent(
        ctx,
        op.pc,
        &[],
        &[(r_args[0], obj), (r_args[1], start), (r_args[2], stop)],
        &[],
        dst,
        dst_bank,
        &BINARY_SLICE_DESCENT,
    )
}

const WRITE_CELL_DESCENT: HelperDescent = HelperDescent {
    path: "write_cell",
    commit_label: "write_cell_commit",
    call_site_label: "write_cell_call_site",
    decline_tag: "WRITE-CELL-SUBWALK",
};

const UNWRAP_CELL_DESCENT: HelperDescent = HelperDescent {
    path: "unwrap_cell",
    commit_label: "unwrap_cell_commit",
    call_site_label: "unwrap_cell_call_site",
    decline_tag: "UNWRAP-CELL-SUBWALK",
};

const FLOAT_FREXP_MANTISSA_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_frexp_mantissa",
    commit_label: "float_frexp_mantissa_commit",
    call_site_label: "float_frexp_mantissa_commit_site",
    decline_tag: "FLOAT-FREXP-MANTISSA-SUBWALK",
};

const INT_FREXP_EXPONENT_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_frexp_exponent",
    commit_label: "int_frexp_exponent_commit",
    call_site_label: "int_frexp_exponent_commit_site",
    decline_tag: "INT-FREXP-EXPONENT-SUBWALK",
};

/// Whether `callable` is the canonical builtin `math.<name>`, asked of the
/// `math` module through the optional-module hooks.
fn is_math_builtin(callable: pyre_object::PyObjectRef, name: &str) -> bool {
    pyre_interpreter::importing::optional_module_hooks()
        .and_then(|hooks| (hooks.math_builtin_name)(callable))
        == Some(name)
}

/// `math.frexp(x)` on an exact int/float argument.  ll_math.py
/// `ll_math_frexp` is two unboxed results; interp_math.py `frexp`
/// then does `newtuple2(newfloat(mant), newint(expo))`.  Walk the two
/// boxing leaves and emit the specialised pair.  Rebound callables,
/// subclasses, and other coercion shapes retain the residual.
pub(crate) fn try_walker_specialize_math_frexp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor {
        return Ok(None);
    }
    if let Some((callable, operands)) = plain_builtin_call_concretes(ctx, code, op, r_args, 1) {
        if is_math_builtin(callable, "frexp") && r_args.len() >= 3 {
            // Every decline that needs no trace runs before the callable
            // guard, so a declined call leaves nothing recorded.
            if let Some((is_int, x)) = frexp_fold_operand(operands[0]) {
                walker_guard_fold_callable(ctx, op.pc, r_args[0], callable)?;
                if try_walker_orthodox_frexp(ctx, op.pc, r_args[2], operands[0], is_int, x, dst)?
                    .is_some()
                {
                    return Ok(Some(()));
                }
            }
        }
    }
    Ok(None)
}

/// The unboxed value of an exact int/bool/float `obj` the frexp leaves
/// accept, with whether it came from an int.  The leaves assume a normal
/// finite non-zero: specials and subnormals stay on the residual, matching
/// ll_math_frexp's first-arm return of `(x, 0)`.
fn frexp_fold_operand(obj: pyre_object::PyObjectRef) -> Option<(bool, f64)> {
    if !unsafe { pyre_object::is_exact_builtin_instance(obj) } {
        return None;
    }
    let (is_int, x) = if unsafe { pyre_object::is_float(obj) } {
        (false, unsafe { pyre_object::w_float_get_value(obj) })
    } else if unsafe { pyre_object::is_int(obj) || pyre_object::is_bool(obj) } {
        (true, unsafe { pyre_object::w_int_get_value(obj) as f64 })
    } else {
        return None;
    };
    if !x.is_finite() || x == 0.0 || !x.abs().is_normal() {
        return None;
    }
    Some((is_int, x))
}

fn try_walker_orthodox_frexp<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    obj: pyre_object::PyObjectRef,
    is_int: bool,
    x: f64,
    dst: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    let dst_bank = 'r';
    let xa =
        walker_coerce_dispatching_operand_to_float(ctx, op_pc, operand, obj, is_int, x, false)?;
    // `MIN_POSITIVE <= |x| < inf`: finite, normal, non-zero.
    let min_normal = ctx
        .trace_ctx
        .const_float(f64::MIN_POSITIVE.to_bits() as i64);
    let infinity = ctx.trace_ctx.const_float(f64::INFINITY.to_bits() as i64);
    let abs_x = ctx.trace_ctx.record_op(OpCode::FloatAbs, &[xa]);
    ctx.trace_ctx
        .set_opref_concrete(abs_x, majit_ir::Value::Float(x.abs()));
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatLt, &[abs_x, infinity], true)?;
    walker_float_cmp_guard(ctx, op_pc, OpCode::FloatLt, &[abs_x, min_normal], false)?;
    // Both leaves write `dst`; roll the pair back together if the
    // second walk declines after the first already boxed.
    let pre_pair = ctx.trace_ctx.get_trace_position();
    let cut_pair = |ctx: &mut WalkContext<'_, '_, Sym>| {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_pair);
        ctx.trace_ctx.heap_cache_mut().reset();
    };
    let mut mantissa = None;
    if try_walker_orthodox_descent_ex(
        ctx,
        op_pc,
        &[],
        &[],
        &[(xa, x)],
        dst,
        dst_bank,
        &FLOAT_FREXP_MANTISSA_DESCENT,
        Some(&mut mantissa),
        true,
    )?
    .is_none()
    {
        cut_pair(ctx);
        return Ok(None);
    }
    let Some(mantissa) = mantissa else {
        cut_pair(ctx);
        return Ok(None);
    };
    let mut exponent = None;
    if try_walker_orthodox_descent_ex(
        ctx,
        op_pc,
        &[],
        &[],
        &[(xa, x)],
        dst,
        dst_bank,
        &INT_FREXP_EXPONENT_DESCENT,
        Some(&mut exponent),
        true,
    )?
    .is_none()
    {
        cut_pair(ctx);
        return Ok(None);
    }
    let Some(exponent) = exponent else {
        cut_pair(ctx);
        return Ok(None);
    };
    let tuple = crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, mantissa, exponent);
    // UNPACK_SEQUENCE reads the pair off the concrete specialised
    // tuple.  Build that host object from the same boxes the descent
    // cached, so getfield_gc agrees with the heapcache.
    let (Some(majit_ir::Value::Ref(mantissa_ref)), Some(majit_ir::Value::Ref(exponent_ref))) = (
        ctx.trace_ctx.box_value(mantissa),
        ctx.trace_ctx.box_value(exponent),
    ) else {
        cut_pair(ctx);
        return Ok(None);
    };
    if mantissa_ref.0 == 0 || exponent_ref.0 == 0 {
        cut_pair(ctx);
        return Ok(None);
    }
    let concrete_tuple = pyre_object::w_specialised_tuple_oo_new(
        mantissa_ref.0 as pyre_object::PyObjectRef,
        exponent_ref.0 as pyre_object::PyObjectRef,
    );
    if concrete_tuple.is_null() {
        cut_pair(ctx);
        return Ok(None);
    }
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
    );
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, tuple)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Descend a generated cell helper (`write_cell` / `unwrap_cell`) the way
/// [`try_walker_orthodox_descent`] enters `binary_value_from_tag`.  The
/// IR comes from the helper's jitcode, not a hand-written getfield/setfield.
fn descend_named_cell_helper<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    jc_index: usize,
    ref_args: &[(OpRef, pyre_object::PyObjectRef)],
    descent: &HelperDescent,
) -> Result<Option<OpRef>, DispatchError> {
    let decline = |why: &str| {
        if fbw_debug_abort_enabled() {
            eprintln!("[decline-why] {}-{why} pc={op_pc}", descent.decline_tag);
        }
        Ok(None)
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_index) else {
        return decline("NO-SUB-BODY");
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return decline("NO-SYM");
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return decline("SYM-NO-JITCODE");
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return decline("NESTED-ENTRY");
    };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    for &(operand, operand_obj) in ref_args {
        ctx.trace_ctx.set_opref_concrete(
            operand,
            majit_ir::Value::Ref(majit_ir::GcRef(operand_obj as usize)),
        );
    }
    let ref_oprefs: Vec<OpRef> = ref_args.iter().map(|&(opref, _)| opref).collect();
    let ref_concretes: Vec<ConcreteValue> = ref_args
        .iter()
        .map(|&(_, obj)| ConcreteValue::Ref(obj))
        .collect();
    let exc_before_subwalk = ctx.last_exc_value();
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        descent.commit_label,
        descent.call_site_label,
        &[],
        &[],
        &ref_oprefs,
        &ref_concretes,
        &[],
    );
    let (walk_outcome, _walk_start) = match walk {
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] {} pc={pc}", descent.decline_tag);
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Ok(pair) => pair,
        Err(error) => {
            if fbw_debug_abort_enabled() {
                match &error {
                    DispatchError::UnsupportedOpname { pc, key } => {
                        eprintln!(
                            "[decline-why] {}-ERR pc={op_pc} error=UnsupportedOpname leaf_pc={pc} key={key}",
                            descent.decline_tag
                        );
                    }
                    _ => {
                        eprintln!(
                            "[decline-why] {}-ERR pc={op_pc} error={}",
                            descent.decline_tag,
                            error.variant_name()
                        );
                    }
                }
            }
            return Err(error);
        }
    };
    let result =
        match promote_published_null_return_since(ctx, walk_outcome, op_pc, exc_before_subwalk)? {
            DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
                .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
            raised @ DispatchOutcome::SubRaise { .. } => {
                let _ = raised;
                return Ok(None);
            }
            _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
        };
    Ok(Some(result))
}

/// Walk generated `unwrap_cell` and return the unwrapped value's operand.
///
/// A ConstPtr cell folds the helper's type tests the way PyPy looks
/// inside `typeobject.py unwrap_cell`; the compiled loop keeps the
/// live getfield (or the identity return for a non-cell).
pub(crate) fn try_walker_orthodox_unwrap_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    stored: pyre_object::PyObjectRef,
) -> Result<Option<(OpRef, pyre_object::PyObjectRef)>, DispatchError> {
    if stored.is_null() {
        return Ok(None);
    }
    let Some(jc) = crate::jitcode_runtime::unwrap_cell_jitcode() else {
        return Ok(None);
    };
    let cell_opref = ctx.trace_ctx.const_ref(stored as i64);
    let Some(result) = descend_named_cell_helper(
        ctx,
        op_pc,
        jc.index(),
        &[(cell_opref, stored)],
        &UNWRAP_CELL_DESCENT,
    )?
    else {
        return Ok(None);
    };
    let result_obj = unsafe { pyre_object::celldict::unwrap_cell(stored) };
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
    );
    Ok(Some((result, result_obj)))
}

/// Walk `write_cell` for an in-place cell store.  Applies the helper
/// afterwards so the walk's remaining concrete reads see the write.
/// A pointer store runs the cell's write barrier inside
/// `celldict::write_cell`; an int store does not.
///
/// `celldict.py getdictvalue_no_unwrapping` promotes `self` and reads
/// `version?` on every lookup. `_setitem_str_cell_known` and `delitem`
/// call `mutated()`. Pin that field before baking `stored`: a later
/// delete or replacing store must revoke this compiled write, which
/// otherwise mutates the detached old cell. A watcher already installed
/// makes `pyjitpl.py opimpl_jit_force_quasi_immutable` abort the trace.
pub(crate) fn try_walker_orthodox_write_cell<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    ns: pyre_object::PyObjectRef,
    slot: usize,
    stored: pyre_object::PyObjectRef,
    value_opref: OpRef,
    new_value: pyre_object::PyObjectRef,
) -> Result<bool, DispatchError> {
    if stored.is_null() || new_value.is_null() {
        return Ok(false);
    }
    let Some(jc) = crate::jitcode_runtime::write_cell_jitcode() else {
        return Ok(false);
    };
    if !walker_pin_namespace_version(ctx, op_pc, ns)? {
        return Ok(false);
    }
    if crate::state::module_dict_cell_value_direct(ns, slot) != Some(stored) {
        return Ok(false);
    }
    let cell_opref = ctx.trace_ctx.const_ref(stored as i64);
    let saved_fbw_mode = ctx.fbw_mode;
    ctx.fbw_mode.cell_store_helper_subwalk = true;
    let descended = descend_named_cell_helper(
        ctx,
        op_pc,
        jc.index(),
        &[(cell_opref, stored), (value_opref, new_value)],
        &WRITE_CELL_DESCENT,
    );
    ctx.fbw_mode = saved_fbw_mode;
    if descended?.is_none() {
        return Ok(false);
    }
    if unsafe { pyre_object::celldict::is_int_mutable_cell(stored) } {
        let cell = stored as *const pyre_object::celldict::IntMutableCell;
        fbw_cell_store_journal_push(stored, unsafe { (*cell).intvalue });
    } else if unsafe { pyre_object::celldict::is_object_mutable_cell(stored) } {
        let cell = stored as *const pyre_object::celldict::ObjectMutableCell;
        fbw_obj_cell_store_journal_push(stored, unsafe { (*cell).w_value });
    }
    let replaced = unsafe { pyre_object::celldict::write_cell(Some(stored), new_value) };
    if replaced.is_some() {
        return Ok(false);
    }
    ctx.clear_last_exc_value();
    Ok(true)
}

/// intobject.py `_truediv` / `descr_truediv` after the `W_IntObject`
/// isinstance.  Walking `binary_value_from_tag` for `/` records the
/// whole `truediv_impl` dispatcher (`bigint_truediv`, dunder lookup)
/// and hung `listcomp_float_element_regression`.
const INT_TRUEDIV_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_truediv",
    commit_label: "int_truediv_commit",
    call_site_label: "int_truediv_call_site",
    decline_tag: "INT-TRUEDIV-SUBWALK",
};

/// floatobject.py `descr_add` / `descr_sub` / `descr_mul` / `descr_div`
/// after `_to_float`.  Exact float pairs walk these instead of the
/// `FloatAdd`+`wrapfloat` hand emit.
const FLOAT_ADD_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_add",
    commit_label: "float_add_commit",
    call_site_label: "float_add_call_site",
    decline_tag: "FLOAT-ADD-SUBWALK",
};

/// intobject.py `descr_add` after the two `intval` reads. The success arm
/// is [`_int_add`]: `checked_add` then `malloc_typed_managed`, which
/// `fuse_boxing_alloc` rewrites to `new_with_vtable`. Overflow stays
/// [`_int_add`]'s `dont_look_inside` arm.
const INT_ADD_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_add",
    commit_label: "int_add_commit",
    call_site_label: "int_add_call_site",
    decline_tag: "INT-ADD-SUBWALK",
};

/// intobject.py `descr_sub` after the two `intval` reads. Same split as
/// [`INT_ADD_DESCENT`]: [`_int_sub`]'s success arm is `malloc_typed_managed`,
/// and overflow stays `_int_sub_ovf`.
const INT_SUB_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_sub",
    commit_label: "int_sub_commit",
    call_site_label: "int_sub_call_site",
    decline_tag: "INT-SUB-SUBWALK",
};

/// intobject.py `descr_mul` after the two `intval` reads. Same split as
/// [`INT_ADD_DESCENT`]: [`_int_mul`]'s success arm is `malloc_typed_managed`,
/// and overflow stays `_int_mul_ovf`.
const INT_MUL_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_mul",
    commit_label: "int_mul_commit",
    call_site_label: "int_mul_call_site",
    decline_tag: "INT-MUL-SUBWALK",
};

const FLOAT_SUB_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_sub",
    commit_label: "float_sub_commit",
    call_site_label: "float_sub_call_site",
    decline_tag: "FLOAT-SUB-SUBWALK",
};

const FLOAT_MUL_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_mul",
    commit_label: "float_mul_commit",
    call_site_label: "float_mul_call_site",
    decline_tag: "FLOAT-MUL-SUBWALK",
};

const FLOAT_TRUEDIV_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_truediv",
    commit_label: "float_truediv_commit",
    call_site_label: "float_truediv_call_site",
    decline_tag: "FLOAT-TRUEDIV-SUBWALK",
};

/// floatobject.py `descr_pow` -> `_pow` (`float_pow_impl`). Distinct from
/// [`FLOAT_POW_DESCENT`], which is the `math.pow` leaf.
const FLOAT_DESCR_POW_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::float_pow_impl",
    commit_label: "float_descr_pow_commit",
    call_site_label: "float_descr_pow_call_site",
    decline_tag: "FLOAT-DESCR-POW-SUBWALK",
};

/// floatobject.py `_compare` after `_to_float` for two floats.
const FLOAT_LT_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_lt",
    commit_label: "float_lt_commit",
    call_site_label: "float_lt_call_site",
    decline_tag: "FLOAT-LT-SUBWALK",
};

const FLOAT_LE_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_le",
    commit_label: "float_le_commit",
    call_site_label: "float_le_call_site",
    decline_tag: "FLOAT-LE-SUBWALK",
};

const FLOAT_GT_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_gt",
    commit_label: "float_gt_commit",
    call_site_label: "float_gt_call_site",
    decline_tag: "FLOAT-GT-SUBWALK",
};

const FLOAT_GE_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_ge",
    commit_label: "float_ge_commit",
    call_site_label: "float_ge_call_site",
    decline_tag: "FLOAT-GE-SUBWALK",
};

const FLOAT_EQ_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_eq",
    commit_label: "float_eq_commit",
    call_site_label: "float_eq_call_site",
    decline_tag: "FLOAT-EQ-SUBWALK",
};

const FLOAT_NE_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_ne",
    commit_label: "float_ne_commit",
    call_site_label: "float_ne_call_site",
    decline_tag: "FLOAT-NE-SUBWALK",
};

/// floatobject.py `descr_pos`.
const FLOAT_POS_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_float_pos",
    commit_label: "float_pos_commit",
    call_site_label: "float_pos_call_site",
    decline_tag: "FLOAT-POS-SUBWALK",
};

/// intobject.py `descr_abs` after `ovfcheck`.
#[allow(dead_code)]
const INT_ABS_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_abs",
    commit_label: "int_abs_commit",
    call_site_label: "int_abs_call_site",
    decline_tag: "INT-ABS-SUBWALK",
};

/// intobject.py `descr_invert`.
const INT_INVERT_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_invert",
    commit_label: "int_invert_commit",
    call_site_label: "int_invert_call_site",
    decline_tag: "INT-INVERT-SUBWALK",
};

pub(crate) fn binary_value_from_tag_jitcode()
-> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
    crate::jitcode_runtime::pathed_runtime_jitcode_cached(BINARY_OP_DESCENT.path)
}

/// The `binary_value_from_tag` wrapper leaf, not `binary_value_from_tag_inner`.
///
/// The wrapper's entry `inline_call` of the inner is already the helper walk
/// (`perform_call`). Opening a descent of the wrapper there would suspend at
/// pc 0 inside the active driver, and the driver would push the wrapper again.
pub(crate) fn jitcode_name_is_binary_value_from_tag(name: &str) -> bool {
    name.ends_with("binary_value_from_tag")
}

/// True when `sub_body` is the `binary_value_from_tag` helper the
/// codewriter inlines for BINARY.  The per-index name table can miss a
/// helper that `pathed_jitcode_cached` still owns, and a name-only
/// check then skipped descent so a declined sub-walk residualized
/// `CallMayForce` (`binary_value_from_tag`) on fib bridges.
pub(crate) fn jitcode_is_binary_value_from_tag(
    pool: super::RawDescrPool<'_>,
    sub_index: usize,
    sub_body: &super::SubJitCodeBody,
) -> bool {
    if pool
        .inline_callee_name(sub_index)
        .is_some_and(jitcode_name_is_binary_value_from_tag)
    {
        return true;
    }
    crate::jitcode_runtime::pathed_jitcode_cached(BINARY_OP_DESCENT.path).is_some_and(|jc| {
        (matches!(pool, super::RawDescrPool::Global) && jc.index() == sub_index)
            || std::ptr::eq(jc.code.as_ptr(), sub_body.code.as_ptr())
    })
}

/// Tag for a declined helper walk that is `binary_value_from_tag` or a
/// named `add`/`sub`/`int_add` body.  Used so a bridge that cannot
/// stamp the helper resume word still emits `int_add` instead of
/// `CallMayForce` (`binary_value_from_tag`).
pub(crate) fn binary_op_tag_for_helper_index(
    pool: super::RawDescrPool<'_>,
    sub_index: usize,
    int_concretes: &[ConcreteValue],
) -> Option<i64> {
    let name = pool.inline_callee_name(sub_index)?;
    if jitcode_name_is_binary_value_from_tag(name) {
        return match int_concretes.first() {
            Some(ConcreteValue::Int(tag)) => Some(*tag),
            _ => None,
        };
    }
    binary_op_tag_for_helper_name(name)
}

/// `int_add` / `int_sub` / `int_mul` overflow arm: after `INT_*_OVF`
/// records overflow, emit `GUARD_OVERFLOW` and the same
/// `w_long_new(bigint_*_int_int(va, vb))` the interpreter takes
/// (`descroperation.rs int_mul`).
fn emit_int_ovf_to_long<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    lhs_raw: OpRef,
    rhs_raw: OpRef,
    la: i64,
    rb: i64,
    lhs_obj: pyre_object::PyObjectRef,
    rhs_obj: pyre_object::PyObjectRef,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    use pyre_interpreter::bytecode::BinaryOperator as B;
    use pyre_interpreter::objspace::descroperation as desc;
    let (helper, binop) = match opcode {
        OpCode::IntAddOvf => (desc::jit_bigint_add_int_int as *const (), B::Add),
        OpCode::IntSubOvf => (desc::jit_bigint_sub_int_int as *const (), B::Subtract),
        OpCode::IntMulOvf => (desc::jit_bigint_mul_int_int as *const (), B::Multiply),
        _ => return Ok(None),
    };
    // `record_int_ovf` on two ConstInts returns a ConstInt and records no
    // `INT_*_OVF`. An operand-less `GUARD_OVERFLOW` after that is
    // `InvalidLoop`. The overflow arm is already selected at record time.
    if !lhs_raw.is_constant() || !rhs_raw.is_constant() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardOverflow, &[])?;
    }
    let Ok(boxed_obj) = pyre_interpreter::opcode_ops::binary_value(lhs_obj, rhs_obj, binop) else {
        return Ok(None);
    };
    if boxed_obj.is_null() || unsafe { !pyre_object::is_long(boxed_obj) } {
        return Ok(None);
    }
    let raw_concrete = unsafe {
        *((boxed_obj as *const u8).add(pyre_object::longobject::LONG_VALUE_OFFSET) as *const i64)
    };
    let raw = ctx.trace_ctx.call_typed_with_effect_pure_can_raise(
        OpCode::CallR,
        helper,
        &[lhs_raw, rhs_raw],
        &[majit_ir::Type::Int, majit_ir::Type::Int],
        majit_ir::Type::Ref,
        majit_metainterp::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        &[
            majit_ir::Value::Int(helper as usize as i64),
            majit_ir::Value::Int(la),
            majit_ir::Value::Int(rb),
        ],
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        raw,
        majit_ir::Value::Ref(majit_ir::GcRef(raw_concrete as usize)),
    );
    if raw.inline_const_to_value().is_none() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoException, &[])?;
    }
    let boxed = crate::helpers::emit_box_long_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_long_size_descr(),
        crate::descr::long_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        boxed,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_obj as usize)),
    );
    Ok(Some(DispatchOutcome::SubReturn {
        result: Some(boxed),
    }))
}

/// The machine-int body of `int_add` / `int_sub` / `int_mul` / bitwise
/// and of `int_floordiv` / `int_mod` (`descroperation.rs`): unbox,
/// `int_*_ovf` or `int_and`/`or`/`xor` or the `OS_INT_PY_DIV` /
/// `OS_INT_PY_MOD` elidable, rebox.  Used when a helper walk cannot
/// stamp its resume word (`GuardResumeCoordinateUnavailable`) so the
/// call would otherwise become `CallMayForce`.  Does not walk
/// `binary_value_from_tag` — that re-enters the same `add` inline and
/// recurses.
pub(crate) fn try_emit_exact_int_binop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) else {
        return Ok(None);
    };
    unsafe {
        for obj in [lhs_obj, rhs_obj] {
            if !pyre_object::is_int(obj) || !pyre_object::is_exact_builtin_instance(obj) {
                return Ok(None);
            }
        }
    }
    let la = unsafe { pyre_object::w_int_get_value(lhs_obj) };
    let rb = unsafe { pyre_object::w_int_get_value(rhs_obj) };
    use pyre_interpreter::bytecode::BinaryOperator as B;
    let op = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag);
    let is_py_div = matches!(
        op,
        Some(B::FloorDivide | B::InplaceFloorDivide | B::Remainder | B::InplaceRemainder)
    );
    // A live zero divisor is `try_walker_specialize_binary_op_int_zero_div`.
    // Emitting `ll_int_py_div` here would dest-write a wrap value.
    if is_py_div && (rb == 0 || (la == i64::MIN && rb == -1)) {
        return Ok(None);
    }
    let opcode = match op {
        Some(B::Add | B::InplaceAdd) => OpCode::IntAddOvf,
        Some(B::Subtract | B::InplaceSubtract) => OpCode::IntSubOvf,
        Some(B::Multiply | B::InplaceMultiply) => OpCode::IntMulOvf,
        Some(B::And | B::InplaceAnd) => OpCode::IntAnd,
        Some(B::Or | B::InplaceOr) => OpCode::IntOr,
        Some(B::Xor | B::InplaceXor) => OpCode::IntXor,
        Some(B::FloorDivide | B::InplaceFloorDivide | B::Remainder | B::InplaceRemainder) => {
            OpCode::CallI
        }
        _ => return Ok(None),
    };
    // bool shares int's `intval` but carries `BOOL_TYPE`.  Unboxing both
    // operands through `&INT_TYPE` plants a `GUARD_CLASS INT` that fails on
    // every `acc + flag` / `flag * 2` and retraces the loop.
    let (lhs_type, lhs_descr) = crate::state::int_or_bool_unbox_type_descr(lhs_obj);
    let (rhs_type, rhs_descr) = crate::state::int_or_bool_unbox_type_descr(rhs_obj);
    let lhs_raw = walker_unbox_int_typed(ctx, op_pc, r_args[0], lhs_type, lhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, r_args[0], walker_numeric_builtin_class(lhs_obj))?;
    let rhs_raw = walker_unbox_int_typed(ctx, op_pc, r_args[1], rhs_type, rhs_descr)?;
    walker_guard_exact_w_class(ctx, op_pc, r_args[1], walker_numeric_builtin_class(rhs_obj))?;
    let (raw, concrete) = if is_py_div {
        // Same `OS_INT_PY_DIV` / `OS_INT_PY_MOD` elidable the descent records.
        // A declined helper walk used to residualize `binary_value_from_tag`
        // as `CallMayForce` of freshly boxed ints; the compiled bridge then
        // returned the divisor (`100 // -1` → `-1`) for `recur(1, 0)`.
        walker_emit_int_div_domain_guards(ctx, op_pc, lhs_raw, rhs_raw, la, rb)?;
        let is_div = matches!(op, Some(B::FloorDivide | B::InplaceFloorDivide));
        walker_emit_int_py_div_or_mod(ctx, lhs_raw, rhs_raw, la, rb, is_div)
    } else if matches!(
        opcode,
        OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf
    ) {
        let (raw, ovf_flag) = record_int_ovf(ctx, op_pc, opcode, lhs_raw, rhs_raw, Some((la, rb)))?;
        let heap_ovf = match opcode {
            OpCode::IntAddOvf => la.checked_add(rb).is_none(),
            OpCode::IntSubOvf => la.checked_sub(rb).is_none(),
            OpCode::IntMulOvf => la.checked_mul(rb).is_none(),
            _ => false,
        };
        // Heap values are passed as `known` so an unstamped unbox InputArg
        // still decides overflow from the live objects (`a * a` with a≈5e9).
        if ovf_flag || heap_ovf {
            return emit_int_ovf_to_long(
                ctx, op_pc, opcode, lhs_raw, rhs_raw, la, rb, lhs_obj, rhs_obj,
            );
        }
        if !raw.is_constant() {
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
        }
        let concrete = match opcode {
            OpCode::IntAddOvf => la.checked_add(rb),
            OpCode::IntSubOvf => la.checked_sub(rb),
            OpCode::IntMulOvf => la.checked_mul(rb),
            _ => None,
        };
        let Some(concrete) = concrete else {
            return Ok(None);
        };
        (raw, concrete)
    } else {
        let raw = ctx.trace_ctx.record_op(opcode, &[lhs_raw, rhs_raw]);
        let concrete = match opcode {
            OpCode::IntAnd => la & rb,
            OpCode::IntOr => la | rb,
            OpCode::IntXor => la ^ rb,
            _ => return Ok(None),
        };
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(concrete));
        (raw, concrete)
    };
    let both_bools = unsafe { pyre_object::is_bool(lhs_obj) && pyre_object::is_bool(rhs_obj) };
    if both_bools && matches!(opcode, OpCode::IntAnd | OpCode::IntOr | OpCode::IntXor) {
        let observed = concrete != 0;
        let Some(boxed) = walker_newbool_guarded(ctx, op_pc, raw, observed, dst_bank)? else {
            return Ok(None);
        };
        let _ = (dst, dst_bank);
        return Ok(Some(DispatchOutcome::SubReturn {
            result: Some(boxed),
        }));
    }
    let boxed_ptr = pyre_object::w_int_new(concrete) as i64;
    let boxed = walker_box_int(ctx, op_pc, raw, concrete)?;
    ctx.trace_ctx
        .set_opref_concrete(boxed, box_int_concrete(concrete, boxed_ptr));
    let _ = (dst, dst_bank);
    Ok(Some(DispatchOutcome::SubReturn {
        result: Some(boxed),
    }))
}

/// Exact builtin `int` `UNARY_INVERT`: walk `_int_invert`.  Bool stays
/// on `invert`'s deprecation-warning slot.
pub(crate) fn try_walker_orthodox_unary_invert<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 1 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(obj) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    let admitted = unsafe {
        pyre_object::is_exact_builtin_instance(obj)
            && pyre_object::is_int(obj)
            && !pyre_object::is_bool(obj)
    };
    if !admitted {
        return Ok(None);
    }
    let x = unsafe { pyre_object::w_int_get_value(obj) };
    let type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let xa = walker_unbox_int(ctx, op_pc, r_args[0], type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, r_args[0], walker_numeric_builtin_class(obj))?;
    try_walker_orthodox_descent(
        ctx,
        op_pc,
        &[(xa, x)],
        &[],
        &[],
        dst,
        dst_bank,
        &INT_INVERT_DESCENT,
    )
}

/// Exact builtin `float` `UNARY_POSITIVE`: walk `_float_pos`.
/// Exact `int` stays residual: `_self_unaryop('pos')` is `self.int(space)`
/// and returns `self`, so a write-through here would be a new fold.
pub(crate) fn try_walker_orthodox_unary_pos<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 1 || dst_bank != 'r' {
        return Ok(None);
    }
    let Some(obj) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_exact_builtin_instance(obj) } {
        return Ok(None);
    }
    if unsafe { pyre_object::is_float(obj) } {
        let x = unsafe { pyre_object::w_float_get_value(obj) };
        let xa = walker_coerce_dispatching_operand_to_float(
            ctx, op_pc, r_args[0], obj, false, x, false,
        )?;
        return try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[],
            &[],
            &[(xa, x)],
            dst,
            dst_bank,
            &FLOAT_POS_DESCENT,
        );
    }
    Ok(None)
}

fn orthodox_list_getitem_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_getitem_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Descend `w_list_getitem_inner` the way list-setitem descends its inner.
/// Returns `Ok(None)` when the body is missing or the walk does not finish.
#[allow(clippy::too_many_arguments)]
fn try_walker_orthodox_list_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    list_op: OpRef,
    key_op: OpRef,
    list_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    sid: i64,
    index: i64,
) -> Result<Option<OpRef>, DispatchError> {
    let Some((sub_body, sym_ptr)) = orthodox_list_getitem_body_and_sym(ctx) else {
        return Ok(None);
    };
    let sym = unsafe { &*sym_ptr };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();

    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    walker_guard_exact_instance(
        ctx,
        op_pc,
        list_op,
        list_type_addr,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
    )?;
    walker_guard_fold_list_strategy(ctx, op_pc, list_op, sid)?;

    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));
    ctx.trace_ctx.set_opref_concrete(
        list_op,
        majit_ir::Value::Ref(majit_ir::GcRef(list_obj as usize)),
    );

    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    };
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "list_getitem_commit",
        "w_list_getitem_call_site",
        &[raw_index],
        &[ConcreteValue::Int(index)],
        &[list_op],
        &[ConcreteValue::Ref(list_obj)],
        &[],
    );
    match walk {
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-GETITEM-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            Ok(None)
        }
        Ok((
            DispatchOutcome::SubReturn {
                result: Some(boxed),
            },
            _,
        )) => Ok(Some(boxed)),
        Ok(_) => {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Exact `list[int]` when an inlined getitem helper declines. Walks
/// `w_list_getitem_inner`. An unfinished walk leaves the inline site on
/// its residual.
pub(crate) fn try_emit_list_int_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    let (Some(list_obj), Some(key_obj)) = (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) else {
        return Ok(None);
    };
    let list_canonical = unsafe {
        std::ptr::eq((*list_obj).ob_type, &pyre_object::pyobject::LIST_TYPE)
            && std::ptr::eq(
                (*list_obj).w_class,
                pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
            )
    };
    if !list_canonical {
        return Ok(None);
    }
    let index = unsafe {
        if !pyre_object::is_int(key_obj) || !pyre_object::is_exact_builtin_instance(key_obj) {
            return Ok(None);
        }
        pyre_object::w_int_get_value(key_obj)
    };
    if index < 0 {
        return Ok(None);
    }
    let concrete_len = unsafe { pyre_object::w_list_len(list_obj) };
    if index as usize >= concrete_len {
        return Ok(None);
    }
    let sid = if unsafe { pyre_object::w_list_uses_int_storage(list_obj) } {
        1i64
    } else if unsafe { pyre_object::w_list_uses_float_storage(list_obj) } {
        2i64
    } else if unsafe { pyre_object::w_list_uses_object_storage(list_obj) } {
        0i64
    } else {
        return Ok(None);
    };
    let Some(boxed) = try_walker_orthodox_list_getitem(
        ctx, op_pc, r_args[0], r_args[1], list_obj, key_obj, sid, index,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(DispatchOutcome::SubReturn {
        result: Some(boxed),
    }))
}

fn binary_op_tag_for_helper_name(name: &str) -> Option<i64> {
    use pyre_interpreter::bytecode::BinaryOperator as B;
    let leaf = name.rsplit([':', '.']).next().unwrap_or(name);
    let leaf = leaf.strip_suffix("_impl").unwrap_or(leaf);
    let leaf = leaf.strip_prefix("shortcut_").unwrap_or(leaf);
    let leaf = leaf.strip_prefix("descr_").unwrap_or(leaf);
    let leaf = leaf.strip_prefix("int_").unwrap_or(leaf);
    let leaf = leaf.strip_prefix("long_").unwrap_or(leaf);
    let op = match leaf {
        "add" => B::Add,
        "getitem" => B::Subscr,
        "sub" => B::Subtract,
        "mul" => B::Multiply,
        "floordiv" => B::FloorDivide,
        "mod" | "mod_" => B::Remainder,
        "truediv" => B::TrueDivide,
        "lshift" => B::Lshift,
        "rshift" => B::Rshift,
        "and" | "and_" => B::And,
        "or" | "or_" => B::Or,
        "xor" => B::Xor,
        "inplace_add" | "iadd" => B::InplaceAdd,
        "inplace_sub" | "isub" => B::InplaceSubtract,
        "inplace_mul" | "imul" => B::InplaceMultiply,
        "inplace_floordiv" | "ifloordiv" => B::InplaceFloorDivide,
        "inplace_mod" | "imod" => B::InplaceRemainder,
        "inplace_truediv" | "itruediv" => B::InplaceTrueDivide,
        "inplace_and" | "iand" => B::InplaceAnd,
        "inplace_or" | "ior" => B::InplaceOr,
        "inplace_xor" | "ixor" => B::InplaceXor,
        _ => return None,
    };
    pyre_interpreter::runtime_ops::binary_op_tag(op)
}

const COMPARE_OP_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::opcode_ops::compare_value_from_tag",
    commit_label: "compare_op_commit",
    call_site_label: "compare_op_call_site",
    decline_tag: "COMPARE-OP-SUBWALK",
};

/// Pin `lo < raw < hi` so a later out-of-range value deopts.
fn walker_guard_int_open_range<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
    concrete: i64,
    lo_exclusive: i64,
    hi_exclusive: i64,
) -> Result<(), DispatchError> {
    let lo = ctx.trace_ctx.const_int(lo_exclusive);
    let hi = ctx.trace_ctx.const_int(hi_exclusive);
    let gt_lo = ctx.trace_ctx.record_op(OpCode::IntLt, &[lo, raw]);
    ctx.trace_ctx.set_opref_concrete(
        gt_lo,
        majit_ir::Value::Int(i64::from(lo_exclusive < concrete)),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[gt_lo])?;
    let lt_hi = ctx.trace_ctx.record_op(OpCode::IntLt, &[raw, hi]);
    ctx.trace_ctx.set_opref_concrete(
        lt_hi,
        majit_ir::Value::Int(i64::from(concrete < hi_exclusive)),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[lt_hi])
}

/// Descend an operator's body whole instead of re-emitting an arm of it by
/// hand.
///
/// Upstream traces *through* `descr_pos`, `descr_add` and their siblings;
/// the guards that select the arm are the body's own.  Here they are too:
/// the class tests the operator opens with read `ob_type`, which the
/// codewriter emits as `guard_class`, and its override probes promote
/// `w_class` (`descroperation.rs try_numeric_unaryop_override`,
/// `needs_numeric_binop_dispatch`), which records a `guard_value`.  Nothing
/// about the operands is asserted at this call site, so a receiver the body
/// handles differently on the next entry side-exits at one of those guards
/// and re-runs the operator in the residual.
///
/// The caller decides *whether* to descend, and that is a policy, not a
/// guard: an operand that is not an exact builtin instance takes an override
/// arm, which calls Python, and a sub-walk that executes a call and then
/// declines has run it twice.  Until a mid-descent decline rewinds such an
/// effect, that operand goes to the residual, unguarded.
///
/// `ref_args` pairs each operand box with its concrete object; `int_args`
/// carries constant-bank operands (an operator tag) the same way.  A body
/// that raises declines (see the `SubRaise` arm).
fn try_walker_orthodox_descent<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    int_args: &[(OpRef, i64)],
    ref_args: &[(OpRef, pyre_object::PyObjectRef)],
    float_args: &[(OpRef, f64)],
    dst: usize,
    dst_bank: char,
    descent: &HelperDescent,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    try_walker_orthodox_descent_ex(
        ctx, op_pc, int_args, ref_args, float_args, dst, dst_bank, descent, None, true,
    )
}

fn try_walker_orthodox_descent_ex<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    int_args: &[(OpRef, i64)],
    ref_args: &[(OpRef, pyre_object::PyObjectRef)],
    float_args: &[(OpRef, f64)],
    dst: usize,
    dst_bank: char,
    descent: &HelperDescent,
    boxed_out: Option<&mut Option<OpRef>>,
    guard_raising_binop: bool,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    let Some(prep) = prepare_orthodox_descent(ctx, op_pc, descent) else {
        return Ok(None);
    };
    run_prepared_orthodox_descent(
        ctx,
        op_pc,
        prep,
        int_args,
        ref_args,
        float_args,
        dst,
        dst_bank,
        descent,
        boxed_out,
        guard_raising_binop,
    )
}

/// Jitcode, body, portal sym, and nested resume entry for one descent.
/// Resolved before any guard so a decline leaves the trace untouched.
struct OrthodoxDescentPrep<Sym: WalkSym> {
    body: SubJitCodeBody,
    sym_ptr: *const Sym,
    nested: HelperEntry,
}

/// Resolve every decline before recording anything. Each decline names
/// itself under `PYRE_FBW_DEBUG_ABORT` so a `consulted=1 fired=0` census
/// line can be attributed without a rebuild.
fn prepare_orthodox_descent<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    descent: &HelperDescent,
) -> Option<OrthodoxDescentPrep<Sym>> {
    let decline = |why: &str| {
        if fbw_debug_abort_enabled() {
            eprintln!("[decline-why] {}-{why} pc={op_pc}", descent.decline_tag);
        }
        None
    };
    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode_cached(descent.path) else {
        return decline("NO-JITCODE");
    };
    let Some(body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return decline("NO-SUB-BODY");
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return decline("NO-SYM");
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return decline("SYM-NO-JITCODE");
    }
    let Ok(nested) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return decline("NESTED-ENTRY");
    };
    Some(OrthodoxDescentPrep {
        body,
        sym_ptr,
        nested,
    })
}

/// Record one already-resolved helper body. `SubRaise` stays a raised
/// outcome (`fuse_kind_ctor_raise`); a caller that used to reject every
/// non-`SubReturn` maps that through [`orthodox_descent_unit`].
fn run_prepared_orthodox_descent<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    prep: OrthodoxDescentPrep<Sym>,
    int_args: &[(OpRef, i64)],
    ref_args: &[(OpRef, pyre_object::PyObjectRef)],
    float_args: &[(OpRef, f64)],
    dst: usize,
    dst_bank: char,
    descent: &HelperDescent,
    boxed_out: Option<&mut Option<OpRef>>,
    guard_raising_binop: bool,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    for &(operand, operand_obj) in ref_args {
        ctx.trace_ctx.set_opref_concrete(
            operand,
            majit_ir::Value::Ref(majit_ir::GcRef(operand_obj as usize)),
        );
    }
    let int_oprefs: Vec<OpRef> = int_args.iter().map(|&(opref, _)| opref).collect();
    let int_concretes: Vec<ConcreteValue> = int_args
        .iter()
        .map(|&(_, value)| ConcreteValue::Int(value))
        .collect();
    let ref_oprefs: Vec<OpRef> = ref_args.iter().map(|&(opref, _)| opref).collect();
    let ref_concretes: Vec<ConcreteValue> = ref_args
        .iter()
        .map(|&(_, obj)| ConcreteValue::Ref(obj))
        .collect();
    let float_oprefs: Vec<OpRef> = float_args.iter().map(|&(opref, _)| opref).collect();
    for &(operand, value) in float_args {
        ctx.trace_ctx
            .set_opref_concrete(operand, majit_ir::Value::Float(value));
    }

    // SAFETY: `prepare_orthodox_descent` rejected a null sym and a null
    // jitcode. `snapshot_sym` stays live for this full-body walk.
    let sym = unsafe { &*prep.sym_ptr };
    let exc_before_subwalk = ctx.last_exc_value();
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &prep.body,
        prep.nested,
        descent.commit_label,
        descent.call_site_label,
        &int_oprefs,
        &int_concretes,
        &ref_oprefs,
        &ref_concretes,
        &float_oprefs,
    );
    let (walk_outcome, _walk_start) = match walk {
        // The body reached a helper this build did not lower.  The arms an
        // exact builtin takes are reads that allocate at most their result,
        // so nothing is committed -- cut the tentative IR, with its
        // snapshots, and let the residual (or the fold behind this descent)
        // serve the operator.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] {} pc={pc}", descent.decline_tag);
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Ok(pair) => pair,
        Err(error) => {
            if fbw_debug_abort_enabled() {
                match &error {
                    DispatchError::UnsupportedOpname { pc, key } => {
                        eprintln!(
                            "[decline-why] {}-ERR pc={op_pc} error=UnsupportedOpname leaf_pc={pc} key={key}",
                            descent.decline_tag
                        );
                    }
                    _ => {
                        eprintln!(
                            "[decline-why] {}-ERR pc={op_pc} error={}",
                            descent.decline_tag,
                            error.variant_name()
                        );
                    }
                }
            }
            return Err(error);
        }
    };
    let result =
        match promote_published_null_return_since(ctx, walk_outcome, op_pc, exc_before_subwalk)? {
            // Void helpers (`opimpl_residual_call_*_v`) have no dst
            // register; `write_residual_call_result_to_dst` no-ops `'v'`.
            DispatchOutcome::SubReturn { result: None } if dst_bank == 'v' => {
                let _ = finish_inline_callee_return(ctx, None);
                return Ok(Some(DispatchOutcome::Continue));
            }
            DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
                .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
            // `front::result_exc::fuse_kind_ctor_raise` removes the Rust
            // `PyError` aggregate from supported literal-message raise paths and
            // materialises the interpreter's `W_BaseException` through one opaque
            // residual.  That is the exception value the ordinary inline-callee
            // machinery propagates too, so preserve the sub-walk's `SubRaise`
            // instead of rolling it back to a hand-written operator fold.
            raised @ DispatchOutcome::SubRaise { .. } => return Ok(Some(raised)),
            _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
        };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    if let Some(slot) = boxed_out {
        *slot = Some(result);
    }
    // `handle_possible_exception`: a successful descent of `//` / `%`
    // still has to carry `GUARD_NO_EXCEPTION` so a later zero divisor
    // deopts instead of dest-writing NULL into the caller's `+=` slot.
    // The first int concrete is a `BinaryOperator` tag only on
    // `binary_value_from_tag`: a descent whose ints are payloads
    // (`w_specialised_tuple_ii_new`) passes `guard_raising_binop` false.
    // COMPARE tags reuse those integers (`==` is 4, which
    // `binary_op_from_tag` reads as Remainder).
    if guard_raising_binop && descent.path == BINARY_OP_DESCENT.path {
        super::inline_call::maybe_guard_no_exception_after_raising_binop(
            ctx,
            op_pc,
            &int_concretes,
        )?;
    }
    Ok(Some(DispatchOutcome::Continue))
}

/// Descend a helper and return its result box without writing a caller
/// dst register.  `allow_void` accepts `void_return` (`SubReturn` with
/// no box) and yields `Ok(Some(None))`.  Walks through
/// [`try_walker_orthodox_descent_ex`] (`dst_bank` `'v'`); a `SubRaise`
/// declines without an extra cut — `run_prepared_orthodox_descent`
/// already returns that outcome.
fn orthodox_helper_boxed<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    int_args: &[(OpRef, i64)],
    ref_args: &[(OpRef, pyre_object::PyObjectRef)],
    descent: &HelperDescent,
    allow_void: bool,
) -> Result<Option<Option<OpRef>>, DispatchError> {
    let mut boxed = None;
    match try_walker_orthodox_descent_ex(
        ctx,
        op_pc,
        int_args,
        ref_args,
        &[],
        0,
        'v',
        descent,
        Some(&mut boxed),
        false,
    )? {
        Some(DispatchOutcome::Continue) => {
            if allow_void {
                Ok(Some(boxed))
            } else {
                match boxed {
                    Some(op) => Ok(Some(Some(op))),
                    None => Ok(None),
                }
            }
        }
        Some(DispatchOutcome::SubRaise { .. }) | None => Ok(None),
        Some(_) => Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    }
}

/// Walk `w_dict_new_kwargs` (`argument.py` `_match_signature`
/// `space.newdict(kwargs=True)`).  Returns the mapping's box; a decline
/// leaves the trace cut back to the call site.  The descent itself must
/// produce the concrete; missing it declines rather than allocating a
/// second shadow.
pub(crate) fn try_walker_orthodox_kwargs_dict_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<Option<OpRef>, DispatchError> {
    match orthodox_helper_boxed(ctx, op_pc, &[], &[], &W_DICT_NEW_KWARGS_DESCENT, false)? {
        Some(Some(dict_op)) => {
            if walker_concrete_ref_object(ctx, dict_op).is_none() {
                return Ok(None);
            }
            Ok(Some(dict_op))
        }
        Some(None) | None => Ok(None),
    }
}

/// Walk `w_dict_store` for one `_match_signature` keyword that named no
/// parameter (`space.setitem(w_kwds, w_key, w_value)`).
pub(crate) fn try_walker_orthodox_kwargs_dict_setitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    dict_obj: pyre_object::PyObjectRef,
    key_op: OpRef,
    key_obj: pyre_object::PyObjectRef,
    value_op: OpRef,
    value_obj: pyre_object::PyObjectRef,
) -> Result<Option<()>, DispatchError> {
    match orthodox_helper_boxed(
        ctx,
        op_pc,
        &[],
        &[
            (dict_op, dict_obj),
            (key_op, key_obj),
            (value_op, value_obj),
        ],
        &W_DICT_STORE_DESCENT,
        true,
    )? {
        Some(_) => Ok(Some(())),
        None => Ok(None),
    }
}

/// `SubReturn` that [`run_prepared_orthodox_descent`] already wrote to `dst`.
/// `None` is the unsupported cut; a caller that recorded guards before the
/// walk passes their position so that cut removes those guards too.
/// `SubRaise` and any other outcome are [`DispatchError::UnexpectedVoidSubReturn`],
/// the contract of the readers that used to match only `SubReturn`.
fn orthodox_descent_unit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    walked: Option<DispatchOutcome>,
    cut_guards: Option<majit_metainterp::recorder::TracePosition>,
) -> Result<Option<()>, DispatchError> {
    match walked {
        Some(DispatchOutcome::Continue) => Ok(Some(())),
        None => {
            if let Some(pos) = cut_guards {
                ctx.trace_ctx.cut_trace_with_snapshots(pos);
                ctx.trace_ctx.heap_cache_mut().reset();
            }
            Ok(None)
        }
        Some(_) => Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    }
}

/// Which `_float_*` leaf a float slot wrapper names, and how its arguments
/// are ordered.
struct FloatSlotLeaf {
    descent: &'static HelperDescent,
    /// `float_binop_rev!` calls the operator on `(args[1], args[0])`.
    reflected: bool,
    /// `cmp_dunder!` / `compare_slot` keep the original operand order.
    compare: bool,
    /// `_float_truediv` raises when the divisor is zero.
    truediv: bool,
}

fn float_slot_leaf(name: &str) -> Option<FloatSlotLeaf> {
    let (descent, reflected, compare, truediv) = match name {
        "__add__" => (&FLOAT_ADD_DESCENT, false, false, false),
        "__radd__" => (&FLOAT_ADD_DESCENT, true, false, false),
        "__sub__" => (&FLOAT_SUB_DESCENT, false, false, false),
        "__rsub__" => (&FLOAT_SUB_DESCENT, true, false, false),
        "__mul__" => (&FLOAT_MUL_DESCENT, false, false, false),
        "__rmul__" => (&FLOAT_MUL_DESCENT, true, false, false),
        "__truediv__" => (&FLOAT_TRUEDIV_DESCENT, false, false, true),
        "__rtruediv__" => (&FLOAT_TRUEDIV_DESCENT, true, false, true),
        "__lt__" => (&FLOAT_LT_DESCENT, false, true, false),
        "__le__" => (&FLOAT_LE_DESCENT, false, true, false),
        "__gt__" => (&FLOAT_GT_DESCENT, false, true, false),
        "__ge__" => (&FLOAT_GE_DESCENT, false, true, false),
        "__eq__" => (&FLOAT_EQ_DESCENT, false, true, false),
        "__ne__" => (&FLOAT_NE_DESCENT, false, true, false),
        _ => return None,
    };
    Some(FloatSlotLeaf {
        descent,
        reflected,
        compare,
        truediv,
    })
}

enum FloatSlotOperand {
    ExactFloat(f64),
    UserFloat(f64),
    ExactInt(i64),
    Bool(i64),
}

/// Payload `as_float` would read for an operand the float slot admits.
///
/// Exact float, user-layout float (`FLOAT_USER_TYPE`, shared by every float
/// subclass), exact machine int, and bool. A long, an int subclass, and
/// anything else stay `None`: `add_builtin` would not reach `_float_*` for a
/// pair of ints, and a long goes through `jit_bigint_to_f64_or_inf`.
fn classify_float_slot_operand(obj: pyre_object::PyObjectRef) -> Option<FloatSlotOperand> {
    if obj.is_null() {
        return None;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Some(FloatSlotOperand::ExactInt(
            pyre_object::tagged_int::untag_int(obj),
        ));
    }
    unsafe {
        if pyre_object::is_long(obj) {
            return None;
        }
        let ob_type = (*obj).ob_type;
        if std::ptr::eq(ob_type, &pyre_object::pyobject::FLOAT_TYPE) {
            return Some(FloatSlotOperand::ExactFloat(
                pyre_object::w_float_get_value(obj),
            ));
        }
        if std::ptr::eq(ob_type, &pyre_object::pyobject::FLOAT_USER_TYPE) {
            return Some(FloatSlotOperand::UserFloat(pyre_object::w_float_get_value(
                obj,
            )));
        }
        if std::ptr::eq(ob_type, &pyre_object::pyobject::BOOL_TYPE) {
            return Some(FloatSlotOperand::Bool(pyre_object::w_int_get_value(obj)));
        }
        if std::ptr::eq(ob_type, &pyre_object::pyobject::INT_TYPE) {
            return Some(FloatSlotOperand::ExactInt(pyre_object::w_int_get_value(
                obj,
            )));
        }
    }
    None
}

fn float_slot_operand_is_float(kind: &FloatSlotOperand) -> bool {
    matches!(
        kind,
        FloatSlotOperand::ExactFloat(_) | FloatSlotOperand::UserFloat(_)
    )
}

fn float_slot_operand_f64(kind: &FloatSlotOperand) -> f64 {
    match *kind {
        FloatSlotOperand::ExactFloat(value) | FloatSlotOperand::UserFloat(value) => value,
        FloatSlotOperand::ExactInt(value) | FloatSlotOperand::Bool(value) => value as f64,
    }
}

fn float_slot_int_widens(kind: &FloatSlotOperand) -> bool {
    match *kind {
        FloatSlotOperand::ExactInt(value) | FloatSlotOperand::Bool(value) => {
            !int_is_exact_as_float(value)
        }
        _ => false,
    }
}

/// Unbox a slot operand the way `as_float` reads it. No `w_class` pin: the
/// slot was already selected by the call. A user float guards `FLOAT_USER_TYPE`
/// and reads [`crate::descr::float_user_floatval_descr`], so a virtual built
/// by [`try_walker_inline_float_subclass_new`] folds. An exact int or bool
/// casts after the unbox; a compare also guards [`int_is_exact_as_float`].
fn unbox_float_slot_operand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete: pyre_object::PyObjectRef,
    kind: &FloatSlotOperand,
    compare: bool,
) -> Result<OpRef, DispatchError> {
    let raw = match *kind {
        FloatSlotOperand::ExactFloat(value) => {
            let type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
            let raw = walker_unbox_float(ctx, op_pc, obj, type_addr)?;
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(value));
            raw
        }
        FloatSlotOperand::UserFloat(value) => {
            let type_addr = &pyre_object::pyobject::FLOAT_USER_TYPE as *const _ as i64;
            if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
                let type_const = ctx.trace_ctx.const_int(type_addr);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardClass,
                    &[obj, type_const],
                )?;
                ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
            }
            let raw = crate::trace_unbox_float(
                ctx.trace_ctx,
                obj,
                type_addr,
                crate::descr::float_user_floatval_descr(),
            );
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(value));
            raw
        }
        FloatSlotOperand::ExactInt(value) | FloatSlotOperand::Bool(value) => {
            let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete);
            let raw_int = walker_unbox_int_typed(ctx, op_pc, obj, type_addr, descr)?;
            if compare {
                walker_guard_int_exact_as_float(ctx, op_pc, raw_int, value)?;
            }
            let raw = ctx.trace_ctx.record_op(OpCode::CastIntToFloat, &[raw_int]);
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Float(value as f64));
            raw
        }
    };
    Ok(raw)
}

fn concrete_is_exact_slot_wrapper(obj: pyre_object::PyObjectRef) -> bool {
    !obj.is_null() && unsafe { std::ptr::eq((*obj).ob_type, &pyre_interpreter::SLOT_WRAPPER_TYPE) }
}

fn call_self_slot_is_populated<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    slot: OpRef,
) -> bool {
    walker_concrete_ref_object(ctx, slot)
        .is_some_and(|null_or_self| !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL)
}

/// Inline a call of float's published arithmetic or compare slot wrapper.
///
/// The call shape is `[wrapper, null, arg0, arg1]` (`bh_call_fn_2`). Identity
/// is pointer equality with `lookup_in_type` of `float`'s type object, so
/// `int.__add__` (also a slot wrapper) stays on the residual. The body is
/// the existing `_float_*` descent: coercion is the unbox above, then
/// [`try_walker_orthodox_descent`]. A reflected arithmetic name swaps the
/// coerced values after the guards. Compare keeps `compare_slot`'s order
/// and declines an int `int_is_exact_as_float` rejects, before any IR.
pub(crate) fn try_walker_inline_float_slot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 4 {
        return Ok(None);
    }
    if call_self_slot_is_populated(ctx, r_args[1]) {
        return Ok(None);
    }
    let Some(callable) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !concrete_is_exact_slot_wrapper(callable) {
        return Ok(None);
    }
    let name = unsafe { pyre_interpreter::function_get_name(callable) };
    let Some(leaf) = float_slot_leaf(name) else {
        return Ok(None);
    };
    let float_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::FLOAT_TYPE);
    if float_type.is_null() {
        return Ok(None);
    }
    if unsafe { pyre_interpreter::lookup_in_type(float_type, name) } != Some(callable) {
        return Ok(None);
    }
    let decline = |why: &str| {
        if fbw_inline_diag_enabled() {
            eprintln!("[float-slot] pc={} name={name} why={why}", op.pc);
        }
        Ok(None)
    };
    let Some(left) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return decline("left operand is not concrete");
    };
    let Some(right) = walker_concrete_ref_object(ctx, r_args[3]) else {
        return decline("right operand is not concrete");
    };
    let Some(left_kind) = classify_float_slot_operand(left) else {
        return decline("left operand is not admitted");
    };
    let Some(right_kind) = classify_float_slot_operand(right) else {
        return decline("right operand is not admitted");
    };
    if !float_slot_operand_is_float(&left_kind) && !float_slot_operand_is_float(&right_kind) {
        return decline("neither operand is a float");
    }
    let x = float_slot_operand_f64(&left_kind);
    let y = float_slot_operand_f64(&right_kind);
    let host_y = if leaf.reflected && !leaf.compare {
        x
    } else {
        y
    };
    if leaf.truediv && host_y == 0.0 {
        return decline("division by zero");
    }
    if leaf.compare && (float_slot_int_widens(&left_kind) || float_slot_int_widens(&right_kind)) {
        return decline("int is not exact as float");
    }

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let xa = unbox_float_slot_operand(ctx, op.pc, r_args[2], left, &left_kind, leaf.compare)?;
    let ya = unbox_float_slot_operand(ctx, op.pc, r_args[3], right, &right_kind, leaf.compare)?;
    let (xa, ya, x, y) = if leaf.reflected && !leaf.compare {
        (ya, xa, y, x)
    } else {
        (xa, ya, x, y)
    };
    let outcome = try_walker_orthodox_descent(
        ctx,
        op.pc,
        &[],
        &[],
        &[(xa, x), (ya, y)],
        dst,
        dst_bank,
        leaf.descent,
    )?;
    if outcome.is_none() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return decline("descent declined");
    }
    if fbw_inline_diag_enabled() {
        eprintln!("[float-slot] pc={} name={name}", op.pc);
    }
    Ok(outcome.map(|outcome| (outcome, op.next_pc)))
}

/// Exact float, or exact machine int, as `builtin_float` boxes it without
/// calling `__float__` or `__index__`. Bool, long, and subclasses are `None`.
fn float_subclass_new_argument(arg: pyre_object::PyObjectRef) -> Option<(bool, f64)> {
    if arg.is_null()
        || (pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(arg))
    {
        return None;
    }
    unsafe {
        let ob_type = (*arg).ob_type;
        if pyre_object::is_exact_type(arg, &pyre_object::pyobject::FLOAT_TYPE)
            && std::ptr::eq(ob_type, &pyre_object::pyobject::FLOAT_TYPE)
        {
            return Some((false, pyre_object::w_float_get_value(arg)));
        }
        if pyre_object::is_exact_type(arg, &pyre_object::pyobject::INT_TYPE)
            && std::ptr::eq(ob_type, &pyre_object::pyobject::INT_TYPE)
            && pyre_object::is_int(arg)
            && !pyre_object::is_bool(arg)
            && !pyre_object::is_long(arg)
        {
            return Some((true, pyre_object::w_int_get_value(arg) as f64));
        }
    }
    None
}

fn same_layout_typedef(a: pyre_object::PyObjectRef, b: pyre_object::PyObjectRef) -> bool {
    let typedef_of = |w: pyre_object::PyObjectRef| {
        let layout = unsafe { pyre_object::typeobject::w_type_get_layout_ptr(w) };
        if layout.is_null() {
            std::ptr::null()
        } else {
            unsafe { (*layout).typedef }
        }
    };
    std::ptr::eq(typedef_of(a), typedef_of(b))
}

/// `user_setup` → `_mapdict_init_empty(w_subtype.terminator)` (`mapdict.py`).
///
/// `emit_walker_instance` bakes the type's terminator as the fresh
/// instance's `map`. A zero here is the deferred-init state
/// `alloc_instance_object` refuses: the next `getfield` of `map` after
/// first attribute access sees the terminator in memory against a
/// heapcache word of 0 (`_opimpl_getfield_gc_any_pureornot`).
///
/// `concrete` is the nursery instance the fold allocated. Store the same
/// terminator word there: `tag_subclass_instance` already did, but a
/// collection between that store and this emit would otherwise leave the
/// live `map` at 0 while the heapcache holds the terminator.
fn walker_emit_user_mapdict_empty<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    new_op: OpRef,
    map_descr: majit_ir::DescrRef,
    storage_descr: majit_ir::DescrRef,
    terminator: *const u8,
    concrete: pyre_object::PyObjectRef,
) {
    let terminator_const = ctx.trace_ctx.const_int(terminator as i64);
    let map_idx = map_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, terminator_const],
        map_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(new_op, map_idx, terminator_const);
    let storage = ctx.trace_ctx.const_null();
    let storage_idx = storage_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new_op, storage],
        storage_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(new_op, storage_idx, storage);
    if !concrete.is_null() {
        ctx.trace_ctx.field_store(
            concrete as i64,
            &map_descr,
            majit_ir::Value::Int(terminator as i64),
        );
        ctx.trace_ctx.field_store(
            concrete as i64,
            &storage_descr,
            majit_ir::Value::Ref(majit_ir::GcRef(0)),
        );
    }
}

/// `MyFloat(x)` for a `float` subclass whose `__new__` is float's and whose
/// `__init__` is object's.
///
/// `float_descr_new` allocates with `w_float_subclass_new` and tags
/// `w_class` via `tag_subclass_instance`. `object_descr_init` returns None
/// for that pair: surplus arguments are accepted once `__new__` is not
/// object's. The trace is `NewWithVtable` of `W_FloatObjectUser`, the
/// user-layout `floatval`, then `w_class` / terminator `map` / empty
/// `storage`. The argument is pinned with
/// [`walker_coerce_dispatching_operand_to_float`] because `builtin_float`
/// dispatches `__float__` / `__index__` on a subclass.
pub(crate) fn try_walker_inline_float_subclass_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 3 {
        return Ok(None);
    }
    if call_self_slot_is_populated(ctx, r_args[1]) {
        return Ok(None);
    }
    let Some(cls) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_type(cls) } {
        return Ok(None);
    }
    let float_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::FLOAT_TYPE);
    let w_object = pyre_interpreter::typedef::w_object();
    let w_metatype = pyre_interpreter::typedef::w_type();
    if float_type.is_null() || w_object.is_null() || w_metatype.is_null() {
        return Ok(None);
    }
    if std::ptr::eq(cls, float_type) {
        return Ok(None);
    }
    if !std::ptr::eq(unsafe { (*cls).w_class }, w_metatype) {
        return Ok(None);
    }
    if unsafe { pyre_object::typeobject::w_type_get_version_tag(cls) } == 0 {
        return Ok(None);
    }
    if unsafe {
        pyre_object::w_type_disallows_instantiation(cls)
            || pyre_object::w_type_is_abstract(cls)
            || pyre_object::typeobject::w_type_has_vectorcall(cls)
            || pyre_object::typeobject::w_type_get_hasuserdel(cls)
    } {
        return Ok(None);
    }
    if !unsafe { pyre_object::typeobject::w_type_issubtype(cls, float_type) } {
        return Ok(None);
    }
    if !same_layout_typedef(float_type, cls) {
        return Ok(None);
    }
    let tp_new = unsafe { pyre_interpreter::lookup_in_type(cls, "__new__") };
    let float_new = unsafe { pyre_interpreter::lookup_in_type(float_type, "__new__") };
    if tp_new.is_none() || tp_new != float_new {
        return Ok(None);
    }
    let tp_init = unsafe { pyre_interpreter::lookup_in_type(cls, "__init__") };
    let obj_init = unsafe { pyre_interpreter::lookup_in_type(w_object, "__init__") };
    if tp_init.is_none() || tp_init != obj_init {
        return Ok(None);
    }
    if unsafe {
        type_attr_is_cell_backed(cls, "__new__") || type_attr_is_cell_backed(cls, "__init__")
    } {
        return Ok(None);
    }
    let terminator =
        unsafe { pyre_interpreter::objspace::std::mapdict::ensure_type_terminator(cls) };
    if terminator.is_null() {
        return Ok(None);
    }
    let Some(arg) = walker_concrete_ref_object(ctx, r_args[2]) else {
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[float-subclass-new] pc={} why=argument is not concrete",
                op.pc
            );
        }
        return Ok(None);
    };
    let Some((is_int, val)) = float_subclass_new_argument(arg) else {
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[float-subclass-new] pc={} why=argument is not exact",
                op.pc
            );
        }
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let type_const = walker_guard_stamped_type_version(ctx, op.pc, r_args[0], cls)?;
    let raw =
        walker_coerce_dispatching_operand_to_float(ctx, op.pc, r_args[2], arg, is_int, val, false)?;

    let concrete = pyre_object::w_float_subclass_new(val);
    if concrete.is_null() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    let concrete = pyre_interpreter::typedef::tag_subclass_instance(concrete, cls);
    // Nursery-born (`w_float_subclass_new`). `emit_box_float_inline`
    // records a collecting `NewWithVtable`; reload after
    // (`shadowstack.py expand_pop_roots`).
    let _roots = pyre_object::gc_roots::push_roots();
    let concrete_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(concrete);
    let new_op = crate::helpers::emit_box_float_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_float_user_size_descr(),
        crate::descr::float_user_floatval_descr(),
    );
    let class_descr = crate::descr::w_class_descr();
    let class_idx = class_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new_op, type_const], class_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new_op, class_idx, type_const);
    let concrete = pyre_object::gc_roots::shadow_stack_get(concrete_slot);
    walker_emit_user_mapdict_empty(
        ctx,
        new_op,
        crate::descr::float_user_map_descr(),
        crate::descr::float_user_storage_descr(),
        terminator,
        concrete,
    );
    let concrete = pyre_object::gc_roots::shadow_stack_get(concrete_slot);
    ctx.trace_ctx.set_opref_concrete(
        new_op,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    if fbw_inline_diag_enabled() {
        eprintln!("[float-subclass-new] pc={} class={}", op.pc, unsafe {
            pyre_object::w_type_get_name(cls)
        });
    }
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

enum IntSlotOperand {
    ExactInt(i64),
    UserInt(i64),
    Bool(i64),
}

fn int_slot_operand_value(kind: &IntSlotOperand) -> i64 {
    match *kind {
        IntSlotOperand::ExactInt(value)
        | IntSlotOperand::UserInt(value)
        | IntSlotOperand::Bool(value) => value,
    }
}

/// Payload `int_value` would read for an operand `add_builtin` sends to
/// `int_add`. Exact int, user-layout int (`INT_USER_TYPE`, shared by every
/// int subclass), and bool. A long goes through `long_add`; anything else
/// makes `int_dunder_add` return NotImplemented.
fn classify_int_slot_operand(obj: pyre_object::PyObjectRef) -> Option<IntSlotOperand> {
    if obj.is_null() {
        return None;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Some(IntSlotOperand::ExactInt(
            pyre_object::tagged_int::untag_int(obj),
        ));
    }
    unsafe {
        if pyre_object::is_long(obj) {
            return None;
        }
        let ob_type = (*obj).ob_type;
        if std::ptr::eq(ob_type, &pyre_object::pyobject::INT_TYPE) {
            return Some(IntSlotOperand::ExactInt(pyre_object::w_int_get_value(obj)));
        }
        if std::ptr::eq(ob_type, &pyre_object::pyobject::INT_USER_TYPE) {
            return Some(IntSlotOperand::UserInt(pyre_object::w_int_get_value(obj)));
        }
        if std::ptr::eq(ob_type, &pyre_object::pyobject::BOOL_TYPE) {
            return Some(IntSlotOperand::Bool(pyre_object::w_int_get_value(obj)));
        }
    }
    None
}

/// Unbox a slot operand the way `int_value` reads it. No `w_class` pin: the
/// slot was already selected by the call. A user int guards `INT_USER_TYPE`
/// and reads [`crate::descr::int_user_intval_descr`], so a virtual built by
/// [`try_walker_inline_int_subclass_new`] folds.
fn unbox_int_slot_operand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    concrete: pyre_object::PyObjectRef,
    kind: &IntSlotOperand,
) -> Result<OpRef, DispatchError> {
    let raw = match *kind {
        IntSlotOperand::ExactInt(value) | IntSlotOperand::Bool(value) => {
            let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(concrete);
            let raw = walker_unbox_int_typed(ctx, op_pc, obj, type_addr, descr)?;
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(value));
            raw
        }
        IntSlotOperand::UserInt(value) => {
            let type_addr = &pyre_object::pyobject::INT_USER_TYPE as *const _ as i64;
            if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
                let type_const = ctx.trace_ctx.const_int(type_addr);
                walker_emit_guard_with_snapshot(
                    ctx,
                    op_pc,
                    OpCode::GuardClass,
                    &[obj, type_const],
                )?;
                ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
            }
            let raw = crate::trace_unbox_int(
                ctx.trace_ctx,
                obj,
                type_addr,
                crate::descr::int_user_intval_descr(),
            );
            ctx.trace_ctx
                .set_opref_concrete(raw, majit_ir::Value::Int(value));
            raw
        }
    };
    Ok(raw)
}

/// Inline a call of int's published arithmetic slot wrapper.
///
/// `__add__` / `__radd__` descend [`INT_ADD_DESCENT`], `__sub__` / `__rsub__`
/// descend [`INT_SUB_DESCENT`], `__mul__` / `__rmul__` descend
/// [`INT_MUL_DESCENT`]. The call shape is `[wrapper, null, arg0, arg1]`
/// (`bh_call_fn_2`). Identity is pointer equality with `lookup_in_type` of
/// `int`'s type object, so `float.__add__` stays on the float slot tried
/// first. `int_binop_rev` swaps before `sub_builtin` / `add_builtin` /
/// `mul_builtin` (`descr_rbinop`'s `op(y, x)`), and an overflowing
/// `checked_*` is left on the residual (`_int_sub_ovf` / `_int_add_ovf` /
/// `_int_mul_ovf`).
pub(crate) fn try_walker_inline_int_slot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 4 {
        return Ok(None);
    }
    if call_self_slot_is_populated(ctx, r_args[1]) {
        return Ok(None);
    }
    let Some(callable) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !concrete_is_exact_slot_wrapper(callable) {
        return Ok(None);
    }
    let name = unsafe { pyre_interpreter::function_get_name(callable) };
    // `descr_rbinop` evaluates `op(y, x)` after the swap below.
    #[derive(Clone, Copy)]
    enum Op {
        Add,
        Sub,
        Mul,
    }
    let (arith, reflected) = match name {
        "__add__" => (Op::Add, false),
        "__radd__" => (Op::Add, true),
        "__sub__" => (Op::Sub, false),
        "__rsub__" => (Op::Sub, true),
        "__mul__" => (Op::Mul, false),
        "__rmul__" => (Op::Mul, true),
        _ => return Ok(None),
    };
    let descent = match arith {
        Op::Add => &INT_ADD_DESCENT,
        Op::Sub => &INT_SUB_DESCENT,
        Op::Mul => &INT_MUL_DESCENT,
    };
    let int_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::INT_TYPE);
    if int_type.is_null() {
        return Ok(None);
    }
    if unsafe { pyre_interpreter::lookup_in_type(int_type, name) } != Some(callable) {
        return Ok(None);
    }
    let decline = |why: &str| {
        if fbw_inline_diag_enabled() {
            eprintln!("[int-slot] pc={} name={name} why={why}", op.pc);
        }
        Ok(None)
    };
    let Some(left) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return decline("left operand is not concrete");
    };
    let Some(right) = walker_concrete_ref_object(ctx, r_args[3]) else {
        return decline("right operand is not concrete");
    };
    let Some(left_kind) = classify_int_slot_operand(left) else {
        return decline("left operand is not admitted");
    };
    let Some(right_kind) = classify_int_slot_operand(right) else {
        return decline("right operand is not admitted");
    };
    let x = int_slot_operand_value(&left_kind);
    let y = int_slot_operand_value(&right_kind);
    let (x, y) = if reflected { (y, x) } else { (x, y) };
    let overflows = match arith {
        Op::Add => x.checked_add(y).is_none(),
        Op::Sub => x.checked_sub(y).is_none(),
        Op::Mul => x.checked_mul(y).is_none(),
    };
    if overflows {
        return decline("overflow");
    }

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let xa = unbox_int_slot_operand(ctx, op.pc, r_args[2], left, &left_kind)?;
    let ya = unbox_int_slot_operand(ctx, op.pc, r_args[3], right, &right_kind)?;
    let (xa, ya) = if reflected { (ya, xa) } else { (xa, ya) };
    let outcome = try_walker_orthodox_descent(
        ctx,
        op.pc,
        &[(xa, x), (ya, y)],
        &[],
        &[],
        dst,
        dst_bank,
        descent,
    )?;
    if outcome.is_none() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return decline("descent declined");
    }
    if fbw_inline_diag_enabled() {
        eprintln!("[int-slot] pc={} name={name}", op.pc);
    }
    Ok(outcome.map(|outcome| (outcome, op.next_pc)))
}

/// Exact machine int, as `int()` boxes it without calling `__int__` or
/// `__index__`. Bool, long, tagged immediates, and subclasses are `None`.
fn int_subclass_new_argument(arg: pyre_object::PyObjectRef) -> Option<i64> {
    if arg.is_null()
        || (pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(arg))
    {
        return None;
    }
    unsafe {
        if pyre_object::is_exact_type(arg, &pyre_object::pyobject::INT_TYPE)
            && std::ptr::eq((*arg).ob_type, &pyre_object::pyobject::INT_TYPE)
            && pyre_object::is_int(arg)
            && !pyre_object::is_bool(arg)
            && !pyre_object::is_long(arg)
        {
            return Some(pyre_object::w_int_get_value(arg));
        }
    }
    None
}

/// `MyInt(n)` for an `int` subclass whose `__new__` is int's and whose
/// `__init__` is object's.
///
/// `int_descr_new` allocates with `w_int_subclass_new` and tags `w_class`
/// via `tag_subclass_instance`. `object_descr_init` returns None for that
/// pair: surplus arguments are accepted once `__new__` is not object's. The
/// trace is `NewWithVtable` of `W_IntObjectUser`, the user-layout `intval`,
/// then `w_class` / terminator `map` / empty `storage`. The argument is an
/// exact machine int with its `w_class` pinned,
/// because `builtin_int` dispatches `__int__` / `__index__` on a subclass
/// argument.
pub(crate) fn try_walker_inline_int_subclass_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 3 {
        return Ok(None);
    }
    if call_self_slot_is_populated(ctx, r_args[1]) {
        return Ok(None);
    }
    let Some(cls) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_type(cls) } {
        return Ok(None);
    }
    let int_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::INT_TYPE);
    let w_object = pyre_interpreter::typedef::w_object();
    let w_metatype = pyre_interpreter::typedef::w_type();
    if int_type.is_null() || w_object.is_null() || w_metatype.is_null() {
        return Ok(None);
    }
    if std::ptr::eq(cls, int_type) {
        return Ok(None);
    }
    if !std::ptr::eq(unsafe { (*cls).w_class }, w_metatype) {
        return Ok(None);
    }
    if unsafe { pyre_object::typeobject::w_type_get_version_tag(cls) } == 0 {
        return Ok(None);
    }
    if unsafe {
        pyre_object::w_type_disallows_instantiation(cls)
            || pyre_object::w_type_is_abstract(cls)
            || pyre_object::typeobject::w_type_has_vectorcall(cls)
            || pyre_object::typeobject::w_type_get_hasuserdel(cls)
    } {
        return Ok(None);
    }
    if !unsafe { pyre_object::typeobject::w_type_issubtype(cls, int_type) } {
        return Ok(None);
    }
    if !same_layout_typedef(int_type, cls) {
        return Ok(None);
    }
    let tp_new = unsafe { pyre_interpreter::lookup_in_type(cls, "__new__") };
    let int_new = unsafe { pyre_interpreter::lookup_in_type(int_type, "__new__") };
    if tp_new.is_none() || tp_new != int_new {
        return Ok(None);
    }
    let tp_init = unsafe { pyre_interpreter::lookup_in_type(cls, "__init__") };
    let obj_init = unsafe { pyre_interpreter::lookup_in_type(w_object, "__init__") };
    if tp_init.is_none() || tp_init != obj_init {
        return Ok(None);
    }
    if unsafe {
        type_attr_is_cell_backed(cls, "__new__") || type_attr_is_cell_backed(cls, "__init__")
    } {
        return Ok(None);
    }
    let terminator =
        unsafe { pyre_interpreter::objspace::std::mapdict::ensure_type_terminator(cls) };
    if terminator.is_null() {
        return Ok(None);
    }
    let Some(arg) = walker_concrete_ref_object(ctx, r_args[2]) else {
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[int-subclass-new] pc={} why=argument is not concrete",
                op.pc
            );
        }
        return Ok(None);
    };
    let Some(val) = int_subclass_new_argument(arg) else {
        if fbw_inline_diag_enabled() {
            eprintln!("[int-subclass-new] pc={} why=argument is not exact", op.pc);
        }
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let type_const = walker_guard_stamped_type_version(ctx, op.pc, r_args[0], cls)?;
    let type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        r_args[2],
        type_addr,
        crate::descr::int_intval_descr(),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw, majit_ir::Value::Int(val));
    walker_guard_exact_w_class(ctx, op.pc, r_args[2], walker_numeric_builtin_class(arg))?;

    let concrete = pyre_object::w_int_subclass_new(val);
    if concrete.is_null() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    let concrete = pyre_interpreter::typedef::tag_subclass_instance(concrete, cls);
    // Nursery-born (`w_int_subclass_new`). `emit_box_int_inline` records
    // a collecting `NewWithVtable`; reload after
    // (`shadowstack.py expand_pop_roots`).
    let _roots = pyre_object::gc_roots::push_roots();
    let concrete_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(concrete);
    let new_op = crate::helpers::emit_box_int_inline(
        ctx.trace_ctx,
        raw,
        crate::descr::w_int_user_size_descr(),
        crate::descr::int_user_intval_descr(),
    );
    let class_descr = crate::descr::w_class_descr();
    let class_idx = class_descr.index();
    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetfieldGc, &[new_op, type_const], class_descr);
    ctx.trace_ctx
        .heapcache_setfield_cached(new_op, class_idx, type_const);
    let concrete = pyre_object::gc_roots::shadow_stack_get(concrete_slot);
    walker_emit_user_mapdict_empty(
        ctx,
        new_op,
        crate::descr::int_user_map_descr(),
        crate::descr::int_user_storage_descr(),
        terminator,
        concrete,
    );
    let concrete = pyre_object::gc_roots::shadow_stack_get(concrete_slot);
    ctx.trace_ctx.set_opref_concrete(
        new_op,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    if fbw_inline_diag_enabled() {
        eprintln!("[int-subclass-new] pc={} class={}", op.pc, unsafe {
            pyre_object::w_type_get_name(cls)
        });
    }
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Exact int `+` inside a binop-rewind inline. `binary_value_from_tag`
/// reaches `int_add`, and that body's collector allocation is a residual
/// the rewind refuses. Unbox the two exact builtins (their `w_class` may
/// be pinned: they are not the slot operands) and descend `_int_add`.
fn try_walker_rewind_int_add<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operands: &[(OpRef, pyre_object::PyObjectRef); 2],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    let x = unsafe { pyre_object::w_int_get_value(operands[0].1) };
    let y = unsafe { pyre_object::w_int_get_value(operands[1].1) };
    if x.checked_add(y).is_none() {
        return Ok(None);
    }
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let unbox = |ctx: &mut WalkContext<'_, '_, Sym>,
                 slot: (OpRef, pyre_object::PyObjectRef),
                 value: i64|
     -> Result<OpRef, DispatchError> {
        let (type_addr, descr) = crate::state::int_or_bool_unbox_type_descr(slot.1);
        let raw = walker_unbox_int_typed(ctx, op_pc, slot.0, type_addr, descr)?;
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(value));
        walker_guard_exact_w_class(ctx, op_pc, slot.0, walker_numeric_builtin_class(slot.1))?;
        Ok(raw)
    };
    let xa = unbox(ctx, operands[0], x)?;
    let ya = unbox(ctx, operands[1], y)?;
    let outcome = try_walker_orthodox_descent(
        ctx,
        op_pc,
        &[(xa, x), (ya, y)],
        &[],
        &[],
        dst,
        dst_bank,
        &INT_ADD_DESCENT,
    )?;
    if outcome.is_none() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    Ok(outcome)
}

/// `a OP b`: descend `binary_value_from_tag` with the operator tag as a
/// constant.  See [`try_walker_orthodox_descent`].
///
/// Policy: both operands must be concrete exact builtin machine ints
/// (int, bool) or exact builtin floats.  Exactness keeps the override
/// arms, which call Python, out of the sub-walk; the numeric restriction
/// keeps out the sequence arms, whose in-place forms (`list += list`)
/// mutate the receiver before any later decline could rewind them.
///
/// An in-place tag is descended as its plain operator.  The body routes
/// tags 13..=24 through the residual `binary_value` (the `__iadd__` probe
/// comes first there), which the sub-walk cannot enter -- every `t += x`
/// declined, 129 cuts in one `synth/trace_segmenting_over_limit_retry` run.
/// An exact builtin numeric has no in-place special, so `binary_value`
/// reaches the same `add`/`sub`/... arm the plain tag selects directly.
pub(crate) fn try_walker_orthodox_binary_op<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    tag: OpRef,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    // `//` and `%` descend since `int_floordiv` / `int_mod` compute the
    // floor result through the `#[oopspec("int.py_div")]` /
    // `int.py_mod` twins of rint.py's `ll_int_py_div` / `ll_int_py_mod`:
    // the body records the same elidable call the fold did, so the
    // optimizer's constant-divisor strength reduction applies to both.
    // Before that the body's `/` / `%` lowered to the truncating
    // `_ll_2_int_*` residual plus the sign-correction branch, measured
    // 63 → 79 ops on `i // (i % 3)`.
    //
    // `**`: a long operand descends `binary_value_from_tag`. Float and mixed
    // `**` descend floatobject.py `descr_pow` -> `_pow` (`float_pow_impl`);
    // a raising pair (`float_pow_would_raise`) is not descended. Machine
    // int `**` int is not descended.
    use pyre_interpreter::bytecode::BinaryOperator as B;
    let plain = match pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) {
        Some(B::Add | B::InplaceAdd) => B::Add,
        Some(B::Subtract | B::InplaceSubtract) => B::Subtract,
        Some(B::Multiply | B::InplaceMultiply) => B::Multiply,
        Some(B::FloorDivide | B::InplaceFloorDivide) => B::FloorDivide,
        Some(B::Remainder | B::InplaceRemainder) => B::Remainder,
        // intobject.py `_truediv`: unboxed zero/mantissa guards then
        // `newfloat(float(x)/float(y))`.  Wide ints raise into residual
        // `int_truediv_ovf2long`, so the helper walk no longer records
        // `rbigint.truediv`.
        Some(B::TrueDivide | B::InplaceTrueDivide) => B::TrueDivide,
        Some(B::Power | B::InplacePower) => B::Power,
        Some(B::Lshift | B::InplaceLshift) => B::Lshift,
        Some(B::Rshift | B::InplaceRshift) => B::Rshift,
        Some(B::And | B::InplaceAnd) => B::And,
        Some(B::Or | B::InplaceOr) => B::Or,
        Some(B::Xor | B::InplaceXor) => B::Xor,
        _ => return Ok(None),
    };
    let Some(plain_tag) = pyre_interpreter::runtime_ops::binary_op_tag(plain) else {
        return Ok(None);
    };
    let mut operands = [(OpRef::NONE, std::ptr::null_mut()); 2];
    for (slot, &operand) in operands.iter_mut().zip(r_args) {
        let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
            return Ok(None);
        };
        // SAFETY: `obj` is a live concrete `PyObjectRef` from the walker
        // shadow.
        //
        // Exact builtin float is admitted: `_float_*` is the unboxed
        // descr_* leaf (`float_*` + in-graph `new_with_vtable`), so the
        // walk no longer hits the synthetic `w_float_new` constructor.
        // Exact builtin long is admitted on the same descent: its arms are
        // `W_LongObject.descr_*` over rbigint.
        let admitted = unsafe {
            pyre_object::is_exact_builtin_instance(obj)
                && (pyre_object::is_int(obj)
                    || pyre_object::is_bool(obj)
                    || pyre_object::is_float(obj)
                    || pyre_object::is_long(obj))
        };
        if !admitted {
            return Ok(None);
        }
        *slot = (operand, obj);
    }
    let lhs_is_float = unsafe { pyre_object::is_float(operands[0].1) };
    let rhs_is_float = unsafe { pyre_object::is_float(operands[1].1) };
    let any_long =
        unsafe { pyre_object::is_long(operands[0].1) || pyre_object::is_long(operands[1].1) };
    // A variable machine-int exponent still declines inside `int_pow_nomod`
    // (its `Option<i64>` result is not a word-ABI residual). Machine int
    // `**` int stays residual. Long `**` continues into
    // `binary_value_from_tag`. Float and mixed `**` continue into
    // `float_pow_impl` below.
    if matches!(plain, B::Power) && !any_long && !lhs_is_float && !rhs_is_float {
        return Ok(None);
    }
    let all_int = !any_long
        && !lhs_is_float
        && !rhs_is_float
        && unsafe {
            (pyre_object::is_int(operands[0].1) || pyre_object::is_bool(operands[0].1))
                && (pyre_object::is_int(operands[1].1) || pyre_object::is_bool(operands[1].1))
        };
    // A live zero divisor is the raising arm (`try_walker_specialize_binary_op_int_zero_div`
    // for an exact int, and the float TrueDivide raise).
    // Descending the success body would dest-write NULL (`sdiv`/`None`) and
    // compile `checksum +=` against an unbound local.
    if all_int
        && matches!(plain, B::FloorDivide | B::Remainder | B::TrueDivide)
        && unsafe { pyre_object::w_int_get_value(operands[1].1) } == 0
    {
        return Ok(None);
    }
    // A binop-rewind inline refuses `w_int_gc_alloc`. Exact int `+` in that
    // region descends `_int_add` (`malloc_typed_managed`) instead of
    // `binary_value_from_tag`.
    if all_int && matches!(plain, B::Add) && ctx.session.borrow().binop_rewind_depth > 0 {
        return try_walker_rewind_int_add(ctx, op_pc, &operands, dst, dst_bank);
    }
    // floatobject.py `descr_{add,sub,mul,div}` after `_to_float`: coerce
    // each operand (float unbox, or int/bool `cast_int_to_float`), then
    // the unboxed leaf whose graph is `float_*` + `new_with_vtable`.
    // Mixed int/float is the same leaf (`float_loop` `i * 0.1`,
    // `spectral_norm` `v[j] / int`).
    if !any_long && !all_int {
        let Some(descent) = (match plain {
            B::Add => Some(&FLOAT_ADD_DESCENT),
            B::Subtract => Some(&FLOAT_SUB_DESCENT),
            B::Multiply => Some(&FLOAT_MUL_DESCENT),
            B::TrueDivide => Some(&FLOAT_TRUEDIV_DESCENT),
            B::Power => Some(&FLOAT_DESCR_POW_DESCENT),
            _ => None,
        }) else {
            return Ok(None);
        };
        let x = if lhs_is_float {
            unsafe { pyre_object::w_float_get_value(operands[0].1) }
        } else {
            unsafe { pyre_object::w_int_get_value(operands[0].1) as f64 }
        };
        let y = if rhs_is_float {
            unsafe { pyre_object::w_float_get_value(operands[1].1) }
        } else {
            unsafe { pyre_object::w_int_get_value(operands[1].1) as f64 }
        };
        if matches!(plain, B::TrueDivide) && y == 0.0 {
            return Ok(None);
        }
        if matches!(plain, B::Power)
            && pyre_interpreter::objspace::descroperation::float_pow_would_raise(x, y)
        {
            return Ok(None);
        }
        let xa = walker_coerce_dispatching_operand_to_float(
            ctx,
            op_pc,
            operands[0].0,
            operands[0].1,
            !lhs_is_float,
            x,
            false,
        )?;
        let ya = walker_coerce_dispatching_operand_to_float(
            ctx,
            op_pc,
            operands[1].0,
            operands[1].1,
            !rhs_is_float,
            y,
            false,
        )?;
        return try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[],
            &[],
            &[(xa, x), (ya, y)],
            dst,
            dst_bank,
            descent,
        );
    }
    // intobject.py `_truediv(space, x, y)` is the unboxed success leaf:
    // zero, two `cast_int_to_float`, float_truediv, in-graph
    // `new_with_vtable`.  Wide ints raise into residual
    // `int_truediv_ovf2long` (`_make_ovf2long`).
    if !any_long && matches!(plain, B::TrueDivide) {
        const MANTISSA_LIM: i64 = 1 << 53;
        let x = unsafe { pyre_object::w_int_get_value(operands[0].1) };
        let y = unsafe { pyre_object::w_int_get_value(operands[1].1) };
        if x <= -MANTISSA_LIM || x >= MANTISSA_LIM || y <= -MANTISSA_LIM || y >= MANTISSA_LIM {
            return Ok(None);
        }
        let type_addr = |obj| {
            if unsafe { pyre_object::is_bool(obj) } {
                &pyre_object::pyobject::BOOL_TYPE as *const _ as i64
            } else {
                &pyre_object::pyobject::INT_TYPE as *const _ as i64
            }
        };
        let xa = walker_unbox_int(ctx, op_pc, operands[0].0, type_addr(operands[0].1))?;
        walker_guard_exact_w_class(
            ctx,
            op_pc,
            operands[0].0,
            walker_numeric_builtin_class(operands[0].1),
        )?;
        let ya = walker_unbox_int(ctx, op_pc, operands[1].0, type_addr(operands[1].1))?;
        walker_guard_exact_w_class(
            ctx,
            op_pc,
            operands[1].0,
            walker_numeric_builtin_class(operands[1].1),
        )?;
        // Host-side mantissa check only admits the recording operands.
        // Later values must deopt into `int_truediv_ovf2long`.
        walker_guard_int_open_range(ctx, op_pc, xa, x, -MANTISSA_LIM, MANTISSA_LIM)?;
        walker_guard_int_open_range(ctx, op_pc, ya, y, -MANTISSA_LIM, MANTISSA_LIM)?;
        return try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[(xa, x), (ya, y)],
            &[],
            &[],
            dst,
            dst_bank,
            &INT_TRUEDIV_DESCENT,
        );
    }
    let tag = if plain_tag == op_tag {
        tag
    } else {
        ctx.trace_ctx.const_int(plain_tag)
    };
    let outcome = try_walker_orthodox_descent(
        ctx,
        op_pc,
        &[(tag, plain_tag)],
        &operands,
        &[],
        dst,
        dst_bank,
        &BINARY_OP_DESCENT,
    )?;
    // `_ovf_zer` belongs to `//` / `%` only.  Emitting it after every
    // descent promoted the freshly boxed `2` of `n < 2` / `n - 2`
    // (`optimize_GUARD_VALUE`: promote of a virtual) and aborted the
    // fib function-entry trace before any add could compile.
    if matches!(
        (&outcome, plain),
        (
            Some(DispatchOutcome::Continue),
            B::FloorDivide | B::Remainder
        )
    ) {
        walker_guard_int_div_domain_if_exact(ctx, op_pc, r_args)?;
    }
    Ok(outcome)
}

/// `==` / `!=` of two exact array-backed tuples no longer than
/// [`pyre_object::tupleobject::UNROLL_CUTOFF`].
///
/// Element equality is only folded for exact ints and `None`. Anything
/// else — a specialised layout, a subclass, a nested tuple — stays the
/// residual `compare_value_from_tag`, which would force both operands.
/// Item `__eq__` of an int or `None` has no side effect, so comparing
/// every element (instead of returning at the first difference) is the
/// same answer `compare_tuples` produces.
fn try_walker_fold_small_tuple_eq<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    lhs: OpRef,
    rhs: OpRef,
    lhs_obj: pyre_object::PyObjectRef,
    rhs_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let negate = match op_tag {
        4 => false,
        5 => true,
        _ => return Ok(None),
    };
    let tuple_type = &pyre_object::TUPLE_TYPE as *const pyre_object::pyobject::PyType;
    let mut lhs_items = Vec::new();
    let mut rhs_items = Vec::new();
    unsafe {
        if !pyre_object::is_exact_tuple(lhs_obj)
            || !pyre_object::is_exact_tuple(rhs_obj)
            || !std::ptr::eq((*lhs_obj).ob_type, tuple_type)
            || !std::ptr::eq((*rhs_obj).ob_type, tuple_type)
        {
            return Ok(None);
        }
        let lhs_len = pyre_object::w_tuple_len(lhs_obj);
        let rhs_len = pyre_object::w_tuple_len(rhs_obj);
        if lhs_len > pyre_object::tupleobject::UNROLL_CUTOFF
            || rhs_len > pyre_object::tupleobject::UNROLL_CUTOFF
        {
            return Ok(None);
        }
        for index in 0..lhs_len {
            let Some(item) = pyre_object::w_tuple_getitem(lhs_obj, index as i64) else {
                return Ok(None);
            };
            lhs_items.push(item);
        }
        for index in 0..rhs_len {
            let Some(item) = pyre_object::w_tuple_getitem(rhs_obj, index as i64) else {
                return Ok(None);
            };
            rhs_items.push(item);
        }
    }

    enum Pair {
        None,
        Int(i64, i64, pyre_object::PyObjectRef, pyre_object::PyObjectRef),
    }
    let mut pairs = Vec::new();
    if lhs_items.len() == rhs_items.len() {
        let int_typeobj = pyre_object::get_instantiate(&pyre_object::INT_TYPE);
        let exact_int = |obj: pyre_object::PyObjectRef| -> Option<i64> {
            if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj)
            {
                return None;
            }
            unsafe {
                // `_descr_eq` (`tupleobject.py`) calls `space.eq_w` per item.
                // `W_IntObject` (`intobject.py`) is the machine-int class;
                // `W_LongObject` is a sibling. `is_exact_type(..., INT_TYPE)`
                // is Python `type is int` and is true for both, so a long
                // would be unboxed through `W_IntObject.intval`.
                if !pyre_object::is_int(obj)
                    || pyre_object::is_bool(obj)
                    || pyre_object::is_long(obj)
                    || !std::ptr::eq((*obj).w_class, int_typeobj)
                {
                    return None;
                }
                Some(pyre_object::w_int_get_value(obj))
            }
        };
        for (&left, &right) in lhs_items.iter().zip(rhs_items.iter()) {
            if unsafe { pyre_object::is_none(left) && pyre_object::is_none(right) } {
                pairs.push(Pair::None);
            } else if let (Some(left_raw), Some(right_raw)) = (exact_int(left), exact_int(right)) {
                pairs.push(Pair::Int(left_raw, right_raw, left, right));
            } else {
                return Ok(None);
            }
        }
    }

    let tuple_type_addr = tuple_type as i64;
    let tuple_class = pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE);
    walker_guard_exact_instance(ctx, op_pc, lhs, tuple_type_addr, tuple_class)?;
    walker_guard_exact_instance(ctx, op_pc, rhs, tuple_type_addr, tuple_class)?;

    let items_descr = crate::descr::tuple_wrappeditems_descr();
    let array_descr = crate::state::pyobject_gcarray_descr();
    let load_block = |ctx: &mut WalkContext<'_, '_, Sym>,
                      tuple: OpRef,
                      tuple_obj: pyre_object::PyObjectRef|
     -> OpRef {
        let block = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, tuple, items_descr.clone());
        let wrapped = unsafe {
            (*(tuple_obj as *const pyre_object::tupleobject::W_TupleObject)).wrappeditems
        };
        ctx.trace_ctx.set_opref_concrete(
            block,
            majit_ir::Value::Ref(majit_ir::GcRef(wrapped as usize)),
        );
        block
    };
    let lhs_block = load_block(ctx, lhs, lhs_obj);
    let rhs_block = load_block(ctx, rhs, rhs_obj);
    let pin_len = |ctx: &mut WalkContext<'_, '_, Sym>,
                   block: OpRef,
                   len: usize|
     -> Result<(), DispatchError> {
        let len_op = crate::state::opimpl_arraylen_gc(ctx.trace_ctx, block, array_descr.clone());
        if !len_op.is_constant() {
            let expected = ctx.trace_ctx.const_int(len as i64);
            let same = ctx.trace_ctx.record_op(OpCode::IntEq, &[len_op, expected]);
            ctx.trace_ctx
                .set_opref_concrete(same, majit_ir::Value::Int(1));
            walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[same])?;
        }
        Ok(())
    };
    pin_len(ctx, lhs_block, lhs_items.len())?;
    pin_len(ctx, rhs_block, rhs_items.len())?;

    let (truth, observed) = if lhs_items.len() != rhs_items.len() {
        let value = i64::from(negate);
        (ctx.trace_ctx.const_int(value), negate)
    } else {
        let none_obj = pyre_object::w_none();
        let mut acc = ctx.trace_ctx.const_int(1);
        let mut acc_value = 1i64;
        for (index, pair) in pairs.iter().enumerate() {
            let index_op = ctx.trace_ctx.const_int(index as i64);
            let left = crate::state::trace_items_block_getitem_value_pure(
                ctx.trace_ctx,
                lhs_block,
                index_op,
            );
            let right = crate::state::trace_items_block_getitem_value_pure(
                ctx.trace_ctx,
                rhs_block,
                index_op,
            );
            match *pair {
                Pair::None => {
                    ctx.trace_ctx.set_opref_concrete(
                        left,
                        majit_ir::Value::Ref(majit_ir::GcRef(none_obj as usize)),
                    );
                    ctx.trace_ctx.set_opref_concrete(
                        right,
                        majit_ir::Value::Ref(majit_ir::GcRef(none_obj as usize)),
                    );
                    walker_guard_stamped_ref_hold(ctx, op_pc, left, none_obj)?;
                    walker_guard_stamped_ref_hold(ctx, op_pc, right, none_obj)?;
                }
                Pair::Int(left_raw, right_raw, left_obj, right_obj) => {
                    ctx.trace_ctx.set_opref_concrete(
                        left,
                        majit_ir::Value::Ref(majit_ir::GcRef(left_obj as usize)),
                    );
                    ctx.trace_ctx.set_opref_concrete(
                        right,
                        majit_ir::Value::Ref(majit_ir::GcRef(right_obj as usize)),
                    );
                    let (left_type, left_descr) =
                        crate::state::int_or_bool_unbox_type_descr(left_obj);
                    let (right_type, right_descr) =
                        crate::state::int_or_bool_unbox_type_descr(right_obj);
                    let left_unboxed = walker_unbox_int_exact(
                        ctx,
                        op_pc,
                        left,
                        left_type,
                        left_descr,
                        walker_numeric_builtin_class(left_obj),
                    )?;
                    let right_unboxed = walker_unbox_int_exact(
                        ctx,
                        op_pc,
                        right,
                        right_type,
                        right_descr,
                        walker_numeric_builtin_class(right_obj),
                    )?;
                    let eq = ctx
                        .trace_ctx
                        .record_op(OpCode::IntEq, &[left_unboxed, right_unboxed]);
                    let eq_value = i64::from(left_raw == right_raw);
                    ctx.trace_ctx
                        .set_opref_concrete(eq, majit_ir::Value::Int(eq_value));
                    let next = ctx.trace_ctx.record_op(OpCode::IntAnd, &[acc, eq]);
                    acc_value &= eq_value;
                    ctx.trace_ctx
                        .set_opref_concrete(next, majit_ir::Value::Int(acc_value));
                    acc = next;
                }
            }
        }
        if negate {
            let one = ctx.trace_ctx.const_int(1);
            let inverted = ctx.trace_ctx.record_op(OpCode::IntXor, &[acc, one]);
            let inverted_value = acc_value ^ 1;
            ctx.trace_ctx
                .set_opref_concrete(inverted, majit_ir::Value::Int(inverted_value));
            (inverted, inverted_value != 0)
        } else {
            (acc, acc_value != 0)
        }
    };
    let boxed = match walker_newbool_guarded(ctx, op_pc, truth, observed, dst_bank)? {
        Some(boxed) => boxed,
        None => {
            let boxed =
                crate::helpers::emit_trace_bool_value_from_truth(ctx.trace_ctx, truth, false);
            let result_obj = pyre_object::w_bool_from(observed);
            ctx.trace_ctx.set_opref_concrete(
                boxed,
                majit_ir::Value::Ref(majit_ir::GcRef(result_obj as usize)),
            );
            boxed
        }
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, boxed)?;
    Ok(Some(()))
}

/// `COMPARE_OP` on two exact builtin machine ints (`int`, `bool`): descend
/// `compare_value_from_tag` → `compare` → `compare_slot` → `int_lt` and its
/// siblings.  Floats, longs and a pair of exact `str`s descend the same
/// helper.  The hand-emitted int, long and str compare folds are retired.  See
/// [`try_walker_orthodox_binary_op`] for the operand policy; the body's
/// override probe is promoted away for such a pair
/// (`descroperation.rs compare`), and the `bool`-vs-`int` subtype ordering
/// it keeps is decided on the promoted classes.
///
/// Tags 0..=5 are the six rich comparisons.  `in` / `not in` (6, 7)
/// descend the same [`COMPARE_OP_DESCENT`] helper — `compare_value_from_tag`
/// routes those tags to `baseobjspace::contains`.  `is` / `is_not` (8, 9)
/// descend it too: `compare_value_from_tag` calls `ObjSpace.is_w` and
/// `w_bool_from`, the same pair `runtime_ops::is_op` records for bytecode
/// `IS_OP`.  CHECK_EXC_MATCH (10) keeps its own fold.
pub(crate) fn try_walker_orthodox_compare_op<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op_tag: i64,
    tag: OpRef,
    r_args: &[OpRef],
    dst: usize,
    dst_bank: char,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 || dst_bank != 'r' {
        return Ok(None);
    }
    if pyre_interpreter::runtime_ops::compare_op_tag_is_contains(op_tag) {
        let (Some(needle_obj), Some(haystack_obj)) = (
            walker_concrete_ref_object(ctx, r_args[0]),
            walker_concrete_ref_object(ctx, r_args[1]),
        ) else {
            return Ok(None);
        };
        if !walker_contains_descent_callback_free(needle_obj, haystack_obj) {
            return Ok(None);
        }
        return try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[(tag, op_tag)],
            &[(r_args[0], needle_obj), (r_args[1], haystack_obj)],
            &[],
            dst,
            dst_bank,
            &COMPARE_OP_DESCENT,
        );
    }
    // `pyopcode.py IS_OP`. Bytecode emits `inline_call` of
    // `runtime_ops::is_op`. A `compare_fn` residual still carries tags 8/9
    // (`compare_op_tag_for_opname`), and `compare_value_from_tag` records
    // `ObjSpace.is_w` for those tags. Same-box `ptr_eq` folds in
    // `opimpl_ptr_eq` (`b1 is b2`, `FASTPATHS_SAME_BOXES`). A null operand
    // raises in the body; decline before the sub-walk so the raise is not
    // executed twice.
    if pyre_interpreter::runtime_ops::compare_op_tag_is_identity(op_tag) {
        let (Some(lhs_obj), Some(rhs_obj)) = (
            walker_concrete_ref_object(ctx, r_args[0]),
            walker_concrete_ref_object(ctx, r_args[1]),
        ) else {
            return Ok(None);
        };
        if lhs_obj.is_null() || rhs_obj.is_null() {
            return Ok(None);
        }
        return try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[(tag, op_tag)],
            &[(r_args[0], lhs_obj), (r_args[1], rhs_obj)],
            &[],
            dst,
            dst_bank,
            &COMPARE_OP_DESCENT,
        );
    }
    if !(0..=5).contains(&op_tag) {
        return Ok(None);
    }
    // `compare_tuples` is `@jit.look_inside_iff(unroll_condition)`, but
    // `unroll_condition` is false outside the JIT (`isconstant` is
    // residual). The concrete lengths are known here, so a short exact
    // pair is folded directly. A `CallMayForce` would force both tuples.
    if let (Some(lhs_obj), Some(rhs_obj)) = (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) && try_walker_fold_small_tuple_eq(
        ctx, op_pc, op_tag, r_args[0], r_args[1], lhs_obj, rhs_obj, dst, dst_bank,
    )?
    .is_some()
    {
        return Ok(Some(DispatchOutcome::Continue));
    }
    let mut operands = [(OpRef::NONE, std::ptr::null_mut()); 2];
    for (slot, &operand) in operands.iter_mut().zip(r_args) {
        let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
            return Ok(None);
        };
        // SAFETY: `obj` is a live concrete `PyObjectRef` from the walker
        // shadow.  Exact builtin float walks `_float_lt` and siblings
        // (`descr_*` after `_to_float`); mixed int/float does too once
        // the int is exact as a double (`int_between(-1, i2 >> 48, 1)`).
        // Exact builtin long walks `compare_slot`'s loop-free arms
        // (`rbigint.lt` / `rbigint.int_lt`). Exact str walks the `_utf8`
        // reads and `ll_streq` / `ll_strcmp` (`descr_eq` / `descr_lt`).
        let admitted = unsafe {
            pyre_object::is_exact_builtin_instance(obj)
                && (pyre_object::is_int(obj)
                    || pyre_object::is_bool(obj)
                    || pyre_object::is_float(obj)
                    || pyre_object::is_long(obj)
                    || pyre_object::is_str(obj))
        };
        if !admitted {
            return Ok(None);
        }
        *slot = (operand, obj);
    }
    // A str paired with a number leaves `compare_slot` for
    // [`compare_slot_rest`], whose graph contains loops.
    let lhs_is_str = unsafe { pyre_object::is_str(operands[0].1) };
    let rhs_is_str = unsafe { pyre_object::is_str(operands[1].1) };
    if lhs_is_str != rhs_is_str {
        return Ok(None);
    }
    let lhs_is_float = unsafe { pyre_object::is_float(operands[0].1) };
    let rhs_is_float = unsafe { pyre_object::is_float(operands[1].1) };
    let any_long =
        unsafe { pyre_object::is_long(operands[0].1) || pyre_object::is_long(operands[1].1) };
    // A long paired with a float leaves `compare_slot` for
    // [`compare_slot_rest`]. That graph contains loops
    // (`policy.py look_inside_graph`), so the descent would not record the
    // arm. That pair stays on the residual.
    if any_long && (lhs_is_float || rhs_is_float) {
        return Ok(None);
    }
    if !any_long && (lhs_is_float || rhs_is_float) {
        let Some(descent) = (match op_tag {
            0 => Some(&FLOAT_LT_DESCENT),
            1 => Some(&FLOAT_LE_DESCENT),
            2 => Some(&FLOAT_GT_DESCENT),
            3 => Some(&FLOAT_GE_DESCENT),
            4 => Some(&FLOAT_EQ_DESCENT),
            5 => Some(&FLOAT_NE_DESCENT),
            _ => None,
        }) else {
            return Ok(None);
        };
        if !lhs_is_float {
            let value = unsafe { pyre_object::w_int_get_value(operands[0].1) };
            if !int_is_exact_as_float(value) {
                return Ok(None);
            }
        }
        if !rhs_is_float {
            let value = unsafe { pyre_object::w_int_get_value(operands[1].1) };
            if !int_is_exact_as_float(value) {
                return Ok(None);
            }
        }
        let x = if lhs_is_float {
            unsafe { pyre_object::w_float_get_value(operands[0].1) }
        } else {
            unsafe { pyre_object::w_int_get_value(operands[0].1) as f64 }
        };
        let y = if rhs_is_float {
            unsafe { pyre_object::w_float_get_value(operands[1].1) }
        } else {
            unsafe { pyre_object::w_int_get_value(operands[1].1) as f64 }
        };
        let xa = walker_coerce_dispatching_operand_to_float(
            ctx,
            op_pc,
            operands[0].0,
            operands[0].1,
            !lhs_is_float,
            x,
            true,
        )?;
        let ya = walker_coerce_dispatching_operand_to_float(
            ctx,
            op_pc,
            operands[1].0,
            operands[1].1,
            !rhs_is_float,
            y,
            true,
        )?;
        return try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[],
            &[],
            &[(xa, x), (ya, y)],
            dst,
            dst_bank,
            descent,
        );
    }
    try_walker_orthodox_descent(
        ctx,
        op_pc,
        &[(tag, op_tag)],
        &operands,
        &[],
        dst,
        dst_bank,
        &COMPARE_OP_DESCENT,
    )
}

/// Descend `baseobjspace::getitem_str` (`descr_getitem` after
/// `getindex_w`) for an exact `str` and an exact `int` index.
/// A missing jitcode declines to the generic residual.
///
/// The boxed index is not frozen: a loop over `s[i]` must keep the
/// live key so the generated length test stays in the body.
/// Specialised-tuple descent freezes because that reader is a
/// two-slot `match`.
fn try_walker_orthodox_str_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_int(key_obj) } {
        return Ok(None);
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(key_obj) {
        // A tagged key has no header for the class guard the generated
        // body does not emit. Keep that shape on the residual until the
        // tag-aware unbox is the body itself.
        return Ok(None);
    }
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let raw_key = unsafe {
        if !std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*key_obj).w_class, int_typeobj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(key_obj)
    };
    let len = unsafe { pyre_object::w_str_len(seq_obj) } as i64;
    let index = if raw_key < 0 { raw_key + len } else { raw_key };
    if usize::try_from(index).ok().is_none() {
        return Ok(None);
    }
    if unsafe { pyre_object::w_str_codepoint_at(seq_obj, index as usize) }.is_none() {
        return Ok(None);
    }

    let Some(jc_arc) = crate::jitcode_runtime::str_getitem_jitcode() else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    walker_guard_exact_str(ctx, op_pc, seq_op)?;
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    walker_guard_exact_instance(ctx, op_pc, key_op, int_type_addr, int_typeobj)?;
    ctx.trace_ctx.set_opref_concrete(
        seq_op,
        majit_ir::Value::Ref(majit_ir::GcRef(seq_obj as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        key_op,
        majit_ir::Value::Ref(majit_ir::GcRef(key_obj as usize)),
    );
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "str_getitem_commit",
        "getitem_str_call_site",
        &[],
        &[],
        &[seq_op, key_op],
        &[ConcreteValue::Ref(seq_obj), ConcreteValue::Ref(key_obj)],
        &[],
    );
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] STR-GETITEM-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Descend `baseobjspace::getitem_bytes_like` (`stringmethods.py descr_getitem`).
/// The scalar arm is `strgetitem` plus `newint(ord)`.  A missing jitcode
/// declines to the generic residual.
fn try_walker_orthodox_bytes_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_int(key_obj) } {
        return Ok(None);
    }
    let tagged_key =
        pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(key_obj);
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let raw_key = unsafe {
        if !tagged_key
            && (!std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
                || !std::ptr::eq((*key_obj).w_class, int_typeobj))
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(key_obj)
    };
    let len = unsafe { pyre_object::bytesobject::w_bytes_len(seq_obj) } as i64;
    let index = if raw_key < 0 { raw_key + len } else { raw_key };
    if index < 0 || index >= len {
        return Ok(None);
    }

    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::baseobjspace::getitem_bytes_like",
    ) else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let bytes_type_addr = &pyre_object::bytesobject::BYTES_TYPE as *const _ as i64;
    let bytes_typeobj =
        pyre_object::pyobject::get_instantiate(&pyre_object::bytesobject::BYTES_TYPE);
    walker_guard_exact_instance(ctx, op_pc, seq_op, bytes_type_addr, bytes_typeobj)?;
    if !tagged_key {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        walker_guard_exact_instance(ctx, op_pc, key_op, int_type_addr, int_typeobj)?;
    }
    ctx.trace_ctx.set_opref_concrete(
        seq_op,
        majit_ir::Value::Ref(majit_ir::GcRef(seq_obj as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        key_op,
        majit_ir::Value::Ref(majit_ir::GcRef(key_obj as usize)),
    );
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "bytes_getitem_commit",
        "getitem_bytes_call_site",
        &[],
        &[],
        &[seq_op, key_op],
        &[ConcreteValue::Ref(seq_obj), ConcreteValue::Ref(key_obj)],
        &[],
    );
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] BYTES-GETITEM-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Drop a speculative `FrameLocalsProxy.__getitem__` descent and let the
/// generic residual record the subscript.
///
/// The hit arm returns. A miss raises, and a helper the walk cannot
/// record (`LoopHeaderJdIndexUnresolved`, an unscannable op) would abort
/// the portal if it propagated. Cut the trace back, drop the heap cache
/// the helper filled, put the walker's exception slot back to what it held
/// before the descent, and clear `take_call_error` so the residual path does
/// not observe the miss.
fn decline_frame_locals_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pre_fold_pos: majit_metainterp::recorder::TracePosition,
    exc_before: (Option<OpRef>, ConcreteValue),
) -> Result<Option<()>, DispatchError> {
    ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
    ctx.trace_ctx.heap_cache_mut().reset();
    ctx.restore_last_exc_value(exc_before.0, exc_before.1);
    let _ = pyre_interpreter::call::take_call_error();
    Ok(None)
}

/// Descend `FrameLocalsProxy.__getitem__` for an exact proxy and an
/// exact `str` key.
///
/// `locals_plus_value` is `@jit.unroll_safe` and reads the viewed
/// frame's localsplus array, the same scan `pyframe.py fast2locals`
/// unrolls. The 3.14 extras miss path is that body's own
/// `get_extra_locals` getfield. A walk the helper cannot record
/// declines to the generic residual rather than answering a vable
/// slot by hand.
fn try_walker_orthodox_frame_locals_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    seq_op: OpRef,
    key_op: OpRef,
    seq_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    let str_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE);
    let exact_str = unsafe {
        !str_typeobj.is_null()
            && std::ptr::eq((*key_obj).ob_type, &pyre_object::pyobject::STR_TYPE)
            && std::ptr::eq((*key_obj).w_class, str_typeobj)
    };
    if !exact_str {
        return Ok(None);
    }
    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::pyframe::frame_locals_proxy::<Impl>::__getitem__",
    ) else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // The slot is session-wide: inside an `except` whose type expression
    // reads the proxy, it holds the exception the match is about to test.
    // `finish_inline_callee_return` clears it, so put it back after the walk.
    let exc_before = (ctx.last_exc_value(), ctx.last_exc_value_concrete());
    let pytype = <pyre_interpreter::pyframe::frame_locals_proxy::FrameLocalsProxy as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE;
    let proxy_typeobj = pyre_object::pyobject::get_instantiate(unsafe { &*pytype });
    walker_guard_exact_instance(ctx, op_pc, seq_op, pytype as i64, proxy_typeobj)?;
    walker_guard_exact_str(ctx, op_pc, key_op)?;
    ctx.trace_ctx.set_opref_concrete(
        seq_op,
        majit_ir::Value::Ref(majit_ir::GcRef(seq_obj as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        key_op,
        majit_ir::Value::Ref(majit_ir::GcRef(key_obj as usize)),
    );
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "frame_locals_getitem_commit",
        "frame_locals_getitem_call_site",
        &[],
        &[],
        &[seq_op, key_op],
        &[ConcreteValue::Ref(seq_obj), ConcreteValue::Ref(key_obj)],
        &[],
    );
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { .. }) => {
            return decline_frame_locals_getitem(ctx, pre_fold_pos, exc_before);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return decline_frame_locals_getitem(ctx, pre_fold_pos, exc_before),
    };
    ctx.restore_last_exc_value(exc_before.0, exc_before.1);
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    Ok(Some(()))
}

/// Descend `baseobjspace::list_iter_descr_next`
/// (`iterobject.py` `W_FastListIterObject.descr_next`).
///
/// The graph key is the helper root registered in `prepass.rs`
/// (`pyre_interpreter::baseobjspace::list_iter_descr_next`). The step
/// (seq/index reads, bounds, index store) is the interpreter body. The
/// traced index must stay a red getfield; a constant index makes the
/// loaded element a loop constant.
pub(crate) fn try_walker_orthodox_list_iter_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }
    let iter_op = r_args[0];
    let Some(iter_obj) = walker_concrete_ref_object(ctx, iter_op) else {
        return Ok(None);
    };
    if unsafe { !pyre_object::is_list_iter(iter_obj) } {
        return Ok(None);
    }
    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::baseobjspace::list_iter_descr_next",
    ) else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let journal_mark = fbw_effect_journal_mark();
    let index_before = unsafe { pyre_object::w_list_iter_index(iter_obj) };
    let seq_before = unsafe { pyre_object::w_list_iter_seq(iter_obj) };
    // A new consume completes the previous in-flight iteration before this
    // step, matching the residual executor.
    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);
    let iter_type_addr = &pyre_object::iterobject::LIST_ITER_TYPE as *const _ as i64;
    walker_guard_class(ctx, op_pc, iter_op, iter_type_addr)?;
    ctx.trace_ctx.set_opref_concrete(
        iter_op,
        majit_ir::Value::Ref(majit_ir::GcRef(iter_obj as usize)),
    );
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "list_iter_descr_next_commit",
        "list_iter_descr_next_call_site",
        &[],
        &[],
        &[iter_op],
        &[ConcreteValue::Ref(iter_obj)],
        &[],
    );
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-ITER-DESCR-NEXT pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    // The walked `setfield_gc` moved the cursor as it was recorded; the
    // stores below are for a body whose store the walk could not execute.
    let iter_now = walker_concrete_ref_object(ctx, iter_op).unwrap_or(iter_obj);
    fbw_gc_store_journal_keep_since(journal_mark, iter_now);
    let index_after = unsafe { pyre_object::w_list_iter_index(iter_now) };
    let seq_after = unsafe { pyre_object::w_list_iter_seq(iter_now) };
    let cursor_unchanged = seq_after == seq_before && index_after == index_before;
    let item = walker_concrete_ref_object(ctx, result).filter(|item| !item.is_null());
    // Exhaustion (`list_iter_stop`, index >= 0).  A negative `__setstate__`
    // cursor stays attached; that arm returns null without the store.
    let exhausted = item.is_none()
        && matches!(
            ctx.trace_ctx.concrete_of_opref(result),
            Some(majit_ir::Value::Ref(r)) if r.as_usize() == 0
        );
    // The store the body recorded, for a walk that could not execute it.
    let replay = cursor_unchanged && !seq_before.is_null() && index_before >= 0;
    // A kept store has no undo entry, so the bridge cursor snapshot is the
    // only way back for a walk that aborts with its delivery refused; it is
    // taken from the pre-walk pair whichever side moved the cursor.
    if ctx.trace_ctx.is_bridge_trace
        && (!cursor_unchanged || (replay && (item.is_some() || exhausted)))
    {
        fbw_bridge_list_iter_journal_push(iter_now, seq_before, index_before);
    }
    if let Some(item) = item {
        if replay {
            unsafe { pyre_object::w_list_iter_set_index(iter_now, index_before + 1) };
        }
        fbw_foriter_inflight_capture(item, body, true);
    } else if exhausted && replay {
        unsafe { pyre_object::w_list_iter_set_seq(iter_now, pyre_object::PY_NULL) };
    }
    Ok(Some(result))
}

/// Whether `callable` is the `dict.get` method object.
///
/// The typedef registers the slot as
/// `make_builtin_function("get", __majit_wrap_dict_descr_get)`, so that leaf
/// is what `BuiltinCode.func` holds. Naming the `dict_method_get` body it
/// forwards to instead compares two different functions, and answered true
/// only while a build happened to give them one address.
fn is_builtin_dict_get_function(callable: pyre_object::PyObjectRef) -> bool {
    if callable.is_null() || !unsafe { pyre_interpreter::is_function(callable) } {
        return false;
    }
    let code = unsafe { pyre_interpreter::function_get_code(callable) } as pyre_object::PyObjectRef;
    !code.is_null()
        && unsafe { pyre_interpreter::is_builtin_code(code) }
        && unsafe { pyre_interpreter::builtin_code_get(code) as usize }
            == pyre_interpreter::type_methods::__majit_wrap_dict_descr_get as *const () as usize
}

#[derive(Clone, Copy)]
enum DictFoldKeyProbe {
    Int,
    Unicode,
}

#[derive(Clone, Copy)]
struct DictFoldHit {
    concrete_value: pyre_object::PyObjectRef,
    key_probe: DictFoldKeyProbe,
}

fn walker_probe_exact_dict_hit(
    dict: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
) -> Result<Option<DictFoldHit>, DispatchError> {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    if canonical_dict.is_null()
        || !unsafe {
            std::ptr::eq((*dict).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*dict).w_class, canonical_dict)
        }
    {
        return Ok(None);
    }

    // Only the two homogeneous strategies fold: their lookups probe a native
    // table and so cannot run Python-level `__hash__` or `__eq__`.  Every other
    // strategy, a mapdict-backed instance dict included, keeps the real lookup.
    let strategy_kind =
        unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(dict).strategy_kind() };
    let int_probe = strategy_kind == pyre_object::dictmultiobject::StrategyKind::Int
        && unsafe { pyre_object::listobject::is_plain_int1(key) && pyre_object::is_int(key) };
    let unicode_probe = strategy_kind == pyre_object::dictmultiobject::StrategyKind::Unicode
        && unsafe {
            pyre_object::is_exact_type(key, &pyre_object::pyobject::STR_TYPE)
                && pyre_object::w_str_get_value_opt(key).is_some()
                && pyre_object::dict_eq_hook::try_hash_str(
                    pyre_object::w_str_get_value_opt(key).unwrap().as_bytes(),
                )
                .is_some()
        };

    let found = if int_probe {
        let index =
            unsafe { pyre_object::dictmultiobject::w_dict_index_of_int_strategy(dict, key) };
        index.and_then(|index| {
            unsafe { pyre_object::dictmultiobject::w_dict_nth_value(dict, index) }
                .map(|value| (value, DictFoldKeyProbe::Int))
        })
    } else if unicode_probe {
        let index =
            unsafe { pyre_object::dictmultiobject::w_dict_index_of_unicode_strategy(dict, key) };
        index.and_then(|index| {
            unsafe { pyre_object::dictmultiobject::w_dict_nth_value(dict, index) }
                .map(|value| (value, DictFoldKeyProbe::Unicode))
        })
    } else {
        None
    };

    let Some((concrete_value, key_probe)) = found else {
        return Ok(None);
    };
    Ok(Some(DictFoldHit {
        concrete_value,
        key_probe,
    }))
}

fn walker_emit_exact_dict_hit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    key_op: OpRef,
    dict: pyre_object::PyObjectRef,
    hit: DictFoldHit,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // Int-strategy hit is the fail arm of the same `dict.lookup` the miss
    // fold emits (`ll_dict_getitem_with_hash`: `int_lt(index, 0)` then
    // `d.entries[index].value`).  A separate `lookup_or_null` would pin
    // the first hit's value and storm the guard on every other key.
    if matches!(hit.key_probe, DictFoldKeyProbe::Int) {
        return walker_emit_exact_dict_int_hit(
            ctx, op_pc, dict_op, key_op, dict, hit, dst, dst_bank,
        );
    }

    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    walker_guard_exact_instance(
        ctx,
        op_pc,
        dict_op,
        &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        canonical_dict,
    )?;

    let strategy_ref = &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF as *const _ as i64;
    let lookup_helper = crate::helpers::jit_dict_exact_unicode_lookup_or_null as *const ();

    walker_guard_stamped_dict_strategy(ctx, op_pc, dict_op, strategy_ref)?;

    walker_guard_exact_str(ctx, op_pc, key_op)?;

    let value = ctx.trace_ctx.call_ref_typed_with_effect(
        lookup_helper,
        &[dict_op, key_op],
        &[majit_ir::Type::Ref, majit_ir::Type::Ref],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    walker_guard_stamped_nonnull(ctx, op_pc, value, hit.concrete_value)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// Hit arm of `ll_dict_getitem_with_hash` after `IntDictStrategy.getitem`
/// found the key: the same `dict.lookup` the miss fold emits, then
/// `int_lt(index, 0)` / `guard_false` and `d.entries[index].value`.
fn walker_emit_exact_dict_int_hit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    key_op: OpRef,
    dict: pyre_object::PyObjectRef,
    hit: DictFoldHit,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    // A virtual dict records `ll_dict_lookup` from the int-strategy body.
    // This fold reads a concrete table.
    if ctx.trace_ctx.is_likely_virtual(dict_op) {
        return Ok(None);
    }
    let Some(key) = walker_concrete_ref_object(ctx, key_op) else {
        return Ok(None);
    };
    let Some(index_concrete) =
        (unsafe { pyre_object::dictmultiobject::w_dict_index_of_int_strategy(dict, key) })
    else {
        return Ok(None);
    };
    let (index_op, storage_op) = walker_emit_int_dict_lookup_index(
        ctx,
        op_pc,
        dict_op,
        key_op,
        dict,
        key,
        index_concrete as i64,
    )?;
    let zero = ctx.trace_ctx.const_int(0);
    let is_miss = ctx.trace_ctx.record_op(OpCode::IntLt, &[index_op, zero]);
    ctx.trace_ctx
        .set_opref_concrete(is_miss, majit_ir::Value::Int(0));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[is_miss])?;
    let value = ctx.trace_ctx.call_ref_typed_with_effect(
        crate::helpers::jit_dict_int_value_at as *const (),
        &[storage_op, index_op],
        &[majit_ir::Type::Ref, majit_ir::Type::Int],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    walker_guard_stamped_nonnull(ctx, op_pc, value, hit.concrete_value)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, value)?;
    Ok(Some(()))
}

/// `IntDictStrategy.getitem` miss: the key is a plain int and the table
/// has no entry.  `descr_getitem` then does `space.raise_key_error(w_key)`.
fn walker_probe_exact_dict_int_miss(
    dict: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
) -> bool {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    if canonical_dict.is_null()
        || !unsafe {
            std::ptr::eq((*dict).ob_type, &pyre_object::pyobject::DICT_TYPE)
                && std::ptr::eq((*dict).w_class, canonical_dict)
        }
    {
        return false;
    }
    let strategy_kind =
        unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(dict).strategy_kind() };
    strategy_kind == pyre_object::dictmultiobject::StrategyKind::Int
        && unsafe { pyre_object::listobject::is_plain_int1(key) && pyre_object::is_int(key) }
        && unsafe { pyre_object::dictmultiobject::w_dict_index_of_int_strategy(dict, key) }
            .is_none()
}

/// `rordereddict.py ll_dict_lookup` for an Int strategy: class / strategy /
/// key pins, `getfield dstorage`, then the four-argument `dict.lookup`
/// oopspec on the unerased table.  Returns `(index, storage)`.
///
/// Shape matches `ll_dict_getitem_with_hash`:
/// `call_i(lookup, dstorage, plain_int_w(key), hash, FLAG_LOOKUP)`.
/// Int hash is identity (`ll_int_hash`), so the hash operand is the same
/// unboxed word as the key.
fn walker_emit_int_dict_lookup_index<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    key_op: OpRef,
    dict: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
    index_concrete: i64,
) -> Result<(OpRef, OpRef), DispatchError> {
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    walker_guard_exact_instance(
        ctx,
        op_pc,
        dict_op,
        &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
        canonical_dict,
    )?;
    walker_guard_stamped_dict_strategy(
        ctx,
        op_pc,
        dict_op,
        &pyre_object::dictmultiobject::INT_DICT_STRATEGY_REF as *const _ as i64,
    )?;
    let int_type = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    walker_guard_exact_instance(
        ctx,
        op_pc,
        key_op,
        int_type,
        pyre_object::get_instantiate(&pyre_object::pyobject::INT_TYPE),
    )?;
    let storage_op = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        dict_op,
        crate::descr::dict_dstorage_descr(),
    );
    let storage_ptr =
        unsafe { pyre_object::dictmultiobject::w_dict_int_storage(dict) as *const _ as usize };
    ctx.trace_ctx.set_opref_concrete(
        storage_op,
        majit_ir::Value::Ref(majit_ir::GcRef(storage_ptr)),
    );
    let key_int = walker_unbox_int(ctx, op_pc, key_op, int_type)?;
    let mut lookup_effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::CannotRaise,
        majit_ir::OopSpecIndex::DictLookup,
    );
    lookup_effect.extradescrs = Some(vec![
        crate::descr::dict_lookup_namespace_descr(),
        crate::descr::dict_lookup_entries_array_descr(),
    ]);
    let lookup_flag = ctx.trace_ctx.const_int(0);
    let index_op = ctx.trace_ctx.call_typed_with_effect(
        OpCode::CallI,
        crate::helpers::jit_dict_exact_int_lookup_index as *const (),
        &[storage_op, key_int, key_int, lookup_flag],
        &[
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ],
        majit_ir::Type::Int,
        lookup_effect,
    );
    ctx.trace_ctx
        .set_opref_concrete(index_op, majit_ir::Value::Int(index_concrete));
    let _ = key;
    Ok((index_op, storage_op))
}

/// BINARY_SUBSCR miss on an exact int-strategy dict: `descr_getitem` after
/// `IntDictStrategy.getitem` returned None.  Emits `dict.lookup` +
/// `int_lt(index, 0)` then `raise_key_error`, and surfaces `SubRaise` so
/// the inlined `except KeyError` catches it.
pub(crate) fn try_walker_specialize_subscr_int_miss<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 2 {
        return Ok(None);
    }
    // Same decline as the hit fold: a virtual dict stays on the inlined
    // getitem body. `Err` here would abort the trace.
    if ctx.trace_ctx.is_likely_virtual(r_args[0]) {
        return Ok(None);
    }
    let (Some(dict), Some(key)) = (
        walker_concrete_ref_object(ctx, r_args[0]),
        walker_concrete_ref_object(ctx, r_args[1]),
    ) else {
        return Ok(None);
    };
    if !walker_probe_exact_dict_int_miss(dict, key) {
        return Ok(None);
    }
    walker_emit_exact_dict_key_error(ctx, op_pc, r_args[0], r_args[1], dict, key)
}

/// `descr_getitem` miss after `IntDictStrategy.getitem` returned None:
/// `space.raise_key_error(w_key)` — `KeyError(w_key)` then raise.  The
/// exception is built inline so a locally-caught `except KeyError` DCEs it,
/// matching the look-inside raise PyPy records after `int_lt(index, 0)`.
fn walker_emit_exact_dict_key_error<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dict_op: OpRef,
    key_op: OpRef,
    dict: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };
    let (index_op, _storage_op) =
        walker_emit_int_dict_lookup_index(ctx, op_pc, dict_op, key_op, dict, key, -1)?;
    let zero = ctx.trace_ctx.const_int(0);
    let is_miss = ctx.trace_ctx.record_op(OpCode::IntLt, &[index_op, zero]);
    ctx.trace_ctx
        .set_opref_concrete(is_miss, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[is_miss])?;

    let concrete = pyre_interpreter::PyError::key_error_with_key(key).exc_object;
    if concrete.is_null() || !unsafe { pyre_object::is_exception(concrete) } {
        return Ok(None);
    }
    // Static typeptr: read it before `emit_rlist_inline` can collect.
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete) };
    let exc_type = unsafe { (*concrete).ob_type } as *const _ as i64;
    let kind = pyre_object::interp_exceptions::ExcKind::KeyError;
    let args_list = crate::helpers::emit_rlist_inline(ctx.trace_ctx, &[key_op]);
    let class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind);
    let class = ctx.trace_ctx.const_ref(class as i64);
    let raised =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class, args_list, user);
    ctx.trace_ctx.heap_cache_mut().class_now_known(raised);
    ctx.trace_ctx.set_opref_concrete(
        raised,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
    );
    fbw_built_exc_insert(raised);
    walker_chain_exception_context(ctx, ec, raised, concrete, kind, user);
    fbw_count_executed_residual(true, true);
    ctx.set_last_exc_value(raised, ConcreteValue::Ref(concrete));
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(concrete as i64));
    Ok(Some(DispatchOutcome::SubRaise {
        exc: raised,
        exc_concrete: ConcreteValue::Ref(concrete),
    }))
}

/// `dict.get` on an exact dictionary and an Int/Unicode strategy hit.
///
/// `dictmultiobject.py getitem` probes an Int strategy with the unboxed key
/// value, and `dictmultiobject.py getitem_str` probes a Unicode strategy with
/// exact-str bytes. The trace guards the exact dict, strategy vtable, and exact
/// key type, then performs a live strategy-specific lookup and guards that it
/// hit. Misses and object/identity strategy keys remain residual so their
/// hash/equality effects stay observable (`rdict.py:576`).
pub(crate) fn try_walker_specialize_builtin_dict_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if !(r_args.len() == 3 || r_args.len() == 4) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (
        ConcreteValue::Ref(callable_operand),
        ConcreteValue::Ref(null_or_self),
        ConcreteValue::Ref(key),
    ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
    else {
        return Ok(None);
    };
    if callable_operand.is_null() || key.is_null() {
        return Ok(None);
    }
    // LOAD_ATTR's generic method path produces `[Method, PY_NULL, key]`;
    // LOAD_METHOD's split path produces `[Function, receiver, key]`.
    // `_Method._immutable_fields_` lets both converge on the same live
    // function/receiver reads.
    let bound_method =
        null_or_self.is_null() && unsafe { pyre_object::is_method(callable_operand) };
    let (callable, dict) = if bound_method {
        (
            unsafe { pyre_object::w_method_get_func(callable_operand) },
            unsafe { pyre_object::w_method_get_self(callable_operand) },
        )
    } else {
        (callable_operand, null_or_self)
    };
    if callable.is_null() || dict.is_null() || !is_builtin_dict_get_function(callable) {
        return Ok(None);
    }
    let Some(hit) = walker_probe_exact_dict_hit(dict, key)? else {
        return Ok(None);
    };

    let mut callable_op = r_args[0];
    let dict_op;
    if bound_method {
        walker_guard_class(
            ctx,
            op.pc,
            r_args[0],
            &pyre_object::function::METHOD_TYPE as *const _ as i64,
        )?;
        callable_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_function_descr(),
        );
        dict_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            dict_op,
            majit_ir::Value::Ref(majit_ir::GcRef(dict as usize)),
        );
    } else {
        dict_op = r_args[1];
    }
    walker_guard_stamped_ref(ctx, op.pc, callable_op, callable)?;
    walker_emit_exact_dict_hit(ctx, op.pc, dict_op, r_args[2], dict, hit, dst, 'r')
}

/// `isinstance(x, C)` for an ordinary class `C` — the trace shape of
/// `typeobject.py` `issubtype`:
///
/// ```text
/// def issubtype(self, w_type):
///     promote(self); promote(w_type)
///     if we_are_jitted():
///         version_tag1 = self.version_tag()
///         version_tag2 = w_type.version_tag()
///         if version_tag1 is not None and version_tag2 is not None:
///             return _pure_issubtype(self, w_type, version_tag1, version_tag2)
///     return _issubtype(self, w_type)
/// ```
///
/// Both types are promoted and the elidable result is keyed on BOTH version
/// tags, so this pins both: the receiver's class through `guard_class` plus the
/// exact-`w_class` guard, `C` through a `guard_value`, and each type's
/// `_version_tag?` through one quasi-immutable marker rather than a load and a
/// `guard_value` per read. [`pyre_interpreter::mutated`] recursively
/// invalidates subclasses, so the receiver's watcher also covers a base class
/// changing its dict or its bases. What remains is a green answer.
///
/// A metaclass other than exactly `type` runs its own `__instancecheck__` —
/// `ABCMeta` among them, whose `register` changes the answer without touching
/// either pinned class — and declines here. So do a tuple or union classinfo,
/// which are not types at all, and a type with no version tag to watch.
///
/// `abstract_isinstance_w` additionally reads `w_inst.__class__` on a MISS
/// (`abstractinst.py`), which a `@property def __class__` turns into user code.
/// [`pyre_interpreter::baseobjspace::isinstance_miss_class_lookup_is_pure`] is
/// the memoized proof that the receiver's type inherits both
/// `object.__getattribute__` and the canonical `object.__class__` descriptor;
/// a miss without it declines. `dispatch_residual_call_iRd_kind`'s replay-safe
/// classification already admits this exact shape under the same three
/// predicates; the version-tag markers are what let the answer be baked rather
/// than re-derived.
///
/// Read-only, so no sub-walk restriction: it cannot raise and introduces no
/// side effect a resume would repeat.
pub(crate) fn try_walker_specialize_builtin_isinstance<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, obj, classinfo)` shape only.
    let Some((concrete_callable, [obj, classinfo])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 2)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_isinstance_function(concrete_callable) {
        return Ok(None);
    }
    // The class the answer comes from has to be the one the guards pin, so read
    // it out of the slot `walker_guard_exact_w_class` compares and require
    // `typedef::type` to agree: an exception instance's `ExcKind` tag names a
    // class the slot does not, and the MRO walked at record time would then
    // belong to a class no guard holds.
    let w_class = unsafe { (*obj).w_class };
    if w_class.is_null()
        || pyre_interpreter::typedef::r#type(obj)
            .is_none_or(|actual| !std::ptr::eq(actual.as_ptr(), w_class))
    {
        return Ok(None);
    }
    let metaclass_is_type = pyre_interpreter::typedef::r#type(classinfo)
        .is_some_and(|meta| std::ptr::eq(meta.as_ptr(), pyre_interpreter::typedef::w_type()));
    if !metaclass_is_type {
        return Ok(None);
    }
    let answer = unsafe { pyre_interpreter::baseobjspace::isinstance_w(obj, classinfo) };
    if !answer
        && !unsafe { pyre_interpreter::baseobjspace::isinstance_miss_class_lookup_is_pure(w_class) }
    {
        return Ok(None);
    }
    // Both markers need a live tag to attach to; a type that carries none has
    // no channel through which a later mutation reaches this trace.
    if unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) } == 0
        || unsafe { pyre_object::typeobject::w_type_get_version_tag(classinfo) } == 0
    {
        return Ok(None);
    }

    // emit the specialized IR (walker-native)
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let classinfo_const = ctx.trace_ctx.const_ref(classinfo as i64);
    walker_guard_fold_callable(ctx, op.pc, r_args[3], classinfo)?;
    let obj_op = r_args[2];
    walker_guard_exact_instance(
        ctx,
        op.pc,
        obj_op,
        unsafe { (*obj).ob_type } as i64,
        w_class,
    )?;
    let w_class_const = ctx.trace_ctx.const_ref(w_class as i64);
    walker_pin_type_version_tag(ctx, op.pc, w_class_const)?;
    walker_pin_type_version_tag(ctx, op.pc, classinfo_const)?;

    let result = ctx
        .trace_ctx
        .const_ref(pyre_object::boolobject::w_bool_from(answer) as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;
    Ok(Some(()))
}

/// Fold plain `getattr(type, name)` when
/// [`pyre_interpreter::type_attr_value_fast_path`] resolves
/// `typeobject.py` `getattribute`'s `space.get(w_value, w_None, self)`.  The exact
/// callable, exact receiver, exact name object, and receiver version are pinned
/// before the value is written as a green constant.  Pinning the callable makes
/// a rebound `getattr` side-exit instead of continuing to use the folded value.
/// The operand guards are tautologies when their inputs are already constants
/// and disappear during optimization.
/// [`pyre_interpreter::mutated`] recursively invalidates subclasses, so the
/// receiver's one quasi-immutable version watcher covers base-class mutation
/// and emits no per-iteration operations.
///
/// Like the `len` fold this is safe in an inlined callee sub-walk: the oracle
/// proves a read-only present attribute, so it cannot raise or introduce a
/// side effect that resume would repeat.  Every other shape declines before
/// emitting IR and falls through to the generic residual.
pub(crate) fn try_walker_specialize_builtin_type_getattr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, obj, name)` shape only.
    let Some((concrete_callable, [concrete_obj, concrete_name])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 2)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_getattr_function(concrete_callable) {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_exact_type(concrete_name, &pyre_object::pyobject::STR_TYPE) } {
        return Ok(None);
    }
    let name = unsafe { pyre_object::w_str_get_wtf8(concrete_name) };
    let Some((w_type, _version_tag, w_value, binding)) =
        (unsafe { pyre_interpreter::type_attr_value_fast_path(concrete_obj, name) })
    else {
        return Ok(None);
    };

    walker_guard_stamped_ref(ctx, op.pc, r_args[0], concrete_callable)?;

    let w_type_const = walker_guard_stamped_ref(ctx, op.pc, r_args[2], w_type)?;

    // The baked WTF-8 bytes remain constant only while this exact string is
    // the name operand.  Constant operands make this guard a removable
    // tautology, so it costs nothing in the steady loop.
    let name_ref = r_args[3];
    walker_guard_stamped_ref_pin(ctx, op.pc, name_ref, concrete_name)?;

    // typeobject.py `promote(self.version_tag())`: this quasi-immutable watcher
    // emits no per-iteration op. `mutated` (baseobjspace.rs) recurses through
    // subclasses, so changing the attribute on any base invalidates this pin.
    walker_pin_type_version_tag(ctx, op.pc, w_type_const)?;
    walker_pin_type_attr_binding(ctx, op.pc, binding)?;

    let value_const = ctx.trace_ctx.const_ref(w_value as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', value_const)?;
    Ok(Some(()))
}

/// `getattr(obj, "name")` — the builtin spelling of the `LOAD_ATTR` fold.
///
/// `space.getattr` is the one operation both `obj.name` and this builtin
/// reach, so a constant `str` name admits exactly the instance-shape read
/// [`try_walker_specialize_load_attr`] already emits, and the two forms stay on
/// one implementation rather than drifting the way a fast-path pair can
/// (`getattr(obj, 'm')` versus `obj.m` is the classic discriminator).
///
/// The three-argument `getattr(obj, name, default)` stays on the residual: the
/// fold's map guard proves the attribute is *present*, which says nothing about
/// the branch that supplies the default.
pub(crate) fn try_walker_specialize_builtin_getattr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, obj, name)` shape only; the
    // three-argument form arrives one operand longer and declines here.
    let Some((concrete_callable, [concrete_obj, concrete_name])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 2)
    else {
        return Ok(None);
    };
    if !pyre_interpreter::builtins::is_builtin_getattr_function(concrete_callable) {
        return Ok(None);
    }
    // The name is rejected before any lookup unless it is a string, and the
    // resolved bytes below stay valid only while this exact string is the
    // operand.  Keep the WTF-8 view: PyPy's RPython string carries lone
    // surrogates through the same traced descriptor lookup as ASCII names.
    if !unsafe { pyre_object::is_exact_type(concrete_name, &pyre_object::pyobject::STR_TYPE) } {
        return Ok(None);
    }
    let name = unsafe { pyre_object::w_str_get_wtf8(concrete_name) };

    // Resolve the descriptor case before emitting anything.  A function-valued
    // class attribute is not a plain class-attribute read: getattr binds it to
    // the receiver.  PyPy traces that Method allocation and virtualizes it into
    // the following CALL, so reproduce the same guarded allocation here.
    let bound_method = unsafe {
        pyre_interpreter::baseobjspace::bound_method_attr_fast_path_wtf8(concrete_obj, name)
    };
    let bound_method = match bound_method {
        Some((w_type, version_tag, w_descr, owes_shadow_guard)) => {
            let shadow = if owes_shadow_guard {
                let Some(shadow) = (unsafe { walker_classify_shadow_guard(concrete_obj) }) else {
                    return Ok(None);
                };
                Some(shadow)
            } else {
                None
            };
            let Some(header) = super_attr_method_header(w_descr) else {
                return Ok(None);
            };
            Some((w_type, version_tag, w_descr, shadow, header))
        }
        None => None,
    };

    let pre_emit_pos = ctx.trace_ctx.get_trace_position();
    walker_guard_stamped_ref(ctx, op.pc, r_args[0], concrete_callable)?;
    let name_ref = r_args[3];
    walker_guard_stamped_ref_pin(ctx, op.pc, name_ref, concrete_name)?;

    if let Some((w_type, _version_tag, w_descr, shadow, header)) = bound_method {
        walker_emit_constant_descr_bound_method(
            ctx,
            op.pc,
            r_args[2],
            concrete_obj,
            w_type,
            w_descr,
            header,
            shadow,
            dst,
            'r',
            None,
        )?;
        return Ok(Some(()));
    }

    let Ok(name) = name.as_str() else {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    };

    // Every shape the read declines has to leave the trace as it found it: the
    // two guards above are the premise of a fold that is no longer there, and
    // the residual the caller falls through to recomputes the lookup from the
    // unguarded operands.
    if (try_walker_specialize_load_attr(ctx, op.pc, r_args[2], name, dst, 'r')?).is_none() {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    Ok(Some(()))
}

/// Record an overflow-checked machine-int operation and guard it.
///
/// [`record_int_ovf`] folds a both-constant operand pair to a constant without
/// recording anything, and `GuardNoOverflow` carries no operands — it reads the
/// flag of the operation immediately before it — so an unconditional guard
/// after a folded pair would attach to whatever was recorded last instead.
/// `None` means the operation cannot be represented and the caller must rewind.
fn record_int_ovf_guarded<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    opcode: OpCode,
    b1: OpRef,
    b2: OpRef,
) -> Result<Option<OpRef>, DispatchError> {
    let (result, overflow) = record_int_ovf(ctx, op_pc, opcode, b1, b2, None)?;
    if overflow {
        return Ok(None);
    }
    if !result.is_constant() {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;
    }
    Ok(Some(result))
}

/// Fall back to the opaque `range` residual from a decline point that the
/// specializer only reaches after it has already emitted.
///
/// Every decline in `try_walker_specialize_builtin_range` past `pre_emit_pos`
/// has to rewind: the callable `GuardValue`, the per-bound class guards and
/// `intval` reads, and — for a bound converted by a user `__index__` — that
/// callee's whole inlined body sit in the trace, and the residual the caller
/// falls through to recomputes all of it.  Leaving them behind would pair the
/// residual with guards for a specialization that no longer exists.
fn walker_range_decline<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pre_emit_pos: majit_metainterp::recorder::TracePosition,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
    ctx.trace_ctx.heap_cache_mut().reset();
    Ok(None)
}

/// Emit the machine-int trace of `functional.py compute_range_length`
/// for a path whose converted bounds all fit signed machine words.  Each
/// source conditional becomes the guard chosen by the recording values; the
/// overflow guards side-exit to the interpreter's wrapped-int implementation.
fn walker_emit_range_length<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    start: OpRef,
    stop: OpRef,
    step: OpRef,
    concrete_start: i64,
    concrete_stop: i64,
    concrete_step: i64,
) -> Result<Option<OpRef>, DispatchError> {
    if concrete_step == 0 {
        return Ok(None);
    }
    let (normalized_start, normalized_stop, normalized_step) = if concrete_step < 0 {
        let Some(step) = concrete_step.checked_neg() else {
            return Ok(None);
        };
        (concrete_stop, concrete_start, step)
    } else {
        (concrete_start, concrete_stop, concrete_step)
    };
    let concrete_length = if normalized_start < normalized_stop {
        let Some(diff) = normalized_stop
            .checked_sub(normalized_start)
            .and_then(|diff| diff.checked_sub(1))
        else {
            return Ok(None);
        };
        let Some(length) = (diff / normalized_step).checked_add(1) else {
            return Ok(None);
        };
        length
    } else {
        0
    };

    let zero = ctx.trace_ctx.const_int(0);
    let one = ctx.trace_ctx.const_int(1);
    let step_has_recorded_sign = if concrete_step < 0 {
        ctx.trace_ctx.record_op(OpCode::IntLt, &[step, zero])
    } else {
        ctx.trace_ctx.record_op(OpCode::IntGt, &[step, zero])
    };
    ctx.trace_ctx
        .set_opref_concrete(step_has_recorded_sign, majit_ir::Value::Int(1));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[step_has_recorded_sign])?;
    let (lo, hi, positive_step) = if concrete_step < 0 {
        let Some(negated) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntSubOvf, zero, step)?
        else {
            return Ok(None);
        };
        (stop, start, negated)
    } else {
        (start, stop, step)
    };

    let nonempty = ctx.trace_ctx.record_op(OpCode::IntLt, &[lo, hi]);
    ctx.trace_ctx.set_opref_concrete(
        nonempty,
        majit_ir::Value::Int((normalized_start < normalized_stop) as i64),
    );
    if normalized_start >= normalized_stop {
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[nonempty])?;
        return Ok(Some(zero));
    }
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[nonempty])?;

    let Some(span) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntSubOvf, hi, lo)? else {
        return Ok(None);
    };
    let Some(diff) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntSubOvf, span, one)? else {
        return Ok(None);
    };
    // `//` through the same `ll_int_py_div` pure call the `BINARY_OP` integer
    // fold emits.  `OpCode::IntFloorDiv` would be the direct spelling, but the
    // dynasm backend carries a regalloc arm for it and no assembler arm on
    // either architecture, so it lowers to nothing and the destination keeps
    // an operand: this quotient came back as its own dividend for a positive
    // step and as the divisor for a negative one.  Both operands are
    // non-negative here (`positive_step` is guarded above and `diff` follows
    // the ordering guard), where floor and truncation agree, so the two
    // spellings compute the same value.
    let (quotient, _) = walker_emit_int_py_div_or_mod(
        ctx,
        diff,
        positive_step,
        normalized_stop - normalized_start - 1,
        normalized_step,
        true,
    );
    let Some(length) = record_int_ovf_guarded(ctx, op_pc, OpCode::IntAddOvf, quotient, one)? else {
        return Ok(None);
    };
    ctx.trace_ctx
        .set_opref_concrete(length, majit_ir::Value::Int(concrete_length));
    Ok(Some(length))
}

/// `range(stop)` / `range(start, stop)` / `range(start, stop, step)` with
/// exact canonical machine-word ints or strict inlinable user `__index__`
/// conversions: lower the opaque constructor residual
/// to a virtual `W_Range` and four virtual wrapped-int fields.  This lets the
/// existing GET_ITER specialization consume the range without forcing either
/// allocation.  All other callables and argument shapes fall through to the
/// generic residual.
pub(crate) fn try_walker_specialize_builtin_range<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if !(3..=5).contains(&r_args.len()) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    let range_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::functional::RANGE_TYPE);
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !std::ptr::eq(concrete_callable, range_type)
    {
        return Ok(None);
    }

    let exact_int_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    enum BoundPlan {
        Exact {
            op: OpRef,
            concrete: pyre_object::PyObjectRef,
        },
        UserIndex(IndexInlineCandidate),
    }
    let mut plans = Vec::with_capacity(r_args.len() - 2);
    for (&arg_op, concrete) in r_args[2..].iter().zip(&arg_concretes[2..]) {
        let ConcreteValue::Ref(arg_obj) = *concrete else {
            return Ok(None);
        };
        if walker_is_exact_machine_int_concrete(arg_obj) {
            plans.push(BoundPlan::Exact {
                op: arg_op,
                concrete: arg_obj,
            });
        } else if let Some(candidate) = prepare_walker_inline_index(ctx, arg_op, arg_obj) {
            plans.push(BoundPlan::UserIndex(candidate));
        } else {
            return Ok(None);
        }
    }

    // Every non-int bound has been resolved and statically preflighted before
    // this first emission.  `functional.py W_Range.descr_new` applies
    // `space.index` independently to start/stop/step; mirror that order and
    // retain each returned box as an intermediate feeding the constructor.
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let mut concrete_args = Vec::with_capacity(plans.len());
    let mut concrete_values = Vec::with_capacity(plans.len());
    let mut raw_args = Vec::with_capacity(plans.len());
    for plan in plans {
        let (arg_op, arg_obj) = match plan {
            BoundPlan::Exact { op, concrete } => (op, concrete),
            BoundPlan::UserIndex(candidate) => {
                let Some((result, ConcreteValue::Ref(concrete))) = try_walker_inline_index(
                    ctx, op, code, funcptr, r_args, call_descr, dst, candidate,
                )?
                else {
                    return walker_range_decline(ctx, pre_emit_pos);
                };
                (result, concrete)
            }
        };
        // A trace-constant bound carries its class in the constant itself, so
        // record the class as known without proving it: `walker_guard_class`
        // would emit a `GuardClass` that can never fail plus the tagged-operand
        // low-bit test that guards a later entry's untagged arrival, and a
        // constant has no later arrival.  A bound returned by an inlined
        // `__index__` is live and takes the full guard.
        if arg_op.is_constant() {
            ctx.trace_ctx.heap_cache_mut().class_now_known(arg_op);
        } else {
            walker_guard_class(ctx, op.pc, arg_op, int_type_addr)?;
        }
        walker_guard_exact_w_class(ctx, op.pc, arg_op, exact_int_class)?;
        let concrete_value = unsafe { pyre_object::w_int_get_value(arg_obj) };
        let raw = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            arg_op,
            crate::descr::int_intval_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(concrete_value));
        concrete_args.push(arg_obj);
        concrete_values.push(concrete_value);
        raw_args.push(raw);
    }

    // Run only the remaining builtin range body on the converted exact ints;
    // executing the original arguments here would call user `__index__` a
    // second time during recording.
    let authentic_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &concrete_args)
    };
    if concrete_values.len() == 3 && concrete_values[2] == 0 {
        let Err(mut err) = authentic_result else {
            return walker_range_decline(ctx, pre_emit_pos);
        };
        let exc = err.to_exc_object();
        let kind = pyre_object::interp_exceptions::ExcKind::ValueError;
        if !walker_recorded_builtin_raise_is_supported(exc, kind) {
            return walker_range_decline(ctx, pre_emit_pos);
        }
        let Some(ec) = walker_ensure_execution_context(ctx) else {
            return walker_range_decline(ctx, pre_emit_pos);
        };

        let step_raw = raw_args[2];
        let zero = ctx.trace_ctx.const_int(0);
        let is_zero = ctx.trace_ctx.record_op(OpCode::IntEq, &[step_raw, zero]);
        ctx.trace_ctx
            .set_opref_concrete(is_zero, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[is_zero])?;
        return Ok(Some(walker_emit_recorded_builtin_raise(ctx, ec, exc, kind)));
    }
    let Ok(authentic_range) = authentic_result else {
        return walker_range_decline(ctx, pre_emit_pos);
    };
    let (authentic_start, authentic_stop, authentic_step) =
        unsafe { pyre_object::functional::w_range_fields(authentic_range) };
    let authentic_length = unsafe { pyre_object::functional::w_range_length(authentic_range) };
    let authentic_fields = [
        authentic_start,
        authentic_stop,
        authentic_step,
        authentic_length,
    ];
    if authentic_fields.iter().any(|&field| unsafe {
        !std::ptr::eq((*field).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*field).w_class, exact_int_class)
    }) {
        return walker_range_decline(ctx, pre_emit_pos);
    }
    let concrete_fields =
        authentic_fields.map(|field| unsafe { pyre_object::w_int_get_value(field) });
    let [
        concrete_start,
        concrete_stop,
        concrete_step,
        concrete_length,
    ] = concrete_fields;

    let zero = ctx.trace_ctx.const_int(0);
    let one = ctx.trace_ctx.const_int(1);
    let (start, stop, step) = match raw_args.as_slice() {
        [stop] => (zero, *stop, one),
        [start, stop] => (*start, *stop, one),
        [start, stop, step] => (*start, *stop, *step),
        _ => unreachable!("range arity gate admitted an invalid argument count"),
    };
    // Trace-constant bounds retain the existing zero-op length.  Every live
    // bound -- a local read, a loop-carried box, the box a user `__index__`
    // returned -- follows `compute_range_length`, so its value feeds all four
    // virtual fields instead of being paired with a stale record-time length.
    // Where the bound came from is not a property the emitted arithmetic
    // depends on: it reads the three raw ints and guards the step's sign and
    // the start/stop ordering it recorded, which is what a live bound needs
    // whether an `__index__` produced it or a `LOAD_FAST` did.
    //
    // A range whose emptiness alternates across iterations fails the ordering
    // guard and takes a bridge, which is the cost this branch carries and the
    // reason the shape used to be left to the residual.  The residual is not
    // cheaper: `range(n)` for a local `n` records a `MayForce` call that
    // allocates the range for real, forces the virtualizable, and re-reads the
    // four fields through class guards -- 10006 instructions an iteration in a
    // hot loop, against 24 for the same loop under pypy.
    let length = if start.is_constant() && stop.is_constant() && step.is_constant() {
        ctx.trace_ctx.const_int(concrete_length)
    } else {
        let Some(length) = walker_emit_range_length(
            ctx,
            op.pc,
            start,
            stop,
            step,
            concrete_start,
            concrete_stop,
            concrete_step,
        )?
        else {
            return walker_range_decline(ctx, pre_emit_pos);
        };
        length
    };

    let new = ctx
        .trace_ctx
        .execute_new_with_vtable(crate::descr::w_range_size_descr());

    let field_descrs = [
        crate::descr::range_start_descr(),
        crate::descr::range_stop_descr(),
        crate::descr::range_step_descr(),
        crate::descr::range_length_descr(),
    ];
    let raw_fields = [start, stop, step, length];
    for (((descr, raw), concrete_value), authentic_field) in field_descrs
        .into_iter()
        .zip(raw_fields)
        .zip(concrete_fields)
        .zip(authentic_fields)
    {
        let boxed = crate::state::wrapint(ctx.trace_ctx, raw);
        ctx.trace_ctx.set_opref_concrete(
            boxed,
            majit_ir::Value::Ref(majit_ir::GcRef(authentic_field as usize)),
        );
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new, boxed], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new, descr_index, boxed);
    }

    // `descr_new`'s `promote_step` — a property of the call shape, not of the
    // bounds, so it is a constant here.  `descr_iter` reads it to pick the
    // iterator shape; leaving it unwritten would hand a forced range an
    // uninitialized byte.
    let promote_step = ctx.trace_ctx.const_int((raw_args.len() != 3) as i64);
    let promote_step_descr = crate::descr::range_promote_step_descr();
    let promote_step_index = promote_step_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[new, promote_step],
        promote_step_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(new, promote_step_index, promote_step);

    let range_type_addr = &pyre_object::functional::RANGE_TYPE as *const _ as i64;
    ctx.trace_ctx.heap_cache_mut().class_now_known(new);
    ctx.trace_ctx.set_opref_concrete(
        new,
        majit_ir::Value::Ref(majit_ir::GcRef(authentic_range as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new)?;
    Ok(Some(DispatchOutcome::Continue))
}

/// Cut the trace from INSIDE the `locals()` expansion once the recorded
/// length crosses `trace_limit`.
///
/// `pyjitpl.py` `MetaInterp._interpret` asks `blackhole_if_trace_too_long()`
/// after every jitcode step, so the `@jit.unroll_safe` `fast2locals` it looks
/// into is interrupted between its own steps and the raise leaves a partly
/// filled mapping behind.  Both arms here record the whole unroll inside ONE
/// Python opcode and `mod.rs` asks only once that opcode returns, so with no
/// cut the overshoot is the frame's own `co_nlocals` and nothing bounds it.
///
/// The cut is upstream's, not a refusal standing in for it: nothing is
/// estimated, the same `history.length() > trace_limit` decides, and what
/// happens on a yes is the abort `mod.rs` performs one opcode later --
/// `latch_abort_blackhole`, `note_root_trace_too_long`
/// and the reason-bearing `TraceTooLong` unwind. The trace is
/// discarded whole, so the half-built expansion above this point is never
/// published; resuming re-executes the opcode from `pc`, which is the position
/// every guard this expansion emits already side-exits to
/// (`walker_emit_fold_guard_with_snapshot`), and the concrete mapping the fold
/// built before emitting is a pure function of the fastlocals, so the residual
/// redoes it with the same outcome.
///
/// `trace_too_long_abort_safe`'s bar is kept: with effects already executed and
/// no blackhole image to hand them to,
/// `run_blackhole_interp_to_cancel_tracing` has nowhere to resume, so the walk
/// goes on recording exactly as it does there.
/// Name the gate a `locals()` / `vars()` / `dir()` expansion declined at, so a
/// census over a corpus attributes every refusal to one gate rather than
/// re-deriving it by reading.  `arm` distinguishes the standard-virtualizable
/// expansion from the inlined-callee one, which decline for different reasons.
fn locals_expansion_declined(arm: &str, why: &str) {
    if fbw_debug_abort_enabled() {
        eprintln!("[decline-why] LOCALS-{arm} {why}");
    }
}

fn locals_expansion_cut_if_too_long<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if !ctx.trace_ctx.is_too_long() {
        return Ok(());
    }
    let latched = residual_call::latch_abort_blackhole(ctx, pc, "locals-expansion");
    if !latched && super::fbw_state::fbw_executed_effect_count() != 0 {
        majit_metainterp::mc_diag_bump(26);
        return Ok(());
    }
    let ops = ctx.trace_ctx.num_recorded_ops();
    crate::state::note_root_trace_too_long(
        ctx.trace_ctx.current_merge_points_first_green_key_pair(),
        ctx.trace_ctx.resumekey_original_loop_token().cloned(),
    );
    ctx.session.borrow_mut().trace_too_long = true;
    Err(DispatchError::TraceTooLong { pc, ops })
}

/// Which builtin [`try_walker_specialize_builtin_locals`] is standing in for.
///
/// All three resolve their frame through the same `topframe_for_locals` and
/// read the same fastlocals, so one modelled expansion serves them; they
/// differ only in what they make of the resulting mapping.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameLocalsBuiltin {
    /// `locals()` / `vars()` — the mapping itself is the result.
    Mapping,
    /// `dir()` — the mapping's sorted key set is the result.
    SortedNames,
}

/// One localsplus slot of the portal frame that the modelled `fast2locals`
/// reproduces.
struct PortalLocalSlot {
    /// The value the slot's key is bound to -- the slot itself for a plain
    /// fastlocal, `Cell.contents` for a cell slot.  `PY_NULL` when the name is
    /// unbound and `fast2locals` deletes it instead.
    value: pyre_object::PyObjectRef,
    /// Whether the slot holds a `Cell` whose contents its key takes.
    cell: bool,
}

/// `jit_locals_dict_setitem_local` / `_cell`: `(dict, code, index, value)`
/// answering the (possibly reallocated) mapping, or null on failure.
type LocalsDictSetitem = extern "C" fn(
    pyre_object::PyObjectRef,
    *const pyre_interpreter::CodeObject,
    i64,
    pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef;

/// The `fast2locals` binder for localsplus slot `index` and the index it names
/// its key with: `code.varnames[index]` below `numlocals` -- a cellvar sharing
/// a varname slot included, since that is the name it carries -- and
/// `cell_slot_names(code)[index - numlocals]` above it.
///
/// The same split [`ModelledLocalSlot::binder`] makes for an inlined callee.
fn portal_slot_binder(index: usize, numlocals: usize) -> (LocalsDictSetitem, i64) {
    if index < numlocals {
        (
            pyre_interpreter::pyframe::jit_locals_dict_setitem_local,
            index as i64,
        )
    } else {
        (
            pyre_interpreter::pyframe::jit_locals_dict_setitem_cell,
            (index - numlocals) as i64,
        )
    }
}

/// Zero-argument `locals()` / `vars()` / `dir()` on the walk's own portal
/// frame: model
/// `pyframe.py fast2locals` in the trace instead of residualizing
/// `interp_inspect.py locals` → `pyframe.py getdictscope`.
///
/// `fast2locals` is `@jit.unroll_safe`, and `policy.py:60-67` cancels
/// `contains_loop` for unroll_safe graphs, so upstream LOOKS INSIDE it: each
/// `self.locals_cells_stack_w[i]` lowers to `getarrayitem_vable_r`
/// (`jtransform.py do_fixed_list_getitem`), answered from
/// `metainterp.virtualizable_boxes`, and `jtransform.py
/// rewrite_op_jit_force_virtualizable` returns `[]` for a read the tracer is
/// inside.  There is no residual and no virtualizable force anywhere on the
/// upstream locals-read path.
///
/// Pyre residualizes the same read as one opaque `bh_call_fn(locals, PY_NULL)`
/// `CallMayForce`, which arms `virtualizable.py:281-291
/// force_virtualizable_if_necessary` for the whole call; the read barrier
/// `force_frame_before_locals_read` then clears `TOKEN_TRACING_RESCALL` and
/// `tracing_after_residual_call` reads that clear as an escape
/// (`VableEscapedDuringResidualCall`), losing the loop.  The deviation is the
/// residual BOUNDARY, not the barrier — `rvirtualizable.py hook_access_field` injects the
/// same hook on reads upstream and `pyjitpl.py vable_after_residual_call` aborts
/// unconditionally on a detected force — so this removes the boundary and
/// leaves the barrier live for every shape it declines.
///
/// Emitted shape, mirroring `pyframe.py`: `guard_value(callable)`;
/// `getorcreatedebug()` (the `debugdata` virtualizable field, answered from
/// `virtualizable_boxes`) followed by `getfield_gc_r(w_locals)` under the
/// guard that pins which of the two mapping arms this is; one
/// `getarrayitem_vable_r(frame, ConstInt(i))` per localsplus slot (the same
/// lowering `emit_load_fast_ref!` already emits for LOAD_FAST); for a cell
/// slot a `guard_class(Cell)` and one `getfield_gc_r(contents)` on top of it,
/// which is the whole of `fast2locals`' cell half (pyframe.py); a
/// `guard_isnull` / `guard_nonnull` pinning the bound-ness of whichever of the
/// two the key takes; THEN the mapping itself, under its exact-dict guard on
/// the frame-owned arm; and a plain non-forcing `Call` per slot —
/// `setitem_str` when bound, `delitem` when not.  Every read and every guard
/// precedes the mapping, so a guard that fails allocates nothing and leaves no
/// half-rewritten mapping for the residual to redo.  None of those ops can
/// reach `force_frame`, so nothing arms the vable protocol.
///
/// The mapping is the FRAME's whenever the frame already carries one:
/// `fast2locals` rewrites only the varname keys, so a foreign key — one an
/// `f_locals` write put there (PEP 667) — survives every call, and an
/// expansion that always started from an empty dict would drop it for as long
/// as the loop stayed compiled.  A frame that carries none keeps the empty
/// `newdict` (pyframe.py) the residual would have materialised, under a
/// `guard_isnull` that side-exits if one appears mid-loop; nothing else
/// references that dict, so it is already the independent copy
/// `frame_locals_snapshot` hands back and its `delitem` arm is a no-op.
///
/// One further non-forcing `Call` turns that mapping into the published
/// result: `jit_locals_dict_snapshot` (the independent PEP 667 copy
/// `frame_locals_snapshot` builds) for `locals()` / `vars()`, and
/// `jit_dir_names_from_locals` (the split-out tail of `builtin_dir`'s
/// no-argument path, which reads `getdictscope` rather than the copy) for
/// `dir()`.  Both take the mapping and not the frame, so they too cannot
/// reach `force_frame`.
///
/// An inline level answers from its own frame model instead
/// ([`try_walker_specialize_builtin_locals_in_callee`]), because the frame
/// `locals()` reports on there is the callee's and a trace has exactly one
/// standard virtualizable.
///
/// Returns `None` (fall through to the generic residual, SAFE — exactly
/// today's behaviour) for every other shape: a rebound `locals` / `vars` /
/// `dir` name,
/// a bound receiver, any argument, a frame that is not the
/// standard virtualizable the boxes describe, a hidden top frame, a
/// non-OPTIMIZED (module / class / exec) frame, a `CO_FAST_HIDDEN` slot, a
/// slot the shadow cannot answer with a Ref, a shadow whose mapping is not the
/// frame's, and a frame-owned mapping that is not an exact dict.
///
/// Cellvars and freevars are modelled rather than declined; what their band
/// still declines is a cell slot not holding a `Cell` — unreachable past an
/// OPTIMIZED frame's `MAKE_CELL` / `COPY_FREE_VARS` prologue, and the same
/// answer the callee arm gives at `cell-slot-not-a-cell` — and an unbound one
/// on the frame-owned arm, which would need a `delitem` naming its key through
/// `cell_slot_names` where `jit_locals_dict_delitem_local` names it through
/// `varnames`.
pub(crate) fn try_walker_specialize_builtin_locals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    macro_rules! decline {
        ($why:literal) => {{
            locals_expansion_declined("PORTAL", $why);
            return Ok(None);
        }};
    }
    // Plain zero-argument `bh_call_fn(callable, PY_NULL)` shape only.
    // These four are "not this fold", the same silent `Ok(None)` `super()`
    // uses for a non-matching CALL.  Naming them LOCALS-PORTAL would
    // attribute every residual CallFn in the corpus to this arm.
    let Some((concrete_callable, _)) = plain_builtin_call_concretes(ctx, code, op, r_args, 0)
    else {
        return Ok(None);
    };
    // `vars()` with no argument delegates straight to `builtin_locals`
    // (`app_inspect.py`), so both names share the fold; `vars(obj)` and
    // `dir(obj)` carry an extra operand and are already excluded by the arity
    // gate.
    let fold = if pyre_interpreter::builtins::is_builtin_locals_function(concrete_callable)
        || pyre_interpreter::builtins::is_builtin_vars_function(concrete_callable)
    {
        FrameLocalsBuiltin::Mapping
    } else if pyre_interpreter::builtins::is_builtin_dir_function(concrete_callable) {
        FrameLocalsBuiltin::SortedNames
    } else {
        return Ok(None);
    };
    // Inside an inline sub-walk or an inlined callee body, the frame
    // `gettopframe_nohidden()` resolves is the CALLEE's, which the expansion
    // below cannot answer for: it reads from the standard virtualizable, and a
    // trace has exactly one of those.  `MIFrame._nonstandard_virtualizable`
    // tests against `metainterp.virtualizable_boxes[-1]`, and
    // `_opimpl_recursive_call` / `perform_call` push a MIFrame without
    // rebinding those boxes, so upstream an inlined callee's frame is an
    // ordinary virtual and its `fast2locals` traces through reading that
    // virtual's fields.  Pyre's counterpart of the virtual is the level's own
    // [`CalleeLocalsShadow`], so the callee arm answers from it; every gate it
    // fails declines to the generic residual, exactly as this refusal did.
    if ctx.fbw_mode.inline_subwalk || current_inline_concrete_frame() != 0 {
        return try_walker_specialize_builtin_locals_in_callee(
            ctx,
            op,
            fold,
            r_args,
            concrete_callable,
            dst,
        );
    }
    let (Some(vable_op), Some(vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        decline!("no-standard-virtualizable");
    };
    // The frame `locals()` reports on is `ec.gettopframe_nohidden()`
    // (`interp_inspect.py`).  Resolve it the same way and require it to BE
    // the standard virtualizable: a hidden portal frame, or any deeper frame
    // handed out through the backref chain, resolves elsewhere and declines.
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        decline!("no-exec-context");
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != vable_ptr {
        decline!("top-frame-not-the-virtualizable");
    }
    let frame_ref = unsafe { &*frame };
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(frame_ref) };
    let code_obj = unsafe { &*code_ptr };
    if !pyre_interpreter::PyFrame::code_locals_are_modelled_fastlocals(code_obj) {
        decline!("not-modelled-fastlocals");
    }
    let numlocals = code_obj.varnames.len();
    // `fast2locals` binds one key per localsplus slot: `varnames` below
    // `numlocals`, then the pure cellvars followed by the freevars, which is
    // what `cell_slot_names` names and the order `fast2locals` walks them in.
    //
    // A cell slot holds the `Cell` and its key takes `Cell.contents` -- one
    // `GETFIELD_GC_R` over a slot the vable read already holds, which is the
    // whole of `fast2locals`' cell half.  Modelling that read is what lets this
    // arm ask `code_locals_are_modelled_fastlocals` -- the question the callee
    // arm already asks -- instead of the narrower one that also required the
    // cellvar and freevar lists to be empty.  Under that narrower question a
    // closure's frame answered `locals()` from the opaque residual, which
    // forces the virtualizable and aborts the trace with ABORT_ESCAPE
    // (`vable_after_residual_call`).  A `varnames` slot that `MAKE_CELL`
    // turned into a cell is in the band below `numlocals` and keeps its
    // `varnames` name, exactly as `fast2locals` reads it.
    let nslots = numlocals + pyre_interpreter::PyFrame::cell_slot_names(code_obj).count();
    let is_cell_slot = |slot: usize| {
        slot >= numlocals
            || (slot < code_obj.localspluskinds.len()
                && code_obj.localspluskinds[slot] & pyre_interpreter::bytecode::CO_FAST_CELL != 0)
    };
    // No width ceiling.  `pyframe.py` `PyFrame.fast2locals` carries
    // `@jit.unroll_safe` and no ceiling of its own, and the length question is
    // `trace_limit`'s.  A fixed ceiling of 32 slots asked a different one, so a
    // frame answered `locals()` from a residual because of its own local count
    // while a trace many times longer was recorded beside it.
    //
    // No preflight takes its place either.  Upstream has nothing in that
    // place: `blackhole_if_trace_too_long` runs from `MetaInterp._interpret`
    // AFTER `run_one_step`, and no refusal on an estimated cost appears
    // anywhere in that path.  One refusing when `recorded + 2 * nslots >
    // trace_limit` was written and dropped, because near the limit it turned a
    // read that would have fitted into the forcing residual.
    //
    // The length question is answered where upstream answers it -- on the
    // recorded length, inside the unroll.  Upstream's step is one jitcode, so
    // the `fast2locals` it looks into is interrupted inside its own loop; this
    // fold is one Python opcode, so `locals_expansion_cut_if_too_long` runs
    // that same check per slot and aborts the trace from there rather than
    // leaving the overshoot to `mod.rs` one opcode later.
    // `locals_cells_stack_w` is PyFrame's only virtualizable array
    // (`virtualizable_gen.rs arrays`), so array index 0 names it.
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        decline!("no-virtualizable-info");
    };
    let Some(lengths) = ctx
        .trace_ctx
        .virtualizable_array_lengths()
        .map(<[usize]>::to_vec)
    else {
        decline!("no-virtualizable-array-lengths");
    };
    if info.num_arrays() != 1 || lengths.first().copied().unwrap_or(0) < nslots {
        decline!("vable-array-shape-too-narrow");
    }
    let (Some(fdescr), Some(adescr)) = (
        info.array_field_descrs().first().cloned(),
        info.array_descrs.first().cloned(),
    ) else {
        decline!("no-vable-array-descrs");
    };
    // `fast2locals` opens on `self.getorcreatedebug()` (pyframe.py) and
    // writes into ITS `w_locals`: the mapping is the FRAME's, carried across
    // calls, so a key written through `f_locals` outlives every `fast2locals`
    // that does not name it.  `debugdata` is a virtualizable field, so the read
    // answers from `virtualizable_boxes` and records no op — exactly like the
    // slot reads below — and the frame never becomes an operand, so nothing
    // here can reach `force_frame`.
    let Some((debugdata_op, majit_ir::Value::Ref(debugdata_ref))) = ctx
        .trace_ctx
        .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
    else {
        decline!("no-debugdata-vable-entry");
    };
    // Read the mapping through the SHADOW's payload, which is what the emitted
    // `getfield_gc_r` reads, and require it to be the one the residual would
    // have used.  The two payloads are not the same object: a root portal seed
    // bakes the vable identity against the live frame but expands the shadow
    // from the `snapshot_for_tracing` copy, whose `clone_debugdata_ptr` hands
    // out a fresh `FrameDebugData` around the same `w_locals`.  Comparing the
    // holders would decline every portal trace; comparing the mapping is the
    // invariant that actually has to hold.
    let shadow_debugdata =
        debugdata_ref.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
    let w_locals = if shadow_debugdata.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*shadow_debugdata).w_locals }
    };
    if !std::ptr::eq(w_locals, frame_ref.get_w_locals()) {
        decline!("shadow-mapping-not-the-frames");
    }
    // `f_extra_locals` is the OTHER half of what the residual reports.  A
    // proxy write whose key names no writable fast local lands there
    // (`framelocalsproxy_setitem`) and sets NEITHER `w_locals` nor a slot, so
    // the mapping this fold rebuilds from slots alone would silently drop the
    // key.  Read through the same shadow payload as `w_locals`; a holder
    // mismatch declines.  A present extras dict is merged AFTER the slot chain
    // and skips a key the slots already bound, which is the order and the
    // precedence `frame_locals_proxy_snapshot` gives the two halves; the null
    // direction is pinned by a guard below so a write mid-loop side-exits
    // instead.
    let w_extra_locals = if shadow_debugdata.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*shadow_debugdata).w_extra_locals }
    };
    if !std::ptr::eq(w_extra_locals, frame_ref.get_extra_locals()) {
        decline!("shadow-extra-not-the-frames");
    }
    let extras_present = !w_extra_locals.is_null();
    // Two shapes, each pinned by a guard so the compiled loop side-exits when
    // the frame moves to the other one:
    //
    // * the frame already carries its mapping — rewrite THAT, so a foreign key
    //   an `f_locals` write left in it survives, as it does across the
    //   residual's `fast2locals`;
    // * the frame carries none — `fast2locals` would materialise an empty dict
    //   (pyframe.py `d.w_locals = space.newdict()`) and fill it from
    //   the fastlocals, and the expansion builds exactly that dict instead of
    //   modelling the store.  Nothing else references it, so it is already the
    //   independent copy `frame_locals_snapshot` would hand back, and a
    //   `delitem` on a key it never held is a no-op.
    //
    // The slot helpers are dict-keyed, so a frame-owned mapping that is not an
    // exact dict declines.
    let canonical_dict = pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE);
    let frame_owned = !w_locals.is_null();
    if frame_owned
        && (canonical_dict.is_null()
            || !unsafe {
                std::ptr::eq((*w_locals).ob_type, &pyre_object::pyobject::DICT_TYPE)
                    && std::ptr::eq((*w_locals).w_class, canonical_dict)
            })
    {
        decline!("frame-mapping-not-exact-dict");
    }
    // Resolve every slot's shadow entry BEFORE emitting anything, so a slot the
    // shadow cannot answer declines from a clean trace position.  The read is
    // the standard-virtualizable arm of `_opimpl_getarrayitem_vable`
    // (`virtualizable_boxes[index]`), which records no op — the emit pass below
    // re-runs it through the real entry point.
    let mut slots: Vec<PortalLocalSlot> = Vec::with_capacity(nslots);
    for i in 0..nslots {
        let flat = info.get_index_in_array(0, i, &lengths);
        let Some((slot_op, entry_value)) = ctx.trace_ctx.virtualizable_entry_at(flat) else {
            decline!("slot-no-vable-entry");
        };
        // The value comes from the SHADOW, never from `locals_w!(frame)`.  An
        // unsynchronized virtualizable's heap array holds whatever the frame
        // last wrote out — measured one FOR_ITER iteration behind on the loop
        // variable — which is exactly the staleness the read barrier's
        // `force_now` repairs before the residual reads it.  The shadow already
        // holds the repaired value, so sourcing from it reproduces the forced
        // residual's answer without the force, and it is what upstream's
        // traced-in `fast2locals` reads (`getarrayitem_vable_r` answered from
        // `virtualizable_boxes`).
        //
        // Prefer the OpRef's own concrete: the box is the GC-forwarded
        // channel, so a Ref that moved across an earlier residual is current.
        let held = match ctx
            .trace_ctx
            .concrete_of_opref(slot_op)
            .filter(|v| matches!(v, majit_ir::Value::Ref(_)))
            .unwrap_or(entry_value)
        {
            majit_ir::Value::Ref(gcref) => gcref.as_usize() as pyre_object::PyObjectRef,
            _ => decline!("slot-concrete-not-ref"),
        };
        let cell = is_cell_slot(i);
        let value = if cell {
            // `fast2locals` falls back to the raw slot when it does not hold a
            // cell.  That shape is unreachable past an OPTIMIZED frame's
            // `MAKE_CELL` / `COPY_FREE_VARS` prologue, and modelling it needs a
            // second arm with its own guard, so decline instead -- the same
            // answer the callee arm gives at `cell-slot-not-a-cell`.
            if held.is_null() || !unsafe { pyre_object::is_cell(held) } {
                decline!("cell-slot-not-a-cell");
            }
            unsafe { pyre_object::w_cell_get(held) }
        } else {
            held
        };
        // An unbound slot is a `delitem` on the frame-owned arm, and the cell
        // band has no `delitem` binder: `jit_locals_dict_delitem_local` names
        // its key through `varnames`, which does not reach past `numlocals`.
        // The fresh arm never held the key, so it skips the delete and needs
        // none.
        if cell && i >= numlocals && value.is_null() && frame_owned {
            decline!("unbound-cell-slot-needs-delitem");
        }
        slots.push(PortalLocalSlot { value, cell });
    }

    // Which helper turns the mapping into the published result.  `dir()` reads
    // `getdictscope` — the mapping itself — through `builtin_dir`'s split-out
    // sorted-key-set tail.  `locals()` / `vars()` hand back
    // `frame_locals_snapshot`'s independent PEP 667 copy, which a mapping the
    // expansion just built for itself already is.
    let tail_fn: Option<extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef> =
        match fold {
            FrameLocalsBuiltin::Mapping if frame_owned => {
                Some(pyre_interpreter::pyframe::jit_locals_dict_snapshot)
            }
            FrameLocalsBuiltin::Mapping => None,
            FrameLocalsBuiltin::SortedNames => {
                Some(pyre_interpreter::builtins::jit_dir_names_from_locals)
            }
        };
    // Authentic mapping, built on the plain eval loop exactly as the skipped
    // residual would — through the SAME helpers the emitted calls invoke, so
    // the recording-time value and the compiled loop's value cannot diverge.
    // On the frame-owned arm this MUTATES the frame's own mapping, which is
    // exactly what the residual `fast2locals` does; the rewrite is a pure
    // function of the fastlocals, so a decline below — or a discarded walk —
    // leaves the residual free to redo it with the same outcome.
    let (concrete_locals, concrete_result) = {
        let _roots = pyre_object::gc_roots::push_roots();
        // Pinned before the mapping below, because materialising that mapping
        // allocates and the extras merge runs after the slot chain, which
        // allocates on every store.
        let extra_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_extra_locals);
        let locals_root = pyre_object::gc_roots::shadow_stack_len();
        // Re-read rather than reuse the gate's `w_locals`: the slot resolution
        // above sits between the two, so the pin takes the address the frame
        // holds NOW.
        let _ = pyre_object::gc_roots::pin_root(if frame_owned {
            frame_ref.get_w_locals()
        } else {
            unsafe { pyre_object::w_dict_new() }
        });
        let value_roots: Vec<usize> = slots
            .iter()
            .map(|modelled| {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(modelled.value);
                slot
            })
            .collect();
        let mut result = pyre_object::PY_NULL;
        let mut slot_failed = false;
        for (i, &value_root) in value_roots.iter().enumerate() {
            let value = pyre_object::gc_roots::shadow_stack_get(value_root);
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root);
            // pyframe.py:566-574 — a bound slot is stored, an unbound one
            // deleted.  Both allocate, so the mapping is re-read from its
            // pinned slot on every pass.  A fresh mapping never held the key,
            // so its `delitem` arm is skipped rather than emitted.
            let updated = if !value.is_null() {
                let (setitem, name_index) = portal_slot_binder(i, numlocals);
                setitem(
                    locals,
                    code_ptr as *const pyre_interpreter::CodeObject,
                    name_index,
                    value,
                )
            } else if frame_owned {
                pyre_interpreter::pyframe::jit_locals_dict_delitem_local(
                    locals,
                    code_ptr as *const pyre_interpreter::CodeObject,
                    i as i64,
                )
            } else {
                locals
            };
            if updated.is_null() {
                slot_failed = true;
                break;
            }
        }
        // `frame_locals_proxy_snapshot` appends `f_extra_locals` AFTER the
        // fastlocals and keeps the fastlocal wherever both name one key, so the
        // merge runs with the whole slot chain already behind it.
        if !slot_failed && extras_present {
            let updated = pyre_interpreter::pyframe::jit_locals_dict_update(
                pyre_object::gc_roots::shadow_stack_get(locals_root),
                pyre_object::gc_roots::shadow_stack_get(extra_root),
            );
            slot_failed = updated.is_null();
        }
        if !slot_failed {
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root);
            // The tail runs here too, so the recorded result is produced by the
            // very helper the emitted call names.
            result = match tail_fn {
                Some(tail) => tail(locals),
                None => locals,
            };
        }
        // The rewrite above allocates on every bound slot, and the tail
        // allocates again, so a collection can have forwarded any value the
        // resolution pass captured as a bare pointer.  The emit pass stamps
        // these onto the `Cell.contents` reads it records; take them from the
        // pins that tracked the move, exactly as `locals_root` is taken below.
        for (modelled, &value_root) in slots.iter_mut().zip(&value_roots) {
            modelled.value = pyre_object::gc_roots::shadow_stack_get(value_root);
        }
        (pyre_object::gc_roots::shadow_stack_get(locals_root), result)
    };
    // A slot rewrite or the tail reports a failure as PY_NULL instead of
    // publishing it; nothing has been emitted yet, so decline and let the
    // residual raise.
    if concrete_result.is_null() {
        decline!("concrete-result-null");
    }
    let concrete_locals_value = majit_ir::Value::Ref(majit_ir::GcRef(concrete_locals as usize));

    // emit the specialized IR (walker-native)
    // Pin the callable identity (LOAD_GLOBAL `locals` is usually already a
    // constant via the namespace cell fold).
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    // The code object is the jitdriver green this trace is keyed on
    // (`interp_jit.py:23 greens = ['next_instr', 'is_being_profiled',
    // 'pycode']`), so its address is a constant for the compiled loop and
    // carries no guard of its own.
    let code_const = ctx.trace_ctx.const_int(code_ptr as i64);
    // `d = self.getorcreatedebug()` — pyframe.py.  An absent payload has no
    // `w_locals` to read, so the guard pins that direction and the fresh-dict
    // arm below stands in for the materialisation.
    let debugdata_present = debugdata_ref.as_usize() != 0;
    walker_guard_stamped_presence(ctx, op.pc, debugdata_op, debugdata_present)?;
    // `d.w_locals` — pyframe.py:556.  Read whenever there is a payload to read
    // it from, and guarded in the direction recorded, so a frame that
    // materialises its mapping mid-loop side-exits instead of going on writing
    // into the expansion's own dict.
    let mut field_op = None;
    let mut extra_field_op = None;
    if debugdata_present {
        let op_ref = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            debugdata_op,
            crate::descr::frame_debug_data_w_locals_descr(),
        );
        walker_guard_stamped_presence(ctx, op.pc, op_ref, frame_owned)?;
        field_op = Some(op_ref);
        // `d.w_extra_locals` — read here so the merge below the slot chain
        // has it, and pinned absent when there is none so a mid-loop proxy
        // write side-exits instead of dropping the new key.
        let extra_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            debugdata_op,
            crate::descr::frame_debug_data_w_extra_locals_descr(),
        );
        walker_guard_stamped_presence(ctx, op.pc, extra_op, extras_present)?;
        extra_field_op = Some(extra_op);
    }
    // `w_cell_get` per cell slot, and every slot's own boundness guard, BEFORE
    // the mapping is built: a guard that fails after a `newdict` side exits to
    // a residual that allocates a second mapping, and one that fails after a
    // store has already rewritten the frame's own mapping leaves the residual
    // to redo the part that landed.  The reads touch no frame, so none of them
    // can re-arm the escape this expansion exists to remove.
    let mut value_ops: Vec<OpRef> = Vec::with_capacity(slots.len());
    let mut index_consts: Vec<OpRef> = Vec::with_capacity(slots.len());
    for (i, modelled) in slots.iter().enumerate() {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        // `self.locals_cells_stack_w[i]` — `jtransform.py do_fixed_list_getitem
        // do_fixed_list_getitem`, the identical lowering `emit_load_fast_ref!`
        // emits for LOAD_FAST.  On the standard virtualizable this resolves to
        // `virtualizable_boxes[index]` and records no op.
        let index_const = ctx.trace_ctx.const_int(i as i64);
        index_consts.push(index_const);
        let (slot_op, _) = vable_ops::with_replace_frames(ctx, |ctx| {
            let nonstandard =
                vable_ops::walker_nonstandard_virtualizable(ctx, op.pc, vable_op, &fdescr)?;
            Ok(ctx.trace_ctx.vable_getarrayitem_ref_checked(
                nonstandard,
                op.pc,
                vable_op,
                index_const,
                i as i64,
                fdescr.clone(),
                adescr.clone(),
            ))
        })?;
        // `pyframe.py:566-571` branches on the slot being bound; pin the
        // direction so a slot that changes bound-ness side-exits instead of
        // publishing a mapping with the wrong key set.  A slot the trace
        // already holds as a constant needs no guard.
        //
        // For a cell slot the branch is on the CONTENTS, not on the slot: the
        // frame prologue put the `Cell` there and nothing in the body replaces
        // it, so the slot itself is bound on every execution of this path.
        let bound = !modelled.value.is_null();
        walker_guard_stamped_presence(ctx, op.pc, slot_op, bound || modelled.cell)?;
        // `w_cell_get` -- `Cell.contents`, the whole of `fast2locals`' cell
        // half.  The compiled loop re-reads the slot, so the class the walk saw
        // is stated rather than assumed.
        let value_op = if modelled.cell {
            walker_pin_cell_contents(ctx, op.pc, slot_op, modelled.value)?
        } else {
            slot_op
        };
        value_ops.push(value_op);
    }
    // The last read's guard lands after that loop's own check, so the mapping
    // and the chain below would otherwise reach the opcode-level check in
    // `mod.rs` unweighed.
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    let mut dict_op = match field_op.filter(|_| frame_owned) {
        Some(op_ref) => op_ref,
        // pyframe.py `self.space.newdict(instance=True)` — the mapping
        // `fast2locals` would have materialised, built here instead of
        // modelling the store back into the debug payload.
        None => ctx.trace_ctx.call_ref_typed_with_effect(
            pyre_interpreter::pyframe::jit_locals_dict_new as *const (),
            &[],
            &[],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ),
    };
    ctx.trace_ctx
        .set_opref_concrete(dict_op, concrete_locals_value);
    if frame_owned {
        walker_guard_exact_instance(
            ctx,
            op.pc,
            dict_op,
            &pyre_object::pyobject::DICT_TYPE as *const _ as i64,
            // Re-derived rather than reusing the gate's binding: the
            // record-time rewrite above allocates, so this takes the address
            // `dict` has NOW.
            pyre_object::get_instantiate(&pyre_object::pyobject::DICT_TYPE),
        )?;
    }
    for ((i, modelled), (&index_const, &value_op)) in slots
        .iter()
        .enumerate()
        .zip(index_consts.iter().zip(&value_ops))
    {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        let bound = !modelled.value.is_null();
        // `pyframe.py:566-574` — a bound slot is stored, an unbound one
        // deleted.  The delete is what keeps a key from a since-unbound local
        // out of a mapping the frame carries across calls; on the fresh arm
        // the mapping never held the key, so it is skipped.
        if !bound && !frame_owned {
            continue;
        }
        let (helper, args, arg_types): (_, Vec<OpRef>, Vec<majit_ir::Type>) = if bound {
            let (setitem, name_index) = portal_slot_binder(i, numlocals);
            let name_const = if name_index == i as i64 {
                index_const
            } else {
                ctx.trace_ctx.const_int(name_index)
            };
            (
                setitem as *const (),
                vec![dict_op, code_const, name_const, value_op],
                vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                    majit_ir::Type::Ref,
                ],
            )
        } else {
            (
                pyre_interpreter::pyframe::jit_locals_dict_delitem_local as *const (),
                vec![dict_op, code_const, index_const],
                vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ],
            )
        };
        dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
            helper,
            &args,
            &arg_types,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        // Every link of the chain names the SAME mapping, so the post-build
        // address is the live one for all of them.
        ctx.trace_ctx
            .set_opref_concrete(dict_op, concrete_locals_value);
        if !bound {
            // The delete reports a raising comparison as PY_NULL instead of
            // publishing it; side-exit so the residual re-runs and raises.
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[dict_op])?;
        }
    }
    // `frame_locals_proxy_snapshot` appends `f_extra_locals` AFTER the
    // fastlocals, so the merge is emitted with the whole slot chain in front of
    // it and the mapping already carries the fastlocal for any key both halves
    // name.  The helper can raise (a user `__eq__` on a colliding key), so a
    // PY_NULL return side-exits to the residual.
    if extras_present && let Some(extra_op) = extra_field_op {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        let updated = ctx.trace_ctx.call_ref_typed_with_effect(
            pyre_interpreter::pyframe::jit_locals_dict_update as *const (),
            &[dict_op, extra_op],
            &[majit_ir::Type::Ref, majit_ir::Type::Ref],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[updated])?;
        ctx.trace_ctx
            .set_opref_concrete(updated, concrete_locals_value);
        dict_op = updated;
    }
    // Asked again with the loop behind it: the check above runs BEFORE a slot
    // emits, so the last slot's own ops -- and the tail below -- would
    // otherwise reach the opcode-level check in `mod.rs` unweighed.
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    // `frame_locals_snapshot`'s PEP 667 copy for `locals()` / `vars()`, or
    // `builtin_dir`'s no-argument tail for `dir()` — each split out so the
    // trace and the eval loop run one implementation.  Both report a failure
    // as PY_NULL instead of publishing it, so the guarded side exit re-runs
    // the residual and raises from the eval loop.
    let result_op = match tail_fn {
        Some(tail) => {
            let op_ref = ctx.trace_ctx.call_ref_typed_with_effect(
                tail as *const (),
                &[dict_op],
                &[majit_ir::Type::Ref],
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::CannotRaise,
                    majit_ir::OopSpecIndex::None,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                op_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
            );
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[op_ref])?;
            op_ref
        }
        None => dict_op,
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result_op)?;
    Ok(Some(()))
}

/// [`try_walker_specialize_builtin_locals`]'s arm for an inlined callee level.
///
/// The frame `builtin_locals` reports on here is the level's own, never the
/// standard virtualizable, so none of the portal expansion's vable reads
/// apply.  Upstream has the same asymmetry and resolves it the same way: one
/// standard virtualizable per trace, an inlined callee frame left an ordinary
/// virtual, and `pyframe.py fast2locals` — `@jit.unroll_safe`, so
/// `policy.py` looks inside it — traced through reading that virtual's
/// fields.  Pyre's counterpart of the virtual is [`CalleeLocalsShadow`]: every
/// visible fastlocal of this level is already an SSA value the walk holds, and
/// `getarrayitem_vable_via_metainterp`'s strict fresh-frame fold is what
/// answers the level's own `LOAD_FAST` from it.  Sourcing the expansion from
/// the same map is therefore the same read the callee's own bytecode makes.
///
/// Emitted shape: `guard_value(callable)` when the name is not already a trace
/// constant; `jit_locals_dict_new` (the `space.newdict(instance=True)` a fresh
/// frame's `fast2locals` materialises); one non-forcing `jit_locals_dict_setitem_local`
/// `Call` per BOUND slot, taking the slot's SSA value straight as an operand;
/// and `jit_dir_names_from_locals` for `dir()`.  No vable read, no `PyFrame`
/// operand, so nothing on it can reach `force_frame` — which is the point,
/// since the opaque residual this replaces forces the published callee frame
/// and `tracing_after_residual_call` reads that as an escape.
///
/// No per-slot boundness guard: a slot's SSA value exists precisely because a
/// param seed or a `STORE_FAST` on the traced path produced it, and the guards
/// already on that path pin which of them ran.  That is the difference from
/// the portal arm, whose slots come out of a virtualizable array the compiled
/// loop re-reads.
///
/// A level whose frame the seed block materialised is NOT excluded.  The
/// `frame_materialized` flag governs STORES — `folded_store_is_observable_local`
/// demotes a `STORE_FAST` into a recorded `SETARRAYITEM_GC` so a frame reached
/// later through a traceback, `f_locals` or `sys._getframe` sees the value —
/// and that demoted store still re-seeds `opref`, so the shadow and the heap
/// array hold the same value.  Reading is what this arm does, and it reads the
/// channel the level's own `LOAD_FAST` reads.
///
/// Returns `None` (fall through to the generic residual, SAFE — exactly the
/// behaviour this arm replaced) for every other shape: a sub-walk whose guards
/// would collapse to the caller's CALL boundary, a level with no shadow, an
/// unseeded frame box, a top frame that is not this level's own, a shadow
/// describing another code object, a non-OPTIMIZED frame, cellvars / freevars
/// / `CO_FAST_HIDDEN` slots, a frame that already carries a locals mapping or
/// an `f_extra_locals` dict, and a written slot the shadow cannot resolve back
/// to a Ref.  A multiframe level keeps the strict fold off
/// (`fold_frame_reg == u16::MAX`) and still answers: its `frame_box` and
/// per-slot `opref` map are the virtual `fast2locals` reads.  `PYRE_FBW_DEBUG_ABORT`
/// names which of them declined.
fn try_walker_specialize_builtin_locals_in_callee<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    fold: FrameLocalsBuiltin,
    r_args: &[OpRef],
    concrete_callable: pyre_object::PyObjectRef,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if let Some(done) = try_walker_specialize_builtin_locals_in_callee_expand(
        ctx,
        op,
        fold,
        r_args,
        concrete_callable,
        dst,
    )? {
        return Ok(Some(done));
    }
    // The expansion declined, so this level is about to run the opaque
    // residual -- and that residual's `force_frame_before_locals_read` clears
    // `TOKEN_TRACING_RESCALL` on the level's own published frame, which
    // `tracing_after_residual_call` reads as `VableEscapedDuringResidualCall`.
    // Falling through therefore does not cost one residual call; it costs the
    // enclosing loop, because the escape is a property of the callee BODY and
    // every retry rebuilds the same framestack and escapes again until
    // `MAX_TRACE_ABORT_COUNT` retires the caller.
    //
    // Refuse the callee HERE instead, before the residual runs.  The caller
    // then records the plain `bh_call_fn` it would have recorded had this body
    // never been admitted, and the escape never happens: same answer, one
    // decline instead of an abort.  What reaches this line is a shape the
    // expansion models no part of -- a non-OPTIMIZED frame, a `CO_FAST_HIDDEN`
    // slot, a frame already carrying an `f_locals` mapping or an
    // `f_extra_locals` dict, or a slot whose value this walk never saw.  Each
    // of those is named under `PYRE_FBW_DEBUG_ABORT`, because the refusal is
    // not the answer: it stands in for an expansion that does not model the
    // shape yet, and widening the expansion until this line is unreachable is
    // the convergence path.
    //
    // A multiframe-seeded level does not land here: the strict fold witness
    // (`fold_frame_reg`) is off for a virtual callee frame on purpose, and
    // the shadow still holds the SSA values `fast2locals` reads.  Each gate
    // of the expansion records
    // why it does not fire: the non-OPTIMIZED refusal is the answer rather
    // than a gap, its `CO_FAST_HIDDEN` half is a bit this compiler never
    // sets, and the two frame-payload gates sit behind writers that need a
    // reference to the level's own live frame.
    if let Some(callee) = super::fbw_state::fbw_innermost_inline_callee_key(ctx) {
        return Err(super::fbw_state::fbw_decline_inline_callee(
            ctx,
            op.pc,
            Some(callee),
        ));
    }
    Ok(None)
}

/// One slot of an inlined callee's frame that the modelled `fast2locals`
/// reproduces.
struct ModelledLocalSlot {
    /// The localsplus slot index, which is also the `varnames` index below
    /// `numlocals` and, above it, `numlocals + cell_slot_names` index.
    index: i64,
    /// What the walk holds AT the slot: the bound value for a plain
    /// fastlocal, the `Cell` for a cell slot.
    slot_op: OpRef,
    /// Whether `slot_op` is a `Cell` whose contents this slot's key takes.
    cell: bool,
    /// The recording-time value the key would be bound to, `PY_NULL` for an
    /// empty cell (which binds no key).
    value: pyre_object::PyObjectRef,
}

impl ModelledLocalSlot {
    /// The `fast2locals` binder for this slot and the index it names its key
    /// with: `code.varnames[index]` for a slot below `numlocals` — a shared
    /// cellvar slot included, since that is the name it carries — and
    /// `cell_slot_names(code)[index - numlocals]` above it.
    fn binder(&self, numlocals: usize) -> (LocalsDictSetitem, i64) {
        if (self.index as usize) < numlocals {
            (
                pyre_interpreter::pyframe::jit_locals_dict_setitem_local,
                self.index,
            )
        } else {
            (
                pyre_interpreter::pyframe::jit_locals_dict_setitem_cell,
                self.index - numlocals as i64,
            )
        }
    }
}

/// The expansion itself: `Ok(None)` means "this shape is not modelled", which
/// its caller turns into an inline refusal rather than a residual.
fn try_walker_specialize_builtin_locals_in_callee_expand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    fold: FrameLocalsBuiltin,
    r_args: &[OpRef],
    concrete_callable: pyre_object::PyObjectRef,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    /// Name the gate that declined.  A decline here costs the caller its whole
    /// inlined callee, so "the expansion models no part of this shape" has to
    /// be attributable to one gate rather than re-derived by reading.
    macro_rules! decline {
        ($why:literal) => {{
            locals_expansion_declined("IN-CALLEE", $why);
            return Ok(None);
        }};
    }
    // Under a single-frame collapse the resume re-executes the whole call, so
    // a guard emitted here re-runs every side effect the inline region already
    // sequenced.  Same gate the other folds that run under a sub-walk take.
    if ctx.fbw_mode.inline_subwalk && !walker_inline_guard_resumes_in_callee(ctx) {
        decline!("subwalk-no-callee-resume");
    }
    let (fold_frame_reg, shadow_code_ptr) = {
        let state = ctx.frame_state.borrow();
        let Some(shadow) = state.callee_shadow.as_ref() else {
            decline!("no-callee-shadow");
        };
        // A `NONE` frame box is a frame register that was never seeded, so
        // the shadow is not the authority for this level's slots.
        // `u16::MAX` is only the strict fresh-frame fold switched off: a
        // multiframe callee still seeds `frame_box` and the per-slot `opref`
        // map, and `fast2locals` (`pyframe.py`, `@jit.unroll_safe`) reads
        // those SSA values.  `rewrite_op_jit_force_virtualizable`
        // (`jtransform.py`) erases the force inside a graph the tracer
        // looks into, so refusing this shape would abort the enclosing
        // walk instead of recording the look-inside.
        if shadow.frame_box.is_none() {
            decline!("unseeded-frame-register");
        }
        (shadow.fold_frame_reg, shadow.code_ptr)
    };
    // `interp_inspect.py locals` reaches its frame through
    // `gettopframe_nohidden`, and `walker_ec_enter` has published THIS level's
    // concrete frame there for the whole sub-walk.  Require the two to name
    // one frame, so a level that never entered the chain — or one whose top
    // frame is someone else's — declines instead of answering for the wrong
    // frame.  Read the identity back through the guard's root, which the
    // collector forwards.
    let inline_frame = current_inline_concrete_frame();
    let ec = pyre_interpreter::call::getexecutioncontext();
    if inline_frame == 0 || ec.is_null() {
        decline!("no-inline-frame-or-ec");
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() || frame as usize != inline_frame {
        decline!("top-frame-not-this-level");
    }
    let frame_ref = unsafe { &*frame };
    let code_ptr = unsafe { pyre_interpreter::pyframe::pyframe_get_pycode(frame_ref) };
    // The shadow's slots index the code object it was opened for; a
    // disagreement means the map does not describe this frame's fastlocals.
    if code_ptr.is_null() || !std::ptr::eq(code_ptr, shadow_code_ptr) {
        decline!("shadow-names-other-code");
    }
    let code_obj = unsafe { &*code_ptr };
    // Two conditions, and only the first can fire.  A non-OPTIMIZED frame
    // answers `locals()` from `PyFrame::getdictscope` — its LIVE namespace,
    // not the independent copy this arm builds — so declining is the answer
    // rather than a gap.  The `CO_FAST_HIDDEN` half is inert: `pycode.rs`
    // builds `localspluskinds` out of `CO_FAST_LOCAL` and `CO_FAST_CELL`
    // alone, so nothing this compiler produces carries the bit, and
    // `PyFrame::fast2locals` skips such a slot only on a frame this arm has
    // already refused.
    if !pyre_interpreter::PyFrame::code_locals_are_modelled_fastlocals(code_obj) {
        decline!("not-modelled-fastlocals");
    }
    let numlocals = code_obj.varnames.len();
    // The pure cellvars and the freevars occupy the slots above `varnames` in
    // the unified layout, and `fast2locals` binds each of them under the name
    // `cell_slot_names` gives it.  A cellvar that shares a varname slot is
    // named by `varnames` and is only a CELL there, which the per-slot kind
    // below picks up.
    let nslots = numlocals + pyre_interpreter::PyFrame::cell_slot_names(code_obj).count();
    // No width ceiling here, and none on the portal arm either: upstream
    // bounds this unroll with `@jit.unroll_safe` on `pyframe.py`
    // `PyFrame.fast2locals` and nothing else, and leaves the length question
    // to `trace_limit`.  What made the ceiling worth removing HERE first is
    // the price of the refusal: the portal arm's decline falls through to the
    // generic residual, while a decline on this arm denies the callee for the
    // rest of the thread's tracing, so the ceiling decided inlinability from a
    // callee's local count.
    //
    // No ceiling and no preflight, for the reason the portal arm records; the
    // per-slot `locals_expansion_cut_if_too_long` below answers the length
    // question instead.  That price is also what makes a preflight worse on
    // this arm than on that one: refusing on the walk's remaining budget would
    // let the length of the trace so far decide a callee's inlinability, the
    // way the ceiling let its local count decide it.  The cut is not that --
    // it ends the trace rather than the callee, so nothing is remembered
    // against the body.

    // Fresh mapping only.  A frame that already carries one — an `f_locals`
    // write (PEP 667), a `setdictscope` — is the portal arm's frame-owned
    // shape, whose rewrite has to reach the frame's own dict across calls;
    // this level's frame is rebuilt from scratch by the compiled trace when it
    // is built at all, so that shape has no counterpart here and declines.
    //
    // Never observed to fire, and the reason is structural: this arm has
    // already required an OPTIMIZED frame, and on one of those `w_locals` has
    // no writer the answer can follow.  `bind_unoptimized_locals_scope`
    // returns before binding it, `PyFrame::fget_getdictscope` hands an
    // optimized frame a `FrameLocalsProxy` rather than calling
    // `getdictscope`, and the line-tracing call to `fast2locals` is guarded on
    // `w_locals` being non-null already, so it cannot be the first setter.
    if !frame_ref.get_w_locals().is_null() {
        decline!("frame-has-w-locals");
    }
    // A null `w_locals` is NOT on its own a fresh mapping.  A proxy write
    // whose key names no writable fast local goes to `f_extra_locals`
    // (`framelocalsproxy_setitem`) and leaves `w_locals` null, and
    // `frame_locals_proxy_snapshot` copies that dict into every mapping it
    // hands back — so rebuilding from the shadow's slots alone would drop the
    // key.  Read the LIVE callee frame, which is the only holder: this level's
    // frame is built by the compiled trace, so its payload starts empty every
    // iteration and only a residual on the recorded path can have filled it.
    //
    // Never observed to fire either, and an earlier gate is why.  The one
    // writer is `FrameLocalsProxy::setitem_value`, so filling the dict takes a
    // reference to this level's own live frame, and that write calls
    // `force_locals` before it stores.  `locals_proxy_extra_key_hot` drives
    // exactly that shape one frame in and reports no decline at all: the
    // expansion is not reached there, because the callee is no longer an
    // un-escaped inline level by the time `locals()` is recorded.
    if !frame_ref.get_extra_locals().is_null() {
        decline!("frame-has-extra-locals");
    }
    // Collect each slot's shadow entry first, so a slot the shadow cannot
    // answer declines from a clean trace position and nothing is emitted.
    let is_cell_slot = |slot: usize| {
        slot >= numlocals
            || (slot < code_obj.localspluskinds.len()
                && code_obj.localspluskinds[slot] & pyre_interpreter::bytecode::CO_FAST_CELL != 0)
    };
    let mut slot_oprefs: Vec<Option<OpRef>> = Vec::with_capacity(nslots);
    {
        let state = ctx.frame_state.borrow();
        let Some(shadow) = state.callee_shadow.as_ref() else {
            decline!("shadow-vanished");
        };
        for slot in 0..nslots as i64 {
            match (shadow.opref.get(&slot).copied(), shadow.concrete.get(&slot)) {
                // Absent from both: this walk never wrote the slot, and the
                // frame it would otherwise have kept a value in is fresh, so
                // the slot is UNBOUND and `fast2locals` binds no key for it.
                //
                // A CELL slot is not fresh in that sense: the frame setup
                // built the cell (`MAKE_CELL`) or copied it out of the
                // closure (`COPY_FREE_VARS`) before the first opcode ran, so
                // absence here means the walk never READ it, not that the name
                // is unbound.  There is no SSA value to bind and guessing
                // "unbound" would drop a live name, so decline.
                (None, None) if is_cell_slot(slot as usize) => decline!("cell-slot-unread"),
                (None, None) => slot_oprefs.push(None),
                // Only an entry recorded through THIS level's frame register
                // describes this frame — the same per-frame isolation the
                // `getarrayitem_vable` read fallback applies.  When the
                // strict fold is off the shadow is still this level's, and
                // `concrete.frame_reg` names the portal red rather than the
                // fold witness; accept the opref and let `concrete_of_opref`
                // resolve the value.
                (Some(opref), Some(concrete))
                    if fold_frame_reg == u16::MAX || concrete.frame_reg == fold_frame_reg =>
                {
                    slot_oprefs.push(Some(opref))
                }
                (Some(opref), None) if fold_frame_reg == u16::MAX => slot_oprefs.push(Some(opref)),
                // Written with no reconstructable concrete half, or through
                // another frame's register: decline rather than guess.
                _ => decline!("slot-not-this-frame"),
            }
        }
    }
    // Every slot that `fast2locals` binds a key for, in slot order.
    //
    // `slot_op` is what the walk holds AT the slot: the value itself for a
    // plain fastlocal, the CELL for a cell slot.  The emit below turns the
    // latter into its contents with one `GETFIELD_GC_R`, which is why the read
    // is not done here — nothing may be emitted while a later slot can still
    // decline.
    let mut slots: Vec<ModelledLocalSlot> = Vec::with_capacity(nslots);
    for (index, entry) in slot_oprefs.iter().enumerate() {
        let Some(slot_op) = *entry else {
            continue;
        };
        // Resolve through the op table, not the shadow's raw `concrete` copy:
        // the table is the GC-forwarded channel, so a Ref that moved across an
        // earlier residual is current there.
        let Some(majit_ir::Value::Ref(gcref)) = ctx.trace_ctx.concrete_of_opref(slot_op) else {
            decline!("slot-concrete-not-ref");
        };
        if gcref == majit_ir::GcRef::NO_CONCRETE {
            decline!("slot-no-concrete");
        }
        let held = gcref.as_usize() as pyre_object::PyObjectRef;
        let cell = is_cell_slot(index);
        if cell {
            // `fast2locals` falls back to the raw slot when it does not hold a
            // cell.  That shape is unreachable for an OPTIMIZED frame past its
            // `MAKE_CELL` / `COPY_FREE_VARS` prologue, and modelling it would
            // need a second arm with its own guard, so decline instead.
            if held.is_null() || !unsafe { pyre_object::is_cell(held) } {
                decline!("cell-slot-not-a-cell");
            }
        }
        let value = if cell {
            unsafe { pyre_object::w_cell_get(held) }
        } else {
            held
        };
        if value.is_null() && !cell {
            // A slot the walk unbound (`DELETE_FAST`).  The mapping is fresh,
            // so it binds no key for it — but only a NULL the trace holds as a
            // constant is unbound on every execution of the compiled path.
            if !slot_op.is_constant() {
                decline!("unbound-slot-not-constant");
            }
            continue;
        }
        slots.push(ModelledLocalSlot {
            index: index as i64,
            slot_op,
            cell,
            value,
        });
    }

    // `dir()` takes `builtin_dir`'s split-out sorted-name tail, which reads
    // `getdictscope` — the mapping itself.  `locals()` / `vars()` need none:
    // nothing else references a mapping the expansion just built, so it is
    // already the independent copy `frame_locals_snapshot` hands back.
    let tail_fn: Option<extern "C" fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef> =
        match fold {
            FrameLocalsBuiltin::Mapping => None,
            FrameLocalsBuiltin::SortedNames => {
                Some(pyre_interpreter::builtins::jit_dir_names_from_locals)
            }
        };
    // The slots that bind a key.  An empty cell binds none — `fast2locals`
    // deletes the name there, and this mapping is fresh, so there is nothing
    // to delete.
    let bound: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.value.is_null())
        .map(|(i, _)| i)
        .collect();
    // Authentic mapping, built through the SAME helpers the emitted calls
    // name, so the recording-time value and the compiled loop's value cannot
    // diverge.  Nothing here touches the frame, so a decline below — or a
    // discarded walk — leaves the residual free to redo it with the same
    // outcome.
    let (concrete_locals, concrete_result) = {
        let _roots = pyre_object::gc_roots::push_roots();
        // Values first: the `w_dict_new` below allocates, so a slot value
        // still held only as a bare pointer could be moved out from under the
        // pin that was about to take it.
        let value_roots: Vec<usize> = bound
            .iter()
            .map(|&i| {
                let root = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(slots[i].value);
                root
            })
            .collect();
        let locals_root = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(unsafe { pyre_object::w_dict_new() });
        let mut result = pyre_object::PY_NULL;
        let mut slot_failed = false;
        for (&i, &value_root) in bound.iter().zip(&value_roots) {
            // The store allocates, so both the mapping and the value are
            // re-read from their pinned slots on every pass.
            let (setitem, name_index) = slots[i].binder(numlocals);
            let updated = setitem(
                pyre_object::gc_roots::shadow_stack_get(locals_root),
                code_ptr as *const pyre_interpreter::CodeObject,
                name_index,
                pyre_object::gc_roots::shadow_stack_get(value_root),
            );
            if updated.is_null() {
                slot_failed = true;
                break;
            }
        }
        if !slot_failed {
            let locals = pyre_object::gc_roots::shadow_stack_get(locals_root);
            result = match tail_fn {
                Some(tail) => tail(locals),
                None => locals,
            };
        }
        // The rewrite above allocates on every bound slot, and the tail
        // allocates again, so a collection can have forwarded any value the
        // resolution pass captured as a bare pointer.  The emit pass stamps
        // these onto the `Cell.contents` reads it records; take them from the
        // pins that tracked the move, exactly as `locals_root` is taken below.
        for (&i, &value_root) in bound.iter().zip(&value_roots) {
            slots[i].value = pyre_object::gc_roots::shadow_stack_get(value_root);
        }
        (pyre_object::gc_roots::shadow_stack_get(locals_root), result)
    };
    // A slot rewrite or the tail reports a failure as PY_NULL instead of
    // publishing it; nothing has been emitted yet, so decline and let the
    // caller record the plain call, which raises the same way.
    if concrete_result.is_null() {
        decline!("concrete-result-null");
    }
    let concrete_locals_value = majit_ir::Value::Ref(majit_ir::GcRef(concrete_locals as usize));

    // emit the specialized IR (walker-native)
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    // The callee's code object is fixed for this inline level, so its address
    // is a constant for the compiled loop and carries no guard of its own.
    let code_const = ctx.trace_ctx.const_int(code_ptr as i64);
    // `w_cell_get` per cell slot, BEFORE the mapping is allocated: each one
    // carries a guard, and a guard that fails after the `newdict` would side
    // exit to a residual that allocates a second mapping.  The read is the
    // whole of `fast2locals`' cell half — `Cell.contents` — and it touches no
    // frame, so it cannot re-arm the escape this expansion exists to remove.
    let mut value_ops: Vec<OpRef> = Vec::with_capacity(slots.len());
    for slot in &slots {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        if !slot.cell {
            value_ops.push(slot.slot_op);
            continue;
        }
        // The slot holds a `Cell` on every execution of this path: the frame
        // prologue put it there and nothing in the body replaces it, but the
        // compiled loop re-reads the slot, so say so.
        value_ops.push(walker_pin_cell_contents(
            ctx,
            op.pc,
            slot.slot_op,
            slot.value,
        )?);
    }
    // Same reason as the portal arm: the last cell read's guard lands after
    // that loop's own check.
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    // pyframe.py `self.space.newdict(instance=True)` — the mapping a fresh
    // frame's `fast2locals` materialises before filling it.
    let mut dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
        pyre_interpreter::pyframe::jit_locals_dict_new as *const (),
        &[],
        &[],
        majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        ),
    );
    ctx.trace_ctx
        .set_opref_concrete(dict_op, concrete_locals_value);
    for (slot, &value_op) in slots.iter().zip(&value_ops) {
        locals_expansion_cut_if_too_long(ctx, op.pc)?;
        if slot.value.is_null() {
            continue;
        }
        // pyframe.py:566-571 — bind this slot's name to its value.  For a
        // plain fastlocal the value is the SSA operand the level's own
        // `LOAD_FAST` would have folded to; for a cell slot it is the
        // `Cell.contents` read emitted above.
        let (setitem, name_index) = slot.binder(numlocals);
        let index_const = ctx.trace_ctx.const_int(name_index);
        dict_op = ctx.trace_ctx.call_ref_typed_with_effect(
            setitem as *const (),
            &[dict_op, code_const, index_const, value_op],
            &[
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
            ],
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::CannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        // Every link of the chain names the SAME mapping.
        ctx.trace_ctx
            .set_opref_concrete(dict_op, concrete_locals_value);
    }
    locals_expansion_cut_if_too_long(ctx, op.pc)?;
    let result_op = match tail_fn {
        Some(tail) => {
            let op_ref = ctx.trace_ctx.call_ref_typed_with_effect(
                tail as *const (),
                &[dict_op],
                &[majit_ir::Type::Ref],
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::CannotRaise,
                    majit_ir::OopSpecIndex::None,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                op_ref,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
            );
            // The tail reports a failure as PY_NULL instead of publishing it,
            // so the guarded side exit re-runs the residual and raises from
            // the eval loop.
            walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[op_ref])?;
            op_ref
        }
        None => dict_op,
    };
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result_op)?;
    Ok(Some(()))
}

/// `sys._getframe()` / `sys._getframe(0)` at the top walk level: publish the
/// portal virtualizable itself instead of residualizing `vm.py getframe`.
///
/// `getframe` is `@jit.look_inside_iff(lambda space, depth:
/// jit.isconstant(depth))` (`pypy/module/sys/vm.py`), so a constant depth is
/// traced THROUGH: at the portal, `ec.gettopframe_nohidden()` is a vref read
/// that `pyjitpl.py _do_jit_force_virtual` answers with
/// `virtualizable_boxes[-1]` under a `ptr_eq` + `implement_guard_value`; in an
/// inline MIFrame its live `JitVirtualRef` is known non-standard and follows
/// the residual `jit_force_virtual` path, which `virtualize.py` removes after
/// `vrefs_after_residual_call` publishes the forced pair.  The `depth == 0`
/// test folds away, and `mark_as_escaped` is one `setfield_gc`.
/// No call survives optimization and no virtualizable is forced — pypy3
/// reports `forcings: 0` and `abort: vable escape: 0` on the fixtures where
/// pyre loses the loop.
///
/// Pyre residualizes the same walk as one opaque `bh_call_fn(_getframe,
/// PY_NULL, depth)` `CallMayForce`, and [`pyre_interpreter::module::sys::vm::getframe`]'s
/// `force_frame` on the frame it returns — the stand-in for the injection
/// `rvirtualizable.py hook_access_field` performs and pyre's rtyper
/// cannot build — clears `TOKEN_TRACING_RESCALL` inside that call whenever the
/// returned frame is the traced one, which `tracing_after_residual_call` reads
/// as an escape (`VableEscapedDuringResidualCall`).  At depth 0 the returned
/// frame is always the portal, so the residual always escapes.  Removing it
/// removes the force with it, and nothing has to replace it: `last_instr` is
/// published onto the portal frame at every may-force boundary
/// (`LiveLastInstrGuard`).  A generic reader of either getter still retains
/// that residual boundary.  Upstream does not: `pyframe.py fget_f_lasti` and
/// `fget_f_lineno` are loop-free and carry no hint, so `policy.py
/// look_inside_graph` admits them, `jtransform.py
/// rewrite_op_jit_force_virtualizable` deletes the injected force, and
/// `pyjitpl.py opimpl_getfield_vable_i` answers `last_instr` out of
/// `virtualizable_boxes`.  Measured over a 200k-iteration read against a
/// same-shape loop that does not read the frame: pypy3 answers `f_lasti`
/// faster than that control loop -- the trace constant -- and pyre was 53x it,
/// while `f_lineno` keeps one non-forcing residual on both and pyre is 2.1x.
/// The two halves are closed by two different emissions, because the shapes
/// they are closing to differ: [`try_walker_specialize_frame_lasti`] emits the
/// constant, while [`try_walker_specialize_frame_lineno`] emits the one
/// non-forcing residual upstream also keeps for the line-table decode.  Either
/// is an optimization over a correct path, not a fix, and both owe two
/// coordinates the boundary hides.  `last_instr` is an instruction-unit index
/// here and the
/// app-level getter reports it doubled (`typedef.rs`, matching `location.py
/// offset2lineno`'s `stopat // 2` on the byte offset upstream stores), so an
/// emission at the app level owes the factor.  And the field has two writers
/// on two conventions: `flush_walk_end_state_to_frame` writes
/// `resume_py_pc - 1` while `LiveLastInstrGuard::enter_frame` writes the
/// executing pc unshifted, and a getter owes the executing one.
/// An exact optimized-frame `f_locals` read is specialized
/// below instead: `pyframe.py fast2locals` is `@jit.unroll_safe`, so upstream
/// traces through it and reads the virtualizable boxes rather than forcing.
/// The force it drops was also the only writer of the frame's locals region,
/// which pyre's `FrameLocalsProxy` reads; the fold therefore performs that
/// write-back itself (`walker_write_back_standard_frame_locals`).
///
/// Emitted shape, following `getframe`'s body line by line:
/// `guard_value(callable)`; `guard_class` + exact-class + `getfield_gc_i` on
/// the depth box, whose resulting RAW int must be a trace constant — that
/// unboxed value is what `jit.isconstant(depth)` tests upstream, where
/// `@unwrap_spec(depth=int)` has already run OUTSIDE the looked-inside graph
/// (the wrapped `W_IntObject` the residual receives is built in-trace by
/// `NewWithVtable` + `SetfieldGc` and is never constant, so testing the box
/// declines 100% of the time); `getfield_gc_r(frame, execution_context)` +
/// `getfield_gc_r(ec, topframeref)` + `ptr_eq` + `guard_true`, the port of
/// `_do_jit_force_virtual`'s identity check; and one non-forcing void `Call`
/// for `mark_as_escaped`.  At depth 0 the result IS
/// `standard_virtualizable_box()`, exactly as `_do_jit_force_virtual` returns
/// `standard_box`.
///
/// For constant depth >= 1, the same top-frame proof seeds the walk, then each
/// hop emits the `ExecutionContext.getnextframe_nohidden(frame)` body shape:
/// raw `frame.f_backref` read, the residual `OS_JIT_FORCE_VIRTUAL`
/// `CallMayForceR(jit_force_vref, raw)` bracketed by FORCE_TOKEN/SETFIELD_GC and
/// `GuardNotForced`, `GuardNonnull` for the "call stack is not deep enough"
/// arm, then `frame.hide()` as `frame.pycode.hidden_applevel` with a
/// `GuardFalse`.  The residual force stays in the trace: `pyjitpl.py
/// _do_jit_force_virtual` returns `None` for a known non-standard
/// virtualizable, and that `None` result is precisely the caller's signal to
/// emit the residual `jit_force_virtual` call.  `optimize_jit_force_virtual`
/// only elides it later for a trace Virtual, matching upstream and giving the
/// cheap per-hop cost instead of the full `_getframe` residual.
///
/// The guard reads `topframeref` raw, without the vref force or the
/// hidden-frame walk `gettopframe_nohidden` performs, so the gate below
/// requires the record-time chain to need neither: `topframeref` must BE the
/// portal pointer (not a `JitVirtualRef` naming it) and the nohidden walk must
/// land on the same frame.  Any other chain declines, and at runtime a
/// `topframeref` that stops matching side-exits.
///
/// Returns `None` (fall through to the generic residual) for every other
/// shape: a rebound `sys._getframe`, a bound receiver, a negative / non-int /
/// inexact / non-constant depth, a walk with no frame identity, a missing
/// audit holder, a top-level `topframeref` mismatch with the portal frame, a
/// hop whose forced `f_backref` is null, or a hop whose result is hidden.
/// Armed hooks stay on this arm: the walk and `mark_as_escaped` are traced
/// and only `trigger_audit_events` is residual.
/// Declines after emission rewind to the pre-specialization trace position and
/// reset the heap cache before falling through.
pub(crate) fn try_walker_specialize_sys_getframe<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    // `sys._getframe()` (2) or `sys._getframe(depth)` (3).
    if !(2..=3).contains(&r_args.len()) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl` prepends
    // as arg0, not a plain `sys._getframe(...)` call.
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::module::sys::vm::is_builtin_getframe_function(concrete_callable)
    {
        return Ok(None);
    }
    // The depth has to be an exact plain non-negative int before anything is
    // emitted; the guards below pin both facts for the compiled loop.
    let exact_int_class = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let (depth_arg, depth_value) = if r_args.len() == 3 {
        let ConcreteValue::Ref(depth_obj) = arg_concretes[2] else {
            return Ok(None);
        };
        if depth_obj.is_null()
            || unsafe {
                !std::ptr::eq((*depth_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
                    || !std::ptr::eq((*depth_obj).w_class, exact_int_class)
            }
        {
            return Ok(None);
        }
        let depth = unsafe { pyre_object::w_int_get_value(depth_obj) };
        if depth < 0 {
            return Ok(None);
        }
        (Some(r_args[2]), depth)
    } else {
        (None, 0)
    };
    // `vm.py audit(space, "sys._getframe", [f])`.  With no hook installed
    // `audit` takes its `holder.hooks_w is None` early-out (`vm.py`) and the
    // event costs nothing; the emission below pins that read so a later
    // `addaudithook` revokes this loop instead of silently missing the event.
    // With a hook already installed the walk and `mark_as_escaped` stay in
    // the trace and only `trigger_audit_events` (`@objectmodel.dont_inline`)
    // is a residual. Declining the whole arm would residualise `getframe`,
    // whose `force_frame` clears the virtualizable token and the retrace
    // never becomes a bridge.
    let audit_holder = pyre_interpreter::module::sys::vm::audit_holder_ptr();
    if audit_holder.is_null() {
        return Ok(None);
    }
    let hooks_armed = pyre_interpreter::module::sys::vm::audit_hooks_armed();
    // Every MIFrame owns one red frame. At the root that is the standard
    // virtualizable; inside an inline sub-walk it is the callee frame seeded in
    // `dispatch_inline_call_dr_kind` and carried by `CalleeLocalsShadow`.
    // Starting the constant-depth walk from that per-level frame is the direct
    // counterpart of `ec.gettopframe_nohidden()` returning the live MIFrame's
    // virtual frame upstream.
    let inline_ptr = current_inline_concrete_frame();
    let inline_level = ctx.fbw_mode.inline_subwalk || inline_ptr != 0;
    let (Some(standard_vable_op), Some(standard_vable_ptr)) = (
        ctx.trace_ctx.standard_virtualizable_box(),
        ctx.trace_ctx.standard_virtualizable_ptr(),
    ) else {
        return Ok(None);
    };
    let (vable_op, vable_ptr) = if inline_level {
        let state = ctx.frame_state.borrow();
        let Some(shadow) = state.callee_shadow.as_ref() else {
            return Ok(None);
        };
        if inline_ptr == 0 || shadow.concrete_frame != inline_ptr || shadow.frame_box == OpRef::NONE
        {
            return Ok(None);
        }
        (shadow.frame_box, inline_ptr)
    } else {
        (standard_vable_op, standard_vable_ptr)
    };
    let ec =
        pyre_interpreter::call::getexecutioncontext() as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return Ok(None);
    }
    // At the portal, prove that the raw execution-context chain still names
    // the standard frame and emit the equivalent runtime guard below. An
    // inline level already has the stronger per-MIFrame identity witness:
    // `frame_box` and `concrete_frame` were seeded together when that level was
    // pushed, and the compiled trace carries the same box directly.
    let frame = if inline_level {
        inline_ptr as *mut pyre_interpreter::PyFrame
    } else {
        if unsafe { (*ec).topframeref } as usize != vable_ptr {
            return Ok(None);
        }
        let frame = unsafe { (*ec).gettopframe_nohidden() };
        if frame.is_null() || frame as usize != vable_ptr {
            return Ok(None);
        }
        frame
    };
    // Validate the entire concrete chain before emitting or forcing anything.
    // A tracing-time `JitVirtualRef` is admissible only when it is still one of
    // `MetaInterp.virtualref_boxes`: that is the pair
    // `vrefs_after_residual_call` will publish if this walk forces it.  Reading
    // `forced` here does not force or change the token; vrefs created during
    // tracing already carry the real recording-time frame there
    // (`virtualref.py virtual_ref_during_tracing`).  This all-or-nothing gate
    // keeps a later decline from shortening the concrete frame chain before
    // the generic residual gets a chance to run.
    //
    // A hidden hop declines outright.  `executioncontext.py
    // getnextframe_nohidden` skips a hidden frame WITHOUT consuming a depth
    // level, so one raw `f_backref` per level only reproduces `getframe`'s walk
    // on a chain that has none; the emitted traversal pins that with its
    // per-hop `guard_false(hidden_applevel)`.
    let final_concrete_frame = {
        let mut scan = frame;
        for _ in 0..depth_value {
            let raw = unsafe { (*scan).f_backref };
            if raw.is_null() {
                return Ok(None);
            }
            if unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(raw as *const u8) } {
                let referent = unsafe {
                    majit_metainterp::virtualref::vref_forced(raw as *const u8)
                        as *mut pyre_interpreter::PyFrame
                };
                if referent.is_null()
                    || (ctx
                        .trace_ctx
                        .live_virtualref_pair_for_ptr(raw as usize)
                        .is_none()
                        && ctx
                            .trace_ctx
                            .virtualref_virtual_for_object_ptr(referent as usize)
                            .is_none())
                {
                    if fbw_debug_abort_enabled() {
                        let pairs = ctx.trace_ctx.snapshot_virtualref_boxes();
                        eprintln!(
                            "[getframe-decline] depth={depth_value} vref {:#x} referent={:#x} has no pair; tracked={pairs:?}",
                            raw as usize, referent as usize,
                        );
                    }
                    return Ok(None);
                }
                scan = referent;
            } else {
                scan = raw;
            }
            if unsafe { (*scan).hide() } {
                return Ok(None);
            }
        }
        scan
    };

    // Until every app-level frame getter is lowered through its own red frame,
    // admitting a landing whose CALL pc is untracked would expose `f_lasti`
    // to the generic heap reader.  `walker_frame_executing_py_pc` now also
    // reads `InlineParentFrame.caller_py_pc` for a farther inlined ancestor.
    if inline_level && depth_value > 0 {
        let landing_ptr = final_concrete_frame as usize;
        let standard_frame = landing_ptr == standard_vable_ptr;
        let tracked_ancestor =
            walker_frame_executing_py_pc(ctx, final_concrete_frame as _, op.pc).is_some();
        let known_red = (standard_frame || tracked_ancestor)
            && unsafe { (*final_concrete_frame).ob_header.ob_type }
                == &pyre_interpreter::pyframe::FRAME_TYPE
            && unsafe {
                (*final_concrete_frame)
                    .code()
                    .flags
                    .contains(pyre_interpreter::CodeFlags::OPTIMIZED)
            };
        let w_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE);
        if !known_red
            || unsafe { (*final_concrete_frame).ob_header.w_class } != w_type
            || unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) } == 0
        {
            return Ok(None);
        }
    }

    // emit the specialized IR (walker-native)
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();

    // `sys` is an ordinary mutable module, so nothing else keeps the name bound
    // to this builtin across iterations.
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    // `@unwrap_spec(depth=int)` and then `jit.isconstant(depth)`: unbox first,
    // and require the UNBOXED value to be the trace constant.  The unbox is
    // only sound behind the class guards, so the constness decline rewinds
    // rather than being hoisted above them.
    if let Some(depth_op) = depth_arg {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        walker_guard_exact_instance(ctx, op.pc, depth_op, int_type_addr, exact_int_class)?;
        let raw = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            depth_op,
            crate::descr::int_intval_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(raw, majit_ir::Value::Int(depth_value));
        if !raw.is_constant() {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    }
    if !inline_level {
        // `ec = space.getexecutioncontext()` is the portal's second red,
        // carried independently of the virtualizable frame.
        let Some(ec_op) = walker_ensure_execution_context(ctx) else {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        };
        // `f = ec.gettopframe_nohidden()` followed by
        // `_do_jit_force_virtual`'s standard-box identity guard.
        let topframeref_op = ctx.trace_ctx.record_op_with_descr(
            OpCode::GetfieldGcR,
            &[ec_op],
            crate::descr::ec_topframeref_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(
            topframeref_op,
            majit_ir::Value::Ref(majit_ir::GcRef(vable_ptr)),
        );
        walker_guard_stamped_ptr_eq(ctx, op.pc, topframeref_op, vable_op)?;
    }

    let mut cur_op = vable_op;
    let mut cur_ptr = frame;
    for _ in 0..depth_value {
        let raw_ptr = unsafe { (*cur_ptr).f_backref };
        let raw_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            cur_op,
            crate::descr::pyframe_f_backref_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(
            raw_op,
            majit_ir::Value::Ref(majit_ir::GcRef(raw_ptr as usize)),
        );

        let raw_is_vref =
            unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(raw_ptr as *const u8) };
        let (next_op, next_ptr) = if raw_is_vref {
            // `_do_jit_force_virtual` sees the vref box as a known
            // non-standard virtualizable and returns None, so
            // `do_residual_call` executes the may-force call.  Run the exact
            // vref bracket around that concrete force: the post half records
            // `VIRTUAL_REF_FINISH(vref, virtual)` before the CALL and replaces
            // the tracked vref with CONST_NULL (`pyjitpl.py`).  The optimizer
            // can then forward JIT_FORCE_VIRTUAL to the paired frame instead
            // of materialising a vref whose `forced` field is null.
            let live_pair = ctx.trace_ctx.live_virtualref_pair_for_ptr(raw_ptr as usize);
            let referent = unsafe {
                majit_metainterp::virtualref::vref_forced(raw_ptr as *const u8)
                    as *mut pyre_interpreter::PyFrame
            };
            let virtual_op = live_pair.map(|pair| pair.0).unwrap_or_else(|| {
                ctx.trace_ctx
                    .virtualref_virtual_for_object_ptr(referent as usize)
                    .expect("the pre-emission frame-chain census accepted this stopped vref")
            });
            // A field read can produce an alias box even though its concrete
            // value is the tracked vref.  Upstream's heapcache normally hands
            // `_do_jit_force_virtual` the tracked box directly.  Preserve
            // that identity for the optimizer after proving the alias at
            // runtime; `VIRTUAL_REF_FINISH` and JIT_FORCE_VIRTUAL must name
            // the same vref box for `optimize_jit_force_virtual` to forward
            // the result to `virtual_op`.
            let force_arg = if let Some((_, vref_op)) = live_pair {
                if raw_op != vref_op {
                    walker_guard_stamped_ptr_eq(ctx, op.pc, raw_op, vref_op)?;
                }
                vref_op
            } else {
                raw_op
            };
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
            ctx.trace_ctx.vrefs_before_residual_call();
            let next_ptr = pyre_interpreter::executioncontext::force_vref(raw_ptr);
            ctx.trace_ctx.vrefs_after_residual_call();
            let force_fn = crate::helpers::jit_force_vref as *const ();
            let forced_op = ctx.trace_ctx.call_typed_with_effect(
                OpCode::CallMayForceR,
                force_fn,
                &[force_arg],
                &[majit_ir::Type::Ref],
                majit_ir::Type::Ref,
                majit_ir::EffectInfo::new(
                    majit_ir::ExtraEffect::ForcesVirtualOrVirtualizable,
                    majit_ir::OopSpecIndex::JitForceVirtual,
                ),
            );
            ctx.trace_ctx.set_opref_concrete(
                forced_op,
                majit_ir::Value::Ref(majit_ir::GcRef(next_ptr as usize)),
            );
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            // `VirtualRefFinish(vref, virtual)` immediately before the call is
            // the optimizer proof that `forced_op == virtual_op`.  Preserve
            // the orthodox force in IR while letting the source walker follow
            // the same forwarded box immediately.
            (virtual_op, next_ptr)
        } else if raw_ptr as usize == standard_vable_ptr {
            // `_do_jit_force_virtual`: the standard virtualizable identity
            // short-circuits before residual-call preparation.  Heapcache
            // normally gives us the same OpRef; keep the runtime proof for an
            // alias box, matching its PTR_EQ + implement_guard_value arm.
            if raw_op != standard_vable_op {
                walker_guard_stamped_ptr_eq(ctx, op.pc, raw_op, standard_vable_op)?;
            }
            (standard_vable_op, raw_ptr)
        } else {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[getframe-decline] depth={depth_value} non-vref hop {:#x} != standard {standard_vable_ptr:#x}",
                    raw_ptr as usize
                );
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        };
        if next_ptr.is_null() || unsafe { (*next_ptr).hide() } {
            unreachable!("the pre-emission frame-chain census accepted this hop")
        }
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[next_op])?;

        let code_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            next_op,
            crate::descr::pyframe_code_descr(),
        );
        let code_ptr = unsafe { (*next_ptr).pycode };
        ctx.trace_ctx.set_opref_concrete(
            code_op,
            majit_ir::Value::Ref(majit_ir::GcRef(code_ptr as usize)),
        );
        // `optimizer.py` reaches for `descr.get_parent_descr()` only
        // when arg0 carries no pointer info yet; a preceding `GUARD_CLASS`
        // gives it `info.InstancePtrInfo()` and that lookup does not run here.
        // The PyCode field group also carries its parent for paths that reach
        // the read without pointer info.  Keep the guard in this path because
        // it validates the concrete pointer before the hidden flag is read —
        // the same order the code-field arm of
        // [`try_walker_specialize_traceback_walk`] uses.
        let code_type_addr = &pyre_interpreter::pycode::CODE_TYPE as *const _ as i64;
        if code_ptr.is_null()
            || unsafe {
                !std::ptr::eq(
                    (*code_ptr.cast::<pyre_object::PyObject>()).ob_type,
                    &pyre_interpreter::pycode::CODE_TYPE,
                )
            }
        {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        walker_guard_class(ctx, op.pc, code_op, code_type_addr)?;
        let hidden_op = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            code_op,
            crate::descr::pycode_hidden_applevel_descr(),
        );
        ctx.trace_ctx
            .set_opref_concrete(hidden_op, majit_ir::Value::Int(0));
        walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[hidden_op])?;

        cur_op = next_op;
        cur_ptr = next_ptr;
    }

    // A depth-zero inline result exposes this callee frame. Publish its current
    // coordinate and any locals that were still held in the strict-fold shadow
    // before the frame becomes observable. This is the same per-frame state
    // the ordinary residual force path flushes, without forcing the outer
    // portal virtualizable or aborting its trace.
    if inline_level && depth_value == 0 {
        residual_call::record_and_publish_inline_callee_last_instr(ctx, op.pc);
        disarm_folded_inline_callee_after_escape(ctx, op.pc)?;
    }

    // A positive-depth inline walk can land on the standard portal frame.
    // Its symbolic `last_instr` is already current in `virtualizable_boxes`
    // (the caller CALL boundary was mirrored there before descending), and
    // residual-call preparation will emit that shadow's store before any
    // runtime frame reader.  Keep the recording-time concrete frame in step
    // with the same value so a getter executed while recording observes the
    // coordinate the compiled trace will publish, rather than baking the
    // frame's stale pre-inline heap value into the trace.
    if cur_op == standard_vable_op
        && cur_ptr as usize == standard_vable_ptr
        && let Some((_, majit_ir::Value::Int(last_instr))) = ctx
            .trace_ctx
            .virtualizable_entry_at(crate::virtualizable_spec::LAST_INSTR_VABLE_FIELD_INDEX)
    {
        // Journaled like the per-opcode publication: this store lands whether
        // or not the walk commits, and a walk that does not commit replays the
        // frame from its pre-walk coordinate.
        crate::jitcode_dispatch::fbw_note_last_instr_undo(cur_ptr as usize);
        unsafe { (*cur_ptr).last_instr = last_instr as isize };
    }

    // `f.mark_as_escaped()` — vm.py.  `escaped` is not one of the six fields
    // `interp_jit.py:25-30` declares, so the store cannot force; it is
    // load-bearing at `executioncontext.py leave`, which forces the
    // leaving frame's own vref only for a frame that escaped.  Upstream traces
    // it as the ordinary `setfield_gc` on the flag, so it is emitted as the
    // read/or/store the `tb_frame` fold above already uses — an opaque call
    // would hide the update from the optimizer and its heap cache.
    let flags_descr = crate::descr::pyframe_flags_descr();
    let live_flags = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, cur_op, flags_descr.clone());
    let escaped_bit = ctx
        .trace_ctx
        .const_int(i64::from(pyre_interpreter::PyFrame::FLAG_ESCAPED));
    let new_flags = ctx
        .trace_ctx
        .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[cur_op, new_flags],
        flags_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(cur_op, flags_descr.index(), new_flags);
    // The walk IS the interpreter running, so the recorded store has to take
    // effect here too — the residual would have applied it before returning.
    unsafe { (*cur_ptr).mark_as_escaped() };

    // `return f` — at depth 0 `cur_op` is still the standard virtualizable
    // `_do_jit_force_virtual` hands back as `standard_box`; each hop above
    // advanced it to the frame the walk settled on. The result stays that
    // box: returning the audit call's box would hide the virtualizable, and
    // the loop's later `f.f_lineno` would residualise into a force.
    if !hooks_armed {
        // The no-hook early-out: pin the quasi-immutable read so a later
        // `addaudithook` revokes this loop.
        walker_pin_audit_hooks(ctx, op.pc, audit_holder)?;
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', cur_op)?;
        return Ok(Some(DispatchOutcome::Continue));
    }

    // Depth 0 returns the frame executing this call. Publish that opcode
    // before the hook reads `f_lineno`. An ancestor (depth > 0) keeps the
    // coordinate its own caller CALL left on it.
    if depth_value == 0 {
        residual_call::publish_portal_executing_last_instr(ctx, cur_op, cur_ptr);
    }
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
    let returned = pyre_interpreter::module::sys::vm::jit_audit_sys_getframe(
        cur_ptr as pyre_object::PyObjectRef,
    );
    // `graphanalyze.py analyze_external_call`: the hook is arbitrary Python,
    // so the effect is `MOST_GENERAL`. `record_call_with_descr` then
    // invalidates the tracing heap cache, and the optimizer's random-effects
    // arm flushes lazy sets before the call. An empty write set left a later
    // `state.x` read on the pre-hook value.
    //
    // The opcode stays `CallN`. `CallMayForceN` stores the following
    // `GuardNotForced` resume descr (`_store_force_index_if_next_guard`),
    // which is what an escaping hook needs, but it also arms the token.
    // This hook then reads `f_lineno` from outside the trace and forces the
    // portal frame on every iteration. That guard is `is_guard_forced` and
    // is never bridged (`forced_never_compiled`). PyPy does not take that
    // path: `trigger_audit_events` is `@dont_inline` but not
    // `dont_look_inside`, and the bridge traces the hook, lowering
    // `f_lineno` to `offset2lineno` on the call's constant pc
    // (`virtualizables forced: 0`). Arming the force descr on this opaque
    // residual is the opposite shape.
    ctx.trace_ctx.call_void_typed_with_effect(
        pyre_interpreter::module::sys::vm::jit_audit_sys_getframe as *const (),
        &[cur_op],
        &[majit_ir::Type::Ref],
        majit_ir::EffectInfo::MOST_GENERAL.clone(),
    );
    if returned.is_null() {
        let exc = pyre_interpreter::eval::get_current_exception();
        let exc_op = ctx.trace_ctx.const_ref(exc as i64);
        ctx.set_last_exc_value(exc_op, ConcreteValue::Ref(exc));
        walker_record_guard_exception(ctx, op.pc)?;
        let exc_concrete = ctx.last_exc_value_concrete();
        let exc_box = ctx.last_exc_value().unwrap_or(exc_op);
        return Ok(Some(DispatchOutcome::SubRaise {
            exc: exc_box,
            exc_concrete,
        }));
    }
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', cur_op)?;
    ctx.live_after_jit_pc = op.next_pc;
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    Ok(Some(DispatchOutcome::Continue))
}

// ── the generated `math` float folds ──────────────────────────────────
//
/// Lower PyPy's `sys.exc_info()` at pyre's generated `CallFn` boundary.
///
/// Upstream `function.py funccall_valuestack` recognizes
/// `space._code_of_sys_exc_info` and calls `vm.py exc_info_direct`. That
/// helper still looks inside `exc_info_with_tb` when the following bytecode
/// can observe traceback slot 2 — it only *omits* the traceback on the
/// proven `exc_info()[0]` / `[:2]` shapes. A null `sys_exc_operror` is
/// `executioncontext.py sys_exc_info`: if `current_gen_or_coroutine` is
/// also None the answer is `(None, None, None)`; a live generator chain
/// walks `_get_topmost_exception` and stays residual because that graph
/// contains a loop.
///
/// The walker fold uses the same `ec_sys_exc_value_descr` Arc as
/// PUSH_EXC_INFO / POP_EXCEPT so a handler restore forwards through the
/// heapcache instead of re-reading a second identity.
pub(crate) fn try_walker_specialize_sys_exc_info<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    macro_rules! decline {
        ($reason:literal) => {{
            if std::env::var_os("PYRE_FBW_INLINE_DIAG").is_some() {
                eprintln!("[sys-exc-info-decline] pc={} why={}", op.pc, $reason);
            }
            return Ok(None);
        }};
    }
    if r_args.len() != 2 {
        decline!("not a zero-argument CallFn shape");
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        decline!("callable or null_or_self is not concrete");
    };
    if concrete_callable.is_null()
        || !null_or_self.is_null()
        || !pyre_interpreter::module::sys::vm::is_builtin_exc_info_function(concrete_callable)
    {
        decline!("not the canonical plain sys.exc_info callable");
    }

    // Every inlined Python level owns its own red frame. Its concrete twin is
    // used only for the green bytecode/look-ahead decision; exception state is
    // read below from the symbolic EC red.
    let Some((_frame_op, frame_ptr)) = walker_executing_frame_box(ctx) else {
        decline!("no per-level red frame");
    };
    let frame = frame_ptr as *mut pyre_interpreter::PyFrame;
    if frame.is_null() {
        decline!("per-level red frame is null");
    }
    let Some((_frame_op, call_py_pc)) =
        walker_frame_executing_py_pc(ctx, frame as pyre_object::PyObjectRef, op.pc)
    else {
        decline!("no executing Python coordinate for red frame");
    };
    let include_traceback =
        pyre_interpreter::module::sys::vm::exc_info_result_needs_traceback_for_call(
            unsafe { &*frame },
            call_py_pc as usize,
        );

    let concrete_exc = pyre_interpreter::eval::get_current_exception();
    let ec_live = pyre_interpreter::call::getexecutioncontext();
    let concrete_gen = if ec_live.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*ec_live).current_gen_or_coroutine }
    };
    let slot_is_exception =
        !concrete_exc.is_null() && unsafe { pyre_object::is_exception(concrete_exc) };
    let gen_is_empty = concrete_gen.is_null() || unsafe { pyre_object::is_none(concrete_gen) };
    if !slot_is_exception && !gen_is_empty {
        // `sys_exc_info` walks `_get_topmost_exception` here. That graph
        // contains a loop, so PyPy's JitPolicy leaves it residual.
        decline!("generator chain may hide a parked exception");
    }
    if !slot_is_exception && !concrete_exc.is_null() {
        decline!("direct EC exception slot is not an exception");
    }

    let Some(ec) = walker_ensure_execution_context(ctx) else {
        decline!("walk carries no EC red");
    };

    let (concrete_class, kind, concrete_tb, concrete_tb_frame, concrete_tuple, concrete_layout) =
        if slot_is_exception {
            let concrete_class = pyre_interpreter::baseobjspace::exception_getclass(concrete_exc);
            if concrete_class.is_null() || unsafe { (*concrete_exc).w_class } != concrete_class {
                // `typedef::type` can obtain a registry class when the physical
                // exception's w_class is still a generic stub. The direct field
                // read below cannot reproduce that branch, so leave it to the
                // wrapper.
                decline!("exception_getclass is not the live w_class field");
            }
            let kind =
                unsafe { pyre_object::interp_exceptions::w_exception_get_kind(concrete_exc) };
            let (concrete_tb, concrete_tb_frame) = if include_traceback {
                let tb = unsafe {
                    pyre_object::interp_exceptions::w_exception_get_traceback(concrete_exc)
                };
                if tb.is_null() || unsafe { pyre_object::is_none(tb) } {
                    (pyre_object::w_none(), std::ptr::null_mut())
                } else if unsafe { pyre_interpreter::pytraceback::is_pytraceback(tb) } {
                    let frame =
                        unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_frame(tb) };
                    if frame.is_null() {
                        decline!("traceback fold has no frame");
                    }
                    (tb, frame)
                } else {
                    decline!("traceback slot is not a PyTraceback");
                }
            } else {
                (pyre_object::w_none(), std::ptr::null_mut())
            };
            let concrete_tuple = pyre_object::w_tuple_new_array_backed(vec![
                concrete_class,
                concrete_exc,
                concrete_tb,
            ]);
            if concrete_tuple.is_null() {
                decline!("concrete three-tuple allocation failed");
            }
            let concrete_layout = unsafe { (*concrete_exc).ob_type } as *const _ as i64;
            (
                concrete_class,
                Some(kind),
                concrete_tb,
                concrete_tb_frame,
                concrete_tuple,
                concrete_layout,
            )
        } else {
            let concrete_tuple = pyre_object::w_tuple_new_array_backed(vec![
                pyre_object::w_none(),
                pyre_object::w_none(),
                pyre_object::w_none(),
            ]);
            if concrete_tuple.is_null() {
                decline!("concrete none-tuple allocation failed");
            }
            (
                std::ptr::null_mut(),
                None,
                pyre_object::w_none(),
                std::ptr::null_mut(),
                concrete_tuple,
                0,
            )
        };

    // commit: no declines below this point
    walker_guard_stamped_ref(ctx, op.pc, r_args[0], concrete_callable)?;

    let exc = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        exc,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_exc as usize)),
    );

    if !slot_is_exception {
        // `exc_info_with_tb` / `sys_exc_info` empty arm: both the handled
        // exception and the generator head are None.
        walker_guard_stamped_isnull(ctx, op.pc, exc)?;
        let gen_head = ctx.trace_ctx.record_op_with_descr(
            OpCode::GetfieldGcR,
            &[ec],
            crate::descr::ec_current_gen_or_coroutine_descr(),
        );
        ctx.trace_ctx.set_opref_concrete(
            gen_head,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_gen as usize)),
        );
        walker_guard_stamped_isnull(ctx, op.pc, gen_head)?;
        let none = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
        let tuple = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &[none, none, none]);
        ctx.trace_ctx.set_opref_concrete(
            tuple,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
        );
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
        return Ok(Some(()));
    }

    walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[exc])?;
    walker_guard_class(ctx, op.pc, exc, concrete_layout)?;

    let exc_class = walker_record_getfield_gc_r_uncached(ctx, exc, crate::descr::w_class_descr());
    ctx.trace_ctx.set_opref_concrete(
        exc_class,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_class as usize)),
    );
    walker_guard_stamped_ref_hold(ctx, op.pc, exc_class, concrete_class)?;
    let tb_op = if include_traceback && !unsafe { pyre_object::is_none(concrete_tb) } {
        let raw_tb = walker_record_getfield_gc_r_uncached(
            ctx,
            exc,
            crate::descr::w_exception_traceback_descr_for(
                kind.expect("handled exception has a kind"),
                unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete_exc) },
            ),
        );
        walker_guard_stamped_nonnull(ctx, op.pc, raw_tb, concrete_tb)?;
        // `error.py OperationError.get_traceback` marks the node's frame
        // escaped so `ExecutionContext.leave` forces its vref. The bit has
        // to be set by the compiled loop, not only on this walk.
        let frame_ref = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            raw_tb,
            crate::descr::pytraceback_frame_descr(),
        );
        walker_guard_stamped_nonnull(
            ctx,
            op.pc,
            frame_ref,
            concrete_tb_frame as pyre_object::PyObjectRef,
        )?;
        let flags_descr = crate::descr::pyframe_flags_descr();
        let live_flags =
            crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, frame_ref, flags_descr.clone());
        let escaped_bit = ctx
            .trace_ctx
            .const_int(i64::from(pyre_interpreter::PyFrame::FLAG_ESCAPED));
        let new_flags = ctx
            .trace_ctx
            .record_op(OpCode::IntOr, &[live_flags, escaped_bit]);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[frame_ref, new_flags],
            flags_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setfield_cached(frame_ref, flags_descr.index(), new_flags);
        unsafe { pyre_interpreter::pytraceback::mark_traceback_escaped(concrete_tb) };
        raw_tb
    } else {
        ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
    };
    let tuple = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &[exc_class, exc, tb_op]);
    ctx.trace_ctx.set_opref_concrete(
        tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_tuple as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', tuple)?;
    Ok(Some(()))
}

/// `int(x)` for an exact float whose truncated value fits a machine Signed.
///
/// The `-2**63 <= x < 2**63` guards run first. The success arm walks
/// `_int_from_trunc` (`to_int_unchecked` plus the managed int alloc).
/// NaN, infinity, out-of-range values, subclasses, and rebound
/// constructors stay on the residual, which owns `newlong_from_float`.
pub(crate) fn try_walker_orthodox_int_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((concrete_callable, [arg_obj, _])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };
    let int_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    if !std::ptr::eq(concrete_callable, int_type_obj) {
        return Ok(None);
    }
    let value = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) || !pyre_object::is_float(arg_obj) {
            return Ok(None);
        }
        pyre_object::w_float_get_value(arg_obj)
    };
    // `2**63` is exactly representable while `i64::MAX` is not; use a strict
    // upper bound, matching ovfcheck_float_to_int on a signed 64-bit target.
    const SIGNED_MIN_AS_FLOAT: f64 = -9223372036854775808.0;
    const SIGNED_LIMIT_AS_FLOAT: f64 = 9223372036854775808.0;
    if !(value >= SIGNED_MIN_AS_FLOAT && value < SIGNED_LIMIT_AS_FLOAT) {
        return Ok(None);
    }
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_int(boxed_result) } {
        return Ok(None);
    }

    let pre_guards = ctx.trace_ctx.get_trace_position();
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let arg_op = r_args[2];
    let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
    let raw_float = walker_unbox_float(ctx, op.pc, arg_op, float_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op.pc,
        arg_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(raw_float, majit_ir::Value::Float(value));
    let low = ctx
        .trace_ctx
        .const_float(SIGNED_MIN_AS_FLOAT.to_bits() as i64);
    let high = ctx
        .trace_ctx
        .const_float(SIGNED_LIMIT_AS_FLOAT.to_bits() as i64);
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatGe, &[raw_float, low], true)?;
    walker_float_cmp_guard(ctx, op.pc, OpCode::FloatLt, &[raw_float, high], true)?;

    if try_walker_orthodox_descent(
        ctx,
        op.pc,
        &[],
        &[],
        &[(raw_float, value)],
        dst,
        'r',
        &INT_FROM_TRUNC_DESCENT,
    )?
    .is_none()
    {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_guards);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    Ok(Some(()))
}

const INT_FROM_TRUNC_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::_int_from_trunc",
    commit_label: "int_from_trunc_commit",
    call_site_label: "int_from_trunc_call_site",
    decline_tag: "INT-FROM-TRUNC-SUBWALK",
};

/// Read a plain `bh_call_fn(callable, PY_NULL, args…)` shape's concrete
/// operands.  `None` means the call is not that shape — a bound receiver in
/// `null_or_self`, a NULL operand, or a non-`Ref` concrete.
fn plain_builtin_call_concretes<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    arity: usize,
) -> Option<(pyre_object::PyObjectRef, [pyre_object::PyObjectRef; 2])> {
    if r_args.len() != arity + 2 {
        return None;
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return None;
    };
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return None;
    }
    let mut operands = [pyre_object::PY_NULL; 2];
    for (slot, concrete) in operands.iter_mut().zip(&arg_concretes[2..arity + 2]) {
        let ConcreteValue::Ref(obj) = *concrete else {
            return None;
        };
        if obj.is_null() {
            return None;
        }
        *slot = obj;
    }
    Some((concrete_callable, operands))
}

/// Pin a concrete ref a fold baked in.  Guard only when the box is not
/// already that constant.
fn walker_guard_fold_callable<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    callable_op: OpRef,
    concrete_callable: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    if callable_op.is_constant() {
        return Ok(());
    }
    let expected = ctx.trace_ctx.const_ref(concrete_callable as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[callable_op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(callable_op, expected);
    Ok(())
}

/// [`walker_guard_fold_callable`] recorded through
/// [`walker_emit_fold_guard_with_snapshot`], which stamps the constant onto
/// the guarded box for bridge recipes.  Returns the interned expected box
/// so a later pin can reuse it.
fn walker_guard_stamped_ref<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let expected = ctx.trace_ctx.const_ref(concrete as i64);
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
    if !op.is_constant() {
        ctx.trace_ctx.heap_cache_mut().replace_box(op, expected);
    }
    Ok(expected)
}

/// [`walker_guard_stamped_ref`] skipped when [`walker_ref_box_is`] already holds.
fn walker_guard_stamped_ref_unless_is<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let expected = ctx.trace_ctx.const_ref(concrete as i64);
    if !walker_ref_box_is(ctx, op, concrete) {
        walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
        ctx.trace_ctx.heap_cache_mut().replace_box(op, expected);
    }
    Ok(expected)
}

/// [`walker_guard_stamped_ref`] skipped when the box is already constant.
fn walker_guard_stamped_ref_unless_const<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let expected = ctx.trace_ctx.const_ref(concrete as i64);
    if !op.is_constant() {
        walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
        ctx.trace_ctx.heap_cache_mut().replace_box(op, expected);
    }
    Ok(expected)
}

/// [`walker_guard_stamped_ref_unless_const`] without `replace_box`.
fn walker_guard_stamped_ref_pin<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let expected = ctx.trace_ctx.const_ref(concrete as i64);
    if !op.is_constant() {
        walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
    }
    Ok(expected)
}

/// Pin a concrete int a fold baked in (strategy word, length, version tag).
/// These boxes are getfield results, so the guard always records.
fn walker_guard_fold_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    value: i64,
) -> Result<(), DispatchError> {
    let expected = ctx.trace_ctx.const_int(value);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[op, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    ctx.trace_ctx.heap_cache_mut().replace_box(op, expected);
    Ok(())
}

/// Walker-native `guard_list_strategy`: getfield `strategy` then GuardValue.
fn walker_guard_fold_list_strategy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    list_op: OpRef,
    sid: i64,
) -> Result<(), DispatchError> {
    let strategy = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_strategy_descr(),
    );
    walker_guard_fold_int(ctx, pc, strategy, sid)
}

/// GETFIELD `version_tag` then unstamped GuardValue.
/// Distinct from [`walker_pin_type_version_tag`] (quasiimmut + GuardNotInvalidated).
fn walker_guard_fold_type_version<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    w_type: pyre_object::PyObjectRef,
    version_tag: i64,
) -> Result<OpRef, DispatchError> {
    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    let descr = crate::descr::type_version_tag_descr();
    let vt_op = walker_record_getfield_gc_i_uncached(ctx, type_const, descr);
    walker_guard_fold_int(ctx, pc, vt_op, version_tag)?;
    Ok(type_const)
}

/// [`walker_guard_fold_int`] recorded through
/// [`walker_emit_fold_guard_with_snapshot`].
fn walker_guard_stamped_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    value: i64,
) -> Result<OpRef, DispatchError> {
    let expected = ctx.trace_ctx.const_int(value);
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
    if !op.is_constant() {
        ctx.trace_ctx.heap_cache_mut().replace_box(op, expected);
    }
    Ok(expected)
}

/// Getfield dict `strategy` then stamped GuardValue.
fn walker_guard_stamped_dict_strategy<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dict_op: OpRef,
    sid: i64,
) -> Result<(), DispatchError> {
    let descr = crate::descr::dict_strategy_word_descr();
    let strategy = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, dict_op, descr);
    walker_guard_stamped_int(ctx, pc, strategy, sid)?;
    Ok(())
}

/// Pin an arraylen a fold baked in. `GuardValue` without `replace_box`:
/// later item reads still use the length box.
fn walker_guard_stamped_len<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    value: i64,
) -> Result<(), DispatchError> {
    let expected = ctx.trace_ctx.const_int(value);
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
    Ok(())
}

/// [`walker_guard_stamped_len`] for a ref. `GuardValue` without `replace_box`.
fn walker_guard_stamped_ref_hold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let expected = ctx.trace_ctx.const_ref(concrete as i64);
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardValue, &[op, expected])?;
    Ok(expected)
}

/// Emit an unstamped `GuardClass` when the box is not constant and its class
/// is not yet known. Always stamps `class_now_known`.
fn walker_guard_fold_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    obj: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    if !obj.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    Ok(())
}

/// Unstamped `GuardClass` plus GETFIELD `w_class` + `GuardValue`.
/// `is_plain_int1` / `is_plain_float_strict` read `value.w_class`.
fn walker_guard_fold_value_w_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    value_op: OpRef,
    value: pyre_object::PyObjectRef,
    value_type_addr: i64,
) -> Result<(), DispatchError> {
    walker_guard_fold_class(ctx, pc, value_op, value_type_addr)?;
    let w_class_ref =
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, value_op, crate::descr::w_class_descr());
    walker_guard_fold_callable(ctx, pc, w_class_ref, unsafe { (*value).w_class })
}

/// Unstamped `GuardClass` tagged with a FOR_ITER green-key FailDescr when
/// one is available. Skips when the box is constant or the class is already
/// known; always stamps `class_now_known`.
///
/// The FailDescr is minted ahead of the guard so a runtime failure — a
/// definitive polymorphism witness — demotes the specialization by descr
/// identity, independent of the guard's per-trace fail index.
/// `store_final_boxes_in_guard` preserves an existing `ResumeGuardDescr`
/// (only refreshing `fail_arg_types`), so the tag survives optimizer
/// guard-folding and unroll; a copied guard chases `prev` to this donor.
/// With no green key the guard is untagged and the site is never demoted.
fn walker_guard_fold_class_foriter<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    obj: OpRef,
    type_addr: i64,
    green_key: Option<u64>,
) -> Result<(), DispatchError> {
    if !obj.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(type_addr);
        match green_key {
            Some(green_key) => {
                let descr = majit_metainterp::make_resume_guard_descr_range_foriter(green_key);
                ctx.trace_ctx.record_guard_with_descr(
                    OpCode::GuardClass,
                    &[obj, type_const],
                    descr,
                );
            }
            None => {
                ctx.trace_ctx
                    .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
            }
        }
        walker_capture_snapshot_for_last_guard(ctx, pc)?;
    }
    ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    Ok(())
}

/// Emit an unstamped `GuardClass` when the box's class is not yet known.
/// Stamps `class_now_known` with the guard.
fn walker_guard_fold_class_if_unknown<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    obj: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.heap_cache().is_class_known(obj) {
        return Ok(());
    }
    let type_const = ctx.trace_ctx.const_int(type_addr);
    ctx.trace_ctx
        .record_guard(OpCode::GuardClass, &[obj, type_const], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    Ok(())
}

/// Emit a stamped `GuardClass` when the box's class is not yet known.
fn walker_guard_stamped_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    obj: OpRef,
    type_addr: i64,
) -> Result<(), DispatchError> {
    if !ctx.trace_ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.trace_ctx.const_int(type_addr);
        walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardClass, &[obj, type_const])?;
        ctx.trace_ctx.heap_cache_mut().class_now_known(obj);
    }
    Ok(())
}

/// Pin a non-null ref a fold already holds. `GuardNonnull` plus stamp the
/// concrete onto the box for bridge recipes.
fn walker_guard_stamped_nonnull<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardNonnull, &[op])?;
    ctx.trace_ctx
        .set_opref_concrete(op, majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)));
    Ok(())
}

/// Prove two ref boxes name the same pointer. `PtrEq` plus `GuardTrue`.
fn walker_guard_stamped_ptr_eq<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    left: OpRef,
    right: OpRef,
) -> Result<(), DispatchError> {
    let same = ctx.trace_ctx.record_op(OpCode::PtrEq, &[left, right]);
    ctx.trace_ctx
        .set_opref_concrete(same, majit_ir::Value::Int(1));
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardTrue, &[same])?;
    Ok(())
}

/// Pin a null ref a fold already holds. `GuardIsnull` plus `replace_box`
/// onto the interned null, matching `_establish_nullity` (`TraceCtx`,
/// not heapcache-only). Callers stamp their own snapshot concrete first.
fn walker_guard_stamped_isnull<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
) -> Result<OpRef, DispatchError> {
    walker_emit_fold_guard_with_snapshot(ctx, pc, OpCode::GuardIsnull, &[op])?;
    let null_const = ctx.trace_ctx.const_ref(0);
    ctx.trace_ctx.replace_box(op, null_const);
    Ok(null_const)
}

/// Stamped `GuardNonnull` or `GuardIsnull` when the box is not already a
/// constant. Direction follows the recorded presence.
fn walker_guard_stamped_presence<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    present: bool,
) -> Result<(), DispatchError> {
    if op.is_constant() {
        return Ok(());
    }
    let opcode = if present {
        OpCode::GuardNonnull
    } else {
        OpCode::GuardIsnull
    };
    walker_emit_fold_guard_with_snapshot(ctx, pc, opcode, &[op])?;
    Ok(())
}

/// Pin a `Cell`: stamped `GuardClass` of `CELL_TYPE`, uncached GETFIELD
/// `contents`, then `GuardNonnull`/`GuardIsnull` of the recorded boundness.
/// `fast2locals` reads `w_cell_get` (`Cell.contents`); a cell can be rebound
/// or deleted between iterations.
fn walker_pin_cell_contents<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    slot_op: OpRef,
    value: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let cell_type = &pyre_object::nestedscope::CELL_TYPE as *const _ as i64;
    walker_guard_stamped_class(ctx, pc, slot_op, cell_type)?;
    let contents =
        walker_record_getfield_gc_r_uncached(ctx, slot_op, crate::descr::cell_contents_descr());
    ctx.trace_ctx.set_opref_concrete(
        contents,
        majit_ir::Value::Ref(majit_ir::GcRef(value as usize)),
    );
    let guard = if value.is_null() {
        OpCode::GuardIsnull
    } else {
        OpCode::GuardNonnull
    };
    walker_emit_fold_guard_with_snapshot(ctx, pc, guard, &[contents])?;
    Ok(contents)
}

/// Pin a bound method: unstamped `GuardClass` of `METHOD_TYPE`, then
/// always-record `GuardValue` on `w_function`. Returns the `w_self` box.
fn walker_guard_bound_method<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    callable_op: OpRef,
    inner_func: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    walker_guard_fold_class(ctx, pc, callable_op, method_type_addr)?;
    let func_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_function_descr(),
    );
    let expected = ctx.trace_ctx.const_ref(inner_func as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[func_ref, expected], 0);
    walker_capture_snapshot_for_last_guard(ctx, pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(func_ref, expected);
    Ok(crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    ))
}

/// Layout `GuardClass` plus the exact canonical `w_class` pin.
fn walker_guard_exact_instance<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    type_addr: i64,
    w_class: pyre_object::PyObjectRef,
) -> Result<(), DispatchError> {
    walker_guard_class(ctx, pc, op, type_addr)?;
    walker_guard_exact_w_class(ctx, pc, op, w_class)
}

/// Stamped type identity plus `walker_pin_type_version_tag`.
fn walker_guard_stamped_type_version<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    op: OpRef,
    concrete: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let expected = walker_guard_stamped_ref(ctx, pc, op, concrete)?;
    walker_pin_type_version_tag(ctx, pc, expected)?;
    Ok(expected)
}

/// Pin payload layout, uncached `w_class`, and `version_tag`.
fn walker_pin_stamped_instance_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    obj: OpRef,
    concrete_obj: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
) -> Result<OpRef, DispatchError> {
    let phys_type = unsafe { (*concrete_obj).ob_type } as i64;
    walker_guard_stamped_class(ctx, pc, obj, phys_type)?;
    let w_class = walker_record_getfield_gc_r_uncached(ctx, obj, crate::descr::w_class_descr());
    walker_guard_stamped_type_version(ctx, pc, w_class, w_type)
}

/// `allocate_instance` stamps the realbase vtable when `w_class` is that
/// realbase's own type, and `_getusercls` otherwise. Matches
/// `exc_instance_pytype` so a canonical fieldless class and an exact
/// realbase both fold.
fn walker_exc_canonical_layout(
    exc: pyre_object::PyObjectRef,
    kind: pyre_object::interp_exceptions::ExcKind,
) -> Option<bool> {
    let header =
        unsafe { &(*(exc as *const pyre_object::interp_exceptions::W_BaseException)).ob_header };
    let exc_type_ptr = header.ob_type;
    if !std::ptr::eq(
        exc_type_ptr,
        pyre_object::interp_exceptions::exc_instance_pytype(kind, header.w_class),
    ) {
        return None;
    }
    Some(pyre_object::interp_exceptions::exc_typeptr_is_user_layout(
        exc_type_ptr,
    ))
}

/// Pin the authentic exception, intern `message_wtf8` as a ConstPtr, emit
/// `NewWithVtable` plus the `__context__` SETFIELD, and route as `SubRaise`.
/// Shared by the immutable-type and read-only-descriptor attr-raise folds
/// after `exc_instance_pytype` has already passed.
fn walker_emit_canonical_message_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    ec: OpRef,
    err: &pyre_interpreter::PyError,
    exc: pyre_object::PyObjectRef,
    kind: pyre_object::interp_exceptions::ExcKind,
    user: bool,
) -> DispatchOutcome {
    // Message as a trace constant: deterministic under the caller's
    // predicate, so one shared immutable string is exact (the same sharing
    // a `raise TypeError("...")` gets from co_consts). Pin the fresh
    // exception across the string allocation; the recorded ConstPtr slot
    // is forwarded across minor collections by the op-graph walker and
    // rooted by the compiled loop's gcref table thereafter.
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_root = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let msg = pyre_object::w_str_from_wtf8(err.message_wtf8());
    // The root keeps the exception alive across that allocation but does
    // not fix its address: a minor collection moves the object and rewrites
    // the slot, which leaves this local naming a forwarded corpse. Read the
    // address back out of the slot the pin claimed.
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_root);
    let msg_const = ctx.trace_ctx.const_ref(msg as i64);
    let args_list = crate::helpers::emit_rlist_inline(ctx.trace_ctx, &[msg_const]);
    let class_const = ctx
        .trace_ctx
        .const_ref(pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) as i64);
    let new_op = crate::helpers::emit_exception_new_inline(
        ctx.trace_ctx,
        kind,
        class_const,
        args_list,
        user,
    );
    ctx.trace_ctx.heap_cache_mut().class_now_known(new_op);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    walker_chain_exception_context(ctx, ec, new_op, exc, kind, user);

    // Inline-built marker: the downstream raise routing records the frame
    // node via the virtual `record_fresh_application_traceback` instead of
    // the forcing runtime hook (mirrors `try_walker_trace_raise_bare_class`).
    fbw_built_exc_insert(new_op);
    // Residual-executor Err-arm state minus the call itself: seed the
    // standing exception for `SubRaise` (`execute_raised` analogue) and
    // restore the blackhole cell so an aborting walk still delivers the
    // pending raise. The `NewWithVtable` vtable pins the class.
    fbw_count_executed_residual(true, true);
    ctx.set_last_exc_value(new_op, ConcreteValue::Ref(exc));
    ctx.fbw_mode.class_of_last_exc_is_const = true;
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));
    DispatchOutcome::SubRaise {
        exc: new_op,
        exc_concrete: ConcreteValue::Ref(exc),
    }
}

const NEWFLOAT_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::floatobject::newfloat",
    commit_label: "newfloat_commit",
    call_site_label: "newfloat_call_site",
    decline_tag: "NEWFLOAT-SUBWALK",
};

/// `float(x)` on an exact int/float argument.
///
/// An exact int walks `floatobject.py newfloat` after `CastIntToFloat`
/// (`intobject.py descr_float`: `space.newfloat(float(self.intval))`).
/// An exact float is `float(f) is f`. A rebound name or a float subclass
/// (which reboxes) falls through to the generic residual.
pub(crate) fn try_walker_orthodox_float_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((concrete_callable, [arg_obj, _])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };
    // The callable must be the canonical `float` type object.
    let float_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::FLOAT_TYPE);
    if !std::ptr::eq(concrete_callable, float_type_obj) {
        return Ok(None);
    }
    let (is_int, val) = unsafe {
        if !pyre_object::is_exact_builtin_instance(arg_obj) {
            return Ok(None);
        }
        if pyre_object::is_int(arg_obj) {
            (true, pyre_object::w_int_get_value(arg_obj) as f64)
        } else if pyre_object::is_float(arg_obj) {
            (false, pyre_object::w_float_get_value(arg_obj))
        } else {
            return Ok(None);
        }
    };
    // Authentic boxed result (float() is side-effect-free on int/float).
    let boxed_result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &[arg_obj])
    };
    let Ok(boxed_result) = boxed_result else {
        return Ok(None);
    };

    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    let arg_op = r_args[2];
    if is_int {
        // Exact int: `CastIntToFloat`, then `floatobject.py newfloat`.
        // The coercion pins `w_class`: `builtin_float` reads an int payload only
        // for an exact builtin, so an `int` subclass overriding `__float__` must
        // side-exit.  The float arm below pins its own for the same reason.
        // A decline cuts the cast so the generic residual is the only writer.
        let pre_cast = ctx.trace_ctx.get_trace_position();
        let raw = walker_coerce_operand_to_float(ctx, op.pc, arg_op, arg_obj, true, val, false)?;
        if try_walker_orthodox_descent(
            ctx,
            op.pc,
            &[],
            &[],
            &[(raw, val)],
            dst,
            'r',
            &NEWFLOAT_DESCENT,
        )?
        .is_none()
        {
            ctx.trace_ctx.cut_trace_with_snapshots(pre_cast);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    } else {
        // exact float → `float(f) is f`: forward the argument unchanged.  Only
        // sound when the constructor actually returned the same object; a
        // divergence (should not happen for an exact float) declines.
        if !std::ptr::eq(boxed_result, arg_obj) {
            return Ok(None);
        }
        let float_type_addr = &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64;
        walker_guard_fold_class(ctx, op.pc, arg_op, float_type_addr)?;
        walker_guard_exact_w_class(ctx, op.pc, arg_op, float_type_obj)?;
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', arg_op)?;
    }
    Ok(Some(()))
}

const NEWCOMPLEX_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::complexobject::newcomplex",
    commit_label: "newcomplex_commit",
    call_site_label: "newcomplex_call_site",
    decline_tag: "NEWCOMPLEX-SUBWALK",
};

const COMPLEX_REAL_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::complexobject::complex_descr_get_real",
    commit_label: "complex_real_commit",
    call_site_label: "complex_real_call_site",
    decline_tag: "COMPLEX-REAL-SUBWALK",
};

const COMPLEX_IMAG_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::complexobject::complex_descr_get_imag",
    commit_label: "complex_imag_commit",
    call_site_label: "complex_imag_call_site",
    decline_tag: "COMPLEX-IMAG-SUBWALK",
};

fn walker_complex_decline<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pre: majit_metainterp::recorder::TracePosition,
) -> Result<Option<()>, DispatchError> {
    ctx.trace_ctx.cut_trace_with_snapshots(pre);
    ctx.trace_ctx.heap_cache_mut().reset();
    Ok(None)
}

/// `unpackcomplex` calls `__complex__` before `__index__`.
fn complex_arg_has_complex_dunder(obj: pyre_object::PyObjectRef) -> bool {
    let Some(w_type) = pyre_interpreter::typedef::r#type(obj) else {
        return false;
    };
    unsafe {
        pyre_interpreter::baseobjspace::lookup_in_type(w_type.as_ptr(), "__complex__").is_some()
    }
}

fn descend_newcomplex<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    raw: OpRef,
    val: f64,
    dst: usize,
    pre: majit_metainterp::recorder::TracePosition,
) -> Result<Option<()>, DispatchError> {
    let imag = ctx.trace_ctx.const_float(0.0f64.to_bits() as i64);
    if !matches!(
        try_walker_orthodox_descent(
            ctx,
            op_pc,
            &[],
            &[],
            &[(raw, val), (imag, 0.0)],
            dst,
            'r',
            &NEWCOMPLEX_DESCENT,
        )?,
        Some(DispatchOutcome::Continue)
    ) {
        return walker_complex_decline(ctx, pre);
    }
    Ok(Some(()))
}

/// `complex(x)` for one positional on the canonical `complex` type.
///
/// `complexobject.py descr__new__` returns an exact complex unchanged.
/// `unpackcomplex` then reads a bool, an exact machine int, or an exact
/// float and allocates through `newcomplex`. A user `__index__` is inlined
/// even when `__float__` is also present, because `unpackcomplex` calls
/// `space.index` before `space.float`. `__complex__`, an int or float
/// subclass, a long, a string, keywords, and a second argument stay on the
/// residual.
pub(crate) fn try_walker_orthodox_complex_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((concrete_callable, [arg_obj, _])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };
    let complex_type_obj =
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::COMPLEX_TYPE);
    if !std::ptr::eq(concrete_callable, complex_type_obj) {
        return Ok(None);
    }
    let arg_op = r_args[2];
    enum Plan {
        Identity,
        Numeric { is_int: bool, val: f64 },
        Index(IndexInlineCandidate),
    }
    let plan = unsafe {
        if pyre_object::is_exact_type(arg_obj, &pyre_object::pyobject::COMPLEX_TYPE) {
            Some(Plan::Identity)
        } else if pyre_object::is_bool(arg_obj) {
            Some(Plan::Numeric {
                is_int: true,
                val: pyre_object::w_bool_get_value(arg_obj) as i64 as f64,
            })
        } else if pyre_object::is_int(arg_obj) && pyre_object::is_exact_builtin_instance(arg_obj) {
            Some(Plan::Numeric {
                is_int: true,
                val: pyre_object::w_int_get_value(arg_obj) as f64,
            })
        } else if pyre_object::is_float(arg_obj) && pyre_object::is_exact_builtin_instance(arg_obj)
        {
            Some(Plan::Numeric {
                is_int: false,
                val: pyre_object::w_float_get_value(arg_obj),
            })
        } else if pyre_object::is_long(arg_obj)
            || pyre_object::is_float(arg_obj)
            || pyre_object::is_complex(arg_obj)
            || pyre_object::is_int(arg_obj)
            || pyre_object::is_str(arg_obj)
            || pyre_object::is_bytes(arg_obj)
            || pyre_object::is_bytearray(arg_obj)
            || complex_arg_has_complex_dunder(arg_obj)
        {
            // An int subclass stays on the residual. The numeric arm admits
            // only an exact int, and its class guard is the builtin `int`.
            if fbw_inline_diag_enabled() {
                eprintln!("[complex-call-decline] why=conversion-dunder-or-other-type");
            }
            None
        } else {
            match prepare_walker_inline_index(ctx, arg_op, arg_obj) {
                Some(candidate) => Some(Plan::Index(candidate)),
                None => {
                    if fbw_inline_diag_enabled() {
                        eprintln!("[complex-call-decline] why=index-prepare-none");
                    }
                    None
                }
            }
        }
    };
    let Some(plan) = plan else {
        return Ok(None);
    };

    let pre = ctx.trace_ctx.get_trace_position();
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    match plan {
        Plan::Identity => {
            let complex_type_addr = &pyre_object::pyobject::COMPLEX_TYPE as *const _ as i64;
            walker_guard_fold_class(ctx, op.pc, arg_op, complex_type_addr)?;
            walker_guard_exact_w_class(ctx, op.pc, arg_op, complex_type_obj)?;
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', arg_op)?;
            Ok(Some(()))
        }
        Plan::Numeric { is_int, val } => {
            let raw =
                walker_coerce_operand_to_float(ctx, op.pc, arg_op, arg_obj, is_int, val, false)?;
            descend_newcomplex(ctx, op.pc, raw, val, dst, pre)
        }
        Plan::Index(candidate) => {
            let Some((result, ConcreteValue::Ref(concrete))) = try_walker_inline_index(
                ctx, op, code, funcptr, r_args, call_descr, dst, candidate,
            )?
            else {
                return walker_complex_decline(ctx, pre);
            };
            if !walker_is_exact_machine_int_concrete(concrete) {
                return walker_complex_decline(ctx, pre);
            }
            let val = unsafe { pyre_object::w_int_get_value(concrete) } as f64;
            let raw =
                walker_coerce_operand_to_float(ctx, op.pc, result, concrete, true, val, false)?;
            descend_newcomplex(ctx, op.pc, raw, val, dst, pre)
        }
    }
}

/// `complex.real` / `complex.imag` on an exact complex.
///
/// `complexobject.py complexwprop` boxes the lane with `space.newfloat`.
/// The traced leaf is `complex_descr_get_real` / `complex_descr_get_imag`.
/// A subclass receiver stays on the residual getset.
pub(crate) fn try_walker_orthodox_complex_member<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    obj: OpRef,
    name: &str,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' {
        return Ok(None);
    }
    let descent = match name {
        "real" => &COMPLEX_REAL_DESCENT,
        "imag" => &COMPLEX_IMAG_DESCENT,
        _ => return Ok(None),
    };
    let Some(concrete) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    if unsafe { !pyre_object::is_exact_type(concrete, &pyre_object::pyobject::COMPLEX_TYPE) } {
        return Ok(None);
    }
    let pre = ctx.trace_ctx.get_trace_position();
    let complex_type_obj =
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::COMPLEX_TYPE);
    let complex_type_addr = &pyre_object::pyobject::COMPLEX_TYPE as *const _ as i64;
    walker_guard_fold_class(ctx, op_pc, obj, complex_type_addr)?;
    walker_guard_exact_w_class(ctx, op_pc, obj, complex_type_obj)?;
    if !matches!(
        try_walker_orthodox_descent(ctx, op_pc, &[], &[(obj, concrete)], &[], dst, 'r', descent,)?,
        Some(DispatchOutcome::Continue)
    ) {
        return walker_complex_decline(ctx, pre);
    }
    Ok(Some(()))
}

/// `str(i)` / `repr(i)` on an exact `int`: walk `intobject.py descr_str`.
///
/// The generated body is `ll_int2dec` then `newutf8`. The residual it
/// replaces is a `CallMayForce`, so it clears the heap cache and forces
/// virtualizables across itself.
///
/// The callable must be the canonical `str` type object or the `repr`
/// builtin: a rebound name or a `str` subclass reboxes through `__new__`.
/// `repr(i)` shares the body because it renders the same decimal text for
/// an exact `int`. The argument must be an exact `int`: `bool` renders
/// `True`/`False`, an `int` subclass may override `__str__` / `__repr__`,
/// and a `W_LongObject`'s payload is a pointer where `intval` would be.
/// Any other shape falls through to the generic residual.
pub(crate) fn try_walker_orthodox_str_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((concrete_callable, [arg_obj, _])) =
        plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };
    let str_type_obj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE);
    let renders_an_int = std::ptr::eq(concrete_callable, str_type_obj)
        || pyre_interpreter::jit_builtin_folds::is_repr_builtin(concrete_callable);
    if !renders_an_int {
        return Ok(None);
    }
    // A tagged immediate has no header for the `w_class` and unbox guards to
    // read, and this emit is not tag-aware.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(arg_obj) {
        return Ok(None);
    }
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    unsafe {
        if !std::ptr::eq((*arg_obj).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*arg_obj).w_class, int_typeobj)
        {
            return Ok(None);
        }
    }

    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    try_walker_orthodox_int_descr_str(ctx, op.pc, r_args[2], arg_obj, dst)
}

/// Walk `intobject.py descr_str` / `descr_repr`. The generated body is
/// `ll_int2dec` then `newutf8`. A missing jitcode declines.
///
/// `descr_str` reads `intval` with no class test, so the int guards stay
/// here. A failed walk cuts them.
fn try_walker_orthodox_int_descr_str<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
    obj: pyre_object::PyObjectRef,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some(prep) = prepare_orthodox_descent(ctx, op_pc, &INT_DESCR_STR_DESCENT) else {
        return Ok(None);
    };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    walker_guard_exact_instance(ctx, op_pc, operand, int_type_addr, int_typeobj)?;
    let walked = run_prepared_orthodox_descent(
        ctx,
        op_pc,
        prep,
        &[],
        &[(operand, obj)],
        &[],
        dst,
        'r',
        &INT_DESCR_STR_DESCENT,
        None,
        false,
    )?;
    orthodox_descent_unit(ctx, op_pc, walked, Some(pre_fold_pos))
}

const INT_DESCR_STR_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::intobject::descr_str",
    commit_label: "int_descr_str_commit",
    call_site_label: "int_descr_str_call_site",
    decline_tag: "INT-DESCR-STR-SUBWALK",
};

/// Descend `space.newutf8` / `W_UnicodeObject.__init__` instead of the
/// residual wrap.  Banks are int then ref: `length`, then `_utf8`.
pub(crate) fn try_walker_orthodox_newutf8<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    storage: OpRef,
    length: OpRef,
    boxed_result: pyre_object::PyObjectRef,
) -> Result<Option<OpRef>, DispatchError> {
    if boxed_result.is_null()
        || !unsafe { pyre_object::is_exact_type(boxed_result, &pyre_object::STR_TYPE) }
    {
        return Ok(None);
    }
    let Some(prep) = prepare_orthodox_descent(ctx, op_pc, &NEWUTF8_DESCENT) else {
        return Ok(None);
    };
    let payload = unsafe { pyre_object::unicodeobject::w_str_storage(boxed_result) };
    let concrete_len = unsafe {
        (*(boxed_result as *const pyre_object::unicodeobject::W_UnicodeObject)).len as i64
    };
    // The length is a payload, not a `BinaryOperator` tag. Stamp it before
    // the walk; the shared descent stamps ref operands itself and must not
    // read this int as a raising operator.
    ctx.trace_ctx
        .set_opref_concrete(length, majit_ir::Value::Int(concrete_len));
    let mut produced = None;
    // Void bank: the caller writes `dst` after the concrete restamp below.
    let walked = run_prepared_orthodox_descent(
        ctx,
        op_pc,
        prep,
        &[(length, concrete_len)],
        &[(storage, payload as pyre_object::PyObjectRef)],
        &[],
        0,
        'v',
        &NEWUTF8_DESCENT,
        Some(&mut produced),
        false,
    )?;
    match walked {
        Some(DispatchOutcome::Continue) => {}
        None => return Ok(None),
        Some(_) => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    }
    let result = produced.ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?;
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(boxed_result as usize)),
    );
    Ok(Some(result))
}

const NEWUTF8_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::unicodeobject::w_str_from_storage_and_length",
    commit_label: "newutf8_commit",
    call_site_label: "newutf8_call_site",
    decline_tag: "NEWUTF8-SUBWALK",
};

/// `_parse_spec("d", ">")` (`newformat.py`) then `_type == "d"` (default
/// included) with no thousands separator, precision, or `z`.
///
/// Output-shape matching alone is not enough: `format(1, "x") == "1"`
/// and `format(1, "b") == "0b1"` look like an unpadded / left-padded
/// `str(1)`, so a later `10` in the same compiled loop would print
/// decimal instead of `"a"` / `"0b1010"`.
fn spec_is_decimal_int_format(spec: &str) -> bool {
    let chars: Vec<char> = spec.chars().collect();
    let n = chars.len();
    if n == 0 {
        return true;
    }
    let mut i = 0;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        i = 2;
    } else if matches!(chars[0], '<' | '>' | '=' | '^') {
        i = 1;
    }
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        i += 1;
    }
    if i < n && chars[i] == 'z' {
        return false;
    }
    if i < n && chars[i] == '#' {
        i += 1;
    }
    if i < n && chars[i] == '0' {
        i += 1;
    }
    while i < n && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i < n && matches!(chars[i], ',' | '_') {
        return false;
    }
    if i < n && chars[i] == '.' {
        return false;
    }
    let ty = if i < n {
        if i + 1 < n {
            return false;
        }
        chars[i]
    } else {
        'd'
    };
    ty == 'd'
}

/// `intobject::format_int_decimal`: `ll_int2dec`, then `_fill_number`.
const FORMAT_INT_DECIMAL_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_object::intobject::format_int_decimal",
    commit_label: "format_int_decimal_commit",
    call_site_label: "format_int_decimal_call_site",
    decline_tag: "FORMAT-INT-DECIMAL-SUBWALK",
};

/// `newformat.py` `_parse_spec("d", ">")` reduced to the decimal
/// machine-int shape `format_int_decimal` traces: fill, width, align,
/// and a forced `+` / space sign. `'^'` and a non-ASCII fill stay on
/// the residual formatter.
struct DecimalIntFormat {
    fill: char,
    width: i64,
    align: i64,
    forced_sign: Option<char>,
}

fn parse_decimal_int_format(spec: &str) -> Option<DecimalIntFormat> {
    let chars: Vec<char> = spec.chars().collect();
    let n = chars.len();
    if n == 0 {
        return None;
    }
    let mut i = 0;
    let mut fill = ' ';
    let mut align_ch = '>';
    let mut got_align = false;
    let mut got_fill = false;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        fill = chars[0];
        align_ch = chars[1];
        got_align = true;
        got_fill = true;
        i = 2;
    } else if matches!(chars[0], '<' | '>' | '=' | '^') {
        align_ch = chars[0];
        got_align = true;
        i = 1;
    }
    let mut forced_sign = None;
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        if chars[i] != '-' {
            forced_sign = Some(chars[i]);
        }
        i += 1;
    }
    if i < n && chars[i] == 'z' {
        return None;
    }
    if i < n && chars[i] == '#' {
        i += 1;
    }
    if !got_fill && i < n && chars[i] == '0' {
        fill = '0';
        if !got_align {
            align_ch = '=';
        }
        i += 1;
    }
    let mut width: i64 = 0;
    while i < n && chars[i].is_ascii_digit() {
        width = width
            .checked_mul(10)?
            .checked_add((chars[i] as i64) - (b'0' as i64))?;
        i += 1;
    }
    if i < n && (chars[i] != 'd' || i + 1 != n) {
        return None;
    }
    if align_ch == '^' || !fill.is_ascii() {
        return None;
    }
    let align = match align_ch {
        '<' => pyre_object::FORMAT_INT_ALIGN_LEFT,
        '=' => pyre_object::FORMAT_INT_ALIGN_SIGN,
        _ => pyre_object::FORMAT_INT_ALIGN_RIGHT,
    };
    Some(DecimalIntFormat {
        fill,
        width,
        align,
        forced_sign,
    })
}

/// FORMAT_WITH_SPEC on an exact `int` plus a constant decimal spec.
///
/// An empty spec declines. `newformat.py` `format_int_or_long` with no
/// width and no forced sign is `space.str`, so that arm descends
/// `descr_str`. Every other admitted spec descends
/// `format_int_decimal` (`_calc_num_width` / `_fill_number`), which
/// splits the sign off before padding. A bool, a subclass, or a spec
/// that is not a constant exact `str` declines (SAFE).
pub(crate) fn try_walker_specialize_format_with_spec_int<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 2 {
        return Ok(None);
    }
    let Some(concrete) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    let Some(concrete_spec) = walker_concrete_ref_object(ctx, r_args[1]) else {
        return Ok(None);
    };
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(concrete) {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_exact_type(concrete_spec, &pyre_object::STR_TYPE) } {
        return Ok(None);
    }
    let Some(spec_text) = (unsafe { pyre_object::w_str_get_value_opt(concrete_spec) }) else {
        return Ok(None);
    };
    let value = r_args[0];
    let spec = r_args[1];
    // An empty spec is `format_w`'s empty-spec arm; the residual serves it.
    if spec_text.is_empty() || !spec_is_decimal_int_format(spec_text) {
        return Ok(None);
    }
    let Some(parsed) = parse_decimal_int_format(spec_text) else {
        return Ok(None);
    };

    let int_typeobj = pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::INT_TYPE);
    let int_value = unsafe {
        if !std::ptr::eq((*concrete).ob_type, &pyre_object::pyobject::INT_TYPE)
            || !std::ptr::eq((*concrete).w_class, int_typeobj)
        {
            return Ok(None);
        }
        pyre_object::w_int_get_value(concrete)
    };
    walker_guard_stamped_ref_pin(ctx, op.pc, spec, concrete_spec)?;
    if parsed.width == 0 && parsed.forced_sign.is_none() {
        return try_walker_orthodox_int_descr_str(ctx, op.pc, value, concrete, dst);
    }

    let Some(jc) = crate::jitcode_runtime::pathed_jitcode_cached(FORMAT_INT_DECIMAL_DESCENT.path)
    else {
        return Ok(None);
    };
    if jc.calldescr.arg_classes != "iriir" {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[decline-why] FORMAT-INT-DECIMAL-ARG-CLASSES pc={} classes={}",
                op.pc, jc.calldescr.arg_classes
            );
        }
        return Ok(None);
    }

    let pre_body = ctx.trace_ctx.get_trace_position();
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    walker_guard_exact_instance(ctx, op.pc, value, int_type_addr, int_typeobj)?;
    let int_raw = walker_unbox_int_typed(
        ctx,
        op.pc,
        value,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )?;
    ctx.trace_ctx
        .set_opref_concrete(int_raw, majit_ir::Value::Int(int_value));
    let fill_box = pyre_object::w_str_new(&parsed.fill.to_string());
    let fill_storage = unsafe { pyre_object::unicodeobject::w_str_storage(fill_box) };
    let sign_storage = match parsed.forced_sign {
        Some(ch) => unsafe {
            pyre_object::unicodeobject::w_str_storage(pyre_object::w_str_new(&ch.to_string()))
        },
        None => std::ptr::null_mut(),
    };
    let width_op = ctx.trace_ctx.const_int(parsed.width);
    let align_op = ctx.trace_ctx.const_int(parsed.align);
    let fill_op = ctx.trace_ctx.const_ref(fill_storage as i64);
    let sign_op = ctx.trace_ctx.const_ref(sign_storage as i64);
    let outcome = try_walker_orthodox_descent(
        ctx,
        op.pc,
        &[
            (int_raw, int_value),
            (width_op, parsed.width),
            (align_op, parsed.align),
        ],
        &[
            (fill_op, fill_storage as pyre_object::PyObjectRef),
            (sign_op, sign_storage as pyre_object::PyObjectRef),
        ],
        &[],
        dst,
        'r',
        &FORMAT_INT_DECIMAL_DESCENT,
    )?;
    if !matches!(outcome, Some(DispatchOutcome::Continue)) {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_body);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }
    Ok(Some(()))
}

/// `space.divmod(w_x, w_y)`, the body operation.py `divmod` returns.
const DIVMOD_DESCENT: HelperDescent = HelperDescent {
    path: "pyre_interpreter::objspace::descroperation::divmod",
    commit_label: "divmod_commit",
    call_site_label: "divmod_call_site",
    decline_tag: "DIVMOD-SUBWALK",
};

/// `divmod(a, b)`: operation.py `divmod(space, w_x, w_y)` is
/// `space.divmod(w_x, w_y)`, so after pinning the builtin's identity the call
/// descends that body with the recorded operands.  Its own class tests and
/// override probes select the `_divmod` / `_int_divmod` arm.
///
/// Admission is the policy [`try_walker_orthodox_descent`] documents: only an
/// exact builtin numeric operand, whose arms call no Python code. The
/// descent's [`DispatchOutcome::SubRaise`] — a zero divisor's
/// `ZeroDivisionError` included — is returned to the caller.
pub(crate) fn try_walker_orthodox_builtin_divmod<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, a, b)` shape only.
    if r_args.len() != 4 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `divmod(a, b)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }
    if !pyre_interpreter::builtins::is_builtin_divmod_function(concrete_callable) {
        return Ok(None);
    }
    let mut operands = [(OpRef::NONE, std::ptr::null_mut()); 2];
    for (slot, &operand) in operands.iter_mut().zip(&r_args[2..]) {
        let Some(obj) = walker_concrete_ref_object(ctx, operand) else {
            return Ok(None);
        };
        // SAFETY: `obj` is a live concrete `PyObjectRef` from the walker
        // shadow.
        let admitted = unsafe {
            pyre_object::is_exact_builtin_instance(obj)
                && (pyre_object::is_int(obj)
                    || pyre_object::is_bool(obj)
                    || pyre_object::is_float(obj)
                    || pyre_object::is_long(obj))
        };
        if !admitted {
            return Ok(None);
        }
        *slot = (operand, obj);
    }
    walker_guard_fold_callable(ctx, op.pc, r_args[0], concrete_callable)?;
    try_walker_orthodox_descent(ctx, op.pc, &[], &operands, &[], dst, 'r', &DIVMOD_DESCENT)
}

/// #171 ORTHODOX descent of the real `w_list_append` charon body (WIP).
///
/// Instead of hand-rolling the int-storage append IR (the fold below), walk
/// the compiled `w_list_append` jitcode (`list_append_jitcode()`): its
/// strategy `switch` folds to `guard_value(strategy==Integer)` over the
/// concrete receiver, the `is_plain_int1` / `plain_int_w` leaves recurse via
/// `inline_call`, the `ll_list_int_*` leaves are oopspec-lowered to
/// getfield/setfield/setarrayitem, and the capacity `goto_if_not` guards the
/// spare-capacity fast path.
///
/// The sub-walk's guards must resume at the `lst.append` CALL site (re-execute
/// the append generically on deopt — any of strategy / plain-int / capacity
/// failing).  The inline-subwalk capture (`walker_capture_snapshot_for_last_
/// guard_impl` single-frame fallthrough) reads `ctx.{outer_active_boxes,
/// outer_jitcode_index,entry_py_pc}` + the vable shadow directly, so this
/// pre-publishes that ONE call-site coordinate (mapped from `op.pc`) before
/// the sub-walk with inline-subwalk mode enabled.
///
/// Like the fold, the walker only RECORDS the array-op IR; this applies the
/// append to the concrete list + journals the rewind. Recognition declines
/// before emitting IR; an unsupported body sub-walk rolls its tentative IR
/// back before falling through to the residual call.
///
/// STATUS: the descr-pool
/// wiring, the host-static const relocation, and the list header field
/// descr-group bridge (`make_descr_from_bh` strategy/length/items →
/// `W_LIST_DESCR_GROUP`) are all in place — the strategy `switch` and the
/// inlined `is_int`/`is_bool` type predicates fold over the concrete receiver,
/// the `W_ListObject.strategy` read resolves a parent_descr, and the walk
/// descends the full append into the Integer fast-path.  The unit-`()` return
/// aggregate (`SyntheticTransparentCtor "Tuple"`) is elided to `ConstRefNull`
/// at build time (`jtransform.rs`), so the descent completes and commits a
/// working trace.  Safety net: if a stale build-time jitcode kept that ctor as
/// a symbolic (tagged) fnaddr, `try_execute_residual_call_via_executor`
/// declines it (`OrthodoxSubWalkTraceUnsupported`) and the method-call form
/// records the append as a residual call instead of baking the hash as a code
/// address and branching to garbage.
fn list_append_resume_declines(error: &DispatchError) -> bool {
    matches!(
        error,
        DispatchError::OrthodoxSubWalkTraceUnsupported { .. }
            | DispatchError::GuardResumeCoordinateUnavailable { .. }
            | DispatchError::LoopBearingCalleeInlineUnsupported { .. }
            | DispatchError::GuardSnapshotVableUntyped { .. }
    )
}

fn rollback_list_append_attempt<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pre_fold_pos: majit_metainterp::recorder::TracePosition,
    list: pyre_object::PyObjectRef,
    len_before: usize,
    allocated_before: isize,
    promote_journal_before: usize,
) {
    ctx.trace_ctx.cut_trace(pre_fold_pos);
    ctx.trace_ctx.heap_cache_mut().reset();
    // `w_list_uses_empty_storage` at entry is not "this attempt pushed a
    // promotion". `orthodox_list_append_commit` pushes
    // `fbw_append_promote_journal_push` only after the strategy switch, and
    // the callable guards above can decline before that. Popping here used
    // to assert on an empty journal or drop another list's entry.
    let promoted = fbw_append_promote_journal_len() > promote_journal_before;
    while fbw_append_promote_journal_len() > promote_journal_before {
        fbw_append_promote_journal_rollback_newest();
    }
    // The promotion clear restored Empty. The caller's pointer can be the
    // pre-move reference (`w_list_switch_to_strategy_for` may relocate), so
    // it is not a safe target for a length store.
    if !promoted && unsafe { pyre_object::w_list_len(list) } > len_before {
        fbw_rewind_unjournaled_list_append(list, len_before, allocated_before);
    }
    // A declined inlined callee may have latched a blackhole image. This
    // recovery continues the walk, so that image must not answer for a
    // later abort (`abort_blackhole_latched`).
    reset_single_frame_blackhole();
}

/// `Ok(false)` rolls the attempt back to the residual.  A resume coordinate
/// the inlined callee cannot name is that decline; anything else still aborts
/// the walk.
fn list_append_capture_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    pre_fold_pos: majit_metainterp::recorder::TracePosition,
    list: pyre_object::PyObjectRef,
    len_before: usize,
    allocated_before: isize,
    promote_journal_before: usize,
) -> Result<bool, DispatchError> {
    match walker_capture_snapshot_for_last_guard(ctx, op_pc) {
        Ok(()) => Ok(true),
        Err(error) if list_append_resume_declines(&error) => {
            rollback_list_append_attempt(
                ctx,
                pre_fold_pos,
                list,
                len_before,
                allocated_before,
                promote_journal_before,
            );
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

/// Trace position and journal watermarks from the first pass of a list-append
/// residual that yielded its body to `SubWalkDriver`. Replay finishes the
/// concrete append, or cuts back here when the body declines.
struct ListAppendSuspendBookmark {
    pc: usize,
    position: majit_metainterp::recorder::TracePosition,
    promote_journal_before: usize,
    allocated_before: isize,
    len_before: usize,
}

thread_local! {
    static LIST_APPEND_SUSPEND_BOOKMARK: std::cell::RefCell<Vec<ListAppendSuspendBookmark>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn begin_list_append_suspend_bookmark(
    pc: usize,
    position: majit_metainterp::recorder::TracePosition,
    promote_journal_before: usize,
    allocated_before: isize,
    len_before: usize,
) {
    if !subwalk_driver_is_active() {
        return;
    }
    LIST_APPEND_SUSPEND_BOOKMARK.with(|slot| {
        slot.borrow_mut().push(ListAppendSuspendBookmark {
            pc,
            position,
            promote_journal_before,
            allocated_before,
            len_before,
        });
    });
}

fn end_list_append_suspend_bookmark(pc: usize) -> Option<ListAppendSuspendBookmark> {
    LIST_APPEND_SUSPEND_BOOKMARK.with(|slot| {
        let mut bookmarks = slot.borrow_mut();
        if bookmarks.last().is_some_and(|bookmark| bookmark.pc == pc) {
            bookmarks.pop()
        } else {
            None
        }
    })
}

/// Journal and apply the append once the descended body has been recorded.
fn finish_recorded_list_append<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    walk_outcome: DispatchOutcome,
    list: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
    len_before: usize,
    allocated_before: isize,
) -> Result<(), DispatchError> {
    match walk_outcome {
        DispatchOutcome::SubReturn { result } => {
            if finish_inline_callee_return(ctx, result).is_some() {
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op_pc });
            }
        }
        _ => return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op_pc }),
    }
    // The sub-walk records the store. Apply it once when the concrete list
    // has not already grown (`w_list_append` still runs under the lock the
    // wrapper held; the body walk does not).
    fbw_list_journal_push_append(list, len_before, allocated_before);
    if unsafe { pyre_object::w_list_len(list) } == len_before {
        unsafe { pyre_object::w_list_append(list, value) };
    }
    Ok(())
}

/// `Ok(None)` is the first pass. `Ok(Some(true))` finished the concrete
/// append after the body was recorded. `Ok(Some(false))` rolled a declined
/// body back to the first pass's position.
fn take_list_append_replay<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    list: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
) -> Result<Option<bool>, DispatchError> {
    let Some(done) = take_completed_nested_subwalk(ctx, op.pc) else {
        return Ok(None);
    };
    let bookmark = end_list_append_suspend_bookmark(op.pc);
    match done {
        Err(error) if list_append_resume_declines(&error) => {
            if let Some(bookmark) = bookmark {
                rollback_list_append_attempt(
                    ctx,
                    bookmark.position,
                    list,
                    bookmark.len_before,
                    bookmark.allocated_before,
                    bookmark.promote_journal_before,
                );
            }
            Ok(Some(false))
        }
        Err(error) => Err(error),
        Ok(outcome) => {
            let (len_before, allocated_before, list) = match &bookmark {
                Some(bookmark) => {
                    let list = fbw_append_promote_journal_at(bookmark.promote_journal_before)
                        .unwrap_or(list);
                    (bookmark.len_before, bookmark.allocated_before, list)
                }
                None => (
                    unsafe { pyre_object::w_list_len(list) },
                    unsafe { pyre_object::listobject::w_list_allocated(list) },
                    list,
                ),
            };
            finish_recorded_list_append(
                ctx,
                op.pc,
                outcome,
                list,
                value,
                len_before,
                allocated_before,
            )?;
            Ok(Some(true))
        }
    }
}

pub(crate) fn try_walker_orthodox_list_append<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((callable, [value, _])) = plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };

    // Recognition: the callable must be the bound builtin `list.append`; the
    // receiver + value then pass the shared storage/spare-capacity gate.
    let (inner_func, inner_self, len_before) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || inner_self.is_null() {
            return Ok(None);
        }
        let list_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE);
        if pyre_interpreter::lookup_in_type(list_type, "append") != Some(inner_func) {
            return Ok(None);
        }
        let Some(len_before) = orthodox_list_append_recognize(inner_self, value) else {
            return Ok(None);
        };
        (inner_func, inner_self, len_before)
    };

    // Resolve the compiled `w_list_append` body + the full-body sym (the
    // resume-coordinate source) BEFORE emitting any guard — a decline must
    // leave the trace untouched.
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    match take_list_append_replay(ctx, op, inner_self, value)? {
        Some(true) => {
            let none_ref = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', none_ref)?;
            return Ok(Some(()));
        }
        Some(false) => return Ok(None),
        None => {}
    }

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // Sampled before any guard. The commit promotes and may store before a
    // later resume decline; these are the rewind inputs for that decline.
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(inner_self) };
    let promote_journal_before = fbw_append_promote_journal_len();

    // ── tentative commit ──
    let callable_op = r_args[0];
    let value_op = r_args[2];

    // Pin the callable to `list.append`: guard_class METHOD + guard_value on
    // the stable function slot.  `list_append_capture_guard` uses
    // `walker_capture_snapshot_for_last_guard`, so an inlined callee resumes
    // at its Python CALL and a top frame resumes at this op.
    let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
    if !callable_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(callable_op) {
        let type_const = ctx.trace_ctx.const_int(method_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[callable_op, type_const], 0);
        if !list_append_capture_guard(
            ctx,
            op.pc,
            pre_fold_pos,
            inner_self,
            len_before,
            allocated_before,
            promote_journal_before,
        )? {
            return Ok(None);
        }
    }
    ctx.trace_ctx.heap_cache_mut().class_now_known(callable_op);
    let func_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_function_descr(),
    );
    let func_const = ctx.trace_ctx.const_ref(inner_func as i64);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[func_ref, func_const], 0);
    if !list_append_capture_guard(
        ctx,
        op.pc,
        pre_fold_pos,
        inner_self,
        len_before,
        allocated_before,
        promote_journal_before,
    )? {
        return Ok(None);
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(func_ref, func_const);

    // Recover the receiver list OpRef; the sub-walk reads it as ref-arg 0.
    let self_ref = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        callable_op,
        crate::descr::method_w_self_descr(),
    );

    begin_list_append_suspend_bookmark(
        op.pc,
        pre_fold_pos,
        promote_journal_before,
        allocated_before,
        len_before,
    );
    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, self_ref, value_op, inner_self, value, len_before,
    );
    match commit_result {
        Ok(()) => {
            end_list_append_suspend_bookmark(op.pc);
        }
        Err(error @ DispatchError::SubWalkSuspended { .. }) => return Err(error),
        Err(error) if list_append_resume_declines(&error) => {
            end_list_append_suspend_bookmark(op.pc);
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK {}", error.variant_name());
            }
            rollback_list_append_attempt(
                ctx,
                pre_fold_pos,
                inner_self,
                len_before,
                allocated_before,
                promote_journal_before,
            );
            return Ok(None);
        }
        Err(error) => {
            end_list_append_suspend_bookmark(op.pc);
            return Err(error);
        }
    }

    // The `list.append(x)` call's `None` return (the residual's Ref dst).
    let none_ref = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', none_ref)?;
    Ok(Some(()))
}

/// The substituted residual a re-routing arm hands back: the funcbox, a minted
/// MayForce call descr, and the arglist that goes with them (the funcbox
/// first, as the generic path builds it).  The generic residual path records
/// and executes it exactly as it would the call it replaces, so the
/// force/exception guards, the heapcache invalidation and the result
/// writeback all stay where they are.
///
/// Two arms produce one: [`try_walker_specialize_set_add_method`] and
/// [`try_walker_specialize_bare_super_call`].
pub(crate) struct DirectResidualSubst {
    pub(crate) funcptr: OpRef,
    pub(crate) descr: DescrRef,
    pub(crate) allboxes: Vec<OpRef>,
}

/// What [`try_walker_specialize_set_add_method`] recorded.
pub(crate) enum SetAddMethodSpec {
    /// MayForce [`pyre_interpreter::runtime_ops::jit_set_add_method`]. The
    /// generic tail still executes it.
    Subst(DirectResidualSubst),
    /// The traced element is already in an integer-strategy set. Either the
    /// intval, `set_id`, and `content_gen` guards stand in for the insert, or
    /// [`pyre_interpreter::runtime_ops::jit_int_set_add_already_present`]
    /// does, and the generic tail does not run.
    Elided,
}

/// Pin `op` to `expected` unless the trace already folded it to that
/// constant. `GUARD_VALUE` of a constant the optimizer has proved is
/// `InvalidLoop` (`optimize_GUARD_VALUE`). A constant that is some other
/// int is not guarded and not treated as pinned: the caller keeps the
/// contains helper instead of the field guards.
fn pin_int_guard_value<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    op: OpRef,
    expected: i64,
) -> Result<bool, DispatchError> {
    if op.is_constant() {
        let pinned = matches!(
            ctx.trace_ctx.box_value(op),
            Some(majit_ir::Value::Int(n)) if n == expected
        );
        return Ok(pinned);
    }
    let expected_op = ctx.trace_ctx.const_int(expected);
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardValue, &[op, expected_op])?;
    Ok(true)
}

/// `s.add(x)`: record the direct `set_add` residual the SET_ADD accumulator
/// opcode records, in place of the generic `bh_call_fn` dispatch the
/// bound-method spelling otherwise leaves behind.
///
/// `pyopcode.py SET_ADD` is `space.call_method(w_set, 'add', w_value)`, so the
/// two spellings name one operation.  The codewriter lowers the opcode to a
/// `set_add` residual (`bh_set_add_fn`), while the method call reaches the
/// same store through `bh_call_fn`, which re-reads the bound method's
/// function, rejects keywords and rebuilds the argument vector on every
/// iteration.  This arm pins the callable to the `set.add` builtin and the
/// receiver to an exact `set`, then hands `(receiver, value)` to
/// [`pyre_interpreter::runtime_ops::jit_set_add_method`] — the same
/// `set_add_value` store, entered directly.  Measured on a 3M-iteration
/// `s.add(i & 3)` loop, that is the whole difference between the method-call
/// form and the comprehension: 0.220s -> 0.130s against `list.append`'s
/// 0.040s.
///
/// A miss stays a MayForce residual, and deliberately so: `set_add_value`
/// hashes the element, which can run a user `__hash__`.  That is the other
/// half of the gap against `list.append` (`GuardNotForced`, which even the
/// dispatch-free comprehension carries), and this arm does not claim it.
///
/// A hit does not hash.  When the traced value is an exact `int`
/// [`pyre_object::plain_int_already_in_int_set`] already finds in an
/// [`pyre_object::setobject::IntegerSetStrategy`] set, and that set's
/// [`pyre_object::setobject::W_SetObject::len_relaxed`] is 1, the arm guards
/// the unboxed intval, [`pyre_object::setobject::W_SetObject::set_id`], and
/// [`pyre_object::setobject::W_SetObject::content_gen`], then writes `None`.
/// No helper runs.  The single element is the traced int, so a different
/// int is a miss: the intval guard side-exits and the interpreter performs
/// the real add.  A membership change fails the `content_gen` guard the
/// same way.
///
/// A hit in a set that already holds more than one int keeps
/// [`pyre_interpreter::runtime_ops::jit_int_set_add_already_present`] and
/// `GuardTrue`.  Those keys share one trace (`i & 15`); pinning the traced
/// sample side-exits on every other present key.  The helper does not
/// insert.  A fitting `W_LongObject` is stored unboxed too, but it is not a
/// `W_IntObject`, so it takes the helper rather than the intval guard.  A
/// concrete miss stays the MayForce substitution below.
///
/// Recognition declines before emitting IR. It admits an exact `set`
/// (`ob_type == &SET_TYPE`). A subclass instance carries `SET_USER_TYPE`
/// (`typedef.py` `_getusercls`) and does not match the `GuardClass` below.
/// A subclass that overrides `add` is excluded by the `GuardValue` pinning
/// the bound function, and a frozenset receiver by the layout guard, which
/// matters because
/// [`pyre_interpreter::opcode_ops::set_add_value`] itself accepts
/// `is_set_or_frozenset` and would mutate one.  Anything else falls through to
/// the generic residual, which still runs the builtin's receiver and arity
/// checks.
pub(crate) fn try_walker_specialize_set_add_method<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<SetAddMethodSpec>, DispatchError> {
    let Some((callable, [value, _])) = plain_builtin_call_concretes(ctx, code, op, r_args, 1)
    else {
        return Ok(None);
    };

    // Recognition: the callable must be the bound builtin `set.add`, over an
    // exact `set`. `py_type_check(..., &SET_TYPE)` is the layout the
    // `GuardClass` below pins, so a `SET_USER_TYPE` receiver is not
    // substituted. It excludes a frozenset, which `set_add_value` would
    // otherwise mutate.
    let (inner_func, inner_self) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null()
            || !pyre_object::py_type_check(inner_self, &pyre_object::setobject::SET_TYPE)
        {
            return Ok(None);
        }
        let set_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
        if pyre_interpreter::lookup_in_type(set_type, "add") != Some(inner_func) {
            return Ok(None);
        }
        (inner_func, inner_self)
    };
    // Before any IR. A concrete miss keeps today's MayForce substitution;
    // guarding a contains check that just returned 0 would side-exit forever.
    let already = pyre_object::plain_int_already_in_int_set(inner_self, value);

    // ── tentative commit ──
    // Pin the callable to `set.add`: guard_class METHOD + guard_value on the
    // stable function slot, both resuming at the call site so a deopt
    // re-executes the call generically.
    let self_ref = walker_guard_bound_method(ctx, op.pc, r_args[0], inner_func)?;
    let set_type_addr = &pyre_object::setobject::SET_TYPE as *const _ as i64;
    walker_guard_fold_class(ctx, op.pc, self_ref, set_type_addr)?;

    // `is_plain_int1` also accepts a fitting long. `walker_unbox_int` reads
    // `W_IntObject.intval`, so only an exact int is pinned here, and only
    // when it is the set's sole element. Any other present int shares this
    // trace; a GuardValue of the sample fails on the next key.
    if already && unsafe { pyre_object::is_int(value) } {
        let (traced_int, traced_id, traced_gen, sole) = unsafe {
            let n = pyre_object::w_int_get_value(value);
            let set = &*(inner_self as *const pyre_object::setobject::W_SetObject);
            (
                n,
                set.set_id as i64,
                set.content_gen_relaxed() as i64,
                set.len_relaxed() == 1,
            )
        };
        if sole {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let raw = walker_unbox_int(ctx, op.pc, r_args[2], int_type_addr)?;
            if pin_int_guard_value(ctx, op.pc, raw, traced_int)? {
                let id_op = crate::state::opimpl_getfield_gc_i(
                    ctx.trace_ctx,
                    self_ref,
                    crate::descr::set_id_descr(),
                );
                if pin_int_guard_value(ctx, op.pc, id_op, traced_id)? {
                    let gen_op = crate::state::opimpl_getfield_gc_i(
                        ctx.trace_ctx,
                        self_ref,
                        crate::descr::set_content_gen_descr(),
                    );
                    if pin_int_guard_value(ctx, op.pc, gen_op, traced_gen)? {
                        let none_ref = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
                        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', none_ref)?;
                        return Ok(Some(SetAddMethodSpec::Elided));
                    }
                }
            }
        }
    }

    if already {
        let mut effect = majit_metainterp::cannot_raise_effect_info();
        effect.can_collect = false;
        let present = ctx.trace_ctx.call_typed_with_effect(
            OpCode::CallI,
            pyre_interpreter::runtime_ops::jit_int_set_add_already_present as *const (),
            &[self_ref, r_args[2]],
            &[majit_ir::Type::Ref, majit_ir::Type::Ref],
            majit_ir::Type::Int,
            effect,
        );
        ctx.trace_ctx
            .set_opref_concrete(present, majit_ir::Value::Int(1));
        walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardTrue, &[present])?;
        let none_ref = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', none_ref)?;
        return Ok(Some(SetAddMethodSpec::Elided));
    }

    let funcptr = ctx
        .trace_ctx
        .const_int(pyre_interpreter::runtime_ops::jit_set_add_method as *const () as i64);
    Ok(Some(SetAddMethodSpec::Subst(DirectResidualSubst {
        funcptr,
        descr: set_add_method_descr(),
        allboxes: vec![funcptr, self_ref, r_args[2]],
    })))
}

/// The descr the `s.add(x)` substitution installs: `(Ref, Ref) -> Ref`,
/// `MOST_GENERAL`, tagged [`majit_ir::RuntimeHelperKind::SetAddMethod`].
///
/// The EI `bind(..., CallFlavor::MayForce)` gives the SET_ADD residual this one
/// stands in for: `EffectInfo::MOST_GENERAL`, not the analyzer-empty forcing
/// shape.  Hashing the element runs arbitrary Python, so no write set was ever
/// computed for it and an empty one would assert something false
/// (`effect_info_for_call_flavor`).
///
/// `SetAddMethod` on top of it: the call inserts into the live set and returns
/// `None`, so the `Void`-result write proxy in `writes_live_heap` misses it and
/// the helper tag is all that discriminator has left to read.  The generic
/// `bh_call_fn` this stands in for was counted through its own `CallFn` tag;
/// dropping to an untagged descr would take a completed insert out of the
/// executed-effect odometer, which is what a nested abort consults before
/// rewinding.  Built here rather than inline so that invariant is testable.
pub(crate) fn set_add_method_descr() -> DescrRef {
    majit_metainterp::make_call_descr_with_effect(
        &[Type::Ref, Type::Ref],
        Type::Ref,
        majit_ir::EffectInfo {
            runtime_helper: majit_ir::RuntimeHelperKind::SetAddMethod,
            ..default_effect_info()
        },
    )
}

/// Shared recognition for the list-append descent: the receiver must be a
/// list whose storage strategy matches the value's strict type predicate
/// (Integer / Float / Ascii / Object).  Returns the list length before the
/// append (the journal rewind point) on a match, or `None` (decline)
/// otherwise.
/// No IR is emitted.  Capacity is not a gate: `ll_append` records
/// `conditional_call` of `_ll_list_resize_hint_really` (`rlist.py`).
///
/// # Safety
/// `inner_self` / `value` must be live `PyObjectRef`s.
unsafe fn orthodox_list_append_recognize(
    inner_self: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
) -> Option<usize> {
    // `is_plain_int1` accepts an exact `W_IntObject` or a fits-int
    // `W_LongObject`; both route to Integer storage. The commit path pins
    // `guard_class(value, LONG_TYPE)` for a long value (vs `INT_TYPE` for an
    // int) so the descended `w_list_append` body observes the right `ob_type`.
    // The body's `is_plain_int1(value)` / `plain_int_w(value)` then unbox the
    // long through the compiled `_fits_int` / `toint` path; when that path
    // reaches a helper the sub-walk cannot lower it declines
    // (`OrthodoxSubWalkTraceUnsupported`) and rolls back to the generic
    // residual (correctness-safe for any element).
    if !pyre_object::pyobject::is_list(inner_self) {
        return None;
    }
    // Empty-strategy first-append promotion. `w_list_can_append_without_realloc`
    // is false for Empty (no backing block yet), so classify by the value's
    // type using switch_to_correct_strategy's int -> float -> bytes ->
    // ascii -> object order (listobject.py) and let the commit path install
    // the typed storage. Exact bytes still decline: this descent has no
    // BytesBlock store.
    if pyre_object::w_list_uses_empty_storage(inner_self) {
        let int_ok = pyre_object::is_plain_int1(value)
            && !(pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(value));
        // NaNs select Object storage to preserve identity.
        let float_ok = pyre_object::is_float_strategy_item(value);
        // `AsciiListStrategy.is_correct_type`: exact `str` whose `_length`
        // equals `len(_utf8)`.
        let ascii_ok = pyre_object::is_ascii_strategy_item(value);
        // switch_to_correct_strategy routes `is_plain_int1` (exact int or
        // fits-in-word long) -> Integer with no tagged exclusion. Exclude any
        // plain-int / float / bytes / ascii from the object fallback so a
        // tagged-int DECLINES (generic residual) instead of mis-routing to
        // Object and diverging the traced strategy from the concrete one the
        // commit installs.
        let obj_ok = !value.is_null()
            && !pyre_object::is_plain_int1(value)
            && !pyre_object::is_float_strategy_item(value)
            && !pyre_object::is_bytes_strategy_item(value)
            && !ascii_ok;
        if !int_ok && !float_ok && !ascii_ok && !obj_ok {
            return None;
        }
        // Empty length is 0 (the journal rewind point).
        return Some(0);
    }
    // Int-storage specialization: `is_plain_int1` value (exact `W_IntObject`
    // or fits-int `W_LongObject`) stored unboxed. A tagged-immediate value
    // would need a tag-aware unboxed store and no `w_class` pin; decline to
    // the generic residual append instead.
    let int_ok = pyre_object::w_list_uses_int_storage(inner_self)
        && pyre_object::is_plain_int1(value)
        && !(pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(value));
    // Object-storage extension: any non-null `Ref` value stored into the
    // object items block — no unboxing, so the value carries no type
    // precondition.
    let obj_ok = pyre_object::w_list_uses_object_storage(inner_self) && !value.is_null();
    // Match `FloatListStrategy.is_correct_type`; NaNs take the residual path
    // that converts the receiver to Object storage.
    let float_ok = pyre_object::w_list_uses_float_storage(inner_self)
        && pyre_object::is_float_strategy_item(value);
    let ascii_ok = pyre_object::w_list_uses_ascii_storage(inner_self)
        && pyre_object::is_ascii_strategy_item(value);
    if !int_ok && !obj_ok && !float_ok && !ascii_ok {
        return None;
    }
    Some(pyre_object::w_list_len(inner_self))
}

/// Resolve the compiled `w_list_append` body + the full-body snapshot sym
/// (the resume-coordinate source) shared by both list-append fold forms.
/// Returns `None` (decline — no IR emitted yet) when the body jitcode is not
/// compiled or the snapshot sym is absent.  The returned `sym_ptr` is
/// non-null with a set `jitcode` field.  Word size does not enter here: the
/// `d` operands resolve through a descr pool built from the target's own
/// Charon layouts (`jitcode_runtime::build_time_field_offset`), whose array
/// descrs place the items at the first element-aligned offset past the length
/// word — the offset a 4-byte word and an 8-byte item disagree on.
pub(crate) fn orthodox_list_append_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_append_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    // SAFETY: set for the lifetime of the enclosing full-body walk.
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Where the guards of a helper sub-walk resume.
pub(crate) enum HelperEntry {
    /// Root level: the caller-boundary resume at the full-body sym's
    /// coordinate for the call (see `try_walker_inline_builtin_call`).
    Root,
    /// Inside an inlined Python callee: the callee paused at the helper's
    /// CALL, pushed as the helper's entry frame, so a guard in the helper
    /// rebuilds it and re-executes the helper.
    Callee(InlineParentFrame),
    /// Inside another canonical helper body.  That helper has no Python
    /// frame of its own, so a guard here resumes where the enclosing
    /// helper's guards do: the coordinates and paused levels it already
    /// published, re-executing the outermost helper call.
    EnclosingHelper,
}

/// The [`HelperEntry`] for a helper call at `op_pc` of the current walk.
/// An `Err` is a decline: nothing has been recorded.
fn orthodox_helper_nested_entry<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<HelperEntry, InlineCallerFrameDecline> {
    if ctx.fbw_mode.transparent_helper_subwalk {
        return Ok(HelperEntry::EnclosingHelper);
    }
    if !ctx.fbw_mode.inline_subwalk {
        return Ok(HelperEntry::Root);
    }
    compute_inline_helper_call_entry_frame(ctx, op_pc).map(HelperEntry::Callee)
}

/// Post-merge `-live-` after a loop-header preamble.
///
/// `jtransform` emits that triple as the preamble `-live-`, then
/// `jit_merge_point`, then the guard-resume `-live-`.
/// `resume_marker_for_jitcode_pc` answers the preamble.
/// `bhimpl_jit_merge_point` on a bottommost blackhole raises
/// `ContinueRunningNormally` there, before `FOR_ITER` runs, and the
/// interpreter takes the backward edge back into the compiled loop.
/// The following `-live-` is where the item class check inside
/// `list_iter_descr_next` resumes. The walker records that check as
/// `GuardNonnull` plus `GuardClass`; `optimize_GUARD_CLASS` strengthens
/// the pair into `GuardNonnullClass` and keeps the `GuardNonnull`
/// resume. Bounds and invalidation guards stay on the preamble:
/// resuming those after the merge runs the loop body in the blackhole
/// and records extra warmup guard failures.
fn loop_header_guard_resume_marker(
    jitcode: &majit_metainterp::jitcode::JitCode,
    marker: usize,
) -> usize {
    let code = jitcode.code.as_slice();
    let op_live = crate::state::op_live();
    if code.get(marker) != Some(&op_live) {
        return marker;
    }
    let Some(preamble) = crate::jitcode_runtime::decode_op_at(code, marker) else {
        return marker;
    };
    if preamble.opname != "live" {
        return marker;
    }
    let Some(merge) = crate::jitcode_runtime::decode_op_at(code, preamble.next_pc) else {
        return marker;
    };
    if merge.opname != "jit_merge_point" {
        return marker;
    }
    let Some(resume) = crate::jitcode_runtime::decode_op_at(code, merge.next_pc) else {
        return marker;
    };
    if resume.opname == "live" && jitcode.can_decode_live_vars(resume.pc, op_live) {
        resume.pc
    } else {
        marker
    }
}

/// Enter a canonical helper body as a sub-jitcode walk from a walker fold.
///
/// Publishes the call-site resume coordinate the enclosing full-body walk needs
/// to rebuild this frame, mirrors the virtualizable's `last_instr` /
/// `valuestackdepth` when the enclosing frame owns the shadow, swaps in the
/// callee's GLOBAL descr pool for the duration, runs the walk, then restores
/// every field it moved.  A build-time canonical body carries no per-fn descr
/// pool, so its `d` / `j` operands resolve through `all_descr_refs()` /
/// `RawDescrPool::Global` -- not the parent loop's per-fn pool, which
/// mis-resolves the first `residual_call` descr.
///
/// Returns the walk outcome together with the trace position taken immediately
/// before the walk, for a caller that has to reason about which ops the callee
/// contributed.  The position is captured after the resume mirroring above, so
/// it names the callee's first op and not the mirror's.
///
/// `fallback_label` names this site in the empty-twin coordinate note;
/// `call_site_label` names it in the active-box collection.
///
/// `nested_entry` is [`orthodox_helper_nested_entry`]'s answer for `op_pc`:
/// inside an inlined callee or another helper the helper's guards resume at
/// the enclosing coordinate (the entry frame pushed for a callee, plus the
/// sub-walk's outer coordinate/active boxes), not at the full-body sym's --
/// the same model `try_walker_inline_builtin_call` applies.  Resolving the
/// full-body coordinate from a callee `op_pc` restored the wrong frame image after
/// an overflow guard failed in `step` of `a, b = step(a, b)`.
#[allow(clippy::too_many_arguments)]
fn run_orthodox_helper_subwalk<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    nested_entry: HelperEntry,
    fallback_label: &'static str,
    call_site_label: &'static str,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
    float_args: &[OpRef],
) -> Result<(DispatchOutcome, majit_metainterp::recorder::TracePosition), DispatchError> {
    let nested_helper = !matches!(nested_entry, HelperEntry::Root);
    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker, class_guard_marker) =
        if nested_helper {
            (
                ctx.entry_py_pc(),
                0,
                ctx.outer_jitcode_index,
                ctx.outer_resume_marker_jit_pc,
                None,
            )
        } else {
            unsafe {
                let jc = &*sym.jitcode();
                let jc_index = jc.index as u32;
                let marker = jc.payload.resume_marker_for_jitcode_pc(op_pc);
                // Only the item-class guard moves. The shared marker stays the
                // preamble so bounds and invalidation guards still resume there.
                let class_guard_marker = if call_site_label == "list_iter_descr_next_call_site" {
                    marker.and_then(|marker| {
                        let bumped =
                            loop_header_guard_resume_marker(jc.payload.jitcode.as_ref(), marker);
                        (bumped != marker).then_some(bumped)
                    })
                } else {
                    None
                };
                // Forward py twin first (#73 phase-3): equals the containing
                // coordinate plus trivia normalization by construction; the containing
                // lookup survives for the empty-twin class, and the trivia skip below
                // is an identity on the twin path.
                let mut py = jc
                    .payload
                    .forward_py_pc_for_jitcode_pc(op_pc)
                    .unwrap_or_else(|| {
                        crate::py_coord::note_empty_twin_fallback(
                            fallback_label,
                            jc.index,
                            op_pc as i32,
                        );
                        crate::py_coord::containing_py_pc_for_jitcode_pc(
                            &jc.payload.metadata,
                            op_pc,
                        )
                    });
                if jc.payload.code_ptr.is_null() {
                    (
                        py,
                        sym.valuestackdepth() as i64,
                        jc_index,
                        marker,
                        class_guard_marker,
                    )
                } else {
                    let codeobj = &*jc.payload.code_ptr;
                    py = skip_python_trivia_forward(codeobj, py as usize) as u32;
                    // Read the depth off the jitcode-pc-keyed trivia twin, which equals
                    // `depth_at_py_pc()[skip_python_trivia_forward(containing_py_pc_for_jitcode_pc(op_pc))]`
                    // by construction; fall back to the py_pc-keyed static-liveness read
                    // where the twin is unpopulated (skeleton / fixture install).
                    let depth = if jc.payload.depth_trivia_populated() {
                        jc.payload.depth_trivia_for_jitcode_pc(op_pc)
                    } else {
                        crate::liveness::liveness_for(jc.payload.code_ptr)
                            .depth_at_py_pc()
                            .get(py as usize)
                            .copied()
                    };
                    let vsd = match depth {
                        Some(d) => (sym.nlocals() + d as usize) as i64,
                        None => sym.valuestackdepth() as i64,
                    };
                    (py, vsd, jc_index, marker, class_guard_marker)
                }
            }
        };
    let saved_vable = if !nested_helper && sym.owns_virtualizable_shadow() {
        let saved = crate::trace_opcode::save_vable_resume_scalars(ctx.trace_ctx);
        let li = call_site_py_pc as i64 - 1;
        let li_op = ctx.trace_ctx.const_int(li);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            li_op,
            Value::Int(li),
        );
        let vsd_op = ctx.trace_ctx.const_int(vsd_value);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "valuestackdepth",
            vsd_op,
            Value::Int(vsd_value),
        );
        Some(saved)
    } else {
        None
    };
    let (active, class_guard_resume) = if nested_helper {
        (ctx.frame_state.borrow().outer_active_boxes.clone(), None)
    } else {
        let mut collect_at = |carried_word: i32| {
            let vstack_boxes = ctx.frame_state.borrow().vstack_boxes.clone();
            let vstack = ctx.vstack_valid.then_some(vstack_boxes.as_slice());
            collect_outer_active_boxes(
                sym,
                ctx.trace_ctx,
                ctx.registers_i,
                ctx.registers_r,
                ctx.registers_f,
                outer_jitcode_index,
                false,
                carried_word,
                op_pc as i32,
                OuterActiveBoxesEntryTwin::Plain,
                call_site_label,
                vstack,
                &[],
                // Not a branch-guard reconstruction: this is the pre-call site
                // snapshot, so there is no kept operand-stack slot to report as
                // unsourced.
                None,
            )
        };
        let call_site_word = call_site_marker
            .map(|marker| marker as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let active = collect_at(call_site_word);
        // Same call-site entry, different carried live. The class guard's
        // decoder reads the post-merge `-live-`, so its boxes come from
        // that pc and not from the preamble collection above.
        let class_guard_resume = class_guard_marker.map(|marker| {
            let boxes = collect_at(marker as i32);
            (marker, boxes)
        });
        (active, class_guard_resume)
    };

    let saved_entry = ctx.entry_py_pc;
    let saved_marker = ctx.outer_resume_marker_jit_pc;
    let saved_oji = ctx.outer_jitcode_index;
    let saved_active = std::mem::take(&mut ctx.frame_state.borrow_mut().outer_active_boxes);
    let saved_class_resume = ctx
        .frame_state
        .borrow_mut()
        .list_iter_class_guard_resume
        .take();
    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    if !(ctx.fbw_mode.inline_subwalk && matches!(saved_entry, EntryPyPc::Jit(_))) {
        ctx.entry_py_pc = EntryPyPc::Jit(op_pc);
    }
    ctx.outer_resume_marker_jit_pc = call_site_marker;
    ctx.outer_jitcode_index = outer_jitcode_index;
    ctx.frame_state.borrow_mut().outer_active_boxes = active;
    ctx.frame_state.borrow_mut().list_iter_class_guard_resume = if nested_helper {
        saved_class_resume.clone()
    } else {
        class_guard_resume
    };
    ctx.descr_refs = crate::jitcode_runtime::descr_ref_table();
    ctx.raw_descrs = RawDescrPool::Global;
    ctx.sub_jitcode_lookup = &GLOBAL_SUB_JITCODE_LOOKUP_FN;

    let walk_start = ctx.trace_ctx.get_trace_position();
    let saved_fbw_mode = ctx.fbw_mode;
    ctx.fbw_mode.inline_subwalk = true;
    let helper_frame = match nested_entry {
        HelperEntry::Callee(frame) => {
            Some(InlineFrameGuard::enter(ctx.session, 0, false, vec![frame]))
        }
        HelperEntry::Root | HelperEntry::EnclosingHelper => None,
    };
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op_pc,
        sub_body,
        int_args,
        int_arg_concretes,
        ref_args,
        ref_arg_concretes,
        float_args,
    );
    drop(helper_frame);
    ctx.fbw_mode = saved_fbw_mode;
    ctx.entry_py_pc = saved_entry;
    ctx.outer_resume_marker_jit_pc = saved_marker;
    ctx.outer_jitcode_index = saved_oji;
    ctx.frame_state.borrow_mut().outer_active_boxes = saved_active;
    ctx.frame_state.borrow_mut().list_iter_class_guard_resume = saved_class_resume;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;
    if let Some(saved) = saved_vable {
        crate::trace_opcode::restore_vable_resume_scalars(ctx.trace_ctx, saved);
    }

    // `abort/` in a helper body is an un-lowered `OpKind` — the same class
    // as a symbolic residual (`try_execute_residual_call_via_executor` →
    // `OrthodoxSubWalkTraceUnsupported`). Propagating `AbortMarkerReached`
    // kills the enclosing portal/bridge walk; the fold contract
    // (`try_walker_orthodox_list_append` / `_opcode`) is to residualize
    // the helper instead, matching `inline_call.rs` rolling a declined
    // descent back to the ordinary residual.
    let walk_result = match walk_result {
        Err(DispatchError::AbortMarkerReached { pc }) => {
            Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, symbolic: 0 })
        }
        other => other,
    };

    Ok((walk_result?, walk_start))
}

/// Execute a canonical helper reached through a real codewriter
/// `inline_call_*` opcode with the same caller-boundary and descriptor-pool
/// setup as specialization-driven orthodox descent.
///
/// Once `flatten` emits the inline-call directly, the opcode handler owns only
/// the callee body and argument lists; this wrapper recovers the full-body
/// symbol and delegates to the already-proven boundary machinery instead of
/// growing a second, subtly different helper-frame model.
pub(crate) fn run_codewriter_helper_inline_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    sub_body: &SubJitCodeBody,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
    float_args: &[OpRef],
) -> Result<DispatchOutcome, DispatchError> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() || unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    }
    let sym = unsafe { &*sym_ptr };
    let nested_entry = orthodox_helper_nested_entry(ctx, op_pc)
        .map_err(|_| DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc })?;
    run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        sub_body,
        nested_entry,
        "codewriter_helper_inline_commit",
        "codewriter_helper_inline_call_site",
        int_args,
        int_arg_concretes,
        ref_args,
        ref_arg_concretes,
        float_args,
    )
    .map(|(outcome, _)| outcome)
}

/// Commit core of the #171 orthodox list-append fold, shared by the
/// method-call (`try_walker_orthodox_list_append`) and LIST_APPEND-opcode
/// (`try_walker_orthodox_list_append_opcode`) forms.  Stamps the receiver
/// concrete, pins the value's class (Integer/Float storage), publishes the
/// single append-site resume coordinate, descends the real `w_list_append`
/// body as a sub-jitcode walk recording its native array store, then journals
/// + applies the concrete append.  `self_ref` is the receiver list OpRef the
/// caller supplies (the bound method's `w_self` field, or the opcode's list
/// operand); `sym` / `sub_body` are the pre-resolved resume source + callee
/// body.  The caller writes any residual result (the method form's `None`; the
/// opcode form is void).  Records IR unconditionally — a body sub-walk abort
/// propagates as `DispatchError` (graceful interpreter fallback), never a wrong
/// trace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orthodox_list_append_commit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    self_ref: OpRef,
    value_op: OpRef,
    mut inner_self: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
    len_before: usize,
) -> Result<(), DispatchError> {
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(inner_self) };
    // Keep the original Ref box across the helper-frame boundary.  For a
    // virtual W_IntObject/W_FloatObject, its cached payload field is the live
    // SSA box recorded by `trace_box_int`/`trace_box_float`; the descended
    // `plain_int_w`/float unbox therefore forwards that field exactly like
    // `OptVirtualize.optimize_GETFIELD_GC_I/F` in
    // `rpython/jit/metainterp/optimizeopt/virtualize.py`.  Making the Ref's
    // identity observable here would force an otherwise non-escaping virtual.
    // Stamp the receiver concrete (the sub-walk reads it as ref-arg 0; its
    // strategy switch needs the concrete receiver).
    ctx.trace_ctx.set_opref_concrete(
        self_ref,
        majit_ir::Value::Ref(majit_ir::GcRef(inner_self as usize)),
    );

    // Empty-strategy first-append promotion (gated): install typed storage on
    // the receiver BEFORE the value-class pin / storage read below, so those
    // observe the post-promotion strategy. Classify the target strategy from
    // the value with recognize's int -> float -> ascii -> object guards
    // (switch_to_correct_strategy, listobject.py), then emit the
    // transition IR mutating the existing wrapper, promote the concrete list,
    // and journal the rewind to Empty. Exact bytes never reach here.
    use pyre_object::listobject::ListStrategy;
    let promote_empty = unsafe { pyre_object::w_list_uses_empty_storage(inner_self) };
    if promote_empty {
        let target = unsafe {
            let int_ok = pyre_object::is_plain_int1(value)
                && !(pyre_object::tagged_int::CAN_BE_TAGGED
                    && pyre_object::tagged_int::is_tagged_int(value));
            if int_ok {
                ListStrategy::Integer
            } else if pyre_object::is_float_strategy_item(value) {
                ListStrategy::Float
            } else if pyre_object::is_ascii_strategy_item(value) {
                ListStrategy::Ascii
            } else {
                ListStrategy::Object
            }
        };
        // Guard the current (Empty) strategy so a deopt re-enters the empty
        // path (mirror of `guard_list_strategy`: getfield strategy +
        // GuardValue + replace_box).
        walker_guard_fold_list_strategy(ctx, op.pc, self_ref, ListStrategy::Empty as i64)?;
        // Emit the transition IR mutating the existing wrapper (helpers.rs).
        // It stages the same first 0 -> 4 RPython grow as the concrete helper,
        // leaving the append body to record the length/item stores.
        crate::helpers::emit_promote_empty_list_inline(ctx.trace_ctx, self_ref, target);
        // Concrete promotion of the real list, then journal so a non-commit
        // walk rolls back to Empty.
        inner_self = unsafe { pyre_object::w_list_switch_to_strategy_for(inner_self, value) };
        ctx.trace_ctx.set_opref_concrete(
            self_ref,
            majit_ir::Value::Ref(majit_ir::GcRef(inner_self as usize)),
        );
        fbw_append_promote_journal_push(inner_self);
    }

    // Pin the appended value's class so the inlined `is_plain_int1` type
    // predicate folds during the sub-walk: guard_class(value, <TYPE>) +
    // class_now_known, so its `is_int`/`is_long`/`is_bool` typeptr reads fold
    // to the pinned const (the typeptr fold in `getfield_gc_via_heapcache`).
    // The recognition gate already proved `is_plain_int1(value)`; this guard
    // enforces the observed ob_type at runtime.  The value's integer payload
    // stays symbolic — only its class is pinned.
    //
    // Object-storage append stores the value as a
    // plain GC ref with no unboxing, so it carries no type precondition —
    // skip the class pin (the sub-walk's object-storage store path does
    // not read the value's class).
    let is_obj_storage = unsafe { pyre_object::w_list_uses_object_storage(inner_self) };
    if !is_obj_storage {
        // Integer, Float, and Ascii storage pin the value's class so the
        // body's strict type test folds during the sub-walk. The ob_type
        // const is FLOAT_TYPE for float storage, STR_TYPE for ascii storage
        // (`AsciiListStrategy.is_correct_type` is `type(w_obj) is
        // W_UnicodeObject` plus `is_ascii`), and INT_TYPE / LONG_TYPE for int
        // storage depending on whether the value is an exact int or a
        // fits-int `W_LongObject` (both pass `is_plain_int1` -> Integer
        // storage, but carry distinct `ob_type`s the sub-walk's
        // `is_plain_int1` folds on).
        let is_float_storage = unsafe { pyre_object::w_list_uses_float_storage(inner_self) };
        let is_ascii_storage = unsafe { pyre_object::w_list_uses_ascii_storage(inner_self) };
        let value_is_long = unsafe { pyre_object::pyobject::is_long(value) };
        let value_type_addr = if is_float_storage {
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64
        } else if is_ascii_storage {
            &pyre_object::pyobject::STR_TYPE as *const _ as i64
        } else if value_is_long {
            &pyre_object::pyobject::LONG_TYPE as *const _ as i64
        } else {
            &pyre_object::pyobject::INT_TYPE as *const _ as i64
        };
        // The strict predicate (`is_plain_int1` / `is_plain_float_strict`)
        // rejects subclasses by reading `value.w_class` and requiring it null
        // or == `get_instantiate(<type>)`. The ob_type pin above only folds the
        // `is_int`/`is_float` typeptr reads; the w_class compare stays symbolic,
        // so the inlined predicate is non-concrete and the strategy arm's
        // `if <pred>(value)` branch cannot fold — the sub-walk then descends the
        // dead else-leg `switch_to_object_strategy`, whose `ListStrategy::Object`
        // unit-variant ctor is a symbolic fnaddr the descent declines
        // (`OrthodoxSubWalkTraceUnsupported`). Pin w_class to the concrete
        // value's field so the subclass test folds too (the recognition gate
        // already proved the strict predicate).
        walker_guard_fold_value_w_class(ctx, op.pc, value_op, value, value_type_addr)?;
    }

    // Pre-publish the ONE append-site resume coordinate the sub-walk's guards
    // collapse to (mirror the full-body path's last_instr / valuestackdepth
    // publication, keyed to the append op's py_pc — the CALL for the method
    // form, the LIST_APPEND for the opcode form).
    let nested_entry = orthodox_helper_nested_entry(ctx, op.pc)
        .map_err(|_| DispatchError::callee_inline_unsupported(op.pc))?;
    // Nested in an active `SubWalkDriver` (list.__init__ descending
    // `proxy_list_append`), the yield would cut this prefix and restore a
    // heap cache that still names it. The append body reads those boxes.
    keep_residual_recordings_across_suspend::<Sym>();
    let (walk_outcome, _walk_start) = run_orthodox_helper_subwalk(
        ctx,
        op.pc,
        sym,
        sub_body,
        nested_entry,
        "list_append_commit",
        "w_list_append_call_site",
        &[],
        &[],
        &[self_ref, value_op],
        &[ConcreteValue::Ref(inner_self), ConcreteValue::Ref(value)],
        &[],
    )?;

    // Reaching here means the body sub-walk completed without hitting an
    // un-lowered helper: the strategy switch folded over the concrete
    // receiver, the strict type-predicate leaves recursed (`is_plain_int1`
    // for Integer / `is_plain_float_strict` for Float /
    // `is_ascii_strategy_item` for Ascii; Object stores with no type test),
    // the `ll_list_{int,float,obj,ascii}_*` leaves lowered to
    // getfield/setfield/setarrayitem, and the unit-`()` return aggregate
    // (`SyntheticTransparentCtor "Tuple"`) was elided to `ConstRefNull` at
    // build time.  Any residual that does NOT lower —
    // e.g. a stale build-time jitcode whose tuple ctor kept a symbolic
    // symbolic-tagged funcbox — is declined by `try_execute_residual_call_via_executor`
    // (`OrthodoxSubWalkTraceUnsupported`) and the sub-walk helper propagates that
    // abort before this point (graceful interpreter fallback, never a wrong
    // trace).  The descr-pool wiring above (strategy/header field descrs) is
    // exercised on the way in.

    // Tracing is execution: apply the append + journal the rewind.  The
    // journal entry is unconditional — it rewinds the receiver to
    // `len_before` on an aborted walk, whichever side actually grew it.
    //
    // The sub-walk normally records the store as IR without touching the
    // concrete list, so the append below is what applies it.  It is not
    // guaranteed to: the per-strategy store the descended arm reaches
    // (`W_ListObject::object_push`, `IntArray::push`, `FloatArray::push`) is a
    // `residual_call`, and a residual whose funcptr resolves to a real address
    // is EXECUTED by `try_execute_residual_call_via_executor` rather than only
    // recorded.  Those three carry runtime bindings, so on a target where the
    // arm keeps them as residuals the sub-walk has already appended, and
    // appending again puts the value in twice — one extra element per compiled
    // append, which is how it surfaces (`len(keep)` 20048 for 20000
    // iterations, a traceback name list with its last frame doubled).
    // Re-read the length instead of assuming which side ran: it is the
    // receiver's own state, so it answers for both.
    finish_recorded_list_append(
        ctx,
        op.pc,
        walk_outcome,
        inner_self,
        value,
        len_before,
        allocated_before,
    )
}

/// Descend the Integer- or Object-strategy `w_list_pop_end_inner` body for a
/// bound `list.pop()` call, recording its length/item array operations instead
/// of an opaque residual call.
///
/// "Guard-free" elsewhere about this body (`listobject.rs` `w_list_pop_end`,
/// `jitcode_runtime.rs` `list_pop_end_jitcode`) names the *lock* guard: a
/// `w_list_lock` pair inside the body would decline the sub-walk. It is not a
/// claim about trace guards, and the two must not be conflated here, because
/// what the fold's soundness rests on is a trace-guard ordering property:
///
/// The sub-walk gets no callee frame — `outer_*` and `snapshot_sym` below stay
/// the caller's — so a guard recorded inside it resumes at this CALL boundary
/// and re-executes the whole `pop()`. That is only sound while every guard
/// lands *before* the body's first committed store. It does today: the Integer
/// arm's `ll_list_int_set_len` is a native `setfield_gc_i`, and the sole op
/// after it is the `w_int_new` call, which `dispatch_inline_call_dir_kind`
/// short-circuits into `walker_box_int` (`NewWithVtable` + `SetfieldGc`,
/// recording no guard) and returns before `run_sub_jitcode_walk`. Take that
/// short-circuit away and the walk records the boxing body's own null and
/// exception guards after the length is already shrunk, and a failure there
/// pops twice. The Object arm returns the element itself, so no boxing call
/// follows its stores and the hazard cannot arise.  A Float element reaches
/// neither arm: `PoppedValue` carries the Integer and Object shapes only.
/// The pre-fold `GuardClass` / `GuardValue` and the strategy
/// switch's guard are ahead of the store by construction; the body's own ops
/// are checked instead of assumed —
/// [`subwalk_guard_follows_store`] reads the window the sub-walk recorded and
/// declines the fold if a guard landed past the first store.
pub(crate) fn try_walker_orthodox_list_pop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let Some((callable, _)) = plain_builtin_call_concretes(ctx, code, op, r_args, 0) else {
        return Ok(None);
    };

    let (inner_func, inner_self, len_before, popped) = unsafe {
        if !pyre_object::function::is_method(callable) {
            return Ok(None);
        }
        let inner_func = pyre_object::function::w_method_get_func(callable);
        let inner_self = pyre_object::function::w_method_get_self(callable);
        if inner_func.is_null() || inner_self.is_null() {
            return Ok(None);
        }
        let list_type = pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE);
        if pyre_interpreter::lookup_in_type(list_type, "pop") != Some(inner_func) {
            return Ok(None);
        }
        let Some((len_before, popped)) = orthodox_list_pop_recognize(inner_self) else {
            return Ok(None);
        };
        (inner_func, inner_self, len_before, popped)
    };

    // Resolve every possible decline before recording a guard.
    let Some((sub_body, sym_ptr)) = orthodox_list_pop_body_and_sym(ctx) else {
        return Ok(None);
    };
    let sym = unsafe { &*sym_ptr };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let self_ref = walker_guard_bound_method(ctx, op.pc, r_args[0], inner_func)?;

    match orthodox_list_pop_commit(
        ctx, op, sym, &sub_body, self_ref, inner_self, len_before, popped, dst,
    ) {
        Ok(()) => Ok(Some(())),
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-POP-SUBWALK pc={pc}");
            }
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// The popped element, sampled before the descended executor can mutate the
/// live list. The Integer scalar is GC-immune and boxed after the sub-walk;
/// the Object element is the value itself, so the commit pins it instead.
#[derive(Clone, Copy)]
pub(crate) enum PoppedTail {
    Int(i64),
    Object(pyre_object::PyObjectRef),
}

/// Recognize a non-empty Integer- or Object-strategy list and sample its final
/// item before the descended executor can mutate the live list.
///
/// These two are the strategies whose `w_list_pop_end_inner` arm is decomposed
/// into `ll_list_{int,obj}_*` leaves; the rest still pop through a fused helper
/// the sub-walk cannot lower.
unsafe fn orthodox_list_pop_recognize(
    inner_self: pyre_object::PyObjectRef,
) -> Option<(usize, PoppedTail)> {
    if !pyre_object::pyobject::is_list(inner_self) {
        return None;
    }
    let int_storage = pyre_object::w_list_uses_int_storage(inner_self);
    if !int_storage && !pyre_object::w_list_uses_object_storage(inner_self) {
        return None;
    }
    let len_before = pyre_object::w_list_len(inner_self);
    if len_before == 0 {
        return None;
    }
    let list = &*(inner_self as *const pyre_object::listobject::W_ListObject);
    let tail = if int_storage {
        PoppedTail::Int(pyre_object::listobject::ll_list_int_getitem_fast(
            list,
            len_before - 1,
        ))
    } else {
        PoppedTail::Object(pyre_object::listobject::ll_list_obj_getitem_fast(
            list,
            len_before - 1,
        ))
    };
    Some((len_before, tail))
}

/// Resolve the pop body and enclosing full-body snapshot before IR emission.
pub(crate) fn orthodox_list_pop_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_pop_end_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() || unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Whether the ops recorded since `start` put a guard after a store.
///
/// A sub-walk that gets no callee frame resumes its guards at the caller's CALL
/// boundary, so a failure re-executes the whole call. That is sound only while
/// every guard precedes the body's first store: past one, the resumed call
/// re-applies an effect the body already recorded.
///
/// This reads the recorded IR only. A `setfield` here says the walk *recorded*
/// a store, not that one reached the heap — `setfield_gc_via_heapcache` writes
/// through only for boxes the walk itself allocated — so a caller that wants to
/// decline on the answer still owes its own proof that nothing observable has
/// been applied yet.
///
/// A `start` past the end means the trace was cut below the capture point, so
/// the window this answers about is gone; report the guard rather than read an
/// empty one as "sound".
pub(crate) fn subwalk_guard_follows_store(
    trace_ctx: &TraceCtx,
    start: majit_metainterp::recorder::TracePosition,
) -> bool {
    // Byte-mode `_pos` is the opencoder cursor, not an `ops` index.
    // `opcode_at` reads `FrontendSlot` without materializing `Rc<Op>`.
    let start_i = start.tree_loop_op_index(trace_ctx.num_inputargs());
    let end = trace_ctx.num_ops();
    if start_i > end {
        return true;
    }
    let mut stored = false;
    for i in start_i..end {
        let Some(opcode) = trace_ctx.opcode_at(i) else {
            return true;
        };
        if opcode.is_guard() && stored {
            return true;
        }
        stored |= opcode.is_setfield() || opcode.is_setarrayitem() || opcode.is_setinteriorfield();
    }
    false
}

/// Publish the pop call-site resume coordinate, descend the real helper body,
/// write its Ref result, and journal/apply the concrete shrink exactly once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orthodox_list_pop_commit<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    sym: &Sym,
    sub_body: &SubJitCodeBody,
    self_ref: OpRef,
    inner_self: pyre_object::PyObjectRef,
    len_before: usize,
    popped: PoppedTail,
    dst: usize,
) -> Result<(), DispatchError> {
    // The Object sample is a live ref and the sub-walk below allocates, so pin
    // it and read it back out of its slot rather than reusing the local
    // (`pin_root` normalizes the address it publishes; `gc_roots.rs`). The
    // Integer sample is a scalar and needs none of this.
    let _roots = pyre_object::gc_roots::push_roots();
    let popped_slot = match popped {
        PoppedTail::Object(item) => {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(item);
            Some(slot)
        }
        PoppedTail::Int(_) => None,
    };

    ctx.trace_ctx
        .set_opref_concrete(self_ref, Value::Ref(majit_ir::GcRef(inner_self as usize)));

    let nested_entry = orthodox_helper_nested_entry(ctx, op.pc)
        .map_err(|_| DispatchError::callee_inline_unsupported(op.pc))?;
    let (walk_outcome, walk_start) = run_orthodox_helper_subwalk(
        ctx,
        op.pc,
        sym,
        sub_body,
        nested_entry,
        "list_pop_commit",
        "w_list_pop_end_call_site",
        &[],
        &[],
        &[self_ref],
        &[ConcreteValue::Ref(inner_self)],
        &[],
    )?;

    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc }),
    };
    // The Integer arm commits `ll_list_int_set_len` before it boxes, so a guard
    // recorded after that store would resume at this CALL boundary and pop a
    // second time. Today none is: the boxing call is short-circuited into
    // `NewWithVtable` + `SetfieldGc`, and the Object arm boxes nothing. Decline
    // rather than inherit that as an assumption — the caller cuts back to the
    // generic residual, which pops exactly once.
    //
    // Declining is only safe while the receiver is untouched, so it takes the
    // same length re-read the commit below does: on a target whose
    // `ll_list_int_set_len` keeps a runtime binding, the sub-walk executed it
    // for real rather than recording it, and cutting back to a residual that
    // pops again is the very double-pop this fold already had to fix on the
    // append side. In that case the store is a `call`, not a `setfield`, so
    // the ordering read has nothing to say about it either.
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before
        && subwalk_guard_follows_store(ctx.trace_ctx, walk_start)
    {
        // This decline is an ordering verdict on the receiver, not a descent
        // that reached an unlowered helper, so there is no symbolic address to
        // carry.  Zero is unambiguous: a real symbolic hash always carries the
        // `SYMBOLIC_FNADDR_BASE` tag.
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported {
            pc: op.pc,
            symbolic: 0,
        });
    }
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;

    let w_item = match popped {
        PoppedTail::Int(raw_item) => pyre_object::w_int_new(raw_item),
        PoppedTail::Object(_) => pyre_object::gc_roots::shadow_stack_get(
            popped_slot.expect("an Object sample is pinned on entry"),
        ),
    };
    fbw_list_journal_push_pop_end(inner_self, len_before, w_item);
    if unsafe { pyre_object::w_list_len(inner_self) } == len_before {
        unsafe { pyre_object::w_list_pop_end(inner_self) };
    }
    Ok(())
}

/// LIST_APPEND-opcode form of the #171 orthodox list-append fold (comprehension
/// append, e.g. `[f(x) for x in xs]` inlines LIST_APPEND into the enclosing
/// function).  The codewriter lowers LIST_APPEND to a void
/// `jit_list_append(list, value)` residual tagged `ListAppendValue`; here
/// `r_args = [list, value]` (the peeked receiver + the popped value — no
/// bound-method callable).  Recognises the receiver/value against the shared
/// gate and descends the same `w_list_append` body as the method-call form
/// ([`try_walker_orthodox_list_append`]).  Returns `None` (fall through to the
/// generic residual, SAFE — identical to the retired MIFrame tracer's `jit_list_append`)
/// for any non-matching shape, and likewise after rolling the tentative IR back
/// when the body sub-walk hits an un-lowered helper; the residual is void so no
/// result is written.
pub(crate) fn try_walker_orthodox_list_append_opcode<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let _ = dst; // LIST_APPEND residual is void — no result to write.
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(list), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if list.is_null() || value.is_null() {
        return Ok(None);
    }

    // Recognition: no bound-method callable to pin — the list and value are the
    // residual's two Ref operands directly.
    let Some(len_before) = (unsafe { orthodox_list_append_recognize(list, value) }) else {
        return Ok(None);
    };

    // Resolve the compiled body BEFORE emitting any IR — the opcode form emits
    // no guard before the commit, so this is the only decline point.
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    match take_list_append_replay(ctx, op, list, value)? {
        Some(true) => return Ok(Some(())),
        Some(false) => return Ok(None),
        None => {}
    }

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(list) };
    let promote_journal_before = fbw_append_promote_journal_len();

    // ── tentative commit ──
    // The receiver list OpRef + value OpRef are the residual's Ref operands.
    begin_list_append_suspend_bookmark(
        op.pc,
        pre_fold_pos,
        promote_journal_before,
        allocated_before,
        len_before,
    );
    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, r_args[0], r_args[1], list, value, len_before,
    );
    match commit_result {
        Ok(()) => {
            end_list_append_suspend_bookmark(op.pc);
        }
        Err(error @ DispatchError::SubWalkSuspended { .. }) => return Err(error),
        Err(error) if list_append_resume_declines(&error) => {
            end_list_append_suspend_bookmark(op.pc);
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK {}", error.variant_name());
            }
            rollback_list_append_attempt(
                ctx,
                pre_fold_pos,
                list,
                len_before,
                allocated_before,
                promote_journal_before,
            );
            return Ok(None);
        }
        Err(error) => {
            end_list_append_suspend_bookmark(op.pc);
            return Err(error);
        }
    }
    Ok(Some(()))
}

/// `frame_locals_proxy::proxy_list_append` form of the list-append descent.
///
/// The slot scan keeps the lock inside that `dont_look_inside` wrapper and
/// records the call. The wrapper is `w_list_append` of one item, returning
/// the same list. When the funcptr is that leaf, descend
/// `w_list_append_inner` through [`orthodox_list_append_commit`] and write
/// the list OpRef back as the result. A guard inside the helper resumes at
/// the Python `CALL` that entered it (`HelperEntry::EnclosingHelper`), which
/// has not bound the temporary yet, so a declined attempt abandons the
/// partial list. Any other funcptr, or a resume the parent chain cannot
/// name, returns `None` and the wrapper residual runs.
pub(crate) fn try_walker_orthodox_proxy_list_append<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if !residual_call::is_proxy_list_append_residual(ctx, funcptr) {
        return Ok(None);
    }
    if r_args.len() != 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(list), ConcreteValue::Ref(value)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    if list.is_null() || value.is_null() {
        return Ok(None);
    }

    let Some(len_before) = (unsafe { orthodox_list_append_recognize(list, value) }) else {
        return Ok(None);
    };
    let Some((sub_body, sym_ptr)) = orthodox_list_append_body_and_sym(ctx) else {
        return Ok(None);
    };
    // SAFETY: `sym_ptr` is non-null with a set `jitcode` (checked in the
    // resolver) and stays live for the enclosing full-body walk.
    let sym = unsafe { &*sym_ptr };

    match take_list_append_replay(ctx, op, list, value)? {
        Some(true) => {
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', r_args[0])?;
            return Ok(Some(()));
        }
        Some(false) => return Ok(None),
        None => {}
    }

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let allocated_before = unsafe { pyre_object::listobject::w_list_allocated(list) };
    let promote_journal_before = fbw_append_promote_journal_len();

    begin_list_append_suspend_bookmark(
        op.pc,
        pre_fold_pos,
        promote_journal_before,
        allocated_before,
        len_before,
    );
    let commit_result = orthodox_list_append_commit(
        ctx, op, sym, &sub_body, r_args[0], r_args[1], list, value, len_before,
    );
    match commit_result {
        Ok(()) => {
            end_list_append_suspend_bookmark(op.pc);
        }
        Err(error @ DispatchError::SubWalkSuspended { .. }) => return Err(error),
        Err(error) if list_append_resume_declines(&error) => {
            end_list_append_suspend_bookmark(op.pc);
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-APPEND-SUBWALK {}", error.variant_name());
            }
            rollback_list_append_attempt(
                ctx,
                pre_fold_pos,
                list,
                len_before,
                allocated_before,
                promote_journal_before,
            );
            return Ok(None);
        }
        Err(error) => {
            end_list_append_suspend_bookmark(op.pc);
            return Err(error);
        }
    }
    // The wrapper returns the same list it appended to.
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', r_args[0])?;
    Ok(Some(()))
}

fn canonical_kind_of_class(
    cls: pyre_object::PyObjectRef,
) -> Option<pyre_object::interp_exceptions::ExcKind> {
    (0..pyre_object::interp_exceptions::EXC_KIND_COUNT).find_map(|disc| {
        let kind: pyre_object::interp_exceptions::ExcKind =
            unsafe { std::mem::transmute(disc as u8) };
        let candidate = pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind);
        if !candidate.is_null() && std::ptr::eq(candidate, cls) {
            Some(kind)
        } else {
            None
        }
    })
}

fn heap_exc_class_mixes_layouts(cls: pyre_object::PyObjectRef) -> bool {
    let mro = unsafe { pyre_object::typeobject::w_type_get_mro(cls) };
    if mro.is_null() {
        return true;
    }
    let mut seen_extended: Option<bool> = None;
    for &base in unsafe { (*mro).as_slice() } {
        let Some(kind) = canonical_kind_of_class(base) else {
            continue;
        };
        let extended = pyre_object::interp_exceptions::exc_kind_uses_extended_layout(kind);
        match seen_extended {
            None => seen_extended = Some(extended),
            Some(prev) if prev != extended => return true,
            _ => {}
        }
    }
    false
}

/// Walker-native exception-construction fold.  A
/// `Type(args)` `CallFn` residual for a canonical builtin exception class or
/// a heap subclass with the same `__new__` / `__init__` descriptors becomes a
/// traced `NewWithVtable` + `SetfieldGc` (kind / w_class / args_w) the
/// optimizer can virtualize when the exception never escapes, instead of
/// the opaque `bh_call_fn` constructor residual + its
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`.
///
/// The `CallFn` arglist is `r_args = [callable, PY_NULL, args...]`
/// (the `bh_call_fn_N` shape — see `try_walker_specialize_list_append`);
/// the positional args are `r_args[2..]` (the `PY_NULL` self slot is
/// skipped).  Records the fresh `NewWithVtable` OpRef in
/// [`FBW_BUILT_EXC`] so a following `RaiseVarargs` takes the instance
/// fast path; writes the trace-time concrete exception into the dst
/// shadow so the `raise/r` GUARD_CLASS reads it.
///
/// PyPy's `W_TypeObject.descr_call` promotes the class, then resolves
/// `__new__` and `__init__` through its versioned MRO
/// (`typeobject.py`).  When both resolve to
/// `W_BaseException.descr_new` / `descr_init`
/// (`interp_exceptions.py`), a trivial subclass has the same traced
/// allocation and `args_w` store as its builtin base; only `w_class` differs.
///
/// Returns `None` (fall through to the generic residual) for any non-matching
/// shape: an overriding or uncacheable subclass, an unsupported
/// non-trivial-args kind, or a null concrete arg.  OSError's parsed fields and
/// SystemExit's code field are emitted alongside the base exception fields;
/// the remaining non-trivial constructors stay on the runtime path.
pub(crate) fn try_walker_trace_exception_new<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    // Plain `bh_call_fn(callable, PY_NULL, args...)` shape only.
    if r_args.len() < 2 {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (ConcreteValue::Ref(concrete_callable), ConcreteValue::Ref(null_or_self)) =
        (arg_concretes[0], arg_concretes[1])
    else {
        return Ok(None);
    };
    // A non-null `null_or_self` is a bound receiver `bh_call_fn_impl`
    // prepends as arg0 — not a plain `Type(args)` call.
    if concrete_callable.is_null() || !null_or_self.is_null() {
        return Ok(None);
    }

    // Concrete positional args (skip callable + PY_NULL self).  The
    // residual `args_w` list must match the runtime `descr_init` list
    // exactly, so reject any null.
    let args = &r_args[2..];
    let concrete_args: Vec<pyre_object::PyObjectRef> = arg_concretes[2..]
        .iter()
        .map(|c| match c {
            ConcreteValue::Ref(p) => *p,
            _ => std::ptr::null_mut(),
        })
        .collect();

    let is_exc_class = unsafe {
        pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(concrete_callable)
    };
    if !is_exc_class || concrete_args.iter().any(|a| a.is_null()) {
        return Ok(None);
    }

    // OSError can rebind `args_w` after parsing a filename, so its final
    // slice is selected below, once the concrete constructor has exposed the
    // value-dependent branch result.

    let is_canonical = pyre_object::interp_exceptions::is_canonical_exc_class(concrete_callable);
    let mut subclass_lookups = None;
    let subclass_version_tag = if is_canonical {
        None
    } else if heap_exc_class_mixes_layouts(concrete_callable) {
        // `class VS(ValueError, StopIteration)` shares descr_new/descr_init
        // with every `_new_exception` class, but ValueError is slim and
        // StopIteration is extended. Folding it as one kind writes the
        // other layout's extra slots through the instance and corrupts
        // the adjacent type-9 `args_w` items block.
        return Ok(None);
    } else {
        // A heap subclass is safe to construct concretely only after both MRO
        // lookups have been proved identical to a canonical exception class.
        // Consequently force_plain_eval below can execute only the builtin
        // Rust `descr_new` / `descr_init`, never user Python code.  This is the
        // promoted-class lookup contract of typeobject.py.
        if !unsafe { pyre_object::typeobject::w_type_is_heaptype(concrete_callable) } {
            return Ok(None);
        }
        let version_tag =
            unsafe { pyre_object::typeobject::w_type_get_version_tag(concrete_callable) };
        if version_tag == 0 {
            return Ok(None);
        }
        // Both answers are baked under `version_tag`, and an in-place
        // `write_cell` store moves no tag: a `__new__` rebound inside its cell
        // would run user Python where this proof admitted only `descr_new`.
        if unsafe { type_attr_is_cell_backed(concrete_callable, "__new__") }
            || unsafe { type_attr_is_cell_backed(concrete_callable, "__init__") }
        {
            return Ok(None);
        }
        let Some(class_new) = (unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(concrete_callable, "__new__")
        }) else {
            return Ok(None);
        };
        let Some(class_init) = (unsafe {
            pyre_interpreter::baseobjspace::lookup_in_type(concrete_callable, "__init__")
        }) else {
            return Ok(None);
        };
        let matches_canonical = (0..pyre_object::interp_exceptions::EXC_KIND_COUNT).any(|disc| {
            // ExcKind is repr(u8) with contiguous discriminants through
            // EXC_KIND_COUNT, as required by the kind-indexed registry.
            let candidate_kind: pyre_object::interp_exceptions::ExcKind =
                unsafe { std::mem::transmute(disc as u8) };
            let candidate =
                pyre_object::interp_exceptions::lookup_exc_class_for_kind(candidate_kind);
            if candidate.is_null() {
                return false;
            }
            unsafe {
                pyre_interpreter::baseobjspace::lookup_in_type(candidate, "__new__")
                    == Some(class_new)
                    && pyre_interpreter::baseobjspace::lookup_in_type(candidate, "__init__")
                        == Some(class_init)
            }
        });
        if !matches_canonical {
            return Ok(None);
        }
        subclass_lookups = Some((class_new, class_init));
        Some(version_tag)
    };
    // Build the exception concretely on the plain eval loop (no tracer
    // re-entry) to read its kind and confirm a flat builtin instance.
    // Trace-time only; discarded after the read.
    let exc = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_callable, &concrete_args)
    };
    let Ok(exc) = exc else { return Ok(None) };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    let canonical_class = pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind);
    if is_canonical {
        // Preserve the canonical arm's registry identity check.
        if canonical_class != concrete_callable {
            return Ok(None);
        }
    } else {
        // The pre-construction descriptor check excludes Python execution;
        // repeat it for the concrete result's eventual kind so aliases whose
        // builtin wrapper produces a different physical kind still decline.
        let Some((class_new, class_init)) = subclass_lookups else {
            return Ok(None);
        };
        if canonical_class.is_null()
            || unsafe {
                pyre_interpreter::baseobjspace::lookup_in_type(canonical_class, "__new__")
                    != Some(class_new)
                    || pyre_interpreter::baseobjspace::lookup_in_type(canonical_class, "__init__")
                        != Some(class_init)
            }
        {
            return Ok(None);
        }
    }
    let Some(user) = walker_exc_canonical_layout(exc, kind) else {
        return Ok(None);
    };
    let is_os_error_family = matches!(
        kind,
        pyre_object::interp_exceptions::ExcKind::OSError
            | pyre_object::interp_exceptions::ExcKind::FileNotFoundError
    );
    let is_system_exit = kind == pyre_object::interp_exceptions::ExcKind::SystemExit;
    // `W_OSError._parse_init_args` / `_init_error`
    // (`interp_exceptions.py`) fill the flattened slots only for 2..=5
    // arguments.  Outside that range the ordinary args-only emit is exact.
    // Unicode constructors still require their dedicated parsing and remain
    // residual.
    let fills_os_error_slots = is_os_error_family && (2..=5).contains(&args.len());

    // Admit the kind exactly when the concretely built instance left its extra
    // slots defaulted — the slot-content test [`try_walker_trace_raise_bare_class`]
    // already runs, in place of a per-kind tag that rejected a whole kind on
    // faith and so kept `AttributeError(msg)` / `NameError(msg)` /
    // `StopIteration()` on the opaque constructor residual.  A `NULL` slot needs
    // no store; a `None` one takes an explicit `SetfieldGc` below.
    //
    // The bare-class sibling censuses an instance built with NO arguments, so
    // every slot it sees is a trace-time constant.  Here the instance is built
    // from the runtime operands `args`, which nothing pins — only the callable
    // is guarded.  A slot that reads `None` because an ARGUMENT was `None`
    // would therefore be emitted as a constant `None` store while `args_w`
    // keeps the live operand: `StopIteration(x)` traced with `x is None` would
    // answer `e.value is None` for every later `x`.  Each of these
    // constructors fills a slot with either a constant default or one of the
    // passed values, so requiring every argument to be non-`None` makes a
    // `None` slot provably a default.  The check is read only once a defaulted
    // slot is actually found, leaving the all-`NULL` kinds this fold already
    // admitted on their existing path.
    //
    // OSError / SystemExit fill their slots from the arguments, and the emit
    // tail writes them from the argument OpRefs; they skip the census.
    let w_none = pyre_object::w_none();
    let mut w_none_slot_descrs = Vec::new();
    if !is_os_error_family && !is_system_exit {
        let any_none_arg = concrete_args.iter().any(|a| std::ptr::eq(*a, w_none));
        for (offset, value) in
            unsafe { pyre_object::interp_exceptions::w_exception_traced_construction_slots(exc) }
        {
            if value.is_null() {
                continue;
            }
            if !std::ptr::eq(value, w_none) || any_none_arg {
                return Ok(None);
            }
            let Some(descr) = crate::descr::w_exception_slot_descr_for(kind, offset, user) else {
                return Ok(None);
            };
            w_none_slot_descrs.push(descr);
        }
    }

    // `interp_exceptions.py W_SystemExit.descr_init` stores one
    // argument verbatim and several as the tuple selected by `newtuple`.
    // Settle the multi-argument representation before emitting any guards so
    // an unsupported unboxed pair can still decline without leaving trace
    // state behind.
    let system_exit_code = if !is_system_exit || args.is_empty() {
        None
    } else if args.len() == 1 {
        Some((Some(args[0]), None))
    } else {
        let concrete_code = unsafe { pyre_object::interp_exceptions::w_exception_get_code(exc) };
        let code_type = unsafe { (*concrete_code).ob_type };
        if std::ptr::eq(code_type, &pyre_object::TUPLE_TYPE) {
            Some((None, Some((false, concrete_code))))
        } else if std::ptr::eq(
            code_type,
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
        ) {
            Some((None, Some((true, concrete_code))))
        } else {
            return Ok(None);
        }
    };

    let exact_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .is_some_and(|w_os_error| std::ptr::eq(concrete_callable, w_os_error));
    if fills_os_error_slots && exact_os_error {
        // PyPy traces the errno-to-subclass lookup with a loop-variant errno.
        // The flat NewWithVtable emit needs a constant w_class, so pinning the
        // unboxed errno deliberately creates per-errno traces/bridges.
        let errno = concrete_args[0];
        let exact_int = pyre_object::tagged_int::CAN_BE_TAGGED
            && pyre_object::tagged_int::is_tagged_int(errno)
            || unsafe {
                pyre_object::is_plain_int1(errno)
                    && std::ptr::eq(
                        (*errno).ob_type,
                        &pyre_object::pyobject::INT_TYPE as *const _,
                    )
            };
        if !exact_int {
            return Ok(None);
        }
    }

    let concrete_w_class = unsafe { (*exc).w_class };
    let is_blocking_io_error = pyre_interpreter::builtins::lookup_exc_class("BlockingIOError")
        .is_some_and(|blocking| std::ptr::eq(concrete_w_class, blocking));
    // `W_OSError._init_error` gives an exact BlockingIOError's numeric third
    // argument the characters_written meaning.  Keep every three-or-more-arg
    // instance of that concrete class on the complete runtime path.
    if fills_os_error_slots && args.len() >= 3 && is_blocking_io_error {
        return Ok(None);
    }

    // Where the platform reads the fourth argument, `_parse_init_args` derives
    // the errno and the retagged class from it and stores it in its own slot.
    // Both depend on a value this emit neither guards nor writes, so those
    // instances stay on the runtime path.
    if cfg!(windows) && fills_os_error_slots && args.len() >= 4 {
        return Ok(None);
    }

    let has_filename = fills_os_error_slots
        && args.len() >= 3
        && !unsafe { pyre_object::is_none(concrete_args[2]) };
    let final_args_len = if has_filename { 2 } else { args.len() };
    let final_args = &args[..final_args_len];

    // GuardClass pins each None-sensitive `_init_error` branch.  A tagged
    // immediate cannot be consumed by GuardClass; retain the residual path for
    // that uncommon filename shape.
    if fills_os_error_slots {
        for index in [2usize, 4] {
            if index >= args.len() || (index == 4 && args.len() != 5) {
                continue;
            }
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(concrete_args[index])
            {
                return Ok(None);
            }
        }
    }
    // commit to the specialization: emit IR (no further declines)
    // Pin the callable identity so the trace-time kind / vtable stay
    // valid across iterations (`implement_guard_value`).
    let callable_op = r_args[0];
    walker_guard_fold_callable(ctx, op.pc, callable_op, concrete_callable)?;
    if subclass_version_tag.is_some() {
        // Pin the promoted class version that made both MRO descriptor
        // identities constant.  `W_TypeObject.mutated` recursively changes
        // subclass tags (`typeobject.py`), so mutating this class or a
        // base revokes the loop before the folded constructor is reused.
        let class_const = ctx.trace_ctx.const_ref(concrete_callable as i64);
        walker_pin_type_version_tag(ctx, op.pc, class_const)?;
    }

    if fills_os_error_slots && exact_os_error {
        let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        let raw_errno = walker_unbox_int(ctx, op.pc, args[0], int_type_addr)?;
        let errno_value = unsafe { pyre_object::w_int_get_value(concrete_args[0]) };
        walker_guard_stamped_int(ctx, op.pc, raw_errno, errno_value)?;
    }
    if fills_os_error_slots {
        for index in [2usize, 4] {
            if index >= args.len() || (index == 4 && args.len() != 5) {
                continue;
            }
            let arg = args[index];
            let physical_type = unsafe { (*concrete_args[index]).ob_type } as i64;
            walker_guard_stamped_class(ctx, op.pc, arg, physical_type)?;
        }
    }

    // `args_w` is the fixed item array (`ll_fixed_newlist`), the shape
    // `descr_new` stores after `make_sure_not_resized`.
    let args_list = crate::helpers::emit_rlist_inline(ctx.trace_ctx, final_args);

    // `W_OSError.descr_new` can retag exact OSError by errno while retaining
    // the OSError physical kind.  The guarded errno makes the concrete final
    // class a valid constant; dedicated classes and subclasses keep the called
    // class operand as in the ordinary constructor emit.
    let emitted_w_class = if fills_os_error_slots && exact_os_error {
        ctx.trace_ctx.const_ref(concrete_w_class as i64)
    } else {
        callable_op
    };
    let new_op = crate::helpers::emit_exception_new_inline(
        ctx.trace_ctx,
        kind,
        emitted_w_class,
        args_list,
        user,
    );

    // The slots the constructor defaulted to `None`.  `NewWithVtable` leaves
    // them null, which reads as "unset" rather than `None`, so each one the
    // census collected needs its own store.
    let w_none_const = ctx.trace_ctx.const_ref(w_none as i64);
    for descr in w_none_slot_descrs {
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_none_const], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new_op, descr_index, w_none_const);
    }

    if let Some((direct_code, tuple_shape)) = system_exit_code {
        let code = if let Some((specialised_oo, concrete_code)) = tuple_shape {
            let code = if specialised_oo {
                crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, args[0], args[1])
            } else {
                crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, args)
            };
            ctx.trace_ctx.set_opref_concrete(
                code,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_code as usize)),
            );
            code
        } else {
            direct_code.expect("SystemExit code has neither direct nor tuple value")
        };
        let descr = crate::descr::w_exception_attr_slot_descr_for(
            kind,
            pyre_interpreter::baseobjspace::ExceptionAttrSlot::Code,
            user,
        );
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new_op, code], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new_op, descr_index, code);
    }

    if fills_os_error_slots {
        use pyre_interpreter::baseobjspace::ExceptionAttrSlot;
        let mut stores = vec![
            (ExceptionAttrSlot::Errno, args[0]),
            (ExceptionAttrSlot::Strerror, args[1]),
        ];
        if has_filename {
            stores.push((ExceptionAttrSlot::Filename, args[2]));
            // The fourth positional argument is winerror and is ignored on
            // non-Windows builds, matching W_OSError._parse_init_args.
            if args.len() == 5 && !unsafe { pyre_object::is_none(concrete_args[4]) } {
                stores.push((ExceptionAttrSlot::Filename2, args[4]));
            }
        }
        for (slot, value) in stores {
            let descr = crate::descr::w_exception_attr_slot_descr_for(kind, slot, user);
            let descr_index = descr.index();
            ctx.trace_ctx
                .record_op_with_descr(OpCode::SetfieldGc, &[new_op, value], descr);
            ctx.trace_ctx
                .heapcache_setfield_cached(new_op, descr_index, value);
        }
    }

    // Mark the class known so the following `raise/r` skips its
    // redundant GUARD_CLASS (mirrors the retired raise path's
    // `heapcache.class_now_known`).  The vtable on the NewWithVtable
    // already pins the class for the optimizer; this keeps the heapcache
    // model in agreement.
    ctx.trace_ctx.heap_cache_mut().class_now_known(new_op);

    // Record the fresh instance so a following `RaiseVarargs` recovers
    // the concrete and takes the instance fast path; stamp the dst shadow
    // so the `raise/r` GUARD_CLASS reads it.
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    fbw_built_exc_insert(new_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    Ok(Some(()))
}

/// Walker-native RAISE_VARARGS inline-built-exception fast path. The
/// `RaiseVarargs` residual is `normalize_raise_varargs_jit(frame, exc,
/// cause)` — `r_args = [frame, exc, cause]`.  When `exc` was built inline by
/// [`try_walker_trace_exception_new`] (∈ [`FBW_BUILT_EXC`]) and there is
/// no explicit `from` cause (concrete `cause` is `PY_NULL`), skip the
/// residual publish + its `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` and
/// emit `__context__` as a `SetfieldGc` on the (still virtual) exception:
///
///   active = GETFIELD_GC_R(ec, sys_exc_value)
///   SETFIELD_GC(exc, active, w_exception.w_context)
///
/// For a fresh exception `w_context` is null and the self-cycle is
/// impossible, so `attach_raise_cause`'s conditional `w_context = active`
/// reduces to the unconditional store (a null store when no exception is
/// active is a no-op that DCEs).  The normalized result is the same
/// instance for a flat builtin, so the inline-built `exc` OpRef is
/// written straight to the dst that fed the following `raise/r`.
///
/// Returns `None` (fall through to the residual) when `exc` was not
/// inline-built or a `from` cause is present.
/// `BaseException___reduce___impl` packs `(cls, args)` when `dict` is
/// NULL and `(cls, args, dict)` when the pointer is set, empty included.
/// `W_BaseException.descr_reduce` appends `w_dict` only when
/// `space.is_true(self.w_dict)` and has no `@jit` hint. The only
/// `@jit.unroll_safe` in `interp_exceptions.py` is `W_ImportError.descr_init`.
/// Null dict: `GuardIsnull` plus specialised-OO 2-tuple so
/// `exception_reduce` DCEs. Set dict: `GuardNonnull` plus array-backed
/// 3-tuple so a pickle trace records the packing instead of residual
/// `bh_call_fn(__reduce__)` while `WalkFrameState` is borrowed
/// (`walk_frame_state_roots`).
pub(crate) fn try_walker_specialize_exception_reduce<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let (callable_op, self_op, concrete_self, from_bound_method) = match r_args.len() {
        2 => {
            let (ConcreteValue::Ref(callable), ConcreteValue::Ref(second)) =
                (arg_concretes[0], arg_concretes[1])
            else {
                return Ok(None);
            };
            if callable.is_null() {
                return Ok(None);
            }
            if second.is_null() {
                // Bound method, no extra args: `bh_call_fn(method, NULL)`.
                if !unsafe { pyre_object::function::is_method(callable) } {
                    return Ok(None);
                }
                let inner_func = unsafe { pyre_object::function::w_method_get_func(callable) };
                let inner_self = unsafe { pyre_object::function::w_method_get_self(callable) };
                if inner_func.is_null()
                    || inner_self.is_null()
                    || !pyre_interpreter::builtins::is_builtin_base_exception_reduce_function(
                        inner_func,
                    )
                {
                    return Ok(None);
                }
                (r_args[0], r_args[0], inner_self, true)
            } else if pyre_interpreter::builtins::is_builtin_base_exception_reduce_function(
                callable,
            ) && unsafe { pyre_object::is_exception(second) }
            {
                (r_args[0], r_args[1], second, false)
            } else {
                return Ok(None);
            }
        }
        3 => {
            let (
                ConcreteValue::Ref(callable),
                ConcreteValue::Ref(null_or_self),
                ConcreteValue::Ref(self_obj),
            ) = (arg_concretes[0], arg_concretes[1], arg_concretes[2])
            else {
                return Ok(None);
            };
            if callable.is_null() || !null_or_self.is_null() || self_obj.is_null() {
                return Ok(None);
            }
            if !pyre_interpreter::builtins::is_builtin_base_exception_reduce_function(callable) {
                return Ok(None);
            }
            (r_args[0], r_args[2], self_obj, false)
        }
        _ => return Ok(None),
    };
    if !unsafe { pyre_object::is_exception(concrete_self) } {
        return Ok(None);
    }
    let w_dict = unsafe { pyre_object::interp_exceptions::w_exception_peek_dict(concrete_self) };
    let stored_args =
        unsafe { pyre_object::interp_exceptions::w_exception_get_args_storage(concrete_self) };
    if stored_args.is_null() {
        return Ok(None);
    }
    // `descr_reduce` does `space.newtuple(self.args_w)` then
    // `space.newtuple([cls, args])` or `space.newtuple([cls, args, dict])`.
    // `newtuple` specialises arity 2, so settle both representations
    // before any guard: emitting the array-backed shape for an arity the
    // runtime specialises leaves `len(r)` guarding a vtable this trace
    // never builds (`emit_specialised_tuple_oo_inline`). A set `w_dict`
    // is a live nursery object across that `w_tuple_new`.
    let args_len = unsafe { pyre_object::interp_exceptions::rlist_len(stored_args) };
    let mut concrete_items = Vec::with_capacity(args_len);
    for index in 0..args_len {
        let item = unsafe { pyre_object::interp_exceptions::rlist_getitem(stored_args, index) };
        if item.is_null() {
            return Ok(None);
        }
        concrete_items.push(item);
    }
    let _dict_roots = if w_dict.is_null() {
        None
    } else {
        Some(pyre_object::gc_roots::push_roots())
    };
    let dict_slot = if w_dict.is_null() {
        None
    } else {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_dict);
        Some(slot)
    };
    let concrete_args_tuple = pyre_object::w_tuple_new(concrete_items);
    let args_specialised_oo = if args_len == 2 {
        let ob_type = unsafe { (*concrete_args_tuple).ob_type };
        if std::ptr::eq(
            ob_type,
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
        ) {
            true
        } else if std::ptr::eq(ob_type, &pyre_object::TUPLE_TYPE) {
            false
        } else {
            return Ok(None);
        }
    } else {
        false
    };
    let kind = unsafe { pyre_object::w_exception_get_kind(concrete_self) };
    let phys_type = unsafe { (*concrete_self).ob_type as i64 };
    let w_class = unsafe { (*concrete_self).w_class };

    let self_box = if from_bound_method {
        if !callable_op.is_constant() {
            walker_guard_stamped_class(
                ctx,
                op.pc,
                callable_op,
                &pyre_object::function::METHOD_TYPE as *const _ as i64,
            )?;
        }
        crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            callable_op,
            crate::descr::method_w_self_descr(),
        )
    } else {
        self_op
    };

    walker_guard_stamped_class(ctx, op.pc, self_box, phys_type)?;
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(concrete_self) };
    let (_, _, w_class_descr, args_descr) = crate::descr::w_exception_descrs_for(kind, user);
    let dict_descr = crate::descr::w_exception_dict_descr_for(kind, user);
    let dict_ref = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, self_box, dict_descr);
    let dict_guard = if w_dict.is_null() {
        OpCode::GuardIsnull
    } else {
        OpCode::GuardNonnull
    };
    walker_emit_fold_guard_with_snapshot(ctx, op.pc, dict_guard, &[dict_ref])?;

    let cls_ref = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, self_box, w_class_descr);
    let cls_const = walker_guard_stamped_ref(ctx, op.pc, cls_ref, w_class)?;

    let args_list = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, self_box, args_descr);
    let length = crate::state::opimpl_arraylen_gc(
        ctx.trace_ctx,
        args_list,
        crate::state::pyobject_gcarray_descr(),
    );
    walker_guard_stamped_len(ctx, op.pc, length, args_len as i64)?;
    let mut items = Vec::with_capacity(args_len);
    for index in 0..args_len {
        let index_op = ctx.trace_ctx.const_int(index as i64);
        items.push(crate::state::trace_items_block_getitem_value(
            ctx.trace_ctx,
            args_list,
            index_op,
        ));
    }
    let args_tuple = if args_specialised_oo {
        crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, items[0], items[1])
    } else {
        crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &items)
    };
    ctx.trace_ctx.set_opref_concrete(
        args_tuple,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_args_tuple as usize)),
    );
    // `(cls, args)` is always arity 2 and never a plain-int / plain-float
    // pair, so `makespecialisedtuple2` builds `Cls_oo`. A set dict is the
    // third item (`BaseException___reduce___impl`), arity 3, array-backed.
    let (result, concrete_result) = if w_dict.is_null() {
        (
            crate::helpers::emit_specialised_tuple_oo_inline(ctx.trace_ctx, cls_const, args_tuple),
            pyre_object::w_tuple_new(vec![w_class, concrete_args_tuple]),
        )
    } else {
        let w_dict = pyre_object::gc_roots::shadow_stack_get(
            dict_slot.expect("set dict is pinned across w_tuple_new"),
        );
        (
            crate::helpers::emit_object_tuple_inline(
                ctx.trace_ctx,
                &[cls_const, args_tuple, dict_ref],
            ),
            pyre_object::w_tuple_new(vec![w_class, concrete_args_tuple, w_dict]),
        )
    };
    drop(_dict_roots);
    ctx.trace_ctx.set_opref_concrete(
        result,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
    );
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', result)?;
    Ok(Some(()))
}

/// `raise X` without `from` lowers the cause operand to const `PY_NULL`.
/// A live non-null Ref is an explicit cause and stays on the residual,
/// which also writes `__cause__` and `__suppress_context__`.
fn raise_varargs_cause_is_absent<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    cause_op: OpRef,
) -> bool {
    let cause_concrete = read_ref_var_list_concrete(code, op, 1, ctx);
    match cause_concrete.get(2) {
        Some(ConcreteValue::Ref(p)) => p.is_null(),
        Some(ConcreteValue::Null) | None => matches!(
            ctx.trace_ctx.box_value(cause_op),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
        ),
        _ => false,
    }
}

/// `chain_exceptions` skips the write when `space.is_w(w_value, w_context)`.
///
/// `except E as e: raise e` raises the instance already being handled.
/// `normalize_raise_varargs_jit` is `MayForce` because its other arm calls
/// the exception class; this arm never does. Guard `sys_exc_value` against
/// the raised box so a later iteration that chains a different exception
/// side-exits to that residual.
fn try_trace_reraise_of_handled_instance<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    let exc_op = r_args[1];
    if !raise_varargs_cause_is_absent(ctx, code, op, r_args[2]) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let Some(ConcreteValue::Ref(exc)) = arg_concretes.get(1).copied() else {
        return Ok(None);
    };
    if exc.is_null() || unsafe { !pyre_object::is_exception(exc) } {
        return Ok(None);
    }
    // `chain_context` reads `get_sys_exception`. `sys_exc_info` returns the
    // `sys_exc_value` slot whenever that slot is set, which is the field
    // the guard below reads. A null slot whose logical exception lives on
    // a generator stays on the residual.
    let active = pyre_interpreter::eval::get_current_exception();
    if active.is_null() || !std::ptr::eq(active, exc) {
        return Ok(None);
    }
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };
    let active_op = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[ec],
        crate::descr::ec_sys_exc_value_descr(),
    );
    if exc_op != active_op {
        walker_guard_stamped_ptr_eq(ctx, op.pc, exc_op, active_op)?;
    }
    ctx.trace_ctx
        .set_opref_concrete(exc_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', exc_op)?;
    Ok(Some(()))
}

pub(crate) fn try_walker_trace_raise_builtin<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let exc_op = r_args[1];
    // The inline-built marker is this OpRef's first `raise` of an exception
    // `try_walker_trace_exception_new` just allocated. A different OpRef —
    // `except E as e: raise e` — is not that allocation. `chain_exceptions`
    // then skips the write when the raised instance is the one already
    // being handled.
    if !fbw_built_exc_take(exc_op) {
        return try_trace_reraise_of_handled_instance(ctx, code, op, r_args, dst);
    }
    // Explicit `raise X from Y` (concrete non-null cause) keeps the
    // residual: `attach_raise_cause` sets both `__cause__` and
    // `__suppress_context__`, which the inline `__context__` store alone
    // does not reproduce.  Re-insert the marker so the raise still routes
    // through the residual (the marker was consumed above).
    if !raise_varargs_cause_is_absent(ctx, code, op, r_args[2]) {
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    }

    // Recover the concrete exception + kind for the per-kind w_context
    // descr.  Always present (the construct fold stamped the dst shadow).
    let Some(exc) = walker_concrete_ref_object(ctx, exc_op) else {
        // No concrete recovered — re-insert and decline so the residual
        // runs (defensive; should not happen for a construct-fold exc).
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            fbw_built_exc_insert(exc_op);
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    let user = unsafe { pyre_object::interp_exceptions::exc_obj_is_user_layout(exc) };

    // commit: emit the `__context__` chaining, skip the publish
    // active = GETFIELD_GC_R(ec, sys_exc_value).
    //
    // Route the EC through `walker_ensure_execution_context` so the
    // `__context__` read shares the ONE seeded EC OpRef the PUSH_EXC_INFO /
    // POP_EXCEPT exc-info lowering already consumes (`try_walker_lower_exc_
    // info_residual`).  A fresh `GETFIELD_GC_R(frame, execution_context)` here
    // would mint a DISTINCT OpRef from the seeded `input_arg` EC, so the POP
    // `sys_exc_value` store would `possible_aliasing`-mismatch the buffered
    // PUSH store and force it to materialize — keeping the virtual exception
    // escaped and defeating the balanced save/restore dead-store elimination
    // that lets the locally-caught exception DCE.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        fbw_built_exc_insert(exc_op);
        return Ok(None);
    };
    walker_chain_exception_context(ctx, ec, exc_op, exc, kind, user);

    // The normalized publish result is the same flat builtin instance;
    // forward the inline-built exc OpRef (carrying its concrete shadow)
    // to the dst that feeds the following `raise/r`.
    fbw_built_exc_insert(exc_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', exc_op)?;
    Ok(Some(()))
}

/// Walker-native fold for a bare-class `raise Type`
/// (no call parentheses).  Unlike `raise Type()`, a bare class has no
/// preceding `CallFn` construct residual — `normalize_raise_varargs_jit`
/// instantiates the class itself — so no virtualizable `NewWithVtable`
/// exists and `try_walker_trace_raise_builtin` declines it to the residual
/// (a per-iteration heap alloc + may-force).
///
/// `do_raise` instantiates a raised class with no arguments, so a bare
/// `raise ValueError` is `raise ValueError()`.  When the operand is a
/// canonical builtin exception class whose concrete zero-argument instance
/// can be reproduced exactly and with no explicit `from` cause, build it
/// inline using the `try_walker_trace_exception_new` Empty-args shape and chain
/// `__context__` (the `try_walker_trace_raise_builtin` tail), so the whole
/// exception virtualizes and DCEs when it never escapes.  A subclass or an
/// instance carrying any other pointer-slot value declines to the residual.
///
/// Returns `None` (fall through to the generic residual) for any
/// non-matching shape.
pub(crate) fn try_walker_trace_raise_bare_class<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    r_args: &[OpRef],
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let class_op = r_args[1];
    // The residual arg concretes are `[frame, exc, cause]`.  Recover the live
    // exception operand (index 1) from the residual list rather than the opref
    // shadow: a bare class comes straight from `LOAD_GLOBAL`, not the inline
    // construct fold, so its shadow is not stamped.
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let Some(ConcreteValue::Ref(concrete_class)) = arg_concretes.get(1).copied() else {
        return Ok(None);
    };
    // The operand must be a canonical builtin exception CLASS.  An already
    // built instance (`raise ValueError()`) is not in the class registry and
    // is handled by `try_walker_trace_raise_builtin`; a non-exception operand
    // (`raise obj`) also declines here.
    if concrete_class.is_null()
        || !pyre_object::interp_exceptions::is_canonical_exc_class(concrete_class)
    {
        return Ok(None);
    }

    // Explicit `raise X from Y` keeps the residual: `attach_raise_cause` sets
    // `__cause__` and `__suppress_context__`, which the inline `__context__`
    // store alone does not reproduce.  The cause operand is a const `PY_NULL`
    // (or a `Null` / `Ref(null)` shadow) when there is no cause; any concrete
    // non-null Ref is an explicit cause.
    let cause_op = r_args[2];
    let cause_is_null = match arg_concretes.get(2) {
        Some(ConcreteValue::Ref(p)) => p.is_null(),
        Some(ConcreteValue::Null) | None => matches!(
            ctx.trace_ctx.box_value(cause_op),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
        ),
        _ => false,
    };
    if !cause_is_null {
        return Ok(None);
    }

    // Build the exception concretely on the plain eval loop (no tracer
    // re-entry) to read its kind and confirm a flat builtin instance.  A
    // canonical class has the builtin `descr_new` / `descr_init`, so a
    // zero-argument construction runs no user code.  Trace-time only.
    let exc = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_impl_result(concrete_class, &[])
    };
    let Ok(exc) = exc else { return Ok(None) };
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    if pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind) != concrete_class {
        return Ok(None);
    }
    let Some(user) = walker_exc_canonical_layout(exc, kind) else {
        return Ok(None);
    };

    let w_none = pyre_object::w_none();
    let mut w_none_slot_descrs = Vec::new();
    for (offset, value) in
        unsafe { pyre_object::interp_exceptions::w_exception_traced_construction_slots(exc) }
    {
        if value.is_null() {
            continue;
        }
        if !std::ptr::eq(value, w_none) {
            return Ok(None);
        }
        let Some(descr) = crate::descr::w_exception_slot_descr_for(kind, offset, user) else {
            return Ok(None);
        };
        w_none_slot_descrs.push(descr);
    }

    // Resolve the EC while declining is still free.  `walker_ensure_execution_
    // context` returns `None` on a null snapshot sym or a frameless walk, and
    // its recovery records a `GETFIELD_GC_R` that must not land after a guard
    // referencing it — `ensure_execution_context` recovers eagerly at walk
    // entry for that reason.  A decline past the commit below would also leave
    // the construction ops orphaned and the heap-cache shadows describing an
    // object the caller's generic-residual fall-through never built.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // commit: pin the class identity, emit the construction + raise
    // Guard the class operand so the trace-time kind / vtable stay valid
    // across iterations (`implement_guard_value`).
    walker_guard_fold_callable(ctx, op.pc, class_op, concrete_class)?;

    let args_list = crate::helpers::emit_rlist_inline(ctx.trace_ctx, &[]);

    let new_op =
        crate::helpers::emit_exception_new_inline(ctx.trace_ctx, kind, class_op, args_list, user);
    let w_none_const = ctx.trace_ctx.const_ref(w_none as i64);
    for descr in w_none_slot_descrs {
        let descr_index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new_op, w_none_const], descr);
        ctx.trace_ctx
            .heapcache_setfield_cached(new_op, descr_index, w_none_const);
    }
    ctx.trace_ctx.heap_cache_mut().class_now_known(new_op);
    ctx.trace_ctx
        .set_opref_concrete(new_op, majit_ir::Value::Ref(majit_ir::GcRef(exc as usize)));
    walker_chain_exception_context(ctx, ec, new_op, exc, kind, user);

    // Mark the inline-built instance FBW-built so the following `raise/r`
    // records its frame node via the virtual `record_fresh_application_
    // traceback` (an inline PyTraceback `NewWithVtable` + SETFIELDs on the
    // exception) rather than `record_top_level_application_traceback`, whose
    // runtime hook passes the exception to a `CallN` and forces it to
    // materialize — defeating the save/restore DCE that virtualizes a
    // locally-caught raise.  Mirrors `try_walker_trace_raise_builtin`.
    fbw_built_exc_insert(new_op);
    write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', new_op)?;
    Ok(Some(()))
}

/// Walker-native fold for the deterministic immutable-type
/// STORE_ATTR / DELETE_ATTR raise (`int.x = v` / `del str.x` →
/// TypeError from the `object_setattr` / `object_delattr` non-heaptype
/// guard, `typeobject.py:416/437`).
///
/// The generic path records the raise as an opaque
/// `CallMayForceN(bh_store_attr_fn)` + `GuardNotForced` +
/// `GuardException`, whose result box (the exception materialised
/// *inside* the residual by `PyError::to_exc_object`) can never
/// virtualize — every compiled iteration re-allocates the TypeError,
/// its message string, and its args list through the runtime GC hooks.
/// PyPy traces `space.setattr` itself, so the same raise shows up as
/// `new_with_vtable` + `setfield_gc` ops its optimizer removes when the
/// exception never escapes.
///
/// This fold restores that shape for the one attribute-store raise
/// whose outcome is provably iteration-invariant: when
/// `type_immutable_attr_raise_is_stable` holds (constant non-heaptype
/// receiver, canonical `type` metaclass, no metaclass descriptor for
/// `name` — every consulted dict frozen), the raise and its message
/// depend only on trace-time constants.  Pin the receiver with
/// `GuardValue` (when not already constant), run the authentic
/// `setattr_str` / `delattr_str` concretely for the authoritative
/// walk's exception, and emit the [`try_walker_trace_exception_new`]
/// construction (`NewWithVtable` + args-list `SetfieldGc`s, message as
/// a rooted trace constant) in place of the residual + guards.  The
/// raise then routes through the ordinary `SubRaise` path with a
/// virtualizable exception OpRef, and a locally-caught `except` DCEs
/// the whole allocation exactly as the explicit-`raise` fold does.
///
/// Returns `None` (fall through to the generic residual) for any
/// non-matching or unprovable shape.
pub(crate) fn try_walker_trace_immutable_type_attr_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    obj_op: OpRef,
    store_value: Option<OpRef>,
    w_code_ptr: usize,
    name_idx: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj_op) else {
        return Ok(None);
    };
    // STORE_ATTR runs the authentic concrete `setattr_str` below only for
    // its raising exception.  The value plays no role in that raise — the
    // stability predicate proves no data descriptor for `name`, so the
    // terminal raises before consulting the value — so a non-constant store
    // value (the common `int.x = i` loop case) still folds: a `None`
    // concrete value substitutes a placeholder for the authentic run, and
    // the value operand never enters the emitted trace.
    let concrete_value = match store_value {
        Some(value_op) => {
            Some(walker_concrete_ref_object(ctx, value_op).unwrap_or_else(pyre_object::w_none))
        }
        None => None,
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    if !pyre_interpreter::baseobjspace::type_immutable_attr_raise_is_stable(
        concrete_obj,
        &name,
        store_value.is_none(),
    ) {
        return Ok(None);
    }

    // The raise decision also reads the metaclass-MRO descriptor state — the
    // branch-F `lookup_in_type_where(type, name)` walk and the forwarding
    // `type.__setattr__` / `type.__delattr__` — which the receiver
    // `GuardValue` below does not cover.  A `version_tag` guard on the
    // metaclass pins that state (the guard the sibling method/attr folds
    // carry, `typeobject.py promote(self.version_tag())`): mutating `type`'s
    // dict bumps its tag directly, and mutating `object`'s dict bumps it too
    // because `mutated()` propagates down to the `type` subclass — so one
    // guard covers the whole `(type, object)` MRO the walk reads.  Branch C
    // proved the metaclass is the canonical `type`.  A tagless metaclass
    // (`version_tag == 0`) is uncacheable, so decline before emitting guards.
    let metaclass = pyre_object::get_instantiate(&pyre_object::pyobject::TYPE_TYPE);
    let metaclass_version_tag =
        unsafe { pyre_object::typeobject::w_type_get_version_tag(metaclass) };
    if metaclass_version_tag == 0 {
        return Ok(None);
    }

    // Resolve the EC while declining is still free, for the `__context__` tail
    // below.  `walker_ensure_execution_context` returns `None` on a null
    // snapshot sym or a frameless walk, and its recovery records a
    // `GETFIELD_GC_R` that must not land after a guard referencing it
    // (`try_walker_trace_raise_bare_class` resolves it at the same boundary).
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // commit: pin the receiver, run the authentic raise, emit inline
    // The stability predicate makes the raise a pure function of `(obj,
    // name)`; `GuardValue` pins the one live input (`name` is a co_names
    // constant).
    walker_guard_fold_callable(ctx, op.pc, obj_op, concrete_obj)?;
    // Pin the metaclass `version_tag` (see above): a `GETFIELD_GC_I` +
    // `GuardValue` that side-exits on any `type`/`object` dict mutation.
    walker_guard_fold_type_version(ctx, op.pc, metaclass, metaclass_version_tag as i64)?;

    // The authoritative walk's concrete execution — the same call the
    // residual executor would have made, raising before any heap
    // mutation.  Plain eval: the predicate excludes every user-code path.
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        match concrete_value {
            Some(value) => pyre_interpreter::baseobjspace::setattr_str(concrete_obj, &name, value),
            None => pyre_interpreter::baseobjspace::delattr_str(concrete_obj, &name),
        }
    };
    let Err(mut err) = result else {
        // Unreachable under the predicate (a non-heaptype dict rejects
        // every mutation).  Fail loud rather than falling through: the
        // generic residual would re-run the (somehow) committed effect.
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "immutable-type attr raise fold: stable raise unexpectedly succeeded",
        });
    };
    let exc = err.to_exc_object();
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    // The folded raise is exactly the immutable-type TypeError; any other
    // kind means the runtime path diverged from the predicate's model.
    if kind != pyre_object::interp_exceptions::ExcKind::TypeError {
        return Ok(None);
    }
    let Some(user) = walker_exc_canonical_layout(exc, kind) else {
        return Ok(None);
    };
    Ok(Some((
        walker_emit_canonical_message_raise(ctx, ec, &err, exc, kind, user),
        op.next_pc,
    )))
}

/// Walker-native fold for the read-only-data-descriptor STORE_ATTR raise.
///
/// `objspace.py:723-739` and `descroperation.py descr__setattr__` raise  allow-line-citation
/// AttributeError after resolving a descriptor with no `__set__` and a
/// reachable `__delete__`.  The interpreter predicate excludes every shortcut
/// and user-code branch; class-version guards pin the two MRO lookups, while
/// the descriptor type's `w_name` guard pins the rendered message across a
/// `type.__name__` assignment that does not change its version tag.
pub(crate) fn try_walker_trace_readonly_descr_attr_raise<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    obj_op: OpRef,
    value_op: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || w_code_ptr == 0 {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj_op) else {
        return Ok(None);
    };
    let concrete_value =
        walker_concrete_ref_object(ctx, value_op).unwrap_or_else(pyre_object::w_none);
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some(descr) =
        pyre_interpreter::baseobjspace::readonly_descr_attr_raise_is_stable(concrete_obj, &name)
    else {
        return Ok(None);
    };

    let w_type = unsafe { pyre_object::w_instance_get_type(concrete_obj) };
    let Some(descr_type) = (unsafe { pyre_interpreter::typedef::r#type(descr) }) else {
        return Ok(None);
    };
    let descr_type = descr_type.as_ptr();
    let w_type_version_tag = unsafe { pyre_object::w_type_get_version_tag(w_type) };
    let descr_type_version_tag = unsafe { pyre_object::w_type_get_version_tag(descr_type) };
    if w_type_version_tag == 0 || descr_type_version_tag == 0 {
        return Ok(None);
    }
    let descr_type_w_name = unsafe { pyre_object::typeobject::w_type_peek_name_obj(descr_type) };

    // Resolve the execution context while declining is still effect-free.  Its
    // recovery may record an op, which must precede the fold's guard sequence.
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };

    // commit: pin both MRO decisions, run the authentic raise, emit inline
    // GuardClass pins the receiver payload without pinning its identity.
    let physical_type = unsafe { (*concrete_obj).ob_type } as i64;
    let physical_type_const = ctx.trace_ctx.const_int(physical_type);
    walker_emit_fold_guard_with_snapshot(
        ctx,
        op.pc,
        OpCode::GuardClass,
        &[obj_op, physical_type_const],
    )?;
    ctx.trace_ctx.heap_cache_mut().class_now_known(obj_op);

    // `typeobject.py` promotes the version tag before an MRO lookup.
    // Pinning the receiver type covers both the named descriptor resolution
    // and the default-`__setattr__` answer.
    walker_guard_fold_type_version(ctx, op.pc, w_type, w_type_version_tag as i64)?;

    // The descriptor type's tag pins its general `__set__` / `__delete__` MRO
    // answers (`descroperation.py:117-125`).
    let descr_type_const =
        walker_guard_fold_type_version(ctx, op.pc, descr_type, descr_type_version_tag as i64)?;

    // `typeobject.py:1046-1058` rewrites `w_name` without mutating the class
    // dictionary or its version tag.  Pin the raw slot, including its initial
    // null state, because it shadows the type name rendered in the message.
    let descr_type_name_op = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[descr_type_const],
        crate::descr::type_name_obj_descr(),
    );
    walker_guard_fold_callable(ctx, op.pc, descr_type_name_op, descr_type_w_name)?;

    // Pyre stores the Python-visible class separately from the physical class
    // GuardClass reads.  Pin that class after the mandated MRO/name guard
    // sequence; unlike GuardValue on `obj_op`, this still accepts every
    // receiver of the same class and ties `w_type` to the receiver.
    walker_guard_exact_w_class(ctx, op.pc, obj_op, w_type)?;

    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::baseobjspace::setattr_str(concrete_obj, &name, concrete_value)
    };
    let Err(mut err) = result else {
        // The concrete store has already run, so falling through would execute
        // it twice.  The predicate promises the descriptor terminal instead.
        return Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: "read-only descriptor attr raise fold: stable raise unexpectedly succeeded",
        });
    };
    let exc = err.to_exc_object();
    let kind = unsafe {
        if !pyre_object::is_exception(exc) {
            return Ok(None);
        }
        pyre_object::interp_exceptions::w_exception_get_kind(exc)
    };
    if kind != pyre_object::interp_exceptions::ExcKind::AttributeError {
        return Ok(None);
    }
    let Some(user) = walker_exc_canonical_layout(exc, kind) else {
        return Ok(None);
    };
    Ok(Some((
        walker_emit_canonical_message_raise(ctx, ec, &err, exc, kind, user),
        op.next_pc,
    )))
}

/// Lower the PUSH_EXC_INFO / POP_EXCEPT
/// exc-info-stack residuals to GETFIELD_GC_R / SETFIELD_GC on the EC's
/// `sys_exc_value` slot (`ec_sys_exc_value_descr`), and consume pyre's
/// propagation-root clear without recording a runtime call.
/// Recognised by the codewriter-stamped `runtime_helper` tag, NOT a funcptr
/// address (the residual calls the cross-crate `cpu.*_current_exception*_fn`
/// wrappers in `pyre-jit`, which `pyre-jit-trace` cannot name).
///
///   * `CurrentExceptionOrNone` — `current_exception_or_none()` (`[]→Ref`,
///     dst_bank `'r'`): the PUSH_EXC_INFO `prev` save.  It owns a matching
///     store and POP_EXCEPT restore, so it pushes the field onto the
///     saved-prev stack: emit `GETFIELD_GC_R(ec, sys_exc_value)` for the
///     restore to reinstate.  The operand `PUSH_EXC_INFO` pushes is `None`
///     for an empty slot and the field otherwise; the test becomes the
///     nullity guard upstream records (see the read arm).
///   * `GetCurrentException` — `get_current_exception()` (`[]→Ref`,
///     dst_bank `'r'`): the read a catch-covered bare `raise` uses to obtain
///     the exception it re-raises.  Emit `GETFIELD_GC_R(ec, sys_exc_value)`
///     and stamp the live value concrete (the residual executor would have
///     returned it) so a downstream read of the dst sees the right value.
///   * `SetCurrentException` — `set_current_exception(exc)` (`[Ref]→void`,
///     dst_bank `'v'`): the PUSH_EXC_INFO store and the POP_EXCEPT restore.
///     Emit `SETFIELD_GC(ec, exc, sys_exc_value)` and apply the concrete
///     write the authoritative walk's residual executor would have done.  A
///     restore with no matching save tests its operand against `None`
///     (`set_sys_exc_info3`) and stores NULL for it.
///   * `ClearInFlightException` — `set_in_flight_exception(PY_NULL)`
///     (`[]→void`, dst_bank `'v'`): apply the clear to the authoritative
///     recording walk, but emit no IR.  PyPy keeps the propagating exception
///     in the local `OperationError` and PUSH_EXC_INFO transfers it directly
///     to `ExecutionContext.sys_exc_operror` (`pyopcode.py, 836-863`),
///     so there is no equivalent residual clear in its compiled trace.  Pyre's
///     extra TLS carrier only exposes the Rust `PyError`'s GC children while
///     the interpreter unwinds.  The walk's inline traceback construction
///     never publishes that carrier at compiled runtime; leaving its clear as
///     a CallN would therefore execute an unmatched TLS write on every caught
///     exception.
///
/// A balanced save (`GETFIELD`) + store + restore (`SETFIELD`) on the same
/// descr-identity field with no intervening read is dead-store-eliminated,
/// so a non-escaping exception virtualizes and DCEs (no per-raise
/// `CallMallocNursery`).  Declines (`None` → generic residual) when the EC
/// cannot be recovered or the operand shape does not match (SAFE).
///
/// "No intervening read" is a precondition this lowering cannot check, and
/// there is a second way to spell the same word: `ec_sys_exc_value_descr` is a
/// hand-minted `EC_DESCR_GROUP` field, while a translated body that reaches
/// `ExecutionContext::sys_exc_info` reads the slot through the LLBC layout
/// descr (`executioncontext::ExecutionContext.sys_exc_value`).  The heap
/// optimizer keys its field cache on `descr_identity`, which is the `Arc`
/// pointer, so those two Arcs do not alias: a read through the translated
/// descr does NOT force the store pending under this one, the pair is
/// eliminated, and the read answers with the pre-handler slot.  Inlining a
/// builtin whose body reaches `get_sys_exception` therefore returns `None`
/// inside an `except` block (measured: `sys.exception()` counted 1840 of
/// 30000).  Until the two identities are one Arc, an inlined builtin body must
/// not reach that field.
pub(crate) fn try_walker_lower_exc_info_residual<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    code: &[u8],
    op: &DecodedOp,
    runtime_helper: majit_ir::RuntimeHelperKind,
    r_args: &[OpRef],
    dst_bank: char,
    dst: usize,
) -> Result<Option<()>, DispatchError> {
    if runtime_helper == majit_ir::RuntimeHelperKind::ClearInFlightException {
        // The authoritative walk executed record_application_traceback and
        // published its concrete exception in the interpreter-only carrier.
        // Complete that concrete ownership transfer now.  Compiled traceback
        // recording is emitted as GC IR and never publishes the carrier, so
        // there is deliberately no corresponding runtime operation to record.
        if !r_args.is_empty() || dst_bank != 'v' {
            return Ok(None);
        }
        pyre_interpreter::eval::set_in_flight_exception(pyre_object::PY_NULL);
        return Ok(Some(()));
    }

    if matches!(
        runtime_helper,
        majit_ir::RuntimeHelperKind::GetCurrentException
            | majit_ir::RuntimeHelperKind::CurrentExceptionOrNone
    ) {
        // PUSH_EXC_INFO `prev = ec.sys_exc_value`, or a covered bare raise's
        // read of it — `[]→Ref`.
        if !r_args.is_empty() || dst_bank != 'r' {
            return Ok(None);
        }
        // The two Python instructions that lower to these helpers want
        // different things from a bridge seed.  A bare `raise` wants
        // the exception the bridge is resuming with — the compiled loop is free
        // to elide its `sys_exc_value` store (a balanced save/store/restore
        // DCEs), so the live slot is not a source there and only the seed
        // names the exception to re-raise.  `PUSH_EXC_INFO`'s `prev` save wants
        // the field itself, and at a bridge that resumes AT the handler the
        // seed is the exception this opcode is two ops away from publishing:
        // saving it as `prev` makes the matching `POP_EXCEPT` reinstate the
        // exception the handler just finished with.  Read the live slot for
        // that one — `_prepare_pendingfields` (state.rs, its `execute` block)
        // runs every decoded pending write through `bh_setfield_gc_r` at bridge
        // entry, so the slot is current.  A seed this walk stored itself is a
        // view of the field either way, and reusing its OpRef keeps the
        // save/store/restore triple balanced.
        // `GetCurrentException` is emitted only by a catch-covered bare
        // `RAISE_VARARGS 0`; PUSH_EXC_INFO saves through
        // `CurrentExceptionOrNone`.
        let is_covered_bare_raise_read =
            runtime_helper == majit_ir::RuntimeHelperKind::GetCurrentException;
        let seed_answers_this_read =
            ctx.fbw_mode.current_exception_seed_from_walk_store || is_covered_bare_raise_read;
        let (prev, prev_obj) = if let Some(seed) = ctx
            .frame_state
            .borrow()
            .current_exception_seed
            .filter(|_| seed_answers_this_read)
        {
            (
                seed,
                ctx.frame_state.borrow().current_exception_seed_concrete,
            )
        } else {
            let Some(ec) = walker_ensure_execution_context(ctx) else {
                return Ok(None);
            };
            let prev = ctx.trace_ctx.record_op_with_descr(
                OpCode::GetfieldGcR,
                &[ec],
                crate::descr::ec_sys_exc_value_descr(),
            );
            (prev, pyre_interpreter::eval::get_current_exception())
        };
        // Stamp the concrete `prev` so a downstream read sees the value the
        // residual executor would have returned at this resume point.
        ctx.trace_ctx.set_opref_concrete(
            prev,
            majit_ir::Value::Ref(majit_ir::GcRef(prev_obj as usize)),
        );
        // Only PUSH_EXC_INFO owns a matching set + POP_EXCEPT pair.  A covered
        // bare raise reads the same field to obtain the exception it
        // re-raises, but has no following PUSH store.  Treating that read as a
        // save arms the next POP as a PUSH and leaves the bare raise's value on
        // this stack, so a second enclosing POP restores the inner exception.
        // For PUSH_EXC_INFO, save the field (OpRef, concrete) for the matching
        // restore and mark the immediately-following set as this PUSH's slot
        // store.  The codewriter pushes `prev` then `exc` onto the operand
        // stack and POP_EXCEPT pops them, but the walker resolves the popped
        // `prev` operand to the caught exception, not the saved prev; the LIFO
        // stack carries the authoritative value instead.
        if !is_covered_bare_raise_read {
            FBW_EXC_PREV.with(|s| s.borrow_mut().push((prev, prev_obj)));
            FBW_EXC_PENDING_PUSH_SET.with(|c| c.set(true));
            // The value `PUSH_EXC_INFO` pushes is `space.w_None` for an empty
            // slot and the field otherwise.  Trace that test as the nullity
            // guard `if prev_operr is not None` records.  It has to carry the
            // walk's own `-live-` anchor: this opcode is an exception-table
            // target, so its block head is the handler landing, which reads a
            // `last_exception` value no guard failure carries.  With no anchor
            // to carry, leave the test in the helper — decline the fold and
            // let the call stand, which also keeps the pushed operand a value
            // the resume image can name.  The field read above balances the
            // save/store/restore triple either way.
            let is_null = prev_obj.is_null();
            let guard = if is_null {
                OpCode::GuardIsnull
            } else {
                OpCode::GuardNonnull
            };
            if !walker_emit_anchored_fold_guard(ctx, op.pc, guard, &[prev])? {
                return Ok(None);
            }
            let w_prev = if is_null {
                ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
            } else {
                prev
            };
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', w_prev)?;
            return Ok(Some(()));
        }
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', prev)?;
        return Ok(Some(()));
    }

    // `SetCurrentException`: PUSH_EXC_INFO store (stores the caught EXC) or
    // POP_EXCEPT restore (restores the saved prev).  The two are identical at
    // the residual level; `FBW_EXC_PENDING_PUSH_SET` (set by the immediately-
    // preceding PUSH_EXC_INFO prev save) tells them apart.
    if r_args.len() != 1 || dst_bank != 'v' {
        return Ok(None);
    }
    let Some(ec) = walker_ensure_execution_context(ctx) else {
        return Ok(None);
    };
    let is_push_set = FBW_EXC_PENDING_PUSH_SET.with(|c| c.replace(false));
    // POP_EXCEPT restore consumes the prev its matching PUSH_EXC_INFO saved.
    // If unbalanced (no saved prev — e.g. a POP whose PUSH was not lowered),
    // or this is the PUSH's own store, fall back to the operand value.
    let restore = if is_push_set {
        None
    } else {
        FBW_EXC_PREV.with(|s| s.borrow_mut().pop())
    };
    let (mut store_op, mut store_concrete) = match restore {
        // POP_EXCEPT: restore the saved prev, NOT the operand-stack value
        // (which the walker resolves to the just-caught exception).  Restoring
        // the saved prev makes the PUSH store + this restore a balanced no-op,
        // so a locally-caught exception de-escapes and DCEs, and keeps the slot
        // (`sys.exc_info()`) correct after the handler unwinds.
        Some((prev_op, prev_concrete)) => (prev_op, prev_concrete),
        None => {
            let exc_concrete = match read_ref_var_list_concrete(code, op, 1, ctx).first() {
                Some(ConcreteValue::Ref(p)) => *p,
                Some(ConcreteValue::Null) | None => std::ptr::null_mut(),
                _ => return Ok(None),
            };
            if is_push_set {
                (r_args[0], exc_concrete)
            } else {
                walker_restore_exc_info_operand(ctx, op.pc, r_args[0], exc_concrete)?
            }
        }
    };
    // A PUSH_EXC_INFO store publishes the exception being handled, which IS the
    // tracked active exception (`ctx.last_exc_value()`, the walker's mirror of
    // RPython `metainterp.last_exc_box`).  The graph-side codewriter binds the
    // popped `exc_value`'s producer to a `last_exc_value` re-read for exactly
    // this reason (`codewriter.rs` PushExcInfo arm), but that producer is
    // graph-only — the walker reads the operand-stack slot directly on the
    // assumption that runtime register threading already holds the caught
    // exception there.  At a bridge resume into a handler the slot's per-PC
    // resume reconstruction can alias a non-exception constant (e.g. the vable
    // `f_code` scalar when the catch-landing exception slot shares its color),
    // so the published current exception would become a code object.  The
    // reconstruction can also leave the slot NULL (a bare handler entry whose
    // caught-exception slot was filled with a null sentinel), which would
    // publish `set_current_exception(NULL)` and lose the active exception for a
    // following bare `raise` / `sys.exc_info()`.  When the PUSH store's operand
    // resolves to NULL or a non-exception, recover the authoritative exception
    // from the tracked channel, matching the graph-side producer.
    if is_push_set
        && (store_concrete.is_null() || !unsafe { pyre_object::is_exception(store_concrete) })
    {
        if let (Some(tracked_op), ConcreteValue::Ref(tracked_obj)) =
            (ctx.last_exc_value(), ctx.last_exc_value_concrete())
        {
            if !tracked_obj.is_null() && unsafe { pyre_object::is_exception(tracked_obj) } {
                store_op = tracked_op;
                store_concrete = tracked_obj;
            }
        }
    }
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[ec, store_op],
        crate::descr::ec_sys_exc_value_descr(),
    );
    // The walk is authoritative: apply the concrete store the residual
    // executor would have performed, so the live EC tracks the symbolic
    // SETFIELD in lock-step (a following `get_current_exception` /
    // POP_EXCEPT restore reads the right value).  Journal the displaced
    // prior value first: this store mutates the LIVE per-thread EC, so a
    // non-commit walk exit must restore it (the store journal's discipline).
    // Without the undo an exception propagating OUT of an except-handler
    // aborts the walk before its POP_EXCEPT restore, leaking the caught
    // exception into the next frame's `sys_exc_value`.
    fbw_sys_exc_journal_push(pyre_interpreter::eval::get_current_exception());
    pyre_interpreter::eval::set_current_exception(store_concrete);
    ctx.frame_state.borrow_mut().current_exception_seed = Some(store_op);
    ctx.frame_state.borrow_mut().current_exception_seed_concrete = store_concrete;
    ctx.fbw_mode.current_exception_seed_from_walk_store = true;
    Ok(Some(()))
}

/// `POP_EXCEPT` → `PyFrame._restore_exc_info` →
/// `ExecutionContext.set_sys_exc_info3(w_prev)` for a restore whose value
/// comes off the operand stack: `space.is_none(w_prev)` clears the slot, and
/// any other value becomes the handled exception.  Returns the value to store
/// and its concrete.
fn walker_restore_exc_info_operand<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    w_prev: OpRef,
    w_prev_concrete: pyre_object::PyObjectRef,
) -> Result<(OpRef, pyre_object::PyObjectRef), DispatchError> {
    let is_none = !w_prev_concrete.is_null() && unsafe { pyre_object::is_none(w_prev_concrete) };
    if !w_prev.is_constant() {
        let none_const = ctx.trace_ctx.const_ref(pyre_object::w_none() as i64);
        let is_w = ctx
            .trace_ctx
            .record_op(OpCode::PtrEq, &[w_prev, none_const]);
        ctx.trace_ctx
            .set_opref_concrete(is_w, majit_ir::Value::Int(is_none as i64));
        let guard = if is_none {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, guard, &[is_w])?;
    }
    Ok(if is_none {
        (ctx.trace_ctx.const_ref(0), pyre_object::PY_NULL)
    } else {
        (w_prev, w_prev_concrete)
    })
}

/// #62: walker-native speculative specialization for the `STORE_SUBSCR`
/// helper residual_call (oopspec `StoreSubscr`, void result).  Records the
/// Resolve the compiled `w_list_setitem_inner` body + the full-body snapshot
/// sym.  `None` when the helper is absent from this build (hand fold stays).
fn orthodox_list_setitem_body_and_sym<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<(SubJitCodeBody, *const Sym)> {
    let jc_arc = crate::jitcode_runtime::list_setitem_jitcode()?;
    let sub_body = sub_jitcode_body_by_index(jc_arc.index())?;
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return None;
    }
    Some((sub_body, sym_ptr))
}

/// Descend `w_list_setitem_inner` the way `orthodox_list_append_commit`
/// descends `w_list_append_inner`: pin class/strategy, unbox the index,
/// walk the lock-free body, journal the displaced element.
///
/// Returns `Ok(None)` when the body is missing or the walk hits an unlowered
/// helper. The generic residual then serves the site.
#[allow(clippy::too_many_arguments)]
fn try_walker_orthodox_list_setitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    list_op: OpRef,
    key_op: OpRef,
    value_op: OpRef,
    list_obj: pyre_object::PyObjectRef,
    key_obj: pyre_object::PyObjectRef,
    value_obj: pyre_object::PyObjectRef,
    sid: i64,
    index: i64,
) -> Result<Option<()>, DispatchError> {
    let Some((sub_body, sym_ptr)) = orthodox_list_setitem_body_and_sym(ctx) else {
        return Ok(None);
    };
    let Some(displaced) = (unsafe { pyre_object::w_list_getitem(list_obj, index) }) else {
        return Ok(None);
    };
    // Typed getitem boxes the displaced int/float and may move the operands.
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };
    // Root the original element before the sub-walk executes the store
    // (`w_list_setitem_inner`).  A post-walk getitem would read the new
    // value and rollback would restore that, leaving the list mutated.
    fbw_store_journal_root(list_obj, key_obj, displaced);
    let sym = unsafe { &*sym_ptr };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();

    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    walker_guard_fold_class(ctx, op_pc, list_op, list_type_addr)?;
    walker_guard_exact_w_class(
        ctx,
        op_pc,
        list_op,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE),
    )?;
    walker_guard_fold_list_strategy(ctx, op_pc, list_op, sid)?;

    let (idx_type, idx_descr) = crate::state::int_or_bool_unbox_type_descr(key_obj);
    let raw_index = walker_unbox_int_typed(ctx, op_pc, key_op, idx_type, idx_descr)?;
    ctx.trace_ctx
        .set_opref_concrete(raw_index, majit_ir::Value::Int(index));

    if sid != 0 {
        let is_float_storage = sid == 2;
        let value_is_long = unsafe { pyre_object::pyobject::is_long(value_obj) };
        let value_type_addr = if is_float_storage {
            &pyre_object::pyobject::FLOAT_TYPE as *const _ as i64
        } else if value_is_long {
            &pyre_object::pyobject::LONG_TYPE as *const _ as i64
        } else {
            &pyre_object::pyobject::INT_TYPE as *const _ as i64
        };
        walker_guard_fold_value_w_class(ctx, op_pc, value_op, value_obj, value_type_addr)?;
    }

    ctx.trace_ctx.set_opref_concrete(
        list_op,
        majit_ir::Value::Ref(majit_ir::GcRef(list_obj as usize)),
    );
    ctx.trace_ctx.set_opref_concrete(
        value_op,
        majit_ir::Value::Ref(majit_ir::GcRef(value_obj as usize)),
    );

    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    };
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "list_setitem_commit",
        "w_list_setitem_call_site",
        &[raw_index],
        &[ConcreteValue::Int(index)],
        &[list_op, value_op],
        &[ConcreteValue::Ref(list_obj), ConcreteValue::Ref(value_obj)],
        &[],
    );
    let walk_outcome = match walk {
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] LIST-SETITEM-SUBWALK pc={pc}");
            }
            fbw_store_journal_pop();
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Ok((outcome, _)) => outcome,
        Err(error) => {
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[decline-why] LIST-SETITEM-SUBWALK-ERR pc={op_pc} error={}",
                    error.variant_name()
                );
            }
            fbw_bump_executed_effect("store_journal");
            return Err(error);
        }
    };
    match walk_outcome {
        DispatchOutcome::SubReturn { result } => {
            // `w_list_setitem_inner` returns bool; the opcode discards it.
            let _ = result;
        }
        _ => {
            fbw_store_journal_pop();
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
    }

    let (Some(list_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        fbw_store_journal_pop();
        ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    };
    fbw_bump_executed_effect("store_journal");
    // Overwrite is idempotent: a residual the sub-walk already executed
    // wrote this same value.
    let stored = unsafe { pyre_object::w_list_setitem(list_obj, index, value_obj) };
    debug_assert!(stored, "orthodox list setitem: in-bounds store failed");
    Ok(Some(()))
}

/// STORE_SUBSCR on an exact list with a non-negative in-bounds int index.
/// Object storage accepts any value; int storage requires an exact non-bool
/// int; float storage requires a float-strategy item.
///
/// Walks `w_list_setitem_inner` (the lock-free body, same split as
/// `w_list_append_inner`). A missing body or an unfinished walk returns
/// `Ok(None)` so the generic `CALL_MAY_FORCE` residual serves the site.
/// Negative indices, list subclasses, and strategy mismatches stay on that
/// residual.
pub(crate) fn try_walker_orthodox_store_subscr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 3 {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let value_op = r_args[2];
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };

    // Gate: list[int] = value, non-negative index in bounds. Object storage
    // accepts every reference; the unboxed strategies additionally require a
    // matching value type (int storage ← W_IntObject, float storage ←
    // W_FloatObject). This is jtransform.py `do_resizable_list_setitem`'s
    // kind=`r` arm, not a separate object-list shortcut.
    let (sid, index) = unsafe {
        // A bool index is fine: bool shares int's `intval`, unboxed below via
        // its own &BOOL_TYPE guard.  A bool *value* into int storage must still
        // route through the generic path — PyPy's IntegerListStrategy rejects a
        // W_BoolObject (`is_correct_type` is exact-type), switching the list to
        // object storage, so the int-storage fast path would drop the bool type.
        // Float subclasses and NaNs switch the list to Object storage.
        // EXACT list only: a list SUBCLASS instance shares `ob_type ==
        // &LIST_TYPE` but retags `w_class` and may override `__setitem__`;
        // `is_exact_list` excludes it so it falls to the generic residual
        // (which honours the override) instead of this direct-storage store.
        if !pyre_object::is_exact_list(list_obj) || !pyre_object::is_int(key_obj) {
            return Ok(None);
        }
        let index = pyre_object::w_int_get_value(key_obj);
        if index < 0 {
            return Ok(None);
        }
        let concrete_len = pyre_object::w_list_len(list_obj);
        if index as usize >= concrete_len {
            return Ok(None);
        }
        // Object storage keeps the value boxed, so a subclass survives it; the
        // unboxed strategies write the raw payload and would drop the subclass
        // identity the read-back must return.  `is_int`/`is_float` read
        // `ob_type`, which a subclass shares, so exactness is a separate check.
        let sid = if pyre_object::w_list_uses_object_storage(list_obj) {
            0i64
        } else if !pyre_object::is_exact_builtin_instance(value_obj) {
            return Ok(None);
        } else if pyre_object::w_list_uses_int_storage(list_obj)
            && pyre_object::is_int(value_obj)
            && !pyre_object::is_bool(value_obj)
        {
            1i64
        } else if pyre_object::w_list_uses_float_storage(list_obj)
            && pyre_object::is_float_strategy_item(value_obj)
        {
            // The subclass term of that predicate is enforced on replay by
            // pinning `w_class` below.  `is_plain_float_strict` also admits the
            // null spelling of "exact float", which no pin can express, so
            // decline such an operand here rather than emit a guard it would
            // fail itself (see `walker_guard_exact_w_class`).
            if walker_exact_builtin_class(value_obj).is_none() {
                return Ok(None);
            }
            2i64
        } else {
            return Ok(None);
        };
        (sid, index)
    };

    try_walker_orthodox_list_setitem(
        ctx, op_pc, list_op, key_op, value_op, list_obj, key_obj, value_obj, sid, index,
    )
}

/// Walker-native `GetIter` for an exact machine-word `range`.
///
/// Emits the virtual `W_IntRangeIterator` allocation shape directly — the
/// iterator PyPy's inlined `descr_iter` would trace — so a locally consumed
/// iterator stays a removable virtual `New`.
pub(crate) fn try_walker_specialize_get_iter<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor
        || dst_bank != 'r'
        || r_args.len() != 1
        || ctx.fbw_mode.inline_subwalk
    {
        return Ok(None);
    }

    let range_op = r_args[0];
    let Some(range_obj) = walker_concrete_ref_object(ctx, range_op) else {
        return Ok(None);
    };

    // `W_Zip.iter_w` is identity; exact-class guards preserve overrides.
    let zip_type = &pyre_object::functional::ZIP_TYPE as *const pyre_object::PyType;
    let zip_class = pyre_object::get_instantiate(&pyre_object::functional::ZIP_TYPE);
    if unsafe {
        !range_obj.is_null()
            && std::ptr::eq((*range_obj).ob_type, zip_type)
            && std::ptr::eq((*range_obj).w_class, zip_class)
    } {
        walker_guard_exact_instance(ctx, op_pc, range_op, zip_type as i64, zip_class)?;
        ctx.frame_state.borrow_mut().vstack_last_ref = range_op;
        return Ok(Some(range_op));
    }

    let (
        concrete_start,
        concrete_step,
        concrete_length,
        concrete_mul,
        concrete_one_past,
        concrete_promote_step,
    ) = unsafe {
        if !pyre_object::functional::is_w_range(range_obj)
            || !pyre_object::functional::is_exact_w_range(range_obj)
        {
            return Ok(None);
        }
        let (start_obj, _stop_obj, step_obj) = pyre_object::functional::w_range_fields(range_obj);
        let length_obj = pyre_object::functional::w_range_length(range_obj);
        if !pyre_object::is_int(start_obj)
            || pyre_object::is_bool(start_obj)
            || !pyre_object::is_int(step_obj)
            || pyre_object::is_bool(step_obj)
            || !pyre_object::is_int(length_obj)
            || pyre_object::is_bool(length_obj)
        {
            return Ok(None);
        }
        let Some((start, _stop, step)) = pyre_object::functional::w_range_fields_i64(range_obj)
        else {
            return Ok(None);
        };
        let Some(length) = pyre_object::functional::w_range_length_i64(range_obj) else {
            return Ok(None);
        };
        let one_past_i128 = start as i128 + length as i128 * step as i128;
        let Ok(one_past) = i64::try_from(one_past_i128) else {
            return Ok(None);
        };
        let Some(mul) = length.checked_mul(step) else {
            return Ok(None);
        };
        let Some(one_past_checked) = start.checked_add(mul) else {
            return Ok(None);
        };
        debug_assert_eq!(one_past_checked, one_past);
        (
            start,
            step,
            length,
            mul,
            one_past,
            pyre_object::functional::w_range_promote_step(range_obj),
        )
    };

    let range_type_addr = &pyre_object::functional::RANGE_TYPE as *const _ as i64;
    walker_guard_fold_class(ctx, op_pc, range_op, range_type_addr)?;

    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;

    let start_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_start_descr(),
    );
    walker_guard_fold_class_if_unknown(ctx, op_pc, start_r, int_type_addr)?;
    let start_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        start_r,
        crate::descr::int_intval_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(start_i, majit_ir::Value::Int(concrete_start));

    let step_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_step_descr(),
    );
    walker_guard_fold_class_if_unknown(ctx, op_pc, step_r, int_type_addr)?;
    let step_i =
        crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, step_r, crate::descr::int_intval_descr());
    ctx.trace_ctx
        .set_opref_concrete(step_i, majit_ir::Value::Int(concrete_step));

    let length_r = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_length_descr(),
    );
    walker_guard_fold_class_if_unknown(ctx, op_pc, length_r, int_type_addr)?;
    let length_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        length_r,
        crate::descr::int_intval_descr(),
    );
    ctx.trace_ctx
        .set_opref_concrete(length_i, majit_ir::Value::Int(concrete_length));

    let mul = ctx
        .trace_ctx
        .record_op(OpCode::IntMulOvf, &[length_i, step_i]);
    ctx.trace_ctx
        .set_opref_concrete(mul, majit_ir::Value::Int(concrete_mul));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;

    let one_past = ctx.trace_ctx.record_op(OpCode::IntAddOvf, &[start_i, mul]);
    ctx.trace_ctx
        .set_opref_concrete(one_past, majit_ir::Value::Int(concrete_one_past));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardNoOverflow, &[])?;

    // `descr_iter` reads `promote_step` to choose the iterator shape.  The
    // field never changes after construction, so this read and its guard lift
    // out of the loop, and fold away entirely when the range is a virtual
    // `range(...)` allocation of this same trace.
    let promote_step_i = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        range_op,
        crate::descr::range_promote_step_descr(),
    );
    ctx.trace_ctx.set_opref_concrete(
        promote_step_i,
        majit_ir::Value::Int(concrete_promote_step as i64),
    );
    let promoted = ctx
        .trace_ctx
        .record_op(OpCode::IntIsTrue, &[promote_step_i]);
    ctx.trace_ctx
        .set_opref_concrete(promoted, majit_ir::Value::Int(concrete_promote_step as i64));
    let promote_guard = if concrete_promote_step {
        OpCode::GuardTrue
    } else {
        OpCode::GuardFalse
    };
    walker_emit_guard_with_snapshot(ctx, op_pc, promote_guard, &[promoted])?;

    // A promoted step means `descr_new` saw no step argument, so the walk is
    // `start, start+1, ... start+length`.  The one-argument shape additionally
    // needs `start == 0`, which the guard below pins for the trace.
    let one_arg = concrete_promote_step && concrete_start == 0;
    if concrete_promote_step {
        let zero = ctx.trace_ctx.const_int(0);
        let starts_at_zero = ctx.trace_ctx.record_op(OpCode::IntEq, &[start_i, zero]);
        ctx.trace_ctx
            .set_opref_concrete(starts_at_zero, majit_ir::Value::Int(one_arg as i64));
        let start_guard = if one_arg {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };
        walker_emit_guard_with_snapshot(ctx, op_pc, start_guard, &[starts_at_zero])?;
    }

    let (size_descr, iter_type_addr) = if !concrete_promote_step {
        (
            crate::descr::w_range_iter_size_descr(),
            &pyre_object::functional::RANGE_ITER_TYPE as *const _ as i64,
        )
    } else if one_arg {
        (
            crate::descr::w_range_iter_one_arg_size_descr(),
            &pyre_object::functional::RANGE_ITER_ONE_ARG_TYPE as *const _ as i64,
        )
    } else {
        (
            crate::descr::w_range_iter_step_one_size_descr(),
            &pyre_object::functional::RANGE_ITER_STEP_ONE_TYPE as *const _ as i64,
        )
    };
    let new = ctx
        .trace_ctx
        .record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.trace_ctx.heap_cache_mut().new_object(new);

    // `stop` is `start + length` rather than the range's own stop: a promoted
    // step is one, so the two agree over any non-empty span, and an empty or
    // backwards span this way ends the walk on its first compare instead of
    // carrying a bound below `start`.
    let iter_fields: Vec<(majit_ir::DescrRef, OpRef)> = if !concrete_promote_step {
        vec![
            (crate::descr::range_iter_current_descr(), start_i),
            (crate::descr::range_iter_remaining_descr(), length_i),
            (crate::descr::range_iter_step_descr(), step_i),
        ]
    } else if one_arg {
        vec![
            (crate::descr::range_iter_one_arg_current_descr(), start_i),
            (crate::descr::range_iter_one_arg_stop_descr(), one_past),
        ]
    } else {
        vec![
            (crate::descr::range_iter_step_one_current_descr(), start_i),
            (crate::descr::range_iter_step_one_stop_descr(), one_past),
            (crate::descr::range_iter_step_one_start_descr(), start_i),
        ]
    };
    for (descr, value) in iter_fields {
        let index = descr.index();
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[new, value], descr);
        ctx.trace_ctx.heapcache_setfield_cached(new, index, value);
    }

    ctx.trace_ctx.heap_cache_mut().class_now_known(new);

    let real_iter = unsafe { pyre_object::functional::w_range_iter(range_obj) };
    ctx.trace_ctx.set_opref_concrete(
        new,
        majit_ir::Value::Ref(majit_ir::GcRef(real_iter as usize)),
    );
    ctx.frame_state.borrow_mut().vstack_last_ref = new;

    Ok(Some(new))
}

/// `ForIterNext` for an arity-two `zip` over two `W_TupleIterObject`
/// cursors. Admission stays here; the step is `functional.py`
/// `W_Zip.next_w` recorded from `baseobjspace::zip_two_tuple_next`.
///
/// Mixed lengths and a non-strict exhaust decline, so `strict`'s
/// `ValueError` stays on the interpreter. Guards resume at this FOR_ITER:
/// `zip_two_tuple_next_call_site` is not the list-iter class-guard marker.
/// `setfield_gc` into the live cursors is record-only, so the concrete
/// index store or `seq` clear is applied once after the walk.
fn try_walker_specialize_zip_two_tuple_iters<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    zip_op: OpRef,
    zip_obj: pyre_object::PyObjectRef,
) -> Result<Option<OpRef>, DispatchError> {
    if ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }

    let zip_type = &pyre_object::functional::ZIP_TYPE as *const pyre_object::PyType;
    let zip_class = pyre_object::get_instantiate(&pyre_object::functional::ZIP_TYPE);
    let (_iterators_obj, inner_objs, steps, strict) = unsafe {
        if zip_obj.is_null()
            || !std::ptr::eq((*zip_obj).ob_type, zip_type)
            || !std::ptr::eq((*zip_obj).w_class, zip_class)
        {
            return Ok(None);
        }
        let iterators_obj = pyre_object::functional::w_zip_get_iterators(zip_obj);
        if iterators_obj.is_null()
            || !pyre_object::is_list(iterators_obj)
            || !pyre_object::is_exact_builtin_instance(iterators_obj)
        {
            return Ok(None);
        }
        let list = &*(iterators_obj as *const pyre_object::listobject::W_ListObject);
        if list.strategy != pyre_object::listobject::ListStrategy::Object
            || pyre_object::w_list_len(iterators_obj) != 2
        {
            return Ok(None);
        }
        let Some(inner0) = pyre_object::w_list_getitem(iterators_obj, 0) else {
            return Ok(None);
        };
        let Some(inner1) = pyre_object::w_list_getitem(iterators_obj, 1) else {
            return Ok(None);
        };

        let mut steps = Vec::with_capacity(2);
        for inner in [inner0, inner1] {
            if !pyre_object::is_tuple_iter(inner)
                || !std::ptr::eq((*inner).ob_type, &pyre_object::iterobject::TUPLE_ITER_TYPE)
            {
                return Ok(None);
            }
            let seq = pyre_object::w_tuple_iter_seq(inner);
            let index = pyre_object::w_tuple_iter_index(inner);
            // `W_FastTupleIterObject.descr_next` indexes `tupleitems` captured
            // by `W_AbstractTupleObject.descr_iter` from `tolist()`. Only
            // `W_TupleObject.wrappeditems` is that array. An arity-2
            // `W_SpecialisedTupleObject_*` stores `value0`/`value1` inline, so
            // a `wrappeditems` load reads the payload as a pointer.
            if seq.is_null()
                || !std::ptr::eq((*seq).ob_type, &pyre_object::TUPLE_TYPE)
                || !pyre_object::is_exact_builtin_instance(seq)
                || index < 0
            {
                return Ok(None);
            }
            let len = pyre_object::w_tuple_len(seq) as isize;
            let item = pyre_object::w_tuple_getitem(seq, pyre_object::seq_index_to_i64(index));
            steps.push((seq, index, len, item));
        }
        (
            iterators_obj,
            [inner0, inner1],
            steps,
            pyre_object::functional::w_zip_get_strict(zip_obj),
        )
    };
    let concrete_continues = steps.iter().all(|step| step.3.is_some());
    let concrete_exhausted = steps.iter().all(|step| step.1 >= step.2);
    // Mixed bounds must fall through to the interpreter's strict error path.
    if !concrete_continues && !(strict && concrete_exhausted) {
        return Ok(None);
    }

    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode(
        "pyre_interpreter::baseobjspace::zip_two_tuple_next",
    ) else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    if unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };

    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    // One class guard, as `try_walker_orthodox_list_iter_next` does. The
    // exact-class and cursor checks above are the admission; the helper
    // body records the list, tuple, and bounds guards.
    walker_guard_class(ctx, op_pc, zip_op, zip_type as i64)?;
    ctx.trace_ctx
        .set_opref_concrete(zip_op, Value::Ref(majit_ir::GcRef(zip_obj as usize)));
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "zip_two_tuple_next_commit",
        "zip_two_tuple_next_call_site",
        &[],
        &[],
        &[zip_op],
        &[ConcreteValue::Ref(zip_obj)],
        &[],
    );
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc, .. }) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[decline-why] ZIP-TWO-TUPLE-NEXT pc={pc}");
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let result = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };

    // `setfield_gc` into a pre-existing iterator is record-only. Apply the
    // cursor the helper recorded, once, when the live object did not move.
    for (inner, step) in inner_objs.into_iter().zip(steps.iter()) {
        let index_after = unsafe { pyre_object::w_tuple_iter_index(inner) };
        let seq_after = unsafe { pyre_object::w_tuple_iter_seq(inner) };
        if seq_after != step.0 || index_after != step.1 {
            continue;
        }
        if ctx.trace_ctx.is_bridge_trace {
            fbw_bridge_tuple_iter_journal_push(inner, step.0, step.1);
        }
        if concrete_continues {
            unsafe { pyre_object::w_tuple_iter_set_index(inner, step.1 + 1) };
        } else {
            unsafe { pyre_object::w_tuple_iter_set_seq(inner, pyre_object::PY_NULL) };
        }
    }

    if concrete_continues {
        let concrete_tuple = if let Some(obj) = walker_concrete_ref_object(ctx, result)
            && !obj.is_null()
        {
            obj
        } else {
            let item0 = steps[0].3.expect("continue-arm zip step has item 0");
            let item1 = steps[1].3.expect("continue-arm zip step has item 1");
            let allocated = pyre_object::w_specialised_tuple_oo_new(item0, item1);
            if allocated.is_null() {
                return Err(DispatchError::ConcreteShadowAllocationFailed { pc: op_pc });
            }
            ctx.trace_ctx
                .set_opref_concrete(result, Value::Ref(majit_ir::GcRef(allocated as usize)));
            allocated
        };
        unsafe { pyre_object::functional::w_zip_set_iteration_progress(zip_obj, 1) };
        fbw_foriter_inflight_capture(concrete_tuple, body, true);
        ctx.frame_state.borrow_mut().vstack_last_ref = result;
    } else {
        unsafe { pyre_object::functional::w_zip_set_iteration_progress(zip_obj, 0) };
        if !matches!(
            ctx.trace_ctx.concrete_of_opref(result),
            Some(majit_ir::Value::Ref(r)) if r.as_usize() == 0
        ) {
            ctx.trace_ctx
                .set_opref_concrete(result, Value::Ref(majit_ir::GcRef(0)));
        }
    }
    Ok(Some(result))
}

/// Which of the two `step == 1` iterator shapes a FOR_ITER is walking.
/// They differ only in the class the FOR_ITER guard compares.
/// `W_IntRangeOneArgIterator` additionally promises a non-negative cursor.
#[derive(Clone, Copy)]
pub(crate) enum RangeStepOneShape {
    StepOne,
    OneArg,
}

impl RangeStepOneShape {
    fn type_addr(self) -> i64 {
        match self {
            Self::StepOne => &pyre_object::functional::RANGE_ITER_STEP_ONE_TYPE as *const _ as i64,
            Self::OneArg => &pyre_object::functional::RANGE_ITER_ONE_ARG_TYPE as *const _ as i64,
        }
    }

    fn next_path(self) -> &'static str {
        match self {
            Self::StepOne => "pyre_object::functional::w_range_iter_step_one_next",
            Self::OneArg => "pyre_object::functional::w_range_iter_one_arg_next",
        }
    }

    unsafe fn replay(self, iter_obj: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
        match self {
            Self::StepOne => unsafe {
                pyre_object::functional::w_range_iter_step_one_next(iter_obj)
            },
            Self::OneArg => unsafe { pyre_object::functional::w_range_iter_one_arg_next(iter_obj) },
        }
    }
}

/// Walker-native `ForIterNext` for the two `step == 1` range-iterator shapes.
///
/// Step-1 `range` `FOR_ITER`. `stop` is immutable. The trace is that
/// shape's own `next`: compare `current` with `stop`, advance, then box.
/// The class guard uses the FOR_ITER green key. The cursor journal is the
/// pre-advance `(current, remaining)` so a later abort can restore it.
fn try_walker_orthodox_for_iter_range_step_one<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    iter_op: OpRef,
    iter_obj: pyre_object::PyObjectRef,
    range_green_key: Option<u64>,
    shape: RangeStepOneShape,
) -> Result<Option<OpRef>, DispatchError> {
    let Some(jc_arc) = crate::jitcode_runtime::pathed_jitcode_cached(shape.next_path()) else {
        return Ok(None);
    };
    let Some(sub_body) = sub_jitcode_body_by_index(jc_arc.index()) else {
        return Ok(None);
    };
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() || unsafe { (&*sym_ptr).jitcode().is_null() } {
        return Ok(None);
    };

    let (concrete_current, concrete_remaining, _concrete_step) =
        unsafe { pyre_object::functional::w_range_iter_fields(iter_obj) };
    let concrete_continues = concrete_remaining != 0;

    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    let type_addr = shape.type_addr();
    walker_guard_fold_class_foriter(ctx, op_pc, iter_op, type_addr, range_green_key)?;
    let sym = unsafe { &*sym_ptr };
    let Ok(nested_entry) = orthodox_helper_nested_entry(ctx, op_pc) else {
        return Ok(None);
    };
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let journal_mark = fbw_effect_journal_mark();
    ctx.trace_ctx
        .set_opref_concrete(iter_op, Value::Ref(majit_ir::GcRef(iter_obj as usize)));
    let walk = run_orthodox_helper_subwalk(
        ctx,
        op_pc,
        sym,
        &sub_body,
        nested_entry,
        "for_iter_range_step_one_commit",
        "range_iter_next_call_site",
        &[],
        &[],
        &[iter_op],
        &[ConcreteValue::Ref(iter_obj)],
        &[],
    );
    // The subwalk can collect; the op's stamped value is the forwarded address.
    let iter_now: Option<pyre_object::PyObjectRef> = if iter_op.is_constant() {
        Some(iter_obj)
    } else {
        match ctx.trace_ctx.concrete_of_opref(iter_op) {
            Some(Value::Ref(r)) => Some(r.as_usize() as pyre_object::PyObjectRef),
            _ => None,
        }
    };
    let (walk_outcome, _) = match walk {
        Ok(pair) => pair,
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { .. }) => {
            if let Some(iter_obj) = iter_now {
                unsafe {
                    pyre_object::functional::w_range_iter_set_cursor(
                        iter_obj,
                        concrete_current,
                        concrete_remaining,
                    );
                }
            }
            ctx.trace_ctx.cut_trace_with_snapshots(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let Some(iter_obj) = iter_now else {
        return Ok(None);
    };
    let item = match walk_outcome {
        DispatchOutcome::SubReturn { result } => finish_inline_callee_return(ctx, result)
            .ok_or(DispatchError::UnexpectedVoidSubReturn { pc: op_pc })?,
        _ => return Err(DispatchError::UnexpectedVoidSubReturn { pc: op_pc }),
    };
    // The walked body records the compare, the store and the box, and its
    // field stores moved the live cursor.  Run the same step once for a walk
    // that could not execute them, i.e. while the cursor is still the
    // pre-iteration value.
    fbw_gc_store_journal_keep_since(journal_mark, iter_obj);
    if concrete_continues {
        let (after, _, _) = unsafe { pyre_object::functional::w_range_iter_fields(iter_obj) };
        if after == concrete_current {
            let concrete_item = unsafe { shape.replay(iter_obj) };
            if concrete_item.is_null() {
                return Ok(None);
            }
            ctx.trace_ctx
                .set_opref_concrete(item, Value::Ref(majit_ir::GcRef(concrete_item as usize)));
        }
    }
    let Some(concrete_item_ptr) = walker_concrete_ref_object(ctx, item) else {
        return Ok(Some(item));
    };
    if !concrete_continues {
        return Ok(Some(item));
    }

    // Journal on root walks too: a non-commit root abort leaves
    // delivery as the only way to keep this item, and delivery
    // refuses when a body-effect signal stands.  Without a cursor
    // snapshot that refuse cannot restore, so the next FOR_ITER
    // yields the following item (`fbw_foriter_item_dropped`).
    fbw_bridge_iter_journal_push(iter_obj, concrete_current, concrete_remaining);
    fbw_foriter_inflight_capture(concrete_item_ptr, body, true);
    ctx.frame_state.borrow_mut().vstack_last_ref = item;

    Ok(Some(item))
}

pub(crate) fn try_walker_specialize_for_iter_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
    _dst: usize,
    dst_bank: char,
) -> Result<Option<OpRef>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 1 {
        return Ok(None);
    }

    // A range class-guard failure at this FOR_ITER green key is a definitive
    // polymorphism witness.  Once the failure path has demoted it, retain the
    // generic residual rather than recreating the range guard on retrace.
    let range_green_key = walker_foriter_green_key(ctx, op_pc);
    if range_green_key.is_some_and(crate::trace::range_foriter_demoted) {
        return Ok(None);
    }

    let iter_op = r_args[0];
    let Some(iter_obj) = walker_concrete_ref_object(ctx, iter_op) else {
        return Ok(None);
    };
    if unsafe { pyre_object::functional::is_zip(iter_obj) } {
        return spec_gate(SpecFold::ZipTwoTupleIters, || {
            try_walker_specialize_zip_two_tuple_iters(ctx, op_pc, iter_op, iter_obj)
        });
    }
    // `iterobject.py` `W_FastListIterObject.descr_next` is recorded by
    // `try_walker_orthodox_list_iter_next` once this fold declines.
    if unsafe { pyre_object::is_list_iter(iter_obj) } {
        return Ok(None);
    }
    if unsafe { pyre_object::functional::is_range_iter_one_arg(iter_obj) } {
        return spec_gate(SpecFold::ForIterNext, || {
            try_walker_orthodox_for_iter_range_step_one(
                ctx,
                op_pc,
                iter_op,
                iter_obj,
                range_green_key,
                RangeStepOneShape::OneArg,
            )
        });
    }
    if unsafe { pyre_object::functional::is_range_iter_step_one(iter_obj) } {
        return spec_gate(SpecFold::ForIterNext, || {
            try_walker_orthodox_for_iter_range_step_one(
                ctx,
                op_pc,
                iter_op,
                iter_obj,
                range_green_key,
                RangeStepOneShape::StepOne,
            )
        });
    }
    let (concrete_current, concrete_remaining, concrete_step) = unsafe {
        if !pyre_object::functional::is_range_iter_general(iter_obj) {
            return Ok(None);
        }
        pyre_object::functional::w_range_iter_fields(iter_obj)
    };
    let concrete_continues = concrete_remaining != 0;

    // A new consume attempt completes the prior in-flight iteration before
    // this irreversible concrete advance, matching the residual executor.
    let body = fbw_foriter_body_from_op_pc(ctx, op_pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body);

    // guard_class W_IntRangeIterator, unless the operand is already known.
    let range_iter_type_addr = &pyre_object::functional::RANGE_ITER_TYPE as *const _ as i64;
    walker_guard_fold_class_foriter(ctx, op_pc, iter_op, range_iter_type_addr, range_green_key)?;

    if !concrete_continues {
        // Exhausted arrival: the walker concretely reached remaining==0 (a nested
        // inner loop run to completion inside the outer body).  Record the
        // routing guard for the false continue predicate, then present the
        // exhaustion edge exactly as the residual does: a NULL Ref that the
        // codewriter's trailing GuardNonnull consumes as the loop exit.  The
        // iterator is already exhausted, so no cursor advance and no in-flight
        // capture.
        let zero = ctx.trace_ctx.const_int(0);
        let remaining = crate::state::opimpl_getfield_gc_i(
            ctx.trace_ctx,
            iter_op,
            crate::descr::range_iter_remaining_descr(),
        );
        let continues = ctx.trace_ctx.record_op(OpCode::IntGt, &[remaining, zero]);
        ctx.trace_ctx.set_opref_concrete(continues, Value::Int(0));
        walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardFalse, &[continues])?;
        let null_item = ctx.trace_ctx.record_op(OpCode::CastIntToPtr, &[zero]);
        ctx.trace_ctx
            .set_opref_concrete(null_item, Value::Ref(majit_ir::GcRef(0)));
        return Ok(Some(null_item));
    }

    // Guard the continue arm before constructing the item.  The false arm
    // resumes at this FOR_ITER, where the interpreter takes the existing
    // exhaustion edge (iterator retained, no item pushed).  This avoids the
    // pointer-mask representation which forced the item to be materialized.
    let current = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_current_descr(),
    );
    let remaining = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_remaining_descr(),
    );
    let step = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        iter_op,
        crate::descr::range_iter_step_descr(),
    );
    let zero = ctx.trace_ctx.const_int(0);
    let continues = ctx.trace_ctx.record_op(OpCode::IntGt, &[remaining, zero]);
    ctx.trace_ctx
        .set_opref_concrete(continues, Value::Int(concrete_continues as i64));
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[continues])?;

    // The continue guard establishes `continues == 1` on the trace path. Keep
    // the wrapping IntAdd and live-iterator SetfieldGc updates intact.
    let delta = ctx.trace_ctx.record_op(OpCode::IntMul, &[step, continues]);
    ctx.trace_ctx.set_opref_concrete(
        delta,
        Value::Int(concrete_step.wrapping_mul(concrete_continues as i64)),
    );
    let next_current = ctx.trace_ctx.record_op(OpCode::IntAdd, &[current, delta]);
    let next_current_concrete =
        concrete_current.wrapping_add(concrete_step.wrapping_mul(concrete_continues as i64));
    ctx.trace_ctx
        .set_opref_concrete(next_current, Value::Int(next_current_concrete));
    let current_descr = crate::descr::range_iter_current_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_current],
        current_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, current_descr.index(), next_current);

    let next_remaining = ctx
        .trace_ctx
        .record_op(OpCode::IntSub, &[remaining, continues]);
    let next_remaining_concrete = concrete_remaining.wrapping_sub(concrete_continues as i64);
    ctx.trace_ctx
        .set_opref_concrete(next_remaining, Value::Int(next_remaining_concrete));
    let remaining_descr = crate::descr::range_iter_remaining_descr();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_remaining],
        remaining_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, remaining_descr.index(), next_remaining);

    // `wrapint` is the transparent `NewWithVtable(W_IntObject)` +
    // `SetfieldGc(intval=current)` shape allocation removal virtualizes.  Do
    // not feed it through pointer arithmetic: locally consumed items stay
    // virtual, while normal forcing materializes escaping items.
    let item = crate::state::wrapint(ctx.trace_ctx, current);

    // Tracing executes the real range cursor advance.  The direct helper is
    // the same `W_IntRangeIterator.next` implementation used by the residual;
    // do not journal it, because abort recovery forwards this exact item.
    let concrete_item = unsafe { pyre_object::functional::w_range_iter_next(iter_obj) };
    debug_assert_eq!(concrete_item.is_some(), concrete_continues);
    let concrete_item_ptr = concrete_item.expect("GuardTrue(continues) implies a range item");
    ctx.trace_ctx.set_opref_concrete(
        item,
        Value::Ref(majit_ir::GcRef(concrete_item_ptr as usize)),
    );

    // Keep the virtual payload's concrete shadow paired with the concrete New.
    // A later body guard can then encode the virtual `i` in its snapshot and
    // blackhole will rematerialize the right item on deopt.
    ctx.trace_ctx
        .set_opref_concrete(current, Value::Int(concrete_current));

    // Same journal as the step-one shape: a root abort that then
    // refuses in-flight delivery must be able to restore the cursor.
    fbw_bridge_iter_journal_push(iter_obj, concrete_current, concrete_remaining);
    fbw_foriter_inflight_capture(concrete_item_ptr, body, true);
    // Range iteration stays at the C level, so the operand-stack mirror
    // remains valid and must receive the item produced by FOR_ITER.  Its
    // virtual state is captured by subsequent body-guard snapshots.
    ctx.frame_state.borrow_mut().vstack_last_ref = item;

    Ok(Some(item))
}

/// Admission for `COMPARE_OP_DESCENT` on tags 6/7.  Same job as the
/// exact-numeric gate on tags 0..=5: do not start a sub-walk whose body
/// can run Python (`__hash__` / `__eq__` / a subclass `__contains__`).
/// A declining residual would re-run those side effects.  This is not
/// a type-specialization fold: it is the callback-free gate the other
/// compare-op descent already uses.  Removing it would re-run a stored
/// `__eq__` when the sub-walk then declines.
///
/// Exact `str`/`bytes` plus a needle whose membership is an elidable
/// find (another exact `str`/`bytes`, or a byte in `range(256)`), and
/// an exact `IntegerListStrategy` list plus a plain `int`, are
/// callback-free.  `dict`/`set` stay on the residual: a stored
/// element's `__eq__` can still run on a hash collision.
fn walker_contains_descent_callback_free(
    needle: pyre_object::PyObjectRef,
    haystack: pyre_object::PyObjectRef,
) -> bool {
    let exact = |obj: pyre_object::PyObjectRef, tp: &pyre_object::pyobject::PyType| unsafe {
        pyre_object::is_exact_type(obj, tp)
            && std::ptr::eq((*obj).w_class, pyre_object::get_instantiate(tp))
    };
    if exact(haystack, &pyre_object::pyobject::STR_TYPE) {
        return exact(needle, &pyre_object::pyobject::STR_TYPE)
            && unsafe { pyre_object::w_str_get_value_opt(needle).is_some() };
    }
    if exact(haystack, &pyre_object::bytesobject::BYTES_TYPE) {
        if exact(needle, &pyre_object::bytesobject::BYTES_TYPE) {
            return true;
        }
        return unsafe {
            pyre_object::listobject::is_plain_int1(needle) && pyre_object::is_int(needle)
        } && (0..=255).contains(&unsafe { pyre_object::w_int_get_value(needle) });
    }
    if exact(haystack, &pyre_object::pyobject::LIST_TYPE) {
        return unsafe {
            pyre_object::listobject::w_list_strategy(haystack)
                == pyre_object::listobject::ListStrategy::Integer
                && pyre_object::listobject::is_plain_int1(needle)
                && pyre_object::is_int(needle)
        };
    }
    // A `set` stays on the residual even for a plain int needle: its single
    // object strategy can hold a user object whose `__hash__` collides with
    // the needle's, and the same-hash probe then runs that object's `__eq__`.
    false
}

/// `guard_class(&STR_TYPE)` + the exact canonical `w_class` guard, the pair
/// that keeps a `str` subclass out of a fold written for the builtin body.
fn walker_guard_exact_str<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    operand: OpRef,
) -> Result<(), DispatchError> {
    walker_guard_exact_instance(
        ctx,
        op_pc,
        operand,
        &pyre_object::pyobject::STR_TYPE as *const _ as i64,
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::STR_TYPE),
    )
}

/// Specialize `STORE_SUBSCR target[const_slice] = source` for a same-length,
/// step-1 slice between two Integer-strategy exact lists, eliding the
/// `CALL_MAY_FORCE` `store_subscr` residual that would force the virtualizable
/// source list (the freshly built BUILD_LIST temp from
/// [`try_walker_specialize_newlist`]) every iteration.  The same-length gate
/// makes the assignment `slice_len` independent in-bounds setitems —
/// `target[start + j] = source[j]` — with no resize and no strategy change, so
/// it rides the existing `FBW_STORE_JOURNAL` per-element undo log.
///
/// Reads the source elements through `getfield_gc(int_items)` +
/// `getarrayitem_gc` ops keyed on the source `OpRef`, so when the source is the
/// freshly built virtual list the optimizer folds the reads against its
/// recorded `SetarrayitemGc` stores and removes the whole temporary.
///
/// The slice key must be a trace constant (a `slice(...)` from `co_consts`);
/// `start` / `stop` are read off the slice object and baked into the emitted
/// index constants.  Falls through to the generic residual (returns `Ok(None)`)
/// for anything outside the gate: a non-constant / `None` / negative bound, a
/// non-unit step, a resizing (length-changing) slice, an empty slice, a
/// non-Integer-storage target or source, or a list subclass (which may override
/// `__setitem__` / `__iter__`).
pub(crate) fn try_walker_specialize_setslice<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    r_args: &[OpRef],
) -> Result<Option<()>, DispatchError> {
    if !ctx.is_authoritative_executor || r_args.len() != 3 {
        return Ok(None);
    }
    let list_op = r_args[0];
    let key_op = r_args[1];
    let value_op = r_args[2];
    // The slice key must be a trace constant (a `slice(...)` from `co_consts`):
    // its `start` / `stop` are baked into the emitted index constants, so a
    // non-constant slice (whose bounds could differ at runtime) cannot be
    // specialized this way.
    if !key_op.is_constant() {
        return Ok(None);
    }
    let (Some(list_obj), Some(key_obj), Some(value_obj)) = (
        walker_concrete_ref_object(ctx, list_op),
        walker_concrete_ref_object(ctx, key_op),
        walker_concrete_ref_object(ctx, value_op),
    ) else {
        return Ok(None);
    };

    // Gate, all read from the concrete shadows: `target[start:stop:1] =
    // source`, both exact-list Integer storage, `stop - start == len(source)`
    // (no resize), `1 <= slice_len`, `0 <= start <= stop <= len(target)`.
    let (start, slice_len) = unsafe {
        // EXACT list for BOTH target and source: a list subclass shares
        // `ob_type == &LIST_TYPE` but retags `w_class` and may override
        // `__setitem__` (target) or `__iter__` (source); both must route
        // through the generic residual.
        if !pyre_object::pyobject::is_exact_list(list_obj)
            || !pyre_object::is_slice(key_obj)
            || !pyre_object::pyobject::is_exact_list(value_obj)
        {
            return Ok(None);
        }
        // step == 1 (None defaults to 1; an explicit non-1 step needs the
        // strided path).
        let step_o = pyre_object::w_slice_get_step(key_obj);
        let step_is_one = pyre_object::is_none(step_o)
            || (pyre_object::is_int(step_o)
                && !pyre_object::is_bool(step_o)
                && pyre_object::w_int_get_value(step_o) == 1);
        if !step_is_one {
            return Ok(None);
        }
        // start / stop must be explicit non-negative plain ints (None bounds and
        // negative indices route through the generic residual, which normalises
        // them).
        let start_o = pyre_object::w_slice_get_start(key_obj);
        let stop_o = pyre_object::w_slice_get_stop(key_obj);
        if !(pyre_object::is_int(start_o)
            && !pyre_object::is_bool(start_o)
            && pyre_object::is_int(stop_o)
            && !pyre_object::is_bool(stop_o))
        {
            return Ok(None);
        }
        let start = pyre_object::w_int_get_value(start_o);
        let stop = pyre_object::w_int_get_value(stop_o);
        let target_len = pyre_object::w_list_len(list_obj) as i64;
        if start < 0 || stop < start || stop > target_len {
            return Ok(None);
        }
        let slice_len = stop - start;
        let src_len = pyre_object::w_list_len(value_obj) as i64;
        // Same-length only — a resizing slice changes the target length and can
        // switch strategy.
        if slice_len != src_len || slice_len < 1 {
            return Ok(None);
        }
        if !(pyre_object::w_list_uses_int_storage(list_obj)
            && pyre_object::w_list_uses_int_storage(value_obj))
        {
            return Ok(None);
        }
        (start, slice_len)
    };

    // emit the specialized IR (walker-native)
    // For BOTH target (`list_op`) and source (`value_op`): guard_class LIST +
    // exact `w_class` (a list subclass sharing `ob_type == &LIST_TYPE` but with
    // an overridden `__setitem__` / `__iter__` side-exits to the generic
    // residual) + guard strategy == Integer.  Folds away when the operand is the
    // just-built virtual list.
    let list_type_addr = &pyre_object::pyobject::LIST_TYPE as *const _ as i64;
    let list_instantiate =
        pyre_object::pyobject::get_instantiate(&pyre_object::pyobject::LIST_TYPE);
    let sid_const_val = pyre_object::listobject::ListStrategy::Integer as i64;
    for &lst_op in &[list_op, value_op] {
        walker_guard_exact_w_class(ctx, op_pc, lst_op, list_instantiate)?;
        walker_guard_fold_class(ctx, op_pc, lst_op, list_type_addr)?;
        walker_guard_fold_list_strategy(ctx, op_pc, lst_op, sid_const_val)?;
    }

    // Bounds guard on the target: the highest written index `start + slice_len -
    // 1` must be in range.  For an Integer-strategy list the `W_ListObject`
    // `length` field is 0 — the authoritative length is `int_items.len`, so read
    // it via `list_int_items_len_descr` (exactly as store_subscr's bounds
    // guard).  IntLt(start+slice_len-1, target.int_items.len).
    let tgt_len_box = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_int_items_len_descr(),
    );
    let last_idx_const = ctx.trace_ctx.const_int(start + slice_len - 1);
    let in_bounds = ctx
        .trace_ctx
        .record_op(OpCode::IntLt, &[last_idx_const, tgt_len_box]);
    let concrete_target_len = unsafe { pyre_object::w_list_len(list_obj) as i64 };
    ctx.trace_ctx.set_opref_concrete(
        in_bounds,
        majit_ir::Value::Int(((start + slice_len - 1) < concrete_target_len) as i64),
    );
    walker_emit_guard_with_snapshot(ctx, op_pc, OpCode::GuardTrue, &[in_bounds])?;

    // Length guard on the source: source.int_items.len == slice_len (folds for
    // the virtual temp; protects a non-virtual source).
    let src_len_box = crate::state::opimpl_getfield_gc_i(
        ctx.trace_ctx,
        value_op,
        crate::descr::list_int_items_len_descr(),
    );
    walker_guard_fold_int(ctx, op_pc, src_len_box, slice_len)?;

    // items[start + j] = source.items[j] for j in 0..slice_len, through the
    // int_items blocks (`list_int_items_block_descr`, matching
    // `emit_typed_list_inline`'s `SetfieldGc`), so a virtual source temp's
    // `SetarrayitemGc` stores fold against these reads.
    let src_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        value_op,
        crate::descr::list_int_items_block_descr(),
    );
    let tgt_block = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        list_op,
        crate::descr::list_int_items_block_descr(),
    );
    for j in 0..slice_len {
        let src_idx = ctx.trace_ctx.const_int(j);
        let src_raw =
            crate::state::trace_int_block_getitem_value(ctx.trace_ctx, src_block, src_idx);
        let tgt_idx = ctx.trace_ctx.const_int(start + j);
        crate::state::trace_int_block_setitem_value(ctx.trace_ctx, tgt_block, tgt_idx, src_raw);
    }

    // Tracing is execution (pyjitpl.py execute_and_record): apply the
    // assignment to the concrete lists now as `slice_len` in-bounds setitems,
    // journaling each displaced element first so a non-committing walk's legacy
    // replay re-executes against the pre-walk heap (FBW_STORE_JOURNAL).  Each
    // `w_list_getitem` / `w_int_new` boxes, and a minor collection there can
    // move any live GC object.  Following the push_roots/pop_roots reload
    // discipline, every live ref is reloaded after each boxing allocation,
    // before its next use: walker operands (`list_obj`/`value_obj`) from the
    // forwarded shadow via `walker_concrete_ref_object`, and the pinned fresh
    // boxes (`src_item`/`displaced`) from their shadow-stack slot via
    // `shadow_stack_get` (the slot index captured just before the pin).
    {
        let _roots = pyre_object::gc_roots::push_roots();
        for j in 0..slice_len {
            let tgt_index = start + j;
            let Some(value_obj) = walker_concrete_ref_object(ctx, value_op) else {
                unreachable!("setslice specialization: operand concrete vanished from the shadow");
            };
            let Some(src_item) = (unsafe { pyre_object::w_list_getitem(value_obj, j) }) else {
                unreachable!("setslice specialization: source index {j} has no element");
            };
            let src_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(src_item);
            let Some(list_obj) = walker_concrete_ref_object(ctx, list_op) else {
                unreachable!("setslice specialization: operand concrete vanished from the shadow");
            };
            let Some(displaced) = (unsafe { pyre_object::w_list_getitem(list_obj, tgt_index) })
            else {
                unreachable!(
                    "setslice specialization: target index {tgt_index} has no element \
                     (bounds gate admitted it)"
                );
            };
            let disp_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(displaced);
            let key_box = pyre_object::w_int_new(tgt_index);
            let key_box = pyre_object::gc_roots::pin_root(key_box);
            let Some(list_obj) = walker_concrete_ref_object(ctx, list_op) else {
                unreachable!("setslice specialization: list concrete vanished mid-apply");
            };
            let src_item = pyre_object::gc_roots::shadow_stack_get(src_slot);
            let displaced = pyre_object::gc_roots::shadow_stack_get(disp_slot);
            fbw_store_journal_push(list_obj, key_box, displaced);
            let stored = unsafe { pyre_object::w_list_setitem(list_obj, tgt_index, src_item) };
            debug_assert!(stored, "setslice specialization: in-bounds store failed");
        }
    }
    Ok(Some(()))
}

/// #62 LoadGlobal cell-cache fold — walker mirror of the retired trait
/// LOAD_GLOBAL fast path.
///
/// When `ns` is a `W_ModuleDictObject` still in `ModuleDictStrategy` mode
/// whose slot for `name` holds a raw value or an `ObjectMutableCell`, emit
/// `QUASIIMMUT_FIELD(ns, slot)` + `RECORD_KNOWN_RESULT` + an elidable cell
/// lookup that the optimizer folds to the constant cell pointer.  The
/// strategy's `version?` watcher invalidates the loop (GUARD_NOT_INVALIDATED)
/// on any rebind, so the fold is sound while `load_global_fn` itself stays
/// `CallFlavor::Plain`.  Returns `Ok(true)` when the fold was emitted;
/// `Ok(false)` when the receiver is not a foldable cell (the caller then
/// falls through to the generic residual, which stays correct).
///
/// Callers fall back to the residual call when this fold declines. When the
/// loaded global is a function that is then CALLed, folding it to a
/// loop-invariant constant callee routes the call through the FBW call-inlining
/// path (#68).
///
/// Builtins fallback: when `name` is ABSENT from the
/// module dict but resolves through `frame.get_builtin()` (e.g.
/// `raise ValueError` / `except ValueError`), the same cell fold is emitted
/// against the BUILTINS dict, guarded additionally by a `QUASIIMMUT_FIELD` on
/// the module dict so adding `name` to globals (shadowing the builtin) bumps
/// the module-dict `version` and fails the loop's GUARD_NOT_INVALIDATED.  This
/// mirrors `bh_load_global_fn`'s `finditem_str(globals)` →
/// `get_builtin().getdictvalue` fallback chain.
pub(crate) fn try_walker_load_global_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    ns_ptr: usize,
    w_code_ptr: usize,
    frame_ptr: usize,
    namei: i64,
) -> Result<bool, DispatchError> {
    if w_code_ptr == 0 {
        return Ok(false);
    }
    let w_globals = ns_ptr as pyre_object::PyObjectRef;
    // The namespace operand is the fold's authority: both legs end at
    // `guard_current_frame_globals_identity`, which bakes it as the expected
    // `ConstPtr` and declines outright on a null one.  An inlined callee whose
    // namespace register is unseeded presents it as a null `Ref`, so decline
    // here instead of walking the builtins leg, which reads `__builtins__`
    // straight out of it.  The residual re-resolves the globals from the frame
    // it runs on, so declining stays correct.
    if w_globals.is_null() {
        return Ok(false);
    }
    // Raw dict access in the inlined-callee builtins leg below is valid only
    // for an exact plain dict or module dict.  Dict subclasses are legal exec
    // namespaces but use a different object layout, so leave them to the live
    // residual lookup.
    if !unsafe { pyre_object::is_dict(w_globals) } {
        return Ok(false);
    }
    // `namei` is the raw `LOAD_GLOBAL` oparg; bit 0 is the push-NULL flag,
    // so the `co_names` index is `namei >> 1` (mirror `bh_load_global_fn`).
    let name_idx = (namei as usize) >> 1;
    // The wrapper being non-null does not make its `code_ptr` non-null:
    // `w_code_new_with_hidden_applevel` leaves the field null for a
    // gateway builtin or a test fixture, and every sibling name lookup
    // screens it the same way.
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(false);
    };
    if code_deletes_name_from_ptr(w_code_ptr, &name) {
        return Ok(false);
    }
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, &name)? {
        return Ok(true);
    }

    // Builtins fallback: the name is absent from the
    // `ns_ptr` module dict.  Mirror `bh_load_global_fn`'s second leg —
    // `frame.get_builtin().getdictvalue(name)` — and fold the builtins cell
    // when the name resolves there.  Requires the live frame operand.
    // The builtins fallback needs the module `pick_builtin(w_globals)` picks
    // (`frame.get_builtin()`).  A live frame supplies it directly and also lets
    // us double-check the operand against the frame's AUTHORITATIVE globals
    // — `bh_load_global_fn` re-resolves the globals it consults from the LIVE
    // frame (`frame.get_w_globals()` when the frame owns `w_code`, else the
    // code's bound globals) and IGNORES the `namespace_ptr` operand.  The
    // `ns_ptr` hint usually equals that live dict; when it does not, nothing
    // here can prove what the residual would read, so decline.
    // An INLINED callee has no materialised frame (`frame_ptr == 0`, its
    // `portal_frame_reg` unseeded); derive the builtin module from the concrete
    // globals' `__builtins__` cell instead — the same object `pick_builtin`
    // resolves (`pick_builtin_obj` in baseobjspace.rs) and the one the
    // interpreter fallback would
    // rebuild for the resumed callee frame.  #670 keeps `__builtins__` in every
    // module dict, `ns_ptr` is the callee's own namespace field (so it is the
    // authoritative globals), and guard (a) below watches the globals `version`,
    // so a later `__builtins__` rebind fails the loop exactly as a
    // shadowing-name insert would.
    let w_builtin = if frame_ptr != 0 {
        let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::PyFrame) };
        let live_globals = if frame.pycode as usize == w_code_ptr {
            frame.get_w_globals()
        } else {
            unsafe {
                pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef)
            }
        };
        // Only the SAME dict makes the absence provable.  `module_dict_cell_slot_direct`
        // answers `None` both for a name that is absent and for a dict it cannot
        // read at all — a plain dict, or a module dict that ran
        // `switch_to_object_strategy` — so on a different dict its `None` says
        // nothing.  Guard (a) below pins `w_globals`' version, which watches the
        // wrong dict in that case, and the residual it replaces resolves
        // `live_globals`; a name present there would read the builtin instead of
        // the global.
        if live_globals.is_null() || live_globals as usize != w_globals as usize {
            return Ok(false);
        }
        frame.get_builtin()
    } else {
        unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }
            .unwrap_or(pyre_object::PY_NULL)
    };
    emit_builtins_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, w_builtin, &name)
}

/// Walk `pyframe::PyFrame::get_w_globals` when `debugdata` is absent.
///
/// The body is `pyframe.py PyFrame.get_w_globals`: a null `debugdata`
/// continues to `jit.promote(self.pycode).w_globals`. The portal frame is
/// the standard virtualizable, so that arm sub-walks the generated body
/// against that red frame. An inlined callee is a different red frame;
/// [`inlined_callee_frame_w_globals`] records the same promote off that
/// frame's own pycode. A missing jitcode, an empty body, or
/// [`DispatchError::OrthodoxSubWalkTraceUnsupported`] leaves the residual
/// call. Any other walk error aborts the portal.
pub(crate) fn try_walker_descend_frame_get_w_globals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    frame_op: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.trace_ctx.standard_virtualizable_box() != Some(frame_op) {
        return inlined_callee_frame_w_globals(ctx, op_pc, frame_op, dst, dst_bank);
    }
    let Some(frame_ptr) = ctx.trace_ctx.standard_virtualizable_ptr() else {
        return Ok(None);
    };
    let Some((_, majit_ir::Value::Ref(debugdata_ref))) = ctx
        .trace_ctx
        .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
    else {
        return Ok(None);
    };
    if debugdata_ref == majit_ir::GcRef::NO_CONCRETE || debugdata_ref.as_usize() != 0 {
        return Ok(None);
    }

    let Some(prep) = prepare_orthodox_descent(ctx, op_pc, &GET_W_GLOBALS_DESCENT) else {
        return Ok(None);
    };
    if prep.body.code.is_empty() {
        return Ok(None);
    }
    let frame_obj = frame_ptr as pyre_object::PyObjectRef;
    let walked = run_prepared_orthodox_descent(
        ctx,
        op_pc,
        prep,
        &[],
        &[(frame_op, frame_obj)],
        &[],
        dst,
        dst_bank,
        &GET_W_GLOBALS_DESCENT,
        None,
        false,
    )?;
    orthodox_descent_unit(ctx, op_pc, walked, None)
}

const GET_W_GLOBALS_DESCENT: HelperDescent = HelperDescent {
    path: "pyframe::PyFrame::get_w_globals",
    commit_label: "get_w_globals_commit",
    call_site_label: "get_w_globals_call_site",
    decline_tag: "GET-W-GLOBALS-SUBWALK",
};

/// `PyFrame.get_w_globals` for an inlined callee that is not the portal
/// virtualizable.
///
/// `emit_new_pyframe_inline_with_params` builds that frame with no
/// `debugdata` and stores `pycode` on it. The null-`debugdata` arm is
/// `jit.promote(self.pycode).w_globals`: a quasi-immutable `PyCode.w_globals`
/// read, the operand `globals_read_keeps_recorded_namespace` already accepts.
/// The pycode is this frame's. A non-null `debugdata` (a namespace override)
/// stays on the residual, which reads `debugdata.w_globals`.
fn inlined_callee_frame_w_globals<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    frame_op: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<Option<()>, DispatchError> {
    let Some(frame_obj) = walker_concrete_ref_object(ctx, frame_op) else {
        return Ok(None);
    };
    let frame = unsafe { &*(frame_obj as *const pyre_interpreter::pyframe::PyFrame) };
    if !frame.debugdata.is_null() || frame.pycode.is_null() {
        return Ok(None);
    }
    let pycode_obj = frame.pycode as pyre_object::PyObjectRef;
    if unsafe { !pyre_interpreter::is_code(pycode_obj) } {
        return Ok(None);
    }
    let w_globals = unsafe { pyre_interpreter::w_code_get_w_globals(pycode_obj) };
    if w_globals.is_null() {
        return Ok(None);
    }
    let pycode_bits = pycode_obj as i64;

    let debug_descr = crate::descr::pyframe_debugdata_descr();
    let debug_idx = debug_descr.index();
    let cached_debug = ctx.trace_ctx.heapcache_getfield_cached(frame_op, debug_idx);
    if let Some(cached) = cached_debug {
        if let Some(word) = ctx.trace_ctx.const_value(cached) {
            if word != 0 {
                return Ok(None);
            }
        } else if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.concrete_of_opref(cached) {
            if r != majit_ir::GcRef::NO_CONCRETE && r.as_usize() != 0 {
                return Ok(None);
            }
        }
    }

    let code_descr = crate::descr::pyframe_code_descr();
    let code_idx = code_descr.index();
    let cached_pycode = ctx.trace_ctx.heapcache_getfield_cached(frame_op, code_idx);
    if let Some(cached) = cached_pycode {
        let matches = if let Some(word) = ctx.trace_ctx.const_value(cached) {
            word == pycode_bits
        } else {
            match ctx.trace_ctx.concrete_of_opref(cached) {
                Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
                    r.as_usize() as i64 == pycode_bits
                }
                Some(_) => false,
                None => true,
            }
        };
        if !matches {
            return Ok(None);
        }
    }

    let debug_op = if let Some(cached) = cached_debug {
        cached
    } else {
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, frame_op, debug_descr)
    };
    if ctx.trace_ctx.const_value(debug_op) != Some(0) {
        walker_emit_fold_guard_with_snapshot(ctx, op_pc, OpCode::GuardIsnull, &[debug_op])?;
    }

    let pycode_op = if let Some(cached) = cached_pycode {
        cached
    } else {
        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, frame_op, code_descr)
    };
    let pycode_const = ctx.trace_ctx.const_ref(pycode_bits);
    let promoted = if ctx.trace_ctx.const_value(pycode_op) == Some(pycode_bits) {
        pycode_op
    } else {
        walker_emit_fold_guard_with_snapshot(
            ctx,
            op_pc,
            OpCode::GuardValue,
            &[pycode_op, pycode_const],
        )?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(pycode_op, pycode_const);
        pycode_const
    };
    let globals_op = crate::state::opimpl_getfield_gc_r(
        ctx.trace_ctx,
        promoted,
        crate::descr::pycode_w_globals_quasi_descr(),
    );
    if walker_concrete_ref_object(ctx, globals_op).is_none() {
        ctx.trace_ctx.set_opref_concrete(
            globals_op,
            majit_ir::Value::Ref(majit_ir::GcRef(w_globals as usize)),
        );
    }
    walker_flush_guard_not_invalidated(ctx, op_pc)?;
    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, globals_op)?;
    Ok(Some(()))
}

/// Trace `pyopcode.py IMPORT_NAME`'s frame reads.
///
/// `get_builtin` and a non-null debugdata read are field reads from the
/// live red frame. Null-debugdata `LoadImportGlobals` declines before the
/// nullity guard so the caller descends `pyframe::PyFrame::get_w_globals`.
/// An inlined callee keeps its own red frame; the descent reads that
/// frame's pycode rather than this portal's.
pub(crate) fn try_walker_import_frame_read_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    frame_op: OpRef,
    dst: usize,
    dst_bank: char,
) -> Result<bool, DispatchError> {
    if dst_bank != 'r'
        || ctx.trace_ctx.standard_virtualizable_box() != Some(frame_op)
        || ctx.trace_ctx.standard_virtualizable_ptr().is_none()
    {
        return Ok(false);
    }
    let frame_ptr = ctx
        .trace_ctx
        .standard_virtualizable_ptr()
        .expect("standard virtualizable pointer was checked above");
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };

    if helper == majit_ir::RuntimeHelperKind::LoadImport {
        // `PyFrame.get_builtin` returns this field directly on every normal
        // frame.  The null/EC fallback is uncommon and remains residual.
        let w_builtin = frame.w_builtin;
        if w_builtin.is_null() || !unsafe { pyre_object::is_module(w_builtin) } {
            return Ok(false);
        }
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if w_dict.is_null() {
            return Ok(false);
        }
        let Some(slot) = crate::state::module_dict_cell_slot_direct(w_dict, "__import__") else {
            return Ok(false);
        };
        let Some(stored) = crate::state::module_dict_cell_value_direct(w_dict, slot) else {
            return Ok(false);
        };
        if stored.is_null() {
            return Ok(false);
        }

        // The cell fold below bakes the builtin module's dict, so pin the
        // frame field that owns that choice.  A custom per-frame builtin (or
        // any future rebinding of the field) side-exits before using the old
        // dict.  Rebinding builtins.__import__ itself is covered by the module
        // dict's quasi-immutable version and mutable-cell read.
        let live_builtin = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            frame_op,
            crate::descr::pyframe_w_builtin_descr(),
        );
        walker_guard_stamped_ref_unless_const(ctx, op_pc, live_builtin, w_builtin)?;
        if live_builtin.is_constant()
            && ctx.trace_ctx.const_value(live_builtin) != Some(w_builtin as i64)
        {
            return Ok(false);
        }
        return emit_namespace_cell_fold(
            ctx, op_pc, dst, dst_bank, w_dict, slot, stored, false, true,
        );
    }

    if !matches!(
        helper,
        majit_ir::RuntimeHelperKind::LoadImportLocals
            | majit_ir::RuntimeHelperKind::LoadImportGlobals
    ) {
        return Ok(false);
    }

    // Null `debugdata` is `pyframe.py get_w_globals`'s promote arm. Decline
    // before the nullity guard so that guard is not left behind when the
    // caller walks the generated body. `LoadImportLocals` still guards.
    if helper == majit_ir::RuntimeHelperKind::LoadImportGlobals {
        let Some((_, majit_ir::Value::Ref(debugdata_ref))) = ctx
            .trace_ctx
            .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
        else {
            return Ok(false);
        };
        if debugdata_ref == majit_ir::GcRef::NO_CONCRETE || debugdata_ref.as_usize() == 0 {
            return Ok(false);
        }
    }

    // `debugdata` is a virtualizable field.  Read its shadow entry rather
    // than the heap field, which may be stale while compiled code owns the
    // frame.  The nullity guard is exactly PyPy's `d is None` branch.
    let Some((debugdata_op, majit_ir::Value::Ref(debugdata_ref))) = ctx
        .trace_ctx
        .virtualizable_entry_at(crate::virtualizable_spec::DEBUGDATA_VABLE_FIELD_INDEX)
    else {
        return Ok(false);
    };
    if debugdata_ref == majit_ir::GcRef::NO_CONCRETE {
        return Ok(false);
    }
    let debugdata_present = debugdata_ref.as_usize() != 0;
    walker_guard_stamped_presence(ctx, op_pc, debugdata_op, debugdata_present)?;
    if debugdata_op.is_constant()
        && ctx.trace_ctx.const_value(debugdata_op) != Some(debugdata_ref.as_usize() as i64)
    {
        return Ok(false);
    }

    let result = match helper {
        majit_ir::RuntimeHelperKind::LoadImportLocals => {
            if !debugdata_present {
                ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
            } else {
                let shadow =
                    debugdata_ref.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
                let w_locals = unsafe { (*shadow).w_locals };
                let live = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    debugdata_op,
                    crate::descr::frame_debug_data_w_locals_descr(),
                );
                if w_locals.is_null() {
                    walker_guard_stamped_presence(ctx, op_pc, live, false)?;
                    if live.is_constant() && ctx.trace_ctx.const_value(live) != Some(0) {
                        return Ok(false);
                    }
                    ctx.trace_ctx.const_ref(pyre_object::w_none() as i64)
                } else {
                    if live.is_constant()
                        && ctx.trace_ctx.const_value(live) != Some(w_locals as i64)
                    {
                        return Ok(false);
                    }
                    ctx.trace_ctx.set_opref_concrete(
                        live,
                        majit_ir::Value::Ref(majit_ir::GcRef(w_locals as usize)),
                    );
                    live
                }
            }
        }
        majit_ir::RuntimeHelperKind::LoadImportGlobals => {
            debug_assert!(debugdata_present);
            let shadow =
                debugdata_ref.as_usize() as *const pyre_interpreter::pyframe::FrameDebugData;
            let w_globals = unsafe { (*shadow).w_globals };
            if w_globals.is_null() {
                return Ok(false);
            }
            let live = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                debugdata_op,
                crate::descr::frame_debug_data_w_globals_descr(),
            );
            if live.is_constant() && ctx.trace_ctx.const_value(live) != Some(w_globals as i64) {
                return Ok(false);
            }
            ctx.trace_ctx.set_opref_concrete(
                live,
                majit_ir::Value::Ref(majit_ir::GcRef(w_globals as usize)),
            );
            live
        }
        _ => unreachable!("helper was filtered above"),
    };

    write_residual_call_result_to_dst(ctx, op_pc, dst, dst_bank, result)?;
    ctx.clear_last_exc_value();
    Ok(true)
}

/// Builtins-fallback half of the LOAD_GLOBAL and module-scope LOAD_NAME cell
/// folds: the name resolves through the frame's builtin module rather than the
/// module dict.  Mirrors `_load_global`'s second leg,
/// `get_builtin().getdictvalue(varname)`, and is reached only once
/// [`emit_module_dict_cell_fold`] has declined.
///
/// Two guards carry it.  (a) The name must stay ABSENT from the module dict,
/// which pinning that dict's `version?` is what proves: the insert that would
/// shadow the builtin runs `mutated()` and fails GUARD_NOT_INVALIDATED.
/// (b) The builtins value itself folds through [`emit_namespace_cell_fold`],
/// whose `QUASIIMMUT_FIELD` on the builtins dict fails the loop on a rebind or
/// delete there.
///
/// Returns `Ok(false)` — the caller then keeps the live residual — for a name
/// still present in the module dict, a missing or non-module builtin, or an
/// unfoldable builtins slot (absent / null / `IntMutableCell`).
fn emit_builtins_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    w_globals: pyre_object::PyObjectRef,
    w_builtin: pyre_object::PyObjectRef,
    name: &str,
) -> Result<bool, DispatchError> {
    // `emit_module_dict_cell_fold` returns `false` for BOTH an absent name and
    // a present-but-unfoldable one (`IntMutableCell` / strategy switched).
    // Only an ABSENT name may fall through to the builtins fold — a
    // present global shadows the builtin, so keep the residual (which reads the
    // live globals slot) when the slot still exists.
    if crate::state::module_dict_cell_slot_direct(w_globals, name).is_some() {
        return Ok(false);
    }
    if w_builtin.is_null() || !unsafe { pyre_object::is_module(w_builtin) } {
        return Ok(false);
    }
    let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
    if w_builtin_dict.is_null() {
        return Ok(false);
    }
    let Some(b_slot) = crate::state::module_dict_cell_slot_direct(w_builtin_dict, name) else {
        return Ok(false);
    };
    let Some(b_stored) = crate::state::module_dict_cell_value_direct(w_builtin_dict, b_slot) else {
        return Ok(false);
    };
    if b_stored.is_null() || unsafe { pyre_object::celldict::is_int_mutable_cell(b_stored) } {
        return Ok(false);
    }
    // Guard (a): the name must stay ABSENT from the module dict so the lookup
    // keeps falling through to builtins.  `celldict.py getdictvalue_no_unwrapping`
    // reads `version?` on every lookup.  A new-key insert (`_setitem_str_cell_known`)
    // or `delitem` calls `mutated()` and fails the quasi-immut guard.  A watcher
    // already installed makes `pyjitpl.py opimpl_jit_force_quasi_immutable`
    // abort the trace.  The same field a present-name fold on this namespace
    // pins, so the two share one marker.
    if !guard_current_frame_globals_identity(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
    if !walker_pin_namespace_version(ctx, op_pc, w_globals)? {
        return Ok(false);
    }
    // Guard (b): the builtins value for `name` must be unchanged.  The
    // `emit_namespace_cell_fold` below records a `QUASIIMMUT_FIELD` on the
    // builtins dict + the elidable cell lookup, so a rebind/del of the
    // builtin bumps the builtins-dict `version` and fails the loop.  The
    // baked `ConstPtr`s that fold emits are forwarded; movability does not
    // decide it.
    if !emit_namespace_cell_fold(
        ctx,
        op_pc,
        dst,
        dst_bank,
        w_builtin_dict,
        b_slot,
        b_stored,
        false,
        true,
    )? {
        return Ok(false);
    }
    Ok(true)
}

/// Resolve a `w_code` wrapper pointer to its live `CodeObject`.
fn code_from_w_code_ptr(w_code_ptr: usize) -> Option<&'static pyre_interpreter::CodeObject> {
    if w_code_ptr == 0 {
        return None;
    }
    let code_ptr =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef) };
    if code_ptr.is_null() {
        return None;
    }
    Some(unsafe { &*(code_ptr as *const pyre_interpreter::CodeObject) })
}

/// Yield each `DELETE_NAME` / `DELETE_GLOBAL` `co_names` index in `code`.
fn code_delete_name_indices(
    code: &pyre_interpreter::CodeObject,
) -> impl Iterator<Item = usize> + '_ {
    (0..code.instructions.len()).filter_map(|pc| {
        let (ins, arg) = pyre_interpreter::decode_instruction_at(code, pc)?;
        let namei = match ins {
            pyre_interpreter::Instruction::DeleteName { namei }
            | pyre_interpreter::Instruction::DeleteGlobal { namei } => namei,
            _ => return None,
        };
        Some(namei.get(arg) as usize)
    })
}

/// `except as` compiles to `STORE_NAME` + `DELETE_NAME` of that name
/// (`pyopcode.py DELETE_NAME`).  Baking its cell would read the detached
/// object after the delete and the next store allocated a replacement.
fn code_deletes_name(code: &pyre_interpreter::CodeObject, name: &str) -> bool {
    code_delete_name_indices(code)
        .any(|idx| pyre_interpreter::pyframe::load_name_from_code(code, idx) == Some(name))
}

fn code_deletes_name_from_ptr(w_code_ptr: usize, name: &str) -> bool {
    code_from_w_code_ptr(w_code_ptr).is_some_and(|code| code_deletes_name(code, name))
}

fn frame_code_deletes_name(frame: &pyre_interpreter::pyframe::PyFrame, name: &str) -> bool {
    code_deletes_name_from_ptr(frame.pycode as usize, name)
}

/// LoadName cell fold — module-scope LOAD_NAME mirror of
/// [`try_walker_load_global_cell_fold`].  At module scope the frame's
/// `w_locals` is null and `w_locals` aliases `w_globals`
/// (`createframe` sets `debugdata.w_locals = w_globals_storage`,
/// pyframe.rs), so `load_name_value`'s probe + LOAD_GLOBAL fallthrough
/// both resolve in `w_globals` — the same dict the global cell fold reads.
/// A non-module frame (class body / `exec(code, g, l)` with separate locals)
/// has a non-null `w_locals`, so the gate routes it to the live
/// residual `bh_load_name_fn`.
///
/// Builtins fallback: when `name` is absent from the module dict, module-scope
/// `LOAD_NAME` falls through via `load_global_value` to
/// `frame.get_builtin().getdictvalue(name)`.  The builtins cell fold pins the
/// module dict `version?` so a later global insertion that shadows the builtin
/// fails GUARD_NOT_INVALIDATED, then folds the builtins dict cell like the
/// LOAD_GLOBAL fallback.
pub(crate) fn try_walker_load_name_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    dst: usize,
    dst_bank: char,
    frame_ptr: usize,
    w_name_ptr: usize,
) -> Result<bool, DispatchError> {
    if frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return Ok(false);
    }
    // Only module scope (w_locals IS w_globals) is foldable. Module frames bind
    // `w_locals = w_globals` (pyframe.py); a `w_locals`
    // that is a DIFFERENT object means the LOAD_NAME probe targets a separate
    // locals namespace the module-dict cell fold (keyed on `w_globals`) would
    // skip. (Class bodies / `exec(code, g, l)` set a separate one; they also do
    // not portal-trace, so the only LOAD_NAME the walker reaches in practice is
    // module-scope.)
    let w_locals = frame.get_w_locals();
    if !w_locals.is_null() && !std::ptr::eq(w_locals, w_globals) {
        return Ok(false);
    }
    let Some(name) = (unsafe {
        pyre_object::unicodeobject::w_str_get_value_opt(w_name_ptr as pyre_object::PyObjectRef)
    }) else {
        return Ok(false);
    };
    if frame_code_deletes_name(frame, name) {
        return Ok(false);
    }
    if emit_module_dict_cell_fold(ctx, op_pc, dst, dst_bank, w_globals, name)? {
        return Ok(true);
    }
    emit_builtins_cell_fold(
        ctx,
        op_pc,
        dst,
        dst_bank,
        w_globals,
        frame.get_builtin(),
        name,
    )
}

/// StoreName/StoreGlobal: descend `typeobject.py write_cell` for an
/// in-place cell.  A replacing write (new cell, version bump) stays on
/// the residual so `mutated()` still runs.
pub(crate) fn try_walker_store_name_cell_fold<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::RuntimeHelperKind,
    frame_ptr: usize,
    w_name_ptr: usize,
    value_opref: OpRef,
) -> Result<bool, DispatchError> {
    if frame_ptr == 0 {
        return Ok(false);
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return Ok(false);
    }
    // STORE_NAME writes `get_or_create_w_locals`, so only a module frame —
    // where `w_locals` aliases `w_globals` — targets this dict.  STORE_GLOBAL
    // names globals outright; its frame may have a null `w_locals`.
    if helper == majit_ir::RuntimeHelperKind::StoreName {
        let w_locals = frame.get_w_locals();
        if !std::ptr::eq(w_locals, w_globals) {
            return Ok(false);
        }
    }
    let Some(name) = (unsafe {
        pyre_object::unicodeobject::w_str_get_value_opt(w_name_ptr as pyre_object::PyObjectRef)
    }) else {
        return Ok(false);
    };
    if frame_code_deletes_name(frame, name) {
        return Ok(false);
    }
    let Some(slot) = crate::state::module_dict_cell_slot_direct(w_globals, name) else {
        return Ok(false);
    };
    let Some(stored) = crate::state::module_dict_cell_value_direct(w_globals, slot) else {
        return Ok(false);
    };
    if stored.is_null() {
        return Ok(false);
    }
    let Some(majit_ir::Value::Ref(majit_ir::GcRef(p))) = ctx.trace_ctx.box_value(value_opref)
    else {
        return Ok(false);
    };
    if p == 0 {
        return Ok(false);
    }
    let new_value = p as pyre_object::PyObjectRef;
    // In-place arms only.  An `IntMutableCell` plus a non-plain-int
    // replaces the cell and must bump `version?` (`write_cell`).
    let in_place = unsafe {
        pyre_object::celldict::is_object_mutable_cell(stored)
            || (pyre_object::celldict::is_int_mutable_cell(stored)
                && pyre_object::listobject::is_plain_int1(new_value)
                && !pyre_object::is_long(new_value))
    };
    if !in_place {
        return Ok(false);
    }
    try_walker_orthodox_write_cell(ctx, op_pc, w_globals, slot, stored, value_opref, new_value)
}
