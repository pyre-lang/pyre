//! MIFrame trace-time helpers for the full-body walker.
//!
//! Frame construction, guard-snapshot capture, register banks, and the
//! stack-slot helpers the walker calls.  The opcode-handler traits
//! (`SharedOpcodeHandler`, `LocalOpcodeHandler`) are implemented only
//! by the real interpreter frame in `pyre-interpreter`; the trace-time
//! mirror that once implemented them here is retired.

use crate::state::*;
use pyre_interpreter::locals_w;

use std::borrow::Cow;

use majit_ir::{GcRef, OpCode, OpRef, Type, Value};
use majit_metainterp::{
    CANNOT_RAISE_NO_HEAP_EFFECT_INFO, TraceAction, TraceCtx, default_effect_info,
};

use pyre_interpreter::bytecode::{CodeObject, ComparisonOperator, Instruction};

#[allow(dead_code)]
extern "C" fn trace_function_get_defaults(func: PyObjectRef) -> PyObjectRef {
    unsafe { function_get_defaults(func) }
}

#[allow(dead_code)]
extern "C" fn trace_function_get_kwdefaults(func: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_interpreter::function_get_kwdefaults(func) }
}

#[allow(dead_code)]
extern "C" fn trace_dict_lookup_jit(dict: PyObjectRef, key: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_object::w_dict_lookup(dict, key).unwrap_or(PY_NULL) }
}

/// floatobject.py `descr_pow` → `_pow(space, x, y)` parity.
///
/// `_pow` in floatobject.py takes two raw floats and returns a
/// raw float (can raise OverflowError / ValueError / ZeroDivisionError).
/// The JIT trace records this as `CALL_F(float_pow_jit, lhs, rhs)`
/// (pyjitpl.py `do_residual_call` CALL_F branch, taken because
/// `check_forces_virtual_or_virtualizable()` is False for ll_math_pow,
/// and `exc=True` because EF_CAN_RAISE), followed by `GUARD_NO_EXCEPTION`
/// via pyjitpl.py `handle_possible_exception`.
///
/// ll_math_pow (ll_math.py) is the can-raise helper (EF_CAN_RAISE),
/// NOT elidable and NOT force-virtual. Using Rust's native `x.powf(y)`
/// would drop the Python exception semantics (negative base fractional
/// exponent → ValueError, 0.0 raised to negative → ZeroDivisionError,
/// overflow → OverflowError). Using CALL_MAY_FORCE_F would be wrong
/// because the optimizer postpones that family until GUARD_NOT_FORCED
/// arrives (heap.py CALL_MAY_FORCE branch), which is the virtualizable
/// protocol — ll_math_pow does not touch virtualizables.
///
/// Extracted to module level for stable function pointer identity.
///
/// Must match `float_pow_impl` semantics in `baseobjspace.rs`: any
/// divergence would cause the JIT compiled code to produce a different
/// result from the interpreter for the same input (correctness bug).
/// ll_math.py `math_pow = llexternal('pow', [DOUBLE, DOUBLE], DOUBLE)`
/// — the raw libm pow an inline-traced `_pow` fast path residualizes as
/// `call_f(ConstClass(ccall_pow), x, y)` with an EF_CANNOT_RAISE descr
/// (no `guard_no_exception` follows).
#[allow(dead_code)] // ll_math.py math_pow
pub(crate) extern "C" fn ccall_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// `sqrt_nonneg` (ll_math.rs) — `@jit.elidable`, `oopspec
/// "math.sqrt_nonneg(x)"` (ll_math.py).  The trace records this as a
/// pure `CALL_F(sqrt_nonneg_jit, x)` (EF_ELIDABLE_CANNOT_RAISE, no trailing
/// guard) behind the domain-pinning guards emitted by
/// `try_walker_specialize_math_sqrt`: `x >= 0` (excludes the ValueError
/// direction) and `isfinite(x)` (excludes NaN/±inf).  Reached only with a
/// finite, non-negative operand, so `x.sqrt()` cannot raise and matches
/// `ll_math_sqrt`'s finite-nonneg branch bit-for-bit.
#[allow(dead_code)] // ll_math.py ll_math_sqrt
pub(crate) extern "C" fn sqrt_nonneg_jit(x: f64) -> f64 {
    x.sqrt()
}

/// RPython `ll_math_{log,cos,sin}` fast-path calls after their exceptional
/// branches have been pinned by the walker.  These are the raw libm calls the
/// translated PyPy trace carries; boxing stays outside so optimizeopt can
/// virtualize the resulting `W_FloatObject`.
#[allow(dead_code)] // ll_math.py ll_math_log
pub(crate) extern "C" fn math_log_positive_jit(x: f64) -> f64 {
    x.ln()
}

#[allow(dead_code)] // ll_math.py ll_math_cos
pub(crate) extern "C" fn math_cos_finite_jit(x: f64) -> f64 {
    x.cos()
}

#[allow(dead_code)] // ll_math.py ll_math_sin
pub(crate) extern "C" fn math_sin_finite_jit(x: f64) -> f64 {
    x.sin()
}

#[allow(dead_code)] // ll_math.py math_pow float fast path
pub(crate) extern "C" fn float_pow_jit(x: f64, y: f64) -> f64 {
    match pyre_interpreter::float_pow_raw(x, y) {
        Ok(z) => z,
        Err(mut err) => {
            // llmodel.py _store_exception parity: set JIT exception
            // state so the following GuardNoException sees it and fails,
            // propagating the raise into the meta-interpreter.
            let exc_obj = err.to_exc_object();
            #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
            majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
            #[cfg(all(feature = "dynasm", not(target_arch = "wasm32")))]
            majit_backend_dynasm::jit_exc_raise(exc_obj as i64);
            #[cfg(target_arch = "wasm32")]
            majit_backend_wasm::jit_exc_raise(exc_obj as i64);
            let _ = exc_obj; // suppress unused warning when no backend
            // Return value is discarded by GuardNoException path; use NaN
            // as a safe sentinel in case the guard is elided.
            f64::NAN
        }
    }
}
use pyre_interpreter::eval::{get_current_exception, set_current_exception};

/// Runtime helper for traced `RAISE_VARARGS`.
///
/// The trace records the Python `CALL` that constructs an exception object,
/// then `RAISE_VARARGS` itself must materialize a real JIT exception before
/// `handle_possible_exception` emits `GuardException`. Without this explicit
/// helper the compiled bridge contains only the constructor call plus
/// `GuardException`, so the guard sees no pending exception and incorrectly
/// resumes down the normal path.
#[allow(dead_code)]
pub(crate) extern "C" fn raise_exception_jit(exc_obj: i64) {
    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    majit_backend_cranelift::jit_exc_raise(exc_obj);
    #[cfg(all(feature = "dynasm", not(target_arch = "wasm32")))]
    majit_backend_dynasm::jit_exc_raise(exc_obj);
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::jit_exc_raise(exc_obj);
    let _ = exc_obj;
}

/// Runtime helper for traced `RAISE_VARARGS`.
///
/// Mirrors `eval.rs`'s `raise_varargs` on the compiled path:
/// normalize the exception operand, normalize/attach the optional
/// cause, and publish the final exception via `jit_exc_raise` so the
/// following `GUARD_EXCEPTION` sees it.
#[allow(dead_code)]
pub(crate) extern "C" fn normalize_raise_varargs_jit(
    _frame_ptr: *const pyre_interpreter::pyframe::PyFrame,
    exc: pyre_object::PyObjectRef,
    raw_cause: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    // pyopcode.py:704-722 — cause and exc normalization both run against
    // `self.space.getexecutioncontext()`: execution context belongs to the
    // current activation/thread, independently of the frame object.
    let frame_ctx = pyre_interpreter::call::getexecutioncontext();
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !frame_ctx.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(frame_ctx);
    }

    let cause = if raw_cause.is_null() {
        None
    } else {
        // pyopcode.py:706-707 — cause class-call must mirror the exc
        // class-call (pyopcode.py:711-713) on compiled traces. Force
        // both onto the plain interpreter path so the constructor
        // cannot re-enter the tracer.
        let result = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            normalize_raise_cause(raw_cause)
        };
        match result {
            Ok(cause) => Some(cause),
            Err(mut err) => {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                let exc = err.to_exc_object();
                raise_exception_jit(exc as i64);
                return exc;
            }
        }
    };

    let mut final_exc: pyre_object::PyObjectRef = unsafe {
        if pyre_object::is_exception(exc) {
            exc
        } else if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
            if frame_ctx.is_null() {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                let err =
                    PyError::runtime_error("raise helper missing current frame").to_exc_object();
                raise_exception_jit(err as i64);
                return err;
            }
            let result = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(exc, &[])
            };
            match result {
                Ok(obj) if pyre_object::is_exception(obj) => obj,
                Ok(obj) => pyre_interpreter::error::exception_from_call_type_error(exc, obj)
                    .to_exc_object(),
                Err(mut err) => err.to_exc_object(),
            }
        } else {
            PyError::type_error("exceptions must derive from BaseException").to_exc_object()
        }
    };

    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);

    if let Err(mut err) = attach_raise_cause(final_exc, cause) {
        final_exc = err.to_exc_object();
    }
    raise_exception_jit(final_exc as i64);
    final_exc
}

/// Runtime helper for traced `PUSH_EXC_INFO`: read the per-thread
/// `CURRENT_EXCEPTION` slot so the compiled bridge preserves
/// `pyopcode.py` / `eval.rs`'s `push_exc_info` semantics (save the
/// previous sys_exc_info before `CURRENT_EXCEPTION` is overwritten).
#[allow(dead_code)]
pub(crate) extern "C" fn trace_get_current_exception_jit() -> PyObjectRef {
    pyre_interpreter::eval::get_current_exception()
}

/// Runtime helper for traced `PUSH_EXC_INFO` / `POP_EXCEPT`: write the
/// per-thread `CURRENT_EXCEPTION` slot so the compiled bridge preserves
/// `pyopcode.py/:778` / `eval.rs`'s `push_exc_info` / `pop_except`
/// semantics.
#[allow(dead_code)]
pub(crate) extern "C" fn trace_set_current_exception_jit(exc: PyObjectRef) {
    pyre_interpreter::eval::set_current_exception(exc);
}
use pyre_interpreter::eval::{attach_raise_cause, normalize_raise_cause};
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_interpreter::{
    PyError, call_function, decode_instruction_at, function_get_defaults, function_get_globals_obj,
    is_builtin_code, is_function, range_iter_continues,
};

use pyre_object::PyObjectRef;
use pyre_object::function::{is_method, w_method_get_func, w_method_get_self};
use pyre_object::functional::RANGE_ITER_TYPE;
use pyre_object::listobject::w_list_getitem;
use pyre_object::pyobject::{
    FLOAT_TYPE, INT_TYPE, LIST_TYPE, LONG_TYPE, PyType, TUPLE_TYPE, get_instantiate, is_float,
    is_int, is_list, is_long, is_tuple,
};
use pyre_object::specialisedtupleobject::{
    SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
};
use pyre_object::tupleobject::w_tuple_getitem;
use pyre_object::{
    PY_NULL, w_list_len, w_list_uses_float_storage, w_list_uses_int_storage,
    w_list_uses_object_storage, w_tuple_len,
};

#[allow(dead_code)]
fn trace_abort_error(reason: &'static str) -> PyError {
    PyError::internal_trace_abort(reason)
}

/// Emit `GetfieldGcR(w_class) → PtrEq(expected) → GuardTrue` so the trace
/// only stays specialised for instances whose Python-level `w_class`
/// matches `expected_typeobj`. Mirrors the `type(w) is W_IntObject` /
/// `type(w) is W_FloatObject` half of `listobject.py is_plain_int1`
/// and `specialisedtupleobject.py:176`. Without this guard a later
/// int/float subclass with the same payload layout would re-enter a
/// trace specialised for the exact payload type and silently lose
/// subclass identity when the trace rewraps via `wrapint` / `wrapfloat`.
///
/// Only `Type::Ref` items can carry a divergent `w_class`: a raw
/// `Type::Int` / `Type::Float` trace value is an unboxed payload produced
/// by arithmetic or a guarded unbox, and its concrete shadow can only be
/// `Int` / `Float` (the `write_int_reg` / `write_ref_reg` sanitizers
/// collapse a boxed subclass to `Null`), never a subclass pointer. Reading
/// `w_class` off such a value would force the box that OptVirtualize is
/// meant to remove, so skip the guard there.
#[allow(dead_code)]
fn trace_guard_exact_w_class(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    expected_typeobj: PyObjectRef,
) {
    if expected_typeobj.is_null() || frame.value_type(obj) != Type::Ref {
        return;
    }
    if ctx.heap_cache().is_unescaped(obj) {
        return;
    }
    let descr = crate::descr::w_class_descr();
    let actual = crate::state::opimpl_getfield_gc_r(ctx, obj, descr);
    let expected = ctx.const_ref(expected_typeobj as i64);
    frame.generate_guard(ctx, OpCode::GuardValue, &[actual, expected]);
    // `pyjitpl.py` `MIFrame.implement_guard_value` parity.
    ctx.heap_cache_mut().replace_box(actual, expected);
}

#[allow(dead_code)]
fn positional_defaults_to_load(
    callable: PyObjectRef,
    code: &CodeObject,
    nargs: usize,
) -> Option<Vec<PyObjectRef>> {
    let nparams = code.arg_count as usize;
    if nargs >= nparams {
        return None;
    }

    let defaults = unsafe { function_get_defaults(callable) };
    if defaults.is_null() {
        return None;
    }

    let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
        unsafe { w_tuple_len(defaults) }
    } else {
        0
    };
    if ndefaults == 0 {
        return None;
    }

    let first_default = nparams.saturating_sub(ndefaults);
    if nargs < first_default {
        return None;
    }

    let defaults_to_load = nparams - first_default;
    let default_start = ndefaults - defaults_to_load;
    let mut loaded = Vec::with_capacity(nparams - nargs);
    for i in nargs..nparams {
        let default_idx = default_start + (i - first_default);
        loaded.push(unsafe { w_tuple_getitem(defaults, default_idx as i64) }.unwrap_or(PY_NULL));
    }
    Some(loaded)
}

#[allow(dead_code)]
fn fill_positional_defaults_for_trace_call<'a>(
    callable: PyObjectRef,
    code: &CodeObject,
    args: &'a [PyObjectRef],
) -> Cow<'a, [PyObjectRef]> {
    let Some(defaults) = positional_defaults_to_load(callable, code, args.len()) else {
        return Cow::Borrowed(args);
    };
    let mut full = Vec::with_capacity(args.len() + defaults.len());
    full.extend_from_slice(args);
    full.extend(defaults);
    Cow::Owned(full)
}

#[allow(dead_code)]
fn const_step_one_slice_bounds(
    concrete_obj: PyObjectRef,
    concrete_key: PyObjectRef,
    concrete_value: PyObjectRef,
) -> Option<(i64, i64, i64, i64, bool)> {
    unsafe {
        if concrete_obj.is_null()
            || concrete_key.is_null()
            || concrete_value.is_null()
            || !is_list(concrete_obj)
            || !pyre_object::sliceobject::is_slice(concrete_key)
            || !is_list(concrete_value)
        {
            return None;
        }
        let step = pyre_object::sliceobject::w_slice_get_step(concrete_key);
        let step_is_none = pyre_object::is_none(step);
        let step_is_one = if step_is_none {
            true
        } else if is_int(step) {
            pyre_object::w_int_get_value(step) == 1
        } else {
            false
        };
        if !step_is_one {
            return None;
        }
        let start = pyre_object::sliceobject::w_slice_get_start(concrete_key);
        let stop = pyre_object::sliceobject::w_slice_get_stop(concrete_key);
        if pyre_object::is_none(start)
            || pyre_object::is_none(stop)
            || !is_int(start)
            || !is_int(stop)
        {
            return None;
        }
        let start = pyre_object::w_int_get_value(start);
        let stop = pyre_object::w_int_get_value(stop);
        if start < 0 || stop < start {
            return None;
        }
        let len = w_list_len(concrete_obj) as i64;
        // PyPy `W_ListObject.descr_setitem` routes slices through
        // `_unpack_slice` before `setslice`, so storage-level code sees
        // adjusted positive-step bounds, not the raw slice fields.  This
        // helper only accepts non-negative constant bounds, so adjustment
        // reduces to CPython/PyPy's upper clamp.
        Some((start, stop, start.min(len), stop.min(len), step_is_none))
    }
}

#[allow(dead_code)]
fn concrete_list_strategy_id(concrete: PyObjectRef) -> Option<i64> {
    unsafe {
        if w_list_uses_object_storage(concrete) {
            Some(0)
        } else if w_list_uses_int_storage(concrete) {
            Some(1)
        } else if w_list_uses_float_storage(concrete) {
            Some(2)
        } else {
            None
        }
    }
}

use crate::descr::{
    float_floatval_descr, int_intval_descr, list_strategy_descr, slice_w_start_descr,
    slice_w_step_descr, slice_w_stop_descr, w_float_size_descr, w_int_size_descr,
};
use crate::frame_layout::PYFRAME_PYCODE_OFFSET;

/// pyjitpl.py `_opimpl_setfield_vable` parity helper.
///
/// `PyreSym.vable_*` is a pyre-only parallel symbolic cache that
/// RPython does not have — RPython's `metainterp.virtualizable_boxes`
/// is the single canonical source for vable static state.  Setfield-
/// vable opcodes (`_opimpl_setfield_vable`) and the JUMP-time
/// shadow flush (`flush_to_frame`) publish into the boxes shadow and
/// synchronize the static back to the live frame, so future readers
/// (JUMP-arg dedup, `close_loop_args_at`) observe the same identity as
/// `s.vable_*`.  Callers gate on
/// `s.owns_virtualizable_shadow()` before calling — upstream
/// `_opimpl_setfield_vable` short-circuits on
/// `_nonstandard_virtualizable` so callee inline frames never reach
/// the `metainterp.virtualizable_boxes[index] = valuebox` write, and
/// the call-site gate is pyre's analog (callee inline syms allocated
/// via `PyreSym::new_uninit` keep `vable_array_base` /
/// `bridge_local_oprefs` `None` so `owns_virtualizable_shadow()`
/// returns false for them).
///
/// Snapshot capture (`flush_to_frame_for_guard`) intentionally does
/// NOT mirror into the shared `ctx.virtualizable_boxes`.  Upstream
/// `rpython/jit/metainterp/pyjitpl.py capture_resumedata` reads
/// `metainterp.virtualizable_boxes` and hands it to
/// `rpython/jit/metainterp/opencoder.py
/// _list_of_boxes_virtualizable(boxes)` with no fallback heap source.
/// Pyre matches that single-source model in spirit for the two
/// per-opcode-advancing fields: `last_instr` / `valuestackdepth` are
/// rewritten in `s.vable_*` to their pre-opcode value at `orgpc - 1`
/// before the snapshot is built. The other three scalars (`pycode`,
/// `debugdata`, `w_globals`) keep whatever OpRef
/// `init_vable_indices` seeded at trace start because the tracer
/// never reaches their mutators under pyre's 3.14 bytecode
/// (`pycode` / `w_globals`: only `pyframe.rs::frame_reinit`;
/// `debugdata`: only `getorcreate_debug_data` on debug paths).
/// Convergence to
/// RPython's pure single-source model requires emitting
/// `_opimpl_setfield_vable` for those handlers if/when they are
/// re-introduced, after which the heap remains authoritative through
/// `metainterp.virtualizable_boxes` rather than through the snapshot
/// reader's own state.
///
/// The snapshot does not mirror its `s.vable_*` overrides into the
/// shared shadow because the shared shadow is the JUMP/JIT-time view
/// (live virtualizable values that `close_loop_args_at`'s JUMP-arg
/// derivation consumes), while `s.vable_last_instr/vsd`
/// carry the pre-opcode override that the snapshot reader needs.
/// The two stores stay distinct deliberately: branch-guard recording
/// saves `s.vable_last_instr/vsd` before flushing and restores them
/// after the snapshot is built, but the shared shadow has no
/// symmetric save/restore — mirroring the override there would leak
/// the pre-opcode value into the JUMP path.
///
/// `static_field_name` matches the canonical PyFrame virtualizable
/// spec at `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`
/// (`last_instr`, `pycode`, `valuestackdepth`, `debugdata`, `w_globals`).
///
/// No-op when `virtualizable_boxes` is not seeded (non-virtualizable
/// trace, or before `init_virtualizable_boxes`). When a slot is mirrored,
/// pair the write with `synchronize_virtualizable()`, matching
/// `TraceCtx::vable_setfield` / `_opimpl_setfield_vable`, so the
/// live virtualizable cannot lag behind `virtualizable_boxes`.
pub(crate) fn mirror_vable_static_to_boxes(
    ctx: &mut TraceCtx,
    static_field_name: &str,
    opref: OpRef,
    concrete: Value,
) {
    if !ctx.has_virtualizable_shadow() {
        return;
    }
    let idx = ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_index_by_name(static_field_name));
    if let Some(idx) = idx {
        ctx.set_virtualizable_entry_at(idx, opref, concrete);
        // `_opimpl_setfield_vable` writes only the slot it stored
        // (`synchronize_virtualizable_at`). A full `synchronize_virtualizable`
        // flushes every array item from `box_value` copies after `const_int`
        // may have collected.
        ctx.synchronize_virtualizable_at(idx);
    }
}

/// Snapshot-only write of a STANDARD static: update the boxes the snapshot
/// copies, and do not write the live `PyFrame`. `pyjitpl.py`
/// `capture_resumedata` overrides `frame.pc` for the snapshot and restores
/// it; the heap `pushvalue`/`popvalue` depth stays the walk's.
pub(crate) fn mirror_vable_static_to_boxes_nosync(
    ctx: &mut TraceCtx,
    static_field_name: &str,
    opref: OpRef,
    concrete: Value,
) {
    if !ctx.has_virtualizable_shadow() {
        return;
    }
    let idx = ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_index_by_name(static_field_name));
    if let Some(idx) = idx {
        ctx.set_virtualizable_entry_at(idx, opref, concrete);
    }
}

/// `pyframe.py popvalue_maybe_none` leaves `locals_cells_stack_w[i] is None`
/// for every `i >= valuestackdepth`. `fill_virtualizable_boxes_to_declared_layout`
/// only fills `OpRef::NONE` holes; an occupied leftover stays in
/// `virtualizable_boxes` and `reached_loop_header` concatenates it.
///
/// `fill()` must not overwrite occupied slots: a stale snapshot vsd
/// below `nlocals` used as `dead_array_tail_from` nulled live local `i`
/// and the `range` iterator at `locals_cells_stack_w[nlocals]`. This
/// writes CONST_NULL into the dead tail using the merge-point's live
/// `valuestackdepth` so `virtualizable_data_boxes()` already holds the
/// null tail `read_boxes` reports.
fn null_vable_dead_array_tail(ctx: &mut TraceCtx, from: usize) {
    let nstatic = match ctx.virtualizable_info() {
        Some(info) => info.num_static_extra_boxes,
        None => return,
    };
    let mut data_len = 0;
    while ctx.virtualizable_box_at(data_len).is_some() {
        data_len += 1;
    }
    if data_len == 0 {
        return;
    }
    data_len -= 1;
    let start = nstatic.saturating_add(from);
    if start >= data_len {
        return;
    }
    let null = ctx.const_null();
    let concrete = Value::Ref(GcRef::NULL);
    for idx in start..data_len {
        ctx.set_virtualizable_entry_at(idx, null, concrete);
    }
}

/// Live STANDARD `last_instr` / `valuestackdepth` boxes saved across a
/// resume-image publish.
///
/// `pyjitpl.py` `capture_resumedata` saves `frame.pc` before rewriting it for
/// the snapshot and restores it afterwards. The walker publishes those
/// scalars into `virtualizable_boxes` because `build_snapshot_vable_vref_boxes`
/// reads that shadow. Restore is boxes-only (`nosync`): the heap copy stays
/// whatever `mirror_vable_static_to_boxes` / `_opimpl_setfield_vable`
/// `synchronize_virtualizable` last wrote.
#[derive(Clone, Copy, Default)]
pub(crate) struct SavedVableResumeScalars {
    last_instr: Option<(OpRef, Value)>,
    valuestackdepth: Option<(OpRef, Value)>,
}

fn vable_static_entry(ctx: &TraceCtx, static_field_name: &str) -> Option<(OpRef, Value)> {
    let idx = ctx
        .virtualizable_info()?
        .static_field_index_by_name(static_field_name)?;
    ctx.virtualizable_entry_at(idx)
}

pub(crate) fn save_vable_resume_scalars(ctx: &TraceCtx) -> SavedVableResumeScalars {
    SavedVableResumeScalars {
        last_instr: vable_static_entry(ctx, "last_instr"),
        valuestackdepth: vable_static_entry(ctx, "valuestackdepth"),
    }
}

pub(crate) fn restore_vable_resume_scalars(ctx: &mut TraceCtx, saved: SavedVableResumeScalars) {
    if let Some((opref, concrete)) = saved.last_instr {
        mirror_vable_static_to_boxes_nosync(ctx, "last_instr", opref, concrete);
    }
    if let Some((opref, concrete)) = saved.valuestackdepth {
        mirror_vable_static_to_boxes_nosync(ctx, "valuestackdepth", opref, concrete);
    }
}

/// Resolve the mutable frame-mirror index for a stack slot.
///
/// RPython `pyjitpl.py` keeps each kind-specific register bank indexed by
/// post-regalloc register number. Pyre still uses `registers_r` as a
/// semantic mirror for `locals_cells_stack_w`, so stack writers must not use
/// post-regalloc colors here: stack colors can legally coalesce with dead
/// local colors and would overwrite the local mirror before loop-close and
/// guard snapshots consume it. The encoder builds the color-indexed Ref bank
/// separately from liveness and the virtualizable shadow.
pub(crate) fn stack_slot_reg_idx(sym: &PyreSym, stack_idx: usize) -> usize {
    sym.nlocals + stack_idx
}

/// Write a Ref-boxed value to the symbolic operand stack at depth
/// offset `stack_idx`. Centralizes the dual-shadow update that
/// `push_typed_value`, `finishframe_exception`'s exception/lasti push,
/// the `caller_result_stack_idx` writeback (pyjitpl.rs) and
/// inline-call setup all duplicated:
///
/// - `registers_r[reg_idx]` — the semantic frame mirror slot
///   (`reg_idx == nlocals + stack_idx`).
/// - `virtualizable_boxes[NUM_VABLE_SCALARS + semantic_idx]` —
///   `locals_cells_stack_w` heap mirror, ALWAYS semantic-indexed
///   (`pyjitpl.py _opimpl_setarrayitem_vable`).
/// - `symbolic_stack_types[stack_idx]` set to `Type::Ref` (every slot
///   of `locals_cells_stack_w` is W_Root per
///   `virtualizable.py read_boxes`).
/// - `concrete_stack[stack_idx]` set to `concrete` for Box-identity
///   tracking.
///
/// Caller is responsible for:
/// - Wrapping Int/Float values via `wrapint` / `wrapfloat` BEFORE
///   calling so `boxed` is always Ref-typed.
/// - Advancing `valuestackdepth` (push) or leaving it (positional
///   write into an existing slot).
/// - Emitting the separate `_opimpl_setfield_vable_i(vsd, depth±1)`
///   IR op via `mirror_vable_static_to_boxes` when the operation
///   logically advances the frame's vsd field (push / pop).
#[allow(dead_code)]
pub(crate) fn write_stack_slot(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    stack_idx: usize,
    boxed: OpRef,
    concrete: ConcreteValue,
) {
    let semantic_idx = sym.nlocals + stack_idx;
    let reg_idx = stack_slot_reg_idx(sym, stack_idx);
    // Portal frames carry the authoritative
    // stack shadow on `virtualizable_boxes` (`pyjitpl.py:1242
    // _opimpl_setarrayitem_vable`). The companion read paths
    // (`read_stack_slot`, `read_live`, and the
    // `get_list_of_active_boxes` snapshot fallback) source
    // their portal-frame view from the vable shadow, so the pyre-only
    // `registers_r[reg_idx]` semantic-mirror write is dead for portal
    // frames.  Non-portal frames retain the lazy-fill mirror because
    // their read path still consults `registers_r[reg_idx]`.
    if !sym.owns_virtualizable_shadow() {
        if reg_idx >= sym.registers_r.len() {
            sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
        }
        sym.registers_r.set(reg_idx, boxed);
    }
    if stack_idx >= sym.symbolic_stack_types.len() {
        sym.symbolic_stack_types.resize(stack_idx + 1, Type::Ref);
    }
    sym.symbolic_stack_types[stack_idx] = Type::Ref;
    if stack_idx >= sym.concrete_stack.len() {
        sym.concrete_stack
            .resize(stack_idx + 1, ConcreteValue::Null);
    }
    sym.concrete_stack[stack_idx] = concrete;
    if sym.owns_virtualizable_shadow() {
        let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_idx;
        // A correct trace never pushes beyond the frame's `co_stacksize`, so
        // `flat_idx` stays within the virtualizable shadow. A multi-frame
        // bridge resume whose inlined-callee return accounting is incomplete
        // can leak an operand-stack slot per loop iteration (the unrolled trace
        // re-pushes without the matching pop), driving `flat_idx` past the
        // shadow. Rather than panic in `set_virtualizable_entry_at`, request a
        // graceful trace abort: the trace is discarded before any code is
        // installed, so the guard resolves through the interpreter instead of
        // crashing the process (mirrors the cross-frame snapshot-gap abort).
        //
        // The bound is `len - 1`, not `len`: the last shadow slot is the
        // virtualizable identity (`virtualizable_boxes[-1]`), the box
        // `_nonstandard_virtualizable` compares every later vable op against.
        // One slot past `co_stacksize` lands exactly there, and a store that
        // reaches it renames the standard virtualizable rather than corrupting
        // one value — the compiled trace then writes frame fields into
        // whatever object the push carried.
        if ctx
            .virtualizable_boxes_len()
            .is_some_and(|len| flat_idx + 1 >= len)
        {
            crate::state::request_trace_abort();
            return;
        }
        // `_opimpl_setarrayitem_vable`: a Ref/Null concrete carries a real
        // W_Root heap pointer; store the box and stamp it. Int/Float concrete
        // means lazy wrapint/wrapfloat emitted a NewWithVtable OpRef without
        // allocating yet — store only the OpRef so synchronize_virtualizable
        // keeps writing the existing W_Root.
        if ctx.has_virtualizable_boxes() {
            match concrete.to_ir_ref_value() {
                Some(v) => {
                    ctx.set_virtualizable_entry_at(flat_idx, boxed, v);
                }
                _ => {
                    ctx.set_virtualizable_box_at(flat_idx, boxed);
                }
            }
        }
    }
}

/// Read the symbolic OpRef at depth offset `stack_idx`, with lazy
/// heap-fill from `locals_cells_stack_w` when the slot is empty.
/// Symmetric counterpart of `write_stack_slot`.
///
/// Reads the semantic frame mirror via `stack_slot_reg_idx`.  On
/// NONE-fill, the IR `getarrayitem` op still uses the SEMANTIC array
/// index (`locals_cells_stack_w[nlocals + stack_idx]`) — the heap layout
/// the array descr describes — and stores the result in the mirror slot
/// subsequent stack reads consult.
///
/// `init_symbolic` (state.rs) leaves
/// `locals_cells_stack_array_ref = OpRef::NONE` for active-owner
/// traces because their locals come from `OpRef::input_arg_ref` and
/// stack writes route through the vable shadow, so the lazy-fill
/// path is normally unused.  In the rare case it does fire (e.g.
/// `pop_value` / `swap_stack_slots` reading a stack slot whose
/// `registers_r` entry was never written), emit `GETFIELD_GC_R` for
/// the array base on demand and cache it on the sym so subsequent
/// fills reuse the same GCREF.  Without this guard,
/// `trace_array_getitem_value(NONE, idx)` would record a malformed
/// `GetarrayitemGcR` with a NONE base operand.
#[allow(dead_code)]
pub(crate) fn read_stack_slot(sym: &mut PyreSym, ctx: &mut TraceCtx, stack_idx: usize) -> OpRef {
    let semantic_idx = sym.nlocals + stack_idx;
    // For portal frames, read the stack slot
    // directly from the `virtualizable_boxes` shadow — PyPy-orthodox
    // (`pyjitpl.py _opimpl_getarrayitem_vable`).  Empirical verification
    // (`PYRE_PATH3_VERIFY_STACK_READ`) showed zero mismatch between
    // the vable shadow and the legacy `registers_r[reg_idx]` semantic-mirror
    // value across 9 benches.  Routing through vable retires one dependency
    // on the `registers_r` semantic-mirror deviation.
    //
    // Non-portal frames keep the `registers_r` lazy-fill path below — they
    // don't own a vable shadow.  Their semantic-mirror is not yet retired.
    if sym.owns_virtualizable_shadow() {
        let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
        if let Some(v) = ctx.virtualizable_box_at(nvs + semantic_idx) {
            return v;
        }
    }
    let reg_idx = stack_slot_reg_idx(sym, stack_idx);
    if reg_idx >= sym.registers_r.len() {
        sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
    }
    if sym.registers_r.get(reg_idx).expect("bound Ref register") == OpRef::NONE {
        if sym.locals_cells_stack_array_ref == OpRef::NONE {
            sym.locals_cells_stack_array_ref = frame_locals_cells_stack_array(ctx, sym.frame);
        }
        let idx_const = ctx.const_int(semantic_idx as i64);
        sym.registers_r.set(
            reg_idx,
            trace_array_getitem_value(ctx, sym.locals_cells_stack_array_ref, idx_const),
        );
    }
    sym.registers_r.get(reg_idx).expect("bound Ref register")
}

/// Swap two operand-stack slots — third member of the
/// `read_stack_slot` / `write_stack_slot` family. Pre-fills both
/// slots through `read_stack_slot`, swaps the registers_r entries,
/// `symbolic_stack_types`, `concrete_stack`, and the vable shadow's
/// `(OpRef, Value)` pairs atomically.
///
/// `virtualizable_boxes` is the single source of truth for the frame's
/// Ref array (opencoder.py _list_of_boxes_virtualizable); reading each half via
/// `concrete_of_opref` separately would drop non-const Box identity
/// into the sentinel fallback, hence the `virtualizable_entry_at`
/// pair-read+pair-write.
///
/// `reg_top` / `reg_other` are semantic frame-mirror indices.
#[allow(dead_code)]
pub(crate) fn swap_stack_slots(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    top_idx: usize,
    other_idx: usize,
) {
    let _ = read_stack_slot(sym, ctx, top_idx);
    let _ = read_stack_slot(sym, ctx, other_idx);
    let semantic_top = sym.nlocals + top_idx;
    let semantic_other = sym.nlocals + other_idx;
    let reg_top = stack_slot_reg_idx(sym, top_idx);
    let reg_other = stack_slot_reg_idx(sym, other_idx);
    if reg_top != reg_other {
        sym.registers_r.swap(reg_top, reg_other);
    }
    if top_idx < sym.symbolic_stack_types.len() && other_idx < sym.symbolic_stack_types.len() {
        sym.symbolic_stack_types.swap(top_idx, other_idx);
    }
    if top_idx < sym.concrete_stack.len() && other_idx < sym.concrete_stack.len() {
        sym.concrete_stack.swap(top_idx, other_idx);
    }
    if sym.owns_virtualizable_shadow() {
        let flat_top = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_top;
        let flat_other = crate::virtualizable_gen::NUM_VABLE_SCALARS + semantic_other;
        // Same `len - 1` bound `write_stack_slot` takes, and for the same
        // reason: the trailing shadow slot is the virtualizable identity, so a
        // swap that reaches it would hand the standard-virtualizable box to a
        // stack slot and a stack value to `virtualizable_boxes[-1]`.
        if ctx
            .virtualizable_boxes_len()
            .is_some_and(|len| flat_top + 1 >= len || flat_other + 1 >= len)
        {
            crate::state::request_trace_abort();
            return;
        }
        if let (Some((op_top, val_top)), Some((op_other, val_other))) = (
            ctx.virtualizable_entry_at(flat_top),
            ctx.virtualizable_entry_at(flat_other),
        ) {
            // Swap the boxes. A box keeps only its own result; an unstamped
            // slot's current concrete lives on the virtualizable, so write
            // the swapped values there (`write_box_at` over the slot, not
            // `_make_op` on the box).
            ctx.set_virtualizable_box_at(flat_top, op_other);
            ctx.set_virtualizable_box_at(flat_other, op_top);
            ctx.write_virtualizable_heap_value_at(flat_top, val_other);
            ctx.write_virtualizable_heap_value_at(flat_other, val_top);
        } else if let (Some(op_top), Some(op_other)) = (
            ctx.virtualizable_box_at(flat_top),
            ctx.virtualizable_box_at(flat_other),
        ) {
            ctx.set_virtualizable_box_at(flat_top, op_other);
            ctx.set_virtualizable_box_at(flat_other, op_top);
        } else {
            panic!(
                "swap_stack_slots: missing virtualizable_boxes entries for stack slots {top_idx} and {other_idx}"
            );
        }
    }
}

impl MIFrame {
    pub fn from_sym(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        concrete_frame: usize,
        fallthrough_pc: usize,
        opcode_start_pc: usize,
    ) -> Self {
        // sym was initialized when its owning MetaInterpFrame was pushed
        // (trace.rs root push / pyjitpl.rs perform_call). MIFrame is a
        // borrowed per-instruction view; no re-initialization here.
        // RPython pyjitpl.py: orgpc = opcode start PC passed to each handler.
        let orgpc = opcode_start_pc;
        Self {
            ctx,
            sym,
            fallthrough_pc,
            concrete_frame_addr: concrete_frame,
            orgpc,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            loop_close_registers_r: Vec::new(),
            close_merge_point_vsd: None,
        }
    }

    #[doc(hidden)]
    pub fn capture_current_fail_args(&mut self) -> Vec<OpRef> {
        self.with_ctx(|this, ctx| this.current_fail_args(ctx))
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn set_resume_marker_for_test(&mut self, jit_pc: usize) {
        self.loop_close_marker_jit_pc = Some(jit_pc);
    }

    #[doc(hidden)]
    pub fn capture_guard_class(
        &mut self,
        obj: OpRef,
        expected_type: *const pyre_object::pyobject::PyType,
    ) {
        self.with_ctx(|this, ctx| this.guard_class(ctx, obj, expected_type));
    }

    #[doc(hidden)]
    pub fn capture_trace_guarded_int_payload(&mut self, int_obj: OpRef) -> OpRef {
        self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj))
    }

    #[doc(hidden)]
    pub fn capture_generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        self.with_ctx(|this, ctx| this.generate_guard(ctx, opcode, args));
    }

    #[doc(hidden)]
    pub fn capture_close_loop_args_at(
        &mut self,
        target_pc: Option<usize>,
        header_marker_jit_pc: Option<usize>,
    ) -> Vec<OpRef> {
        self.with_ctx(|this, ctx| {
            this.close_loop_args_at(ctx, target_pc, header_marker_jit_pc, &[])
        })
    }

    #[doc(hidden)]
    pub fn symbolic_nlocals(&self) -> usize {
        self.sym().nlocals
    }

    #[doc(hidden)]
    pub fn symbolic_valuestackdepth(&self) -> usize {
        self.sym().valuestackdepth
    }

    /// Read `PyFrame.valuestackdepth` directly from the concrete frame at
    /// `concrete_frame_addr`.  The orthodox
    /// PyPy-parity replacement for `self.sym().valuestackdepth`.
    ///
    /// RPython has no symbolic mirror of the Python stack — `MIFrame` only
    /// holds the per-jitcode-invocation register banks (`registers_r/i/f`),
    /// and the user-side stack lives in `PyFrame.locals_cells_stack_w` /
    /// `PyFrame.valuestackdepth` accessed via IR `getfield/setfield` on the
    /// virtualizable.  Pyre's `PyreSym.valuestackdepth` is a newly
    /// introduced divergence (a symbolic mirror) that drifts from
    /// `PyFrame.valuestackdepth` whenever the production walker handles an
    /// opcode that mutates the concrete stack (the walker records the
    /// residual_call but does not run `MIFrame::pop_value`'s `sym.valuestackdepth -= 1`).
    ///
    /// Returns `None` when `concrete_frame_addr == 0` (tests constructing a
    /// sym-only `MIFrame`).
    ///
    /// Production tracer paths always seed `concrete_frame_addr` from the
    /// live `PyFrame` for the top frame.
    pub(crate) fn concrete_valuestackdepth(&self) -> Option<usize> {
        // `close_loop_args_at` publishes the merge point's own depth here for
        // the duration of the JUMP-arg derivation; outside that window the
        // concrete frame is the reader.
        if let Some(depth) = self.close_merge_point_vsd {
            return Some(depth);
        }
        crate::state::concrete_stack_depth(self.concrete_frame_addr)
    }

    #[doc(hidden)]
    pub fn symbolic_registers_r(&self) -> &crate::jitcode_dispatch::RegisterList {
        &self.sym().registers_r
    }

    #[doc(hidden)]
    pub fn capture_value_type(&self, opref: OpRef) -> Type {
        self.value_type(opref)
    }

    #[allow(dead_code)]
    pub(crate) fn ctx(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    pub(crate) fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self, &mut TraceCtx) -> R) -> R {
        let ctx = self.ctx;
        unsafe { f(self, &mut *ctx) }
    }

    #[inline]
    pub(crate) fn sym(&self) -> &PyreSym {
        unsafe { &*self.sym }
    }

    #[inline]
    pub(crate) fn sym_mut(&mut self) -> &mut PyreSym {
        unsafe { &mut *self.sym }
    }

    /// `pypy/module/pypyjit/interp_jit.py reds = ['frame', 'ec']` requires
    /// every CALL_ASSEMBLER red-args list and JUMP-args list to carry ec.
    /// Trace and bridge setup seed `sym.execution_context` from the portal's
    /// second red. Adapter paths must preserve that red rather than deriving
    /// it from the virtualizable frame.
    ///
    /// A bridge resumed at a guard whose resume data named no value at the
    /// red's color arrives with none, and there is no second live source to
    /// read it back from — `ec` is a red, not a `PyFrame` field. Recover it
    /// the way the interpreter does, by asking the thread
    /// ([`crate::helpers::emit_current_execution_context`]), and keep the
    /// result so the rest of the frame reuses one read.
    pub(crate) fn ensure_execution_context(&mut self, ctx: &mut TraceCtx) -> OpRef {
        let ec = self.sym().execution_context;
        if !ec.is_none() {
            return ec;
        }
        let recovered =
            crate::helpers::emit_current_execution_context(ctx, "ExecutionContext::MIFrame");
        self.sym_mut().execution_context = recovered;
        recovered
    }

    #[inline]
    fn clear_pre_opcode_state(&mut self) {
        self.pre_opcode_registers_r = None;
        self.pre_opcode_semantic_depth = None;
    }

    /// Pre-opcode stack depth: snapshot-captured `pre_opcode_semantic_depth`
    /// when available, otherwise the concrete `PyFrame.valuestackdepth`
    /// (which holds the same pre-opcode state because the interpreter step
    /// for this opcode has not run yet); falls back to the symbolic
    /// `sym.valuestackdepth` only when `concrete_frame_addr == 0`
    /// (unit tests constructing sym-only `MIFrame`s).
    ///
    /// Replaces the `pre_opcode_depth_or(self.sym().valuestackdepth)`
    /// pattern.  The `pre_opcode_*` machinery exists precisely because
    /// pyre's `MIFrame::pop_value` / `push_typed_value` mutate
    /// `sym.valuestackdepth` mid-opcode and the guard/snapshot writers
    /// need the *pre-mutation* value.  Reading directly from PyFrame
    /// makes that pre-mutation guarantee structural rather than
    /// snapshot-bookkeeping-dependent.
    #[inline]
    fn pre_opcode_concrete_depth(&self) -> usize {
        self.pre_opcode_semantic_depth.unwrap_or_else(|| {
            self.concrete_valuestackdepth()
                .unwrap_or_else(|| self.sym().valuestackdepth)
        })
    }

    fn materialize_fail_arg_slot(
        &mut self,
        ctx: &mut TraceCtx,
        slot: OpRef,
        slot_type: Type,
        abs_idx: usize,
    ) -> OpRef {
        if !slot.is_none() {
            return slot;
        }
        let concrete_value = self.concrete_at(abs_idx).unwrap_or(PY_NULL);
        let typed_value = extract_concrete_typed_value(slot_type, concrete_value);
        fail_arg_opref_for_typed_value(ctx, typed_value)
    }

    /// `pyjitpl.py` `get_list_of_active_boxes` parity. Returns
    /// compact register boxes for live registers only.
    ///
    /// Both the tracer (here) and the blackhole bridge-resume decoder
    /// (`consume_one_section`, `resume.py`) read the same
    /// `all_liveness` byte stream via `jitcode.get_live_vars_info(pc,
    /// op_live)` (`jitcode.py`) and iterate the per-bank register
    /// indices with `LivenessIterator` (`liveness.py`). One
    /// source, same order.
    fn get_list_of_active_boxes(
        &mut self,
        ctx: &mut TraceCtx,
        in_a_call: bool,
        after_residual_call: bool,
        top_frame_marker_call_pc: Option<usize>,
    ) -> Vec<OpRef> {
        // resume.py consume_one_section invariant: every register
        // reported as live must be reachable via a valid OpRef. RPython
        // trivially satisfies this because every read populates
        // `registers_r[i]`. pyre's `registers_r` is the unified
        // abstract register file — locals occupy `[..nlocals]` and the
        // live stack tail occupies `[nlocals..nlocals+stack_only]`
        // (pyjitpl.py MIFrame parity). A live register that the
        // trace has not yet produced (forward-live local across a
        // superinstruction edge, live stack slot resurrected after a
        // guard backtrack) keeps `OpRef::NONE`, poisoning the guard's
        // fail_args. Mirror RPython's invariant by forcing lazy init
        // for every live register via the same `_opimpl_getarrayitem
        // _vable` mirror read that LOAD_FAST uses (load_local_value
        // at trace_opcode.rs), BEFORE snapshotting registers_r
        // below. Source for the live indices is the same packed
        // `all_liveness` byte stream (`jitcode.get_live_vars_info(pc,
        // op_live)` at `jitcode.py`) that resume.py uses at
        // decode time — pyjitpl.py `get_list_of_active_boxes`
        // analog, walking the full live register-file set.
        #[derive(Clone, Copy)]
        enum LiveBank {
            Int,
            Ref,
            Float,
        }
        impl LiveBank {
            #[allow(dead_code)]
            fn ty(self) -> Type {
                match self {
                    LiveBank::Int => Type::Int,
                    LiveBank::Ref => Type::Ref,
                    LiveBank::Float => Type::Float,
                }
            }
        }

        let jitcode_ptr_pre = self.sym().jitcode;
        // `pyjitpl.py:194-198`: a single `pc` drives both the result-box
        // clear and the liveness decode.  Resolve the resume jitcode pc once
        // here so the lazy-load preamble fills exactly the registers the
        // snapshot below reads, routed through the SAME `-live-` the snapshot
        // pc resolves to. A py_pc-only observer cannot represent a post-call
        // catch marker after the JitCode-keyed migration; it declines below
        // rather than publishing an ambiguous snapshot.
        let resume_jit_pc: Option<usize> = unsafe {
            let jc = &*jitcode_ptr_pre;
            if !jc.payload.is_populated() {
                None
            } else {
                let marker_call_pc = if in_a_call {
                    self.residual_call_pc
                } else if after_residual_call {
                    top_frame_marker_call_pc
                } else {
                    // Pre-call top-frame guard: resume at the plain `live_pc`
                    // `-live-`. Routing through the post-call marker here would
                    // make the box list shorter than the box count the decoder
                    // reads at the recorded (pre-call) snapshot position.
                    None
                };
                if marker_call_pc.is_some() {
                    // This retired MIFrame trait-interpret leg has no production driver.
                    // Production guard capture (`collect_outer_active_boxes`) captures rather than
                    // aborts residual-call and inline guards as pyjitpl.py/opencoder.py capture_resumedata;
                    // only trait-leg unit tests reach this abort before Phase-6 deletion.
                    crate::state::request_trace_abort();
                    return Vec::new();
                }
                // The loop-close twin is the complete
                // resume coordinate at this capture seam. A missing twin uses
                // the existing trace-abort path below rather than guessing a
                // block-head position from `live_pc`.
                match self.loop_close_marker_jit_pc {
                    Some(jit_pc) => Some(jit_pc),
                    None => {
                        // This (parent) frame reports a `live_pc` the jitcode
                        // has no resume entry for — the cross-frame snapshot
                        // coordinate gap (#124/#130): an inlined callee +
                        // exception-resume shape whose parent resume pc was
                        // never recorded.  Building the guard from this frame
                        // would emit incorrect resume data, so request a trace
                        // abort and return no active boxes.  The recorded guard
                        // is thrown away with the aborted (pre-install) trace,
                        // so the empty list — already a valid return for the
                        // skeleton / short-liveness paths below — is harmless.
                        crate::state::request_trace_abort();
                        return Vec::new();
                    }
                }
            }
        };
        let live_regs_for_banks: Vec<(LiveBank, usize)> = unsafe {
            let jc = &*jitcode_ptr_pre;
            // Skeleton payload (no resume maps yet) skips the lazy-load
            // preamble; the main path's skeleton branch handles the same case.
            if !jc.payload.is_populated() {
                Vec::new()
            } else {
                // RPython `pyjitpl.py:218-225` reads each liveness bank
                // from its matching register file. Pyre's unified semantic
                // stack can hold an OpRef before the kind bank has been
                // populated, so collect every listed bank/index and complete
                // the matching bank immediately before the direct snapshot.
                let jit_pc =
                    resume_jit_pc.expect("is_populated() branch above ensures lookup hits");
                let op_live = crate::state::op_live();
                let off = jc.payload.jitcode.get_live_vars_info(jit_pc, op_live);
                let all_liveness = crate::state::liveness_info_snapshot();
                if off + 2 >= all_liveness.len() {
                    Vec::new()
                } else {
                    let length_i = all_liveness[off] as u32;
                    let length_r = all_liveness[off + 1] as u32;
                    let length_f = all_liveness[off + 2] as u32;
                    let mut cursor = off + 3;
                    let mut out: Vec<(LiveBank, usize)> =
                        Vec::with_capacity((length_i + length_r + length_f) as usize);
                    use majit_jitcode::liveness::LivenessIterator;
                    if length_i != 0 {
                        // SAFETY: `all_liveness` bank at `cursor` is a complete
                        // `encode_liveness` record (`liveness.py` `LivenessIterator.__init__`).
                        let mut it =
                            unsafe { LivenessIterator::new(cursor, length_i, &all_liveness) };
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Int, reg_idx as usize));
                        }
                        cursor = it.offset();
                    }
                    if length_r != 0 {
                        // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
                        let mut it =
                            unsafe { LivenessIterator::new(cursor, length_r, &all_liveness) };
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Ref, reg_idx as usize));
                        }
                        cursor = it.offset();
                    }
                    if length_f != 0 {
                        // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
                        let mut it =
                            unsafe { LivenessIterator::new(cursor, length_f, &all_liveness) };
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Float, reg_idx as usize));
                        }
                    }
                    out
                }
            }
        };
        let (nlocals, valid_stack_only, jitcode_ptr, pcdep_entries, has_color_map) = {
            let s = self.sym();
            let (metadata_stack_depth, pcdep_entries, has_color_map) = if s.jitcode.is_null() {
                (None, Vec::new(), false)
            } else {
                unsafe {
                    let jc = &*s.jitcode;
                    let has_color_map = jc.payload.metadata.has_color_map;
                    match resume_jit_pc {
                        Some(jit_pc) => (
                            jc.payload
                                .depth_for_jitcode_pc_pred(jit_pc)
                                .map(|d| d as usize),
                            jc.payload.pcdep_for_jitcode_pc(jit_pc).unwrap_or_default(),
                            has_color_map,
                        ),
                        None => (None, Vec::new(), has_color_map),
                    }
                }
            };
            let valid_stack_only = if self.pre_opcode_registers_r.is_some() {
                metadata_stack_depth.unwrap_or_else(|| s.valuestackdepth.saturating_sub(s.nlocals))
            } else {
                s.valuestackdepth.saturating_sub(s.nlocals)
            };
            (
                s.nlocals,
                valid_stack_only,
                s.jitcode,
                pcdep_entries,
                has_color_map,
            )
        };
        let pcdep_opt: Option<&[(u8, u16, u16)]> =
            (!pcdep_entries.is_empty()).then(|| pcdep_entries.as_slice());
        // SSA-authoritative live_r: Ref bank entries go
        // through the read_live / lazy-fill / materialize pipeline to
        // populate registers_r[color].  Int/Float banks already live in
        // their own register arrays (registers_i / registers_f) and the
        // clone at lines 1225-1227 captures them; materialization would
        // only corrupt those values by overwriting with a Ref-derived
        // fallback.  Skip non-Ref banks entirely.
        let mut bank_materializations: Vec<(LiveBank, usize, OpRef)> =
            Vec::with_capacity(live_regs_for_banks.len());
        for (bank, idx) in live_regs_for_banks {
            if !matches!(bank, LiveBank::Ref) {
                continue;
            }
            let color_idx = idx;
            // Derive semantic index for vable shadow / concrete_at.
            // A stack slot color may be coalesced with a local identity
            // color when the local is not live.  Mirror the decoder's
            // `semantic_ref_slot_for_reg_color`: consult the live stack
            // prefix first, and only fall back through the local color
            // map if no live stack slot owns this color.
            let Some(semantic_idx) = crate::state::semantic_ref_slot_for_reg_color(
                nlocals,
                valid_stack_only,
                pcdep_opt.unwrap_or(&[]),
                color_idx,
            ) else {
                continue;
            };
            {
                let s = self.sym_mut();
                if color_idx >= s.registers_r.len() {
                    s.registers_r.resize(color_idx + 1, OpRef::NONE);
                }
            }
            // pyjitpl.py:218-234 parity for the snapshot/fallback:
            // produce the color-indexed Ref bank, but source active
            // virtualizable frame slots from the semantic shadow. A
            // stack color may be coalesced with a dead local color;
            // reading `registers_r[color]` for the fallback would pick
            // up the stale local mirror for a live stack slot.
            let live_max = nlocals + valid_stack_only;
            let read_live = |this: &MIFrame, ctx: &TraceCtx| -> OpRef {
                let s = this.sym();
                if s.owns_virtualizable_shadow() && semantic_idx < live_max {
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    return ctx
                        .virtualizable_box_at(nvs + semantic_idx)
                        .expect("get_list_of_active_boxes: missing vable frame box");
                }
                let val = s.registers_r.get(semantic_idx).unwrap_or(OpRef::NONE);
                if val != OpRef::NONE {
                    return val;
                }
                OpRef::NONE
            };
            let live_value_pre = read_live(self, ctx);
            if live_value_pre == OpRef::NONE {
                if semantic_idx < nlocals {
                    let value = MIFrame::load_local_value(self, ctx, semantic_idx)
                        .expect("get_list_of_active_boxes: failed to lazy-load live local");
                    self.sym_mut().registers_r.set(color_idx, value);
                } else {
                    // Stack lazy-fill: heap read at semantic index,
                    // store in both the semantic mirror (read_live) and
                    // the color bank (Ref-bank fail args).
                    let s = self.sym_mut();
                    if s.locals_cells_stack_array_ref == OpRef::NONE {
                        let frame_ref = s.frame;
                        s.locals_cells_stack_array_ref =
                            frame_locals_cells_stack_array(ctx, frame_ref);
                    }
                    let idx_const = ctx.const_int(semantic_idx as i64);
                    let arr = s.locals_cells_stack_array_ref;
                    let value = trace_array_getitem_value(ctx, arr, idx_const);
                    if semantic_idx >= s.registers_r.len() {
                        s.registers_r.resize(semantic_idx + 1, OpRef::NONE);
                    }
                    s.registers_r.set(semantic_idx, value);
                    s.registers_r.set(color_idx, value);
                }
            }
            let live_value = if live_value_pre == OpRef::NONE {
                read_live(self, ctx)
            } else {
                live_value_pre
            };
            let semantic_value = self
                .pre_opcode_registers_r
                .as_ref()
                // `capture_pre_opcode_state` stores a semantic frame
                // snapshot for both vable-owner and non-owner traces. A
                // live stack color may reuse a dead local color, so reading
                // the snapshot by color would capture the wrong local value.
                .and_then(|pre_r| pre_r.get(semantic_idx).copied())
                .filter(|value| !value.is_none())
                .unwrap_or(live_value);
            let bank_value =
                self.materialize_fail_arg_slot(ctx, semantic_value, Type::Ref, semantic_idx);
            bank_materializations.push((bank, idx, bank_value));
        }
        let (registers_i, registers_r_bank, _registers_r_semantic, registers_f) = {
            let s = self.sym();
            // Unified abstract register file view.
            // When a guard is being captured mid-opcode, read from
            // `pre_opcode_registers_r` (the full snapshot of
            // `registers_r` at opcode start). Otherwise read the live
            // `registers_r`. Both variants share a single indexing rule
            // so `live_{i,r,f}_regs` indices (which live in the
            // stack_base=nlocals register space) can be resolved with
            // one lookup instead of the legacy
            // `idx < nlocals ? locals : stack` split.
            //
            // Dual-writes grow `registers_r` monotonically on
            // stack pushes; pop does not shrink it. Bound the view to
            // the valid locals + live stack_only range so stale slots
            // above the current (or pre-opcode) stack depth cannot
            // surface as active OpRefs. This matches the OLD
            // `stack_values.len()` bound on the
            // `stack_values[idx - nlocals]` read path.
            //
            let source_len = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r.len()
            } else {
                s.registers_r.len()
            };
            let valid_len = (s.nlocals + valid_stack_only).min(source_len);
            let mut registers_i = s.registers_i.clone();
            let mut registers_r_bank = s.registers_r.to_vec();
            let mut registers_f = s.registers_f.clone();
            for &(bank, reg_idx, value) in &bank_materializations {
                match bank {
                    LiveBank::Int => {
                        if reg_idx >= registers_i.len() {
                            registers_i.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_i[reg_idx] = value;
                    }
                    LiveBank::Ref => {
                        if reg_idx >= registers_r_bank.len() {
                            registers_r_bank.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_r_bank[reg_idx] = value;
                    }
                    LiveBank::Float => {
                        if reg_idx >= registers_f.len() {
                            registers_f.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_f[reg_idx] = value;
                    }
                }
            }
            let mut registers_r_semantic: Vec<OpRef> =
                if let Some(ref pre_r) = self.pre_opcode_registers_r {
                    pre_r[..valid_len.min(pre_r.len())].to_vec()
                } else if s.owns_virtualizable_shadow() {
                    // Portal frames have the
                    // authoritative semantic-indexed shadow in
                    // `virtualizable_boxes` (`pyjitpl.py:1242
                    // _opimpl_setarrayitem_vable`).  When no opcode-start
                    // snapshot is available, source the encoder's
                    // semantic view directly from the vable shadow rather
                    // than the pyre-only `registers_r` semantic mirror.
                    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                    (0..valid_len)
                        .map(|idx| ctx.virtualizable_box_at(nvs + idx).unwrap_or(OpRef::NONE))
                        .collect()
                } else {
                    s.registers_r.iter().take(valid_len).collect()
                };
            if in_a_call {
                if let Some(result_idx) = self.pending_result_stack_idx {
                    let abs_idx = s.nlocals + result_idx;
                    match self.pending_result_type.unwrap_or(Type::Ref) {
                        Type::Int => {
                            if result_idx >= registers_i.len() {
                                registers_i.resize(result_idx + 1, OpRef::NONE);
                            }
                            registers_i[result_idx] = ctx.const_int(0);
                        }
                        Type::Ref => {
                            let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
                            if abs_idx < registers_r_semantic.len() {
                                registers_r_semantic[abs_idx] = null_ref;
                            }
                            // PyPy uses `_result_argcode` plus the bytecode
                            // dst register to clear the typed result bank.
                            // Pyre's Python-frame path receives a semantic
                            // stack depth, so Ref must be translated through
                            // the per-jitcode stack color map before touching
                            // the bank used by packed liveness.
                            // #73: the not-yet-produced call result is not a
                            // live Variable at the resume PC, so it carries no
                            // `pcdep_color_slots` entry; its color comes from
                            // the precomputed `result_color_at_pc` table (the
                            // `_result_argcode` analog, same source as
                            // `compute_inline_caller_frame`). The carried
                            // JitCode marker names the result's live stack
                            // position. `u16::MAX` = empty stack / skeleton,
                            // skip the bank null.
                            let color_idx_opt = (!jitcode_ptr.is_null())
                                .then(|| unsafe { &*jitcode_ptr })
                                .and_then(|jc| {
                                    resume_jit_pc.and_then(|jit_pc| {
                                        jc.payload.result_color_trivia_for_jitcode_pc(jit_pc)
                                    })
                                })
                                .and_then(|c| (c != u16::MAX).then_some(c as usize));
                            if let Some(color_idx) = color_idx_opt {
                                if color_idx >= registers_r_bank.len() {
                                    registers_r_bank.resize(color_idx + 1, OpRef::NONE);
                                }
                                registers_r_bank[color_idx] = null_ref;
                            }
                        }
                        Type::Float => {
                            if result_idx >= registers_f.len() {
                                registers_f.resize(result_idx + 1, OpRef::NONE);
                            }
                            registers_f[result_idx] = ctx.const_float(0);
                        }
                        Type::Void => {}
                    }
                }
            }
            (
                registers_i,
                registers_r_bank,
                registers_r_semantic,
                registers_f,
            )
        };
        // pyjitpl.py:202-203: read the 2-byte offset from JitCode.code
        // (upstream uses `decode_offset(self.jitcode.code, pc + 1)`) and
        // then read the `[len_i][len_r][len_f]` header from the shared
        // all_liveness byte string. Pyre stores the packed bytes on its
        // MetaInterpStaticData JitCode entry; upstream stores them on
        // metainterp_sd.
        //
        // Skeleton payload (not yet populated) → fall back to the
        // pyre-jit-trace LiveVars analysis. With the call.py-parity
        // jitcode_for callback wired up, this branch only fires for
        // sentinel/null jitcodes (PyreSym::new_uninit) that never
        // reach final code emission.
        let jc = unsafe { &*jitcode_ptr };
        if jc.payload.is_skeleton() {
            // `CallControl.get_jitcode` drain populates the jitcode before any
            // guard capture (pyjitpl.py:199 parity). Phase X-0 eliminated
            // the out-of-range-pc source. Phase X-1(a) migrated the
            // remaining guard/resume tests to the real compile path in
            // `pyre-jit`. Unconditional panic — any hit is a bug.
            panic!(
                "get_list_of_active_boxes: skeleton jitcode (not populated) \
                 at jitcode_pc={:?} — Phase X-0/X-1 removed all known triggers; \
                 further hits are bugs.",
                resume_jit_pc
            );
        }
        // `pyjitpl.py:199-233` parity: decode the `-live-` offset from
        // the jitcode byte stream via `jitcode.get_live_vars_info(pc,
        // op_live)` (`jitcode.py`), read the `[len_i][len_r]
        // [len_f]` header in `all_liveness`, then iterate per-bank
        // register indices with `LivenessIterator` (`liveness.py-
        // 201`). Register indices snapshot into `registers_r` in
        // int → ref → float bank order to match the encoder/decoder
        // contract (`all_liveness` byte layout).
        // Mirror RPython `pyjitpl.py:194-195 pc=self.pc`: an `in_a_call`
        // parent whose CALL sits in a try-block reads liveness at that call's
        // post-residual-call `-live-`/catch, so the encoded box count and
        // bank layout match the blackhole's carried jitcode resume position.
        // The pyre split between a narrowed
        // fallthrough `-live-` and the un-narrowed post-call `-live-`
        // otherwise crosses the two markers — the post-call `-live-` keeps
        // the CALL result ref the next opcode pops, so the decoder reads one
        // more ref than the encoder wrote. Without a catch marker, use the
        // plain fallthrough `-live-`.
        let jit_pc =
            resume_jit_pc.expect("get_list_of_active_boxes: no carried JitCode resume marker");
        let op_live = crate::state::op_live();
        let off = jc.payload.jitcode.get_live_vars_info(jit_pc, op_live);
        let all_liveness = crate::state::liveness_info_snapshot();
        assert!(
            off + 2 < all_liveness.len(),
            "get_list_of_active_boxes: liveness offset {} + header 3 bytes exceeds all_liveness length {}",
            off,
            all_liveness.len()
        );
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        let mut cursor = off + 3;
        let mut boxes = Vec::with_capacity((length_i + length_r + length_f) as usize);
        // `pyjitpl.py:216-233` line-by-line parity: each live register
        // is read from its kind-specific bank via direct list indexing
        // (`self.registers_i[index]` etc).  Rust slice indexing
        // bounds-checks and panics on OOB, matching Python's
        // IndexError contract — a liveness-listed index out of bank
        // range is an encoder/codewriter invariant violation, not a
        // silent NONE.
        use majit_jitcode::liveness::LivenessIterator;
        if length_i != 0 {
            // SAFETY: `all_liveness` bank at `cursor` is a complete
            // `encode_liveness` record (`liveness.py` `LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(cursor, length_i, &all_liveness) };
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_i[reg_idx as usize]);
            }
            cursor = it.offset();
        }
        if length_r != 0 {
            // PyPy parity: portal red args (`pypy/module/pypyjit/
            // interp_jit.py:67 reds = ['frame', 'ec']`) are JitCode
            // inputargs that appear in every `-live-` op's R-bank
            // (`liveness.py compute_liveness`). pyre's MIFrame stores
            // them on dedicated PyreSym fields (`sym.frame`,
            // `sym.execution_context`) rather than at color positions
            // in `sym.registers_r` — adapt by substituting at the
            // encoder boundary. After guard capture the wire-format
            // payload contains the OpRefs at the canonical portal
            // color positions; `_prepare_next_section` (resume.py)
            // fills the BH bank from there, mirroring RPython exactly.
            let portal_frame_reg = jc.payload.metadata.portal_frame_reg as u32;
            let portal_ec_reg = jc.payload.metadata.portal_ec_reg as u32;
            let sym_frame = self.sym().frame;
            // [frame, ec] portal-reds contract: `sym.execution_context`
            // may be OpRef::NONE on adapter paths (CALL_ASSEMBLER bridge
            // attach, bridge-from-guard). `ensure_execution_context`
            // recovers it via GETFIELD_GC(frame, execution_context_descr)
            // when needed; otherwise returns the seeded value.
            let sym_ec = self.ensure_execution_context(ctx);
            // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(cursor, length_r, &all_liveness) };
            while let Some(reg_idx) = it.next() {
                // Portal-red substitution applies only to the force-alived
                // SCRATCH case. The register allocator reuses these low
                // colors for real frame slots (a call result live across a
                // later call); at such PCs the bank materialization above
                // already wrote the slot's box at this color, and
                // substituting sym frame/ec would clobber it in the snapshot
                // (same scratch gate as `collect_outer_active_boxes`).
                let is_portal_red = reg_idx == portal_frame_reg || reg_idx == portal_ec_reg;
                let is_portal_red_scratch = is_portal_red
                    && crate::state::semantic_ref_slot_for_reg_color(
                        nlocals,
                        valid_stack_only,
                        pcdep_opt.unwrap_or(&[]),
                        reg_idx as usize,
                    )
                    .is_none();
                let opref = if is_portal_red_scratch {
                    if reg_idx == portal_frame_reg {
                        sym_frame
                    } else {
                        sym_ec
                    }
                } else {
                    if is_portal_red && crate::jitcode_dispatch::p2_diag_enabled() {
                        eprintln!(
                            "[p2-trait-scratch] jitcode_pc={} color={} owned by frame slot; keeping bank box",
                            jit_pc, reg_idx
                        );
                    }
                    // `registers_r_bank` is slot-indexed (`sym.registers_r`).
                    // `pyjitpl.py` `MIFrame.get_list_of_active_boxes` records
                    // `registers_r[index]`, the color-indexed register. An
                    // unmapped live color is a jitcode temporary no frame
                    // slot backs, so the slot mirror at `color` is a leftover
                    // local and is never read. At loop close the color-
                    // indexed bank is the walked merge-point snapshot
                    // `loop_close_registers_r` (`CloseLoop`
                    // `merge_point_registers_r`, the same `ctx.registers_r`
                    // `collect_outer_active_boxes` reads via
                    // `regs_r.get_box(color)`).
                    let mapped = crate::state::semantic_ref_slot_for_reg_color(
                        nlocals,
                        valid_stack_only,
                        pcdep_opt.unwrap_or(&[]),
                        reg_idx as usize,
                    )
                    .is_some()
                        // `reconstructed_ref_slot_color`: empty metadata is
                        // identity coloring (the color is the semantic slot)
                        // for skeleton/fixture compatibility. A populated
                        // `has_color_map` is authoritative, so an absent
                        // color stays unmapped.
                        || (!has_color_map
                            && (reg_idx as usize) < nlocals + valid_stack_only);
                    if mapped {
                        registers_r_bank[reg_idx as usize]
                    } else if let Some(v) = self
                        .loop_close_registers_r
                        .get(reg_idx as usize)
                        .copied()
                        .filter(|r| !r.is_none())
                    {
                        // Color-indexed merge-point register
                        // (`pyjitpl.py` `MIFrame.get_list_of_active_boxes`
                        // `registers_r[index]`). Includes a loop inputarg
                        // the virtualizable shadow still holds.
                        v
                    } else {
                        // `-live-` is a union over a PC range
                        // (`liveness.py`); a color live in that union
                        // and dead at this position has no register.
                        ctx.const_ref(0)
                    }
                };
                boxes.push(opref);
            }
            cursor = it.offset();
        }
        if length_f != 0 {
            // SAFETY: complete `all_liveness` bank (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(cursor, length_f, &all_liveness) };
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_f[reg_idx as usize]);
            }
        }
        boxes
    }

    /// RPython Box.type parity: build fail_arg_types matching compact
    /// active_boxes length. Each box carries its own immutable type.
    /// Header layout matches `virtualizable_gen.rs`'s `inputargs` (frame +
    /// `extra_reds` + `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`):
    /// `[frame:Ref, ec:Ref, last_instr:Int, pycode:Ref,
    ///   valuestackdepth:Int, debugdata:Ref, w_globals:Ref]` — line-by-line
    /// parity with `interp_jit.py:25-30` plus
    /// `interp_jit.py reds = ['frame', 'ec']`.
    fn build_fail_arg_types_for_active_boxes(&self, active_boxes: &[OpRef]) -> Vec<Type> {
        let mut types = crate::virtualizable_gen::virt_live_value_types(0);
        for &opref in active_boxes {
            types.push(self.value_type(opref));
        }
        types
    }

    pub(crate) fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        // history.py ConstInt.type / 262 ConstPtr.type / 308
        // ResOperation.type parity: a Box's type is an intrinsic
        // property of the Box itself, not a property of the slot it
        // happens to occupy. `ctx.get_opref_type` resolves the type
        // from the OpRef's producing op (constant kind, recorded
        // result_type, or `Forwarded::Info(PtrInfo)` for virtualized
        // See the PtrInfo fallback at the end of
        // `OptContext::opref_type`). Position-based scans of
        // `registers_r` / `virtualizable_boxes` were a pyre-only
        // adaptation that papered over earlier `get_opref_type` gaps;
        // those gaps are closed at the source now.
        let ctx_ref: &TraceCtx = unsafe { &*self.ctx };
        ctx_ref.get_opref_type(value).unwrap_or(Type::Ref)
    }

    pub(crate) fn load_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
    ) -> Result<OpRef, PyError> {
        // pyjitpl.py `_opimpl_getarrayitem_vable` (standard path):
        //     return self.metainterp.virtualizable_boxes[index]
        //
        // When the standard virtualizable is active, read the current OpRef
        // straight from the virtualizable_boxes cache (seeded by
        // initialize_virtualizable at setup_tracing, mirrored by
        // store_local_value on every STORE_FAST). This is the RPython
        // orthodox read path: no extra IR op, shadow-state only.
        //
        // Pyre's flat layout puts locals at flat indices
        // `num_static_extra_boxes .. num_static + nlocals`, i.e.
        // `NUM_VABLE_SCALARS + idx`. The scalar static fields live before
        // the array items, the standard-vable identity
        // (`virtualizable_boxes[-1]`) after.
        let vable_entry = {
            let s = self.sym();
            if s.is_active_vable_owner && s.bridge_local_oprefs.is_none() {
                let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + idx;
                ctx.virtualizable_box_at(flat_idx)
            } else {
                None
            }
        };
        if let Some(op) = vable_entry {
            let s = self.sym_mut();
            if idx >= s.registers_r.len() {
                return Err(PyError::type_error("local index out of range in trace"));
            }
            // Do NOT write registers_r[idx] = op here.  For active
            // virtualizable-owner traces, the vable shadow is the
            // authoritative source for locals, and stack colors can still
            // coalesce with local colors in the encoder's temporary bank.
            // Reintroducing a local mirror write here can overwrite the
            // value that guard capture is about to materialize for a stack
            // slot sharing the same color.
            return Ok(op);
        }
        let s = self.sym_mut();
        if idx >= s.registers_r.len() {
            return Err(PyError::type_error("local index out of range in trace"));
        }
        if s.registers_r.get(idx).expect("bound Ref register") == OpRef::NONE {
            if s.bridge_local_oprefs.is_some() {
                // Bridge trace: OpRef::NONE means this local is a constant
                // or virtual from resume data, not a missing vable slot.
                // Read from the concrete frame via the locals_cells_stack_w
                // array.  RPython `virtualizable.py read_boxes` does the
                // array-field access in two steps:
                //
                //   for _, fieldname in unroll_array_fields:
                //       lst = getattr(virtualizable, fieldname)   # :94
                //       for i in range(len(lst)):
                //           boxes.append(wrap(cpu, lst[i], ...))  # :96
                //
                // Step 1 (`getattr`) is `GETFIELD_GC_R` of the array
                // field (`frame_locals_cells_stack_array` /
                // `opimpl_getfield_gc_r`). Step 2 (`lst[i]`) is
                // `GETARRAYITEM_GC_R` via `trace_array_getitem_value`,
                // which indexes the array GCREF (`pyobject_gcarray_descr`
                // `base_size` = items offset), not the frame pointer.
                let frame_ref = s.frame;
                let array_ref = crate::state::frame_locals_cells_stack_array(ctx, frame_ref);
                let idx_const = ctx.const_int(idx as i64);
                s.registers_r
                    .set(idx, trace_array_getitem_value(ctx, array_ref, idx_const));
            } else {
                // Active vable owner whose registers_r[idx] is NONE cannot
                // exist: init_symbolic (state.rs) seeds
                // registers_r[idx] = OpRef::from_raw(base + idx) for every i in
                // 0..nlocals before any load_local_value runs. Reachability
                // audit (`MAJIT_PROBE_VABLE_FALLBACK`) confirmed
                // this empirically: 0 firings across debug unit tests
                // (debug_assert!(false)) and 0 firings across 28 release
                // benchmark runs (env-gated eprintln). The remaining
                // fallthrough is the non-vable-owner path —
                // `s.locals_cells_stack_array_ref` is the callee's own
                // locals_cells_stack_w array (seeded by Stage 1 at
                // the retired inline-call path).
                let idx_const = ctx.const_int(idx as i64);
                s.registers_r.set(
                    idx,
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const),
                );
            }
        }
        Ok(s.registers_r.get(idx).expect("bound Ref register"))
    }

    #[allow(dead_code)]
    pub(crate) fn set_next_instr(&mut self, _ctx: &mut TraceCtx, target: usize) {
        self.sym_mut().pending_next_instr = Some(target);
    }

    /// Update virtualizable last_instr and valuestackdepth.
    /// RPython parity: always use orgpc (opcode start PC) as the semantic
    /// next instruction, so the heap frame stores `last_instr = orgpc - 1`.
    /// The trace loop advancement uses pending_next_instr separately
    /// (in pyjitpl.rs step_*_frame).
    pub(crate) fn flush_to_frame(&mut self, ctx: &mut TraceCtx) {
        let resume_pc = self.orgpc;
        let frame_addr = self.concrete_frame_addr;
        // virtualizable.py read_boxes reads statics from the LIVE
        // virtualizable.  The root MIFrame's concrete frame is the
        // trace-stepping snapshot (`snapshot_for_tracing`), whose
        // `debugdata` is an owned clone freed when tracing
        // ends — a const captured from the snapshot dangles in the
        // compiled trace's resume data, and the guard-failure vable
        // write (`write_from_resume_data_partial`) then stamps the
        // dangling pointer into the live frame.  Read the pointer-valued
        // statics from the live virtualizable instead; `pycode` is
        // copied by the snapshot so either source gives the same value.
        // `last_instr` / `valuestackdepth` are plain values that evolve
        // with the trace, so they stay snapshot-sourced below.
        let statics_addr = {
            let live = self.sym().current_live_vable_frame();
            if live != 0 && self.sym().owns_virtualizable_shadow() {
                live
            } else {
                frame_addr
            }
        };
        let code_ptr = if statics_addr != 0 {
            unsafe { *((statics_addr + PYFRAME_PYCODE_OFFSET) as *const usize) }
        } else {
            0
        };
        let ns_ptr = self.sym().concrete_namespace as i64;
        // Read from the concrete `PyFrame.valuestackdepth` rather than the
        // symbolic mirror. `resume_pc == self.orgpc` (the
        // start PC of the current opcode) so PyFrame holds the correct
        // pre-opcode value (the interpreter step for this opcode has not
        // run yet).  Falls back to the symbolic value only when
        // `concrete_frame_addr == 0` (test-only sym-only MIFrames).
        let vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth) as i64;
        // `debugdata` keeps the box `virtualizable.py` `read_boxes` /
        // `initialize_virtualizable` (and a crossed-frame
        // `gen_load_from_other_virtualizable`) installed on
        // `virtualizable_boxes`. DELETE_NAME resolves the name through
        // `debugdata.w_locals`, so a recording-time ConstPtr pins the
        // origin frame's mapping. `pyjitpl.py` `reached_loop_header`
        // concatenates that shadow (`live_arg_boxes += virtualizable_boxes;
        // pop()`); `s.vable_debugdata` is only the origin-frame seed and
        // is not rebased when those loads replace the shadow. Writing it
        // back here made the closing JUMP name a pre-cut `InputArgRef`
        // that `original_boxes` no longer held (`history.py`
        // `cut_trace_from` / `opencoder.py` `TraceIterator._get`).
        let last_instr_value = resume_pc as i64 - 1;
        let last_instr_op = ctx.const_int(last_instr_value);
        let pycode_op = ctx.const_ref(code_ptr as i64);
        let vsd_op = ctx.const_int(vsd);
        let debugdata_idx = ctx
            .virtualizable_info()
            .and_then(|info| info.static_field_index_by_name("debugdata"));
        let debugdata_op = debugdata_idx
            .and_then(|i| ctx.virtualizable_box_at(i))
            .filter(|&op| !op.is_none() && !ctx.opref_is_void_producer(op))
            .unwrap_or_else(|| self.sym().vable_debugdata);
        let w_globals_op = ctx.const_ref(ns_ptr);
        let owns = {
            let s = self.sym_mut();
            s.vable_last_instr = last_instr_op;
            s.vable_pycode = pycode_op;
            s.vable_valuestackdepth = vsd_op;
            s.vable_debugdata = debugdata_op;
            s.frame_w_globals = w_globals_op;
            s.owns_virtualizable_shadow()
        };
        // pyjitpl.py `_opimpl_setfield_vable` parity:
        // mirror the republished statics into the canonical
        // `metainterp.virtualizable_boxes` shadow so subsequent readers
        // (snapshot, JUMP-arg dedup) see the same identity that
        // `s.vable_*` carries. Leave `debugdata` on the shadow box
        // `read_boxes` already installed — it is not republished.
        if owns {
            mirror_vable_static_to_boxes(
                ctx,
                "last_instr",
                last_instr_op,
                Value::Int(last_instr_value),
            );
            mirror_vable_static_to_boxes(ctx, "pycode", pycode_op, Value::Ref(GcRef(code_ptr)));
            mirror_vable_static_to_boxes(ctx, "valuestackdepth", vsd_op, Value::Int(vsd));
        }
    }

    /// capture_resumedata(resumepc=orgpc) parity: flush vable fields for guards.
    ///
    /// When a pre-opcode snapshot is present, sets vable_last_instr = orgpc - 1
    /// and vable_valuestackdepth = the snapshot depth. The guard's fail_args
    /// then carry the pre-opcode stack state so the blackhole interpreter
    /// can re-execute the opcode from orgpc.
    ///
    /// Note: branch-guard recording does NOT call this — branch guards
    /// build their own fail_args with post-pop state and other_target PC
    /// (see the comment there for why).
    fn flush_to_frame_for_guard(&mut self, ctx: &mut TraceCtx) {
        // RPython capture_resumedata(resumepc=orgpc) parity:
        // Always use orgpc (opcode start PC) as the resume PC.
        let resume_pc = self.orgpc;
        let vsd = self.pre_opcode_concrete_depth() as i64;
        // pyjitpl.py `capture_resumedata` parity: RPython reads
        // `metainterp.virtualizable_boxes` without mutating it. The two
        // fields that advance per-opcode (`last_instr`, `valuestackdepth`)
        // need a guard-time-correct override here because pyre's tracer
        // is itself the dispatch loop: at guard time the active opcode is
        // the one at `orgpc`, so the snapshot must encode the pre-opcode
        // state (`last_instr = orgpc - 1`, `valuestackdepth = pre-opcode
        // depth via `pre_opcode_registers_r`).
        // The other three scalars (`pycode`, `debugdata`, `w_globals`)
        // keep the inputarg OpRefs `init_vable_indices`
        // seeded at trace start because pyre-jit-trace never enters
        // their mutators under pyre's 3.14 bytecode: `pycode` /
        // `w_globals` are set only by `pyframe.rs::frame_reinit`;
        // `debugdata` only by `getorcreate_debug_data` on debug paths.
        // This matches RPython's
        // "boxes carry vable inputargs" model — see
        // `mirror_vable_static_to_boxes` doc for the convergence path
        // when those handlers are re-introduced.
        let last_instr_value = resume_pc as i64 - 1;
        let last_instr_op = ctx.const_int(last_instr_value);
        let vsd_op = ctx.const_int(vsd);
        let s = self.sym_mut();
        s.vable_last_instr = last_instr_op;
        s.vable_valuestackdepth = vsd_op;
        // The shared `ctx.virtualizable_boxes` shadow is intentionally
        // not mirrored here — see `mirror_vable_static_to_boxes` doc
        // for the convention.  The two stores have distinct roles:
        // `s.vable_*` is the snapshot reader's view (carries pre-opcode
        // overrides set here, save/restored by branch-guard recording),
        // `ctx.virtualizable_boxes` is the JUMP/JIT-time view (consumed
        // by `close_loop_args_at`'s JUMP-arg derivation).
    }

    /// Loop-carried values must follow the typed live-state contract used by
    /// PyreMeta::slot_types / restore_values().
    ///
    /// In pyre's typed INT/REF/FLOAT model, integer locals cross a loop JUMP
    /// as raw Int values, not freshly boxed W_Int objects.
    fn materialize_loop_carried_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        slot_type: Type,
    ) -> OpRef {
        match slot_type {
            Type::Int => match self.value_type(value) {
                Type::Int => value,
                Type::Ref => {
                    // Convert boxed W_Int back to its raw payload so the loop
                    // header sees the typed INT stream expected by restore_values().
                    self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, value))
                }
                _ => value,
            },
            Type::Ref => match self.value_type(value) {
                Type::Int => {
                    // Virtualizable slots are Ref — re-box raw Int for the
                    // loop header which expects boxed W_IntObject.
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    crate::trace_box_int(
                        ctx,
                        value,
                        w_int_size_descr(),
                        int_intval_descr(),
                        int_type_addr,
                    )
                }
                Type::Float => {
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    crate::trace_box_float(
                        ctx,
                        value,
                        w_float_size_descr(),
                        float_floatval_descr(),
                        float_type_addr,
                    )
                }
                _ => value,
            },
            _ => value,
        }
    }

    /// Pure-read shape predictor for `close_loop_args_at` output.
    ///
    /// Returns the LENGTH that `close_loop_args_at` would produce at
    /// the current sym/ctx state, without mutating either.  Used by
    /// the merge-point seed sites (`trace.rs::trace_bytecode`,
    /// `TraceCtx::new`, `TraceCtx::with_green_key`) that need to
    /// allocate `original_boxes` of the same shape future
    /// `close_loop_args` calls will produce, so
    /// `pyjitpl.py assert len(original_boxes) == len(live_arg_boxes)`
    /// can fire (see memory
    /// `merge_point_shape_assert_prerequisite_2026_05_03.md`).
    ///
    /// Shape derivation matches `close_loop_args_at`:
    /// `1 (frame) + extra_reds (ec) + NUM_VABLE_SCALARS + target_array_capacity`
    /// where the vable scalars are
    /// `[next_instr, code, stack_depth, debugdata, namespace]`
    /// and `target_array_capacity` is either the virtualizable array
    /// lengths sum (when known) or the fallback `nlocals + stack_only`.
    pub(crate) fn live_args_shape_at(&self, ctx: &TraceCtx) -> usize {
        let extra_reds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        let nlocals = self.sym().nlocals;
        // Pure-read of stack depth comes from
        // the concrete `PyFrame` (no symbolic mirror).  RPython's
        // pyjitpl.py:2957-2965 `live_arg_boxes` shape derives directly
        // from PyFrame's `locals_cells_stack_w` length + `valuestackdepth`
        // — there is no symbolic mirror to consult.
        let vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth);
        let stack_only = vsd.saturating_sub(nlocals);
        let target_array_capacity = ctx
            .virtualizable_array_lengths()
            .map(|lengths| lengths.iter().copied().sum::<usize>())
            .filter(|&len| len >= nlocals)
            .unwrap_or(nlocals + stack_only);
        1 + extra_reds + crate::virtualizable_gen::NUM_VABLE_SCALARS + target_array_capacity
    }

    /// Value to attach to one loop-carried array box.
    ///
    /// Int and float boxes take the unboxed word in
    /// `PyreSym::concrete_value_at`. Those words are not GC pointers.
    /// A ref box takes the virtualizable shadow for the same slot.
    /// `walk_active_trace_refs` forwards that cell. `concrete_value_at`'s
    /// ref is an off-heap copy from trace start, so stamping it replaces
    /// the forwarded word. The jump arg may be a `SameAs` of the shadow
    /// box; the slot's value is still the one `read_boxes` would report.
    fn loop_carried_runtime_value(
        &self,
        ctx: &TraceCtx,
        opref: OpRef,
        flat_idx: usize,
        slot: usize,
    ) -> Option<Value> {
        if opref == OpRef::NONE || opref.is_constant() {
            return None;
        }
        match opref.ty()? {
            Type::Int => match self.sym().concrete_value_at(slot) {
                ConcreteValue::Int(v) => Some(Value::Int(v)),
                _ => None,
            },
            Type::Float => match self.sym().concrete_value_at(slot) {
                ConcreteValue::Float(v) => Some(Value::Float(v)),
                _ => None,
            },
            Type::Ref => {
                let value = ctx.virtualizable_entry_at(flat_idx)?.1;
                match value {
                    Value::Ref(r) if r != GcRef::NO_CONCRETE => Some(value),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// TODO: bundles `pyjitpl.py:2957-2965` `live_arg_boxes`
    /// construction (`greenboxes + redboxes + virtualizable_boxes`,
    /// `pop()` the trailing token) with the `vable_last_instr` pin
    /// (`pyjitpl.py:2973`). RPython performs both inline within
    /// `reached_loop_header`; pyre extracts them because pyre's "args"
    /// are `OpRef`s pulled from the unified `registers_r` register
    /// file, and the merge `target_pc` must be threaded through
    /// explicitly (RPython has it implicitly in `redboxes`).
    pub(crate) fn close_loop_args_at(
        &mut self,
        ctx: &mut TraceCtx,
        target_pc: Option<usize>,
        header_marker_jit_pc: Option<usize>,
        merge_point_registers_r: &[OpRef],
    ) -> Vec<OpRef> {
        // The `--TICK--` poll now lives in JUMP_BACKWARD's jitcode
        // (`emit_jump_absolute_tick`: `raw_load_i` / `uint_ge` / `goto_if_not`
        // then residual `bytecode_trace` on the armed arm).  Synthesizing it
        // here at loop close duplicated that guard and resumed it at the
        // JUMP_BACKWARD *pre-opcode* state, so the blackhole and any bridge
        // walked `loop_header` + `goto` with no dispatcher.
        //
        // Load-bearing invariant (still applies to the walked `raw_load_i`):
        // RawLoadI must remain outside the always-pure range and its
        // descriptor must remain non-pure. Otherwise CSE can forward the
        // preamble's guarded-zero value into the loop body and
        // `optimize_guard_false` deletes the body's poll.
        self.loop_close_marker_jit_pc = header_marker_jit_pc;
        self.loop_close_registers_r = merge_point_registers_r.to_vec();
        // pyjitpl.py reached_loop_header: virtualizable_boxes
        // (read from locals_cells_stack_w[*] by virtualizable.py read_boxes
        // read_boxes) are carried into the JUMP unchanged, including
        // stack slots. Do NOT truncate to nlocals here.
        //
        // Read the user-side
        // `valuestackdepth` from the concrete `PyFrame` to match
        // `live_args_shape_at`'s reader.  Both helpers share the same
        // shape derivation; reading from different sources lets the two
        // diverge whenever the symbolic mirror drifts from PyFrame.
        // RPython's pyjitpl.py derives `live_arg_boxes` from
        // PyFrame's `locals_cells_stack_w` length + `valuestackdepth`
        // — no symbolic mirror in the loop.
        let concrete_nlocals = self.sym().nlocals;
        // interp_jit.py reaches `jit_merge_point` from inside the
        // dispatch loop, so the frame it promotes `valuestackdepth` off is
        // current at every merge point. Here the concrete `PyFrame` is only
        // stepped by the interpreter. A walk that resumed from a guard keeps
        // the `valuestackdepth` the resume wrote at the resume PC and never
        // advances it, so a close that lands on a DIFFERENT bytecode offset —
        // a cross-loop retarget, where `cut_trace_from` re-heads the trace at
        // the merge point it walked into — reads a depth belonging to the
        // offset it started from. Every operand-stack slot at or above that
        // depth is then treated as dead capacity and force-nulled into the
        // JUMP, while the target LABEL (built from the merge point's own
        // `virtualizable_boxes`) binds them: the loop body reads a null where
        // the entry reload put a live value.
        //
        // A bytecode offset has exactly one operand-stack depth, so the merge
        // point's depth is available statically from the same liveness table
        // `maybe_compile_and_run` gates interpreter entry on. Use that exact
        // depth in either direction. The full virtualizable array capacity is
        // still carried below; slots above the header's live depth are dead
        // capacity and are null-padded rather than counted as live stack.
        self.close_merge_point_vsd = target_pc.and_then(|pc| {
            crate::state::merge_point_stack_depth_to_recover(self.concrete_frame_addr, pc)
        });
        let concrete_vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth)
            .max(concrete_nlocals);
        {
            let s = self.sym_mut();
            s.nlocals = concrete_nlocals;
            s.valuestackdepth = concrete_vsd;
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            // virtualizable.py:44 + interp_jit.py:25-30: locals_cells_stack_w[*]
            // is a W_Root array → every item is declared Ref. The loop-carried
            // types passed to the JUMP / merge point MUST be Ref for every
            // array slot; tracker-observed Int/Float types are internal to
            // unboxing lowering and must not leak into the inputarg contract.
            //
            // The type-map reset is deferred until AFTER the materialize loop
            // below. Resetting here would poison `self.value_type(value)` used
            // by `materialize_loop_carried_value` — an Int-typed OpRef on the
            // symbolic stack (e.g. a GetarrayitemRawI result LOAD_FASTed onto
            // the stack) would appear Ref to the materializer and skip the
            // Int→Ref boxing, handing the MIFrame merge-point a raw Int value
            // in a Ref-typed slot. Cut_trace_from then installs that raw Int
            // at a cut-inputarg position whose declared type is Ref, and the
            // downstream unroll pass produces JUMP args that pass an Int into
            // a Ref slot at runtime → SIGSEGV (str x24, [x12, #0x10] with x12
            // carrying the unboxed int payload).
            //
            // `registers_r` is the unified abstract register file;
            // reserve `[nlocals..nlocals+stack_only)` for the stack so
            // merge-point JUMP args have a stable slice.
            let min_regs_len = concrete_nlocals + stack_only;
            if s.registers_r.len() < min_regs_len {
                s.registers_r.resize(min_regs_len, OpRef::NONE);
            }
        }
        self.flush_to_frame(ctx);
        // pyjitpl.py reached_loop_header: a merge-point resume enters
        // the target loop at `pc`, so last_instr must be `pc - 1` so the
        // interpreter's `next_instr() = last_instr + 1` returns the merge
        // point. flush_to_frame already stored `orgpc - 1`; override with
        // the merge target.
        //
        // Propagation gap #1: propagate the override
        // into the virtualizable_boxes shadow so the writeback below
        // emits the merge-target PC, not the orgpc placed by
        // flush_to_frame. virtualizable_boxes[0] = vable_last_instr per
        // `virtualizable_gen.rs`'s `inputargs` ordering.
        if let Some(pc) = target_pc {
            let last_instr_value = pc as i64 - 1;
            let opref = ctx.const_int(last_instr_value);
            {
                let s = self.sym_mut();
                s.vable_last_instr = opref;
            }
            // `interp_jit.py` `jump_absolute` writes `self.last_instr =
            // intmask(jumpto)` before `can_enter_jit`. On a virtualizable
            // that is `_opimpl_setfield_vable` / `synchronize_virtualizable`,
            // so `virtualizable_boxes[0]` already holds the header pc when
            // `reached_loop_header` concatenates it. The walker records that
            // store in `setfield_vable_via_metainterp`; this pin is the same
            // write for the source-level close, including traces whose
            // `owns_virtualizable_shadow` gate would otherwise leave the
            // shadow on the pre-pin box. Snapshot capture still
            // save/restores the boxes.
            mirror_vable_static_to_boxes(ctx, "last_instr", opref, Value::Int(last_instr_value));
        }
        // No vable heap-writeback before the closing JUMP: the
        // virtualizable stays virtual across the loop/bridge edge. The
        // live vable scalars/array items reach the target LABEL through
        // the JUMP-arg derivation below; loop-invariant fields fold into
        // the resume-data payload of each guard. A guard failure rebuilds
        // the heap frame from that payload (`consume_vref_and_vable_boxes`
        // → `write_boxes`), and a forcing residual call materializes it
        // via `synchronize_virtualizable` — neither needs a pre-written
        // heap frame at the JUMP boundary.
        // An active virtualizable owner must have its array base seeded by
        // `init_symbolic` / `become_active_vable_owner` before the JUMP-arg
        // derivation below reads `nlocals`/`vable_array_base`.  `nlocals` is
        // NOT a usable proxy for "init ran": module-scope (`<module>`) frames
        // are vable owners (M1 portal gate) yet have `co_nlocals == 0`, the
        // same value as the struct default — names go through globals, not
        // fast locals.  `target_array_capacity` below handles `nlocals == 0`
        // via the `valuestackdepth` saturating-sub, so the seeded base is the
        // real precondition.
        debug_assert!(
            !self.sym().is_active_vable_owner || self.sym().vable_array_base.is_some(),
            "an active vable owner must have a seeded vable_array_base before close_loop_args_at"
        );
        // RPython close_loop_args parity: JUMP args must match the target
        // label's types (inputarg_types). materialize_loop_carried_value
        // boxes values to match (e.g. Int → Ref for virtualizable locals).
        //
        // For bridge traces, ctx.inputarg_types() returns the bridge's
        // guard fail_arg types, NOT the root loop's label types. The JUMP
        // targets the root loop label, so resolve the root loop's LABEL/
        // inputargs types via `front_target_inputarg_types` (peeled-entry
        // LABEL when unrolled, root TreeLoop.inputargs otherwise — see
        // `MetaInterp::front_target_inputarg_types` doc).
        //
        // A close reached through the walk targets the merge point that matched
        // (`original_boxes`, pyjitpl.py), whichever kind of trace is
        // recording: `reached_loop_header` scans `current_merge_points` first
        // and only falls through to the previously compiled loop when nothing
        // matched. So the merge point recorded at the closing header answers
        // ahead of the bridge's front target; a bridge that walked into a loop
        // header it had itself passed closes at THAT label, not at the loop its
        // guard came from. With no such merge point the close targets the root
        // loop's LABEL/inputargs, which `front_target_inputarg_types` resolves
        // (a bridge's own `inputarg_types()` are its guard fail_arg types).
        let inputarg_types = target_pc
            .and_then(|pc| ctx.merge_point_arg_types_at_header(pc))
            .or_else(|| {
                let (driver, _) = crate::driver::driver_pair();
                driver
                    .is_bridge_tracing()
                    .then(|| driver.current_trace_green_key())
                    .flatten()
                    .and_then(|gk| driver.front_target_inputarg_types(gk))
            })
            .unwrap_or_else(|| ctx.inputarg_types());
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // `extra_reds` reflects the canonical ec/red layout (NUM_EXTRA_REDS).
        // Drives the conditional ec push at args[1] and the dedup-side
        // OpRef ↔ virtualizable_box mapping below.
        let extra_reds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        // pyjitpl.py reached_loop_header parity: once the
        // descriptor-driven virtualizable path is active, JUMP args must carry
        // the full virtualizable array capacity. compile.rs later expands the
        // loop entry from the same heap lengths; emitting only the live stack
        // window here leaves too few source args for that expansion.
        let target_array_capacity = ctx
            .virtualizable_array_lengths()
            .map(|lengths| lengths.iter().copied().sum::<usize>())
            .filter(|&len| len >= self.sym().nlocals)
            .unwrap_or_else(|| {
                self.sym().nlocals
                    + self
                        .sym()
                        .valuestackdepth
                        .saturating_sub(self.sym().nlocals)
            });
        // [frame, ec] portal-reds contract: recover ec before the sym()
        // snapshot below so JUMP args never carry OpRef::NONE in the ec
        // slot on adapter / bridge-from-guard paths.
        let recovered_ec = self.ensure_execution_context(ctx);
        // A void `sym.frame` (typically `DebugMergePoint` occupying the
        // next `op_count` slot) cannot be a JUMP red. Recover the
        // virtualizable identity (`virtualizable_boxes[-1]`), which is
        // seeded once from the portal frame.
        if ctx.opref_is_void_producer(self.sym().frame) {
            if let Some(identity) = ctx.standard_virtualizable_box() {
                if !ctx.opref_is_void_producer(identity) {
                    self.sym_mut().frame = identity;
                }
            }
        }
        // pyjitpl.py reached_loop_header builds one `live_arg_boxes` from
        // reds + `virtualizable_boxes[:-1]`. Fill any short shadow before
        // both this JUMP and the walker's `append_virtualizable_boxes` read
        // that same list. Array index `>= valuestackdepth` is the dead
        // stack tail (`pyframe.py popvalue_maybe_none`); majit is generic
        // and takes that boundary from here.
        // The stack depth reads from `PyFrame.valuestackdepth`
        // (via `concrete_valuestackdepth()`) rather than the symbolic
        // mirror.  `close_loop_args_at` runs at the orgpc anchor where
        // PyFrame still holds the pre-opcode state.
        let concrete_vsd = self
            .concrete_valuestackdepth()
            .unwrap_or_else(|| self.sym().valuestackdepth);
        ctx.fill_virtualizable_boxes_to_declared_layout(Some(concrete_vsd));
        // `popvalue_maybe_none` stored None at every `i >= valuestackdepth`.
        // Fill only covers NONE holes; occupied leftovers of a loop-carried
        // local stay in the shadow and split peel/body GuardTrue.
        null_vable_dead_array_tail(ctx, concrete_vsd);
        let (
            frame,
            execution_context,
            next_instr,
            code,
            stack_depth,
            debugdata,
            nlocals,
            locals,
            stack,
            _local_types,
            _stack_types,
        ) = {
            let s = self.sym();
            let nlocals = s.nlocals;
            let stack_only = concrete_vsd.saturating_sub(s.nlocals);
            // virtualizable.py `read_boxes` + pyjitpl.py:2954-2965
            // `reached_loop_header`: `virtualizable_boxes` length is the
            // target vable array capacity (`nlocals + ncells + co_stacksize`),
            // not the live Python stack depth. JUMP args carry that full
            // capacity so every target LABEL slot has a matching source.
            // Slots beyond the live prefix are left as `OpRef::NONE` and
            // filled by `materialize_fail_arg_slot` below, which reads
            // `concrete_value_at` and falls back to `PY_NULL` for dead
            // capacity slots — mirroring RPython's null-padded
            // virtualizable_boxes tail.
            let target_stack_capacity = target_array_capacity.saturating_sub(nlocals);
            let mut stack_types_vec =
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec();
            stack_types_vec.resize(target_stack_capacity, Type::Ref);
            // pyjitpl.py `reached_loop_header` parity: read
            // both locals and stack values from the virtualizable shadow
            // (`virtualizable_boxes[NUM_VABLE_SCALARS + i]`). The shadow
            // is RPython's single source of truth for the
            // `locals_cells_stack_w` view; `push_typed_value` /
            // `store_local_value` mirror every write into it for traces
            // that satisfy `owns_virtualizable_shadow()` (the loop
            // portal AND every bridge that seeded its own
            // `bridge_local_oprefs`). Non-owner traces (rare —
            // inline-callee scaffolding before the inline path takes
            // over) keep the legacy semantic registers_r read.
            //
            // Shadow path bounds: virtualizable_boxes is sized to
            // `target_array_capacity` (NUM_VABLE_SCALARS + nlocals +
            // co_stacksize) at vable init, so `stack_only.min(
            // target_stack_capacity)` is the correct live-prefix
            // length; the legacy `reg_len.saturating_sub(locals_len)`
            // cap silently dropped OpRefs once the operand stack
            // overgrew the registers_r slice (RPython
            // `reached_loop_header` carries the full
            // `virtualizable_boxes[:-1]` regardless of register-file
            // occupancy).  The non-shadow registers_r read keeps the
            // reg_len cap because reading past `registers_r.len()`
            // panics there.
            let (locals_vec, mut stack_vec) = if s.owns_virtualizable_shadow() {
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                let shadow_stack_len = stack_only.min(target_stack_capacity);
                let locals_vec: Vec<OpRef> = (0..nlocals)
                    .map(|i| {
                        ctx.virtualizable_box_at(nvs + i)
                            .expect("close_loop_args_at: missing virtualizable local box")
                    })
                    .collect();
                let stack_vec: Vec<OpRef> = (0..shadow_stack_len)
                    .map(|d| {
                        ctx.virtualizable_box_at(nvs + nlocals + d)
                            .expect("close_loop_args_at: missing virtualizable stack box")
                    })
                    .collect();
                (locals_vec, stack_vec)
            } else {
                let read_color = |color: usize| s.registers_r.get(color).unwrap_or(OpRef::NONE);
                let locals_vec: Vec<OpRef> = (0..nlocals).map(|i| read_color(i)).collect();
                let live_stack_len = stack_only.min(target_stack_capacity);
                let stack_vec: Vec<OpRef> = (0..live_stack_len)
                    .map(|d| read_color(nlocals + d))
                    .collect();
                (locals_vec, stack_vec)
            };
            stack_vec.resize(target_stack_capacity, OpRef::NONE);
            (
                s.frame,
                recovered_ec,
                s.vable_last_instr,
                s.vable_pycode,
                ctx.virtualizable_box_at(2)
                    .unwrap_or(s.vable_valuestackdepth),
                s.vable_debugdata,
                nlocals,
                locals_vec,
                stack_vec,
                s.symbolic_local_types.clone(),
                stack_types_vec,
            )
        };
        let mut args = vec![frame];
        // NUM_EXTRA_REDS == 1 (crate const-assert): `reds = ['frame', 'ec']`.
        args.push(execution_context);
        if ctx.has_virtualizable_boxes() {
            // Same list the walker JUMP already built:
            // `append_virtualizable_boxes` = reds + `virtualizable_boxes[:-1]`.
            // `pyjitpl.py reached_loop_header` concatenates that list after
            // `jump_absolute` has written `last_instr` (`_opimpl_setfield_vable`)
            // and `popvalue_maybe_none` has stored None (`_opimpl_setarrayitem_vable`).
            args.extend(ctx.virtualizable_data_boxes());
        } else {
            args.extend_from_slice(&[next_instr, code, stack_depth, debugdata]);
            for (idx, value) in locals.into_iter().enumerate() {
                let target_type = inputarg_types
                    .get(num_scalars + idx)
                    .copied()
                    .unwrap_or(Type::Ref);
                // Materialize NONE slots from concrete frame before boxing.
                // RPython's live_arg_boxes never contains holes at loop closure
                // because MIFrame.run_one_step always updates all live registers.
                let value = self.materialize_fail_arg_slot(ctx, value, target_type, idx);
                args.push(self.materialize_loop_carried_value(ctx, value, target_type));
            }
            // Live value-stack window: slots at index >= live_stack_len are dead
            // capacity (Python index >= valuestackdepth). `interpreter/pyframe.py`
            // `popvalue_maybe_none` nulls a popped slot, so `read_boxes` reports
            // None for every dead stack slot and `virtualizable_boxes` carries a
            // null tail; the target loop LABEL's stack tail is therefore all-null
            // and folds away. pyre's bridge value-stack clear is gated off
            // (`is_active_vable_owner` excludes bridges to avoid the null-base
            // `InvalidLoop` abort), so the concrete frame still holds the stale
            // popped pointers (e.g. a caught exception) in those slots. Reading
            // them through `materialize_fail_arg_slot` would put live pointers in
            // the JUMP tail where the loop label expects null, blocking
            // `optimize_bridge` retarget, and would also disagree with resume,
            // which reconstructs the dead tail as null. Force the null here: these
            // slots reach only the terminal JUMP args, never an in-trace field
            // base, so no `get_const_info_mut` null-base abort.
            let live_stack_len = concrete_vsd.saturating_sub(nlocals);
            for (stack_idx, value) in stack.into_iter().enumerate() {
                let target_type = inputarg_types
                    .get(num_scalars + nlocals + stack_idx)
                    .copied()
                    .unwrap_or(Type::Ref);
                let value = if stack_idx >= live_stack_len {
                    let typed_null = extract_concrete_typed_value(target_type, PY_NULL);
                    fail_arg_opref_for_typed_value(ctx, typed_null)
                } else {
                    self.materialize_fail_arg_slot(ctx, value, target_type, nlocals + stack_idx)
                };
                args.push(self.materialize_loop_carried_value(ctx, value, target_type));
            }
        }
        // virtualizable.py:44 parity (delayed): now that all materialize_loop_
        // carried_value calls have consulted each OpRef's actual type, flip the
        // symbolic type maps to the post-loop invariant where every array slot
        // is Ref. Downstream consumers (reached_loop_header's live_types, the
        // merge-point snapshot it stores) observe the Ref contract while the
        // box/unbox decisions above still see the pre-loop truth.
        {
            let s = self.sym_mut();
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            s.symbolic_local_types = vec![Type::Ref; concrete_nlocals];
            s.symbolic_stack_types = vec![Type::Ref; stack_only];
        }
        // pyjitpl.py remove_consts_and_duplicates:
        //     def remove_consts_and_duplicates(self, boxes, endindex, duplicates):
        //         for i in range(endindex):
        //             box = boxes[i]
        //             if isinstance(box, Const) or box in duplicates:
        //                 boxes[i] = self.history.record_same_as(box)
        //             else:
        //                 duplicates[box] = None
        //
        //     def reached_loop_header(self, greenboxes, redboxes):
        //         duplicates = {}
        //         self.remove_consts_and_duplicates(redboxes, len(redboxes),
        //                                           duplicates)
        //         live_arg_boxes = greenboxes + redboxes
        //         if self.jitdriver_sd.virtualizable_info is not None:
        //             self.remove_consts_and_duplicates(
        //                 self.virtualizable_boxes,
        //                 len(self.virtualizable_boxes)-1,
        //                 duplicates)
        //             live_arg_boxes += self.virtualizable_boxes
        //             live_arg_boxes.pop()
        //
        // RPython dedups ALL of redboxes (1 = vable_box = frame) AND
        // ALL of virtualizable_boxes[:-1] (static_fields + array_items),
        // sharing one `duplicates` dict across both calls. In pyre's flat
        // layout `args = [frame, ni, code, vsd, ns, locals..., stack...]`,
        // that corresponds to every index 0..args.len(). Previously pyre
        // skipped the scalar header slots (frame plus the vable statics),
        // which is a line-by-line divergence from RPython.
        // Track slots that the dedup actually mutated so we can mirror the
        // `put_back_list_of_boxes3` mutation below (pyjitpl.py writes
        // the deduped redboxes back to the frame's registers; RPython's
        // `remove_consts_and_duplicates` additionally mutates
        // `self.virtualizable_boxes` in place so subsequent reads see the
        // SameAs-wrapped identities).
        for (i, arg) in args.iter_mut().enumerate() {
            if !ctx.opref_is_void_producer(*arg) {
                continue;
            }
            *arg = if i == 0 {
                ctx.standard_virtualizable_box()
                    .filter(|id| !ctx.opref_is_void_producer(*id))
                    .unwrap_or(*arg)
            } else {
                let tp = inputarg_types
                    .get(i)
                    .copied()
                    .filter(|t| *t != Type::Void)
                    .unwrap_or(Type::Ref);
                let typed_null = extract_concrete_typed_value(tp, PY_NULL);
                fail_arg_opref_for_typed_value(ctx, typed_null)
            };
        }
        let before_dedup = args.clone();
        // pyjitpl.py `remove_consts_and_duplicates` over reds, then over
        // `virtualizable_boxes[:-1]`, sharing one `duplicates` set.
        // `args` is already that concatenated list (`frame` + extra reds
        // + `virtualizable_boxes[:-1]`), so one pass matches both calls.
        ctx.remove_consts_and_duplicates_untyped(&mut args);
        let dedup_changed: Vec<(usize, OpRef)> = args
            .iter()
            .enumerate()
            .filter(|(i, opref)| before_dedup[*i] != **opref)
            .map(|(i, opref)| (i, *opref))
            .collect();
        // `remove_consts_and_duplicates` mutates `virtualizable_boxes[:-1]`
        // in place. `args[1+extra_reds..]` is that element block.
        let payload_start = 1 + extra_reds;
        if let Some(boxes) = ctx.collect_virtualizable_boxes() {
            let end = boxes.len().saturating_sub(1);
            if args.len() >= payload_start + end {
                ctx.adopt_normalized_virtualizable_elements(
                    &args[payload_start..payload_start + end],
                );
            }
        }
        // pyjitpl.py put_back_list_of_boxes3: write dedup'd values back
        // to frame symbolic state so subsequent tracing sees the SameAs-wrapped
        // identities. RPython runs this on the "continue tracing" path after
        // reached_loop_header returns without closing. Harmless on the "close
        // loop" path since the frame won't be reused.
        {
            // `num_scalars` (NUM_SCALAR_INPUTARGS) already counts extra_reds
            // (frame + extra_reds + vable static fields).
            let total_scalar_prefix = num_scalars;
            let s = self.sym_mut();
            for &(idx, new_opref) in &dedup_changed {
                if idx < total_scalar_prefix {
                    match idx {
                        0 => s.frame = new_opref,
                        // NUM_EXTRA_REDS == 1 (crate const-assert).
                        1 => s.execution_context = new_opref,
                        _ => match idx - extra_reds {
                            1 => s.vable_last_instr = new_opref,
                            2 => s.vable_pycode = new_opref,
                            3 => s.vable_valuestackdepth = new_opref,
                            4 => s.vable_debugdata = new_opref,
                            _ => {}
                        },
                    }
                } else {
                    let local_idx = idx - total_scalar_prefix;
                    // `registers_r` is the unified abstract register
                    // file; locals + stack tail share the same addr
                    // space, so the dedup'd rename writes to the single
                    // slot regardless of whether the dedup refers to a
                    // local or a stack entry.
                    if local_idx < s.registers_r.len() {
                        s.registers_r.set(local_idx, new_opref);
                    }
                }
            }
        }
        // pyjitpl.py:2967-2969: generate a dummy GUARD_FUTURE_CONDITION
        // just before the JUMP so that unroll can use it when it's
        // creating artificial guards (patchguardop). record_guard calls
        // capture_resumedata which captures the full framestack +
        // virtualizable_boxes + virtualref_boxes.
        //
        // RPython only emits GUARD_FUTURE_CONDITION here. GUARD_NOT_INVALIDATED
        // is *not* unconditionally emitted before JUMP — pyjitpl.py
        // emits it only inside `opimpl_record_quasi_immutable_field`, after a
        // quasi-immut field read sets `heapcache.need_guard_not_invalidated`.
        // The pyre frontend does the same via `flush_guard_not_invalidated`,
        // so an unconditional emit here would (a) leak resume data for traces
        // that have no quasi-immut dep at all, and (b) leave a runtime guard
        // whose flag is decoupled from any watcher, which can spuriously
        // exit a hot inner loop with no chance of re-tracing.
        //
        // RPython parity: orgpc must be the loop header TARGET, not the
        // JUMP_BACKWARD's PC. The patchguardop from this GuardFutureCondition
        // provides the resume_position for all peeled body virtual state guards.
        // If orgpc is wrong, all those guards resume at the wrong PC.
        if let Some(pc) = target_pc {
            self.orgpc = pc;
        }
        self.generate_guard(ctx, majit_ir::OpCode::GuardFutureCondition, &[]);
        // pyjitpl.py:2995 assert len(self.virtualref_boxes) == 0,
        //     "missing virtual_ref_finish()?"
        // Reached loop header must not have dangling virtualrefs — every
        // `enter` must have been matched by a `leave`, or one closed by
        // vrefs_after_residual_call / stop_tracking_virtualref.
        debug_assert_eq!(
            ctx.virtualref_boxes_len(),
            0,
            "missing virtual_ref_finish()? close_loop_args_at reached with open virtualrefs"
        );
        // Verify `live_args_shape_at` formula matches actual output.
        // If this fires, the helper's shape derivation is stale relative
        // to `close_loop_args_at`'s args layout — update both in lockstep.
        debug_assert_eq!(
            args.len(),
            self.live_args_shape_at(ctx),
            "live_args_shape_at must predict close_loop_args_at output length",
        );
        // virtualstate.py get_runtime_item — populate the `Box.value` stamp
        // so the optimizer can route `cpu.cls_of_box(runtime_box)` /
        // `runtime_box.getref_base()` through the materialised BoxRef's
        // per-type mixin slot. Writes go into `ctx.opref_concrete`; the
        // optimizer stamps them onto BoxRefs before virtualstate matching.
        //
        // virtualstate.py generate_guards requires `runtime_boxes` to be fully
        // parallel with `boxes`. Int and float array slots take
        // `concrete_value_at`. A ref takes the virtualizable shadow for
        // that slot, which a minor collection has already forwarded.
        // The frame, ec, and vable scalars are stamped below:
        //   args[0]                                ↔ frame (raw ptr)
        //   args[1..1+extra_reds]                  ↔ ec (raw ptr)
        //   args[1+extra_reds..num_scalars]        ↔ vable scalars (shadow)
        //   args[num_scalars + i]                  ↔ locals slot i
        //   args[num_scalars + nlocals + j]        ↔ stack slot nlocals+j
        // `num_scalars` already counts `extra_reds` (per dedup loop above).
        {
            let header_off = num_scalars;
            // The runtime-value walk uses the symbolic stack range so the
            // locals/stack slot indices align with `args[header_off + ..]`.
            let stack_only = self
                .sym()
                .valuestackdepth
                .saturating_sub(self.sym().nlocals);
            let record = |ctx: &mut TraceCtx, opref: OpRef, value: majit_ir::Value| {
                if opref != OpRef::NONE && !opref.is_constant() {
                    ctx.set_opref_concrete(opref, value);
                }
            };
            // args[0] frame
            let frame_addr = self.concrete_frame_addr;
            if frame_addr != 0 {
                if let Some(&opref) = args.first() {
                    record(
                        ctx,
                        opref,
                        majit_ir::Value::Ref(majit_ir::GcRef(frame_addr)),
                    );
                }
            }
            // args[1..1+extra_reds] ec — NUM_EXTRA_REDS == 1.
            let ec_ptr = self.sym().concrete_execution_context as usize;
            if ec_ptr != 0 {
                if let Some(&opref) = args.get(1) {
                    record(ctx, opref, majit_ir::Value::Ref(majit_ir::GcRef(ec_ptr)));
                }
            }
            // args[1+extra_reds..num_scalars] vable static fields — the
            // shadow `ctx.virtualizable_entry_at(i)` tracks the JIT's
            // current belief about each vable scalar, kept in sync with
            // `s.vable_*` OpRefs across setfield_vable updates.
            let vable_start = 1 + extra_reds;
            let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            for i in 0..nvs {
                let slot_idx = vable_start + i;
                let Some(&opref) = args.get(slot_idx) else {
                    break;
                };
                if let Some((shadow_opref, value)) = ctx.virtualizable_entry_at(i) {
                    if shadow_opref == opref && ctx.box_carries_runtime_concrete(opref) {
                        record(ctx, opref, value);
                    }
                }
            }
            // Locals/stack InputArgs are the vable array boxes. Stamping them
            // from `concrete_locals` (seed-time, not updated by setarrayitem)
            // overwrote `_res*` after the last vable store and made
            // `flush_walk_end_state_to_frame` read a stale W_Root.
            // Same as the scalar loop: the box already carries
            // `virtualizable_entry_at`.
            for i in 0..nlocals {
                let slot_idx = header_off + i;
                if let Some(&opref) = args.get(slot_idx) {
                    let vable_idx = nvs + i;
                    if let Some((slot_op, value)) = ctx.virtualizable_entry_at(vable_idx) {
                        if slot_op == opref {
                            if ctx.box_carries_runtime_concrete(opref) {
                                record(ctx, opref, value);
                            }
                            continue;
                        }
                    }
                    if let Some(v) = self.loop_carried_runtime_value(ctx, opref, vable_idx, i) {
                        record(ctx, opref, v);
                    }
                }
            }
            for j in 0..stack_only {
                let slot_idx = header_off + nlocals + j;
                if let Some(&opref) = args.get(slot_idx) {
                    let vable_idx = nvs + nlocals + j;
                    if let Some((slot_op, value)) = ctx.virtualizable_entry_at(vable_idx) {
                        if slot_op == opref {
                            if ctx.box_carries_runtime_concrete(opref) {
                                record(ctx, opref, value);
                            }
                            continue;
                        }
                    }
                    if let Some(v) =
                        self.loop_carried_runtime_value(ctx, opref, vable_idx, nlocals + j)
                    {
                        record(ctx, opref, v);
                    }
                }
            }
        }
        self.loop_close_marker_jit_pc = None;
        self.loop_close_registers_r = Vec::new();
        self.close_merge_point_vsd = None;
        args
    }

    /// pyjitpl.py capture_resumedata: build fail_args for CURRENT
    /// top frame. Returns the scalar header plus active_boxes —
    /// `[frame, (ec)?, last_instr, pycode, valuestackdepth, debugdata,
    /// active_boxes...]` — matching
    /// `interp_jit.py PyFrame._virtualizable_` /
    /// `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS` line-by-line.
    /// `NUM_EXTRA_REDS` controls whether the ec slot
    /// (interp_jit.py `reds = ['frame', 'ec']`) is present between
    /// frame and the vable static fields. Dormant under
    /// NUM_EXTRA_REDS=0 (skips ec push, preserves pre-ec 7-scalar
    /// layout). virtualizable.py read_boxes: all static fields in
    /// order.
    pub(crate) fn current_fail_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes = self.get_list_of_active_boxes(ctx, false, false, None);
        // [frame, ec] portal-reds contract. Recover ec before snapshotting
        // sym fields so guard fail_args never carry OpRef::NONE in the ec
        // slot (adapter/bridge-from-guard paths).
        let ec = self.ensure_execution_context(ctx);
        let s = self.sym();
        let mut fa =
            Vec::with_capacity(crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + active_boxes.len());
        fa.push(s.frame);
        // NUM_EXTRA_REDS == 1 (crate const-assert in `lib.rs`).
        // `interp_jit.py reds = ['frame', 'ec']`.
        fa.push(ec);
        fa.extend_from_slice(&[
            s.vable_last_instr,
            s.vable_pycode,
            s.vable_valuestackdepth,
            s.vable_debugdata,
        ]);
        fa.extend_from_slice(&active_boxes);
        fa
    }

    /// pyjitpl.py:1087 parity: after a field read that might have set the
    /// needs_guard_not_invalidated flag (quasi-immutable field), emit the
    /// guard with full snapshot via record_guard.
    pub(crate) fn flush_guard_not_invalidated(&mut self, ctx: &mut TraceCtx) {
        if let Some(saved_orgpc) = ctx.pending_guard_not_invalidated_pc() {
            ctx.set_pending_guard_not_invalidated(None);
            // pyjitpl.py:1087 parity: use the field read's orgpc so the
            // snapshot captures the correct liveness state.
            let current_orgpc = self.orgpc;
            self.orgpc = saved_orgpc;
            self.generate_guard(ctx, OpCode::GuardNotInvalidated, &[]);
            self.orgpc = current_orgpc;
        }
    }

    pub(crate) fn generate_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
        // pyjitpl.py generate_guard parity:
        //     if isinstance(box, Const):    # no need for a guard
        //         return
        // The first arg of every data guard (GUARD_CLASS, GUARD_TRUE,
        // GUARD_NONNULL, GUARD_VALUE, ...) is the box being checked.
        // Control-flow guards (GUARD_NOT_FORCED, GUARD_NO_OVERFLOW,
        // GUARD_NOT_INVALIDATED, ...) call generate_guard with `args=&[]`,
        // so `args.first()` is None and the check is skipped — matching
        // RPython where `box=None` for those guards.
        if let Some(&first) = args.first() {
            if first.is_constant() {
                return;
            }
        }
        // pyjitpl.py:1087 parity: flush pending guard_not_invalidated
        // before recording any new guard (the quasi-immut guard should be
        // emitted with its own snapshot before the current guard).
        if opcode != OpCode::GuardNotInvalidated {
            self.flush_guard_not_invalidated(ctx);
        }
        // pyjitpl.py:2575-2578: determine after_residual_call from guard opcode.
        // opencoder.py create_top_snapshot: when true, all boxes in top frame are live
        // (liveness filter disabled for residual call guards).
        let after_residual_call = matches!(
            opcode,
            OpCode::GuardException
                | OpCode::GuardNoException
                | OpCode::GuardNotForced
                | OpCode::GuardAlwaysFails
        );
        if after_residual_call {
            // pyjitpl.py:2586-2602: residual-call guards snapshot the state
            // AFTER the call, using the auto-advanced pc and post-call
            // register file. The opcode-start snapshot is only for
            // re-executing the current opcode from orgpc.
            self.clear_pre_opcode_state();
        }
        // pyjitpl.py capture_resumedata(resumepc) parity:
        // Normal guards: resumepc = orgpc (re-execute the opcode from start).
        // after_residual_call guards (GUARD_NOT_FORCED, GUARD_NO_EXCEPTION):
        //   RPython generate_guard passes resumepc=-1, and capture_resumedata
        //   skips the "frame.pc = resumepc" assignment — frame.pc stays at
        //   the auto-advanced next instruction (pyre fallthrough_pc equivalent).
        //   This ensures the liveness PC, header ni, and blackhole resume PC
        //   all point to the instruction AFTER the call, not the call itself.
        let resume_pc = if after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        self.generate_guard_core(ctx, opcode, args, resume_pc, after_residual_call);
    }

    /// Core guard recording with explicit resume PC.
    ///
    /// pyjitpl.py generate_guard parity: record guard op,
    /// then call capture_resumedata.
    fn generate_guard_core(
        &mut self,
        ctx: &mut TraceCtx,
        opcode: OpCode,
        args: &[OpRef],
        resume_pc: usize,
        after_residual_call: bool,
    ) {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes =
            self.get_list_of_active_boxes(ctx, false, after_residual_call, Some(self.orgpc));
        let snapshot_full_types = self.build_fail_arg_types_for_active_boxes(&active_boxes);

        // Snapshot is the source of truth — the
        // optimizer's `store_final_boxes_in_guard`
        // (`optimizeopt/mod.rs`) overwrites `op.fail_args` from the
        // snapshot built below via `op.store_final_boxes(liveboxes)`
        // (`resoperation.rs`). `generate_guard` records the guard with
        // no inline fail_args; `capture_resumedata` + `_number_boxes`
        // populate them from the snapshot chain. Types come from the
        // live boxes at `store_final_boxes` (`compile.py`).
        ctx.record_guard(opcode, args, 0);

        // pyjitpl.py: self.capture_resumedata(resumepc, after_residual_call)
        self.capture_resumedata(
            ctx,
            resume_pc,
            after_residual_call,
            &active_boxes,
            &snapshot_full_types,
        );
        // pyjitpl.py `count_ops(opnum, Counters.GUARDS)` is bumped
        // inside `TraceCtx::record_guard` (the record chokepoint),
        // so no explicit count here.
    }

    /// pyjitpl.py capture_resumedata parity.
    ///
    /// Temporarily sets frame.pc = resumepc, captures this frame plus
    /// virtualizable_boxes + virtualref_boxes into a snapshot, then
    /// restores frame.pc.  Matches opencoder.py
    /// `capture_resumedata(framestack, virtualizable_boxes,
    /// virtualref_boxes, after_residual_call=False)`.
    fn capture_resumedata(
        &mut self,
        ctx: &mut TraceCtx,
        resume_pc: usize,
        after_residual_call: bool,
        active_boxes: &[OpRef],
        snapshot_full_types: &[Type],
    ) {
        // pyjitpl.py:2594-2596: saved_pc = frame.pc; frame.pc = resumepc
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_last_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        // This retired MIFrame trait-interpret leg has no production driver.
        // Production guard capture carries the genuine post-call jitcode
        // coordinate; preserve the trait-leg after-residual decline without
        // rewriting the resume word.
        if after_residual_call {
            crate::state::request_trace_abort();
        }
        let snapshot_live_pc = saved_orgpc;

        // pyjitpl.py: history.trace.capture_resumedata(
        //     self.framestack, virtualizable_boxes, self.virtualref_boxes,
        //     after_residual_call)
        let snapshot = self.build_framestack_snapshot(
            ctx,
            snapshot_live_pc,
            active_boxes,
            snapshot_full_types,
        );
        let snapshot_id = ctx.capture_resumedata(snapshot);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_last_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// Build the single-frame `Snapshot` — this frame plus virtualizable
    /// and virtualref boxes.  Mirrors opencoder.py capture_resumedata
    /// `capture_resumedata(framestack, virtualizable_boxes,
    /// virtualref_boxes, ...)`.
    ///
    /// The caller is responsible for swapping `self.orgpc` if the
    /// snapshot pc differs from the current orgpc (RPython
    /// pyjitpl.py MIFrame.capture_resumedata does the same
    /// before calling `history.trace.capture_resumedata`), and for
    /// computing `top_active_boxes` / `top_snapshot_types_full` under
    /// the liveness that applies to the swapped pc.
    fn build_framestack_snapshot(
        &mut self,
        ctx: &mut TraceCtx,
        top_pc: usize,
        top_active_boxes: &[OpRef],
        top_snapshot_types_full: &[Type],
    ) -> majit_metainterp::recorder::Snapshot {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let top_snapshot_types = &top_snapshot_types_full[n..];
        let top_jitcode_index = unsafe { (*self.sym().jitcode).index } as u32;
        let top_word = self
            .loop_close_marker_jit_pc
            .map(|m| m as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let payload = unsafe { &(&*self.sym().jitcode).payload };
        let resolved = payload.resolve_resume_pc_with_jitcode_pc(top_word, crate::state::op_live());
        let top_pc_word = resolved.map(|offset| offset as u32).unwrap_or_else(|| {
            // A missing carried marker declines this trace before the
            // provisional snapshot can be installed.
            crate::state::request_trace_abort();
            top_pc as u32
        });
        // Decline a resolved JitCode pc that has no forward Python-pc
        // marker. Resume recovers py_pc from the jitcode word; the
        // recorder does not store a copy.
        if let Some(offset) = resolved
            && payload.resume_position_for_jitcode_pc(offset).is_none()
        {
            crate::state::request_trace_abort();
        }
        let top_frame = majit_metainterp::recorder::SnapshotFrame {
            jitcode_index: top_jitcode_index,
            pc: top_pc_word,
            boxes: Self::fail_args_to_snapshot_boxes_typed(
                top_active_boxes,
                top_snapshot_types,
                ctx,
            ),
        };
        // Single-frame snapshot: the walker records one frame per guard; the
        // multi-frame parent chain was the retired trait-interpret leg.
        let frames = vec![top_frame];
        let vable_boxes = self.list_of_boxes_virtualizable(ctx);
        // `pyjitpl.py:2623` passes `self.virtualref_boxes` to the snapshot
        // alongside the virtualizable boxes; the vable half here comes from
        // the trait leg's own mirror, so take only the vref half.
        let (_, vref_boxes) = ctx.build_snapshot_vable_vref_boxes();
        // PHASE 1.4 candidate D probe: detect snapshot-time divergence
        // between vable_boxes (heap mirror) and registers_r (machine
        // register source). Both should be populated by store_local_value's
        // dual-write. Any divergence here means a code path updated one
        // shadow without the other — most likely load_local_value's lazy
        // fallback (trace_opcode.rs) which writes registers_r
        // but does NOT call set_virtualizable_box_at.
        if std::env::var("PYRE_PROBE_SNAPSHOT").ok().as_deref() == Some("1") {
            let num_static = ctx
                .virtualizable_info()
                .map(|info| info.num_static_extra_boxes)
                .unwrap_or(0);
            let nlocals = self.sym().nlocals;
            let registers_r_src: Vec<OpRef> = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r[..pre_r.len().min(nlocals)].to_vec()
            } else {
                let s = self.sym();
                s.registers_r.iter().take(nlocals).collect()
            };
            let mut diverge = 0usize;
            for i in 0..registers_r_src.len() {
                let reg_op = registers_r_src[i];
                let vable_op = ctx
                    .virtualizable_box_at(num_static + i)
                    .unwrap_or(OpRef::NONE);
                if !reg_op.is_none() && reg_op != vable_op {
                    eprintln!(
                        "[PROBE-D] vable/reg divergence top_pc={} local={} reg_opref={:?} vable_opref={:?}",
                        top_pc, i, reg_op, vable_op
                    );
                    diverge += 1;
                }
            }
            eprintln!(
                "[PROBE-D] ENTER top_pc={} nlocals={} reg_len={} num_static={} diverge_count={}",
                top_pc,
                nlocals,
                registers_r_src.len(),
                num_static,
                diverge
            );
        }
        majit_metainterp::recorder::Snapshot {
            resume_position: -1,
            frames,
            vable_boxes,
            vref_boxes,
        }
    }

    /// virtualizable.py load_list_of_boxes _get_virtualizable_field_boxes parity:
    /// [static_fields..., array_items..., virtualizable_ptr].
    /// pyjitpl.py:2586: self.virtualizable_boxes → vable_array.
    /// opencoder.py _encode parity: encode OpRef as SnapshotTagged.
    /// Constant-pool OpRefs → Const(value, type) from pool.
    /// NONE → Const(0, Ref). Regular → Box.
    fn opref_to_snapshot_tagged(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
    ) -> majit_metainterp::recorder::SnapshotTagged {
        Self::opref_to_snapshot_tagged_for_slot(opref, ctx, None)
    }

    /// virtualizable.py `read_boxes(cpu, virtualizable, startindex)` parity:
    /// each slot is wrapped via `wrap(cpu, value, startindex + i)` where the
    /// lltype (ARRAYITEMTYPE or static field `FIELDTYPE`) is declared and
    /// determines the resulting Const's INT/REF/FLOAT kind. pyre stores
    /// constants in a unified pool whose stored `const_type` may disagree
    /// with the slot's declared type (e.g. pointer constants opened via
    /// `const_int`), so snapshot encoding must prefer the slot's declared
    /// type when it is known — otherwise `_gettagged` → `getconst(val, tp)`
    /// picks TAGINT for a Ref-typed slot and the resume reader decodes a
    /// raw i64 where a PyObjectRef is expected.
    fn opref_to_snapshot_tagged_for_slot(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
        declared_type: Option<majit_ir::Type>,
    ) -> majit_metainterp::recorder::SnapshotTagged {
        if opref.is_none() {
            majit_metainterp::recorder::SnapshotTagged::Const(
                0,
                declared_type.unwrap_or(majit_ir::Type::Ref),
            )
        } else if ctx.constant_value(opref).is_some() {
            if let Some(index) = opref.const_ptr_index() {
                return majit_metainterp::recorder::SnapshotTagged::Const(
                    i64::from(index),
                    majit_ir::Type::Ref,
                );
            }
            let val = ctx.constant_value(opref).unwrap_or(0);
            // resume.py `getconst(const)` dispatches on `const.type`.
            // Prefer the pool's actual const type over `declared_type`:
            // Box.type is immutable, so an Int-typed constant (e.g. an
            // intbounds-promoted local) must stay Int even when the slot
            // layout declares Ref. Retyping it here would seed the bridge
            // optimizer's const_pool with `Value::Ref(GcRef(small_int))`
            // and later trip the getintbound forwarding assertion. Fall
            // back to `declared_type` only when the pool has no type for
            // this OpRef (e.g. raw-pointer constants seeded without a
            // const_type entry).
            let tp = ctx
                .const_type(opref)
                .or(declared_type)
                .unwrap_or(majit_ir::Type::Int);
            majit_metainterp::recorder::SnapshotTagged::Const(val, tp)
        } else {
            // resume.py:211,214: box.type lives on the Box itself; the
            // typed `OpRef` carries the matching variant tag and the
            // explicit `tp` is the lockstep authority for any
            // transitional `Untyped` opref (resoperation.py InputArgInt/727/739).
            let tp = ctx
                .get_opref_type(opref)
                .unwrap_or_else(|| panic!("missing snapshot box type for {:?}", opref));
            majit_metainterp::recorder::SnapshotTagged::Box(opref, tp)
        }
    }

    /// RPython pyjitpl.py:2586 virtualizable_boxes parity.
    ///
    /// RPython creates SEPARATE Box objects for virtualizable_boxes via
    /// read_boxes()/wrap() — these are distinct from frame register boxes.
    /// _number_boxes dedup uses object identity, so vable and frame get
    /// independent TAGBOX indices → deadframe stores both.
    ///
    /// pyre uses the SAME OpRefs for both → _number_boxes dedup merges them
    /// → vable and frame sections share TAGBOX indices. Recovery uses frame
    /// sections with liveness-based mapping (restore_guard_failure_values),
    /// matching RPython's consume_boxes(position_info) architecture.
    ///
    /// Fresh identity approaches (VABLE_FRESH_BIT, VABLE_KEY_OFFSET)
    /// expand num_boxes → larger fail_args → deadframe/exit layout mismatch.
    /// Fix requires backend exit block recompilation after numbering,
    /// or trace-time SameAs emission for fresh vable OpRefs.
    fn list_of_boxes_virtualizable(
        &self,
        ctx: &mut majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        let sym = self.sym();
        // opencoder.py _list_of_boxes_virtualizable parity:
        // RPython format: [virtualizable_ptr, static_fields..., array_items...]
        // (virtualizable_ptr moved from end to front).
        // virtualizable.py/139 read_boxes / load_list_of_boxes:
        // Memory order: [static_field_0, ..., array_items..., vable_ptr]
        // read_boxes creates fresh Box objects for each field via wrap().
        // opencoder.py _list_of_boxes_virtualizable: reorders
        //   vable_ptr from end to front → snapshot = [vable_ptr, fields..., items...]
        let stack_only = sym.valuestackdepth.saturating_sub(sym.nlocals);
        let mut boxes = Vec::new();
        // opencoder.py:722: virtualizable_ptr FIRST.
        // The virtualizable frame pointer is always a GCREF.
        //
        // RPython parity: the vable identity is the virtualizable OWNER
        // (portal) frame — `metainterp.virtualizable_boxes[-1]` — recorded once
        // at toplevel, NOT the current frame. For an inlined callee `sym` (the
        // separate-inline-frame path), `sym.frame` is the callee frame, whose
        // heap `locals_cells_stack_w` length differs from the owner frame's;
        // the static-field count and array length below are sourced from the
        // owner (`ctx.virtualizable_*`), so using `sym.frame` here makes the
        // decoder's `get_total_size(virtualizable)` read the callee's shorter
        // array and trip `consume_vable_info` (vable_size-1 mismatch). Source
        // the identity from the seeded owner; fall back to `sym.frame` only in
        // the unseeded test path.
        let identity_opref = ctx.virtualizable_owner_identity().unwrap_or(sym.frame);
        boxes.push(Self::opref_to_snapshot_tagged_for_slot(
            identity_opref,
            ctx,
            Some(majit_ir::Type::Ref),
        ));
        // Static fields in declared order (virtualizable.py:90-93).
        // virtualizable.py:131-133 wraps each value with its declared
        // `FIELDTYPE`; pyre mirrors that by consulting
        // `VirtualizableInfo::static_fields[i].field_type`.
        //
        // opencoder.py `_list_of_boxes_virtualizable(boxes)`
        // parity: read from `ctx.virtualizable_boxes` (the canonical
        // analog of RPython's `metainterp.virtualizable_boxes`) for
        // the three invariant scalars (`pycode`, `debugdata`,
        // `w_globals`), and recompute the two
        // per-opcode-advancing scalars (`last_instr`, `valuestackdepth`)
        // from `self.orgpc` / `pre_opcode_registers_r` so the snapshot encodes
        // the pre-opcode state at `resume_pc` (the PROBE-VABLE-DIV
        // diagnostic confirmed slot 0 / slot 2 are the
        // only divergence sources between the shared shadow and
        // `s.vable_*` — slots 1/3/4 always agree because their
        // mutators are unreachable under pyre's 3.14 bytecode).
        //
        // The slot-0 inline override re-derives `resume_pc - 1`
        // because `flush_to_frame_for_guard` swaps `self.orgpc` to
        // `resume_pc` (capture_resumedata at line 2773), and writes
        // `s.vable_last_instr = const_int(resume_pc - 1)` without
        // mirroring to `ctx.virtualizable_boxes[0]`.  Reading
        // `ctx.virtualizable_boxes[0]` directly would pick up the
        // value `publish_last_instr_to_vable` wrote at the original
        // (pre-swap) orgpc, which is one off when `resume_pc !=
        // orgpc` (most branch guards).  The read-time recompute
        // matches `flush_to_frame_for_guard` so the snapshot stays
        // self-consistent.
        //
        // The slot-2 inline override re-derives the pre-opcode
        // valuestackdepth from the same source `flush_to_frame_for_guard`
        // uses to set `s.vable_valuestackdepth`.
        //
        // Test-fixture fallback: `TraceCtx::for_test_types` callers
        // construct a ctx without registering `VirtualizableInfo` and
        // without seeding `ctx.virtualizable_boxes`.  In that mode
        // the `_with_compiled_trace_jitcode` fixtures expect
        // `sym.vable_field_oprefs()` to drive the snapshot — fall
        // back when the shared shadow is unseeded.
        let (vable_static_types, vsd_field_index, ni_field_index, num_static): (
            Vec<majit_ir::Type>,
            Option<usize>,
            Option<usize>,
            usize,
        ) = match ctx.virtualizable_info() {
            Some(info) => (
                info.static_fields.iter().map(|f| f.field_type).collect(),
                info.static_field_index_by_name("valuestackdepth"),
                info.static_field_index_by_name("last_instr"),
                info.num_static_extra_boxes,
            ),
            None => (Vec::new(), None, None, 0),
        };
        let pre_opcode_vsd: Option<i64> = if vsd_field_index.is_some() {
            let resume_pc = self.orgpc;
            Some(self.pre_opcode_concrete_depth() as i64)
        } else {
            None
        };
        let pre_opcode_last_instr: Option<i64> = ni_field_index.map(|_| self.orgpc as i64 - 1);
        if num_static > 0 && ctx.has_virtualizable_boxes() {
            for idx in 0..num_static {
                let declared = vable_static_types.get(idx).copied();
                let opref = if Some(idx) == ni_field_index {
                    let li = pre_opcode_last_instr
                        .expect("pre_opcode_last_instr seeded when ni_field_index is Some");
                    ctx.const_int(li)
                } else if Some(idx) == vsd_field_index {
                    let vsd =
                        pre_opcode_vsd.expect("pre_opcode_vsd seeded when vsd_field_index is Some");
                    ctx.const_int(vsd)
                } else {
                    ctx.virtualizable_box_at(idx).unwrap_or(OpRef::NONE)
                };
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    opref, ctx, declared,
                ));
            }
        } else {
            // Test fallback: ctx has no vinfo and no seeded shadow.
            let fallback_types: Vec<majit_ir::Type> = if !vable_static_types.is_empty() {
                vable_static_types.clone()
            } else {
                vec![majit_ir::Type::Ref; sym.vable_field_oprefs().len()]
            };
            for (idx, opref) in sym.vable_field_oprefs().iter().enumerate() {
                let declared = fallback_types.get(idx).copied();
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    *opref, ctx, declared,
                ));
            }
        }
        // Array items: locals + stack (virtualizable.py read_boxes).
        let _ = stack_only;
        let symbolic_stack_len = if self.pre_opcode_registers_r.is_some() {
            self.pre_opcode_concrete_depth().saturating_sub(sym.nlocals)
        } else {
            sym.registers_r.len().saturating_sub(sym.nlocals)
        };
        let concrete_frame_ptr = if !sym.concrete_vable_ptr.is_null() {
            sym.concrete_vable_ptr as usize
        } else {
            self.concrete_frame_addr
        };
        let concrete_frame = if concrete_frame_ptr != 0 {
            Some(unsafe { &*(concrete_frame_ptr as *const pyre_interpreter::pyframe::PyFrame) })
        } else {
            None
        };
        // virtualizable.py read_boxes parity: encoder must emit one
        // box per slot in the heap-side `locals_cells_stack_w` array
        // because the decoder reads `vinfo.get_total_size(virtualizable)`
        // (= static_fields + heap array length) on the runtime PyFrame.
        // Using the symbolic current stack depth here was off by
        // (max_stackdepth - current_stack_depth) and produced
        // `vable_size - 1 != vinfo.get_total_size` panics whenever a
        // bridge tried to consume the snapshot at a state where the
        // physical frame had been allocated with stack room beyond the
        // current symbolic depth. Read the physical frame length and
        // pad missing slots with the live concrete value (or NULL).
        let physical_array_len = ctx
            .virtualizable_array_lengths()
            .and_then(|lengths| lengths.first().copied())
            .or_else(|| concrete_frame.map(|f| locals_w!(f).len()))
            .unwrap_or_else(|| {
                if !sym.jitcode.is_null() {
                    let code = unsafe { &*(*sym.jitcode).raw_code() };
                    code.varnames.len()
                        + pyre_interpreter::pyframe::ncells(code)
                        + code.max_stackdepth as usize
                } else {
                    let current_vsd = self.pre_opcode_concrete_depth();
                    let stack_depth = current_vsd
                        .saturating_sub(sym.nlocals)
                        .min(symbolic_stack_len);
                    sym.nlocals + stack_depth
                }
            });
        let full_array_len = physical_array_len;
        // virtualizable.py:135-137 `lst[j] = reader.load_next_value_of_type(
        // ARRAYITEMTYPE)` — every array slot is the array's declared item
        // type (GCREF for pyre's `locals_cells_stack_w`), regardless of
        // what the optimizer chose for the OpRef's own kind. Enforce this
        // at encoding time so a `LOAD_CONST 0` whose OpRef is Int-typed
        // still lands in the snapshot as a Ref constant.
        let array_item_type = ctx
            .virtualizable_info()
            .and_then(|info| info.array_fields.first().map(|a| a.item_type))
            .unwrap_or(majit_ir::Type::Ref);
        // virtualizable.py read_boxes / opencoder.py _list_of_boxes_virtualizable
        // parity: RPython snapshot reads directly from `self.virtualizable_boxes`,
        // which is the single source of truth mirrored by every
        // `_opimpl_setarrayitem_vable` via `synchronize_virtualizable`.
        // pyre's tracer mirrors EVERY write into `locals_cells_stack_w`
        // (locals via `store_local_value`, stack via `push_typed_value`
        // + `pop_value` + `swap_values` + `finishframe_exception`) into
        // `virtualizable_boxes`, so the shadow is the only source we
        // read here.
        let num_static = ctx
            .virtualizable_info()
            .map(|info| info.num_static_extra_boxes)
            .unwrap_or(0);
        // `popvalue_maybe_none` nulls every popped slot, so `read_boxes`
        // wraps None for index >= valuestackdepth. Encode that CONST_NULL
        // here: a leftover TAGBOX would name a jitframe slot that is dead
        // in compiled code and missing from the COND_CALL gcmap, and
        // `write_from_resume_data_partial` would copy the stale word into
        // the type-9 array. `close_loop_args_at` already force-nulls the
        // same JUMP tail.
        let live_array_depth = pre_opcode_vsd.map(|v| v as usize);
        for i in 0..full_array_len {
            // virtualizable_boxes layout:
            //   [field0, ..., fieldN, arr[0..M], vable_ref]
            // so array slot `i` lives at `num_static + i`. The trailing
            // `vable_ref` is at `virtualizable_boxes[-1]` and NEVER covers
            // an array slot — skip it via `.get()`.
            if live_array_depth.is_some_and(|vsd| i >= vsd) {
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    OpRef::NONE,
                    ctx,
                    Some(array_item_type),
                ));
                continue;
            }
            let opref = ctx
                .virtualizable_box_at(num_static + i)
                .unwrap_or(OpRef::NONE);
            if !opref.is_none() {
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    opref,
                    ctx,
                    Some(array_item_type),
                ));
            } else {
                // TODO: legacy `register == PyFrame
                // slot` conflation lets STORE_FAST
                // write an unboxed int into `locals_cells_stack_w[i]`
                // when the trace IR optimizer promoted that local's
                // OpRef to Int. Reading those raw bits back here and
                // const-seeding them as `Const(val, Ref)` mistypes
                // the value in the bridge optimizer's const_pool —
                // verified root cause of the LoadFastLoadFast vable
                // conversion regression.
                //
                // Emit a NULL sentinel; the bridge resume will
                // re-fetch the actual slot value via
                // `vable_getarrayitem_r` against the live frame.
                let _ = concrete_frame;
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    OpRef::NONE,
                    ctx,
                    Some(array_item_type),
                ));
            }
        }
        boxes
    }

    /// RPython pyjitpl.py get_list_of_active_boxes parity:
    #[allow(dead_code)]
    fn fail_args_to_snapshot_boxes(
        fail_args: &[OpRef],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        fail_args
            .iter()
            .map(|&opref| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// snapshot boxes from active_boxes = [locals, stack].
    /// RPython: each Box carries type ('r'/'i'/'f') — pyre passes types
    /// explicitly so _number_boxes can detect virtual vs int correctly.
    fn fail_args_to_snapshot_boxes_typed(
        active_boxes: &[OpRef],
        types: &[majit_ir::Type],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        active_boxes
            .iter()
            .enumerate()
            .map(|(i, &opref)| {
                if opref.is_none() {
                    majit_metainterp::recorder::SnapshotTagged::Const(0, majit_ir::Type::Ref)
                } else if let Some(index) = opref.const_ptr_index() {
                    majit_metainterp::recorder::SnapshotTagged::Const(
                        i64::from(index),
                        majit_ir::Type::Ref,
                    )
                } else if ctx.constant_value(opref).is_some() {
                    let val = ctx.constant_value(opref).unwrap_or(0);
                    // resume.py `getconst(const)` dispatches on
                    // `const.type`; pyre's plain `const_int(v)` has an
                    // intrinsic INT type (see `opref_to_snapshot_tagged`).
                    let tp = ctx.const_type(opref).unwrap_or(majit_ir::Type::Int);
                    majit_metainterp::recorder::SnapshotTagged::Const(val, tp)
                } else {
                    let tp = types.get(i).copied().unwrap_or_else(|| {
                        panic!("missing fail-arg box type at index {} for {:?}", i, opref)
                    });
                    majit_metainterp::recorder::SnapshotTagged::Box(opref, tp)
                }
            })
            .collect()
    }

    /// pyjitpl.py implement_guard_value parity.
    /// executor.py constant_from_op(box): dispatches on box.type.
    #[allow(dead_code)]
    pub(crate) fn implement_guard_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        expected: i64,
    ) {
        let expected_ref = match self.value_type(value) {
            majit_ir::Type::Ref => ctx.const_ref(expected),
            _ => ctx.const_int(expected),
        };
        self.generate_guard(ctx, OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py: replace_box
        ctx.heap_cache_mut().replace_box(value, expected_ref);
    }

    /// RPython registers[idx] parity: read concrete value from Box arrays.
    fn concrete_at(&self, abs_idx: usize) -> Option<PyObjectRef> {
        let v = self.sym().concrete_value_at(abs_idx);
        if !v.is_null() {
            return Some(v.to_pyobj());
        }
        None
    }

    /// pyjitpl.py opimpl_guard_class
    pub(crate) fn guard_class(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        expected_type: *const PyType,
    ) {
        // heapcache.py: skip guard if class already known for this object
        if ctx.heap_cache().is_class_known(obj) {
            return;
        }
        // pyjitpl.py generate_guard parity:
        //     if isinstance(box, Const):    # no need for a guard
        //         return
        // The concrete value (and therefore its class) is known at trace
        // time, so the runtime type check is guaranteed to pass. RPython
        // also short-circuits before capture_resumedata, so no snapshot is
        // attached for the skipped guard. heapcache.class_now_known is
        // still called below — pyjitpl.py opimpl_guard_class invokes
        // it unconditionally after generate_guard.
        if obj.is_constant() {
            // pyjitpl.py:1087 parity: a pending GUARD_NOT_INVALIDATED from a
            // preceding quasi-immut field read must still be flushed even
            // when the type guard is skipped, otherwise the watcher and the
            // trace's quasi-immut dependency would be silently dropped.
            self.flush_guard_not_invalidated(ctx);
            ctx.heap_cache_mut().class_now_known(obj);
            return;
        }
        let expected_type_const = ctx.const_int(expected_type as usize as i64);
        // pyjitpl.py:1521 records GUARD_CLASS. The obj is non-null by
        // construction here (every caller passes a value-stack operand or a
        // freshly read object, the same invariant under which the codewriter
        // emits guard_class at jtransform.py handle_getfield_typeptr).
        // A genuinely null-fed class-guarded slot is rejected structurally by
        // the cross-loop-CUT abort in compile_loop_body, not by the guard form.
        // The optimizer strengthens a separately-recorded preceding GUARD_NONNULL
        // into GUARD_NONNULL_CLASS (rewrite.py:408-444 / optimize_guard_class).
        self.generate_guard(ctx, OpCode::GuardClass, &[obj, expected_type_const]);
        // heapcache.py: class_now_known sets class + nullity.
        ctx.heap_cache_mut().class_now_known(obj);
    }

    pub(crate) fn trace_guarded_int_payload(
        &mut self,
        ctx: &mut TraceCtx,
        int_obj: OpRef,
    ) -> OpRef {
        if self.value_type(int_obj) == Type::Int {
            return int_obj;
        }
        if pyre_object::tagged_int::CAN_BE_TAGGED {
            if let Some(Value::Ref(r)) = ctx.concrete_of_opref(int_obj) {
                if r != GcRef::NO_CONCRETE {
                    let o = r.as_usize() as PyObjectRef;
                    if !o.is_null() {
                        if pyre_object::tagged_int::is_tagged_int(o) {
                            let lowbit = crate::helpers::emit_tag_lowbit_test(ctx, int_obj, true);
                            self.generate_guard(ctx, OpCode::GuardTrue, &[lowbit]);
                            return crate::helpers::emit_untag_int(
                                ctx,
                                int_obj,
                                pyre_object::tagged_int::untag_int(o),
                            );
                        } else {
                            let lowbit = crate::helpers::emit_tag_lowbit_test(ctx, int_obj, false);
                            self.generate_guard(ctx, OpCode::GuardFalse, &[lowbit]);
                        }
                    }
                }
            }
        }
        self.guard_class(ctx, int_obj, &INT_TYPE as *const PyType);
        opimpl_getfield_gc_i(ctx, int_obj, int_intval_descr())
    }
}

/// Returns (is_int, is_float) for the fused-dispatch fuseability gate.
/// Mirrors the concrete-type classification codegen.rs uses to pick
/// between the int and float fast paths in the retired compare helper,
/// but reads the ConcreteValue variant directly so the check does not
/// have to allocate an intermediate w_int/w_float box.
#[allow(dead_code)]
fn classify_concrete(cv: ConcreteValue) -> (bool, bool) {
    match cv {
        ConcreteValue::Int(_) => (true, false),
        ConcreteValue::Float(_) => (false, true),
        ConcreteValue::Ref(obj) if !obj.is_null() => unsafe { (is_int(obj), is_float(obj)) },
        _ => (false, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_miframe<'a>(ctx: &'a mut TraceCtx, sym: &'a mut PyreSym) -> MIFrame {
        MIFrame {
            ctx,
            sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            loop_close_registers_r: Vec::new(),
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        }
    }

    #[cfg(feature = "cranelift")]
    fn clear_pending_jit_exception() {
        majit_backend_cranelift::jit_exc_raise(0);
    }

    #[cfg(all(not(feature = "cranelift"), feature = "dynasm"))]
    fn clear_pending_jit_exception() {
        majit_backend_dynasm::jit_exc_raise(0);
    }

    #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
    fn clear_pending_jit_exception() {}

    #[cfg(feature = "cranelift")]
    fn pending_jit_exception_raw() -> i64 {
        majit_backend_cranelift::jit_exc_value_raw()
    }

    #[cfg(all(not(feature = "cranelift"), feature = "dynasm"))]
    fn pending_jit_exception_raw() -> i64 {
        majit_backend_dynasm::jit_exc_value_raw()
    }

    #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
    fn pending_jit_exception_raw() -> i64 {
        0
    }

    #[test]
    fn exact_w_class_guard_skips_primitive_values() {
        let mut ctx = TraceCtx::for_test_types(&[Type::Int, Type::Float]);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        let mut frame = test_miframe(&mut ctx, &mut sym);

        let fake_type = 0x1234usize as PyObjectRef;
        let before = unsafe { &*frame.ctx }.num_ops();
        let frame_ptr: *mut MIFrame = &mut frame;
        let ctx_ptr = frame.ctx;
        unsafe {
            trace_guard_exact_w_class(
                &mut *frame_ptr,
                &mut *ctx_ptr,
                OpRef::input_arg_int(0),
                fake_type,
            );
            trace_guard_exact_w_class(
                &mut *frame_ptr,
                &mut *ctx_ptr,
                OpRef::input_arg_float(1),
                fake_type,
            );
        }

        assert_eq!(unsafe { &*frame.ctx }.num_ops(), before);
    }

    #[test]
    fn stack_slot_writers_stamp_the_box_when_seeded_without_concretes() {
        // An owner seeded with no live values still has boxes; a Ref push
        // stores the box and the box carries the concrete (`ConstPtr` inline /
        // `*FrontendOp._resref`). Production never reaches the unstamped
        // InputArg seed; this hardens that a write does not panic.
        let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let placeholder = ctx.const_ref(0);
        // Boxes cover stack slots 0 and 1 at `nlocals == 0`; the empty values
        // slice leaves InputArg/`*FrontendOp` boxes unstamped. These slots
        // are `ConstPtr(0)` so each already carries a concrete.
        let boxes = vec![placeholder; nvs + 4];
        ctx.install_virtualizable_info(info.clone());
        ctx.set_virtualizable_boxes_with_info(boxes, Vec::new(), &info, &[4]);

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.bridge_local_oprefs.replace(Vec::new()); // owns via the bridge half
        sym.nlocals = 0;
        sym.registers_r.replace(vec![OpRef::NONE; 2]); // swap's semantic mirror needs both slots
        assert!(sym.owns_virtualizable_shadow());

        let ptr = 0xdead_beef_usize as *mut pyre_object::pyobject::PyObject;
        let boxed0 = ctx.const_ref(0x1111);
        let boxed1 = ctx.const_ref(0x2222);
        write_stack_slot(&mut sym, &mut ctx, 0, boxed0, ConcreteValue::Ref(ptr));
        write_stack_slot(&mut sym, &mut ctx, 1, boxed1, ConcreteValue::Ref(ptr));
        assert_eq!(ctx.virtualizable_box_at(nvs), Some(boxed0));
        assert_eq!(ctx.virtualizable_box_at(nvs + 1), Some(boxed1));
        assert_eq!(
            ctx.virtualizable_entry_at(nvs),
            Some((boxed0, majit_ir::Value::Ref(majit_ir::GcRef(0x1111)))),
        );
        assert_eq!(
            ctx.virtualizable_entry_at(nvs + 1),
            Some((boxed1, majit_ir::Value::Ref(majit_ir::GcRef(0x2222)))),
        );

        swap_stack_slots(&mut sym, &mut ctx, 0, 1);
        assert_eq!(ctx.virtualizable_box_at(nvs), Some(boxed1));
        assert_eq!(ctx.virtualizable_box_at(nvs + 1), Some(boxed0));
        assert_eq!(
            ctx.virtualizable_entry_at(nvs),
            Some((boxed1, majit_ir::Value::Ref(majit_ir::GcRef(0x2222)))),
        );
        assert_eq!(
            ctx.virtualizable_entry_at(nvs + 1),
            Some((boxed0, majit_ir::Value::Ref(majit_ir::GcRef(0x1111)))),
        );
    }

    #[test]
    fn normalize_raise_varargs_jit_null_frame_still_publishes_pending_exception() {
        clear_pending_jit_exception();
        let execution_context = pyre_interpreter::PyExecutionContext::default();
        let exc_class = unsafe {
            pyre_object::w_dict_getitem_str(execution_context.get_builtin_dict(), "ValueError")
        }
        .expect("builtins should contain ValueError");

        let result = normalize_raise_varargs_jit(std::ptr::null(), exc_class, pyre_object::PY_NULL);

        assert_eq!(result as i64, pending_jit_exception_raw());
        let err = unsafe { pyre_interpreter::PyError::from_exc_object(result as PyObjectRef) };
        assert_eq!(err.kind, pyre_interpreter::PyErrorKind::RuntimeError);
        assert_eq!(err.message_text(), "raise helper missing current frame");
        clear_pending_jit_exception();
    }

    #[test]
    fn get_list_of_active_boxes_reads_kind_specific_register_banks() {
        use indexmap::IndexMap;
        use majit_jitcode::liveness::encode_liveness;
        use std::sync::Arc;

        let mut all_liveness = vec![1, 1, 1];
        all_liveness.extend(encode_liveness(&[2]));
        all_liveness.extend(encode_liveness(&[1]));
        all_liveness.extend(encode_liveness(&[3]));
        let mut insns: IndexMap<String, u8> = IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("get_list_of_active_boxes_test");
            inner.set_body(majit_jitcode::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_i: 4,
                c_num_regs_r: 4,
                c_num_regs_f: 4,
                // RPython `jitcode.py:85-90` `assert pc in self._startpoints`:
                // hand-crafted bodies must declare each opcode's offset.  The
                // single BC_LIVE here sits at byte 0.
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.n_py_instrs = 1;
        pyjit.metadata.block_head_py_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.is_drained = true;
        let inner_jc = crate::state::JitCode {
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let int_box = OpRef::int_op(10);
        let ref_box = OpRef::ref_op(20);
        let float_box = OpRef::float_op(30);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        // SSA-authoritative live_r: the encoder reads the
        // color-indexed Ref bank. Set nlocals=2 so the Ref liveness
        // color 1 reads the temporary bank at index 1 (identity for locals).
        // Int and Float banks stay kind-specific (no unification),
        // so their bank-indexed setup is unchanged.
        sym.nlocals = 2;
        sym.valuestackdepth = 2;
        sym.execution_context = OpRef::input_arg_ref(0);
        sym.registers_i = vec![OpRef::NONE, OpRef::NONE, int_box];
        sym.registers_r.replace(vec![OpRef::NONE, ref_box]);
        sym.registers_f = vec![OpRef::NONE, OpRef::NONE, OpRef::NONE, float_box];

        let mut ctx = crate::trace_ctx_for_test(1);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            // The hand-crafted body carries a single `-live-` at JitCode byte 0
            // (`block_head_py_by_jit_pc = [(0, 0)]`); seed that jitcode resume
            // marker so `get_list_of_active_boxes` reads liveness at offset 0
            // instead of declining on an unset marker.
            loop_close_marker_jit_pc: Some(0),
            loop_close_registers_r: Vec::new(),
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        assert_eq!(active, vec![int_box, ref_box, float_box]);

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }

    #[test]
    fn get_list_of_active_boxes_reads_an_unmapped_color_from_the_walked_register_or_nulls_it() {
        use indexmap::IndexMap;
        use majit_jitcode::liveness::encode_liveness;
        use std::sync::Arc;

        let mut all_liveness = vec![0, 2, 0];
        all_liveness.extend(encode_liveness(&[1, 2]));
        let mut insns: IndexMap<String, u8> = IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new(
                "get_list_of_active_boxes_unmapped_merge_point_color_test",
            );
            inner.set_body(majit_jitcode::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_r: 3,
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.n_py_instrs = 1;
        pyjit.metadata.block_head_py_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.is_drained = true;
        pyjit.metadata.has_color_map = true;
        // Color 1 owns local slot 1. Color 2 is live with no slot at this PC:
        // a temporary such as the `pycode` green read before
        // `jit_merge_point`.
        pyjit.metadata.pcdep_by_jit_pc.push((0, vec![(1, 1, 1)]));
        pyjit.metadata.depth_pred_by_jit_pc.push((0, 0));
        let inner_jc = crate::state::JitCode {
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let mapped = OpRef::ref_op(20);
        let stale_slot = OpRef::ref_op(99);
        let walked_green = OpRef::ref_op(50);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        sym.nlocals = 2;
        sym.valuestackdepth = 2;
        sym.execution_context = OpRef::input_arg_ref(0);
        // The slot-indexed mirror holds an unrelated box at index 2.
        sym.registers_r
            .replace(vec![OpRef::NONE, mapped, stale_slot]);

        let mut ctx = crate::trace_ctx_for_test(1);
        let null_ref = ctx.const_ref(0);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: Some(0),
            // The walked registers: color 2 is the green the merge point
            // reads; a mapped color keeps its frame slot's box.
            loop_close_registers_r: vec![OpRef::NONE, OpRef::ref_op(21), walked_green],
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        assert_eq!(active, vec![mapped, walked_green]);

        // Outside the loop-close window there is no walked register, and the
        // unmapped color does not record the mirror's unrelated slot box.
        frame.loop_close_registers_r = Vec::new();
        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        assert_eq!(active, vec![mapped, null_ref]);

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }

    /// Color 1 owns local slot 1; color 2 is live with no slot. Returns the
    /// snapshot boxes from `get_list_of_active_boxes` at the merge-point
    /// `-live-`.
    fn snapshot_live_ref_colors_1_and_2(
        slot_at_2: OpRef,
        loop_close_registers_r: Vec<OpRef>,
        shadow: Option<OpRef>,
    ) -> Vec<OpRef> {
        use indexmap::IndexMap;
        use majit_jitcode::liveness::encode_liveness;
        use std::sync::Arc;

        let mut all_liveness = vec![0, 2, 0];
        all_liveness.extend(encode_liveness(&[1, 2]));
        let mut insns: IndexMap<String, u8> = IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new(
                "get_list_of_active_boxes_unmapped_color_helper",
            );
            inner.set_body(majit_jitcode::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_r: 3,
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.n_py_instrs = 1;
        pyjit.metadata.block_head_py_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.is_drained = true;
        pyjit.metadata.has_color_map = true;
        pyjit.metadata.pcdep_by_jit_pc.push((0, vec![(1, 1, 1)]));
        pyjit.metadata.depth_pred_by_jit_pc.push((0, 0));
        let inner_jc = crate::state::JitCode {
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let mapped = OpRef::ref_op(20);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        sym.nlocals = 2;
        sym.valuestackdepth = 2;
        sym.execution_context = OpRef::input_arg_ref(0);
        sym.registers_r
            .replace(vec![OpRef::NONE, mapped, slot_at_2]);

        let mut ctx = crate::trace_ctx_for_test(1);
        if let Some(held) = shadow {
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            ctx.install_virtualizable_info(info.clone());
            ctx.set_virtualizable_boxes_with_info(vec![held], Vec::new(), &info, &[0]);
        }
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: Some(0),
            loop_close_registers_r,
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
        active
    }

    #[test]
    fn get_list_of_active_boxes_reads_an_unmapped_inputarg_color_from_the_walked_register() {
        let leftover = OpRef::input_arg_ref(3);
        let walked = OpRef::ref_op(50);
        let mapped = OpRef::ref_op(20);
        let active = snapshot_live_ref_colors_1_and_2(
            leftover,
            vec![OpRef::NONE, OpRef::ref_op(21), walked],
            None,
        );
        assert_eq!(active, vec![mapped, walked]);
    }

    #[test]
    fn get_list_of_active_boxes_keeps_an_unmapped_inputarg_color_the_shadow_still_holds() {
        let held = OpRef::input_arg_ref(3);
        let mapped = OpRef::ref_op(20);
        let stale_slot = OpRef::ref_op(99);
        let active = snapshot_live_ref_colors_1_and_2(
            stale_slot,
            vec![OpRef::NONE, OpRef::ref_op(21), held],
            Some(held),
        );
        assert_eq!(active, vec![mapped, held]);
    }

    #[test]
    fn get_list_of_active_boxes_records_const_null_for_an_unmapped_dead_color() {
        let leftover = OpRef::input_arg_ref(3);
        let mapped = OpRef::ref_op(20);
        let active = snapshot_live_ref_colors_1_and_2(leftover, Vec::new(), None);
        assert_eq!(active, vec![mapped, OpRef::const_ptr(majit_ir::GcRef(0))]);
    }

    #[test]
    fn pre_opcode_snapshot_reads_coalesced_stack_color_by_semantic_slot() {
        use indexmap::IndexMap;
        use majit_jitcode::liveness::encode_liveness;
        use std::sync::Arc;

        let mut all_liveness = vec![0, 1, 0];
        all_liveness.extend(encode_liveness(&[0]));
        let mut insns: IndexMap<String, u8> = IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new(
                "pre_opcode_snapshot_coalesced_stack_color_test",
            );
            inner.set_body(majit_jitcode::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_r: 3,
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.n_py_instrs = 1;
        pyjit.metadata.block_head_py_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.py_floor_by_jit_pc = vec![(0, 0)];
        pyjit.metadata.is_drained = true;
        // Per-PC (color, slot) entries the codewriter publishes at pc 0:
        // local 0 -> color 0 (slot 0), local 1 -> color 1 (slot 1), and the
        // live operand-stack slot (depth 0 = abs slot nlocals+0 = 2) -> color
        // 0, reusing dead local 0's color. Sorted by (color, slot).
        pyjit.metadata.has_color_map = true;
        // JitCode-native twins the migrated `get_list_of_active_boxes` reads
        // (`pcdep_for_jitcode_pc` / `depth_for_jitcode_pc_pred`): mirror the
        // py_pc-keyed tables above at the single `-live-` JitCode offset 0.
        pyjit
            .metadata
            .pcdep_by_jit_pc
            .push((0, vec![(1, 0, 0), (1, 0, 2), (1, 1, 1)]));
        pyjit.metadata.depth_pred_by_jit_pc.push((0, 1));
        let inner_jc = crate::state::JitCode {
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let local0 = OpRef::ref_op(10);
        let local1 = OpRef::ref_op(11);
        let stack0 = OpRef::ref_op(20);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        sym.nlocals = 2;
        sym.valuestackdepth = 3;
        sym.execution_context = OpRef::input_arg_ref(0);
        // Semantic mirror: local0 is at slot 0, while stack depth 0 is at
        // semantic slot 2. Liveness color 0 belongs to the live stack slot,
        // reusing dead local0's color.
        sym.registers_r.replace(vec![local0, local1, stack0]);

        let mut ctx = crate::trace_ctx_for_test(1);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            // Single `-live-` at JitCode byte 0 (`block_head_py_by_jit_pc =
            // [(0, 0)]`); seed the jitcode resume marker so the snapshot reads
            // liveness at offset 0 rather than declining on an unset marker.
            loop_close_marker_jit_pc: Some(0),
            loop_close_registers_r: Vec::new(),
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: Some(vec![local0, local1, stack0]),
            pre_opcode_semantic_depth: Some(3),
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false, None);
        assert_eq!(active, vec![stack0]);

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }
}
