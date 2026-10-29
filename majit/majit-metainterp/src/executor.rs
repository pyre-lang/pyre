//! Operation execution for the blackhole interpreter.
//!
//! Mirrors RPython's `executor.py`: executes individual JIT IR operations
//! by dispatching on the opcode and computing the result.

use majit_backend::Backend;
use majit_ir::{OpCode, OpRef};

/// rpython/jit/metainterp/executor.py `execute_varargs(cpu, metainterp, opnum, argboxes, descr)`.
///
/// ```python
/// def execute_varargs(cpu, metainterp, opnum, argboxes, descr):
///     # only for opnums with a variable arity (calls, typically)
///     check_descr(descr)
///     func = get_execute_function(opnum, -1, True)
///     return func(cpu, metainterp, argboxes, descr)
/// ```
///
/// For CALL_* opcodes, `func` ultimately calls `cpu.bh_call_*(funcaddr,
/// args)` (`executor.py _do_call`).  Split the typed argboxes into I/R/F
/// and hand them to `cpu.bh_call_{i,r,v}`.  The Float arm shares the
/// i64-return ABI: helper concrete pointers built by `#[jit_module]`
/// pre-pack the f64 result via `f64::to_bits` (majit-macros
/// `helper_return_to_i64`), so that arm uses `cpu.bh_call_i` with an
/// Int descr — the leftover CA BH packed-bits convention. Callers
/// recover the f64 with `f64::from_bits` (`miframe_execute_varargs`).
/// `argboxes[0]` is the funcbox (carrying the function pointer in its
/// `i64` slot) and the remaining slots are the typed call arguments.
///
/// `metainterp` mirrors RPython's `self` parameter: helper-side
/// exceptions published on the `BH_LAST_EXC_VALUE` thread-local seam
/// (the convention `bh_call_fn_impl` and friends use) are transcribed
/// onto `metainterp.last_exc_value` before returning, the same way
/// RPython's cpu auto-publishes onto `cpu.last_exc_value`.  Pyre has
/// no separate `cpu` object; the metainterp owns the field directly.
///
/// Returns the concrete result value as an `i64` — for void-returning
/// calls returns `0` (caller ignores it); for float-returning calls
/// the i64 carries the f64 bits via `f64::to_bits` and the caller
/// must unpack with `f64::from_bits(resvalue as u64)`.  When the
/// helper raises (BH_LAST_EXC_VALUE seam fires), the post-call hook
/// transcribes onto `metainterp.last_exc_value` and overrides the
/// result with `0` — matching `executor.py:52-78`'s neutral-zero
/// behavior across INT, REF (NULL), FLOAT (longlong.ZEROF), and VOID.
// executor.py:188-190
//
//     def do_getfield_gc_i(cpu, _, structbox, fielddescr):
//         struct = structbox.getref_base()
//         return cpu.bh_getfield_gc_i(struct, fielddescr)
//
// `structbox.getref_base()` returns the GCREF the box carries; pyre's
// flat box analog is the concrete `i64` shadow paired with the symbolic
// OpRef (the (OpRef, i64) tuple returned by `read_ref_reg`). The
// caller has already projected `structbox.getref_base()` and passes it
// as `structbox` here so the function shape matches RPython 1:1.
pub fn do_getfield_gc_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
) -> i64 {
    let struct_ = structbox;
    cpu.bh_getfield_gc_i(struct_, fielddescr)
}

// executor.py:192-194
pub fn do_getfield_gc_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
) -> majit_ir::GcRef {
    let struct_ = structbox;
    cpu.bh_getfield_gc_r(struct_, fielddescr)
}

// executor.py:196-198
pub fn do_getfield_gc_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
) -> f64 {
    let struct_ = structbox;
    cpu.bh_getfield_gc_f(struct_, fielddescr)
}

// executor.py:215-222
//
//     def do_setfield_gc(cpu, _, structbox, itembox, fielddescr):
//         struct = structbox.getref_base()
//         if fielddescr.is_pointer_field():
//             cpu.bh_setfield_gc_r(struct, itembox.getref_base(), fielddescr)
//         elif fielddescr.is_float_field():
//             cpu.bh_setfield_gc_f(struct, itembox.getfloatstorage(), fielddescr)
//         else:
//             cpu.bh_setfield_gc_i(struct, itembox.getint(), fielddescr)
//
// One RPython function covering all three banks, so the type dispatch is
// the descr's own `is_pointer_field()` / `is_float_field()` — not the
// caller's.  `itembox` arrives here already projected to its concrete
// `Value`, the flat-box analog of the `getref_base()` / `getint()` /
// `getfloatstorage()` accessors upstream calls on the box.  Returns
// `false` when the value's bank disagrees with the descr's, which
// upstream cannot express (its boxes are typed by construction).
pub fn do_setfield_gc(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    itembox: majit_ir::Value,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
    field_type: majit_ir::Type,
) -> bool {
    let struct_ = structbox;
    match (field_type, itembox) {
        (majit_ir::Type::Ref, majit_ir::Value::Ref(r)) => {
            cpu.bh_setfield_gc_r(struct_, r, fielddescr);
            true
        }
        (majit_ir::Type::Float, majit_ir::Value::Float(f)) => {
            cpu.bh_setfield_gc_f(struct_, f, fielddescr);
            true
        }
        (majit_ir::Type::Int, majit_ir::Value::Int(i)) => {
            cpu.bh_setfield_gc_i(struct_, i, fielddescr);
            true
        }
        _ => false,
    }
}

//     def do_setarrayitem_gc(cpu, _, arraybox, indexbox, itembox, arraydescr):
//         array = arraybox.getref_base()
//         index = indexbox.getint()
//         if arraydescr.is_array_of_pointers():
//             cpu.bh_setarrayitem_gc_r(array, index, itembox.getref_base(),
//                                      arraydescr)
//         elif arraydescr.is_array_of_floats():
//             cpu.bh_setarrayitem_gc_f(array, index, itembox.getfloatstorage(),
//                                      arraydescr)
//         else:
//             cpu.bh_setarrayitem_gc_i(array, index, itembox.getint(), arraydescr)
//
// The bank comes from the descr, as in `do_setfield_gc`; `false` when the
// value's bank disagrees with it.
pub fn do_setarrayitem_gc(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    itembox: majit_ir::Value,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
    item_type: majit_ir::Type,
) -> bool {
    let array = arraybox;
    let index = indexbox;
    match (item_type, itembox) {
        (majit_ir::Type::Ref, majit_ir::Value::Ref(r)) => {
            cpu.bh_setarrayitem_gc_r(array, index, r, arraydescr);
            true
        }
        (majit_ir::Type::Float, majit_ir::Value::Float(f)) => {
            cpu.bh_setarrayitem_gc_f(array, index, f, arraydescr);
            true
        }
        (majit_ir::Type::Int, majit_ir::Value::Int(i)) => {
            cpu.bh_setarrayitem_gc_i(array, index, i, arraydescr);
            true
        }
        _ => false,
    }
}

// executor.py:206-212 do_getarrayitem_gc_{i,r,f}: project box → gcref +
// concrete index, dispatch to `cpu.bh_getarrayitem_gc_*`.  Pyre's flat
// box analog passes `(array, index)` as plain `i64` values projected
// from the symbolic OpRefs by the caller.
pub fn do_getarrayitem_gc_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getarrayitem_gc_i(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_gc_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
) -> majit_ir::GcRef {
    cpu.bh_getarrayitem_gc_r(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_gc_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getarrayitem_gc_f(arraybox, indexbox, arraydescr)
}

// blackhole.py bhimpl_arraylen_gc(cpu, array, arraydescr): direct
// `cpu.bh_arraylen_gc(array, arraydescr)`.  RPython has no explicit
// `do_arraylen_gc` in executor.py; the dispatch goes through the
// blackhole fallback wrapper.  Pyre exposes it here in `executor.rs`
// for `TraceCtx::arraylen_sanity_load` to consume directly without
// importing the blackhole module.  Array is projected to `i64` from
// the operand (`arraybox.getref_base()` analog).
pub fn do_arraylen_gc(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
) -> i64 {
    cpu.bh_arraylen_gc(arraybox, arraydescr)
}

// executor.py:132 do_getarrayitem_raw_{i,f}: project arraybox → int
// (raw pointer), dispatch to `cpu.bh_getarrayitem_raw_*`.  Distinct
// from `do_getarrayitem_gc_*` (executor.py:117) which projects via
// `arraybox.getref_base()` — raw arrays carry their pointer as a
// plain integer.  Pyre passes the raw pointer as `i64`.
pub fn do_getarrayitem_raw_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getarrayitem_raw_i(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_raw_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_jitcode::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getarrayitem_raw_f(arraybox, indexbox, arraydescr)
}

// executor.py:200 do_getfield_raw_{i,r,f}: project structbox → int
// (raw pointer via `structbox.getint()`), dispatch to
// `cpu.bh_getfield_raw_*`.  Distinct from `do_getfield_gc_*`
// (executor.py:188) which projects via `structbox.getref_base()` —
// raw structs carry their pointer as a plain integer.  Pyre's caller
// projects the symbolic OpRef carrier to `i64` directly.
pub fn do_getfield_raw_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getfield_raw_i(structbox, fielddescr)
}

pub fn do_getfield_raw_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
) -> majit_ir::GcRef {
    cpu.bh_getfield_raw_r(structbox, fielddescr)
}

pub fn do_getfield_raw_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_jitcode::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getfield_raw_f(structbox, fielddescr)
}

/// Walker leftover of `_do_call` when there is no `MetaInterp`/`cpu`:
/// split positional `args` by `descr.arg_types()` into I/R/F and call
/// `bh_call_*_by_classes`. Float results use the real f64 ABI
/// (`executor.py` `cpu.bh_call_f`) and return packed bits.
pub(crate) fn do_call_positional(
    descr: &dyn majit_ir::descr::CallDescr,
    func_ptr: i64,
    args: &[i64],
) -> i64 {
    let arg_types = descr.arg_types();
    let mut args_i = Vec::new();
    let mut args_r = Vec::new();
    let mut args_f = Vec::new();
    let mut arg_classes = String::new();
    for (i, ty) in arg_types.iter().enumerate().take(args.len()) {
        let concrete = args[i];
        match ty {
            majit_ir::Type::Int => {
                args_i.push(concrete);
                arg_classes.push('i');
            }
            majit_ir::Type::Ref => {
                args_r.push(concrete);
                arg_classes.push('r');
            }
            majit_ir::Type::Float => {
                args_f.push(concrete);
                arg_classes.push('f');
            }
            majit_ir::Type::Void => panic!("typed call: void argument class at slot {i}"),
        }
    }
    let func = func_ptr as usize;
    match descr.result_type() {
        majit_ir::Type::Int => unsafe {
            majit_backend::call_stub::bh_call_i_by_classes(
                func,
                &arg_classes,
                leftover_bh_args(&args_i),
                leftover_bh_args(&args_r),
                leftover_bh_args(&args_f),
            )
        },
        majit_ir::Type::Ref => unsafe {
            majit_backend::call_stub::bh_call_ref_by_classes(
                func,
                &arg_classes,
                leftover_bh_args(&args_i),
                leftover_bh_args(&args_r),
                leftover_bh_args(&args_f),
            )
        },
        majit_ir::Type::Void => {
            unsafe {
                majit_backend::call_stub::bh_call_v_by_classes(
                    func,
                    &arg_classes,
                    leftover_bh_args(&args_i),
                    leftover_bh_args(&args_r),
                    leftover_bh_args(&args_f),
                );
            }
            0
        }
        majit_ir::Type::Float => unsafe {
            majit_backend::call_stub::bh_call_f_by_classes(
                func,
                &arg_classes,
                leftover_bh_args(&args_i),
                leftover_bh_args(&args_r),
                leftover_bh_args(&args_f),
            )
            .to_bits() as i64
        },
    }
}

fn leftover_bh_args(v: &[i64]) -> Option<&[i64]> {
    if v.is_empty() { None } else { Some(v) }
}

/// executor.py `do_cond_call(cpu, metainterp, argboxes, descr)`:
///
/// ```python
/// def do_cond_call(cpu, metainterp, argboxes, descr):
///     condbox = argboxes[0]
///     if condbox.getint():
///         do_call_n(cpu, metainterp, argboxes[1:], descr)
/// ```
///
/// Trace-time `emit_force_virtualizable` has the funcptr / calldescr /
/// virtualizable argument at the record site and no `MetaInterp`. When
/// a Backend CPU is installed (`TraceCtx::blackhole_cpu`, wired by
/// `TraceCtx::set_cpu` from `MetaInterp.backend`) the call goes through
/// `cpu.bh_call_v`. Otherwise `_do_call` is the walker leftover
/// `do_call_positional`.
pub fn do_cond_call(
    cpu: Option<&dyn majit_backend::Backend>,
    cond: i64,
    func_ptr: i64,
    args: &[i64],
    descr: &dyn majit_ir::descr::CallDescr,
) {
    if cond == 0 {
        return;
    }
    match cpu {
        Some(cpu) => do_call_n(cpu, func_ptr, args, descr),
        None => {
            let _ = do_call_positional(descr, func_ptr, args);
        }
    }
}

/// executor.py `do_call_n` → `_do_call(..., rettype='v')`: split
/// positional args by `descr.arg_types()` and `cpu.bh_call_v`.
fn do_call_n(
    cpu: &dyn majit_backend::Backend,
    func_ptr: i64,
    args: &[i64],
    descr: &dyn majit_ir::descr::CallDescr,
) {
    let mut args_i = Vec::new();
    let mut args_r = Vec::new();
    let mut args_f = Vec::new();
    for (i, ty) in descr.arg_types().iter().enumerate().take(args.len()) {
        match ty {
            majit_ir::Type::Int => args_i.push(args[i]),
            majit_ir::Type::Ref => args_r.push(args[i]),
            majit_ir::Type::Float => args_f.push(args[i]),
            majit_ir::Type::Void => panic!("do_call_n: void argument class at slot {i}"),
        }
    }
    let calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(descr);
    cpu.bh_call_v(
        func_ptr,
        leftover_bh_args(&args_i),
        leftover_bh_args(&args_r),
        leftover_bh_args(&args_f),
        &calldescr,
    );
}

/// `executor.py _do_call`: split argboxes[1..] into I/R/F and call
/// `cpu.bh_call_*`. `argboxes[0]` is the funcbox.
fn do_call<M: Clone>(
    metainterp: &crate::pyjitpl::MetaInterp<M>,
    argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
    descr: &dyn majit_ir::descr::CallDescr,
    result_type: majit_ir::Type,
) -> i64 {
    let func = argboxes[0].2;
    let mut args_i = Vec::new();
    let mut args_r = Vec::new();
    let mut args_f = Vec::new();
    for (kind, _, concrete) in &argboxes[1..] {
        match kind {
            crate::jitcode::JitArgKind::Int => args_i.push(*concrete),
            crate::jitcode::JitArgKind::Ref => args_r.push(*concrete),
            crate::jitcode::JitArgKind::Float => args_f.push(*concrete),
        }
    }
    let cpu = metainterp.blackhole_cpu();
    // Float leftover wrappers return packed i64 bits (`f64::to_bits`);
    // leftover CA BH uses an Int descr + `cpu.bh_call_i` so
    // `verify_result_type` accepts the i64 ABI.
    let calldescr = if result_type == majit_ir::Type::Float {
        majit_jitcode::jitcode::BhCallDescr::from_signature(
            descr.arg_classes(),
            majit_ir::Type::Int,
            descr.get_extra_info().clone(),
        )
    } else {
        majit_jitcode::jitcode::BhCallDescr::from_call_descr(descr)
    };
    match result_type {
        majit_ir::Type::Int => cpu.bh_call_i(
            func,
            leftover_bh_args(&args_i),
            leftover_bh_args(&args_r),
            leftover_bh_args(&args_f),
            &calldescr,
        ),
        majit_ir::Type::Ref => {
            cpu.bh_call_r(
                func,
                leftover_bh_args(&args_i),
                leftover_bh_args(&args_r),
                leftover_bh_args(&args_f),
                &calldescr,
            )
            .0 as i64
        }
        majit_ir::Type::Void => {
            cpu.bh_call_v(
                func,
                leftover_bh_args(&args_i),
                leftover_bh_args(&args_r),
                leftover_bh_args(&args_f),
                &calldescr,
            );
            0
        }
        majit_ir::Type::Float => cpu.bh_call_i(
            func,
            leftover_bh_args(&args_i),
            leftover_bh_args(&args_r),
            leftover_bh_args(&args_f),
            &calldescr,
        ),
    }
}

pub fn execute_varargs<M: Clone>(
    metainterp: &mut crate::pyjitpl::MetaInterp<M>,
    opnum: OpCode,
    argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
    descr: &dyn majit_ir::descr::CallDescr,
) -> i64 {
    debug_assert!(opnum.is_call(), "execute_varargs requires a call opcode");
    // RPython's `cpu` parameter is the seam through which helper-side
    // exceptions reach `metainterp.last_exc_value` (`cpu.bh_call_*`
    // writes onto `cpu.last_exception` and the metainterp's
    // post-execute hook copies it).  Pyre has no separate cpu; we use
    // the `BH_LAST_EXC_VALUE` thread-local that production helpers
    // (`bh_call_fn_impl` etc.) publish on, mirroring the same convention
    // every blackhole.rs CALL_* arm uses (its `handler_residual_call_*`
    // family and their shared `check_residual_call_exception_after`).
    // Clear before dispatch so a stale value from a prior call cannot
    // bleed into this one.
    crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    // Inner closure carries every existing return path so the outer
    // wrapper can run BH_LAST_EXC_VALUE transcription regardless of
    // which arm fires.
    let result = (|| -> i64 {
        // COND_CALL / COND_CALL_VALUE_* layout (pyjitpl.py do_conditional_call):
        //   argboxes[0] = condbox / valuebox
        //   argboxes[1] = funcbox
        //   argboxes[2..] = call args
        // RPython's executor handles cond-call dispatch via a per-opcode
        // execute function (`do_cond_call_*`); pyre inlines the same
        // semantics here.
        //
        // Two distinct shapes (blackhole.py:1257-1276):
        //   bhimpl_conditional_call_ir_v(condition, func, ...):
        //       if condition: cpu.bh_call_v(func, ...)        # void
        //   bhimpl_conditional_call_value_ir_{i,r}(value, func, ...):
        //       if value == 0: value = cpu.bh_call_*(func, ...)
        //       return value
        if matches!(opnum, OpCode::CondCallN) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_N requires [condbox, funcbox, *args]",
            );
            // `executor.py do_cond_call`: skip when the condbox is 0,
            // else `do_call_n(..., argboxes[1:], descr)`.
            if argboxes[0].2 == 0 {
                return 0;
            }
            return do_call(metainterp, &argboxes[1..], descr, majit_ir::Type::Void);
        }
        if matches!(opnum, OpCode::CondCallValueI | OpCode::CondCallValueR) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_VALUE_* requires [valuebox, funcbox, *args]",
            );
            let value = argboxes[0].2;
            if value != 0 {
                // `do_cond_call_value_{i,r}`: nonzero short-circuits.
                return value;
            }
            let result_type = if matches!(opnum, OpCode::CondCallValueR) {
                majit_ir::Type::Ref
            } else {
                majit_ir::Type::Int
            };
            return do_call(metainterp, &argboxes[1..], descr, result_type);
        }
        debug_assert!(
            !argboxes.is_empty(),
            "execute_varargs: argboxes must include funcbox at slot 0",
        );
        do_call(metainterp, argboxes, descr, descr.result_type())
    })();
    // Mirror RPython's executor.py:52-78 post-call exception flow:
    // each `cpu.bh_call_*` arm wraps the call in `try: ... except
    // Exception as e: metainterp.execute_raised(e); result = ZERO`.
    // Pyre observes the same condition through the BH_LAST_EXC_VALUE
    // thread-local seam.  Two pieces have to fire together:
    //   1. `metainterp.execute_raised(bh_exc, constant=False)` — sets
    //      `last_exc_value` AND clears `class_of_last_exc_is_const`,
    //      so a stale `True` from a prior GUARD_EXCEPTION cannot make
    //      `handle_possible_exception` treat the new exception's
    //      class as constant (pyjitpl.py).
    //   2. Override the returned value with the type's neutral zero —
    //      `make_result_of_lastop` (`pyjitpl/frame.rs`) snapshots the
    //      concrete result *before* `handle_possible_exception` runs,
    //      so leaving the helper's return value in place can pin a
    //      garbage value into the resume snapshot.  `i64 == 0` covers
    //      INT=0, REF=NULL, and FLOAT=longlong.ZEROF (0.0_f64.to_bits()
    //      as i64 == 0); VOID callers ignore the slot.
    let bh_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 {
        // The shared `bh_*` helper published this raise into the backend
        // `_store_exception` cells as well as `BH_LAST_EXC_VALUE`
        // (`publish_residual_call_exception`).  Those cells belong to
        // compiled / blackhole execution; RPython tracing never touches
        // them (executor.py:52-78 hands the exception straight to
        // `metainterp.execute_raised`).  Drain them here, or a raise that
        // tracing consumed — for example one handled by a recorded
        // GUARD_EXCEPTION — outlives the trace, and the next compiled
        // trace's `must_save_exception` guard delivers it as that frame's
        // own raise: an exception surfacing out of a frame that raised
        // nothing.
        metainterp.backend.clear_stored_exception();
        metainterp.execute_raised(bh_exc, false);
        return 0;
    }
    result
}

/// One row of `executor.EXECUTE_BY_NUM_ARGS`, resolved.
///
/// `_execute_arglist` raises `NotImplementedError` when the table holds no
/// helper for the opcode — a different event from a registered helper refusing
/// its operands, which upstream cannot express at all because every upstream
/// helper is total. The `Option`-returning wrappers below collapse the two into
/// `None`, so [`execute_nonspec_const`] consults the `*_row` functions instead
/// and keeps [`NoConstExecutor`] for the missing helper alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstFold<T> {
    /// This row has no entry for the opcode.
    Unregistered,
    /// The entry exists and refused these operands.
    Declined,
    /// The folded constant.
    Folded(T),
}

impl<T> ConstFold<T> {
    /// The `Option` view: a decline and a missing row are both `None`.
    pub fn folded(self) -> Option<T> {
        match self {
            ConstFold::Folded(value) => Some(value),
            ConstFold::Unregistered | ConstFold::Declined => None,
        }
    }
}

/// `executor.py::execute_nonspec_const` for binary integer opcodes.
///
/// Returns the folded `i64` result when the operation is recognized and
/// the result is well-defined; returns `None` to abort folding when:
///   * the opcode is not a recognized binary int op
///   * an OVF arithmetic op (IntAddOvf/SubOvf/MulOvf) overflows —
///     RPython's `do_int_add_ovf` then hits its
///     `assert metainterp is not None`, which
///     AssertionErrors in the `constant_fold` path (metainterp=None);
///     pyre prefers the softer `None` skip so the op stays in the
///     trace and the runtime guard fires
///   * a shift count is outside `0..64`, as rejected by
///     `blackhole.check_shift_count`
///   * IntFloorDiv / IntMod with a zero divisor
///
/// Non-OVF IntAdd/IntSub/IntMul match `bhimpl_int_add/_sub/_mul`
/// compute `intmask(a + b)`, i.e. wrapping i64 arithmetic. Using
/// `checked_*` for these non-overflowing opcodes would abort the fold on a
/// representable wrapping result.
///
/// Mirrors the overflow helpers and binary-int rows generated in
/// `executor.EXECUTE_BY_NUM_ARGS`.
pub fn execute_binary_int_const(opcode: OpCode, a: i64, b: i64) -> Option<i64> {
    execute_binary_int_const_row(opcode, a, b).folded()
}

/// [`execute_binary_int_const`] with its two `None`s separated — see
/// [`ConstFold`]. Every opcode named below has an entry, so an arm answering
/// `None` is that entry declining, never a missing one.
pub fn execute_binary_int_const_row(opcode: OpCode, a: i64, b: i64) -> ConstFold<i64> {
    let folded = match opcode {
        OpCode::IntAdd => Some(a.wrapping_add(b)),
        OpCode::IntSub => Some(a.wrapping_sub(b)),
        OpCode::IntMul => Some(a.wrapping_mul(b)),
        OpCode::IntAddOvf => a.checked_add(b),
        OpCode::IntSubOvf => a.checked_sub(b),
        OpCode::IntMulOvf => a.checked_mul(b),
        OpCode::IntAnd => Some(a & b),
        OpCode::IntOr => Some(a | b),
        OpCode::IntXor => Some(a ^ b),
        OpCode::IntLshift => (0..64).contains(&b).then(|| a << b),
        OpCode::IntRshift => (0..64).contains(&b).then(|| a >> b),
        OpCode::UintRshift => (0..64).contains(&b).then(|| (a as u64 >> b as u64) as i64),
        OpCode::IntLt => Some((a < b) as i64),
        OpCode::IntLe => Some((a <= b) as i64),
        OpCode::IntGt => Some((a > b) as i64),
        OpCode::IntGe => Some((a >= b) as i64),
        OpCode::IntEq => Some((a == b) as i64),
        OpCode::IntNe => Some((a != b) as i64),
        OpCode::UintLt => Some(((a as u64) < (b as u64)) as i64),
        OpCode::UintLe => Some(((a as u64) <= (b as u64)) as i64),
        OpCode::UintGe => Some(((a as u64) >= (b as u64)) as i64),
        OpCode::UintGt => Some(((a as u64) > (b as u64)) as i64),
        // Truncating, as `_ll_2_int_floordiv` / `_ll_2_int_mod` are and as the
        // blackhole and every backend execute them. Wrapping keeps
        // `i64::MIN / -1` off the panic path; the trace guards that corner out
        // before it reaches either op.
        OpCode::IntFloorDiv => (b != 0).then(|| a.wrapping_div(b)),
        OpCode::IntMod => (b != 0).then(|| a.wrapping_rem(b)),
        // `blackhole.bhimpl_int_signext` delegates to `support.int_signext`.
        OpCode::IntSignext => (1..=8)
            .contains(&b)
            .then(|| crate::support::int_signext(a, b)),
        OpCode::UintMulHigh => {
            // blackhole.py bhimpl_uint_mul_high — high 64 of (a as u64) * (b as u64).
            Some((((a as u64) as u128 * (b as u64) as u128) >> 64) as i64)
        }
        _ => return ConstFold::Unregistered,
    };
    match folded {
        Some(value) => ConstFold::Folded(value),
        None => ConstFold::Declined,
    }
}

/// Pointer-comparison row of `executor.EXECUTE_BY_NUM_ARGS`.
/// Mirrors `blackhole.bhimpl_ptr_eq`, `bhimpl_ptr_ne`, and their instance forms:
/// straight pointer identity once both args are constant references.
pub fn execute_ptr_compare_const(opcode: OpCode, a: usize, b: usize) -> Option<i64> {
    let result = match opcode {
        OpCode::PtrEq | OpCode::InstancePtrEq => a == b,
        OpCode::PtrNe | OpCode::InstancePtrNe => a != b,
        _ => return None,
    };
    Some(result as i64)
}

/// `executor.py execute` for the pointer-compare row: unbox two values
/// (`getref_base()` / the Int word a non-moving pointer constant carries)
/// and run [`execute_ptr_compare_const`]. `blackhole.py bhimpl_ptr_ne`
/// does not take `cpu`.
pub fn execute_ptr_compare_values(
    opcode: OpCode,
    a: majit_ir::Value,
    b: majit_ir::Value,
) -> Option<i64> {
    execute_ptr_compare_const(opcode, const_pointer_word(a)?, const_pointer_word(b)?)
}

/// The machine word a constant carries when it stands for a pointer: a GC
/// reference holds it in `Value::Ref`, a pointer to non-moving memory in
/// `Value::Int`.  A float or void constant is not a pointer.
fn const_pointer_word(value: majit_ir::Value) -> Option<usize> {
    match value {
        majit_ir::Value::Ref(r) => Some(r.0),
        majit_ir::Value::Int(i) => Some(i as usize),
        majit_ir::Value::Float(_) | majit_ir::Value::Void => None,
    }
}

/// Unary-int row of `executor.EXECUTE_BY_NUM_ARGS`. Mirrors the corresponding
/// `blackhole.bhimpl_int_*` helpers.
///
/// Returns `None` for unrecognized opcodes so the caller can fall
/// through to other dispatch paths; upstream's unrolled dispatcher reserves
/// its terminal assertion for opcodes with no matching row.
pub fn execute_unary_int_const(opcode: OpCode, a: i64) -> Option<i64> {
    let result = match opcode {
        OpCode::IntNeg => a.wrapping_neg(),
        OpCode::IntInvert => !a,
        OpCode::IntIsZero => (a == 0) as i64,
        OpCode::IntIsTrue => (a != 0) as i64,
        OpCode::IntForceGeZero => a.max(0),
        _ => return None,
    };
    Some(result)
}

/// Unary-float row of `executor.EXECUTE_BY_NUM_ARGS`, mirroring
/// `blackhole.bhimpl_float_neg` and `bhimpl_float_abs`.
pub fn execute_unary_float_const(opcode: OpCode, a: f64) -> Option<f64> {
    let result = match opcode {
        OpCode::FloatNeg => -a,
        OpCode::FloatAbs => a.abs(),
        _ => return None,
    };
    Some(result)
}

/// Binary-float row of `executor.EXECUTE_BY_NUM_ARGS`, mirroring the
/// corresponding `blackhole.bhimpl_float_*` helpers.
/// FLOAT_TRUEDIV with `b == 0.0` is NOT folded — see upstream
/// `test_optimizebasic.test_float_division_by_multiplication` which
/// preserves `float_truediv(f, 0.0)` in the optimized loop rather than
/// freezing the IEEE inf/nan constant. The runtime executor still
/// performs `a / b` through `blackhole.bhimpl_float_truediv`;
/// only trace-time folding is suppressed.
pub fn execute_binary_float_const(opcode: OpCode, a: f64, b: f64) -> Option<f64> {
    execute_binary_float_const_row(opcode, a, b).folded()
}

/// [`execute_binary_float_const`] with its two `None`s separated — see
/// [`ConstFold`].
pub fn execute_binary_float_const_row(opcode: OpCode, a: f64, b: f64) -> ConstFold<f64> {
    let folded = match opcode {
        OpCode::FloatAdd => Some(a + b),
        OpCode::FloatSub => Some(a - b),
        OpCode::FloatMul => Some(a * b),
        OpCode::FloatTrueDiv => (b != 0.0).then(|| a / b),
        _ => return ConstFold::Unregistered,
    };
    match folded {
        Some(value) => ConstFold::Folded(value),
        None => ConstFold::Declined,
    }
}

/// Float-comparison row of `executor.EXECUTE_BY_NUM_ARGS`, mirroring the
/// corresponding `blackhole.bhimpl_float_*` comparison helpers.
pub fn execute_float_compare_const(opcode: OpCode, a: f64, b: f64) -> Option<i64> {
    let result = match opcode {
        OpCode::FloatLt => a < b,
        OpCode::FloatLe => a <= b,
        OpCode::FloatEq => a == b,
        OpCode::FloatNe => a != b,
        OpCode::FloatGt => a > b,
        OpCode::FloatGe => a >= b,
        _ => return None,
    };
    Some(result as i64)
}

/// Constant-fold dispatch corresponding to `executor.execute_nonspec_const`
/// and `executor._execute_arglist`. RPython raises `NotImplementedError` when
/// `EXECUTE_BY_NUM_ARGS[arity, withdescr][opnum]` has no helper; Pyre preserves
/// that distinction without using an exception:
///
/// - `Err(NoConstExecutor)` means no helper is registered for the opcode.
/// - `Ok(None)` means a registered helper declined to fold its operands.
/// - `Ok(Some(value))` contains the folded constant.
///
/// `_type` is accepted for signature parity with RPython's `type`
/// parameter; it is not consulted because the Value variant in
/// `argboxes` already determines the result type via the helper that
/// fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoConstExecutor;

pub fn execute_nonspec_const(
    cpu: &dyn crate::cpu::Cpu,
    opnum: OpCode,
    argboxes: &[majit_ir::Value],
    descr: Option<&majit_ir::descr::DescrRef>,
    _type: majit_ir::Type,
) -> Result<Option<majit_ir::Value>, NoConstExecutor> {
    use majit_ir::Value;
    let arity = argboxes.len();

    // ── arity == 1 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 1 {
        let a = argboxes[0];
        // executor.py `do_same_as_i/r/f` — identity fold for
        // SAME_AS_I/R/F (`bhimpl_int_same_as` etc., `blackhole.py`).
        match opnum {
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => return Ok(Some(a)),
            _ => {}
        }
        if let Value::Int(i) = a
            && let Some(folded) = execute_unary_int_const(opnum, i)
        {
            return Ok(Some(Value::Int(folded)));
        }
        if let Value::Float(f) = a
            && let Some(folded) = execute_unary_float_const(opnum, f)
        {
            return Ok(Some(Value::Float(folded)));
        }
        match execute_cast_const_row(opnum, a) {
            ConstFold::Folded(folded) => return Ok(Some(folded)),
            ConstFold::Declined => return Ok(None),
            ConstFold::Unregistered => {}
        }
        // GETFIELD_GC_{I,R,F} — withdescr arity-1.
        // The executor helper is registered under the plain opnum
        // (resoperation.py's table has no PURE getfield opnums; the
        // fold decision is the caller's, gated on
        // `descr.is_always_pure()` at `heap.py:641
        // optimize_GETFIELD_GC_I`), so the fold must accept the plain
        // spelling used for immutable fields.
        // `optimizer.py protect_speculative_operation` has
        // validated the gcref is non-null and of a valid type for
        // `fielddescr.parent_descr` (`llmodel.py`); the
        // dereference below cannot fault.  Unsupported field_size is
        // `llmodel.py read_int_at_mem`'s NotImplementedError and
        // would propagate to crash upstream — pyre matches via
        // fail-loud `unreachable!()`.
        if let (Value::Ref(struct_ref), Some(d)) = (a, descr)
            && let Some(fd) = d.as_field_descr()
        {
            return Ok(Some(match opnum {
                OpCode::GetfieldGcI => match fd.field_size() {
                    1 | 2 | 4 | 8 => Value::Int(cpu.bh_getfield_gc_i(struct_ref.0, fd)),
                    sz => unreachable!(
                        "GETFIELD_GC_I: unsupported field_size {} \
                             (llmodel.py:478 raises NotImplementedError)",
                        sz
                    ),
                },
                OpCode::GetfieldGcR => Value::Ref(cpu.bh_getfield_gc_r(struct_ref.0, fd)),
                OpCode::GetfieldGcF => Value::Float(cpu.bh_getfield_gc_f(struct_ref.0, fd)),
                _ => return Err(NoConstExecutor),
            }));
        }
        // ARRAYLEN_GC — withdescr arity-1.
        // `executor.py:do_arraylen_gc` → `cpu.bh_arraylen_gc(array, ad)`.
        // `protect_speculative_array` validated the gcref + tid; the
        // arraydescr is expected to carry a `len_descr` (registered
        // by the backend's array metadata), so a missing one is a
        // bug — fail-loud per `llmodel.py assert isinstance(...)`.
        if let (Value::Ref(array), Some(d)) = (a, descr)
            && let Some(ad) = d.as_array_descr()
            && opnum == OpCode::ArraylenGc
        {
            let len = cpu
                .bh_arraylen_gc(array, ad)
                .expect("ARRAYLEN_GC: arraydescr missing len_descr (llmodel.py:585 asserts)");
            return Ok(Some(Value::Int(len)));
        }
        // STRLEN / UNICODELEN — by the time the fold is entered,
        // `protect_speculative_string / _unicode` has already validated
        // `str_descr / unicode_descr` is registered (under
        // `supports_guard_gc_type == true`; the false case is gated
        // out at `OptContext::protect_speculative_operation`).
        if let Value::Ref(s) = a {
            match opnum {
                OpCode::Strlen => {
                    let len = cpu
                        .bh_strlen(s)
                        .expect("STRLEN: str_descr unregistered after protect_speculative_string");
                    return Ok(Some(Value::Int(len)));
                }
                OpCode::Unicodelen => {
                    let len = cpu.bh_unicodelen(s).expect(
                        "UNICODELEN: unicode_descr unregistered after protect_speculative_unicode",
                    );
                    return Ok(Some(Value::Int(len)));
                }
                _ => {}
            }
        }
    }

    // ── arity == 2 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 2 {
        if let (Value::Int(a), Value::Int(b)) = (argboxes[0], argboxes[1]) {
            match execute_binary_int_const_row(opnum, a, b) {
                ConstFold::Folded(folded) => return Ok(Some(Value::Int(folded))),
                ConstFold::Declined => return Ok(None),
                ConstFold::Unregistered => {}
            }
        }
        if let (Value::Float(a), Value::Float(b)) = (argboxes[0], argboxes[1]) {
            match execute_binary_float_const_row(opnum, a, b) {
                ConstFold::Folded(folded) => return Ok(Some(Value::Float(folded))),
                ConstFold::Declined => return Ok(None),
                ConstFold::Unregistered => {}
            }
            if let Some(folded) = execute_float_compare_const(opnum, a, b) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        // PTR_EQ / PTR_NE and their instance forms compare two machine words.
        // `do_ptr_eq` reads `getref_base()` off two boxes because every RPython
        // pointer constant is a ref box; pyre models a pointer to non-moving
        // memory -- a `&'static PyType` vtable, say -- as `Value::Int` holding
        // the same address, so `py_type_check`'s `ob_type == &TYPE_TYPE` folds
        // with one arg of each shape once the receiver itself is constant.
        if let (Some(a), Some(b)) = (
            const_pointer_word(argboxes[0]),
            const_pointer_word(argboxes[1]),
        ) && let Some(folded) = execute_ptr_compare_const(opnum, a, b)
        {
            return Ok(Some(Value::Int(folded)));
        }
        // GETARRAYITEM_GC_PURE_I/R/F — withdescr arity-2 (array, index).
        // `executor.py:do_getarrayitem_gc_pure_*` →
        //   `cpu.bh_getarrayitem_gc_*(array, index, ad)`.
        // `protect_speculative_array` + the array-bounds check at
        // `optimizer.py:865-867` validated the gcref/index pre-fold;
        // unsupported `item_size` matches `llmodel.py`'s
        // NotImplementedError, fail-loud via `unreachable!()`.
        if let (Value::Ref(array), Value::Int(index), Some(d)) = (argboxes[0], argboxes[1], descr)
            && let Some(ad) = d.as_array_descr()
        {
            return Ok(Some(match opnum {
                OpCode::GetarrayitemGcPureI => {
                    let v = cpu.bh_getarrayitem_gc_i(array, index, ad).expect(
                        "GETARRAYITEM_GC_PURE_I: unsupported item_size \
                             (llmodel.py:478 raises NotImplementedError)",
                    );
                    Value::Int(v)
                }
                OpCode::GetarrayitemGcPureR => {
                    Value::Ref(cpu.bh_getarrayitem_gc_r(array, index, ad))
                }
                OpCode::GetarrayitemGcPureF => {
                    Value::Float(cpu.bh_getarrayitem_gc_f(array, index, ad))
                }
                _ => return Err(NoConstExecutor),
            }));
        }
        // STRGETITEM / UNICODEGETITEM — protect_speculative_string /
        // _unicode has validated str_descr / unicode_descr is
        // registered (under `supports_guard_gc_type == true`).
        if let (Value::Ref(s), Value::Int(index)) = (argboxes[0], argboxes[1]) {
            match opnum {
                OpCode::Strgetitem => {
                    let v = cpu.bh_strgetitem(s, index).expect(
                        "STRGETITEM: str_descr unregistered after protect_speculative_string",
                    );
                    return Ok(Some(Value::Int(v)));
                }
                OpCode::Unicodegetitem => {
                    let v = cpu.bh_unicodegetitem(s, index).expect(
                        "UNICODEGETITEM: unicode_descr unregistered after protect_speculative_unicode",
                    );
                    return Ok(Some(Value::Int(v)));
                }
                _ => {}
            }
        }
    }

    // ── arity == 3 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 3 {
        // executor.py `do_int_between` -> blackhole.py bhimpl_int_between
        // `bhimpl_int_between(a, b, c): return a <= b < c`.
        if let (Value::Int(a), Value::Int(b), Value::Int(c)) =
            (argboxes[0], argboxes[1], argboxes[2])
            && opnum == OpCode::IntBetween
        {
            return Ok(Some(Value::Int((a <= b && b < c) as i64)));
        }
    }

    // Preserve `_execute_arglist`'s missing-helper outcome separately from the
    // `Ok(None)` outcome returned by a registered helper.
    Err(NoConstExecutor)
}

/// executor.py cross-type cast fold:
///   CAST_FLOAT_TO_INT / CAST_INT_TO_FLOAT — true numeric conversion
///   CAST_FLOAT_TO_SINGLEFLOAT / CAST_SINGLEFLOAT_TO_FLOAT — f64↔f32 bits
///   CONVERT_FLOAT_BYTES_TO_LONGLONG / CONVERT_LONGLONG_BYTES_TO_FLOAT —
///       reinterpret-bits pass-through (Value carries f64 bits already
///       as i64 in pyre's TraceValues; the cast just relabels)
///   CAST_PTR_TO_INT / CAST_INT_TO_PTR — pointer reinterpret
/// Mirrors blackhole.py bhimpl_cast_*.
pub fn execute_cast_const(opcode: OpCode, arg: majit_ir::Value) -> Option<majit_ir::Value> {
    execute_cast_const_row(opcode, arg).folded()
}

/// [`execute_cast_const`] with its two `None`s separated — see [`ConstFold`].
pub fn execute_cast_const_row(opcode: OpCode, arg: majit_ir::Value) -> ConstFold<majit_ir::Value> {
    use majit_ir::{GcRef, Value};
    match (opcode, arg) {
        (OpCode::CastFloatToInt, Value::Float(f)) => {
            // blackhole.py `bhimpl_cast_float_to_int = int(int(a))`
            // lowers to `lltype.cast_float_to_int` (C `(long)f`) post-
            // translation. Skip fold on non-finite or out-of-range
            // floats: the runtime path's `as i64` saturates (lenient
            // IEEE policy, see `blackhole.rs`'s `bhimpl_cast_float_to_int`) but
            // trace-time fold would freeze that saturation as a
            // constant — emit the cast op instead so the runtime
            // computes consistently. The safe i64 window is
            // `i64::MIN ..= i64::MAX`; `i64::MAX + 1` rounds to the
            // same f64 as `i64::MAX` (precision loss), so use the
            // strictly-less-than upper bound `9.223372036854776e18`.
            if !f.is_finite() || f < (i64::MIN as f64) || f >= 9.223372036854776e18_f64 {
                return ConstFold::Declined;
            }
            ConstFold::Folded(Value::Int(f as i64))
        }
        (OpCode::CastIntToFloat, Value::Int(i)) => ConstFold::Folded(Value::Float(i as f64)),
        (OpCode::CastFloatToSinglefloat, Value::Float(f)) => {
            ConstFold::Folded(Value::Int(crate::blackhole::singlefloat2int(f as f32)))
        }
        (OpCode::CastSinglefloatToFloat, Value::Int(i)) => {
            ConstFold::Folded(Value::Float(f32::from_bits(i as u32) as f64))
        }
        (OpCode::ConvertFloatBytesToLonglong, Value::Float(f)) => {
            ConstFold::Folded(Value::Int(f.to_bits() as i64))
        }
        (OpCode::ConvertLonglongBytesToFloat, Value::Int(i)) => {
            ConstFold::Folded(Value::Float(f64::from_bits(i as u64)))
        }
        // `assembler.py:1528-1529 genop_cast_ptr_to_int =
        // _genop_same_as` / `genop_cast_int_to_ptr = _genop_same_as`.
        // `rewrite.py optimize_CAST_PTR_TO_INT` emits the cast; it does not
        // fold it. A managed pointer folded to `ConstInt` leaves the gc
        // table (`remove_constptr` only rewrites a live `ConstPtr` to
        // `LoadFromGcTable`). The immediate then dangles after the nursery
        // reuses the slot, and `jit_bigint_divrem_returns_lhs_remainder`
        // reads `_digits == 0x29`. Untagged non-GC addresses still fold:
        // `runner_test.py` `cast_int_to_ptr(21)` round-trips through
        // `cast_ptr_to_int`.
        (OpCode::CastPtrToInt, Value::Ref(r)) => {
            if majit_gc::gc_owns_object(r.0) {
                ConstFold::Declined
            } else {
                ConstFold::Folded(Value::Int(r.0 as i64))
            }
        }
        (OpCode::CastIntToPtr, Value::Int(i)) => ConstFold::Folded(Value::Ref(GcRef(i as usize))),
        _ => ConstFold::Unregistered,
    }
}

/// Narrow [`execute_varargs`] carve-out usable without `&mut MetaInterp`.
///
/// RPython `executor.execute_varargs(cpu, ..., exc=False)`
/// (`executor.py:75-78`) skips the `metainterp.execute_raised` coordination
/// when the helper provably cannot raise — `MIFrame.execute_varargs` with
/// `pure=True` reaches this path for every `EF_ELIDABLE_CANNOT_RAISE`
/// callee. Pyre's walker (`pyre-jit-trace::jitcode_dispatch::
/// dispatch_residual_call_*`) cannot thread `&mut MetaInterp` through the
/// trace recorder seam, so this helper exposes the `exc=False` shape via
/// `cpu.bh_call_*`.
///
/// **Caller contract** (debug-asserted): `descr.get_extra_info()` must report
/// both `check_is_elidable()` true AND `check_can_raise(false)` false.  Any
/// other EI risks landing in `BH_LAST_EXC_VALUE` with no metainterp around
/// to transcribe it, which would silently swallow the exception.
///
/// `args` follows `_build_allboxes` (`pyjitpl.py`) layout
/// **excluding** the funcbox: the funcbox concrete int is `func_ptr` and the
/// remaining concrete operand values pass straight through to
/// `cpu.bh_call_*` (`executor.py _do_call`).  Up to `MAX_HOST_CALL_ARITY`
/// (16) operand slots.
pub fn execute_pure_call(
    descr: &dyn majit_ir::descr::CallDescr,
    func_ptr: i64,
    args: &[i64],
) -> i64 {
    debug_assert!(
        descr.get_extra_info().check_is_elidable()
            && !descr.get_extra_info().check_can_raise(false),
        "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI"
    );
    do_call_positional(descr, func_ptr, args)
}

/// `executor.execute_varargs` parity for the walker layer, simplified
/// for callers that do not hold a `MetaInterp` (record-time concrete
/// execution of residual_calls).
///
/// PyPy upstream (`rpython/jit/metainterp/pyjitpl.py`
/// `do_residual_call`) calls `executor.execute_varargs(opnum,
/// argboxes, descr, exc=can_raise, pure=is_elidable)` for every
/// residual_call regardless of EI branch — concrete execution always
/// runs at trace-record time, only the recorded opcode kind and the
/// post-call guard emission differ.  See `select_residual_call_opcode`
/// (`pyre-jit-trace/src/jitcode_dispatch.rs`) for the per-EI-branch
/// inventory.
///
/// The walker has no `MetaInterp` to thread; instead its per-walk
/// `WalkSession.last_exc_value` is that slot.  This function therefore
/// returns the BH_LAST_EXC_VALUE seam value via `Result::Err` so the
/// caller can write it into the walker's exc shadow + emit
/// `GUARD_NO_EXCEPTION`, mirroring upstream's
/// `metainterp.execute_raised(bh_exc, constant=False)` +
/// `handle_possible_exception` flow.
///
/// Differences from `execute_varargs`:
///   * No `MetaInterp` argument; caller handles exception state.
///   * No `cond_call` / `cond_call_value` dispatch — residual_call
///     opcodes are not cond-call variants.
///   * Returns `Result<i64, i64>` instead of side-effecting
///     `metainterp.execute_raised`.
///
/// Differences from `execute_pure_call`:
///   * No `is_elidable + cannot_raise` debug_assert.
///   * Clears BH_LAST_EXC_VALUE before dispatch (`execute_pure_call`
///     does not because elidable_cannot_raise cannot raise).
pub fn execute_residual_call(
    descr: &dyn majit_ir::descr::CallDescr,
    func_ptr: i64,
    args: &[i64],
) -> Result<i64, i64> {
    crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let result = do_call_positional(descr, func_ptr, args);
    let bh_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 { Err(bh_exc) } else { Ok(result) }
}

#[cfg(test)]
mod execute_residual_call_tests {
    use super::*;
    use majit_ir::descr::SimpleCallDescr;
    use majit_ir::{EffectInfo, ExtraEffect, Type};

    extern "C" fn add2_i64(a: i64, b: i64) -> i64 {
        a.wrapping_add(b)
    }

    // A may-force helper that raises: publishes a non-zero exception
    // pointer on `BH_LAST_EXC_VALUE` (the blackhole CALL_* convention)
    // and returns the 0 result sentinel, exactly as a raising
    // `bh_call_*` arm does.
    extern "C" fn raises_stopiteration() -> i64 {
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xDEAD_BEEF));
        0
    }

    fn make_may_force_descr(arg_types: Vec<Type>, result_type: Type) -> SimpleCallDescr {
        let effect = EffectInfo {
            // EF_CAN_RAISE / forces — the non-pure classification.
            extraeffect: ExtraEffect::CanRaise,
            ..EffectInfo::default()
        };
        SimpleCallDescr::new(0, arg_types, result_type, false, 8, effect)
    }

    #[test]
    fn non_raising_call_returns_ok_result() {
        let descr = make_may_force_descr(vec![Type::Int, Type::Int], Type::Int);
        let r = execute_residual_call(&descr, add2_i64 as *const () as i64, &[40, 2]);
        assert_eq!(
            r,
            Ok(42),
            "a non-raising add2_i64(40, 2) must return Ok(42)"
        );
    }

    #[test]
    fn raising_call_reports_published_exception_pointer() {
        let descr = make_may_force_descr(vec![], Type::Int);
        let r = execute_residual_call(&descr, raises_stopiteration as *const () as i64, &[]);
        assert_eq!(
            r,
            Err(0xDEAD_BEEF),
            "the helper's BH_LAST_EXC_VALUE publication must surface as Err"
        );
    }

    #[test]
    fn clears_published_exception_after_consuming_it() {
        // The Err return consumes the publication; the TLS slot must not
        // keep the pointer for the next (unrelated) call to observe.
        let descr = make_may_force_descr(vec![], Type::Int);
        let r = execute_residual_call(&descr, raises_stopiteration as *const () as i64, &[]);
        assert_eq!(r, Err(0xDEAD_BEEF));
        let stale = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
        assert_eq!(
            stale, 0,
            "BH_LAST_EXC_VALUE must be cleared after the exception is consumed"
        );
    }

    #[test]
    fn clears_stale_exception_before_dispatch() {
        // A prior call's exception must not bleed into a clean call.
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0x1234));
        let descr = make_may_force_descr(vec![Type::Int, Type::Int], Type::Int);
        let r = execute_residual_call(&descr, add2_i64 as *const () as i64, &[1, 2]);
        assert_eq!(
            r,
            Ok(3),
            "stale BH_LAST_EXC_VALUE must be cleared before dispatch"
        );
    }
}

#[cfg(test)]
mod do_cond_call_tests {
    use super::*;
    use majit_ir::descr::SimpleCallDescr;
    use majit_ir::{EffectInfo, ExtraEffect, Type};
    use std::sync::atomic::{AtomicI64, Ordering};

    static LAST: AtomicI64 = AtomicI64::new(0);

    extern "C" fn probe(vable: i64) {
        LAST.store(vable, Ordering::SeqCst);
    }

    fn void_ref_descr() -> SimpleCallDescr {
        SimpleCallDescr::new(
            0,
            vec![Type::Ref],
            Type::Void,
            false,
            0,
            EffectInfo {
                extraeffect: ExtraEffect::CannotRaise,
                ..EffectInfo::default()
            },
        )
    }

    #[test]
    fn do_cond_call_respects_the_condition() {
        LAST.store(0, Ordering::SeqCst);
        let descr = void_ref_descr();
        do_cond_call(None, 0, probe as *const () as i64, &[0xBEEF], &descr);
        assert_eq!(LAST.load(Ordering::SeqCst), 0);
        do_cond_call(None, 1, probe as *const () as i64, &[0xBEEF], &descr);
        assert_eq!(LAST.load(Ordering::SeqCst), 0xBEEF);
    }
}

#[cfg(test)]
mod execute_pure_call_tests {
    use super::*;
    use majit_ir::descr::SimpleCallDescr;
    use majit_ir::{EffectInfo, ExtraEffect, Type};

    extern "C" fn double_i64(x: i64) -> i64 {
        x.wrapping_mul(2)
    }

    extern "C" fn add3_i64(a: i64, b: i64, c: i64) -> i64 {
        a.wrapping_add(b).wrapping_add(c)
    }

    extern "C" fn double_f64(x: f64) -> f64 {
        x * 2.0
    }

    /// `jit_bigint_to_f64_or_inf`'s shape: an integer/ref parameter with a
    /// genuine `f64` return, so the result arrives in xmm0 / d0 while nothing
    /// occupies the FP argument file.
    extern "C" fn halve_int_to_f64(x: i64) -> f64 {
        x as f64 / 2.0
    }

    extern "C" fn void_no_op(_x: i64) {}

    /// An Int result with a Float parameter — the direction the Float-result
    /// work did not cover.  The return encodes BOTH arguments: a callee taking
    /// a single `f64` cannot tell a correct call from a wrong-register one,
    /// because the value it wants may happen to be left in xmm0 by the caller.
    extern "C" fn float_then_int_to_i64(x: f64, n: i64) -> i64 {
        x as i64 * 100 + n
    }

    /// The interleaved void shape.  Records both arguments so the test can tell
    /// a right-register call from one that shuffled them.
    static VOID_INTERLEAVED_SEEN: std::sync::atomic::AtomicI64 =
        std::sync::atomic::AtomicI64::new(0);

    extern "C" fn void_float_then_int(x: f64, n: i64) {
        VOID_INTERLEAVED_SEEN.store(x as i64 * 100 + n, std::sync::atomic::Ordering::SeqCst);
    }

    fn make_descr(arg_types: Vec<Type>, result_type: Type) -> SimpleCallDescr {
        let effect = EffectInfo {
            extraeffect: ExtraEffect::ElidableCannotRaise,
            ..EffectInfo::default()
        };
        SimpleCallDescr::new(0, arg_types, result_type, false, 8, effect)
    }

    #[test]
    fn executes_single_int_arg_and_returns_doubled_result() {
        let descr = make_descr(vec![Type::Int], Type::Int);
        let result = execute_pure_call(&descr, double_i64 as *const () as i64, &[21]);
        assert_eq!(result, 42, "double_i64(21) must return 42");
    }

    #[test]
    fn executes_three_int_args_routing_through_call_int_function() {
        let descr = make_descr(vec![Type::Int, Type::Int, Type::Int], Type::Int);
        let result = execute_pure_call(&descr, add3_i64 as *const () as i64, &[100, 20, 3]);
        assert_eq!(result, 123, "add3_i64(100, 20, 3) must return 123");
    }

    /// `executor.py` hands `descr` to `cpu.bh_call_i` just as `:66-72`
    /// does to `cpu.bh_call_f`, so a Float parameter is class-`'f'` whatever
    /// the result type is.
    #[test]
    fn int_result_reads_a_float_argument_from_the_float_register_file() {
        let descr = make_descr(vec![Type::Float, Type::Int], Type::Int);
        let result = execute_pure_call(
            &descr,
            float_then_int_to_i64 as *const () as i64,
            &[7.0_f64.to_bits() as i64, 3],
        );
        assert_eq!(
            result, 703,
            "float_then_int_to_i64(7.0, 3) == 703; an all-i64 pass hands the \
             callee the raw bits as its integer argument"
        );
    }

    /// `executor.py cpu.bh_call_v` takes the same `descr`, so a void
    /// callee's Float parameter is placed the same way.
    #[test]
    fn void_result_places_an_interleaved_float_argument_in_order() {
        VOID_INTERLEAVED_SEEN.store(0, std::sync::atomic::Ordering::SeqCst);
        let descr = make_descr(vec![Type::Float, Type::Int], Type::Void);
        execute_pure_call(
            &descr,
            void_float_then_int as *const () as i64,
            &[7.0_f64.to_bits() as i64, 3],
        );
        assert_eq!(
            VOID_INTERLEAVED_SEEN.load(std::sync::atomic::Ordering::SeqCst),
            703,
            "void_float_then_int(7.0, 3) must see 7.0 and 3 in declaration order"
        );
    }

    /// A Float result is fetched through `cpu.bh_call_f` (`executor.py`),
    /// i.e. the floating-point return register.  What puts a Float *argument*
    /// in the FP argument file is the stub's `lltype.Float` parameter
    /// (`descr.py:604-605 ARGS`/`FUNC`).  `args` carries the raw `f64::to_bits`
    /// in both directions (`longlong` float-storage parity).
    #[test]
    fn float_result_and_float_arg_use_the_floating_point_register_file() {
        let descr = make_descr(vec![Type::Float], Type::Float);
        let result = execute_pure_call(
            &descr,
            double_f64 as *const () as i64,
            &[3.5_f64.to_bits() as i64],
        );
        assert_eq!(f64::from_bits(result as u64), 7.0, "double_f64(3.5) == 7.0");
    }

    /// The mixed shape that motivated the fix: an integer parameter with an
    /// `f64` return.  The integer return register is undefined after a call to
    /// an `f64`-returning callee, so reading it here read residue rather than
    /// the result.
    #[test]
    fn float_result_with_an_integer_arg_reads_the_float_return_register() {
        let descr = make_descr(vec![Type::Int], Type::Float);
        let result = execute_pure_call(&descr, halve_int_to_f64 as *const () as i64, &[7]);
        assert_eq!(
            f64::from_bits(result as u64),
            3.5,
            "halve_int_to_f64(7) == 3.5; an integer-register read would see 7",
        );
    }

    #[test]
    fn void_return_routes_through_call_void_function_and_returns_zero_sentinel() {
        let descr = make_descr(vec![Type::Int], Type::Void);
        let result = execute_pure_call(&descr, void_no_op as *const () as i64, &[99]);
        assert_eq!(result, 0, "void execute_pure_call returns the 0 sentinel");
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI")]
    fn non_elidable_ei_panics_debug_assertion() {
        let effect = EffectInfo {
            extraeffect: ExtraEffect::CannotRaise,
            ..EffectInfo::default()
        };
        let descr = SimpleCallDescr::new(0, vec![Type::Int], Type::Int, false, 8, effect);
        let _ = execute_pure_call(&descr, double_i64 as *const () as i64, &[1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI")]
    fn elidable_can_raise_panics_debug_assertion() {
        let effect = EffectInfo {
            extraeffect: ExtraEffect::ElidableCanRaise,
            ..EffectInfo::default()
        };
        let descr = SimpleCallDescr::new(0, vec![Type::Int], Type::Int, false, 8, effect);
        let _ = execute_pure_call(&descr, double_i64 as *const () as i64, &[1]);
    }
}

#[cfg(test)]
mod execute_nonspec_const_tests {
    use super::*;
    use majit_ir::descr::{DescrRef, SimpleFieldDescr};
    use majit_ir::{Type, Value};
    use std::sync::Arc;

    /// The getfield fold arm must accept the plain spelling: the
    /// executor helper is registered under the plain opnum upstream,
    /// and `heap.py optimize_GETFIELD_GC_I` folds a plain-spelled
    /// op whenever `descr.is_always_pure()` holds and the base becomes
    /// constant. A plain-only miss here surfaces as the
    /// `constant_fold` "no helper registered" panic.
    #[test]
    fn getfield_gc_fold_accepts_plain_spelling() {
        let cpu = crate::cpu::default_cpu();
        // One-field struct layout: header word + i64 payload at offset 8.
        let storage: Box<[i64; 2]> = Box::new([0, 42]);
        let base = storage.as_ref().as_ptr() as usize;
        let descr: DescrRef = Arc::new(SimpleFieldDescr::new(0, 8, 8, Type::Int, true));
        let opnum = OpCode::GetfieldGcI;
        let folded = execute_nonspec_const(
            cpu.as_ref(),
            opnum,
            &[Value::Ref(majit_ir::GcRef(base))],
            Some(&descr),
            Type::Int,
        );
        match folded {
            Ok(Some(Value::Int(42))) => {}
            other => panic!("{opnum:?} must fold to Ok(Some(Int(42))), got {other:?}"),
        }
    }

    /// Every registered helper that refuses its operands must answer
    /// `Ok(None)`. `Err(NoConstExecutor)` is upstream's
    /// `NotImplementedError`, and `Optimizer::constant_fold` spells it as a
    /// `panic!` — so conflating the two turns a decline into a crash.
    #[test]
    fn a_registered_helper_that_refuses_its_operands_answers_ok_none() {
        let cpu = crate::cpu::default_cpu();
        let declines: &[(OpCode, Vec<Value>, Type)] = &[
            // Shift count outside `blackhole.check_shift_count`'s range.
            (
                OpCode::IntLshift,
                vec![Value::Int(1), Value::Int(64)],
                Type::Int,
            ),
            (
                OpCode::IntRshift,
                vec![Value::Int(1), Value::Int(-1)],
                Type::Int,
            ),
            // Zero divisor.
            (
                OpCode::IntFloorDiv,
                vec![Value::Int(5), Value::Int(0)],
                Type::Int,
            ),
            (
                OpCode::IntMod,
                vec![Value::Int(5), Value::Int(0)],
                Type::Int,
            ),
            // OVF opcodes are spelled with `checked_*`.
            (
                OpCode::IntAddOvf,
                vec![Value::Int(i64::MAX), Value::Int(1)],
                Type::Int,
            ),
            (
                OpCode::IntSubOvf,
                vec![Value::Int(i64::MIN), Value::Int(1)],
                Type::Int,
            ),
            (
                OpCode::IntMulOvf,
                vec![Value::Int(i64::MAX), Value::Int(2)],
                Type::Int,
            ),
            // `test_optimizebasic.test_float_division_by_multiplication`
            // keeps `float_truediv(f, 0.0)` in the optimized loop.
            (
                OpCode::FloatTrueDiv,
                vec![Value::Float(1.0), Value::Float(0.0)],
                Type::Float,
            ),
            // Non-finite and out-of-range `cast_float_to_int`.
            (
                OpCode::CastFloatToInt,
                vec![Value::Float(f64::INFINITY)],
                Type::Int,
            ),
            (
                OpCode::CastFloatToInt,
                vec![Value::Float(f64::NAN)],
                Type::Int,
            ),
            (OpCode::CastFloatToInt, vec![Value::Float(1e30)], Type::Int),
        ];
        for (opnum, args, ty) in declines {
            match execute_nonspec_const(cpu.as_ref(), *opnum, args, None, *ty) {
                Ok(None) => {}
                other => panic!("{opnum:?}{args:?} must decline with Ok(None), got {other:?}"),
            }
        }
    }

    /// The counterpart: an opcode with no row at all still reaches
    /// `Err(NoConstExecutor)`, so the distinction is real in both directions.
    #[test]
    fn an_opcode_with_no_row_still_answers_err() {
        let cpu = crate::cpu::default_cpu();
        let folded = execute_nonspec_const(
            cpu.as_ref(),
            OpCode::SetfieldGc,
            &[Value::Int(1), Value::Int(2)],
            None,
            Type::Void,
        );
        assert_eq!(folded, Err(NoConstExecutor));
    }
}
