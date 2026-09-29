//! Trace-time box / unbox / binop / compare recording for boxed primitives.
//!
//! Interpreter-agnostic: each helper takes the boxed primitive's type
//! address and field descrs as parameters and emits only generic IR —
//! `GuardClass` + `GetfieldGc` to unbox, the typed op plus an overflow
//! guard, and `NewWithVtable` + `SetfieldGc` to rebox. Per-operator
//! dispatch and concrete computation stay with the interpreter; only the
//! recording sequence lives here. (`int`/`float` name the IR primitive
//! width, not any interpreter object type.)

/// Unbox a Python int object: emit GuardClass + GetfieldGc(I|PureI).
///
/// Auto-generated equivalent of PyPy's int_unbox annotation.
/// Returns the raw i64 OpRef, with heapcache integration.
fn getfield_gc_i_pureornot(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    use majit_ir::Value;
    // heapcache: check if this field was already read/written in this trace
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (int arm). The
        // line-by-line port runs `executor.execute(cpu, mi, opnum,
        // fielddescr, box)` and asserts `resvalue ==
        // upd.currfieldbox.getint()`. Pyre projects the struct
        // pointer through `concrete_of_opref(obj)`; the cached Box's
        // intrinsic value is fetched via `box_value(cached)` —
        // covering const pool, standard-virtualizable shadow, and
        // the frontend object's `value` field (RPython
        // `currfieldbox.getint()` dispatch parity).
        let expected_int = match ctx.box_value(cached) {
            Some(Value::Int(n)) => Some(n),
            _ => None,
        };
        if let Some(cached_int) = expected_int
            && let Some(Value::Ref(struct_ref)) = ctx.box_value(obj)
        {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64
                && struct_ptr != 0
                && let Some(Value::Int(loaded)) =
                    ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
            {
                assert_eq!(
                    loaded, cached_int,
                    "_opimpl_getfield_gc_any_pureornot sanity \
                             check (int): loaded {loaded} != cached \
                             {cached_int} (field_index={field_index}, \
                             struct_ptr={struct_ptr:#x})"
                );
            }
        }
        // pyjitpl.py:946 profiler.count_ops(rop.GETFIELD_GC_I,
        // Counters.HEAPCACHED_OPS) — folded-away op accounting on cache hit.
        ctx.profiler()
            .count_ops(OpCode::GetfieldGcI, crate::counters::HEAPCACHED_OPS);
        return cached;
    }
    // Record the plain GETFIELD_GC_I opnum regardless of
    // `descr.is_always_pure()`; purity is re-derived from the descr by
    // OptHeap. There is no pure getfield opnum.
    // pyjitpl.py cache-miss `execute_with_descr` routes through
    // `execute_and_record`, which counts OPS and RECORDED_OPS.
    ctx.profiler()
        .count_ops(OpCode::GetfieldGcI, crate::counters::OPS);
    ctx.profiler()
        .count_ops(OpCode::GetfieldGcI, crate::counters::RECORDED_OPS);
    let live_value = if let Some(Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
        } else {
            None
        }
    } else {
        None
    };
    let result =
        ctx.record_op_with_descr_value(OpCode::GetfieldGcI, &[obj], descr.clone(), live_value);
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

pub fn trace_unbox_int(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    int_type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    // GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0 (llgraph/runner.py execute_guard_class).
    // Callers that route through `trace_unbox_int_with_resume_descr`
    // already emit the GuardClass via
    // `frame.generate_guard` — by the time control reaches here,
    // `is_class_known(obj)` returns true and this branch is dead.  In
    // unit tests (`pyre/pyre-jit/src/trace_verify.rs`) the GuardClass is
    // recorded with no resume context; the test asserts only on the
    // emitted opcode shape.
    if !obj.is_constant() && !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(int_type_addr);
        ctx.record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        ctx.heap_cache_mut().class_now_known(obj);
    }
    getfield_gc_i_pureornot(ctx, obj, intval_descr)
}

/// Box a raw i64 into a Python int object.
///
/// Auto-generated equivalent of PyPy's int_box annotation.
/// The helper route preserves pyre's small-int cache instead of always
/// materializing a fresh heap object in the trace.
pub fn trace_box_int(
    ctx: &mut crate::TraceCtx,
    value: majit_ir::OpRef,
    size_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
    int_type_addr: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;

    // Inline W_Int allocation so OptVirtualize can see the object shape
    // and fold later GetfieldRawI(intval) reads back to `value`.
    // RPython parity: NEW_WITH_VTABLE (not NEW) for classes with vtable.
    // jtransform.py rewrite_op_setfield parity: typeptr setfield filtered in trace.
    // rewrite.py:479-484 GC rewriter emits vtable via fielddescr_vtable.
    let obj = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    // pyjitpl.py `execute_new_with_vtable` stamps BOTH halves —
    // `heapcache.new(resbox)` AND `heapcache.class_now_known(resbox)`; only
    // plain `execute_new` (:2720-2723) stops at `new`. The vtable the GC
    // rewriter writes from `size_descr` IS the class, so it is known by
    // construction and upstream records it unconditionally.
    ctx.heap_cache_mut().new_object(obj);
    ctx.heap_cache_mut().class_now_known(obj);
    let intval_idx = intval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[obj, value], intval_descr);
    // `upd.setfield(valuebox)` parity — the cache stores the Box
    // identity (`value` OpRef); cache-hit readers fetch the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setfield_cached(obj, intval_idx, value);
    obj
}

/// Residual `Ref -> Ref` helpers that return the argument when no
/// collection moved it (`reload_top_root`, `try_gc_current_object_address`).
/// `interp_jit.py PyFrame.dispatch` has no per-opcode frame reload; record
/// the identity instead of `CallR` when the live result equals the arg.
#[derive(Clone, Default)]
pub struct IdentityRefResidual {
    pub fnaddrs: Vec<i64>,
}

impl IdentityRefResidual {
    pub fn matches(&self, fnaddr: i64) -> bool {
        self.fnaddrs.contains(&fnaddr)
    }
}

static IDENTITY_REF_RESIDUAL: std::sync::OnceLock<IdentityRefResidual> = std::sync::OnceLock::new();

pub fn register_identity_ref_residual(spec: IdentityRefResidual) {
    let _ = IDENTITY_REF_RESIDUAL.set(spec);
}

pub fn identity_ref_residual() -> Option<&'static IdentityRefResidual> {
    IDENTITY_REF_RESIDUAL.get()
}

/// Residual `() -> Int` helpers whose result is process-constant after
/// first touch (`gc_interp::enabled`). `@elidable` (`rlib/jit.py`) folds
/// the traced concrete; without a live LLBC extract the frozen jitcode
/// still residualizes, so interpret records the constant instead of `CallI`.
#[derive(Clone, Default)]
pub struct ElidableIntResidual {
    pub fnaddrs: Vec<i64>,
}

impl ElidableIntResidual {
    pub fn matches(&self, fnaddr: i64) -> bool {
        self.fnaddrs.contains(&fnaddr)
    }
}

static ELIDABLE_INT_RESIDUAL: std::sync::OnceLock<ElidableIntResidual> = std::sync::OnceLock::new();

pub fn register_elidable_int_residual(spec: ElidableIntResidual) {
    let _ = ELIDABLE_INT_RESIDUAL.set(spec);
}

pub fn elidable_int_residual() -> Option<&'static ElidableIntResidual> {
    ELIDABLE_INT_RESIDUAL.get()
}

/// Residual `() -> Void` helpers that are no-ops on the recorded
/// trace when they succeed (`frame_anchor_release`, `stack_check`).
/// `interp_jit.py` `dispatch` has neither; compiled loops poll the
/// breaker on the back-edge instead.
#[derive(Clone, Default)]
pub struct VoidSkipResidual {
    pub fnaddrs: Vec<i64>,
}

impl VoidSkipResidual {
    pub fn matches(&self, fnaddr: i64) -> bool {
        self.fnaddrs.contains(&fnaddr)
    }
}

static VOID_SKIP_RESIDUAL: std::sync::OnceLock<VoidSkipResidual> = std::sync::OnceLock::new();

pub fn register_void_skip_residual(spec: VoidSkipResidual) {
    let _ = VOID_SKIP_RESIDUAL.set(spec);
}

pub fn void_skip_residual() -> Option<&'static VoidSkipResidual> {
    VOID_SKIP_RESIDUAL.get()
}

/// Residual `Int -> Ref` helpers that return the standard virtualizable
/// frame (`frame_anchor_live`). `interp_jit.py` has no shadow-stack
/// slot; the frame is the loop's red input. Reuse that OpRef when the
/// live concrete equals the vable pointer.
#[derive(Clone, Default)]
pub struct FrameAnchorLiveResidual {
    pub fnaddrs: Vec<i64>,
}

impl FrameAnchorLiveResidual {
    pub fn matches(&self, fnaddr: i64) -> bool {
        self.fnaddrs.contains(&fnaddr)
    }
}

static FRAME_ANCHOR_LIVE_RESIDUAL: std::sync::OnceLock<FrameAnchorLiveResidual> =
    std::sync::OnceLock::new();

pub fn register_frame_anchor_live_residual(spec: FrameAnchorLiveResidual) {
    let _ = FRAME_ANCHOR_LIVE_RESIDUAL.set(spec);
}

pub fn frame_anchor_live_residual() -> Option<&'static FrameAnchorLiveResidual> {
    FRAME_ANCHOR_LIVE_RESIDUAL.get()
}

/// Portal-interpret exception-state residuals: the EC `sys_exc_value`
/// field ops and the builtin exception-class `LOAD_GLOBAL`.
pub struct ExceptionTraceResidual {
    /// Canonical exception class from a residual `LOAD_GLOBAL`.
    /// `pyopcode.py LOAD_GLOBAL` looks inside the module-dict cell.
    pub load_global_fnaddrs: Vec<i64>,
    pub emit_load_global_exc:
        fn(ctx: &mut crate::TraceCtx, raw_i: &[i64]) -> Option<(majit_ir::OpRef, i64)>,
    /// `eval.rs get_current_exception` / `set_current_exception`.
    /// FBW `try_walker_lower_exc_info_residual` emits EC field ops.
    pub get_current_exception_fnaddrs: Vec<i64>,
    pub set_current_exception_fnaddrs: Vec<i64>,
    pub emit_get_current_exception:
        fn(ctx: &mut crate::TraceCtx, ec: majit_ir::OpRef) -> (majit_ir::OpRef, i64),
    pub emit_set_current_exception:
        fn(ctx: &mut crate::TraceCtx, ec: majit_ir::OpRef, exc: majit_ir::OpRef, exc_ptr: i64),
    /// Live `ExecutionContext` pointer (`interp_jit.py reds = ['frame', 'ec']`).
    pub current_ec_ptr: fn() -> i64,
}

impl ExceptionTraceResidual {
    pub fn matches_load_global(&self, fnaddr: i64) -> bool {
        self.load_global_fnaddrs.contains(&fnaddr)
    }

    pub fn matches_get_current_exception(&self, fnaddr: i64) -> bool {
        self.get_current_exception_fnaddrs.contains(&fnaddr)
    }

    pub fn matches_set_current_exception(&self, fnaddr: i64) -> bool {
        self.set_current_exception_fnaddrs.contains(&fnaddr)
    }
}

static EXCEPTION_TRACE_RESIDUAL: std::sync::OnceLock<ExceptionTraceResidual> =
    std::sync::OnceLock::new();

pub fn register_exception_trace_residual(spec: ExceptionTraceResidual) {
    let _ = EXCEPTION_TRACE_RESIDUAL.set(spec);
}

pub fn exception_trace_residual() -> Option<&'static ExceptionTraceResidual> {
    EXCEPTION_TRACE_RESIDUAL.get()
}

/// `compare_op_from_tag` 0..=5 → `IntLt`/`IntLe`/`IntGt`/`IntGe`/`IntEq`/`IntNe`.
pub fn int_compare_op_kind(tag: i64) -> Option<majit_ir::OpCode> {
    Some(match tag {
        0 => majit_ir::OpCode::IntLt,
        1 => majit_ir::OpCode::IntLe,
        2 => majit_ir::OpCode::IntGt,
        3 => majit_ir::OpCode::IntGe,
        4 => majit_ir::OpCode::IntEq,
        5 => majit_ir::OpCode::IntNe,
        _ => return None,
    })
}

/// Emit an overflow-checked binary int operation.
///
/// Auto-generated: unbox a, unbox b, emit ovf op, guard no overflow, box result.
pub fn trace_int_binop_ovf(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    int_type_addr: i64,
    intval_descr: majit_ir::DescrRef,
    size_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    let a_val = trace_unbox_int(ctx, a, int_type_addr, intval_descr.clone());
    let b_val = trace_unbox_int(ctx, b, int_type_addr, intval_descr.clone());
    let folded = match (ctx.box_value(a_val), ctx.box_value(b_val)) {
        (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) => {
            Some(majit_ir::Value::Int(crate::eval_binop_i(opcode, la, rb)))
        }
        _ => None,
    };
    let result = ctx.record_op_with_value(opcode, &[a_val, b_val], folded);
    // No production caller of this AST→trace helper: pyre-jit-trace
    // routes overflow guards through the retired int fast path's
    // `frame.generate_guard` instead.  The bare
    // `record_guard` keeps the guard shape correct in unit tests
    // (`trace_verify.rs`) without a synthetic snapshot — convergence
    // path is convergence to register-machine jitcode.
    ctx.record_guard(OpCode::GuardNoOverflow, &[], 0);
    trace_box_int(ctx, result, size_descr, intval_descr, int_type_addr)
}

/// Emit a non-overflow binary int operation (bitwise ops, shifts).
pub fn trace_int_binop(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    int_type_addr: i64,
    intval_descr: majit_ir::DescrRef,
    size_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_int(ctx, a, int_type_addr, intval_descr.clone());
    let b_val = trace_unbox_int(ctx, b, int_type_addr, intval_descr.clone());
    let folded = match (ctx.box_value(a_val), ctx.box_value(b_val)) {
        (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) => {
            Some(majit_ir::Value::Int(crate::eval_binop_i(opcode, la, rb)))
        }
        _ => None,
    };
    let result = ctx.record_op_with_value(opcode, &[a_val, b_val], folded);
    trace_box_int(ctx, result, size_descr, intval_descr, int_type_addr)
}

/// Emit a comparison between two Python ints.
pub fn trace_int_compare(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    int_type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_int(ctx, a, int_type_addr, intval_descr.clone());
    let b_val = trace_unbox_int(ctx, b, int_type_addr, intval_descr.clone());
    let folded = match (ctx.box_value(a_val), ctx.box_value(b_val)) {
        (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) => {
            Some(majit_ir::Value::Int(crate::eval_binop_i(opcode, la, rb)))
        }
        _ => None,
    };
    ctx.record_op_with_value(opcode, &[a_val, b_val], folded)
}

/// Unbox a Python float object: emit GuardClass + GetfieldGc(F|PureF).
///
/// Returns the raw f64 OpRef.
pub fn getfield_gc_f_pureornot(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    use majit_ir::Value;
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:941-945 cache-hit sanity check (float arm).
        // ConstFloat.same_constant compares via longlong.extract_bits
        // (history.py:283-294); pyre's Value Eq for Float uses
        // to_bits — bit-identical, NaN==NaN, 0.0!=-0.0.  The cached
        // Box's intrinsic value is fetched via `box_value(cached)` —
        // covering const pool, standard-virtualizable shadow, and
        // the frontend object's `value` field (RPython
        // `currfieldbox.getfloat_storage()` dispatch parity).
        let expected_float = match ctx.box_value(cached) {
            Some(Value::Float(f)) => Some(f),
            _ => None,
        };
        if let Some(cached_float) = expected_float
            && let Some(Value::Ref(struct_ref)) = ctx.box_value(obj)
        {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64
                && struct_ptr != 0
                && let Some(Value::Float(loaded)) =
                    ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Float)
            {
                assert_eq!(
                    loaded.to_bits(),
                    cached_float.to_bits(),
                    "_opimpl_getfield_gc_any_pureornot sanity \
                             check (float): loaded {loaded} != cached \
                             {cached_float} (field_index={field_index}, \
                             struct_ptr={struct_ptr:#x})"
                );
            }
        }
        // pyjitpl.py:946 profiler.count_ops(rop.GETFIELD_GC_I,
        // Counters.HEAPCACHED_OPS) — the opnum literal in upstream is
        // GETFIELD_GC_I regardless of type, so wire the float variant
        // to the same counter bucket.
        ctx.profiler()
            .count_ops(OpCode::GetfieldGcI, crate::counters::HEAPCACHED_OPS);
        return cached;
    }
    // Record the plain GETFIELD_GC_F opnum regardless of
    // `descr.is_always_pure()`; purity is re-derived from the descr by
    // OptHeap. There is no pure getfield opnum.
    // pyjitpl.py cache-miss `execute_with_descr` routes through
    // `execute_and_record`, which counts OPS and RECORDED_OPS.
    ctx.profiler()
        .count_ops(OpCode::GetfieldGcF, crate::counters::OPS);
    ctx.profiler()
        .count_ops(OpCode::GetfieldGcF, crate::counters::RECORDED_OPS);
    let live_value = if let Some(Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Float)
        } else {
            None
        }
    } else {
        None
    };
    let result =
        ctx.record_op_with_descr_value(OpCode::GetfieldGcF, &[obj], descr.clone(), live_value);
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

pub fn trace_unbox_float(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    float_type_addr: i64,
    floatval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    if !obj.is_constant() && !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(float_type_addr);
        ctx.record_guard(OpCode::GuardClass, &[obj, type_const], 0);
        ctx.heap_cache_mut().class_now_known(obj);
    }
    getfield_gc_f_pureornot(ctx, obj, floatval_descr)
}

/// Box a raw f64 into a Python float object: emit New + SetfieldGc.
pub fn trace_box_float(
    ctx: &mut crate::TraceCtx,
    value: majit_ir::OpRef,
    size_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
    float_type_addr: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    // RPython parity: NEW_WITH_VTABLE + jtransform.py typeptr filter +
    // rewrite.py GC rewriter fielddescr_vtable emission.
    let obj = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    // pyjitpl.py: `execute_new_with_vtable` records the class too.
    // Same reasoning as `trace_box_int`.
    ctx.heap_cache_mut().new_object(obj);
    ctx.heap_cache_mut().class_now_known(obj);
    let floatval_idx = floatval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[obj, value], floatval_descr);
    // `upd.setfield(valuebox)` parity — the cache stores the Box
    // identity (`value` OpRef); cache-hit readers fetch the intrinsic
    // value via `box_value(cached)` at hit time.
    ctx.heapcache_setfield_cached(obj, floatval_idx, value);
    obj
}

/// Emit a binary float operation: unbox a, unbox b, emit float op, box result.
pub fn trace_float_binop(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    float_type_addr: i64,
    floatval_descr: majit_ir::DescrRef,
    size_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_float(ctx, a, float_type_addr, floatval_descr.clone());
    let b_val = trace_unbox_float(ctx, b, float_type_addr, floatval_descr.clone());
    let folded = match (ctx.box_value(a_val), ctx.box_value(b_val)) {
        (Some(majit_ir::Value::Float(a)), Some(majit_ir::Value::Float(b))) => {
            let bits = crate::eval_binop_f(opcode, a.to_bits() as i64, b.to_bits() as i64);
            Some(majit_ir::Value::Float(f64::from_bits(bits as u64)))
        }
        _ => None,
    };
    let result = ctx.record_op_with_value(opcode, &[a_val, b_val], folded);
    trace_box_float(ctx, result, size_descr, floatval_descr, float_type_addr)
}

/// Emit a comparison between two Python floats.
pub fn trace_float_compare(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    float_type_addr: i64,
    floatval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_float(ctx, a, float_type_addr, floatval_descr.clone());
    let b_val = trace_unbox_float(ctx, b, float_type_addr, floatval_descr.clone());
    let folded = match (ctx.box_value(a_val), ctx.box_value(b_val)) {
        (Some(majit_ir::Value::Float(a)), Some(majit_ir::Value::Float(b))) => {
            Some(majit_ir::Value::Int(crate::eval_float_cmp(
                opcode,
                a.to_bits() as i64,
                b.to_bits() as i64,
            )))
        }
        _ => None,
    };
    ctx.record_op_with_value(opcode, &[a_val, b_val], folded)
}
