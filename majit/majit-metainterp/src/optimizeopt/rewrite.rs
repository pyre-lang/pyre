use indexmap::IndexMap;
use majit_ir::operand::Operand;
/// OptRewrite: algebraic simplification and constant folding.
///
/// Translated from rpython/jit/metainterp/optimizeopt/rewrite.py.
/// Rewrites operations into equivalent, cheaper operations.
/// This includes constant folding for pure ops and algebraic identities.
use majit_ir::{Const, Op, OpCode, OpRc, OpRef, Value};

use crate::optimizeopt::info::{PreambleOp, PtrInfo, PtrInfoExt};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult, intdiv};

/// rewrite.py: loop_invariant_results value.
/// RPython stores PreambleOp or regular Box (AbstractResOp) directly
/// in the dict. In Rust, we use an enum to distinguish.
#[derive(Clone, Debug)]
enum LoopInvariantEntry {
    /// Regular result (already forced or body-computed).
    Direct(OpRef),
    /// shortpreamble.py: LoopInvariantOp.produce_op stores
    /// PreambleOp(op, preamble_op, invented_name) in the dict.
    Preamble(PreambleOp),
}

/// Yield the `OptimizationResult::InvalidLoop` control value so the driver
/// (`propagate_from_pass_range`) converts it to `Err(InvalidLoop)` at the
/// pass barrier.  RPython `raise InvalidLoop`; threaded as a value here so
/// it works under `panic=abort`.  Use as `return raise_invalid_loop(msg, op, ctx)`.
///
/// The payload is a `&'static str`, so the abandon reason is structurally
/// unable to carry the operation it was raised on: the driver prints
/// `[jit] abort trace at key=… (InvalidLoop: <msg>)`, which names a rule and
/// no subject, and nothing downstream can rejoin the two.  Every caller here
/// decides on `arg(0)`, so log that operand and the constant it resolved to
/// at the point of decision — that resolved value *is* the proof the guard
/// can never pass, and it is gone by the time the driver sees the reason.
#[cold]
#[inline(never)]
fn raise_invalid_loop(msg: &'static str, op: &Op, ctx: &OptContext) -> OptimizationResult {
    if crate::majit_log_enabled() {
        let arg0 = op.arg(0);
        let resolved = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b));
        eprintln!(
            "[jit] InvalidLoop raised at {:?} pos={:?}: {} (arg0={:?} resolved_const={:?})",
            op.opcode,
            op.pos().get(),
            msg,
            arg0.get_box_replacement(false).to_opref(),
            resolved,
        );
    }
    OptimizationResult::InvalidLoop(msg)
}

/// optimizer.py `_check_subclass`: whether `vtable1` is a subclass of
/// `vtable2`.
///
/// The preorder numbering makes a subclass test a range test — every subclass
/// of a class numbers inside that class's own span.  The span includes its
/// `max`, which no real class is ever numbered with.
///
/// Answers `None` when either class has no recorded range, which decides
/// nothing: a caller must keep its guard rather than read that as a pass.
fn check_subclass(vtable1: i64, vtable2: i64) -> Option<bool> {
    // optimizer.py `_check_subclass`: read `subclassrange_min` / `max` off
    // the class constants when `cpu.subclassrange_min_offset` is set.
    // The span includes `max`; no real class is numbered with that value.
    if let Some(offset) = majit_backend::cpu_subclassrange_min_offset() {
        // A zero vtable is a class with no vtable object, so the guard stays.
        // `optimizer.py` `Optimizer._check_subclass` has no such class: it
        // always reads `subclassrange_min` / `subclassrange_max` off the class.
        if vtable1 == 0 || vtable2 == 0 {
            return None;
        }
        let (known_min, _) = majit_backend::read_vtable_subclass_range(vtable1, offset);
        let (expected_min, expected_max) =
            majit_backend::read_vtable_subclass_range(vtable2, offset);
        return Some(expected_min <= known_min && known_min <= expected_max);
    }
    let (known_min, _) = majit_gc::subclass_range(vtable1 as usize)?;
    let (expected_min, expected_max) = majit_gc::subclass_range(vtable2 as usize)?;
    Some(expected_min <= known_min && known_min <= expected_max)
}

/// The GC type id a `PtrInfo` descr names, or `None` when it names none.
///
/// Raw and headerless layouts carry no header word and so no id at all.  A
/// stamped `0` means "not resolved yet", not type zero, so it is recovered
/// through the structural cache key the same way the short preamble's guard
/// builder recovers it.
fn descr_gc_type_id(descr: &majit_ir::DescrRef) -> Option<u32> {
    if let Some(size) = descr.as_size_descr() {
        if !size.is_gc_managed() || size.headerless() {
            return None;
        }
        return crate::optimizeopt::info::resolve_gc_tid(
            size.type_id(),
            size.cache_key(),
            |cache, key| cache.resolve_struct_tid(key),
        );
    }
    let array = descr.as_array_descr()?;
    if !array.is_gc_managed() {
        return None;
    }
    crate::optimizeopt::info::resolve_gc_tid(array.type_id(), array.cache_key(), |cache, key| {
        cache.resolve_array_tid(key)
    })
}

/// info.py: INFO_NULL / INFO_NONNULL / INFO_UNKNOWN
/// optimizer.py: getnullness()
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nullness {
    Null,
    Nonnull,
    Unknown,
}

/// Rewrite operations into equivalent, cheaper forms.
///
/// Handles:
/// - Constant folding for pure integer/boolean ops
/// - Algebraic simplifications (identity, absorbing elements)
/// - Strength reduction for power-of-two `IntFloorDiv` / `IntMod`
///   (every other integer fold lives in OptIntBounds / OptPure)
/// - Guard simplification when argument is known constant
/// - Boolean operation rewrites (inverse/reflex)
/// - Conditional call elimination when condition/value is constant
/// - Pointer equality on same OpRef
/// - Cast and convert round-trip elimination
/// - Guard-no-exception removal after removed calls
pub struct OptRewrite {
    /// rewrite.py:39: loop_invariant_results — cache for CALL_LOOPINVARIANT results.
    /// Key: function pointer (arg0 as i64).
    /// Value: Direct(OpRef) or Preamble(PreambleOp) — RPython isinstance check.
    loop_invariant_results: indexmap::IndexMap<i64, LoopInvariantEntry>,
    /// rewrite.py:40: loop_invariant_producer — maps func_ptr → emitted Call op.
    /// Used by produce_potential_short_preamble_ops (rewrite.py).
    loop_invariant_producer: indexmap::IndexMap<i64, OpRc>,
    /// rewrite.py `CallLoopinvariantOptimizationResult` — the key its
    /// `_callback` records once the rewritten call has been emitted.
    pending_loopinvariant_key: Option<i64>,
}

/// The loop `rewrite.py` runs over an `intdiv` result: send each operation
/// from the head of the pass chain and take the last one, which computes the
/// value.
fn send_extra_operations(ctx: &mut OptContext, operations: Vec<majit_ir::OpRc>) -> Operand {
    let mut newop = None;
    for operation in operations {
        ctx.send_extra_operation_rc(operation.clone());
        newop = Some(operation);
    }
    Operand::from_bound_op(&newop.expect("intdiv returns at least two operations"))
}

impl OptRewrite {
    pub fn new() -> Self {
        OptRewrite {
            loop_invariant_results: indexmap::IndexMap::new(),
            loop_invariant_producer: indexmap::IndexMap::new(),
            pending_loopinvariant_key: None,
        }
    }

    // ── Constant folding for binary integer ops ──

    /// Constant-fold for the pyre-opcode strength-reduction rules below
    /// (`IntFloorDiv` / `IntMod` only — every other integer fold lives in
    /// OptIntBounds / OptPure, as upstream).
    fn try_fold_binary_int(&self, opcode: OpCode, lhs: i64, rhs: i64) -> Option<i64> {
        match opcode {
            OpCode::IntFloorDiv => {
                if rhs != 0 {
                    Some(lhs.wrapping_div(rhs))
                } else {
                    None
                }
            }
            OpCode::IntMod => {
                if rhs != 0 {
                    Some(lhs.wrapping_rem(rhs))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try algebraic simplification for INT_FLOORDIV.
    /// `x // 1 -> x`, constant fold when both operands are known.
    fn optimize_int_floor_div(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.resolve_operand_operand_opt(&arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.resolve_operand_operand_opt(&arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) && let Some(result) = self.try_fold_binary_int(OpCode::IntFloorDiv, a, b)
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(result));
            return OptimizationResult::Remove;
        }

        // x // 1 -> x (identity)
        if let Some(1) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = Operand::from_bound_op(op_rc);
            let b_arg = ctx.resolve_operand_operand(&arg0);
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // x // (-1) -> INT_NEG(x)
        if let Some(-1) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let mut neg = Op::new(OpCode::IntNeg, &[arg0]);
            neg.pos().set(op.pos().get());
            return OptimizationResult::Replace(neg);
        }

        // 0 // x -> 0 (zero dividend)
        if let Some(0) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x // x -> 1 (self-division, x != 0 guaranteed by semantics)
        if ctx
            .resolve_operand_operand(&arg0)
            .same_box(&ctx.resolve_operand_operand(&arg1))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(1));
            return OptimizationResult::Remove;
        }

        // Strength reduction for constant divisor >= 2.
        //
        // Both arms below compute a FLOOR quotient, which `IntFloorDiv` is
        // not: they are only equal to it for a non-negative dividend, so both
        // are gated on that.
        let known_nonneg = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.peek_intbound_box(&b))
            .is_some_and(|bound| bound.known_nonnegative());
        if let Some(divisor) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
            && known_nonneg
        {
            if divisor > 1 && divisor.count_ones() == 1 {
                // Power-of-2 division: x // (2^n) = x >> n
                let shift = divisor.trailing_zeros();
                let arg_shift = Operand::const_(Const::Int(shift as i64));
                let result_ref = ctx.emit(Op::new(OpCode::IntRshift, &[arg0, arg_shift.clone()]));
                let b_old = Operand::from_bound_op(op_rc);
                let b_res = ctx.get_box_replacement_operand(result_ref);
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }

            // General constant divisor >= 3: magic number multiplication
            if divisor >= 3 && ctx.supports_efficient_uint_mul_high {
                // rewrite.py:770 `known_nonneg = b1.known_nonnegative()`:
                // a non-negative dividend skips the sign-correction ops.
                let b_res = send_extra_operations(
                    ctx,
                    intdiv::division_operations(&arg0, divisor, known_nonneg),
                );
                let b_old = Operand::from_bound_op(op_rc);
                ctx.make_equal_to(&b_old, &b_res);
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// Try algebraic simplification for INT_MOD.
    ///
    /// Strength reduction from rpython/jit/metainterp/optimizeopt/intdiv.py.
    fn optimize_int_mod(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // Constant fold
        if let (Some(a), Some(b)) = (
            ctx.resolve_operand_operand_opt(&arg0)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            ctx.resolve_operand_operand_opt(&arg1)
                .and_then(|b| ctx.get_constant_int_box(&b)),
        ) && let Some(result) = self.try_fold_binary_int(OpCode::IntMod, a, b)
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(result));
            return OptimizationResult::Remove;
        }

        // x % 1 -> 0 (any integer mod 1 is 0)
        if let Some(1) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x % (-1) -> 0 (any integer mod -1 is 0)
        if let Some(-1) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // 0 % x -> 0 (zero dividend)
        if let Some(0) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // x % x -> 0 (self-modulo)
        if ctx
            .resolve_operand_operand(&arg0)
            .same_box(&ctx.resolve_operand_operand(&arg1))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            return OptimizationResult::Remove;
        }

        // Strength reduction for constant divisor >= 3 (non-power-of-2).
        //
        // `modulo_operations` computes a FLOOR remainder, which `IntMod` is
        // not, so the rewrite holds only for a non-negative dividend — the
        // case where the two agree.  rewrite.py:809
        // `known_nonneg = b1.known_nonnegative()` reads the same bound to
        // decide whether the expansion needs its sign-correction ops.
        let known_nonneg = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.peek_intbound_box(&b))
            .is_some_and(|bound| bound.known_nonnegative());
        if let Some(divisor) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
            && divisor >= 3
            && divisor.count_ones() != 1
            && known_nonneg
            && ctx.supports_efficient_uint_mul_high
        {
            let b_res =
                send_extra_operations(ctx, intdiv::modulo_operations(&arg0, divisor, known_nonneg));
            let b_old = Operand::from_bound_op(op_rc);
            ctx.make_equal_to(&b_old, &b_res);
            return OptimizationResult::Remove;
        }

        OptimizationResult::PassOn
    }

    /// Fold the tagged-int box/unbox round-trip
    /// `IntRshift(IntOr(IntAddOvf(x, x), 1), 1) => x`.
    ///
    /// 5b's box path emits `IntAddOvf(raw, raw)` (== 2*x, guarded by its
    /// `GuardNoOverflow` so the low bit is 0), then `IntOr(.., 1)` (set the
    /// tag bit, == 2*x + 1), and the next iteration re-unboxes with an
    /// arithmetic `IntRshift(.., 1)` (== x, since `(2x+1) >> 1 == x` for all
    /// non-overflowing i64 x, including negatives). The upstream
    /// `GuardNoOverflow` on the `IntAddOvf` makes `2*x` exact on the compiled
    /// fast path, so the whole chain reduces to `x`. The `IntAddOvf`/`IntOr`/
    /// `CastIntToPtr` box ops stay (the tagged store still needs the boxed
    /// value); only this redundant unbox is eliminated, and DCE reclaims the
    /// store leg if it later dies.
    ///
    /// Only the exact chain folds: the `IntOr` and `IntRshift` second args
    /// must both be the constant `1`, and the `IntAddOvf`'s two args must be
    /// the same box.
    fn optimize_int_rshift_tag_unbox(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        // IntRshift(A, 1): shift amount must be the constant 1.
        if ctx.get_constant_int_box(&op.arg(1).get_box_replacement(false)) != Some(1) {
            return None;
        }
        // A resolves to IntOr(B, 1): second arg must be the constant 1.
        let a = ctx.resolve_operand_operand(&op.arg(0));
        let or_op = ctx.get_producing_op(&a)?;
        if or_op.opcode != OpCode::IntOr {
            return None;
        }
        if ctx.get_constant_int_box(&or_op.arg(1).get_box_replacement(false)) != Some(1) {
            return None;
        }
        // B resolves to IntAddOvf(x, x): both args the same box.
        let b = ctx.resolve_operand_operand(&or_op.arg(0));
        let add_op = ctx.get_producing_op(&b)?;
        if add_op.opcode != OpCode::IntAddOvf {
            return None;
        }
        let x = ctx.resolve_operand_operand(&add_op.arg(0));
        if !x.same_box(&ctx.resolve_operand_operand(&add_op.arg(1))) {
            return None;
        }
        // The IntRshift result is x.
        let b_old = Operand::from_bound_op(op_rc);
        ctx.make_equal_to(&b_old, &x);
        Some(OptimizationResult::Remove)
    }

    // ── Unary operations ──

    /// Constant fold INT_IS_ZERO.
    /// rewrite.py `optimize_INT_IS_ZERO`:
    ///     return self._optimize_nullness(op, op.getarg(0), False)
    fn optimize_int_is_zero(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        self.optimize_nullness(op, op_rc, op.arg(0).to_opref(), false, ctx)
    }

    /// rewrite.py `optimize_INT_IS_TRUE`:
    ///     if (not self.is_raw_ptr(op.getarg(0)) and
    ///         self.getintbound(op.getarg(0)).is_bool()):
    ///         self.make_equal_to(op, op.getarg(0))
    ///         return
    ///     return self._optimize_nullness(op, op.getarg(0), True)
    fn optimize_int_is_true(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py optimize_INT_IS_TRUE:
        //     if (not self.is_raw_ptr(op.getarg(0)) and
        //         self.getintbound(op.getarg(0)).is_bool()):
        //         self.make_equal_to(op, op.getarg(0))
        //         return
        //     return self._optimize_nullness(op, op.getarg(0), True)
        //
        // is_raw_ptr (optimizer.py) checks for an
        // AbstractRawPtrInfo on the box's forwarded slot — NOT a
        // Ref-typed Box check. A raw-pointer 'i'-typed Box (i.e. one
        // pointing into a virtual raw buffer) skips the is_bool
        // shortcut because the buffer pointer's intbound is unrelated
        // to its boolean truthiness.
        let arg0_is_raw = ctx.is_raw_ptr(&op.arg(0).get_box_replacement(false));
        if !arg0_is_raw
            && let Some(bound) = ctx
                .resolve_operand_operand_opt(&arg0)
                .and_then(|b| ctx.peek_intbound_box(&b))
            && bound.is_bool()
        {
            // make_equal_to: replace INT_IS_TRUE result with arg0.
            let b_old = Operand::from_bound_op(op_rc);
            let b_arg = ctx.resolve_operand_operand(&arg0);
            ctx.make_equal_to(&b_old, &b_arg);
            return OptimizationResult::Remove;
        }

        // is_true_and_minint: int_is_true(int_and(x, MININT)) => int_lt(x, 0)
        if let Some(inner) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|pb| ctx.get_producing_op(&pb))
            && inner.opcode == OpCode::IntAnd
            && ctx.get_constant_int_box(&inner.arg(1).get_box_replacement(false)) == Some(i64::MIN)
        {
            let arg_zero = Operand::const_(majit_ir::Const::Int(0));
            let mut new_op = Op::new(OpCode::IntLt, &[inner.arg(0), arg_zero.clone()]);
            new_op.pos().set(op.pos().get());
            return OptimizationResult::Emit(new_op);
        }

        self.optimize_nullness(op, op_rc, arg0.to_opref(), true, ctx)
    }

    /// rewrite.py: _optimize_oois_ooisnot(op, expect_isnot, instance)
    ///
    /// Pointer equality optimization using virtual/null/class information.
    fn optimize_oois_ooisnot(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        expect_isnot: bool,
        instance: bool,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // rewrite.py _optimize_oois_ooisnot:
        //     arg0 = get_box_replacement(op.getarg(0))
        //     arg1 = get_box_replacement(op.getarg(1))
        //     info0 = getptrinfo(arg0)
        //     info1 = getptrinfo(arg1)
        let info0 = ctx.getptrinfo(&op.arg(0).get_box_replacement(false));
        let info1 = ctx.getptrinfo(&op.arg(1).get_box_replacement(false));

        let is_virtual0 = info0.as_ref().is_some_and(|i| i.is_virtual());
        let is_virtual1 = info1.as_ref().is_some_and(|i| i.is_virtual());

        // rewrite.py:530-535: virtual objects
        if is_virtual0 {
            let intres = if is_virtual1 {
                // rewrite.py: `intres = (info0 is info1) ^ expect_isnot`
                // — PtrInfo identity (info.py `same_info` is `self is
                // other` for non-Const infos), via getptrinfo_handle which
                // preserves the `_forwarded` cell identity.
                let same = match (
                    ctx.getptrinfo_handle(&op.arg(0).get_box_replacement(false)),
                    ctx.getptrinfo_handle(&op.arg(1).get_box_replacement(false)),
                ) {
                    (Some(h0), Some(h1)) => h0.same_info(&h1),
                    _ => false,
                };
                same ^ expect_isnot
            } else {
                expect_isnot
            };
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(intres as i64));
            return OptimizationResult::Remove;
        }
        if is_virtual1 {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(expect_isnot as i64));
            return OptimizationResult::Remove;
        }

        // rewrite.py:528-531: null checks — fall back to OpRef for downstream
        let arg0 = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        let arg1 = ctx.resolve_operand_operand(&op.arg(1)).to_opref();
        if info1.as_ref().is_some_and(|i| i.is_null()) {
            return self.optimize_nullness(op, op_rc, arg0, expect_isnot, ctx);
        }
        if info0.as_ref().is_some_and(|i| i.is_null()) {
            return self.optimize_nullness(op, op_rc, arg1, expect_isnot, ctx);
        }

        // rewrite.py: `elif arg0 is arg1:` — box identity
        // (resoperation.py `same_box` base = `self is other`).
        if ctx.box_is(arg0, arg1) {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(!expect_isnot as i64));
            return OptimizationResult::Remove;
        }

        // rewrite.py:535-553: instance comparison — different classes → not same
        if instance {
            let cls0 = info0
                .as_ref()
                .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
            let cls1 = info1
                .as_ref()
                .and_then(|i| i.get_known_class(ctx.cpu.as_ref()));
            if let (Some(c0), Some(c1)) = (cls0, cls1)
                && c0 != c1
            {
                let b = Operand::from_bound_op(op_rc);
                ctx.make_constant_box(&b, Value::Int(expect_isnot as i64));
                return OptimizationResult::Remove;
            }
        } else {
            // rewrite.py:550-553: non-instance array pointer comparison.
            // If both are ArrayPtrInfo with known-different length bounds,
            // they cannot be the same object.
            let lb0 = info0.clone().and_then(|mut i| i.getlenbound(None));
            let lb1 = info1.clone().and_then(|mut i| i.getlenbound(None));
            if let (Some(lb0), Some(lb1)) = (lb0, lb1)
                && lb0.known_ne(&lb1)
            {
                let b = Operand::from_bound_op(op_rc);
                ctx.make_constant_box(&b, Value::Int(expect_isnot as i64));
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    /// rewrite.py `_optimize_nullness(op, box, expect_nonnull)`:
    ///     info = self.getnullness(box)
    ///     if info == INFO_NONNULL: self.make_constant_int(op, expect_nonnull)
    ///     elif info == INFO_NULL: self.make_constant_int(op, not expect_nonnull)
    ///     else: return self.emit(op)
    fn optimize_nullness(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        arg: OpRef,
        expect_nonnull: bool,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match self.getnullness(arg, ctx) {
            Nullness::Nonnull => {
                let b = Operand::from_bound_op(op_rc);
                ctx.make_constant_box(&b, Value::Int(expect_nonnull as i64));
                OptimizationResult::Remove
            }
            Nullness::Null => {
                let b = Operand::from_bound_op(op_rc);
                ctx.make_constant_box(&b, Value::Int(!expect_nonnull as i64));
                OptimizationResult::Remove
            }
            Nullness::Unknown => OptimizationResult::PassOn,
        }
    }

    /// Constant fold INT_FORCE_GE_ZERO.
    fn optimize_int_force_ge_zero(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);

        if let Some(a) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(if a < 0 { 0 } else { a }));
            return OptimizationResult::Remove;
        }

        // force_ge_zero_pos: int_force_ge_zero(x) => x (if x known nonneg)
        if let Some(bound) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.peek_intbound_box(&b))
        {
            if bound.known_nonnegative() {
                let b_old = Operand::from_bound_op(op_rc);
                let b_arg = ctx.resolve_operand_operand(&arg0);
                ctx.make_equal_to(&b_old, &b_arg);
                return OptimizationResult::Remove;
            }
            // force_ge_zero_neg: int_force_ge_zero(x) => 0 (if x known negative)
            if bound.upper < 0 {
                let b = Operand::from_bound_op(op_rc);
                ctx.make_constant_box(&b, Value::Int(0));
                return OptimizationResult::Remove;
            }
        }

        OptimizationResult::PassOn
    }

    // ── Guards ──

    /// Optimize `GUARD_TRUE` through `OptRewrite.optimize_guard(op, CONST_1)`.
    /// If the condition is a known constant 0, the trace is impossible and must abort.
    ///
    /// `OptRewrite.optimize_guard` performs the contradiction check and emission
    /// at call time. Its `make_constant(box, CONST_1)` step
    ///   of the upstream `optimize_guard` is split into
    ///   `OptRewrite::propagate_postprocess` per RPython's
    ///   `have_postprocess` model.
    fn optimize_guard_true(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // The integer path in `OptRewrite.optimize_guard` checks
        // `intbound.is_constant()`,
        // which catches values narrowed to a single point by bounds analysis,
        // not just the constant pool.
        if let Some(val) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
        {
            if val != 0 {
                return OptimizationResult::Remove;
            }
            return raise_invalid_loop("GUARD_TRUE proven to always fail", op, ctx);
        }

        OptimizationResult::PassOn
    }

    /// Optimize `GUARD_FALSE` through `OptRewrite.optimize_guard(op, CONST_0)`.
    fn optimize_guard_false(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);

        // rewrite.py:165-168: box.type=='i' checks intbound.is_constant().
        if let Some(val) = ctx
            .resolve_operand_operand_opt(&arg0)
            .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
        {
            if val == 0 {
                return OptimizationResult::Remove;
            }
            return raise_invalid_loop("GUARD_FALSE proven to always fail", op, ctx);
        }

        OptimizationResult::PassOn
    }

    /// rewrite.py: optimize_GUARD_VALUE + replace_old_guard_with_guard_value
    ///
    /// If both args are constants and equal, the guard is redundant → remove.
    /// If arg0 is Ref-typed with a prior guard_nonnull/guard_class, replace
    /// that old guard with guard_value (rewrite.py:307-347).
    /// If the expected value is boolean, replace with GUARD_TRUE/FALSE.
    fn optimize_guard_value(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if op.num_args() < 2 {
            return OptimizationResult::PassOn;
        }
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        // rewrite.py optimize_guard(op, constbox) — base contradiction
        // check called from optimize_GUARD_VALUE at line 301. `arg1` is the
        // asserted Const. For box.type=='i' (rewrite.py:165-168) the check is
        // intbound.is_constant()/get_constant_int(), catching values narrowed
        // by bounds analysis, not just the constant pool. For 'r'
        // (rewrite.py) it is get_box_replacement(box).is_constant()/
        // same_constant. For 'f', rewrite.py:295-298 returns silently when
        // arg0 is constant without checking equality, so we mirror that by
        // removing on equality but never raising on mismatch.
        if let Some(expected_int) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            if let Some(actual_int) = ctx
                .resolve_operand_operand_opt(&arg0)
                .and_then(|b| ctx.get_constant_int_or_bound_box(&b))
            {
                if actual_int == expected_int {
                    return OptimizationResult::Remove;
                }
                return raise_invalid_loop("GUARD_VALUE proven to always fail", op, ctx);
            }
        } else if let (Some(actual), Some(expected)) = (
            ctx.resolve_operand_operand_opt(&arg0)
                .and_then(|b| ctx.get_constant_box(&b)),
            ctx.resolve_operand_operand_opt(&arg1)
                .and_then(|cb| cb.const_value()),
        ) {
            if actual == expected {
                return OptimizationResult::Remove;
            }
            match actual {
                Value::Int(_) | Value::Ref(_) => {
                    return raise_invalid_loop("GUARD_VALUE proven to always fail", op, ctx);
                }
                Value::Float(_) => {
                    return OptimizationResult::Remove;
                }
                Value::Void => {}
            }
        }

        // rewrite.py: optimize_GUARD_VALUE for Ref args.
        // getptrinfo synthesizes ConstPtrInfo for constant Refs, matching
        // `if info:` in RPython (which is True for ConstPtrInfo too).
        let obj_box = ctx.resolve_operand_operand_opt(&arg0);
        let obj_info = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b));
        if let Some(info) = obj_info {
            if info.is_virtual() {
                return raise_invalid_loop("promote of a virtual", op, ctx);
            }
            // rewrite.py: replace_old_guard_with_guard_value
            if let Some(old_guard) = obj_box
                .as_ref()
                .and_then(|b| ctx.get_last_guard(b))
                .cloned()
            {
                // rewrite.py:320: c_value = op.getarg(1) — generic Const.
                // Under typed seeding c_value can be Int OR Ref; the
                // previous gating on get_constant_int dropped the Ref
                // case entirely so prior GUARD_NONNULL/GUARD_CLASS were
                // never strengthened to GUARD_VALUE for Ref-typed args.
                if let Some(c_value) = ctx
                    .resolve_operand_operand_opt(&arg1)
                    .and_then(|b| b.const_value())
                {
                    // rewrite.py:321-323: c_value.nonnull(). ConstInt.nonnull
                    // == (value != 0); ConstPtr.nonnull == (gcref != null).
                    let c_nonnull = match c_value {
                        Value::Int(i) => i != 0,
                        Value::Ref(g) => !g.is_null(),
                        Value::Float(_) => true,
                        Value::Void => true,
                    };
                    if !c_nonnull {
                        return raise_invalid_loop(
                            "GUARD_VALUE(..., NULL) follows some other guard that it is not NULL",
                            op,
                            ctx,
                        );
                    }
                    // rewrite.py:324-332: previous_classbox = info.get_known_class(cpu)
                    // expected_classbox = cpu.cls_of_box(c_value)
                    // get_known_class on the c_value side dispatches through
                    // getptrinfo → ConstPtrInfo.get_known_class (info.py)
                    // which is exactly cls_of_box for constant pointers.
                    if let Some(prev_cls) = info.get_known_class(ctx.cpu.as_ref())
                        && let Some(arg1_box) = ctx.resolve_operand_operand_opt(&arg1)
                        && let Some(expected_cls) = ctx.get_known_class(&arg1_box)
                        && prev_cls != expected_cls
                    {
                        return raise_invalid_loop(
                            "GUARD_VALUE proven to always fail (class mismatch)",
                            op,
                            ctx,
                        );
                    }
                    // rewrite.py:333-334: can_replace_guards check.
                    if !ctx.can_replace_guards {
                        return OptimizationResult::PassOn;
                    }
                    // rewrite.py:335-347: replace old guard with GUARD_VALUE.
                    // last_guard_pos is a _newoperations index (info.py get_last_guard).
                    // rewrite.py:339-340: old descr must not be ResumeAtPositionDescr
                    // — RPython's fresh ResumeGuardDescr() at line 335 must
                    // not overwrite a RAPD marker.
                    if let Some(old_idx) = obj_box.as_ref().and_then(|b| ctx.last_guard_pos(b))
                        && !ctx.is_resume_at_position_guard(old_idx as i32)
                    {
                        // rewrite.py:345-354 + optimizer.py:713-718:
                        // RPython creates a fresh ResumeGuardDescr() (an
                        // empty descr OBJECT, not None) for the strengthened
                        // guard, then replace_guard / replace_guard_op copies
                        // the resume payload from the old guard descr into the
                        // new one. This path writes directly into
                        // new_operations (rather than deferring through
                        // replaces_guard), so perform the descr copy inline
                        // before replacing the op — identical to the
                        // GUARD_CLASS strengthening below. Passing None here
                        // would leave the strengthened guard with no descr →
                        // no rd_numb → exit_layout.storage None → resume panic
                        // when the promoted value actually changes at runtime.
                        let new_descr = crate::compile::make_resume_guard_descr_typed(
                            old_guard
                                .get_fail_arg_types()
                                .map(|t| t.to_vec())
                                .unwrap_or_default(),
                        );
                        let old_descr = old_guard
                            .getdescr()
                            .expect("strengthened GUARD_VALUE donor must carry a descr");
                        crate::compile::copy_all_attributes_from(&new_descr, &old_descr);
                        // rewrite.py:346-348 GuardResOp.copy_and_change:
                        // shallow copy with new opcode/args/descr; fail_args,
                        // fail_arg_types, rd_resume_position, rd_numb, rd_consts,
                        // rd_virtuals, rd_pendingfields are carried from
                        // old_guard automatically.
                        let replacement = old_guard.copy_and_change(
                            OpCode::GuardValue,
                            Some(&[old_guard.arg(0), arg1]),
                            Some(Some(new_descr)),
                        );
                        // rewrite.py:343: self.optimizer.replace_guard(op, info)
                        ctx.replace_new_operation(old_idx, OpRc::new(replacement));
                        // rewrite.py:345-346: info.reset_last_guard_pos()
                        if let Some(b) = obj_box.as_ref() {
                            ctx.with_ptr_info_mut(b, |info_mut| info_mut.reset_last_guard_pos());
                        }
                        // postprocess_GUARD_VALUE (rewrite.py): make_constant
                        // with the actual c_value (preserving Int vs Ref typing).
                        ctx.make_constant_arg(&arg0, c_value);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // rewrite.py postprocess_GUARD_VALUE `make_constant(box,
        // op.getarg(1))` runs in `propagate_postprocess` below, AFTER the
        // guard has been emitted with its argument resolved. Calling it
        // here (pre-emit) installs the Const forwarding before
        // `emit_operation` resolves the guard's own arg0, collapsing the
        // emitted guard to `guard_value(Const, Const)` — a no-op the
        // backend compiles to nothing, so promoted values were never
        // re-checked at runtime (#210 unguarded zombie loops).
        OptimizationResult::PassOn
    }

    /// rewrite.py optimize_GUARD_CLASS / postprocess_GUARD_CLASS.
    ///
    /// Shared by GuardClass and GuardNonnullClass — RPython
    /// `optimize_GUARD_NONNULL_CLASS` (rewrite.py) delegates to
    /// `optimize_GUARD_CLASS` after the null check, so both opcodes go
    /// through the same known-class / strengthening / postprocess logic.
    fn optimize_guard_class(
        &self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // rewrite.py:402: info = self.ensure_ptr_info_arg0(op) — installs
        // an InstancePtrInfo on arg0 when one is missing. Discard the
        // EnsuredPtrInfo borrow; downstream lookups re-acquire via
        // `getptrinfo` / `get_ptr_info` against the resolved OpRef.
        let _ = ctx.ensure_ptr_info_arg0(op);
        let obj = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        // rewrite.py:397-407: ensure_ptr_info_arg0 → info.py getptrinfo.
        // `getptrinfo(ConstPtr)` returns a synthesized ConstPtrInfo, so a
        // constant Ref arg0 is handled uniformly with virtual / instance
        // info: ConstPtrInfo.get_known_class(cpu) (info.py) reads
        // the typeptr at offset 0 via cls_of_box and compares against
        // expectedclassbox. Mismatch → proven-fail guard → InvalidLoop.
        let obj_info_for_class = ctx.getptrinfo(&op.arg(0).get_box_replacement(false));
        if let Some(known_class) =
            obj_info_for_class.and_then(|i| i.get_known_class(ctx.cpu.as_ref()))
            && op.num_args() >= 2
        {
            // RPython GuardClass / GuardNonnullClass class operands are
            // ConstInt vtable addresses (`expectedclassbox.getint()`).
            let expected = ctx.get_constant_int_box(&op.arg(1));
            if let Some(expected) = expected {
                if known_class == expected {
                    // `rewrite.py` `optimize_GUARD_CLASS`: known class matches — drop the
                    // guard. TRACE-level GUARD_CLASS is VOID
                    // (`resoperation.py` `'GUARD_CLASS/2d/n'`); there is
                    // no result box to forward.
                    return OptimizationResult::Remove;
                }
                // rewrite.py:404-407: known class mismatch is a
                // proven-fail guard — abort the trace.
                return raise_invalid_loop("GUARD_CLASS proven to always fail", op, ctx);
            }
        }
        // rewrite.py:408-427: guard strengthening.
        // If there was a previous GUARD_NONNULL on the same value,
        // replace it with GUARD_NONNULL_CLASS (combining both checks).
        // rewrite.py:409-410: skip replacement if old descr is a
        // ResumeAtPositionDescr — RPython's fresh ResumeGuardDescr() at
        // line 417 must not overwrite a RAPD marker (rewrite.py:421-422
        // "old descr must not be ResumeAtPositionDescr").
        if let Some(old_guard) = ctx.get_last_guard(&op.arg(0).get_box_replacement(false))
            && old_guard.opcode == OpCode::GuardNonnull
            && op.num_args() >= 2
            && ctx.can_replace_guards
        {
            // last_guard_pos is a _newoperations index.
            let old_guard_idx = ctx.last_guard_pos(&op.arg(0).get_box_replacement(false));
            if let Some(old_idx) = old_guard_idx
                && !ctx.is_resume_at_position_guard(old_idx as i32)
            {
                // rewrite.py:417-426 + optimizer.py:713-718:
                // RPython creates a fresh ResumeGuardDescr for the
                // strengthened guard, then replace_guard_op copies the
                // resume payload from the old guard descr into the new
                // one. This path writes directly into new_operations, so
                // perform the descr copy inline before replacing the op.
                let new_descr = crate::compile::make_resume_guard_descr_typed(
                    old_guard
                        .get_fail_arg_types()
                        .map(|t| t.to_vec())
                        .unwrap_or_default(),
                );
                let old_descr = old_guard
                    .getdescr()
                    .expect("strengthened GUARD_CLASS donor must carry a descr");
                crate::compile::copy_all_attributes_from(&new_descr, &old_descr);
                let combined = old_guard.copy_and_change(
                    OpCode::GuardNonnullClass,
                    Some(&[old_guard.arg(0), op.arg(1)]),
                    Some(Some(new_descr)),
                );
                ctx.replace_new_operation(old_idx, OpRc::new(combined));
                // rewrite.py postprocess_GUARD_CLASS parity
                // (invoked inline here because the replacement path
                // rewrites `new_operations[old_idx]` directly instead
                // of going through `emit_operation`, which would have
                // triggered the regular postprocess dispatch).
                //
                // The replacement happened because the old guard is
                // not a ResumeAtPositionDescr (guarded at
                // `is_resume_at_position_guard(old_idx)` above), so
                // rewrite.py:434-435 `update_last_guard = not
                // old_guard_op or isinstance(descr, RAPD)` evaluates
                // to False — pass `update_last_guard=false` so that
                // make_constant_class preserves the strengthened
                // guard's position in last_guard_pos (optimizer.py
                // parity) rather than snapping it to the tail of
                // new_operations.
                if let Some(class_val) = ctx.get_constant_int_box(&op.arg(1))
                    && let Some(b) = ctx.get_box_replacement_operand_opt(obj)
                {
                    crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                        ctx, &b, class_val, /* update_last_guard = */ false,
                    );
                }
                return OptimizationResult::Remove;
            }
        }
        // rewrite.py postprocess_GUARD_CLASS: runs AFTER emit.
        // Register deferred postprocess — executed by emit_operation
        // after the guard is added to new_operations.  Upstream
        // `postprocess_GUARD_CLASS` runs unconditionally (no
        // virtual-skip): `make_constant_class` already preserves
        // existing `InstancePtrInfo` whether or not `is_virtual=True`
        // (`optimizer.py`), so the Rust port skips the local
        // `is_virtual` guard and lets `Optimizer::make_constant_class`
        // dispatch on the live `Instance` / `Virtual` arm.
        if op.num_args() >= 2
            && let Some(class_val) = ctx.get_constant_int_box(&op.arg(1))
        {
            ctx.pending_guard_class_postprocess =
                Some(crate::optimizeopt::PendingGuardClassPostprocess { obj, class_val });
        }
        OptimizationResult::PassOn
    }

    // ── SAME_AS identity ──

    /// optimizer.py `getnullness(op)` wrapper. Delegates to
    /// `OptContext::getnullness`, which implements the upstream
    /// `op.type == 'r' or is_raw_ptr(op)` dispatch line-by-line, then
    /// converts the upstream `INFO_NULL` / `INFO_NONNULL` /
    /// `INFO_UNKNOWN` integer return into the local `Nullness` enum.
    fn getnullness(&self, opref: OpRef, ctx: &mut OptContext) -> Nullness {
        // optimizer.py `getnullness` has no missing-Box branch —
        // every `op` has a backing `AbstractValue`, because
        // `resoperation.py` derives `AbstractResOpOrInputArg` from it and
        // `AbstractResOp` from that. `get_box_replacement_operand_opt`
        // resolves the opref to its bound host; the read-only `getnullness`
        // below never writes, so an unresolvable opref (OpRef::NONE
        // sentinel, no upstream equivalent) maps to `INFO_UNKNOWN`.
        let info = match ctx.get_box_replacement_operand_opt(opref) {
            Some(b) => ctx.getnullness(&b),
            None => crate::optimizeopt::INFO_UNKNOWN,
        };
        Self::nullness_from_info(info)
    }

    /// Convert an `info.py` INFO_NULL/INFO_NONNULL/INFO_UNKNOWN return
    /// into the local `Nullness` enum used by the rewrite pass.
    fn nullness_from_info(value: i8) -> Nullness {
        if value == crate::optimizeopt::INFO_NULL {
            Nullness::Null
        } else if value == crate::optimizeopt::INFO_NONNULL {
            Nullness::Nonnull
        } else {
            Nullness::Unknown
        }
    }

    /// Check if an OpRef is Ref-typed.
    /// optimizer.py:128: op.type == 'r'
    ///
    /// Routes through the canonical `OptContext::opref_type` accessor
    /// (constant → value_types → producer op result_type) and falls back
    /// to PtrInfo presence — a Ref-only side channel populated for input
    /// args that do not appear in `new_operations`.
    fn is_ref_typed(&self, opref: OpRef, ctx: &OptContext) -> bool {
        if ctx.opref_type(opref) == Some(majit_ir::Type::Ref) {
            return true;
        }
        // has_ptr_info takes &Operand per info.py getptrinfo.
        ctx.get_box_replacement_operand_opt(opref)
            .as_ref()
            .is_some_and(|b| ctx.has_ptr_info(b))
    }

    /// rewrite.py: _optimize_CALL_INT_UDIV
    /// x / 1 → x
    fn optimize_call_int_udiv(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> bool {
        if op.num_args() < 3 {
            return false;
        }
        let arg2 = op.arg(2);
        if let Some(1) = ctx
            .resolve_operand_operand_opt(&arg2)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            let b_old = Operand::from_bound_op(op_rc);
            let b_arg = ctx.resolve_operand_operand(&op.arg(1));
            ctx.make_equal_to(&b_old, &b_arg);
            ctx.last_op_removed = true;
            return true;
        }
        false
    }

    /// rewrite.py: _optimize_CALL_INT_PY_MOD
    fn optimize_call_int_py_mod(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 3 {
            return None;
        }
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);
        let b1 = {
            let b = ctx.resolve_operand_operand(&arg1);
            ctx.getintbound_handle(&b).borrow().clone()
        };
        let b2 = {
            let b = ctx.resolve_operand_operand(&arg2);
            ctx.getintbound_handle(&b).borrow().clone()
        };

        // rewrite.py:774-777: b1.known_eq_const(0) → 0
        if b1.known_eq_const(0) {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:780-781: if not b2.is_constant(): return False
        if !b2.is_constant() {
            return None;
        }
        let val = b2.get_constant_int();
        // rewrite.py:783-784
        if val <= 0 {
            return None;
        }
        // rewrite.py:785-788: x % 1 → 0
        if val == 1 {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:789-796: x % power_of_two → x & (power_of_two - 1)
        // Python's modulo: valid even for negative x.
        // RPython: replace_op_with + send_extra_operation (routes through passes).
        if val & (val - 1) == 0 {
            let arg_mask = Operand::const_(Const::Int(val - 1));
            let mut and_op = Op::new(OpCode::IntAnd, &[arg1, arg_mask.clone()]);
            and_op.pos().set(op.pos().get());
            ctx.emit_extra(ctx.current_pass_idx, and_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:797-805: intdiv.modulo_operations fallback, which emits
        // UINT_MUL_HIGH unconditionally.  The residual call always goes away.
        let known_nonneg = b1.known_nonnegative();
        let b_res = send_extra_operations(ctx, intdiv::modulo_operations(&arg1, val, known_nonneg));
        let b_old = Operand::from_bound_op(op_rc);
        ctx.make_equal_to(&b_old, &b_res);
        ctx.last_op_removed = true;
        Some(OptimizationResult::Remove)
    }

    /// rewrite.py: _optimize_CALL_INT_PY_DIV
    fn optimize_call_int_py_div(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 3 {
            return None;
        }
        let arg1 = op.arg(1);
        let arg2 = op.arg(2);
        let b1 = {
            let b = ctx.resolve_operand_operand(&arg1);
            ctx.getintbound_handle(&b).borrow().clone()
        };
        let b2 = {
            let b = ctx.resolve_operand_operand(&arg2);
            ctx.getintbound_handle(&b).borrow().clone()
        };

        // rewrite.py:726-729: b1.known_eq_const(0) → 0
        if b1.known_eq_const(0) {
            let b = Operand::from_bound_op(op_rc);
            ctx.make_constant_box(&b, Value::Int(0));
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:730-741: non-constant divisor (shift optimization)
        if !b2.is_constant() {
            // rewrite.py:731-740: x // (1 << y) → x >> y
            // when 0 <= y < LONG_BIT - 1
            if let Some(shift_op) = ctx
                .resolve_operand_operand_opt(&arg2)
                .and_then(|pb| ctx.get_producing_op(&pb))
                && shift_op.opcode == OpCode::IntLshift
                && shift_op.num_args() >= 2
                && shift_op.arg(0).get_box_replacement(false).const_int() == Some(1)
            {
                let shiftvar = ctx.resolve_operand_operand(&shift_op.arg(1));
                let shiftbound = ctx.getintbound_handle(&shiftvar).borrow().clone();
                if shiftbound.known_nonnegative() && shiftbound.known_lt_const(63) {
                    let arg_shift = shiftvar.clone();
                    let mut rshift_op = Op::new(OpCode::IntRshift, &[arg1, arg_shift.clone()]);
                    rshift_op.pos().set(op.pos().get());
                    ctx.emit_extra(ctx.current_pass_idx, rshift_op);
                    ctx.last_op_removed = true;
                    return Some(OptimizationResult::Remove);
                }
            }
            return None;
        }
        let val = b2.get_constant_int();
        // rewrite.py:743-749: x // -1 → -x (if x > MININT)
        if val == -1 && b1.known_gt_const(i64::MIN) {
            let mut neg_op = Op::new(OpCode::IntNeg, &[arg1]);
            neg_op.pos().set(op.pos().get());
            ctx.emit_extra(ctx.current_pass_idx, neg_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:750-751
        if val <= 0 {
            return None;
        }
        // rewrite.py:752-755: x // 1 → x
        if val == 1 {
            let b_old = Operand::from_bound_op(op_rc);
            let b_arg = ctx.resolve_operand_operand(&arg1);
            ctx.make_equal_to(&b_old, &b_arg);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:756-757: x // power_of_two → x >> shift
        if val & (val - 1) == 0 {
            let shift = val.trailing_zeros() as i64;
            let arg_shift = Operand::const_(Const::Int(shift));
            let mut rshift_op = Op::new(OpCode::IntRshift, &[arg1, arg_shift.clone()]);
            rshift_op.pos().set(op.pos().get());
            ctx.emit_extra(ctx.current_pass_idx, rshift_op);
            ctx.last_op_removed = true;
            return Some(OptimizationResult::Remove);
        }
        // rewrite.py:758-766: intdiv.division_operations fallback, which emits
        // UINT_MUL_HIGH unconditionally.  The residual call always goes away.
        let known_nonneg = b1.known_nonnegative();
        let b_res =
            send_extra_operations(ctx, intdiv::division_operations(&arg1, val, known_nonneg));
        let b_old = Operand::from_bound_op(op_rc);
        ctx.make_equal_to(&b_old, &b_res);
        ctx.last_op_removed = true;
        Some(OptimizationResult::Remove)
    }

    /// rewrite.py: _optimize_call_arrayop
    ///
    /// Element-by-element unrolling for small constant-length array
    /// copy/move operations. Handles both virtual and non-virtual arrays.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn optimize_call_arrayop(
        &mut self,
        op: &Op,
        source_box: OpRef,
        dest_box: OpRef,
        source_start_box: OpRef,
        dest_start_box: OpRef,
        length_box: OpRef,
        ctx: &mut OptContext,
    ) -> bool {
        // rewrite.py:601-602: length = self.get_constant_box(length_box)
        let length_int = match ctx
            .get_box_replacement_operand_opt(length_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(l) => l,
            None => return false,
        };
        // rewrite.py:605-606: 0-length → remove
        if length_int == 0 {
            return true;
        }

        // One chain walk each; the position view falls back to the source.
        let source_b = ctx.get_box_replacement_operand_opt(source_box);
        let source_box = source_b.as_ref().map_or(source_box, |b| b.to_opref());
        let dest_b = ctx.get_box_replacement_operand_opt(dest_box);
        let dest_box = dest_b.as_ref().map_or(dest_box, |b| b.to_opref());
        let source_is_virtual = source_b.as_ref().is_some_and(|b| ctx.is_virtual(b));
        let dest_is_virtual = dest_b.as_ref().is_some_and(|b| ctx.is_virtual(b));

        // rewrite.py:610-611: constant start indices required
        let source_start = match ctx
            .get_box_replacement_operand_opt(source_start_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(s) => s,
            None => return false,
        };
        let dest_start = match ctx
            .get_box_replacement_operand_opt(dest_start_box)
            .and_then(|b| ctx.get_constant_int_box(&b))
        {
            Some(d) => d,
            None => return false,
        };

        // rewrite.py:613-617: both start constant, at least one virtual or length <= 8
        if (!source_is_virtual || !dest_is_virtual) && length_int > 8 {
            return false;
        }

        // rewrite.py:612,617: extrainfo.single_write_descr_array sanity check
        let call_descr = match op.getdescr() {
            Some(d) => d,
            None => return false,
        };
        let cd = match call_descr.as_call_descr() {
            Some(cd) => cd,
            None => return false,
        };
        let ei = cd.get_extra_info();
        // rewrite.py:617: extrainfo.single_write_descr_array is not None
        // effectinfo.py:201-206: set when exactly one write array descriptor.
        let arraydescr = match &ei.single_write_descr_array {
            Some(d) => d.clone(),
            None => {
                // Fallback: check bitstring — must have exactly one array write.
                let count: u32 = ei
                    .write_descrs_arrays
                    .as_ref()
                    .map(|w| w.iter().map(|b| b.count_ones()).sum())
                    .unwrap_or(0);
                if count != 1 {
                    return false;
                }
                // No actual DescrRef available — cannot emit typed ops.
                return false;
            }
        };

        // rewrite.py:621-635: arraydescr.is_array_of_structs()
        if arraydescr
            .as_array_descr()
            .is_some_and(|ad| ad.is_array_of_structs())
        {
            // rewrite.py:624-627: only if both virtual, not memmove
            if !(source_is_virtual && dest_is_virtual && source_box != dest_box) {
                return false;
            }
            // rewrite.py:628-629: all_fdescrs = arraydescr.get_all_fielddescrs()
            // → all_interiorfielddescrs in descr.py.
            let all_fdescr_indices: Vec<u32> = arraydescr
                .as_array_descr()
                .and_then(|ad| ad.get_all_interiorfielddescrs())
                .map(|fds| fds.iter().map(|d| d.index()).collect())
                .or_else(|| {
                    // Fallback: get from virtual's metadata
                    ctx.get_box_replacement_operand_opt(source_box)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .and_then(|info| match info {
                            crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(v) => {
                                if v.fielddescrs.is_empty() {
                                    None
                                } else {
                                    Some(v.fielddescrs.iter().map(|d| d.index()).collect())
                                }
                            }
                            _ => None,
                        })
                })
                .unwrap_or_default();
            if all_fdescr_indices.is_empty() {
                return false;
            }
            // rewrite.py:631-634: copy interior fields element by element
            for index in 0..length_int {
                for &fdescr_idx in &all_fdescr_indices {
                    let val = ctx
                        .get_box_replacement_operand_opt(source_box)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .and_then(|info| {
                            info.getinteriorfield_virtual(
                                (index + source_start) as usize,
                                fdescr_idx,
                            )
                        });
                    if let Some(val) = val {
                        let idx = (index + dest_start) as usize;
                        let val = ctx.materialize_operand_at(val);
                        if let Some(b) = ctx.get_box_replacement_operand_opt(dest_box) {
                            ctx.with_ptr_info_mut(&b, |info| {
                                info.setinteriorfield_virtual(idx, fdescr_idx, val.clone());
                            });
                        }
                    }
                }
            }
            return true;
        }

        // rewrite.py:636-643: iteration direction
        let mut index_current: i64 = 0;
        let mut index_delta: i64 = 1;
        let mut index_stop: i64 = length_int;
        if source_box == dest_box && source_start < dest_start {
            // ARRAYMOVE with overlapping regions: iterate in reverse
            index_current = index_stop - 1;
            index_delta = -1;
            index_stop = -1;
        }

        // rewrite.py:646-670: element-by-element copy
        // RPython routes synthesized ops through send_extra_operation()
        // so they pass through downstream optimization passes.
        // We use ctx.emit_extra(current_pass_idx, op) for the same effect.
        let pass_idx = ctx.current_pass_idx;
        while index_current != index_stop {
            let index = index_current;
            index_current += index_delta;
            debug_assert!(index >= 0);

            // Read source element
            let val = if source_is_virtual {
                // rewrite.py:650-651: source_info.getitem(arraydescr, index + source_start)
                ctx.get_box_replacement_operand_opt(source_box)
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b))
                    .and_then(|info| info.getitem((index + source_start) as usize))
                    .and_then(|e| e.as_opref())
                    .map(|v| ctx.materialize_operand_at(v))
            } else {
                // rewrite.py:653: opnum = OpHelpers.getarrayitem_for_descr(arraydescr)
                // Select I/R/F opcode based on item type.
                let item_type = arraydescr
                    .as_array_descr()
                    .map(|ad| ad.item_type())
                    .unwrap_or(majit_ir::Type::Int);
                let opcode = OpCode::getarrayitem_for_type(item_type);
                let arg_source = ctx.materialize_operand_at(source_box);
                let arg_idx = Operand::const_(Const::Int(index + source_start));
                let mut getop = Op::new(opcode, &[arg_source.clone(), arg_idx.clone()]);
                getop.setdescr(arraydescr.clone());
                let getop = OpRc::new(getop);
                ctx.emit_extra_rc(pass_idx, getop.clone());
                Some(Operand::from_bound_op(&getop))
            };

            let val = match val {
                Some(v) => v,
                None => continue, // rewrite.py:660-661: if val is None: continue
            };

            // Write to destination
            if dest_is_virtual {
                // rewrite.py:662-665: dest_info.setitem(...)
                let idx = (index + dest_start) as usize;
                if let Some(b) = ctx.get_box_replacement_operand_opt(dest_box) {
                    ctx.with_ptr_info_mut(&b, |info| info.setitem(idx, val.clone()));
                }
            } else {
                // rewrite.py:666-670: emit SETARRAYITEM_GC
                let arg_dest = ctx.materialize_operand_at(dest_box);
                let arg_idx = Operand::const_(Const::Int(index + dest_start));
                let arg_val = val;
                let mut setop = Op::new(
                    OpCode::SetarrayitemGc,
                    &[arg_dest.clone(), arg_idx.clone(), arg_val.clone()],
                );
                setop.setdescr(arraydescr.clone());
                ctx.emit_extra(pass_idx, setop);
            }
        }
        true
    }

    fn optimize_same_as(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() == 0 {
            return OptimizationResult::PassOn;
        }
        let arg0 = op.arg(0);
        let b_old = Operand::from_bound_op(op_rc);
        let b_arg = ctx.resolve_operand_operand(&arg0);
        ctx.make_equal_to(&b_old, &b_arg);
        OptimizationResult::Remove
    }

    // ── Boolean inverse/reflex rewrites ──

    /// `Optimization.get_pure_result` for a two-argument synthetic op.
    ///
    /// Builds the op `OptRewrite.try_boolinvers` / `find_rewritable_bool`
    /// would hand to `get_pure_result`, then reads the ring `OptPure`
    /// publishes. The ring is bounded (`pureop_historylength`); a miss
    /// after that many pure ops is the same miss upstream takes. The
    /// ring is cleared in `OptPure.setup` at the start of each optimize
    /// run — `OptPure.flush` does not clear it, and no label op does
    /// either — so this pass does not clear it again.
    fn get_pure_result(
        &self,
        opcode: OpCode,
        args: [Operand; 2],
        ctx: &mut OptContext,
    ) -> Option<Operand> {
        let synthetic = Op::new(opcode, &args);
        ctx.get_pure_result(&synthetic)
    }

    /// `OptRewrite.try_boolinvers`: look up the synthetic inverse and,
    /// when its intbound is exactly 0 or 1, fold this op to the other bit.
    fn try_boolinvers(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        top: &Op,
        ctx: &mut OptContext,
    ) -> bool {
        let Some(old) = self.get_pure_result(top.opcode, [top.arg(0), top.arg(1)], ctx) else {
            return false;
        };
        let resolved = ctx.resolve_operand_operand(&old);
        let Some(bound) = ctx.peek_intbound_box(&resolved) else {
            return false;
        };
        let value = if bound.known_eq_const(1) {
            0
        } else if bound.known_eq_const(0) {
            1
        } else {
            return false;
        };
        let b = Operand::from_bound_op(op_rc);
        ctx.make_constant_box(&b, Value::Int(value));
        true
    }

    /// `OptRewrite.find_rewritable_bool`, three phases:
    /// 1. `boolinverse` on the same args, via `try_boolinvers`
    /// 2. `boolreflex` on the swapped args, aliased with `make_equal_to`
    /// 3. `boolreflex`'s `boolinverse` on the swapped args, via `try_boolinvers`
    fn find_rewritable_bool(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> Option<OptimizationResult> {
        if op.num_args() < 2 {
            return None;
        }
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);

        if let Some(inverse) = op.opcode.bool_inverse() {
            let top = Op::new(inverse, &[arg0.clone(), arg1.clone()]);
            if self.try_boolinvers(op, op_rc, &top, ctx) {
                return Some(OptimizationResult::Remove);
            }
        }

        let Some(reflex) = op.opcode.bool_reflex() else {
            return None;
        };
        let top = Op::new(reflex, &[arg1.clone(), arg0.clone()]);
        if let Some(old) = self.get_pure_result(top.opcode, [top.arg(0), top.arg(1)], ctx) {
            let b_old = Operand::from_bound_op(op_rc);
            let b_cached = ctx.resolve_operand_operand(&old);
            ctx.make_equal_to(&b_old, &b_cached);
            return Some(OptimizationResult::Remove);
        }
        if let Some(reflex_inverse) = reflex.bool_inverse() {
            let top = Op::new(reflex_inverse, &[arg1, arg0]);
            if self.try_boolinvers(op, op_rc, &top, ctx) {
                return Some(OptimizationResult::Remove);
            }
        }
        None
    }

    // ── Float algebraic simplifications ──
    // rewrite.py:103-161 — only FLOAT_MUL, FLOAT_TRUEDIV, FLOAT_NEG, FLOAT_ABS.
    // Constant folding for all float ops is handled by execute_nonspec_const.

    /// rewrite.py optimize_FLOAT_MUL
    fn optimize_float_mul(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        // rewrite.py:109: for lhs, rhs in [(arg1, arg2), (arg2, arg1)]:
        for (lhs, rhs) in [(&arg0, &arg1), (&arg1, &arg0)] {
            if let Some(v) = ctx
                .resolve_operand_operand_opt(lhs)
                .and_then(|b| b.const_value())
                .and_then(|v| match v {
                    Value::Float(f) => Some(f),
                    _ => None,
                })
            {
                if v == 1.0 {
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_v2 = ctx.resolve_operand_operand(rhs);
                    ctx.make_equal_to(&b_old, &b_v2);
                    return OptimizationResult::Remove;
                }
                if v == -1.0 {
                    let mut neg = Op::new(OpCode::FloatNeg, std::slice::from_ref(rhs));
                    neg.pos().set(op.pos().get());
                    return OptimizationResult::Replace(neg);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py optimize_FLOAT_TRUEDIV
    fn optimize_float_truediv(&self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let arg0 = op.arg(0);
        let arg1 = op.arg(1);
        if let Some(divisor) = ctx
            .resolve_operand_operand_opt(&arg1)
            .and_then(|b| b.const_value())
            .and_then(|v| match v {
                Value::Float(f) => Some(f),
                _ => None,
            })
        {
            // rewrite.py:135-141: frexp check that divisor AND reciprocal
            // are both exact powers of 2. Bit-level equivalent: mantissa
            // bits are all zero and exponent is normal (not zero/subnormal/inf/nan).
            if Self::is_exact_power_of_two(divisor) {
                let reciprocal = 1.0 / divisor;
                if Self::is_exact_power_of_two(reciprocal) {
                    let arg_recip = Operand::const_(Const::Float(reciprocal));
                    let mut new_op = Op::new(OpCode::FloatMul, &[arg0, arg_recip.clone()]);
                    new_op.pos().set(op.pos().get());
                    return OptimizationResult::Emit(new_op);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py: `math.frexp(divisor)[0]` == ±0.5 iff exact power of 2.
    fn is_exact_power_of_two(v: f64) -> bool {
        let bits = v.to_bits();
        let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;
        let exponent = ((bits >> 52) & 0x7FF) as u32;
        mantissa == 0 && exponent > 0 && exponent < 0x7FF
    }

    /// rewrite.py optimize_FLOAT_NEG
    fn optimize_float_neg(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let v = ctx
            .resolve_operand_operand_opt(&op.arg(0))
            .or_else(|| Some(ctx.resolve_operand_operand(&op.arg(0))));
        if let Some(arg_op) = v.and_then(|pb| ctx.get_producing_op(&pb))
            && arg_op.opcode == OpCode::FloatNeg
        {
            let b_old = Operand::from_bound_op(op_rc);
            let b_inner = ctx.resolve_operand_operand(&arg_op.arg(0));
            ctx.make_equal_to(&b_old, &b_inner);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    /// rewrite.py optimize_FLOAT_ABS
    fn optimize_float_abs(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let v = ctx.resolve_operand_operand_opt(&op.arg(0));
        if let Some(v) = v
            && let Some(arg_op) = ctx.get_producing_op(&v)
            && arg_op.opcode == OpCode::FloatAbs
        {
            let b_old = Operand::from_bound_op(op_rc);
            ctx.make_equal_to(&b_old, &v);
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }
}

impl Optimization for OptRewrite {
    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // Track last_op_removed for GuardNoException optimization.
        // Reset for non-guard ops (guards don't count as "the last op").
        if !op.opcode.is_guard() {
            ctx.last_op_removed = false;
        }

        // Try boolean inverse/reflex rewrites for comparisons
        if (op.opcode.bool_inverse().is_some() || op.opcode.bool_reflex().is_some())
            && let Some(result) = self.find_rewritable_bool(op, op_rc, ctx)
        {
            return result;
        }

        match op.opcode {
            // Integer arithmetic rewrites (the autogenintrules.py ruleset)
            // live in OptIntBounds, as upstream; rewrite.py has no
            // optimize_INT_ADD/SUB/MUL/AND/OR/XOR/shift/NEG/INVERT methods.
            // IntFloorDiv / IntMod are pyre opcodes (upstream lowers // and %
            // to residual calls, see optimize_call_int_py_div / _py_mod), so
            // their strength-reduction rules stay here with the opcode.
            OpCode::IntFloorDiv => self.optimize_int_floor_div(op, op_rc, ctx),
            OpCode::IntMod => self.optimize_int_mod(op, op_rc, ctx),

            // Tagged-int box/unbox round-trip fold. The autogen IntRshift
            // rules live in OptIntBounds (autogenintrules.rs); this pyre-only
            // tag fold is gated on the active GC's taggedpointers config and
            // is unreachable with it off (nothing emits the matched chain).
            OpCode::IntRshift => {
                if majit_gc::taggedpointers_enabled()
                    && let Some(result) = self.optimize_int_rshift_tag_unbox(op, op_rc, ctx)
                {
                    return result;
                }
                OptimizationResult::PassOn
            }

            OpCode::IntIsZero => self.optimize_int_is_zero(op, op_rc, ctx),
            OpCode::IntIsTrue => self.optimize_int_is_true(op, op_rc, ctx),
            OpCode::IntForceGeZero => self.optimize_int_force_ge_zero(op, op_rc, ctx),

            // ── Guards ──
            OpCode::GuardTrue => self.optimize_guard_true(op, ctx),
            OpCode::GuardFalse => self.optimize_guard_false(op, ctx),
            OpCode::GuardValue => self.optimize_guard_value(op, ctx),
            // RPython rewrite.py guard optimizations:
            // If the guarded condition is already known to be true (constant),
            // the guard can be removed entirely.
            OpCode::GuardNonnull => {
                // rewrite.py optimize_GUARD_NONNULL
                //     opinfo = getptrinfo(op.getarg(0))
                //     if opinfo is not None:
                //         if opinfo.is_nonnull(): return
                //         elif opinfo.is_null(): raise InvalidLoop(...)
                //     return self.emit(op)
                let obj_box = op.arg(0).get_box_replacement(false);
                let obj = obj_box.to_opref();
                if let Some(info) = ctx.getptrinfo(&obj_box) {
                    if info.is_nonnull() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_null() {
                        return raise_invalid_loop("GUARD_NONNULL proven to always fail", op, ctx);
                    }
                }
                // rewrite.py postprocess_GUARD_NONNULL:
                // make_nonnull runs immediately; mark_last_guard deferred
                // until emit adds the guard to new_operations.
                let has_info = ctx.has_ptr_info(&obj_box);
                if !has_info {
                    ctx.set_ptr_info(&obj_box, crate::optimizeopt::info::PtrInfo::nonnull());
                }
                // rewrite.py:282: mark_last_guard deferred to emit_operation
                ctx.pending_mark_last_guard = Some(obj);
                OptimizationResult::PassOn
            }
            OpCode::GuardIsnull => {
                // rewrite.py optimize_GUARD_ISNULL
                //     info = getptrinfo(op.getarg(0))
                //     if info is not None:
                //         if info.is_null(): return
                //         elif info.is_nonnull(): raise InvalidLoop(...)
                //     return self.emit(op)
                let obj_box = ctx.resolve_operand_operand_opt(&op.arg(0));
                if let Some(info) = obj_box.as_ref().and_then(|b| ctx.getptrinfo(b)) {
                    if info.is_null() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_nonnull() {
                        return raise_invalid_loop("GUARD_ISNULL proven to always fail", op, ctx);
                    }
                }
                // rewrite.py `postprocess_GUARD_ISNULL` runs its
                // `make_constant(op.getarg(0), CONST_NULL)` in
                // `propagate_postprocess` below, for the same reason
                // postprocess_GUARD_VALUE does: pre-emit it installs the Const
                // forwarding before `emit_operation` resolves the guard's own
                // arg0, and the guard reaches the backend as
                // `guard_isnull(ConstPtr(NULL))` — a tautology that never fails,
                // so the null the trace was specialized on was never re-checked.
                OptimizationResult::PassOn
            }
            OpCode::GuardClass => self.optimize_guard_class(op, op_rc, ctx),
            OpCode::GuardNonnullClass => {
                // rewrite.py optimize_GUARD_NONNULL_CLASS:
                //     info = getptrinfo(op.getarg(0))
                //     if info and info.is_null():
                //         raise InvalidLoop(...)
                //     return self.optimize_GUARD_CLASS(op)
                if let Some(info) = ctx.getptrinfo(&op.arg(0).get_box_replacement(false))
                    && info.is_null()
                {
                    return raise_invalid_loop(
                        "GUARD_NONNULL_CLASS proven to always fail",
                        op,
                        ctx,
                    );
                }
                self.optimize_guard_class(op, op_rc, ctx)
            }
            OpCode::GuardIsObject => {
                // rewrite.py optimize_GUARD_IS_OBJECT.
                let arg0 = op.arg(0).get_box_replacement(false);
                if let Some(info) = ctx.getptrinfo(&arg0) {
                    if info.is_constant() {
                        let Some(constant) = ctx.get_constant_box(&arg0) else {
                            return OptimizationResult::PassOn;
                        };
                        let gcref = constant.as_ref();
                        if gcref.is_null() {
                            return raise_invalid_loop("A GUARD_IS_OBJECT(NULL) found", op, ctx);
                        }
                        if majit_gc::check_is_object(gcref) {
                            return OptimizationResult::Remove;
                        }
                        return raise_invalid_loop(
                            "A GUARD_IS_OBJECT(not-an-object) found",
                            op,
                            ctx,
                        );
                    }
                    // An info that is about an object has already established
                    // what this guard tests; one that is precise and is not
                    // about an object never can.
                    if info.is_about_object() {
                        return OptimizationResult::Remove;
                    }
                    if info.is_precise() {
                        return raise_invalid_loop(
                            "GUARD_IS_OBJECT proven to always fail",
                            op,
                            ctx,
                        );
                    }
                }
                OptimizationResult::PassOn
            }
            OpCode::GuardGcType => {
                // rewrite.py optimize_GUARD_GC_TYPE.
                let arg0 = op.arg(0).get_box_replacement(false);
                let Some(expected) = ctx.get_constant_int_box(&op.arg(1)) else {
                    return OptimizationResult::PassOn;
                };
                if let Some(info) = ctx.getptrinfo(&arg0) {
                    if info.is_constant() {
                        let Some(constant) = ctx.get_constant_box(&arg0) else {
                            return OptimizationResult::PassOn;
                        };
                        let Some(type_id) = majit_gc::get_actual_typeid(constant.as_ref()) else {
                            return OptimizationResult::PassOn;
                        };
                        if i64::from(type_id) != expected {
                            return raise_invalid_loop(
                                "wrong GC type ID found on a constant",
                                op,
                                ctx,
                            );
                        }
                        return OptimizationResult::Remove;
                    }
                    // The descr names the layout the value is already known to
                    // carry, so the tid it resolves to settles the guard either
                    // way.  Take the resolved tid, never the stamped one: a
                    // deserialized descr carries only its structural cache key,
                    // and reading the 0 that stands for "not yet resolved" as a
                    // tid would report a type conflict against every real one.
                    if let Some(type_id) = info.get_descr().and_then(descr_gc_type_id) {
                        if i64::from(type_id) != expected {
                            return raise_invalid_loop("wrong GC types passed around!", op, ctx);
                        }
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }
            OpCode::GuardSubclass => {
                // rewrite.py optimize_GUARD_SUBCLASS.
                let arg0 = op.arg(0).get_box_replacement(false);
                let Some(expected) = ctx.get_constant_int_box(&op.arg(1)) else {
                    return OptimizationResult::PassOn;
                };
                let Some(info) = ctx.getptrinfo(&arg0) else {
                    return OptimizationResult::PassOn;
                };
                // A constant's class is read off the value itself; an info that
                // is about an object may name the class exactly.  Both settle
                // the guard, so they share the verdict below.
                let known_class = if info.is_constant() {
                    ctx.cls_of_box(&arg0)
                } else if info.is_about_object() {
                    info.get_known_class(ctx.cpu.as_ref())
                } else {
                    None
                };
                if let Some(known_class) = known_class {
                    let msg = if info.is_constant() {
                        "GUARD_SUBCLASS(const) proven to always fail"
                    } else {
                        "GUARD_SUBCLASS(known_class) proven to always fail"
                    };
                    return match check_subclass(known_class, expected) {
                        Some(true) => OptimizationResult::Remove,
                        Some(false) => raise_invalid_loop(msg, op, ctx),
                        None => OptimizationResult::PassOn,
                    };
                }
                // Without an exact class the descr still bounds the value: it is
                // that class or a subclass of it.  The guard always passes when
                // the whole bound is inside the tested class, and can only be
                // decided at runtime when the tested class is inside the bound.
                if info.is_about_object()
                    && let Some(base) = info
                        .get_descr()
                        .and_then(|descr| descr.as_size_descr())
                        .map(|descr| descr.vtable() as i64)
                {
                    return match check_subclass(base, expected) {
                        Some(true) => OptimizationResult::Remove,
                        Some(false) => match check_subclass(expected, base) {
                            Some(false) => raise_invalid_loop(
                                "GUARD_SUBCLASS(base_class) proven to always fail",
                                op,
                                ctx,
                            ),
                            _ => OptimizationResult::PassOn,
                        },
                        None => OptimizationResult::PassOn,
                    };
                }
                OptimizationResult::PassOn
            }

            // ── Float arithmetic ──
            OpCode::FloatMul => self.optimize_float_mul(op, op_rc, ctx),
            OpCode::FloatTrueDiv => self.optimize_float_truediv(op, ctx),
            OpCode::FloatNeg => self.optimize_float_neg(op, op_rc, ctx),
            OpCode::FloatAbs => self.optimize_float_abs(op, op_rc, ctx),

            // ── Identity ops ──
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
                self.optimize_same_as(op, op_rc, ctx)
            }

            // ── Conditional calls ──
            OpCode::CondCallN => {
                if let Some(0) = ctx.get_constant_int_box(&op.arg(0).get_box_replacement(false)) {
                    ctx.last_op_removed = true;
                    return OptimizationResult::Remove;
                }
                if let Some(c) = ctx.get_constant_int_box(&op.arg(0).get_box_replacement(false))
                    && c != 0
                {
                    let mut call_op = Op::new(
                        OpCode::CallN,
                        &(1..op.num_args()).map(|i| op.arg(i)).collect::<Vec<_>>(),
                    );
                    call_op.pos().set(op.pos().get());
                    if let Some(d) = op.getdescr() {
                        call_op.setdescr(d);
                    }
                    ctx.last_op_removed = false;
                    return OptimizationResult::Replace(call_op);
                }
                ctx.last_op_removed = false;
                OptimizationResult::PassOn
            }
            // rewrite.py: optimize_COND_CALL_VALUE_I/R
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                let nullness = self.getnullness(op.arg(0).to_opref(), ctx);
                // rewrite.py:486-489: INFO_NONNULL → result is arg(0)
                if nullness == Nullness::Nonnull {
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_arg = ctx.resolve_operand_operand(&op.arg(0));
                    ctx.make_equal_to(&b_old, &b_arg);
                    ctx.last_op_removed = true;
                    return OptimizationResult::Remove;
                }
                // rewrite.py:490-493: INFO_NULL → demote to CALL_PURE
                if nullness == Nullness::Null {
                    let call_opcode = if op.opcode == OpCode::CondCallValueI {
                        OpCode::CallPureI
                    } else {
                        OpCode::CallPureR
                    };
                    let mut call_op = Op::new(
                        call_opcode,
                        &(1..op.num_args()).map(|i| op.arg(i)).collect::<Vec<_>>(),
                    );
                    call_op.pos().set(op.pos().get());
                    if let Some(d) = op.getdescr() {
                        call_op.setdescr(d);
                    }
                    ctx.last_op_removed = false;
                    return OptimizationResult::Replace(call_op);
                }
                ctx.last_op_removed = false;
                OptimizationResult::PassOn
            }

            // ── Pointer equality (rewrite.py: _optimize_oois_ooisnot) ──
            OpCode::PtrEq | OpCode::InstancePtrEq => {
                let instance = matches!(op.opcode, OpCode::InstancePtrEq);
                if instance {
                    // rewrite.py optimize_INSTANCE_PTR_EQ:
                    //     arg0 = get_box_replacement(op.getarg(0))
                    //     arg1 = get_box_replacement(op.getarg(1))
                    //     self.pure_from_args2(rop.INSTANCE_PTR_EQ, arg1, arg0, op)
                    let arg0 = ctx.resolve_operand_operand(&op.arg(0));
                    let arg1 = ctx.resolve_operand_operand(&op.arg(1));
                    ctx.register_pure_from_args2(
                        OpCode::InstancePtrEq,
                        Operand::from_bound_op(op_rc),
                        arg1,
                        arg0,
                    );
                }
                self.optimize_oois_ooisnot(op, op_rc, false, instance, ctx)
            }
            OpCode::PtrNe | OpCode::InstancePtrNe => {
                let instance = matches!(op.opcode, OpCode::InstancePtrNe);
                if instance {
                    // rewrite.py optimize_INSTANCE_PTR_NE: same swap.
                    let arg0 = ctx.resolve_operand_operand(&op.arg(0));
                    let arg1 = ctx.resolve_operand_operand(&op.arg(1));
                    ctx.register_pure_from_args2(
                        OpCode::InstancePtrNe,
                        Operand::from_bound_op(op_rc),
                        arg1,
                        arg0,
                    );
                }
                self.optimize_oois_ooisnot(op, op_rc, true, instance, ctx)
            }

            // ── Cast round-trip elimination ──
            // rewrite.py:807-813: register pure inverse for CSE, then emit.
            OpCode::CastPtrToInt => {
                ctx.register_pure_from_args1(
                    OpCode::CastIntToPtr,
                    Operand::from_bound_op(op_rc),
                    op.arg(0),
                );
                OptimizationResult::PassOn
            }
            OpCode::CastIntToPtr => {
                ctx.register_pure_from_args1(
                    OpCode::CastPtrToInt,
                    Operand::from_bound_op(op_rc),
                    op.arg(0),
                );
                OptimizationResult::PassOn
            }
            // jtransform.py rewrite_op_cast_opaque_ptr: CAST_OPAQUE_PTR is identity (no-op).
            OpCode::CastOpaquePtr => {
                let b_old = Operand::from_bound_op(op_rc);
                let b_arg = ctx.resolve_operand_operand(&op.arg(0));
                ctx.make_equal_to(&b_old, &b_arg);
                OptimizationResult::Remove
            }

            // ── Float-bytes conversion round-trip elimination ──
            // rewrite.py:815-821: register inverse pure relationship for CSE.
            // CONVERT_FLOAT_BYTES_TO_LONGLONG(x) does NOT reduce to x —
            // it changes the bit representation. But if we later see
            // CONVERT_LONGLONG_BYTES_TO_FLOAT(result), pure.rs can
            // recognize the round-trip and recover x.
            OpCode::ConvertFloatBytesToLonglong => {
                ctx.register_pure_from_args1(
                    OpCode::ConvertLonglongBytesToFloat,
                    Operand::from_bound_op(op_rc),
                    op.arg(0),
                );
                OptimizationResult::PassOn
            }
            OpCode::ConvertLonglongBytesToFloat => {
                ctx.register_pure_from_args1(
                    OpCode::ConvertFloatBytesToLonglong,
                    Operand::from_bound_op(op_rc),
                    op.arg(0),
                );
                OptimizationResult::PassOn
            }

            // rewrite.py optimize_GUARD_NO_EXCEPTION:
            //
            //     def optimize_GUARD_NO_EXCEPTION(self, op):
            //         if self.last_emitted_operation is REMOVED:
            //             return  # the prior op was a CALL_PURE that
            //                     # was killed; kill the guard too
            //         return self.emit(op)
            //
            // `last_emitted_operation` is set by every pass's emit
            // (optimizer.py), so the flag reflects the
            // PREVIOUS op's fate regardless of which pass dropped it.
            // pyre's ctx.last_op_removed is the cross-pass equivalent.
            OpCode::GuardNoException => {
                if ctx.last_op_removed {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }
            // rewrite.py: optimize_GUARD_FUTURE_CONDITION
            OpCode::GuardFutureCondition => {
                ctx.patchguardop = Some(op.clone());
                OptimizationResult::Remove
            }

            // INT_SIGNEXT belongs on `OptIntBounds`, not `OptRewrite`.
            // rewrite.py has no `optimize_INT_SIGNEXT`; the handler lives
            // at intbounds.py (optimize + postprocess). pyre's
            // intbounds.rs's `optimize_int_signext` /
            // `postprocess_int_signext` already implement the full upstream
            // logic (is_within_range check), so this `OptRewrite` arm
            // is redundant — its weaker `nbytes == 8` shortcut admits a
            // strict subset of intbounds's removals. Removed for
            // line-by-line dispatch-shape parity.

            // rewrite.py: optimize_CALL_PURE_I
            // Dispatch based on oopspecindex to specialized handlers.
            // Constant-fold and CSE are handled by pure.rs; here we
            // only do oopspec-specific simplifications.
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                let __descr_arc_descr = op.getdescr();
                if let Some(descr) = __descr_arc_descr.as_ref()
                    && let Some(cd) = descr.as_call_descr()
                {
                    let ei = cd.get_extra_info();
                    match ei.oopspecindex {
                        // rewrite.py:688: OS_INT_UDIV
                        majit_ir::OopSpecIndex::IntUdiv => {
                            if self.optimize_call_int_udiv(op, op_rc, ctx) {
                                return OptimizationResult::Remove;
                            }
                        }
                        // rewrite.py:689: OS_INT_PY_DIV
                        majit_ir::OopSpecIndex::IntPyDiv => {
                            if let Some(result) = self.optimize_call_int_py_div(op, op_rc, ctx) {
                                return result;
                            }
                        }
                        // rewrite.py:692: OS_INT_PY_MOD
                        majit_ir::OopSpecIndex::IntPyMod => {
                            if let Some(result) = self.optimize_call_int_py_mod(op, op_rc, ctx) {
                                return result;
                            }
                        }
                        _ => {}
                    }
                }
                OptimizationResult::PassOn
            }

            // rewrite.py: optimize_CALL_LOOPINVARIANT_I
            OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN => {
                if let Some(func_val) = op.arg(0).get_box_replacement(false).const_int() {
                    // RPython: LoopInvariantOp.produce_op stores PreambleOp
                    // in loop_invariant_results during import. Transfer from
                    // ctx.imported_loop_invariant_results on first access.
                    if let Some(&(_, source)) = ctx
                        .imported_loop_invariant_results
                        .iter()
                        .find(|(k, _)| *k == func_val)
                        && !self
                            .loop_invariant_results
                            .iter()
                            .any(|(k, _)| *k == func_val)
                    {
                        // RPython shortpreamble.py:158-159. Dual-slot case:
                        // `produce_loop_invariant` installs
                        // `make_equal_to(source, result_opref)`, so the source
                        // box's `_forwarded` slot now holds
                        // `Forwarded::Op(result_op)`.
                        // Build the synthetic SameAsI replay at
                        // `result_opref` (= get_box_replacement(source))
                        // so `take_preamble_forwarded_opinfo` reads the
                        // info seeded at result_opref's slot per the
                        // dual-slot rule (`mod.rs`'s `replay_pos` closure).
                        let replay_pos = ctx.get_replacement_opref(source);
                        let source_op = ctx.materialize_operand_at(source);
                        let mut replay = Op::new(OpCode::SameAsI, std::slice::from_ref(&source_op));
                        replay.pos().set(replay_pos);
                        self.loop_invariant_results.insert(
                            func_val,
                            LoopInvariantEntry::Preamble(PreambleOp {
                                op: source_op,
                                invented_name: false,
                                preamble_op: OpRc::new(replay),
                                // Non-invented loop-invariant producer: the
                                // SameAs arm is never taken, so no source.
                                same_as_source: None,
                            }),
                        );
                    }
                    // rewrite.py:453-458: isinstance(resvalue, PreambleOp)
                    // → force_op_from_preamble → replace in dict
                    if let Some(entry) = self.loop_invariant_results.get(&func_val).cloned() {
                        let cached_result = match entry {
                            LoopInvariantEntry::Preamble(ref pop) => {
                                // unroll.py: force_op_from_preamble(preamble_op)
                                let forced = ctx.force_op_from_preamble_op(pop).to_opref();
                                self.loop_invariant_results
                                    .insert(func_val, LoopInvariantEntry::Direct(forced));
                                forced
                            }
                            LoopInvariantEntry::Direct(r) => r,
                        };
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_cached = ctx.get_box_replacement_operand(cached_result);
                        ctx.make_equal_to(&b_old, &b_cached);
                        ctx.last_op_removed = true;
                        return OptimizationResult::Remove;
                    }
                    // Cache miss: demote and record result
                    self.loop_invariant_results
                        .insert(func_val, LoopInvariantEntry::Direct(op.pos().get()));
                    // rewrite.py `CallLoopinvariantOptimizationResult`:
                    // `_callback` records the emitted producer.
                    self.pending_loopinvariant_key = Some(func_val);
                }
                let call_opcode = OpCode::call_for_type(op.result_type());
                let new_op = op.copy_and_change(call_opcode, None, None);
                new_op.pos().set(op.pos().get());
                ctx.last_op_removed = false;
                OptimizationResult::Emit(new_op)
            }

            // ── rewrite.py: optimize_ASSERT_NOT_NONE ──
            OpCode::AssertNotNone => {
                // RPython: self.make_nonnull(op.getarg(0))
                let obj_box = op.arg(0).get_box_replacement(false);
                let has_info = ctx.has_ptr_info(&obj_box);
                if !has_info {
                    ctx.set_ptr_info(&obj_box, crate::optimizeopt::info::PtrInfo::nonnull());
                }
                OptimizationResult::Remove
            }

            // rewrite.py optimize_RECORD_EXACT_CLASS:
            //     opinfo = getptrinfo(op.getarg(0))
            //     expectedclassbox = op.getarg(1)
            //     if opinfo is not None:
            //         realclassbox = opinfo.get_known_class(cpu)
            //         if realclassbox is not None:
            //             assert realclassbox.same_constant(expectedclassbox)
            //             return
            //     self.make_constant_class(op.getarg(0), expectedclassbox,
            //                              update_last_guard=False)
            OpCode::RecordExactClass => {
                if op.num_args() >= 2 {
                    // RPython `RECORD_EXACT_CLASS` carries the same ConstInt
                    // vtable address shape as GUARD_CLASS.
                    let expected_class = ctx.get_constant_int_box(&op.arg(1));
                    if let Some(expected_class) = expected_class {
                        // getptrinfo synthesizes ConstPtrInfo for constant
                        // Refs so `get_known_class` reads cls_of_box for them.
                        let obj_box = op.arg(0).get_box_replacement(false);
                        if let Some(known) = ctx
                            .getptrinfo(&obj_box)
                            .and_then(|i| i.get_known_class(ctx.cpu.as_ref()))
                        {
                            debug_assert_eq!(known, expected_class);
                            return OptimizationResult::Remove;
                        }
                        crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                            ctx,
                            &obj_box,
                            expected_class,
                            false, // update_last_guard=False
                        );
                    }
                }
                OptimizationResult::Remove
            }

            // rewrite.py: optimize_record_exact_value
            //   box = op.getarg(0)
            //   expectedconstbox = op.getarg(1)
            //   assert isinstance(expectedconstbox, Const)
            //   self.make_constant(box, expectedconstbox)
            //
            // `make_constant` walks the forwarding chain internally
            // (optimizer.py `box = get_box_replacement(box)`), so
            // upstream passes `op.getarg(0)` raw without a prior
            // `get_box_replacement` resolution. Pyre matches.
            OpCode::RecordExactValueI | OpCode::RecordExactValueR => {
                let val = op.arg(1).get_box_replacement(false).const_value().expect(
                    "rewrite.py:400 — RECORD_EXACT_VALUE expectedconstbox \
                     must be a Const",
                );
                ctx.make_constant_arg(&op.arg(0), val);
                OptimizationResult::Remove
            }

            // rewrite.py: optimize_CALL_N — dispatch on oopspecindex
            OpCode::CallN | OpCode::CallI | OpCode::CallR => {
                let __descr_arc_descr = op.getdescr();
                if let Some(descr) = __descr_arc_descr.as_ref()
                    && let Some(cd) = descr.as_call_descr()
                {
                    let ei = cd.get_extra_info();
                    match ei.oopspecindex {
                            // rewrite.py:580-590: OS_ARRAYCOPY / OS_ARRAYMOVE
                            majit_ir::OopSpecIndex::Arraycopy => {
                                if op.num_args() >= 6
                                    && self.optimize_call_arrayop(
                                        op,
                                        op.arg(1).to_opref(),
                                        op.arg(2).to_opref(), // source, dest
                                        op.arg(3).to_opref(),
                                        op.arg(4).to_opref(),
                                        op.arg(5).to_opref(), // src_start, dst_start, length
                                        ctx,
                                    ) {
                                        return OptimizationResult::Remove;
                                    }
                            }
                            majit_ir::OopSpecIndex::Arraymove
                                // rewrite.py:592-597: ARRAYMOVE: source == dest
                                if op.num_args() >= 5 => {
                                    let array_box = op.arg(1);
                                    if self.optimize_call_arrayop(
                                        op,
                                        array_box.to_opref(),
                                        array_box.to_opref(), // source == dest
                                        op.arg(2).to_opref(),
                                        op.arg(3).to_opref(),
                                        op.arg(4).to_opref(),
                                        ctx,
                                    ) {
                                        return OptimizationResult::Remove;
                                    }
                                }
                            _ => {}
                        }
                }
                OptimizationResult::PassOn
            }

            // Everything else: pass on to next optimization pass
            _ => OptimizationResult::PassOn,
        }
    }

    fn setup(&mut self) {
        // ctx.last_op_removed is initialised by OptContext::new() and
        // maintained cross-pass by propagate_from_pass_range +
        // emit_operation — no per-pass setup needed.
        self.loop_invariant_results.clear();
        self.loop_invariant_producer.clear();
        self.pending_loopinvariant_key = None;
    }

    fn name(&self) -> &'static str {
        "rewrite"
    }

    fn have_postprocess(&self) -> bool {
        true
    }

    /// rewrite.py postprocess_GUARD_VALUE,
    /// rewrite.py postprocess_GUARD_TRUE / postprocess_GUARD_FALSE.
    ///
    /// The `make_constant(box, CONST_*)` call is the second half of PyPy's
    /// `optimize_guard` (rewrite.py). PyPy emits the guard, then
    /// records that the guard's input box is now known constant. The Rust
    /// port keeps the same split that `have_postprocess` requires — the
    /// emit happens via `optimize_guard_true/false/value` and the
    /// `make_constant` happens here.
    ///
    /// Safety on stable OpRefs: PyPy uses fresh `Box` objects per loop
    /// iteration; majit uses positional `OpRef` slots. The constant lands
    /// on the resolved OpRef of the comparison result (e.g. the position
    /// of an `int_lt`), which is itself fresh per iteration: preamble and
    /// body optimization emit comparison ops into disjoint OpRef ranges.
    /// CSE within a single phase is the only way
    /// for two uses to share the same OpRef, in which case they describe
    /// the same value and the constant is correct for all of them. PyPy's
    /// stable-Box vs majit's stable-OpRef yield the same observable
    /// behavior.
    fn propagate_postprocess(&mut self, op: &Op, op_rc: &OpRc, ctx: &mut OptContext) {
        // rewrite.py `CallLoopinvariantOptimizationResult._callback`:
        // `loop_invariant_producer[key] = self.optimizer.getlastop()`.
        if let Some(key) = self.pending_loopinvariant_key.take() {
            self.loop_invariant_producer.insert(key, op_rc.clone());
        }
        match op.opcode {
            OpCode::GuardTrue => {
                ctx.make_constant_arg(&op.arg(0), majit_ir::Value::Int(1));
            }
            OpCode::GuardFalse => {
                ctx.make_constant_arg(&op.arg(0), majit_ir::Value::Int(0));
            }
            OpCode::GuardValue => {
                if op.num_args() >= 2
                    && let Some(val) = op.arg(1).get_box_replacement(false).const_value()
                {
                    ctx.make_constant_arg(&op.arg(0), val);
                }
            }
            // rewrite.py postprocess_GUARD_ISNULL
            //     self.make_constant(op.getarg(0), CONST_NULL)
            // Ref-typed → Value::Ref(NULL); Int-typed → Value::Int(0).
            OpCode::GuardIsnull => {
                let obj = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
                let null = if self.is_ref_typed(obj, ctx) {
                    majit_ir::Value::Ref(majit_ir::GcRef(0))
                } else {
                    majit_ir::Value::Int(0)
                };
                ctx.make_constant_arg(&op.arg(0), null);
            }
            _ => {}
        }
    }

    /// rewrite.py: produce_potential_short_preamble_ops
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        for (_, op) in &self.loop_invariant_producer {
            sb.add_loopinvariant_op(ctx, op);
        }
    }

    /// rewrite.py serialize_optrewrite
    fn serialize_optrewrite(&self) -> Vec<(i64, OpRef)> {
        self.loop_invariant_results
            .iter()
            .map(|(func_ptr, entry)| match entry {
                LoopInvariantEntry::Direct(r) => (*func_ptr, *r),
                LoopInvariantEntry::Preamble(pop) => (*func_ptr, pop.op.to_opref()),
            })
            .collect()
    }

    /// rewrite.py deserialize_optrewrite
    fn deserialize_optrewrite(&mut self, entries: &[(i64, OpRef)]) {
        for &(func_ptr, result) in entries {
            self.loop_invariant_results
                .insert(func_ptr, LoopInvariantEntry::Direct(result));
        }
    }
}

impl Default for OptRewrite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::GcRef;

    #[test]
    fn check_subclass_reads_cpu_subclassrange_fields() {
        #[repr(C)]
        struct Class {
            subclassrange_min: i64,
            subclassrange_max: i64,
        }
        let parent = Class {
            subclassrange_min: 0,
            subclassrange_max: 4,
        };
        let child = Class {
            subclassrange_min: 2,
            subclassrange_max: 3,
        };
        let other = Class {
            subclassrange_min: 10,
            subclassrange_max: 11,
        };
        let prev = majit_backend::cpu_subclassrange_min_offset();
        majit_backend::set_cpu_subclassrange_min_offset(Some(0));
        assert_eq!(
            check_subclass(
                &child as *const Class as i64,
                &parent as *const Class as i64
            ),
            Some(true)
        );
        assert_eq!(
            check_subclass(
                &other as *const Class as i64,
                &parent as *const Class as i64
            ),
            Some(false)
        );
        majit_backend::set_cpu_subclassrange_min_offset(prev);
    }

    /// Producer-position trace spec. A consumer's `args` name the result
    /// positions of earlier producers in the same spec slice, so no op-arg
    /// is constructed as a position-only `Operand::from_opref(...)` box;
    /// [`build_specs`] later binds each arg to its producing `OpRc`.
    #[derive(Clone)]
    struct OpSpec {
        opcode: OpCode,
        args: Vec<u32>,
    }

    fn op_spec(opcode: OpCode, args: &[u32]) -> OpSpec {
        OpSpec {
            opcode,
            args: args.to_vec(),
        }
    }

    /// Turn a producer-position spec slice into a bound `OpRc` graph,
    /// oparser-faithful (`rpython/jit/tool/oparser.py`): each producer op
    /// is appended at its index position and every consumer arg references
    /// the producing op's bound result box (`from_bound_op`), mirroring
    /// oparser's `self.vars[name] = resop; args.append(self.vars[arg])`
    /// object-identity wiring — so each arg sheds to `Operand::Op` at
    /// construction, never the position-only `Operand::Box`.
    fn build_specs(specs: &[OpSpec]) -> Vec<majit_ir::OpRc> {
        let mut ops: Vec<majit_ir::OpRc> = Vec::new();
        for (pos, spec) in specs.iter().enumerate() {
            let arg_ops: Vec<Operand> = spec
                .args
                .iter()
                .map(|&p| Operand::from_bound_op(&ops[p as usize]))
                .collect();
            let op = OpRc::new(Op::new(spec.opcode, &arg_ops));
            op.pos()
                .set(OpRef::op_typed(pos as u32, spec.opcode.result_type()));
            ops.push(op);
        }
        ops
    }

    /// The operand the spec graph already carries at `opref`. Seeding a
    /// freshly materialized box would forward a different object than the
    /// one consumer args walk.
    fn producer_operand(ops: &[majit_ir::OpRc], opref: OpRef) -> Operand {
        ops.iter()
            .find(|op| op.pos().get() == opref)
            .map(|op| Operand::from_bound_op(op))
            .unwrap_or_else(|| panic!("no producer at {opref:?}"))
    }

    fn same_i() -> OpSpec {
        op_spec(OpCode::SameAsI, &[])
    }

    fn same_f() -> OpSpec {
        op_spec(OpCode::SameAsF, &[])
    }

    fn same_r() -> OpSpec {
        op_spec(OpCode::SameAsR, &[])
    }

    fn bin_i(opcode: OpCode, left: u32, right: u32) -> OpSpec {
        op_spec(opcode, &[left, right])
    }

    fn bin_f(opcode: OpCode, left: u32, right: u32) -> OpSpec {
        op_spec(opcode, &[left, right])
    }

    fn bin_r(opcode: OpCode, left: u32, right: u32) -> OpSpec {
        op_spec(opcode, &[left, right])
    }

    fn unary_i(opcode: OpCode, arg: u32) -> OpSpec {
        op_spec(opcode, &[arg])
    }

    fn unary_f(opcode: OpCode, arg: u32) -> OpSpec {
        op_spec(opcode, &[arg])
    }

    fn unary_r(opcode: OpCode, arg: u32) -> OpSpec {
        op_spec(opcode, &[arg])
    }

    fn run_one(
        specs: Vec<OpSpec>,
        target: usize,
        constants: &[(OpRef, Value)],
    ) -> (OptimizationResult, OptContext) {
        let ops = build_specs(&specs);
        let mut ctx = OptContext::new(ops.len());
        // Emit the producer the args already wrap. A cloned `Op` would be a
        // second object at the same position.
        for op in &ops[..target] {
            ctx.emit_rc(op.clone());
        }
        for &(opref, value) in constants {
            let b = producer_operand(&ops, opref);
            ctx.make_constant_box(&b, value);
        }
        let mut passes = test_pass_chain();
        let mut op = (*ops[target]).clone();
        resolve_op_args_in_ctx(&mut op, &mut ctx);
        let mut op_rc = OpRc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let result = loop {
            let mut result = OptimizationResult::PassOn;
            for pass in passes.iter_mut() {
                result = pass.propagate_forward(&op, &op_rc, &mut ctx);
                if !matches!(result, OptimizationResult::PassOn) {
                    break;
                }
            }
            match result {
                OptimizationResult::Restart(new_op) => {
                    // optimizer.py `send_extra_operation`: autogen rewrites
                    // return Restart so chained rules re-fire on the new op
                    // (`sub_const_canonicalize` then `add_zero` / OptPure).
                    op_rc = OpRc::new(new_op);
                    ctx.supersede_restart_producer(&op_rc);
                    op = (*op_rc).clone();
                    ctx.bind_input_resops(std::slice::from_ref(&op_rc));
                }
                other => break other,
            }
        };
        (result, ctx)
    }

    /// The int-rewrite slice of the production pipeline: integer rule
    /// rewrites live in OptIntBounds (autogenintrules.py), all-constant
    /// pure folding in OptPure (pure.py), with OptRewrite between
    /// them as in default_pipeline. Tests in this module assert the
    /// chain's observable result, wherever the individual rule lives.
    fn test_pass_chain() -> Vec<Box<dyn crate::optimizeopt::Optimization>> {
        vec![
            Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()),
            Box::new(OptRewrite::new()),
            Box::new(crate::optimizeopt::pure::OptPure::new()),
        ]
    }

    /// `run_one` against OptRewrite alone, for tests that assert what the
    /// rewrite pass itself must NOT do (a chained pass would mask it).
    fn run_one_rewrite_only(
        specs: Vec<OpSpec>,
        target: usize,
        constants: &[(OpRef, Value)],
    ) -> (OptimizationResult, OptContext) {
        run_one_rewrite_only_with_mul_high(specs, target, constants, &[], true)
    }

    fn run_one_rewrite_only_with_mul_high(
        specs: Vec<OpSpec>,
        target: usize,
        constants: &[(OpRef, Value)],
        nonneg: &[OpRef],
        supports_efficient_uint_mul_high: bool,
    ) -> (OptimizationResult, OptContext) {
        let ops = build_specs(&specs);
        let mut ctx = OptContext::new(ops.len());
        ctx.supports_efficient_uint_mul_high = supports_efficient_uint_mul_high;
        for op in &ops[..target] {
            ctx.emit_rc(op.clone());
        }
        for &(opref, value) in constants {
            let b = producer_operand(&ops, opref);
            ctx.make_constant_box(&b, value);
        }
        for &opref in nonneg {
            let b = producer_operand(&ops, opref);
            let _ = ctx
                .getintbound_handle(&b)
                .borrow_mut()
                .expect("a materialized int operand has a live bound cell")
                .make_ge_const(0);
        }
        let mut pass = OptRewrite::new();
        let mut op = (*ops[target]).clone();
        resolve_op_args_in_ctx(&mut op, &mut ctx);
        let op_rc = OpRc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let result = pass.propagate_forward(&op, &op_rc, &mut ctx);
        (result, ctx)
    }

    fn resolve_op_args_in_ctx(op: &mut Op, ctx: &mut OptContext) {
        // optimizer.py:651-652 setarg loop parity. Direct unit tests that
        // bypass Optimizer::propagate_from_pass_range still need the same
        // canonical operand args that production passes receive.
        for i in 0..op.num_args() {
            let arg = op.arg(i);
            let resolved = match ctx.resolve_operand_operand_opt(&arg) {
                Some(b) => b,
                None => {
                    let argref = arg.to_opref();
                    if argref.is_none() {
                        arg.clone()
                    } else {
                        ctx.materialize_operand_at(argref)
                            .get_box_replacement(false)
                    }
                }
            };
            op.setarg(i, resolved);
        }
    }

    fn assert_remove(result: &OptimizationResult) {
        assert!(matches!(result, OptimizationResult::Remove));
    }

    fn assert_pass_on(result: &OptimizationResult) {
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    fn assert_int_const(ctx: &OptContext, opref: OpRef, expected: i64) {
        assert_eq!(
            ctx.get_box_replacement_operand_opt(opref)
                .and_then(|cb| cb.const_int()),
            Some(expected)
        );
    }

    fn assert_forward(ctx: &OptContext, from: OpRef, to: OpRef) {
        assert_eq!(ctx.get_replacement_opref(from), to);
    }

    /// Run the rewrite pass on a sequence of ops and return the optimized ops.
    #[allow(dead_code)]
    fn run_rewrite(specs: &[OpSpec]) -> (Vec<Op>, OptContext) {
        let ops = build_specs(specs);
        let mut ctx = OptContext::new(ops.len());
        let mut passes = test_pass_chain();

        for op in ops.iter() {
            // Resolve forwarded arguments
            let mut resolved = (**op).clone();
            resolve_op_args_in_ctx(&mut resolved, &mut ctx);

            let __pf_rc = OpRc::new(resolved.clone());
            ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
            let mut result = OptimizationResult::PassOn;
            for pass in passes.iter_mut() {
                result = pass.propagate_forward(&resolved, &__pf_rc, &mut ctx);
                if !matches!(result, OptimizationResult::PassOn) {
                    break;
                }
            }
            match result {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replacement)
                | OptimizationResult::Restart(replacement) => {
                    ctx.emit(replacement);
                }
                OptimizationResult::Remove => {
                    // removed, nothing emitted
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
        }

        let new_ops: Vec<Op> = ctx.new_operations.iter().map(|rc| (**rc).clone()).collect();
        (new_ops, ctx)
    }

    // ── Binary integer operation tests (consolidated) ──
    // RPython rewrite.py: identity, absorbing, constant-fold rules for all binops.

    /// Helper: test a binary op where one arg is constant → expect Remove + forwarding.
    fn assert_binop_identity(
        opcode: OpCode,
        const_pos: usize,
        const_val: i64,
        expected_forward_to: u32,
    ) {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
            2,
            &[(OpRef::int_op(const_pos as u32), Value::Int(const_val))],
        );
        assert_remove(&result);
        assert_forward(&ctx, OpRef::int_op(2), OpRef::int_op(expected_forward_to));
    }

    /// Helper: test constant fold → expect Remove + constant result.
    fn assert_binop_const_fold(opcode: OpCode, a: i64, b: i64, expected: i64) {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(a)),
                (OpRef::int_op(1), Value::Int(b)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), expected);
    }

    /// Helper: test same-arg binop → expect Remove.
    fn assert_binop_self(opcode: OpCode, expected_const: Option<i64>) {
        let (result, ctx) = run_one(vec![same_i(), bin_i(opcode, 0, 0)], 1, &[]);
        assert_remove(&result);
        if let Some(val) = expected_const {
            assert_int_const(&ctx, OpRef::int_op(1), val);
        }
    }

    #[test]
    fn test_int_add_identities() {
        // x + 0 = x
        assert_binop_identity(OpCode::IntAdd, 1, 0, 0);
        // 0 + x = x
        assert_binop_identity(OpCode::IntAdd, 0, 0, 1);
        // constant fold
        assert_binop_const_fold(OpCode::IntAdd, 10, 20, 30);
    }

    #[test]
    fn test_int_sub_identities() {
        // x - 0 = x
        assert_binop_identity(OpCode::IntSub, 1, 0, 0);
        // x - x = 0
        assert_binop_self(OpCode::IntSub, Some(0));
        // constant fold (sub_const_canonicalize then OptPure)
        assert_binop_const_fold(OpCode::IntSub, 30, 10, 20);
    }

    #[test]
    fn test_int_mul_identities() {
        // x * 0 = 0
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntMul, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);

        // x * 1 = x
        assert_binop_identity(OpCode::IntMul, 1, 1, 0);
        // constant fold
        assert_binop_const_fold(OpCode::IntMul, 6, 7, 42);
    }

    #[test]
    fn test_int_mul_power_of_two() {
        // x * 8 → lshift(x, 3)
        let (result, _) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntMul, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(8))],
        );
        match result {
            OptimizationResult::Replace(ref new_op) | OptimizationResult::Emit(ref new_op) => {
                assert_eq!(new_op.opcode, OpCode::IntLshift);
            }
            _ => {} // may also Remove with forwarding
        }
    }

    #[test]
    fn test_int_floordiv_identities() {
        // x / 1 = x
        assert_binop_identity(OpCode::IntFloorDiv, 1, 1, 0);
        // 0 / x = 0
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntFloorDiv, 0, 1)],
            2,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
        // x / x = 1
        assert_binop_self(OpCode::IntFloorDiv, Some(1));
        // x / -1 = neg(x)
        // constant fold
        assert_binop_const_fold(OpCode::IntFloorDiv, 42, 7, 6);
    }

    #[test]
    fn test_int_mod_identities() {
        // x % 1 = 0
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntMod, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(1))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
        // x % x = 0
        assert_binop_self(OpCode::IntMod, Some(0));
    }

    /// Whether the pass emitted `opcode`.  `intdiv` queues its sequence
    /// through `emit_extra`, so it is still in `extra_operations_after` when
    /// the pass returns.
    fn emitted_opcode(ctx: &OptContext, opcode: OpCode) -> bool {
        ctx.new_operations
            .iter()
            .map(|op| op.opcode)
            .chain(ctx.extra_operations_after.iter().map(|(_, op)| op.opcode))
            .any(|emitted| emitted == opcode)
    }

    /// Whether the magic-number expansion ran.
    fn emitted_uint_mul_high(ctx: &OptContext) -> bool {
        emitted_opcode(ctx, OpCode::UintMulHigh)
    }

    /// A `CALL_PURE_I` carrying one of the Python division oopspecs, with the
    /// divisor already constant-folded — the shape `optimize_call_int_py_div`
    /// and `optimize_call_int_py_mod` are reached through.
    fn run_int_py_call(
        oopspecindex: majit_ir::OopSpecIndex,
        divisor: i64,
        supports_efficient_uint_mul_high: bool,
    ) -> (OptimizationResult, OptContext) {
        let ops = build_specs(&[same_i(), same_i(), same_i()]);
        let mut ctx = OptContext::new(4);
        ctx.supports_efficient_uint_mul_high = supports_efficient_uint_mul_high;
        for op in &ops {
            ctx.emit_rc(op.clone());
        }
        let b = Operand::from_bound_op(&ops[2]);
        ctx.make_constant_box(&b, Value::Int(divisor));

        let mut call = Op::new(
            OpCode::CallPureI,
            &[
                Operand::from_bound_op(&ops[0]),
                Operand::from_bound_op(&ops[1]),
                Operand::from_bound_op(&ops[2]),
            ],
        );
        call.pos().set(OpRef::int_op(3));
        call.setdescr(std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            0,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            true,
            std::mem::size_of::<i64>(),
            majit_ir::EffectInfo::new(majit_ir::ExtraEffect::ElidableCanRaise, oopspecindex),
        )));
        resolve_op_args_in_ctx(&mut call, &mut ctx);
        let op_rc = OpRc::new(call.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let mut pass = OptRewrite::new();
        let result = pass.propagate_forward(&call, &op_rc, &mut ctx);
        (result, ctx)
    }

    /// `IntFloorDiv` and `IntMod` truncate, while both strength reductions
    /// below them compute a floor.  The two agree only for a non-negative
    /// dividend, so an unknown-sign one keeps the native op no matter what
    /// the backend can multiply.
    #[test]
    fn constant_division_and_modulo_keep_the_native_op_for_an_unknown_sign_dividend() {
        for opcode in [OpCode::IntFloorDiv, OpCode::IntMod] {
            for mul_high in [false, true] {
                let (result, ctx) = run_one_rewrite_only_with_mul_high(
                    vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
                    2,
                    &[(OpRef::int_op(1), Value::Int(100))],
                    &[],
                    mul_high,
                );
                assert_pass_on(&result);
                assert!(
                    !emitted_uint_mul_high(&ctx),
                    "{opcode:?} took the floor expansion for a possibly-negative \
                     dividend (mul_high={mul_high})"
                );
            }
        }
    }

    #[test]
    fn backend_without_mul_high_keeps_native_constant_division_and_modulo() {
        for opcode in [OpCode::IntFloorDiv, OpCode::IntMod] {
            let (result, ctx) = run_one_rewrite_only_with_mul_high(
                vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
                2,
                &[(OpRef::int_op(1), Value::Int(100))],
                &[OpRef::int_op(0)],
                false,
            );
            assert_pass_on(&result);
            assert!(
                !emitted_uint_mul_high(&ctx),
                "{opcode:?} expanded through UintMulHigh on a backend that lacks it"
            );
        }
    }

    #[test]
    fn backend_with_mul_high_expands_constant_division_and_modulo() {
        for opcode in [OpCode::IntFloorDiv, OpCode::IntMod] {
            let (_result, ctx) = run_one_rewrite_only_with_mul_high(
                vec![same_i(), same_i(), bin_i(opcode, 0, 1)],
                2,
                &[(OpRef::int_op(1), Value::Int(100))],
                &[OpRef::int_op(0)],
                true,
            );
            assert!(
                emitted_uint_mul_high(&ctx),
                "{opcode:?} kept the native division on a backend that has UintMulHigh"
            );
        }
    }

    /// `rewrite.py` / `:758-766` reach `intdiv.modulo_operations` /
    /// `division_operations` with no capability test of any kind, so the
    /// residual call goes away and the expansion is the same one on every
    /// backend.  A backend that has to emulate `UintMulHigh` pays for it in
    /// its own lowering, not by keeping the call.
    #[test]
    fn call_int_py_div_and_mod_always_expand_through_mul_high() {
        for oopspecindex in [
            majit_ir::OopSpecIndex::IntPyDiv,
            majit_ir::OopSpecIndex::IntPyMod,
        ] {
            for mul_high in [false, true] {
                let (result, ctx) = run_int_py_call(oopspecindex, 100, mul_high);
                assert_remove(&result);
                assert!(
                    emitted_uint_mul_high(&ctx),
                    "{oopspecindex:?} left the residual call (mul_high={mul_high})"
                );
            }
        }
    }

    #[test]
    fn test_int_bitwise_identities() {
        // AND: x & 0 = 0, x & -1 = x, x & x = x
        assert_binop_identity(OpCode::IntAnd, 0, -1i64, 1); // -1 & x = x
        assert_binop_self(OpCode::IntAnd, None); // x & x = x (forward to x)

        // OR: x | 0 = x, x | -1 = -1, x | x = x
        assert_binop_identity(OpCode::IntOr, 1, 0, 0);
        assert_binop_self(OpCode::IntOr, None);

        // XOR: x ^ 0 = x, x ^ x = 0, x ^ -1 = ~x
        assert_binop_identity(OpCode::IntXor, 1, 0, 0);
        assert_binop_self(OpCode::IntXor, Some(0));
        assert_binop_const_fold(OpCode::IntXor, 0xFF, 0x0F, 0xF0);
    }

    #[test]
    fn test_shift_identities() {
        // x << 0 = x
        assert_binop_identity(OpCode::IntLshift, 1, 0, 0);
        // x >> 0 = x
        assert_binop_identity(OpCode::IntRshift, 1, 0, 0);
        // constant fold
        assert_binop_const_fold(OpCode::IntLshift, 1, 4, 16);
    }

    #[test]
    fn test_unary_constant_fold() {
        // neg constant
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntNeg, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(42))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), -42);

        // invert constant
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntInvert, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(0xFF))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), !0xFF);
    }

    #[test]
    fn test_int_is_zero_and_is_true() {
        // is_zero(0) = 1
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntIsZero, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 1);

        // is_zero(5) = 0
        let (result, ctx) = run_one(
            vec![same_i(), unary_i(OpCode::IntIsZero, 0)],
            1,
            &[(OpRef::int_op(0), Value::Int(5))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 0);
    }

    #[test]
    fn test_comparison_constant_fold() {
        assert_binop_const_fold(OpCode::IntLt, 3, 5, 1);
        assert_binop_const_fold(OpCode::IntLt, 5, 3, 0);
        assert_binop_const_fold(OpCode::IntEq, 7, 7, 1);
        assert_binop_const_fold(OpCode::IntEq, 7, 8, 0);
        assert_binop_const_fold(OpCode::UintLt, 3, 5, 1);
    }

    // ── Guard tests ──

    #[test]
    fn test_guard_true_known_true() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardTrue, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(1))],
        );
        assert_remove(&result);
    }

    #[test]
    fn test_guard_true_known_false() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardTrue, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert!(
            matches!(result, OptimizationResult::InvalidLoop(_)),
            "guard_true(0) should abort as InvalidLoop, got {result:?}"
        );
    }

    #[test]
    fn test_guard_true_unknown() {
        let (result, _) = run_one(vec![same_i(), op_spec(OpCode::GuardTrue, &[0])], 1, &[]);
        assert_pass_on(&result);
    }

    #[test]
    fn test_guard_false_known_false() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardFalse, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
    }

    #[test]
    fn test_guard_false_known_true() {
        let (result, _) = run_one(
            vec![same_i(), op_spec(OpCode::GuardFalse, &[0])],
            1,
            &[(OpRef::int_op(0), Value::Int(1))],
        );
        assert!(
            matches!(result, OptimizationResult::InvalidLoop(_)),
            "guard_false(1) should abort as InvalidLoop, got {result:?}"
        );
    }

    /// `rewrite.py` splits GUARD_ISNULL in two: `optimize_GUARD_ISNULL` emits the
    /// guard, `postprocess_` runs `make_constant(op.getarg(0), CONST_NULL)`.
    /// Running that `make_constant` in the forward half installs the Const
    /// forwarding before `emit_operation` resolves the guard's own arg 0, and
    /// the guard reaches the backend as `guard_isnull(ConstPtr(NULL))` — a
    /// tautology that never fails, so the null the trace was specialized on is
    /// never re-checked. Same shape as the GUARD_VALUE half of #210.
    #[test]
    fn guard_isnull_leaves_its_own_operand_unfolded() {
        // p0 = inputarg(Ref); guard_isnull(p0); finish(p0)
        let p0 = crate::history::test_support::TraceBuilder::new().input(majit_ir::Type::Ref, 0);
        let ops = {
            let mut guard = Op::new(OpCode::GuardIsnull, &[p0.clone()]);
            guard.pos().set(OpRef::void_op(0));
            let mut finish = Op::new(OpCode::Finish, &[p0]);
            finish.pos().set(OpRef::void_op(1));
            vec![guard, finish]
        };
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.trace_inputargs = OpRef::inputarg_refs(&[majit_ir::Type::Ref]);
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1);

        let guard = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardIsnull)
            .expect("GUARD_ISNULL must survive: nothing here proves the pointer null");
        assert_eq!(
            guard.arg(0).const_value(),
            None,
            "the guard the backend receives still has to carry the pointer it checks"
        );

        let finish = result
            .iter()
            .find(|o| o.opcode == OpCode::Finish)
            .expect("the FINISH must survive");
        assert_eq!(
            finish.arg(0).const_value(),
            Some(Value::Ref(majit_ir::GcRef(0))),
            "postprocess_GUARD_ISNULL still has to hand every later op CONST_NULL"
        );
    }

    #[test]
    fn guard_isnull_uses_the_inputargs_own_forwarding_chain() {
        let mut ctx = OptContext::with_inputarg_types(4, &[majit_ir::Type::Ref]);
        let input =
            majit_ir::InputArgRc::new(majit_ir::InputArg::from_type(majit_ir::Type::Ref, 0));
        let arg = Operand::from_bound_inputarg(&input);
        arg.set_forwarded_const(majit_ir::Const::from_gcref(majit_ir::GcRef(0)));
        let mut guard = Op::new(OpCode::GuardIsnull, std::slice::from_ref(&arg));
        guard.pos().set(OpRef::void_op(0));
        let guard_rc = OpRc::new(guard.clone());
        let mut pass = OptRewrite::new();

        let result = pass.propagate_forward(&guard, &guard_rc, &mut ctx);
        assert_remove(&result);
    }

    #[test]
    fn guard_value_uses_the_inputargs_own_forwarding_chain() {
        let mut ctx = OptContext::with_inputarg_types(4, &[majit_ir::Type::Ref]);
        let input =
            majit_ir::InputArgRc::new(majit_ir::InputArg::from_type(majit_ir::Type::Ref, 0));
        let arg = Operand::from_bound_inputarg(&input);
        let value = majit_ir::GcRef(42);
        arg.set_forwarded_const(majit_ir::Const::from_gcref(value));
        let mut guard = Op::new(
            OpCode::GuardValue,
            &[arg, Operand::const_from_value(majit_ir::Value::Ref(value))],
        );
        guard.pos().set(OpRef::void_op(0));
        let guard_rc = OpRc::new(guard.clone());
        let mut pass = OptRewrite::new();

        let result = pass.propagate_forward(&guard, &guard_rc, &mut ctx);
        assert_remove(&result);
    }

    #[test]
    fn test_guard_value_match() {
        let (result, _) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::GuardValue, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(42)),
                (OpRef::int_op(1), Value::Int(42)),
            ],
        );
        assert_remove(&result);
    }

    // ── SAME_AS tests ──

    #[test]
    fn test_same_as_i() {
        let (result, ctx) = run_one(vec![same_i(), op_spec(OpCode::SameAsI, &[0])], 1, &[]);
        assert_remove(&result);
        assert_forward(&ctx, OpRef::int_op(1), OpRef::int_op(0));
    }

    // ── Integration test: full optimizer with OptRewrite ──

    #[test]
    fn test_optimizer_integration_add_zero() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptRewrite::new()));

        // Create a trace: x = SameAsI(), y = SameAsI(constant 0), z = IntAdd(x, y)
        let ops = build_specs(&[
            same_i(),                         // op0: x
            same_i(),                         // op1: 0
            op_spec(OpCode::IntAdd, &[0, 1]), // op2: x + 0
        ]);

        // We need to set up constants before the optimizer runs.
        // The optimizer creates its own context, so we need a way to
        // inject constants. Since we're testing through the optimizer,
        // let's test the pass directly instead.
        let mut ctx = OptContext::new(3);

        let mut passes = test_pass_chain();

        // Simulate the optimizer loop on the spec OpRcs. Cloning into a fresh
        // OpRc would register a second producer at the same position.
        for (i, op) in ops.iter().enumerate() {
            for a in 0..op.num_args() {
                let resolved_arg = ctx.resolve_operand_operand(&op.arg(a));
                op.setarg(a, resolved_arg);
            }
            ctx.bind_input_resops(std::slice::from_ref(op));
            let mut result = OptimizationResult::PassOn;
            // The production loop absorbs SameAs* before the passes
            // (optimizer.py:864-867); the argless SameAsI fixtures here
            // stand in for inputs and are emitted directly.
            if op.opcode != OpCode::SameAsI {
                for pass in passes.iter_mut() {
                    result = pass.propagate_forward(&**op, op, &mut ctx);
                    if !matches!(result, OptimizationResult::PassOn) {
                        break;
                    }
                }
            }
            match result {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replacement)
                | OptimizationResult::Restart(replacement) => {
                    ctx.emit(replacement);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit_rc(op.clone());
                }
                OptimizationResult::InvalidLoop(_) => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
            // op1 is the zero constant the following IntAdd folds.
            if i == 1 {
                let b = Operand::from_bound_op(op);
                ctx.make_constant_box(&b, Value::Int(0));
            }
        }

        // The SameAsI(x) should be removed and forwarded, but we only
        // have SameAsI with no args (acting as input). Let's verify
        // the IntAdd was removed and the result is forwarded.
        // op0 is emitted, op1 is emitted (just a constant), op2 is removed.
        // After forwarding, any reference to op2 should resolve to op0.
        assert_eq!(
            ctx.get_replacement_opref(OpRef::int_op(2)),
            OpRef::int_op(0)
        );
    }

    #[test]
    fn test_optimizer_integration_chain() {
        // RPython parity: x - x -> 0, then guard_true(0) makes the trace impossible.
        let ops = build_specs(&[
            same_i(),                         // op0: x
            op_spec(OpCode::IntSub, &[0, 0]), // op1: x - x -> 0
            op_spec(OpCode::GuardTrue, &[1]), // op2: guard_true(0)
        ]);

        let mut ctx = OptContext::new(3);
        let mut passes = test_pass_chain();

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for op in &ops {
                for a in 0..op.num_args() {
                    let resolved_arg = ctx.resolve_operand_operand(&op.arg(a));
                    op.setarg(a, resolved_arg);
                }
                ctx.bind_input_resops(std::slice::from_ref(op));
                let mut result = OptimizationResult::PassOn;
                // SameAsI input fixtures bypass the passes, as in the
                // production loop's SameAs absorption (optimizer.py:864-867).
                if op.opcode != OpCode::SameAsI {
                    for pass in passes.iter_mut() {
                        result = pass.propagate_forward(&**op, op, &mut ctx);
                        if !matches!(result, OptimizationResult::PassOn) {
                            break;
                        }
                    }
                }
                match result {
                    OptimizationResult::Emit(emitted) => {
                        ctx.emit(emitted);
                    }
                    OptimizationResult::Replace(replacement)
                    | OptimizationResult::Restart(replacement) => {
                        ctx.emit(replacement);
                    }
                    OptimizationResult::Remove => {}
                    OptimizationResult::PassOn => {
                        ctx.emit_rc(op.clone());
                    }
                    OptimizationResult::InvalidLoop(_) => {
                        std::panic::panic_any(crate::optimize::InvalidLoop(
                            "guard proven to always fail",
                        ));
                    }
                }
            }
        }))
        .expect_err("guard_true(0) should abort the optimized trace");
        assert!(err.downcast_ref::<crate::optimize::InvalidLoop>().is_some());
    }

    // ── Wrapping arithmetic tests ──

    #[test]
    fn test_int_add_wrapping() {
        // wrapping
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntAdd, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(i64::MAX)),
                (OpRef::int_op(1), Value::Int(1)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), i64::MIN);
    }

    // ── Shift of zero constant tests ──

    #[test]
    fn test_zero_lshift_anything() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntLshift, 0, 1)],
            2,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
    }

    #[test]
    fn test_zero_rshift_anything() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntRshift, 0, 1)],
            2,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0);
    }

    // ── Non-optimizable cases (should PassOn) ──

    #[test]
    fn test_int_add_no_constants() {
        let (result, _) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntAdd, 0, 1)],
            2,
            &[],
        );
        assert_pass_on(&result);
    }

    #[test]
    fn test_unknown_opcode_passthrough() {
        // SETFIELD_GC(struct, value): struct is a Ref producer, value an Int
        // producer. OptRewrite has no rule for it → PassOn.
        let (result, _) = run_one(
            vec![same_r(), same_i(), op_spec(OpCode::SetfieldGc, &[0, 1])],
            2,
            &[],
        );
        assert_pass_on(&result);
    }

    // ── INT_AND constant fold ──

    #[test]
    fn test_int_and_constant_fold() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntAnd, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(0xFF)),
                (OpRef::int_op(1), Value::Int(0x0F)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0x0F);
    }

    // ── INT_OR constant fold ──

    #[test]
    fn test_int_or_constant_fold() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::IntOr, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(0xF0)),
                (OpRef::int_op(1), Value::Int(0x0F)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 0xFF);
    }

    // ── UINT_RSHIFT tests ──

    #[test]
    fn test_uint_rshift_zero() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::UintRshift, 0, 1)],
            2,
            &[(OpRef::int_op(1), Value::Int(0))],
        );
        assert_remove(&result);
        assert_forward(&ctx, OpRef::int_op(2), OpRef::int_op(0));
    }

    #[test]
    fn test_uint_rshift_constant_fold() {
        let (result, ctx) = run_one(
            vec![same_i(), same_i(), bin_i(OpCode::UintRshift, 0, 1)],
            2,
            &[
                (OpRef::int_op(0), Value::Int(-1)), // all ones
                (OpRef::int_op(1), Value::Int(1)),
            ],
        );
        assert_remove(&result);
        // u64::MAX >> 1 = i64::MAX
        assert_int_const(&ctx, OpRef::int_op(2), i64::MAX);
    }

    // ── Float optimization tests ──

    #[test]
    fn test_float_mul_one() {
        let cases = [
            ("right", OpRef::float_op(1), OpRef::float_op(0)),
            ("left", OpRef::float_op(0), OpRef::float_op(1)),
        ];
        for (name, const_op, forward_to) in cases {
            let (result, ctx) = run_one(
                vec![same_f(), same_f(), bin_f(OpCode::FloatMul, 0, 1)],
                2,
                &[(const_op, Value::Float(1.0))],
            );
            assert!(matches!(result, OptimizationResult::Remove), "case {name}");
            assert_eq!(
                ctx.get_replacement_opref(OpRef::float_op(2)),
                forward_to,
                "case {name}"
            );
        }
    }

    #[test]
    fn test_float_neg_double_negation() {
        // FloatNeg(FloatNeg(x)) -> x
        let ops = build_specs(&[
            same_f(),                        // op0: x
            op_spec(OpCode::FloatNeg, &[0]), // op1: -x
            op_spec(OpCode::FloatNeg, &[1]), // op2: -(-x) -> x
        ]);
        let mut ctx = OptContext::new(3);
        ctx.emit_rc(ops[0].clone());

        let mut pass = OptRewrite::new();
        // Process op1 first (pass it through) on the producer the next
        // op's arg already wraps.
        ctx.bind_input_resops(std::slice::from_ref(&ops[1]));
        let result1 = pass.propagate_forward(&ops[1], &ops[1], &mut ctx);
        assert!(matches!(result1, OptimizationResult::PassOn));
        ctx.emit_rc(ops[1].clone());

        // Process op2: should detect double negation
        for i in 0..ops[2].num_args() {
            let resolved = ctx.resolve_operand_operand(&ops[2].arg(i));
            ops[2].setarg(i, resolved);
        }
        ctx.bind_input_resops(std::slice::from_ref(&ops[2]));
        let result2 = pass.propagate_forward(&ops[2], &ops[2], &mut ctx);
        assert!(matches!(result2, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_replacement_opref(OpRef::float_op(2)),
            OpRef::float_op(0)
        );
    }

    #[test]
    fn test_float_truediv_power_of_two() {
        // x / 2.0 → x * 0.5
        let (result, _) = run_one(
            vec![same_f(), same_f(), bin_f(OpCode::FloatTrueDiv, 0, 1)],
            2,
            &[(OpRef::float_op(1), Value::Float(2.0))],
        );
        assert!(matches!(result, OptimizationResult::Emit(_)));
    }

    #[test]
    fn test_float_no_opt_passthrough() {
        // FloatAdd with no constants: no RPython rewrite → PassOn
        let (result, _) = run_one(
            vec![same_f(), same_f(), bin_f(OpCode::FloatAdd, 0, 1)],
            2,
            &[],
        );
        assert_pass_on(&result);
    }

    // ── COND_CALL tests ──

    #[test]
    fn test_cond_call_constant_false_removed() {
        // CondCallN(condition=0, func, arg1) -> removed (dead call)
        let (result, _) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallN, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        assert_remove(&result);
    }

    #[test]
    fn test_cond_call_constant_true_to_direct_call() {
        // CondCallN(condition=1, func, arg1) -> CallN(func, arg1)
        let (result, _) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallN, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(1))],
        );
        match result {
            OptimizationResult::Replace(op) => {
                assert_eq!(op.opcode, OpCode::CallN);
                // Should have args [func, arg1] (condition arg stripped)
                assert_eq!(op.num_args(), 2);
                assert_eq!(op.arg(0).to_opref(), OpRef::int_op(1));
                assert_eq!(op.arg(1).to_opref(), OpRef::int_op(2));
            }
            other => panic!("expected Replace(CallN), got {:?}", other),
        }
    }

    // ── COND_CALL_VALUE tests ──

    #[test]
    fn test_cond_call_value_nonnull_returns_value() {
        // CondCallValueI(value=42, func, arg1) -> value itself (no call needed)
        let (result, ctx) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallValueI, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(42))],
        );
        assert_remove(&result);
        let resolved = ctx.get_replacement_opref(OpRef::int_op(3));
        assert!(resolved.is_constant());
        assert_eq!(
            ctx.get_box_replacement_operand_opt(resolved)
                .and_then(|b| ctx.get_constant_int_box(&b)),
            Some(42)
        );
    }

    #[test]
    fn test_cond_call_value_null_to_direct_call() {
        // CondCallValueI(value=0, func, arg1) -> CallPureI(func, arg1)
        let (result, _) = run_one(
            vec![
                same_i(),
                same_i(),
                same_i(),
                op_spec(OpCode::CondCallValueI, &[0, 1, 2]),
            ],
            3,
            &[(OpRef::int_op(0), Value::Int(0))],
        );
        match result {
            OptimizationResult::Replace(op) => {
                assert_eq!(op.opcode, OpCode::CallPureI);
                assert_eq!(op.num_args(), 2);
                assert_eq!(op.arg(0).to_opref(), OpRef::int_op(1));
                assert_eq!(op.arg(1).to_opref(), OpRef::int_op(2));
            }
            other => panic!("expected Replace(CallPureI), got {:?}", other),
        }
    }

    // ── PTR_EQ / PTR_NE tests ──

    #[test]
    fn test_ptr_eq_same_opref() {
        // PtrEq(x, x) -> 1
        // resoperation.py InputArgRef / 615 RefOp `type = 'r'`: ptr
        // boxes carry the Ref variant tag, not Int.
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::PtrEq, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 1);
    }

    #[test]
    fn test_ptr_ne_same_opref() {
        // PtrNe(x, x) -> 0
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::PtrNe, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 0);
    }

    #[test]
    fn test_instance_ptr_eq_same_opref() {
        // InstancePtrEq(x, x) -> 1
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::InstancePtrEq, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 1);
    }

    #[test]
    fn test_instance_ptr_ne_same_opref() {
        // InstancePtrNe(x, x) -> 0
        let (result, ctx) = run_one(vec![same_r(), bin_r(OpCode::InstancePtrNe, 0, 0)], 1, &[]);
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(1), 0);
    }

    #[test]
    fn test_ptr_eq_ref_against_int_modelled_pointer_folds() {
        // `py_type_check` compares a heap object's `ob_type` against a
        // `&'static PyType`.  The field read is a `Value::Ref`; the static's
        // address is a pointer to non-moving memory, which pyre models as
        // `Value::Int` holding the same word.  Once the receiver itself is a
        // constant both args are constant and the pure pass folds the compare,
        // so the fold has to read the word out of either shape -- the
        // `(Ref, Ref)`-only spelling reached `execute_nonspec_const` with no
        // matching row and panicked as an unregistered opcode.
        let (result, ctx) = run_one(
            vec![same_r(), same_r(), bin_r(OpCode::PtrEq, 0, 1)],
            2,
            &[
                (OpRef::ref_op(0), Value::Ref(GcRef(4330386864))),
                (OpRef::ref_op(1), Value::Int(4330386864)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 1);
    }

    #[test]
    fn test_ptr_ne_ref_against_distinct_int_modelled_pointer_folds() {
        let (result, ctx) = run_one(
            vec![same_r(), same_r(), bin_r(OpCode::PtrNe, 0, 1)],
            2,
            &[
                (OpRef::ref_op(0), Value::Ref(GcRef(4330386864))),
                (OpRef::ref_op(1), Value::Int(4330408704)),
            ],
        );
        assert_remove(&result);
        assert_int_const(&ctx, OpRef::int_op(2), 1);
    }

    #[test]
    fn test_ptr_eq_distinct_constants_not_folded_in_rewrite() {
        // PtrEq(const 100, const 200): two distinct non-null ConstPtr.
        // rewrite.py _optimize_oois_ooisnot has no value-compare
        // branch — `arg0 is arg1` (line 542) is object identity, which is
        // False for distinct ConstPtr, so it falls through to `emit(op)`
        // (line 564). The actual constant fold lives in the pure pass
        // (pure.py → execute_ptr_compare_const), not rewrite.
        // history.py ConstPtr — Value::Ref must land on a Ref-tagged
        // OpRef so the box class identity matches the resoperation.py:615
        // RefOp mixin of the producer SameAsR.
        // OptRewrite alone: this asserts the REWRITE pass's behavior; the
        // chained harness would let OptPure fold it, masking a regression
        // where rewrite value-compares ConstPtr.
        let (result, ctx) = run_one_rewrite_only(
            vec![same_r(), same_r(), bin_r(OpCode::PtrEq, 0, 1)],
            2,
            &[
                (OpRef::ref_op(0), Value::Ref(GcRef(100))),
                (OpRef::ref_op(1), Value::Ref(GcRef(200))),
            ],
        );
        // rewrite.py `return self.emit(op)` — rewrite passes the op on
        // unchanged; it does not fold distinct constants.
        assert_pass_on(&result);
        assert_eq!(
            ctx.get_box_replacement_operand_opt(OpRef::int_op(2))
                .and_then(|cb| cb.const_int()),
            None
        );
    }

    // ── CAST round-trip tests ──

    #[test]
    fn test_cast_ptr_to_int_passes_through() {
        // rewrite.py:807-809: CastPtrToInt registers pure inverse, emits.
        // arg0 is a Ref box (resoperation.py RefOp `type = 'r'`).
        let (result, _) = run_one(vec![same_r(), unary_r(OpCode::CastPtrToInt, 0)], 1, &[]);
        assert_pass_on(&result);
    }

    #[test]
    fn test_cast_int_to_ptr_passes_through() {
        // rewrite.py:811-813: CastIntToPtr registers pure inverse, emits.
        let (result, _) = run_one(vec![same_i(), unary_i(OpCode::CastIntToPtr, 0)], 1, &[]);
        assert_pass_on(&result);
    }

    #[test]
    fn test_cast_opaque_ptr_eliminated() {
        // CastOpaquePtr(x) -> x
        let (result, ctx) = run_one(vec![same_r(), unary_r(OpCode::CastOpaquePtr, 0)], 1, &[]);
        assert_remove(&result);
        assert_forward(&ctx, OpRef::ref_op(1), OpRef::ref_op(0));
    }

    // ── CONVERT_FLOAT_BYTES tests ──
    // rewrite.py:815-821: these conversions are NOT eliminated —
    // they actually change bit representation. Only round-trips
    // (A→B→A) are eliminated via pure.rs CSE.

    #[test]
    fn test_convert_float_bytes_to_longlong_passes_through() {
        let (result, _) = run_one(
            vec![same_f(), unary_f(OpCode::ConvertFloatBytesToLonglong, 0)],
            1,
            &[],
        );
        assert_pass_on(&result);
    }

    #[test]
    fn test_convert_longlong_bytes_to_float_passes_through() {
        let (result, _) = run_one(
            vec![same_i(), unary_i(OpCode::ConvertLonglongBytesToFloat, 0)],
            1,
            &[],
        );
        // PassOn: op is emitted, no replacement registered.
        assert_pass_on(&result);
    }

    // ── GUARD_NO_EXCEPTION tests ──

    #[test]
    fn test_guard_no_exception_after_removed_call() {
        // CondCallN(condition=0, ...) -> removed, then GuardNoException -> removed
        let ops = build_specs(&[
            same_i(),                               // op0: condition (const 0)
            same_i(),                               // op1: func
            op_spec(OpCode::CondCallN, &[0, 1]),    // op2: removed
            op_spec(OpCode::GuardNoException, &[]), // op3: should be removed
        ]);
        let mut ctx = OptContext::new(4);
        ctx.emit_rc(ops[0].clone());
        ctx.emit_rc(ops[1].clone());
        ctx.make_constant_box(&Operand::from_bound_op(&ops[0]), Value::Int(0));

        let mut pass = OptRewrite::new();
        // Process CondCallN -> removed
        for i in 0..ops[2].num_args() {
            let resolved = ctx.resolve_operand_operand(&ops[2].arg(i));
            ops[2].setarg(i, resolved);
        }
        ctx.bind_input_resops(std::slice::from_ref(&ops[2]));
        let result2 = pass.propagate_forward(&ops[2], &ops[2], &mut ctx);
        assert!(matches!(result2, OptimizationResult::Remove));

        // Process GuardNoException -> should also be removed
        let __pf_rc = OpRc::new((*ops[3]).clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result3 = pass.propagate_forward(&ops[3], &__pf_rc, &mut ctx);
        assert!(matches!(result3, OptimizationResult::Remove));
    }

    #[test]
    fn test_guard_no_exception_after_emitted_call() {
        // CallN(...) -> emitted, then GuardNoException -> kept
        let ops = build_specs(&[
            same_i(),                               // op0: func
            op_spec(OpCode::CallN, &[0]),           // op1: call
            op_spec(OpCode::GuardNoException, &[]), // op2: should NOT be removed
        ]);
        let mut ctx = OptContext::new(3);
        ctx.emit_rc(ops[0].clone());

        let mut pass = OptRewrite::new();
        // Process CallN -> PassOn (not handled by OptRewrite)
        ctx.bind_input_resops(std::slice::from_ref(&ops[1]));
        let result1 = pass.propagate_forward(&ops[1], &ops[1], &mut ctx);
        assert!(matches!(result1, OptimizationResult::PassOn));
        ctx.emit_rc(ops[1].clone());

        // Process GuardNoException -> should NOT be removed
        let __pf_rc = OpRc::new((*ops[2]).clone());
        ctx.bind_input_resops(std::slice::from_ref(&__pf_rc));
        let result2 = pass.propagate_forward(&ops[2], &__pf_rc, &mut ctx);
        assert!(matches!(result2, OptimizationResult::PassOn));
    }

    #[test]
    fn test_guard_future_condition_records_and_removes() {
        // rewrite.py: GUARD_FUTURE_CONDITION → record in patchguardop + remove
        let (result, ctx) = run_one(vec![op_spec(OpCode::GuardFutureCondition, &[])], 0, &[]);
        assert_remove(&result);
        assert!(ctx.patchguardop.is_some());
        assert_eq!(
            ctx.patchguardop.unwrap().opcode,
            OpCode::GuardFutureCondition
        );
    }

    #[test]
    fn test_guard_value_to_guard_false() {
        // GUARD_VALUE(v, 0) on a bool-bounded v → GUARD_FALSE(v)
        // (_maybe_replace_guard_value, optimizer.py). The [0,1]
        // bound on v comes from the `int_gt` producer the intbounds pass
        // analyzed; the guard's own make_constant runs in
        // postprocess_GUARD_VALUE (rewrite.py), after emit, so it
        // does not bound v at emit time.
        // Bound oparser graph: i0/i1 are header InputArgs, v = IntGt(i0, i1)
        // a live producer, and GUARD_VALUE's expected operand is the literal
        // ConstInt(0) — so every arg sheds to Operand::{InputArg,Op,Const}.
        use crate::history::test_support::bound_inputarg_operand;
        let (_i0, i0_rc) = bound_inputarg_operand(majit_ir::Type::Int, 0);
        let (_i1, i1_rc) = bound_inputarg_operand(majit_ir::Type::Int, 1);
        // A live producer for v (IntGt result) at int_op(2), held in
        // `int_gt` for the body of the test.
        let int_gt = OpRc::new(Op::new(
            OpCode::IntGt,
            &[
                Operand::from_bound_inputarg(&i0_rc),
                Operand::from_bound_inputarg(&i1_rc),
            ],
        ));
        int_gt.pos().set(OpRef::int_op(2));
        let guard_value = Op::new(
            OpCode::GuardValue,
            &[
                Operand::from_bound_op(&int_gt),
                Operand::const_from_value(Value::Int(0)),
            ],
        );
        let finish = Op::new(OpCode::Finish, &[]);
        let ops = {
            let mut gv = guard_value;
            gv.pos().set(OpRef::void_op(0));
            let mut fin = finish;
            fin.pos().set(OpRef::void_op(1));
            vec![(*int_gt).clone(), gv, fin]
        };
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&[majit_ir::Type::Int; 2]);
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert!(
            result.iter().any(|o| o.opcode == OpCode::GuardFalse),
            "GUARD_VALUE(v, 0) should become GUARD_FALSE(v)"
        );
    }

    #[test]
    fn test_int_mul_neg_one() {
        // x * (-1) → INT_NEG(x)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let x = b.input(majit_ir::Type::Int, 0);
        let neg_one = b.const_int(-1);
        let prod = b.op(OpCode::IntMul, &[x, neg_one]);
        b.op(OpCode::Finish, &[prod]);
        let (ops, inputs) = b.build();

        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        // mul_minus_one lives in OptIntBounds (autogenintrules.py).
        opt.add_pass(Box::new(crate::optimizeopt::intbounds::OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");

        assert!(
            result.iter().any(|o| o.opcode == OpCode::IntNeg),
            "x * (-1) should become INT_NEG(x)"
        );
    }

    #[test]
    fn test_float_mul_neg_one() {
        // x * (-1.0) → FLOAT_NEG(x)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let x = b.input(majit_ir::Type::Float, 0);
        let neg_one = Operand::const_from_value(Value::Float(-1.0));
        let prod = b.op(OpCode::FloatMul, &[x, neg_one]);
        b.op(OpCode::Finish, &[prod]);
        let (ops, inputs) = b.build();

        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_cond_call_n_zero_removes() {
        // COND_CALL_N(0, func, args...) → removed (condition is false)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let cond = Operand::const_from_value(Value::Int(0));
        let func = b.input(majit_ir::Type::Int, 0);
        let arg = b.input(majit_ir::Type::Int, 1);
        b.op(OpCode::CondCallN, &[cond, func, arg]);
        b.op(OpCode::Finish, &[]);
        let (ops, inputs) = b.build();
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::CondCallN),
            "COND_CALL_N(0, ...) should be removed"
        );
    }

    #[test]
    fn test_cond_call_n_nonzero_converts() {
        // COND_CALL_N(1, func, args...) → CALL_N(func, args...)
        let mut b = crate::history::test_support::TraceBuilder::new();
        let cond = Operand::const_from_value(Value::Int(1));
        let func = b.input(majit_ir::Type::Int, 0);
        let arg = b.input(majit_ir::Type::Int, 1);
        b.op(OpCode::CondCallN, &[cond, func, arg]);
        b.op(OpCode::Finish, &[]);
        let (ops, inputs) = b.build();
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.add_pass(Box::new(OptRewrite::new()));
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::default();
        let num_inputs = inputs.len();
        let result = opt
            .optimize_with_constants_and_inputs_oprc(&ops, &mut constants, num_inputs)
            .expect("test: unexpected InvalidLoop");
        assert!(
            result.iter().any(|o| o.opcode == OpCode::CallN),
            "COND_CALL_N(1, ...) should become CALL_N"
        );
    }
}
