/// earlyforce.py: OptEarlyForce — force virtual args before escaping.
///
/// RPython earlyforce.py forces ALL arguments of most operations to ensure
/// virtual objects are materialized before they can escape. Exempt ops:
/// SETFIELD_GC, SETARRAYITEM_GC, SETARRAYITEM_RAW, QUASIIMMUT_FIELD,
/// SAME_AS_*, raw_free calls.
///
/// earlyforce.py: self.optimizer.optearlyforce = self
/// The pass registers itself so force_at_the_end_of_preamble can route
/// forced operations starting from earlyforce.next (= heap).
use majit_ir::{Op, OpCode};

use crate::optimizeopt::info::PtrInfoExt;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

#[derive(Clone, Copy)]
pub struct OptEarlyForce;

/// earlyforce.py: is_raw_free(op, opnum).
/// Raw free calls should not force their arguments.
pub fn is_raw_free(op: &Op, opnum: OpCode) -> bool {
    if !opnum.is_real_call() {
        return false;
    }
    let __descr_arc_descr = op.getdescr();
    if let Some(descr) = __descr_arc_descr.as_ref()
        && let Some(cd) = descr.as_call_descr()
    {
        let ei = cd.get_extra_info();
        return ei.oopspecindex == majit_ir::OopSpecIndex::RawFree;
    }
    false
}

impl OptEarlyForce {
    pub fn new() -> Self {
        OptEarlyForce
    }

    /// earlyforce.py `propagate_forward`: `self.optimizer.force_box(arg, self)`.
    /// `force_box_impl` reaches the owning optimizer (`optforce.optimizer`)
    /// and `emit_extra` runs immediately. This pass already popped
    /// `potential_extra_ops`, so it calls `force_box_impl` rather than
    /// `Optimizer::force_box` (which would pop again). Direct unit tests
    /// pass no optimizer and keep the queue.
    fn propagate_inner(
        &mut self,
        optimizer: &mut Option<&mut crate::optimizeopt::optimizer::Optimizer>,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if Self::should_force_args(op) {
            for i in 0..op.num_args() {
                let arg_opnd = ctx.resolve_operand_operand_opt(&op.arg(i));
                let arg = arg_opnd
                    .as_ref()
                    .map(|o| o.to_opref())
                    .unwrap_or_else(|| op.arg(i).to_opref());
                // optimizer.py: force_box pops the arg's
                // potential_extra_op and hands it to the short-preamble
                // producer BEFORE (and independent of) the virtual check, so a
                // non-virtual short-box arg is added to the preamble too.
                // `take_potential_extra_op` is a no-op when not unrolling or
                // when no extra op is queued for this arg.
                let arg_key = arg_opnd.clone().unwrap_or_else(|| op.arg(i).clone());
                if let Some(tracked) = ctx.take_potential_extra_op(&arg_key) {
                    // shortpreamble.py:434: the resolved Box is handed
                    // to the builder; fall back to the operand itself.
                    let arg_b = arg_opnd
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| op.arg(i).clone());
                    if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                        builder.add_preamble_op_from_pop(&tracked, arg_b);
                    } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                        builder.add_preamble_op_from_pop(&tracked, arg_b);
                    }
                }
                // optimizer.py:363-366: if the arg carries a virtual PtrInfo,
                // force it into the trace. `is_virtual` / `take_ptr_info`
                // chain-walk the operand themselves (get_box_replacement), so
                // the raw arg drives them directly — no box round-trip.
                let arg_is_virtual = ctx.is_virtual(&op.arg(i));
                if arg_is_virtual {
                    // A virtual resolves to a bound alloc op, so the native
                    // resolver yields its terminal operand directly (no box
                    // round-trip); force_box reads the operand's own opref and
                    // drives every make_equal_to / set_ptr_info receiver off it.
                    let arg_op = ctx
                        .resolve_operand_operand_opt(&op.arg(i))
                        .expect("arg_is_virtual implies a resolved box");
                    let mut info = ctx.take_ptr_info(&arg_op).unwrap();
                    let _forced = if let Some(opt) = optimizer.as_deref_mut() {
                        crate::optimizeopt::info::force_box_impl(&mut info, &arg_op, ctx, opt)
                    } else {
                        info.force_box(&arg_op, ctx)
                    };
                }
            }
        }
        // earlyforce.py:29: return self.emit(op)
        OptimizationResult::PassOn
    }

    /// earlyforce.py propagate_forward: should we force args for this op?
    /// RPython exempt set: SETFIELD_GC, SETARRAYITEM_GC, SETARRAYITEM_RAW,
    /// QUASIIMMUT_FIELD, SAME_AS_I/R/F, and raw_free. Note that
    /// SETFIELD_RAW is NOT exempt in RPython.
    pub(crate) fn should_force_args(op: &Op) -> bool {
        !matches!(
            op.opcode,
            OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                | OpCode::QuasiimmutField
                | OpCode::SameAsI
                | OpCode::SameAsR
                | OpCode::SameAsF
        ) && !is_raw_free(op, op.opcode)
    }
}

impl Default for OptEarlyForce {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptEarlyForce {
    /// earlyforce.py: propagate_forward.
    /// Force all virtual args of non-exempt operations, then emit.
    fn propagate_forward(
        &self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        opt: &mut Option<&mut crate::optimizeopt::optimizer::Optimizer>,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let mut pass = *self;
        pass.propagate_inner(opt, op, op_rc, ctx)
    }

    fn name(&self) -> &'static str {
        "earlyforce"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_inputarg_operand;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::OpRef;
    use majit_ir::Type;
    use majit_ir::descr::make_call_descr;
    use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex};

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            let pos = i as u32;
            op.pos().set(OpRef::op_typed(pos, op.result_type()));
        }
    }

    #[test]
    fn test_is_raw_free_matches_call_descr_oopspec() {
        let op = Op::new(OpCode::CallN, &[]);
        op.setdescr(make_call_descr(
            Vec::new(),
            Type::Void,
            EffectInfo::new(ExtraEffect::CanRaise, OopSpecIndex::RawFree),
        ));

        assert!(is_raw_free(&op, op.opcode));
        assert!(!is_raw_free(&op, OpCode::IntAdd));
    }

    #[test]
    fn test_earlyforce_resolves_call_may_force_args() {
        let mut ops = vec![Op::new(
            OpCode::CallMayForceN,
            &[
                rooted_inputarg_operand(Type::Ref, 100),
                rooted_inputarg_operand(Type::Ref, 101),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallMayForceN);
    }

    #[test]
    fn test_earlyforce_passthrough_non_call() {
        let mut ops = vec![Op::new(
            OpCode::IntAdd,
            &[
                rooted_inputarg_operand(Type::Int, 100),
                rooted_inputarg_operand(Type::Int, 101),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_earlyforce_call_assembler_handled() {
        let mut ops = vec![Op::new(
            OpCode::CallAssemblerI,
            &[rooted_inputarg_operand(Type::Ref, 100)],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallAssemblerI);
    }

    #[test]
    fn test_earlyforce_guard_not_forced() {
        // GUARD_NOT_FORCED should have its fail_args resolved.
        let mut guard = Op::new(OpCode::GuardNotForced, &[]);
        guard.setfailargs(Default::default());
        let mut ops = vec![guard];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNotForced);
        assert!(result[0].has_failargs());
    }

    #[test]
    fn test_earlyforce_all_call_may_force_types() {
        for opcode in [
            OpCode::CallMayForceI,
            OpCode::CallMayForceR,
            OpCode::CallMayForceF,
            OpCode::CallMayForceN,
        ] {
            let mut ops = vec![Op::new(opcode, &[rooted_inputarg_operand(Type::Ref, 100)])];
            assign_positions(&mut ops);

            let mut opt = Optimizer::new();
            opt.add_pass(Box::new(OptEarlyForce::new()));
            let result = opt.optimize_with_constants_and_inputs(
                &ops,
                &mut majit_ir::ConstMap::default(),
                1024,
            );
            assert_eq!(result.len(), 1, "{opcode:?} should be handled");
        }
    }

    #[test]
    fn test_earlyforce_exempt_setfield() {
        // SETFIELD_GC should NOT force args (earlyforce.py:18)
        let mut ops = vec![Op::new(
            OpCode::SetfieldGc,
            &[
                rooted_inputarg_operand(Type::Ref, 100),
                rooted_inputarg_operand(Type::Int, 101),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::default(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }
}
