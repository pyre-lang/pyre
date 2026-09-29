/// Late-stage simplification pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/simplify.py.
///
/// This is the simplest optimization pass. It runs after the other passes
/// and cleans up operations that are no longer needed:
///
/// 1. Converts CALL_PURE_* to the corresponding CALL_* (purity has already
///    been exploited by OptPure).
/// 2. Converts CALL_LOOPINVARIANT_* to the corresponding CALL_*.
/// 3. Removes hint operations (RECORD_EXACT_CLASS, RECORD_EXACT_VALUE_*,
///    RECORD_KNOWN_RESULT, VIRTUAL_REF_FINISH, QUASIIMMUT_FIELD,
///    ASSERT_NOT_NONE) that were consumed by earlier passes.
/// 4. Rewrites VIRTUAL_REF to SAME_AS_R (the virtualisation is resolved).
use majit_ir::{Op, OpCode};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub struct OptSimplify;

impl OptSimplify {
    pub fn new() -> Self {
        OptSimplify
    }

    /// Convert a CALL_PURE_* or CALL_LOOPINVARIANT_* to the corresponding CALL_*.
    fn rewrite_call(op: &Op) -> Op {
        let new_opcode = OpCode::call_for_type(op.result_type());
        let new_op = op.copy_and_change(new_opcode, None, None);
        new_op.pos().set(op.pos().get());
        new_op
    }
}

impl Default for OptSimplify {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptSimplify {
    fn propagate_forward(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        _ctx: &mut OptContext,
    ) -> OptimizationResult {
        match op.opcode {
            // CALL_PURE_* -> CALL_* (`simplify.py optimize_CALL_PURE_I`)
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                OptimizationResult::Emit(Self::rewrite_call(op))
            }

            // CALL_LOOPINVARIANT_* -> CALL_*
            OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN => OptimizationResult::Emit(Self::rewrite_call(op)),

            // VIRTUAL_REF -> SAME_AS_R (just forward the first arg)
            OpCode::VirtualRefR => {
                let mut new_op = Op::new(OpCode::SameAsR, &[op.arg(0)]);
                new_op.pos().set(op.pos().get());
                OptimizationResult::Emit(new_op)
            }

            // simplify.py: GUARD_FUTURE_CONDITION — record in patchguardop,
            // then remove. Unroll uses patchguardop to attach resume data to
            // extra guards from short preamble.
            OpCode::GuardFutureCondition => {
                _ctx.patchguardop = Some(op.clone());
                OptimizationResult::Remove
            }

            // Hint operations that are simply removed
            OpCode::VirtualRefFinish
            | OpCode::QuasiimmutField
            | OpCode::AssertNotNone
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueR
            | OpCode::RecordExactValueI
            | OpCode::RecordKnownResult => OptimizationResult::Remove,

            _ => OptimizationResult::PassOn,
        }
    }

    fn name(&self) -> &'static str {
        "simplify"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::TraceBuilder;
    use majit_ir::operand::Operand;
    use majit_ir::{OpRc, OpRef, Type};

    /// Seed empty guard snapshots over the canonical `OpRc` slice in place
    /// (each guard's `rd_resume_position` is on GuardExtra, so the assignment is
    /// shared through the `Rc`), mirroring `seed_empty_guard_snapshots` for
    /// the `OpRc`-threaded driver.
    fn seed_oprc(ops: &[OpRc]) -> super::super::SnapshotBoxes {
        let scratch: Vec<Op> = ops.iter().map(|op| (**op).clone()).collect();
        let (seeded, snapshots) = super::super::seed_empty_guard_snapshots(&scratch);
        for (op, seed) in ops.iter().zip(seeded.iter()) {
            op.set_rd_resume_position(seed.rd_resume_position());
        }
        snapshots
    }

    /// Drive `OptSimplify` over a bound oparser graph
    /// (`rpython/jit/tool/oparser.py`): header `InputArg`s and live producer
    /// `OpRc`s built by [`TraceBuilder`], threaded through the canonical
    /// `optimize_with_constants_and_inputs_oprc` entry (`input_ops_from_ops =
    /// true`) so the test's own `OpRc`s are the producers the optimizer
    /// indexes. Every op-arg sheds to `Operand::{InputArg,Op,Const}` — never
    /// the position-only `Operand::Box`.
    fn run_trace(builder: TraceBuilder) -> Vec<Op> {
        let (ops, inputs) = builder.build();
        let mut opt = crate::optimizeopt::optimizer::Optimizer::new();
        opt.add_pass(Box::new(OptSimplify::new()));
        opt.trace_inputargs = OpRef::inputarg_refs(&inputs);
        opt.snapshot_boxes = seed_oprc(&ops);
        let num_inputs = inputs.len();
        opt.optimize_with_constants_and_inputs_oprc(
            &ops,
            &mut majit_ir::ConstMap::default(),
            num_inputs,
        )
        .expect("test: unexpected InvalidLoop")
        .into_iter()
        .map(|rc| (*rc).clone())
        .collect()
    }

    /// The `OpRef`s a header-input slice of `n` `Type::Int` inputargs resolves
    /// to once emitted — the bound-graph counterpart of the old fixtures'
    /// `int_op(0..n)` free positions (`to_opref` of an `InputArg` box).
    fn input_oprefs(n: u32) -> Vec<OpRef> {
        (0..n).map(OpRef::input_arg_int).collect()
    }

    #[test]
    fn test_call_pure_to_call() {
        for (pure_op, expected_op) in [
            (OpCode::CallPureI, OpCode::CallI),
            (OpCode::CallPureR, OpCode::CallR),
            (OpCode::CallPureF, OpCode::CallF),
            (OpCode::CallPureN, OpCode::CallN),
        ] {
            // pure_op(i0, i1) over two header inputargs; Finish keeps
            // the result live so rewrite (not unused-Remove) runs.
            let mut b = TraceBuilder::new();
            let i0 = b.input(Type::Int, 0);
            let i1 = b.input(Type::Int, 1);
            let v = b.op(pure_op, &[i0, i1]);
            b.op(OpCode::Finish, &[v]);
            let result = run_trace(b);
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].opcode, expected_op);
            assert_eq!(
                &result[0]
                    .args_slice()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect::<Vec<_>>()[..],
                &input_oprefs(2)[..]
            );
        }
    }

    #[test]
    fn test_call_loopinvariant_to_call() {
        for (loopinv_op, expected_op) in [
            (OpCode::CallLoopinvariantI, OpCode::CallI),
            (OpCode::CallLoopinvariantR, OpCode::CallR),
            (OpCode::CallLoopinvariantF, OpCode::CallF),
            (OpCode::CallLoopinvariantN, OpCode::CallN),
        ] {
            let mut b = TraceBuilder::new();
            let i0 = b.input(Type::Int, 0);
            b.op(loopinv_op, &[i0]);
            let result = run_trace(b);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].opcode, expected_op);
        }
    }

    #[test]
    fn test_virtual_ref_to_same_as() {
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        b.op(OpCode::VirtualRefR, &[i0, i1]);
        let result = run_trace(b);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SameAsR);
        assert_eq!(
            result[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::input_arg_int(0)]
        );
    }

    #[test]
    fn test_removed_ops() {
        let removed_opcodes = [
            OpCode::VirtualRefFinish,
            OpCode::QuasiimmutField,
            OpCode::AssertNotNone,
            OpCode::RecordExactClass,
            OpCode::RecordExactValueR,
            OpCode::RecordExactValueI,
            OpCode::RecordKnownResult,
        ];
        for opcode in removed_opcodes {
            let arity = opcode.arity().unwrap_or(0) as u32;
            // `arity` header inputargs, then the op consuming them.
            let mut b = TraceBuilder::new();
            let args: Vec<Operand> = (0..arity).map(|i| b.input(Type::Int, i)).collect();
            b.op(opcode, &args);
            let result = run_trace(b);
            assert!(result.is_empty(), "{:?} should be removed", opcode);
        }
    }

    #[test]
    fn test_passthrough() {
        // v = IntAdd(i0, i1), GuardTrue(v).
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        let v = b.op(OpCode::IntAdd, &[i0, i1]);
        b.op(OpCode::GuardTrue, &[v]);
        let result = run_trace(b);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
    }

    #[test]
    fn test_preserves_args_on_call_rewrite() {
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        let i2 = b.input(Type::Int, 2);
        let v = b.op(OpCode::CallPureI, &[i0, i1, i2]);
        b.op(OpCode::Finish, &[v]);
        let result = run_trace(b);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallI);
        assert_eq!(
            result[0]
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            input_oprefs(3)
        );
    }

    #[test]
    fn test_mixed_ops() {
        // IntAdd(i0,i1), CallPureI(i0), RecordExactClass(i0,i1), IntSub(i0,i1).
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        b.op(OpCode::IntAdd, &[i0.clone(), i1.clone()]);
        let call = b.op(OpCode::CallPureI, std::slice::from_ref(&i0));
        b.op(OpCode::RecordExactClass, &[i0.clone(), i1.clone()]);
        b.op(OpCode::IntSub, &[i0, i1]);
        b.op(OpCode::Finish, &[call]);
        let result = run_trace(b);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::CallI);
        assert_eq!(result[2].opcode, OpCode::IntSub);
    }

    #[test]
    fn test_guard_future_condition_removed() {
        // GuardFutureCondition, IntAdd(i0, i1), Finish.
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        b.op(OpCode::GuardFutureCondition, &[]);
        b.op(OpCode::IntAdd, &[i0, i1]);
        b.op(OpCode::Finish, &[]);
        let result = run_trace(b);
        // GUARD_FUTURE_CONDITION should be removed
        assert!(
            !result
                .iter()
                .any(|o| o.opcode == OpCode::GuardFutureCondition)
        );
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_assert_not_none_removed() {
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        b.op(OpCode::AssertNotNone, &[i0]);
        b.op(OpCode::Finish, &[]);
        let result = run_trace(b);
        assert!(!result.iter().any(|o| o.opcode == OpCode::AssertNotNone));
    }

    #[test]
    fn test_virtual_ref_r_to_same_as() {
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        b.op(OpCode::VirtualRefR, &[i0, i1]);
        b.op(OpCode::Finish, &[]);
        let result = run_trace(b);
        assert!(result.iter().any(|o| o.opcode == OpCode::SameAsR));
    }

    #[test]
    fn test_call_pure_rewrites_to_call() {
        // `simplify.py optimize_CALL_PURE_I` always rewrites to CALL_*
        // and emits. Unused results stay; they are not deleted.
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        b.op(OpCode::CallPureR, &[i0]);
        b.op(OpCode::Finish, &[]);
        let result = run_trace(b);
        assert!(result.iter().any(|o| o.opcode == OpCode::CallR));
        assert!(!result.iter().any(|o| o.opcode == OpCode::CallPureR));
    }

    #[test]
    fn test_record_exact_class_removed() {
        let mut b = TraceBuilder::new();
        let i0 = b.input(Type::Int, 0);
        let i1 = b.input(Type::Int, 1);
        b.op(OpCode::RecordExactClass, &[i0, i1]);
        b.op(OpCode::Finish, &[]);
        let result = run_trace(b);
        assert!(!result.iter().any(|o| o.opcode == OpCode::RecordExactClass));
    }
}
