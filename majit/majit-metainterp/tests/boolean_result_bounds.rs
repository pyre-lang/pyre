//! Regression coverage for pure.py callbacks and RecentPureOps lookup.
use majit_ir::operand::Operand;
use majit_ir::{ConstMap, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};
use majit_metainterp::optimizeopt::optimizer::Optimizer;
use majit_metainterp::optimizeopt::{OptContext, Optimization, OptimizationResult, pure::OptPure};

fn const_int(value: i64) -> Operand {
    Operand::const_from_value(Value::Int(value))
}

/// An emitted int box standing in for the op whose result the pure cache
/// records.
fn emit_result_box(ctx: &mut OptContext) -> Operand {
    let pos = ctx.emit(Op::new(OpCode::SameAsI, &[const_int(0)]));
    ctx.get_box_replacement_operand_opt(pos).unwrap()
}

#[test]
fn postponed_boolean_result_is_not_retested_for_truth() {
    for opcode in [OpCode::IntIsZero, OpCode::IntIsTrue] {
        let input = InputArg::from_type_rc(Type::Int, 0);
        let comparison = OpRc::new(Op::new(opcode, &[Operand::from_bound_inputarg(&input)]));
        comparison.pos().set(OpRef::int_op(1));
        let truth = OpRc::new(Op::new(
            OpCode::IntIsTrue,
            &[Operand::from_bound_op(&comparison)],
        ));
        truth.pos().set(OpRef::int_op(2));
        let finish = OpRc::new(Op::new(OpCode::Finish, &[Operand::from_bound_op(&truth)]));
        finish.pos().set(OpRef::op_typed(3, Type::Void));
        let mut optimizer = Optimizer::default_pipeline();
        optimizer.trace_inputargs = OpRef::inputarg_refs(&[Type::Int]);
        let mut constants = ConstMap::<Value>::default();
        let optimized = optimizer
            .optimize_with_constants_and_inputs_oprc(
                &[comparison, truth, finish],
                &mut constants,
                1,
            )
            .unwrap();
        assert_eq!(
            optimized.iter().map(|op| op.opcode).collect::<Vec<_>>(),
            [opcode, OpCode::Finish]
        );
        assert_eq!(optimized[1].arg(0).to_opref(), optimized[0].pos().get());
    }
}

#[test]
fn pure_lookup_keeps_forwarding_and_commutative_matching() {
    let mut ctx = OptContext::new(8);
    let mut pure = OptPure::new();
    let x_pos = ctx.emit(Op::new(
        OpCode::SameAsI,
        &[Operand::const_from_value(Value::Int(9))],
    ));
    let x = ctx.get_box_replacement_operand_opt(x_pos).unwrap();
    let alias_pos = ctx.emit(Op::new(OpCode::SameAsI, &[x.clone()]));
    let alias = ctx.get_box_replacement_operand_opt(alias_pos).unwrap();
    ctx.make_equal_to(&alias, &x);
    let result = emit_result_box(&mut ctx);
    pure.pure_from_args2(OpCode::IntAdd, &x, &const_int(5), &result, &mut ctx);
    let query = Op::new(
        OpCode::IntAdd,
        &[Operand::const_from_value(Value::Int(5)), alias],
    );
    assert_eq!(
        pure.get_pure_result(&query, &mut ctx).map(|r| r.to_opref()),
        Some(result.to_opref())
    );
    let different = Op::new(
        OpCode::IntAdd,
        &[Operand::const_from_value(Value::Int(6)), x],
    );
    assert_eq!(pure.get_pure_result(&different, &mut ctx), None);
}

#[test]
fn unchecked_arithmetic_reuses_checked_arithmetic() {
    for (plain, checked) in [
        (OpCode::IntAdd, OpCode::IntAddOvf),
        (OpCode::IntSub, OpCode::IntSubOvf),
        (OpCode::IntMul, OpCode::IntMulOvf),
    ] {
        let mut ctx = OptContext::new(8);
        let mut pure = OptPure::new();
        let x_pos = ctx.emit(Op::new(
            OpCode::SameAsI,
            &[Operand::const_from_value(Value::Int(9))],
        ));
        let x = ctx.get_box_replacement_operand_opt(x_pos).unwrap();
        let result = emit_result_box(&mut ctx);
        pure.pure_from_args2(checked, &x, &const_int(5), &result, &mut ctx);
        let query = Op::new(plain, &[x, const_int(5)]);
        assert_eq!(
            pure.get_pure_result(&query, &mut ctx).map(|r| r.to_opref()),
            Some(result.to_opref())
        );
    }
}

#[test]
fn checked_arithmetic_does_not_reuse_unchecked_arithmetic() {
    for (plain, checked) in [
        (OpCode::IntAdd, OpCode::IntAddOvf),
        (OpCode::IntSub, OpCode::IntSubOvf),
        (OpCode::IntMul, OpCode::IntMulOvf),
    ] {
        let mut ctx = OptContext::new(8);
        let mut pure = OptPure::new();
        let x_pos = ctx.emit(Op::new(
            OpCode::SameAsI,
            &[Operand::const_from_value(Value::Int(9))],
        ));
        let x = ctx.get_box_replacement_operand_opt(x_pos).unwrap();
        let result = emit_result_box(&mut ctx);
        pure.pure_from_args2(plain, &x, &const_int(5), &result, &mut ctx);
        let checked = OpRc::new(Op::new(
            checked,
            &[x, Operand::const_from_value(Value::Int(5))],
        ));
        checked.pos().set(OpRef::int_op(3));
        assert!(matches!(
            pure.propagate_forward(&checked, &checked, &mut None, &mut ctx),
            OptimizationResult::Remove
        ));
        let guard = OpRc::new(Op::new(OpCode::GuardNoOverflow, &[]));
        guard.pos().set(OpRef::op_typed(4, Type::Void));
        assert!(matches!(
            pure.propagate_forward(&guard, &guard, &mut None, &mut ctx),
            OptimizationResult::PassOn
        ));
    }
}
