//! Pins the optimized body of a peeled integer loop whose null reference is
//! loop-invariant. The null check belongs in the preamble; the exported
//! virtual state checks every entry to the body label.
//!
//! The trace here is hand-built, so its phase 2 numbers input args from zero
//! and the body reads the same slot the import wrote. That is the case this
//! file covers. A recorded trace shifts phase 2 by `inputarg_base`, and the
//! two halves are rejoined in `OptContext::imported_inputarg_operand` —
//! `a_peeled_body_reads_the_slot_the_import_forwarded` in
//! `optimizeopt::mod` is what holds that, and this test passes with or
//! without it.

use majit_ir::operand::Operand;
use majit_ir::{ConstMap, GcRef, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};
use majit_metainterp::optimizeopt::unroll::UnrollOptimizer;

fn positioned(opcode: OpCode, args: &[Operand], raw: u32) -> Op {
    let op = Op::new(opcode, args);
    op.pos().set(OpRef::op_typed(raw, opcode.result_type()));
    op
}

#[test]
fn peeled_integer_loop_drops_the_loop_invariant_null_guard() {
    let acc = InputArg::from_type_rc(Type::Int, 0);
    let index = InputArg::from_type_rc(Type::Int, 1);
    let null = InputArg::from_type_rc(Type::Ref, 2);
    acc.set_value(Value::Int(0));
    index.set_value(Value::Int(0));
    null.set_value(Value::Ref(GcRef(0)));

    let acc_arg = Operand::from_bound_inputarg(&acc);
    let index_arg = Operand::from_bound_inputarg(&index);
    let null_arg = Operand::from_bound_inputarg(&null);

    let is_null = positioned(OpCode::GuardIsnull, std::slice::from_ref(&null_arg), 3);
    is_null.set_rd_resume_position(0);
    let less_than = OpRc::new(positioned(
        OpCode::IntLt,
        &[
            index_arg.clone(),
            Operand::const_from_value(Value::Int(100)),
        ],
        4,
    ));
    let in_range = positioned(OpCode::GuardTrue, &[Operand::from_bound_op(&less_than)], 5);
    in_range.set_rd_resume_position(1);
    let next_acc = OpRc::new(positioned(OpCode::IntAdd, &[acc_arg, index_arg.clone()], 6));
    let next_index = OpRc::new(positioned(
        OpCode::IntAdd,
        &[index_arg, Operand::const_from_value(Value::Int(1))],
        7,
    ));
    let jump = positioned(
        OpCode::Jump,
        &[
            Operand::from_bound_op(&next_acc),
            Operand::from_bound_op(&next_index),
            null_arg,
        ],
        8,
    );
    let ops = vec![
        is_null,
        (*less_than).clone(),
        in_range,
        (*next_acc).clone(),
        (*next_index).clone(),
        jump,
    ];

    let mut optimizer = UnrollOptimizer::new();
    optimizer.trace_inputargs = vec![
        Operand::from_bound_inputarg(&acc),
        Operand::from_bound_inputarg(&index),
        Operand::from_bound_inputarg(&null),
    ];
    optimizer.snapshot_boxes = vec![
        Some(majit_metainterp::optimizeopt::SnapshotBoxList::new()),
        Some(majit_metainterp::optimizeopt::SnapshotBoxList::new()),
    ];
    let mut constants: ConstMap<Value> = ConstMap::default();
    let (optimized, _) =
        optimizer.optimize_trace_with_constants_and_inputs(&ops, &mut constants, 3);

    let labels = optimized
        .iter()
        .enumerate()
        .filter_map(|(index, op)| (op.opcode == OpCode::Label).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(labels.len(), 2, "expected preamble and peeled-body labels");
    assert_eq!(
        optimized[..labels[1]]
            .iter()
            .filter(|op| op.opcode == OpCode::GuardIsnull)
            .count(),
        1,
        "the preamble must still check the null specialization"
    );
    assert_eq!(
        optimized[labels[1]].num_args(),
        2,
        "the body label must carry only the accumulator and index"
    );
    let body = optimized[labels[1] + 1..]
        .iter()
        .map(|op| op.opcode)
        .collect::<Vec<_>>();
    assert_eq!(
        body,
        vec![
            OpCode::IntLt,
            OpCode::GuardTrue,
            OpCode::IntAdd,
            OpCode::IntAdd,
            OpCode::Jump,
        ],
        "the peeled body must trust the null constant in its exported virtual state"
    );
    assert_eq!(
        optimized.last().expect("body must end in Jump").num_args(),
        2,
        "the closing jump must not carry the exported null constant"
    );
}
