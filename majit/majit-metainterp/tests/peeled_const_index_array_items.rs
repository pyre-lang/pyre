//! Pins the peeled body of a loop that reads several constant-index items
//! of a loop-invariant array. Every item is exported as a heap short box,
//! so the body label carries them all and the body reloads none.

use majit_ir::descr::{make_array_descr_full, make_field_descr_with_parent, make_size_descr_full};
use majit_ir::operand::Operand;
use majit_ir::{ArrayFlag, ConstMap, GcRef, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};
use majit_metainterp::optimizeopt::unroll::UnrollOptimizer;

fn positioned(opcode: OpCode, args: &[Operand], raw: u32) -> OpRc {
    let op = Op::new(opcode, args);
    op.pos().set(OpRef::op_typed(raw, opcode.result_type()));
    OpRc::new(op)
}

fn with_descr(opcode: OpCode, args: &[Operand], raw: u32, descr: &majit_ir::DescrRef) -> OpRc {
    let op = Op::new(opcode, args);
    op.setdescr(descr.clone());
    op.pos().set(OpRef::op_typed(raw, opcode.result_type()));
    OpRc::new(op)
}

/// `acc += arr[0] + arr[1] + arr[2]` over `i < 100`, where `arr` is either
/// the loop-invariant input itself or a field read off it.
fn optimize_sum_of_three_items(array_is_a_field: bool) -> Vec<OpRc> {
    let acc = InputArg::from_type_rc(Type::Float, 0);
    let index = InputArg::from_type_rc(Type::Int, 1);
    let obj = InputArg::from_type_rc(Type::Ref, 2);
    acc.set_value(Value::Float(0.0));
    index.set_value(Value::Int(0));
    obj.set_value(Value::Ref(GcRef(0x1000)));

    let acc_arg = Operand::from_bound_inputarg(&acc);
    let index_arg = Operand::from_bound_inputarg(&index);
    let obj_arg = Operand::from_bound_inputarg(&obj);

    // A field descr holds its parent weakly, so the parent lives until
    // the optimizer is done.
    let list_descr = make_size_descr_full(0, 16, 7);
    let items_descr = make_field_descr_with_parent(
        8,
        8,
        Type::Ref,
        ArrayFlag::Pointer,
        0,
        "items".to_string(),
        &list_descr,
    );
    let float_array_descr = make_array_descr_full(0, 16, 8, 8, Type::Float);

    let mut ops: Vec<OpRc> = Vec::new();
    let array = if array_is_a_field {
        let storage = with_descr(
            OpCode::GetfieldGcR,
            std::slice::from_ref(&obj_arg),
            3,
            &items_descr,
        );
        ops.push(storage.clone());
        Operand::from_bound_op(&storage)
    } else {
        obj_arg.clone()
    };
    let items: Vec<OpRc> = (0..3)
        .map(|i| {
            with_descr(
                OpCode::GetarrayitemGcF,
                &[array.clone(), Operand::const_from_value(Value::Int(i))],
                4 + i as u32,
                &float_array_descr,
            )
        })
        .collect();
    ops.extend(items.iter().cloned());
    let mut sum = acc_arg;
    for (i, item) in items.iter().enumerate() {
        let add = positioned(
            OpCode::FloatAdd,
            &[sum, Operand::from_bound_op(item)],
            7 + i as u32,
        );
        ops.push(add.clone());
        sum = Operand::from_bound_op(&add);
    }
    let less_than = positioned(
        OpCode::IntLt,
        &[
            index_arg.clone(),
            Operand::const_from_value(Value::Int(100)),
        ],
        10,
    );
    ops.push(less_than.clone());
    let in_range = positioned(OpCode::GuardTrue, &[Operand::from_bound_op(&less_than)], 11);
    in_range.set_rd_resume_position(0);
    ops.push(in_range);
    let next_index = positioned(
        OpCode::IntAdd,
        &[index_arg, Operand::const_from_value(Value::Int(1))],
        12,
    );
    ops.push(next_index.clone());
    // The recorder closes every loop with GUARD_FUTURE_CONDITION; its
    // resume data is what a replayed short-preamble guard patches onto.
    let future = positioned(OpCode::GuardFutureCondition, &[], 13);
    future.set_rd_resume_position(1);
    ops.push(future);
    ops.push(positioned(
        OpCode::Jump,
        &[sum, Operand::from_bound_op(&next_index), obj_arg],
        14,
    ));
    let ops: Vec<Op> = ops.iter().map(|op| (**op).clone()).collect();

    let mut optimizer = UnrollOptimizer::new();
    optimizer.trace_inputargs = vec![
        Operand::from_bound_inputarg(&acc),
        Operand::from_bound_inputarg(&index),
        Operand::from_bound_inputarg(&obj),
    ];
    optimizer.snapshot_boxes = vec![
        Some(majit_metainterp::optimizeopt::SnapshotBoxList::new()),
        Some(majit_metainterp::optimizeopt::SnapshotBoxList::new()),
    ];
    let mut constants: ConstMap<Value> = ConstMap::default();
    let (optimized, _) =
        optimizer.optimize_trace_with_constants_and_inputs(&ops, &mut constants, 3);
    drop(list_descr);
    optimized
}

/// The peeled body's opcodes, after checking that its closing jump carries
/// the body's own sum: a body result renumbered onto a short-preamble replay
/// position must not come back as an unrelated imported item.
fn peeled_body_opcodes(optimized: &[OpRc]) -> Vec<OpCode> {
    let labels = optimized
        .iter()
        .enumerate()
        .filter_map(|(index, op)| (op.opcode == OpCode::Label).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(labels.len(), 2, "expected preamble and peeled-body labels");
    let body = &optimized[labels[1] + 1..];
    let last_sum = body
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::FloatAdd)
        .expect("the body must still add the items")
        .pos()
        .get();
    let jump = body.last().expect("body must end in Jump");
    assert_eq!(jump.opcode, OpCode::Jump);
    assert_eq!(
        jump.arg(0).to_opref(),
        last_sum,
        "the closing jump must carry the body's own sum"
    );
    body.iter().map(|op| op.opcode).collect()
}

#[test]
fn peeled_body_reloads_no_item_of_an_invariant_input_array() {
    let body = peeled_body_opcodes(&optimize_sum_of_three_items(false));
    assert!(
        !body.contains(&OpCode::GetarrayitemGcF),
        "every constant-index item must come from the short preamble: {body:?}"
    );
}

#[test]
fn peeled_body_reloads_no_item_of_an_invariant_field_array() {
    let body = peeled_body_opcodes(&optimize_sum_of_three_items(true));
    assert!(
        !body.contains(&OpCode::GetarrayitemGcF) && !body.contains(&OpCode::GetfieldGcR),
        "the storage field and every constant-index item must come from \
         the short preamble: {body:?}"
    );
}
