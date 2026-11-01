use indexmap::IndexMap;
use majit_ir::operand::Operand;
use majit_ir::{InputArgRc, OpRc, OpRef, Type};

/// Rebind every arg/failarg box to this trace's own `InputArgRc`/`OpRc`
/// producers (`runner_test.py` builds args from the producer box object).
pub fn rebind_trace_arg_boxes(inputargs: &[InputArgRc], operations: &[OpRc]) {
    let mut map = IndexMap::<OpRef, Operand>::new();
    for ia in inputargs {
        map.insert(ia.opref(), Operand::from_bound_inputarg(ia));
    }
    for op in operations {
        if op.opcode.result_type() != Type::Void {
            map.insert(op.pos().get(), Operand::from_bound_op(op));
        }
    }
    for op in operations {
        for slot in op.args_slice_mut() {
            if slot.is_constant() || slot.is_none() {
                continue;
            }
            if let Some(mapped) = map.get(&slot.to_opref()) {
                *slot = mapped.clone();
            }
        }
        op.map_failargs_in_place(|slot| {
            if slot.is_constant() || slot.is_none() {
                return;
            }
            if let Some(mapped) = map.get(&slot.to_opref()) {
                *slot = mapped.clone();
            }
        });
    }
}
