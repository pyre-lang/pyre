//! Verification: auto-generated trace functions produce correct IR.

#[cfg(test)]
mod tests {
    use crate::jit::descr::{
        make_field_descr, make_immutable_field_descr, make_size_descr_with_type_and_vtable,
    };
    use majit_ir::{OpCode, OpRef};
    use majit_metainterp::TraceCtx;

    fn intval_descr() -> majit_ir::DescrRef {
        make_field_descr(8, 8, majit_ir::Type::Int, true)
    }
    fn immutable_intval_descr() -> majit_ir::DescrRef {
        make_immutable_field_descr(8, 8, majit_ir::Type::Int, true)
    }
    fn floatval_descr() -> majit_ir::DescrRef {
        make_field_descr(8, 8, majit_ir::Type::Float, false)
    }
    fn immutable_floatval_descr() -> majit_ir::DescrRef {
        make_immutable_field_descr(8, 8, majit_ir::Type::Float, false)
    }
    fn w_int_size_descr() -> majit_ir::DescrRef {
        make_size_descr_with_type_and_vtable(16, 0, 0)
    }
    fn w_float_size_descr() -> majit_ir::DescrRef {
        make_size_descr_with_type_and_vtable(16, 0, 0)
    }
    const FAKE_INT_TYPE: i64 = 0x1234_5678;
    const FAKE_FLOAT_TYPE: i64 = 0x1234_9876;

    /// GuardClass recording allocates the nursery-backed `Trace._ops`
    /// GcArray (`opencoder.py` Trace), so this thread has to be a mutator
    /// with the collector installed.
    fn test_ctx(num_inputs: usize) -> TraceCtx {
        crate::eval::init_gc_subsystem();
        TraceCtx::for_test(num_inputs)
    }

    /// Get ops from TraceCtx, excluding the dummy Finish we add for finalization.
    fn get_ops(ctx: TraceCtx) -> Vec<OpCode> {
        let mut recorder = ctx.into_recorder();
        let dummy_descr = make_size_descr_with_type_and_vtable(0, 0, 0);
        recorder.finish(&[], dummy_descr);
        let trace = recorder.get_trace();
        trace
            .ops
            .iter()
            .map(|o| o.opcode)
            .filter(|op| *op != OpCode::Finish)
            .collect()
    }

    #[test]
    fn test_unbox_int_ops() {
        let mut ctx = test_ctx(1);
        let obj = OpRef::input_arg_ref(0);
        let _intval = crate::trace_unbox_int(&mut ctx, obj, FAKE_INT_TYPE, intval_descr());
        let ops = get_ops(ctx);
        // GUARD_CLASS(box, cls): backend loads typeptr from obj at offset 0
        // (llgraph/runner.py execute_guard_class), no explicit pre-read. Followed by the
        // intval field read.
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcI]);
        eprintln!("✓ trace_unbox_int: {:?}", ops);
    }

    #[test]
    fn test_box_int_ops() {
        let mut ctx = test_ctx(1);
        let _obj = crate::trace_box_int(
            &mut ctx,
            OpRef::input_arg_ref(0),
            w_int_size_descr(),
            intval_descr(),
            FAKE_INT_TYPE,
        );
        let ops = get_ops(ctx);
        // rewrite.py:479-484 GC rewriter: NEW_WITH_VTABLE emits the vtable
        // field via fielddescr_vtable, so the trace records
        // NEW_WITH_VTABLE(size) + only the non-type field setter.
        assert_eq!(ops, vec![OpCode::NewWithVtable, OpCode::SetfieldGc]);
        eprintln!("✓ trace_box_int: {:?}", ops);
    }

    #[test]
    fn test_int_binop_ovf_ops() {
        let mut ctx = test_ctx(2);
        let _result = crate::trace_int_binop_ovf(
            &mut ctx,
            OpRef::input_arg_ref(0),
            OpRef::input_arg_ref(1),
            OpCode::IntAddOvf,
            FAKE_INT_TYPE,
            intval_descr(),
            w_int_size_descr(),
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![
                OpCode::GuardClass,
                OpCode::GetfieldGcI, // unbox a
                OpCode::GuardClass,
                OpCode::GetfieldGcI, // unbox b
                OpCode::IntAddOvf,
                OpCode::GuardNoOverflow, // add + guard
                OpCode::NewWithVtable,
                OpCode::SetfieldGc, // box (NEW_WITH_VTABLE + intval setter)
            ]
        );
        eprintln!("✓ trace_int_binop_ovf: {} ops", ops.len());
    }

    #[test]
    fn test_unbox_float_ops() {
        let mut ctx = test_ctx(1);
        let obj = OpRef::input_arg_ref(0);
        let _floatval = crate::trace_unbox_float(&mut ctx, obj, FAKE_FLOAT_TYPE, floatval_descr());
        let ops = get_ops(ctx);
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcF]);
        eprintln!("✓ trace_unbox_float: {:?}", ops);
    }

    #[test]
    fn test_unbox_int_ops_read_immutable_descrs_as_plain_getfield() {
        let mut ctx = test_ctx(1);
        let obj = OpRef::input_arg_ref(0);
        let _intval =
            crate::trace_unbox_int(&mut ctx, obj, FAKE_INT_TYPE, immutable_intval_descr());
        let ops = get_ops(ctx);
        // The tracer records the plain GetfieldGcI even for an immutable
        // descr; OptHeap re-derives purity from the descr. GuardClass takes
        // the object directly (no pre-read of ob_type).
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcI]);
    }

    #[test]
    fn test_unbox_float_ops_read_immutable_descrs_as_plain_getfield() {
        let mut ctx = test_ctx(1);
        let obj = OpRef::input_arg_ref(0);
        let _floatval =
            crate::trace_unbox_float(&mut ctx, obj, FAKE_FLOAT_TYPE, immutable_floatval_descr());
        let ops = get_ops(ctx);
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcF]);
    }

    #[test]
    fn test_box_float_ops() {
        let mut ctx = test_ctx(1);
        let _obj = crate::trace_box_float(
            &mut ctx,
            OpRef::input_arg_ref(0),
            w_float_size_descr(),
            floatval_descr(),
            FAKE_FLOAT_TYPE,
        );
        let ops = get_ops(ctx);
        assert_eq!(ops, vec![OpCode::NewWithVtable, OpCode::SetfieldGc]);
        eprintln!("✓ trace_box_float: {:?}", ops);
    }

    #[test]
    fn test_float_binop_ops() {
        let mut ctx = test_ctx(2);
        let _result = crate::trace_float_binop(
            &mut ctx,
            OpRef::input_arg_ref(0),
            OpRef::input_arg_ref(1),
            OpCode::FloatAdd,
            FAKE_FLOAT_TYPE,
            floatval_descr(),
            w_float_size_descr(),
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![
                OpCode::GuardClass,
                OpCode::GetfieldGcF, // unbox a
                OpCode::GuardClass,
                OpCode::GetfieldGcF, // unbox b
                OpCode::FloatAdd,
                OpCode::NewWithVtable,
                OpCode::SetfieldGc, // box
            ]
        );
        eprintln!("✓ trace_float_binop: {} ops", ops.len());
    }
}
