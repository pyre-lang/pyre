//! `assembler.py _build_wb_slowpath`: the slow path of `COND_CALL_GC_WB`
//! and `COND_CALL_GC_WB_ARRAY` `CALL`s a helper shared by every site. It
//! must reach the GC with the pushed object, come back to the
//! card-marking branch, and hand back the live core and float registers.
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use majit_backend::{Backend, JitCellToken};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_gc::GcFlags;
use majit_gc::collector::MiniMarkGC;
use majit_gc::header::{GcHeader, header_of};
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::{GcRef, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};

/// Same card-carrying large array as x86/assembler.rs `CARD_ARRAY_LENGTH`.
const LENGTH: usize = 17024;

fn op(opcode: OpCode, args: &[OpRef], pos: OpRef) -> Op {
    let bx: Vec<_> = args.iter().map(|a| rb(*a)).collect();
    let op = Op::new(opcode, &bx);
    op.pos().set(pos);
    op
}

#[test]
fn cond_call_gc_wb_slowpath_reaches_the_gc_and_keeps_live_registers() {
    let mut gc = MiniMarkGC::new();
    let item_size = std::mem::size_of::<GcRef>();
    let array_tid = gc.register_type(majit_gc::TypeInfo::varsize(
        8,
        item_size,
        0,
        true,
        Vec::new(),
    ));
    let total_size = GcHeader::SIZE + 8 + item_size * LENGTH;
    let mut alloc_old = |gc: &mut MiniMarkGC| {
        let obj = gc.alloc_in_oldgen_with_cards(array_tid, total_size, LENGTH, true);
        unsafe { *(obj.0 as *mut usize) = LENGTH };
        let hdr = unsafe { &*header_of(obj.0) };
        assert!(hdr.has_flag(GcFlags::GCFLAG_HAS_CARDS));
        assert!(hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS));
        assert!(!hdr.has_flag(GcFlags::GCFLAG_CARDS_SET));
        obj
    };
    let plain = alloc_old(&mut gc);
    let array = alloc_old(&mut gc);

    let mut backend = DynasmBackend::new();
    // A collector without a write barrier first: the `setup_once` in
    // `attach_default_test_descrs` builds no helper for it, and they must
    // still be built once MiniMark is in.
    backend.set_gc_allocator(Box::new(majit_backend::jitframe::HostHeapGc));
    backend.attach_default_test_descrs();
    // jitframe.py — a collector with a type table carries JITFRAME before install.
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    backend.set_gc_allocator(Box::new(gc));
    let card_page_shift = majit_gc::WriteBarrierDescr::for_current_gc().jit_wb_card_page_shift;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Float, 2),
    ];
    let index: i64 = 1152;
    for (trace_id, opcode, obj) in [
        (1606, OpCode::CondCallGcWb, plain),
        (1607, OpCode::CondCallGcWbArray, array),
    ] {
        let barrier_args: &[OpRef] = if opcode == OpCode::CondCallGcWbArray {
            &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)]
        } else {
            &[OpRef::input_arg_ref(0)]
        };
        // The sum crosses the barrier in a core register, the float one in
        // a float register; a failing guard hands both back.
        let fail_args = [
            OpRef::float_op(4),
            OpRef::int_op(3),
            OpRef::input_arg_ref(0),
        ];
        let guard = op(OpCode::GuardTrue, &[OpRef::int_op(6)], OpRef::void_op(7));
        guard.set_fail_arg_types(vec![Type::Float, Type::Int, Type::Ref]);
        guard.setfailargs(fail_args.iter().map(|r| rb(*r)).collect::<Vec<_>>().into());
        let finish = op(OpCode::Finish, &[], OpRef::void_op(8));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());
        let ops = vec![
            op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(1), OpRef::input_arg_int(1)],
                OpRef::int_op(3),
            ),
            op(
                OpCode::FloatAdd,
                &[OpRef::input_arg_float(2), OpRef::input_arg_float(2)],
                OpRef::float_op(4),
            ),
            op(opcode, barrier_args, OpRef::void_op(5)),
            op(
                OpCode::IntLt,
                &[OpRef::input_arg_int(1), OpRef::const_int(0)],
                OpRef::int_op(6),
            ),
            guard,
            finish,
        ];
        let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
        let token = JitCellToken::new(trace_id as i64);
        backend.compile_loop(&inputargs, &ops, &token).unwrap();
        let frame = backend.execute_token(
            &token,
            &[Value::Ref(obj), Value::Int(index), Value::Float(1.25)],
        );
        assert!(!backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_float_value(&frame, 0), 2.5);
        assert_eq!(backend.get_int_value(&frame, 1), 2 * index);
        assert_eq!(backend.get_ref_value(&frame, 2), obj);
    }

    let hdr = unsafe { &*header_of(plain.0) };
    assert!(
        !hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
        "remember_young_pointer must have run on the pushed object"
    );
    let hdr = unsafe { &*header_of(array.0) };
    assert!(
        hdr.has_flag(GcFlags::GCFLAG_CARDS_SET),
        "remember_young_pointer_from_array must have run on the pushed array"
    );
    // incminimark.py `get_card`: card bytes sit below the header, the
    // first one at `header - 1`.
    let card = index as usize >> card_page_shift;
    let card_byte = unsafe { *((array.0 - GcHeader::SIZE - 1 - (card >> 3)) as *const u8) };
    assert_eq!(
        card_byte,
        1 << (card & 7),
        "the card-set branch must dirty the index's card"
    );
}
