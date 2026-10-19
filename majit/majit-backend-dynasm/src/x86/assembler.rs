/// x86/assembler.py: Assembler386 — x86_64 JIT code generation backend.
///
/// RPython: Assembler386(BaseAssembler, VectorAssemblerMixin)
/// in x86/assembler.py.
///
/// Key methods:
///   assemble_loop — assembler.py:501
///   assemble_bridge — assembler.py:623
///   _assemble — assembler.py:779 (walk ops + emit code)
///   patch_jump_for_descr — assembler.py:965
///   redirect_call_assembler — assembler.py:1138
use crate::regloc::ebp_loc_pat;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::sync::Arc;

// x86/assembler.py parity: x86_64-only backend.
/// `codebuf.py MachineCodeBlockWrapper`: code is assembled into a plain byte
/// vector (every relocation is PC-relative) and copied into the arena block
/// at `materialize`; no per-trace executable mapping.
pub(crate) type Assembler = dynasmrt::VecAssembler<dynasmrt::x64::X64Relocation>;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, dynasm};

use super::regloc::{LocationCodeBuilder, Operand, addr_add, addr_add_const, heap, imm};
use super::rx86;

/// `rx86.py X86_64_CodeBuilder.MULTIBYTE_NOPs`, index == length, 1..=15.
fn multibyte_nop(len: usize) -> &'static [u8] {
    const NOPS: [&[u8]; 16] = [
        &[],
        &[0x90],
        &[0x66, 0x90],
        &[0x0f, 0x1f, 0x00],
        &[0x0f, 0x1f, 0x40, 0x00],
        &[0x0f, 0x1f, 0x44, 0x00, 0x00],
        &[0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00],
        &[0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00],
        &[0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        &[0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        &[0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        &[
            0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
        &[
            0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
        &[
            0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
        &[
            0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
        &[
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00,
            0x00,
        ],
    ];
    NOPS[len]
}

use majit_backend::{
    AsmMemoryBlock, AsmMemoryManager, BackendError, JitCellToken, MachineDataBlockWrapper,
};
use majit_ir::{
    FailDescr, FailDescrStore, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, OpTypeIndex,
    TargetArgLoc, Type,
};

use crate::arch::*;
use crate::codebuf;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::jitframe::{
    FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FRAME_OFS, JF_GCMAP_OFS,
    JF_GUARD_EXC_OFS,
};
use crate::jump::RegallocMoves;
use crate::regalloc::{RegAlloc, RegAllocAssembler};
use crate::regloc::Loc;
use crate::runner::GuardGcTypeInfo;

/// x86/assembler.py: managed general-purpose registers.
const X86_GEN_REGS: [crate::regloc::RegLoc; 16] = [
    crate::regloc::RegLoc::new(0, false),
    crate::regloc::RegLoc::new(1, false),
    crate::regloc::RegLoc::new(2, false),
    crate::regloc::RegLoc::new(3, false),
    crate::regloc::RegLoc::new(4, false),
    crate::regloc::RegLoc::new(5, false),
    crate::regloc::RegLoc::new(6, false),
    crate::regloc::RegLoc::new(7, false),
    crate::regloc::RegLoc::new(8, false),
    crate::regloc::RegLoc::new(9, false),
    crate::regloc::RegLoc::new(10, false),
    crate::regloc::RegLoc::new(11, false),
    crate::regloc::RegLoc::new(12, false),
    crate::regloc::RegLoc::new(13, false),
    crate::regloc::RegLoc::new(19, false),
    crate::regloc::RegLoc::new(20, false),
];

/// x86/assembler.py: managed XMM/float registers.
const X86_FLOAT_REGS: [crate::regloc::RegLoc; 8] = [
    crate::regloc::RegLoc::new(0, true),
    crate::regloc::RegLoc::new(1, true),
    crate::regloc::RegLoc::new(2, true),
    crate::regloc::RegLoc::new(3, true),
    crate::regloc::RegLoc::new(4, true),
    crate::regloc::RegLoc::new(5, true),
    crate::regloc::RegLoc::new(6, true),
    crate::regloc::RegLoc::new(7, true),
];

/// A LABEL publishes an in-buffer offset and only becomes an absolute address
/// in `fixup_target_tokens`; 0 means its target was never compiled.  The first
/// page is never mapped, so any value below it is one of those two and can only
/// be a wild branch.  Dynasm bakes the immediate at codegen, so it cannot be
/// repaired later.
const MIN_RELOCATED_JUMP_TARGET: usize = 4096;

/// Resolved argument: either a frame slot (frame-pointer-relative offset) or a constant.
enum ResolvedArg {
    /// Frame-pointer-relative byte offset: [rbp + offset] on x64, [x29, #offset] on aarch64.
    Slot(i32),
    /// Immediate constant value.
    Const(i64),
}

#[derive(Clone, Copy)]
struct CallAssemblerTargetAddr {
    immediate: Option<usize>,
}

impl CallAssemblerTargetAddr {
    fn is_available(self) -> bool {
        self.immediate.is_some()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use majit_backend::{Backend, JitCellToken};
    use majit_ir::forwarding::bound_operand_from_opref;
    use majit_ir::operand::Operand;
    use majit_ir::{
        GcRef, InputArg, Op, OpCode, OpRc, OpRef, Type, Value, make_array_descr_signed,
    };

    use crate::regloc::{EDI, Loc, R10};
    use crate::runner::DynasmBackend;
    use dynasmrt::dynasm;

    /// x86/regalloc.py `consider_call_malloc_nursery` binds the result with
    /// `force_allocate_reg(op, selected_reg=ecx)`; only FrameManager may spill
    /// it later.  Storing it into a slot here as well grew every recursive
    /// loop's JitFrame by one slot per allocation, which changes how much the
    /// process allocates and so shifts the minor-collection schedule the
    /// jitcounter decay is driven by.
    #[test]
    fn malloc_nursery_result_does_not_grow_frame_depth() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        let malloc = OpRc::new(Op::new(
            OpCode::CallMallocNursery,
            &[Operand::from_opref(OpRef::const_int(32))],
        ));
        malloc.pos().set(OpRef::ref_op(0));

        let finish = Op::new(OpCode::Finish, &[Operand::from_bound_op(&malloc)]);
        finish.pos().set(OpRef::void_op(1));
        finish.set_fail_arg_types(vec![Type::Ref]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(517);
        backend
            .compile_loop(&[], &[malloc, OpRc::new(finish)], &token)
            .expect("compile register-resident nursery result trace");

        let compiled = token
            .compiled
            .get()
            .expect("compiled code")
            .downcast_ref::<super::CompiledCode>()
            .expect("dynasm compiled code");
        assert_eq!(
            compiled.frame_depth.load(Ordering::Acquire),
            super::JITFRAME_FIXED_SIZE,
            "a register-resident nursery result must not allocate a shadow frame slot",
        );
    }

    fn compile_eval_breaker_poll_trace(
        trace_id: u64,
        word_addr: usize,
    ) -> (DynasmBackend, JitCellToken) {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let mut token = JitCellToken::new(trace_id as i64);

        let word = OpRc::new(Op::with_descr(
            OpCode::RawLoadI,
            &[
                Operand::from_opref(OpRef::const_int(word_addr as i64)),
                Operand::from_opref(OpRef::const_int(0)),
            ],
            make_array_descr_signed(0, 8, Type::Int, true),
        ));
        word.pos().set(OpRef::int_op(0));
        let armed = OpRc::new(Op::new(OpCode::IntIsTrue, &[Operand::from_bound_op(&word)]));
        armed.pos().set(OpRef::int_op(1));
        let guard = Op::new(OpCode::GuardFalse, &[Operand::from_bound_op(&armed)]);
        guard.pos().set(OpRef::void_op(2));
        guard.set_fail_arg_types(vec![]);
        guard.setfailargs(vec![].into());
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(3));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        backend
            .compile_loop(
                &[],
                &[word, armed, OpRc::new(guard), OpRc::new(finish)],
                &mut token,
            )
            .expect("compile eval-breaker poll IR");
        (backend, token)
    }

    #[test]
    fn eval_breaker_poll_deopts_when_bitmask_set() {
        let test_word = AtomicUsize::new(0);
        let word_addr = &test_word as *const AtomicUsize as usize;

        let (backend, token) = compile_eval_breaker_poll_trace(518, word_addr);
        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());

        test_word.fetch_or(majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);
        let frame = backend.execute_token(&token, &[]);
        assert!(!backend.get_latest_descr(&frame).is_finish());
        test_word.fetch_and(!majit_ir::eval_breaker_word::EB_STW, Ordering::Relaxed);

        let (backend_after_clear, token_after_clear) =
            compile_eval_breaker_poll_trace(519, word_addr);
        let frame = backend_after_clear.execute_token(&token_after_clear, &[]);
        assert!(backend_after_clear.get_latest_descr(&frame).is_finish());

        test_word.fetch_or(majit_ir::eval_breaker_word::EB_ASYNC, Ordering::Relaxed);
        let frame = backend_after_clear.execute_token(&token_after_clear, &[]);
        assert!(!backend_after_clear.get_latest_descr(&frame).is_finish());
    }

    #[test]
    fn eval_breaker_poll_cross_trace_uses_same_word() {
        let test_word = AtomicUsize::new(0);
        let word_addr = &test_word as *const AtomicUsize as usize;

        let (backend_a, token_a) = compile_eval_breaker_poll_trace(520, word_addr);
        test_word.fetch_or(majit_ir::eval_breaker_word::EB_ASYNC, Ordering::Relaxed);
        let (backend_b, token_b) = compile_eval_breaker_poll_trace(521, word_addr);

        let frame_a = backend_a.execute_token(&token_a, &[]);
        assert!(!backend_a.get_latest_descr(&frame_a).is_finish());
        let frame_b = backend_b.execute_token(&token_b, &[]);
        assert!(!backend_b.get_latest_descr(&frame_b).is_finish());
    }

    /// `X86XMMRegisterManager.convert_to_imm` parks the bits in the machine
    /// data block (`ConstFloatLoc`) and `MOVSD`s that address. The eight
    /// bytes of the constant are not in the instruction stream.
    #[test]
    fn float_constant_loads_from_literal_pool() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let inputargs = vec![majit_ir::InputArg::new_float_rc(0)];
        let add = OpRc::new(Op::new(
            OpCode::FloatAdd,
            &[
                majit_ir::forwarding::bound_operand_from_opref(OpRef::input_arg_float(0)),
                Operand::from_opref(OpRef::const_float(1.5)),
            ],
        ));
        add.pos().set(OpRef::float_op(1));
        let finish = Op::new(OpCode::Finish, &[Operand::from_bound_op(&add)]);
        finish.pos().set(OpRef::void_op(2));
        finish.set_fail_arg_types(vec![Type::Float]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(522);
        backend
            .compile_loop(&inputargs, &[add, OpRc::new(finish)], &token)
            .expect("compile float-add constant");

        let frame = backend.execute_token(&token, &[majit_ir::Value::Float(2.0)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(
            backend.get_float_value(&frame, 0).to_bits(),
            3.5f64.to_bits()
        );

        let compiled = token
            .compiled
            .get()
            .expect("compiled code")
            .downcast_ref::<super::CompiledCode>()
            .expect("dynasm compiled code");
        let code = unsafe {
            std::slice::from_raw_parts(
                compiled.buffer.ptr(dynasmrt::AssemblyOffset(0)),
                compiled.buffer.len(),
            )
        };
        let bits = 1.5f64.to_le_bytes();
        assert!(
            !code.windows(8).any(|window| window == bits),
            "float bits must not sit in the code stream"
        );
        let mut pool_addr = None;
        for block in &compiled.data_blocks {
            let bytes = unsafe { std::slice::from_raw_parts(block.ptr(), block.len()) };
            if let Some(pos) = bytes.windows(8).position(|window| window == bits) {
                pool_addr = Some(block.ptr() as usize + pos);
            }
        }
        let pool_addr = pool_addr.expect("float bits must live in the machine data block");
        // `Assembler386.mov` consumes that `ConstFloatLoc`: `MOVSD_xj` when
        // the address fits disp32, otherwise `_addr_as_reg_offset` plus
        // `MOVSD_xm`. `_addr_as_reg_offset` either loads the address with
        // `MOV_ri` or reuses an r11 value set earlier (`MOV_ri r11, base`)
        // with the difference as disp32.
        let addr_i = pool_addr as i64;
        let referenced = if addr_i == addr_i as i32 as i64 {
            let disp = (addr_i as i32).to_le_bytes();
            code.windows(4).any(|window| window == disp)
        } else {
            code.windows(10).enumerate().any(|(pos, window)| {
                // `MOV_ri r11, imm64`: REX.W+B `0x49`, `0xBB`.
                if window[..2] != [0x49, 0xbb] {
                    return false;
                }
                let base = i64::from_le_bytes(window[2..].try_into().unwrap());
                let offset = addr_i.wrapping_sub(base);
                if offset == 0 {
                    return true;
                }
                offset == offset as i32 as i64
                    && code[pos + 10..]
                        .windows(4)
                        .any(|disp| disp == (offset as i32).to_le_bytes())
            })
        };
        assert!(
            referenced,
            "mov must reference the convert_to_imm address {pool_addr:#x}"
        );
    }

    /// SysV: eight XMM argument registers, then the stack. The ninth float
    /// constant is a `ConstFloatLoc` spilled by `regalloc_immedmem2mem`
    /// (two `MOV32_si`). The first eight reach xmm0–xmm7 via `MOVSD`.
    #[test]
    fn float_const_stack_call_returns_ninth() {
        extern "C" fn ninth(
            a: f64,
            b: f64,
            c: f64,
            d: f64,
            e: f64,
            f: f64,
            g: f64,
            h: f64,
            i: f64,
        ) -> f64 {
            let _ = (a, b, c, d, e, f, g, h);
            i
        }

        let mut effect = majit_ir::EffectInfo::new(
            majit_ir::ExtraEffect::CannotRaise,
            majit_ir::OopSpecIndex::None,
        );
        effect.can_collect = false;

        let mut args = Vec::with_capacity(10);
        args.push(Operand::from_opref(OpRef::const_int(
            ninth as *const () as i64,
        )));
        for i in 0..8 {
            args.push(Operand::from_opref(OpRef::const_float((i as f64) + 10.0)));
        }
        args.push(Operand::from_opref(OpRef::const_float(1.5)));

        let call = OpRc::new(Op::new(OpCode::CallF, &args));
        call.pos().set(OpRef::float_op(0));
        call.setdescr(majit_ir::make_call_descr(
            vec![Type::Float; 9],
            Type::Float,
            effect,
        ));
        let finish = Op::new(OpCode::Finish, &[Operand::from_bound_op(&call)]);
        finish.pos().set(OpRef::void_op(1));
        finish.set_fail_arg_types(vec![Type::Float]);
        finish.setfailargs(vec![].into());

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let token = JitCellToken::new(523);
        backend
            .compile_loop(&[], &[call, OpRc::new(finish)], &token)
            .expect("compile float-const stack call");
        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(
            backend.get_float_value(&frame, 0).to_bits(),
            1.5f64.to_bits()
        );
    }

    /// `RegAlloc.make_sure_var_in_reg` returns `FloatImmedLoc` for a
    /// `ConstFloat`, and `save_into_mem` writes those bits. The base is
    /// the trace `InputArg` (`Operand::from_bound_inputarg`); a
    /// position-only `InputArgRef` is not an operand the recorder emits.
    #[test]
    fn gc_store_float_constant_writes_bits() {
        let mut buf = [0u8; 8];
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let inputargs = vec![majit_ir::InputArg::new_ref_rc(0)];
        let store = OpRc::new(Op::new(
            OpCode::GcStore,
            &[
                Operand::from_bound_inputarg(&inputargs[0]),
                Operand::from_opref(OpRef::const_int(0)),
                Operand::from_opref(OpRef::const_float(1.5)),
                Operand::from_opref(OpRef::const_int(8)),
            ],
        ));
        store.pos().set(OpRef::void_op(1));
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(2));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(524);
        backend
            .compile_loop(&inputargs, &[store, OpRc::new(finish)], &token)
            .expect("compile gc_store of a float constant");
        let frame = backend.execute_token(
            &token,
            &[majit_ir::Value::Ref(majit_ir::GcRef(
                buf.as_mut_ptr() as usize
            ))],
        );
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(f64::from_le_bytes(buf).to_bits(), 1.5f64.to_bits());
    }

    /// `compute_vars_longevity` asserts a fail arg is not a `Const`, and
    /// `Renamer.start_renaming` refuses to place one there. The fail arg
    /// is the `SameAsF` box that received the constant; `locs_for_fail`
    /// records that box and `get_float_value` reads it back.
    #[test]
    fn guard_fail_arg_float_box() {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        let same = OpRc::new(Op::new(
            OpCode::SameAsF,
            &[Operand::from_opref(OpRef::const_float(1.5))],
        ));
        same.pos().set(OpRef::float_op(0));
        let armed = OpRc::new(Op::new(
            OpCode::IntIsTrue,
            &[Operand::from_opref(OpRef::const_int(0))],
        ));
        armed.pos().set(OpRef::int_op(1));
        let guard = Op::new(OpCode::GuardTrue, &[Operand::from_bound_op(&armed)]);
        guard.pos().set(OpRef::void_op(2));
        guard.set_fail_arg_types(vec![Type::Float]);
        guard.setfailargs(vec![Operand::from_bound_op(&same)].into());
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(3));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(525);
        backend
            .compile_loop(
                &[],
                &[same, armed, OpRc::new(guard), OpRc::new(finish)],
                &token,
            )
            .expect("compile guard with a float-box fail arg");
        let frame = backend.execute_token(&token, &[]);
        assert!(!backend.get_latest_descr(&frame).is_finish());
        assert_eq!(
            backend.get_float_value(&frame, 0).to_bits(),
            1.5f64.to_bits()
        );
    }

    // ── COND_CALL_GC_WB_ARRAY inline card marking ──────────────────────

    /// Array length that satisfies both clauses of the card question that a
    /// fixture controls (incminimark.py:1017-1019). Same fixture as
    /// aarch64/assembler.rs `CARD_ARRAY_LENGTH`.
    const CARD_ARRAY_LENGTH: usize = 17024;

    fn alloc_old_card_array(gc: &mut majit_gc::collector::MiniMarkGC, type_id: u32) -> GcRef {
        let item_size = std::mem::size_of::<GcRef>();
        let total_size = majit_gc::header::GcHeader::SIZE + 8 + item_size * CARD_ARRAY_LENGTH;
        assert!(
            total_size >= majit_gc::GcAllocator::max_nursery_object_size(gc),
            "CARD_ARRAY_LENGTH no longer describes a large object, so it gets no cards"
        );
        let obj = gc.alloc_in_oldgen_with_cards(type_id, total_size, CARD_ARRAY_LENGTH, true);
        assert!(
            unsafe {
                (*majit_gc::header::header_of(obj.0)).has_flag(majit_gc::GcFlags::GCFLAG_HAS_CARDS)
            },
            "the fixture array must carry cards"
        );
        unsafe { *(obj.0 as *mut usize) = CARD_ARRAY_LENGTH };
        obj
    }

    fn run_cond_call_gc_wb_array(trace_id: u64, obj: GcRef, index: i64, index_in_register: bool) {
        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();

        let mut inputargs = vec![InputArg::new_ref_rc(0)];
        let mut values = vec![Value::Ref(obj)];
        let index_operand = if index_in_register {
            inputargs.push(InputArg::new_int_rc(1));
            values.push(Value::Int(index));
            bound_operand_from_opref(OpRef::input_arg_int(1))
        } else {
            bound_operand_from_opref(OpRef::const_int(index))
        };

        let barrier = Op::new(
            OpCode::CondCallGcWbArray,
            &[
                bound_operand_from_opref(OpRef::input_arg_ref(0)),
                index_operand,
            ],
        );
        barrier.pos().set(OpRef::void_op(2));

        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(3));
        finish.set_fail_arg_types(vec![]);
        finish.setfailargs(vec![].into());

        let token = JitCellToken::new(trace_id as i64);
        backend
            .compile_loop(&inputargs, &[OpRc::new(barrier), OpRc::new(finish)], &token)
            .expect("compile COND_CALL_GC_WB_ARRAY trace");
        let frame = backend.execute_token(&token, &values);
        assert!(
            backend.get_latest_descr(&frame).is_finish(),
            "the barrier trace must run to its FINISH"
        );
    }

    /// WriteBarrierSlowPath register arm: `SHR; XOR -8; BTS [header]`.
    ///
    /// r10 is in `ALL_CORE_REGS`. The previous byte-OR sequence copied
    /// `loc_base` into r10 and then reloaded `loc_index` from r10, so an
    /// index that already lived there became the array pointer. The
    /// scratch-r11 BTS sequence must leave r10 untouched after the copy.
    #[test]
    fn wb_array_card_mark_reg_index_in_r10_matches_bts_sequence() {
        let page_shift = majit_gc::collector::DEFAULT_CARD_PAGE_SHIFT;
        let mut got = super::Assembler::new(0);
        super::encode_wb_array_card_mark(&mut got, EDI.value, &Loc::Reg(R10), page_shift);
        let got = got.finalize().unwrap();

        // The previous byte-OR arm began `push r10` / `mov r10, loc_base`.
        assert!(
            !got.windows(2).any(|window| window == [0x41, 0x52]),
            "must not push r10; the BTS arm uses scratch r11"
        );
        assert!(
            got.windows(2).any(|window| window == [0x0F, 0xAB]),
            "WriteBarrierSlowPath emits BTS [header], tmp"
        );

        let mut mov = super::Assembler::new(0);
        dynasm!(mov ; .arch x64 ; mov r11, r10);
        let mov = mov.finalize().unwrap();
        assert!(
            got.starts_with(&mov),
            "index in r10 must be copied into scratch r11 before SHR/XOR/BTS"
        );
    }

    /// opassembler.py:996-1015 / x86/assembler.py:2382-2386 inline card marking.
    ///
    /// The register arm shifts the index at runtime; the immediate arm folds
    /// the same two quantities at assembly time. Both must dirty exactly the
    /// card `mark_card` (incminimark.py:1574-1598) would dirty.
    #[test]
    fn cond_call_gc_wb_array_immed_index_marks_same_card_as_reg_index() {
        let wb = crate::runner::dynasm_write_barrier_descr()
            .expect("a write barrier descriptor must be resolvable");
        assert_ne!(
            wb.jit_wb_cards_set, 0,
            "card marking must be enabled for this test to exercise the card arms"
        );
        let card_page_shift = wb.jit_wb_card_page_shift;

        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let item_size = std::mem::size_of::<GcRef>();
        let type_id = gc.register_type(majit_gc::TypeInfo::varsize(
            8,
            item_size,
            0,
            true,
            Vec::new(),
        ));
        let obj_immed = alloc_old_card_array(&mut gc, type_id);
        let obj_reg = alloc_old_card_array(&mut gc, type_id);
        let obj_interp = alloc_old_card_array(&mut gc, type_id);

        for obj in [obj_immed, obj_reg] {
            unsafe {
                (*majit_gc::header::header_of(obj.0)).set_flag(majit_gc::GcFlags::GCFLAG_CARDS_SET);
            }
        }
        for obj in [obj_immed, obj_reg, obj_interp] {
            assert!(
                gc.dirty_cards(obj).is_empty(),
                "a freshly allocated card array starts with every card clean"
            );
        }

        const INDICES: [i64; 5] = [0, 5, 200, 1152, 2047];
        for (n, &index) in INDICES.iter().enumerate() {
            let trace_id = 9200 + 2 * n as u64;
            run_cond_call_gc_wb_array(trace_id, obj_immed, index, false);
            run_cond_call_gc_wb_array(trace_id + 1, obj_reg, index, true);
            gc.do_write_barrier_card(obj_interp, index as usize, card_page_shift);
            assert_eq!(
                gc.dirty_cards(obj_immed),
                gc.dirty_cards(obj_reg),
                "index {index} must dirty the same cards through both arms"
            );
        }

        let mut expected: Vec<usize> = INDICES
            .iter()
            .map(|&index| (index as usize) >> card_page_shift)
            .collect();
        expected.sort_unstable();
        expected.dedup();

        let immed_cards = gc.dirty_cards(obj_immed);
        let interp_cards = gc.dirty_cards(obj_interp);
        assert_eq!(
            immed_cards, interp_cards,
            "the compiled card bits must match remember_young_pointer_from_array2"
        );
        assert_eq!(
            immed_cards, expected,
            "each index must dirty exactly its own card, and nothing else"
        );
    }

    /// `push_gcmap(store=True)` is `MOV(raw_stack(ofs), imm(gcmap))`: a
    /// 64-bit gcmap goes through `_load_scratch`, so a repeated gcmap reuses
    /// r11 and a nearby one is `LEA r11, [r11+d]`.
    #[test]
    fn push_gcmap_loads_the_gcmap_through_the_scratch_cache() {
        let stats = Arc::new(majit_backend::AsmMemoryManagerStats::default());
        let arena = majit_backend::AsmMemoryManager::new_isolated(stats);
        let inputargs: Vec<super::InputArgRc> = Vec::new();
        let operations: Vec<OpRc> = Vec::new();
        let mut asm = super::Assembler386::new(
            arena,
            0,
            0,
            Default::default(),
            None,
            None,
            Default::default(),
            None,
            Default::default(),
            crate::guard::AttachedDescrPtrs::default(),
            Arc::new(crate::guard::CpuDescrCell::default()),
            0,
            0,
            0,
            0,
            0,
            [0; 5],
            0,
            0,
            0,
            [0; 4],
            [0; 4],
            &inputargs,
            &operations,
        );
        let gcmap = 0x1234_5678_9abc_usize;
        let store = {
            let ofs = super::JF_GCMAP_OFS;
            assert!((-128..128).contains(&ofs));
            [0x4C, 0x89, 0x5D, ofs as u8]
        };
        let mut expected = vec![0x49, 0xBB];
        expected.extend_from_slice(&(gcmap as u64).to_le_bytes());
        expected.extend_from_slice(&store);
        expected.extend_from_slice(&store);
        expected.extend_from_slice(&[0x4D, 0x8D, 0x5B, 0x10]);
        expected.extend_from_slice(&store);
        expected.extend_from_slice(&[0x48, 0xC7, 0x45, store[3], 0x00, 0x10, 0x00, 0x00]);
        asm.push_gcmap(gcmap as *mut usize);
        asm.push_gcmap(gcmap as *mut usize);
        asm.push_gcmap((gcmap + 16) as *mut usize);
        asm.push_gcmap(0x1000 as *mut usize);
        let code = asm.mc.take().finalize().expect("finalize push_gcmap bytes");
        let entry = asm.self_entry_addr_ptr;
        drop(asm);
        drop(unsafe { Box::from_raw(entry) });
        assert_eq!(code, expected);
    }
}

#[derive(Clone, Copy)]
enum AbiArgPlacement {
    Gpr(u8),
    Stack(i32),
}

/// `x86/assembler.py _push_all_regs_to_frame` parity — free-fn
/// variant that emits into an arbitrary `dynasmrt::x64::Assembler`,
/// for use by helper-buffer builders that operate outside of an
/// `Assembler386` instance (e.g. `_build_malloc_slowpath`,
/// `_build_wb_slowpath`).  Logic is identical to
/// `Assembler386::push_all_regs_to_jitframe`; the per-arch slot table
/// and `FIRST_ITEM_OFFSET`-relative addressing match.
pub(crate) fn push_all_regs_to_jitframe_raw(
    asm: &mut Assembler,
    ignored_regs: &[crate::regloc::RegLoc],
    withfloats: bool,
    callee_only: bool,
) {
    let regs = if callee_only {
        crate::x86::regalloc::SAVE_AROUND_CALL_CORE_REGS
    } else {
        crate::x86::regalloc::ALL_CORE_REGS
    };
    for reg in regs.iter() {
        if ignored_regs.contains(reg) {
            continue;
        }
        let slot = core_reg_position(*reg).expect("push_all_regs: managed x86_64 GPR");
        let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
        rx86::mov_br(asm, ofs, reg.value);
    }
    if withfloats {
        for reg in crate::x86::regalloc::ALL_FLOAT_REGS.iter() {
            let slot = float_reg_position(*reg).expect("push_all_regs: managed x86_64 XMM");
            let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
            rx86::movsd_bx(asm, ofs, reg.value);
        }
    }
}

/// `x86/assembler.py _pop_all_regs_from_frame` parity — free-fn
/// variant; see `push_all_regs_to_jitframe_raw` for usage notes.
pub(crate) fn pop_all_regs_from_jitframe_raw(
    asm: &mut Assembler,
    ignored_regs: &[crate::regloc::RegLoc],
    withfloats: bool,
    callee_only: bool,
) {
    let regs = if callee_only {
        crate::x86::regalloc::SAVE_AROUND_CALL_CORE_REGS
    } else {
        crate::x86::regalloc::ALL_CORE_REGS
    };
    for reg in regs.iter() {
        if ignored_regs.contains(reg) {
            continue;
        }
        let slot = core_reg_position(*reg).expect("pop_all_regs: managed x86_64 GPR");
        let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
        rx86::mov_rb(asm, reg.value, ofs);
    }
    if withfloats {
        for reg in crate::x86::regalloc::ALL_FLOAT_REGS.iter() {
            let slot = float_reg_position(*reg).expect("pop_all_regs: managed x86_64 XMM");
            let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
            rx86::movsd_xb(asm, reg.value, ofs);
        }
    }
}

/// `x86/assembler.py _call_footer_shadowstack` parity (free-fn
/// variant for use outside of an `Assembler386` borrow — used by
/// `emit_call_footer_raw`, which the standalone propagate / malloc
/// slowpath trampolines reach for).  Subtracts `2 * WORD` from the
/// shadow-stack top, undoing the `gen_shadowstack_header` push that
/// the trace prologue emitted. `ebx` is restored by the footer after it.
pub(crate) fn emit_footer_shadowstack_raw(asm: &mut Assembler) {
    let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    if rx86::fits_in_32bits(rst) {
        rx86::sub_ji8(asm, rst as i32, (WORD * 2) as i32); // SUB [rootstacktop], WORD * 2
    } else {
        rx86::mov_ri(asm, rx86::EBX, rst); // MOV ebx, rootstacktop
        rx86::sub_mi(asm, (rx86::EBX, 0), (WORD * 2) as i32); // SUB [ebx], WORD * 2
    }
}

/// `CPU_X86_64.CALLEE_SAVE_REGISTERS`. Win64 index 4/5 are not contiguous:
/// `_call_footer` reads them from `SHADOWSTORE2_OFS` / `SHADOWSTORE3_OFS`.
fn callee_save_regs() -> &'static [u8] {
    #[cfg(target_os = "windows")]
    {
        &[rx86::EBX, rx86::ESI, rx86::EDI, rx86::R12, 14, 15]
    }
    #[cfg(not(target_os = "windows"))]
    {
        &[rx86::EBX, rx86::R12, 13, 14, 15]
    }
}

fn callee_save_ofs(index: usize) -> i32 {
    #[cfg(target_os = "windows")]
    {
        match index {
            4 => SHADOWSTORE2_OFS as i32,
            5 => SHADOWSTORE3_OFS as i32,
            _ => ((PASS_ON_MY_FRAME + index + 1) * WORD) as i32,
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        ((PASS_ON_MY_FRAME + index + 1) * WORD) as i32
    }
}

/// `_call_header`'s callee-save stores, after `ebp` and the thread-local
/// word are already in the frame.
fn emit_save_callee_saves(asm: &mut Assembler) {
    for (index, reg) in callee_save_regs().iter().copied().enumerate() {
        rx86::mov_sr(asm, callee_save_ofs(index), reg);
    }
}

/// `_call_footer` restore: callee-saves in reverse, then `ebp`, then
/// `ADD rsp, FRAME_FIXED_SIZE * WORD`. Does not emit `ret`.
fn emit_restore_callee_saves(asm: &mut Assembler) {
    let regs = callee_save_regs();
    for index in (0..regs.len()).rev() {
        rx86::mov_rs(asm, regs[index], callee_save_ofs(index));
    }
    rx86::mov_rs(asm, rx86::EBP, (PASS_ON_MY_FRAME * WORD) as i32);
    rx86::add_ri(asm, rx86::ESP, (FRAME_FIXED_SIZE * WORD) as i32);
}

/// `x86/assembler.py _call_footer` parity (free-fn variant).
/// Restores the callee-save set established by `_call_header` and
/// returns the jitframe pointer in `rax`.  Must be entered with `rsp`
/// at trace-body alignment (i.e. the same value the trace's
/// `_call_header` left after its `SUB rsp, FRAME_FIXED_SIZE * WORD`).
pub(crate) fn emit_call_footer_raw(asm: &mut Assembler) {
    // the return value is the jitframe
    dynasm!(asm ; .arch x64 ; mov rax, rbp);
    emit_footer_shadowstack_raw(asm);
    emit_restore_callee_saves(asm);
    dynasm!(asm ; .arch x64 ; ret);
}

/// `_build_failure_recovery(exc, withfloats)`.
///
/// Entered by `jmp`, not `call`, with the two words `generate_quick_failure`
/// pushed still on the stack (`jf_descr` underneath `jf_gcmap`). Saves every
/// managed GPR and, only when `withfloats`, the XMMs. On `exc`, loads
/// `pos_exc_value` into `ebx` (already saved), clears `pos_exception` then
/// `pos_exc_value`, and stores `ebx` to `jf_guard_exc`. Pops the two words
/// into `jf_gcmap` and `jf_descr`, then `_call_footer`. Index in
/// `failure_recovery_code` is `exc + 2 * withfloats`.
fn emit_failure_recovery(mc: &mut LocationCodeBuilder, exc: bool, withfloats: bool) {
    push_all_regs_to_jitframe_raw(mc, &[], withfloats, false);
    if exc {
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        // We might have an exception pending.  Load it into ebx...
        mc.MOV(crate::regloc::EBX, heap(exc_value_addr));
        mc.MOV(heap(exc_type_addr), imm(0));
        mc.MOV(heap(exc_value_addr), imm(0));
        // save ebx into 'jf_guard_exc'
        rx86::mov_br(mc, JF_GUARD_EXC_OFS, rx86::EBX);
    }
    // Top of stack is the gcmap, then the descr (`generate_quick_failure`).
    rx86::pop_b(mc, JF_GCMAP_OFS);
    rx86::pop_b(mc, JF_DESCR_OFS);
    emit_call_footer_raw(mc);
}

/// `setup_once`'s four `_build_failure_recovery` entries:
/// `(exc, withfloats)` = `(False, False)`, `(True, False)`,
/// `(False, True)`, `(True, True)`.
pub(crate) fn build_failure_recoveries(
    arena: &Arc<AsmMemoryManager>,
) -> (Vec<codebuf::ArenaExecutableBuffer>, [usize; 4]) {
    let mut buffers = Vec::with_capacity(4);
    let mut addrs = [0usize; 4];
    for (index, (exc, withfloats)) in [(false, false), (true, false), (false, true), (true, true)]
        .into_iter()
        .enumerate()
    {
        let mut mc = LocationCodeBuilder::new();
        emit_failure_recovery(&mut mc, exc, withfloats);
        let buffer = codebuf::finalize_executable(mc.into_inner(), arena)
            .expect("failure_recovery: finalize");
        let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
        assert!(ptr != 0, "failure_recovery_code[{index}] entry is null");
        addrs[index] = ptr;
        buffers.push(buffer);
    }
    (buffers, addrs)
}

/// `assembler.py:328 _build_propagate_exception_path` parity —
/// pure builder.  Caching/ownership is the caller's responsibility:
/// `X86CpuExt::ensure_propagate_exception_path` (`x86/cpu_ext.rs`)
/// stores the resulting address in its `propagate_exception_path`
/// field, matching PyPy's `self.propagate_exception_path` attribute
/// on `Assembler386`.
///
/// **Calling convention (matches PyPy line 328-345):**
/// - Entry: reached only via `JMP` (not `CALL`) from a slowpath that
///   has already restored `rsp` to the trace body's alignment level
///   (i.e. the same value the trace's `_call_header` left after its
///   SUB).  `rbp` = jitframe (possibly reloaded after a GC move).
///   No other register conventions are assumed: every callee-save
///   was already restored by the slowpath's `_pop_all_regs_from_frame`,
///   and the live trace-body values were spilled to the jitframe.
/// - Exit: `_call_footer` semantics — restores callee-save GPRs
///   from `[rsp+...]`, sets `rax = rbp` (jitframe pointer return),
///   adds the prologue's SUB back, and `RET`s to the function that
///   originally entered the trace (PyPy: the C JIT shim;
///   pyre: the same role via `Asm::_call_header`/`_call_footer`).
///
/// `propagate_exception_descr` is read from `cpu_handle` and baked
/// as an i64 immediate into `[rbp + JF_DESCR_OFS]`.  Caller must
/// guarantee the descr is installed (`MetaInterp::finish_setup`,
/// `pyjitpl.py`) before invoking this builder; the build asserts
/// otherwise.  PyPy itself would silently bake 0 in this case (and
/// fail at `handle_fail` dispatch time); pyre prefers a build-time
/// fail-fast for the same invariant.
///
/// Returns `(buffer, entry_addr)`. The caller (`X86CpuExt`) owns the arena
/// block for the lifetime of the per-CPU stash, matching the helper blocks
/// rooted by PyPy's `asmmemmgr`.
pub(crate) fn build_propagate_exception_path(
    cpu_handle: &crate::guard::CpuDescrHandle,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    let mut asm = LocationCodeBuilder::new();
    let propagate_descr = cpu_handle.read().descr_ptrs().propagate_exception_descr as i64;
    assert!(
        propagate_descr != 0,
        "build_propagate_exception_path: cpu_handle.propagate_exception_descr \
         must be installed (pyjitpl.py:2283) before the trampoline is built \
         on this CPU"
    );
    // `_store_and_reset_exception(self.mc, eax)`. On real OOM
    // pos_exc_value is typically already NULL — the propagate descr's
    // `handle_fail` raises MemoryError directly.
    store_and_reset_exception(&mut asm, Some(crate::regloc::EAX), None, None);
    rx86::mov_br(&mut asm, JF_GUARD_EXC_OFS, rx86::EAX);
    asm.MOV(Operand::B(JF_DESCR_OFS), imm(propagate_descr));
    // assembler.py:342 `self._call_footer()` — restore callee-save,
    // set rax = rbp, ADD rsp, prologue_size, RET.
    emit_call_footer_raw(&mut asm);
    let buffer = codebuf::finalize_executable(asm.into_inner(), arena)
        .expect("propagate_exception_path: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `assembler.py _store_and_reset_exception(mc, excvalloc, exctploc,
/// tmploc)`: move `pos_exc_value` into `excvalloc`, or through `tmploc` into
/// `jf_guard_exc`, and `pos_exception` into `exctploc`; then clear both.
fn store_and_reset_exception(
    mc: &mut LocationCodeBuilder,
    excvalloc: Option<crate::regloc::RegLoc>,
    exctploc: Option<crate::regloc::RegLoc>,
    tmploc: Option<crate::regloc::RegLoc>,
) {
    let exc_value_addr = crate::jit_exc_value_addr() as i64;
    let exc_type_addr = crate::jit_exc_type_addr() as i64;
    if tmploc.is_none() {
        assert!(excvalloc.is_some());
    }
    if let Some(excvalloc) = excvalloc {
        assert!(!excvalloc.is_xmm);
        mc.MOV(excvalloc, heap(exc_value_addr));
    } else if let Some(tmploc) = tmploc {
        // if both are None, just ignore
        mc.MOV(tmploc, heap(exc_value_addr));
        mc.MOV(Operand::B(JF_GUARD_EXC_OFS), tmploc);
    }
    if let Some(exctploc) = exctploc {
        assert!(!exctploc.is_xmm);
        mc.MOV(exctploc, heap(exc_type_addr));
    }
    mc.MOV(heap(exc_type_addr), imm(0));
    mc.MOV(heap(exc_value_addr), imm(0));
}

/// `assembler.py _restore_exception(mc, excvalloc, exctploc, tmploc)`.
fn restore_exception(
    mc: &mut LocationCodeBuilder,
    excvalloc: Option<crate::regloc::RegLoc>,
    exctploc: crate::regloc::RegLoc,
    tmploc: Option<crate::regloc::RegLoc>,
) {
    let exc_value_addr = crate::jit_exc_value_addr() as i64;
    let exc_type_addr = crate::jit_exc_type_addr() as i64;
    if let Some(excvalloc) = excvalloc {
        mc.MOV(heap(exc_value_addr), excvalloc);
    } else {
        let tmploc = tmploc.expect("_restore_exception: tmploc is not None");
        mc.MOV(tmploc, Operand::B(JF_GUARD_EXC_OFS));
        rx86::mov_bi(mc, JF_GUARD_EXC_OFS, 0);
        mc.MOV(heap(exc_value_addr), tmploc);
    }
    mc.MOV(heap(exc_type_addr), exctploc);
}
/// `build_frame_realloc_slowpath`. The caller has executed
/// `IncreaseStackSlowPath.generate_body`: depth at `[rsp + WORD]` (the
/// `PASS_ON_MY_FRAME` scratch) and `push_gcmap` already done. `CALL` then
/// pushes the return address, so the depth is at `[rsp + WORD*2]`.
pub(crate) fn build_frame_realloc_slowpath(
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    let mut asm = LocationCodeBuilder::new();
    push_all_regs_to_jitframe_raw(&mut asm, &[], true, false);
    #[cfg(target_os = "windows")]
    let (arg0, arg1, tmpreg, align) = (rx86::ECX, rx86::EDX, crate::regloc::R12, 40);
    #[cfg(not(target_os = "windows"))]
    let (arg0, arg1, tmpreg, align) = (rx86::EDI, rx86::ESI, crate::regloc::ECX, 8);
    // `MOV_rs(ARG1, WORD*2)` then `MOV ARG0, ebp`.
    rx86::mov_rs(&mut asm, arg1, (WORD * 2) as i32);
    dynasm!(asm ; .arch x64 ; mov Rq(arg0), rbp);
    rx86::sub_ri(&mut asm, rx86::ESP, align);
    let ebx = crate::regloc::EBX;
    store_and_reset_exception(&mut asm, None, Some(ebx), Some(tmpreg));
    asm.CALL(imm(crate::runner::dynasm_realloc_frame as *const () as i64));
    dynasm!(asm ; .arch x64 ; mov rbp, rax);
    restore_exception(&mut asm, None, ebx, Some(crate::regloc::ECX));
    rx86::add_ri(&mut asm, rx86::ESP, align);
    // `_load_shadowstack_top_in_ebx` then `MOV [ebx - WORD], eax`.
    // `eax` still holds the new frame; restore ran before this store.
    let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    asm.MOV(ebx, heap(rst));
    rx86::mov_mr(&mut asm, (rx86::EBX, -(WORD as i32)), rx86::EAX);
    rx86::mov_bi(&mut asm, JF_GCMAP_OFS, 0);
    pop_all_regs_from_jitframe_raw(&mut asm, &[], true, false);
    dynasm!(asm ; .arch x64 ; ret);
    let buffer = codebuf::finalize_executable(asm.into_inner(), arena)
        .expect("frame_realloc_slowpath: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `_build_stack_check_slowpath`. The registered function is
/// `extern "C" fn(current: usize) -> u8`; this helper passes the entry
/// `rsp` and then tests `pos_exception`, not `al`.
pub(crate) fn build_stack_check_slowpath(
    slowpath_addr: usize,
    propagate_exception_path: usize,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    assert!(slowpath_addr != 0, "stack_check_slowpath address is 0");
    assert!(
        propagate_exception_path != 0,
        "build_stack_check_slowpath: propagate_exception_path is 0"
    );
    let mut asm = LocationCodeBuilder::new();
    #[cfg(target_os = "windows")]
    let (arg0, align) = (rx86::ECX, 40);
    #[cfg(not(target_os = "windows"))]
    let (arg0, align) = (rx86::EDI, 8);
    // `MOV ARG0, esp` before the alignment `sub`.
    dynasm!(asm ; .arch x64 ; mov Rq(arg0), rsp);
    rx86::sub_ri(&mut asm, rx86::ESP, align);
    asm.CALL(imm(slowpath_addr as i64));
    rx86::add_ri(&mut asm, rx86::ESP, align);
    asm.MOV(crate::regloc::EAX, heap(crate::jit_exc_type_addr() as i64));
    let overflow = asm.new_dynamic_label();
    dynasm!(asm
        ; .arch x64
        ; test rax, rax
        ; jnz =>overflow
        ; ret
        ; =>overflow
    );
    // From now on this function is basically "merged" with its caller:
    // drop my own return address, then `JMP(imm(propagate_exception_path))`
    // through r11, since the two helpers are separate arena blocks.
    rx86::add_ri(&mut asm, rx86::ESP, WORD as i32);
    asm.JMP(imm(propagate_exception_path as i64));
    let buffer = codebuf::finalize_executable(asm.into_inner(), arena)
        .expect("stack_check_slowpath: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `_build_cond_call_slowpath` for `(supports_floats, callee_only)` pairs
/// `(False, False)`, `(False, True)`, `(True, False)`, `(True, True)`.
///
/// The helper runs on a fresh `pending_slowpaths`: `reload_frame_if_necessary`
/// enqueues a frame `WriteBarrierSlowPath`, and `flush_pending_slowpaths`
/// emits that body into this helper, not into the trace. `wb_slowpath` must
/// already be built (`ensure_wb_slowpath` before this).
pub(crate) fn build_cond_call_slowpaths(
    wb_slowpath: [usize; 5],
    arena: &Arc<AsmMemoryManager>,
) -> (Vec<codebuf::ArenaExecutableBuffer>, [usize; 4]) {
    let inputargs: Vec<InputArgRc> = Vec::new();
    let operations: Vec<OpRc> = Vec::new();
    let constants: majit_ir::ConstMap<majit_ir::Const> = Default::default();
    let classptr_to_typeid: IndexMap<i64, u32> = IndexMap::new();
    let classptr_to_subclass_range: IndexMap<i64, (i64, i64)> = IndexMap::new();
    let mut asm = Assembler386::new(
        Arc::clone(arena),
        0,
        0,
        constants,
        None,
        None,
        classptr_to_typeid,
        None,
        classptr_to_subclass_range,
        crate::guard::AttachedDescrPtrs::default(),
        // `CpuDescrCell::new` leaks the attachments box. One cell per process:
        // `ensure_cond_call_slowpath` builds this assembler once.
        Arc::new(crate::guard::CpuDescrCell::default()),
        0,
        0,
        0,
        0,
        0,
        wb_slowpath,
        0,
        0,
        0,
        [0; 4],
        // This helper assembler never emits `generate_quick_failure`.
        [0; 4],
        &inputargs,
        &operations,
    );
    let mut buffers = Vec::with_capacity(4);
    let mut addrs = [0usize; 4];
    for (index, (supports_floats, callee_only)) in
        [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
    {
        asm.emit_cond_call_helper_variant(supports_floats, callee_only);
        let mc = asm.mc.take();
        let buffer = codebuf::finalize_executable(mc, arena).expect("cond_call_slowpath: finalize");
        let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
        assert!(ptr != 0, "cond_call_slowpath[{index}] entry is null");
        addrs[index] = ptr;
        buffers.push(buffer);
    }
    let entry = asm.self_entry_addr_ptr;
    drop(asm);
    // `Assembler386::new` leaks this box. The helper assembler never reads
    // it back, and it has no `Drop` that does.
    drop(unsafe { Box::from_raw(entry) });
    (buffers, addrs)
}

/// `assembler.py _build_wb_slowpath(withcards, withfloats, for_frame)` —
/// pure builder. Caching/ownership is the caller's responsibility:
/// `X86CpuExt::ensure_wb_slowpath` stores the entry in `wb_slowpath`.
///
/// The helper is called from the slow path of a write barrier. It saves
/// the registers the GC function may clobber, calls it and restores them.
/// The `for_frame=false` variants take the object as an argument pushed
/// just before the `CALL` and return with `RET 8`; the `withcards`
/// variants end with the `TEST8` of `GCFLAG_CARDS_SET` the caller's `JNS`
/// reads. The `for_frame` variant takes `rbp`.
///
/// Returns `None` where upstream returns without building anything: a GC
/// without card marking has no `withcards` helper.
pub(crate) fn build_wb_slowpath(
    withcards: bool,
    withfloats: bool,
    for_frame: bool,
    arena: &Arc<AsmMemoryManager>,
) -> Option<(codebuf::ArenaExecutableBuffer, usize)> {
    let descr = crate::runner::dynasm_write_barrier_descr()?;
    let func = if !withcards {
        // `descr.get_write_barrier_fn(cpu)`. The frame takes the guarded
        // entry, an ordinary store the one `gc.py get_write_barrier_fn`
        // names.
        if for_frame {
            crate::runner::dynasm_write_barrier as *const () as i64
        } else {
            crate::runner::dynasm_jit_remember_young_pointer as *const () as i64
        }
    } else {
        if descr.jit_wb_cards_set == 0 {
            return None;
        }
        crate::runner::dynasm_write_barrier_from_array as *const () as i64
    };
    let mut mc = LocationCodeBuilder::new();
    let word = WORD as i32;
    // win64: 4 extra unused words before CALL
    let shadow_save: i32 = if cfg!(target_os = "windows") {
        4 * word
    } else {
        0
    };
    // `callbuilder.CallBuilder64.ARG0`.
    #[cfg(target_os = "windows")]
    let arg0 = rx86::ECX;
    #[cfg(not(target_os = "windows"))]
    let arg0 = rx86::EDI;
    let (exc0, exc1) = (crate::regloc::EBX, crate::regloc::R12);
    let add_to_esp;
    if !for_frame {
        push_all_regs_to_jitframe_raw(&mut mc, &[], withfloats, true);
        add_to_esp = shadow_save;
        if add_to_esp != 0 {
            // the 4-words shadow store
            rx86::sub_ri(&mut mc, rx86::ESP, add_to_esp);
        }
        rx86::mov_rs(&mut mc, arg0, word + add_to_esp);
    } else {
        // Don't save registers on the jitframe here: it might override
        // already-saved values that will be restored later.
        //
        // This version is called after a CALL. The registers the call
        // destroyed are dead and the callee-saved ones survive the helper,
        // so it saves only eax and xmm0 (possible results of the call) and
        // the two callee-saved registers that carry the exception from the
        // CALL across the helper.
        assert!(!withcards);
        // we have one word to align
        add_to_esp = shadow_save + 7 * word;
        rx86::sub_ri(&mut mc, rx86::ESP, add_to_esp);
        rx86::mov_sr(&mut mc, shadow_save + word, rx86::EAX);
        rx86::movsd_sx(&mut mc, shadow_save + 2 * word, 0);
        dynasm!(mc ; .arch x64 ; mov Rq(arg0), rbp);
        rx86::mov_sr(&mut mc, shadow_save + 5 * word, exc0.value);
        rx86::mov_sr(&mut mc, shadow_save + 6 * word, exc1.value);
        store_and_reset_exception(&mut mc, Some(exc0), Some(exc1), None);
    }

    mc.CALL(imm(func));

    if withcards {
        // A final TEST8 before the RET, for the caller. Careful to not
        // follow this instruction with another one that changes the status
        // of the CPU flags!
        rx86::mov_rs(&mut mc, rx86::EAX, word + add_to_esp);
        rx86::test8_mi(&mut mc, (rx86::EAX, descr.jit_wb_if_flag_byteofs), -0x80);
    }

    if !for_frame {
        if add_to_esp != 0 {
            // ADD touches CPU flags
            rx86::lea_rs(&mut mc, rx86::ESP, add_to_esp);
        }
        pop_all_regs_from_jitframe_raw(&mut mc, &[], withfloats, true);
        // `RET16_i(WORD)` pops the pushed argument.
        dynasm!(mc ; .arch x64 ; ret 8);
    } else {
        rx86::movsd_xs(&mut mc, 0, shadow_save + 2 * word);
        rx86::mov_rs(&mut mc, rx86::EAX, shadow_save + word);
        restore_exception(&mut mc, Some(exc0), exc1, None);
        rx86::mov_rs(&mut mc, exc0.value, shadow_save + 5 * word);
        rx86::mov_rs(&mut mc, exc1.value, shadow_save + 6 * word);
        rx86::lea_rs(&mut mc, rx86::ESP, add_to_esp);
        dynasm!(mc ; .arch x64 ; ret);
    }

    let buffer =
        codebuf::finalize_executable(mc.into_inner(), arena).expect("wb_slowpath: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    Some((buffer, ptr))
}

/// `assembler.py:231 _build_malloc_slowpath(kind='fixed')` parity —
/// pure builder.  Caching/ownership is the caller's responsibility:
/// `X86CpuExt::ensure_malloc_slowpath_fixed` (`x86/cpu_ext.rs`)
/// stores the resulting address in its `malloc_slowpath_fixed`
/// field, matching PyPy's `self.malloc_slowpath` attribute on
/// `Assembler386`.
///
/// **Calling convention (matches PyPy line 233-243):**
/// - Entry: `rbp` = jitframe, `rcx` = old `nursery_head`,
///   `rdx` = `nursery_head + total_size` (set by the per-call-site
///   fast path's `lea rdx, [rcx + total_size]`).  `total_size`
///   recovered at runtime via `sub rdx, rcx`.  The caller has already
///   pushed the gcmap to `[rbp + JF_GCMAP_OFS]`.
/// - Exit: `rcx` = payload pointer (or 0 on OOM), matching PyPy line
///   304 `MOV_rr(ecx, eax)`; `rbp` reloaded from shadow stack top in
///   case a minor GC fired and moved the frame; every other GPR/XMM
///   restored from the jitframe save area, including `rax` which is
///   reset to its pre-call value.
///
/// `ecx` and `edx` are the only registers excluded from
/// push_all_regs / pop_all_regs: `ecx` carries the return value
/// (assembler.py `_push_all_regs_to_frame(mc, [ecx, edx], floats)`),
/// `edx` is the runtime size carrier and the caller's regalloc has
/// already spilled any live value out of both before the call via
/// `MALLOC_NURSERY_CLOBBER` (regalloc.rs).
///
/// On OOM (`rax == 0` after the helper call) the trampoline tail-jumps
/// to `propagate_path` — the standalone `_build_propagate_exception_path`
/// trampoline returned by `build_propagate_exception_path`.  Caller
/// (`X86CpuExt::ensure_malloc_slowpath_fixed`) builds the propagate
/// path first and threads its address here, matching PyPy's
/// `setup_once` ordering.
///
/// Returns `(buffer, entry_addr)`.  Same ownership rule as
/// `build_propagate_exception_path` — the caller (`X86CpuExt`) holds
/// the arena buffer so the RX page lives as long as the CPU.
pub(crate) fn build_malloc_slowpath_fixed(
    cpu_handle: &crate::guard::CpuDescrHandle,
    propagate_path: usize,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    // `cpu_handle` is still threaded through for symmetry with PyPy
    // (where `_build_malloc_slowpath` reads several `self.cpu`-rooted
    // attrs).  The propagate descr itself is now baked once into the
    // standalone propagate trampoline, not re-baked here.
    let _ = cpu_handle;
    let mut asm = LocationCodeBuilder::new();

    // assembler.py:264 `SUB_rr(edx, ecx)` — recover total_size at
    // runtime.  Both `malloc_cond` and `malloc_cond_varsize_frame`
    // hand off `ecx = old_nursery_free` / `edx = old_nursery_free +
    // total_bytes`, so the same `SUB` recovers the byte count for
    // either caller (PyPy line 2554 / 2578 both route through this
    // shared `malloc_slowpath`).
    dynasm!(asm ; .arch x64 ; sub rdx, rcx);

    let slowpath_fn = crate::runner::dynasm_nursery_slowpath as *const () as i64;
    build_malloc_slowpath_body(
        &mut asm,
        slowpath_fn,
        propagate_path,
        MallocSlowpathKind::Fixed,
    );

    let buffer =
        codebuf::finalize_executable(asm.into_inner(), arena).expect("malloc_slowpath: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

pub(crate) fn build_malloc_slowpath_headerless(
    cpu_handle: &crate::guard::CpuDescrHandle,
    propagate_path: usize,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    let _ = cpu_handle;
    let mut asm = LocationCodeBuilder::new();

    dynasm!(asm ; .arch x64 ; sub rdx, rcx);

    let slowpath_fn = crate::runner::dynasm_nursery_slowpath_headerless as *const () as i64;
    build_malloc_slowpath_body(
        &mut asm,
        slowpath_fn,
        propagate_path,
        MallocSlowpathKind::Fixed,
    );

    let buffer = codebuf::finalize_executable(asm.into_inner(), arena)
        .expect("malloc_slowpath_headerless: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `_build_malloc_slowpath(kind)` for `'var'`, `'str'` and `'unicode'`.
/// Entry is `malloc_cond_varsize`'s contract: length in `edx`, tid in
/// `ecx`, and for `'var'` the itemsize at `[rsp+WORD]` on the trace.
pub(crate) fn build_malloc_slowpath_varsize(
    cpu_handle: &crate::guard::CpuDescrHandle,
    kind: MallocSlowpathKind,
    propagate_path: usize,
    arena: &Arc<AsmMemoryManager>,
) -> (codebuf::ArenaExecutableBuffer, usize) {
    let _ = cpu_handle;
    let mut asm = LocationCodeBuilder::new();
    // `get_malloc_slowpath_array_addr()` / `get_malloc_fn_addr('malloc_str')`
    // / `get_malloc_fn_addr('malloc_unicode')`.
    let slowpath_fn = match kind {
        MallocSlowpathKind::Var => crate::runner::dynasm_malloc_array as *const () as i64,
        MallocSlowpathKind::Str => crate::runner::dynasm_malloc_str as *const () as i64,
        MallocSlowpathKind::Unicode => crate::runner::dynasm_malloc_unicode as *const () as i64,
        MallocSlowpathKind::Fixed => {
            panic!("build_malloc_slowpath_varsize: 'fixed' is build_malloc_slowpath_fixed")
        }
    };
    build_malloc_slowpath_body(&mut asm, slowpath_fn, propagate_path, kind);
    let buffer = codebuf::finalize_executable(asm.into_inner(), arena)
        .expect("malloc_slowpath_varsize: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// How `_build_malloc_slowpath` turns the entry registers into the
/// helper arguments. `Fixed` is `kind='fixed'` (size already in `edx`
/// via the caller's `SUB rdx, rcx`). `Var` is `kind='var'`: length in
/// `edx`, tid in `ecx`, itemsize in the caller's stack word. `Str` /
/// `Unicode` are `kind='str'` / `'unicode'`: length in `edx`, and the tid
/// `malloc_str` / `malloc_unicode` capture in `ecx` (`dynasm_malloc_str`
/// takes it as an argument).
#[derive(Clone, Copy)]
pub(crate) enum MallocSlowpathKind {
    Fixed,
    Str,
    Unicode,
    Var,
}

/// Body of `_build_malloc_slowpath` for `kind='fixed'` and `kind='var'`.
/// `CallMallocNursery` and `CallMallocNurseryVarsizeFrame` share the
/// fixed trampoline (`SUB rdx, rcx` is emitted by their builders).
/// `kind='var'` leaves `edx` as the length.
///
/// Emits `_push_all_regs_to_frame([ecx, edx])` → CALL helper →
/// reload_frame → inline WB → OOM check → `MOV ecx, eax` →
/// `_pop_all_regs_from_frame` → RET.

/// `slowpath_fn` is the absolute address of the GC helper called by
/// `MOV rax, imm64; CALL rax`.  `Fixed` passes the size in ARG0
/// (RCX on Win64, RDI on SysV).  `Var` passes
/// `(itemsize, tid, num_elem)`; `Str` / `Unicode` pass `(tid, length)`.
/// Result in RAX.
fn build_malloc_slowpath_body(
    asm: &mut LocationCodeBuilder,
    slowpath_fn: i64,
    propagate_path: usize,
    kind: MallocSlowpathKind,
) {
    let ignored = [crate::regloc::ECX, crate::regloc::EDX];

    // assembler.py `_push_all_regs_to_frame(mc, [ecx, edx], floats)`.
    // Saves every managed GPR/XMM except ECX/EDX so the inner CALL's
    // caller-clobber set is contained.  EAX is included in the save
    // set (unlike pyre's prior `[EAX, EDX]` mask), preserving any
    // live caller value across the slowpath — the regalloc only
    // promises ECX/EDX clobber to the caller.
    push_all_regs_to_jitframe_raw(asm, &ignored, true, false);

    // assembler.py `add_to_esp = 16 - WORD` plus Win64 shadow
    // space.  pyre's JIT body is 0-mod-16 (`_call_header`'s
    // `SUB rsp, FRAME_FIXED_SIZE * WORD`) and the outer `call rax` pushes
    // the return address, leaving rsp 8-mod-16 inside the trampoline.  The
    // SUB below brings rsp back to 0-mod-16 before the inner CALL — `8`
    // alignment alone on SysV, `8 + 32` shadow on Win64.
    #[cfg(target_os = "windows")]
    let align: i32 = 40;
    #[cfg(not(target_os = "windows"))]
    let align: i32 = 8;

    if align != 0 {
        rx86::sub_ri(asm, rx86::ESP, align);
    }
    // `_build_malloc_slowpath`: `kind='fixed'` passes the size
    // (`MOV ARG0, edx` after the caller's `SUB rdx, rcx`).
    // `kind='var'` reads the itemsize the callsite stored at `[rsp+WORD]`
    // on the trace; after this `SUB` it sits at `[rsp+align+2*WORD]`.
    match kind {
        MallocSlowpathKind::Fixed => {
            #[cfg(target_os = "windows")]
            let arg0_reg: u8 = 1; // rcx
            #[cfg(not(target_os = "windows"))]
            let arg0_reg: u8 = 7; // rdi
            dynasm!(asm ; .arch x64 ; mov Rq(arg0_reg), rdx);
        }
        MallocSlowpathKind::Str | MallocSlowpathKind::Unicode => {
            // `MOV_rr(ARG0, edx)` for the length; pyre's `malloc_str` takes
            // the captured tid first, so `(tid, length)` is `(ecx, edx)`.
            // On Win64 ARG0/ARG1 are ecx/edx already.
            #[cfg(not(target_os = "windows"))]
            dynasm!(asm ; .arch x64
                ; mov rdi, rcx
                ; mov rsi, rdx
            );
        }
        MallocSlowpathKind::Var => {
            // `malloc_array(itemsize, tid, num_elem)`. Careful ordering for
            // Win64, where ARG0 is ecx and ARG1 is edx: ARG2 <- edx (length),
            // ARG1 <- ecx (tid), then ARG0 <- the itemsize the callsite
            // stored at `[rsp+WORD]`, now above the return address.
            let item_ofs = align + (WORD as i32) * 2;
            #[cfg(target_os = "windows")]
            {
                dynasm!(asm ; .arch x64
                    ; mov r8, rdx
                    ; mov rdx, rcx
                );
                rx86::mov_rs(asm, rx86::ECX, item_ofs);
            }
            #[cfg(not(target_os = "windows"))]
            {
                // ARG2 is edx already.
                dynasm!(asm ; .arch x64 ; mov rsi, rcx);
                rx86::mov_rs(asm, rx86::EDI, item_ofs);
            }
        }
    }
    rx86::mov_ri(asm, rx86::EAX, slowpath_fn);
    dynasm!(asm ; .arch x64 ; call rax);
    if align != 0 {
        rx86::add_ri(asm, rx86::ESP, align);
    }

    // assembler.py:296 `_reload_frame_if_necessary(mc)` — rebind rbp
    // from the shadow stack in case a minor GC moved the jitframe.
    // ECX is used as scratch here (matches PyPy `_reload_frame_if_necessary`
    // assembler.py:1375); the helper return value still lives in RAX at
    // this point and is moved into ECX only after the reload (and after
    // the WB inline below, which preserves RAX via push/pop).
    let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    rx86::mov_ri(asm, rx86::ECX, rst_addr);
    dynasm!(asm ; .arch x64
        ; mov rcx, [rcx]
        ; mov rbp, [rcx - 8]
    );

    // assembler.py:_reload_frame_if_necessary line 1376 (Win64 + Linux
    // share): non-array write barrier on the reloaded jf so subsequent
    // Ref writes into frame slots are tracked by minor GC.  The barrier
    // body is conditional on the GC exposing a write-barrier descr.
    let wb_descr = crate::runner::dynasm_write_barrier_descr();
    if let Some(wb) = wb_descr {
        let byteofs = wb.jit_wb_if_flag_byteofs;
        let mask = wb.jit_wb_if_flag_singlebyte as i8;
        let skip_wb = asm.new_dynamic_label();
        rx86::test8_mi(asm, (rx86::EBP, byteofs), i32::from(mask));
        dynasm!(asm ; .arch x64 ; jz =>skip_wb);
        // Inline WB helper call: rbp -> ARG0, save/restore rax across.
        // Stack accounting at this point: the trampoline's pre-call
        // alignment SUB has been fully reversed by the matching ADD,
        // so rsp is back at the trampoline-entry value of 8-mod-16
        // (pyre JIT body 0-mod-16 + outer-CALL push).  `push rax`
        // brings rsp to 0-mod-16 — already aligned for the inner CALL
        // on SysV, while Win64 still needs its 32-byte shadow space.
        let wb_fn = crate::runner::dynasm_write_barrier as *const () as i64;
        #[cfg(target_os = "windows")]
        {
            dynasm!(asm ; .arch x64
                ; push rax
                ; sub rsp, 32           // 32 shadow (push rax already aligned)
                ; mov rcx, rbp
            );
            rx86::mov_ri(asm, rx86::EAX, wb_fn);
            dynasm!(asm ; .arch x64
                ; call rax
                ; add rsp, 32
                ; pop rax
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            dynasm!(asm ; .arch x64
                ; push rax
                ; mov rdi, rbp
            );
            rx86::mov_ri(asm, rx86::EAX, wb_fn);
            dynasm!(asm ; .arch x64
                ; call rax
                ; pop rax
            );
        }
        dynasm!(asm ; .arch x64 ; =>skip_wb);
    }

    // assembler.py:298-322 — TEST/JZ-to-OOM-tail with the common (success)
    // case as fall-through, matching PyPy's branch layout exactly.  The
    // `JZ` lands on the OOM tail emitted after RET; the fall-through path
    // does `MOV ecx, eax` → `_pop_all_regs_from_frame` → `pop_gcmap` →
    // `RET`.
    let oom_tail = asm.new_dynamic_label();
    dynasm!(asm ; .arch x64
        ; test rax, rax
        ; jz =>oom_tail
    );

    // assembler.py:304 `MOV_rr(ecx, eax)` — deliver the helper return
    // value through ECX so it survives `pop_all` (which restores RAX
    // from the save area).  RAX is still valid at this point because
    // the WB inline above brackets its inner CALL with `push rax /
    // pop rax`.
    dynasm!(asm ; .arch x64 ; mov rcx, rax);

    // assembler.py `_pop_all_regs_from_frame(mc, [ecx, edx], floats)`.
    pop_all_regs_from_jitframe_raw(asm, &ignored, true, false);
    // assembler.py:308 `self.pop_gcmap(mc)` — clear `JF_GCMAP_OFS`
    // before RET so the caller's regalloc layout (which never sees the
    // trampoline's saved-reg slots) is the only gcmap the next
    // collecting call walks.  Matches PyPy "trampoline owns the
    // gcmap-clear before RET" structure exactly.
    rx86::mov_bi(asm, crate::jitframe::JF_GCMAP_OFS, 0);
    dynasm!(asm ; .arch x64 ; ret);

    // assembler.py:309-322 OOM tail — patched JZ target above.  When the
    // slowpath helper returns NULL (`libc::calloc` / `gc.alloc_nursery_*`
    // OOM) PyPy tail-JMPs to the standalone `propagate_exception_path`
    // (line 322).  Stage `propagate_path` into a scratch register and
    // jump indirectly so the transfer is range-independent: dynasm-rs's
    // `jmp extern` would lower to `E9 + rel32`, but the malloc slowpath
    // and propagate trampoline are separate arena blocks that may belong to
    // different retained mappings and are not guaranteed to land within ±2GB,
    // so a relative external jump could otherwise fail with
    // `ImpossibleRelocation` and trip the caller's `expect("malloc_slowpath:
    // finalize")` on affected layouts.
    // `ADD esp, WORD` pops the trampoline return address so
    // `_call_footer` in `propagate_exception_path` sees the trace body
    // rsp `_call_header` left. The `'var'` itemsize lives in the
    // `PASS_ON_MY_FRAME` scratch, so this tail does not pop a callsite
    // reservation.
    dynasm!(asm ; .arch x64 ; =>oom_tail);
    rx86::add_ri(asm, rx86::ESP, WORD as i32);
    asm.JMP(imm(propagate_path as i64));
}

/// Pointer-identity key for `target_tokens_currently_compiling`. PyPy
/// x86/assembler.py:93 keys it by the descr Python object itself; we use
/// the underlying allocation address of the `Arc<dyn Descr>` so two
/// distinct TargetToken descriptors are never confused.
fn loop_target_id(op: &Op) -> Option<usize> {
    op.getdescr().as_ref().map(majit_ir::descr_identity)
}

fn target_argloc_from_loc(loc: Loc) -> TargetArgLoc {
    match loc {
        Loc::Reg(r) => TargetArgLoc::Reg {
            regnum: r.value,
            is_xmm: r.is_xmm,
        },
        Loc::Ebp(e) => TargetArgLoc::Ebp {
            ebp_offset: e.value,
            is_float: e.is_float,
        },
        Loc::Esp(e) => panic!("LABEL target argloc must not be a RawEspLoc, got {e:?}"),
        Loc::Frame(f) => TargetArgLoc::Frame {
            position: f.get_position(),
            ebp_offset: f.ebp_loc.value,
            is_float: f.ebp_loc.is_float,
        },
        // `consider_label`: `assert not isinstance(arg, Const)`, so
        // `_x86_arglocs` never holds an `ImmedLoc`, `FloatImmedLoc` or
        // `ConstFloatLoc`.
        Loc::Immed(_) | Loc::ImmedFloat(_) | Loc::ConstFloat(_) => {
            panic!("LABEL target argloc must not be a constant location, got {loc:?}")
        }
        Loc::Addr(a) => TargetArgLoc::Addr {
            base: a.base,
            index: a.index,
            scale: a.scale,
            offset: a.offset,
        },
    }
}

fn loc_from_target_argloc(loc: &TargetArgLoc) -> Loc {
    match *loc {
        TargetArgLoc::Reg { regnum, is_xmm } => {
            Loc::Reg(crate::regloc::RegLoc::new(regnum, is_xmm))
        }
        TargetArgLoc::Ebp {
            ebp_offset,
            is_float,
        } => Loc::Ebp(crate::regloc::RawEbpLoc {
            value: ebp_offset,
            is_float,
        }),
        TargetArgLoc::Frame {
            position,
            ebp_offset,
            is_float,
        } => Loc::Frame(crate::regloc::FrameLoc::new(position, ebp_offset, is_float)),
        TargetArgLoc::Immed { .. } => {
            panic!("LABEL target argloc must not be a constant location, got {loc:?}")
        }
        TargetArgLoc::Addr {
            base,
            index,
            scale,
            offset,
        } => Loc::Addr(crate::regloc::AddressLoc {
            base,
            index,
            scale,
            offset,
        }),
    }
}

fn core_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    crate::x86::regalloc::ALL_CORE_REGS
        .iter()
        .position(|candidate| *candidate == reg)
}

fn float_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    crate::x86::regalloc::ALL_FLOAT_REGS
        .iter()
        .position(|candidate| *candidate == reg)
        .map(|idx| crate::x86::regalloc::ALL_CORE_REGS.len() + idx)
}

fn reg_position_in_jitframe(reg: crate::regloc::RegLoc) -> Option<usize> {
    if reg.is_xmm {
        float_reg_position(reg)
    } else {
        core_reg_position(reg)
    }
}

fn deadframe_slot_for_loc(loc: &Loc) -> Option<u16> {
    match loc {
        Loc::Reg(reg) => Some(
            reg_position_in_jitframe(*reg).expect("deadframe slot: register is not managed") as u16,
        ),
        Loc::Frame(frame) => {
            // `store_info_on_descr`: `(loc.value - base_ofs) // WORD`, with
            // the misalignment and overflow assertions; 0xFFFF is the hole.
            assert_eq!(
                frame.ebp_loc.value & (WORD as i32 - 1),
                0,
                "store_info_on_descr: misaligned"
            );
            let position = frame.get_position() + JITFRAME_FIXED_SIZE;
            assert!(
                0 < position && position < 0xFFFF,
                "store_info_on_descr: overflow!"
            );
            Some(position as u16)
        }
        Loc::Immed(_)
        | Loc::ImmedFloat(_)
        | Loc::ConstFloat(_)
        | Loc::Ebp(_)
        | Loc::Esp(_)
        | Loc::Addr(_) => None,
    }
}

/// x86/assembler.py WriteBarrierSlowPath card-marking body.
///
/// Register and frame-index arms emit `SHR tmp, card_page_shift; XOR tmp, -8;
/// BTS [header], tmp`. `tmp` is `X86_64_SCRATCH_REG` (r11), which is outside
/// `ALL_CORE_REGS`, so the index register is never rewritten. A previous
/// byte-OR sequence copied `loc_base` into r10 (an allocatable GPR) and then
/// reloaded `loc_index` from that same register, so an index that already
/// lived in r10 became the array pointer and the card bit was taken from the
/// pointer instead of the index.
///
/// GCREF is the payload, so BTS uses displacement `-GcHeader::SIZE` — the
/// same header-relative bias `WriteBarrierDescr::extract_flag_byte` and the
/// Immed `byte_ofs` apply. Immediate indices keep the `OR8` fold
/// (`x86/assembler.py` ImmedLoc).
fn encode_wb_array_card_mark(mc: &mut Assembler, loc_base: u8, loc_index: &Loc, page_shift: u32) {
    match loc_index {
        Loc::Reg(idx) => {
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            if idx.value != scratch {
                dynasm!(mc
                    ; .arch x64
                    ; mov Rq(scratch), Rq(idx.value as u8)
                );
            }
            rx86::shr_ri(mc, scratch, page_shift as i32);
            rx86::xor_ri(mc, scratch, -8);
            rx86::bts_mr(
                mc,
                (loc_base, -(majit_gc::header::GcHeader::SIZE as i32)),
                scratch,
            );
        }
        Loc::Frame(idx) => {
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            rx86::mov_rb(mc, scratch, idx.ebp_loc.value);
            rx86::shr_ri(mc, scratch, page_shift as i32);
            rx86::xor_ri(mc, scratch, -8);
            rx86::bts_mr(
                mc,
                (loc_base, -(majit_gc::header::GcHeader::SIZE as i32)),
                scratch,
            );
        }
        Loc::Immed(idx) | Loc::ImmedFloat(idx) => {
            let byte_index = idx.value >> page_shift;
            let byte_ofs = !((byte_index >> 3) as i64) - majit_gc::header::GcHeader::SIZE as i64;
            let byte_val = 1_i64 << (byte_index & 7);
            rx86::or8_mi(mc, (loc_base, byte_ofs as i32), i32::from(byte_val as i8));
        }
        // x86/assembler.py:2387-2388
        // `raise AssertionError("index is neither RegLoc nor ImmedLoc")`
        _ => panic!("index is neither RegLoc nor ImmedLoc"),
    }
}

// ── Abstract condition codes ──
// Architecture-independent CC values used throughout the assembler.
// Converted to arch-specific encoding at emission time.
const CC_O: u8 = 0;
const CC_NO: u8 = 1;
const CC_B: u8 = 2; // unsigned <
const CC_AE: u8 = 3; // unsigned >=
const CC_E: u8 = 4; // ==
const CC_NE: u8 = 5; // !=
const CC_BE: u8 = 6; // unsigned <=
const CC_A: u8 = 7; // unsigned >
const CC_S: u8 = 8;
const CC_NS: u8 = 9;
const CC_L: u8 = 10; // signed <
const CC_GE: u8 = 11; // signed >=
const CC_LE: u8 = 12; // signed <=
const CC_G: u8 = 13; // signed >

/// Invert a condition code.
fn invert_cc(cc: u8) -> u8 {
    match cc {
        CC_O => CC_NO,
        CC_NO => CC_O,
        CC_B => CC_AE,
        CC_AE => CC_B,
        CC_E => CC_NE,
        CC_NE => CC_E,
        CC_BE => CC_A,
        CC_A => CC_BE,
        CC_S => CC_NS,
        CC_NS => CC_S,
        CC_L => CC_GE,
        CC_GE => CC_L,
        CC_LE => CC_G,
        CC_G => CC_LE,
        _ => CC_E, // fallback
    }
}

/// `codebuf.SlowPath`: a not-taken conditional jump to a body emitted after
/// the loop or bridge, then a jump back to the fast path.
///
/// `saved_scratch_value_1` is the r11 cache at the `jcc`
/// (`get_scratch_register_known_value`). `saved_scratch_value_2` is the
/// cache at `set_continue_here`; `-1` means unknown and
/// `load_scratch_if_known` emits nothing.
struct SlowPath {
    slow_label: DynamicLabel,
    continue_label: DynamicLabel,
    saved_scratch_value_1: i64,
    saved_scratch_value_2: i64,
    kind: SlowPathKind,
}

/// rewrite.py `FLAG_ARRAY` / `FLAG_STR` / `FLAG_UNICODE`: the `kind`
/// argument of `CALL_MALLOC_NURSERY_VARSIZE`.
const FLAG_ARRAY: i64 = 0;
const FLAG_STR: i64 = 1;
const FLAG_UNICODE: i64 = 2;

/// Body selected by `SlowPath.generate`.
enum SlowPathKind {
    WriteBarrier {
        /// `arglocs[0]`. A register, or the immediate `addr_add_const` encodes.
        loc_base: Loc,
        loc_index: Option<Loc>,
        helper_num: usize,
        card_marking: bool,
        card_page_shift: u32,
    },
    /// `StackCheckSlowPath`: one `call` of `stack_check_slowpath`.
    StackCheck,
    /// `IncreaseStackSlowPath`. `gcmap` is the pointer `_check_frame_depth`
    /// captured (`*mut usize` stored as `usize` so the queue stays `Send`).
    IncreaseStack { gcmap: usize },
    /// `CondCallSlowPath`. `gcmap` is the map `cond_call` read
    /// (`*mut usize` stored as `usize`). `arglocs` are the extra-arg
    /// locations `consider_cond_call` captured; `generate_body` remaps
    /// them into `cond_call_register_arguments`. `variant_num` indexes
    /// `cond_call_slowpath` (`floats * 2 + callee_only`).
    /// `should_be_saved` is `rm.reg_bindings.values()` as a bitmask
    /// over GPR numbers (`1 << gpr.value`).
    CondCall {
        gcmap: usize,
        imm_func: i64,
        arglocs: Vec<Loc>,
        resloc: Option<Loc>,
        variant_num: usize,
        should_be_saved: i64,
        guard_token_no_exception: Option<GuardToken>,
        expects_guard_no_exception: bool,
    },
    /// `MallocCondSlowPath`. `gcmap == 0` stores a null map.
    /// `synthesize` is the `nf_addr == 0` arm of `malloc_cond_varsize_frame`:
    /// `(sizeloc, GcHeader::SIZE)` so `sub rdx, rcx` still yields the
    /// byte count `_build_malloc_slowpath` passes on.
    MallocCond {
        gcmap: usize,
        result_reg: u8,
        synthesize: Option<(u8, i32)>,
    },
    /// `MallocCondVarsizeSlowPath`. `kind` is `FLAG_ARRAY` / `FLAG_STR` /
    /// `FLAG_UNICODE`. `lengthloc` is the length operand captured before
    /// the fast path rewrites `edx`.
    MallocCondVarsize {
        gcmap: usize,
        result_reg: u8,
        kind: i64,
        itemsize: i64,
        type_id: i64,
        lengthloc: Loc,
    },
}

/// assembler.py Assembler386.
/// In Rust, this is a transient builder — created per compilation,
/// not a long-lived object like RPython's.
///
/// Borrows the trace's `inputargs` and `operations` for its lifetime so
/// `OpRef → Type` resolves through `op.type_` / `inputarg.tp` directly
/// (RPython `box.type` parity); no `value_types: HashMap` side-table.
pub struct Assembler386<'a> {
    /// The dynasm assembler (rx86.py + codebuf.py combined).
    pub(crate) mc: LocationCodeBuilder,
    /// `BaseAssembler.asmmemmgr`: destination arena for materialization.
    asm_memory_manager: Arc<AsmMemoryManager>,
    /// assembler.py:83 pending_guard_tokens — guards awaiting recovery stubs.
    pending_guard_tokens: Vec<GuardToken>,
    /// `x86/regalloc.py` `min_bytes_before_label`. The next label
    /// (`consider_label` / `flush_loop`) is padded out to this offset so
    /// `redirect_call_assembler` and a `GUARD_NOT_INVALIDATED` patch cannot
    /// overwrite it.
    min_bytes_before_label: usize,
    /// GC bitmap to push before the current collecting call (e.g.,
    /// CallMallocNursery slow path). Set by `regalloc_perform` when the
    /// regalloc captured one, cleared after.
    pending_malloc_nursery_gcmap: Option<usize>,
    /// Index of the next guard / FINISH descr in `fail_descrs`.
    fail_index: u32,
    /// Frame depth (in WORD units) for the current trace.
    frame_depth: usize,
    /// `BaseAssembler._previous_rd_locs`.
    previous_rd_locs: majit_ir::RdLocs,
    /// Fail descriptors built during assembly — wrapped in `FailDescrCell`
    /// so `Arc::as_ptr` is a thin pointer suitable for direct
    /// `Arc::from_raw` recovery (`history.py AbstractDescr.show`).
    fail_descrs: FailDescrStore,
    /// trace_id for this compilation.
    trace_id: u64,
    /// header_pc (green_key) for this compilation.
    header_pc: u64,
    /// Input argument types.
    input_types: Vec<Type>,
    /// assembler.py rebuild_faillocs_from_descr parity:
    /// bridge input locations recovered from the source guard descr.
    bridge_input_locs: Option<Vec<Loc>>,

    // ── State tracking for code generation ──
    /// Maps OpRef → jitframe slot index.
    opref_to_slot: IndexMap<OpRef, usize, rustc_hash::FxBuildHasher>,
    /// Trace inputargs — borrowed for `opref_type` lookups.
    inputargs: &'a [InputArgRc],
    /// Trace operations — borrowed for `opref_type` lookups (reads
    /// `op.type_` directly, RPython `box.type` parity).
    operations: &'a [OpRc],
    /// `arg.index` raw -> idx in inputargs, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots. Mirrors
    /// `OpTypeIndex::inputarg_pos`.
    inputarg_pos: majit_ir::PosIndex,
    /// `op_pos[op.pos.raw()] = idx in operations`, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots and Void/None ops.
    /// Mirrors `OpTypeIndex::op_pos`.
    op_pos: majit_ir::PosIndex,
    /// Constants: OpRef index (>= 10000) → typed `Const` value. The box
    /// variant carries its own type (`Const::get_type`), so no separate
    /// constant-type map is needed.
    constants: majit_ir::ConstMap<majit_ir::Const>,
    /// Next available frame slot index.
    next_slot: usize,
    /// Condition code from the most recent CMP/TEST instruction,
    /// consumed by a following GUARD_TRUE/GUARD_FALSE.
    /// Stores an abstract condition code (CC_* constants).
    guard_success_cc: Option<u8>,
    /// x86/assembler.py:93 target_tokens_currently_compiling parity.
    /// Keyed by descriptor pointer identity (PyPy uses Python `is`).
    target_tokens_currently_compiling: IndexMap<usize, DynamicLabel>,
    compiled_target_tokens: Vec<majit_ir::DescrRef>,
    /// First cross-buffer JUMP whose target still points into an assembler
    /// buffer instead of executable memory.
    unrelocated_jump_target: Option<(usize, usize)>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
    /// `AbstractLLCPU.subclassrange_min_offset`. `None` keeps the
    /// TYPE_INFO arm of `genop_guard_guard_subclass`.
    subclassrange_min_offset: Option<usize>,
    /// llsupport/gc.py get_typeid_from_classptr_if_gcremovetypeptr vtable→typeid table, materialized by the runner
    /// via gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr. Used by
    /// the gcremovetypeptr branch of `_cmp_guard_class`.
    classptr_to_typeid: IndexMap<i64, u32>,
    /// TYPE_INFO / CLASSTYPE constants for `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`, fetched by the runner from the active gc_ll_descr.
    guard_gc_type_info: Option<GuardGcTypeInfo>,
    /// Constant classptr → `(subclassrange_min, subclassrange_max)`, matching
    /// `loc_check_against_class.getint()` field reads in
    /// `x86/assembler.py:1971-1974`.
    classptr_to_subclass_range: IndexMap<i64, (i64, i64)>,
    /// Dynamic label at the function entry for self-recursive CALL_ASSEMBLER.
    self_entry_label: Option<DynamicLabel>,
    /// Leaked pointer holding the resolved entry address for self-recursive
    /// CALL_ASSEMBLER via the execute trampoline. Written after finalization.
    self_entry_addr_ptr: *mut usize,
    /// opassembler.py:1177 _finish_gcmap.
    finish_gcmap: Option<*mut usize>,
    /// opassembler.py:1215 gcmap_for_finish.
    gcmap_for_finish: *mut usize,
    /// assembler.py _store_force_index parity:
    /// Pre-allocated fail descr for the next GUARD_NOT_FORCED, created
    /// at CALL_ASSEMBLER emission time so we can store its pointer to
    /// jf_force_descr before the call. Consumed by the subsequent
    /// GUARD_NOT_FORCED guard emission.
    pending_force_descr: Option<majit_ir::DescrRef>,
    /// Pre-wrapped `FailDescrCell` for the same pending guard.  Codegen
    /// bakes the cell's thin pointer into `JF_FORCE_DESCR_OFS` so that
    /// `force_token_to_dead_frame` (cranelift/compiler.rs) can
    /// recover the descr via `recover_fail_descr_cell` without the
    /// fat-pointer mismatch a bare `Arc<dyn Descr>` ptr would cause.
    /// The same cell is consumed by `append_guard_token_with_faillocs`
    /// so jf_force_descr and jf_descr resolve to the same identity.
    pending_force_cell: Option<usize>,
    /// `compile.py:665-674` + `pyjitpl.py:2283`: construction-time
    /// snapshot of the six descr pointers attached to the owning cpu
    /// instance.  Retained for constructor signature stability across
    /// runner / call_assembler callsites — emission helpers
    /// (`done_with_this_frame_descr_ptr_for_type`,
    /// `exit_frame_with_exception_descr_ref_ptr`,
    /// `propagate_exception_descr_ptr`) read live from `cpu_handle`
    /// instead so the raw ptr baked into `JF_DESCR_OFS` and the Arc
    /// stamped into `meta_descr` come from the same snapshot.
    #[allow(dead_code)]
    attached_descrs: crate::guard::AttachedDescrPtrs,
    /// `Arc` clone of the owning cpu's attachment handle.  Its heap
    /// address is baked into the CALL_ASSEMBLER helper call site as a
    /// compile-time immediate (`Arc::as_ptr`) and the `Arc` is moved
    /// into the resulting `CompiledCode` so the pointee outlives any
    /// subsequent `DynasmBackend` drop — same role as RPython's
    /// `self.cpu` attribute-access after whole-program translation,
    /// where the `cpu` object's identity is guaranteed by Python.
    cpu_handle: crate::guard::CpuDescrHandle,
    /// `assembler.py:94 setup()` `self.frame_depth_to_patch = []` —
    /// list of code-buffer byte offsets at which a placeholder 32-bit
    /// `0xffffff` was written for the stack-depth check / slowpath
    /// trampoline.  After materialisation, `patch_stack_checks`
    /// overwrites each entry with the final frame depth.
    ///
    /// Each entry stores an offset *relative to the start of the
    /// machine-code buffer* (i.e. pre-`rawstart`); `patch_stack_checks`
    /// adds `rawstart` to obtain the absolute address.
    frame_depth_to_patch: Vec<usize>,
    /// `Assembler386.pending_slowpaths`, created empty in `setup`.
    /// `flush_pending_slowpaths` is the first act of
    /// `write_pending_failure_recoveries`: a slow-path body may append a
    /// guard token, so recovery stubs are written after these paths.
    pending_slowpaths: Vec<SlowPath>,
    /// `pending_memoryerror_trampoline_from`: one label per compilation,
    /// bound by `generate_propagate_error_64` after the recovery stubs.
    /// `CHECK_MEMORY_ERROR` is not a `SlowPath`.
    pending_memoryerror_trampoline: Option<DynamicLabel>,
    /// assembler.py:1003-1008 `_assemble`: the frame depth of a cross-loop
    /// JUMP target (its `target_frame_depth`), or 0 when the trace has no
    /// external JUMP.  The closing `JMP` enters the target loop's body,
    /// which may use deeper frame slots than this trace; `_assemble` grows
    /// `frame_depth` to fit so a bridge's prologue `_check_frame_depth`
    /// reallocs the in-flight JITFRAME large enough before the `JMP`.
    jump_target_frame_depth: usize,
    /// `x86/assembler.py:63` `self.malloc_slowpath` parity — entry
    /// pointer of the per-CPU malloc slowpath trampoline used by both
    /// `CallMallocNursery` and `CallMallocNurseryVarsizeFrame` (PyPy
    /// line 2554 / 2578 both route through the same `malloc_slowpath`).
    /// Resolved by `X86CpuExt::ensure_malloc_slowpath_fixed` and
    /// passed in at construction time so the emit path bakes it as a
    /// 64-bit immediate without re-touching the backend.
    malloc_slowpath_fixed: usize,
    /// Headerless fixed-size nursery malloc slowpath trampoline.
    malloc_slowpath_headerless: usize,
    /// `_build_malloc_slowpath(kind='var')`. `malloc_cond_varsize` calls it.
    malloc_slowpath_varsize: usize,
    /// `_build_malloc_slowpath(kind='str')`.
    malloc_slowpath_str: usize,
    /// `_build_malloc_slowpath(kind='unicode')`.
    malloc_slowpath_unicode: usize,
    /// `assembler.py self.wb_slowpath`, resolved by
    /// `X86CpuExt::ensure_wb_slowpath`.
    wb_slowpath: [usize; 5],
    /// `Assembler386.propagate_exception_path`. 0 when the descr is not
    /// installed; `emit_propagate_exception_if_zero` is then a no-op.
    propagate_exception_path: usize,
    /// `Assembler386._frame_realloc_slowpath`.
    frame_realloc_slowpath: usize,
    /// `Assembler386.stack_check_slowpath`. 0 skips the probe
    /// (`_call_header_with_stack_check`).
    stack_check_slowpath: usize,
    /// `Assembler386.cond_call_slowpath`, the four
    /// `_build_cond_call_slowpath` entries. Index is
    /// `floats * 2 + callee_only`.
    cond_call_slowpath: [usize; 4],
    /// `Assembler386.failure_recovery_code`, the four
    /// `_build_failure_recovery` entries. Index is `exc + 2 * withfloats`.
    failure_recovery_code: [usize; 4],
    /// `assembler.py reserve_gcref_table`: one label per slot of the
    /// reference-constant table reserved at the start of this code block,
    /// which the `LoadFromGcTable` genop reads PC-relative. Empty when the
    /// trace references no reference constants.
    gcref_table: Vec<dynasmrt::DynamicLabel>,
    /// `genop_finish`'s `MOV_rp` targets. Each word is the singleton descr
    /// pointer, bound after the recoveries so the load stays rip-relative
    /// and the word is not executed.
    finish_descr_words: Vec<(dynasmrt::DynamicLabel, u64)>,
    /// `assembler.py` `datablockwrapper`. Float constants live here
    /// (`X86XMMRegisterManager.convert_to_imm` → `ConstFloatLoc`), not in
    /// the instruction stream.
    datablockwrapper: MachineDataBlockWrapper,
}

/// assembler.py GuardToken — represents a pending guard needing
/// a recovery stub to be written after the main loop body.
struct GuardToken {
    /// Dynamic label that the guard's Jcc jumps to — bound in
    /// write_pending_failure_recoveries to the recovery stub.
    fail_label: DynamicLabel,
    /// Descr for stub bookkeeping (`set_adr_jump_offset`).
    fail_descr: majit_ir::DescrRef,
    /// [`FailDescrCell::thin_ptr`] baked into `jf_descr`. The `Box` lives
    /// on `Asm::fail_descrs` so the address stays valid.
    fail_cell_ptr: usize,
    /// opassembler.py:515 GuardToken.gcmap.
    gcmap: *mut usize,
    /// `assembler.py implement_guard`'s `guard_token.pos_jump_offset` — the
    /// offset of the 4-byte target field of the guard's `Jcc`/`JMP`, which
    /// `patch_jump_for_descr` later redirects into a bridge. For
    /// `GUARD_NOT_INVALIDATED` it is the byte after the opcode of the
    /// not-yet-written `JMP` (`genop_guard_guard_not_invalidated`).
    /// `None` only for a guard that emits no branch at all.
    pos_jump_offset: Option<usize>,
    /// `GuardToken.guard_not_invalidated()`.
    guard_not_invalidated: bool,
    /// llsupport/assembler.py must_save_exception: true for
    /// GUARD_EXCEPTION / GUARD_NO_EXCEPTION / GUARD_NOT_FORCED.  Selects the
    /// exc=True failure-recovery variant that stages pos_exc_value into
    /// jf_guard_exc (store_info_on_descr:236) so grab_exc_value can read it.
    must_save_exception: bool,
    /// `implement_guard` / `genop_guard_guard_not_invalidated` store
    /// `get_scratch_register_known_value()` here. `-1` is unknown.
    /// `generate_quick_failure` assigns it back
    /// (`restore_scratch_register_known_value` emits no code). A token
    /// whose recovery is not entered from its own jump with r11 unchanged
    /// stays `-1`.
    known_scratch_value: i64,
}

/// What `assembler.py patch_pending_failure_recoveries` reads off a guard
/// token once the buffer has been materialised.  Upstream still has the tokens
/// there; `write_pending_failure_recoveries` consumes pyre's, so the three
/// fields the walk needs survive it as this.
struct RecoveryStub {
    /// `tok.faildescr`.
    fail_descr: majit_ir::DescrRef,
    /// `tok.pos_recovery_stub`.
    pos_recovery_stub: usize,
    /// `tok.pos_jump_offset`.
    pos_jump_offset: Option<usize>,
    /// `tok.guard_not_invalidated()`.
    guard_not_invalidated: bool,
}

fn fail_cell_capacity(ops: &[OpRc]) -> usize {
    ops.iter()
        .filter(|op| op.opcode.is_guard() || op.opcode == OpCode::Finish)
        .count()
}

/// Compiled output from assemble_loop/assemble_bridge.
pub struct CompiledCode {
    /// Executable memory buffer (keeps code alive).
    pub buffer: codebuf::ArenaExecutableBuffer,
    /// Entry point offset within the buffer.
    pub entry_offset: AssemblyOffset,
    /// Fail descriptors for guards + FINISH ops.
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at conversion
    /// from the in-progress `Assembler386.fail_descrs` Vec.
    pub fail_descrs: std::sync::Arc<FailDescrStore>,
    /// Input argument types.
    pub input_types: Vec<Type>,
    /// `compile.py` parity: `Arc` clone of the owning cpu's
    /// attachment handle.  Keeps the heap pointee alive for the whole
    /// lifetime of this compiled trace so the `cpu_handle` immediate
    /// baked into the CALL_ASSEMBLER helper call site never dangles,
    /// even if the emitting `DynasmBackend` is dropped first.
    pub cpu_attachments: crate::guard::CpuDescrHandle,
    /// trace_id.
    pub trace_id: u64,
    /// header_pc (green_key).
    pub header_pc: u64,
    /// Frame depth (number of jitframe slots used).
    /// AtomicUsize for redirect_call_assembler's update_frame_info
    /// parity: may be updated through &CompiledCode (shared ref).
    pub frame_depth: std::sync::atomic::AtomicUsize,
    /// `None` for root loops; bridges set `(source_trace_id, source_fail_index_per_trace)`.
    pub source_guard: Option<(u64, u32)>,
    /// `assembler.py patch_pending_failure_recoveries` — the
    /// `GUARD_NOT_INVALIDATED` sites this trace left for
    /// `clt.invalidate_positions`.
    pub invalidate_positions: Vec<majit_backend::InvalidatePosition>,
    /// `MachineDataBlockWrapper.done`: the ranges that hold `ConstFloatLoc`
    /// bytes. They stay mapped for as long as this code does.
    pub data_blocks: Vec<AsmMemoryBlock>,
}

impl CompiledCode {
    /// `looptoken._ll_function_addr = rawstart + functionpos`: the first
    /// instruction, past the reserved gcref table.
    pub fn entry_ptr(&self) -> *const u8 {
        self.buffer.ptr(self.entry_offset)
    }
}

/// `_build_float_constants`: 16-byte, 16-aligned sign masks. Both halves are
/// the same pattern so a 128-bit XORPD/ANDPD updates the low double.
#[repr(C, align(16))]
struct AlignedPdConst([u8; 16]);

static FLOAT_CONST_NEG: AlignedPdConst = AlignedPdConst([
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
]);

static FLOAT_CONST_ABS: AlignedPdConst = AlignedPdConst([
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f,
]);

/// The `genop_*` methods here are line-by-line ports of the RPython emitters
/// (`x86/assembler.py` `genop_*`).  Emission runs through `regalloc_perform`,
/// which works from regalloc `arglocs`, so the ports whose opcode that
/// dispatch reaches by another route carry `#[allow(dead_code)]`
/// individually.  They are annotated rather than deleted so the upstream
/// method boundary stays where a later porter looks for it; the attribute is
/// per method so the lint still reports anything else in this `impl` that
/// stops being reached.
impl<'a> Assembler386<'a> {
    /// rpython/jit/metainterp/history.py is_constant `box.type` parity.
    /// Single source of truth: `op.type_` for ops, `inputarg.tp` for
    /// inputargs, the `Const` variant tag for constants.
    ///
    /// Unreached since the call emitters moved onto regalloc arglocs, which
    /// carry the type in the `Loc` itself; kept per the note above.
    #[allow(dead_code)]
    #[inline]
    fn opref_type(&self, opref: OpRef) -> Option<Type> {
        self.opref_type_at(opref, None)
    }

    #[inline]
    fn opref_type_at(&self, opref: OpRef, at_op_index: Option<usize>) -> Option<Type> {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        match at_op_index {
            Some(at) => type_index.opref_type_at(opref, at),
            None => type_index.opref_type(opref),
        }
    }
    /// assembler.py:54 __init__
    pub(crate) fn new(
        asm_memory_manager: Arc<AsmMemoryManager>,
        trace_id: u64,
        header_pc: u64,
        constants: majit_ir::ConstMap<majit_ir::Const>,
        vtable_offset: Option<usize>,
        subclassrange_min_offset: Option<usize>,
        classptr_to_typeid: IndexMap<i64, u32>,
        guard_gc_type_info: Option<GuardGcTypeInfo>,
        classptr_to_subclass_range: IndexMap<i64, (i64, i64)>,
        attached_descrs: crate::guard::AttachedDescrPtrs,
        cpu_handle: crate::guard::CpuDescrHandle,
        malloc_slowpath_fixed: usize,
        malloc_slowpath_headerless: usize,
        malloc_slowpath_varsize: usize,
        malloc_slowpath_str: usize,
        malloc_slowpath_unicode: usize,
        wb_slowpath: [usize; 5],
        propagate_exception_path: usize,
        frame_realloc_slowpath: usize,
        stack_check_slowpath: usize,
        cond_call_slowpath: [usize; 4],
        failure_recovery_code: [usize; 4],
        inputargs: &'a [InputArgRc],
        operations: &'a [OpRc],
    ) -> Self {
        let inputarg_pos = OpTypeIndex::<OpRc, InputArgRc>::build_inputarg_pos(inputargs);
        let op_pos = OpTypeIndex::<OpRc, InputArgRc>::build_op_pos(operations);
        let datablockwrapper = MachineDataBlockWrapper::new(Arc::clone(&asm_memory_manager));
        Assembler386 {
            mc: LocationCodeBuilder::new(),
            asm_memory_manager,
            pending_guard_tokens: Vec::new(),
            min_bytes_before_label: 0,
            pending_malloc_nursery_gcmap: None,
            fail_index: 0,
            frame_depth: JITFRAME_FIXED_SIZE,
            previous_rd_locs: majit_ir::RdLocs::new(),
            fail_descrs: FailDescrStore::default(),
            trace_id,
            header_pc,
            input_types: Vec::new(),
            bridge_input_locs: None,
            opref_to_slot: IndexMap::with_hasher(rustc_hash::FxBuildHasher),
            inputargs,
            operations,
            inputarg_pos,
            op_pos,
            constants,
            next_slot: 0,
            guard_success_cc: None,
            target_tokens_currently_compiling: IndexMap::new(),
            compiled_target_tokens: Vec::new(),
            unrelocated_jump_target: None,
            vtable_offset,
            subclassrange_min_offset,
            classptr_to_typeid,
            guard_gc_type_info,
            classptr_to_subclass_range,
            self_entry_label: None,
            self_entry_addr_ptr: Box::into_raw(Box::new(0usize)),
            finish_gcmap: None,
            gcmap_for_finish: {
                let gcmap = allocate_gcmap(1, JITFRAME_FIXED_SIZE);
                gcmap_set_bit(gcmap, 0);
                gcmap
            },
            pending_force_descr: None,
            pending_force_cell: None,
            attached_descrs,
            cpu_handle,
            frame_depth_to_patch: Vec::new(),
            pending_slowpaths: Vec::new(),
            pending_memoryerror_trampoline: None,
            jump_target_frame_depth: 0,
            malloc_slowpath_fixed,
            malloc_slowpath_headerless,
            malloc_slowpath_varsize,
            malloc_slowpath_str,
            malloc_slowpath_unicode,
            wb_slowpath,
            propagate_exception_path,
            frame_realloc_slowpath,
            stack_check_slowpath,
            cond_call_slowpath,
            failure_recovery_code,
            gcref_table: Vec::new(),
            finish_descr_words: Vec::new(),
            datablockwrapper,
        }
    }

    /// Two `int32` halves of a `ConstFloatLoc`. `regalloc_immedmem2mem`
    /// reads them from the pool address (`CArrayPtr(INT)`).
    fn const_float_halves(from: crate::regloc::ConstFloatLoc) -> (i32, i32) {
        let low = unsafe { (from.value as *const i32).read_unaligned() };
        let high = unsafe { (from.value as *const i32).add(1).read_unaligned() };
        (low, high)
    }

    /// `regalloc_immedmem2mem`: a `ConstFloatLoc` moved directly to a
    /// `RawEbpLoc` or `RawEspLoc` is two 32-bit immediate stores of the
    /// halves already written into the data block (even on x86-64, because
    /// the immediates are encoded as 32 bits).
    fn regalloc_immedmem2mem(&mut self, from: crate::regloc::ConstFloatLoc, to_loc: &Loc) {
        let (low, high) = Self::const_float_halves(from);
        match to_loc {
            ebp_loc_pat!(e) => {
                rx86::mov32_bi(&mut self.mc, e.value, low);
                rx86::mov32_bi(&mut self.mc, e.value.wrapping_add(4), high);
            }
            Loc::Esp(e) => {
                rx86::mov32_si(&mut self.mc, e.value, low);
                rx86::mov32_si(&mut self.mc, e.value.wrapping_add(4), high);
            }
            _ => panic!("regalloc_immedmem2mem to {to_loc:?}"),
        }
    }

    /// `assembler.py reserve_gcref_table`: reserve `n` zeroed words,
    /// padded to a multiple of 16 bytes, at the start of the machine code,
    /// so the `LoadFromGcTable` genop can address them PC-relative. The
    /// runner writes the gcrefs into them once the block is materialized
    /// (`patch_gcref_table`; `GcTable::in_code`). Must run before any
    /// instruction is emitted.
    pub(crate) fn reserve_gcref_table(&mut self, n: usize) {
        assert_eq!(
            self.mc.offset().0,
            0,
            "the gcref table opens the code block"
        );
        for _ in 0..n {
            let slot = self.mc.new_dynamic_label();
            self.mc.forget_scratch_register();
            dynasm!(self.mc ; =>slot ; .u64 0);
            self.gcref_table.push(slot);
        }
        if n % 2 == 1 {
            dynasm!(self.mc ; .u64 0);
        }
    }

    /// Bind `genop_finish` descr words after every executable path.
    fn emit_finish_descr_words(&mut self) {
        let words = std::mem::take(&mut self.finish_descr_words);
        for (slot, ptr) in words {
            self.mc.forget_scratch_register();
            dynasm!(self.mc ; =>slot);
            for byte in ptr.to_le_bytes() {
                DynasmApi::push(&mut *self.mc, byte);
            }
        }
    }

    /// `compile.py:665` parity: heap-pinned address of `self.cpu`'s
    /// attachment handle, derived from the Arc clone.  Baked into the
    /// CALL_ASSEMBLER helper call site.
    fn cpu_handle_ptr(&self) -> i64 {
        Arc::as_ptr(&self.cpu_handle) as *const () as i64
    }

    /// `compile.py:665-674` parity: attach the six metainterp descrs on
    /// the emission side.  Mirrors `self.cpu.done_with_this_frame_descr_*`
    /// reads in `rpython/jit/backend/x86/assembler.py`.  Reads from the
    /// live `cpu_handle` snapshot so the raw pointer baked into
    /// `JF_DESCR_OFS` and the Arc returned by
    /// `done_with_this_frame_descr_arc_for_type` resolve to the same
    /// metainterp singleton.
    fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> i64 {
        self.cpu_handle
            .read()
            .descr_ptrs()
            .done_with_this_frame_descr_ptr_for_type(tp) as i64
    }

    /// `compile.py:665-674` `make_and_attach_done_descrs` Arc lookup —
    /// returns the metainterp `DoneWithThisFrameDescr*` Arc the
    /// optimizer attached for the given result type.  Used to stamp
    /// `meta_descr` on backend FINISH descrs so trait forwarding routes
    /// `is_finish` / `fail_arg_types` through the metainterp class
    /// hierarchy (`compile.py:624 final_descr=True`).
    fn done_with_this_frame_descr_arc_for_type(&self, tp: Type) -> Option<majit_ir::DescrRef> {
        let attachments = self.cpu_handle.read();
        match tp {
            Type::Void => attachments.done_with_this_frame_descr_void.clone(),
            Type::Int => attachments.done_with_this_frame_descr_int.clone(),
            Type::Ref => attachments.done_with_this_frame_descr_ref.clone(),
            Type::Float => attachments.done_with_this_frame_descr_float.clone(),
        }
    }

    /// `compile.py:658` parity: `self.cpu.exit_frame_with_exception_descr_ref`.
    fn exit_frame_with_exception_descr_ref_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .descr_ptrs()
            .exit_frame_with_exception_descr_ref as i64
    }

    /// `pyjitpl.py` parity: `self.cpu.propagate_exception_descr`.
    /// `build_propagate_exception_path` stamps it into `jf_descr`.
    /// `genop_discard_check_memory_error` jumps there when the checked
    /// value is zero.
    fn propagate_exception_descr_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .descr_ptrs()
            .propagate_exception_descr as i64
    }

    // Helper methods

    /// Frame-pointer-relative byte offset for a given slot index.
    /// Slots are absolute jf_frame indices, including the fixed
    /// JITFRAME-managed prefix. FIRST_ITEM_OFFSET accounts for the object
    /// header and array-length word that precede jf_frame[0].
    fn slot_offset(slot: usize) -> i32 {
        FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32
    }

    /// Resolve an OpRef to either a frame slot offset or an immediate constant.
    /// Cranelift resolve_opref parity: check constants map FIRST
    /// (regardless of CONST_BIT), then fall back to slot mapping.
    fn resolve_opref(&self, opref: OpRef) -> ResolvedArg {
        // Op results take precedence over constants (Cranelift parity).
        if let Some(&slot) = self.opref_to_slot.get(&opref) {
            return ResolvedArg::Slot(Self::slot_offset(slot));
        }
        // history.py/268/314 — inline-Const variants carry value inline.
        if let Some(val) = opref
            .inline_const_bits()
            .or_else(|| self.constants.get(&opref.raw()).map(|c| c.as_raw_i64()))
        {
            return ResolvedArg::Const(val);
        }
        // history.py/268/314 — Const always carries a value, so a
        // constant OpRef with no resolvable value is an invariant break,
        // not a `#0`.
        if opref.is_constant() {
            panic!(
                "resolve_opref: legacy constant {opref:?} missing from constants pool — \
                 Const always carries a value (history.py:227/268/314)"
            );
        }
        // regalloc.py:102 `FrameManager.loc(must_exist=True)` raises KeyError
        // for a non-constant box that is neither register- nor frame-resident:
        // a used box is always slot-mapped or constant. Silently materializing
        // an unmapped box as `#0` would hide a wrong value — the same hazard
        // that moved genop_call_assembler onto regalloc arglocs. Fail loud.
        panic!(
            "resolve_opref: unmapped non-constant OpRef {opref:?} — every used \
             box must be slot-mapped or constant (regalloc.py:102 loc must_exist)"
        );
    }

    /// Allocate a frame slot for an OpRef and return the slot index.
    /// Reuses existing slot if the OpRef already has one.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        if let Some(&existing) = self.opref_to_slot.get(&opref) {
            return existing;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref, slot);
        slot
    }

    /// Emit: ADD/SUB/AND/OR/XOR reg, loc
    fn emit_binop_reg_loc(&mut self, opcode: OpCode, dst_reg: u8, src: &Loc) {
        let dst = Operand::R(dst_reg);
        match opcode {
            OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                self.mc.ADD(dst, src)
            }
            OpCode::IntSub | OpCode::IntSubOvf => self.mc.SUB(dst, src),
            OpCode::IntMul | OpCode::IntMulOvf => self.mc.IMUL(dst, src),
            // genop_int_and: AND with (1<<32)-1 is one zero-extending MOV32.
            // and r, imm32 would sign-extend 0xffffffff into an all-ones no-op.
            OpCode::IntAnd if matches!(src.as_immed(), Some((v, _)) if v == (1i64 << 32) - 1) => {
                self.mc.MOV32(dst, dst)
            }
            OpCode::IntAnd => self.mc.AND(dst, src),
            OpCode::IntOr => self.mc.OR(dst, src),
            OpCode::IntXor => self.mc.XOR(dst, src),
            _ => {}
        }
    }

    /// Emit: CMP loc0, loc1
    fn emit_cmp_loc_loc(&mut self, loc0: &Loc, loc1: &Loc) {
        // `_binaryop('CMP')` has no xmm form: a FLOAT compare is `UCOMISD`
        // (`genop_guard_guard_value`, `_cmpop_float`).
        assert!(
            !matches!(loc0, Loc::Reg(r) if r.is_xmm) && !matches!(loc1, Loc::Reg(r) if r.is_xmm),
            "CMP on an xmm register ({loc0:?}, {loc1:?}); a FLOAT compare is UCOMISD"
        );
        match (loc0, loc1) {
            (Loc::Immed(_) | Loc::ImmedFloat(_), _)
            | (Loc::Frame(_) | Loc::Ebp(_), Loc::Frame(_) | Loc::Ebp(_)) => {
                // No `CMP_i*` / `CMP_bb` encoding: the left operand is staged
                // through the scratch register, which `INSN('CMP')` refuses,
                // so the compare itself is the raw `CMP_r*` form.
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                self.mc.MOV(Operand::R(scratch), loc0);
                match Operand::from(loc1) {
                    Operand::R(s) => dynasm!(self.mc ; .arch x64 ; cmp Rq(scratch), Rq(s)),
                    Operand::B(ofs) => rx86::cmp_rb(&mut self.mc, scratch, ofs),
                    Operand::I(v) if rx86::fits_in_32bits(v) => {
                        rx86::cmp_ri(&mut self.mc, scratch, v as i32)
                    }
                    // `_missing_binary_insn`: a second 64-bit operand would
                    // need the scratch register too.
                    _ => panic!("CMP has no encoding for ({loc0:?}, {loc1:?})"),
                }
            }
            _ => self.mc.CMP(loc0, loc1),
        }
    }

    /// `test_location`: `TEST` a register; `CMP [mem], imm0` for a frame
    /// location. Upstream rejects `ImmedLoc`; a constant is staged through
    /// the scratch register so the following branch does not read stale flags.
    fn emit_test_loc(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) => self.mc.TEST(*r, *r),
            Loc::Immed(_) | Loc::ImmedFloat(_) => {
                // Staged through the scratch register, which `INSN('TEST')`
                // refuses, so the test is the raw `TEST_rr` form.
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                self.mc.MOV(Operand::R(scratch), loc);
                dynasm!(self.mc ; .arch x64 ; test Rq(scratch), Rq(scratch));
            }
            _ => self.mc.CMP(loc, imm(0)),
        }
    }

    // ── AArch64 helper: load a 64-bit immediate into register Xn ──

    /// Emit: load the value of `opref` into RAX (x64) / X0 (aarch64).
    fn load_arg_to_rax(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                rx86::mov_rb(&mut self.mc, rx86::EAX, offset);
            }
            ResolvedArg::Const(val) => {
                rx86::mov_ri(&mut self.mc, rx86::EAX, val);
            }
        }
    }

    /// Emit: load a regalloc Loc into RAX (x64) / X0 (aarch64).
    /// Unlike load_arg_to_rax, this uses the regalloc-determined location
    /// instead of resolve_opref(), so register-carried values are preserved.
    fn emit_load_to_rax(&mut self, loc: Loc) {
        let rax = Loc::Reg(crate::regloc::RegLoc {
            value: 0,
            is_xmm: false,
        });
        match loc {
            Loc::Reg(r) if r.value == 0 && !r.is_xmm => {
                // already in rax/x0
            }
            Loc::Immed(imm) | Loc::ImmedFloat(imm) => {
                rx86::mov_ri(&mut self.mc, rx86::EAX, imm.value);
            }
            _ => self.regalloc_mov(&loc, &rax),
        }
    }

    /// Emit: load a regalloc Loc into the dedicated scratch (R11), which is
    /// outside `ALL_CORE_REGS`, so the load cannot clobber a value the
    /// regalloc still has live in an allocatable register.  Counterpart of
    /// the AArch64 `emit_load_loc_to_ip0`.
    fn emit_load_loc_to_scratch(&mut self, loc: Loc) {
        let scratch = crate::regloc::X86_64_SCRATCH_REG;
        match loc {
            // already there
            Loc::Reg(r) if !r.is_xmm && r.value == scratch.value => {}
            _ => self.regalloc_mov(&loc, &Loc::Reg(scratch)),
        }
    }

    /// Emit: load the value of `opref` into RCX (x64) / X1 (aarch64).
    fn load_arg_to_rcx(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                rx86::mov_rb(&mut self.mc, rx86::ECX, offset);
            }
            ResolvedArg::Const(val) => {
                rx86::mov_ri(&mut self.mc, rx86::ECX, val);
            }
        }
    }

    /// Emit: store RAX/X0 to the frame slot for `result_opref`.
    /// Allocates a new slot if needed.
    fn store_rax_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        rx86::mov_br(&mut self.mc, offset, rx86::EAX);
    }

    // assembler.py `_call_header` — function prologue

    fn setup_input_state(&mut self, inputargs: &[InputArgRc]) {
        // opref_to_slot stores ABSOLUTE jitframe slot indices so that
        // slot_offset(slot) returns the correct byte offset directly.
        // User position `p` maps to absolute slot `p + JITFRAME_FIXED_SIZE`.
        if let Some(ref input_locs) = self.bridge_input_locs {
            let mut max_abs_slot = JITFRAME_FIXED_SIZE;
            for (ia, loc) in inputargs.iter().zip(input_locs.iter()) {
                if let Loc::Frame(floc) = loc {
                    let abs_slot = JITFRAME_FIXED_SIZE + floc.get_position();
                    self.opref_to_slot.insert(ia.opref(), abs_slot);
                    if abs_slot + 1 > max_abs_slot {
                        max_abs_slot = abs_slot + 1;
                    }
                }
            }
            self.next_slot = max_abs_slot;
        } else {
            for (i, ia) in inputargs.iter().enumerate() {
                self.opref_to_slot
                    .insert(ia.opref(), JITFRAME_FIXED_SIZE + i);
            }
            self.next_slot = JITFRAME_FIXED_SIZE + inputargs.len();
        }
    }

    /// Emit the function prologue.
    /// x64: System V AMD64 ABI — first arg (jf_ptr) in RDI.
    /// aarch64: AAPCS64 — first arg (jf_ptr) in X0.
    ///
    /// `_call_header_with_stack_check`: SP probe after `_call_header`,
    /// which ends with `_call_header_shadowstack`. Deep compiled-to-compiled
    /// recursion returns the caller-provided jf_ptr in RAX so the glue
    /// drains the overflow on the way back to the interpreter.
    ///
    /// Fast path (`StackCheckSlowPath`, condition `A`):
    /// ```text
    ///   MOV  eax, [endaddr]
    ///   SUB  rax, rsp
    ///   CMP  eax, [lengthaddr]
    ///   JA   slow                 ; not taken when eax <= [length]
    /// continue:
    /// ```
    /// The slow body is one `call` of `stack_check_slowpath`. On overflow
    /// that helper drops its return address and jumps to
    /// `propagate_exception_path`, whose `_call_footer` pops the shadow
    /// stack entry pushed above.
    fn _call_header(&mut self, inputargs: &[InputArgRc]) {
        // x86/assembler.py `_call_header`. One
        // `SUB rsp, FRAME_FIXED_SIZE * WORD` reserves the
        // `PASS_ON_MY_FRAME` scratch and the callee-save homes. `ebp`
        // lands at `PASS_ON_MY_FRAME * WORD`, arg2 (the thread-local
        // address) at `THREADLOCAL_OFS` before that register is reused,
        // then `ebp` becomes arg1. `CALLEE_SAVE_REGISTERS` follow at
        // `(PASS_ON_MY_FRAME + i + 1) * WORD`; Win64 i == 4/5 use
        // `SHADOWSTORE2_OFS` / `SHADOWSTORE3_OFS`. `FRAME_FIXED_SIZE` is
        // odd, so the 8-mod-16 entry rsp becomes a 0-mod-16 body.
        // `arch.py` says "never use r13", so Win64 does not save it.
        let frame = (FRAME_FIXED_SIZE * WORD) as i32;
        rx86::sub_ri(&mut self.mc, rx86::ESP, frame);
        rx86::mov_sr(&mut self.mc, (PASS_ON_MY_FRAME * WORD) as i32, rx86::EBP);
        #[cfg(target_os = "windows")]
        {
            rx86::mov_sr(&mut self.mc, THREADLOCAL_OFS as i32, rx86::EDX);
            dynasm!(self.mc ; .arch x64 ; mov rbp, rcx);
        }
        #[cfg(not(target_os = "windows"))]
        {
            rx86::mov_sr(&mut self.mc, THREADLOCAL_OFS as i32, rx86::ESI);
            dynasm!(self.mc ; .arch x64 ; mov rbp, rdi);
        }
        emit_save_callee_saves(&mut self.mc);
        // `_call_header_shadowstack`, the tail of `_call_header`.
        self.gen_shadowstack_header();
        // `_call_header_with_stack_check`: emit nothing while
        // `stack_check_slowpath` is 0 (descr missing, or
        // `insert_stack_check` not registered). The helper is built
        // only when both are ready.
        if self.stack_check_slowpath != 0 {
            let addrs = crate::stack_check_addresses()
                .expect("stack_check_slowpath is built only after insert_stack_check");
            let eax = crate::regloc::EAX;
            self.mc.MOV(eax, heap(addrs.end_adr as i64)); // MOV eax, [start]
            self.mc.SUB(eax, crate::regloc::ESP); // SUB eax, current
            self.mc.CMP(eax, heap(addrs.length_adr as i64)); // CMP eax, [length]
            // Condition `A`: not taken when `eax <= [length]`.
            // `set_continue_here` does not forget; the slow-path
            // epilogue reloads the r11 cache before jumping back.
            let mut sp = self.emit_slow_jcc(CC_A, SlowPathKind::StackCheck);
            self.set_continue_here(&mut sp);
            self.pending_slowpaths.push(sp);
        }
        self.setup_input_state(inputargs);
    }

    fn abi_int_arg(idx: usize) -> AbiArgPlacement {
        #[cfg(target_os = "windows")]
        match idx {
            0 => AbiArgPlacement::Gpr(1), // rcx
            1 => AbiArgPlacement::Gpr(2), // rdx
            2 => AbiArgPlacement::Gpr(8),
            3 => AbiArgPlacement::Gpr(9),
            _ => AbiArgPlacement::Stack(32 + ((idx - 4) * WORD) as i32),
        }
        #[cfg(not(target_os = "windows"))]
        match idx {
            0 => AbiArgPlacement::Gpr(7), // rdi
            1 => AbiArgPlacement::Gpr(6), // rsi
            2 => AbiArgPlacement::Gpr(2), // rdx
            3 => AbiArgPlacement::Gpr(1), // rcx
            4 => AbiArgPlacement::Gpr(8),
            5 => AbiArgPlacement::Gpr(9),
            _ => AbiArgPlacement::Stack(((idx - 6) * WORD) as i32),
        }
    }

    fn emit_abi_arg_from_reg(
        &mut self,
        placement: AbiArgPlacement,
        src: crate::regloc::RegLoc,
        arg_type: Type,
    ) {
        match placement {
            AbiArgPlacement::Gpr(dst) if src.is_xmm => self.mc.MOVDQ(Operand::R(dst), src),
            AbiArgPlacement::Gpr(dst) => self.mc.MOV(Operand::R(dst), src),
            AbiArgPlacement::Stack(offset) if src.is_xmm && arg_type != Type::Float => {
                rx86::movq_mx(&mut self.mc, (rx86::ESP, offset), src.value);
            }
            AbiArgPlacement::Stack(offset) if src.is_xmm => self.mc.MOVSD(Operand::S(offset), src),
            AbiArgPlacement::Stack(offset) => self.mc.MOV(Operand::S(offset), src),
        }
    }

    fn emit_abi_arg_from_mem(&mut self, placement: AbiArgPlacement, offset: i32, arg_type: Type) {
        let src = Operand::B(offset);
        match placement {
            AbiArgPlacement::Gpr(dst) => self.mc.MOV(Operand::R(dst), src),
            AbiArgPlacement::Stack(dst_offset) => {
                let dst = Operand::S(dst_offset);
                if arg_type == Type::Float {
                    let scratch = crate::regloc::X86_64_XMM_SCRATCH_REG;
                    self.mc.MOVSD(scratch, src);
                    self.mc.MOVSD(dst, scratch);
                } else {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG;
                    self.mc.MOV(scratch, src);
                    self.mc.MOV(dst, scratch);
                }
            }
        }
    }

    fn emit_abi_arg_from_imm(&mut self, placement: AbiArgPlacement, val: i64, arg_type: Type) {
        let _ = arg_type;
        let dst = match placement {
            AbiArgPlacement::Gpr(dst) => Operand::R(dst),
            AbiArgPlacement::Stack(offset) => Operand::S(offset),
        };
        self.mc.MOV(dst, imm(val));
    }

    fn emit_abi_int_arg_from_reg(&mut self, idx: usize, src: u8) {
        self.emit_abi_arg_from_reg(
            Self::abi_int_arg(idx),
            crate::regloc::RegLoc::new(src, false),
            Type::Int,
        );
    }

    fn emit_abi_int_arg_from_imm(&mut self, idx: usize, val: i64) {
        self.emit_abi_arg_from_imm(Self::abi_int_arg(idx), val, Type::Int);
    }

    fn emit_abi_int_arg_from_mem(&mut self, idx: usize, offset: i32) {
        self.emit_abi_arg_from_mem(Self::abi_int_arg(idx), offset, Type::Int);
    }

    /// Bytes to subtract before a C call so the callee's entry rsp is
    /// 8 mod 16 and, on Win64, its shadow is writable.
    ///
    /// `extra_pushes == 0` means rsp is still the body rsp from
    /// `_call_header`. `CallBuilderX86.subtract_esp_aligned` then
    /// subtracts only `on_stack - PASS_ON_MY_FRAME` words (zero when the
    /// outgoing area fits in that scratch).
    ///
    /// `extra_pushes != 0` means those pushes sit below the scratch, so
    /// the outgoing area is not `PASS_ON_MY_FRAME`. Reserve the full
    /// `outgoing_stack_words` block under the pushes, plus one word when
    /// that would leave rsp not 0 mod 16. One push is 40 bytes on Win64
    /// and 8 on SysV; two pushes are the 32-byte Win64 shadow alone.
    fn abi_reserved_call_area_size(extra_pushes: usize, stack_slots: usize) -> i32 {
        let on_stack = super::callbuilder::outgoing_stack_words(stack_slots);
        if extra_pushes == 0 {
            return super::callbuilder::aligned_esp_bytes(on_stack - PASS_ON_MY_FRAME as i32);
        }
        let base = on_stack * WORD as i32;
        let needs_pad = if extra_pushes & 1 == 0 {
            base % 16 != 0
        } else {
            base % 16 == 0
        };
        base + if needs_pad { WORD as i32 } else { 0 }
    }

    fn emit_reserve_abi_call_area(&mut self, extra_pushes: usize, stack_slots: usize) -> i32 {
        let adjust = Self::abi_reserved_call_area_size(extra_pushes, stack_slots);
        if adjust != 0 {
            rx86::sub_ri(&mut self.mc, rx86::ESP, adjust);
        }
        adjust
    }

    fn emit_release_abi_call_area(&mut self, adjust: i32) {
        if adjust != 0 {
            rx86::add_ri(&mut self.mc, rx86::ESP, adjust);
        }
    }

    fn emit_abi_call_rax_with_extra_pushes(&mut self, extra_pushes: usize) {
        let adjust = Self::abi_reserved_call_area_size(extra_pushes, 0);
        if adjust != 0 {
            rx86::sub_ri(&mut self.mc, rx86::ESP, adjust);
        }
        self.mc.CALL(crate::regloc::EAX);
        if adjust != 0 {
            rx86::add_ri(&mut self.mc, rx86::ESP, adjust);
        }
    }

    fn emit_abi_call_rax(&mut self) {
        self.emit_abi_call_rax_with_extra_pushes(0);
    }

    fn emit_abi_call_rax_aligned(&mut self) {
        // Body rsp is already 0 mod 16 (`_call_header`). A register-only
        // call fits in `PASS_ON_MY_FRAME`, including the Win64 shadow.
        self.emit_abi_call_rax();
    }

    fn emit_abi_call_rax_after_one_push(&mut self) {
        self.emit_abi_call_rax_with_extra_pushes(1);
    }

    fn emit_abi_call_reg_with_extra_pushes(&mut self, reg: u8, extra_pushes: usize) {
        let adjust = Self::abi_reserved_call_area_size(extra_pushes, 0);
        if adjust != 0 {
            rx86::sub_ri(&mut self.mc, rx86::ESP, adjust);
        }
        self.mc.CALL(Operand::R(reg));
        if adjust != 0 {
            rx86::add_ri(&mut self.mc, rx86::ESP, adjust);
        }
    }

    fn emit_abi_call_reg(&mut self, reg: u8) {
        self.emit_abi_call_reg_with_extra_pushes(reg, 0);
    }

    // assembler.py:2153 _call_footer — function epilogue

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    /// Thin wrapper around the free-fn `emit_call_footer_raw` so the
    /// backend-owned malloc trampoline can emit byte-identical epilogue
    /// sequences when exiting through `propagate_exception_descr`.
    fn _call_footer(&mut self) {
        emit_call_footer_raw(&mut self.mc);
        // `RET` ends the basic block.
        self.mc.forget_scratch_register();
    }

    /// x86/assembler.py:254 `_push_all_regs_to_jitframe` parity. Writes
    /// every managed GPR (and optionally XMM) into its canonical
    /// jitframe save slot so a subsequent collecting helper call can
    /// trace live Refs via the gcmap. Skips registers in `ignored_regs`
    /// (typically the slow-path's argument / result register, which
    /// holds non-Ref data across the call).
    ///
    /// Iterates `crate::x86::regalloc::ALL_CORE_REGS` / `ALL_FLOAT_REGS`
    /// (the allocator pool — drops R13 and XMM5..XMM14 on Win64), and
    /// indexes the slot via `core_reg_position` / `float_reg_position`,
    /// which look up positions in the same Win64-aware lists.  This
    /// mirrors `regalloc.py all_reg_indexes`, which is built from the
    /// post-`remove(r13)` `all_regs` on Win64 (so R14→10, R15→11).
    /// `_build_failure_recovery`, the `core_reg_index`-driven gcmap, and the
    /// post-call pop all consume positions through this same Win64-aware
    /// list, keeping the three in agreement.
    fn push_all_regs_to_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        push_all_regs_to_jitframe_raw(&mut self.mc, ignored_regs, withfloats, false);
    }

    /// x86/assembler.py:283 `_pop_all_regs_from_jitframe` parity.
    fn pop_all_regs_from_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        pop_all_regs_from_jitframe_raw(&mut self.mc, ignored_regs, withfloats, false);
    }

    /// `_check_frame_depth` — bridge entry. The fast path is
    /// `CMP_bi` plus a not-taken `jl` (`IncreaseStackSlowPath`,
    /// condition `L`). The body, emitted with the other slow paths,
    /// stores the depth and calls `build_frame_realloc_slowpath`.
    ///
    /// ```text
    ///   CMP QWORD [rbp + JF_FRAME_OFS + LENGTHOFS], 0xffffff
    ///                              ; → frame_depth_to_patch[]
    ///   JL   slow                  ; not taken when the frame is deep enough
    /// continue:
    /// ```
    ///
    /// The second `0xffffff` is `MOV_si(WORD)` in the slow body. Both
    /// immediates are in this compilation's buffer, so
    /// `patch_stack_checks` rewrites them together. Loops do not emit
    /// this check (`_check_frame_depth_debug` is not ported).
    fn emit_check_frame_depth(&mut self, gcmap: *mut usize) {
        let frame_len_ofs = (JF_FRAME_OFS + crate::jitframe::LENGTHOFS) as i32;
        let placeholder: i32 = 0xffffff;

        // `CMP_bi(ofs, 0xffffff)`. Dynasm emits `48 81 /7` with a
        // disp32; the 4-byte immediate is still the tail of the
        // instruction, so `offset - 4` is the patch site.
        dynasm!(self.mc ; .arch x64
            ; cmp QWORD [rbp + frame_len_ofs], placeholder
        );
        let cmp_imm_ofs = self.mc.offset().0 - 4;
        self.frame_depth_to_patch.push(cmp_imm_ofs);

        // `IncreaseStackSlowPath(mc, Conditions['L'])`. Not taken when
        // `[jf_frame.length]` is already >= the patched depth.
        let mut sp = self.emit_slow_jcc(
            CC_L,
            SlowPathKind::IncreaseStack {
                gcmap: gcmap as usize,
            },
        );
        self.set_continue_here(&mut sp);
        self.pending_slowpaths.push(sp);
    }

    /// x86/assembler.py:1422 `gen_shadowstack_header` parity (mirrors
    /// aarch64). Pushes two words onto the jitframe shadow stack on
    /// every JIT function entry: an `is_minor` marker (`1`) and the
    /// current jitframe pointer (rbp). The GC walks this stack during
    /// minor-collect to update jf pointers — without it, a minor GC
    /// inside a recursive call (e.g. fib_recursive) leaves rbp dangling
    /// at the freed nursery slot.
    ///
    /// `_call_header_shadowstack`: `_load_shadowstack_top_in_ebx`, then
    /// the two stores through `ebx`, which `_call_header` has saved.
    fn gen_shadowstack_header(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        let ebx = crate::regloc::EBX;
        self.mc.MOV(ebx, heap(rst)); // MOV ebx, [rootstacktop]
        // the '1' is to benefit from the shadowstack 'is_minor' optimization
        rx86::mov_mi(&mut self.mc, (rx86::EBX, 0), 1); // MOV [ebx], 1
        rx86::mov_mr(&mut self.mc, (rx86::EBX, WORD as i32), rx86::EBP); // MOV [ebx + WORD], ebp
        rx86::add_ri(&mut self.mc, rx86::EBX, (WORD * 2) as i32);
        self.mc.MOV(heap(rst), ebx); // MOV [rootstacktop], ebx
    }

    /// x86/assembler.py `_call_footer_shadowstack` parity:
    ///
    /// ```python
    /// if rx86.fits_in_32bits(rst):
    ///     self.mc.SUB_ji8(rst, WORD * 2)       # SUB [rootstacktop], 16
    /// else:
    ///     self.mc.MOV_ri(ebx.value, rst)       # MOV ebx, rootstacktop
    ///     self.mc.SUB_mi8((ebx.value, 0), WORD * 2)  # SUB [ebx], 16
    /// ```
    ///
    /// One in-memory subtract — no need to load the current top into a
    /// register, decrement, and store back.
    #[allow(dead_code)]
    fn gen_footer_shadowstack(&mut self) {
        emit_footer_shadowstack_raw(&mut self.mc);
        // `emit_footer_shadowstack_raw` writes `X86_64_SCRATCH_REG`.
        self.mc.forget_scratch_register();
    }

    /// `push_gcmap(store=True)`: `MOV(raw_stack(ofs), imm(gcmap))`. A
    /// 64-bit immediate goes through `insn_with_64_bit_immediate`, which
    /// `_load_scratch`es it and stores the scratch register.
    fn push_gcmap(&mut self, gcmap: *mut usize) {
        self.mc.MOV(Operand::B(JF_GCMAP_OFS), imm(gcmap as i64));
    }

    /// assembler.py:1000 pop_gcmap.
    fn pop_gcmap(&mut self) {
        rx86::mov_bi(&mut self.mc, JF_GCMAP_OFS, 0);
    }

    /// RPython `AbstractCallBuilder.emit`: CALL_ASSEMBLER is a collecting
    /// call, so the caller jitframe must publish the regalloc gcmap before
    /// entering the callee/helper and clear it only after reloading a possibly
    /// moved frame pointer.
    pub(crate) fn push_pending_call_gcmap(&mut self) -> bool {
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
            true
        } else {
            false
        }
    }

    pub(crate) fn pop_pending_call_gcmap_after_collect(&mut self, pushed: bool) {
        self.reload_frame_if_necessary();
        if pushed {
            self.pop_gcmap();
        }
    }

    /// x86/assembler.py `_reload_frame_if_necessary` parity:
    ///
    /// ```python
    ///   MOV ecx, [rootstacktop]   // shadow stack top pointer
    ///   MOV ebp, [ecx - WORD]     // jf_ptr at top - WORD
    ///   _write_barrier_fastpath(mc, wbdescr, [ebp], array=False,
    ///                           is_frame=True)
    /// ```
    ///
    /// After a collecting helper call the GC may have copied the
    /// jitframe from nursery to old gen. PyPy minor-GC does not write
    /// `jf_forward` on a move — that field is reserved for the
    /// `grow_jitframe` realloc path — so chasing `jf_forward` here
    /// reads the freed nursery slot. The shadow-stack entry IS
    /// rewritten by the GC visitor during copy, so the live jf_ptr
    /// lives at `*(root_stack_top - WORD)`. Reload `rbp` from there.
    ///
    /// Then re-apply the non-array write barrier on the new jitframe
    /// (`is_frame=True`): subsequent stores of nursery refs into
    /// jitframe slots must be tracked by minor GC, otherwise an
    /// old-gen jitframe holding a nursery pointer is missed during
    /// the next collection and the slot ends up dangling.
    fn reload_frame_if_necessary(&mut self) {
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        rx86::mov_ri(&mut self.mc, rx86::ECX, rst_addr);
        dynasm!(self.mc ; .arch x64
        ; mov rcx, [rcx]            // rcx = *rst_addr = root_stack_top
        );
        dynasm!(self.mc ; .arch x64
            ; mov rbp, [rcx - 8]        // rbp = *(top - WORD) = jf_ptr
        );
        // assembler.py:1378-1383 `_reload_frame_if_necessary` parity:
        //
        // ```python
        // wbdescr = self.cpu.gc_ll_descr.write_barrier_descr
        // if gcrootmap and wbdescr:
        //     # frame never uses card marking, so we enforce this is not
        //     # an array
        //     self._write_barrier_fastpath(mc, wbdescr, [ebp], array=False,
        //                                  is_frame=True)
        // ```
        //
        // After a collecting helper call the jitframe may have been
        // promoted from nursery to old-gen.  Subsequent stores of young
        // Refs into `[rbp + ofs]` would create old→young pointers
        // invisible to the GC — the WB fastpath re-arms the `TRACK_YOUNG_PTRS`
        // bit so the next collection scans the frame.  Reuses the shared
        // `emit_write_barrier_fastpath_kind` helper — `assembler.py:2388-2419`
        // expresses both `is_frame=True` and `is_frame=False` in a single
        // `_write_barrier_fastpath` whose addressing degenerates naturally
        // when `loc_base == ebp`.  `is_array=false` skips card marking
        // (assembler.py:2401 `if array and jit_wb_cards_set` gate), and
        // `is_frame=true` calls `wb_slowpath[4]`.
        if crate::runner::dynasm_write_barrier_descr().is_some() {
            let rbp_loc = Loc::Reg(crate::regloc::EBP);
            self.emit_write_barrier_fastpath_kind(None, &[rbp_loc], false, true);
        }
    }

    /// `llsupport/assembler.py GuardToken.compute_gcmap`: skip the hole
    /// a virtual leaves (`if arg is None: continue`), mark every remaining
    /// `REF`-typed failarg, narrow nothing else.  A force token is marked like
    /// any other — `resoperation.py FORCE_TOKEN/0/r` returns the jitframe,
    /// itself a moving GC object.
    ///
    /// The skip is carried by the location rather than the type: only a
    /// `Reg` or `Frame` location names a slot, so a fail arg that has neither
    /// contributes no bit.
    fn guard_gcmap_from_faillocs(
        &self,
        fail_arg_types: &[Type],
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) -> *mut usize {
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (tp, loc) in fail_arg_types.iter().zip(faillocs.iter()) {
            if *tp != Type::Ref {
                continue;
            }
            match loc {
                Some(Loc::Reg(r)) => {
                    if let Some(position) = reg_position_in_jitframe(*r) {
                        gcmap_set_bit(gcmap, position);
                    }
                }
                Some(Loc::Frame(f)) => {
                    gcmap_set_bit(gcmap, f.get_position() + JITFRAME_FIXED_SIZE);
                }
                None => {}
                Some(other) => panic!(
                    "guard_gcmap_from_faillocs: a Ref fail argument at {other:?} \
                carries no gcmap bit"
                ),
            }
        }
        gcmap
    }

    /// `x86/assembler.py fixup_target_tokens`.
    /// A LABEL assembled inside a bridge is a jump target for later traces exactly like a loop's.
    fn fixup_target_tokens(tokens: Vec<majit_ir::DescrRef>, frame_depth: usize, rawstart: usize) {
        for descr in tokens {
            if let Some(loop_descr) = descr.as_loop_target_descr() {
                // assembler.py:1003-1008 reads the target loop's
                // `frame_info.jfi_frame_depth` at a cross-loop JUMP; publish
                // this loop's full frame depth so a later trace's JUMP can
                // size its frame for this target.  Carries the depth grown by
                // `_assemble` for this loop's own onward JUMP.
                //
                // Store the companion `target_frame_depth` BEFORE the
                // `ll_loop_code` gate (both Release stores): a reader
                // Acquire-loads `ll_loop_code` and only reads
                // `target_frame_depth` once the gate is non-zero, so the depth
                // must become visible first — otherwise the reader pairs the
                // new code pointer with a stale 0 depth and bypasses the
                // frame-capacity check (descr.rs set_dispatch_target ordering
                // contract).  Dynasm ignores `label_block_id` (descr.rs —
                // it bakes the LABEL address straight into `ll_loop_code`), so
                // that companion is not published here.
                let old = loop_descr.ll_loop_code();
                let new = old + rawstart;
                loop_descr.set_target_frame_depth(frame_depth);
                loop_descr.set_ll_loop_code(new);
                if majit_ir::debug::have_debug_prints() {
                    majit_ir::debug::debug_print(&format!(
                        "[dynasm] fixup_target_tokens: ll_loop_code {old} -> {new:#x}"
                    ));
                }
            }
        }
    }

    fn check_unrelocated_jump_target(&self) -> Result<(), BackendError> {
        let Some((target, descr)) = self.unrelocated_jump_target else {
            return Ok(());
        };
        Err(BackendError::CompilationFailed(format!(
            "cross-buffer JUMP target {target:#x} for descr {descr:#x} is not a relocated executable address"
        )))
    }

    // assembler.py:501 assemble_loop

    /// assembler.py:501 assemble_loop: compile a loop trace.
    ///
    /// Returns compiled code with fail descriptors and entry point.
    pub fn assemble_loop(mut self) -> Result<CompiledCode, BackendError> {
        self.input_types = self.inputargs.iter().map(|ia| ia.tp.get()).collect();

        // assembler.py:537 prepare_loop — set up regalloc
        // For now, simplified: all args in frame slots

        // assembler.py:547 _assemble — generate code for all ops
        // Create a dynamic label at the entry point for self-recursive
        // CALL_ASSEMBLER (redirect_call_assembler parity).
        let entry_label = self.mc.new_dynamic_label();
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; =>entry_label);
        self.self_entry_label = Some(entry_label);
        let entry = self.mc.offset();
        // `regalloc.py prepare_loop`: 64-bit `redirect_call_assembler` writes
        // at most 13 bytes at `_ll_function_addr`. The first label must sit
        // past them.
        self.min_bytes_before_label = entry.0 + 13;
        self._assemble(true)?;
        self.check_unrelocated_jump_target()?;

        // regalloc sets fail_arg_locs in append_guard_token_with_faillocs,
        // which stamps `rd_locs` from them right there
        // (`llsupport/assembler.py:279`), so no post-regalloc fixup pass runs.

        // assembler.py:553 write_pending_failure_recoveries
        let stub_offsets = self.write_pending_failure_recoveries();
        // `materialize_loop` calls `datablockwrapper.done()` before the
        // code block is copied out. The constants are not in that stream.
        let data_blocks = self.datablockwrapper.done();

        // assembler.py:556 materialize_loop — finalize to executable memory
        let tokens = std::mem::take(&mut self.compiled_target_tokens);
        let frame_depth = self.frame_depth;
        self.emit_finish_descr_words();
        let mut buffer =
            codebuf::finalize_writable(self.mc.into_inner(), &self.asm_memory_manager)?;

        // assembler.py:849 patch_pending_failure_recoveries
        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);
        let invalidate_positions = Self::collect_invalidate_positions(rawstart, &stub_offsets);

        // assembler.py:556 patch_stack_checks — overwrite the 32-bit
        // `0xffffff` placeholders in any `_check_frame_depth` /
        // `_check_frame_depth_debug` emission with the loop's final
        // absolute frame depth (already includes `JITFRAME_FIXED_SIZE`).
        // No-op when the assembler did not emit a check (empty list).
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);

        // Write resolved entry address for self-recursive CALL_ASSEMBLER
        // trampoline. The JIT code loads from this pointer at runtime.
        unsafe { *self.self_entry_addr_ptr = rawstart + entry.0 };

        Self::fixup_target_tokens(tokens, frame_depth, rawstart);
        buffer.make_executable()?;

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: std::sync::Arc::new(self.fail_descrs),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: None,
            invalidate_positions,
            data_blocks,
        })
    }

    fn resolve_call_assembler_target_addr(
        &self,
        descr: Option<&majit_ir::DescrRef>,
    ) -> CallAssemblerTargetAddr {
        if let Some(token) = descr
            .and_then(|d| d.as_loop_token_descr())
            .and_then(|ltd| ltd.token_handle_any())
            .and_then(|any| any.downcast_ref::<Arc<JitCellToken>>())
        {
            let descr_addr = token.ll_function_addr();
            if descr_addr != 0 {
                return CallAssemblerTargetAddr {
                    immediate: Some(descr_addr),
                };
            }
            return CallAssemblerTargetAddr { immediate: None };
        }

        let _ = descr;
        CallAssemblerTargetAddr { immediate: None }
    }

    /// llsupport/assembler.py rebuild_faillocs_from_descr — reconstruct
    /// the locations of bridge inputargs from the guard's recovery layout.
    ///
    /// patch_jump_for_descr redirects the guard's jump into the bridge,
    /// so the register-save subroutine never runs.
    /// The bridge sees live registers exactly as they were at guard time.
    /// Return Reg locs for register positions, matching RPython.
    pub fn rebuild_faillocs_from_descr(
        descr: &dyn majit_ir::FailDescr,
        inputargs: &[InputArgRc],
    ) -> Vec<Loc> {
        let mut locs = Vec::new();
        let gpr_regs = crate::x86::regalloc::ALL_CORE_REGS;
        let float_regs = crate::x86::regalloc::ALL_FLOAT_REGS;
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        let mut input_i = 0usize;
        for &pos in descr.rd_locs() {
            if pos == 0xFFFF {
                continue;
            }
            let pos = pos as usize;
            if pos < gpr_regs.len() {
                // llsupport/assembler.py:211 — GPR: return register location
                locs.push(Loc::Reg(gpr_regs[pos]));
            } else if pos < gpr_regs.len() + float_regs.len() {
                // llsupport/assembler.py:213 — FPR: return float register
                locs.push(Loc::Reg(float_regs[pos - gpr_regs.len()]));
            } else {
                // llsupport/assembler.py:217 — frame slot
                let slot = pos - JITFRAME_FIXED_SIZE;
                let tp = inputargs
                    .get(input_i)
                    .map(|ia| ia.tp.get())
                    .unwrap_or(Type::Int);
                locs.push(Loc::Frame(crate::regloc::FrameLoc::new(
                    slot,
                    crate::regalloc::get_ebp_ofs(base_ofs, slot),
                    tp == Type::Float,
                )));
            }
            input_i += 1;
        }
        locs
    }

    /// assembler.py:623 assemble_bridge: compile a bridge trace.
    pub fn assemble_bridge(
        mut self,
        fail_descr: &dyn FailDescr,
        arglocs: &[Loc],
    ) -> Result<CompiledCode, BackendError> {
        self.input_types = self.inputargs.iter().map(|ia| ia.tp.get()).collect();
        self.bridge_input_locs = if arglocs.is_empty() {
            None
        } else {
            Some(arglocs.to_vec())
        };

        // `regalloc.py prepare_bridge`: `min_bytes_before_label = 0`.
        self.min_bytes_before_label = 0;
        let entry = self.mc.offset();
        self._assemble(false)?;
        self.check_unrelocated_jump_target()?;
        let stub_offsets = self.write_pending_failure_recoveries();
        let data_blocks = self.datablockwrapper.done();

        let tokens = std::mem::take(&mut self.compiled_target_tokens);
        let frame_depth = self.frame_depth;
        self.emit_finish_descr_words();
        let mut buffer =
            codebuf::finalize_writable(self.mc.into_inner(), &self.asm_memory_manager)?;

        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);
        let invalidate_positions = Self::collect_invalidate_positions(rawstart, &stub_offsets);

        // assembler.py:658 patch_stack_checks — same as the loop path,
        // applied with the bridge's own absolute frame depth.  Bridges
        // routinely grow the depth past the original loop value; the
        // patch rewrites the placeholder `0xffffff` immediate(s) so the
        // CMP at bridge entry reflects the bridge's true requirement.
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);
        Self::fixup_target_tokens(tokens, frame_depth, rawstart);
        buffer.make_executable()?;

        if crate::majit_dump_enabled() {
            let code = unsafe { std::slice::from_raw_parts(rawstart as *const u8, buffer.len()) };
            eprintln!(
                "[dynasm] BRIDGE CODE DUMP ({} bytes at {:#x}, entry +{:?}):",
                code.len(),
                rawstart,
                entry
            );
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{:08x} ", word);
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: std::sync::Arc::new(self.fail_descrs),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: Some((fail_descr.trace_id(), fail_descr.fail_index_per_trace())),
            invalidate_positions,
            data_blocks,
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    ///
    /// Uses the register allocator (regalloc.rs) to assign registers/frame
    /// locations, then emits code using those locations. This replaces the
    /// old frame-slot model where every value went through [rbp+offset].
    fn _assemble(&mut self, emit_prologue: bool) -> Result<(), BackendError> {
        let inputargs: &'a [InputArgRc] = self.inputargs;
        let ops: &'a [OpRc] = self.operations;
        self.unrelocated_jump_target = None;
        if emit_prologue {
            self._call_header(inputargs);
        } else {
            self.setup_input_state(inputargs);
        }
        let input_slot_depth = self.next_slot;

        // ── Run register allocator ──
        // assembler.py:537 prepare_loop / assembler.py:638 prepare_bridge
        if crate::majit_j2plan_log_enabled() {
            let plan = crate::j2plan::TracePlan::build(inputargs, ops);
            // Independent debug toggle — not gated by MAJIT_LOG.
            eprintln!("[dynasm:j2plan] {}", plan.summary());
        }

        // RegAlloc keeps the raw `i64` value map; project it from the
        // typed pool at this boundary (each Const carries its own type).
        let ra_constants: indexmap::IndexMap<u32, i64> = self
            .constants
            .iter()
            .map(|(&k, c)| (k, c.as_raw_i64()))
            .collect();
        let mut ra = RegAlloc::new(ra_constants, inputargs, ops);
        let is_bridge = self.bridge_input_locs.is_some();
        if let Some(ref arglocs) = self.bridge_input_locs {
            ra.prepare_bridge(arglocs);
        } else {
            ra.prepare_loop();
        }
        // assembler.py:647 — bridges emit `_check_frame_depth` between
        // `prepare_bridge` and `_update_at_exit` so the JIT can grow the
        // in-flight JITFRAME if the bridge's frame_depth exceeds the
        // loop's allocation.  Loops skip this (PyPy line 544 uses
        // `_check_frame_depth_debug`, a no-op outside DEBUG_FRAME_DEPTH).
        if is_bridge {
            let gcmap = ra.get_gcmap(&[], false);
            self.emit_check_frame_depth(gcmap);
        }
        if crate::majit_ops_log_enabled() {
            eprintln!("[dynasm] _assemble: {} ops", ops.len());
        }
        // assembler.py:374 walk_operations — the regalloc drives this
        // assembler op by op (`regalloc_perform*`). The XMM manager reaches
        // `assembler.datablockwrapper` during it, so the wrapper leaves
        // `self` for the walk instead of being reachable two ways.
        // `LocationCodeBuilder` drops `_scratch_register_value` when
        // `X86_64_SCRATCH_REG` is written and at the end of `CALL`/`JMP`
        // (`_relative_unaryop`), not before every op.
        self.fail_descrs = FailDescrStore::with_capacity(fail_cell_capacity(ops));
        self.fail_index = 0;
        let mut datablockwrapper = std::mem::replace(
            &mut self.datablockwrapper,
            MachineDataBlockWrapper::new(Arc::clone(&self.asm_memory_manager)),
        );
        let walked = ra.walk_operations_with_datablockwrapper(&mut datablockwrapper, self);
        self.datablockwrapper = datablockwrapper;
        walked?;
        // ra.get_final_frame_depth() returns a USER-position count; convert
        // to absolute by adding JITFRAME_FIXED_SIZE before comparing.
        let frame_slot_depth =
            input_slot_depth.max(JITFRAME_FIXED_SIZE + ra.get_final_frame_depth());
        self.frame_depth = self.frame_depth.max(frame_slot_depth);

        // Sync regalloc frame positions to opref_to_slot, the map `resolve_opref`
        // reads, once the walk is over. No live emitter reaches it any more: every remaining consumer
        // (`load_arg_to_rax` / `load_arg_to_rcx` / `resolve_const_or` and the
        // genops that call them) is `#[allow(dead_code)]`, kept for the upstream
        // method boundary. The map cannot express what the regalloc actually
        // decides — `before_call` leaves a value bound to a callee-saved member
        // of the allocation pool in its register, with no frame slot at all — so
        // an emitter that needs an operand location must take it from `arglocs`.
        // opref_to_slot stores ABSOLUTE jitframe slots (user position +
        // JITFRAME_FIXED_SIZE) so slot_offset(slot) gives the correct byte
        // offset without further adjustment.
        for (position, iarg) in inputargs.iter().enumerate() {
            self.opref_to_slot
                .insert(iarg.opref(), JITFRAME_FIXED_SIZE + position);
        }
        // Also sync any frame allocations from regalloc's FrameManager.
        for (&opref, lifetime) in ra.longevity.lifetimes_iter() {
            if let Some(floc) = lifetime.current_frame_loc {
                self.opref_to_slot
                    .insert(opref, JITFRAME_FIXED_SIZE + floc.get_position());
            }
        }
        // frame_slot_depth is already absolute (see calculation above).
        self.next_slot = frame_slot_depth;

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] _assemble done: pending_guard_tokens={} fail_index={}",
                self.pending_guard_tokens.len(),
                self.fail_index
            );
        }
        // Closes the last op's span, the way `--end of the loop--` closes a
        // `jit-log-opt` listing.
        if crate::majit_dump_enabled() {
            eprintln!("[dynasm] @{:#06x} end of ops", self.mc.offset().0);
        }
        // `regalloc.py` `_walk_operations` ends in `flush_loop`, so a
        // `GUARD_NOT_INVALIDATED` with nothing after it still has five
        // bytes before the recovery stubs.
        self.flush_loop();

        // assembler.py:1003-1008 `_assemble`: grow the frame to fit a
        // cross-loop JUMP target.  The closing `JMP` jumps into the target
        // loop's body, which can use deeper frame slots than this trace; for
        // a bridge the prologue `_check_frame_depth` then reallocs the live
        // JITFRAME to this grown depth on entry.  `target_frame_depth` is
        // full-width (includes JITFRAME_FIXED_SIZE), matching `frame_depth`,
        // so no adjustment is needed.  Zero (no external JUMP) is a no-op.
        self.frame_depth = self.frame_depth.max(self.jump_target_frame_depth);

        if self.pending_slowpaths.iter().any(|sp| {
            matches!(
                &sp.kind,
                SlowPathKind::CondCall {
                    expects_guard_no_exception: true,
                    guard_token_no_exception: None,
                    ..
                }
            )
        }) {
            panic!("a cond-call asked for GUARD_NO_EXCEPTION and the guard never stored the token");
        }

        Ok(())
    }

    /// assembler.py `genop_list[op.getopnum()]` — emit code for a non-guard
    /// op from its regalloc locations (`regalloc_perform`).
    fn genop_list(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        fail_index: u32,
        ops: &[OpRc],
    ) {
        match op.opcode {
            OpCode::IntAddOvf => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntSubOvf => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntMulOvf => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            // `_binaryop_or_lea(asmop='ADD', is_add=True)`, also bound as
            // `genop_nursery_ptr_increment`. ADD when `result_loc is
            // arglocs[0]`; otherwise `_consider_lea` produced a fresh
            // result and the form is `LEA result, [arglocs[0] + delta]`.
            OpCode::IntAdd | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(a0), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let same_as_lhs = matches!(a0, Loc::Reg(a) if a.value == dst.value);
                    if same_as_lhs {
                        self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    } else {
                        match (a0, src) {
                            (Loc::Reg(a), Loc::Immed(i) | Loc::ImmedFloat(i)) => {
                                // `_consider_lea` is guarded by
                                // `fits_in_32bits` of the immediate.
                                let v = i32::try_from(i.value).expect(
                                    "IntAdd LEA requires an immediate \
                                     encodable as signed disp32",
                                );
                                self.mc.LEA(*dst, Operand::M(a.value, i64::from(v)));
                            }
                            _ => panic!(
                                "IntAdd: result_loc != arglocs[0] requires LEA form \
                                 (arglocs[0]=Reg, arglocs[1]=Immed); got a0={a0:?} src={src:?}",
                            ),
                        }
                    }
                }
            }
            // x86/assembler.py `_binaryop_or_lea(asmop='SUB',
            // is_add=False)`: when `result_loc is arglocs[0]` emit
            // `SUB dst, src` in place; otherwise the regalloc routed
            // through `_consider_lea` (consider_int_sub at
            // x86/regalloc.py) and produced a fresh result register,
            // and we must emit `LEA result_loc, [arglocs[0] - delta]`
            // — never `SUB dst, src`, which would corrupt `dst`'s stale
            // value (the bug seen in fannkuch as `q.int_items.ptr - 1`
            // landing in a fresh result register that previously held
            // the base pointer).  IntMul / IntAnd / IntOr / IntXor never
            // take the LEA path (regalloc.rs routes them through
            // `consider_binop_symm` which keeps result==arglocs[0]), so
            // a plain in-place op is correct for them.
            OpCode::IntSub => {
                if let (Some(Loc::Reg(dst)), Some(a0), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let same_as_lhs = matches!(a0, Loc::Reg(a) if a.value == dst.value);
                    if same_as_lhs {
                        self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    } else {
                        match (a0, src) {
                            (Loc::Reg(a), Loc::Immed(i) | Loc::ImmedFloat(i)) => {
                                // regalloc.py — `_consider_lea` is guarded
                                // by `rx86.fits_in_32bits(-y.value)`. The
                                // wrapping negate matches PyPy `-y.value`;
                                // `y.value = 2147483648` is valid because
                                // disp32 = `-2147483648`. `i32::try_from`
                                // panics if the regalloc guard somehow let
                                // through a non-encodable value.
                                let v = i32::try_from(i.value.wrapping_neg()).expect(
                                    "IntSub LEA requires an immediate \
                                         encodable as signed disp32 after negation",
                                );
                                self.mc.LEA(*dst, Operand::M(a.value, i64::from(v)));
                            }
                            _ => panic!(
                                "IntSub: result_loc != arglocs[0] requires LEA form \
                                 (arglocs[0]=Reg, arglocs[1]=Immed); got a0={a0:?} src={src:?}",
                            ),
                        }
                    }
                }
            }
            OpCode::IntMul | OpCode::IntAnd | OpCode::IntOr | OpCode::IntXor => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                }
            }
            // ── Unary integer (result in arglocs[0] register) ──
            OpCode::IntNeg => {
                if let Some(Loc::Reg(r)) = result_loc {
                    self.mc.NEG(*r);
                }
            }
            OpCode::IntInvert => {
                if let Some(Loc::Reg(r)) = result_loc {
                    self.mc.NOT(*r);
                }
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                if let (Some(Loc::Reg(dst)), Some(shift_loc)) = (result_loc, arglocs.get(1)) {
                    // The count is an immediate or `ecx`; any other location
                    // is moved into `ecx` first.
                    let ecx = crate::regloc::ECX;
                    let count = match shift_loc {
                        Loc::Immed(i) | Loc::ImmedFloat(i) => imm(i64::from(i.value as i8)),
                        Loc::Reg(s) if *s == ecx => Operand::from(ecx),
                        _ => {
                            self.regalloc_mov(shift_loc, &Loc::Reg(ecx));
                            Operand::from(ecx)
                        }
                    };
                    match op.opcode {
                        OpCode::IntLshift => self.mc.SHL(*dst, count),
                        OpCode::IntRshift => self.mc.SAR(*dst, count),
                        OpCode::UintRshift => self.mc.SHR(*dst, count),
                        _ => {}
                    }
                }
            }
            // ── Integer comparisons ──
            // `_cmpop` + `flush_cc`. When the regalloc picks `frame_reg`
            // (rbp) as the result sentinel, the comparison's outcome lives
            // in the condition flags and the following guard consumes it.
            // Otherwise `flush_cc` materialises with `MOV_ri` 0 + `SET_ir`.
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                let mut opcode = op.opcode;
                if arglocs.len() >= 2 {
                    // `cmp` reaches an immediate only on its right, so a
                    // constant on the left is moved through a scratch
                    // register first. Swapping the operands folds it back
                    // into the instruction, and the order the swap destroys
                    // is carried by `resoperation.py`'s reflex table.
                    let (cmp0, cmp1) = if matches!(arglocs[0], Loc::Immed(_) | Loc::ImmedFloat(_))
                        && !matches!(arglocs[1], Loc::Immed(_) | Loc::ImmedFloat(_))
                        && let Some(reflexed) = op.opcode.bool_reflex()
                    {
                        opcode = reflexed;
                        (&arglocs[1], &arglocs[0])
                    } else {
                        (&arglocs[0], &arglocs[1])
                    };
                    self.emit_cmp_loc_loc(cmp0, cmp1);
                }
                let cc = Self::opcode_to_cc(opcode);
                self.flush_cc(cc, result_loc);
            }
            OpCode::IntIsTrue => {
                if let Some(src) = arglocs.first() {
                    self.emit_test_loc(src);
                    self.flush_cc(CC_NE, result_loc);
                }
            }
            OpCode::IntIsZero => {
                if let Some(src) = arglocs.first() {
                    self.emit_test_loc(src);
                    self.flush_cc(CC_E, result_loc);
                }
            }
            OpCode::UintMulHigh => {
                if let Some(Loc::Reg(dst)) = result_loc {
                    if let Some(src) = arglocs.first() {
                        if let Loc::Immed(_) | Loc::ImmedFloat(_) = src {
                            // Staged through the scratch register, which
                            // `INSN('MUL')` refuses, so the multiply is the
                            // raw `MUL_r` form.
                            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                            self.mc.MOV(Operand::R(scratch), src);
                            dynasm!(self.mc ; .arch x64 ; mul Rq(scratch));
                        } else {
                            self.mc.MUL(src);
                        }
                        self.mc.MOV(*dst, crate::regloc::EDX);
                    }
                }
            }
            OpCode::IntForceGeZero => {
                // genop_int_force_ge_zero: TEST src, src; MOV res, 0; CMOVNS res, src.
                // consider_int_force_ge_zero forbids the result from aliasing src:
                // MOV 0 would otherwise clobber the value CMOVNS reads.
                let (Some(Loc::Reg(src)), Some(Loc::Reg(res))) = (arglocs.first(), result_loc)
                else {
                    panic!(
                        "int_force_ge_zero: expected a register source and result, \
                         got arglocs={arglocs:?} result={result_loc:?}"
                    );
                };
                if src == res {
                    panic!("int_force_ge_zero: result register aliases the source");
                }
                self.mc.TEST(*src, *src);
                // mov(): MOV of 0 is MOV_riu32 (zero-extending), and it does not
                // clobber the flags TEST just set.
                self.mc.MOV(*res, imm(0));
                self.mc.CMOVNS(*res, *src);
            }
            OpCode::IntSignext => {
                // x86/assembler.py `genop_int_signext`: numbytes is an immediate,
                // and 1/2/4 select MOVSX8 / MOVSX16 / MOVSX32 (movsxd).
                let (Some(argloc), Some(numbytes_loc), Some(Loc::Reg(res))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                else {
                    panic!(
                        "int_signext: expected [arg, numbytes] and a register result, \
                         got arglocs={arglocs:?} result={result_loc:?}"
                    );
                };
                let (Loc::Immed(numbytes) | Loc::ImmedFloat(numbytes)) = numbytes_loc else {
                    panic!("int_signext: numbytes loc must be an immediate, got {numbytes_loc:?}");
                };
                match numbytes.value {
                    1 => self.mc.MOVSX8(*res, argloc),
                    2 => self.mc.MOVSX16(*res, argloc),
                    4 => self.mc.MOVSX32(*res, argloc),
                    _ => panic!("bad number of bytes"),
                }
            }
            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                // `_binaryop("ADDSD"|"SUBSD"|"MULSD"|"DIVSD")`: the instruction
                // reads `arglocs[1]` in place. `_consider_float_op` leaves that
                // operand as `xrm.loc` (register, frame, or `ConstFloatLoc`).
                if let (Some(dst_loc), Some(src_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let Loc::Reg(dst) = *dst_loc else {
                        panic!(
                            "float binop arglocs[0] must be a register \
                             (_consider_float_op force_result_in_reg), got {dst_loc:?}"
                        );
                    };
                    let src = *src_loc;
                    match op.opcode {
                        OpCode::FloatAdd => self.mc.ADDSD(dst, &src),
                        OpCode::FloatSub => self.mc.SUBSD(dst, &src),
                        OpCode::FloatMul => self.mc.MULSD(dst, &src),
                        OpCode::FloatTrueDiv => self.mc.DIVSD(dst, &src),
                        _ => {}
                    }
                }
            }
            OpCode::FloatNeg => {
                // genop_float_neg: XORPD against heap(float_const_neg_addr).
                // The sign mask from _build_float_constants flips ±0.0 and a NaN sign.
                if let Some(Loc::Reg(r)) = result_loc {
                    let addr = &FLOAT_CONST_NEG as *const AlignedPdConst as i64;
                    self.mc.XORPD(*r, heap(addr));
                }
            }
            OpCode::FloatAbs => {
                // genop_float_abs: ANDPD against heap(float_const_abs_addr).
                // The mask clears the sign bit, so -0.0 becomes +0.0 and a NaN payload stays.
                if let Some(Loc::Reg(r)) = result_loc {
                    let addr = &FLOAT_CONST_ABS as *const AlignedPdConst as i64;
                    self.mc.ANDPD(*r, heap(addr));
                }
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                if let (Some(a_loc), Some(b_loc)) = (arglocs.first(), arglocs.get(1)) {
                    // `_cmpop_float`: `need_direct_p = 'A' not in cond` and
                    // `need_rev_p = 'A' not in rev_cond` (substring, so 'BE'
                    // does not contain 'A' and 'AE' does). The chosen UCOMISD
                    // keeps a non-register source; `_if_parity_clear_zero_and_carry`
                    // runs only when `need_p` is set.
                    let (cond, rev_cond, need_direct_p, need_rev_p) = match op.opcode {
                        OpCode::FloatLt => (CC_B, CC_A, true, false),
                        OpCode::FloatLe => (CC_BE, CC_AE, true, false),
                        OpCode::FloatEq => (CC_E, CC_E, true, true),
                        OpCode::FloatNe => (CC_NE, CC_NE, true, true),
                        OpCode::FloatGt => (CC_A, CC_B, false, true),
                        OpCode::FloatGe => (CC_AE, CC_BE, false, true),
                        _ => unreachable!("float compare opcode"),
                    };
                    let direct_case = if need_direct_p {
                        !b_loc.is_reg()
                    } else {
                        a_loc.is_reg()
                    };
                    let (lhs, rhs, checkcond, need_p) = if direct_case {
                        (*a_loc, *b_loc, cond, need_direct_p)
                    } else {
                        (*b_loc, *a_loc, rev_cond, need_rev_p)
                    };
                    let Loc::Reg(dst) = lhs else {
                        panic!(
                            "UCOMISD first operand must be RegLoc (_cmpop_float / \
                             _consider_float_cmp), got {lhs:?}"
                        );
                    };
                    self.mc.UCOMISD(dst, &rhs);
                    if need_p {
                        self.emit_if_parity_clear_zero_and_carry();
                    }
                    // `genop_cmp_float` ends in `flush_cc`, so a comparison
                    // whose only consumer is the next guard keeps its answer
                    // in the flags.
                    self.flush_cc(checkcond, result_loc);
                }
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let sr = match src {
                        Loc::Reg(s) => s.value,
                        _ => {
                            self.regalloc_mov(
                                src,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    // cvtsi2sd preserves the destination's high 64 bits, so it
                    // carries a false dependency on the register's prior value.
                    // In a loop that reuses the same xmm, this serialises the
                    // conversion against the previous iteration. Break it with a
                    // zeroing idiom (eliminated at register rename, zero latency).
                    rx86::pxor_xx(&mut self.mc, dst.value, dst.value);
                    rx86::cvtsi2sd_xr(&mut self.mc, dst.value, sr);
                }
            }
            // `genop_cast_float_to_int`: `CVTTSD2SI(resloc, arglocs[0])`.
            // rx86 has only `CVTTSD2SI_rx` / `CVTTSD2SI_rb`; any other
            // source is `_missing_binary_insn`.
            OpCode::CastFloatToInt => {
                let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) else {
                    panic!(
                        "CAST_FLOAT_TO_INT expects a source and a register result, \
                         got {arglocs:?} -> {result_loc:?}"
                    );
                };
                self.mc.CVTTSD2SI(*dst, src);
            }
            // `genop_cast_float_to_singlefloat`: CVTSD2SS into the xmm temp,
            // then MOVD32_rx into the GPR. The f32 bits occupy the low 32.
            OpCode::CastFloatToSinglefloat => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(tmp)), Some(Loc::Reg(dst))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                {
                    self.mc.CVTSD2SS(*tmp, *src);
                    self.mc.MOVD32(*dst, *tmp);
                }
            }
            // `genop_cast_singlefloat_to_float`: MOVD32_xr, then CVTSS2SD
            // of the result into itself.
            OpCode::CastSinglefloatToFloat => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    self.mc.MOVD32(*dst, *src);
                    self.mc.CVTSS2SD(*dst, *dst);
                }
            }
            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::CastOpaquePtr
            | OpCode::VirtualRefR
            | OpCode::ConvertFloatBytesToLonglong
            | OpCode::ConvertLonglongBytesToFloat => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // `assembler.py genop_load_from_gc_table`: load the reference
            // constant from slot `index` of the table reserved at the start
            // of this code block (`reserve_gcref_table`), PC-relative. The
            // gc_table root walker forwards the slot in place, so each load
            // observes the relocated object.
            OpCode::LoadFromGcTable => {
                let (Some(Loc::Immed(idx) | Loc::ImmedFloat(idx)), Some(Loc::Reg(dst))) =
                    (arglocs.first(), result_loc)
                else {
                    panic!(
                        "LoadFromGcTable expects [Immed(index)] and a register result, \
                         got arglocs={arglocs:?} result={result_loc:?}"
                    );
                };
                let slot = *self
                    .gcref_table
                    .get(idx.value as usize)
                    .expect("LoadFromGcTable index inside the reserved gcref table");
                dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [=>slot]);
            }
            // `assembler.py genop_cast_ptr_to_int = _genop_same_as`
            // / `:1529 genop_cast_int_to_ptr = _genop_same_as`.  PyPy's
            // x86 backend treats both casts as plain `mov` — the
            // AddressAsInt low-bit tag is a `blackhole.py bhimpl_cast_ptr_to_int`
            // interpreter-side software invariant, not a backend
            // codegen step.  Tagging at codegen would fold a fake odd
            // pointer back into the raw aligned-pointer space and
            // could collide with a real GC pointer.  `runner_test.py:
            // 1957 cast_int_to_ptr(-17) -> cast_ptr_to_int == -17`
            // expects strict identity through the compiled trace.
            OpCode::CastPtrToInt | OpCode::CastIntToPtr => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // ── Memory loads: getfield pattern ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen => {
                // regalloc.py `_consider_gc_load` parity: both
                // `base_loc = self.rm.make_sure_var_in_reg(op.getarg(0), args)`
                // and `result_loc = self.force_allocate_reg(op)` force
                // register materialisation — pyre's
                // `consider_getfield` (x86/regalloc.rs) does the
                // same.  Silently no-op'ing on a non-Reg base or result
                // would mask a regalloc bug (e.g. a fresh `GETFIELD_GC`
                // arm that forgot the `make_sure_var_in_reg` call), so
                // surface the invariant violation explicitly.
                let base = match arglocs.first() {
                    Some(Loc::Reg(r)) => *r,
                    other => panic!(
                        "GetfieldGc/Strlen/Unicodelen/ArraylenGc base must be Loc::Reg \
                         (regalloc.py:1156 make_sure_var_in_reg invariant), got {other:?}",
                    ),
                };
                let dst = match result_loc {
                    Some(Loc::Reg(r)) => *r,
                    other => panic!(
                        "GetfieldGc/Strlen/Unicodelen/ArraylenGc result_loc must be Loc::Reg \
                         (regalloc.py:1158 force_allocate_reg invariant), got {other:?}",
                    ),
                };
                let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                // A 4-byte field is sign-extended whatever its descr says.
                let sign = field_size == 4;
                self.load_from_mem(
                    dst,
                    Operand::M(base.value, i64::from(ofs)),
                    field_size,
                    sign,
                );
            }
            // ── Memory loads: getarrayitem pattern ──
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => {
                if let (Some(base_loc), Some(index_loc), Some(Loc::Reg(dst))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                {
                    let (base_size, item_size, signed) = op
                        .with_array_descr(|ad| {
                            (
                                ad.base_size() as i32,
                                ad.item_size() as i32,
                                op.opcode.result_type() == Type::Int && ad.is_item_signed(),
                            )
                        })
                        .unwrap_or((0, 8, false));
                    // A non-register base (a constant pointer from the green-pc
                    // inline dispatch reading `program[const]`, or a spilled
                    // Frame slot) is staged into rax below. Relocate an index
                    // already occupying rax so that staging cannot clobber it.
                    let base_in_reg = matches!(base_loc, Loc::Reg(_));
                    let index_reg = match index_loc {
                        Loc::Reg(r) if base_in_reg || r.value != crate::regloc::EAX.value => {
                            r.value
                        }
                        _ => {
                            self.regalloc_mov(
                                index_loc,
                                &Loc::Reg(crate::regloc::X86_64_SCRATCH_REG),
                            );
                            crate::regloc::X86_64_SCRATCH_REG.value
                        }
                    };
                    if item_size != 1 {
                        // `index_reg` may be the scratch register it was just
                        // `MOV`ed into, which `INSN('IMUL')` refuses.
                        rx86::imul_ri(&mut self.mc, index_reg, item_size);
                    }
                    // Resolve the base into a register. A Loc::Immed base
                    // (constant array pointer) previously failed the Loc::Reg
                    // pattern, so the address computation and load were skipped
                    // and dst kept a stale value (mirrors the GcLoad
                    // immediate-base handling below).
                    let base_reg = match base_loc {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(base_loc, &Loc::Reg(crate::regloc::EAX));
                            crate::regloc::EAX.value
                        }
                    };
                    if base_size != 0 {
                        rx86::lea_ra(
                            &mut self.mc,
                            rx86::EAX,
                            (i16::from(base_reg), index_reg, 0, base_size),
                        );
                    } else {
                        rx86::lea_ra(
                            &mut self.mc,
                            rx86::EAX,
                            (i16::from(base_reg), index_reg, 0, 0),
                        );
                    }
                    let src = Operand::M(rx86::EAX, 0);
                    self.load_from_mem(*dst, src, item_size as usize, signed);
                }
            }
            // ── Memory stores: opassembler.rs emit_op_setfield_regalloc ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                // `rewrite.rs transform_to_gc_load` lowers both opcodes to
                // GC_STORE / GC_STORE_INDEXED through `emit_gc_store_or_indexed`
                // before the regalloc ever sees them, and `consider_setfield`
                // force-allocates a register for every non-constant base, so the
                // only shape that could reach the else arm is a constant base on
                // an op the rewriter did not consume.  Emitting it from
                // `resolve_opref` would read whichever slot the lifetime's
                // end-of-allocation `current_frame_loc` happens to name and
                // clobber rax/rcx behind the regalloc's back; decline instead.
                let [base_loc, val_loc] = arglocs else {
                    panic!(
                        "{:?} expects two regalloc locations, got {arglocs:?}",
                        op.opcode,
                    );
                };
                let Loc::Reg(base) = base_loc else {
                    panic!(
                        "{:?} reached the backend with a non-register base {base_loc:?} \
                         — `rewrite.rs transform_to_gc_load` must have consumed it",
                        op.opcode,
                    );
                };
                let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                self.emit_op_setfield_regalloc(base, val_loc, ofs, field_size);
            }
            // arglocs = [base_loc, ofs_loc, res_loc, imm(nsize)].
            // `base_loc` may be Loc::Immed when the load is from a
            // constant pointer (e.g. `GcLoadI(jfi_descr_ptr, 8, 8)` for
            // the JITFRAME size in CallMallocNurseryVarsizeFrame).
            // llsupport/regalloc.py return_constant returns the
            // bare Loc::Immed in that case and the assembler is
            // responsible for materializing it. Mirror aarch64 by
            // staging the constant through the scratch register (R11);
            // dropping the load left the destination register holding
            // stale heap pointers, which the varsize-frame slowpath
            // then dereferenced as an allocation size and tripped a
            // multi-terabyte OOM (fib_recursive on x86).
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let Some(ofs_loc) = arglocs.get(1) {
                    let dst = match arglocs.get(2) {
                        Some(Loc::Reg(r)) => r,
                        _ => match result_loc {
                            Some(Loc::Reg(r)) => r,
                            _ => return,
                        },
                    };
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                        _ => op
                            .with_array_descr(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    match arglocs.first() {
                        Some(Loc::Reg(base)) => {
                            self.emit_op_gcload_regalloc(&Loc::Reg(*base), ofs_loc, dst, nsize);
                        }
                        Some(Loc::Immed(base_i) | Loc::ImmedFloat(base_i)) => {
                            let scratch = crate::regloc::X86_64_SCRATCH_REG;
                            // regalloc.rs materializes out-of-range
                            // offsets into LARGE_IMM_SCRATCH (R11). If
                            // we land in that case while base is also
                            // an immediate we cannot share R11.
                            if let Loc::Reg(r) = ofs_loc {
                                assert!(
                                    r.value != scratch.value,
                                    "GcLoad: base=Immed and ofs already occupies R11",
                                );
                            }
                            // `addr_add(base_loc, ofs_loc)` on an immediate
                            // base is an absolute `'j'` address, or an `'a'`
                            // with the base as its static offset.
                            let base_loc = Loc::Immed(*base_i);
                            self.emit_op_gcload_regalloc(&base_loc, ofs_loc, dst, nsize);
                        }
                        other => {
                            panic!("GcLoad base_loc must be Loc::Reg or Loc::Immed, got {other:?}",)
                        }
                    }
                }
            }
            // ── x86/assembler.py genop_discard_gc_store ──
            // arglocs = [value_loc, base_loc, ofs_loc, size_loc].
            // value_loc may be Loc::Immed when the source is a Const
            // (llsupport/regalloc.py return_constant), so the emitter
            // must materialize it before the store — silently dropping
            // such writes left newly-allocated objects without vtables
            // and triggered downstream GuardClass failures.
            OpCode::GcStore | OpCode::RawStore => {
                let (value_loc, base_loc, ofs_loc, size_loc) = match arglocs {
                    [v, b, o, s] => (v, b, o, s),
                    _ => panic!(
                        "GcStore arglocs must be [value, base, ofs, size] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcStore base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                let size = match size_loc {
                    Loc::Immed(i) | Loc::ImmedFloat(i) => i.value.unsigned_abs() as usize,
                    other => panic!(
                        "GcStore size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                // `save_into_mem` stores an immediate as an integer move. A
                // singlefloat bit pattern is int-kind (`get_type_flag` →
                // FLAG_UNSIGNED), not an xmm `MOVSS`.
                let dest_addr = addr_add(&Loc::Reg(*base), ofs_loc, 0, 0);
                self.save_into_mem(dest_addr, value_loc, size);
            }
            // ── x86/assembler.py genop_discard_gc_store_indexed ──
            // `base_loc, ofs_loc, value_loc, factor_loc, offset_loc, size_loc = arglocs`.
            // `dest_addr = AddressLoc(base_loc, ofs_loc, scale=get_scale(factor_loc.value), disp=offset_loc.value)`
            // emits `[base + ofs * 2**scale + disp]`. `load_supported_factors =
            // (1, 2, 4, 8)` (x86/runner.py:31), so the rewriter passes raw
            // byte strides in that set straight through here and the native
            // SIB scaled-index addressing does the multiply. Any other factor
            // must have been pre-scaled away in `cpu_simplify_scale`.
            OpCode::GcStoreIndexed => {
                if let (Some(Loc::Reg(base)), Some(Loc::Reg(ofs_reg)), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let factor = match arglocs.get(3) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
                        _ => 1,
                    };
                    let offset = match arglocs.get(4) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value as i32,
                        _ => 0,
                    };
                    let size = match arglocs.get(5) {
                        Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value.unsigned_abs() as usize,
                        _ => 8,
                    };

                    let scale = match factor {
                        1 => 0,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        other => panic!(
                            "x86 GcStoreIndexed: unsupported factor {other}; \
                             load_supported_factors = (1, 2, 4, 8)"
                        ),
                    };
                    let dest_addr = addr_add(
                        &Loc::Reg(*base),
                        &Loc::Reg(*ofs_reg),
                        i64::from(offset),
                        scale,
                    );
                    self.save_into_mem(dest_addr, value_loc, size);
                }
            }
            // ── x86/assembler.py _genop_gc_load_indexed ──
            // Line-by-line port:
            //   base_loc, ofs_loc, scale_loc, offset_loc, size_loc, sign_loc = arglocs
            //   scale = get_scale(scale_loc.value)
            //   src_addr = addr_add(base_loc, ofs_loc, offset_loc.value, scale)
            //   self.load_from_mem(resloc, src_addr, size_loc, sign_loc)
            //
            // The regalloc passes the raw byte stride in `scale_loc`
            // (x86/regalloc.py:1184); `get_scale` converts 1/2/4/8 →
            // 0/1/2/3 SIB exponents. PyPy keeps the (1,2,4,8) check
            // implicit through `valid_addressing_size`; we surface the
            // unsupported factors as a panic to keep miscompiles loud.
            //
            // KNOWN ISSUE: this emission triggers a timing-dependent
            // segfault on `spectral_norm`-style traces (function call +
            // many iterations + later loop). Adding any eprintln in the
            // dispatch hides the bug, so it is likely a stale
            // base-array pointer surviving a minor GC during the inline
            // jitframe-alloc fast path.
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                let (base_loc, ofs_loc, scale_loc, offset_loc, size_loc, sign_loc) = match arglocs {
                    [b, o, sc, of, sz, sg] => (b, o, sc, of, sz, sg),
                    _ => panic!(
                        "GcLoadIndexed arglocs must be [base, ofs, scale, offset, size, sign] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcLoadIndexed base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                let ofs_reg = match ofs_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcLoadIndexed ofs_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                let factor = match scale_loc {
                    Loc::Immed(i) | Loc::ImmedFloat(i) => i.value,
                    other => panic!(
                        "GcLoadIndexed scale_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let offset = match offset_loc {
                    Loc::Immed(i) | Loc::ImmedFloat(i) => i.value as i32,
                    other => panic!(
                        "GcLoadIndexed offset_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let size = match size_loc {
                    Loc::Immed(i) | Loc::ImmedFloat(i) => i.value as usize,
                    other => panic!(
                        "GcLoadIndexed size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let sign = match sign_loc {
                    Loc::Immed(i) | Loc::ImmedFloat(i) => i.value != 0,
                    other => panic!(
                        "GcLoadIndexed sign_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let dst = match result_loc {
                    Some(Loc::Reg(r)) => r,
                    other => panic!("GcLoadIndexed result_loc must be Loc::Reg, got {other:?}",),
                };

                let shift = match factor {
                    1 => 0,
                    2 => 1,
                    4 => 2,
                    8 => 3,
                    other => panic!(
                        "x86 GcLoadIndexed: unsupported factor {other}; \
                         load_supported_factors = (1, 2, 4, 8)"
                    ),
                };
                // `addr_add(base_loc, ofs_loc, offset, scale)`.
                let src_addr = addr_add(
                    &Loc::Reg(*base),
                    &Loc::Reg(*ofs_reg),
                    i64::from(offset),
                    shift,
                );
                self.load_from_mem(*dst, src_addr, size, sign);
            }
            // `rewrite.py transform_to_gc_load` lowers SETARRAYITEM_GC to
            // GC_STORE_INDEXED (after `handle_write_barrier_setarrayitem`)
            // for every compiled loop, so the backend has no handler for it.
            OpCode::SetarrayitemGc => {
                panic!("dynasm: SetarrayitemGc must have been lowered by rewrite_ops_for_gc");
            }
            OpCode::SetarrayitemRaw => {
                if let (Some(Loc::Reg(base)), Some(index_loc), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let (base_size, item_size) = op
                        .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
                        .unwrap_or((0, 8));
                    let index_reg = crate::regloc::X86_64_SCRATCH_REG.value;
                    // r11 is already the computed destination address here.
                    // Stage non-register values through a saved GPR: r12 is
                    // allocatable on x86-64, so using SCRATCH_REG_2 would
                    // clobber live regalloc state.
                    let value_reg = crate::regloc::EAX.value;

                    self.regalloc_mov(
                        index_loc,
                        &Loc::Reg(crate::regloc::RegLoc::new(index_reg, false)),
                    );
                    if item_size != 1 {
                        // `index_reg` may be the scratch register it was just
                        // `MOV`ed into, which `INSN('IMUL')` refuses.
                        rx86::imul_ri(&mut self.mc, index_reg, item_size);
                    }
                    if base_size != 0 {
                        rx86::add_ri(&mut self.mc, index_reg, base_size);
                    }
                    dynasm!(self.mc ; .arch x64 ; add Rq(index_reg), Rq(base.value));

                    match value_loc {
                        Loc::Reg(val) if val.is_xmm => {
                            rx86::movsd_mx(&mut self.mc, (index_reg, 0), val.value);
                        }
                        Loc::Reg(val) => match item_size {
                            1 => rx86::mov8_mr(&mut self.mc, (index_reg, 0), val.value),
                            2 => rx86::mov16_mr(&mut self.mc, (index_reg, 0), val.value),
                            4 => rx86::mov32_mr(&mut self.mc, (index_reg, 0), val.value),
                            _ => rx86::mov_mr(&mut self.mc, (index_reg, 0), val.value),
                        },
                        _ => {
                            dynasm!(self.mc ; .arch x64 ; push rax);
                            self.regalloc_mov(
                                value_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(value_reg, false)),
                            );
                            match item_size {
                                1 => rx86::mov8_mr(&mut self.mc, (index_reg, 0), value_reg),
                                2 => rx86::mov16_mr(&mut self.mc, (index_reg, 0), value_reg),
                                4 => rx86::mov32_mr(&mut self.mc, (index_reg, 0), value_reg),
                                _ => rx86::mov_mr(&mut self.mc, (index_reg, 0), value_reg),
                            }
                            dynasm!(self.mc ; .arch x64 ; pop rax);
                        }
                    }
                }
            }
            // ── Control flow ──
            OpCode::Jump => {
                let descr_arc = op.getdescr();
                let jump_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                let target_arglocs = jump_descr
                    .map(|descr| {
                        descr
                            .target_arglocs()
                            .into_iter()
                            .map(|loc| loc_from_target_argloc(&loc))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let mut src_locations1 = Vec::new();
                let mut dst_locations1 = Vec::new();
                let mut src_locations2 = Vec::new();
                let mut dst_locations2 = Vec::new();
                // x86/regalloc.py:1287: assert len(arglocs) == jump_op.numargs()
                // RPython enforces arity equality at regalloc time;
                // the assembler never sees surplus args.
                let remap_count = if target_arglocs.is_empty() {
                    arglocs.len()
                } else {
                    assert_eq!(
                        arglocs.len(),
                        target_arglocs.len(),
                        "JUMP args ({}) != target LABEL args ({})",
                        arglocs.len(),
                        target_arglocs.len(),
                    );
                    target_arglocs.len()
                };
                for (i, src_loc) in arglocs[..remap_count].iter().enumerate() {
                    // `consider_jump`: the set comes from `box.type == FLOAT`,
                    // not from the location. Set 2 carries the floats and
                    // gets the float scratch. When the destination has to be
                    // synthesized below, the same answer is the slot's kind:
                    // `regalloc_push` / `regalloc_pop` read it to choose
                    // their scratch register, and `loc_width` for the width.
                    let is_float = op.arg(i).to_opref().ty() == Some(Type::Float);
                    let dst_loc = if i < target_arglocs.len() {
                        target_arglocs[i]
                    } else {
                        // The canonical base for a frame position, the
                        // one `FrameManager` was built with
                        // (`get_baseofs_of_frame_field`). Passing 0 here named
                        // a slot `FIRST_ITEM_OFFSET` bytes below the one the
                        // regalloc means by the same position, so a source and
                        // this destination could denote the same value and
                        // different storage.
                        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
                        let dst_ofs = crate::regalloc::get_ebp_ofs(base_ofs, i);
                        Loc::Frame(crate::regloc::FrameLoc::new(i, dst_ofs, is_float))
                    };
                    let (srcs, dsts) = if is_float {
                        (&mut src_locations2, &mut dst_locations2)
                    } else {
                        (&mut src_locations1, &mut dst_locations1)
                    };
                    srcs.push(*src_loc);
                    dsts.push(dst_loc);
                }
                let tmpreg1 = Loc::Reg(crate::regloc::X86_64_SCRATCH_REG);
                let tmpreg2 = Loc::Reg(crate::regloc::X86_64_XMM_SCRATCH_REG);
                if majit_ir::debug::have_debug_prints() {
                    let _s = majit_ir::debug::scope("jit-backend");
                    majit_ir::debug::debug_print(&format!(
                        "Jump remap: {} int src→dst, {} float src→dst",
                        src_locations1.len(),
                        src_locations2.len()
                    ));
                    for (i, (s, d)) in src_locations1.iter().zip(dst_locations1.iter()).enumerate()
                    {
                        majit_ir::debug::debug_print(&format!("  int[{i}]: {s:?} → {d:?}"));
                    }
                }
                crate::jump::remap_frame_layout_mixed(
                    self,
                    &src_locations1,
                    &dst_locations1,
                    tmpreg1,
                    &src_locations2,
                    &dst_locations2,
                    tmpreg2,
                );
                if let Some(label) = loop_target_id(op)
                    .and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
                {
                    dynasm!(self.mc ; .arch x64 ; jmp =>label);
                    self.mc.forget_scratch_register();
                } else if let (Some(descr_ref), Some(descr)) = (descr_arc.as_ref(), jump_descr) {
                    let target = descr.ll_loop_code();
                    // External JUMP: direct JMP to target loop code.
                    // `closing_jump` is `JMP(imm(target))`.
                    // `LocationCodeBuilder._relative_unaryop` stages a
                    // 64-bit absolute target through `X86_64_SCRATCH_REG`
                    // (r11), never RAX. Using RAX clobbers a loop-carried Ref
                    // the regalloc bound there: a bridge that rejoins the loop
                    // would return with RAX = `target`, and the next
                    // GuardClass(RAX) would read that code address as a Ref.
                    // `_assemble` records the target loop's frame depth so
                    // this trace's frame grows to fit it. A target below
                    // `MIN_RELOCATED_JUMP_TARGET` is still an in-buffer offset
                    // or zero; recording it fails the compile instead of
                    // baking a wild branch.
                    if target < MIN_RELOCATED_JUMP_TARGET {
                        self.unrelocated_jump_target =
                            Some((target, majit_ir::descr_identity(descr_ref)));
                    } else {
                        self.jump_target_frame_depth =
                            self.jump_target_frame_depth.max(descr.target_frame_depth());
                        self.mc.JMP(imm(target as i64));
                    }
                }
            }
            OpCode::Finish => {
                // RPython: genop_finish stores result at jf_frame[0] (base_ofs),
                // writes descr ptr to jf_descr, then calls _call_footer.
                // arglocs[0] = result location (if any)
                let fail_arg_types = self.infer_fail_arg_types(op, Some(op_index));
                let result_type = if fail_arg_types.is_empty() {
                    Type::Void
                } else {
                    fail_arg_types[0]
                };
                // `compile.py` ExitFrameWithExceptionDescrRef identity:
                // route to the metainterp `exit_frame_with_exception_descr_ref`
                // when the FINISH was emitted for
                // `pyjitpl.py compile_exit_frame_with_exception`.  The
                // runtime classifier (`runner.rs::find_descr_by_ptr`) then
                // dispatches into `jitexc.ExitFrameWithExceptionRef` rather
                // than `jitexc.DoneWithThisFrame*`.
                let is_exit_exc = op
                    .with_fail_descr(|fd| fd.is_exit_frame_with_exception())
                    .unwrap_or(false);
                let global_descr_ptr = if is_exit_exc {
                    self.exit_frame_with_exception_descr_ref_ptr()
                } else {
                    self.done_with_this_frame_descr_ptr_for_type(result_type)
                };
                // FINISH op exit (DoneWithThisFrame* / ExitFrameWithExceptionDescr).
                // `compile.py` skips these — not a `ResumeDescr`.
                // `genop_finish` loads the descr with rip-relative `MOV_rp`
                // and stores it at `jf_descr`. The word is the cpu-attached
                // singleton: `find_descr_by_ptr` matches that cell before the
                // registry, so a gc-table index would name a different
                // pointer. `fail_descrs` keeps the Arc alive. Test scaffolds
                // must attach singletons (`attach_default_test_descrs` or
                // `MetaInterp::new` / `finish_setup`) before emitting FINISH.
                let descr: majit_ir::DescrRef = if is_exit_exc {
                    self.cpu_handle
                        .read()
                        .exit_frame_with_exception_descr_ref
                        .clone()
                } else {
                    self.done_with_this_frame_descr_arc_for_type(result_type)
                }
                .expect(
                    "FINISH emission requires cpu-attached singleton — \
                     call `attach_default_test_descrs` or use `MetaInterp::new`",
                );

                // Store result to jf_frame[0]
                if let Some(result) = arglocs.first() {
                    let slot0 = Loc::Frame(crate::regloc::FrameLoc::new(
                        0,
                        crate::jitframe::FIRST_ITEM_OFFSET as i32,
                        result_type == Type::Float,
                    ));
                    self.regalloc_mov(result, &slot0);
                }

                // `genop_finish`: `MOV_rp(eax, 0)` of the descr word, then
                // `mov(eax, RawEbpLoc(jf_descr))`. The word is the singleton
                // pointer selected above, bound after the recoveries.
                let slot = self.mc.new_dynamic_label();
                self.finish_descr_words
                    .push((slot, global_descr_ptr as u64));
                dynasm!(self.mc ; .arch x64 ; mov rax, [=>slot]);
                rx86::mov_br(&mut self.mc, JF_DESCR_OFS, rx86::EAX);

                if result_type == Type::Ref {
                    if let Some(gcmap) = self.finish_gcmap {
                        gcmap_set_bit(gcmap, 0);
                        self.push_gcmap(gcmap);
                    } else {
                        self.push_gcmap(self.gcmap_for_finish);
                    }
                } else if let Some(gcmap) = self.finish_gcmap {
                    self.push_gcmap(gcmap);
                } else {
                    self.pop_gcmap();
                }

                self._call_footer();
                // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
                // not the cell pointer.  `handle_fail_done_with_this_frame`
                // and `handle_fail_exit_frame_with_exception` match by
                // ptr-equality on the singleton, so the cell only carries
                // the keep-alive identity for `clt.asmmemmgr_gcreftracers`.
                self.fail_descrs.push(descr.clone());
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                let descr_arc = op.getdescr();
                let label_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                if label_descr.is_some() {
                    // `regalloc.py consider_label` calls `flush_loop` before
                    // binding: 16-byte alignment, and `min_bytes_before_label`
                    // so a `GUARD_NOT_INVALIDATED` patch or
                    // `redirect_call_assembler` cannot land on the target.
                    self.flush_loop();
                }
                if majit_ir::debug::have_debug_prints() {
                    majit_ir::debug::log_one(
                        "jit-backend",
                        &format!("LABEL: new DynamicLabel({label:?})"),
                    );
                }
                self.mc.forget_scratch_register();
                dynasm!(self.mc ; =>label);
                if let Some(descr) = label_descr {
                    descr.set_target_arglocs(
                        arglocs
                            .iter()
                            .copied()
                            .map(target_argloc_from_loc)
                            .collect(),
                    );
                    descr.set_ll_loop_code(self.mc.offset().0);
                    if let Some(id) = descr_arc.as_ref().map(majit_ir::descr_identity) {
                        self.target_tokens_currently_compiling.insert(id, label);
                    }
                    if let Some(descr_ref) = descr_arc.as_ref() {
                        self.compiled_target_tokens.push(descr_ref.clone());
                    }
                }
            }
            // ── Calls ──
            // `consider_call` (regalloc.rs) has already captured `arglocs =
            // [func_addr_or_descr_info..., arg_locs...]` and run `before_call`
            // for this op, so the register file is in its across-the-call shape
            // by the time the emitter sees it. Which registers that saved is the
            // `SAVE_ALL_REGS / SAVE_GCREF_REGS / SAVE_DEFAULT_REGS` choice made
            // there, not a blanket spill: `spill_or_move_registers_before_call`
            // (`regalloc.py`) drops values that die at the call, leaves
            // callee-saved ones where they are, and prefers moving the rest to a
            // free callee-saved register over spilling them.
            //
            // What is left here is the call and the result placement.
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN => {
                let oopspec = op.with_call_descr(|cd| cd.get_extra_info().oopspecindex);
                let is_raw_free =
                    op.opcode == OpCode::CallN && oopspec == Some(majit_ir::OopSpecIndex::RawFree);
                let is_math_sqrt = oopspec == Some(majit_ir::OopSpecIndex::MathSqrt);
                let is_threadlocalref_get =
                    oopspec == Some(majit_ir::OopSpecIndex::ThreadlocalrefGet);
                let is_nursery_alloc = op.opcode == OpCode::CallR
                    && op.with_call_descr(|cd| cd.get_extra_info().runtime_helper)
                        == Some(majit_ir::RuntimeHelperKind::NurseryAlloc);
                if matches!(
                    op.opcode,
                    OpCode::CallMayForceI
                        | OpCode::CallMayForceR
                        | OpCode::CallMayForceF
                        | OpCode::CallMayForceN
                        | OpCode::CallReleaseGilI
                        | OpCode::CallReleaseGilF
                        | OpCode::CallReleaseGilN
                ) {
                    self._store_force_index_if_next_guard(ops, op_index, fail_index);
                }
                if is_math_sqrt {
                    // assembler.py `genop_math_sqrt`: SQRTSD(arglocs[0], resloc).
                    self.genop_math_sqrt(result_loc);
                } else if is_threadlocalref_get {
                    // `_consider_threadlocalref_get`: [offset, size, sign].
                    let Some(Loc::Reg(resloc)) = result_loc else {
                        panic!("threadlocalref_get needs a register result, got {result_loc:?}");
                    };
                    self.threadlocalref_get(
                        Self::argloc_imm(arglocs, 0),
                        *resloc,
                        Self::argloc_imm(arglocs, 1) as usize,
                        Self::argloc_imm(arglocs, 2) != 0,
                    );
                } else if is_raw_free {
                    self.genop_nursery_free_inline_x86(op, arglocs);
                } else if is_nursery_alloc {
                    self.genop_nursery_alloc_inline_x86(op, arglocs);
                } else {
                    self.genop_call_with_arglocs(op, arglocs);
                }
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // assembler.py _store_force_index parity:
                // store next GUARD_NOT_FORCED's descr ptr to jf_force_descr
                // BEFORE the call, so forcing code knows which guard to resume.
                self._store_force_index_if_next_guard(ops, op_index, fail_index);
                self.genop_call_assembler(op, arglocs, result_loc);
            }
            OpCode::CondCallN => self.genop_discard_cond_call(regalloc, op, arglocs, op_index),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.genop_cond_call_value(regalloc, op, arglocs, op_index);
            }
            // `rewrite.py handle_new` lowers every NEW / NEW_WITH_VTABLE,
            // vtable store included (`gen_initialize_vtable`), and the
            // assembler has no genop for either.
            OpCode::New | OpCode::NewWithVtable => {
                panic!("not implemented operation with res: {:?}", op.opcode)
            }
            // ── Allocation (raw, when GC rewriter is not active) ──
            OpCode::NewArray | OpCode::NewArrayClear => self.genop_new_array(op, arglocs),
            OpCode::Newstr => self.genop_newstr(op, arglocs),
            OpCode::Newunicode => self.genop_newunicode(op, arglocs),
            // ── Allocation (rewritten by GC rewriter) ──
            OpCode::CallMallocNursery => {
                self.genop_call_malloc_nursery(op, result_loc);
            }
            OpCode::CallMallocNurseryHeaderless => {
                self.genop_call_malloc_nursery_headerless(op, result_loc);
            }
            // `malloc_cond_varsize_frame` / `malloc_cond_varsize`.
            OpCode::CallMallocNurseryVarsizeFrame => {
                self.malloc_cond_varsize_frame(arglocs, result_loc);
            }
            OpCode::CallMallocNurseryVarsize => {
                self.malloc_cond_varsize(op, arglocs, result_loc);
            }
            OpCode::CallMallocNurseryVarsizeHeaderless => {
                self.genop_call_malloc_nursery_varsize_headerless(op, arglocs, result_loc);
            }
            // `genop_discard_check_memory_error`: `TEST` plus a
            // not-taken `jz` to the in-buffer trampoline
            // `generate_propagate_error_64` binds after the recovery
            // stubs. That trampoline jumps to `propagate_exception_path`
            // (`_build_propagate_exception_path`).
            OpCode::CheckMemoryError => {
                let reg = match arglocs.first() {
                    Some(Loc::Reg(r)) if !r.is_xmm => r.value,
                    _ => panic!("CheckMemoryError arglocs[0] must be a non-xmm register"),
                };
                self.emit_propagate_exception_if_zero(reg);
            }
            // x86/assembler.py genop_discard_cond_call_gc_wb
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.emit_write_barrier_fastpath(regalloc, op, &arglocs);
            }
            // x86/assembler.py genop_discard_zero_array.  The GC
            // rewriter leaves ZERO_ARRAY in the stream and mutates its range
            // after observing following SETARRAYITEM_GC stores; it must reach
            // codegen even though it has no result.
            OpCode::ZeroArray => self.genop_discard_zero_array(arglocs),
            // assembler.py `load_effective_addr` / aarch64
            // `emit_op_load_effective_address`.  rewrite.py turns
            // COPYSTRCONTENT into LEA + memcpy; a silent no-op here
            // leaves the memcpy address in an unwritten register.
            OpCode::LoadEffectiveAddress => {
                self.genop_load_effective_address(&arglocs, result_loc);
            }
            // ── Misc ──
            OpCode::ForceToken => {
                if let Some(Loc::Reg(r)) = result_loc {
                    self.mc.MOV(*r, crate::regloc::EBP);
                }
            }
            // assembler.py genop_save_exception IS
            // `_store_and_reset_exception(resloc)` — reuse the shared helper.
            OpCode::SaveException => self.emit_store_and_reset_exception(result_loc),
            OpCode::SaveExcClass => self.genop_save_exc_class(result_loc),
            OpCode::RestoreException => self.genop_restore_exception(arglocs),
            // Guards never reach the non-guard dispatch — the regalloc
            // hands them to `regalloc_perform_guard`.
            _ if op.opcode.is_guard() => unreachable!(
                "regalloc_perform reached with guard {:?}; guards must \
                 route through regalloc_perform_guard",
                op.opcode
            ),
            // ── No-ops ──
            _ => {}
        }
    }

    /// assembler.py `genop_guard_list[guard_opnum]` — emit a guard with its
    /// faillocs (`regalloc_perform_guard`).
    fn genop_guard_list(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        faillocs: &[Option<Loc>],
        fail_index: u32,
        frame_depth: usize,
    ) {
        let guard_argloc = arglocs.first().copied();
        match op.opcode {
            // x86/assembler.py `genop_guard_guard_true` is a bare
            // `implement_guard(guard_token)` — the regalloc routed the
            // condition through `load_condition_into_cc`, which either
            // reuses the cc from the prior CompOp (CC fusion) or emits
            // a TEST itself before this point. Mirror that here.
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull => {
                if let Some(loc) = arglocs.first() {
                    self.load_condition_into_cc(loc);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            // x86/assembler.py `genop_guard_guard_false` inverts
            // the published cc, then implements. So a fused IntLt that
            // set CC_L turns into a CC_GE failure jump under GuardFalse.
            OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull => {
                if let Some(loc) = arglocs.first() {
                    self.load_condition_into_cc(loc);
                }
                self.guard_success_cc = self.guard_success_cc.map(invert_cc);
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            // `genop_guard_guard_value`: `UCOMISD` when `getarg(0).type ==
            // FLOAT`, `CMP` otherwise. `consider_guard_value` puts arg 0 in
            // a register; arg 1 keeps its `self.loc()` location code.
            OpCode::GuardValue => {
                if arglocs.len() >= 2 {
                    if op.arg(0).to_opref().ty() == Some(majit_ir::Type::Float) {
                        assert_eq!(op.arg(1).to_opref().ty(), Some(majit_ir::Type::Float));
                        let Loc::Reg(x) = arglocs[0] else {
                            panic!(
                                "GUARD_VALUE float operand must be in a register \
                                 (consider_guard_value), got {:?}",
                                arglocs[0]
                            );
                        };
                        self.mc.UCOMISD(x, &arglocs[1]);
                    } else {
                        self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                    }
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardClass => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardGcType => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_gc_type(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardIsObject => {
                if arglocs.len() >= 2 {
                    self.emit_guard_is_object(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardSubclass => {
                if arglocs.len() >= 3 {
                    self.emit_guard_subclass(&arglocs[0], &arglocs[1], &arglocs[2]);
                    self.guard_success_cc = Some(CC_B);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardException => {
                if arglocs.len() >= 2 {
                    self.emit_guard_exception(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
                self.emit_store_and_reset_exception(result_loc);
            }
            OpCode::GuardNonnullClass => {
                if arglocs.len() >= 2 {
                    // `genop_guard_guard_nonnull_class`: a null object
                    // leaves `CMP obj, 1` with B (and NE), so the forward
                    // `JB` lands on the one guard `Jcc` that
                    // `patch_jump_for_descr` redirects.
                    let Loc::Reg(obj) = &arglocs[0] else {
                        panic!(
                            "GuardNonnullClass: obj_loc must be Loc::Reg, got {:?}",
                            arglocs[0]
                        );
                    };
                    // Patched below: `emit_forward_jump('B')` is a `J_il8`.
                    let jb_location = self.mc.new_dynamic_label();
                    rx86::cmp_ri(&mut self.mc, obj.value, 1);
                    dynasm!(self.mc ; .arch x64 ; jb BYTE =>jb_location);
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    // patch the JB above
                    self.mc.forget_scratch_register();
                    dynasm!(self.mc ; .arch x64 ; =>jb_location);
                    self.guard_success_cc = Some(CC_E);
                    self.implement_guard_with_faillocs(
                        op,
                        op_index,
                        fail_index,
                        guard_argloc,
                        faillocs,
                        frame_depth,
                    );
                }
            }
            OpCode::GuardNoException => {
                // `genop_guard_guard_no_exception`: when the preceding
                // `cond_call` queued a `CondCallSlowPath`, the fast path
                // emits nothing. The token hangs on that slow path and
                // `generate_guard_no_exception` runs after the result move.
                // The don't-call edge never reaches the check.
                if self.pending_cond_call_wants_guard() {
                    let fail_label = self.mc.new_dynamic_label();
                    let token = self.guard_token_with_faillocs(
                        op,
                        op_index,
                        fail_index,
                        fail_label,
                        guard_argloc,
                        faillocs,
                        frame_depth,
                    );
                    // Replayed from `CondCallSlowPath` after the call, which
                    // clobbers r11. The fast-path cache is not that edge, so
                    // `known_scratch_value` stays unknown here.
                    // `generate_guard_no_exception` records it at the
                    // slow-path `jcc`.
                    self.record_guard_token_side_effects(op, &token);
                    self.stash_cond_call_guard_token(token);
                } else {
                    self.emit_guard_no_exception_check();
                    self.implement_guard_with_faillocs(
                        op,
                        op_index,
                        fail_index,
                        guard_argloc,
                        faillocs,
                        frame_depth,
                    );
                }
            }
            OpCode::GuardNoOverflow => {
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardOverflow => {
                // x86/assembler.py:1873-1874 aliases GUARD_NO_OVERFLOW to
                // guard_true and GUARD_OVERFLOW to guard_false. The overflow
                // arithmetic producer leaves the no-overflow success CC in
                // `guard_success_cc`, so invert it for the expected-overflow
                // arm before the common guard emitter derives its fail CC.
                let no_overflow_cc = self
                    .guard_success_cc
                    .take()
                    .expect("GuardOverflow requires a preceding overflow operation");
                self.guard_success_cc = Some(invert_cc(no_overflow_cc));
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNotForced => {
                rx86::cmp_bi(&mut self.mc, JF_DESCR_OFS, 0);
                self.guard_success_cc = Some(CC_E);
                self.implement_guard_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNotForced2 => {
                self.store_force_descr_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardNotInvalidated => {
                self.implement_guard_not_invalidated_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            OpCode::GuardAlwaysFails => {
                self.implement_guard_always_fails_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
            _ => {
                self.implement_guard_nojump_with_faillocs(
                    op,
                    op_index,
                    fail_index,
                    guard_argloc,
                    faillocs,
                    frame_depth,
                );
            }
        }
    }

    /// Helper: guard class comparison.
    /// x86/assembler.py `_cmp_guard_class` emits a single
    /// `CMP [obj + vtable_offset], classptr` so the object register is
    /// never touched. Mirror that: for register and 32-bit-fitting
    /// immediate classptrs we emit the memory-operand CMP directly; for
    /// 64-bit immediates we stage through the dedicated scratch (R11)
    /// rather than RAX, which may itself hold `obj_loc`. The earlier
    /// `mov rax, imm` clobbered `obj_loc` when the regalloc placed the
    /// object in RAX, leaving subsequent uses (e.g. the immediately
    /// following `move: Reg(0) → Frame(pos=N)`) writing the vtable
    /// constant into the deopt slot.
    fn _cmp_guard_class(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        // Caller (genop_guard_guard_class) sets `guard_success_cc =
        // Some(CC_E)` immediately after this returns, so any path that
        // fails to emit a CMP would branch on stale flags from a
        // preceding instruction. Fail closed instead.
        let Loc::Reg(obj) = obj_loc else {
            panic!("GuardClass: obj_loc must be Loc::Reg, got {obj_loc:?}");
        };
        if let Some(vtable_offset) = self.vtable_offset {
            self.mc
                .CMP(Operand::M(obj.value, vtable_offset as i64), class_loc);
        } else {
            let (Loc::Immed(i) | Loc::ImmedFloat(i)) = class_loc else {
                panic!("GuardClass (typeid form): class_loc must be Loc::Immed, got {class_loc:?}",);
            };
            let expected_typeid = self
                .lookup_typeid_from_classptr(i.value as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "GuardClass: missing typeid for classptr {:#x}",
                        i.value as usize
                    )
                });
            self._cmp_guard_gc_type(&Loc::Reg(*obj), &Loc::immed(expected_typeid as i64));
        }
    }

    fn require_guard_gc_type_info(&self, guard_name: &'static str) -> GuardGcTypeInfo {
        self.guard_gc_type_info.unwrap_or_else(|| {
            panic!(
                "{} requires cpu.supports_guard_gc_type and a TYPE_INFO layout",
                guard_name
            )
        })
    }

    fn lookup_subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        self.classptr_to_subclass_range
            .get(&(classptr as i64))
            .copied()
    }

    fn emit_load_gc_typeid_into_reg(&mut self, obj_reg: u8, dst_reg: u8) {
        let tid_ofs = -(majit_gc::header::GcHeader::SIZE as i32);
        self.mc
            .MOV32(Operand::R(dst_reg), Operand::M(obj_reg, i64::from(tid_ofs)));
    }

    /// `_cmp_guard_gc_type`: one `CMP32_mi` of an immediate type id.
    /// The header word lives at `obj - GcHeader::SIZE`; its low 32 bits are the type id.
    fn _cmp_guard_gc_type(&mut self, obj_loc: &Loc, expected_typeid_loc: &Loc) {
        // Callers (guard_class typeid form, guard_gc_type) branch on CC_E.
        let Loc::Reg(obj) = obj_loc else {
            panic!("guard_gc_type: obj_loc must be Loc::Reg, got {obj_loc:?}");
        };
        let (Loc::Immed(expected) | Loc::ImmedFloat(expected)) = expected_typeid_loc else {
            panic!(
                "_cmp_guard_gc_type: expected typeid must be ImmedLoc, got {expected_typeid_loc:?}"
            );
        };
        let tid_ofs = -(majit_gc::header::GcHeader::SIZE as i32);
        rx86::cmp32_mi(&mut self.mc, (obj.value, tid_ofs), expected.value as i32);
    }

    /// `AddressLoc` scale is 0..3 (`addr_add`). A larger `shift_by` is a
    /// `SHL` of the typeid and scale 0, so the address stays
    /// `base + (typeid << shift_by) + offset`.
    fn sib_scale_and_shl(shift_by: u8) -> (u8, u8) {
        if shift_by < 4 {
            (shift_by, 0)
        } else {
            (0, shift_by)
        }
    }

    /// `genop_guard_guard_is_object`: `MOV32` of the typeid, then `TEST8` of
    /// `addr_add(imm(base_type_info), typeid, scale=shift_by, offset=infobits_offset)`.
    /// Success is NZ.
    fn emit_guard_is_object(&mut self, obj_loc: &Loc, typeid_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_IS_OBJECT");
        let (Loc::Reg(obj), Loc::Reg(typeid)) = (obj_loc, typeid_loc) else {
            panic!(
                "guard_is_object: expected [Reg object, Reg typeid], got {obj_loc:?} {typeid_loc:?}"
            );
        };
        self.emit_load_gc_typeid_into_reg(obj.value, typeid.value);
        let (scale, shl) = Self::sib_scale_and_shl(info.shift_by);
        if shl > 0 {
            self.mc.SHL(*typeid, imm(i64::from(shl)));
        }
        let static_offset = (info.base_type_info as i64).wrapping_add(info.infobits_offset as i64);
        let addr = addr_add(&Loc::immed(static_offset), typeid_loc, 0, scale);
        self.mc.TEST8(addr, imm(i64::from(info.is_object_flag)));
    }

    /// x86/assembler.py `genop_guard_guard_subclass`.
    fn emit_guard_subclass(&mut self, obj_loc: &Loc, class_loc: &Loc, tmp_loc: &Loc) {
        // `cpu.vtable_offset` is set and no TYPE_INFO table is installed.
        // `offset2` is `cpu.subclassrange_min_offset`; `check_min` /
        // `check_max` are `vtable_ptr.subclassrange_min/max`.
        if self.guard_gc_type_info.is_none()
            && let (Some(vtable_offset), Some(range_off)) =
                (self.vtable_offset, self.subclassrange_min_offset)
        {
            let (Loc::Reg(obj), Loc::Immed(classptr) | Loc::ImmedFloat(classptr), Loc::Reg(tmp)) =
                (obj_loc, class_loc, tmp_loc)
            else {
                panic!(
                    "GUARD_SUBCLASS expects [Reg object, Immed classptr, Reg tmp] \
                     like x86/assembler.py:1947"
                );
            };
            let (check_min, check_max) =
                majit_backend::read_vtable_subclass_range(classptr.value, range_off);
            let offset = vtable_offset as i32;
            let offset2 = range_off as i32;
            self.mc.MOV(*tmp, Operand::M(obj.value, i64::from(offset)));
            self.mc.MOV(*tmp, Operand::M(tmp.value, i64::from(offset2)));
            self.mc.SUB(*tmp, imm(check_min));
            self.mc.CMP(*tmp, imm(check_max - check_min));
            return;
        }
        let info = self.require_guard_gc_type_info("GUARD_SUBCLASS");
        let (Loc::Reg(obj), Loc::Immed(classptr) | Loc::ImmedFloat(classptr), Loc::Reg(tmp)) =
            (obj_loc, class_loc, tmp_loc)
        else {
            panic!(
                "GUARD_SUBCLASS expects [Reg object, Immed classptr, Reg tmp] \
                 like x86/assembler.py:1947"
            );
        };
        let (check_min, check_max) = self
            .lookup_subclass_range(classptr.value as usize)
            .unwrap_or((0, 0));
        if let Some(vtable_offset) = self.vtable_offset {
            let offset = vtable_offset as i32;
            let offset2 = info.subclassrange_min_offset as i32;
            self.mc.MOV(*tmp, Operand::M(obj.value, i64::from(offset)));
            self.mc.MOV(*tmp, Operand::M(tmp.value, i64::from(offset2)));
        } else {
            // genop_guard_guard_subclass typeid arm: MOV32, then
            // MOV tmp, addr_add(imm(base_type_info), tmp, scale=shift_by,
            // offset=sizeof_ti + offset2).
            self.emit_load_gc_typeid_into_reg(obj.value, tmp.value);
            let (scale, shl) = Self::sib_scale_and_shl(info.shift_by);
            if shl > 0 {
                self.mc.SHL(*tmp, imm(i64::from(shl)));
            }
            let static_offset = (info.base_type_info as i64)
                .wrapping_add(info.sizeof_ti as i64)
                .wrapping_add(info.subclassrange_min_offset as i64);
            let addr = addr_add(&Loc::immed(static_offset), tmp_loc, 0, scale);
            self.mc.MOV(*tmp, addr);
        }
        self.mc.SUB(*tmp, imm(check_min));
        self.mc.CMP(*tmp, imm(check_max - check_min));
    }

    /// x86/assembler.py `genop_guard_guard_exception`.
    fn emit_guard_exception(&mut self, expected_loc: &Loc, tmp_loc: &Loc) {
        let Loc::Reg(tmp) = tmp_loc else {
            return;
        };
        // `MOV(loc1, heap(pos_exception))`: the address goes through
        // `_addr_as_reg_offset`, so r11 stays known for the
        // `_store_and_reset_exception` that follows the guard.
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.mc.MOV(*tmp, heap(exc_type_addr));
        self.mc.CMP(*tmp, expected_loc);
    }

    /// `genop_discard_check_memory_error`: `TEST reg, reg` and a
    /// not-taken `jz` (`Conditions['Z']`) to one in-buffer trampoline.
    /// `generate_propagate_error_64` binds that label after the recovery
    /// stubs and jumps to `propagate_exception_path`. The fallthrough
    /// has no label and does not forget the r11 cache. No-op when
    /// `propagate_exception_descr` is unattached.
    fn emit_propagate_exception_if_zero(&mut self, reg: u8) {
        if self.propagate_exception_descr_ptr() == 0 {
            return;
        }
        let path = self.propagate_exception_path;
        assert!(
            path != 0,
            "propagate_exception_descr is set but propagate_exception_path is 0"
        );
        let tramp = match self.pending_memoryerror_trampoline {
            Some(label) => label,
            None => {
                let label = self.mc.new_dynamic_label();
                self.pending_memoryerror_trampoline = Some(label);
                label
            }
        };
        dynasm!(self.mc ; .arch x64 ; test Rq(reg), Rq(reg));
        // Not taken when `reg` is nonzero. Binding the trampoline is a
        // join, so it forgets; this fallthrough does not.
        self.emit_jcc_to_label(CC_E, tramp);
    }

    /// `_store_and_reset_exception`: result = pos_exc_value; clear both
    /// pos_exception and pos_exc_value on the success fallthrough.
    fn emit_store_and_reset_exception(&mut self, result_loc: Option<&Loc>) {
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        match result_loc {
            Some(Loc::Reg(dst)) => self.mc.MOV(*dst, heap(exc_value_addr)),
            Some(loc @ Loc::Frame(_)) => {
                let tmploc = crate::regloc::X86_64_SCRATCH_REG;
                self.mc.MOV(tmploc, heap(exc_value_addr));
                self.mc.MOV(loc, tmploc);
            }
            None => {}
            Some(other) => panic!(
                "emit_store_and_reset_exception: unhandled result location \
                {other:?} — the exception value would be dropped"
            ),
        }
        self.mc.MOV(heap(exc_type_addr), imm(0));
        self.mc.MOV(heap(exc_value_addr), imm(0));
    }

    /// Hardware condition for `SET_ir` (`rx86.py` `Conditions`).
    /// Abstract `CC_L` is 10; hardware `L` is 12.
    fn hw_condition(cc: u8) -> u8 {
        match cc {
            CC_O => 0,
            CC_NO => 1,
            CC_B => 2,
            CC_AE => 3,
            CC_E => 4,
            CC_NE => 5,
            CC_BE => 6,
            CC_A => 7,
            CC_S => 8,
            CC_NS => 9,
            CC_L => 12,
            CC_GE => 13,
            CC_LE => 14,
            CC_G => 15,
            _ => 4,
        }
    }

    /// `assembler.py _if_parity_clear_zero_and_carry`.
    ///
    /// UCOMISD sets PF on an unordered compare, together with ZF and CF, so
    /// `sete` / `setb` / `setbe` would report NaN as equal / less-than and
    /// `setne` would report it as not-not-equal.  `cmp rbp, 0` on the frame
    /// pointer — never null inside compiled code — clears ZF and CF, and is
    /// jumped over when PF is clear.
    fn emit_if_parity_clear_zero_and_carry(&mut self) {
        // `emit_forward_jump('NP')`: a `J_il8` patched by
        // `patch_forward_jump`.
        let ordered = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64
        ; jnp BYTE =>ordered
        );
        // CMP EBP, 0: as EBP cannot be null here, that operation should
        // always clear zero and carry
        rx86::cmp_ri(&mut self.mc, rx86::EBP, 0);
        // patch the JNP above
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>ordered);
    }

    /// `flush_cc`.
    ///
    /// After a CMP/TEST that leaves a boolean in the flags, call this.
    /// `result_loc is ebp` publishes `guard_success_cc` for the next
    /// guard. Any other register is `MOV_ri(result, 0)` then `SET_ir`
    /// of `lowest8bits` (`rex_fw`, so the low byte is `sil`/`dil`/`bpl`/`spl`).
    /// `MOV` does not clobber the flags.
    fn flush_cc(&mut self, cond: u8, result_loc: Option<&Loc>) {
        // `flush_cc` opens with `guard_success_cc == cond_none`. A
        // condition still pending was published by an earlier op and
        // never consumed.
        debug_assert!(
            self.guard_success_cc.is_none(),
            "flush_cc: guard_success_cc already set",
        );
        let frame_reg_value = crate::x86::regalloc::frame_reg().value;
        if let Some(Loc::Reg(r)) = result_loc {
            if r.value == frame_reg_value {
                self.guard_success_cc = Some(cond);
                return;
            }
            self.mc.MOV(*r, imm(0));
            rx86::set_ir(
                &mut self.mc,
                Self::hw_condition(cond),
                r.value | rx86::BYTE_REG_FLAG,
            );
        }
    }

    /// x86/regalloc.py `load_condition_into_cc` parity for the
    /// emit side. If the previous op already published a cond in
    /// `guard_success_cc`, the guard reads it directly. Otherwise the
    /// guard arg is a materialised boolean and we re-issue
    /// TEST + set CC_NE so `implement_guard` has a flag state to jump
    /// off of.
    fn load_condition_into_cc(&mut self, loc: &Loc) {
        if self.guard_success_cc.is_some() {
            return;
        }
        self.emit_test_loc(loc);
        self.guard_success_cc = Some(CC_NE);
    }

    /// Map an integer comparison OpCode to a condition code.
    fn opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::IntLt => CC_L,
            OpCode::IntLe => CC_LE,
            OpCode::IntGt => CC_G,
            OpCode::IntGe => CC_GE,
            OpCode::IntEq | OpCode::PtrEq | OpCode::InstancePtrEq => CC_E,
            OpCode::IntNe | OpCode::PtrNe | OpCode::InstancePtrNe => CC_NE,
            OpCode::UintLt => CC_B,
            OpCode::UintLe => CC_BE,
            OpCode::UintGt => CC_A,
            OpCode::UintGe => CC_AE,
            _ => CC_E,
        }
    }

    /// Guard with faillocs — emit conditional jump and store faillocs on descr.
    fn implement_guard_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard_with_faillocs: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        let pos = self.mc.offset().0;
        // `implement_guard`: the `Jcc` does not change r11, so the cache
        // at this point is the value the recovery jump leaves in the
        // register.
        let known_scratch_value = self.mc.get_scratch_register_known_value();
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        self.pending_guard_tokens
            .last_mut()
            .expect("guard token appended")
            .known_scratch_value = known_scratch_value;
        self.set_last_guard_jump_offset(pos);
    }

    /// `implement_guard`: `guard_token.pos_jump_offset = pos - 4`, the
    /// target field of the `Jcc`/`JMP` ending at `pos`. dynasm encodes a
    /// branch to a dynamic label with a 32-bit displacement.
    fn set_last_guard_jump_offset(&mut self, pos: usize) {
        self.pending_guard_tokens
            .last_mut()
            .expect("guard token appended")
            .pos_jump_offset = Some(pos - 4);
    }

    /// Guard no-jump with faillocs.
    fn implement_guard_nojump_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        // No branch from this position. Recovery is not entered with the
        // fallthrough r11 cache, so `known_scratch_value` stays unknown.
        // `genop_guard_guard_not_invalidated` overwrites it: the patched
        // `JMP` does enter with r11 unchanged.
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
    }

    /// `assembler.py genop_guard_guard_not_invalidated` — the guard tests
    /// nothing and emits nothing. `cpu.invalidate_loop` later writes
    /// `JMP rel32` (five bytes) over the instructions that follow, while
    /// every other mutator is quiesced (`LoopInvalidation::invalidate`).
    ///
    /// `regalloc.py consider_guard_not_invalidated` calls
    /// `ensure_next_label_is_at_least_at_position(n + 5)` so that patch
    /// cannot overwrite the next label. `flush_loop` honours that floor
    /// when the label is bound, and again at the end of the trace so five
    /// bytes exist before the recovery stubs.
    fn implement_guard_not_invalidated_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        // `genop_guard_guard_not_invalidated`: `pos` is the opcode byte of
        // the not-yet-written `JMP`; `pos + 1` is "after potential jmp".
        let pos = self.mc.offset().0;
        self.implement_guard_nojump_with_faillocs(
            op,
            op_index,
            fail_index,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        let token = self
            .pending_guard_tokens
            .last_mut()
            .expect("guard token appended");
        token.pos_jump_offset = Some(pos + 1);
        token.guard_not_invalidated = true;
        // `genop_guard_guard_not_invalidated` records the scratch cache.
        // `get_relative_pos(break_basic_block=False)` does not forget, and
        // the `JMP rel32` written by `invalidate_loop` does not change r11.
        token.known_scratch_value = self.mc.get_scratch_register_known_value();
        self.ensure_next_label_is_at_least_at_position(pos + 5);
    }

    /// `regalloc.py ensure_next_label_is_at_least_at_position`.
    fn ensure_next_label_is_at_least_at_position(&mut self, at_least_position: usize) {
        self.min_bytes_before_label = self.min_bytes_before_label.max(at_least_position);
    }

    /// `regalloc.py flush_loop`. Pad with one `X86_64_CodeBuilder.MULTIBYTE_NOPs`
    /// entry so the next label is 16-byte aligned and not before
    /// `min_bytes_before_label`.
    fn flush_loop(&mut self) {
        // `RegAlloc.flush_loop` calls `MachineCodeBlockWrapper.get_relative_pos`,
        // which forgets the scratch register when it breaks the basic block.
        self.mc.forget_scratch_register();
        let current_pos = self.mc.offset().0;
        let aligned = (current_pos + 15) & !15;
        let target_pos = aligned.max(self.min_bytes_before_label);
        let insert_nops = target_pos - current_pos;
        assert!(
            insert_nops <= 15,
            "flush_loop pad {insert_nops} exceeds MULTIBYTE_NOPs"
        );
        if insert_nops > 0 {
            self.mc.extend(multibyte_nop(insert_nops).iter().copied());
        }
    }

    fn implement_guard_always_fails_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let fail_label = self.mc.new_dynamic_label();
        // The `jmp` leaves r11 unchanged. Capture the cache before
        // the `forget_scratch_register`, which only drops the fallthrough.
        let known_scratch_value = self.mc.get_scratch_register_known_value();
        dynasm!(self.mc ; .arch x64 ; jmp =>fail_label);
        let pos = self.mc.offset().0;
        self.mc.forget_scratch_register();
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        self.pending_guard_tokens
            .last_mut()
            .expect("guard token appended")
            .known_scratch_value = known_scratch_value;
        self.set_last_guard_jump_offset(pos);
    }

    /// `store_info_on_descr` fields for one guard, without pushing the token.
    /// `append_guard_token_with_faillocs` pushes it. `genop_guard_guard_no_exception`
    /// keeps it on `CondCallSlowPath` until `generate_guard_no_exception`.
    fn guard_token_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        fail_label: DynamicLabel,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) -> GuardToken {
        // assembler.py _store_force_index parity:
        // If a CALL_ASSEMBLER already pre-allocated this guard's descr
        // (stored in pending_force_descr), reuse it — same Arc, same ptr
        // that was written to jf_force_descr.
        // Stamp the per-trace fail_index and trace_id onto the metainterp
        // ResumeGuardDescr (`op.descr`).  `compile.py` reserves these
        // slots for the `ResumeDescr` family; gate the writes accordingly
        // so non-resume meta descrs (Done* / Exit* / Propagate) take the
        // default panic path.  The metainterp's `build_guard_metadata`
        // (`compile.rs`) used to do this after backend codegen with
        // the same sequential counter; doing it here lets readers consume
        // the canonical metainterp identity before metadata builds.
        let descr_arc = op.getdescr();
        if let Some(d) = descr_arc.as_ref() {
            if d.is_resume_guard() || d.is_resume_guard_copied() {
                if let Some(fd) = d.as_fail_descr() {
                    fd.set_fail_index_per_trace(fail_index);
                    fd.set_trace_id(self.trace_id);
                }
            }
        }
        let descr: majit_ir::DescrRef = if let Some(pre) = self.pending_force_descr.take() {
            pre
        } else if let Some(d) = descr_arc {
            // Guard exit — `compile.py` ResumeGuardDescr family.
            // Types already live on `op.descr`; do not
            // `infer_fail_arg_types().to_vec()` a second copy.
            d
        } else {
            // Test scaffold: tests synthesise guard ops without op.descr.
            // Mint a fresh metainterp ResumeGuardDescr to carry the
            // codegen-time identity (fail_index / trace_id / fail_arg_types).
            let fresh = majit_backend::make_resume_guard_descr_typed(
                self.infer_fail_arg_types(op, Some(op_index)).into_vec(),
            );
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        let descr_fd = descr.as_fail_descr().expect("guard descr is FailDescr");
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token: fail_index={} op_index={} opcode={:?} fail_args={:?} fail_arg_types={:?} faillocs={:?}",
                fail_index,
                op_index,
                op.opcode,
                op.guard_fail_args(),
                descr_fd.fail_arg_types(),
                faillocs
            );
        }

        // `BaseAssembler.store_info_on_descr`: encode `None`, registers
        // and stack slots, panicking on anything else as `gen_regs.index(loc)`
        // does, and share `_previous_rd_locs` when every position matches.
        let rd_locs = crate::guard::store_info_on_descr(
            &mut self.previous_rd_locs,
            faillocs,
            deadframe_slot_for_loc,
        );
        // Stamp source_op_index directly on the meta descr (UnsafeCell slot
        // owned by ResumeGuardDescr / ResumeGuardCopiedDescr per
        // resume_guard_descr.rs); `layout_for_fail_descr` reads it back
        // via `fd.source_op_index()` so no side-table is needed.
        if descr_fd.is_resume_guard() || descr_fd.is_resume_guard_copied() {
            descr_fd.set_source_op_index(op_index);
        }
        // `llsupport/assembler.py guardtok.faildescr.rd_locs = positions`
        // — write through the trait accessor so the metainterp
        // `AbstractFailDescr` (`history.py _attrs_`) receives the
        // canonical copy.  Must follow the `meta_descr` stamp above.
        descr_fd.set_rd_locs(rd_locs);
        // `regalloc.py consider_guard_value` records
        // `all_reg_indexes[x.value]`, a deadframe slot; `llmodel.py
        // get_value_direct` reads the raw jitframe word at that slot. The
        // shared failure stub saves every managed register into the jitframe
        // before recovery runs, so the operand remains readable even when it
        // is not a fail-arg. Stamping while laying the guard out means
        // `store_hash` (`compile.py`, gated on `status == 0`) leaves it
        // alone and `must_compile` hashes the (guard, failing value) pair
        // instead of the guard alone.  Without it a guard whose failing value
        // never repeats still accumulates in one bucket and compiles another
        // bridge every `trace_eagerness` failures, without bound.
        if op.opcode == majit_ir::OpCode::GuardValue
            && let Some(slot) = guard_argloc.as_ref().and_then(deadframe_slot_for_loc)
        {
            let type_tag = match op.arg(0).to_opref().ty() {
                Some(majit_ir::Type::Ref) => majit_backend::STATUS_TY_REF,
                Some(majit_ir::Type::Float) => majit_backend::STATUS_TY_FLOAT,
                _ => majit_backend::STATUS_TY_INT,
            };
            descr_fd.make_a_counter_per_value(slot as u32, type_tag);
        }
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token-slots: fail_index={} rd_locs={:?}",
                fail_index,
                descr_fd.rd_locs()
            );
        }
        let gcmap =
            self.guard_gcmap_from_faillocs(descr_fd.fail_arg_types(), faillocs, frame_depth);

        // Reuse the cell pre-allocated by `_store_force_index_if_next_guard`
        // when this guard is paired with a CALL_ASSEMBLER's force-store —
        // jf_force_descr and jf_descr then resolve to the same cell, and
        // `fail_descrs[fail_index]` carries exactly one entry per guard.
        let fail_cell_ptr = self
            .pending_force_cell
            .take()
            .unwrap_or_else(|| self.fail_descrs.push(descr.clone()));
        GuardToken {
            fail_label,
            fail_descr: descr.clone(),
            fail_cell_ptr,
            gcmap,
            pos_jump_offset: None,
            guard_not_invalidated: false,
            must_save_exception: matches!(
                op.opcode,
                OpCode::GuardException | OpCode::GuardNoException | OpCode::GuardNotForced
            ),
            known_scratch_value: -1,
        }
    }

    /// `GuardNotForced2` publishes `finish_gcmap` when its token is built.
    /// A fused `GUARD_NO_EXCEPTION` builds the token on the guard visit and
    /// pushes it later from `CondCallSlowPath.generate_body`.
    fn record_guard_token_side_effects(&mut self, op: &Op, token: &GuardToken) {
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(token.gcmap);
        }
    }

    fn push_built_guard_token(&mut self, op: &Op, token: GuardToken) {
        self.record_guard_token_side_effects(op, &token);
        self.pending_guard_tokens.push(token);
    }

    /// Append guard token with regalloc faillocs instead of opref_to_slot snapshot.
    fn append_guard_token_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        fail_label: DynamicLabel,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let token = self.guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            fail_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        self.push_built_guard_token(op, token);
    }

    /// `genop_guard_guard_no_exception`: the last queued slow path is the
    /// `CondCallSlowPath` that set `expects_guard_no_exception`.
    fn pending_cond_call_wants_guard(&self) -> bool {
        matches!(
            self.pending_slowpaths.last(),
            Some(SlowPath {
                kind: SlowPathKind::CondCall {
                    expects_guard_no_exception: true,
                    guard_token_no_exception: None,
                    ..
                },
                ..
            })
        )
    }

    fn stash_cond_call_guard_token(&mut self, token: GuardToken) {
        let Some(sp) = self.pending_slowpaths.last_mut() else {
            panic!("genop_guard_guard_no_exception: no CondCallSlowPath is pending");
        };
        match &mut sp.kind {
            SlowPathKind::CondCall {
                expects_guard_no_exception: true,
                guard_token_no_exception,
                ..
            } => {
                assert!(
                    guard_token_no_exception.is_none(),
                    "genop_guard_guard_no_exception stored two tokens"
                );
                *guard_token_no_exception = Some(token);
            }
            _ => panic!("genop_guard_guard_no_exception: the last slow path is not the cond-call"),
        }
    }

    /// x86/assembler.py `store_force_descr`: GUARD_NOT_FORCED_2 is not a
    /// conditional exit.  It publishes the resume descriptor and finish
    /// gcmap that FORCE_TOKEN may expose after the wrapper has returned.
    fn store_force_descr_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        guard_argloc: Option<Loc>,
        faillocs: &[Option<Loc>],
        frame_depth: usize,
    ) {
        let unused_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(
            op,
            op_index,
            fail_index,
            unused_label,
            guard_argloc,
            faillocs,
            frame_depth,
        );
        let token = self
            .pending_guard_tokens
            .pop()
            .expect("GUARD_NOT_FORCED_2 descriptor token");
        // `store_force_descr`: `mov r11, descr`, `mov [jf_force_descr], r11`.
        let scratch = crate::regloc::X86_64_SCRATCH_REG;
        self.mc.MOV(scratch, imm(token.fail_cell_ptr as i64));
        self.mc.MOV(Operand::B(JF_FORCE_DESCR_OFS), scratch);
        self.finish_gcmap = Some(token.gcmap);
    }

    // assembler.py:652 write_pending_failure_recoveries

    /// `generate_quick_failure`.
    ///
    /// `restore_scratch_register_known_value` assigns `known_scratch_value`
    /// and emits nothing, so a nearby constant is a `LEA` from the r11 the
    /// guard jump left behind. Then `PUSH` the descr and the gcmap and `JMP`
    /// `failure_recovery_code[exc + 2 * withfloats]` (`store_info_on_descr`).
    fn generate_quick_failure(&mut self, guard_token: GuardToken) -> RecoveryStub {
        let stub_start = self.mc.offset();

        let fail_label = guard_token.fail_label;
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("recovery stub: binding {fail_label:?}"),
            );
        }
        // The fail label is a join, so `get_relative_pos` forgets the
        // fallthrough cache. `restore_scratch_register_known_value` then
        // installs the value recorded at the guard jump. `-1` stays unknown.
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>fail_label);
        self.mc
            .restore_scratch_register_known_value(guard_token.known_scratch_value);

        // `PUSH` descr, then `PUSH` gcmap. The recovery pops gcmap first.
        // Upstream pushes the descr from the gc table (`PUSH_p` +
        // `_patch_load_from_gc_table`), the slot the rewriter filled with the
        // descr's GCREF. The cell here is a prebuilt immortal leaf
        // (`FailDescrCell`, `GC_IMMORTAL_LEAF_HEADER`) owned by `fail_descrs`
        // and minted after the rewriter ran, so it never moves and its
        // address is the value that slot would hold.
        self.mc.PUSH(imm(guard_token.fail_cell_ptr as i64));
        self.mc.PUSH(imm(guard_token.gcmap as i64));

        let index = Self::failure_recovery_index(&guard_token);
        let target = self.failure_recovery_code[index];
        assert!(
            target != 0,
            "failure_recovery_code[{index}] was not built (_build_failure_recovery)"
        );
        self.mc.JMP(imm(target as i64));
        RecoveryStub {
            fail_descr: guard_token.fail_descr,
            pos_recovery_stub: stub_start.0,
            pos_jump_offset: guard_token.pos_jump_offset,
            guard_not_invalidated: guard_token.guard_not_invalidated,
        }
    }

    /// `store_info_on_descr`: `exc + 2 * withfloats`. `withfloats` is set
    /// when a failarg type is FLOAT. `store_info_on_descr` also treats
    /// VECTOR that way; `Type` has no vector variant.
    fn failure_recovery_index(guard_token: &GuardToken) -> usize {
        let withfloats = guard_token
            .fail_descr
            .as_fail_descr()
            .is_some_and(|fd| fd.fail_arg_types().iter().any(|tp| *tp == Type::Float));
        usize::from(guard_token.must_save_exception) + 2 * usize::from(withfloats)
    }

    /// `SlowPath.__init__`: emit `J_il` to `slow_label` and record the r11
    /// cache. Conditional jumps do not forget. `continue_label` stays
    /// unbound until `set_continue_here`.
    fn emit_slow_jcc(&mut self, cc: u8, kind: SlowPathKind) -> SlowPath {
        let slow_label = self.mc.new_dynamic_label();
        let continue_label = self.mc.new_dynamic_label();
        self.emit_jcc_to_label(cc, slow_label);
        SlowPath {
            slow_label,
            continue_label,
            saved_scratch_value_1: self.mc.get_scratch_register_known_value(),
            saved_scratch_value_2: -1,
            kind,
        }
    }

    /// `malloc_cond` / `malloc_cond_varsize`: `Some(cc)` is `SlowPath.__init__`
    /// (`J_il`, conditional jumps do not forget). `None` is an unconditional
    /// `jmp` into the same body when the nursery addresses are absent;
    /// the `jmp` forgets before `saved_scratch_value_1` is recorded.
    fn begin_slow_path(&mut self, cc: Option<u8>, kind: SlowPathKind) -> SlowPath {
        let slow_label = self.mc.new_dynamic_label();
        let continue_label = self.mc.new_dynamic_label();
        match cc {
            Some(cc) => self.emit_jcc_to_label(cc, slow_label),
            None => {
                dynasm!(self.mc ; .arch x64 ; jmp =>slow_label);
                self.mc.forget_scratch_register();
            }
        }
        SlowPath {
            slow_label,
            continue_label,
            saved_scratch_value_1: self.mc.get_scratch_register_known_value(),
            saved_scratch_value_2: -1,
            kind,
        }
    }

    /// `SlowPath.set_continue_addr`. Binds the fast-path continue point
    /// without `forget_scratch_register`: `get_relative_pos` is asked not
    /// to break the block, so the not-taken edge keeps the r11 cache.
    fn set_continue_here(&mut self, sp: &mut SlowPath) {
        sp.saved_scratch_value_2 = self.mc.get_scratch_register_known_value();
        let continue_label = sp.continue_label;
        dynasm!(self.mc ; .arch x64 ; =>continue_label);
    }

    /// `Assembler386.flush_pending_slowpaths`.
    ///
    /// An append during `generate_body` is emitted after the current
    /// epilogue, which is what iterating `pending_slowpaths` does.
    fn flush_pending_slowpaths(&mut self) {
        loop {
            let pending = std::mem::take(&mut self.pending_slowpaths);
            if pending.is_empty() {
                break;
            }
            for sp in pending {
                let slow_label = sp.slow_label;
                let continue_label = sp.continue_label;
                dynasm!(self.mc ; .arch x64 ; =>slow_label);
                // `restore_scratch_register_known_value` emits no code.
                self.mc
                    .restore_scratch_register_known_value(sp.saved_scratch_value_1);
                self.generate_slowpath_body(sp.kind);
                self.mc.load_scratch_if_known(sp.saved_scratch_value_2);
                dynasm!(self.mc ; .arch x64 ; jmp =>continue_label);
                self.mc.forget_scratch_register();
            }
        }
    }

    /// `WriteBarrierSlowPath.generate_body`.
    ///
    /// Card marking tests `GCFLAG_CARDS_SET` with the sign flag of the fast
    /// path's `TEST8` (`js`) and, after the helper, with the helper's own
    /// trailing `TEST8` (`jns`). `jns` lands at the end of this body, so
    /// both arms fall into `SlowPath.generate`'s `load_scratch_if_known`
    /// and the jump back. The frame helper (`helper_num == 4`) is not
    /// passed a stack argument; `build_wb_slowpath` returns from it with
    /// `ret` and from the others with `ret 8`.
    fn generate_slowpath_body(&mut self, kind: SlowPathKind) {
        match kind {
            SlowPathKind::WriteBarrier {
                loc_base,
                loc_index,
                helper_num,
                card_marking,
                card_page_shift,
            } => {
                let card_mark = if card_marking {
                    let card_mark = self.mc.new_dynamic_label();
                    self.emit_jcc_to_label(CC_S, card_mark);
                    Some(card_mark)
                } else {
                    None
                };

                if helper_num != 4 {
                    // `PUSH(loc_base)` accepts an immediate (`_unaryop`).
                    match loc_base {
                        Loc::Reg(r) => {
                            dynasm!(self.mc ; .arch x64 ; push Rq(r.value));
                        }
                        Loc::Immed(i) | Loc::ImmedFloat(i) => self.mc.PUSH(imm(i.value)),
                        other => panic!("write barrier base {other:?}"),
                    }
                }
                let helper = self.wb_slowpath[helper_num];
                assert!(
                    helper != 0,
                    "wb_slowpath[{helper_num}] was not built (X86CpuExt::ensure_wb_slowpath)"
                );
                self.mc.CALL(imm(helper as i64));

                if let Some(card_mark) = card_mark {
                    let after_cards = self.mc.new_dynamic_label();
                    self.emit_jcc_to_label(CC_NS, after_cards);
                    dynasm!(self.mc ; .arch x64 ; =>card_mark);
                    let loc_index = loc_index.expect("card marking records loc_index");
                    // A register base copies the index through r11 and does not
                    // update `_scratch_register_value`. An immediate index is
                    // `OR8` via `addr_add_const`.
                    if matches!(loc_index, Loc::Reg(_) | Loc::Frame(_)) {
                        self.mc.forget_scratch_register();
                    }
                    self.mark_wb_cards(loc_base, &loc_index, card_page_shift);
                    dynasm!(self.mc ; .arch x64 ; =>after_cards);
                }
            }
            // `StackCheckSlowPath.generate_body`: one `call`.
            SlowPathKind::StackCheck => {
                let helper = self.stack_check_slowpath;
                assert!(
                    helper != 0,
                    "StackCheck queued without stack_check_slowpath"
                );
                self.mc.CALL(imm(helper as i64));
            }
            // `IncreaseStackSlowPath.generate_body`: depth at `[rsp+WORD]`
            // (the `PASS_ON_MY_FRAME` scratch), publish the gcmap captured
            // at the check, then call `build_frame_realloc_slowpath`.
            SlowPathKind::IncreaseStack { gcmap } => {
                rx86::mov_si(&mut self.mc, WORD as i32, 0x00ff_ffff);
                let imm_ofs = self.mc.offset().0 - 4;
                self.frame_depth_to_patch.push(imm_ofs);
                self.push_gcmap(gcmap as *mut usize);
                let helper = self.frame_realloc_slowpath;
                assert!(
                    helper != 0,
                    "IncreaseStack queued without frame_realloc_slowpath"
                );
                self.mc.CALL(imm(helper as i64));
            }
            SlowPathKind::CondCall {
                gcmap,
                imm_func,
                arglocs,
                resloc,
                variant_num,
                should_be_saved,
                guard_token_no_exception,
                expects_guard_no_exception,
            } => {
                self.generate_cond_call_slowpath(
                    gcmap,
                    imm_func,
                    arglocs,
                    resloc,
                    variant_num,
                    should_be_saved,
                    guard_token_no_exception,
                    expects_guard_no_exception,
                );
            }
            SlowPathKind::MallocCond {
                gcmap,
                result_reg,
                synthesize,
            } => {
                if let Some((sizeloc, header)) = synthesize {
                    dynasm!(self.mc ; .arch x64 ; xor rcx, rcx);
                    rx86::lea_ra(
                        &mut self.mc,
                        rx86::EDX,
                        (rx86::NO_BASE_REGISTER, sizeloc, 0, header),
                    );
                }
                self.publish_malloc_gcmap(gcmap);
                self.call_malloc_helper(self.malloc_slowpath_fixed);
                self.move_malloc_result(result_reg);
            }
            SlowPathKind::MallocCondVarsize {
                gcmap,
                result_reg,
                kind,
                itemsize,
                type_id,
                lengthloc,
            } => {
                // The maxlength `ja` lands on the nursery `ja`. That edge
                // skipped `_addr_as_reg_offset`, so `saved_scratch_value_1`
                // does not describe r11 here. Drop the cache before
                // `_load_scratch` (`MallocCondVarsizeSlowPath`).
                self.mc.forget_scratch_register();
                self.publish_malloc_gcmap(gcmap);
                debug_assert!(
                    !matches!(lengthloc, Loc::Reg(r) if r.value == rx86::ECX || r.value == rx86::EDX),
                    "malloc_cond_varsize lengthloc must not be ecx or edx"
                );
                let addr = match kind {
                    FLAG_ARRAY => {
                        // `MOV_si(WORD, itemsize)` into the `PASS_ON_MY_FRAME`
                        // scratch; the trampoline reads it above its return
                        // address, so OOM does not pop a callsite reservation.
                        self.mc.MOV(Operand::S(WORD as i32), imm(itemsize));
                        self.malloc_slowpath_varsize
                    }
                    FLAG_STR => self.malloc_slowpath_str,
                    FLAG_UNICODE => self.malloc_slowpath_unicode,
                    other => panic!("malloc_cond_varsize: unknown kind {other}"),
                };
                self.mov_gpr_from_loc(rx86::EDX, lengthloc);
                // `MOV_ri(ecx, arraydescr.tid)`. Upstream loads it for
                // `FLAG_ARRAY` only; `malloc_str` / `malloc_unicode` capture
                // their tid, which `dynasm_malloc_str` takes as an argument.
                rx86::mov_ri(&mut self.mc, rx86::ECX, type_id);
                self.call_malloc_helper(addr);
                self.move_malloc_result(result_reg);
            }
        }
    }

    /// `_build_cond_call_slowpath` body. The caller has already pushed the
    /// gcmap. Save every register this variant does not, except eax and the
    /// four `cond_call_register_arguments`, then `call rax`.
    fn emit_cond_call_helper_variant(&mut self, supports_floats: bool, callee_only: bool) {
        self.pending_slowpaths.clear();
        self.mc.restore_scratch_register_known_value(-1);
        self.guard_success_cc = None;
        self.frame_depth_to_patch.clear();
        let arg_regs = crate::x86::regalloc::cond_call_argument_regs();
        let ignored = [
            arg_regs[0],
            arg_regs[1],
            arg_regs[2],
            arg_regs[3],
            crate::regloc::EAX,
        ];
        push_all_regs_to_jitframe_raw(&mut self.mc, &ignored, supports_floats, callee_only);
        // Win64: alignment plus the 32-byte shadow (`WORD * 5`). SysV: one
        // word of alignment. The trace body is 0 mod 16 after `_call_header`;
        // the `call` into this helper leaves 8 mod 16, and this `sub` makes
        // the inner `call rax` 0 mod 16.
        #[cfg(target_os = "windows")]
        let align = (WORD * 5) as i32;
        #[cfg(not(target_os = "windows"))]
        let align = WORD as i32;
        rx86::sub_ri(&mut self.mc, rx86::ESP, align);
        self.mc.CALL(crate::regloc::EAX);
        self.reload_frame_if_necessary();
        rx86::add_ri(&mut self.mc, rx86::ESP, align);
        self.pop_gcmap();
        pop_all_regs_from_jitframe_raw(
            &mut self.mc,
            &[crate::regloc::EAX],
            supports_floats,
            callee_only,
        );
        dynasm!(self.mc ; .arch x64 ; ret);
        self.mc.forget_scratch_register();
        self.flush_pending_slowpaths();
    }

    /// `CondCallSlowPath.generate_body`. Spill `cond_call_register_arguments
    /// + [eax]` when the GPR is in `should_be_saved` and is not `resloc`,
    /// remap extra args, `mov eax, imm_func`,
    /// `call cond_call_slowpath[variant_num]`, move the result, restore eax,
    /// then `generate_guard_no_exception` when the token is present.
    fn generate_cond_call_slowpath(
        &mut self,
        gcmap: usize,
        imm_func: i64,
        arglocs: Vec<Loc>,
        resloc: Option<Loc>,
        variant_num: usize,
        should_be_saved: i64,
        guard_token_no_exception: Option<GuardToken>,
        expects_guard_no_exception: bool,
    ) {
        assert!(gcmap != 0, "cond_call published a null gcmap");
        self.push_gcmap(gcmap as *mut usize);
        let arg_regs = crate::x86::regalloc::cond_call_argument_regs();
        let gprs = [
            arg_regs[0],
            arg_regs[1],
            arg_regs[2],
            arg_regs[3],
            crate::regloc::EAX,
        ];
        let res_reg = match resloc {
            Some(Loc::Reg(r)) if !r.is_xmm => Some(r),
            _ => None,
        };
        let mut restore_eax = false;
        for gpr in gprs {
            // `CondCallSlowPath.generate_body`: save only when
            // `gpr in should_be_saved and gpr is not self.resloc`.
            let bound = (should_be_saved & (1i64 << gpr.value)) != 0;
            if !bound || res_reg == Some(gpr) {
                continue;
            }
            let slot = core_reg_position(gpr).expect("cond_call gpr is a managed register");
            rx86::mov_br(&mut self.mc, Self::slot_offset(slot), gpr.value);
            if gpr == crate::regloc::EAX {
                restore_eax = true;
            }
        }
        assert!(
            arglocs.len() <= arg_regs.len(),
            "cond_call remaps at most {} arguments, got {}",
            arg_regs.len(),
            arglocs.len()
        );
        let dst: Vec<Loc> = arg_regs[..arglocs.len()]
            .iter()
            .copied()
            .map(Loc::Reg)
            .collect();
        crate::jump::remap_frame_layout(
            self,
            &arglocs,
            &dst,
            Loc::Reg(crate::regloc::X86_64_SCRATCH_REG),
        );
        self.mc.MOV(crate::regloc::EAX, imm(imm_func));
        let helper = self
            .cond_call_slowpath
            .get(variant_num)
            .copied()
            .unwrap_or(0);
        assert!(
            helper != 0,
            "cond_call_slowpath[{variant_num}] was not built (_build_cond_call_slowpath)"
        );
        self.mc.CALL(imm(helper as i64));
        if let Some(resloc) = resloc {
            let is_eax = matches!(resloc, Loc::Reg(r) if r == crate::regloc::EAX);
            if !is_eax {
                self.regalloc_mov(&Loc::Reg(crate::regloc::EAX), &resloc);
            }
        }
        if restore_eax {
            let slot = core_reg_position(crate::regloc::EAX).expect("eax is a managed register");
            rx86::mov_rb(&mut self.mc, rx86::EAX, Self::slot_offset(slot));
        }
        if let Some(mut token) = guard_token_no_exception {
            // `generate_guard_no_exception`: `CMP [pos_exception], 0`,
            // success on `CC_E`, then the failing `jcc`. The displacement
            // is the last four bytes, same as `set_last_guard_jump_offset`.
            // The token is not in `pending_guard_tokens` yet, so the offset
            // is stored here and the token is pushed before the stubs run.
            self.emit_guard_no_exception_check();
            let cc = self
                .guard_success_cc
                .take()
                .expect("generate_guard_no_exception sets CC_E");
            let fail_label = token.fail_label;
            self.emit_jcc_to_label(invert_cc(cc), fail_label);
            let pos = self.mc.offset().0;
            token.pos_jump_offset = Some(pos - 4);
            // `generate_guard_no_exception` calls `implement_guard`, which
            // records the scratch cache at this `jcc`. The call forgot it;
            // `emit_guard_no_exception_check` reloaded the exception address
            // into r11, and the `jcc` leaves that value in place.
            token.known_scratch_value = self.mc.get_scratch_register_known_value();
            self.pending_guard_tokens.push(token);
        } else if expects_guard_no_exception {
            panic!("a cond-call asked for GUARD_NO_EXCEPTION and the guard never stored the token");
        }
    }

    /// assembler.py:1005 write_pending_failure_recoveries.
    /// Returns recovery stub offsets for post-finalize address fixup.
    fn write_pending_failure_recoveries(&mut self) -> Vec<RecoveryStub> {
        // `flush_pending_slowpaths` before the stub loop: a slow-path body
        // may append a guard token to `pending_guard_tokens`.
        self.flush_pending_slowpaths();
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] write_pending_failure_recoveries: {} tokens",
                self.pending_guard_tokens.len()
            );
        }
        let mut stub_offsets = Vec::new();
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            stub_offsets.push(self.generate_quick_failure(guard_token));
        }
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("write_pending done: {} stubs", stub_offsets.len()),
            );
        }
        // `generate_propagate_error_64`: one trampoline after the stubs.
        // `forget` before the bind — the label is a join, not a
        // `SlowPath` continue edge. `JMP(imm(propagate_exception_path))`
        // goes through r11 to its own buffer.
        if let Some(label) = self.pending_memoryerror_trampoline {
            let path = self.propagate_exception_path;
            assert!(
                path != 0,
                "memory-error trampoline without propagate_exception_path"
            );
            self.mc.forget_scratch_register();
            dynasm!(self.mc ; .arch x64 ; =>label);
            self.mc.JMP(imm(path as i64));
        }
        stub_offsets
    }

    /// assembler.py patch_pending_failure_recoveries — set
    /// `tok.faildescr.adr_jump_offset` to the raw address of the 4-byte
    /// target field in the guard's `JMP`/`Jcond`. dynasm already resolved
    /// that field to the recovery stub when it bound `fail_label`.
    fn patch_pending_failure_recoveries(rawstart: usize, stubs: &[RecoveryStub]) {
        for stub in stubs {
            let Some(pos_jump_offset) = stub.pos_jump_offset else {
                continue;
            };
            let addr = rawstart + pos_jump_offset;
            if !stub.guard_not_invalidated {
                debug_assert_eq!(
                    addr as i64 + 4 + unsafe { (addr as *const i32).read_unaligned() } as i64,
                    (rawstart + stub.pos_recovery_stub) as i64,
                    "guard branch target field does not reach its recovery stub"
                );
            }
            if let Some(fd) = stub.fail_descr.as_fail_descr() {
                fd.set_adr_jump_offset(addr);
            }
        }
    }

    /// `assembler.py patch_pending_failure_recoveries` — the
    /// `GUARD_NOT_INVALIDATED` arm of the same walk.  Rather than patching the
    /// guard now, record `(addr-in-the-code-of-the-not-yet-written-jump-target,
    /// relative-target-to-use)` for `clt.invalidate_positions`, already encoded
    /// as the displacement the store will write.
    fn collect_invalidate_positions(
        rawstart: usize,
        stubs: &[RecoveryStub],
    ) -> Vec<majit_backend::InvalidatePosition> {
        stubs
            .iter()
            .filter(|stub| stub.guard_not_invalidated)
            .map(|stub| {
                let pos_jump_offset = stub
                    .pos_jump_offset
                    .expect("GUARD_NOT_INVALIDATED records its jump position");
                let relative_target = stub.pos_recovery_stub as i64 - (pos_jump_offset as i64 + 4);
                let relative_target = i32::try_from(relative_target)
                    .expect("guard recovery stub within JMP rel32 reach of its guard");
                // `JMP_l`: `E9 rel32`, five bytes; `invalidate_loop` writes the
                // opcode at `addr - 1`.
                majit_backend::InvalidatePosition {
                    addr: rawstart + pos_jump_offset - 1,
                    word: 0xE9 | u64::from(relative_target as u32) << 8,
                }
            })
            .collect()
    }

    /// `assembler.py:948 _patch_frame_depth` — overwrite the 32-bit
    /// `0xffffff` placeholder at `adr` with the finalised frame depth.
    ///
    /// PyPy uses `codebuf.MachineCodeBlockWrapper().writeimm32` +
    /// `copy_to_raw_memory(adr)`; here we write the four little-endian
    /// bytes directly inside a `with_writable` guard so the page-RW
    /// permissions match the platform's executable-memory policy.
    fn patch_frame_depth(adr: usize, allocated_depth: usize) {
        codebuf::with_writable(adr as *mut u8, 4, || unsafe {
            (adr as *mut i32).write_unaligned(allocated_depth as i32);
        });
    }

    /// `assembler.py:898 patch_stack_checks` — iterate
    /// `frame_depth_to_patch` and rewrite each placeholder immediate
    /// with the final `framedepth` (already absolute, including
    /// `JITFRAME_FIXED_SIZE`).
    ///
    /// Takes the patch list by slice rather than via `&self` so the
    /// caller (which has already consumed `self.mc` through
    /// `finalize()`) can still drive the patch step without keeping
    /// `Assembler386` partially moved.
    fn patch_stack_checks(framedepth: usize, rawstart: usize, offsets: &[usize]) {
        for &ofs in offsets {
            Self::patch_frame_depth(rawstart + ofs, framedepth);
        }
    }

    // assembler.py:965-987 patch_jump_for_descr

    /// assembler.py:965 patch_jump_for_descr: redirect a guard to a
    /// bridge.
    ///
    /// `adr_jump_offset` is the raw address of the 4-byte target field of
    /// the guard's `JMP`/`Jcond` (set by `patch_pending_failure_recoveries`).
    /// If the bridge is within rel32 reach of the jump, patch that field.
    /// Otherwise leave the field pointing at the recovery stub and clobber
    /// the stub with "MOV r11, bridge_addr; JMP r11"; the stub is at least
    /// that long.
    pub fn patch_jump_for_descr(descr: &dyn majit_ir::FailDescr, adr_new_target: usize) {
        let adr_jump_offset = descr.adr_jump_offset();
        assert!(adr_jump_offset != 0, "guard already patched");
        let offset = adr_new_target as i64 - (adr_jump_offset as i64 + 4);
        if let Ok(offset) = i32::try_from(offset) {
            codebuf::with_writable(adr_jump_offset as *mut u8, 4, || unsafe {
                (adr_jump_offset as *mut i32).write_unaligned(offset);
            });
        } else {
            let rel = unsafe { (adr_jump_offset as *const i32).read_unaligned() };
            let adr_target = (adr_jump_offset as i64 + 4 + rel as i64) as usize;
            codebuf::with_writable(adr_target as *mut u8, 13, || unsafe {
                let stub_ptr = adr_target as *mut u8;
                *stub_ptr = 0x49;
                *stub_ptr.add(1) = 0xBB;
                (stub_ptr.add(2) as *mut u64).write_unaligned(adr_new_target as u64);
                *stub_ptr.add(10) = 0x41;
                *stub_ptr.add(11) = 0xFF;
                *stub_ptr.add(12) = 0xE3;
            });
        }

        // assembler.py:987
        descr.set_adr_jump_offset(0); // "patched"
    }

    /// assembler.py:1138 redirect_call_assembler: patch old loop entry
    /// to JMP to new loop after retrace.
    pub fn redirect_call_assembler(
        old: &majit_backend::JitCellToken,
        new: &majit_backend::JitCellToken,
        old_addr: *const u8,
        new_addr: *const u8,
    ) {
        codebuf::with_writable(old_addr as *mut u8, 16, || {
            let old_ptr = old_addr as *mut u8;
            let offset = new_addr as isize - (old_addr as isize + 5);
            if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                unsafe {
                    *old_ptr = 0xE9;
                    (old_ptr.add(1) as *mut i32).write(offset as i32);
                }
            } else {
                unsafe {
                    *old_ptr = 0x49;
                    *old_ptr.add(1) = 0xBB;
                    (old_ptr.add(2) as *mut u64).write(new_addr as u64);
                    *old_ptr.add(10) = 0x41;
                    *old_ptr.add(11) = 0xFF;
                    *old_ptr.add(12) = 0xE3;
                }
            }
        });
        // `x86/assembler.py redirect_call_assembler`:
        // `asm_adr = newlooptoken._ll_raw_start`.
        majit_backend::redirect_assembler(old, new, new.ll_raw_start() as u64);
    }

    // genop_* — integer arithmetic

    /// INT_ADD: result = arg0 + arg1
    #[allow(dead_code)]
    fn genop_int_add(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; add rax, rcx
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_SUB: result = arg0 - arg1
    #[allow(dead_code)]
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; sub rax, rcx
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_MUL: result = arg0 * arg1
    #[allow(dead_code)]
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; imul rax, rcx
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_AND: result = arg0 & arg1
    #[allow(dead_code)]
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; and rax, rcx
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_OR: result = arg0 | arg1
    #[allow(dead_code)]
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; or rax, rcx
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_XOR: result = arg0 ^ arg1
    #[allow(dead_code)]
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; xor rax, rcx
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_NEG: result = -arg0
    #[allow(dead_code)]
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; neg rax
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_INVERT: result = ~arg0
    #[allow(dead_code)]
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; not rax
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_LSHIFT: result = arg0 << arg1
    #[allow(dead_code)]
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; shl rax, cl
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    #[allow(dead_code)]
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; sar rax, cl
        );
        self.store_rax_to_result(op.pos().get());
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    #[allow(dead_code)]
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; shr rax, cl
        );
        self.store_rax_to_result(op.pos().get());
    }

    // genop_* — overflow arithmetic (assembler.py:1413-1425)

    /// assembler.py genop_int_add_ovf — delegates to genop_int_add,
    /// then sets guard_success_cc = 'NO'. On x86, ADD always sets OF.
    #[allow(dead_code)]
    fn genop_int_add_ovf(&mut self, op: &Op) {
        self.genop_int_add(op); // ADD sets OF on x86
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py genop_int_sub_ovf.
    #[allow(dead_code)]
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        self.genop_int_sub(op);
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py genop_int_mul_ovf.
    #[allow(dead_code)]
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        self.genop_int_mul(op); // IMUL sets OF on x86
        self.guard_success_cc = Some(CC_NO);
    }

    // genop_* — comparisons

    /// Emit SETcc/CSET to materialize a boolean result.
    /// x64: SETcc AL; MOVZX EAX, AL
    /// aarch64: CSET X0, cc
    #[allow(dead_code)]
    fn emit_setcc_to_result(&mut self, cc: u8, result_opref: OpRef) {
        match cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; setl al),
            CC_LE => dynasm!(self.mc ; .arch x64 ; setle al),
            CC_G => dynasm!(self.mc ; .arch x64 ; setg al),
            CC_GE => dynasm!(self.mc ; .arch x64 ; setge al),
            CC_E => dynasm!(self.mc ; .arch x64 ; sete al),
            CC_NE => dynasm!(self.mc ; .arch x64 ; setne al),
            CC_B => dynasm!(self.mc ; .arch x64 ; setb al),
            CC_BE => dynasm!(self.mc ; .arch x64 ; setbe al),
            CC_A => dynasm!(self.mc ; .arch x64 ; seta al),
            CC_AE => dynasm!(self.mc ; .arch x64 ; setae al),
            CC_O => dynasm!(self.mc ; .arch x64 ; seto al),
            CC_NO => dynasm!(self.mc ; .arch x64 ; setno al),
            _ => dynasm!(self.mc ; .arch x64 ; sete al),
        }
        dynasm!(self.mc
            ; .arch x64
            ; movzx eax, al
        );
        self.store_rax_to_result(result_opref);
    }

    /// INT_IS_TRUE: result = (arg0 != 0)
    #[allow(dead_code)]
    fn genop_int_is_true(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos().get().is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos().get());
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    #[allow(dead_code)]
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        self.guard_success_cc = Some(CC_E);
        if !op.pos().get().is_none() {
            self.emit_setcc_to_result(CC_E, op.pos().get());
        }
    }

    // genop_* — guards

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Looks up the materialized table populated by the runner from
    /// the active gc_ll_descr. RPython resolves the same value via
    /// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
    fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        self.classptr_to_typeid.get(&(classptr as i64)).copied()
    }

    fn emit_guard_jcc(&mut self, fail_cc: u8) -> DynamicLabel {
        let fail_label = self.mc.new_dynamic_label();
        match fail_cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; jl =>fail_label),
            CC_LE => dynasm!(self.mc ; .arch x64 ; jle =>fail_label),
            CC_G => dynasm!(self.mc ; .arch x64 ; jg =>fail_label),
            CC_GE => dynasm!(self.mc ; .arch x64 ; jge =>fail_label),
            CC_E => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
            CC_NE => dynasm!(self.mc ; .arch x64 ; jne =>fail_label),
            CC_B => dynasm!(self.mc ; .arch x64 ; jb =>fail_label),
            CC_BE => dynasm!(self.mc ; .arch x64 ; jbe =>fail_label),
            CC_A => dynasm!(self.mc ; .arch x64 ; ja =>fail_label),
            CC_AE => dynasm!(self.mc ; .arch x64 ; jae =>fail_label),
            CC_O => dynasm!(self.mc ; .arch x64 ; jo =>fail_label),
            CC_NO => dynasm!(self.mc ; .arch x64 ; jno =>fail_label),
            CC_S => dynasm!(self.mc ; .arch x64 ; js =>fail_label),
            CC_NS => dynasm!(self.mc ; .arch x64 ; jns =>fail_label),
            _ => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
        }
        // aarch64: b.cond has 19-bit range (±1MB), which is too short
        // for forward references to recovery stubs. Use inverted condition
        // + unconditional branch (26-bit / ±128MB) pattern instead:
        //   b.NOT_cond >skip ; b =>fail_label ; skip:
        fail_label
    }

    fn emit_jcc_to_label(&mut self, fail_cc: u8, fail_label: DynamicLabel) {
        match fail_cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; jl =>fail_label),
            CC_LE => dynasm!(self.mc ; .arch x64 ; jle =>fail_label),
            CC_G => dynasm!(self.mc ; .arch x64 ; jg =>fail_label),
            CC_GE => dynasm!(self.mc ; .arch x64 ; jge =>fail_label),
            CC_E => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
            CC_NE => dynasm!(self.mc ; .arch x64 ; jne =>fail_label),
            CC_B => dynasm!(self.mc ; .arch x64 ; jb =>fail_label),
            CC_BE => dynasm!(self.mc ; .arch x64 ; jbe =>fail_label),
            CC_A => dynasm!(self.mc ; .arch x64 ; ja =>fail_label),
            CC_AE => dynasm!(self.mc ; .arch x64 ; jae =>fail_label),
            CC_O => dynasm!(self.mc ; .arch x64 ; jo =>fail_label),
            CC_NO => dynasm!(self.mc ; .arch x64 ; jno =>fail_label),
            CC_S => dynasm!(self.mc ; .arch x64 ; js =>fail_label),
            CC_NS => dynasm!(self.mc ; .arch x64 ; jns =>fail_label),
            _ => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
        }
    }

    /// Infer fail_arg_types from `op.type_` (via `opref_type`) or
    /// `op.fail_arg_types`.
    fn infer_fail_arg_types(&self, op: &Op, op_index: Option<usize>) -> SmallVec<[Type; 8]> {
        if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            if let Some(descr_types) = op
                .with_fail_descr(|fd| {
                    let dt = fd.fail_arg_types();
                    (!dt.is_empty()).then(|| SmallVec::from_slice(dt))
                })
                .flatten()
            {
                return descr_types;
            }
        }
        let descr_arc = op.getdescr();
        if let Some(fd) = descr_arc.as_ref().and_then(|d| d.as_fail_descr()) {
            // Step A installs op.descr = ResumeGuardDescr with
            // post-numbering fail_arg_types via
            // store_final_boxes_in_guard (optimizeopt/mod.rs).
            // The hash once cited for Step A resolves nowhere in this
            // repository, so that symbol is the reference.
            // Prefer the descr for guards too; fall through to
            // op.fail_arg_types only for sharing-path guards
            // (optimizeopt/mod.rs) where op.descr=None.
            let dt = fd.fail_arg_types();
            let expected_len = op.guard_fail_args().map(|fa| fa.len()).unwrap_or(0);
            if dt.len() == expected_len && !dt.is_empty() {
                return SmallVec::from_slice(dt);
            }
        }
        if let Some(ts) = op.get_fail_arg_types() {
            let expected_len = if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.num_args()
            } else {
                op.guard_fail_args().map(|fa| fa.len()).unwrap_or(0)
            };
            if ts.len() == expected_len {
                SmallVec::from_slice(&ts)
            } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.args_slice()
                    .iter()
                    .map(|opref| {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    })
                    .collect()
            } else if let Some(fa) = op.guard_fail_args() {
                fa.iter()
                    .map(|opref| {
                        if opref.is_none() {
                            // resume.py parity: TAGCONST/TAGVIRTUAL
                            // slots are kept as OpRef::NONE in fail_args
                            // (PyPy filters them out; pyre keeps positional).
                            // `Type::Void` is the "hole" sentinel — value
                            // comes from the resume snapshot, not the
                            // deadframe, so downstream consumers
                            // (`guard_gcmap_from_faillocs`, `typed_outputs`
                            // reconstruction) must skip these slots. Earlier
                            // code used `Type::Ref`, which silently leaked a
                            // NULL `GcRef` into the gcmap and the shadow
                            // stack.
                            Type::Void
                        } else {
                            self.opref_type_at(opref.to_opref(), op_index).unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                     op_index={:?} (fail_arg): RPython box.type is fixed at \
                                     construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                        }
                    })
                    .collect()
            } else {
                SmallVec::new()
            }
        } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            // Finish/Jump carry no failargs; their result kind comes from
            // the argument boxes, whose types are fixed at construction
            // (resoperation.py InputArgInt/727/739).  When neither a fail descr nor
            // a preset fail_arg_types list supplies them, infer from the
            // arglist so the FINISH's done_with_this_frame_descr kind
            // matches the caller's CALL_ASSEMBLER result kind (a Void
            // mismatch routes every return through the assembler helper
            // instead of the result-loading fast path).
            op.args_slice()
                .iter()
                .map(|opref| {
                    self.opref_type_at(opref.to_opref(), op_index)
                        .unwrap_or_else(|| {
                            panic!(
                                "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                opref, op_index
                            )
                        })
                })
                .collect()
        } else if let Some(fa) = op.guard_fail_args() {
            fa.iter()
                .map(|opref| {
                    if opref.is_none() {
                        // resume.py:411-417 parity: see comment above —
                        // Type::Void is the "hole" sentinel.
                        Type::Void
                    } else {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (fail_arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    }
                })
                .collect()
        } else {
            SmallVec::new()
        }
    }

    /// assembler.py generate_guard_no_exception:
    /// `CMP heap(self.cpu.pos_exception()), imm0` with success on zero.
    fn emit_guard_no_exception_check(&mut self) {
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.mc.CMP(heap(exc_type_addr), imm(0));
        self.guard_success_cc = Some(CC_E);
    }

    /// `_store_force_index`: before a call that may force, store the next
    /// GUARD_NOT_FORCED / GUARD_NOT_FORCED_2 fail descr into `jf_force_descr`.
    /// Does not write `jf_descr` (`genop_guard_guard_not_forced` compares that
    /// field with zero; a fresh frame already holds zero).
    fn _store_force_index_if_next_guard(&mut self, ops: &[OpRc], op_idx: usize, fail_index: u32) {
        // assembler.py _find_nearby_operation(+1)
        let next_idx = op_idx + 1;
        if next_idx >= ops.len() {
            return;
        }
        let next_op = &ops[next_idx];
        if next_op.opcode != OpCode::GuardNotForced && next_op.opcode != OpCode::GuardNotForced2 {
            return;
        }
        // Pre-allocate the fail descr for the next GUARD_NOT_FORCED.
        // The full metadata (faillocs, rd_numb, etc.) will be filled in
        // when the guard is actually emitted in append_guard_token_with_faillocs.
        // Pre-allocated GuardNotForced descr — ResumeGuardDescr family.
        // Stamp the metainterp `AbstractFailDescr` Arc from `next_op.descr`
        // here so `append_guard_token_with_faillocs` does not need a second
        // pass through `unsafe { Arc::as_ptr as *mut }`.
        let descr_arc = next_op.getdescr();
        if let Some(d) = descr_arc.as_ref() {
            if d.is_resume_guard() || d.is_resume_guard_copied() {
                if let Some(fd) = d.as_fail_descr() {
                    fd.set_fail_index_per_trace(fail_index);
                    fd.set_trace_id(self.trace_id);
                }
            }
        }
        let descr: majit_ir::DescrRef = if let Some(d) = descr_arc {
            d
        } else {
            let fresh = majit_backend::make_resume_guard_descr_typed(
                self.infer_fail_arg_types(next_op, Some(next_idx))
                    .into_vec(),
            );
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        // `force_token_to_dead_frame` (cranelift/compiler.rs)
        // recovers `jf_force_descr` via `recover_fail_descr_cell`, which
        // requires a `FailDescrCell` thin pointer.  Bake the cell pointer
        // here (not the bare `Arc<dyn Descr>` fat-pointer data half) and
        // hand the cell off to `append_guard_token_with_faillocs` so the
        // inline guard-exit path bakes the same identity into jf_descr.
        let descr_ptr = self.fail_descrs.push(descr.clone()) as i64;
        self.pending_force_descr = Some(descr);
        self.pending_force_cell = Some(descr_ptr as usize);

        // `_store_force_index`: `forget_scratch_register`, load the descr
        // into `X86_64_SCRATCH_REG`, then `mov [jf_force_descr], r11`.
        // R11 is outside the allocatable set, so the call's arguments stay
        // put. The load is a cell pointer rather than a gc-table slot: fail
        // descrs are `FailDescrCell`s, and `recover_fail_descr_cell` reads
        // that thin pointer back.
        let scratch = crate::regloc::X86_64_SCRATCH_REG;
        self.mc.MOV(scratch, imm(descr_ptr));
        self.mc.MOV(Operand::B(JF_FORCE_DESCR_OFS), scratch);
    }

    // genop_* — control flow

    /// FINISH: store result (if any), store descr ptr, return jf_ptr.
    #[allow(dead_code)]
    fn genop_finish(&mut self, op: &Op, fail_index: u32) {
        // compiler.rs parity: trust explicit FINISH types only when
        // they match the actual result arity; otherwise infer from the op args.
        let finish_refs: Vec<OpRef> = op.args_slice().iter().map(|a| a.to_opref()).collect();
        let fail_arg_types = if let Some(explicit) = op.get_fail_arg_types() {
            if explicit.len() == finish_refs.len() {
                explicit.to_vec()
            } else {
                finish_refs
                    .iter()
                    .map(|opref| self.opref_type_at(*opref, None).unwrap_or(Type::Int))
                    .collect()
            }
        } else {
            finish_refs
                .iter()
                .map(|opref| self.opref_type_at(*opref, None).unwrap_or(Type::Int))
                .collect()
        };
        // compile.py parity: use type-specific global singleton.
        // FINISH op exit (DoneWithThisFrame*) — `compile.py:185` skips these.
        // Finish ops write the type-appropriate singleton pointer to jf_descr
        // so CALL_ASSEMBLER's fast path CMP matches the correct variant.
        let result_type = if fail_arg_types.is_empty() {
            Type::Void
        } else {
            fail_arg_types[0]
        };
        let global_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
        // Singleton-direct push (see OpCode::Finish above for rationale).
        let descr: majit_ir::DescrRef = self
            .done_with_this_frame_descr_arc_for_type(result_type)
            .expect(
                "genop_finish requires cpu-attached singleton — \
                 call `attach_default_test_descrs` or use `MetaInterp::new`",
            );

        // If there's a result argument, store it to jf_frame[0].
        // assembler.py:2291-2303 parity: float results use xmm0/MOVSD.
        if op.num_args() > 0 {
            let arg0 = op.arg(0).to_opref();
            let slot0_offset = Self::slot_offset(0);
            if result_type == Type::Float {
                // Float: load to xmm0, store via MOVSD
                self.load_arg_to_rax(arg0); // loads raw bits
                rx86::mov_br(&mut self.mc, slot0_offset, rx86::EAX); // store float bits via GPR
            } else {
                self.load_arg_to_rax(arg0);
                rx86::mov_br(&mut self.mc, slot0_offset, rx86::EAX);
            }
        }

        // Store descr pointer at jf_ptr[0] (jf_descr slot).
        // compile.py:665-674 parity: use global singleton pointer.
        let descr_ptr = global_descr_ptr;
        rx86::mov_ri(&mut self.mc, rx86::EAX, descr_ptr);
        rx86::mov_br(&mut self.mc, JF_DESCR_OFS, rx86::EAX);

        if result_type == Type::Ref {
            if let Some(gcmap) = self.finish_gcmap {
                gcmap_set_bit(gcmap, 0);
                self.push_gcmap(gcmap);
            } else {
                self.push_gcmap(self.gcmap_for_finish);
            }
        } else if let Some(gcmap) = self.finish_gcmap {
            self.push_gcmap(gcmap);
        } else {
            self.pop_gcmap();
        }

        // Emit epilogue (return jf_ptr).
        self._call_footer();

        // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
        // not the cell pointer (see OpCode::Finish comment above).
        self.fail_descrs.push(descr.clone());
    }

    // genop_* — type conversions

    // Float helpers

    /// Load a float value from `opref` into XMM0 (x64) / D0 (aarch64).
    /// Float values are stored as bit-cast i64 in frame slots.
    #[allow(dead_code)]
    fn load_float_arg_to_d0(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                rx86::movsd_xb(&mut self.mc, 0, offset);
            }
            ResolvedArg::Const(_) => {
                panic!(
                    "float constants are ConstFloatLoc from X86XMMRegisterManager.convert_to_imm"
                );
            }
        }
    }

    /// Load a float value from `opref` into XMM1 (x64) / D1 (aarch64).
    #[allow(dead_code)]
    fn load_float_arg_to_d1(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                rx86::movsd_xb(&mut self.mc, 1, offset);
            }
            ResolvedArg::Const(_) => {
                panic!(
                    "float constants are ConstFloatLoc from X86XMMRegisterManager.convert_to_imm"
                );
            }
        }
    }

    /// Store XMM0 (x64) / D0 (aarch64) to the frame slot for `result_opref`.
    #[allow(dead_code)]
    fn store_d0_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        rx86::movsd_bx(&mut self.mc, offset, 0);
    }

    // genop_* — float arithmetic
    // x86/assembler.py:1648 genop_float_add etc.
    // aarch64/assembler.py float equivalents

    /// FLOAT_ADD: result = arg0 + arg1
    #[allow(dead_code)]
    fn genop_float_add(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; addsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_SUB: result = arg0 - arg1
    #[allow(dead_code)]
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; subsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_MUL: result = arg0 * arg1
    #[allow(dead_code)]
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; mulsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    #[allow(dead_code)]
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; divsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    #[allow(dead_code)]
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        // Load the sign-bit mask (0x8000_0000_0000_0000) into XMM1
        // via integer register, then XOR.
        let sign_mask: i64 = i64::MIN; // 0x8000000000000000
        rx86::mov_ri(&mut self.mc, rx86::EAX, sign_mask);
        dynasm!(self.mc
        ; .arch x64
        ; movq xmm1, rax
        );
        dynasm!(self.mc
                    ; .arch x64
                    ; xorpd xmm0, xmm1

        );
        self.store_d0_to_result(op.pos().get());
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    #[allow(dead_code)]
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        // Break cvtsi2sd's false dependency on the destination's prior value
        // (it preserves the high 64 bits) with a zeroing idiom.
        dynasm!(self.mc
        ; .arch x64
        ; pxor xmm0, xmm0
        );
        dynasm!(self.mc
            ; .arch x64
            ; cvtsi2sd xmm0, rax
        );
        self.store_d0_to_result(op.pos().get());
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    #[allow(dead_code)]
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; cvttsd2si rax, xmm0
        );
        self.store_rax_to_result(op.pos().get());
    }

    // genop_* — memory operations
    // x86/assembler.py:1747 genop_getfield_gc etc.

    /// x86/assembler.py:1746 genop_discard_setfield — sized store via regalloc.
    /// Stage non-register values through X86_64_SCRATCH_REG (r11), mirroring
    /// the aarch64 path that uses x16.
    fn emit_op_setfield_regalloc(
        &mut self,
        base: &crate::regloc::RegLoc,
        val_loc: &Loc,
        ofs: i32,
        field_size: usize,
    ) {
        let dest_addr = Operand::M(base.value, i64::from(ofs));
        if let Loc::Reg(_) | Loc::Immed(_) | Loc::ImmedFloat(_) = val_loc {
            self.save_into_mem(dest_addr, val_loc, field_size);
            return;
        }
        // A memory value is staged through the scratch register, which
        // `INSN('MOV*')` refuses to store, so the store is the raw form.
        let scratch = crate::regloc::X86_64_SCRATCH_REG;
        self.mc.MOV(scratch, val_loc);
        let mem = (base.value, ofs);
        match field_size {
            1 => rx86::mov8_mr(&mut self.mc, mem, scratch.value),
            2 => rx86::mov16_mr(&mut self.mc, mem, scratch.value),
            4 => rx86::mov32_mr(&mut self.mc, mem, scratch.value),
            _ => rx86::mov_mr(&mut self.mc, mem, scratch.value),
        }
    }

    /// x86/assembler.py _genop_gc_load — sized load via regalloc.
    /// `size`: byte size (1/2/4/8). Negative = signed load.
    fn emit_op_gcload_regalloc(
        &mut self,
        base_loc: &Loc,
        ofs_loc: &Loc,
        dst: &crate::regloc::RegLoc,
        size: i64,
    ) {
        let src = addr_add(base_loc, ofs_loc, 0, 0);
        self.load_from_mem(*dst, src, size.unsigned_abs() as usize, size < 0);
    }

    /// assembler.py `load_from_mem`.
    fn load_from_mem(
        &mut self,
        resloc: crate::regloc::RegLoc,
        source_addr: Operand,
        size: usize,
        sign: bool,
    ) {
        if resloc.is_xmm {
            // An xmm location is `MOVSD`. `get_type_flag` banks
            // `lltype.Float` only; a singlefloat access is int-kind.
            debug_assert_eq!(size, 8, "load_from_mem: float-kind access is 8 bytes");
            self.mc.MOVSD(resloc, source_addr);
            return;
        }
        match (size, sign) {
            (1, true) => self.mc.MOVSX8(resloc, source_addr),
            (1, false) => self.mc.MOVZX8(resloc, source_addr),
            (2, true) => self.mc.MOVSX16(resloc, source_addr),
            (2, false) => self.mc.MOVZX16(resloc, source_addr),
            (4, true) => self.mc.MOVSX32(resloc, source_addr),
            // zero-extending
            (4, false) => self.mc.MOV32(resloc, source_addr),
            (WORD, _) => self.mc.MOV(resloc, source_addr),
            _ => panic!("load_from_mem size = {size}"),
        }
    }

    /// `ImmedLoc.lowest8bits`: the low byte, sign-extended.
    fn lowest8bits(val: i64) -> i32 {
        i32::from(val as u8 as i8)
    }

    /// Sized store: `[base + ofs]` or `[base + ofs_reg]` — assembler.py save_into_mem.
    fn save_into_mem(&mut self, dest_addr: Operand, value_loc: &Loc, size: usize) {
        if let Loc::Reg(v) = value_loc
            && v.is_xmm
        {
            // An xmm store is `MOVSD`. `get_type_flag` banks only
            // `lltype.Float` (8 bytes).
            debug_assert_eq!(size, 8, "save_into_mem: float-kind access is 8 bytes");
            self.mc.MOVSD(dest_addr, *v);
            return;
        }
        match size {
            // `value_loc.lowest8bits()`
            1 => match value_loc.as_immed() {
                Some((v, _)) => self
                    .mc
                    .MOV8(dest_addr, imm(i64::from(Self::lowest8bits(v)))),
                None => self.mc.MOV8(dest_addr, value_loc),
            },
            2 => self.mc.MOV16(dest_addr, value_loc),
            4 => self.mc.MOV32(dest_addr, value_loc),
            8 => self.mc.MOV(dest_addr, value_loc),
            other => panic!("save_into_mem: unsupported store size {other}"),
        }
    }

    // genop_* — calls
    // x86/assembler.py _genop_call

    fn argloc_imm(arglocs: &[Loc], index: usize) -> i64 {
        match arglocs.get(index) {
            Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
            _ => 0,
        }
    }

    /// `_genop_call`: arglocs are `[resloc, size, sign, func, args...]`,
    /// or `[resloc, size, sign, saveerr, func, args...]` for
    /// CALL_RELEASE_GIL.
    fn genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        let is_call_release_gil = op.opcode.is_call_release_gil();
        let func_index = 3 + usize::from(is_call_release_gil);
        let descr = op.getdescr();
        let descr = descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("_genop_call: op without CallDescr");
        let resloc = match arglocs[0] {
            loc @ Loc::Reg(_) => Some(loc),
            _ => None,
        };
        let mut cb = super::callbuilder::CallBuilder64::new(
            self,
            arglocs[func_index],
            &arglocs[func_index + 1..],
            resloc,
            descr.result_class(),
            Self::argloc_imm(arglocs, 1) as usize,
        );
        cb.argtypes = descr.arg_classes();
        cb.ressign = Self::argloc_imm(arglocs, 2) != 0;
        if is_call_release_gil {
            cb.emit_call_release_gil(Self::argloc_imm(arglocs, 3));
        } else if descr.get_extra_info().check_can_collect() {
            cb.emit();
        } else {
            cb.emit_no_collect();
        }
    }

    /// assembler.py `threadlocalref_get`: load the stack location
    /// `THREADLOCAL_OFS` into `resloc`, then `load_from_mem` the value at
    /// `offset` from it.
    fn threadlocalref_get(
        &mut self,
        offset: i64,
        resloc: crate::regloc::RegLoc,
        size: usize,
        sign: bool,
    ) {
        rx86::mov_rs(&mut self.mc, resloc.value, THREADLOCAL_OFS as i32);
        self.load_from_mem(
            resloc,
            addr_add_const(&Loc::Reg(resloc), offset),
            size,
            sign,
        );
    }

    /// assembler.py `genop_math_sqrt`: `SQRTSD(arglocs[0], resloc)`.
    /// `_consider_math_sqrt` force-results into the source register, so the
    /// input already sits in `resloc`.
    fn genop_math_sqrt(&mut self, result_loc: Option<&Loc>) {
        if let Some(Loc::Reg(r)) = result_loc {
            rx86::sqrtsd_xx(&mut self.mc, r.value, r.value);
        }
    }

    /// Inline nursery bump for a call tagged
    /// [`majit_ir::RuntimeHelperKind::NurseryAlloc`].
    ///
    /// The tag declares a two-word headerless node taken from the
    /// interpreter's own nursery, whose payload is the call's two arguments:
    /// the value at offset 0 and the successor link at offset 8.
    ///
    /// The fast path takes the head of the allocator's recycle list when it
    /// reports one and has one, otherwise advances `nursery_free`, and stores
    /// those two words. Neither can collect, so it emits no gcmap. The slow
    /// path is the ordinary residual call wrapper, which may collect inside
    /// the callee; the callee is free to treat the second argument as a
    /// keep-root, since that is the object the new node links to. This is
    /// sound because the op is still a call: the optimizer's residual-call
    /// emission fences pending setfields before the allocation, so a collector
    /// that walks the interpreter's own structures finds roots that are
    /// current. A collector that is not headerless-aware
    /// (`dynasm_headerless_nursery_addrs`) gets the residual call alone.
    fn genop_nursery_alloc_inline_x86(&mut self, op: &Op, arglocs: &[Loc]) {
        // Two-word headerless node: value@0, link@8.
        const NURSERY_ALLOC_NODE_SIZE: i32 = 16;

        let (nf_addr, nt_addr) = crate::runner::dynasm_headerless_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }

        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        let (Some(&value_loc), Some(&next_loc)) =
            (arglocs.get(func_index + 1), arglocs.get(func_index + 2))
        else {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        };

        let fn_loc = arglocs.get(func_index).copied();
        let source_regs = [
            fn_loc.and_then(|loc| loc.as_reg()),
            value_loc.as_reg(),
            next_loc.as_reg(),
        ];
        let caller_save_scratch = crate::x86::regalloc::SAVE_AROUND_CALL_CORE_REGS;
        let mut picked = Vec::new();
        for &reg in caller_save_scratch {
            if source_regs
                .iter()
                .flatten()
                .any(|source| source.is_core_reg() && source.value == reg.value)
            {
                continue;
            }
            picked.push(reg);
            if picked.len() == 4 {
                break;
            }
        }
        if picked.len() < 4 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }
        let value_reg = picked[0].value;
        let next_reg = picked[1].value;
        let base_reg = picked[2].value;
        let new_free_reg = picked[3].value;
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let nf = nf_addr as i64;
        let nt = nt_addr as i64;

        let value_dst = Loc::Reg(crate::regloc::RegLoc::new(value_reg, false));
        let next_dst = Loc::Reg(crate::regloc::RegLoc::new(next_reg, false));
        self.regalloc_mov(&value_loc, &value_dst);
        self.regalloc_mov(&next_loc, &next_dst);

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        let bump = self.mc.new_dynamic_label();
        let init = self.mc.new_dynamic_label();

        // Take from the recycle list first, exactly as the callee's own
        // allocation order does. A bump-only fast path would agree with the
        // callee only while the current chunk has room: once it fills, an
        // allocator that recycles serves every request from the list, the bump
        // pointer stays at the limit, and the fast path never fires again.
        let recycle_addr = crate::runner::dynasm_nursery_recycle_list_addr();
        if recycle_addr != 0 {
            let rl = recycle_addr as i64;
            self.mc.MOV(crate::regloc::X86_64_SCRATCH_REG, imm(rl));
            rx86::mov_rm(&mut self.mc, base_reg, (scratch, 0)); // cell = *recycle_head
            dynasm!(self.mc ; .arch x64
            ; test Rq(base_reg), Rq(base_reg)
            );
            dynasm!(self.mc ; .arch x64
                            ; jz =>bump                                         // empty -> bump instead
            );
            rx86::mov_rm(&mut self.mc, new_free_reg, (base_reg, 8)); // cell's link word
            rx86::mov_mr(&mut self.mc, (scratch, 0), new_free_reg); // *recycle_head = link
            dynasm!(self.mc ; .arch x64
                            ; jmp =>init

            );
            self.mc.forget_scratch_register();
        }

        // CallR regalloc has already spilled/moved caller-save registers via
        // before_call(). Use only those freed registers plus R11, and load
        // value/next before staging nursery slot addresses so their original
        // argloc source registers stay intact for the slow residual call.
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64
        ; =>bump
        );
        self.mc.MOV(Operand::R(base_reg), heap(nf)); // base = *nursery_free
        rx86::lea_rm(
            &mut self.mc,
            new_free_reg,
            (base_reg, NURSERY_ALLOC_NODE_SIZE),
        );
        self.mc.CMP(Operand::R(new_free_reg), heap(nt));
        dynasm!(self.mc ; .arch x64
                    ; ja =>slow_path
        );
        self.mc.MOV(heap(nf), Operand::R(new_free_reg)); // *nursery_free = base + 16
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64
                    ; =>init
        );
        rx86::mov_mr(&mut self.mc, (base_reg, 0), value_reg); // value @ base+0
        rx86::mov_mr(&mut self.mc, (base_reg, 8), next_reg); // next @ base+8
        dynasm!(self.mc ; .arch x64
                    ; mov rax, Rq(base_reg)                                 // result = base

        );
        dynasm!(self.mc ; .arch x64 ; jmp =>done);
        self.mc.forget_scratch_register();

        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>slow_path);
        self.genop_call_with_arglocs(op, arglocs);
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>done);
    }

    /// Inline recycle push for a `raw_free` whose allocator publishes a
    /// recycle window.
    ///
    /// Publishing the window and the list head declares that releasing a cell
    /// inside the window is exactly `cell.link = *head; *head = cell`, with
    /// the link word at offset 8 — the same two-word headerless cell
    /// [`Self::genop_nursery_alloc_inline_x86`] hands out. An allocator that
    /// publishes neither keeps the whole release on the residual call, and so
    /// does a cell the window rejects, which is how a null pointer and a cell
    /// the allocator no longer owns reach the callee's own tests.
    ///
    /// The op stays a call, so `before_call` has already put the register file
    /// in its across-the-call shape; what this replaces is the branch to the
    /// callee and the callee's own body. It cannot collect and emits no
    /// gcmap, matching the descr the release carries.
    fn genop_nursery_free_inline_x86(&mut self, op: &Op, arglocs: &[Loc]) {
        let window_addr = crate::runner::dynasm_nursery_recycle_window_addr();
        let head_addr = crate::runner::dynasm_nursery_recycle_list_addr();
        if window_addr == 0 || head_addr == 0 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }
        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        let Some(&cell_loc) = arglocs.get(func_index + 1) else {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        };

        // Two caller-save registers the call's own operands do not occupy, so
        // the slow path still finds its arglocs where regalloc left them.
        let fn_loc = arglocs.get(func_index).copied();
        let source_regs = [fn_loc.and_then(|loc| loc.as_reg()), cell_loc.as_reg()];
        let mut picked = Vec::new();
        for &reg in crate::x86::regalloc::SAVE_AROUND_CALL_CORE_REGS {
            if source_regs
                .iter()
                .flatten()
                .any(|source| source.is_core_reg() && source.value == reg.value)
            {
                continue;
            }
            picked.push(reg);
            if picked.len() == 2 {
                break;
            }
        }
        if picked.len() < 2 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }
        let cell_reg = picked[0].value;
        let offset_reg = picked[1].value;
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let window = window_addr as i64;
        let head = head_addr as i64;

        let cell_dst = Loc::Reg(crate::regloc::RegLoc::new(cell_reg, false));
        self.regalloc_mov(&cell_loc, &cell_dst);

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();

        self.mc.MOV(crate::regloc::X86_64_SCRATCH_REG, imm(window));
        dynasm!(self.mc ; .arch x64
                    ; mov Rq(offset_reg), Rq(cell_reg)
        );
        rx86::sub_rm(&mut self.mc, offset_reg, (scratch, 0)); // cell - base
        rx86::cmp_rm(&mut self.mc, offset_reg, (scratch, 8)); // vs width
        dynasm!(self.mc ; .arch x64
                    ; jae =>slow_path                                       // outside the window
        );
        self.mc.MOV(crate::regloc::X86_64_SCRATCH_REG, imm(head));
        rx86::mov_rm(&mut self.mc, offset_reg, (scratch, 0)); // old head
        rx86::mov_mr(&mut self.mc, (cell_reg, 8), offset_reg); // cell's link word
        rx86::mov_mr(&mut self.mc, (scratch, 0), cell_reg); // *recycle_head = cell
        dynasm!(self.mc ; .arch x64
                    ; jmp =>done

        );
        self.mc.forget_scratch_register();

        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>slow_path);
        self.genop_call_with_arglocs(op, arglocs);
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>done);
    }

    /// assembler.py call_assembler: invoke a compiled JIT loop.
    ///
    /// RPython fast path (assembler.py:295-360):
    ///   1. _call_assembler_emit_call — call the target trace
    ///   2. _call_assembler_check_descr — CMP jf_descr == done_with_this_frame_descr
    ///   3. Path A (slow): call assembler_helper
    ///   4. Path B (fast): MOV result, [frame + ofs]
    ///   5. join paths
    ///
    /// llsupport/assembler.py `call_assembler` + x86/assembler.py
    /// `_call_assembler_emit_call` parity. Line-by-line port:
    /// 1. simple_call(target, [jf, threadlocal_loc])
    /// 2. CMP [eax + jf_descr_ofs], done_descr_imm
    /// 3. je fast_path
    /// 4. simple_call(asm_helper, [eax, vloc], result_loc)   ← slow path
    /// 5. jmp merge
    /// 6. fast_path: `_call_assembler_load_result` — one MOV or MOVSD
    ///    from the dead frame's value index 0 into eax / xmm0
    /// 7. merge:
    ///
    /// Caller's rbp is preserved by the callee's _call_header/_call_footer
    /// (which push/pop it). After the call we still need
    /// `reload_frame_if_necessary` because a minor GC during the callee
    /// may have moved the caller jitframe; the popped rbp is the
    /// pre-GC address while the shadow stack carries the updated one.
    fn genop_call_assembler(&mut self, op: &Op, arglocs: &[Loc], result_loc: Option<&Loc>) {
        // handle_call_assembler (rewrite.py) always pre-builds the
        // callee jitframe — storing every inputarg, and for a virtualizable
        // passing the forced vable object as the second arg — so the backend
        // only loads arglocs[0] (the rewritten frame) and invokes the target.
        let __descr_arc_call_descr = op.getdescr();
        let frame_loc = arglocs
            .first()
            .copied()
            .expect("call_assembler missing rewritten jitframe arg");
        let vable_loc = arglocs.get(1).copied();

        let target_addr = self.resolve_call_assembler_target_addr(__descr_arc_call_descr.as_ref());
        let is_resolved = target_addr.is_available() || self.self_entry_label.is_some();
        let result_type = op.opcode.result_type();
        let done_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
        let helper_addr = crate::call_assembler_helper_addr() as i64;
        let green_key = self.header_pc as i64;

        if !is_resolved {
            // Unresolved target: emit force-fn dispatch through
            // r12-saved rbp (kept as-is — this path is rare and not
            // on the recursive hot path).
            self.emit_load_to_rax(frame_loc);
            dynasm!(self.mc ; .arch x64 ; mov rdx, rax);
            let force_addr = crate::call_assembler_force_fn_addr() as i64;
            if force_addr != 0 {
                if let Some(vloc) = vable_loc {
                    self.emit_load_to_rax(vloc);
                    self.emit_abi_int_arg_from_reg(0, 0);
                } else {
                    rx86::mov_rm(
                        &mut self.mc,
                        rx86::EAX,
                        (rx86::EDX, FIRST_ITEM_OFFSET as i32),
                    );
                    self.emit_abi_int_arg_from_reg(0, 0);
                }
                let pushed_gcmap = self.push_pending_call_gcmap();
                self.mc.MOV(crate::regloc::EAX, imm(force_addr));
                self.emit_abi_call_rax_aligned();
                self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
            } else {
                dynasm!(self.mc ; .arch x64 ; xor eax, eax);
            }
            if result_type == Type::Float {
                // The force helper returns the bits in rax.
                // `_call_assembler_load_result` leaves a float in xmm0.
                rx86::movdq_xr(&mut self.mc, 0, rx86::EAX);
            }
            self.move_call_assembler_result(result_type, result_loc);
            return;
        }

        // ── x86/assembler.py `_call_assembler_emit_call` ──
        // `simple_call(addr, [argloc, threadlocal_loc])`. Branch directly
        // to the resolved callee entry — skip the Rust trampoline, which
        // would otherwise add an extra indirect call and (when
        // MAJIT_LOG was probed) a `std::env::var_os` per recursion.
        let eax = Loc::Reg(crate::regloc::EAX);
        if let Some(addr) = target_addr.immediate {
            let threadlocal_loc =
                Loc::Esp(crate::regloc::RawEspLoc::new(THREADLOCAL_OFS as i32, false));
            self.simple_call(
                Loc::immed(addr as i64),
                &[frame_loc, threadlocal_loc],
                Some(eax),
            );
        } else {
            // The self-entry address is only known through
            // `self_entry_addr_ptr`; the arguments go where
            // `CallBuilder64.prepare_arguments` would put them. The frame
            // moves before the thread-local load, which writes
            // `ARGUMENTS_GPR[1]`.
            let arg0 = super::callbuilder::ARGUMENTS_GPR[0].value;
            let arg1 = super::callbuilder::ARGUMENTS_GPR[1].value;
            self.regalloc_mov(
                &frame_loc,
                &Loc::Reg(crate::regloc::RegLoc::new(arg0, false)),
            );
            rx86::mov_rs(&mut self.mc, arg1, THREADLOCAL_OFS as i32);
            let pushed_gcmap = self.push_pending_call_gcmap();
            let addr_ptr = self.self_entry_addr_ptr as i64;
            let eax = crate::regloc::EAX;
            self.mc.MOV(eax, imm(addr_ptr));
            self.mc.MOV(eax, Operand::M(eax.value, 0));
            self.emit_abi_call_rax_aligned();
            // Callee's _call_footer popped caller's rbp (= pre-GC
            // address). Reload from shadow stack so subsequent
            // frame-relative ops hit the moved jitframe.
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
        }

        // ── x86/assembler.py _call_assembler_check_descr ──
        // `CMP(mem(eax, ofs), imm(value))`.
        let fast_path = self.mc.new_dynamic_label();
        let merge = self.mc.new_dynamic_label();
        self.mc.CMP(
            Operand::M(rx86::EAX, i64::from(JF_DESCR_OFS)),
            imm(done_descr_ptr),
        );
        dynasm!(self.mc ; .arch x64
                    ; je =>fast_path

        );

        // ── Path A: x86/assembler.py _call_assembler_emit_helper_call ──
        // simple_call(asm_helper, [tmploc=rax, vloc], result_loc).
        // pyre's helper signature is (cpu_handle, callee_jf,
        // green_key) — see compile.py:665.
        let cpu_ptr = self.cpu_handle_ptr();
        self.simple_call(
            Loc::immed(helper_addr),
            &[Loc::immed(cpu_ptr), eax, Loc::immed(green_key)],
            Some(eax),
        );
        if result_type == Type::Float {
            // `call_assembler_helper_trampoline` returns the bits in rax.
            // `_call_assembler_load_result` leaves a float in xmm0.
            rx86::movdq_xr(&mut self.mc, 0, rx86::EAX);
        }
        dynasm!(self.mc ; .arch x64
            ; jmp =>merge
            ; =>fast_path
        );
        self.mc.forget_scratch_register();

        // x86/assembler.py `_call_assembler_load_result`: one load from the
        // dead frame's value index 0. A float stays in xmm0; int/ref/void
        // stay in eax (`call_assembler` asserts the int/ref result is eax).
        if result_type == Type::Float {
            rx86::movsd_xm(&mut self.mc, 0, (rx86::EAX, FIRST_ITEM_OFFSET as i32));
        } else {
            rx86::mov_rm(
                &mut self.mc,
                rx86::EAX,
                (rx86::EAX, FIRST_ITEM_OFFSET as i32),
            );
        }
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>merge);
        self.move_call_assembler_result(result_type, result_loc);
    }

    /// Move the value `_call_assembler_load_result` left in eax / xmm0 into
    /// the regalloc result register. `after_call` binds eax for an int or ref
    /// and xmm0 for a float, so that move is usually nothing.
    fn move_call_assembler_result(&mut self, result_type: Type, result_loc: Option<&Loc>) {
        match (result_type, result_loc) {
            (Type::Void, None) => {}
            (Type::Float, Some(Loc::Reg(r))) if r.is_xmm => {
                if r.value != 0 {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(r.value), Rx(0));
                }
            }
            (_, Some(Loc::Reg(r))) if !r.is_xmm => {
                self.mc.MOV(*r, crate::regloc::EAX);
            }
            _ => panic!(
                "CALL_ASSEMBLER result must use its regalloc result register: \
                 type={result_type:?} loc={result_loc:?}"
            ),
        }
    }

    /// x86/assembler.py _write_barrier_fastpath parity.
    fn emit_write_barrier_fastpath(&mut self, regalloc: &RegAlloc<'_>, op: &Op, arglocs: &[Loc]) {
        let is_array = op.opcode == majit_ir::OpCode::CondCallGcWbArray;
        self.emit_write_barrier_fastpath_kind(Some(regalloc), arglocs, is_array, false);
    }

    /// `regalloc` is `self._regalloc`, `None` outside a walk.
    fn emit_write_barrier_fastpath_kind(
        &mut self,
        regalloc: Option<&RegAlloc<'_>>,
        arglocs: &[Loc],
        is_array: bool,
        is_frame: bool,
    ) {
        // x86/assembler.py:2399-2401 asserts the descriptor is the collector's
        // write-barrier class. `COND_CALL_GC_WB` only exists because the GC
        // rewriter emitted it, so a missing descriptor here means the two
        // disagree; returning would drop the barrier without a trace.
        let wb = crate::runner::dynasm_write_barrier_descr()
            .expect("COND_CALL_GC_WB emitted without a write barrier descriptor");
        let mut card_marking = false;
        let mut loc_index = None;
        let mut mask = wb.jit_wb_if_flag_singlebyte as i8;
        if is_array && wb.jit_wb_cards_set != 0 {
            // assumptions the rest of the function depends on:
            assert_eq!(wb.jit_wb_cards_set_byteofs, wb.jit_wb_if_flag_byteofs);
            assert_eq!(wb.jit_wb_cards_set_singlebyte, -0x80);
            card_marking = true;
            loc_index = Some(
                *arglocs
                    .get(1)
                    .expect("COND_CALL_GC_WB_ARRAY card marking needs the index loc"),
            );
            mask = wb.jit_wb_if_flag_singlebyte as i8 | -0x80;
        }
        // `loc_base = arglocs[0]`. `is_frame` uses `raw_stack`; otherwise
        // `addr_add_const`, which accepts an immediate base (`AddressLoc`
        // code `'j'`, or `_addr_as_reg_offset` when the address does not
        // fit a signed disp32).
        let loc_base = match arglocs.first() {
            Some(loc @ (Loc::Reg(_) | Loc::Immed(_) | Loc::ImmedFloat(_))) => *loc,
            other => panic!("write barrier base must be a register or an immediate, got {other:?}"),
        };
        debug_assert!(!is_frame || matches!(loc_base, Loc::Reg(r) if r == crate::regloc::EBP));
        let byteofs = wb.jit_wb_if_flag_byteofs;

        let helper_num = if is_frame {
            4
        } else {
            // `self._regalloc is not None and len(self._regalloc.xrm.reg_bindings)`.
            let withfloats = regalloc.is_some_and(|ra| ra.xrm.reg_bindings_len() > 0);
            usize::from(card_marking) + 2 * usize::from(withfloats)
        };
        let helper = self.wb_slowpath[helper_num];
        assert!(
            helper != 0,
            "wb_slowpath[{helper_num}] was not built (X86CpuExt::ensure_wb_slowpath)"
        );

        // `TEST8(loc, imm(mask))`. A frame base is `[rbp + byteofs]`.
        let loc = addr_add_const(&loc_base, i64::from(byteofs));
        self.mc.TEST8(loc, imm(i64::from(mask)));
        // `_write_barrier_fastpath`: `TEST8` then `WriteBarrierSlowPath` on
        // `NZ`. The flag-clear edge falls through `set_continue_addr`.
        // The helper call and the card mark live in `generate_body`.
        let mut sp = self.emit_slow_jcc(
            CC_NE,
            SlowPathKind::WriteBarrier {
                loc_base,
                loc_index,
                helper_num,
                card_marking,
                card_page_shift: wb.jit_wb_card_page_shift,
            },
        );
        self.set_continue_here(&mut sp);
        self.pending_slowpaths.push(sp);
    }

    /// Card mark. A register base keeps `encode_wb_array_card_mark`. An
    /// immediate base uses `addr_add_const` so the address is not treated
    /// as a GPR. The `-GcHeader::SIZE` bias matches that register form.
    fn mark_wb_cards(&mut self, loc_base: Loc, loc_index: &Loc, page_shift: u32) {
        match loc_base {
            Loc::Reg(base) => {
                encode_wb_array_card_mark(&mut self.mc, base.value, loc_index, page_shift);
            }
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                self.mark_wb_cards_imm(i.value, loc_index, page_shift);
            }
            other => panic!("write barrier base {other:?}"),
        }
    }

    fn mark_wb_cards_imm(&mut self, base: i64, loc_index: &Loc, page_shift: u32) {
        let header = majit_gc::header::GcHeader::SIZE as i64;
        match loc_index {
            Loc::Reg(idx) => {
                // Non-register base: save `loc_index`, mutate it, `BTS` the
                // staged address, then pop. r11 only stages the address.
                dynasm!(self.mc ; .arch x64 ; push Rq(idx.value));
                rx86::shr_ri(&mut self.mc, idx.value, page_shift as i32);
                rx86::xor_ri(&mut self.mc, idx.value, -8);
                self.mc.BTS(heap(base.wrapping_sub(header)), *idx);
                dynasm!(self.mc ; .arch x64 ; pop Rq(idx.value));
            }
            Loc::Frame(idx) => {
                // The bit index and the address scratch cannot both be r11.
                dynasm!(self.mc ; .arch x64 ; push rax);
                rx86::mov_rb(&mut self.mc, rx86::EAX, idx.ebp_loc.value);
                rx86::shr_ri(&mut self.mc, rx86::EAX, page_shift as i32);
                rx86::xor_ri(&mut self.mc, rx86::EAX, -8);
                self.mc
                    .BTS(heap(base.wrapping_sub(header)), crate::regloc::EAX);
                dynasm!(self.mc ; .arch x64 ; pop rax);
            }
            Loc::Immed(idx) | Loc::ImmedFloat(idx) => {
                let byte_index = idx.value >> page_shift;
                let byte_ofs = !((byte_index >> 3) as i64) - header;
                let byte_val = 1_i64 << (byte_index & 7);
                self.mc.OR8(
                    heap(base.wrapping_add(byte_ofs)),
                    imm(i64::from(byte_val as i8)),
                );
            }
            _ => panic!("index is neither RegLoc nor ImmedLoc"),
        }
    }

    fn mov_gpr_from_loc(&mut self, dst: u8, loc: Loc) {
        self.mc.MOV(Operand::R(dst), &loc);
    }

    /// `_imul_const_scaled`. Returns the residual scale (0..3) the outer
    /// `LEA_ra` applies. `target = source * itemsize` once that scale is
    /// shifted back in.
    fn imul_const_scaled(&mut self, targetreg: u8, sourcereg: u8, itemsize: i64) -> u8 {
        debug_assert!(itemsize > 0);
        let mut reduced = itemsize as usize;
        let shift = if reduced & 7 == 0 {
            3
        } else if reduced & 3 == 0 {
            2
        } else if reduced & 1 == 0 {
            1
        } else {
            0
        };
        reduced >>= shift;
        if crate::regalloc::valid_addressing_size(reduced.wrapping_sub(1)) {
            let scale = crate::regalloc::get_scale(reduced - 1);
            rx86::lea_ra(
                &mut self.mc,
                targetreg,
                (i16::from(sourcereg), sourcereg, scale, 0),
            );
        } else if crate::regalloc::valid_addressing_size(reduced) {
            let scale = crate::regalloc::get_scale(reduced);
            rx86::lea_ra(
                &mut self.mc,
                targetreg,
                (rx86::NO_BASE_REGISTER, sourcereg, scale, 0),
            );
        } else {
            assert!(
                rx86::fits_in_32bits(reduced as i64),
                "_imul_const_scaled immediate {reduced} does not fit"
            );
            rx86::imul_rri(&mut self.mc, targetreg, sourcereg, reduced as i32);
        }
        shift as u8
    }

    /// `push_gcmap(..., store=True)`. `gcmap == 0` writes a null map.
    fn publish_malloc_gcmap(&mut self, gcmap: usize) {
        if gcmap != 0 {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            rx86::mov_bi(&mut self.mc, crate::jitframe::JF_GCMAP_OFS, 0);
        }
    }

    /// `CALL(imm(malloc_slowpath*))` via r11, so the trampoline's
    /// `_push_all_regs_to_frame` saves the caller's rax.
    fn call_malloc_helper(&mut self, addr: usize) {
        assert!(
            addr != 0,
            "malloc slowpath trampoline was not built (_build_malloc_slowpath)"
        );
        self.mc.CALL(imm(addr as i64));
    }

    /// The trampoline returns the payload in ecx (`MOV ecx, eax`).
    fn move_malloc_result(&mut self, result_reg: u8) {
        self.mc.MOV(Operand::R(result_reg), crate::regloc::ECX);
    }

    /// `malloc_cond`: nursery bump, `CMP`, not-taken `J_il(A)` into
    /// `MallocCondSlowPath`, then fall-through `MOV [nursery_free], edx`.
    ///
    /// The payload `LEA` by `GcHeader::SIZE` stays on that fall-through,
    /// before `set_continue_addr`. The fixed trampoline returns the object
    /// in ecx and leaves edx clobbered, so the store of `nursery_free`
    /// runs only when the `ja` is not taken.
    fn genop_call_malloc_nursery(&mut self, op: &Op, result_loc: Option<&Loc>) {
        let size_ref = op.arg(0).to_opref();
        let total_size = size_ref.inline_const_bits().unwrap_or_else(|| {
            self.constants
                .get(&size_ref.raw())
                .map(|c| c.as_raw_i64())
                .unwrap_or(size_ref.raw() as i64)
        });
        debug_assert!(
            total_size & (WORD as i64 - 1) == 0,
            "malloc_cond size must be word-aligned"
        );
        let result_reg = match result_loc {
            Some(Loc::Reg(r)) => r.value,
            _ => rx86::ECX,
        };
        let gcmap = self.pending_malloc_nursery_gcmap.unwrap_or(0);
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        self.mc.MOV(Operand::R(rx86::ECX), heap(nf_addr as i64));
        rx86::lea_rm(&mut self.mc, rx86::EDX, (rx86::ECX, total_size as i32));
        self.mc.CMP(Operand::R(rx86::EDX), heap(nt_addr as i64));
        let mut sp = self.begin_slow_path(
            Some(CC_A),
            SlowPathKind::MallocCond {
                gcmap,
                result_reg,
                synthesize: None,
            },
        );
        self.mc.MOV(heap(nf_addr as i64), Operand::R(rx86::EDX));
        rx86::lea_rm(
            &mut self.mc,
            result_reg,
            (rx86::ECX, majit_gc::header::GcHeader::SIZE as i32),
        );
        self.set_continue_here(&mut sp);
        self.pending_slowpaths.push(sp);
    }

    /// `malloc_cond_varsize_frame`. When `sizeloc` is ecx it is staged in
    /// edx first. The `LEA` displacement is `GcHeader::SIZE` so
    /// `_build_malloc_slowpath`'s `sub rdx, rcx` recovers the byte count
    /// `dynasm_nursery_slowpath` allocates. `malloc_cond_varsize_frame`
    /// adds with displacement 0; this addend is the excluded header bias,
    /// including when `sizeloc` is edx.
    fn malloc_cond_varsize_frame(&mut self, arglocs: &[Loc], result_loc: Option<&Loc>) {
        let sizeloc = match arglocs.first() {
            Some(Loc::Reg(r)) => r.value,
            other => {
                panic!("CallMallocNurseryVarsizeFrame size arg must be Loc::Reg, got {other:?}")
            }
        };
        let result_reg = match result_loc {
            Some(Loc::Reg(r)) => r.value,
            other => {
                panic!("CallMallocNurseryVarsizeFrame result_loc must be Loc::Reg, got {other:?}")
            }
        };
        let mut sv = sizeloc;
        if sv == rx86::ECX {
            dynasm!(self.mc ; .arch x64 ; mov rdx, rcx);
            sv = rx86::EDX;
        }
        let gc_header_size = majit_gc::header::GcHeader::SIZE as i32;
        let gcmap = self.pending_malloc_nursery_gcmap.unwrap_or(0);
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            let mut sp = self.begin_slow_path(
                None,
                SlowPathKind::MallocCond {
                    gcmap,
                    result_reg,
                    synthesize: Some((sv, gc_header_size)),
                },
            );
            self.set_continue_here(&mut sp);
            self.pending_slowpaths.push(sp);
            return;
        }
        self.mc.MOV(Operand::R(rx86::ECX), heap(nf_addr as i64));
        rx86::lea_ra(
            &mut self.mc,
            rx86::EDX,
            (i16::from(rx86::ECX), sv, 0, gc_header_size),
        );
        self.mc.CMP(Operand::R(rx86::EDX), heap(nt_addr as i64));
        let mut sp = self.begin_slow_path(
            Some(CC_A),
            SlowPathKind::MallocCond {
                gcmap,
                result_reg,
                synthesize: None,
            },
        );
        self.mc.MOV(heap(nf_addr as i64), Operand::R(rx86::EDX));
        // `gen_initialize_tid_keep_flags` stores only the type-id half.
        // Clear the whole header word on this fast path so a recycled
        // nursery slot cannot keep `GCFLAG_HAS_SHADOW`, `GCFLAG_PINNED`,
        // or the high half of `FORWARDED_MARKER`. The slow path does not
        // clear: a young raw-malloced frame carries `YOUNG_RAWMALLOC`,
        // and `set_continue_here` is after this store.
        rx86::mov_mi(&mut self.mc, (rx86::ECX, 0), 0);
        rx86::lea_rm(&mut self.mc, result_reg, (rx86::ECX, gc_header_size));
        self.set_continue_here(&mut sp);
        self.pending_slowpaths.push(sp);
    }

    /// `malloc_cond_varsize`. The maxlength `ja` lands on
    /// the nursery-overflow `ja` (`emit_forward_jump` / `patch_forward_jump`),
    /// so both failures share one not-taken `J_il(A)`.
    fn malloc_cond_varsize(&mut self, op: &Op, arglocs: &[Loc], result_loc: Option<&Loc>) {
        let (base_size, type_id) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.type_id() as i64))
            .expect("CallMallocNurseryVarsize requires an ArrayDescr");
        let itemsize = match arglocs.get(1) {
            Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
            _ => panic!("malloc_cond_varsize itemsize must be an immediate, got {arglocs:?}"),
        };
        let lengthloc = match arglocs.first() {
            Some(loc) => *loc,
            None => panic!("malloc_cond_varsize requires a length loc"),
        };
        let kind = match arglocs.get(2) {
            Some(Loc::Immed(i)) => i.value,
            _ => panic!("malloc_cond_varsize kind must be an immediate, got {arglocs:?}"),
        };
        debug_assert!(itemsize > 0);
        debug_assert!(
            base_size as usize + majit_gc::header::GcHeader::SIZE
                >= majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE
        );
        let result_reg = match result_loc {
            Some(Loc::Reg(r)) => r.value,
            other => {
                panic!("CallMallocNurseryVarsize result_loc must be a register; got {other:?}")
            }
        };
        let gcmap = self.pending_malloc_nursery_gcmap.unwrap_or(0);
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        let max_young = crate::runner::dynasm_max_size_of_young_obj();
        let max_length = max_young.saturating_sub(2 * WORD);
        let slow_kind = || SlowPathKind::MallocCondVarsize {
            gcmap,
            result_reg,
            kind,
            itemsize,
            type_id,
            lengthloc,
        };
        if nf_addr == 0 || nt_addr == 0 || max_length == 0 {
            let mut sp = self.begin_slow_path(None, slow_kind());
            self.set_continue_here(&mut sp);
            self.pending_slowpaths.push(sp);
            return;
        }
        let mut varsizeloc = match lengthloc {
            Loc::Reg(r) => {
                debug_assert_ne!(r.value, rx86::ECX);
                debug_assert_ne!(r.value, rx86::EDX);
                r.value
            }
            other => {
                self.mov_gpr_from_loc(rx86::EDX, other);
                rx86::EDX
            }
        };
        self.mc.CMP(Operand::R(varsizeloc), imm(max_length as i64));
        let nursery_ja = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; ja =>nursery_ja);
        self.mc.MOV(Operand::R(rx86::ECX), heap(nf_addr as i64));
        let shift = if crate::regalloc::valid_addressing_size(itemsize as usize) {
            crate::regalloc::get_scale(itemsize as usize)
        } else {
            let shift = self.imul_const_scaled(rx86::EDX, varsizeloc, itemsize);
            varsizeloc = rx86::EDX;
            shift
        };
        let mut constsize = base_size + majit_gc::header::GcHeader::SIZE as i64;
        let force_realignment = (itemsize as usize) % WORD != 0;
        if force_realignment {
            constsize += WORD as i64 - 1;
        }
        assert!(
            rx86::fits_in_32bits(constsize),
            "malloc_cond_varsize constsize {constsize} does not fit a displacement"
        );
        rx86::lea_ra(
            &mut self.mc,
            rx86::EDX,
            (i16::from(rx86::ECX), varsizeloc, shift, constsize as i32),
        );
        if force_realignment {
            rx86::and_ri(&mut self.mc, rx86::EDX, !(WORD as i32 - 1));
        }
        self.mc.CMP(Operand::R(rx86::EDX), heap(nt_addr as i64));
        dynasm!(self.mc ; .arch x64 ; =>nursery_ja);
        let mut sp = self.begin_slow_path(Some(CC_A), slow_kind());
        self.mc.MOV(Operand::M(rx86::ECX, 0), imm(type_id));
        self.mc.MOV(heap(nf_addr as i64), Operand::R(rx86::EDX));
        rx86::lea_rm(
            &mut self.mc,
            result_reg,
            (rx86::ECX, majit_gc::header::GcHeader::SIZE as i32),
        );
        self.set_continue_here(&mut sp);
        self.pending_slowpaths.push(sp);
    }

    /// Headerless varsize nursery allocation. The fast path keeps the
    /// `imul` / `and` / taken `jmp` sequence; overflow calls
    /// `dynasm_nursery_slowpath_headerless_varsize` through `push_all_regs`
    /// and an ABI call. This is not `malloc_cond_varsize`.
    fn genop_call_malloc_nursery_varsize_headerless(
        &mut self,
        op: &Op,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
    ) {
        let base_size = op
            .with_array_descr(|ad| ad.base_size())
            .expect("CallMallocNurseryVarsizeHeaderless requires an ArrayDescr")
            as i64;
        let itemsize = match arglocs.get(1) {
            Some(Loc::Immed(i) | Loc::ImmedFloat(i)) => i.value,
            _ => 8,
        };
        let (nf_addr, nt_addr) = crate::runner::dynasm_headerless_nursery_addrs();
        let max_young = crate::runner::dynasm_max_size_of_young_obj();
        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        let max_length = max_young.saturating_sub(base_size as usize);
        debug_assert!(itemsize > 0);
        if nf_addr == 0 || nt_addr == 0 || max_length == 0 {
            dynasm!(self.mc ; .arch x64 ; jmp =>slow_path);
            self.mc.forget_scratch_register();
        } else {
            match arglocs.first() {
                Some(Loc::Reg(len_r)) => {
                    debug_assert_ne!(len_r.value, crate::regloc::ECX.value);
                    debug_assert_ne!(len_r.value, crate::regloc::EDX.value);
                    dynasm!(self.mc ; .arch x64 ; mov rdx, Rq(len_r.value));
                }
                Some(Loc::Immed(len_i) | Loc::ImmedFloat(len_i)) => {
                    rx86::mov_ri(&mut self.mc, rx86::EDX, len_i.value);
                }
                Some(Loc::Frame(len_f)) => {
                    rx86::mov_rb(&mut self.mc, rx86::EDX, len_f.ebp_loc.value);
                }
                Some(Loc::Ebp(len_e)) => {
                    rx86::mov_rb(&mut self.mc, rx86::EDX, len_e.value);
                }
                other => {
                    panic!("CallMallocNurseryVarsizeHeaderless length is not a value: {other:?}")
                }
            }
            self.mc.CMP(Operand::R(rx86::EDX), imm(max_length as i64));
            dynasm!(self.mc ; .arch x64
                                    ; ja =>slow_path
            );
            self.mc.MOV(Operand::R(rx86::ECX), heap(nf_addr as i64));
            rx86::imul_ri(&mut self.mc, rx86::EDX, itemsize as i32);
            rx86::add_ri(&mut self.mc, rx86::EDX, (base_size + 7) as i32);
            dynasm!(self.mc ; .arch x64
            ; and rdx, -8
            );
            dynasm!(self.mc ; .arch x64
                                    ; add rdx, rcx
            );
            self.mc.CMP(Operand::R(rx86::EDX), heap(nt_addr as i64));
            dynasm!(self.mc ; .arch x64
                                    ; ja =>slow_path
            );
            self.mc.MOV(heap(nf_addr as i64), Operand::R(rx86::EDX));
            dynasm!(self.mc ; .arch x64
                                    ; jmp =>done

            );
            self.mc.forget_scratch_register();
        }
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>slow_path);
        self.push_all_regs_to_jitframe(&[], true);
        match arglocs.first() {
            Some(Loc::Reg(len_r)) => {
                self.emit_abi_int_arg_from_reg(0, len_r.value as u8);
            }
            Some(Loc::Immed(len_i) | Loc::ImmedFloat(len_i)) => {
                self.emit_abi_int_arg_from_imm(0, len_i.value);
            }
            Some(Loc::Frame(len_f)) => {
                self.emit_abi_int_arg_from_mem(0, len_f.ebp_loc.value);
            }
            Some(Loc::Ebp(len_e)) => {
                self.emit_abi_int_arg_from_mem(0, len_e.value);
            }
            other => panic!("CallMallocNurseryVarsizeHeaderless length is not a value: {other:?}"),
        }
        self.emit_abi_int_arg_from_imm(1, itemsize);
        self.emit_abi_int_arg_from_imm(2, base_size);
        rx86::mov_ri(
            &mut self.mc,
            rx86::EAX,
            crate::runner::dynasm_nursery_slowpath_headerless_varsize as *const () as i64,
        );
        self.emit_abi_call_rax();
        self.reload_frame_if_necessary();
        self.pop_all_regs_from_jitframe(&[crate::regloc::EAX], true);
        self.emit_propagate_exception_if_zero(0);
        let Some(Loc::Reg(r)) = result_loc else {
            panic!(
                "CallMallocNurseryVarsizeHeaderless result_loc must be a register; got {result_loc:?}"
            );
        };
        self.mc.MOV(*r, crate::regloc::EAX);
        dynasm!(self.mc ; .arch x64 ; jmp =>done);
        self.mc.forget_scratch_register();
        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>done);
    }

    /// Headerless fixed-size nursery allocation.  `size` excludes any GC
    /// header; result is the old nursery base.
    ///
    /// The fast path raw-bumps the nursery, so it is correct only when the
    /// active dynasm GC is headerless-aware — its nursery must yield a raw base
    /// carrying no `GcHeader` — what an interpreter that runs its own collector
    /// over its own object arena supplies.  A headered collector such as
    /// MiniMarkGC must never back this op: its nursery walk reads a `GcHeader`
    /// at `base - GcHeader::SIZE`, which a raw base lacks.
    /// `dynasm_headerless_nursery_addrs` answers no nursery for such a
    /// collector, so the op goes straight to the slowpath, where
    /// `alloc_nursery_headerless`'s panicking default fails loudly.
    fn genop_call_malloc_nursery_headerless(&mut self, op: &Op, result_loc: Option<&Loc>) {
        let size_ref = op.arg(0).to_opref();
        let size = size_ref.inline_const_bits().unwrap_or_else(|| {
            self.constants
                .get(&size_ref.raw())
                .map(|c| c.as_raw_i64())
                .unwrap_or(size_ref.raw() as i64)
        });
        let (nf_addr, nt_addr) = crate::runner::dynasm_headerless_nursery_addrs();
        let nf = nf_addr as i64;
        let nt = nt_addr as i64;

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();

        if nf_addr == 0 || nt_addr == 0 {
            // Mirror the headered inactive-nursery path without touching
            // `[nf]`: seed the slowpath's `rdx - rcx` size contract directly.
            dynasm!(self.mc ; .arch x64
            ; xor ecx, ecx
            );
            rx86::mov_ri(&mut self.mc, rx86::EDX, size);
            dynasm!(self.mc ; .arch x64
                            ; jmp =>slow_path

            );
            self.mc.forget_scratch_register();
        } else {
            self.mc.MOV(Operand::R(rx86::ECX), heap(nf));
            rx86::lea_rm(&mut self.mc, rx86::EDX, (rx86::ECX, size as i32));
            self.mc.CMP(Operand::R(rx86::EDX), heap(nt));

            dynasm!(self.mc ; .arch x64 ; ja =>slow_path);
        }

        let result_reg = match result_loc {
            Some(Loc::Reg(r)) => r.value,
            _ => crate::regloc::ECX.value,
        };
        self.mc.MOV(heap(nf), Operand::R(rx86::EDX));
        self.mc.MOV(Operand::R(result_reg), crate::regloc::ECX);
        dynasm!(self.mc ; .arch x64 ; jmp =>done);
        self.mc.forget_scratch_register();

        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>slow_path);
        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            rx86::mov_bi(&mut self.mc, gcmap_ofs, 0);
        }
        let helper_addr = self.malloc_slowpath_headerless as i64;
        self.mc.CALL(imm(helper_addr));
        if let Some(Loc::Reg(r)) = result_loc {
            self.mc.MOV(*r, crate::regloc::ECX);
        }
        let _ = gcmap_ofs;

        self.mc.forget_scratch_register();
        dynasm!(self.mc ; .arch x64 ; =>done);
        // The result register is the delivery contract (see
        // `genop_call_malloc_nursery`); an extra store here grew `frame_depth`
        // by one slot per allocation.
    }

    /// NEW_ARRAY / NEW_ARRAY_CLEAR: allocate a typed GC array.
    ///
    /// Rewrite normally replaces these with `CALL_MALLOC_NURSERY_VARSIZE` /
    /// `malloc_array`. This leftover path must still stamp the collector
    /// type id and write length at the descr's `lendescr` offset — not
    /// libc malloc with a string-header store at +8.
    fn genop_new_array(&mut self, op: &Op, arglocs: &[Loc]) {
        let [len_loc, ..] = arglocs else {
            panic!("varsize allocation expects a length location, got {arglocs:?}");
        };
        let (base_size, item_size, type_id, len_ofs) = op
            .with_array_descr(|ad| {
                (
                    ad.base_size() as i64,
                    ad.item_size() as i64,
                    ad.type_id() as i64,
                    ad.len_descr().map_or(0, |fd| fd.offset() as i64),
                )
            })
            .unwrap_or((8, 8, 0, 0));
        let clear = matches!(op.opcode, OpCode::NewArrayClear) as i64;
        self.emit_load_to_rax(*len_loc);
        // Six integer args. Win64 args 4/5 live at [rsp+32]/[rsp+40].
        // `on_stack` is 6, which fits in `PASS_ON_MY_FRAME`, so the
        // reserve subtracts nothing and the stores land in that scratch.
        // Reserving after the stores would be wrong once `on_stack`
        // exceeds the scratch. SysV keeps args 4/5 in registers.
        #[cfg(target_os = "windows")]
        let call_area_adjust = self.emit_reserve_abi_call_area(0, 2);
        self.emit_abi_int_arg_from_reg(4, 0);
        self.emit_abi_int_arg_from_imm(0, base_size);
        self.emit_abi_int_arg_from_imm(1, item_size);
        self.emit_abi_int_arg_from_imm(2, len_ofs);
        self.emit_abi_int_arg_from_imm(3, type_id);
        self.emit_abi_int_arg_from_imm(5, clear);
        self.mc.MOV(
            crate::regloc::EAX,
            imm(crate::runner::dynasm_malloc_new_array as *const () as i64),
        );
        #[cfg(target_os = "windows")]
        {
            self.mc.CALL(crate::regloc::EAX);
            self.emit_release_abi_call_area(call_area_adjust);
        }
        #[cfg(not(target_os = "windows"))]
        self.emit_abi_call_rax();
    }

    // genop_* — misc

    // assembler.py genop_save_exc_class / genop_save_exception

    /// assembler.py genop_save_exc_class:
    /// `MOV resloc, [pos_exception]`.  The regalloc always assigns the
    /// result a register (`consider_no_arg_result`).
    fn genop_save_exc_class(&mut self, result_loc: Option<&Loc>) {
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        match result_loc {
            Some(Loc::Reg(dst)) => self.mc.MOV(*dst, heap(exc_type_addr)),
            Some(loc @ Loc::Frame(_)) => {
                let tmploc = crate::regloc::X86_64_SCRATCH_REG;
                self.mc.MOV(tmploc, heap(exc_type_addr));
                self.mc.MOV(loc, tmploc);
            }
            None => {}
            Some(other) => {
                panic!("genop_save_exc_class: unhandled result location {other:?}")
            }
        }
    }

    /// `genop_discard_restore_exception` → `_restore_exception(mc,
    /// arglocs[1], arglocs[0])`: `MOV(heap(pos_exc_value), excvalloc)`,
    /// then `MOV(heap(pos_exception), exctploc)`. arglocs = [class, value];
    /// `consider_restore_exception` puts a box in a register and a Const in
    /// an `ImmedLoc` (`return_constant`).
    fn genop_restore_exception(&mut self, arglocs: &[Loc]) {
        if arglocs.len() < 2 {
            return;
        }
        self.mov_heap_loc(crate::jit_exc_value_addr() as i64, &arglocs[1]);
        self.mov_heap_loc(crate::jit_exc_type_addr() as i64, &arglocs[0]);
    }

    /// `MOV(heap(addr), loc)` for a cell that does not fit a disp32:
    /// `_addr_as_reg_offset` turns `'j'` into `'m'`. An immediate that fits
    /// is `MOV_mi`; a wider one is `insn_with_64_bit_immediate`, which
    /// `PUSH`es `find_unused_reg()` (eax), loads it, stores it and `POP`s.
    fn mov_heap_loc(&mut self, addr: i64, loc: &Loc) {
        self.mc.MOV(heap(addr), loc);
    }

    // genop_* — extended integer arithmetic

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    #[allow(dead_code)]
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64
        ; mul rcx
        );
        dynasm!(self.mc ; .arch x64
            ; mov rax, rdx
        );
        self.store_rax_to_result(op.pos().get());
    }

    // genop_* — extended float operations

    /// FLOAT_ABS: result = |arg0|
    #[allow(dead_code)]
    fn genop_float_abs(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        let mask: i64 = i64::MAX; // 0x7FFF_FFFF_FFFF_FFFF
        rx86::mov_ri(&mut self.mc, rx86::EAX, mask);
        dynasm!(self.mc ; .arch x64
        ; movq xmm1, rax
        );
        dynasm!(self.mc ; .arch x64
                    ; andpd xmm0, xmm1

        );
        self.store_d0_to_result(op.pos().get());
    }

    // genop_* — GC memory operations

    /// Emit a sized store of rcx/x1 to [rax]/[x0].
    #[allow(dead_code)]
    fn emit_store_to_rax_sized(&mut self, size: usize) {
        match size {
            1 => dynasm!(self.mc ; .arch x64 ; mov [rax], cl),
            2 => dynasm!(self.mc ; .arch x64 ; mov [rax], cx),
            4 => dynasm!(self.mc ; .arch x64 ; mov [rax], ecx),
            _ => dynasm!(self.mc ; .arch x64 ; mov [rax], rcx),
        }
    }

    /// Resolve an OpRef that is expected to be a compile-time constant.
    fn resolve_const_or(&self, opref: OpRef, default: i64) -> i64 {
        match self.resolve_opref(opref) {
            ResolvedArg::Const(v) => v,
            _ => default,
        }
    }

    /// GC_STORE: store value to base + offset.
    /// 4-arg form: arg(0)=base, arg(1)=offset, arg(2)=value, arg(3)=itemsize.
    #[allow(dead_code)]
    fn genop_discard_gc_store(&mut self, op: &Op) {
        if op.num_args() < 4 {
            return; // 3-arg GC rewrite form — skip for now
        }
        let itemsize = self
            .resolve_const_or(op.arg(3).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch x64 ; pop rax);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// GC_STORE_INDEXED: store to base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=value, arg(3)=scale,
    /// arg(4)=base_offset, arg(5)=itemsize.
    #[allow(dead_code)]
    fn genop_discard_gc_store_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(3).to_opref(), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(4).to_opref(), 0) as i32;
        let itemsize = self
            .resolve_const_or(op.arg(5).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        if scale != 1 {
            rx86::imul_ri(&mut self.mc, rx86::ECX, scale);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if base_offset != 0 {
            rx86::add_ri(&mut self.mc, rx86::EAX, base_offset);
        }
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch x64 ; pop rax);
        self.emit_store_to_rax_sized(itemsize);
    }

    // genop_* — interior field operations

    // genop_* — call variants

    /// `cond_call`: one not-taken `jcc` of `guard_success_cc`, then
    /// `set_continue_addr`. `load_condition_into_cc` emits `test` and
    /// `CC_NE` only when no compare is pending. The call, the gcmap, and
    /// `generate_guard_no_exception` live in `CondCallSlowPath`.
    fn genop_discard_cond_call(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op: &Op,
        arglocs: &[Loc],
        op_index: usize,
    ) {
        if arglocs.len() < 2 {
            panic!(
                "cond_call arglocs are predicate, func, extras (consider_cond_call); got {arglocs:?}"
            );
        }
        self.load_condition_into_cc(&arglocs[0]);
        let cc = self
            .guard_success_cc
            .take()
            .expect("cond_call requires guard_success_cc (load_condition_into_cc)");
        // Predicate at 0, func at 1. Extras are remapped onto
        // `cond_call_register_arguments` in the slow path
        // (`CondCallSlowPath.generate_body`).
        let extras = arglocs[2..].to_vec();
        self.enqueue_cond_call(regalloc, op, op_index, cc, extras, None);
    }

    /// `cond_call` for `COND_CALL_VALUE_I` / `COND_CALL_VALUE_R`.
    /// `consider_cond_call` tests `resloc` and calls when it is zero.
    /// Extra-arg locations follow `[argloc, resloc]`; the slow path remaps
    /// them into `cond_call_register_arguments`. Does not consume
    /// `guard_success_cc`.
    fn genop_cond_call_value(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op: &Op,
        arglocs: &[Loc],
        op_index: usize,
    ) {
        if arglocs.len() < 2 {
            panic!(
                "COND_CALL_VALUE arglocs are [argloc, resloc, extras] (consider_cond_call), got {arglocs:?}"
            );
        }
        let resloc = arglocs[1];
        assert!(
            self.guard_success_cc.is_none(),
            "cond_call value does not consume guard_success_cc"
        );
        self.emit_test_loc(&resloc);
        let extras = arglocs[2..].to_vec();
        self.enqueue_cond_call(regalloc, op, op_index, CC_E, extras, Some(resloc));
    }

    /// `cond_call`: queue `CondCallSlowPath` and bind the continue point
    /// immediately, so the fast path is only the test plus one `jcc`.
    fn enqueue_cond_call(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op: &Op,
        op_index: usize,
        cc: u8,
        arglocs: Vec<Loc>,
        resloc: Option<Loc>,
    ) {
        let imm_func = match self.resolve_opref(op.arg(1).to_opref()) {
            ResolvedArg::Const(val) => val,
            ResolvedArg::Slot(_) => panic!("cond_call func is Const (cond_call)"),
        };
        let Some(gcmap) = self.pending_malloc_nursery_gcmap else {
            panic!("cond_call gcmap is missing (perform_with_gcmap_ptr)");
        };
        if gcmap == 0 {
            panic!("cond_call published a null gcmap");
        }
        let expects_guard_no_exception = self
            .operations
            .get(op_index + 1)
            .is_some_and(|next| next.opcode == OpCode::GuardNoException);
        // `sp.should_be_saved = self._regalloc.rm.reg_bindings.values()`, kept
        // as a bitmask over GPR numbers. `variant_num = floats * 2 +
        // callee_only`: the for/else over the same bindings sets
        // `callee_only` when every bound core register is in
        // `save_around_call_regs` (no bindings takes that arm), and `floats`
        // is `len(self._regalloc.xrm.reg_bindings) != 0`.
        let mut should_be_saved = 0i64;
        let mut callee_only = true;
        for (_, reg) in regalloc.rm.reg_bindings_iter() {
            should_be_saved |= 1i64 << reg.value;
            if !regalloc.rm.save_around_call_regs.contains(&reg) {
                callee_only = false;
            }
        }
        let floats = regalloc.xrm.reg_bindings_len() > 0;
        let variant_num = usize::from(floats) * 2 + usize::from(callee_only);
        let mut sp = self.emit_slow_jcc(
            cc,
            SlowPathKind::CondCall {
                gcmap,
                imm_func,
                arglocs,
                resloc,
                variant_num,
                should_be_saved,
                guard_token_no_exception: None,
                expects_guard_no_exception,
            },
        );
        self.set_continue_here(&mut sp);
        self.pending_slowpaths.push(sp);
    }

    // genop_* — string/array operations

    /// assembler.py `load_effective_addr`:
    /// `result = base + (index << shift) + baseofs`.
    /// `resoperation.py` args `[v_gcptr, v_index, c_baseofs, c_shift]`.
    fn genop_load_effective_address(&mut self, arglocs: &[Loc], result_loc: Option<&Loc>) {
        let Some(Loc::Reg(dst)) = result_loc else {
            panic!("LoadEffectiveAddress result_loc must be Loc::Reg, got {result_loc:?}");
        };
        let [base, index, baseofs, shift] = match arglocs {
            [a, b, c, d, ..] => [a, b, c, d],
            other => panic!("LoadEffectiveAddress expects 4 arglocs, got {other:?}"),
        };
        let shift_amt = match shift {
            Loc::Immed(i) | Loc::ImmedFloat(i) => i.value,
            other => panic!(
                "LoadEffectiveAddress shift must be Immed (rewrite.py ConstInt), got {other:?}"
            ),
        };
        let ofs = match baseofs {
            Loc::Immed(i) | Loc::ImmedFloat(i) => i.value,
            other => panic!(
                "LoadEffectiveAddress baseofs must be Immed (rewrite.py ConstInt), got {other:?}"
            ),
        };
        // `load_effective_addr`: `LEA(result, addr_add(frm, sizereg,
        // baseofs, scale))`. `make_sure_var_in_reg` hands a constant base
        // back as an `ImmedLoc`, which `AddressLoc` folds into the static
        // offset (`NO_BASE_REGISTER`, location code `'a'`).
        let addr = match index {
            Loc::Immed(i) | Loc::ImmedFloat(i) => {
                let scaled = Loc::immed(i.value.wrapping_shl(shift_amt as u32));
                addr_add(base, &scaled, ofs, 0)
            }
            Loc::Reg(r) if !r.is_xmm => {
                let shift = match shift_amt {
                    0..=3 => shift_amt as u8,
                    other => panic!(
                        "LoadEffectiveAddress shift must be 0..=3 (rewrite.py itemscale), got {other}"
                    ),
                };
                addr_add(base, index, ofs, shift)
            }
            other => panic!(
                "LoadEffectiveAddress index must be Loc::Reg after \
                 consider_load_effective_address, got {other:?}"
            ),
        };
        self.mc.LEA(*dst, addr);
    }

    /// NEWSTR: allocate a byte string of given length.
    /// `base_size` / `item_size` come from the injected ArrayDescr
    /// (`builtin_string_array_descr` in `runner.rs`), which encodes
    /// `get_array_token(rstr.STR, ...)` — basesize includes the +1
    /// extra_item_after_alloc null terminator.
    fn genop_newstr(&mut self, op: &Op, arglocs: &[Loc]) {
        let (base_size, item_size, type_id) = Self::array_token_from_descr(op, 16, 1);
        self.genop_alloc_lowlevel_string(arglocs, type_id, base_size, item_size);
    }

    /// NEWUNICODE: allocate a unicode string (4-byte chars).
    /// Basesize = 16 (no extra_item_after_alloc), itemsize = 4.
    fn genop_newunicode(&mut self, op: &Op, arglocs: &[Loc]) {
        let (base_size, item_size, type_id) = Self::array_token_from_descr(op, 16, 4);
        self.genop_alloc_lowlevel_string(arglocs, type_id, base_size, item_size);
    }

    /// Read `(base_size, item_size)` from the injected ArrayDescr.
    /// Fallback used only when the descr is missing (should never happen
    /// for NEWSTR/NEWUNICODE after `inject_builtin_string_descrs`).
    fn array_token_from_descr(op: &Op, fallback_base: i64, fallback_item: i64) -> (i64, i64, i64) {
        op.with_array_descr(|ad| {
            (
                ad.base_size() as i64,
                ad.item_size() as i64,
                ad.type_id() as i64,
            )
        })
        .unwrap_or((fallback_base, fallback_item, 0))
    }

    fn genop_alloc_lowlevel_string(
        &mut self,
        arglocs: &[Loc],
        type_id: i64,
        base_size: i64,
        item_size: i64,
    ) {
        // The length comes from its regalloc location, for the reason
        // `genop_alloc_varsize` states below: `consider_raw_call_like` plans
        // these opcodes, and `before_call` leaves a value bound to a
        // callee-saved member of the allocation pool in its register, where
        // `resolve_opref` cannot see it.
        let [len_loc, ..] = arglocs else {
            panic!("lowlevel string allocation expects a length location, got {arglocs:?}");
        };
        self.emit_load_to_rax(*len_loc);
        self.emit_abi_int_arg_from_reg(3, 0);
        self.emit_abi_int_arg_from_imm(0, type_id);
        self.emit_abi_int_arg_from_imm(1, base_size);
        self.emit_abi_int_arg_from_imm(2, item_size);
        rx86::mov_ri(
            &mut self.mc,
            rx86::EAX,
            crate::runner::dynasm_malloc_lowlevel_string as *const () as i64,
        );
        self.emit_abi_call_rax();
    }

    /// Shared implementation for NEW_ARRAY.
    /// Allocates base_size + length * item_size bytes, zero-fills,
    /// and writes length to the header.
    fn genop_alloc_varsize(&mut self, arglocs: &[Loc], base_size: i64, item_size: i64) {
        // The length comes from its regalloc location.  `consider_raw_call_like`
        // plans these opcodes, and `before_call` leaves a value bound to a
        // callee-saved member of the allocation pool in its register, where
        // `resolve_opref` cannot see it.  The load is consumed by the very next
        // instruction, so rax/x0 doubling as ABI arg0 is safe here.
        let [len_loc, ..] = arglocs else {
            panic!("varsize allocation expects a length location, got {arglocs:?}");
        };
        self.emit_load_to_rax(*len_loc);
        let malloc_ptr = libc::malloc as *const () as i64;
        let memset_ptr = libc::memset as *const () as i64;

        // Save length, compute total_size = base_size + length * item_size
        dynasm!(self.mc ; .arch x64
            ; push rax                           // save length
        );
        rx86::imul_ri(&mut self.mc, rx86::EAX, item_size as i32);
        rx86::add_ri(&mut self.mc, rx86::EAX, base_size as i32);
        dynasm!(self.mc ; .arch x64
            ; push rax                           // save total_size
        );
        self.emit_abi_int_arg_from_reg(0, 0);
        rx86::mov_ri(&mut self.mc, rx86::EAX, malloc_ptr);
        // Two pushes sit below the scratch, so the outgoing shadow is
        // not `PASS_ON_MY_FRAME`.
        self.emit_abi_call_rax_with_extra_pushes(2);
        dynasm!(self.mc ; .arch x64
        ; pop rcx                            // rcx = total_size
        );
        dynasm!(self.mc ; .arch x64
            ; push rax                           // save ptr
        );
        self.emit_abi_int_arg_from_reg(2, 1);
        self.emit_abi_int_arg_from_imm(1, 0);
        self.emit_abi_int_arg_from_reg(0, 0);
        rx86::mov_ri(&mut self.mc, rx86::EAX, memset_ptr);
        self.emit_abi_call_rax_with_extra_pushes(2);
        dynasm!(self.mc ; .arch x64
        ; pop rax                            // rax = ptr
        );
        dynasm!(self.mc ; .arch x64
        ; pop rcx                            // rcx = length
        );
        dynasm!(self.mc ; .arch x64
            // Store length at offset 8 (RPython string header)
            ; mov [rax + 8], rcx
        );
    }

    /// `AddressLoc(base_loc, _scaled_addr(...))` for ZERO_ARRAY, as INSN
    /// encodes it. `_scaled_addr` folds an immediate start into
    /// `start * itemsize`; a register start uses scale `get_scale(itemsize)`.
    /// An immediate base is `'j'` with an immediate start and `'a'` with
    /// `NO_BASE_REGISTER` otherwise. A displacement outside the signed 32-bit
    /// range is `_addr_as_reg_offset` for `'j'` and
    /// `_fix_static_offset_64_m` / `_fix_static_offset_64_a` for `'m'` / `'a'`.
    fn addr_add_zero_array(
        &mut self,
        base: &Loc,
        start: &Loc,
        itemsize: i64,
        offset: i64,
    ) -> Operand {
        let (scaled, shift) = match start {
            Loc::Reg(_) => {
                assert!(
                    crate::regalloc::valid_addressing_size(itemsize as usize),
                    "rewrite did not correctly handle shift/mul!"
                );
                (*start, crate::regalloc::get_scale(itemsize as usize))
            }
            Loc::Immed(s) | Loc::ImmedFloat(s) => (Loc::immed(s.value.wrapping_mul(itemsize)), 0),
            other => panic!("ZERO_ARRAY start must be a GPR or immediate, got {other:?}"),
        };
        addr_add(base, &scaled, offset, shift)
    }

    /// `genop_discard_zero_array`: the unrolled `XORPS` + `MOVUPS` /
    /// `save_into_mem(imm0)` stores through `addr_add`. An eight-loc
    /// `arglocs` is `consider_zero_array`'s memset() path.
    fn genop_discard_zero_array(&mut self, arglocs: &[Loc]) {
        if let [
            base_loc,
            startindex_loc,
            itemsize_loc,
            baseofs_loc,
            dstaddr_loc,
            length_loc,
            len_itemsize_loc,
            bytes_loc,
        ] = arglocs
        {
            return self.zero_array_memset(
                base_loc,
                startindex_loc,
                itemsize_loc,
                baseofs_loc,
                dstaddr_loc,
                length_loc,
                len_itemsize_loc,
                bytes_loc,
            );
        }
        let [
            base_loc,
            startindex_loc,
            Loc::Immed(bytes_loc),
            Loc::Immed(itemsize_loc),
            Loc::Immed(baseofs_loc),
            Loc::Reg(null_loc),
        ] = arglocs
        else {
            panic!(
                "ZERO_ARRAY expects (base, start, bytes, itemsize, baseofs, null), got {arglocs:?}"
            );
        };
        assert!(null_loc.is_xmm);
        let baseofs = baseofs_loc.value;
        let nbytes = bytes_loc.value;
        let null_loc = null_loc.value;
        let mut null_reg_cleared = false;
        let mut i = 0i64;
        while i < nbytes {
            let addr =
                self.addr_add_zero_array(base_loc, startindex_loc, itemsize_loc.value, baseofs + i);
            let mut current = nbytes - i;
            if current >= 16 {
                current = 16;
                if !null_reg_cleared {
                    rx86::xorps_xx(&mut self.mc, null_loc, null_loc);
                    null_reg_cleared = true;
                }
                self.mc.MOVUPS(addr, Operand::X(null_loc));
            } else {
                if current >= WORD as i64 {
                    current = WORD as i64;
                } else if current >= 4 {
                    current = 4;
                } else if current >= 2 {
                    current = 2;
                }
                self.save_into_mem(addr, &Loc::immed(0), current as usize);
            }
            i += current;
        }
    }

    /// The memset() half of `consider_zero_array`: `LEA(dstaddr_loc,
    /// _scaled_addr(startindex_loc, itemsize_loc, base_loc, imm(baseofs)))`,
    /// the `LEA` of the byte length when `len_itemsize > 1`, then
    /// `simple_call_no_collect(imm(memset_addr), [dstaddr_loc, imm0,
    /// length_loc])`. Regalloc has already run `before_call`.
    #[allow(clippy::too_many_arguments)]
    fn zero_array_memset(
        &mut self,
        base_loc: &Loc,
        startindex_loc: &Loc,
        itemsize_loc: &Loc,
        baseofs_loc: &Loc,
        dstaddr_loc: &Loc,
        length_loc: &Loc,
        len_itemsize_loc: &Loc,
        bytes_loc: &Loc,
    ) {
        let (
            Loc::Immed(itemsize),
            Loc::Immed(baseofs),
            Loc::Reg(dstaddr),
            Loc::Immed(len_itemsize),
        ) = (itemsize_loc, baseofs_loc, dstaddr_loc, len_itemsize_loc)
        else {
            panic!(
                "ZERO_ARRAY memset locations: {itemsize_loc:?} {baseofs_loc:?} {dstaddr_loc:?} {len_itemsize_loc:?}"
            );
        };
        let dst_addr =
            self.addr_add_zero_array(base_loc, startindex_loc, itemsize.value, baseofs.value);
        self.mc.LEA(*dstaddr, dst_addr);
        let mut length_loc = *length_loc;
        if len_itemsize.value > 1 {
            let (Loc::Reg(index), Loc::Reg(bytes)) = (length_loc, bytes_loc) else {
                panic!("ZERO_ARRAY scaled length needs registers: {length_loc:?} {bytes_loc:?}");
            };
            assert!(
                crate::regalloc::valid_addressing_size(len_itemsize.value as usize),
                "rewrite did not correctly handle shift/mul!"
            );
            let shift = crate::regalloc::get_scale(len_itemsize.value as usize);
            self.mc
                .LEA(*bytes, addr_add(&Loc::immed(0), &Loc::Reg(index), 0, shift));
            length_loc = *bytes_loc;
        }
        let memset_addr = libc::memset as *const () as i64;
        self.simple_call_no_collect(
            Loc::immed(memset_addr),
            &[*dstaddr_loc, Loc::immed(0), length_loc],
        );
    }

    /// `Assembler386.simple_call`.
    fn simple_call(&mut self, fnloc: Loc, arglocs: &[Loc], result_loc: Option<Loc>) {
        let (result_type, result_size) = match result_loc {
            Some(Loc::Reg(r)) if r == crate::regloc::XMM0 => ('f', 8),
            None => ('v', 0),
            Some(_) => ('i', WORD),
        };
        super::callbuilder::CallBuilder64::new(
            self,
            fnloc,
            arglocs,
            result_loc,
            result_type,
            result_size,
        )
        .emit();
    }

    /// `Assembler386.simple_call_no_collect`.
    fn simple_call_no_collect(&mut self, fnloc: Loc, arglocs: &[Loc]) {
        super::callbuilder::CallBuilder64::new(
            self,
            fnloc,
            arglocs,
            Some(Loc::Reg(crate::regloc::EAX)),
            'i',
            WORD,
        )
        .emit_no_collect();
    }

    // genop_* — address computation
}

/// `jump.py`'s three primitives. The parallel-move algorithm that drives
/// them is one shared implementation in `crate::jump`; only these differ per
/// backend, which is what makes them the trait and it the free function.
impl<'a> crate::jump::RegallocMoves for Assembler386<'a> {
    fn regalloc_mov(&mut self, src: &Loc, dst: &Loc) {
        let scratch = crate::regloc::X86_64_SCRATCH_REG;
        let from_xmm = matches!(src, Loc::Reg(r) if r.is_xmm);
        let to_xmm = matches!(dst, Loc::Reg(r) if r.is_xmm);
        match (src, dst) {
            (Loc::Reg(s), Loc::Reg(d)) if s == d => {}
            // copy 128-bit from -> to
            _ if from_xmm && to_xmm => self.mc.MOVAPD(dst, src),
            // A GPR to or from an xmm register has no `MOVSD` form.
            (Loc::Reg(_), Loc::Reg(_)) if from_xmm || to_xmm => self.mc.MOVDQ(dst, src),
            (Loc::Immed(_), Loc::Reg(_)) if to_xmm => {
                self.mc.MOV(scratch, src);
                self.mc.MOVDQ(dst, scratch);
            }
            _ if from_xmm || to_xmm => self.mc.MOVSD(dst, src),
            (Loc::ConstFloat(from), ebp_loc_pat!(_e)) => {
                self.regalloc_immedmem2mem(*from, dst);
            }
            (ebp_loc_pat!(e1), ebp_loc_pat!(e2)) if e1.value == e2.value => {}
            // `jump.py _move` stages a memory source through `tmpreg`; a
            // frame-to-frame move that reaches here goes through r11.
            (Loc::Frame(_) | Loc::Ebp(_), Loc::Frame(_) | Loc::Ebp(_)) => {
                self.mc.MOV(scratch, src);
                self.mc.MOV(dst, scratch);
            }
            // `mov` -> `MOV(to_loc, from_loc)`: a `FloatImmedLoc` has location
            // code 'i' like an `ImmedLoc`, so `genop_finish`'s `save_into_mem`
            // of a ConstFloat stores its bits the same way.
            _ => self.mc.MOV(dst, src),
        }
    }

    /// `regalloc_push`: an xmm register goes through `SUB esp, 8` and
    /// `MOVSD_sx`; everything else is `PUSH(loc)`. On x86-64 a float
    /// `FrameLoc` is one word, so it is a plain `PUSH_b`.
    fn regalloc_push(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch x64 ; sub rsp, 8 ; movsd [rsp], Rx(r.value));
            }
            _ => self.mc.PUSH(loc),
        }
    }

    /// `regalloc_pop`, the mirror of `regalloc_push`.
    fn regalloc_pop(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch x64 ; movsd Rx(r.value), [rsp] ; add rsp, 8);
            }
            _ => self.mc.POP(loc),
        }
    }

    /// x86/jump.py `_move`. A `ConstFloatLoc` is the one
    /// `ImmediateAssemblerLocation` that is a memory reference; every caller
    /// passes a `tmpreg`, so the push/pop arm for `tmpreg is None` has no
    /// counterpart here.
    fn _move(&mut self, src: &Loc, dst: &Loc, tmpreg: &Loc) {
        let mut src = *src;
        if dst.is_memory_reference() && src.is_memory_reference() {
            if let Loc::ConstFloat(from) = src {
                self.regalloc_immedmem2mem(from, dst);
                return;
            }
            self.regalloc_mov(&src, tmpreg);
            self.mc.forget_scratch_register();
            src = *tmpreg;
        }
        self.regalloc_mov(&src, dst);
    }
}

/// x86/regalloc.py `self.assembler`: `RegAlloc.walk_operations` reaches each
/// op's emitter through these while its register managers still hold the
/// state of that op.
impl RegAllocAssembler for Assembler386<'_> {
    fn regalloc_perform(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        gcmap: Option<usize>,
    ) {
        let ops = self.operations;
        let op = &ops[op_index];
        if crate::majit_ops_log_enabled() {
            let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
            eprintln!(
                "[dynasm] emit[{}]: {:?} args=[{}] result={:?}",
                op_index,
                op.opcode,
                al.join(", "),
                result_loc
            );
        }
        // Byte offset where this op's code begins: the span from
        // the last LABEL to the JUMP is one loop iteration, the
        // window `jit-log-opt` brackets with its `+508:` prefixes.
        if crate::majit_dump_enabled() {
            eprintln!(
                "[dynasm] @{:#06x} op[{}] {:?}",
                self.mc.offset().0,
                op_index,
                op.opcode
            );
        }
        self.pending_malloc_nursery_gcmap = gcmap;
        let fail_index = self.fail_index;
        self.genop_list(
            regalloc,
            op,
            op_index,
            arglocs,
            result_loc.as_ref(),
            fail_index,
            ops,
        );
        self.pending_malloc_nursery_gcmap = None;
    }

    fn regalloc_perform_discard(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        arglocs: &[Loc],
    ) {
        let ops = self.operations;
        let op = &ops[op_index];
        if crate::majit_ops_log_enabled() {
            let al: Vec<String> = arglocs.iter().map(|l| format!("{l:?}")).collect();
            eprintln!(
                "[dynasm] discard[{}]: {:?} args=[{}]",
                op_index,
                op.opcode,
                al.join(", ")
            );
        }
        if crate::majit_dump_enabled() {
            eprintln!(
                "[dynasm] @{:#06x} op[{}] {:?}",
                self.mc.offset().0,
                op_index,
                op.opcode
            );
        }
        let fail_index = self.fail_index;
        self.genop_list(regalloc, op, op_index, arglocs, None, fail_index, ops);
        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            self.fail_index += 1;
        }
    }

    fn regalloc_perform_guard(
        &mut self,
        _regalloc: &RegAlloc<'_>,
        op_index: usize,
        faillocs: &[Option<Loc>],
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        frame_depth: usize,
    ) {
        let ops = self.operations;
        let op = &ops[op_index];
        if crate::majit_ops_log_enabled() {
            eprintln!(
                "[dynasm] guard[{}]: {:?} args=[{}] faillocs={}",
                op_index,
                op.opcode,
                arglocs
                    .iter()
                    .map(|l| format!("{:?}", l))
                    .collect::<Vec<_>>()
                    .join(", "),
                faillocs.len()
            );
        }
        // A guard owns bytes like any other op, so it opens its
        // own span. Without this line the guard's code is read as
        // part of the preceding op's span and every span after the
        // trace's first guard names the wrong operation.
        if crate::majit_dump_enabled() {
            eprintln!(
                "[dynasm] @{:#06x} op[{}] {:?}",
                self.mc.offset().0,
                op_index,
                op.opcode
            );
        }
        let fail_index = self.fail_index;
        self.genop_guard_list(
            op,
            op_index,
            arglocs,
            result_loc.as_ref(),
            faillocs,
            fail_index,
            frame_depth,
        );
        self.fail_index += 1;
    }
}
